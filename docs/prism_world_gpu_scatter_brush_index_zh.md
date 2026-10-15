# World GPU 作者刷子有序 AABB 索引

日期：2026-10-16

## 权威与选择

`GpuScatterField::new_indexed` 与 `GpuScatterHeightfield::new_indexed` 是显式可选的 GPU 刷子查询加速路径。配方仍是 `ScatterDensityField`，scene 仍是唯一作者真相；索引可删除重建，不增加持久身份、算法版本或缓存键字段。原 `new` 保留线性基线，索引不自动启用、不改变笔刷参数、不截断配方。

这不是 CPU `ScatterBrushIndex` 的空间中位数 BVH 的 GPU 移植。CPU 可以收集候选并排序；GPU 在无每点 scratch 的前提下必须直接保留 Replace/Add/species 的稳定 ID 合成顺序，因此对 canonical 升序 ID 的**连续区间**作中点划分。树是平衡前序数组，内部节点存子树 AABB 和 exclusive escape，叶子仍按原 canonical 顺序出现。裁剪跳过的只有不覆盖位置的子树，其余叶子顺序不变；精确球形衰减、取整和 noise/warp 仍复用原 WGSL 整数 kernel。

构造 O(B)、B−1 次内部包围盒合并，递归栈 O(log B)，除预留上传数组外无辅助堆分配；GPU 查询无栈、无 scratch、无排序。查询为 O(访问节点+精确刷子测试+固定噪声)，最坏 O(B)。ID 与空间相关时可大幅裁剪；随机 ID 的连续区间包围盒可能几乎覆盖全域，即使刷子本身稀疏也可能更慢。全重叠必须走全树，不能承诺普遍加速或自动作为默认值。

所有 bounds 以 i128 计算 `center ± outer_radius` 后夹到有符号 i64 世界域；父节点做精确整数 union。WGSL 以 low/high u32 pair 作有符号闭区间比较，不引入浮点世界位置、shader-i64 或 optional feature。AABB 是保守测试，球面零权重仍由原精确 brush kernel 决定。

## 固定资源与工作准入

令 `C=max(capacity,1)`，固定 brush 容量为 b，`T(0)=0`，非零 `T(b)=2b−1`，`R=max(b+T(b),1)`。索引全部打包到既有 brush storage buffer 尾部，作者场仍为五 storage binding、一 uniform；高度场仍为六 storage binding。

| 模式 | owned 逻辑 GPU buffer 字节 |
| --- | --- |
| 线性 field | `88C+64*max(b,1)+352` |
| 索引 field | `88C+64R+352` |
| 索引 heightfield | `108C+64R+4*max(max_surface_samples,1)+496` |

host 上传 scratch 在构造时 fallible 预留 b+T(b) 个64B record，`brush_scratch_bytes()` 返回实际容量载荷。高度样本另有独立 i32 scratch。构造先检查完整复合预算、各 buffer/device limit、binding 和组合 record 的 u32 地址空间，再分配 GPU 资源；单独节点数适合 u32 不代表刷子加节点仍能寻址。零容量保留合法 sentinel buffer，scratch 不需要 sentinel。buffer 逻辑载荷不是物理 VRAM；Vec/Arc 元数据、allocator/driver 内存、CPU 配方和构造栈另计，不是 RSS 上限。

`brush_index_enabled()` 查询对象模式。`max_sample_work(&field)?` 纯检查 brush 容量并返回每 primary 的保守 work：线性为 `field.work_per_sample()`，索引另加实际 B 的 `T(B)`。`write_request` 在任何 queue write 前 checked 计算 `primary_candidates*max_sample_work` 并检查调用者预算，不能仍用线性预算上传非空 indexed 配方；该上界覆盖整树与所有刷子，不利用稀疏或低密度放宽准入。逻辑 work 不是 GPU 指令/周期或毫秒硬限额，也不是实测访问节点数。

所有可恢复预检错误保留旧 source/field/terrain 请求、完整 key 和 GPU 输出；不排入部分 queue write。设备丢失、driver OOM 不属于此恢复承诺。indexed heightfield 的三级资源和地形读取预算同样先检后写。`GpuScatterDeletions` 可直接接收 indexed field/heightfield，不增加第二份索引或删除状态；其请求内 `max_field_work` 应使用上游对象的索引上界。

## 私有上传 ABI 与生命周期

一次实际 B 刷子上传布局为 `[B 个原 brush record][T(B) 个 node record]`，不是按固定容量 b 定位节点。原 brush record 仍64B，不变。128B field uniform 的 word19 是节点基址 B、word20 是节点数，word21..31 为零；线性模式19/20均零，空索引节点数为零。节点64B布局：

| u32 words | 内容 |
| --- | --- |
| 0..5 | min XYZ，各 i64 low/high |
| 6..11 | max XYZ，各 i64 low/high |
| 12 | exclusive subtree escape，相对节点块起始，不相对 brush buffer 或当前节点 |
| 13 | canonical brush index；内部节点为 `u32::MAX` |
| 14..15 | 零 |

只有完整 canonical 刷子列表变化才重新构造、打包并上传 `64*(B+T(B))` 字节；相同列表、仅源范围/seed 变化、仅 field base/noise 变化均不重传节点。列表缩短或清空后旧尾部可以保留，但 uniform 的实际 B/T(B) 防止访问；再次增长重写完整有效前缀。完整请求相等不上传，独立配方比较最坏 O(B)，不是常数摘要身份。构建是 O(B) 全量重建，不是增量编辑/refit。

`encode` 不分配 owned 资源、不构建索引、不上传、不 submit/wait/map/readback；生产路径没有 CPU 逐点采样。先提交当前 encoder，再修改下次请求，顺序提交不要求主机等待。不能累积不同请求的 encoder 后一起提交而假定它们各自保留旧 uniform/brush 内容。

输出 ABI 完全不变：原64B source record 保持源 accepted，16B 属性为 density/species/该阶段 accepted/rank；必须消费当前最终 producer 的 compact/indirect，不能只看原 accepted 标记。高度场和删除层继续只删点、不移动或补点。compact 跨工作组顺序未规定，只有 indirect-count 前缀有效。

## 验证与复现

```sh
cargo test --offline --locked -p prism_world_gpu --features terrain_scatter --lib --test wgsl_compile
WGPU_BACKEND=metal cargo test --offline --locked -p prism_world_gpu --features terrain_scatter --test scatter_brush_index -- --nocapture --test-threads=1
cargo clippy --offline --locked -p prism_world_gpu --lib --features terrain_scatter -- -D warnings
WGPU_BACKEND=metal cargo bench --offline --locked -p prism_world_gpu --features gpu --bench scatter_brush_index
```

2026-10-16 Apple M2 / Metal / IntegratedGpu，无 optional feature。原始源码独立 Cargo harness 定向回归164项全通过、0失败/忽略：库112、WGSL5、索引12、源生成5、作者场7、heightfield8、删除7、组合删除8。另独立验证no-default库73项、仅gpu feature索引11项全通过。harness 直接引用仓库源文件及 path 依赖，避开包 dev-dependency 拉入的无关引擎构建，不是源码副本；不能据此宣称全 workspace 或多后端通过。正常 workspace 的 GPU+terrain 库及 harness all-target strict Clippy（`-D warnings`）均通过。

新增节点单测检查前缀不变、指针/容量不变、精确子树 union/shape/escape、canonical 叶序、signed ABI、世界边缘、数量溢出、无预留 debug 拒绝，以及独立闭区间遍历参考。新增12项公开 GPU 回归严格要求真实设备及所选 `WGPU_BACKEND`，不跳过：空/单点/奇偶树、最大半径/i64全域、noise/warp、非交换乱序ID、稀疏/重叠、分区/最小距离、资源缩放复用、恰好预算/差一/容量/旧请求保留，及 indexed field→heightfield→managed删除对公共 CPU 链路。每次逐 slot 检查完整源记录、sample/rank/accepted，检查 compact 唯一集合与 indirect；失败后再 encode 旧请求，以发现误排入的 queue write。索引专测为 buffer 对照，不额外宣称 indexed 路径做过像素验收；原线性场/地形/删除真实 draw 回归本次继续通过。

## 限定 release A/B

2026-10-16，macOS27.0.1、rustc1.96.1、Apple M2/Metal；`CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2`，独立 harness release profile。B=256/1024，各测空间顺序ID稀疏、空间乱序ID稀疏、全部重叠。每档4对预热、32对测量，交替 A/B 先后顺序；nearest-rank P50/P95/P99，单位 µs。两对象同时常驻。encode+finish 不含 encoder 创建；总窗口连续包含 encode+finish+submit+有界等待，均不含构造、上传、readback、CPU oracle 或 draw。不是 GPU timestamp、隔离执行时间或产品帧时；机器有其他并行任务，不是独占实验。

全部432次 dispatch（含预热）在计时外与预计算完整 CPU oracle 对照，每对再核对 A/B 全输出，不重试、不跳过慢样本；poll/map receive 均30秒硬失败限时。digest 按 source slot 数值顺序生成，不依赖非确定的 compact 顺序，不是无碰撞身份。

| B / 分布 | 模式 | 接受 / digest | encode+finish P50 / P95 / P99 | 总窗口 P50 / P95 / P99 |
| --- | --- | --- | --- | --- |
| 256 / 顺序稀疏 | 线性 | 116 / `c84c9bbc11691737` | 18.209 / 23.834 / 30.333 | 1238.959 / 1317.584 / 1339.250 |
| 256 / 顺序稀疏 | 索引 | 同上 | 19.750 / 53.125 / 54.959 | 596.833 / 658.458 / 670.666 |
| 256 / 乱序稀疏 | 线性 | 116 / `84be557f2bb80188` | 24.125 / 44.917 / 52.042 | 1255.834 / 1345.166 / 1356.208 |
| 256 / 乱序稀疏 | 索引 | 同上 | 29.750 / 48.167 / 78.083 | 2026.125 / 2220.333 / 2354.000 |
| 256 / 全重叠 | 线性 | 98 / `bf15b0d6b96c5ae9` | 22.750 / 51.500 / 52.208 | 1700.375 / 1995.167 / 2154.292 |
| 256 / 全重叠 | 索引 | 同上 | 22.250 / 58.167 / 89.833 | 2206.083 / 2685.000 / 2890.417 |
| 1024 / 顺序稀疏 | 线性 | 114 / `143e5f211f42283e` | 22.125 / 22.916 / 27.042 | 1444.375 / 1569.209 / 1604.625 |
| 1024 / 顺序稀疏 | 索引 | 同上 | 22.625 / 60.500 / 60.750 | 492.750 / 546.583 / 561.791 |
| 1024 / 乱序稀疏 | 线性 | 113 / `1e8f06fed66a7fe4` | 22.875 / 52.125 / 57.042 | 1452.541 / 1546.750 / 1908.833 |
| 1024 / 乱序稀疏 | 索引 | 同上 | 22.708 / 67.166 / 71.000 | 2543.333 / 2644.833 / 2653.333 |
| 1024 / 全重叠 | 线性 | 95 / `89052866e961a4bf` | 37.250 / 53.167 / 64.125 | 5591.125 / 6535.333 / 6628.125 |
| 1024 / 全重叠 | 索引 | 同上 | 23.542 / 119.375 / 186.250 | 8005.875 / 8936.875 / 9519.834 |

本次顺序稀疏总窗口 P50 改善，但乱序稀疏与全部重叠均退化；不据此推荐无条件启用。乱序改变 ID 与几何/操作的关联，故它与顺序配方的结果可以不同；严格 A/B 只在同一完整配方内成立。

每档256个完整 jittered voxel，区域 `[0,0,0]..[256000,1000,1000]` mm，spacing1000mm、min_distance0、无 noise/warp，所有 primary 源接受。源 seed=`0123456789abcdef`、version1、salt22；场 seed=`fedcba9876543210`、version1、base_density32768、species7。稀疏中心均匀分布于 X，YZ=500mm，inner100mm、outer800mm、opacity40000；乱序用源码固定 Fisher-Yates 将稳定ID关联到不同中心。全重叠中心 `[128000,500,500]`、outer256000mm。按空间序号 mod3 依次 Replace(20000+(序号 mod17)*1000)、Add(700)、Add(-500)，species=序号+100，输入逆序后由配方 canonical 排序。CPU实际球内探针分别为355/1401与65536/262144，不是 GPU 节点访问计数。这是一维条带 fixture，不能代替三维生产分布、启用最小距离或场景散布画质验收。

| B | 模式 | 节点 | owned GPU逻辑字节 | owned host上传scratch字节 | 首次request上传估计字节 | 每点 / 全请求work上界 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 256 | 线性 | 0 | 39264 | 16384 | 16704 | 257 / 65792 |
| 256 | 索引 | 511 | 71968 | 49088 | 49408 | 768 / 196608 |
| 1024 | 线性 | 0 | 88416 | 65536 | 65856 | 1025 / 262400 |
| 1024 | 索引 | 2047 | 219424 | 196544 | 196864 | 3072 / 786432 |

首次上传估计是192B source+128B field+刷子/节点逻辑载荷，构造另初始化两个 indirect 共32B；不是实测总线流量，不含隐式清零。计时派发无上传。每档共享验证 staging21520B、CPU oracle容量20480B、seen256B，两模式各有timing数组512B；共享CPU配方载荷为16424/65576B（brush Vec容量+config）。它们不计入生产GPU buffer，也不是完整进程内存。

首次 request 的单次构建/打包/入队墙钟（线性/索引，µs）为：256顺序65.542/29.250、乱序25.042/27.083、重叠34.250/40.750；1024顺序28.417/48.292、乱序26.375/56.750、重叠29.042/44.167。该窗口不是隔离索引构建时间，且没有分位数；完整 constructor 还含 pipeline/buffer 创建，首例线性160611.416µs、后续各例约2001.542..3398.709µs，受到首次编译/缓存影响，不用于声称构造加速。初始上传在预热前单独 submit/wait，完整逐项原始输出由 bench 直接打印。

## 独立边界

本次交付可调用、可复合、有界的 GPU 刷子索引及其验证，不是完整 world 或产品散布完成。增量编辑/refit、贴地放置/朝向/缩放、曲率/数据层/跨层/SDF规则、风/交互、蓝噪声目标、scene schema/编辑器、产品 ECS decoder/render graph、多厂商后端及产品性能/物理显存峰值仍需独立实现和验收。CPU空间树与GPU有序树各有明确选择，不把任一局部稀疏结果推广到另一算法或真实产品负载。
