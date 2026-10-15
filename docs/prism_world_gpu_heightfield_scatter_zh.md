# GPU 真实高度场散布过滤

验收日期：2026-10-16。本文记录已实现的过滤切片，不是整个 world 系统的完成声明。

## 范围与精确契约

`prism_world_gpu/terrain_scatter` 导出 `GpuScatterHeightfield`，依次记录
既有整数源生成、作者密度/种类场及真实高度场过滤三个 compute pass。
高度数据来自 `HeightfieldTile::bake/rebake` 的不可变快照，不是 CPU 预计算
的候选接受标记。生产路径不逐点生成、采样、过滤或回读。

过滤与 `prism_world_terrain::scatter_with_heightfield` revision 1 精确对应：

- 显式平面 embedding，XZ 半开覆盖；按网格 a-c-b / b-c-d 三角拓扑，
  对角线归第一三角。不是双线性插值。
- encoded i16 高度乘正 i64 毫米比例，加世界 origin Y；有理高度不先取整。
  地表高度和垂直离地带均为闭区间。
- 精确比较 `sqrt(dx²+dz²)/spacing <= rise/run`。先检查宽位分量上界，
  再使用平方差避免平方和溢出。
- 全局 i64 坐标、超过 i64 的解码高度及有理比较通过 u32 limb 实现，
  不需要 shader-i64、浮点世界坐标或额外设备 feature。
- 只删除源点，不移动、不旋转、不补点。原 candidate key、重建、最小距离、
  任意 XYZ 分区集合不变；窄离地带可能输出很少甚至零点。

## API、资源和失败语义

`new(ctx, capacity, max_brushes, max_surface_samples, vertex_count, max_buffer_bytes)`
在分配前验证完整复合资源；`write_request` 接收源配置/范围/源工作上限、
作者场/场工作上限、地表/过滤器/地表读取预算。

上下文必须显式请求至少六个 storage binding 和七个 bind-group binding；
普通 `GpuContext` 的四 binding downlevel 默认不变。
`GpuContext::try_headless_with_limits` 可请求所需 limits，不支持时返回失败。
地表 uniform 为 128 字节，所有样本扩展为 i32 上传，避免依赖 16-bit storage。

令 `C=max(capacity,1)`、`B=max(max_brushes,1)`、`S=max(max_surface_samples,1)`，
包含源、场和地形三级的 owned 逻辑 GPU buffer 总计 `108C+64B+4S+496` 字节。
另预留 height upload scratch `4*max_surface_samples` 字节及作者场刷子 scratch；
实际容量可通过 getter 查看。共享原 i16 地形快照、元数据、allocator/driver
内部存储另计；这些不是物理 VRAM 或 RSS 上限。

完整非空请求 XZ 必须被地表覆盖；样本数不得超过固定容量或 u32 ABI。
地表读取准入使用 `4*primary_candidates` 上界，实际每个场幸存点读取四角。
样本容量、过滤参数、覆盖、预算、源/场工作及 dispatch 均在任何 queue write
前验证；返回的可恢复错误保留旧请求、完整缓存键和旧 GPU 输出。
设备丢失、wgpu OOM 和普通元数据分配不属于该恢复承诺。

## 缓存、上传和队列生命周期

完整结构缓存键包含源/作者场、surface/filter 版本、完整地形规格/样本/revision/
tile/embedding 与过滤器，保留共享不可变快照；不以 tile+revision 或摘要代替身份。
同 tile/revision 的独立不同样本不能命中，发布 rebake 新快照不会改变旧键。

源/场上传沿用 `GpuScatterField`。地形请求字变化上传 128 字节；实际样本变化
上传 `4*N` 字节，只有 embedding、比例、过滤器或 metadata 变化而样本相等时
不重传样本。完整结构比较最坏为 `O(N+B)`，不是常数身份检查。

每次 `encode` 清最终 indirect count 并记录三个 pass；自身不分配资源、不上传、
不 submit/wait/map/readback。先提交当前 encoder，再更新下一请求；顺序提交
不要求主机等待。不能先记录多个不同请求的 encoder 再集中提交。

## 消费 ABI

`records()` 保留原 64 字节源记录及源 accepted 标记；`attributes()` 是按源 slot
对齐的 16 字节 `[density,species,final_accepted,rank]`。所有 primary slot
均重写，场/地形拒绝点不会留下旧 accepted。只有本对象的 `compact_indices()`
及 `indirect_args()` 表达最终地形筛选；不能消费中间 source/field accepted
来绕过地形。compact 只有 indirect count 前缀有效，跨工作组顺序未规定。

下游以 compact 源 slot 读取原整数位置和属性；indirect 为
`[vertex_count,final_count,0,0]`。当前 pass 不输出投影位置、朝向、缩放或 raster
coverage，作者 density 也不会自动变成透明度。

## 验证入口与边界

```text
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -p prism_world_gpu --features terrain_scatter -- --nocapture --test-threads=1
cargo test --offline -p prism_world_gpu --features terrain_scatter --lib --test wgsl_compile
cargo clippy --offline -p prism_world_gpu --features terrain_scatter --all-targets -- -D warnings
WGPU_BACKEND=metal cargo bench --offline -p prism_world_gpu --features terrain_scatter --bench scatter_heightfield
```

实机 Apple M2 / Metal / IntegratedGpu，required features 为空。正常 workspace
完整 package 定向回归 **126 passed、0 failed、0 ignored**：库 87、density 8、
parity 1、resident 6、作者场 7、源生成 5、本次地形过滤 8、terrain draw 1、
WGSL 3；无文档测试。no-default 库另外 73 项通过。

新增八项实机测试逐 primary slot 对 CPU 源记录、作者属性及最终接受标记，
验证 compact 唯一集合和 indirect count；直接在同一 encoder 内消费 GPU
records/attributes/compact/indirect 作实际 draw_indirect，并逐像素核对。
覆盖非共面三角与对角线、有理高度/离地带闭端点、双轴坡度和宽整数平方溢出、
i64 世界角点/u32 spacing/解码超 i64、全部轴任意逆序分区、结构缓存/相同
revision 不同内容/公开 rebake、空轴/尾部/重复 encode/无主机等待连续提交，
以及各类预检失败后旧请求、完整 key、资源与实际 GPU 输出保持不变。

原始源码隔离 harness 的 87 项库测试与 3 项 WGSL 检查、全部 target Clippy
`-D warnings` 已通过。正常 workspace 完整测试也已通过，不能再把先前
workspace manifest 暂时损坏视为当前测试阻塞。正常 workspace all-target
Clippy 则在共享依赖 `prism_asset` 处失败（164 项既有 lint/doc 错误，包含
`multiple_bound_locations`、`collapsible_if`、`large_enum_variant` 和缺失文档）；
没有忽略或放宽 lint 来宣称通过，也没有修改无关共享改动。这不是全 workspace
Clippy 或多 GPU 后端验收。

## Release 性能、输出与资源

运行环境为 `aarch64-apple-darwin`、macOS 27.0.1（26A434）、rustc 1.96.1
（31fca3adb，2026-06-26）、LLVM 22.1.2。release benchmark 与生产实现使用
同一源码，通过独立 `/private/tmp` target 和 source-isolation harness 执行。
每档 8 次预热、64 次测量，每次包括预热都完整回读并在计时外逐 slot 校验；
两轮共 288 次 dispatch 全部通过，输出数量和 digest 一致。

实烘 65×65 高度场，sampler 为 `u_q16/256-v_q16/256` 毫米，scale=1，
origin=`[-32000,0,-32000]`，spacing=1000mm。源为 100mm 三维 jittered voxel，
min distance=0、seed=`0x0123456789abcdef`、version=1、salt=22；作者场 seed=
`0xfedcba9876543210`、version=1、density=40000、species=7，noise cell/amp=
311/10000、warp cell/amp=127/201，无刷子。地表高度 `[-100,100]`、垂直离地带
`[-250,250]`、坡度 rise/run=1/10。10k 请求为 20×25×20，范围
`[-1000,-1200,-1000]..[1000,1300,1000]`；100k 为 40×50×50，范围
`[-2000,-2500,-2500]..[2000,2500,2500]`，全部半开，源点全部通过源筛选。

| primary | 场幸存 / 最终点 | 实际逻辑地形读数 / 准入上界 | 源 / 场工作预算 | 全 slot 数值 digest |
|---:|---:|---:|---:|---|
| 10,000 | 6,132 / 1,250 | 24,528 / 40,000 | 10,000 / 330,000 | `6ec66e0aa5c5cb6a` |
| 100,000 | 61,122 / 6,193 | 244,488 / 400,000 | 100,000 / 3,300,000 | `fc3be7280867a56b` |

digest 覆盖全部 slot 与 indirect，不含未定义的 compact 顺序，不充当无碰撞身份。
实际地形读数来自相同 CPU oracle 的逻辑查询计数，不是 GPU 内存访问性能计数器。

以下均为主机墙钟微秒，nearest-rank P50/P95/P99；encode+finish 和 submit+wait
分开计时，包含三级 compute 与计数清零。排除 encoder/pipeline/buffer 创建、
上传、回读、CPU oracle/校验及绘制；每次 native poll/map callback 等待有 30 秒
硬上限，超时/不匹配直接失败，不重试、跳过或截断。

| 轮次 / primary | encode+finish P50 / P95 / P99 µs | submit+wait P50 / P95 / P99 µs |
|---|---:|---:|
| 首轮 / 10k | 164.167 / 344.125 / 808.000 | 1376.750 / 1520.917 / 1626.500 |
| 首轮 / 100k | 239.000 / 406.625 / 1628.000 | 7366.917 / 13192.792 / 19275.916 |
| 回归结束后复测 / 10k | 179.125 / 288.625 / 331.417 | 1442.333 / 2362.833 / 4674.250 |
| 回归结束后复测 / 100k | 242.042 / 399.250 / 1621.625 | 7522.584 / 15448.750 / 25020.584 |

首轮与构建/其他定向回归可能重叠；复测在本轮 GPU 回归结束后运行，但共享机器
仍非独占。P99 波动保留两轮，不挑选更好的数值；没有 GPU timestamp、产品帧时、
无干扰 A/B、CPU/GPU 加速比或吞吐目标达标结论。

| primary | owned GPU 逻辑 buffer B | 验证 staging B | CPU oracle slots 容量 B | 唯一性 seen 容量 B |
|---:|---:|---:|---:|---:|
| 10,000 | 1,097,460 | 840,016 | 800,000 | 10,000 |
| 100,000 | 10,817,460 | 8,400,016 | 8,000,000 | 100,000 |

两档共享原 i16 高度载荷均为 8,450B，owned height upload scratch 均为 16,900B，
brush scratch 为 0B。首次 request 上传 17,348B（source192 + field128 +
terrain128 + height16,900），构造的三级 indirect 初始化另为 48B，合计 17,396B；
不含隐式零初始化及 driver 元数据。计时帧无上传。验证 staging、oracle/seen 仅属
测试/基准，不属于生产 buffer 预算；这些容量载荷不是物理显存、RSS 或峰值内存。

这是精确 GPU 地表过滤，不是完整产品散布。贴地放置、曲率/SDF/数据层和跨散布层
规则、风/交互、持久 override 删除、笔刷空间索引、scene schema/编辑器、产品
renderer graph、蓝噪声频谱与混合 LOD 画质仍需各自实现并验收。局部 parity、
像素和限定基准不能替代这些工作或整个 world 系统的完成证据。
