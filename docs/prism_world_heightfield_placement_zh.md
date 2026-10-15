# 真实 heightfield 的确定性二维贴地放置

## 入口与语义边界

`prism_world_terrain` 的可选 `scatter` feature 导出 `scatter_on_heightfield`、
`estimate_heightfield_placement_work` 及独立的 `HeightfieldPlacement*` 类型。
实现为 `no_std + alloc`，读取真实不可变 `PlanarHeightfieldSurface`，不依赖引擎、
GPU、mesh 或用户查询闭包。既有 `scatter_with_heightfield` /
`scatter_with_indexed_heightfield` 仍只删除原三维候选，不改变其身份、坐标或缓存语义。

此入口先按全局 XZ 网格生成二维候选，再把 Y 放到真实三角面高度加整数垂直偏移。
XZ 半开请求没有 Y slab，因此高山不会因为源随机 Y 不在薄离地带而丢掉所有候选。
每个完整网格方格恰有一个原始候选，但最小距离、地形规则及作者密度仍可删除它；
不保证每平方米最终数量、最大填充、泊松输入过程或蓝噪声频谱。

本片主要记录 CPU 平面放置；对应的固定容量 GPU twin 已由
`prism_world_gpu::GpuHeightfieldPlacement` 交付，ABI、实机回归和限定性能见
`prism_world_gpu_heightfield_placement_zh.md`。两者都不是行星/任意朝向表面投影、曲率/数据层/
跨散布层规则、风/交互、索引增量编辑或产品 scene/render graph 接线。
完整 placement 删除及独立 `PPLD` 持久 codec 现由
`prism_world_placement_deletions_zh.md` 交付；既有三维候选删除 codec 不能替代它。

GPU twin 与 CPU 入口共享 placement/source identity 和精确投影语义，但不是把 CPU 幸存点上传到 GPU：GPU 固定容量地生成 canonical XZ source，执行水平 Matérn-II、真实三角面 projection/过滤、最终 XYZ 作者场，再写 compact/indirect。其 ABI 为 64B source record、64B surface sample、16B density/species/accepted/rank attributes，以及 `[vertex_count, accepted_count, 0, 0]` indirect 参数；owned 逻辑 buffer 预算为 `148*max(C,1)+64*max(B,1)+4*max(S,1)+468` 字节。完整 brush/heightfield snapshot、binding/dispatch 和 candidate/sample/field work 先做预算检查；可恢复上传失败保留旧请求、缓存键和 GPU 输出，projection overflow 在过滤前使整批 fail-closed，不返回部分结果。这里的预算是逻辑 GPU buffer 预算，不包含 host scratch、Arc/Vec 元数据或驱动内部显存，也不是时间上限。

2026-10-16，Apple M2 IntegratedGpu / Metal 真实设备测试 7/7 通过，覆盖精确投影与法线、作者场、分区与 i64 边界、缓存/尾部、失败事务、投影溢出和真实 indirect draw。release benchmark 使用 8 次 warmup、64 次 sample，覆盖 10k/100k primary × 0/150mm；submit+wait P50/P95/P99（µs）依次为 `905.833/935.417/979.333`、`1097.500/1134.333/1161.625`、`1519.209/1893.292/2011.833`、`2059.167/2140.291/2407.000`。这是当前 Apple M2、驱动和 fixture 的限定主机墙钟结果，不是 GPU timestamp、跨硬件承诺或完整产品帧时；GPU twin 仍不等于 scene/render graph 接线、行星曲率、数据层或跨散布层规则已完成。

## 候选身份、顺序与重建

候选沿用 `ScatterCandidateKey` 的全局整数 X/Z jitter，源 grid Y 固定为零；
原随机 Y 被丢弃。`HeightfieldPlacementKey` 由 placement 算法版本和完整源键组成，
与 scene 拥有的放置节点共同定位对象，不使用 Vec 下标、截断 hash 或 PersistentId
冒充候选身份。placement 版本当前为 1；源 kernel 版本也必须匹配。

seed、调用方 version/salt、网格间距、全局 grid X/Z 参与源身份；terrain、offset、
最小距离、请求边界、过滤器与作者场改变几何/成员，不改变同一原始候选身份。
这些可变依赖仍全部进入缓存。不同作者节点应使用各自的 scene owner/namespace。

`key.rebuild(surface, vertical_offset_mm)` 重建贴地几何及精确 surface sample，
不证明请求归属、最小距离、地形规则或作者场接受。未知 placement/source 版本、
非零 grid Y、零间距、越界 raw 坐标、缺表面覆盖或投影溢出明确报错。
直接调用 `key.source.rebuild()` 只得到原三维随机点，不能当作贴地重建。
输出因此是独立的 `HeightfieldPlacementPoint`，不是修改 Y 后的 `ScatterPoint`。

生成按全局 grid X、再 Z 的字典序输出；points 与 density/species attributes 一一对齐。
请求按最终 XZ 整数位置半开归属，允许边界不对齐网格。相同全局配方及一致的地形/
作者快照下，任意 XZ 分区、反向请求或丢弃后重建得到相同键、位置及属性集合。
由于 exclusive endpoint 也是 i64，单独的 X/Z=`i64::MAX` 无法被半开请求拥有；
`i64::MIN` 与 `i64::MAX-1` 可在有覆盖时正常参与，Y 则允许完整 i64 范围。

## 水平最小距离

`min_distance_mm = 0` 关闭距离筛选；正值为水平 XZ 严格排斥半径，距离等于半径
允许共存。采用 Matérn-II 原候选筛选：一个候选只要有更低 rank 的原始近邻便被删除，
不要求该邻居最后幸存。因此不需要递归、请求内贪心状态或跨请求共享接受表。

rank 使用现有 `pcg_hash_u32` 组成的 64-bit mix：
`mix(v) = (pcg(v)<<32) | pcg(v xor 0x9e3779b97f4a7c15)`；初值为
`mix(0x706c6163655f787a xor placement_version)`，依次混入 source kernel 版本、
seed、调用方 version、salt、spacing、grid X、grid Z。负坐标以模 2^64 整数值参与。
rank 相同时以完整 placement key 决定严格全序。半径、请求、terrain 与作者场
不参与 rank；半径零时不计算未被消费的 rank hash。

间距 s、半径 r 的 halo 为 `ceil(r/s)`，扫描包括请求/表面外的全局邻格，
不查询这些邻格的 terrain/field。随后会被地形或作者密度拒绝的近邻仍可阻挡；
不能将 halo 限制在本次请求或先对邻居作地形过滤，否则分区边界会改变结果。
grid 邻域在 i64 范围饱和裁剪；raw 世界坐标超出 i64 的候选不参与距离，但探针仍计数。
坐标差扩宽到 i128；仅在两个绝对分量均小于 u32 半径后以 u128 求平方和。

水平间距至少 r 意味着投影后的三维欧氏距离也至少 r，哪怕地形陡峭或取整；
它不是沿地表测地线均匀密度。原候选筛选可能保留空洞，不作补点。

## 精确投影、过滤及作者采样

每个水平筛选幸存点读取真实 heightfield 的四角，沿已有 a-c-b / b-c-d 三角拓扑
求 i128 有理高度和朝 +Y 几何法线，非双线性插值。详情见
`prism_world_heightfield_scatter_zh.md`。完整非空请求 XZ 必须被 surface 覆盖，
即使作者密度零也不能掩盖缺失地形；合法空请求不要求覆盖。

先对精确表面高度加 `vertical_offset_mm`，再四舍五入到整数毫米，恰好半毫米时
远离零。必须对“精确高度 + offset”的整体取整；例如 -0.5mm + 1mm 得到 +1mm，
不能先把 -0.5mm 取成 -1mm 再加 1mm 得到 0。误差不超过 0.5mm，不先转 float。
精确样本包含的仍是未加 offset 的原高度和几何法线。

高度分子允许超过 i64；只有最终取整后的 Y 必须能表示为 i64。
无法表示时整个请求返回 `ProjectionOverflow`，不夹断或忽略；该检查先于地形/
作者过滤，因此不能用拒绝规则掩盖错误几何。宽位中间算术能容纳 surface 的
i16×i64 解码、插值及 i64×u32 偏移。

随后按 `HeightfieldPlacementFilters` 检查未取整、未加偏移的地表世界高度闭区间，
以及精确的 `max_slope_rise/max_slope_run`。run 必须正，rise=0 只接受水平面；
复用现有避免宽位平方溢出的双轴坡度比较。这里没有 clearance：垂直偏移已明确
决定所有点的离地量。法线不做平滑或 renderer 特定的旋转/缩放转换。

最后 `ScatterDensityField` 在最终取整、已偏移的 `[X,Y,Z]` 采样 noise/warp/刷子/
species，并用完整 canonical source key 的稳定作者 rank 接受密度。
不是在原随机 Y 或未偏移表面采样。作者场使用现有线性整数 kernel；没有隐式
构建 BVH 或复用旧三维候选的场属性。每个最终属性与该点的场采样一致。

## 工作、容量与失败边界

`estimate_heightfield_placement_work` 无分配、无 terrain/field 枚举。令 N 为与请求
相交的 XZ grid 方格数、h=`ceil(r/s)`、W 为 `field.work_per_sample()`，保守上界为：

- candidate probes：r=0 时 N，否则 `N*(2*h+1)^2`，含 primary 与重复邻域重建。
- terrain sample reads：`4*N`。
- author work：`N*W`，W 包括作者选择 rank 的逻辑单位。

所有乘法先 checked；输入、布局算术、完整覆盖和三个工作预算在分配/枚举前检查。
预算不是时间上限或 CPU 指令计数。实际 counters 分别记录执行的 primary/邻格探针、
水平距离幸存者的四角读取、地形规则幸存者的作者工作；作者密度拒绝仍消耗场工作。
请求边界、世界边界和提前拒绝可使实际值小于上界。

仅最终接受的行计入 `max_output_points`，超出时报错，不返回截断前缀。
points 与 attributes 各一次 fallible exact reserve，预留行数为 N、count limit、
`max_output_bytes/(size_of(point)+size_of(attribute))` 三者最小值；枚举中不增长。
分配后仍检查两列的实际 capacity 字节，spare capacity 即使最终无点也计费。
额外接受一行前检查 count/bytes，错误时本次局部结果销毁，旧输出和输入快照不变。
不以最终 len 代替实际 capacity，也不把新 Vec 预留解释成零分配。

容量预算不含 Vec header、分配器元数据、栈、已有 terrain/field payload 或消费者副本，
不构成进程峰值 RSS。构造快照与 field 的预算由各自 owner 负责。
输出为 O(N) 容量，查询无逐点临时分配；线性作者采样为 O(B)，有距离筛选时
候选生成上界为 O(N*(2h+1)^2)，不保证半径远大于间距时仍快。

空请求仍验证 grid spacing、region 与 filters，但其 work/输出容量均为零，无需地形
覆盖；不会用空区域掩盖非法参数。工作不足不通过减少采样或改变配方来“适配预算”。

## 完整缓存与快照

`HeightfieldPlacementCacheKey` 包含 placement/source/surface/field 四个算法版本、
全部 config/region/filter、完整不可变 terrain 与完整作者场。offset、最小距离、
inactive 过滤/场参数、embedding、tile/spec/revision/实际样本全部参与 equality。
预算、Vec 容量及执行次序不参与语义身份。

clone 共享 Arc payload，不深拷贝；结构化相等性最坏与样本数/刷子数成正比，
不是恒定时间摘要。保留输出会延长快照寿命；新 terrain/field 应发布新快照，
不能静默改写旧派生结果。重烘或改 offset 可以保持候选键而改变几何和完整缓存。

## CPU release 测量

2026-10-16，Apple M2、aarch64-apple-darwin、macOS27.0.1（26A434）、
rustc1.96.1（31fca3adb，LLVM22.1.2），正常 workspace 的 release/bench profile。
每档8次预热+32次样本，nearest-rank P50/P95/P99取第16/31/32项。
主机墙钟包含完整生产调用与两列输出分配，排除fixture/oracle构造、结果检查、
打印及销毁；没有时间通过阈值，不是GPU timestamp、产品帧时或跨机器承诺。
未证明整机无其他负载，不能把尾部波动归因于某一kernel。

真实65² i16高度场为PositiveY root，scale=1mm，ramp=`u_q16/256-v_q16/256`，
origin=`[-32000,0,-32000]`，sample spacing=1000mm，共享原高度载荷8450字节。
请求XZ=`[-5000,5000)²`，grid spacing=100mm，恰10000 primary；seed=
`0x0123456789abcdef`、version=1、salt=22、offset=7mm；地表高度闭区间
[-20,20]mm，最大rise/run=1/100。三枚全opacity硬球按ID3/15/24依次合成，
输入刻意逆序；base density=40000/species=7、field seed=`0xfedcba9876543210`、
version=1、无noise/warp。具体球心/半径/Replace/Add/species由benchmark完整打印。

| 水平半径 | 最终点数 | candidate探针 | 四角读取 | 实际field work | P50 / P95 / P99（µs） |
|---|---:|---:|---:|---:|---:|
| 0mm | 4571 | 10000 | 40000 | 30916 | 1036.750 / 1085.833 / 1094.375 |
| 150mm | 683 | 122011 | 5824 | 4516 | 6245.959 / 8721.583 / 10955.792 |

两档输出两Vec capacity均10000：point1760000字节、attribute80000字节，
合计1840000字节；计时内恰2次分配，未因实际点数较少而少计容量。容量上限
16MiB、count上限10000；共享field刷子192字节与高度8450字节另计。
work准入：candidate上界10000/250000，terrain均40000，field均40000。
半径0删除2271个地形不合格点和3158个作者不合格点；半径150先删除8544个距离
不合格点，再删除327/446个地形/作者不合格点，首个阻挡来自请求外的候选有131个。

全部80次计时调用都在计时外比较完整有序点/精确样本/属性/cache/counters，
逐键重建并检查全部输出对的水平间距。参考算法独立实现扩宽PCG、排名、三角
重心插值/叉积法线/整体取整、硬球作者合成与密度接受；正半径用104²原始候选
全对比较成员，不复刻生产halo遍历，并代数推导首个阻挡的探针次序。
这不是先运行生产函数生成“期望值”，也不是图像/频谱或最大填充质量验收。

## 最终回归与 GPU 验证边界

2026-10-16，正常 workspace、离线固定锁依赖、独立 target：`scatter` 配置
109项库测试、26项新放置集成测试、11项既有索引地形集成测试全部通过；
`--no-default-features` 配置78项库测试通过，新旧scatter集成入口均为0项，
feature门控正常。两配置文档测试均0项，两配置all-target严格Clippy
`-D warnings`、本片五个Rust文件的rustfmt检查与定向`git diff --check`通过。
新增实现、测试和benchmark中没有未实现分支或占位实现。

新26项测试覆盖独立三角平面/整数排名/全对距离oracle、负半毫米及跨零偏移、
最终Y作者采样、原候选阻挡、非对齐XZ反向分区、精确高度/双轴坡度、容量与工作
预检、投影溢出及i64边缘、完整缓存与公开rebake快照。不能以这些测试替代
蓝噪声频谱、产品视觉质量、整世界吞吐或其他后端验收。

另对**既有三维GPU地形过滤**运行物理设备回归。正常workspace命令先被并行
修改中的`prism_tasks/src/arena.rs:225`缺少`Layout`导入阻断，未到GPU执行；
没有修改该依赖或资产/GPU文件来消除错误。最小独立Cargo验证包直接引用
未修改的`prism_world_gpu`生产crate和原始`tests/scatter_heightfield.rs`，
只排除其无关开发依赖，使用自身离线生成的锁而非workspace锁。
Apple M2/Metal IntegratedGpu、无optional GPU feature，8项测试连续两轮
全部通过，0失败/忽略；第二轮以`--offline --locked`复跑。包括真实高度的
CPU逐项对照、宽位/世界边缘、缓存/失败保留、XYZ分区及真实indirect draw像素。
GPU crate文件集合的前后SHA-256相同：
`1da7c9bbd5e71f44fb43a9c997733c1b999e79bd308af12b01acd1ec3714e123`。

这是现有GPU过滤链的隔离回归，不是正常workspace GPU整包通过，
也不替代新二维贴地放置的 GPU twin 证据；新 GPU 证据见
`prism_world_gpu_heightfield_placement_zh.md`。
本机验证包manifest/lock保留在`/private/tmp/prism-placement-gpu-regression-5Ncrl4`，
不纳入源码提交；构建输出与CPU测试共用本片专用target。

验收后已用`cargo clean --target-dir`分别清理本片主验证目录
`/private/tmp/prism-heightfield-placement-2P48EW`（918.9MiB）与并行测试目录
`/private/tmp/prism-heightfield-placement-tests-ibV3m8`（150.0MiB），并确认两个
target路径均已不存在。清理对象仅为可重建编译产物，未清理其他任务的target；
上述小型验证包manifest/lock保留供复现。

## 复现

在仓库根目录执行，建议使用独立 `--target-dir` 并在记录后清理对应目录：

```text
cargo test --offline --locked -p prism_world_terrain --features scatter
cargo test --offline --locked -p prism_world_terrain --no-default-features
cargo clippy --offline --locked -p prism_world_terrain --features scatter --all-targets -- -D warnings
cargo clippy --offline --locked -p prism_world_terrain --no-default-features --all-targets -- -D warnings
CARGO_INCREMENTAL=0 cargo bench --offline --locked -p prism_world_terrain --features scatter --bench heightfield_placement
```
