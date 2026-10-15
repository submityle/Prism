# 真实 heightfield 查询与散布过滤

## 范围和入口

`prism_world_terrain` 默认导出 `PlanarHeightfieldSurface`，对真实烘焙的
`HeightfieldTile` 查询精确三角面高度与几何法线，无需构建 mesh。可选
`scatter` feature 导出 `scatter_with_heightfield`，组合既有源散布、作者密度/
种类场和地形过滤。两种配置均为 `no_std + alloc`，不引入引擎或 GPU。

`scatter_with_indexed_heightfield` 则直接组合 `ScatterBrushIndex` CPU 空间 BVH
和同一个地表过滤内核。它显式接收已构建索引与有界可复用 `ScatterBrushScratch`，
不在每次请求中重建索引或增长 scratch；线性入口仍保留，调用者按负载选择。
两入口的 points、attributes、完整 cache key、candidate_evaluations 和
surface_sample_reads 一致，只有 `field_work` 按各自实际探针计数。

这条 CPU 路径只删除原始候选，不移动、投影或补点。因此保留源 candidate key
重建、最小距离和任意 XYZ 半开请求分区的成员不变性。离地带很窄时可能稀疏甚至
为空；这不是保证贴地密度的植被放置器，也不是 GPU 地形过滤或产品 renderer 接入。
源候选中被其他候选阻挡、随后因密度或地形被删除的位置不会被回填。

## 明确的空间与插值契约

`PlanarHeightfieldEmbedding` 显式指定样本 `[0,0]` 的世界整数毫米位置和正的
`u32` 样本间距。tile 的 cube-sphere face/key 不隐含平面世界位置，不能把该接口
解释成行星表面查询。XZ 覆盖为从 origin 到 `origin + (side-1)*spacing` 的半开矩形；
端点超出 `i64` 时构造失败，零间距失败。内部格线归正侧 cell。

每个 quad 的四角为 a=`[x,z]`、b=`[x+1,z]`、c=`[x,z+1]`、d=`[x+1,z+1]`。
它沿用现有 `HeightfieldMesh` 的 a-c-b / b-c-d 三角形，不使用双线性插值。
cell 内余数 `rx+rz <= spacing`（含对角线）属于第一片，否则第二片。
对非共面四角，两片的几何法线可不同，对角线高度连续。

`HeightfieldSurfaceSample` 返回未约分的 `i128` 高度分子和等于 spacing 的
`u32` 分母；分子已包含世界 origin Y。法线为朝 +Y 的未归一化
`[-dhx,spacing,-dhz]`，不是 mesh 的平滑顶点法线。源为实际解码的
`i16 × i64` 高度，所有几何运算扩宽到 `i128`，不先转 float 或把高度夹到 i64。
构造及单次成功查询不分配；每次查询读取四个角样本。

## 三个闭区间/比例过滤

`HeightfieldScatterFilters` 明确控制：

- `surface_height_mm`：地表的绝对世界 Y 高度闭区间，不是候选 Y。
- `clearance_mm`：候选 Y 减精确地表 Y 的有符号垂直距离闭区间，不是到面的最短距离。
- `max_slope_rise / max_slope_run`：几何面的非负最大 rise/run，run 必须正；rise=0
  只保留水平面。它不使用角度或浮点近似。

高度与 clearance 通过分母交叉相乘精确比较；不在毫米取整后才判断端点。
坡度判断等价于 `(dhx²+dhz²)*run² <= spacing²*rise²`，但实现先将每个
缩放分量与 `spacing*rise` 比较，只有可容纳的分量才进入 `u128` 平方，使用
减法避免两平方和溢出。极端解码高度不需要其平方能表示在 i128 中。
区间倒置或 run=0 是显式错误，不静默修正。

非空请求的完整 XZ 足迹必须在 surface 内，即使密度为零也不能掩盖缺地形。
任一轴为空的合法请求无需地形覆盖。地形覆盖、输入和预算失败不返回前缀结果；
函数不修改 surface、field 或既有输出。

## 缓存、快照和预算

surface 持有不可变 `Arc<HeightfieldTile>`，clone 共享样本，不隐式深拷贝。
`HeightfieldScatterCacheKey` 包含完整源/作者场 key、surface/filter 算法版本、
全部过滤字段、完整 surface embedding 和 tile 快照。相等性包含 tile/spec/revision
和全部实际样本，不用 tile+revision、Arc 地址或摘要替代内容身份。
即使同 tile/revision 独立烘焙出不同样本，也不能错误命中缓存。

缓存 key clone 只共享 terrain/field；结构化 equality 最坏 O(terrain 样本数 + brush数)，
并非恒定时间指纹查找。保留输出会延长快照驻留；重烘后调用者应发布新快照，旧结果
仍引用旧地形，不会被悄悄篡改。共享载荷的驻留/淘汰由宿主预算负责。

地形预算在源枚举/分配前以 `4 × primary_candidates` 做 checked 预检，独立于
源/halo 工作和作者场工作预算。它是保守的逻辑样本读数，不是实际执行时间上限；
`surface_sample_reads` 记录最终查询的 field 幸存数 × 4，通常小于预检上界。
实际总成本为既有源生成 + 作者场求值 + O(field幸存数) 的地形查询/过滤。

过滤原地稳定压紧原来的 points/attributes 两个 Vec，不引入第三个逐点输出，保留
对齐、源顺序和原点坐标。truncate 不缩容量，`field_budget.max_output_bytes` 仍覆盖
两 Vec 的实际容量载荷；这不是 allocator/RSS 峰值限制。Arc 快照、field 既有刷子、
分配器元数据和外部副本均不在该输出预算内。

索引入口也先执行同一个地形预检，再进入 indexed source 的工作、scratch 和输出
准入；空请求仍检查 scratch 容量。新增 `HeightfieldScatterError::Index` 保留完整
`ScatterBrushIndexError` 及错误链，地形预检错误仍使用原有错误类型。它没有新的
语义版本或缓存身份；索引、scratch 的容量/布局不进入派生结果 key。既有
`filter_heightfield_scatter` 可直接消费其输出，持久删除仍按完整 candidate key
匹配并保持属性对齐。索引和 scratch 的实际容量载荷另计，不能藏入输出预算；
全重叠时遍历、命中排序及精确合成可能比线性更慢。

## 复现入口

在仓库根目录执行：

```text
cargo test --offline -p prism_world_terrain --no-default-features
cargo test --offline -p prism_world_terrain --features scatter
cargo clippy --offline -p prism_world_terrain --features scatter --all-targets -- -D warnings
CARGO_INCREMENTAL=0 cargo bench --offline -p prism_world_terrain --features scatter --bench heightfield_scatter
```

基准先烘焙真实 65×65 ramp：样本高为 `u_q16/256-v_q16/256` mm，scale=1，
世界 origin `[-32000,0,-32000]`、spacing=1000mm，共享高度载荷 8450 字节。
独立查询为 100k 个跨正负位置/cell/两片三角的固定位置，每次计时内逐个校验预计算
精确结果，不能称为孤立插值指令耗时。完整散布为 20×25×20=10k 个 100mm 源 voxel、
最小距离0、density40000 加 noise311/amp10000 和 warp127/amp201，无刷子；地表高度
[-100,100]mm、clearance[-250,250]mm、slope≤1/10。计时含源生成、作者场、地形查询及
压紧，排除 fixture 构造、oracle 准备、结果校验和销毁。每次完整结果在计时外逐点/属性
与标量 oracle 比较，额外检测分配次数和容量稳定性。

### 2026-10-16 原始线性入口 release 实测

原始切片验收时，正常 workspace 默认配置 78 项、`scatter` 配置 93 项单测全部通过，0 失败/忽略；
两配置 all-target Clippy `-D warnings` 通过。新增 19 项查询与 15 项过滤回归覆盖
精确三角面 oracle、极端整数、完整缓存身份、XYZ 分区、预算及属性对齐。公开
`HeightfieldTile::rebake` 回归另验证独立 owner 发布新 Arc 后，新旧快照/选择隔离，
以及末样本编码失败不会泄漏前缀或 revision 变更。

本机 `aarch64-apple-darwin`、macOS 27.0.1、rustc 1.96.1，8 次预热/64 次测量，
nearest-rank 主机墙钟，单位 µs。正常仓库源码直接编译，使用隔离 target
`/private/tmp/prism-world-sector-build-EFoWO8`，没有复制代码或修改依赖来通过验证。

| 工作负载 | P50 | P95 | P99 | 计时内分配 |
|---|---:|---:|---:|---:|
| 100k 精确地形查询 + 逐结果校验 | 1705.166 | 2613.750 | 3803.084 | 0 |
| 10k 源生成 + 作者场 + 地形过滤 | 3514.916 | 3930.916 | 5631.916 | 2 |

查询 digest 为 `e64046a006c8c0f2`；完整散布每次作者场后 6132 点，过滤后 1250 点，
实际地形读数 24528，保守准入读数 40000，两列容量载荷 880000 字节。查询的 72 次
100k 全结果比较和完整散布的 72 次逐点/属性比较均通过；共享地形载荷 8450 字节
另计，不是内存峰值。原始全局格对齐请求为 `[-1000,-1200,-1000]` 到
`[1000,1300,1000]`；基准对估算候选数显式断言，未放宽预算来掩盖边界格。

首次试运行使用未对齐的 Y `[-1250,1250]`，实际覆盖26格，作者场预检以
`required343200 > limit330000` 拒绝；修正 fixture 到25格后才取得上述完整结果。
这一失败不算通过。并行任务/调度可能影响尾延迟，不据此作跨硬件、无干扰 A/B、
GPU timestamp、产品帧时或完整散布性能达标承诺。

这些入口是局部 CPU 内核验证，不代表全 workspace、GPU 地形过滤、风/交互、曲率/
数据层/跨散布层过滤、贴地放置、蓝噪声画质或产品级性能验收。

## CPU 空间索引与真实地形组合 A/B

复现新增的完整生产调用基准：

```text
cargo test --offline --locked -p prism_world_terrain --features scatter --test indexed_heightfield_scatter
cargo bench --offline --locked -p prism_world_terrain --features scatter --bench indexed_heightfield_scatter
```

本轮正常 workspace `scatter` 配置 109 项库测试和新增 11 项公开集成测试全部通过，
0失败/忽略；`--no-default-features` 78 项库测试通过，两配置 all-target Clippy
`-D warnings`、本轮文件格式和 diff 检查通过。集成测试以原线性入口为结果参考，
覆盖无刷子/空请求、noise/warp与非交换ID顺序、稀疏/全重叠实际work、i64极值、
三轴非对齐逆序分区与源最小距离、精确/少一预算、预检顺序/溢出/错误链、完整
缓存依赖、完整键删除组合及跨索引/失败后的scratch复用。独立源码审查未发现
可复现缺陷；这些是局部契约证据，不是完整系统验收。

2026-10-16，Apple M2、`aarch64-apple-darwin`、rustc 1.96.1，正常 workspace
release/bench profile，独立 target `/private/tmp/prism-indexed-terrain-target-32CI9Y`。
每类负载 8 预热对 + 32 测量对，交替 AB/BA，nearest-rank P50/P95/P99
为第16/31/32名。计时包含源生成、作者刷子场、真实地形过滤和两个输出分配；排除
fixture/index/scratch 构造、参考输出、完整校验、打印和析构。不是 GPU 时间、
无干扰隔离实验、产品帧时或性能阈值门禁。共享宿主并行负载可影响尾延迟。

地形仍为前述真实65×65 ramp、8450字节高度载荷。请求为
`[-1000,-1200,-1000]..[1000,1300,1000]`，100mm格、10000 primary，
最小距离0；seed=`0x0123456789abcdef`、version1、salt22。场 seed=
`0xfedcba9876543210`、base density40000/species7、关闭noise/warp。
1024刷子按逆序输入ID，规范合成为ID升序；opacity40000、species=ID，
ID%3对应 `Replace(30000+ID*17%20000)` / `Add(1000)` / `Add(-300)`。
稀疏中心为 `[-1500+(ID%16)*200,-900+((ID/16)%4)*600,-1500+(ID/64)*200]`，
内/外半径40/80mm；全重叠中心 `[0,0,0]`、半径200/10000mm。
过滤高度[-100,100]、离地带[-250,250]mm、坡度≤1/10。

| 负载 | 线性 P50 / P95 / P99（µs） | 索引 P50 / P95 / P99（µs） | 场后 / 地形后点数 |
| --- | --- | --- | --- |
| 1024稀疏刷子 | 23322.709 / 24135.250 / 24262.958 | 1049.458 / 1170.041 / 1262.417 | 6075 / 1209 |
| 1024全重叠刷子 | 84574.709 / 89875.833 / 93354.000 | 131862.875 / 136042.167 / 138091.750 | 7310 / 1461 |

两边candidate_evaluations均10000；稀疏线性/索引实际field_work为
10250000/124572，全重叠为10250000/30720000；实际surface reads分别为
24300、29240。每次两Vec容量均为10000行，points800000字节、attributes80000字节，
合计880000字节、恰两个分配；scratch固定1024索引/8192字节，无请求内增长。
索引2047节点/139200字节、完整作者刷子载荷65536字节、共享地形另计。

有限预算为source10000 probes/10000 points/16MiB，field30720000 work/16MiB，
terrain40000 reads；构造最多1024刷子/65536字节、索引262144字节、scratch8192字节。
线性/索引各自最坏场准入为10250000/30720000，不以较低实际work绕过预检；排序
比较、CPU指令和时间不是逻辑probe预算。全部160次计时调用（含预热）在计时外
核对完整points/attributes/cache/counters及固定容量，每条路径与自身参考的
field_work也完全一致。全重叠索引明显退化，因此不默认取代线性扫描。
