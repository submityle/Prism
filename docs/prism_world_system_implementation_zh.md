# Prism World 系统实现记录

日期：2026-10-20

本文只把已经存在源码、测试和提交记录的部分标为已交付；设计文档中的路线图、指标和效果目标仍然不能替代实现证据。

2026-10-22 `prism_world_terrain::HeightfieldLodSeamPlan::build_strict_for_residency` 新增从严格排序且唯一的 resident `TileKey` 切片推导 strict 2:1 heightfield seam 邻接：同级、coarse parent、finer children、跨 cube-face neighbor edge/reversal 与 finer coverage 均从 tile topology 派生；不完整 finer pair、重复/歧义覆盖和无序输入 fail-closed。resident 输入由调用方复用有序快照；每个 owner edge 仅做固定数量的二分查找，邻居临时表预留上限为 8，额外工作 O(log N)、输出/临时空间 O(1)。`CARGO_TARGET_DIR=/private/tmp/prism-target-world-lod CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 cargo test -p prism_world_terrain --lib --offline resident_adapter` 为 2 passed，整 crate 同配置 `cargo test -p prism_world_terrain --lib --offline` 为 114 passed；指定文件 rustfmt check 与 `git diff --check` 通过。测试覆盖同级、coarse、finer、跨面、无序、歧义和不完整 finer coverage。此接口只生成 seam plan，不生成/绘制 terrain mesh，不管理 resident snapshot 生命周期，也不证明跨 LOD GPU/render graph 接线或产品性能；mixed-LOD 产品验收仍未完成。

2026-10-23 新增真实 planar `HeightfieldMesh::apply_lod_seam_plan` consumer：要求 seam plan tile 与 mesh 相同、segments 等于 samples-per-edge 减一；将 seam transition 的局部 grid indices 映射回 mesh 既有顶点，剔除对应 coarse edge boundary quads，再附加 transition strips，避免 overlay 与原边界面重叠。每次操作先验证最多两条相对 coarse 边、索引前置长度与输出预算，按 O(S² + T) 生成有界临时 index buffer（S=segments，T=transition indices），完整成功后单次替换；不增加顶点/法线/UV buffer，失败保留旧索引。对 seam 后 payload 使用显式 `max_bytes` admission；mesh 生命周期只接受一次 seam 应用，再次应用须重建，避免重复删面。测试验证单 coarse 边索引数、全部索引落在原顶点集、重复应用与预算失败原子性，以及 transition +Y winding/接口边界。验证：隔离 target `/private/tmp/prism-target-world-seam-consumer` 下 `cargo test -p prism_world_terrain --lib --offline` 117 passed、`cargo check -p prism_world_terrain --no-default-features --offline` 通过；指定文件 rustfmt 与 `git diff --check` 通过。限制：该 CPU consumer 当前只接受零至两条互为对边的 coarse transitions；相邻 coarse corner fan、finer-neighbor stitching、planet cube-sphere mesh index integration、GPU/RHI renderer graph draw 与视觉/帧时验收仍未完成。

## 本批交接与验证范围

2026-10-15 按共享磁盘协调要求停止共享 target 构建；本批只整理 `prism_world_io`、`prism_world_terrain` 与本文，测试使用独立的 `/private/tmp` target，未清理或覆盖共享 target，未 push。

- 地形曲面提取使用 `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/private/tmp/prism-world-terrain-target cargo test --offline -j 1 -p prism_world_terrain --no-default-features`：59 passed、0 failed；随后启用 `mesh` feature 重跑为 62 passed、0 failed，包含原生 Mesh 转换回归。
- 新增 threaded 文件 provider 与 16 项回归测试源码，使用 `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/private/tmp/prism-world-io-target cargo test --offline -j 1 -p prism_world_io` 完成 28 passed、0 failed（其中 threaded 回归 16 项）。
- 本批运行的是指定文件的 `rustfmt --edition 2024 --config skip_children=true --check` 和 `git diff --check`，仅验证格式、解析及补丁空白，不等同于类型检查、clippy 或测试。
- `prism_world_io` 全部 target clippy 与 `prism_world_terrain` no-default 全部 target clippy 已通过；terrain `mesh` feature clippy 在编译依赖 `prism_asset` 时因磁盘耗尽中断。性能/GPU 验证及全 workspace 测试仍未运行；本批使用 `/private/tmp` 隔离 target，没有清理或覆盖共享 target。

后续异步 runtime 接线已补齐至固定 worker 文件到 `SceneHost` 的链路，并接入任务池 Future、共享取消和动态 PriorityCell，详见 M3、`prism_world_async_runtime_zh.md` 与 `prism_world_shared_priority_zh.md`；跨 LOD 地形接缝和产品性能目标仍按下文边界保留。本记录不构成完整系统验收。

## 已交付切片

### M0-M2：空间、流送和常驻

- `prism_world_stream`：`i64` 毫米坐标、负坐标 `div_euclid/rem_euclid`、sector/Morton locality、cell manifest、需求集、预测预取、优先级、预算、迟滞驱逐和可回放 frame trace。当前 scheduler 还记录已提交的 `scene_key/content_version/derived_hash`；同一 sector 的 manifest revision 变化会重新进入既有 load admission，而不会被“按 sector 已驻留”短路。
- manifest 输入现在有统一的唯一性 preflight：`prism_world_stream::validate_manifests`、`StreamScheduler::try_plan`、`CellFrameCoordinator::try_plan_frame` 与 `WorldRuntime::try_tick` 在任何 frame counter、resident/loading 状态、IO admission、SceneHost 调用或 residency policy 推进前拒绝重复 `SectorId`；runtime 同时拒绝重复 `LoadedCell`。错误携带首条/重复输入位置，避免 stream 按最后一条预算而 runtime 按第一条 payload/header 校验的分裂语义。旧的非 fallible `plan/plan_frame/tick` 保留兼容并对非法输入显式 panic；外部输入生产路径应使用对应 `try_` API。本片只解决 manifest 身份一致性，不包含资产解压、异步 IO、renderer 或 GPU 接线。
- canonical Data Layer admission 已进入 `prism_world_stream`：`DataLayerMask` 使用固定 64-bit ABI，拒绝重复及超范围 layer id；`CellManifest::data_layers` 与 `DemandSource::data_layers` 在同一 scheduler 中完成无逐帧 layer-list 分配的子集准入，layer mask 变化会触发 stale reload，空间仍满足但 layer 不可用的 resident 会被确定性驱逐。runtime 的 `LayeredCellManifest`、`RuntimeDataLayers` 和 `LayeredCellFrameCoordinator` 现在复用该 mask，并将可用层集合传给普通 stream coordinator；cell header 扩展为 52 字节并校验 layer mask，避免 manifest 与 payload header 的层集合分裂。该切片仍不解析 layer payload，也不等价于 ECS、terrain、GPU 或 renderer 的产品层接线。
- `prism_world_residency`：A/B/C 三档状态、物化预算、确定性 promotion/demotion、HLOD 连续 alpha/dither、QoS governor、稀疏 SDF brick 和 PCG 图求值/脏传播/cache key。
- `prism_world_runtime` 的 cell reload 缺口已补齐：`CellFrameCoordinator` 用 manifest revision 识别 replacement，仍复用既有 `SceneHost::load_cell` 边界，成功后提交新 manifest；header、provider、取消、背压或 host 失败时恢复旧 manifest 的 resident 状态。该片只覆盖 cell payload replacement，不冒充 `prism_scene::SceneReloadPlan` 的模板级增量 ECS 传播，也不提供 host 内部原子回滚。
- 数据层切换期间的 in-flight 请求现在有明确的 stale-completion fence：层变为不可用时只标记并取消对应 sector，底层 IO 必须消费终结事件后才释放身份/配额；迟到成功或失败不会跨过 `SceneHost` 边界，也不会污染旧 resident 状态。data-layer mask 变化仍参与 manifest replacement identity。`prism_world_stream` 9 项与 `prism_world_runtime` 104 项定向回归通过（2026-10-20，独立 `/private/tmp/prism-target`）；这证明的是 CPU lifecycle contract，不是异步 OS 读抢占、GPU 或 renderer 接线。
- `prism_world_runtime` 新增 engine-neutral `RuntimeDataLayers` 与 `LayeredCellFrameCoordinator`：在既有 `CellFrameCoordinator`/`SceneHost` 之上显式区分 unloaded、loaded、activated，按稳定 `DataLayerId` 排序并取 cell 声明层与 runtime-enabled 层的确定性交集；layer-set host 调用成功后才提交本地状态，失败保留旧状态，可在驻留 eviction 后重新加载并重新激活。该 slice 不解析 layer payload，不修改 terrain、GPU、ECS partition 或 render。
- `prism_world_runtime` 新增 engine-neutral `HlodRuntime`：接入已验证的 `HlodCluster` 声明，按稳定 cluster id 排序请求，执行请求数/输出字节/目录容量预检，提供跨帧 hysteresis 与确定性 transition record，并在整批校验、计算和输出分配成功后一次性提交 active level；`insert_bake` 进一步原子接入完整 bake catalog，并可按选中的 asset key 查询 representation descriptor 和 concrete proxy artifact。重复 cluster/representation/artifact、重复请求、未知 cluster、声明错误和预算错误均 fail-closed，失败不污染旧状态。该 slice 不拥有资产容器、GPU buffer、Scene/ECS 真相或 renderer graph。
- 2026-10-23 `prism_world_residency::bake_hlod` 已从声明目录扩展为确定性的 engine-neutral HLOD artifact bake：输入按稳定 proxy id 规范化，生成可复现的 source hash、cluster/representation id、层级元数据、AABB proxy geometry（8 顶点/36 索引）、bounds、material key/参数及 geometry/material/artifact hash；非有限值、负半径、重复输入、生成 ID 冲突、数量/字节预算不足均 fail-closed，空输入也有稳定结果。验证：独立 target 上 `cargo test --offline --locked -p prism_world_residency --all-features` 为 **119 个库测试通过，16 个集成测试通过，1 个 doctest 通过**；组合 runtime catalog 的 `prism_world_runtime --lib hlod_runtime` 为 **5 passed**。artifact 是可传递给下游的具体 CPU 产物，但不等于最终 mesh/material asset package；资产流送、GPU residency/upload、Scene/ECS 单体 materialization、renderer submission、跨后端画面与受控性能仍需独立验收。
- `prism_world`：组合 policy core 的 `WorldRuntime`，并提供 feature-gated `SceneEcsHost`；cell 加载通过真实 `prism_scene::spawn_typed_plan`/`SceneSpawn::attach_instance`，卸载通过 `prism_scene::despawn_instance`，驻留升降级通过 typed provider 回调，不直接写第二份 scene 真相。
- `SceneEcsHost` 的 ownership registry 已覆盖真实 ECS despawn hook panic：token/binding 在 despawn 成功返回前保持可见；shutdown 逐 cell 提交，失败 cell 与待处理 cell 可由兼容 provider 接手后重试。`WorldRuntime::apply_scene_plan` 只在同一应用内跳过已成功卸载目标的后续 tier action，仍是 fail-fast、非事务接口。

### M4：GPU proxy visibility contract

- `prism_world_gpu` 已纳入根 workspace。
- CPU contract、SoA little-endian upload layout、WGSL compute shader 和 CPU golden 对拍已具备。GPU 物理输出为四个规范 `u32` 与一个未量化 `f32 screen_error`（20 字节步长），原始误差直接来自 GPU，不由 CPU 重算或从量化值还原；cluster SoA 保留几何误差位模式。
- WGSL parse/validation、CPU tests 和 `cargo check --features gpu` 已通过。
- 严格门禁命令：

```text
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test -j 1 -p prism_world_gpu --features gpu --test gpu_parity -- --nocapture
```

2026-10-15 跨沙盒实机结果通过：Apple M2、Metal、IntegratedGpu、无额外必需 feature；真实创建 pipeline、dispatch、map/readback，并逐项比较规范四字与原始屏幕误差。测试覆盖 5 个可见/不可见对象、129 个对象的 3-workgroup 尾部、LOD 阈值、量化半步/饱和与空输入。5 个对象上传 368 字节、读回 100 字节；一般非空上传量 `128 + 48×N`、读回量 `20×N`。测试严格拒绝缺少 adapter/device、CPU adapter 或未遵守 `WGPU_BACKEND`，不把跳过算通过。telemetry 耗时是主机墙钟，不是 GPU timestamp，也不构成吞吐或 P50/P95/P99 性能达标证据。

### M4：固定容量 GPU 驻留与间接绘制

`GpuResidentVisibility` 已提供一次资源分配、验证后增量写入、显式 dense prefix、camera/count 上传与纯 encode。GPU 输出唯一 source-slot compact 流和 `[vertex_count, visible_count, 0, 0]`，真实间接实例 draw 已通过像素验证；每个非空 64-lane 工作组只预留一次全局输出区间。生产 encode 不分配 GPU 资源、不上传、不 submit/wait/map/readback。逐帧必须先提交再上传或 encode 下一帧，不能累积多个不同 camera 的 encoder 后一起提交。

2026-10-15 Apple M2/Metal 严格回归：库单测 7 项、visibility shader validation 1 项、parity 1 项、resident 3 项通过，harness all-target 与正常 workspace GPU lib clippy 通过 `-D warnings`。百万实例/75% 可见、release、8 预热+64 样本，主机 record+finish P50/P95/P99 为 21.167/33.125/35.625 µs，submit+wait 为 1371.750/1511.000/1619.583 µs；逻辑 buffer 72,000,144 字节，首次上传 48,000,128 字节，计时帧无上传/回读/绘制。不是 GPU timestamp、物理 VRAM 或完整产品帧时。API、复现方式、三档实测与未集成边界见 `prism_world_gpu_resident_zh.md`，不据此宣称 M4 全部完成。

### M4：扇区所有权与宿主生命周期

`ProxyRegistry` 提供固定代理容量、独立 sector 数量预算（含空 sector）、sector 内唯一 ID、代际 handle、交换压紧和 dirty range；`GpuSectorProxies` 持有同一注册表、预留 SoA scratch 和固定 resident GPU 资源，无变化 upload 为 O(1)，改变后的扫描/转置/清标记总计 O(N + K)，逻辑列上传为 48K 字节。未上传编辑和整体替换 registry 均拒绝 render/camera/encode，尾删和全卸载仍同步 count。返回的校验/预算错误不改变注册表；map/Arc 的普通分配 OOM 不属于可恢复错误承诺。

`prism_world_runtime/proxy_host` 在 header 校验后借用项目 decoder、原 `SceneHost` 与同一注册表，先完成代理校验、容量准入、handle/generation 预计算和 fallible allocation，再调用权威 host；host 成功后才以无可恢复失败的 commit 发布派生 registry 更新。首次 load 与同 sector reload 共用该事务路径，满代理容量时可复用旧 sector 槽位；host 返回错误或 unwind 时旧代理、handle generation、dirty/upload 状态均保持不变。generation 即将进入 `u64::MAX` 的槽位按退休规则排除，不虚报可复用容量。evict 仍在权威 host 成功后才删代理，失败/unwind 保留。它不另立 scene 真相、不规定 payload 格式、不启用 wgpu，也不能回滚权威 host 内部在返回错误或 panic 前自行产生的部分修改。真实 stream demand/coordinator→decoder/host→上传→间接绘制与策略驱逐像素回归已接通；合约、测试方法与性能边界见 `prism_world_sector_proxies_zh.md`。这不是产品 ECS decoder/render graph 的完整接入。

2026-10-20 专项回归：`prism_world_gpu` 默认与 `gpu` feature 库测试均为 123 passed；`prism_world_runtime` 默认库测试为 104 passed，`gpu_proxy` feature 库测试为 124 passed、7 ignored。随后显式执行 `gpu_proxy_cell_host_tests --ignored --nocapture`，7 项在 IntegratedGpu/DiscreteGpu 门禁下全部通过，覆盖真实 GPU 首次加载、同 sector reload、满容量复用、host error/panic 保留旧派生状态、上传与驱逐生命周期。该证据不包含多厂商 GPU、物理显存峰值或产品 scene/ECS decoder/renderer graph。

### M4：GPU 密度覆盖与距离淡出

`DensityPolicy` 以 `(proxy_id, cluster_id, seed)` 的稳定 PCG rank 计算平滑 coverage，不依赖可变 dense slot；可选距离 fade 与 quality coverage 相乘。构造器校验有限 normal/零参数，明确拒绝非零 subnormal；CPU/WGSL 使用相同 f32 缩放 norm、退化 width 与结果规范化规则。普通 resident 不增加资源；可选路径额外 `4×max(capacity,1)+16` 逻辑 buffer 字节，策略变化只写 16 字节 uniform，不遍历/重传列。原几何 record ABI 不变，compact 取几何可见与正 coverage 交集。下游按 compact source slot 读取 coverage，并可调用导出的 4×4 ordered-dither WGSL；部分 coverage 不被材质消费时不构成连续淡出。

2026-10-15 Apple M2/Metal 原始源码 harness 最终 96 项通过、0 失败/忽略（库 79、density 8、resident 6、parity 1、WGSL 2）。真实 draw_indirect 读取 GPU coverage 并逐像素对拍；另测两 seed、129 尾部、交换压紧、全卸载、固定资源、极小 normal range 与大坐标。实机回归发现并修复窄 transition 消减、极小距离 subnormal 余量及 smoothstep 中间平方冲零；明确检查微小正 coverage 仍在 compact 集合内。release 百万实例 density 模式输出 416,295 个 source slot，record+finish P50/P95/P99 为 48.750/57.125/64.125 µs，submit+wait 为 2058.625/2107.625/2136.208 µs；不含上传、回读或绘制，不是 GPU timestamp/产品帧时或严格 A/B 比较。API、三档双模式实测、harness 隔离说明和边界见 `prism_world_density_zh.md`。整数 GPU 候选生成另见下节；作者密度场、风/交互、HZB 与产品 renderer graph 尚须继续完成。

补充验证：no-default 原始库源码 73 项 CPU 测试通过；原始源码 harness all-target clippy 和正常 workspace GPU lib clippy 均通过 `-D warnings`。这两项局部结论不代表全 workspace 或完整产品集成通过。

### M4：可再生 CPU 散布 kernel

- `prism_world_residency::scatter` 在全局整数毫米三维格中每格生成一个 hash-jitter 候选；显式 seed/version/salt 与完整结构化 candidate key 决定位置。半开位置范围负责唯一归属，负坐标使用 Euclidean 除法；任意拆分、合并和请求顺序不会改变成员集合。完整 cache key 包含 kernel 版本、配置与范围，丢弃后可重建，不充当持久对象身份。
- 可选最小距离采用 jittered lattice 上的 Matérn type-II 式筛选，重建未请求邻格的原始候选，以 `(priority hash, full key)` 排名消除边界依赖；不是 Poisson 输入、蓝噪声保证、最大填充或递归 dart throwing。距离比较使用宽整数，等于半径允许保留。
- 预检工作量上界为 `N × (2 × ceil(radius/spacing) + 1)^3`，无筛选时为 `N`；输出数量和 Vec 容量载荷字节另有硬限制。checked 算术、一次 fallible exact reservation，超限整体失败，不返回随预算变化的截断前缀；字节限制不含分配器元数据、栈或外部消费者副本。
- no-default 测试 35 项（其中散布新增 20 项）、all-target clippy `-D warnings` 和格式检查通过，覆盖全局参考筛选、跨 cell 最小距离、未请求邻格阻挡、任意分区双向请求、缓存重建、预算与 i64 极值。这是 CPU 派生数据 kernel；其整数 GPU twin 与限定百万候选基准见下节，仍未完成产品散布渲染/地表投影及完整性能验收。

### M4：整数 GPU 散布生成

`GpuScatter` 复用 residency revision 1 的配置、范围、完整缓存键及 O(1) 工作预检。WGSL 以 u32 low/high pair 精确实现全局有符号 i64 网格、PCG jitter、请求外原始候选排名和严格最小距离比较，无浮点或 shader-i64 feature。固定容量约束所有 primary slot，不按幸存数量截断；逻辑 buffer 为 `68×max(capacity,1)+208` 字节。请求变化只上传 192 字节；encode 清计数并记录生成/compact，无分配、上传、submit/wait 或回读。消费端通过 compact source slot 直接读取 64 字节整数记录及 indirect args；队列生命周期与既有 resident 相同，必须先提交再更新请求。

2026-10-15 Apple M2/Metal 全套原始源码 harness 104 项通过、0 失败/忽略（库 82、scatter 5、density 8、resident 6、parity 1、WGSL 2）；no-default 库 73 项通过；harness all-target 与正常 workspace GPU lib clippy 均通过 `-D warnings`。GPU 覆盖 i64 两端八角、u32 spacing 极值、未请求 blocker、任意分区双向集合、63/64/65/128/129 尾部、资源复用、失败保留请求，并直接消费 records/compact 做真实间接绘制像素校验。二维 dispatch/填充编号溢出以 synthetic limit 单测检查，未宣称实机跨二维边界。release 百万无距离筛选候选 encode+finish P50/P95/P99 为 130.416/196.583/394.708 µs，submit+wait 为 3572.834/4616.916/4775.042 µs；10k/半径 17mm 筛选后 2,736 点，submit+wait 为 1519.625/1720.250/2515.084 µs。计时内无上传/回读/绘制，不是 GPU timestamp、产品帧时或蓝噪声画质验收。完整契约、四档数据与验证入口见 `prism_world_gpu_scatter_zh.md`。

### M4：CPU 作者密度与种类场

`ScatterDensityField` 是不可变、可丢弃重建的派生描述，使用全局整数毫米、三维 value noise 和三通道 domain warp；跨 i64 边缘的 warp 先扩宽再采样，仅虚拟格 hash 有明确的 2^64 周期。唯一稳定 brush ID 决定 Replace/Add 的合成顺序，球形刷子按平方距离 smoothstep 与 opacity 衰减；species 在有效权重至少 32768 时替换。候选 rank 只依赖完整 candidate key 与 field seed，固定 seed 时密度选择单调，0/65535 端点精确。全结构缓存键不使用摘要充当身份，field clone O(1)、独立配方比较最坏 O(B)。

`scatter_with_density_field` 原地过滤已有源散布，不重填空洞，保持源最小距离和半开范围的分区/逆序求值不变性。构造预算包括原 brush Vec 的 spare capacity，场工作上界在枚举前检查；合计输出预算计算两个 Vec 的实际容量载荷，不是峰值分配/RSS 上限。每次 sample/rank 无分配；该线性路径逐点采样成本为 O(B+固定噪声成本)，显式 CPU 空间索引路径见下节。

no-default 完整回归 65 项单测与 1 项文档测试通过，其中作者场新增 30 项；all-target clippy `-D warnings` 通过。覆盖独立整数 oracle/golden、全部 UNORM16 权重与密度、i64 边缘 warp、完整缓存字段与三轴任意分区双向求值。Apple M2/macOS release 的 10k 点、8 预热/64 样本：constant sample+rank P50/P95/P99 为 200.459/217.209/234.625 µs，warp+32 重叠刷子为 5392.875/32179.334/55792.958 µs；完整源生成+过滤+属性为 5878.709/12008.041/17783.000 µs、恰为两个 Vec 分配、880,000 字节输出容量载荷。并行负载干扰明显，不是 GPU 时间、产品帧时或严格优化 A/B。API、精确取整与完整基准见 `prism_world_scatter_field_zh.md`；该段是 CPU 基线验收，GPU twin 的独立验收见下节，地表过滤和产品 renderer 仍未由这些测试覆盖。

### M4：CPU 作者刷子空间索引

`ScatterBrushIndex` 共享完整不可变 `ScatterDensityField`，以确定性空间 BVH、前序 escape 节点和显式可复用 scratch 裁剪笔刷。i128 包围盒端点夹到 i64 世界域；候选 brush index 在合成前恢复 canonical ID 顺序，复用基线整数 noise/warp/距离/取整/species kernel，不改 field 算法版本或完整缓存身份。

索引与 scratch 分别限制实际 Vec 容量载荷，fallible 预留，采样不分配。`scatter_with_indexed_density_field` 在枚举前以 primary×最坏单样本 work 准入，输出原地稳定压紧、两列容量共同预算；点/属性/cache/source counter 与基线相同，实际 `field_work` 另按节点/精确刷子探针计数。复杂度为 O(访问节点+K log K+K+固定噪声)，全部重叠时可能慢于线性扫描；不是增量编辑、GPU 索引或产品帧时承诺。API、预算、独立对照与 release 复现入口见 `prism_world_scatter_brush_index_zh.md`。

2026-10-16 no-default 通过 99 库+16 集成+1 文档测试，all-target strict Clippy 通过。Apple M2 release：10000稀疏刷子/256查询批次的线性与索引 P50/P95/P99 为 5158.000/11220.250/15476.208 和 88.834/199.500/216.458 µs；全重叠索引更慢，三档负载均保留线性基线。1024稀疏刷子/10000 primary的完整散布为 21017.125/22018.583/25264.375 与 3504.917/3692.458/3754.917 µs，两边4750点、两个分配、880000字节输出容量并且完整输出一致。宿主并行负载、无最小距离邻居探测，不扩张成产品帧时结论。

### M4：GPU 作者密度与种类场

`GpuScatterField` 在 `GpuScatter` 后记录精确整数 field pass，复用 CPU revision 1 的 noise/domain warp、brush ID 顺序、平方距离 smoothstep、signed Add 取整、species 阈值和独立 rank；无 float 坐标、shader-i64 或 optional feature。生产路径无 CPU 逐点生成/采样/回读。显式五 storage binding，因此新增严格 required-limits context acquisition；原 downlevel 四 binding 默认值保持不变。

source 64 字节 record 保留源 accepted；新增 source-slot 对齐16字节属性和最终 compact/indirect。下游必须消费场输出，不能只用源 accepted。复合 owned 逻辑 GPU buffer 为 `88×max(capacity,1)+64×max(max_brushes,1)+352` 字节，另一次 fallible 预留 host 刷子 scratch。源/场工作、容量、brush 数与 dispatch 均在任何 queue write 前验证，可恢复错误保留旧请求。变更驱动上传 source192字节、field uniform128字节、brush64×B字节；不变刷子不随源范围重传。encode 无 owned allocation/upload/submit/wait/readback，调用者仍须先提交当前 encoder 再更新下次请求。

2026-10-15 Apple M2/Metal 原始源码隔离 harness 完整定向回归 113 项通过、0 失败/忽略（库84、scatter5、field7、density8、resident6、parity1、WGSL2）；no-default 库73项通过，harness all-target 与正常 workspace GPU lib Clippy 均通过 `-D warnings`。新场测试逐 slot 对 CPU 全记录/属性/rank，对 compact 唯一集合与 indirect count，并在同一 encoder 内直接消费 GPU records/attributes/compact/indirect 作实际像素验证。覆盖世界角点、u32 极值、最大 brush 半径、signed rounding、尾部/空请求/重复 encode/无 wait 连续提交和失败保留完整旧请求；非完整 workspace 或多后端验收。release 基准覆盖10k/100k三维完整 voxel、constant/noise/warp/warp+32重叠刷子，每档8预热/64样本，全部576次派发逐点对CPU。100k constant 的 submit+wait P50/P95/P99 为911.333/965.750/1211.250 µs；warp+32刷子为6437.792/6715.792/7302.500 µs，owned GPU逻辑buffer为8,802,400字节，host刷子scratch为2,048字节。计时排除上传/回读/校验/绘制，不是GPU timestamp或产品帧时。完整八档分位数、资源与digest见 `prism_world_gpu_scatter_field_zh.md`。

### M4：GPU 作者刷子有序 AABB 索引

`GpuScatterField::new_indexed` 与 `GpuScatterHeightfield::new_indexed` 显式启用canonical ID连续区间中点树，不移植CPU空间中位数树；叶序保持非交换Replace/Add/species合成顺序，signed i64保守AABB与前序exclusive escape支持无栈裁剪，无每点scratch/排序。节点64B，尾随实际brush列表打包在原storage buffer内，不增加binding或optional feature，不改算法版本、输出ABI或完整缓存身份。原`new`仍是线性基线。

构造O(B)、平衡递归栈O(log B)，查询最坏O(B)；资源和host scratch预算包括完整树，单点work另加实际节点数，全部请求预检仍在queue write前完成。只在完整brush列表改变时重建上传；base/noise/source变化不重传树，不是增量refit。indexed作者场→真实heightfield→managed完整键删除组合已与公共CPU链路对照。

2026-10-16 Apple M2/Metal冻结源码独立harness定向164项全通过，0失败/忽略；no-default库另73项与仅gpu feature索引11项通过，正常workspace GPU+terrain库与harness all-target严格Clippy通过。六类release A/B含预热共432次dispatch逐slot/属性/compact/indirect完整对照；256/1024顺序稀疏的总窗口P50线性/索引为1238.959/596.833与1444.375/492.750µs，乱序稀疏和全重叠均退化，因此不默认启用。该窗口是encode+finish+submit+wait主机墙钟，非GPU timestamp或产品帧时；完整资源、分位数、ABI及边界见`prism_world_gpu_scatter_brush_index_zh.md`。

### M4：真实 heightfield 查询与 CPU 地表过滤

`prism_world_terrain::PlanarHeightfieldSurface` 直接查询真实烘焙 tile，显式平面 origin/spacing，XZ 半开归属，沿既有 mesh 三角拓扑进行 i128 精确插值并返回几何法线。默认构建无需 mesh/引擎；可选 `scatter` 仍为 `no_std + alloc`，组合源散布/作者场后按闭区间地表高度、垂直离地带及有理坡度稳定删点，保留源 key 重建、最小距离及 XYZ 分区不变。不是贴地投影；窄带可无点，不作补点密度保证。

地形工作以 primary×4 逻辑读数在枚举前准入，实际查询 field 幸存点；复用两 Vec 原地压紧、容量继续计入原输出预算。完整缓存 key 通过 Arc 共享不可变 tile 快照，比较实际样本/spec/revision/embedding 和所有过滤参数，不用 tile/revision 或摘要替代身份。默认 78 项回归、scatter feature 93 项回归均通过，0失败/忽略；两配置 normal workspace all-target Clippy `-D warnings` 通过。新增查询覆盖非共面/对角线、半开边界、超2^53负坐标、i64/u32极值和解码超i64；过滤覆盖精确闭端点、双轴坡度、属性对齐、源距离/key重建、三轴逆序分区、完整缓存失效、公开 rebake 发布新快照及失败原子性、预算/溢出/空请求/缺覆盖。

2026-10-16 `aarch64-apple-darwin`、macOS27.0.1、rustc1.96.1 release，8预热/64样本，主机墙钟 nearest-rank：100k真实查询含逐结果校验 P50/P95/P99=1705.166/2613.750/3803.084 µs、0分配；10k源+noise/warp作者场+地形过滤为3514.916/3930.916/5631.916 µs，6132个场幸存点筛到1250点，24528次逻辑地形读取，两列容量880000字节、恰2次分配，共享heightfield8450字节另计。每次完整结果在计时外逐点/属性对oracle，计时排除构造/校验/销毁；首次fixture未对齐格被预算拒绝后修正并加入候选数断言。不是GPU时间、峰值内存、无干扰A/B或产品帧时。完整方法与边界见 `prism_world_heightfield_scatter_zh.md`。

### M4：CPU 刷子索引与真实 heightfield 组合

`scatter_with_indexed_heightfield` 显式接收已构建CPU空间BVH及有界可复用scratch，先执行地形完整覆盖/过滤参数/primary×4读取预算预检，再调用索引密度源，最后与线性入口共用精确地表过滤和原地两列压紧。完整点/属性/cache/source candidate counter/地形读数与线性相同，仅实际field_work不同；算法版本不变，支持直接组合完整键删除。新增Index错误保留原始索引错误链，不隐式创建/增长scratch或引入逐点临时输出。

正常workspace scatter配置109项库测试+新增11项公开集成测试通过，no-default配置78项通过，0失败/忽略；两配置all-target严格Clippy与本轮文件格式/diff检查通过。覆盖线性全输出对照、noise/warp/非交换ID顺序、稀疏与全重叠work、极端整数、三轴分区/距离、严格预算/溢出/错误顺序与错误链、完整缓存/删除组合、scratch跨索引/失败复用；独立源码审查未发现可复现缺陷。索引依赖residency另通过99项库测试+16项公开集成测试+1项文档测试，不代表全workspace验证。

2026-10-16 Apple M2 release，1024刷子/10000 primary、真实65² ramp，每类8预热对+32测量对交替AB/BA，全部160次计时调用在计时外全量核对输出/cache/counters与容量。稀疏线性/索引P50/P95/P99为23322.709/24135.250/24262.958与1049.458/1170.041/1262.417µs；全重叠为84574.709/89875.833/93354.000与131862.875/136042.167/138091.750µs，因此索引仍为显式可选。两Vec恰2分配、880000字节容量，索引139200字节/scratch8192字节/地形8450字节另计。无noise/warp、最小距离0，主机墙钟含源/场/地形/输出分配，不是GPU timestamp、RSS、产品帧时或完整系统验收；复现与预算见`prism_world_heightfield_scatter_zh.md`。

### M4：真实 heightfield 的 CPU 二维贴地放置

`scatter_on_heightfield` 是独立于三维稳定过滤的新入口：canonical grid Y=0的全局XZ jitter提供二维候选，无随机Y slab；水平Matérn-II筛选包含请求/表面外以及随后被terrain/field拒绝的原候选。真实三角面高度与整数offset相加后整体取整，半毫米远离零，投影溢出整体报错；地表高度/坡度仍按原精确样本过滤，作者场在最终取整偏移位置采样。`HeightfieldPlacementKey`显式包含placement算法版本和完整source key，`rebuild(surface,offset)`重建投影几何，不修改旧ScatterPoint身份或既有API。

任意非对齐XZ半开分区和请求顺序不改变成员，水平最小距离保证投影后至少同样的三维欧氏间距，但不保证测地线均匀、最大填充、Poisson或蓝噪声。完整缓存含四种算法版本及全部配置/region/filter/terrain/field；候选/四角/作者work上界在分配前检查，两列各一次fallible reserve、实际capacity计费，只对最终接受行检查count上限。CPU `no_std + alloc`，无engine/mesh/GPU依赖或逐点分配。

2026-10-16 Apple M2/macOS27.0.1/rustc1.96.1 release，真实65² ramp、10000候选、三枚非交换作者刷子，半径0/150mm各8预热+32样本；全部80次计时调用均完整逐点对独立整数参考算法、重建key、检查属性/cache/counters与全部输出对水平距离。两档P50/P95/P99为1036.750/1085.833/1094.375与6245.959/8721.583/10955.792µs，最终4571/683点，实际candidate探针10000/122011；两Vec capacity共1840000字节、恰2分配，场/高度共享载荷另计。保留尾延迟，不推导无干扰跨档加速比或产品帧时。方法、资源、语义及复现见`prism_world_heightfield_placement_zh.md`；该源生成片不包括GPU/行星放置、曲率/数据层/跨层规则或产品renderer接线，独立placement持久删除由下述M4/M5后续切片交付。

本片最终正常workspace回归：`scatter`为109库+26新放置集成+11既有索引集成，no-default为78库，均0失败/忽略；两配置all-target Clippy `-D warnings`及定向rustfmt通过。另对既有GPU地形过滤在Apple M2/Metal以原生产crate/原测试文件的最小验证包连续两轮通过8项，含真实indirect draw像素；正常workspace尝试被并行修改的`prism_tasks::arena`缺少`Layout`导入阻断，隔离验证使用自身锁依赖。没有修改资产/GPU源码，不将该结果当作新放置GPU twin或完整workspace通过；验证细节见上述放置文档。

### M4：真实 heightfield 的 GPU 二维贴地放置

`GpuHeightfieldPlacement` 是 CPU `scatter_on_heightfield` revision 1 的固定容量 GPU twin，
独立执行二维 XZ source、水平 Matérn-II、真实三角投影、未偏移高度/坡度过滤、最终 XYZ
作者场和 finalize。source/projection/field/finalize 通过有序 dispatch 连接；source record、
sample、attribute、compact 和 indirect 均保持 source-slot 对齐，生产 encode 不做 owned
分配、等待或读回。WGSL 只用 core u32 limb；Naga ABI 闸门验证 192B/128B uniform、64B
record/sample、binding access、入口和无 float/wide scalar。

投影高度加 offset 后整体半值远离零取整，投影溢出在过滤前检测；任何溢出都会由 finalize
清空 indirect count 和所有 live acceptance，禁止部分成功。完整 source key、terrain/field
快照、filters、embedding 和配置参与缓存依赖；构造和 request preflight 先验证完整 brush/
surface、设备 binding、work budget 与固定 capacity，逻辑 buffer 为
`148C+64B+4S+468`（C/B/S 至少为1）。

2026-10-16 Apple M2/Metal 严格回归 7/7 通过，覆盖三角 oracle、signed normal、坡度/闭区间、
半毫米跨零、最终 Y field、刷子/noise/warp、非对齐分区、i64 边缘、缓存/事务、整体溢出和
真实 indirect draw 像素；无设备 Naga 闸门 6/6 通过。release 8 warmup+64 samples 的
10k/100k×0/150mm 结果、完整资源和复现命令见 `prism_world_gpu_heightfield_placement_zh.md`。
这只完成二维平面 GPU kernel，不代表 GPU placement deletion、行星/曲率/数据层、风/交互、
产品 scene/ECS decoder、生产 renderer 注册/提交或多厂商性能已完成。

本轮另交付 backend-neutral renderer bridge 与原生注册快照。`GpuHeightfieldPlacement::encode`
已用 `reset_main` compute 取代 `clear_buffer`，只清零 `indirect[1]` 与 overflow status，保留另外
三个静态 indirect 字。`registration_snapshot` 暴露 11 个原生 buffer handle、精确逻辑 byte size、
reset/候选四阶段 pipeline 与 bind group、dispatch、request generation 和 candidate count；空请求以
`active=None` 表达 reset-only，不伪造候选 dispatch。

`build_placement_render_graph` 接收 renderer 已注册的 RHI buffer/scene texture 与
pipeline/bind-group identity，按稳定 cell 顺序记录：active placement stream 为
`reset → source → project → field/compact → finalize` 五个 compute pass 加一个 indirect raster；
empty stream 只有一个 reset compute pass。reset hazard 只写 indirect/status；全部后续
bind-group 可见资源进入 RenderGraph hazard/SSA 声明。候选数量/work presence 不一致、同 cell 重复、
零 buffer、零 dispatch 和未对齐 indirect offset 在 graph 发布前拒绝。Null RHI 回归验证两种命令
形态与跨 cell 顺序。

2026-10-20 独立离线 target 验证为 terrain-scatter 库 135/135、WGSL 9/9、GPU feature 库
119/119、world runtime 库 104/104；沙盒外 Apple M2/Metal IntegratedGpu 的真实
`heightfield_placement` 集成 10/10 通过。该 snapshot/bridge 不分配或持久注册生产 RHI 资源，
也不 submit/wait/retire；persistent registration RAII lease、renderer draw pipeline/bind group、
scene target、实际 RenderGraph native record/encode/submit、completion tracker/generation
retirement、`prism_render_scene` mesh/material/unified visibility 消费以及产品 scene/ECS decoder
仍是明确未交付边界。

### M4：真实 heightfield 的 GPU 精确地表过滤

`prism_world_gpu/terrain_scatter` 的 `GpuScatterHeightfield` 组合整数源生成、作者场与地形过滤，直接读取真实 baked 高度样本，不在 CPU 逐点求值或回读生产输出。六 storage binding、128B 地形 uniform、i32 样本；有符号 i64 世界坐标与更宽有理高度/离地带、精确双轴坡度由核心 u32 limb 完成，不需要 shader-i64 或其他 optional feature。与 CPU revision1 三角/闭端点/坡度逐项一致，只删点、不移动/补点，保持源最小距离、key 重建和 XYZ 分区成员不变。

全部资源在构造前验证，owned GPU 逻辑 buffer 为 `108C+64B+4S+496` 字节（C/B/S 为至少1的源/刷子/样本容量）。完整请求/读数/源/场预检在任何 queue write 前进行，可恢复错误保留旧请求、结构 key、固定资源与 GPU 输出。地形 uniform 变化上传128B；真实样本变化上传4N字节，只有 embedding、过滤或 metadata 变化不会重传相同样本。encode 无 owned 分配/上传/submit/wait/map/readback；必须先提交当前 encoder 再改请求。原64B源记录不改写，消费者必须使用最终16B属性、compact及indirect。

2026-10-16 Apple M2/Metal 正常 workspace package 定向回归126项通过、0失败/忽略（库87、density8、parity1、resident6、field7、scatter5、heightfield8、terrain draw1、WGSL3），no-default库另73项通过。八项新增实机回归涵盖全部slot CPU oracle、compact唯一性、真实draw_indirect像素、有理/极值/坡度溢出、缓存与rebake、任意XYZ分区、尾部/空请求/连续无wait提交和失败保留旧输出。隔离原始源码 harness all-target Clippy `-D warnings`通过；正常 workspace all-target Clippy 在共享依赖 `prism_asset` 的164项现有lint/doc错误处失败，不视作通过，也不修改其他工作。

release两轮每档8预热/64测量，共288次dispatch全部逐点对CPU。10k/100k三维候选分别从6132/61122场幸存点筛为1250/6193点；逻辑GPU buffer为1097460/10817460B，host高度scratch16900B，共享原高度8450B，首次显式上传含indirect初始化17396B。回归后复测的submit+wait P50/P95/P99分别1442.333/2362.833/4674.250µs和7522.584/15448.750/25020.584µs；首轮同样保留于 `prism_world_gpu_heightfield_scatter_zh.md`，不挑选较好尾延迟。计时不含上传/回读/CPU检查/绘制，不是GPU timestamp、物理VRAM或产品帧时；贴地放置和完整产品散布仍另行验收。

### M4/M5：可再生散布的完整键持久删除

`ScatterDeletionSet` 以不可变完整结构键集合提供稳定原地过滤；`ScatterDeletionBinding` 将整组键存于调用方选择的作者节点 PersistentId/FieldId 的 override Bytes，而不是散布 dense slot、截断 hash 或第二份内容数据库。格式 `PSDL` revision1 为16B头+每键52B，严格数值排序/唯一、完整键重建及独立 wire/count/Rust容量准入；错误不降级为空。单键实际修改O(D)拷贝、过滤O(N log D)，原points/attributes指针容量与源key/距离语义不变，不补点。runtime输出保留node/field/完整集合额外cache依赖；整字段提交沿用Timeline真实inverse及字段LWW，明确不是逐键并发union。

真实FileJournal关闭重开→回放→丢弃并重新生成缓冲→过滤，以及持久undo/redo回归已通过。并行流送修改前默认runtime 78、no-default 30、proxy_host 98测试通过，residency 91单测+1doctest通过；最后三项回归加入前全feature曾通过115项。默认/no-default all-target严格lint及完整示例随后通过（51点、删除/undo/redo为1/0/1、3条事务回放）。并行流送修改曾因未匹配新增`StreamAction::Reload`阻断重跑，该阻塞现已修复；2026-10-16独立target、offline/locked定向lib重跑默认79、全feature119项，均0失败/忽略，不扩大为当前脏工作区全配置或全workspace通过。完整可执行示例以clone staging、落盘成功后才发布内存Timeline；保留进程内撤销栈、不声称重建重启后的可编辑历史。这是明确的失败发布边界，不是产品增量事务或跨资源崩溃原子指令。

2026-10-16 macOS27.0.1/aarch64/rustc1.96.1 release，每项8预热/64样本：10k/100k输入、25%删除，原地过滤主机P50/P95/P99分别791.917/797.167/922.708µs及9758.500/9875.125/11031.166µs；独立encode+decode为106.000/109.209/113.917µs及1083.584/1092.292/1252.833µs。完整结果/指针/容量在计时外逐样本校验，不含生成/clone/析构，不是GPU时间、RSS或产品帧时。wire/实际容量、完整契约及历史/合并/发布成本见`prism_world_scatter_deletions_zh.md`；GPU及地形删除适配见下节，产品接线仍另行验收。

### M4/M5：二维贴地放置的完整键持久删除

`HeightfieldPlacementDeletionSet`只接收独立placement版本与canonical grid Y=0的完整源键，验证不依赖当前地形覆盖或请求成员；不混用原三维删除。不可变Arc快照、构造去重前count/实际capacity预算、无变化O(1) clone和实际修改O(D)拷贝均有回归。`filter_heightfield_placement`先检查列对齐，再稳定原地压紧真实投影点及作者属性，保留两Vec指针/容量、源缓存和工作计数，不重投影、不补点。

runtime显式`placement` feature提供`PlacementDeletionBinding`、完整namespace/set/filter-version缓存依赖及PPLD revision1（16B头+56B完整键）；默认依赖不扩大，启用placement仍支持no_std+alloc。严格规范顺序、完整身份、版本、长度及独立wire/decoded容量验证，不接受PSDL或静默清空。scene字段仍为唯一权威，沿用Timeline真实inverse/字段LWW，不是逐键union；日志与Timeline不自动原子。完整示例以clone staging→真实FileJournal同步→发布，再关闭重开、回放、丢弃并重新生成400点/属性，删除/undo/redo为1/0/1、3条事务全部核对。

2026-10-16正常workspace独立target回归：terrain scatter库132+二维放置集成26+索引地形集成11、no-default库78项；runtime默认79、placement104、no-default+placement55项全部通过，含新增23+25项删除/持久恢复测试。四种定向配置all-target Clippy `-D warnings`通过。Apple M2/macOS27.0.1/rustc1.96.1 release，10k/100k×0/25/100%删除，每操作8预热/64测量，两轮共1728次计时调用及全部解码后过滤对照通过。最终100k/25%过滤P50/P95/P99=10103.084/13572.792/19537.500µs，独立codec为1233.209/2192.375/2230.084µs；100k全删过滤P99=115402.916µs明确保留。空删除先验证列对齐后O(1)返回；两列capacity1840000/18400000B，全删不释放。主机计时排除生成/clone/外部oracle/销毁，不是GPU时间、峰值内存或帧时承诺；两轮全部档位、资源/历史成本及边界见`prism_world_placement_deletions_zh.md`。本片不改变GPU代码，不将CPU持久恢复冒充GPU二维放置、产品renderer接线或整个系统验收。

### M4/M5：CPU 地形删除与 GPU 完整键删除

`filter_heightfield_scatter` 在真实地形过滤后稳定原地压紧 points/attributes；即使空删除或空列也先检查长度对齐，保留原指针/容量、完整地形/作者场/源缓存键与上游工作计数。删除集合（含请求外键）是额外完整缓存依赖；不重烘、不改变位置或补点。normal workspace terrain no-default 78项、scatter 109项（新增16项）全部通过，两配置 all-target Clippy `-D warnings`通过。

`GpuScatterDeletions<U>` 独占 sealed 源/作者场/地形上游，shader只读取最终compact/count并按完整13-word键二分删除，不复活中间被剔除的点。保持 records/attributes 未改，消费者必须使用最终 compact/indirect。新增逻辑buffer为`52*max(Dcap,1)+4*max(C,1)+144`字节；预算覆盖整链、实际key scratch容量和`N*ceil(log2(D+1))`比较界，D=0仍预算遍历。所有层先检后写，错误保留旧请求/输出；encode无owned分配/上传/submit/wait/readback。context clone共享私有Arc设备身份，拒绝跨Instance局部handle相等误认。调用者仍须先submit当前encoder再write下一请求，raw API不自动检测提交epoch。

2026-10-16 Apple M2/Metal、无optional feature，冻结后原始源码隔离harness共139项全部通过（库100、源删除7、组合删除8、既有源5/场7/地形8、WGSL4），含真实draw_indirect像素、完整记录/属性不改写、CPU全键oracle、独立设备拒绝/clone接纳、预算失败原子性和无wait顺序提交。harness all-target与正常workspace terrain_scatter GPU lib依赖闭包Clippy均通过`-D warnings`，正常workspace no-default/terrain_scatter lib check均通过；不是全workspace验收。

release两组完整记录共864次dispatch逐全键/位置对CPU，10k/100k候选×0/25/100%删除、8预热/64测量。冻结后100k/25%删除encode+finish P50/P95/P99=57.500/156.375/330.167µs，含submit+wait为2004.875/2523.542/7724.459µs；整链逻辑GPU8500352B、新增1700144B、key scratch1300000B。首轮和其他档位完整保留于`prism_world_gpu_scatter_deletions_zh.md`，不挑快轮、不作无干扰A/B；计时排除构造/上传/回读/校验，非GPU timestamp、物理VRAM/RSS或产品帧时。没有因此完成风/交互、逐键并发协议或产品renderer接线。

### M5：持久化

- `prism_world_persist`：`PersistentId`、规范化 `OverrideDelta`、append-only log、确定性 LWW CRDT、snapshot/timeline、undo/redo、little-endian codec/checksum。
- `std` feature 提供 `FileJournal` 和 `SnapshotStore`：journal v2 使用事务记录 + checksummed commit marker 表达批次，限制每批最多 65,536 笔事务和 64 MiB 事务记录；恢复仅发布完整 committed batch，截断或校验失败回退到当前批次起点。v1 journal 明确拒绝，需旧运行时导出有效事务并写入新 v2 文件后迁移；新版本不静默混读/追加。
- `FileJournal::open` 在读取、初始化或恢复之前取得独占 OS 文件锁，并保持到句柄销毁；第二个句柄明确返回 `Locked`，不支持绕过锁写入或替换锁定路径。写入、flush 或 sync 的 I/O 失败会 poison 当前句柄，即使尾部回滚成功也必须销毁并重开；回滚失败同时保留原始错误和回滚错误。错误返回并不证明未提交：若 commit 已完整落盘，重开可见整批；checksum 用于意外损坏检测，不是认证机制，掉电语义仍依赖文件系统和设备。故障注入走同一生产 writer，覆盖 seek/短写/flush/sync 和回滚失败；batch 边界逐字节截断测试保证恢复得到整批或无批。
- 本轮 macOS 验证：`cargo test -j 1 -p prism_world_persist --features std --quiet` 的 18 项测试、真实子进程锁回归 2 项，以及 `cargo check -j 1 -p prism_world_persist --no-default-features` 通过。锁内读测试复用持锁文件句柄并恢复游标；跨进程回归证明父进程持锁时子进程得到 `Locked`，释放后可重开并读取已提交前缀，且文件字节不变。尚未提供 Windows 实机验证。
- 相关提交：`4d0fbfebe`；Scene/ECS 接缝提交：`34656b305`；GPU workspace 集成：`c94d5cf13`。

### M3：异步 IO/provider

- 共享动态优先级：`spawn_future_with_priority` 的初次提交/wake/量子 continuation 与 world 路由/scheduler/native chunk 保留同一个 `Arc<PriorityCell>`，`PrioritizedCellIo` 可直接包裹任务 scope 与 coordinator。IO 按 live lane、静态分数、FIFO 准入；动态队列有界扫描，静态队列保留 heap，活动读取不抢占。原生逐块采样 kernel hint，物理槽等待和 NamedThreads 派发仍 FIFO；任务池显式 priority inbox 始终先于普通队列，并非 IO 的隐式 Normal 排序。生命周期、性能和验证细节见 `prism_world_shared_priority_zh.md`。
- `prism_world_runtime/tasks` 提供 `TaskCellIo` child scope 和 `next_cell_completion`，真实 archive 加载在 `TaskPool` 的 0/1/2 worker 上完成 checksum/header/host 链路。`CancelToken::cancelled()` 可唤醒停驻任务，逐 poll 有界取消，迟到 payload 受抑制，drop 逐个取消未完成请求。零 worker fallback 已修复外部 IO 唤醒只入队不通知 driver 的停顿；runtime tasks 63 项、tasks std 273 项和 11 项文档测试通过。async harness 每 dispatch 最多 32 poll、continuation 进入全局 FIFO；零 worker root 每 32 dispatch 再获 poll，仍为合作式且不保证跨队列严格公平。详细生命周期、panic 清理与 feature 验证边界见 `prism_world_async_runtime_zh.md`。
- `prism_world_runtime/reactor` 提供 `ReactorCellIo`，借用宿主既有 `IoReactor`/`NamedThreads`，IO worker 独占 fd/native future 并在 reap 后关闭；compute 只 poll。取消终态不提前退还原生清理配额，有界 FIFO 同时保留槽与声明字节后锁外唤醒，不自旋重试。真实 native archive 在 0/1/2 compute worker 完成校验和 host 链路，IO 63/runtime 67 测试通过；Apple M2/macOS 的 2 MiB cached archive release P50/P95/P99 为 4.413/4.511/4.525 ms。IO worker 会 park，非全部线程无阻塞；当前 backend 仅 Apple。所有权、2×capacity 准入界、复现与限定测量见 `prism_world_native_io_zh.md`。
- `prism_world_io`：分块读取请求、稳定优先级/FIFO 准入、活动读取 round-robin、单调请求 ID 的 fail-closed 耗尽保护、显式容量背压、可取消 poll 操作、连续 chunk/short-read/checksum 校验和 std 文件 provider。
- 默认构建保持 `no_std + alloc`；`std` 只增加 scheduler 与文件系统 provider，不把 IO、资产数据库或 ECS 引入 world policy core。
- `prism_world_runtime`：真实调用 `StreamScheduler` 生成 load/reload/evict 动作，并通过有界、可取消、优先级排序的 provider 队列提交 cell；完成后执行固定 manifest header 校验，再进入 `SceneHost::load_cell`，失败/驱逐/reload 失败写入可 replay trace。std 构建新增 `IoSchedulerAdapter`，实际桥接 `prism_world_io::IoScheduler` 的 chunk/completion/cancel/backpressure/error 事件，并保留同步 `ChunkProvider` 兼容路径。
- `prism_world_runtime/asset_container` 现提供真实 `AssetContainerCellProvider`：使用已校验的 `RuntimeManifest`/`OverlaySet` 通过 `ManifestReader` 路由 `scene_key`，底层复用 `ContainerReader`/`RangeSource` 的 raw、deflate 与 range-backed 读取；在分配前检查声明长度/上限，拒绝非零物理 offset、路由/身份/范围错误、短读、header/checksum 不匹配和取消，不把部分字节交给 `SceneHost`。可选 `asset_container_mmap` feature 将 v2 mmap reader 接入同一 overlay/manifest/provider 路径，并覆盖真实 cell header 校验与完整 payload 返回。这仍是同步兼容 provider，不宣称可中断正在进行的容器读取、cell 零拷贝、GPU 直传或场景 schema 解析。
- `prism_asset_container` 的可选 `mmap` feature 提供只读 `MmapRangeSource`、`MmapV2DirectoryView`，且 v2 `ContainerReader::from_mmap_file` 直接保留该映射索引：不复制 directory/table，不保留 reader 自身逐 entry/逐 chunk 的 `BTreeMap`，entry 查询在规范 GUID 目录上二分，chunk 查询直接读取固定 table record。挂载仍用有界连续 scratch 完整校验 header/reserved、目录与 table hash、物理区间/零 padding、每个 chunk 与 entry payload hash；压缩 chunk 必须解码校验，不跳过整容器扫描。scratch 容量纳入共享 in-flight budget，分配使用 fallible reserve；映射文件必须在 reader 生命周期内保持不可变。v1 和其他受支持布局仍走标准 range-backed 索引。该 reader 可沿既有 `OverlaySet → ManifestReader → AssetContainerCellProvider` 路由加载 World cell；overlay/runtime manifest 仍保留跨容器优先级所需 GUID 索引，因此端到端内存收益未量化。provider 仍组装完整 `Vec<u8>`，尚未实现应用 schema decoder 的 mmap 视图、cell 零拷贝或 GPU 直传，也无产品吞吐达标证据。
- 2026-10-21 回归：`cargo test -p prism_asset_container --features mmap --offline` 的 90 项 lib、11 项 block-format、10 项 read-budget 测试通过；v2 `ContainerReader` 与普通 reader 的目录/entry/chunk/read_all 对拍覆盖 Raw、Deflate、BlockDeflate，另有 v1 mmap fallback、损坏 payload 和 scratch 预算失败释放。`cargo test -p prism_world_runtime --features asset_container_mmap --lib --offline` 的 117 项 runtime 测试通过，含 mmap v2 container → manifest/overlay → World cell provider → cell header → 完整 cell payload；不启用 mmap 的 runtime `asset_container` feature 116 项也通过。`cargo test -p prism_asset_container --lib --offline` 的 85 项无 feature 测试、`cargo check -p prism_asset_container --no-default-features --offline` 以及两 crate 对应 mmap feature lib Clippy `-D warnings` 通过。未做 release mount/query 基准，结构性分配减少尚无量化收益声明。
- 2026-10-21 GPU resident mmap backing：`GpuBlockDecodeBackend::bind_validated_mmap_chunk` 复用已完整校验的 v2 映射，在 `GpuResidentBlockPayload` 生命周期内保留 `Arc<MmapRangeSource>`，避免再复制整容器到 `Arc<[u8]>`；映射仍只读，挂载校验仍是完整性权威。`cargo check -p prism_asset_container --features 'gpu mmap' --offline` 与非强制设备的 `cargo test -p prism_asset_container --features 'gpu mmap' --test resident_blocks --offline` 通过（8 项）；`PRISM_REQUIRE_GPU=1` 的同一 mmap 消费测试已实际启动，但当前环境报告没有可用 graphics adapter，因此不能计为实机 GPU 通过。该切片只减少 GPU resident 输入的整容器主机复制，不等于 DirectStorage、磁盘到显存直传、cell schema 零拷贝或 World provider 产品链路闭环。
- 2026-10-21 通用 cell decode seam：`prism_world_runtime::DecodedSceneHost` 由调用方注入 `CellPayloadDecoder` 与 typed `DecodedCellHost`，先做独立 decoded-byte/cell-count preflight，再 decode、prepare，最后由无失败的 commit 点发布；同 sector reload、decode/预算/prepare/evict 失败均保留旧 typed resident 与计费。`cargo test -p prism_world_runtime --lib --offline` 独立 target 为 108 项通过。该接缝仍接收 runtime 已组装的 `&[u8]`，不规定应用 schema，不等于 `prism_scene`/ECS/renderer 产品 materializer 已接入，也不提供 mmap payload 零拷贝。
- 2026-10-21 World SDF physics consumer seam：`WorldSdfSnapshot` 固定 origin/voxel size/distance scale/revision，`WorldSdfPhysicsBridge` 与 `SdfPhysicsCellHost` 在每 cell 与 aggregate brick/sample/byte 预算内先 staging，再调用 `SceneHost`，成功后才发布；reload、decode、stale revision、预算和 host 失败保留旧 snapshot。physics query 对未驻留 cell/缺失 brick 保留 `Unknown`，详见 `prism_world_physics_sdf_runtime_zh.md`。该切片尚不等于刚体 solver 接触、renderer graph 或 GI 多消费者接线。
- `FileCellProvider` 复用 `prism_world_io/std` 的真实 `FileProvider`，以 sector 注册完整 manifest/文件 offset/checksum，拒绝过期 manifest 并保留调度优先级。真实临时文件测试覆盖文件→分块 scheduler→header/checksum→`SceneHost`，及短读、打开失败、取消和错误隔离；底层每次 poll 为有界阻塞读取，不宣称已变成线程池异步文件 IO。
- 本批新增独立的 `ThreadedFileProvider` 源码：固定 std worker、预留槽位/队列、受限 chunk、ACK 驱动的后续读取、worker 内文件打开/读取/关闭，以及非 joining 的取消/shutdown。取消的 Ready 缓冲区交由 worker 清理；槽位在结果和文件销毁后才释放。waker 克隆/替换后的销毁及唤醒位于 metadata 锁外，worker/批量 shutdown 隔离唤醒回调 panic。逻辑 payload 约束为 `capacity × max_chunk_size`，不包含调用方已取走的 chunk、线程栈、元数据、分配器开销或 RSS；阻塞 OS 调用只能合作取消，返回前仍占槽位。16 项回归覆盖范围/短读/打开失败、准入、parked ACK、queued/running/ready 取消、重复 shutdown、last-owner drop、唤醒时序及 panic，已与 crate 其余测试合计 28 项通过。
- 新增 `CellIo`/`CellFrameCoordinator` 的 task-context poll 接口；`ScheduledCellIo` 聚合完整 cell，每次最多消费 16 个 chunk，随后唤醒并让出。`WorldFrameCoordinator<P>` 是同步兼容别名，保留 `new/poll_io`；旧路径仍可能阻塞，不把兼容 API 称为异步。同步 runtime 的 ID 也改为单调非零、耗尽显式失败，并用 ID 保持同优先级 FIFO。
- `RoutedCellIo` 在提交前检查完整 manifest 版本，保留注册的 offset/checksum 与调用方优先级；`ThreadedCellIo::threaded` 将其接入独占固定 worker pool，要求 active limit 不超过物理槽位。真实 archive 回归从 stream demand 到 worker 分块、checksum/header 校验和 `SceneHost` 提交，错误内容不会进入 host；不是 scene/ECS 或 GPU 端到端验收。
- `QueuedThreadedFileProvider` 提供取消清理后的有界容量等待：所有 clone 共享 `capacity` 条准入记录与 `capacity` 个物理槽位，FIFO 授予槽位后再唤醒；等待中不保留 payload/file、不使用自唤醒重试。直接 `ThreadedFileProvider::start` 仍保留立即背压契约。IO crate 42 项测试通过，包含阻塞清理后的替换请求、2 MiB 文件校验、多 scheduler、FIFO grant、取消/drop、shutdown 及重入/panic waker。
- `IoSchedulerAdapter` 以带 ID/原始请求的 `Failed` 终态关联错误；scheduler 串行锁内跟踪实际 start/poll 回调，scheduler 的显式取消 ID 优先，provider 不能伪造其他请求的取消 ID。并发 blocking reader 保留各自结果、共享 provider 唤醒可唤醒全部等待者，public poll 不抢占其结果；终态被消费前仍计入 capacity，取消会唤醒停驻 reader。
- adapter public poll 的 waker 由最新 Context 替换，blocking reader 按请求 ID 各有一个注册；锁外执行 clone/drop/wake，锁内 provider 的同步通知延迟到 adapter guard 解锁后派发，隔离回调 panic 并保留重入注册与并发通知。本轮 IO 42 项、runtime 53 项及 no-default runtime 6 项测试通过；两种 feature 的 all-target clippy 通过 `-D warnings`。真实 archive release 测量与限定条件见 `prism_world_async_runtime_zh.md`。
- `IoByteLimits` 为 adapter/file provider 增加独立的单请求和累计声明字节准入，包含 52 字节 header（其中 8 字节为 canonical Data Layer mask）；mutex 内在分配请求 ID/启动 IO 前检查上限及累计溢出，拒绝不消耗 quota/ID。queued、active 和未消费的 Completed/Failed 均保留完整配额，消费 Chunk 不释放；兼容构造器保留无配置字节上限行为，但仍拒绝算术溢出。
- 聚合首次 chunk 时 fallible reserve 完整声明大小，后续 append 也为 fallible；分配失败取消底层操作，待 scheduler 移除后才以原 request ID/分配错误报告终态。累计上限 B 限制逻辑准入，不是 RSS 硬上限：保留 Chunk 副本与聚合结果的逻辑长度可接近 2B，Vec 实际容量/分配器、provider、事件元数据及交还调用方后的缓冲区不在计数内。
- 预算补丁在并行任务验证快照中通过 runtime 全 feature 测试 27 项、no-default 测试 3 项及两种 feature 的 all-target clippy/check/fmt；预算批新增 8 项预算/文件验证，既有 5 项并发/关联测试先前额外重复 100 轮通过。历史收尾重跑曾因并行修改使用 `error.to_string()` 而触发泛型错误类型缺少 Display 的 E0599；后续已恢复 Debug 格式化，并在本批 threaded provider 修改之前通过 runtime 全 feature 27 项测试。该记录不代表本批新 provider 已被编译，也不将当前共享工作区误报为全部通过。上述 runtime 验证不依赖 `prism_asset`。

### M6-M8：地形、仿真与网络边界

2026-10-23 状态更正与复验：当前 planar `HeightfieldMesh::apply_lod_seam_plan` 已支持四边任意 coarse-edge 子集及相邻边 corner fan；紧随其后的“相邻 coarse-edge corner fan 不支持”旧记录已过期，以本条及 `prism_world_terrain_lod_seam_zh.md` 为准。Apple M2/Metal `cargo test -p prism_world_gpu --features gpu --test terrain_draw --offline -- --nocapture --test-threads=1` 为 2 passed：新增 81 顶点/360 索引 corner-fan indexed raster，完整 64² RGBA8 读回均验证覆盖、法线和 alpha。当前 checkout `cargo test -p prism_world_runtime --lib --offline` 为 187 passed，`cargo test -p prism_world_gpu --features gpu --lib --offline` 为 139 passed；`LayeredCellManifest::admitted_layers` 先前报告的 feature 编译错误在当前 checkout 未复现。此处验证不等于 terrain renderer graph/Scene-ECS 产品接线或完整 World 验收。

- `prism_world_terrain`：cube-sphere 六面映射、四叉树 tile/跨面邻接、连续 screen-space LOD/morph、i64/Q32.32 局部坐标误差界、确定性高度采样和小端体素布局。
- M6 纵切已补齐：`TerrainStreamAdapter` 将 `TileKey` 映射为稳定 `SectorId`，计算保守行星毫米 AABB，生成稳定 `CellManifest`/derived hash，并可直接交给 `StreamScheduler`；12 项 terrain 测试、fmt/check/clippy 通过。
- M6 可执行 heightfield kernel：全量烘焙为带正毫米比例的 signed i16 高度，超范围明确失败；`BakeLimits` 在采样和分配前限制数量/载荷字节，分配失败可返回错误。局部脏区按网格 floor/ceil 保守覆盖，先完成 scratch 采样再提交；失败不改变高度或 revision。`GridChange` 是单个稳定 cache owner 的有序 token，不是跨缓存的全局身份。
- `HeightfieldMesh` 输出本地 XZ 米制索引网格，+Y 三角绕序和有限差分单位法线；更新覆盖脏高度及一圈法线邻域、不重分配原 mesh。载荷为 `32×顶点数 + 24×quad数` 字节。`mesh` feature 输出真实 `prism_mesh::Mesh`，不是渲染占位数据。
- 平面 `HeightfieldMesh::apply_lod_seam_plan` 已将 resident-derived 2:1 coarse-neighbor plan 消费为过渡索引：移除对应边界 quad、复用既有顶点并一次性替换完整索引流；计划最多两条相对 coarse 边，预算或拓扑失败保持原 mesh。`clear_lod_seam_plan` 可在独立预算内原子恢复基础规则网格，再应用新计划而不重烘高度/属性。新增 `apply_lod_morph(field, plan, alpha)` 消费 transition morph records，按 coarse segment 两端高度的中点连续移动 odd fine-edge samples；只写边界高度并更新一圈有限差分法线，无 mesh 容量增长，计划身份/边集合/field cache revision 和有限 `[0,1]` alpha 在写入前检查。`alpha=0/1` 与恢复 fine surface、coarse midpoint 的回归通过。该 CPU 平面更新不是 GPU vertex morph，也未验证动画时间连续性、邻瓦共享顶点法线或实际渲染画质；planetary position morph、finer-neighbor stitching、renderer graph 接线仍未完成。
 - `HeightfieldMesh::clear_lod_seam_plan` 现允许 seam mesh 以独立字节预算原子恢复规则网格，之后切换 residency-derived seam plan；dirty heightfield 更新保持当前拓扑，覆盖 height/normal 后与相同 seam plan 的全量重建逐数组一致。验证包含清除预算失败不变性、异边计划切换和更新后 position/normal/index 等价。相邻 coarse-edge corner fan 仍不支持，且相应拓扑边界仍须由 caller 保持拒绝。
- `PlanetHeightfieldMesh` 将已烘焙高度投影为六面 cube-sphere 索引网格；TileKey 整数身份经 f64 UV/归一化投影，减 i64 毫米 origin 后编码为局部 f32 米制位置。半径限 `1..=2^52` 毫米，非正半径+高度、非有限值、载荷超限和超过位置误差预算均失败；误差界包含浮点运算及编码，不包含高度量化、三角插值和 LOD 接缝。测试覆盖六面向外绕序/单位法线、同级相邻 tile 与跨面边界位置、地球半径第 30 层 tile 小于 0.001 mm 的数值误差界及原生 Mesh。`validate_planet_heightfield_edge` 按精确跨面方向和有界完整边比较，在发布相邻 tile 前拒绝 tile/分辨率/毫米高度断裂；`PlanetHeightfieldNormalNeighborhood` 使用邻 tile 内侧一行计算跨 tile 边界法线。新增 `PlanetHeightfieldMesh::update` 消费有序 `GridChange`，只重投影变更矩形并重算一圈法线，身份/邻域校验和完整 preflight 在提交前完成，无 mesh 资源重分配；更新结果与全量重建逐顶点一致。新增 `PlanetMeshPublication` 将多个 replacement/update/remove 组成单次 generation 提交，统一执行 tile/字节/邻边预算和跨 tile 高度校验，失败不改变旧快照。混合 LOD 缝合和 renderer graph 接线仍是独立边界。
- `PlanetMeshPublication::commit_with_delta` 在同一原子提交中返回 generation、按 `TileKey` 排序的新增/替换 tile 与此前确实驻留后被删除的 tile；对不存在 tile 的 remove 不产生删除 delta。delta 与下一快照都在状态切换前完成 fallible reserve，验证/容量/分配失败均保持 generation 和旧快照不变；原 `commit` API 保持兼容且不收集 delta。测试覆盖新增、替换、驻留删除、未知删除、generation 和失败原子性。该 delta 提供下游增量消费所需身份，但尚未接入 renderer scene/entity/asset 生命周期。
- 本次验证：`cargo test -p prism_world_terrain --lib --offline` 137 passed；`cargo check -p prism_world_terrain --no-default-features --offline` 通过；指定 Rust 文件 rustfmt 与三文件 `git diff --check` 通过。一次较早检查尝试曾因共享磁盘空间不足中断，释放空间后上述检查成功；未清理共享 target。
- 行星 `PlanetHeightfieldMesh::apply_lod_seam_plan` 现消费 resident-derived coarse-edge plan：在六面 cube-sphere 网格中移除对应边界 quad，以原始 tile grid 坐标映射 transition 索引，并按三角面与含整数 origin 的球心点积校正 outward winding；构建完整索引流前做 checked size/byte budget 与 vertex-range 验证，失败不改 mesh。最多支持两条 coarse 边；相邻边在共享角省略重叠端段并由四三角 corner fan 覆盖，允许预算内原子恢复基网格后切换计划。新增 `PlanetHeightfieldMesh::apply_lod_morph(field, plan, alpha)` 将 odd fine-edge 顶点沿局部 3D chord 移至 coarse endpoint 的连续中间位置；resident seam neighbor、tile/cache/revision/spec、边界坐标及 morph 位置/法线均在写入前校验，更新 O(S) 边界和一圈法线 stencil，零额外 mesh allocation。活动 seam/morph 期间拒绝 heightfield 局部 update，调用者须先恢复 alpha=0 并清除 seam，再更新并重建 seam/morph，避免混合新旧投影状态；该失败路径保持 mesh 不变。非平坦 ramp 测试覆盖六面×四边、alpha 0/0.5/1、chord 中点、单位朝外法线、非法 alpha 原子性及活动 morph 时 update 拒绝；resident-derived fine/coarse 几何对照覆盖六面×四边、两块 finer child 的完整边界点，逐点匹配 coarse mesh 对应顶点与弦段（`planetary_finer_tiles_close_to_coarse_edges_on_all_faces`）。验证：`cargo test -p prism_world_terrain --lib --offline` 129 passed，`cargo test -p prism_world_terrain --features mesh --lib --offline` 132 passed；此前 `cargo check` 通过。release CPU benchmark warmup 16 + 128 samples：planet morph side 129 为 P50/P95/P99 28.083/28.167/28.209 µs，side 257 为 54.459/54.667/61.209 µs，零额外 mesh allocation。该 benchmark 是 CPU wall-clock，不是 GPU timestamp、产品帧时或跨 adapter 证据。跨瓦 CPU 对照并不证明动态全过程无裂缝、复杂高度场法线/视觉效果、GPU vertex morph 或 renderer graph 接线；这些仍未完成。
- `PlanetMeshPublicationTransaction::stage_build_with_lod_seam_plan` 现将 resident-derived seam plan 与 planet mesh 一起 staging；commit 双向校验最终 snapshot 的 coarse-LOD 邻接与每个受影响 owner 的 transition metadata：新增 coarse tile 时拒绝漏缝 mesh，移除邻居时要求同事务清除 owner seam 或一起删除 owner；replacement 与失败原子性也有回归。拓扑校验仅检查每个变更 tile 周边最多 13 个候选 tile，不扫描整个 resident set、不分配临时拓扑表；每候选至多 4 边、对已排序 publication 做有界二分查找，约为 O(K log N) 时间、O(1) 临时空间（K 为事务变更 tile 数，N 为 resident tile 数）。此边界只覆盖当前 coarse transition consumer，不扩展 corner fan、finer-neighbor 视觉 stitching 或 GPU/RHI renderer graph。验证：无 feature 124 项、`mesh` feature 127 项通过；本批仍未完成真实 GPU、视觉与帧时验收。
- 新增 `validate_planet_heightfield_lod_edge(fine, edge, coarse, max_comparisons)`：严格要求相邻 2:1 TileKey 关系、相同奇数样本数和偶数 segment 数；用整数邻接/跨面边映射定位 coarse child half，只比较 fine 边界偶数样本与 coarse 对应样本，O(S) 工作且在采样前预算准入，不做浮点坐标近似。`PlanetMeshPublication::commit` 将它并入受影响边界预检：替换 fine tile 时校验其 coarse parent-neighbor，替换 coarse tile 时也检查仍驻留的 finer children；同事务同时替换时去重，所有同级与跨 LOD 比较共用 `max_neighbor_comparisons`，任何错配在快照提交前失败。回归覆盖六面×四边、两种 child half、反向跨面映射的非恒定高度 oracle、预算、非邻接、全匹配提交及单点高度不匹配导致整个 publication 保持 generation/snapshot 不变。专项验证：`cargo test -p prism_world_terrain --lib --offline` 134 passed、`cargo test -p prism_world_terrain --features mesh --lib --offline` 137 passed；隔离 target 下 rustfmt 与 `git diff --check` 通过。此检查保证重合高度样本一致，不验证高频细节滤波质量、动态 morph 全程视觉无缝、GPU consumer 或 renderer graph。
- `SdfGrid` 全量烘焙 signed Q16.16 到 X-fast 小端 i32；半开 XYZ 局部区域先 scratch 采样、成功才覆盖并递增 revision。按完整 dimensions、实际 source 调用数、输出+scratch 逻辑峰值字节预检，checked 算术及 fallible allocation 拒绝溢出/分配失败；任何采样失败保留原 bytes/revision。`quantize_sdf_mm` 用 i128 最近舍入、ties-away、超范围拒绝；`VoxelSdfSource` 只将声明为 Q8.8 的距离精确扩展 256 倍，拒绝 density/格式或规格不匹配，不伪造距离变换。空间 embedding 由 source 保持稳定；这一步没有执行曲面提取。
- `prism_world_physics_bridge` 已补齐 terrain → physics heightfield 的受限派生接缝：从 `HeightfieldTile` 的 signed millimetre 样本和 `PlanarHeightfieldEmbedding` 构造 `prism_physics_gpu::Heightfield`，在分配与发布前检查样本数量、逻辑上传字节、正采样间距、checked 算术和毫米到米的有限性。`WorldHeightfieldPhysicsBridge` 按 tile/revision 拒绝同 tile 的重复或倒退发布，所有转换和 fallible allocation 成功后才替换旧 snapshot；失败保持旧 snapshot，eviction 只清除可重建的 physics 派生状态。在此基础上，`WorldHeightfieldPhysicsResidency` 以 `TileKey` 稳定排序管理多个 resident tile，执行 tile 数、sample 总量和逻辑 upload bytes 聚合预算，并将尺寸/预算预检置于高度数组分配之前；替换、拒绝和 eviction 都保持 aggregate accounting 一致。独立 `prism_world_physics_cell_runtime` 由 `TerrainCellDecoder` 产生带 manifest sector 身份的完整 terrain snapshot，在下游 `SceneHost` 调用前完成 duplicate tile、跨 cell tile ownership、bridge budget、同 tile revision 单调性和旧 resident replacement 的 staging；host load/reload/evict error 或 panic 均保留旧 physics snapshot。ownership 索引在 host 成功后提交单 cell delta，不深拷贝全局 `cell_tiles`/`tile_owners`；heightfield residency 深拷贝仍由 aggregate tile/sample/upload-byte budget 约束。新增 PTHC v1 产品 terrain-cell payload codec：固定 64-byte frame/record header、little-endian ABI、CRC32 IEEE 损坏检测（不提供认证）、严格 manifest-sector 与 canonical tile 顺序校验、字段/reserved/尾随字节校验，以及 payload、记录数、单 tile/总 samples、decoded logical bytes 的全量 preflight 和 fallible allocation。codec reload 保留持久 revision、创建新的进程内 terrain cache identity；损坏 payload 不调用 host、不改变旧 resident。PTHC 校验 sector 身份但不把 `CellManifest::derived_hash` 解释为 payload digest；内容认证/签名与 hash 算法绑定由上层资产/transport 契约负责。2026-10-20 定向 offline 回归：terrain lib 112 项、physics cell runtime lib 18 项通过；codec 测试覆盖截断、校验和、ABI/记录字段、排序、预算、revision reload 与失败原子性，terrain 导入测试还覆盖错误报告精确长度的迭代器。这些是 CPU codec/消费契约验证，不等于物理显存/GPU uploader、产品 Scene/ECS decoder 组合、renderer graph 完整链路、多厂商 GPU 或端到端性能验收。
- 新增 `SdfSurfaceMesh` 在同分辨率的固定六四面体格上提取零等值面，规范 sample/edge key 焊接顶点并消除重复端点退化面；半开 cube 区域读取一圈 sample halo，任意等分辨率分区与全量提取的有向三角集合一致。负值在内、零在外，输出外向 CCW 索引与按实际编码三角面面积加权的单位法线；开放区域边界法线仍为单侧平均。i128 先减整数毫米 origin，再作 f64 插值及局部 f32 米制编码，拒绝精度超限、面翻转/塌缩、无定义法线和坐标超界；误差界不覆盖采样、量化或曲面近似误差。
- 新增 engine-neutral `DualContouringMesh` CPU 等值面提取切片：每个符号变化 cell 生成一个受约束 QEF 顶点，按稳定 primal-edge 顺序连接四边形并以坐标奇偶规则拆三角形；checkerboard face 明确 fail-closed 为 `AmbiguousTopology`，避免隐式选择非流形拓扑。cell、确定性遍历 work、顶点、三角形和逻辑峰值字节均在分配前 checked admission，输出使用 fallible reservation，失败不发布部分 mesh；局部 f32 编码、法线、退化/翻面和源 grid revision 均有校验。已通过 no-default terrain lib 142 项、mesh feature terrain lib 146 项及 no-default `cargo check`（2026-10-23，隔离 target）；极端 i32 SDF 差分先提升到 f64 再运算，mesh feature 另验证 native `prism_mesh::Mesh::validate()`。该切片只提供 CPU 提取器与 mesh asset 转换，尚未实现 Transvoxel、跨 LOD 接缝、增量 brick 重提取、GPU/RHI 或 renderer/physics 产品接线。
- 曲面构建对 cube/triangle/vertex/逻辑峰值字节设硬限制，checked 算术与 fallible reservation，失败不修改源 grid。非空构建三次遍历，排序/查边复杂度 `O(C + T log T + T log V)`；输出载荷 `24V + 12T`，峰值 `max(60T + 12V, 12T + 48V)`，不含借用 grid、旧 mesh、分配器开销/容量取整和固定遍历栈。no-default 回归 59 项及 `mesh` feature 回归 62 项已通过：覆盖 256 种符号 mask、距离极值、各向异性平面、零点焊接、闭球流形/Euler、任意区域分区、极大共用 origin、精确预算、错误路径与原生 Mesh 转换；这不是 Dual Contouring、Transvoxel 或混合 LOD 接缝实现，也尚未进入 renderer/physics 产品链路。
- 表示选择和重烘规划现按实际 XY 网格覆盖乘 Z 深度计算工作量/字节，覆盖位置敏感的 floor/ceil 边界、溢出、完整规格比较及输入排列无关的脏区合并；本批 terrain no-default 59 项、启用原生 mesh 后 62 项通过。IO 与 terrain no-default clippy 已通过，terrain mesh clippy 受磁盘耗尽阻断；性能/GPU 仍未验收，原生 mesh 的依赖 `prism_asset` 仍有既有文档告警。
- `prism_world_sim`：统一 tick/output/dirty-region budget，确定性局部侵蚀、持久体素/碎片 delta、水文容量约束、D8 流向/汇流量/流域分析、生态 biomass 更新和小端 delta codec。
- 2026-10-16 D8 分析切片验证：默认配置 20 项测试、`persist` 配置 25 项测试和 all-target/all-features strict Clippy 均通过；独立 `/private/tmp/prism-world-sim-d8-20261016` target 验证固定八邻居预算、区域裁剪、同高 tie-break、流域出口、河网阈值、`i32` 坐标边界及失败不变性。该证据只覆盖 CPU 拓扑分析，不代表 GPU 水面、侵蚀反馈或水引擎接线。
- M7 持久接缝已补齐：`simulation_delta_to_transaction` 将 `SimulationDelta` 编码为稳定 opaque `Bytes` forward + 同字段 `Delete` inverse，`WorldRuntime::append_simulation_delta` 自动使用 log next sequence 并写入 append-only log；simulation feature 测试覆盖 round-trip、错误和 facade 入口。
- M7 仿真状态回放已闭合：`WorldSim::apply_delta` 的正向校验与 `WorldSim::reverse_delta` 的反向校验均在写入前完成；`VoxelDelta` revision 2 保留 hardness，`WorldSim::fragments` 保留 stable-key 碎片，使质量归零的体素和已生成碎片可以无损回滚。no-default 22 项、persist 27 项单测和 all-target/all-features strict Clippy 通过。该切片仍不包含 GPU stencil、SDF 重烘或产品级时间轴调度。
- M7 SDF 只读接缝已补齐：`SparseSdfQuery`/`SdfQueryConsumer` 为多个消费者共享同一 sparse grid，明确区分 `Value`、`Miss` 与缺 brick 的 `Unknown`，覆盖负坐标、范围和缺失数据测试；不会把未知场误报为空场。
- `prism_world_network`：复用 stream/persist 身份的 AOI、delta envelope、authority/read-only/shared-LWW policy、sector handoff ledger 和顺序无关 PersistentId 合并；不拥有传输层或第二份世界真相。
- M8 session seam 已补齐：`Session` 提供 token gate、有限 outbound/inbound 队列、序号、累计 ACK、retry deadline、duplicate suppression 和失败状态；`MemoryTransport` 提供 no_std 可运行的双端测试传输。9 项 network 测试及 no-default-features/clippy 通过。真实 socket/加密 transport 仍由宿主提供。
- 这些 crate 已加入根 workspace，均有可运行算法和单测；它们是 world 接缝的 engine-neutral 实现，不宣称已经完成整个 renderer/physics/network transport 端到端产品。
- `SparseSdfBrickGrid` 已支持负坐标采样、跨 mip dirty propagation、删除后的父 mip 失效，以及 brick/region 的保守 i16 min/max 查询；查询返回 `None` 时表示稀疏数据未知，不能当作空场。
- `CpuSdfClipmap` 与 `SdfToroidalClipmap` 提供同一组进入/离开 delta：前者保留完整排序布局作为参考 planner；后者只存每 mip bounds，以模运算映射固定 slot，并按矩形差集枚举边缘，单格滚动成本与差集大小成正比，整窗跳跃则输出完整 eviction/rebuild 集。两者均采用负坐标 floor 除法、事务式中心提交和确定性 X/Y/Z/mip 顺序；parity、负坐标、多 mip、大跳跃和 slot 回用由同 crate 测试覆盖。`max_update_bytes` 是逻辑布局/delta payload 预算，不是 RSS/allocator 峰值保证；临时计数、bounds、分配器容量和旧/新驻留同时存活不计入该数。它仍是 CPU layout/residency 输入契约，不包含 GPU 纹理分配、SDF 采样、上传或 renderer graph 接线；这些仍需独立实现和真机验收。
- M7 GPU SDF brick lookup 已补齐：`GpuSdfLookup` 以固定容量上传 `BrickCoord` 排序的完整键、元数据和两枚 `i16`/`u32` packed samples，shader 用负坐标 floor division + binary search 做 `O(log B)` lookup，并以显式 `Value/Unknown` 结果区分缺 brick；上传、query、设备 binding 和 buffer budget 均在 queue write 前预检。WGSL ABI 5/5、residency 109/109、GPU crate strict no-deps Clippy 通过；跨沙盒 Apple M2/Metal 两项真实 parity 通过，证明 pipeline/dispatch/packed 解包/readback。该 slice 仍不等于 renderer、physics、GI、nav 或 runtime clipmap eviction 接线。
- terrain 在最大 quadtree 层级返回显式 `NoChildren`，LOD 非有限值和高度乘法溢出有确定性错误/饱和行为；simulation 的 tick、hydrology 和 delta codec 增加回绕/溢出、canonical record、输入大小和记录数限制；network merge 以 `TransactionId -> PersistentId` 索引去重，避免按对象桶线性扫描。

## 当前验证

2026-10-21 设计 §17.3 的局部 SDF 原子重烘已补齐：`WorldSdfPhysicsBridge::rebake_region` 仅枚举编辑区受影响 brick 与所有更粗父 mip，在回调前完成完整坐标、工作/驻留预算及样本缓冲分配预检；求值失败或 unwind 不改变旧快照，成功后一次发布 revision，未受影响样本保留原 allocation。`into_source()` 可转移源样本所有权，经 PSDC 和真实 archive/coordinator/cell host 重载至距离/梯度查询。bridge/cell 联合 lib/tests 共 71 项、rustdoc 示例 1 项及严格 API 文档检查通过；既有 Clippy 阻断和定向诊断结果见专项文档。Apple M2 优化基准中，64/4096/32768 个 mip0 驻留 brick 的固定 3-brick/1536-sample 编辑 P50 均约 31.2 µs，7168-brick/3670016-sample 重烘 P50 约 75.5 ms；此同步 CPU worker API 不等于异步对象脏传播、相机 clipmap 调度、solver/GI 消费或 GPU 上传完成。完整资源、分位数和边界见 `prism_world_physics_sdf_runtime_zh.md`。

2026-10-21 SDF 分离式重烘增加 `prepare_rebake → SdfRebakeWork::evaluate → commit_rebake`：仅持有目标样本缓冲，不借用或复制 resident grid，求值期间旧快照可查询。发布身份加精确 base revision 在首个写入前拒绝外部 bridge、clone、同版本驱逐重载及过期编辑。可选 `tasks` feature 复用现有 TaskPool，提供逐样本/末次采样后/提交前取消、panic payload 隔离、单次非等待结果收集和显式 owner 提交；句柄实际借用 pool，禁止 pool 先销毁而让句柄 drop 隐式 join。零 worker/强制 single 路径同步求值。单个 work scratch 受限，但 aggregate in-flight 数量/字节、prepare 分配与 owner 提交成本仍由运行时管理；未据此宣称对象脏传播、相机 clipmap、cell host 异步发布、多消费者、GPU 或产品实时调度完成。详细语义与质量评估见 `prism_world_physics_sdf_runtime_zh.md`。

该片 bridge/cell 联合回归 85 项、默认关闭 tasks 的 39 项、强制 single 的 48 项及 doctest 3 项通过；single 中 3 项真实 worker gate 测试仅断言 inline 模式后返回，真实后台执行另由正常 tasks 配置覆盖。API rustdoc 严格检查通过，严格 Clippy 仍受 4 处既有库告警阻断，定向诊断通过。Apple M2 优化基准的 7168-brick/3670016-sample 案例中，后台求值观测 P50/P99 约 99.1/201.5 ms，owner prepare/commit 各自 P99 约 2.124/1.471 ms；这说明求值移出 owner 后仍须继续控制提交预算，不构成产品帧时或相对同步路径加速比承诺。完整命令、分阶段分位数与排除项见上述专项文档。

SDF cell 已增加正式 PSDC v1 派生缓存编解码器：完整 sector/brick identity、物理尺度、revision 和量化样本保真；独立 payload/brick/sample/逻辑字节预算与 CRC、规范排序、保留位检查在样本构造前完成。真实 archive 文件经 FileCellProvider、coordinator、SdfPhysicsCellHost 到连续距离/梯度查询的接缝已有专项测试，包含失败保留旧快照及同一 coordinator 修复后重试。2026-10-21 bridge/cell 的 lib 与测试目标共 57 项通过（含 codec 11 项、文件集成 2 项），API rustdoc 严格检查通过；既有 Clippy 阻断及定向诊断结果如实保留。Apple M2 优化基准中，1024 个 brick/524288 个样本的 encode/decode P50 分别为 2.911/2.849 ms，仅衡量 CPU 编解码。格式、完整分位数、资源边界及验证记录见 `prism_world_physics_sdf_runtime_zh.md`；该接缝不代表 SDF solver、渲染或 GI 产品消费者已经完成。

2026-10-21 独立 target 定向复验：`cargo test -p prism_world_runtime --lib --offline` 108 项、`cargo test -p prism_world_gpu --features gpu --lib --offline` 123 项全部通过。跨沙盒 `WGPU_BACKEND=metal cargo test -p prism_world_gpu --features terrain_scatter --offline --test heightfield_placement --test heightfield_placement_deletion -- --nocapture --test-threads=1` 在 Apple M2 / Metal / IntegratedGpu 上分别通过 10 和 3 项，无忽略，实际执行 placement/deletion shader、完整键过滤及 indirect draw 回归；这不验证上述 renderer 产品接线或行星 placement。连续 SDF 查询的 bridge/cell 全目标回归共 44 项通过，解析距离/梯度、失败原子性与查询微基准结果见 `prism_world_physics_sdf_runtime_zh.md`。

2026-10-21 SDF clipmap 增加 toroidal bounds/slot planner 与双向 delta：`cargo test -j1 -p prism_world_residency --offline` 114 项、`cargo clippy -j1 -p prism_world_residency --all-targets --offline -- -D warnings`、`cargo test -j1 -p prism_world_runtime --lib --offline` 108 项及 `cargo test -j1 -p prism_world_gpu --features gpu --lib --offline` 123 项均通过；clipmap parity 覆盖负坐标、多 mip、单步和大幅跳跃，另验证 ring slot 回用及预算失败原子性。这里的 GPU feature 测试是编译/CPU 单测闭环，不是本轮新做的真实设备 GPU dispatch。torus 仍只提供 CPU residency 规划输入，尚未接到运行时调度、SDF 数据 buffer、GPU 上传或消费者。
2026-10-21 CPU clipmap delta 已接入独立的 `GpuSdfToroidalLookup` 固定槽上传/查询 contract：直接 modulo slot、完整 key 防 stale alias、按变化 slot 上传、显式 dirty brick 更新/删除失效；CPU sparse grid 仍为权威源。验证：runtime lib 108 项、residency lib 114 项、GPU feature lib 123 项、WGSL compile/ABI 6 项通过；Apple M2/Metal/IntegratedGpu 真实 GPU dispatch/readback 2 项通过；release CPU planner benchmark 已复测，最新观测值见 `prism_world_sdf_clipmap_performance_zh.md`。GPU ring 仍未接入产品 runtime 相机/焦点调度、physics solver、renderer、GI 或导航消费者，不宣称 M7 端到端完成。
2026-10-21 `prism_world_runtime` 新增 `WorldSdfClipmapScheduler`：以 owner 身份和 revision fence 保护两阶段 center plan/commit，自动标脏 entered brick、移除 exited dirty；批量 mark 排序去重后合并，原始请求与保留集合分别有数量/字节预算；clear 只申请当前 dirty 数量的逻辑字节标记表。集合幂等但有效非空 mark 必须推进版本，防止旧异步完成清除新编辑；`take_dirty` 是工作所有权转移，非完成确认。默认 runtime lib 123 项（含专项 14 项）、no-default lib check 和 runtime lib no-deps Clippy `-D warnings` 通过；no-default 测试被既有 HLOD 测试缺少 vec 宏阻断，asset_container/tasks 和 GPU feature 测试被其他模块 prism_asset scheduler 字段不一致阻断，未修改其文件。预算、复杂度、失败原子性与完整验证命令见 `prism_world_runtime_sdf_clipmap_zh.md`。该片不替代 SDF 烘焙/GPU 上传，未接 runtime 相机、physics solver、renderer、GI 或导航；同中心 plan 仍有 dirty 扫描/复制成本，aggregate in-flight 预算仍由宿主管理。

2026-10-21 World SDF physics bridge 新增 sphere 窄相 manifold 生成：使用连续距离梯度、正向外法线和固定八角点查询，保留缺数据 `Unknown`，支持相切零穿透并在负 signed-distance 内部样本保持外向法线。`cargo test -j 1 -p prism_world_physics_bridge --lib --offline`（独立 target）16 项通过，覆盖法线/witness ABI、大世界原点、相切、内部穿透、分离、缺失数据及非法半径。接缝只产生可供调用方收集的单点 ContactManifold，尚未接入 physics broad phase、接触生命周期或 solver tick；严格 Clippy 因数据盘空间不足未能启动，不能视作通过。详细语义和误差边界见 `prism_world_physics_sdf_runtime_zh.md`。

2026-10-21 SDF manifold 已接入 XPBD 外部 contact generator hook：在每个预测 substep、位置约束求解前，调用方可依据当前预测 pose 将 world-SDF 接触追加到既有接触批次；原 `Solver::step` 不启用该 hook。真实 SDF sphere manifold→两次 XPBD substep→动态球体推出穿透区域的专项回归通过；physics bridge 全库 17 项、`prism_world_runtime` 默认库 183 项和 `prism_world_gpu --features gpu` 库 123 项亦通过（均离线、独立 target）。这证明 SDF 窄相接入 XPBD 求解器的 CPU 路径，不证明相机 clipmap 到物理世界的生产调度、连续接触持久化/摩擦、renderer/GI/nav 消费或 GPU 物理路径已完成；当前 core crate 仍有既有 missing-docs 告警，未把严格 Clippy 标记为通过。

本轮 scene/world streaming 切片的定向验证：`rustfmt --edition 2024 --check pkg/prism_world_stream/src/lib.rs pkg/prism_world_runtime/src/lib.rs`，以及 `cargo test --offline -p prism_world_stream -p prism_world_runtime --lib --no-fail-fast`；结果为 `prism_world_stream` 5 项、`prism_world_runtime` 79 项通过。新增回归覆盖 manifest revision 触发 replacement、成功提交新 cell，以及坏 header 后旧 resident 状态可重试。工作树的全量 `cargo fmt --all -- --check` 仍会报告其他并行改动文件，本轮没有格式化或修改这些非目标包。

通过：

```text
cargo fmt -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world -- --check
cargo test -p prism_world_persist --features std
cargo test -p prism_world --features scene_host
cargo test -p prism_world_gpu
cargo test -p prism_world_gpu --features gpu --test wgsl_compile
cargo check -p prism_world_gpu --features gpu
CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo check -p prism_world_io -p prism_world_network -p prism_world_sim -p prism_world_terrain --no-default-features
cargo test -p prism_world_io --features std
cargo test -p prism_world_network -p prism_world_sim --features prism_world_sim/persist -p prism_world_terrain
CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo clippy -p prism_world_io --features std --all-targets -- -D warnings
CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo clippy -p prism_world_network -p prism_world_sim -p prism_world_terrain --all-targets -- -D warnings
CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo check -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world --no-default-features
```

核心 world policy 与 durable/scene tests 的既有验证包含 5 项 `prism_world` 测试、17 项 `SceneEcsHost` feature 测试、11 项 IO 测试、4 项 network 测试、6 项 simulation 测试、7 项 terrain 测试和 1 项 WGSL validator 测试；本轮持久化回归扩至 18 项并新增 2 项真实子进程锁回归，GPU CPU contract 已有 5 项通过。场景扩展仍不是完整流送到 ECS 的产品级端到端验收，不以这些数量代替完整覆盖。

万级 benchmark（本机单次 best，作为回归基线，不是跨硬件性能承诺）：

- stream demand solve：10,000 manifests，1.964 ms，5.09 M actions/s，3,366 次分配。
- residency schedule：10,000 records，0.368 ms，27.18 M records/s，3 次分配。
- world tick：10,000 items，1.818 ms，11.00 M items/s，3,374 次分配。
- persist CRDT merge：10,000 transactions，2.394 ms，4.18 Mtx/s；encode 7.642 ms，decode 10.246 ms。
- 以上由 `cargo bench -p prism_world_stream -p prism_world_residency -p prism_world_persist -p prism_world --benches` 实际执行，结果随 CPU、编译器和负载变化。

2026-10-15 受限环境中的严格 GPU 重跑曾失败：`No suitable graphics adapter found`（Metal 未返回 adapter）。同日经过用户允许的跨沙盒严格重跑已取得上文 Apple M2/Metal 的 pipeline、dispatch/readback 和 parity 通过证据；保留该受限探测失败记录，不将无 adapter 的运行算作通过。

### 地形原生网格与 Metal 实机回归

```text
CARGO_TARGET_DIR=/private/tmp/prism-world-kernel-target cargo bench -p prism_world_terrain --bench heightfield_bake
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 CARGO_TARGET_DIR=/private/tmp/prism-world-target cargo test -p prism_world_gpu --features gpu --test terrain_draw --test gpu_parity -- --nocapture
```

`terrain_draw` 在 Apple M2/Metal IntegratedGpu 上将真实原生 Mesh 顶点/索引送入 indexed raster，启用背面剔除，读回 64² RGBA8 像素检查覆盖、UV、法线和高度。81 顶点、384 索引；全量顶点 2,592 字节、索引 1,536 字节；3×3 高度重烘只需更新含法线邻域的 5×5 顶点，五段上传共 800 字节到同一个 GPU buffer。局部上传与全量替换的全部 16,384 读回字节一致，远离脏区的像素不变。64 次测量、8 次预热的主机墙钟 encode/submit/执行/同步/map/copy 为 P50/P95/P99 = 590/839/882 µs；这是 debug、小型离屏测试，不是 GPU timestamp、产品帧时或完整 render graph 集成结论。严格 parity 测试同次重跑通过。

CPU release 基准为确定性平面、16 次预热、每项 128 次测量；下表单位 ns，为 nearest-rank P50/P95/P99，本机本轮结果，不代表跨硬件承诺或峰值内存。局部更新均为 9 个高度样本、25 个 mesh 顶点。

| 网格 | 高度载荷 / mesh 载荷字节 | 全量高度 | 全量 mesh | 局部高度 | 局部 mesh |
|---|---:|---:|---:|---:|---:|
| 129² | 33,282 / 925,728 | 119,583 / 184,458 / 222,334 | 265,625 / 412,167 / 494,209 | 167 / 500 / 1,500 | 333 / 541 / 667 |
| 257² | 132,098 / 3,686,432 | 323,416 / 994,083 / 1,312,042 | 761,709 / 1,625,917 / 2,512,334 | 500 / 2,708 / 3,416 | 292 / 1,292 / 1,417 |

新增行星/SDF release 基准同为 16 次预热、128 次测量，2026-10-15 在其他并行任务存在时实测；不可与上表推导为无干扰加速比。行星测试投影地球半径整面，origin 为面中心，允许误差 2,000 mm 以容纳整面远坐标的 f32 编码；它不是局部毫米精度用例，精度由上文独立回归验证。

| 操作 | 逻辑输出字节 | P50 / P95 / P99（ns） |
|---|---:|---:|
| 行星 mesh 129² 全量投影 | 925,728 | 1,783,125 / 6,531,834 / 9,624,625 |
| 行星 mesh 257² 全量投影 | 3,686,432 | 5,563,375 / 10,421,000 / 15,644,250 |
| SDF 33³ 全量 | 143,748 | 14,667 / 33,625 / 40,792 |
| SDF 33³ 局部 27 样本 | 143,748（另加 108 scratch） | 166 / 625 / 2,500 |
| SDF 65³ 全量 | 1,098,500 | 127,500 / 240,291 / 283,083 |
| SDF 65³ 局部 27 样本 | 1,098,500（另加 108 scratch） | 167 / 833 / 2,291 |

行星法线当前每顶点读取邻域并重新投影，耗时明显高于平面 mesh；数据保留 P99，不把该 CPU 烘焙耗时写成产品帧时目标达标。SDF 基准 source 为廉价轴对齐平面，实际复杂 source 成本需要另测；所有字节数为逻辑载荷，不是 allocator/RSS 峰值。

`cargo clippy -D warnings` 在包含 `prism_world` 依赖闭包时会进入仓库其他并行改动的 `prism_asset`/`prism_ecs`，当前被这些既有告警阻断；world 自身新增代码已按相应 crate 的局部检查验证，不能把依赖闭包失败误报为全 workspace 通过。

cell host 后台 SDF 重烘装配增加默认关闭的 `tasks` feature：复用共享 pool，限制未收集任务总数/样本/scratch，终态收集才释放额度；成功 reload/evict/同 cell 重烘使旧工作失效，失败保留，提交前复验 aggregate resident 预算。wrapper 与现有 SceneHost/查询生命周期接通，不复制整张 grid、不重新加载作者内容。专项测试覆盖真实 worker 生命周期与跨 cell 预算竞争；pending 轮询基准覆盖满任务表最差索引，本机容量 1024 的 P99 为 1.482 µs。该能力不代表对象自动脏传播、相机 clipmap、跨消费者调度或 solver/GI/renderer 接入完成；资源与验证边界见 `prism_world_physics_sdf_runtime_zh.md`。

## 尚未交付的边界

2026-10-21 runtime SDF 更新：`WorldSdfGpuClipmap` 已接通 bounded dirty scheduler 与
固定槽 GPU ring 的增量上传、删除失效及同提交 consumer；专项证据与旧共享工作树依赖
阻断记录见 `prism_world_runtime_sdf_gpu_upload_zh.md`（提交 `40af58469`）。
`WorldSdfFocusDriver`/`WorldSdfGpuFocus` 进一步接通全局 i64 毫米坐标、负数 floor、
逐轴迟滞、stationary no-op 和上传成功后的 focus 提交；单 Y 平面覆盖强制要求 Y 无迟滞、
水平有迟滞时 radius >= 1，详见 `prism_world_runtime_sdf_focus_zh.md`。这些接缝不代替
对象自动 dirty/bake、真实消费者、三维窗口或跨消费者实时帧预算。
focus 切片当时的验证：runtime lib 148 passed，`gpu_proxy` lib 168 passed / 7 ignored，
no-default check 与限定范围 Clippy 通过；Apple M2/Metal 跨沙盒 focus 7/7、既有上传
4/4 实卡测试通过，默认忽略的宿主生命周期测试也已显式补跑 7/7 通过。
当时主树 GPU lib 被其他模块 `prism_asset::scheduler` 的编译错误
阻断，隔离副本 123 passed 不算主树通过；这是历史验证边界，不代表当前编译状态。
详细命令、微基准尾部及范围见上述 focus 文档。

2026-10-21 对象 SDF 更新：`WorldSdfObjectStore<T>` 已实现不可变对象 source、
每 brick contributor 索引、旧/新覆盖自动标脏、所有存活贡献者的局部 min 重烘，以及
owner/revision 隔离的 detached work。`WorldSdfLocalVolume` 提供规则量化体积的
i128 三线性采样；`WorldSdfGpuObjects<T>` 将 CPU 发布与 resident GPU dirty 准入连接，
CPU 重烘和 GPU 上传分别完成前均拒绝消费，窗口外修改在进入时加载最新结果。
它不是 ECS object-edit observer 或线程池装配，也不将现有单 Y 平面升级为三维窗口。
默认 runtime lib 183 passed，`gpu_proxy` lib 203 passed / 7 ignored，no-default check
通过，`prism_world_gpu --features gpu` lib 123 passed，release CPU 局部工作量微基准
1 passed；Apple M2/Metal 对象 GPU 6/6、focus 7/7、clipmap 上传 4/4 通过，限定范围
Clippy `-D warnings` 通过（依赖 render graph 的既有 unused-import warning 不在 lint 范围）。
对象测试在 sandbox 因无 Metal adapter 无法初始化，获准宿主环境实卡运行通过。
资源、精度、性能尾部及验收边界见 `prism_world_runtime_sdf_objects_zh.md`。

以下项目仍需要独立硬件或端到端证据，不能用 policy primitive 冒充完成：

行星 GPU 几何专项补充（早期 f32 ABI 历史）：旧非零高度 `planet_placement_gpu` 曾用独立 f64 投影 oracle 检查三轴误差不超过 2,048 mm、法线分量误差不超过 2e-6，并捕获负 i64 原点解码错误。旧 f32 世界坐标 record 已由后续 116-byte signed-i128 frame-local record 取代；当前规定矩阵的 frame-local 精度实测 ≤1 mm，见下方 2026-10-23 复验。仍无行星 placement 的完整产品 renderer consumer，因此数据 primitive 不能冒充产品渲染接口；换 sector 连续性及 renderer graph 产品接线仍待验收。

- M4 散布：行星/曲率/数据层/跨散布层规则、风与交互、逐键并发删除协议、索引增量编辑、蓝噪声目标、产品 proxy renderer 接线及大规模产品散布性能验收；`GpuPlanetPlacement` 已交付固定容量的 source/projection/finalize kernel、signed-i128 上传 ABI、compact/indirect 输出、Naga/device-free 回归，并新增逐候选 identity/rank/height numerator/accepted/invalid 语义 parity 回归。2026-10-22 Apple M2/Metal/IntegratedGpu 实卡 `planet_placement_gpu` 2/2 通过，覆盖平坦 tile 全候选 dispatch、非零高度与阈值 CPU 语义 parity；`planet_placement_deletion` 1/1 实卡通过，覆盖完整 planetary key 删除并保留其他 cell。以上不证明 CPU i64 几何坐标逐位 parity，也没有 scene/ECS decoder 或完整行星产品链路。二维贴地 placement 已有 reset compute、原生 registration snapshot、backend-neutral RenderGraph lowering、Null RHI 命令记录及 Apple M2/Metal 真实 GPU 回归；2026-10-22 可选 `placement_rhi` 进一步交付同设备/table 的持久注册 RAII lease、限定 shader ABI 的 draw pipeline/bind group 与 scene target、真实 RenderGraph record/encode/submit、有界 completion retirement、空删除复位及 submit/wait panic 恢复，Apple M2/Metal 严格集成 14/14，详见 `prism_world_gpu_heightfield_placement_deletions_zh.md`。仍未接入 `prism_render_scene` 产品 mesh/material/unified visibility、scene/ECS decoder 或持久删除的自动消费；有限单 color draw 不替代产品 renderer，没有新增产品性能/画质验收。已有内容中立有界 CPU/GPU 源散布、CPU/GPU 作者密度/种类场及各自可选刷子空间/有序索引、真实heightfield的CPU/GPU高度/离地带/坡度精确稳定过滤、独立身份的CPU二维精确贴地放置及其PPLD完整键持久删除、CPU/GPU二维贴地放置 kernel、CPU源/作者场/地形及GPU完整键删除与CPU持久文件恢复、独立 resident 密度 coverage/dither 与真实消费测试，不等于完整产品散布链路。
- M3：应用 schema 的完整零解析 mmap、GPU 直传和产品级吞吐仍未闭环；v2 `ContainerReader::from_mmap_file` 已直接使用 mmap 目录/table 索引，并有 `asset_container_mmap` feature 下贯穿 manifest/overlay/World cell provider 的 CPU 测试，但仍完整扫描并验证容器，provider 仍组装完整 cell `Vec<u8>`。共享动态优先级已接入但不提供抢占或跨队列无饥饿保证。任务池 Future/共享取消、固定 worker 及宿主原生 reactor → pollable coordinator → manifest/payload → `SceneHost` 已有真实文件端到端证据，同步兼容路径仍独立保留。离线缓存文件基准记录于 `prism_world_async_runtime_zh.md` 与 `prism_world_native_io_zh.md`，不能代替冷盘、资产包或产品负载验收。
- M6：多表示产品接线、Dual Contouring/Transvoxel 与混合 LOD 接缝、实际 renderer graph 接入；本批等分辨率 marching-tetrahedra 已通过 no-default 103 项和 `mesh` feature 106 项 CPU 回归，行星跨 tile 法线、高度局部增量投影和 `PlanetMeshPublication` 的跨 tile 原子 CPU 发布也已补齐，但其性能、混合 LOD 接缝和产品接线仍未验收。既有 heightfield 原子局部重烘、cube-sphere 增量投影、Q16.16 SDF 数据烘焙、native Mesh、离屏真实 GPU 绘制和 terrain-to-world-stream manifest 接缝不构成这些剩余缺口的完成证据。
- M7：SDF 与渲染/物理/GI 的多消费者真实运行时接入，以及大规模实时调度；CPU sparse grid、clipmap layout、GPU sorted lookup 与固定槽 toroidal lookup contract 已交付。World physics bridge/cell host 已有版本化原子发布、三线性距离与解析世界梯度查询，跨 brick 缺角点保持 `Unknown`，详见 `prism_world_physics_sdf_runtime_zh.md`；实际刚体接触生成/solver 响应、渲染/GI 消费及产品多消费者接线仍独立验收；simulation-to-world persistence 接缝已完成。
- M8：真实加密传输、多人 session、跨 sector handoff 的网络系统端到端一致性验证；engine-neutral session/authority/handoff seam 已完成。
- GPU proxy 已有 Apple M2/Metal parity、驻留 compact/真实间接绘制、sector/coordinator/宿主生命周期及限定 release P50/P95/P99 证据；物理显存峰值、多厂商/后端、完整产品性能与产品 scene/ECS decoder/renderer graph 集成仍未验收。

这些边界已有明确契约和可运行基础；只有补齐对应真实运行时证据后，才从本节移除。

## 不变量

- world 不复制 scene 内容；可变权威状态是 override delta，派生索引/代理/SDF/PCG 结果可删除重建。
- policy core 保持 `no_std + alloc`、`forbid(unsafe_code)`；文件系统和 GPU 能力只在明确 feature/provider 边界出现。
- 所有日志、排序、PCG、HLOD 和 readback contract 都有稳定顺序或稳定 key。
- 任何声称 GPU 或性能完成的结论必须同时记录 adapter、pipeline、dispatch/readback、P50/P95/P99、上传字节和显存/资源预算。

## 2026-10-20 当前工作树回归复验

在 HEAD `bcf70c1d` 及当时共享工作树状态下，使用独立临时 Cargo target 离线重跑：

| 命令 | 结果 | 证据边界 |
|---|---:|---|
| `cargo test -p prism_world_runtime --lib --offline` | 104 passed | runtime lib 单测；不代表 workspace 全量测试 |
| `cargo test -p prism_world_gpu --features gpu --lib --offline` | 123 passed | 验证 `gpu` feature 编译与 lib 单测；不代表本次在真实 GPU 上执行 shader |

两条命令均通过；临时 `CARGO_TARGET_DIR` 随后清理，Cargo 报告移除 5,614 个文件、约 1.1 GiB。测试是在含其他并行未提交改动的共享工作树上运行，结论仅适用于当时被编译的工作树内容；本节不声称独立干净 checkout、设备实测或端到端产品链路通过。

## 2026-10-21 SDF sphere batch → XPBD 接触消费

`prism_world_physics_bridge::SdfSphereContactBatch` 新增不可变、按 `(BodyHandle.index, generation)` 规范排序的 sphere 刚体批适配器：构造阶段校验重复/非法句柄、半径和球数/查询数/最坏接触数/逻辑 scratch 字节预算，并预留可复用 manifold scratch；每次调用按当前刚体位置做有界 SDF 查询，将完整可用采样生成的 manifold 暂存，只有全部查询成功且输出容量预留成功后才原子追加。Miss、Unknown、分离和接触分别计数；错误不改调用方输出。使用毫米世界坐标和浮动原点转换，有限性、范围及加法均检查。排序成本 `O(n log n)` 只在构造发生；每次生成 `O(n)` 查询、无排序，scratch 复用，输出按需扩容。

验证：`CARGO_TARGET_DIR=/private/tmp/prism-world-physics-bridge-sphere-batch-target-20261021 cargo test -p prism_world_physics_bridge --lib --offline`，23 passed；`rustfmt --edition 2024` 与指定文件 `git diff --check` 通过。首次编译发现并修复新模块对未声明 `alloc` crate 的误用。依赖 `prism_physics_core` 当前在此工作树会产生既有 lint 警告；bridge 新增字段缺文档告警已补齐，本条测试结果不代表严格 Clippy 通过。

本适配器只支持球心位于刚体原点的球体，且 core 暴露的 mass properties 无法区分“静态”与其他零逆质量 body，调用方须传入正确的静态 SDF body。它证明批量 SDF narrow-phase manifold 能经已有 XPBD contact-generator hook 推动动态球体；尚未接入 ECS 自动注册、broad phase、一般 collider shape、持久接触/摩擦、生产 tick/clipmap 调度，也未覆盖 GPU physics。这是 M7 的一段 CPU 消费链路，不等于 M7 多消费者或 World 系统整体完成。

## 2026-10-21 runtime 与 GPU feature 回归复验

在 HEAD `de3139e3badd` 及当前共享工作树状态下，使用 `CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0`
降低临时构建产物体积，离线重跑：

| 命令 | 结果 | 证据边界 |
|---|---:|---|
| `CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 cargo test -p prism_world_runtime --lib --offline` | 183 passed | runtime lib 测试；非 workspace 全量验证 |
| `CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 cargo test -p prism_world_gpu --features gpu --lib --offline` | 123 passed | GPU feature 编译与 lib 测试；不代表本轮实卡 shader 运行 |

此前报告的 `LayeredCellManifest` 缺少 `admitted_layers` 在当前源码中不可复现：
`pkg/prism_world_runtime/src/data_layers.rs` 暴露 `layers()`，相关 runtime/GPU 路径没有
`admitted_layers` 引用，且本次两条测试均成功编译并通过。该复验没有修改 World runtime/GPU 源文件；
结果适用于本次共享工作树快照，不构成干净 checkout 或 GPU 硬件验收。

## 2026-10-22 Planet placement frame-local 精度调查（历史问题，已由后续实现修复）

当时新增的 `write_request_in_frame` 原型将 `origin_mm - output_origin_mm` 在 CPU 侧精确计算为 i128，但旧 shader 仍观测到约 256 mm 局部误差（要求 ≤1 mm），故当时未通过精度验收。旧 f32/double-single ABI 和以下调查仅为历史过程；后续已采用 U256 定点投影与 signed-i128 frame-local record，当前实卡结果见本节后续 Q37..Q84 记录及 2026-10-23 复验。

旧实现调查曾怀疑 Metal fast-math 破坏 TwoSum/FMA double-single 误差恢复；相关探针已撤除，旧 80-byte f32 record 后来被 116-byte signed-i128 local record 取代。历史 Metal 命令 `CARGO_TARGET_DIR=/private/tmp/prism-target-planet cargo test -p prism_world_gpu --features terrain_scatter --test planet_placement_gpu planetary_placement_frame_preserves_millimetre_local_offsets_at_planet_scale --offline -- --nocapture` 曾失败（X 轴偏差约256 mm），后续生产算法修复后的当前复验则通过。

这一段的多 limb 定点算法、输出 ABI 和指定输入矩阵后来已实现并有 Apple M2/Metal 对照；planetary 产品链路仍因性能预算与 renderer 接线未通过而保持未完成。

2026-10-22 后续复核（旧 shader 结果）：`planet_placement_fixed_math.rs` 独立 Rust reference 的首批 9 项通过；当时旧 production GPU frame-local 仍失败约256 mm。以下记录了从 reference 到 U256 GPU 原语及生产算法替换的历史过程；其精度状态已被后续 17/17 实卡验证取代。性能及 planetary 产品 renderer 接线仍需分别验收。

同日临时 Metal 读回（旧 double-single/f32 shader）曾发现 `df_i128` 负数 two's-complement 转幅值进位错误及疑似 fast-math 影响误差恢复；旧探针已清除，旧 shader/ABI 已由后续整数路径取代。本段故障不是当前精度状态。

frame-local 回归 oracle 已增强为独立 f64 cube-sphere 投影，不再仅用 CPU 最终 i64 坐标减自身作为期望值。历史旧 shader 的 M2/Metal 复测中 X 轴 oracle 约为 `-0.481 mm`、GPU 为 `256 mm`，误差约 `256.481 mm`；后续生产定点 shader 已修复该失败。

精度路径审查补充（历史记录，后续生产 shader 已更新，见下方 2026-10-22 自适应 Q 记录及 2026-10-23 当前源码复验）：不能将方向降为 Q48/Q16 后声称满足全配置 1 mm。半径上限为 `2^52 mm`，Q48 单个方向 ULP 已对应 16 mm 径向位移；Q16 误差更大。Rust Q70 参考保留为高精度 oracle，并已加入 `terrain_scatter` lib 的测试编译（仅 `cfg(test)`）；当时 WGSL limb twin 尚未实现。GPU 输出仍为局部 `f32` position word，故当时 frame-local 精度仍失败；后续已完成整数输出 ABI 和生产 GPU 算法，实卡证据见下方。随后发现 Rust reference 接入 crate 后会对一个未使用的 WGSL 原语字符串产生 dead-code warning，已删除该占位字符串。针对 reference 的 crate 测试 9 项通过；之后重编译因隔离 target 将可用磁盘耗尽而失败（`No space left on device`），专项 target 已再次清理。

2026-10-22 精度算法推进（历史阶段状态）：先发现简单 Q64 cube 坐标量化加 u128 sqrt 在最大合法正 offset 的用例中与 Q70 reference 差 2 mm；已移除该算法。随后方向路径改为 Q84，从精确整数 cell-center 分子建立 cube 向量并使用 U256 计算平方根与方向商。确定性矩阵覆盖 tile level 0/1/8/15/22/30、样本边 2/3/13/257/65535/u32::MAX、六面及 tile/cell 首中末组合、radius 1/2^31/2^52、offset i64 两端、quarter-mm radial 余数和最大正 height-scale；另用 Q96 reference 覆盖极端高度。该阶段 reference 13/13 通过，但 production shader 当时尚未接入且 frame-local 仍失败；此旧状态已被下方生产 U256/adaptive-Q 实现和 M2 实卡复验取代，不代表当前实现。

历史 ABI 闭环问题：当时 80-byte record 的 f32 position 无法保留毫米级局部精度；后续已改为 116-byte record，每轴 signed-i128 local 毫米，并同步更新 deletion identity-prefix 消费。当前 ABI 语义通过指定测试矩阵，但产品 RHI/renderer decoder 尚未证明完整接线，且新增 record 带来每候选资源与计算开销。

2026-10-22 后续专项验证：修复 `planet_placement_u256_gpu.rs` 的 raw-string delimiter，移除未使用 helper，并让 arithmetic parity 向量的乘法结果落在 U256 范围内。另修正 Q70/Q84 frame-origin 单测从 limb 0 截断而造成的错误 oracle，改为完整 signed-magnitude 转 i128。Apple M2/Metal 实测 U256 add/sub/mul/div-rem/isqrt/shift 与 Rust reference 对拍通过；`cargo test -p prism_world_gpu --features terrain_scatter --test planet_placement_u256_gpu --offline -- --nocapture` 为 14 passed。`cargo test -p prism_world_runtime --lib --offline` 为 183 passed；`cargo test -p prism_world_gpu --features gpu --lib --offline` 为 139 passed。

历史结果：U256 原语对拍时 production placement 仍用旧实现，frame-local 测得 X=256 mm、独立 oracle 约 -0.481 mm；之后生产 shader、record ABI 与 deletion consumer 已完成更新，不能继续引用该失败代表当前状态。

后续 standalone WGSL U256 除法采用按 numerator/denominator bit length 收紧循环上界的版本，Metal 对拍通过；该 primitive 的优化不等同生产候选吞吐结论。production placement 后续另行接入 U256/adaptive-Q shader，此处当时状态为历史记录。

2026-10-22 对优化后的除法新增多 limb divisor/high-limb quotient GPU 向量：Apple M2/Metal 对拍及完整 reference suite 14/14 passed，补强了非单 limb 除数路径的实卡证据；仍不是行星生产候选路径或其吞吐基准。

同日新增 `u256_isqrt_scaled(value, 168)`，在不物化超过 256 位的 `value << 168` 的前提下逐 radix-4 pair 恢复平方根；当时测试只覆盖原语向量，production cube-sphere 全配置 kernel 尚未建立，后续生产接线及实卡覆盖见下方记录。

后续将整数 cube 向量的 Q84 归一化、signed limb 解码及 quarter-mm radial 到整毫米 ties-away-from-zero 投影加入实卡 parity kernel；当时 14/14 只证明 standalone 原语。随后这些算术迁入生产 placement shader，并由下方行星 placement GPU 实卡回归验证。
同轮补充负方向向量 [-3,4,-12] 与负 quarter-mm radial 的 M2/Metal 对拍，符号/幅值与 Rust Q84 reference 一致；完整测试仍为 14/14 passed。当时该 primitive 尚未接入 production placement，随后由下方 U256/adaptive-Q 实现取代。

2026-10-22 生产行星 placement 改为 U256 Q84 cube-sphere 整数方向、quarter-mm radial ties-away-from-zero 投影和 116-byte record ABI；xyz 为 signed-i128 frame-local 毫米，identity 前六 word 保持不变，deletion shader 仍按 identity 前缀比较。Apple M2/Metal planet_placement_gpu 3/3 passed，包含此前失败的 frame-local case（≤1 mm）、非零高度/阈值 parity 与全 tile dispatch；planet deletion 1/1 passed；U256/reference/六面方向实卡 suite 14/14 passed。此验收覆盖当前指定合法合同与 fixtures，不是所有数学输入的形式证明。

ABI/性能边界仍需继续审计：新 record 每候选增加 36 bytes，容量预算/编译检查已更新；原始固定 Q84 路径有高整数吞吐成本，之后改为 adaptive Q37..Q84 并测得 62,500 候选 submit+wait P50 约29 ms，仍未达到16.7 ms。产品 RHI consumer 是否有完整 decode/renderer 接线仍未证明；GPU kernel parity 不代表 planetary renderer 产品链路完成。

2026-10-22 固定 Q84 行星 placement 实机性能基准（历史基线，之后由 adaptive-Q 基准改进）：Apple M2/Metal IntegratedGpu、release、warmup 8 + 64 samples；256 / 4,096 / 62,500 candidates submit+wait 分别为 5,129/6,453/6,724 µs、5,549/7,536/9,853 µs、48,705/57,239/67,684 µs。计时是该主机/fixture 的 submit+wait 墙钟，不是 GPU timestamp；62.5k 超预算，后续 adaptive-Q 将 P50 降至约29 ms，但仍未满足16.7 ms。

同轮修正 `GpuPlanetPlacement::new` 的逻辑资源注释：record 116 bytes + compact index 4 bytes，故候选项合计 120 bytes；再加高度样本、params/status/indirect 后为 `120*max(capacity,1)+16*max(samples,1)+216`。

2026-10-22 Q84 shader 乘法热点试验：尝试按操作数有效 limb 缩短 U256 schoolbook 乘法，以及改为逐行 carry 累加；两种版本的 reference/Metal 算术套件均为 14/14 passed，但 Apple M2/Metal 62,500 候选 submit+wait P50 分别约 49.1 ms 与 52.9 ms，未优于基线 48.7 ms，且小档尾延迟波动明显。两种未经证明有收益的实现均已撤销，生产 shader 保持基线算法。该结果确认单纯乘法循环改写不足以闭合吞吐缺口。

2026-10-22 行星投影自适应 Q 精度：生产 kernel 由固定 Q84 改为按 quarter-mm radial numerator 的 bit length 选择 Q precision，范围 Q37..Q84；Q37 下限对最大合法行星半径仍留有数十位方向精度，radial 增大时逐步提高精度，极端高度/offset 自动回退至 Q84。signed-i128 输出 ABI、ties-away-from-zero、identity/deletion 前缀不变。Apple M2/Metal `planet_placement_gpu` 4 个设备测试通过，含 frame-local ≤1 mm、非零高度/rank parity、全 tile dispatch 和 `radius=2^52 mm + vertical_offset=i64::MAX + height_scale=i64::MAX` 的极端 radial；该新增用例以独立 Q96 整数方向/投影 reference 对照位置 ≤1 mm。该文件还包含 13 个 CPU fixed-math tests，integration 二进制总计 17/17 passed。U256 实卡/reference suite 14/14、GPU feature lib 139/139、runtime lib 183/183 passed。

同一 Apple M2/Metal release 基准（warmup 8 + 64 samples、排除 upload/readback）对比：固定 Q84 基线 62,500 candidates submit+wait P50/P95/P99=48,705/57,239/67,684 µs；自适应 Q37..Q84 为 29,095/35,624/41,718 µs，P50 改善约 40%。256 candidates 为 2,632/3,035/3,155 µs，4,096 为 3,837/4,438/4,622 µs。仍未达到 16.7 ms 的 62.5k 单帧预算；计时是 host submit+wait，不是 GPU timestamp，未证明其他 adapter 性能。自适应选择有整数位宽依据且有限极端组合已实卡对拍，但完整合法输入空间形式证明、多厂商验证、产品 renderer consumer 和产品规模 QoS 仍未完成。

2026-10-22 当前工作树专项复验：在 Apple M2/Metal IntegratedGpu 上，`CARGO_TARGET_DIR=/private/tmp/prism-target-planet-world-verify CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 cargo test -p prism_world_gpu --features terrain_scatter --test planet_placement_gpu --offline -- --nocapture` 为 17 passed，包含 frame-local 毫米精度（≤1 mm）、非零高度/rank、极端 radial 与 13 个 reference tests；同配置 `cargo test -p prism_world_gpu --features terrain_scatter --test planet_placement_deletion --offline -- --nocapture` 为 1 passed，验证完整 key 删除保留其他 cell。单独 frame-local 用例亦在宿主 Metal 通过。sandbox 本身没有 GPU adapter，故实卡结果来自获准宿主执行；同一完整文件在 sandbox 执行时 4 个设备用例因 adapter 不可用失败，不能作为算法失败。该复验未修改 GPU 源文件，不增加新的性能或产品 renderer 证据；本轮专用 target 已在结果确认后清理。

2026-10-23 planetary placement 性能复测：Apple M2/Metal IntegratedGpu，release warmup 8 + 64 samples，上传/最终 readback 排除计时；256 / 4,096 / 62,500 候选 submit+wait P50/P95/P99 分别为 3,409/7,887/11,479 µs、7,850/14,801/16,192 µs、29,058/41,002/49,312 µs。62,500 档 encode P50/P95/P99 为 7.792/11.416/2,030.208 µs（P99 尾部波动明显），提交等待仍远高于 16.7 ms；这说明 CPU encode 非主瓶颈，但 submit+wait 仍不是 GPU timestamp。尝试按有效 limb 缩短 U256 乘法循环前，planetary test 编译被共享工作树的 `prism_world_persist/src/durable.rs:410` 类型推断错误阻断，无法取得 shader parity 或性能证据，因此该未验证 shader 实验已撤回。数据来自 `CARGO_TARGET_DIR=/private/tmp/prism-target-planet-perf CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 cargo bench -p prism_world_gpu --features terrain_scatter --bench planet_placement --offline` 在获准 Apple M2/Metal 宿主运行；沙盒无 GPU adapter。隔离 target 已清理。

2026-10-23 planetary U256 乘法有效 limb 剪枝优化已在共享依赖类型推断修复后重试并保留。`u256_mul` 依据两个操作数的 bit length 计算有效 32-bit limb 数，只遍历可能非零的部分积；零操作数不做乘法迭代，常见窄 cell/tile/norm limb 显著减少循环，完整 8-limb 输入仍保持原有 modulo-2^256 结果及 overflow 规则。获准 Apple M2/Metal 实卡验证：`cargo test -p prism_world_gpu --features terrain_scatter --test planet_placement_gpu --offline -- --nocapture` 17 passed，`... --test planet_placement_u256_gpu ...` 14 passed，deletion 1 passed。release 基准命令 `CARGO_TARGET_DIR=/private/tmp/prism-target-planet-retry CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 cargo bench -p prism_world_gpu --features terrain_scatter --bench planet_placement --offline` 连续三轮 8 warmup + 64 samples，62,500 候选 submit+wait P50/P95/P99 分别为 24,343/25,124/25,213 µs、24,436/32,741/46,245 µs、25,418/31,335/39,265 µs；P50 对比同工作树优化前 29,058 µs 稳定降低约 12.5%–16.2%。256 候选三轮 P50 为 1.59/2.33/4.88 ms，4,096 候选第二、三轮为 3.65/3.73 ms；小档 submit 尾部受主机调度影响。此优化显著改善但未达 16.7 ms，P95/P99 仍有波动；指标是 host submit+wait 而非 GPU timestamp，不证明多厂商或完整产品帧吞吐。shader parity 和 performance evidence 均通过后保留。

2026-10-23 planetary RHI consumer 切片：新增 `prism_world_gpu::planet_placement_rhi`，在 placement/deletion producer snapshot 之上提供同 `GpuContext` 的设备身份认证、placement 与完整键 deletion 的提交入口、backend-neutral RenderGraph resource/pass hazard lowering，以及 queue completion 前的 snapshot lease 保活。placement/deletion 共用有界 in-flight 容量；failed drain 同时保留两类 pending snapshot，避免 GPU 仍可能使用 deletion 资源时提前释放。模块把 116-byte signed-i128 frame-local record 视为 opaque storage，不复用 planar 64-byte draw ABI。离线 `cargo test -p prism_world_gpu --features placement_rhi --lib planet_placement_rhi --offline` 为 4 passed，完整 `placement_rhi` feature lib 为 182 passed；Apple M2/Metal planetary RHI integration `cargo test -p prism_world_gpu --features placement_rhi --test planet_placement_rhi --offline -- --nocapture --test-threads=1` 为 2/2 passed，验证 placement/deletion 实际提交、间接计数、图 pass 数与完成后 lease retirement；planar RHI integration 另为 14/14 passed。该切片不是 `prism_render_scene` 产品 renderer：尚无 mesh/material/visibility decoder、真实 planetary draw pipeline、跨 sector publication 或多厂商/产品帧时验收。

2026-10-23 当前共享 checkout 复验（HEAD `079732b04545d89a561a578f963f204c3f773c49`）：`cargo test -p prism_world_runtime --lib --offline` 为 184 passed，`cargo test -p prism_world_gpu --features gpu --lib --offline` 为 139 passed。先前报告的 `LayeredCellManifest::admitted_layers` 编译错误在此快照不可复现：`data_layers.rs` 暴露 `layers()`，runtime 与 GPU feature 测试均成功编译。此为当前共享工作树的离线 CPU/lib 验证，不是干净 checkout、跨沙盒 Metal 实卡或 GPU 产品链路验收；本次未修改 runtime/GPU 源码。

2026-10-23 planetary placement 间接 draw consumer：新增受 `placement_rhi` feature 控制的 `planet_placement_draw`，读取 116-byte/29-word 行星记录中的 signed-i128 frame-local 毫米坐标、compact source-slot 索引及四字 indirect arguments，在真实单采样 color target 上绘制按像素定尺寸的 normal/rank marker。每次提交以 completion callback 保活 producer registration snapshot、bind group 与 uniform；固定 in-flight 上限约束 per-draw 资源数量。输入校验拒绝空 target、MSAA、pipeline/target format 不一致、非正/非有限尺寸与非有限 transform。Apple M2/Metal IntegratedGpu 实卡 `CARGO_TARGET_DIR=/private/tmp/prism-target-world-draw-host CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 cargo test -p prism_world_gpu --features placement_rhi --test planet_placement_draw --offline -- --nocapture --test-threads=1` 为 1 passed；验证 producer compute 后无 readback 的 indirect draw、GPU 实例计数为 4、color target 写入像素且完成回调释放 pending lease。沙盒本身无 Metal adapter，沙盒测试失败为 adapter acquisition，不是 shader/test 断言；获准宿主实卡通过。此仍为固定 billboard 的 GPU consumer slice，不是 `prism_render_scene` mesh/material/visibility/scene-ECS 接线，也未验证深度/透明排序、跨 sector continuity、多厂商、产品画质或规模化帧时；62,500 候选 placement 的既有 submit+wait P50 仍约 24 ms，未满足 16.7 ms 目标。

2026-10-23 planetary U256 除数右移 live-limb 优化实验已撤回：优化前当前 checkout 的 Apple M2/Metal release benchmark（warmup 8 + 64 samples）62,500 candidates submit+wait P50/P95/P99 为 24,218/25,069/25,528 µs；实验版三轮分别为 25,134/25,440/26,206 µs、25,222/25,835/26,078 µs、24,994/25,587/25,790 µs，未显示收益且 P50 均较本轮基线慢约 3.2%–4.1%。实验版 U256/CPU-reference/Metal arithmetic 14/14 和生产 planetary placement Metal 17/17 通过，但性能准入失败，因此 production shader 保持原实现。所有数据为同一 Apple M2/Metal fixture 的 host submit+wait 墙钟，不是 GPU timestamp；未来除法优化需减少实际设备执行成本并在受控重复基准中先证明收益。
