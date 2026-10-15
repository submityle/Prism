# Prism World 系统实现记录

日期：2026-10-15

本文只把已经存在源码、测试和提交记录的部分标为已交付；设计文档中的路线图、指标和效果目标仍然不能替代实现证据。

## 本批交接与验证范围

2026-10-15 按共享磁盘协调要求停止共享 target 构建；本批只整理 `prism_world_io`、`prism_world_terrain` 与本文，测试使用独立的 `/private/tmp` target，未清理或覆盖共享 target，未 push。

- 地形曲面提取使用 `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/private/tmp/prism-world-terrain-target cargo test --offline -j 1 -p prism_world_terrain --no-default-features`：59 passed、0 failed；随后启用 `mesh` feature 重跑为 62 passed、0 failed，包含原生 Mesh 转换回归。
- 新增 threaded 文件 provider 与 16 项回归测试源码，使用 `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/private/tmp/prism-world-io-target cargo test --offline -j 1 -p prism_world_io` 完成 28 passed、0 failed（其中 threaded 回归 16 项）。
- 本批运行的是指定文件的 `rustfmt --edition 2024 --config skip_children=true --check` 和 `git diff --check`，仅验证格式、解析及补丁空白，不等同于类型检查、clippy 或测试。
- `prism_world_io` 全部 target clippy 与 `prism_world_terrain` no-default 全部 target clippy 已通过；terrain `mesh` feature clippy 在编译依赖 `prism_asset` 时因磁盘耗尽中断。性能/GPU 验证及全 workspace 测试仍未运行；本批使用 `/private/tmp` 隔离 target，没有清理或覆盖共享 target。

后续异步 runtime 接线已补齐至固定 worker 文件到 `SceneHost` 的链路，详见 M3 和 `prism_world_async_runtime_zh.md`；任务池整合、跨 LOD 地形接缝和产品性能目标仍按下文边界保留。本记录不构成完整系统验收。

## 已交付切片

### M0-M2：空间、流送和常驻

- `prism_world_stream`：`i64` 毫米坐标、负坐标 `div_euclid/rem_euclid`、sector/Morton locality、cell manifest、需求集、预测预取、优先级、预算、迟滞驱逐和可回放 frame trace。
- `prism_world_residency`：A/B/C 三档状态、物化预算、确定性 promotion/demotion、HLOD 连续 alpha/dither、QoS governor、稀疏 SDF brick 和 PCG 图求值/脏传播/cache key。
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

### M4：可再生 CPU 散布 kernel

- `prism_world_residency::scatter` 在全局整数毫米三维格中每格生成一个 hash-jitter 候选；显式 seed/version/salt 与完整结构化 candidate key 决定位置。半开位置范围负责唯一归属，负坐标使用 Euclidean 除法；任意拆分、合并和请求顺序不会改变成员集合。完整 cache key 包含 kernel 版本、配置与范围，丢弃后可重建，不充当持久对象身份。
- 可选最小距离采用 jittered lattice 上的 Matérn type-II 式筛选，重建未请求邻格的原始候选，以 `(priority hash, full key)` 排名消除边界依赖；不是 Poisson 输入、蓝噪声保证、最大填充或递归 dart throwing。距离比较使用宽整数，等于半径允许保留。
- 预检工作量上界为 `N × (2 × ceil(radius/spacing) + 1)^3`，无筛选时为 `N`；输出数量和 Vec 容量载荷字节另有硬限制。checked 算术、一次 fallible exact reservation，超限整体失败，不返回随预算变化的截断前缀；字节限制不含分配器元数据、栈或外部消费者副本。
- no-default 测试 35 项（其中散布新增 20 项）、all-target clippy `-D warnings` 和格式检查通过，覆盖全局参考筛选、跨 cell 最小距离、未请求邻格阻挡、任意分区双向请求、缓存重建、预算与 i64 极值。这是 CPU 派生数据 kernel；尚无 GPU 散布生成、proxy 渲染/密度淡入、地表投影或百万点性能验收结论。

### M5：持久化

- `prism_world_persist`：`PersistentId`、规范化 `OverrideDelta`、append-only log、确定性 LWW CRDT、snapshot/timeline、undo/redo、little-endian codec/checksum。
- `std` feature 提供 `FileJournal` 和 `SnapshotStore`：版本头、长度/序号/payload/checksum、批量 append、`flush + sync_all`、重启恢复、损坏/截断尾部裁剪、乱序/重复拒绝、同目录临时文件原子替换和父目录同步。
- `FileJournal::open` 在读取、初始化或恢复之前取得独占 OS 文件锁，并保持到句柄销毁；第二个句柄明确返回 `Locked`，不支持绕过锁写入或替换锁定路径。写入、flush 或 sync 的 I/O 失败会 poison 当前句柄，即使尾部回滚成功也必须销毁并重开；回滚失败同时保留原始错误和回滚错误。故障注入走同一生产 writer，覆盖 seek/短写/flush/sync 和 truncate/seek/flush/sync 回滚失败。失败后内存 log 保持旧视图，但回滚失败且磁盘已有完整记录时，重开允许恢复这些记录；batch 不是崩溃原子事务。
- 本轮 macOS 验证：`cargo test -j 1 -p prism_world_persist --features std --quiet` 的 18 项测试、真实子进程锁回归 2 项，以及 `cargo check -j 1 -p prism_world_persist --no-default-features` 通过。锁内读测试复用持锁文件句柄并恢复游标；跨进程回归证明父进程持锁时子进程得到 `Locked`，释放后可重开并读取已提交前缀，且文件字节不变。尚未提供 Windows 实机验证。
- 相关提交：`4d0fbfebe`；Scene/ECS 接缝提交：`34656b305`；GPU workspace 集成：`c94d5cf13`。

### M3：异步 IO/provider

- `prism_world_io`：分块读取请求、稳定优先级/FIFO 准入、活动读取 round-robin、单调请求 ID 的 fail-closed 耗尽保护、显式容量背压、可取消 poll 操作、连续 chunk/short-read/checksum 校验和 std 文件 provider。
- 默认构建保持 `no_std + alloc`；`std` 只增加 scheduler 与文件系统 provider，不把 IO、资产数据库或 ECS 引入 world policy core。
- `prism_world_runtime`：真实调用 `StreamScheduler` 生成 load/evict 动作，并通过有界、可取消、优先级排序的 provider 队列提交 cell；完成后执行固定 manifest header 校验，再进入 `SceneHost::load_cell`，失败/驱逐写入可 replay trace。std 构建新增 `IoSchedulerAdapter`，实际桥接 `prism_world_io::IoScheduler` 的 chunk/completion/cancel/backpressure/error 事件，并保留同步 `ChunkProvider` 兼容路径。
- `FileCellProvider` 复用 `prism_world_io/std` 的真实 `FileProvider`，以 sector 注册完整 manifest/文件 offset/checksum，拒绝过期 manifest 并保留调度优先级。真实临时文件测试覆盖文件→分块 scheduler→header/checksum→`SceneHost`，及短读、打开失败、取消和错误隔离；底层每次 poll 为有界阻塞读取，不宣称已变成线程池异步文件 IO。
- 本批新增独立的 `ThreadedFileProvider` 源码：固定 std worker、预留槽位/队列、受限 chunk、ACK 驱动的后续读取、worker 内文件打开/读取/关闭，以及非 joining 的取消/shutdown。取消的 Ready 缓冲区交由 worker 清理；槽位在结果和文件销毁后才释放。waker 克隆/替换后的销毁及唤醒位于 metadata 锁外，worker/批量 shutdown 隔离唤醒回调 panic。逻辑 payload 约束为 `capacity × max_chunk_size`，不包含调用方已取走的 chunk、线程栈、元数据、分配器开销或 RSS；阻塞 OS 调用只能合作取消，返回前仍占槽位。16 项回归覆盖范围/短读/打开失败、准入、parked ACK、queued/running/ready 取消、重复 shutdown、last-owner drop、唤醒时序及 panic，已与 crate 其余测试合计 28 项通过。
- 新增 `CellIo`/`CellFrameCoordinator` 的 task-context poll 接口；`ScheduledCellIo` 聚合完整 cell，每次最多消费 16 个 chunk，随后唤醒并让出。`WorldFrameCoordinator<P>` 是同步兼容别名，保留 `new/poll_io`；旧路径仍可能阻塞，不把兼容 API 称为异步。同步 runtime 的 ID 也改为单调非零、耗尽显式失败，并用 ID 保持同优先级 FIFO。
- `RoutedCellIo` 在提交前检查完整 manifest 版本，保留注册的 offset/checksum 与调用方优先级；`ThreadedCellIo::threaded` 将其接入独占固定 worker pool，要求 active limit 不超过物理槽位。真实 archive 回归从 stream demand 到 worker 分块、checksum/header 校验和 `SceneHost` 提交，错误内容不会进入 host；不是 scene/ECS 或 GPU 端到端验收。
- `QueuedThreadedFileProvider` 提供取消清理后的有界容量等待：所有 clone 共享 `capacity` 条准入记录与 `capacity` 个物理槽位，FIFO 授予槽位后再唤醒；等待中不保留 payload/file、不使用自唤醒重试。直接 `ThreadedFileProvider::start` 仍保留立即背压契约。IO crate 42 项测试通过，包含阻塞清理后的替换请求、2 MiB 文件校验、多 scheduler、FIFO grant、取消/drop、shutdown 及重入/panic waker。
- `IoSchedulerAdapter` 以带 ID/原始请求的 `Failed` 终态关联错误；scheduler 串行锁内跟踪实际 start/poll 回调，scheduler 的显式取消 ID 优先，provider 不能伪造其他请求的取消 ID。并发 blocking reader 保留各自结果、共享 provider 唤醒可唤醒全部等待者，public poll 不抢占其结果；终态被消费前仍计入 capacity，取消会唤醒停驻 reader。
- adapter public poll 的 waker 由最新 Context 替换，blocking reader 按请求 ID 各有一个注册；锁外执行 clone/drop/wake，锁内 provider 的同步通知延迟到 adapter guard 解锁后派发，隔离回调 panic 并保留重入注册与并发通知。本轮 IO 42 项、runtime 53 项及 no-default runtime 6 项测试通过；两种 feature 的 all-target clippy 通过 `-D warnings`。真实 archive release 测量与限定条件见 `prism_world_async_runtime_zh.md`。
- `IoByteLimits` 为 adapter/file provider 增加独立的单请求和累计声明字节准入，包含 44 字节 header；mutex 内在分配请求 ID/启动 IO 前检查上限及累计溢出，拒绝不消耗 quota/ID。queued、active 和未消费的 Completed/Failed 均保留完整配额，消费 Chunk 不释放；兼容构造器保留无配置字节上限行为，但仍拒绝算术溢出。
- 聚合首次 chunk 时 fallible reserve 完整声明大小，后续 append 也为 fallible；分配失败取消底层操作，待 scheduler 移除后才以原 request ID/分配错误报告终态。累计上限 B 限制逻辑准入，不是 RSS 硬上限：保留 Chunk 副本与聚合结果的逻辑长度可接近 2B，Vec 实际容量/分配器、provider、事件元数据及交还调用方后的缓冲区不在计数内。
- 预算补丁在并行任务验证快照中通过 runtime 全 feature 测试 27 项、no-default 测试 3 项及两种 feature 的 all-target clippy/check/fmt；预算批新增 8 项预算/文件验证，既有 5 项并发/关联测试先前额外重复 100 轮通过。历史收尾重跑曾因并行修改使用 `error.to_string()` 而触发泛型错误类型缺少 Display 的 E0599；后续已恢复 Debug 格式化，并在本批 threaded provider 修改之前通过 runtime 全 feature 27 项测试。该记录不代表本批新 provider 已被编译，也不将当前共享工作区误报为全部通过。上述 runtime 验证不依赖 `prism_asset`。

### M6-M8：地形、仿真与网络边界

- `prism_world_terrain`：cube-sphere 六面映射、四叉树 tile/跨面邻接、连续 screen-space LOD/morph、i64/Q32.32 局部坐标误差界、确定性高度采样和小端体素布局。
- M6 纵切已补齐：`TerrainStreamAdapter` 将 `TileKey` 映射为稳定 `SectorId`，计算保守行星毫米 AABB，生成稳定 `CellManifest`/derived hash，并可直接交给 `StreamScheduler`；12 项 terrain 测试、fmt/check/clippy 通过。
- M6 可执行 heightfield kernel：全量烘焙为带正毫米比例的 signed i16 高度，超范围明确失败；`BakeLimits` 在采样和分配前限制数量/载荷字节，分配失败可返回错误。局部脏区按网格 floor/ceil 保守覆盖，先完成 scratch 采样再提交；失败不改变高度或 revision。`GridChange` 是单个稳定 cache owner 的有序 token，不是跨缓存的全局身份。
- `HeightfieldMesh` 输出本地 XZ 米制索引网格，+Y 三角绕序和有限差分单位法线；更新覆盖脏高度及一圈法线邻域、不重分配原 mesh。载荷为 `32×顶点数 + 24×quad数` 字节。`mesh` feature 输出真实 `prism_mesh::Mesh`，不是渲染占位数据。
- `PlanetHeightfieldMesh` 将已烘焙高度投影为六面 cube-sphere 索引网格；TileKey 整数身份经 f64 UV/归一化投影，减 i64 毫米 origin 后编码为局部 f32 米制位置。半径限 `1..=2^52` 毫米，非正半径+高度、非有限值、载荷超限和超过位置误差预算均失败；误差界包含浮点运算及编码，不包含高度量化、三角插值和 LOD 接缝。测试覆盖六面向外绕序/单位法线、同级相邻 tile 与跨面边界位置、地球半径第 30 层 tile 小于 0.001 mm 的数值误差界及原生 Mesh。边缘法线为单侧差分，尚非跨 tile 平滑法线或混合 LOD 缝合；当前为全量投影。
- `SdfGrid` 全量烘焙 signed Q16.16 到 X-fast 小端 i32；半开 XYZ 局部区域先 scratch 采样、成功才覆盖并递增 revision。按完整 dimensions、实际 source 调用数、输出+scratch 逻辑峰值字节预检，checked 算术及 fallible allocation 拒绝溢出/分配失败；任何采样失败保留原 bytes/revision。`quantize_sdf_mm` 用 i128 最近舍入、ties-away、超范围拒绝；`VoxelSdfSource` 只将声明为 Q8.8 的距离精确扩展 256 倍，拒绝 density/格式或规格不匹配，不伪造距离变换。空间 embedding 由 source 保持稳定；这一步没有执行曲面提取。
- 新增 `SdfSurfaceMesh` 在同分辨率的固定六四面体格上提取零等值面，规范 sample/edge key 焊接顶点并消除重复端点退化面；半开 cube 区域读取一圈 sample halo，任意等分辨率分区与全量提取的有向三角集合一致。负值在内、零在外，输出外向 CCW 索引与按实际编码三角面面积加权的单位法线；开放区域边界法线仍为单侧平均。i128 先减整数毫米 origin，再作 f64 插值及局部 f32 米制编码，拒绝精度超限、面翻转/塌缩、无定义法线和坐标超界；误差界不覆盖采样、量化或曲面近似误差。
- 曲面构建对 cube/triangle/vertex/逻辑峰值字节设硬限制，checked 算术与 fallible reservation，失败不修改源 grid。非空构建三次遍历，排序/查边复杂度 `O(C + T log T + T log V)`；输出载荷 `24V + 12T`，峰值 `max(60T + 12V, 12T + 48V)`，不含借用 grid、旧 mesh、分配器开销/容量取整和固定遍历栈。no-default 回归 59 项及 `mesh` feature 回归 62 项已通过：覆盖 256 种符号 mask、距离极值、各向异性平面、零点焊接、闭球流形/Euler、任意区域分区、极大共用 origin、精确预算、错误路径与原生 Mesh 转换；这不是 Dual Contouring、Transvoxel 或混合 LOD 接缝实现，也尚未进入 renderer/physics 产品链路。
- 表示选择和重烘规划现按实际 XY 网格覆盖乘 Z 深度计算工作量/字节，覆盖位置敏感的 floor/ceil 边界、溢出、完整规格比较及输入排列无关的脏区合并；本批 terrain no-default 59 项、启用原生 mesh 后 62 项通过。IO 与 terrain no-default clippy 已通过，terrain mesh clippy 受磁盘耗尽阻断；性能/GPU 仍未验收，原生 mesh 的依赖 `prism_asset` 仍有既有文档告警。
- `prism_world_sim`：统一 tick/output/dirty-region budget，确定性局部侵蚀、持久体素/碎片 delta、水文容量约束、生态 biomass 更新和小端 delta codec。
- M7 持久接缝已补齐：`simulation_delta_to_transaction` 将 `SimulationDelta` 编码为稳定 opaque `Bytes` forward + 同字段 `Delete` inverse，`WorldRuntime::append_simulation_delta` 自动使用 log next sequence 并写入 append-only log；simulation feature 测试覆盖 round-trip、错误和 facade 入口。
- M7 SDF 只读接缝已补齐：`SparseSdfQuery`/`SdfQueryConsumer` 为多个消费者共享同一 sparse grid，明确区分 `Value`、`Miss` 与缺 brick 的 `Unknown`，覆盖负坐标、范围和缺失数据测试；不会把未知场误报为空场。
- `prism_world_network`：复用 stream/persist 身份的 AOI、delta envelope、authority/read-only/shared-LWW policy、sector handoff ledger 和顺序无关 PersistentId 合并；不拥有传输层或第二份世界真相。
- M8 session seam 已补齐：`Session` 提供 token gate、有限 outbound/inbound 队列、序号、累计 ACK、retry deadline、duplicate suppression 和失败状态；`MemoryTransport` 提供 no_std 可运行的双端测试传输。9 项 network 测试及 no-default-features/clippy 通过。真实 socket/加密 transport 仍由宿主提供。
- 这些 crate 已加入根 workspace，均有可运行算法和单测；它们是 world 接缝的 engine-neutral 实现，不宣称已经完成整个 renderer/physics/network transport 端到端产品。
- `SparseSdfBrickGrid` 已支持负坐标采样、跨 mip dirty propagation、删除后的父 mip 失效，以及 brick/region 的保守 i16 min/max 查询；查询返回 `None` 时表示稀疏数据未知，不能当作空场。
- terrain 在最大 quadtree 层级返回显式 `NoChildren`，LOD 非有限值和高度乘法溢出有确定性错误/饱和行为；simulation 的 tick、hydrology 和 delta codec 增加回绕/溢出、canonical record、输入大小和记录数限制；network merge 以 `TransactionId -> PersistentId` 索引去重，避免按对象桶线性扫描。

## 当前验证

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

## 尚未交付的边界

以下项目仍需要独立硬件或端到端证据，不能用 policy primitive 冒充完成：

- M3：`prism_tasks` 生命周期/取消整合、资产包解压、mmap schema 视图、GPU 直传和产品级吞吐仍需实现或独立运行时验证；固定 worker → pollable coordinator → manifest/payload → `SceneHost` 已有真实文件端到端证据，同步兼容路径仍独立保留。离线缓存文件基准记录于 `prism_world_async_runtime_zh.md`，不能代替冷盘、资产包或产品负载验收。
- M4 散布：GPU 候选生成/proxy 流、连续密度切换和大规模性能验收；当前完成的是内容中立、可再生且有界的 CPU 散布 kernel。
- M6：多表示产品接线、Dual Contouring/Transvoxel 与混合 LOD 接缝、行星跨 tile 法线/增量投影及实际 renderer graph 接入；本批等分辨率 marching-tetrahedra 已通过 no-default 59 项和 `mesh` feature 62 项 CPU 回归，但其性能、LOD 接缝和产品接线仍未验收。既有 heightfield 原子局部重烘、cube-sphere 全量投影、Q16.16 SDF 数据烘焙、native Mesh、离屏真实 GPU 绘制和 terrain-to-world-stream manifest 接缝不构成这些缺口的完成证据。
- M7：SDF 与渲染/物理/GI 的多消费者真实运行时接入，以及大规模实时调度；simulation-to-world persistence 接缝已完成。
- M8：真实加密传输、多人 session、跨 sector handoff 的网络系统端到端一致性验证；engine-neutral session/authority/handoff seam 已完成。
- GPU proxy 的真实 adapter、dispatch/readback 正确性已有上述 Apple M2/Metal 证据；P50/P95/P99 性能、显存峰值、多厂商/后端与 renderer 端到端集成仍未验收。

这些边界已有明确契约和可运行基础；只有补齐对应真实运行时证据后，才从本节移除。

## 不变量

- world 不复制 scene 内容；可变权威状态是 override delta，派生索引/代理/SDF/PCG 结果可删除重建。
- policy core 保持 `no_std + alloc`、`forbid(unsafe_code)`；文件系统和 GPU 能力只在明确 feature/provider 边界出现。
- 所有日志、排序、PCG、HLOD 和 readback contract 都有稳定顺序或稳定 key。
- 任何声称 GPU 或性能完成的结论必须同时记录 adapter、pipeline、dispatch/readback、P50/P95/P99、上传字节和显存/资源预算。
