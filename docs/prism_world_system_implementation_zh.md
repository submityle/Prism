# Prism World 系统实现记录

日期：2026-10-15

本文只把已经存在源码、测试和提交记录的部分标为已交付；设计文档中的路线图、指标和效果目标仍然不能替代实现证据。

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

### M5：持久化

- `prism_world_persist`：`PersistentId`、规范化 `OverrideDelta`、append-only log、确定性 LWW CRDT、snapshot/timeline、undo/redo、little-endian codec/checksum。
- `std` feature 提供 `FileJournal` 和 `SnapshotStore`：版本头、长度/序号/payload/checksum、批量 append、`flush + sync_all`、重启恢复、损坏/截断尾部裁剪、乱序/重复拒绝、同目录临时文件原子替换和父目录同步。
- `FileJournal::open` 在读取、初始化或恢复之前取得独占 OS 文件锁，并保持到句柄销毁；第二个句柄明确返回 `Locked`，不支持绕过锁写入或替换锁定路径。写入、flush 或 sync 的 I/O 失败会 poison 当前句柄，即使尾部回滚成功也必须销毁并重开；回滚失败同时保留原始错误和回滚错误。故障注入走同一生产 writer，覆盖 seek/短写/flush/sync 和 truncate/seek/flush/sync 回滚失败。失败后内存 log 保持旧视图，但回滚失败且磁盘已有完整记录时，重开允许恢复这些记录；batch 不是崩溃原子事务。
- 本轮 macOS 验证：`cargo test -j 1 -p prism_world_persist --features std --quiet` 的 18 项测试、真实子进程锁回归 2 项，以及 `cargo check -j 1 -p prism_world_persist --no-default-features` 通过。锁内读测试复用持锁文件句柄并恢复游标；跨进程回归证明父进程持锁时子进程得到 `Locked`，释放后可重开并读取已提交前缀，且文件字节不变。尚未提供 Windows 实机验证。
- 相关提交：`4d0fbfebe`；Scene/ECS 接缝提交：`34656b305`；GPU workspace 集成：`c94d5cf13`。

### M3：异步 IO/provider

- `prism_world_io`：分块读取请求、稳定优先级/FIFO 排序、显式容量背压、可取消 poll 操作、连续 chunk/short-read/checksum 校验和 std 文件 provider。
- 默认构建保持 `no_std + alloc`；`std` 只增加 scheduler 与文件系统 provider，不把 IO、资产数据库或 ECS 引入 world policy core。
- `prism_world_runtime`：真实调用 `StreamScheduler` 生成 load/evict 动作，并通过有界、可取消、优先级排序的 provider 队列提交 cell；完成后执行固定 manifest header 校验，再进入 `SceneHost::load_cell`，失败/驱逐写入可 replay trace。std 构建新增 `IoSchedulerAdapter`，实际桥接 `prism_world_io::IoScheduler` 的 chunk/completion/cancel/backpressure/error 事件，并保留同步 `ChunkProvider` 兼容路径。

### M6-M8：地形、仿真与网络边界

- `prism_world_terrain`：cube-sphere 六面映射、四叉树 tile/跨面邻接、连续 screen-space LOD/morph、i64/Q32.32 局部坐标误差界、确定性高度采样和小端体素布局。
- `prism_world_sim`：统一 tick/output/dirty-region budget，确定性局部侵蚀、持久体素/碎片 delta、水文容量约束、生态 biomass 更新和小端 delta codec。
- `prism_world_network`：复用 stream/persist 身份的 AOI、delta envelope、authority/read-only/shared-LWW policy、sector handoff ledger 和顺序无关 PersistentId 合并；不拥有传输层或第二份世界真相。
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
cargo test -p prism_world_network -p prism_world_sim -p prism_world_terrain
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

`cargo clippy -D warnings` 在包含 `prism_world` 依赖闭包时会进入仓库其他并行改动的 `prism_asset`/`prism_ecs`，当前被这些既有告警阻断；world 自身新增代码已按相应 crate 的局部检查验证，不能把依赖闭包失败误报为全 workspace 通过。

## 尚未交付的边界

以下项目仍需要独立硬件或端到端证据，不能用 policy primitive 冒充完成：

- M3：真实文件/资产 provider 到 SceneHost 的产品级异步吞吐、解压和 GPU 直传链路仍需独立运行时验证；`IoSchedulerAdapter` 的内存 provider 端到端测试已通过。
- M6：terrain 多表示、增量重烘以及 terrain-to-render/world streaming 的端到端接入。
- M7：simulation-to-world persistence、SDF 一场多用和大规模实时调度接入。
- M8：真实传输层、多人会话、认证/重试、跨 sector handoff 的端到端一致性验证。
- GPU proxy 的真实 adapter、dispatch/readback 正确性已有上述 Apple M2/Metal 证据；P50/P95/P99 性能、显存峰值、多厂商/后端与 renderer 端到端集成仍未验收。

这些边界不是占位 API；每一项必须以可运行算法、明确契约、单测和真实调用路径交付后才从本节移除。

## 不变量

- world 不复制 scene 内容；可变权威状态是 override delta，派生索引/代理/SDF/PCG 结果可删除重建。
- policy core 保持 `no_std + alloc`、`forbid(unsafe_code)`；文件系统和 GPU 能力只在明确 feature/provider 边界出现。
- 所有日志、排序、PCG、HLOD 和 readback contract 都有稳定顺序或稳定 key。
- 任何声称 GPU 或性能完成的结论必须同时记录 adapter、pipeline、dispatch/readback、P50/P95/P99、上传字节和显存/资源预算。
