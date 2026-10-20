# 世界分区 Loom 控制器与视图

本切片提供 `prism_editor_ui` 的真实 world 控制器、Loom Element 子树和交互测试。控制器持有 `EditorWorldData`，直接更新选择、数据层可见性/可编辑性、租约和 opaque cell bytes；加载、卸载、保存和跨 cell 发布通过有界队列交给宿主执行。入队不代表加载完成或保存成功。

## main 集成

本切片仅写 `src/world.rs`、`src/world_view.rs`、`tests/world_panel.rs` 和本文。UI crate 的 `lib.rs`、`Cargo.toml` 由 main 集成。

`pkg/prism_editor_ui/src/lib.rs` 增加：

```rust
mod world;
mod world_view;
pub use world::{
    WorldGridPlane, WorldPanelAction, WorldPanelController, WorldPanelError,
    WorldPanelIntent, WorldPanelIntentKind, WorldPanelLease, WorldPanelLimits,
    WorldPanelViewport, MAX_WORLD_PANEL_CELLS, MAX_WORLD_PANEL_INTENTS,
    MAX_WORLD_PANEL_LAYERS, MAX_WORLD_PANEL_LEASES,
};
pub use world_view::{BoundWorldPanel, WorldCellVisual, WorldPanelViewStats};
```

`pkg/prism_editor_ui/Cargo.toml` 的 dependencies 增加：

```toml
prism_editor_world = { path = "../prism_editor_world" }
prism_asset = { path = "../prism_asset", default-features = false }
prism_ecs = { path = "../prism_ecs", default-features = false, features = ["std", "partition"] }
```

已有 dev dependency `futures-lite = "2"` 可直接用于新增测试。实现不引入 Bevy 或 wgpu 依赖，使用的 partition 类型来自 Prism 自有 ECS。

## 控制器与画面

`WorldPanelController::new(world, owner, viewport, limits)` 接收宿主配置好的真实模型；nil owner 被拒绝。模型的 cell、数据层 ID、actor catalog 和 membership 由宿主提供，没有项目类型或固定业务层。`world()` 提供只读观察，`into_world()` 可移交完整模型。

`build()` 返回 `BoundWorldPanel`。宿主将 `element()` 挂载到 Loom，通过 `handle_event(&view, target_key, &UiEvent)` 分派命中键；每个启用的 binding 都对应真实绘制节点。状态变更后重新 build 并 reconcile。旧 publication 或其他控制器的 view 返回 `StaleView`。纯 `dispatch()` 只解析动作；需要 publication 检查时应走控制器。网格为单选，Activate/Select 都替换当前选择。

视图包含实际绝对定位的网格盒、中心缩放、按 cell 平移、XY/XZ/YZ 平面及固定轴切片操作。盒子几何按 viewport 边界裁切；负数和小数 origin 使用 floor/ceil，右/下边界半开。`hit_test()` 拒绝非有限点和 viewport 外的点，返回网格坐标；`Select` 再检查该 cell 是否已注册。未注册槽位仍绘制背景，但不绑定选择。

背景显示 loaded/dirty/pending/unloaded/absent，选中色覆盖背景；边框显示本控制器持有、过期或未观察的 lease。选中详情保留精确状态、dirty、revision、lease 和队列数量。外部 lease 无枚举接口，因此显示 `Unobserved`，不能解释为 unlocked；实际 acquire/unload/写入仍走 world 的完整锁检查。网格内不放置可能越过边界的文字，坐标和图例位于网格之外。

选中 cell 的数据层按钮直接调用真实 `set_layer_visible` / `set_layer_editable`；这些是共享数据层策略，会影响所有属于该层的 cell。名称最多显示 96 个字符。超过可见层上限时明确显示省略提示；完整层列表/分页仍由宿主层编辑器负责。

## 预算与成本边界

| 资源 | 默认上限 | 硬上限 |
| --- | ---: | ---: |
| viewport 槽位，含空槽 | 512 | 1024 |
| 选中 cell 的数据层控件 | 16 | 64 |
| 待执行 intent | 16 | 32 |
| 控制器保留的 lease token | 32 | 128 |

尺寸范围为每轴 1..8192 logical pixels，cell pitch 为 8..256。构造和 viewport 更新先检查有限数、i32 坐标范围及槽位乘积，失败不发布部分几何。队列满返回 `Capacity`，不静默丢弃旧操作。lease 使用宿主单调 tick，默认 TTL 60，拒绝时钟倒退和 expiry 溢出；过期 token 需要显式 release 才释放保留容量，也可对同一 cell 重新 acquire。

每次 `build()` 只遍历可见槽位、选中层的有界前缀和固定工具栏。cell 查询为 BTreeMap 查找，成本为 O(V log N)，另加可见层查询和本地有界 lease 查找；没有每帧遍历全部已注册 cell。`WorldPanelViewStats` 给出实际槽位 lookup 和层控件数，不是性能 benchmark。20,000 cell 测试确认相同 viewport 的 lookup 次数与小世界相同。

显式执行边界不具有上述每帧成本保证：`begin_residency()` 调用现有全模型 residency planner，为避免单次点击误卸载其他 cell，先保留全部其他 resident/loading cell；数据层修改和 world 事务准备/恢复也沿用 domain 校验与扫描。它们不在 `build()` 内运行。UI 不声称整个交互链成本与世界大小无关，也不声称无分配。

## 宿主执行协议

1. `Load` / `Unload` / `Save` / `CrossCell` 生成带 controller identity 和 model epoch 的 `WorldPanelIntent`，通过 `take_intent()` FIFO 取出。丢弃尚未执行的 ticket 不改变 world。过期 epoch 或其他控制器 ticket 不能开始执行。ticket 可克隆，不声明为单次消费凭证；domain token、revision、CAS 和 receipt 提供执行阶段保护。
2. 加载先调用 `begin_residency(&ticket)`，将真实 transition token 构造为 world load request，交给 `WorldIoDriver`。结果经 `apply_load_completion()` 回写，包括失败/取消。`complete_load()` 仅适合宿主提供的不携带 durable revision 的 bytes；从磁盘加载应走 revision-aware completion。卸载在宿主确实释放消费者后调用 `complete_unload()`；失败或取消调用 `fail_residency()`。dirty、有有效 lease 或其他 domain 阻碍的 cell 不能被强行卸载。
3. 普通保存调用 `prepare_save()` 捕获实际 bytes/revision/lease，再提交给驱动；仅 `apply_save_completion()` 使用真实完成结果清理 dirty。期间有新编辑、lease 到期或其他 domain 冲突时，不把旧 snapshot 的持久化成功当成本地最新内容已保存。文件 provider 的取消、超时、请求/字节界限和存储错误通过原驱动传播。
4. 跨 cell 按钮先捕获 source，再选择 destination；两侧需要本控制器的 live lease。`prepare_transaction(ticket, id, snapshot, edits, provider)` 强制提供两侧的 opaque cell payload，拒绝范围外的 cell 和 actor 原归属；payload 由宿主定义，不推断或修改 membership schema。actor 必须已由宿主注册；移动前应已有当前归属的 durable actor。
5. 准备返回真正的 `PendingWorldTransaction`，提交其 request 给 `FileWorldTransactionProvider`，把真实 receipt 交给 `commit_transaction()`。准备阶段锁定 dirty；只有 domain 全部 revision、lease、catalog 和 residency 检查通过才一次安装本地结果。UI 不自行串联 rename，也不建立第二套 actor 格式。磁盘的 journal/root publication、CAS、恢复与取消 commit point 仍属于 world transaction provider；读取协调存储必须使用该 provider snapshot，不能混用独立 cell 路径作为同一事务视图。
6. 若 durable 发布成功但本地 receipt 因过期 lease 或新编辑被拒绝，保留 dirty 和旧 catalog。宿主重新取得完整 live lease 后，可明确调用 `discard_edits_and_restore_transaction_snapshot()`，丢弃所有本地 resident edits 并恢复完整 durable snapshot。该方法要求全部注册 cell 已 loaded、editable、leased，且 snapshot 精确覆盖；绝不会在失败回执后自动丢弃用户修改。

租约时钟、异步驱动轮询、取消句柄、host-side membership 编辑、错误呈现和原生事件命中接线由应用宿主负责。本切片交付可挂载控制器/view 和真实执行接缝，尚未修改 native app 或声明其窗口操作已验证。

## 本切片实际验证

采用最小隔离 workspace：引用仓库原始 `action.rs` / `support.rs` / 两个 world UI 文件及原始依赖源码，避免构建整个根 workspace 的 GPU dev dependencies。隔离 package 使用与上文一致的真实依赖和现有语义事件；仅对隔离 harness 未用到的旧 `support` helper 加 `allow(dead_code)`，不修改仓库 support 源码。

命令使用 `--offline`、单 job、dev/test debug=0、incremental=false：

```sh
cargo test --offline --manifest-path /private/tmp/prism-foundation-compat.qFtoL5/Cargo.toml -p prism_editor_ui --test world_panel
cargo clippy --offline --no-deps --manifest-path /private/tmp/prism-foundation-compat.qFtoL5/Cargo.toml -p prism_editor_ui --lib --test world_panel -- -D warnings
```

8 个测试覆盖真实 Loom mount/update/布局盒、所有可见控件 binding、缩放平移切片、dirty/层/lease 状态、外部锁、取消/失败 transition、非有限数/坐标/预算/旧 view/旧 ticket，以及临时文件上的普通保存、过期 snapshot 保留新编辑、真实双 cell transaction receipt 和 actor 跨 cell move。actor 测试还验证 durable move 后 expired lease 回执被拒绝，随后重新 acquire 并显式 snapshot restore 才安装新归属。文件测试在 provider outstanding operations 归零后删除自己的临时目录。

当前测试 8/8 通过（session `64350`），`--no-deps` strict clippy 通过（session `91665`）。未加 `--no-deps` 的首轮 clippy（session `98179`）在范围外 `prism_math/src/ops.rs:1` 的 `doc_markdown` lint 失败，未修改该依赖。此证据针对隔离 world UI 切片，不等同于全 UI crate、全 workspace、GPU 像素或原生窗口端到端通过；进程 kill/journal 恢复由既有 world 存储测试负责，本次 UI 测试没有新增 kill 注入。
