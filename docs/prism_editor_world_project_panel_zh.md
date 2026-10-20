# World project 原生面板与 controller

范围：`prism_editor_ui::WorldPanelController` 和 `BoundWorldPanel`。
使用实际 `EditorWorldData`、`WorldProject`、稀疏 `ProjectChange` 与已有
`WorldProjectStorageService`；没有第二个 IO worker、Bevy 或固定游戏数据类型。
app 的窗口挂载、文件路径选择与 runtime consumers 由宿主负责。

## 真实模型与可见行

controller 构造时按 `WorldProjectPanelLimits.project` 捕获一次完整 catalog。
后续 build 不再重捕获 world，也不创建 history 快照。
`project()` 是与实际 world 同步的 identity、layer、cell、actor membership。

选中 cell 后，`world:layer:{id}:member` 修改它的直接 membership；
visible/editable 控件修改实际层策略。每个命令先在有界临时 project 上完成
全量引用、预算和冲突验证，再更新实际 world 和发布 project。
对现有 topology 的策略和 membership 修改原地完成，保留 resident payload、
dirty revisions、锁、edit scope 和 HLOD。没有写入任何伪造 payload。

层列表按实际 project 的 ID 顺序分页，`WorldPanelLimits.visible_layers`
限制每页行数。`ProjectLayerPage(offset)`、Previous/Next 按钮提供真实页面切换；
每行提供 policy 和选中 cell 的 membership 控件。名称最多呈现 96 字符，
存储状态最多呈现 256 字符。grid 仍只查询 viewport 对应坐标，
`WorldPanelViewStats.visible_layers`、`has_more_layers` 报告实际页面。
分页前缀遍历受 project 的 max_layers 约束；不声称随机跳页的遍历成本仅等于页大小。

`BoundWorldPanel` 的 Arc publication identity 随状态变化更新。
另一 controller 或旧 build 的事件不会修改当前模型。界面未提供启用绑定的
控件不能通过 dispatch 伪造命令。命令 API 本身仍执行实际域验证。

## 稀疏历史及 OFPA 同步

`edit_project(&[ProjectEdit])` 使用后端 `ProjectChange` 的 touched-record
compare-before-restore，而不是保留全 project/world 副本。
`project_history()` 返回 undo 数、redo 数、两者合计 charged bytes。
entries/bytes 达上限则拒绝整个命令；失败不清 redo，也不部分改变 world。
新命令只在成功后清 redo，undo/redo 同样先验证 candidate 再移动 history。

`LayerVisible`、`LayerEditable`、`CellLayer` 自动进入该历史。
Undo/redo 会推进 project revision 并更新精确 token，即使值最终相同，
旧保存回执也不能清除当前 dirty。

结构命令（例如新增/删除 cell 或 layer、层重命名/改 parent）
采用实际 `WorldProject::materialize`。要求旧 world 静止，并保留旧 world
供宿主显式交接；没有静默舍弃运行态。局部命令直接操作 `Actor` 被拒绝，
actor 移动/删除必须通过实际 OFPA transaction。

`commit_transaction` 在 domain 提交前从真实 request 的 actor delete/put
构造并验证下一 project，只有真实 receipt 通过 domain 的 lease/revision
围栏后才发布。transaction snapshot 的显式恢复也在恢复前验证 catalog，
然后同步真实 actor owner。未确认或过期 receipt 保留原 project 和 dirty bytes。
不相关的 policy history 可以撤销，同时保留已经提交的 actor owner 变化。
项目文件与 OFPA root 是各自的持久化边界；没有宣称二者被一个跨文件原子事务提交。

## 实际 storage 生命周期

宿主绑定真实服务：

```rust,ignore
controller.bind_project_storage(service, observed_revision, timeout)?;
// observed_revision: Option<TimelineFileRevision>
// None 表示文件状态未知；必须 load/adopt 后才能 save。
// Missing 只能用于宿主明确确认的 create-only 操作。
```

绑定不标记 captured project clean。最多保留一个 request 和一个 candidate，
candidate 未决时拒绝再发 load/save。实际服务自身也执行队列/字节预算。
`ProjectSave` 编码当前 catalog，经共享 worker 条件发布。
`ProjectLoad` 只读入独立 candidate；加载结果再次通过面板自身的 project limits。

宿主每 tick（面板隐藏时也一样）调用 `poll_project_storage()`。
返回 true 表示消费了实际 terminal result，需要重建视图。
`project_io_state()` 保留实际 Failed/Cancelled/Saved/Candidate 状态。
Saved 包含 receipt 的 durability、deadline_exceeded 和是否匹配当前 snapshot。

保存时允许继续编辑。完成后，实际 file revision 始终按 receipt 更新，
只有 `WorldProject::matches_token(receipt.snapshot_token)` 才清当前 dirty。
此判断包含 Arc identity；不能用 GUID、revision 数字或字节相同替代。
PublicationUncertain 清除已知 file revision 和 clean proof；后续 save
被阻止，需重新 load 并采用以恢复基线。普通 CAS 冲突和 codec 错误不替换 live project。

`ProjectCancel` / `cancel_project_storage()` 调用真实 request.cancel()。
即使返回 Cancelled，request 也留到 poll 消费 terminal；返回 TooLate 时
继续消费真实 publication receipt，不把已经发布的文件显示成未发布或已取消。

## 候选安全采用与宿主交接

加载启动时保留当前 project token。采用要求 candidate 仍匹配这个 exact token；
加载后编辑再撤销到相同内容仍会被拒绝，需要 dismiss 并重新加载。
普通 `ProjectAdopt` 还要求 membership clean。
`ProjectAdoptDiscard` 是显式丢弃 membership 改动的操作，不豁免以下域保护：

- 所有 cell 必须 unloaded 且无 dirty；actor 不得 dirty。
- 没有尚未交给宿主的 intent，没有 live 本地 lease，没有待交接旧 world。
- 通过实际锁 capability API 对所有旧 cell 探测，拒绝接管前已存在的外部锁
  （包含相同 owner）。成功探测立即释放。此检查会推进旧 world 的锁序列、
  按当前时间清理 expired leases，不改变 payload、catalog 或 history。

materialize 和所有检查成功后，controller 更换 world/project，清 authoring history，
使用 loaded revision 建立新的 save 基线，并更换 controller generation。
原 world **保留**在 controller 内，所有新 IO 意图/完成被围栏，直至宿主处理：

```rust,ignore
// 先 detach 原 world 的 runtime consumers；这是宿主的真实责任。
if let Some(old_world) = controller.take_retired_world() {
    // 接管完整旧 world，按宿主规则释放 scope/HLOD/运行态等。
}
```

`has_retired_world()` 是此交接的明确状态。`into_world()` 消费 controller 前，
宿主同样应先接管 retired world。完整采用不是对 resident world 的在线热合并。

## Erdos/app API 适配与 main 根导出

UI Cargo 已依赖 `prism_editor_world` 和 `prism_editor_timeline`，无需新增依赖。
main 在既有 `pub use world::{...}` 增加：

```rust,ignore
WorldPanelSave, WorldPanelTransaction, WorldPanelTransition,
WorldProjectIoState, WorldProjectPanelLimits,
```

实际 cell IO 需要 exact controller generation：

```rust,ignore
let transition = controller.begin_residency(&intent)?;
let request = controller.world().load_request(transition.token())?;
// 将 request 提交真实 provider，保留 transition。
controller.apply_load_completion(&transition, actual_completion)?;
// 直接 host bytes completion / unload / fail 接受 owned transition。
```

`prepare_save()` 返回 `WorldPanelSave`，保留 `request()` / `payload()`；
completion 接受 `&WorldPanelSave`。`prepare_transaction()` 返回
`WorldPanelTransaction`，保留 `request()`；commit 接受这个有代际围栏的 wrapper。
域的裸 TransitionToken 在新 materialized world 可以复用相同数字，不能单独作为 UI IO 围栏。

app 退出/切换项目应把 `project_is_dirty()`、`project_io_pending()`、
`has_retired_world()` 和原有 cell/actor dirty/IO 条件共同纳入 fence。
文件路径选择、bind、tick poll、detach/retirement acknowledgment 在 app 所有权内，
本文不把 controller 测试等同于 app 已挂载完成。

## 验证入口

`tests/world_panel.rs` 保留真实 cell 文件/OFPA receipt、过期 lease、snapshot recovery 测试，
并验证恢复后 actor membership 与稀疏 policy history 同步。
`tests/world_project_panel.rs` 使用真实临时目录和 storage worker，覆盖
roundtrip、新编辑遇旧 receipt、CAS 冲突、非法文件、候选 exact fence、
取消/未知 revision、安全采用、跨代数字 token 复用、history 和可见行预算。

使用共享 target，不清理他人的或活跃 target：

```sh
CARGO_TARGET_DIR=/private/tmp/prism-editor-joint-cr1ODK CARGO_INCREMENTAL=0 \
CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 \
cargo test -p prism_editor_ui --test world_panel --test world_project_panel

CARGO_TARGET_DIR=/private/tmp/prism-editor-joint-cr1ODK CARGO_INCREMENTAL=0 \
CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 \
cargo clippy -p prism_editor_ui --all-targets --no-deps -- -D warnings
```
