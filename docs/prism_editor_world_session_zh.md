# World cell IO session：已实现边界与宿主契约

`WorldProjectSession` 是 **detached / no-runtime-consumers** 的 native cell 文件 IO 宿主：使用真实 `FileCellIoProvider` 和 `WorldIoDriver`，消费 `WorldPanelController` 的 Load / Save / Unload 队列。它不是完整 World runtime，也不表示主程序已经接线。

本切片仅包含 `src/world_session.rs`、`tests/world_session.rs` 和本文；模块声明、root 导出、主程序及 UI 接线由主线完成。未修改已有场景 session。

## 接线接口

主线需声明 `mod world_session;`，并从 crate root 导出：

```rust
pub use world_session::{
    MAX_WORLD_SESSION_WORK_PER_POLL, WorldProjectSession,
    WorldProjectSessionFailure, WorldProjectSessionLimits, WorldProjectSessionWork,
};
```

构造与轮询：

```rust
WorldProjectSession::new_detached(root, &controller, limits, now)
    -> Result<WorldProjectSession, String>

session.poll(&mut controller, now) -> Result<DesktopPanelUpdate, String>
session.rebind_detached(&controller) -> Result<(), String>
```

`root` 是已经存在的 World 目录，不是 `cells` 子目录，也不是 membership 文件路径。构造会同步 canonicalize，因此须在帧循环外完成；provider 在其下使用既有 `cells/x_y_z.prismcell` 布局和 envelope、身份、校验及版本基线规则。session 不解析应用 payload。

`DesktopWorldPanel::new` 或 `with_project` 的 provider closure 可以持有 session。例如在创建 controller/session 后：

```rust
let shared = std::rc::Rc::new(std::cell::RefCell::new(session));
let provider = shared.clone();
let panel = DesktopWorldPanel::new(controller, move |controller, now| {
    provider.borrow_mut().poll(controller, now)
});
// 宿主保留 shared 和 panel，错误后可检查票据、修复原因，再 panel.retry()。
```

membership 文件的加载、保存、candidate adoption 仍是 controller/project-storage 的独立职责；`with_project` 可同时绑定该服务。cell 保存成功不会替 membership 清 dirty。session 返回自己的在途/失败/排队及 project 围栏；完整 cell dirty/residency 围栏由 `DesktopWorldPanel::fenced()` 合并，不能只用 `is_busy()` 判断是否可退出。

## 有界工作与时钟

| 配置 | 默认 | 准入约束 |
| --- | --- | --- |
| `max_cell_bytes` | 8 MiB | 非零，满足现有 provider envelope 上限 |
| `max_cells` | 4096 | 1..=65536，绑定项目的 cell 数不得超出 |
| `work_per_poll` | 4 | 1..=32，单次 dequeue 与 terminal delivery 共用预算 |
| `poll_interval` | 10 ms | 大于零且不超过 1 秒 |
| `operation_timeout` | 10 秒 | 大于零且不超过 3600 秒 |
| `clock_tick` | 1 秒 | 大于零且不超过 60 秒 |

每个 session 同时至多一个真实请求/provider worker。开始请求的那次 poll 不立即交付其结果；未准入队列仍留在 controller。等待异步结果时返回 `wake_at`，不 sleep、join 或循环等待 worker。worker 的短暂清理期也不提前创建新 residency transition。

session 独占 controller 的单调 lease 时钟：以构造时 `Instant` 为零，elapsed / `clock_tick` 转为 tick。适用于尚未由其他宿主推进时钟的 controller；不能把一个已经使用非零 tick 的 controller 无说明地移到新零点。倒退、tick 溢出、deadline/wake 溢出均在 dequeue 前拒绝。

`operation_timeout` 是**观察 deadline**，不是硬终止保证。达到后记一次 `deadlines_exceeded`，仍保留真实请求并继续安排唤醒；`deadline_exceeded()` 可查询当前请求。不会因为超时伪造 Cancelled/Saved。底层 future 的取消/Drop 会取得 publication mutex，可能等待已经开始的文件发布；因此本宿主不在 deadline 路径调用 cancel 或丢弃 future。阻塞的操作系统文件 IO 仍可能迟迟不返回，CI/进程级硬超时应由外部宿主实施，并不能把它记为保存成功。

有界不等于实时常数耗时：residency planning 会扫描准入 World；controller 的其他 metadata 限制继续有效。Save 有有界 payload snapshot 和提交副本；Load reconciliation 为保留原始终态，最多再复制一份有界 payload。不宣称零分配、零拷贝或整帧严格时间预算。

## 终态、失败与 dirty

- Load 先取得真实 `WorldPanelTransition`，直到实际 completion 才通过 controller 安装 bytes/revision；失败交付 controller 回滚。启动失败尝试真实 `fail_residency`，回滚失败也保留 transition。
- Save 在复制前检查 session byte budget，保留 `WorldPanelSave` 的 bytes/revision/lease 快照；只有真实 Saved 且 controller 校验通过才清 dirty。
- 文件错误、保存期间新编辑、lease 过期都不会把当前 dirty 数据标为 clean。即使文件已经发布，旧快照不再匹配时也保留 Saved 证据；底层 controller 记录实际 durable revision，供明确重试使用。
- `failure()` 保留 intent、可用的 transition/save、真实 completion（provider error 转成字符串）和 `reconciled`。该标志仅表示无需修复 transition 或 controller 接受结果/回滚成功，**不代表 durable save 成功**。例如真实失败 completion 被接受，仍然 dirty。
- failure 围栏阻止后续命令自动执行。`take_failure()` 是显式责任转交，不是“忽略错误”按钮；未 reconciled 的 transition 必须由原 controller 修复。不能直接丢掉它并假装单元已卸载或保存。
- `Err` 不丢正在执行的请求或待交付终态。`DesktopWorldPanel` 遇错后会暂停 provider，宿主须保留 session、检查/解决原因后 `retry()`；仅 `retry()` 不会清 session 自己的 failure。

`work()` 暴露真实宿主计数：`intents_taken`、`requests_submitted`、`completions_received`、`unloads_completed`、`deadlines_exceeded`。它们不统计 runtime 实体、GPU 或 OFPA 工作。

## 身份、adoption 与 retired_world

每次 poll 先检查 `retired_world` 和精确 `WorldProjectToken`，再碰时钟、队列或 driver。稳定 World ID 和相同 revision 并不足以授权：独立构造/解码的项目没有相同实例 fence。普通 policy/membership 编辑也改变项目 token，需要宿主检查后显式 `rebind_detached`。

rebind 要求相同稳定 World ID、无未处理 failure、无 retired world 且满足 cell 上限。它**不改写**已接收 intent/transition/save 内的 controller-generation ticket。即使显式 rebind 到同身份的另一个 controller，旧 IO 仍会被该 controller 拒绝，原 terminal receipt 留在 session；可重新绑定原 controller 交付，不会悄悄移植旧写入。

本模块绝不调用 `take_retired_world()`。项目替换/结构编辑后，拥有真实 runtime consumers 的宿主须先完成实际 detach，并负责 retired world 生命周期；session 没有依据代为确认。不要在 panel provider 错误路径丢 session/driver：未完成 worker 的 Drop 仍可能阻塞 publication，且票据会失去责任人。

## 明确未完成的 runtime / OFPA 范围

`new_detached` 和 `rebind_detached` 要求调用者保证没有 ECS、render、physics 等消费者引用这些 cell。Load 不创建实体或 attach renderer；Unload 只在该前提下完成 authoring bytes 卸载。接入真实消费者后必须设计可失败的 attach/detach 与回滚协议，不能继续沿用这个无消费者前提冒充生产集成。

CrossCell 需要应用定义两侧 opaque payload 的权威修改，无法由 bytes 自动推导。当前明确拒绝并保留完整 intent 供具备真实 transaction 能力的宿主接管，不静默吞票，不制造空 transaction。未接入 OFPA actor IO、runtime streaming、场景/GPU同步或完整 World 工程工作流。

## 测试与验证状态

`tests/world_session.rs` 使用实际临时目录/provider worker 覆盖：保存后独立 session 加载、detached unload、失败保 dirty 与明确重试、较旧 Saved 不清新编辑、lease 到期、缺失文件 Load 回滚、字节/工作预算、非法参数、时钟倒退/溢出、deadline 观察、同 ID/revision 的外来 controller、project token/retired 围栏、CrossCell 显式拒绝。

2026-10-22，主线导出模块后运行：

```sh
CARGO_TARGET_DIR=/private/tmp/prism-editor-world-wake-target \
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
cargo test --offline -j 1 -p prism_editor_app --features desktop --test world_session
```

14 项通过、0 failed/ignored。CrossCell 回归先通过实际 Load intent 和 residency transition 将两侧 cell 置为 Loaded，再验证宿主明确拒绝并保留跨 cell ticket；没有绕过控制器的驻留或 lease 前置条件。此结果只覆盖上述 detached 文件会话，不能作为 runtime/OFPA 或原生窗口验收证据。
