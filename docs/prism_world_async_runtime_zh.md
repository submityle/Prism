# World 异步文件 runtime

日期：2026-10-15。实现范围为真实 archive 文件到内容中立 `SceneHost` 的加载；未引入第二份 scene 内容或持久状态。

## 接线

`ThreadedCellIo::threaded(path, workers, active, capacity, limits)` 创建固定 worker 的后端。先以 `register(CellReadRequest)` 注册每个 sector 的完整 manifest、archive offset 与可选 checksum，再交给 `CellFrameCoordinator::with_io`。`plan_frame` 提交需求；帧驱动调用 `poll_io_with_context(cx, host)`，遇到 `Pending` 等待 waker，遇到 `Ready` 表示消费一个终态。应用可以每帧消费有限个终态；禁止用无等待循环把 `Pending` 轮询成忙等。

`CellIo` 与路由、协调器保留 `no_std + alloc`；固定文件 worker 与聚合 adapter 位于 `std`。自定义 backend 可用 `RequestId::new(NonZeroU64)` 建立自己的身份，必须保证同一实例内永不复用。`IoRuntime`/`WorldFrameCoordinator` 保留同步兼容接口，provider 回调仍可能阻塞；异步工厂不会调用 `read_to_completion`。

路由在准入前核对完整 manifest，包括内容版本、derived hash、范围与 IO 长度；调用方不能用同 sector 的旧注册绕过版本校验。覆盖注册只影响后续提交，已接受请求保留原始 range/checksum。头部由协调器按最初提交的 manifest 校验，opaque payload 只在全部校验成功后进入 host。单请求失败、取消与内部后端失败具有不同返回/trace 语义。

## 有界行为

| 对象 | 边界 |
|---|---|
| 协调器接受而未消费终态的请求 | `capacity`，取消也不提前退还 |
| 活动 scheduler 请求 | `active`，工厂要求其非零且不超过 worker 物理槽位 |
| 物理文件槽位 | `workers.capacity`，取消清理结束前继续占用 |
| 容量等待记录 | 全部 provider clone 合计 `workers.capacity`，含尚未 poll 的准入与已获 grant |
| worker 数量 | `workers.workers`，无每请求线程 |
| provider 自有 payload | 最多 `workers.capacity × workers.max_chunk_size` 逻辑字节 |
| 聚合准入 | 单请求和累计声明字节由 `IoByteLimits` 限制，包含 44 字节 header |
| 单次 scheduled poll | 最多 16 个 chunk 事件，之后显式唤醒并让出 |

这些字节是逻辑载荷，不是 allocator/RSS 上限；线程栈、容器元数据、Vec 容量取整、已交还调用方的内容不在计数内。直接 adapter 保留 chunk 事件时可能形成第二份完整载荷；`ScheduledCellIo` 立即丢弃已聚合的 chunk 事件。host 的工作与析构在调用线程执行，其耗时必须独立受宿主控制。16-chunk 限额不是墙钟硬实时保证。

取消采用 operation 共享状态，已进入 OS 的阻塞读取只能合作终止；取消终态可以先被 scheduler 消费，但物理槽位要等 worker 销毁文件/结果后才释放。queued wrapper 此时等待 FIFO grant，既不把临时满槽当成永久 cell 失败，也不自旋重试。每次 grant 在锁内占住槽位、锁外唤醒；混用多个 scheduler 时需按总等待数配置容量，超额仍显式返回背压。

adapter 的 public poll 保留一个最新 Context 的 waker；并发 blocking reader 各自按请求 ID 注册，数量受请求 capacity 限制。waker clone、替换后的最终 drop 和 wake 均在 adapter/通知锁外执行；provider 在锁内同步唤醒时先记通知，等 adapter guard 解锁后再派发。重入产生的新注册不被旧 poll 清掉，派发期间的新通知继续交付，单个回调 panic 不阻断其他等待者。

## 已执行验证

独占 target：`/private/tmp/prism-world-async-VQ6A2p`，关闭 incremental 与 debug info，离线单 job 构建。IO 42 项、runtime 53 项测试通过；runtime no-default 的 6 项测试通过。两种 feature 配置的 all-target clippy 均通过 `-D warnings`，指定文件 rustfmt 和 `git diff --check` 通过。真实文件测试覆盖分块穿越 header、archive 非零 offset、checksum 与 stale header 拒绝、仅有效 opaque payload 进入 `SceneHost`；Pending、唤醒、公平性、配额保留、取消和 ID 耗尽另有确定性回归。11 项 adapter 唤醒回归覆盖最新 Context、重入 wake/drop、panic 隔离、锁内同步 poll/cancel 通知延迟、并发通知与多个 blocking reader。

性能入口为 `cargo bench --offline -p prism_world_runtime --bench threaded_archive -j 1`。文件共 2 MiB、32 个 64 KiB cell、2 worker、4 active、16 KiB chunk、32 请求 capacity；8 次预热，64 次测量，nearest-rank 分位数。输入与载荷均校验，host 扫描 payload；worker 创建和源注册在计时外，stream plan、checksum、header 校验、host 和唤醒等待在端到端计时内。

唤醒加固后的 release 本机记录：端到端 P50/P95/P99 为 6.626/10.308/12.405 ms，P50 对应 301.84 MiB/s；poll（含校验与 host）为 35.125/172.459/390.167 µs，共 4,929 次。文件可被 OS cache 命中，并行工作区有其他负载；不将此结果称为磁盘峰值吞吐、无 IO 卡顿达标或跨硬件承诺，也不将不同并行负载下的前后两次测量当作优化倍数证据。

本批没有执行 GPU、冷盘、mmap、资产包解压或加密网络验收。

## 任务池与结构化取消

启用 `prism_world_runtime/tasks` 后，`TaskCellIo::new(backend, &parent)` 取得一个空闲 backend 的独占所有权并创建 child token；非空闲 backend 原样返回。将 `next_cell_completion(&mut io)` 或协调器的 Context poll future 提交给 `prism_tasks::TaskPool::spawn_async`，即可在同一计算池中驱动加载。阻塞文件操作仍留在固定文件 worker 中。此 feature 不开启任务池的全局实例，也不替宿主选择线程数量。

`CancelToken::cancelled()` 为每个 Future 独立登记最新 waker，drop 时 O(1) 注销，槽位存储受峰值同时等待者数量约束。树使用 weak child 与 strong parent 路径：丢弃中间 token 不切断仍存活后代的取消关系，废弃子树可以回收。传播与深链释放采用迭代遍历；clone、drop、wake 回调都在树与 waiter 锁外执行。先标记全部后代，再逐个通知；单个回调 panic 不跳过其他通知，正常调用完成清理后重抛第一个 panic，正在 unwind 时保持原 panic。若被压制的 panic payload 析构本身再次 panic，仅遗留其二次 payload，以免二次展开导致 abort。

父 scope 取消会直接唤醒等待中的 IO task，即使 provider 没有通知。`TaskCellIo` 每次 poll 最多向 16 个已接受请求发出取消；还有剩余时主动唤醒并让出该 poll。取消后拒绝新提交，并将迟到的成功或单请求失败转为 `Cancelled`，不会把成功 payload 交给场景；backend 内部错误仍原样返回。配额直到消费终态才释放。drop 会逐个尝试取消全部剩余请求，即使通知或单次 backend cancel panic；标准 threaded backend 的 drop 不等待阻塞文件读取，自定义 backend 仍受自己的 drop 契约约束。

真实 archive 回归在 0/1/2 个计算 worker 上运行：3 个 cell、每个 512 字节 payload、1 个文件槽、7 字节 chunk，核对 archive offset、checksum、header 和 host payload。过程中修复零 worker fallback 的外部唤醒：ready queue 与直接 Future 的通知共同推进带 epoch 的 condvar，多个 `block_on` driver 各自观察通知，不再等待 5 秒 stall timer 才继续。定向测试覆盖提前通知、直接 Future、排队 harness 与并发 driver。

本轮启用 tasks 的 runtime 63 项通过；`prism_tasks` 默认配置及 `--no-default-features --features std` 的公平性收尾后计数为 273 项单测与 11 项文档测试（原 264 项，加已有公平性补丁 6 项及收尾新增 3 项）。runtime no-default 的 6 项通过；tasks/std 与 runtime/tasks 联合 all-target clippy、runtime no-default all-target clippy 均通过 `-D warnings`，指定文件格式检查与 `git diff --check` 通过。仍未接通共享动态优先级、原生 IO reactor、资产包解压、mmap 或 GPU 直传；取消集成不构成这些功能的验收。

## 任务公平性

`spawn_async` 的 harness 每次 scheduler dispatch 最多执行 32 次 `poll`。只有 `Pending` 且 poll 期间收到 wake 才继续消耗量子；未收到 wake 的 `Pending` 保存 Future 并进入 Idle，等待外部唤醒。第 32 次 poll 若仍为已通知的 `Pending`，在同一锁内保存 Future 并转为 Queued，再解锁入队；并发 wake 合并为一个 continuation，不重复入队、不丢失通知。`Ready` 或 panic 优先结束任务，即使最后一次 poll 同时发出 wake，也不会因量子耗尽而重新入队；完成后的旧 waker 不再触发 poll。

有 worker 时，harness 的首次提交、Idle 唤醒与量子 continuation 都进入全局 injector 的 FIFO 队尾，包括从该池 worker 发起的提交。这样，自唤醒任务不能借本地 LIFO 立即重新执行、越过已在 injector 中等待的普通计算任务或其他 async harness。FIFO 只保证该队列内的取出顺序：调度器仍优先处理 priority inbox、本地 deque，再取 injector，fibers 还会先恢复就绪 fiber；多 worker 的执行/完成顺序也不是全局 FIFO。持续高优先级或本地工作造成的跨队列饥饿不属于本补丁解决范围。

零 worker fallback 中，`block_on` 每批最多执行 32 个 ready harness dispatch，再 poll 一次直接传入的 root Future；ready queue 持续非空也不会把 root 饿死。每批最多为 `32 × 32 = 1024` 次 harness poll，root poll 不计入其中。其他 ready harness 通过 FIFO 插入连续自唤醒任务之间；普通同步 job 在零 worker 下仍由提交线程 inline 执行。root 返回 Pending 后，队列有工作就继续下一批，无工作时沿用通知/epoch 等待机制；多 worker 下 root 由调用线程 poll，Pending 时等待自身 waker，不承担 caller-side help。

以上都是合作式、以 poll/dispatch 次数计的边界，不是毫秒延迟或硬实时保证。执行器无法抢占单次 poll；不返回的 poll、阻塞调用、长时间计算、析构以及 root 自己的单次 poll 都可能阻塞所在 driver/worker。调用方仍需分段工作并主动让出；本补丁不引入抢占、动态优先级或 IO 容量等待策略。

公平性回归共 9 项：零 worker 下自唤醒与 root/计算进展，单 worker 下本地及 injector 普通任务的进展与 root 唤醒，worker 内首次 async 提交的 FIFO 顺序，两种 ready 队列的并发 wake 合并，量子末尾恰好 Ready 或无 wake 的 Pending，以及让出后 panic/旧 waker 隔离。顺序和 poll 次数通过显式握手、手动队列驱动及原子计数断言，不以时间片快慢判定公平性；超时仅保护测试握手，poll fuse 让饥饿回归以计数超限失败。

本次独立验证使用 `/private/tmp/prism-task-fairness-vGTao8`，离线、单构建 job，关闭 incremental/debug info。`cargo test --offline -p prism_tasks -j 1` 通过 273 单测与 11 doc；`--no-default-features --features std` 加 `-- --test-threads=1` 同样通过 273 + 11。std-only 首次并行全量跑在范围外的 `tests::m3::main_lane_runs_only_when_pumped` 挂起超过 60 秒，随后终止；不将那次运行记为通过，也未修改该测试或 named lane。定向 `fairness_tests` 在 `--no-default-features --features single` 下 6 项、`--features fibers` 下 9 项通过（实际强制单线程 feature 名为 `single`）。std-only `cargo clippy --offline -p prism_tasks --all-targets --no-default-features --features std -j 1 -- -D warnings` 通过；fibers all-target clippy 被未改动的 `fiber/stack.rs:92` 与 `fiber/mod.rs:102,241` 三处 `undocumented_unsafe_blocks` 拦住，不声称该配置 lint 通过。三个 Rust 文件的定向 rustfmt 检查与四个文件的 diff 空白检查通过。上文 runtime 63、no-default 6 及联合 clippy 是主代理已有验证，本次公平性收尾未重跑 runtime，也未修改 IO 接入。
