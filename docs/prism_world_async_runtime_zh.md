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

本批没有执行 GPU、冷盘、mmap、资产包解压或加密网络验收。`prism_tasks` 的调度/共享取消整合仍需独立实现，不能把标准 Future/waker 兼容性当成已经接入任务系统。
