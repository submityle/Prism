# World 与任务池共享动态优先级

日期：2026-10-15。

`prism_world_io/tasks` 与 `prism_world_runtime/tasks` 将同一个 `Arc<prism_tasks::PriorityCell>` 从任务 Future 传到 cell 路由、scheduler 和原生文件读取。没有第二份优先级状态；`CellReadRequest` 仍是可复制的内容元数据，不增加任务所有权。默认和 no-default 构建不启用此能力；单独启用 IO 的 `tasks` 不强制开启任务池多线程，`reactor` 才开启已有多线程 IO lane。

## 接线

- `TaskPool::spawn_future_with_priority(cell, future)`：初次提交、外部 wake、32-poll 量子 continuation 均保留同一 cell。普通 `spawn_async` 不变，`spawn_future` 是普通路径的同义入口。
- `PriorityGroup::priority_cell()`：获取组内既有共享 cell，后续继承提升会同时被 Future 和 IO 观察。获取 cell 不会把这些请求或 Future 加进组的完成 counter；Future 仍有自己的 `Task::counter`。
- `ReadRequest::with_shared_priority(cell)`：低层提交入口；原有 `with_priority(u32)` 保留静态流送分数。
- `PriorityCellIo::submit_with_priority(request, cell)`：支持此契约的 backend 必须保留原 Arc，不能只复制当前值。不支持的 backend 不实现这个 trait，不会静默退回静态优先级。
- `PrioritizedCellIo::new(io, cell)`：将普通 `CellIo::submit` 转为共享提交，可直接交给 `CellFrameCoordinator::with_io`。`RoutedCellIo`、`ScheduledCellIo` 和 `TaskCellIo` 均透传同一个 Arc；路由仍检查完整 manifest、恢复注册的 offset/checksum，并保留调用方静态分数。

典型组合为 `PrioritizedCellIo<TaskCellIo<ReactorCellIo>>` 或 threaded backend。宿主从共享 cell clone 一份交给驱动该协调器的 `spawn_future_with_priority`，需求升级时调用 `cell.boost_to(Priority::Critical)`，不需要取消、重发或分配新请求 ID。原生 backend 仍借用宿主服务，不改变其生命周期要求，详见 `prism_world_native_io_zh.md`。

## 排序与进展契约

IO 准入按以下键降序选择：实时 lane、静态 `u32` 分数、较早请求 ID。lane 为 `Background < Low < Normal < High < Critical`；无共享 cell 的 IO 请求属于 `Normal`。静态分数不与 lane 做数值 max，`u32::MAX` 的预取也不能压过已提升为 High 的请求。同 lane、同分数按提交顺序 FIFO。

静态请求继续走 BinaryHeap；动态请求使用受 scheduler capacity 限制的数组，每次准入扫描其当前原子值，避免把可变优先级当作 heap 的不可变排序键。单次选择为 `O(S + log H)`，S 为排队共享请求数、H 为静态 heap 大小；持续排空全部动态请求最坏 `O(S²)`。活动请求仍用原来的 round-robin，不随 lane 改变 chunk 公平性。队列比较对每个候选采样一次；并发 boost 不是全局原子快照，可能在下一次选择才被观察。

任务池的显式 priority inbox 与普通 FIFO 是不同队列：显式 Background Future 也先于普通 Future/job。零 worker 的显式 Future ready queue 同样实时扫描并在同 lane 保留 FIFO；普通 Future 保持既有 FIFO。不能把 IO 的“普通请求属于 Normal”推广为任务池的全局统一排序。单个 poll 不会被抢占，持续高优工作仍可能饿死低优或普通工作；32 次量子只限制 dispatch 内 poll 数，不是跨优先级延迟承诺。

`boost_to` 单调提升，无降级、不安装 listener、不产生独立 wake。已提交任务/IO 自身的入队、依赖完成或资源 grant 负责唤醒；boost 不能代替 provider 的进展通知，也不能让不返回的 poll 让出。它只影响未来的准入/dispatch，不抢占已经运行的 poll、活动读取或已提交的 native read。

原生 provider 每个 ACK 后的新 read turn 重新采样：Background/Low → `IoPriority::Low`，Normal → Normal，High/Critical → High。kernel hint 是 best effort，不保证 OS 执行顺序。取消清理期间的物理槽/字节等待仍按首次 poll FIFO grant，NamedThreads IO 派发仍为 FIFO，不声称端到端所有队列已经统一优先级。

## 生命周期与兼容性

共享提交复用原有 mutex 准入、字节检查、单调 ID、失败关联和锁外唤醒。queued、active、未消费终态继续占 quota；scope 取消仍逐 poll 有界发出取消、抑制迟到 payload，并等待底层终态退休。Arc 不改变 fd/native future 的 worker 内 reap/close 顺序。共享优先级不会绕过 stale manifest 或 checksum/header 校验。

`ReadRequest` 的相等性比较固定元数据和 Arc 身份，不比较可变 lane；boost 不改变既有请求的相等关系。未启用 tasks 时 `with_chunk_size` 保留 const；启用 tasks 后拒绝参数需要析构 Arc，因此该 builder 是非 const。其他默认行为不变。

## 复现入口

2026-10-15 源码验证：IO/runtime 的 reactor 组合为 70/73 项，tasks-only 为 48/69 项，std-only 为 42/53 项，no-default 为 0/6 项，全部通过；四种组合的 all-target clippy 均通过 `-D warnings`。新增回归检查完整 manifest 拒绝、原 Arc 身份、late boost/静态分数/FIFO、native ACK 后 hint 变化、scope 取消/声明字节配额，以及真实任务池到 coordinator/SceneHost 的回放链路。

任务池 default 为 286 项，`fibers,numa,trace,compat-bevy` 为 309 项，std/single 为 272 项；13 项优先级相关测试额外重复 10 轮通过。default 与 std/single 的 all-target clippy 和各 11 项真实包 doctest 通过。当前根 workspace 被并行创建但尚无 manifest 的 `tools/ecs-perf` 阻断，因此本轮使用工作区外独立 manifest，依赖指向真实 crate，并将实际 `src/lib.rs` 作为测试入口；没有复制实现或移动该目录。这是上述模块的验证，不是整个共享脏工作区的验证。

```text
cargo test --offline -p prism_tasks -j1 -- --test-threads=1
cargo test --offline -p prism_world_io -p prism_world_runtime --features prism_world_runtime/reactor -j1
cargo test --offline -p prism_world_io -p prism_world_runtime --no-default-features --features prism_world_runtime/tasks -j1
cargo bench --offline -p prism_world_io --features tasks --bench priority_admission -j1
```

`priority_admission` 测量固定排队深度下第一次 scheduler poll：零字节 provider、active=1、16 次预热、128 样本、nearest-rank P50/P95/P99。队列构造、提交、boost 与销毁在计时外；计时包含选择、start、空读取完成与活动记录处理。动态组最后一个请求在入队后提升，并断言其先完成。它不是磁盘吞吐、真实任务延迟、GPU 或产品帧时测试，不覆盖完整队列排空的二次复杂度。

2026-10-15 Apple M2/macOS release，工作区存在其他并行负载，单位 ns：

| 排队数 | 静态 P50/P95/P99 | 全共享 P50/P95/P99 |
|---|---:|---:|
| 64 | 83 / 125 / 125 | 209 / 250 / 292 |
| 256 | 83 / 125 / 125 | 625 / 708 / 750 |
| 1024 | 125 / 125 / 375 | 2334 / 4209 / 21792 |
| 4096 | 458 / 1083 / 2000 | 15625 / 19167 / 59125 |

纳秒级小样本受计时器粒度、缓存和并行负载影响；保留尾部分位数，不推导普适倍数。运行时须按实际最大排队深度配置 capacity，不能以“小请求通常很快”代替最坏扫描成本预算。
