# World 原生 reactor 文件接线

日期：2026-10-15。此切片把现有 `prism_tasks::IoReactor`、`NamedThreads::Io` 接到世界归档流送，不创建第二套 reactor 或逐请求线程，不改变 scene 的作者态权威。当前真实平台 backend 为 Apple POSIX AIO；其他 Unix 平台构造返回 `Unsupported`，Windows 不导出本接口。不会以同步读取冒充原生 backend。

## 使用与所有权

启用 `prism_world_runtime/reactor`（同时启用 `tasks`、`std` 和 `prism_world_io/reactor`）。宿主创建并保有 `Arc<IoReactor>` 与 `NamedThreads`，将它们的引用传给 `ReactorCellIo::reactor(path, &reactor, &workers, native_config, active, capacity, byte_limits)`。工厂返回 `RoutedCellIo<ScheduledCellIo<QueuedReactorFileProvider>>`；先注册完整 `CellReadRequest` 的 manifest、offset、checksum，再用 `CellFrameCoordinator::with_io` 或 `TaskCellIo` 驱动。源更新只影响后续提交。

构造、start、poll、cancel、operation/provider drop 不打开/关闭文件、不等待原生完成、不创建或 join 线程；这些路径使用短 metadata 锁和现有 lane 的 dispatch，不能视为 lock-free 或硬实时。文件打开、native read、关闭只在已有 IO worker 上发生。**IO worker 在等待 native 完成时会 park**，不等同于无阻塞 reactor continuation；compute worker 仅 poll。宿主必须为 IO lane 保留进展能力，不能把阻塞等待这些结果的消费者也放进同一已饱和 IO lane。

provider/operation 借用服务 owner，不可能自行析构最后一个 `NamedThreads`；派发 job 持有 reactor clone，但不持有 workers owner。宿主在非 compute 线程完成服务 teardown，先排空/析构 workers，再释放最终 reactor。计算任务可持有宿主服务 bundle 的 `Arc`，但宿主必须保留最终 owner，以免在计算线程发生 joining teardown。

## 原生生命周期与取消

`IoReactor` 接受 raw fd，拥有 buffer 但不拥有文件。直接 drop 它的 read future 不保证取消已提交的 OS 操作。本 provider 将 File、read future 与内部 thread-unpark waker 一并限制在 IO worker；`PendingRead` unwind guard 在读取完成并 reap 前不允许文件关闭，包括取消和 worker turn panic。宿主 waker 不交给 reactor。

每个 stream 最多一个 read turn 排队/运行。返回 chunk 后保留 File，直到调用者下一次 poll ACK 才提交下一块，不预读；最后一块或错误在 worker 关闭 File 后发布。取消可立即报告，但物理槽位及声明字节配额要等 native reap、File 与结果析构完成后释放。已发布的终态如果未消费也继续占槽。交给调用者的 chunk 已脱离 provider 配额。

唤醒 clone 位于锁前，替换后的 drop 和 wake 位于锁外。单个宿主 wake panic 被隔离，其他等待者仍被通知；原生 reactor 只收到内部 unpark waker。range 溢出、signed fd offset 超界、单请求/累计声明字节超限在读取前拒绝；零长度不打开文件。checksum、连续 chunk 和 short-read 校验仍由现有 scheduler 执行，完整 manifest/header/payload 校验后才进入 `SceneHost`。

## 有界准入

| 对象 | 约束 |
|---|---|
| 物理流槽位 | `ReactorFileConfig.capacity`，含 Ready、ACK 和取消清理 |
| 排队准入记录 | 同一个 provider 全部 queued clone 合计 `capacity`，含未 poll 的 start 和未消费 grant |
| 未结束 operation | 最多 `2 × capacity`；排队记录不持有文件或数据 buffer |
| provider 自有 payload | 最多 `capacity × max_chunk_size` 逻辑字节 |
| native 声明字节 | `max_request_bytes` 和 `max_total_bytes`，取消清理期间仍计费 |
| scheduler 聚合 | 独立 `IoByteLimits`，含 queued/active/未消费终态和 cell header |
| 线程 | 宿主预配置固定线程数量，无逐请求创建 |

这些是逻辑配额，不是 allocator/RSS 上限；线程栈、metadata、容器容量取整和调用者已取走内容另计。共享 reactor 的其他用户须由宿主另外限流。单次 scheduled poll 仍最多处理 16 个 chunk 事件，不承诺墙钟延迟。

queued provider 按首次 poll 注册顺序 FIFO 等待，**同时**满足空槽和声明字节才授予；在锁内预占资源，然后锁外唤醒。获 grant 后直到 operation 再次 poll 才派发文件任务，grant 被取消/drop 可同步释放。取消队首可让后续多条请求获 grant；大请求造成字节配额 head-of-line blocking 是明确的 FIFO 策略，不通过绕行、定时器或自唤醒重试规避。大于任一字节上限的请求直接失败，不永远等待。

直接 `ReactorFileProvider::start` 保留立即背压 API，与 queued clone 共享物理配额，但可能越过尚未 grant 的请求。`ReactorCellIo` 使用独占 provider，要求 `0 < active <= native.capacity`，因此能在被取消的旧 native 流尚未清理时接收替换请求。多个 scheduler 混用同一 queued provider 需按总准入记录配置容量；超额返回显式 Backpressure。

## 已运行验证

本机 Apple M2、macOS 原生 backend，IO 63 项、启用 reactor 的 runtime 67 项测试通过，涵盖真实非零 archive offset、多块/短读/checksum、打开失败、取消前后 fd 生命周期、unwind drain、并行共享服务、FIFO 槽/字节 grant 和重复取消替换。新增 latest-waker、wake/drop 锁外重入及连续 wake panic 隔离回归，原生 provider 的 21 项测试另重复 50 轮通过。runtime 在 0/1/2 compute worker 上验证有效内容进入 host、错误内容被拒绝、trace replay、父 scope 取消与 32 次替换；provider 单槽循环覆盖 64 次取消替换。任务池公平性另见 `prism_world_async_runtime_zh.md`。

性能入口复用同一个基准，避免复制归档和宿主逻辑：

```text
cargo test --offline -p prism_world_io -p prism_world_runtime --features prism_world_runtime/reactor -j1
cargo clippy --offline -p prism_world_io -p prism_world_runtime --features prism_world_runtime/reactor --all-targets -j1 -- -D warnings
cargo bench --offline -p prism_world_runtime --features reactor --bench threaded_archive -j1 -- --reactor
cargo bench --offline -p prism_world_runtime --features reactor --bench threaded_archive -j1
```

2026-10-15 同机 release：2 MiB、32 × 64 KiB cell、2 IO worker、4 active、16 KiB chunk、8 预热、64 样本、nearest-rank 分位数。创建归档/服务、构造 provider 和源注册均在计时外；stream plan、checksum/header、host payload 扫描及唤醒等待在端到端计时内。每次 poll 独立计时，包含校验及 host。native 的服务跨样本共享；threaded 在各样本计时前创建独占 worker，二者均不计服务创建/teardown 成本。

| backend | 端到端 P50/P95/P99（ms） | P50 对应 MiB/s | poll P50/P95/P99（µs） | poll 样本 |
|---|---:|---:|---|---:|
| native reactor | 4.413209 / 4.510916 / 4.525250 | 453.18 | 20.667 / 103.792 / 116.000 | 6,021 |
| threaded | 4.563000 / 4.642125 / 4.985500 | 438.31 | 20.209 / 126.958 / 133.375 | 5,909 |

归档可被 OS cache 命中，工作区存在其他并行任务；不是冷盘、磁盘峰值、无干扰 A/B、GPU timestamp 或产品帧时测试，不从两行推导普遍加速比。当前 native worker 逐 turn 提交，不宣称批量提交吞吐已经最优。

首次原仓库测试 IO 60/runtime 67 已通过。随后共享工作区新建 `tools/ecs-perf` 的非 Cargo 目录被根 `tools/*` 匹配，Cargo 在成员解析阶段失败；未修改该外部改动。后续回归、benchmark 与 all-target clippy 用独立 `/private/tmp/prism-world-native-yHWe0a/Cargo.toml` 引用真实 path 依赖，unit target 直接指向原始 lib.rs，无复制实现；独占编译 target 为 `/private/tmp/prism-world-async-VQ6A2p`，关闭 incremental/debug info。独立 harness 原生测试 IO 63/runtime 67、std-only IO 42/runtime 53、no-default IO 0/runtime 6 通过；原生、std-only、no-default 的 all-target clippy 均通过。不把这些结果扩展为根 workspace 通过。

## 边界

已完成原生 reactor → 有界 scheduler → manifest/header/checksum → opaque `SceneHost` 和任务 scope 的接线。仍不代表资产包解压、mmap schema 视图、GPU 直传、动态优先级继承、完整 scene/ECS/GPU 产品管线或跨平台后端已经实现。同步 `FileProvider` 兼容路径仍独立存在；此次没有改变它的阻塞语义。
