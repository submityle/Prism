# Timeline 异步文件存储

实现范围为 `pkg/prism_editor_timeline/src/storage.rs` 及其文件回归测试。存储服务不拥有编辑器，不自动替换当前时间轴，也不自动标记已保存。没有新增依赖或 Bevy 依赖。2026-10-21 已接入 `prism_editor_timeline` 根模块，使用真实 TimelineDocument codec；app 的文件会话接线另行验收。

## 宿主 API 与资源归属

- `TimelineStorageService::new(absolute_path, limits)`：为一个文件启动一个串行 worker。路径必须绝对、无父目录跳转；父目录须已存在且可信、稳定。构造、请求提交、poll 不做调用线程文件 IO。
- `load(timeout)`：返回 `TimelineStorageRequest`。成功结果为 `Loaded(TimelineLoadCandidate { document, revision })`，document 是真实 codec 解码候选。
- `save(bytes, snapshot_token, expected_revision, timeout)`：接收拥有的 Vec 和编辑器快照身份，worker 先验证实际 document bytes，再有条件保存。成功返回 `Saved(TimelineSaveReceipt)`。
- request 的 `id()` 是服务内单调身份；`poll(&mut self)` 一次取走终态，再次 poll 为 `Consumed`；`cancel()` 返回 `Cancelled` 或 `TooLate`。

宿主保存提交时记录当前 token；收到 receipt 后，只有 token 仍等于当前编辑器 token 才能确认相应快照保存状态。旧保存成功仍更新磁盘修订，但不能把后来编辑标记为已保存。加载结果由宿主按当前项目、服务代次、请求 ID 和未保存编辑策略明确采纳；ID 不能跨不同 service 比较。候选的 ticks_per_second 也由宿主决定如何应用。

已集成的完整根模块清单（无 Cargo 变更）：

```rust
mod storage;
pub use storage::{
    TimelineCancelOutcome, TimelineFileRevision, TimelineLoadCandidate,
    TimelineSaveDurability, TimelineSaveReceipt, TimelineStorageError,
    TimelineStorageLimits, TimelineStorageRequest, TimelineStorageResult,
    TimelineStorageService, MAX_TIMELINE_STORAGE_BYTES,
    MAX_TIMELINE_STORAGE_REQUESTS,
};
```

## 输入、输出与生命周期边界

`TimelineStorageLimits` 默认最多 2 个请求、每份序列化数据 8 MiB，硬上限分别为 8 和 64 MiB。请求额度合计运行中、队列中和仍由 handle 保留的输出；即使已 poll，仍需 drop handle 释放额度。取消中的 worker 仍持有其额度，避免无法中止的旧 IO 积累无界工作。队列使用有界通道和非阻塞提交，超额返回 Busy。

保存同时检查 Vec 长度与容量、expected revision 字节长度。读取检查 metadata 长度并分块执行流式上限检查，稀疏文件也受限。真实 codec 同时应用 TimelineLimits 的轨道、关键帧、维度、事件和文本限制；解码内存还包含这些受限结构，不能将序列化字节限额等同于精确进程内存上限。宿主取走的结果和自己保留的克隆由宿主负责释放。

超时使用单调时钟，从提交开始计算，包含排队。发布前取消或过期立即围住该请求的结果；worker 在分块 IO、解码完成及发布前检查，随后返回的旧结果不会覆盖取消/超时终态。不能强制中断已进入内核的 read/fsync，取消不会阻塞 UI 等待线程退出。service drop 围住未发布工作并关闭队列，不在调用线程 join。

最终比较通过后，worker 在互斥状态下跨入 Publishing，随后释放互斥锁执行 rename。自此取消为 TooLate；超时和 service drop 不能把真实磁盘发布隐藏成未保存失败。成功 receipt 的 deadline_exceeded 表示完成时已超期。跨围栏的 OS 调用可能继续等待，宿主应继续保留并 poll handle 取得实际结局。

## 修订、原子发布与错误语义

`TimelineFileRevision::Present(Arc<[u8]>)` 持有并比较完整文件字节；Missing 与现存空文件不同。不会使用 mtime、长度或非抗碰撞摘要代替修订。保存开始和 stage fsync 后各读一次目标，比对 expected revision；发现改变或删除返回 Conflict，保留外部文件。坏格式加载返回 Codec；保存坏格式也返回 Codec 并保留原目标。

每次操作由 worker 解析父目录，再持有固定 sidecar 的非阻塞 OS 文件锁：`<filename>.prism-timeline.lock`。相同目标的合作 service/process 通过该锁串行，锁忙返回 Busy。sidecar 保留在磁盘；不能在并发使用时删除它，否则不同 inode 的锁会失去协调。目标和锁的符号链接/非普通文件被拒绝；这不是针对恶意、并发替换目录树的沙箱。

发布过程为同目录 create_new stage → 分块写入 → stage sync_all → 再比较目标完整字节 → 发布围栏 → rename → 父目录 sync_all。取消、冲突和发布前失败通过 RAII 清理本次 stage，避免破坏目标。正常读者在同文件系统 rename 语义下只看见完整旧版或新版。

- 发布前 IO/codec/取消/超时错误：本次操作没有替换目标。
- rename 返回错误或发布期间 worker 异常：PublicationUncertain，宿主须重新加载核对，不能假设未发布并盲目重试。
- rename 成功、父目录同步失败：仍返回 Saved，durability 为 Uncertain，明确表示字节已发布但持久性未获确认。
- 两次同步都成功：durability 为 Synced，表示 OS 已确认相应同步调用成功。

sidecar 只协调遵守该协议的写入者。不合作的外部写入仍可能发生于最终比较与 rename 之间；此实现不能承诺对任意外部写入者的原子 compare-and-swap。加载时原地写入的外部程序也不受该锁保护。崩溃可能留下 stage，服务不会扫描或删除来历不明的遗留文件。宿主若要求更强跨进程事务语义，应统一写入协议或使用支持条件提交的存储设施。

## 实际验证

18 项 storage 测试使用真实临时目录、文件、OS 文件锁和真实 TimelineDocument codec：11 项集成测试覆盖往返、Unicode/cubic/极值 ID、同尺寸同 mtime 冲突、删除修订、并发 worker、请求及字节上限、坏格式/伪造计数、旧快照、独立子进程、原子读观察、符号链接与路径拒绝。7 项内部测试在真实 stage fsync、发布围栏、成功 rename 后设置仅测试构建存在的同步点，验证取消、真实超时、service drop、stage 后冲突、实际 rename 失败及实际父目录同步打开失败。没有生产 mock 文件系统或伪造成功结果。

最初使用独立且无新依赖的 `prism_timeline_storage_checks` harness：直接编译本仓库 storage.rs 与 tests/storage.rs，codec 和编辑器 token 来自实际 prism_editor_timeline 路径依赖；不复制或替代 codec。严格 Clippy 使用同一 harness。随后主仓库实际根模块集成后的 `cargo test --offline -p prism_editor_timeline -p prism_editor_ui --all-targets`（session 70535）同样通过全部 18 项 storage 回归，时间轴合计 55 项；这不代表 native app 接线已完成。

2026-10-21 再次直接验证真实 checkout：`cargo test --offline -p prism_editor_timeline -p prism_editor_world --all-targets`（session 17937）exit 0，时间轴 91 项、世界 97 项全部通过，包括通用 document worker、完整 sequence 项目存储及 world membership 的真实文件回归。共享根模块同时公开 sequence 作者/文档/存储 API，world 复用同一个 document worker，并用 `DocumentCodec { domain, message }` 保留实际域错误。随后 session 46011 的 profiler/timeline/world 联合 Clippy（all-targets、all-features、no-deps、`-D warnings`）exit 0；依赖包既有警告不属于此 lint 范围。

隔离 target 由本任务单独拥有；验证结束只清理本任务 target，不清理主线程或其他任务缓存。app 已有 Timeline/Graph/Sequence 项目路径、异步轮询、精确候选和保存回执接线，实际验证见 `prism_editor_native_panels_zh.md`；World membership 原生 UI 集成另行验收。
