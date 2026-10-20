# Sequence 项目创作与真实异步文件

本切片拥有 sequence_document.rs、sequence_storage.rs、两个对应集成测试和本文；经主线程明确交接，也负责 storage.rs 的通用 worker 提取，保留此前已有的链式 lint 修改。lib.rs 仍由主线程集成；没有 Bevy、额外依赖、领域设备后端或第二套文件 worker。

## 数据与创作接口

SequenceProjectDocument 拥有稳定 SequenceId → 真实 TimelineDocument 的有序表、全局稳定 SequenceClipId → (parent, clip) 的完整拓扑、可选 root，以及两类 ID 高水位。默认构造为空项目；数据只读暴露，修改必须通过 SequenceProjectEditor。没有外部路径引用代替文档内容，完整文件无需另外加载子文档。

SequenceDocumentLimits 同时约束真实 SequenceRegistry、真实 TimelineDocument codec、整个项目字节、命令长度和逆历史。project.registry(limits) 借用项目中原始文档，调用现有 SequenceRegistry::insert/add_clip，复用真实嵌套整数时钟、采样和事件。项目/编辑器必须比借用 registry/player 活得久；编辑前结束借用。无 self-referential 结构或伪采样器。

SequenceProjectCommand { label, edits } 原子应用，支持 PutDocument/RemoveDocument、PutClip/RemoveClip、SetRoot、SetFrequency 和 Timeline { id, edit: TimelineEdit }。PutClip 可以移动/替换现有稳定 ID；可在同一事务中移除文档并重接引用，只要求最终状态有效。根不存在、来源缺失、重复次数无效、零频率、环、深度、真实映射溢出、文档 codec 或预算错误均拒绝。

编辑器按顺序修改并同时生成反序 inverse；失败逐条逆向还原。普通轨道/关键帧/事件编辑只保留实际受影响负载。替换或删除整个文档保留该文档，移除 clip 只保留 clip；从不为一个命令克隆整个项目。undo/redo 继续交换 inverse，重新检查最终项目，不制造全项目 snapshot。history_entries/history_bytes 同时计入两方向，成功分支清 redo、按预算淘汰最老 undo；失败保留文档、历史、revision、token 和高水位。

SequenceProjectToken 使用 Arc 指针身份；执行、undo、redo 成功均换 token。revision 只用于递增显示，不授权异步保存确认。保存开始后继续编辑时，旧 receipt.token 与 live token 不相等，宿主不能据旧 receipt 标记当前状态 clean。

next_sequence_id/next_clip_id 根据已采纳 ID 高水位建议新 ID；成功删除或 undo 不降低高水位，失败事务不消耗 ID。显式导入已有 ID 和稳定 ID 更新由宿主负责身份一致性；本 API 不把替换操作误认成分配。零 ID 合法；u128::MAX 高水位返回 Exhausted，不 wrap。高水位针对 sequence/clip 命名空间；子文档内部 TrackId/KeyId/event ID 由已有真实 Timeline API 管理。

## 文件协议和资源上限

PRSSEQ01 为 version 1，固定小端整数；不序列化 usize。随后为 sequence 高水位、clip 高水位和 root，各占一字节 present flag 加固定 16 字节 ID；absent 的 ID 字节必须全零。固定宽度防止 undo 保留高水位导致文件元数据突然增长。之后为 u64 文档数量，每项为 u128 SequenceId、u64 文档字节长度及真实 TimelineDocument 编码。最后是 u64 clip 数量，每 clip 依次存 id/parent/source 三个 u128，parent.start/end、source.start/end、约分后的正 rate 分子/分母、正 repetitions 七个 u64，共 104 字节。

文档按 SequenceId，clip 按全局 SequenceClipId 严格递增；解码拒绝重复、乱序、未约分 rate、非法 flag、额外尾字节及截断。读取数量先核对策略和剩余字节，再分配/解析。载入是隔离候选，只在完整图验证通过后返回；现存项目不会在 decode 失败时受影响。encode/decode 保留子文档浮点位模式、插值、切线、真实事件、时钟、拓扑和高水位；同一项目重编码逐字节一致。

整个项目和历史各硬限 64 MiB，单命令 edits/history entries 各 4096，label 4096 字节；零值无效。默认项目/历史各 16 MiB、1024 edits、256 entries、4096 label bytes。registry 与每文档另受 SequenceLimits/TimelineLimits 约束。拓扑构造借用每份真实文档，临时编码和检查工作有界；没有无限循环、无界项目扫描后台线程或隐式导入。

## 共享异步存储

SequenceProjectCodec 实现 DocumentStorageCodec，关联完整 SequenceProjectDocument 和 SequenceProjectToken。SequenceProjectStorageService 是共享 DocumentStorageService 的薄封装，new(path,max_requests,limits)、load(timeout)、save(owned_bytes,token,exact_expected_revision,timeout)；request 的 id/poll/cancel 与共享实现相同。构造和提交不做调用线程文件 IO。构造会验证空项目能满足文件策略，保存前 worker 再解码真实完整项目，损坏项目不会被写入。

DocumentStorageCodec: Send + Debug + 'static，关联 Document: Send + Debug + 'static、Token: Send + Clone + Debug + 'static；decode(&self, bytes, max_bytes) 运行于 worker。DocumentStorageService::with_codec(path, TimelineStorageLimits, codec) 保持相同队列、槽位和文件流程。TimelineStorageCodec 使用 limits.document；其他 codec 自带其领域限额，并服从 worker 的 max_bytes。TimelineStorageService/Request/Result/LoadCandidate/SaveReceipt 保留为该默认 codec 的别名，原构造器 new(path, limits) 和既有调用保持可用。

根导出由主线程维护：sequence_document 的 SequenceDocumentError、SequenceDocumentLimits、SequenceProjectCommand、SequenceProjectDocument、SequenceProjectEdit、SequenceProjectEditor、SequenceProjectToken；sequence_storage 的 SequenceProjectCodec、SequenceProjectLoadCandidate、SequenceProjectSaveReceipt、SequenceProjectStorageRequest、SequenceProjectStorageResult、SequenceProjectStorageService。若宿主需要直接复用通用 backend，再导出 storage 的 DocumentStorageCodec、TimelineStorageCodec、DocumentStorageService、DocumentStorageRequest、DocumentStorageResult、DocumentLoadCandidate、DocumentSaveReceipt。Cargo 无变更。

共享 worker 对 running/queued/未消费结果统一限额；结果消费/持有句柄、Busy、Consumed、Missing、UnsafePath、ByteLimit、SequenceCodec、IO、取消、超时和不可确定发布都有显式结果。CAS 比较完整字节，缺失文件不同于空文件。协作锁、stage 创建、分块写、文件 fsync、第二次字节检查、取消发布围栏、rename、目录 fsync 全部复用原 storage.rs；不会另开一个弱化版存储路径。

发布前 cancel/timeout 保证后来不能替换文件；进入不可撤回发布后 cancel 返回 TooLate，须看最终 receipt/PublicationUncertain。目录同步失败时文件已经发布，durability=Uncertain；不能向用户宣称确定失败并盲重试。CAS 的最终检查与 rename 之间仍要求外部写者使用同一个协作锁；不声称能排斥忽略协议的任意进程。

宿主显式采用 Loaded candidate，保存后显式比较 snapshot_token；没有自动清 dirty 或覆盖 live editor 的逻辑。项目路径、保存/加载按钮、领域 provider 与原生 UI 由宿主集成，本切片不修改 app。

## 测试

项目回归覆盖位精确完整 roundtrip、原始文档借用和实际嵌套事件、逐负载逆操作、引用重接、高水位删除/undo/重载/耗尽、所有截断前缀、计数与大小限额、损坏图、非规范顺序和 rational、失败状态不变、历史淘汰与旧 token。

文件回归使用真实临时文件、共享异步 worker 和真实项目 codec，覆盖完整读写后实际 registry/事件、同长度外部字节冲突、坏格式读写不覆盖、missing/empty 区分、结果槽限额、旧 token/request identity、cancel/timeout 后串行 barrier 确认无迟到发布、stage 清理、协作锁及 symlink。通用 worker 继续运行原 timeline 围栏回归，并新增 sequence codec 的确定性 checkpoint：真实 stage fsync 后取消，确认旧项目保留和 stage 移除；进入发布后取消返回 TooLate，超时及 service drop 仍返回真实新字节和原项目 token。

2026-10-21 验证：timeline crate 的 lib 与全部 integration tests 共 91 项通过，包括项目文档 13 项、项目文件存储 8 项和 worker 内新增 2 项 sequence 围栏回归；全部 targets 的 Clippy 以 -D warnings 通过。共享 target 为 /private/tmp/prism-editor-joint-cr1ODK，不属于本切片可清理目录。
