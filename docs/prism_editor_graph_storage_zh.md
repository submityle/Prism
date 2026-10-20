# 通用节点图完整文档与异步文件存储

实现位于 `pkg/prism_editor_graph/src/graph/storage.rs`，通过 graph crate 根公开。它保存实际完整图，不依赖命令重放、Bevy、GPU、项目节点类型或材质域。节点 payload 是不透明 `Vec<u8>`；端口类型键由宿主提供。

## API 与宿主接线

- `GraphDocument::{new, encode, decode, graph, positions, next_ids, into_parts}`：拥有完整图、位置及四类 ID 高水位。`into_parts` 转移所有权，不克隆整图。
- `GraphEditor::from_document(document, limits)`：显式接纳候选文档，创建新的状态身份，历史为空。`GraphEditor::snapshot(limits, max_bytes)` 从借用的图编码，返回 `GraphSnapshot { bytes, token }`。原有 undo 仍保存受影响记录的逆操作。
- `GraphStorageService::new(absolute_path, GraphStorageLimits)`：每个服务一个串行后台 worker。`load(timeout)` 与 `save(bytes, snapshot_token, expected_revision, timeout)` 返回可轮询的 `GraphStorageRequest`；构造、入队和轮询不做文件系统 I/O。
- `GraphStorageResult::Loaded(GraphLoadCandidate)` 返回完整文档及精确文件版本；`Saved(GraphSaveReceipt)` 返回状态 token、精确版本、持久化结果与是否越过期限。
- `GraphStorageRequest::{id, poll, cancel}`：单次消费结果；`id` 仅在同一服务内递增。跨服务或切换文档的 load 代次由宿主另行维护。

这些 API 已由 `prism_editor_graph` 根导出，没有新增依赖；本切片不需要修改共享 UI lib/Cargo。宿主保存时应提交同一次 `snapshot` 的 bytes/token，expected revision 使用上次 load/save 返回值。token 是进程内状态身份，并非内容哈希，API 不验证任意调用者传入的 bytes/token 是否配对。

保存结果不会自动清除 dirty。宿主先确认文档身份、请求代次，再比较 receipt token 与当前 `snapshot_token()`；仅匹配时可确认当前状态已保存。成功 edit、undo、redo 都生成新 token，失败保持 token，新的 editor/import 也有不同 token。undo 回到相同内容时仍保守地视为另一状态。load 候选不会替换当前 editor，接纳和 dirty 冲突处理由宿主明确决定。

## v1 完整文档协议

所有整数小端；字符串为 u32 字节长度加 UTF-8，payload 为 u32 长度加原始字节。

| 顺序 | 内容 |
| --- | --- |
| 0..6 | `PGRD`、版本字节 `1, 0` |
| 6..38 | 四个 u64：next node / port / edge / group |
| 38..50 | 三个 u32：node / group / edge 数量 |
| nodes | 按 ID 升序：ID、名称、payload、x/y i32、group u64（0 表示无）、port 数量、端口记录 |
| ports | 保持声明顺序：ID、名称、类型键、direction u8（input=0/output=1）、cardinality u8（single=0/multiple=1） |
| groups | 按 ID 升序：ID、名称、parent u64（0 表示无） |
| edges | 按 ID 升序：ID、output port ID、input port ID |

解码拒绝未知版本/tag、非法 UTF-8、空名称/类型键、尾随或截断字节、重复或零/MAX 活跃 ID、非规范节点/组/边排序，以及不大于现有 ID 的高水位。允许高水位为 u64::MAX，表达该类别身份耗尽。端口声明顺序不强制按 ID 排序；全图端口 ID 必须唯一。

解码重建端口名称、入出边、组成员和子组索引，实际检查端口类型、方向、连接数量政策、悬空引用、组循环和节点拓扑循环。组的父记录可以出现在后方。每个活跃节点必须恰好有一个位置，坐标绝对值不超过一百万。文档不包含 undo 历史、运行时状态 token 或宿主资源反序列化逻辑。

## 资源界限

默认最多 2 个存活请求槽，硬上限 8；运行、排队、未消费结果和已消费但未 drop 的 handle 都占槽。若 handle 被丢弃而 worker 仍持有请求，槽要等 worker 完成才释放。

默认每份文档和 expected revision 最多 8 MiB，硬上限 64 MiB；入队检查保存 Vec 的长度及 capacity。读取先检查文件长度，再以最多 16 KiB 的块读取，并检查增长超限。codec 在分配记录/复制字段之前检查计数和字段长度，编码每次追加检查总字节上限。

`GraphDocumentLimits` 复用 authoring 的节点、边、每节点端口、字符串、payload 政策，另有 groups（默认 4096、硬上限 16384，0 禁用）。command/history 的预算不作为文档总字节预算；其配置仍须合法。字节/记录上限不代表进程 RSS 或分配次数测量，也不保证操作系统调用在期限内返回。

## 文件协议、发布 fence 与恢复

父目录必须已存在、可信且保持稳定。worker 解析父目录，拒绝目标、锁和 stage 的 symlink/目录/非普通文件。为目标 `a.graph` 保留两个宿主不得另作他用的同级路径：

- `a.graph.prism-graph.lock`：持久 sidecar，由真实 OS `File::try_lock()` 锁定。load/save 都持排他锁；忙时返回 Busy，不阻塞等待。锁文件不主动删除，避免拆成两个锁 inode。
- `a.graph.prism-graph.stage`：持锁者唯一的未发布 stage。每次 load/save 获锁后仅检查并删除此精确路径上的遗留普通文件，不扫描其他文件。

保存依次执行：取得锁 → 清理遗留 stage → 有界读目标并比较完整 expected bytes → 解码验证新文档 → create_new stage → 分块写入 → stage `sync_all` → 再次完整读取比较 expected → 在与 cancel/poll 共用的 mutex 内切换到 Publishing → rename 替换目标 → 父目录 `sync_all` → receipt。Missing 与现存空文件不同；同长度、同 mtime 的修改仍会冲突。

Publishing 是取消协议的不可撤回 fence；rename 是单文件读者实际看到新版本的发布点。fence 之前取消/超时可以立即返回确定的未发布结果，worker 以后即使完成系统调用也不能 rename；RAII 尝试清理 stage，清理本身遇到 OS 错误时留待下次持锁恢复。fence 之后 cancel 返回 TooLate，即使超时或服务被 drop，poll 也要等实际发布结果，不能返回一个误导性的“已取消”。此时期限状态写入 receipt。rename 报错返回 `PublicationUncertain`，宿主须重新 load 确认；rename 成功而目录同步失败仍返回 Saved，但 durability 为 Uncertain。

进程死于 rename 前：目标保持旧版本，下次合作读写在 OS 自动释放锁后丢弃 stage。死于 rename 后：下次 load 读取新版本，不能推断中断前父目录同步是否完成。这是对未发布 stage 的丢弃恢复，不是重放事务日志。请求和服务 drop 不阻塞 join，不能强制中断已进入内核的 I/O。

原子可见性限于单个目标文件及提供原子 rename 语义的文件系统。合作 provider 的 CAS 在共享锁下串行；不遵守锁的外部 writer 仍可能在最终比较与 rename 之间写入。可信稳定目录要求不覆盖恶意目录替换、hardlink 别名或 sidecar 命名碰撞。没有多文件/跨图事务、网络文件系统一致性承诺或断电持久化实验。

## 验证证据

在独立 workspace 中复用仓库实际 `src`/`tests`，只构建无第三方依赖的 `prism_editor_graph`，使用 offline、单 job、debug=0、incremental=0。真实测试包含完整分组图往返、四类高水位及坐标、端口声明序与拓扑评估、接纳后 grouped node 删除/undo、旧 snapshot receipt、输入与句柄界限、完整前缀截断/逐字节变异、类型/方向/组循环/节点循环错误、精确字节 CAS、两个 provider、真实 OS 锁、symlink/目录拒绝、进程间读写、原子旧/新代读取。

内部测试在真实 stage fsync、Publishing、rename 后设测试专用 gate，验证取消/超时 fence、最终 CAS、rename 失败、目录同步打开失败。子进程分别停在 stage fsync 后和 rename 后，父进程确认其持锁，kill+wait 回收，再验证旧/新版本、stage 恢复、锁释放和下一次条件保存。gate 不进入生产构建。测试不等同于机器断电、真实设备 I/O 错误或性能测试。

2026-10-21 最终隔离 `cargo test --offline -p prism_editor_graph --lib --tests`：lib 17、authoring 9、graph_kernel 1、storage 16，共 43 项通过（session 97944，exit 0）。另执行 `cargo clippy --offline -p prism_editor_graph --lib --tests --no-deps -- -D warnings` 以及限定文件 rustfmt/diff 检查；最终结果及 target 清理量随本切片交付报告给出。此前 UI alltargets 的通过结果属于上一切片，本次不据此宣称 native 图文档加载/保存界面已经接入。材质域 native authoring 仍待后续宿主集成。
