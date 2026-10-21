# World 项目 membership 与作者历史

`prism_editor_world::project` 保存真实 `EditorWorldData` 的项目目录：world GUID、全部数据层及父层/可见/可编辑策略、cell 的直接层成员集合、全部 actor GUID 到唯一 owner cell 的关系。捕获覆盖干净 actor，不使用只枚举 dirty 写操作的 `plan_ofpa()`。所有类型不依赖 Bevy，也不固化任何游戏对象种类。

## API 与根接线

world 根已声明 `pub mod project;`，可使用模块内全部公开类型：`WorldProject`、`ProjectLayer`、`ProjectLimits`、`ProjectError`、`ProjectEdit`、`ProjectChange`、`ProjectHistory`、`WorldProjectToken`、`WorldProjectCodec`、`WorldProjectStorageService`、`WorldProjectStorageRequest`、`WorldProjectStorageResult`、`WorldProjectLoadCandidate`、`WorldProjectSaveReceipt`、`WorldProjectStorageError`。world Cargo 已依赖 `prism_editor_timeline = { path = "../prism_editor_timeline" }`，复用其通用 document worker，无新增根导出或依赖要求。

`EditorWorldData::actor_records()` 已转发已有的 `self.ofpa.actors()`，提供所有注册 actor 的确定性只读迭代。通用存储错误已有 `TimelineStorageError::DocumentCodec { domain: &'static str, message: String }`，world 使用 domain `world-project`，不伪装成 timeline/sequence 错误。以上四项根与既有模块接线由主线程负责，本切片不提交这些共享文件。

典型流程：

1. `WorldProject::capture(&world, limits)` 捕获目录；或 `new(world_guid)` 后通过作者命令建立目录。
2. `ProjectHistory::execute(&mut project, expected_revision, &edits, limits)` 原子更改；`undo`/`redo` 应用稀疏逆。
3. `WorldProjectStorageService::save(&project, expected_file_revision, timeout)` 编码真实目录并捕获精确 host token；保留 request，调用 `poll()` 或 `cancel()`。
4. 保存成功只在 `project.matches_token(&receipt.snapshot_token)` 时允许宿主标记该目录为 clean。即使 token 已过时，实际出版的 `receipt.revision` 仍是该次文件状态，必须妥善跟踪，不能假装保存未发生。
5. `load()` 返回独立验证过的 candidate 与精确文件字节 revision。宿主确认现有 dirty 内容、异步任务与租约已处理后才能采用 candidate。`candidate.document.materialize(world_limits, project_limits)` 创建新的真实 world，恢复真实 cell/actor 文件身份，所有 cell 初始为 unloaded。

## 有界、确定性 wire

版本 1 为 little endian。固定头是 magic `PEWPROJ\0`、u32 版本、u64 文件总长、u128 world GUID、u64 项目修订、三个 u32 数量（layer/cell/actor），最后为 u64 FNV-1a 校验。空项目共 64 字节。校验用于损坏检测，不提供认证。

layer 按 u32 ID 严格升序：ID、u32 UTF-8 名称长度及名称、parent tag（0/1，存在时跟 u32 parent）、策略位（visible/editable）。cell 按有符号 `(x,y,z)` 字典序：三个 i32 坐标、u32 membership 数量、严格升序 u32 layer ID。actor 按 GUID 严格升序：u128 GUID 和三个 i32 owner 坐标。零 layer ID 合法，world/actor GUID 不得为 nil。

解码前先检查 byte 上限、最小头、完整长度、版本与校验；分配前检查记录数量、名称长度、membership 总量及剩余字节。拒绝重复或乱序记录、非法 UTF-8/tag/flags、尾随字节、重复层名、非法层名、父层环、缺失层/cell 引用。层名沿用真实 world 的规则。层级验证不用递归，深层输入不会占用递归调用栈。

`ProjectLimits` 默认上限：8 MiB wire、4096 层、100000 cells、200000 actors、400000 memberships、每层名 1024 UTF-8 bytes、每条命令 4096 records。公开配置本身也有硬上限；编码、捕获、作者修改、逆历史回放和物化都重新验证适用预算。计数/长度溢出或无效配置明确返回错误。

## 作者命令与稀疏逆历史

`ProjectEdit::Layer/Cell/Actor` 完整替换或删除一个记录。一个命令按最终状态验证引用，因此允许同批新增 cell 后移动 actor、删除层并修复全部引用。删除不隐式级联；同批重复 target、无变化命令、非法引用或预算超限被拒绝。失败保持内容、修订、host token 和 undo/redo 栈。

`apply(expected_revision, edits, limits, max_change_bytes)` 返回 `ProjectChange`。每条 change 仅保留 touched records 的 before/after，不复制整个项目。撤销先比较 world GUID 与全部 touched 当前值，然后恢复；保留其他记录上的独立改动。恢复后仍须通过完整引用与预算验证；失败回滚。成功修改、撤销和重做都递增 u64 revision，耗尽拒绝，更新精确 Arc identity fence。

这是记录值 CAS，而非每记录 ABA 版本跟踪：独立修改若将 touched 值恢复到同值，内容 CAS 可以接受；保存 token 的 Arc/revision fence 仍拒绝旧 receipt。历史不能跨 world 使用；加载后宿主应显式清空旧历史。

`ProjectHistory` 同时限制 undo+redo 条数和确定性 admission charge。charge 包含两侧记录、名称、集合节点 allowance，不是 allocator/RSS 实测值。满预算拒绝新命令，不静默丢弃历史。成功分支清空 redo，失败保留 redo。验证需要扫描完整关系图；仅持久保留的逆数据随 touched records 增长，不能将其描述为所有操作都与世界大小无关。准备新分支时旧 redo 暂留直至提交，因此瞬时准备内存也受两个已配置预算控制。

## 实际存储与运行时消费契约

`WorldProjectStorageService` 是已有 `DocumentStorageService<WorldProjectCodec>` 的薄适配，不另写文件 IO/线程/事务实现。保存 API 将同一项目的实际编码与 token 一起提交；worker 在真实出版前再次解码，条件出版比较全部 expected bytes，复用现有槽位/取消/超时/锁/临时文件/rename/目录同步机制。取消过晚时必须查看实际结果，不能据取消请求推断未写文件。durability 与 deadline 结果由通用 receipt 原样传出。

项目目录不包含 resident payload、payload revision、dirty 标记、租约、edit scope、HLOD 构建结果或在途操作。cell/actor 的版本化 payload 继续使用现有 OFPA/Cell provider 和 WorldTransaction backend。物化只恢复目录和真实身份，不生成占位 payload，也不强行覆盖活跃 world。宿主需要按目录定位既有 payload 文件，经过已有加载/租约流程安装实际内容。

membership 项目文件出版与独立 OFPA payload 事务不是同一个跨文件原子事务。宿主必须为自己的保存策略排序、处理缺失 payload 和故障后恢复，不可将本模块宣称为跨目录/payload 的原子保存、原生应用挂载或 runtime 流式安装已完成。

## 原生 world 面板接线审计

真实调用链为 `pkg/prism_editor_app/src/desktop_ui/panels.rs` 的 `DesktopWorldPanel`，持有 `pkg/prism_editor_ui/src/world.rs` 的 `WorldPanelController`，再由 `world_view.rs` 生成带精确 stamp 围栏的 `BoundWorldPanel`。`DesktopWorldPanel::poll` 先轮询控制器绑定的真实 project storage，再调用宿主 provider；project I/O 完成也会触发面板刷新。app 的 `fenced()` 检查 project membership dirty、project I/O pending、loaded candidate、retired world，以及原有 pending cell intents、cell dirty/residency 和 provider fence。

`WorldPanelController` 已持有与真实 world 同步的 membership project、稀疏 `ProjectHistory`、精确 token、storage request、独立 loaded candidate 和 retired-world handoff。原生 app 现在驱动该 storage 生命周期，并将未保存 membership、在途 I/O、待处理 candidate 和未接管的 retired world 纳入关闭/卸载 fence；exact save receipt 才能清除对应快照的 dirty fence。真实临时目录 app 测试覆盖自动 poll、save、load candidate、adopt、retired-world handoff 和 fence 解除。

app 现在可通过 `DesktopWorldProjectBinding` 与 `DesktopWorldPanel::with_project` 一次安装 membership 文件路径、已观察 revision、worker admission、项目预算与超时；面板持续保留真实路径，`project_path()` 与 `project_file_revision()` 分别暴露绑定路径和最新成功 load/save receipt 学到的精确 revision。`None` 表示文件状态未知，首次保存前必须 load/adopt；`TimelineFileRevision::Missing` 只允许宿主已经明确确认的 create-only 路径，不能用来跳过现有文件协调。

宿主仍负责决定 open-existing/create-new 模式、消费 cell intent，并接入实际 payload/provider/runtime consumer；这不是跨 membership 文件与 OFPA payload 的原子事务，也不代表 runtime 流式安装已完成。candidate 采用仍须服从 controller 的精确 token 和旧 world/payload 保护条件。

宿主需要保留同一个 `WorldProject`、有界 `ProjectHistory`、last-clean token、精确文件 revision、唯一在途 request 与独立 loaded candidate，复用当前 document worker。`WorldProjectStorageService::save(&project, expected_file_revision, timeout)` 和 `load(timeout)` 已可调用，request 支持 `poll()`/`cancel()`。加载后调用 `candidate.document.materialize(EditorWorldLimits, ProjectLimits)`，失败时保留旧模型和 candidate，成功构建也不等于宿主已经采用。

接线必须满足：

1. 作者入口有界地接收 `ProjectEdit`，以项目 revision admission，通过 `ProjectHistory::execute/undo/redo` 验证完整关系。UI 所有者还需提供与真实 world 同步提交、成功后失效旧 bound view/intent 的接口；现有 project 历史只修改目录，不会更新 resident world。不得先改一侧、再以可能失败的另一侧更新宣称原子成功。尤其 actor owner 变化须复用既有 world transaction/租约守卫，不能绕过 OFPA 直接搬目录。
2. 捕获用于显式建立目录或受控对账。每次 `capture()` 都创建 revision=0 的独立目录与新 token，不能每次 `poll()` 重建来代替持续作者历史。已有层策略按钮和 cross-cell transaction 成功后须同步目录，再允许保存或 clean 认定；只检查一个未同步的缓存 project token，不能证明当前真实 world 已保存。
3. 保存先保留 request 和本次 token。实际 receipt 总要更新所观察到的文件 revision；仅当当前目录匹配 `receipt.snapshot_token`、目录与真实 world 已同步时才更新 last-clean。过期 receipt 不清理更新后的编辑，`PublicationUncertain` 要求重新读取并对账；取消后仍轮询终态。membership dirty、候选待采用和不确定出版均进入宿主 `fenced`。
4. candidate 采用前，宿主显式处理旧 world 的 dirty payload、在途 IO/事务、保留租约和运行时消费者；不能从 cell dirty=false 推断目录 clean。完整替换路径可在这些条件满足后，以 materialize 的全 unloaded 新 world 构造新 `WorldPanelController`，同时替换其 bound view、清空旧历史、拒绝旧 controller tickets/回执，并重新启动真实 payload 加载。未完成这些条件时保留 candidate，不自动覆盖。

world 保留自己的 `WorldProjectStorageService` request/token 语义，不套用要求 `PartialEq` 的通用 bytes+token 面板宏，也不新增 worker。当前 app 回归会销毁第一套 `EditorDesktopUi`/controller，再创建完全独立的第二套会话，从磁盘 load candidate、adopt 并验证 layer/cell/actor membership、identity、精确 revision、旧 controller intent 失效和 retired-world fence；这比单 controller 测试更接近重启，但不冒充真实 native event loop 或进程重启。

`DesktopWorldPanel::next_wake` 在 project request 保留期间以 10ms 单调 deadline 驱动原生 `WaitUntil`，因此 save/load 不再依赖额外窗口输入才能收取完成结果；provider 自身更早的 deadline 仍优先。正式 CLI 入口的 `WorldProjectHost`/session、retired runtime detach/attach，以及真实 OFPA payload 的重启 round-trip 仍未接线，继续作为 Editor 未完成项。

## 验证范围

`tests/project.rs` 覆盖真实 world 干净 actor 捕获、完整 canonical roundtrip、真实身份物化、每个截断前缀与逐字节损坏、非法 wire/预算、批量最终引用关系、失败原子性、稀疏逆冲突及非关联编辑保留、历史 byte/entry 预算、深层层级与修订耗尽。

`tests/project_storage.rs` 使用真实临时目录：共享 worker 保存/重启重载、实际文件字节与 CAS 冲突、codec domain/错误身份/损坏/文件预算、槽位与取消出版围栏、旧 receipt 拒绝，以及经重载目录解析并读取既有真实 OFPA actor 文件。测试不会以假 payload 或假 IO 代替后端。具体执行结果以本切片提交报告为准。

初次实现提交 `63113dd24` 时根接线尚未合入，曾使用临时源码副本验证；以下是真实 checkout 在四项根接线到位后的复核结果，取代此前仅副本验证的状态：

| 真实 checkout 命令 | session | 结果 |
| --- | --- | --- |
| `cargo test -p prism_editor_world --all-targets --offline` | `3877` | exit 0；39 unit + 7 executable_data_layer + 12 filesystem + 12 ofpa_filesystem + 13 project + 6 project_storage + 8 recovery，共 97 项通过 |
| `cargo clippy -p prism_editor_world --all-targets --no-deps --offline -- -D warnings` | `71814` | exit 0，无 warning |
| `cargo test -p prism_editor_world --doc --offline` | `96226` | exit 0，1 项 doctest 通过 |

三条命令均使用 `CARGO_TARGET_DIR=/private/tmp/prism-editor-joint-cr1ODK`、`CARGO_BUILD_JOBS=2`、`CARGO_PROFILE_DEV_DEBUG=0`、`CARGO_PROFILE_TEST_DEBUG=0`、`CARGO_INCREMENTAL=0`。测试没有失败或忽略项，所有上述 session 已结束；没有创建新的 build target，也不清理共享 target。根接线的独立提交仍由主线程负责。
