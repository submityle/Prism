# World 项目 membership 与作者历史

`prism_editor_world::project` 保存真实 `EditorWorldData` 的项目目录：world GUID、全部数据层及父层/可见/可编辑策略、cell 的直接层成员集合、全部 actor GUID 到唯一 owner cell 的关系。捕获覆盖干净 actor，不使用只枚举 dirty 写操作的 `plan_ofpa()`。所有类型不依赖 Bevy，也不固化任何游戏对象种类。

## API 与根接线

world 根声明 `pub mod project;` 即可使用模块内全部公开类型：`WorldProject`、`ProjectLayer`、`ProjectLimits`、`ProjectError`、`ProjectEdit`、`ProjectChange`、`ProjectHistory`、`WorldProjectToken`、`WorldProjectCodec`、`WorldProjectStorageService`、`WorldProjectStorageRequest`、`WorldProjectStorageResult`、`WorldProjectLoadCandidate`、`WorldProjectSaveReceipt`、`WorldProjectStorageError`。world Cargo 依赖 `prism_editor_timeline = { path = "../prism_editor_timeline" }`，复用其通用 document worker。

`EditorWorldData::actor_records()` 需要转发已有的 `self.ofpa.actors()`，提供所有注册 actor 的确定性只读迭代。通用存储错误需要 `TimelineStorageError::DocumentCodec { domain: &'static str, message: String }`，world 使用 domain `world-project`，不伪装成 timeline/sequence 错误。根与既有模块由主线程接线。

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

## 验证范围

`tests/project.rs` 覆盖真实 world 干净 actor 捕获、完整 canonical roundtrip、真实身份物化、每个截断前缀与逐字节损坏、非法 wire/预算、批量最终引用关系、失败原子性、稀疏逆冲突及非关联编辑保留、历史 byte/entry 预算、深层层级与修订耗尽。

`tests/project_storage.rs` 使用真实临时目录：共享 worker 保存/重启重载、实际文件字节与 CAS 冲突、codec domain/错误身份/损坏/文件预算、槽位与取消出版围栏、旧 receipt 拒绝，以及经重载目录解析并读取既有真实 OFPA actor 文件。测试不会以假 payload 或假 IO 代替后端。具体执行结果以本切片提交报告为准。

本切片实现时，共享根接线尚未合入，因此使用临时源码副本补入上文的最小根接线，复用 `/private/tmp/prism-editor-joint-cr1ODK`，jobs=2、debug=0、incremental=0。`cargo test -p prism_editor_world --offline` 完整副本回归 97 项测试及 1 项 doctest 通过，其中新增项目/存储测试 19 项；`cargo clippy -p prism_editor_world --all-targets --no-deps --offline -- -D warnings` 通过。副本验证不是主仓库已接线的证据；主线程接线后仍需在真实 crate 联合验证。共享 target 保留。
