# prism_editor_world：OFPA 真实存储与执行边界

本切片提供真实 actor 文件读写、逐文件持久化，以及隔离 version root 上的跨 cell/OFPA 多文件事务与重放恢复。`OfpaPlan` 仍只描述待保存元数据；只有 provider 返回的实际保存/事务回执经 world 核对后才能清除 dirty。payload 始终是不透明字节，不依赖项目 gameplay 类型、Bevy、wgpu 或运行时实体结构。

## 文件与版本

路径沿用 catalog 的 `cells/x_y_z/actors/<十进制 u128 actor GUID>`。`.lock` 是持久 OS advisory 锁侧文件，`.pending` 是同目录暂存文件。actor 与 cell 共享 envelope codec 和持久替换路径，没有复制另一套私有 scene 格式假设。actor magic 为 ASCII `PEWACTR` 后接 NUL，cell 仍为 ASCII `PEWCELL` 后接 NUL，原 cell 格式保持兼容。

固定 64 字节头全部使用小端序：magic 8、版本 u32、坐标 i32×3、GUID u128、revision u64、payload 长度 u64、校验值 u64，随后是恰好声明长度的 opaque payload。版本为 1，校验是头前 56 字节与 payload 的 FNV-1a-64；它检测偶发损坏，不提供抗恶意篡改认证。nil GUID 拒绝，所有 i32 坐标合法，包括负数与极值；读取时坐标和 GUID 必须与请求完全相同。未知版本、身份不符、截断、尾随数据、长度和校验错误均拒绝。

## 有界异步执行

`FileActorIoProvider` 的构造只同步解析已存在的私有 world 根目录；提交/轮询不执行文件 I/O，每个获准请求由有界 worker 执行。`ActorIoLimits` 分别限制单文件 payload、未释放请求数和总预留 payload 字节。load 预留请求上限，save 预留 `Arc<[u8]>` 长度且不复制 payload。stat 和头验证先于 fallible payload 分配；最大 envelope 沿用 scene 的 64 MiB 上限。

clone 共享预算；独立 provider/process 各有预算，通过 OS 锁协调同一文件。丢弃 future 后，worker 清理期间仍占预算；完成但未轮询的结果也占预算。已交还调用者的 snapshot/receipt、调用者保留的 pending 不计入在途预算；每 worker 的 64 KiB scratch、头、线程栈和元数据是额外开销。world 对每个 resident cell 的 actor payload 总和另设 `max_cell_bytes` 上限，原 cell payload 有独立同额上限；二者合计最多两倍该值，另加元数据和外部保留快照。

每请求要求非零 timeout，使用提交时建立的单调 deadline。worker 在工作前、分块间、发布 fence 前检查取消/超时。超时返回 `TimedOut`；阻塞内核 I/O 无法强制中断，已进入 fence 的 rename 与目录 sync 会完成。drop 取消阻止返回后的新发布；若 commit 已进入 fence，drop 等它退出，因此不能保证取消前磁盘从未改变。worker 清理可继续，关闭根目录前应等待操作数与预留字节数都归零。

## revision、回执与编辑租约

保存必须显式选择 `Missing` 或 `Revision(base)`。持有独占 OS 锁时完整校验旧文件，拒绝基线不符、倒退 revision、同 revision 不同字节。已存在同 revision 且完全相同的字节可作为幂等重试，即使请求基线已过时，也必须重新 sync 目录后才返回回执。多个独立 writer 不能凭较大的本地 revision 覆盖已变化的基线。

world 的 `replace_actor_payload` 要求 resident、editable cell 与有效 lease，原子推进 actor 和 cell revision。`prepare_actor_save` 捕获 bytes、actor/cell revision、residency generation、canonical root 和 lease；`commit_ofpa` 只接受与其完全匹配的实际回执。编辑版本变化、旧 residency、过期 lease 或 policy 变化保留 dirty；匹配的实际持久结果仍记录 durable base，供后续显式重试。成功 actor 回执不清除 cell dirty；所有 dirty actor 完成后还需单独保存 cell。独立 catalog 的 `commit` 只是外部元数据确认，不能绕过 world 的回执核对。

load 由项目提供 actor catalog，先 `prepare_actor_load`，再异步 `load_actor`，最后 `apply_actor_load`。dirty actor 禁止覆盖；失效 lease、错误 root/identity、变化的 actor/cell revision 或 residency 拒绝安装且无状态变更。成功安装恢复 durable actor revision/base。cell unload 释放其 resident actor bytes。此处不实现 catalog 扫描发现，也不解析 cell payload 中的 membership。

## 持久化与恢复

reader 使用共享 OS 锁，writer 使用非阻塞独占 OS 锁，忙时返回 `Busy`。锁 inode 不删除、不截断；进程死亡由 OS 释放锁。writer 在锁内验证旧文件，回收该 actor 的精确 regular-file 暂存路径，以 `create_new` 创建 stage，分块写入并 file sync，经过取消 fence 后原子 rename，再 directory sync。新建目录逐层 sync 父目录。rename 前失败保留旧 final，RAII 清理已拥有的 stage；进程崩溃残留 stage 从不作为 load 来源，后续非幂等保存仅回收自己的 stage，不清扫其他文件。

rename 后 sync 失败返回 `CommitUncertain`，不能据此清 dirty；保留并重试相同快照，重新确认持久性。损坏 final 拒绝覆盖，需外部恢复已知良好版本或选择新 root，不自动“修复”为新空文件。root/祖先/内容必须是应用拥有的可信工作目录；现有 symlink 与 special file 拒绝，但 std 的路径检查不是抵御恶意并发 inode/目录替换的 capability sandbox，非合作 writer 也可绕过 advisory 锁。

## 已实现的跨 cell/OFPA 协调事务

上述单文件 API 的原子单位仍是一个 actor/cell，不能将多次保存当作事务。新增 FileWorldTransactionProvider 在独立 root 内实际执行多文件 put/delete、actor 跨 cell move、redo 恢复和单 generation 读取；它不是原路径多次 rename 的包装。

### API 与使用顺序

- WorldFileTarget::{Cell, Actor} 保留数值 identity；WorldFileChange::{put, delete} 显式携带每目标 Missing 或 Revision(base)，put revision 严格推进。同一 actor GUID 跨 owner 的新 revision 也必须大于旧值。
- WorldTransactionRequest::new(id, base_generation, changes) 拒绝 nil、空列表、重复目标和溢出 generation，将目标按 kind、坐标、GUID 排序。每次提交使用独立 GUID；仅最新 generation 保留精确幂等重试信息，不提供无限历史 ID 去重账本。
- FileWorldTransactionProvider::load_snapshot(timeout) 先恢复已决定事务，再返回一整个拥有 opaque bytes 的 WorldTransactionSnapshot。commit(request, timeout) 返回不可自行构造的 WorldTransactionReceipt；同一 generation 的每个目标一起发布。
- world 初次向空 store 保存可直接准备；连接非空 store 先显式 discard_edits_and_restore_transaction_snapshot(snapshot, leases, now)。它明确丢弃 resident 编辑，要求完整 registered cell identity 覆盖及全部 live leases；恢复 actor catalog、ownership、payload、revision/base。此操作不会隐式扫描/注册项目 cell 或解析 membership。
- prepare_world_transaction(id, snapshot, edits, leases, now, provider) 捕获不可变请求、root/generation、每 cell revision/residency/lease，以及涉及 cell 的 actor records/bytes。WorldTransactionEdit::Actor 指定 destination 和 opaque bytes；跨 cell move 必须同时提供源、目标 Cell edit。引擎不替项目修改 membership schema。DeleteActor 要求 actor 已持久存在；新 actor 先在当前 owner 保存，再移动或删除。
- 准备完整验证后才推进涉及 cell revision 并令其 dirty，保留本地原 payload/ownership，防止 pending intent 因 lease 过期被安全卸载路径丢弃。commit_world_transaction(pending, receipt, now) 先核对全部 guards，再整体应用 cell bytes、actor put/delete/move、durable bases 和 clean 状态。不能遗漏未持久或 dirty 的现有 actor。新编辑、过期 lease、错误 root/request、旧 residency 或 actor catalog 变化时，整组保持 dirty，不能部分清理。

world 绑定一个 canonical root 与已确认 generation；旧 generation 和跨 root 快照不能直接继续准备。回执过时不回滚已提交磁盘：保留 dirty，在外部解决编辑冲突后显式恢复最新快照，或者重试原 pending 请求并重新核对。pending intent 本身不是持久本地编辑日志，进程退出后以已发布 snapshot 为准。低层 provider 允许通用文件集合变更；只有 world move API 强制两侧 cell opaque 更新，低层调用者仍负责 payload 的业务一致性。

### 实际布局与隔离读路径

root 下持久 .prism-world.lock 是全局 OS gate，.prism-world/ 是事务隔离标记与数据目录。事务 provider 拒绝含 legacy cells 的根；已有 legacy cell/actor provider 在每次实际 I/O 期间持共享 gate，检测标记后返回 TransactionalRoot。初始化持独占 gate，因此旧 provider 已创建但尚未开始的请求也不能绕过隔离。旧数据需由调用者显式导出并提交到新 root；没有原位迁移。

目录包含 CURRENT、journal、固定 .pending 暂存名、slot0/slot1 及最多 64 个持久 stripe lock 文件。generation 奇偶选择 slot；每 slot 包含完整 manifest 和 0..count-1 数值 envelope 文件。actor/cell envelope 直接复用原 codec。CURRENT、journal、manifest 也复用 version 1 envelope，采用独立 magic PEWROOT/PEWJRNL/PEWMANF 后接 NUL，metadata revision=0、coord=(0,0,0)、GUID=1。

CURRENT body 是 generation u64、transaction GUID u128、manifest FNV64；journal body 是 parent 与 next 两个 CURRENT body。manifest body 为文件数 u32、按序 target(kind u8、coord i32×3、GUID u128)/revision u64/length u64/payload FNV64，再接变更数 u32 和每目标 base/delete-or-put/revision 描述；整数全小端，tag 只接受 0/1。put 原始 bytes 引用最终 snapshot，不复制存入 manifest；重试会将读取的全部 bytes 与请求精确比较，而非只信任 hash。metadata 单文件最多 1 MiB，计数/长度/总字节先于 payload 分配验证，拒绝尾随数据、坏校验与版本。一个坐标只能有一个 cell identity，一个 actor GUID 在整个 generation 只能有一个 owner，所有 actor owner 必须有 cell 文件。

所有 reader/writer 都先取得非阻塞独占 gate，reader 因需要恢复而同样串行，竞争返回 Busy。reader 只读 CURRENT 指定完整 manifest，校验每个 envelope 后一次交回拥有 bytes 的 snapshot；后续 slot 回收不影响旧 snapshot。禁止绕过 provider 从 slot 或 legacy 路径拼接读取。因此原子可观察边界是单一 version root，而不是一组文件 rename。

### 决定提交、发布与恢复顺序

1. writer 持全局 gate，先恢复 journal，校验旧完整 generation。核对 exact root base_generation，再按升序取得所有目标涉及的唯一 hash stripe 锁；先核对全部目标 CAS，再构建与校验新完整集合。stripe 冲突只扩大互斥范围，固定集合不随 actor 数无限增长。
2. 只回收 inactive slot 的数值文件和 manifest。目录枚举最多 MAX_ENTRIES+2 项，完整验证 inventory 后才删除，拒绝未知文件、目录、symlink；不递归扫描其他目录。写完整新 slot、逐文件 sync、slot directory sync、store directory sync。旧 CURRENT slot 始终不动。
3. 写入并 file sync journal.pending，进入取消 fence 前最后检查 deadline/cancel；在 fence 内 rename 为 journal 并 store directory sync。这一持久 redo 决定是 commit point。之前中止只留下未发布 slot/pending；reader 仍选旧 CURRENT，后续 writer 回收自己的 inactive slot。
4. 决定之后不再取消/回滚。写并 sync CURRENT.pending，原子替换 CURRENT，sync store，再删除 journal 并 sync store 后返回回执。任何 journal rename 之后的失败返回 CommitUncertain，调用者不能清 dirty；必须恢复/重试确认。
5. 任意新 reader/writer 若发现 journal，先完整校验它及 next slot 的全部 envelope，并要求当前 root 为 journal 的 parent 或 next。随后在 fence 内重放 CURRENT 发布（若已发布则无需重复），删除并 sync journal。坏 journal、缺失/损坏候选、分叉 root 均 fail closed，不返回混合 snapshot，也不擅自丢弃已决定事务。

drop 在 fence 外取消可阻止此次提交，已进入 fence 则等待发布/不确定结果退出；deadline 在 fence 内过期不会撤销决定。恢复可在进入自身 fence 前因取消/超时退出，journal 保留供下次请求恢复。阻塞内核 I/O 仍无法强制中断。commit 前不需要 rollback 多个 final 文件，因为 reader 从未选择 unpublished slot。

### 有界资源与剩余边界

WorldTransactionLimits 限制最多 4096 files、单文件 bytes、整个 snapshot payload bytes（最多约 64 MiB）、请求数和总在途预留；每请求最多两倍 entries 的变更，以容纳删除旧目标和创建新目标。read 预留一个 snapshot 上限；commit 预留旧 snapshot 上限加全部输入 bytes；clone 共享预算，独立 provider/process 各有预算。metadata、集合、worker 栈、64 KiB scratch 额外有界；读入 Vec 转成 Arc 时还可能短暂复制最多一个 file_bytes 的 payload，预留量不是进程 RSS 上限。返回给调用者的 snapshot/receipt/pending 不计 admission。world 恢复和 receipt 安装会复制 resident bytes，受自身 cell/actor limits 约束，不能宣称零分配。

两个 slot 限制 store 自有历史为两份完整 snapshot、两份 manifest 和固定控制/锁文件；不提供无限增量日志。每次提交复制完整 generation，适用于当前有界编辑基础，未验证大世界吞吐量。空目标集合可由非空删除事务产生；cell/actor nil identity 拒绝，所有 i32 坐标合法。两份 slot 不是损坏时的自动回退备份；CURRENT 或已决定 journal 损坏时拒绝打开，外部需恢复已知完整副本。

当前真实文件测试只能证明当前 macOS 文件系统上的执行与故障处理；未做物理断电、网络文件系统、其他 OS 或恶意非合作 writer 验证。故障 hook 模拟 rename 前/后错误和 worker panic，不等于硬件持久性认证。

## 逐文件存储切片的历史验证（2026-10-20）

使用独立小工作区 `/private/tmp/prism-foundation-compat.qFtoL5`：复制依赖 manifests，源码链接到本共享 checkout；未构建整个仓库或 app。固定 `CARGO_INCREMENTAL=0`、dev/test debug=0、jobs=1，独立 target 为该目录下 `target-ofpa`。主命令前缀是 `cargo ... --offline --manifest-path /private/tmp/prism-foundation-compat.qFtoL5/Cargo.toml -p prism_editor_world`。

- `test --all-targets`：58 通过，0 失败；库 19、executable_data_layer 7、filesystem 12、ofpa_filesystem 12、recovery 8。两个集成测试入口是受父测试驱动的锁子进程辅助入口，计数包含其无环境变量的空跑；父测试实际启动、终止并 wait 子进程。最终构建会话 86997，exit 0。
- `test --doc`：1 个编译型 doctest 通过，exit 0。
- `clippy --all-targets --no-deps -- -D warnings`：exit 0；仅对本 crate 与测试启用严格 lint。
- 定向 `rustfmt --check`（lib 模块树与修改的集成测试）及限定写集 `git diff --check`：exit 0。

新增测试覆盖实际文件的 binary/empty/extreme-coordinate roundtrip、重开 provider、坏 magic/version/identity/checksum/length、稀疏超限文件、CAS 冲突与幂等重试、独立 writer 竞争、request/byte admission、取消 stage 清理、超时、rename 前错误、rename 后不确定提交、panic、OS lock 与进程死亡清理、symlink 拒绝、错误 receipt root/bytes、过期 lease、字节相同的新编辑、旧 load、actor/cell 独立提交与 restart 恢复。最后一轮前曾发现清理测试只等操作计数而未等字节预留析构的竞态，现已同时等待两者；完成但未取结果的预算测试改为确定性的 worker 唤醒同步。

验证后以精确 `cargo clean --target-dir /private/tmp/prism-foundation-compat.qFtoL5/target-ofpa` 清理自有构建产物，释放 172.9 MiB；无自有构建会话继续运行。没有清理其他任务 target。

## 协调事务切片的实际验证（2026-10-20）

复用同一独立小工作区的 manifests 与源码链接，单独使用 /private/tmp/prism-foundation-compat.qFtoL5/target-transaction；没有构建全仓、app 或 content。环境固定 CARGO_INCREMENTAL=0、CARGO_PROFILE_DEV_DEBUG=0、CARGO_PROFILE_TEST_DEBUG=0、CARGO_BUILD_JOBS=1。下述 cargo 命令均带 --offline --manifest-path /private/tmp/prism-foundation-compat.qFtoL5/Cargo.toml -p prism_editor_world。

- 最终 test --all-targets：session 51050，exit 0，78 通过、0 失败。库 39、executable_data_layer 7、filesystem 12、ofpa_filesystem 12、recovery 8；含 3 个无子进程环境时空跑的辅助入口，父测试实际 spawn、kill 并 wait/reap。事务新增 20 个库测试（含一个辅助入口）。此前 session 38783 也为 78/78，最终一轮增加了完整 generation 核对。
- clippy --all-targets --no-deps -- -D warnings：session 99878，exit 0，涵盖最终源码与全部测试。严格 lint 只施加到本 crate；未声称全依赖无告警。
- test --doc：session 97202，exit 0；现有 1 个编译型 doctest 通过。
- rustfmt --edition 2024 --check pkg/prism_editor_world/src/lib.rs（递归模块树）和限定文件 git diff --check：exit 0。

新增真实临时 FS 测试覆盖：跨 cell actor move、同 GUID 两方向 revision 推进、整批 target/root CAS、精确幂等与分叉重试、全局 actor ownership、整 snapshot 文件/字节/请求界限、保留旧 snapshot 后多次 slot 回收、独立 provider 的 reader/writer 互斥、预先存在的 legacy provider 隔离、实际 stripe lock 冲突、未决定/已决定/已发布故障、坏 journal/manifest/envelope 拒绝恢复、symlink 拒绝且保留外部文件、决定前取消/超时和决定后 deadline/drop fence，以及 world 新建/删除/move、错误 root receipt、stale edit/lease 和显式 restart restore。

kill 验证在 prepared、decided、published 三个实际 checkpoint 分别启动独立子进程，父进程先确认 OS gate 返回 Busy，再 kill、wait；重开 provider 校验 transaction ID、完整 cell/actor identity、revision 和 bytes。prepared 只看到旧 generation，decided/published 恢复到新 generation，并验证 journal 已清理、后续幂等提交可继续。不是仅检查文件数，也未把错误注入当作物理断电。

早一轮并行 FS 测试中两个 100ms deadline 在到达 hook 前到期；改为 2s deadline，并将 inactive slot 清理从固定 4096 次 stat 改为有界 inventory 检查。最终运行通过。clippy 发现的一处等价边界条件写法也已修正。未验证大世界性能、其他 OS/文件系统、真实硬件断电或恶意非合作 writer。

全部验证结束后立即清理上述精确 target：cargo clean 报告移除 531 files、176.3 MiB，exit 0。所有本切片验证 session 已结束，无自有构建继续运行；未清理其他任务 target。隔离工作区 manifests/source links 保留供后续小规模验证复用。
