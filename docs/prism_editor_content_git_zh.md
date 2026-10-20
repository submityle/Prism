# E7 Git 状态 provider

实现位于 `pkg/prism_editor_content/src/git.rs`，通过主包的 `pub mod git;` 导出，仅依赖 Rust 标准库与包内 VCS 合同，不依赖 Bevy 或异步 runtime。集成测试直接导入公开模块。此适配器只读取仓库，生产代码不执行 add、commit、checkout、lock 或 LFS 写动作。

## 调用与原子应用

`GitProvider::new(GitLimits)` 创建并发域，克隆共享容量。`start(root)` 立即返回请求或 Busy/启动错误，所有文件系统操作、子进程启动、读取与解析均在监督线程进行。root 必须是含 `.git` 文件或目录的 worktree 根；接受 linked worktree，拒绝子目录自动向上发现，避免路径根歧义。

`GitRequest::poll()` 非阻塞，返回 None 表示尚未完成（或结果已消费），最多交付一次 `Result<GitSnapshot, GitError>`。持有请求的 UI 必须自行记录完成状态。`cancel()` 非阻塞取消，消费前的取消也会拒绝已经就绪的成功结果。drop 自动取消；监督线程仍拥有子进程并负责回收。需要观察终止错误时，应 cancel 后继续 poll，而不是直接丢弃句柄。

启动前捕获 `catalog.snapshot_token()`。成功时，先验证当前项目/root 与调用者的请求 epoch 仍匹配，再通过主模块提供的精确 token 发布入口调用：

```rust
catalog.publish_vcs(&expected, snapshot.into_vcs_statuses(), VcsSnapshotLimits::default())?;
```

token 过期、发布预算或批量路径错误全部原子拒绝，保留旧状态。旧接口 `replace_vcs_statuses` 也可应用快照，但调用者需在同次独占 catalog 访问中先自行验证 token；新增异步入口应优先使用 `publish_vcs`。Git 模块自身不增加发布方法。

所有读取、退出、预算、解析和路径校验完成前，不交付快照，也不接触 catalog。任何 Err 都应显示给用户并保留原 catalog。诊断保留原始 stderr bytes；成功时也通过 diagnostics 显式返回，不静默丢弃警告；消费 `into_vcs_statuses` 前应取出需要展示的 diagnostics。revision 和 lock_owner 均为 None，本次读取不采集提交 ID 或锁所有者。

## Native host 刷新合同

本包提供采集与 catalog 发布合同；原生事件循环、刷新动作和诊断控件由 app 持有。上述 provider 回归不代表原生 UI 接入已经完成。host 每次挂载或切换项目建立新的 session；每次请求刷新立即更换 refresh epoch，即使 catalog 没有变化也必须撤销旧请求的发布资格。exact token 仅校验 catalog 快照身份，不能代替请求 epoch：新结果发布为 `Ok(false)` 时 token 不变，旧请求仍必须被拒绝。

每个 host session 保留一个在途句柄和一个最新刷新意图。刷新到来时取消旧句柄、覆盖待启动意图，并在事件循环中继续非阻塞 poll 旧句柄直至终态；连续点击刷新不能积攒请求列表。消费旧终态后再尝试启动最新意图，并在实际启动时捕获 catalog token。共享 provider 仍可能被其他使用者占满；Busy 应成为可见的等待状态，保留一个最新意图并由后续 wake 重试，不能在 UI 线程自旋或另建 provider 绕过容量。

结果只能在同次 UI 线程访问中，先检查 session、root 绑定、refresh epoch，再调用 `publish_vcs` 检查捕获的 exact token。重新挂载 catalog、换 root、显式取消或更新的刷新请求均撤销旧结果资格。GitSnapshot.root 是 worker 规范化后的绝对路径；host 应绑定请求提交时的 root 身份，不能把成功结果的 root 反过来作为当前项目身份，也不能在 UI 线程为比较路径执行同步 canonicalize。若 host 需要与项目根作规范路径相等校验，应由已有项目来源或后台任务提供该身份。

发布成功但未变化仍应结束当前请求，并展示本次诊断；发布失败应保留整个旧 catalog。当前请求的启动、Git 退出、预算、超时、清理与发布错误均需要进入可见诊断。成功 stderr 必须在消费 snapshot 前取出；非 UTF-8 诊断可转义展示并保留原 bytes，不能静默丢失。旧 session/epoch 的结果不能覆盖新会话的状态或诊断；显式用户取消可以展示已取消，内部替换产生的取消属于已撤销请求。诊断应替换当前有界状态，显示文本另设长度预算，不能无限追加历史或直接将最大输出量全部加入布局。

原生验收应使用实际 Git 仓库经过刷新动作、事件循环 poll 和 `publish_vcs`，验证 Clean/Modified 可见变化、ready 未消费时刷新、在途刷新与旧 epoch、同内容 `Ok(false)`、换项目或 catalog 后拒绝旧结果，以及失败后旧状态保留和可见诊断。阻塞/取消测试可沿用真实 Git 的 FIFO index，不需要假进程或伪造 snapshot。仅调用 content 层 exact-token 回归不能替代这些 app 行为验证。

## 真实数据与路径语义

每个请求顺序执行以下真实 Git 子进程，不经过 shell：

```text
git ... ls-files --cached --deduplicate -z
git ... status --porcelain=v1 -z --untracked-files=all --ignored=traditional --ignore-submodules=dirty --renames
```

先为 ls-files 确认的索引路径建立 Clean，再覆盖 status 返回的变化。冲突 stages 通过 deduplicate 合并，不会遗留 Clean 冲突项。不能从快照缺省推断任意路径 Clean；被省略的目录、目录外路径等仍是 Unknown。ignored 使用 traditional 与 untracked-files=all 的真实逐文件输出，目录内资产可 exact-path 命中，不根据父目录伪造文件状态。不递归采集子模块内部脏文件状态；dirty 模式仍检测父仓库 gitlink 提交变化，Clean 仅表示父仓库状态。

porcelain 按 NUL 分隔，不按行或空白分割；保留空格、tab、换行和 Unicode。重命名/copy 读取目标、来源两个路径。重命名目标保留 original_path 与原始 XY；来源增加 Deleted（index rename 为 staged），来源重新出现时真实记录优先。copy 不删除来源。XY 两字节完整保留，VcsRecord 只能表达一个状态，投影优先级为冲突、特殊 ??/!!、Renamed、Deleted、Added（含 copy）、Modified。未解决冲突 staged=false，其他 tracked index 非空变化 staged=true；MM/AM 等同时变化不会丢失 XY。

拒绝非 UTF-8、反斜杠、绝对路径、空路径 component、. 与 ..，错误包含原始 bytes；不做 lossy 替换或 trim。忽略目录输出若出现尾 / 仅去一个语法尾斜线。跟踪路径与资产路径必须使用相同仓库根和正斜杠表示。

## 预算与生命周期

默认整请求 stdout+stderr 合计 8 MiB、100000 个路径出现次数、同一 provider 2 个请求、10 秒截止时间；均可显式配置，零限额拒绝。两条命令共同计费，rename 来源单独计路径，同一路径出现在 ls-files/status 中计两次。超预算是完整请求失败，不截断成可应用结果；解析与返回前也检查取消/截止。

并发 permit 同时覆盖 worker、请求句柄和未消费结果，成功和错误结果均占槽，防止积攒 ready 快照绕过容量。消费结果后释放；drop 时需等 worker 关闭管道、终止并回收 Git 才释放。调用者主动消费并另行缓存快照后的内存归调用者管理。单个 worker 只有一个 Git 子进程；Unix socketpair 两端替代 stdout/stderr 管道，父端 nonblocking，监督线程公平读取两流，空闲最多休眠 2 ms，避免阻塞读线程泄漏和 stderr 填满死锁。字节预算限制逻辑载荷，不承诺 allocator 精确 RSS；读取 scratch 固定 8 KiB，索引结构开销随路径数和路径 bytes 有界。

超时、取消、预算或 I/O 错误触发 child.kill 并 child.wait；wait 即使 kill 失败也执行。清理错误显式携带原始原因。process_id 仅诊断，不应被外部用于信号操作（PID 可能复用）。无同步 Drop 等待 UI；超时不承诺内核不可中断 I/O、进程创建或文件系统调用的硬实时终止。

命令禁用 optional locks、fsmonitor、untracked cache 与 pager，stdin 为 null，移除继承的 GIT_* 重定向/配置变量，以免读到别的 index/worktree。读取不刷新写 index，测试核对 index 内容与修改时间。使用 PATH 中可信 Git；需要支持 --deduplicate、--ignored=traditional 的版本（测试环境 Apple Git 2.39.5）。不覆盖用户 HOME/config，不运行网络操作。

## 平台与一致性限制

当前可运行实现仅 Unix（含 macOS/Linux），非 Unix 返回 UnsupportedPlatform；尚非 Windows 全平台交付，不引入 unsafe 或平台依赖来伪装支持。仓库必须可访问 Git 元数据，bare 仓库不支持。Git 没有跨 ls-files/status 与外部修改的事务快照；扫描期间外部改动可能使结果反映相邻时间点，调用者应在变更事件后再次刷新，切换项目/新请求时拒绝旧 epoch。不存在目录所有文件的 Clean 推断，也不遍历嵌套仓库/子模块内部；不采集 LFS 锁状态。

## 验证

2026-10-20：主线程已在实际主包验证 `tests/git` 的 12 项真实 Git 仓库回归，并通过 `--all-targets --all-features --no-deps` strict Clippy。主模块已公开导出 git，测试直接导入主包公共 API。Git 测试在沙盒外完整通过；沙盒内特殊文件路径创建与 ps 可能被系统拒绝，不能将该拒绝当成 provider 行为失败或跳过后宣称完整通过。独占 Rust 文件 rustfmt 检查也已通过。

真实临时仓库回归覆盖 Clean、索引+工作区双变化、重命名来源/目标及来源重建、空白/换行/Unicode、未跟踪、逐文件忽略、真实 merge 冲突、非 UTF-8/反斜杠拒绝、合并预算失败保留 catalog、dirty 回到 Clean、原子批量路径拒绝、精确 token 过期拒绝及遗漏路径 Unknown、真实 Git 非零退出、取消/超时/drop 回收与共享容量。生命周期测试让真实 Git 打开 FIFO index 阻塞，再验证终止后 ps 中无存活或 zombie PID；不使用假 Git 或状态 fixture。APFS 拒绝创建非 UTF-8 文件名时，通过真实 `git update-index --cacheinfo` 建立该索引路径；允许该文件名的平台额外覆盖未跟踪路径。

建议以独立小 target 验证，结束立即清理该 target：

```sh
CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 cargo test --offline -p prism_editor_content --test git --target-dir <owned-target>
CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 cargo clippy --offline -p prism_editor_content --all-targets --all-features --no-deps --target-dir <owned-target> -- -D warnings
rustfmt --edition 2024 --check pkg/prism_editor_content/src/git.rs pkg/prism_editor_content/tests/git.rs
cargo clean --target-dir <owned-target>
```
