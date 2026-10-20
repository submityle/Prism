# E7 项目源文件系统异步发现

本切片以标准库后台线程枚举真实目录，生成完整源元数据快照。不读取文件内容，不分配 AssetId，不推断导入类型，不使用路径哈希冒充重命名稳定身份。宿主负责稳定身份、目录到资产的映射以及目录模型的原子替换和代际校验。

本提交只拥有以下三个文件：

- `pkg/prism_editor_content/src/filesystem.rs`
- `pkg/prism_editor_content/tests/filesystem.rs`
- `docs/prism_editor_content_filesystem_zh.md`

`lib.rs` 的 `pub mod filesystem;` 由 main 集成。本切片不修改 crate manifest、README、Git 适配器或宿主的 `source.rs`。

## 调用与生命周期

使用 `FilesystemDiscovery::new(limits, excludes)` 创建服务，调用 `start(root)` 启动真实 OS 工作线程。目录枚举和元数据查询在工作线程执行；调用方仅校验根路径、解析其绝对路径并预留任务槽位。

`FilesystemScan::poll(&mut self)` 返回 `Poll::Pending` 或一次 `Poll::Ready(Result<FilesystemSnapshot, FilesystemError>)`。宿主在更新循环中轮询，无异步运行时依赖。这不是带 waker 的 Future；`is_finished()` 只观察线程结束，不消费结果。重复消费返回 `AlreadyConsumed`。

同一个服务的所有 clone 共用原子并发预算。已经完成但尚未消费的结果仍占槽位，避免无人领取的快照绕过限制。成功、失败和取消的终态消费均释放槽位；线程启动失败也回收预留槽位。独立服务各自拥有预算，宿主应复用服务限制项目级总并发。消费后由宿主持有的历史快照不属于在途预算。

`cancel()` 设置共享取消标志，即使线程已经完成，尚未消费的成功快照也会被取消丢弃。`Drop` 先取消再 join，回收工作线程、目录句柄和未消费快照后才释放槽位，不遗留 detached worker。

超时从启动线程前计算，包含调度等待，在目录/条目/元数据操作之间以及排序前后检查；零时限产生 `TimedOut`。期限针对工作线程执行，完成后等待消费的时间不计入超时。标准库不能中断已经进入内核的阻塞文件系统调用，因此取消和超时是协作式的，drop 可能等待该调用返回，不能承诺硬实时截止或有界 join 延迟。正常轮询只 join 已结束的线程。

## 快照与路径契约

快照包含绝对原生根路径、文件及目录条目、扫描计数和输出路径字节总数。条目只有相对路径、文件/目录种类和文件大小；目录大小统一为零。根自身不生成条目，空目录保留，空项目成功返回空列表。

相对路径按 UTF-8 字节顺序排序，保留组件中的原始空格、换行和 Unicode 字符，以 `/` 连接。不 trim、不大小写折叠、不做 Unicode 规范化、不替换反斜杠、不做有损字符串转换。非法路径、非 UTF-8 名称和反斜杠名称均产生显式错误。排除配置也拒绝空路径、绝对路径、空组件、点组件、父组件、NUL 和反斜杠。原生 Windows 根可以使用系统分隔符；项目相对路径始终使用斜杠。

根必须非空、可表示为 UTF-8，不能包含父组件或 NUL；Unix 根也拒绝反斜杠。相对根在启动时解析为绝对路径。错误保留原生 `PathBuf`、I/O 操作名称和原始 OS 错误；显示转义路径，换行和非 UTF-8 字节不会变成含糊文本。

线程私有累积快照，全部成功并排序后才交付一个结果。遍历、路径、预算、取消或超时错误均丢弃整批条目，绝不返回部分成功。这是发布的原子性，不是文件系统事务：外部进程仍可能增删改名，元数据不保证同一时间点一致。宿主应在失败时保留旧快照，并校验自己的刷新代际。

## 预算与遍历范围

上限均包含边界。只有 `max_in_flight` 必须大于零，其余零预算有明确含义，例如零输出预算允许空目录成功。

| 字段 | 默认值 | 作用范围 |
| --- | --- | --- |
| `max_in_flight` | 2 | 同服务及 clone 的在途任务及未消费结果 |
| `max_entries` | 100,000 | 输出文件和目录总数 |
| `max_path_bytes` | 4,096 | 完整 UTF-8 相对路径，包含所有斜杠 |
| `max_total_path_bytes` | 16 MiB | 所有输出路径字节数之和，包含目录 |
| `max_scanned_entries` | 200,000 | 每个枚举条目，包含排除项和符号链接 |
| `max_depth` | 64 | 根为零，直接子文件/目录为一 |
| `max_file_size` | 16 GiB | 常规文件元数据长度 |
| `max_root_path_bytes` | 16 KiB | 输入根及解析后的原生绝对根路径字节数 |
| `timeout` | 30 秒 | 线程耗时，按协作式检查执行 |

深度优先迭代遍历避免递归栈增长，不预收集完整目录列表。最多保留 `max_depth + 1` 个遍历目录句柄。输出数量和总路径预算限制快照内存；路径和深度预算限制遍历栈与暂存路径。Vec 容量、线程栈、OS 目录迭代器和排除配置不是精确字节配额，路径总字节数不能理解为进程内存上限。

先计入枚举预算并验证名称和单路径字节数，再判断排除。被排除的目录不打开，后代不枚举、不占后代预算。非排除条目以不跟随链接的元数据操作分类；符号链接直接跳过，不输出、不读取目标，包括悬空和循环链接。特殊文件（例如 Unix socket）报错，不打开内容。

根和每次将要打开的目录逐级检查所有祖先，遇到符号链接即报 `SymlinkDirectory`。macOS 的系统目录别名也受此约束；调用方需明确提供选定目录的真实绝对路径，模块不会通过 canonicalize 悄悄跟随链接。标准库路径式检查与打开间仍有 TOCTOU 窗口，无法抵御恶意并发替换为链接；本模块不是敌对目录树的文件系统沙箱。

## 可配置排除

默认在任意深度排除精确 basename：`.git`、`.hg`、`.svn`、`target`、`node_modules`、`build`、`dist`、`.DS_Store`。同时适用于文件和目录（例如 worktree 的 Git 元数据文件）。

`FilesystemExcludes::none()` 从空策略开始；`exclude_name()` 增加任意深度的精确 basename，`exclude_subtree()` 增加相对路径及其整棵子树。匹配大小写敏感，无 glob，不把相似前缀误当作同一名字。剪枝目录内部的非法名称不可见；已枚举的非法名称不会以排除为由静默忽略。

## 验证证据与边界

2026-10-20，在 macOS 以真实临时目录运行 28 项集成测试，全部通过。覆盖嵌套和空目录、文件大小、空格/换行/Unicode 精确保留和排序、默认及自定义排除、所有预算、共享服务并发竞争、完成结果占槽、立即和完成后取消、超时、drop 后复用、根及 I/O 错误、符号链接目录/文件/悬空/循环/根/祖先、Unix socket 拒绝。没有替代文件系统 I/O 的 mock 或 fixture。

APFS 拒绝创建非 UTF-8 文件名（实测 EILSEQ），本机验证非 UTF-8 原生根参数拒绝及字节保留；真实非 UTF-8 目录条目测试仅在非 macOS Unix 启用，本次未运行。未验证 Windows、敌对并发替换、网络文件系统硬截止、线程启动失败注入或整个仓库构建。

为避免重建共享工作区依赖，使用独立无依赖 Cargo harness，直接将仓库实际 `src/lib.rs` 配置为库目标、实际 `tests/filesystem.rs` 配置为集成测试，通过 main 集成的公开 API 调用。没有复制实现，没有修改工作区 manifest 或 lockfile。临时 manifest 如下：

```toml
[package]
name = "prism-editor-filesystem-verification"
version = "0.0.0"
edition = "2024"
publish = false
[workspace]
[lib]
name = "prism_editor_content"
path = "/Users/wangkailong/Documents/GitHub/Prism/pkg/prism_editor_content/src/lib.rs"
[[test]]
name = "filesystem"
path = "/Users/wangkailong/Documents/GitHub/Prism/pkg/prism_editor_content/tests/filesystem.rs"
[profile.dev]
debug = 0
incremental = false
```

首次实现的验证命令（临时目录仅属于本任务，收尾删除；复验应新建独有目录）：

```sh
cargo test --offline --manifest-path /private/tmp/prism-editor-filesystem-PaGAJa/Cargo.toml --target-dir /private/tmp/prism-editor-filesystem-PaGAJa/target --test filesystem
cargo clippy --offline --manifest-path /private/tmp/prism-editor-filesystem-PaGAJa/Cargo.toml --target-dir /private/tmp/prism-editor-filesystem-PaGAJa/target --test filesystem -- -D warnings
rustc --edition 2024 --crate-type lib --crate-name prism_filesystem_validation -F unsafe_code -D warnings -W missing_docs -W unused_qualifications pkg/prism_editor_content/src/filesystem.rs -o /private/tmp/prism-editor-filesystem-PaGAJa/libprism_filesystem_validation.rlib
rustfmt --edition 2024 --check pkg/prism_editor_content/src/filesystem.rs pkg/prism_editor_content/tests/filesystem.rs
git diff --check -- pkg/prism_editor_content/src/filesystem.rs pkg/prism_editor_content/tests/filesystem.rs docs/prism_editor_content_filesystem_zh.md
```

Clippy 使用独立 harness 默认 lint 集合并将 warning 视为错误，不等同于执行工作区全部 lint。独立 rustc 检查另外覆盖本模块公开文档、unsafe 禁止和未使用限定名。文件 watcher、内容读取、导入器、资产身份分配及增量内容/重命名识别不在本切片范围。

### 严格 lint 补验（2026-10-20）

根据 main 的集成检查反馈，将 BTreeSet/Arc 改从 alloc 引入，fmt、原子类型、Poll、Duration 和 Error trait 使用 core；文档中的 `max_in_flight` 使用代码格式。测试继续通过公开的 `prism_editor_content::filesystem` 导入，不直接包含源文件，测试内的等价 core 类型也同步调整。行为和 OS 调用边界不变：阻塞系统调用不能被强行中断，取消/超时只在调用边界检查，drop 的 join 可能等待该调用返回。

本次独立 harness 仍指向实际 crate 库和真实集成测试，并将当前根 manifest 的 `[workspace.lints.clippy]`、`[workspace.lints.rust]` 完整镜像为 harness 的 `[lints.clippy]`、`[lints.rust]`。因此以下 Clippy 补验应用工作区相同 lint 集合，并通过 `-D warnings` 将 warning 升为错误；不再仅检查上面的默认 lint 集合。

```sh
cargo clippy --offline --manifest-path /private/tmp/prism-editor-filesystem-lints-Sbol9j/Cargo.toml --target-dir /private/tmp/prism-editor-filesystem-lints-Sbol9j/target --lib --test filesystem -- -D warnings
cargo test --offline --manifest-path /private/tmp/prism-editor-filesystem-lints-Sbol9j/Cargo.toml --target-dir /private/tmp/prism-editor-filesystem-lints-Sbol9j/target --test filesystem
rustc --extern alloc --edition 2024 --crate-type lib --crate-name prism_filesystem_validation -F unsafe_code -D warnings -W missing_docs -W unused_qualifications pkg/prism_editor_content/src/filesystem.rs -o /private/tmp/prism-editor-filesystem-lints-Sbol9j/libprism_filesystem_validation.rlib
```

结果：严格 Clippy、28/28 文件系统集成测试、独立 rustc 及指定文件的格式/空白检查均通过。未重新运行 main 的其他测试，main 报告的 49 项总数不是本次独立复验范围。补验目录和 target 仅属于本任务，完成后删除；没有执行全仓构建。
