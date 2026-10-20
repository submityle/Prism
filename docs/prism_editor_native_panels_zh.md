# 原生作者与诊断面板

本接入位于 `pkg/prism_editor_app`。`EditorDesktopUi` 挂载真实 Loom 子树与现有 UI controller；它不持有 Gameplay 类型，也不推断项目字段或把按钮意图转换成磁盘成功。默认 CLI 不凭空创建项目时间轴、世界或采集结果，宿主需要显式挂载实际数据。

## 宿主契约

`mount_native_panel(id, label, Box<dyn DesktopPanel>)` 转移实际对象所有权；`native_panel<T>` 读取原对象，`native_panel_mut<T>` 在清除指针预览并校验文本草稿围栏后提供可变访问。操作后刷新会重建精确绑定。`unmount_native_panel` 仅在草稿、dirty、pending I/O 与错误围栏均解除后返回原对象。

卸载后若剩余面板重建失败，恢复原 slot、活动 tab 与原宿主对象，并返回可见错误；刷新失败不会丢弃被移除对象。调用方仍可从原 ID 取回相同宿主并修复问题。

| 面板 | 实际输入 | 宿主责任 | app 行为 |
| --- | --- | --- | --- |
| `DesktopTimelinePanel` | `TimelineEditor`、`TimelinePanelController` | 原子接收真实 `DesktopTimelineFrame`，持久化作者文档 | 路由作者/传输操作；单调时钟 advance；返回真实采样与跨越事件 |
| `DesktopWorldPanel` | `WorldPanelController`、非阻塞 provider | 消费真实 intent，保留 transition/save token，轮询真实 I/O，发布匹配的完成结果 | 显示真实 world 树/地图并路由操作；不制造完成回执 |
| `DesktopProfilerPanel` | `ProfilerController`、非阻塞 provider | 提供实际完成的 CPU/GPU `ProfilerSources` | 发布不可变 store 并路由 frame/span/domain 导航；CPU/GPU 时钟不混用 |
| `DesktopGraphPanel` | `GraphEditor`、宿主节点模板、视口和预算 | 提供领域无关的端口类型与 payload；显式绑定真实文件 | 实际节点/连线、指针预览、一次逆命令提交与独立存储 |
| `DesktopPrefabPanel` | `PrefabPanelController`、真实 store 与显式传播目标 | 提供实际资产/实例；确认精确持久化结果 | 路由选择、revert、apply review/confirm、传播与历史；不创建虚假资产 |
| `DesktopFrameGraphPanel` | `FrameGraphPanelController`、完成的 `Arc<GraphCaptureReport>` provider | 发布真实编译/执行采集；明确 capture 的证据等级 | 真实 pass/resource 导航，生产失败保留旧 publication 并围栏 |

provider 返回 `DesktopPanelUpdate { changed, fenced, wake_at }`。宿主必须用 `fenced` 表达自身未提交元数据、在途发布或尚未解决的 I/O；world 的排队 intent、dirty cell、pending residency 另由 app 直接检查。回调必须非阻塞；延迟工作应提交给宿主服务，并通过单调 `Instant` deadline 或唤醒原生事件循环继续轮询。provider 错误保留 controller/source，显示错误并围栏；宿主解决原因后通过 `retry` 恢复。

timeline 的 consumer 返回 `Err` 必须表示整帧没有被接受。app 保留完全相同的 tick、iteration、sample 与 event delivery，停止自动唤醒；显式 `retry` 后重投同一帧，成功后才清除。宿主不得先写入部分外部状态再返回错误。advance 错误保留旧时钟锚点；不存在丢弃 elapsed 后假称成功的路径。

timeline 编辑用精确 `TimelineSnapshotToken` 加 `ticks_per_second` 与保存基线比较。`acknowledge_saved(&token, ticks_per_second)` 只有两者均匹配当前状态才更新保存基线；只有宿主已经收到实际持久化成功回执才能调用。单独修改频率也会变 dirty。可使用下述显式 `TimelineStorageService` 绑定，不会把 scene 保存等同于 timeline 保存。

接入顺序是先创建真实宿主数据与 controller，再挂入 `EditorDesktopUi`，最后把该 UI 传给 `run_desktop` 或 `run_desktop_with_document`。下例的 editor、controller、host 均由调用方提供，`accept_frame` 必须满足上面的整帧原子契约：

```rust,ignore
use prism_editor_app::{DesktopTimelinePanel, EditorDesktopUi};

ui.mount_native_panel(
    "timeline",
    "Timeline",
    Box::new(DesktopTimelinePanel::new(editor, controller, move |frame| {
        host.accept_frame(frame)
    })),
)?;
```

面板契约与布局不依赖 desktop feature；原生窗口入口使用 `prism_editor_app/desktop`。app 已直接依赖 `prism_editor_timeline`，本接入不需要修改共享 UI/content/timeline 的 Cargo 或根导出。自定义事件循环调用 `poll_native_panels(now)`、`refresh()`，并把 `next_native_panel_wake()` 纳入自己的等待期限；有在途工作的 provider 应持续返回适当的 `wake_at`，直到取得真实完成回执。

## 挂载、布局和输入

底部 dock 在 content 与贡献面板之间切换，保留每个面板的滚动位置。最多八个 slot，ID 为不超过 48 字节的 ASCII 字母/数字/连字符，标签不超过 64 字节。贡献实际树限制 16,384 节点、128 深度、4,096 绑定；显式键不得重复或侵占 `editor.`、`content:`、`native:` 保留空间。不同 trigger 可共享同一语义 target，例如 world cell 的 Select/Activate。

命中、文本框与绘制共用 Loom 已计算的逻辑像素矩形和 pane clip；没有另造固定坐标控件。输入使用各面板真实 bound view 的精确绑定。文本草稿支持 IME、Enter/Tab 与 Escape；错误提交保留文本。面板切换不丢弃尚未解决的草稿。timeline、graph、prefab 活动时原生及工具栏 Undo/Redo 路由至其独立作者历史；只读诊断面板不会落入 scene 历史。Prefab review 确认前不改 store，确认只发布一笔逆命令，未确认 review 本身也围栏关闭/卸载。

`DesktopPanel` 允许额外实际 Loom 贡献，保留自己的绑定、poll、deadline、dirty 与作者版本。它不是空 custom 节点占位。

## 时间与指针手势

原生 `about_to_wait` 调用 `poll_native_panels(Instant::now())`，合并最早 deadline 至 `WaitUntil`。Playing 以 16ms 目标节奏推进，后台 tab 仍采样和投递；Paused/Stopped 不维持周期唤醒。时钟倒退显示错误并围栏，直到宿主处理。隐藏面板的数据变化不导致浏览器持续重建；没有变更的刷新不重建保留树。

真实 native 左键优先捕获 timeline 的 key/tangent，普通按钮仍经过 click 语义路由。key 命中来自 `timeline:key:{track}:{key}`，局部 x 原点取真实 `timeline:lane:{track}`；tangent 命中来自 `timeline:tangent:incoming/outgoing`，原点取真实 `timeline:curve`。坐标捕获后固定，移出 pane 仍可更新，不因 preview 重布局而漂移。

切线只允许上述两个精确 handle key 捕获；`timeline:tangent:*:stem:*` 是绘制连接线的装饰节点，没有拖动命中语义。

按下调用 UI 的 `begin_key_drag` / `begin_tangent_drag`，移动调用 `update_drag`，实际 Loom marker/curve 绘制 controller preview；模型与 Undo 历史在此期间不写入。release 先更新最终坐标，成功才 `commit_drag`，多选只有一笔作者事务。最终事件无效或没有可信坐标时取消，绝不提交先前有效预览。Escape、focus lost、resize、suspend、host mutable replacement 取消捕获。拖动期间禁止 tab/滚轮/其他键命令，并围栏 scene save/reload/close。

宿主使用公开 `panel_pointer_press/move/release/cancel_panel_pointer` 可复用相同路径；release 应提供当前逻辑坐标，平台取消或无法确认位置时传 `None`。宿主直接替换 timeline 文档前应先调用取消，或通过会自动取消的 `native_panel_mut` 获取模型。

graph 捕获使用真实 `graph:canvas` 的保留矩形及 clip，再由 `BoundGraphPanel::hit_test` 判断实际节点/端口；不把项目端口名硬编码进宿主。左键移动/多选/连线、中键平移及画布内滚轮缩放调用实际 controller，画布外滚轮继续滚动 dock。移动只改 preview，释放使用最终位置和最新 bound view 验证后写一笔真实作者命令。无效释放取消；原生中键不会与 scene 拖动或文档 I/O 同时捕获。

## 独立异步文件路由

`DesktopGraphPanel::mount_storage` 与 `DesktopTimelinePanel::mount_storage` 接收宿主指定的绝对路径、codec/字节预算、精确已观察文件 revision 与超时。构造只建立服务，实际 Save/Load 才排队后台工作；没有每帧克隆完整作者文档。保存显式编码一次快照，主循环按 10ms deadline 轮询在途请求；停止播放仍会收取文件回执。

原生 Save/Open 先检查 scene I/O 与 reconciliation，再路由活动面板的独立文件绑定；没有绑定时显示错误。工具栏 `graph:file:*` / `timeline:file:*` 提供保存、加载候选、取消、采用与丢弃。加载只发布隔离候选，不隐式替换现场或清除 dirty。未保存编辑存在时禁用原生“采用”；宿主可以读取精确当前模型 token 后显式 `adopt_loaded(&token)` 批准替换。该 token 必须同时匹配加载开始状态与当前状态，timeline 还检查加载开始的频率；加载期间的任何作者改动都要求重新加载。验证失败同时保留现场和候选。采用成功后重建真实 editor/controller，保留宿主节点模板或传输配置，清空历史；timeline 停止并向实际 consumer 投递新现场。

只有匹配所保存快照的 `Synced` 回执能更新保存基线；保存期间产生的新编辑或新频率仍保持 dirty。revision 冲突不覆盖磁盘。已发布但目录同步失败保留失败围栏并采用真实新 revision；`PublicationUncertain` 清除可写 revision，必须加载并明确协调后才能再次保存。取消后仍持有请求直到终态，`TooLate` 仍检查真实持久化结果，不伪造取消成功。候选、在途请求、失败、不确定 revision 与作者预览均围栏关闭、卸载及现场替换。

Sequence 作者面板沿用同一 `DesktopPanel` 贡献槽位与 `sequence:` 命名空间，宿主可接真实 controller/bound view；Newton 的 UI 导出完成前，本轮不把子序列作者或 runtime 执行宣称为 app 已交付。`DocumentStorage*` 通用类型已从 timeline 根公开，后续接线复用其真实存储能力。帧图当前接入真实 capture 导航，生产资源缩略预览的 GPU provider/图像桥接另行验收；编译图声明捕获不代表 GPU 已执行。

## 验证状态

此前版本的 8 项 app 面板测试已通过，覆盖实际 Loom 布局/语义路由、传输与作者 Undo/Redo、错误文本/IME 围栏、真实 world intent、贡献 slot、三面板切换与最小化。真实单调时钟测试记录 elapsed=49.740334ms、tick=4、sample=4、跨越 event=1；真实 `CpuProducer` 测试记录 frame duration=5,441,625ns。这些数值是该次执行证据，不是性能承诺。

前一提交 `eecaa827e` 的 `desktop_ui::panel_tests` 共 13 项全部通过，包含多选 key 手势、取消/无效 release、真实曲线原点与切线 stem 排除、时钟倒退、卸载失败保留所有权，以及树/绑定校验。该次默认 `cargo test -p prism_editor_app --all-targets` 通过 52 unit + 15 controller + 16 document，共 83 项。

使用主线程释放的 `/private/tmp/prism-editor-joint-cr1ODK`，在 `PRISM_REQUIRE_GPU=1`、Apple M2 / Metal 原生进程环境完成 app `--all-targets --features desktop`：70 unit + 2 content images + 15 controller + 18 document + 4 Gizmo，共 109 项通过；强制 GPU 不允许静默跳过。新增面板的证据仍是实际 Loom 布局、语义/指针路由与真实 CPU 采集；该 GPU 回归覆盖既有场景、Gizmo 和 PNG 内容绘制，不应称为新增三面板的 GPU 像素测试。

app `cargo clippy -p prism_editor_app --no-deps --all-targets --features desktop -- -D warnings` 通过。包含依赖的严格检查曾被非 app 的 `prism_math/src/ops.rs:1` 文档格式阻塞，该文件由其他任务维护；这里只声明 app 的严格检查。限定文件 rustfmt、`git diff --check` 通过。自有 `/private/tmp/prism-app-panels.Ylp1B0` 已用 `cargo clean` 回收 432.0MiB；joint target 仍由主线程管理，不清理他人的缓存。

collections/Git 并发错误已捕获结构化证据。旧测试二进制中，collections 两项独跑 100 轮通过，排除 Git 的 47 项跑 50 轮通过；五项并行约第十轮失败。诊断版五项 100 轮与全量 50 项 200 轮通过，说明单次未复现不能证明修复。新增实际 Git spawn 与每轮 100 次保存/回读最新快照的测试后，六项并行在第 42 轮失败，精确错误为 `CollectionStorageError::Busy`，保存路径位于该测试独占的 `.prism/content-collections.bin`。失败约半秒，与十秒测试期限无关。已确认问题是把暂态锁竞争直接变成终态失败围栏；锁竞争与进程创建相关，但未直接证明操作系统内部的描述符继承路径。

修复仅在后台 writer 对结构化 `Busy` 做最多八次尝试、总计 126ms 退避等待；UI 不阻塞，串行队列继续保留最新快照。持续占锁仍报告带文件路径和错误变体的终态失败，并要求显式重试；损坏文件、路径错误及其他 I/O 错误不自动重试。实际文件锁回归覆盖短暂占锁释放、持续占锁保留旧字节/dirty 围栏、停止每帧重试，以及释放锁后显式保存最新快照。没有延长测试期限或把失败吞成成功。

修复后同一原生测试二进制的 collections/Git 七项并行回归连续 100 轮通过（session 45948，exit 0），其中每轮包含实际 Git 进程创建与 100 次集合快照保存，最终回读验证最新快照。该压力回归与真实文件锁失败/恢复测试共同覆盖暂态重试和持续占锁两条路径；有限轮次并不意味着已经证明所有进程创建场景下绝无竞争。

当前新增面板证据是实际 retained 布局/输入路由与 CPU 数据采集，尚无本轮新增的原生面板 GPU 像素或解锁桌面 OS 输入验收结果。此前 native smoke 在锁屏环境只完成 M2/Metal 初始化，0 presented frame、0 native input 后达到时间上限，不能据此宣称原生展示或交互验收完成。

本轮 Graph/Timeline 文件与作者面板接入新增 `desktop_ui::authoring_panel_tests` 九项真实回归：节点移动/最终释放/取消/UndoRedo，中键平移/滚轮缩放/宿主替换取消，宿主类型端口连线单逆事务，真实文件保存/重载/陈旧回执与候选替换围栏，真实外部 revision 冲突和在途取消，频率独立于作者 token 的保存/采用检查，Prefab 实际选择/revert/历史，Prefab review→confirm→Undo/Redo，以及生产 `render_fabric` 编译捕获导航与 provider 失败保留 publication。Prefab Undo 校验实际组合后的基值与子层 pin；Undo 是新发布，不错误要求版本头回到旧字节或旧精确 token。帧图用例是无资源 compute 声明编译捕获，未执行 GPU 工作。

本轮最终验证复用 shared target，`jobs=2`、dev/test `debug=0`、`incremental=0`，没有建立另一个 target：

- 默认 `cargo test -p prism_editor_app --all-targets`：61 unit + 15 controller + 16 document，92 项通过，session 48290 exit 0，日志 `/private/tmp/prism-app-default-check.log`。
- 原生环境 `PRISM_REQUIRE_GPU=1 cargo test -p prism_editor_app --all-targets --features desktop`：79 unit + 2 content images + 15 controller + 18 document + 4 Gizmo，118 项通过，session 10449 exit 0，日志 `/private/tmp/prism-app-desktop-native-check.log`。沙箱首次强制 GPU 检查因 Metal adapter 不可见而失败，原生环境重跑全项成功；不把失败重写为跳过。新增九项仍是作者/文件/布局路由证据，GPU 像素证据来自既有 scene/Gizmo/PNG 用例。
- `cargo clippy -p prism_editor_app --no-deps --all-targets --features desktop -- -D warnings`：session 31470 exit 0，日志 `/private/tmp/prism-app-authoring-clippy.log`。依赖仍有其他任务的警告，不宣称全仓零警告。

当前没有自有活动验证进程或待清理自有 target；共享 `/private/tmp/prism-editor-joint-cr1ODK` 保留给主线程，不执行清理。生产帧图 GPU runtime preview 的 provider 与图像桥接、Sequence 根导出后的专用宿主接线仍单独验收；本轮不将它们计入完成项。
