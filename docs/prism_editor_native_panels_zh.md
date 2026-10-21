# 原生作者与诊断面板

本接入位于 `pkg/prism_editor_app`。`EditorDesktopUi` 挂载真实 Loom 子树与现有 UI controller；它不持有 Gameplay 类型，也不推断项目字段或把按钮意图转换成磁盘成功。默认 CLI 不凭空创建项目时间轴、世界或采集结果，宿主需要显式挂载实际数据。

## 宿主契约

`mount_native_panel(id, label, Box<dyn DesktopPanel>)` 转移实际对象所有权；`native_panel<T>` 读取原对象，`native_panel_mut<T>` 在清除指针预览并校验文本草稿围栏后提供可变访问。操作后刷新会重建精确绑定。`unmount_native_panel` 仅在草稿、dirty、pending I/O 与错误围栏均解除后返回原对象。

卸载后若剩余面板重建失败，恢复原 slot、活动 tab 与原宿主对象，并返回可见错误；刷新失败不会丢弃被移除对象。调用方仍可从原 ID 取回相同宿主并修复问题。

| 面板 | 实际输入 | 宿主责任 | app 行为 |
| --- | --- | --- | --- |
| `DesktopTimelinePanel` | `TimelineEditor`、`TimelinePanelController` | 原子接收真实 `DesktopTimelineFrame`，持久化作者文档 | 路由作者/传输操作；单调时钟 advance；返回真实采样与跨越事件 |
| `DesktopWorldPanel` | `WorldPanelController`、非阻塞 provider | 消费 cell intent，保留 transition/save token 并发布匹配的完成结果 | 每次 panel poll 同时轮询真实 project storage；project dirty、I/O pending、loaded candidate 与 retired world 均参与全局 fence；不制造完成回执 |
| `DesktopProfilerPanel` | `ProfilerController`、非阻塞 provider | 提供实际完成的 CPU/GPU `ProfilerSources` | 发布不可变 store 并路由 frame/span/domain 导航；CPU/GPU 时钟不混用 |
| `DesktopGraphPanel` | `GraphEditor`、宿主节点模板、视口和预算 | 提供领域无关的端口类型与 payload；显式绑定真实文件 | 实际节点/连线、指针预览、一次逆命令提交与独立存储 |
| `DesktopPrefabPanel` | `PrefabPanelController`、真实 store 与显式传播目标 | 提供实际资产/实例；确认精确持久化结果 | 路由选择、revert、apply review/confirm、传播与历史；不创建虚假资产 |
| `DesktopFrameGraphPanel` | `FrameGraphPanelController`、完成的 `Arc<GraphCaptureReport>` provider；可选 `DesktopRuntimePreviewSource` | 发布真实编译/执行采集；runtime source 必须交给 app 同步提供实际 `RuntimePreviewFrame` | 真实 pass/resource 导航；生产 capture 按 reserve-before-submit、bind、poll/cancel/drain 接入，并将真实 RGBA8 publication 交给共享 `ImageRenderer` |
| `DesktopSequencePanel` | `SequenceProjectEditor`、显式 `SequencePanelLimits` | 提供真实完整项目、时钟/拓扑预算及独立文件绑定 | 子文档/clip 作者、一次逆命令、嵌套诊断与完整项目异步存储 |

provider 返回 `DesktopPanelUpdate { changed, fenced, wake_at }`。宿主必须用 `fenced` 表达自身未提交元数据、在途发布或尚未解决的 I/O；world 的排队 intent、dirty cell、pending residency 与 project 生命周期状态由 app 直接检查。回调必须非阻塞；延迟工作应提交给宿主服务，并通过单调 `Instant` deadline 或唤醒原生事件循环继续轮询。provider 错误保留 controller/source，显示错误并围栏；宿主解决原因后通过 `retry` 恢复。

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

命中、文本框与绘制共用 Loom 已计算的逻辑像素矩形和 pane clip；没有另造固定坐标控件。输入使用各面板真实 bound view 的精确绑定。文本草稿支持 IME、Enter/Tab 与 Escape；错误提交保留文本。面板切换不丢弃尚未解决的草稿。timeline、graph、prefab、sequence 活动时原生及工具栏 Undo/Redo 路由至其独立作者历史；只读诊断面板不会落入 scene 历史。Prefab review 确认前不改 store，确认只发布一笔逆命令，未确认 review 本身也围栏关闭/卸载。

`DesktopPanel` 允许额外实际 Loom 贡献，保留自己的绑定、poll、deadline、dirty 与作者版本。它不是空 custom 节点占位。

## 时间与指针手势

原生 `about_to_wait` 调用 `poll_native_panels(Instant::now())`，合并最早 deadline 至 `WaitUntil`。Playing 以 16ms 目标节奏推进，后台 tab 仍采样和投递；Paused/Stopped 不维持周期唤醒。时钟倒退显示错误并围栏，直到宿主处理。隐藏面板的数据变化不导致浏览器持续重建；没有变更的刷新不重建保留树。

真实 native 左键优先捕获 timeline 的 key/tangent，普通按钮仍经过 click 语义路由。key 命中来自 `timeline:key:{track}:{key}`，局部 x 原点取真实 `timeline:lane:{track}`；tangent 命中来自 `timeline:tangent:incoming/outgoing`，原点取真实 `timeline:curve`。坐标捕获后固定，移出 pane 仍可更新，不因 preview 重布局而漂移。

切线只允许上述两个精确 handle key 捕获；`timeline:tangent:*:stem:*` 是绘制连接线的装饰节点，没有拖动命中语义。

按下调用 UI 的 `begin_key_drag` / `begin_tangent_drag`，移动调用 `update_drag`，实际 Loom marker/curve 绘制 controller preview；模型与 Undo 历史在此期间不写入。release 先更新最终坐标，成功才 `commit_drag`，多选只有一笔作者事务。最终事件无效或没有可信坐标时取消，绝不提交先前有效预览。Escape、focus lost、resize、suspend、host mutable replacement 取消捕获。拖动期间禁止 tab/滚轮/其他键命令，并围栏 scene save/reload/close。

宿主使用公开 `panel_pointer_press/move/release/cancel_panel_pointer` 可复用相同路径；release 应提供当前逻辑坐标，平台取消或无法确认位置时传 `None`。宿主直接替换 timeline 文档前应先调用取消，或通过会自动取消的 `native_panel_mut` 获取模型。

graph 捕获使用真实 `graph:canvas` 的保留矩形及 clip，再由 `BoundGraphPanel::hit_test` 判断实际节点/端口；不把项目端口名硬编码进宿主。左键移动/多选/连线、中键平移及画布内滚轮缩放调用实际 controller，画布外滚轮继续滚动 dock。移动只改 preview，释放使用最终位置和最新 bound view 验证后写一笔真实作者命令。无效释放取消；原生中键不会与 scene 拖动或文档 I/O 同时捕获。

## 独立异步文件路由

`DesktopGraphPanel::mount_storage` 与 `DesktopTimelinePanel::mount_storage` 接收宿主指定的绝对路径、codec/字节预算、精确已观察文件 revision 与超时。构造只建立服务，实际 Save/Load 才排队后台工作；没有每帧克隆完整作者文档。保存显式编码一次快照，主循环按 10ms deadline 轮询在途请求；停止播放仍会收取文件回执。Graph/Timeline/Sequence 的结构化 `Busy` 仅在发布前重试，最多八次尝试，退避 1/2/4/8/16/32/64ms，始终受原始绝对截止时间限制。等待期间通过单调 deadline 唤醒，不在 UI 线程 sleep；保留原始编码字节、模型 token 与磁盘 revision，不重新捕获更新后的现场。冲突、取消、codec/path 错误及发布不确定均不自动重试。耗尽后保留带绑定路径的可见失败围栏并停止唤醒，要求显式 Save/Load；等待重试期间取消直接移除计划，没有遗留 worker 能在后续唤醒时发布。

原生 Save/Open 先检查 scene I/O 与 reconciliation，再路由活动面板的独立文件绑定；没有绑定时显示错误。工具栏 `graph:file:*` / `timeline:file:*` 提供保存、加载候选、取消、采用与丢弃。加载只发布隔离候选，不隐式替换现场或清除 dirty。未保存编辑存在时禁用原生“采用”；宿主可以读取精确当前模型 token 后显式 `adopt_loaded(&token)` 批准替换。该 token 必须同时匹配加载开始状态与当前状态，timeline 还检查加载开始的频率；加载期间的任何作者改动都要求重新加载。验证失败同时保留现场和候选。采用成功后重建真实 editor/controller，保留宿主节点模板或传输配置，清空历史；timeline 停止并向实际 consumer 投递新现场。

只有匹配所保存快照的 `Synced` 回执能更新保存基线；保存期间产生的新编辑或新频率仍保持 dirty。revision 冲突不覆盖磁盘。已发布但目录同步失败保留失败围栏并采用真实新 revision；`PublicationUncertain` 清除可写 revision，必须加载并明确协调后才能再次保存。取消后仍持有请求直到终态，`TooLate` 仍检查真实持久化结果，不伪造取消成功。候选、在途请求、失败、不确定 revision 与作者预览均围栏关闭、卸载及现场替换。

`DesktopSequencePanel::new(editor, limits)` 接收实际完整项目与显式 UI 预算，持有真实 `SequencePanelController` 和当前 `BoundSequencePanel`。所有作者事件最终调用 `handle_event` 校验 controller 身份与精确项目 token，包含错误输入；错误保留在真实 controller 并围栏关闭/保存。原生 Undo/Redo 操作完整项目历史。clip 数字输入仅暂存，`SaveClip`/`AddClip` 才写一次真实逆命令。未应用草稿阻止丢弃草稿的导航/历史操作、存储与卸载；显式“Discard clip draft / clear rejected action”还原实际选中 clip 或新 clip 基线，不把作者模型或文件错误标为已保存。检查只读 clip 元数据，不在轮询中克隆时间轴 payload。

Sequence 的 `mount_storage(path, DesktopSequenceStorageLimits, observed_revision, timeout)` 复用实际 `SequenceProjectStorageService`。`document` 是完整项目 codec/拓扑/历史预算，`max_requests` 是 worker 准入预算；没有新增共享依赖。Save 编码一次完整项目，包含所有子文档、频率、轨道/事件、clip 拓扑及 ID 高水位。Open 取得隔离候选，复用同一 pending/cancel/durability/revision 围栏；采用同时验证加载开始 token、当前 token、草稿与 controller 错误，再构造独立 editor/controller，成功后一次替换并清空历史及诊断。失败保留现场与候选。`acknowledge_saved` 只认可精确项目 token；改变任一文档频率同样改变 token，旧保存回执不能清除新编辑。

Sequence `Evaluate` 显式运行真实嵌套求值器并显示实际样本/事件诊断，未挂接持续 transport，也不调用 Gameplay 或写入 runtime。没有在途文件请求时不会安排周期唤醒。`DocumentStorage*` 通用类型已从 timeline 根公开。帧图生产预览的 capture 不是截图或手工 TexturePreview：宿主通过 `DesktopRuntimePreviewSource` 提供当前真实队列借用的 `RuntimePreviewFrame`，app 在同一调用中依据当前 pass key/resource slot 找到精确 binding，先调用 preview_mut().reserve(...)，再用 reservation 的 binding、mip/layer 与 capture_limits() 调用 RuntimePreviewFrame::submit，最后 bind_submission。提交前失败只 abandon；bind 拒绝则保留原 `RuntimePreviewSubmission` 并继续取消/轮询，直到 staging slots、bytes 和 graph completion 都归零。

poll_native_panels 对隐藏面板也驱动 poll_submission，取消只隔离 publication，不释放 GPU admission；新的报告或选择会取消旧 request 并等待 drain。next_native_panel_wake 使用单调 deadline。成功 publication 的 `FrameGraphPreviewImage` 是 controller 的精确 image Arc；原生 `DesktopContentImages` 通过 viewport 的同一个 `ImageRenderer` 先 upload 后替换旧 handle，通道/曝光/range 生成新 image Arc 会重新上传，单纯 zoom/pan/DPI 几何变化不会上传，image_quad 应用实际 canvas/clip 与 DPI 后再和缩略图一起 Load-composite 到 caller target。resident custom 节点只是有界布局锚点，像素证据必须来自 compositor 的实际 target 读回。

`DesktopRuntimePreviewSource` 是显式 host-owned seam：app 不创建 queue、report、Gameplay 或伪造运行时 frame；source 负责让 `RuntimePreviewFrame<'queue>` 在 callback 中立即提交。source 必须在失败前不提交 frame，并保持 imported resources 的宿主所有权。若 host 需要把 frame 跨 callback 保存，当前 profiler API 不允许将借用 queue 的 frame 存为 'static，应在 host 内同步完成 callback，不能复制 report 来绕过 exact-Arc 校验。

## 验证状态

此前版本的 8 项 app 面板测试已通过，覆盖实际 Loom 布局/语义路由、传输与作者 Undo/Redo、错误文本/IME 围栏、真实 world intent、贡献 slot、三面板切换与最小化。真实单调时钟测试记录 elapsed=49.740334ms、tick=4、sample=4、跨越 event=1；真实 `CpuProducer` 测试记录 frame duration=5,441,625ns。这些数值是该次执行证据，不是性能承诺。

前一提交 `eecaa827e` 的 `desktop_ui::panel_tests` 共 13 项全部通过，包含多选 key 手势、取消/无效 release、真实曲线原点与切线 stem 排除、时钟倒退、卸载失败保留所有权，以及树/绑定校验。该次默认 `cargo test -p prism_editor_app --all-targets` 通过 52 unit + 15 controller + 16 document，共 83 项。

使用主线程释放的 `/private/tmp/prism-editor-joint-cr1ODK`，在 `PRISM_REQUIRE_GPU=1`、Apple M2 / Metal 原生进程环境完成 app `--all-targets --features desktop`：70 unit + 2 content images + 15 controller + 18 document + 4 Gizmo，共 109 项通过；强制 GPU 不允许静默跳过。新增面板的证据仍是实际 Loom 布局、语义/指针路由与真实 CPU 采集；该 GPU 回归覆盖既有场景、Gizmo 和 PNG 内容绘制，不应称为新增三面板的 GPU 像素测试。

app `cargo clippy -p prism_editor_app --no-deps --all-targets --features desktop -- -D warnings` 通过。包含依赖的严格检查曾被非 app 的 `prism_math/src/ops.rs:1` 文档格式阻塞，该文件由其他任务维护；这里只声明 app 的严格检查。限定文件 rustfmt、`git diff --check` 通过。自有 `/private/tmp/prism-app-panels.Ylp1B0` 已用 `cargo clean` 回收 432.0MiB；joint target 仍由主线程管理，不清理他人的缓存。

collections/Git 并发错误已捕获结构化证据。旧测试二进制中，collections 两项独跑 100 轮通过，排除 Git 的 47 项跑 50 轮通过；五项并行约第十轮失败。诊断版五项 100 轮与全量 50 项 200 轮通过，说明单次未复现不能证明修复。新增实际 Git spawn 与每轮 100 次保存/回读最新快照的测试后，六项并行在第 42 轮失败，精确错误为 `CollectionStorageError::Busy`，保存路径位于该测试独占的 `.prism/content-collections.bin`。失败约半秒，与十秒测试期限无关。已确认问题是把暂态锁竞争直接变成终态失败围栏；锁竞争与进程创建相关，但未直接证明操作系统内部的描述符继承路径。

修复仅在后台 writer 对结构化 `Busy` 做最多八次尝试、总计 126ms 退避等待；UI 不阻塞，串行队列继续保留最新快照。持续占锁仍报告带文件路径和错误变体的终态失败，并要求显式重试；损坏文件、路径错误及其他 I/O 错误不自动重试。实际文件锁回归覆盖短暂占锁释放、持续占锁保留旧字节/dirty 围栏、停止每帧重试，以及释放锁后显式保存最新快照。没有延长测试期限或把失败吞成成功。

修复后同一原生测试二进制的 collections/Git 七项并行回归连续 100 轮通过（session 45948，exit 0），其中每轮包含实际 Git 进程创建与 100 次集合快照保存，最终回读验证最新快照。该压力回归与真实文件锁失败/恢复测试共同覆盖暂态重试和持续占锁两条路径；有限轮次并不意味着已经证明所有进程创建场景下绝无竞争。

当前新增面板证据是实际 retained 布局/输入路由与 CPU 数据采集；生产预览接线还通过共享 viewport `ImageRenderer` 完成了真实 M2/Metal runtime framegraph 像素回归。app-owned fixture 使用 host-owned imported `Rgba8Unorm` texture（4×2、2 layers、2 mips），在真实 pass binding 上执行 reserve → `RuntimePreviewFrame::submit` → bind → poll；随后对实际 ImageRenderer target readback 的首像素为 `[0,255,0,255]`，缩放重排未产生第二次 texture upload。第二次 capture 的 submission 经过 cancel 与 GPU drainage 后没有发布半帧；新的 runtime report 选择会清除旧 image，这是 controller 的显式 publication/selection 围栏，不以旧像素冒充新 frame。强制 GPU targeted test 与 app all-targets 均在 Apple M2/Metal 原生 adapter 上通过。此前 native smoke 在锁屏环境只完成 M2/Metal 初始化，0 presented frame、0 native input 后达到时间上限，不能据此宣称原生展示或交互验收完成。

本轮 Graph/Timeline 文件与作者面板接入新增 `desktop_ui::authoring_panel_tests` 九项真实回归：节点移动/最终释放/取消/UndoRedo，中键平移/滚轮缩放/宿主替换取消，宿主类型端口连线单逆事务，真实文件保存/重载/陈旧回执与候选替换围栏，真实外部 revision 冲突和在途取消，频率独立于作者 token 的保存/采用检查，Prefab 实际选择/revert/历史，Prefab review→confirm→Undo/Redo，以及生产 `render_fabric` 编译捕获导航与 provider 失败保留 publication。Prefab Undo 校验实际组合后的基值与子层 pin；Undo 是新发布，不错误要求版本头回到旧字节或旧精确 token。帧图用例是无资源 compute 声明编译捕获，未执行 GPU 工作。

本轮最终验证复用 shared target，`jobs=2`、dev/test `debug=0`、`incremental=0`，没有建立另一个 target：

- 默认 `cargo test -p prism_editor_app --all-targets`：61 unit + 15 controller + 16 document，92 项通过，session 48290 exit 0，日志 `/private/tmp/prism-app-default-check.log`。
- 原生环境 `PRISM_REQUIRE_GPU=1 cargo test -p prism_editor_app --all-targets --features desktop`：79 unit + 2 content images + 15 controller + 18 document + 4 Gizmo，118 项通过，session 10449 exit 0，日志 `/private/tmp/prism-app-desktop-native-check.log`。沙箱首次强制 GPU 检查因 Metal adapter 不可见而失败，原生环境重跑全项成功；不把失败重写为跳过。新增九项仍是作者/文件/布局路由证据，GPU 像素证据来自既有 scene/Gizmo/PNG 用例。
- `cargo clippy -p prism_editor_app --no-deps --all-targets --features desktop -- -D warnings`：session 31470 exit 0，日志 `/private/tmp/prism-app-authoring-clippy.log`。依赖仍有其他任务的警告，不宣称全仓零警告。

上述 Graph/Timeline 提交结束时没有自有活动验证进程或待清理自有 target；共享 `/private/tmp/prism-editor-joint-cr1ODK` 保留给主线程，不执行清理。生产帧图 GPU runtime preview 的 provider 与图像桥接仍单独验收。World membership 的完整项目保存/采用需要 UI controller 提供精确围栏的历史/同步/采用入口，现有 provider/intent 接线不等于此能力。

本次 runtime preview 接线的定向真实测试为 `desktop_ui::runtime_preview_tests::native_runtime_capture_drains_and_composites_real_frame_pixels`：Apple M2/Metal、`1 passed`。同一配置的 `cargo test -p prism_editor_app --all-targets --features runtime-preview` 为 88 unit + 2 content-image + 15 controller + 18 document + 4 Gizmo，共 127 项通过；main/examples 无测试也正常结束。之后复用同一 shared target 的 app-only strict Clippy 未到达 app 检查，因共享 target 写入 `serde_core` 临时文件时 `No space left on device` 退出 101；这不是 app lint 诊断，也不宣称本次改动已通过该轮 Clippy。共享 target 仍由主线程管理，不在本任务中清理。

Sequence 本轮新增六项真实 app 回归，定向 session 84029 exit 0，日志 `/private/tmp/prism-app-sequence-check.log`。覆盖实际 retained 节点非零命中矩形、原生 wheel/点击/文本/Enter、草稿应用单逆事务与 Undo/Redo，错误输入经过真实 controller 并保留围栏，陈旧 controller/project bound 拒绝，完整项目真实文件 round trip，异步旧回执/加载后新编辑/草稿拒绝采用，外部文件冲突与取消直到请求终态。真实三层项目 root/child/leaf 频率 2/4/8Hz，显式 elapsed=2 秒得到 leaf tick=16；leaf event=11 映射至 root tick=2，clip 路径为 `[20,30]`。求值前后项目 token 不变，无持续 transport 唤醒。这些是 CPU/文件/保留 UI 路由证据，不是 Sequence runtime 或新增 GPU 渲染验收。

该次全量默认测试 session 44452 曾在 Graph 加载和 Timeline 保存出现真实 `Busy` 失败；未通过延长期限或串行测试掩盖。结构化有限重试修复后再新增两项实际 `.prism-timeline.lock` 回归：短暂占锁时现场继续编辑，释放后磁盘仍是原保存快照且旧回执不能清除新频率的 dirty；持续占锁耗尽后保留路径/Busy/围栏且无周期唤醒，显式重试并取消等待计划后释放锁也不写文件，最终显式保存恢复。该轮没有捕获足以证明具体 OS 锁继承根因的证据，不把有限重试称作系统性消除所有竞争。

Sequence 与重试修复的最终默认 `cargo test --offline -p prism_editor_app --all-targets` 为 session 42691 exit 0：69 unit + 15 controller + 16 document，共 100 项通过，日志 `/private/tmp/prism-app-sequence-default.log`。八项 Sequence 测试全部通过。复用共享 target、jobs=2、debug=0、incremental=0，没有新建自有 target。磁盘紧急指令后未启动新编译；后续尝试直接复用测试二进制时，该文件已不存在，因此计划中的重复并发循环未执行，本轮 desktop strict Clippy 也未重跑。上文 118 项强制 GPU 与 strict Clippy 属于此前 Graph/Timeline 提交，不能冒充本轮 Sequence 的验收结果。

本轮结束时无 app 活跃验证进程或可清理自有 target，共享 `/private/tmp/prism-editor-joint-cr1ODK` 未删除。Sequence runtime 的显式实体绑定、整帧预检/应用与 publication barrier 由独立任务提供，app 尚未挂接，等待正式 API 及真实 DSP/ViewData 证据；这里的 Evaluate 只表示实际 kernel 诊断。World membership controller/history/storage 候选入口也等待 UI 所有者交接，不能把现有 intent/provider 路由记作完整项目持久化。

磁盘恢复并获准验证后，补跑相同共享 target/config 的 `cargo clippy --offline -p prism_editor_app --no-deps --all-targets --features desktop -- -D warnings`，session 96266 exit 101。日志 `/private/tmp/prism-app-sequence-clippy.log` 显示检查尚未到 app，被并行开发中的 `prism_editor_ui/src/world.rs:375`（const wrapper 调用非 const 的 `PendingWorldTransaction::request`，E0015）和 `world.rs:549`（新增 Project actions 尚未匹配，E0004）阻断；未修改其他任务的源文件，不宣称本轮 app strict Clippy 已通过。Sequence 代码已限定提交 `33913731f`，需要 World 所有者补齐后再联合复检。
