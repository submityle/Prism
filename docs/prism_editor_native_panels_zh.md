# 原生 timeline / world / profiler 面板

本接入位于 `pkg/prism_editor_app`。`EditorDesktopUi` 挂载真实 Loom 子树与现有 UI controller；它不持有 Gameplay 类型，也不推断项目字段或把按钮意图转换成磁盘成功。默认 CLI 不凭空创建项目时间轴、世界或采集结果，宿主需要显式挂载实际数据。

## 宿主契约

`mount_native_panel(id, label, Box<dyn DesktopPanel>)` 转移实际对象所有权；`native_panel<T>` 读取原对象，`native_panel_mut<T>` 在清除指针预览并校验文本草稿围栏后提供可变访问。操作后刷新会重建精确绑定。`unmount_native_panel` 仅在草稿、dirty、pending I/O 与错误围栏均解除后返回原对象。

卸载后若剩余面板重建失败，恢复原 slot、活动 tab 与原宿主对象，并返回可见错误；刷新失败不会丢弃被移除对象。调用方仍可从原 ID 取回相同宿主并修复问题。

| 面板 | 实际输入 | 宿主责任 | app 行为 |
| --- | --- | --- | --- |
| `DesktopTimelinePanel` | `TimelineEditor`、`TimelinePanelController` | 原子接收真实 `DesktopTimelineFrame`，持久化作者文档 | 路由作者/传输操作；单调时钟 advance；返回真实采样与跨越事件 |
| `DesktopWorldPanel` | `WorldPanelController`、非阻塞 provider | 消费真实 intent，保留 transition/save token，轮询真实 I/O，发布匹配的完成结果 | 显示真实 world 树/地图并路由操作；不制造完成回执 |
| `DesktopProfilerPanel` | `ProfilerController`、非阻塞 provider | 提供实际完成的 CPU/GPU `ProfilerSources` | 发布不可变 store 并路由 frame/span/domain 导航；CPU/GPU 时钟不混用 |

provider 返回 `DesktopPanelUpdate { changed, fenced, wake_at }`。宿主必须用 `fenced` 表达自身未提交元数据、在途发布或尚未解决的 I/O；world 的排队 intent、dirty cell、pending residency 另由 app 直接检查。回调必须非阻塞；延迟工作应提交给宿主服务，并通过单调 `Instant` deadline 或唤醒原生事件循环继续轮询。provider 错误保留 controller/source，显示错误并围栏；宿主解决原因后通过 `retry` 恢复。

timeline 的 consumer 返回 `Err` 必须表示整帧没有被接受。app 保留完全相同的 tick、iteration、sample 与 event delivery，停止自动唤醒；显式 `retry` 后重投同一帧，成功后才清除。宿主不得先写入部分外部状态再返回错误。advance 错误保留旧时钟锚点；不存在丢弃 elapsed 后假称成功的路径。

timeline 编辑用精确 `TimelineSnapshotToken` 与保存基线比较。`acknowledge_saved(&token)` 只有匹配当前文档才解除围栏；只有宿主已经收到实际持久化成功回执才能调用。`TimelineStorageService` 的提交、超时、不确定持久性及 revision 冲突由宿主管理，本面板不会自动把 scene 保存等同于 timeline 保存。

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

命中、文本框与绘制共用 Loom 已计算的逻辑像素矩形和 pane clip；没有另造固定坐标控件。输入继续使用真实 `BoundTimelinePanel`、`BoundWorldPanel`、`BoundProfilerPanel` 的精确绑定。文本草稿支持 IME、Enter/Tab 与 Escape；错误提交保留文本。面板切换不丢弃尚未解决的草稿。timeline 活动时原生及工具栏 Undo/Redo 路由至时间轴历史；world/profiler 活动时不会落入 scene 历史。

`DesktopPanel` 允许额外实际 Loom 贡献，保留自己的绑定、poll、deadline、dirty 与作者版本。它不是空 custom 节点占位。

## 时间与指针手势

原生 `about_to_wait` 调用 `poll_native_panels(Instant::now())`，合并最早 deadline 至 `WaitUntil`。Playing 以 16ms 目标节奏推进，后台 tab 仍采样和投递；Paused/Stopped 不维持周期唤醒。时钟倒退显示错误并围栏，直到宿主处理。隐藏面板的数据变化不导致浏览器持续重建；没有变更的刷新不重建保留树。

真实 native 左键优先捕获 timeline 的 key/tangent，普通按钮仍经过 click 语义路由。key 命中来自 `timeline:key:{track}:{key}`，局部 x 原点取真实 `timeline:lane:{track}`；tangent 命中来自 `timeline:tangent:incoming/outgoing`，原点取真实 `timeline:curve`。坐标捕获后固定，移出 pane 仍可更新，不因 preview 重布局而漂移。

切线只允许上述两个精确 handle key 捕获；`timeline:tangent:*:stem:*` 是绘制连接线的装饰节点，没有拖动命中语义。

按下调用 UI 的 `begin_key_drag` / `begin_tangent_drag`，移动调用 `update_drag`，实际 Loom marker/curve 绘制 controller preview；模型与 Undo 历史在此期间不写入。release 先更新最终坐标，成功才 `commit_drag`，多选只有一笔作者事务。最终事件无效或没有可信坐标时取消，绝不提交先前有效预览。Escape、focus lost、resize、suspend、host mutable replacement 取消捕获。拖动期间禁止 tab/滚轮/其他键命令，并围栏 scene save/reload/close。

宿主使用公开 `panel_pointer_press/move/release/cancel_panel_pointer` 可复用相同路径；release 应提供当前逻辑坐标，平台取消或无法确认位置时传 `None`。宿主直接替换 timeline 文档前应先调用取消，或通过会自动取消的 `native_panel_mut` 获取模型。

## 验证状态

此前版本的 8 项 app 面板测试已通过，覆盖实际 Loom 布局/语义路由、传输与作者 Undo/Redo、错误文本/IME 围栏、真实 world intent、贡献 slot、三面板切换与最小化。真实单调时钟测试记录 elapsed=49.740334ms、tick=4、sample=4、跨越 event=1；真实 `CpuProducer` 测试记录 frame duration=5,441,625ns。这些数值是该次执行证据，不是性能承诺。

当前 `desktop_ui::panel_tests` 共 13 项已全部通过，包含多选 key 手势、取消/无效 release、真实曲线原点与切线 stem 排除、时钟倒退、卸载失败保留所有权，以及树/绑定校验。默认 `cargo test -p prism_editor_app --all-targets` 通过 52 unit + 15 controller + 16 document，共 83 项。

使用主线程释放的 `/private/tmp/prism-editor-joint-cr1ODK`，在 `PRISM_REQUIRE_GPU=1`、Apple M2 / Metal 原生进程环境完成 app `--all-targets --features desktop`：70 unit + 2 content images + 15 controller + 18 document + 4 Gizmo，共 109 项通过；强制 GPU 不允许静默跳过。新增面板的证据仍是实际 Loom 布局、语义/指针路由与真实 CPU 采集；该 GPU 回归覆盖既有场景、Gizmo 和 PNG 内容绘制，不应称为新增三面板的 GPU 像素测试。

app `cargo clippy -p prism_editor_app --no-deps --all-targets --features desktop -- -D warnings` 通过。包含依赖的严格检查曾被非 app 的 `prism_math/src/ops.rs:1` 文档格式阻塞，该文件由其他任务维护；这里只声明 app 的严格检查。限定文件 rustfmt、`git diff --check` 通过。自有 `/private/tmp/prism-app-panels.Ylp1B0` 已用 `cargo clean` 回收 432.0MiB；joint target 仍由主线程管理，不清理他人的缓存。

collections/Git 并发错误已捕获结构化证据。旧测试二进制中，collections 两项独跑 100 轮通过，排除 Git 的 47 项跑 50 轮通过；五项并行约第十轮失败。诊断版五项 100 轮与全量 50 项 200 轮通过，说明单次未复现不能证明修复。新增实际 Git spawn 与每轮 100 次保存/回读最新快照的测试后，六项并行在第 42 轮失败，精确错误为 `CollectionStorageError::Busy`，保存路径位于该测试独占的 `.prism/content-collections.bin`。失败约半秒，与十秒测试期限无关。已确认问题是把暂态锁竞争直接变成终态失败围栏；锁竞争与进程创建相关，但未直接证明操作系统内部的描述符继承路径。

修复仅在后台 writer 对结构化 `Busy` 做最多八次尝试、总计 126ms 退避等待；UI 不阻塞，串行队列继续保留最新快照。持续占锁仍报告带文件路径和错误变体的终态失败，并要求显式重试；损坏文件、路径错误及其他 I/O 错误不自动重试。实际文件锁回归覆盖短暂占锁释放、持续占锁保留旧字节/dirty 围栏、停止每帧重试，以及释放锁后显式保存最新快照。没有延长测试期限或把失败吞成成功。

修复后同一原生测试二进制的 collections/Git 七项并行回归连续 100 轮通过（session 45948，exit 0），其中每轮包含实际 Git 进程创建与 100 次集合快照保存，最终回读验证最新快照。该压力回归与真实文件锁失败/恢复测试共同覆盖暂态重试和持续占锁两条路径；有限轮次并不意味着已经证明所有进程创建场景下绝无竞争。

当前新增面板证据是实际 retained 布局/输入路由与 CPU 数据采集，尚无本轮新增的原生面板 GPU 像素或解锁桌面 OS 输入验收结果。此前 native smoke 在锁屏环境只完成 M2/Metal 初始化，0 presented frame、0 native input 后达到时间上限，不能据此宣称原生展示或交互验收完成。
