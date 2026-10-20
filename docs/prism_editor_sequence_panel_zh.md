# Sequence project 作者面板

`prism_editor_ui` 的 `SequencePanelController` / `BoundSequencePanel` 操作真实
`SequenceProjectEditor`，包含多个拥有独立整数时钟的 `TimelineDocument` 和完整
subsequence clip 拓扑。面板不持有项目副本，不直接写文档；宿主持有 editor，所有作者操作
提交 `SequenceProjectCommand`，沿用 kernel 的逆命令 Undo/Redo。无新增依赖或 Bevy 依赖。

## 实际功能

- 创建、打开、删除文档，设置项目根和文档 ticks/second；被引用文档的删除由 kernel 原子拒绝。
- 按稳定 `SequenceId` / 全项目唯一 `SequenceClipId` 浏览子序列和父引用；定位会切换到
  clip 所属文档及正确页，打开 child 会进入真实引用目标。
- clip 草稿包含 source ID、parent/source 半开区间、正整数比例分子/分母和重复次数。
  字段输入先只改草稿；Add/Apply 一次提交完整 clip。非法区间、零比例、环等拒绝后
  项目与 history 不产生部分改动，下一棵树展示实际错误。
- 根区间、seek、elapsed nanoseconds 和根 Once/Loop 模式通过实际输入节点设置；
  Evaluate 调用真实 `SequencePlayer::new/seek/play/advance`。
- 事件诊断保留真实 sequence/event ID、事件本地 tick、映射 root tick、root iteration、
  clip/iteration 实例路径和事件名；采样诊断保留真实目标 tick、track 数与实例路径。
  两类诊断均能打开文档、定位最后一级嵌套 clip。没有合成事件或伪图形占位。

Evaluate 是一次显式区间诊断，不是持续运行的原生播放时钟。seek 本身不发 crossed events；
随后 elapsed advance 才遍历边界、重复和 root loop。时钟映射、裁剪、溢出检查和事件排序
完全由 sequence kernel 执行。面板不执行 camera/audio provider，也不冒充运行时后端。

## API 与宿主接点

根导出由主线程接线，无 Cargo 变更：

```rust,ignore
SequencePanelController, SequencePanelAction, SequencePanelError, SequencePanelLimits,
SequenceClipDraft, SequenceClipField, SequencePreview, SequencePage,
SequenceEventDiagnostic, SequenceSampleDiagnostic, BoundSequencePanel
```

宿主保留一份 editor、controller 和当前 bound；加载候选经 host 接受后替换 editor，
重新 build，不能继续提交旧 bound 的事件。最小调用顺序：

```rust,ignore
let mut controller = SequencePanelController::default();
let bound = controller.build(&editor)?;
ui.mount(bound.element());
// target 是实际命中节点 key，event 是宿主转换后的 UiEvent。
controller.handle_event(&mut editor, &bound, target, &event)?;
let bound = controller.build(&editor)?;
ui.update(bound.element());
```

`BoundSequencePanel::bindings()` 返回真实节点的 key/semantic trigger。所有键使用
`sequence:` 命名空间；项目撤销/重做键固定为 `sequence:undo` / `sequence:redo`。
`element()` 是真实 box/text/SearchField Loom 树，按钮有实际背景，未启用按钮没有绑定。
`dispatch()` 仅解释输入；原生事件应走 `handle_event()`，由它验证 controller 身份及
精确 `SequenceProjectToken`。相同 revision 的另一 editor、Undo ABA、UI 选择或草稿变化
均不能复用旧 bound。无变化时重建相同 Element，RecordingBackend 重协调零操作。

`BoundSequencePanel::snapshot_token()` 可作为宿主异步保存确认的快照证据。持久化继续由
host 的 `SequenceProjectStorageService` 完成：保存拥有的编码数据和 token，完成后验证
token，不能把旧保存完成信号误标记为当前编辑状态已保存。面板不做文件 I/O。

原生挂载由 Erdos 负责，通过实际 `DesktopPanel` / `mount_native_panel` 接入宿主所拥有
的真实 editor；该切片提供 Element、绑定和控制器路径，不宣称原生窗口挂载已经验证。
单文档 timeline 面板和项目存储不在此切片内改动。

## 预算与失效处理

`SequencePanelLimits` 在构造时验证：每种列表独立分页，每页 1–32 行；诊断事件与采样
各最多 1–128 行；实际 registry traversal 最多 1–65536 visits。默认 12 行、128 条
诊断和 4096 visits。文档/clip 数、嵌套深度仍受项目 kernel policy 限制；评估将 events、
track samples、visits 上限再与项目限额取最小值。kernel 的 max_samples 统计 track，
所以面板另外检查 instance sample 行数，避免空 track 文档绕过输出上限。超过预算
明确报错并清空旧诊断，不静默截断为成功。

文档和 clip 列表使用 kernel 确定性顺序；五种页分别管理，不把行索引当资产身份。
事件名仅为显示截取前 120 个字符，真实 ID 和实例路径保持完整。source ID 输入支持
完整 u128，整数输入溢出拒绝。不存在外部项目更新后的幽灵选择：build 检测精确快照
变化，清理旧诊断、已删除文档/clip 选择和无效草稿 source，夹紧页面。

## 验证与资源归属

`tests/sequence_panel.rs` 从 crate 根导入生产 API，在真实 checkout 上运行；不复制模块
或构造替代 UI。测试宿主挂载实际 Element，检查绑定均有唯一节点、计算布局、向实际 key
投递 semantic event，执行 controller → kernel → 更新树，并检查 RecordingBackend 操作。
另用 RetainedScene/rasterize 检查全部绑定具有非零布局及按钮实际背景像素。

回归包括真实两级嵌套 fixture：root/child/leaf 时钟 2/4/8 Hz，parent/source 区间、
1/2 局部比例和重复；检查真实 leaf tick、跨越事件、root loop、seek 零事件、
可定位实例路径；另覆盖作者命令与逆历史、无环约束、有界分页、大 ID、错误输入、
身份/快照围栏、外部引用删除、空 track 输出限额和不变树零重协调。

验证复用 `/private/tmp/prism-editor-joint-cr1ODK`，jobs=2，debug=0，incremental=0：

```sh
CARGO_TARGET_DIR=/private/tmp/prism-editor-joint-cr1ODK CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 cargo test -p prism_editor_ui --test sequence_panel --offline -j 2
CARGO_TARGET_DIR=/private/tmp/prism-editor-joint-cr1ODK CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 cargo clippy -p prism_editor_ui --test sequence_panel --offline -j 2 --no-deps -- -D warnings
```

本次真实 checkout 上 11 项测试全部通过；targeted strict Clippy exit 0。依赖包已有
warning 未在此切片修改，strict 使用 `--no-deps -- -D warnings`。

该 target 是活跃共享资源，面板任务未创建私有 target，不能清理共享目录。限定提交
哈希由交付报告给出；原生挂载/项目路径/异步存储完成确认仍需宿主联合验证。
