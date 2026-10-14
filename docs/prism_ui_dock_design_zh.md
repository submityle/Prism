# Prism 停靠系统完整设计规格（`prism_ui_dock`）

> v1.2 · 面向 Loom Studio 与任意 Loom 工具应用的可停靠工作台内核。
>
> 本文定义布局模型、交互语义、渲染接缝、持久化格式、版本转换、性能、扩展、可达性、窗口恢复、可靠性和验收方法。源码中的纯数据模型、确定性 reducer、`DockHost`、codec、缓存和可选 a11y 投影是本文的实施真相。

## 1. 目标与职责

`prism_ui_dock` 提供：

- 主停靠树、浮动窗口、自动隐藏栏和最大化恢复快照；
- 纯函数 reducer、完整校验、不变量维护和结构化错误；
- renderer-independent `RenderPlan`、surface、tab、splitter、focus metadata 和 drop preview；
- `render` feature 下的 `DockView`、Panel/Document factory 和 `DockPanel` 装配；
- `PanelRegistry`、默认布局、Workspace/Perspective 和 host-driven autosave；
- v1 稳定二进制 layout/workspace codec、资源限制、注册表过滤和显式 schema migration；
- `DockHistory`、`DockHost`、精确输入 RenderPlan cache、DPI/monitor placement restore；
- `a11y` feature 下的 tab、panel、document、autohide 语义树和键盘动作解析。

上层窗口与 renderer 只消费 `DockHost` 的布局、计划和 Element 结果。上层不得直接修改 `DockLayout`；所有状态变化都转换为 `Action`，由 reducer 一次性验证并提交。

停靠内核不保存业务对象、GPU 句柄、平台窗口对象或插件实例；它保存稳定 ID、布局状态和可复现的几何输入。

## 2. 核心原则

1. 布局是 `no_std + alloc` 纯数据，和 renderer、窗口系统、业务 ECS 解耦。
2. UI 产生意图，reducer 负责状态转移；相同初始状态和 action 序列得到相同布局。
3. codec 把磁盘、云同步和插件输入视为不可信数据，在分配前检查长度、数量、深度、UTF-8、枚举、比例和尾随字节。
4. 面板使用稳定 ID；descriptor、factory、Element 和 GPU 资源分别由各自层拥有。
5. 拖拽预览不改变模型；只有合法 Drop action 才改变布局。
6. 几何、Element、DrawList、rasterizer 和 window present 使用独立数据层，便于测试、替换和性能定位。
7. active tab 和 active auto-hide reveal 懒构建；inactive tab 只生成标题元数据。
8. 所有失败都返回结构化错误，失败 action 不改变输入布局。

## 3. 布局模型

```rust
pub enum DockNode {
    Split { axis: Axis, children: Vec<SplitChild> },
    TabStack { tabs: Vec<PanelId>, active: usize },
    Document { tabs: Vec<DocumentId>, active: usize },
    Empty,
}

pub struct SplitChild {
    pub ratio: Ratio,
    pub node: DockNode,
}

pub struct DockLayout {
    pub root: DockNode,
    pub floating: Vec<FloatingWindow>,
    pub autohide: AutoHideBars,
    pub maximized: Option<Box<MaximizeRestore>>,
}

pub struct FloatingWindow {
    pub rect: Rect,
    pub root: DockNode,
    pub z: u32,
}
```

规则：

- `Split` 至少有两个子节点；同轴相邻 split 在 reducer 收敛时合并。
- `TabStack` 是普通面板的叶子；单面板仍使用单元素 tab stack。
- `Document` 是文档叶子；普通 panel 不能与 document tabify。
- `Empty` 只作为空主根或过滤过程中的中间值，提交、编码、解算前会收敛。
- panel/document ID 在主树、浮窗和 autohide 中只出现一次。
- floating 按 `z` 升序绘制；同 z 使用数组顺序稳定排序。
- maximize 保存完整旧布局；最大化状态下当前可见面严格只有目标 panel。

`Ratio` 使用百万分固定点。整数像素使用最大余数法分配，顺序稳定；最小尺寸无法满足时返回 `RenderPlanError`，不静默钳制。

## 4. 校验与 reducer

`DockLayout::validate_with` 在 reducer、codec、默认布局、RenderPlan 输入处执行：

- ratio 为正并总和为 `Ratio::SCALE`；
- active 索引有效；
- panel/document 唯一归属；
- split 没有空子、单子或非法比例；
-浮窗 rect 具有正宽高且 z 唯一；
- autohide active 必须指向现有 entry；
- maximize 快照和当前可见树相互一致；
- 深度、节点、面板、文档和浮窗数量不超过 `Limits`。

公开 action 包括 `Split`、`Tabify`、`Move`、`ActivateTab`、`Close`、`Resize`、`Float`、`Dock`、`Maximize`、`Restore`、`AutoHide` 和 `ShowAutoHide`。

reducer 使用候选 clone、动作应用、normalize、完整 validate、一次性返回的事务流程。关闭最后 tab、空浮窗、单子 split、拖拽取消和未知 ID 过滤都有确定收敛结果。

`ActivateTab` 是键盘和鼠标 tab 选择的统一 reducer 入口；它只修改目标 stack 的 active 索引。

## 5. 拖拽与停靠

```text
Idle -> PickUp -> Dragging -> Drop -> Idle
                         \-> Cancel -> Idle
```

- 中心生成 `CenterTab`；上、下、左、右生成 split zone。
- tab header 生成 `TabInsert(index)`，索引范围为 `[0, tabs.len]`。
- Document 目标只接受 split；Panel 目标接受中心、四边和 tab insertion。
- floating 目标按 z 逆序命中，再在窗口内部按 RenderPlan 命中。
- `resolve_drop_preview` 与 `map_drop_action` 使用相同的目标几何和 zone 数据。
- `drop_preview_element` 只绘制 absolute preview，不注册事件、不抢焦点、不修改布局。
- preview host 必须在输入树中设置 `pointer-events:none` 或从 hit-test 树排除；`prism_ui_input::PointerEvents::None` 可直接表达该语义。
- Esc、非法区域和窗口失焦都回到原状态。

## 6. DockHost、历史和 frame pipeline

`DockHost` 是应用接入停靠状态的唯一聚合器：

1. 保存当前 `DockLayout`；
2. 通过 `DockHistory` 执行 reducer action；
3. 成功提交后递增 revision 并清理 RenderPlan cache；
4. 通过 `RenderPlanCache` 求解 exact layout/viewport/metrics 输入；
5. 向 Element、a11y、window 和 renderer adapter 提供只读结果。

历史采用固定容量 snapshot：新 action 清空 redo 分支；undo/redo 不访问 factory、文件或 GPU；容量为零直接返回结构化错误。

标准帧顺序为：

```text
input snapshot
 -> hit test / gesture
 -> preview 或 Action
 -> DockHost::dispatch
 -> RenderPlanCache::get_or_solve
 -> DockView::render
 -> retained layout tree
 -> DrawList
 -> CPU reference 或 wgpu rasterizer
 -> window surface present
```

## 7. RenderPlan 与 Element

`RenderPlan` 输出 main/floating/autohide/maximized surface、panel/document rect、tab header、splitter line/hit rect、visibility、focus target 和 size diagnostics。solver 只做整数几何，不创建窗口、Element 或 GPU 资源。

`DockView` 通过 `PanelViewRegistry`、`DocumentViewRegistry` 调用 factory：

- active tab 调用一次 factory；inactive tab 只保留标题；
- active auto-hide reveal 调用对应 factory；
- floating 按 z 顺序装配；
- `DockPanel` 负责玻璃表面、标题栏、内容容器和 DockSide 样式；
- unknown ID、missing factory、重复注册和非法布局返回 `DockViewError`；
- factory 失败不返回部分 Element。

Render backend 的 `ElementRenderer` 将 Element 更新、布局和 DrawList 生成统一到 retained host；CPU rasterizer 和可选 wgpu rasterizer 共享 DrawList 契约。

## 8. 持久化与 schema migration

layout 使用 `PRDK`，workspace 使用 `PRWS`；两个格式都采用 little-endian、magic、version、checked count、UTF-8、枚举和严格尾部检查。当前 canonical version 都为 `1`。

`MigrationRegistry` 与 `WorkspaceMigrationRegistry` 只接受单调递增、来源版本唯一的迁移边；迁移函数是纯 bytes-to-bytes 转换。迁移流程为：读取 header → 查找版本边 → 执行转换 → 检查目标版本 → canonical decode → 完整 validate。缺少迁移边、目标版本错误和非法输出都会返回结构化错误。

`WorkspaceFileStore::load_with_migration` 与 `load_with_registry_and_migration` 在 primary/last-good/default 回退前执行显式迁移，因此文件加载、registry 过滤和版本转换使用同一条安全链。

registry-aware decode 的顺序为：移除未知 ID、修复 active、移除空分支、重算比例、重新检查唯一归属和 maximize 状态。未知 ID 不会调用 factory。

文件保存顺序为：sibling advisory lock → encode/validate → unique temp → flush/sync → last-good → atomic replace primary → parent directory sync → unlock。autosave 使用单调时间 debounce；写入失败保留 dirty 状态。

## 9. Registry 与扩展

core registry 只保存 `PanelDescriptor`：stable ID、标题、图标、实例策略、默认槽位。render registry 保存 factory。factory 只能通过只读 context 读取布局、主题、instance ID 和诊断信息，不能直接修改布局。

扩展约束：注册幂等、ID 稳定、descriptor 可版本化、factory 生命周期可回收、卸载前关闭实例并重新运行 registry filter/default layout。`Instancing::Multi` 允许宿主建立多实例；默认布局为每个 descriptor 生成一个默认实例。

## 10. 性能与容量

- validate、reducer、solver、drop preview、codec 的主要复杂度分别为 `O(N)`、`O(N)`、`O(N+P+S)`、`O(P+S)`、`O(B)`。
- 默认 Limits 限制深度、节点、tab、panel、document、浮窗、字符串和编码字节数；应用只能收紧限制。
- `RenderPlanCache` 以 layout fingerprint、viewport、LayoutMetrics 为完整 key；layout、viewport、metrics 任意变化都会 miss。
- `DockHost` 在 action 提交后清除 cache，避免 stale plan。
- tab header 可使用 `prism_ui_virtual` 窗口化；active、keyboard neighbor、drag source、drop target 保留优先级。
- factory 可使用 generation token 异步构建；旧 generation 的结果不得覆盖新布局。
- 宿主记录 solve ns、Element update ns、DrawList command count、visible count、factory count、GPU submission ns、present latency 和 dropped frames，并输出 P50/P95/P99。
- 同一 manager 的提交顺序串行；不同 Workspace 的 encode/validate 可并行。

## 11. 视觉、动画和主题

`DockPanel` 使用主题 token；drop preview 使用中心/边缘区分和对比度合格的边框；split、dock、maximize、restore、reveal 动画只插值 RenderPlan 几何，不修改模型。

动画档位包含 disabled、reduced-motion、normal 和 high-quality。新 action 到达时，当前插值状态成为新时间线起点，避免动画对象累积。HDR、主题和 DPI 只影响渲染输入，不改变布局 ratio。

## 12. 可达性与窗口恢复

启用 `a11y` feature 后，`DockA11ySnapshot::build` 生成 stable tab、panel、document 和 autohide semantic keys，并计算 `FocusOrder`。`resolve` 使用 `KeyboardNav`：Tab/Shift-Tab 移焦，Enter 激活 tab，Escape 返回 dismiss action，tab 的 selected 状态与 RenderPlan 同步。

tab activation 转换为 `Action::ActivateTab`；FocusTrap 由 overlay host 用于 modal/floating context。语义树不持有 Element 或布局可变引用。

`FloatingPlacement` 记录 monitor ID、逻辑 rect 和 scale milli。`restore_placement` 按首选 monitor → 第一可用 monitor 选择目标，按新旧 scale 缩放，并将窗口 clamp 到 work area，同时保证最小可见像素。显示器消失不会改变 dock tree。

逻辑布局使用逻辑像素；window backend 负责物理像素、surface scale、HDR、refresh rate 和 present mode 转换。

## 13. 安全与可靠性

- 布局字符串只作为 ID，不执行任何内容。
- 所有长度先 checked，再分配；递归深度和集合数量先检查。
- 非法 ratio、rect、enum、UTF-8、active、maximize、duplicate ownership、trailing bytes 均拒绝。
- reducer、codec、solver、factory 装配使用全有或全无结果，不产生半成品状态。
- load 返回来源和 primary/last-good 诊断；default 只在两份快照均不可用时使用。
- 诊断默认只记录 ID、revision、surface 和计数，不记录用户文件内容。

## 14. Crate 结构与 feature

```text
pkg/prism_ui_dock/src/
├── model.rs / validate.rs / reducer.rs
├── drag.rs / drop_preview.rs / drop_preview_view.rs
├── render_plan.rs / render_cache.rs / frame.rs
├── view.rs / registry.rs / default_layout.rs
├── history.rs / display.rs / migration.rs
├── persist.rs / workspace.rs / workspace_persist.rs
└── workspace_store.rs / autosave.rs
```

features：`std` 提供 fs4 文件存储和 autosave；`render` 提供 Element 装配；`a11y` 依赖 `render` 并提供 DockA11ySnapshot。关闭 default features 后，model、validate、reducer、drag、solver、codec、workspace、history、display、migration 和 cache 保持 `no_std + alloc`。

## 15. 验收矩阵与验证记录

| 层级 | 结果 |
|---|---|
| model/validate/reducer | ratio、唯一归属、Document、浮窗、maximize、ActivateTab 和错误不变性测试通过 |
| drag/drop/preview | 状态机、五向 zone、tab insertion、preview geometry 测试通过 |
| render plan/view | 固定点舍入、最小尺寸、z 顺序、active factory、floating、autohide 测试通过 |
| persistence/store | layout/workspace round-trip、截断、尾随、未知 ID、migration、lock、last-good、并发保存测试通过 |
| history/host/cache | undo/redo、branch clearing、revision、cache hit/miss 测试通过 |
| a11y/display | tab semantic、keyboard activation、DPI 缩放、monitor fallback、可见区域 clamp 测试通过 |
| CPU/GPU | Dock Element DrawList 与真实 wgpu rasterizer parity 测试通过 |

2026 年 10 月 14 日本机记录：

- `cargo check -p prism_ui_dock --no-default-features`：通过；
- `cargo test -p prism_ui_dock --features render`：116 passed；
- `cargo test -p prism_ui_dock --features a11y`：117 passed；
- `cargo test -p prism_ui_render_backend --features gpu --test dock_gpu_parity -- --nocapture`：1 passed，真实 GPU adapter 执行。

## 16. 版本与维护规则

layout、workspace、RenderPlan 和 Rust API 版本分开维护。wire 变化必须增加版本、提供迁移函数、添加旧版/损坏/round-trip fixture、更新 `pkg/prism_ui_dock/docs/version/`、运行 no-default/render/a11y/GPU 验证并执行 `git diff --check`。

一个领域只允许一个状态拥有者：布局归 reducer/model，几何归 solver/cache，语义归 a11y snapshot，窗口归 window backend，GPU 资源归 render backend。适配器只做转换，不复制第二份可变状态。

## 17. 术语

- **DockNode**：主区或浮窗中的布局节点。
- **DockLayout**：主树、浮窗、autohide 和 maximize 快照。
- **RenderPlan**：布局到几何、surface、focus 和命中数据的确定性中间表示。
- **DockHost**：布局、历史、revision 和 RenderPlan cache 的唯一聚合器。
- **Workspace/Perspective**：命名的布局快照及活动选择。
- **DropZone**：拖拽指针对应的中心、边缘或 tab 插入区域。
- **last-good**：最近一次成功验证并提交的 Workspace 快照。

本文与编辑器外壳、DockPanel、Loom layout/overlay/store/input/virtual 和 `prism_window` capability 文档协同。Prism crate 只借鉴公开架构形态和经典数值，不包含 Unreal Engine、Unity、VS Code 或 JetBrains 的源码及衍生代码。
