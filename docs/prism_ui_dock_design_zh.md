# Prism 停靠系统完整设计规格（`prism_ui_dock`）

> v1.1 · 面向 Loom Studio 与任意 Loom 工具应用的可停靠工作台内核。
>
> 本文同时定义数据模型、交互语义、渲染接缝、持久化格式、性能预算、扩展协议、可达性、可靠性与验证方式。`prism_ui_dock` 的行为以纯数据模型和确定性 reducer 为真相源；窗口、GPU surface、输入分发和宿主线程调度通过明确的适配协议接入。

## 1. 目标与职责

`prism_ui_dock` 负责：

- 可序列化的主布局、浮动布局、自动隐藏栏和最大化恢复状态；
- 纯函数 reducer、校验、不变量维护和结构化错误；
- 渲染无关的几何计划、焦点元数据、分隔条命中矩形和 drop preview；
- `render` feature 下从注册表到 `Element` 的确定性装配；
- 面板/文档贡献点、默认布局物化、Workspace/Perspective 管理；
- 稳定二进制 codec、资源上限、注册表过滤、原子文件保存与 debounce autosave；
- 可脱离 GPU 单测的拖拽状态机与 reducer action 映射。

宿主负责把这些确定性产物接入窗口、surface、输入树、portal、焦点管理、命令栈和应用级 frame submission。宿主不直接修改 `DockLayout`，所有改变都转换为 `Action` 并提交 reducer。

不引入业务类型、GPU 句柄、平台窗口对象或第三方编辑器代码；面板内容由注册表 factory 提供，停靠内核只保存稳定 ID 和几何/交互状态。

## 2. 产品形态与原则

参考 VS Code、Unreal Slate、Dear ImGui docking、JetBrains Tool Window、Blender Area、Qt ADS、Zed/Figma Pane Group 的公开架构形态，采用以下原则：

1. 布局是 `no_std + alloc` 纯数据，和 renderer、窗口系统、业务 ECS 解耦。
2. UI 只产生意图，reducer 负责状态转移；相同初始状态与 action 序列必须得到字节级等价布局。
3. 所有输入均视为不可信数据；codec 在分配前检查长度、数量、深度、UTF-8、枚举、比例、唯一归属和尾随字节。
4. 面板采用稳定 ID；标题、图标、默认槽位、实例策略和 factory 分离，插件可以增加面板而无需修改内核。
5. 预览过程不改变布局；只有合法 Drop action 才提交结构变更。
6. 几何、文本、动画和 GPU 提交使用不同的数据层，方便测试、替换和性能定位。
7. 面板内容按 active/reveal 状态懒构建；隐藏 tab 不占用内容 view 生命周期。
8. 所有错误都返回结构化值；不会通过比例钳制、静默丢节点或隐式默认化掩盖损坏状态。

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

### 3.1 节点规则

- `Split` 的子项数量至少为 2；同轴相邻 split 在 reducer 收敛时合并，保留稳定顺序。
- `TabStack` 是普通面板的唯一叶子载体；单个面板仍以单元素 tab 栈表示。
- `Document` 是文档叶子，工具面板可以环绕它，但工具面板不能与文档 tabify。
- `Empty` 只允许作为空主根或过滤后的空浮窗中间态；提交 reducer、编码和渲染前会收敛到合法形态。
- 一个面板实例只能出现在主树、一个浮窗或一个自动隐藏条中的一个位置。
- `floating` 按 `z` 升序绘制；相同 `z` 使用向量顺序作为稳定 tie-breaker。
- `maximized` 保存最大化前的完整 `DockLayout` 快照和目标面板 ID；最大化期间可见面严格只有目标面板。

### 3.2 比例与几何

`Ratio` 使用固定点正整数。每次 split/resize 都以固定总量归一，整数像素分配使用最大余数法，余数按子节点顺序稳定分配。比例不因尺寸不足而静默钳制；当最小尺寸约束无法满足时，solver 返回 `RenderPlanError`。

## 4. 不变量与收敛

`validate` 在 reducer 输出、decode 输出、默认布局物化和 render plan 输入处执行同一组规则：

- 所有 ratio 为正且总量正确；
- 所有 tab active 索引有效；
- 所有 panel/document ID 在各表面只归属一次；
- split 不包含空子节点、单子节点或非法轴；
- 浮窗 rect 具有正宽高，浮窗根是合法节点；
- autohide entry 只出现一次，active reveal 必须指向现有 entry；
- maximize 快照、目标面板和当前可见树相互一致；
- 面板数量、文档数量、节点数量、深度、浮窗数量和编码字节数不超过 `Limits`；
- Document 不接受普通 Panel 的 tabify，普通 Panel 不会被误当作 Document。

关闭最后一个 tab、浮窗清空、split 只剩一个子、拖拽取消和过滤未知 ID 都有确定收敛结果：空节点被移除，比例重算，active 索引重定位，状态不会停留在半成品。

## 5. Action 与 reducer

```rust
pub enum Action {
    Split { target: Target, axis: Axis, placement: Placement },
    Tabify { panel: PanelId, target: Target },
    Move { panel: PanelId, destination: Target, placement: Placement },
    Resize { split: NodePath, ratios: Vec<Ratio> },
    Float { panel: PanelId, rect: Rect },
    Dock { floating_index: usize, target: Target, placement: Placement },
    Close { panel: PanelId },
    AutoHide { panel: PanelId, side: Side },
    RevealAutoHide { panel: PanelId },
    HideAutoHide,
    Maximize { target: Target },
    Restore,
}
```

每个 action 遵循“先验证输入、计算候选布局、完整校验、一次性提交”的事务语义。失败返回 `LayoutError`，原布局保持不变。reducer 不产生 IO、线程、GPU 操作或 factory 调用。

撤销/重做通过编辑器命令内核包装 action：命令保存 reducer 前后的 `DockLayout` 或压缩快照，执行、撤销、重做均重新运行校验。命令栈只拥有布局快照，不拥有面板 view 和窗口资源，因此不会造成资源泄漏。

## 6. 拖拽与停靠

```text
Idle
  -> PickUp(panel, origin)
  -> Dragging { ghost, hover, pointer }
  -> Drop(action)
  -> Idle

Dragging -> Cancel -> Idle
Dragging -> Reset -> Idle
```

### 6.1 命中与 drop zone

- 中心区生成 `CenterTab`；上、下、左、右生成 split zone。
- tab header 生成 `TabInsert(index)`，索引在 `[0, tabs.len]` 内。
- Document 目标只接受 split；Panel 目标接受中心、四边和 tab 插入。
- 主区四边保留整区 edge zone，用于没有 leaf 命中时的自动创建停靠 lane。
- floating window 目标先按 z 逆序命中，再按窗口内部计划命中；没有目标时可生成新的浮窗。
- 预览与实际 action 共享同一 `DropZone` 和 `RenderPlan`，避免“预览位置”和“提交位置”不一致。

### 6.2 预览与输入隔离

`resolve_drop_preview` 只消费 `RenderPlan` 和 `DropZone`，返回 renderer-independent `DropPreview`。`drop_preview_element` 生成绝对定位 Element，并带有固定 preview key。

宿主把 preview portal 放在内容树之后、输入分发树之外，或明确设置 `pointer-events:none`。真实 target 的 hit-test 在 preview 生成前完成；preview 不能抢焦点、触发 action 或修改布局。Esc、非法区域和窗口失焦都执行无副作用取消。

## 7. 解算、Element 与 frame pipeline

完整路径固定为：

```text
DockLayout
  -> validate
  -> render_plan::solve / solve_with
  -> RenderPlan
  -> DockView::render
  -> prism_ui::Element
  -> retained layout tree
  -> DrawList
  -> CPU reference 或 wgpu rasterizer
  -> window surface present
```

### 7.1 RenderPlan

`RenderPlan` 包含主区、浮窗、auto-hide bar/reveal、maximized surface、panel/document rect、tab header、splitter/hit rect、surface ID、visibility、focus metadata 和最小尺寸诊断。solver 不创建 Element、不访问窗口、不分配 GPU 资源。

### 7.2 DockView

`DockView` 通过 `PanelViewRegistry` 与 `DocumentViewRegistry` 调用 factory：

- active tab 调用一次 factory；inactive tab 只产生标题；
- active auto-hide reveal 调用对应 factory；折叠项只产生按钮；
- floating window 按稳定 z 顺序生成；
- `DockPanel` 只负责玻璃表面、标题栏、内容容器和 `DockSide` 样式；
- unknown ID、missing factory、非法布局和重复 view registration 均返回 `DockViewError`；
- factory 失败不会提交部分 Element，调用方可重试或显示错误面板。

### 7.3 Host frame contract

每帧 host 按以下顺序工作：读取输入快照、计算 hit-test、生成 action/preview、提交 action、求解 RenderPlan、更新 retained tree、生成 DrawList、提交 surface。窗口 resize、scale factor、HDR 和 surface lost 作为外部事件进入下一帧；布局模型始终使用逻辑像素。

`ElementRenderer` 是 retained draw-list host；它负责 Element update、布局、绘制命令和 CPU/wgpu 两个 raster path。应用级 surface/present 由 `prism_window` 与 renderer host 组合完成，dock crate 不持有窗口对象。

## 8. 持久化与 schema

### 8.1 v1 wire format

布局和 Workspace 使用稳定 little-endian 二进制格式：magic、version、checked counts、UTF-8 字符串、枚举、节点递归数据和严格尾部校验。encoder 与默认 decoder 使用同一组资源预算，保证“能写出的数据默认能读回”。

当前 canonical version 为 `1`。未知 version 返回 `UnsupportedVersion`，不会猜测字段含义。升级流程固定为：识别旧版本 → 迁移到内存中的 canonical model → 完整 `validate` → 使用当前 encoder 重写。迁移函数按版本独立注册，禁止在 decoder 内隐式改变语义；没有对应迁移函数时返回结构化错误并保留原字节诊断。

### 8.2 注册表过滤

`decode_layout_with_registry` 与 `decode_workspace_with_registry` 在模型构造后、renderer 消费前过滤未知 panel/document ID：

1. 从叶子移除未知项；
2. 修复 active 索引；
3. 移除空分支；
4. 重新归一剩余 split ratio；
5. 重新验证唯一归属和 maximize 状态；
6. 将过滤计数写入诊断结果。

过滤是安全降级，不会调用未注册 factory。

### 8.3 Workspace 与文件保存

`WorkspaceManager` 以插入顺序保存命名 Workspace/Perspective，切换和保存使用深拷贝隔离。`WorkspaceFileStore` 的保存事务为：获取 sibling advisory lock、编码并校验、写唯一临时文件、flush/sync、更新 last-good、原子替换 primary、同步父目录、释放锁。

加载顺序为 primary → last-good → caller default，并保留每次失败的结构化诊断。锁文件保持稳定路径，由 OS 在进程退出时释放持有锁；绕过该协议的外部写入不进入一致性保证范围。

`WorkspaceAutosave` 只维护 dirty 标记与单调时钟 deadline。宿主通过 `poll(now)` 或 `flush()` 调用保存；失败保留 dirty 状态，成功清除 dirty 状态。它不在 reducer 内阻塞，也不创建后台线程，worker/executor 由应用层决定。

## 9. 注册表与扩展协议

```rust
pub struct PanelDescriptor {
    pub id: PanelId,
    pub title: String,
    pub icon: Option<IconId>,
    pub instancing: Instancing,
    pub default_slot: DefaultSlot,
}

pub trait ViewFactory {
    fn build(&self, context: &ViewContext) -> Result<Element, ViewError>;
}
```

core registry 只保存纯数据。render registry 保存 factory；factory 通过 `ViewContext` 获取只读布局信息、主题 token、panel instance ID、诊断句柄和取消令牌，不能直接修改 `DockLayout`。

扩展必须满足：稳定 ID、幂等注册、版本化 descriptor、显式资源释放、可序列化状态与独立测试。插件卸载前先关闭其 panel instance，重新运行 registry filter 和 default layout policy，禁止遗留悬挂 ID。

默认布局按注册顺序把 descriptor 映射到 Left、Right、Bottom、Document、Floating 槽位；同槽位稳定合并为 tab stack，空槽位不生成非法节点。`Instancing::Multi` 允许宿主创建多个 instance，但默认布局只创建一个默认实例。

## 10. 性能与容量

### 10.1 复杂度

- validator：`O(N)` 节点遍历，附带 ID 集合检查；
- reducer：`O(N)` 候选布局复制与校验；
- render solver：`O(N + P + S)`，其中 N 为节点数、P 为可见 panel/document 数、S 为 splitter 数；
- drop preview：`O(P + S)`，只扫描当前 `RenderPlan`；
- codec：`O(B)`，B 为编码字节数；
- active view 装配：`O(V + H)`，V 为可见内容数，H 为可见标题数。

### 10.2 预算

默认 `Limits` 固定最大节点深度、panel/document 数、tab 数、浮窗数、字符串长度和总字节数。应用可以在创建 decode/render policy 时收紧预算，不允许放宽到整数溢出或无限分配。

### 10.3 变化量优化

当前完整 solver 和 owned clone 提供稳定基线；宿主缓存每个 `NodePath` 的 fingerprint、父 rect、min-size summary 和子计划，按 dirty path 重新计算局部 subtree，再更新祖先 rect。缓存必须以 layout revision、metrics revision、theme revision 和 registry revision 为 key，任何版本变化都使相关缓存失效。

拖拽期间只更新 pointer、hover 和 preview；Drop 前不重建内容 view。分隔条调整只更新受影响 split 的 ratio 与几何，提交时仍运行完整校验。

### 10.4 虚拟化与监控

Tab header 使用 `prism_ui_virtual` 做窗口化，active、keyboard neighbor、drag source 和 drop target 具有保留优先级。宿主每帧记录 layout solve ns、Element update ns、draw-list command count、visible panel count、factory build count、GPU submission ns、present latency 和 dropped frame count，按 P50/P95/P99 输出诊断。指标不改变布局结果。

### 10.5 内存与并发

core model 是单线程可移动纯数据；不同 Workspace 可并行 encode/validate，但同一 manager 的提交顺序由宿主串行化。factory 不在 reducer 线程同步执行；可异步构建的内容使用 loading Element 和 generation token，旧 generation 完成时丢弃结果，避免拖拽后的过期内容覆盖新布局。

## 11. 视觉、动画与主题

- `DockPanel` 使用主题 token 提供 frosted glass、separator、标题栏和 `DockSide`。
- drop preview 使用半透明填充、方向边框和明确的中心/边缘色彩，颜色同时满足明暗主题对比度要求。
- split、dock、maximize、restore、auto-hide reveal 使用 `prism_ui_anim`/`prism_ui_motion` 的时间线；动画只插值 RenderPlan 几何，不修改模型。
- 动画有 disabled、reduced-motion、normal、high-quality 四档；CI 使用 disabled 档并比较稳态 snapshot。
- HDR 使用线性颜色 token 和 surface capability 协商；布局几何始终保持逻辑像素。
- 动效中若收到新 action，旧时间线按当前插值状态合并到新目标，避免跳变和累积动画对象。

## 12. 可达性、DPI 与多显示器

### 12.1 语义树

DockView 为 root、surface、tablist、tab、tabpanel、splitter、autohide button、floating window 生成稳定语义 ID。每个 tab 暴露 selected、posinset、setsize、label；每个 splitter 暴露 orientation、value、min、max；Document 与 Panel 保留清晰 role。

### 12.2 键盘与焦点

- Tab/Shift-Tab 在当前 surface 的语义顺序内移动；
- Ctrl/Command+PageUp/PageDown 切换 tab；
- Ctrl/Command+Shift+方向键移动 panel；
- splitter 由方向键微调、PageUp/PageDown 大步调节；
- Esc 取消拖拽、关闭 reveal 或退出 maximize；
- 模态浮窗使用 `FocusTrap`，关闭后焦点回到启动它的 tab；
- focus ring 独立于 hover 样式，满足对比度和 reduced-motion 要求。

### 12.3 DPI 与窗口恢复

`FloatingWindow.rect` 使用逻辑像素。宿主保存 monitor stable ID、logical rect、scale factor、work area 和窗口状态；恢复时按以下顺序校正：匹配 monitor → clamp 到 work area → 按新 scale factor 转换 → 保证最小可见区域 → 提交窗口移动。显示器消失时自动迁移到主显示器，不修改 dock tree。

混合 DPI resize 只重新求解 RenderPlan，不改变 ratio。surface scale、HDR、refresh rate 和 present mode 变化通过 `prism_window` capability 事件进入 renderer host。

## 13. 安全与可靠性

- 不执行布局中的字符串；布局只包含 ID、枚举、数值和结构。
- 读入长度先经过 checked arithmetic，再进行分配；深度、数量、字节上限在递归前检查。
- 非法 ratio、rect、enum、UTF-8、active、maximize、duplicate ownership 和 trailing bytes 返回错误。
- 文件保存采用临时文件、flush/sync、last-good、原子替换和 advisory lock。
- 写入失败不清除 dirty；加载失败提供来源和诊断，不丢弃 caller default。
- reducer、codec、solver、view factory 都遵循全有或全无提交，不产生半成品状态。
- 运行期诊断使用不含用户内容的 panel ID、surface ID、revision 和计数；日志默认不记录路径中的敏感信息。

## 14. Crate 结构与公共 API

```text
pkg/prism_ui_dock/
├── src/model.rs             # DockNode、DockLayout、浮窗、autohide、maximize
├── src/validate.rs          # 不变量、Limits、结构化 LayoutError
├── src/reducer.rs           # Action 与确定性状态转移
├── src/drag.rs              # 拖拽状态机、drop zone、action 映射
├── src/drop_preview.rs      # renderer-independent preview
├── src/drop_preview_view.rs # render feature 的 preview Element
├── src/render_plan.rs       # solver、surface、几何、focus metadata
├── src/view.rs              # Panel/Document registry 与 DockView
├── src/registry.rs          # 纯数据 registry
├── src/default_layout.rs    # 默认槽位物化
├── src/persist.rs           # v1 layout codec 与 registry filter
├── src/workspace.rs         # Workspace/Perspective manager
├── src/workspace_persist.rs # workspace codec
├── src/workspace_store.rs   # std 文件事务与回退加载
└── src/autosave.rs          # host-driven debounce/flush
```

features：`default = ["std"]`；`std` 启用 `fs4` 文件存储；`render` 启用 `prism_ui`、组件和 `DockView`。关闭 default features 后，model、validate、reducer、drag、solver、codec、workspace 仍保持 `no_std + alloc`。

公共入口保持稳定：`reduce`、`solve/solve_with`、`encode/decode_layout`、`encode/decode_workspace`、`PanelRegistry`、`WorkspaceManager`、`WorkspaceFileStore`、`WorkspaceAutosave` 和 `DockView`。新增字段通过版本化结构或 builder 扩展，避免破坏已有调用方。

## 15. 测试与验收矩阵

| 层级 | 验收内容 | 当前验证方式 |
|---|---|---|
| Model/validate | ratio、唯一归属、Document、浮窗、maximize、资源上限 | 单元测试与系统化 action 序列 |
| Reducer | split/tabify/move/close/resize/float/dock/autohide/maximize/restore | 纯函数测试、错误不变性测试 |
| Drag/drop | 状态机终态、五向 zone、tab insertion、非法目标 | 无 GPU 单元测试 |
| RenderPlan | 固定点舍入、最小尺寸、hit rect、z 顺序、focus metadata | 确定性几何测试 |
| Element | split、floating、autohide、active factory、missing factory | `render` feature 单元测试 |
| Persistence | round-trip、截断、尾随、UTF-8、未知 ID、maximize restore | layout/workspace codec 测试 |
| File store | lock、last-good、primary/default 回落、并发保存 | `std` 文件存储测试 |
| CPU/GPU | Dock Element draw-list 与真实 wgpu rasterizer 对拍 | `dock_gpu_parity` Metal/wgpu 测试 |
| Accessibility | 语义角色、焦点回收、键盘 action 映射、reduced-motion | host adapter contract tests |
| Regression | 每次 public API、wire version、RenderPlan fixture 变化 | CI snapshot 与版本检查 |

2026 年 10 月 14 日本机验证记录：

- `cargo test -p prism_ui_dock`：94 passed；
- `cargo test -p prism_ui_dock --features render`：102 passed；
- `cargo test -p prism_ui_render_backend --features gpu --test dock_gpu_parity -- --nocapture`：1 passed，真实 GPU adapter 执行通过。

## 16. 版本、迁移与维护

布局 wire version、Workspace wire version、RenderPlan schema version 和 public Rust API version 分开管理。任何不兼容修改必须：

1. 增加版本号；
2. 提供旧版到 canonical model 的显式迁移函数；
3. 增加旧版 fixture、损坏输入 fixture 和迁移后 round-trip fixture；
4. 更新 `docs/version/` 变更记录；
5. 运行默认、render、GPU parity 三组验证；
6. 执行 `git diff --check` 后再提交。

维护原则：一个领域只有一个状态拥有者；布局真相源属于 reducer/model，几何真相源属于 solver，视觉真相源属于 Element/theme，窗口真相源属于 `prism_window`，GPU 资源真相源属于 renderer backend。适配器只能转换，不复制第二份可变状态。

性能回归以同一硬件、同一 viewport、同一 panel registry、同一主题和同一 GPU adapter 比较 P50/P95/P99；只比较平均值不能作为通过依据。任何优化都必须保留确定性输出和错误语义。

## 17. 术语

- **DockNode**：主区或浮窗中的纯数据布局节点。
- **DockLayout**：主树、浮窗、auto-hide 和 maximize 快照的完整状态。
- **TabStack**：普通面板 tab 叶子。
- **Document**：文档 tab 叶子，不能与普通 panel tabify。
- **RenderPlan**：布局到几何、surface、焦点和命中数据的确定性中间表示。
- **Workspace/Perspective**：命名的多套布局快照及其活动选择。
- **DropZone**：拖拽指针对应的中心、边缘或 tab 插入区域。
- **last-good**：最近一次成功验证并提交的 Workspace 快照。
- **Host**：把 dock 纯数据产物接入输入、窗口、renderer、线程和应用生命周期的上层。

## 18. 规范归属

本文与以下文档协同：`prism_editor_framework_design_zh.md` 的编辑器外壳、`prism_ui_component_kit_design_zh.md` 的 DockPanel 外观、`prism_ui_loom_design_zh.md` 的 layout/overlay/store/input/virtual 地基、`prism_window_refactor_zh.md` 的窗口 capability。所有 Prism crate 只借鉴公开架构形态和经典数值，不包含 Unreal Engine、Unity、VS Code 或 JetBrains 的源码及衍生代码。
