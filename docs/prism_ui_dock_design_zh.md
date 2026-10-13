# Prism 停靠系统设计方案（prism_ui_dock）
> v1 / 顶级次世代 AAA 级 — 可停靠 / 可拖拽 / 可持久化 / 可协同的工作台布局内核

> 面向 Prism(Bevy fork)Loom Studio 编辑器(`prism_editor_framework_design_zh.md` §5「编辑器外壳 Shell」)的**停靠宿主(Dock host)**地基。
> `pkg/prism_ui_dock` 已提供纯数据的布局模型、校验、reducer、几何计划、拖拽状态机、Element 装配、host-driven autosave 和持久化边界；
> `prism_ui_component_kit::editor::DockPanel` 仍是**零尺寸纯展示组件**(标题栏 + 内容区 + `DockSide` 修饰类)，作为每块面板的视觉外壳复用。
> `DockView` 当前只实例化每个 TabStack 的 active panel；切换 tab 会按新 active 项调用 factory，inactive panel 不保持挂载，也没有本 crate 生命周期保证。Drop-preview Element 只是 portal 可消费的几何提示，不会自动进入输入 hit-test 路径；上层必须将其作为 pointer-events:none 的视觉 overlay，或在 hit-test 树中显式排除。真实 renderer host、overlay/hit-test 接线与 schema 迁移仍由后续里程碑完成。
>
> **命名定位**:停靠是 **UI 家族的通用能力**(VS Code/Fleet 用它拼整个 IDE,不限游戏编辑器),故 crate 归入 `prism_ui_dock`
> (与 `prism_ui_overlay`/`prism_ui_virtual`/`prism_ui_layout` 同层),编辑器外壳(E 系列)**消费**它。
>
> 借形态不抄码,系统性对标:
> **VS Code**（SerializableGrid 布局树 / 可移动视图容器 / editor group / 布局 JSON 持久化 / 活动栏自动隐藏）、
> **Unreal Slate**（FTabManager / 布局层级 / 主次 Tab / Nomad Tab / 停靠区）、
> **JetBrains Fleet·Rider**（Tool Window / 停靠条 / 工具窗模式）、
> **Unity**（EditorWindow 停靠 / 分隔条 / `.wlt` 保存布局）、
> **Blender**（非重叠平铺 Area/Region / Workspace 切换 / 边角拖拽切分）、
> **Dear ImGui docking**（DockBuilder / 二叉 DockNode / DockSpace / 中心+四边 drop zone）、
> **Qt ADS·wxAUI**（Dock Manager / Perspective 透视图 / 浮动容器）、**Figma·Zed**（面板分组 / Pane Group）。
>
> 本文为**设计规格**。地基层(`prism_ui_layout` 确定性 flexbox 解算、`prism_ui_overlay` portal/浮层 + FocusTrap、
> `prism_ui_store` 响应式状态、`prism_ui_virtual` 虚拟化、`prism_ui_input` hit_test/focus/gesture、
> `prism_ui_component_kit::DockPanel` 面板外观)和 `prism_ui_dock` 的 M0 核心、M1 render plan + Element 装配、M2 纯数据拖拽、M3 core reducer/RenderPlan/DockView Element、M4 Workspace codec/文件存储/host-driven autosave 边界均为**已交付**(SHIPPED)，schema migration 仍 planned; retained draw-list host 已由 `prism_ui_render_backend::ElementRenderer` 提供，窗口/GPU submission、overlay/hit-test 与 schema migration 仍为 PLANNED；其中 retained draw-list host 已交付，但应用级窗口 surface/frame submission 尚未接线。
> 全文严格区分「已实现并通过测试」与「规划中」。
>
> **非目标**:不含 AI/ML/LLM;不做窗口管理器(多显示器/原生窗口由平台/`prism_window` 层承载,本层只给布局模型);
> 不内置协同冲突消解(本层给可序列化布局 + 命令,协同在上层,见 §12)。

- 版本: v1(初始设计)
- 适用: Prism / Loom UI 生态(编辑器外壳 + 任意 Loom 工具应用)
- Crate 名: `prism_ui_dock`
- 关键地基 — 已就绪(SHIPPED): `prism_ui_layout`(确定性 flexbox,`no_std`)、`prism_ui_overlay`(OverlayKind z 序 + FocusTrap,浮动/预览层)、`prism_ui_store`(响应式状态 + select + middleware)、`prism_ui_virtual`(列表虚拟化)、`prism_ui_input`(hit_test z 序/pointer-events、FocusRing、gesture/velocity/swipe)、`prism_ui_scoped`(作用域样式)、`prism_ui_component_kit::editor::DockPanel`(面板外观)、`prism_ui_snapshot`(可逆文本编码,持久化参照)、`prism_ui_anim`/`prism_ui_motion`(拖拽/贴靠动效)
- 剩余宿主能力与跨层依赖: 见 §15 路线图和 §18 现状复核；M0–M4 的纯数据、解算、Element 装配与持久化边界按交付标记区分，未把宿主集成误报为 crate 已完成
- 核心契约:布局是**纯数据树**,一切改动走**确定性 reducer**,持久化输入必须可校验；当前 crate 已提供 active 内容懒构建，虚拟化、增量解算与 schema 迁移属于宿主/后续能力。

---

## 目录
1. 设计哲学与核心契约
2. 顶级产品对标:借形态、取什么、不抄什么
3. 布局模型:DockNode 树(split / tabstack / float / document)
4. 操作与不变量:split / tabify / move / float / resize / maximize / autohide / close
5. 拖拽停靠交互:drop zone / 命中 / 预览 / 状态机
6. 渲染:布局解算 → Element(复用 layout / overlay / DockPanel)
7. 持久化:序列化 / 校验 / 迁移 / Workspace 透视图
8. 面板注册与贡献点(Panel registry)
9. 性能工程:懒构建 / 虚拟化 / 脏子树 / 拖拽解耦
10. 效果(视觉):玻璃 / drop 高亮 / 贴靠动效 / 浮窗 / 最大化过渡
11. 安全:不可信布局输入 / 资源上限 / 注册表白名单
12. 稳定性:不变量 / 确定性 / 优雅收敛 / 撤销 / 多显示器 / 崩溃安全
13. 可达性 / DPI / 主题
14. Crate 结构 / 模块 / feature
15. 路线图(M0–M5)
16. 公共 API 草图
17. 风险与取舍
18. 诚实边界(SHIPPED vs PLANNED)
19. 术语表

---

## 1. 设计哲学与核心契约

- **K1 布局即纯数据**:整个工作台布局是一棵**可序列化、可比较、可快照**的 `DockNode` 树(`no_std + alloc`),与渲染/平台无关,可脱离运行时单元测试。
- **K2 一切改动走确定性 reducer**:split/move/float/resize/close 都是**纯状态转移**;相同操作序列 → 相同布局(可 snapshot 回归)。UI 不直接改树,发**意图(action)**。
- **K3 成本 ∝ 变化量（目标）**:宿主未来可对拖分隔条做局部重算；当前 `render_plan::solve` 与 reducer 是整棵布局的确定性处理。切 tab 的 Element 装配只调用 active factory，隐藏 tab 内容不构建。
- **K4 面板懒构建**:render feature 下只有当前 active tab/autohide reveal 内容进入 `Element` 树；后台 tab/折叠面板只保留标题。tab 虚拟化、selector 与持久结构共享仍是宿主/后续优化，不由本 crate 提供。
- **K5 不变量始终成立**:有效布局满足比例为正且归一、面板实例唯一归属、浮窗各自成合法树；`Empty` 是允许的占位根，reducer/validator 负责校验与折叠可收敛空节点。
- **K6 持久化输入不可信**:从磁盘/云同步读回的布局当作**不可信数据**——未知 ID 可按 registry 过滤、深度/数量/字节上限、不 panic、非法比例/模型报结构化错误。当前 version=1 不提供 schema 迁移；`WorkspaceFileStore` 才提供 primary → last-good → caller default 的宿主回落链。
- **K7 贡献点而非硬编码**:core registry 只保存稳定 ID、标题、默认槽位、实例策略等纯数据；render feature 的 `PanelViewRegistry`/`DocumentViewRegistry` 在宿主边界提供 factory。
- **K8 可测优先**:布局可 snapshot；拖拽状态机可脱离 GPU 测试；`render_plan` 提供确定性 focus metadata，但键盘导航和完整 a11y 接线仍属宿主能力。

---

## 2. 顶级产品对标:借形态、取什么、不抄什么

| 来源 | 借鉴形态 | 取什么到 `prism_ui_dock` | 不抄什么 |
|---|---|---|---|
| **VS Code** | SerializableGrid 布局树、可移动视图容器、活动栏自动隐藏、布局 JSON | §3 节点树;§7 JSON 持久化 + 校验;§4 autohide | 其 Electron/DOM 实现 |
| **Unreal Slate** | FTabManager、布局层级、主次 Tab、Nomad Tab、停靠区 | §3 split/tabstack 分层;§8 面板注册 | Slate 源码 |
| **Dear ImGui docking** | 二叉 DockNode、中心+四边 drop zone、DockBuilder | §3 二叉 split 树;§5 中心+四边 drop zone | 其即时模式模型 |
| **JetBrains Fleet/Rider** | Tool Window 模式、停靠条、固定/浮动/窗口化 | §4 autohide/float 态;§8 工具窗语义 | 商业实现 |
| **Blender** | 非重叠平铺、Workspace、边角切分 | §3 平铺不重叠约束;§7 Workspace 透视图 | 其 Area/Region 具体 |
| **Qt ADS / wxAUI** | Dock Manager、Perspective、浮动容器 | §7 Perspective(命名布局);§3 浮窗子树 | 其 C++ 栈 |
| **Zed / Figma** | Pane Group、低延迟拖拽 | §5 低延迟拖拽解耦;§9 性能 | 其内部实现 |

> **版权红线**:所有 Prism crate 不含任何 Unreal/Unity/VS Code/JetBrains 源码或衍生代码;仅借鉴**公开架构形态**。`prism_ui_dock` 为自研,`DockPanel` 外观复用自已交付的 `prism_ui_component_kit`(本仓库)。

---

## 3. 布局模型:DockNode 树

一棵 `DockNode` 树描述整个工作台;浮窗各自是独立子树;文档区为特殊 leaf。

```rust
pub enum DockNode {
    /// 内部节点:沿一个轴把空间切成 N 份,各带归一比例。
    Split { axis: Axis, children: Vec<(Ratio, DockNode)> },
    /// 叶子:一个 tab 栈,多个面板叠为标签页,记录激活项。
    TabStack { tabs: Vec<PanelId>, active: usize },
    /// 文档区:编辑器主视口/多文档的特殊 tab 栈(可被环绕但不可被吞并)。
    Document { tabs: Vec<DocumentId>, active: usize },
    /// 占位空根(无任何面板时)。
    Empty,
}

pub struct DockLayout {
    pub root: DockNode,                 // 主停靠区(含文档区)
    pub floating: Vec<FloatingWindow>,  // 浮动窗口,各自一棵 DockNode
    pub autohide: AutoHideBars,         // 四边自动隐藏栏(折叠的工具窗)
}

pub struct FloatingWindow { pub rect: Rect, pub root: DockNode, pub z: u32 }
```

**设计选择**:
- **二叉优先、N 叉可退化**:多数操作产出二叉 split(便于拖拽/合并),但同轴相邻 split 可合并为 N 叉以压平深度(减少嵌套、利 §9 脏子树)。
- **TabStack 为唯一叶子载体**:单面板也是只含一个 tab 的栈,统一模型,避免「单面板 vs 多面板」分叉。
- **文档区特殊化**:对齐 VS Code editor group / Unreal 主 Tab —— 文档区可被工具面板**环绕**,但工具面板不能把文档区吞成自己的 tab(保持中心稳定)。
- **浮窗独立成树**:浮窗内部仍可 split/tab;对齐 Qt ADS 浮动容器。多显示器定位交平台层(§12.5)。

---

## 4. 操作与不变量

所有操作是 reducer:`fn reduce(layout, action) -> layout`,纯函数、确定性。

| 操作 | 语义 | 不变量维护 |
|---|---|---|
| `Split(target, axis, side)` | 在目标 leaf 旁切出新空间 | 新建二叉 split,比例默认 0.5 |
| `Tabify(panel, target)` | 把面板并入目标 tab 栈 | 源节点若空则折叠 |
| `Move(panel, dest, pos)` | 跨节点移动(拖拽落点) | 源折叠 + 目标插入,树重归一 |
| `Float(panel, rect)` | 面板脱离成浮窗 | 源折叠,浮窗入 `floating` |
| `Dock(floating, target)` | 浮窗停靠回主区 | 浮窗出列,内容并入 |
| `Resize(split, ratios)` | 拖分隔条调比例 | 正值校验并按固定点总量归一；最小尺寸由 render plan 报错 |
| `Maximize/Restore(leaf)` | 全屏某 leaf,保存前态 | 可逆,存 restore 快照 |
| `AutoHide(panel, side)` | 折叠到边栏(悬停弹出) | 对齐 VS Code 面板/Fleet 工具窗 |
| `Close(panel)` | 关面板 | 空 leaf 折叠,空间回邻居 |

**折叠(collapse)规则**:关掉 tab 栈最后一个 tab → leaf 删除,其空间按比例并回兄弟;split 只剩一个子 → 用该子替换 split(压平)。保证**无悬挂空节点**。

---

## 5. 拖拽停靠交互

拖拽是独立**状态机**,与模型解耦,可无 GPU 单测:

```
Idle → PickUp(panel) → Dragging{ghost, hover} → Drop(action) | Cancel
```

- **抓取（宿主规划）**:tab/标题栏按下进入拖拽；当前 crate 的状态机只接收 `PickUp/Move/Drop/Cancel/Reset` 与逻辑指针位置，不实现 gesture velocity/swipe 阈值或惯性。
- **命中（宿主规划）**:`prism_ui_input::hit_test`(z 序 + pointer-events)接线由上层 host 完成；当前 crate 只提供确定性几何与 drop zone 计算。
- **drop zone（当前已实现）**:悬停面板目标显示中心(tabify)、上/下/左/右(split)以及面板 tab 栏内插入位(tab 间重排)；Document 目标不支持 center/tab-insert，只允许 split。整区外缘停靠仍是规划能力。
- **预览（适配器已实现，接线规划）**:`drop_preview_element` 生成 absolute Element；portal、pointer-events/hit-test 排除和实时事件由上层 host 负责。
- **落下**:映射为 §4 的 reducer action；当前 `map_drop_action` 覆盖 center/split/tab-insert，Float/Dock 由宿主根据交互上下文编排。
- **取消**:Esc / 落到非法区 → 回原位,无副作用(呼应确定性)。

分隔条拖拽的 reducer action 只提交新比例；`prism_ui_store` 接线与局部重算是宿主优化目标，当前 crate 不提供增量缓存。

---

## 6. 渲染:布局解算 → Element

- **几何**:`render_plan::solve` 将 `DockNode` 树映射为 crate 自有的 renderer-independent `RenderPlan`（面板、文档、tab、splitter 与 surface 几何）；`DockView` 独立装配 Element，当前并未直接构造或消费 `prism_ui_layout::LayoutTree`。实际 layout host 接线由上层负责。
- **面板外观**:每个可见 tab 内容包进 `prism_ui_component_kit::editor::DockPanel`(复用其玻璃面 + 标题栏 + `DockSide`),本层只负责**摆放与装配**。
- **浮窗 / 拖拽预览**:当前 `DockView` 生成浮窗 Element，`drop_preview_element` 生成视觉提示；portal 层、z 序、FocusTrap、pointer dispatch 和窗口 surface 由上层 host 接线。
- **分隔条**:可拖拽把手元素,带命中区 padding(易抓)。
- **增量（宿主规划）**:Element 装配当前每次生成完整树；宿主未来可对 tab/几何变化做局部 patch。

---

## 7. 持久化:序列化 / 校验 / 迁移 / Workspace

- **稳定二进制编码（已交付）**:`encode_layout`/`decode_layout` 及 workspace codec 使用 magic/version、checked 长度、资源上限和模型校验；当前不是 JSON codec。
- **schema 版本化（规划）**:当前 version=1 对未知版本返回错误，尚无旧版迁移链。
- **校验(§11)**:加载时执行 registry 过滤、比例/深度/数量/字节上限和不变量校验；非法输入返回结构化错误，默认回落由文件存储或宿主编排。
- **Workspace / Perspective(借 Qt ADS / Blender)**:命名多套布局(「编码」「调试」「美术」),一键切换；Workspace/Perspective 为内存模型，可通过 codec/store 持久化整个 manager 快照，自动保存由 host 编排。浮窗层叠在 Element 装配时按 `z` 升序生成（低层先绘制、高层后绘制）。
- **自动保存**:`WorkspaceAutosave` 是 `std` 下由宿主 poll/flush 的 debounce 控制器，不是 store middleware 或后台线程。
- **重置**:始终可一键回**默认布局**(由面板注册表的默认位置生成)。

---

## 8. 面板注册与贡献点

```rust
pub struct PanelDescriptor {
    pub id: PanelId,
    pub title: String,
    pub icon: Option<IconId>,
    pub instancing: Instancing,     // Singleton | Multi
    pub default_slot: DefaultSlot,  // LeftBar | RightBar | Bottom | Document | Floating
    // core registry 不持有 Element factory；factory 由 render host registry 提供
}
pub struct PanelRegistry {
    // 真实实现保持注册顺序；注册时拒绝空白 ID 与重复 ID。
}
impl PanelRegistry {
    pub fn register(&mut self, desc: PanelDescriptor) -> Result<(), PanelRegistryError>;
    pub fn get(&self, id: &PanelId) -> Option<&PanelDescriptor>; // 持久化白名单
    pub fn iter(&self) -> core::slice::Iter<'_, PanelDescriptor>;
}
```

- **贡献点**:插件/域编辑器注册面板,不改停靠内核。对齐编辑器 D3「贡献点而非硬编码」。
- **单例 vs 多例**:Inspector 单例;文档/图表可多例。
- **纯数据边界**:当前 core registry 只保存稳定 ID、标题、实例策略、默认槽位和初始可见性；不伪造 `Element`、GPU 句柄或 factory。懒构建由 `PanelViewRegistry`/`DocumentViewRegistry` 提供。
- **白名单**:持久化只认注册过的 id,反序列化未知 id 优雅丢弃(§11)。

---

## 9. 性能工程:成本 ∝ 变化量

- **懒构建**:仅可见 tab 的 `factory()` 执行;后台 tab/autohide 折叠面板内容不进树,激活时才建、隐藏可回收。render feature 下 active autohide 会生成独立 reveal sibling 与 `DockPanel` 外壳；仅 active 项要求 panel factory，折叠 tab 仍可无 factory 展示标题。
- **虚拟化（规划）**:tab 很多时可由宿主用 `prism_ui_virtual` 窗口化；当前 crate 不提供 tab 虚拟化。
- **脏子树重算（规划）**:宿主可对拖分隔条做局部几何重算；当前 `render_plan::solve` 每次处理完整布局。
- **拖拽解耦**:纯数据状态机与 `drop_preview_element` 不修改布局；overlay、hit-test 排除、ghost 合成和 reducer commit 由宿主接线。
- **结构共享（规划）**:当前 reducer 使用 owned clone，不提供持久化结构共享或 selector。
- **可测性能（规划）**:宿主未来可暴露可见面板数、每帧重算节点数和拖拽帧延迟；当前 crate 提供确定性求解与测试，不提供运行时指标。

---

## 10. 效果(视觉)

- **玻璃表面**:复用 `DockPanel` 的 frosted glass + 主题 token(`color.separator`/`glass.tint`),停靠区整体材质统一。
- **drop 高亮**:落区半透明填充 + 边框脉冲,`prism_ui_anim`/`prism_ui_motion` 做淡入淡出与方向提示。
- **贴靠/切分动效（规划）**:面板落位、split 产生/折叠的 spring 过渡由宿主动画层实现。
- **浮窗/最大化过渡（规划）**:投影、ghost、几何插值不属于当前 crate 的渲染或窗口边界。
- **分隔条（规划）**:悬停样式、光标和性能档开关由 host 接线；当前 crate 只输出确定性几何。
- 动效必须支持 CI 确定性模式；本 crate 当前不提供动画实现。

---

## 11. 安全

停靠内核的攻击面主要是**持久化布局(来自磁盘/云同步,可能损坏或被构造)**:

- **不可信反序列化**:加载布局时——
  - **未知面板 id**:注册表白名单外的一律丢弃,不实例化任何未注册内容(无「从布局执行任意代码」路径)。
  - **数值校验**:比例要求正值并按固定点总量归一；最小尺寸由 `render_plan` 求解时结构化报错。浮窗 rect、z 序和模型不变量非法时拒绝，不静默钳制或去重。
  - **资源上限**:最大面板数、最大嵌套深度、最大浮窗数、最大 tab 数——防构造深树/海量节点导致的栈溢出/OOM(DoS)。
  - **不 panic**:畸形结构返回 `LayoutError`/`PersistError`；回落默认布局由 `WorkspaceFileStore` 或宿主编排，不由 codec 隐式完成。
  - **schema 迁移（规划）**:当前 version=1 对未知版本返回 `UnsupportedVersion`；迁移链尚未交付。
- **无代码注入**:布局只含 id/数值/结构,不含可执行内容;面板行为完全由注册表(进程内可信代码)决定。
- **云同步隔离**:多设备同步的布局同样过上述校验,一台的损坏布局不污染内核。

---

## 12. 稳定性

### 12.1 不变量(invariants)
reducer/validator 每次边界强制:无悬挂空节点、比例归一、每面板实例唯一归属、浮窗各自合法树；`Document` 节点是可选的，若存在则不可与工具 tabify。可用 `debug_assert` + 属性测试(proptest 式随机操作序列后校验不变量)兜底。

### 12.2 确定性
相同 action 序列 → 相同 `DockLayout`;可 snapshot 回归(复用 `prism_ui_snapshot`)。拖拽状态机脱离 GPU 可测。

### 12.3 优雅收敛
关最后一个 tab、折叠空 split、浮窗清空自动关闭、拖拽取消回原位——所有边界都有确定收敛行为,不留中间态。

### 12.4 撤销 / 重做(可选)
布局操作可记入命令栈(对齐编辑器命令内核),结构共享让历史廉价;「误拖散布局」可一键撤销。

### 12.5 多显示器 / DPI
浮窗跨屏定位、DPI 缩放交平台/`prism_window` 层;本层只存逻辑 rect + 显示器 id,恢复时由平台层校正到当前显示器拓扑(屏幕消失则拉回主屏)。

### 12.6 崩溃安全持久化
原子写(临时文件 + rename)+ 保留 last-good（当前 Unix 目标替换语义完整；Windows 在目标已存在时显式返回 `AlreadyExists`，待 `ReplaceFileW` 适配后再宣称重复保存支持）；`WorkspaceFileStore` 加载失败回落 last-good → caller default。稳定 sibling lock 使用 OS advisory lock（Unix/Windows 由 `fs4` 适配），锁文件本身不删除；进程崩溃时由 OS 释放持有的锁，避免 mtime/PID stale reclaim 竞态。

---

## 13. 可达性 / DPI / 主题

- **键盘导航（规划）**:`prism_ui_input::FocusRing` 可由宿主接线到面板/tab；快捷键移动/切分面板、切 tab、聚焦相邻面板尚未由本 crate 接线。
- **角色/语义（部分底座，宿主接线规划）**:`DockPanel::role() == Role::Group` 可复用；tablist/tab 语义和辅助技术事件尚未由 `DockView` 暴露。
- **FocusTrap（规划）**:模态浮窗可由宿主用 `prism_ui_overlay::FocusTrap` 困住焦点；当前 crate 不管理焦点生命周期。
- **DPI/HDR/主题**:几何用逻辑像素由布局解算,渲染后端做 DPI 缩放;颜色走主题 token,支持明暗/HDR。

---

## 14. Crate 结构 / 模块 / feature

```
prism_ui_dock/
├── src/model.rs / validate.rs / reducer.rs  # 纯数据、校验、确定性 reducer
├── src/drag.rs / drop_preview.rs            # 拖拽状态机、drop zone、预览几何
├── src/render_plan.rs                       # renderer-independent 几何计划
├── src/view.rs                              # `render` feature 的 Element 装配
├── src/persist.rs / workspace.rs            # codec 与内存 Workspace/Perspective
├── src/workspace_store.rs / autosave.rs     # `std` 文件边界与 host-driven autosave
└── src/registry.rs                          # 纯数据 PanelDescriptor/PanelRegistry
```
features: `std`(文件存储)/ `render`(Element 装配)/ `snapshot_tests`。当前无独立 `store/`、`a11y/` 模块；核心 model/reducer/persist 为 `no_std + alloc`。

---

## 15. 路线图(M0–M5)

| 里程碑 | 内容 | 依赖 |
|---|---|---|
| **M0** | model + reduce(split/tabify/close/resize)+ 不变量 + 属性测试 | — |
| **M1** | renderer-independent 几何解算 + render plan + tab/header/splitter 元数据 | Dock model ✅ / DockPanel ✅ / `render` feature Element 装配 ✅ / retained draw-list host ✅；窗口/GPU submission 仍待完成 |
| **M2** | 拖拽状态机 + 中心/四边 drop zone + tab 插入 + overlay 预览 | 状态机/drop-zone/reducer action 已交付；renderer overlay 预览仍待集成 |
| **M3** | 浮窗 + autohide 边栏 + maximize | core reducer、RenderPlan 几何与 DockView Element 已交付；原生窗口、portal overlay、pointer/focus 接线仍待宿主完成 |
| **M4** | 持久化(序列化/校验/迁移)+ Workspace + 自动保存 | persistence core ✅ / Workspace 纯数据与 codec ✅ / host-driven autosave ✅；schema migration 仍待完成 |
| **M5** | 动效 + 可达性 + 撤销集成 + 多显示器校正 | 平台层 / 命令内核 |

---

## 16. 公共 API 草图（历史示意，非可编译 API）

```rust
// 示意：实际 API 使用 PanelRegistry/PanelDescriptor，factory 与 store 由 host 提供。
let mut registry = PanelRegistry::new();
registry.register(PanelDescriptor::new("inspector", "Inspector"))?;

// 实际 reducer 直接接收 DockLayout 与 Action；不存在 DockStore/DockRegistry builder。
let next = reduce(&layout, Action::Resize { split: split_id, ratios })?;

// render feature：实际 DockView::new 接收 layout、PanelRegistry、PanelViewRegistry、DocumentViewRegistry。
let view = DockView::new(&layout, &registry, &panel_views, &document_views);
let element = view.render()?; // → Result<Element, DockViewError>

// 加载(不可信输入,校验/过滤；迁移与回落由 host/store 编排)
let layout = decode_layout(&bytes, DecodeOptions {
    known_panels: Some(registry.ids()),
    ..DecodeOptions::default()
})?;
```

---

## 17. 风险与取舍

1. **二叉 vs N 叉**:二叉易拖拽/合并但易生深树;靠「同轴合并压平 + 深度上限」平衡,代价是 reducer 复杂度。
2. **文档区特殊化**:避免中心被吞提升可用性,但引入「普通 leaf vs 文档 leaf」双语义,需测试覆盖边界。
3. **拖拽解耦的实现成本**:ghost 在 overlay 独立合成、落下才 reduce,是流畅度关键,也是最易出 bug 处(命中/预览/落点三者一致性)。
4. **持久化健壮性**:布局是用户最易感知丢失的状态;必须 last-good + 原子写 + 回落默认三重兜底。
5. **多显示器**:逻辑 rect 跨屏恢复依赖平台层拓扑;屏幕热插拔的拉回策略需明确,否则浮窗「消失」。
6. **动效 vs 确定性**:动效提升体验但引入时间态;CI/确定性模式必须可关,snapshot 测稳态而非动画中帧。

---

## 18. 诚实边界(SHIPPED vs PLANNED)

- ✅ **已存在底座(非本 crate)**:`prism_ui_layout`(确定性 flexbox,`no_std`)、`prism_ui_overlay`(OverlayKind z 序 + FocusTrap + portal 渲染)、`prism_ui_store`(Store/select/middleware)、`prism_ui_virtual`(视口窗口化 + overscan + spacer)、`prism_ui_input`(hit_test z 序/pointer-events、FocusRing tabindex、gesture/velocity/swipe/multitap)、`prism_ui_scoped`、`prism_ui_snapshot`(serialize_tree/parse_tree 可逆编码)、`prism_ui_anim`/`prism_ui_motion`、`prism_ui_component_kit::editor::DockPanel`(**仅视觉外壳**:标题栏 + 内容区 + DockSide 修饰类 + role=Group,无任何布局逻辑)。
- ✅ **M0 已交付**:`pkg/prism_ui_dock` 的 `model` / `validate` / `reduce`：纯数据 `DockNode`/`DockLayout`、split/tab/document/floating/autohide/maximize 模型、资源上限与唯一归属校验，以及确定性 split/tabify/move/close/resize/float/dock/maximize/restore/autohide reducer。
- ✅ **M1 几何/计划垂直切片已交付**:`render_plan` 暴露确定性 `solve`/`solve_with`，输入经校验的模型、逻辑 viewport 和显式 metrics，输出主区/浮窗/auto-hide/maximize 的面板与文档矩形、tab headers、splitter/hit rect、active/focus 元数据。比例使用固定点整数分配与最大余数稳定舍入；最小尺寸不足、无效模型、bar/reveal 溢出均结构化报错，不静默钳制比例或几何。
- ✅ **M1 Element 装配已交付（`render` feature）**:`DockView` 通过独立的 `PanelViewRegistry`/`DocumentViewRegistry` 调用懒 factory，把已校验的 `DockLayout` 递归装配为 `prism_ui::Element`，覆盖 split/tab/document/floating/autohide，并复用 `DockPanel` 外壳；TabStack 只装配 active tab，inactive tab 只生成标题，不保持内容 view 生命周期；未知 panel、缺失 factory 和非法布局均返回结构化错误。核心默认构建仍保持 no_std + alloc 与 renderer-independent。
- ✅ **Retained draw-list host 已交付**:`prism_ui_render_backend::ElementRenderer` 复用 Loom retained tree，提供 Element update、布局解算和 `DrawList` 输出；Dock GPU parity 已改为走该生产 host API。
- ⬜ **窗口/GPU submission host 仍未交付**:GPU rasterizer 与真实 parity 已验证，但窗口 surface、overlay portal、命中测试接线、focus/a11y 和应用级 frame submission 仍由上层 UI host 完成。
- ✅ **Dock Element 的真实 GPU parity fixture 已交付（范围限定）**:`pkg/prism_ui_render_backend/tests/dock_gpu_parity.rs` 将一个 320×220 的简单水平 split `DockView` Element（scene/inspector，玻璃主题与标题 glyph）经过 Loom layout 和 draw list，比较 CPU golden 与真实 wgpu rasterizer；本次本机 GPU feature 测试通过，最大通道差阈值为 `2e-3`。它只证明该 Element→retained draw-list→rasterizer fixture，不覆盖 `render_plan::solve`、浮窗、auto-hide、maximize、document 或 drag/drop preview，也不等同窗口 surface/frame submission parity。无 GPU adapter 时当前测试会 return，因此 CI 通过本身不证明实际执行了 GPU 对拍。GPU feature 下 CPU golden 自动使用与 shader 相同的 SDF 文本 tier；默认 CPU-only 构建仍保留 vector tier。
- ✅ **M2 纯数据拖拽内核已交付**:`drag` 提供确定性拖拽状态机、中心/四边 drop zone、tab 插入边界和 reducer action 映射；当前测试覆盖状态机与边界行为。overlay 预览和真实 UI hit-test/render 集成尚未交付。
- ✅ **M2 drop-preview 几何接缝已交付**:`resolve_drop_preview` 将 `DropZone` 与已求解的 `RenderPlan` panel/document/tab rect 连接为 deterministic `DropPreview`，覆盖 center/split/tab-insert、目标缺失和索引越界错误；portal Element、pointer dispatch 与实际 reducer commit 仍由上层 host 接线。
- ✅ **M2 drop-preview Element 几何适配器已交付（`render` feature）**:`drop_preview_element` 将 `DropPreview` 转换为 absolute Element，供 portal host 作为视觉提示消费；Element 本身未绑定输入层 pointer-events/hit-test-through 语义，不自行注册事件或修改 `DockLayout`。上层必须显式从 hit-test 树排除此 overlay，避免它遮挡真实 drop target。TabInsert 预览使用 panel/document 的 header `tab_rect`，center/split 使用 leaf rect。
- ✅ **M4 persistence core 已交付**:`pkg/prism_ui_dock::encode_layout`/`decode_layout` 提供带 magic/version 的确定性二进制布局编码、checked 长度/深度/节点预算、UTF-8 与枚举校验、解码后模型不变量校验，以及带已知 panel/document 注册表的未知 ID 过滤和安全分支折叠；主树、浮窗、autohide、maximize restore 均可 round-trip。最大化状态要求当前可见面严格收敛为快照中的唯一面板，并由 model/persistence validation 在进入 renderer 前拒绝非法恢复态。该切片不提供文件 IO、schema 迁移链或自动保存。
- ✅ **Workspace / Perspective 纯数据管理已交付**:`WorkspaceManager`、`Workspace` 与 `Perspective` 按确定性插入顺序管理命名布局；拒绝空白/重复名称，并以 `DockLayout` 深拷贝隔离保存与切换边界。该切片只负责内存中的选择与快照，不提供文件 IO、自动保存、factory 或 renderer。
- ✅ **Workspace 持久化 codec 已交付**:`encode_workspace`/`decode_workspace` 提供带 magic/version 的稳定 Workspace/Perspective 快照，逐个复用布局 codec，执行 active 引用、资源上限、UTF-8、枚举、截断和尾随字节校验；encoder 与默认 decoder 的 workspace/perspective/name/layout/总字节预算对称，Writer 在每次 append 前拒绝超出总预算；`decode_workspace_with_registry` 将 `PanelRegistry` 作为嵌套布局的 panel allow-list。该切片只处理内存字节。
- ✅ **Workspace 文件存储边界已交付**（`std`）:`WorkspaceFileStore` 通过 sibling lock 串行化保存、唯一临时文件、`sync_all` 与 rename 提交（当前 Unix 目标替换完成；Windows 已存在目标会返回 `AlreadyExists`，尚未接入 `ReplaceFileW`）；并在 Unix 上同步父目录；覆盖 primary 前先验证旧快照，再以独立临时文件更新 `last-good`，失败路径清理临时文件。加载按 primary → last-good → caller default 回退并保留诊断，并提供 Registry-aware load 防止未知 panel 进入上层。稳定 sibling lock 使用 `fs4` 提供的跨平台 OS advisory lock；锁文件本身不删除，进程崩溃时由 OS 释放持有的锁，避免 mtime/PID stale reclaim 竞态。该锁只协调同样遵守该协议的 writer，不能约束绕过锁的外部写入。它不是 autosave middleware，不提供 schema 迁移或 renderer 装配；断电语义仍取决于底层文件系统。
- ✅ **Host-driven autosave 调度已交付**:`WorkspaceAutosave` 维护 dirty/debounce deadline，宿主通过单调 `Instant` 调用 `poll` 或显式 `flush`；写入失败保留 dirty 状态以便重试。它不创建后台线程、不在 reducer 内隐式阻塞、不伪造 schema migration；宿主负责把调用放入 worker/executor 边界。
- ✅ **Registry 基础契约已交付**:`PanelDescriptor`/`PanelRegistry` 提供 no_std + alloc 的纯数据贡献点、确定性注册顺序、空白/重复 ID 拒绝、查询和迭代；不包含 Element/factory/renderer。
- ✅ **Registry → 默认布局物化已交付**:`PanelRegistry::default_layout(DefaultLayoutPolicy)` 按注册顺序把可见面板映射到 Left/Right/Bottom/Document/Floating 槽位；同槽位稳定合并为 tab stack，空槽位不产生非法空节点，浮窗 z 序确定，并在返回前执行完整布局校验。`Instancing::Multi` 在此只生成一个默认实例，不伪造多实例语义。
- ✅ **Workspace 默认布局重置接线已交付**:`WorkspaceManager::reset_active_perspective_to_default` 与 `reset_perspective_to_default` 在提交前完整物化并校验 Registry 默认布局，成功后以深拷贝替换目标 perspective；策略错误、无活动 perspective 或未知名称均返回结构化错误且不留下半成品状态。
- ⬜ **M3–M5 其余能力仍 PLANNED**:`a11y` 以及动画/多显示器/撤销集成；绕过锁的外部写入不受保护；`store` 文件存储边界与 host-driven autosave 已交付，但不等同于 a11y、动画、多显示器、撤销等能力。完整窗口/portal UI 装配和 renderer frame submission 仍属于后续工作，不因 M1 计划数据或 persistence core 已存在而视为完成。当前已修复 layout encoder 与默认 decoder 的字符串/集合/深度资源限制不对称，并拒绝未知 inactive panel 与不一致 maximize snapshot 归属。
- ⬜ **依赖未落地项**:撤销集成需编辑器命令内核;多显示器校正需平台/`prism_window` 层;动效档位依赖性能分级。
- 现状复核(本次已核验):`pkg/prism_ui_dock` 已提供 M0 宿主核心、M2 GPU-independent drag/drop 内核与 drop-preview 几何接缝、M1 renderer-independent render plan、`render` feature 下的 Element 装配、`ElementRenderer` retained draw-list host、Dock Element 的真实 GPU parity、M4 persistence core、Workspace/Perspective 纯数据管理、codec、`std` 文件存储边界、host-driven autosave、Registry 基础契约及 Registry → 默认布局物化；布局校验会在 renderer 消费前拒绝非法 maximize 恢复态。窗口/GPU submission host、portal overlay/pointer dispatch、schema 迁移和上层 focus/a11y 接线仍未交付。

---

## 19. 术语表

- **DockNode / DockLayout**:描述工作台的纯数据布局树与其根 + 浮窗 + 自动隐藏栏的聚合。
- **TabStack**:叶子节点,多个面板叠为标签页,记录激活项;单面板也是单 tab 栈。
- **drop zone**:拖拽悬停时的落区(中心 tabify / 四边 split / tab 间重排；外缘整区停靠为规划能力)。
- **reducer**:把 action 作用到布局的纯函数,维护所有不变量,确定性可测。
- **Workspace / Perspective**:命名的多套布局,一键切换(借 Qt ADS/Blender)。
- **autohide 边栏**:折叠到四边、悬停弹出的工具窗模式(借 VS Code/Fleet)。
- **last-good**:上一份成功加载的布局,崩溃/损坏时的恢复兜底。

---

> 本文为 `prism_ui_dock` 的 v1 设计规格。实现须遵循 `prism_bevy_refactor_plan_zh.md` 的「pkg 优先、`prism_*` 唯一真相源」方向,与 `prism_editor_framework_design_zh.md`(§5 外壳)、`prism_ui_component_kit_design_zh.md`(DockPanel 外观)、`prism_ui_loom_design_zh.md`(layout/overlay/store/input/virtual 地基)协同演进。所有 Prism crate 不含任何 Unreal Engine / Unity / VS Code / JetBrains 源码或衍生代码;仅借鉴公开架构形态与经典数值。
