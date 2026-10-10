# Prism 停靠系统设计方案（prism_ui_dock）
> v1 / 顶级次世代 AAA 级 — 可停靠 / 可拖拽 / 可持久化 / 可协同的工作台布局内核

> 面向 Prism(Bevy fork)Loom Studio 编辑器(`prism_editor_framework_design_zh.md` §5「编辑器外壳 Shell」)的**停靠宿主(Dock host)**地基。
> 当前 `pkg` 仅有 `prism_ui_component_kit::editor::DockPanel` —— 一个**零尺寸纯展示组件**(标题栏 + 内容区 + `DockSide` 修饰类),
> **没有任何布局管理**:无 split/tab/拖拽重排/浮动窗口/布局持久化。本 crate 补齐这整套「宿主」逻辑,
> `DockPanel` 则作为「每块面板的外观」被复用。
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
> `prism_ui_component_kit::DockPanel` 面板外观)为**已交付**(SHIPPED);**`prism_ui_dock` 本体为全新规划项(PLANNED)**。
> 全文严格区分「已实现并通过测试」与「规划中」。
>
> **非目标**:不含 AI/ML/LLM;不做窗口管理器(多显示器/原生窗口由平台/`prism_window` 层承载,本层只给布局模型);
> 不内置协同冲突消解(本层给可序列化布局 + 命令,协同在上层,见 §12)。

- 版本: v1(初始设计)
- 适用: Prism / Loom UI 生态(编辑器外壳 + 任意 Loom 工具应用)
- Crate 名: `prism_ui_dock`
- 关键地基 — 已就绪(SHIPPED): `prism_ui_layout`(确定性 flexbox,`no_std`)、`prism_ui_overlay`(OverlayKind z 序 + FocusTrap,浮动/预览层)、`prism_ui_store`(响应式状态 + select + middleware)、`prism_ui_virtual`(列表虚拟化)、`prism_ui_input`(hit_test z 序/pointer-events、FocusRing、gesture/velocity/swipe)、`prism_ui_scoped`(作用域样式)、`prism_ui_component_kit::editor::DockPanel`(面板外观)、`prism_ui_snapshot`(可逆文本编码,持久化参照)、`prism_ui_anim`/`prism_ui_motion`(拖拽/贴靠动效)
- 前置待建(本 crate 落地): 见 §15 路线图 M0–M5
- 核心契约: 继承 Loom「**成本 ∝ 变化量**」;布局是**纯数据树**,一切改动走**确定性 reducer**,一切持久化**可校验可迁移**,一切可见面板**懒构建 + 虚拟化**。

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
- **K3 成本 ∝ 变化量**:拖分隔条只重算受影响子树;切 tab 只重建该 leaf;隐藏 tab 内容**不构建**。继承 Loom 契约。
- **K4 面板懒构建 + 虚拟化**:只有**当前可见**的 tab 内容进入 Element 树;后台 tab/折叠面板的内容延迟实例化。十数十面板不等于十数十次每帧构建。
- **K5 不变量始终成立**:树永远合法——无空 leaf(占位根除外)、比例归一、每个面板实例恰在一个节点、浮窗各自成树。reducer 每次操作后**重归一 + 折叠空节点**。
- **K6 持久化输入不可信**:从磁盘/云同步读回的布局当作**不可信数据**——未知面板 id 优雅丢弃、比例钳制、深度上限、不 panic、schema 迁移、损坏则回落默认布局。
- **K7 贡献点而非硬编码**:面板经注册表贡献(id ↔ 工厂 + 默认位置 + 图标 + 单例/多例),对齐编辑器 D3。
- **K8 可达 + 可测**:键盘可在面板间导航/移动面板;布局可 snapshot;拖拽状态机可脱离 GPU 测试。

---

## 2. 顶级产品对标:借形态、取什么、不抄什么

| 来源 | 借鉴形态 | 取什么到 `prism_ui_dock` | 不抄什么 |
|---|---|---|---|
| **VS Code** | SerializableGrid 布局树、可移动视图容器、活动栏自动隐藏、布局 JSON | §3 节点树;§7 JSON 持久化 + 校验;§4 autohide | 其 Electron/DOM 实现 |
| **Unreal Slate** | FTabManager、布局层级、主次 Tab、Nomad Tab、停靠区 | §3 split/tabstack 分层;§8 面板注册 | Slate 源码 |
| **Dear ImGui docking** | 二叉 DockNode、中心+四边 drop zone、DockBuilder | §3 二叉 split 树;§5 五向 drop | 其即时模式模型 |
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
| `Resize(split, ratios)` | 拖分隔条调比例 | 比例钳制 [min,1-min] 并归一 |
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

- **抓取**:tab/标题栏按下进入拖拽;`prism_ui_input::gesture` 的 velocity/swipe 提供惯性与阈值(避免误触)。
- **命中**:`prism_ui_input::hit_test`(z 序 + pointer-events)找悬停的 leaf/split/边缘。
- **drop zone(五向 + 外缘,借 ImGui)**:悬停目标显示——中心(tabify)、上/下/左/右(在目标旁 split)、以及**整区外缘**(停靠到整个停靠区的边)。tab 栏内悬停显示**插入位**(tab 间重排)。
- **预览**:`prism_ui_overlay` 画半透明 ghost + 高亮落区;落点合法性实时反馈(非法落区灰显)。
- **落下**:映射为 §4 的 `Move/Float/Dock/Tabify` action,过 reducer。
- **取消**:Esc / 落到非法区 → 回原位,无副作用(呼应确定性)。

分隔条拖拽是轻量子状态机:只改相邻两子比例,反应式写 `prism_ui_store`,**不触发无关子树重算**(§9)。

---

## 6. 渲染:布局解算 → Element

- **几何**:`DockNode` 树映射到 `prism_ui_layout` 的 `LayoutTree`——`Split` → flex 行/列 + 子节点显式比例尺寸;`TabStack` → tab 栏 + 内容区。确定性解算,相同树相同几何。
- **面板外观**:每个可见 tab 内容包进 `prism_ui_component_kit::editor::DockPanel`(复用其玻璃面 + 标题栏 + `DockSide`),本层只负责**摆放与装配**。
- **浮窗 / 拖拽预览**:走 `prism_ui_overlay`(portal 层 + z 序 + FocusTrap);浮窗带投影(DockPanel 的 `--floating` 已有 lift 样式)。
- **分隔条**:可拖拽把手元素,带命中区 padding(易抓)。
- **增量**:Loom 协调器只 patch 变化区域;切 tab 只换该 leaf 内容子树,拖分隔条只更新两侧盒几何。

---

## 7. 持久化:序列化 / 校验 / 迁移 / Workspace

- **可逆文本/JSON 编码**:参照 `prism_ui_snapshot` 的 `serialize_tree`/`parse_tree` 可逆编码,把 `DockLayout`(树 + 比例 + tab 顺序 + 激活项 + 浮窗 rect/z + autohide)写成稳定文本。对齐 VS Code 布局 JSON / Unity `.wlt`。
- **schema 版本化**:每份布局带 `schema_version`;读旧版走迁移链(字段改名/默认补全),对齐 reflect 迁移心智。
- **校验(§11)**:加载即校验——未知面板 id 丢弃、比例钳制归一、深度/数量上限、损坏回落默认。
- **Workspace / Perspective(借 Qt ADS / Blender)**:命名多套布局(「编码」「调试」「美术」),一键切换;每套独立持久化。
- **自动保存**:`prism_ui_store` middleware 在布局变更后节流落盘(原子写,见 §12.6)。
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
    pub factory: PanelFactory,      // 懒构建:() -> Element 内容
}
pub trait PanelRegistry {
    fn register(&mut self, desc: PanelDescriptor);
    fn resolve(&self, id: &PanelId) -> Option<&PanelDescriptor>; // 持久化白名单
}
```

- **贡献点**:插件/域编辑器注册面板,不改停靠内核。对齐编辑器 D3「贡献点而非硬编码」。
- **单例 vs 多例**:Inspector 单例;文档/图表可多例。
- **懒工厂**:内容只在面板首次可见时构建(§9/K4)。
- **白名单**:持久化只认注册过的 id,反序列化未知 id 优雅丢弃(§11)。

---

## 9. 性能工程:成本 ∝ 变化量

- **懒构建**:仅可见 tab 的 `factory()` 执行;后台 tab/autohide 折叠面板内容不进树,激活时才建、隐藏可回收。
- **虚拟化**:tab 很多时 tab 栏用 `prism_ui_virtual` 窗口化;超长面板内容本就由各面板自行虚拟化。
- **脏子树重算**:拖分隔条只重算该 split 子树几何(`prism_ui_layout` 局部);其余节点几何缓存不动。
- **拖拽解耦**:拖拽 ghost 在 overlay 层独立合成,**不触发**底层布局重排,直到落下才过一次 reducer。分隔条实时比例走 store signal,面板内容经 selector 精确隔离——拖分隔条**不重渲面板内容**。
- **结构共享**:reducer 返回新树时共享未变子树(持久化数据结构),diff 便宜,利撤销栈(§12.4)。
- **可测性能**:暴露指标(可见面板数、每帧重算节点数、拖拽帧延迟),CI 门控回归。

---

## 10. 效果(视觉)

- **玻璃表面**:复用 `DockPanel` 的 frosted glass + 主题 token(`color.separator`/`glass.tint`),停靠区整体材质统一。
- **drop 高亮**:落区半透明填充 + 边框脉冲,`prism_ui_anim`/`prism_ui_motion` 做淡入淡出与方向提示。
- **贴靠/切分动效**:面板落位、split 产生/折叠用弹性过渡(spring),非瞬跳,降低空间认知负担。
- **浮窗**:`--floating` 投影 lift;拖拽时轻微缩放/透明 ghost。
- **最大化过渡**:leaf 全屏/还原做几何插值动画,保留心智连续性。
- **分隔条**:悬停加粗 + 光标反馈;拖拽实时跟手(无延迟,靠 §9 解耦)。
- 所有动效**可按性能档关闭**(低端设备/CI 确定性模式),不牺牲功能。

---

## 11. 安全

停靠内核的攻击面主要是**持久化布局(来自磁盘/云同步,可能损坏或被构造)**:

- **不可信反序列化**:加载布局时——
  - **未知面板 id**:注册表白名单外的一律丢弃,不实例化任何未注册内容(无「从布局执行任意代码」路径)。
  - **数值钳制**:比例 `[min, 1-min]` 并归一;浮窗 rect 钳到屏幕可见范围;z 序去重。
  - **资源上限**:最大面板数、最大嵌套深度、最大浮窗数、最大 tab 数——防构造深树/海量节点导致的栈溢出/OOM(DoS)。
  - **不 panic**:任何畸形结构走 `LayoutError` 回落默认布局,绝不崩溃。
  - **schema 迁移**:版本不符走迁移链,不可迁移字段隔离。
- **无代码注入**:布局只含 id/数值/结构,不含可执行内容;面板行为完全由注册表(进程内可信代码)决定。
- **云同步隔离**:多设备同步的布局同样过上述校验,一台的损坏布局不污染内核。

---

## 12. 稳定性

### 12.1 不变量(invariants)
reducer 每次操作后强制:无悬挂空节点、比例归一、每面板实例唯一归属、文档区始终存在且不可被吞并、浮窗各自合法树。可用 `debug_assert` + 属性测试(proptest 式随机操作序列后校验不变量)兜底。

### 12.2 确定性
相同 action 序列 → 相同 `DockLayout`;可 snapshot 回归(复用 `prism_ui_snapshot`)。拖拽状态机脱离 GPU 可测。

### 12.3 优雅收敛
关最后一个 tab、折叠空 split、浮窗清空自动关闭、拖拽取消回原位——所有边界都有确定收敛行为,不留中间态。

### 12.4 撤销 / 重做(可选)
布局操作可记入命令栈(对齐编辑器命令内核),结构共享让历史廉价;「误拖散布局」可一键撤销。

### 12.5 多显示器 / DPI
浮窗跨屏定位、DPI 缩放交平台/`prism_window` 层;本层只存逻辑 rect + 显示器 id,恢复时由平台层校正到当前显示器拓扑(屏幕消失则拉回主屏)。

### 12.6 崩溃安全持久化
原子写(临时文件 + rename)+ 保留 last-good;加载失败回落 last-good → 默认。编辑器崩溃重启布局不丢。

---

## 13. 可达性 / DPI / 主题

- **键盘导航**:`prism_ui_input::FocusRing` 在面板/tab 间 tab 序移动;快捷键移动/切分面板、切 tab、聚焦相邻面板。
- **角色/语义**:`DockPanel::role() == Role::Group`;tab 栏暴露 tablist/tab 角色,激活态可被辅助技术读出。
- **FocusTrap**:模态浮窗用 `prism_ui_overlay::FocusTrap` 困住焦点。
- **DPI/HDR/主题**:几何用逻辑像素由布局解算,渲染后端做 DPI 缩放;颜色走主题 token,支持明暗/HDR。

---

## 14. Crate 结构 / 模块 / feature

```
prism_ui_dock/
├── model/        # DockNode / DockLayout / FloatingWindow(纯数据, no_std)
├── reduce/       # action + 确定性 reducer + 不变量 + 折叠/归一
├── drag/         # 拖拽状态机 + drop zone 计算(用 input hit_test/gesture)
├── render/       # DockNode → LayoutTree → Element(用 layout/overlay/DockPanel)
├── persist/      # 序列化/反序列化 + 校验 + schema 迁移 + Workspace
├── registry/     # PanelDescriptor / PanelRegistry / 贡献点
├── store/        # prism_ui_store 集成 + 自动保存 middleware
└── a11y/         # 键盘导航 + 角色
```
features: `std`(持久化落盘)/ `anim`(动效)/ `snapshot_tests`。核心 model/reduce/persist 为 `no_std + alloc`。

---

## 15. 路线图(M0–M5)

| 里程碑 | 内容 | 依赖 |
|---|---|---|
| **M0** | model + reduce(split/tabify/close/resize)+ 不变量 + 属性测试 | — |
| **M1** | render(layout 解算 + DockPanel 装配 + tab 栏) | prism_ui_layout ✅ / DockPanel ✅ |
| **M2** | 拖拽状态机 + 五向 drop zone + overlay 预览 | prism_ui_input ✅ / prism_ui_overlay ✅ |
| **M3** | 浮窗 + autohide 边栏 + maximize | M1/M2 |
| **M4** | 持久化(序列化/校验/迁移)+ Workspace + 自动保存 | prism_ui_store ✅ |
| **M5** | 动效 + 可达性 + 撤销集成 + 多显示器校正 | 平台层 / 命令内核 |

---

## 16. 公共 API 草图(示意,非最终)

```rust
let mut registry = DockRegistry::new();
registry.register(PanelDescriptor::new("inspector").title("Inspector")
    .slot(DefaultSlot::RightBar).singleton().factory(|| inspector_view()));
registry.register(PanelDescriptor::new("outliner").title("Outliner")
    .slot(DefaultSlot::LeftBar).singleton().factory(|| outliner_view()));

// 布局状态进 store(响应式 + 自动保存)
let dock = DockStore::new(DockLayout::default_from(&registry));
dock.autosave_to(path, Throttle::ms(500));

// action 驱动
dock.dispatch(DockAction::Split { target: leaf, axis: Axis::Horizontal, side: Side::Right });
dock.dispatch(DockAction::Move { panel: "inspector".into(), dest, pos: DropPos::Center });

// 渲染
let view = DockView::new(&dock, &registry).render(); // → prism_ui::Element

// 加载(不可信输入,自动校验/迁移/回落)
let layout = DockLayout::load(&bytes, &registry)?; // 未知面板丢弃,比例钳制,损坏回落默认
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
- ⬜ **全部 PLANNED(本 crate 尚未创建)**:`prism_ui_dock` 的 model / reduce / drag / render / persist / registry / store / a11y。本文所有 `rust` 代码块、API、drop zone/状态机均为**设计意图**,不代表已实现。
- ⬜ **依赖未落地项**:撤销集成需编辑器命令内核;多显示器校正需平台/`prism_window` 层;动效档位依赖性能分级。
- 现状复核(本次已核验):`pkg` 全局搜 dock 布局管理、split/tab/float/持久化**均无实现**;唯一相关是 `DockPanel` 展示组件。本设计不推翻它,而是在其上补「宿主」层。

---

## 19. 术语表

- **DockNode / DockLayout**:描述工作台的纯数据布局树与其根 + 浮窗 + 自动隐藏栏的聚合。
- **TabStack**:叶子节点,多个面板叠为标签页,记录激活项;单面板也是单 tab 栈。
- **drop zone**:拖拽悬停时的落区(中心 tabify / 四边 split / 外缘整区停靠 / tab 间重排)。
- **reducer**:把 action 作用到布局的纯函数,维护所有不变量,确定性可测。
- **Workspace / Perspective**:命名的多套布局,一键切换(借 Qt ADS/Blender)。
- **autohide 边栏**:折叠到四边、悬停弹出的工具窗模式(借 VS Code/Fleet)。
- **last-good**:上一份成功加载的布局,崩溃/损坏时的恢复兜底。

---

> 本文为 `prism_ui_dock` 的 v1 设计规格。实现须遵循 `prism_bevy_refactor_plan_zh.md` 的「pkg 优先、`prism_*` 唯一真相源」方向,与 `prism_editor_framework_design_zh.md`(§5 外壳)、`prism_ui_component_kit_design_zh.md`(DockPanel 外观)、`prism_ui_loom_design_zh.md`(layout/overlay/store/input/virtual 地基)协同演进。所有 Prism crate 不含任何 Unreal Engine / Unity / VS Code / JetBrains 源码或衍生代码;仅借鉴公开架构形态与经典数值。
