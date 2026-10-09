# Prism `prism_ui_component_kit` 组件库设计方案

> 面向 Loom 声明式 UI 的**开箱即用控件库**(Component Kit),工作名沿用 Loom 体系。
> 定位类比 SwiftUI Controls / shadcn primitives / Ant Design:**只做组装,严格复用下层**,不另起炉灶。
> 视觉基调取自用户 Sketch / DTCG 导出的「Liquid Glass」风格(日夜双模式),
> 但**命名一律用 `glass`,禁用 `ios`**。
>
> 本文为**设计规格**,严格区分「已交付」(SHIPPED)与「规划中」(PLANNED),
> 不把未落地能力描述为已落地。

- 版本: v0.5(**控件层已交付 SHIPPED**;§5 全部清单落地,141 个控件文件,585 测试全绿,Clippy 零告警,`--no-default-features` no_std 通过)
- crate 名: **`prism_ui_component_kit`**(单一控件库 crate,内部按目录分模块)
- 适用引擎: Prism / Bevy ECS 生态,Loom 声明式 UI
- 铁律: `#![forbid(unsafe_code)]`、`no_std` 友好(`extern crate alloc`)、每控件配测试、Clippy 零告警、三层解耦
- v0.4 变更: 组件清单改为表格;对齐大型库补齐 A–E 共约 30 个缺口组件;新增 `motion/` 与 `utils/` 原语目录

---

## 目录
1. 定位与现状结论
2. 设计原则(铁律)
3. 分层架构与模块目录
4. 组件 API 规约
5. 组件总清单(按目录分组,表格)
6. 玻璃材质作为一等变体
7. 横切能力(日夜 / a11y / 交互态 / 受控)
8. 专题:取色(ColorPicker / ColorWheel / ColorInput)
9. 专题:时间(DatePicker / TimePicker / DateRangePicker / Calendar)
10. 专题:表单结构件(Form 布局基建)
11. 专题:对齐大型库补齐组件(v0.4,A–E)
12. 专题:引擎 / 编辑器专用组件
13. 专题:节点编辑器(node_editor)
14. Workbench 集成与质量门禁
15. 路线图(P0–P6)
16. 风险与取舍
17. 术语表

---

## 1. 定位与现状结论

Loom 的组件相关基础设施**已齐全**,但**尚无任何具体控件**。本库补齐「把积木拼成开箱即用控件」这一层,
并把节点编辑器等引擎专用件也一并收入(同一 crate,内部按目录隔离)。

### 1.1 现状盘点(已核实,SHIPPED)

| 层 | crate | 已有能力 |
|---|---|---|
| 组件模型 | `prism_ui_component`(单数) | `Component` trait、`FnComponent`、`Props`、`ContextMap`(DI)、`Slots`、`lifecycle` |
| 表单 | `prism_ui_form` | `Form` 响应式状态、`required/min_len/max_len/int_range/int_step/email/luhn/iban/pattern/custom` 校验 |
| 工作台 | `prism_ui_workbench` | Storybook 式 `Story / ArgSet / ControlValue / Workbench / render_story`、`fuzzy` 模糊匹配 |
| 无障碍 | `prism_ui_a11y` | `Role`(Button/Dialog/Checkbox/Tab/Menu/MenuItem…)、焦点序、键盘导航 |
| 样式 | `prism_ui_style` | `Class.with_state(InteractionState)`、`with_glass`、`with_shadow`、token 级联解析 |
| 主题 | `prism_ui_theme` | `glass()` 全套日夜 token |
| 浮层 | `prism_ui_overlay` | 浮层定位 / 种类(Portal / FocusScope 底座) |
| 虚拟化 | `prism_ui_virtual` | 长列表虚拟化(Table / Tree / AssetGrid / InfiniteScroll 复用) |
| 树 | `prism_ui_tree` | 分代 Arena 树(TreeView / TreeSelect / Cascader 复用) |
| 文本 | `prism_ui_text` | 文本整形(RichText / Code / Highlight 复用) |
| 反射 / 内省 | `prism_reflect` / `prism_ui_inspector` | 属性反射(PropertyGrid 复用) |

### 1.2 关键结论(PLANNED 的理由)

对 `Button / Widget` 的全仓检索命中的都是**输入事件**、**a11y 角色**、**overlay 种类**——
**没有任何可复用控件**。因此需要新建 `prism_ui_component_kit`。

### 1.3 命名辨析

```
prism_ui_component       (已存在,单数)  → 组件模型:Component trait / Props / Slots / DI
prism_ui_component_kit   (新建)          → 控件库:Button / TextField / ColorPicker / NodeEditor ...
```
`_kit` 后缀清晰表达「成套控件」,与单数的「模型层」无歧义。

---

## 2. 设计原则(铁律)

- **K1 样式不进控件**:控件只贴 class 名,外观由主题侧 `Class` 定义,颜色全部 `StyleValue::token(...)`。
  → 日夜切换、换肤对控件**零改动**。
- **K2 复用优先**:状态走 `prism_ui_form`,角色走 `prism_ui_a11y`,浮层 / Portal / 焦点陷阱走 `prism_ui_overlay`,
  长列表走 `prism_ui_virtual`,树走 `prism_ui_tree`,文本走 `prism_ui_text`,交互态走 `InteractionState`;**不自造轮子**。
- **K3 变体即枚举**:外观差异用 `Variant / Size` 枚举表达,映射到固定 class 名。
- **K4 可组合**:复杂控件拆成可独立测试的小件(`CalendarGrid`、`SwatchGrid`、`NodeCanvas`…),再组合。
- **K5 no_std 友好**:不引第三方日期 / 颜色库,纯算法内置并配单测。
- **K6 每控件配 stories**:变体 × 尺寸 × 交互态 × 日夜全覆盖,快照即验收。
- **K7 底座先行**:`Popover` 是 Tooltip / Menu / ContextMenu / HoverCard / Popconfirm / 各 Picker 浮层的统一底座,优先落地。

---

## 3. 分层架构与模块目录

**所有控件(含节点编辑器)都在同一个 crate `prism_ui_component_kit`,内部按目录分模块**:

```
prism_ui_component_kit
   ├─ dep  prism_ui / prism_ui_component / prism_ui_style / prism_ui_theme
   ├─ dep  prism_ui_a11y / prism_ui_form / prism_ui_overlay
   ├─ dep  prism_ui_virtual / prism_ui_tree / prism_ui_text
   ├─ dep  prism_reflect / prism_ui_inspector
   └─ dev-dep prism_ui_workbench

src/
 ├─ lib.rs            # 门面 re-export
 ├─ kit.rs            # ControlSize / 公共变体枚举 / interactive() 态 helper
 ├─ preset.rs         # 控件级 Class 预设(贴主题 token)
 ├─ basics/           # L0:button / label / icon / divider / spacer / avatar / tag / kbd / code
 │                    #     heading / link / image(async_image) / video(media_player)
 │                    #     blockquote / highlight(mark)
 ├─ inputs/           # L1:text_field / text_area / toggle / checkbox / radio / slider / range_slider
 │                    #     stepper / number_field / select / multi_select / combobox / segmented
 │                    #     search_field / tag_input / file_field / pin_input
 │                    #     password_field / masked_input
 │                    #     transfer / cascader / tree_select / mentions / dropzone / toggle_group
 │                    #     form/(form_field / form_label / form_error / fieldset / input_group)
 ├─ pickers/          # color_picker / color_wheel / color_input
 │                    #     date_picker / date_range_picker / time_picker
 │                    #     gradient_editor / curve_editor + calendar.rs(纯日历算法)
 ├─ display/          # 数据展示:table / tree_view / list / section / badge / stat / timeline
 │                    #     empty_state / carousel / rating
 │                    #     descriptions / calendar(完整视图) / image_list(masonry) / number_ticker
 ├─ containers/       # L2:card / sheet / stack / scroll / scroll_area / grid / aspect_ratio
 │                    #     accordion / split_view / tabs(tab_list / tab_panel) / infinite_scroll
 ├─ feedback/         # L3:popover / tooltip / hover_card / menu / context_menu / dialog / alert
 │                    #     toast / spinner / progress / skeleton / command_palette / banner
 │                    #     popconfirm / result / tour / float_button(speed_dial)
 ├─ nav/              # L4:nav_bar / tab_bar / toolbar / status_bar / sidebar / breadcrumb
 │                    #     pagination / drawer / wizard / anchor(scroll_spy) / affix / back_top
 ├─ editor/           # 引擎/编辑器专用:property_grid / vector_field / transform_field
 │                    #     knob / dial / gauge / dock_panel / chart
 │                    #     timeline_sequencer / asset_browser(asset_grid) / console(log_view)
 │                    #     ruler(guides)
 ├─ node_editor/      # 节点编辑器:canvas / node / port / edge / minimap / graph(图数据模型)
 ├─ motion/           # 动效原语:transition / fade / slide / scale / collapse
 └─ utils/            # 工具原语:portal / visually_hidden / watermark / qr_code / copy_button
```

> 节点编辑器虽重,但按要求收在同一 crate 的 `node_editor/` 目录,靠模块边界隔离;视觉外壳复用 kit 玻璃卡片 + token。
> `node_editor` / `editor` / `media` / `qr` 等重模块以 **feature gate** 控制编译面。

---

## 4. 组件 API 规约

统一「**声明式 builder + token 驱动样式 + 变体枚举**」。每控件遵循同一骨架:

```rust
pub enum ButtonVariant { Filled, Tinted, Gray, Plain, Glass }
pub enum ControlSize   { Small, Medium, Large }

pub struct Button {
    label: String,
    variant: ButtonVariant,
    size: ControlSize,
    disabled: bool,
    leading: Option<Element>,   // Slots:图标插槽
    role: Role,                 // 默认 Role::Button
}

impl Button {
    pub fn new(label: impl Into<String>) -> Self { /* 默认 Filled / Medium */ }
    pub fn variant(mut self, v: ButtonVariant) -> Self { self.variant = v; self }
    pub fn size(mut self, s: ControlSize) -> Self { self.size = s; self }
    pub fn glass(self) -> Self { self.variant(ButtonVariant::Glass) }
    pub fn disabled(mut self, d: bool) -> Self { self.disabled = d; self }
    pub fn leading(mut self, icon: Element) -> Self { self.leading = Some(icon); self }
}

impl Component for Button {
    type Props = ();
    fn render(&self, _: &()) -> Element {
        Element::box_()
            .class(self.variant.class_name())   // 如 "btn-glass"
            .class(self.size.class_name())       // 如 "ctl-md"
            .child(Element::text(&self.label))
    }
}
```

要点:外观由 `preset.rs` 的 `Class` 决定,颜色全走 token;控件代码**不含像素**。
交互态复用 `Class.with_state(...)`;子内容 / 图标用 `Slots`。

---

## 5. 组件总清单(按目录分组)

> 图例:★ = 编辑器 / 引擎刚需;◇ = 玻璃默认浮层;**v0.4** = 本版对齐大型库新增。

### 5.1 `basics/` — L0 基础与排版

| 组件 | 说明 / 与易混项区分 | 复用下层 | 标记 |
|---|---|---|---|
| `Button` | 五变体三尺寸五态 | `prism_ui_style` / a11y | — |
| `Label` / `Text` | 文本标签 | `prism_ui_text` | — |
| `Icon` | 图标 | — | — |
| `Divider` | 分隔线(`color.separator`) | token | — |
| `Spacer` | 弹性间距 | — | — |
| `Avatar` | 头像(图/字/占位) | — | — |
| `Tag` / `Chip` | 标签徽记 | token | — |
| `Kbd` | 键位提示 | — | — |
| `Code` | 等宽代码(行内) | `prism_ui_text` | — |
| `Heading` | 标题层级 h1–h6 | `prism_ui_text` | — |
| `Link` | 超链接 | a11y | — |
| `Image` / `AsyncImage` | 占位 + 懒加载 | `prism_ui_async`(media feature) | — |
| `Video` / `MediaPlayer` | 媒体播放 | media feature | — |
| `Blockquote` | 排版引用块 | `prism_ui_text` | **v0.4** |
| `Highlight` / `Mark` | 搜索命中文本高亮 | `prism_ui_text` | **v0.4** |

### 5.2 `inputs/` — L1 输入(接 `prism_ui_form`)

| 组件 | 说明 / 与易混项区分 | 复用下层 | 标记 |
|---|---|---|---|
| `TextField` | 单行输入 | form | — |
| `TextArea` | 多行输入 | form | — |
| `Toggle`(Switch) | 开关 | form | — |
| `Checkbox` | 勾选 | form / a11y | — |
| `Radio` / `RadioGroup` | 单选 | form / a11y | — |
| `Slider` | 单值滑块 | form | — |
| `RangeSlider` | 双端区间滑块 | form | — |
| `Stepper` | 步进加减 | form | — |
| `NumberField` | 数值输入 | form(`int_range`/`int_step`) | — |
| `Select` / `Picker` | 单选下拉 | form / overlay ◇ | ◇ |
| `MultiSelect` | 多选下拉 | form / overlay ◇ | ◇ |
| `SegmentedControl` | 单选分段 | form | — |
| `Combobox` / `Autocomplete` | 可输入筛选下拉 | form / overlay ◇ | ◇ |
| `SearchField` | 搜索框 | form | — |
| `TagInput` / `TokenField` | 标签录入 | form | — |
| `FileField` | 文件选择按钮 | — | ★ |
| `PinInput` / `OTP` | 分格验证码 | form | — |
| `PasswordField` | 显隐切换 | form | — |
| `MaskedInput` | 电话 / 卡号格式化 | form(`pattern`/`luhn`) | — |
| `Transfer` | 穿梭框(左右列表搬移) | form / virtual | ★ **v0.4** |
| `Cascader` | 级联选择(省市区层级) | form / tree ◇ | ◇ **v0.4** |
| `TreeSelect` | 树形下拉(≠ 平铺 `Select`) | form / tree ◇ | ◇ **v0.4** |
| `Mentions` | `@` 提及输入 | form / overlay ◇ | ◇ **v0.4** |
| `Dropzone` / `FileDropzone` | 拖拽上传区(≠ `FileField` 按钮) | form | ★ **v0.4** |
| `ToggleGroup` | 可按下按钮组(≠ 单选 `Segmented`) | form / a11y | **v0.4** |

**表单结构件(`inputs/form/`,布局基建 ≠ 校验状态,详见第 10 节)**

| 组件 | 说明 | 复用下层 |
|---|---|---|
| `FormField` | 字段容器:label + 控件 + error,映射 `Form` 态 | form |
| `FormLabel` | 字段标签 | — |
| `FormError` | 错误文案(`color.red`,`touched && invalid` 才显示) | form |
| `Fieldset` | 分组 + 组标题 | a11y 分组角色 |
| `InputGroup` | 前后缀 addon(图标 / 单位 / 按钮) | — |

### 5.3 `pickers/` — 选择器(详见第 8、9 节)

| 组件 | 说明 / 与易混项区分 | 复用下层 | 标记 |
|---|---|---|---|
| `ColorPicker` | swatch 网格(吃 `sys.*` 色板 token) | overlay ◇ | ◇ |
| `ColorWheel` | 连续色环 HSV(≠ swatch 网格) | — | — |
| `ColorInput` | 文本色值输入(≠ 面板,`#RRGGBB`/token) | form | **v0.4** |
| `DatePicker` | 单日选择(`CalendarGrid` 弹层) | overlay ◇ | ◇ |
| `DateRangePicker` | 起止区间(双 `CalendarGrid`) | overlay ◇ | ◇ |
| `TimePicker` | 时 / 分 / 秒三列 | overlay ◇ | ◇ |
| `GradientEditor` | 渐变编辑 | — | — |
| `CurveEditor` | 曲线编辑(缓动 / 动画) | — | ★ |
| `calendar.rs` | `no_std` 纯日历算法(内部件) | — | — |

### 5.4 `display/` — 数据展示

| 组件 | 说明 / 与易混项区分 | 复用下层 | 标记 |
|---|---|---|---|
| `Table` / `DataGrid` | 可排序 + 虚拟化 | virtual | ★ |
| `TreeView` | 树形展示 | tree | ★ |
| `List` / `ListRow` | 列表 | virtual | — |
| `Section` | 分区 | — | — |
| `Badge` | 角标 | token | — |
| `Stat` / `KPI` | 指标卡 | — | — |
| `Timeline` | **事件**时间轴(≠ editor 的 `Sequencer`) | — | — |
| `EmptyState` | 空状态占位 | — | — |
| `Carousel` | 轮播 | — | — |
| `Rating` | 评分星 | form | — |
| `Descriptions` | 详情页键值对版式 | — | **v0.4** |
| `Calendar` | **完整月 / 周视图**(≠ `DatePicker` 弹层小 grid) | calendar.rs | ★ **v0.4** |
| `ImageList` / `Masonry` | 瀑布流图墙 | virtual | **v0.4** |
| `NumberTicker` | 数字滚动动画(配 `Stat`) | motion | **v0.4** |

### 5.5 `containers/` — L2 容器

| 组件 | 说明 / 与易混项区分 | 复用下层 | 标记 |
|---|---|---|---|
| `Card` | 卡片 | token | — |
| `Sheet` | 玻璃面板 | overlay ◇ | ◇ |
| `Stack` | VStack / HStack | 布局 | — |
| `ScrollView` | 原生滚动 | — | — |
| `ScrollArea` | 自定义滚动条(≠ 裸 `ScrollView`) | — | — |
| `Grid` | 网格布局 | 布局 | — |
| `AspectRatio` | 宽高比盒 | — | — |
| `Accordion` / `Disclosure` | 折叠面板 | motion(collapse) | — |
| `SplitView` / `Resizable` | 可拖拽分栏 | — | ★ |
| `Tabs` / `TabList` / `TabPanel` | **内容**标签页(≠ nav 的 `TabBar`) | a11y `Role::Tab` | — |
| `InfiniteScroll` / `LoadMore` | 滚动加载 | virtual | **v0.4** |

### 5.6 `feedback/` — L3 反馈(接 `prism_ui_overlay`)

| 组件 | 说明 / 与易混项区分 | 复用下层 | 标记 |
|---|---|---|---|
| `Popover` | 通用锚定浮层(**下列浮层的底座**) | overlay | ◇ |
| `Tooltip` | 轻提示 | overlay | ◇ |
| `HoverCard` | 富内容悬浮卡(≠ `Tooltip`) | overlay | ◇ |
| `Menu` / `MenuItem` | 菜单 | overlay / a11y | ◇ |
| `ContextMenu` | 右键菜单 | overlay / a11y | ★◇ |
| `Dialog` | 模态对话框 | overlay / FocusScope | ◇ |
| `Alert` | 行内警示 | token | — |
| `Toast` | 瞬时通知 | overlay | ◇ |
| `Spinner` / `ProgressBar` | 加载 / 进度 | — | — |
| `Skeleton` | 骨架屏 | — | — |
| `CommandPalette` | 命令面板(`fuzzy`) | overlay / workbench | ★◇ |
| `Banner` / `InlineAlert` | 横幅提示 | token | — |
| `Popconfirm` | 锚定气泡确认(轻量于 `Dialog`) | Popover 底座 | ★◇ **v0.4** |
| `Result` / `StatusPage` | 成功 / 404 / 500 / 空结果页 | — | **v0.4** |
| `Tour` / `Coachmark` | 分步引导高亮 | overlay | ★ **v0.4** |
| `FloatButton` / `SpeedDial` | 悬浮操作按钮 + 展开 | overlay | ◇ **v0.4** |

### 5.7 `nav/` — L4 导航

| 组件 | 说明 / 与易混项区分 | 复用下层 | 标记 |
|---|---|---|---|
| `NavBar` | 顶栏(玻璃) | overlay | ◇ |
| `TabBar` | 底部导航(玻璃,≠ 内容 `Tabs`) | overlay | ◇ |
| `Toolbar` | 工具条 | — | — |
| `StatusBar` | 状态栏 | — | — |
| `Sidebar` / `NavRail` | 侧栏导航 | — | ★ |
| `Breadcrumb` | 面包屑 | — | — |
| `Pagination` | 分页器 | — | — |
| `Drawer` | 抽屉 | overlay | ◇ |
| `Wizard` / `Steps` | 分步流程 | — | — |
| `Anchor` / `ScrollSpy` | 锚点侧栏 + 滚动高亮 | — | **v0.4** |
| `Affix` | 吸顶 / 吸附固定 | — | **v0.4** |
| `BackTop` | 回到顶部 | — | **v0.4** |

### 5.8 `editor/` — 引擎 / 编辑器专用(详见第 12 节)

| 组件 | 说明 / 与易混项区分 | 复用下层 | 标记 |
|---|---|---|---|
| `PropertyGrid` / `PropertyInspector` | 反射驱动属性面板 | reflect / inspector | ★ |
| `VectorField` | xyz(w) 成组数值 | form | ★ |
| `TransformField` | 位移 / 旋转 / 缩放 | form | ★ |
| `Knob` / `Dial` | 旋钮 | — | — |
| `Gauge` | 旋针仪表 | — | — |
| `DockPanel` | 可停靠 / 拖拽重排面板 | — | ★ |
| `Chart` | 图表(对接 visualize) | — | — |
| `Timeline` / `Sequencer` | **动画 / 音频**时间轴(≠ display 事件 `Timeline`) | — | ★ |
| `AssetBrowser` / `AssetGrid` | 资产浏览 | virtual | ★ |
| `Console` / `LogView` | 日志面板(等级过滤 + 搜索) | `Code` | ★ |
| `Ruler` / `Guides` | 标尺 / 辅助线 | — | — |

### 5.9 `node_editor/` — 节点编辑器(详见第 13 节)

| 组件 | 说明 | 复用下层 | 标记 |
|---|---|---|---|
| `NodeCanvas` | 无限画布,平移 / 缩放 | — | ★ |
| `Node` | 节点卡片(玻璃壳) | kit 玻璃 | — |
| `Port` | 端口(按类型着色) | `sys.*` token | — |
| `Edge` / `Wire` | 贝塞尔连线 | — | — |
| `Minimap` | 缩略导航(≠ 游戏小地图) | — | — |
| `graph` | 图数据模型(独立可测) | — | — |

### 5.10 `motion/` — 动效原语(**v0.4** 新增)

| 组件 | 说明 | 复用下层 | 标记 |
|---|---|---|---|
| `Transition` | 进出场过渡调度(统一时长 / 缓动) | — | **v0.4** |
| `Fade` | 淡入淡出 | Transition | **v0.4** |
| `Slide` | 滑入滑出 | Transition | **v0.4** |
| `Scale` | 缩放出现 | Transition | **v0.4** |
| `Collapse` | 高度折叠(供 `Accordion` 复用) | Transition | **v0.4** |

### 5.11 `utils/` — 工具原语(**v0.4** 新增)

| 组件 | 说明 / 与易混项区分 | 复用下层 | 标记 |
|---|---|---|---|
| `Portal` | 传送门(薄封装,若 `prism_ui_overlay` 已暴露则直接转发) | overlay | **v0.4** |
| `VisuallyHidden` | a11y 视觉隐藏文本 | a11y | **v0.4** |
| `Watermark` | 水印 | — | **v0.4** |
| `QRCode` | 二维码(`qr` feature) | — | **v0.4** |
| `CopyButton` / `Clipboard` | 一键复制 + 反馈 | — | **v0.4** |

---

## 6. 玻璃材质作为一等变体

直接复用主题 `glass()` 的 token + `with_glass` / `with_shadow`:

```rust
Class::new("surface-glass")
    .with_glass(20.0,
                StyleValue::token("glass.tint"),
                Some(StyleValue::token("glass.highlight")))
    .with_shadow(0.0, 8.0, 24.0, StyleValue::token("glass.shadow"));
```

默认用法:标 ◇ 的浮层 / 导航类默认玻璃;实体控件(Button / Card)玻璃为**可选**变体,默认 `color.fill.*` 实体填充。
玻璃为**近似实现**(半透明叠色 + 高光描边 + 投影),~90% 观感、~10% 成本;真 backdrop blur 留给 GPU 后端提示。

---

## 7. 横切能力(每个控件默认具备)

- **日夜双模式**:零成本——颜色皆 token,`glass()` 的 light / dark override 自动生效。
- **a11y**:构造时写死合理 `Role` 默认,可覆盖 label;容器接 `FocusOrder`。
- **交互态**:统一五态;`kit.rs` 提供 `interactive(class)` helper,自动补各态 fill / opacity token。
- **受控 / 非受控**:输入类统一接 `prism_ui_form::Form`,`TextField::bind(form, "email")` 即双向绑定 + 校验。
- **动效**:进出场统一走 `motion/` 原语,时长 / 缓动 token 化,低端可一键关闭。

---

## 8. 专题:取色(ColorPicker / ColorWheel / ColorInput)

- `ColorPicker`:**预设色板直接吃现有 token**——`glass()` 的 `sys.*` 色板日夜双值,swatch 网格日夜自动跟随;
  结构 `Swatch` × N → `SwatchGrid` → `ColorPicker`;值模型存 `StyleValue`(具体色或 token 引用);浮层 `Popover + .glass()`。
- `ColorWheel`:连续色环(HSV / HSB),**≠** swatch 离散网格;与 `Slider`(明度 / Alpha)组合;共享同一值模型。
- `ColorInput`:**文本色值输入框**(`#RRGGBB` / `rgba()` / token 名),≠ 面板;接 `form` 校验与解析,可挂 `ColorPicker` 弹层。

```rust
pub struct ColorPicker { value: StyleValue, swatches: Vec<StyleValue>, allow_custom: bool, role: Role }
pub struct ColorWheel  { value: StyleValue, show_alpha: bool }
pub struct ColorInput  { value: StyleValue, with_picker: bool }
```

---

## 9. 专题:时间(DatePicker / TimePicker / DateRangePicker / Calendar)

- **值 + 校验走 `prism_ui_form`**(`pattern` / `custom`,范围步长复用 `int_range` / `int_step`)。
- 拆小可组合:`TimePicker` = 时 / 分 / 秒三列 `Picker`;`DatePicker` = `CalendarGrid`(7×6)+ 月份翻页;
  **`DateRangePicker`** = 双 `CalendarGrid` + 起止高亮(`color.tint` / `color.fill`)。
- **`Calendar`**(display):**完整月 / 周视图**面板(事件 / 日程承载),≠ `DatePicker` 的弹层小 grid,共享 `calendar.rs`。
- 浮层 `Popover + .glass()`;`calendar.rs` 为 `no_std` 纯日历算法,独立模块 + 单测。

```rust
pub struct Date { pub year: i32, pub month: u8, pub day: u8 }
pub fn month_grid(y: i32, m: u8) -> [[Option<u8>; 7]; 6];
```

---

## 10. 专题:表单结构件(Form 布局基建)

`prism_ui_form` 只管**状态 + 校验**;本组补**结构与布局**,把字段标签、控件、错误信息组织成一致版式:

- `FormField`:单字段容器,串起 `FormLabel` + 任意输入控件 + `FormError`,并把 `Form` 的 `touched/dirty/error` 映射到 class 态。
- `FormLabel` / `FormError`:标签与错误文案(错误用 `color.red` token,仅在 `touched && invalid` 时显示)。
- `Fieldset`:分组 + 组标题(接 a11y 分组角色)。
- `InputGroup`:前后缀 addon(图标 / 单位 / 按钮),包裹单个输入控件。

```rust
FormField::new("email", form)
    .label("邮箱")
    .child(TextField::bind(form, "email"))   // 校验来自 form 已注册的 validators
// 错误自动显示 form.first_error("email")
```

---

## 11. 专题:对齐大型库补齐组件(v0.4,A–E)

对齐 Ant Design / MUI / Radix·shadcn / Mantine / Fluent 后补齐的缺口,**仍严格遵守 K1–K7**(复用下层、不自造轮子):

### A. 数据录入(`inputs/`)
- `Transfer`:左右双列表 + 中间搬移,内部两张 `List`(接 `virtual`)+ 选择态。
- `Cascader`:层级下拉,逐级展开,数据接 `prism_ui_tree`;浮层 `Popover + .glass()`。
- `TreeSelect`:树形下拉选择(≠ 平铺 `Select`),复用 `TreeView` 作为弹层内容。
- `Mentions`:`@` 触发联想浮层,命中插入 token,接 `fuzzy`。
- `Dropzone`:拖拽文件区(≠ `FileField` 按钮),含拖入高亮态与多文件队列。
- `ToggleGroup`:一组可按下按钮(单选 / 多选,≠ `Segmented` 的单选分段)。

### B. 数据展示(`display/`)
- `Descriptions`:标签-值网格,详情页高频(行 / 列数、边框变体)。
- `Calendar`:**完整月 / 周视图**(≠ `DatePicker` 弹层),承载事件 / 日程,共享 `calendar.rs`。
- `ImageList` / `Masonry`:瀑布流图墙,接 `virtual` + `AsyncImage`。
- `NumberTicker`:数值滚动动画,配 `Stat`,走 `motion/` 调度。

### C. 反馈 / 操作(`feedback/`)
- `Popconfirm`:锚定气泡确认(轻量于 `Dialog`),直接复用 `Popover` 底座 + 两枚 `Button`。
- `Result` / `StatusPage`:操作结果页(成功 / 404 / 500 / 空),图标 + 文案 + 操作区。
- `Tour` / `Coachmark`:分步引导,高亮目标 + 浮层说明 + 上一步 / 下一步。
- `FloatButton` / `SpeedDial`:常驻悬浮操作按钮,点开放射子项(玻璃)。

### D. 导航 / 布局(`nav/`)
- `Anchor` / `ScrollSpy`:文档锚点侧栏,滚动联动高亮当前节。
- `Affix`:元素吸顶 / 吸附固定在视口。
- `BackTop`:回到顶部浮钮(可与 `FloatButton` 共壳)。

### E. 工具 / 原语(`motion/` + `utils/`)
- **动效原语 `motion/`**:`Transition` 统一调度,派生 `Fade` / `Slide` / `Scale` / `Collapse`;
  时长 / 缓动 token 化;供 `Dialog` / `Toast` / `Drawer` / `Accordion` / `NumberTicker` 复用,**不在各控件各写一套**。
- **`utils/`**:`Portal`(薄封装 `prism_ui_overlay`,若已暴露直接转发)、`VisuallyHidden`(薄封装 `prism_ui_a11y`)、
  `Watermark`、`QRCode`(`qr` feature)、`CopyButton` / `Clipboard`。

> **先确认再薄封装**:`Portal` / `VisuallyHidden` / `FocusScope` 很可能已在 `prism_ui_overlay` / `prism_ui_a11y`,
> 若已直接可用则只做 re-export,不重复实现。

### 范围取舍
- **暂不收**:移动端触控组(`PullToRefresh` / `SwipeAction` / `ActionSheet` / `IndexBar` / `NoticeBar`)——
  属 Vant / Antd-Mobile 领域,桌面编辑器 + 引擎未必需要,留作可选 feature,视引擎定位再定。
- **引擎正数**:`editor/` 与 `node_editor/` 是大型通用库普遍没有的能力,属本库相对主流库的超集部分。

---

## 12. 专题:引擎 / 编辑器专用组件(`editor/`)

- **`PropertyGrid`**:接 `prism_reflect` 描述符 + `prism_ui_inspector`,按字段类型自动选控件(数值→`NumberField`、颜色→`ColorPicker`、向量→`VectorField`、枚举→`Select`)。
- **`VectorField` / `TransformField`**:xyz(w)成组数值,支持拖拽标量、统一步进。
- **`Knob` / `Dial` / `Gauge`**:旋钮 / 仪表(音频与游戏调参)。
- **`DockPanel`**:可停靠 / 拖拽重排面板系统。
- **`Chart`**:图表,可对接 visualize 能力。
- **`Timeline` / `Sequencer`**:动画 / 音频时间轴(轨道 / 关键帧 / 播放头),**≠** display 的事件 `Timeline`。
- **`AssetBrowser` / `AssetGrid`**:资产浏览(缩略图网格 + 过滤 + 虚拟化,接 `prism_ui_virtual`)。
- **`Console` / `LogView`**:日志面板(等级过滤 + 搜索,等宽 `Code`)。
- **`Ruler` / `Guides`**:标尺与辅助线(画布 / 布局对齐)。

均复用 kit token、玻璃外壳与 `InteractionState`。

---

## 13. 专题:节点编辑器(`node_editor/`)

对齐 `docs/prism_visual_scripting_design_zh.md`,作为**通用节点图编辑器**:

- `NodeCanvas`:无限画布,平移 / 缩放(视图变换矩阵),网格背景。
- `Node`:节点卡片(玻璃外壳,标题 + 输入 / 输出端口区),可内嵌 `PropertyGrid`。
- `Port`:端口,按数据类型用 `sys.*` 色板 token 着色。
- `Edge` / `Wire`:贝塞尔连线,命中测试、拖拽重连。
- `Minimap`:缩略导航(**≠** 游戏运行时小地图)。
- 交互:框选、吸附对齐、多选拖动;图数据模型 `graph` 独立可测。

---

## 14. Workbench 集成与质量门禁

- 每控件带 `stories()`,变体 × 尺寸 × 交互态 × 日夜注册为 `Story`,`ControlValue` 暴露可调参数。
- 验收:stories 覆盖全变体且 `render_story` 快照稳定。
- 门禁:`cargo test -p prism_ui_component_kit` 全绿、`cargo clippy --all-targets` 零告警、`#![forbid(unsafe_code)]`、`no_std` 构建通过。

---

## 15. 路线图(P0–P6)

| 阶段 | 内容 | 产出 |
|---|---|---|
| P0 | 建 `prism_ui_component_kit` + `kit.rs` + `preset.rs` 控件级 `Class` 预设 | 骨架编译通过 |
| P1 | `basics/`:L0 + 排版富媒体(Heading / Link / Image / AsyncImage / Video / **Blockquote / Highlight·Mark**)+ `Button` 五变体 + `motion/`(Transition 原语)+ stories | 玻璃按钮日夜切换 + 过渡 |
| P2 | `inputs/` 基础 + **表单结构件** + PasswordField / MaskedInput / MultiSelect / **ToggleGroup / Mentions** + `Form` 绑定 | 带校验的玻璃表单 |
| P2.5 | `pickers/`:ColorPicker / ColorWheel / **ColorInput** + `calendar.rs` + DatePicker / DateRangePicker / TimePicker | 取色与时间选择器 |
| P3 | **底座与刚需**:`Popover` 底座 → Tooltip / HoverCard / Menu / ContextMenu / **Popconfirm**;`containers/Tabs` + `ScrollArea`;`display/TreeView` / `Table` / **Calendar**;`editor/PropertyGrid`;`containers/SplitView` | 编辑器地基可跑 |
| P3.5 | `containers/`(+ **InfiniteScroll**)+ `nav/`(NavBar / TabBar / Sidebar / Drawer + **Anchor / Affix / BackTop**)+ `inputs/` 补充(RangeSlider / Combobox / TagInput / FileField / **Transfer / Cascader / TreeSelect / Dropzone**) | 完整玻璃界面壳 |
| P4 | `feedback/` 补充(+ **Result / Tour / FloatButton**)+ `display/` 补充(+ **Descriptions / ImageList·Masonry / NumberTicker**)+ `editor/`(VectorField / TransformField / Knob / Gauge / Chart / DockPanel / Timeline / Sequencer / AssetBrowser / Console·LogView / Ruler·Guides)+ `pickers/`(GradientEditor / CurveEditor)+ `utils/`(**Portal / VisuallyHidden / Watermark / QRCode / CopyButton**) | 组件库 v1 |
| P5 | `node_editor/`:`graph` → NodeCanvas / Node / Port / Edge / Minimap + 框选吸附 | 通用节点编辑器 |
| P6 | 全库 a11y 审查 + Table / Tree / AssetGrid / InfiniteScroll 虚拟化落地 + 性能回归用例 | 组件库 v1.1 |

---

## 16. 风险与取舍

- **crate 体量**:全部收入单 crate 会变大(v0.4 后约 140+ 控件);以**目录 + feature gate**(`node_editor` / `editor` / `media` / `qr`)控制编译期与依赖面。
- **命名易混**:`prism_ui_component` vs `prism_ui_component_kit`;`Tabs`(内容)vs `TabBar`(导航)、`Timeline`(事件)vs `Timeline/Sequencer`(时间轴)、`ColorWheel` vs `ColorPicker` vs `ColorInput`、`ScrollArea` vs `ScrollView`、`HoverCard` vs `Tooltip`、`Calendar`(完整视图)vs `DatePicker`(弹层)、`TreeSelect` vs `Select`、节点 `Minimap` vs 游戏小地图——文档与命名已显式区分。
- **薄封装去重**:`Portal` / `VisuallyHidden` / `FocusScope` 先确认下层是否已暴露,已有则仅 re-export,避免重复造轮子。
- **媒体件依赖**:`Image/AsyncImage/Video/ImageList` 需异步加载与解码,接 `prism_ui_async`,置于 `media` feature,低端可降级为占位。
- **玻璃近似**:CPU 参考光栅器忽略 `glass.blur`,仅 GPU 后端用作提示;低端降级为纯 `fill`。
- **动效集中**:所有进出场走 `motion/`,避免各控件各写一套;低端可全局关闭过渡。
- **范围蔓延**:凡涉及新状态机 / 新样式原语,回到下层 crate 实现,保持本库薄;移动端触控组默认不收。

---

## 17. 术语表

- **Kit**:成套开箱即用控件(本库)。
- **Variant**:控件外观变体枚举(如 `ButtonVariant::Glass`)。
- **Glass**:近似「Liquid Glass」材质(半透明叠色 + 高光 + 投影),命名禁用 `ios`。
- **Token**:主题设计令牌,颜色 / 标量的间接名,日夜双值自动解析。
- **InteractionState**:交互五态(Normal / Hover / Focus / Pressed / Disabled)。
- **Popover 底座**:通用锚定浮层,Tooltip / HoverCard / Menu / ContextMenu / Popconfirm / Picker 浮层的共同实现基础。
- **FormField**:表单结构件,串联 label + 控件 + error,把 `Form` 状态映射为 class 态。
- **Transition**:`motion/` 的过渡调度原语,Fade / Slide / Scale / Collapse 的共同基础。
- **Story**:Workbench 中一个控件状态用例,用于预览与快照测试。
