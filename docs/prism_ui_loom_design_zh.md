# Prism Loom 次世代声明式 UI / 场景系统设计方案

> 面向 Prism(Bevy fork)的**保留模式、细粒度响应式、引擎弱耦合**的声明式 UI 与场景系统。
> 工作名 **Loom**(织机,寓意「编织实体树」)。
> 借鉴 Bevy BSN(官方对照基线)、SolidJS / Leptos(细粒度响应式)、SwiftUI(布局协议 / 环境)、
> Jetpack Compose(位置记忆 / Slot Table)、Flutter(RelayoutBoundary 增量布局 / 手势竞技场)、
> React Fiber(可中断渲染 / 时间切片)、Dioxus / Xilem(Rust 原生声明式)、Zed GPUI / Servo WebRender
> (GPU 驱动的保留绘制流)、cosmic-text / HarfBuzz(文本整形 / BiDi / IME)取长补短。
>
> 本文为**设计规格**:既是对**已交付**(SHIPPED)架构的权威总览,也是对 **v2 升级**(PLANNED)的
> 工程化方案。全文严格区分「已实现并通过测试」与「规划中」,不把未落地能力描述为已落地。

- 版本: v2.0(升级设计阶段;v1 核心与高级层已交付)
- 适用引擎: Prism / Bevy ECS 生态
- 当前实现规模: **25 个 `prism_ui_*` crate**,累计 **600+ 测试函数**,`#![forbid(unsafe_code)]` 为主、`no_std` 友好、Clippy 零告警
- 关键依赖: `prism_ui_reactive`(自研无毛刺响应图)、`prism_ui_tree`(分代 Arena + LIS 协调)、`prism_ui_layout`(纯 Rust Flexbox)、可插拔 `Backend` trait(与渲染后端解耦)

---

## 目录
1. 设计哲学与核心契约
2. Bevy BSN 评测:它好在哪、弱在哪
3. 分层架构总览(v1 已交付)
4. 数据流与协调内核
5. 性能工程:成本 ∝ 变化量
6. 效果:动画 / 过渡作为一等公民
7. 易用性:像写前端一样写引擎 UI
8. 易维护:三层解耦 + 可预测 + 可内省
9. **v2 升级专题**(本次设计重点)
   - 9.1 双模式编译:开发期解释 / 发布期宏固化
   - 9.2 静态子树提升与编译期常量化
   - 9.3 增量布局:RelayoutBoundary + 脏树
   - 9.4 可组合布局协议(Layout trait / Modifier 链)
   - 9.5 可中断渲染与时间切片(优先级调度)
   - 9.6 文本栈:整形 / 富文本 / BiDi / IME
   - 9.7 输入 / 命中测试 / 声明式手势竞技场
   - 9.8 GPU 驱动的保留绘制流
   - 9.9 `$` 自动字段绑定糖(宏层闭环)
   - 9.10 服务端驱动 UI(SDUI)沙箱
   - 9.11 组件生命周期钩子 / 并发资源
   - 9.12 主题管线 / 设计令牌编译
10. Crate 全景与状态矩阵
11. 与 Bevy / BSN 的兼容与共存
12. 路线图(M1–M8)
13. 风险与取舍
14. 术语表

---

## 1. 设计哲学与核心契约

对标成熟前端框架与现代 GPU UI 栈,确立六条铁律:

- **P1 静态/动态显式分界**:编译期就知道「哪里会变」。静态部分零运行时构造;动态部分细粒度订阅。
- **P2 拒绝 archetype 搬迁**:运行期更新只改组件「值」,不插/删组件;频繁切换的状态用枚举 / 标志字段表达。
  这是对 BSN「增删组件触发 archetype 搬迁」陷阱的结构性规避。
- **P3 关注点分离**:结构、响应、样式三层各自独立编写、测试、热重载;改一层不动其它层。
- **P4 可预测优先于极致**:宁可牺牲一点理论峰值,也要让性能行为稳定、可解释、可断言为回归用例。
- **P5 渐进采用**:能与现有命令式 ECS 代码共存,局部替换,不强制全量迁移。
- **P6 失败可见**:错误边界、校验、报错 span 映射,错误不静默。

### 核心契约

> **成本 ∝ 变化量,而非场景规模。**

每帧构建一棵 `Element` 树是廉价的(纯数据,无 ECS 写入);运行时把它与保留的状态做协调(reconcile),
只把**真正的增量**(`SetText` / `SetPaint` / `Reorder` / `SetLayout` …)告诉后端。这是 Loom 全部性能
设计的出发点,也是它区别于「整树 diff」或「场景级 patch」方案的根本。关键在于:这个契约是**可测试的**——
`RecordingBackend` 把「本次产生了哪些操作」变成可断言的事实(见 §5)。

---

## 2. Bevy BSN 评测:它好在哪、弱在哪

BSN(Bevy Scene Notation,随 `bevy_scene` 下一代场景格式演进)是 Bevy 官方的声明式实体/场景方案,
用 `bsn!` 宏与 `.bsn` 资产描述实体树、组件与 required components。客观评价:

### 2.1 BSN 的优点(必须尊重)
- **与 ECS 原生同构**:节点即实体,属性即组件,`spawn` 即落地,无额外运行时模型,心智最短。
- **required components**:声明一个组件自动拉起其依赖组件,减少样板。
- **资产化 + 热重载潜力**:`.bsn` 作为资产,天然可被 `bevy_asset` 管理、复用、热重载。
- **生态整合度**:与 Bevy 调度器、查询、系统、反射(`bevy_reflect`)无缝衔接,这是任何第三方方案难以企及的。
- **数据驱动**:场景可被工具链(编辑器)读写,利于可视化编辑。

### 2.2 BSN 的弱点(升级的切入点)
- **更新成本不可预测**:显隐 / 选中态等高频切换若用「增删组件」表达,会触发 **archetype 搬迁**,
  成本随实体规模与组件组合爆炸,且难以被使用者直观度量。性能取决于使用者是否踩中陷阱。
- **缺少细粒度响应式**:没有一等的 Signal / Memo / Effect 依赖图;「值变了只重算依赖子图」需使用者手搭系统 + 变更检测拼接。
- **结构即场景,缺少中间 IR**:`bsn!` 直接 spawn 实体,缺少一层廉价的、可每帧重建、可 diff 的**数据层视图描述**,
  因此很难做「整树重建 + 最小增量落地」这种成本可控的模型。
- **高级能力非一等公民**:组件/插槽/Context、Store、路由、i18n、表单、Overlay、虚拟化、动画编排……
  在 BSN 里基本靠宏片段拼接或外接系统,缺少独立、可组合、可测试的分层。
- **动画/过渡需外接**:进出场、隐式过渡、FLIP 布局动画、共享元素过渡不是内建一等能力。
- **稳定身份与热重载对齐**:结构变更后如何**保留运行时状态**、精确对齐节点身份,BSN 尚无成熟位置记忆机制。
- **文本/输入/a11y 深度**:复杂文本整形、BiDi、IME、命中测试、手势、读屏语义等「真实 UI 难题」覆盖有限。

### 2.3 结论
BSN 的优势是**生态原生集成度**,这是 Loom 不与之正面竞争、而选择**共存**的理由(见 §11)。
Loom 的差异化取舍是:**引擎弱耦合 + 字段级可绑定 + 可测试的最小增量更新 + 成熟前端的分层能力**。
两者定位不同,可长期并存:底层仍可落到 ECS 实体,但中间多了一层成本可控、可内省、可热重载的声明式栈。

---

## 3. 分层架构总览(v1 已交付)

```
┌──────────────────────────────────────────────────────────┐
│ 结构层 Structure(编译期 / 可解释)                           │
│   loom! 宏 / Element 构建器 → 数据层视图描述(含 $ 响应式读取)  │
├──────────────────────────────────────────────────────────┤
│ 响应层 Reactivity(运行期,细粒度,无毛刺)                     │
│   Signal / Memo / Effect;字段级双向绑定 EcsBridge;Show / For │
├──────────────────────────────────────────────────────────┤
│ 样式层 Style(资产,可热重载)                                │
│   token / class / 作用域(scoped) / @media 响应式断点 / 级联   │
└──────────────────────────────────────────────────────────┘
            ↓ 全部落到 ↓
┌──────────────────────────────────────────────────────────┐
│ 保留树 Tree(分代 Arena + LIS Keyed 协调器 + 稳定节点 ID)      │
│ 布局引擎(纯 Rust Flexbox)→ 最小化 BackendOp 流 → 渲染后端    │
└──────────────────────────────────────────────────────────┘
```

每一层都是独立 crate,可单独使用、单独测试、单独演进。这是「易维护」的结构性保证。
高级层(组件 / Store / 路由 / i18n / 表单 / Overlay / 虚拟化 / 异步 / a11y / 动画 / 工具链)
**不新造状态模型**,而是复用核心响应图——保证成本契约对高级层同样成立。

### 结构层 API(两种写法,等价降解)

构建器 API(数据层,始终可用):

```rust
use prism_ui::{Element, RecordingBackend, Ui};
use prism_ui::layout::{AvailableSpace, Size};

let view = Element::box_().child(Element::text("hello"));
let mut ui = Ui::new(RecordingBackend::new());
ui.mount(&view);
ui.compute_layout(Size::new(AvailableSpace::Definite(800.0), AvailableSpace::Definite(600.0)));
assert_eq!(ui.node_count(), 2);

// 重新渲染一棵相同的树,不产生任何新建操作(成本契约的最小证明)。
let before = ui.backend().len();
ui.update(&view);
assert_eq!(ui.backend().len(), before);
```

`loom!` 宏(语法层,编译期降解为同样的构建器调用,保留 ident span 以精确报错):

```rust
use prism_ui::{loom, ElementKind};

let view = loom! {
    box {
        class: "card";
        style: { flex_direction: column; width: px(300.0); background_color: token("color.surface"); };
        text("标题");
        box { class: "row"; }
    }
};
assert_eq!(view.kind(), &ElementKind::Box);
```

`$` **响应式读取语法糖**(已交付):`text($sig)` 降级为受追踪的 `(sig).get()`——在 `ReactiveView`
的 effect 内构建时自动登记依赖,信号变更触发该子树重算。纯 `.get()` 糖,无隐藏状态。

---

## 4. 数据流与协调内核

```
用户输入 / 事件
      │
      ▼
  Signal 更新 ──→ 标记依赖的 Memo / Effect 为脏(无毛刺拓扑调度)
      │
      ▼
  构建新的 Element 树(廉价、数据层、可每帧重建)
      │
      ▼
  Ui::update(&tree)  ← 核心协调步骤
      ├─ keyed 协调(LIS 最小化):复用能复用的节点,只对差异发出 BackendOp
      ├─ 身份优先级:Stable(编译期位置记忆)> Keyed(显式 key)> Positional(下标)
      ├─ 种类变更(Box↔Text↔Custom)→ 局部重建
      └─ 其它字段差异 → SetText / SetPaint / Reorder 等最小操作
      ▼
  Ui::compute_layout(viewport) → 仅对几何变化的节点发 SetLayout
      ▼
  Backend::apply(ops) → 渲染 / 无头记录
```

**关键:没有整树 diff。** 协调器以保留树为基准,逐层做 keyed 比对,输出的 `BackendOp` 流严格正比于
「实际变化」。列表增删/重排用 **LIS(最长递增子序列)** 求最少移动次数。

---

## 5. 性能工程:成本 ∝ 变化量

| 机制 | 作用 | 对应 crate | 状态 |
|---|---|---|---|
| 数据层 `Element` | 每帧重建视图描述是廉价的(无 ECS 写入) | `prism_ui` | ✅ |
| LIS 最小化 keyed 协调 | 列表增删/重排按 key 复用节点,最小 move | `prism_ui_tree` | ✅ |
| 字段级 BackendOp | 只发出 `SetText` / `SetPaint` / `Reorder` 等真实增量 | `prism_ui` | ✅ |
| 布局几何差分 | `compute_layout` 只对坐标/尺寸变化节点发 `SetLayout` | `prism_ui(_layout)` | ✅ |
| 无毛刺响应传播 | 值未变不传播;只扰动真正依赖者 | `prism_ui_reactive` | ✅ |
| 字段回写相等性守卫 | ECS 回写值未变不置脏,掐断「写→tick→又拉取」振荡 | `prism_ui_ecs` | ✅ |
| 文本测量只用 ×÷ | 确定性测量,无超越函数依赖 | `prism_ui` | ✅ |
| 静态子树提升 | 无绑定子树编译期常量化 | `prism_ui_macro` | 🔜 v2(§9.2) |
| 双模式编译 | 发布期宏固化、零解析开销 | `prism_ui_macro` | 🔜 v2(§9.1) |
| 增量布局 RelayoutBoundary | 布局脏传播止于边界,子树尺寸稳定不重算 | `prism_ui_layout` | 🔜 v2(§9.3) |

**可测试的性能契约**(已钉死为回归用例):
- 「相同输入零新增操作」
- 「keyed 反转:0 新建 / 0 删除 / 1 reorder」
- 「显隐改字段而非增删组件:0 archetype 搬迁」

BSN 的性能取决于使用者是否踩中 archetype 搬迁等陷阱,较难直接度量;Loom 把它变成**测试契约**。

---

## 6. 效果:动画 / 过渡作为一等公民(已交付)

- `prism_ui_anim`:缓动曲线(含 Newton / 二分求解的 CubicBezier)、解析式阻尼弹簧、关键帧时间线、
  进出场 `Transition`、编排(stagger / sequence / parallel)。
- `prism_ui_motion`:**隐式过渡**(属性变化即自动补间,打断可重定目标)、**FLIP 布局动画**
  (First/Last/Invert/Play)、**共享元素 / Hero 过渡**(按 `Key` 配对源→目标 + 进出场回退)。

这些在 BSN 里通常要外接动画系统并手动驱动;在 Loom 里是声明式、可组合、可测试的内建能力。

---

## 7. 易用性:像写前端一样写引擎 UI

- `loom!` DSL 可读、报错精准(保留 ident span)。
- 组件 / 插槽 / Context / Store / 路由 / i18n / 表单 / Overlay 一应俱全,心智模型与 React/Solid 对齐。
- 工作台(`prism_ui_workbench`)提供 Storybook 式隔离预览。
- v2 的 `$` 自动字段绑定糖(§9.9)让「组件字段 ⇄ 信号」一行声明。

---

## 8. 易维护:三层解耦 + 可预测 + 可内省

- **三层各自单测、各自演进**:迭代样式系统不必动响应模型或结构宏。
- **单一更新机制**:Store / Router / I18n 全部以 `Signal` / `Memo` 暴露,接入同一张无毛刺依赖图——
  高级层没有引入第二套更新机制,这是「性能可预测」与「易维护」的根本保证。
- **全链路可内省**:树快照、`OpTrace` 操作轨迹、`PerfReport` 性能面板、signal 依赖图
  (`DependencyGraph`:传递闭包 / Kahn 拓扑序 / Graphviz 导出)、时间旅行回放与相邻帧 diff、
  快照回归测试。调试体验对标 React/Redux DevTools。

---

## 9. v2 升级专题(本次设计重点)

v1 已把「细粒度响应 + 最小增量 + 分层高级能力」做实。v2 的目标是补齐**真实 UI 栈的硬骨头**
(文本 / 输入 / 布局性能 / 渲染后端),并把三项规划中的编译期优化落地,使 Loom 从「可用」走向
「在 AAA 游戏 UI 预算下可预测地优秀」。

### 9.1 双模式编译:开发期解释 / 发布期宏固化

**借鉴**:Compose 的 Slot Table、Flutter 的 hot reload、以及「解释器快改 + AOT 快跑」的通用取舍。

- **开发期(interpret)**:`.loom` 结构 + `.loom.style` 作为**资产**被解释执行,文件变更即热重载,
  复用已交付的 `prism_ui_hotreload`(`NodePath` 身份比对 + `ReloadPlan` + `StateStore` 状态裁剪 +
  `StyleDiff`),**保留运行时状态**。极速反馈闭环,无需重编译。
- **发布期(freeze)**:同一份 `.loom` 经构建脚本**宏固化**为 Rust 构建器调用,解析/构建开销归零,
  静态子树被常量化提升(§9.2)。
- **一致性保证**:两条路径共享同一个 AST 下降器(`prism_ui_macro::lower`)与同一套语义测试向量,
  用快照测试(`prism_ui_snapshot`)钉死「解释结果 == 固化结果」。

**设计要点**:解释器需沙箱化(§9.10),固化器需稳定排序以保证可复现构建(reproducible build)。

### 9.2 静态子树提升与编译期常量化

**借鉴**:SolidJS 的「静态模板克隆」、编译器的常量折叠。

- `loom!` 宏在下降期对**不含任何 `$` 绑定 / 信号依赖**的子树标注 `Static`,提升为 `once_cell` 常量模板;
  运行期直接克隆引用,跳过逐节点构造与 diff。
- 动态与静态在同一棵树混排:只有动态「岛屿」(islands)参与协调,静态骨架零成本。
- 与 §9.3 增量布局协同:静态子树天然是 `RelayoutBoundary`,其内部几何在父尺寸不变时永不重算。

### 9.3 增量布局:RelayoutBoundary + 脏树

**借鉴**:Flutter 的 `RelayoutBoundary` 与 `markNeedsLayout` 脏传播、Taffy 的缓存。

当前 `prism_ui_layout` 每次 `compute_layout` 全量求解。v2 引入:

- **脏标记传播**:节点尺寸约束变化时 `mark_needs_layout()`,脏沿父链上溯,**止于 RelayoutBoundary**
  (尺寸由自身约束唯一决定、不受子内容影响的节点,如定宽定高盒子)。
- **测量缓存**:按 `(约束, 内容哈希)` 缓存 flexbox 子问题结果;命中则跳过。
- **两级失效**:`needs_layout`(几何重算)与 `needs_paint`(仅重绘)分离,避免「改颜色却重排版」。

成本模型从 O(节点数) 降到 O(受影响子树),与核心契约「成本 ∝ 变化量」一致。

### 9.4 可组合布局协议(Layout trait / Modifier 链)

**借鉴**:SwiftUI 的 `Layout` 协议、Compose 的 `Modifier` 链、Flutter 的 `RenderObject`。

现状只有 Flexbox。v2 抽象一个**布局协议**,让自定义布局成为一等扩展点:

```rust
pub trait LayoutProtocol {
    fn measure(&self, children: &[MeasureHandle], constraints: Constraints) -> Size;
    fn place(&self, children: &mut [PlaceHandle], bounds: Rect);
}
```

- 内建实现:`Flex`(已有)、`Grid`、`Stack`(z 叠放)、` Absolute`、`Wrap`。
- `Modifier` 链:`padding / margin / size / aspect_ratio / align / clip` 以**可组合、顺序敏感**的
  修饰符表达,降解为约束变换,而非魔法字段。对标 Compose「Modifier 顺序即语义」。
- 自定义布局可被单测:给定约束与子尺寸,断言 `place` 输出的矩形集合。

### 9.5 可中断渲染与时间切片(优先级调度)

**借鉴**:React Fiber 的可中断协调、时间切片、`useTransition`/优先级车道(lanes)。

游戏 UI 必须守住帧预算(16.6ms / 8.3ms)。v2 让协调可被**预算打断**:

- `update_budgeted(tree, deadline)`:协调以子树为单位增量推进,逼近 deadline 时让出,
  下帧继续,保证输入与动画不被大列表重建阻塞。
- **优先级车道**:输入响应 / 动画 > 可见区内容 > 屏外内容。高优先级更新可抢占低优先级的协调。
- 与 `prism_ui_virtual`(虚拟化)协同:屏外项以最低优先级构建,overscan 空闲时预热。

**约束**:保留模式 + 不可变 `Element` 树使「暂停-恢复」安全(无半完成的可变状态);这是相对
命令式即时模式 UI 的结构优势。

### 9.6 文本栈:整形 / 富文本 / BiDi / IME

**借鉴**:HarfBuzz(整形)、cosmic-text / swash(Rust 文本栈)、ICU(BiDi / 分段)。

现状文本测量是「只用 ×÷ 的确定性近似」,够测试但不足以驱动真实多语言 UI。v2 引入 `prism_ui_text`:

- **整形(shaping)**:字形定位、连字、kerning;可插拔整形后端(默认 swash/HarfBuzz,测试用确定性桩)。
- **富文本**:span 级样式(字重 / 斜体 / 颜色 / 下划线),段落级对齐 / 行高 / 截断(ellipsis)。
- **双向文本(BiDi)**:Unicode BiDi 算法,RTL / 混排;与 §9.12 的 RTL 主题联动。
- **IME / 文本编辑**:输入法预编辑(composition)区间、光标 / 选区模型、命中测试到字形。
- **成本控制**:整形结果按 `(文本, 样式, 宽度)` 缓存;只有脏文本重整形,接入 §9.3 的 `needs_layout`。

### 9.7 输入 / 命中测试 / 声明式手势竞技场

**借鉴**:Flutter 的 **手势竞技场(gesture arena)**、SwiftUI 手势组合子、浏览器事件捕获/冒泡。

现状输入未进核心模型(roadmap 风险 6 列为「走适配层」)。v2 引入 `prism_ui_input`:

- **命中测试**:保留树 + 布局矩形 → 空间索引(可选 BVH),O(log n) 命中;支持 `pointer-events` 透传。
- **事件分发**:捕获(capture)→ 目标 → 冒泡(bubble)三阶段,可 `stop_propagation`。
- **声明式手势**:`on_tap / on_drag / on_long_press / on_pinch`,多手势**竞技场仲裁**
  (如「水平拖动」胜出则「点击」出局),避免手势冲突的经典噩梦。
- **焦点系统**:与已交付的 `prism_ui_a11y` 焦点/键盘导航合流,统一焦点环与 Tab 序。

### 9.8 GPU 驱动的保留绘制流

**借鉴**:Zed GPUI、Servo WebRender、Vello(compute 光栅化)、分层合成(layer compositing)。

`Backend` trait 已把渲染解耦。v2 提供一等 GPU 后端 `prism_ui_render_backend`:

- **保留绘制流**:协调输出的 `BackendOp` 增量落到 GPU 可见的**绘制指令缓冲**,只重传变化的实例数据。
- **批处理与实例化**:同材质/图集的矩形、文本字形合批;圆角 / 阴影 / 边框走 SDF。
- **分层合成**:动画中的子树提升为独立 layer,仅重合成不重绘制(与 §9.3 `needs_paint` 对齐)。
- **与 Prism 渲染器集成**:复用 `prism_render_*` 的 wgpu 管线,UI 作为最终合成层;离屏/无头后端用于测试。

**安全提示**:GPU 后端涉及 `unsafe` 的最小面(缓冲映射),隔离在该 crate 内并以 parity 测试对拍。

### 9.9 `$` 自动字段绑定糖(宏层闭环)

现状:`$` 的**响应式读取**已交付(§3);其**自动登记 `EcsBridge` 字段绑定**仍在设计。v2 闭环:

- 语法:`text($entity.Health.current)` → 宏在拿到桥/实体上下文时,自动登记一条 `FieldBinding`
  (读路径复用 `Ref` tick 变更检测;写路径经 `Mut` + 相等性守卫回写),零 archetype 搬迁。
- 双向:`bind!(input.value <-> $entity.Name.0)` 生成读写双通道,接入已交付的 `prism_ui_ecs::schedule`
  (`LoomSyncSet{Pull, Push}`,`.chain()` 保证拉取先于回写)。
- 宏层需解析字段路径类型,给出编译期类型错误而非运行期 panic——报错 span 映射到具体字段。

### 9.10 服务端驱动 UI(SDUI)沙箱

**借鉴**:Airbnb SDUI、Server-Driven UI 的能力白名单模型。

远端下发 `.loom` 可做 A/B、运营位、热修 UI。v2 的安全模型:

- **沙箱**:远端结构只能引用**能力白名单**内的组件 / token / 事件,禁止任意代码。
- **版本协商**:客户端声明支持的 schema 版本,服务端按版本降级;未知节点优雅回退占位。
- **校验**:下发内容经 schema 校验 + 结构沙箱裁剪后才进协调器;失败走错误边界(`prism_ui_async`)。

### 9.11 组件生命周期钩子 / 并发资源

现状组件模型(`prism_ui_component`)无生命周期钩子;路由(`prism_ui_router`)无守卫/深链接。v2 补齐:

- **生命周期**:`on_mount / on_unmount / on_update`,`on_cleanup` 自动在卸载时释放订阅(对标 Solid `onCleanup`)。
- **路由守卫 + 深链接**:`before_enter` 异步守卫、`:param` 类型化解析、可恢复的深链接状态。
- **并发资源**:已交付的 `prism_ui_async`(`Resource` + Suspense + Error Boundary)增加**竞态取消**
  (最新请求胜出)、**SWR 式缓存**(stale-while-revalidate),接入 §9.5 优先级车道。

### 9.12 主题管线 / 设计令牌编译

**借鉴**:Tailwind、Material 3 动态配色、Style Dictionary 令牌管线。

- **令牌编译**:设计令牌(`color.* / space.* / radius.*`)在构建期编译为常量表 + 运行期可切换主题的
  间接层;暗色/高对比/品牌主题切换是一次信号写入,成本 ∝ 受影响节点。
- **语义令牌**:`color.surface` → `color.gray.100`(亮)/ `color.gray.900`(暗)的语义映射,
  避免组件硬编码具体色值。
- **RTL / 国际化联动**:逻辑属性(`inline-start` 而非 `left`)随 BiDi(§9.6)自动镜像。
- 复用已交付的 `prism_ui_scoped`(作用域样式 + `@media` 断点)与 token 环检测。

---

## 10. Crate 全景与状态矩阵

> 测试数为本地粗测(`#[test]` / `#[tokio::test]` 计数),随开发推进更新;以各 crate 实际 CI 为准。

**核心层(Core)**

| Crate | 职责 | 状态 | 测试≈ |
|---|---|---|---|
| `prism_ui_reactive` | 无毛刺 Signal / Memo / Effect;只读依赖图 introspection | ✅ 已交付 | 19 |
| `prism_ui_tree` | 分代 Arena、保留树、LIS 最小化 keyed 协调 | ✅ 已交付 | 9 |
| `prism_ui_style` | design token / class / 选择器 / 级联(含 token 环检测) | ✅ 已交付 | 13 |
| `prism_ui_layout` | 纯 Rust Flexbox 求解器 | ✅ 已交付 | 15 |
| `prism_ui_anim` | 缓动 / 弹簧 / 时间线 / 过渡 / 编排 | ✅ 已交付 | 33 |
| `prism_ui` | 伞 crate:`Element` / `Ui` 运行时 / `Backend` / 最小化 op 流 | ✅ 已交付 | 28 |
| `prism_ui_macro` | `loom!` DSL + `$` 响应式读取糖 + 稳定节点 ID 注入 | ✅ 已交付 | 17 |

**结构/响应/样式增强层**

| Crate | 职责 | 状态 | 测试≈ |
|---|---|---|---|
| `prism_ui_ecs` | 字段级双向绑定 `EcsBridge` + 调度器集成 + `Show`/`For` 结构绑定 | ✅ 已交付 | 64 |
| `prism_ui_scoped` | 作用域样式 `ScopeId` + `@media` 响应式断点 | ✅ 已交付 | 26 |
| `prism_ui_hotreload` | `.loom` / `.loom.style` 热重载,保留运行时状态 | ✅ 已交付 | 45 |
| `prism_ui_a11y` | 角色 / 焦点 / 键盘导航 / 读屏标签基线 | ✅ 已交付 | 49 |

**高级功能层(Advanced)**

| Crate | 对标 | 状态 | 测试≈ |
|---|---|---|---|
| `prism_ui_component` | React / SolidJS 组件 | ✅ 已交付 | 14 |
| `prism_ui_store` | Redux / Zustand / Pinia | ✅ 已交付 | 11 |
| `prism_ui_i18n` | FormatJS / Fluent | ✅ 已交付 | 9 |
| `prism_ui_router` | React Router / Vue Router | ✅ 已交付 | 15 |
| `prism_ui_form` | React Hook Form / VeeValidate | ✅ 已交付 | 20 |
| `prism_ui_overlay` | Radix / Floating UI | ✅ 已交付 | 22 |
| `prism_ui_virtual` | TanStack Virtual | ✅ 已交付 | 25 |
| `prism_ui_async` | SolidJS Resource / Suspense | ✅ 已交付 | 13 |
| `prism_ui_motion` | Framer Motion | ✅ 已交付 | 37 |

**工具链(Tooling)**

| Crate | 对标 | 状态 | 测试≈ |
|---|---|---|---|
| `prism_ui_devtools` | React DevTools(基线) | ✅ 已交付 | 9 |
| `prism_ui_inspector` | 树检查器 + 性能面板 + 依赖图 | ✅ 已交付 | 57 |
| `prism_ui_timetravel` | Redux DevTools 时间旅行 | ✅ 已交付 | 24 |
| `prism_ui_snapshot` | Jest 快照 | ✅ 已交付 | 29 |
| `prism_ui_workbench` | Storybook | ✅ 已交付 | 27 |

**v2 规划 crate(本设计新增)**

| Crate | 职责 | 对应专题 | 状态 |
|---|---|---|---|
| `prism_ui_text` | 整形 / 富文本 / BiDi / IME | §9.6 | 🔜 规划 |
| `prism_ui_input` | 命中测试 / 事件分发 / 手势竞技场 / 焦点 | §9.7 | 🔜 规划 |
| `prism_ui_render_backend` | GPU 保留绘制流 / 合批 / 分层合成 | §9.8 | 🔜 规划 |
| `prism_ui_sdui` | 服务端驱动 UI 沙箱 + 版本协商 | §9.10 | 🔜 规划 |
| `prism_ui_theme` | 令牌编译 / 语义令牌 / 动态主题 / RTL | §9.12 | 🔜 规划 |

> `prism_ui_layout`(增量布局 §9.3、布局协议 §9.4)、`prism_ui_macro`(双模式 §9.1、静态提升 §9.2、
> `$` 绑定闭环 §9.9)、`prism_ui_component`/`prism_ui_router`/`prism_ui_async`(生命周期/守卫/竞态 §9.11)
> 为**在既有 crate 内增强**,不新建 crate。

---

## 11. 与 Bevy / BSN 的兼容与共存

Loom 不取代 BSN,而是**在其之上/之侧**提供成本可控、可内省的声明式栈。共存策略:

- **底层仍可落到 ECS**:`prism_ui_ecs` 的 `EcsBridge` 把 Loom 节点字段与 ECS 组件字段双向绑定;
  Loom 负责「变化量驱动」,ECS 负责「实体即数据」。
- **渐进采用(P5)**:可在局部用 `loom!` 描述一块 HUD / 菜单,其余仍用命令式 spawn;两者经桥共享状态。
- **互补定位**:BSN 擅长场景/关卡的静态实体布置与编辑器可视化;Loom 擅长高频交互 UI、动画、
  数据驱动列表、热重载开发闭环。
- **迁移路径**:`.bsn` → `.loom` 的半自动转换器可作为后续工具(非本设计承诺项)。

---

## 12. 路线图(M1–M8)

> 原则:未落地并本地提交前,不计入「已胜出」;提交后同步更新本文与 crate roadmap。

- **M1 结构层**:✅ `loom!` 宏 + 构建器 + `$` 读取糖 + 编译期稳定节点 ID。🔜 静态子树提升(§9.2)。
- **M2 响应→ECS 绑定**:✅ Signal/Memo/Effect、字段级双向绑定、调度器集成、`Show`/`For`、a11y 基线。
  🔜 `$` 自动字段绑定糖闭环(§9.9)。
- **M3 样式层**:✅ token/class/级联/scoped/@media/热重载核心。🔜 文件系统监听集成、主题管线(§9.12)。
- **M4 效果层**:✅ 缓动/弹簧/时间线/过渡/隐式过渡/FLIP/共享元素/编排。
- **M5 高级功能**:✅ 组件/Store/虚拟化/异步/Overlay/表单/路由/i18n。🔜 生命周期钩子、路由守卫、竞态取消(§9.11)。
- **M6 工具链**:✅ DevTools/检查器/时间旅行/依赖图/快照/工作台。🔜 双模式编译(§9.1)。
- **M7 真实 UI 栈(v2 新增)**:🔜 文本栈(§9.6)、输入/手势(§9.7)、增量布局(§9.3)、布局协议(§9.4)。
- **M8 渲染与分发(v2 新增)**:🔜 GPU 保留绘制流(§9.8)、可中断渲染(§9.5)、SDUI 沙箱(§9.10)。

**v2 建议优先级**:M7 文本栈与输入系统是「真实可用」的前置硬需求,优先于 M8;增量布局(§9.3)
与双模式编译(§9.1)是性能/体验的高杠杆项,可并行推进。

---

## 13. 风险与取舍

1. **文本栈复杂度**:整形/BiDi/IME 是独立的大工程,引入 swash/HarfBuzz 会增加依赖与 `unsafe` 面;
   取舍:隔离在 `prism_ui_text`,提供确定性测试桩,核心契约不依赖具体整形后端。
2. **可中断渲染的正确性**:时间切片下的「暂停-恢复」需保证不产生撕裂态;依赖保留模式 + 不可变树,
   并以快照测试覆盖「打断后恢复 == 一次性完成」。
3. **GPU 后端的 `unsafe` 与可移植性**:隔离最小面,用 parity 测试对拍 CPU 参考;离屏后端保证可测试。
4. **双模式一致性**:解释路径与固化路径必须语义等价;用共享下降器 + 快照向量钉死,否则「开发期对、
   发布期错」会摧毁信任。
5. **宏编译开销**:大文件宏展开可能拖慢编译 → 双模式(开发期解释)缓解。
6. **SDUI 安全**:远端 `.loom` 必须沙箱 + 能力白名单 + 版本协商,否则是远程代码执行面。
7. **范围蔓延**:v2 覆盖面大;建议按 M7→M8 分批交付,每批以「crate + 测试 + 文档」三件套闭环,
   不堆半成品。

---

## 14. 术语表

- **保留模式(retained-mode)**:维护一棵持久的节点树,更新时计算增量,而非每帧全量重绘(即时模式)。
- **协调(reconcile)**:把新构建的 `Element` 树与保留树比对,产出最小 `BackendOp` 增量的过程。
- **LIS**:最长递增子序列;用于 keyed 列表重排的最少移动求解。
- **无毛刺(glitch-free)**:响应传播保证观察者只看到一致的最终值,不经历中间不一致态。
- **archetype 搬迁**:ECS 中增删组件导致实体在原型表间搬迁的开销;Loom 以「改值不改结构」规避。
- **RelayoutBoundary**:布局脏传播的截断点;其内部几何在父约束不变时不重算。
- **手势竞技场(gesture arena)**:多手势识别器竞争同一指针序列、按规则仲裁唯一胜者的机制。
- **SDUI**:服务端驱动 UI;结构/样式由服务端下发,客户端在沙箱内渲染。
- **双模式编译**:开发期解释(热重载)、发布期宏固化(零解析)两条共享语义的编译路径。
- **岛屿(islands)**:静态骨架中参与协调的动态子树;其余静态部分零运行时成本。

---

> 本文「已交付」项均已实现并通过测试(25 crate / 600+ 测试、Clippy 零告警);详见各 crate 的
> doctest / 集成测试与 `pkg/prism_ui/docs/`。「规划中(🔜)」项在落地并本地提交前不计入「已胜出」。
