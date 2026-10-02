# Loom — Prism 的声明式 UI / 场景系统

> 工作名 **Loom**(织机,寓意「编织实体树」)。
> 目标:在 Prism / ECS 架构上,提供一套 **比 Bevy BSN 更好用、性能更可控、表现力更强、更易维护** 的声明式 UI 与场景系统。

Loom 是一套 **保留模式(retained-mode)** 的视图系统,设计灵感来自
SolidJS / Leptos、SwiftUI、Jetpack Compose、Flutter 与 Tailwind,并针对
游戏引擎的性能预算重新设计。它 **与引擎解耦**:运行时只计算「发生了什么变化」,
并向可插拔的 [`Backend`] 发出最小化的操作流,因此同一份视图代码既能跑在
无头测试后端上,也能驱动 GPU 渲染器。

## 核心契约

> **成本 ∝ 变化量,而非场景规模。**

每帧构建一棵 `Element` 树是廉价的;运行时把它与保留的状态做协调(reconcile),
只把 **真正的增量** 告诉后端。这是 Loom 全部性能设计的出发点,也是它区别于
「整树 diff」或「场景级 patch」方案的根本。

## 快速上手

构建器 API(数据层,始终可用):

```rust
use prism_ui::{Element, RecordingBackend, Ui};
use prism_ui::layout::{AvailableSpace, Size};

// 一个纵向盒子,内含一个文本子节点。
let view = Element::box_().child(Element::text("hello"));

let mut ui = Ui::new(RecordingBackend::new());
ui.mount(&view);
ui.compute_layout(Size::new(
    AvailableSpace::Definite(800.0),
    AvailableSpace::Definite(600.0),
));

// 物化了两个节点(盒子 + 文本)。
assert_eq!(ui.node_count(), 2);

// 重新渲染一棵相同的树,不会产生任何新建操作。
let before = ui.backend().len();
ui.update(&view);
assert_eq!(ui.backend().len(), before);
```

`loom!` 宏(语法层,编译期降解为同样的构建器调用):

```rust
use prism_ui::{loom, ElementKind};

let view = loom! {
    box {
        class: "card";
        style: {
            flex_direction: column;
            width: px(300.0);
            background_color: token("color.surface");
        };
        text("标题");
        box { class: "row"; }
    }
};

assert_eq!(view.kind(), &ElementKind::Box);
```

## 架构:三层分离

```
结构层 Structure  →  loom! 宏 / Element 构建器(编译期可解释)
响应层 Reactivity →  Signal / Memo / Effect(运行期,细粒度)
样式层 Style      →  token / class / 级联 / 断点(资产,可热重载)
         ↓ 全部落到 ↓
保留树 + Keyed 协调器 → Flexbox 布局 → 最小化 BackendOp 流 → 渲染后端
```

详见 [docs/architecture.md](docs/architecture.md)。

## Crate 组成与状态

Loom 以多个独立 crate 分层实现,每一层都可单独使用。下表为 **已交付(SHIPPED)** 的部分:

**核心层(Core)** — 数据层 `Element` → 保留树 → 最小化 `BackendOp` 的完整链路:

| Crate | 职责 | 状态 | 测试 |
|---|---|---|---|
| [`prism_ui_reactive`](../prism_ui_reactive) | 无毛刺(glitch-free)的 Signal / Memo / Effect 图 | ✅ 已交付 | 19 |
| [`prism_ui_tree`](../prism_ui_tree) | 分代 Arena、保留树、LIS 最小化的 keyed 协调器 | ✅ 已交付 | 9 |
| [`prism_ui_style`](../prism_ui_style) | design token、class、选择器、级联 | ✅ 已交付 | 13 |
| [`prism_ui_layout`](../prism_ui_layout) | 纯 Rust Flexbox 求解器 | ✅ 已交付 | 15 |
| [`prism_ui_anim`](../prism_ui_anim) | 缓动、弹簧、时间线、过渡 | ✅ 已交付 | 33 |
| [`prism_ui`](.) | 伞 crate:`Element` / `Ui` 运行时 / `Backend` | ✅ 已交付 | 28 |
| [`prism_ui_macro`](../prism_ui_macro) | `loom!` 声明式 DSL(proc-macro);`$` **响应式读取语法糖**(`$expr` → 受追踪 `Signal` 读取) | ✅ 已交付 | 17 |

**高级层(Advanced)** — 对标 React/SolidJS/Vue 生态的一等能力,全部构建在上述核心层之上:

| Crate | 职责 | 状态 | 测试 |
|---|---|---|---|
| [`prism_ui_component`](../prism_ui_component) | 组件模型:`Component` trait、props、具名多插槽、Context 注入 | ✅ 已交付 | 14 |
| [`prism_ui_store`](../prism_ui_store) | 可预测全局状态:`Store` + 细粒度选择器 + 中间件 | ✅ 已交付 | 11 |
| [`prism_ui_i18n`](../prism_ui_i18n) | 国际化:响应式消息目录、插值、CLDR 复数选择 | ✅ 已交付 | 9 |
| [`prism_ui_router`](../prism_ui_router) | 响应式客户端路由:路径匹配、`:param` / `*wildcard`、历史栈 | ✅ 已交付 | 15 |
| [`prism_ui_devtools`](../prism_ui_devtools) | 内省工具:树快照、`render_tree` 美化输出、`BackendOp` 轨迹统计 | ✅ 已交付 | 9 |
| [`prism_ui_overlay`](../prism_ui_overlay) | Portal / Overlay:模态 / popover / tooltip / toast、z-order、backdrop、`FocusTrap` | ✅ 已交付 | 22 |
| [`prism_ui_form`](../prism_ui_form) | 响应式表单:双向绑定、`Validator`(required/min/max/int_range/pattern/custom)、touched/dirty、errors memo | ✅ 已交付 | 20 |
| [`prism_ui_virtual`](../prism_ui_virtual) | 列表虚拟化:定高/变高(前缀和二分)、overscan、回收池、spacer+可见项 | ✅ 已交付 | 25 |
| [`prism_ui_async`](../prism_ui_async) | 异步健壮性:`AsyncState` / `Resource`、Suspense、Error Boundary、`all` 聚合 | ✅ 已交付 | 13 |
| [`prism_ui_a11y`](../prism_ui_a11y) | 无障碍:`Role` / `AriaState` / `Label`、`A11yTree`、`FocusOrder`、`KeyboardNav`、`LiveRegion` | ✅ 已交付 | 49 |
| [`prism_ui_scoped`](../prism_ui_scoped) | 组件作用域样式(`ScopeId` 稳定散列、class 命名空间化)、响应式 `@media` 断点解析(mobile-first 级联) | ✅ 已交付 | 26 |
| [`prism_ui_motion`](../prism_ui_motion) | 自动过渡:隐式样式过渡(记忆旧值→Tween)、FLIP 布局动画、共享元素(Hero)过渡 | ✅ 已交付 | 37 |
| [`prism_ui_snapshot`](../prism_ui_snapshot) | 快照测试:可逆文本序列化、行级 LCS diff、golden 比对(`Comparison`)、布局快照(`LayoutQuery`) | ✅ 已交付 | 29 |
| [`prism_ui_workbench`](../prism_ui_workbench) | 组件工作台(Storybook 风格):`ControlValue`/`ArgSet` 类型校验、`Story`/`StoryBuilder`、两级分层注册、隔离 harness | ✅ 已交付 | 27 |
| [`prism_ui_timetravel`](../prism_ui_timetravel) | 时间旅行调试:帧时间线(编辑器式 undo/redo)、跳转、重放(相邻帧 `diff`/变更摘要) | ✅ 已交付 | 24 |
| [`prism_ui_inspector`](../prism_ui_inspector) | 元素树检查器:`NodePath` 寻址、`Query` 过滤、`PerfReport`/`TreeMetrics` 性能面板、`DependencyGraph`(signal 依赖图:传递闭包/拓扑序/Graphviz 导出) | ✅ 已交付 | 57 |
| [`prism_ui_hotreload`](../prism_ui_hotreload) | `.loom` / `.loom.style` 热重载:节点身份(`NodePath`)比对、`ReloadPlan`(保留/新增/移除/重建)、跨重载状态保留(`StateStore`)、样式 `StyleDiff` | ✅ 已交付 | 45 |
| [`prism_ui_ecs`](../prism_ui_ecs) | **ECS 桥接(M2 headline)**:组件字段 <-> `Signal` 字段级双向绑定(`FieldBinding`/`EcsBridge`),复用 ECS tick 变更检测作传输、相等性守卫防振荡;Bevy 调度器集成(`LoomSyncSet` / `NonSend` + exclusive system);`Show`/`For` 信号驱动结构绑定(两阶段批量 spawn/despawn,`diff_keyed` LIS 最小移动,`StructuralScope`) | ✅ 已交付 | 64 |

全部 25 个 crate 累计 **630 个 lib + 集成测试通过**(表中「测试」列为各 crate `--lib --tests` 计数;另有 42 个 doctest 通过,合计 672)。每个 crate 均:`#![forbid(unsafe_code)]`、
`no_std` 友好(`default = ["std"]`,proc-macro crate 除外)、通过严格 Clippy(零告警)。

> **诚实声明**:仍为 **设计阶段(PLANNED)** 的能力包括:宏层 `$` 语法糖的 **ECS 字段绑定登记**
> 形态(当前已交付的 `$` 为 **响应式信号读取** 语法糖;自动登记 `EcsBridge` 字段绑定的形态尚需
> 宏层拿到桥/实体上下文,仍在设计)、`.loom` / `.loom.style` 热重载的 **文件系统监听集成**、
> **静态子树提升**、**双模式编译**(dev 解释 / release 宏固化)。
> 已交付并通过测试的能力包括:字段级 **ECS ↔ Signal 双向绑定** 及其 **Bevy 调度器集成**
> (`LoomSyncSet` / `NonSend` + exclusive system)、`Show` / `For` **信号驱动结构绑定**
> (两阶段批量 spawn/despawn + `diff_keyed` LIS 最小移动)、`loom!` 宏 `$` **响应式读取语法糖**、
> **编译期稳定节点 ID**、`.loom` / `.loom.style`
> **热重载核心**(节点身份比对 + 状态保留 + 样式 diff)、列表 **虚拟化**、
> **Suspense / Error Boundary**、**Portal / Overlay**、**表单校验**、**a11y 基线**、
> **作用域样式 / 响应式 @media**、**隐式过渡 / FLIP 布局动画 / 共享元素过渡**,均为引擎弱耦合 crate。见
> [docs/roadmap.md](docs/roadmap.md)。本文档不会把未实现的能力描述为已实现。
> 已交付的高级层(组件 / Store / i18n / 路由 / DevTools)目前为 **引擎弱耦合的
> 独立运行时能力**,与 Prism/ECS 实体的深度绑定仍在 roadmap 中推进。

## 文档索引

- [架构设计](docs/architecture.md) — 三层分离、数据流、成本契约。
- [与 Bevy BSN 的对比](docs/bsn-comparison.md) — 逐维度评析与 Loom 的改进。
- [分层指南](docs/layers.md) — 核心各 crate 的真实 API 与代码示例。
- [高级功能](docs/advanced-features.md) — 组件模型 / Store / i18n / 路由 / DevTools 的真实 API。
- [路线图](docs/roadmap.md) — 已交付 vs 规划中,里程碑 M1–M6。

## 许可

`MIT OR Apache-2.0`。
