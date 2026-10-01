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

| Crate | 职责 | 状态 | 测试 |
|---|---|---|---|
| [`prism_ui_reactive`](../prism_ui_reactive) | 无毛刺(glitch-free)的 Signal / Memo / Effect 图 | ✅ 已交付 | 9 |
| [`prism_ui_tree`](../prism_ui_tree) | 分代 Arena、保留树、LIS 最小化的 keyed 协调器 | ✅ 已交付 | 9 |
| [`prism_ui_style`](../prism_ui_style) | design token、class、选择器、级联 | ✅ 已交付 | 13 |
| [`prism_ui_layout`](../prism_ui_layout) | 纯 Rust Flexbox 求解器 | ✅ 已交付 | 15 |
| [`prism_ui_anim`](../prism_ui_anim) | 缓动、弹簧、时间线、过渡 | ✅ 已交付 | 24 |
| [`prism_ui`](.) | 伞 crate:`Element` / `Ui` 运行时 / `Backend` | ✅ 已交付 | 7+2 |
| [`prism_ui_macro`](../prism_ui_macro) | `loom!` 声明式 DSL(proc-macro) | ✅ 已交付 | 8 |

每个 crate 均:`#![forbid(unsafe_code)]`、`no_std` 友好(`default = ["std"]`,
proc-macro crate 除外)、通过严格 Clippy(零告警)。

> **诚实声明**:上表以外的高级功能(响应式到 ECS 的绑定、热重载、Store、
> 列表虚拟化、Suspense、Portal、路由、i18n、DevTools、共享元素过渡)目前仍为
> **设计阶段(PLANNED)**,见 [docs/roadmap.md](docs/roadmap.md)。本文档不会把
> 未实现的能力描述为已实现。

## 文档索引

- [架构设计](docs/architecture.md) — 三层分离、数据流、成本契约。
- [与 Bevy BSN 的对比](docs/bsn-comparison.md) — 逐维度评析与 Loom 的改进。
- [分层指南](docs/layers.md) — 各 crate 的真实 API 与代码示例。
- [路线图](docs/roadmap.md) — 已交付 vs 规划中,里程碑 M1–M6。

## 许可

`MIT OR Apache-2.0`。
