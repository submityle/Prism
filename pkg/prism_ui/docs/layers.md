# Loom 分层指南

> 每个 crate 的职责与 **真实、已交付** 的 API 示例。下文示例均取自各 crate 现有的
> 公共接口与测试;标注「规划中」的能力尚未实现。

## 1. `prism_ui_reactive` — 响应层

无毛刺(glitch-free)的 Signal / Memo / Effect 图,单线程、`no_std` 友好(需 `alloc`),
模型对标 SolidJS / Leptos 的细粒度响应,而非粗粒度 VDOM diff。

```rust
use prism_ui_reactive::Runtime;
use std::cell::RefCell;
use std::rc::Rc;

let rt = Runtime::new();
let count = rt.signal(0i32);
let doubled = rt.memo({
    let count = count.clone();
    move || count.get() * 2
});

let log = Rc::new(RefCell::new(Vec::new()));
let _effect = rt.effect({
    let doubled = doubled.clone();
    let log = log.clone();
    move || log.borrow_mut().push(doubled.get())
});

assert_eq!(*log.borrow(), vec![0]);
count.set(5);
assert_eq!(doubled.get(), 10);
assert_eq!(*log.borrow(), vec![0, 10]); // 仅在值真正变化时传播
```

要点:节点只在自身值 **确实变化** 时才扰动观察者;一次更新内不会出现中间态
(无毛刺)。这正是「字段级 patch 而非整树重建」得以成立的前提。

## 2. `prism_ui_tree` — 保留树与协调器

分代 `Arena`、`Tree<K, T>`,以及 **LIS 最小化** 的 keyed 协调器。

```rust
use prism_ui_tree::{diff_keyed};

// 旧顺序 [a, b, c, d] -> 新顺序 [a, c, b, d]
let old = ["a", "b", "c", "d"];
let new = ["a", "c", "b", "d"];
let diff = diff_keyed(&old, &new);

// 没有新建、没有删除,只有最小次数的移动。
assert_eq!(diff.create_count(), 0);
assert!(diff.removals.is_empty());
assert!(diff.move_count() >= 1);
```

`Diff` 的 `ops` 由 `DiffOp::{Keep, Move, Create}` 组成,`Move` 次数经由最长递增
子序列最小化,等价于 React 的 keyed reconciliation,但为零分配紧凑实现。
`reconcile_children` 在此之上驱动保留树的子节点复用。

## 3. `prism_ui_style` — 样式层

design token、`Class` / `StyleSheet`、选择器与级联(`Cascade` → `ComputedStyle`),
支持 token 引用(含环检测)、交互态与响应式断点匹配上下文。

```rust
use prism_ui_style::{Theme, StyleProp, StyleValue};

// 带默认调色板的主题,token 可被 class 引用。
let theme = Theme::with_default_palette();
assert!(theme.tokens.resolve_value(&StyleValue::token("color.bg")).is_ok());

// 内联属性是强类型的键值对。
let w = StyleValue::px(300.0);
let bg = StyleValue::token("color.surface");
let _ = (StyleProp::Width, w);
let _ = (StyleProp::BackgroundColor, bg);
```

`StyleProp` 为 `#[non_exhaustive]`,因此跨 crate 的匹配需带通配臂,这保证样式属性
可以向后兼容地增补。级联 `resolve(sheet, tokens, &classes, &ctx)` 返回
`ComputedStyle`,可 `.iter()` / `.get()`。
> 作用域样式、`@media` 响应式断点、`.loom.style` 资产热重载为 **规划中**。

## 4. `prism_ui_layout` — 布局层

纯 Rust 的 Flexbox 求解器,无外部依赖,`no_std` 友好。核心类型:`Point` / `Size<T>` /
`Rect` / `Edges<T>` / `Dimension` / `AvailableSpace`,样式 `LayoutStyle`
(`Display` / `FlexDirection` / `JustifyContent` / `AlignItems` 等)。

```rust
use prism_ui_layout::{LayoutTree, LayoutStyle, Size, AvailableSpace, Dimension};

let mut tree = LayoutTree::new();
let child = tree.new_leaf(LayoutStyle {
    size: Size { width: Dimension::Points(100.0), height: Dimension::Points(40.0) },
    ..Default::default()
});
let root = tree.new_node(LayoutStyle::default(), &[child]);

tree.compute_layout(root, Size {
    width: AvailableSpace::Definite(800.0),
    height: AvailableSpace::Definite(600.0),
});

let layout = tree.layout(child);
assert_eq!(layout.size.width, 100.0);
```

`new_leaf_with_measure` + `Measure` trait 支持内容测量(如文本);`prism_ui` 的
文本测量实现刻意只用 ×÷,保证确定性且不依赖超越函数。

## 5. `prism_ui_anim` — 效果层

缓动、弹簧物理、关键帧时间线与进出场过渡,`no_std` 友好(无 `std` 时用内部多项式
近似替代超越函数,保证有/无默认特性结果都可用)。

```rust
use prism_ui_anim::{Easing, Spring, Lerp};

// 缓动曲线在 [0,1] 上采样。
let e = Easing::EaseInOut;
let mid = e.sample(0.5);
assert!(mid > 0.0 && mid < 1.0);

// 解析式阻尼弹簧:从当前值朝目标推进一步。
let spring = Spring::gentle();
let mut st = spring.state_at(0.0);
st.step(&spring, 1.0 /* target */, 1.0 / 60.0 /* dt */);
assert!(st.is_settled(1.0, 1e-3) == false); // 一步还没到位

// 标量 / 数组 / 元组的线性插值。
assert_eq!(0.0f32.lerp(&10.0, 0.5), 5.0);
```

还提供 `Timeline` / `Keyframe`(逐段缓动的关键帧)与 `Tween` / `Transition`
(时间驱动补间 + 进出场阶段)。
> 隐式过渡(样式 `transition:` 声明自动补间)、共享元素过渡、编排(stagger /
> sequence)为 **规划中**。

## 6. `prism_ui` — 伞 crate 与运行时

把上述各层整合到一个 `Ui<B: Backend>` 运行时,以廉价、数据层的 `Element` 驱动。

```rust
use prism_ui::{Element, RecordingBackend, Ui};
use prism_ui::layout::{AvailableSpace, Size};

let view = Element::box_()
    .class("card")
    .child(Element::text("hello"))
    .child(Element::box_().key_int(1));

let mut ui = Ui::new(RecordingBackend::new());
ui.mount(&view);
ui.compute_layout(Size::new(
    AvailableSpace::Definite(800.0),
    AvailableSpace::Definite(600.0),
));

// 相同输入 → 零新增操作(成本 ∝ 变化量)。
let before = ui.backend().len();
ui.update(&view);
assert_eq!(ui.backend().len(), before);
```

`Element` 是只读的数据层视图描述:`ElementKind::{Box, Text, Custom}`,`Key::{Index,
Int, Str}`,构建器 `box_()` / `text()` / `custom()` / `.key_int()` / `.key_str()` /
`.class()` / `.style()` / `.child()` / `.children()`。`Ui::update` 做 keyed 递归协调,
`compute_layout` 做布局几何差分,二者都只向 `Backend` 发出最小化 `BackendOp`。

## 7. `prism_ui_macro` — `loom!` DSL

`loom!` 过程宏把一套可读的 DSL 在 **编译期** 降解为上面的 `Element` 构建器调用,
所有生成路径全限定(`::prism_ui::...`),不依赖调用方的 `use`。

```rust
use prism_ui::loom; // 从伞 crate 直接拿到宏

let items = ["a", "b", "c"];
let view = loom! {
    box {
        class: "card", "elevated";
        key: 42;
        style: {
            flex_direction: column;      // 关键字 → Keyword::Column
            width: px(300.0);            // StyleValue::px
            background_color: token("color.bg"); // StyleValue::token
            opacity: 0.5;                // StyleValue::Number
        };
        text("标题");
        for_each(items.iter().map(|s| loom! { text(*s) })); // 动态列表 splice
    }
};
assert_eq!(view.class_names(), &["card".to_string(), "elevated".to_string()]);
```

语法要点:
- 节点:`box { .. }`、`text(EXPR)`、`custom(EXPR)`;
- 属性:`class: a, b;`、`key: 42` 或 `key: "id";`、`style: { prop: value; }`;
- 样式值:`px(..)` / `token(..)` / `rgba8(..)` / 裸数字 / 裸关键字;
- `snake_case` 属性与关键字自动转 `PascalCase`(`background_color` → `BackgroundColor`),
  且保留 span,报错精准;
- `for_each(EXPR)` 把一个 `IntoIterator<Item = Element>` 拼接为子节点。
