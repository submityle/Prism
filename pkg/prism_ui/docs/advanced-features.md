# Loom 高级功能层

> 本文描述 Loom **已交付** 的高级功能层。它们对标 React / SolidJS / Vue / Pinia
> 生态的一等能力,全部构建在核心层([`prism_ui_reactive`](../../prism_ui_reactive)、
> [`prism_ui`](../../prism_ui))之上,`#![forbid(unsafe_code)]`、`no_std` 友好、
> Clippy 零告警。文中所有代码示例均来自各 crate 的真实 doctest / 集成测试。

| Crate | 对标 | 一句话 |
|---|---|---|
| `prism_ui_component` | React / SolidJS 组件 | `Component` trait + props + 具名多插槽 + Context 注入 |
| `prism_ui_store` | Redux / Zustand / Pinia | 可预测状态容器 + 细粒度选择器 + 中间件 |
| `prism_ui_i18n` | FormatJS / Fluent | 响应式消息目录 + 插值 + CLDR 复数 |
| `prism_ui_router` | React Router / Vue Router | 路径匹配 + `:param` / `*wildcard` + 历史栈 |
| `prism_ui_devtools` | React DevTools | 树快照 + 美化渲染 + `BackendOp` 轨迹 |

设计贯穿一条主线:**高级能力不新造一套状态模型,而是复用核心层的细粒度响应式**。
Store 的订阅、i18n 的 locale 切换、router 的当前位置,全部是 `prism_ui_reactive`
的 `Signal` / `Memo`,因此「变了什么就只重算什么」的成本契约对高级层同样成立。

---

## 1. `prism_ui_component` — 组件模型

把一段视图封装为可复用、可组合、可注入上下文的单元。

- `trait Component { type Props; fn render(&self, props: &Self::Props) -> Element; }`
  —— 类型安全的组件契约,props 类型随组件而定。
- `FnComponent::new(|p: &P| -> Element)` —— 把闭包直接升格为组件,免写 struct。
- `Props` —— 构建器式属性包:`class(..)` / `child(..)` / `children(..)`,
  读取用 `class_names()` / `child_elements()`。
- `mount_component(&component, props) -> Element` —— 实例化为可挂载的 `Element`。
- `ContextMap` —— 按类型注入/读取的上下文:`provide::<T>` / `inject::<T>()`
  / `child()`(克隆继承、就地覆盖),实现跨层级依赖传递而不逐层透传 props。
- `Slots` / `SlottedComponent` —— 具名多插槽:`with(name, el)` / `with_default(el)`,
  渲染时用 `named(name)` / `default_slot()` 取回,支持「卡片的 header/body/footer」
  这类布局型组件。
- `ComponentCtx` —— 把 `slots` 与 `context` 一起交给渲染闭包:
  `render_with_context(ctx, |cx| { .. })`。

```rust
use prism_ui::Element;
use prism_ui_component::{FnComponent, Props, mount_component, Component};

// 一个把 props 里的 class 与子节点转发到 box 的组件。
let card = FnComponent::new(|p: &Props| {
    let mut el = Element::box_();
    for c in p.class_names() { el = el.class(c); }
    for child in p.child_elements() { el = el.child(child.clone()); }
    el
});

let props = Props::new()
    .class("card")
    .child(Element::text("标题"));
let view = mount_component(&card, props);
```

**为什么比 BSN 强**:BSN 没有独立的组件 / 插槽 / Context 概念,复用靠宏片段拼接;
Loom 的组件是一等值,可带类型化 props、具名插槽与按类型注入的上下文,组合性与
可测试性都更接近成熟前端框架。

---

## 2. `prism_ui_store` — 可预测全局状态

Loom 版的 Redux / Zustand / Pinia:单一状态容器,写入可预测,读取被细粒度追踪。

- `Store::new(&rt, initial)` —— 持有一个由 `Signal` 承载的状态 `S`。
- `store.get()` / `store.with(|s| ..)` —— 读取(在 memo/effect 内读会自动建立依赖)。
- `store.update(|s| ..)` / `store.set(s)` —— 提交写入:**无条件**跑中间件 + 通知。
- `store.set_if_changed(s)`(`S: PartialEq`)—— 相等则丢弃,零中间件、零通知。
- `store.select(|s| ..) -> Memo<U>` —— **细粒度选择器**:状态变了才重算,
  且只有「选出的切片」真的变化时才打扰下游。
- `Middleware<S>` / `LoggingMiddleware` —— 环绕每次提交的钩子,内置日志中间件
  记录 `(prev, next)` 快照,便于审计与时间旅行。

```rust
use prism_ui_reactive::Runtime;
use prism_ui_store::Store;

let rt = Runtime::new();
let store = Store::new(&rt, 0i32);

let doubled = store.select(|&n| n * 2);   // 细粒度派生
store.update(|n| *n += 5);

assert_eq!(store.get(), 5);
assert_eq!(doubled.get(), 10);            // 自动跟随
```

**提交策略取舍**:默认 `update`/`set` 无条件提交(无 `PartialEq` 约束时无法判等,
保证通用场景可预测);当 `S: PartialEq` 时用 `set_if_changed` 做零成本变更门控。
这条「可预测优先、按需门控」的策略与引擎的性能预算一致。

---

## 3. `prism_ui_i18n` — 国际化

响应式消息目录,带参数插值与 CLDR 风格复数选择;当前 locale 存于 `Signal`,
切换语言时所有翻译 `Memo` 自动重算。

- `LocaleId` —— 轻量 locale 标识。
- `Catalog::new().with(key, "模板 {name}")` —— 单条消息或复数消息集合。
- `PluralCategory` / `PluralRules` —— **纯整数** 复数选择(无浮点,确定性)。
- `Args::new().with(k, v)` / `Value` —— 插值参数表。
- `I18n::new(&rt, locale)` —— 响应式门面:`register(..)`、`set_locale(..)`、
  `translation(key, args) -> Memo<String>`。

```rust
use prism_ui_i18n::{Args, Catalog, I18n, PluralRules};
use prism_ui_reactive::Runtime;

let rt = Runtime::new();
let mut i18n = I18n::new(&rt, "en");
i18n.register("en", Catalog::new().with("greet", "Hello, {name}!"), false, PluralRules::English);
i18n.register("fr", Catalog::new().with("greet", "Bonjour, {name}!"), false, PluralRules::English);

let hello = i18n.translation("greet", Args::new().with("name", "Ada"));
assert_eq!(hello.get(), "Hello, Ada!");

i18n.set_locale("fr");
assert_eq!(hello.get(), "Bonjour, Ada!"); // 切换 locale,翻译自动更新
```

**与引擎契合点**:复数选择刻意不碰浮点,与 `prism_ui_layout` / `prism_ui_anim`
的确定性约束一脉相承,保证跨平台 / 回放一致。

---

## 4. `prism_ui_router` — 响应式客户端路由

当前位置存于 `Signal`,路由解析、渲染、副作用都随导航自动更新;`no_std` 友好,
路由匹配是纯整数 / 字符串运算(无浮点)。

- `Location` —— 解析后的路径:segments + query + fragment。
- `RoutePattern` / `RouteMatch` —— 编译后的模式(静态段、`:name` 参数、
  末尾 `*name` 通配)与其捕获。
- `RouteTable` / `RouteId` —— **有序、首次匹配胜出** 的路由表。
- `Router::new(&rt, path, table)` —— 绑定路由表到响应式运行时,暴露:
  `current_match() -> Memo<Option<(RouteId, RouteMatch)>>`、`navigate(path)`、
  `location()`、`back()` / `forward()` 历史栈。

```rust
use prism_ui_reactive::Runtime;
use prism_ui_router::{RouteId, RouteTable, Router};

let rt = Runtime::new();
let table = RouteTable::new()
    .route("/", RouteId::new(0))
    .route("/users/:id", RouteId::new(1));

let router = Router::new(&rt, "/", table);
let current = router.current_match();

router.navigate("/users/42");
let (id, matched) = current.get().expect("user route matches");
assert_eq!(id, RouteId::new(1));
assert_eq!(matched.param("id"), Some("42"));

assert!(router.back());
assert_eq!(router.location().path(), "/");
```

> 守卫(guards)与深链接协商为 roadmap M5 的后续增强。

---

## 5. `prism_ui_devtools` — 内省与调试

把运行时转瞬即逝的状态变成 **可存储、确定性、人类可读** 的产物,直接服务于
快照测试(golden tests)。

- `snapshot(&Element) -> TreeSnapshot` —— 把活的 `Element` 树走成 owned 快照,
  可 `node_count()` / `depth()` 查询、可比较。
- `render_tree(&TreeSnapshot) -> String` —— 缩进式多行美化输出,用于调试与
  golden 比对。
- `OpTrace` —— 按变体统计一段 `BackendOp`,`summary()` 给出稳定单行摘要,
  把「本帧产生了哪些操作」变成可断言的事实。

```rust
use prism_ui::Element;
use prism_ui_devtools::{render_tree, snapshot};

let view = Element::box_()
    .child(Element::text("hello"))
    .child(Element::text("world"));

let snap = snapshot(&view);
assert_eq!(snap.node_count(), 3);
assert_eq!(snap.depth(), 2);
assert_eq!(render_tree(&snap), "Box\n  Text \"hello\"\n  Text \"world\"\n");
```

**与性能契约闭环**:`OpTrace` + `RecordingBackend` 把 README 里「成本 ∝ 变化量」
的口号钉成可回归的测试——相同输入零新增 op、keyed 反转仅 1 次 reorder 等,
都能被 DevTools 产物断言。完整的实体/组件树检查器、signal 依赖图可视化、
状态时间旅行回放为 roadmap M6 的后续增强。

---

## 组合示例:高级层如何协同

一个典型的「可国际化、带全局状态、按路由切换」的视图,其数据流为:

```
Router.current_match (Memo)  ─┐
Store.select(切片)   (Memo)  ─┼─►  组件 render(&Props)  ─►  Element 树
I18n.translation     (Memo)  ─┘                              │
                                                             ▼
                               Ui::update  →  keyed 协调  →  最小化 BackendOp
                                                             │
                               DevTools.OpTrace  ◄───────────┘(可断言)
```

因为 Store / Router / I18n 全部以 `Signal` / `Memo` 暴露,它们天然接入核心层的
无毛刺依赖图:任意一个发生变化,只会触发依赖它的那部分组件重算,再经协调器
落为最小化操作流。高级层没有引入第二套更新机制,这是「性能可预测」与「易维护」
在架构层面的根本保证。

## 诚实边界

- 本文所列能力均 **已交付并通过测试**,但当前为 **引擎弱耦合的独立运行时能力**:
  它们消费 / 产出 `Element` 与 `Signal`,**尚未** 与 Prism/ECS 实体字段做深度自动绑定
  (该绑定为 roadmap M2)。
- 守卫 / 深链接(router)、时间旅行 UI(devtools)、Suspense / Portal / 虚拟化 /
  共享元素过渡等仍为 **规划中**,见 [roadmap.md](roadmap.md)。
