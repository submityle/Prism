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

## 6. `prism_ui_overlay` — Portal / Overlay 栈

把「浮层」从普通文档流里解耦出来,统一到一个 **z-order 受控的覆盖平面**,
对标 Radix / Headless UI 的 Portal 模型。

- `OverlayManager` 持有一组 `OverlayEntry`,每条带 `OverlayId` 与 `OverlayKind`
  (`Modal` < `Popover` < `Tooltip` < `Toast`,按 z 优先级升序,同级按插入序)。
- `render(base)` 产出 `base + portal` 的根 box;portal 内按 z-order 逐层生成
  **keyed** 浮层 box,`Modal` 自动在其前插入 `prism-overlay-backdrop`。全部节点
  携带由 overlay id 派生的稳定 key,供协调器精确复用。
- 消解策略:`on_escape()` 关闭顶部**可消解**浮层(跳过 pinned);`on_scrim_click()`
  仅关闭顶部 modal。`FocusTrap<K>` 泛型环形焦点循环,空集合优雅返回 `None`。

```rust
use prism_ui_overlay::{OverlayKind, OverlayManager};

let mut mgr = OverlayManager::new();
let dialog = mgr.push(OverlayKind::Modal, prism_ui::Element::text("confirm?"));
let _tip = mgr.push(OverlayKind::Tooltip, prism_ui::Element::text("hint"));
// tooltip 在 modal 之上;Esc 关掉最顶层可消解项(tooltip)。
mgr.on_escape();
assert!(mgr.get(dialog).is_some()); // modal 仍在
```

---

## 7. `prism_ui_form` — 响应式表单与声明式校验

对标 React Hook Form / VeeValidate:字段值是 **reactive signal**,校验是纯函数的
组合,错误集合是随值自动重算的 `Memo`。

- `Form::register(id, initial, validators)` 注册字段;`value` / `set` / `binding`
  提供读 / 写 / 双向绑定;`touch` 与内部 `dirty` 标志记录交互态。
- 内置 `Validator`:`required` / `min_len` / `max_len` / `int_range` / `pattern` /
  `custom`,可任意组合为 `Vec<BoxedValidator>`。`first_error` 短路、`all_errors` 收集。
- `errors_memo()` 缓存且随 `revision` 信号重订阅新字段;`is_valid()` 在 `untrack`
  下运行以免产生伪订阅。`set` 先释放 map 借用再通知,规避 effect flush 期间的
  `RefCell` 再入 panic。

```rust
use prism_ui_form::{pattern, required, Form};
use prism_ui_reactive::Runtime;

let rt = Runtime::new();
let form = Form::new(rt);
let email = form.register(
    "email",
    "",
    vec![required(), pattern(|v| v.contains('@'), "需为邮箱")],
);
form.set(email.clone(), "a@b.com");
assert!(form.all_errors(email).is_empty());
```

---

## 8. `prism_ui_virtual` — 列表虚拟化

只实例化可视区内的项,对标 react-window / TanStack Virtual,但以 **整数 + 前缀和**
实现(契合仓库禁用 f32 超越函数的约束)。

- `FixedList` 定高:`total_size` / `offset_of` / `visible_range`(含 overscan 与
  双端 clamp)全部 O(1)。
- `VariableList` 变高:构造时把每项尺寸累加成**前缀和偏移表**,`offset_of` O(1),
  `index_at` 用 `partition_point` 做**二分**(O(log n))。
- `RecyclePool`:LIFO 复用插槽、幂等 `acquire`、`release` 归还、`active` 绑定映射,
  为增量滚动提供稳定的节点身份。
- `virtualize_fixed` / `virtualize_variable` 产出 `(可见范围, Element)`:leading
  spacer + 每个可见项的 **keyed slot** + trailing spacer,三段尺寸恒等于 `total_size`,
  使滚动条几何与完整物化列表一致。

```rust
use prism_ui_virtual::{virtualize_fixed, FixedList, Viewport};

let list = FixedList::new(10_000, 24.0, 4.0); // 1 万项,项高 24px,间距 4px
let vp = Viewport::new(480.0, 600.0).with_overscan(3); // 滚到 480px,可视 600px
let (range, _tree) = virtualize_fixed(&list, &vp, |i| prism_ui::Element::text(i.to_string()));
assert!(range.len() < 40); // 一万项,仅物化几十个
```

---

## 9. `prism_ui_async` — 异步状态 / Suspense / 错误边界

把「加载中 / 成功 / 失败」建模成一等公民,对标 SolidJS `createResource` + Suspense。
以**显式驱动的状态机**实现,不绑定任何 futures 运行时,因而 `no_std` 纯净。

- `AsyncState<T, E>`:`is_pending` / `is_ready` / `is_failed`、`ready()` / `failed()`、
  `map` / `map_err`。
- `Resource<T, E>` 基于 `Signal`:`resolve` / `fail` / `reload`,状态转移经响应式图
  自动通知观察者。
- `suspense` / `suspense_all`(聚合多个资源)、`error_boundary` / `guarded`
  在失败时回退到降级视图。`all` + `pending_count` / `failed_count` / `ready_count`
  做批量状态聚合。

```rust
use prism_ui_async::{suspense, Resource};
use prism_ui_reactive::Runtime;

let rt = Runtime::new();
let user: Resource<&str, &str> = Resource::pending(&rt);
let view = suspense(&user, || prism_ui::Element::text("loading…"), |u| prism_ui::Element::text(*u));
assert_eq!(view.text_content(), Some("loading…"));
```

---

## 10. `prism_ui_a11y` — 无障碍基线

把可访问性做进模型层(而非事后补丁),对标 ARIA Authoring Practices。

- `Role` / `AriaState`(`disabled` / `expanded` / `selected` / `hidden` …)/ `Label`
  (文本 / `labelledby` / `describedby`)构成 `A11yNode`;`A11yNodeBuilder` 流式构建。
- `A11yTree` 有序 keyed 树,`label_text` 解析带**防环**;`FocusOrder` 实现 tab-index
  规则、跳过 inert、环绕、first/last。
- `KeyboardNav` 做 **role 感知** 的方向键 / Tab 导航;`LiveRegion` + `Politeness`
  管理读屏播报队列。`screen_reader_text` / `describe_node` 产出读屏文本;`derive`
  模块从 `prism_ui::Element` 桥接。

```rust
use prism_ui::Key;
use prism_ui_a11y::{A11yNode, Label, Role};

let btn = A11yNode::builder(Key::Str("save".into()), Role::Button)
    .label(Label::text("Save"))
    .focusable(true)
    .build();
assert!(btn.is_tab_stop());
```

---

---

## 11. `prism_ui_motion` — 自动过渡 / 布局动画 / 共享元素

对标 **Framer Motion** 的隐式过渡、**FLIP** 布局动画,以及 Flutter `Hero` /
SwiftUI `matchedGeometryEffect` 的共享元素过渡。三者都构建在 `prism_ui_anim`
的确定性 `Tween` / `Easing` 之上,纯算术、`no_std` 友好、无 `unsafe`。

**隐式样式过渡** —— `TransitionTracker` 记住每个属性上一帧的值,值变化时自动补间;
中途打断会从**当前动画值**重定目标(连续性):

```rust
use prism_ui_motion::{TransitionSpec, TransitionTracker};
use prism_ui_style::{StyleProp, StyleValue};

let mut tracker = TransitionTracker::new()
    .with_transition(StyleProp::Width, TransitionSpec::linear(1.0));

tracker.observe(StyleProp::Width, StyleValue::px(0.0));   // 首帧:不动画
tracker.observe(StyleProp::Width, StyleValue::px(100.0)); // 变化:开始补间
tracker.step(0.5);                                        // 推进 0.5s
assert_eq!(tracker.value(StyleProp::Width), Some(StyleValue::px(50.0)));
```

可连续插值的值(`px`/`percent`/`number`/`color`)按数值混合,其余(`keyword`/`auto`/
`token`/单位不匹配)按 eased 进度越过阈值离散切换。

**FLIP 布局动画** —— 记录旧矩形,布局变化后计算把新框「反转」回旧框的
`Transform`,再播放回 identity:

```rust
use prism_ui_motion::{FlipAnimation, Rect};
use prism_ui_anim::Easing;

let prev = Rect::new(0.0, 0.0, 100.0, 100.0);
let current = Rect::new(200.0, 0.0, 100.0, 100.0);
let flip = FlipAnimation::new(prev, current, 1.0, Easing::Linear);
assert_eq!(flip.sample(0.0).tx, -200.0); // 起始视觉上仍在旧位置
assert!(flip.sample(1.0).is_identity()); // 结束落到新位置
```

**共享元素(Hero)过渡** —— 按 `Key` 配对两套布局中的同一元素,得到源→目标补间;
只在一侧出现的 key 记为 entering / leaving 并给出进出场标量;`stagger` 复用
`Choreography` 做级联编排:

```rust
use prism_ui_motion::SharedElementTransition;
use prism_ui::Key;
use prism_ui_anim::Easing;

let from = [(Key::Str("card".into()), prism_ui_motion::Rect::new(0.0, 0.0, 100.0, 80.0))];
let to   = [(Key::Str("card".into()), prism_ui_motion::Rect::new(300.0, 200.0, 200.0, 160.0))];
let shared = SharedElementTransition::from_frames(&from, &to, 0.4, Easing::Linear);

assert_eq!(shared.matched_len(), 1);
let mid = shared.sample(&Key::Str("card".into()), 0.5).unwrap();
assert!(!mid.is_identity()); // 中途处于飞行状态
assert!(shared.sample(&Key::Str("card".into()), 1.0).unwrap().is_identity());
```

> 对标:Framer Motion `layout` / `AnimatePresence`、Flutter `Hero`、SwiftUI
> `matchedGeometryEffect`。差异:我们把「取值→补间→反转变换」拆成可单测的纯函数,
> 不绑定任何渲染后端,`Transform` 可直接下发给 Loom 的 paint 层。

---

## 12. `prism_ui_scoped` — 组件作用域样式 / 响应式 @media

对标 **Vue `<style scoped>`** / **CSS Modules** 的 class 命名空间化,以及
**Tailwind** 的 mobile-first 响应式断点。构建在 `prism_ui_style` 之上,只决定
「哪个值生效」,不重造级联。

**作用域样式** —— `ScopeId` 用稳定散列(FNV-1a,等价 Vue `data-v-xxxxxxxx`)为组件
生成唯一后缀;`Scope` 只重写**本作用域登记**的 class,全局 / 未知 class 原样透传。
同一份样式表给两个组件作用域化后不会相互冲突:

```rust
use prism_ui::Element;
use prism_ui_scoped::Scope;
use prism_ui_style::{Class, StyleProp, StyleSheet, StyleValue};

let sheet = StyleSheet::new()
    .with_class(Class::new("title").with(StyleProp::FontSize, StyleValue::px(16.0)));

let scope = Scope::from_name("Hero").with_local("title");
let scoped = scope.scope(&sheet);

let scoped_name = scoped.scoped_name("title").unwrap().to_string();
assert_ne!(scoped_name, "title");                     // 已命名空间化
assert_eq!(scoped.local_name(&scoped_name), Some("title")); // 可反查

// 同样的重写应用到视图树,引用保持一致。
let view = Element::box_().child(Element::text("Hi").class("title"));
let out = scope.apply(&view);
assert_eq!(out.child_elements()[0].class_names(), &[scoped_name]);
```

**响应式 @media 解析** —— `MediaResolver` 以视口宽度为输入,按 mobile-first 级联
(base 恒生效,较宽断点逐层覆盖)解出当前生效的属性集,语义与 Tailwind 一致:

```rust
use prism_ui_scoped::MediaResolver;
use prism_ui_style::{Breakpoint, Class, StyleProp, StyleValue};

let class = Class::new("title")
    .with(StyleProp::FontSize, StyleValue::px(14.0))
    .with_breakpoint(Breakpoint::Md, StyleProp::FontSize, StyleValue::px(24.0));

let narrow = MediaResolver::new(640.0).active_props(&class);
let wide = MediaResolver::new(768.0).active_props(&class);
assert_eq!(narrow.get(&StyleProp::FontSize), Some(&StyleValue::px(14.0)));
assert_eq!(wide.get(&StyleProp::FontSize), Some(&StyleValue::px(24.0)));
```

> 对标:Vue scoped / CSS Modules(作用域),Tailwind `sm:`/`md:`/`lg:`(断点)。
> 差异:作用域仅重写**登记过**的 local class(确定、无全局副作用),断点解析是
> cascade 的互补视图(回答「此宽度下谁生效」),二者皆为可单测的纯数据变换。

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
- Suspense / Portal / Overlay / 虚拟化 / 表单校验 / a11y 基线 **均已交付**(本文 6–10 节)。
  仍为 **规划中** 的是:守卫 / 深链接(router)、时间旅行 UI(devtools)、
  静态子树提升 / 编译期稳定节点 ID,见 [roadmap.md](roadmap.md)。隐式过渡 / FLIP 布局动画 /
  共享元素过渡(第 11 节)与作用域样式 / 响应式 @media(第 12 节)**均已交付**。
