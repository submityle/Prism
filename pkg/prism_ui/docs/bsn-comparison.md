# Loom 与 Bevy BSN:评析、借鉴与重新设计

> 本文分三部分:(1) 剖析 **Bevy BSN** 的机制、长处与结构性局限;
> (2) 说明 Loom 从 **React / SolidJS / Vue / Flutter / SwiftUI / Jetpack Compose /
> Tailwind / Framer Motion / Storybook / Redux DevTools** 借鉴了什么;
> (3) 给出 Loom 的 **重新设计**,并沿 **性能 / 效果 / 易用性 / 易维护** 四维逐项论证。
>
> 原则:严格区分 **已交付(SHIPPED)**、**进行中(IN PROGRESS)** 与 **规划中(PLANNED)**。
> 对 BSN 的评析立足其公开可见的设计方向,服务于 Loom 的取舍,不代表对 BSN 的最终评价。
> 当前 Loom 规模:**25 个 crate / 486 个测试通过**,Clippy 零告警,核心层 `no_std` 友好。

---

## 1. 什么是 BSN

BSN(Bevy Scene Notation)是 Bevy 新一代的 **声明式场景 / UI 记法**。它用一套宏
(`bsn!`)与配套 `.bsn` 资产,把「**结构**(spawn 哪些实体)+ **构造**(挂哪些组件)
+ **数据**(组件初值)」统一表达,目标是取代样板繁多的命令式 `spawn` 链与难手写的
`.scn.ron`,并成为 Bevy UI 的声明式基座。其核心概念大致是:

- **实体-组件直映射**:一个节点就是一个 Entity,花括号里列出要挂的 Component。
- **Construct / Patch**:`Construct` trait 从「配方」构造组件,`patch` 允许在模板基础上
  覆盖字段;配合 **required components** 做隐式补齐。
- **模板与继承**:`.bsn` 资产可被复用 / 组合 / 覆写,支持资产级 **热重载**。
- **与 ECS 变更检测集成**:数据驱动的刷新复用 Bevy 既有的 tick 变更检测(规划 / 演进中)。

**长处**:

- **ECS 原生**:直接映射 Entity + Component,与 Bevy 世界模型天然一致,零阻抗失配。
- **类型安全的宏**:结构在编译期检查,组件 / 字段写错即编译报错。
- **一处声明**:实体层级 + 初始组件集中书写,比命令式 `spawn` 链紧凑得多。
- **资产化**:`.bsn` 可独立于代码分发、热重载,利于美术 / 关卡迭代。

**结构性局限**(Loom 要改进的点):

1. **三件事耦合在一套语法**:结构、构造、数据初值绑在一起,想单独迭代样式系统或
   响应模型都要动同一套宏,演进被锁死。
2. **无独立样式层**:样式即「组件字段的初值」,缺少 token / class / 级联 / 断点这类
   设计系统一等概念,主题化与响应式断点要在实体层手工搭。
3. **更新粒度偏实体 / 场景级**:以 patch 覆盖为主,缺少「只发出本次真实字段差异」的
   最小增量协调器,大列表重排 / 局部文本变更的成本不易被钉死。
4. **archetype 搬迁风险**:用「插入 / 移除 marker 组件」表达显隐 / 选中态时,运行期改变
   组件集合会触发 archetype 搬迁(内存搬运),在热路径上代价高且隐蔽。
5. **动画 / 异步 / 路由 / i18n 非一等公民**:需各自外接系统,缺少统一的声明式表达与
   可预测的更新契约。
6. **引擎强绑定**:离开 Bevy App / World 难以独立单测视图逻辑。

---

## 2. 借鉴了哪些优秀产品

Loom 不发明新范式,而是把各生态**被验证过**的最佳实践,收敛到一套对引擎友好的实现里:

| 来源 | 借鉴的核心思想 | 在 Loom 的落点 |
|---|---|---|
| **SolidJS / Leptos** | 细粒度响应式:signal 变化只重算其依赖,无 VDOM 整树 diff | `prism_ui_reactive` 无毛刺 Signal / Memo / Effect(拓扑 + 批处理) |
| **React** | 组件模型、Context、Suspense、Error Boundary、DevTools | `prism_ui_component` / `prism_ui_async` / `prism_ui_devtools` |
| **Flutter** | Widget(数据) / Element(保留)分离;每帧廉价重建 + 协调 | `Element` 数据层 + 保留树 + keyed 协调器 |
| **SwiftUI** | 声明式视图 + 隐式过渡 + 环境注入 | `loom!` + `prism_ui_motion`(隐式过渡)+ `ContextMap` |
| **Jetpack Compose** | 位置记忆(positional memoization)、稳定节点身份、重组跳过 | roadmap M1 **编译期稳定节点 ID**(进行中)+ keyed 协调 |
| **Tailwind** | 原子化 class + 设计 token + 响应式断点 | `prism_ui_style`(token / class / 级联)+ `prism_ui_scoped`(@media) |
| **Framer Motion** | 动画一等公民:spring / layout(FLIP) / shared-element / 编排 | `prism_ui_anim` + `prism_ui_motion` |
| **Redux / Zustand / Pinia** | 可预测状态容器 + 细粒度选择器 + 中间件 | `prism_ui_store` |
| **React Router / Vue Router** | 路径匹配 + 参数 / 通配 + 历史栈 | `prism_ui_router` |
| **FormatJS / Fluent** | 响应式消息目录 + 插值 + CLDR 复数 | `prism_ui_i18n` |
| **Storybook** | 组件隔离预览 / 用例工作台 | `prism_ui_workbench` |
| **Redux DevTools** | 时间旅行 / 快照 / 依赖图检查 | `prism_ui_timetravel` / `prism_ui_snapshot` / `prism_ui_inspector` |

---

## 3. Loom 的重新设计:三层分离 + ECS 字段级绑定

Loom 把 BSN 耦合在一套宏里的三件事拆成 **可独立编写、测试、演进** 的三层,再用一座
**字段级双向绑定桥** 把响应层接回 ECS:

```
结构层 Structure  →  loom! 宏 / Element 构建器(编译期降解为全限定构建器调用)
响应层 Reactivity →  Signal / Memo / Effect(运行期细粒度,无毛刺)
样式层 Style      →  token / class / 级联 / 断点 / @media(资产,可热重载)
         ↓ 全部落到 ↓
保留树 + Keyed 协调器(LIS 最小化)→ Flexbox 布局 → 最小化 BackendOp 流 → 渲染后端
         ↑ 字段级双向绑定 ↑
ECS 组件字段  ⇄  Signal   (prism_ui_ecs:复用 ECS tick 变更检测作传输)
```

与 BSN 最关键的差异点是 **`prism_ui_ecs`(已交付,roadmap M2 headline)**:它不是另造一套
脏标记,而是**复用 ECS 的 tick 变更检测作为响应式的传输**。`FieldBinding<C,T>` 把组件 `C`
的单个字段经 `reader`/`writer` 闭包投影到 / 回写自 `Signal<T>`;拉取查变更 tick(空闲实体
零成本),回写走相等性守卫(绝不无谓推进 tick),双向都带守卫故 `ECS→signal→ECS` 往返
**收敛而非振荡**。这把「声明式视图」与「引擎实体」在字段粒度接通,又不牺牲最小增量契约。

> 进行中:`EcsBridge` 的 **Bevy 调度器集成**(NonSend 资源 + exclusive system,帧首 pull /
> 帧末 push)与 roadmap M1 **编译期稳定节点 ID** 正在落地;落地并提交后本文与 roadmap 同步勾选。

---

## 4. 逐维度评析与 Loom 的改进

| 维度 | BSN | Loom 的做法 | 状态 |
|---|---|---|---|
| 关注点分离 | 结构 + 构造 + 数据耦合在同一套语法 | **结构 / 响应 / 样式三层分离**,各自独立单测与演进 | 已交付 |
| 更新粒度 | 偏实体 / 场景级 patch | **字段级增量**:协调器只发出 `SetText`/`SetPaint`/`Reorder` 等真实差异 | 已交付 |
| archetype 搬迁 | 用增删 marker 组件表显隐 / 选中态时可能触发 | **显式规避**:只改组件「值」,可切换态用枚举 / 标志字段 | 已交付(策略)|
| ECS ↔ 响应式 | 规划 / 演进中 | **字段级双向绑定** `FieldBinding`/`EcsBridge`,复用 tick 变更检测 | 已交付;调度器集成进行中 |
| 独立样式层 | 无,样式即组件字段初值 | **token + class + 级联 + 交互态 + 断点**;**scoped + @media** | 已交付 |
| 热重载 | 资产级(逐步成熟) | **结构 + 样式热重载且保留运行时状态**:`NodePath` 身份 + `ReloadPlan` + `StateStore` + `StyleDiff` | 已交付(核心);文件监听 + 稳定 ID 对齐进行中 |
| 动画 / 效果 | 非一等公民,需外接 | **一等公民**:缓动 / 弹簧 / 时间线 / 过渡 + 隐式过渡 + FLIP 布局动画 + 共享元素 + 编排 | 已交付 |
| 组件 / 状态 / 路由 / i18n | 无独立层,靠宏片段拼接 | **一等独立 crate**:`component` / `store` / `router` / `i18n` | 已交付 |
| 异步 / 错误边界 | 较弱 | **`Resource` + Suspense + Error Boundary**(`prism_ui_async`) | 已交付 |
| Portal / Overlay | 需手搭 | **模态 / popover / tooltip / toast + z-order + FocusTrap**(`prism_ui_overlay`) | 已交付 |
| 列表虚拟化 | 需手搭 | **定高 / 变高 + 前缀和二分 + overscan + 回收池**(`prism_ui_virtual`) | 已交付 |
| 表单 | 需手搭 | **双向绑定 + 声明式校验**(`prism_ui_form`) | 已交付 |
| 无障碍 a11y | 较弱 | **角色 / 焦点 / 键盘导航 / 读屏标签基线**(`prism_ui_a11y`) | 已交付 |
| 工具链 | 发展中 | **DevTools + 树检查器 + 性能面板 + 时间旅行 + 依赖图 + 快照 + 工作台** | 已交付 |
| 稳定节点身份 | required components / 资产对齐 | **编译期稳定节点 ID**(借 Compose 位置记忆),用于结构变更 / 热重载精确对齐 | 进行中(M1) |
| 宏层绑定糖 | `bsn!` 内建 | `loom!` 的 **`$` 绑定语法糖**(自动登记 `EcsBridge` 绑定) | 规划中(M2) |
| 双模式编译 | 宏 + 资产 | **开发期解释(极速热重载)/ 发布期宏固化(零解析)** | 规划中(M6) |

---

## 5. 四维论证

### 5.1 性能:成本 ∝ 变化量,而非场景规模

- **细粒度响应式**:信号变化只触发其依赖子图重算(SolidJS 路线),无 VDOM 整树 diff。
- **最小增量协调**:keyed 协调器用 **LIS(最长递增子序列)** 求最少移动;`RecordingBackend`
  把「本次产生了哪些操作」变成**可断言的事实**——「相同输入零新增」「keyed 反转
  0 新建 / 0 删除 / 1 reorder」都是被钉死的回归用例。BSN 的性能取决于使用者是否踩中
  archetype 搬迁等陷阱,较难直接度量;Loom 把它变成测试契约。
- **零无谓搬迁 / 零无谓 tick**:显隐 / 选中态改字段而非增删组件;`prism_ui_ecs` 回写带
  相等性守卫,值未变就不置脏,从源头掐断「写入 → tick → 又触发拉取」的无谓循环。
- **编译期常量化(规划)**:无绑定的静态子树提升 + 双模式编译的「发布期宏固化」把
  解析 / 构建开销降到零。

### 5.2 效果:动画与过渡是一等公民

- `prism_ui_anim`:缓动曲线(含 Newton / 二分求解的 CubicBezier)、解析式阻尼弹簧、
  关键帧时间线、进出场 `Transition`、编排(stagger / sequence / parallel)。
- `prism_ui_motion`:**隐式过渡**(属性变化即自动补间,打断可重定目标)、
  **FLIP 布局动画**(First/Last/Invert/Play)、**共享元素 / Hero 过渡**。
- 这些在 BSN 里通常要外接动画系统并手动驱动;在 Loom 里是声明式、可组合、可测试的内建能力。

### 5.3 易用性:像写前端一样写引擎 UI

- `loom!` DSL 可读、报错精准(保留 ident span,错误即编译期定位)。
- 组件 / 插槽 / Context / Store / 路由 / i18n / 表单 / Overlay 一应俱全,心智模型与 React/Solid 对齐。
- `$` 绑定语法糖(规划)让「组件字段 ⇄ 信号」一行声明;Storybook 式工作台让组件可隔离预览。

### 5.4 易维护:三层解耦 + 可预测 + 可内省

- **三层各自单测、各自演进**:想迭代样式系统不必动响应模型或结构宏。
- **单一更新机制**:Store / Router / I18n 全部以 `Signal` / `Memo` 暴露,天然接入同一张
  无毛刺依赖图——高级层没有引入第二套更新机制,这是「性能可预测」与「易维护」的根本保证。
- **全链路可内省**:树快照、`OpTrace` 操作轨迹、`PerfReport` 性能面板、
  signal **依赖图**(`DependencyGraph`:传递闭包 / Kahn 拓扑序 / Graphviz 导出)、
  **时间旅行**回放与相邻帧 diff、**快照回归测试**。调试体验对标 React/Redux DevTools。

---

## 6. 诚实的边界

- 本文「已交付」项均 **已实现并通过测试**(25 crate / 486 测试、Clippy 零告警);
  详见 [advanced-features.md](advanced-features.md) 与各 crate 的 doctest / 集成测试。
- **进行中**:`EcsBridge` 的 Bevy 调度器集成、roadmap M1 编译期稳定节点 ID / 静态子树提升。
  这些在落地并本地提交前,**不计入「已胜出」**;提交后同步更新本文与 [roadmap.md](roadmap.md)。
- **规划中**:`loom!` 的 `$` 绑定语法糖、`Show` / `For` 结构绑定(批量 spawn/despawn 到帧末)、
  双模式编译、router 守卫 / 深链接、组件生命周期钩子、`t!` 宏糖 / RTL。
- BSN 作为 Bevy 官方方案,在 **与 Bevy 生态的原生集成度** 上天然领先;Loom 的取舍是
  **引擎弱耦合 + 字段级可绑定 + 可测试的最小增量更新**,两者定位不同、可长期并存。
