# Loom 路线图

> 本文清晰区分 **已交付(SHIPPED)** 与 **规划中(PLANNED)**。
> 原则:不把未实现能力描述为已实现。

## 已交付(SHIPPED)

7 个 crate,全部通过测试、Clippy 零告警,`no_std` 友好(proc-macro crate 除外):

| Crate | 能力 | 测试 |
|---|---|---|
| `prism_ui_reactive` | 无毛刺 Signal / Memo / Effect | 9 |
| `prism_ui_tree` | 分代 Arena、保留树、LIS 最小化 keyed 协调 | 9 |
| `prism_ui_style` | token / class / 选择器 / 级联(含 token 环检测) | 13 |
| `prism_ui_layout` | 纯 Rust Flexbox 求解器 | 15 |
| `prism_ui_anim` | 缓动 / 弹簧 / 时间线 / 过渡 | 24 |
| `prism_ui` | `Element` / `Ui` 运行时 / `Backend` / 最小化 op 流 | 7 + 2 doctest |
| `prism_ui_macro` | `loom!` 声明式 DSL | 8 |

**高级层** 5 个 crate 亦已交付(详见 [advanced-features.md](advanced-features.md)):

| Crate | 能力 | 测试 |
|---|---|---|
| `prism_ui_component` | 组件模型:props / 具名多插槽 / Context 注入 | 15 |
| `prism_ui_store` | 可预测状态容器 + 细粒度选择器 + 中间件 | 12 |
| `prism_ui_i18n` | 响应式目录 + 插值 + CLDR 复数 | 10 |
| `prism_ui_router` | 路径匹配 + `:param` / `*wildcard` + 历史栈 | 16 |
| `prism_ui_devtools` | 树快照 + 美化渲染 + `BackendOp` 轨迹 | 10 |

12 个 crate 累计 **176 个测试通过**,Clippy 零告警。

对应的核心价值已可验证:
- 数据层 `Element` → 保留树 → 最小化 `BackendOp` 的完整链路;
- 「相同输入零新增操作」「keyed 反转 0 新建 / 0 删除 / 1 reorder」等性能契约测试;
- 响应 / 样式 / 布局 / 动画四层均可独立使用与单测;
- `loom!` 宏编译期降解为全限定构建器调用,报错精准。

## 规划中(PLANNED)

### M1 结构层(部分已交付)
- [x] `loom!` 宏 + `Element` 构建器。
- [ ] 静态子树提升(无绑定子树编译期常量化)。
- [ ] 编译期稳定节点 ID(借 Compose 位置记忆),用于结构变更 / 热重载精确对齐。

### M2 响应层到 ECS 的绑定(部分已交付)
- [x] 独立的 Signal / Memo / Effect 运行时。
- [x] 字段级 **双向** 绑定:ECS 组件字段 <-> `Signal` 的 `FieldBinding`/`EcsBridge`——读路径复用 `Ref` 的 tick 变更检测(仅变更帧才拉取),写路径用相等性守卫经 `Mut` 回写(不触发无谓 tick、不振荡),零 archetype 搬迁(`prism_ui_ecs`,已交付)。宏层 `$` 语法糖自动登记绑定仍规划中。
- [ ] `Show` / `For` 结构绑定,keyed reconcile 批量 spawn/despawn 到帧末。
- [x] 可访问性(a11y)基线:角色 / 焦点 / 键盘导航 / 读屏标签(`prism_ui_a11y`,已交付)。

### M3 样式层增强(scoped / @media / 热重载核心已交付)
- [x] token / class / 级联 / 交互态 / 断点匹配上下文。
- [x] 作用域样式(scoped):`ScopeId` 稳定散列 + class 命名空间化(`prism_ui_scoped`,已交付)。
- [x] `@media` 响应式断点解析:mobile-first 级联解出生效属性集(`prism_ui_scoped`,已交付)。
- [x] `.loom` 结构 + `.loom.style` 样式热重载,**保留运行时状态**:`NodePath` 身份比对 + `ReloadPlan`(保留/新增/移除/重建)+ `StateStore` 状态裁剪 + `StyleDiff`(`prism_ui_hotreload`,已交付)。文件系统监听集成仍规划中。

### M4 效果层增强(已交付)
- [x] 缓动 / 弹簧 / 时间线 / 进出场 `Transition`。
- [x] 隐式过渡:`TransitionTracker` 记忆旧值,属性变化自动补间,打断可重定目标(`prism_ui_motion`,已交付)。
- [x] 布局动画:FLIP(First/Last/Invert/Play)反转变换回归 identity(`prism_ui_motion`,已交付)。
- [x] 编排(choreography):stagger / sequence / parallel(`prism_ui_anim::Choreography`,已交付)。
- [x] 共享元素过渡(shared element / Hero):按 `Key` 配对源→目标补间 + 进出场回退(`prism_ui_motion`,已交付)。

### M5 高级功能(多数已交付)
- [x] 组件模型:props、具名多插槽、`children`、Context 注入(`prism_ui_component`,已交付)。生命周期钩子规划中。
- [x] Store + 细粒度选择器 + 中间件(`prism_ui_store`,已交付)。
- [x] 列表虚拟化:定高/变高 + 前缀和二分 + overscan + 回收池(`prism_ui_virtual`,已交付)。
- [x] 异步与健壮性:`Resource` + Suspense + Error Boundary(`prism_ui_async`,已交付)。
- [x] Portal / Overlay 管理器(模态 / popover / tooltip / toast、z-order、backdrop、FocusTrap,`prism_ui_overlay`,已交付)。
- [x] 表单双向绑定 + 声明式校验(`prism_ui_form`,已交付)。
- [x] 声明式路由 + 导航/历史栈(`prism_ui_router`,已交付)。守卫 + 深链接规划中。
- [x] 国际化:响应式目录 + 插值 + CLDR 复数(`prism_ui_i18n`,已交付)。`t!` 宏糖 / RTL 规划中。

### M6 工具链(部分已交付)
- [x] DevTools 基线:树快照 + `render_tree` 美化输出 + `OpTrace` 操作轨迹(`prism_ui_devtools`,已交付)。
- [x] DevTools 进阶 · 树检查器 + 性能面板:`NodePath` 寻址 / `Query` 过滤 / `PerfReport` / `TreeMetrics`(`prism_ui_inspector`,已交付)。
- [x] DevTools 进阶 · 状态时间旅行回放:帧时间线 undo/redo + 跳转 + 相邻帧 diff / 变更摘要(`prism_ui_timetravel`,已交付)。
- [x] DevTools 进阶 · signal 依赖图:`prism_ui_reactive` 暴露只读 `GraphSnapshot` introspection,`prism_ui_inspector::DependencyGraph` 提供传递闭包 / 拓扑序(Kahn)/ Graphviz 导出(已交付)。
- [x] 快照测试(渲染树 / 布局结果序列化比对)(`prism_ui_snapshot`,已交付)。
- [x] 组件工作台(Storybook 式隔离预览)(`prism_ui_workbench`,已交付)。
- [ ] 双模式编译:开发期解释(极速热重载)/ 发布期宏固化(零解析开销)。

## 风险与开放问题

1. **响应式实现策略**:自研细粒度 vs 复用 ECS 变更检测。倾向复用变更检测承载、
   薄封装 Signal,降低与引擎的割裂。
2. **宏编译开销**:大文件宏展开可能拖慢编译 → 用双模式缓解。
3. **宏报错可读性**:必须做 span 映射(现已保留 ident span),否则开发体验崩塌。
4. **服务端驱动 UI 安全**:远端 `.loom` 需沙箱 + 能力白名单 + 版本协商。
5. **a11y 深度**:焦点 / 读屏需尽早进模型,后补代价高。
6. **跨后端抽象边界**:核心机制与渲染后端解耦(已通过 `Backend` trait 落地),
   输入 / a11y 后续走适配层。
