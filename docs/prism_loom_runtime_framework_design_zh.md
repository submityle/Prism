# Prism Loom 游戏运行时框架设计方案（Loom Runtime）
> v1 / 顶级次世代 AAA 级运行时框架 — 以 Loom 声明式·响应式内核为「应用外壳 + 组合层 + 绑定层」，Bevy/Prism ECS 为「权威仿真内核」

> 面向 Prism(Bevy fork)的**模块化、数据驱动、确定性、ECS 权威**的游戏运行时框架。
> 工作名 **Loom Runtime**。与 **Loom Studio**(编辑器框架,见 `prism_editor_framework_design_zh.md`)、
> **Prism Gameplay**(玩法框架,见 `prism_gameplay_design_zh.md`)构成「编辑—运行—玩法」三联。
> 借形态不抄码,系统性对标:
> **Unreal**(GameInstance / World / Level / Subsystem / UMG+MVVM / Slate 保留 UI / 蓝图数据驱动)、
> **Unity**(PlayerLoop / Scene 组合 / UI Toolkit 保留模式 + 数据绑定 / Addressables / ScriptableObject)、
> **Godot**(SceneTree 保留树 / 信号 / 节点组合)、
> **Flutter·SwiftUI·Jetpack Compose**(声明式应用外壳 / 状态驱动视图 / 导航栈)、
> **SolidJS·Leptos·React**(细粒度响应 / Suspense / Error Boundary / 路由)、
> **Redux·Zustand·Pinia**(可预测全局状态 / 单向数据流)、
> **ReactiveX·MobX**(响应式玩法管线 / 派生计算)。
>
> 本文为**设计规格**。地基层(Loom 25 crate + 相关 Bevy crate + Prism 子系统 crate)为**已交付**(SHIPPED);
> 运行时框架专有层(R 系列)为**规划项**(PLANNED);**域运行时**成熟度受对应引擎子系统成熟度约束,文中显式标注。
> 全文严格区分「已实现并通过测试」与「规划中」,不把未落地能力描述为已落地。
>
> **非目标(与引擎文档一致)**:纯经典数值与确定性运行时,**不含 AI / ML / 神经网络 / LLM** 功能;
> Loom **不替代 ECS 仿真内核**——仿真(物理/动画/渲染/网络 tick)权威在 Bevy/Prism ECS;
> Loom 承担**应用外壳 + 声明式组合 + 响应式绑定 + 表现层(HUD/菜单/叙事)**,二者单向、低耦合。

- 版本: v1.0(运行时框架设计阶段)
- 适用引擎: Prism / Bevy ECS 生态
- 关键地基(SHIPPED): `prism_ui*`(响应/树/布局/样式/后端/调度/虚拟化/输入/文本)、`prism_ui_store`(可预测全局状态)、
  `prism_ui_router`(响应式路由)、`prism_ui_async`(资源状态机/Suspense/错误边界)、`prism_ui_reactive`(无毛刺响应图)、
  `prism_ui_ecs`(字段级 EcsBridge)、`prism_ui_hotreload`、`prism_ui_scheduler`(帧预算时间切片)、
  `prism_ui_i18n`、`prism_ui_a11y`、`prism_ui_sdui`、`prism_ui_overlay`、`prism_ui_motion`、`prism_ui_render_backend`;
  `bevy_app`、`bevy_ecs`、`bevy_state`、`bevy_time`、`bevy_tasks`、`bevy_input`、`bevy_scene`、`bevy_asset`、
  `bevy_reflect`、`bevy_remote`(BRP)、`bevy_diagnostic`、`bevy_window`、`bevy_winit`
- 域子系统(运行时组合对象,成熟度各异): 渲染架构/帧图、材质/WESL、`Ember` 粒子、`Animation` 动画、
  `Resonance` 音频、Lumen GI、物理、地形/世界系统、虚拟几何/体积/毛发 GPU 孪生
- 核心契约: 继承 Loom「**成本 ∝ 变化量**」;表现层一切刷新走**响应式派生**,一切跨帧任务**异步可取消**,
  一切运行态**可内省/可快照**;仿真与表现**单向数据流**,表现永不回写仿真(除显式命令通道)。

---

## 目录
1. 设计哲学与核心契约(运行时加固)
2. 为什么用 Loom 做运行时框架(可行性与边界)
3. 顶级产品对标:借形态、取什么、不抄什么
4. 分层架构总览(仿真 ↔ 绑定 ↔ 表现)
5. 应用外壳 App Shell:GameInstance / 生命周期 / PlayerLoop 集成
6. 场景组合:BSN 权威场景 + Loom 表现树的共生
7. 响应式绑定层:ECS → ViewModel → View(MVVM 对标)
8. 全局状态与数据流:Store / 单向流 / 派生选择器
9. 导航与模式栈:路由 / 菜单栈 / 模态 / 场景切换
10. HUD / 菜单 / 叙事 UI 运行时(保留模式 + 虚拟化)
11. 异步资源与流送:Suspense / 错误边界 / 加载屏
12. 数据驱动作者层:DataAsset / 配置 / 热重载
13. 输入运行时:上下文栈 / 焦点 / 手势 / 重绑定
14. 域运行时组合(挂接引擎子系统)
15. 确定性与可回放:固定步长 / 快照 / 录制重放
16. 性能工程:成本 ∝ 变化量在运行时的落地
17. 网络与表现同步(运行时视角)
18. 可观测性:运行态内省 / 诊断 HUD / 时间旅行
19. 平台与外壳:窗口 / DPI / HDR / 可达性 / i18n
20. Crate 全景与状态矩阵
21. 与 Bevy / BSN / Gameplay / Editor 的关系
22. 路线图(R1–R12)
23. 风险与取舍
24. 术语表

---

## 1. 设计哲学与核心契约(运行时加固)

在 Loom 六条铁律(P1–P6)基础上,为运行时规模追加四条:

- **R1 仿真权威,表现派生**:ECS 是唯一事实源;Loom 表现层只**读**仿真、只**派生**视图,
  绝不旁路修改仿真状态。玩家操作经**显式命令/事件通道**回到仿真,与编辑器 Command 同构。
- **R2 单向数据流**:`ECS 组件 → EcsBridge 信号 → ViewModel 派生 → View 增量`。
  闭环只在显式输入边界处发生,杜绝双向隐式耦合导致的时序噪声。
- **R3 成本 ∝ 变化量**:继承 Loom 契约;HUD/菜单按脏传播,长列表(背包/排行榜/技能栏)虚拟化,
  表现层刷新**与世界规模解耦**,**可测试**(`RecordingBackend` 断言本帧产生的最小操作)。
- **R4 异步非阻塞**:资源流送、存档、RPC、场景切换走 `prism_ui_async` 状态机 + Suspense,
  永不冻结主循环;一切长任务可取消、有错误边界与回退 UI。
- **R5 确定性守恒**:表现层不参与仿真数值;需确定性的玩法逻辑留在 ECS,整数/定点/排序优先,
  表现层的插值/动画为**纯展示**,不回写、不影响回放。
- **R6 可内省 / 可快照**:运行态(Store / 路由 / ViewModel / 元素树)可快照、可时间旅行
  (`prism_ui_timetravel` / `prism_ui_snapshot`),线上问题可复现。

### 核心契约

> **运行时 = 权威 ECS 仿真 + 响应式绑定桥 + 声明式表现树 + 异步资源状态机。**
> 仿真每帧推进世界;EcsBridge 把被读字段变为细粒度信号;ViewModel 派生出 UI 友好视图;
> Loom 协调产出**最小增量**喂给渲染后端。玩家输入经命令通道单向回到仿真。四段单向、各自可测。

---

## 2. 为什么用 Loom 做运行时框架(可行性与边界)

**可以,且很合适**——但要钉死边界:Loom 是**运行时的「应用框架 + 绑定层 + 表现层」**,不是仿真内核。

- **适合**(Loom 已交付能力正中靶心):
  - 应用外壳与生命周期:`bevy_app` 调度 + `bevy_state` 状态机 + Loom 根组件装配。
  - 状态驱动表现:`prism_ui_reactive` 细粒度响应 + `prism_ui_store` 可预测全局状态。
  - ECS ↔ UI 绑定:`prism_ui_ecs` 已提供**字段级 EcsBridge**,天然做 MVVM。
  - 异步资源:`prism_ui_async` 的资源状态机/Suspense/错误边界正是加载屏/流送 UI 所需。
  - 导航:`prism_ui_router` 做菜单栈/模式切换;虚拟化 `prism_ui_virtual` 做大列表 HUD。
  - 热重载:`prism_ui_hotreload` 让表现层/数据资产免重启迭代。
- **不适合**(明确不做,交给 ECS 与子系统):
  - 物理/动画/渲染/网络 tick 等**仿真数值**——权威在 Bevy/Prism ECS 与域子系统。
  - 确定性玩法逻辑——留在 ECS 系统内,表现层不介入。
- **结论**:Loom Runtime = 「声明式应用外壳 + 响应式绑定 + 表现层」的运行时框架;
  与 `prism_gameplay`(ECS 玩法)分工互补——玩法在 ECS 跑,表现经 Loom 绑定呈现。

---

## 3. 顶级产品对标:借形态、取什么、不抄什么

| 产品 | 值得学的运行时能力 | Loom Runtime 的取法 | 落点 |
|---|---|---|---|
| **Unreal** | GameInstance/World/Level 生命周期、UMG + MVVM ViewModel、Slate 保留 UI、Subsystem、蓝图数据驱动 | App Shell(§5)、MVVM 绑定(§7)、保留表现树(§10)、DataAsset(§12) | §5,§7,§10,§12 |
| **Unity** | PlayerLoop 可编排、Scene 组合、UI Toolkit 保留模式 + 数据绑定、Addressables、ScriptableObject | PlayerLoop 集成(§5)、场景组合(§6)、绑定(§7)、异步流送(§11) | §5,§6,§7,§11 |
| **Godot** | SceneTree 保留树、信号、节点组合、分组 | 保留表现树 + 响应信号(§4,§10) | §4,§10 |
| **Flutter / SwiftUI / Compose** | 声明式应用外壳、状态驱动视图、导航栈、生命周期钩子 | 根组件装配 + 导航栈(§5,§9)、生命周期(§5) | §5,§9 |
| **SolidJS / Leptos / React** | 细粒度响应、Suspense、错误边界、客户端路由 | 响应内核(SHIPPED)+ Suspense/错误边界(§11)+ 路由(§9) | §9,§11 |
| **Redux / Zustand / Pinia** | 可预测全局状态、单向流、时间旅行调试 | `prism_ui_store` + 单向流(§8)+ 时间旅行(§18) | §8,§18 |
| **ReactiveX / MobX** | 响应式管线、派生计算、选择器记忆化 | 派生 memo/选择器(§7,§8) | §7,§8 |

**不抄的教训**:即时模式 HUD(每帧重建、状态难留)→ 保留模式 + 稳定 ID + 脏传播;
UI 直接写游戏状态(时序噪声/回放污染)→ 单向流 + 命令通道(R1/R2);
把游戏逻辑写进 UI 层(难测、难复用)→ 玩法留 ECS,表现只绑定;
全局可变单例满天飞 → `Store` 可预测容器 + 作用域服务。

---

## 4. 分层架构总览(仿真 ↔ 绑定 ↔ 表现)

```
┌────────────────────────────────────────────────────────────────────────┐
│ L4 域运行时表现(贡献点,挂接引擎子系统)                                       │
│   HUD/准星 · 技能条/状态栏 · 过场/叙事 UI · 小地图 · 库存/商店 · 对话/任务面板   │
├────────────────────────────────────────────────────────────────────────┤
│ L3 表现运行时:路由/模式栈 · 菜单/模态 · 加载屏/Suspense · 通知/Toast · 虚拟列表 │
├────────────────────────────────────────────────────────────────────────┤
│ L2 绑定层:EcsBridge(字段级信号) · ViewModel 派生 · Store(全局状态) · 选择器   │
├────────────────────────────────────────────────────────────────────────┤
│ L1 应用外壳:GameInstance · 生命周期/状态机 · PlayerLoop 集成 · 服务定位 · 输入上下文 │
├────────────────────────────────────────────────────────────────────────┤
│ L0 权威内核(SHIPPED,不属本框架新增):Bevy/Prism ECS 仿真 · 子系统 · 资产 · 窗口 │
└────────────────────────────────────────────────────────────────────────┘
                    单向数据流:L0 ──读/派生──▶ L2 ──▶ L3/L4
                    输入回路:L3/L4 ──命令/事件──▶ L0(仅显式边界)
```

- **L0 权威内核**:已交付,仿真事实源。本框架**不重做**,只定义绑定/组合协议。
- **L1 应用外壳**:`prism_runtime_core`——GameInstance、生命周期、PlayerLoop 阶段编排、服务定位。
- **L2 绑定层**:`prism_runtime_bind`——基于 `prism_ui_ecs` 的字段级桥 + ViewModel + 选择器。
- **L3 表现运行时**:`prism_runtime_shell` + `prism_runtime_ui`——路由/模式栈/加载屏/通知/虚拟化。
- **L4 域运行时**:`prism_runtime_domains`——各域 HUD/叙事/库存,随子系统点亮。

---

## 5. 应用外壳 App Shell:GameInstance / 生命周期 / PlayerLoop 集成

- **GameInstance**:跨场景存活的根上下文(存档槽、玩家账号、设置、全局 Store),对标 UE GameInstance。
- **生命周期**:`startup → load → running → paused → unloading → shutdown`,以 `bevy_state` 状态机表达,
  每态可挂 Loom 根组件(主菜单 / 加载屏 / 游戏内 HUD / 暂停菜单),态切换即表现树切换。
- **PlayerLoop 集成**:在 `bevy_app` 调度图中为表现层开辟固定阶段:
  `SimAdvance(ECS)` → `BridgeSync(采集被读字段)` → `ViewDerive(ViewModel)` → `UiReconcile(Loom)` → `Present`。
  绑定与协调在仿真之后、呈现之前,保证读到的是**本帧收敛后的世界**。
- **服务定位**:作用域服务注册表(全局 / 关卡 / 会话),对标 UE Subsystem;服务是 `Disposable`,态卸载自动回收。
- **帧预算**:`prism_ui_scheduler` 时间切片——表现层刷新受帧预算约束,过载时降优先级分帧完成,不抢仿真预算。

---

## 6. 场景组合:BSN 权威场景 + Loom 表现树的共生

- **BSN/`.scn` 为世界权威**:实体/组件/关卡用 `bevy_scene` 的 BSN 描述、`spawn` 落地(ECS 原生,数据驱动)。
- **Loom 为表现权威**:HUD/菜单/叙事用 `loom!` 构建,享受成本契约与工具链;**二者不混写**。
- **桥接点**:场景加载完成 → 发信号 → Loom 根据世界状态装配对应 HUD;场景卸载 → `Disposable` 回收表现树。
- **分层组合**:关卡可叠加「表现层预设」(战斗 HUD / 载具 HUD / 过场模式),按模式栈组合,源数据可回溯(非破坏式)。
- **与编辑器一致**:编辑器用 `.bsn`/`.scn` 作权威场景格式,运行时同格式加载,**编辑—运行往返无损**(对齐 Editor D10)。

---

## 7. 响应式绑定层:ECS → ViewModel → View(MVVM 对标)

- **EcsBridge(SHIPPED,`prism_ui_ecs`)**:字段级把 ECS 组件暴露为细粒度信号;只有**被 UI 读到**的字段才建立订阅,
  天然「成本 ∝ 变化量」——血量变了只刷血条,不触碰其它 HUD。
- **ViewModel**:在桥之上派生 UI 友好视图(格式化数值、聚合状态、本地化文案),对标 UE MVVM / Unity 数据绑定。
  ViewModel 为纯派生(`memo`),输入不变则不重算,输出不变则不刷新。
- **选择器(selector)**:记忆化的派生读取(如「可见敌人中最近的 N 个」),避免每帧全量扫描。
- **单向为主、双向可选**:HUD 单向(世界→UI);设置/表单可双向(`prism_ui_form` + 显式命令回写),双向仅限输入边界。
- **零样板**:杜绝「每帧手写刷新 UI」;变更自动推送,未变更零开销。

---

## 8. 全局状态与数据流:Store / 单向流 / 派生选择器

- **Store(SHIPPED,`prism_ui_store`)**:Loom 版 Redux/Zustand——单值容器 + 细粒度响应,
  commit 只通知受影响观察者。用于**跨场景、非 ECS 的会话状态**(设置、UI 偏好、临时流程状态、菜单导航)。
- **职责切分**:世界/玩法状态在 **ECS**(权威、确定性、可回放);应用/会话/UI 状态在 **Store**(表现层私有)。
  二者不混:仿真数据不进 Store,UI 偏好不进 ECS。
- **单向流**:`action → reducer → 新状态 → 订阅派生`,可预测、可时间旅行(§18)。
- **派生选择器**:跨 Store 与 EcsBridge 的组合派生(如「当前可购买且金币足够的商品」),记忆化、按需重算。

---

## 9. 导航与模式栈:路由 / 菜单栈 / 模态 / 场景切换

- **路由(SHIPPED,`prism_ui_router`)**:响应式客户端路由,位置在信号里;用于菜单/界面导航(主菜单/设置/存档/多人大厅)。
- **模式栈(mode stack)**:游戏内界面以栈管理(HUD 底 → 暂停 → 库存 → 模态确认),返回即出栈,输入上下文随栈切换(§13)。
- **模态与遮罩**:`prism_ui_overlay` 做模态层/焦点陷阱/遮罩,保证模态期间底层不可交互且可达性正确。
- **场景切换**:切场景 = 状态机转移 + 加载屏(Suspense,§11)+ 旧表现树 `Disposable` 回收 + 新根装配;
  支持无缝切换(保留持久 HUD)与硬切换(全屏加载)。

---

## 10. HUD / 菜单 / 叙事 UI 运行时(保留模式 + 虚拟化)

- **保留模式**:表现树持久存在、稳定 ID,只做增量更新,避免即时模式的「每帧重建 + 状态丢失」。
- **虚拟化(SHIPPED,`prism_ui_virtual`)**:背包/技能树/排行榜/聊天/日志等长列表只构造可视窗口,
  万级条目滚动延迟与总量解耦。
- **动效一等公民(SHIPPED,`prism_ui_motion`/`prism_ui_anim`)**:伤害数字、过渡、状态图标脉冲为声明式过渡,
  纯展示、不回写仿真(R5)。
- **文本栈(SHIPPED,`prism_ui_text`)**:整形/富文本/BiDi/IME,支撑对话系统、字幕、多语言 UI。
- **叙事 UI**:对话/任务/过场面板绑定玩法 ViewModel(当前装备/外观出现在过场),过场可接管/交还相机与输入(对齐 Gameplay §45)。

---

## 11. 异步资源与流送:Suspense / 错误边界 / 加载屏

- **资源状态机(SHIPPED,`prism_ui_async`)**:`no_std` 友好的显式驱动状态机,建模「资产流送 / 存档加载 / RPC 解析」,
  不拖入 std futures 运行时。
- **Suspense**:表现层声明「数据未就绪时的回退 UI」(骨架屏/加载旋转/进度条),就绪自动替换,零手写轮询。
- **错误边界**:加载失败/RPC 失败不崩表现树,降级到错误 UI + 重试命令;失败可见、可恢复(R4)。
- **加载屏与流送**:大世界背景流送(对接 World Partition)期间显示进度/提示,前台不冻结;
  配合 `bevy_asset` 异步加载句柄。

---

## 12. 数据驱动作者层:DataAsset / 配置 / 热重载

- **DataAsset**:技能/物品/UI 布局/本地化/主题皆为可反射数据资产(`bevy_reflect` + `bevy_asset`),
  策划改数据不改码(对齐 Gameplay §13)。
- **主题与设计令牌(SHIPPED,`prism_ui_theme`/`prism_ui_style`/`prism_ui_scoped`)**:皮肤/令牌编译,换肤不改结构。
- **本地化(SHIPPED,`prism_ui_i18n`)**:文案/复数/方向性数据驱动,运行时切换语言即时生效。
- **热重载(SHIPPED,`prism_ui_hotreload`)**:表现结构/样式/数据资产免重启迭代,开发期秒级反馈。
- **SDUI(SHIPPED,`prism_ui_sdui`)**:服务端驱动 UI 沙箱,可用于活动页/公告/可远端配置的运营界面(受沙箱约束)。

---

## 13. 输入运行时:上下文栈 / 焦点 / 手势 / 重绑定

- **输入上下文栈**:随模式栈切换的输入映射(游戏/菜单/载具/过场),对标 UE Enhanced Input;栈顶优先消费。
- **焦点与导航(SHIPPED,`prism_ui_input`/`bevy_input_focus`)**:手柄/键鼠/触屏统一焦点模型,方向导航 + Tab 环,
  可达性友好(§19)。
- **声明式手势竞技场**:命中测试 + 手势消解(点击/拖拽/长按/滑动),表现层声明手势,冲突由竞技场裁决。
- **重绑定**:映射上下文/修饰器/触发器为 DataAsset,可运行时重绑定(玩家自定义按键,对齐 Gameplay §10),热重载。

---

## 14. 域运行时组合(挂接引擎子系统)

> 域运行时**只做表现与绑定**,算法在子系统;子系统未成熟则域表现为空壳,显式标注不冒进。

| 域 | 运行时表现职责 | 挂接子系统 | 成熟度约束 |
|---|---|---|---|
| 战斗 HUD | 血条/护盾/技能冷却/增益图标/伤害数字 | Gameplay(GAS/Tag) | 受 GAS 成熟度约束 |
| 小地图/罗盘 | 实体投影/兴趣点/视野 | 世界系统 / `bevy_camera` | 受世界系统约束 |
| 库存/商店/装备 | 虚拟化网格 + 拖拽 + 预览 | Gameplay 物品系统 | 受物品系统约束 |
| 对话/任务/叙事 | 分支对话/任务追踪/字幕 | Gameplay 叙事(§31) | 受叙事系统约束 |
| 过场 UI | 字幕/黑边/可跳过/数据绑定 | Timeline / 相机 | 受序列系统约束 |
| 载具/建造 HUD | 仪表/建造网格/资源 | 对应玩法子系统 | 受子系统约束 |
| 诊断 HUD | 帧率/内存/网络/实体数(§18) | `bevy_diagnostic` | SHIPPED 可点亮 |

---

## 15. 确定性与可回放:固定步长 / 快照 / 录制重放

- **固定步长仿真**:确定性玩法用固定步长 tick(ECS 侧),表现层以插值平滑呈现(纯展示,不参与数值,R5)。
- **表现不污染回放**:录制/回放只记录输入与仿真状态;表现层的动画/插值不入录,保证回放逐位复现。
- **运行态快照(SHIPPED,`prism_ui_snapshot`)**:表现树可快照为确定性文本,用于回归黄金测试与线上问题复现。
- **时间旅行(SHIPPED,`prism_ui_timetravel`)**:Store/路由状态可回溯调试(开发态),与仿真回放互补。

---

## 16. 性能工程:成本 ∝ 变化量在运行时的落地

- **脏传播**:EcsBridge 字段级订阅 → 只刷新真正变化的 HUD 片段;静态子树编译期提升、零运行时构造。
- **虚拟化**:长列表仅构造可视窗口;瓦片/脏矩形重绘(GPU 后端),大画布只重绘变化区域。
- **帧预算调度(SHIPPED,`prism_ui_scheduler`)**:表现层刷新受预算约束,过载分帧,优先级抢占,绝不饿死仿真。
- **可测性能(SHIPPED,`RecordingBackend`)**:把「本帧产生哪些后端操作」变为可断言事实,性能回归钉成测试。
- **预算示例(目标)**:HUD 刷新 <0.5ms/帧(典型),长列表滚动与总量解耦,场景切换表现装配 <1 帧可感知冻结。
- **GPU 后端 parity**:`prism_ui_render_backend` wgpu 后端需真机对拍 CPU 参考后端后方可承诺(与引擎文档一致,未验证不写死)。

---

## 17. 网络与表现同步(运行时视角)

> 网络仿真/预测/回滚权威在 ECS 与 Gameplay(§16/§33);本节只讲**表现层如何消费网络态**。

- **复制态 → 表现**:复制组件经 EcsBridge 暴露为信号,远端实体 HUD(队友血条/名牌)自动绑定更新。
- **预测与回滚不穿透表现**:预测/回滚在 ECS 内收敛,表现层读的是收敛后状态;回滚抖动由插值吸收,不闪 UI。
- **延迟补偿提示**:ping/丢包/重连状态经 Store 暴露给网络状态 HUD(连接中/重连/掉线),走 Suspense/错误边界。
- **无缝迁移/切服**:表现层以状态机 + 加载屏覆盖迁移窗口,持久 HUD 保留,会话状态在 Store 跨迁移存活。

---

## 18. 可观测性:运行态内省 / 诊断 HUD / 时间旅行

- **诊断 HUD(SHIPPED 可点亮)**:帧率/帧时间/内存预算/实体数/网络(`bevy_diagnostic`),叠加 `prism_ui_overlay`。
- **运行态内省(SHIPPED,`prism_ui_devtools`)**:表现树/信号图可 dump 为确定性文本,线上问题可复现。
- **时间旅行(SHIPPED,`prism_ui_timetravel`)**:Store/路由回溯(开发态),定位「界面为何到了这个状态」。
- **与编辑器 BRP 互通**:运行时经 `bevy_remote`(BRP)暴露给 Loom Studio,实现运行态远程检查/调参(对齐 Editor §9/§10)。

---

## 19. 平台与外壳:窗口 / DPI / HDR / 可达性 / i18n

- **窗口/平台(SHIPPED,`bevy_window`/`bevy_winit`/`bevy_android`)**:多分辨率/多 DPI/全屏/多窗。
- **DPI/HDR/色彩**:表现层按逻辑像素布局,DPI 缩放无锯齿;HDR UI 叠加遵循色彩管理(与渲染架构对齐)。
- **可达性(SHIPPED,`prism_ui_a11y`/`bevy_a11y`)**:语义树/焦点序/朗读;手柄导航与可缩放 UI。
- **国际化(SHIPPED,`prism_ui_i18n`)**:多语言/BiDi/字体回退(`prism_ui_text`),运行时切换。

---

## 20. Crate 全景与状态矩阵

| crate | 职责 | 状态 |
|---|---|---|
| `prism_runtime_core` | GameInstance、生命周期/状态机、PlayerLoop 阶段编排、服务定位、Disposable | 🔜 规划 |
| `prism_runtime_bind` | EcsBridge 之上的 ViewModel/选择器/绑定约定、单向流回写命令通道 | 🔜 规划 |
| `prism_runtime_shell` | 路由/模式栈/模态/场景切换/加载屏装配 | 🔜 规划 |
| `prism_runtime_ui` | HUD/菜单/叙事通用控件、虚拟列表、通知/Toast、诊断 HUD | 🔜 规划 |
| `prism_runtime_input` | 输入上下文栈、焦点导航、手势竞技场、重绑定 | 🔜 规划 |
| `prism_runtime_data` | DataAsset/配置/主题/i18n 运行时装载 + 热重载编排 | 🔜 规划 |
| `prism_runtime_net` | 复制态 → 表现绑定、网络状态 HUD、迁移窗口外壳(表现侧) | 🔜 规划 |
| `prism_runtime_domains` | 各域运行时表现装配(挂接子系统) | 🔜 规划 |
| `prism_runtime_app` | 二进制:装配模块、窗口、与 `bevy_app` 集成、PIE 对接 | 🔜 规划 |
| — 复用地基 — | `prism_ui*` 全家桶(含 store/router/async/ecs/reactive/virtual/scheduler/snapshot/timetravel/motion/text/i18n/a11y/overlay/hotreload/sdui) | ✅ 已交付 |
| — 复用内核 — | `bevy_app/ecs/state/time/tasks/input/scene/asset/reflect/remote/diagnostic/window/winit` | ✅ 已交付 |

> 原则:每个新 crate 以「crate + 测试 + 文档」三件套闭环交付;域运行时成熟度受子系统约束;未落地并本地提交前不计入「已胜出」。

---

## 21. 与 Bevy / BSN / Gameplay / Editor 的关系
- **Loom 是唯一表现地基**:全部运行时 UI/HUD 用 `loom!` 构建,享受成本契约与工具链。
- **ECS 是唯一仿真内核**:世界/玩法/网络在 Bevy/Prism ECS 跑;Loom 只读/派生,经命令通道回写(R1)。
- **与 Gameplay 分工**:`prism_gameplay`(Actor/GAS/输入/AI/网络)在 ECS 实现**玩法逻辑**;
  Loom Runtime 实现其**表现与绑定**。本文第 7/14/17 节是对 Gameplay §35(MVVM)的运行时框架化落地,**不重复**玩法算法。
- **与 Editor 共生**:Loom Studio 编辑 `.bsn`/`.scn` 权威场景与 DataAsset,Loom Runtime 同格式加载——**编辑即所得、往返无损**;
  运行态经 BRP 回连编辑器实现 PIE/远程调参。
- **与子系统分工**:运行时**不实现**渲染/物理/动画/音频算法,只提供其**运行态表现 + 绑定 + 诊断**;域表现随子系统成熟点亮。
- **渐进采用**:可先在现有 app 内嵌「诊断 HUD + 一块绑定面板」,逐步长成完整运行时框架。

---

## 22. 路线图(R1–R12)
- **R1 应用外壳**:`prism_runtime_core`——GameInstance、生命周期状态机、PlayerLoop 阶段(BridgeSync/ViewDerive/UiReconcile)、服务定位、Disposable。配套生命周期回归测试。
- **R2 绑定层**:`prism_runtime_bind`——EcsBridge→ViewModel 约定、选择器记忆化、单向流命令通道。打通「组件变→HUD 刷新」最小闭环并断言最小操作集。
- **R3 表现外壳**:`prism_runtime_shell`——路由 + 模式栈 + 模态 + 场景切换 + 加载屏(Suspense)。
- **R4 核心 HUD/菜单**:`prism_runtime_ui`——主菜单/暂停/设置 + 血条/技能条 + 虚拟化库存 + 诊断 HUD。
- **R5 输入运行时**:`prism_runtime_input`——上下文栈 + 焦点导航 + 手势竞技场 + 重绑定。
- **R6 数据驱动 + 热重载**:`prism_runtime_data`——DataAsset/主题/i18n 装载 + 热重载编排。
- **R7 异步流送**:加载屏/错误边界/大世界背景流送 UI 对接 `bevy_asset` + World Partition。
- **R8 可观测性**:诊断 HUD 全量 + 运行态快照/时间旅行 + BRP 回连编辑器(PIE/远程调参)。
- **R9 网络表现**:`prism_runtime_net`——复制态绑定 + 网络状态 HUD + 迁移窗口外壳。
- **R10 域运行时**:`prism_runtime_domains`——战斗/库存/对话/过场,随子系统点亮。
- **R11 确定性回放**:固定步长 + 插值分离 + 录制重放表现不污染 + 黄金快照。
- **R12 平台收口**:多 DPI/HDR/可达性/i18n/移动端外壳 + `prism_runtime_app` 装配二进制。

**建议优先级**:R1→R2→R4 = 可用「绑定式 HUD」最短路径;R3 外壳并行;R5/R6 = 可玩界面分水岭;
R7/R8 = AAA 运行态前提;R9/R10/R11 = 高级生产力;R12 为收口。

---

## 23. 风险与取舍
1. **仿真/表现边界侵蚀**:最大风险是表现层偷偷写仿真。以 R1/R2 单向流 + 命令通道从架构上杜绝,代码评审钉死。
2. **绑定粒度**:EcsBridge 字段级订阅覆盖面决定「成本 ∝ 变化量」兑现度;默认字段级 + 可选粗粒度降订阅开销。
3. **确定性守恒**:表现层插值/动画必须纯展示;回放只录输入与仿真态,表现不入录,黄金测试守恒。
4. **GPU 后端 parity**:`prism_ui_render_backend` wgpu 后端需真机对拍 CPU 参考后端后方可承诺(未验证不写死)。
5. **Store 与 ECS 职责混淆**:严格切分——世界/玩法进 ECS,应用/会话/UI 进 Store;混用将破坏回放与可测性。
6. **域运行时与子系统耦合**:运行时只做表现,算法在子系统;子系统未成熟则域表现为空壳,显式标注。
7. **网络表现抖动**:预测/回滚在 ECS 收敛,表现读收敛态 + 插值吸收抖动,避免 HUD 闪烁。
8. **范围蔓延**:覆盖面大;严格按 R1→R12 分批,每批三件套闭环,不堆半成品。
9. **与 Gameplay 文档重叠**:本文只做「运行时框架化 + 表现绑定」,玩法算法引用 `prism_gameplay_design_zh.md`,不复制。

---

## 24. 术语表
- **GameInstance**:跨场景存活的根运行时上下文(存档/账号/设置/全局 Store)。
- **PlayerLoop 阶段**:在 `bevy_app` 调度中为「仿真→绑定→派生→协调→呈现」划定的有序阶段。
- **EcsBridge(SHIPPED)**:`prism_ui_ecs` 的字段级桥,把 ECS 组件字段暴露为细粒度响应信号。
- **ViewModel / 选择器**:ECS/Store 之上的 UI 友好派生视图 / 记忆化派生读取(MVVM 对标)。
- **单向数据流**:世界→绑定→表现单向更新,仅在显式输入边界经命令通道回写仿真。
- **Store(SHIPPED)**:`prism_ui_store` 可预测全局状态容器,承载应用/会话/UI 状态(非仿真)。
- **模式栈 / 输入上下文栈**:界面与输入映射以栈管理,栈顶优先,随模式切换。
- **Suspense / 错误边界(SHIPPED)**:`prism_ui_async` 的未就绪回退 UI / 失败降级 UI。
- **保留模式**:表现树持久 + 稳定 ID + 增量更新(对立于即时模式每帧重建)。
- **虚拟化(SHIPPED)**:`prism_ui_virtual` 只构造可视窗口,延迟与列表总量解耦。
- **BRP(SHIPPED)**:`bevy_remote` 的 JSON-RPC 2.0 运行时远程检查/变更协议,运行态回连编辑器。
- **Disposable**:服务/订阅生命周期句柄,态卸载自动回收防泄漏。
- **成本 ∝ 变化量**:Loom 核心契约,刷新量正比于真实变化量而非世界规模,且可测(`RecordingBackend`)。

---

> 本文为设计规格。L0 地基项均已实现并通过测试(Loom 25 crate / 600+ 测试,Clippy 零告警;
> 相关 Bevy crate 随引擎交付);运行时框架专有层(R1–R12)与域运行时在落地并本地提交前不计入「已胜出」,
> 且域运行时成熟度显式受对应子系统成熟度约束。本框架为**纯经典数值 / 确定性**运行时,不含 AI/ML/LLM 功能;
> Loom 承担应用外壳/绑定/表现,**ECS 为唯一仿真权威**。
