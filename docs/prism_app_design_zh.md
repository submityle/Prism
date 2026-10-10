# Prism App 顶级次世代 AAA 级应用外壳 / 插件 / 调度驱动设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **应用外壳 + 插件系统 + 调度驱动 + 主循环 + 子应用管线 + 状态机** 内核设计。它是 `bevy_app` 的自研替代，位于 ECS 之上、游戏运行时框架之下。
> 借形态不抄码。借鉴：
> - **插件/调度/子应用**：Bevy App（Plugin / PluginGroup / SubApp / Main 调度集 / States / 固定时间步 / Runner）
> - **主循环与 tick 组**：Unity（PlayerLoop 可插拔子系统）、Unreal（Engine Tick / Tick Group / World Tick）、Godot（MainLoop / SceneTree / process vs physics_process）
> - **固定步长积分**：Glenn Fiedler「Fix Your Timestep」（累加器 + 插值 alpha + 死亡螺旋防护）
> - **帧节奏/呈现**：主机与移动端 frame pacing（vsync 对齐、present 时间戳、低延迟管线）
> - **流水线并行**：Bevy pipelined rendering（渲染子应用与下一帧仿真并行，配合 ECS 提取管线）
> - **声明式外壳**：Flutter / SwiftUI / Jetpack Compose（应用生命周期 + 状态驱动）
> - **管线/阶段**：flecs pipeline（阶段化 system 分组）
> 本文为纯经典调度/循环路线，**不含任何 AI/ML 内容**。

- 版本: v0.3（设计阶段，未进入编码；v0.1→v0.2 新增第 24 章「AAA 高级功能增补」：插件依赖图与分组/系统与模块热重载/子应用流水线深化/运行模式(无头服务器/编辑器内嵌/专用服务器)/优雅生命周期/控制台变量(cvar)与配置级联/崩溃隔离与系统恢复/子状态与状态作用域实体/多窗口多世界；均为 PLANNED，无代码；v0.2→v0.3 新增第 25 章「跨 crate 契约对齐」：主循环编排 time 帧节奏/VRR/帧预算/挂起恢复 + 向 tasks 车道喂预算 / 子应用流水线以 ECS 提取为边界·tasks 线程类分离为执行·transform 双缓冲保读一致 / 配置与 cvar 经 reflect 特性校验 / 多世界各持独立 time 时间域）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: `prism_ecs`（World/Schedule/States）、`prism_tasks`（执行器/流水线）、`prism_time`（时钟/固定步长）、`prism_diagnostic`（可选）、`prism_reflect`（可选，配置/设置反射）
- 层级定位: ECS 文档 L5「App / Plugin / 固定阶段」；**上层** `prism_loom_runtime_framework`（GameInstance/HUD/路由）跑在本层之上，不要混淆
- 明确约束: 应用图(App graph/Plugin 注册)核心 `no_std + alloc` 可行；**Runner/主循环/帧节奏需 `std`**；`multi_thread` / `pipelined` / `winit` / `headless` / `determinism` / `reflect` / `trace` 为 feature；**不依赖任何 `bevy_*` crate**

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier）
4. 分层架构与边界（与 Loom Runtime 的分工）
5. 核心模型：App / SubApp / World / Schedule
6. 插件系统（Plugin / PluginGroup / 依赖解析 / 去重 / 排序）
7. 主循环与 tick 组（First→Last + FixedMain）
8. 固定时间步与可变步长（累加器 / 插值 alpha / 死亡螺旋防护）
9. 子应用与流水线并行（渲染提取 / 服务器子应用）
10. Runner 抽象（窗口 / 无头 / 专用服务器 / 测试）
11. 状态机（States / 计算态 / 子态 / 状态作用域实体）
12. 生命周期与平台事件（启动 / 挂起恢复 / 退出 / 低内存）
13. 帧节奏与呈现（vsync / 帧限 / 低延迟 / present 时间戳）
14. 配置与设置分层（默认 → 平台 → 用户 → 命令行）
15. 确定性与可回放（固定序 / 录制重放 / 回滚钩子）
16. 可观测性（阶段火焰图 / 帧统计 / 启动耗时）
17. 高级功能增补
18. 性能工程
19. 易用性与 Bevy 迁移策略
20. crate 分层与模块布局
21. 契约、不变量与版本化
22. 路线图（M0–M6）与基准即规格
23. 诚实边界与风险
24. AAA 高级功能增补（v0.2）
25. 跨 crate 契约对齐（v0.3：与 ecs/tasks/time/reflect/transform 升级后的新契约对齐）

---

## 1. 设计哲学与目标

`prism_app` 是把一堆 `prism_ecs` World 与 Schedule「组织成一个能跑起来的引擎实例」的外壳：它定义**由谁、以什么顺序、在什么时间步、在哪个线程、以什么帧节奏**驱动仿真与表现，并提供**插件**作为引擎能力的唯一装配单元。

**一句话定位**：`prism_app` 是 Prism 的「可组合运行时装配器 + 确定性心跳」——用插件声明式装配引擎能力，用阶段化调度 + 固定/可变双时间步驱动，用子应用把仿真与渲染流水线并行，并对外保持近 Bevy 的 `App::new().add_plugins(..).add_systems(..).run()` 手感。

四条总目标（按权重）：

1. **性能**：子应用流水线并行（仿真/渲染错帧重叠）、阶段内 system 经 ECS 冲突图 + fiber 并行、启动期插件装配零运行时开销、帧节奏对齐降低延迟与卡顿。
2. **效果（规模能力）**：固定步长确定性心跳支撑回滚网络；多 World/子应用支撑大世界分区与渲染提取；平台生命周期支撑移动端挂起恢复与主机认证。
3. **易用**：插件即装配、`add_systems(Update, ..)`、声明式状态机、与 `bevy_app` 近乎一致的 API；高级能力默认关闭、按档位开启。
4. **可移植 + 档位化**：同一外壳经 capability/quality tier/feature 三重门控，从无头服务器/移动端缩放到高端桌面 AAA；Runner 可换（窗口/无头/服务器/测试）。

非目标：不做 UI/HUD/路由（归 Loom Runtime）；不做 gameplay 玩法（归 Prism Gameplay）；不内置具体窗口后端（经 `prism_window` 注入）。

---

## 2. 参考产品取舍

| 产品 | 吸收 | 规避 |
|---|---|---|
| Bevy App | Plugin/PluginGroup、SubApp、Main 调度集、States、固定步长、Runner、事件更新 | 插件装配偶有隐式顺序坑、SubApp 提取样板 |
| Unity PlayerLoop | 可插拔子系统循环、Update/FixedUpdate/LateUpdate 分组 | C# 绑定、GC 停顿 |
| Unreal Engine | Tick Group（PrePhysics/DuringPhysics/PostPhysics/PostUpdate）、World/Engine 分层、子系统生命周期 | 与 UObject 耦合、专有 |
| Godot | MainLoop/SceneTree、process vs physics_process、notification 生命周期 | 场景树即真相（我们 ECS 权威） |
| Fix Your Timestep | 累加器固定步 + 渲染插值 alpha + 死亡螺旋防护 | —— |
| 主机/移动 frame pacing | present 时间戳对齐、低延迟渲染管线、自适应帧限 | 平台专有 API（经 RHI/窗口隔离） |
| flecs pipeline | 阶段化 system 分组、阶段即查询 | C ABI 风格 |
| Flutter/SwiftUI/Compose | 应用生命周期 + 状态驱动外壳 | 前端语境、重 GC |

综合：**Bevy App 人体工学 + Unity/Unreal 的 tick 组表达力 + Fix-Your-Timestep 的确定性心跳 + 主机级 frame pacing** 为四支柱，叠加 **子应用流水线 + 平台生命周期 + 可换 Runner** 的 AAA 能力层，全部走 feature/档位门控。

---

## 3. 档位化（capability / quality tier）

| 维度 | 说明 | 示例 |
|---|---|---|
| **capability** | 运行时探测的硬件/环境能力 | 线程数、是否有窗口/显示器、是否支持高精度计时器、是否移动端 |
| **quality tier** | 运行形态档 | server（无头定帧）/ mobile（省电可变步 + 生命周期）/ desktop（流水线 + frame pacing） |
| **feature flag** | 编译期裁剪 | `multi_thread` / `pipelined` / `winit` / `headless` / `determinism` / `reflect` |

目标：无头服务器只付固定步长心跳成本；桌面端叠加子应用流水线 + frame pacing + 窗口生命周期；移动端走省电可变步 + 挂起/恢复。

---

## 4. 分层架构与边界

```
L7  游戏运行时   prism_loom_runtime_framework（GameInstance/HUD/路由/MVVM）   ← 在 App 之上
----------------------------------------------------------------------------------
L6  App 外壳     prism_app：App / SubApp / Plugin / Runner / 主循环 / 状态机   ← 本文
L5  调度         prism_ecs：Schedule / Executor / Fiber 作业图
L4  仿真         prism_ecs：World / Query / System / Command
L3  运行时服务   prism_tasks（执行器）/ prism_time（时钟/固定步）/ prism_diagnostic
L2  平台 I/O     prism_window / prism_input（经插件注入，不被 App 直接 new）
L1  地基         prism_math / prism_reflect / prism_utils / prism_platform
```

**边界契约**：
- `prism_app` 向下只调度 `prism_ecs` 的 World/Schedule，向上只暴露 App/Plugin/Runner API。
- 窗口/输入/渲染/音频等**都以插件形式注入**，App 本身不 `new` 任何平台对象——保证无头/服务器/测试可运行。
- Loom Runtime 把自己注册为一个（组）插件挂到 App 上；App 不反向依赖 Loom。

---

## 5. 核心模型：App / SubApp / World / Schedule

```rust
pub struct App {
    sub_apps: SubApps,                 // main + 若干具名子应用（如 Render）
    runner: Box<dyn Fn(App) -> AppExit>,
    plugin_registry: PluginRegistry,   // 已注册插件、状态、构建顺序
    plugins_state: PluginsState,       // Adding → Ready → Finished → Cleaned
}

pub struct SubApp {
    world: World,                      // 一个 prism_ecs World
    schedules: Schedules,              // 具名 Schedule 集（Main/Fixed/...）
    extract: Option<ExtractFn>,        // 从 main world 提取到本子应用（见 §9）
    update_schedule: ScheduleLabel,    // run() 时跑哪张图
}
```

- **App** = 一组 SubApp + 一个 Runner + 插件装配状态机。
- **main SubApp** 承载权威仿真 World；**Render SubApp** 承载渲染 World（经 extract 单向喂数据）。
- **Schedule** 复用 `prism_ecs` 的调度图；App 只负责「每帧按什么顺序跑哪些 Schedule」。

---

## 6. 插件系统（唯一装配单元）

```rust
pub trait Plugin: Send + Sync {
    fn build(&self, app: &mut App);               // 装配：加系统/资源/事件/状态
    fn ready(&self, app: &App) -> bool { true }   // 异步就绪门（如资产后端初始化）
    fn finish(&self, app: &mut App) {}            // 全部 ready 后的收尾（拿设备句柄等）
    fn cleanup(&self, app: &mut App) {}           // 启动完成后释放构建期临时资源
    fn name(&self) -> &str { type_name } 
    fn is_unique(&self) -> bool { true }          // 默认禁止重复添加
}
```

高级能力：

- **PluginGroup**：成组装配（如 `PrismDefaultPlugins`），可 `.disable::<X>()` / `.add_before/after::<Y>()` 调整。
- **依赖解析**：插件声明依赖/先后约束，App 做拓扑排序与缺失检测；循环依赖在装配期报错（而非运行期崩）。
- **去重**：`is_unique` 默认防重复添加；重复显式报错，杜绝「加了两次相机插件」类隐性 bug。
- **异步就绪**：`ready/finish` 两段式解决「插件 B 需要插件 A 初始化出的 GPU 设备」这类跨插件时序。
- **档位装配**：插件可按 §3 capability/tier 条件装配不同 system 集。

---

## 7. 主循环与 tick 组

每帧 main SubApp 跑 `Main` 调度，内部按固定顺序运行一组阶段 Schedule（Bevy 形态 + Unreal/Unity tick 组表达力）：

```
启动一次:  PreStartup → Startup → PostStartup
每帧:      First
           → RunFixedMainLoop{ 内部按累加器跑 0..N 次 FixedMain(见 §8) }
           → PreUpdate → StateTransition → Update → PostUpdate
           → Last
```

`FixedMain` 内部再细分 tick 组（借 Unreal）：`FixedFirst → FixedPreUpdate → FixedUpdate → FixedPostUpdate → FixedLast`，物理/网络/确定性 gameplay 挂在这里。可变步表现（相机平滑、插值、输入采样）挂 `Update`。

阶段是**可扩展**的：插件可 `insert phase before/after`，但核心阶段顺序为不变量（§21）。

---

## 8. 固定时间步与可变步长

借「Fix Your Timestep」：

```
accumulator += min(frame_dt, max_frame_dt)   // max_frame_dt 防死亡螺旋
while accumulator >= fixed_dt {
    run(FixedMain)
    accumulator -= fixed_dt
    if steps++ > max_substeps { break }       // 螺旋熔断，宁可慢放不卡死
}
alpha = accumulator / fixed_dt                // 渲染插值因子，供 Update 平滑
```

- `fixed_dt` 默认 1/60（可按 quality tier/网络 tickrate 配），`determinism` 档下为精确有理步长。
- 暴露 `Time<Fixed>` / `Time<Virtual>`（可暂停/缩放）/ `Time<Real>`（墙钟）三种时钟（经 `prism_time`）。
- `alpha` 让表现层对「上一固定态 ↔ 当前固定态」插值，消除低 tickrate 下的视觉抖动。
- 螺旋熔断保证即使单帧超时也不会无限追帧卡死（主机认证关注项）。

---

## 9. 子应用与流水线并行

```
帧 N:   [main: 仿真 World]  ──extract(只读)──▶  [Render SubApp: 渲染 World]
帧 N+1: [main: 仿真 N+1]  与  [Render: 渲染 N]  在不同线程重叠执行（pipelined）
```

- **extract**：main world → render world 的单向只读拷贝（对接 ECS 文档 §23.5 提取管线 + `ExtractComponent` trait）。这是解耦渲染与仿真、让 `prism_render_scene` 脱 Bevy 的关键接缝。
- **流水线并行**：`pipelined` 档下渲染子应用与下一帧仿真并行执行，吞吐接近翻倍（延迟换吞吐）；低延迟档可关。
- 子应用也可用于**专用服务器**（无渲染子应用，只 main）、**大世界分区**（多 World）。

---

## 10. Runner 抽象

Runner 决定「谁来驱动帧」——App 把控制权交给它：

| Runner | 场景 | 行为 |
|---|---|---|
| `WinitRunner` | 窗口客户端 | 由窗口事件循环驱动，集成 §12 生命周期、§13 frame pacing |
| `HeadlessRunner` | 工具/测试/CI | 跑 N 帧或直到退出条件，无窗口 |
| `DedicatedServerRunner` | 专用服务器 | 定 tickrate 固定步长心跳，无渲染子应用 |
| `ScheduleRunnerOnce` | 单元测试 | 跑一帧，断言 World 状态 |

Runner 经 `app.set_runner(..)` 注入；默认按 capability（有无显示器）自动选择。保证**同一套插件/系统在窗口与无头下行为一致**，只是驱动方式不同。

---

## 11. 状态机

复用并增强 `prism_ecs` 的 States（ECS 文档 §8），在 App 层提供生命周期编排：

- **States**：枚举态（如 `Loading/MainMenu/InGame/Paused`），`OnEnter/OnExit/OnTransition` 调度钩子。
- **计算态（Computed）**：由其它态派生（如 `InGame && !Paused ⇒ Simulating`），单向推导无毛刺。
- **子态（Sub-states）**：仅在父态存在时有效（如 `InGame` 下的 `Combat/Explore`）。
- **状态作用域实体**：标记实体「仅在某态存在」，`OnExit` 自动 despawn（对接 ECS §23.6 层级 despawn），消除手写清理。
- **`run_if(in_state(..))`**：system 按态条件运行。

---

## 12. 生命周期与平台事件

主机/移动端认证关注项，经窗口插件转成 App 事件：

- `Startup`（一次）/ `Suspended`（移动端后台/主机休眠：暂停仿真、释放瞬态 GPU 资源）/ `Resumed`（重建交换链/上下文）/ `AppExit`（优雅退出：flush 命令、保存、关设备）。
- `LowMemory`（移动端：主动卸载流送 cell）、`FocusChanged`、`WillRenderFirstFrame`。
- 退出有**优雅路径**：排空命令队列、运行 `Last` 收尾 system、按插件逆序 `cleanup`，保证存档/网络断连一致。

---

## 13. 帧节奏与呈现

- **frame pacing**：对齐 present 时间戳，稳定帧间隔（防 1%low 抖动），而非一味追高平均帧率。
- **帧限（frame limiter）**：无 vsync 时按目标帧率限速，省电/控温（移动端重要）。
- **低延迟管线**：可配「采样输入 → 仿真 → 渲染 → present」的最短路径档，减少输入延迟（竞技场景）。
- **自适应**：按最近帧时间直方图在 quality tier 内微调（分辨率缩放交给渲染层，这里只管节奏）。
- 这些都经 `prism_window`/RHI 的平台 API 实现，App 只持有策略与时钟。

---

## 14. 配置与设置分层

分层覆盖（后者覆盖前者）：`引擎默认 → 平台档位（quality tier） → 用户设置 → 命令行/环境变量`。经 `prism_reflect` 做类型化读写与热重载；设置变更以事件广播给关心的子系统。无头/服务器可纯命令行驱动。

---

## 15. 确定性与可回放

- **固定序**：`determinism` 档下阶段顺序、system 顺序、固定步长、随机源全部确定；跨平台一致依赖定点/软件数学路径（与 ECS §14 一致）。
- **录制重放**：记录每固定步的输入帧 + 周期性 World 快照（ECS §16.5），可逐帧重放复现 bug / 做回归。
- **回滚钩子**：App 暴露「回退到快照 → 重跑 FixedMain 到当前」的编排点，供网络层（`prism_replication`）实现回滚预测。

---

## 16. 可观测性

- **阶段火焰图**：每阶段/每 Schedule 耗时，接 ECS §16.6 system 火焰图与 `prism_ui_devtools`。
- **帧统计**：帧时间、固定步 substep 数、extract 耗时、流水线重叠率、present 延迟。
- **启动耗时**：各插件 `build/finish` 耗时，定位启动瓶颈。
- **诊断 HUD**：运行态内省（当前态、World 实体数、调度图），对接 Loom Runtime 诊断 HUD。

---

## 17. 高级功能增补

- **插件热重载**：开发档下重建部分 SubApp / 重跑插件 `build`，配合 ECS 动态组件（§16.2）与资产热重载，改逻辑免重启。
- **多 App 实例**：同进程跑多个 App（如编辑器内嵌 PIE「Play In Editor」沙盒），World/资源隔离。
- **系统单步**：接 ECS §23.4，按 system 粒度单步整帧，调试时序 bug。
- **启动屏/异步装配**：`ready` 门 + 加载态（§11），资产/设备就绪前显示加载屏（对接 Loom Suspense）。
- **崩溃安全**：panic 捕获 → 优雅退出路径（§12），尽量 flush 存档与崩溃转储。
- **特性探测装配**：按 capability 在装配期选择 system 集/子应用拓扑（如无 GPU 常驻能力则走 CPU 提取路径）。

---

## 18. 性能工程

- **装配零运行时开销**：插件/依赖解析只在启动期做；运行期是纯 Schedule 执行，无动态查表。
- **流水线重叠**：§9 让仿真/渲染错帧并行，吞吐接近翻倍。
- **阶段内并行**：阶段 system 走 ECS 冲突图 + fiber 作业图，近线性扩展。
- **extract 最小化**：只提取「可见 + 变化」数据（接 ECS 脏块访问器），成本 ∝ 变化量。
- **帧节奏稳定**：优先稳定 1%low 而非平均 FPS；熔断防死亡螺旋。
- **无头精简**：server 档不建渲染子应用/窗口，纯心跳。

诚实边界：流水线重叠率、frame pacing 稳定性、启动耗时等指标需真实设备 + 大场景实测，均标注 PLANNED。

---

## 19. 易用性与 Bevy 迁移策略

对外 API 刻意贴近 `bevy_app`：

```rust
App::new()
    .add_plugins(PrismDefaultPlugins)
    .insert_state(GameState::MainMenu)
    .add_systems(Startup, setup)
    .add_systems(Update, (input, ui).chain())
    .add_systems(FixedUpdate, (physics, net))
    .run();
```

迁移路径：
1. 提供 `prism_app::prelude` 与 bevy_app 近同名导出（`App/Plugin/PluginGroup/Startup/Update/FixedUpdate/States`）。
2. 把 `prism_bevy`（本就是 Bevy 桥接）整体由 `prism_app` 取代/废弃。
3. 渲染/音频对接层的插件（`prism_render_scene` 等）把 `bevy_app::Plugin` 换 `prism_app::Plugin`，`add_systems` 签名不变，内部换内核。

---

## 20. crate 分层与模块布局

```
pkg/prism_app/
  src/
    app.rs                 # App、PluginsState 装配状态机、run()
    sub_app.rs             # SubApp、SubApps、extract 接线
    plugin.rs              # Plugin trait、PluginRegistry、依赖解析/去重/拓扑排序
    plugin_group.rs        # PluginGroup、enable/disable/reorder
    schedule_order.rs      # Main 阶段序、tick 组、phase 插入
    fixed.rs               # 固定步长累加器、alpha、死亡螺旋熔断
    state.rs               # States/Computed/SubState/状态作用域实体（接 prism_ecs）
    runner/                # winit / headless / dedicated_server / once
    lifecycle.rs           # Suspended/Resumed/LowMemory/AppExit 事件
    pacing.rs              # frame pacing / 帧限 / 低延迟策略
    settings.rs            # 分层配置（接 prism_reflect）
    determinism.rs         # 固定序 / 录制重放 / 回滚编排钩子
    diagnostics.rs         # 阶段火焰图 / 帧统计
  features = ["std","multi_thread","pipelined","winit","headless","determinism","reflect","trace"]
```

依赖：`prism_ecs`、`prism_tasks`、`prism_time`，可选 `prism_diagnostic`/`prism_reflect`。窗口/输入经插件注入，**不**在 Cargo 层硬依赖具体后端。**不碰任何 `bevy_*`。**

---

## 21. 契约、不变量与版本化

- **核心阶段顺序为不变量**：`First → RunFixedMainLoop → PreUpdate → StateTransition → Update → PostUpdate → Last`；插件只能相对插入，不能重排核心序。
- **插件装配状态机单调**：`Adding → Ready → Finished → Cleaned`，不可回退（热重载走显式重建路径）。
- **extract 单向只读**：渲染子应用不得写回 main world。
- **退出优雅性**：`AppExit` 必排空命令、运行 `Last`、逆序 `cleanup`。
- **版本化契约**：`PluginsState`、`SubAppLabel`、`ExtractFn` 签名、`AppExit` 码、tick 组标签集、`Time<Fixed/Virtual/Real>`。

---

## 22. 路线图（M0–M6）与基准即规格

- **M0 外壳**：App + SubApp(单 main) + Plugin + `HeadlessRunner`/`ScheduleRunnerOnce` + Startup/Update 阶段 → 跑通空 World 的帧循环 + 单测。
- **M1 调度序 + 状态**：完整 Main 阶段序、PluginGroup/依赖解析/去重、States/OnEnter/OnExit、Events 更新。
- **M2 固定步长**：累加器 + alpha + 死亡螺旋熔断、`Time<Fixed/Virtual/Real>`、FixedMain tick 组。
- **M3 子应用 + 流水线**：Render SubApp + extract 接线 + `pipelined` 并行（接 ECS §23.5）。
- **M4 窗口 Runner + 生命周期**：`WinitRunner`、Suspended/Resumed/LowMemory/AppExit、frame pacing/帧限。
- **M5 确定性 + 服务器**：固定序/录制重放/回滚编排、`DedicatedServerRunner`、设置分层。
- **M6 迁移/工具**：bevy_app 兼容 prelude；`prism_bevy` 退役；阶段火焰图/帧统计/诊断 HUD；插件热重载 + PIE 多实例。

**基准即规格**：每里程碑以微基准红绿为完成判据——空帧循环开销、固定步长在 jitter 输入下步数正确、流水线重叠率、无头与窗口行为对拍一致、确定性双跑帧哈希一致。核心价值集中在 **M2（确定性心跳）+ M3（流水线/提取）+ M5（确定性/服务器）**。

---

## 23. 诚实边界与风险

- 本文为设计规格，**当前无代码**；M0–M6 均为 PLANNED。
- **高风险项**：
  1. **extract 管线 + 流水线并行（M3）**：跨 World 只读提取 + 跨线程生命周期是正确性要害；须先定义稳定 `ExtractComponent` trait，再上并行；可先串行提取跑通，再开 `pipelined`。这是 `prism_render_scene` 能否干净脱 Bevy 的决定性接缝。
  2. **固定步长熔断（M2）**：死亡螺旋参数（max_frame_dt/max_substeps）需按内容压测标定，错配会在卡顿时放大为慢放或卡死。
  3. **窗口生命周期（M4）**：移动端 Suspended/Resumed 下 GPU 上下文/交换链重建极易崩，需与 RHI/窗口层联调，设备丢失恢复要全链路演练。
  4. **确定性（M5）**：跨平台浮点一致仅在 `determinism` 档 + 定点/软件数学路径保证；录制重放靠逐固定步输入 + 周期快照，快照频率权衡内存与回放精度。
  5. **插件依赖解析（M1）**：循环依赖/缺失依赖须在装配期静态报错，不能拖到运行期；去重语义要明确（显式报错 vs 静默忽略）。
- **与既有文档关系**：本层是 `prism_loom_runtime_framework`（L7 游戏运行时）的**地基**，二者不重叠——Loom 作为插件挂到本层；`prism_bevy` 在 M6 被本层取代。渲染提取与 ECS 文档 §23.5 为同一接缝的两侧表述，须保持契约一致。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 24. AAA 高级功能增补（v0.2）

本章补齐顶级 App/运行时框架常被忽视、却在真实 AAA 项目里缺一不可的能力。均 feature/档位门控，默认不付成本；与前文 App/Plugin/Schedule/主循环/子应用/状态机互补。它是 `bevy_app` 替代，区别于上层 `prism_loom_runtime_framework`（本 crate 是地基，后者在其之上）。

### 24.1 插件依赖图与分组

插件规模化后「谁先初始化」是硬问题：

- 插件声明依赖（`depends_on` / `after` / `before`），框架**拓扑排序**决定初始化序，环依赖提交期报错。
- **插件组**（PluginGroup）：一次加入一组相关插件（如 `DefaultPlugins`），可单独禁用/替换组内成员（对标 Bevy PluginGroup）。
- 特性门控：插件按 feature/平台条件加入（服务器不加渲染插件）。

### 24.2 系统与模块热重载

迭代速度是生产力核心：

- **系统热重载**:gameplay 系统编译进 dylib,运行时检测变更、卸载旧符号、载入新符号、重接 Schedule,保 World 状态不丢(接 `prism_reflect` 做状态迁移)。
- **资产/配置热重载**:接 `prism_asset` 文件监视,改 shader/材质/配置即时生效。
- 热重载仅开发档启用;发行档静态链接零成本。

### 24.3 子应用流水线深化（Sub-App Pipelining）

子应用（SubApp）是独立 World + Schedule，可与主应用流水线并行：

- **渲染子应用**:仿真(主)与渲染(子)流水线并行——第 N 帧渲染与第 N+1 帧仿真同时跑(接 ECS §23.5 提取、`prism_tasks` 车道),吞吐近翻倍。
- **服务器子应用**:单进程内跑「客户端 + 内嵌服务器」(listen server),子应用隔离世界。
- 子应用间经**提取/同步点**单向传数据,不共享可变状态,避免竞争。

### 24.4 运行模式（Headless / 编辑器内嵌 / 专用服务器）

同一 App 骨架支持多形态启动：

| 模式 | 特征 |
|---|---|
| `Client` | 全渲染 + 输入 + 音频 |
| `DedicatedServer` | 无头(无渲染/音频/窗口),仅仿真 + 网络 |
| `EditorEmbedded` | App 跑在编辑器窗口内,受编辑器驱动暂停/单步 |
| `Headless` | CI/测试/批处理,无显示 |

模式决定加载哪些插件组(§24.1)、主循环是否驱动渲染;无头模式零渲染依赖(脱 Bevy 后尤其干净)。

### 24.5 优雅启动 / 关闭生命周期

- **启动阶段**:`PreStartup → Startup → PostStartup`,配合 loading/splash 态(接状态机),资产预热后再进主循环。
- **优雅关闭**:捕获退出请求 → 跑 `OnShutdown` 系统(存档、断网、刷盘、释放 GPU) → 确认退出,避免数据丢失/句柄泄漏。
- 退出可被系统取消(「有未保存更改,确认退出?」)。

### 24.6 控制台变量（cvar）与配置级联

对标 Quake/Source 的 cvar 体系 + 现代配置级联：

- **cvar**:运行时可改的命名变量(`r.shadows 2`),分类(渲染/网络/调试),带默认/范围/权限(作弊保护),控制台/配置文件/命令行可设。
- **配置级联**:默认 < 平台 < 用户 < 命令行 < 运行时,逐层覆盖;经 `prism_reflect` 做类型化读写与校验。
- cvar 变更可触发回调(改分辨率即时重建交换链)。

### 24.7 崩溃隔离与系统恢复

AAA 要求「一个系统崩了不拖垮整个进程」:

- **panic 隔离**:单 system panic 可捕获为错误、隔离该 system(禁用并上报),而非直接终止(可选,开发档默认 fail-fast)。
- **崩溃转储**:panic hook 收集堆栈、最近日志、World 摘要、cvar 快照,写 crash dump 供事后分析。
- **看门狗**:主循环卡死检测(帧超时),导出诊断。

### 24.8 子状态与状态作用域实体

状态机深化（接前文状态机章）：

- **子状态**(SubState):`Menu` 下有 `Main/Settings/Credits`,层级状态机,父状态退出时子状态一并退出。
- **状态作用域实体**:实体标记归属某状态,状态退出时自动 despawn(接 ECS 层级 despawn),杜绝「切场景残留实体」。
- **转换钩子**:`OnEnter/OnExit/OnTransition` 系统集,过渡动画/资源加卸载挂此。

### 24.9 多窗口 / 多世界

- **多窗口**:编辑器多视口、游戏多显示器;每窗口关联渲染目标,主循环统一驱动(接 `prism_window`)。
- **多世界**:主世界 + 编辑器预览世界 + 服务器世界并存,各持独立 ECS World 与时间域(接 time §24.8),经子应用隔离。

### 24.10 诚实边界

本章全部为 PLANNED 设计目标,无代码。**24.1 插件依赖图 + 24.3 子应用流水线**是框架骨架能力,建议随 M0/M1 优先落地;24.4 运行模式(尤其无头服务器)随脱 Bevy 清理渲染耦合时自然成立;24.6 cvar、24.8 子状态随 gameplay/编辑器落地;24.2 热重载、24.7 崩溃隔离是开发期生产力/稳定性增强,随工具链(M6)落地。本 crate 是地基运行时,勿与上层 `prism_loom_runtime_framework` 职责重叠(后者做更高层游戏流程编排)。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码;仅借鉴公开架构形态与经典数值。

---

## 25. 跨 crate 契约对齐（v0.3）

`prism_ecs`(v0.4)/`prism_tasks`/`prism_time`/`prism_reflect`/`prism_transform`(均 v0.2) 升级后，App 作为**总编排者**需显式对齐这些新契约。本章不引入 App 新机制，只固定「App 驱动谁、真相归谁」，与 ECS §24 跨 crate 对齐互为镜像。

### 25.1 主循环编排 time（帧节奏 / VRR / 帧预算 / 挂起恢复）

App 主循环（§7）、固定步（§8）、帧节奏与呈现（§13）是 `prism_time` 新能力的**驱动方**，而非实现方：

- 帧节奏/低延迟提交对齐 time §24.2，VRR 调度对齐 time §24.3——App 向 time 取「下一 present 预估时刻」做提交对齐。
- 固定步累加器实现在 time（§8 与 time §7 是同一机制两侧，单一真相）；App 只负责每帧推进与调用。
- 挂起/恢复（§12）把平台事件转给 time §24.7，由 time 钳制「恢复首帧巨 delta」。
- App 把本帧**帧预算**（time §24.4）下发给 `prism_tasks` 车道（§25.2），超预算则顺延 `Background`。

### 25.2 子应用流水线：ECS 提取为边界，tasks 为执行，transform 双缓冲保一致

App 子应用流水线（§9 / §24.3）的三方契约：

- **边界**：仿真→渲染的数据搬运以 ECS §23.5 提取管线为唯一接缝（渲染子应用只依赖提取 trait）。
- **执行**：子应用并行跑在 `prism_tasks` 线程类分离之上（tasks §24.2），`Critical` 车道（tasks §24.1）保关键路径。
- **读一致**：渲染侧读 `prism_transform` 双缓冲世界矩阵（transform §24.3），避免读到半更新的仿真写。
- 收益：第 N 帧渲染与第 N+1 帧仿真流水线并行，吞吐近翻倍，且三方职责不交叉。

### 25.3 配置 / cvar 经 reflect 做类型化读写与校验

App 配置分层（§14）与 cvar（§24.6）的读写/校验**委托 `prism_reflect`**：

- cvar/配置字段的范围钳制、默认值、改名兼容走 reflect §24.7 特性（`clamp`/`default`/`rename`）。
- 命令行/文件/运行时的级联覆盖经 reflect 类型化反序列化，非法输入按安全边界（reflect §24.8）拒绝而非崩溃。

### 25.4 多世界各持独立 time 时间域

App 多世界（§24.9）与 ECS 多 World（ECS §16）协同时，每个 World 持**独立时间上下文**（time §24.8）：主世界、编辑器预览世界、服务器子应用世界互不干扰（一个暂停不冻结另一个），经子应用隔离。

### 25.5 诚实边界

本章全部为 PLANNED 对齐说明，无代码；不改变 App 已有机制，仅固定编排职责与引用。落地时序：25.1 随 M2 主循环/平台接线；25.2 随 M3 渲染子应用；25.3 随 M1 配置/编辑器；25.4 随多世界/服务器模式落地。App 是地基运行时编排者，更高层游戏流程编排仍归 `prism_loom_runtime_framework`。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

