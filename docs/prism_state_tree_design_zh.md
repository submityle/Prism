# Prism StateTree 次世代通用决策运行时设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的**层级状态图 + 任务承载 + 类型化数据绑定 + 可快照/可回滚/可批量**通用决策运行时。它不服务单一领域：同一执行内核同时承载 AI 决策、玩法流程、交互契约、能力编排、动画状态机与 UI 流程，运行时形态是「**一份编译后的不可变计划（`TreePlan`）被 N 个实例共享**」，目标是**语义正确、成本 ∝ 变化量、帧内零分配、确定性可回放、万级代理可批处理**。
>
> 借形态不抄码。借鉴：
> - **UE5 StateTree**（HFSM + Task/Evaluator/Transition + 数据绑定 + Schema/Context + 可视化调试）——核心基线：状态生命周期清晰、绑定表达力强、资产可数据驱动
> - **Harel 状态图 / W3C SCXML / XState**（正交并行区域、深/浅历史、entry/exit/do、事件队列与 run-to-completion、无事件迁移）——补齐 UE StateTree 缺失的**形式语义**与**并行区域**
> - **UE Behavior Tree / BehaviorTree.CPP / Groot**（`Success/Failure/Running` 三态、reactive fallback、装饰器/服务、黑板、tick 策略、可视化调试）——**只借语义，不设独立系统**：其有用语义全部归约为本设计的激活模式与叶节点三态（§8.1/§17.2）
> - **Unity Behavior / Mecanim Animator**（子图、黑板变量、状态机迁移的 duration / interruption source / priority）——作者化与中断语义
> - **Unity DOTS / UE Mass**（SoA 批处理、按处理器反转迭代、同构代理规模化）——群体批量执行器
> - **flecs / Bevy ECS**（pair 关系、通配/传递、observer/reaction、prefab `IsA`、快照回滚）——宿主集成与增量重算
> - **Utility AI（IAUS / Dave Mark）、GOAP、HTN**——只借「选择策略」与「规划器扩展点」，**不内建规划器**
> - **Temporal / Durable Execution 心智**（确定性重放、checkpoint、失败可续）——长流程存档与回滚语义；**实现复用** `prism_visual_scripting` 的 durable 层，不另起一套
> - **本仓既有资产**：`prism_ecs`（observer/reaction/relation/snapshot 的 rollback+determinism）、`prism_app`（schedule/fixed/state/determinism）、`prism_reflect`（path/schema/dynamic/property_bridge/net_delta）、`prism_tasks`（work-stealing）、`prism_asset`（热重载）、`prism_diagnostic`（budget/hitch/profiler/replay）、`prism_time`（timeline/recording）
>
> 本文聚焦「**语义正确 + 数据导向 + 可作者化 + 可调试 + 可网络化**」的工程路线；不复刻 UE/Unity 的对象模型与源码，只借其**状态语义边界、任务生命周期、绑定心智与调试形态**；核心抽象 `no_std + alloc` 友好，ECS/资产/编辑器/调试层需 `std`；**不依赖任何 `bevy_*` crate**。

- 版本: v0.2（ST0 核心语义内核与 Schema/Context 基础契约同步版）
- 状态: **ST0 CORE SHIPPED / ST1 ECS 最小装配切片已实现 / ST2 数据层基础契约、实例帧布局与反射读取切片已实现 / ST2–ST8 其余能力 PLANNED**（ST1 的共享 Plan、逐实体 runtime 组件、CommandQueue 延迟 spawn、显式 tick/event 路径、确定性 ECS gather、显式 Schedule、opt-in `StateTreePlugin`、payload lifecycle Events、typed ECS Context gather 及 core Schema/Context 基础契约已交付；ST2 已交付 host-neutral binding path/slot/dependency contract、无 unsafe 的 instance frame 布局与可选 `reflect` feature 的编译型 `prism_reflect::AccessPlan` 读写和目标类型校验；反射值到 frame 的自动桥接、zero-config 自动插件、observer marker 迁移桥接等仍按路线图推进，依赖项落地状态见 §2.1 与 §20）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 层级定位: **通用决策运行时，非 AI 专属**。AI 是首要消费者而非所有者。上承 `prism_gameplay_*`（AI/能力/流程/交互）、动画与 UI 流程；内嵌 `prism_ecs` 调度，**不进内核仿真内循环**（ECS 存储遍历、物理 solver、渲染 RG 永远走原生路径）
- 关键依赖（均为 `pkg/` 下的 prism 原生 crate，落地状态见 §2.1）: `prism_ecs`（observer/reaction/relation/snapshot：迁移事件的即时响应与回滚底座）、`prism_app`（Plugin/Schedule/`fixed`/`determinism`/`state`：相位与固定步）、`prism_reflect`（`path`/`schema`/`dynamic`/`property_bridge`：动态绑定路径、资产 schema、版本迁移）、`prism_tasks`（off-thread 编译、批量并行执行）、`prism_asset`（树资产加载/mtime 监视/热重载）、`prism_diagnostic`（budget/hitch/profiler/replay：调试器与帧预算）、`prism_time`（timeline/recording/确定性时间源）、`prism_ui*` + `prism_editor_framework`（图编辑器宿主与调试 HUD）
- 可选依赖: `prism_visual_scripting`（图 IR / durable / 画布，feature `graph`；**未就绪，属前置待建**）、`prism_gameplay_tags`（GameplayTag 门控，AI/交互域使用；**未就绪**）、`prism_loom`（响应式绑定内核，feature `reactive`；**未就绪**）
- 相关文档: `prism_gameplay_design_zh.md`（§14 AI 框架（黑板/感知）、§29.1 StateTree、§29.2 Smart Objects、§30 交互、§22 Crate 拆分）、`prism_ecs_design_zh.md`（observer/reaction/relation/snapshot）、`prism_app_design_zh.md`（schedule/state/fixed/determinism）、`prism_reflect_design_zh.md`（path/schema/迁移）、`prism_tasks_design_zh.md`、`prism_asset_design_zh.md`、`prism_diagnostic_design_zh.md`、`prism_visual_scripting_design_zh.md`（图作者化 / durable / LLM 作者化）、`prism_animation_engine_design_zh.md`（动画状态机消费者）、`prism_loom_runtime_framework_design_zh.md`（§7 响应式绑定）、`prism_network_design_zh.md`（复制/预测/回滚）、`directory_structure_rules.md`（§5.1 文档结构 / §12 八维评估为本文强制规范）
- 明确约束:
  - 语义内核（定义模型 / 编译 / `TreePlan` / 游标运行时 / 确定性 / 快照）`no_std + alloc` 友好；ECS/资产/编辑器/调试需 `std`。
  - **守卫纯函数铁律**：迁移守卫与选择策略只读上下文、零副作用；一切写操作经延迟命令队列。
  - **目标：热路径零分配、无锁、无 panic**：这是完整运行时与基准门槛，不是当前 ST0 已兑现保证。当前 ST0 的配置切换/日志仍会分配；调用者提供的 guard/callback 也可自行 panic。实现进度与实测边界见 §20。
  - **默认零成本**：未启用任何 `prism_state_tree*` crate 时，引擎侧无决策运行时开销。
  - **不做规划器**（GOAP/HTN 只留扩展点）与**通用工作流引擎**（后者是 `prism_visual_scripting` durable 的职责）。

---

## 目录

1. 设计哲学
2. 现状基线与差距
3. 顶级产品对标：取什么、不抄什么
4. 分层架构
5. 语义内核：状态图
6. Task：状态内的行为承载
7. 数据层：Schema / Context / Binding / Evaluator / InstanceData
8. 选择与抢占：候选 · 触发 · 重入
9. 执行引擎
10. 快照、回滚与存档
11. 网络：权威、预测与版本一致性
12. 确定性
13. 性能工程
14. 工具链：作者化、静态校验、调试
15. 一致性验证：语义测试套件
16. 安全与沙箱
17. 与既有子系统的装配
18. Crate 拆分与 feature 矩阵
19. 公共 API 草案
20. 路线图（ST0–ST8）
21. 关键扩展点、风险与非目标
22. 多维质量评估（§12 八维）
23. 术语表
24. 与既有设计文档的关系与偏差
25. 高级功能

---

## 1. 设计哲学

### 1.1 五条铁律

1. **语义先行**：层级、并行、历史、微步这些语义用 Harel 状态图 / SCXML 的形式定义收敛，而不是「各家的 FSM 各说各话」。语义有歧义的地方一律取状态图规范解释，并可被一致性测试套件（§15）证伪。
2. **数据导向**：定义不可变且共享（`Arc<TreePlan>`）；实例状态最小化、布局在编译期确定；批量路径走 SoA；快照即一次 `memcpy` 加游标。
3. **类型优先、动态兜底**：Rust 侧绑定编译期校验；作者层走 `prism_reflect` 路径，但**在编译期解析为 slot 索引**，热路径不出现字符串查找。
4. **成本 ∝ 变化量**：事件驱动唤醒；Evaluator 声明依赖并只在其变化时重算；无事件、无定时器、无变化时状态不 tick。
5. **可观测即交付物**：迁移因果、断点、录制重放、静态校验器与语义测试套件，和运行时同属交付范围，不是事后附加。

### 1.2 一句话形态

```
TreeDef(文本/图资产) ──compile(off-thread)──> TreePlan(不可变·扁平·带版本)
                                                  │ Arc 共享
                    ┌─────────────────────────────┼─────────────────────────────┐
                    ▼                             ▼                             ▼
             Instance(游标运行时)          Instance(游标运行时)          BatchRuntime(SoA 批量)
             复杂个体 AI / 流程            能力编排 / UI 流程            万级同构代理
```

### 1.3 非目标（明确不做）

- 不做 GOAP/HTN 规划器；不做神经网络/ML 行为。
- 不做通用持久化工作流引擎（长流程/审批/事件溯源归 `prism_visual_scripting` durable）。
- 不在热路径解释图；图只是**编辑期视图**，运行期只有 `TreePlan`。
- 不引入独立行为树系统——其有用语义由 §8 的「候选 / 触发 / 重入」三属性 + §6.1 三态覆盖（§17.2 说明）；不替代 `prism_app::state`（后者是全局相位，粒度不同）。

---

## 2. 现状基线与差距

### 2.1 引擎内现状（依赖可用性）

| 依赖能力 | 提供方 | 现状 | 依据 |
|---|---|---|---|
| 即时事件 / 变更反应 | `prism_ecs` | 已有代码 | `src/observer.rs`、`src/reaction.rs` |
| 关系模型（pair/通配/传递） | `prism_ecs` | 已有代码 | `src/relation.rs`、`src/world/relations.rs` |
| 世界快照 / 回滚 / 确定性 | `prism_ecs` | 已有代码 | `src/world/snapshot/{capture,delta,rollback,determinism,hash}.rs` |
| 调度 / 固定步 / 全局状态 / 确定性开关 | `prism_app` | 已有代码 | `src/schedule.rs`、`fixed.rs`、`state/`、`determinism.rs` |
| 属性路径 / schema / 动态反射 / 桥接 | `prism_reflect` | 已有代码 | `src/path.rs`、`schema/`、`dynamic/`、`property_bridge/` |
| 作业系统（并行/主线程亲和） | `prism_tasks` | 已有 crate | `pkg/prism_tasks` |
| 资产加载 / mtime / 热重载 | `prism_asset` | 已有 crate | `pkg/prism_asset` |
| 帧预算 / hitch / profiler / replay | `prism_diagnostic` | 已有 crate | `pkg/prism_diagnostic` |
| 时间源 / timeline / 录制 | `prism_time` | 已有 crate | `pkg/prism_time` |
| 编辑器 UI 底座 | `prism_ui*`、`prism_editor_framework` | 已有 crate | `pkg/prism_ui*`、`docs/prism_editor_framework_design_zh.md` |
| 图 IR / durable / 画布 | `prism_visual_scripting` | **未就绪**（PLANNED） | 仅设计文档，无 `pkg/prism_visual_script` |
| 响应式绑定内核 | `prism_loom` | **未就绪**（PLANNED） | 仅设计文档，无 `pkg/prism_loom` |
| GameplayTag 门控 | `prism_gameplay_tags` | **未就绪**（PLANNED） | 仅设计文档 |
| 决策运行时本体 | `prism_state_tree_core` + `prism_state_tree_ecs` | **ST0 CORE SHIPPED**；ST1 最小装配切片与 ST2 binding 基础契约/反射读取切片已实现（显式 Schedule、opt-in `StateTreePlugin`、typed ECS Context、payload Events、host-neutral binding path/slot/dependency plan、可选 `reflect` feature 的编译型 `AccessPlan` 与目标类型校验已交付；instance frame/写入器/zero-config/observer marker 仍 PLANNED）；ST2–ST8 其余能力 PLANNED | `pkg/prism_state_tree_core`、`pkg/prism_state_tree_ecs`；见 §20 ST0/ST1/ST2 实现进度 |

> **诚实边界**：上表「已有代码」指模块存在，其完整能力以各 crate 自身文档与测试为准；StateTree 的 ST0 语义内核与 ST1 最小逐实体 ECS 装配已落地，不等于完整决策产品；`prism_visual_scripting` / `prism_loom` / `prism_gameplay_tags` 三项未就绪，对应 feature 在落地前为**前置待建**，不阻塞 ST0–ST4 核心（它们只影响 `graph` / `reactive` / AI 域 `GameplayTag` 门控）。

### 2.2 为什么不复用现成方案

| 候选 | 能提供 | 不满足 | 结论 |
|---|---|---|---|
| `prism_app::state` / `bevy_state` | 全局相位切换、`OnEnter/OnExit` | 非每实体、无层级/并行/绑定、无实例数据 | 用作**宿主相位**，不做决策 |
| Rust `sm` / `statig` | 代码生成式 FSM | 无资产、无层级、无绑定、无并行区域、无调试/快照/确定性 | 不采用 |
| 通用行为树 crate | BT 三态、黑板 | 无状态生命周期、无 Schema/绑定、无确定性回滚、无静态校验 | 不采用（语义并入 §8 的候选/触发/重入机制与 §6.1 三态） |
| UE5 StateTree 本身 | 完整产品形态 | 闭源 + UObject/反射耦合；单活跃路径（正交并行需自行模拟）；Evaluator 默认逐帧求值易成热点 | **借形态，重实现** |

### 2.3 差距清单

- 缺**通用决策运行时**：现有 `prism_app::state` 粒度太粗；没有任何每实体状态生命周期承载。
- 缺**类型化数据绑定**：从组件/资源到行为的绑定目前每个系统手写。
- 缺**可快照的决策状态**：GAS/动画/AI 的状态无法统一进 rollback 快照环。
- 缺**作者化与校验**：无资产格式、无静态校验、无调试器、无语义一致性测试。
- 缺**规模化路径**：无同构代理的批处理形态。

---

## 3. 顶级产品对标：取什么、不抄什么

### 3.1 核心语义对标

| 对标系统 | 取什么 | 不抄什么 | 落点 |
|---|---|---|---|
| UE5 StateTree | State/Task/Evaluator/Transition 心智；Schema+Context 数据契约；绑定表达力；共享定义+轻量实例；调试器形态 | UObject/反射运行时；全帧 Evaluator 求值；单活跃路径限制；编辑器即真相源 | §5–§9 |
| Harel 状态图 / SCXML / XState | 正交并行区域；深/浅历史伪状态；entry/exit/do；事件队列 + run-to-completion；无事件（eventless）迁移；LCA 迁移算法 | SCXML 的 XML 语法与 `<script>` 内联脚本；解释器式实现 | §5 |
| BehaviorTree.CPP / UE BT | `Success/Failure/Running` 三态；reactive fallback（条件重判即可抢占）；装饰器/服务分层 | XML 作者格式；每节点虚调用 + 每帧全树遍历；**不引入独立 BT 运行时** | §6 / §8 |
| Unity Behavior / Mecanim | 子图引用与参数覆盖；迁移 duration / interruption source / priority；编辑器节点搜索与类型安全端口 | GameObject/序列化体系；Animator 的运行期专有结构 | §5.2、§13.1、§17.5 |
| Unity DOTS / UE Mass | SoA 实例列；按节点反转迭代；同构代理的 LOD 分档 | 强制全体批处理（复杂个体仍需游标）；ECS 组件粒度爆炸 | §9.3 |
| flecs / Bevy ECS | pair 关系、通配、传递、observer/reaction 的即时响应；prefab `IsA` 复用 | 关系语义的隐式副作用；过度深层关系图 | §7.1、§17.1 |
| Utility AI（IAUS） | 打分式选择：归一化 consideration + 权重 + 响应曲线 | 把 Utility 当唯一决策范式；隐藏式全局打分 | §8 |
| Temporal / Durable | 确定性重放、checkpoint、失败可续、版本迁移 | 分布式编排器、网络服务、Saga 全量语义 | §10、§11 |
| 本仓 `prism_reflect` | 路径寻址、schema 版本化迁移、net_delta | 运行期字符串解析进热路径 | §7.2 |
| 本仓 `prism_ecs` reaction/observer | 变更驱动重算、事件即时响应 | 用反应图替代显式相位 | §7.3、§9.4 |

### 3.2 深化对标（§25 高级功能来源）

> 首轮对标注重「语义」（§3.1）；本节补注「规模化、工具链与验证」这一类工程实践的来源。

| 借鉴对象 | 借鉴点 | 落点 |
|---|---|---|
| Source 引擎 NPC 的 schedule + condition + interrupt | 用「可中断的日程 + 优先级 + 中断条件」统一调度——**一套机制覆盖 BT/FSM** 的最早工业实践之一 | §8.1、§25.2 |
| Halo 2 / Halo 3 行为树（Damian Isla 的 GDC 分享） | BT 工业化的起点：优先级、可中断、可读性 | §8.1 |
| F.E.A.R. GOAP（Jeff Orkin） | 把「规划」当作选择的一种，而不是独立运行时 | §8.2、§21 |
| Killzone 2/3 HTN（Guerrilla） | 层级任务网络，且面向策划可编辑 | §21 |
| Left 4 Dead AI Director | 行为层之上的全局节奏/强度调节器 | §25.5、§25.3 |
| Unreal Mass / Zone Graph / Crowd | LOD 分档、批处理、群体与 Smart Object 协同 | §9.3、§25.1、§25.3 |
| Total War 式编队 AI | 编队级决策向个体下派命令，个体可拒绝 | §25.3 |
| 动作游戏取消窗口 / Unreal AnimBP `Interruption Source` | 「可被谁打断」的显式声明 | §25.2 |
| XState `@xstate/graph` | 图探索：枚举路径、找不可达/死锁、生成测试用例 | §25.4 |
| SPIN / TLA+ / UPPAAL | 有界模型检验：死锁、活锁、守卫矛盾 | §25.4 |
| rr / 反向调试、rollback netcode 重放 | 反事实调试：回放并改写某次决策看分叉 | §25.8 |
| Unreal Game Features / 版本化内容包 | 行为包的可插拔与语义版本契约、灰度与回滚 | §25.6 |
| 服务治理的熔断/舱壁 | 坏树隔离与预算降级 | §25.7 |
| UE EQS / 影响图（influence map） | 空间战术查询作为一等数据源 | §25.9 |
| `big_brain`（Rust Utility AI） | thinker/utility 的频率分层 | §8.2、§25.1 |
| 本仓 `prism_diagnostic` / `prism_time` | 遥测、预算、录制回放、时间旅行 | §25.5、§25.8 |

---

## 4. 分层架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L4 工具与作者层 (Authoring & Tooling)                                        │
│   文本资产(RON/schema) = 真相源   ·   图视图(prism_visual_scripting 画布)      │
│   静态校验器(错误码) · StateTree Debugger · 迁移因果视图 · 录制/重放 · LLM 作者化 │
│      │ 产出：TreeDef（可 diff / 可 merge / 可评审）                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ L3 编译层 (Compile) —— off-thread，prism_tasks                                │
│   validate   静态校验（可达性/守卫恒假/槽位未绑/并行写冲突/无界实例）            │
│   lower      TreeDef → TreePlan（扁平状态表/迁移表/守卫表/槽位表/实例帧布局）    │
│   optimize   绑定槽位化 · 常量折叠 · 死状态消除 · 批处理可提升性标注 · 版本签名   │
│      │ 产出：TreePlan（不可变 · 可缓存 · 可签名 · 可热替换）                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ L2 运行层 (Runtime)                                                          │
│   CursorRuntime   每实例游标（活跃路径 + 每 region 游标 + 实例帧 blob）          │
│   BatchRuntime    按节点反转迭代，SoA 实例列，prism_tasks 并行（feature batch）  │
│   Scheduler       相位挂接(prism_app) · 固定步 · AccessSet 并行冲突消解          │
│   CommandQueue    延迟写（结构性变更）· 事件环（定容，零分配）                    │
│   Snapshot        实例帧 + 游标 + 待处理事件 + RNG → 可序列化快照                │
├──────────────────────────────────────────────────────────────────────────────┤
│ L1 宿主集成层 (Host Integration)                                             │
│   Schema/Context  ← prism_ecs（Entity/查询/事件/关系/prefab）                  │
│   BindingResolver ← prism_reflect（路径 → slot 索引，编译期解析）               │
│   TaskRegistry    ← 原生 Task / WASM Task(prism_script) / 图 Task              │
│   Replication     ← prism_network（策略标注、预测、回滚）                       │
│   观测/预算       ← prism_diagnostic / prism_time                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

分层铁律：

- **L4→L3 单向**：作者层只产出 `TreeDef`，不碰运行期；编译一律 off-thread，编译失败不影响已运行计划。
- **L3 对后端透明**：编译器只认 Task 契约与 `AccessSet`，不认 Task 来自原生/WASM/图。
- **L2 不进内核内循环**：运行时跑在 gameplay/服务相位；触碰世界一律经 L1 的受控边界与 `CommandQueue`。
- **L1 是唯一引擎接缝**：宿主 API 经 `prism_reflect` 与 Schema 暴露，不为每个 API 手写绑定。

---

## 5. 语义内核：状态图

> 本节定义「什么算正确」。所有语义可被 §15 一致性套件验证。

### 5.1 状态类型

| 类型 | 语义 | 用途 |
|---|---|---|
| `Atomic` | 叶子状态，携带 Task 列表 | 具体行为段 |
| `Compound` | 有子状态，**恰好一个**活跃子 | 模式切换（Idle/Combat/Dead） |
| `Parallel` | 有多个 **region**，**全部**活跃 | 正交关注点（移动 ∥ 战斗 ∥ 表情 ∥ 语音） |
| `History`（伪状态） | 记录子状态进入点，`Shallow` / `Deep` | 被打断后恢复原分支（UE StateTree 缺失） |

`Compound` 与 `Parallel` 可互相嵌套：`Combat`（compound）的某个子状态可以是 `Parallel`，其 region 内又是 compound。**禁止混合**——一个状态要么 compound 要么 parallel，避免「活跃路径」定义歧义。

**活跃配置（active configuration）** = 当前所有活跃状态的集合。游标运行时必须能枚举它（`active_states()`），作为调试、绑定与网络同步的真相。

### 5.2 迁移

```rust
pub struct TransitionDef {
    pub source: StateId,                    // 可挂在祖先上：子状态自动继承
    pub target: TransitionTarget,           // State(StateId) | History(StateId, Shallow|Deep) | Self
    pub trigger: Trigger,                   // Event(&'static str) | Eventless | OnTimer | OnTaskDone(TaskId) | OnSignal
    pub guard: GuardRef,                    // 纯函数：&GuardCtx -> bool
    pub priority: u16,                      // 越小越优先；同优先级按声明序
    pub kind: TransitionKind,               // External(退出并重入) | Internal(目标已活跃则不动)
    pub interrupt: InterruptPolicy,         // 中断语义（§6.3）
    pub actions: &'static [ActionRef],      // on_transition 副作用（经 CommandQueue）
}
```

取定规则（**确定性铁律**）：

1. 候选集 = 活跃配置中所有状态及其**祖先链**上、trigger 匹配且 guard 为真的迁移。
2. 排序键：`(priority ↑, 声明序 ↑, 源状态深度 ↓)`；取第一条，**一个微步只取一条**。
3. `Eventless` 迁移**不逐帧轮询**：只在「事件排空后 / 定时器到期 / 其守卫声明的依赖变化」时重判（cadence 可显式覆盖，§7.3/§8.1）。同一微步内可级联，但有**最大级联深度**（编译期校验 + 运行期上限），防止震荡。
4. 迁移路径按 **LCA（最近公共祖先）** 计算：

```
exit  : 自 source 向上退到 LCA 的子节点（逐层 on_exit，自深至浅）
enter : 自 LCA 向下进到 target（逐层 on_enter，自浅至深）
```

5. `TransitionKind`：`External` 走完整 exit → enter（活跃子任务按 §6.3 `InterruptPolicy` 中断）；`Internal` **不退出、不重入**，只执行 `actions`——用于「目标已活跃则什么都不做」，这正是 BT「不重启正在运行的子节点」的精确对应。
6. `Self` 迁移 = `External` 且 target == source（退出并重入），用于「重跑当前段」；`Internal` + `Self` 即「只跑 actions、不重跑行为」。
7. 进入 `Compound` 后**立即求值一次 `Eventless` 迁移**（SCXML 语义）——这就是「初始子状态的选择」；若一条都不成立，必须有声明式兜底（`default` 子状态或失败迁移，校验 `ST0013`）。Event 迁移完成后触发的 eventless 级联必须在 `current_event = None` 的上下文中执行；事件只对正在选择/应用的那一条迁移可见，不能泄漏到后续无事件 guard，否则会破坏可重放性。
8. 迁移期间**不允许**在状态树上做结构性改动；所有写操作进 `CommandQueue`，微步结束后统一 flush。

### 5.3 历史伪状态

- `Shallow History`：仅记住 LCA 直接子层的进入点。
- `Deep History`：记住完整活跃配置。
- 用途：`受击 → 硬直 → 恢复` 时回到原来的 `Patrol/Chase/Cover` 分支，而不是回到 `Combat` 的初始子状态。
- 未经访问的 history 回退到编译期声明的 `default` 子状态（不得为「未定义行为」）。

### 5.4 微步与事件队列（run-to-completion）

```
loop {
    drain_events();                  // 定容事件环 → 分类到各实例
    step_transitions();              // 选一条迁移（§5.2）
    if no_transition { break }
    exit_states(); enter_states();   // 触发 on_exit / on_enter / on_transition
    flush_command_queue();           // 结构性写在此统一发生
}
```

- 事件在微步内**不插队**：本微步产生的迁移不消费本微步新事件，保证可重放。
- 队列**定容**（编译期上限 + 溢出显式错误），热路径零分配。
- 溢出策略可配：`DropOldest`（表现类）/ `Error`（权威类，默认）。

### 5.5 语义样例

```ron
TreeDef(
  schema: AiSchema,
  params: { alert_radius: 800.0, combat_history: Deep },
  root: Compound(
    initial: "Idle",
    states: [
      State(
        name: "Idle",
        tasks: [ PlayMontage("Idle"), LookAround(interval: 3.0) ],
        transitions: [
          Transition(target: "Combat", trigger: Event("Perception.Seen"), guard: "target_in_range", priority: 0),
        ],
      ),
      State(
        name: "Combat",
        history: Deep,                          // 被打断后可回到原分支
        on_enter: [ SetBlackboard("mode", "combat") ],
        states: [ Patrol, Chase, Attack, Cover ],   // compound：选一
        transitions: [
          Transition(target: "Idle", trigger: Eventless, guard: "!has_target", priority: 0),
          Transition(target: "Dead", trigger: Event("Health.Zero"), priority: 0, interrupt: All),
        ],
      ),
      State(name: "Dead", tasks: [ Ragdoll ], transitions: []),
    ],
  ),
)
```

语义保证（可被一致性测试断言）：`Combat` 期间收到 `Health.Zero` 时，无论 `Combat` 内在哪个子状态、`Parallel` region 有多少个，`Combat` 子树整体退出（`on_exit` 深度序）后进入 `Dead`，且 `Combat` 的深历史被更新，供复活后 `Transition(target: History(Combat), trigger: Event("Revive"))` 精确恢复。

---

## 6. Task：状态内的行为承载

> **命名与歧义消解**：本节的 Task 对标 UE StateTree Task / BT Task，指「状态内的行为单元」。它与 `prism_tasks`（work-stealing 作业系统）**无关**。为避免混淆，代码中一律 `use prism_state_tree::Task;`；需要跨引用时写 `prism_tasks::Task` 显式限定。`prism_state_tree_core` **不依赖** `prism_tasks`（后者只在 `batch` / off-thread 编译特性下引入）。

### 6.1 状态与生命周期

```rust
pub enum Outcome { Success, Failure, Running }

pub trait Task: Send + Sync + 'static {
    /// 每实例每状态一份；编译期确定布局，必须可快照、无堆指针。
    type Instance: Default + Snapshot;

    fn on_enter(&self, i: &mut Self::Instance, cx: &mut TaskCtx<'_>) -> Outcome;
    fn on_tick (&self, i: &mut Self::Instance, cx: &mut TaskCtx<'_>) -> Outcome { Outcome::Running }
    fn on_exit (&self, i: &mut Self::Instance, cx: &mut TaskCtx<'_>);
    fn on_abort(&self, i: &mut Self::Instance, reason: AbortReason, cx: &mut TaskCtx<'_>) { self.on_exit(i, cx) }

    fn cadence(&self) -> Cadence { Cadence::Never }             // 声明 tick 频率，默认不 tick
    fn access (&self) -> AccessSet { AccessSet::READ_WORLD }    // 并行调度冲突声明
    fn replay (&self) -> ReplaySafety { ReplaySafety::ReplaySafe }
}
```

- `Cadence`: `Never` / `EveryFrame` / `EveryNFrames(u8)` / `OnChange(DepSet)` / `OnEvent(&'static str)`。**默认不 tick**——这是与 UE「Evaluator 全帧求值」的关键性能取舍。
- 一个状态的 Task 列表按序执行：遇 `Failure` 中止后续；全 `Success` 视为该状态「行为完成」（可由 `OnTaskDone` 迁移消费）；出现 `Running` 表示持续。
- 同一状态内**禁止两个 latent Task 同时写同一绑定槽**（编译期校验，§14.2）。

### 6.2 状态请求而非直接改状态

Task 不直接翻转状态，而是声明请求：

```rust
cx.request(RequestKind::Transition("Combat"));
cx.request(RequestKind::PushSubtree(Handle<TreeAsset>, params));   // 压入子流程
cx.request(RequestKind::Complete);                                  // 本段完成
```

请求进 `CommandQueue`，在微步末由语义内核统一裁决（可被守卫否决、可被更高优先级迁移抢占）。这样「行为」与「转移裁决」解耦，避免「Task 完成」与「过渡条件」互相打架的常见困惑。

### 6.3 中断策略

| `InterruptPolicy` | 语义 | 典型用法 |
|---|---|---|
| `None` | 已在执行的任务不可被同优先级打断 | 不可中断的施法/动画 |
| `Self` | 仅同级替换触发 | 移动模式切换 |
| `Child` | 子状态迁移不打断 Task | 移动 ∥ 战斗共存 |
| `Region` | 仅打断同 region | 并行区域局部打断 |
| `All` | 强制打断并走 `on_abort` | 死亡/眩晕/受击硬直 |

`All` 必须可被 `on_abort` 处理，且 `on_abort` 之后不再收到 `on_tick`（运行时断言，测试可验证）。

---

## 7. 数据层：Schema / Context / Binding / Evaluator / InstanceData

### 7.1 Schema 与 Context

Schema 定义「这棵树在什么宿主上运行、上下文是谁」，等价于 UE 的 Context/ExternalData，但**类型化**：

```rust
pub trait Schema: Send + Sync + 'static {
    type Context<'w>: ContextRef<'w>;   // 借用宿主世界的上下文视图
    fn gather<'w>(world: &'w World, owner: Entity) -> Result<Self::Context<'w>, SchemaError>;
}

/// AI 域（对标 UE StateTree AI 形态）
pub struct AiContext<'w> { pub controller: Entity, pub pawn: Entity, pub blackboard: BlackboardRef<'w> }

/// Actor 域
pub struct ActorContext<'w> { pub actor: Entity }

/// 流程域（GameMode/GameState 驱动的长流程）
pub struct FlowContext<'w> { pub game_state: Entity, pub players: &'w [Entity] }
```

Schema 决定：Tag 门控是否生效、复制策略默认值、允许的 Task 集合（编译期白名单）、调试面板字段。

### 7.2 绑定：静态优先，动态兜底

| 模式 | 写法 | 校验时机 | 热路径成本 |
|---|---|---|---|
| 静态（Rust） | `bind(ctx.target.position)` | 编译期 | 直接字段访问 |
| 动态（作者层） | `path: "Context.Target.Position"`（`prism_reflect`） | **加载/编译期**解析为 slot 索引 | 与静态同构：slot 读写 |
| 事件/信号 | `OnSignal("Alarm.Raised")` | 编译期注册表校验 | 定容队列匹配 |

**铁律**：任何 `path` 字符串都不得进入帧循环。编译器把路径解析为 `(SlotIndex, TypeTag, CopyOp)`，运行期只做定长拷贝/引用；未解析成功即编译失败（错误码见 §14.2）。

### 7.3 Evaluator 作为反应式提供者

Evaluator = 全局（每实例一份，或跨实例共享）的只读数据提供者，供 Task/Guard 绑定。

```rust
pub trait Evaluator: Send + Sync + 'static {
    type Instance: Default + Snapshot;
    fn deps(&self) -> DepSet;                     // 声明依赖：组件/tracked resource/事件
    fn cadence(&self) -> Cadence;                 // 默认 OnChange(deps)
    fn update(&self, i: &mut Self::Instance, cx: &mut EvalCtx<'_>);
}
```

与 UE 的关键差异：

- **依赖声明式**：`deps()` 命中 `prism_ecs` 变更检测/reaction，只在依赖变化时重算（成本 ∝ 变化量）。
- **cadence 可退档**：远距离/低 LOD 代理降为 `EveryNFrames`，与 Mass 式 LOD 对齐。
- **纯读**：Evaluator 不得写世界；需要写就回到 Task。

### 7.4 InstanceData：编译期帧布局

- 每个状态的所有 Task/Evaluator 的 `Instance` 需求，在编译期汇总为一张**实例帧布局表**（偏移 + 大小 + 对齐）。
- 运行期实例状态 = 一段连续 blob（`Vec<u8>` / 定容 arena），游标 + blob 即完整状态。
- 好处：快照 = 一次 `memcpy`（§10）；SoA 友好（同构代理的同槽位天然连续，可批量 SIMD）；无每状态堆分配。
- `Snapshot` derive 保证 blob 内不含指针（编译期 lint：`Instance` 类型不得含 `Box/Rc/Arc/&/String`）。

---

## 8. 选择与抢占：候选 · 触发 · 重入

> 本节回答「为什么不设 `ActivationMode` 这类模式开关」：子节点的**候选**（选谁）、**触发**（何时重判）、**重入**（目标已活跃时是否重启）是三个正交属性，分别由守卫、`Trigger`、`TransitionKind` 承担（§5.2）。IR 里不需要再多一个「模式」概念——那只会把同一套语义重新切成两个子系统。

### 8.1 三属性覆盖两种写法

| 属性 | 由什么承担 | 状态机写法 | BT selector / reactive fallback 写法 |
|---|---|---|---|
| 候选 | 守卫 / 评分（§8.2） | 每个子状态一条迁移 | 同左 |
| 触发 | `Trigger`（§5.2） | `Event` / `OnTimer` / `OnTaskDone` / `OnSignal` | `Eventless`（唤醒即重判） |
| 重入 | `TransitionKind`（§5.2） | `External`（真正切换） | `Internal`（已活跃则不动） |

- 抢占顺序 = `priority` + 声明序（§5.2），两种写法共用同一套裁决规则。
- 于是「BT」= `Compound` + 有序候选 + `Eventless` + `Internal`；「状态机」= 候选 + `Event` + `External`。**同一套机制、两种作者写法**，可在同一棵树里逐状态混用。
- 中断不再另设机制：`External` 迁移按 §6.3 `InterruptPolicy` 中断活跃任务，`Internal` 迁移**不打断**任务（正是「不重启正在运行的子节点」）。
- 混用示例：`Combat` 用 `Event` + `External`（有生命周期、要进战吼动画、要占资源），其内部出招选择用 `Eventless` + `Internal`（随时被闪避抢占，改编排但不退出战斗模式）。
- 性能不靠模式而靠触发：`Eventless` 默认 `OnChange(deps)`（依赖驱动，非常量重走）；显式 `EveryFrame` 是**可见且可被校验器标红**的选择（等价传统 BT 的每帧重判）。

### 8.2 候选：守卫与评分

- **有序候选**：按声明序，首个守卫为真者胜（= 传统 BT / UE StateTree 的顺序选择）。
- **评分候选**：由 Evaluator 计算（归一化 consideration + 权重 + 响应曲线），产出互斥的选中判据（= Utility AI / `Weighted`）；评分必须纯函数且确定性，可批处理（§9.3）。
- **记住上次**：历史伪状态（§5.3）。
- **兜底**：`Compound` 必须声明 `default` 子状态或失败迁移，禁止「选择失败即未定义」（校验 `ST0013`）。

```rust
/// 自定义评分仍是扩展点，但只产出「谁被选中」的判据，不引入运行期模式。
pub trait SelectionPolicy: Send + Sync {
    fn score(&self, cands: &[StateId], cx: &GuardCtx<'_>, rng: &mut Rng) -> Selection;
}
```

### 8.3 作者层糖（编辑期展开，非运行期模式）

编辑器/文本层可提供 `Selector` / `ReactiveSelector` / `UtilitySelector` / `Repeat(n)` / `Cooldown(t)` 等宏，**编辑期展开**为 §8.1 的三属性组合 + §6.1 三态 + 实例数据；展开后 IR 与运行语义不新增任何概念（对齐 §4 的 L3 单向与 §14.1 文本真相源）。

**不做**：内建 GOAP/HTN 规划器。留 `Planner` 扩展点（§21），由游戏层以评分 + 树结构自行逼近。

---

## 9. 执行引擎

### 9.1 `TreePlan` 编译产物

```rust
pub struct TreePlan {
    pub rev: PlanRev,                        // 内容哈希：网络一致性 + 热重载比对（§11）
    pub states: Box<[StateNode]>,            // 扁平：kind/父子/region/初始子/history
    pub transitions: Box<[TransitionNode]>,  // guard 索引 + 目标 + 优先级 + interrupt
    pub guards: Box<[GuardProgram]>,         // 编译后的守卫（含绑定 slot 引用）
    pub slots: Box<[SlotDesc]>,              // 绑定槽（来源 + 类型 + 拷贝语义）
    pub tasks: Box<[TaskRef]>,               // Arc<dyn Task> + 参数
    pub eval: Box<[EvaluatorRef]>,
    pub instance_layout: FrameLayout,        // 实例帧偏移表
    pub caps: Capabilities,                  // 最大级联深度、事件环容量、并行度提示
}
```

不可变、`Send + Sync`、`Arc` 共享；编译期完成绑定槽位化与批处理可提升性标注。

### 9.2 `CursorRuntime`（默认）

- 每实例：活跃路径（`SmallVec<StateId>`）+ 每 region 游标 + 实例帧 blob + 定时器表 + 待处理事件。
- 只在**有事件 / 定时器到期 / `OnChange` 依赖变化**时进入微步；否则该实例完全静默。
- 复杂个体 AI、能力编排、流程/UI 用此路径，语义完备（含并行区域与历史）。

### 9.3 `BatchRuntime`（feature `batch`，群体）

反转迭代：以**节点**为主循环，遍历处于该节点的实例集。

```
for node in plan.batchable_nodes() {          // 编译期标注：纯读 + 无实例私有跨节点状态
    let slice = agents_in(node);              // SoA 实例列（同槽位连续）
    par_for(slice, |chunk| node.tick_batch(chunk));  // prism_tasks 并行
}
```

- **节点级批处理提升**：只有被标注为 `Batchable` 的状态/任务进批路径；复杂个体仍走 `CursorRuntime`。同一棵树内两种路径共存，按实例 LOD 切换（档位、迟滞与摊销见 §25.1）。
- 切换 LOD 时用实例帧 blob 迁移（同布局），不重建。
- 目标场景：万级同构代理（人群、兽群、弹幕、粒子式单位）。

### 9.4 调度与相位

- 默认挂 `Update`；确定性玩法挂 `FixedUpdate`（对齐 `prism_app::fixed`）。
- 系统的 `AccessSet` 由树的 Task/Evaluator 汇总，交给 `prism_app` 调度器做并行冲突消解；写冲突的实例自动串行。
- 运行期**不在树内直接 spawn/despawn/加组件**：结构性变更一律 `CommandQueue` → ECB（对齐 `prism_ecs` 的 deferred 模型）。

### 9.5 热重载

- `TreePlan` 带 `PlanRev`；资产热重载编译新 plan，按**稳定 StateGuid**（非索引）重映射在跑实例：状态仍存在则保留实例帧字节，消失则走 `on_exit` + 降级到最近存活祖先。
- 重映射失败策略可配：`KeepOld`（默认，安全）或 `ForceMigrate`（编辑器/开发期，快速迭代）。

---

## 10. 快照、回滚与存档

`InstanceSnapshot`（紧凑、可序列化、可跨进程）：

```
- PlanRev + 活跃路径(StateGuid 化) + 每 region 游标
- 实例帧 blob（编译期布局 → 定长字节）
- 待处理事件环（定容）
- 定时器表（相对固定步的相位）
- 实例 RNG 状态（种子化，确定性）
- 历史栈（深/浅历史的进入点）
```

- 进 `prism_ecs::world::snapshot` 的快照环，服务预测回滚（§11）与存档（§17.6）。
- **重放安全分级**（`ReplaySafety`）：`Pure`（无外部依赖）/ `ReplaySafe`（可重放，副作用经 CommandQueue）/ `AuthorityOnly`（依赖网络/物理瞬时结果，禁入预测路径）。编译期标注，回滚路径遇到 `AuthorityOnly` 状态即降级（该实例不预测、等服务端）。
- 存档兼容：`prism_reflect` schema 版本 + `SaveMigration`（对齐 `prism_gameplay` §17/§25）。

---

## 11. 网络：权威、预测与版本一致性

- **默认服务器权威**：影响玩法结果的迁移在服务端裁决；`ReplicationPolicy` 逐节点标注（`ServerOnly` / `ReplicatedState` / `ReplicatedEvent` / `ClientLocal`）。
- **预测**：本地拥有代理可走客户端预测 + 回滚重放（复用 `prism_ecs` 快照回路与 §10 的 `ReplaySafety`）。回滚粒度是**实例**，不是世界。
- **版本一致性**：`PlanRev` 参与握手；不一致时服务端拒绝或降级为 `ServerOnly`（绝不「半懂」地跑对方语义）。
- **状态同步二选一**（按树的性质）：① 同步**迁移事件**（流量小，适合状态少但迁移频繁）；② 两端由**复制输入 + 确定性**各自推导（流量最小，适合确定性玩法，对齐 lockstep）。同一棵树不得混用。
- 与 `prism_network` 的关系：本设计只提供**策略标注与快照接口**，传输/定序/对账归 `prism_network`。

---

## 12. 确定性

1. **全序**：迁移排序键（§5.2）无并列；实例遍历序按稳定 Entity id 排序；禁止 `HashMap` 迭代序进入裁决。
2. **RNG**：每实例种子化 RNG，禁用系统随机与系统时间；`Utility` 打分的随机性是种子化抖动。
3. **时钟**：一律读 `prism_time` 的固定步时间；`Cadence` 计数基于步数而非墙钟。
4. **浮点**：确定性档位下用软浮点/定点的评分路径；非确定性档位默认高性能浮点（对齐 `prism_app::determinism` 的编译期/配置切换）。
5. **可验证**：同种子 + 同输入 → 同 `active_states` 轨迹与同快照哈希（§15.2 golden 测试）。

---

## 13. 性能模型与帧预算

### 13.1 成本模型

单实例一次推进的成本可分解为：

```
C_step = Σ_active Task.tick + Σ_evaluated Guard.eval
       + Σ_transition (exit + enter) + Σ_binding_copy
```

关键性质是**每一项都只在「有事发生」时非零**，于是性能可以按档位推理，而不是靠笼统的"优化"：

| 档 | 触发条件 | 成本量级 | 说明 |
|---|---|---|---|
| L0 静默 | 无事件 / 无定时器到期 / 无依赖变化 | ≈ 0（仅队列判空） | 绝大多数远距离、无关代理 |
| L1 唤醒 | 事件到达或定时器到期 | 微秒级 | 只重判受影响实例的候选迁移（§8.1） |
| L2 依赖重判 | 守卫/Evaluator 的 `DepSet` 变化 | ∝ 变化量 | 依赖追踪，不做常量全量重判（§7.3） |
| L3 迁移 | 命中一条迁移 | exit/enter + 实例数据构造 | 走 LCA（§5.2），任务按 `InterruptPolicy` 中断（§6.3） |
| L4 批处理 | 同构节点批量推进 | 分摊常数 + SIMD/并行 | `BatchRuntime`（§9.3） |
| L5 规划 | 显式请求规划器 | 需预算上限 | `Planner` 扩展点，默认不启用（§8.2） |

反模式（编译期或 CI 拦截）：热路径字符串路径解析（§7.2）、每帧全树遍历、每实例每帧 `Vec` 分配、在守卫里写世界、`AuthorityOnly` 状态进入预测路径（§10）。

### 13.2 量化目标（**设计目标，待基准验证**）

| 指标 | 目标 | 验证方式 |
|---|---|---|
| 静默实例 | 10 万常驻实例，推进成本 < 1 ms/帧 | `benches/` + `prism_diagnostic::budget` |
| 单实例单步 | 堆分配 0、锁 0、panic 0 | 分配计数断言（CI，§15.4） |
| 单次迁移 | < 1 µs（不含用户 actions） | 微基准 |
| 同构代理吞吐 | 1 万代理 @ 60 Hz 在批处理预算内 | 批处理基准（§15.3/§15.4） |
| 深树（≤ 8 层） | 成本不随深度线性劣化 | 祖先链预计算前后对照基准 |
| 决策降档 | 远距离代理成本降至 1/N | 摊销基准（§25.1） |
| 帧预算 | 决策运行时纳入全局预算并可降档 | 逐树超时告警 + 降档（§25.7） |

> 上表是**设计目标而非实测值**。落地时必须以 `benches/` 数据替换；不达标则按 §22.2 的流程降档或调整方案，禁止把目标当结论写进验收。

### 13.3 缓存与内存布局

- 实例帧 blob 的槽位按访问热度排序，热槽位靠前（编译期布局；可依据遥测 §25.5 反哺重排）。
- 迁移表按源状态分组，并**预计算祖先链与 LCA**，避免运行期回溯。
- 事件环、实例池、迁移临时栈全部编译期定容，帧内零分配。
- 批处理路径的实例列按槽位切分（SoA），chunk 边界对齐 SIMD 宽度。

### 13.4 并行

- `AccessSet`（§6.1）汇总出读/写集，交 `prism_app` 调度器做冲突消解；写冲突的实例自动串行（§9.4）。
- 批处理按 chunk 并行（`prism_tasks`），主线程亲和仅授予必须触碰渲染/音频的**表现类** Task。
- 编组树与成员树的写入域不同（§25.3），因此可并行推进。

### 13.5 分歧检测（确定性护栏）

- 开发构建下按固定步对实例快照取哈希，与权威路径/重放哈希比对；不一致即 dump 该实例的迁移轨迹（§14.3）并中止。
- 目的：把"确定性退化"从线上玄学问题变成开发期可复现的失败（与 §10/§12 配套）。

---

## 14. 工具链：作者化、静态校验、调试

### 14.1 资产格式：文本是真相源

- 真相源：`TreeDef` 文本（RON，schema 由 `prism_reflect` 驱动），**可 diff / 可 merge / 可 code review / 可 LLM 生成**。
- 图视图：`prism_state_tree_graph`（feature `graph`）经 `prism_visual_scripting` 画布 + `prism_editor_framework` 呈现；图 ↔ 文本 round-trip，图不引入文本表达不了的语义。
- 子图复用：状态可引用另一棵树资产（`LinkedTree(Handle<TreeAsset>, params)`），参数可覆盖；等价于 UE 5.5 的 subtree 与 Unity 的 subgraph。
- 与 `prism_asset` 集成：mtime 监视 + 热重载 + `PlanRev` 比对（§9.5）。

### 14.2 静态校验器（错误码）

| 码 | 检查 | 时机 |
|---|---|---|
| `ST0001` | 不可达状态（无任何迁移/初始路径可达） | 编译 |
| `ST0002` | 守卫恒假（常量折叠 + 类型/取值域分析） | 编译 |
| `ST0003` | 绑定位路径无法解析或类型不匹配 | 编译 |
| `ST0004` | 状态缺 `on_exit` 处理但存在 latent Task（资源泄漏风险） | 编译 |
| `ST0005` | 并行 region 内两个 latent Task 写同一槽位 | 编译 |
| `ST0006` | 无事件迁移级联可能超上限（震荡风险） | 编译 |
| `ST0007` | `Parallel` 的 region 引用同一 `History`（语义歧义） | 编译 |
| `ST0008` | `AuthorityOnly` 状态可达预测路径 | 编译 |
| `ST0009` | 实例帧含指针/堆类型（违反 §7.4） | 编译 + CI lint |
| `ST0010` | 事件环容量小于最大单微步事件数 | 编译 |
| `ST0011` | Task 未声明 `AccessSet`（并行调度不安全） | 编译 |
| `ST0012` | 子图引用循环或参数类型不匹配 | 编译 |
| `ST0013` | `Compound` 无 `default`/无兜底迁移，存在选择失败可能 | 编译 |

校验器同时可在 `cargo test` 下以 lint 形式跑（对齐仓库测试收口要求，§18）。

### 14.3 StateTree Debugger

- 实时活跃配置高亮（含并行 region 分色）；每状态计时、每 Task 状态、每绑定槽当前值。
- **迁移因果视图**：逐条候选迁移列出「trigger 是否匹配 / guard 返回什么 / 为何败给另一条」，直接回答「为什么不切换」这一最费时的排查。
- 断点：状态进入/退出、Task 完成、守卫求值；命中断点由 `prism_diagnostic` 暂停世界或仅暂停该实例。
- 录制/重放：接 `prism_time::recording`，可回放一次战斗的决策轨迹（与确定性 §12 共用）。
- 对比：两个实例的活跃配置/绑定值 diff（排查「只在这只怪身上出错」）。
- 反事实分叉（"如果当时选另一条分支会怎样"）见 §25.8。

### 14.4 LLM 友好性

复用 `prism_visual_scripting` §15 的节点元数据导出与自修复 loop：文本真相源 + 结构化错误码（§14.2）使「自然语言 → TreeDef → 编译报错 → 自动修复」闭环可行。

---

## 15. 一致性验证：语义测试套件

> 这是「语义先行」（§1.1）的落地证据，也是 `text/integration/` 的主要交付物。

### 15.1 SCXML 语义移植（对标 W3C 测试套件）

把 W3C SCXML 的语义测试用例（状态图部分：层级、并行、历史、eventless、微步、LCA）**手工移植**为 Prism 语义测试：同一输入 → 同一活跃配置序列。目的：让「层级/并行/历史」的正确性不依赖个人理解，而是有可执行的规范对照。

### 15.2 决定性 golden 测试

- 同种子 + 同事件序列 → 相同 `active_states` 轨迹 + 相同快照哈希（跨 3 次运行与跨机器）。
- 快照往返：`snapshot → restore → 继续 tick` 与「不中断 tick」结果逐位一致。
- 回滚重放：预测路径回滚后重放，结果与权威路径一致（`ReplaySafe` 子集内）。

### 15.3 批处理 parity（golden-twin）

`BatchRuntime` 与 `CursorRuntime` 跑同一棵树同一输入序列，活跃配置与最终快照必须一致（公差：无，语义级一致）。该测试对应仓库 `text/reference/` 的 golden-twin 规则，须注释对照关系。

### 15.4 性能回归

基准用例（静默实例成本、单实例单步分配数、万级代理吞吐）进 `benches/`，CI 记录分配次数（应为 0）与吞吐趋势。

覆盖率驱动的用例生成（迁移/活跃配置/守卫真值组合覆盖）见 §25.4。

---

## 16. 安全与沙箱

- **地基层零 `unsafe`**：`prism_state_tree_core` / `prism_state_tree_ecs` 不含 `unsafe`；如个别 SIMD/布局代码必须用，收口到最小块 + `// SAFETY:` 注释 + 最小单测（对齐仓库 §7.2）。
- **实例帧内存安全**：blob 访问一律经布局表 + 边界检查（`get_slot::<T>() -> Result<&T, LayoutError>`），禁裸指针强转；`ST0009` 静态拦截指针类型。
- **外部输入校验**：`path`（`prism_reflect`）、图资产、WASM Task 参数、网络下发的迁移事件、行为包契约（§25.6）一律校验后再用；失败降级不 panic。
- **脚本/图 Task 沙箱**：非原生 Task（WASM/图）走 `prism_script` / `prism_visual_scripting` 的能力模型与预算门控；树为其提供 `AccessSet` 白名单，超界即拒。
- **依赖审计**：新增依赖遵循仓库 `deny.toml` 策略；`no_std` 层零平台假设。

---

## 17. 与既有子系统的装配

### 17.1 ECS 集成

```rust
#[derive(Component)] pub struct StateTreeInstance { pub asset: Handle<TreeAsset>, plan: Arc<TreePlan>, /* blob… */ }
// 关系：RunningTree / SupervisedBy / SmartObjectClaim
// 事件：OnStateEntered / OnStateExited / OnTaskCompleted / OnTransitionRejected
```

当前 ST1 通过确定性 Entity 顺序发布带 payload 的双缓冲 `Events<StateTreeLifecycleNotification>`；它不是 `prism_ecs` 无载荷 marker observer，也不承诺迁移回调期间的即时 reaction。observer/reaction 桥接保留为后续里程碑。

### 17.2 不设独立行为树：BT 语义归约为激活模式

- BT 的全部有用语义可归约为三件事，本设计均已覆盖：① 条件驱动、可抢占的子节点选择 → §8.1 的 `Eventless` + `Internal` 迁移组合；② `Success/Failure/Running` 三态 → §6.1；③ 有序/权重/评分候选 → §8.2。
- 因此**不引入独立的 BT 运行时**：`Eventless` + `Internal` 让任一 `Compound` 状态按「依赖变化即重判 + 可抢占且不重启子节点」运行，语义等价于一棵 selector 树；`Event` + `External` 即状态机。两者在同一内核中逐节点混用。
- 收益：只有一个运行时、一个绑定模型、一个调试器、一个快照模型，没有语义漂移与重复基建（UE 的 BT 与 StateTree 各一套基础设施正是反面教材）。
- 外部 BT 资产（若有）经**转换器**编译为 `Eventless`/`Internal` 迁移组合，属工具层（§14.1）能力，不引入第二运行时。

### 17.3 AI 域装配（对齐 `prism_gameplay_design_zh.md` §14/§29）

```
Perception ──写──> Blackboard ──绑定──> StateTree(Evaluator/Guard) ──Task──> 与玩家一致的输入意图
```

### 17.4 交互与 Smart Object（对齐 §29.2 / §30）

- 玩家 `Interactor` 与 AI Smart Object 共用**同一执行体**：交互 = 一次 GAS 能力 = 树上的一个 Task。
- 槽位预定（claim）用关系 + GameplayTag 需求匹配；抢占与释放是迁移，不是特判代码。

### 17.5 动画与 UI

- 动画状态机：作为专用 Schema 驱动，迁移的 `duration` / `interruption source` / `priority` 映射到本设计的 `InterruptPolicy` + 混合参数；不重造 Animator。
- UI/流程：菜单栈、任务、对话、过场流程 = `FlowContext` 上的树；长流程持久化交 `prism_visual_scripting` durable（避免两套存档语义）。

### 17.6 存档

`InstanceSnapshot` + `prism_reflect` schema 版本 + `SaveMigration`；与 `prism_gameplay` §17 的存档框架共用同一序列化通道。

### 17.7 可编程性

Task 可由 Rust 原生、图脚本或 WASM 提供（`prism_script`）；热路径默认原生，脚本 Task 受预算与白名单门控。

---

## 18. Crate 拆分与 feature 矩阵

> 遵循仓库 §1 命名（`snake_case` + `prism_` 前缀）与 §3 单职责；每个 crate 独立可用、独立 Plugin，齐备三件套（§6.4）后方可合入 `main`。

```
pkg/
  prism_state_tree          # 门面：默认插件组 + 预置 Task/Guard/Schema + 常用策略（稳定 API 面）
  prism_state_tree_core     # 语义内核：TreeDef/TreePlan/游标运行时/确定性/快照（no_std + alloc）
  prism_state_tree_ecs      # ECS 集成：组件/Schema/Context/CommandQueue/observer 事件/关系
  prism_state_tree_author   # 作者层：资产 schema/序列化/校验器/迁移（prism_reflect + prism_asset）
  prism_state_tree_debug    # 调试：活跃配置/迁移因果/断点/对比/录制重放（prism_diagnostic）
  prism_state_tree_batch    # 群体：SoA 批量执行器（feature batch，prism_tasks）   [可选]
  prism_state_tree_graph    # 图编辑器宿主（feature graph，prism_visual_scripting）[可选·前置待建]
  prism_state_tree_utility  # Utility 选择策略（feature utility）                 [可选]
```

落地优先级：**必设** `core` + `ecs` + `author` + 门面 `prism_state_tree`；**按需** `batch` / `debug` / `utility`；**前置待建** `graph`。

feature 矩阵（默认最小化，不暴露矛盾组合；以 `docs/cargo_features.md` 为准）：

| feature | 影响 crate | 内容 | 默认 |
|---|---|---|---|
| `std` | all | ECS/资产/调试/编辑器 | 门面与 ECS 层开启；`core` 保持 `no_std` |
| `serde` | core/author | 快照与资产序列化 | 开 |
| `reflection` | author | `prism_reflect` 路径绑定与 schema 迁移 | 开 |
| `batch` | batch | SoA 批量执行器（需 `std` + `prism_tasks`） | 关 |
| `utility` | utility | Utility 选择策略与响应曲线 | 关 |
| `graph` | graph | 图编辑器宿主（需 `prism_visual_scripting`） | 关 |
| `determinism` | core | 软浮点/定点评分路径、全序断言 | 关 |
| `net` | ecs | 复制策略标注、`PlanRev` 握手、快照接口 | 关 |
| `replay` | debug | 录制/重放与轨迹导出 | 关 |
| `hot-reload` | author/ecs | `PlanRev` 重映射与资产热重载 | 关 |
| `trace` | debug | 迁移因果与预算事件（`tracing::event!` 门控） | 关 |
| `telemetry` | debug | 聚合遥测与报告导出（§25.5） | 关 |
| `verify` | author | 模型化验证器与测试生成（§25.4，编译期/CI） | 关 |

---

## 19. 公共 API 草案

> 面向三类消费者：① 引擎集成方（挂 schedule / 接资产）② Task 作者（注册行为）③ 作者化工具（编辑器 / LLM）。以下为意图草案，非最终签名。

```rust
// ── 19.1 零配置起步（引擎集成方）─────────────────────────────
App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(StateTreeDefaultPlugins)          // core + ecs + author + debug
    .run();

commands.spawn((
    StateTreeInstance::new(tree_asset.clone()),    // Handle<TreeAsset>
    AIControllerBundle::default(),                 // 提供 AiContext（§7.1）
));

// ── 19.2 编译（引擎 / 工具，off-thread）──────────────────────
pub fn compile(def: &TreeDef, schema: &SchemaRef, reg: &TaskRegistry, opts: CompileOpts)
    -> Result<TreePlan, Vec<StError>>;             // 结构化错误 → 调试器 / LLM 自修复

// ── 19.3 运行（ECS 侧）──────────────────────────────────────
impl StateTreeInstance {
    pub fn plan(&self) -> &Arc<TreePlan>;
    pub fn active_states(&self) -> impl Iterator<Item = StateId>;
    pub fn request(&mut self, req: RequestKind);                 // §6.2
    pub fn snapshot(&self) -> InstanceSnapshot;                  // §10
    pub fn restore(&mut self, s: &InstanceSnapshot) -> Result<(), RestoreError>;
}

// ── 19.4 Task 作者 ──────────────────────────────────────────
pub trait Task { /* §6.1 */ }
pub trait Evaluator { /* §7.3 */ }
pub trait Guard: Send + Sync { fn eval(&self, cx: &GuardCtx<'_>) -> bool; }
pub trait SelectionPolicy { /* §8 */ }

pub struct TaskRegistry;   // 内容寻址，产 registry_digest；支持原生 / WASM / 图 Task
impl TaskRegistry { pub fn register<T: Task + 'static>(&mut self, t: T) -> Result<(), RegError>; }

// ── 19.5 调试 / 观测（feature debug）─────────────────────────
pub struct Trace { pub transition_attempts: Vec<TransitionAttempt>, pub active_path: Vec<StateId> }

// ── 19.6 批量（feature batch）───────────────────────────────
pub struct BatchRuntime;
impl BatchRuntime { pub fn tick_batch(&mut self, world: &mut World, agents: &[Entity]) -> BatchOutcome; }
```

- **稳定性分层**：`compile` / `TreePlan` / `Task` / `Evaluator` / `Guard` / `StateTreeInstance` 为稳定核心；`BatchRuntime` / `Utility` / `graph` / `Trace` 为门控扩展。
- **错误即数据**：所有 `*Error` 均可序列化，带 `code` + 定位（对齐 §14.2），服务调试器与 LLM 自修复。

---

## 20. 路线图（ST0–ST8）

> 与 `prism_gameplay_design_zh.md` §24 的 M4（AI）对齐：ST0–ST3 是 M4 的前置；ST4 起为超越基础 AI 的差异化能力。

| 里程碑 | 内容 | 出口标准 |
|---|---|---|
| **ST0 语义内核** | `TreeDef`/`TreePlan`/游标运行时的 `no_std` 内核；层级 + 并行 region + 历史 + 微步 | §15.1 语义套件绿；`cargo test --doc` 绿 |
| **ST1 ECS 装配** | `StateTreeInstance` 组件、Schema/Context、CommandQueue、带 payload 的生命周期 Event stream、调度挂接 | 显式注册后可跑：spawn + 资产 → 状态可切换；无载荷 observer marker 与 zero-config 仍为后续里程碑 |
| **ST2 数据层** | 静态绑定 + `prism_reflect` 路径编译为 slot；Evaluator 依赖驱动；实例帧布局 | **基础契约、帧布局与反射读写切片已交付**：严格 binding path、稳定 slot、依赖反向索引与确定性 dirty evaluator；core 提供无 unsafe 的 `FrameLayout`/`InstanceFrame` 字节布局和边界检查，`prism_state_tree_ecs/reflect` 提供一次编译的 `AccessPlan`、稳定 slot 排序、重复 slot/路径错误、目标 reflected type mismatch 校验及 typed `get`/`set`（缺失 slot/底层 apply 失败结构化返回）；反射值与实例帧自动桥接、`ST0003/ST0009` 完整诊断和静默实例零 tick 基准仍待交付 |
| **ST3 作者层与调试** | RON 资产 + schema 迁移 + 校验器 + Debugger（活跃配置/因果/断点） | 一份完整敌人树可作者化、可调试、热重载生效 |
| **ST4 AI 域闭环** | 感知 → 黑板 → 树 → 与玩家一致的输入意图（含 §8 的 `Eventless` + `Internal` 抢占语义） | 敌人「巡逻→警戒→追击→战斗→脱战」全链路 Demo |
| **ST5 快照与网络** | `InstanceSnapshot` 进快照环；`ReplaySafety`；复制策略 + `PlanRev` 握手 + 预测回滚 | 回滚重放与权威一致（§15.2）；双端演示 |
| **ST6 规模化** | `BatchRuntime` + LOD 分档 + `utility` 选择 | §15.3 parity 绿；万级代理吞吐基线 |
| **ST7 跨域复用** | 交互/Smart Object、动画 Schema、UI/流程 Schema；`graph` 图编辑器（前置待建就绪后） | 至少两个非 AI 域跑通同一内核 |
| **ST8 高级功能** | §25 的规模化与工具链能力（决策 LOD、中断层级、编组、验证器、遥测、行为包、熔断、反事实调试、空间查询） | §25 各项验收证据齐备；性能目标以基准数据替换设计目标（§13.2） |

每里程碑产出对应 `pkg/<crate>/docs/version/v<X.Y.Z>.md`（仓库 §5.2）并把设计文档状态推进（`PLANNED` → `SHIPPED`，逐段标注）。

§25 各高级功能与里程碑的依赖关系：§25.2/§25.4 → ST4；§25.1/§25.3/§25.5/§25.9 → ST6；§25.6/§25.7/§25.8 → ST7–ST8。

### ST0 / ST1 / ST2 实现进度（Implementation Progress）

`pkg/prism_state_tree_core` 已作为根 workspace member 落地，定位为 ST0 的纯语义内核切片（`no_std + alloc`、`#![forbid(unsafe_code)]`、不依赖 ECS/渲染/网络/编辑器）。当前实现拆分为 `model.rs`（定义校验与不可变 `Plan`）、`runtime.rs`（游标运行时）、`snapshot.rs`（快照编解码与校验恢复）和 `tests.rs`（语义回归），而不是把所有逻辑堆在单文件中。

已交付并由测试覆盖的语义：

- `TreeDef → Plan` 的稠密 ID、父子一致性、单父、连通性、环、初始子状态、Parallel region 去重和 History owner 校验；
- Atomic / Compound / Parallel 活跃配置，祖先迁移继承，声明顺序与 priority 的确定性裁决；
- Event / Eventless 迁移、LCA exit/enter、Internal / External / Self 生命周期差异；
- Shallow / Deep history、定容 FIFO event queue、微步预算（超限时保留尚未执行的事件）、纯 guard 和生命周期/迁移日志；
- 确定性语义快照编码、针对 plan 的配置/history 校验、容量受限队列恢复；恢复错误不改变实例状态，且保留已安装 guards。
- Parallel Deep History 恢复会跳过默认 region 初始化，避免历史子状态与默认子状态叠加；微步预算错误不返回部分 `StepReport`，调用者应检查队列与日志后重试。

当前证据：本轮 `cargo test -p prism_state_tree_core` 为 **38 tests + 0 doctest 全绿**；回归覆盖 Schema/Context 描述符与上下文身份校验、首次 `run()` 的启动 Eventless 稳定化、事件共享同一微步预算、事件迁移后 eventless guard 观察到 `GuardContext.event == None`、首次运行时先完成初始 eventless 稳定化再消费已排队事件、Parallel owner 转移后所有 region 仍有活跃配置和 History 请求深度与 owner 配置不匹配时的编译期拒绝。启用 `prism_state_tree_ecs/reflect` feature 的集成单测为 **23 tests + 0 doctest 全绿**，其中 2 项覆盖一次编译的 dotted path、稳定 slot 读取、重复 slot/路径失败与目标 reflected type mismatch；core 另有 2 项覆盖实例帧对齐布局、重复声明拒绝和边界读写，adapter 已提供经过编译路径的 typed reflected `get`/`set`（测试覆盖实际写入、错误类型写入和缺失 slot），但尚未把 reflected value 自动编码进 `InstanceFrame`。`cargo clippy -p prism_state_tree_core --offline -- -D warnings` 通过；`prism_state_tree_ecs` 默认 feature 当前集成单测为 **21 tests + 0 doctest 全绿**（启用 `app` feature 时为 22 tests），覆盖显式批量 tick 的顺序、重复项的实际二次执行、stale/未 materialize/缺组件三种缺失原因、错误隔离、失败后的状态快照、实例独立性、基于真实 ECS query 的 `tick_all` 确定性 gather、显式 Schedule 系统每次运行一次和报告资源保留结果、生命周期队列顺序、Schedule tick 新增事件的带 Entity payload `Events<StateTreeLifecycleNotification>` 发布，以及 typed ECS context gather。迁移/历史热路径目前使用临时 `Vec` 且日志无界增长，零分配仍是尚未满足的设计目标；完整 SCXML 一致性套件和分配计数基准也尚未交付。

ST1 最小装配切片：`pkg/prism_state_tree_ecs` 提供共享 `Arc<Plan>` 资产句柄、持有独立且长期存活 core cursor 的 `StateTreeInstance` ECS 组件、经 `CommandQueue` deferred spawn 的 `queue_spawn`，以及显式逐实体 `tick_entity` / `step`、显式按调用顺序批量执行的 `tick_entities`、基于 `World::query_filtered` 的确定性 `tick_all` gather、有界事件请求 API、`install_tick_system` Schedule 接线、`StateTreeLifecycleEvent` 实例队列、带 Entity payload 的 `Events<StateTreeLifecycleNotification>` 桥接和显式 `EcsSchema`/`gather_context` typed context gather。`tick_all` 收集所有已 materialize 的实例，按 generational `Entity` 排序后执行；Schedule 系统每次运行复用同一路径，并把最后一次 `BatchTickReport` 或 gather 错误放入 `StateTreeTickReport`，但仍由宿主显式注册。生命周期队列保留 core enter/exit/transition 回调顺序；Schedule 在每次 tick 前记录各实例队列边界，只以确定性 Entity 顺序发布该次 tick 新增事件，并 rotation 双缓冲事件（标准两次 update grace window）；显式 tick 遗留事件仍由显式 drain API 管理，这仍不是无载荷 marker observer。core `Guard` 已增加 `Send + Sync` 约束，允许运行时组件满足 ECS 线程安全边界；集成测试验证 spawn 应用后初始进入，再请求事件并在后续 tick 切换状态，并覆盖事件队列满、预算错误、stale/未 materialize/缺组件三种缺失原因、重复输入顺序、重复句柄实际执行、错误隔离、失败后状态快照、实例独立性、gather 排序、Schedule 运行次数、生命周期顺序、事件发布和 typed context gather（默认 feature 21 个测试；启用 `app` feature 时为 22 个测试，另含 plugin 回归测试）。该 crate 及其 `prism_ecs` 依赖保持 `no_std + alloc` 构建路径。

ST2 数据层基础契约：`pkg/prism_state_tree_core/src/binding.rs` 提供严格的 `BindingPath`、稳定 `BindingSlot`、`EvaluatorDependency` 与 `DependencyPlan`。路径只接受规范 dotted identifier；依赖计划在编译时建立 slot → evaluator 反向索引，变化 slot 只唤醒受影响 evaluator，输出按 evaluator id 确定性排序并去重。该层保持 host-neutral，不读取 `World`，不引入 `prism_reflect`。`pkg/prism_state_tree_core/src/frame.rs` 进一步提供无 unsafe 的 `FrameLayout`/`InstanceFrame`：声明按 slot 排序，编译时计算对齐/偏移/总大小，运行时 byte range 访问经过 slot 存在性、边界和长度校验。ECS crate 的可选 `reflect` feature 在 `reflect_binding.rs` 中把路径编译为 `prism_reflect::AccessPlan`，按 slot 排序后二分查找并重放索引链；`compile_typed` 在加载/编译阶段拒绝重复 slot、非法/不可解析路径和目标 reflected type mismatch。typed reflected writer、实例帧与 reflected value 的自动桥接、ST0003/ST0009 完整诊断和静默实例零 tick 基准仍待交付。当前 `ReflectBindingPlan::set` 已提供经过编译路径的反射字段写入，并将缺失 slot/底层 apply 错误结构化返回；它尚未把值编码为 `InstanceFrame` 字节。

Schema/Context 基础契约已落地：`prism_state_tree_core` 提供稳定 `SchemaId`/`SchemaVersion`、能力位、只读 `ContextRef`、描述符校验和上下文 schema 身份校验；它不依赖 ECS，也不提供 World gather。

诚实边界：这不是完整 StateTree 产品的假实现，也不宣称 ST0–ST8 全部完成。ST2 已交付 host-neutral binding path/slot/dependency 基础契约、无 unsafe 的 `FrameLayout`/`InstanceFrame` 字节布局与可选 `reflect` feature 的只读 typed adapter（一次编译 `AccessPlan`、稳定 slot 重放、目标类型校验），但尚未实现 reflected value 到实例帧的 typed writer、ST0003/ST0009 完整类型/布局诊断或静默实例基准；ST1 尚未实现生命周期 observer 迁移事件桥接、资产服务器/热重载、ECS 快照回滚或 zero-config 插件；这些仍为 `PLANNED`。`install_tick_system` 已提供显式 Schedule 注册入口，每次 schedule run 只执行一次与 `tick_all` 相同的确定性 ECS gather，并把 `BatchTickReport` 或 gather 失败写入 `StateTreeTickReport`；`StateTreeInstance::drain_lifecycle_events` 已保留 core enter/exit/transition 回调；Schedule 仅发布其本次 tick 新增的回调，这不等于完整 observer、自动/隐式 Context gather 或自动插件。当前 ST0 同样不承诺 transition 期间无堆分配或外部 callback panic 可恢复；ST1 的 typed `EcsSchema`/`gather_context` 已交付，但自动/隐式 Context gather 与完整 binding 仍未交付。

---

## 21. 关键扩展点、风险与非目标

### 21.1 扩展点

- `Task` / `Evaluator` / `Guard` / `SelectionPolicy`：行为四件套。
- `Schema`：接入新宿主域（Actor / AI / Flow / 动画 / UI / 自定义）。
- `Planner`：GOAP/HTN 预留接缝（非内建）。
- `ReplicationPolicy`：逐节点网络策略。
- `ReplaySafety`：自定义重放安全判定。
- `TaskBackend`：原生 / WASM / 图 Task 后端。
- `LodPolicy`：实例在游标/批处理间的切换依据。
- `TelemetrySink`：遥测聚合与导出后端（§25.5）。
- `Verifier`：自定义静态性质检查（§25.4）。
- `Query`：空间/战术查询实现（§25.9）。
- `BehaviorPack` 契约：对外槽位/事件契约与迁移脚本（§25.6)。

### 21.2 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 状态图语义（并行/历史/微步）实现复杂度高 | 正确性风险 | §15.1 规范对照套件；语义先行；禁止「差不多实现」 |
| 双运行时（游标 + 批量）语义漂移 | 同一棵树两种行为 | §15.3 parity 测试作为强制门禁 |
| 实例帧布局变更破坏存档/回滚 | 兼容性风险 | `prism_reflect` schema 版本 + `SaveMigration`；`FrameLayout::fingerprint()` 布局指纹进快照 |
| 网络版本不一致 | 行为分叉 | `PlanRev` 握手 + 不一致降级 `ServerOnly` |
| 作者化工具链工作量 | 迭代速度 | 复用 `prism_visual_scripting` 画布与 `prism_ui*`，不自建编辑器 |
| `graph`/`reactive` 前置未就绪 | 部分 feature 阻塞 | ST0–ST4 不依赖它们；`reactive` 缺失时 Evaluator 退化为轮询 cadence |

### 21.3 非目标（重申）

不内建规划器；不做通用工作流引擎；不做运行期图解释；不引入独立行为树运行时；不替代 `prism_app::state`；不把决策逻辑放进 ECS 存储内循环。

---

## 22. 多维质量评估（§12 八维）

> 本文为 **PLANNED** 设计文档：所有维度「现状」均为**规划中（尚无实现）**，故下表逐维给出**设计已论证的手段**与**验收所需证据**，不把未落地能力描述为已落地（仓库 §5.1 铁律）。

| 维度 | 现状 | 设计手段（论证） | 验收证据要求 | 未覆盖项 |
|---|---|---|---|---|
| **性能** | 规划中 | 默认 `Cadence::Never`；事件/定时器/依赖变化唤醒；扁平表寻址；实例帧 blob；定容事件环；`BatchRuntime` SoA（§9/§13） | `benches/` 记录静默实例成本、单步分配数（应 0）、万级吞吐；纳入 `prism_diagnostic::budget` | 真实基准与帧预算数值；GPU 侧评分（未规划） |
| **效果** | 规划中 | 状态图语义（层级/并行/历史/LCA/微步）对齐 SCXML；§8 三属性对「BT 式 reactive」与「状态机式迁移」的等价覆盖；双运行时 parity（§5/§8/§15） | §15.1 语义套件、§15.3 parity 全绿 | W3C 套件移植覆盖率目标未定；BT 存量资产的转换保真度 |
| **易用性** | 部分实现（ST1 显式装配） | ST1 提供 `queue_spawn`、typed Context、显式 Schedule/Plugin 接线；后续再提供零配置插件组与预置 Schema；文本资产可读可 diff；错误码可定位；doctest 覆盖（§18/§19） | 当前：默认 feature 21 个 ECS 集成测试（`app` feature 22 个）、显式 Schedule/Plugin 回归与 typed Context 用例；后续：`cargo test --doc` 绿及 ST1 零配置出口标准 | 图编辑器体验、零配置运行时（依赖后续里程碑） |
| **易维护性** | 规划中 | 单一职责分层（语义内核 / ECS / 作者 / 调试 / 批量）；`core` 零 `unsafe`；注释讲 why（§4/§16） | CI：Clippy 零告警、`src/**` 无测试（测试收口 `text/`） | — |
| **易拓展性** | 规划中 | `Task/Evaluator/Guard/SelectionPolicy/Schema` trait 扩展点；`TaskBackend` 支持 WASM/图；feature 门控（§21.1/§18） | 新 Schema 接入不改内核；新后端不改编译器 | 跨语言 ABI 稳定承诺（依赖 `prism_script`） |
| **易迭代性** | 规划中 | 文本真相源 + `PlanRev` 热重载 + 稳定 `StateGuid` 重映射；每里程碑 version 文档（§9.5/§20） | 热重载重映射测试；version 文档与 crate 版本一致（仓库 §5.2） | 多版本 plan 并存灰度（未规划） |
| **健壮性** | 规划中 | 全路径 `Result`（`thiserror`），无静默吞错；校验器 13 条错误码；快照往返与回滚重放测试（§14.2/§10/§15） | `ST0001`–`ST0013` 用例覆盖；快照往返逐位一致 | 长跑稳定性/内存碎片数据 |
| **安全** | 规划中 | 地基层零 `unsafe`；实例帧访问带边界检查；外部输入（路径/图/网络事件）校验后使用；脚本 Task 走能力沙箱与预算（§16） | `// SAFETY` 完备性审查；`ST0009` CI lint 生效；依赖审计通过 `deny.toml` | 图/WASM Task 沙箱的越权测试（依赖前置） |

### 22.1 关键取舍（仓库 §12.1）

1. **偏向语义正确，牺牲实现简易度**：引入并行 region / 历史 / 微步使内核复杂度显著高于「UE 式子集」，换取语义可被规范验证、跨域复用（AI/动画/UI 同一内核）与调试确定性。可接受，因为语义歧义一旦扩散到全引擎（多个子系统各写一套状态机）成本更高。
2. **偏向运行时性能，牺牲表达自由度**：守卫纯函数 + 写操作延迟到 `CommandQueue` + 禁运行期结构性写。代价是「守卫里顺手做点事」不合法；收益是确定性、回滚与并行调度可行。
3. **偏向确定性与可回滚，牺牲 Task 自由度**：`ReplaySafety` 强制分级、实例必须可快照。代价是含指针/外部副作用的 Task 不能用；收益是预测回滚与存档统一。
4. **偏向复用既有地基，牺牲独立性**：深度绑定 `prism_ecs` / `prism_reflect` / `prism_app`。代价是上游接口变更会影响本设计；收益是不重造事件、反射、快照、调度四套基建。
5. **默认关闭批处理与 Utility**：默认构建零额外成本与零复杂度；需要规模化的项目显式开启。代价是两套执行路径需 parity 测试守门。
6. **单轨决策系统，牺牲与既有 BT 生态的兼容**：不设独立行为树运行时，BT 语义归约为 `Eventless` + `Internal` 迁移组合（§17.2）。代价是存量 BT 资产与团队习惯需要转换/再学习，且「`Eventless` 重判 + `Internal` 不重启」的语义必须与 BT 的 abort 语义逐条对齐（靠 §15 一致性套件守门）；收益是单一运行时、单一绑定模型、单一调试器、单一快照模型与可验证语义，避免两套决策系统长期漂移。

### 22.2 CR 重点质疑（仓库 §12.2）

- 热路径性能：单实例单步分配数必须为 0（基准佐证）；`Cadence::Never` 默认是否被预置 Task 破坏（逐个审查）。
- 语义正确性：并行 region 下的 LCA 与中断语义是否有规范对照用例（§15.1）而非「实现者自证」。
- `unsafe`：地基层是否真为零，`// SAFETY` 是否覆盖全部前提。
- 双运行时 parity：`BatchRuntime` 引入时是否同步新增 parity 用例（缺则不合入）。
- 破坏性变更：`TreePlan` 布局 / `InstanceSnapshot` 变更是否在 version 文档给出迁移指引（仓库 §5.2）。

### 22.3 适用范围（仓库 §12.3）

本文适用于 `pkg/prism_state_tree*` 全部 crate 的设计与 CR；纯文档改动可简化，但不得省略「安全」与「健壮性」两维。

### 22.4 维度细化：手段与高级功能的映射

> 把 §25 的高级功能回填到维度，作为 CR 时的核对清单（每条都有对应的验收证据要求）。

- **性能**：L0 静默零成本（§13.1）；依赖驱动重判 L2（§7.3）；实例帧 blob + 槽位热度重排（§13.3）；批处理 SoA（§9.3）；决策 LOD 与摊销（§25.1）；熔断与预算降档（§25.7）；分配计数 CI 门禁（§13.2/§15.4）。
- **效果**：状态图语义对齐 SCXML（§15.1）；双运行时 parity（§15.3）；模型化验证覆盖死锁/活锁/守卫矛盾（§25.4）；中断层级与取消窗口的层级矩阵测试（§25.2）；编组协同用例（§25.3）；反事实调试定位根因（§25.8）。
- **易用性**：零配置插件组（§19.1）；文本真相源可 diff/可评审（§14.1）；13 条错误码定位（§14.2）；调试器的迁移因果视图（§14.3）；一条命令产出验证报告与遥测报告（§25.4/§25.5）；doctest 覆盖（§19）。
- **易拓展性**：行为四件套 + `Schema`/`Planner`/`LodPolicy`/`Query`/`TelemetrySink`/`Verifier` 扩展点（§21.1）；`TaskBackend` 支持 WASM/图（§17.7）；行为包契约使第三方可并行贡献（§25.6）；feature 门控（§18）。
- **易维护性**：单一职责分层（§4）；地基层零 `unsafe`（§16）；稳定 Guid 契约与迁移路径（§25.6）；遥测量化重构收益（§25.5）；CR 门禁（同仓库 `directory_structure_rules.md` §11）。
- **易迭代性**：`PlanRev` 灰度与一键回滚（§25.6）；稳定 `StateGuid` 重映射（§9.5）；每里程碑 version 文档（§5.2 仓库规则）。
- **健壮性**：`Result` 纪律与无静默吞错（仓库 §7.1）；熔断与坏树隔离（§25.7）；有界验证（§25.4）；分歧检测（§13.5）。
- **安全**：地基层零 `unsafe` 与 `// SAFETY` 完备（§16）；实例帧边界检查；外部输入（路径/图/事件/行为包）校验后使用；脚本 Task 沙箱与预算；依赖审计（§16）。

---

## 23. 术语表

| 术语 | 含义 |
|---|---|
| `TreeDef` | 作者层定义（文本真相源），可编译为 `TreePlan` |
| `TreePlan` | 编译后的不可变扁平计划，运行期共享 |
| `Instance` | 一个实体上的运行实例：活跃配置 + 游标 + 实例帧 blob |
| `State` | `Atomic` / `Compound` / `Parallel` 之一；`History` 为伪状态 |
| `Region` | `Parallel` 状态下的正交子区域，各自独立活跃 |
| `Task` | 状态内的行为单元（对标 UE StateTree Task），**非** `prism_tasks` 作业 |
| `Evaluator` | 跨状态只读数据提供者，依赖声明式重算 |
| `Guard` | 迁移/选择守卫，纯函数 |
| `Schema` / `Context` | 宿主契约与类型化上下文视图 |
| `Slot` | 绑定编译产物，运行期定长读写单元 |
| `CommandQueue` | 延迟写队列，微步末统一 flush |
| `Microstep` | 一次「选一条迁移并执行」的原子步骤 |
| `ReplaySafety` | 重放安全分级：`Pure` / `ReplaySafe` / `AuthorityOnly` |
| `active configuration` | 当前全部活跃状态的集合 |

---

## 24. 与既有设计文档的关系与偏差

1. **与 `prism_gameplay_design_zh.md` §29.1 的偏差（需 reviewer 确认）**：该节把 StateTree 描述为 AI 域的轻量对象，并把 crate 归入 `prism_gameplay_ai`（§22）。本文主张 StateTree 是**通用决策运行时**（AI、流程、交互、动画、UI 共用），crate 独立为 `pkg/prism_state_tree*`，`prism_gameplay_ai` 只提供 AI 域 Schema、预置 Task 与感知/黑板装配。**理由**：① 跨域复用避免多套状态机；② 不把通用内核压进 `prism_gameplay_*` 依赖链，专用服务器与工具层可单独使用；③ 与 UE 将 StateTree 独立于 AI 模块的方向一致。**建议**：在 §29.1 增补指向本文的引用，并把 §22 的 `prism_gameplay_ai` 描述改为「建于 `prism_state_tree` 之上的 AI 装配层」。
2. **与 `prism_gameplay_design_zh.md` §14 的偏差（需 reviewer 确认）**：该文档目前规划了**独立行为树**——§14 标题「AI 框架：行为树 / 黑板 / 感知」、§14.2 Behavior Tree、§22 `prism_gameplay_ai` 注释「黑板/行为树/StateTree/感知/EQS」、§24 M4「黑板 + 行为树 + 感知」、§25 扩展点 `BehaviorNode`，另 §20/§27/§29.1 有引用。本文主张**不设独立 BT 系统**：其有用语义由 §8 的候选/触发/重入三属性 + §6.1 三态覆盖（理由见 §17.2）。**建议**：把 §14 收敛为「AI 框架：黑板 / 感知 / 决策（`prism_state_tree`）」，删除 §14.2 与 `BehaviorNode` 扩展点，§22/§24 措辞同步改为「黑板/感知/决策装配」。
3. **与 `prism_visual_scripting_design_zh.md` 的关系**：图 IR、durable、LLM 作者化**复用**该文档成果；本文不自建画布、不自建持久化执行引擎（§17.5/§14.4）。
4. **与 `prism_loom_runtime_framework_design_zh.md` §7 的关系**：Evaluator 的依赖驱动重算在 `prism_loom` 就绪后可升级为其响应式绑定内核的消费者（feature `reactive`）；未就绪时退化为 `prism_ecs` 变更检测 + cadence 轮询。
5. **与 `directory_structure_rules.md` 的合规声明**：本文命名 `<topic>_design_zh.md`（§1）、含固定头部（版本 + 适用引擎，§5.1）、标 `PLANNED` 状态（§5.1）、含「多维质量评估」八维段与取舍记录（§5.1/§12.1）、不把规划描述为已落地（§5.1）。后续 crate 落地必须齐备三件套（§6.4）并补 `docs/version/v<X.Y.Z>.md`（§5.2）。

---

# 扩展篇：规模化与工程化能力

## 25. 高级功能

> 本节每个特性都遵守同一条铁律：**不新增语义概念**——一律由 §5–§8 的状态/迁移/任务/绑定/实例数据与 §9 的执行引擎组合而成，新增的只是调度层、工具层与契约层。每节固定给出：借鉴来源、机制、与核心模型的关系、五维影响（性能/效果/易用/拓展/维护）、验收证据。五维用 ↑↑/↑/=/↓ 表示相对影响，`↑` 越大表示该维度收益越显著，`=` 表示中性。

### 25.1 决策 LOD 与摊销调度（Decision LOD）

- **借鉴**：Unreal Mass 的 LOD / off-LOD 仿真；`big_brain` 的 thinker 频率分层。
- **机制**：每实例持有一个 `DecisionLod` 档（`Near` / `Far` / `Offscreen` / `Dormant`），由重要度函数（距离、可见性、玩家关注度、剧情标记）计算。档位决定四件事：① 推进频率；② `Eventless` 重判频率（`EveryNFrames(n)`）；③ 是否转入批处理（§9.3）；④ 是否降级为统计式近似。
- **与核心模型**：档位只改变**什么时候算**，不改变**算什么**；它是 `Cadence`（§6.1）与 `LodPolicy` 扩展点（§21.1）的组合，不需要新概念。
- **抖动与确定性**：升格/降格带**迟滞**与稳定期；档位变化本身是一次显式事件并进遥测（§25.5）。档位函数必须是固定步快照上的确定性输入函数；被跳过的帧要么用**确定性补算**在下次唤醒时摊销，要么由该档**显式声明**语义可粗化（校验器记录，禁止隐式粗化）。
- **五维**：性能 ↑↑（远距离成本降至 1/N）；效果 =（`Near` 档语义不变，粗化项须显式声明）；易用 ↑（默认策略开箱可用）；拓展 ↑（自定义 `LodPolicy`）；维护 ↑（档位与语义解耦，可整体关闭）。
- **验收**：同输入下 `Near` 与 `Far` 的活跃配置轨迹一致（除显式粗化项）；摊销基准进 `benches/`。

### 25.2 中断层级与取消窗口（Interrupt Tiers & Cancel Windows）

- **借鉴**：动作游戏的取消窗口；Unreal AnimBP 的 `Interruption Source`；UE BT 的 abort levels。
- **机制**：在 §5.2 的 `priority`（同层次序）之上补一层 **`Tier`（跨层"谁能打断谁"）**，并引入 `CancelWindow { opens_at, closes_at, allowed_tiers }`（依据任务进度、动画通知或计时器开启）。
- **与核心模型**：`Tier` 与 `CancelWindow` 只是**守卫的输入**（守卫读取消窗口状态），不引入第二套裁决器——迁移裁决仍是 §5.2 的排序键。
- **典型用途**：受击/死亡（`Tier::Critical`）可打断任何决策，且不被 `Tier::Normal` 打断；处决动画在 `closes_at` 之前不可打断。
- **五维**：性能 =（常数比较）；效果 ↑↑（中断行为可预测、可逐条测试）；易用 ↑（声明式，不靠团队口头约定）；拓展 ↑（自定义 `Tier` 表）；维护 ↑（层级集中定义，全局一致）。
- **验收**：层级矩阵测试（谁能打断谁，穷举组合）+ 取消窗口开闭边界测试。

### 25.3 编组决策、共享黑板与占用（Group Trees & Shared Blackboard）

- **借鉴**：Total War 式编队 AI；UE Mass / Zone Graph 的群体协同；Smart Object 的 claim 机制。
- **机制**：**编组树**（`Schema = GroupContext`）与个体树**同构**。编组树的命令通过关系（`SupervisedBy`）向成员树下派迁移请求（`RequestKind::Transition`，§6.2），成员可接受或由守卫拒绝；共享黑板带**访问契约**（读/写/订阅权限 + 并发写策略 `LastWriteWins` / `Reject` / `Merge`）；占用（claim）用关系 + 超时释放。
- **与核心模型**：编组树就是一棵普通树；下派就是事件；占用就是关系 + 计时器。不需要"编组专用 API"或第二套运行时。
- **五维**：性能 ↑（编组层决策频率低，天然摊销到成员）；效果 ↑↑（协同战术：包夹、轮替、掩护）；易用 ↑（与个体树同一心智模型与调试器）；拓展 ↑（新 Schema 即可接新编组形态）；维护 ↑（编组与个体共用验证器与遥测）。
- **验收**：编组下派 → 成员拒绝/接受 → 重新协商的集成用例；共享黑板并发写的确定性用例；claim 超时释放用例。

### 25.4 模型化验证与覆盖率驱动的测试生成

- **借鉴**：XState `@xstate/graph` 的图探索；SPIN / TLA+ / UPPAAL 的有界模型检验；覆盖率驱动的测试生成。
- **机制**（编译期/测试期，**不进入运行期**）：
  1. **路径枚举**：从 `TreePlan` 枚举迁移图（含并行 region 的交错），产出可达活跃配置集合。
  2. **静态性质**：不可达状态、死锁（无迁移可触发且无未完成任务的滞留）、活锁（级联震荡）、守卫矛盾（互斥守卫不可同时为真）、恒假守卫、无兜底选择（§14.2 `ST0013`）。
  3. **测试生成**：以「迁移覆盖 / 活跃配置覆盖 / 守卫真值组合覆盖」为目标，种子化生成确定性用例序列，直接落 `text/integration/`。
  4. **有界验证**：给定事件字母表与深度上限，穷举检查不变量（如「任意状态都可由某条路径回到 `Idle`」）。
- **与核心模型**：验证器只读 `TreePlan`，零运行期语义、零运行期成本。
- **五维**：性能 =（编译期/CI）；效果 ↑↑（把结构性问题挡在设计期）；易用 ↑（一条命令出报告 + 自动生成用例）；拓展 ↑（`Verifier` 扩展点可加自定义性质）；维护 ↑（重构树后立即回归）。
- **验收**：验证器进 CI，对全部示例树零高危告警；生成用例并入 §15 套件。
- **未覆盖项**：无界深度的完整模型检验（当前为有界近似，记 `PLANNED`）。

### 25.5 运行时遥测与设计反哺（Behavior Telemetry）

- **借鉴**：Left 4 Dead AI Director 的强度监控；性能火焰图思路；产品埋点。
- **机制**（受 feature `trace` + `prism_diagnostic` 门控）：每状态驻留时长分布、每迁移触发次数、守卫求值次数与**假阳性率**（"总是假"的守卫是设计坏味道）、Task tick 成本、档位停留分布、实例老化（长期卡在同一状态的实例）。
- **输出**：① 运行期 HUD / 直方图；② 可导出的聚合报告（供设计评审）；③ 反哺编译——按访问热度重排实例帧槽位（§13.3）；④ 异常告警（迁移从未触发、驻留超阈值）。
- **五维**：性能 ↑（关闭时零代码路径；开启时采样聚合零分配）；效果 ↑（用数据改行为而非拍脑袋）；易用 ↑（一条命令出报告）；拓展 ↑（`TelemetrySink` 可接外部分析平台）；维护 ↑（量化重构收益，避免"改完不知道好不好"）。
- **验收**：关闭时的零成本验证（编译期移除）；报告样例进文档。

### 25.6 行为包与语义版本契约（Behavior Pack & SemVer）

- **借鉴**：Unreal Game Features 的可插拔；Temporal 的版本迁移；SemVer 契约。
- **机制**：行为以**包**为单位（`BehaviorPack` = 树 + 节点库 + Schema + 资源 + 迁移脚本），声明 `requires`（引擎/依赖版本）与 `provides`（对外槽位与事件契约）。加载时校验契约，不匹配即拒载并给出可操作错误。
- **版本演进**：稳定 `StateGuid` / `SlotGuid` / `EventId` 构成对外契约；破坏性变更必须升主版本并提供 `Migration`（`prism_reflect` schema 迁移 + `SaveMigration`，§10）。
- **灰度与一键回滚**：`PlanRev` 支持按实例组灰度（新 plan 先对部分实例生效），异常阈值触发自动回滚到上一 `PlanRev`（与 §9.5 热重载同一机制）。
- **五维**：性能 =（灰度只影响 plan 选择）；效果 ↑（出问题可回滚）；易用 ↑（契约错误在加载期暴露，而非运行期偶发）；拓展 ↑↑（第三方/多团队并行贡献行为包）；维护 ↑↑（契约稳定 + 迁移路径明确）。
- **验收**：契约不匹配的拒载用例；灰度 → 回滚演练；跨版本存档迁移用例。

### 25.7 熔断、预算与坏树隔离（Circuit Breaker & Isolation）

- **借鉴**：服务治理的熔断/舱壁；引擎帧预算。
- **机制**：① **微步预算**：单实例单帧最大迁移数与级联深度（§5.2 规则 3）；② **树级时间预算**：超预算即降档（降 `DecisionLod`、关 `Eventless` 重判）并告警；③ **隔离**：某棵树反复超预算或触发不变量违反时，仅降级该树（`SafeMode`：冻结在最后已知良好活跃配置），不拖垮整帧；④ 修复后自动恢复。
- **与核心模型**：预算是 `Capabilities`（§9.1）的一部分；隔离是调度层行为（§9.4），不改语义。
- **五维**：性能 ↑（防单点拖垮帧）；效果 =（降级可观测、可恢复）；易用 ↑（默认开启，作者无需关心）；拓展 =（阈值可配）；维护 ↑（问题精确定位到具体树）。
- **验收**：注入"级联震荡树"验证熔断与恢复；超预算告警在遥测中可见（§25.5）。

### 25.8 反事实调试与时间旅行（Counterfactual Debugging）

- **借鉴**：rr / 反向调试；rollback netcode 的重放；`prism_time::recording`。
- **机制**：建立在确定性重放之上（§12）：① 回放到任意步；② 在该步**改写一次决策**（强制某条迁移 / 否决某条守卫 / 改某个绑定值）；③ 分叉重放并 diff 两条轨迹的活跃配置与关键槽位。
- **用途**：回答"如果当时选另一个分支会怎样"；定位某次误判的根因；给策划演示改动效果而不必改资产。
- **与核心模型**：完全依赖既有的"守卫纯函数 + 快照 + 确定性重放"三件套，不改运行期。
- **五维**：性能 =（仅开发期）；效果 ↑↑（根因定位能力）；易用 ↑（编辑器内交互）；拓展 =（`DebugProbe` 扩展点）；维护 ↑（复杂问题不再靠猜）。
- **验收**：分叉重放的轨迹 diff 用例；`ReplaySafe` 子集内的确定性保证。

### 25.9 空间与战术查询（Spatial/Tactical Queries as Evaluators）

- **借鉴**：UE EQS；影响图（influence map）；本仓 `prism_physics` 的几何查询。
- **机制**：把"找掩体 / 找最佳射击位 / 评估威胁分布"建模为**一类 Evaluator**（`QueryEvaluator`）：声明查询形状与采样，输出绑定槽（如 `best_cover: Entity`、`threat: f32`）。采样与评分可批处理（§9.3），可随档位降频（§25.1）。
- **与核心模型**：不新增"查询节点"类型——它**就是** Evaluator（§7.3），因此天然获得依赖驱动重算、缓存、批处理与降档能力。
- **五维**：性能 ↑（批处理 + 依赖驱动 + 可降频）；效果 ↑（战术质量）；易用 ↑（复用绑定心智，不学新 API）；拓展 ↑（新查询 = 新 crate，不改门面）；维护 ↑（查询与决策解耦，可单独 golden 测试）。
- **验收**：查询结果的 golden 测试（固定输入 → 固定选择）；批处理 parity（§15.3）。
