# Prism Gameplay 次世代游戏框架设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的数据导向 Gameplay 框架设计：在 prism 原生 ECS 内核（`prism_ecs`）之上，构建组合式 Actor、游戏流程、GAS、增强输入、AI、序列编排、模块化玩法与 AAA 级表现/性能整合层，全部以独立 Plugin 组合、可逐层下钻。
> 借鉴 UE5（Actor/GameMode/GAS/Enhanced Input/Subsystem/Blueprint/Game Features/World Partition/StateTree/Mass）、Unity（GameObject/MonoBehaviour/ScriptableObject/DOTS/Timeline/Input System/MVVM），在 ECS 内核上取长补短；但不复刻其 OOP 继承体，也不依赖任何 `bevy_*` crate。
> 本文档为设计规格，暂不进入编码阶段。

- 版本: v0.2（设计阶段，重构为 prism 原生依赖）
- 适用引擎: Prism / `prism_ecs` 生态（脱离 Bevy 后的独立引擎，不依赖任何 `bevy_*` crate）
- 关键依赖（均为 `pkg/` 下的 prism 原生 crate）: `prism_ecs`（数据导向 ECS：组件/查询/关系 `relation`/观察者 `observer`/反应 `reaction`/生命周期钩子 `component_hooks`/预制 `prefab`(`IsA`)/世界分区 `partition`(WorldPartition/DataLayers/HLOD/FloatingOrigin/InterestGrid/streaming/Dormant/LOD)/世界快照 `snapshot`(WorldSnapshot/SnapshotRing/SnapshotDelta)）、`prism_app`（Plugin/PluginGroup/Schedule/SubApp、状态机 `state`(States/SubStates/ComputedStates/StateScoped/OnTransition)、固定步 `fixed`、确定性 `determinism`、`cvar`/`settings`、`lifecycle`、`platform_tier`）、`prism_time`（固定 tick / 时间线 `timeline` / 录制回放 `recording` / 网络时间 `net` / 确定性 `determinism`）、`prism_input`（原始设备输入底座）、`prism_reflect`（反射 / 序列化 / `net_delta` / `schema`，支撑作者层/存档/网络 delta）、`prism_asset`（DataAsset 加载 / 热重载）、`prism_tasks`（并行 job）、`prism_transform`（层级 / 传播 / `interpolation` / `large_world`）、`prism_math` / `prism_math_gpu`、`prism_diagnostic`（`budget`/`hitch`/`profiler`/`replay`/`telemetry` 可观测性）
- 相关文档: `prism_ecs_design_zh.md`（ECS 内核）、`prism_app_design_zh.md`（App/状态/调度）、`prism_time_design_zh.md`、`prism_transform_design_zh.md`、`prism_reflect_design_zh.md`、`prism_asset_design_zh.md`、`prism_physics_design_zh.md`（物理内核/几何查询）、`prism_character_controller_design_zh.md`（§7 Pawn/Controller/Character 的 CCT 落地规格）、`prism_network_design_zh.md`（§16/§33 网络复制/预测/回滚/延迟补偿/无缝迁移的权威实现）、`prism_anticheat_design_zh.md`（权威反作弊）、`prism_animation_engine_design_zh.md`（§41 Motion Matching/分层/IK/Root Motion）、`prism_camera_design_zh.md`（§44 运镜）、`prism_particle_engine_design_zh.md`（§42 VFX）、`prism_audio_engine_design_zh.md`（§43 音频）、`prism_world_system_design_zh.md` / `prism_terrain_world_design_zh.md`（§28 大世界/流送）、`prism_scene_design_zh.md`（场景/Prefab 真相源）、`prism_ui_loom_design_zh.md` / `prism_loom_runtime_framework_design_zh.md`（§35 响应式 UI 数据绑定）、`prism_diagnostic_design_zh.md`（§38 可观测性）、`prism_engine_component_gap_zh.md`（落地缺口追踪）、`prism_game_creator_design_zh.md`（Game Creator：建于本框架之上的游戏模板/内容系统/创作工作流套件，§31/§32 的正式落地）

---

## 目录
1. 设计哲学
2. 现状基线与差距
3. 分层架构
4. 核心概念映射（对标 UE/Unity）
5. Actor 模型：ECS 原生的组合式 Actor
6. 游戏流程：GameMode / GameState / PlayerState
7. 控制与附身：Pawn / Controller / Character
8. Gameplay Tag 系统
9. Gameplay Ability System（GAS 对标）
10. 增强输入系统（Enhanced Input 对标）
11. Subsystem / 服务定位
12. 消息与事件总线
13. 数据驱动作者层：DataAsset / Prefab / 热重载
14. AI 框架：行为树 / 黑板 / 感知
15. 序列与编排：Timeline / Playable
16. 网络复制与预测回滚
17. 存档与持久化
18. 可编程性：脚本与可视化蓝图
19. 确定性
20. 性能策略
21. 易用性分层与默认体验
22. Crate 拆分与落地形态
23. 公共 API 草案
24. 路线图
25. 关键扩展点清单
26. 模块化玩法：Game Features 运行时插拔（旗舰）
27. 大规模实体与人群（Mass 对标）
28. 大世界：World Partition / Data Layers / HLOD / 后台仿真
29. AI 进阶：StateTree / Smart Objects / 群体导航
30. 交互与世界契约
31. 叙事与进程：任务 / 对话 / 阵营声望（游戏模板层）
32. 物品：背包 / 装备 / 掉落表（游戏模板层）
33. 网络进阶：预测 Key / Lag Compensation / 双模 / 无缝迁移
34. 时间操控与确定性回放
35. 响应式玩法与 UI 数据绑定（MVVM 对标）
36. 程序化内容（PCG）与程序化叙事
37. 手感与命中判定层
38. 可观测性与调试：Gameplay Debugger / Visual Logger / 实时调参 / 自动化测试
39. 高级 ECS 玩法机制：Aspect 组合 / 批量结算 / GPU-driven
40. AAA 质量基线与对标产品
41. 动画系统：Motion Matching / 分层 / IK / Root Motion
42. VFX 特效：GPU 粒子与 Cue 整合
43. 音频：空间化 / 交互混音 / 程序化（MetaSound 对标）
44. 相机与运镜（Cinemachine 对标）
45. 过场与叙事演出整合
46. 角色次世代细节：布料 / 毛发 / 破坏 / 肌肉
47. 帧预算与性能工程
48. 跨系统整合：一帧内的数据流与时序
49. 落地与验收：Demo 与质量/性能清单
50. 术语表
51. 游戏设置框架（GameUserSettings 对标）

---

## 1. 设计哲学

对标商用旗舰引擎的 Gameplay 层，确立五条铁律：

- **ECS 原生，而非移植 OOP**：不复刻 UE 的 UObject 继承树或 Unity 的 GameObject 装箱，而是把 Actor/Ability/Effect 全部落到组件与关系上。逻辑即 System/Observer，状态即 Component/Resource。
- **易用性即 API 分层**：默认"零配置能跑"（一个 `Character` bundle 即可移动+跳跃+被控制），进阶用户逐层下钻替换调度、属性、能力。对标 Bevy 的 modular 哲学。
- **数据驱动优先**：能用 DataAsset/Scene/Prefab 描述的绝不写死在代码里。能力、效果、输入映射、AI 行为均可序列化、可热重载、可被编辑器编辑。
- **组合优于继承**：UE 的 `ACharacter : APawn : AActor` 继承链，改为 `Bundle`/`Component` 组合 + `require` 依赖 + 关系（Relationship）。功能通过附加组件获得，而非通过继承获得。
- **确定性可选**：Gameplay 状态推进提供确定性调度路径（固定步长 + 有序系统 + 定点数值可选），服务联机回滚与录像回放；默认路径为高性能非确定性。编译期/配置切换。

设计取舍总表：

| 维度 | Prism 选择 | 对标与理由 |
|---|---|---|
| 对象模型 | 组合式 Actor（Entity + Component + Relationship） | 取 UE Actor 的组织性 + Unity 组件化 + ECS 性能 |
| 生命周期 | 组件钩子 + 观察者事件 | 替代 MonoBehaviour 的 Awake/Start/Update 心智，且并行安全 |
| 游戏流程 | States + Resource + 关系 | GameMode/GameState 用状态机 + 资源表达 |
| 能力系统 | 数据导向 GAS（Attribute/Effect/Ability 全组件化） | 保留 GAS 的表达力，去掉其运行时开销与样板 |
| 标签 | 分层 interned GameplayTag | 对标 UE GameplayTag，O(1) 匹配 |
| 输入 | 上下文栈 + 抽象 Action + 修饰器/触发器 | 对标 Enhanced Input / Unity Input System |
| 服务 | Subsystem（World/App 级） | 对标 UE Subsystem，替代单例/全局 |
| 作者层 | DataAsset + Scene Patch + 热重载 | 对标 ScriptableObject / DataTable / Prefab |
| 网络 | 组件级复制 + 客户端预测 + 快照回滚 | 对标 UE 复制 + Unity Netcode，鲁棒可扩展 |
| 脚本 | Rust 一等公民 + 可选 WASM/图脚本 | 保留原生性能，图脚本作可选层 |

---

## 2. 现状基线与差距

### 2.1 已有基础（prism 原生内核）
- `prism_ecs`：Archetype 存储、并行调度、**关系（`relation`）**、**观察者/触发器（`observer`）**、**反应图（`reaction` / `ReactionGraph`）**、**生命周期钩子（`component_hooks`：`on_add`/`on_insert`/`on_remove`/`on_replace`）**、**预制（`prefab`：`IsA`，Template/Patch 语义）**、`event`/`message`、`require` 组件依赖、Entity disabling。
- `prism_ecs::partition`：**世界分区原语已内建**——`WorldPartition`（Cell 网格）、`DataLayers`、`Hlod`、`FloatingOrigin`/`GridCell`（大世界精度）、`InterestGrid`（兴趣）、`CellEntityIndex`/`WeakRefs`（流送）、`Dormant`/`DormancySet`（休眠）、`EntityLodProcessor`（LOD 分级）。§27/§28 大规模实体与大世界直接复用，不再从零造轮子。
- `prism_ecs::world::snapshot`：**世界快照原语已内建**——`WorldSnapshot`/`SnapshotRing`/`SnapshotDelta`。§16 回滚、§19 确定性、§34 倒带/回放直接复用。
- `prism_app`：Plugin、PluginGroup、Schedule、`sub_app`、`run_mode`；状态机 `state`（`States`/`SubStates`/`ComputedStates`/`StateScoped`/`OnTransition` + 状态作用域清理）；固定步 `fixed`；`determinism`（确定性开关）；`cvar`/`settings`、`lifecycle`、`platform_tier`（平台降档）、`watchdog`。
- `prism_time`：固定 tick、`timeline`（时间线编排，§15）、`recording`（录制/回放，§34）、`net`（网络时间）、`determinism`、`adaptive_quality`（降档）。
- `prism_reflect`：运行时反射 + `schema` + `net_delta` + `property_bridge`，支撑序列化 / 编辑器 Inspector / 存档 / 脚本 / 网络 delta。
- `prism_transform`：层级（`ChildOf`/`Children`）与传播、`interpolation`（网络插值）、`double_buffer`、`large_world`（大世界坐标）。
- `prism_input`：原始设备输入底座（keyboard/mouse/gamepad/touch/axis/button）——增强输入抽象（Context/Action/Modifier/Trigger）尚缺，由本框架 §10 填补。
- `prism_asset` / `prism_tasks` / `prism_math` / `prism_diagnostic`：资产热加载、并行 job、SIMD 数学、可观测性（`budget`/`hitch`/`profiler`/`replay`）。

### 2.2 关键缺口（Gameplay 层缺失）
`prism_ecs`/`prism_app` 是低层数据导向内核与 App 框架，提供了 ECS、关系、观察者、状态机、分区、快照等**原语**，但**没有**成体系的 Gameplay 框架。当前缺（本设计填补）：
- 无 Actor/Pawn/Controller/Character 等高层抽象与默认体验。
- 无游戏流程编排（GameMode/GameState/PlayerState 生命周期与规则挂点）。
- 无能力/属性/效果系统（GAS 对标）。
- 无 GameplayTag 分层标签系统。
- 无增强输入抽象（`prism_input` 当前只有原始设备输入）。
- 无 Subsystem 服务模型、无 gameplay 消息总线。
- 无 AI（行为树/黑板/感知/StateTree/Smart Object）、无序列编排、无存档框架。

> **与早期版本的差异**：大世界流送、兴趣管理、LOD/休眠、世界快照回滚、状态机、确定性/固定步**不再属于"内核缺口"**——它们已作为原语存在于 `prism_ecs::partition` / `prism_ecs::world::snapshot` / `prism_app::state` / `prism_app::fixed`+`determinism` / `prism_time`。本框架在其上做**玩法语义封装**（升降格策略、复制标注、回滚边界、规则挂点），而非重复实现底座。网络复制/预测/延迟补偿/无缝迁移由 `prism_network` 承载（§16/§33 只定义玩法侧接缝），权威反作弊由 `prism_anticheat` 承载。

本设计即填补 Layer 3–5 的 Gameplay 框架空白，同时严格复用内核的关系/观察者/钩子/状态/分区/快照能力，不重复造轮子。

---

## 3. 分层架构

```
Layer 5  作者/编辑层     Prefab、DataAsset、可视化蓝图、热重载、编辑器 Inspector
Layer 4  高层玩法框架     GameMode/State/PlayerState、Pawn/Controller/Character、GAS、AI、Timeline
Layer 3  玩法服务层       Subsystem、GameplayTag、Enhanced Input、消息总线、复制/预测、存档
Layer 2  玩法内核         Actor 组合模型、生命周期编排、确定性调度、gameplay 事件模型
Layer 1  ECS 内核（既有）  prism_ecs: Component/System/relation/observer/reaction/hook/prefab/partition/snapshot；prism_app: state/fixed/determinism
Layer 0  平台/时间/资产    prism_time/prism_input/prism_asset/prism_tasks/prism_reflect/prism_transform/prism_math
```

- Layer 2 只依赖 `prism_ecs`，是"引擎无关玩法内核"，可被专用服务器逻辑复用。
- Layer 3 各系统之间松耦合，均以独立 Plugin 形式接入，可单独启用/替换。
- Layer 4 是"有观点"层：提供开箱即用的默认玩法组件，但每一处都能被下钻覆盖。
- 跨语言/脚本边界只出现在 Layer 5（图脚本/WASM），不侵入内核。

---

## 4. 核心概念映射（对标 UE/Unity）

| 概念 | UE5 | Unity | Prism Gameplay |
|---|---|---|---|
| 世界中的对象 | AActor | GameObject | Entity + Actor 标记组件 |
| 行为单元 | UActorComponent / Tick | MonoBehaviour / Update | Component + System / Observer |
| 场景层级 | USceneComponent 附着 | Transform 父子 | `ChildOf`/`Children` 关系 + Transform |
| 可被控制体 | APawn / ACharacter | 自定义 | `Pawn` / `Character` bundle |
| 控制器 | APlayerController/AIController | 自定义 | `Controller` + `Possesses` 关系 |
| 游戏规则 | AGameModeBase | 自定义/Manager | `GameMode`（资源 + 规则系统集） |
| 全局状态 | AGameStateBase | 自定义 | `GameState`（资源 + States） |
| 玩家状态 | APlayerState | 自定义 | `PlayerState`（每玩家实体） |
| 能力 | GAS GameplayAbility | 自定义 | `Ability`（组件 + 状态机 System） |
| 效果 | GameplayEffect | 自定义 | `GameplayEffect`（数据 + 应用管线） |
| 属性 | AttributeSet | 自定义 | `Attribute` 组件 + 聚合器 |
| 标签 | GameplayTag | Tag/Layer | `GameplayTag`（分层 interned） |
| 输入 | Enhanced Input | Input System | `InputContext`/`InputAction` |
| 服务 | Subsystem | 单例/Manager | `Subsystem`（World/App 级） |
| 蓝图 | Blueprint | Visual Scripting | 可选图脚本层（Layer 5） |
| 数据资产 | DataAsset/DataTable | ScriptableObject | `GameDataAsset` + 反射 |
| 预制 | Blueprint Class/Prefab | Prefab | Scene/Template + Patch |
| 消息 | GameplayMessageSubsystem | Event/UnityEvent | `GameplayMessage` 总线 + Observer |

---

## 5. Actor 模型：ECS 原生的组合式 Actor

### 5.1 心智模型
- **Actor = 一个被标记为可参与玩法的 Entity**，通过 `Actor` 标记组件与关系树组织。
- 不使用继承。UE 的 `ACharacter : APawn : AActor`，在 Prism 表达为：`Character` bundle `require` `Pawn` bundle `require` `Actor`。
- 组件即"UActorComponent"，但不 Tick 自己——逻辑集中在 System/Observer，天然并行、可批处理。

### 5.2 生命周期编排（替代 MonoBehaviour 生命周期）
用组件钩子 + 观察者事件表达 UE/Unity 的生命周期，且并行安全：

| UE/Unity | Prism | 实现机制 |
|---|---|---|
| Constructor / Awake | `on_add` 钩子 | 组件插入即触发，同步执行 |
| BeginPlay / Start | `OnBeginPlay` 观察者事件 | 首次进入运行态时触发（延迟到世界就绪） |
| Tick / Update | 常规 System（`Update` 集） | 并行、可按需 run-if |
| FixedUpdate | 固定步长 System（`FixedUpdate` 集） | 与物理对齐 |
| EndPlay / OnDestroy | `on_remove` 钩子 + `OnEndPlay` | 销毁前回收资源、触发事件 |
| OnEnable/OnDisable | Entity disabling + 观察者 | 复用内核 entity_disabling |

关键差异：BeginPlay 语义用观察者事件而非钩子，因为需要"世界已就绪 + 相关系统已注册"，钩子在插入瞬间过早。

### 5.3 组合与依赖
```rust
// 伪代码：默认角色由组合构成，非继承
#[derive(Bundle)]
struct CharacterBundle {
    actor: Actor,                 // 标记
    pawn: Pawn,                   // require Actor
    movement: CharacterMovement,  // 移动能力
    transform: Transform,
    // require 机制自动补齐缺失依赖组件
}
```
- `require` 保证依赖闭包完整（缺 Transform 自动补默认）。
- 功能扩展 = 附加组件；不需要的功能不产生任何开销（无空 Tick、无虚表）。

### 5.4 层级与附着
- 场景层级复用内核 `ChildOf`/`Children` 关系 + Transform 传播。
- 逻辑关系（如"武器属于角色"）用自定义关系组件表达，避免污染 Transform 层级。

---

## 6. 游戏流程：GameMode / GameState / PlayerState

### 6.1 GameMode（规则，仅服务器权威）
- 表达为 `Resource` + 一组"规则系统"（`GameRules` 系统集）。
- 职责：出生点选择、玩家进入/离开、胜负判定、重开局。规则以可替换的 trait 对象或类型参数注入。
- 提供挂点事件：`OnPlayerJoin`/`OnPlayerLeave`/`OnMatchStart`/`OnMatchEnd`（观察者事件）。

### 6.2 GameState（全局共享状态，客户端可见）
- 复用 `prism_app::state` 做阶段机（Lobby/Warmup/Playing/PostMatch）。
- 附加 `GameState` 资源承载可复制的全局数据（比分、剩余时间）。
- 状态作用域实体：某阶段专属的实体随状态退出自动清理（复用 state_scoped）。

### 6.3 PlayerState（每玩家）
- 每个玩家一个 `PlayerState` 实体，持有分数、队伍、属性引用。
- 与 `Controller` 用关系连接（`Controls`/`ControlledBy`）。
- 与网络连接实体（`Connection`）用关系连接，支持热插拔（掉线保留 PlayerState）。

### 6.4 流程编排图
```
App 启动 → Plugin 注册 → GameMode 资源就绪
  → 进入 GameState::Lobby (state)
  → 玩家连接 → 生成 PlayerState → OnPlayerJoin 规则
  → GameState::Playing → GameMode 选出生点 → 生成 Pawn → Controller 附身 Pawn
  → 运行循环（Update/FixedUpdate 系统集）
  → 胜负条件满足 → GameState::PostMatch → 清理 state_scoped 实体 → 重开或退出
```

---

## 7. 控制与附身：Pawn / Controller / Character

### 7.1 附身模型（对标 UE Possess）
- `Controller`（玩家或 AI）与 `Pawn` 通过关系连接：`Possesses`（Controller→Pawn）/ `PossessedBy`（Pawn→Controller）。
- 一个 Controller 同一时刻附身一个 Pawn；切换即改关系（观察者广播 `OnPossess`/`OnUnpossess`）。
- 输入从 Controller 流向被附身 Pawn；AI 与玩家控制器共用同一 Pawn 接口，可无缝切换。

### 7.2 PlayerController
- 持有本地玩家的 `InputContext` 栈、相机管理、HUD/UI 关联。
- 仅在本地客户端存在完整输入；服务器侧存在权威副本用于校验。

### 7.3 AIController
- 持有黑板引用、行为树运行时（见第 14 节），把 AI 决策转成与玩家一致的输入意图（`MovementIntent`/`AbilityIntent`），复用同一 Pawn 执行路径。

### 7.4 Character 与移动
- `Character` = `Pawn` + `CharacterMovement`（胶囊体 + 地面检测 + 状态机：Walking/Falling/Swimming）。
- 移动通过"输入意图 → 运动求解 → 与物理耦合"三段式；`CharacterMovement` 的权威 CCT 落地（胶囊体求解、固定步 `(state,input,dt)->state` 纯函数、与物理确定性/快照路径对齐）见 `prism_character_controller_design_zh.md` 与 `prism_physics_design_zh.md`，本节只定义玩法侧移动意图与模式接口。
- 移动模式可插拔（自定义飞行/攀爬），对标 UE `UCharacterMovementComponent` 的 MovementMode，但用组件枚举 + System 分派。

---

## 8. Gameplay Tag 系统

### 8.1 目标
对标 UE `FGameplayTag`：分层字符串标签（如 `Ability.Skill.Fireball`、`State.Debuff.Stunned`），支持父子匹配、容器、查询。

### 8.2 实现
- **Interning**：字符串在加载期 intern 成 `GameplayTag(u32)`，运行时全部 O(1) 整数比较。
- **分层匹配**：预计算每个标签的祖先位集；`matches_tag`/`matches_any`/`matches_all` 均为位运算或前缀比较。
- **TagContainer**：紧凑位集/小向量，支持增删、集合运算。
- **组件化**：实体可挂 `GameplayTags` 组件；系统可用查询过滤"拥有某标签"。
- **注册表**：`GameplayTagRegistry` 资源，从 DataAsset 声明式加载标签树，支持热重载新增（不改已分配 id）。

### 8.3 用途
- 能力激活条件（`ActivationRequiredTags`/`BlockedTags`）。
- 效果分类与免疫（`ImmunityTags`）。
- 状态标记（眩晕/无敌/隐身）驱动系统 run-if。
- 事件路由（消息总线按标签订阅，见第 12 节）。

---

## 9. Gameplay Ability System（GAS 对标）

这是本框架的旗舰模块。目标：保留 UE GAS 的表达力（属性/效果/能力/线索/标签），去掉其运行时开销（无 UObject、无反射热路径、无深继承）与作者样板。

### 9.1 三大支柱

**(1) Attribute（属性）**
- 属性即组件字段（HP/Mana/AttackPower/MoveSpeed）。
- 每个属性区分 **BaseValue** 与 **CurrentValue**；CurrentValue = Base 经修饰器聚合后的结果。
- **AttributeAggregator**：把作用于该属性的所有 modifier 按 `Add→Multiply→Override` 顺序聚合，缓存脏标记，仅在变更时重算。
- 属性变更触发 `OnAttributeChanged` 观察者（用于 UI、死亡判定、连锁效果）。
- 元属性（Meta，如 Damage/Healing）：不持久化，作为一次性输入进入聚合管线，转化为对 HP 的实际增减。

**(2) GameplayEffect（效果）**
- 纯数据（DataAsset），描述"如何修改属性/标签"。三种时长模型：
  - **Instant**：立即改 BaseValue（伤害/治疗）。
  - **Duration**：限时修改 CurrentValue（buff/debuff），到期自动移除。
  - **Infinite**：持续到手动移除（装备加成）。
- 支持：周期性（每 N 秒结算一次）、层数堆叠（Stacking 策略：叠加/刷新/独立）、施加条件（tag 要求）、赋予/移除标签、触发 Cue。
- 运行时表达为"Effect 实例实体"，与目标实体用关系连接（`AppliedTo`/`ActiveEffects`），到期由计时系统回收。

**(3) GameplayAbility（能力）**
- 描述"一个技能怎么执行"：激活条件（tag/冷却/消耗）、执行阶段（状态机：Committed→Executing→Ended）、产生的 Effect、动画/特效挂点。
- 运行时是"能力实例组件 + 驱动 System 的小型状态机"，而非协程/蓝图字节码。
- 提供 **AbilityTask** 抽象（对标 UE AbilityTask）：等待事件、等待延时、等待目标确认、播放蒙太奇——用异步任务或 System 状态推进实现，避免每个能力手写状态机样板。
- 冷却/消耗本身用 GameplayEffect 表达（冷却=赋予冷却标签的 Duration 效果），统一模型。

### 9.2 承载者：AbilitySystem 组件（对标 ASC）
- 挂在拥有能力的实体上，聚合：`GrantedAbilities`、`ActiveEffects`（关系）、`AttributeSet`、`OwnedTags`。
- 提供 API：`grant_ability`/`try_activate`/`apply_effect_to`/`get_attribute`。

### 9.3 数据流
```
输入意图/AI → try_activate(Ability)
  → 检查 tag 要求 + 冷却 + 消耗
  → 提交消耗（应用 Cost Effect）→ 进入 Executing
  → AbilityTask 推进（播放动画/生成投射物/等待命中）
  → 命中 → 构造 Damage 元属性 → apply_effect_to(target, DamageEffect)
  → 目标 AttributeAggregator 结算 → HP 变更 → OnAttributeChanged
  → 触发 GameplayCue（特效/音效，见 9.4）→ 死亡判定
```

### 9.4 GameplayCue（表现层线索）
- 把"表现"（粒子/音效/震屏）与"逻辑"解耦：Effect/Ability 只发 Cue 事件（按 GameplayTag 标识），表现系统订阅播放。
- 本地/网络分离：逻辑在服务器，Cue 可在客户端本地预测播放，天然适配网络。

### 9.5 相对 UE GAS 的代差优势
| 维度 | UE GAS | Prism GAS |
|---|---|---|
| 属性存储 | UObject 反射字段 | ECS 组件（SoA，可 SIMD 批量结算） |
| 效果实例 | UObject 分配 | 实体 + 关系（零 GC，可批处理到期） |
| 能力执行 | 蓝图/协程 | System 状态机（并行、可确定性） |
| 样板 | 大量 boilerplate | DataAsset 声明 + 派生宏 |
| 热路径开销 | 反射/虚调用 | 静态分派 + 脏标记增量 |

---

## 10. 增强输入系统（Enhanced Input 对标）

### 10.1 抽象层次
- **InputAction**：抽象动作（`Move`(Vec2)/`Jump`(bool)/`Look`(Vec2)），与物理设备解耦。
- **InputMappingContext**：一组"设备输入→Action"的映射，可整体压栈/出栈（如进入载具切换上下文）。
- **Modifier**：对原始值加工（死区、灵敏度、取反、SwizzleAxis、标量缩放）。
- **Trigger**：定义 Action 触发时机（Pressed/Released/Hold(时长)/Tap/Combo/Chorded）。

### 10.2 运行时
- Controller 持有 InputContext 栈；每帧从上到下解析映射，产出 `ActionState`（Started/Ongoing/Triggered/Completed + value）。
- ActionState 转成"意图组件"（`MovementIntent`/`AbilityIntent`）写到被附身 Pawn，Gameplay 系统只读意图，与设备完全解耦。
- 支持输入缓冲（Buffer）与消费语义（一次性动作防重复触发）。

### 10.3 数据驱动
- 映射上下文、修饰器、触发器均为 DataAsset，可热重载、可重绑定（玩家自定义按键）。
- 与 Unity Input System 的 Action Maps / Control Schemes 心智一致，但零 GC、可确定性回放（录制 ActionState 即可重放）。

---

## 11. Subsystem / 服务定位

### 11.1 动机
替代全局单例/静态 Manager，提供"有生命周期、可依赖注入、可发现"的服务（对标 UE Subsystem）。

### 11.2 层级
- **AppSubsystem**：进程级，随 App 生命周期（如资源管理器、网络客户端）。
- **WorldSubsystem**：世界级，随 World 创建/销毁（如伤害仲裁、任务管理器）。
- **PlayerSubsystem**：玩家级，随 PlayerState（如玩家背包服务）。

### 11.3 实现
- Subsystem 本质是"带初始化/销毁钩子的 Resource / 每玩家组件"，通过 `SubsystemRegistry` 统一初始化顺序、依赖声明、可被 mod 替换实现。
- 提供 `world.subsystem::<T>()` 便捷访问，编译期类型安全。

---

## 12. 消息与事件总线

三种事件机制，按场景选用，避免"一把锤子"：

| 机制 | 内核基础 | 适用 | 特点 |
|---|---|---|---|
| **Observer/Trigger** | prism_ecs observer | 针对实体的即时响应（受击、附身） | 同步、可冒泡（traversal）、低延迟 |
| **Message（缓冲事件）** | prism_ecs message | 帧内批量、跨系统解耦 | 双缓冲、可并行读、有序 |
| **GameplayMessage 总线** | 自建于 observer + tag | 按 GameplayTag 广播的松耦合玩法消息 | 发布/订阅、UI 与逻辑彻底解耦，对标 UE GameplayMessageSubsystem |

- GameplayMessage 总线：`broadcast(tag, payload)` → 所有按 `tag`（含祖先）订阅者收到。UI、成就、音频等横切系统零耦合接入。
- 冒泡：命中事件可沿"武器→角色→队伍"关系链冒泡（复用内核 traversal）。

---

## 13. 数据驱动作者层：DataAsset / Prefab / 热重载

### 13.1 GameDataAsset（对标 ScriptableObject / DataAsset / DataTable）
- 任意 `Reflect + Serialize` 类型即可作为数据资产：能力定义、效果定义、属性初值表、输入映射、AI 行为、掉落表。
- 通过 `prism_asset` 加载，支持热重载：改文件即时生效（编辑期迭代）。
- DataTable 变体：行 = 结构化记录，主键索引，支持 CSV/JSON/RON 导入。

### 13.2 Prefab（对标 Prefab / Blueprint Class）
- 复用内核 **Scene + Template + Patch**：Prefab = 可重复 spawn 的 Template。
- **Patch/Override**：Prefab 变体（对标 Prefab Variant）通过 Scene Patch 表达差异，不复制全量。
- 嵌套 Prefab、Prefab 内引用（entity path template）均由内核 Template 支撑。

### 13.3 热重载与编辑器
- 结合 `prism_reflect`，Inspector 可读写任意组件字段（对标 Unity Inspector / UE Details）。
- 数据资产热重载 + 组件热替换，支撑"改数据不重启"的迭代闭环。

---

## 14. AI 框架：行为树 / 黑板 / 感知

### 14.1 Blackboard（黑板）
- 每个 AIController 一个黑板：类型化键值（目标、位置、阈值），可被行为树/EQS/技能共享读写。
- 黑板即组件集合或 typed map，变更可触发行为树重评估（事件驱动，避免每帧全量重跑）。

### 14.2 Behavior Tree（行为树）
- 数据驱动（DataAsset）：Selector/Sequence/Parallel/Decorator/Service/Task 节点。
- 运行时为"每 AI 一个轻量执行游标 + 共享只读树"，事件驱动 + 惰性 tick，避免深递归开销。
- Task 叶子把决策转成与玩家一致的输入意图（复用第 7 节路径）。

### 14.3 感知（Perception，对标 AIPerception）
- 视觉/听觉/伤害刺激源与感知组件；空间查询用物理/加速结构。
- 感知结果写黑板，触发行为树。

### 14.4 空间查询（EQS 对标，可选后期）
- 生成采样点 → 多维打分（距离/视线/掩体）→ 选优。作为可选高级模块。

---

## 15. 序列与编排：Timeline / Playable

- **Timeline**（对标 Unity Timeline / UE Sequencer）：时间轴轨道编排动画、音频、事件、相机、属性曲线。
- 数据驱动 DataAsset；运行时为"时间游标 + 轨道求值系统"，可暂停/缩放/倒放。
- **落地映射**：时间游标/轨道编排建于 `prism_time::timeline`，倒带/回放复用 `prism_time::recording`（见 `prism_time_design_zh.md`）；本节定义玩法侧轨道语义（能力触发、相机/输入接管）。
- **Cutscene/Cinematics**：接管相机与输入上下文，结束恢复。
- 与 GAS 联动：能力可触发 Timeline（连招/终结技演出）。

---

## 16. 网络复制与预测回滚

> 网络是 Gameplay 框架的一等公民，从设计初期即嵌入，而非事后补丁。**但复制/传输/预测回滚/延迟补偿/无缝迁移的权威实现属于 `prism_network`（见 `prism_network_design_zh.md`），本节只定义玩法侧与网络层的接缝与不变量，不复制其内核细节。**

### 16.1 分层立场（谁拥有什么）
- `prism_network` 拥有：传输（可靠 UDP/QUIC 多通道）、服务器权威仿真、兴趣管理（AOI）、快照 delta 双通道、预测与权威和解（rollback）、延迟补偿、服务器网格分片与权威移交、确定性对账。
- Gameplay 层拥有：**什么该复制、如何预测、回滚边界在哪**——把玩法状态（属性/效果/能力/移动意图）标注为可复制单元，并声明其预测/和解策略，交给 `prism_network` 执行。
- 权威持久状态复用 `prism_reflect::net_delta` + persist 的 override delta 语义（同一套序列化、同一套确定性合并），不另立网络专属真相源。

### 16.2 玩法侧接缝
- **复制标注**：组件级 `Replicated` 标记 + 复制策略（相关性/优先级/压缩），由 `prism_network` 的 provider 消费；兴趣过滤复用 `prism_ecs::partition::InterestGrid`。
- **预测/和解回路**：本地输入即时预测执行（移动/技能），服务器权威回传后由网络层比对；不一致时回滚到权威快照（复用 `prism_ecs::world::snapshot::SnapshotRing`）并重放本地输入（见 §19 确定性 + §33.1 预测 Key）。
- **属性/效果复制**：GAS 的 Effect 在服务器权威结算，`GameplayCue` 在客户端本地表现预测，天然分层省带宽。
- **RPC/消息**：可靠/不可靠 RPC 与跨网 `GameplayMessage` 广播经网络层通道投递。

### 16.3 与物理 / CCT 的接缝
- 移动预测回滚与 `prism_character_controller`（固定步 `(state,input,dt)->state` 纯函数）、`prism_physics` 的确定性/快照路径对齐（见 `prism_character_controller_design_zh.md` §16、`prism_physics_design_zh.md`）。
- 玩法状态推进（§3 Layer 3 FixedUpdate）与 `prism_app::fixed` + `prism_app::determinism` 共用同一时间步与种子化 RNG，保证回滚可复现。

---

## 17. 存档与持久化

- **反射驱动序列化**：任意 `Reflect` 组件可存档；存档 = 世界子集快照（Scene 序列化）。
- **版本迁移**：字段增删的迁移器（migration），旧存档可升级。
- **分层存档**：全局存档（进度/解锁）、关卡存档（世界状态）、玩家存档（背包/属性）。
- 与 Prefab/Scene 共用序列化基建，避免两套。

---

## 18. 可编程性：脚本与可视化蓝图

- **Rust 一等公民**：核心玩法用 Rust（性能 + 类型安全），派生宏消除样板。
- **可选图脚本层（Blueprint 对标）**：基于反射 + 节点图 DataAsset，编译/解释为 System 调用；面向策划快速迭代，不进热路径。
- **可选 WASM 脚本**：mod/热更场景，沙箱安全；通过稳定 ABI 访问受限 API。
- 分层原则：脚本层只调用 Layer 3/4 的稳定 API，不触碰内核热路径，保证性能不被脚本拖垮。

---

## 19. 确定性

- **确定性调度路径**：固定步长 + 全序系统 + 稳定实体迭代序（按稳定 id 排序）+ 定点/软浮点数值（可选）。
- **落地映射**：固定步走 `prism_app::fixed`，确定性开关走 `prism_app::determinism`（并与 `prism_time::determinism`、`prism_transform::determinism`、`prism_physics` 确定性路径对齐）；状态快照/回放走 `prism_ecs::world::snapshot`（`SnapshotRing`/`SnapshotDelta`）+ `prism_time::recording`。
- 编译期/配置切换：默认高性能非确定性；联机/回放开启确定性路径。
- 服务于：客户端预测回滚（§16）、录像回放（§34）、联机同步、自动化测试可复现（§38.4）。
- 与物理确定性路径共享同一时间步与随机数源（种子化 RNG 作为资源/组件，禁用非确定性全局随机）。

---

## 20. 性能策略

- **数据导向**：属性/效果 SoA 布局，批量结算可 SIMD；到期效果批处理回收。
- **无每对象 Tick**：逻辑集中在 System，按需 run-if；不需要的功能零成本。
- **脏标记增量**：属性聚合、行为树、复制均增量计算，避免全量每帧重跑。
- **并行调度**：读写不冲突的系统自动并行（内核能力）；能力/AI 大量实体天然并行。
- **关系代替查找**：附身/效果/所属用关系直连，避免线性搜索与哈希查找。
- **分配控制**：效果/能力实例复用池化实体；事件双缓冲；避免热路径堆分配。
- **Cue/表现与逻辑分离**：表现可在低频/客户端本地，逻辑保持紧凑。

---

## 21. 易用性分层与默认体验

对标 Bevy "简单又灵活"，提供三档体验：

- **零配置档**：`app.add_plugins(GameplayDefaultPlugins)` + spawn 一个 `CharacterBundle` → 立即获得可移动、可跳跃、可被玩家控制、可受击的角色。
- **配置档**：通过 DataAsset 调能力/属性/输入映射，不写代码即可改玩法数值与手感。
- **下钻档**：替换移动求解、自定义能力状态机、自定义复制策略、接管调度——每层都有稳定 trait 扩展点。

设计红线：任何默认行为都可被覆盖；任何高层封装都不隐藏底层 ECS（用户随时可直接操作组件/系统）。

---

## 22. Crate 拆分与落地形态

> 遵循仓库"一子系统一 crate"惯例，玩法框架落在 `pkg/prism_gameplay_*`（与 `prism_engine_component_gap_zh.md` 中登记的 `prism_gameplay` 对齐）。所有 crate 构建于 prism 原生内核之上，不依赖任何 `bevy_*` crate。

```
pkg/
  prism_gameplay_core      # Layer2: Actor/生命周期/确定性调度骨架/gameplay 事件模型（仅依赖 prism_ecs）
  prism_gameplay_tags      # Layer3: GameplayTag 注册表与容器（interned 分层标签）
  prism_gameplay_input     # Layer3: 增强输入（Context/Action/Modifier/Trigger，建于 prism_input 之上）
  prism_gameplay_abilities # Layer4: GAS（Attribute/Effect/Ability/Cue/ASC）
  prism_gameplay_flow      # Layer4: GameMode/GameState/PlayerState/Pawn/Controller/Character（建于 prism_app::state）
  prism_gameplay_ai        # Layer4: 黑板/行为树/StateTree/感知/EQS/Smart Object
  prism_gameplay_sequence  # Layer4: Timeline/Playable（建于 prism_time::timeline）
  prism_gameplay_save      # Layer3: 存档/持久化/迁移（建于 prism_reflect 序列化）
  prism_gameplay_subsystem # Layer3: Subsystem 注册与生命周期
  prism_gameplay_message   # Layer3: GameplayMessage 总线（建于 prism_ecs event/observer）
  prism_gameplay_features  # Layer3/4: Game Features 运行时插拔 + 组件注入器（§26）
  prism_gameplay_script    # Layer5: 图脚本/WASM（可选 feature）
  prism_gameplay_interaction # Layer4: 交互机制（检测管线 + Interactable 契约，复用 GAS / §29 Smart Object）
  prism_gameplay_feel      # Layer4: 手感机制 + 命中判定基建（hitbox/hurtbox/输入缓冲/i-frame；表现经 Cue/Message 下放）
  prism_gameplay_settings  # Layer3: 游戏设置框架（建于 prism_app::settings/cvar + prism_reflect + §35 MVVM）
  prism_gameplay           # 门面聚合 + GameplayDefaultPlugins（对标 bevy_internal / UE Gameplay 模块组）
```
- 网络复制/预测由 `prism_network` 承载（不设 `prism_gameplay_net`），玩法只提供 §16 的复制标注与预测/回滚策略接缝。
- 每个 crate 独立可用、独立 Plugin，遵循内核 modular 原则。
- `prism_gameplay_core` 不依赖 `prism_ecs` 之外的东西，可被专用服务器逻辑复用。
- 门面 crate 提供"零配置能跑"的默认插件组。

---

## 23. 公共 API 草案

```rust
// 1) 零配置起步
App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(GameplayDefaultPlugins)  // tags/input/abilities/flow/...
    .run();

// 2) 生成一个开箱即用角色
commands.spawn((
    CharacterBundle::default(),
    GameplayTags::from(["Team.Blue", "Class.Warrior"]),
));

// 3) 授予并激活能力（GAS）
asc.grant_ability(fireball_ability);          // fireball_ability: Handle<AbilityDef>
asc.try_activate(tag!("Ability.Skill.Fireball"));

// 4) 定义效果（DataAsset，声明式）
GameplayEffectDef {
    duration: Duration::Instant,
    modifiers: vec![Modifier::add(attr!("Health"), -50.0)],
    granted_tags: tags!["State.Hit"],
    cue: Some(tag!("Cue.Impact.Fire")),
    ..default()
}

// 5) 输入映射（DataAsset）
InputMappingContext::new()
    .bind(Key::W, action!("Move"), [Modifier::swizzle_y()])
    .bind(GamepadButton::South, action!("Jump"), [Trigger::pressed()]);

// 6) 附身
controller.possess(pawn_entity);   // 触发 OnPossess 观察者

// 7) 订阅玩法消息（UI 零耦合）
messages.subscribe(tag!("Player.Damaged"), |msg: &DamageMsg, ui: &mut Hud| { ... });

// 8) Subsystem
let inv = world.subsystem::<InventorySubsystem>();

// 9) 下钻：自定义移动模式
impl MovementMode for GrapplingHook { fn solve(&mut self, ctx: MoveCtx) -> MoveResult { ... } }
```

---

## 24. 路线图

- **M0 内核就绪校验**：确认 relationship/observer/hook/state/template 满足需求，补齐 gameplay 事件模型与确定性调度骨架（`prism_gameplay_core`）。
- **M1 基础玩法闭环**：GameplayTag + Enhanced Input + Pawn/Controller/Character + 零配置角色可跑可控。
- **M2 GAS**：Attribute/Effect/Ability/Cue/ASC，跑通"技能→伤害→死亡→表现"完整链路。
- **M3 流程与服务**：GameMode/GameState/PlayerState、Subsystem、GameplayMessage 总线。
- **M4 AI**：黑板 + 行为树 + 感知，AIController 复用玩家执行路径。
- **M5 网络**：对接 `prism_network`（组件复制 + 客户端预测 + 回滚）与 `prism_anticheat`，玩法侧提供复制标注与预测/回滚策略，与 GAS/移动联动。
- **M6 编排与作者层**：Timeline、DataAsset/Prefab 编辑器闭环、热重载、存档。
- **M7 可编程性**：图脚本/WASM 可选层，稳定 ABI。
- **M8 打磨**：性能基线（万级实体能力结算）、确定性联机压测、示例游戏（第三人称动作 Demo）。

---

## 25. 关键扩展点清单

- `MovementMode` trait：自定义移动/飞行/攀爬。
- `AbilityTask`：自定义能力异步阶段。
- `EffectModifier` / `Aggregator`：自定义属性聚合规则。
- `GameRules` trait：自定义胜负/出生/进入规则。
- `ReplicationPolicy`：自定义相关性/优先级/压缩。
- `BehaviorNode`：自定义行为树节点。
- `InputModifier` / `InputTrigger`：自定义输入加工与触发。
- `Subsystem` trait：注入自定义世界/玩家级服务。
- `Cue Handler`：自定义表现响应。
- `SaveMigration`：自定义存档版本迁移。

---

# 扩展篇 II：Next-Gen 高级功能

> 本篇为超出"基础玩法闭环"的进阶差异化能力。每个模块都独立可选（feature/Plugin），默认关闭不产生成本，与主篇的分层与扩展点原则一致。

---

## 26. 模块化玩法：Game Features 运行时插拔（旗舰）

对标 UE **Game Features & Modular Gameplay**：把一整块玩法（一张地图的活动、一个赛季内容、一个 DLC、一个 mod）打包成"可在运行时激活/停用"的功能单元，且能向**已存在的 Actor**注入组件/能力/输入映射，而无需修改这些 Actor 的定义。

### 26.1 核心构件
- **GameFeature**：一个可加载单元 = 一组 Plugin + DataAsset + 资源清单 + 激活/停用钩子。
- **GameFeatureAction**：激活时执行的动作（注册能力、添加输入上下文、生成管理器实体、注册 Cue 处理器）。
- **ComponentInjector（组件注入器）**：声明"对所有匹配 `Query<With<Character>>` 的实体，激活期间注入组件 X / 授予能力 Y / 压入输入上下文 Z"，停用时自动回滚。
- **FeatureState**：`Registered → Loading → Active → Deactivating`，可热切换。

### 26.2 实现要点
- 注入基于内核**关系 + 生命周期钩子**：功能激活时对匹配实体建立"注入来源"关系，停用时按关系批量撤销，保证幂等与可回滚。
- 后加入的实体也会被"追加注入"（观察者监听 `on_add`），保证一致性。
- 与热重载、mod、A/B 内容实验天然契合：开一个 feature 即一个内容切片上线。

### 26.3 价值
- 内容团队并行开发互不干扰；主干精简，玩法按需拼装。
- 是"次世代"相对传统单体工程的关键工程化优势。

---

## 27. 大规模实体与人群（Mass 对标）

对标 UE **Mass Entity / MassAI**、Unity DOTS 群体：以数据导向方式仿真数千至数十万个轻量代理（NPC、鸟群、士兵、粒子化生物）。

> **落地映射**：升降格、LOD 分级、休眠直接复用 `prism_ecs::partition` 的 `EntityLodProcessor`（LOD 分档 + `PhasePolicy` 分帧）、`Dormant`/`DormancySet`（休眠远处代理）、`InterestGrid`（兴趣）。本节只描述玩法语义，底座不重复实现。

### 27.1 双层实体模型
- **轻量层（Mass）**：群体代理只持有极简 SoA 数据（位置/速度/目标/状态枚举），批量并行推进，无独立能力/物理。
- **完整层（Actor）**：玩家附近或交互中的代理"升格"为完整 Actor（挂能力/物理/动画），远离后"降格"回轻量层。
- 升/降格是组件迁移（archetype move），由 LOD 系统驱动，状态无损。

### 27.2 LOD 分级仿真
- 按到玩家/相机距离分档：近处每帧完整仿真，中距低频，远处仅统计级推进（"这片区域大致发生了什么"）。
- 时间片轮转：远处代理错帧更新，均摊 CPU。

### 27.3 群体行为
- Flow Field / 导航流场做大规模寻路（避免逐个 A*）。
- RVO/ORCA 或密度场做避障；编队、跟随、羊群行为。
- 与行为树/StateTree 分层：群体用轻量 FSM，升格后接完整 AI。

---

## 28. 大世界：World Partition / Data Layers / HLOD / 后台仿真

对标 UE5 **World Partition + Data Layers + HLOD**、Unity 大世界流式。

> **落地映射**：本节能力直接复用 `prism_ecs::partition` 原语——`WorldPartition`（Cell）、`DataLayers`、`Hlod`/`HlodProxyId`、`FloatingOrigin`/`GridCell`（大世界精度，配合 `prism_transform::large_world`）、`InterestGrid`、`CellEntityIndex`/`WeakRefs`（流送）、`Dormant`（休眠）、`EntityLodProcessor`（LOD 分级）。空间组织/流送编排的权威设计见 `prism_world_system_design_zh.md` 与 `prism_terrain_world_design_zh.md`；本节只描述玩法语义（升降格、后台抽象仿真），不复制底座。

### 28.1 World Partition（空间分区流式）
- 世界按网格/八叉树分 Cell，按玩家位置与预测轨迹异步流入/流出实体与资源。
- 无缝无 loading 界面；流式在 `prism_tasks` 后台线程完成，主线程零卡顿地"提交"就绪 Cell。
- 与 Prefab/Scene Patch 结合：Cell 内容即一批可序列化实体。

### 28.2 Data Layers（数据层）
- 同一空间叠加多套内容层（昼/夜版本、剧情前/后、难度变体），运行时切换某层可见/激活。
- 与 Game Features 联动：一个 feature 可挂一个 Data Layer。

### 28.3 HLOD（层级化 LOD）
- 远处 Cell 用合并代理（合并网格 + 简化逻辑）表示，近处替换为完整 Cell。
- 玩法侧同理：远处村庄用"聚合状态"表示，靠近时展开为个体 NPC。

### 28.4 后台世界仿真
- 未加载区域的经济/生态/派系战争以低频"抽象仿真"推进（无实体，仅统计模型），玩家回到时按结果重建。
- 让开放世界"离开也在运转"，是次世代活世界的关键。

---

## 29. AI 进阶：StateTree / Smart Objects / 群体导航

### 29.1 StateTree（对标 UE StateTree）
- 分层状态机 + 选择器的融合：比纯行为树更高效、更可预测、更易调试。
- 事件驱动状态转移 + 数据驱动（DataAsset），运行时为轻量游标，适合大规模代理。
- 与行为树并存：BT 擅长复杂决策，StateTree 擅长清晰的状态生命周期，二者可嵌套。

### 29.2 Smart Objects（智能物体 / 世界契约）
- 环境交互点声明"我能提供什么交互"（椅子=坐下、门=开、炮塔=操作），并附带槽位、条件标签、动画契约。
- AI 与玩家**共用同一契约**：AI 查询附近 Smart Object 满足需求（找椅子休息），玩家走近同一物体触发交互。
- 槽位预定（claim）避免多代理争用；与 GameplayTag 需求匹配。

### 29.3 群体导航
- 导航网格（NavMesh）+ 流场 + 分层寻路（区域图 + 局部细化）。
- 动态障碍/门/电梯用导航连接（NavLink），支持跳跃/攀爬语义。

---

## 30. 交互与世界契约

> **定位**：本节属 base 框架——提供交互*机制*（检测管线 + 契约 + 执行复用 GAS），不预设具体交互内容。具体可交互对象、动词、提示文案与数值由游戏层以 DataAsset 配置（内容属游戏模板层）。

**数据模型**
- `Interactable`：挂在可交互实体上，声明候选条件（GameplayTag 门控）、交互原型引用（指向一个能力）、提示/耐时等参数（供表现层读取，不含 UI 实现）。
- `Interactor`：发起者组件，持有当前焦点候选与交互状态。
- **交互 = 一次 GameplayAbility**：复用 §9 GAS 的授权/消耗/冷却/网络管线——交互即"对目标施加 Effect / 触发 Cue / 转移状态"，不另造执行通道。

**检测与执行管线**
- 焦点/瞄准检测：射线 / 形状查询（建于 `prism_physics` 查询，玩法侧只做候选筛选）→ 按 `Interactable` 条件 Tag 过滤 → 产出高亮候选。
- 高亮/提示：通过 §12 GameplayMessage 广播候选变化，表现层（HUD/高亮）订阅渲染，玩法不耦合 UI。
- 确认 → 激活能力：支持长按、多阶段充能、可打断（Tag 门控中断）。

**契约统一与网络**
- 玩家侧 `Interactor` 与 AI 侧 Smart Object（§29）走**同一执行体**：Interaction 是玩家入口，Smart Object 是 AI 入口，底层同一能力/StateTree。
- 多人并发交互的授权由服务器权威裁定；交互请求 = 能力激活，走 `prism_network` 的预测/回滚（§16/§33），不新增网络通道。

> **落地映射**：机制落 `pkg/prism_gameplay_interaction`（可按需并入 `prism_gameplay_abilities`）；表现（高亮/提示 UI）经 §12 消息 + §35 MVVM 下放。**边界**：具体交互目录、动词语义、提示内容属游戏模板层，不在 base 范围内。

---

## 31. 叙事与进程：任务 / 对话 / 阵营声望（游戏模板层，非 base 框架）

> **范围说明**：类比 UE，本文档对标**引擎级 Gameplay 框架**（GameFramework/GAS/Enhanced Input/AI/Mass/World Partition/复制…），而 UE 引擎本身**不内置**任务/对话/阵营声望——这些属于**游戏模板与内容系统层**（Lyra 等价层，由具体游戏自建）。因此本节不在 base 框架范围内，仅保留编号以维持交叉引用稳定。此类系统应**基于**本框架的 §8 GameplayTag、§9 GAS、§12 GameplayMessage、§13 DataAsset、§17 存档搭建，归入独立的游戏模板/内容系统文档 `prism_game_creator_design_zh.md`（Game Creator 套件，本节的正式落地见其 §10）。

---

## 32. 物品：背包 / 装备 / 掉落表（游戏模板层，非 base 框架）

> **范围说明**：同 §31，物品/背包/装备/掉落表属于**游戏模板与内容系统层**（Lyra 等价层），UE 引擎的 Gameplay 框架本身不内置。本节不在 base 框架范围内，仅保留编号。此类系统应**基于**本框架的 §9 GAS（装备 = Infinite Effect + 能力注入）、ECS 关系（`ContainedBy`/`Contains`）、§13 DataAsset、§17 存档搭建，归入独立的游戏模板/内容系统文档 `prism_game_creator_design_zh.md`（Game Creator 套件，本节的正式落地见其 §9）。

---

## 33. 网络进阶：预测 Key / Lag Compensation / 双模 / 无缝迁移

> 以下为玩法侧需要的网络能力**契约**；传输/定序/分片/对账的权威实现见 `prism_network_design_zh.md`，命中回溯的审计与反作弊接缝见 `prism_anticheat_design_zh.md`。本节只描述"玩法如何使用"。

### 33.1 GAS 预测（Prediction Key，对标 UE GAS 预测）
- 客户端激活能力时生成预测 key，本地预测应用效果/Cue；服务器确认后合并，拒绝则回滚该 key 关联的所有预测变更（回滚复用 §16.2 快照回路）。
- 让技能"零延迟手感"同时保持服务器权威。

### 33.2 Lag Compensation（命中回溯）
- 服务器保存近 N 帧实体位置历史（可复用 `SnapshotRing`）；处理客户端命中判定时"回溯"到该客户端当时看到的世界状态做判定，补偿延迟。回溯窗口与审计日志对接 `prism_anticheat`。
- 对标竞技射击的服务器回退命中。

### 33.3 双模网络内核
- **Rollback**：适合快节奏对战（预测 + 回滚重放），依赖确定性路径（§19）。
- **Lockstep**：适合大规模同步 RTS（只传输入，全端确定性推进）。
- 二者共用 `prism_app::determinism` 确定性调度与 `prism_ecs` 快照基建，按玩法选择；具体传输由 `prism_network` provider 实现。

### 33.4 无缝迁移（Seamless Travel / Server Handover）
- 换图/跨服不断线：玩家连接与 PlayerState 在服务器间迁移，客户端无缝过渡。
- 与 `prism_ecs::partition::WorldPartition` 结合支撑"无缝大世界分服"（权威移交实现见 `prism_network`）。

---

## 34. 时间操控与确定性回放

### 34.1 时间操控
- 全局/局部 **TimeDilation**（子弹时间、局部时缓/时快区域），逻辑与物理时间步统一缩放。
- **Rewind/倒带**：环形缓冲记录关键实体状态，支持回退（时间系机制、死亡回溯）。
- 与命中停顿（hitstop）、慢动作演出联动。

### 34.2 确定性录像与回放（Demo/Replay）
- 基于确定性路径：只记录初始状态 + 每帧输入（ActionState），即可精确重放整局。
- 用途：观战、精彩集锦、bug 复现、平衡性分析、反作弊回放审查。
- 回放可暂停/快进/自由镜头/时间刻度调节。

### 34.3 时间旅行调试
- 开发期记录世界快照序列，可在调试器中前后拖动时间轴检视任意帧的实体状态（配合第 38 节 Visual Logger）。

---

## 35. 响应式玩法与 UI 数据绑定（MVVM 对标）

> 本节的运行时框架化落地(应用外壳/绑定层/表现树/异步流送)见 **Loom Runtime**:`prism_loom_runtime_framework_design_zh.md`。玩法算法在本文(ECS 权威),表现与绑定在 Loom Runtime,二者单向、分工互补。

### 35.1 响应式玩法管线
- 属性/标签/状态变更以**观察者 + 脏标记**驱动下游派生（如"HP 变化 → 更新血条 → 检查处决阈值 → 触发濒死状态"），避免每帧轮询。
- 派生状态（Computed）：由基础状态自动重算的只读状态（复用 `ComputedStates` 思路扩展到任意数据）。

### 35.2 UI 数据绑定（对标 UE MVVM ViewModel / Unity 数据绑定）
- **ViewModel** 层：把玩法数据映射为 UI 友好视图模型，UI 只绑定 ViewModel 字段，变更自动推送。
- 双向绑定可选（设置界面），单向为主（HUD），零手写"每帧刷新 UI"样板。
- 与 GameplayMessage 总线结合：UI 订阅消息驱动，逻辑与表现彻底解耦。

---

## 36. 程序化内容（PCG）与程序化叙事

- **PCG 图**（对标 UE PCG）：数据驱动的程序化生成管线（散布植被、生成地牢、布置敌人），可运行时/编辑期执行，结果可缓存与流式。
- 与 World Partition 结合：Cell 内容可程序化生成，减少手工摆放。
- **程序化叙事/事件**：基于世界状态与规则动态生成遭遇、事件、动态任务（emergent gameplay）。
- 种子化确定性生成，保证多端/回放一致。

---

## 37. 手感与命中判定层

> **定位**：本节属 base 框架——提供手感*机制与命中判定基建*，不预设品类手感。具体数值（缓冲时长 / hitstop 时长 / 连招窗口 / 削韧阈值）由游戏层 DataAsset 调参；动作游戏向的默认（coyote time / i-frames / 辅助瞄准）做成**可选 feature、默认关闭**，不把动作品类假设强加给所有游戏。

**输入手感（机制）**
- input buffering（预输入）、跳跃缓存、连招输入窗口：建于 §10 增强输入（`prism_gameplay_input`）的 trigger/buffer 能力。
- coyote time（离台缓冲）：作为可选输入修饰器，默认关闭。

**命中判定（基建）**
- hitbox / hurtbox：独立于物理碰撞的*玩法判定体*（几何查询建于 `prism_physics`，语义归玩法层）。
- 多段命中去重、命中优先级、无敌帧（i-frames）= GameplayTag 状态（§8/§9），护甲 / 削韧（poise）= GAS Attribute。
- 判定结算走确定性路径（§19）：客户端预测命中、服务器权威确认，Lag Compensation 见 §33。

**命中反馈（下放表现层）**
- hitstop（命中卡顿）、屏幕震动、后坐力、受击顿帧：经 §9 GameplayCue + §12 Message 触发，具体表现下放——镜头冲击见 `prism_camera_engine_design_zh.md`、VFX 见 `prism_particle_engine_design_zh.md`。
- 注记：hitstop/震屏多为**本地表现**（不占网络通道），命中**判定权威**在服务器。

**曲线资产**
- 伤害随距离/蓄力/时间的曲线（DataAsset，§13），可视化编辑、热重载调手感；全部数据驱动，策划改数值不改代码。

> **落地映射**：机制落 `pkg/prism_gameplay_feel`（输入手感 + 命中判定基建；表现经 Cue/Message 下放）。**边界**：具体数值、连招表、品类专属手感属游戏模板层。`prism_character_controller_design_zh.md` 引用本节为"手感层"。

---

## 38. 可观测性与调试：Gameplay Debugger / Visual Logger / 实时调参 / 自动化测试

> **落地映射**：建于 `prism_diagnostic` 之上——`profiler`/`span`/`trace`（分系统耗时）、`budget`（帧预算守卫，§47）、`hitch`（尖刺检测）、`replay`（事件回放）、`telemetry`/`remote`（远程面板）、`metrics`。Inspector/反射调参走 `prism_reflect`，热重载走 `prism_asset`。

### 38.1 Gameplay Debugger（对标 UE Gameplay Debugger）
- 分类别（AI/能力/网络/输入/属性）的运行时叠加显示：选中实体即看其黑板、激活能力、当前属性、当前状态、复制状态。
- 世界空间可视化：感知范围、导航路径、Smart Object 槽位、命中判定体。

### 38.2 Visual Logger（时间轴事件回溯）
- 记录带时间戳的结构化玩法事件与形状/文本快照，可在时间轴上回放"这个 AI 当时为什么这么决策"。
- bug 复现利器：出问题回看事件流而非加断点。

### 38.3 实时调参（Live Tuning）
- Inspector 直接改属性/曲线/DataAsset 字段实时生效（反射驱动）；配合热重载改数据不重启。
- 参数集可导出为 DataAsset，调好即固化。

### 38.4 自动化玩法测试
- 基于确定性路径：脚本化输入序列 → 断言世界状态；回放即回归测试。
- 无头(headless)运行，CI 中跑玩法回归、平衡性验证、性能基线（万级实体能力结算耗时）。

---

## 39. 高级 ECS 玩法机制：Aspect 组合 / 批量结算 / GPU-driven

### 39.1 Aspect（对标 Unity DOTS Aspect）
- 把"一组常一起访问的组件 + 便捷方法"封装成 Aspect（如 `MovementAspect` = Transform+Velocity+MovementConfig），提升 ECS 代码可读性，同时保持零成本。

### 39.2 Archetype 批量结算
- 相同能力/效果的实体天然聚集在同一 archetype，按 archetype 批量结算属性聚合与效果到期，SIMD 友好，cache 友好。

### 39.3 GPU-driven 玩法（前沿/可选）
- 海量同质代理（粒子化生物、弹幕、群体）的移动/避障/生命周期可下放 GPU compute（wgpu），与 Prism 渲染同后端。
- CPU 只管理"需要交互"的少数升格实体，GPU 跑其余统计级仿真，回读用于稀疏事件。

### 39.4 关系图算法
- 基于内核关系构建玩法图（编队树、传送网络、电路/管线、派系关系），提供图遍历/连通性/最短路的通用玩法算法层。

---

# 扩展篇 III：AAA 级表现、性能与整合

> 本篇把玩法框架拔高到顶级次世代 AAA 的"效果 + 性能"双达标线。表现子系统（动画/VFX/音频/相机/演出）与第二篇 GAS/Cue/手感层深度咬合，并给出可量化的帧预算与验收清单。
> 渲染与物理细节分别见 `prism_rendering_architecture_zh.md` 与 `prism_physics_design_zh.md`，本篇只谈"玩法如何驱动表现"与"整帧整合"。

---

## 40. AAA 质量基线与对标产品

### 40.1 对标产品与借鉴点
| 产品 | 借鉴维度 |
|---|---|
| UE5（黑神话悟空 / Hellblade II / Fortnite） | Nanite/Lumen 画质基线、MetaHuman、Chaos 破坏、大规模内容工程 |
| God of War: Ragnarök | 无缝一镜到底相机、分层动画与战斗手感、过场无缝切入 |
| The Last of Us Part II | 情境动画覆盖、程序化面补、细腻命中反馈、音频叙事 |
| Marvel's Spider-Man | 无缝流式大世界、运动中无 loading、运动匹配位移 |
| Elden Ring / 只狼 | 动作判定帧、削韧/弹反、i-frame、boss 多阶段编排 |
| DOOM Eternal | 120fps 稳帧的性能工程、极致输入响应、资源回收手感 |
| Horizon Forbidden West | 大规模机械体 AI、程序化植被、群体与生态 |
| Overwatch / Valorant / CS2 | 网络预测 + 命中回溯 + sub-tick、技能表现与逻辑分离 |
| Destiny 2 | 大规模 PvE 实体、能力系统（GAS 式）、无缝流式 |

### 40.2 AAA 效果达标线（玩法侧）
- **响应**：输入到画面反馈 ≤ 1 帧逻辑延迟；本地可预测动作 0 感知延迟。
- **动画**：无脚底打滑、无生硬切换、全身 IK 贴合地形、命中有分层反馈，支持运动匹配位移。
- **反馈**：每次命中/技能有 VFX + 音频 + 相机 + hitstop 的多通道协同反馈（"juice"）。
- **一致性**：逻辑在服务器权威，表现可本地预测，不一致时回滚无"橡皮筋"观感。
- **规模**：战斗场景同屏数百~数千可交互/半交互单位，画质与帧率不塌。

### 40.3 AAA 性能达标线
- 目标：主机/PC 60fps 稳帧（16.6ms），竞技模式 120fps（8.3ms），掌机/移动 30–60fps 降级可运行。
- 玩法逻辑（不含渲染/物理）帧预算 ≤ 3–4ms；其余留给渲染/物理/音频。
- 零 GC 卡顿、零主线程阻塞式加载、无帧级尖刺（p99 帧时达标，而非平均）。

---

## 41. 动画系统：Motion Matching / 分层 / IK / Root Motion

对标 UE5 Animation Blueprint + Motion Matching、育碧 Motion Matching、GoW 分层动画。动画是 AAA 手感的半壁江山。

> **落地映射**：动画评估/Motion Matching/分层混合/IK/Root Motion 的权威实现见 `prism_animation_engine_design_zh.md`；本节只定义玩法侧如何驱动动画（能力→蒙太奇、AnimNotify→命中帧、位移权威约定）。

### 41.1 架构
- **数据导向动画评估**：骨骼姿势 SoA，蒙皮矩阵批量计算，评估可并行、可 Job 化、可 LOD（远处降采样/降骨骼）。
- **异步评估**：动画求值在 `prism_tasks` 并行跑，主线程只做提交，避免阻塞玩法。
- **动画图（AnimGraph）**：状态机 + 混合空间 + 分层(Layer) + 混合树，数据驱动（DataAsset），可热重载调手感。

### 41.2 Motion Matching
- 用动作数据库 + 运动特征匹配，按当前速度/朝向/轨迹实时选最佳姿势片段，天然消除脚底打滑与生硬转向。
- 轨迹来自输入意图预测（与第 10 节输入意图统一），与 CharacterMovement 位移对齐（Root Motion warping）。

### 41.3 分层与叠加
- 下半身移动 + 上半身持武/瞄准/受击可独立分层叠加（对标 GoW/TLOU）。
- 加性动画（受击抖动、呼吸、后坐力）叠加在基础姿势上。

### 41.4 IK 与程序化修正
- 足部 IK（贴合坡面/台阶）、手部 IK（握武器/攀爬点）、注视 IK（看向目标）、全身 IK（Full-Body IK）。
- 命中方向程序化反应（受击位移/面补），减少逐个制作命中动画的成本。

### 41.5 Root Motion 与位移权威
- Root Motion 驱动精确位移（处决/翻滚/攻击前冲），但网络下以胶囊体权威为准、动画为表现，避免预测分叉。
- Motion Warping：动态拉伸 Root Motion 对齐目标（翻越不同高度、处决对位）。

### 41.6 与 GAS/手感整合
- 能力激活播放蒙太奇；**AnimNotify**（动画通知）在精确帧触发命中判定、生成投射物、施加 Effect、播放 Cue——判定与动画帧对齐（对标格斗/动作游戏的"出招帧"）。
- 命中停顿(hitstop)直接作用于动画时间刻度。

---

## 42. VFX 特效：GPU 粒子与 Cue 整合

对标 UE Niagara / Unity VFX Graph。

> **落地映射**：GPU 粒子仿真/发射器图/LOD 的权威实现见 `prism_particle_engine_design_zh.md`；本节只定义玩法侧如何经 GameplayCue 触发与回灌特效。

- **GPU 粒子**：粒子仿真跑 GPU compute（wgpu，与渲染同后端），百万级粒子不占玩法 CPU 预算。
- **数据驱动特效图**：发射器/模块化节点（DataAsset），美术可编辑、可热重载。
- **与 GameplayCue 整合**：Effect/Ability 只发 Cue（按 GameplayTag），VFX 系统订阅播放；本地可预测播放、网络只传 Cue 事件（省带宽）。
- **事件化特效**：粒子命中/生命周期事件可回灌玩法（如火花落地点燃），形成表现→玩法的反馈闭环（可选、受控）。
- **LOD 与剔除**：按距离/屏占比降级粒子数与更新频率；视锥外停更。

---

## 43. 音频：空间化 / 交互混音 / 程序化（MetaSound 对标）

对标 Wwise/FMOD、UE MetaSounds。

> **落地映射**：空间化/HRTF/混音/程序化音频图的权威实现见 `prism_audio_engine_design_zh.md`（及 `pkg/prism_audio_*` 子系统族）；本节只定义玩法侧如何经 Cue/事件驱动音频。

- **程序化音频图（MetaSound 对标）**：节点式音频合成/处理（DataAsset），参数由玩法实时驱动（速度→引擎音高、生命值→心跳）。
- **空间音频**：HRTF、遮挡/阻挡（occlusion/obstruction）、混响区（reverb zone）、距离衰减曲线。
- **交互式混音**：按游戏状态动态混音（战斗压低环境音、濒死低通滤波、过场 ducking）。
- **与 Cue/事件整合**：音效同样走 Cue 总线，与 VFX/相机同一时刻协同触发；音频可本地预测。
- **性能**：声源虚拟化（远处/超量声源降级为虚拟声），语音池化，混音在音频线程，零玩法阻塞。

---

## 44. 相机与运镜（Cinemachine 对标）

对标 Unity Cinemachine、God of War 无缝相机。

> **落地映射**：相机栈/混合/构图/碰撞避让/抖动的权威实现见 `prism_camera_design_zh.md`；本节只定义玩法侧如何经事件（技能/处决/boss 登场）驱动运镜。

- **相机栈 + 混合**：多个虚拟相机按优先级/事件激活，自动平滑混合（战斗/锁定/过场/载具相机切换无跳切）。
- **程序化构图**：目标跟随 + 构图框（framing）+ 预测前瞻（look-ahead）+ 死区；锁定目标自动构图。
- **碰撞避让**：相机与几何碰撞自动拉近/避让，防穿墙（spring arm）。
- **程序化抖动**：命中/爆炸/脚步的相机冲击，叠加 + 衰减曲线（与手感层第 37 节统一）。
- **玩法事件驱动**：技能/处决/boss 登场触发运镜事件；与 GAS/Timeline 联动。
- **一镜到底支持**：无缝从玩法切入过场再切回（对标 GoW），不黑屏不切镜。

---

## 45. 过场与叙事演出整合

- 深化第 15 节 Timeline：过场可**无缝接管**玩法（接管相机/输入上下文/动画层），结束无缝交还，支持可跳过与运行时数据绑定（过场里出现玩家当前装备/外观）。
- **互动过场（QTE / 可操作演出）**：过场中保留受限输入，复用增强输入上下文栈。
- **动态叙事演出**：根据世界状态/阵营/任务进度选择演出变体（程序化叙事，呼应第 36 节）。
- **多轨协同**：动画 + VFX + 音频 + 相机 + 后处理 + 玩法事件在同一时间轴精确对齐。

---

## 46. 角色次世代细节：布料 / 毛发 / 破坏 / 肌肉

> **落地映射**：布料见 `prism_cloth_engine_design_zh.md`，毛发见 `prism_hair_engine_design_zh.md`，破坏/软组织数值求解见 `prism_physics_design_zh.md`；本节只给玩法整合点（受击耦合、破坏回灌玩法、与装备系统联动）。

- **布料/毛发**：披风、头发、植被随动，与角色速度/风场/受击耦合；可 GPU 求解、可 LOD 降级。
- **破坏（Chaos 对标）**：可破坏物件预断裂（fracture），命中/爆炸触发碎裂，碎块与玩法交互（阻挡/掉落）；破坏事件可回灌玩法。
- **肌肉/次级动画**：拉伸挤压、脂肪抖动、受击软组织形变（高端角色），LOD 控制。
- **程序化外观**：MetaHuman 式分层角色（可换装/损伤累积/泥污血迹），与装备/换装系统（游戏模板层）联动显示。

---

## 47. 帧预算与性能工程

### 47.1 帧预算分解（60fps = 16.6ms 示例，玩法占比）
| 子系统 | 预算 | 策略 |
|---|---|---|
| 玩法逻辑（System/GAS/AI） | 2.5–3.5ms | 并行调度、脏标记增量、archetype 批处理 |
| 动画评估 | 1.5–3ms | 异步 Job、LOD、骨骼降采样 |
| 输入/相机/UI | <1ms | 事件驱动、数据绑定免轮询 |
| 网络（复制/预测） | <1ms（均摊） | 增量、兴趣管理、后台序列化 |
| 其余留给渲染/物理/音频 | 8–10ms | 见各自文档 |

### 47.2 性能工程原则
- **Job 化一切可并行**：动画、AI、能力结算、复制序列化全部 `prism_tasks` 并行。
- **p99 稳帧而非均值**：预算针对最坏帧；用时间片轮转摊平尖刺（远处 AI/效果到期分帧处理）。
- **零分配热路径**：实体/效果/事件池化，双缓冲事件，避免帧内堆分配与 GC 风格停顿。
- **数据局部性**：SoA + archetype 聚集，缓存友好；关系直连代替查找。
- **可伸缩降级**：移动/掌机自动降档（粒子数、动画 LOD、AI 频率、群体规模），同一套玩法代码跨平台。
- **预算守卫**：运行时每子系统耗时统计 + 超预算告警（接第 38 节可观测性）。

---

## 48. 跨系统整合：一帧内的数据流与时序

AAA 的难点不在单系统，而在**整帧协同**。确立清晰的一帧时序，避免表现与逻辑错位：

```
帧开始
 1. 采集输入 → 解析增强输入 → 产出输入意图（写被附身 Pawn）
 2. 网络接收 → 应用服务器权威快照 / 确认预测 / 回滚重放
 3. FixedUpdate（固定步）：移动求解 → 物理 → 能力/效果结算 → 属性聚合 → 命中判定
 4. 事件分发：OnAttributeChanged / 命中 / 死亡 → GameplayMessage 总线
 5. 表现驱动：Cue 派发（VFX/音频/相机冲击）、动画状态更新、UI 数据绑定推送
 6. 动画异步评估（Job）→ 蒙皮
 7. 相机求解（跟随/混合/碰撞/抖动）
 8. 网络发送 → 本地预测状态入环形缓冲（供回滚/回放/倒带）
 9. 渲染提交（见渲染文档 GPU Scene）
帧结束
```

- **逻辑先行、表现跟随**：所有表现（动画/VFX/音频/相机）读上一步的玩法结果，单向依赖，避免循环与竞态。
- **预测与权威分离**：步骤 2 的回滚只影响玩法状态，表现层通过平滑/插值吸收修正，杜绝可见跳变。
- **确定性边界**：步骤 3 走确定性路径（固定步 + 全序 + 种子化 RNG），步骤 5–7 的表现层允许非确定性（抖动/粒子随机），互不污染。

---

## 49. 落地与验收：Demo 与质量/性能清单

### 49.1 垂直切片 Demo（Vertical Slice）
一个第三人称动作战斗 Demo，贯通全栈，作为"能达 AAA 线"的证明：
- 可控角色（Motion Matching 移动 + 分层动画 + 足部 IK）
- 一套 GAS 技能（含预测、Cue、命中帧、hitstop、相机冲击）
- 若干完整 AI（StateTree + Smart Object）+ 一群轻量代理（Mass）
- 无缝过场切入/切出 + 一个可破坏场景
- 联机 2–4 人（预测回滚 + 命中回溯）
- 流式大世界一个 Cell 的无缝进出

### 49.2 验收清单（效果）
- [ ] 移动无脚底打滑，转向无生硬切换，坡面台阶 IK 贴合
- [ ] 每次命中具备 VFX+音频+相机+hitstop 四通道协同
- [ ] 本地动作 0 感知延迟；联机修正无橡皮筋/无瞬移
- [ ] 过场无缝（不黑屏不切镜），可跳过
- [ ] 同屏数百单位画质帧率不塌

### 49.3 验收清单（性能）
- [ ] 60fps 稳定，玩法逻辑帧预算 ≤ 3.5ms（p99）
- [ ] 竞技模式 120fps 达标
- [ ] 无帧级尖刺（p99 帧时 / 最坏帧达标）
- [ ] 零 GC 风格停顿、零主线程阻塞加载
- [ ] 移动/掌机降档可运行，同一套玩法代码
- [ ] 万级实体能力结算在预算内（基准测试固化为 CI 回归）

### 49.4 验收清单（可信度）
- [ ] 确定性回放逐帧一致（录像可精确重放）
- [ ] 联机回滚重放结果与权威一致（自动化压测）
- [ ] 玩法回归测试在 CI 无头运行通过

## 50. 术语表

- **Actor**：被标记为可参与玩法的 Entity（非 UE UObject 继承体）。
- **ASC (AbilitySystemComponent)**：能力系统承载组件，聚合能力/效果/属性/标签。
- **GameplayTag**：分层 interned 标签，O(1) 匹配。
- **GameplayEffect**：声明式属性/标签修改数据，含 Instant/Duration/Infinite。
- **GameplayCue**：与逻辑解耦的表现线索（特效/音效）。
- **AttributeAggregator**：把 modifier 聚合为属性 CurrentValue 的增量计算器。
- **Possess/附身**：Controller 与 Pawn 的控制关系。
- **InputContext**：可压栈的输入映射集合。
- **Subsystem**：有生命周期、可发现的服务（App/World/Player 级）。
- **Reconciliation/回滚**：客户端预测与服务器权威不一致时的纠正重放。
- **确定性路径**：固定步长 + 全序系统 + 可选定点数值，服务联机与回放。

---

## 51. 游戏设置框架（GameUserSettings 对标）

> **定位**：base 级玩家设置框架，建于 `prism_app::cvar/settings` + `platform_tier`（平台降档）。对标 UE `UGameUserSettings` + Enhanced Input 重绑 + 控制台变量。提供设置的**模型 / 持久化 / 重绑 / 档位 / 实时应用 / UI 绑定**机制；具体选项清单与难度语义由游戏层扩展（属游戏模板层）。

**分层设置模型（优先级合并）**
- 默认值 → 平台档（`platform_tier` 自动降档）→ 用户持久化覆盖 → 会话临时覆盖，按优先级合并取值。
- 类型化 schema（建于 `prism_reflect`），支持校验、范围约束与版本迁移。

**设置类别**
- **输入**：键位重绑（建于 §10 增强输入的可重绑 DataAsset）、灵敏度、死区、轴反转。
- **图像/画质**：分辨率 / 全屏 / VSync / 帧率上限 / 画质档；画质档联动渲染档位与 `platform_tier`，具体渲染项清单下放到渲染文档。
- **音频**：主 / 音乐 / SFX / 语音总线音量；总线实现见音频文档（§43）。
- **无障碍（accessibility）**：字幕、色盲模式、UI 缩放、震动开关、辅助瞄准（= §37 手感可选 feature 的开关）。
- **玩法/难度**：难度档——base 提供选项模型与持久化，具体难度数值语义由游戏层定义。

**持久化与实时应用**
- 持久化建于 §17 存档 / `prism_reflect` 序列化，带版本迁移。
- 实时应用（live apply）：cvar 热应用、`OnSettingChanged` 事件（§12 Message）；需重启项显式标注。

**UI 绑定**
- 经 §35 MVVM 双向绑定到设置界面（设置项是双向绑定的典型场景），零手写"每帧刷新 UI"。

> **落地映射**：crate `pkg/prism_gameplay_settings`（建于 `prism_app::settings/cvar` + `prism_reflect` + §35 MVVM）。**边界**：具体选项清单、难度数值、品类专属项属游戏模板层；base 提供模型/持久化/绑定/重绑/档位机制。

