# Prism Game Creator 次世代游戏创作套件设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的**游戏创作套件（Game Creator）**设计：在 base 级 `prism_gameplay` 引擎框架之上，构建"开箱即做游戏"的**游戏模板 + 内容系统 + 创作工作流**层——即 Prism 生态中的 **Lyra 等价层**。
> 类比 UE：`prism_gameplay` 对应 **UE 引擎自带的 Gameplay 框架**（GameFramework / GAS / Enhanced Input / AI / Mass / World Partition / 复制），而 **Game Creator 对应 Epic 官方的 Lyra 示例项目 + Game Features 内容包 + 内容创作工作流**——引擎本身不内置任务/对话/物品/经济/连招编辑器，这些属于游戏模板与内容系统层，由此套件统一提供。
> 借形态不抄码，系统性对标顶级次世代 AAA 产品的**内容系统形态与经典数值路径**，不复刻其源码、不引入任何 `bevy_*` 直接依赖，内容层一律建于 `prism_*` 原生 crate 之上。
> 本文为**设计规格**，暂不进入编码阶段；全文严格区分「已落地的下层能力」与「本套件规划项（PLANNED）」。

- 版本: v0.1（设计阶段，建立于 `prism_gameplay` v0.2 之上）
- 适用引擎: Prism / `prism_ecs` 生态（脱离 Bevy 后的独立引擎，不依赖任何 `bevy_*` crate）
- 定位: **第四联**——与 Loom Studio（编辑器）、Loom Runtime（运行时）、Prism Gameplay（玩法框架）并列，是建于 Gameplay 之上的**内容/模板层**
- 关键依赖（下层 base 框架与内核）: `prism_gameplay_*`（Actor/GAS/增强输入/流程/AI/序列/消息/存档/Game Features/交互 §30/手感 §37/设置 §51）、`prism_ecs`（关系/预制/观察者/分区）、`prism_app`（state/settings/cvar/platform_tier）、`prism_reflect`（序列化/schema/net_delta，支撑内容资产与存档）、`prism_asset`（DataAsset 热重载）、`prism_time`（timeline/recording）、`prism_network`（复制/预测/回滚，内容层只做标注）、`prism_ui_*` + Loom Runtime（§35 MVVM 的 HUD/菜单/叙事表现）
- 非目标: **不替代 base 框架**——仿真/能力/输入/网络权威在 `prism_gameplay` 与各引擎子系统；Game Creator 只做**内容定义 + 模板脚手架 + 作者工作流 + LiveOps/UGC**；与引擎文档一致，纯经典数值路径，**不含 AI / ML / 神经网络 / LLM** 功能。
- 相关文档: `prism_gameplay_design_zh.md`（base 玩法框架，本套件的直接地基）、`prism_loom_runtime_framework_design_zh.md`（运行时外壳/表现）、`prism_editor_framework_design_zh.md`（Loom Studio 编辑器，内容作者工具挂接点）、`prism_ecs_design_zh.md`、`prism_reflect_design_zh.md`、`prism_asset_design_zh.md`、`prism_network_design_zh.md`、`prism_ui_loom_design_zh.md`、`prism_engine_component_gap_zh.md`（落地缺口追踪）

---

## 目录
1. 设计哲学与定位
2. 与 Prism 生态的关系：四联架构与分层边界
3. 对标借鉴矩阵：每个内容系统对标哪些顶级产品
4. 总体架构：`pkg/prism_gc_*` 布局
5. 内容创作工作流：从模板到可玩
6. 游戏模板系统：FPS / TPS / ARPG 脚手架
7. 模块化内容包：建于 Game Features 的运行时插拔
8. 战斗与手感内容：连招编辑器 / 命中内容 / 打击感
9. 物品与经济：背包 / 装备 / 掉落表 / 货币
10. 进程与叙事：任务 / 对话 / 阵营声望
11. 角色成长：等级 / 技能树 / 天赋
12. 交互内容：可交互物 / 场景契约内容
13. 世界内容与遭遇：刷怪 / 遭遇编排 / 兴趣点
14. AI 内容库：行为模板 / 战术 / 群体
15. 相机与演出内容：运镜预设 / 过场
16. UI / HUD 内容：菜单流 / HUD 组件库
17. 音频内容：混音快照 / 交互音乐 / 对白
18. 存档与游戏状态内容
19. LiveOps 实时服务：赛季 / 活动 / 远程配置
20. UGC / Mod：用户生成内容与安全沙箱
21. 性能工程：内容层的预算与流式
22. 易用性与创作者体验
23. 模块化与可维护性
24. 迭代速度：热重载 / PIE / 内容快速试玩
25. Crate 拆分与落地形态
26. 公共 API 草案
27. 路线图
28. 落地与验收：Demo 与清单
29. 武器系统：射击 / 近战 / 改装
30. 载具、坐骑与机甲
31. 位移与跑酷（Traversal）
32. 术语表

---

## 1. 设计哲学与定位

Game Creator 回答一个具体问题：**Gameplay 框架已经把"引擎级能力"做全了，但从"能力"到"一款能玩的游戏"之间还差一层。这一层是什么、谁来做、怎么做得又快又好。**

类比 UE 生态可一句话说清边界：

- **引擎 Gameplay 框架**（= `prism_gameplay`）：GameMode、GAS、Enhanced Input、行为树、Mass、World Partition、复制——这些是**机制地基**，任何类型的游戏都用得上，但它本身**不是一款游戏**。
- **Lyra 示例项目 + Game Features 内容包**（= Game Creator）：把上面的机制组装成"射击游戏模板""武器插件""经验系统插件""前端菜单"——**开箱即玩、可拆可换的内容**。

因此确立六条铁律：

- **内容即数据，模板即组装**：Game Creator 不发明新的运行时机制；它用 `prism_gameplay` 已有的 Tag/GAS/Message/DataAsset/关系/Game Features，把它们**组装**成可直接试玩的游戏骨架。能用 DataAsset 描述的内容，绝不写死为代码。
- **可拆解优先（Lyra 式）**：每个内容域（武器、经验、队伍、计分、前端）都是独立的**内容包（Experience / Game Feature）**，可运行时插拔、可被另一款游戏整包复用，彼此之间通过 Tag 与 Message 松耦合。
- **不越界 base 框架**：机制 bug 在 `prism_gameplay` 修，表现 bug 在引擎子系统修；Game Creator 只负责**内容定义、脚手架、作者工具与装配**。网络仍走 `prism_network`，表现仍走 Cue/UI/音频子系统。
- **创作者体验是第一等公民**：设计目标不是"工程师能写出来"，而是"策划/关卡/叙事能在编辑器里拖出来"。每个内容系统都必须有对应的 Loom Studio 作者面板与数据资产 schema。
- **零配置能跑，逐层可下钻**：`cargo run --example fps_template` 应直接得到一个能走能打的射击 Demo；进阶用户逐层替换武器表、能力集、HUD 布局，而不碰框架代码。
- **无 AI 立场沿用**：与 renderer/gameplay 文档一致，内容生成走经典规则（掉落表、权重、状态机、规则引擎），**不含 AI/ML/LLM**。程序化内容指 PCG/规则化编排，不是神经生成。

一句话定位：**`prism_gameplay` 让 Prism "能做任何游戏"；Game Creator 让 Prism "开箱做出某一款游戏"。**

---

## 2. 与 Prism 生态的关系：四联架构与分层边界

Prism 生态此前为"三联"，Game Creator 补齐为**四联**：

| 联 | 工作名 | 文档 | 职责 | 类比 UE |
|---|---|---|---|---|
| 编辑器 | Loom Studio | `prism_editor_framework_design_zh.md` | 内容创作工具、视口、面板、协同 | UE Editor |
| 运行时 | Loom Runtime | `prism_loom_runtime_framework_design_zh.md` | 应用外壳、PlayerLoop、HUD/菜单表现、响应式绑定 | GameInstance + UMG |
| 玩法框架 | Prism Gameplay | `prism_gameplay_design_zh.md` | **base 级机制**：Actor/GAS/输入/AI/网络接缝/Game Features | UE Gameplay 模块组 |
| **内容套件** | **Game Creator** | **本文档** | **游戏模板 + 内容系统 + 作者工作流 + LiveOps/UGC** | **Lyra + Game Features 内容包** |

分层边界（数据流自上而下依赖，不反向）：

```
┌─────────────────────────────────────────────────────────┐
│  Game Creator（内容/模板层）                               │
│   游戏模板 · 内容包 · 连招/物品/任务/成长内容 · LiveOps · UGC │
├─────────────────────────────────────────────────────────┤
│  Prism Gameplay（base 机制层）                             │
│   Actor/GAS/Input/AI/Sequence/Message/Save/Features/交互/手感 │
├─────────────────────────────────────────────────────────┤
│  Prism 内核 + 引擎子系统                                   │
│   ecs/app/reflect/asset/time/network · 渲染/物理/动画/音频/相机 │
└─────────────────────────────────────────────────────────┘
```

四条硬边界：

- **机制 vs 内容**：若一段逻辑"任何类型游戏都要用"，它属于 `prism_gameplay`；若它"只服务某类玩法/某款游戏的配置"，它属于 Game Creator。例：GAS 的 Effect 结算引擎在 gameplay；"火球术 = 50 伤害 + 3 秒燃烧"这张数据表在 Game Creator。
- **运行时 vs 表现**：Game Creator 定义"任务完成要弹什么 HUD"的内容资产，但 HUD 的渲染与绑定由 Loom Runtime + `prism_ui_*` 执行。
- **内容 vs 网络**：Game Creator 的内容资产带复制标注（谁权威、是否预测），实际复制/回滚由 `prism_network` 执行；套件不自建网络栈。
- **作者 vs 运行**：内容资产的编辑在 Loom Studio，运行时只消费已序列化/已烘焙的 DataAsset，二者经 `prism_asset` 热重载打通。

> 对 `prism_gameplay §31/§32` 的呼应：gameplay 文档把"叙事与进程""物品/背包/装备/掉落表"标为**游戏模板层、非 base 框架**并保留编号 stub。本文档 §9/§10 即这两节的**正式归属地**；gameplay 相应 stub 指向本文档。

---

## 3. 对标借鉴矩阵：每个内容系统对标哪些顶级产品

借形态不抄码，仅取**内容系统形态、经典数值路径与作者工作流**：

| 本文档章 | 内容系统 | 对标借鉴（取其形态/工作流） | 不抄什么 |
|---|---|---|---|
| §6 游戏模板 | FPS/TPS/ARPG 脚手架 | **Lyra**（Experience + GameFeature 可拆模板）、Unity 的 Starter/URP 模板、Unreal 模板项目 | 其资产二进制、具体关卡 |
| §7 内容包 | 运行时插拔内容 | Lyra Experience、UEFN Verse 设备、Destiny 的 Activity 配置 | 其后端服务实现 |
| §8 战斗/手感内容 | 连招、命中、打击感 | **DMC5 / God of War / Sekiro / Doom Eternal / 街霸6** 的连招表/取消窗口/命中停顿 | 其动作资产与动画 |
| §9 物品/经济 | 背包/装备/掉落/货币 | **Diablo / Destiny / Path of Exile / Elden Ring** 的词缀/品质/掉落权重/货币生态 | 其具体数值与美术 |
| §10 任务/叙事 | 任务/对话/声望 | **Witcher 3 / Cyberpunk 2077 / BG3 / Disco Elysium / Mass Effect** 的任务图/对话树/声望门槛 | 其剧本文本 |
| §11 角色成长 | 技能树/天赋 | **PoE 天赋树 / Borderlands / Destiny / 暗黑4 巅峰** 的节点图与加点规则 | 其具体天赋数值 |
| §12 交互内容 | 可交互物契约 | **RDR2 / 塞尔达旷野之息** 的世界交互动词、Half-Life 的 use 契约 | 其具体物件实现 |
| §13 世界/遭遇 | 刷怪/遭遇编排 | **Left 4 Dead AI Director / Elden Ring 兴趣点 / Horizon** 的遭遇与巡逻 | 其具体地图 |
| §14 AI 内容库 | 行为模板/战术 | **Halo / F.E.A.R. / The Last of Us 2 / Alien Isolation** 的战术行为库与感知配置 | 其专有行为树资产 |
| §15 相机/演出 | 运镜预设/过场 | **Cinemachine 预设 / God of War 一镜到底 / 神秘海域** 过场编排 | 其过场序列资产 |
| §16 UI/HUD | 菜单流/HUD 库 | **Lyra 前端 / Destiny 导航 / 现代 3A 无障碍选项面板** | 其美术主题 |
| §17 音频内容 | 混音/交互音乐 | **Wwise/FMOD 工作流形态、DOOM/战神 的动态音乐分层** | 其音频中间件实现 |
| §19 LiveOps | 赛季/活动/远程配置 | **Destiny 2 / Fortnite / Warframe / Apex** 的赛季与远程开关 | 其后端与经济服务 |
| §20 UGC/Mod | 用户内容/沙箱 | **UEFN / Roblox / Garry's Mod / Steam Workshop** 的 Mod 装载与沙箱 | 其平台后端 |
| §29 武器系统 | 射击/近战/改装 | **Lyra 武器 / COD·Destiny gunsmith / DMC5·只狼 moveset / 怪物猎人 武器类别** | 其武器资产与动画 |
| §30 载具/坐骑/机甲 | 驾驶/骑乘/机甲 | **GTA·极限竞速 / BOTW·RDR2·WoW 坐骑 / Titanfall·Armored Core 机甲** | 其载具资产与物理数值 |
| §31 位移/跑酷 | 翻越/墙跑/抓钩 | **镜之边缘 / 刺客信条 / 消逝的光芒 / Titanfall** 的位移动词与 flow | 其动画资产 |

对标纪律（与各引擎文档一致）：只研究公开可得的**设计形态、数值模型与编辑工作流**；不拉取任何产品的源码、资产或衍生数据；对标产品名仅用于标注"借鉴了哪种形态"。

---

## 4. 总体架构：`pkg/prism_gc_*` 布局

遵循仓库"一子系统一 crate"惯例，内容套件落在 `pkg/prism_gc_*`（gc = Game Creator），门面 crate 为 `prism_game_creator`，与 `prism_gameplay_*` / `prism_ui_*` 命名对齐。全部建于 base 框架之上，不新增底层机制。

分层（Layer 延续 gameplay 文档的 L1-L5 之上，内容层记为 **L6**）：

```
L6 内容/模板层（prism_gc_*）
  ├─ prism_gc_core        内容包/Experience 定义与装配器（建于 prism_gameplay_features §26）
  ├─ prism_gc_templates   FPS/TPS/ARPG 等模板脚手架 + 示例装配
  ├─ prism_gc_combat      连招/命中内容/打击感内容（建于 §9 GAS + §37 手感）
  ├─ prism_gc_weapons     武器原型/射击弹道/近战 moveset/改装插槽（建于 §8 + §9 + GAS）
  ├─ prism_gc_vehicles    载具/坐骑/机甲内容 + 武器硬点（建于 gameplay §7 附身 + physics）
  ├─ prism_gc_traversal   位移/跑酷动词库 + 表面标注 + flow（建于 gameplay §7 CCT + §37 + §41）
  ├─ prism_gc_items       物品/背包/装备/掉落/货币内容（gameplay §32 的落地）
  ├─ prism_gc_progression 等级/技能树/天赋/成就内容
  ├─ prism_gc_quests      任务/对话/阵营声望内容（gameplay §31 的落地）
  ├─ prism_gc_interaction 交互内容库（建于 gameplay §30）
  ├─ prism_gc_encounter   世界内容/刷怪/遭遇编排/兴趣点
  ├─ prism_gc_ai_content  AI 行为模板/战术/感知预设库（建于 gameplay §14/§29）
  ├─ prism_gc_cine        运镜预设/过场内容（建于 gameplay §44/§45）
  ├─ prism_gc_ui_content  菜单流/HUD 组件内容（建于 Loom §35）
  ├─ prism_gc_audio_content 混音快照/交互音乐/对白内容（建于 gameplay §43）
  ├─ prism_gc_liveops     赛季/活动/远程配置内容（远程配置拉取经平台 I/O）
  ├─ prism_gc_ugc         UGC/Mod 装载 + 内容沙箱（建于 gameplay §18 脚本 + §26）
  └─ prism_game_creator   门面：GameCreatorDefaultPlugins + 模板选择器
```

每个 crate 的产物都是三件套：**(1) 运行时组件/系统**（极薄，大多是装配 gameplay 已有机制）、**(2) DataAsset schema**（内容真相源，经 `prism_reflect`）、**(3) Loom Studio 作者面板契约**（编辑器挂接点）。

与 base 的依赖关系：`prism_gc_*` **只能依赖** `prism_gameplay_*`、`prism_*` 内核、`prism_ui_*`；**禁止反向**（gameplay 不得依赖 gc）。门面 crate `prism_game_creator` 聚合默认内容包，提供"选一个模板即可跑"的开发者入口。

> 落地状态登记：`prism_engine_component_gap_zh.md` 中为本套件追加一行 `prism_gc`（内容/模板层），初始标 ⬜（PLANNED，设计阶段，无代码）。

---

## 5. 内容创作工作流：从模板到可玩

Game Creator 的核心价值是把"分散的机制"收敛为一条**可重复的创作流水线**。目标：一名策划 + 一名关卡 + 一名叙事，不写 Rust 也能产出可玩垂直切片。

工作流五阶段：

1. **选模板（Scaffold）**：从 §6 模板库选一个（如 TPS），`prism_gc_templates` 生成一个最小可玩工程——含默认 GameMode/HUD/输入/相机/一把武器/一个敌人。对标 Unreal/Unity 的 New Project 模板。
2. **装内容包（Compose）**：以 Experience/Game Feature 为单位，运行时勾选启用内容包（武器包、经验包、队伍包、计分包）。对标 Lyra 的 Experience 装配。
3. **填数据（Author）**：在 Loom Studio 面板里编辑 DataAsset——武器表、掉落表、技能树、任务图、对话树、混音快照。所有编辑即时热重载进 PIE。
4. **编排（Orchestrate）**：用序列/遭遇编辑器编排关卡遭遇、过场、触发器；用任务图连接节点。
5. **调试与试玩（Iterate）**：PIE（Play-In-Editor）即点即玩，Gameplay Debugger（gameplay §38）叠加内容态可视化（当前任务步、掉落 roll、连招窗口）。

贯穿工作流的三条机制：

- **内容 ID 与引用完整性**：所有内容资产用稳定 `AssetId`（`prism_asset`）互引；编辑器在保存时做引用校验（断链、循环、缺失 Tag），避免"武器引用了不存在的能力"。
- **内容 schema 版本化**：每类 DataAsset 带 `schema_version`，经 `prism_reflect` 的 schema 迁移在加载时自动升级旧内容，保障长期可维护（对标 Unity ScriptableObject 的版本迁移痛点，提前解决）。
- **内容校验器（Validator）**：可运行的 lint 规则集（如"每把武器必须有 fire 能力""每个任务必须可达终点"），在编辑器、CI、打包三处运行，杜绝脏内容进入构建。

> 作者工作流的 UI 一律由 Loom Studio 承载；本套件只定义**面板契约 + 校验规则 + schema**，不自建编辑器 UI 框架。

---

## 6. 游戏模板系统：FPS / TPS / ARPG 脚手架

模板 = 一组预装配的内容包 + 默认资产 + 示例关卡，开箱即跑，是新项目的起点（Lyra 定位）。

初始模板矩阵：

| 模板 | 核心装配 | 对标 | 默认内容包 |
|---|---|---|---|
| `fps` 第一人称射击 | 第一人称相机 + 枪械 GAS 能力 + 命中内容 + 计分 | Lyra / COD 式 | 武器包、弹药包、计分包、团队包 |
| `tps` 第三人称动作 | 第三人称相机（§44）+ 近战连招（§8）+ 闪避 i-frame | 战神 / 仁王 | 连招包、锁定包、处决包 |
| `arpg` 动作角色扮演 | 俯视相机 + 技能槽 + 掉落/装备 + 技能树 | 暗黑 / PoE | 物品包、掉落包、技能树包、任务包 |
| `platformer` 平台跳跃 | 2.5D 相机 + 精确手感（§37）+ 收集品 | 蔚蓝 / 马力欧 | 手感包、收集品包、关卡计时包 |
| `survival` 生存建造（可选） | 交互（§30）+ 背包 + 制作 + 昼夜 | 森林 / 瓦尔海姆 | 交互包、制作包、资源包 |

模板设计原则：

- **薄装配、厚内容**：模板本身几乎无代码，是一个"内容包清单 + 默认 DataAsset 集"。切换模板 = 换一份清单。
- **可渐进替换**：从 `fps` 起步想做 extraction shooter，只需加"背包包 + 撤离点遭遇包"，不推倒重来。
- **示例即文档**：每个模板附一个 `examples/<tpl>_demo`，是可运行的最佳实践参考（对标 Bevy examples 文化）。
- **平台档适配**：模板默认读取 `prism_app::platform_tier`，在低端档自动降内容密度（刷怪上限、特效 Cue 等级）。

---

## 7. 模块化内容包：建于 Game Features 的运行时插拔

内容包（Experience / Content Pack）是 Game Creator 的组织单元，直接建于 `prism_gameplay_features`（gameplay §26 Game Features 运行时插拔）。

内容包定义（DataAsset）包含：

- **启用动作集**：启用时注入哪些组件/系统/输入上下文/能力集（经 §26 的组件注入器）。
- **Tag 契约**：声明本包提供与依赖的 GameplayTag（如武器包 `provides: Weapon.*`，计分包 `requires: Team.*`），装配器据此做依赖解析与冲突检测。
- **资产清单**：本包携带的 DataAsset（武器表、掉落表、HUD 片段）。
- **优先级与覆盖**：多包叠加时的覆盖规则（如"硬核模式包"覆盖默认难度曲线）。

关键能力：

- **运行时热插拔**：PIE 中勾选/取消内容包即时生效（经 `prism_asset` 热重载 + §26 的注入/回收），无需重启——这是迭代速度的核心（§24）。
- **包隔离**：每个包的系统运行在独立的 SystemSet，卸载时精确回收其注入的实体与组件，不泄漏（复用 §26 的生命周期与 `StateScoped`）。
- **跨游戏复用**：一个"经验曲线包"可被 TPS 与 ARPG 两款游戏整包复用，因为它只依赖 Tag 契约而非具体游戏。
- **冲突裁决**：装配器检测 Tag/输入绑定/HUD 槽位冲突，给出明确诊断（对标 UE Game Feature 的 action 冲突报错）。

> 本章不重复实现插拔机制——插拔是 gameplay §26 的能力；本章定义的是**内容包的 schema、依赖契约与装配策略**。

---

## 8. 战斗与手感内容：连招编辑器 / 命中内容 / 打击感

把 base 的 §9 GAS 与 §37 手感/命中判定基建，组装为**可编辑的战斗内容**。对标 DMC5/战神/只狼/Doom Eternal/街霸6。（武器类别的具体 moveset、射击弹道与改装见 §29。）

内容系统：

- **连招表（Combo Graph）**：有向图资产，节点 = 招式（绑定一个 GAS Ability + 动画 Tag），边 = 取消窗口/输入条件（轻→轻→重、方向键派生）。运行时由状态推进消费；编辑器提供可视化连招图。对标 DMC/街霸的 move list 与取消规则。
- **命中内容（Hit Data）**：每招的 hitbox 激活帧窗口、伤害/削韧/硬直、击退向量、命中停顿（hitstop）、受击反应 Tag。基建（hitbox/hurtbox、i-frame=Tag、削韧=Attribute）在 §37，本章提供**可编辑的数值资产**。
- **打击感内容（Feel Profile）**：命中反馈套件——hitstop 时长、屏幕抖动曲线、顿帧、受击闪白、Cue（粒子/音效）引用、手柄震动模式。全部经 Cue/Message 下放给表现与音频子系统，本章只持有数值与引用。
- **输入手感内容**：输入缓冲窗口、宽容期（coyote time）、连招预输入——数值化为资产，底层由 §10 增强输入 + §37 执行。

设计要点：

- **数据驱动的取消系统**：取消规则（哪招能取消哪招、第几帧开窗）是数据，不是硬编码的 if-else，策划可直接调。
- **帧精确与可确定**：命中窗口以逻辑帧（gameplay §19 确定性）计数，支持回放校验，联机走 §33 的 lag compensation。
- **分层默认**：动作向重手感（取消树、派生）作为可选内容包，默认模板只给"轻击/重击/闪避"最小集，逐层加码。
- **调试叠层**：Gameplay Debugger 叠加显示当前连招节点、取消窗口剩余帧、hitbox/hurtbox、i-frame 状态。

---

## 9. 物品与经济：背包 / 装备 / 掉落表 / 货币

> **归属声明**：本章是 `prism_gameplay §32`（物品/背包/装备/掉落表，游戏模板层）的正式落地。gameplay 保留 §32 编号 stub 并指向此处。

建于 base：装备 = GAS Infinite Effect + 能力注入（§9）；容器关系 = ECS `Contains`/`ContainedBy`（§5 关系）；资产 = DataAsset（§13）；持久化 = 存档（§17）。对标 Diablo/Destiny/PoE/Elden Ring。（武器类物品的射击/近战/改装专属内容见 §29；载具/机甲内容见 §30。）

内容系统：

- **物品定义（Item Def）**：类型、堆叠、品质、重量/体积、图标、使用能力引用、装备槽、词缀槽。纯数据资产。
- **背包/容器**：槽位网格或重量制，经 ECS 关系表达"物品属于容器"；拾取/丢弃/转移走 §30 交互 + §12 消息。
- **装备系统**：装备即对角色 ASC 施加一个 Infinite GameplayEffect（属性加成）+ 可选能力授予（如"穿上靴子获得冲刺"），卸下即移除——完全复用 GAS，不另造系统。
- **词缀与品质（Affix / Rarity）**：Diablo/PoE 式词缀池 + 权重 + 品质分级；掉落时按规则 roll 词缀。纯经典随机，无 AI。
- **掉落表（Loot Table）**：分层权重表（怪物→掉落组→物品+概率+数量区间），支持保底（pity）、幸运加成、首杀奖励。对标 Destiny/暗黑掉落。
- **货币与经济**：多币种、商店买卖、分解/合成、绑定规则。经济曲线为可调资产，便于平衡与 LiveOps 调参（§19）。

设计要点：

- **复制与权威**：掉落 roll、交易在服务器权威（经 `prism_network`），客户端只预测拾取动画；物品实例带稳定 ID 便于存档与反作弊（`prism_anticheat`）。
- **可平衡性**：所有数值（词缀范围、掉率、价格）集中在资产，支持 A/B 与远程配置覆盖（§19）。
- **存档友好**：物品实例经 `prism_reflect` 序列化，schema 版本化支持跨版本迁移（新增词缀不损坏旧档）。

---

## 10. 进程与叙事：任务 / 对话 / 阵营声望

> **归属声明**：本章是 `prism_gameplay §31`（叙事与进程，游戏模板层）的正式落地。gameplay 保留 §31 编号 stub 并指向此处。

建于 base：Tag（§8）表达状态、GAS/Message（§9/§12）驱动事件、DataAsset（§13）承载内容、存档（§17）持久化进度。对标 Witcher 3/Cyberpunk/BG3/Disco Elysium/Mass Effect。

内容系统：

- **任务图（Quest Graph）**：节点 = 目标（击杀/到达/交付/护送），边 = 前置条件与分支；支持并行目标、可选目标、失败分支、时限任务。状态机落到 Tag + 任务组件，推进由 Message 触发。对标 Witcher 的任务阶段机。
- **对话系统（Dialogue）**：对话树/图资产——节点 = 台词 + 说话人 + 条件 + 效果（给任务/改声望/给物品）；选项带技能检定（对标 Disco Elysium 的属性检定、BG3 的骰检）。本地化经 `prism_ui_i18n`。
- **阵营与声望（Faction / Reputation）**：阵营图 + 声望数值 + 门槛事件（声望到敌对触发攻击、到友好解锁商店）。声望变化经经典规则（动作→声望增量表），无 AI。
- **叙事状态与世界状态（World State / Flags）**：全局剧情 flag 存储，供任务/对话/遭遇查询；经存档持久化。对标各 RPG 的 quest variable 系统。
- **分支与后果（Branching）**：选择后果通过世界状态 flag 影响后续内容（商店、对话、结局），而非硬编码。

设计要点：

- **纯数据、可本地化、可测**：任务/对话全为 DataAsset，叙事团队在编辑器编写；校验器（§5）检查"每个任务可达终点""每个对话节点有出口"。
- **与存档强绑定**：任务进度、对话历史、世界 flag 是存档的一等内容（§18）。
- **事件而非轮询**：目标达成经 Message 事件推进，避免每帧轮询，契合数据导向性能原则。

---

## 11. 角色成长：等级 / 技能树 / 天赋

建于 GAS 属性（§9）与 Tag（§8）。对标 PoE 天赋树/Borderlands/Destiny/暗黑4 巅峰系统。

内容系统：

- **经验与等级曲线**：XP 来源表（击杀/任务/探索）+ 等级曲线资产；升级经 Message 广播，触发属性成长（GAS 永久 Effect）。
- **技能树 / 天赋图（Talent Graph）**：节点图资产——节点 = 加点效果（属性/能力/被动 Tag），边 = 解锁依赖；支持大型互联天赋树（PoE 式）与分支专精树（暗黑式）。加点 = 施加对应 GAS 效果/授予能力。
- **技能槽与装配（Loadout）**：主动技能槽位、被动插槽、符文/宝石镶嵌；装配即数据，运行时转为能力授予。
- **重置与洗点（Respec）**：规则化移除已加点效果并返还点数，纯数据操作。

设计要点：

- **成长=授予 GAS 效果**，不另造数值系统，与装备（§9）共用同一属性管线，天然可叠加结算。
- **可平衡**：曲线与节点数值集中资产，支持 LiveOps 调参与赛季重置（§19）。
- **UI 自动派生**：技能树 UI 从天赋图资产自动生成布局（经 Loom §35），策划改图即改 UI。

---

## 12. 交互内容：可交互物 / 场景契约内容

建于 gameplay §30（交互与世界契约 base 机制：Interactable/Interactor 数据模型、检测执行管线、与 §29 Smart Object 共用执行体）。本章提供**可编辑的交互内容库**。对标 RDR2/塞尔达的世界动词、Half-Life 的 use 契约。

内容系统：

- **交互动词库（Verb Library）**：开门/拾取/对话/撬锁/采集/乘骑/搬运等预制交互，每个 = 一组 §30 的 Interactable 配置 + 提示文案 + 触发能力。
- **交互提示内容（Prompt）**：按钮提示、距离/朝向门槛、高亮描边 Cue、长按/连打 QTE 参数——数据化，表现下放 UI/Cue。
- **上下文交互**：同一物件按状态给不同动词（门：开/锁/撬），由 Tag 条件选择，复用 §30 的检测管线。
- **场景触发器内容**：区域触发、压力板、机关联动——以 Smart Object（§29）+ 遭遇（§13）编排，纯数据。

设计要点：零新机制，全部是 §30 的配置资产；提供编辑器内"交互点可视化"叠层与批量放置工具契约。

---

## 13. 世界内容与遭遇：刷怪 / 遭遇编排 / 兴趣点

建于 gameplay §28（World Partition/Data Layers/后台仿真）与 §15（序列编排）。对标 L4D 的 AI Director、Elden Ring 兴趣点、Horizon 巡逻。

内容系统：

- **刷怪与填充（Spawning）**：刷怪点、刷怪波次、密度曲线；按玩家进度/平台档/难度缩放。随 World Partition 流送加载，远处休眠（Dormant）。
- **遭遇编排（Encounter）**：序列化的遭遇脚本——触发条件→阶段→胜负判定→奖励，支持多阶段 Boss、护送、守点。建于 §15 Timeline/Playable。
- **动态节奏导演（Director）**：L4D 式强度曲线——按经典规则（玩家血量/击杀速率/时间）调整刷怪与补给节奏；纯规则，无 AI。
- **兴趣点与世界事件（POI / World Event）**：地图标记、随机世界事件、巡逻路线资产；经 Data Layers 分层管理。

设计要点：遭遇全为数据资产，关卡设计师在编辑器编排；运行时经 §28 流送与 §14 AI 内容驱动；强度/密度集中为可调曲线，支持 LiveOps 与难度包覆盖。

---

## 14. AI 内容库：行为模板 / 战术 / 群体

建于 gameplay §14（行为树/黑板/感知）与 §29（StateTree/Smart Objects/群体导航）。本章是**可复用的 AI 行为内容库**，不是新 AI 机制。对标 Halo/F.E.A.R./TLOU2/Alien Isolation。

内容系统：

- **行为模板库（Behavior Presets）**：预制敌人原型——近战冲锋、远程掩体、治疗辅助、精英指挥；每个 = 行为树/StateTree 资产 + 黑板默认值 + 感知配置。
- **战术内容（Tactics）**：F.E.A.R. 式小队战术——包抄、压制、呼叫增援；经 Smart Object + 黑板共享（§29），纯规则协调。
- **感知配置（Perception Profile）**：视野角/距离/记忆时长/听觉灵敏度数据集，支持 Alien Isolation 式的搜索/警戒/猎杀状态曲线。
- **群体与巡逻（Crowd / Patrol）**：Mass（§27）驱动的人群填充与巡逻路线内容，分平台档控制上限。

设计要点：全部是已有 AI 机制的**配置资产 + 预设库**；提供编辑器内行为预览与黑板实时监视叠层；明确红线——行为选择是经典行为树/效用（utility）规则，**不含神经网络决策**。

---

## 15. 相机与演出内容：运镜预设 / 过场

建于 gameplay §44（相机/运镜，Cinemachine 对标）与 §45（过场与叙事演出整合）。对标 Cinemachine 预设、战神一镜到底、神秘海域过场。

内容系统：

- **运镜预设库（Camera Rigs）**：第一/第三人称、锁定、越肩、俯视、过肩瞄准等预制相机资产 + 混合规则；模板直接引用。
- **过场内容（Cutscene）**：Timeline（§15）编排的镜头/动画/音频/字幕轨道资产；支持可跳过、运行时过场（非预渲染）。
- **动态演出（Dynamic Framing）**：战斗取景、处决镜头、锁定目标构图规则——数据驱动，经 §44 执行。
- **情境触发**：进洞穴切相机、Boss 入场运镜——由遭遇（§13）/触发器（§12）调用相机预设。

设计要点：相机机制在 §44，本章只提供预设与过场**内容**；过场可被叙事（§10）与遭遇（§13）引用；提供编辑器时间线作者面板契约。

---

## 16. UI / HUD 内容：菜单流 / HUD 组件库

建于 Loom Runtime（§35 MVVM 绑定、HUD/菜单/叙事 UI 运行时）与 `prism_ui_*`。本章提供**可装配的前端与 HUD 内容**，对标 Lyra 前端、Destiny 导航、现代 3A 无障碍面板。

内容系统：

- **前端菜单流（Frontend Flow）**：主菜单→大厅→设置→加载的导航图资产，经 `prism_ui_router`；可被模板/内容包增删页面。
- **HUD 组件库（HUD Kit）**：血条、弹药、小地图、准星、技能冷却、任务追踪、伤害数字等预制 HUD 片段，经 §35 绑定到 GAS 属性/任务/物品数据。
- **提示与通知系统（Toast / Objective）**：拾取提示、任务更新、成就弹窗——订阅 Message（§12）自动显示。
- **设置界面内容**：从 gameplay §51 游戏设置框架自动生成设置面板（画质/输入/音频/无障碍/难度），本章提供布局内容与分组。

设计要点：UI 渲染/绑定由 Loom + `prism_ui_*` 执行，本章只持有**布局内容与数据绑定声明**；HUD 片段可由内容包按槽位注入（§7 冲突裁决解决槽位争用）；无障碍（a11y）与本地化（i18n）为默认项，非事后补丁。

---

## 17. 音频内容：混音快照 / 交互音乐 / 对白

建于 gameplay §43（音频：空间化/交互混音/程序化，MetaSound 对标）与 `prism_audio_*`。本章提供**音频内容资产**，对标 Wwise/FMOD 工作流、DOOM/战神动态音乐。

内容系统：

- **混音快照（Mix Snapshot）**：战斗/探索/菜单/过场的混音状态资产，按游戏状态切换（闪避时压低背景、低血量高通滤波）。
- **交互音乐（Interactive Music）**：分层/分段音乐——按强度曲线（与 §13 导演联动）切换 stem 与过渡；经 §43 执行。
- **对白内容（Dialogue Audio）**：台词音频 + 字幕 + 口型 Tag，绑定叙事（§10）对话节点；本地化多语轨。
- **声音 Cue 库**：命中/脚步/UI/环境 Cue 预设，供战斗（§8）/交互（§12）/HUD（§16）引用。

设计要点：音频 DSP/空间化在 §43，本章只持有**内容与切换规则**；混音切换订阅游戏状态与 Message，不轮询；程序化=规则化分层，不含神经音频生成。

---

## 18. 存档与游戏状态内容

建于 gameplay §17（存档与持久化，`prism_gameplay_save` + `prism_reflect` 序列化）。本章定义**哪些内容进存档、如何版本化与迁移**。

内容系统：

- **存档内容清单（Save Schema）**：声明进档的内容域——角色进度（§11）、物品背包（§9）、任务/世界 flag（§10）、设置（§51）、解锁与成就。
- **存档槽与配置**：多槽、自动存档、快速存档、云存档接口（云同步 I/O 下放平台层）。
- **版本迁移（Migration）**：每个内容 schema 带版本，加载旧档经 `prism_reflect` schema 迁移链升级；校验器确保迁移完备（§5）。
- **防篡改**：单机校验和 + 联机权威在服务器（`prism_anticheat`/`prism_network`），避免信任客户端存档。

设计要点：存档机制在 §17，本章只定义**内容契约与迁移策略**；明确"可玩进度=一等内容"，长期可维护性靠 schema 版本化保障。

---

## 19. LiveOps 实时服务：赛季 / 活动 / 远程配置

对标 Destiny 2/Fortnite/Warframe/Apex 的赛季与远程运营。本章提供**内容侧的 LiveOps 装配**；后端服务与经济账本不在本套件范围，经平台 I/O 接入。

内容系统：

- **赛季内容（Season）**：赛季定义 = 一个大内容包（§7）——新武器/任务/地图/奖励轨道，有生效时间窗；到期自动卸载/切换。
- **活动与限时玩法（Events）**：节日活动、限时模式，经内容包热插拔上线/下线，无需发版（契合 §24 热重载哲学）。
- **远程配置（Remote Config）**：掉率/价格/难度/开关的远程覆盖层，叠加在本地默认资产之上；拉取经平台网络 I/O，运行时经 cvar/设置热应用（gameplay §51）。
- **战令/奖励轨道（Battle Pass）**：进度轨道 + 奖励表资产，复用成长（§11）与物品（§9）。
- **A/B 与灰度**：同一内容的多版本按分组下发，纯数据切换。

设计要点：LiveOps=内容包的时间化与远程覆盖，不是新机制；安全上远程配置只能覆盖**已声明的白名单字段**（防止远程注入任意行为），经校验器约束；经济权威在服务器。

> 安全提示：远程配置拉取属于出站网络；需明确只读白名单字段、签名校验与回退默认，避免把远程内容当可信指令执行。

---

## 20. UGC / Mod：用户生成内容与安全沙箱

建于 gameplay §18（可编程性：脚本/可视化蓝图，可选 feature）与 §26（Game Features）。对标 UEFN/Roblox/Garry's Mod/Steam Workshop。

内容系统：

- **Mod 包格式（Mod Package）**：UGC = 一个受限内容包——可带 DataAsset、脚本（WASM 沙箱，gameplay §18）、资产引用；经 §7 装配器装载。
- **创作者 API 表面**：暴露给 UGC 的受限 API（生成物品、改分、放置实体），**不暴露**文件系统/网络/反射任意写——最小权限原则。
- **安全沙箱**：UGC 脚本跑在 WASM 沙箱（§18），CPU/内存/实体配额限制；资产经校验器扫描（引用完整性、禁用 API、资源上限）。
- **分发与审核**：Workshop 式订阅、本地导入、版本与依赖；审核规则化（大小/格式/API 白名单），人工审核接口留给平台。

设计要点：UGC 是**最高风险内容面**，默认拒绝一切未显式授权的能力；脚本不可触达底层系统、不可发起出站网络、不可读任意文件；配额与超时强制回收；明确不含"AI 生成内容"路径。

---

## 21. 性能工程：内容层的预算与流式

内容层不引入新的热路径机制，但**内容密度**是 AAA 性能的主要变量。本章定义内容侧的性能契约，建于 gameplay §47（帧预算）与 §20（性能策略）、`prism_diagnostic`（budget/hitch/profiler）。

- **内容预算（Content Budget）**：每类内容声明预算——同屏敌人上限、同帧掉落 roll 数、活跃任务数、HUD 刷新频率；超预算降级（降密度、合批、延迟处理）。
- **数据导向优先**：内容驱动走 Message/事件与批处理，杜绝每帧轮询（任务目标、交互检测、声望检查均事件驱动）。
- **流式与分层**：遭遇/刷怪/POI 随 World Partition（§28）流送与 Dormant 休眠；远处内容降级或卸载。
- **平台档缩放**：所有密度/特效 Cue 等级读取 `prism_app::platform_tier`，低端档自动降档，模板内置默认曲线。
- **成本可观测**：每个内容域暴露诊断计数（活跃实体、roll 次数、事件量）进 `prism_diagnostic`，Gameplay Debugger 可视化热点。
- **烘焙而非运行时解析**：掉落表/技能树/任务图在打包时烘焙为紧凑运行时结构，避免运行时解析 schema。

---

## 22. 易用性与创作者体验

易用性是本套件的核心 KPI——目标是"不写 Rust 也能产出可玩内容"。

- **三档作者分层**：策划（填表/调曲线）、关卡/叙事（编排图/树）、工程（写内容包与自定义能力）；每档有对应面板与 API 深度。
- **零配置默认**：选模板即跑；每个内容系统都有"合理默认值"，空资产也不崩（给占位内容 + 校验警告）。
- **即时反馈**：所有内容编辑热重载进 PIE（§24），编辑器内可视化叠层（交互点、刷怪点、连招窗口、任务态）。
- **强校验与明确报错**：校验器（§5）在编辑/CI/打包三处拦截脏内容；报错指向具体资产与字段，不是运行时崩溃。
- **示例驱动**：每个模板与内容系统附可运行示例（Bevy examples 文化），文档即代码。
- **可发现性**：内容资产带分类/标签/搜索元数据，大项目也易检索（对标 UE 资产浏览器）。

---

## 23. 模块化与可维护性

- **一域一 crate、内容包为单元**：`prism_gc_*` 各司其职，内容以可拆内容包组织（§7），删一个包不影响其余。
- **Tag/Message 松耦合**：内容系统间不直接函数调用，经 GameplayTag 契约与 Message 事件通信，便于独立演进与测试。
- **schema 版本化**：所有 DataAsset 带版本与迁移链，长期维护不怕旧内容腐化（§5/§18）。
- **契约测试**：每个内容系统提供校验规则集 + 示例资产的回归测试，CI 跑内容 lint。
- **禁止反向依赖**：`prism_gc_*` 只依赖 base，不被 base 依赖，保持内容层可整体替换（另一套模板库可平行存在）。
- **文档纪律**：沿用仓库 SHIPPED/PLANNED 区分；本套件当前全 PLANNED，不把设计描述为已实现。

---

## 24. 迭代速度：热重载 / PIE / 内容快速试玩

迭代速度直接决定内容产能，是本套件与 Loom Studio 协同的重点。

- **热重载内容**：DataAsset 改动经 `prism_asset` 热重载即时生效，PIE 中无需重启看到新数值/新连招/新任务。
- **运行时插拔内容包**：PIE 中勾选内容包即时装/卸（§7），快速对比"开/关某系统"的手感。
- **PIE（Play-In-Editor）**：编辑器内即点即玩，支持从任意关卡/遭遇起玩、注入测试内容（给满级、给全物品）。
- **内容快照与回放**：复用 gameplay §34 时间操控/回放，录制一段战斗反复调连招数值。
- **脚本热更**：UGC/图脚本（§18/§20）WASM 热替换，调逻辑无需重编。
- **快速校验**：保存即增量校验（§5），秒级反馈断链/缺失，而非打包时才发现。

---

## 25. Crate 拆分与落地形态

延续 §4 布局，给出 crate 职责与依赖边界（全部 PLANNED）：

```text
prism_gc_core        内容包/Experience schema + 装配器（依赖 prism_gameplay_features/tags/message）
prism_gc_templates   模板脚手架 + 默认装配清单（依赖 prism_gc_core + 相关内容 crate）
prism_gc_combat      连招图/命中内容/打击感内容（依赖 prism_gameplay_abilities/feel）
prism_gc_weapons     武器原型/射击弹道/近战 moveset/改装插槽（依赖 prism_gc_combat + prism_gc_items + abilities）
prism_gc_vehicles    载具/坐骑/机甲 + 武器硬点（依赖 prism_gameplay_flow + prism_physics_* + prism_gc_weapons）
prism_gc_traversal   位移/跑酷动词库 + 表面标注 + flow（依赖 prism_gameplay_feel + CCT + prism_animation）
prism_gc_items       物品/背包/装备/掉落/货币（依赖 prism_gameplay_abilities + prism_ecs 关系 + save）
prism_gc_progression 等级/技能树/天赋（依赖 prism_gameplay_abilities/tags + items）
prism_gc_quests      任务/对话/声望/世界 flag（依赖 prism_gameplay_message/save + ui_i18n）
prism_gc_interaction 交互内容库（依赖 prism_gameplay_interaction §30）
prism_gc_encounter   刷怪/遭遇/POI/导演（依赖 prism_gameplay_sequence + partition + ai_content）
prism_gc_ai_content  AI 行为模板/战术/感知预设（依赖 prism_gameplay_ai §14/§29）
prism_gc_cine        运镜预设/过场内容（依赖 prism_camera + prism_gameplay_sequence §44/§45）
prism_gc_ui_content  菜单流/HUD 库/设置面板内容（依赖 Loom Runtime + prism_ui_* + gameplay §51）
prism_gc_audio_content 混音快照/交互音乐/对白（依赖 prism_audio_* §43）
prism_gc_liveops     赛季/活动/远程配置/战令（依赖 prism_gc_core + 平台网络 I/O）
prism_gc_ugc         Mod 包/沙箱/分发（依赖 prism_gameplay_script §18 + prism_gc_core §26）
prism_game_creator   门面：GameCreatorDefaultPlugins + 模板选择器 + 默认内容包集
```

落地原则：每个 crate 独立可用、独立 Plugin；门面提供"选模板即跑"的默认组；内容 crate 极薄（装配 + schema + 校验），重逻辑在 base；`prism_gc_core` 不依赖具体内容 crate，可被自定义模板复用。

---

## 26. 公共 API 草案

> 示意性 Rust 伪代码，表达形态与边界，非最终签名；强调"内容=数据资产，运行时=薄装配"。

内容包与装配：

```rust
// 内容包定义（DataAsset，经 prism_reflect 序列化）
#[derive(Asset, Reflect)]
pub struct ExperienceDef {
    pub id: ExperienceId,
    pub provides_tags: TagSet,     // 本包提供的 GameplayTag 契约
    pub requires_tags: TagSet,     // 依赖的 Tag 契约
    pub features: Vec<GameFeatureRef>, // 启用的 Game Features（gameplay §26）
    pub assets: Vec<AssetId>,      // 携带的内容资产清单
    pub priority: i32,             // 叠加覆盖优先级
}

// 装配器：解析依赖、检测冲突、运行时装/卸
pub trait ExperienceAssembler {
    fn activate(&mut self, exp: &ExperienceDef) -> Result<(), AssembleError>;
    fn deactivate(&mut self, id: ExperienceId);
    fn validate(&self, exp: &ExperienceDef) -> Vec<ContentDiagnostic>;
}
```

战斗内容（示意连招/命中资产）：

```rust
#[derive(Asset, Reflect)]
pub struct ComboGraph { pub nodes: Vec<ComboNode>, pub edges: Vec<ComboEdge> }
#[derive(Reflect)]
pub struct ComboNode { pub ability: AbilityId, pub anim_tag: Tag, pub hit: HitData }
#[derive(Reflect)]
pub struct ComboEdge { pub from: NodeId, pub to: NodeId, pub cancel_window: FrameRange, pub input: InputCond }
#[derive(Reflect)]
pub struct HitData { pub active: FrameRange, pub damage: f32, pub poise: f32, pub hitstop: Frames, pub knockback: Vec3, pub cue: CueRef }
```

物品与掉落（示意）：

```rust
#[derive(Asset, Reflect)]
pub struct ItemDef { pub kind: ItemKind, pub rarity: Rarity, pub equip: Option<EquipDef>, pub affix_slots: u8, pub use_ability: Option<AbilityId> }
#[derive(Asset, Reflect)]
pub struct LootTable { pub groups: Vec<LootGroup>, pub pity: Option<PityRule> }
#[derive(Reflect)]
pub struct LootGroup { pub entries: Vec<(ItemRef, Weight, CountRange)> }
```

任务与对话（示意）：

```rust
#[derive(Asset, Reflect)]
pub struct QuestGraph { pub nodes: Vec<QuestNode>, pub edges: Vec<QuestEdge> }
#[derive(Asset, Reflect)]
pub struct DialogueGraph { pub nodes: Vec<DialogueNode> }
#[derive(Reflect)]
pub struct DialogueNode { pub speaker: Tag, pub line: LocKey, pub choices: Vec<DialogueChoice> }
#[derive(Reflect)]
pub struct DialogueChoice { pub text: LocKey, pub condition: Cond, pub effects: Vec<NarrativeEffect>, pub check: Option<SkillCheck> }
```

内容校验（贯穿编辑/CI/打包）：

```rust
pub trait ContentValidator {
    fn id(&self) -> &str;
    fn validate(&self, ctx: &ContentDb) -> Vec<ContentDiagnostic>; // 断链/循环/缺 Tag/不可达
}
```

API 分层对应 §22 的三档作者：策划面对 `*Def` 数据资产；关卡/叙事面对 `*Graph` 编排；工程实现 `Validator`/自定义 `GameFeature`。

---

## 27. 路线图

分阶段，每阶段可独立交付一个可玩里程碑（全部 PLANNED）：

- **M0 地基对齐**：确认 `prism_gameplay` §26/§30/§37/§51 接缝稳定；建立 `prism_gc_core` 内容包 schema 与装配器 + 校验器骨架。
- **M1 首个模板（TPS 垂直切片）**：`prism_gc_templates` 的 TPS + `prism_gc_combat` 连招/命中 + `prism_gc_cine` 相机 + 最小 HUD；目标：能走能打能被打，PIE 热调连招；武器系统（§29 近战/射击）与基础位移/跑酷（§31）在此落地。
- **M2 RPG 内容骨架**：`prism_gc_items`（物品/掉落/装备）+ `prism_gc_progression`（等级/技能树）+ `prism_gc_quests`（任务/对话）+ 存档（§18）；目标：捡装备、加点、接任务、存读档。
- **M3 世界与 AI 内容**：`prism_gc_encounter`（刷怪/遭遇/导演）+ `prism_gc_ai_content`（行为模板）+ `prism_gc_interaction` 库 + 音频内容（§17）；目标：一个有遭遇节奏的关卡；载具/坐骑/机甲内容包（§30）在此落地。
- **M4 前端与设置**：`prism_gc_ui_content`（前端流/HUD 库/设置面板，建于 gameplay §51）+ 无障碍/本地化；目标：完整菜单到游戏的闭环。
- **M5 LiveOps 与 UGC**：`prism_gc_liveops`（赛季/远程配置/战令）+ `prism_gc_ugc`（Mod 沙箱）；目标：可运营、可被玩家扩展。
- **M6 FPS/ARPG 模板补全 + 性能/易用打磨**：补齐模板矩阵，内容预算与平台档缩放落地，作者体验与校验器完善。

每个里程碑同步更新 `prism_engine_component_gap_zh.md` 的 `prism_gc` 行状态（⬜→🟡→✅）。

---

## 28. 落地与验收：Demo 与清单

验收以**可运行 Demo + 清单**为准，不以文档为准。

Demo 目标：

- **模板 Demo**：每个模板一个 `examples/<tpl>_demo`，`cargo run` 即玩。
- **内容垂直切片**：一个整合 Demo——选 TPS 模板、装 3 个内容包、填一张武器表/掉落表/一条任务线、编排一场遭遇，全程不改框架代码、全程热重载可调。

验收清单（节选）：

- 易用性：策划能否不写 Rust 在编辑器产出一把可用新武器？（填表→热重载→PIE 可用）
- 模块化：删除"任务内容包"后游戏是否仍可运行且无残留实体？
- 性能：满密度遭遇是否守住内容预算、低端平台档是否自动降档、无 hitch 回归（`prism_diagnostic`）？
- 可维护：旧版本存档/旧 schema 内容能否自动迁移加载？
- 边界：内容层是否零新增底层机制、是否无反向依赖 base、网络是否仍走 `prism_network`？
- 安全：远程配置是否仅覆盖白名单字段、UGC 脚本是否受沙箱与配额约束？
- 创作闭环：从"选模板"到"可玩切片"是否可在无源码改动下完成？

---

## 29. 武器系统：射击 / 近战 / 改装

建于 base 的 §8（战斗/手感：连招图、命中数据、打击感）与 §9（物品/背包/装备）：**一把武器 = 可装备物品（§9）+ 招式/命中内容（§8）+ GAS 能力授予**的组装，本章不新增底层机制，只提供"武器类"的可编辑内容形态。对标 Lyra 武器系统 / COD·Destiny 的 gunsmith / DMC5·只狼 的 moveset / 怪物猎人 的武器类别分化。

内容系统：

- **武器原型（Weapon Archetype）**：类别（手枪/步枪/霰弹/狙击/弓弩/单手剑/大剑/双刀/长柄/拳套…）、持握姿态、装备即授予的 GAS 能力集（开火/瞄准/换弹/招式）、HUD 契约（弹药/准星/连段表）。纯 DataAsset，经 §9 装备通道挂载到角色 ASC。
- **射击内容（Ranged）**：开火模式（单发/连发/爆发/蓄力）、hitscan 与 projectile 两类弹道（弹速/重力/穿透/反弹）、伤害衰减曲线（距离/部位倍率）、后坐与散布曲线（随连射累积、ADS 收敛）、弹药与换弹（弹匣/备弹/战术换弹打断）、瞄准（ADS/腰射/变倍镜）。全部为可调曲线资产。
- **近战内容（Melee）**：moveset 直接复用 §8 的连招图 + 命中数据 + 判定盒；补充武器专属的招架/弹反/处决窗口（底层 i-frame/削韧见 base §37）、蓄力重击、方向派生。不同武器类别 = 不同连招图资产 + 不同手感档。
- **改装 / Gunsmith（Attachment）**：插槽模型（瞄具/枪管/枪口/弹匣/握把/枪托/配件…），每个配件 = 一个 GAS Modifier（改散布/后坐/射速/装弹/机动）+ 可选能力授予；装配即对武器属性做叠加计算。改装 UI 由 §16 从插槽 schema 自动生成（填表即出界面）。
- **武器成长与词缀**：等级/熟练度、品质词缀、模组镶嵌——复用 §9 的词缀/品质池与 §11 的成长通道，武器只是带词缀槽的可装备物品。

设计要点：

- **零新机制**：武器是 §8+§9+GAS 的"内容组合包"，不另立武器子系统；删除 `prism_gc_weapons` 不影响 base。
- **网络与反作弊**：命中判定走 base §33 的 lag compensation + 服务器权威，弹道/伤害在服务器裁决，客户端只预测表现；稳定武器实例 ID 供 `prism_anticheat` 校验。本章只做标注，不自建复制。
- **平台档调参**：散布/辅助瞄准辅助力度（非 AI、纯规则吸附）、弹道精度按 `prism_app` platform_tier 缩放；明确**不含任何 AI 瞄准/ML**。
- **数值集中可运营**：所有曲线（伤害/后坐/衰减/改装增减）集中在资产，支持 §19 LiveOps 远程平衡与 A/B。
- **调试叠层**：Gameplay Debugger 叠显弹道射线、散布锥、命中部位、改装后合成属性。

---

## 30. 载具、坐骑与机甲

建于 gameplay §7（Pawn/Controller 附身模型：Controller 可在不同 Pawn 间切换）与 `prism_physics_*`（载具/刚体仿真权威）；武器硬点复用 §29。本章提供"可骑乘/可驾驶实体"的内容形态，物理与网络权威均不在本层。对标 GTA·极限竞速（驾驶）/ BOTW·RDR2·WoW（坐骑）/ Titanfall·Armored Core（机甲）。

内容系统：

- **附身进出（Possession）**：进入/离开载具 = Controller 在角色 Pawn 与载具 Pawn 间切换附身（gameplay §7）；多座位载具用座位关系（ECS 关系）表达"驾驶位/副驾/炮位"，各座位绑定不同能力集与相机预设（§15）。
- **载具类型与操控档（Handling Profile）**：地面（轮式/履带）/悬浮/飞行/水面四大类，每类一张可调 handling 资产（加速/极速/抓地/转向/漂移/浮力/升力）。仿真由 `prism_physics` 执行，本章只持有参数与绑定，不写物理解算。
- **坐骑（Mount）**：召唤/收起、骑乘进出、体力/耐力（GAS 属性）、骑乘战斗（马上攻击 = 复用 §29 近战/射击 + 骑乘状态 Tag）、驯服/亲密度（经典规则数值）。对标 RDR2 的马匹羁绊、WoW 的坐骑召唤。
- **机甲（Mech）**：驾驶舱附身 + 重型位移（惯性/步伐冲击）+ 武器硬点 + 过热/能量/护盾（GAS 属性与冷却）；改装 = 硬点装配（腿/臂/背），每个硬点挂一件 §29 武器或模组。机甲本质 = **载具 + 硬点武器 + 过热能量**的内容组合包。
- **武器硬点（Hardpoint）**：载具/机甲上的武器挂载点 = 复用 §29 的武器原型，绑定到座位或固定炮塔；瞄准/开火权限随座位分配。

设计要点：

- **预测密集→网络只标注**：载具/机甲是高频预测对象，复制/预测/回滚走 base §33，本章仅标注同步策略与权威边界，不自建网络。
- **物理权威在 `prism_physics`**：操控手感最终由物理子系统裁决，本层只提供数值与绑定；避免双重真相源。
- **平台档缩放**：载具密度、仿真步进精度、特效（尾迹/碎屑）按 platform_tier 降档。
- **零新机制、可拆**：附身用 gameplay §7、属性用 GAS、关系用 ECS；删除 `prism_gc_vehicles` 后游戏仍可步行运行、无残留实体。
- **调试叠层**：显示 handling 曲线当前工作点、硬点占用、过热/能量状态。

---

## 31. 位移与跑酷（Traversal）

建于 gameplay §7 的角色控制器（CCT，详见 `prism_character_controller_design_zh.md`）+ base §37 手感 + §41 动画（Motion Matching / Root Motion / Motion Warping）。本章把"高级位移动词"做成可复用内容库，底层运动与贴合由动画/CCT 子系统执行。对标 镜之边缘 / 刺客信条 / 消逝的光芒 / Titanfall 的位移手感与 flow。

内容系统：

- **位移动词库（Traversal Verbs）**：vault（翻越）/ mantle（攀上）/ climb（攀爬）/ wall-run（墙跑）/ slide（滑铲）/ grapple（抓钩）/ swing（摆荡）/ ledge-grab（抓边）等，每个动词 = 一个 GAS 能力 + §41 motion warping（把动画末端对齐到实际落点/边缘）+ 表面探测触发条件。纯内容，可按模板增删。
- **表面可达性标注（Surface Tags）**：关卡几何上标注"可翻越/可攀爬/可墙跑/可抓钩点"的 Surface Tag + 探针，运行时用物理射线/体检测（复用 §12 检测与物理射线）匹配动词触发，无需每物件手工挂脚本。
- **手感宽容（Assist）**：预输入缓冲、coyote time（离地后仍可跳的宽容帧）、落点吸附/边缘捕捉——数值化为资产，底层执行走 §37。
- **流动动量（Flow）**：连续位移的保速/增速曲线（墙跑接跳跃保持动量、滑铲接翻越不掉速），对标镜之边缘/Titanfall 的 flow；纯曲线资产。
- **攀爬系统（Climb）**：自由面攀爬与点位攀爬两模式，体力消耗 = GAS 属性（对标 BOTW）；坠落/抓握由动画状态 + CCT 约束。

设计要点：

- **零新机制**：位移动词 = GAS 能力 + 动画 warping + 表面探测的内容组合，不新增运动内核。
- **确定性 + 预测**：动词触发与位移遵循 §19 确定性，联机预测/回滚走 §33；落点对齐统一依赖 §41 的 motion warping 作为唯一真相源，避免脚本各自硬编码偏移。
- **可访问性联动**：自动翻越/简化位移等辅助项作为纯规则开关，联动 §51 游戏设置（无障碍），不改动词本身。
- **平台档缩放**：探针密度与 warping 迭代精度可按 platform_tier 降档。
- **调试叠层**：叠显可达表面 Tag、当前触发的动词、warping 目标点与动量曲线工作点。

---

## 32. 术语表

- **Game Creator**：本套件工作名；建于 `prism_gameplay` 之上的游戏模板/内容系统/创作工作流层，Prism 生态的 Lyra 等价层。
- **内容包 / Experience**：可运行时插拔的内容组织单元，建于 Game Features（gameplay §26），声明 Tag 契约与资产清单。
- **模板 / Template**：预装配的内容包清单 + 默认资产 + 示例关卡，新项目起点（对标 Lyra/UE 模板）。
- **DataAsset（内容资产）**：内容真相源，经 `prism_reflect` 序列化、`prism_asset` 热重载、带 schema 版本。
- **校验器 / Validator**：内容 lint 规则集，在编辑/CI/打包三处拦截脏内容（断链/循环/缺 Tag/不可达）。
- **连招图 / Combo Graph**：战斗内容的招式有向图，节点=招式+命中数据，边=取消窗口+输入条件。
- **掉落表 / Loot Table**：分层权重掉落资产，支持保底（pity），纯经典随机。
- **任务图 / Quest Graph**：任务目标与分支的有向图资产，状态落到 Tag + 组件，由 Message 推进。
- **声望 / Reputation**：阵营数值 + 门槛事件，经经典规则变化，无 AI。
- **LiveOps**：赛季/活动/远程配置的内容侧运营，经内容包时间化 + 白名单远程覆盖。
- **UGC / Mod**：用户生成内容，受限内容包 + WASM 沙箱（gameplay §18）+ 配额，最小权限。
- **平台档 / platform_tier**：`prism_app` 的平台能力分级，驱动内容密度/特效等级缩放。
- **PIE（Play-In-Editor）**：编辑器内即点即玩，支撑热重载快速迭代。
- **四联架构**：Loom Studio（编辑器）+ Loom Runtime（运行时）+ Prism Gameplay（玩法）+ Game Creator（内容），自上而下单向依赖。
- **武器原型 / Weapon Archetype**：武器类别的内容定义（§29），= 可装备物品 + 招式/命中 + GAS 能力的组装。
- **改装 / Gunsmith（Attachment）**：武器插槽配件内容（§29），每件 = GAS Modifier + 可选能力，UI 由插槽 schema 自动生成。
- **载具操控档 / Handling Profile**：载具一类可调操控参数资产（§30），仿真由 `prism_physics` 执行，本层只持参数。
- **武器硬点 / Hardpoint**：载具/机甲上的武器挂载点（§30），复用 §29 武器原型并绑定座位/炮塔。
- **位移动词 / Traversal Verb**：翻越/墙跑/抓钩等高级位移内容单元（§31），= GAS 能力 + motion warping + 表面探测。
- **运动扭曲 / Motion Warping**：把动画末端对齐到实际落点/边缘的动画技术（§41 权威），位移动词落点贴合的唯一真相源（§31）。
