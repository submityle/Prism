# Prism Scene 顶级次世代 AAA 级场景/预制体系统设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **场景/预制体模型 + 实体子树模板 + 属性级覆盖（override）+ 嵌套/变体继承 + 数据导向快速实例化 + 常驻档位与按需物化 + 子场景流送 + 热重载 + 确定性回放** 内核设计。它是 `bevy_scene` 的自研替代，是「作者态内容」接入 ECS 的唯一一等模型，也是世界系统的「作者内容砖块」。
> 借形态不抄码。借鉴：
> - **统一场景模型**：Godot（一切皆场景、场景即预制体、场景可嵌套与继承、local-to-scene 资源、`PackedScene` 预打包）
> - **属性级覆盖与嵌套变体**：Unity（Prefab / Prefab Variant / Nested Prefab / modifications 稀疏修改跟踪 / 一键还原与应用回基体 / Prefab Stage）
> - **流送与持久化粒度**：Unreal（World Partition 隐式空间网格 / Level Instance / Packed Level Actor / HLOD 分层代理 / OFPA 每对象文件化协作粒度 / 数据层开关）
> - **数据驱动模板**：Unreal 数据驱动 archetype 与 flecs `IsA` prefab 继承（模板即数据、非脚本类）
> - **数据导向实例化**：数据库列存 SoA（Arrow/Parquet 形态）按 archetype 列块 memcpy、COW 结构共享、mmap 零解析
> - **确定性回放**：事件溯源（Event Sourcing，仅借「delta 可确定性重放」形态，不引入其并发真相源）
> 本文为纯经典数据结构 + 序列化 + 实例化路线，**不含任何 AI/ML 内容**，不含任何 Unreal/Unity 源码或衍生代码。

- 版本： v0.3（设计阶段，未进入编码；本轮 §24 深化为 14 子节：预制体接口暴露参数(§24.1)/构造式参数化预制体(§24.2)/散布规则库(§24.3)/菱形继承 C3 消歧(§24.4)/prefab LOD 链(§24.5)/稳定子对象寻址(§24.6)/引用完整性级联(§24.7)/三方合并软锁(§24.8)/数据层+Level Instance 对称(§24.9)/Prefab Stage apply-revert(§24.10)/HLOD 代理(§24.11)/分帧实例化去重(§24.12)/网络复制+序列绑定(§24.13)/场景查询标签代数(§24.14)；承前深化：§8.5 列去重与缓存微架构、§11.4 HLOD 簇代理、§12.3 预测式预取、§16.4 网络复制对称、§19.7 撤销/重做与安全重命名；clean-slate 全新 crate，无旧 API 需保留。承接 `prism_ecs`/`prism_asset`/`prism_transform` 已定稿设计，向下复用其身份/存储/依赖/句柄/层级核；向上为重做后的世界系统提供「作者内容单元」契约）
- 适用引擎： Prism（后 Bevy 时代，独立运行时）
- 关键依赖： `prism_ecs`（批量预留/chunk 列写入/`serialize`/`determinism`）、`prism_asset`（`StableGuid`/软引用/依赖闭包就绪/EDL/单文件资产包/热重载 `Modified`）、`prism_transform`（关系驱动层级）、`prism_reflect`（字段路径/override 值序列化/编辑器属性枚举）、`prism_tasks`（并行实例化）、`prism_diagnostic`（计数器/trace）
- 层级定位： 内容层 L5；上接世界系统（流送基座）/渲染/物理/脚本/编辑器，下接 `prism_ecs`/`prism_asset`/`prism_transform`
- 明确约束： 核心 `no_std + alloc`；`std`/`async_io`/`hot_reload`/`editor`/`determinism`/`persist` 为 feature；工作区 `forbid(unsafe_code)`；**不依赖任何 `bevy_*` crate**
- 架构立场： **scene 是单一逻辑真相**（作者编辑的实体内容），世界系统降级为架在其上的「空间流送 + 常驻残留 + 持久化」基座，不另立平行真相源（取代旧 `prism_world_system_design_zh.md` 的 WorldDB-as-truth 定位，该文将按本 crate 契约重写为去 Bevy 多 crate 版本）

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier）
4. 分层架构
5. 核心模型：Scene / Template / Instance / Override 三层
6. 身份与引用（LocalEntityId / 具名引用 / StableGuid / PersistentId / 跨场景软引用）
7. 覆盖系统：属性级 override / 结构 override / 嵌套变体继承
8. 实例化管线与数据导向快路径
9. 嵌套、变体与组合（prefab 套 prefab / variant / level）
10. 序列化格式（架在 prism_asset 单文件资产包之上）
11. 常驻档位与按需物化（residency tier / proxy ⇄ ECS）
12. 子场景流送与世界系统接缝（additive streaming）
13. 热重载与 override 保全传播
14. 确定性与回滚
15. 持久化与存档（override delta / PersistentId / 与 world_persist 分工）
16. 事件、变更与世界集成
17. 可观测性与诊断
18. 性能工程
19. 易用性与 API 人体工学（具名实体 / override DSL / prelude）
20. crate 分层与模块布局
21. 契约、不变量与版本化
22. 路线图（M0–M6）与基准即规格
23. 诚实边界与风险
24. AAA 高级功能增补
25. 基准即规格：量化目标与验收门槛
26. 测试与验证策略

---

## 1. 设计哲学与目标

场景系统是「作者在编辑器里摆出来的世界」与「运行时活在 ECS 里的实体」之间的桥梁。关卡、预制体、敌人模板、UI 层、测试场景——它们都不应各写一套「序列化实体→反序列化→接层级→改一个字段→重新实例化」。`prism_scene` 把这件事统一成**一个一等概念 Scene（实体子树模板）**：作者只认 scene，关卡与预制体不是两种东西，而是同一模型的组合与被组合（借 Godot 的「一切皆场景」）。

**一句话定位**：`prism_scene` 是 Prism 的「作者态内容真相」——以实体子树模板为单位，提供稳定的局部身份、属性级覆盖、嵌套变体继承、数据导向的确定性快速实例化、保 override 的热重载、以及常驻档位 + 按需物化的大规模支撑；一次 `spawn_scene()`，关卡/预制体/流送块/海量 proxy 走同一套语义。

与旧 `bevy_scene` 的根本差异在于三条翻转：

| 维度 | `bevy_scene`（弃用） | `prism_scene`（clean-slate） |
|---|---|---|
| 序列化 | 反射 serde（逐实体逐组件、慢、体积大、需运行时类型解析） | schema 稳定二进制 + 稳定字段 id + 列块 mmap 零解析；热路径零反射 |
| 实例化 | 逐实体 spawn + 逐组件 insert | 按 archetype 列块 memcpy + 一次性 remap（数据导向快路径） |
| 覆盖 | 场景覆盖粒度粗、丢失基体更新 | **属性级稀疏 override**（Unity modifications 形态）+ 嵌套变体叠栈 |
| 规模 | 实体即对象，百万静态撑爆 archetype | 常驻档位 + proxy + 按需物化，交互内容才进 ECS |
| 真相定位 | 场景文件为中心、无空间索引 | scene 是逻辑真相，世界系统为其上的流送基座 |

四条总目标（按权重）：

1. **易用（本设计首要权重）**：属性级 override 做满——实例改一个字段，基体后续更新其余内容时实例自动吃到、被改字段纹丝不动；具名局部实体让 override API 用名字而非裸编号；`#[derive]` 不强加于用户（Scene 是数据资产），组件只需 `prism_reflect` 可见即可被覆盖/热改；失败软降级发诊断不 panic。所见即所得的编辑器 live 迭代是第一公民。
2. **性能**：实例化走「列块 memcpy + 偏移驱动 remap」热路径，烘焙期把纯属性 override 预合成进计划、运行期不碰反射；COW 载荷让 N 个相同 prefab 的内存亚线性增长；archetype 预留 + `prism_tasks` 分片并行；mmap 零解析冷加载。
3. **效果（能力）**：嵌套 prefab、变体继承、关卡组合、additive 子场景流送、常驻档位（交互/流送/proxy）、proxy ⇄ ECS 按需物化、保 override 热重载、确定性回放——覆盖从「单个预制体」到「行星级大世界作者内容」的全谱。
4. **确定性**：固定模板 GUID + schema 版本 + 实例化参数 ⇒ 实例化顺序、Entity 分配、`PersistentId`、override 重放全可复现，对接 `prism_ecs` 的 `determinism`/rollback 档位与存档/网络。

设计信条（承接 Prism 范式）：

- **一切皆 scene**：level = 一棵顶层 scene（散放实体 + 嵌套实例节点）；prefab = 可被嵌套的 scene；variant = 「基体软引用 + override 集」的薄 scene。单一模型，组合对称。
- **override 是稀疏 delta**：实例只存「相对模板的差异」，不存全量；基体是唯一数据源，实例是叠加其上的透明层。
- **成本正比于差异**：没有 override 的实例几乎零额外内存（COW 共享模板列）；没被改的字段热重载自动传播。
- **三重门控**：capability × quality tier × feature flag——属性级 override / 热重载 / 流送 / 持久化 / 按需物化任意路径可降级或编译期裁剪。
- **诚实标注**：二进制模板不可逐行 diff、整模板粒度的某些合并局限、沙盒无 GPU 的性能目标均为 design target——如实写明（§23）。

---

## 2. 参考产品取舍

借形态不抄码，只在**架构/算法层**借鉴，不复制任何源码。

| 产品 | 借鉴点 | Prism 落法（本文） |
|---|---|---|
| **Godot：一切皆场景 / 场景即预制体 / 场景嵌套与继承 / `PackedScene`** | level 与 prefab 统一为「实体子树模板」，组合与被组合对称；`PackedScene` 预打包为实例化快路径 | §5 单一 Scene 概念；§8 实例化计划即 `PackedScene` 形态 |
| **Godot：local-to-scene 资源** | 某些资产按实例唯一化（如每实例一份可变材质） | §11 per-instance 复制标记 |
| **Unity：Prefab modifications 稀疏修改跟踪** | 实例只存「相对基体的属性 delta」，基体更新非改字段自动传播 | §7 属性级 override 核心 |
| **Unity：Prefab Variant / Nested Prefab** | 变体 = 基体 + override；嵌套 prefab 多层叠栈 | §7/§9 override 叠栈、变体即薄模板 |
| **Unity：一键还原 / 应用回基体 / Prefab Stage** | 编辑器可枚举 override 做还原、应用、隔离编辑 | §7 冲突与可视化、§19 编辑器 API |
| **Unreal：World Partition 隐式空间网格 / 数据层** | 子场景按空间网格隐式流送、数据层开关 | §12 流送、§11 档位（委托世界系统基座触发） |
| **Unreal：Level Instance / Packed Level Actor** | 子关卡作为可复用、可流送、可烘焙打包的实例单元 | §9 嵌套实例节点、§8 烘焙打包 |
| **Unreal：HLOD 分层代理** | 远景用廉价分层代理替代,交互时才物化 | §11 proxy 档位 + 按需物化 |
| **Unreal：OFPA 每对象文件化** | 每被改对象的 override 独立持久、协作粒度细 | §15 持久化（override delta + PersistentId） |
| **数据库列存 SoA（Arrow/Parquet）** | 按 archetype 列块连续、只拉需要的列、mmap 零解析、内容寻址去重 | §8 实例化计划列式、§10 序列化 |
| **COW / 持久化数据结构** | 共享不可变模板列，写时才复制 | §8 COW 载荷、§18 内存 |
| **事件溯源（Event Sourcing）** | override 作为可确定性重放的 delta 序列（仅借回放形态，不引入并发真相源） | §7 override replay、§14 确定性 |
| **flecs `IsA` prefab 继承** | 模板即数据、继承链、通配 | §9 变体继承（数据层，不走运行时查询） |
| **UE Nanite / Horizon-Decima：tile 流式代理** | 远景簇用合并代理/impostor、近景物化真实例 | §11.4 HLOD 簇 + 切换契约 |
| **Parquet / git：内容寻址去重** | 相同列块全库单份、内容哈希索引、冷热分段 | §8.5 列去重与缓存微架构 |

**综合**：以 **Godot 统一场景模型** 为骨架，**Unity 属性级 override + 嵌套变体** 为易用性皇冠，**Unreal World Partition / Level Instance / HLOD / OFPA** 为规模与流送与协作粒度，**列存 SoA + COW + mmap** 为实例化与内存性能，**事件溯源回放形态** 为确定性与持久化。全部走 feature/档位门控，默认只付「加载 + 实例化 + 整树卸载」的最小成本。

---

## 3. 档位化（capability / quality tier）

三重门控：**capability（平台/构建能力）× quality tier（运行时质量档）× feature flag（编译期裁剪）**。任何高级路径都可降级或编译期移除，核心路径永远可用。

| 能力 | feature | 默认 | 降级形态 |
|---|---|---|---|
| 核心模型（模板/实例/实例化/整树卸载） | 恒开（`no_std+alloc`） | 开 | 不可降级，是最小内核 |
| 属性级 override | `override`（默认开，易用优先） | 开 | 关则退化为「整组件覆盖」粗粒度 |
| 结构 override（增删实体/组件） | `override` | 开 | 关则实例只能改属性不能改结构 |
| 嵌套/变体 | 恒开 | 开 | 单层无变体仍可用 |
| 异步加载/流送 | `async_io` | 开（std） | 关则仅同步 `spawn_scene_now` |
| 热重载 | `hot_reload` | 开（dev） | release 关，省监听与 diff 开销 |
| 编辑器集成（override 枚举/导出模板） | `editor` | 关 | 关则无反射枚举、无运行时导出 |
| 确定性/回滚 | `determinism` | 关 | 关则实例化顺序不保证跨平台一致 |
| 持久化/存档 | `persist` | 关 | 关则无 `PersistentId`/override delta 存档 |
| 按需物化（proxy ⇄ ECS） | `residency`（需世界系统基座） | 关 | 关则所有 scene 内容直接整体实例化 |

quality tier（运行时）示例：实例化预算（每帧物化上限）、流送半径、proxy 密度、COW 粒度均随 tier 调。capability 示例：无 `prism_asset` 的 `gpu_stream` 能力时，proxy 走 CPU 常驻而非 GPU 直传。

---

## 4. 分层架构

```
作者态 ───────────────────────────────────────────────
  prism_scene            唯一内容模型:Template / Instance / Override / 嵌套 / 变体
         │               (作者只编辑 scene;level = 顶层 scene + 嵌套实例节点)
         ▼
烘焙期 ───────────────────────────────────────────────
  prism_scene_bake       scene 图 → 扁平化「实例化计划」+ 预合成纯属性 override
         │               + 空间切片 + 常驻档位标注(interactive/streamed/proxy)
         ▼
运行态 ───────────────────────────────────────────────
  prism_scene(runtime)   spawn_scene / override 读写 / 热重载 / despawn_scene
         │               ▲ 依赖闭包就绪事件(prism_asset EDL)
         ▼               │
  prism_ecs              批量预留 + chunk 列写入 + serialize + determinism
  prism_asset            StableGuid / 软引用 / 依赖图 / 单文件包 / Modified
  prism_transform        关系驱动层级接层级
         ▲
         │ (本 crate 向上提供「作者内容单元」契约)
世界系统基座(另文,去 Bevy 多 crate)─────────────────────
  prism_world_stream     空间索引 + 流送 VM + GPU 直传 → 触发 Tier B 实例化
  prism_world_residency  proxy ⇄ ECS 升降级 → 调用本 crate spawn/despawn + override
  prism_world_persist    存档/多人 → 存本 crate 产出的 override delta + PersistentId
```

模块职责：

- **`prism_scene`（本文核心 crate）**：模板/实例/override 的运行期数据结构与 API、实例化引擎、热重载、确定性重放。`no_std + alloc`，`forbid(unsafe_code)`。
- **`prism_scene_bake`（配套 crate，`std`）**：离线把 scene 作者图编译为 `InstantiationPlan` + 预合成 override + 空间切片 + 档位标注，产出 `.scene.prism`/`.prefab.prism` 资产包。对标 Godot `PackedScene`、UE Level 打包。
- **边界清晰**：本 crate 不定义组件语义（只搬运 reflect 可见数据）、不做文件 IO（经 `prism_asset` + VFS）、不做格式解码、不做大世界空间索引/流送调度（归世界系统基座，本 crate 只被其调用）。

---

## 5. 核心模型：Scene / Template / Instance / Override 三层

只有一个一等概念 **Scene**（= 实体子树模板），运行时分三层表示：

```
Template（模板,磁盘/常驻态,不可变,可共享,可 mmap）
   │  含:InstantiationPlan + 依赖清单 + 具名表 + 嵌套清单 + 档位标注
   │ spawn_scene()
   ▼
Instance（实例,运行态,活在 ECS 的实体子树）
   │  根实体挂 SceneInstance{ template, root, remap, overrides }
   ▼
Override（覆盖,稀疏 delta,挂在 Instance 上)
      含:属性覆盖集 + 结构覆盖集(§7)
```

### 5.1 Template（模板）

```rust
/// 一个 scene 资产的运行期常驻表示；由 prism_scene_bake 产出、prism_asset 加载。
pub struct SceneTemplate {
    guid: StableGuid,                 // 复用 prism_asset 身份,随文件走
    plan: InstantiationPlan,          // 扁平化实例化计划(§8)
    names: NameTable,                 // 具名局部实体: name → LocalEntityId
    nested: Box<[NestedSlot]>,        // 嵌套实例节点(§9)
    deps: Box<[SoftHandle<ErasedAsset>]>, // 依赖闭包(嵌套子模板 + 引用的网格/材质/音频…)
    schema: SchemaVersion,            // reflect 字段 id/版本,迁移用(§21)
    tiers: ResidencyPlan,             // 常驻档位标注(§11),可空
}
```

Template **不可变、可被任意多实例共享**。它本身是一种 `prism_asset` 资产类型（`Asset for SceneTemplate`），因此天然获得 `Handle`/`SoftHandle`/依赖图/热重载/单文件包。

### 5.2 Instance（实例）

实例化在 ECS 里产出一棵实体子树；根实体挂一个锚组件：

```rust
#[derive(Component)]
pub struct SceneInstance {
    template: SoftHandle<SceneTemplate>, // 来源模板(软引用,支持卸载/热重载)
    remap: RemapTable,                   // LocalEntityId → Entity(本实例专属,见 §6)
    overrides: Option<Box<OverrideSet>>, // 该实例的覆盖集(无覆盖时为 None,零成本)
    persistent: Option<PersistentId>,    // 存档/网络定位(§15,persist 档位)
    residency: ResidencyState,           // 当前常驻档位(§11)
}
```

`remap` 让「模板内的 LocalEntityId」映射到「本实例的活 Entity」；`overrides` 为 `None` 时实例零额外内存（COW 共享模板列，§8）。

### 5.3 Override（覆盖）

见 §7。核心是**稀疏**：只存被改过的字段/结构,基体是唯一数据源,实例是叠加其上的透明层。

**对称性（借 Godot）**：Template 内部可引用别的 Template（§9 `NestedSlot`）。一个 level 就是一棵顶层 Template：散放若干实体 + 若干嵌套实例节点（各带局部 override）。level 与 prefab 不是两套机制。

---

## 6. 身份与引用

四种 id，各司其职——这是「实例化后内部引用不串台、跨会话可定位」的正确性基石。

| id | 作用域 | 位宽/形态 | 用途 |
|---|---|---|---|
| `LocalEntityId` | 单个 Template 内 | `u32`，从 0 连续 | 模板内实体编号；模板内部的实体值字段（父指针/关系目标/挂点）全存它 |
| 具名引用 `NameTag` | 单个 Template 内 | 字符串 → `LocalEntityId`（烘焙期入 `NameTable`） | 让 override/查询 API 用名字而非裸编号（§19） |
| `AssetIndex`/`Entity` | 进程内 | `prism_ecs` 代际槽 | 实例化后的活实体 |
| `StableGuid` | 跨构建 | 128-bit | 模板本身的身份（复用 `prism_asset` 内嵌 GUID） |
| `PersistentId` | 跨会话/存档/网络 | 128-bit（确定性派生） | 定位「某实例内的某实体」用于存档回灌/网络复制（§15） |

### 6.1 实例化 remap（内部引用不串台）

模板的 `InstantiationPlan` 里所有「实体值字段」存 `LocalEntityId`。实例化：

1. 向 `prism_ecs` **批量预留** `plan.entity_count` 个实体；
2. 建 `RemapTable`（本质 `Box<[Entity]>`，下标即 `LocalEntityId`，O(1) 查）；
3. 列块 memcpy 时，对标记为「实体值」的字段槽做一遍 remap（偏移在烘焙期已算好，无反射）。

**关键不变量**：实例化后，模板内部的引用**仍指向本实例内部**（经本实例 `remap`），绝不串到别的实例。多次实例化同一模板，各自独立 remap 表、各自独立子树。

### 6.2 `PersistentId`：确定性派生

```
PersistentId = hash128(
    实例根所在的 StableGuid 链(顶层场景 GUID → 嵌套路径的各子模板 GUID)
    ‖ 该实体的 LocalEntityId 在其所属模板内的路径
)
```

- **确定性**：相同作者内容 + 相同实例化位置 ⇒ 相同 `PersistentId`，跨会话/跨机一致。
- **稳定**：实体在场景内移动/改名不变（基于 LocalEntityId 而非运行期 Entity）；只有作者重排 LocalEntityId 才变（迁移时提供映射，§21）。
- **用途**：存档只需存「`PersistentId` → override delta」；回灌时按 id 定位实体施加 delta；网络复制按 id 寻址（OFPA 形态）。

### 6.3 跨场景软引用

模板 A 想引用「别的场景里的某实体」→ 存 `CrossSceneRef { scene: SoftHandle<SceneTemplate>, local: LocalEntityId }`。实例化按 `prism_asset` 依赖闭包就绪后解析为活 `Entity`；目标未加载/已卸载时降级为 `None` 并发 `DanglingRef` 诊断，**不 panic**。跨场景引用是弱约束，生命周期由各自实例独立管理。

---

## 7. 覆盖系统（皇冠：易用性天花板）

本设计把 override 粒度**做满到属性级**（易用优先，§1 首要权重）。目标是 Unity modifications 级体验：**实例改一个字段，基体 prefab 后续更新其余内容时实例自动吃到，被改字段保留**。做好这个，scene 才算 AAA。

### 7.1 两类 override

**(a) 属性覆盖（property override）** —— 走 `prism_reflect` 字段路径：

```rust
pub struct PropertyOverride {
    target: LocalEntityId,       // 哪个实体
    component: ComponentTypeId,  // 哪个组件
    path: FieldPath,             // reflect 字段路径,如 "transform.translation.x"
    value: ReflectValue,         // 覆盖值(序列化态)
}
```

一个实例的属性 override 集**稀疏**（只存被改过的）。实例化/重建时：先按模板铺基值，再 replay 属性 override 集；被覆盖的 `(target, component, path)` 打标记，基体热重载时**跳过**这些标记、其余照更新（§13）。

**(b) 结构覆盖（structural override）** —— 增删实体/组件：

```rust
pub enum StructuralOp {
    AddEntity    { local: LocalEntityId, parent: LocalEntityId, archetype: ArchetypeSignature, data: RowBlob },
    RemoveEntity { local: LocalEntityId },                 // 删掉模板里的实体
    AddComponent { target: LocalEntityId, component: ComponentTypeId, data: ComponentBlob },
    RemoveComponent { target: LocalEntityId, component: ComponentTypeId },
}
```

`AddEntity` 分配的 `LocalEntityId` 落在模板编号空间之外的保留段（§6），避免与基体冲突、且热重载重排基体时实例新增实体不受影响。

### 7.2 override 集结构与去重

```rust
pub struct OverrideSet {
    properties: SparseMap<(LocalEntityId, ComponentTypeId, FieldPath), ReflectValue>,
    structural: Vec<StructuralOp>,
}
```

- 属性 override 按 `(target, component, path)` 唯一键去重，后写覆盖前写。
- 存储用稀疏结构（`SparseMap`），没被改的字段零占用——这是「成本正比于差异」的落点。

### 7.3 嵌套变体继承（Prefab Variant）

变体 B「继承」A，本质上 **B 的模板就是「A 的 `SoftHandle` + 一组 override」**（薄模板，不复制 A 的数据）：

```
实例化 variant 实例 I(基于 B,B 基于 A):
  1. 实例化 A 的 plan            → 基值
  2. replay B 的 override 集      → 变体层
  3. replay I 自己的 override 集  → 实例层
```

override 可**叠栈多层**（A ← B 变体 ← C 变体 ← … ← 实例），每层都是稀疏 delta，自底向上重放。这就是 Unity nested prefab variant 的形态，但用 `prism_reflect` 字段路径表达、纯数据、可确定性重放（§14）。

**冲突语义**：上层 override 覆盖下层同键；结构删除（`RemoveEntity`）后，针对被删实体的下层属性 override 成为「孤儿」，重放时丢弃 + 发 `OrphanOverride` 诊断（§23）。

### 7.4 编辑器可视化与操作（`editor` 档位）

override 集是结构化、可枚举的，为编辑器预留一等操作（借 Unity Prefab Stage）：

- **枚举**：列出某实例所有 override（哪些字段/结构被改），复用 `prism_reflect` 做属性面板高亮。
- **一键还原（revert）**：删除某条 override → 下次重建回落基值。
- **应用回基体（apply）**：把实例的某条 override 写回其模板（改 `.prefab.prism`），所有兄弟实例随之更新。
- **隔离编辑（prefab stage）**：以空实例化方式单独编辑一个模板，保存即热重载所有实例。

### 7.5 预合成优化（性能,见 §8）

纯属性 override（无结构变化、无跨实例依赖）在**烘焙期可预合成**进实例化计划的列值——即把 override 直接写进该实例专属的列快照，运行期**不再 replay、不碰反射**。只有「运行时动态改的 override」或「含结构变化」才走运行期 replay。这让「作者态改好的 prefab 实例」实例化等同于「普通模板」实例化（§8 量化目标）。

---

## 8. 实例化管线与数据导向快路径

性能皇冠。核心思想：**实例化不是「逐实体 spawn + 逐组件 insert」，而是「按 archetype 列块 memcpy + 一次性 remap」**，借列存 SoA 形态（对标 Godot `PackedScene` 的预打包 + 数据库列块）。

### 8.1 烘焙期：扁平化为实例化计划

```rust
pub struct InstantiationPlan {
    entity_count: u32,
    groups: Box<[ArchetypeGroup]>,       // 按 archetype 分组,组内列式连续
    entity_valued_fields: Box<[FieldSlot]>, // 需 remap 的实体值字段位置(偏移已算好)
    hierarchy: Box<[(LocalEntityId, LocalEntityId)]>, // (child, parent)
    reserved_local_base: u32,            // 结构 override 新增实体的 LocalEntityId 起点
}

pub struct ArchetypeGroup {
    signature: ArchetypeSignature,       // 该组组件集
    locals: Box<[LocalEntityId]>,        // 组内 n 个实体
    columns: Box<[Column]>,              // 每组件一列,n 个值连续(SoA blob,对齐,可 mmap)
}
```

### 8.2 运行期：实例化热路径

```
spawn_scene_now(template, root_transform) -> SceneInstanceId:
  1. prism_ecs.reserve(entity_count)                 // 一次性批量预留,不逐个分配
  2. remap = Box<[Entity]>  (下标 = LocalEntityId)    // O(1) 映射表
  3. for group in plan.groups:
        chunk = prism_ecs.reserve_chunks(group.signature, group.locals.len())
        for col in group.columns:
            memcpy(chunk.column(col.id), col.blob)    // 整列拷贝,无逐实体 set
  4. for slot in plan.entity_valued_fields:
        patch(slot, remap[slot.local_value])          // 偏移驱动 remap,无反射
  5. 接层级(prism_transform 关系,按 hierarchy)
  6. 施加运行期 override(若有,§7.5 预合成的已在列里)
  7. 递归嵌套槽(走依赖闭包,§9/§12)
  8. 挂 SceneInstance 根组件,返回 id
```

### 8.3 性能手段细化

- **COW 载荷**：`Column.blob` 不可变、被多实例共享（`Arc<[u8]>` 形态）；只有「含 override 的列」或「含 remap 的实体值列」走写时复制，其余实例间直接共享底层缓冲。**N 个相同 prefab 的内存亚线性增长**（仅差异 + remap 表 ×N）。
- **archetype 预留**：计划带 `signature`，实例化前一次性为目标 archetype 预留容量，避免途中反复搬迁 chunk。
- **并行实例化**：多个独立实例（如一片森林 100 棵相同树）走 `prism_tasks` 分片并行；各实例 remap 表互不相交、写入各自 chunk 段，**无锁**。
- **零反射热路径**：reflect 只在烘焙期用于生成计划、算字段偏移、预合成 override；运行期全是偏移驱动 memcpy + remap。
- **mmap 零解析冷加载**：`Column.blob` 对齐布局，`prism_asset` 单文件包可直接映射为运行期 column，不解析（§10）。
- **确定性顺序**：`LocalEntityId` 连续、remap 按序、Entity 批量预留顺序确定 ⇒ 对接 `prism_ecs` 的 `determinism`/rollback（§14）。

### 8.4 异步实例化（`async_io` 档位）

当模板或其嵌套/引用资产未常驻时，`spawn_scene(soft_handle)` 走异步：`load_soft` → `prism_asset` 依赖闭包就绪事件 → 回到 §8.2 热路径实例化 → 发 `SceneInstantiated` 事件。调度走 `prism_asset` EDL，不自建。

### 8.5 内存与缓存微架构（content-addressed 去重 / SIMD remap / 预取）

更细的性能工程，借数据库列存引擎（Parquet/git 内容寻址）与 Nanite 流式思路，全在 CPU 侧经典数据结构，无 GPU 依赖假设。

- **内容寻址列去重**：烘焙期对每条 `Column.blob` 取内容哈希，全资产库内相同列只存一份（相同网格引用列、默认组件列大量重复）；实例化按哈希索引直接映射 mmap 页，不复制。磁盘与常驻双降。
- **冷热列分段**：列按 64 字节对齐，按组件访问热度分「热列/冷列」两段（借 ECS archetype 的 hot/cold split）；remap 只触碰「实体值列」这一热段、不扫冷段，减少 cache miss。
- **SIMD remap**：`entity_valued_fields` 的偏移表排序后批量加偏移，remap 为「向量化 gather + add」而非逐字段分支；`LocalEntityId → Entity` 表连续、下标直取、无哈希。
- **软件预取与 madvise**：实例化按组顺序 `prefetch` 下一列块首址掩盖内存延迟；mmap 冷加载对首批交互列 `madvise(WILLNEED)`、对 proxy 列 `madvise(RANDOM)` 惰性拉取。
- **双缓冲 remap 快照**：确定性档位下 remap 表可快照供回滚复用，不重算（§14）。
- **跨嵌套零拷贝**：嵌套子模板的只读列与父实例共享同一 mmap 页，仅差异 + remap 独立（§8.3 COW 延伸到跨层）。
- **分配器友好**：实例化批量预留一次成型（§8.2），避免途中 realloc；despawn 整树归还连续区间给 `prism_ecs` 空槽池，降碎片。

---

## 9. 嵌套、变体与组合

### 9.1 嵌套实例节点

Template 内一个嵌套槽：

```rust
pub struct NestedSlot {
    child: SoftHandle<SceneTemplate>, // 子模板(软引用,可独立卸载/热重载)
    root_local: LocalEntityId,        // 子树实例化后挂到本模板哪个实体下
    overrides: OverrideSet,           // 本模板对子模板的覆盖(变体层)
    residency: ResidencyHint,         // 该嵌套的档位提示(§11)
}
```

- **深度嵌套**：A 含 B 含 C，实例化按依赖闭包**自底向上就绪**、**自顶向下接层级**。
- **防环**：嵌套图在烘焙期做环检测（复用 `prism_asset` 依赖图的 Kahn 拓扑 + 环检测），自引用/循环报错 + 诊断。

### 9.2 变体即薄模板

变体 B 的磁盘载荷只有 `{ base: SoftHandle<A>, overrides }`，不复制 A 的数据：

- **省磁盘**：变体文件极小。
- **基体更新自动继承**：改 A，所有变体 + 变体的实例在热重载后吃到（除被各层 override 的字段）。
- **多层继承**：A ← B ← C 链式，实例化按 §7.3 自底向上叠栈重放。

### 9.3 组合：level 即顶层 scene

一个关卡 = 一棵顶层 Template：

- 散放实体（关卡专属几何/灯光/触发器）直接在 plan 的 groups 里；
- 大量重复内容（树、路灯、建筑模块）作为嵌套实例节点，各带位置/微调 override；
- 子关卡（室内、建筑内部）作为可流送的嵌套槽（§12），进范围才实例化。

这正是 UE Level Instance / Packed Level Actor 的形态：关卡可复用、可嵌套、可烘焙打包，但统一在 scene 模型下，无单独「关卡」类型。

---

## 10. 序列化格式（架在 prism_asset 单文件资产包之上）

**不另造格式**，直接复用已定稿的 UE 式单文件 `.prism` 资产包（见 `prism_asset_design_zh.md` §13.1）。Scene 是一种资产类型，其载荷段按本节布局。

### 10.1 包内布局

| 段 | 内容 | 对应 §prism_asset |
|---|---|---|
| 身份段 | `StableGuid`（随文件走，移动/改名零成本） | 内嵌 GUID |
| ImportSettings 段 | 烘焙参数（压缩档/档位标注策略/预合成开关） | ImportSettings |
| SourceRecord 段 | 作者源文件哈希/定位（支持重导入） | SourceRecord |
| Dependencies 段 | 所有 `SoftHandle`（嵌套子模板 + 引用的网格/材质/音频…），自动进依赖图 | Dependencies |
| Payload 段 | **实例化计划（或变体 delta）+ override 集 + 嵌套清单 + 具名表 + schema 版本** | Payload |

### 10.2 schema 稳定与零解析

- 组件布局靠 `prism_reflect` 的**稳定字段 id + 版本号**，字段增删走版本迁移（§21），不靠顺序；
- 列 blob **对齐布局**，可直接 mmap 为运行期 `Column`，**不解析**（§8.3）；
- override 的 `FieldPath`/`ReflectValue` 以稳定字段 id 编码，跨版本可迁移。

### 10.3 编辑态/运行态分离

- **编辑态**：`.scene.prism`/`.prefab.prism`，含完整 override 集 + 具名表 + 可重导入源记录，供编辑器热改。
- **运行态**：发行打包时经 `prism_scene_bake` 产出预合成、剥离编辑元数据的紧凑计划（可进 `prism_asset` 的 `.ucas` 容器），冷加载 mmap 即用。
- **文本旁视图**：复用 `prism_asset` 的文本旁视图做 diff（二进制主体不可逐行 diff 的缓解，§23）。

### 10.4 后缀

默认 `.scene.prism`（关卡/场景）、`.prefab.prism`（预制体/变体），复合后缀走 `prism_asset` §9.1 的最长匹配 + 可自定义注册；**类型靠包头而非后缀判定**，后缀仅用于默认 loader 分发。

---

## 11. 常驻档位与按需物化（residency tier）

解决「百万静态内容不能都当活 ECS 实体」的真问题——**不引入平行真相源**，而是给 scene 内容分「常驻档位」，交互内容才进 ECS（借 UE World Partition + HLOD + 按需物化）。

### 11.1 三档常驻

同一棵 scene 可混合三档，烘焙期标注或作者意图：

| 档位 | 语义 | 是否进 ECS | 典型内容 |
|---|---|---|---|
| **Tier A interactive** | 整实例化为活实体 | 是 | 可交互对象、任务点、角色、动态物件 |
| **Tier B streamed** | 子场景块，进范围实例化、出范围卸载 | 进范围时是 | 室内、建筑内部、关卡区块 |
| **Tier C proxy** | 扁平成 SoA proxy 流，贡献 GPU 常驻实例 + 空间记录，**不进 ECS** | 否（除非物化） | 海量静态散布（树/石/草/路灯） |

档位标注写入 `ResidencyPlan`，由世界系统基座的流送 VM 消费（§12）。

### 11.2 按需物化（proxy ⇄ ECS）

Tier C 的 proxy 对象被靠近/交互时，**物化**为一个 Tier A scene 实例：

```
materialize(proxy_obj):
  1. 查 proxy 对应的 SceneTemplate + 变换 + 持久 override(若曾被改)
  2. 走 §8 快路径实例化为活实体子树
  3. 其 PersistentId 定位该 proxy,施加存档里的 override delta(§15)
dematerialize(instance):   // 退离时
  1. 序列化实例的 override delta 回 proxy 持久层
  2. despawn 整树,退回 GPU 常驻 proxy 表示
```

**物化/反物化全是 scene 的 spawn/despawn + override 读写**，无需第二套序列化。迟滞（hysteresis）+ 预算防临界距离抖动（§18/§23）。

### 11.3 local-to-scene（per-instance 唯一化，借 Godot）

某些资产需按实例唯一化（如每实例一份可变材质、独立随机种子）。模板里标 `#[local_to_scene]` 的字段/子资产，在实例化时复制一份实例专属副本，而非共享模板引用。默认共享（省内存），显式标注才唯一化。

### 11.4 HLOD 簇与 impostor 代理（渲染侧接缝）

大世界远景不能逐实例绘制。本 crate 提供「簇声明 + 切换契约」，代理资产由渲染侧烘焙，职责单向（借 UE HLOD / Horizon-Decima 的 tile 代理形态）。

- **簇声明**：烘焙期把一片 Tier C proxy 按空间网格聚成簇（cluster），每簇声明 `{ 代理资产 SoftHandle, 切换阈值, 成员 PersistentId 列 }`；代理资产（合并网格 / impostor 公告板 / 体素化壳）由渲染侧离线产出，本 crate 只持引用与阈值。
- **分层代理**：簇可多级（近→中→远）对应 HLOD 层级，每级一个代理资产，按屏幕空间误差（screen-space error）阈值切换。
- **切换事件**：越过阈值发 `HLODSwitch{ cluster, from_level, to_level }`，渲染侧换绘制表示；本 crate 不绘制、不管 GPU 资源，只发信号与维护成员归属。
- **物化穿透**：玩家进簇交互半径时，被触碰的 proxy 单体按 §11.2 物化为 Tier A 实例，其余仍走代理；物化单体从代理绘制集排除（渲染侧按 `PersistentId` 剔除），避免双重绘制。
- **确定性成簇**：聚簇用固定空间网格 + 稳定排序，跨机一致、簇 id 可复现，供网络/存档按簇寻址（§16.4）。

---

## 12. 子场景流送与世界系统接缝（additive streaming）

### 12.1 additive 加载/卸载

- 子场景作为 `SoftHandle<SceneTemplate>` 是**流送单元**：进范围 → `load_soft` → 依赖闭包就绪 → §8 实例化 → 发 `SceneInstantiated`；出范围 → 按根 `despawn_scene` 整树一次回收（复用 `prism_asset` 释放队列 + `prism_ecs` 层级 despawn）。
- **加法加载**：多个子场景叠加进同一 World，各自一个 `SceneInstance` 根，互不干扰；卸载按根批量回收。

### 12.2 与世界系统基座的契约（本 crate 向上提供）

本 crate **不自建**空间索引/流送调度，由重做后的世界系统基座（`prism_world_stream`/`residency`/`persist`，另文）驱动。契约：

| 世界系统基座调用 | 本 crate 提供 |
|---|---|
| 进入 cell / Tier B 进范围 | `spawn_scene(handle, transform)` → `SceneInstanceId` |
| 离开 cell / 出范围 | `despawn_scene(id)` |
| Tier C proxy 升级 | `materialize(proxy) → SceneInstanceId`（§11.2） |
| proxy 降级 | `dematerialize(id)` → 回收 override delta |
| 存档/多人 | 读写 `PersistentId → OverrideDelta`（§15） |
| 热重载传播 | `SceneReloaded{ instance }` 事件（§13） |

**边界**：本 crate 是「作者内容单元」的实例化/卸载/覆盖引擎；「何时、在哪、按什么预算触发」归世界系统基座。scene 是逻辑真相，基座是其上的流送/常驻/持久化层。

### 12.3 预测式预取、优先级与无缝切换

- **预测式预取**：世界基座给出速度/朝向，本 crate 对「即将进范围」的子场景提前 `load_soft` 走依赖闭包就绪，但延后实例化到真正进范围；预取与实例化解耦，掩盖 IO 延迟（借 UE World Partition 预加载边带）。
- **优先级队列**：实例化/物化任务带优先级（交互 > 视野中心 > 边缘 > 预取），每帧按预算从高优到低优消费（§18.5）；低优任务可被更高优抢占顺延。
- **分帧切片（time-sliced）**：大子场景实例化按 archetype 组切片跨帧完成，每帧只填预算内列块，`SceneInstantiated` 待整树就绪后一次性发（不暴露半成品）。
- **无缝卸载**：出范围子场景先降可见/停更一帧、再整树回收，避免「正被引用时拔除」导致上层句柄瞬断；回收走延迟队列，与加载错峰。
- **接缝一致性**：跨子场景引用（§6.3）在接缝处按 `PersistentId` 延迟解析，两侧加载时序无关，先到先挂、后到补联。

---

## 13. 热重载与 override 保全传播（`hot_reload` 档位）

编辑器 live 迭代的核心：改 prefab 基体，所有实例实时更新，**各实例手改过的字段纹丝不动**（Unity Prefab modifications 体验）。

### 13.1 传播流程

```
prism_asset 对 SceneTemplate 发 Modified:
  1. diff 新旧模板计划(结构 diff + 属性 diff)
  2. for 每个引用该模板的活 SceneInstance:
       a. 保留该实例的 OverrideSet
       b. 用新模板重建基值(对未 override 字段);被标记 override 的字段跳过(§7.1)
       c. 结构变化(模板新增/删除实体):增量施加到实例,尽量原地改而非整树重建
       d. 嵌套/变体:沿继承链自底向上重放(§7.3)
  3. 发 SceneReloaded{ instance } → 渲染/物理刷新驻留
```

### 13.2 保全策略

- **属性保全**：被 override 的 `(target, component, path)` 在重建时跳过，其余照新模板更新。
- **结构保全**：实例 `AddEntity` 的新增实体用保留 LocalEntityId 段（§7.1），基体重排不影响；基体删了某实体但实例对它有 override → 孤儿处理（§23）。
- **句柄保全**：尽量原地改组件数据而非 despawn/respawn，避免实例内 `Entity`/`Handle` 失效（减少上层句柄悬空）。

### 13.3 变体与嵌套的级联热重载

改基体 A → A 的变体 B、B 的实例、A 的嵌套引用方全部级联重建（按依赖图拓扑序，`prism_asset` 失效传播驱动）。单测对拍：改一个字段后，所有实例该字段更新、其余 override 字段不变。

---

## 14. 确定性与回滚（`determinism` 档位）

- **确定性实例化**：固定模板 GUID + schema 版本 + 实例化参数（变换/种子）⇒ `LocalEntityId` 顺序、Entity 批量预留顺序、`PersistentId`、override 重放顺序全可复现，跨平台一致（对接 `prism_ecs` 定点路径）。
- **override 重放可复现**：override 集按稳定键（`(target, component, path)`）排序重放，叠栈顺序固定（§7.3），无 HashMap 迭代序依赖。
- **回滚友好**：实例化/物化是「批量预留 + 列写入」的纯结构操作，可纳入 `prism_ecs` 的 snapshot/rollback 档位；反物化/卸载对称可回滚。
- **种子**：`#[local_to_scene]` 的随机种子由 `hash(PersistentId ‖ 全局 seed)` 派生，确定且实例间去相关。


---

## 15. 持久化与存档（override delta / PersistentId / 与 world_persist 分工）

存档的本质，是把「玩家/系统在运行时对作者内容做的差异」持久下来，下次会话按同一模板叠回去。`prism_scene` 复用覆盖系统这套既有机制：**运行时变更 = 一份新的 override delta**，与作者 override 同形同构，天然可存、可叠、可确定性重放。

### 15.1 存档即 override delta

- 运行时某实例被玩法改了（开过的箱子、击碎的墙、拾走的道具），这些变更以 `(PersistentId, 目标, component, path) → 新值` 记成一份 **运行时 override delta**，键结构与 §7 作者 override 完全一致。
- 读档：实例化模板 → 叠作者 override → 再叠运行时 delta。三层顺序固定（模板基值 → 作者 override → 运行时 delta），确定性可复现（§14）。
- 只存差异：没被玩法碰过的实例，存档里零字节。成本正比于「玩家真正改变过的世界」，不是世界总量。

### 15.2 PersistentId 作为跨会话锚点

- `PersistentId = hash(场景实例链路 ‖ LocalEntityId ‖ 可选作者稳定键)`（§6），跨会话、跨进程稳定，是存档/网络定位实体的唯一键。
- 关卡重排、模板内实体增删，只要作者稳定键不变，`PersistentId` 不漂移；无稳定键的程序化实体由链路派生，重建顺序确定则 id 稳定。
- 孤儿 delta（指向已被模板删除的实体）：读档时软降级，发诊断并丢弃该条，不阻断整档加载（§23）。

### 15.3 与 world_persist 的分工

| 关注点 | 本 crate（prism_scene） | world_persist（另文） |
|---|---|---|
| 产生什么 | 单实例的 override delta（键 = `PersistentId`） | 哪些 cell/区块脏了、何时落盘、分片与压缩 |
| 存什么形态 | 稀疏键值 delta（同作者 override 结构） | delta 的空间分桶 + 版本 + 校验 |
| 何时存 | 不决策（被动响应变更事件） | 决策（预算/节流/退出/检查点触发） |
| 读档物化 | 叠 delta 到实例（§15.1） | 调度哪些块先物化、按需拉取 |

**边界**：本 crate 定义「一份实例 delta 长什么样、怎么确定性叠回」；「海量 delta 怎么分桶落盘、何时落、怎么压缩校验」归 `world_persist`。scene 给语义，持久化基座给工程。

### 15.4 版本迁移下的存档

- 模板 schema 升版（字段改名/类型变化）：存档里的旧字段 id 经 §21 迁移映射转译后再叠加；无法迁移的字段软降级丢弃并诊断。
- `LocalEntityId` 重排迁移映射（§21）保证旧 delta 的目标仍能定位到正确实体。

---

## 16. 事件、变更与世界集成

本 crate 通过 `prism_ecs` 事件总线对外广播生命周期与变更信号，供渲染/物理/脚本/世界基座/编辑器订阅；本 crate 自身不轮询、不反向依赖订阅方。

### 16.1 生命周期事件

| 事件 | 触发时机 | 典型订阅方 |
|---|---|---|
| `SceneInstantiated{ instance, root, template }` | 实例化完成、实体可见 | 渲染注册、物理建代理、脚本 on_spawn |
| `SceneDespawned{ instance, root }` | 整树回收前 | 释放 GPU/物理资源、脚本 on_despawn |
| `SceneReloaded{ instance, changed }` | 热重载重建后（§13） | 渲染/物理刷新受影响驻留 |
| `SceneMaterialized{ proxy, instance }` | proxy 升级进 ECS（§11.2） | 交互系统接管 |
| `SceneDematerialized{ instance, proxy }` | 降级回 proxy | 回收交互态、沉淀 delta |
| `OverrideApplied{ instance, target, path }` | 运行时 override 落地（§15.1） | 存档记脏、网络广播 |

### 16.2 变更粒度与批处理

- 事件按「本帧批」聚合发送（一次流送加载的几千实体 → 一个 `SceneInstantiated` 批 + 实体范围），订阅方批量处理，避免逐实体事件风暴。
- `changed` 载荷携带受影响的 `(target, component)` 集合，订阅方可精确刷新而非全量重扫。

### 16.3 世界集成的单向性

世界基座调用本 crate 的命令式 API（spawn/despawn/materialize，§12.2），本 crate 回以事件。控制流单向：基座决策 → 调用 → 本 crate 执行 → 事件回传。本 crate 不知道「cell/预算/相机」等世界概念，保持内容层纯净。

### 16.4 网络复制与权威（`persist` + 网络层接缝）

多人场景的复制复用覆盖系统，不另立协议真相（借 UE OFPA + replication 形态，纯数据）。

- **复制单元 = override delta**：实体网络状态变化表达为按 `PersistentId` 寻址的 override delta（§15.1），与存档、作者 override 同形；网络层只传稀疏键值 delta，不传全量实体。
- **权威与冲突**：delta 带来源与序号，权威端按稳定键 `(target, component, path)` 合并，高序号覆盖；本 crate 提供确定性合并序（§14），冲突策略由网络层注入（last-write / 权威仲裁）。
- **兴趣管理**：客户端按空间簇（§11.4）或实例订阅相关 delta 流——远处收代理级粗粒度状态，交互区收实例级细粒度。
- **延迟物化一致**：客户端对未物化 proxy 累积 delta，物化（§11.2）时一次性叠加，保证迟到玩家看到的世界状态与权威端逐位一致。
- **边界**：本 crate 定义「状态怎么表达、怎么确定性叠加」；「怎么传、何时传、可靠/非可靠通道、带宽预算」归网络层，另文。

---

## 17. 可观测性与诊断

对接 `prism_diagnostic`，所有热路径与异步路径带计数器与 trace span；失败软降级并留痕，不静默吞错。

### 17.1 计数器

| 计数器 | 含义 |
|---|---|
| `scene.instances.live` | 当前活实例数（分档位：interactive/streamed/proxy） |
| `scene.entities.resident` | 进 ECS 的实体数 vs proxy 态实体数 |
| `scene.instantiate.ns` | 单次实例化耗时分布（p50/p99） |
| `scene.override.count` | 活 override 条目总数（作者 + 运行时） |
| `scene.cow.sharing_ratio` | COW 列块平均共享度（N 实例 / 实际物理列块） |
| `scene.hot_reload.propagations` | 热重载传播次数 / 受影响实例数 |
| `scene.materialize.ns` | proxy ⇄ ECS 物化/反物化耗时 |
| `scene.orphan_overrides` | 孤儿 override 条目数（告警项，§23） |

### 17.2 trace span

- `scene.load`（含依赖闭包就绪子 span）、`scene.instantiate`（含 remap/COW/并行分片子 span）、`scene.hot_reload`、`scene.materialize`、`scene.bake`（离线）。
- span 带模板 GUID + 实例数标签，可在诊断面板按模板聚合定位热点。

### 17.3 不变量断言（debug 档位）

- 实例化后：每个实例内 `LocalEntityId → Entity` 映射双射、层级无环、override 目标全部可解析。
- 热重载后：被 override 字段值未被基体更新覆盖；未 override 字段 == 新基值。
- 这些断言仅在 `debug_assertions` 或显式诊断档位开启，release 热路径零开销。

---

## 18. 性能工程

性能是「数据布局 + 批处理 + 零反射热路径」的结构性结果，不是事后优化。所有数字为 design target（沙盒无 GPU，未实测），编码期以 §25 基准验收。

### 18.1 实例化快路径

- **列块 memcpy**：模板按 archetype 预排成列（SoA），实例化时整列 `copy_from_slice` 进 `prism_ecs` 预留 chunk，而非逐实体逐组件 insert。
- **一次性 remap**：`LocalEntityId → Entity` 批量预留后，所有内部引用（父子、实体引用字段）一次性加偏移重写，不逐个查表。
- **预合成 override**：烘焙期把纯属性 override 合进模板列的「变体视图」，运行期直接 memcpy 合成后的列，不在热路径施加 override（§7.4）。

### 18.2 COW 与内存

- N 个相同 prefab 共享同一份只读模板列（mmap），仅各自的 override delta 与被写字段独立；内存亚线性于实例数。
- 目标：100× 相同 prefab 的常驻内存 < 单份的 ×1.3（§25）。
- proxy 档位（§11）用 SoA 紧凑数组，单 proxy 常驻字节数远小于一个 ECS 实体（无 archetype 元数据/句柄槽）。

### 18.3 并行

- 大批实例化按 archetype 列块分片，`prism_tasks` 并行填充不同 chunk；remap 阶段按实例独立并行；合并阶段无锁（各写各的预留区间）。
- 目标：100× 相同 prefab 并行实例化近线性加速（§25）。

### 18.4 冷加载

- mmap 模板包，零解析直接把列映射进地址空间（§10）；依赖闭包并行预取（`prism_asset` EDL）。
- 首次 touch 触发按页加载，交互内容优先、proxy 数据惰性。

### 18.5 预算、迟滞与回收

- 每帧实例化/物化有时间预算，超预算的任务顺延下帧（世界基座调度，§12.2）。
- 物化/反物化带迟滞阈值，避免 proxy 在升降级边界抖动（§11.2）。
- despawn 整树批量回收，归还 chunk 空槽给 `prism_ecs` 复用；COW 列块引用计数归零才真正释放。

---

## 19. 易用性与 API 人体工学（具名实体 / override DSL / prelude）

易用是本设计首要权重（§1）。目标：常见操作一行、override 用名字不用编号、失败不 panic、所见即所得。以下代码示意「手感」，非最终签名。

### 19.1 具名局部实体

模板内实体可带 `NameTag`，override 与引用用名字而非裸 `LocalEntityId`：

```rust
// 实例化一个预制体,拿到实例句柄
let goblin = scene::spawn(&mut world, goblin_prefab, Transform::at(pos));

// 用名字 override 属性(易用核心):改"武器"子实体的伤害字段
goblin.set(&mut world, local!("Weapon"), |w: &mut Weapon| w.damage = 42.0);

// 名字解析失败 → 软降级发诊断,返回 Err,不 panic
if goblin.set(&mut world, local!("NotExist"), ...).is_err() {
    // 诊断已记录,调用方自行决定
}
```

### 19.2 override DSL

批量 override 用声明式 builder，烘焙期可预合成（§7.4）：

```rust
let elite = scene::variant(goblin_prefab)      // 建变体
    .set("Weapon", Weapon { damage: 80.0, .. })  // 属性 override
    .set("Body",   Health { max: 300.0, .. })
    .add_child("Aura", aura_prefab)              // 结构 override:加子实体
    .remove("Backpack")                          // 结构 override:删子实体
    .bake();                                      // 预合成为新模板视图
```

### 19.3 作者/运行时对称

同一套 `.set()` 既用于作者变体，也用于运行时存档 delta（§15）——用户只学一套语义。编辑器里拖拽改属性 = 调同一 API 产生同形 override。

### 19.4 prelude 与渐进披露

```rust
use prism_scene::prelude::*;   // spawn / despawn / variant / local! / Scene
```

- 常见 90% 场景（spawn/despawn/set 一个字段/建变体）在 prelude 一把梭。
- 高级 10%（手调 archetype 预排、自定义物化策略、确定性种子）在子模块，不污染默认命名空间。

### 19.5 不侵入用户类型

- Scene 是**数据资产**，不要求用户组件 `#[derive(Scene)]`；组件只需 `prism_reflect` 可见即可被覆盖/序列化/热改（§7、§13）。
- 新加一个组件字段，无需改 scene 代码：反射自动纳入覆盖/持久化面。

### 19.6 错误即诊断

所有可失败 API 返回 `Result` 且失败时发结构化诊断（§17），默认软降级（跳过该 override/该实体），绝不因单条内容错误 panic 整个关卡加载。

### 19.7 可发现性、撤销/重做与安全重命名

易用不止于简洁 API，更在于「改错能回退、重命名不碎引用、构建期能提示」。

- **撤销/重做即 override 栈**：编辑器每次改动产生一条带反向补丁的 override 操作入事务栈；undo 施加反向 delta、redo 重放，复用 §7 重放语义，无需单独 undo 子系统。
- **安全重命名**：具名实体 `NameTag` 改名时，编辑器按 `LocalEntityId` 回填所有引用（引用绑 id 不绑字符串，§6），改名零碎引用。
- **构建期可发现**：`local!("Weapon")` 在 `editor` 档位可由构建脚本对模板校验，拼错名字构建期报错而非运行期软降级；release 回落运行期解析。
- **override 预算提示**：实例 override 条目数、偏离基体是否过多（可能该升级为新变体）在编辑器以计数提示（§17 `scene.override.count`），引导作者保持继承而非散改。
- **渐进式复杂度**：90% 用 `spawn/set/variant`；需要时才下沉到 `InstantiationPlan`、自定义物化策略、确定性种子，默认命名空间不被高级 API 污染（§19.4）。

---

## 20. crate 分层与模块布局

### 20.1 crate 拆分

| crate | 职责 | std |
|---|---|---|
| `prism_scene` | 运行时：模型/身份/覆盖/实例化/常驻/流送接缝/热重载/确定性/持久化语义 | `no_std + alloc` 核 |
| `prism_scene_bake` | 离线：模板编译（反射 → 稳定列 schema + 预排 archetype + 预合成 override + 依赖闭包） | `std`（工具链） |

世界系统拆为另文的 `prism_world_stream` / `prism_world_residency` / `prism_world_persist`，本 crate 只定与其接缝的契约（§12.2、§15.3），不实现它们。

### 20.2 feature 矩阵

| feature | 作用 | 默认 |
|---|---|---|
| `std` | 文件 IO / 线程 / 诊断后端 | 否（核 no_std） |
| `async_io` | 异步流送加载（配合 `prism_tasks`） | 否 |
| `hot_reload` | 监听 `Modified`、保 override 传播（§13） | 否 |
| `editor` | Prefab Stage、拖拽 override、一键 apply/revert（§7.5） | 否 |
| `determinism` | 确定性实例化 + 回滚钩子（§14） | 否 |
| `persist` | 运行时 override delta 产生与读档叠加（§15） | 否 |
| `residency` | 常驻档位 + proxy ⇄ ECS 物化（§11） | 否 |
| `override_runtime` | 运行时施加 override（关掉则只能用预合成模板，极致省运行期） | 是 |

### 20.3 模块布局（`prism_scene`）

```
prism_scene/
  model/        Scene / SceneTemplate / SceneInstance / 节点树
  id/           LocalEntityId / NameTag / PersistentId / 解析器
  override/     OverrideSet / 属性 override / 结构 override / 叠栈
  instantiate/  列块 memcpy / remap / 并行分片 / COW
  residency/    tier A/B/C / proxy SoA / materialize
  stream/       additive load/unload / 世界基座接缝
  reload/       热重载 diff + 保 override 传播
  determinism/  确定性种子 / 重放序 / rollback 钩子
  persist/      override delta / 读档叠加
  serialize/    稳定列 schema 读 / mmap 零解析
  diag/         计数器 / span / 不变量断言
  prelude.rs
```

---

## 21. 契约、不变量与版本化

### 21.1 对下游 crate 的契约

- 向 `prism_ecs`：只经「批量预留 + chunk 列写入 + 层级 despawn」公共 API 写实体，不碰其内部存储；确定性路径走其定点/序分配接口。
- 向 `prism_asset`：模板是一等资产，经 `SoftHandle`/依赖闭包/`Modified`/单文件包；本 crate 不自建 IO/缓存。
- 向 `prism_transform`：层级经其关系驱动 API 建；本 crate 不自建父子存储。
- 向 `prism_reflect`：字段路径/override 值序列化/编辑器属性枚举全经反射；运行时热路径不触反射（仅烘焙/编辑器期）。

### 21.2 核心不变量

- **双射**：实例内 `LocalEntityId ↔ Entity` 一一对应；`PersistentId` 全局唯一且跨会话稳定。
- **叠栈确定**：override 按稳定键排序重放，叠栈顺序固定（§7.3、§14）。
- **稀疏性**：实例只存 delta；零 override 实例零额外内存（§18.2）。
- **软降级**：任何单条内容错误（孤儿 override、解析失败、迁移失败）发诊断并跳过，不阻断整体（§19.6、§23）。
- **单向集成**：世界基座调命令、本 crate 回事件，内容层不反依赖世界概念（§16.3）。

### 21.3 版本化与迁移

- **schema 版本**：每模板带 schema version；字段用稳定 id（非名字）编码，改名不破二进制。
- **字段迁移映射**：升版提供「旧字段 id → 新字段 id / 转换函数」表，读旧包与旧存档时转译；无映射字段软降级丢弃并诊断。
- **LocalEntityId 重排迁移**：模板内实体增删导致 id 重排时，烘焙产出「旧 id → 新 id」映射，使旧 override/旧存档 delta 仍能定位（§13.2、§15.4）。
- **包格式版本**：序列化容器带 magic + version，拒绝不兼容大版本并给出清晰诊断，不静默误读。

---

## 22. 路线图（M0–M6）与基准即规格

每里程碑以可运行基准为验收门槛（§25），不达标不进下一阶段。

| 里程碑 | 内容 | 验收门槛 |
|---|---|---|
| M0 | 模型骨架：Scene/Template/Instance/LocalEntityId + 朴素实例化（逐实体） | 能 spawn/despawn 一棵静态子树，层级正确 |
| M1 | 数据导向快路径：列块 memcpy + 一次性 remap + archetype 预排 | 纯属性 prefab 实例化 < 1µs/实体（§25） |
| M2 | 覆盖系统：属性 override + 结构 override + 叠栈 + 具名引用 | override 一字段、重放确定、API 一行（§19） |
| M3 | 嵌套/变体 + 热重载保 override（`hot_reload`/`editor`） | 改基体一字段，所有实例更新、override 字段不变 |
| M4 | 序列化：稳定列 schema + mmap 零解析 + 单文件包 + 依赖闭包 | 冷加载零解析，依赖闭包并行预取 |
| M5 | 常驻档位 + proxy ⇄ ECS 物化 + additive 流送接缝（`residency`） | 百万 proxy 常驻，交互区按需物化不抖动 |
| M6 | 持久化 delta + 确定性/回滚 + 迁移（`persist`/`determinism`） | 存档只存 delta、读档确定重放、跨平台一致 |

---

## 23. 诚实边界与风险

直陈局限，不粉饰；每条给缓解。

| 边界/风险 | 说明 | 缓解 |
|---|---|---|
| 二进制包不可逐行 diff | 稳定列二进制对 VCS 不友好，难人工 review | 提供文本化 dump 工具 + 结构化 diff 工具；协作粒度靠单文件包 + override 稀疏（改一处只动一处） |
| 整模板合并局限 | 两人同时改同一模板的不同实体仍可能冲突 | 鼓励嵌套拆分（小 prefab 组合），冲突面缩到单实体；提供三方合并辅助（基于稳定字段 id） |
| 孤儿 override | 基体删了某实体，实例/存档仍对它有 override | 读取时软降级丢弃 + 诊断告警（§17 `orphan_overrides`）；编辑器提示作者清理 |
| 物化抖动 | proxy 在升降级边界反复物化 | 迟滞阈值 + 预算节流（§18.5）；世界基座调度兜底 |
| 作者手搓百万不现实 | 大世界靠人手摆百万实体不可行 | 程序化散布节点（scatter node）在模板内声明「规则 + 种子」，烘焙期展开为 proxy 列，确定且稀疏存储（非 AI，纯规则） |
| 预合成与运行时 override 的张力 | 预合成快但失去运行期灵活；全运行时灵活但慢 | 二者共存：纯属性走预合成、结构/动态走运行时（§7.4、§20.2 `override_runtime`） |
| 无 GPU 实测 | 本设计数字均为 target | §25 基准即规格，编码期以真机校准 |

---

## 24. AAA 高级功能增补

在三层核心模型（§5–§23）之上，增补大世界 / 协作 / 品质所需的进阶能力。全部为经典数据结构 + 确定性算法，**无 AI/ML**、**不依赖 Bevy**、不复制任何引擎源码；均走 feature/档位门控（多数在 `editor`/`determinism`/`persist` 下），默认不付费，启用后仍服从「scene 是单一真相、override delta 是唯一持久可写、其余皆可弃派生」三原则。

### 24.1 预制体接口与暴露参数（prefab interface / exposed params）

prefab 不应是黑盒：模板可声明一组**暴露参数**（exposed parameters）作为「实例化契约」，调用方只填这组参数，内部结构被封装（借 Houdini HDA 暴露参数 / Godot `@export` / UE spawn-time 暴露形态，纯数据非脚本）。

```text
PrefabInterface {
  params: [ { name, 类型, 默认值, 取值域/约束, 作用的内部字段路径集 } ],
  slots:  [ { name, 期望 prefab 契约 } ],   // 可插拔子 prefab 挂点
  events: [ 具名生命周期钩子(数据声明,非回调代码) ],
}
```

- **一处改、多处生效**：一个暴露参数可驱动多个内部字段（如「色调」同时改材质参数、灯光色、粒子色），映射在模板声明，实例化期一次性展开为具体 override。
- **类型化与可发现**：参数带类型/取值域/默认，编辑器自动生成检视面板（§7.4），错误即诊断（§19.6）。
- **封装边界**：未暴露的内部结构不进实例化契约，重构内部不破坏调用方——prefab 有了稳定「公开 API」。

### 24.2 构造式参数化预制体（construction prefab，确定性烘焙生成）

某些 prefab 的结构**由参数确定性生成**（如「n 层楼梯」「长度 L 的栅栏」「半径 r 的环形布阵」）。用数据驱动的构造图（construction graph）在**烘焙期/实例化期确定性展开**，无运行时脚本（借 UE construction script 的「参数→结构」形态，但以确定性数据流图落地，非命令式脚本）：

```text
参数(§24.1) → 构造图(重复/阵列/沿样条分布/条件节点,确定性) → 展开为实体子树模板
  ⇒ 同参数同结构(§14 确定性),可缓存,可增量重算(改参数只重展开受影响子树)
```

构造结果是派生的实例化计划（§8.1），非平行真相；作者对展开结果的手改仍是 override delta（§7），重展开时按 `PersistentId` 保全（§13.2）。

### 24.3 程序化散布节点与规则库（scatter node）

模板内声明「在体积 / 样条 / 地表上按密度 + 种子散布某 prefab」，烘焙期**确定性**展开为 proxy 列（§11 Tier C、§23），作者只维护规则不手摆实例（借 UE PCG 的「规则 + 点云」形态，纯算法）。

- **可组合规则库**：密度图 × 样条引导 × 排除体积 × 坡度/高度/曲率约束 × 其他散布层互斥，多规则叠加，烘焙期确定性求值（接 world §14/§15）。
- **散布即 scene 节点**：scatter 节点是 scene 模板的一等节点，被 world 流送基座消费为 Tier C proxy，不进 ECS 除非物化（§11.2）。
- **确定性**：`seed + 空间锚点 + 规则版本`（§14）⇒ 可复现、可增量（改一条规则只重散受影响分块）。

### 24.4 嵌套变体继承与菱形消歧（diamond resolution）

变体即薄模板（§9.2），prefab 可多层继承（基体 → 变体 → 变体的变体）。多重继承时的**菱形依赖**（A 同时经 B、C 继承自 D）用确定性线性化消歧：

- **C3 式线性化**：对继承 DAG 做确定性拓扑线性化（borrow C3/MRO 形态），得唯一覆盖顺序，override 按此序叠加，结果与声明顺序无关、跨机一致。
- **冲突可诊断**：同一字段被多条继承路径以不同值覆盖时，按线性序取胜者并发诊断（§19.6），编辑器可视化覆盖来源链（§7.4）。
- **深继承性能**：烘焙期预合成（§7.5）把继承链压平为单层实例化计划，运行时无链式查找开销（§8）。

### 24.5 预制体 LOD 链与多分辨率（prefab LOD chain）

同一 prefab 可声明多分辨率表示（全细节 / 中 / 远 / impostor），作为「内容侧 LOD」与 world 的流送/HLOD（world §10/§25.1）对接：

- 模板声明 `lod_chain: [ { 表示, 屏幕误差阈值 } ]`，实例化时按当前细节需求选表示，连续切换 + dither（禁 pop，world §1）。
- 与虚拟几何（world §25.1）互补：虚拟几何管单网格内部连续簇 LOD，prefab LOD 链管「整 prefab 换表示」（如建筑换 impostor）。
- LOD 表示是 soft 引用的资产变体，缺失软降级到最近可用级（§6）。

### 24.6 稳定子对象寻址与跨场景软引用（stable addressing）

实例内任意子实体/子资产可被**稳定路径**寻址（借 UE soft object path / Unity property path 形态）：

```text
SubPath = SceneInstanceId · [具名节点段…] · 字段路径     // 跨会话稳定
解析: 经 PersistentId(§6.2) 锚定,延迟解析,目标未加载则软降级(§6.3)
```

- **跨场景链接**：实例字段可软引用另一场景的具名子对象（任务目标、传送点、序列轨道绑定 §24.13），两侧加载时序无关，先到先挂、后到补联（§12.3 接缝一致性）。
- **稳定锚点**：寻址经 `PersistentId` 而非易变 Entity，热重载 / 重实例化后自动重连（§13、§16）。

### 24.7 引用完整性与级联策略（cascade / nullify）

跨对象引用需要「删除语义」防悬垂（borrow 数据库外键 ON DELETE 形态）：

| 策略 | 语义 | 典型 |
|---|---|---|
| `nullify` | 被引用对象删除 → 引用置空 + 诊断 | 弱引用（UI 指向、可选目标） |
| `cascade` | 被引用对象删除 → 引用方一并删除 | 强组合（挂件随宿主销毁） |
| `restrict` | 存在引用则禁止删除（或转孤儿待处理） | 关键依赖保护 |
| `orphan-keep` | 删除后引用变孤儿软引用，目标重现则重连 | 跨场景延迟链接（§24.6） |

策略在字段声明，编辑期安全重命名/删除按策略级联修复引用（§19.7），存档/多人同策略（§15、§16.4）。

### 24.8 三方合并与协作冲突消解（3-way merge / soft-lock）

OFPA 每对象文件化（§15、world §11.1）让多人并行编辑同一世界的不同对象天然无冲突；对**同一对象的并发编辑**，提供协作消解（borrow Git 三方合并 + 版本控制 checkout 形态）：

- **稀疏 override 三方合并**：override 是字段级稀疏记录（§7.2），合并以「基体 + 分支 A delta + 分支 B delta」三方按字段归并——不相交字段自动合并，同字段冲突按策略（LWW / 手动，接 world §11.3 CRDT）。
- **软锁 checkout**：可选对某对象/子树声明软锁（编辑意图广播），避免同字段冲突于发生前；软锁是协作提示非强制真相。
- **冲突可视化**：编辑器并排展示冲突字段的基值 / 双改值，一键择一或合并（§7.4）。

### 24.9 数据层 / Level Instance / Packed Level Actor 对称

- **数据层开关**：实例/节点打数据层标签（昼夜 / 剧情阶段 / 难度 / DLC），运行期按掩码启停可见与物化，不改模板（接 world §6.4 的图层代数 §25.5）。
- **打包 ⇄ 解包一键互转**：一组实体可「打包为可嵌套子场景」（Packed Level Actor），或「解包为可独立编辑的实例」（unpacked），协作粒度随意切换，borrow UE Level Instance / Godot 场景即预制体形态。

### 24.10 Prefab Stage 隔离编辑与一键 apply/revert

- **Prefab Stage**：在独立上下文（隔离世界）编辑 prefab 基体，所见即所得，保存后经热重载传播到所有实例（§13、`editor`）。
- **对称 apply/revert**：实例改动一键 `apply` 回基体（提升为模板级，影响所有实例），或一键 `revert` 还原到基体值（丢弃该实例 override），完全对称可逆（§7.5），是易用性天花板的一部分。

### 24.11 HLOD 代理分层（渲染侧接缝，驱动 world §10/§25.1）

远距离把一簇子场景声明为单个代理（合并网格 / impostor / 体素化壳），近距离换回真实例。本 crate 只提供「簇 → 代理资产 SoftHandle + 切换阈值 + 成员 PersistentId 列」的声明与 `HLODSwitch` 事件，代理资产由渲染侧离线烘焙，职责单向（§11.4）。物化穿透：进簇交互半径的单体按 §11.2 物化，其余走代理（world §10.3）。

### 24.12 分帧流式实例化与内容寻址去重

- **time-sliced spawn**：超大子场景实例化按 archetype 组切片跨帧完成，每帧只填预算内列块，整树就绪才一次性发 `SceneInstantiated`（不暴露半成品），消除加载卡顿（§12.3）。
- **内容寻址列去重 + 冷热分段**：全库相同列（相同网格引用 / 相同默认组件列）单份存储，热 / 冷列分段，降磁盘与常驻内存（§8.5）；典型库 > 40% 列可去重（§25）。

### 24.13 网络复制对称与序列（sequencer）绑定

- **复制即 override delta**：实体网络状态就是 override delta，与存档 / 作者 override 同形，一套语义覆盖单机 / 存档 / 多人（§16.4、world §12）。按簇 / 实例兴趣订阅，带宽正比于交互变化量。
- **序列绑定稳定**：过场 / 动画按 `PersistentId` 绑定 scene 实体轨道，实体热重载 / 重实例化后绑定经 id 自动重连（不经 Entity）；绑定由动画侧消费，本 crate 只保证 id 稳定（§6.2）。

### 24.14 场景查询与标签代数（scene query / tag algebra）

作者与 gameplay 需按语义检索场景内容，而非记硬编码路径：

```text
查询 = 标签集 ∩/∪/∖ + 空间谓词(半径/体积内) + 类型/组件谓词
  例: tag(敌人) ∩ tag(精英) ∖ tag(已死亡) ∩ within(警戒区体积)
```

- **标签即 scene 节点 tag**（非平行真相），查询在物化实体上走 `prism_ecs` 查询、在 proxy 上走空间索引（world §5）+ 标签位集，统一接口。
- **确定性结果序**：查询结果按稳定排序（空间 Morton + id tiebreak，§14），跨机一致，供存档 / 多人 / 回放复现（§15、§16.4）。
- gameplay 的 EQS 式环境查询、任务目标检索、编辑器批量选择全复用此代数。

---

## 25. 基准即规格：量化目标与验收门槛

沙盒无 GPU，以下为 design target，编码期以真机基准校准并作为里程碑门槛（§22）。CPU 侧可直接测，GPU 相关标注待测。

| 指标 | 目标 | 条件 |
|---|---|---|
| 纯属性 prefab（已预合成）实例化 | < 1µs/实体 | 列块 memcpy + 一次性 remap，热路径零反射 |
| 带结构 override 实例化 | < 3µs/实体 | 运行时施加结构 override |
| 100× 相同 prefab 并行实例化 | 近线性加速 | `prism_tasks` 分片，合并无锁 |
| 100× 相同 prefab 常驻内存 | < 单份 ×1.3 | COW 列共享，仅 override delta 独立 |
| 冷加载（mmap 零解析） | 零解析、仅页触发开销 | 依赖闭包并行预取 |
| proxy 单位常驻字节 | ≪ 一个 ECS 实体 | SoA 紧凑、无 archetype 元数据 |
| proxy → ECS 物化 | < 预算内一帧完成一批 | 迟滞防抖，不卡帧 |
| 热重载传播（1 模板 → N 实例） | O(受影响实例)，不整树重建 | 原地增量施加 |
| 存档 delta 体积 | 正比于玩家实际改动，非世界总量 | 稀疏 override delta |
| override 施加/还原 | O(delta 条目)，确定性序 | 稳定键排序重放 |
| 内容寻址列去重率 | 典型库 > 40% 列可去重 | 相同网格/默认组件列全库单份（§8.5） |
| 预测式预取命中 | 进范围前依赖闭包已就绪 | 速度/朝向驱动预取（§12.3） |
| HLOD 切换 | 无可见突变（阈值滞回） | 屏幕误差阈值 + 滞回（§11.4） |
| 网络 delta 带宽 | 正比于交互实体变化量 | 按簇/实例兴趣订阅（§16.4） |
| 分帧实例化卡顿 | 单帧实例化不超预算 | time-sliced 跨帧（§12.3） |

---

## 26. 测试与验证策略

- **对拍（确定性）**：固定模板 + 参数两次实例化，`LocalEntityId`/Entity 序/`PersistentId`/override 重放逐位一致；跨平台（x86/ARM）对拍（§14）。
- **override 保全**：改基体一字段 → 断言所有实例该字段更新、被 override 字段不变、未 override 字段 == 新基值（§13 单测）。
- **结构 diff**：模板增删实体后热重载 → 断言实例原地增量、句柄尽量不失效、孤儿 override 走软降级路径（§13.2、§23）。
- **COW 内存**：实例化 N 份相同 prefab，断言物理列块共享度与常驻内存满足 §25 目标。
- **性能基准**（`criterion` 类）：实例化/物化/热重载各一组基准，纳入 CI 回归门禁，退化即失败（§22、§25）。
- **序列化往返**：模板 bake → mmap 读 → 实例化，与朴素路径结果逐位对拍；跨 schema 版本迁移往返（§21.3）。
- **存档往返**：产生 delta → 存 → 读 → 叠回，断言世界状态逐位复现；孤儿/迁移软降级覆盖（§15.4）。
- **模糊/软降级**：喂损坏包/孤儿 override/缺依赖，断言只发诊断不 panic、不脏其余内容（§19.6）。
- **规模冒烟**：百万 proxy 常驻 + 交互区物化，断言内存/帧预算达标、物化不抖动（§25、§18.5）。

---

> 本文档为 `prism_scene` 的 clean-slate 设计稿（v0.2，纯文档，未进入编码）。它确立 scene 为 Prism 作者态内容的单一逻辑真相，世界系统降级为架其上的流送/常驻/持久化基座；旧 `prism_world_system_design_zh.md`（仍依赖 Bevy、以 WorldDB 为平行真相）将按本文契约在后续任务中重写为去 Bevy 多 crate 版本。所有性能数字为 design target，须以 §25 基准在真机校准。
