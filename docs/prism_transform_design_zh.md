# Prism Transform 顶级次世代 AAA 级变换 / 层级 / 传播设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **局部变换（Local TRS）+ 全局变换（GlobalTransform/Affine3）+ 层级传播 + 脏标记增量 + 并行分块 + 大世界原点重定位 + 插值 + GPU 常驻** 变换内核设计。它是 `bevy_transform` 的自研替代，是渲染、物理、动画、相机、空间查询共同吃的「空间真相」。
> 借形态不抄码。借鉴：
> - **Local/Global 分离 + 传播**：Bevy `Transform` / `GlobalTransform`、`propagate_transforms`
> - **场景组件层级**：Unreal `USceneComponent` 层级、`FTransform`（分离 Location/Rotation/Scale3D）
> - **Transform 组件**：Unity `Transform`（localPosition/localRotation/localScale + 世界缓存）
> - **大世界精度**：Star Citizen 64-bit 浮点 + 容器/网格原点重定位（Origin Rebasing）、Unreal 5 `FLargeWorldCoordinates`（double）
> - **仿射运算**：glam `Affine3A`（SIMD）、经典 3×4 仿射矩阵
> 本文为纯经典线性代数 / 层级传播路线，**不含任何 AI/ML 内容**。

- 版本: v0.4（核心 M0–M6 已落地并验证；§24 高级增补中 **§24.1 静态变换烘焙与批合并（`bake`）**、**§24.3 双缓冲/多缓冲变换（`double_buffer`）**、**§24.2 姿态量化压缩（`quantize`）**、**§24.4 变换 Observer 钩子（`observer`）**、**§24.5 空间加速结构增量同步（`spatial_sync`）**、**§24.6 轻量约束（`constraint`）**、**§24.7 GPU 侧层级传播（`compute_hierarchy`，可移植调度 + 上传载荷 + CPU 参考层）**、**§24.8 扫掠变换（`sweep`）** 已交付并验证，其余 §24 项（及 §24.7 真实 GPU dispatch）仍为设计阶段；v0.1→v0.2 新增第 24 章「AAA 高级功能增补」：静态变换烘焙与批合并/姿态量化压缩/双缓冲读取一致性/变换 Observer 钩子/空间加速结构增量同步/轻量约束(look-at/aim/parent-blend)/GPU 侧层级传播/扫掠变换(CCD/运动模糊)；§24.1/24.2/24.3/24.4/24.5/24.6/24.7(可移植层)/24.8 已交付，其余（24.7 真实 GPU dispatch）仍为 PLANNED）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: `prism_math`（Vec3/Quat/Affine3/Mat4、SIMD）、`prism_ecs`（组件存储 + `ChildOf` 关系 + 变更检测 + 并行查询）、`prism_tasks`（分块并行传播）；可选 `prism_time`（插值 alpha）、`prism_diagnostic`
- 层级定位: ECS 文档 L3「仿真」；渲染提取（ECS §15）与物理/相机的空间输入供给方
- 明确约束: 核心 `no_std + alloc`；`f64`（大世界）/ `determinism`（定点）/ `2d` / `trace` 为 feature；**不依赖任何 `bevy_*` crate**

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier）
4. 分层架构
5. 核心模型：Transform / GlobalTransform / Affine3
6. 层级与关系（基于 ECS `ChildOf`，传播解耦存储）
7. 脏标记增量传播（成本正比于变化量）
8. 并行分块传播（经 prism_tasks）
9. 大世界：64-bit 坐标 + 网格原点重定位
10. 插值（配合 time alpha / 固定步表现层平滑）
11. 2D 变换变体
12. 确定性定点路径
13. GPU 常驻变换列直传（接 ECS 提取）
14. Helper API（look_at / transform_point / 空间互转）
15. 与 ECS / App / Time / 渲染 / 物理集成
16. 可观测性（传播统计 / 脏集规模）
17. 高级功能增补
18. 性能工程
19. 易用性与 Bevy 迁移策略
20. crate 分层与模块布局
21. 契约、不变量与版本化
22. 路线图（M0–M6）与基准即规格
23. 诚实边界与风险
24. AAA 高级功能增补（v0.2）

---

## 1. 设计哲学与目标

变换是引擎里**被读取最频繁、被修改最密集、却最容易悄悄算错**的数据。几乎每个子系统（渲染、物理、相机、动画、音频空间化、空间查询）都要「某物现在在世界的哪里、朝哪、多大」。`prism_transform` 的使命是：提供**一份权威的空间真相**，让「局部意图（我想相对父节点移动）」与「全局结果（我在世界的位置）」显式分离，并以**最小代价**把前者传播成后者。

**一句话定位**：`prism_transform` 是 Prism 的「空间真相内核」——Local(TRS) 作为可写意图、Global(Affine3) 作为只读结果；层级传播基于 ECS 关系而非内嵌指针；脏标记让传播成本正比于「这帧真正动了多少」；大世界用 64-bit + 原点重定位守住浮点精度。

四条总目标（按权重）：

1. **性能**：传播成本 ∝ 脏子树规模，而非实体总数；仿射运算走 SIMD；分块并行；全局矩阵缓存可直供 GPU。
2. **效果（能力）**：无限层级；大世界无抖动（64-bit + rebasing）；固定步 + 插值表现层丝滑；确定性定点可回滚。
3. **易用**：`transform.translation.x += v` 一把梭；父子一行 `commands.entity(c).insert(ChildOf(p))`；全局只读、绝不手改。
4. **可移植 + 档位化**：核心 `no_std`；f64 / 定点 / 2D 按 feature 裁剪。

**不做什么**：不做场景序列化（归 `prism_scene`）、不做骨骼蒙皮（归 `prism_anim_runtime`，但它复用本 crate 的层级）、不做物理积分（归 `prism_physics`，本 crate 只提供空间读写接缝）。

---

## 2. 参考产品取舍

| 来源 | 吸收 | 规避 |
|---|---|---|
| Bevy `Transform`/`GlobalTransform` | Local/Global 分离、`propagate_transforms`、变更检测驱动传播 | 旧版「全量遍历所有根」在超大场景的浪费；本 crate 用脏集增量 |
| Unreal `USceneComponent` | 任意层级附着、相对/世界双向 setter、`AttachToComponent` 语义 | 组件对象内嵌父子指针（难并行、难序列化）；本 crate 用 ECS 关系 |
| Unity `Transform` | 直观 local/world API、`TransformPoint/InverseTransformPoint` | 层级与 GameObject 强绑、脏标记对用户不透明；本 crate 脏集可观测 |
| Star Citizen 64-bit + 容器 | double 世界坐标 + 容器相对 + 原点重定位抗抖 | 其客户端-服务器坐标体系过重；本 crate 只取「原点 cell + f32 相对」形态 |
| Unreal 5 LWC（double） | 大世界 double 存储、相机相对渲染 | 全量 double 的带宽/缓存代价；本 crate 默认 f32，f64 为档位 |
| glam `Affine3A` | 3×4 SIMD 仿射、`transform_point3`/`mul` 快 | 无；直接经 `prism_math` 门面提供 |

**取舍原则**：层级关系用 ECS 数据（可并行、可序列化、可查询），绝不用对象内嵌指针；默认 f32 保带宽，大世界/确定性作为可选档位；传播只碰脏子树。

---

## 3. 档位化（capability / quality tier / feature）

- **capability（能力探测）**：是否启用 f64 大世界、是否启用 SIMD 路径、是否可直供 GPU 列缓冲（由渲染后端告知对齐/布局）。
- **quality tier（质量档）**：
  - *Low*：仅 f32 + 单线程传播（移动端/弱机、小场景）。
  - *Medium*：f32 + 并行分块 + 脏集增量（主流 AAA）。
  - *High*：+ GPU 常驻变换列、插值表现层、原点重定位。
  - *Ultra*：+ f64 大世界 + 定点确定性（开放世界 / 回滚网络）。
- **feature flags**：`std`、`f64`（大世界坐标）、`determinism`（定点传播）、`2d`（Transform2d 变体）、`gpu`（变换列直传布局）、`trace`（传播统计）。

档位只改**精度与并行策略**，不改公共 API 形态——用户代码 `transform.translation` 在任何档位下都成立。

---

## 4. 分层架构

```
L4 集成接缝   render 提取(ECS §15) / physics 读写 / camera / audio 空间化
L3 传播引擎   脏集收集 → 拓扑分层 → 并行分块传播 → 全局缓存写回
L2 层级模型   基于 prism_ecs 关系 ChildOf / Children（无内嵌指针）
L1 变换代数   Transform(TRS) / GlobalTransform(Affine3) / 组合 / 求逆 / 插值
L0 数学门面   prism_math: Vec3/Quat/Mat4/Affine3（SIMD, f32/f64/fixed）
```

规则：严格向下依赖；传播引擎（L3）是唯一允许写 `GlobalTransform` 的地方；用户只写 L1 的 `Transform` 与 L2 的关系。

---

## 5. 核心模型：Transform / GlobalTransform / Affine3

```rust
/// 局部变换：相对父节点的可写意图。无父则相对世界。
/// 存 TRS 三元组（而非矩阵）：直观、可独立插值、避免矩阵分解。
#[derive(Component, Clone, Copy, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation:    Quat,
    pub scale:       Vec3,
}

/// 全局变换：传播产出的世界空间结果。只读（引擎写，用户读）。
/// 用 3×4 仿射（Affine3）存：比 4×4 省 1/4 内存与带宽，含非均匀缩放/错切。
#[derive(Component, Clone, Copy, PartialEq)]
pub struct GlobalTransform(pub Affine3);

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE,
    };
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self { /* PLANNED */ unimplemented!() }
    pub fn affine(&self) -> Affine3 { /* TRS → Affine3，SIMD */ unimplemented!() }
    pub fn mul(&self, child: &Transform) -> Transform { /* 组合（仅均匀缩放可闭合为 TRS） */ unimplemented!() }
}

impl GlobalTransform {
    pub fn translation(&self) -> Vec3 { /* PLANNED */ unimplemented!() }
    pub fn to_mat4(&self) -> Mat4 { /* 供 GPU/旧接口 */ unimplemented!() }
    pub fn mul_transform(&self, local: &Transform) -> GlobalTransform { /* 父Global ∘ 子Local */ unimplemented!() }
}
```

设计要点：
- **Transform 存 TRS 而非矩阵**：可独立插值旋转（slerp）、避免每帧矩阵分解、与动画/物理语义一致。
- **GlobalTransform 存 Affine3（3×4）**：承载层级累乘后的非均匀缩放与错切（纯 TRS 不足以表达），且省带宽、SIMD 友好。
- **组合注意**：父子均含非均匀缩放时，结果可能含错切，无法无损写回子的 TRS——故子的权威是「Local TRS」，Global 只做结果缓存。

---

## 6. 层级与关系（基于 ECS `ChildOf`，传播解耦存储）

层级**不内嵌在 Transform 里**，而是 ECS 关系（见 `prism_ecs_design_zh.md` §排他关系 / `ChildOf`）：

```rust
// 建立父子（排他关系：一个实体至多一个父）
commands.entity(child).insert(ChildOf(parent));
// 反向 Children 由 ECS 关系自动维护，供传播遍历
```

好处：
- **可并行**：关系是数据，传播可对不相交子树并行（§8）。
- **可序列化**：`prism_scene` 直接存关系，无需处理对象指针。
- **可查询**：`Query<(&Transform, &Children)>` 自然遍历。
- **删除策略**：`despawn` 父节点时按 ECS 关系删除策略级联（见 ECS §关系删除策略），避免悬挂。

无父实体（无 `ChildOf`）即「根」，其 `GlobalTransform = Transform.affine()`。

---

## 7. 脏标记增量传播（成本正比于变化量）

核心性能命题：**一帧只有少数实体动，传播就应只碰这些实体的子树**，而非每帧全量遍历。

机制：
1. **脏来源**：复用 ECS 变更检测——`Changed<Transform>` 或新增 `ChildOf` 关系。
2. **脏子树收集**：对每个脏实体，向下标记其子树需重算（父变则全子孙的 Global 失效）。用世代戳/位集避免重复入队。
3. **跳过干净子树**：若一个子树根未脏且其祖先链未脏，整棵跳过。
4. **根直算**：脏的根实体 `Global = Local.affine()`；脏子节点 `Global = parent.Global ∘ Local`。

复杂度：O(脏子树节点数)，而非 O(实体总数)。静止场景几乎零成本。

> 与渲染一致性：传播必须在渲染提取（ECS §15）**之前**完成，保证提取读到的 Global 是本帧最终值。

---

## 8. 并行分块传播（经 prism_tasks）

层级传播天然可并行：**不相交子树互不影响**。

策略（见 `prism_tasks_design_zh.md` 的 work-stealing 作业图）：
- **按根分块**：每个脏根子树是一个独立作业，投递到 `prism_tasks` worker 池并行跑。
- **分层并行（深层级）**：对单棵巨树，按 BFS 深度分层——同层节点互不依赖，可在层内并行，层间同步（屏障）。
- **粒度控制**：子树过小则合并成批，避免调度开销 > 计算；阈值随平台标定。
- **无锁写回**：每个节点只写自己的 `GlobalTransform`，不同作业写不同实体，天然无数据竞争（ECS 并行查询保证不相交访问）。

与 ECS §并行调度协同：传播系统声明 `Query<&mut GlobalTransform>` + `Query<&Transform>` + 关系读，调度器据此与其他系统并行编排。

---

## 9. 大世界：64-bit 坐标 + 网格原点重定位

f32 在距原点 >~16 km 处精度退化（抖动、Z-fighting、物理不稳）。两条可选档位：

**(A) f64 世界坐标（`f64` feature）**：`Transform`/`GlobalTransform` 的 translation 升为 `DVec3`，旋转/缩放保持 f32。渲染时转「相机相对坐标」再降 f32 送 GPU（camera-relative rendering），抵消大数相减误差。对标 UE5 LWC。

**(B) 网格原点重定位（Origin Rebasing，Star Citizen 形态）**：世界划分为网格单元（cell），实体存 `(cell: IVec3, local: Vec3<f32>)`。当相机/玩家跨越阈值，全场减去位移、原点归零（rebase），使活跃区始终贴近原点。接 ECS §13.3 的大世界分区。

```rust
#[cfg(feature = "f64")]
pub struct GlobalTransformHp { pub cell: IVec3, pub translation: DVec3, pub affine_f32: Affine3 }
```

两条路线可叠加：cell 定位 + cell 内 f32 + 渲染相机相对。默认关闭（f32 单精度），仅开放世界开启。

---

## 10. 插值（配合 time alpha / 固定步表现层平滑）

固定步长仿真（物理 60 Hz）与可变帧率渲染（144 Hz）之间，用 `prism_time`（见 `prism_time_design_zh.md` §12 的 `overstep_fraction` alpha）做**表现层插值**，消除低 tickrate 抖动：

```rust
// 存上一固定步与当前固定步的变换，渲染用 alpha 线性/球面插值
pub struct TransformInterpolation { prev: Transform, curr: Transform }
// render: lerp(prev.translation, curr.translation, alpha); slerp(prev.rotation, curr.rotation, alpha)
```

- 平移 lerp、旋转 slerp、缩放 lerp；仅对标记了插值的实体做（避免全量开销）。
- 插值产出的是**渲染用临时 Global**，不污染仿真权威。
- 瞬移（teleport）需能跳过插值（否则会「拉线」），提供 `Transform::teleport` 标记。

---

## 11. 2D 变换变体（`2d` feature）

2D 游戏用 3D 变换浪费且易误用 Z。提供轻量 `Transform2d`：

```rust
#[cfg(feature = "2d")]
pub struct Transform2d { pub translation: Vec2, pub rotation: f32 /*rad*/, pub scale: Vec2, pub z_layer: f32 }
```

- 复用同一套层级关系与传播引擎，仅代数降维（SE(2)+scale），更省、更快。
- `z_layer` 单独管绘制顺序，不参与 2D 平面变换。
- 与 3D 可共存（UI/世界混排时各走各的传播通道）。

---

## 12. 确定性定点路径（`determinism` feature）

回滚网络/录像重放要求**跨平台位级一致**。f32 的平台差异（FMA、编译器重排、超越函数）破坏确定性。

- 定点坐标（如 Q32.32）+ 整数/定点三角函数表，传播全程走定点代数（见 `prism_math` 定点路径）。
- 层级累乘顺序固定、无浮点、无并行不确定序（或并行但 commutative/确定归并）。
- 与 ECS §确定性、`prism_replication`（Quantum/GGPO 形态）契约一致：同输入 → 同 Global（位等价）。
- 代价：精度/范围受限、超越函数慢；仅联机确定性场景启用，默认关闭。

---

## 13. GPU 常驻变换列直传（接 ECS 提取）

渲染要把每个可见实体的世界矩阵送 GPU（实例化/间接绘制）。`GlobalTransform` 的 Affine3 可**直接按 GPU 布局打包**，省一次 CPU 侧转换：

- 传播写回时，可选同时写入一段 **GPU 友好的变换列缓冲**（SoA、对齐、`Mat3x4`/`Mat4` 按后端要求）。
- 配合 ECS §15 的渲染提取管线：提取阶段只搬「本帧脏」的变换列（增量上传），而非全量重传。
- 大世界档位：送 GPU 前转相机相对 f32（§9），布局不变。
- `gpu` feature 由 `prism_render_driver`（RHI）告知对齐/行主序列约定，避免布局分叉。

---

## 14. Helper API（look_at / transform_point / 空间互转）

易用性门面（对标 Unity/Unreal 直觉）：

```rust
impl Transform {
    pub fn look_at(&mut self, target: Vec3, up: Vec3) { /* 朝向目标 */ }
    pub fn looking_at(target: Vec3, up: Vec3) -> Self { /* 构造 */ }
    pub fn rotate_around(&mut self, point: Vec3, rot: Quat) { /* 绕点转 */ }
    pub fn forward(&self) -> Vec3; pub fn right(&self) -> Vec3; pub fn up(&self) -> Vec3;
}
impl GlobalTransform {
    pub fn transform_point(&self, p: Vec3) -> Vec3;        // 局部→世界
    pub fn inverse_transform_point(&self, p: Vec3) -> Vec3; // 世界→局部
    pub fn reparent_keeping_world(&self, child: Entity, new_parent: Entity); // 换父保持世界姿态
}
```

`reparent_keeping_world` 是 AAA 常用操作（拾取物体挂手上、下车保持位置）：换父时反解新 Local = new_parent.Global⁻¹ ∘ child.Global，用户无感知世界跳变。

---

## 15. 与 ECS / App / Time / 渲染 / 物理集成

- **ECS**：Transform/GlobalTransform 是组件；层级用关系（§6）；传播用变更检测（§7）+ 并行查询（§8）；GPU 列接提取（§13）。
- **App**：传播系统注册在 `PostUpdate`（见 `prism_app_design_zh.md` §Schedule），**在渲染提取前、在物理写回后**，顺序由 system set 固定。
- **Time**：插值吃固定步 alpha（§10，见 time §12）；瞬移标记跳过插值。
- **渲染**：`prism_render_scene`/相机/可见性读 GlobalTransform（剔除、视图矩阵、实例矩阵）。
- **物理**：物理积分写回姿态 → 本 crate 传播到子节点；或物理拥有权威时，Transform 从物理同步（接缝由 system set 顺序裁定，单一真相）。

关键顺序契约：`物理写回 → transform 传播 → 渲染提取`，三者在同一帧内严格串行（跨系统集可并行其余工作）。

---

## 16. 可观测性（传播统计 / 脏集规模）

`trace` feature 导出每帧：
- 脏实体数 / 脏子树节点数 / 实际重算节点数（验证增量有效性）。
- 并行分块数 / 各 worker 负载 / 最深层级（检测长链病态）。
- 插值实体数、GPU 变换列上传字节数（增量上传量）。
- 大世界：当前 cell、本帧是否 rebase、rebase 影响实体数。

供 `prism_profiler`/编辑器检视器定位「为什么这帧传播很贵」（通常是误把静态物标脏，或超深层级）。

---

## 17. 高级功能增补（AAA）

1. **换父保持世界姿态**（§14 `reparent_keeping_world`）——拾取/挂载标配。
2. **传播 LOD / 冻结**：远距或静态子树标记 `TransformFrozen`，传播直接跳过（配合世界分区休眠）。
3. **插值与瞬移分流**（§10）——避免瞬移拉线。
4. **大世界原点重定位**（§9）——开放世界无抖动。
5. **确定性定点传播**（§12）——回滚网络位级一致。
6. **GPU 常驻增量上传**（§13）——只传脏变换列。
7. **2D 降维通道**（§11）——2D 游戏省算力。
8. **非均匀缩放错切承载**（§5 Affine3）——层级含非均匀缩放仍正确。
9. **分层并行 + work-stealing**（§8）——超深/超宽层级都快。
10. **空间锚点 / 相对坐标系**：支持多原点（载具内坐标系、太空站本地系），对标 SC 容器——物体可相对动态父系存姿态。

---

## 18. 性能工程

- **增量 > 全量**：传播成本 ∝ 脏子树，静止场景近零（§7）。
- **Affine3（3×4）而非 Mat4**：省 25% 内存/带宽，SIMD 乘更快。
- **TRS 存储**：避免每帧矩阵分解；旋转直接 slerp。
- **SoA 变换列**：GPU 直传、缓存友好、增量上传（§13）。
- **并行分块 + 粒度自适应**（§8）：大场景线性扩展到多核。
- **变更检测过滤**：只对 `Changed<Transform>` 入脏集，避免扫全表。
- **冻结/LOD**：静态/远景子树零传播。
- **无堆热路径**：脏集用预分配位集/世代戳，传播循环无分配。

**反模式告警**：每帧无意义写 Transform（哪怕写回相同值）会标脏触发子树重算——文档与 lint 提示用「仅在真正变化时写」。

---

## 19. 易用性与 Bevy 迁移策略

- **API 近 Bevy**：`Transform`、`GlobalTransform`、`Transform::from_xyz`、`look_at`、`ChildOf` 命名/语义尽量对齐，降低迁移成本。
- **一行建父子**：`commands.entity(c).insert(ChildOf(p))`（Bevy 用户零学习）。
- **全局只读约定**：`GlobalTransform` 无公开可变 setter，防「手改全局被下帧传播覆盖」的经典坑；需要设世界姿态时用 `reparent_keeping_world` 或设根 `Transform`。
- **prelude**：`use prism_transform::prelude::*;` 带出常用类型 + helper。
- **迁移垫片**（可选）：提供 `bevy_transform` 兼容别名层，老代码可渐进替换。

---

## 20. crate 分层与模块布局

```
pkg/prism_transform/
  src/
    transform.rs            # Transform(TRS)、构造/组合/helper（look_at/forward…）
    global.rs               # GlobalTransform(Affine3)、transform_point/inverse
    hierarchy.rs            # 基于 prism_ecs 关系的 ChildOf/Children 接线、reparent
    propagate.rs            # 脏集收集 + 拓扑分层 + 写回（传播引擎核心）
    parallel.rs             # 经 prism_tasks 的分块/分层并行（无 std 时退单线程）
    interpolate.rs          # 固定步 alpha 插值、瞬移分流
    large_world.rs          # f64 / cell + 原点重定位（f64 feature）
    determinism.rs          # 定点传播（determinism feature）
    transform2d.rs          # 2D 变体（2d feature）
    gpu_columns.rs          # GPU 变换列布局/增量上传（gpu feature）
    diagnostics.rs          # 传播统计/脏集规模（trace feature）
    prelude.rs
  features = ["std","f64","determinism","2d","gpu","trace"]
```

依赖：`prism_math` + `prism_ecs` + `prism_tasks`；可选 `prism_time`/`prism_diagnostic`。**不碰任何 `bevy_*`。**

---

## 21. 契约、不变量与版本化

- **全局只读**：`GlobalTransform` 仅由传播引擎写；用户写只认 `Transform` + 关系。
- **传播时机**：渲染提取前、物理写回后，Global 必为本帧最终值。
- **无父即根**：无 `ChildOf` 的实体 `Global = Local.affine()`。
- **换父守恒**：`reparent_keeping_world` 后世界姿态不变（数值误差内）。
- **确定性契约**：`determinism` 档同输入位级同 Global，与 ECS §确定性、`prism_replication` 一致。
- **插值契约**：插值仅影响渲染用临时值，不改仿真权威；瞬移跳过插值。
- **版本化**：Transform/GlobalTransform 字段、Affine3 布局、关系语义、GPU 列布局、cell 划分约定均为版本化契约。

---

## 22. 路线图（M0–M6）与基准即规格

- **M0 代数**：Transform(TRS)/GlobalTransform(Affine3) + 组合/求逆/helper → 单测（往返、组合结合律）。
- **M1 层级传播**：基于 ECS 关系的单线程全量传播 + 变更检测驱动 → 正确性（多层父子世界姿态）。
- **M2 增量脏集**：脏子树收集 + 跳过干净子树 → 基准（静止场景近零成本）。
- **M3 并行**：经 prism_tasks 分块/分层并行 + 粒度自适应 → 基准（多核线性扩展）。
- **M4 插值**：固定步 alpha 插值 + 瞬移分流 → 低 tickrate 无抖动视觉验证。
- **M5 大世界 + 确定性**：f64/cell + 原点重定位；定点传播 → 远距无抖动 + 双跑位等价。
- **M6 GPU + 工具**：变换列增量上传 + 2D 变体 + 传播统计 + bevy 兼容 prelude。

**基准即规格**：静止场景传播成本、脏子树传播吞吐、并行多核扩展比、插值视觉平滑、大世界抖动阈值、确定性双跑位等价、GPU 增量上传字节。核心价值在 **M2（增量）+ M3（并行）+ M5（大世界/确定性）**。

---

## 23. 诚实边界与风险

- M0–M6 核心路线图**已全部落地并通过验证**：实现 + 单测（135 项 lib 测试全绿）+ 基准，`cargo clippy --all-targets` 零告警、`cargo test` 零失败。状态随代码演进；§24「AAA 高级功能增补」中 **§24.1 静态烘焙（`bake`）**、**§24.3 双缓冲（`double_buffer`）** 已随 M2/M3 优先落地并验证，**§24.2 姿态量化（`quantize`）**、**§24.4 变换 Observer（`observer`）**、**§24.6 轻量约束（`constraint`）** 本次作为纯/可测优先项落地并验证，其余 §24 项仍为 PLANNED，按本文优先级随消费方接线落地。
- **高风险项**：
  1. **非均匀缩放 + 层级（M1）**：父子均非均匀缩放会产生错切，无法无损写回子 TRS；须坚持「Global 仅为缓存、Local 为权威」，并在文档/API 明确告警，否则会出现「子物体被意外拉斜」类顽疾。
  2. **大世界精度（M5）**：f64 全量代价大、cell rebasing 边界处理（跨 cell 物理/碰撞/网络）复杂；需真实开放世界里程压测，错配会在边界出现瞬移或抖动。
  3. **确定性定点（M5）**：精度/范围受限、超越函数误差；跨平台位级一致需严格验证，且与并行归并顺序冲突（并行必须确定序）。
  4. **传播时序（M1/物理接缝）**：物理与 transform 谁是权威、写回顺序若不单一真相，会出现「物体回弹/双重积分」；须由 App system set 严格固定 `物理→传播→提取`。
  5. **误标脏导致全量重算（M2）**：每帧无意义写 Transform 会退化为全量传播，吃掉增量收益；需 lint + 诊断（§16）兜底。
  6. **插值拉线（M4）**：瞬移未跳过插值会出现跨屏拉线；teleport 标记必须覆盖所有瞬移路径（含物理 teleport、换关卡）。
- **与既有文档关系**：层级关系依赖 `prism_ecs_design_zh.md` 的 `ChildOf`/排他关系/删除策略/并行查询/变更检测/提取管线（§15）；并行执行依赖 `prism_tasks_design_zh.md`；插值 alpha 来自 `prism_time_design_zh.md` §12；大世界接 ECS §13.3 分区与 `prism_world_system_design_zh.md`；GPU 列布局由 `prism_render_driver`(RHI) 约定；确定性与 `prism_replication` 契约一致。传播在 App 主循环中的落点见 `prism_app_design_zh.md` §Schedule。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 24. AAA 高级功能增补（v0.2）

本章补齐顶级变换系统常被忽视、却在真实 AAA 项目里缺一不可的能力。均 feature/档位门控，默认不付成本；与 §17 的高级功能互补、不重叠（§17 偏「传播正确性与大世界」，本章偏「流水线、带宽、空间集成与 GPU 卸载」）。

### 24.1 静态变换烘焙与批合并（Static Baking）✅ 已交付（`bake`）

对标记 `TransformStatic`（关卡几何、建筑、地形装饰）的实体，在加载/构建期一次性烘焙世界矩阵并**退出传播集**：

- 静态子树的 `GlobalTransform` 预算好后冻结，运行时零传播、零脏检查。
- 可进一步把同材质静态网格**预合并到世界空间**（drop 掉各自 Transform），减少绘制调用与实例矩阵上传（对标 UE 静态网格合并 / Unity Static Batching）。
- 由 `prism_scene`/`prism_asset_bake` 在离线/加载期触发；运行时若需移动，显式「解冻」转回动态集。

收益:大世界里 90%+ 实体是静态的,把它们移出热路径是最大的一笔传播省时。

> **交付状态（`pkg/prism_transform/src/bake.rs`）：✅ 已交付并验证。**
> `StaticBaker` 以并行数组伴随 `TransformGraph`：`is_static` 标记、冻结世界变换 `baked`、逐节点 `valid` 位与一个 `generation` 计数器（供合并批次 / GPU 常驻缓存廉价探测失效）。核心 API：`mark_static` / `clear_static` / `is_static` / `is_baked` / `baked` / `try_baked` / `static_roots` / `bake` / `invalidate_subtree` / `unfreeze_subtree`。
> - **烘焙 & 冻结**：`bake` 是**增量**的——仅重算 `valid == false` 的节点（新标记或被 `invalidate_subtree` 置脏的子树闭包），父优先 BFS 用 `baked[parent].mul_transform(local)` 预乘；静态根若挂在动态父上，则从调用方给的 `parent_globals` 取世界种子。重算 0 个节点时不 bump `generation`，故「静止场景重烘焙幂等」。返回 `BakeStats { baked, roots }`。
> - **批合并**：`merge_instances`（按 `BatchKey(u64)` 分组、按 key 升序排序、组内保持输入序，产出各组世界实例矩阵 `InstanceBatch`）与 `merge_meshes`（把各 `MeshSource` 的 local 顶点用 baked 世界变换烘焙到世界空间并拼接成单个 `MergedMesh`，记录每源顶点 `range`）。两者都只纳入 `is_baked` 的源，对标 UE 静态网格合并 / Unity static batching。
> - **可逆**：`unfreeze_subtree` 把子树退回动态集，`invalidate_subtree` 置脏并 bump 代际，供运行时「解冻移动」。
> - **纯确定性**：`no_std + alloc`、无线程 / 无时钟 / 无 ECS 依赖，核心对拍逐级传播 oracle 单测（`tests_bake.rs`，12 项）。

### 24.2 姿态量化压缩（Quantization）—— 网络 / 存储 / GPU 带宽 ✅ 已交付（`quantize`）

姿态在网络复制、存档、GPU 上传时按需量化,显著压带宽:

- **旋转**:四元数 smallest-three 编码（存 3 分量 + 2 bit 符号/最大位索引，约 32 bit），对标主流联机引擎。
- **平移**:按世界/cell 分层量化（粗格 + 格内定点），精度随距离/重要度分档。
- **缩放**:多数实体缩放=1,用 1 bit「是否单位缩放」旁路,仅异常者存全量。
- 解码在接收/读取侧,误差有界且可配置;与 §12 定点、`prism_replication` 快照格式契约一致。

> **交付状态（`pkg/prism_transform/src/quantize.rs`）：✅ 已交付并验证。**
> 全部编解码为纯 `no_std` 整型/`f32` 运算，固定「非负映射值 round-half-up」舍入规则，故编/解码跨平台位一致、往返误差永不超过广告界。
> - **旋转 smallest-three**：`QuatQuantizer` / `QuatQuantized`。先归一化并符号规范化（使被丢弃分量非负），丢掉绝对值最大的分量、存其 2 bit 索引 + 其余三分量在 `[-1/√2, 1/√2]` 上的量化码；解码由单位长约束重建被丢分量。默认 `QuatQuantizer::DEFAULT`（10 bit/分量）配 `to_bits`/`from_bits` 打包成单个 32 bit 字。`max_component_error = (1/√2)/(2^bits − 1)`。
> - **平移**：`BoundedQuantizer` / `QuantizedVec3` 在调用方给定的 AABB 盒内每轴均匀量化（越界饱和、落在栅格点精确、`max_abs_error` 为半步）；`LayeredQuantizer` / `LayeredTranslation` 用「粗 `cell_size` 整数格 + 格内定点」分层，精度处处一致（不随世界范围增长），`max_abs_error = 0.5·cell_size/(2^bits−1)`。
> - **缩放**：`ScaleQuantizer` / `ScaleQuantized` 用 1 bit「是否单位缩放」旁路（在 `tolerance` 内精确解回 `(1,1,1)`），仅非单位缩放付三分量码。
> - **整姿态**：`PoseQuantizer` / `QuantizedPose` 组合三者编解码整个 `Transform`，并暴露 `max_translation_error` / `max_scale_error` / `max_rotation_component_error` 误差界查询。
> - `no_std + alloc`（标量 `sqrt`/`floor`/`abs` 走 `libm`）。确定性往返与误差界单测（`tests_quantize.rs`，13 项，含手算 oracle 码值）。

### 24.3 双缓冲 / 多缓冲变换（渲染-仿真读取一致性）✅ 已交付（`double_buffer`）

配合 ECS §23.5 的渲染提取流水线:渲染线程读取的世界变换必须是**一帧内不被仿真写撕裂**的稳定快照。

- 维护「仿真写缓冲」与「渲染读缓冲」双份 Global 列,提取阶段原子切换(或 copy-on-extract)。
- 渲染可与下一帧仿真流水线并行(见 App §子应用流水线),互不读到半更新状态。
- 仅对参与渲染的变换列双缓冲,非渲染实体不付此内存成本。

> **交付状态（`pkg/prism_transform/src/double_buffer.rs`）：✅ 已交付并验证。**
> 泛型 `MultiBuffer<T: Copy>` 维护 `buffer_count` 份等长列（钳制到 ≥ 2），以 `write_idx` / `read_idx` / `version` 记录所有权。构造 `new` / `double` / `triple`，写侧 `write` / `write_at` / `write_all` / `prime_write_from_read`（把当前读列拷进写列，供只改部分实体的帧携带前帧其余项前进），读侧 `read` / `columns` / `read_token`，发布 `publish` / `publish_from`，另有 `resize` / `buffer_count` / `len` / `version`。
> - **无撕裂发布**：`publish` 执行 `read_idx = write_idx; write_idx = (write_idx + 1) % count; version += 1`——它只把「已写完」的列翻为可读，绝不改写读者可能持有的列。写列与读列初始即不混叠（`write_idx = 1, read_idx = 0`），故首次 `publish` 前读旧写新互不干扰。
> - **快照存活语义**：读者取 `Copy` 的 `ReadToken { buffer, version }`，之后用 `columns(token)` 重新以不可变借用读回那一列；`is_token_live` 据 `version` 判断仿真是否已旋转回该列。双缓冲下快照在下一次 `publish` 即失效；三（多）缓冲下仿真可边写 N+1 帧边让渲染读 N 帧快照，快照存活 `buffer_count - 1` 次 publish。
> - **借用即纪律**：类型本身无原子、无自带线程，仅编码「哪列可写、哪些是冻结快照」的所有权纪律，由调度器跨线程维持；写者可变借用写列、读者不可变借用快照列，借用检查器已禁止「边写边读同一列」。
> - 别名 `TransformDoubleBuffer` / `TransformTripleBuffer` = `MultiBuffer<GlobalTransform>`。`no_std + alloc`，确定性读写一致性单测（`tests_double_buffer.rs`，12 项）。

### 24.4 变换 Observer / 钩子（OnTransformChanged）✅ 已交付（`observer`）

接 ECS §12 Observer:变换变化可触发派生更新,免轮询:

```rust
world.observe::<OnChanged<GlobalTransform>>(|e, world| {
    // 更新空间索引 / 重定位音源 / 标记阴影缓存失效 / 唤醒休眠物理
});
```

- 典型订阅方:空间加速结构(§24.5)、音频空间化、阴影/反射探针缓存失效、触发器体积进出检测。
- Observer 批量合并(本帧多次变化只回调一次),避免抖动放大。

> **交付状态（`pkg/prism_transform/src/observer.rs`）：✅ 已交付并验证。**
> `TransformObserver` 是产生「哪些节点动了、动了多少」信号的纯数据结构：按节点维护「上次已上报的基线世界姿态」，一帧内一次或多次 `observe(dirty, globals)` 把脏节点及其新世界姿态折叠进待发集（**本帧内同一节点多次触碰合并**），`flush()` 按 `NodeId` 升序对净姿态异于基线者各发一条 `TransformChange`、推进基线、递增 `epoch`。
> - **确定性**：派发顺序只取决于节点下标；一帧内多次变化合并成一条净变化；一帧内「动了又动回基线」净零则不通知。
> - **API**：`new` / `with_capacity` / `push`（追加带基线的节点）/ `ensure_len`（以 `IDENTITY` 为基线扩容，新出现节点下次 flush 报告从单位姿态到真实姿态的变化）/ `prime`（以当前 `globals` 设基线且不发事件，用于首帧传播后播种）/ `observe` / `flush`（返回 `Vec<TransformChange>`）/ `dispatch`（回调队列形态）/ `epoch` / `len`。
> - `TransformChange { node, old, new }` 配 `translation_delta` / `moved(eps)` / `basis_changed(eps)` / `mask(eps) -> ChangeMask { translation, basis }`（`basis` 为 3×3 线性部整体变化；精细 rotation/scale 拆分留给订阅方从 `old`/`new` 自行分解）。
> - `no_std + alloc`，无原子、无线程、无 ECS 绑定——脏集与世界姿态由调用方（如 `dirty`）提供。确定性单测（`tests_observer.rs`，12 项）。

### 24.5 空间加速结构增量同步（BVH / Grid Hash）✅ 已交付（`spatial_sync`）

空间查询(拾取、范围检索、宽相位碰撞、剔除)依赖加速结构;变换变化应**增量更新**而非每帧重建:

- 动态实体变换变化(经 §24.4 钩子或脏集)驱动其在 BVH/网格哈希中的 refit/重插入。
- 静态实体(§24.1)入独立静态结构,一次构建永不动。
- 与 `prism_render_visibility`(剔除)、`prism_physics`(宽相位)、`prism_navigation`(空间查询)共享同一套同步契约,避免每个子系统各维护一份空间索引。

> **交付状态（`pkg/prism_transform/src/spatial_sync.rs`）：✅ 已交付并验证。**
> `SpatialSync` 是「变换脏集 → 空间加速结构增量指令」的纯确定性桥接层（不做真实 BVH/网格重建——那属空间索引 crate）。每个被追踪实体登记其**对象空间**（局部）`Aabb3`；`observe(dirty, globals)` 对脏且被追踪的代理，用世界 `GlobalTransform` 变换局部盒的八个角点并重拟合出**世界空间**紧致盒；`drain()` 产出按 `NodeId` 升序排序的 `SpatialCommand` 流（`Insert`/`Update`/`Remove`）与 `SyncStats` 计数，输出是「观测到的变化」的确定性函数，与登记/观测/移除的调用顺序无关。
> - **重拟合策略（`margin`）**：`margin == 0`（精确模式）任意世界盒变化（增或缩）都发 `Update`；`margin > 0`（胖盒模式）仿动态树纪律，仅当新紧致盒**逃出**已存胖盒时才发 `Update` 并重新加胖，盒内抖动不产生指令——与 `prism_physics` 宽相位动态树的 fat-AABB refit 一致。
> - **增量 diff**：`transform_aabb`（角点变换重拟合，对仿射精确）/ `world_aabb` 为可复用公共辅助；`register`/`remove` 支持热插拔，脏集可直接喂整棵层级（未追踪的脏节点被忽略，无需预过滤）；一帧内对同一节点多次 `observe` 取最后姿态；`drain_into` 复用调用方缓冲避免每帧分配。
> - `no_std + alloc`，无原子、无线程、无 ECS 绑定——脏集与世界姿态由调用方（如 `dirty`/`observer`）提供，产出的指令由空间索引 crate 消费。确定性 + 几何 oracle 单测（`tests_spatial_sync.rs`，17 项）。

### 24.6 轻量变换约束（Constraints: look-at / aim / parent-blend）✅ 已交付（`constraint`）

对标 Unity Constraints / Unreal 控制绑定的**运行时轻量约束**(非完整 rig,归 `prism_anim_runtime`):

| 约束 | 行为 |
|---|---|
| `LookAt` | 持续朝向目标实体/点(带轴/up 约束) |
| `Aim` | 局部轴对准目标(炮塔、摄像机) |
| `ParentBlend` | 在多个父之间按权重混合世界姿态(换乘过渡) |
| `PositionLimit` | 位置/旋转范围钳制 |

约束在传播**之后**求值并回写 Global(或反解 Local),顺序由 system set 固定;可链式但需显式声明依赖避免环。

> **交付状态（`pkg/prism_transform/src/constraint.rs`）：✅ 已交付并验证。**
> 对标生产引擎的运行时轻量约束（非完整 rig，rig 归 `prism_anim_runtime`）。每个求解器都是其输入的**纯确定性函数**：世界目标点/父姿态由调用方解析好传入（求解器从不做实体查找），算术为定序 `f32`，故同输入恒同输出。
> - `LookAt { target, up }`：保平移/缩放，重写旋转使节点前向（Prism 约定 `-Z`）指向世界目标，`up` 作滚转提示；节点落在目标上（视向为零）则保持原姿态；前向与 `up` 共线时确定性回退另一上轴。
> - `Aim { aim_axis, target }`：对某**局部轴**施加最小旋转使其世界像对准目标方向（炮塔/摄像机），不约束绕该轴的扭转。
> - `ParentBlend { poses: Vec<(GlobalTransform, f32)> }`：按权重线性混合平移/缩放、以符号对齐的归一化四元数均值（nlerp 式）混合旋转；无候选或权重和近零时回退 `IDENTITY`。
> - `PositionLimit { min, max }`：把世界平移钳进 AABB 盒，保旋转/缩放。
> - 辅助 `look_at_rotation` / `rotation_arc`（最短弧，含 180° 稳定回退）；`Constraint` 枚举 + `solve_chain(&[Constraint], current)` 按切片序折叠（顺序显著、调用方负责定序与避免环）。
> - `no_std + alloc`（标量数学走 `libm`），复用 `prism_math` 的 `Vec3`/`Quat`/`Mat3`。几何 oracle 单测（`tests_constraint.rs`，17 项）。

### 24.7 GPU 侧层级传播（Compute Hierarchy）✅ 已交付（`compute_hierarchy`，可移植调度 + 上传载荷层）

超大层级(集群动画、植被、人群)在 CPU 传播会成瓶颈。可把层级传播下放 GPU compute:

- 层级拓扑(父索引数组)与 Local 变换上传 GPU,compute shader 按层并行累乘出 Global。
- 结果留在 GPU 常驻缓冲,直接供实例化/间接绘制,免回读 CPU。
- 与 §13 GPU 变换列、ECS §15 GPU 驱动一致;仅对「只在 GPU 消费、CPU 无需读回」的子树启用(如纯视觉植被)。
- `gpu` feature + RHI(`prism_render_driver`)compute 能力探测门控。

> **交付状态（`pkg/prism_transform/src/compute_hierarchy.rs`）：✅ 已交付并验证（可移植的「调度 + 上传载荷 + CPU 参考」数据模型层；真实 WGSL dispatch 与设备常驻缓冲归消费侧，见 §24.9）。**
> §24.7 的层级传播在逻辑上与 CPU 并行传播（`parallel::LevelPlan`）同构：按**深度分层** dispatch，一层内的节点互相独立（父都在更浅、已求解的层），故可整层并行累乘。真正需要 GPU 的只是 compute dispatch 与设备缓冲；而「怎么分层、以什么顺序、上传什么字节」是**完全可移植、无需设备**的，本模块把这一层完整落地：
> - `LevelSchedule::build(&Hierarchy)`：`O(n log n)` 建每节点 `depth`（root=0，BFS 前向一遍）、计数排序进连续深度桶，**层内按 `NodeId` 升序**使 dispatch 顺序跨运行位相同（确定性）。暴露 `order()`（扁平 dispatch 序）/`level(i)`/`level_ranges()`/`node_depth()`/`level_count()`/`max_depth()`。它是 `no_std` 的 GPU dispatch 调度，`parallel::LevelPlan` 则是 `std` 线程池调度——同一分层事实、两套消费层，不互相桩。
> - `propagate_by_levels(hier,&schedule,locals,globals)`：**GPU twin 的 CPU 参考 / parity oracle**，逐层做与 shader 完全一致的「父世界 × 本地」累乘；用相同的 `Affine3` 复合与算子次序，故与串行 `propagate` **位对位相等**（仅求值分组不同），单测固定森林手算对拍。
> - `ComputeHierarchyInput::pack(hier,&schedule,locals,layout)`：**device-free 上传载荷**——`parents: Vec<i32>`（root=`-1`）、`local_matrices: Vec<u8>`（节点序、行主序打包本地 `Affine3`，复用输出路径同款 `MatrixLayout{RowMajor3x4=48B,RowMajor4x4=64B}` 以便 shader 共用解码）、`dispatch_order: Vec<u32>` + `level_ranges: Vec<(u32,u32)>`（驱动按层 dispatch）。驱动可 `memcpy` 直传、逐层 dispatch，无需再做 CPU 计算。
> - `no_std + alloc`，从不分配设备 / 开线程 / 读时钟；节点上限由 `i32` 父索引约束（`pack` 断言）。oracle + parity + 字节布局单测（`tests_compute_hierarchy.rs`，7 项），`cargo test` 零失败、`cargo clippy --all-targets` 零告警、无桩。

### 24.8 扫掠变换（Swept Transform）—— CCD / 运动模糊 ✅ 已交付（`sweep`）

保存实体「上一位姿 → 当前位姿」的扫掠信息,供两类消费:

- **连续碰撞检测(CCD)**:高速物体用扫掠体做宽相位,防穿透(供 `prism_physics`)。
- **运动模糊 / TAA 速度向量**:渲染用前后帧世界矩阵算屏幕空间速度(供 `prism_render_scene` 的 motion vector pass)。
- 复用 §10 插值已存的 prev/curr,无额外存储;瞬移(teleport)标记须清零扫掠,避免假速度拉花。

> **交付状态（`pkg/prism_transform/src/sweep.rs`）：✅ 已交付并验证。**
> `SweptMotion` 存实体「上一帧 → 本帧」世界姿态 + 对象空间 `Aabb3`，复用 §10 插值已有的 prev/curr 快照，无额外每实体存储。
> - **保守扫掠包围**：`bounds()` 返回**可证明保守**的世界 `Aabb3`——沿插值轨迹（平移/缩放 `lerp`、旋转最短弧 `slerp`，与 `GlobalTransform::interpolate` 一致）的每一时刻该盒都包含代理。构造复刻生产物理引擎的经典 temporal-AABB 界：取两端点世界盒之并集（平移沿直线段、缩放 `lerp` 使中间角点落在端点像的连线上，范数不超过端点，故线性/缩放运动已被并集覆盖），再按旋转弧长 `θ·r`（`θ` 扫掠角、`r` 角点到帧原点的半径）向各轴外扩以保守吞下旋转外凸（弧矢高 ≤ 弧长）。
> - **TOI 时间细分数据模型**：`bounds_sub(t0,t1)` 对任意子区间给保守盒，`subdivide(n)` 走 `[0,1]` 的等分区间产出每片 `SweepSegment{t0,t1,bounds}`——子区间越小 `θ` 与端点跨度越小、盒越紧，正是保守推进（conservative advancement）夹逼 TOI 所需；另有 `pose_at(t)`/`box_at(t)` 精确采样、`angular_motion()`/`translation_delta()` 运动度量。
> - **瞬移**：`teleported(curr, local)` 构造的扫掠包围塌缩为当前盒、报告速度为零，杜绝假速度拉花（供运动模糊 / TAA 速度向量）；`still(pose, local)` 为静止退化。
> - `no_std + alloc`，纯 `prism_math` 数学，无线程无时钟，从不改动权威仿真态。保守正确性由**稠密采样对拍**验证（平移/旋转/平移+旋转+缩放组合，每时刻采样盒 ⊆ 扫掠盒）；窄相位扫掠测试与真实 motion vector pass 归 `prism_physics`/`prism_render_scene`。几何 + 保守性单测（`tests_sweep.rs`，13 项）。

### 24.9 诚实边界

本章 **24.1 静态烘焙（`bake`）**、**24.3 双缓冲（`double_buffer`）**、**24.2 姿态量化（`quantize`）**、**24.4 变换 Observer（`observer`）**、**24.5 空间加速结构增量同步（`spatial_sync`）**、**24.6 轻量约束（`constraint`）**、**24.7 GPU 侧层级传播（`compute_hierarchy`，可移植调度 + 上传载荷层）**、**24.8 扫掠变换（`sweep`）** 已落地并通过验证（实现 + 单测，`cargo clippy --all-targets` 零告警、`cargo test` 零失败）；其余为 PLANNED 设计目标,无代码。诚实边界：

- **24.1 已交付的边界**：`StaticBaker` 是纯计算核心（并行数组 + 增量 `bake` + 批合并），不含「静态标记来源」——`TransformStatic` 组件语义、离线/加载期触发点仍属 `prism_scene`/`prism_asset_bake`，由消费方接线；静态根挂动态父时的世界种子需调用方显式传入 `parent_globals`。`merge_meshes` 产出 CPU 侧世界顶点缓冲，GPU 常驻/绘制合并由渲染侧消费。
- **24.3 已交付的边界**：`MultiBuffer` 是纯数据结构，自身无原子、无线程，只编码所有权纪律；跨线程的原子切换 / copy-on-extract 时序由 ECS §23.5 提取流水线与调度器维持，本 crate 不越界实现线程同步。
- **24.2 已交付的边界**：`quantize` 是纯编解码核心（smallest-three 旋转 + 盒内/分层平移 + 1 bit 单位缩放旁路 + 整姿态组合），只保证确定性与误差界；网络/快照的位流格式契约、分档策略与 §12 定点一致性由 `prism_replication` 消费方定义，本 crate 不越界定义线格式。
- **24.4 已交付的边界**：`observer` 是纯数据结构，无原子、无线程、无 ECS Observer 绑定——脏集与世界姿态由调用方（如 `dirty`/传播流水线）提供；本帧合并与升序派发是确定性契约，但精细的 rotation/scale 拆分、以及实际订阅方（空间索引 §24.5、音频、探针缓存失效）的接线留给消费方。
- **24.6 已交付的边界**：`constraint` 只做轻量无 rig 约束（`LookAt`/`Aim`/`ParentBlend`/`PositionLimit`），完整骨骼 rig 归 `prism_anim_runtime`；求解器在传播之后求值并回写 `GlobalTransform`，约束间的定序与避免环由调用方负责（`solve_chain` 不做环检测）；`to_scale_rotation_translation` 分解假设无 shear，父级非均匀缩放叠子级旋转产生的 shear 场景只在 TRS 可表示精度内保真。
- **24.5 已交付的边界**：`spatial_sync` 只是「变换脏集 → 空间加速结构增量指令」的纯桥接 diff 层，**不做真实 BVH/网格哈希的树重建、节点分裂/合并、SAH 优化**——那属空间索引 crate（`prism_render_visibility` 剔除树、`prism_physics` 宽相位动态树、`prism_navigation` 空间哈希），本 crate 只产出 `Insert`/`Update`/`Remove` 指令流由它们消费接线；脏集与世界 `GlobalTransform` 姿态由调用方（`dirty`/`observer`）提供，`SpatialSync` 不做实体查找、不触发传播；胖盒 `margin` 的重拟合纪律与真实树的 fat-AABB 一致，但最终树形态与查询性能取决于消费侧实现。
- **24.7 已交付的边界**：`compute_hierarchy` 只拥有**可移植、无设备**的三层——逐层 dispatch 调度（`LevelSchedule`）、与串行位对位相等的 CPU 参考 / parity oracle（`propagate_by_levels`）、以及 device-free 上传载荷（`ComputeHierarchyInput`：父索引 + 行主序本地矩阵 + dispatch 序 + 层范围）。**真实的 WGSL compute dispatch、GPU 存储缓冲上传/绑定、以及世界矩阵常驻 GPU 直供实例化/间接绘制（免回读）属消费侧 RHI/渲染驱动（`prism_render_driver` compute 能力探测 + `gpu` feature 门控），本 crate 不越界实现设备侧代码**；GPU twin 的正确性由上述 CPU 参考对拍保障，真实设备 parity 测试落在接线的渲染/驱动 crate。本层按节点序打包、层内按 `NodeId` 升序 dispatch，保证与串行传播确定性一致，但多队列/多 workgroup 的实际并行度与占用率取决于消费侧 shader 实现。
- **24.8 已交付的边界**：`sweep` 只给**保守扫掠包围与 TOI 时间细分数据模型**，不做窄相位求解——真实 CCD 命中（扫掠体 vs 几何的最早接触求根）归 `prism_physics`，真实 motion vector pass / 速度缓冲写出归 `prism_render_scene`；`bounds()` 的 temporal-AABB 界**保证保守（含整段轨迹）但非最紧**（旋转按弧长 `θ·r` 外扩，是上界非精确凸包），需更紧包围时应 `subdivide` 细分夹逼；插值假设与 `GlobalTransform::interpolate` 一致（平移/缩放 `lerp`、旋转最短弧 `slerp`），非此插值模型的运动不在保守性证明范围内；`SweptMotion` 从不改动权威仿真态，瞬移须由调用方以 `teleported` 显式标记。
- 其余 PLANNED：**24.7 的真实 GPU compute dispatch 与设备常驻缓冲**（可移植调度 + 上传载荷 + CPU 参考已交付如上，仅剩需 RHI/设备的接线部分）随 M6 GPU 驱动落地。

所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码;仅借鉴公开架构形态与经典数值。

