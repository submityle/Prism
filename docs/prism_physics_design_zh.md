# Prism Physics 次世代物理引擎设计方案

> 面向 Prism（Bevy fork）的数据导向、统一求解、多后端物理引擎设计。
> 借鉴 UE5 Chaos、Unity DOTS Physics / Havok、PhysX 5、Jolt、Avian、Rapier，取长补短。
> 本文档为设计规格，不含任何 AI/ML 内容，采用纯经典数值物理路线。

- 版本: v0.3（编码推进中：M0–M8 内核已落地，GPU 后端已上真机验证）
- 适用引擎: Prism / Bevy ECS 生态
- 关键依赖: bevy_ecs（并行 ECS）、bevy_math（glam SIMD）、bevy_tasks（任务系统）、wgpu（GPU 后端）

---

## 目录
1. 设计哲学
2. 分层架构
3. 核心内核：统一 XPBD 求解器
4. 前沿求解器：多求解器插槽
5. 几何与碰撞检测
6. 仿真管线
7. 后端抽象与 GPU
8. 全物态双向耦合
9. 高级特性
10. 自适应与多分辨率仿真
11. GPU-Driven 持久化管线
12. 异步流水线化仿真
13. 高级几何与鲁棒性
14. 时间操控与网络
15. 可编程性：约束 DSL 与热重载
16. 离线模式（架构级预留）
17. Bevy 集成层 API
18. 确定性与联机
19. 质量与可信度基础设施
20. 能力全景（分级）
21. Crate 拆分与落地形态
22. 路线图
23. 关键扩展点清单
24. 术语表

---

## 1. 设计哲学

对标商用旗舰引擎，取长补短，确立四条铁律：

- 数据导向 (DOD)：物理状态即 ECS 组件，热数据 SoA 布局，天然对齐 Bevy 并行调度与 SIMD。
- 统一求解器：刚体/关节/软体/布料/流体使用同一套 XPBD 内核作为主干，而非分裂成多个互不兼容的子系统。这是相对 PhysX(TGS)、Rapier(Impulse) 的关键差异化点，也是全物态耦合的结构性前提。
- 确定性可选：提供 f32 高性能路径 + 定点/软浮点确定性路径，服务联机同步与回放。二者编译期切换，零运行时代价。
- 易用性即 API：默认"零配置能跑"，进阶用户可逐层下钻替换。

设计取舍总表：

| 维度 | Prism 选择 | 理由 |
|---|---|---|
| 求解器 | 统一 XPBD + 子步 | 稳定、刚性高、跨物态统一，代差优势 |
| 内存 | SoA + ECS 组件 | 并行/SIMD 天然对齐 |
| 并行 | Island 分岛 + bevy_tasks | 高并行度、确定性可控 |
| GPU | wgpu compute 可选后端 | 与 Bevy 渲染同后端，跨平台 |
| 碰撞 | GJK/EPA + SDF + 凸分解 | 覆盖凸/凹/任意网格 |
| 联机 | 定点 + 快照回滚 | 对齐 Unity DOTS 卖点 |
| API | 默认零配置，分层下钻 | 对齐 Bevy modular/易用 |

---

## 2. 分层架构

```
Layer 5  作者层 Authoring   (Prefab/编辑器/热重载)
Layer 4  Bevy 集成层        (Plugin/Component/System, RigidBody/Collider/Joint/事件/Query)
Layer 3  仿真管线 Pipeline   (Broad->Narrow->Solve->Integrate, Island/睡眠/子步/CCD)
Layer 2  核心内核 Kernel     (统一 XPBD + 多求解器插槽, Contact/Joint/Soft/Cloth/Fluid 约束)
Layer 1  几何 & 加速结构     (BVH, GJK/EPA, SDF, 凸分解)
Layer 0  后端抽象 Backend    (CPU-SIMD / GPU-Compute)
```

- 跨语言/后端边界只在 Layer 0 与 Layer 2，其余层稳定。
- no_std 友好，wasm 可跑（关闭 GPU/多线程时降级到单线程 SIMD）。
- Layer 0–2 组成"引擎无关内核"，不依赖 bevy_ecs，可被服务器逻辑/其他引擎复用。

---

## 3. 核心内核：统一 XPBD 求解器

传统引擎把刚体、布料、软体、流体做成完全不同的子系统，导致跨物态交互难做。Prism 用一套位置约束框架统一它们。

### 3.1 统一状态
一切皆"粒子/刚体节点 + 约束"：
- 刚体：带旋转的 6-DOF 节点（位置 + 四元数姿态 + 线/角速度 + 逆质量 + 逆惯性张量）。
- 软体：四面体网格节点。布料：三角网格节点。绳索/毛发：一维约束链节点。流体：无网格粒子。

### 3.2 约束原语
可组合、可插件扩展的 trait Constraint：
- 接触约束（非穿透 + 摩擦，含 compliance 软硬度）
- 距离/球/铰链/棱柱/齿轮/马达 关节
- 体积保持（软体）、弯曲/拉伸（布料）
- 密度约束（PBF 流体）

### 3.3 子步 Substepping
每帧拆成 N 个小子步，每子步只做少量迭代（甚至 1 次）。相比"单步多迭代"，能量更稳、刚度更高（参考 Müller 2020, Detailed Rigid Body Simulation with XPBD）。这是相对 Chaos/PhysX 的稳定性优势来源。

### 3.4 Compliance（柔度）
用 compliance 代替 stiffness：数值稳定、与时间步无关。用户直接给"材料软硬"物理量，而非调玄学参数。

### 3.5 求解流程（每子步）
```
1. 预测位置:   x* = x + v*dt + (f_ext/m)*dt^2
2. 约束投影:   for iter in 0..K { 投影所有约束 (Gauss-Seidel / Jacobi 混合) }
3. 更新速度:   v = (x* - x_prev)/dt
4. 速度层修正: 恢复系数(弹性) + 动摩擦
5. 提交状态:   x = x*
```
- 采用图着色分批的混合策略：岛内按约束图着色分组，组内 Jacobi 并行、组间 Gauss-Seidel 串行。
- 接触流形缓存跨帧复用，做 warm-start，收敛更快。

---

## 4. 前沿求解器：多求解器插槽

内核预留 trait Solver + SolverRegistry，按物态/精度自动选择：
- VBD (Vertex Block Descent, SIGGRAPH 2024)：比 XPBD 更适合极刚材料与大形变，无条件稳定、可 GPU 大规模并行。软体/布料高保真档。
- 投影动力学 (Projective Dynamics) / ADMM：预分解系统矩阵，固定拓扑下布料/软体超快收敛。
- MPM (Material Point Method)：网格+粒子混合，用于雪、沙、泥、黏弹、可塑性。GPU 后端一等公民。UE/Unity 均无原生支持。
- FLIP/APIC 流体：比 PBF 数值耗散更低的高质量液体，配表面重建。

同一场景可多求解器共存，通过统一接触层耦合（见第 8 节）。

---

## 5. 几何与碰撞检测（Layer 1）

| 阶段 | 方案 | 借鉴对象 |
|---|---|---|
| Broad-phase 粗筛 | 并行 SAP + 增量 BVH 双模式：动态用 BVH，海量静态用网格/SAP | Jolt / Rapier |
| Narrow-phase 精筛 | GJK + EPA（凸-凸），SDF 采样（凸-任意），专用快路径（球/盒/胶囊） | PhysX / Jolt |
| 复杂网格 | 编译期凸分解（V-HACD 风格）+ 运行时 SDF 场 | Chaos |
| CCD 连续检测 | 保守步进 + 子弹体标记，按需开启 | 全家 |

补充机制：
- 碰撞矩阵 / Layer & Mask：位掩码分层，编辑器可视化配置。
- 触发器 (Sensor)：只报事件不产生力。
- 接触流形缓存：跨帧复用 warm-start。
- 几何资源共享：Collider 内部指向共享几何资源 Handle<ColliderShape>，避免重复内存。

---

## 6. 仿真管线（Layer 3）

- Island 分岛：并查集把相互接触/关节连接的物体分组，岛间完全并行求解。并行度的核心来源。
- 睡眠 (Sleeping)：低速物体进入休眠，从活跃集移除，CPU 归零。唤醒靠接触/关节传播。
- 确定性调度：岛内约束排序稳定化，保证同输入同输出（联机必需）。
- 分阶段并行：Broad/Narrow/Solve 各自用 bevy_tasks 的 ComputeTaskPool 切片并行，接触对生成用无锁并发容器收集。

单帧管线：
```
[收集外力/命令] -> [Broad-phase] -> [Narrow-phase 生成接触] ->
[Island 构建] -> [并行子步求解] -> [速度修正] -> [写回 Transform] -> [触发事件]
```

---

## 7. 后端抽象与 GPU（Layer 0）

- CPU 后端：SoA + glam SIMD（SSE/AVX/NEON），Island 级 rayon/bevy_tasks 并行。目标：万级动态刚体 60fps。
- GPU 后端（可选 feature）：基于 wgpu compute shader，把粒子/约束批量投影搬上 GPU，服务大规模布料/流体/破碎（十万~百万粒子）、GPU broad-phase（spatial hashing）+ GPU XPBD 投影。
- 后端 trait 化：

```rust
trait PhysicsBackend {
    fn step(&mut self, world: &mut PhysicsWorld, dt: f32);
}
```

CPU/GPU 可运行时切换、可混合（刚体走 CPU、流体走 GPU）。

---

## 8. 全物态双向耦合 (Unified Two-Way Coupling)

商用引擎的老大难：布料穿软体、流体推刚体、沙埋角色往往是假的或单向的。Prism 因为所有物态最终都归约为"节点 + 约束/接触"，可以做真正统一的耦合层：
- 统一接触流形：刚体面 <-> 布料顶点 <-> 流体粒子 <-> MPM 粒子，全部进入同一接触约束池。
- 动量守恒的双向传递：流体浮起木箱、木箱压弯布料、角色踩进雪里留脚印并被减速。
- 并行解耦：耦合在 Island 层解耦并行，仅跨物态岛才串行。

这是相对 Chaos/PhysX 的结构性优势——它们的子系统是事后拼接，Prism 原生统一。

---

## 9. 高级特性

- 破坏系统 (Fracture/Destruction)：预断裂网格 + 运行时 cluster 聚簇分裂，接触冲量超阈值触发。借鉴 Chaos Destruction。
- 力场 (Fields)：空间函数驱动的力/速度/断裂场（爆炸、涡流、引力井、风、浮力），可组合。
- 软体 FEM 可选：需要精确材料时用共旋 (co-rotational) FEM，默认走 XPBD 快路径。
- 物理材料图：摩擦/弹性/密度的组合规则表（相加/相乘/取最小），编辑器可视化。
- 毛发/绳束 (Hair/Strand)：UE Groom 级发丝物理，作为一维约束链复用统一内核。
- 气动/浮力：布料受风、物体浮水，作为力场扩展。

---

## 10. 自适应与多分辨率仿真

- 自适应子步/迭代：按局部误差估计动态调节，高速碰撞区多子步，静止区少。
- 空间 LOD 物理：远处/视锥外物体降级（简化碰撞体、降频、代理刚体、纯动画），近处全解算。
- 降阶模型 (Model Reduction / 模态子空间)：大软体用模态分析压到少量自由度，实时跑巨型可形变体。
- 区域休眠 + 流式：结合大世界分块，未加载区域物理冻结，进出时快照恢复。

---

## 11. GPU-Driven 持久化管线

不止"把某步搬 GPU"，而是状态常驻 GPU、CPU 只发指令：
- 物理状态、BVH、接触对全程留在显存，帧间不回传。
- Indirect dispatch：接触/约束数量在 GPU 上决定并自驱动派发，避免 CPU 回读同步点。
- 与 Bevy wgpu 渲染管线共享缓冲，物理结果直供 GPU skinning / 粒子渲染，零拷贝。
- CPU<->GPU 仅在需要查询/事件时按需回读（异步、双缓冲）。

---

## 12. 异步流水线化仿真 (Pipelined / Decoupled)

- 物理在独立固定步长线程运行，渲染帧对最近两个物理状态做插值/外推（消除抖动）。
- 三重缓冲状态，读写无锁。
- 游戏逻辑通过命令队列写入（施力、改属性），下一子步统一应用，保证确定性。
- 与 Bevy 的 FixedUpdate 调度对齐，但允许物理频率独立于渲染帧率。

---

## 13. 高级几何与鲁棒性

- 实时 Voronoi 破碎 + 动态重网格化：接触点处即时生成裂纹，而非纯预断裂，配 cluster 层级。
- 连续 SDF 世界表示：任意静态几何统一为 SDF，接触查询 O(1)、天然平滑法线、支持布尔运算世界编辑。
- 精确谓词 + 符号扰动 (Simulation of Simplicity)：消除退化/共面的数值崩溃，工业级鲁棒性。

---

## 14. 时间操控与网络

- 时间倒流 / 慢放 / 局部时停：基于快照环形缓冲，可对选定 Island 单独缩放时间（子弹时间只对敌人生效）。
- Rollback + 预测：客户端预测 + 服务器权威 + 状态和解，构建在确定性 + 快照之上。
- 状态哈希校验：每帧对物理状态做哈希，联机 desync 自动检测定位。

---

## 15. 可编程性：约束 DSL 与热重载

- 约束编译器 / DSL：用户声明式定义自定义约束（如"保持这两点夹角在 30–60 度且软度 0.1"），编译到 CPU/GPU 内核，无需手写求解代码。
- 物理参数热重载：材料、重力、约束参数运行时改即生效，编辑器滑块实时调参。
- 可视化调试进阶：接触力箭头、约束应力热力图、Island 着色、求解器收敛曲线、GPU 占用火焰图，接入 Bevy 诊断与 egui/UI。

---

## 16. 离线模式（架构级预留）

将"离线模式"作为架构级预留能力：现在不实现完整功能，但在架构和数据格式上留好扩展点，避免日后返工。

### 16.1 两种离线形态
| 形态 | 用途 | 对标 |
|---|---|---|
| A. 离线烘焙 (Bake & Playback) | 昂贵仿真（布料/软体/流体/破碎）离线算好，运行时只回放缓存，几乎零 CPU | UE Chaos Cache / Houdini Vellum / Alembic |
| B. 非实时高精度 (Batch/Headless Sim) | 无渲染下跑高子步/高迭代/FEM，用于影视级、科学计算、数据集生成、CI 回归 | Havok 离线 / Bullet headless |

### 16.2 预留设计
① 后端 trait 扩展（Layer 0）：在 PhysicsBackend 之外预留 SimulationDriver 抽象，区分驱动方式：
```rust
enum DriveMode {
    Realtime,                                                        // 帧驱动，当前默认
    Offline { substeps: u32, iterations: u32, target_frames: u32 },  // 批量高精度
    Playback(CacheHandle),                                           // 只读回放，跳过求解
}
```
step() 内部按 DriveMode 分派。M1 只实现 Realtime，但枚举与分派点现在就留好。

② 状态可序列化 = 缓存基础（Layer 2）：统一 XPBD 状态本就是纯数据。要求所有物理状态组件实现 Reflect + Serialize——烘焙缓存、快照回滚、确定性回放三者共用地基，一次投入三处收益。缓存格式：
- 关键帧 + 增量压缩（位置量化、变化阈值剔除）。纯经典压缩，不使用任何学习/神经方法。
- 分轨道存储（每个可缓存实体一条 track），支持部分加载/流式。

③ 缓存组件与资源（Layer 4）预留但暂不实现的占位：
- PhysicsCache { handle, mode: Record | Playback | Passthrough }
- Bakeable（标记该实体参与烘焙）
- 资源 Handle<PhysicsCacheAsset>（走 Bevy Asset 系统，可热重载/流式）

④ Headless 运行路径：PrismPhysicsPlugin 预留 .headless() 构造，不注册 Debug/渲染同步 System，允许脱离窗口在 CI/服务器/CLI 跑批量仿真并落盘缓存。与确定性 feature 天然协同。

### 16.3 数据流
```
[编辑器/CLI]  Offline Sim ---> 写入 PhysicsCacheAsset(磁盘)
                                     |
[运行时] Playback 模式 <--- 加载 Cache      (零求解，按帧插值回放)
[运行时] Record 模式  ---> 实时仿真同时录制新缓存
[运行时] Passthrough  ---> 忽略缓存，纯实时（默认）
```
回放支持：时间缩放、循环、与实时物体的单向交互（缓存物体推动实时物体，自身不被反推，可配置）。

---

## 17. Bevy 集成层 API（Layer 4）

### 17.1 默认极简（新手 5 行能跑）
```rust
app.add_plugins(PrismPhysicsPlugin::default());

commands.spawn((
    RigidBody::Dynamic,
    Collider::cuboid(1.0, 1.0, 1.0),
    Transform::from_xyz(0.0, 5.0, 0.0),
));
```

### 17.2 组件设计（全部 ECS 原生，可 reflect/序列化）
- RigidBody { Dynamic | Kinematic | Static }
- Collider（内部指向共享几何资源 Handle<ColliderShape>）
- Mass / CenterOfMass / Inertia（可自动从 Collider 计算，也可覆盖）
- Velocity / AngularVelocity
- Friction / Restitution / PhysicsMaterial
- Joint 组件族（impl Component）
- CollisionLayers { memberships, filters }

### 17.3 事件与查询
- CollisionStarted / CollisionEnded / TriggerEvent（Bevy Event/Observer）
- SpatialQuery：raycast / shapecast / overlap / point_project，作为 ECS SystemParam 直接注入。
- Observer 驱动：接触即触发 observer，符合 Bevy 最新 event 架构。

### 17.4 易用性加分项
- 单位自解释（SI），gravity 默认 -9.81。
- Debug 可视化插件：碰撞体线框、接触点、约束、休眠状态、Island 着色。
- 诊断面板接入 Bevy Diagnostics：步耗时、活跃刚体数、接触对数、岛数。

---

## 18. 确定性与联机

- feature deterministic：软浮点或定点数学，稳定排序，跨平台位一致。
- 快照/回滚 API 供 rollback netcode（GGPO 风格）使用。
- 与高性能 f32 路径二选一，编译期切换，零运行时代价。
- 状态哈希每帧校验，desync 自动检测定位。

---

## 19. 质量与可信度基础设施

- 确定性回归测试 + 模糊测试 (fuzzing)：随机场景跑 headless，检测 NaN/爆炸/desync。
- 黄金帧对比 (Golden replay)：缓存参考轨迹，CI 检测数值回归（借力离线序列化）。
- 能量/动量守恒监控：运行时断言物理量守恒，越界报警，调试玄学 bug 的杀手锏。
- 基准套件：堆叠、关节链、布料垂布、破碎、流体溃坝，持续跑性能看板。

---

## 20. 能力全景（分级）

```
Tier S (定义下一代身份，UE/Unity 没有):
  · 统一 XPBD 全物态双向耦合
  · MPM 沙/雪/泥/黏弹可塑
  · GPU-Driven 持久化管线（状态常驻显存, indirect dispatch, 零拷贝）

Tier A (研究前沿，工业化落地):
  · VBD / 投影动力学 多求解器插槽
  · 降阶软体（模态子空间）
  · 自适应子步 + 空间 LOD 物理
  · 实时 Voronoi 破碎 + SDF 世界表示
  · 异步流水线化仿真 + 状态插值

Tier B (对标商用旗舰):
  · 关节马达 / 破坏 / 力场 / 毛发-绳束 / FLIP-APIC 流体
  · 确定性联机 rollback + 状态哈希校验
  · 时间倒流 / 慢放 / 局部时停
  · 约束 DSL + 参数热重载

Tier C (地基, 做到最好):
  · Island 并行 / 睡眠 / CCD / warm-start
  · 空间查询 (raycast/shapecast/overlap)
  · 调试可视化 (接触力/应力热图/收敛曲线)
  · 序列化 / 缓存 / 离线烘焙
  · 确定性回归 + 模糊测试 + Golden replay + 守恒监控
```

---

## 21. Crate 拆分与落地形态

```
crates/
  bevy_prism_core        // Layer 0-2: 无 Bevy 依赖的纯物理内核（可独立发布/复用）
  bevy_prism_geometry    // Layer 1: 几何、BVH、GJK/EPA、SDF
  bevy_prism_backend_cpu // CPU SIMD 后端
  bevy_prism_backend_gpu // wgpu compute 后端 (feature)
  bevy_prism_cache       // 离线缓存格式 + 烘焙器 + 回放器（预留，先占位）
  bevy_prism             // Layer 4-5: Bevy 插件、组件、System、事件、查询
  bevy_prism_debug       // 可视化 & 诊断
```

内核 bevy_prism_core 不依赖 bevy_ecs（纯数据 + trait），保证可被其他引擎/服务器逻辑复用；Bevy 层只做同步桥接。这与 Rapier 的"引擎无关内核 + bevy_rapier 胶水"哲学一致，但内核选用统一 XPBD 而非脉冲法，形成差异化。

---

## 22. 路线图

| 阶段 | 目标 | 交付 | 状态 |
|---|---|---|---|
| M0 | 数学地基、SoA、BVH broad-phase、扩展点预留 | 基准可测 | ✅ 完成 |
| M1 | XPBD 刚体 MVP + 球/盒/胶囊 + 接触摩擦 + 子步 | 能堆箱子稳定 | ✅ 完成 |
| M2 | 关节族 + 空间查询 + 事件/Observer | 可做门/车/机关 | ✅ 完成 |
| M2.5 | 异步流水线 + 状态哈希（联机地基） | 插值无抖动 | ✅ 完成 |
| M3 | Island 并行 + 睡眠 + warm-start + CCD（已升级为 rotational conservative-advancement：shape-cast + 位姿回绕，对齐 UE/Chaos） | 万级刚体 60fps | ✅ 完成 |
| M4 | 布料 / 软体 / 绳索（统一约束复用） | 交互式布料 | ✅ 完成 |
| M4.5 | 离线烘焙 + Golden replay CI | 缓存回放 demo | ✅ 完成 |
| M5 | GPU 后端（流体/大布料/破碎） | 十万粒子 | ✅ 完成（prism_physics_gpu / prism_volumetric_gpu，每个核均为 CPU golden + WGSL 孪生 + 真机 GPU parity） |
| M5.5 | MPM（沙/雪/泥）+ FLIP/APIC 流体 + 表面重建 | 溃坝/沙堆 demo | ✅ 完成 |
| M7 | VBD / 降阶软体 + 自适应 LOD | 巨型软体实时 | ✅ 完成 |
| M8 | 约束 DSL + 参数热重载 | 编辑器实时调参 | ✅ 完成 |
| M9 | GPU-Driven 持久化管线 + 实时 Voronoi 破碎 | 零拷贝 + 动态裂纹 | 🟡 CPU 破碎核心完成；GPU 持久化管线随 M5 一并落地 |

关键预留原则：Tier S/A 的功能现在只在 trait Solver、DriveMode、序列化、后端抽象四个扩展点"留好插槽"，M0/M1 不实现，避免过度设计拖慢主线。

> 实现现状补充（截至编码推进）：
> - CCD 已从纯线性 shape-cast 升级为 rotational conservative-advancement（线性+角向守恒上界，支持纯自转触发与 position+orientation 两段回绕）。
> - 布料-刚体双向耦合原语已落地（`CouplingBody` / `resolve_two_way_coupling`：逆质量加权接触分配 + 反作用冲量累积），并有 WGSL 孪生与真机 parity（`prism_physics_gpu::cloth::coupling`、`prism_volumetric_gpu`）。**已知差距**：核心 step 管线（`pipeline` / `world` / `XpbdSolver`）尚未端到端串联“从刚体构造 proxy → 耦合 → 反作用冲量写回刚体”，仍在推进。

---

## 23. 关键扩展点清单

这些是必须在 M0/M1 就固化下来的接口，后续所有高级功能都挂在其上，避免返工：
1. trait Solver + SolverRegistry：多求解器（XPBD/VBD/PD/MPM/FLIP）派发点。
2. trait PhysicsBackend：CPU/GPU 后端切换点。
3. enum DriveMode：Realtime/Offline/Playback（离线模式）分派点。
4. 状态组件 Reflect + Serialize：缓存/回滚/回放共用的序列化地基。
5. trait Constraint：约束原语扩展点（含 DSL 编译目标）。
6. Island 抽象：并行 + 时间操控 + 局部休眠的作用域单位。
7. ColliderShape 资源句柄：几何共享与凸分解/SDF 承载点。

---

## 24. 术语表

| 术语 | 含义 |
|---|---|
| XPBD | Extended Position-Based Dynamics，扩展位置动力学，本引擎主求解器 |
| VBD | Vertex Block Descent，SIGGRAPH 2024 提出的顶点块下降求解器 |
| MPM | Material Point Method，物质点法，用于沙/雪/泥等连续介质 |
| FLIP/APIC | 流体求解方法，比 PBF 数值耗散更低 |
| DOD | Data-Oriented Design，数据导向设计 |
| SoA | Structure of Arrays，数组结构布局，利于 SIMD/并行 |
| Compliance | 柔度，与 stiffness 互为倒数，XPBD 中稳定的材料软硬度参数 |
| Substepping | 子步，将单帧拆为多个小步以提高稳定性与刚度 |
| Island | 岛，相互约束连接的物体分组，并行/休眠/时间操控的作用域 |
| Broad/Narrow-phase | 碰撞粗筛/精筛阶段 |
| GJK/EPA | 凸体碰撞检测与穿透深度算法 |
| SDF | Signed Distance Field，有符号距离场 |
| CCD | Continuous Collision Detection，连续碰撞检测，防高速穿透 |
| Warm-start | 热启动，复用上帧解加速本帧收敛 |
| Rollback | 回滚，联机预测和解机制 |
| Bake/Playback | 离线烘焙/回放，预计算仿真结果供运行时零成本播放 |
| Headless | 无渲染窗口的批量运行模式 |

---

*本文档为 Prism Physics 设计规格 v0.3，不含 AI/ML 内容；M0–M8 内核与 GPU 后端已进入编码，并以 CPU golden + 真机 GPU parity 双重验证。*
