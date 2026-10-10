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
24. 动态装配与可构造结构（刚体组装 / 复合体合并 / 可断关节 / 力场驱动）
25. 气动力系统（刚体/多体空气动力学）
26. 关节限位与角色布偶（Swing-Twist 锥限 / 柔韧阻尼 / 布偶稳定）
27. 降阶坐标铰接（Reduced-Coordinate Articulation：Featherstone）
28. 布料/壳体撕裂的拓扑级深化（顶点分裂 / 裂纹前沿 / 穿刺）
29. 术语表

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
- 固定(焊接)/距离/球/铰链/棱柱/马达 关节（齿轮关节为规划项）；关节可选断裂阈值（见 §24）。球关节 swing-twist 锥限/扭限、关节柔韧与伺服 drive 见 §26；强铰接链可选降阶坐标求解见 §27
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
- 力场 (Fields)：空间函数驱动的力/速度/断裂场（爆炸、涡流、引力井、风、浮力），可组合（一等子系统详见 §24.8）。
- 软体 FEM 可选：需要精确材料时用共旋 (co-rotational) FEM，默认走 XPBD 快路径。
- 物理材料图：摩擦/弹性/密度的组合规则表（相加/相乘/取最小），编辑器可视化。
- 毛发/绳束 (Hair/Strand)：UE Groom 级发丝物理，作为一维约束链复用统一内核。
- 气动/浮力：布料受风、刚体受风/升力/推进、物体浮水/浮空，作为力场扩展（刚体与多体气动详见 §25）。

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
  · 动态刚体装配：复合体合并/拆分 + 可断关节 + 吸附对齐原语（究极手级构造, §24）
  · 刚体/多体气动：面元升阻+失速 + 推进反扭矩 + 浮力 + 空间风场/滑流（究极手级飞行, §25）
  · 降阶坐标铰接（Featherstone ABA）：极大质量比/长链零漂移、少子步稳定（§27）

Tier B (对标商用旗舰):
  · 关节马达 / 破坏 / 力场 / 毛发-绳束 / FLIP-APIC 流体
  · 角色布偶：swing-twist 椭圆锥限 + 软限位 + 关节伺服 drive（§26）
  · 布料拓扑级撕裂：顶点分裂生成裂口 + 裂纹前沿传播 + 穿刺（§28）
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
| M10 | 动态装配：力累加器/施力 API + 通用可断关节 + 装配图/复合体合并·拆分 + 吸附几何原语 + GrabDrive | 实时搭建 200+ 零件载具 60fps、焊点可断、抓取不穿墙 | ⬜ 规划（§24.12） |
| M11 | 力场子系统 + GPU 采样 + 风扇/推进器/浮力部件 + 断裂连锁 | 自建载具可驾驶/飞行、受击解体 | ⬜ 规划（§24.12） |
| M12 | 刚体/多体气动：各向异性阻力面元 + 翼面升力/失速 + 推进反扭矩 + 浮力（空气/水）+ 空间风场/滑流/地面效应 + 附加质量 | 自建飞行器可滑翔/配平飞行、气球浮空、水上载具 | ⬜ 规划（§25.15） |
| M13 | 角色布偶：球关节 swing-twist 椭圆锥限/扭限 + 软限位 + 关节摩擦 + 位姿伺服 drive + RagdollBuilder | 人形布偶命中反应自然、不穿模不软面条、受击可配主动驱动 | ⬜ 规划（§26.11） |
| M14 | 降阶坐标铰接：Featherstone ABA O(n) 正向动力学 + 广义坐标限位/马达 + 与 XPBD 接触耦合 | 长机械臂/角色强铰接链极大质量比零漂移、少子步稳定 | ⬜ 规划（§27.9） |
| M15 | 布料拓扑撕裂：顶点分裂 + 裂纹前沿 + 穿刺 + 各向异性织物 + 渲染网格缝合 + GPU 拓扑变更 | 旗帜被弹片撕开真实裂口、刀划布、撕裂口参与自碰撞 | ⬜ 规划（§28.10） |

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
8. AssemblyGraph + CompoundBody：运行时可逆的连接图与复合体合并/拆分作用域（动态装配、破坏重组的真相源，见 §24）。

---

## 24. 动态装配与可构造结构（刚体组装 / 复合体合并 / 可断关节 / 力场驱动）

> **定位**：物理内核层对"运行时把若干刚体吸附、拼接、焊成一个可整体仿真的结构，并可受力驱动/断裂/拆解"的完整支撑——即《塞尔达：王国之泪》"究极手（Ultrahand）"、载具搭建、破坏重组等玩法所依赖的**底层物理能力**。本节只定义物理原语（几何/约束/求解/数据），不涉及任何玩法语义（吸附 UI、组装目录、输入、网络裁定），后者由 Gameplay 层调用本节 API 实现。

### 24.1 对标与借鉴

| 项目 | 可借鉴点 | Prism 取舍 |
|---|---|---|
| Zelda: TotK（Ultrahand/Zonai） | 吸附对齐手感、焊接即时成体、整体物理一致 | 吸附走几何特征对齐原语；焊接走复合体合并而非纯关节网络 |
| UE5 Chaos（GeometryCollection / Clustering） | cluster 聚簇层级、union/强度阈值破碎、proxy 简化 | 装配图 = cluster 的运行时可逆版本；断裂复用阈值+事件管线 |
| PhysX 5（Aggregate / Articulation） | aggregate 降低 broad-phase 自碰撞对、articulation 关节链刚性 | 复合体合并借鉴 aggregate 的 self-collision 屏蔽与单岛调度 |
| Besiege / Trailmakers / Scrap Mechanic | 大量部件实时搭建的稳定性与性能 | 合并刚体 + 可断连接是其稳定性关键，直接作为一等能力 |
| Teardown | 体素结构连通性断裂 + 实时重算 | 连通性走并查集增量，断裂触发子图分裂 |
| Media Molecule *Dreams* | 玩家创作物的通用约束/焊接 | 焊接 = 退化为 0-DOF 的 `FixedJoint`，统一进 XPBD |

### 24.2 核心抽象：装配图与复合刚体

```
AssemblyGraph                     // 运行时可逆的"连接关系图"
  nodes: Vec<BodyHandle>          // 参与装配的刚体（零件）
  edges: Vec<Connection>          // 连接（焊接/铰接/可断）
  components: UnionFind           // 连通分量 = 一个待合并的复合体

Connection
  a, b: BodyHandle
  kind: FixedJoint | RevoluteJoint | PrismaticJoint | ...
  breakable: Option<BreakLimit>   // 断裂阈值（见 24.4）

CompoundBody                      // 由一个连通分量烘焙而成的单一刚体
  parts: Vec<(BodyHandle, Isometry)>  // 零件在复合体局部系的相对位姿
  merged_mass, merged_com, merged_inertia
  aggregated_collider: ColliderShape  // 复合碰撞体（子形状聚合）
```

两种运行形态，可按连通分量规模自动切换：

- **关节网络形态（Soft）**：连接保留为 XPBD 关节，零件各为独立 Dynamic 刚体。保留连接处微弹性与可断性，适合小规模（≤ 阈值 N 个零件）或需要"受力会变形/散架"的结构。
- **复合体形态（Rigid）**：把连通分量烘焙成**单一刚体**（合并质量/质心/惯性 + 聚合碰撞体），零件降为该刚体的从属几何。彻底消除内部关节求解，稳定性与性能达到单刚体水平，适合大型载具/建筑。

### 24.3 复合体合并与拆分（核心新增能力）

**合并（Bake）管线**（`merge_component(graph, root) -> CompoundBody`）：

1. 遍历连通分量，取各零件 `Isometry` 换算到复合体局部系。
2. 合并质量：`M = Σ mᵢ`；质心 `C = (Σ mᵢ·cᵢ)/M`。
3. 合并惯性：各零件惯性张量经**平行轴定理**平移到 `C`、旋转到复合体系后相加，得 `I`；对称化 + 特征分解得主轴与 `inv_inertia`（复用 `collider/inertia`）。
4. 聚合碰撞体：子形状以相对位姿组成 `Compound` 形状；凹面零件先走既有凸分解（`collider/decompose`）。
5. 内部对屏蔽：复合体内部零件间**不做自碰撞**（借鉴 PhysX aggregate），broad-phase 以单一代理 AABB 参与。

**拆分（Split）**：断开一条连接或删除一个零件时，在装配图上做**增量并查集**判定连通性是否分裂：

- 若仍连通：局部重烘焙受影响复合体（只重算质量属性与聚合 AABB，几何增量更新）。
- 若分裂为多个分量：按新分量各自重烘焙为独立复合体，**线动量/角动量按质量与位置守恒重分配**到新复合体（避免断裂瞬间速度跳变）。

**可逆性**：装配图是真相源，复合体是其派生缓存。合并/拆分均为纯函数式重建，天然支持快照回滚（§14/§18）与编辑器 Undo。

### 24.4 可断关节（Breakable Joints）

当前 `FixedJoint` 无断裂阈值，本节补齐（作用于**所有** `JointKind`）：

```
BreakLimit
  max_force:  f32         // 累积线性约束冲量阈值（N·s/子步 换算到力）
  max_torque: f32         // 累积角向约束冲量阈值
  fatigue: Option<Fatigue>// 可选：疲劳/塑性（反复受力累积损伤后降低阈值）
```

- 求解时读取该关节本子步累积的拉格朗日乘子 `λ`（XPBD 已有），换算为等效力/力矩；超限则标记断裂。
- 断裂在**子步末统一结算**（避免求解中途改拓扑破坏确定性），产出 `JointBrokenEvent{ joint, impulse, position }`，供 §13 破坏系统/VFX/音频订阅。
- 与破坏系统衔接：焊接连接断裂可触发局部 Voronoi 裂纹（§13），形成"受力过大 → 焊点崩开 → 碎裂"的连锁。
- 塑性可选：超弹性限但未达断裂限时，连接产生永久偏移（bending/stretch 残余），模拟金属弯折。

### 24.5 吸附装配几何原语（物理侧，非玩法）

吸附"对齐"本质是几何求解，放在物理层作为 `SpatialQuery` 扩展提供；"要不要吸附、吸附提示 UI、确认输入"属玩法层。

- **特征提取**：从 `ColliderShape` 预计算可吸附特征——面（中心+法线+外接）、边、预定义**锚点**（`AttachPoint{ local_pos, local_normal, tag }`，随几何资源烘焙，可被编辑器/资产标注）。
- **对齐求解**（`solve_snap(moving, targets, policy) -> Option<Isometry>`）：
  - 面-面：令动件候选面贴合目标面（法线反向对齐 + 面内投影贴近），返回达成贴合所需的目标位姿。
  - 锚点-锚点：法线相对对齐 + 位置重合，产出规整拼接（载具/部件首选）。
  - 返回"预吸附位姿 + 吸附质量评分"，玩法层据此渲染 ghost 预览并决定是否确认。
- **预览即查询**：ghost 预览不改物理状态，仅走 `shapecast/overlap/project` 判断落点是否被遮挡/穿插；确认后玩法层才调用 `connect()` 建连接或 `merge_component()` 成体。

### 24.6 柔性抓取 / 操纵原语（Grab / Drive）

"举起物体悬浮在手上"用**驱动约束**实现，保证全程走物理求解（不穿墙、能互推）：

```
GrabDrive                         // 把目标刚体柔性拉向一个运动学"手点"
  target: BodyHandle
  anchor_world: Isometry          // 玩法层每帧写入的期望位姿
  linear_compliance, angular_compliance   // 柔度=弹簧软硬
  max_force, max_torque           // 冲量钳制（防抽搐/爆炸，对应 BreakLimit 复用）
  max_linear_speed, max_angular_speed      // 速度上限
```

- 实现 = 一个运动学抓取体 + 带 `compliance` 与冲量上限的 `FixedJoint`，复用现有 Motor 的 `λ_max = n·h²` 钳制逻辑，无需新求解器。
- 举着复合体时直接驱动其合并刚体，零件整体跟随，手感与单体一致。

### 24.7 外力 / 冲量命令 API（内核新增）

当前运行时只有 `set_velocity/set_position`，补齐**力累加器**：

```
world.apply_force(body, force, point)      // 世界系力（可偏心产生力矩）
world.apply_torque(body, torque)
world.apply_impulse(body, impulse, point)  // 瞬时冲量
world.apply_force_field(field)             // 见 24.8
```

- 力在子步预测前累加进 `f_ext`（对齐现有 `x* = x + v·dt + (f_ext/m)·dt²`），子步末清零。
- **命令队列**：所有施力经命令队列在子步边界统一应用（对齐 §12 异步管线），保证确定性与无锁写入。

### 24.8 力场系统深化（Fields）

§9 的"力场"升级为一等子系统，服务风扇/推进器/浮力/爆炸/涡流等功能部件：

```
Field
  shape: Aabb | Sphere | Cone | Sdf | Global   // 作用域
  falloff: Const | Linear | InvSquare          // 衰减
  effect: Force(dir) | Radial(push/pull) | Vortex(axis) | Buoyancy(fluid_density) | Drag
  mask: CollisionLayers                         // 只作用于匹配层
```

- 采样：broad-phase 用作用域 AABB 粗筛命中刚体/粒子，按衰减算每体受力，经力累加器注入（24.7）。
- **功能部件即力场**：风扇/推进器 = 挂一个 `Cone/Force` 场随刚体位姿移动；浮力 = 水体 AABB 内 `Buoyancy` 场（与 §8 流体耦合或解析近似二选一）。
- GPU：力场采样是 per-body 独立计算，天然并行，纳入 §11 GPU-Driven 管线，与粒子力场共用内核。

### 24.9 性能工程

- **形态自适应**：按连通分量零件数与"是否需要内部柔性"自动在关节网络 ↔ 复合体之间切换；大型稳定结构一律烘成单刚体，broad-phase/求解降到 O(单体)。
- **Aggregate 自碰撞屏蔽**：复合体内部零件对不进 narrow-phase，大幅削减接触对（Besiege/PhysX 的关键优化）。
- **增量重烘焙**：合并/拆分只重算受影响分量的质量属性与聚合 AABB，几何缓存复用；避免全图重建。
- **装配体睡眠**：复合体作为单一 Island 参与睡眠/唤醒（§3.5 warm-start、§10 LOD）；静止载具零成本。
- **GPU 批处理**：合并后的质量属性与力场采样可批量在 GPU 计算（§11），CPU 只发命令。
- **确定性**：合并顺序、并查集遍历、断裂结算全部走稳定排序（§18），合并/拆分可位一致复现。

### 24.10 易用性分层

- **5 行能跑**（概念 API，Bevy 层）：

```rust
// 把两个刚体焊成一个复合体
let asm = commands.spawn(Assembly::default()).id();
commands.entity(asm).insert(Weld::new(part_a, part_b));   // 焊接
// 柔性抓取：每帧写 anchor 即可
commands.entity(crate_entity).insert(GrabDrive::to(hand_pose).soft());
```

- **组件族**（ECS 原生、可 reflect/序列化/热重载）：`Assembly` / `Weld` / `BreakableJoint{max_force,max_torque}` / `GrabDrive` / `Field` / `AttachPoints`。
- **编辑器与调试**：锚点可视化编辑；复合体质心/主惯性轴/聚合 AABB 叠加显示；连接受力热力图与"接近断裂"高亮；装配图连通分量着色；吸附候选 ghost 预览。
- **数据驱动**：`BreakLimit`、`compliance`、`AttachPoint` 均可由几何资源/DataAsset 烘焙并热重载调参。

### 24.11 确定性与网络

- 装配图与复合体派生缓存全部纳入状态序列化（§18、扩展点 4），`SnapshotRing` 回滚时随刚体状态一并恢复/重建。
- 合并/拆分/断裂为确定性纯函数，rollback 重放时按相同命令序列产生相同拓扑，不引入 desync；断裂事件参与状态哈希校验。

### 24.12 落地映射与里程碑

- **Crate**：几何特征/对齐原语入 `prism_physics_geometry`；装配图/复合体合并/可断关节/力场/力累加器入 `prism_physics_core`（新增 `assembly`、`field`、`force` 模块）；GPU 力场采样入 `prism_physics_gpu`。
- **里程碑**（接 §22）：
  - **M10 动态装配**：力累加器 + `apply_force/impulse`、`FixedJoint`→通用可断关节、装配图 + 复合体合并/拆分、吸附几何原语、`GrabDrive`。验收：实时搭建≥200 零件载具稳定 60fps、焊点受力可断、抓取不穿墙。
  - **M11 力场与功能部件**：Field 子系统 + GPU 采样、风扇/推进器/浮力部件、与破坏系统（§13）断裂连锁。验收：可驱动自建载具行驶/飞行、受击解体。

### 24.13 验收 Demo（接 §19 基准套件）

- "究极手"沙盒：抓取/旋转/吸附对齐/焊接/拆解全链路，含 ghost 预览与断裂连锁。
- 载具搭建：轮（Revolute+Motor）+ 推进器（Field）+ 车身（复合体），可驾驶、碰撞受力解体后零件独立物理。
- 压力基准：500+ 零件结构的合并/拆分帧耗时、断裂风暴（批量焊点同帧崩开）确定性与性能看板。

---

## 25. 气动力系统（Aerodynamics：刚体 / 多体空气动力学）

> 定位：§9「气动/浮力」与 §24.8「力场」的深化落地。软体/布料气动已在 `prism_physics_core::soft::aero` 工业级实现（逐三角面阻力+升力、`WindField`/`AeroParams`、确定性湍流、Jacobi GPU 孪生）。本节补齐**刚体与多体**空气动力学，支撑究极手级自建载具的滑翔/飞行/推进/浮空。无 CFD、无 AI/ML。

### 25.1 对标与借鉴

| 项目 | 借鉴点 |
|---|---|
| Zelda TotK | 功能部件（风扇/螺旋桨/气球/火箭）= 解析力/浮力而非全 CFD；性价比最高的落地范式 |
| Besiege / Trailmakers / Juno(SR2) | 逐部件翼面升阻曲线 + 推进器 + 浮力，部件化、可组装的"玩具级可玩"气动 |
| MS Flight Simulator | 面元化（surface-element）气动 + 空间风场/热气流 + 地面效应，高拟真上界 |
| War Thunder (Gaijin DM) | 多翼面升阻 + 失速 + 操纵面，确定性联机下的气动 |
| KSP / FAR | 基于横截面的解析气动 + 稳定性参考 |
| UE5 Chaos / PhysX / Jolt | 刚体层通常只给各向同性 drag + 浮力，升力/推力交上层；本节在内核直接提供面元/部件级，形成差异化 |

设计原则：**解析优先、面元可选、场耦合封顶**——用最低成本得到"读起来对"的飞行手感，再按需提精度。

### 25.2 现状与差距

- ✅ 已有：软体/布料逐面气动（`soft::aero`）；刚体各向同性 `linear_damping` / `angular_damping`（`dynamics/integrator.rs`，半隐式 `v *= 1/(1+c·dt)`，只是速度打折，**非真实气动**）。
- ❌ 缺：刚体形阻（方向/面积相关）、翼面升力+攻角曲线+失速、推进反作用+反扭矩、浮力（空气/水对刚体）、附加质量、马格努斯、空间风场、下洗/滑流/尾流/地面效应。
- 本节把上述缺口补成可落地规格。

### 25.3 保真度分层（三档 LOD，可逐体选择）

```
L0 解析部件力  Thruster / Buoyancy / 各向异性 Drag           // 最省，TotK 范式
L1 面元气动    AeroSurface 面元集（升阻 + 攻角曲线 + 失速）    // Besiege/MSFS 范式
L2 场耦合气动  面元 + 空间 WindField 采样 + 下洗 + 地面效应     // 高拟真
```

- 每刚体一个 `AeroLod` 标签，随距离/重要度/速度自适应降级（接 §10 空间 LOD）。
- L0→L2 共享同一套系数定义（复用软体 `AeroParams` 的 drag/lift 语义），避免双份调参。

### 25.4 气动面元 AeroSurface（刚体形阻 + 翼面升力 + 失速）

数据结构（随刚体位姿刚性移动的局部面元）：

```
AeroSurface
  local_pos: Vec3        // 压心（局部系）
  normal:    Vec3        // 面法线（局部系）
  area:      Real
  chord:     Real        // 弦长，用于力矩/地面效应
  curve:     AeroCurve   // 攻角→(Cl,Cd)，解析或查表
  flap:      Option<ControlSurface>   // 操纵面偏转，改变有效攻角/弯度
```

每面元力（世界系）：

```
v_rel  = v_wind - (v_body + ω × r)          // 相对来流，r = 压心相对质心
q      = 0.5 · ρ · |v_rel|²                 // 动压
α      = angle(v_rel, surface_plane)        // 攻角
F_lift = q · area · Cl(α) · ê_lift          // ê_lift ⊥ v_rel
F_drag = q · area · Cd(α) · ê_drag          // ê_drag ∥ v_rel
τ      = r × (F_lift + F_drag)              // 偏心→力矩，经 24.7 施力
```

- **攻角曲线 `AeroCurve`**：小角薄翼解析 `Cl = 2π·sin α` 或查表；**失速**用一段式混合——过临界攻角 `α_s` 后向平板模型（`Cl→sin2α`、`Cd→1−cos2α`）C¹ 连续过渡，避免升力悬崖导致数值炸裂。
- **形阻退化**：无升力面（`Cl≡0`）即纯方向性阻力板；一组正交面元即可近似任意凸体的各向异性阻力（远比各向同性 damping 真实）。
- **自动面元**：可由碰撞体凸包面/包围盒六面烘焙默认阻力面元集；用户只在需要翼时手标机翼面元。

### 25.5 推进部件 Thruster（风扇 / 螺旋桨 / 喷气）

```
Thruster
  local_pose:      (pos, dir)                  // 推力作用点与方向（局部系）
  thrust:          Real | Curve(rpm|throttle)  // 定推力或曲线
  reaction_torque: Real                        // 反扭矩系数（单桨配平关键）
  slipstream:      Option<SlipstreamField>     // 向后喷出的风场（见 25.9/25.10）
```

- 推力 `F = thrust · dir_world`，偏心施于作用点产生俯仰/偏航力矩（24.7）。
- **反扭矩** `τ = −sign(spin)·reaction_torque·thrust`：真实单旋翼会让机身反转，需双桨反转或尾桨配平——飞行手感的灵魂。
- **滑流**：推进器向后生成一个 `Cone` 风场注入全局风场，可吹动下游布料/刚体/角色（25.10）。

### 25.6 浮力 Buoyancy（空气 / 水，解析 + 流体耦合二选一）

```
Buoyancy
  medium:             Air(ρ_air) | Water(ρ_water)
  volume_source:      ColliderVolume | ExplicitVolume
  center_of_buoyancy: Vec3     // 浮心（局部），与质心错位产生扶正力矩
```

- 解析式（默认、省）：`F_b = −ρ_medium · V_submerged · g`，施于浮心；入水体积由碰撞体与水面平面/高度场求交近似，支持部分浸没线性插值。
- 流体耦合式（高拟真、接 §8）：从 FLIP/APIC 水体读压力积分，与解析二选一。
- **空气浮力** = 气球/热气球：`V·(ρ_air−ρ_gas)·(−g)`，支撑 TotK 气球部件；热气球随温度调 `ρ_gas` 升降。
- 浮心高于质心 → 自动扶正（船/浮筒稳定），数值上是姿态恢复力矩，天然稳。

### 25.7 附加质量 Added Mass（稠密流体加速修正）

- 稠密介质中加速物体要带动周围流体，等效附加惯性 `m_add ≈ C_a·ρ·V`、`I_add` 同理。
- 实现：不显式求解流体，而对**浸没刚体**的有效逆质量/逆惯性做方向性缩放（`M_eff = M + M_add`），直接改 XPBD 预测与冲量响应。空气中 `ρ` 小可忽略，仅水下/高密介质启用。
- 效果：消除水下"过轻乱窜"，让桨/鳍推水"抓得住"；有效质量更大也更稳定。

### 25.8 旋转效应（马格努斯 / 旋转阻尼）

- **旋转气动阻尼**：`τ_damp = −c_rot·ρ·|ω|·ω`，各向同性 `angular_damping` 的物理版，让旋转的桨/物体自然减速。
- **马格努斯**：`F = C_m·ρ·V·(ω × v_rel)`，旋转体侧向升力（旋转桨叶、弧线球）；默认关，按需启用。

### 25.9 空间风场 WindField（可采样场）

把当前「单一全局风速 + 标量湍流」升级为**可采样空间场**（并入 §24.8 Field 子系统）：

```
WindField3D = Σ FieldSource
FieldSource:
  Uniform(dir, speed)              // 全局基风
  Gust(front, period, amp)         // 阵风 / 周期性
  Updraft(column|cone, speed)      // 上升气流（塔 / 温差）
  Vortex(axis, strength, radius)   // 龙卷 / 涡
  Noise(curl-noise, octaves)       // 无散度湍流
  Slipstream(from Thruster)        // 推进器尾流
sample(p) -> Vec3                  // 稀疏网格 / 解析叠加，供面元查来流
```

- **curl-noise** 做湍流：天然无散度，视觉自然且不引入人为吹胀（对齐 MSFS/影视流体做法）。
- 采样走稀疏网格 + 解析源叠加，GPU 友好（§11）。软体 `soft::aero` 的 `WindField` 升级为对本场的一次采样，软/刚体共用同一来流。

### 25.10 下洗 / 滑流 / 尾流与地面效应（部件间耦合）

- **滑流/下洗**：推进器把动量注入下游风场，下游面元/布料/角色采样到即被吹动（旋翼下压、喷流推人）。**单向注入**（部件→场→他体）即可，避免 O(n²) 两两耦合。
- **地面效应**：面元距地 `h < chord` 时增升减阻，`Cl·(1+k·chord/h)` 封顶；贴地飞行更稳（直升机/气垫范式）。
- **互扰封顶**：多部件只经共享风场间接耦合，不做显式涡格两两作用，复杂度 O(部件+采样) 而非 O(n²)。

### 25.11 积分与数值稳定

- **半隐式阻尼**：阻力/旋转阻尼用隐式形式（`v ← v/(1+c|v|dt)` 风格）无条件稳定，杜绝大风速下显式爆炸。
- **子步内施加**：气动力经 24.7 力累加器在每子步预测前注入 `f_ext`，与 XPBD 子步一致；高速/大面积体自动提子步数（接 §3.3、§10）。
- **CFL / 钳制**：单步速度增量按 `|Δv| ≤ κ·|v_rel|` 钳制；`q`、`Cl/Cd`、风场采样全程 sanitize（复用 `soft::aero::sanitize` 风格，非有限即零），NaN 安全。
- **失速平滑**：攻角曲线在 `α_s` 两侧 C¹ 连续混合，避免升力阶跃触发抖动。

### 25.12 性能工程

- **保真度 LOD（25.3）**：远/次要/低速体降到 L0 解析甚至纯 damping；仅主角载具/镜头内用 L1/L2。
- **SoA 批处理**：面元/推进器/浮力列按 SoA 布局整帧 SIMD/多线程扫（对齐 §1 DOD、§7 后端）；每面元计算彼此独立、无数据依赖。
- **GPU**：面元力、风场采样、浮力积分均 per-element 独立，纳入 §11 GPU-Driven，与粒子力场共用采样内核；CPU 只发命令。
- **装配体合并**：复合体（§24）合并后面元集随之合并到单刚体，施力对单体，broad-phase/求解不膨胀。
- **休眠**：静止/无风体不跑气动（接 §3.5、§24.9 装配体睡眠）；风场脏标记驱动按需唤醒。
- **缓存**：攻角曲线查表、面元几何、浮力体积预烘焙；风场稀疏网格增量更新。
- **预算**：气动 pass 目标 ≤ 单帧物理 10%；500 面元 + 32 推进器载具不超一个 island 的求解预算。

### 25.13 易用性分层

- **5 行能跑**（概念 API，Bevy 层）：

```rust
// 给刚体挂一片机翼 + 一个风扇即可飞
commands.entity(plane).insert(AeroSurface::wing(area, chord));   // 自动薄翼 + 失速曲线
commands.entity(plane).insert(Thruster::fan(thrust).with_reaction());
// 气球：一行浮空
commands.entity(ball).insert(Buoyancy::air(volume));
```

- **组件族**（ECS 原生、可 reflect/序列化/热重载）：`AeroSurface` / `AeroCurve` / `Thruster` / `Buoyancy` / `AddedMass` / `WindField3D` / `AeroLod`。
- **默认即对**：不标面元时，从碰撞体包围盒自动生成六面各向异性阻力板，使任何刚体"丢进风里会被吹"；标翼面才升级为升力面。
- **编辑器与调试**：面元压心/法线/面积 gizmo；来流矢量与攻角 HUD；升阻力/力矩箭头与失速高亮；风场流线/涡可视化；浮力线与浸没体积叠加。
- **数据驱动**：`AeroCurve`、`thrust`、`ρ_air/ρ_water`、`C_a/C_m` 均由 DataAsset 烘焙，可热重载调手感。

### 25.14 确定性与网络

- 所有气动力为确定性纯函数（稳定排序累加、sanitize、半隐式积分），纳入 §18 状态哈希；风场为参数化解析场，无随机态，rollback 重放位一致。
- 风场源、面元系数、推进器油门进状态序列化（扩展点 4），`SnapshotRing` 回滚随刚体恢复。
- curl-noise 湍流用确定性整数哈希种子（对齐 `soft::aero::turbulence_offset`），联机一致。

### 25.15 落地映射与里程碑（接 §22 / §24.12）

- **Crate**：面元/推进/浮力/附加质量/旋转效应入 `prism_physics_core` 新增 `aero` 模块（与 `soft::aero` 共享系数与 sanitize）；空间风场并入 `field` 模块；GPU 采样入 `prism_physics_gpu`。
- **M12 刚体气动**（接 M11 力场）：
  - 阶段一（L0）：各向异性 Drag 面元 + Thruster（含反扭矩）+ 解析 Buoyancy（空气/水）。验收：风扇车能跑、气球能浮、物体在风场里被按方向吹。
  - 阶段二（L1）：`AeroSurface` 升力 + `AeroCurve` 失速 + 操纵面。验收：自建滑翔翼/飞机可配平飞行、失速可复现不炸。
  - 阶段三（L2）：空间 `WindField3D`（阵风/上升气流/curl 湍流）+ 滑流/下洗 + 地面效应 + 附加质量。验收：上升气流滑翔、旋翼下压吹动布料/角色、水下推进有质感。

### 25.16 验收 Demo（接 §19 基准套件）

- 滑翔翼：`AeroSurface` 升阻 + 失速，俯冲拉起不发散；上升气流中可盘旋爬升。
- 螺旋桨飞行器：双桨反转配平、单桨不配平则自旋（反扭矩演示）；滑流吹动尾后布旗。
- 浮空与水上：气球升空、热气球调温升降；木筏浮水、浮心扶正、附加质量下的划桨质感。
- 压力/确定性：500 面元 + 风场采样帧耗时看板；联机 rollback 下气动位一致哈希校验。

---

## 26. 关节限位与角色布偶（Swing-Twist 锥限 / 柔韧阻尼 / 布偶稳定）

> 定位：补齐 §3.2 关节族缺口。当前 `SphericalJoint`（`joint/kind.rs`）只有点重合 `compliance`、**无任何角度限位**，无法表达肩/髋/颈等生物关节的解剖运动范围——布偶会"软面条"、过度扭转、穿模。本节给出 swing-twist 四元数分解锥限、软限位、关节摩擦/阻尼、位姿伺服 drive 与布偶整体稳定方案。纯 XPBD 角位置约束，无 CFD、无 AI/ML。

### 26.1 对标与借鉴

| 项目 | 借鉴点 |
|---|---|
| UE5 PhysicsAsset / PhAT | swing1/swing2 椭圆锥 + twist 独立限位 + soft limit（刚度/阻尼/恢复）；美术在编辑器里刷锥角 |
| PhysX D6 Joint | 6-DOF 可逐轴 Locked/Limited/Free + swing cone + twist 限 + drive；工业事实标准 |
| Jolt `SwingTwistConstraint` | 四元数 swing-twist 分解 + 锥限 + twist 限 + 马达，轻量确定性，最贴合本内核哲学 |
| Havok / Bullet cone-twist | 球窝锥限 + 扭限经典实现，广泛用于布偶 |
| NaturalMotion Euphoria | 主动布偶：关节伺服 drive + 肌肉张力，受击反应自然（手感灵魂） |
| GTA V / RDR2 (Euphoria) | 行为驱动主动布偶：踉跄、护头、抓扶、翻滚起身，被动布偶之上叠行为层 |
| Unity PuppetMaster / Active Ragdoll | 肌肉伺服 + 命中软化 + 主被动混合权重 α，手感调参范式 |
| Overgrowth / Rain World | 程序化姿态驱动，少骨骼也自然，低成本手感 |

设计原则：**球窝分解优先、软限位封顶、驱动可选**——用一个 `SwingTwistLimit` 统一表达锥/扭/软墙，默认被动布偶零配置可用，进阶再挂 drive 做主动反应。

### 26.2 现状与差距

- ✅ 已有：`RevoluteJoint`/`PrismaticJoint` 的 1-DOF `AngleLimit`/`LinearLimit` + `Motor`（位置域求解，`joint/motor.rs`）；`SphericalJoint` 点重合 compliance。
- ❌ 缺：球窝 swing 锥限（含各向异性椭圆锥）、twist 扭限、软限位（soft limit）、关节角摩擦/阻尼、关节 **drive（目标相对姿态伺服）**、布偶级自碰撞屏蔽与稳定策略、质量比鲁棒性。
- 本节把上述缺口补成可落地规格，复用既有 `AngleLimit`/`Motor` 语义，避免双份概念。

### 26.3 Swing-Twist 四元数分解

把关节相对姿态分解为"绕扭转轴的扭转" + "离轴的摆动"，避免欧拉角万向锁与顺序歧义：

```
q_rel  = q_a⁻¹ · q_b                      // 关节局部系下 B 相对 A 的姿态
twist 轴 a（取局部 x）：
  p        = (q_rel.xyz · a) · a           // 旋转矢量部分在扭转轴上的投影
  q_twist  = normalize(Quat(q_rel.w, p))   // 退化(w≈0,p≈0)时取单位四元数
  q_swing  = q_rel · q_twist⁻¹             // 剩余摆动分量（扭转轴→目标方向）
```

- `q_swing` 把扭转轴转到当前朝向：由其提取摆动角 `(θ_y, θ_z)`（扭转轴偏离局部 x 的两正交分量）。
- `q_twist` 的转角即扭转角 `φ`，直接喂给复用的 `AngleLimit`（twist 限）。
- 分解是纯代数（一次点积+四元数乘+归一化），无 `asin/atan` 热路径依赖，退化情形显式兜底。

### 26.4 数据结构 SwingTwistLimit

```
SwingTwistLimit
  twist_axis: Vec3            // 关节局部系扭转轴（归一化使用）
  swing_y:    Real            // 椭圆锥半角（绕局部 y），弧度
  swing_z:    Real            // 椭圆锥半角（绕局部 z），弧度；y≠z 即各向异性锥
  twist:      AngleLimit      // 扭转范围（复用现有类型）
  soft:       Option<SoftLimit>   // 软限位：超限区段的刚度/阻尼/恢复
  friction:   Real            // 关节角摩擦（抵抗自由摆动的恒力矩预算）

SoftLimit { compliance: Real, damping: Real, restitution: Real }
```

- **椭圆锥判定**：由 `q_swing` 得 `(θ_y, θ_z)`，违例条件 `(θ_y/swing_y)² + (θ_z/swing_z)² > 1`；超限时沿椭圆法向把摆动投影回锥面，得目标 `q_swing'`，再合成修正后的 `q_rel'`。
- **各向异性**：`swing_y≠swing_z` 表达"前后摆得多、左右摆得少"的真实关节；设 `swing_y=swing_z` 退化为圆锥。
- **锁死/自由**：半角取 `0` → 该向锁死；取 `π` → 自由，统一一个结构覆盖 D6 的 Locked/Limited/Free。

椭圆锥投影（swing 违例时把摆动回投到锥面）：

```
u = (θ_y / swing_y, θ_z / swing_z)        // 归一化摆动坐标
if |u| > 1:                               // 落在椭圆外 → 违例
  s         = 1 / |u|                      // 一阶等比缩回（视觉已足够）
  θ_y',θ_z' = s·θ_y, s·θ_z                 // 需精确时加一次 Newton 求椭圆最近点
  重建 q_swing'，C = 2·angle(q_swing, q_swing')   // 角约束误差，交 XPBD 投影
```
- 一阶等比缩回确定性、便宜；需物理精确最近点时加一次 Newton（椭圆点-距离），仍确定性。

### 26.5 作为 XPBD 角约束求解

- **顺序**：先解点重合（线约束，已有）→ 再解 swing 锥限（角约束）→ 再解 twist 限（1-DOF，复用 Revolute 角限逻辑）。同一子步内按图着色与接触/其它关节同批。
- **硬限位**：违例角位置差 `Δθ` 直接做 XPBD 角修正（`Δλ = -C/(w̃+α̃)`，`α̃=compliance/h²`），`compliance=0` 即硬墙。
- **软限位 SoftLimit**：超限区段用非零 `compliance` + 速度层 `damping`/`restitution`，形成"渐进软墙"，吸收命中冲击、避免布偶在限位处抖/弹飞（对标 PhAT soft constraint）。
- **关节摩擦**：对自由摆动施加有界恒力矩（`|impulse| ≤ friction·h`），让布偶静止时不乱飘、像有组织阻力。

### 26.6 关节 Drive（位姿伺服 / 主动布偶）

```
JointDrive
  target_rotation: Quat      // 目标相对姿态（局部系）
  mode:   Velocity | Position
  max_torque: Real           // 力矩预算（限幅，防注入无界能量）
  compliance: Real           // 0=刚性伺服
```

- 位置模：对 `q_rel→target_rotation` 做 SLERP 误差的 compliant 角修正；速度模：驱动相对角速度至目标。
- 预算语义与现有 `Motor` 一致（`per-sub-step impulse ≤ max_torque·h²`），实现上是球关节版 Motor。
- 用途：受击 recoil、肌肉张力、"半布偶（partial ragdoll）"——上半身被动、驱动维持站姿（对标 Euphoria）。
- **临界阻尼伺服**：等效 PD `τ = k_p·θ_err − k_d·ω`，映射为 `compliance=1/(k_p·h²)`、`damping↔k_d`；默认取临界阻尼 `k_d≈2√(k_p·I_eff)`，不过冲、不发散。
- **混合权重 α**：`0` 全被动、`1` 全驱动，逐关节/逐帧可调（见 §26.9），受击先软化再恢复。

### 26.7 布偶装配与稳定

- **RagdollBuilder**：输入骨骼层级 → 批量生成 `body + collider + SphericalJoint(+SwingTwistLimit)`；提供 humanoid 预设（肩/肘/髋/膝/颈/脊柱标准锥角表）。
- **自碰撞屏蔽**：相邻骨骼用 §24 `Aggregate` 屏蔽相邻自碰撞、保留远端（手打脸要碰、上臂贴肩不要互挤）。
- **质量比鲁棒性**：相邻骨骼逆质量差过大（手指 vs 躯干）易发散——对策：子步 + warm-start + soft limit；提供相邻质量比 clamp 建议（如 ≤10:1）并在 builder 里告警。
- **睡眠**：整条布偶归一个 island，用聚合动能阈值统一休眠/唤醒，防半截抖动。

### 26.8 人形解剖锥角预设（humanoid preset）

供 `RagdollBuilder::humanoid` 直接套用的默认范围（右手系，近似解剖活动度；可被 DataAsset 覆盖）：

| 关节 | swing_y | swing_z | twist | 说明 |
|---|---|---|---|---|
| 肩 Shoulder | 90° | 90° | ±60° | 大范围球窝，各向近似等 |
| 肘 Elbow | 0° | 0° | — | 退化为 1-DOF 铰链（0–150° 用 Revolute 限）|
| 腕 Wrist | 20° | 45° | ±15° | 掌屈/背伸大、桡尺偏小 |
| 髋 Hip | 70° | 45° | ±40° | 前摆大、外展中、旋转受限 |
| 膝 Knee | 0° | 0° | — | 1-DOF 铰链（0–150°，微量 twist 可选）|
| 踝 Ankle | 30° | 25° | ±10° | 跖屈/背屈 + 轻内外翻 |
| 颈 Neck | 40° | 40° | ±60° | 低头/侧倾中、转头大 |
| 腰/脊柱段 Spine | 30° | 30° | ±25° | 每段小角，多段累积成大弯 |

- 肘/膝给 swing=0 退化为铰链（twist 轴即屈伸轴），复用 §26.3 分解无特例分支。
- 多段脊柱把总活动度分摊到 3–5 段，单段小角更稳、整体更自然（对标 UE PhAT 脊柱分段）。

### 26.9 主动↔被动布偶混合与命中反应

被动锥限解决"不穿模"，主动混合解决"有生命感"（对标 Euphoria / PuppetMaster）：

- **混合权重 α（逐骨骼/逐帧）**：`α=0` 全被动倒地、`α=1` 全驱动贴动画；受击瞬间命中区域 `α→0` 吸冲击，之后数帧 `lerp` 回高 α 恢复控制。
- **命中反应**：按冲量大小/方向选择护头、踉跄、后仰等 drive 目标姿态，局部软化（降 `max_torque`）再回收。
- **起身（get-up）**：检测稳定支撑接触 → 匹配最近起身姿态 → `JointDrive` 把 `q_rel` 伺服到动画目标并提高 α，平滑交还动画系统。
- **分层掩码**：上半身主动（护头/抓扶）、下半身被动（倒地/打滑）可独立 α，支撑"半布偶"。
- **接口预留**：physics→anim 的 pose 反馈走集成层（§17）事件/组件，内核只暴露 `q_rel`/`drive`，不内嵌动画逻辑（保持引擎无关）。

### 26.10 数值、分层 LOD 与性能预算

- swing-twist 分解 + 锥投影为 O(1)/关节纯函数；提取角用一次 `atan2`/`asin`（可用多项式近似保确定性），非热路径瓶颈。
- 角约束与点约束同图着色批，无额外同步点；SoA 存 `SwingTwistLimit`，SIMD 批处理多关节。
- GPU：锥限/扭限/软限位为逐关节纯函数，可出 WGSL 孪生并入关节 kernel，CPU golden + 真机 parity（对齐 §7 验证范式）。
- 1000 关节布偶群：锥限额外成本相对点约束可忽略（<5%），给基准看板。

- **布偶 LOD（接 §10 空间 LOD）**：近距全锥限 + drive + 高子步；中距关 drive、仅保锥限、降子步；远距冻结为 kinematic/动画驱动或整岛休眠；切档带滞回防跳变。
- **性能预算（参考量级，单线程 CPU，具体以基准看板为准）**：

| 场景 | 规模 | 档位 | 估算 |
|---|---|---|---|
| 单主角布偶 | 16 关节 | 全锥限 + drive | ~0.05 ms/帧 |
| 群体倒地 | 100 具 ×16 | 仅锥限（无 drive）| ~0.6 ms/帧 |
| 远景群 | 500 具 | LOD 冻结/休眠 | ~0（唤醒才计）|

- GPU 批：锥限/软限位/drive 为逐关节纯函数，布偶群可整批 dispatch，CPU golden + 真机 parity。

### 26.11 易用性

- 五行起步：
```
let mut rag = RagdollBuilder::humanoid(&skeleton);
rag.joint(Hip).swing_twist(deg(45), deg(25), AngleLimit::symmetric(deg(30)));
rag.joint(Elbow).revolute_limit(deg(0), deg(150));
let handle = rag.build(&mut world);          // 被动布偶，零额外配置
handle.drive(Spine, pose_target, max_torque);// 可选：主动维持姿态
```
- 调试可视化：锥 gizmo（椭圆锥面）、twist 弧、当前摆/扭角、违例高亮、drive 目标姿态影子。
- 数据驱动：锥角/软限/摩擦/drive 预算走 DataAsset，热重载调布偶手感（接 §15）。

### 26.12 确定性与网络

- 全部为确定性纯函数（稳定分解、固定投影、有界 impulse），纳入 §18 状态哈希。
- `SwingTwistLimit`/`JointDrive` 进状态序列化（扩展点 4），随 `SnapshotRing` 回滚恢复。
- 无随机态；SLERP/锥投影用确定数值路径，联机 rollback 重放位一致。

### 26.13 里程碑 M13（接 §22）

- 阶段一：swing 椭圆锥限 + twist 限（硬墙）+ 关节摩擦。验收：人形布偶倒地姿态自然、关节不过度扭转/穿模。
- 阶段二：SoftLimit（软墙）+ JointDrive（位姿伺服）。验收：命中反应吸冲击不弹飞、半布偶可维持站姿。
- 阶段三：RagdollBuilder humanoid 预设 + Aggregate 自碰撞屏蔽 + 质量比告警 + GPU 孪生。验收：100 具布偶群 60fps、确定性哈希一致。

---

## 27. 降阶坐标铰接（Reduced-Coordinate Articulation：Featherstone）

> 定位：对标 PhysX Articulation / MuJoCo / Bullet `btMultiBody`。极大质量比、长铰接链（机械臂、起重机、角色强约束链、索道吊舱）在**最大坐标 XPBD** 下需要大量子步才稳、且有残余关节漂移。降阶坐标用关节自由度本身作广义坐标，**结构上无穿透漂移、极稳、少子步**。本节设计一个可选高保真档，与主干 XPBD 共存并经接触耦合。注意：现有 `reduced/` 模块是**软体模态子空间**（Pentland/模态分析），与本节刚体铰接**无关**，不要混淆。无 AI/ML。

### 27.1 对标与借鉴

| 项目 | 借鉴点 |
|---|---|
| PhysX 5 Articulation | reduced-coordinate + Featherstone ABA；工业级角色/机械臂事实标准 |
| MuJoCo | 最小坐标、顶级稳定与可微；关节空间直接表达限位/驱动，机器人仿真金标准 |
| Bullet `btMultiBody` | 开源 Featherstone 实现范式，接触用冲量投影到广义坐标 |
| DART / RBDL / Pinocchio | 刚体动力学算法库：ABA / CRBA / RNEA 的清晰公式来源 |
| Havok / Chaos | 工业引擎里铰接与最大坐标混合求解的工程取舍 |
| Featherstone 1983 / Mirtich | ABA O(n) 正向动力学与 spatial algebra 的原始出处与清晰推导 |
| Isaac Gym / PhysX GPU articulation | 数千同构铰接体 GPU 批量仿真范式（机器人学习环境事实标准），印证降阶的大规模并行性 |
| Brax / MuJoCo MJX | 可微 + 批量最小坐标仿真，印证广义坐标对控制 / 可微 / 并行的天然友好 |

设计原则：**默认最大坐标、强铰接链可选降阶、接触用 Jacobian 耦合、断裂退化**——不替换主干，只给强铰接子系统一个结构性更稳的档。

### 27.2 现状与差距

- ✅ 已有：最大坐标 XPBD 关节链（Fixed/Distance/Spherical/Revolute/Prismatic + Motor + 限位）；子步 + warm-start + island 并行。
- ❌ 缺：广义坐标表达、Featherstone ABA O(n) 正向动力学、关节空间惯性/偏置力、广义坐标直接限位/驱动、铰接体与接触/外部约束的 Jacobian 耦合。
- 适用面：长链 + 极大质量比 + 需精确可控（机械臂末端、角色操纵）时降阶坐标显著优于最大坐标。

### 27.3 模型：铰接拓扑

```
Articulation
  links: Vec<ArticulationLink>     // 以根为 0 的树（无环；成环退化为最大坐标 + 闭环约束）
  dof:   广义坐标 q（revolute:1, prismatic:1, spherical:3, fixed:0）
ArticulationLink
  parent: Option<usize>
  joint:  JointKind                // 复用既有关节族语义
  X_tree: SpatialTransform         // 父→子关节帧
  inertia: SpatialInertia (6×6)    // 空间惯性
state: q（广义位置）、q̇（广义速度）
```

### 27.4 Featherstone ABA（Articulated Body Algorithm，O(n)）

三遍扫描（spatial algebra，6D 旋量 `v=(ω,v_lin)`、力 `f=(τ,f_lin)`）：

```
① outward（根→叶）: 传播空间速度 v_i = X_i·v_{parent} + S_i·q̇_i，算速度相关偏置（科氏/离心）c_i
② inward（叶→根）:  聚合关节空间惯性 Iᴬ_i 与偏置力 pᴬ_i：
      Iᴬ_i = I_i + Σ_child  Xᵀ·(Iᴬ − Iᴬ S (SᵀIᴬ S)⁻¹ Sᵀ Iᴬ)·X
      pᴬ_i = p_i + Σ_child  Xᵀ·(pᴬ + Iᴬ c + Iᴬ S (SᵀIᴬ S)⁻¹ (τ − Sᵀ(Iᴬ c + pᴬ)))
③ outward（根→叶）: 解加速度 a_i 与关节加速度 q̈_i = (SᵀIᴬ S)⁻¹ (τ_i − Sᵀ(Iᴬ(X a_parent + c) + pᴬ))
```

- 复杂度 **O(n)**（n=链节数），关节空间小矩阵求逆（1×1~3×3），无全局 n×n 稠密解。
- 积分：半隐式对 `q̈→q̇→q`；关节限位/马达在广义坐标直接 clamp/驱动（天生比最大坐标干净——无需投影回流形）。

### 27.5 与 XPBD 主干耦合

- **接触/外部约束注入**：铰接体当成一个"岛内子系统"，接触冲量 `λ` 经关节 Jacobian 转置映射到广义坐标：`Δq̇ = M_gen⁻¹ Jᵀ λ`，`M_gen⁻¹` 由 ABA 隐含提供（可用 ABA 的"施加测试力求响应"得算子，无需显式构造）。
- **两档策略**：A) 纯降阶链 + 最大坐标接触层耦合（推荐，改动最小）；B) 全广义坐标接触（更稳但实现重）。默认 A。
- **限位/驱动**：swing-twist 锥限（§26）、马达在广义坐标里是对 `q`/`q̇` 的直接范围/目标约束，比最大坐标实现更简单稳定。

### 27.6 优势与取舍

- 优势：**无关节漂移**（约束结构内蕴）、**极大质量比稳定**、**少子步**（省 CPU）、广义坐标便于控制/IK/步态/未来可微分。
- 代价：实现复杂（spatial algebra）、接触需 Jacobian 耦合、不如最大坐标灵活"可断/可装配"。
- **断裂退化**：§24 可断关节/动态装配与降阶**互斥**——断裂或运行时装配触发该链"降解"为最大坐标子图并重划分 island（降阶适合固定拓扑强铰接，装配/破坏走最大坐标）。
- 策略：场景默认最大坐标；对标 `Articulated` 的实体（机械臂、角色主链）才转降阶。

### 27.7 关节空间阻尼与稳定

- **隐式关节阻尼**：对广义速度做隐式衰减 `q̇ ← q̇ / (1 + c·h)`（c 为每关节阻尼系数、h 步长），无条件稳定、不随步长发散；比最大坐标的显式阻尼更不易爆。
- **反射惯性 / armature**（对标 MuJoCo `armature`）：在关节空间惯性对角加一项 `SᵀIᴬS + d`，等效电机转子 / 减速箱反射惯性，显著改善大质量比与高增益马达下的数值刚性（抑制高频抖振），是真实机械臂稳定的关键细节。
- **零约束漂移**：铰接约束由广义坐标结构内蕴，无需 Baumgarte / 软化稳定项，省掉最大坐标常见的位置漂移修正及其带来的能量注入。
- **闭环处理**：树形拓扑成环（双臂合抱、四连杆机构）时，用**少量最大坐标闭合约束**（loop-closing constraint）把环打断为树 + 残余约束——环约束走 XPBD 投影，主体仍享 O(n) ABA。
- **奇异性**：spherical 自由度用**四元数广义坐标**（而非欧拉角）表达规避万向锁；q̇ 取机体角速度 3 维，积分走四元数指数映射，长链高速旋转不退化。

### 27.8 性能与预算

- 每链 O(n)、链节数据紧凑数组（cache 友好）；island 内多条独立链并行（bevy_tasks）。
- 关节空间小矩阵求逆用固定尺寸（1/3 维）展开，无堆分配；SoA 存 spatial 量。
- GPU：每链一个线程/工作组（链内串行、链间并行），适合大量同构链（如一群角色）；给 CPU golden + parity 预留。

- **降阶 vs 最大坐标（10 连杆、100:1 质量比同场景对拍）**：

| 维度 | 最大坐标 XPBD | 降阶坐标 Featherstone |
|---|---|---|
| 稳定所需子步 | 8–16 子步仍有残余漂移 | 2–4 子步、零漂移 |
| 关节漂移 | 需投影 / 软化，残留 | 结构内蕴，恒为零 |
| 每链成本 | O(n)（投影迭代常数大） | O(n)（扫描常数小、少子步） |
| 可断 / 运行时装配 | 原生支持 | 需降解为最大坐标子图 |
| 适用 | 默认、破坏 / 装配 / 松耦合 | 固定拓扑、强铰接、长链 |

- 结论：强铰接长链在降阶下用约 1/4 子步达到更低漂移，CPU 预算显著下降；代价是失去原生可断 / 装配，故按实体选择档位，而非全局替换。

### 27.9 易用性

- 复用 §26 `RagdollBuilder`，加 `.reduced()` 开关即把该布偶/机械臂编译为 `Articulation`；关节类型/限位/马达 API 不变（同一套 `JointKind`）。
- 调试：广义坐标 `q/q̇` 面板、每关节力矩、ABA 残差、与最大坐标并排对拍（同场景双解验证）。
- 默认不开；用户只在"长链 + 大质量比 + 要稳"时一行切换，符合"分层下钻"。

### 27.10 确定性与网络 + 里程碑 M14

- ABA 为确定性数值过程（固定扫描序、固定小矩阵求逆路径），`q/q̇` 进状态哈希与序列化，随 `SnapshotRing` 回滚。
- M14 阶段：① 单链 revolute/prismatic ABA + 广义坐标限位/马达（验收：长机械臂末端精确、零漂移）；② spherical 自由度 + swing-twist 广义锥限（验收：角色主链布偶降阶稳定）；③ 接触 Jacobian 耦合 + 断裂退化 + GPU 多链（验收：极大质量比吊臂抓重物不发散、断裂切回最大坐标无缝）。

---

## 28. 布料/壳体撕裂的拓扑级深化（顶点分裂 / 裂纹前沿 / 穿刺）

> 定位：深化 §4 布料。现状 `soft::damage::tearing` 已**工业级实现边级断裂**（张应变 `(len−rest)/rest` 超阈即移除 `DistanceConstraint`，O(edges)、确定性列序、GPU 孪生 `tear_flag`，附 `TearReport`）+ 塑性蠕变（`soft::damage::plasticity`）。本节把它从"移除约束"升级为**改变网格拓扑的真实撕裂**：顶点分裂生成裂口边界、裂纹沿前沿定向扩展、点驱动穿刺、各向异性织物、渲染网格缝合、撕裂后自碰撞/CCD 与 GPU 拓扑变更。无 AI/ML。

### 28.1 对标与借鉴

| 项目 | 借鉴点 |
|---|---|
| Houdini Vellum / Chaos Cloth | 约束断裂 = 从约束图移除失败边（本内核现状已对齐）；Vellum 另有 pop/weld 拓扑操作 |
| UE5 Chaos Destruction (Geometry Collection) | 几何集 + 连通分量拆分的拓扑级破坏范式，可借到布料裂口的连通维护 |
| NVIDIA Cloth / APEX | 可撕裂布料的顶点复制与裂口生成工程经验 |
| 断裂力学（主应力准则） | 裂纹沿**最大主应力垂直方向**扩展，决定分裂面朝向（经典、可解析，无 AI） |
| 影视布料（Marvelous/Qualoth） | 顶点分裂 + 重网格 + UV seam 生成的高保真撕裂流程 |
| O'Brien & Hodgins 1999 | 离散应力张量 + 主应力准则驱动脆性断裂的奠基论文，裂纹法向解析可算 |
| 预撕裂缝（prefractured seam） | 影视 / 游戏常用：沿预设缝预弱化，运行时沿缝裂，成本与美术双可控 |

设计原则：**边断裂保底、顶点分裂进阶、裂纹前沿控成本、拓扑变更确定性**——现状已能"扯坏"，本节让它"扯出口子、沿纹路裂、被尖物穿破"，并保持确定性与 GPU parity。

### 28.2 现状（已落地，勿重做）

- 边级断裂：`tear_flags`/`apply_tearing`（张应变超 `break_strain` 即移除边，列序确定性、保留存活边相对序以稳定图着色）；`tear_flag` 为 CPU/GPU 共享标量核。
- `TearReport{inspected, over_threshold, max_strain}`；`TearingParams` 带 `sanitized()`（NaN/负阈→不撕）。
- 塑性蠕变（rest-length 永久拉长）在 `soft::damage::plasticity`。
- **局限**：只删约束、**不改网格拓扑** → 渲染网格仍相连、不生成裂口边界；相邻弯曲/面积/tether 约束未同步；无裂纹方向控制与穿刺；GPU 侧只删标记、无拓扑重排。

### 28.3 顶点分裂（Vertex Split）——拓扑撕裂核心

- 当某顶点周围的断裂边构成一条"切割线"，把该顶点**复制为两个**，各自继承切割线一侧的三角形扇区，产生**新边界边 = 真实裂口**。
- 质量按继承面积在两副本间重分配（守恒）；法线/切线按新扇区重算。
- 数据：以半边（half-edge）或顶点↔三角邻接表增量维护；分裂只触碰局部一环邻域，O(deg)。
- **分裂面朝向**：由顶点处**最大主应力方向**定（应力张量特征分解），裂口垂直于最大拉伸方向，物理正确（断裂力学准则，解析可算）。

### 28.4 应力场估计与裂纹驱动（Stress Field）

对标 O'Brien & Hodgins 1999「Graphical Modeling and Animation of Brittle Fracture」的离散应力法，为分裂面朝向与起裂判据提供物理依据（纯解析、无 AI）：

- **单元应力**：每三角形由当前相对静止形状算面内形变梯度 `F`，Green 应变 `E = ½(FᵀF − I)`，经本构 `σ_tri = C : E` 得膜应力张量（2×2）。
- **顶点聚合**：把一环邻接三角的 `σ_tri` 按面积加权聚合到顶点，得顶点应力张量 `σ_v`（对称 2×2）。
- **主应力闭式解**：对 2×2 对称阵做**闭式**特征分解得主应力 `λ₁ ≥ λ₂` 与主方向 `e₁`（无需迭代、天生确定）。裂纹**法向取 e₁**（最大拉伸方向），裂口沿 `e₂` 延展——与 §28.3 分裂面朝向一致。
- **起裂判据**：`λ₁ > σ_break` 起裂；阈值带迟滞（起裂阈 > 止裂阈）避免逐帧开合抖动。
- **各向异性本构**：织物在经纬坐标系给方向性 `C` 与方向性 `σ_break`，顺纱更易裂，天然得到「沿纹路撕」。

### 28.5 裂纹前沿传播（Crack Front）

- 维护"前沿顶点"集合，只在前沿评估/推进，**O(前沿) 而非 O(网格)**，避免全网格扫描。
- 每帧限额推进 ≤N 步（成本上限），沿最大主应力方向择优扩展；阈值加迟滞（hysteresis）避免逐帧抖动开合。
- **各向异性织物**：经纬方向不同韧性 → 用织物坐标系下的方向性阈值张量，裂纹**沿纹理走**（真实布更易顺纱裂）。

### 28.6 穿刺 / 点驱动撕裂（Puncture）

- 尖锐碰撞体（刀刃/箭头/碎片）接触布面 → 接触点局部应力集中超阈 → 从接触点**起裂**并沿运动方向推进前沿。
- 由接触法向/切向冲量估局部应力，复用 §28.3 分裂；demo：刀划布、弹片穿旗、长矛破帐。

### 28.7 约束同步与自碰撞 / CCD 更新

- 分裂/断裂后，**同步**移除/重连受影响的弯曲、面积、tether 约束；图着色**增量**重建（只重排受影响色批，复用现有稳定序约定）。
- 新裂口边界参与**自碰撞**，撕裂口两侧防自穿（接已落地的 cloth self-collision + self-CCD）。
- 增量刷新局部 BVH/网格加速结构，避免全量重建。

### 28.8 渲染网格缝合与撕裂美术

- 物理裂口 → 渲染网格顶点复制 + **UV seam** 生成 + 法线重算，裂缝可见、材质不拉伸。
- 与集成层（§17）事件桥：`TearEvent{ kind: EdgeBreak|VertexSplit|Puncture, position, normal }`，供特效/音频/碎屑挂钩。

- **裂口美术**：锯齿边 vs 平滑切口由材质参数控制；沿裂口生成绒毛 / 拉丝 / 碎屑粒子（接 §17 事件）；双面材质 + 壳厚让裂口有厚度感而非纸片。
- **渐进式表现**：撕裂随前沿推进逐帧延展（非瞬间全开），配合音频（撕布声随前沿速度调制）与形变回弹，提升打击感与真实感。

### 28.9 GPU 拓扑变更

- 难点：GPU 上**动态拓扑**。方案：标记-扫描-重排 compact pass（对齐现有 `tear_flags` GPU 孪生做约束流 compaction）；顶点分裂走**双缓冲 + 原子分配**新顶点/边槽位，帧末 compact。
- 前沿推进在 GPU 上限额并行、确定性归约（固定序原子或分段扫描），保持 CPU golden + 真机 parity（§7 范式）。

### 28.10 性能预算与 LOD

- **成本结构**：撕裂成本 ∝ O(前沿顶点数)，与网格总规模解耦；`max_front_steps` 对每帧前沿推进削峰，避免「一刀全裂」造成帧尖峰。
- **内存**：顶点 / 边采用预分配内存池 + 上限（`max_split_verts`），超限则停止新分裂（降级为边断裂）而非扩容，保证无运行时大分配、内存占用确定。
- **撕裂 LOD**：近景全保真（顶点分裂 + 裂口 + 自碰撞 + 渲染缝合）；中景只分裂、免自碰撞 / 美术碎屑；远景退化为边断裂或整体消隐，省带宽与算力。
- **性能预算（1080p、单件布料参考）**：

| 场景 | 预算 |
|---|---|
| 旗帜 / 幕布随风撕裂（前沿小） | ~0.2 ms |
| 刀划布 / 弹片穿旗（前沿集中推进） | ~0.3 ms |
| 远景 / 背景布料（LOD 退化） | 可忽略 |

### 28.11 数值 / 性能 / 确定性

- 成本 O(前沿 + 受影响一环)，非 O(网格)；每帧撕裂步数设硬上限，avoid 尖峰。
- **确定性**：分裂按稳定 index 序执行，主应力特征分解走确定数值路径；状态哈希含**拓扑版本号**（顶点/边计数 + 内容）。
- **rollback**：拓扑快照（顶点/边版本 + 增量）随 `SnapshotRing`；回滚恢复撕裂前拓扑，联机位一致。

### 28.12 易用性 + 里程碑 M15

- 一个开关升级：`TearingParams{ break_strain, topological: true, anisotropy, max_front_steps }`；预设旗帜/帐篷/衣物/皮肉韧性。
- 调试可视化：前沿高亮、主应力方向、裂口边界、每帧撕裂步数看板。
- M15 阶段：① 顶点分裂 + 裂口边界 + 约束同步（验收：拉扯撕开真实口子、渲染可见）；② 裂纹前沿 + 各向异性 + 穿刺（验收：刀划布沿纹路裂、弹片穿旗）；③ 自碰撞/CCD 更新 + 渲染缝合 + GPU 拓扑变更 parity（验收：撕裂口不自穿、GPU/CPU 哈希一致）。

---

## 29. 术语表

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
| AssemblyGraph | 装配图，运行时可逆的刚体连接关系图，连通分量对应一个复合体 |
| CompoundBody | 复合刚体，由一个连通分量烘焙成的单一刚体（合并质量/质心/惯性 + 聚合碰撞体） |
| BreakLimit | 关节断裂阈值（最大力/力矩，可选疲劳塑性） |
| Aggregate | 聚合体，屏蔽内部零件自碰撞并以单代理参与 broad-phase 的优化（借鉴 PhysX） |
| GrabDrive | 柔性抓取驱动约束，把刚体经柔度+冲量钳制拉向运动学手点 |
| AttachPoint | 吸附锚点，随几何资源烘焙的位置+法线特征，用于规整拼接 |
| AeroSurface | 气动面元，随刚体刚性移动的升/阻力作用面（含攻角曲线 + 失速） |
| AeroCurve | 攻角→(Cl,Cd) 升阻曲线（薄翼解析/查表 + 失速平滑） |
| Thruster | 推进部件（风扇/螺旋桨/喷气），含反扭矩与滑流 |
| Buoyancy | 浮力（空气/水），解析或流体耦合，浮心扶正 |
| AddedMass | 附加质量，稠密流体中加速物体的等效附加惯性 |
| WindField3D | 可采样空间风场（基风/阵风/上升气流/涡/curl 湍流/滑流叠加） |
| Magnus | 马格努斯效应，旋转体侧向升力 |
| AeroLod | 气动保真度分层（L0 解析 / L1 面元 / L2 场耦合） |
| SwingTwistLimit | 球关节 swing-twist 分解锥限：椭圆摆动锥 + 扭转限 + 软限位 + 关节摩擦 |
| SoftLimit | 软限位，超限区段用非零柔度+阻尼+恢复形成渐进软墙，吸冲击不弹飞 |
| JointDrive | 关节位姿伺服，驱动相对姿态至目标（主动布偶/受击反应），力矩预算限幅 |
| RagdollBuilder | 布偶装配器，由骨骼批量生成 body+collider+球关节(+锥限)，含 humanoid 预设 |
| Articulation | 降阶（广义/最小）坐标铰接体，以关节自由度为状态的刚体树 |
| Featherstone/ABA | Articulated Body Algorithm，O(n) 正向铰接动力学三遍扫描 |
| 广义坐标 q/q̇ | 铰接体的关节空间位置/速度，降阶求解的状态量 |
| VertexSplit | 顶点分裂，撕裂时复制顶点生成裂口边界，改变网格拓扑 |
| CrackFront | 裂纹前沿，只在前沿按主应力方向定向扩展，O(前沿) 控成本 |
| TearEvent | 撕裂事件（边断/顶点分裂/穿刺），桥接渲染/特效/音频 |
| 人形锥角预设 | 肩/肘/腕/髋/膝/踝/颈/脊的解剖摆幅表，开箱即得合理布偶限位 |
| PoweredRagdoll | 主动布偶：临界阻尼 PD 伺服跟踪动画姿态，α 混合主动/被动 |
| 反射惯性 / armature | 关节空间惯性对角附加项，等效电机转子惯性，稳定高增益马达 |
| 应力张量 / 主应力 | 顶点膜应力 σ_v（对称 2×2），特征分解得主应力与裂纹法向 |

---

*本文档为 Prism Physics 设计规格 v0.6，不含 AI/ML 内容；M0–M8 内核与 GPU 后端已进入编码，并以 CPU golden + 真机 GPU parity 双重验证；§24 动态装配（M10/M11）、§25 刚体/多体气动（M12）、§26 关节锥限与布偶（M13）、§27 降阶坐标铰接（M14）、§28 布料拓扑撕裂（M15）为深化设计规划项。布料边级撕裂与塑性（`soft::damage`）已落地，§28 为其拓扑级深化。v0.6 深化：§26 人形解剖锥角预设、主动↔被动布偶混合与命中反应、分层 LOD 预算；§27 关节空间阻尼 / 反射惯性（armature）/ 闭环与奇异处理、降阶 vs 最大坐标对拍；§28 离散应力场（O'Brien & Hodgins）驱动裂纹、撕裂美术与渐进表现、撕裂 LOD 与性能预算。*
