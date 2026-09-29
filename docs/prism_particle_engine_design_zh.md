# Prism Ember 次世代粒子引擎设计方案

> 面向 Prism（Bevy fork）的数据导向、GPU-first、图编译式高性能粒子/特效引擎设计。
> 借鉴 UE5 Niagara、Unity VFX Graph（HDRP）、Houdini POPs/DOPs、EmberGen、PopcornFX、Frostbite/Destiny GPU 粒子，取长补短。
> **着色一等公民**：粒子渲染复用共享材质系统的 closure/shading-model，PBR / NPR / 自定义 / 混合四者皆为一等公民，可达顶级次世代 AAA 质量。
> 本文档为设计规格，尚未进入编码；采用纯经典数值路线，不含任何 AI/ML 内容。

- 版本: v0.2（设计阶段，未进入编码；本版重点补齐 PBR/NPR/混合/自定义着色、高级仿真、体积光照、时序集成、性能与效果验收）
- 范围: 一步到位（含 Simulation Stages / 网格流体 / 光追碰撞受光 / 四等公民着色）
- 适用引擎: Prism / Bevy ECS 生态
- 关键依赖: bevy_ecs（并行 ECS）、bevy_render（wgpu/WESL、render graph、render_phase、pipeline_cache）、bevy_math（glam SIMD）、bevy_tasks（任务系统）、bevy_shader（WESL/naga_oil）、bevy_solari（实时光追，可选）、bevy_light（聚类光照）、bevy_core_pipeline（prepass/OIT/motion vector）、bevy_diagnostic（诊断）
- 复用的现有设施: render_resource::{storage_buffer, buffer_vec, pipeline_cache, pipeline_specializer}、batching::gpu_preprocessing（indirect）、meshlet（indirect draw 经验）、occlusion_culling（HZB）、gpu_readback（GPU→CPU）、prepass（depth/normal/motion）、共享材质 closure 与 OIT
- 关联文档: `prism_material_pipeline_design_zh.md`（材质与着色一等公民契约）、`prism_aaa_advanced_features_zh.md`（PBR/NPR/混合高级特性分册）、`prism_physics_design_zh.md`（统一 XPBD/VBD 求解原语）

---

## 目录
1. 设计哲学与目标
2. 业界参考与采纳映射
3. 分层架构
4. 概念模型：System / Emitter / Stage / Renderer
5. 属性系统与 GPU 数据布局
6. 节点图与编译器：Graph → IR → WESL
7. Simulation Stages：通用迭代求解框架
8. Module / Force / DataInterface 插件体系
9. GPU 仿真管线（每帧全链路）
10. 高级仿真特性（XPBD/VBD、不可压缩流体、燃烧、涡量、群体）
11. 内存与缓冲池管理
12. 排序与透明：接入共享 OIT
13. 剔除与包围盒
14. 事件系统与数据通道
15. 渲染器矩阵
16. 粒子着色与材质集成：PBR/NPR/自定义/混合四等公民
17. PBR 粒子与体积着色
18. NPR 粒子与风格化着色
19. 混合与自定义着色
20. 体积光照与阴影（6-way / deep shadow / phase）
21. Motion Vector 与时序集成（TAA / 上采样）
22. 光追碰撞与受光（Solari）
23. GPU→CPU 反馈
24. CPU / GPU 双模拟
25. 数值积分与稳定性
26. 模拟空间与大世界
27. 属性压缩与带宽优化
28. 伸缩性与平台档案
29. 确定性与网络
30. 资产格式与作者工作流
31. Bevy 集成层：API 与渲染图接线
32. 性能预算与效果验收
33. 质量与可信度基础设施
34. Crate 拆分与落地形态
35. 路线图
36. 关键扩展点清单
37. 开放问题
38. 术语表

---

## 1. 设计哲学与目标

对标商用旗舰特效引擎，确立六条铁律：

- **GPU-first, 零回读**：默认全流程 GPU（compute + indirect draw），百万级粒子不需要 CPU 逐帧回读。CPU 只提交少量 uniform 与 dispatch。
- **数据驱动 + ECS 原生**：Emitter 实例是 ECS 组件，System 是 Asset，完全融入 Prism 的 extract → prepare → queue 渲染管线。
- **图编译而非解释**：节点图在加载期编译成少量 WESL compute kernel，运行时零解释开销、可 specialize、可异步预热。这是相对 Niagara 运行时 VM 的关键差异化。
- **属性按需分配（Attribute-on-demand）**：图里没用到的属性不占显存，SoA 布局对齐带宽。
- **着色四等公民**：粒子不是"只会 additive 混合的贴片"。渲染阶段复用共享材质系统的 shading closure，**PBR / NPR / 自定义 / 混合**同为一等公民；风格化 VFX 与物理烟火在同一架构下皆可达 AAA 水准（§16-19）。
- **可扩展**：Module / Force / Renderer / DataInterface / Stage / ShadingModel 均为 trait 插件点，第三方可注册代码片段而无需改内核。

设计取舍总表：

| 维度 | Ember 选择 | 理由 |
|---|---|---|
| 执行位置 | GPU 默认 + CPU 可选回退 | 规模优先，兼容无 compute 平台 |
| 图执行 | 编译到 WESL kernel | 零解释、可 specialize、复用 pipeline_cache |
| 内存布局 | SoA + 按需分配 + 位宽量化 | 带宽是百万粒子瓶颈 |
| 分配策略 | free-list + 可选 compaction | 避免每帧全数组压缩 |
| 计数/派发 | GPU indirect dispatch/draw | 零回读、无 CPU 同步气泡 |
| 求解模型 | Spawn/Update + Simulation Stages + XPBD/VBD | 从粒子扩展到网格/邻域/约束求解器 |
| 着色模型 | 复用共享材质 closure（PBR/NPR/自定义/混合） | 四等公民、与主渲染一致、避免重写 |
| 透明 | 接入共享 OIT，不自建 | 与场景其他半透明统一排序/合成 |
| 时序 | 输出 motion vector | 与 TAA / 时序上采样正确耦合 |
| 图模型 | 先"堆叠"后"任意 DAG" | 易用先行，灵活演进 |
| 碰撞 | depth/SDF/光追 三档 | 精度与硬件兼容分级 |
| 确定性 | hash RNG + 固定 dt 可选 | 回放/网络同步 |

---

## 2. 业界参考与采纳映射

| 来源引擎 | 关键机制 | Ember 采纳映射 |
|---|---|---|
| UE **Niagara** | Emitter/System、Module 堆叠、Simulation Stages、Data Interfaces、Data Channels、Scalability、Lightweight Emitters | §4/§7/§8/§14/§28 |
| Unity **VFX Graph (HDRP)** | Context/Block、图→HLSL 代码生成、GPU Events、Strips、Attribute Map/Point Cache、Sample Mesh/SDF/Texture、**六向光照贴图烟雾、Shader Graph 输出（PBR/自定义 lit）** | §6/§14/§8/§16/§17/§20/§30 |
| Houdini **POPs/DOPs** | 属性中心、microsolver 组合、VEX wrangle、VDB/SDF 碰撞、Groups、Pyro（燃烧/涡量） | §7/§8/§5/§10 |
| **EmberGen** | 实时 GPU 体素流体（烟/火/爆炸）、涡量约束、六向光照导出 | §10/§20 |
| **PopcornFX** | Evolver、平台档案、CPU+GPU 混合、属性采样器 | §28/§24 |
| Frostbite / Destiny GPU 粒子 | persistent-thread、GPU 排序、indirect、tile 化、**体积雾/粒子光照注入** | §9/§12/§20 |
| Guerrilla（Decima）体积 | 六向光照 rig、体积自阴影、phase function | §20 |
| GPU Gems / 实时体积 | curl-noise、软粒子、深度碰撞、vector field、deep shadow maps | §8/§20 |
| **风格化标杆**（原神 / 崩坏3 / 明日方舟 / Guilty Gear Xrd / Arc System Works） | 渐变色阶、描边、程序化 SDF 形状、速度线、能量环、天使环高光 | §18 |
| UE **Lumen / VSM / Nanite / TSR** | 粒子接入 GI / 虚拟阴影 / 时序上采样的共享基底服务 | §17/§20/§21/§22 |

**次世代差异化**（相对 Niagara/VFX Graph 的组合优势）：
- 用 `bevy_solari` 光追做**粒子碰撞与受光/软阴影**（§22）。
- 用 `occlusion_culling` HZB 做**粒子遮挡剔除**（§13）。
- 用 `gpu_readback` 做**GPU→gameplay 事件反馈**（§23）。
- **粒子着色复用共享材质 closure**，PBR/NPR/自定义/混合四等公民（§16），且天然继承基底高级服务：聚类+RT 光照、虚拟阴影、体积 GI、时序上采样——这是把 VFX 从"贴片时代"抬到"次世代 AAA"的关键。

---

## 3. 分层架构

```
Layer 7  作者层 Authoring   图编辑器 / 模板继承 / 热重载 / 点缓存导入 / 材质预览
Layer 6  着色层 Shading     ShadingModel（PBR/NPR/自定义/混合）→ 共享材质 closure / OIT
Layer 5  编译层 Compiler     节点图 → IR → 优化 → WESL codegen + 资源布局
Layer 4  ECS 集成层          EmberSystem Asset + Instance Component + Plugin
Layer 3  仿真调度层          extract → prepare → simulate(compute 子图) → queue
Layer 2  GPU 执行层          缓冲池 / free-list / compute passes / indirect / 排序 / 剔除
Layer 1  渲染层 Renderers    Sprite / Mesh / Ribbon / Light / Decal / Volume
```

拟新增 crate：

| Crate | 职责 |
|---|---|
| `bevy_ember` | 用户 API、Asset、Instance Component、Module/Force/DataInterface/ShadingModel trait、CPU 求解器 |
| `bevy_ember_compiler` | 图 → IR → WESL codegen、属性活跃性分析、着色特化 key、pipeline specialization key |
| `bevy_ember_render` | extract/prepare/queue、渲染图节点、GPU 缓冲、WESL kernel、indirect draw、OIT/motion vector 接线 |
| `bevy_ember_editor`（可选） | 基于 bevy_feathers/bevy_ui 的图编辑器与预览 |

**边界声明（避免重复造轮子）**：Ember 只负责"生几何 + 跑仿真 + 提交实例"。**着色走共享材质 closure、透明走共享 OIT、光照/阴影/GI/上采样走共享基底服务**（见 §16 与关联文档）。Ember 不重写 BRDF、不重写 OIT、不重写虚拟阴影，只作为这些服务的消费者与实例提供者。

---

## 4. 概念模型：System / Emitter / Stage / Renderer

三层结构，语义收紧自 Niagara：

```
EmberSystem  (资产: 一整个特效, 如"爆炸")
 └─ Emitter[]  (一类粒子: 火焰芯 / 火星 / 烟)
     ├─ Attributes 布局 (position, velocity, color, age, ...)
     ├─ Stages[] (有序执行阶段, 见 §7)
     ├─ Events / Data Channels (见 §14)
     ├─ ShadingModel (PBR/NPR/自定义/混合, 见 §16)
     └─ Renderers[] (一个 emitter 可挂多个渲染器)
```

一个 System 内**不同 Emitter 可用不同 ShadingModel**（火焰 PBR + 卡通烟 NPR），这正是"混合引擎"在特效层的体现（§19）。

**Stage 频率语义**：

| Stage 类别 | 频率 | 典型用途 |
|---|---|---|
| EmitterSpawn | 系统生成 1 次 | 初始化 emitter 级参数 |
| EmitterUpdate | 每帧 1 次（单线程） | 计算本帧 spawn 数、发射器动画、写 indirect args |
| ParticleSpawn | 每新粒子 1 次 | 初始位置/速度/生命/颜色 |
| ParticleUpdate | 每存活粒子每帧 1 次 | 力/积分/碰撞/曲线/事件 |
| SimulationStage | 见 §7 | 网格流体 / 邻域 / 约束迭代 / 事件消费 |

---

## 5. 属性系统与 GPU 数据布局

### 5.1 属性系统
- 属性 = `(name, type, semantic)`。内置语义：Position / Velocity / Age / Lifetime / Color / Size / Rotation / Scale / Alive / ParticleId / RibbonId / SortKey / **Normal / Tangent / Emissive / Roughness / Metallic / MaterialId / ShadingParams**，也支持自定义属性。
- **SoA 布局**：每个属性一段 storage buffer，`buf[particle_index]`，带宽友好。
- **按需分配**：编译器分析图，仅为被读写的属性分配缓冲；未使用语义零占用。着色需要的属性（法线/粗糙度等）仅在对应 ShadingModel 启用时分配。
- **原地 vs ping-pong**：默认原地更新；仅"读旧写新"的属性做双缓冲，由编译器自动决定，最小化显存。

### 5.2 粒子池布局（per Emitter, GPU）
```
ParticlePool
├─ attribute buffers   : SoA, 容量 = capacity（编译期上限）
├─ alive_list  (u32[])  : 存活索引，紧凑
├─ free_list   (u32 栈) : 空闲槽栈
├─ counters (atomic)    : alive_count / spawn_count / dead_count
├─ indirect_dispatch    : workgroup 数（由 alive_count 推导）
├─ indirect_draw        : 实例数（渲染 pass 填）
├─ prev_transform       : 上帧位置/矩阵（motion vector, §21）
└─ bounds (AABB)        : GPU reduction 或固定 bounds
```

### 5.3 CPU 侧组件
```
EmberSystem(Handle<EmberSystemAsset>)   // 引用编译后的系统
EmberSystemInstance {
    transform, playback(time/rate/loop/seed),
    emitter_states[], user_params(exposed 参数),
    sim_space(Local/World/Hybrid), bounds_mode,
    lod_bias, budget_override, quality_override,
    shading_override,   // 运行期切 PBR/NPR/混合档
}
```

---

## 6. 节点图与编译器：Graph → IR → WESL

### 6.1 为什么编译
Niagara 运行时 VM 有解释开销；VFX Graph 走"图 → HLSL codegen"。Ember 走 **图 → IR → WESL kernel**，配合 Prism 现有 `pipeline_cache` + `pipeline_specializer` + naga_oil，做到零解释、可特化、异步预热。着色 kernel 同样由图/材质编译产出（§16）。

### 6.2 编译流水线
```
Graph(.ember)
  → 语义校验     (类型 / 属性读写 / 环路检测 / ShadingModel 兼容)
  → IR (SSA-like, 属性读写标注)
  → 优化         (常量折叠 / 死代码消除 / 属性活跃性 → 缓冲裁剪 / ping-pong 决策)
  → codegen      (每个 Stage 生成一个 WESL compute entry, 内联所有 Module;
                   每个 Renderer×ShadingModel 生成 draw kernel, 内联 closure)
  → 特化 key      (属性布局 / 宏开关 / 着色模型 / 平台档), 交给 pipeline_cache
```
产物：`spawn.wesl` / `update.wesl` / 各 Stage kernel / 各 Renderer draw kernel（含 shading closure）/ 缓冲布局 / bind group layout / indirect 参数 kernel。

### 6.3 表达式与绑定
- **表达式语言**：仿 VEX/HLSL 的小型表达式（`@position += @velocity * dt`），编译进 kernel。
- **曲线/渐变**：Curve/Gradient 烘焙成 1D 纹理或 SSBO LUT，kernel 采样避免分支。NPR 色阶 ramp 亦复用此机制（§18）。
- **Property Binder / Data Interface**：CPU 把场景数据（相机/光/时间/音频/骨骼/SDF）写入 uniform/storage，供图只读采样（见 §8）。

---

## 7. Simulation Stages：通用迭代求解框架

只有 spawn/update 不足以做流体、邻域、绳索。引入 Niagara Simulation Stages + Houdini microsolver 的通用迭代模型。每个 Stage 声明**迭代域(Iteration Domain)**：

| 迭代域 | dispatch 依据 | 用途 |
|---|---|---|
| `PerParticle` | alive_count（indirect） | 常规力/积分/生命 |
| `PerNeighborCell` | 空间哈希网格单元数 | 建网格 / 邻域查询（flocking、SPH-lite、碰撞对） |
| `PerGridVoxel` | 3D 网格分辨率 | 体素流体（烟/火）：平流、扩散、压力投影 |
| `PerConstraint` | 约束批次数 | XPBD/VBD 约束迭代（绳索/布片/软体，见 §10） |
| `PerEvent` | 事件缓冲计数 | 从事件生成粒子 |
| `Custom(count)` | 用户指定 | 通用 compute |

- **迭代次数**：Stage 可设 `iterations = N`（如流体 Jacobi 压力投影多次、XPBD 约束多次），编译器展开成多次 dispatch。
- **网格属性**：属性除 per-particle SoA 外，可声明为 **grid attribute（3D 纹理/SSBO）**，Stage 间 ping-pong。烟火 = 粒子采样速度场 + 速度场自身用网格 Stage 求解。
- **邻域网格**：build（散列插入）→ query（遍历 27 邻域），供碰撞对/群体/SPH。
- **读写作用域**：每 Stage 声明读/写集（粒子/网格/全局），编译器据此插入最少 barrier 与 ping-pong。

这样火花(per-particle)、烟(网格流体)、群体(邻域)、绳索/布(约束迭代)统一在一个框架里。

---

## 8. Module / Force / DataInterface 插件体系

### 8.1 统一插件 trait
```
trait EmberModule {
    fn stage(&self) -> StageKind;
    fn required_attributes(&self) -> &[AttrSemantic];  // 用于按需分配
    fn resources(&self) -> &[ResourceDecl];            // uniform/curve/texture
    fn emit_wesl(&self, ctx: &mut CodegenCtx);         // 生成内联代码
}
trait DataInterface {
    fn bindings(&self) -> &[ResourceDecl];
    fn emit_wesl_functions(&self, ctx: &mut CodegenCtx); // 生成 sample_* 函数
}
```

### 8.2 内置 Module 库（首发）
- **生成**：Rate / Burst / Distribution（点/球/盒/圆锥/网格表面/骨骼/SDF 表面采样）
- **初始化**：Set Position/Velocity/Color/Size/Life/Rotation、Random Range、Inherit Velocity
- **力**：Gravity、Drag、Vortex、Point/Line Attractor、Curl Noise、Wind、Vector Field(3D 纹理)
- **碰撞**：Plane、Depth-buffer、SDF、（可选）光追（§22）、反弹/摩擦
- **约束**：Distance/Bend/Volume（XPBD 原语，对接 §10 与 physics_core）
- **生命周期**：Age/Kill、Color/Size/Velocity-over-life（曲线 LUT）
- **事件**：Emit on Death/Collision/Spawn/Condition → 触发其他 emitter（§14）

### 8.3 DataInterface / 采样器清单
- **几何**：Sample Mesh（表面/顶点/三角形重心插值）、Skeletal Mesh（蒙皮位置/速度）、Spline/Curve
- **场**：Texture2D/3D（vector field / 密度）、SDF/VDB（碰撞与吸附，法线=梯度）、Point Cache（烘焙点云）
- **场景**：Camera、Depth/Normal（碰撞/软粒子）、Light/Clustered（bevy_light）、Ray-Trace Scene（bevy_solari）、GI Probe / Irradiance（受光，§17）
- **信号**：Audio 频谱、Curve/Gradient LUT、Global（time/dt/frame/随机流）
- **交互**：UserParam（Property Binder）、自建 Grid/Neighbor 结构

---

## 9. GPU 仿真管线（每帧全链路）

渲染图中新增一条 compute 子图，置于主渲染之前。每个 Emitter 一帧：

```
1. EmitterUpdate     1 线程/emitter：算 spawn_count，写 indirect args
2. Spawn/Emit        dispatch spawn_count：从 free_list 取槽，跑 ParticleSpawn kernel
3. SimulationStages  按 §7 顺序执行（含 ParticleUpdate、网格、邻域、约束、事件消费）
4. Event Scatter     写事件 ring buffer / Data Channel（§14）
5. Compaction(可选)  prefix-sum 重建 alive_list（碎片率高时）
6. Bounds            GPU reduction 求 AABB（或固定 bounds）
7. Cull              视锥 / 距离 / HZB 遮挡（§13），填可见实例
8. Sort(条件)        半透明按视深排序（§12）
9. Fill Draw Args    写 indirect_draw 实例数与实例缓冲（含 prev_transform for MV）
10. Render Draw       indirect draw，运行选定 ShadingModel closure（§14-19）
```

**关键性能技术**
- **Indirect dispatch/draw**：workgroup 数与实例数全 GPU 侧算，CPU 不知道粒子数，零回读、零同步气泡（复用 `gpu_preprocessing`/`meshlet` 经验）。
- **Async compute**：仿真与上一帧后处理/阴影 overlap（后端支持独立 compute 队列时）。
- **Persistent-thread（可选）**：固定 workgroup 数 + grid-stride 循环，减少 dispatch 数与 tail 效应。
- **Barrier 最小化**：编译器按 Stage 读写集合并同类 Stage、插最少屏障。

---

## 10. 高级仿真特性（XPBD/VBD、不可压缩流体、燃烧、涡量、群体）

对标 Houdini Pyro / EmberGen / Niagara，补齐"看起来次世代"的仿真质感。**约束求解原语对齐 `prism_physics_core` 的统一 XPBD/VBD**，粒子系统只产几何/约束数据、不重写求解器。

- **XPBD / VBD 约束**：距离/弯曲/体积/各向异性约束，供绳索、飘带布片、软体碎片、可撕裂网格。子步 compliance、`PerConstraint` 迭代域批处理（图着色避免写冲突）。VBD 作为高稳定档插槽。
- **不可压缩流体**：`PerGridVoxel` 平流（semi-Lagrangian / MacCormack 去数值耗散）→ 扩散 → **压力投影**（Jacobi 近似档 / multigrid 精确档）→ 速度回写。粒子从速度场平流形成烟火。开放问题 §37 给出近似先行、精确演进的路线。
- **燃烧模型（combustion）**：温度/燃料/烟三通道耦合，温度驱动浮力与 blackbody 自发光（§17），燃料消耗生成烟与热浪扭曲。
- **涡量约束（vorticity confinement）**：向速度场注入涡量补偿数值耗散，保住烟的卷曲细节——次世代烟火的关键质感来源。
- **群体 / Boids**：邻域网格支撑分离/对齐/聚合三规则 + 目标追踪，做鸟群/鱼群/萤火。
- **大规模碎裂**：碎块粒子挂 mesh 渲染器 + XPBD 刚体近似 + 光追/SDF 碰撞，做爆炸崩塌。
- **两向耦合（可选高档）**：流体/布/碎块对角色施加反作用力，经 §23 少量回读或 GPU 侧共享缓冲对接 physics_core。

---

## 11. 内存与缓冲池管理

- **BufferPool**：按容量分档复用 storage buffer，减少分配抖动；系统销毁归还。
- **Free-list 分配**：spawn kernel 从 dead 栈弹槽；update kernel 将死亡粒子压回。避免每帧全数组压缩。
- **Compaction**：碎片率超阈值时做一次 stream-compaction（prefix-sum）重建 alive_list，保证渲染实例连续。
- **容量策略**：每 emitter 固定 capacity 上限；超发丢弃或复用最老粒子（作者可选）。
- **显存预算闭环**：与 §28/§32 预算对接，超预算触发降级。

---

## 12. 排序与透明：接入共享 OIT

**Ember 不自建 OIT**，接入场景共享的透明合成路径，使粒子与其他半透明物体正确互相排序/合成。

| 混合模式 | 是否需排序 | 方案 |
|---|---|---|
| Additive / Premultiplied | 否 | 顺序无关，最省，直接 additive phase |
| Alpha blend 半透明 | 是 | 优先接入共享 OIT（MBOIT / WBOIT / per-pixel linked list）；无 OIT 时回退视深 key 的 one-sweep radix（大量）/ bitonic（少量） |
| Alpha mask / Opaque | 否 | 深度测试，无需排序 |

- **与场景统一**：走共享 OIT 时，粒子与半透明网格、毛发、布料在同一合成中按深度加权，避免"粒子永远盖在物体前/后"的经典穿插错误。
- **优化**：仅排序**可见（剔除后）**粒子；远距离 LOD 关闭排序；排序 key 量化到 16-bit 降带宽；per-tile/per-slice 局部排序。
- **深度感知**：软粒子/体积走共享 depth prepass 做深度淡出，避免硬边穿插。

---

## 13. 剔除与包围盒

- **包围盒**：GPU reduction 求 alive AABB，或固定 bounds（省算力，作者可选）。
- **多级剔除**：视锥 → 距离/屏占比 → **HZB 遮挡剔除**（复用 `occlusion_culling`，被遮挡的 emitter/tile 不渲染甚至不模拟）。
- **显著性睡眠**：屏占比过小自动睡眠，重新可见唤醒（保留/重置由作者定）。
- 剔除结果决定 indirect draw 实例数与是否跳过排序/仿真。

---

## 14. 事件系统与数据通道

两级事件体系：

- **GPU Event（emitter 内，同帧轻量）**：类型 OnSpawn/OnDeath/OnCollision/OnCondition。写 per-emitter append buffer（原子），下一 `PerEvent` Stage 消费并 spawn，继承发起者属性。
- **Data Channel（跨 system，可跨帧，命名）**：append/consume 环形缓冲。System A 写（如"每个爆炸点"），System B/C 读并 spawn（火星/冲击波/烟）。
  - 背压：溢出丢弃并计数上报。
  - 跨帧：可保留 1 帧延迟以打断依赖环、便于并行。

---

## 15. 渲染器矩阵

一个 emitter 可挂多个渲染器，均走 indirect draw，接入 `render_phase`（Transparent3d / AlphaMask3d / Opaque3d）。**每个渲染器都可搭配任意 ShadingModel（PBR/NPR/自定义/混合，§16）**。

| 渲染器 | 说明 | 支持的着色 |
|---|---|---|
| Sprite/Billboard | 相机对齐/速度对齐/固定轴；flipbook；软粒子；SDF 形状（NPR 锐利形状） | PBR-lit / NPR / additive / 自定义 |
| Mesh | 每粒子一网格实例，复用实例化 + gpu_preprocessing | 全 closure（含各向异性/清漆） |
| Ribbon/Trail | 按 RibbonId 串链，GPU 生成条带几何 | PBR-lit / NPR 速度线 / additive |
| Beam | 两点/多点链，闪电/激光 | NPR 能量 / additive |
| Light | 粒子驱动点光/体积光，对接 bevy_light clustered，数量受限 + LOD | — |
| Decal | 落地贴花，接入 deferred/forward decal | PBR / NPR |
| Volume | 网格流体密度场光线步进（对接 §10 与 §20） | 体积 PBR / 风格化体积 |

**Ribbon GPU 生成**：
1. 构建 pass：按 (RibbonId, age) 分段，写每条链 head/count 与前后邻居索引。
2. 几何 pass：每段生成四边形条带，宽度=SizeOverLife，法线朝相机或速度，UV 沿弧长参数化，切线供各向异性着色。
3. 断裂检测（相邻年龄差超阈值断开）、tessellation LOD。

- 排序结果 → 实例缓冲直接被 indirect draw 消费。
- 软粒子/体积复用 core_pipeline 的 depth prepass；所有渲染器可写 motion vector（§21）。

---

## 16. 粒子着色与材质集成：PBR/NPR/自定义/混合四等公民

**核心立场**：粒子渲染阶段调用共享材质系统的 shading closure，与主渲染完全一致。ShadingModel 是 emitter（或单粒子）级选项，四者皆为一等公民：

```
enum EmberShadingModel {
    Unlit,          // 传统 additive/自发光（最省，能量特效常用）
    Pbr,            // 物理光照，走共享 PBR closure（§17）
    Npr,            // 风格化，走共享 NPR closure（§18）
    Custom(handle), // 用户 WESL closure，走材质扩展点（§19）
    Hybrid { base, overlay, weight },  // 混合（§19）
}
```

- **编译期特化**：`Renderer × ShadingModel` 生成对应 draw kernel，内联所选 closure；未用属性不分配（如 Unlit 不占 Normal/Roughness）。
- **继承基底高级服务**：无论 PBR 还是 NPR，粒子都能消费共享基底服务——聚类光照 + RT 光照、虚拟阴影（VSM）接收与投射、体积/屏幕空间 GI、时序上采样（§21）、光追反射与受光（§22）。**这些是着色轴之上的共享服务，NPR 同样享有**（差异只在"如何响应光照"，见关联的 AAA 高级特性分册）。
- **材质契约对齐**：属性里的 MaterialId/ShadingParams 映射到共享材质参数，粒子与场景材质走同一 bind group 组织思路，避免两套体系。

四等公民与"混合引擎"关系：一个 System 内多 Emitter 各选 ShadingModel，即在特效层实现 PBR/NPR/混合共存；单粒子级混合权重实现更细粒度过渡（§19）。

---

## 17. PBR 粒子与体积着色

面向物理真实的烟、火、火星、碎屑、水花。

- **受光贴片/网格粒子**：Sprite 用 billboard 法线或法线贴图，Mesh 用真实法线；走共享 PBR closure（GGX + 多散近似），接入 bevy_light 聚类光照、阴影贴图/VSM、GI probe/irradiance、Solari RT 受光。
- **体积烟雾 PBR**：
  - **单次 + 多次散射**：单散射逐光源采样透过率，多散射用近似（如 blur/ambient 项或 Guerrilla 风格能量补偿），避免烟"发死黑"。
  - **相位函数**：Henyey-Greenstein（各向异性 g），或 double-lobe（前向+后向）表现真实烟的边缘辉光。
  - **自阴影**：deep shadow / deep opacity maps（§20），烟内部正确变暗。
  - **六向光照 rig**：预积分六方向进出光照，运行期按光向插值（VFX Graph / Guerrilla 方案），廉价近似体积散射。
- **火焰 blackbody**：温度（§10 燃烧通道）→ 普朗克黑体谱 → emissive 颜色/强度，物理正确的红→黄→白热渐变，接 HDR/bloom。
- **折射 / 热浪扭曲**：粒子写屏幕空间偏移（法线/密度驱动），采样场景颜色实现折射与热浪，走共享折射路径。
- **参与介质耦合**：粒子体积可注入场景体积雾/大气，接收方向光体积阴影。

效果目标：物理烟火在直射/背光/环境光下响应正确，边缘有散射辉光，内部有自阴影层次，与场景光照/GI/雾统一。

---

## 18. NPR 粒子与风格化着色

对标原神 / 崩坏3 / 明日方舟 / Guilty Gear Xrd 等顶级风格化标杆。**NPR 是完整一等公民**，享有几何、剔除、排序、时序、体积、阴影全部基底能力，仅"光照响应"风格化。

- **渐变色阶 / Ramp**：按受光量或年龄/速度采样 1D ramp（§6.2 LUT），得色阶量化的卡通明暗；支持双 ramp（受光/背光）。
- **描边（Outline）**：
  - Sprite：SDF 距离场生成锐利轮廓与描边（比 alpha-test 边缘更干净），或屏幕空间边缘检测。
  - Mesh 粒子：背面外扩描边 / 法线-深度边缘检测。
- **程序化 SDF 形状**：用 SDF 定义星芒、环、多边形能量形状，分辨率无关、边缘锐利——风格化能量特效的核心。
- **速度线 / 冲击波环**：Ribbon/Beam + NPR ramp 表现打击感速度线、扩散冲击环、能量波纹（UV 流动 + 菲涅尔边缘增强）。
- **天使环 / 形状化高光**：把高光响应重塑为艺术可控形状（对偶 §17 的物理高光），用于魔法/能量粒子。
- **卡通体积**：风格化烟走离散阴影带 + 描边轮廓的"cel volumetric"，而非物理散射，保持手绘感。
- **法线卡通化 / 平滑法线**：Mesh 粒子用平滑或自定义法线避免碎面，配合色阶得干净卡通面。
- **光照响应可调**：NPR closure 仍读取共享光照/阴影/GI 数据，但以风格化传递函数映射（阈值化、量化、ramp 重映射），因此虚拟阴影、体积 GI、RT 受光对 NPR 一样生效，只是"呈现"不同。

效果目标：风格化 VFX 达到主机级二次元/卡通渲染质感，轮廓干净、色阶分明、能量形状锐利，且能吃到场景阴影与光照信息。

---

## 19. 混合与自定义着色

- **Emitter 级混合**：同 System 内火焰用 Pbr、烟用 Npr、能量核用 Unlit——天然"混合引擎"。
- **单粒子级混合**：`Hybrid { base, overlay, weight }`，先算 base（如 PBR 受光）再叠 overlay（如 NPR rim/gradient），weight 可由属性（年龄/速度/受光）驱动，实现"物理受光 + 风格化描边/辉光"的过渡效果，或从写实烟渐变为风格化消散。
- **自定义 closure**：`Custom(handle)` 指向用户 WESL closure，经共享材质扩展点注入，可实现任意 BxDF / 传递函数（如虹彩、薄膜干涉、屏幕扭曲特效），编译进 draw kernel。
- **一致性保证**：混合与自定义都走同一属性/绑定契约与 OIT/motion vector 接线，行为可预测、可回归（§32/§33）。

---

## 20. 体积光照与阴影（6-way / deep shadow / phase）

体积/烟火质感的专章，供 §17/§18 的体积着色调用。

- **六向光照贴图**：预积分/运行期烘焙六个主轴的进出光照到贴片，运行期按主光/环境方向插值，成本远低于逐样本光线步进，质量接近——VFX Graph 与 Guerrilla 均采用。
- **Deep shadow / deep opacity maps**：从光源视角记录沿光线的透过率函数，供烟内部自阴影与相互投影；分层 opacity 采样廉价且平滑。
- **相位函数**：Henyey-Greenstein（可调 g）、双叶前后向散射；NPR 档用离散阶跃替代得卡通带状阴影。
- **接收/投射场景阴影**：粒子接收方向光 shadow map / VSM，也可向场景投体积软阴影（高档）。
- **体积雾/大气耦合**：粒子密度注入 froxel 体积，统一参与场景体积散射与消光，避免"粒子与雾两套光照"割裂。
- **光追增强（可选）**：Solari 提供体积受光可见性与 bounce 色（§22）。

---

## 21. Motion Vector 与时序集成（TAA / 上采样）

次世代抗锯齿与上采样（TAA / TSR / 时序上采样）要求所有可见几何写正确的运动矢量，粒子不能例外，否则运动粒子会拖影或被时序滤波抹糊。

- **每粒子 motion vector**：保存 prev_transform / prev_position（§5.2），draw kernel 输出当前与上帧裁剪空间位置差；Sprite/Mesh/Ribbon/Beam 均支持。
- **flipbook / UV 动画的时序处理**：快速变化的 flipbook 标注为"时序不稳定"，供上采样器降低历史权重，避免鬼影。
- **响应式/半透明遮罩**：向时序上采样提供 reactive mask，让高频粒子少吃历史、减少拖尾。
- **与 §12 OIT 协同**：半透明粒子的 motion vector 用于时序合成时的历史重投影。

效果目标：高速火星、飘带、烟在 TAA/上采样下清晰不拖影，静止时无抖动。

---

## 22. 光追碰撞与受光（bevy_solari，差异化亮点）

- **光追碰撞**：per-particle 沿运动方向发短射线查询 `bevy_solari::scene`，命中即碰撞/反弹。比屏幕空间深度碰撞更准（背面/离屏也可）。作为**高质量档**，普通档回退 SDF/depth。
- **光追受光/阴影**：粒子（尤其烟/体积）向光源采样可见性，得软阴影与 GI bounce 色，接入 bevy_light 聚类光照与 Solari realtime GI。PBR 与 NPR 均可消费（NPR 以风格化映射呈现）。
- 硬件不支持光追时自动降级；作为可选依赖，不阻塞核心管线。

---

## 23. GPU→CPU 反馈（gpu_readback）

- 少量高价值事件（首次落地/命中角色）压缩后经 `gpu_readback` **异步回读**，触发 gameplay/音效，多帧延迟容忍、不阻塞。
- 统计计数（alive/spawn/溢出/耗时）回读用于诊断 HUD 与预算控制。

---

## 24. CPU / GPU 双模拟

- **GPU 模拟**（默认）：大规模、无需 CPU 交互。
- **CPU 模拟**（回退/特殊）：小规模、需 gameplay 交互（拾取回调）、或无 compute 平台。用 `bevy_tasks` 并行 + SoA。
- 编译器对同一图 IR 产出**两套后端**（WESL kernel / Rust SIMD 循环），语义一致由 §29 保证。

---

## 25. 数值积分与稳定性

- 积分器可选：Semi-implicit Euler（默认，稳定廉价）/ Verlet（约束）/ RK2（高精度）。
- **子步进(substepping)**：物理向特效固定 dt 多子步，避免穿透与不稳定；XPBD 约束同样子步。
- **CFL/夹紧**：速度/位移夹紧防爆；流体平流遵守 CFL。

---

## 26. 模拟空间与大世界

- **模拟空间**：Local（随 transform，手持火把）/ World（脱离父物体，尾迹）/ Hybrid（spawn 用 local，update 用 world）。
- **大世界重定位**：origin rebasing，仿真用相对相机/区块坐标，避免远距离浮点精度崩溃。

---

## 27. 属性压缩与带宽优化

- 位置 fp16/相对量化，颜色 RGBA8，法线 oct-encode，年龄归一化 fp16——按语义选编码，编译器生成打包/解包代码。
- 百万粒子瓶颈通常在带宽；量化 + 按需分配（§5）叠加显著降带宽。
- 着色属性（法线/粗糙度）仅在对应 ShadingModel 启用时分配并量化。

---

## 28. 伸缩性与平台档案

- **质量档位**（Low→Ultra）矩阵：全局粒子上限、是否 GPU 排序/OIT、碰撞方案(depth/SDF/光追)、网格流体分辨率、体积光照档(六向/光线步进/光追)、着色档(Unlit→PBR+全高级特性)、渲染器精简。
- **平台覆盖**（PopcornFX 式）：每平台/每档覆盖 emitter 的 spawn rate、capacity、更新频率、ShadingModel 降级（如移动端把体积 PBR 降为六向近似或 Unlit）。
- **运行期降级**：超预算按 (优先级, 屏占比, 距离) 排序：降 spawn → 关排序/OIT → 降体积分辨率 → 简化着色 → 降更新频率(每 N 帧) → 剔渲染器 → 暂停仿真。
- **全局预算**：粒子/显存/GPU 时间三重预算，对接 §23 计数反馈闭环与 §32。

---

## 29. 确定性与网络

- **Hash-based 无状态 RNG**：`hash(particle_id, global_seed, stream_id, frame)`，GPU/CPU/跨后端一致。
- **确定性模式**：固定 dt + 固定种子 + 稳定 Stage 顺序，支持回放/录制。
- **网络**：表现层默认不参与同步（省带宽）；需同步的少量粒子走 CPU 确定性路径。

---

## 30. 资产格式与作者工作流

- **`.ember`**：图 + 参数 + 渲染器配置 + ShadingModel 设置，`bevy_reflect` 序列化（RON/二进制）。
- **模板与继承**：System/Emitter 继承模板并局部覆盖（Niagara Parent Emitter），改模板批量生效。
- **暴露参数**：作者标记可外部驱动参数，运行期由 gameplay/材质/时间线绑定。
- **点缓存/属性贴图**：从 Houdini/外部烘焙点云或属性纹理驱动初始状态。
- **材质/着色预览**：编辑器内切 PBR/NPR/混合实时预览，含光照环境切换。
- **Module 版本与迁移**：Module 带版本号 + 迁移函数，升级不炸旧资产。
- **热重载**：改图/改着色 → 重编译 kernel → pipeline_cache 换管线，运行时无缝替换。
- **调试**：单粒子追踪、Stage 逐帧步进、力场/网格/包围盒 gizmo（bevy_gizmos）、每 Stage GPU timestamp（bevy_diagnostic）。

---

## 31. Bevy 集成层：API 与渲染图接线

- **Extract**：`ExtractComponent`/`ExtractResource` 抽取实例 transform、user_params、playback、shading_override 到 render world。
- **Prepare**：分配/复用 GPU 缓冲，写 uniform（dt/time/camera/property binders），备 bind group（复用 combined_bind_group / material_bind_groups 思路）。
- **Simulate Node**：渲染图新增 `EmberSimulationNode`（ComputePass 序列），置于主 pass 前。
- **Queue/Phase**：每种渲染器×ShadingModel 生成 phase item，接入 Transparent3d/AlphaMask3d/Opaque3d 与共享 OIT，走 indirect draw；写 motion vector 到 prepass 目标。
- **Shader**：全 WESL + naga_oil import，复用 globals.wesl / view.wesl / maths.wesl / 共享材质 closure / 光照 / 阴影 include。
- **PipelineCache**：compute 与 draw 管线走 pipeline_cache + specializer，异步预热避免卡顿。

面向用户的最小 API（示意）：
```
app.add_plugins(EmberPlugin::default());
commands.spawn((
    EmberSystem(asset_server.load("explosion.ember")),
    EmberSystemInstance::default(),
    Transform::from_translation(pos),
));
```

---

## 32. 性能预算与效果验收

> 以下帧时间/显存数字为**设计目标（design target），非实测**（当前沙盒无 GPU）。实测须在目标硬件用 §33 基准闭环校准。

**帧时间预算（单帧，桌面 Ultra 档，仅供设计对齐）**

| 阶段 | 目标预算 | 说明 |
|---|---|---|
| 仿真（100 万粒子 per-particle） | ≤ 1.0 ms | indirect + persistent-thread |
| 网格流体（128³ 烟，含压投影） | ≤ 1.5 ms | Jacobi 近似档 |
| 排序（可见半透明） | ≤ 0.4 ms | one-sweep radix，仅可见 |
| 剔除 + 包围盒 | ≤ 0.2 ms | HZB + reduction |
| 体积着色（六向档） | ≤ 0.8 ms | 六向插值，非逐样本步进 |
| draw（含 PBR/NPR closure） | ≤ 1.5 ms | indirect draw + OIT |
| **合计目标** | **≤ ~5.5 ms** | 与主渲染 async overlap 后净增更低 |

**显存预算**：百万粒子在按需分配 + 量化下目标 < 数十 MB 量级（取决于启用属性/着色档）。

**效果验收（分档）**
- **PBR**：物理烟火在直射/背光/环境下响应正确、边缘散射辉光、内部自阴影层次；火焰黑体渐变物理可信；折射/热浪不穿帮；与场景 GI/雾/阴影统一。
- **NPR**：轮廓干净无锯齿、色阶分明、能量 SDF 形状锐利分辨率无关；吃得到场景阴影/GI 但以风格化呈现；速度线/冲击环打击感到位。
- **混合/自定义**：Emitter 级与单粒子级混合过渡平滑、weight 驱动可控；自定义 closure 行为可回归。
- **时序**：高速粒子在 TAA/上采样下清晰不拖影，静止无抖动（motion vector + reactive mask 生效）。
- **稳定性**：容量溢出、free-list 竞态、事件背压、compaction 正确性无崩溃；确定性模式可复现。

---

## 33. 质量与可信度基础设施

- **Golden image 回归**：确定性模式渲染比对；PBR/NPR/混合各留基准场景。
- **性能基准**：接入现有 `benches/`，跑 1e5/1e6/1e7 粒子仿真+渲染耗时与显存，防回退；各着色档分别基准。
- **数值一致性**：GPU vs CPU 求解同图，误差阈值内一致（可移植 compute 桶可 CPU golden；RT traversal 不可）。
- **压力/边界测试**：容量溢出、free-list 竞态、事件背压、compaction 正确性、OIT 层数溢出。

---

## 34. Crate 拆分与落地形态

```
bevy_ember            用户 API / Asset / Component / trait(含 ShadingModel) / CPU 求解器
bevy_ember_compiler   图 → IR → WESL codegen / 活跃性分析 / 着色特化 key
bevy_ember_render     extract/prepare/queue / 渲染图节点 / GPU 缓冲 / WESL / indirect / OIT / MV 接线
bevy_ember_editor     (可选) 图编辑器 / 预览(含着色预览) / 性能 HUD
```
依赖方向：`editor → ember → {compiler, ember_render} → bevy_render/bevy_ecs/bevy_light/bevy_solari/...`。可作为 optional feature 接入 `bevy_internal`。新代码落在 `pkg/` 下对应 crate。

---

## 35. 路线图

| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 骨架 | crate 布局、Asset/Component、CPU 简单求解器、Sprite Unlit 渲染 | CPU 1 万粒子火花 demo |
| M1 GPU 核心 | SoA 缓冲池、free-list、spawn/update compute、indirect draw | GPU 100 万粒子无回读 |
| M2 图编译 | 图→IR→WESL codegen、按需分配、曲线 LUT | 图驱动 update kernel |
| M3 力/碰撞 | 力场库、curl noise、depth/SDF 碰撞、事件系统 | 烟火/瀑布/碎片 demo |
| M4 渲染器 | Mesh/Ribbon/Beam/Light、GPU 排序、软粒子、motion vector | 完整特效场景 + TAA 无拖影 |
| M5 PBR 着色 | 共享 PBR closure 接入、聚类受光、软粒子折射、共享 OIT | 物理受光火星/水花 demo |
| M6 NPR 着色 | ramp/色阶、SDF 形状、描边、速度线/能量环、卡通体积 | 二次元风格化 VFX demo |
| M7 混合/自定义 | Emitter 级 + 单粒子级混合、Custom WESL closure | 火焰 PBR + 卡通烟同场 demo |
| M8 体积光照 | 六向 rig、deep shadow、相位函数、体积雾耦合 | 次世代体积烟火 demo |
| M9 高级求解器 | Simulation Stages、邻域网格、不可压缩流体、涡量、燃烧、XPBD、Data Channels | 体素烟火 + 群体 + 布/绳 demo |
| M10 次世代 | Solari 光追碰撞/受光、HZB 剔除、GPU→CPU 反馈、平台档案、golden-image + 基准 | 光追特效 + 回归门禁 |

---

## 36. 关键扩展点清单

- `EmberModule` / `DataInterface` / `EmberForce`：注册 WESL 代码片段。
- `StageKind` / IterationDomain：新增迭代域（如自定义求解器）。
- `EmberRenderer`：新增渲染器类型（含 phase 接线与 WESL）。
- `EmberShadingModel` / Custom closure：新增/替换着色模型，接入共享材质扩展点。
- `IntegratorKind`：新增积分器。
- `SortStrategy` / `CullStrategy`：替换排序/剔除算法。
- `PlatformProfile` / `QualityLevel`：伸缩档位覆盖（含着色档）。
- Asset 序列化与迁移钩子（bevy_reflect）。

---

## 37. 开放问题

1. **首发图模型**：先"线性 Module 堆叠"（易用、快出成果）再演进"任意 DAG"（灵活）——建议堆叠先行，IR 预留 DAG。
2. **光追依赖**：Solari 碰撞/受光作为可选高质量档（默认关闭）以兼容无光追硬件——建议采纳。
3. **体素流体范围**：M9 是否包含完整压力投影（不可压缩烟），还是先做无散度近似——建议先近似（Jacobi），后 multigrid 精确。
4. **CPU 后端优先级**：是否 M0 即交付 CPU 后端，还是仅作回退——建议 M0 交付最小 CPU 后端便于调试与确定性。
5. **体积着色默认档**：桌面默认六向 rig 还是逐样本光线步进——建议默认六向（性价比），Ultra 档开步进/光追。
6. **NPR 与共享光照耦合度**：NPR ramp 完全替换光照 vs 在共享光照结果上做风格化映射——建议后者（吃到阴影/GI，呈现风格化），保架构统一。

---

## 38. 术语表

| 术语 | 含义 |
|---|---|
| System / Emitter | 特效整体 / 一类粒子的发射器 |
| Stage | 有序执行阶段（spawn/update/simulation stage） |
| Iteration Domain | Stage 的 dispatch 依据（粒子/网格/邻域/约束/事件） |
| Attribute (SoA) | 粒子属性，结构分离数组布局 |
| Module | 可堆叠的行为单元，编译为内联 WESL |
| DataInterface | 只读外部数据采样器（网格/SDF/相机/光追等） |
| Data Channel | 跨 system 的命名事件/数据总线 |
| ShadingModel | 粒子着色模型（Unlit/PBR/NPR/自定义/混合），复用共享材质 closure |
| Closure | 共享材质系统的着色函数单元 |
| OIT | 顺序无关透明合成（共享） |
| Motion Vector | 屏幕空间运动矢量，供 TAA/时序上采样 |
| 六向光照 | 预积分六方向进出光照的廉价体积散射近似 |
| Deep shadow / opacity | 沿光线记录透过率函数，供体积自阴影 |
| Phase function | 体积散射角分布（如 Henyey-Greenstein） |
| Vorticity confinement | 涡量约束，补偿数值耗散保烟卷曲细节 |
| XPBD / VBD | 统一约束求解原语（对齐 physics_core） |
| Free-list | 空闲槽栈式分配器 |
| Compaction | prefix-sum 压缩存活数组 |
| Indirect dispatch/draw | 由 GPU 决定线程/实例数的派发/绘制 |
| HZB | 层级 Z 缓冲，用于遮挡剔除 |
| Property Binder | CPU→GPU 参数绑定 |
| Point Cache | 外部烘焙点云/属性贴图 |
