# Prism 渲染引擎 — 布料引擎子系统完整设计（v1 / wgpu + WESL）

> 状态：架构提案（Draft，允许破坏性重构）
> 定位：AAA / 次世代 XPBD 布料/服装引擎，与 PBR/NPR/自定义/混合四前端正交协同
> Shader：WESL（naga → WGSL/SPIR-V/Metal/DXIL）；运行时：wgpu（Metal/VK/DX12/WebGPU）
> 关联文档：`prism_rendering_architecture_zh.md`（总）、`prism_material_pipeline_design_zh.md`（§6.2 子系统注册表）、`prism_hair_engine_design_zh.md`（对称子系统）、`prism_physics_design_zh.md`（§3 统一 XPBD 内核）、`prism_aaa_advanced_features_zh.md`

---

## 0. 定位：这是什么级别的布料引擎

是**完整 XPBD 服装/布料引擎**（三角网格 + 约束求解 + 服装管线），不是"给衣服贴一张 sheen 高光"的着色贴皮。对标业界成熟产品，仅在**算法层**借鉴，不复制其代码：

| 产品 | 借鉴点（算法层） |
|---|---|
| **UE5 Chaos Cloth** | XPBD 布料求解、GPU 化、自碰撞、空气动力学（风/升阻）、LOD、绘制刷权重、长距离约束（LRA）防拉伸 |
| **NVIDIA NvCloth / PhysX Clothing / APEX** | 距离约束 + 三角弯曲、虚拟粒子自碰撞、GPU 并行批处理、tether（系绳）约束 |
| **Marvelous Designer / CLO** | 2D 版片缝合 → 3D 服装的**版片资产模型**、缝合线、内外层与褶皱 |
| **Havok Cloth** | 骨骼驱动局部空间 sim、碰撞代理体（capsule/sphere/convex） |
| **Houdini Vellum（影视 XPBD）** | 高保真离线 XPBD、撕裂/塑性、多层服装耦合、气泡/薄膜 |

**判据（承接材质设计 §6.2）**：一等子系统必须拥有**自己的几何 + 自己的 sim + 特殊渲染**。布料三者全占：三角网格服装几何、XPBD 布料 sim（拉伸/弯曲/剪切/自碰撞）、双面薄透射 + sheen/fuzz 着色 + 法线/褶皱细节。因此是一等子系统；而"布料看起来的绒感高光"只是 über-BSDF 的 sheen closure 瓣，不构成子系统。**布料与毛发是两个独立子系统**（三角网格+布料 XPBD vs 发丝几何+strand sim），仅共享形变预算这一资源仲裁层。

---

## 1. 与整体架构的关系（子系统内部同构范式）

布料严格遵循"共享 GPU-driven 基底 + PBR/NPR 分叉响应"范式，架构一致性拉满：

```
共享基底（PBR/NPR/自定义/混合 都吃）:
    服装资产（版片/缝合）→ sim mesh（低模）+ render mesh（高模）
    → 约束图构建（拉伸/弯曲/剪切/LRA/tether）
    → XPBD 布料 sim（子步 + compliance）→ 形变顶点（汇入 gpu_scene）
    → 碰撞（身体代理体 + 自碰撞）
    → 连续 LOD（sim 分辨率 + render mesh LOD + 远距蒙皮代理，禁硬切换 pop）
    → render mesh 嵌入/蒙皮跟随 sim mesh
    → 进虚拟几何 vis-buffer；褶皱法线细节

分叉响应（C 类，真分家）:
    PBR: cloth sheen/fuzz BRDF（Charlie/Ashikhmin）+ 双面薄透射 + 微纤维各向异性 + 褶皱法线
    NPR: 风格化布料 ramp + 手绘褶皱线 + 顶点色控阴影段 + 描边（复用 material id 边）

fallback:
    高配 sim + 自碰撞；基线 sim 无自碰撞；兜底纯蒙皮（不 sim）
```

布料**不拥有**光照/阴影/OIT/motion/虚拟几何这些公共设施——它**消费**基底服务，只在"几何+sim+响应"处分家。这与"每个前端各自重算高级特性"相反，是本引擎"管线级混合"的优越性来源。

---

## 2. 数据与资产模型

- **服装资产（garment）**：一组 2D 版片（panel）+ 缝合线（seam）+ 材料参数（拉伸/弯曲刚度、密度、摩擦、风阻），对标 Marvelous Designer / CLO 的版片管线；也支持直接导入已缝合的 3D 服装网格。
- **Sim mesh（仿真网格）**：低分辨率三角网格，真正被 XPBD 模拟的粒子集合；决定 sim 成本。
- **Render mesh（渲染网格）**：高分辨率外观网格，通过**嵌入/蒙皮**跟随 sim mesh 形变；决定视觉细节与光栅成本，本身不参与约束求解。
- **约束图**：距离/拉伸（结构边）、弯曲（二面角/交叉边）、剪切（对角）、LRA 长距离约束防过拉伸、tether 系绳（固定点约束）。
- **碰撞代理**：绑定骨骼的 capsule/sphere/convex 身体代理体 + 自碰撞（虚拟粒子/空间哈希）。
- **绑定**：服装绑定到蒙皮 mesh，固定点（腰带/领口）随蒙皮移动（形变父级 = 材质设计 §6.1 蒙皮地基）。

代码落点（`prism_render_architecture::cloth`，**待建，与 `hair` 对称**）：
- `ClothPieceHandle`、`ClothPiece{ sim_vertex_count, render_vertex_count, constraint_count, native_form, deformation }`（规划 `cloth/mod.rs`）。
- `ClothLodTier{ FullSim, ReducedSim, SkinnedProxy }` + `is_simulated()`（规划 `cloth/mod.rs`）。
- 复用**已落**的 `deformation::DeformationKind::Cloth`（枚举值已就位）与 `deformation::schedule` 形变预算仲裁。

---

## 3. 管线阶段（端到端）

| 阶段 | 内容 | 对标 | 代码落点 |
|---|---|---|---|
| 1. 导入/缝合 | 版片缝合 → sim mesh；render mesh 嵌入绑定 | Marvelous / CLO | 规划 `cloth/asset.rs` |
| 2. 约束构建 | 拉伸/弯曲/剪切/LRA/tether 约束图 + 图着色分批 | Chaos / NvCloth | 规划 `cloth/constraints.rs` |
| 3. 布料动力学 | XPBD 子步求解（compliance，边长/弯曲/剪切） | Chaos Cloth / Vellum | 规划 `cloth/dynamics.rs`（→ `deformation::schedule`；求解原语对齐 `prism_physics_core` §3 统一 XPBD）|
| 4. 碰撞 | 身体代理体碰撞 + 自碰撞（空间哈希/虚拟粒子）| PhysX Clothing / Havok | 规划 `cloth/collision.rs` |
| 5. LOD | 覆盖度/距离选档 + 连续 sim 降分辨率，禁硬切换 | Chaos LOD | 规划 `cloth/lod.rs`（对称 `hair/lod.rs`）|
| 6. 嵌入 | render mesh 跟随 sim mesh（重心坐标/蒙皮嵌入）| UE5 | 规划 `cloth/embed.rs` |
| 7. 着色 | cloth sheen/fuzz closure + 双面薄透射 | 材质系统 closure | `material` 的 cloth closure（不在 cloth 内重写）|
| 8. 透明/合成 | 薄纱走共享 OIT | `transparency` | **已落** `transparency::routing`（薄透布走透明路径）|

**关键边界**：着色（阶段 7）走材质系统的 cloth sheen/fuzz closure（Charlie/Ashikhmin-Shirley），透明（阶段 8）走共享 OIT——布料模块**不重写**着色与透明，只产几何与形变。sim/LOD/嵌入落"compute 可移植"桶，可 CPU golden。

---

## 4. LOD 策略（禁 pop 硬切换）

三档：`FullSim`（全分辨率 sim + 自碰撞）→ `ReducedSim`（降分辨率 sim，弱/无自碰撞）→ `SkinnedProxy`（纯蒙皮跟随，不 sim）。仅 `is_simulated()` 档（`FullSim`/`ReducedSim`）发形变请求。

**`native_form` 钳制**（对称毛发）：授权某服装的原生形态（如背景 NPC 衣物 `native_form=SkinnedProxy`——就该纯蒙皮，不是降级）。`resolve_cloth_lod` 用 `tier.coarser_of(piece.native_form)` 把覆盖度/距离选出的档钳制成"不比授权更细"——满屏也不会把纯蒙皮 NPC 衣物提升到它根本没配的全 sim，但距离拉远仍可继续降档。这是**授权几何选择，与 PBR/NPR 着色响应正交**，两种风格同等遵循。

分桶按黄金范式（对齐 `virtual_geometry/bins.rs`）：per-tier `Vec` 桶 + `push`/`bin_cloth_lod` 纯函数、确定性输入序、越界跳过不 panic。

---

## 5. PBR / NPR / 自定义 / 混合 分叉响应

- **PBR**（对标 UE5 Chaos + film sheen）：cloth sheen/fuzz BRDF（Charlie sheen / Ashikhmin-Shirley 各向异性）+ 双面薄透射（薄纱/丝绸透光）+ 微纤维各向异性 + 褶皱法线细节；有 RT 时参与硬件阴影/反射。
- **NPR**（对标 miHoYo / Arc Sys 卡通服装）：风格化布料 ramp + 手绘褶皱线（沿应力方向叠笔刷）+ 顶点色控阴影段；描边复用 vis-buffer 的 material id 边界；卡通褶皱可与真实法线解耦（美术手编）。
- **自定义**：项目注入 `illumination=Custom` 的 WESL closure。
- **混合**：同一服装可按 material id 分 tile 路由到不同前端（因共享光/影/GI 而连贯）；写实场景里的卡通角色衣物与 PBR 世界吃同一份 GI/阴影，不断裂。

这四者共享 §1 的全部基底（同一份服装几何、XPBD sim、LOD、碰撞），**只在着色响应处分家**——四者都能享受布料级 sim 与褶皱几何，不是各做各的。

---

## 6. 性能预算与降级矩阵

| 能力 | 高配 | 基线 | 兜底 |
|---|---|---|---|
| 几何 | 高模 render mesh + 全 sim mesh | 中模 + 降分辨率 sim | 蒙皮壳 |
| sim | XPBD + 自碰撞 + 空气动力学 | XPBD 仅身体碰撞 | 静态蒙皮（不 sim）|
| 约束 | 拉伸+弯曲+剪切+LRA+tether | 拉伸+弯曲 | — |
| 碰撞 | 身体代理 + 自碰撞 | 身体代理 | 无 |
| 褶皱 | 动态褶皱法线（应力驱动）| 预烘焙褶皱贴图 | 无 |
| 透明 | OIT（薄纱）| 排序 alpha | 不透明近似 |

- 自碰撞是布料最重特性：**基线只做身体碰撞务实、自碰撞高配可选**（承接材质设计 §10 风险条）。
- 求解稳定性靠**子步 substepping + compliance**（对齐 `prism_physics_core` §3.3/§3.4）——子步多、单步少迭代，能量更稳、刚度更高。
- 形变预算由 `deformation::schedule` 统一仲裁：单帧顶点上限 + BLAS refit 独立配额；多套服装超预算时最高优先项无条件先跑（防饿死），其余延后。

---

## 7. 可测性

布料 sim / 约束 / LOD / 嵌入落"compute 可移植"桶（材质设计 §8.1）：数组进数组出，可写手写 Rust CPU golden 逐值对数。规划 `cloth/lod.rs` 首批确定性单测（对称 `hair/lod.rs` 13 测）：阈值分档、连续降分辨率、代理档不发 sim 请求、sim 档发 `DeformationKind::Cloth` 形变请求（`vertex_count = sim_vertex_count`）、分桶保序、越界跳过、空输入、`native_form` 钳制、`coarser_of` 秩比较。XPBD 约束投影可 golden（固定拓扑 + 固定子步）；RT traversal 不可（驱动 BVH）——与全引擎三桶边界一致。

---

## 8. 落地路线图

1. **子系统骨架** `cloth/mod.rs`：`ClothPieceHandle`/`ClothPiece`/`ClothLodTier` + `is_simulated()`/`coarseness()`/`coarser_of()`（复用 `DeformationKind::Cloth`）。
2. **LOD** `cloth/lod.rs`：档选择 + 连续降分辨率 + `native_form` 钳制 + `bin_cloth_lod` 分桶 + `cloth_deformation_request`；确定性 CPU golden（先落，最低冲突、与毛发对称）。
3. **约束构建** `cloth/constraints.rs`：拉伸/弯曲/剪切/LRA/tether 约束图 + 图着色分批（对齐物理 §3.5）。
4. **布料动力学** `cloth/dynamics.rs`：XPBD 子步求解（compliance），先仅身体碰撞，输出接 `deformation::schedule`；确定性 golden。
5. **碰撞** `cloth/collision.rs`：身体代理体（capsule/sphere/convex）+ 自碰撞（空间哈希/虚拟粒子）。
6. **嵌入** `cloth/embed.rs`：render mesh 重心坐标/蒙皮嵌入跟随 sim mesh。
7. **着色分叉**：material cloth sheen/fuzz closure（Charlie/Ashikhmin）+ `prism_render_npr` 风格化布料响应。
8. **空气动力学与褶皱**：风/升阻力、应力驱动动态褶皱法线；跨档 dither 过渡消 pop。

**优先级**：先落骨架 + `cloth/lod.rs`（最低冲突、与毛发对称、`Cloth` 枚举已备），XPBD 动力学与自碰撞随物理内核对齐逐步点亮；每步 wgpu 可编译 + 单测绿。

---

## 9. 风险

- **自碰撞 + 稳定性是重头**：布料自碰撞算力/稳定性折衷大，需空间哈希与虚拟粒子先就位；子步数 vs 预算需折衷。
- **与物理内核的边界**：XPBD 求解原语应对齐 `prism_physics_core` §3 统一内核，避免渲染侧重复造求解器——渲染侧 cloth 模块负责服装几何/LOD/嵌入/形变调度，求解可复用物理内核原语。
- **NPR 布料无成熟范式**：顶级二次元服装褶皱多为美术手编 + 有限 sim，PBR 侧可抄业界、NPR 侧需自趟。
- **确定性要求**：联机/回放需固定子步 + 固定图着色顺序 + 固定种子（承接物理 §18 确定性）。

---

## 附录：术语

- **版片 / 缝合线**：2D 服装裁片与缝合边，缝合成 3D 服装（Marvelous 式）。
- **sim mesh / render mesh**：被模拟的低模 vs 跟随嵌入的高模外观网格。
- **LRA / tether**：长距离约束防过拉伸 / 系绳固定点约束。
- **compliance（柔度）**：XPBD 中代替 stiffness 的材料软硬量，与时间步无关。
- **sheen / fuzz**：布料绒面各向异性微高光（Charlie/Ashikhmin closure 瓣）。
- **FullSim / ReducedSim / SkinnedProxy 档**：LOD 阶梯；仅前两档做 sim。
