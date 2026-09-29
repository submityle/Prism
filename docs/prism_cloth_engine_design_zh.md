# Prism 渲染引擎 — 布料引擎子系统完整设计（v2 / wgpu + WESL）

> 状态：架构提案（Draft，允许破坏性重构）
> 定位：AAA / 次世代 XPBD 布料/服装引擎，与 PBR/NPR/自定义/混合四前端正交协同
> Shader：WESL（naga → WGSL/SPIR-V/Metal/DXIL）；运行时：wgpu（Metal/VK/DX12/WebGPU）
> 关联文档：`prism_rendering_architecture_zh.md`（总）、`prism_material_pipeline_design_zh.md`（§6.2 子系统注册表）、`prism_hair_engine_design_zh.md`（对称子系统）、`prism_physics_design_zh.md`（§3 统一 XPBD 内核、§4 多求解器插槽、§10 自适应、§12 异步流水线）、`prism_aaa_advanced_features_zh.md`
> v2 变更：新增 §6 高级仿真特性、§7 布料着色模型（sheen/fiber/薄透射），§8 性能路径升级为 GPU-driven 持久化 + 异步流水线 + 图着色批处理，补 §10 效果验收口径。

---

## 0. 定位：这是什么级别的布料引擎

是**完整 XPBD 服装/布料引擎**（三角网格 + 约束求解 + 服装管线），不是"给衣服贴一张 sheen 高光"的着色贴皮。对标业界成熟产品，仅在**算法层**借鉴，不复制其代码：

| 产品 | 借鉴点（算法层，v2 细化） |
|---|---|
| **UE5 Chaos Cloth** | XPBD 求解、GPU 化、自碰撞、空气动力学（逐三角升/阻力）、LOD、绘制刷权重、长距离约束（LRA）、backstop/max-distance/blend-weight 绘制约束、animation drive |
| **NVIDIA NvCloth / PhysX Clothing / APEX** | 距离约束 + 三角弯曲、虚拟粒子自碰撞、GPU 图着色分批并行、tether（系绳）、strain limiting 应变限制 |
| **Marvelous Designer / CLO** | 2D 版片缝合 → 3D 服装的版片资产模型、缝合线、内外层与预褶皱、warp/weft 经纬各向异性 |
| **Havok Cloth** | 骨骼驱动局部空间 sim、碰撞代理体（capsule/sphere/convex）、休眠/激活 |
| **Houdini Vellum（影视 XPBD）** | 高保真离线 XPBD、撕裂/塑性形变、多层服装耦合、压力约束（气泡/薄膜）、自适应重网格 |
| **VBD（Vertex Block Descent, SIGGRAPH 2024）** | 极刚材料/大形变无条件稳定、GPU 大规模并行——作为**高保真求解插槽**（对齐物理 §4） |
| **影视 fiber-level（Sony/迪士尼）** | Estevez-Kulla sheen、多重散射织物、逐纤维近似——作为**着色**高配 closure |

**判据（承接材质设计 §6.2）**：一等子系统必须拥有**自己的几何 + 自己的 sim + 特殊渲染**。布料三者全占：三角网格服装几何、XPBD 布料 sim（拉伸/弯曲/剪切/自碰撞）、双面薄透射 + sheen/fuzz 着色 + 张力驱动褶皱法线。因此是一等子系统；而"布料看起来的绒感高光"只是 über-BSDF 的 sheen closure 瓣，不构成子系统。**布料与毛发是两个独立子系统**（三角网格+布料 XPBD vs 发丝几何+strand sim），仅共享形变预算这一资源仲裁层。

---

## 1. 与整体架构的关系（子系统内部同构范式）

布料严格遵循"共享 GPU-driven 基底 + PBR/NPR 分叉响应"范式：

```
共享基底（PBR/NPR/自定义/混合 都吃）:
    服装资产（版片/缝合）→ sim mesh（低模）+ render mesh（高模）
    → 约束图构建（拉伸/弯曲/剪切/LRA/tether）+ 图着色分批
    → XPBD/VBD 布料 sim（子步 + compliance + strain limiting）→ 形变顶点（汇入 gpu_scene）
    → 碰撞（身体代理体 + 自碰撞 CCD + backstop）
    → 连续 LOD（sim 分辨率 + render mesh LOD + 远距蒙皮代理，禁硬切换 pop）
    → render mesh 嵌入/蒙皮跟随 sim mesh
    → 进虚拟几何 vis-buffer；张力驱动褶皱法线

分叉响应（C 类，真分家）:
    PBR: cloth sheen/fuzz BRDF（Charlie/Estevez-Kulla）+ 双面薄透射 + 经纬各向异性 + 褶皱法线
    NPR: 风格化布料 ramp + 手绘褶皱线 + 顶点色控阴影段 + 描边（复用 material id 边）

fallback:
    高配 VBD/XPBD sim + 自碰撞；基线 XPBD 无自碰撞；兜底纯蒙皮（不 sim）
```

布料**不拥有**光照/阴影/OIT/motion/虚拟几何这些公共设施——它**消费**基底服务，只在"几何+sim+响应"处分家。这是本引擎"管线级混合"优越性的来源。

---

## 2. 数据与资产模型

- **服装资产（garment）**：一组 2D 版片（panel）+ 缝合线（seam）+ 材料参数（经/纬拉伸刚度、弯曲刚度、密度、摩擦、风阻），对标 Marvelous Designer / CLO；也支持直接导入已缝合的 3D 服装网格。
- **Sim mesh**：低分辨率三角网格，真正被求解的粒子集合；决定 sim 成本。
- **Render mesh**：高分辨率外观网格，通过嵌入/蒙皮跟随 sim mesh；决定视觉细节与光栅成本，不参与约束求解。
- **约束图**：距离/拉伸（结构边，经纬可各向异性）、弯曲（二面角 / Bergou 等距弯曲）、剪切（对角）、LRA 长距离约束防过拉伸、tether 系绳固定点、pressure 压力（气泡/薄膜可选）。
- **绘制约束（painted，对标 Chaos）**：`max_distance`（限制点离蒙皮姿态的最大位移）、`backstop`（防穿透身体的背挡距离）、`blend_weight`（sim 姿态与蒙皮姿态的混合权重，稳定性/贴合折衷）、`anim_drive`（向动画目标位置的驱动）。
- **碰撞代理**：绑定骨骼的 capsule/sphere/convex 身体代理体 + 自碰撞（空间哈希 + 虚拟粒子，含 CCD）。
- **绑定**：服装绑定到蒙皮 mesh，固定点随蒙皮移动（形变父级 = 材质设计 §6.1 蒙皮地基）。

代码落点（`prism_render_architecture::cloth`，**待建，与 `hair` 对称**）：
- `ClothPieceHandle`、`ClothPiece{ sim_vertex_count, render_vertex_count, constraint_count, native_form, deformation }`（规划 `cloth/mod.rs`）。
- `ClothLodTier{ FullSim, ReducedSim, SkinnedProxy }` + `is_simulated()`（规划 `cloth/mod.rs`）。
- 复用**已落**的 `deformation::DeformationKind::Cloth` 与 `deformation::schedule` 形变预算仲裁。

---

## 3. 管线阶段（端到端）

| 阶段 | 内容 | 对标 | 代码落点 |
|---|---|---|---|
| 1. 导入/缝合 | 版片缝合 → sim mesh；render mesh 嵌入绑定 | Marvelous / CLO | 规划 `cloth/asset.rs` |
| 2. 约束构建 | 拉伸/弯曲/剪切/LRA/tether/pressure 约束图 + 图着色分批 | Chaos / NvCloth | 规划 `cloth/constraints.rs` |
| 3. 布料动力学 | XPBD/VBD 子步求解（compliance）+ strain limiting | Chaos / Vellum / VBD | 规划 `cloth/dynamics.rs`（→ `deformation::schedule`；原语对齐 `prism_physics_core` §3/§4）|
| 4. 碰撞 | 身体代理体 + 自碰撞 CCD + backstop | PhysX Clothing / Havok | 规划 `cloth/collision.rs` |
| 5. LOD | 覆盖度/距离选档 + 连续降分辨率 + 自适应重网格，禁硬切换 | Chaos LOD / Vellum | 规划 `cloth/lod.rs`（对称 `hair/lod.rs`）|
| 6. 嵌入 | render mesh 跟随 sim mesh（重心坐标/蒙皮嵌入）| UE5 | 规划 `cloth/embed.rs` |
| 7. 着色 | cloth sheen/fuzz closure + 双面薄透射（见 §7）| 材质系统 closure | `material` cloth closure（不在 cloth 内重写）|
| 8. 透明/合成 | 薄纱走共享 OIT | `transparency` | **已落** `transparency::routing` |

**关键边界**：着色（阶段 7）走材质系统 cloth closure，透明（阶段 8）走共享 OIT——布料模块**不重写**着色与透明，只产几何与形变。sim/LOD/嵌入落"compute 可移植"桶，可 CPU golden。

---

## 4. LOD 策略（禁 pop 硬切换）

三档：`FullSim`（全分辨率 + 自碰撞）→ `ReducedSim`（降分辨率，弱/无自碰撞）→ `SkinnedProxy`（纯蒙皮，不 sim）。仅 `is_simulated()` 档发形变请求。

**`native_form` 钳制**（对称毛发）：授权某服装原生形态（背景 NPC 衣物 `native_form=SkinnedProxy`——就该纯蒙皮，非降级）。`resolve_cloth_lod` 用 `tier.coarser_of(piece.native_form)` 钳成"不比授权更细"，满屏也不越级提升，远距仍可降档。**授权几何选择，与 PBR/NPR 着色响应正交**。

**自适应分辨率**（对标 Vellum / 物理 §10）：褶皱聚集/高曲率区可局部加密 sim（可选高配），平坦区粗化——在固定预算内把粒子花在视觉最需要处。分桶按黄金范式（对齐 `virtual_geometry/bins.rs`）：per-tier `Vec` 桶 + `bin_cloth_lod` 纯函数、确定性输入序、越界跳过不 panic。

---

## 5. PBR / NPR / 自定义 / 混合 分叉响应

- **PBR**：cloth sheen/fuzz + 双面薄透射 + 经纬各向异性 + 张力褶皱法线（详见 §7）；有 RT 时参与硬件阴影/反射。
- **NPR**：风格化布料 ramp + 手绘褶皱线（沿应力方向叠笔刷）+ 顶点色控阴影段；描边复用 vis-buffer material id 边；卡通褶皱可与真实法线解耦（美术手编）。
- **自定义**：项目注入 `illumination=Custom` 的 WESL closure。
- **混合**：同一服装按 material id 分 tile 路由不同前端（共享光/影/GI 而连贯）；写实场景卡通角色衣物与 PBR 世界吃同一份 GI/阴影，不断裂。

四者共享 §1 全部基底（同一份服装几何、sim、LOD、碰撞），**只在着色响应处分家**。

---

## 6. 高级仿真特性（v2 新增，含性能/效果取舍）

> 这些是"顶级布料引擎"与"能动的三角网格"的分水岭。逐条：算法 / 产品对标 / 成本 / 效果 / 落点。

1. **多求解器插槽（XPBD 基线 / VBD 高保真）**：默认 XPBD 子步 + compliance（稳、快、跨平台）；极刚材料/大形变切 VBD（无条件稳定、GPU 大并行）。**对齐物理 §4 SolverRegistry**。成本：VBD 每顶点更贵；效果：皮革/厚重织物不软塌。落点 `cloth/dynamics.rs`。
2. **自碰撞 + CCD（连续碰撞检测）**：空间哈希 + 虚拟粒子；CCD 防高速穿插隧穿。**最重特性**——高配可选、基线只做身体碰撞。成本：空间哈希构建 + 大量对测；效果：多层裙摆不互穿。落点 `cloth/collision.rs`。
3. **空气动力学（逐三角升/阻力）**：按三角法线与相对风速算升力/阻力（Chaos）。成本：低（逐三角一次）；效果：飘动/鼓风/降落伞感。
4. **风场耦合**：全局风 + 局部风场（体积/噪声），可与粒子/植被共享风数据。成本：低；效果：环境一致的摆动。
5. **strain limiting（应变限制）**：约束投影后二次钳制边长，防"超弹"拉丝。成本：一趟后处理；效果：布料不像橡皮。
6. **backstop / max-distance / blend-weight（绘制约束）**：美术刷权重控制贴合与位移上限、防穿身体、sim↔蒙皮混合。成本：数据驱动近零；效果：紧身处稳、飘逸处自由——**AAA 角色服装质感的关键调参手段**。
7. **多层服装耦合**：内衬/外套分层，层间约束防互穿并保持叠放次序（Vellum）。成本：层数线性；效果：外套盖住内衬不穿帮。
8. **撕裂 / 塑性形变（可选高配）**：应力超阈断约束（撕裂）、超弹性形变保留（塑性褶皱）。成本：拓扑变更需重建约束图；效果：布料破损/永久褶皱。
9. **压力约束（pressure）**：闭合网格体积保持，做气球/羽绒/薄膜。成本：低；效果：充气/鼓胀。
10. **两向耦合**：布料受刚体推挤、也反推轻质刚体（对齐物理 §8 统一耦合）。成本：耦合迭代；效果：物件压在布上凹陷。
11. **休眠 / 激活（sleep）**：静止服装停 sim、被扰动再激活（Havok）。成本：负（省算力）；效果：大量 NPC 服装可扩展。
12. **子步 substepping + compliance**：子步多、单步少迭代，能量更稳、刚度更高、与时间步无关（物理 §3.3/§3.4）——**稳定性的地基**。

---

## 7. 布料着色模型（v2 新增，材质系统 closure 侧）

> 布料 sim 产几何，**着色由材质系统 cloth closure 承载**（本模块不重写）。这里列 closure 目标形态与产品对标，供材质 ABI 侧实现。

1. **sheen / 绒面各向异性微高光**：基线 Charlie sheen（UE/glTF）；高配 **Estevez-Kulla sheen**（Sony Imageworks，能量守恒、掠射边缘绒光）。效果：天鹅绒/丝绒边缘泛光。
2. **经纬各向异性高光（woven）**：Ashikhmin-Shirley / 双切线各向异性，沿经纬方向拉丝高光。效果：缎面/丝绸方向性反光。
3. **双面薄透射（thin transmission）**：薄纱/雪纺背面透光，双面法线；走共享 OIT 合成。效果：透光轻纱。
4. **多重散射织物 / 逐纤维近似（高配）**：影视 fiber-level 的 dual-scatter 近似，厚织物次表面透光。效果：毛线/厚呢的柔透。
5. **薄膜干涉（silk iridescence，可选）**：thin-film 让丝绸/尼龙出彩虹晕。效果：变色丝绸。
6. **张力驱动褶皱法线**：按 sim 应变/曲率混合褶皱法线贴图强度（松弛→平、拉紧→显褶）。效果：动态褶皱，非静态贴图。
7. **接触自阴影 / AO**：褶皱凹陷处接触阴影，消费共享阴影/AO 服务。效果：布料层叠有体积感。

**NPR 对偶**：以上物理 closure 在 NPR 前端替换为 ramp 量化 + 手绘褶皱线 + 解耦高光带，消费同一 sim 应变数据。

---

## 8. 性能预算与降级矩阵（v2 升级：GPU-driven + 异步）

**GPU-driven 持久化管线**（对齐物理 §11）：sim 状态常驻 GPU 缓冲，约束按图着色分批（组内 Jacobi 并行、组间 Gauss-Seidel 串行），避免逐帧回读 CPU。
**异步流水线化**（对齐物理 §12）：sim 与渲染解耦，sim 可落后渲染 1 帧或以固定步长跑，形变顶点通过双缓冲交给 `gpu_scene`——摊平尖峰、不阻塞主渲染。

| 能力 | 高配 | 基线 | 兜底 |
|---|---|---|---|
| 求解器 | VBD / XPBD 多子步 | XPBD 少子步 | 蒙皮（不 sim）|
| 几何 | 高模 render + 全 sim | 中模 + 降分辨率 sim | 蒙皮壳 |
| 碰撞 | 身体代理 + 自碰撞 CCD | 身体代理 | 无 |
| 约束 | 拉伸+弯曲+剪切+LRA+tether+strain limit | 拉伸+弯曲 | — |
| 空气动力学 | 逐三角升/阻 + 风场 | 全局风 | 无 |
| 褶皱 | 张力驱动动态褶皱法线 | 预烘焙褶皱贴图 | 无 |
| 着色 | Estevez-Kulla sheen + 薄透射 + fiber 多散 | Charlie sheen | 各向同性近似 |
| 透明 | OIT（薄纱）| 排序 alpha | 不透明近似 |
| 调度 | GPU 持久化 + 异步流水线 + 休眠 | 同步 sim | 静态 |

- 形变预算由 `deformation::schedule` 统一仲裁：单帧顶点上限 + BLAS refit 独立配额；多套服装超预算时最高优先项无条件先跑（防饿死），其余延后。休眠服装不占预算。
- **务实取舍**：基线只身体碰撞、Charlie sheen、预烘焙褶皱即可达"良好"；自碰撞 CCD + Estevez sheen + 张力褶皱是"顶级"的可选加法（承接材质设计 §10 风险条）。

---

## 9. 可测性

布料 sim / 约束 / LOD / 嵌入落"compute 可移植"桶（材质设计 §8.1）：数组进数组出，可写手写 Rust CPU golden 逐值对数。规划 `cloth/lod.rs` 首批确定性单测（对称 `hair/lod.rs` 13 测）：阈值分档、连续降分辨率、代理档不发 sim 请求、sim 档发 `DeformationKind::Cloth` 请求（`vertex_count = sim_vertex_count`）、分桶保序、越界跳过、空输入、`native_form` 钳制、`coarser_of` 秩比较。XPBD/VBD 约束投影可 golden（固定拓扑 + 固定子步 + 固定图着色序）；RT traversal 不可（驱动 BVH）——与全引擎三桶边界一致。

---

## 10. 落地路线图

1. **子系统骨架** `cloth/mod.rs`：`ClothPieceHandle`/`ClothPiece`/`ClothLodTier` + `is_simulated()`/`coarseness()`/`coarser_of()`（复用 `DeformationKind::Cloth`）。
2. **LOD** `cloth/lod.rs`：档选择 + 连续降分辨率 + `native_form` 钳制 + `bin_cloth_lod` + `cloth_deformation_request`；确定性 CPU golden（先落，最低冲突、与毛发对称）。
3. **约束构建** `cloth/constraints.rs`：拉伸（经纬各向异性）/弯曲/剪切/LRA/tether 约束图 + 图着色分批。
4. **布料动力学** `cloth/dynamics.rs`：XPBD 子步 + compliance + strain limiting，先仅身体碰撞，接 `deformation::schedule`；确定性 golden。VBD 高保真插槽随物理 §4 对齐后接入。
5. **碰撞** `cloth/collision.rs`：身体代理体 + 自碰撞（空间哈希/虚拟粒子/CCD）+ backstop。
6. **嵌入** `cloth/embed.rs`：render mesh 重心坐标/蒙皮嵌入。
7. **着色分叉**（材质侧）：cloth sheen/fuzz closure（Charlie → Estevez-Kulla）+ 薄透射 + `prism_render_npr` 风格化响应。
8. **高级项**：空气动力学/风场、绘制约束（backstop/max-dist/blend）、多层耦合、张力褶皱、GPU 持久化 + 异步流水线、撕裂/塑性（可选）。

**优先级**：先落骨架 + `cloth/lod.rs`（最低冲突、与毛发对称、`Cloth` 枚举已备），XPBD 动力学与自碰撞随物理内核对齐逐步点亮；每步 wgpu 可编译 + 单测绿。

---

## 11. 效果验收口径（v2 新增）

- **物理正确**：不超弹（strain limiting 生效）、不穿身体（backstop/碰撞生效）、多层不互穿（自碰撞生效）；静止收敛不抖。
- **AAA 质感**：紧身处贴合稳、飘逸处自然摆动（绘制约束调参到位）；缎面/丝绒有方向性 sheen；薄纱透光；褶皱随动作动态生成而非静态贴图。
- **NPR 观感**：卡通服装 ramp/手绘褶皱线达顶级二次元；运动下不被 TAA 抹糊（reactive mask 生效）。
- **性能**：GPU 持久化 + 异步下 sim 不阻塞主渲染；大量 NPC 服装靠休眠 + SkinnedProxy 降级可扩展。
- **跨平台**：sim/LOD/嵌入落 compute-可移植桶可 CPU golden；自碰撞 CCD 在弱 RT 平台仍纯 compute 可跑。

---

## 12. 风险

- **自碰撞 + 稳定性是重头**：算力/稳定性折衷大，需空间哈希与虚拟粒子先就位；子步数 vs 预算需折衷。
- **与物理内核的边界**：XPBD/VBD 求解原语应对齐 `prism_physics_core` §3/§4 统一内核，渲染侧 cloth 模块只负责服装几何/LOD/嵌入/形变调度，**不重复造求解器**。
- **NPR 布料无成熟范式**：顶级二次元服装褶皱多为美术手编 + 有限 sim，PBR 侧可抄业界、NPR 侧需自趟。
- **确定性要求**：联机/回放需固定子步 + 固定图着色顺序 + 固定种子（承接物理 §18）。
- **绘制约束美术成本**：backstop/max-distance/blend-weight 需 DCC 侧刷权重管线配套。

---

## 附录：术语

- **版片 / 缝合线**：2D 服装裁片与缝合边，缝合成 3D 服装（Marvelous 式）。
- **sim mesh / render mesh**：被模拟的低模 vs 跟随嵌入的高模外观网格。
- **LRA / tether / pressure**：长距离防拉伸 / 系绳固定点 / 压力体积约束。
- **strain limiting**：约束后二次钳制边长，防超弹。
- **backstop / max-distance / blend-weight / anim-drive**：绘制约束——背挡防穿身、位移上限、sim↔蒙皮混合、动画驱动。
- **compliance（柔度）**：XPBD 中代替 stiffness 的材料软硬量，与时间步无关。
- **CCD**：连续碰撞检测，防高速穿插隧穿。
- **VBD**：Vertex Block Descent，极刚/大形变高保真求解插槽。
- **sheen / fuzz / Estevez-Kulla**：布料绒面各向异性微高光模型（closure 瓣）。
- **FullSim / ReducedSim / SkinnedProxy 档**：LOD 阶梯；仅前两档做 sim。
