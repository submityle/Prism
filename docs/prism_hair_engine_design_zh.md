# Prism 渲染引擎 — 毛发引擎子系统完整设计（v2 / wgpu + WESL）

> 状态：架构提案（Draft，允许破坏性重构）
> 定位：AAA / 次世代 strand-based（发丝级）毛发引擎，与 PBR/NPR/自定义/混合四前端正交协同
> Shader：WESL（naga → WGSL/SPIR-V/Metal/DXIL）；运行时：wgpu（Metal/VK/DX12/WebGPU）
> 关联文档：`prism_rendering_architecture_zh.md`（总）、`prism_material_pipeline_design_zh.md`（§6.2 子系统注册表、§6.3 毛发 mini 支柱）、`prism_cloth_engine_design_zh.md`（对称子系统）、`prism_physics_design_zh.md`（§3 统一 XPBD、§4 多求解器、§11 GPU 持久化、§12 异步流水线）、`prism_aaa_advanced_features_zh.md`
> v2 变更：新增 §6 高级仿真特性、§7 毛发着色模型（Marschner/Chiang/dual-scatter/fiber-level + NPR 天使环），§8 性能升级为 GPU-driven 持久化 + 异步流水线，补 §11 效果验收口径；§0 产品对标细化（TressFX 4 / 影视 fiber-level / Frostbite）。

---

## 0. 定位：这是什么级别的毛发引擎

是**完整 strand-based（发丝级）毛发引擎**，不是"给头发一个各向异性高光"的着色贴皮。对标业界成熟产品，仅在**算法层**借鉴，不复制其代码：

| 产品 | 借鉴点（算法层，v2 细化） |
|---|---|
| **UE5 Groom** | Alembic groom 导入、guide→render 插值、compute 可见性光栅、deep opacity/voxel 透射、strand→card→mesh LOD、Niagara/物理驱动 groom sim、strand 软光栅 vis-buffer |
| **AMD TressFX（含 4.x）** | PPLL/OIT 发丝透明、Kajiya-Kay + Marschner 着色、局部/全局形状约束的 XPBD 式 strand 动力学、SDF 碰撞、近似自阴影 |
| **NVIDIA HairWorks** | 样条细分、密度 LOD、自阴影、guide 插值 |
| **影视 Marschner / d'Eon / Chiang** | 物理毛发 BSDF（R/TT/TRT 分量）、能量守恒近眼模型、吸收/髓质参数化 |
| **影视 Zinke dual-scattering** | 多重散射近似（浅色/金发的全局散射项）——顶级真实感必需 |
| **影视 fiber-level（Yan et al.）** | 逐纤维散射高保真——作为**高配着色插槽**（近景特写） |
| **Frostbite / God of War 系** | 主机预算下的 strand↔card 混合、density LOD、卡带宽的自阴影近似 |

**判据（承接材质设计 §6.2）**：一等子系统必须拥有**自己的几何 + 自己的 sim + 特殊渲染（透射/OIT/RT 代理）**。毛发三者全占，因此是一等子系统；而"毛发看起来的高光"只是 über-BSDF 的一个 closure 瓣，不构成子系统。**毛发与布料是两个独立子系统**（发丝几何+strand sim vs 三角网格+XPBD 布料 sim），仅共享形变预算这一资源仲裁层。

---

## 1. 与整体架构的关系（子系统内部同构范式）

毛发严格遵循"共享 GPU-driven 基底 + PBR/NPR 分叉响应"范式（材质设计 §6.3）：

```
共享基底（PBR/NPR/自定义/混合 都吃）:
    groom 资产 → guide strand + 插值 render strand（丛聚/卷曲/frizz）
    → guide-strand sim（XPBD：边长 + 局部/全局形状约束 + 碰撞）→ 形变顶点（汇入 gpu_scene）
    → 连续 LOD（strand→减股→card→mesh，禁硬切换 pop）
    → compute 发丝亚像素可见性软光栅
    → 共享 deep-transmittance 自阴影服务 + 插进共享 OIT

分叉响应（C 类，真分家）:
    PBR: Marschner/Chiang R/TT/TRT + Zinke dual-scattering 多重散射（浅色/金发必需）
    NPR: 风格化各向异性高光带（可多条）+ ramp + 阴影偏移（可与切线解耦=天使环/发环）

fallback:
    strand 高配、card 基线；RT 反射里毛发用 proxy 或排除
```

毛发**不拥有**光照/阴影/OIT/motion 这些公共设施——它**消费**基底服务，只在"几何+sim+响应"处分家。这是本引擎"管线级混合"优越性的来源。

---

## 2. 数据与资产模型

- **Groom 资产**：一组 guide strand（模拟源）+ 插值参数 + 每 strand 属性（根/尖半径、根 UV、随机化种子、卷曲/丛聚参数）。导入源对标 Alembic groom（UE5 同源）。导入时经 `hair/groom_import.rs` 归一化：任意非均匀控制点的 guide 折线按弧长重采样到统一定长（首末 pin 根/尖，退化输入不 panic），per-strand 属性（`StrandAttributes`：根/尖半径 taper、根 UV、seed）随几何前向携带，落成 strand-major 定长 stride 的 `ResampledGroom`，喂 dynamics 的 rest-length 边约束与 interpolation 的逐点混合。
- **Guide strand**：少量（数百~数千）被真正模拟的引导发丝；决定 sim 成本。
- **Render strand**：由 guide 在运行时插值出的大量（数万~数十万）渲染发丝；决定视觉密度与光栅成本，本身**不模拟**、只跟随插值形变。
- **绑定**：groom 绑定到蒙皮 mesh，根点随蒙皮移动（形变父级 = 材质设计 §6.1 蒙皮地基）。**已落** `hair/binding.rs`：导入期把每根 strand 根点用 Ericson 最近点投影到最近 scalp 三角，存重心坐标 + 沿面法向的有符号高度（`MeshBinding`，拓扑稳定、越界/无三角=`UNBOUND` 不丢根）；运行时用同拓扑的蒙皮形变顶点逐帧重解根世界位置与右手正交根帧（`RootFrame`），喂 dynamics 的 pinned 根粒子（`inverse_mass=0`）与根形状约束的朝向；静止/退化输入恒等兜底不 panic，rest 顶点重解精确复现原根位（golden 往返性质）。

代码落点（`prism_render_architecture::hair`）：
- `HairGroupHandle`、`HairGroup{ guide_strand_count, max_render_strands, segments_per_strand, native_form, deformation }`（**已落** `hair/mod.rs`）。
- `HairLodTier{ Strands, ReducedStrands, Cards, Mesh }` + `is_strand_based()`/`coarseness()`/`coarser_of()`（**已落**）。

---

## 3. 管线阶段（端到端）

| 阶段 | 内容 | 对标 | 代码落点 |
|---|---|---|---|
| 1. 导入/插值 | groom 导入归一化（guide 弧长重采样到定长控制点 + per-strand 属性携带）→ guide→render strand 插值（丛聚/卷曲/frizz/随机化） | UE5 Groom / Alembic | **已落** `hair/groom_import.rs`（导入弧长重采样，喂 dynamics/interpolation）+ `hair/interpolation.rs`（插值）|
| 2. Strand 动力学 | guide 的 XPBD sim（边长 + 局部/全局形状约束 + LRA + 代理体碰撞可选） | TressFX | **已落** `hair/dynamics.rs`（→ `deformation::schedule`）+ `hair/collision.rs`（sphere/capsule 代理体）+ `hair/binding.rs`（groom 根绑定蒙皮 scalp，逐帧重解根世界变换）|
| 3. LOD | 覆盖度选档 + 连续减股 + 跨档 dither 过渡，禁硬切换；Cards 档从 strand 相干帧网格化代理带 | UE5/HairWorks | **已落** `hair/lod.rs` + `hair/transition.rs`（跨档 crossfade + per-strand dither）+ `hair/ribbon.rs`（strand→ribbon/card 网格化）+ `hair/mesh_shell.rs`（Cards→Mesh 最远档矩形箱体管壳，补全 Strands→减股→Cards→Mesh 几何梯）+ `hair/decimation.rs`（重要度加权嵌套抽稀序，修复 stride 采样跨档 pop）+ `hair/density_lod.rs`（guide 几何→binding 重要度传播→importance 抽稀端到端密度 LOD 驱动）|
| 4. 光栅 | 发丝亚像素 compute 软光栅进 visibility；per-vertex 相干帧供 ribbon/card 宽度 + 各向异性切线 | UE5 Groom / TressFX | **已落** `hair/raster.rs`（挂 `virtual_geometry` 软光栅桶）+ `hair/frames.rs`（双反射 RMF 旋转最小化帧）|
| 5. 着色 | HairPbr 闭包（Chiang/Marschner）+ dual-scattering（见 §7）| 影视 BSDF | `material` 的 `HairPbr` closure（不在 hair 内重写）|
| 6. 透射/自阴影 | deep opacity map / voxel 透射 | UE5/TressFX | **已落** `hair/deep_transmittance.rs`（per-ray 透射服务）+ `hair/deep_opacity_layout.rs`（跨纹素定长层 slab host 打包）|
| 7. 透明合成 | OIT，走 `HairVisibility` 路径 | TressFX PPLL | **已落** `transparency::routing`（`TransparentKind::Hair`）|

**关键边界**：着色（阶段 5）走材质系统的 `HairPbr` closure，透明（阶段 7）走共享 OIT——毛发模块**不重写**着色与透明，只产几何、形变与透射。sim/LOD/光栅落"compute 可移植"桶，可 CPU golden。

---

## 4. LOD 策略（禁 pop 硬切换）

四档：`Strands`（全发丝，做 sim）→ `ReducedStrands`（减股，做 sim）→ `Cards`（卡片）→ `Mesh`（网格壳）。仅 `is_strand_based()` 档（`Strands`/`ReducedStrands`）做 sim 与发形变请求。

**`native_form` 钳制**：授权某 groom 的原生形态——远景/背景角色授权 `Cards` 就是原生形态而非降级（消除 PBR 偏见，NPR/二次元卡片授权同理）。`resolve_hair_lod` 用 `tier.coarser_of(group.native_form)` 把覆盖度选出的档钳制成"不比授权更细"：满屏也不会把卡片 groom 提升到它根本没有的 strands，但距离拉远仍可继续降到 mesh。**授权几何选择，与 PBR/NPR 着色响应正交**，两种风格同等遵循。

**已落**：`select_hair_lod_tier` / `resolve_hair_lod`（含 `native_form` 钳制）/ `bin_hair_lod`（按档分桶，越界跳过不 panic，保输入序）/ `hair_deformation_request`（strand 档→ `DeformationKind::Hair`，`vertex_count = render_strands × segments`，`needs_blas_refit=true`）。分桶对齐 `virtual_geometry/bins.rs` 黄金范式。

---

## 5. PBR / NPR / 自定义 / 混合 分叉响应

- **PBR**（对标 UE5 Groom、film-grade）：R/TT/TRT 三分量 + dual-scattering 多重散射，各向异性沿切线；deep opacity map 自阴影，有 RT 时走硬件光线自阴影（详见 §7）。**（已实现）** 物理 Marschner R/TT/TRT + Zinke dual-scattering closure 落 `prism_render_shading::hair` + `hair.wesl` twin（§10 item6）。
- **NPR**（对标 miHoYo / Arc Sys）：风格化各向异性高光带（可多条）+ ramp + 阴影偏移，高光位置**可与真实切线解耦**（天使环/发环风格）；描边复用 vis-buffer 的 material id 边界（`native_form=Cards` 的卡片发同样吃 material id 描边）；per-strand 或 per-card 顶点色控高光/阴影段。 **（已实现）** 风格化高光带 + cel ramp + 不受阴影 rim closure 落 `prism_render_shading::stylized_hair` + `stylized_hair.wesl` twin（§10 item6）。
- **自定义**：项目注入 `illumination=Custom` 的 WESL closure。
- **混合**：同一 groom 可按 material id 分 tile 路由到不同前端（因共享光/影/GI 而连贯）；写实场景里的卡通角色发与 PBR 世界吃同一份 GI/阴影，不断裂。

四者共享 §1 全部基底（同一份 strand 几何、sim、LOD、透射），**只在着色响应处分家**——四者都能享受发丝级几何与自阴影。

---

## 6. 高级仿真特性（v2 新增，含性能/效果取舍）

> 顶级 strand 动力学与"能动的样条"的分水岭。逐条：算法 / 产品对标 / 成本 / 效果 / 落点。

1. **局部 + 全局形状约束（XPBD）**：局部约束保发型段间刚度、全局约束把发丝拉回原始造型（TressFX 核心）。成本：每 guide 逐段一趟；效果：发型不散架、甩动后回弹。落点 `hair/dynamics.rs`。
2. **碰撞（capsule/sphere/SDF + 自碰撞近似）**：身体代理体碰撞 + SDF 碰撞（TressFX 4）；发丝自碰撞用体素/近似（全对测太贵）。成本：SDF 采样低、自碰撞中；效果：头发不穿脸/肩。
3. **风场耦合**：全局风 + 局部风场/湍流噪声，与布料/粒子共享风数据。成本：低；效果：环境一致的飘动。
4. **子步 substepping + compliance**：子步多、单步少迭代，能量更稳、刚度更高、与时间步无关（物理 §3.3/§3.4）——快速甩头不炸毛的地基。
5. **LRA / tether（长距离约束）**：防发丝过拉伸失控。成本：低；效果：高速运动不拉丝。
6. **guide 插值模式（丛聚/卷曲/frizz）**：clump（成绺）、curl（卷曲）、frizz（毛躁）程序化参数，render strand 由 guide 插值 + 噪声偏移。成本：光栅前一趟；效果：自然发束层次而非均匀分布。
7. **多求解器插槽（XPBD 基线 / VBD 高保真）**：极端刚度（辫子/发胶造型）可切 VBD（物理 §4）。成本：更贵；效果：硬造型不软塌。
8. **GPU-driven 持久化 + 异步流水线**：guide 状态常驻 GPU、约束图着色分批，sim 与渲染解耦异步跑（物理 §11/§12）——摊平尖峰、不阻塞主渲染。
9. **休眠 / 激活**：静止角色停 sim、被扰动再激活。成本：负（省算力）；效果：大量 NPC 毛发可扩展。

---

## 7. 毛发着色模型（v2 新增，材质系统 closure 侧）

> strand sim 产几何与透射，**着色由材质系统 `HairPbr` closure 承载**（本模块不重写）。这里列 closure 目标形态与产品对标。

1. **Kajiya-Kay 兜底档（已实现）**：各向异性高光基线（实时最便宜档，降级矩阵兜底）。效果：基础发丝高光。**已落**：`prism_render_shading::hair_kajiya` CPU golden + `hair_kajiya.wesl` GPU twin——单各向异性 `sin(T,L)` 圆柱漫反射（base 着色，垂直切线峰值）+ 单 shifted 各向异性 specular（Kajiya & Kay 1989 的 `(T'·L)(T'·V)+sin(T',L)sin(T',V)` 的 exponent 次幂，切线沿法线 tilt `specular_shift` 作 cuticle-tilt 廉价替身，white↔base 由 `specular_tint` 混）；无吸收/透射/髓质，最省；6 确定性单测绿 + WESL naga 编译并解析 brdf import 绿（`shader_tests::hair_kajiya_wesl_compiles_and_resolves_imports`）。着色五档全齐（兜底 Kajiya-Kay → 基线 Marschner → 近眼 Chiang → 特写 fiber-level → NPR 风格化）。
2. **Marschner R/TT/TRT**：反射 R + 透射 TT + 内反射 TRT 三分量物理毛发 BSDF。效果：真实主高光 + 次级彩色高光。
3. **Chiang 近眼模型（已实现）**：能量守恒、吸收/髓质参数化，特写更准。效果：影视级近景发丝。**已落**：`prism_render_shading::hair_chiang` CPU golden + `hair_chiang.wesl` GPU twin——色→σ_a 吸收拟合（Chiang 2016 eq.9，越黑越吸收、能量守恒，区别于 baseline 的 pow-tint）+ Beer-Lambert 皮质透射 + R/TT/TRT 三 cuticle lobe + 空心髓质各向同性前散射填充（`medulla_ratio`/`medulla_scatter` 控）；6 确定性单测绿 + WESL naga 编译并解析 brdf import 绿（`shader_tests::hair_chiang_wesl_compiles_and_resolves_imports`，见 §10 item6）。待续：fiber-level 逐纤维特写。
4. **Zinke dual-scattering 多重散射**：全局散射项——浅色/金发**必需**（否则发黑失真）。成本：预计算/近似一趟；效果：金发通透不发死。
5. **fiber-level（Yan et al.，高配插槽，已实现）**：逐纤维散射，近景特写最高保真。成本：贵，仅特写；效果：顶级真实感。**已落**：`prism_render_shading::hair_fiber` CPU golden + `hair_fiber.wesl` GPU twin——Yan et al. 2015 双圆柱（外皮质 + 内散射髓质）：R/TT/TRT 三 cuticle lobe（复用 Chiang σ_a 吸收/Beer-Lambert）+ 两个髓质**散射 lobe**（`TTs`/`TRTs`：加宽纵向 Gaussian + `Henyey-Greenstein` 方位相函数 + 核吸收 + κ·albedo 权重），`medulla_ratio=0` 时优雅退化为三 cuticle lobe；6 确定性单测绿 + WESL naga 编译并解析 brdf import 绿（`shader_tests::hair_fiber_wesl_compiles_and_resolves_imports`，见 §10 item6）。着色五档（Kajiya-Kay 兜底/Marschner 基线/Chiang 近眼/fiber-level 特写/NPR 风格化）齐全。
6. **deep opacity map / voxel 透射自阴影**：发丝间自阴影透射累积（阶段 6 产出，着色消费）。效果：发内体积阴影。
7. **NPR 对偶（已实现基线）**：风格化各向异性高光带（可多条、可与切线解耦=天使环）+ ramp + 阴影偏移 + MatCap；消费同一 strand 切线/覆盖数据。效果：顶级二次元发丝（原神/GG Xrd）。**已落**：`stylized_hair` CPU golden + `stylized_hair.wesl` GPU twin（wrap cel ramp + 双 shifted `Kajiya-Kay` 高光环 + 不受阴影 `Fresnel` rim，见 §10 item6）；MatCap/多环扩展与 material-id 描边接线待后续。

**RT 侧**：有硬件 RT 时毛发可走光线自阴影/反射代理；弱 RT 平台降级 deep opacity（compute 可跑，跨平台）。

---

## 8. 性能预算与降级矩阵（v2 升级：GPU-driven + 异步）

**GPU-driven 持久化**（物理 §11）：guide sim 状态常驻 GPU、约束图着色分批（组内 Jacobi 并行、组间 Gauss-Seidel 串行）。**异步流水线**（物理 §12）：sim 与渲染解耦、双缓冲交形变顶点给 `gpu_scene`，摊平尖峰。

> **进度（GPU sim kernel 契约已落）**：guide-XPBD 仿真 compute kernel 已以 `hair_sim.wesl` 落地并通过 naga 编译验证（CPU `dynamics.rs` 的忠实 twin，见 §10.8）。当前为契约优先——kernel 本身完备，但接进渲染图（dispatch/双缓冲/`gpu_scene` 顶点交换）属跨子系统调度，待并发线稳定后接线。

| 能力 | 高配 | 基线 | 兜底 |
|---|---|---|---|
| 几何 | 全 strand 软光栅 | card | mesh 壳 |
| sim | guide XPBD + SDF 碰撞 + 自碰撞近似 + 风场 | guide XPBD 仅身体碰撞 | 静态（root 跟蒙皮）|
| 求解器 | VBD（硬造型）/ XPBD 多子步 | XPBD 少子步 | 无 sim |
| 着色 | Chiang + dual-scatter（+fiber 特写）| Marschner R/TT/TRT | Kajiya-Kay |
| 自阴影 | deep opacity/voxel 透射 | 近似 dither 阴影 | 参与普通阴影图 |
| 透明 | OIT（PPLL/moment）| 排序 alpha | alpha test |
| RT 反射 | proxy 参与 | 排除（只主视图）| 排除 |
| 调度 | GPU 持久化 + 异步 + 休眠 | 同步 sim | 静态 |

- strand + deep transmittance 是 AAA 最重特性：**card 基线务实、strand 高配可选**，别一上来全 strand（承接材质设计 §10 风险条）。
- 形变预算由 `deformation::schedule` 统一仲裁：单帧顶点上限 + BLAS refit 独立配额；密集 groom 超预算时最高优先项无条件先跑（防饿死），其余延后。休眠 groom 不占预算。

---

## 9. 可测性

毛发 sim / LOD / 光栅落"compute 可移植"桶（材质设计 §8.1）：数组进数组出，可写手写 Rust CPU golden 逐值对数。当前 `hair/lod.rs` 已有约 13 个确定性单测（阈值分档、抽稀因子、代理档丢几何、strand 档发形变请求、分桶保序、越界跳过、空输入、`native_form` 卡片钳制、`coarser_of` 秩比较）。着色 BSDF closure 可 golden；RT traversal 不可（驱动 BVH）——与全引擎三桶边界一致。

---

## 10. 落地路线图

1. **（已完成）子系统骨架**：`hair/mod.rs` 核心类型（含 `native_form`）+ `hair/lod.rs` LOD 阶梯/分桶/形变绑定/钳制（commit 已落，单测绿）。
2. **（已完成）strand 动力学** `hair/dynamics.rs`：guide XPBD（边长 + 局部/全局形状约束 + LRA/tether），先无碰撞，接 `deformation::schedule`；确定性 CPU golden，单测绿。 **（根绑定蒙皮已落）** `hair/binding.rs`：groom 根随蒙皮头动（`UE5` Groom binding asset / `TressFX` 根蒙皮范式）——`bind_roots` 导入期对每根做 Ericson `closest_point_on_triangle` 投影到最近 scalp 三角（暴力扫描，ties 取最小三角 idx 确定性），存 `MeshBinding{triangle,bary,height}`（重心 + 沿面法向有符号高度，拓扑稳定、无三角/越界=`UNBOUND` 保根不丢、索引对齐 strand 数组）；`resolve_root_frames`/`resolve_root_positions` 逐帧用同拓扑蒙皮形变顶点重建根世界位（重心插值 + 法向偏移）与右手正交根帧 `RootFrame{position,tangent,normal,bitangent}`，位置喂 dynamics pinned 根粒子、根帧旋转 strand 授权 rest 形态；rest 顶点重解精确复现原根位（往返 golden 性质）、刚体平移根随之同移、空 mesh/越界/零面积面全部恒等兜底不 panic；9 确定性单测绿（在面高度≈0 & 重心和=1/往返复现根位/刚体平移根跟随/根帧正交且法向=面法向/空 mesh 全 `UNBOUND` 不 panic/越界解恒等/退化零面积面 finite 不 panic/外部投影钳到边/多三角取最近且重绑位精确幂等）。设备端 GPU 蒙皮重解同一契约属独立调度，此处只落 CPU golden 几何契约。
3. **（已完成，CPU golden + GPU 插值双片齐）插值**：**（导入归一化前置步已落）** `hair/groom_import.rs`——任意非均匀控制点的 authored guide 折线按弧长重采样到统一定长（`resample_strand` 首末 pin 根/尖、退化/空/单点/全重合输入不 panic；`resample_groom` 批量 flat 缓冲 + per-strand range 越界 skip 不 panic，产 strand-major 定长 stride 的 `ResampledGroom`），per-strand 属性 `StrandAttributes`（根/尖半径 taper + `radius_at`、根 UV、seed）随几何前向携带，`strand_rest_lengths` 派生喂 dynamics XPBD 边约束的 per-segment rest length（8 确定性单测绿：pin 根尖/均匀弧长间距/非均匀重参数化/target 钳 2/空&单点&重合不 panic/批量定长+属性携带/越界 skip/密度无关幂等）；确保 dynamics 与 interpolation 的定长控制点前提成立。CPU 侧 `hair/interpolation.rs` guide→render 丛聚/卷曲/frizz/随机化，与 LOD 抽稀联动（确定性 CPU golden，单测绿）；GPU 侧 **`prism_render_scene/src/shaders/hair_interp.wesl`** 落地 compute twin——一 invocation 一根 render strand，逐值镜像 CPU 变换链（加权混合→长度抖动→丛聚→卷曲→逐点位置抖动），curl/jitter 从 root(0)→tip(full) 线性增长；确定性哈希 `splitmix64` 以 `vec2<u32>`(lo,hi) 手写 64-bit 加/异或/右移/低 64 位乘法逐位复刻（baseline WGSL 无原生 u64），`sin_turns` 沿用 CPU 范围规约 9 阶 Taylor（禁 `f32::sin`，`round_away` 半数远零对齐 Rust `f32::round`），故 CPU golden 直接验其数学；buffer 布局 flat `guide_points: array<vec4<f32>>`（.xyz 控制点）+ 每 guide `guide_ranges`(offset,count) 切片 + `bindings`(4 guide idx + 4 weight + root_uv + seed + out_offset) + out `out_points: array<vec4<f32>>` + `var<immediate> HairInterpParams`；naga 编译验证绿（`shader_tests::hair_interp_wesl_compiles_standalone`）。
4. **（已完成，CPU 分类 + GPU 光栅二片齐）发丝光栅**：CPU 侧 `hair/raster.rs` 亚像素软光栅**分类/分桶** ABI（沿用 `virtual_geometry/bins.rs` 范式，`HairSoftRasterAbi`=subpixel_samples 8/coverage_epsilon/tile_size 16/max_segments_per_tile 256，确定性单测绿）；GPU 侧物理光栅 **`prism_render_scene/src/shaders/hair_raster.wesl`** 落地——发丝段作屏幕空间 capsule 散射进逐像素 vis-buffer，双 kernel 无锁最近片解算（`hair_raster_depth` atomicMin 量化深度 + `hair_raster_resolve` 深度胜出者发布 seg_id/coverage，绕开 baseline WGSL 无 64-bit 原子的限制，Nanite 式 vis-buffer）；解析 capsule 亚像素覆盖（`subpixel_samples` 分层采样 + 1px AA 带）、`coverage_epsilon` 剔除、radius 扩张 AABB 扫描；naga 编译验证绿（`shader_tests::hair_raster_wesl_compiles_standalone`）。tiled 分桶（tile_size/max_segments_per_tile）属后端 dispatch 调度，不改逐像素覆盖数学。 **（相干帧工具已落）** `hair/frames.rs`：沿 strand 用**双反射旋转最小化帧**（Wang 2008 RMF）从根帧逐点传播参考法线——`strand_tangents` 前向差分单位切线 + `build_strand_frames` 两次反射（点偏移平分面 + 切线对齐）产 per-vertex 右手正交帧 `StrandFrame{tangent,normal,bitangent}`，避免朴素逐点基在切线越过 helper 轴处翻转导致 ribbon 自折/闪烁；`StrandFrame::ribbon_edges(center,half_width)` 沿 bitangent 对称展开成 card/ribbon 两边缘，供 strand→card LOD 与各向异性着色切线消费；8 确定性单测绿（帧数=点数/直线帧恒定不扭/切线前向差分/平面曲线出平面法线恒定即 RMF 最小旋转性质/空&单点不 panic/重合点不 panic 仍正交/ribbon 对称跨宽/位精确幂等）。设备端可作 compute twin，此处只落 CPU golden 的几何契约。 **（Cards 代理网格化已落）** `hair/ribbon.rs`：消费 `frames.rs` 的 per-vertex 相干帧把 strand 折线网格化成 Cards LOD 档的代理带几何（TressFX/UE5 Groom 卡片渲染范式）——`build_ribbon(points,frames,radii)` 沿每点 ±bitangent·radius 双边缘展开（left=2i/right=2i+1 的 SoA `RibbonMesh{positions,tangents,uvs,indices}`，2 顶点/控制点、triangle-list 6·(n-1) 索引、弧长归一化 v 坐标），`build_ribbon_tapered` 用 `StrandAttributes::radius_at` 沿归一化弧长做根/尖 taper 宽度；view-independent（帧已定，非广告牌）故可 CPU golden 逐值对，长度不匹配/<2 点/空输入产空 mesh 不 panic；8 确定性单测绿（顶点/三角计数=2n 与 6(n-1)/双边缘沿 bitangent 对称跨 2·radius/弧长 v 单调 0→1/taper 根尖半径命中/退化&空输入空 mesh 不 panic/位精确幂等）。GPU 侧卡片网格化 compute twin **`prism_render_scene/src/shaders/hair_ribbon.wesl`** 已落——一 invocation 一根 strand，两趟（弧长累加驱动 v + ±bitangent·radius 边缘展开）逐值镜像 `build_ribbon`，按 vertex_offset/index_offset 批量切片、索引偏置 vertex_offset 使共享输出池正确寻址（offset=0 时与单丝 CPU golden 位同构，故该 golden 直接验此 kernel），naga 编译验证绿（`shader_tests::hair_ribbon_wesl_compiles_standalone`），喂 Cards 档 vis-buffer 与各向异性着色切线。 **（Mesh 档壳网格化已落）** `hair/mesh_shell.rs`：LOD 几何梯最远一级——远景发丛不可辨单丝时把 strand 塌成低面数实心壳（对齐 UE5 Groom strand→card→mesh 的 mesh 档）。沿中心线逐控制点扫出四角矩形截面（bitangent 张 half_width、normal 张 half_thickness），逐段缝成封闭箱体管 + 根/尖平封盖；矩形截面刻意免三角函数、每环仅 4 顶点、对角逐顶点外法线对远景足够。`ShellMesh{positions,normals,uvs,indices}` SoA（vertex_count=4n/triangle_count=8(n-1)+4/is_empty），`build_shell(points,frames,half_width,half_thickness)` 均匀截面 + `build_shell_tapered` 用 `StrandAttributes::radius_at` 沿归一化弧长逐环锥缩；退化（<2 点/长度不匹配）产空 mesh 不 panic；9 确定性单测绿（顶点=4n/三角=8(n-1)+4/角点落截面箱 extent 且沿切线零漂移/对角角点关于中心线对称/法线单位长/uv 周长四分与弧长 v 单调/锥缩命中根尖半径/位精确幂等/索引不越界/空&单点&长度不匹配空 mesh）。GPU 侧壳网格化 compute twin **`prism_render_scene/src/shaders/hair_mesh_shell.wesl`** 已落——一 invocation 一根 strand，两趟（弧长累加驱动 v + 逐环四角矩形截面 ±bitangent·half_width/±normal·half_thickness 展开、对角外法线 `normalize_or` 兜底、八侧三角/段 + 根尖平封盖 index 偏置 vertex_offset）逐值镜像 `build_shell`，corner 符号用 `select` 算术推导（免 const 数组动态索引、保持纯 const-expression twin），offset=0 时与单丝 CPU golden 位同构故该 golden 直接验此 kernel，naga 编译验证绿（`shader_tests::hair_mesh_shell_wesl_compiles_standalone`）。view-independent 可 CPU golden 逐值对，喂 Mesh 档 vis-buffer 与各向异性着色切线。
5. **（已完成，CPU golden + GPU voxel 双片齐）deep transmittance**：CPU 侧 `hair/deep_transmittance.rs` deep opacity 分层透射 + voxel 近似 + per-texel 分桶（确定性 golden，单测绿）；GPU 侧 **`prism_render_scene/src/shaders/hair_transmittance.wesl`** 落地 voxel 路径的 compute twin——一 invocation 一条光线 texel，按 `ranges[texel]` 切片读 flat `samples`，把 in-slab 不透明度散射进定长 voxel slab 的光学密度 `sigma`，再以 running `product(1-sigma_j)` 输出单调不增透射曲线（sort-free/exp-free，与 CPU `accumulate_voxel_density`+`voxel_transmittance` 逐值同构，故 CPU golden 直接验其数学）；naga 编译验证绿（`shader_tests::hair_transmittance_wesl_compiles_standalone`）。**（host 分层布局构建步已落）** `hair/deep_opacity_layout.rs`：把 `bin_samples` 的 per-texel 桶跨纹素打包成上传就绪的定长层 slab——`build_deep_opacity_map(bins,layer_count,start_offset)` 对每纹素做稳定深度排序 + 等宽分层 + `product(1-alpha)` 复合（与 per-ray `build_deep_opacity` 逐值同构，单测交叉对拍），产 texel-major/layer-minor 扁平 `transmittance` 网格 + per-texel `near_depth`/`layer_step` 三平行缓冲（各纹素自括其发丝深度跨度、恒定 stride 可上传），空桶=全透射行；`DeepOpacityMap::layers`/`map_transmittance` 供着色侧按接收深度解码（越界/退化纹素全亮不 panic，深度单调不增，无 exp/确定性）；6 确定性单测绿。设备端 GPU 排序构建同一 slab 属独立调度、不在此拥有。接共享阴影/OIT。
6. **着色分叉**（材质侧）：**（PBR 侧已完成）** `prism_render_shading::hair`（CPU golden）+ `prism_render_scene/src/shaders/hair.wesl`（GPU twin）落地物理 **Marschner R/TT/TRT**（Karis 能量守恒实时式，UE 算法授权借鉴）+ **Zinke dual-scattering** 多重散射填充（浅色/金发通透不发死）：R 白色主高光 + TT 背光透射 rim + TRT 彩色次高光 + 逐 lobe 纵向 Gaussian/cuticle-tilt/Fresnel/吸收着色；纵向宽度随 `perceptual_roughness`，emissive 只加一次；6 个确定性 golden 单测绿 + WESL naga 编译绿。**（NPR 侧已完成）** `prism_render_shading::stylized_hair`（CPU golden）+ `prism_render_scene/src/shaders/stylized_hair.wesl`（GPU twin）落地风格化**天使环各向异性高光带**：wrap 半 Lambert 漫反射量化成 `bands` 级 cel ramp（可软化边）+ 两条 shifted 阈值化 `Kajiya-Kay` 各向异性高光带（近根白色主环 + 近尖着色次环，各过 `smoothstep` 成锐利墨线发环，高光位置沿法线偏移与真实切线解耦）+ 不受阴影的视角 `Fresnel` rim（背光轮廓保持可读），消费同一 strand 切线/覆盖数据、emissive 只加一次；6 个确定性 golden 单测绿（matte 纯 ramp / ramp 量化 ≤ bands 级 / 高光加能量 / 遮挡只剩 emissive / 背面只剩 emissive / 幂等）+ WESL naga 编译并解析 brdf import 绿（`shader_tests::stylized_hair_wesl_compiles_and_resolves_imports`）。**（Chiang 近眼高配已完成）** `prism_render_shading::hair_chiang`（CPU golden）+ `prism_render_scene/src/shaders/hair_chiang.wesl`（GPU twin）落地 **Chiang 2016 能量守恒近眼毛发 BSDF**：由 base color + 方位粗糙度拟合每通道吸收系数 σ_a（eq.9，越黑吸收越强、透射 lobe 随之衰减，物理正确、区别于 baseline 的 pow-tint 残留）+ Beer-Lambert 皮质透射 `T=exp(-σ_a·seg)` + R（白色 Fresnel glint）/TT（`(1-F)²·T`）/TRT（`(1-F)²·F·T²`）三 lobe + 可选空心髓质各向同性前散射填充（`sqrt(T)` 半路径×wrap×INV_PI，`medulla_ratio`/`medulla_scatter` 参数化灰发/兽毛），`longitudinal_roughness>0` 时可与表面粗糙度解耦，emissive 只加一次；6 确定性单测绿（跨参数含髓质 finite&非负 / 遮挡只剩 emissive / 越黑越吸收 / 髓质加填充 / 黑发仍有白 R glint / 幂等）+ WESL naga 编译并解析 brdf import 绿（`shader_tests::hair_chiang_wesl_compiles_and_resolves_imports`）。**（fiber-level 逐纤维特写高配已完成）** `prism_render_shading::hair_fiber`（CPU golden）+ `prism_render_scene/src/shaders/hair_fiber.wesl`（GPU twin）落地 **Yan et al. 2015 双圆柱 fur/hair BSDF**：外皮质 + 内散射髓质双圆柱，三 cuticle lobe（R/TT/TRT，复用 Chiang σ_a 色→吸收拟合 + Beer-Lambert 皮质透射 + `(1-F)²`/`(1-F)²·F` 权重）之上叠加两个髓质**散射 lobe**——`TTs`（散射前向透射：加宽纵向 Gaussian `beta_s=r²(1+3κ)` + `Henyey-Greenstein` 方位相函数 `hg(-cosφ,g)` 前向峰 + 核吸收 `exp(-σ_med·(0.5+κ))` + κ·albedo 权重，用皮质半路径 `sqrt(T)`×base 着色，即灰发/兽毛的柔和去饱和前向辉光）与 `TRTs`（多一次内反射，F 加权，全皮质路径 `T`×base 着色）；`medulla_ratio=0` 两散射 lobe 归零、优雅退化为三 cuticle lobe，emissive 只加一次；6 确定性单测绿（跨参数含 κ&g finite&非负 / 遮挡只剩 emissive / 髓质加散射填充 / 前向各向异性偏前散射 / 越黑越吸收 / 幂等）+ WESL naga 编译并解析 brdf import 绿（`shader_tests::hair_fiber_wesl_compiles_and_resolves_imports`）。**（Kajiya-Kay 兜底档已完成）** `prism_render_shading::hair_kajiya`（CPU golden）+ `prism_render_scene/src/shaders/hair_kajiya.wesl`（GPU twin）落地 **Kajiya & Kay 1989 各向异性 strand 着色**（降级矩阵最便宜兜底档）：单各向异性 `sin(T,L)` 圆柱漫反射（base 着色，光垂直切线时峰值）+ 单 shifted 各向异性 specular（`(T'·L)(T'·V)+sin(T',L)sin(T',V)` 的 `specular_exponent` 次幂，切线沿法线 tilt `specular_shift` 作 cuticle-tilt 廉价替身，高光 white↔base 由 `specular_tint` 混），无吸收/透射/髓质、emissive 只加一次；6 确定性单测绿（跨参数 finite&非负 / 遮挡只剩 emissive / diffuse 垂直切线峰值 / specular 加能量 / 越尖 exponent off-peak 越暗 / 幂等）+ WESL naga 编译并解析 brdf import 绿（`shader_tests::hair_kajiya_wesl_compiles_and_resolves_imports`）。**着色五档全部落地（兜底 Kajiya-Kay + 基线 Marschner + 近眼 Chiang + 特写 fiber-level + NPR 风格化）。**
7. **（已完成）碰撞与连续 LOD 过渡** `hair/collision.rs` + `hair/transition.rs`：sphere/capsule 代理体碰撞（每子步约束后投影，pinned 不动，空集 no-op）集成进 `dynamics.rs`；`resolve_hair_lod_transition` 在阈值过渡带内计算相邻档 blend + `strand_survives_dither` 确定性 per-strand screen-door 抖动淡入淡出消 pop（尊重 `native_form` 钳制）；确定性 CPU golden，单测绿。 **（重要度加权嵌套抽稀已落）** `hair/decimation.rs`：把连续密度 LOD 的减股从 `interpolation::decimate_bindings` 的 stride 采样（仅整数倍档嵌套、任意计数换 strand 而 pop）升级为**重要度加权的嵌套抽稀序**——按长度/曲率/authored 优先级加权混合 + per-strand hash jitter 空间去相关，产一条稳定 permutation，其**每前缀即有效 kept 集**（keep 前 k ⊂ keep 前 k+1），任意 target 计数零 pop（对齐 UE5 Groom / HairWorks 密度 LOD「重要丝存活到最低档」偏置，重抽稀时均匀减薄而非剃出秃斑）；免超越函数（仅 +/*/sqrt）、确定性、空/长度不匹配输入优雅退化不 panic。**已接线**：`interpolation::decimate_bindings`(减股档消费入口)已从 stride 采样切到委托 `decimate_bindings_nested`(几何无关的纯 seed hash 嵌套排序，`DEFAULT_DECIMATION_JITTER`)，任意 target 计数零 pop、保持 out 清空/proxy 空/满档全保语义；有几何时改走 `decimate_bindings_importance` 得重要度加权存活；共 14 关联测试绿(含 10⊂20、7⊂13 跨非整数倍档嵌套子集不变量)。**（几何驱动端到端已落）** `hair/density_lod.rs`：把上面 decimation 的重要度机制接上真实 groom 几何——`guide_metrics` 逐 guide 从重采样控制点测弧长 + 曲率、root_radius 按全局 max 归一化作 authored 厚度优先级[0,1]；`binding_importances` 按 binding 的 4 guide 权重把 guide 度量混到每根 render strand（越界 guide 索引零贡献）再经 `compute_importance` 归一到[0,1]；`plan_density_lod` / `plan_density_lod_default` 一站式 measure→propagate→`decimate_bindings_importance`，让 UE5 Groom「近景保全、远景按重要度存活」端到端可用（跨 dynamics 存储 / interpolation 度量两套 Vec3 逐点转换一次、免超越函数、确定性、5 golden 单测绿：长度/厚度归一/长丝重要度更高/最长丝存活到 count1/7⊂19 跨非整数倍档嵌套 + 确定性/越界 + 空 + proxy 优雅退化不 panic）。自碰撞近似与 SDF 碰撞均已在此基础分层落地（见 item8 `self_collision` / `sdf_collision`）。
8. **高级项（逐条落地中）**：
   - **（已完成）风场耦合** `hair/wind.rs`：`WindField`（方向/风速 + gust 脉动 + per-axis flutter 湍流）→ `wind_acceleration` / `apply_wind`（free 粒子加 accel·dt² 位移，pinned 跳过，dt<=0 no-op），手写 Taylor `sin_turns`；确定性单测绿（对应 §6.3）。
   - **（已完成）休眠 / 激活门控** `hair/sleep.rs`：`groom_motion_energy`（Σ 隐式速度平方）+ `SleepThresholds`/`GroomSleepState` 迟滞门（wake 优先、quiet 连续帧累积到 `frames_to_sleep` 才睡、dead band 保持、NaN 保持唤醒）；`should_simulate` 供调度层跳过静止 groom；确定性单测绿（对应 §6.9 / §8 休眠不占预算）。
   - **（已完成）RT 反射代理策略** `hair/rt_proxy.rs`：`RtReflectionRole{FullStrands,Proxy,Excluded}` + `RtProxyPolicy{min_coverage_for_proxy,allow_strands_in_rt}` → `resolve_rt_role(tier,coverage,policy)`（覆盖度门下排除；strand 档默认降级 proxy，仅 opt-in 才 trace 真发；card/mesh 本身即 proxy）；确定性单测绿（对应 §7 RT 侧 / §8 RT 反射行：高配 proxy 参与、基线/兜底排除）。BVH 构建与 traversal 属非可移植 RT 后端桶，本模块只裁决注册哪种表示。
   - **（已完成）多求解器插槽（VBD 高保真）** `hair/solver.rs`：`HairSolverKind{Xpbd,Vbd}` + `SolverSelection.choose(stretch_stiffness)`（刚度过阈才路由 VBD，非有限值回落 XPBD）；`simulate_strand_vbd` 为逐顶点块下降（Vertex Block Descent）——每子步预测惯性目标后按 Gauss-Seidel 序对每个自由顶点做一次针对其惯性+拉伸+弯曲 3x3 Hessian 的精确 Newton 步（PSD 投影的弹簧 Hessian、手写 3x3 cofactor 求逆带奇异守卫），刚造型（辫子/发胶）不软塌；pinned 固定、空/单点/dt<=0 no-op、碰撞复用 `hair/collision.rs`、越界与奇异全 NaN 安全；确定性单测绿（含越刚越不拉伸差分对照）（对应 §6.7）。
   - **（已完成）近似 strand 自碰撞** `hair/self_collision.rs`：`SelfCollisionParams{particle_radius,stiffness,cell_size}` + `resolve_self_collision(particles,params)`——用确定性有序 `BTreeMap` 均匀空间哈希把粒子按整数网格分桶，每粒子只与其自身及 27 邻格内 `j>i` 候选（gather+sort_unstable 定序）做球体软斥（min_sep=2r，按 inverse_mass 分权推开，pinned 权 0 不动，coincident/both-pinned/非有限跳过），把全对 O(n^2) 降到局部密度级；standalone per-frame 后处理服务（类比 `collision.rs`，不接进 substep）；空/坏参 no-op、越界与 NaN 全安全；确定性单测绿（对应 §6.2 自碰撞近似 / §8 sim 高配行）。
   - **（已完成）SDF 身体碰撞** `hair/sdf_collision.rs`：`SdfPrimitive{Sphere,Capsule,HalfSpace,Box}`（各带解析 `signed_distance`，内负外正、退化半径/法线/半轴报 `f32::INFINITY` 失效）+ `union_signed_distance`（取并集最小值）+ `push_out_of_field`（沿中心差分场梯度按穿透深度迭代推出，默认 `DEFAULT_SDF_ITERATIONS=4` 趟以在重叠并集处收敛，零梯度点沿 +Y 逃逸不产 NaN）+ `resolve_sdf_collisions` / `SdfCollider`（owned 便捷桶）；比解析球/胶囊更贴合下颌/锁骨等紧配面，作为 item7 解析代理之上的更重档（`TressFX` 4 风格）；pinned 不动、空集/零迭代 no-op、越界与 NaN 全安全；确定性单测绿（对应 §6.2 SDF 碰撞 / §8 sim 高配行）。
   - **（已完成）帧内编排 kernel** `hair/groom.rs`：`GroomStepConfig{wind,xpbd,sleep,self_collision:Option<SelfCollisionParams>,sdf_iterations}` + `step_groom(particles,strand_lengths,rest_lengths,goal_positions,colliders,sdf,time,sleep_state,config)->GroomSleepState`——把上面各独立服务按毛发域固定的每帧 pass 序编排成一次 guide 步：①先测 `groom_motion_energy`→`update_sleep` 过休眠门；②若睡则立即返回、不碰任何粒子（不占形变预算，对齐 §8）；③否则 `apply_wind`（外力预处理）→`simulate_guides`（XPBD 主解，解析代理体在每子步内投影）→`resolve_sdf_collisions`（更重的 SDF 身体档）→`resolve_self_collision`（可选 strand-vs-strand 分离）；④返回推进后的 sleep 状态。同帧被扰动的睡眠 groom 会因入帧动能越过 `wake_above` 立即唤醒并当帧仿真（不掉拍）。纯确定性数组进数组出（§9），只组合既有 kernel、绝不重造求解器；空集/零迭代/睡眠各分支均安全；确定性单测绿（对应 §3 管线 pass 序 / §8 休眠不占预算）。
   - **（已完成，GPU 侧契约落地）GPU guide-XPBD 仿真 compute kernel** `prism_render_scene/src/shaders/hair_sim.wesl`：CPU golden `hair/dynamics.rs::simulate_guides` 的忠实 compute twin——一个 invocation owns 一根 guide strand，`@compute @workgroup_size(64)`，`substeps × iterations` 的 Gauss-Seidel 序完全对齐 CPU（integrate→solve_edges→solve_local→solve_global→solve_lra），`alpha=compliance/dt_sub²` / `gravity_step=gravity·dt_sub²` / `velocity_retain=1-clamp(damping)` 与 CPU 逐值同构；buffer 布局 `positions: array<vec4<f32>>`（.xyz=位置, .w=inverse_mass，0=pinned）+ `prev_positions` + `goals` + `rest_lengths` + `strands`（particle/rest/goal 偏移切片，同 CPU offset 纪律）+ `var<immediate> HairXpbdParams`；沙盒无 GPU，靠 naga 编译验证（`shader_tests::hair_sim_wesl_compiles_standalone` 绿）。这是 §8 GPU-driven 持久化的持久 sim stage 契约首片。**待续（跨子系统，hair 侧只留契约）**：sim↔渲染图接线 + 双缓冲交形变顶点属调度/`gpu_scene`；GPU 持久化/约束图着色分批/异步流水线（物理 §11/§12）；fiber-level 特写着色（材质系统 / `prism_render_npr`）。

**优先级**：先 card 基线打通端到端（务实），strand 高配随光栅/透射逐步点亮；每步 wgpu 可编译 + 单测绿。

---

## 11. 效果验收口径（v2 新增）

- **物理真实**：金发/浅色靠 dual-scattering 通透不发死；主高光（R）+ 次级彩色高光（TRT）到位；发内自阴影有体积感；快速甩头不炸毛（子步 + 形状约束生效）。
- **AAA 几何**：发丝级密度、无 LOD pop（连续减股 + dither 过渡）；不穿脸/肩（碰撞生效）。
- **NPR 观感**：天使环/各向异性高光带达顶级二次元（原神/GG Xrd）；卡片发吃 material id 描边；运动下不被 TAA 抹糊（reactive mask 生效）。
- **性能**：GPU 持久化 + 异步下 sim 不阻塞主渲染；大量 NPC 靠休眠 + card/mesh 降级可扩展。
- **跨平台**：sim/LOD/光栅/deep opacity 落 compute-可移植桶可 CPU golden；弱 RT 平台自阴影降级 deep opacity 仍可跑。

---

## 12. 风险

- **strand 光栅 + deep transmittance 是重头**：需 compute 软光栅与自阴影服务先就位，属"桩接管线 + vis-buffer"这条真·出血点（材质设计 §10）。
- **与物理内核的边界**：strand XPBD 求解原语应对齐 `prism_physics_core` §3/§4，渲染侧 hair 模块只负责 groom 几何/插值/LOD/光栅/透射/形变调度，**不重复造求解器**。
- **NPR 毛发无成熟范式**：顶级二次元毛发多为自研/魔改，PBR 侧可抄业界、NPR 侧需自趟。
- **sim 稳定性与确定性**：XPBD 约束刚度/迭代数需与形变预算折衷；联机/回放需固定子步 + 固定图着色序 + 固定随机种子。

---

## 附录：术语

- **guide strand / render strand**：被模拟的引导发丝（插值源）vs 插值出的渲染发丝（跟随不模拟）。
- **strand 档 / 减股档 / card 档 / mesh 档**：LOD 阶梯；仅前两档做 sim。
- **局部/全局形状约束 / LRA**：保段间刚度 / 拉回原造型 / 长距离防过拉伸。
- **compliance（柔度）**：XPBD 中代替 stiffness 的材料软硬量，与时间步无关。
- **R/TT/TRT**：Marschner 毛发 BSDF 的反射/透射/内反射三分量。
- **dual-scattering**：Zinke 多重散射近似，浅色发必需。
- **fiber-level**：逐纤维散射高保真（Yan et al.），近景特写高配。
- **deep opacity map / voxel 透射**：发丝自阴影的透射累积近似。
- **天使环 / 各向异性高光带**：NPR 可与切线解耦的风格化高光（发环）。
- **VBD**：Vertex Block Descent，硬造型高保真求解插槽。
