# Prism 渲染引擎 — 毛发引擎子系统完整设计（v1 / wgpu + WESL）

> 状态：架构提案（Draft，允许破坏性重构）
> 定位：AAA / 次世代 strand-based 毛发引擎，与 PBR/NPR/自定义/混合四前端正交协同
> Shader：WESL（naga → WGSL/SPIR-V/Metal/DXIL）；运行时：wgpu（Metal/VK/DX12/WebGPU）
> 关联文档：`prism_rendering_architecture_zh.md`（总）、`prism_material_pipeline_design_zh.md`（§6.2 子系统注册表、§6.3 毛发 mini 支柱）、`prism_physics_design_zh.md`

---

## 0. 定位：这是什么级别的毛发引擎

是**完整 strand-based（发丝级）毛发引擎**，不是"给头发一个各向异性高光"的着色贴皮。对标业界成熟产品，仅在**算法层**借鉴，不复制其代码：

| 产品 | 借鉴点（算法层） |
|---|---|
| **UE5 Groom** | Alembic groom 导入、guide→render 插值、compute 可见性光栅、deep opacity/voxel 透射、strand→card→mesh LOD、Niagara/物理驱动 groom sim |
| **AMD TressFX** | PPLL/OIT 发丝透明、Kajiya-Kay + Marschner 着色、局部/全局形状约束的 XPBD 式 strand 动力学、近似自阴影 |
| **NVIDIA HairWorks** | 样条细分、密度 LOD、自阴影 |
| **影视 Chiang / Marschner** | 物理毛发 BSDF（R/TT/TRT 分量）、dual-scattering 多重散射近似 |

**判据（承接材质设计 §6.2）**：一个东西要成为"一等子系统"，必须拥有**自己的几何 + 自己的 sim + 特殊渲染（透射/OIT/RT 代理）**。毛发三者全占，因此是一等子系统；而"毛发看起来的高光"只是 über-BSDF 的一个 closure 瓣，不构成子系统。**毛发与布料是两个独立子系统**（发丝几何+strand sim vs 三角网格+XPBD 布料 sim），仅共享形变预算这一资源仲裁层。

---

## 1. 与整体架构的关系（子系统内部同构范式）

毛发严格遵循全引擎的"共享 GPU-driven 基底 + PBR/NPR 分叉响应"范式（材质设计 §6.3），架构一致性拉满：

```
共享基底（PBR/NPR/自定义/混合 都吃）:
    groom 资产 → guide strand + 插值 render strand
    → guide-strand sim（XPBD）→ 形变顶点（汇入 gpu_scene）
    → 连续 LOD（strand→减股→card→mesh，禁硬切换 pop）
    → compute 发丝可见性光栅
    → 共享 deep-transmittance 自阴影服务 + 插进共享 OIT

分叉响应（C 类，真分家）:
    PBR: Marschner/Chiang R/TT/TRT + dual-scattering 多重散射
    NPR: 风格化各向异性高光带 + ramp + 阴影偏移（可与切线解耦=天使环）

fallback:
    strand 高配、card 基线；RT 反射里毛发用 proxy 或排除
```

毛发**不拥有**光照/阴影/OIT/motion 这些公共设施——它**消费**基底服务，只在"几何+sim+响应"处分家。这与"每个前端各自重算高级特性"相反，是本引擎"管线级混合"的优越性来源。

---

## 2. 数据与资产模型

- **Groom 资产**：一组 guide strand（模拟源）+ 插值参数 + 每 strand 属性（根/尖半径、根 UV、随机化种子、卷曲/丛聚参数）。导入源对标 Alembic groom（UE5 同源）。
- **Guide strand**：少量（数百~数千）被真正模拟的引导发丝；决定 sim 成本。
- **Render strand**：由 guide 在运行时插值出的大量（数万~数十万）渲染发丝；决定视觉密度与光栅成本，本身**不模拟**、只跟随插值形变。
- **绑定**：groom 绑定到蒙皮 mesh，根点随蒙皮移动（形变父级 = 材质设计 §6.1 蒙皮地基）。

代码落点（`prism_render_architecture::hair`）：
- `HairGroupHandle`、`HairGroup{ guide_strand_count, max_render_strands, segments_per_strand, deformation }`（**已落** `hair/mod.rs`）。
- `HairLodTier{ Strands, ReducedStrands, Cards, Mesh }` + `is_strand_based()`（**已落**）。

---

## 3. 管线阶段（端到端）

| 阶段 | 内容 | 对标 | 代码落点 |
|---|---|---|---|
| 1. 导入/插值 | guide→render strand 插值（丛聚/卷曲/随机化） | UE5 Groom | 规划 `hair/interpolation.rs` |
| 2. Strand 动力学 | guide 的 XPBD sim（边长 + 局部/全局形状约束 + 碰撞） | TressFX | 规划 `hair/dynamics.rs`（→ `deformation::schedule`）|
| 3. LOD | 覆盖度选档 + 连续减股，禁硬切换 | UE5/HairWorks | **已落** `hair/lod.rs` |
| 4. 光栅 | 发丝亚像素 compute 软光栅进 visibility | UE5 Groom / TressFX | 规划 `hair/raster.rs`（挂 `virtual_geometry` 软光栅桶）|
| 5. 着色 | HairPbr 闭包（Chiang/Marschner）+ dual-scattering | 影视 BSDF | `material` 的 `HairPbr` closure（不在 hair 内重写）|
| 6. 透射/自阴影 | deep opacity map / voxel 透射 | UE5/TressFX | 规划 `hair/deep_transmittance.rs`（共享阴影服务）|
| 7. 透明合成 | OIT，走 `HairVisibility` 路径 | TressFX PPLL | **已落** `transparency::routing`（`TransparentKind::Hair`）|

**关键边界**：着色（阶段 5）走材质系统的 `HairPbr` closure，透明（阶段 7）走 `transparency` 子系统的 `HairVisibility` 路径，形变预算走 `deformation::schedule`。`hair` 模块只**拥有**几何、LOD、sim 绑定与发丝专属透射，**引用而非重写**这些公共设施。

---

## 4. LOD 策略（禁 pop 硬切换）

- **离散阶梯**：`Strands`（全股）→ `ReducedStrands`（减股，本引擎按 1/4 股 + 1/2 控制点整数抽稀，确定可复现）→ `Cards`（相机朝向卡片，无逐股几何）→ `Mesh`（静态网格壳，远/离屏兜底）。
- **连续过渡**：阶梯内再叠加连续 strand 抽稀/宽度补偿（strand 变稀时增宽保覆盖），跨档用 dither/alpha 过渡避免 pop（本轮先落离散阶梯，连续过渡随光栅一并做）。
- **覆盖度驱动**：输入为屏幕覆盖度 `0..=1`（调用方投影得出），本层是纯确定性分类、无投影/超越函数——契合"compute 可移植 + CPU golden 可测"桶（材质设计 §8.1）。
- **sim 只在 strand 档**：`Cards`/`Mesh` 不发形变请求（`hair_deformation_request` 返回 `None`），彻底省掉远处 sim 成本。
- **原生形态钳制（消 PBR 偏见）**：阶梯默认把 strands 当"顶级"、cards/mesh 当"降级"，但对 NPR/二次元 groom，**卡片本身就是授权的原生外观，不是降级**。`HairGroup::native_form` 声明该 groom 最细可用的几何表示：strand-authored 用 `Strands` 走全阶梯；卡片授权用 `Cards`。`resolve_hair_lod` 用 `tier.coarser_of(native_form)` 把覆盖度选出的档钳制成"不比授权更细"——满屏也不会把卡片 groom 提升到它根本没有的 strands，但距离拉远仍可继续降到 mesh。这是**授权几何选择，与 PBR/NPR 着色响应正交**，两种风格同等遵循。

**已落**：`select_hair_lod_tier` / `resolve_hair_lod`（含 `native_form` 钳制）/ `bin_hair_lod`（按档分桶，越界跳过不 panic，保输入序）/ `hair_deformation_request`（strand 档→ `DeformationKind::Hair`，`vertex_count = render_strands × segments`，`needs_blas_refit=true`）。

---

## 5. PBR / NPR / 自定义 / 混合 分叉响应

- **PBR**（对标 UE5 Groom、film-grade Chiang/Marschner）：R/TT/TRT 三分量 + dual-scattering 多重散射（浅色/金发必需），各向异性沿切线；deep opacity map 自阴影 + 有 RT 时走硬件光线自阴影。
- **NPR**（对标 miHoYo 原神/星穹铁道发丝、Arc Sys Guilty Gear Xrd）：风格化各向异性高光带（可多条）+ ramp + 阴影偏移，高光位置**可与真实切线解耦**（天使环/发环风格）；描边复用 vis-buffer 的 material id 边界（`native_form=Cards` 的卡片发同样吃 material id 描边）；per-strand 或 per-card 顶点色控高光/阴影段。
- **自定义**：项目注入 `illumination=Custom` 的 WESL closure。
- **混合**：同一 groom 甚至可按 material id 分 tile 路由到不同前端（因共享光/影/GI 而连贯）；写实场景里的卡通角色发与 PBR 世界吃同一份 GI/阴影，不断裂。

这四者共享 §1 的全部基底（同一份 strand 几何、sim、LOD、透射），**只在着色响应处分家**——所以四者都能享受发丝级几何与自阴影，不是各做各的。

---

## 6. 性能预算与降级矩阵

| 能力 | 高配 | 基线 | 兜底 |
|---|---|---|---|
| 几何 | 全 strand 软光栅 | card | mesh 壳 |
| sim | guide XPBD + 碰撞 | guide XPBD 无碰撞 | 静态（root 跟蒙皮）|
| 自阴影 | deep opacity/voxel 透射 | 近似 dither 阴影 | 参与普通阴影图 |
| 透明 | OIT（PPLL/moment）| 排序 alpha | alpha test |
| RT 反射 | proxy 参与 | 排除（只主视图）| 排除 |

- strand 是 AAA 最重特性之一：**card 基线务实、strand 高配可选**，别一上来全 strand（承接材质设计 §10 风险条）。
- 形变预算由 `deformation::schedule` 统一仲裁：单帧顶点上限 + BLAS refit 独立配额；密集 groom 超预算时最高优先项无条件先跑（防饿死），其余延后。

---

## 7. 可测性

毛发 sim / LOD / 光栅落"compute 可移植"桶（材质设计 §8.1）：数组进数组出，可写手写 Rust CPU golden 逐值对数。当前 `hair/lod.rs` 已有 13 个确定性单测（阈值分档、抽稀因子、代理档丢几何、strand 档发形变请求、分桶保序、越界跳过、空输入、`native_form` 卡片钳制、`coarser_of` 秩比较）。着色 BSDF 的 closure 可 golden；RT traversal 不可（驱动 BVH）——与全引擎三桶边界一致。

---

## 8. 落地路线图

1. **（已完成）子系统骨架**：`hair/mod.rs` 核心类型 + `hair/lod.rs` LOD 阶梯/分桶/形变绑定（commit 已落，136 单测绿）。
2. **strand 动力学** `hair/dynamics.rs`：guide XPBD（边长 + 局部/全局形状约束），先无碰撞，输出接 `deformation::schedule`；确定性 CPU golden。
3. **插值** `hair/interpolation.rs`：guide→render 丛聚/卷曲/随机化参数与 LOD 抽稀联动。
4. **发丝光栅** `hair/raster.rs`：亚像素软光栅桶，挂 `virtual_geometry` compute 软光栅路径，进 visibility。
5. **deep transmittance** `hair/deep_transmittance.rs`：自阴影透射服务，接共享阴影/OIT。
6. **着色分叉**：material `HairPbr`（Chiang/Marschner + dual-scattering）+ `prism_render_npr` 风格化毛发响应。
7. **碰撞与连续 LOD 过渡**：sim 碰撞体、跨档 dither/alpha 过渡消 pop。
8. **RT 代理**：反射中的 proxy/排除策略。

**优先级**：先 card 基线打通端到端（务实），strand 高配随光栅/透射逐步点亮；每步 wgpu 可编译 + 单测绿。

---

## 9. 风险

- **strand 光栅 + deep transmittance 是重头**：需要 compute 软光栅与自阴影服务先就位，属"桩接管线 + vis-buffer"这条真·出血点（材质设计 §10）。
- **NPR 毛发无成熟范式**：顶级二次元毛发多为自研/魔改，PBR 侧可抄业界、NPR 侧需自趟。
- **sim 稳定性**：XPBD 约束刚度/迭代数需与形变预算折衷；确定性要求限制随机化用固定种子。

---

## 附录：术语

- **guide strand**：被模拟的引导发丝，插值源。
- **render strand**：插值出的渲染发丝，跟随不模拟。
- **strand 档 / card 档 / mesh 档**：LOD 阶梯；仅 strand 档做 sim。
- **deep opacity map / voxel 透射**：发丝自阴影的透射累积近似。
- **dual-scattering**：毛发多重散射的两级近似（全局 + 局部），浅发必需。
