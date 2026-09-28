# Prism 渲染引擎 — 材质与管线完整架构设计（Slang 版 / v1）

> 状态：架构提案（Draft，允许破坏性重构）
> 面向版本：Bevy/Prism 下一代渲染底座
> Shader 工具链：**Slang**（唯一 shader 语言，替换现有 WESL）
> 后端策略：**Vulkan 优先**（VK 先写代码，架构后端中立，后续扩 Metal/D3D12/WebGPU/主机/移动）
> 本文依据：对 `pkg/` 下渲染 crate 的静态阅读 + 架构讨论收敛结论；未运行示例/测试/基准
> 关联文档：`docs/prism_rendering_architecture_zh.md`（较早 draft，本文在其之上收敛材质/管线/子系统/Slang 决策）
> 最后更新：2026-09-28

---

## 0. 设计目标与非目标

### 目标（硬约束）

- 同一引擎**同时**是顶级 AAA 的 PBR / NPR / 混合 / 自定义引擎；四者皆一等公民，无谁是底座、无谁是补丁。赛道由具体项目/场景选择，不由引擎替用户选。
- 后端中立，**Vulkan 优先**（先出 SPIR-V/VK，架构须能无痛扩到 Metal/D3D12/WebGPU/主机/移动）。
- 吃满高级特性：GI/RT、虚拟阴影（VSM）、虚拟几何、时序上采样、动态光照、粒子、透明、毛发、布料、体积。

### 非目标（主动不做，避免撞墙）

- 不做完整无界 Substrate（性能重、撞弱平台、对"混合"无用——混合靠逐像素多材质，不靠单材质无限瓣）。
- 不追求 RT 里对 NPR 的物理正确（NPR 在次级光线里本就是降级的，见 §10）。
- 不追求"物理外观一致"的 PBR/NPR 统一（这是错误目标，见 §4）。

---

## 1. 顶层架构：共享 GPU-driven 基底 + 多并存前端 + 正交风格轴

```
                        ┌─────────────────────────────────────────────┐
                        │   资产 / 材质图 (node graph, Slang 生成)       │
                        └───────────────┬─────────────────────────────┘
                                        │ normalize → closure IR
        ┌───────────────────────────────┼───────────────────────────────┐
        │                    支柱一：材质 ABI (正交轴)                     │
        │   domain × closure(IR) × illumination × blend + specialization │
        └───────────────────────────────┼───────────────────────────────┘
                                        │
   ┌────────────────────────────────────┼────────────────────────────────────┐
   │              共 享 G P U - d r i v e n  基 底 (算一次, 全前端消费)          │
   │  支柱四: 可变形几何/模拟(蒙皮父级 → 布料/毛发/粒子 sim) → gpu_scene       │
   │  → 可见性(vis-buffer / virtual_geometry) → material id                   │
   │  支柱二: 光照/阴影"数据"服务(光源SoA+cluster / VSM页 / RT阴影反射 /       │
   │          GI·IBL·GTAO irradiance / BVH)  ——只出数据, 不出响应             │
   │  支柱三: 跨切面基底(motion+reactive mask / OIT / RT去噪 / capability)     │
   └────────────────────────────────────┼────────────────────────────────────┘
                                        │  material id + tile 分类路由
   ┌──────────────┬──────────────────────┼───────────────────┬────────────────┐
   │ 延迟 PBR 前端 │  forward+ 前端        │  NPR 风格化前端    │  自定义前端槽   │
   │ (虚拟几何/RT/ │ (一等:透明+粒子+可选  │ (描边/ramp/SDF面阴 │ (项目注入)      │
   │  GI/VSM)      │  NPR 承载)           │  影/风格化高光/post)│                │
   └──────────────┴──────────────────────┴───────────────────┴────────────────┘
        正交风格轴 illumination = Lit / Stylized / Unlit / Custom (叠在任意前端)
        混合 = 逐像素 material id, tile 分类到对应前端, 因共享光/影/GI而连贯
```

**一句话**：基底装满、算一次；每个前端从基底取自己要的数据、叠自己独有的响应和通道。这是"四者皆顶级 + 可混合 + 多平台"的唯一解——单独管线会让每条前端各自重造高级特性，没一条能到 AAA。

---

## 2. Slang 工具链（本版核心决策，会反向重塑材质 ABI）

选 Slang 不只是换语法，它的四个能力直接决定架构长相。

### 2.1 为什么 Slang（对着四个硬需求）

| 硬需求 | Slang 的对应能力 | 对 WESL 的优势 |
|---|---|---|
| 闭包 IR 要能被延迟 pass **和** RT hit shader 共用 | `interface` + 泛型（关联类型），一份 closure 定义两处实例化 | WESL 无接口/泛型，只能宏拼或复制 |
| 后端中立、VK 优先 | 同一份源码 → SPIR-V(先) / DXIL / Metal / WGSL / CPU | WESL 经 naga，目标面窄、RT 支持弱 |
| CPU golden reference 不漂移 | Slang **CPU/C++ host target**，从**同一份 shader**生成 CPU 参考 | 现在 CPU 参考是独立 Rust(`resolve.rs`)，靠人肉对齐，必漂 |
| specialization 排列编译 | **link-time specialization** + 类型参数 + link-time 常量 | WESL 靠预处理宏，组合爆炸难管 |

**最大的白捡收益**：现在 `prism_render_shading/resolve.rs` 里的 CPU 参考实现和 WESL shader 是两套代码人肉对齐（`evaluate_toon_direct` 等），这是 golden reference 漂移的结构性隐患。Slang 能把同一份 closure 源码编到 CPU host，**CPU 参考 = shader 本身的另一个编译目标**，漂移风险从"靠纪律"变成"编译器保证"。这一条单独就值回迁移成本。

### 2.2 Slang 如何映射到各支柱

- **闭包 IR = Slang `interface IMaterialClosure`**：`evalBSDF / sample / pdf / evalStylized` 是接口方法。über-BSDF、有界 slab、毛发专用 closure 都是它的 `struct ... : IMaterialClosure` 实现。延迟着色 pass 与 RT hit shader **import 同一个 module**，各自实例化——这正是"闭包 IR 必须 RT 可求值"的落地方式。
- **specialization 桶 = link-time specialization**：材质图 normalize 出的排列（如 `über + clearcoat瓣 + Lit` vs `über + Stylized`）在链接期特化成独立 variant，`specialization_id` 就是这个 variant 的键。排列编译器 = 遍历合法轴组合 → 批量特化。
- **ABI 头 = Slang reflection 自动生成 Rust `#[repr(C)]`**：不再手写 `GpuMaterialHeader`/`GpuSurfaceParameters` 两边对齐。用 slang-reflect 从 shader 侧结构反射出 Rust 绑定，GPU/CPU ABI **无法**漂移。`MATERIAL_ABI_VERSION` 由反射内容哈希驱动，改结构自动 bump。

### 2.3 构建集成

- 新 crate `prism_render_slang`（或 `prism_shader`）：封装 slangc 调用、reflection→Rust codegen、variant 缓存。
- 构建期：`.slang` → SPIR-V（VK）+ CPU host lib（golden ref）+ reflection json → Rust `build.rs` 生成 ABI 绑定。
- `.wesl` 现有 shader（`pkg/prism_render_scene/src/shaders/*.wesl`）**分批迁移**，不一次性推倒。迁移期两者并存，新 closure 一律 Slang。

### 2.4 迁移成本（诚实）

- slangc 进工具链、build.rs codegen、variant 缓存是**真实一次性成本**，约等于建一套 shader 构建子系统。
- naga/WESL 的现有对齐测试要改造成"Slang CPU target 对齐 GPU target"。
- 网络受限环境下拉 slang 二进制/依赖需要升级权限（`require_escalated`）。

---

## 3. 支柱一：材质 ABI（破坏性重构）

### 3.1 正交轴取代互斥枚举

删掉 `MaterialShadingModel`（`pkg/prism_render_material/src/record.rs`，9 值单标量互斥枚举）这个**性能+易用性双病根**。改为四条正交轴：

```
Material = {
    domain:        Surface | Decal | Volume | PostProcess       // 保留, 现有已对
    closure:       ClosureGraph (IR, 见 3.2)                     // 取代 shading_model
    illumination:  Lit | Stylized | Unlit | Custom              // 新增, 正交风格轴
    blend:         Opaque | Masked | Transmissive | Transparent | Additive
}
specialization_id: u64   // 由上面轴的合法排列特化产出
```

- **`illumination` 是"风格轴"**，正交于 closure。`Stylized` 不再挤在 shading_model 里，而是"用什么方式解读同一份光照数据"（BRDF 积分 vs ramp 量化）。这解决了一个关键直觉——**NPR 不该和 Water 同级**：Water 是 closure/子系统，NPR 是 illumination 轴。
- `MaterialRenderClass` 里的 `NprOpaque/NprTransparent/CustomOpaque/CustomTransparent` **全删**——它们是 `blend × illumination` 的笛卡尔积被错误地拍平进一个枚举。拆回两条正交轴后自然消失。

### 3.2 闭包 IR：über-BSDF 一等 + 有界 slab 可选

`pkg/prism_render_material/src/ir.rs` 的 `ClosureKind` 保留大部分，但语义改变：

- **地基 = OpenPBR 锚定的 über-BSDF 一等 closure**。metallic/roughness/clearcoat/sheen/subsurface/transmission/anisotropy 全是它的**参数/瓣**，不是独立 shading model。`Cloth sheen`、`ClearCoat` 降级为 über 的瓣（不再是顶层 model）。
- **有界 slab 栈（封顶 3–4 瓣）作可选升级**：`Layer`/`Mix` 节点保留，但**编译期强制封顶**。超过上限 → validation error，不进无界 Substrate。
- **`normalize()` 必须改**：现在它把闭包压成单 `shading_model`（`ir.rs` 里 `if kind==Npr { shading_model = Npr }`）——**这是把正交信息毁掉的元凶**。新 `normalize` 产出 `ClosureGraph`（保留多瓣结构）+ `illumination` 轴 + `specialization_id`，**不再塌缩**。
- **毛发是真例外**：`ClosureKind::Hair` = 专用重 closure（Marschner/Chiang R/TT/TRT + dual-scattering），**不是** über 的瓣。NPR 头发 = `illumination=Stylized` 下的各向异性高光带（可与物理切线解耦，做"天使环"），是另一条 closure 实现。见 §6.3。

### 3.3 GPU 数据结构（拆胖结构）

- `GpuSurfaceParameters`（18 字段胖结构，每像素全带）拆成 **über 核心参数（紧凑）+ 按瓣可选 blob**。specialization variant 决定读哪些 blob，不再无脑塞 18 字段。
- `GpuMaterialHeader`：`shading_model` 字段删除；新增 `illumination`、`closure_graph_offset`、`specialization_id`。`closure_mask` 保留（RT/分类用）。
- 所有这些结构由 **Slang reflection 生成**（§2.2），不手写。

---

## 4. 支柱二：光照/阴影 数据服务 + 各前端响应（比材质更关键）

**核心原则：共享的是数据/加速结构，不是响应。** "一致"的正确定义 = **同一批授权光源驱动，外观按各自风格解读**（NPR 本就该长得不一样），不是物理外观相同。

### 4.1 基底共享（算一次）

光源 SoA + cluster 剔光 / VSM 深度页 / RT 阴影·反射结果 / GI·IBL·GTAO irradiance / BVH。

### 4.2 各前端响应分家（A/B/C 三类）

| 基底数据 | A 类·同能力不同实现 | B 类·能用但固有降级 | C 类·各自专属 |
|---|---|---|---|
| 光源+cluster | PBR: BRDF 积分 / NPR: ramp 量化 | — | NPR: light layer 分层打光 |
| VSM 深度页 | PBR: PCF 软阴影 / NPR: 阈值硬阴影+染色 | — | NPR: **SDF 面部阴影**(独立数据源叠加) |
| GI/IBL | PBR: 物理环境光 / NPR: 收进来做风格化底光 | **NPR 对 GI 只收不发**(防能量爆炸) | — |
| RT 反射 | PBR: 物理反射 | RT 里 **NPR 降级为风格化 albedo 近似** | — |
| RT 去噪 | 共享去噪器 | **NPR 需单独调锐利分段**参数 | — |

- **A 类占大头**——"同能力不同实现"在这里成立。
- **B 类必须认**：RT×NPR 是"数据喂风格化响应"，不是物理追 NPR，换任何设计绕不掉。
- **NPR 专属通道**（SDF 面阴影 / shadow ramp / light layer）挂基底，**只 NPR 前端消费**。

### 4.3 Slang 表达

`interface ILightResponse { float3 shade(ShadingCtx, LightSample, Closure); }`。PBR/NPR 是两个实现，import 同一份光源/阴影数据结构。前端切换 = 换 `ILightResponse` 实现，光照数据零改动。

---

## 5. 支柱三：跨切面基底（解一次，全前端共享）

- **motion vector + reactive/stencil mask**：粒子/透明/NPR/毛发**全和 TAA 打架**，这是**最该早做**的基底。粒子子系统会逼你先把它立起来（§11）。
- **OIT**：透明 + 粒子 + 毛发共同消费的公共设施（**不是**并列子系统）。
- **RT 去噪**：共享去噪器，每前端调参（NPR 见 §4.2 B 类）。
- **能力分层 / fallback**：见 §8。

---

## 6. 支柱四：可变形几何 / 模拟阶段 + 子系统注册表

### 6.1 蒙皮父级（所有形变子系统的根）

```
可变形几何阶段 (骨骼蒙皮 / morph / 形变输出)
   ├── 布料 sim (XPBD)        → 形变顶点
   ├── 毛发 sim (guide strand)→ 形变顶点
   ├── 粒子 sim (GPU)         → 实例/条带
   └── 普通蒙皮角色           → 形变顶点
          ↓ 全部汇入 gpu_scene → 可见性 → 各前端
```

落 `prism_physics_core / prism_physics_geometry`。**任何角色都要蒙皮**，所以这是 VK 竖切绕不过的第一个真子系统。

### 6.2 子系统注册表（四档，钉死）

| 档 | 成员 | 判据 | 落地优先级 |
|---|---|---|---|
| **0 地基** | 可变形几何(蒙皮/morph) | 所有形变子系统父级 | **最先** |
| **1 一等子系统** | 粒子·毛发·体积·布料sim | 自己的几何+sim+特殊渲染 | 粒子最先(带起 reactive mask) |
| **2 后置留槽** | 水/海洋·植被·贴花 | 是子系统但重/非通用 | 竖切不碰 |
| **3 陷阱·别做子系统** | SSS/sheen/清漆(=closure)、大气(=光照数据服务)、OIT/motion(=基底服务) | 无几何无sim / 是公共设施 | — |

**判据**：需要自己的「几何+sim+特殊渲染(透射/OIT/RT代理)」才是子系统；只是着色变化 → closure；大家都消费 → 基底服务。**布料要拆**：sim 是子系统、sheen 是 über 一个瓣。

### 6.3 毛发 mini 支柱（子系统内部同构范例）

```
共享基底(PBR/NPR 都吃): strand几何(连续LOD→card, 禁硬切换) + guide-strand sim
                       + 共享 deep-transmittance 自阴影服务 + 插进共享 OIT
分叉响应(C类, 真分家):  PBR: Marschner/Chiang R/TT/TRT + dual-scattering
                       NPR: 风格化各向异性高光带 + ramp + 阴影偏移(可与切线解耦=天使环)
fallback:              strand 高配, card 基线; RT 反射里毛发用 proxy 或排除
```

**所有一等子系统内部都长这个样**（共享基底 + PBR/NPR 分叉响应），架构一致性拉满。

---

## 7. 前端（并存，皆一等）

- **延迟 PBR 前端**：承载虚拟几何(Nanite 级)/RT/GI/VSM。虚拟几何与 forward+ 互斥，**这才是逼向延迟的真原因，不是 PBR 本身**。
- **forward+ 前端（一等公民）**：透明**强制** forward，所以这条路本就必建；NPR 复用它、粒子挂它，**不是额外成本**。
- **NPR 风格化前端**：描边 / ramp / SDF 面阴影 / 风格化高光 / rim / 风格化 post。**vis-buffer 的 material id 边界白送描边**。
- **自定义前端槽**：项目注入 `illumination=Custom`，走自定义 Slang closure + 自定义 pass。
- **混合 = 逐像素 material id + tile 分类路由**：GPU 把像素按 material id 分 tile，路由到对应前端。因共享光/影/GI 而连贯——这是"管线级混合"的具体机制，也是它比"各前端各算高级特性"优越的地方。

---

## 8. 后端抽象（破坏性重构）

- 删 `BackendMode{VulkanFirst,WgpuCompatibility}` 二值枚举 → **后端注册表 + capability 查询**。
- 保留并扩展 `VulkanTier{Core13,MeshShader,RayQuery,Full}`，泛化成跨后端 capability bits（mesh shader / ray query / bindless / wave ops …）。
- **fallback 矩阵**：每个高级特性声明所需 capability + 降级路径（无 RT → SSR/SSGI；无 mesh shader → 传统 index draw；无 bindless → 描述符表）。
- Slang 同源多目标（§2.1）让"VK 先，后面 Metal/D3D12/WebGPU"从"重写 shader"变成"加编译目标 + 补 capability 分支"。

---

## 9. 代码层破坏性重构清单（可编译可测每步）

1. `pkg/prism_render_material/src/record.rs`：删 `MaterialShadingModel`；`MaterialRenderClass` 删 `Npr*/Custom*`；`GpuMaterialHeader` 去 `shading_model`、加 `illumination`/`closure_graph_offset`/`specialization_id`；`GpuSurfaceParameters` 拆核心+瓣 blob。
2. `pkg/prism_render_material/src/ir.rs`：`normalize()` 停止塌缩成单 `shading_model`，产出 `ClosureGraph`+`illumination`+`specialization_id`；`Layer/Mix` 加封顶校验。
3. `pkg/prism_render_shading/src/classification.rs`：`MaterialShadingClass` 固定 9 桶 → 按 `specialization_id`/tile 动态分桶；`classify_material_header` 重写。
4. `pkg/prism_render_shading/src/resolve.rs`：`evaluate_toon_direct` 从"一个特例分支"提升为 `ILightResponse(Stylized)` 实现；扩描边/ramp/SDF 面阴影/rim/post（现在几乎是空的）。
5. `pkg/prism_physics_*/.../backend/mod.rs`：`BackendMode` → 后端注册表 + capability。
6. 新 crate `prism_render_slang`：slangc 封装 + reflection→Rust codegen + variant 缓存。
7. 新 crate `prism_render_npr`：NPR 前端 ABI 骨架 + `evaluate_stylized_direct` + 屏幕空间描边 CPU 参考（由 Slang CPU target 出）。
8. `.wesl` → `.slang` 分批迁移（新代码一律 Slang）。

---

## 10. 诚实的固有摩擦与风险（不糊）

- **B 类降级不可消除**：RT×NPR、NPR 对 GI 只收不发、RT 反射里 NPR 降级 albedo、去噪器 NPR 单独调参。
- **NPR 一等公民无成熟范式**：PBR 侧有 UE/业界抄作业，**NPR 一等前端这条基本得自己趟**——这是全设计风险最高、参考最少的部分（顶级二次元 NPR 大作多为自研或魔改引擎，不用 UE 招牌管线）。
- **Slang 迁移是真成本**：一套 shader 构建子系统 + CPU 对齐改造。收益（RT 共用 closure、多目标、CPU 参考不漂移、reflection 生成 ABI）大于成本，但成本要认。
- **strand 毛发是 AAA 最重特性之一**：card 基线务实、strand 高配可选，别一上来就 strand。
- **真正的落地风险不在材质模型**（业界已收敛），而在**还不存在的 Vulkan 后端 + vis-buffer 基底**。

### 与 UE 的关系（定位参考）

- **对齐部分**（PBR 骨架，业界共识）：GPU-driven 基底 + 后端中立（≈ UE RHI）、延迟为主 + 虚拟几何逼向延迟、透明走 forward、材质从互斥枚举转向可组合（≈ UE Substrate 的方向）。
- **主动分歧**：① 把 NPR 从补丁抬成一等前端（UE 不做）；② 把 Substrate 从近乎无界收敛成**有界 slab** 以吃满多平台。
- 结论：PBR 骨架"是同一标准"，NPR 一等 + 材质封顶策略"刻意不同"，且不同得有道理。

---

## 11. 落地路线图（VK 优先，每步可编译+测试绿）

**破除分析瘫痪的关键：先一条极薄端到端竖切，让竖切反过来钉死 ABI。**

1. **建 `prism_render_slang`**：slangc + reflection codegen + variant 缓存跑通（先编一个 über closure 到 SPIR-V + CPU target，对齐）。
2. **极薄竖切**：真·Vulkan → 蒙皮(支柱四地基) → vis-buffer → material id → **一个 PBR 延迟着色 + 一个 NPR forward 着色**，点亮光照/阴影**数据服务最小版**（一盏方向光 + 一张 VSM）。
3. **材质 ABI 破坏性重构**（§9 第 1–3 步），用竖切验证正交轴 + specialization。
4. **粒子子系统**：优先，因为它带起 reactive mask（§5 最该早做的基底）。
5. **`prism_render_npr` 骨架**：描边（material id 边界白送）+ ramp + `evaluate_stylized_direct`。
6. **frame graph 装配节点**，最后逐节点 Vulkan 实装。
7. 毛发(card)/布料/froxel 体积按 §6.2 优先级跟进；水/植被/贴花后置。

---

## 附录 A：术语与轴速查

- **前端 (frontend)**：延迟 PBR / forward+ / NPR / 自定义。决定"怎么组织 pass 与着色"。
- **illumination 轴**：Lit / Stylized / Unlit / Custom。决定"用什么方式解读光照数据"。正交于前端与 closure。
- **closure**：über-BSDF（一等）/ 有界 slab / 毛发专用。决定"表面 BSDF 长什么样"。
- **子系统**：有自己几何+sim+特殊渲染的横切模块（毛发/粒子/体积/水/布料sim）。
- **基底服务**：大家都消费的公共设施（OIT/motion+mask/透射/去噪/光照数据）。**不是**子系统。
