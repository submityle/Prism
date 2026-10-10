# Prism 材质系统 — 破坏性重构设计（v10 / OpenPBR Surface 规范参数化 · 开放 über · 每材质静态特化 · 无全局硬顶 · 无塌缩 · 四线正交 · 传递无关缓存 · 统一执行模型版）

> 状态：架构提案（Draft，**允许破坏性重构，不保留旧 API**）
> 面向版本：Prism 下一代渲染底座（Bevy fork，`pkg/` workspace）
> Shader 语言：**WESL**（经 naga → WGSL/SPIR-V/Metal/DXIL），不引入 Slang
> 运行时后端：**wgpu**（Metal/Vulkan/D3D12/WebGPU/主机/移动全覆盖）
> 本文依据：对 `pkg/prism_render_material`、`prism_render_shading`、`prism_render_scene/src/shaders/*.wesl`、`prism_render_architecture` 的静态阅读（以代码为准）
> 核心命题：**没有全局硬顶，不做运行时塌缩，也不做 per-pixel 预算裁剪。模型开放可扩展，每材质按声明静态特化成它确切需要的瓣集与特征——"界"是每材质编译期事实，不是全局天花板。分层在烘焙期解析成同参数化纹理空间混合（层数天然不限）；距离/性能靠纹理预过滤 + 高光抗锯齿 + 静态置换 + 采样缩放，全程无损、确定、跨平台。**
> 规范参数化：**运行时 über 以 OpenPBR Surface（ASWF/Adobe/Autodesk 2024）为规范核心，Prism über 是 OpenPBR 的真超集**——`Prism über = OpenPBR Surface（全字段） ∪ { FACE 私有瓣 } ∪ { 开放词汇表后续瓣 }`。现有 7 瓣只是今天未补齐的实现子集；补齐缺口（thin_film 虹彩、coat/fuzz/subsurface 全字段等）是对开放词汇表的**加法扩充**，不是改架构。超集成立的三条硬约束见 §4.5（纯 OpenPBR 子集合规对拍 / 扩展位段隔离 / 向下投影）。FACE 作为 OpenPBR 之外的 Prism 私有瓣服务 NPR。
> 四线正交：**PBR / NPR / 混合 / 自定义不是四条管线，而是同一条 OpenPBR über 上的正交响应模式**，由 `axis.rs` 的 `illumination` 轴 + `render_class` + 可选 `custom_program` 钩子选择；共享同一套参数解包、per-tile 分类与降级。
> 最后更新：2026-10-10（v10 结构性重构：PCM 能力基座从“升级方向”提升为顶层 §5（“超集”的完整模型）；原 §5–§15 顺延为 §6–§16；原 §16“设计升级方向”收敛为纯可选演进并顺延为 §17；全文章节与交叉引用整体重编号。）

---

## 0. 设计目标与非目标

### 目标（硬约束）
- **顶级次世代 AAA 质量**：PBR / NPR / 混合 / 自定义四条线皆一等公民，效果不设天花板。四线是**同一条 OpenPBR über 上的正交响应模式**（§3.5），不是四套管线。
- **规范参数化 = OpenPBR Surface**：运行时 über 的权威参数集采用 OpenPBR Surface（2024），作者端 MaterialX 节点图亦以 OpenPBR Surface 为目标 BSDF。现 7 瓣是其子集，缺口按开放词汇表加法补齐（§4.4）。
- **性能确定、跨平台**：Web / 移动 / 主机 / 桌面同一套管线；每材质代价编译期可知，无 per-pixel 非确定预算裁剪。
- **易用**：美术用行业标准作者层（OpenPBR / MaterialX）+ 原型模板，不手搓内部 IR。
- **无全局硬顶**：没有 `MAX_LOBE`、`MAX_TEXTURE`、`MAX_DEPTH`、`MAX_LAYER` 这类人为上限；模型是开放可扩展的词汇表。
- **一条线**：作者 → 烘焙 → 运行只有一条管线，一个 über-BSDF 模型族；没有"固定深度 slab 线"与"不计数 über 线"的二分。
- **无损降级**：靠纹理预过滤（mip/VT）+ 高光抗锯齿自动降细节，**瓣数/特征在距离上恒定**；绝不做有损的瓣/层塌缩。
- **双线可测**：保留 CPU 金标准（`prism_render_shading`）+ WESL GPU twin 的字节级对拍，作为确定性与回归门禁。

### 非目标（主动不做）
- **不做 BSDF 塌缩阶梯**（offline/runtime 把多瓣合成少瓣）：有损、档位跳变、不可预期——明确否决。
- **不做 per-pixel 运行时预算裁剪**（Substrate 式动态砍瓣）：非确定、撞弱平台——明确否决。
- **不设全局硬顶来"定界"**：定界靠每材质编译期静态特化，不靠全局 MAX 常量。
- 不保留 ABI v4 / 旧 `MaterialRecord` / 旧 `MaterialGraph` 对外契约；**删除 `bevy_bridge.rs:40 lower_standard_material`** 这类"外部材质 1:1 直映"翻译层，统一走 OpenPBR Surface 导入烘焙（§4）。
- 不追求 RT 次级光线里 NPR 的物理正确。

> **"无界"与"确定"如何兼得**：无界 ≠ 运行时动态无界。无界指**作者端层数/节点不限**、**模型词汇表可扩展**、**没有全局数值天花板**；确定来自每材质在编译期被特化成它声明的确切瓣集——代价是编译期常量，不是运行时逐像素赌博。

---

## 1. 现状基线（代码实测，作为重构起点）

| 事实 | 位置 | 说明 | 本次处理 |
|---|---|---|---|
| `MATERIAL_ABI_VERSION = 4` | `record.rs:7` | GPU header 契约版本 | bump 到 5（破坏性） |
| `MAX_MATERIAL_TEXTURES = 8` | `record.rs:8` | 纹理槽硬上限 | **删**，改 VT 页引用（§9） |
| `LobeMask::COUNT = 7` | `surface.rs:153` | EMISSION/CLEARCOAT/ANISOTROPY/SHEEN/SUBSURFACE/TRANSMISSION/FACE | 改为**开放 enum**（词汇表可增，mask 位宽留足），不作天花板 |
| 核心 12 words + 每 lobe 4 words，变长打包 | `surface.rs` | WESL `material_unpack.wesl::prism_unpack_surface` 字节级镜像 | 保留变长打包，去掉 COUNT 作为上限的语义 |
| `MAX_CLOSURE_SLAB_DEPTH = 4`，仅 Mix/Layer 计深度 | `ir.rs:9` / `slab_depth()` | 超限 → `ClosureSlabTooDeep` | **删**，运行时无闭包图,无需定界 |
| 9-way 着色分类（prefix-sum/scatter，无 readback） | `classification.rs` / `material_classification.wesl` | `classify_count → prefix_classes → scatter_work` 共享 ABI | 保留，不加 tier 维度 |
| 虚拟纹理地基已存在 | `texture_streaming/{indirection,residency,feedback,scheduler,pool}.rs` | 页表（`PAGE_TABLE_ENTRY_WORDS=4`，二分可查）+ 残留表状态机 | 复用，承接距离 LOD |
| 正交轴 + `SpecializationId` | `axis.rs` | illumination[0..8] / closure_mask[8..40] / render_class[40..48] | 保留，不放 tier |
| 7 瓣 über = OpenPBR Surface 子集 | `surface.rs` | 核心 base/metallic/roughness/reflectance + 6 可选瓣(coat/aniso/sheen/sss/transmission/emission) 近似 OpenPBR 的 base/specular/coat/fuzz/subsurface/transmission/emission | 规范化到 OpenPBR Surface，缺字段加法补齐（§4.4） |
| `Illumination` 四值轴 | `axis.rs:24` | Lit(0)/Stylized(1)/Unlit(2)/Custom(3)；注释明示 Stylized 是**轴上的值**，非竞争着色模型 | 作为 PBR/NPR/自定义四线的正交选择器（§3.5） |
| `lower_standard_material`（Bevy StandardMaterial→Record 1:1 直映） | `bevy_bridge.rs:40` | 外部材质翻译层 | **删**，外部材质统一经 OpenPBR Surface 导入烘焙（§4） |

**两条线的问题（本次重构要消灭的）**：
1. **固定线**：Mix/Layer 深度硬顶 ≤4，超限校验拒绝（人为天花板）；
2. **不固定线**：über lobe + 算术 + 其它 closure 不计数，图本身无界，lowering 时静默收口进 7-lobe ABI。

两套互不相干的定界机制（一个硬顶拒绝、一个静默收口），行为不一致，且都带全局硬顶的味道。**重构的收敛：取消运行时闭包图；运行时只保留一个开放 über 参数块（变长、按材质静态特化）；分层在烘焙期解析掉。既不留"图"要定界，也不留全局 MAX 来封顶——两条线并成一条，且这一条没有天花板。**

---

## 2. 对标优秀项目（借鉴点与取舍）

| 项目 / 技术 | 借鉴点 | Prism 取舍 |
|---|---|---|
| **Frostbite 分层材质** | 层栈在**烘焙/纹理空间**解析成单一参数集 + 遮罩，运行时不留图，层数天然不限 | **核心借鉴**：分层 = 同参数化 über 的无损 lerp，任意层数归一成逐纹素参数 |
| **Google Filament** | uber + 内建能量补偿 + 移动/Web 友好 | 取其单一 über 模型族 + 能量守恒，但去掉固定瓣数的天花板 |
| **UE5 Nanite / vis-buffer** | 逐 tile 材质分类，波前一致 | 直接复用（`classification.rs` 已有 9-way），性能地基 |
| **UE Virtual Texturing / idTech MegaTexture** | 虚拟纹理，距离 LOD 靠 mip 预过滤 | 复用现有 `texture_streaming`，距离降级全靠纹理，模型不变 |
| **UE 材质 permutation / Filament variant** | 按特征静态特化 shader，无运行时动态分支预算 | **核心借鉴**：每材质编译期特化成确切瓣集；这是"无界但确定"的实现 |
| **Activision 多散射 GGX（Turquin/Kulla-Conty）** | 高 roughness 能量守恒 | BSDF 必补项 + furnace CI |
| **Toksvig / LEAN / 高光抗锯齿** | 法线细节 → 等效 roughness | 距离细节丢失用它补，**取代**法线塌缩 |
| **OpenPBR Surface + MaterialX** | 2024 统一 BSDF + 作者标准，分组词汇表开放，可扩展 | **核心借鉴**：作者前端 + **运行时 über 的权威参数化**；现 7 瓣是其子集，缺口加法补齐（§4.4）|
| **UE5 Substrate** | slab 代数、作者端无界、模型可扩展 | **借其作者端无界与模型开放性，弃其运行时 per-pixel 预算裁剪与塌缩思路** |

**一句话**：作者层拥抱 OpenPBR/MaterialX（易用、无界）→ import 烘焙成 **OpenPBR Surface über**（一条线，无天花板）→ 每材质编译期静态特化成确切瓣集（无界但确定）→ 四线（PBR/NPR/混合/自定义）作为 illumination 轴正交响应复用同一 über → 距离/性能靠纹理预过滤 + 静态置换 + 采样缩放（无损、确定）。

---

## 3. 三层架构总览

```
┌─────────────────────────────────────────────────────────────┐
│ 作者层 (Authoring)  —  prism_render_material_authoring (新)   │
│  OpenPBR Surface 参数集 / MaterialX 节点图 / archetype 模板   │
│  作者端无界：层数/节点不限，lobe 词汇表可扩展                 │
└───────────────────────────┬─────────────────────────────────┘
                            │ import/bake：解析分层 → über 参数 + 遮罩纹理
┌───────────────────────────▼─────────────────────────────────┐
│ 烘焙层 (Bake)  —  prism_render_material (重构)                │
│  层栈（任意层数）→ 纹理空间逐纹素混合（同参数化 lerp，无损）  │
│  产出：开放 über 参数块（变长）+ VT 页引用 + lobe_mask        │
│  按材质声明的瓣集生成静态置换 key，无 tier、无全局 MAX        │
└───────────────────────────┬─────────────────────────────────┘
                            │ 开放 ABI v5（变长 über 参数块）
┌───────────────────────────▼─────────────────────────────────┐
│ 运行层 (Runtime)  —  prism_render_scene / prism_render_arch   │
│  vis-buffer per-tile 9-way 分类（已有）→ 波前一致 über 着色   │
│  每材质跑它编译期特化的确切瓣集（无运行时动态砍瓣）           │
│  距离 LOD = VT mip 预过滤 + 高光抗锯齿；贵瓣采样数按画质缩放   │
│  瓣数/特征在距离上恒定；绝无运行时瓣/层塌缩                   │
└─────────────────────────────────────────────────────────────┘
```

核心不变量（一条线、无硬顶、无塌缩）：
- **一条管线、一个 über 模型族**；没有闭包图、没有深度、没有 tier、**没有全局 MAX**。
- **界 = 每材质编译期静态特化**：代价是编译期常量，不是运行时逐像素预算。
- **降级只改输入，不改模型**：距离靠纹理 mip/VT，细节靠高光抗锯齿，性能靠静态置换 + 采样缩放。
- 分层在烘焙期解析成同参数化 über 的无损混合（任意层数），运行时不留图。

---

## 3.5 四条线：PBR / NPR / 混合 / 自定义（一条 über 上的正交响应，不是四条管线）

> 这是对"一条线"的澄清，不是回退。**线 = 管线；模式 = 响应。** 四种外观需求共享同一条 OpenPBR über + 同一 per-tile 分类 + 同一降级，差异只在"用这些输入算什么光照响应"，由 `axis.rs::Illumination` 轴（Lit/Stylized/Unlit/Custom，代码实测四值）+ `render_class` + 可选 `custom_program` 选择。

| 线 | 轴选择（代码） | 作者层（OpenPBR Surface +） | 烘焙产物 | 运行时响应 | 降级 |
|---|---|---|---|---|---|
| **PBR 物理** | `Illumination::Lit(0)` | OpenPBR 全瓣参数 | OpenPBR über 块 + lobe_mask | 能量守恒 BSDF（多散射 GGX、菲涅尔、瓣叠加） | VT mip + 高光抗锯齿 + 采样缩放（§7） |
| **NPR 风格化** | `Illumination::Stylized(1)` | OpenPBR base + Prism 私有 FACE 瓣 + ramp/描边参数 | 同 über 块 + FACE 瓣 + ramp LUT(VT 页) | ramp 量化 / SDF 面阴影 / 描边：对**同一光照输入**的非物理响应 | ramp LUT mip；描边宽随距离；瓣集恒定 |
| **混合 Hybrid** | 逐材质/逐区域 `illumination`（PBR 底 + NPR 覆盖，遮罩选择） | 两套响应参数并存 + 选择遮罩 | 单一 über，illumination 可按区域切 | per-tile 分类把 Lit/Stylized 分到不同波前，避免发散 | 两路各自无损降级，合成相加 |
| **自定义 Custom** | `Illumination::Custom(3)` + `custom_program` 句柄 | 作者挂自定义 WESL 片段/子图（编译期编入置换） | über 参数 + `custom_program` id | 走自定义着色入口，仍**复用 OpenPBR über 解包 + per-tile 分类** | 自定义程序声明自己的采样缩放钩子 |

**统一不变量（四线共享，不复制管线）**：
- 同一 **OpenPBR über 参数解包**（`material_unpack.wesl::prism_unpack_surface`，变长、mask 驱动）。
- 同一 **per-tile 9-way 分类**（`classification.rs` 已含 `Npr` / `Custom` 类，`material_classification.wesl` prefix-sum/scatter，无 readback）。
- 同一 **降级**（VT mip + 高光抗锯齿 + 静态置换 + 采样缩放，无损、无塌缩、无 per-pixel 裁剪）。
- 同一 **双线对拍**（CPU 金标准 + WESL twin），四线响应各自进回归门禁。

**为什么不拆成四条管线**：拆管线会复制解包/分类/降级/对拍四套基础设施，permutation 与维护成本翻倍，且混合线需要在同一画面里逐区域切换——四管线反而要跨管线合成。正交轴方案下，混合只是"同一 über 的 illumination 逐纹素/逐区域取值"，per-tile 分类天然处理波前发散。这正是 `axis.rs` 注释"Stylized 是 illumination 轴上的值、不是竞争着色模型"的工程含义。

---

### 3.5.1 PBR 物理线（深化：性能 / 效果 / 易用）

> 对标 **Frostbite**（Lagarde/de Rousiers 统一 PBR）、**Filament**（Google 文档级 BRDF 规范）、**Activision/Kulla-Conty 多散射能量补偿**、**Belcour-Barla 薄膜虹彩**。目标：能量守恒、远近稳定、跨平台确定。

- **效果（必达物理项，进 §10 CI 门禁）**：
  - 多散射 GGX 能量补偿（Kulla-Conty 2017 解析近似 / Activision 查表），高糙度金属不发黑；
  - coat 分层含吸收与色移（OpenPBR `coat_color/coat_ior/coat_darkening`，§4.4），湿表面/车漆正确变暗；
  - 各向异性（Burley GGX，`anisotropy/anisotropy_rotation`）、sheen（Estévez-Kulla charlie 布料）、SSS（Christensen-Burley 可分离近似）；
  - thin_film 虹彩（Belcour-Barla 2017）作为开放词汇表首个新瓣（§4.4/§10）。
- **性能**：多散射/IBL 全走**预积分 LUT 或解析近似**（一次查表，不迭代）；coat 二层但未用则编译期静态置换掉（§8）；远处 **Toksvig/LEAN** 把法线细节转等效 roughness 消闪烁（§10）；视角无关项（diffuse/irradiance/SSS 漫透）可进 §14.3 对象空间缓存，高光走 split-sum 屏幕空间。
- **易用**：`Metal/Skin/Glass/CarPaint/Water` archetype 填参（§4.2），OpenPBR 全字段作者端可见；furnace 白炉子 CI 守能量守恒，美术改参不破物理。

### 3.5.2 NPR 风格化线（深化：性能 / 效果 / 易用）

> 对标 **Arc System Works《Guilty Gear Xrd》**（顶点色控阴影 / 法线编辑 / 背面挤出描边）、**miHoYo《原神》**（ramp 图集 / matcap 金属 / SDF 面阴影）、**UE Toon / Blender NPR**。目标：风格可控且与同一 über 光照输入解耦，不另造 NPR 管线。

- **效果**：
  - **光照量化 ramp**：缓存"量化前的连续辐照度"（N·L 带状化前），采样期查 ramp LUT 施加量化曲线——连续量缓存更稳、避免档边闪烁（§14.3 已定此切分）；
  - **SDF 面阴影（FACE 瓣，`surface.rs::softness`）**：miHoYo 式脸部阴影随光照方位平滑过渡；FACE 是 **OpenPBR 之外的 Prism 私有瓣**（§4.4），由 `Illumination::Stylized(1)` 消费；
  - **描边**：屏幕空间几何（背面挤出 / 后处理法线·深度边缘），**永远屏幕空间**，宽度随距离收敛；
  - **风格化高光**：阶跃锐边 / matcap / Kajiya-Kay 发丝各向异性高光。
- **性能**：ramp/matcap/FACE 均为 **LUT 或单张贴图采样**（廉价）；量化在采样期逐像素查表（便宜且防档跳）；描边为一次后处理；NPR 可缓存项（ramp 输入辐照度、材质色、面阴影采样）进 §14.3，描边与锐边高光屏幕空间。**不变量：PBR/NPR 精度从架构上就一致，无按线特判**——缓存只存传递无关的感知均匀入射辐照度（四线共用一份，§14.3/§14.8），ramp/BRDF 等响应全在采样期施加；NPR 台阶锐度由采样期对解码辐照度的屏幕梯度做解析抗锯齿生成，**与缓存位深解耦**（§14.8），故既不降 NPR 精度、PBR/NPR 又走同一条路径。降级（§7）只作用于采样数/纹理 mip，不碰响应质量。
- **易用**：`Toon` archetype；ramp 曲线美术可画、matcap 直给贴图；描边参数与物理参数解耦互不污染；美术不碰 BSDF 也能出稳定风格。

### 3.5.3 混合线（深化：性能 / 效果 / 易用）

> 对标 **UE 多 ShadingModel 混排 / `ShadingModelFromMaterialExpression`**、写实场景 + 卡通角色同屏。目标：一个画面内 PBR 与 NPR 响应共存，仍是一条 über。

- **效果**：逐材质 / 逐区域 `illumination` 取值（PBR 底 + NPR 覆盖，遮罩选择），同一 über 块里按区域切响应，支持"写实环境 + 风格化主角"。
- **性能（关键）**：**per-tile 9-way 分类**（`classification.rs` 已含 `Npr/Custom`，`material_classification.wesl` prefix-sum/scatter 无 readback）把 Lit/Stylized **分到不同波前**，消除 warp 内分支发散——这是混合高性能的根（对标 Nanite 材质 per-tile classify）。缓存层按 `illumination` 分通道存，命中与否不破坏一致性（§14.3）。
- **易用**：美术画 `illumination` 遮罩即可；合成期按遮罩相加；无需跨管线合成，无额外作者心智。

### 3.5.4 自定义线（深化：性能 / 效果 / 易用）

> 对标 **UE Custom HLSL node / Material Function / 自定义 ShadingModel**、**Unity ShaderGraph Custom Function**。目标：开放逃生舱口，但不破一条线的解包/分类/降级/对拍基建。

- **效果**：作者挂自定义 WESL 片段/子图，**编译期编入置换**；仍复用 OpenPBR über 解包（`material_unpack.wesl`）+ per-tile 分类，自定义只替换"用输入算什么响应"。
- **采样期响应契约**（与 §14.9/§16 统一）：缓存只存传递无关入射辐照度，自定义程序实现 `respond_custom(surface, inc, aux, view)`（采样期响应，必实现）/ 可选 `shade_cacheable_aux(surface, light_env)`（声明额外视角无关量进缓存）；**不实现 `respond_custom` → 自动全屏幕空间安全退化**。
- **性能**：编译期编入=静态置换，无运行时解释；自定义声明自己的采样缩放钩子（§7）与 cacheable 切分（§14），默认不拖慢其它三线。
- **易用**：节点图或片段两种入口；默认安全（屏幕空间、无需懂缓存）；契约为 opt-in 加速，渐进采用。
- **升级为 PCM 扩展点**：自定义线不再是“唯一逃生舱口 + 硬编码 `custom_program` 句柄”，而是 **PCM（§5）里 response-mode/custom 一类扩展点**；`respond_custom`/`shade_cacheable_aux` 契约升为该 kind 的类型化 hook，受限 Rust 瓣（§5.4.1，`proc-macro → WESL`，CPU 金标准即原生 Rust）自动获双线对拍（裸 WESL 为 `golden="none"` 的不安全逃生口，仅烘焙期）。

### 3.5.5 四线共享：性能 / 效果 / 易用总矩阵与治理

| 维度 | PBR | NPR | 混合 | 自定义 |
|---|---|---|---|---|
| **效果关键项** | 多散射 GGX / coat 分层 / 虹彩 | ramp 量化 / SDF 面阴影 / 描边 | 区域级 PBR⊕NPR 共存 | 作者定义响应 |
| **性能手段** | LUT/解析近似 + Toksvig + split-sum | LUT/matcap + 后处理描边 | per-tile 分波前消发散 | 静态置换编入 + 契约切分 |
| **§14 缓存切分** | diffuse/irradiance/SSS 缓存，高光屏幕 | ramp 输入辐照度/面阴影缓存，描边屏幕 | 按 illumination 分通道缓存 | 契约声明 cacheable，否则屏幕 |
| **易用入口** | Metal/Skin/Glass… archetype | Toon archetype + ramp 曲线 | illumination 遮罩 | 节点图/WESL 片段 |
| **对拍门禁** | furnace + 字节对拍 | ramp/面阴影响应对拍 | 分区域各自对拍 | 自定义输出对拍 |

**治理（四线不加管线维度）**：四线共享同一 permutation 池，键是 `axis.rs::SpecializationId`（`illumination[0..8]` / `closure_mask[8..40]` / `render_class[40..48]`，**无 tier**）；NPR/Custom 只是 `illumination` 值与 `render_class`/`custom_program` 的取值，不新增管线轴。permutation 成本由内容哈希去重 + archetype 聚类 + 按需异步编译 + 全瓣 über 回退吸收（§8.4）。**四线正交、一条基建**：解包/分类/降级/对拍/缓存五套地基只建一次。

---

## 4. 作者层：OpenPBR / MaterialX 前端（易用 + "无界"的落点）

### 4.1 新 crate `prism_render_material_authoring`
- 输入：OpenPBR Surface 参数集，或 MaterialX 节点图（`.mtlx`），**层数/节点/瓣类不限**。
- 输出：烘焙层的输入（层栈 + 参数曲线 + 遮罩来源 + 声明的瓣集）。
- OpenPBR 与现 über 近乎一一对应（base/specular→core，coat→CLEARCOAT，fuzz→SHEEN，subsurface→SUBSURFACE，transmission→TRANSMISSION，anisotropy→ANISOTROPY，emission→EMISSION，Prism 扩展 FACE 给 NPR）；**新瓣可作为开放词汇表的一等成员加入**，不必塞进固定槽。

### 4.2 archetype 模板
预置 `Metal / Skin / Cloth / Glass / Foliage / Toon / CarPaint / Water`，美术选模板填参，不从空图搭。模板也作为**置换聚类锚点**（见 §8.4），把常见瓣集组合收敛成少量热门 permutation。

### 4.3 "无界"如何满足而不失确定性
作者端的层数/节点/瓣类自由，在 **import/bake** 阶段被解析（见 §6）成 über 参数 + 遮罩纹理 + **该材质声明的瓣集**。运行时按这个瓣集**编译期静态特化**：用到哪些瓣就编哪些瓣，不用的不编。于是：
- 没有全局天花板（满足"不要有界"）；
- 每材质代价编译期已知（满足"确定、跨平台"）；
- 不做运行时逐像素砍瓣（否决 Substrate 式预算裁剪）。
自由在作者侧，特化在编译期，中间靠烘焙——这是关键区别。

### 4.4 OpenPBR Surface 规范映射与补瓣（加法，不改架构；终态为 OpenPBR 超集）

运行时 über 以 **OpenPBR Surface** 为规范核心，终态是其**真超集**：`Prism über = OpenPBR Surface（全字段） ∪ { FACE } ∪ { 开放词汇表后续瓣 }`。现 `surface.rs` 的核心 12 words + 7 瓣只是今天未补齐的**实现子集**；补齐 = 把现有 4-word lobe 扩成 OpenPBR 全字段 lobe，走变长打包 + 静态置换，**是加法，不改模型族**。补齐完成后，OpenPBR 字段构成超集的「规范核心区」，FACE / 后续开放瓣构成「Prism 扩展区」（位段划分见 §4.5）。

| OpenPBR 组 / lobe | 现 über 字段（`surface.rs`） | 状态 | 待补字段（加法） |
|---|---|---|---|
| **base**（diffuse+metal） | `base_color` / `metallic` / `perceptual_roughness` | 子集 | `base_weight`、`diffuse_roughness`（Oren-Nayar 糙漫反射） |
| **specular** | `reflectance`（IOR 代理）/ `anisotropy` / `anisotropy_rotation` | 子集 | `specular_weight`、`specular_color`、显式 `specular_ior`（取代 reflectance 代理） |
| **coat** | `clearcoat` / `clearcoat_roughness` | 子集 | `coat_color`、`coat_ior`、`coat_roughness_anisotropy`、`coat_normal`、`coat_darkening` |
| **fuzz** | `sheen` | 子集 | `fuzz_color`、`fuzz_roughness` |
| **subsurface** | `subsurface` | 子集 | `subsurface_color`、`subsurface_radius`（vec3，RGB 自由程）、`scatter_anisotropy` |
| **transmission** | `transmission` / `thickness` / `index_of_refraction` / `dispersion` | 较全 | `transmission_color`、`transmission_scatter`、`thin_walled` 标志 |
| **emission** | `emission`（4 words） | ✓ | `emission_luminance` 单位统一（nit / 相对） |
| **thin_film**（虹彩） | — | **缺整瓣** | 新增 `thin_film_weight` / `thin_film_thickness` / `thin_film_ior`（§10 首个开放瓣示例） |
| **geometry** | `normal_scale` / `alpha_cutoff` + 法线纹理 | 子集 | `geometry_opacity`、`geometry_thin_walled`、`geometry_coat_normal`（随 coat 补） |
| **FACE**（NPR） | `softness`（SDF 面阴影） | ✓ | OpenPBR 之外的 **Prism 私有瓣**，NPR 专用，**不并入 OpenPBR**，与之共存 |

要点：
- **开放词汇表让补瓣是加法**：新字段扩进对应 lobe 的变长块，位掩码/静态置换天然容纳，老材质不受影响（它们的 lobe_mask 不含新位）。
- **thin_film 作为开放瓣的首个落地示例**（§10）：验证"新物理项 = 新 lobe 进词汇表"的扩展路径。
- **FACE 保持 OpenPBR 外**：OpenPBR 是物理 BSDF 标准，不含 NPR；FACE 作为 Prism 私有瓣，由 `Illumination::Stylized` 线消费（§3.5），与 OpenPBR 物理瓣在同一 über 块里共存。
- **删 `lower_standard_material`**：外部材质（含 Bevy `StandardMaterial`、glTF）统一经 OpenPBR Surface 导入器映射，不再逐格式写 1:1 翻译层。
- **超集 = PCM 的一个 kind**：OpenPBR 补瓣只是 `kind=lobe` 的加法扩展；完整的九类扩展点能力基座（瓣只是其一）见 §5。

### 4.5 超集成立的三条硬约束（把"是超集"从口号变成可验证契约）

"Prism über 是 OpenPBR 超集"要可验证、可维护，必须满足三条硬约束，缺一则退化为"又一个私有材质模型"：

1. **纯 OpenPBR 子集合规对拍（Conformance Gate）**：对"只用 OpenPBR 字段、不含任何 Prism 扩展位"的材质，Prism 运行时的 BSDF 响应必须与 OpenPBR 参考实现（ASWF `OpenPBR` reference / MaterialX 节点）在 furnace + 定向光对拍下字节级一致。这进 CI 门禁（§10）——证明"关掉扩展，Prism 就是标准 OpenPBR"，这是"真超集"的充要前提。
2. **扩展位段隔离（Bit-Region Isolation）**：`LobeMask` 开放枚举按**规范核心区 / Prism 扩展区**静态分段——OpenPBR 规范瓣占低位固定区段（随上游版本演进只增不改语义），FACE 与开放词汇表后续瓣占高位扩展区段。两区位段互不重叠，变长解包 (`prism_unpack_surface`) 按区段路由，杜绝"私有瓣污染规范语义"或"上游新字段撞车扩展位"。
3. **向下投影（Lossy Downgrade Projection）**：提供 `project_to_openpbr`——把含扩展瓣的 Prism über **有损投影**成纯 OpenPBR 参数块（丢弃 FACE/私有瓣、把私有效果近似并入最接近的规范瓣），用于导出到外部纯 OpenPBR/MaterialX 消费端。投影是"超集→基集"的单向收敛，须显式标注有损项，不保证视觉等价、只保证"是合法 OpenPBR"。

> **维护义务**：OpenPBR 是活标准（版本演进）。超集定位意味着 Prism 承担"跟随规范核心区"的义务——上游新增/修订字段须在规范核心区加法吸收并过合规门禁；私有扩展只能在扩展区演进，永不侵入核心区语义。

> **升级为能力基座**：§4.5 三约束不止约束“瓣”这一类扩展——应提升为**全管线能力/扩展点基座（PCM）的全局不变量**，统辖九类扩展点（瓣 / 参数通道 / 响应模式 / 传递重映射 / 缓存量 / 数据提供者 / 降级策略 / 作者节点 / 合成 pass），且 OpenPBR 核心瓣本身也自举为 `KHR` 命名空间的注册能力。详见 §5。

### 4.6 作者迭代闭环与可观测性/验证（易用 + 易维护的 AAA 落点）

> 对标 **UE5 Material Editor**（实时预览 + 指令/采样器统计 + 平台 stats）、**Frostbite** 材质验证与 FrameGraph 调试、**RenderDoc / PIX** 抓帧式着色调试、**Guerrilla Decima** 的置换/变体可视化工具、**id Tech** 材质热重载。取其“迭代快 + 可观测 + 可验证”，但全部做成**内建、确定、跨平台**（wgpu + WESL），不依赖外部抓帧，也不引入编辑器专用近似路径。

前文（§5–§14）把性能/易扩展打到次时代地基水准；AAA 材质系统的另一半竞争力在**迭代速度与可调试性**——本节把易用/易维护补到同一水准，且全部复用既有不变量（烘焙期单态化、pass 级 uniform 不进 permutation、双线对拍），不新增任何运行时成本。

**1. 热重载与即时预览（易用）**：manifest / MaterialX 节点图 / 受限 Rust 瓣改动触发**烘焙期增量重编**（仅改动材质，内容哈希命中则跳过，§8.4）；编译在途由“全瓣 über 回退 PSO”兜底着色（§8.4）——改参不黑屏、不卡顿，后台编完热切换。关键不变量：热重载走**同一条烘焙管线**，无“编辑器专用快速近似”分支，从结构上杜绝 editor-vs-runtime 漂移。

**2. 内建确定性调试视图（效果调试 + 易维护）**：一组只读旁路可视化，均为 **pass 级 uniform 切换、不进 permutation key**（§8.3），跨平台字节确定：
- **lobe_mask / 置换热图**：每像素实际特化的瓣集 + 所属 PSO + permutation 热度（定位“某材质悄悄炸出一堆变体”，§8.4）；
- **解耦着色缓存视图**（§14）：Resident / Stale / 未命中着色、生效 mip 层级、overshading 热图——把 §14 的缓存行为变成肉眼可见；
- **per-tile 分类/波前视图**（§8.3/§8.5）：Lit / Stylized / Custom 波前分布与 occupancy，直接看四线混排是否发散；
- **能量守恒残差热图**（§10）：furnace / 定向光对拍残差就地上色，哪里不守恒一目了然；
- **VT 缺页 / mip 回退视图**（§9）：缺页与降细节回退可视化。
相当于“内建的 RenderDoc/PIX”，但确定、跨平台、且随每个 PCM kind 自动获得（见第 5 点）。

**3. 验证门禁前移（shift-left，易维护）**：§4.5 合规对拍 / §10 furnace / §11 双线 twin / §8.4 cost 预算这些 CI 门禁，在作者**保存时即跑增量子集**（只测改动材质），把回归反馈从“提交后 CI”提前到“编辑器内即时标红”。manifest 的 `requires_caps` 在作者端即对**目标平台矩阵**做能力协商预检（§5.3）——弱平台不满足当场报错，而非上线才暴雷。

**4. 代价可观测（性能 + 易维护）**：每材质 / 每 permutation 的**静态代价**（瓣数、指令估计、采样器数、VT 页预算、cache-aux 字长）在作者端可见并入 `cost_hint` CI（§5.5/§8.4）——UE5 式指令计数，但确定且跨平台可枚举；预算超标在作者端即报，防某扩展悄悄变贵。

**5. 对扩展一视同仁（易扩展的乘数效应）**：以上热重载 / 调试视图 / 验证门禁 / 代价可观测**不为任何单一 kind 定制**——任何新注册的 PCM 能力（新瓣 / 新响应模式 / 新 provider……§5）凭其 manifest 的类型化 hook 与 schema（§17.5）**自动继承整套工具链**，无需逐 kind 手写编辑器/调试器支持。这是把 §5“单一真相源 + codegen”从运行时推广到工具链的红利：扩展越多，工具复用越划算。

> 五维度收益：**性能**=回退 PSO 零卡顿迭代 + 代价可观测防悄悄变贵；**效果**=furnace/对拍残差与缓存/波前行为就地可视化；**易用**=热重载 + 即时预览 + 当场合规/能力反馈；**易扩展**=新 kind 自动继承全套工具链（无逐 kind 手写）；**易维护**=单一烘焙管线无 editor/runtime 漂移 + 内建确定性调试取代外部抓帧。

---

## 5. PCM 能力基座：全管线能力/扩展点统一注册（“超集”的完整模型）

§4.4/§4.5 的"超集"目前只覆盖**一类**扩展——往 über 加瓣（`closure_mask` 的位）。但 §8.5 六步执行路径 + 作者层（§4）+ VT（§9）里，今天有**九种**各自为政、多数封闭的"扩展机制"，每种用不同的硬编码枚举/散落特判定界——与 §1 批判的"固定线硬顶 + 不固定线静默收口两套互不相干机制"同病。PCM 的升级：**把九类扩展点统一成同一套声明式能力契约，瓣只是其中一类**；全部烘焙期单态化消失，运行时仍是零分发的特化 WESL。

**九类扩展点（kind）× §8.5 落点 × 今天的封闭定界方式**：

| 扩展点 kind | §8.5 落点 | 今天怎么扩（封闭/散落） | 锚点 |
|---|---|---|---|
| **lobe 瓣** | 步骤3 瓣求值 | 改 `LobeMask` enum + `surface.rs` | surface.rs:153 |
| **channel 参数通道** | 步骤1 解包 | 手改 ABI header 动态段 | §17.1 |
| **response-mode 响应模式** | 步骤2 分类 + 步骤5 响应 | 改 `Illumination` **封闭四值枚举** | axis.rs:24 |
| **transfer 传递重映射** | 步骤5/6 | 无正式位置（塞 respond/custom） | §14.3 |
| **cache-aux 缓存量** | 步骤4 光照源 | `shade_cacheable_aux` 散在契约 | §14.3 |
| **provider 数据提供者** | 步骤1 输入 / §9 | 逐种硬写（SDF 图集/ramp LUT/测量页） | texture_streaming/* |
| **downgrade 降级策略** | 步骤1 降级 / §7 | 全局统一，扩展瓣无自定义钩子 | §7.2 |
| **author-node 作者节点** | §4 作者层 | 逐格式写映射 | §4 |
| **composite 合成 pass** | 步骤6 合成 | 描边等硬编进步骤6 | §8.5 步骤6 |

**九个点、九套定界 → 一个注册表**：`PRISM_lobe_face` 只是 `kind=lobe` 的一条 manifest；新增响应模式（`X_studio_mode_watercolor`）、动态通道（`PRISM_channel_wetness`）、描边 pass（`PRISM_pass_outline`）流程完全同构，走注册表而非改封闭 enum。超集（加瓣）由此降为 PCM 的一个 kind。

### 5.1 核心自举（dogfood）：消灭"核心 vs 扩展"特判
OpenPBR 的 base/specular/coat/... 本身也注册为 `kind=lobe` 能力，命名空间 `KHR`（规范核心）；FACE 为 `PRISM`，私有瓣为 `X_studio_`。代码里不再有"核心瓣硬编码、扩展瓣走注册"的二分——只有一个注册表，区别仅在**命名空间 + 稳定级 + 位区**（§4.5 位段隔离）。这是 Vulkan（core 由 EXT 晋升）与 USD（内置 schema 也只是 schema）的红利：**唯一真相源 + codegen**，把 §17.5 自描述 schema 升为**所有 kind 的 manifest 格式**，`surface.rs`↔WESL 手写漂移从结构上消失。§4.5 三约束自然成为 PCM 全局不变量（合规对拍 = 只用 `KHR` 项 → 字节级等于 OpenPBR 参考；位区隔离 = 命名空间各占独立位/槽区、永不回收；向下投影 = 丢非 `KHR` 项投影回纯 OpenPBR）。

### 5.2 性能不变量：烘焙期单态化，不是运行时分发（对九类一视同仁）
PCM 的全部治理机器（命名空间解析、能力协商、used/required、位区分配）在**烘焙/加载期**求值完毕并蒸发；运行时只剩**已特化好的那一份 WESL**，零 dispatch、零解释、零注册表查询——这是与"插件 VM"划清界限的红线。严守 §8.5"permutation 边界只由步骤1/3 决定"：transfer/cache-aux/provider/downgrade/composite 全部编入烘焙期置换 key 或 pass 图，运行时无新增 PSO、不改分类 key。置换聚类按"扩展组合原型"做（§8.4）。
> **唯一热路径结构改动——response-mode**：步骤 2 的 9-way 分类今天是定值（`material_classification.wesl`）；开放响应模式意味着**类数本身变成注册驱动**（prefix-sum/scatter 跑在"注册的 class 集"上）。它仍是**烘焙期确定的固定表**，非运行时动态分支——分类表在管线构建期生成。这是本升级唯一触碰热路径结构的点，须单独立项并验证不回归 §8.5 波前 occupancy。

### 5.3 能力协商（capability）：跨平台确定性降级
借 Vulkan feature/limit 模型。每条 manifest 声明 `requires_caps`（对象空间缓存 / VT / f16 / compute scatter 等），目标平台广播 caps，**烘焙/加载期**协商：`required` 不满足 → 烘焙/加载期**显式报错**（如 VK required ext），不留运行时惊喜；`used`（非 required）不满足 → **graceful ignore**（glTF 式），材质投影回不含此扩展的变体。弱平台由此从 §8.5"关缓存只走步骤4(b)"升级为**全管线能力感知的确定性降级**——一条材质在 Web/移动上特化成什么，是烘焙期可枚举、可 CI 的事实。

### 5.4 manifest 与 hook ABI（声明式、codeless）
每条扩展 = 一份 TOML manifest（codeless）。字段分**不可变身份**与**可变归属**两组，是本节相较早期草案的关键升级——命名空间不再是 `id` 的前缀，而是独立字段，晋升只翻字段、不改身份：

| 组 | 字段 | 可变性 | 语义 |
| --- | --- | --- | --- |
| **身份（永久钉死）** | `uid` | 不可变 | 全局唯一稳定标识（内容无关的分配号），一经发放永不改写，是所有交叉引用 / 依赖 / 位掩码的锚 |
| | `registry_slot` | 不可变 | 永久钉死的位 / 槽区，永不回收；与 `uid` 一一绑定 |
| **归属（可晋升）** | `namespace` | 可变（枚举） | `KHR`（规范核心）/`EXT`（跨厂）/`PRISM`（本体私有）/`X_studio`（工作室私有） |
| | `vendor` | 可变 | 发布方标识（晋升 / 并厂时更新） |
| | `stability` | 可变 | 实验 / 候选 / 稳定级 |
| | `id` | **派生** | 由 `namespace` + 语义名规范化得到的展示名（如 `KHR_lobe_face`），仅供人读与 glTF / MaterialX 互操作，**不作身份锚** |
| **能力描述** | `kind` / `version` / `requires_caps` / `energy_policy` / `view_dependent` / `replaces_core` | — | 同前；`version` 独立于 OpenPBR 版本轴；完整 schema = §17.5 |

**晋升 = 翻字段，不是改名**：`X_studio → EXT → KHR` 只改 `namespace`/`vendor`/`stability` 三个可变字段，`uid` 与 `registry_slot`（位区）保持不变——因此晋升**不破坏任何交叉引用、不触碰已钉死的位掩码绑定**，根除了“晋升即重命名即全链路失效”的自相矛盾（早期把命名空间塞进 `id` 前缀会导致此问题）。`id` 作为派生展示名随 `namespace` 自动重算，供 glTF 风格互操作，不参与任何绑定。

**§4.5 三约束落为字段谓词**：命名空间升为字段后，§4.5 的全局不变量从“人工审读的约定”变成**可机检的字段谓词**——合规对拍（约束 1）= “凡 `namespace==KHR` 必与 OpenPBR 参考实现字节对拍一致”；位区隔离（约束 2）= 位分配以 `namespace` 为分区键静态控制；晋升闸门 = 一台只允许 `X_studio→EXT→KHR` 单向跃迁的字段状态机。CI 直接枚举字段即可守门，无需人工审读。

每类 kind 有类型化 hook 签名（lobe: unpack/classify/sample/respond；provider: produce_page；transfer: remap；composite: record 等）。受限 Rust 瓣（§5.4.1）编译期由 `proc-macro` 产出 WESL 孪生、CPU 金标准即作者原生 Rust → `golden="auto"`，任何 kind 自动获字节级 twin 对拍（§11）；裸 WESL 为不安全逃生口（`golden="none"`、CI 标红、**仅烘焙期**，运行时永不加载任意 shader——安全边界）。ABI header 加 `extensions_used_mask`/`extensions_required_mask`；OpenPBR 核心位与 PCM 注册扩展位走双命名空间 / 双版本轴，结构性不撞车（根除 §15 一条风险）。

#### 5.4.1 行为作者模型：节点图组合 → 受限 Rust 瓣 → 裸 WESL（取代"受限 DSL"）

早期草案把自定义行为寄托于一门"受限 DSL"。本版**取消自造 DSL**：DSL 唯一买到的东西是"单源 → CPU/WESL 字节一致"，而**布局**一致已由 §17.5 schema 从结构上保证，只剩**行为数学**需要单源。与其发明一门新语言，不如直接用团队本就在用的 Rust。行为作者按"由易到难、由封闭到开放"分三档：

| 档 | 作者写什么 | 覆盖面 | 新金标准/孪生 | 落点 |
|---|---|---|---|---|
| **① 节点图组合** | 在节点图里连接**已验证的瓣/原语**（数据，非代码） | ~95%，OpenPBR 超集 | 不需要（组合是确定性降级，只验证降级器一次） | §4 作者层 / §6 烘焙 |
| **② 受限 Rust 瓣** | 用**受限 Rust 子集**写新瓣行为数学，`proc-macro → WESL` | 需要新数学的长尾 | 需要（仅这一个瓣一次性） | 本节 hook |
| **③ 裸 WESL 逃生口** | 直接手写 WESL + 手写 CPU 金标准 | 极端 / 临时 | 手写双份，`golden="none"` CI 标红 | §11 |

**为什么是 Rust 而非另一门 DSL**：CPU 金标准 `prism_render_shading` 本就是原生 Rust。让 ② 的作者直接写 Rust，则**金标准侧零转译**——真值就是作者亲手写的那份 Rust，最可信；**只有 GPU 侧的 WESL 孪生是生成物**。这比"WESL 子集 → 转译回 CPU"安全（后者让"真值"本身成了生成产物）。先例：CubeCL（Rust→WGSL）、rust-gpu（Rust→SPIR-V）。

**② 是"受检子集"，不是任意 Rust**：为可转译 WESL + 可字节对拍，② 禁止 heap / `dyn trait` / 递归 / 无界循环，`proc-macro` 必须**拒绝并报错**而非静默降级。浮点字节确定性（sin/cos/exp/FMA/舍入/fast-math）是语言无关的硬问题，两侧共用一个 `no_std` 软件数学库消解。

**离线 author-node 例外**：只在烘焙期跑、永不进 GPU 运行时的扩展点（§5 的 author-node / provider / transfer 烘焙类）无需 WESL 孪生，可放开到近乎完整 Rust；但凡进金标准 crate 的部分仍须过同一确定性数学库闸，以免污染真值。

> 边界：节点图（①）是**组合层**，发明不出新闭式瓣；真正的新数学必然下沉到 ②/③。① 覆盖绝大多数作者，②/③ 只服务长尾；`golden="auto"` 适用于 ②（同源生成 WESL），`golden="none"` 适用于 ③。节点图须为**受类型约束的图**（端口带类型、连接受限），且烘焙期做**规范化线性化**（拓扑序确定、浮点求和顺序稳定），否则作者可连出违反 §3.5 四线正交 / 导致瓣坍塌的结构，或引入 CPU/GPU 浮点顺序漂移。

#### 5.4.2 编辑器渲染插件与"打包即烘焙"（双运行时 · 同源 WESL · 信任边界）

②/③ 的作者产物如何进引擎？**不是运行时插件 VM（§5.2 红线），而是编辑器期加载、打包期烘焙**。关键是分清**两个运行时**：

| | 编辑器运行时（开发期） | 出货运行时（打包后） |
|---|---|---|
| 渲染插件 | **可动态加载 / 热重载** | 不存在插件 |
| dispatch | 允许动态（图快速迭代） | 零 dispatch（§5.2/§8.5） |
| 着色来源 | 插件提供的 WESL **动态组装 / 编译** | 烘焙出的静态单态化特化 |
| 字节确定性 | 预览可放宽 | 必须双线一致（§11） |

**打包 = 烘焙**：出货打包这一步**就是**单态化——把编辑器里动态用到的瓣冻结成静态 WESL + 零 dispatch 的每材质特化，进最终构建；**插件本身不随出货**，只有其烘焙产物进包。

**唯一真正的难点 = WYSIWYG 同源**：编辑器动态预览与打包烘焙**必须同源于一份 WESL**，只是两种编译策略（编辑器动态编译 / 打包静态单态化）。预览**绝不能走插件自带的 native 旁路**，否则作者所见 ≠ 出货所得。这与 §4.6"热重载走同一条烘焙管线、无编辑器专用近似分支"是同一条不变量，此处把它推广到**第三方渲染插件**。

**门禁落在打包口，不在预览**：凡会被烘进出货的 ②/③ 产物，必须过 CPU 金标准 + WESL 孪生 + 确定性数学库闸 + §11 字节对拍；编辑器预览可不要求字节确定（它只是预览）。好处：确定性高门槛只压在打包这一刻，开发期迭代可以快。

**插件边界 = 信任边界**：原生渲染插件是把本地代码灌进编辑器且最终喂给金标准 crate（真值）。第三方插件须**签名受信**或 **wasm 沙箱**，且其原语须过确定性闸 + CI 字节对拍后方可启用——否则污染的是全链路真值，比运行时出错更难查。

> 优先级：**中（易用 / 易扩展杠杆，P2+）**。① 节点图组合 + ② 受限 Rust 瓣的 CI 字节对拍是地基（随 §5 PCM 落地）；编辑器渲染插件 + 打包即烘焙是其**分发形态**，在第三方作者需求出现时再落。它不新增运行时架构，只新增"可加载的 PCM 包格式 + loader + 信任 / 沙箱策略"。

### 5.5 五维度小结
- **性能**：运行时零分发；成本全在烘焙期，archetype 置换聚类 + 内容哈希去重吸收（§8.4）；`cost_hint` 进 CI 防某扩展悄悄变贵。
- **效果**：FACE 升为 `PRISM_lobe_face@2`（SDF 图集 provider + 方向场）；glint/薄膜/测量 BRDF/水彩描边各作对应 kind 落地，均进 §10/§11 门禁，效果不设天花板。
- **易用**：三级作者入口（archetype → MaterialX 节点图 → 受限 Rust 瓣，§5.4.1；裸 WESL 为逃生口）；`required` 不满足给可诊断报错；MaterialX 往返（§17.6）对未知扩展优雅降级。
- **易扩展**：新增任何一类能力 = 写一条 manifest，不碰封闭 enum；晋升路径 `X_studio_ → EXT_ → KHR_`（thin_film 可作首个走完晋升路径样例）。
- **易维护**：单一真相源 + codegen 杀漂移；两条版本轴（OpenPBR 规范轴 / PCM 扩展轴）显式分离；合规门禁自动枚举 `KHR` 字段。

> 优先级：**高（架构杠杆）**。把 §4.4/§4.5 的"超集"从"能加瓣"升级为"整条管线被同一套治理扩展"。唯一硬骨头是 §5.2 的 response-mode 分类注册化（动热路径，单独立项，见 §13 P2-j）；其余八类严守烘焙期单态化、运行时零新增。诚实代价：需为六步执行模型 + 作者层 + VT 各定义稳定 hook ABI，前期设计量明显更大。

---

## 6. 烘焙层：分层 → 开放 über（一条线的落点，无 N 层硬顶）

这是"两线合一 + 无硬顶 + 无塌缩"的核心。把运行时闭包图彻底移出运行时：

### 6.1 分层解析为纹理空间混合（任意层数）
- 层栈（**任意层数，无 MAX_LAYER**）在烘焙期按遮罩**逐纹素混合**：
  `p_texel = Σ mask_i · p_i / Σ mask_i`。
- 关键：纹理空间混合天然"归一"——不管多少层参与，一个纹素产出**一组** über 参数。层数不限不会增加运行时代价,也不需要人为封 N 层。
- 混合的是**同一套 über 参数**（base_color/metallic/roughness/…）的 lerp，**无损、线性、可预期**：和"把 clearcoat 近似并进 base"那种异构瓣塌缩完全不同——这里全程同一参数化，插值就是插值，不发明新模型。
- 瓣存在性（如某层有 coat、某层没有）通过遮罩加权后决定该纹素的 `lobe_mask`：有能量的瓣进 mask，驱动静态置换。

### 6.2 产出：开放 über 参数块
- 规范化产物是**一个变长 OpenPBR über 参数块**（扩展现 `SurfaceParameterBlock` 变长打包为 OpenPBR 全字段 lobe，§4.4）+ `lobe_mask` + VT 页引用。瓣集按材质实际声明，**不受固定 7 瓣上限约束**；OpenPBR 的 base/specular/coat/fuzz/subsurface/transmission/emission/thin_film 各组按需进 mask。
- 没有 `lod_ladder`、没有 `TierPlan`、没有 `SlabOp`、没有 `MAX_*` 封顶。
- `MAX_CLOSURE_SLAB_DEPTH` / `ClosureSlabTooDeep` / `MAX_MATERIAL_TEXTURES` 全删——运行时没有图、没有固定槽，无需这些定界常量。

```rust
// ir.rs —— 运行时无闭包图；烘焙产物是单一开放 über
pub struct BakedMaterial {
    pub surface: SurfaceParameterBlock, // 变长 über 参数块（瓣集按声明，无固定上限）
    pub lobe_mask: LobeMask,            // 开放位掩码，驱动静态特征置换（无损）
    pub features: MaterialFeatureFlags,
    pub render_class: MaterialRenderClass,
    pub illumination: Illumination,
    pub vt_pages: Vec<VtPageRef>,       // 取代固定 8 纹理槽，数量不限
    pub specialization: SpecializationId, // 含 lobe_mask，不含 tier、不含全局 MAX
}
```

### 6.3 作者节点图去哪了
MaterialX/OpenPBR 的程序化节点（noise/curve/blend）在烘焙期**烘到纹理**（VT 页）或**固化成常量参数**。运行时只采样纹理 + 跑静态特化的 über，不解释图。这正是 Frostbite/UE 的实际做法。

---

## 7. 降级：改输入不改模型（取代塌缩与 per-pixel 裁剪）

每材质跑它编译期特化的瓣集，瓣数/特征在距离上**恒定**。降级只作用在"输入"和"采样质量"上，全部无损或标准：

### 7.1 距离 LOD = 纹理预过滤（无损、自动）
- VT mip 链自动降细节；带宽随距离下降，着色瓣数不变。
- **高光抗锯齿**（Toksvig / LEAN）：把 mip 下采样丢失的法线细节转成等效 roughness 增量，远处高光不闪烁。
  `α_eff² = α² + 2·σ_normal²`（σ 由法线 mip 的方差估计）。
- 这替代了塌缩里"法线聚合"的角色，但不改模型、不改瓣数。

### 7.2 性能 LOD = 静态置换 + 采样缩放（无损 / 可控）
- **静态特征置换**：按 `lobe_mask` 编译期裁掉用不到的瓣（没有 clearcoat 的材质不编 clearcoat 分支）。这是"界"的真正所在——每材质编译成它的确切瓣集，无损,确定。
- **贵瓣采样缩放**：SSS/透射/多次散射的**采样数**按画质档缩放（如 SSS 卷积半径采样 16→8→4）。模型不变,只是数值精度降,平滑可控,不跳变。
- **per-tile 分类**（已有 9-way）：保证波前一致,降低分支代价。
- 注意：这里**没有** per-pixel 动态砍瓣——瓣集在编译期就定了，运行时不赌预算。

### 7.3 为什么这比塌缩 / per-pixel 裁剪都好（对比）

| 维度 | 塌缩阶梯（否决） | per-pixel 预算裁剪（否决） | 本方案（静态特化 + 改输入） |
|---|---|---|---|
| 外观一致性 | 异构瓣合并有损，掠射角走样 | 砍瓣阈值附近跳变 | 同一模型，始终物理一致 |
| 档位过渡 | 两个模型间跳变 | 像素间不连续 | mip 连续 + 采样数平滑，无跳变 |
| 可预期性 | 塌缩结果对美术是黑盒 | 逐像素预算不可控 | 远处=模糊版自己，直觉可预期 |
| 跨平台确定性 | 需能量守恒塌缩算子 | 弱平台预算抖动 | 每材质代价编译期常量 |
| 硬顶 | 需固定档位数 | 需 per-pixel 预算上限 | **无全局硬顶，界=编译期特化** |
| 实现复杂度 | 塌缩算子 + CPU/WESL 双实现 | 动态调度 + 噪点 | 复用 VT + 高光抗锯齿 + 采样缩放 |

---

## 8. 运行层 ABI v5（破坏性，不保留 v4，无硬顶）

### 8.1 SpecializationId（去掉 tier，纯正交轴 + 开放 lobe_mask）
```
bits  0..8   illumination
bits  8..40  closure_mask (= lobe_mask 投影，驱动静态置换；位宽留足，支持开放瓣集扩展)
bits 40..48  render_class
bits 48..64  reserved（不放 tier；降级不产生新 permutation）
```
降级不改 permutation：同一材质在近景/远景用**同一个** PSO，只是采样的 VT mip 和采样数不同。瓣集扩展时在 `closure_mask` 位域内扩展，不改轴布局。

### 8.2 GpuMaterialHeader v5
相对 v4 的破坏性改动：
- 保留单一 `parameter_offset` / `parameter_size`，但其长度**由瓣集决定、变长**（不是固定 7 瓣，也不是 tier 数组）。
- `lobe_mask` 保留并作为**开放位掩码**，驱动变长解包 + 静态置换。
- **参数块布局 = OpenPBR Surface 瓣序**：核心 base/specular/geometry 常驻，可选瓣（coat/fuzz/subsurface/transmission/emission/thin_film）按 mask 低位优先顺序变长排布；WESL `prism_unpack_surface` 已是低位优先 mask 驱动聚集，补瓣只是加长每瓣字段数（§4.4），解包循环不变。
- FACE（NPR 私有瓣）在 OpenPBR 物理瓣之后排布，由 `Illumination::Stylized` 消费（§3.5）；`custom_program` 字段承接自定义线（§3.5）。
- 纹理段 `texture_offset/texture_count` → **VT 页引用**（指向 `texture_streaming` 的页表句柄），不再是固定 8 槽索引，数量不限。
- 删除一切 `MAX_*` 常量语义（`MAX_MATERIAL_TEXTURES` / `MAX_CLOSURE_SLAB_DEPTH`）。
- `MATERIAL_ABI_VERSION = 5`。
- `material_classification.wesl` 的 `MaterialHeader` 镜像同步更新字段名（现仍是 v4 布局：`parameter_offset/parameter_size/lobe_mask`）。

### 8.3 per-tile 分类（复用现有，不加 tier 维度）
- `classify_count → prefix_classes → scatter_work`（`material_classification.wesl`）保持 9-way，不扩成 9×tier。
- 质量档（采样缩放）作为 pass 级 uniform,不进分类 key——避免桶数爆炸。
- 波前一致性由 9-way 分类保证；分支代价由静态置换（lobe_mask）压低。

### 8.4 置换治理（无界的代价：permutation 爆炸的缓解）
"每材质静态特化"的真实成本是 shader permutation 数量。治理手段：
- **置换 key = 瓣集（lobe_mask）+ 特征 flags**，内容哈希去重；相同瓣集的材质共享 PSO。
- **archetype 聚类**：模板把常见瓣集组合收敛成少量热门 permutation，长尾稀少。
- **按需异步编译 + über 回退**：冷 permutation 未编译完时，临时用一个"全瓣 über"PSO 兜底着色（略贵但正确），后台编译完成后切换。保证不卡顿、不丢画面。
- **置换缓存**：以内容哈希为键落盘，跨进程/跨机复用（CI 预热）。
- 这样"无界"落在作者/模型层，permutation 数量落在可治理的工程层——而不是用一个全局 MAX 去粗暴封顶。

### 8.5 运行层统一执行模型（一条执行路径，四线共用，光照来源可切换）

关键定调：**运行层只有一条执行路径**。四线（PBR/NPR/混合/自定义）的差异只落在"采样期响应函数"（步骤 5），缓存命中与否只改"光照来源"（步骤 4），降级只改"输入纹理 mip/采样数"——这三者**都不新增 PSO、不改分类 key、不产生并行着色栈**。这是把 §3.5（四线正交）、§8.3（分类）、§14（解耦着色）、§7（降级）收敛成单一执行契约的落点。

```
1 解包    prism_unpack_surface(words, lobe_mask) → Surface（变长 OpenPBR 瓣集，低位优先聚集，§4.4/§8.2）
2 分类    per-tile 9-way（按 illumination/render_class 分波前）；质量档/预算 = pass uniform，不进 key（§8.3）
3 瓣求值  热=spec-constant 置换（编译掉未用瓣，无损）⇔ 冷=全瓣 über + mask 瓣循环兜底（二者等价，§8.4）
4 光照源  (a) 对象空间缓存命中 → 读"传递无关入射辐照度"IncomingRadiance（§14.3）
          (b) 未命中/首触/视角相关 → 屏幕空间直接积分（§8 über 回退，非并行栈）
          两源归一为同一 IncomingRadiance 结构 → 下游响应函数无感（唯一分支点，不改 permutation/key）
5 响应    respond()：PBR 乘 albedo·评估 BRDF ｜ NPR 施 ramp(+fwidth 解析 AA) ｜ 混合按 illumination 遮罩 ｜ 自定义走契约（§3.5/§14.3）
6 合成    + 视角相关项（GGX/coat 高光、描边）→ 输出；主 pass TAA
```

- **一条路径的意义**：弱平台 / 关闭缓存 = 只走步骤 4(b) 的屏幕空间源，其余步骤完全不变——缓存只是把步骤 4 的来源从"逐帧屏幕积分"换成"对象空间复用"。没有第二套着色栈，不保留旧着色 API（破坏性）。
- **四线正交落在步骤 5**：步骤 1/2/3/4 四线完全共用，差异只在 `respond()` 内部的响应函数（§3.5）；缓存层对四线中立（§14.3），这是 PBR/NPR 精度能一致的结构根因。
- **permutation 边界只由步骤 1/3 决定**：key = 瓣集（lobe_mask）+ 特征 flags；步骤 2 的质量档、步骤 4 的缓存命中、步骤 6 的降级都**不参与 key** —— permutation 数量与"降级 / 缓存状态 / 画质档"彻底解耦（§8.3/§8.4）。
- **波前一致性**：步骤 2 的 9-way 分类把 Lit/Stylized/Custom 分到不同波前，步骤 3 的静态置换压低瓣分支，步骤 4 的两源归一避免"命中/未命中"在波前内分叉成本——三者叠加保证四线混排场景下的 occupancy。
- **响应模式开放化的热路径代价（§5.2）**：若把步骤 2 的 9-way 分类从定值升级为“注册驱动的响应模式集”（PCM response-mode kind），类数变成**烘焙期确定的固定表**（非运行时动态分支）——这是全管线扩展里唯一触碰热路径结构的点，须单独立项并验证不回归本节波前 occupancy。

---

## 9. 纹理：虚拟纹理（距离 LOD 的载体，复用现有地基，无槽上限）

现成基础设施（无需从零造）：
- `texture_streaming/indirection.rs`：GPU 页表，每项 `PAGE_TABLE_ENTRY_WORDS=4` 词，二分可查，CPU `lookup` 与 shader 二分同序。
- `texture_streaming/residency.rs`：`NotResident→Requested→Resident` 状态机 + 优先级 + 字节成本。
- `feedback.rs / scheduler.rs / pool.rs`：反馈优先级 + 预算调度 + 物理页池。

重构动作：
- 删除 `MAX_MATERIAL_TEXTURES = 8` 的语义约束，材质引用**任意多** VT 页，不占固定槽。
- **距离 LOD 完全交给 VT mip**：远→采样高 mip，页更小、带宽更低，着色模型不变（§7.1）。
- VT 缺页（`Requested` 未到 `Resident`）时采样回退到已驻留的更高 mip——降细节而非卡顿，仍是同一模型。

---

## 10. 效果：必补的物理项（进 CI 门禁）

- **多散射 GGX 能量补偿**（Turquin / Kulla-Conty）：修正单散射 GGX 高 roughness 丢能量。查表 `E(μ,α)`、`E_avg(α)`，补偿项
  `f_ms = (1-E(μ_o))(1-E(μ_i)) / (π(1-E_avg))`，乘以 `F_avg` 的多次反射级数。
- **Furnace test（白炉）CI 门禁**：均匀环境光下验证反照率守恒。CPU 金标准 + WESL twin 都跑。
- **高光抗锯齿**（Toksvig/LEAN，§7.1）：远处高光不闪烁；这是距离 LOD 的无损手段。
- **原理化分层 BSDF（coat/base 能量耦合，非瓣求和）**（§17.2）：coat 对其下 base 的能量不是简单叠加，而是按 Fresnel 透射做能量预算耦合（coat 反射掉的能量不再喂给 base），并统一 coat 吸收/变暗（`coat_darkening`）。取代"各瓣独立求和"的能量不自洽；进 furnace 门禁与双线对拍。
- **NPR**（`Illumination::Stylized`，§3.5）：ramp 量化 / SDF 面部阴影（Prism 私有 FACE 瓣，OpenPBR 之外）/ 描边，作为对**同一 OpenPBR über 输入**的正交非物理响应；不进物理 furnace 门禁，但进 ramp/描边视觉回归。
- **thin_film 虹彩（OpenPBR thin_film 瓣，开放瓣首个落地示例）**：薄膜干涉随 `thin_film_thickness` / `thin_film_ior` 产生彩虹色高光（肥皂泡、氧化金属、涂层）。按 OpenPBR 作为新 lobe 加入开放词汇表——新增 `LobeMask::THIN_FILM` 位 + 3-word 块，`prism_unpack_surface` 低位优先循环自动容纳，老材质 mask 不含该位、零影响。这是"新物理项 = 加法补瓣"路径的验证用例（§4.4）。
- **其余待补 OpenPBR 字段**（coat_color/coat_ior、fuzz_color、subsurface_radius vec3、specular_color/weight、diffuse_roughness）同样以加法扩字段方式补齐，各进 furnace / 对拍门禁。
- **开放瓣扩展机制**：任何新物理项（如各向异性 SSS）走同一路径——新位 + 变长字段 + 静态置换，不冲击现有材质、不加全局硬顶。

---

## 11. 双线确定性（保留，这是优势）

- CPU 金标准（`prism_render_shading`）+ WESL GPU twin 字节级对拍，重构中保留并扩展覆盖：
  - über BSDF、多散射补偿、高光抗锯齿、VT 页寻址（`indirection::lookup` 已 CPU/shader 同序）、烘焙期分层混合、变长解包，均 CPU/WESL 双实现对拍。
  - furnace / 能量守恒 / VT 寻址一致性作为回归门禁。
- 注意：这与旧文档"两套并行物理栈"无关——物理侧已收敛（`prism_render_architecture` 依赖 `prism_physics_core`，cloth 为薄 façade）。

---

## 12. 性能 / 效果 / 易用性权衡总表

| 维度 | 重构前 | 重构后 | 收益来源 |
|---|---|---|---|
| 模型线数 | 固定线 + 不固定线（二分） | **一条**开放 über | §6 烘焙单一化 |
| 全局硬顶 | `MAX_LOBE=7`/`MAX_TEX=8`/`MAX_DEPTH=4` | **无**；界=每材质编译期特化 | §6/§8 |
| 作者自由 | 内部 IR 手搓，深度≤4 拒绝 | OpenPBR/MaterialX，层数/瓣类不限 | §4 作者层 |
| 降级方式 | （无统一机制） | 纹理预过滤 + 高光抗锯齿 + 采样缩放（**无损,无塌缩,无 per-pixel 裁剪**） | §7 |
| 外观一致 | 塌缩/裁剪会走样（已否决） | 全程同模型，物理一致 | §7.3 |
| 性能确定 | über worst-case 全瓣 | 每材质静态特化 + per-tile 分类 | §7.2/§8.3 |
| permutation 治理 | — | 内容哈希去重 + archetype 聚类 + 按需编译 + über 回退 | §8.4 |
| 跨平台 | wgpu+WESL 已达成 | 维持；VT/补偿走 compute 可测 | §9/§11 |
| 高端画质 | 单散射丢能量 | 多散射补偿 + 开放瓣扩展 | §10 |
| 纹理规模 | 8 槽硬墙 | 虚拟纹理（复用现有，无槽上限） | §9 |
| 回归安全 | CPU golden + WESL twin | 维持并覆盖烘焙/VT/补偿/变长解包 | §11 |

---

## 13. 落地路线图（破坏性，分阶段）

| 阶段 | 内容 | ABI 影响 | 优先级 |
|---|---|---|---|
| P0-a | 多散射能量补偿 + furnace CI 门禁 | 无（改 BSDF+测试） | P0 |
| P0-b | 高光抗锯齿（Toksvig/LEAN）接入 über | 无 | P0 |
| P1-a | 距离 LOD 全面切到 VT mip（删 `MAX_MATERIAL_TEXTURES`） | 纹理段改 VT 页引用 | P1 |
| P1-b | ABI v5：header/分类 shader 字段同步（变长 über，无 tier/MAX） | **破坏性 v4→v5** | P1 |
| P1-c | 烘焙层：分层 → 纹理空间混合 → 开放 über（删闭包图/所有硬顶） | 内部，删 `MAX_CLOSURE_SLAB_DEPTH` | P1 |
| P1-d | 置换治理：内容哈希去重 + 按需编译 + über 回退 | 工具链/PSO 缓存 | P1 |
| P1-e | 贵瓣采样数按画质档缩放（SSS/透射/多散射） | 无（pass uniform） | P1 |
| P1-f | über 参数块规范化到 **OpenPBR Surface** 瓣序（补 base_weight/specular_color/coat_color/subsurface_radius 等字段，删 `lower_standard_material`） | 内部布局（随 v5） | P1 |
| P2-d | **thin_film 虹彩瓣**落地（开放瓣首验：新位 + 3-word 块 + furnace/对拍） | 内部 enum 开放化 | P2 |
| P2-e | 四线正交落地：`Illumination` 四值 + `custom_program` 钩子 + 混合线逐区域 illumination（per-tile 分类消化发散） | 运行时/分类 | P2 |
| P2-a | OpenPBR/MaterialX 作者前端 + archetype 模板（兼作置换聚类锚点） | 新 crate | P2 |
| P2-b | 作者节点图烘焙到 VT（noise/curve/blend 固化） | 工具链 | P2 |
| P2-c | 开放瓣词汇表扩展机制（新 lobe 作为一等成员接入静态置换） | 内部 enum 开放化 | P2 |
| P3-a | 解耦着色地基：纹理空间着色 pass，着色结果写入 VT 页缓存（复用 `texture_streaming`） | 新 pass/缓存，不改 ABI | P3 |
| P3-b | 先解耦视角无关项（diffuse/SSS/emission/GI 辐照度），高光仍走屏幕空间 | 分离求值 | P3 |
| P3-c | 缓存失效 + 时序复用 + mip LOD 策略（光照/动态变化时的页重算） | 调度/一致性 | P3 |
| P1-g | 自描述反射瓣 schema：lobe 定义（字段名/位宽/置换 key）数据化，解包/分类/对拍从 schema 生成（§17.5） | 工具链+内部布局 | P1 |
| P2-f | 超集合规门禁 + 向下投影导出器（纯 OpenPBR 对拍 + `project_to_openpbr`，§4.5/§17） | 新测试+导出路径 | P2 |
| P2-g | 运行时动态参数通道（wetness/snow/damage/dissolve 等少量动态轴，烘焙静态 über + 运行时解析式调制，§17.1） | header 加动态段 | P2 |
| P2-h | 原理化分层 BSDF（coat/base 能量耦合 + coat_darkening，§17.2） | 无（改 BSDF+测试） | P2 |
| P3-d | 光谱提升（hero-wavelength，仅 RT twin / 离线参考，不进实时主线，§17.3） | 无（RT-only 分支） | P3 |
| P3-e | 随机单瓣采样（blue-noise，贵瓣按预算随机选瓣 + TAA，§17.4） | 无（pass uniform） | P3 |
| 开放瓣 | glint / 测量 BRDF·BTF 瓣作为开放词汇表成员落地（§17.6） | 内部 enum 开放化 | P2+ |
| P1-h | **PCM 地基**：ABI v5 加 `extensions_used/required_mask`；先把 lobe/channel 两类迁进注册表，OpenPBR 核心瓣自举为 `KHR` 命名空间项（§5） | **随 v5**（header 加掩码） | P1 |
| P2-i | 全 kind manifest 格式（§17.5 schema 升级）+ codegen（解包/分类/对拍从 manifest 生成）+ MaterialX nodedef 往返 | 工具链+内部布局 | P2 |
| P2-j | **response-mode 分类注册化**（步骤 2 类数从定值改为注册驱动固定表，§5.2；唯一动热路径项，单独立项） | 分类 shader/运行时 | P2 |
| P2+ 扩展点 | transfer / cache-aux / provider / downgrade / composite 五类逐类接入 PCM；glint/测量/水彩描边作各 kind 首样例；thin_film 走完 `EXT_→KHR_` 晋升路径（§5） | 内部（烘焙期单态化） | P2+ |

### 13.a 实现进度（滚动更新 · code-truth）

> 下表只记录**已落地到代码并带测试门禁**的项；设计未动工的项一律按上面的路线图状态为「未开始」。实现以仓库代码为准，与本文叙述冲突时以代码为准。

**✅ 已完成**

- **P1-g（S1 + S2）自描述反射瓣 schema + codegen**（§17.5）
  - 提交：`399db87e2`（self-describing lobe schema + S1 对拍门禁）、`cd13ba525`（WESL unpack + Rust layout codegen，S2）。
  - 落地：`pkg/prism_material_schema`（schema crate）+ `prism-material-codegen` 二进制；由 schema 生成 `material_unpack.wesl` / `surface_layout.rs`；新鲜度（generated-vs-committed）+ CPU/WESL 对拍门禁。
  - 说明：这是"自描述 lobe schema + codegen 先行"路径的首个落地，解包/布局不再手写，后续加字段/加瓣走 schema 加法。
- **P0-a 多散射能量补偿 + furnace CI 门禁**（§10）
  - 范围：以 Kulla-Conty 2017 无色多散射瓣 + Turquin 2019 彩色多次反射 Fresnel（`F_avg² E_avg / (1 - F_avg(1-E_avg))`，逐通道）替换原先粗糙的 `energy = 1 - 0.28·rough²` 直接光补偿。ABI 中性（只改 BSDF + 测试）。
  - 代码：
    - CPU 金标准 `pkg/prism_render_shading/src/gi/env_brdf/multiscatter.rs` —— 新增解析直接光 API `multiscatter_direct(f0, n·v, n·l, perceptual_roughness)`、`directional_albedo_analytic`、`average_albedo_analytic`（8 节点 Gauss-Legendre，节点/权重与 WESL twin 逐位一致）。
    - 调用点 `pkg/prism_render_shading/src/lighting.rs::evaluate_principled_direct`。
    - WESL twin `pkg/prism_render_scene/src/shaders/brdf.wesl::multiscatter_direct`（与 CPU 同名、同算法、同常量）。
  - 门禁：白炉 furnace 测试（无色全反射 `单散射 E(μo) + 多散射瓣方向反照率 = 1`，各 roughness/视角 ≤2e-2）、近镜面消失、彩色金属保色/有限/非负、`E_avg` 有界且随 roughness 单调下降；WESL 侧经 `shading::resolve::shader_tests`（48 个 shader 编译/类型检查）覆盖。
  - 验证：`cargo test -p prism_render_shading --lib multiscatter`（21 通过）、`--lib lighting`（21 通过）、`cargo test -p prism_render_scene --lib shading::resolve::shader_tests`（48 通过）；touched 文件 clippy 干净、已 rustfmt。

- **P0-b 高光抗锯齿（Toksvig / Kaplanyan-Tokuyoshi / LEAN）接入 über**（§13 表 P0-b）
  - 范围：把已落地的高光抗锯齿核（Toksvig mip 法线长度、Kaplanyan/Tokuyoshi 屏幕空间法线方差、LEAN）从"独立核"接入 über 原理化直接光瓣。ABI 中性——法线方差是**着色期量**（GGX `alpha^2` 域的 `sigma^2`），由子像素/纹素足迹推导，**不进 GPU 材质 ABI**。
  - 设计：`SurfaceSample` 新增 `normal_variance: f32`（CPU 与 WESL twin 同步），直接光把基准 `alpha^2` 经 `filter_alpha_sq`/`spec_aa_filter_alpha_sq` 加上 `min(2·variance, kappa_max)` 的核方差再开方回 `alpha`。`normal_variance == 0` 为**逐位恒等**（钳位地板 `MIN_ALPHA^2` 远低于任何 `alpha >= MIN_ROUGHNESS^2`），既有 golden 全部保持不变；`kappa_max`（`DEFAULT_KAPPA_MAX = SPEC_AA_KAPPA_MAX = 0.18`）封顶，极大方差饱和到同一钳位 `alpha`。
  - 代码：
    - CPU 金标准 `pkg/prism_render_shading/src/lighting.rs`：`SurfaceSample.normal_variance` 字段 + `Default`；`evaluate_principled_direct` 用 `gi::specular_aa::normal_variance::filter_alpha_sq` 过滤 `alpha`。
    - 构造点同步：`resolve.rs::surface_sample_from_parameters`、`cloth_advanced.rs` 测试基面（其余均走 `..Default::default()`）。
    - WESL twin `pkg/prism_render_scene/src/shaders/brdf.wesl`：`SurfaceSample.normal_variance` + `principled_direct` 调 `prism_render_scene::shaders::specular_aa::spec_aa_filter_alpha_sq`（与 CPU 逐位同算术、同常量、同参序）。
    - 核本体（本阶段前已落地、本阶段接入）：CPU `gi/specular_aa/{mod,toksvig,normal_variance,lean}.rs` + WESL `shaders/specular_aa.wesl` + 对拍 `shading/specular_aa/{mod,shader_tests}.rs`。
  - 诚实边界：`normal_variance` 当前恒为 `0.0`（恒等旁路），**待 P1-a 纹理足迹 / P1-f 烘焙法线方差图接入后被真实喂数**才激活抗锯齿；这是分阶段诚实接线，不是假实现。编辑器 MaterialX 作者前端（P2-a）仍按 goal 保持 TODO。
  - 门禁：`cargo test -p prism_render_shading --lib lighting`（22 通过，含 `specular_aa_variance_is_identity_at_zero_and_coarsens_the_peak`：恒等 + 单调粗化 + κ 饱和）、`--lib gi::specular_aa`；`cargo test -p prism_render_scene --lib shading::resolve::shader_tests`（48 通过，编译 brdf.wesl + specular_aa.wesl 全链路）、`--lib shading::specular_aa`；touched 文件 clippy 干净、已 rustfmt。

**⬜ 未开始（见上表）**：P1-a~P1-f、P1-h（PCM 地基）、P2-a~P2-j、P3-a~P3-e、开放瓣（glint / 测量 BRDF）。

**破坏性说明**：P1-b 一次性弃 v4，不做兼容垫片；旧 `MaterialRecord`/`GpuMaterialHeader`/`lower_standard_material`/所有 `MAX_*` 常量直接改写或删除。

### 13.5 P3 远期：解耦着色 / texel-space shading（性能上限最高，工程最重）

不在屏幕空间逐像素着色，而在**对象/纹理空间**着色，结果写入着色缓存页、跨像素/跨帧复用；按 mip 摊销 → **天然 LOD + 消除 overshading**。性能天花板最高但工程最重，列为 **P3 远期选项**，严格作为**独立性能子系统**推进，不与 über 重构（P0–P2）耦合——über 先落地，解耦着色作为可选 pass 叠加。**完整深化设计（架构 / 四线分离求值 / 着色率 / 失效 / seam / 一致性 / 易用 / 门禁 / 风险解法）见 §14。**

---

## 14. 解耦着色（texel-space / object-space shading）深化设计

> 定位：**P3 性能上限层**。把"着色"从屏幕空间逐像素解绑，改为在**对象/纹理空间**按需着色、写入**着色缓存**，跨像素跨帧复用，按 mip 摊销。目标是在不改 OpenPBR über 模型（§4/§8）、不改四线正交语义（§3.5）的前提下，拿到"消除 overshading + 天然 LOD + 可摊销光照"的性能天花板。全程**复用现有 `texture_streaming` 地基**，不另造 VT；不保留任何旧着色路径对外 API（屏幕空间着色退化为"缓存未命中回退路径"，不是并行栈）。

### 14.1 对标与借鉴（取其形，去其短）

| 项目 / 技术 | 借鉴点 | Prism 取舍 |
|---|---|---|
| **育碧 Deferred/Texel Shading**（Far Cry 系） | 着色与光栅解耦，纹理空间着色由 VT 承载，feedback 驱动"要着哪些 texel" | **核心借鉴**：复用 VT feedback→priority→budget 链(`feedback.rs`/`scheduler.rs`)驱动"着色需求"，不是只驱动"采样需求" |
| **UE5 Lumen Surface Cache** | 对象空间低频光照缓存 + 预算重着色 + 时序复用 | 借其"按预算分帧重着 + 时序累积"的失效摊销；但我们缓存**材质响应**而非仅间接光，分辨率随 mip 自适应 |
| **Decoupled Deferred Shading（Liktor/Dachsbacher 2012）** | 着色样本 memoization 缓存，着色率与可见性率解耦 | 借其"着色样本去重/复用"思想，落到 VT 页粒度（而非 micropolygon） |
| **RenderMan Reyes / Pixar Ptex** | 纹理空间着色 + **per-face 无缝参数化**（Ptex 无显式 UV seam） | 借 Ptex 式**按几何面片分配缓存 tile**，从根上回避 UV seam（§14.6） |
| **id Tech MegaTexture / 现 `texture_streaming`** | 虚拟纹理页表 + residency + 预算调度 | **直接复用**：着色缓存 = "可写 VT 页"，页表/残留/调度/图集全部沿用 |
| **COD/VRS、Nanite 材质** | 屏幕空间可见性率与着色率解耦（VRS）、per-tile 分类 | 作为**回退路径**与补充：缓存未命中/视角相关项走屏幕空间 + per-tile 分类(§8.3) |

一句话：**用 VT 的"按需、分级、预算"机制去调度"着色"而不只是"采样"**；对象空间缓存复用 Lumen 的分帧重着 + 时序摊销；用 Ptex 式 per-face tile 回避 seam。

### 14.2 架构：着色缓存 = 可写 VT 页（复用地基，加法）

数据流（在现 `texture_streaming` 上加一个"着色"生产者，消费端不变）：

```
几何/可见性(vis-buffer) ──► texel feedback: 哪些对象空间页本帧可见 + 期望 mip
        │                         (复用 feedback.rs 的 PageDemand / mip_error / importance)
        ▼
着色需求表(ShadeResidencyTable)  ──► 预算调度(复用 scheduler.rs 贪心按字节/按 texel 预算)
        │  标记 Requested/Resident/Stale            │ 产出 ShadePlan{要着色的页, 要驱逐的页}
        ▼                                           ▼
对象空间着色 pass(compute) ──写入──► 着色缓存图集(复用 atlas.rs 的 tile 布局/slot_placement)
        │  对每个 texel: 解包 OpenPBR über(§8) → 按四线响应(§3.5)算"可缓存项"
        ▼
主 pass 采样着色缓存(复用 indirection.rs 二分页表) + 屏幕空间补"视角相关项" → 合成
```

复用点（全部已存在，详见 §9）：
- **页表** `indirection.rs::GpuPageTable`：二分可查；其 `w1` 低 8 位现为保留位 → 存**着色缓存 generation / valid 标志**（§14.7），零额外带宽。
- **残留状态机** `residency.rs::PageResidency`（`NotResident→Requested→Resident`）→ 扩一个 **`Stale`** 语义（已驻留但光照过期需重着），走同一优先级容器。
- **预算调度** `scheduler.rs::StreamingPlan` 贪心按字节预算 → 复用为**着色预算**（§14.4）。
- **图集** `atlas.rs::AtlasGeometry/slot_placement`：着色结果写进 tile；`COPY_BYTES_PER_ROW_ALIGNMENT`/block 对齐沿用；tile 边缘留 gutter（§14.6）。
- **保底 mip** `mip_tail.rs::mip_tail_covers`：保证粗 mip 常驻 → 缓存未命中时的降细节回退（§14.7）。
- **时序/驱逐保护** `streamer.rs` 的 retention decay + eviction protection window → 直接用于**缓存失效摊销**（§14.5）。

关键：**着色缓存不是新子系统的新 ABI，而是给 VT 页加一个"可写 + 带 generation"的生命周期**。über 解包与四线响应核心(§3.5/§8)原样搬进对象空间 compute pass，**着色数学不变**。

### 14.3 四线 × 分离求值矩阵（视角无关项缓存，视角相关项屏幕空间）

解耦的物理前提：**视角无关项**（不依赖 view/half-vector）可缓存复用；**视角相关项**依赖观察方向，缓存需存方向性表示，成本/误差高。

**架构约束（本次重构定调，取代上一版的 `banding_sensitivity` 按页特判）**：缓存层只存**传递无关的入射辐射量**——入射辐照度 / 低频辐射（需要高光时加方向性 SH-L1/L2），**不存"已经过响应/传递的结果"**。所有响应与传递都搬到**采样期**：PBR 在采样期全分辨率乘 albedo·评估 BRDF，NPR 在采样期对缓存辐照度施加 ramp，自定义走契约。推论：

- **缓存对四线完全中立**：同一表面的缓存内容 PBR / NPR / 混合区域**共用同一份辐照度**，不按线分通道存不同"响应"（§14.3 旧表的分线只是"采样期各取所需"，不是缓存内容分叉）。
- **精度是单一规范编码的全局决定**，不存在按线 / 按材质的精度策略——`banding_sensitivity` 这类标志被删除（见 §14.8）。
- **albedo/材质细节不被摊销**：只有昂贵的光照被缓存按 mip 摊销，albedo/法线等高频细节仍在采样期全分辨率取——解耦着色的本意（省光照、不省细节）。

四条线（§3.5）在采样期各自的切分：

| 线 | 可缓存（对象空间，复用） | 屏幕空间（视角相关，不缓存或方向性缓存） | 高效/高质解法 |
|---|---|---|---|
| **PBR** | 入射辐照度 / GI、SSS 漫透低频项、emission（**albedo/base 不进缓存，采样期全分辨率乘入**） | GGX specular / coat 高光 / anisotropy（依赖 H 向量） | 缓存辐照度直接复用，采样期乘 albedo（反照率细节不糊）；高光项用**缓存辐照度 + 屏幕空间 BRDF 评估**（split-sum 式）；需缓存高光时存**方向性辐照度（SH-L1/L2 或主方向 lobe）** |
| **NPR** | toon ramp 的**光照量化输入**（N·L 带状化前的辐照度）、面阴影 SDF 采样、材质色 | 描边（屏幕空间几何）、风格化高光锐边（视角相关） | ramp 的"输入辐照度"视角无关 → 缓存；**量化曲线在采样期施加**（曲线是逐像素便宜的 LUT 查表，缓存连续量更稳、避免档边闪烁）；描边永远屏幕空间 |
| **混合** | 逐区域取 PBR/NPR 的可缓存项，按 illumination 遮罩 | 两线各自的视角相关项 | 缓存层按 `illumination` 分通道存；合成期按遮罩选择；**per-tile 分类(§8.3)把 Lit/Stylized 分波前**，缓存命中与否不破坏一致性 |
| **自定义** | 由自定义程序**声明**的"视角无关输出"（契约，§14.9） | 声明为视角相关的输出 | 自定义 WESL 片段实现采样期 `respond_custom()`（必）+ 可选 `shade_cacheable_aux()`（声明额外视角无关量）；默认只用共享入射辐照度，未实现则安全退化为屏幕空间 |

统一原则——**split evaluation + 传递无关缓存**：对象空间只缓存"传递无关的入射辐射量" + 时序复用；响应（albedo·BRDF / ramp / 自定义）与视角相关项全在采样期算；合成阶段相加。四线共享**同一份缓存内容**与同一 über 解包，差异只在采样期的响应函数——缓存层对四线中立，这是 PBR/NPR 精度能完全一致的根（§14.8）。

### 14.4 着色率管理（风险①的高性能解法）

问题：纹理空间若盲目全量重着，比屏幕空间更贵。解法 = **把 VT 的"按需 + 分级 + 预算"直接当着色率控制器**：

- **按需**：只着色 vis-buffer 反馈为**本帧可见**的对象空间页（复用 `feedback.rs` 的可见性→`PageDemand`）。不可见页不着色。
- **分级（天然 LOD）**：期望 mip 由屏幕投影面积定（`feedback.rs::mip_error`/`clamped_importance`）。远处/密集几何落到粗 mip → **一个粗 texel 覆盖多像素，overshading 被消除**（这是性能天花板的来源）。
- **预算（硬上限）**：每帧着色 texel 数由 `scheduler.rs` 贪心按**着色预算**（texel/字节/compute 时间）切；超预算的页**本帧不重着，继续采样上一帧缓存**（时序复用兜底）。预算是 pass 级 uniform，不进材质 permutation（与 §8.3 一致）。
- **优先级**：复用 `feedback.rs::SemanticWeights` 思路，给"着色需求"打分（屏幕重要度 × mip 紧迫度 × 失效紧迫度），高分先着。
- **VRS 协同**：缓存命中区域主 pass 几乎只做"采样 + 视角相关补项"，可叠加硬件 VRS 进一步降屏幕着色率。

效果：**着色总量 ≈ O(可见对象空间 texel at mip)**，与屏幕分辨率/过绘制解耦；密集几何、远景、高几何复杂场景收益最大。

### 14.5 缓存失效与时序摊销（风险②的高性能解法）

问题：光照/材质/动态变化时缓存过期；全量重着=退回逐帧全着。解法 = **细粒度失效 + 分帧摊销 + 时序累积**：

- **generation / epoch 失效**：每着色缓存页存一个 `shade_generation`（放 `indirection.rs` 页表 `w1` 保留低 8 位 + 页元数据）。失效源各自维护 epoch：
  - 光照 epoch（光源移动/强度变、天光变）；动态材质 epoch（材质参数动画）；几何 epoch（蒙皮/形变）。
  - 页的 `shade_generation` < 影响它的最大 epoch → 标 **`Stale`**（复用残留状态机新增态）。
- **局部失效，不全局**：只有**受影响的页**进 `Stale`（如只有被移动光源包围盒覆盖的对象空间页），其余命中不动。静态光 + 静态几何的页**永不重着**（最大收益场景）。
- **分帧重着（Lumen 式预算摊销）**：`Stale` 页进同一预算调度(§14.4)，按优先级**每帧只重着一部分**；重着前继续用旧缓存（短暂偏旧，视觉可接受）。重着紧迫度随 staleness 上升。
- **时序累积**：对象空间着色天然稳定（无屏幕空间抖动），重着结果与旧值做 **temporal blend**，高频光照变化下用历史平滑；配合主 pass TAA。
- **保守重着频率分层**：diffuse/GI 低频 → 低频重着；emission/快速动画 → 高频或直接走屏幕空间。频率是**每语义/每 archetype 可配**（§14.9）。

效果：**重着成本 ∝ 实际变化量**，而非场景规模；静态区零成本，动态区被预算 + 时序摊平，无档位跳变。

### 14.6 UV seam / 参数化（风险③的高质解法）

问题：纹理空间着色在 UV 接缝处出现裂缝/漏光。解法 = **Ptex 式 per-face tile + gutter，从参数化根上消除 seam**：

- **per-face / per-chart 缓存 tile**：着色缓存按**几何面片（或 chart）**分配 tile，而非共享全局 UV。借 Pixar **Ptex**：无显式全局 UV、无跨 chart seam；相邻面片的滤波通过**邻接表**在采样期跨 tile 取邻。
- **gutter（边缘外扩）**：每 tile 着色时多算 1–2 圈边缘 texel（`atlas.rs` tile 已有 block 对齐，预留 gutter 带），双线性/各向异性采样不越界、不漏缝。
- **mip 一致的 seam 处理**：粗 mip 的 gutter 同步生成，避免远处接缝重现。
- **退路**：对不便 per-face 参数化的资产（如导入的单 UV 网格），用**接缝感知滤波**（采样期检测 chart 边界，钳到同 chart）作为次优解。

效果：接缝在**生成期**解决（gutter + 邻接），采样期零特判或仅轻量钳制，视觉无缝。

### 14.7 缓存一致性与回退（风险④的高性能解法）

问题：页在"请求着色→着色完成→被驱逐"过程中，主 pass 可能采到未就绪/已失效页。解法 = **就绪位 + 粗 mip 保底 + 无锁双代**：

- **就绪/有效位**：页表 `w1` 保留位存 `valid` + `shade_generation`；主 pass 采样时若页 `!valid`，**回退到已驻留的更粗 mip**（`mip_tail.rs::mip_tail_covers` 保证粗 mip 常驻）→ 降细节而非裂帧，与 §7.1 距离 LOD 同手段。
- **着色与采样无锁解耦**：着色 pass 写入**新 slot**，完成后原子更新页表指向新 slot（generation 自增），主 pass 永远读到**一致的某一代**（旧代或新代，不读半写）。驱逐沿用 `scheduler.rs` 的 evict 顺序。
- **首触延迟（disocclusion）**：相机骤转/新物体入场导致大量页未着色 → 本帧这些像素**走屏幕空间 über 回退**（即普通 §8 路径），后台补着色，下帧起命中缓存。保证不卡顿、不黑块。
- **一致性校验**：对象空间缓存值可与屏幕空间直算做**抽样对拍**（复用 §11 双线框架的思路，CPU 金标准可在对象空间复算），作为 CI 回归。

效果：主 pass 永远有可用数据（新代/旧代/粗 mip/屏幕回退四级兜底），**无裂缝、无卡顿、无非确定黑块**。

### 14.8 内存与压缩

- **单一规范编码（四线共用一份，天然无带状）**：缓存只存传递无关的入射辐照度（§14.3），用一种**感知均匀的 HDR 编码**（log-luminance / PQ 式曲线，色度分量线性）——在整个 HDR 范围给出**均匀相对精度**。因为编码与下游响应无关、且感知均匀，任何采样期传递（PBR tonemap、NPR ramp、自定义）都**不会放大带状**：量化步长感知上恒定，无"陡区被压出台阶"的问题。于是**不再需要按线/按页的精度策略**（删除上一版的 `banding_sensitivity` 与 NPR 排除分支）——精度是这一个编码选择的全局结果，PBR/NPR 从源头一致。
- **有损压缩对四线一致（只看重着频率，不看线）**：GPU BC6H 作用在上述感知编码上，误差预算对所有消费者一致；是否压**只由页的重着频率/价值**决定（低频久驻页压、高频重着页不压），**与 PBR/NPR 无关**。因为缓存存的是感知均匀辐照度而非"响应结果"，BC6H 的误差同样不被 ramp 放大。
- **ramp 台阶的锐度不依赖缓存位深（架构保证）**：NPR 量化的"硬边"由 ramp LUT 的阈值定义，并在**采样期用解码辐照度的屏幕空间梯度做解析抗锯齿**（阈值交越处按 `fwidth` 平滑），因此台阶边缘清晰度由屏幕导数决定、**与缓存 bit 深度解耦**。这把"NPR 要高精度缓存"的需求从根上移除——缓存只要感知均匀，锐边在采样期生成。
- **预算即显存墙**：图集容量 = `pool.rs::PhysicalPagePool` capacity，按平台配置；弱平台调小预算 → 更多走屏幕空间回退（优雅降级，非崩溃）。
- **方向性缓存的代价**：若对高光启用 SH-L1 方向性缓存，字节成本 ×4（4 系数）→ 仅对高价值材质/archetype 开启（§14.9）。

### 14.9 易用性与作者契约（默认安全，opt-in 加速）

- **默认屏幕空间，archetype opt-in 解耦**：解耦着色是**性能优化开关**，不改作者心智模型。美术仍写 OpenPBR Surface(§4)；是否走对象空间缓存由 **archetype/材质标志**声明（如 `Skin/Foliage/静态建筑` 开，`快速动画/强视角相关` 关）。
- **自定义线的采样期响应契约**（§3.5 自定义线落点 / §16 附录A）：缓存只存传递无关入射辐照度，自定义程序只负责响应
  - `respond_custom(surface, inc: IncomingRadiance, aux, view) -> Color`（采样期施加响应，读共享入射辐照度，必实现）
  - `shade_cacheable_aux(surface, light_env) -> CustomAux`（**可选**：声明额外视角无关量进缓存，分通道不污染共享辐照度）
  - 不实现 `respond_custom` → 安全退化为屏幕空间 über（§8）；不实现 `shade_cacheable_aux` → 只用共享入射辐照度。
- **可观测性**：调试视图叠加"缓存命中率 / Stale 页 / 重着预算占用 / mip 分布"，美术/工程按场景调预算与重着频率。
- **零作者 seam 负担**：per-face tile + gutter(§14.6) 由系统处理，美术不手动排 UV gutter。

### 14.10 落地阶段与门禁（细化 §13 的 P3-a/b/c）

| 阶段 | 内容 | 门禁 |
|---|---|---|
| P3-a 地基 | 着色缓存 = 可写 VT 页：扩 `residency` 加 `Stale`、页表 `w1` 存 generation/valid、着色 compute pass 写 `atlas` tile | 命中/未命中回退正确；无裂帧（粗 mip 兜底）CI |
| P3-b 视角无关项 | 先缓存 diffuse/SSS/emission/GI 辐照度（PBR+NPR 可缓存项），高光走屏幕空间 split-sum | 与纯屏幕空间抽样对拍误差阈值；furnace 不回归 |
| P3-c 失效摊销 | generation/epoch 局部失效 + 分帧预算重着 + 时序累积 | 动态光场景帧时稳定、无档跳；staleness 上限 CI |
| P3-d seam/一致性 | per-face tile + gutter + 无锁双代 + disocclusion 屏幕回退 | seam 视觉零裂；骤转无黑块 CI |
| P3-e 四线/自定义契约 | NPR ramp 输入缓存、混合逐区域、自定义两段式契约 | 四线各自对拍 + 命中率/预算可观测 |
| P3-f 内存/压缩（可选） | BC6H 低频页压缩、方向性 SH 高光缓存（高价值材质） | 显存预算内；方向性误差阈值 |

**破坏性说明**：屏幕空间 über 路径（§8）**保留为缓存未命中/视角相关的回退路径**，不是并行栈，不维护旧着色 API；对象空间缓存是其上的加法 pass。弱平台关闭开关即纯 §8 路径。

### 14.11 风险 → 高性能/高质解决方案总表

| 风险 | 朴素做法的坑 | 本设计的解法 | 复用地基 |
|---|---|---|---|
| 纹理空间着色率失控 | 全量重着比屏幕还贵 | 按需(可见)+分级(mip 消 overshading)+预算(硬上限)+VRS 协同 | `feedback.rs`/`scheduler.rs` |
| 缓存失效（光照/动态变化） | 全量重着退回逐帧全着 | generation/epoch 局部失效 + 分帧预算摊销 + 时序累积 + 频率分层 | `streamer.rs` decay/保护窗 |
| UV seam 裂缝/漏光 | 采样期特判昂贵且不彻底 | Ptex 式 per-face tile + gutter 生成期消缝 + 邻接采样 | `atlas.rs` tile/对齐 |
| 缓存一致性（半写/未就绪） | 裂帧/黑块/非确定 | valid 位 + 无锁双代 generation + 粗 mip 保底 + disocclusion 屏幕回退 | `indirection.rs` w1/`mip_tail.rs` |
| 视角相关项不可缓存 | 强缓存高光→拖影/错误 | split evaluation：视角无关缓存 + 视角相关屏幕空间；需要时方向性 SH 缓存 | §3.5 四线 + §8 über |
| 显存压力 | 图集爆显存 | HDR 紧凑格式 + 低频页 BC6H + 预算即墙 + 弱平台回退屏幕空间 | `pool.rs` capacity |
| 作者复杂度上升 | 美术要懂缓存/UV gutter | 默认屏幕空间、archetype opt-in、自定义两段式契约、seam 系统托管 | §4 archetype/§3.5 |

效果小结：**性能**上消除 overshading + 按 mip/变化量摊销，密集/远景/静态光场景拿到天花板；**效果**上 split evaluation + 方向性缓存保高光正确、Ptex gutter 消缝、时序累积抗抖；**易用**上默认安全、opt-in 加速、seam 托管。整条路**复用 `texture_streaming` 全部地基**，是加法 pass，不引入并行着色栈，不保留旧着色 API。

---

## 15. 风险与回退

| 风险 | 缓解 |
|---|---|
| **permutation 爆炸**（无全局硬顶的主要代价） | 内容哈希去重 + archetype 聚类 + 按需异步编译 + 全瓣 über 回退 PSO 兜底（§8.4）；CI 预热缓存 |
| 冷 permutation 首帧回退略贵 | über 回退正确但稍慢，后台编译完切换；可对热门 archetype 预编译 |
| 多散射补偿与 WESL twin 对不齐 | furnace CI + 查表字节对拍 |
| VT 缺页抖动 | 回退到已驻留高 mip（降细节不卡顿）；复用现有 residency 优先级 |
| 采样缩放在低档出现噪点 | 配合 TAA/时序累积；缩放曲线可调 |
| 变长 über 解包边界错误 | CPU/WESL twin 字节级对拍覆盖变长路径 |
| 破坏性 ABI 迁移面大 | 一次性迁移 + WESL twin 对拍兜底,分阶段 P0→P2 |
| 解耦着色缓存失效/一致性（P3） | 作为独立子系统推进；首期只缓存视角无关项，高光仍屏幕空间；光照/动态变化触发页重算 + 时序复用兜底 |
| 运行时动态通道破坏确定性/双线对拍（§17.1） | 动态轴限定为少量解析式调制（无分支爆炸）；调制函数 CPU/WESL 双实现字节对拍；动态段进 ABI 显式声明，不走隐式状态 |
| 原理化分层 BSDF 与 twin 对不齐（§17.2） | 能量耦合项查表化 + furnace 守恒门禁；coat/base 耦合系数 CPU 金标准先行、WESL twin 对拍 |
| 光谱提升实时成本过高（§17.3） | 严格限定 RT twin / 离线参考，不进实时主线；实时仍 RGB，光谱仅作对拍基准与色散高保真分支 |
| 随机单瓣采样低档噪点（§17.4） | blue-noise + TAA/时序累积；采样档可调；仅对贵瓣启用，廉价瓣仍确定性求和 |
| OpenPBR 上游演进与扩展位撞车（§4.5） | 规范核心区/扩展区位段隔离；上游新字段只进核心区并过合规门禁；私有瓣永不占核心区位段 |
| PCM response-mode 分类注册化动热路径（§5.2） | 类数仅烘焙期注册、运行时查固定表（非动态分支）；单独立项，用 §11 对拍 + occupancy 回归守住波前一致性（§8.5） |
| PCM 九类 hook ABI 面大、前期设计重（§5） | 分阶段：P1 先 lobe/channel，其余八类 P2+ 逐类接入；每类 hook 签名类型化，受限 Rust 瓣由 `proc-macro` 产 WESL、金标准即原生 Rust，自动双线对拍 |
| 扩展能力在弱平台不满足（§5.3） | 能力协商在烘焙/加载期完成：required 不满足显式报错、used 不满足 graceful ignore 投影回变体；特化结果烘焙期可枚举可 CI |

---

## 16. 附录 A：关键数据结构草案

```rust
// ir.rs —— 删除 MAX_CLOSURE_SLAB_DEPTH / ClosureSlabTooDeep / slab_depth / MAX_MATERIAL_TEXTURES
// 运行时无闭包图；烘焙产物是单一开放 über（变长，瓣集按声明）
pub struct BakedMaterial {
    pub surface: SurfaceParameterBlock,  // 变长 OpenPBR Surface über 块（§4.4 全字段，无 7 瓣上限）
    pub lobe_mask: LobeMask,             // 开放位掩码，驱动静态置换与变长解包
    pub features: MaterialFeatureFlags,
    pub render_class: MaterialRenderClass,
    pub illumination: Illumination,
    pub vt_pages: Vec<VtPageRef>,        // 取代固定 8 纹理槽，数量不限
    pub specialization: SpecializationId,// illumination/closure_mask/render_class（无 tier/MAX）
}

// 烘焙期分层混合（同参数化 lerp，无损；离线、可预览；任意层数）
pub fn bake_layer_stack(layers: &[LayerParams], masks: &[MaskSource])
    -> (SurfaceParameterBlock, LobeMask);
```

```rust
// record.rs —— ABI v5（破坏性，单一开放 über，无 tier、无 MAX 常量）
pub const MATERIAL_ABI_VERSION: u32 = 5;

#[repr(C)]
pub struct GpuMaterialHeader {
    pub generation: u32, pub revision: u32,
    pub illumination: u32, pub render_class: u32,
    pub feature_flags: u32, pub closure_mask: u32,
    pub parameter_offset: u32, pub parameter_size: u32, // 单一变长 über 块（长度随瓣集）
    pub vt_page_offset: u32, pub vt_page_count: u32,     // 取代 texture_offset/count，数量不限
    pub sampler_offset: u32, pub sampler_count: u32,
    pub custom_program: u32, pub active: u32,
    pub material_epoch_low: u32, pub material_epoch_high: u32,
    pub closure_graph_offset: u32,                       // 仅 RT/离线消费，可为 0
    pub specialization_low: u32, pub specialization_high: u32,
    pub lobe_mask: u32,                                  // 开放位掩码
}
```

```rust
// === 解耦着色数据结构（§14 落点，P3；全部复用 texture_streaming 地基，加法）===

// residency.rs —— 残留状态机新增 Stale 态（已驻留但光照/材质/几何过期需重着）
// 原: NotResident -> Requested -> Resident
pub enum PageResidency {
    NotResident,
    Requested,
    Resident,
    Stale,          // 新增：缓存命中仍可采样，但进重着队列（§14.5）
}

// indirection.rs —— 页表项 w1 的保留低 8 位用作着色缓存元数据（零额外带宽，§14.2/14.7）
// w1 = (mip << 24) | (layer << 8) | shade_meta8
//   shade_meta8: bit0   = valid（本页着色结果已就绪，可采样）
//                bit1   = generation_parity（无锁双代，避免半写撕裂）
//                bit2..7= shade_generation 低位（粗判 Stale；细判在页元数据）
pub const SHADE_VALID_BIT: u32 = 1 << 0;
pub const SHADE_GEN_PARITY_BIT: u32 = 1 << 1;

// 着色缓存页元数据（与 VT 页一一对应，不新建 ABI，仅给页加生命周期）
pub struct ShadePageMeta {
    pub shade_generation: u32,   // 本页着色时各失效源 epoch 的快照（§14.5）
    pub last_shaded_frame: u32,  // 时序摊销/驱逐保护窗（复用 streamer.rs decay）
    pub residency: PageResidency,
}

// 失效源 epoch：页的 shade_generation < 影响它的最大 epoch -> 标 Stale（局部，不全局）
pub struct ShadeEpochs {
    pub lighting_epoch: u32,     // 光源移动/强度/天光变
    pub material_epoch: u32,     // 材质参数动画
    pub geometry_epoch: u32,     // 蒙皮/形变
}
```

```rust
// scheduler.rs —— 着色需求表与着色计划（复用贪心按预算调度，预算是 pass 级 uniform，不进 permutation）
pub struct ShadeResidencyTable {
    pub demands: Vec<ShadeDemand>,   // 本帧可见对象空间页的着色需求（来自 vis-buffer feedback）
}

pub struct ShadeDemand {
    pub page: VtPageRef,
    pub desired_mip: u8,             // 由屏幕投影面积定（feedback.rs::mip_error）—— 天然 LOD/消 overshading
    pub priority: u32,              // 屏幕重要度 × mip 紧迫度 × staleness（复用 SemanticWeights 思路）
    pub stale: bool,
}

// 复用 StreamingPlan 的结构语义：本帧要着色的页 + 要驱逐的页，受着色预算（texel/字节/compute 时间）硬切
pub struct ShadePlan {
    pub to_shade: Vec<VtPageRef>,    // 超预算的页本帧不重着，继续采样上一帧缓存（时序复用兜底，§14.4）
    pub to_evict: Vec<VtPageRef>,
    pub shade_budget_texels: u32,    // pass 级 uniform
}
```

```rust
// === 传递无关着色缓存：规范编码（v9/v10 定调，§14.3/§14.8）===
// 缓存只存"视角无关的入射辐射量"，不存任何已响应/已传递的结果。
// 四线（PBR/NPR/混合/自定义）共用同一份内容；精度是这一个编码的全局结果，无按线/按页策略。

// 单一规范编码：感知均匀 HDR（log-luminance / PQ 式亮度曲线，色度分量线性）。
// 在整个 HDR 范围给出均匀相对精度 → 任何采样期传递都不放大带状（§14.8）。
pub enum ShadeCacheEncoding {
    // 基础（必存，四线中立）：聚合入射辐照度（漫反射/GI/SSS 低频/emission）
    IrradiancePerceptualHdr,            // R=感知编码亮度, G/B=色度(线性)
    // 可选：方向性入射辐射，支持采样期高光 BRDF 评估（高价值材质才开，§14.9）
    DirectionalSh { bands: u8 },        // 1 或 2 阶 SH；字节 ×(系数数)
}

// 着色缓存页负载：共享入射辐照度常驻；方向性 / 自定义 aux 分通道，不混进共享内容
pub struct ShadeCachePagePayload {
    pub incoming: ShadeCacheEncoding,   // 四线中立，必存（唯一被按 mip 摊销的量）
    pub directional_words: u32,         // 可选高光方向性 SH 的字长，0 = 不存
    pub custom_aux_offset: u32,         // 自定义线声明的额外视角无关量（可选，分通道，不污染共享）
    pub custom_aux_words: u32,
    pub shade_generation: u32,          // 失效源 epoch 快照（§14.5），粗判 Stale
    // 注意：albedo/法线等高频细节【不在此】—— 采样期全分辨率取（省光照，不省细节，§14.3）
}
```

```wgsl
// === 采样期响应：四线统一入口（v9/v10，取代旧 CachedResponse 两段式契约）===
// 缓存读出的是"传递无关的入射辐射量"，不是任何线的"响应结果"。
// 响应/传递（albedo·BRDF / ramp / 自定义）一律在采样期施加 → 四线精度从源头一致。

// 视角无关入射辐射量（阶段 1 产物；由引擎光照积分写入，与材质线无关）
struct IncomingRadiance {
    irradiance: vec3<f32>,              // 解码后的入射辐照度（感知编码在存取两侧，中间线性）
    dir_sh: array<vec3<f32>, 4>,        // 可选方向性（SH-L1=4 系数），has_sh=false 时忽略
    has_sh: bool,
}

// 阶段 2：采样期四线统一响应入口。读共享入射辐照度 + 采样期全分辨率细节 + 视角 → 颜色
fn respond(surface: Surface, inc: IncomingRadiance, view: View) -> vec4<f32>;
//   PBR : 全分辨率乘 albedo·评估 BRDF；高光用 inc.dir_sh / split-sum（视角相关在此算）
//   NPR : 对 inc.irradiance 施 ramp LUT；阈值交越处用 fwidth 做解析抗锯齿（锐边与缓存位深解耦，§14.8）
//   混合: 按 illumination 遮罩在采样期选 PBR/NPR 响应（缓存不分叉）
//   自定义: 走下方 custom 契约

// 自定义线（可选额外缓存视角无关量；最简情形只实现 respond_custom，直接用共享入射辐照度）
fn shade_cacheable_aux(surface: Surface, light_env: LightEnv) -> CustomAux;        // 可选：额外视角无关量进缓存
fn respond_custom(surface: Surface, inc: IncomingRadiance, aux: CustomAux, view: View) -> vec4<f32>;
//   不实现 cacheable_aux → 只用共享入射辐照度；不实现 respond_custom → 安全退化为屏幕空间 über（§8）

// 四线共享：主 pass 命中缓存 → respond() 合成；未命中/首触 → §8 über 屏幕空间回退（同一执行路径，§8.5；非并行栈）
```


---

## 17. 设计升级方向（在不违反"确定性 / 无硬顶 / 无塌缩 / 四线正交 / 无 AI"前提下的演进）

以下为对现设计的可选升级，均保持"经典、解析、可对拍"，不引入 AI/ML/神经网络。按收益/成本给出优先级建议（见各节末）。

### 17.1 运行时动态参数通道（打破"全烘焙"的刚性）

现设计把一切固化进烘焙 über（§6），对**静态材质**最优，但对"同一材质需随运行时状态连续变化"的场景（**湿度 wetness、积雪 snow、战损 damage、溶解 dissolve、生长**）过于刚性——这些轴若靠"每状态烘一份"会 permutation 爆炸，靠纹理 blend 又丢物理。

**升级**：在烘焙静态 über 之外，开一条**少量**（建议 ≤4）**运行时动态轴**，运行时按解析式调制既有 über 字段（如 wetness 降 roughness/升 specular/压暗 base_color、snow 叠加白 base + 高 roughness 顶面瓣、dissolve 驱动 alpha + 自发光边）。关键约束：动态轴是**解析式调制**（无分支爆炸、无逐像素预算），调制函数 CPU/WESL 双实现对拍，动态段在 ABI header 显式声明（P2-g）。这保留"无塌缩/确定性"，只是把"输入"从纯静态扩成"静态 + 少量解析动态"。

> 优先级：**高**。覆盖大量实际美术诉求，成本可控（解析调制 + 显式 ABI 段）。

### 17.2 原理化分层 BSDF（coat/base 能量耦合，取代瓣求和）

现 über 倾向"各瓣独立求值再和"。物理上 **coat 之下的 base 应收到 coat 透射后的能量**（coat 反射掉的不再喂 base），且 coat 吸收会使底色变暗（`coat_darkening`）。纯瓣求和在强 coat/高 IOR 下能量不自洽。

**升级**：对**分层关系**（coat↔base、fuzz↔base）引入**能量耦合**——按 coat Fresnel 透射率做能量预算，coat 吸收/变暗统一建模。耦合项查表化（类似多散射补偿 §10），进 furnace 守恒门禁 + 双线对拍（P2-h）。这不是加瓣，是"把已有瓣的组合方式从求和升级为分层预算"，仍确定性、仍无硬顶。

> 优先级：**高**。直接提升画质物理正确性，与现 furnace 门禁机制同构、落地路径清晰。

### 17.3 光谱提升（hero-wavelength，仅 RT twin / 离线参考）

RGB 渲染在**强色散、薄膜虹彩、荧光**下与真值有偏差。全实时转光谱成本过高，且与现 RGB 主线/对拍冲突。

**升级**：在**离线/RT twin**（光线追踪对照线）引入 **hero-wavelength 光谱提升**——把 RGB 反照率提升为光谱、按 hero 波长采样，作为"色散/薄膜/荧光"的**高保真对拍基准**。实时主线仍 RGB；光谱仅跑在 RT-only 分支（P3-d）。这给 thin_film（§10）、dispersion 一个真值标尺，不冲击实时确定性。

> 优先级：**中**。价值集中在高保真离线对拍，不影响实时主线；工程上是独立 RT 分支。

### 17.4 随机单瓣采样（blue-noise，摊销贵瓣）

多瓣材质"每像素全瓣求和"在瓣数增长时线性变贵。

**升级**：对**贵瓣**（SSS/透射/多散射）按预算做**随机单瓣采样**——每像素按瓣权重用 **blue-noise** 概率选 1 个贵瓣求值，配合 TAA/时序累积收敛。廉价瓣（diffuse/specular）仍确定性求和。采样档可调（P3-e），低档靠 blue-noise + TAA 控噪。这是"降采样质量"而非"塌缩瓣集"，不违反无塌缩（瓣集不变，只变采样策略）。

> 优先级：**中**。对高瓣数材质的性能上限有用，但依赖 TAA、需控噪，列为可选性能档。

### 17.5 自描述 schema：PCM 的单一声明源与全链路代码生成（深化设计）

今天 lobe 的字段布局、位掩码、置换 key、默认值、合法区间分散在 Rust（`surface.rs`/`record.rs`）、WESL（`material_unpack.wesl`/`material_classification.wesl`/`material_sample.wesl`）、CPU 金标准（`prism_render_shading`）、文档字段表（§4.4）、合规枚举（§4.5）**六处各自手写**；补一个瓣要六处同步改，任一处漂移就是一个 CPU/GPU 不对拍或合规漏判的隐性 bug。§5.1/§5.4 已把它定调为"所有 kind 的 manifest 格式 = §17.5"并反复依赖——本节把这块从一句方向**补成那条被依赖的主干**：**一个声明式 schema 作为 PCM 的唯一真相源，六处产物全部编译期代码生成**。这是"开放词汇表"从"能扩"跃迁到"易扩且结构上不出错"的关键基建，也是 §11 双线对拍从"手工维护两份"升级为"同源生成、字节一致是构造性结果"的根因。

#### 17.5.1 对标与借鉴（取其形，去其短）

| 来源 | 借鉴点 | 去其短 |
|---|---|---|
| **USD `Sdr`/`Ndr` 着色定义注册** | 着色节点的输入/输出/元数据**自描述、可查询、可注册** | 运行时查询；我们在烘焙期蒸发为常量 |
| **MaterialX `nodedef`** | 声明式节点接口 + 版本 + 命名空间，天然对接 §4 作者层 | 仅描述接口不生成运行时布局；我们延伸到位级 codegen |
| **LLVM TableGen** | 声明式 record → **多后端** codegen（一个源出多目标）的范式本体 | 领域是指令选择；我们把范式搬到瓣/能力布局 |
| **glTF KHR + JSON Schema** | 注册表 + 校验 + `namespace` + 向后兼容的扩展治理 | 纯数据交换；我们加编译期布局与特化 |
| **FlatBuffers / Cap'n Proto** | schema → **确定性二进制布局** codegen，跨语言零漂移 | 通用序列化；我们绑定 ABI v5 位段与置换 key |
| **Slang reflection / UE `FShaderParametersMetadata`** | CPU↔GPU 参数布局**单源绑定**，杜绝两侧手对 offset | 偏运行时反射/宏；我们编译期一次性生成、运行时零反射 |

一句话取舍：**取"自描述 + 单源多后端 codegen + 命名空间治理"，去"运行时反射/查询"**——所有生成与校验落在烘焙/加载期（§5.2），运行时只剩已特化好的那份 WESL，零反射、零注册表查询。

#### 17.5.2 schema 内容模型（声明什么）

schema 是 §5.4 manifest 的**字段级完整形态**：manifest 描述一条能力的身份/归属/kind，schema 进一步描述该能力**每个字段**的布局与契约。每字段一条声明 record：

```toml
# 示意：KHR coat 瓣的一个字段声明（声明式，无行为代码）
[[lobe.KHR_coat.field]]
name          = "coat_roughness"
type          = "f16"            # 类型 -> 位宽/打包规则由类型表派生
bit_region    = "KHR"           # 位区分区键 = namespace（§5.4）
pack_slot     = 3               # registry_slot 内的稳定偏移，永不回收
spec_subfield = "closure_mask"  # 贡献的 SpecializationId 子区（§8.1）
consumed_by   = ["unpack", "sample"]   # §8.5 哪些步骤消费
default       = 0.0
valid         = "0.0..=1.0"     # 合法性谓词 -> 校验 + 调试断言
energy_policy = "coupled"       # 对接 §17.2 分层能量预算
view          = "invariant"     # §14.3 视角无关 -> 可进对象空间缓存
```

字段维度覆盖：**名/类型/位宽/位区(=namespace)/打包偏移/贡献的 SpecializationId 子区/§8.5 消费步骤/默认值/合法谓词/能量策略/视角相关性/调试视图绑定/金标准绑定**。schema 之于 PCM 的九类 kind 一视同仁——lobe 声明字段，provider 声明页格式，composite 声明 pass I/O，response-mode 声明分类 class 条目，皆同构。

#### 17.5.3 codegen 管线（从一个源生成什么）

schema → **一个确定性纯函数** → 六处产物，全部编译期生成、运行时蒸发：

| # | 生成目标 | 取代今天的手写处 | 锚点 |
|---|---|---|---|
| ① | Rust 解包/打包布局 | `surface.rs` / `record.rs` 手写 offset | surface.rs:153 |
| ② | WESL 解包/分类/采样布局 | `material_unpack`/`classification`/`sample.wesl` | §8.5 步骤1/2/3 |
| ③ | CPU 金标准布局 | `prism_render_shading` 手写镜像 | §11 |
| ④ | 内建调试视图字段表 | §4.6 调试视图逐字段手列 | §4.6 |
| ⑤ | 文档字段表 + 合规枚举 | §4.4 字段表 / §4.5 OpenPBR 枚举 | §4.4/§4.5 |
| ⑥ | 校验谓词（三约束可机检） | §4.5 人工审读约定 | §4.5 |

关键红利：② 与 ③ **同源生成** → §11 的"CPU 金标准 vs WESL twin 字节级一致"从"人工维护两份、靠对拍测试兜底"升级为**构造性保证**（同一 schema 过同一 codegen，布局字节相同是编译期事实，对拍测试退化为冗余防线而非唯一防线）。⑥ 让 §4.5 三约束（合规对拍/位区隔离/向下投影）从"人工审读"变为"CI 枚举字段即可守门"，与 §5.4 的字段谓词闭环。

#### 17.5.4 不变量守恒（逐条）

| 不变量 | schema+codegen 如何守 |
|---|---|
| **无 AI** | 纯声明式数据 + 确定性 codegen，经典元编程，无任何学习/推断 |
| **无全局硬顶** | schema 无 `MAX_*`；位区按 `namespace` 分区增长，位宽是每材质编译期常量，不是全局天花板 |
| **确定性 + 双线** | 单源过纯函数 codegen -> 两线布局字节同构（§17.5.3 红利），drift 这一 bug 类被结构性消灭 |
| **无塌缩** | schema 描述瓣集布局，不改求值；降级仍只改输入（§7） |
| **四线正交** | schema 以位区/消费步骤描述正交响应，不生成"第二条管线"（§8.5 单路径） |
| **烘焙期单态化** | codegen + 静态特化全在烘焙期；运行时零反射、零注册表查询（§5.2 红线） |
| **每材质静态特化** | 生成的布局按 `lobe_mask` 在烘焙期切片，未用字段编译掉（无损） |

#### 17.5.5 边界：schema 管布局与契约，不管任意数学（诚实划界）

schema **只声明布局 + 元数据 + 契约**（在哪、多宽、谁消费、合法区间、能量/视角策略），**不表达瓣的求值数学**——后者仍写在 §5.4 的类型化 hook（受限 Rust 瓣或逃生口 WESL，§5.4.1）里。这条界必须清楚：把任意数学塞进 schema 会让它退化成又一门图灵完备语言，既难确定性对拍又难 codegen。schema 负责"字段契约 + 全链路布局同步"，hook 负责"行为"，二者各司其职——这正是 USD `Sdr`（描述接口）与着色实现分离的取舍。

#### 17.5.6 五维度

- **性能**：运行时**零成本**——codegen 与校验全在烘焙期蒸发，生成的是最优打包布局 + 每材质单态化 WESL（§5.2）；顺带消灭"手写 offset 错位导致的隐性慢路径/错解包"。
- **效果**：不直接改画质，但消灭 CPU/GPU 布局漂移这一**正确性 bug 类**；并因补瓣成本骤降，间接加速 §17.2 分层能量、§17.6 glint/测量瓣等画质项的落地迭代。
- **易用**：补瓣/补能力 = 改一条 schema record + 重新生成，不再六处手对；作者/工程看到的字段表、调试视图、文档**自动同源**，无"文档与实现不符"。
- **易扩展**：这是**开放词汇表的放大器**——§5 把"9 扩展点扩到 10/20"落为"注册表加一条 manifest"，schema 进一步把"加一条 manifest"落为"加一组声明 record 后重新生成"，位区按 `namespace` 自然隔离、永不撞车（§5.4）。
- **易维护**：**单一真相源 + codegen 杀整类漂移 bug**；schema 可 diff、可版本化、可复现；CI 校验"生成物与 schema 同步"（stale codegen 直接标红），且 `KHR` 区自动对 OpenPBR 参考枚举（§4.5 约束1）。

#### 17.5.7 风险 → 解决方案总表

| 风险 | 解决方案 |
|---|---|
| codegen 增编译/构建成本 | 按 schema 内容哈希做**增量生成 + 缓存**；生成物入库，CI 只校验"新鲜度"（与 schema 同步），非每次全量重生 |
| schema 表达力不够（某瓣需复杂逻辑） | 明确划界（§17.5.5）：schema 管布局/契约，行为留在类型化 hook；schema 不追求图灵完备 |
| 版本迁移 / ABI 演进 | schema 版本与 **ABI v5** 耦合（§8），迁移 = schema diff；`registry_slot` 永不回收保证老位绑定稳定（§5.4） |
| 生成代码难调试 | 生成物**可读 + 源映射回 schema 行 + 确定性**（同源同输出）；调试视图亦同源生成（§4.6） |
| schema 自身偏离 OpenPBR 规范 | `namespace==KHR` 区由 ⑤/⑥ 自动对 OpenPBR 参考**枚举对拍**，偏离即 CI 标红（§4.5 约束1） |

#### 17.5.8 落地阶段（细化 §13 P1-g / P2-i）

- **S1（随 P1-g）**：定义 schema IDL + 注册表格式；把现有九类 kind 的既有字段**声明化**（先不生成），与手写布局做**平价校验**（schema 推导布局 == 手写布局，字节比对）。（✅ **已落地**：`pkg/prism_material_schema` 的 `schema/surface.toml` + `surface_schema()`；平价门禁 `prism_render_material/tests/schema_parity.rs`，对 2^7 全瓣遮罩字节比对。）
- **S2（P1-g 完成）**：codegen 产出 ① Rust 布局常量/瓣表 + ② WESL 解包；CI 以字节平价对现有手写布局守门，确认 codegen 无回归后切换为生成物。（✅ **已落地**：`prism_material_schema::codegen`（`emit_wesl_unpack`/`emit_rust_layout` 纯函数）+ `prism-material-codegen` 二进制（`write`/`check`）；生成物 `prism_render_scene/.../material_unpack.wesl` 与 `prism_render_material/src/surface_layout.rs` 已切换为生成，`surface.rs` 改为 re-export 生成常量；新鲜度门禁 `tests/codegen_freshness.rs` 守 committed==regenerated，`check` 子命令供 CI 调用。瓣表与 `LobeMask` 由 `surface.rs::layout_matches_generated_table` 对拍。生成 WESL 与手写行为一致，仅修正过期注释 36/`12+6*4`→40/`12+7*4`。）
- **S3（P2-i）**：扩展 codegen 到 ③ 金标准 / ② 分类·采样 / ④ 调试视图 / ⑤ 文档·合规枚举；**删除手写布局**，schema 成为唯一源；§11 双线对拍转为构造性保证 + 冗余回归。
- **S4（P2-i 收尾）**：schema 驱动 §4.5 三约束谓词 + §5 PCM 全 kind 注册；至此"加扩展点 = 加 schema record + 重新生成"全链路成立，`extensions_used/required_mask`（§5.4）由 schema 生成。

> 优先级：**高（工程杠杆，先行基建）**。§5.1/§5.4/§8.5 多处已把它当主干依赖——它是补瓣、合规门禁、双线对拍、PCM 全 kind 扩展共同的地基，应与 P1-h PCM 地基并行先行。诚实代价：需先把六处手写布局的隐契约完整反向声明为 schema（S1 平价校验量不小），且需长期维护 codegen 工具链本身。

### 17.6 glint / 测量 BRDF·BTF 瓣 + MaterialX 往返

- **glint / 闪光瓣**：车漆金属片、雪、沙的离散闪烁，现 GGX 微表面模型无法表达。作为开放词汇表新瓣（解析 glint，如 Zirr/Jakob 多尺度法线分布），走 §4.4 加法路径。
- **测量 BRDF / BTF 瓣**：对无法解析建模的真实材质（织物、复杂涂层），引入**测量数据瓣**（表格化 BRDF/BTF），作为开放瓣的"数据驱动但非 AI"成员——纯查表插值，确定性、可对拍。
- **MaterialX 往返（round-trip）**：作者层（§4）不止"导入"MaterialX，还支持**导出**——结合 §4.5 向下投影，Prism über ↔ MaterialX 可往返（扩展瓣标注为私有节点），打通外部 DCC 协作。

> 优先级：glint **中**（视觉收益明确）；测量瓣 **中低**（场景特定）；MaterialX 往返 **中**（协作价值，依赖 §4.5 投影）。

**升级优先级小结**：先做 **17.5 自描述 schema**（本轮已深化为 PCM 单一声明源与全链路 codegen，见 §17.5.1–8；工程基建，放大后续所有补瓣效率）、**17.2 原理化分层**（画质正确性，机制同构）、**17.1 运行时动态通道**（覆盖大量美术诉求）；再按需 17.6 glint / 17.3 光谱 RT / 17.4 随机采样。全部保持确定性、可双线对拍、无全局硬顶、无 AI。

---

## 附录 B：术语速查
- **一条线**：作者→烘焙→运行只有一条管线、一个 über 模型族；没有"固定深度 slab 线"与"不计数 über 线"的二分。
- **无全局硬顶**：没有 `MAX_LOBE/MAX_TEXTURE/MAX_DEPTH/MAX_LAYER` 等人为天花板；界靠每材质编译期静态特化。
- **开放 über**：lobe 词汇表可扩展、参数块变长，瓣集按材质声明，不受固定 7 瓣约束。
- **每材质静态特化**：按 lobe_mask 编译掉用不到的瓣（无损）；代价是编译期常量，不是运行时逐像素预算。
- **无塌缩降级**：降级只改输入（纹理 mip/VT）与采样质量，不改 BSDF 瓣集；取代有损的瓣/层塌缩与 per-pixel 裁剪。
- **纹理空间分层混合**：层栈在烘焙期按遮罩逐纹素 lerp 同参数化 über（无损，任意层数），运行时不留图。Frostbite 模型。
- **高光抗锯齿**：Toksvig/LEAN，把法线细节丢失转成等效 roughness，远处不闪烁。
- **置换治理**：内容哈希去重 + archetype 聚类 + 按需异步编译 + 全瓣 über 回退，用来吸收"无界"带来的 permutation 成本。
- **双线对拍**：CPU 金标准 + WESL GPU twin 字节级对拍（这是测试双线，刻意保留）。
- **OpenPBR Surface**：ASWF/Adobe/Autodesk 2024 统一 PBR 作者 + BSDF 规范（base/specular/coat/fuzz/subsurface/transmission/emission/thin_film/geometry 分组）。Prism 运行时 über 的权威参数化，现 7 瓣是其子集，缺口加法补齐。
- **四线正交**：PBR/NPR/混合/自定义不是四条管线，而是同一条 OpenPBR über 上由 `Illumination` 轴 + `render_class` + `custom_program` 选择的四种响应模式；共享解包/分类/降级/对拍。
- **补瓣加法**：开放词汇表下，新物理瓣 = 新 mask 位 + 变长字段 + 静态置换，不影响老材质、不加全局硬顶。thin_film 虹彩是首个示例。
- **FACE 私有瓣**：OpenPBR（纯物理）之外的 Prism NPR 专用瓣（SDF 面阴影），由 Stylized 线消费，与 OpenPBR 物理瓣同块共存。
- **OpenPBR 超集**：Prism über = OpenPBR Surface（全字段）∪ FACE 私有瓣 ∪ 开放词汇表后续瓣；关掉扩展即退化为纯 OpenPBR（§4.4/§4.5）。
- **合规门禁**：对纯 OpenPBR 子集材质，Prism 响应与 OpenPBR 参考实现 furnace/定向光字节对拍，进 CI，证明"真超集"（§4.5）。
- **私有位区 / 位段隔离**：`LobeMask` 分规范核心区（OpenPBR）与扩展区（FACE/开放瓣），位段互不重叠，解包按区路由（§4.5）。
- **向下投影**：`project_to_openpbr`，把含扩展瓣的 über 有损投影为合法纯 OpenPBR 参数块，用于外部导出（§4.5）。
- **运行时动态通道**：烘焙静态 über 之外的少量（≤4）解析式动态轴（wetness/snow/damage/dissolve），运行时调制既有字段，确定性可对拍（§17.1）。
- **原理化分层**：coat/base 按 Fresnel 透射做能量耦合（非瓣求和）+ coat 吸收变暗，进 furnace 守恒门禁（§17.2）。
- **光谱提升**：hero-wavelength 光谱采样，仅 RT twin / 离线参考，作色散/薄膜高保真对拍基准，实时主线仍 RGB（§17.3）。
- **随机单瓣采样**：贵瓣按 blue-noise 概率选单瓣求值 + TAA 收敛；降采样质量而非塌缩瓣集（§17.4）。
- **自描述 schema / PCM 单一声明源**：把每字段的名/类型/位宽/位区/打包偏移/置换 key/消费步骤/合法谓词数据化为唯一声明源，Rust 解包·WESL 解包/分类/采样·CPU 金标准·调试视图·文档·合规谓词六处产物全部编译期 codegen；双线字节一致从"手工维护"升为"同源生成的构造性结果"，运行时零反射（§17.5）。
- **PCM 能力基座**：Prism Capability Model——把九类扩展点（瓣/参数通道/响应模式/传递重映射/缓存量/数据提供者/降级策略/作者节点/合成 pass）统一成同一套声明式能力契约；超集加瓣只是其中一类；全部烘焙期单态化、运行时零分发（§5）。
- **扩展点 kind**：PCM 中一类可注册能力的类别，各有类型化 hook 签名与 §8.5 落点，由 manifest 的 `kind` 字段指定（§5）。
- **核心自举 / dogfood**：OpenPBR 核心瓣本身也注册为 `KHR` 命名空间能力，代码无“核心 vs 扩展”特判，仅以命名空间/稳定级/位区区分（§5.1）。
- **能力协商 capability**：manifest 声明 `requires_caps`，烘焙/加载期与平台 caps 协商；required 不满足报错、used 不满足优雅降级（§5.3）。
- **命名空间与晋升**：`namespace` 为 manifest 的独立可变字段——`KHR`（规范核心）/`EXT`（跨厂）/`PRISM`（本体私有）/`X_studio`（工作室私有）；晋升 `X_studio→EXT→KHR` 只翻字段，不改 `uid`/`registry_slot` 身份与位区（§5.4）。
- **响应模式注册化**：步骤2 分类类数从封闭四值 `Illumination` 升级为注册驱动的烘焙期固定表（唯一动热路径项，§5.2）。
- **作者迭代闭环 / 可观测性**：热重载 + über 回退 PSO 的零卡顿迭代、内建确定性调试视图（lobe_mask / 置换 / 缓存 / 波前 / furnace 残差）、验证门禁前移、代价可观测；对所有 PCM kind 一视同仁自动继承，不逐 kind 手写工具（§4.6）。
- **行为作者三档 / 受限 Rust 瓣**：自定义瓣行为分三档——① 节点图组合已验证原语（覆盖 ~95%，无新金标准）→ ② **受限 Rust 子集**经 `proc-macro` 生成 WESL 孪生（CPU 金标准即原生 Rust，零转译，`golden="auto"`）→ ③ 裸 WESL + 手写金标准逃生口（`golden="none"`）；取代早期"受限 DSL"，不自造语言；② 为受检子集（禁 heap/dyn/递归/无界循环），浮点确定性靠两侧共享 `no_std` 数学库（§5.4.1）。
- **编辑器渲染插件 / 打包即烘焙**：渲染插件在**编辑器运行时**动态加载 / 热重载，**打包即单态化烘焙**成出货运行时的静态零 dispatch 特化；插件不随出货、只烘产物进包；编辑器预览与出货**同源一份 WESL**（WYSIWYG），双线门禁落在打包口；插件边界即**信任边界**（签名 / wasm 沙箱 + CI 字节对拍）（§5.4.2）。
