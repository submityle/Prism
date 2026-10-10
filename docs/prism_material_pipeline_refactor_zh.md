# Prism 材质系统 — 破坏性重构设计（v10 / OpenPBR Surface 规范参数化 · 开放 über · 每材质静态特化 · 无全局硬顶 · 无塌缩 · 四线正交 · 传递无关缓存 · 统一执行模型版）

> 状态：架构提案（Draft，**允许破坏性重构，不保留旧 API**）
> 面向版本：Prism 下一代渲染底座（Bevy fork，`pkg/` workspace）
> Shader 语言：**WESL**（经 naga → WGSL/SPIR-V/Metal/DXIL），不引入 Slang
> 运行时后端：**wgpu**（Metal/Vulkan/D3D12/WebGPU/主机/移动全覆盖）
> 本文依据：对 `pkg/prism_render_material`、`prism_render_shading`、`prism_render_scene/src/shaders/*.wesl`、`prism_render_architecture` 的静态阅读（以代码为准）
> 核心命题：**没有全局硬顶，不做运行时塌缩，也不做 per-pixel 预算裁剪。模型开放可扩展，每材质按声明静态特化成它确切需要的瓣集与特征——"界"是每材质编译期事实，不是全局天花板。分层在烘焙期解析成同参数化纹理空间混合（层数天然不限）；距离/性能靠纹理预过滤 + 高光抗锯齿 + 静态置换 + 采样缩放，全程无损、确定、跨平台。**
> 规范参数化：**运行时 über 的权威参数集 = OpenPBR Surface（ASWF/Adobe/Autodesk 2024）。** 现有 7 瓣是 OpenPBR 的实现子集；补齐缺口（thin_film 虹彩、coat/fuzz/subsurface 全字段等）是对开放词汇表的**加法扩充**，不是改架构。FACE 作为 OpenPBR 之外的 Prism 私有瓣服务 NPR。
> 四线正交：**PBR / NPR / 混合 / 自定义不是四条管线，而是同一条 OpenPBR über 上的正交响应模式**，由 `axis.rs` 的 `illumination` 轴 + `render_class` + 可选 `custom_program` 钩子选择；共享同一套参数解包、per-tile 分类与降级。
> 最后更新：2026-10-04（v7：OpenPBR Surface 规范化 + 四线正交深化）

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
| `MAX_MATERIAL_TEXTURES = 8` | `record.rs:8` | 纹理槽硬上限 | **删**，改 VT 页引用（§8） |
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
| **PBR 物理** | `Illumination::Lit(0)` | OpenPBR 全瓣参数 | OpenPBR über 块 + lobe_mask | 能量守恒 BSDF（多散射 GGX、菲涅尔、瓣叠加） | VT mip + 高光抗锯齿 + 采样缩放（§6） |
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

- **效果（必达物理项，进 §9 CI 门禁）**：
  - 多散射 GGX 能量补偿（Kulla-Conty 2017 解析近似 / Activision 查表），高糙度金属不发黑；
  - coat 分层含吸收与色移（OpenPBR `coat_color/coat_ior/coat_darkening`，§4.4），湿表面/车漆正确变暗；
  - 各向异性（Burley GGX，`anisotropy/anisotropy_rotation`）、sheen（Estévez-Kulla charlie 布料）、SSS（Christensen-Burley 可分离近似）；
  - thin_film 虹彩（Belcour-Barla 2017）作为开放词汇表首个新瓣（§4.4/§9）。
- **性能**：多散射/IBL 全走**预积分 LUT 或解析近似**（一次查表，不迭代）；coat 二层但未用则编译期静态置换掉（§7）；远处 **Toksvig/LEAN** 把法线细节转等效 roughness 消闪烁（§9）；视角无关项（diffuse/irradiance/SSS 漫透）可进 §13.3 对象空间缓存，高光走 split-sum 屏幕空间。
- **易用**：`Metal/Skin/Glass/CarPaint/Water` archetype 填参（§4.2），OpenPBR 全字段作者端可见；furnace 白炉子 CI 守能量守恒，美术改参不破物理。

### 3.5.2 NPR 风格化线（深化：性能 / 效果 / 易用）

> 对标 **Arc System Works《Guilty Gear Xrd》**（顶点色控阴影 / 法线编辑 / 背面挤出描边）、**miHoYo《原神》**（ramp 图集 / matcap 金属 / SDF 面阴影）、**UE Toon / Blender NPR**。目标：风格可控且与同一 über 光照输入解耦，不另造 NPR 管线。

- **效果**：
  - **光照量化 ramp**：缓存"量化前的连续辐照度"（N·L 带状化前），采样期查 ramp LUT 施加量化曲线——连续量缓存更稳、避免档边闪烁（§13.3 已定此切分）；
  - **SDF 面阴影（FACE 瓣，`surface.rs::softness`）**：miHoYo 式脸部阴影随光照方位平滑过渡；FACE 是 **OpenPBR 之外的 Prism 私有瓣**（§4.4），由 `Illumination::Stylized(1)` 消费；
  - **描边**：屏幕空间几何（背面挤出 / 后处理法线·深度边缘），**永远屏幕空间**，宽度随距离收敛；
  - **风格化高光**：阶跃锐边 / matcap / Kajiya-Kay 发丝各向异性高光。
- **性能**：ramp/matcap/FACE 均为 **LUT 或单张贴图采样**（廉价）；量化在采样期逐像素查表（便宜且防档跳）；描边为一次后处理；NPR 可缓存项（ramp 输入辐照度、材质色、面阴影采样）进 §13.3，描边与锐边高光屏幕空间。**不变量：PBR/NPR 精度从架构上就一致，无按线特判**——缓存只存传递无关的感知均匀入射辐照度（四线共用一份，§13.3/§13.8），ramp/BRDF 等响应全在采样期施加；NPR 台阶锐度由采样期对解码辐照度的屏幕梯度做解析抗锯齿生成，**与缓存位深解耦**（§13.8），故既不降 NPR 精度、PBR/NPR 又走同一条路径。降级（§6）只作用于采样数/纹理 mip，不碰响应质量。
- **易用**：`Toon` archetype；ramp 曲线美术可画、matcap 直给贴图；描边参数与物理参数解耦互不污染；美术不碰 BSDF 也能出稳定风格。

### 3.5.3 混合线（深化：性能 / 效果 / 易用）

> 对标 **UE 多 ShadingModel 混排 / `ShadingModelFromMaterialExpression`**、写实场景 + 卡通角色同屏。目标：一个画面内 PBR 与 NPR 响应共存，仍是一条 über。

- **效果**：逐材质 / 逐区域 `illumination` 取值（PBR 底 + NPR 覆盖，遮罩选择），同一 über 块里按区域切响应，支持"写实环境 + 风格化主角"。
- **性能（关键）**：**per-tile 9-way 分类**（`classification.rs` 已含 `Npr/Custom`，`material_classification.wesl` prefix-sum/scatter 无 readback）把 Lit/Stylized **分到不同波前**，消除 warp 内分支发散——这是混合高性能的根（对标 Nanite 材质 per-tile classify）。缓存层按 `illumination` 分通道存，命中与否不破坏一致性（§13.3）。
- **易用**：美术画 `illumination` 遮罩即可；合成期按遮罩相加；无需跨管线合成，无额外作者心智。

### 3.5.4 自定义线（深化：性能 / 效果 / 易用）

> 对标 **UE Custom HLSL node / Material Function / 自定义 ShadingModel**、**Unity ShaderGraph Custom Function**。目标：开放逃生舱口，但不破一条线的解包/分类/降级/对拍基建。

- **效果**：作者挂自定义 WESL 片段/子图，**编译期编入置换**；仍复用 OpenPBR über 解包（`material_unpack.wesl`）+ per-tile 分类，自定义只替换"用输入算什么响应"。
- **采样期响应契约**（与 §13.9/§15 统一）：缓存只存传递无关入射辐照度，自定义程序实现 `respond_custom(surface, inc, aux, view)`（采样期响应，必实现）/ 可选 `shade_cacheable_aux(surface, light_env)`（声明额外视角无关量进缓存）；**不实现 `respond_custom` → 自动全屏幕空间安全退化**。
- **性能**：编译期编入=静态置换，无运行时解释；自定义声明自己的采样缩放钩子（§6）与 cacheable 切分（§13），默认不拖慢其它三线。
- **易用**：节点图或片段两种入口；默认安全（屏幕空间、无需懂缓存）；契约为 opt-in 加速，渐进采用。

### 3.5.5 四线共享：性能 / 效果 / 易用总矩阵与治理

| 维度 | PBR | NPR | 混合 | 自定义 |
|---|---|---|---|---|
| **效果关键项** | 多散射 GGX / coat 分层 / 虹彩 | ramp 量化 / SDF 面阴影 / 描边 | 区域级 PBR⊕NPR 共存 | 作者定义响应 |
| **性能手段** | LUT/解析近似 + Toksvig + split-sum | LUT/matcap + 后处理描边 | per-tile 分波前消发散 | 静态置换编入 + 契约切分 |
| **§13 缓存切分** | diffuse/irradiance/SSS 缓存，高光屏幕 | ramp 输入辐照度/面阴影缓存，描边屏幕 | 按 illumination 分通道缓存 | 契约声明 cacheable，否则屏幕 |
| **易用入口** | Metal/Skin/Glass… archetype | Toon archetype + ramp 曲线 | illumination 遮罩 | 节点图/WESL 片段 |
| **对拍门禁** | furnace + 字节对拍 | ramp/面阴影响应对拍 | 分区域各自对拍 | 自定义输出对拍 |

**治理（四线不加管线维度）**：四线共享同一 permutation 池，键是 `axis.rs::SpecializationId`（`illumination[0..8]` / `closure_mask[8..40]` / `render_class[40..48]`，**无 tier**）；NPR/Custom 只是 `illumination` 值与 `render_class`/`custom_program` 的取值，不新增管线轴。permutation 成本由内容哈希去重 + archetype 聚类 + 按需异步编译 + 全瓣 über 回退吸收（§7.4）。**四线正交、一条基建**：解包/分类/降级/对拍/缓存五套地基只建一次。

---

## 4. 作者层：OpenPBR / MaterialX 前端（易用 + "无界"的落点）

### 4.1 新 crate `prism_render_material_authoring`
- 输入：OpenPBR Surface 参数集，或 MaterialX 节点图（`.mtlx`），**层数/节点/瓣类不限**。
- 输出：烘焙层的输入（层栈 + 参数曲线 + 遮罩来源 + 声明的瓣集）。
- OpenPBR 与现 über 近乎一一对应（base/specular→core，coat→CLEARCOAT，fuzz→SHEEN，subsurface→SUBSURFACE，transmission→TRANSMISSION，anisotropy→ANISOTROPY，emission→EMISSION，Prism 扩展 FACE 给 NPR）；**新瓣可作为开放词汇表的一等成员加入**，不必塞进固定槽。

### 4.2 archetype 模板
预置 `Metal / Skin / Cloth / Glass / Foliage / Toon / CarPaint / Water`，美术选模板填参，不从空图搭。模板也作为**置换聚类锚点**（见 §7.4），把常见瓣集组合收敛成少量热门 permutation。

### 4.3 "无界"如何满足而不失确定性
作者端的层数/节点/瓣类自由，在 **import/bake** 阶段被解析（见 §5）成 über 参数 + 遮罩纹理 + **该材质声明的瓣集**。运行时按这个瓣集**编译期静态特化**：用到哪些瓣就编哪些瓣，不用的不编。于是：
- 没有全局天花板（满足"不要有界"）；
- 每材质代价编译期已知（满足"确定、跨平台"）；
- 不做运行时逐像素砍瓣（否决 Substrate 式预算裁剪）。
自由在作者侧，特化在编译期，中间靠烘焙——这是关键区别。

### 4.4 OpenPBR Surface 规范映射与补瓣（加法，不改架构）

运行时 über 的权威参数集 = **OpenPBR Surface**。现 `surface.rs` 的核心 12 words + 7 瓣是 OpenPBR 的**实现子集**；补齐 = 把现有 4-word lobe 扩成 OpenPBR 全字段 lobe，走变长打包 + 静态置换，**是加法，不改模型族**。

| OpenPBR 组 / lobe | 现 über 字段（`surface.rs`） | 状态 | 待补字段（加法） |
|---|---|---|---|
| **base**（diffuse+metal） | `base_color` / `metallic` / `perceptual_roughness` | 子集 | `base_weight`、`diffuse_roughness`（Oren-Nayar 糙漫反射） |
| **specular** | `reflectance`（IOR 代理）/ `anisotropy` / `anisotropy_rotation` | 子集 | `specular_weight`、`specular_color`、显式 `specular_ior`（取代 reflectance 代理） |
| **coat** | `clearcoat` / `clearcoat_roughness` | 子集 | `coat_color`、`coat_ior`、`coat_roughness_anisotropy`、`coat_normal`、`coat_darkening` |
| **fuzz** | `sheen` | 子集 | `fuzz_color`、`fuzz_roughness` |
| **subsurface** | `subsurface` | 子集 | `subsurface_color`、`subsurface_radius`（vec3，RGB 自由程）、`scatter_anisotropy` |
| **transmission** | `transmission` / `thickness` / `index_of_refraction` / `dispersion` | 较全 | `transmission_color`、`transmission_scatter`、`thin_walled` 标志 |
| **emission** | `emission`（4 words） | ✓ | `emission_luminance` 单位统一（nit / 相对） |
| **thin_film**（虹彩） | — | **缺整瓣** | 新增 `thin_film_weight` / `thin_film_thickness` / `thin_film_ior`（§9 首个开放瓣示例） |
| **geometry** | `normal_scale` / `alpha_cutoff` + 法线纹理 | 子集 | `geometry_opacity`、`geometry_thin_walled`、`geometry_coat_normal`（随 coat 补） |
| **FACE**（NPR） | `softness`（SDF 面阴影） | ✓ | OpenPBR 之外的 **Prism 私有瓣**，NPR 专用，**不并入 OpenPBR**，与之共存 |

要点：
- **开放词汇表让补瓣是加法**：新字段扩进对应 lobe 的变长块，位掩码/静态置换天然容纳，老材质不受影响（它们的 lobe_mask 不含新位）。
- **thin_film 作为开放瓣的首个落地示例**（§9）：验证"新物理项 = 新 lobe 进词汇表"的扩展路径。
- **FACE 保持 OpenPBR 外**：OpenPBR 是物理 BSDF 标准，不含 NPR；FACE 作为 Prism 私有瓣，由 `Illumination::Stylized` 线消费（§3.5），与 OpenPBR 物理瓣在同一 über 块里共存。
- **删 `lower_standard_material`**：外部材质（含 Bevy `StandardMaterial`、glTF）统一经 OpenPBR Surface 导入器映射，不再逐格式写 1:1 翻译层。

---

## 5. 烘焙层：分层 → 开放 über（一条线的落点，无 N 层硬顶）

这是"两线合一 + 无硬顶 + 无塌缩"的核心。把运行时闭包图彻底移出运行时：

### 5.1 分层解析为纹理空间混合（任意层数）
- 层栈（**任意层数，无 MAX_LAYER**）在烘焙期按遮罩**逐纹素混合**：
  `p_texel = Σ mask_i · p_i / Σ mask_i`。
- 关键：纹理空间混合天然"归一"——不管多少层参与，一个纹素产出**一组** über 参数。层数不限不会增加运行时代价,也不需要人为封 N 层。
- 混合的是**同一套 über 参数**（base_color/metallic/roughness/…）的 lerp，**无损、线性、可预期**：和"把 clearcoat 近似并进 base"那种异构瓣塌缩完全不同——这里全程同一参数化，插值就是插值，不发明新模型。
- 瓣存在性（如某层有 coat、某层没有）通过遮罩加权后决定该纹素的 `lobe_mask`：有能量的瓣进 mask，驱动静态置换。

### 5.2 产出：开放 über 参数块
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

### 5.3 作者节点图去哪了
MaterialX/OpenPBR 的程序化节点（noise/curve/blend）在烘焙期**烘到纹理**（VT 页）或**固化成常量参数**。运行时只采样纹理 + 跑静态特化的 über，不解释图。这正是 Frostbite/UE 的实际做法。

---

## 6. 降级：改输入不改模型（取代塌缩与 per-pixel 裁剪）

每材质跑它编译期特化的瓣集，瓣数/特征在距离上**恒定**。降级只作用在"输入"和"采样质量"上，全部无损或标准：

### 6.1 距离 LOD = 纹理预过滤（无损、自动）
- VT mip 链自动降细节；带宽随距离下降，着色瓣数不变。
- **高光抗锯齿**（Toksvig / LEAN）：把 mip 下采样丢失的法线细节转成等效 roughness 增量，远处高光不闪烁。
  `α_eff² = α² + 2·σ_normal²`（σ 由法线 mip 的方差估计）。
- 这替代了塌缩里"法线聚合"的角色，但不改模型、不改瓣数。

### 6.2 性能 LOD = 静态置换 + 采样缩放（无损 / 可控）
- **静态特征置换**：按 `lobe_mask` 编译期裁掉用不到的瓣（没有 clearcoat 的材质不编 clearcoat 分支）。这是"界"的真正所在——每材质编译成它的确切瓣集，无损,确定。
- **贵瓣采样缩放**：SSS/透射/多次散射的**采样数**按画质档缩放（如 SSS 卷积半径采样 16→8→4）。模型不变,只是数值精度降,平滑可控,不跳变。
- **per-tile 分类**（已有 9-way）：保证波前一致,降低分支代价。
- 注意：这里**没有** per-pixel 动态砍瓣——瓣集在编译期就定了，运行时不赌预算。

### 6.3 为什么这比塌缩 / per-pixel 裁剪都好（对比）

| 维度 | 塌缩阶梯（否决） | per-pixel 预算裁剪（否决） | 本方案（静态特化 + 改输入） |
|---|---|---|---|
| 外观一致性 | 异构瓣合并有损，掠射角走样 | 砍瓣阈值附近跳变 | 同一模型，始终物理一致 |
| 档位过渡 | 两个模型间跳变 | 像素间不连续 | mip 连续 + 采样数平滑，无跳变 |
| 可预期性 | 塌缩结果对美术是黑盒 | 逐像素预算不可控 | 远处=模糊版自己，直觉可预期 |
| 跨平台确定性 | 需能量守恒塌缩算子 | 弱平台预算抖动 | 每材质代价编译期常量 |
| 硬顶 | 需固定档位数 | 需 per-pixel 预算上限 | **无全局硬顶，界=编译期特化** |
| 实现复杂度 | 塌缩算子 + CPU/WESL 双实现 | 动态调度 + 噪点 | 复用 VT + 高光抗锯齿 + 采样缩放 |

---

## 7. 运行层 ABI v5（破坏性，不保留 v4，无硬顶）

### 7.1 SpecializationId（去掉 tier，纯正交轴 + 开放 lobe_mask）
```
bits  0..8   illumination
bits  8..40  closure_mask (= lobe_mask 投影，驱动静态置换；位宽留足，支持开放瓣集扩展)
bits 40..48  render_class
bits 48..64  reserved（不放 tier；降级不产生新 permutation）
```
降级不改 permutation：同一材质在近景/远景用**同一个** PSO，只是采样的 VT mip 和采样数不同。瓣集扩展时在 `closure_mask` 位域内扩展，不改轴布局。

### 7.2 GpuMaterialHeader v5
相对 v4 的破坏性改动：
- 保留单一 `parameter_offset` / `parameter_size`，但其长度**由瓣集决定、变长**（不是固定 7 瓣，也不是 tier 数组）。
- `lobe_mask` 保留并作为**开放位掩码**，驱动变长解包 + 静态置换。
- **参数块布局 = OpenPBR Surface 瓣序**：核心 base/specular/geometry 常驻，可选瓣（coat/fuzz/subsurface/transmission/emission/thin_film）按 mask 低位优先顺序变长排布；WESL `prism_unpack_surface` 已是低位优先 mask 驱动聚集，补瓣只是加长每瓣字段数（§4.4），解包循环不变。
- FACE（NPR 私有瓣）在 OpenPBR 物理瓣之后排布，由 `Illumination::Stylized` 消费（§3.5）；`custom_program` 字段承接自定义线（§3.5）。
- 纹理段 `texture_offset/texture_count` → **VT 页引用**（指向 `texture_streaming` 的页表句柄），不再是固定 8 槽索引，数量不限。
- 删除一切 `MAX_*` 常量语义（`MAX_MATERIAL_TEXTURES` / `MAX_CLOSURE_SLAB_DEPTH`）。
- `MATERIAL_ABI_VERSION = 5`。
- `material_classification.wesl` 的 `MaterialHeader` 镜像同步更新字段名（现仍是 v4 布局：`parameter_offset/parameter_size/lobe_mask`）。

### 7.3 per-tile 分类（复用现有，不加 tier 维度）
- `classify_count → prefix_classes → scatter_work`（`material_classification.wesl`）保持 9-way，不扩成 9×tier。
- 质量档（采样缩放）作为 pass 级 uniform,不进分类 key——避免桶数爆炸。
- 波前一致性由 9-way 分类保证；分支代价由静态置换（lobe_mask）压低。

### 7.4 置换治理（无界的代价：permutation 爆炸的缓解）
"每材质静态特化"的真实成本是 shader permutation 数量。治理手段：
- **置换 key = 瓣集（lobe_mask）+ 特征 flags**，内容哈希去重；相同瓣集的材质共享 PSO。
- **archetype 聚类**：模板把常见瓣集组合收敛成少量热门 permutation，长尾稀少。
- **按需异步编译 + über 回退**：冷 permutation 未编译完时，临时用一个"全瓣 über"PSO 兜底着色（略贵但正确），后台编译完成后切换。保证不卡顿、不丢画面。
- **置换缓存**：以内容哈希为键落盘，跨进程/跨机复用（CI 预热）。
- 这样"无界"落在作者/模型层，permutation 数量落在可治理的工程层——而不是用一个全局 MAX 去粗暴封顶。

### 7.5 运行层统一执行模型（一条执行路径，四线共用，光照来源可切换）

关键定调：**运行层只有一条执行路径**。四线（PBR/NPR/混合/自定义）的差异只落在"采样期响应函数"（步骤 5），缓存命中与否只改"光照来源"（步骤 4），降级只改"输入纹理 mip/采样数"——这三者**都不新增 PSO、不改分类 key、不产生并行着色栈**。这是把 §3.5（四线正交）、§7.3（分类）、§13（解耦着色）、§6（降级）收敛成单一执行契约的落点。

```
1 解包    prism_unpack_surface(words, lobe_mask) → Surface（变长 OpenPBR 瓣集，低位优先聚集，§4.4/§7.2）
2 分类    per-tile 9-way（按 illumination/render_class 分波前）；质量档/预算 = pass uniform，不进 key（§7.3）
3 瓣求值  热=spec-constant 置换（编译掉未用瓣，无损）⇔ 冷=全瓣 über + mask 瓣循环兜底（二者等价，§7.4）
4 光照源  (a) 对象空间缓存命中 → 读"传递无关入射辐照度"IncomingRadiance（§13.3）
          (b) 未命中/首触/视角相关 → 屏幕空间直接积分（§7 über 回退，非并行栈）
          两源归一为同一 IncomingRadiance 结构 → 下游响应函数无感（唯一分支点，不改 permutation/key）
5 响应    respond()：PBR 乘 albedo·评估 BRDF ｜ NPR 施 ramp(+fwidth 解析 AA) ｜ 混合按 illumination 遮罩 ｜ 自定义走契约（§3.5/§13.3）
6 合成    + 视角相关项（GGX/coat 高光、描边）→ 输出；主 pass TAA
```

- **一条路径的意义**：弱平台 / 关闭缓存 = 只走步骤 4(b) 的屏幕空间源，其余步骤完全不变——缓存只是把步骤 4 的来源从"逐帧屏幕积分"换成"对象空间复用"。没有第二套着色栈，不保留旧着色 API（破坏性）。
- **四线正交落在步骤 5**：步骤 1/2/3/4 四线完全共用，差异只在 `respond()` 内部的响应函数（§3.5）；缓存层对四线中立（§13.3），这是 PBR/NPR 精度能一致的结构根因。
- **permutation 边界只由步骤 1/3 决定**：key = 瓣集（lobe_mask）+ 特征 flags；步骤 2 的质量档、步骤 4 的缓存命中、步骤 6 的降级都**不参与 key** —— permutation 数量与"降级 / 缓存状态 / 画质档"彻底解耦（§7.3/§7.4）。
- **波前一致性**：步骤 2 的 9-way 分类把 Lit/Stylized/Custom 分到不同波前，步骤 3 的静态置换压低瓣分支，步骤 4 的两源归一避免"命中/未命中"在波前内分叉成本——三者叠加保证四线混排场景下的 occupancy。

---

## 8. 纹理：虚拟纹理（距离 LOD 的载体，复用现有地基，无槽上限）

现成基础设施（无需从零造）：
- `texture_streaming/indirection.rs`：GPU 页表，每项 `PAGE_TABLE_ENTRY_WORDS=4` 词，二分可查，CPU `lookup` 与 shader 二分同序。
- `texture_streaming/residency.rs`：`NotResident→Requested→Resident` 状态机 + 优先级 + 字节成本。
- `feedback.rs / scheduler.rs / pool.rs`：反馈优先级 + 预算调度 + 物理页池。

重构动作：
- 删除 `MAX_MATERIAL_TEXTURES = 8` 的语义约束，材质引用**任意多** VT 页，不占固定槽。
- **距离 LOD 完全交给 VT mip**：远→采样高 mip，页更小、带宽更低，着色模型不变（§6.1）。
- VT 缺页（`Requested` 未到 `Resident`）时采样回退到已驻留的更高 mip——降细节而非卡顿，仍是同一模型。

---

## 9. 效果：必补的物理项（进 CI 门禁）

- **多散射 GGX 能量补偿**（Turquin / Kulla-Conty）：修正单散射 GGX 高 roughness 丢能量。查表 `E(μ,α)`、`E_avg(α)`，补偿项
  `f_ms = (1-E(μ_o))(1-E(μ_i)) / (π(1-E_avg))`，乘以 `F_avg` 的多次反射级数。
- **Furnace test（白炉）CI 门禁**：均匀环境光下验证反照率守恒。CPU 金标准 + WESL twin 都跑。
- **高光抗锯齿**（Toksvig/LEAN，§6.1）：远处高光不闪烁；这是距离 LOD 的无损手段。
- **NPR**（`Illumination::Stylized`，§3.5）：ramp 量化 / SDF 面部阴影（Prism 私有 FACE 瓣，OpenPBR 之外）/ 描边，作为对**同一 OpenPBR über 输入**的正交非物理响应；不进物理 furnace 门禁，但进 ramp/描边视觉回归。
- **thin_film 虹彩（OpenPBR thin_film 瓣，开放瓣首个落地示例）**：薄膜干涉随 `thin_film_thickness` / `thin_film_ior` 产生彩虹色高光（肥皂泡、氧化金属、涂层）。按 OpenPBR 作为新 lobe 加入开放词汇表——新增 `LobeMask::THIN_FILM` 位 + 3-word 块，`prism_unpack_surface` 低位优先循环自动容纳，老材质 mask 不含该位、零影响。这是"新物理项 = 加法补瓣"路径的验证用例（§4.4）。
- **其余待补 OpenPBR 字段**（coat_color/coat_ior、fuzz_color、subsurface_radius vec3、specular_color/weight、diffuse_roughness）同样以加法扩字段方式补齐，各进 furnace / 对拍门禁。
- **开放瓣扩展机制**：任何新物理项（如各向异性 SSS）走同一路径——新位 + 变长字段 + 静态置换，不冲击现有材质、不加全局硬顶。

---

## 10. 双线确定性（保留，这是优势）

- CPU 金标准（`prism_render_shading`）+ WESL GPU twin 字节级对拍，重构中保留并扩展覆盖：
  - über BSDF、多散射补偿、高光抗锯齿、VT 页寻址（`indirection::lookup` 已 CPU/shader 同序）、烘焙期分层混合、变长解包，均 CPU/WESL 双实现对拍。
  - furnace / 能量守恒 / VT 寻址一致性作为回归门禁。
- 注意：这与旧文档"两套并行物理栈"无关——物理侧已收敛（`prism_render_architecture` 依赖 `prism_physics_core`，cloth 为薄 façade）。

---

## 11. 性能 / 效果 / 易用性权衡总表

| 维度 | 重构前 | 重构后 | 收益来源 |
|---|---|---|---|
| 模型线数 | 固定线 + 不固定线（二分） | **一条**开放 über | §5 烘焙单一化 |
| 全局硬顶 | `MAX_LOBE=7`/`MAX_TEX=8`/`MAX_DEPTH=4` | **无**；界=每材质编译期特化 | §5/§7 |
| 作者自由 | 内部 IR 手搓，深度≤4 拒绝 | OpenPBR/MaterialX，层数/瓣类不限 | §4 作者层 |
| 降级方式 | （无统一机制） | 纹理预过滤 + 高光抗锯齿 + 采样缩放（**无损,无塌缩,无 per-pixel 裁剪**） | §6 |
| 外观一致 | 塌缩/裁剪会走样（已否决） | 全程同模型，物理一致 | §6.3 |
| 性能确定 | über worst-case 全瓣 | 每材质静态特化 + per-tile 分类 | §6.2/§7.3 |
| permutation 治理 | — | 内容哈希去重 + archetype 聚类 + 按需编译 + über 回退 | §7.4 |
| 跨平台 | wgpu+WESL 已达成 | 维持；VT/补偿走 compute 可测 | §8/§10 |
| 高端画质 | 单散射丢能量 | 多散射补偿 + 开放瓣扩展 | §9 |
| 纹理规模 | 8 槽硬墙 | 虚拟纹理（复用现有，无槽上限） | §8 |
| 回归安全 | CPU golden + WESL twin | 维持并覆盖烘焙/VT/补偿/变长解包 | §10 |

---

## 12. 落地路线图（破坏性，分阶段）

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

**破坏性说明**：P1-b 一次性弃 v4，不做兼容垫片；旧 `MaterialRecord`/`GpuMaterialHeader`/`lower_standard_material`/所有 `MAX_*` 常量直接改写或删除。

### 12.5 P3 远期：解耦着色 / texel-space shading（性能上限最高，工程最重）

不在屏幕空间逐像素着色，而在**对象/纹理空间**着色，结果写入着色缓存页、跨像素/跨帧复用；按 mip 摊销 → **天然 LOD + 消除 overshading**。性能天花板最高但工程最重，列为 **P3 远期选项**，严格作为**独立性能子系统**推进，不与 über 重构（P0–P2）耦合——über 先落地，解耦着色作为可选 pass 叠加。**完整深化设计（架构 / 四线分离求值 / 着色率 / 失效 / seam / 一致性 / 易用 / 门禁 / 风险解法）见 §13。**

---

## 13. 解耦着色（texel-space / object-space shading）深化设计

> 定位：**P3 性能上限层**。把"着色"从屏幕空间逐像素解绑，改为在**对象/纹理空间**按需着色、写入**着色缓存**，跨像素跨帧复用，按 mip 摊销。目标是在不改 OpenPBR über 模型（§4/§7）、不改四线正交语义（§3.5）的前提下，拿到"消除 overshading + 天然 LOD + 可摊销光照"的性能天花板。全程**复用现有 `texture_streaming` 地基**，不另造 VT；不保留任何旧着色路径对外 API（屏幕空间着色退化为"缓存未命中回退路径"，不是并行栈）。

### 13.1 对标与借鉴（取其形，去其短）

| 项目 / 技术 | 借鉴点 | Prism 取舍 |
|---|---|---|
| **育碧 Deferred/Texel Shading**（Far Cry 系） | 着色与光栅解耦，纹理空间着色由 VT 承载，feedback 驱动"要着哪些 texel" | **核心借鉴**：复用 VT feedback→priority→budget 链(`feedback.rs`/`scheduler.rs`)驱动"着色需求"，不是只驱动"采样需求" |
| **UE5 Lumen Surface Cache** | 对象空间低频光照缓存 + 预算重着色 + 时序复用 | 借其"按预算分帧重着 + 时序累积"的失效摊销；但我们缓存**材质响应**而非仅间接光，分辨率随 mip 自适应 |
| **Decoupled Deferred Shading（Liktor/Dachsbacher 2012）** | 着色样本 memoization 缓存，着色率与可见性率解耦 | 借其"着色样本去重/复用"思想，落到 VT 页粒度（而非 micropolygon） |
| **RenderMan Reyes / Pixar Ptex** | 纹理空间着色 + **per-face 无缝参数化**（Ptex 无显式 UV seam） | 借 Ptex 式**按几何面片分配缓存 tile**，从根上回避 UV seam（§13.6） |
| **id Tech MegaTexture / 现 `texture_streaming`** | 虚拟纹理页表 + residency + 预算调度 | **直接复用**：着色缓存 = "可写 VT 页"，页表/残留/调度/图集全部沿用 |
| **COD/VRS、Nanite 材质** | 屏幕空间可见性率与着色率解耦（VRS）、per-tile 分类 | 作为**回退路径**与补充：缓存未命中/视角相关项走屏幕空间 + per-tile 分类(§7.3) |

一句话：**用 VT 的"按需、分级、预算"机制去调度"着色"而不只是"采样"**；对象空间缓存复用 Lumen 的分帧重着 + 时序摊销；用 Ptex 式 per-face tile 回避 seam。

### 13.2 架构：着色缓存 = 可写 VT 页（复用地基，加法）

数据流（在现 `texture_streaming` 上加一个"着色"生产者，消费端不变）：

```
几何/可见性(vis-buffer) ──► texel feedback: 哪些对象空间页本帧可见 + 期望 mip
        │                         (复用 feedback.rs 的 PageDemand / mip_error / importance)
        ▼
着色需求表(ShadeResidencyTable)  ──► 预算调度(复用 scheduler.rs 贪心按字节/按 texel 预算)
        │  标记 Requested/Resident/Stale            │ 产出 ShadePlan{要着色的页, 要驱逐的页}
        ▼                                           ▼
对象空间着色 pass(compute) ──写入──► 着色缓存图集(复用 atlas.rs 的 tile 布局/slot_placement)
        │  对每个 texel: 解包 OpenPBR über(§7) → 按四线响应(§3.5)算"可缓存项"
        ▼
主 pass 采样着色缓存(复用 indirection.rs 二分页表) + 屏幕空间补"视角相关项" → 合成
```

复用点（全部已存在，详见 §8）：
- **页表** `indirection.rs::GpuPageTable`：二分可查；其 `w1` 低 8 位现为保留位 → 存**着色缓存 generation / valid 标志**（§13.7），零额外带宽。
- **残留状态机** `residency.rs::PageResidency`（`NotResident→Requested→Resident`）→ 扩一个 **`Stale`** 语义（已驻留但光照过期需重着），走同一优先级容器。
- **预算调度** `scheduler.rs::StreamingPlan` 贪心按字节预算 → 复用为**着色预算**（§13.4）。
- **图集** `atlas.rs::AtlasGeometry/slot_placement`：着色结果写进 tile；`COPY_BYTES_PER_ROW_ALIGNMENT`/block 对齐沿用；tile 边缘留 gutter（§13.6）。
- **保底 mip** `mip_tail.rs::mip_tail_covers`：保证粗 mip 常驻 → 缓存未命中时的降细节回退（§13.7）。
- **时序/驱逐保护** `streamer.rs` 的 retention decay + eviction protection window → 直接用于**缓存失效摊销**（§13.5）。

关键：**着色缓存不是新子系统的新 ABI，而是给 VT 页加一个"可写 + 带 generation"的生命周期**。über 解包与四线响应核心(§3.5/§7)原样搬进对象空间 compute pass，**着色数学不变**。

### 13.3 四线 × 分离求值矩阵（视角无关项缓存，视角相关项屏幕空间）

解耦的物理前提：**视角无关项**（不依赖 view/half-vector）可缓存复用；**视角相关项**依赖观察方向，缓存需存方向性表示，成本/误差高。

**架构约束（本次重构定调，取代上一版的 `banding_sensitivity` 按页特判）**：缓存层只存**传递无关的入射辐射量**——入射辐照度 / 低频辐射（需要高光时加方向性 SH-L1/L2），**不存"已经过响应/传递的结果"**。所有响应与传递都搬到**采样期**：PBR 在采样期全分辨率乘 albedo·评估 BRDF，NPR 在采样期对缓存辐照度施加 ramp，自定义走契约。推论：

- **缓存对四线完全中立**：同一表面的缓存内容 PBR / NPR / 混合区域**共用同一份辐照度**，不按线分通道存不同"响应"（§13.3 旧表的分线只是"采样期各取所需"，不是缓存内容分叉）。
- **精度是单一规范编码的全局决定**，不存在按线 / 按材质的精度策略——`banding_sensitivity` 这类标志被删除（见 §13.8）。
- **albedo/材质细节不被摊销**：只有昂贵的光照被缓存按 mip 摊销，albedo/法线等高频细节仍在采样期全分辨率取——解耦着色的本意（省光照、不省细节）。

四条线（§3.5）在采样期各自的切分：

| 线 | 可缓存（对象空间，复用） | 屏幕空间（视角相关，不缓存或方向性缓存） | 高效/高质解法 |
|---|---|---|---|
| **PBR** | 入射辐照度 / GI、SSS 漫透低频项、emission（**albedo/base 不进缓存，采样期全分辨率乘入**） | GGX specular / coat 高光 / anisotropy（依赖 H 向量） | 缓存辐照度直接复用，采样期乘 albedo（反照率细节不糊）；高光项用**缓存辐照度 + 屏幕空间 BRDF 评估**（split-sum 式）；需缓存高光时存**方向性辐照度（SH-L1/L2 或主方向 lobe）** |
| **NPR** | toon ramp 的**光照量化输入**（N·L 带状化前的辐照度）、面阴影 SDF 采样、材质色 | 描边（屏幕空间几何）、风格化高光锐边（视角相关） | ramp 的"输入辐照度"视角无关 → 缓存；**量化曲线在采样期施加**（曲线是逐像素便宜的 LUT 查表，缓存连续量更稳、避免档边闪烁）；描边永远屏幕空间 |
| **混合** | 逐区域取 PBR/NPR 的可缓存项，按 illumination 遮罩 | 两线各自的视角相关项 | 缓存层按 `illumination` 分通道存；合成期按遮罩选择；**per-tile 分类(§7.3)把 Lit/Stylized 分波前**，缓存命中与否不破坏一致性 |
| **自定义** | 由自定义程序**声明**的"视角无关输出"（契约，§13.9） | 声明为视角相关的输出 | 自定义 WESL 片段实现采样期 `respond_custom()`（必）+ 可选 `shade_cacheable_aux()`（声明额外视角无关量）；默认只用共享入射辐照度，未实现则安全退化为屏幕空间 |

统一原则——**split evaluation + 传递无关缓存**：对象空间只缓存"传递无关的入射辐射量" + 时序复用；响应（albedo·BRDF / ramp / 自定义）与视角相关项全在采样期算；合成阶段相加。四线共享**同一份缓存内容**与同一 über 解包，差异只在采样期的响应函数——缓存层对四线中立，这是 PBR/NPR 精度能完全一致的根（§13.8）。

### 13.4 着色率管理（风险①的高性能解法）

问题：纹理空间若盲目全量重着，比屏幕空间更贵。解法 = **把 VT 的"按需 + 分级 + 预算"直接当着色率控制器**：

- **按需**：只着色 vis-buffer 反馈为**本帧可见**的对象空间页（复用 `feedback.rs` 的可见性→`PageDemand`）。不可见页不着色。
- **分级（天然 LOD）**：期望 mip 由屏幕投影面积定（`feedback.rs::mip_error`/`clamped_importance`）。远处/密集几何落到粗 mip → **一个粗 texel 覆盖多像素，overshading 被消除**（这是性能天花板的来源）。
- **预算（硬上限）**：每帧着色 texel 数由 `scheduler.rs` 贪心按**着色预算**（texel/字节/compute 时间）切；超预算的页**本帧不重着，继续采样上一帧缓存**（时序复用兜底）。预算是 pass 级 uniform，不进材质 permutation（与 §7.3 一致）。
- **优先级**：复用 `feedback.rs::SemanticWeights` 思路，给"着色需求"打分（屏幕重要度 × mip 紧迫度 × 失效紧迫度），高分先着。
- **VRS 协同**：缓存命中区域主 pass 几乎只做"采样 + 视角相关补项"，可叠加硬件 VRS 进一步降屏幕着色率。

效果：**着色总量 ≈ O(可见对象空间 texel at mip)**，与屏幕分辨率/过绘制解耦；密集几何、远景、高几何复杂场景收益最大。

### 13.5 缓存失效与时序摊销（风险②的高性能解法）

问题：光照/材质/动态变化时缓存过期；全量重着=退回逐帧全着。解法 = **细粒度失效 + 分帧摊销 + 时序累积**：

- **generation / epoch 失效**：每着色缓存页存一个 `shade_generation`（放 `indirection.rs` 页表 `w1` 保留低 8 位 + 页元数据）。失效源各自维护 epoch：
  - 光照 epoch（光源移动/强度变、天光变）；动态材质 epoch（材质参数动画）；几何 epoch（蒙皮/形变）。
  - 页的 `shade_generation` < 影响它的最大 epoch → 标 **`Stale`**（复用残留状态机新增态）。
- **局部失效，不全局**：只有**受影响的页**进 `Stale`（如只有被移动光源包围盒覆盖的对象空间页），其余命中不动。静态光 + 静态几何的页**永不重着**（最大收益场景）。
- **分帧重着（Lumen 式预算摊销）**：`Stale` 页进同一预算调度(§13.4)，按优先级**每帧只重着一部分**；重着前继续用旧缓存（短暂偏旧，视觉可接受）。重着紧迫度随 staleness 上升。
- **时序累积**：对象空间着色天然稳定（无屏幕空间抖动），重着结果与旧值做 **temporal blend**，高频光照变化下用历史平滑；配合主 pass TAA。
- **保守重着频率分层**：diffuse/GI 低频 → 低频重着；emission/快速动画 → 高频或直接走屏幕空间。频率是**每语义/每 archetype 可配**（§13.9）。

效果：**重着成本 ∝ 实际变化量**，而非场景规模；静态区零成本，动态区被预算 + 时序摊平，无档位跳变。

### 13.6 UV seam / 参数化（风险③的高质解法）

问题：纹理空间着色在 UV 接缝处出现裂缝/漏光。解法 = **Ptex 式 per-face tile + gutter，从参数化根上消除 seam**：

- **per-face / per-chart 缓存 tile**：着色缓存按**几何面片（或 chart）**分配 tile，而非共享全局 UV。借 Pixar **Ptex**：无显式全局 UV、无跨 chart seam；相邻面片的滤波通过**邻接表**在采样期跨 tile 取邻。
- **gutter（边缘外扩）**：每 tile 着色时多算 1–2 圈边缘 texel（`atlas.rs` tile 已有 block 对齐，预留 gutter 带），双线性/各向异性采样不越界、不漏缝。
- **mip 一致的 seam 处理**：粗 mip 的 gutter 同步生成，避免远处接缝重现。
- **退路**：对不便 per-face 参数化的资产（如导入的单 UV 网格），用**接缝感知滤波**（采样期检测 chart 边界，钳到同 chart）作为次优解。

效果：接缝在**生成期**解决（gutter + 邻接），采样期零特判或仅轻量钳制，视觉无缝。

### 13.7 缓存一致性与回退（风险④的高性能解法）

问题：页在"请求着色→着色完成→被驱逐"过程中，主 pass 可能采到未就绪/已失效页。解法 = **就绪位 + 粗 mip 保底 + 无锁双代**：

- **就绪/有效位**：页表 `w1` 保留位存 `valid` + `shade_generation`；主 pass 采样时若页 `!valid`，**回退到已驻留的更粗 mip**（`mip_tail.rs::mip_tail_covers` 保证粗 mip 常驻）→ 降细节而非裂帧，与 §6.1 距离 LOD 同手段。
- **着色与采样无锁解耦**：着色 pass 写入**新 slot**，完成后原子更新页表指向新 slot（generation 自增），主 pass 永远读到**一致的某一代**（旧代或新代，不读半写）。驱逐沿用 `scheduler.rs` 的 evict 顺序。
- **首触延迟（disocclusion）**：相机骤转/新物体入场导致大量页未着色 → 本帧这些像素**走屏幕空间 über 回退**（即普通 §7 路径），后台补着色，下帧起命中缓存。保证不卡顿、不黑块。
- **一致性校验**：对象空间缓存值可与屏幕空间直算做**抽样对拍**（复用 §10 双线框架的思路，CPU 金标准可在对象空间复算），作为 CI 回归。

效果：主 pass 永远有可用数据（新代/旧代/粗 mip/屏幕回退四级兜底），**无裂缝、无卡顿、无非确定黑块**。

### 13.8 内存与压缩

- **单一规范编码（四线共用一份，天然无带状）**：缓存只存传递无关的入射辐照度（§13.3），用一种**感知均匀的 HDR 编码**（log-luminance / PQ 式曲线，色度分量线性）——在整个 HDR 范围给出**均匀相对精度**。因为编码与下游响应无关、且感知均匀，任何采样期传递（PBR tonemap、NPR ramp、自定义）都**不会放大带状**：量化步长感知上恒定，无"陡区被压出台阶"的问题。于是**不再需要按线/按页的精度策略**（删除上一版的 `banding_sensitivity` 与 NPR 排除分支）——精度是这一个编码选择的全局结果，PBR/NPR 从源头一致。
- **有损压缩对四线一致（只看重着频率，不看线）**：GPU BC6H 作用在上述感知编码上，误差预算对所有消费者一致；是否压**只由页的重着频率/价值**决定（低频久驻页压、高频重着页不压），**与 PBR/NPR 无关**。因为缓存存的是感知均匀辐照度而非"响应结果"，BC6H 的误差同样不被 ramp 放大。
- **ramp 台阶的锐度不依赖缓存位深（架构保证）**：NPR 量化的"硬边"由 ramp LUT 的阈值定义，并在**采样期用解码辐照度的屏幕空间梯度做解析抗锯齿**（阈值交越处按 `fwidth` 平滑），因此台阶边缘清晰度由屏幕导数决定、**与缓存 bit 深度解耦**。这把"NPR 要高精度缓存"的需求从根上移除——缓存只要感知均匀，锐边在采样期生成。
- **预算即显存墙**：图集容量 = `pool.rs::PhysicalPagePool` capacity，按平台配置；弱平台调小预算 → 更多走屏幕空间回退（优雅降级，非崩溃）。
- **方向性缓存的代价**：若对高光启用 SH-L1 方向性缓存，字节成本 ×4（4 系数）→ 仅对高价值材质/archetype 开启（§13.9）。

### 13.9 易用性与作者契约（默认安全，opt-in 加速）

- **默认屏幕空间，archetype opt-in 解耦**：解耦着色是**性能优化开关**，不改作者心智模型。美术仍写 OpenPBR Surface(§4)；是否走对象空间缓存由 **archetype/材质标志**声明（如 `Skin/Foliage/静态建筑` 开，`快速动画/强视角相关` 关）。
- **自定义线的采样期响应契约**（§3.5 自定义线落点 / §15 附录A）：缓存只存传递无关入射辐照度，自定义程序只负责响应
  - `respond_custom(surface, inc: IncomingRadiance, aux, view) -> Color`（采样期施加响应，读共享入射辐照度，必实现）
  - `shade_cacheable_aux(surface, light_env) -> CustomAux`（**可选**：声明额外视角无关量进缓存，分通道不污染共享辐照度）
  - 不实现 `respond_custom` → 安全退化为屏幕空间 über（§7）；不实现 `shade_cacheable_aux` → 只用共享入射辐照度。
- **可观测性**：调试视图叠加"缓存命中率 / Stale 页 / 重着预算占用 / mip 分布"，美术/工程按场景调预算与重着频率。
- **零作者 seam 负担**：per-face tile + gutter(§13.6) 由系统处理，美术不手动排 UV gutter。

### 13.10 落地阶段与门禁（细化 §12 的 P3-a/b/c）

| 阶段 | 内容 | 门禁 |
|---|---|---|
| P3-a 地基 | 着色缓存 = 可写 VT 页：扩 `residency` 加 `Stale`、页表 `w1` 存 generation/valid、着色 compute pass 写 `atlas` tile | 命中/未命中回退正确；无裂帧（粗 mip 兜底）CI |
| P3-b 视角无关项 | 先缓存 diffuse/SSS/emission/GI 辐照度（PBR+NPR 可缓存项），高光走屏幕空间 split-sum | 与纯屏幕空间抽样对拍误差阈值；furnace 不回归 |
| P3-c 失效摊销 | generation/epoch 局部失效 + 分帧预算重着 + 时序累积 | 动态光场景帧时稳定、无档跳；staleness 上限 CI |
| P3-d seam/一致性 | per-face tile + gutter + 无锁双代 + disocclusion 屏幕回退 | seam 视觉零裂；骤转无黑块 CI |
| P3-e 四线/自定义契约 | NPR ramp 输入缓存、混合逐区域、自定义两段式契约 | 四线各自对拍 + 命中率/预算可观测 |
| P3-f 内存/压缩（可选） | BC6H 低频页压缩、方向性 SH 高光缓存（高价值材质） | 显存预算内；方向性误差阈值 |

**破坏性说明**：屏幕空间 über 路径（§7）**保留为缓存未命中/视角相关的回退路径**，不是并行栈，不维护旧着色 API；对象空间缓存是其上的加法 pass。弱平台关闭开关即纯 §7 路径。

### 13.11 风险 → 高性能/高质解决方案总表

| 风险 | 朴素做法的坑 | 本设计的解法 | 复用地基 |
|---|---|---|---|
| 纹理空间着色率失控 | 全量重着比屏幕还贵 | 按需(可见)+分级(mip 消 overshading)+预算(硬上限)+VRS 协同 | `feedback.rs`/`scheduler.rs` |
| 缓存失效（光照/动态变化） | 全量重着退回逐帧全着 | generation/epoch 局部失效 + 分帧预算摊销 + 时序累积 + 频率分层 | `streamer.rs` decay/保护窗 |
| UV seam 裂缝/漏光 | 采样期特判昂贵且不彻底 | Ptex 式 per-face tile + gutter 生成期消缝 + 邻接采样 | `atlas.rs` tile/对齐 |
| 缓存一致性（半写/未就绪） | 裂帧/黑块/非确定 | valid 位 + 无锁双代 generation + 粗 mip 保底 + disocclusion 屏幕回退 | `indirection.rs` w1/`mip_tail.rs` |
| 视角相关项不可缓存 | 强缓存高光→拖影/错误 | split evaluation：视角无关缓存 + 视角相关屏幕空间；需要时方向性 SH 缓存 | §3.5 四线 + §7 über |
| 显存压力 | 图集爆显存 | HDR 紧凑格式 + 低频页 BC6H + 预算即墙 + 弱平台回退屏幕空间 | `pool.rs` capacity |
| 作者复杂度上升 | 美术要懂缓存/UV gutter | 默认屏幕空间、archetype opt-in、自定义两段式契约、seam 系统托管 | §4 archetype/§3.5 |

效果小结：**性能**上消除 overshading + 按 mip/变化量摊销，密集/远景/静态光场景拿到天花板；**效果**上 split evaluation + 方向性缓存保高光正确、Ptex gutter 消缝、时序累积抗抖；**易用**上默认安全、opt-in 加速、seam 托管。整条路**复用 `texture_streaming` 全部地基**，是加法 pass，不引入并行着色栈，不保留旧着色 API。

---

## 14. 风险与回退

| 风险 | 缓解 |
|---|---|
| **permutation 爆炸**（无全局硬顶的主要代价） | 内容哈希去重 + archetype 聚类 + 按需异步编译 + 全瓣 über 回退 PSO 兜底（§7.4）；CI 预热缓存 |
| 冷 permutation 首帧回退略贵 | über 回退正确但稍慢，后台编译完切换；可对热门 archetype 预编译 |
| 多散射补偿与 WESL twin 对不齐 | furnace CI + 查表字节对拍 |
| VT 缺页抖动 | 回退到已驻留高 mip（降细节不卡顿）；复用现有 residency 优先级 |
| 采样缩放在低档出现噪点 | 配合 TAA/时序累积；缩放曲线可调 |
| 变长 über 解包边界错误 | CPU/WESL twin 字节级对拍覆盖变长路径 |
| 破坏性 ABI 迁移面大 | 一次性迁移 + WESL twin 对拍兜底,分阶段 P0→P2 |
| 解耦着色缓存失效/一致性（P3） | 作为独立子系统推进；首期只缓存视角无关项，高光仍屏幕空间；光照/动态变化触发页重算 + 时序复用兜底 |

---

## 15. 附录 A：关键数据结构草案

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
// === 解耦着色数据结构（§13 落点，P3；全部复用 texture_streaming 地基，加法）===

// residency.rs —— 残留状态机新增 Stale 态（已驻留但光照/材质/几何过期需重着）
// 原: NotResident -> Requested -> Resident
pub enum PageResidency {
    NotResident,
    Requested,
    Resident,
    Stale,          // 新增：缓存命中仍可采样，但进重着队列（§13.5）
}

// indirection.rs —— 页表项 w1 的保留低 8 位用作着色缓存元数据（零额外带宽，§13.2/13.7）
// w1 = (mip << 24) | (layer << 8) | shade_meta8
//   shade_meta8: bit0   = valid（本页着色结果已就绪，可采样）
//                bit1   = generation_parity（无锁双代，避免半写撕裂）
//                bit2..7= shade_generation 低位（粗判 Stale；细判在页元数据）
pub const SHADE_VALID_BIT: u32 = 1 << 0;
pub const SHADE_GEN_PARITY_BIT: u32 = 1 << 1;

// 着色缓存页元数据（与 VT 页一一对应，不新建 ABI，仅给页加生命周期）
pub struct ShadePageMeta {
    pub shade_generation: u32,   // 本页着色时各失效源 epoch 的快照（§13.5）
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
    pub to_shade: Vec<VtPageRef>,    // 超预算的页本帧不重着，继续采样上一帧缓存（时序复用兜底，§13.4）
    pub to_evict: Vec<VtPageRef>,
    pub shade_budget_texels: u32,    // pass 级 uniform
}
```

```rust
// === 传递无关着色缓存：规范编码（v9/v10 定调，§13.3/§13.8）===
// 缓存只存"视角无关的入射辐射量"，不存任何已响应/已传递的结果。
// 四线（PBR/NPR/混合/自定义）共用同一份内容；精度是这一个编码的全局结果，无按线/按页策略。

// 单一规范编码：感知均匀 HDR（log-luminance / PQ 式亮度曲线，色度分量线性）。
// 在整个 HDR 范围给出均匀相对精度 → 任何采样期传递都不放大带状（§13.8）。
pub enum ShadeCacheEncoding {
    // 基础（必存，四线中立）：聚合入射辐照度（漫反射/GI/SSS 低频/emission）
    IrradiancePerceptualHdr,            // R=感知编码亮度, G/B=色度(线性)
    // 可选：方向性入射辐射，支持采样期高光 BRDF 评估（高价值材质才开，§13.9）
    DirectionalSh { bands: u8 },        // 1 或 2 阶 SH；字节 ×(系数数)
}

// 着色缓存页负载：共享入射辐照度常驻；方向性 / 自定义 aux 分通道，不混进共享内容
pub struct ShadeCachePagePayload {
    pub incoming: ShadeCacheEncoding,   // 四线中立，必存（唯一被按 mip 摊销的量）
    pub directional_words: u32,         // 可选高光方向性 SH 的字长，0 = 不存
    pub custom_aux_offset: u32,         // 自定义线声明的额外视角无关量（可选，分通道，不污染共享）
    pub custom_aux_words: u32,
    pub shade_generation: u32,          // 失效源 epoch 快照（§13.5），粗判 Stale
    // 注意：albedo/法线等高频细节【不在此】—— 采样期全分辨率取（省光照，不省细节，§13.3）
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
//   NPR : 对 inc.irradiance 施 ramp LUT；阈值交越处用 fwidth 做解析抗锯齿（锐边与缓存位深解耦，§13.8）
//   混合: 按 illumination 遮罩在采样期选 PBR/NPR 响应（缓存不分叉）
//   自定义: 走下方 custom 契约

// 自定义线（可选额外缓存视角无关量；最简情形只实现 respond_custom，直接用共享入射辐照度）
fn shade_cacheable_aux(surface: Surface, light_env: LightEnv) -> CustomAux;        // 可选：额外视角无关量进缓存
fn respond_custom(surface: Surface, inc: IncomingRadiance, aux: CustomAux, view: View) -> vec4<f32>;
//   不实现 cacheable_aux → 只用共享入射辐照度；不实现 respond_custom → 安全退化为屏幕空间 über（§7）

// 四线共享：主 pass 命中缓存 → respond() 合成；未命中/首触 → §7 über 屏幕空间回退（同一执行路径，§7.5；非并行栈）
```


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
