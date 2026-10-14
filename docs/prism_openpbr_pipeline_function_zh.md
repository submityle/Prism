# Prism — OpenPBR Surface 参数在各渲染管线中的功能对照（PBR / NPR / 混合 / 自定义）

> **一句话**：OpenPBR Surface 的参数是 **closure 轴**（表面闭合长什么样）的输入；PBR 与 NPR 是 **illumination 轴**（同一份光照数据怎么被解读）的选择。所以"某个 OpenPBR 参数在 PBR / NPR 里干什么"，答案不在参数表里，而在**同一次解包之后、由哪条响应函数去读它**——本文就是把这张"谁读谁、读了当什么用"的账逐参数对平。

---

## 0. 本文回答什么 / 不回答什么

**回答**：每个 OpenPBR Surface 参数在 PBR（Lit）线、NPR（Stylized）线、混合线、Unlit / 自定义线上**分别承担什么功能**——它是喂给哪个瓣/哪一项、影响直接光还是间接光、在 NPR 下是被重新解读、被忽略、还是有等效的美术替身。

**不回答**（已由既有文档覆盖，本文不重复）：

| 问题 | 去哪看 |
|---|---|
| 规范参数叫什么、什么类型、Prism 字段叫什么、默认值差多少、是否已对齐 | `docs/prism_material_pipeline_refactor_zh.md` §4.4（逐参数对账表，基线 OpenPBR v1.1.1）+ §4.5（超集三条硬约束） |
| 材质 ABI / über 打包 / slab 闭包图 / PCM 声明式 schema | 同上 §3、§5、§17.5；字段真相源 `pkg/prism_material_schema/schema/surface.toml` |
| PBR / NPR / 混合前端各有哪些高级特性、对标谁 | `docs/prism_material_pipeline_design_zh.md` §13、§13.1、§14、§15 |
| 轴的定义、前端与 closure 的正交关系 | 同上 §3.1、§7、附录 A |

**本文基线**（全部可 grep）：

- **规范**：OpenPBR Surface **v1.1.1（2026-04-17，ASWF 维护、MaterialX 子项目，Autodesk + Adobe 联合开发）**，取自 `AcademySoftwareFoundation/OpenPBR` 的 `parametrization` 表。v1.0→v1.1→v1.1.1 **没有任何参数增删**，只有一处默认值变化（`thin_film_ior` 1.5→1.4，v1.1 #211）与若干澄清（#248 新增 "Flexibility of implementation"、#258 澄清薄壁与 subsurface 交互、#260 允许 `emission_color` > 1、#262 重写薄膜章节）。Prism 取 v1.1.1 口径。
- **⚠️ 先纠一个常见混淆**：**`specular_ior_level` 不是 OpenPBR 任何版本的参数**。它是 Blender 4.x Principled BSDF 的叫法。OpenPBR 里"介电 F0 的 level 控制"由 **`specular_weight` 调制 `specular_ior`** 实现（规范公式见 §3.3），其硬范围是 [0, ∞) 而非 [0,1]。
- **CPU 金标准**：`pkg/prism_render_shading/src/`（`resolve.rs` / `stylized.rs` / `clearcoat.rs` / `subsurface.rs` / `face_shadow.rs` / `outline.rs` / `quality.rs` / `openpbr_reference.rs`）。
- **GPU 孪生**：`pkg/prism_render_scene/src/shaders/`（`shading_resolve.wesl` / `brdf.wesl` / `clearcoat.wesl` / `material_unpack.wesl` / `outline.wesl` / `lobe_budget.wesl`）。
- **作者/导入层**：`pkg/prism_render_material_authoring/src/import.rs`。
- **口径**：只描述**已落地代码**的行为；未落地的明确标"计划/降级/未消费"，不把设计意图写成现状。

---

## 1. 坐标系：为什么"同一个参数、不同管线"是成立的

### 1.1 三级流水：解包一次，分派一次，响应分家

```text
MaterialX / glTF / Bevy StandardMaterial
        │  import.rs: adapt_openpbr_surface → import_openpbr_surface（只认 OpenPBR 词汇表，其余报错）
        ▼
über 参数块（surface.toml: core words + 各 lobe words）    ← 参数的「布局与契约」真相源
        │  material_unpack.wesl / surface.rs：解包成同一个 SurfaceSample
        ▼
classification（CPU classification.rs:17 MaterialShadingClass ／ GPU switch params.shading_class）
        │  ← 分岔点在这里：illumination 轴 + 主导 closure 投影成 9 个 shade class
        ├─ Lit  · Principled/Subsurface/ClearCoat/Cloth/Hair/Water ─→ principled_direct（BRDF 积分）
        ├─ Stylized (Npr) ─────────────────────────────────────────→ stylized_direct（ramp 量化）
        ├─ Unlit ──────────────────────────────────────────────────→ 只取 emissive/albedo
        └─ Custom ─────────────────────────────────────────────────→ 项目注入的 WESL closure
```

`Illumination` 轴定义在 `pkg/prism_render_material/src/axis.rs:26-37`（`Lit` / `Stylized` / `Unlit` / `Custom`，注释明写 Stylized "Consumes the same lighting data as `Lit`"）。9 个 shade class 见 `pkg/prism_render_shading/src/classification.rs:17-26`；GPU 侧是 `shading_resolve.wesl` 里 9 个 `SHADING_CLASS_*` 常量 + `switch params.shading_class`（`PRINCIPLED=0 … CUSTOM=8`）。

**关键**：解包（`material_unpack.wesl`）**只有一份**。OpenPBR 参数不分"PBR 版 / NPR 版"——它们是同一份 `SurfaceSample`，差别只在下面哪个响应函数去读它。

### 1.2 一句话结论（本文最重要的一条）

> **今天，OpenPBR 的绝大多数瓣只在 PBR（Lit）线被真正消费；NPR 直接光响应只读三个东西：`base_color`、`specular_roughness`、`emissive`。** NPR 的"风格"不住在 OpenPBR 参数里，而住在 `StylizedParams`（`stylized.rs:71-100`）——那是一组**全局/风格层**参数（band 数、wrap、ramp 软度、阴影阈值、风格高光、rim），不是 per-material 材质参数。

这不是规范要求——OpenPBR 规范**不含 NPR**，也**没有**风格化章节；但规范 v1.1.1 的 "Flexibility of implementation" 章节**明文允许**把参数改作它用（§4 引言引了原文）。所以准确的说法是：**"风格住在哪里"由 Prism 决定，"可以不住在 OpenPBR 参数里"则是规范许可的**。第 4 章给"未消费清单 + 如果要接的建议映射"，第 11 章给缺口与路线图。

### 1.3 五条线（"PBR、NPR 等"展开）

| 线 | 判据（code-truth） | 参数功能定位 |
|---|---|---|
| **PBR / Lit（物理线）** | `illumination = Lit` + 主导 closure → `Principled`（或 `Subsurface`/`ClearCoat`/`Cloth`/`Hair`/`Water`） | 参数 = **物理 BSDF 的系数**；受规范约束（§4.5 合规门禁） |
| **NPR / Stylized（风格线）** | `illumination = Stylized` → `Npr` class（`resolve.rs:1391`、`classification.rs`） | 参数 = **美术可读的调制量**；只有 3 个直连，其余未消费或走专属通道 |
| **混合线** | 同帧不同物体各自取 `illumination` 值；分桶在 `classify_material_header` / `ShadingWorkPlan` | 参数的**两种功能在同帧并存**（不同像素走不同 arm），共享光源/阴影/GI |
| **Unlit 线** | `illumination = Unlit`（`resolve.rs:1271`、`:1522`） | 只有 albedo/emission 有意义，光照项一律不读 |
| **自定义线** | `illumination = Custom` → 投影 `Custom` class | 参数只完成解包与分类；响应由项目 hook 决定，"功能"由项目自己定义 |

---

## 2. 总矩阵：参数 × 管线功能（速查）

**图例**：`✅` 该线响应直接消费 ／ `◐` 经上游（lobe、阴影、合成、alpha）间接影响 ／ `○` 该线当前不消费。

### 2.1 Base + Specular

| 规范参数 | PBR（Lit）线的功能 | NPR（Stylized）线的功能 | 混合线 | 视依赖 | 缓存通道 |
|---|---|---|---|---|---|
| `base_weight` | 漫反射基底权重，与 `base_metalness` 一起把基底在介电漫反射/金属镜面之间分配 | ○（不读） | 同 PBR | invariant | `LitDiffuse` |
| `base_color` | 漫反射 albedo / 金属 F0；三通道吸收与色调 | ✅ **cel ramp 的乘性色调**（`stylized.rs:206`） | 两边都读，读法不同 | invariant | `LitDiffuse` / **`StylizedDiffuse`** |
| `base_metalness` | 决定基底是介电还是导体：F0 由 `base_color` 取代 `specular_color`，漫反射份额趋零 | ○（NPR 金属通常走 MatCap/整材质切换） | — | invariant | `LitDiffuse` |
| `base_diffuse_roughness` | EON（Fujii Oren-Nayar + 解析多次散射补偿）漫反射粗糙；Prism 丢掉规范 `ρ_ms` 的 `1/π` 勘误并烘 17×9 反照率表 | ○ | — | invariant | `LitDiffuse` |
| `specular_weight` | 通过 **modulated-IOR** 缩放介电法向入射反射率（`ξ_s=0 ⇒ 界面整体消失`，§3.3）；金属侧作整体乘子 | ○（NPR 高光强度住在 `StylizedParams::specular_intensity`） | — | dependent | 屏幕空间 |
| `specular_color` | 介电：**只**染"自上首次反射"的 Fresnel（规范承认非物理但有用）；金属：F82-tint 掠射边染色（§3.3） | ○（**同名不同物**：NPR 用的是 `StylizedParams::specular_color`） | — | dependent | 屏幕空间 |
| `specular_roughness` | GGX α（`α = r²`）＋各向异性伸长；决定高光锐度与能量分布 | ✅ **风格高光的锐度**：`gloss = 1−r`，`exponent = mix(MIN, MAX, gloss²)`（`stylized.rs:213-216`） | 两边都读 | dependent | 屏幕空间 |
| `specular_ior` | 介电 F0 = `((η−1)/(η+1))²`（`dielectric_f0`，`brdf.wesl:217`）；η<1 走 TIR 分支 | ○ | — | dependent | 屏幕空间 |
| `specular_roughness_anisotropy` | 用规范 `openpbr-anisotropy-formula` 把 α 伸长成 `α_t`/`α_b`（`brdf.wesl:878-886`、`lighting.rs:899-917`；base 与 coat 共用） | ○ | — | dependent | 屏幕空间 |

### 2.2 Transmission + Subsurface

| 规范参数 | PBR（Lit）线的功能 | NPR（Stylized）线的功能 | 视依赖 | 缓存通道 |
|---|---|---|---|---|
| `transmission_weight` | 折射透射瓣权重；决定表面从"不透明介电"转向"玻璃/水" | ○（美术常用"背后透光/描边透光"假替） | dependent | 屏幕空间 |
| `transmission_color` | 透射色（Beer-Lambert 色调） | ○ | invariant（色调本身） | — |
| `transmission_depth` | 达到 `transmission_color` 的参考路径长度；实时约定 **路径 = 1 场景单位**，`T = slab_color^(1/depth)`，`depth==0` 走规范非物理 tint 分支 | ○ | invariant | 可烘焙进贴图 |
| `transmission_scatter` | 体内散射色（半透明浑浊） | ○ | invariant | — |
| `transmission_scatter_anisotropy` | 体散射相函数各向异性（前/后向） | ○ | invariant | — |
| `transmission_dispersion_scale` | 色散强度（RGB 折射率分离） | ○（NPR 常用手绘色边代替） | dependent | 屏幕空间 |
| `transmission_dispersion_abbe_number` | 色散由 Abbe 数推导（默认 20），比 scale 更物理 | ○ | dependent | 屏幕空间 |
| `subsurface_weight` | 次表面瓣权重，把漫反射层替换为参与介质的归一化扩散 | ○（NPR 皮肤多用 ramp 假 SSS） | invariant | `Subsurface` |
| `subsurface_color` | 体反照率（进入扩散系数） | ○ | invariant | `Subsurface` |
| `subsurface_radius` | 平均自由程尺度（实时瓣作相对值） | ○ | invariant | `Subsurface` |
| `subsurface_radius_scale` | 逐通道半径比（默认 `(1.0, 0.5, 0.25)` 的红扩散更远） | ○ | invariant | `Subsurface` |
| `subsurface_scatter_anisotropy` | 介质内散射相函数各向异性（`g`） | ○ | invariant | `Subsurface` |

### 2.3 Coat + Fuzz + Thin-film + Emission + Geometry

| 规范参数 | PBR（Lit）线的功能 | NPR（Stylized）线的功能 | 视依赖 |
|---|---|---|---|
| `coat_weight` | 清漆层权重：叠一层介电镜面，并**吸收其下的基底** | ○（"二阶高光"通常用第二套 stylized 高光手摆） | dependent |
| `coat_color` | 清漆介质的法向透射率平方 $T^2$，**染的是其下的基底**（不是 coat 自己的高光） | ○ | dependent |
| `coat_roughness` | 清漆层自身 GGX 锐度（默认 0，即镜面清漆）；**并会把基底磨粗** `r'_B = lerp(r_B, (r_B⁴+2r_C⁴)^{1/4}, C)` | ○（"二阶高光"通常用第二套 stylized 高光手摆） | dependent |
| `coat_roughness_anisotropy` | 清漆层各向异性（与 base 同一公式，`clearcoat.wesl:317-326`） | ○ | dependent |
| `coat_ior` | 清漆层 F0（默认 1.6） | ○ | dependent |
| `coat_darkening` | 规范的**能量暗化**：`lerp(1, Δ, C·δ)`，`Δ=(1−K)/(1−E_b·K)`（`openpbr_reference.rs` 的 `modulated_darkening_factor` / `general_darkening_formula`） | ○（NPR 用调色/ramp 直接压暗） | dependent |
| `fuzz_weight` | 绒毛瓣权重：掠射角的柔光边（velvet/sheen） | ○（NPR 用 rim 近似） | dependent |
| `fuzz_color` | 绒毛散射色 | ○ | dependent |
| `fuzz_roughness` | 绒毛纤维锐度/朝向集中度 | ○ | dependent |
| `thin_film_weight` | 薄膜虹彩瓣权重（KHR 瓣，slot 6） | ○ | dependent |
| `thin_film_thickness` | 膜厚（µm→nm 换算后进 Airy BSDF） | ○ | dependent |
| `thin_film_ior` | 膜折射率（决定虹彩相位） | ○ | dependent |
| `emission_luminance` | 自发光辐亮度（nits；`emissive = emission_color × luminance`，`resolve.rs`） | ✅ 直接加到风格化输出（`stylized.rs:242`），不参与 ramp | invariant |
| `emission_color` | 自发光有色调 | ✅ 同上 | invariant |
| `geometry_opacity` | 覆盖率：乘进 alpha，决定混合/遮罩与透射路径 | ◐（经 alpha/遮罩影响描边与排序，不参与 ramp） | invariant |
| `geometry_thin_walled` | **切换整套 slab 语义的模式位**（非质量开关）：透射退化为薄片、双面、**`subsurface_radius*` 被忽略**（v1.1.1 #258） | ◐ | invariant |

### 2.4 规范之外的 Prism 扩展区 + shade-time 输入

| 名称 | 归属 | PBR 线 | NPR 线 | 备注 |
|---|---|---|---|---|
| `npr_face`（`npr_face_softness`） | Prism 私有瓣（`surface.toml:885`，uid `studio.prism.face_shadow`，slot 16） | ○ | ✅ **专属通道**：SDF 面部阴影的翻转软度 | `face_shadow.rs` + `brdf.wesl:1212-1245` + `shading_resolve.wesl:2560` |
| `anisotropy_rotation` | Prism 扩展 | ✅ 切线各向异性旋转 | ○ | `resolve.rs:313-314` |
| `ambient_occlusion` | Prism 扩展 | ✅ 间接/环境项遮蔽 | ○ | `resolve.rs` / `environment/mod.rs` |
| `normal_scale` | Prism 扩展 | ✅ 法线强度 | ◐ 影响 ramp 输入法线 | `resolve.rs` / `texture_sample.rs` |
| `alpha_cutoff` | Prism 扩展 | ✅ 遮罩阈值 | ◐ | — |
| `glint_weight/roughness/density/bin_half_width` | Prism 私有瓣（slot 17） | ✅ 离散 NDF 闪光 | ○ | `energy_policy = coupled`（等效重整形，不改能量） |
| `measured_brdf` | Prism 私有瓣（slot 18） | ✅ 实测 BRDF 查表混入 | ○ | provider 同名（`surface.toml:1218`） |
| `geometry_normal` / `geometry_tangent` / `geometry_coat_normal` / `geometry_coat_tangent` | **shade-time 输入，不是材质参数** | ✅ 构造 `ShadingFrame` / coat frame | ◐ 法线影响 ramp，切线多不读 | 见 §3.10 |

### 2.5 一眼看懂的四条规律

1. **PBR 线：参数的"功能"= 规范定义的物理角色**，几乎 1:1 落在某个 BRDF 项（GGX / Fresnel / V-Smith / Oren-Nayar / Airy / 扩散）。
2. **NPR 线：参数的"功能"= 它作为"美术可读量"被重新解释**。今天只有 `base_color`（→ ramp 色调）、`specular_roughness`（→ 高光锐度）、`emission`（→ 直接叠加）三种解释落地。
3. **视依赖（`view` 元数据）决定了能不能缓存**：invariant 的 OpenPBR 瓣（base / subsurface / emission）才是解耦着色（texel-space cache）的候选，dependent 的（specular / coat / transmission / fuzz / thin_film / glint）必须留在屏幕空间（§7）。
4. **合规门禁只约束 PBR 线**：纯 OpenPBR 子集必须与参考实现对拍（§4.5）。NPR 线是**规范之外的自由区**（规范不含 NPR），而"参数可被改作它用"又由规范的 "Flexibility of implementation"（#248）**明文许可**（§4 引言）——因此"某参数在 NPR 下应该干什么"是**产品决策**，不是规范义务（§8）。

---

## 3. 逐参数详解

每条格式：**规范语义** → **PBR 功能** → **NPR 功能** → **视依赖 / 降级** → **消费点**。

### 3.1 元参数（core words，`surface.toml:30` `words = 20`）

core 区放的是"所有瓣都要用的公共槽位"（`surface.toml:32-215`）：`base_color`(33)、`base_metalness`(46)、`specular_roughness`(57)、`specular_weight`(69)、`ambient_occlusion`(82)、`normal_scale`(93)、`alpha_cutoff`(104)、`base_weight`(115)、`base_diffuse_roughness`(129)、`specular_color`(143)、`specular_ior`(156)、`geometry_thin_walled`(170)、`geometry_opacity`(189)。

功能上它们是**über 的"常驻系数"**，无论材质带哪些瓣都必须解包出来——这解释了为什么 NPR 直连能只读 core 里的三项而不需要任何 lobe。

### 3.2 Base 组

- **`base_weight`**（float，默认 1）
  - PBR：漫反射基底的权重。与 `base_metalness` 共同决定基底在"介电漫反射 + 介电镜面"和"金属镜面（无漫反射）"之间的分配；也是涂层/次表面接管基底的接口。
  - NPR：不读。NPR 的"底光强度"由 ramp 与全局风格参数决定。
  - 视依赖：invariant → 可进 `ShadeChannel::LitDiffuse` / `StylizedDiffuse` 页。
  - 消费点：`resolve.rs`、`clearcoat.rs`、`lighting.rs`、`cloth_advanced.rs`、`dynamic.rs`。
- **`base_color`**（color3，Prism 用 vec4 且 w 为 alpha，默认 `(0.8,0.8,0.8)`）
  - PBR：**双重身份**——介电时它是漫反射 albedo，同时也是规范里的"观测反射色 $C$"（white-furnace 归一化的锚点：`E_multi-scatter = (1 − E_spec)·C`）；金属时它是 `F0 = base_weight × base_color`（**由 `base_weight` 缩放**，不是别的参数）。规范硬义务：任意上述混合再加 coat/fuzz 后，`C = 1` 时必须通过 **white furnace test**（方向反照率 ≡ 1）。
  - NPR：**直连 ramp**。`color = base_color × illuminance × ramp × shadow`（`stylized.rs:206`）——即 `base_color` 在 NPR 下就是"二分/三分色块的基色"，不做任何能量归一化。同时它是皮肤/头发在 NPR 下唯一的逐材质色调来源。
  - 视依赖：invariant；NPR 用 **`ShadeChannel::StylizedDiffuse`**，PBR 用 `LitDiffuse`——同一参数在缓存层就分成两个通道（`texel_cache/key.rs:5-24`）。
  - 消费点：`resolve.rs`、`stylized.rs:206`、`clearcoat.rs`、`cloth.rs`、`water.rs`、`hair*.rs`、`subsurface.rs`、`gi/area_light/*`。
- **`base_metalness`**（float，默认 0）
  - PBR：导体/介电开关。规范口径是**两个 slab 的统计混合** $M$（不是"渐变过渡"），中间值近似金属斑块/锈蚀/掉漆，Substance 的 metalness 贴图接的就是它。metalness 越高，F0 越由 `base_color` 接管、漫反射份额趋零。
  - **参数集收缩（易漏）**：规范明确金属只使用 6 个参数——`base_weight`、`base_color`、`specular_weight`、`specular_color`、`specular_roughness`、`specular_roughness_anisotropy`。因此 `base_metalness → 1` 时，§2 矩阵里 transmission / subsurface / fuzz 等参数的功能**是被顶掉的**，不是"也参与了"。而介电基底走 `specular_ior` 路径，金属**不用** `specular_ior`。
  - NPR：不读。二次元金属通常靠 MatCap 或整材质替换，而不是靠 metalness 连续过渡。
- **`base_diffuse_roughness`**（float，默认 0）
  - PBR：漫反射层的微面斜率 σ。规范指定用 **EON**（Fujii Oren-Nayar + 解析多次散射补偿 `EON_comp`），并反解 albedo 使观测色恰为 `base_color`；`σ = 0` **精确**退化为 Lambert。转写见 `openpbr_reference.rs:62-159`（`fon_a`/`fon_b`/`fon_unit_brdf`/`eon_directional_albedo`/`eon_average_albedo`/`eon_rho_ms`）。
  - **Prism 的两处自主决策（code-truth）**：(a) 规范里 `ρ_ms` 的 LaTeX 写作 `(ρ²/π)·…`，但那个 `1/π` 与同节自身的不变量"ρ→1 ⇒ E_diffuse→1"**矛盾**（照字面实现白炉只能到 **0.70**）且量纲不自洽（`f_comp` 已带 `1/π`，`ρ_ms` 必须无量纲）——**Prism 丢掉 `1/π`**，推导写在 `eon_rho_ms` 的 doc 注释里；(b) 规范把解析 albedo 推给 [Portsmouth2024]，Prism 改为**数值求积烘成 17×9 表**（`eon_albedo_table.rs`，与 `brdf.wesl` 内联镜像由 `eon_albedo_table_mirrors_the_wesl` 测试守漂移），并由 `eon_lobe_passes_the_white_furnace`（96×96 求积）验证 ρ=1 时方向反照率达 1。
  - NPR：不读。这是 §4.2 里唯一"**无建议**"的一项，理由是 ramp 量化后该信息**确实被丢弃**了，不是不愿接。

### 3.3 Specular 组

- **`specular_weight`**（float，默认 1；规范硬范围 **[0, ∞)**、Norm [0,1]；**Prism schema 钳成 `0..=1`**）
  - PBR：**它不是直接乘在 F0 上的开关，而是通过调制 IOR 来缩放介电法向入射反射率**。规范公式（Prism 逐字转写在 `openpbr_reference.rs:175` 的 `modulated_ior`）：
    ```
    F_s  = |(1 − η_s) / (1 + η_s)|²
    ε    = sgn(η_s − 1) · sqrt(ξ_s · F_s)        ξ_s = specular_weight
    η'_s = (1 + ε) / (1 − ε)                     之后一律用 η'_s 代入 F(μ, η'_s)
    约束：ξ_s ≤ 1 / F_s                          （否则 ε 翻转，反射反而消失）
    ```
    因此 `ξ_s = 0 ⇒ η' = 1 ⇒ F ≡ 0`，界面**整体消失**；`ξ_s = 1 ⇒ η' = η_s`（恒等）。透射/折射必须**与 η' 保持一致**，否则会出现"只缩放了高光、折射仍按旧 IOR"的劈裂。
  - **Prism 的合规教训（code-truth）**：实时瓣曾用 Schlick 近似 Fresnel 并把 `specular_weight` 当作 F0 的线性乘子，结果 `specular_weight == 0` 时掠射角仍趋全反射（"幽灵边缘"），**违反规范"ξ_s = 0 ⇒ 界面消失、F ≡ 0"，最大偏差 0.904**。现改为上面的 `modulated_ior` + Walter-2007 精确介质 Fresnel。`openpbr_reference.rs` 的 `dielectric_fresnel_has_the_expected_shape` 测试注释原文即"精确曲线**不是** Schlick 多项式——正是门禁要抓的偏差"。
  - 金属侧**不用**这个调制：`F0 = base_weight × base_color`，掠射边由 `specular_color` 在 `μ̄ = 1/7` 处染色（F82-tint，见下条），最后整体乘 `specular_weight`。
  - NPR：不读。等价的美术旋钮是 `StylizedParams::specular_intensity`（0 = 关高光）。
  - ⚠️ **Prism 偏差**：规范硬范围 [0, ∞)，Prism schema 写 `0..=1`（`surface.toml:69`，`valid` 在 `:77`）——艺术夸张 `> 1` 会被合法性谓词拒绝。
- **`specular_color`**（color3，默认白）
  - PBR（介电）：**只**给"来自上方的首次反射"的 Fresnel 染色；**透射、以及来自下方的反射不受影响**。规范自己承认这在技术上非物理（原文："technically unphysical if altered from the default white color … but can be useful in practice to artificially tint the specular highlight"）——即这是**规范许可的美术旋钮**，不是滥用。
  - PBR（金属）：**F82-tint**（Kutz 2021）。`F0 = base_weight × base_color`，`specular_color` 定义掠射边（约 82°、`μ̄ = 1/7`）的染色，再整体乘 `specular_weight`；默认值下**精确退化为 Schlick，且边缘不会亮于 Schlick**（`F_82(μ) = F_Schlick(μ) − [μ(1−μ)⁶ / (μ̄(1−μ̄)⁶)]·(F_Schlick(μ̄) − F(μ̄))`）。转写在 `openpbr_reference.rs:262` 的 `base_metal_fresnel_rgb`。
  - NPR：`SurfaceSample::specular_color` 不读；NPR 高光色来自同名的 `StylizedParams::specular_color`。**这是最容易踩的坑**（改材质里的 `specular_color` 在 NPR 下没反应）——但注意规范已明确它就是"染色高光"用途，所以 §4.2 把它接进 NPR 高光是**顺着规范意图**。
- **`specular_roughness`**（float，默认 0.3）
  - PBR：GGX 粗糙度 α；直接决定高光瓣宽度、间接光探针/环境 BRDF 近似（`env_brdf_approx`）的锐度。
  - NPR：**直连**，但语义变成"高光锐度"：
    ```rust
    // stylized.rs:213-216
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let gloss = 1.0 - roughness;
    let exponent = mix(MIN_SPECULAR_EXPONENT, MAX_SPECULAR_EXPONENT, gloss * gloss);
    ```
    然后 `pow(N·H, exponent)` 过 `smoothstep(specular_threshold ± softness)` 成"天使环"式色块（`stylized.rs:218-227`）。**所以粗糙度在 NPR 下不是能量参数，是形状参数**——它把高光斑调大调小，而强度完全由风格参数决定。
  - 降级：`MIN_ROUGHNESS` 夹紧保证 `gloss²` 不退化成奇异指数。
- **`specular_ior`**（float，默认 1.5；规范硬范围 (0, ∞)、Norm [1,3]；Prism 钳成 `1..=3`；`MIN_IOR = 1.0 + 1e-3`）
  - PBR：介电基底内部介质 $V_\mathrm{dielectric}$ 的折射率 $n_b$。它是**介电 F0 的唯一来源**（`dielectric_f0`），也是**色散**（Cauchy / Abbe，§3.4）里 $n_d$ 的定义者，还是 coat 界面相对 IOR 与透射折射的基准；η < 1 走全内反射分支。它 **同时**驱动透射瓣的折射——见 §3.4 的 `index_of_refraction`。
  - NPR：不读。若要做"边缘更亮/更暗"，用 `StylizedParams::rim_power`。
- **`specular_roughness_anisotropy`**（float [0,1]；Prism 字段解码后短名 `anisotropy`）
  - PBR：按规范 `openpbr-anisotropy-formula` 伸长 GGX：
    ```
    α_t = r² · sqrt(2 / (1 + (1−a)²))
    α_b = (1−a) · α_t
    ```
    （`brdf.wesl:878-886`、`lighting.rs:899-917`、`clearcoat.wesl:317-326`；base 与 coat 共用同一映射，`a == 0` 逐位退化为各向同性）。方向来自 `geometry_tangent` / `geometry_coat_tangent`。
  - NPR：不读。NPR 的"发丝天使环"是**把切线解耦出来手摆**的（design §14.1 第 3 条），与这个参数无关——这是有意的：物理各向异性在高光被阈值化成色块后不再有多余信息量。
  - ⚠️ **同名不同义警告**：`pkg/prism_render_shading/src/gi/**` 里大量出现的 `anisotropy` 是**体散射相函数各向异性**（Henyey-Greenstein 的 g），不是本参数。检索时勿混。

### 3.4 Transmission 组

- **`transmission_weight`**（float，默认 0）
  - PBR：把介电基底从"不透明镜面 + 漫反射"切到折射透射（玻璃/水/冰）；同时改变阴影透射、排序（`blend: Transmissive`）与 IBL 折射路径。
  - NPR：不读。NPR 里"透明"通常由 `blend` 轴（Transparent/Additive）+ 美术贴图承担；卡通玻璃靠"背后透光"手绘。
- **`transmission_color`**（color3）
  - PBR：吸收色。与 `transmission_depth` 组成 Beer-Lambert：`T = color^(path/depth)`。
  - NPR：不读；若需要，用 ramp 或顶点色调色。
- **`transmission_depth`**（float，schema 标识符 `transmission_depth`，解码后 Prism 字段 `thickness`）
  - PBR：达到 `transmission_color` 的参考路径长度。实时瓣无场景单位路径，故**显式约定路径 = 1 场景单位**（`T = slab_color^(1/depth)`）；`depth == 0` 走规范的非物理 tint 分支。
  - NPR：不读。
  - 消费点：`lighting.rs`（slab 吸收助手）、`openpbr_reference.rs`。
- **`index_of_refraction`**（float，默认 1.5，Prism 传输瓣 slot 2，`surface.toml:734`）
  - PBR：透射瓣自己的 IOR——决定折射偏折与 TIR。⚠️ **它不是 OpenPBR 的参数名**：规范里透射用的就是基底的 `specular_ior`（$n_b$）。Prism 在传输瓣里**再存一份**，作者层也接受 `index_of_refraction` 作为输入名（`import.rs:281`），所以两条路都能写进去——**两者不一致时以透射瓣的值为准**。这是"同名/异名参数分裂"的一个具体风险点，接玻璃材质时要留意。
  - NPR：不读。
- **`transmission_scatter`**（color3，Prism 用 vec4，w 未用）
  - PBR：体内散射色（浑浊玻璃/玉石）；与 `subsurface_*` 共用扩散类数学但作用在透射路径上。
  - NPR：不读。
- **`transmission_scatter_anisotropy`**（float [-1,1]）
  - PBR：体散射相函数前后向（`g`）。影响折射光在介质内的方向分布与多重散射近似。
  - NPR：不读。
- **`transmission_dispersion_scale`**（float）
  - PBR：色散强度——按 RGB 分离 IOR 制造色边。实时代价高，通常按质量档降级为单 IOR。
  - NPR：不读（色边是美术手绘的）。
- **`transmission_dispersion_abbe_number`**（float，默认 20）
  - PBR：色散改由 Abbe 数推导，比 `dispersion_scale` 更贴近真实玻璃目录口径。两者共同决定分离量。
  - NPR：不读。

### 3.5 Subsurface 组

- **`subsurface_weight`**（float，默认 0）
  - PBR：把漫反射层替换为参与介质（归一化扩散 / 随机游走近似），同时把 shade class 投影成 `Subsurface`（`lit_class_from_closures`），走 `shade_subsurface` arm 与 `LobeSampleBudget.subsurface_taps` 预算。
  - NPR：不读。经典二次元皮肤走 **ramp 假 SSS**（design §13.1 第 11 条：更省更可控）；需要时也能吃真 SSS。
  - 消费点：`resolve.rs`（唯一消费点——因为真计算在 GI/扩散模块里由预算驱动）。
- **`subsurface_color`**（color3，默认 `(0.8,0.8,0.8)`）
  - PBR：体反照率，进扩散系数（`σ_s'` 相关推导）。
  - NPR：不读。
  - 消费点：`subsurface.rs`、`lighting.rs`、`gi/translucency/transmission.rs`、`gi/cloth/*`。
- **`subsurface_radius`**（float，作相对值）
  - PBR：平均自由程尺度——决定光在皮下扩散多远（红比蓝远）。
  - NPR：不读。
- **`subsurface_radius_scale`**（color3，默认 `(1.0, 0.5, 0.25)`）
  - PBR：逐通道半径比。规范默认即"红光扩散更远"，是皮肤偏红的原因。
  - NPR：不读。
- **`subsurface_scatter_anisotropy`**（float [-1,1]）
  - PBR：介质内相函数各向异性（`g`），影响扩散剖面的前向性。
  - NPR：不读。

### 3.6 Coat 组

- **`coat_weight`**（float，默认 0）
  - PBR：叠一层清漆介电镜面，并**按 `coat_darkening` 吸收其下的基底能量**；同时投影成 `ClearCoat` shade class（`clearcoat.rs` 为 CPU 金标准）。
  - NPR：不读。二次元的"二阶高光"是用第二组手摆 stylized 高光实现的（可完全脱离物理层）。
- **`coat_color`**（color3，默认白）
  - PBR：**规范定义它是"涂层介质在法向入射透射率的平方"**（$T_\mathrm{coat}^2$），作用是**给其下的基底染色**——原文明确 "the specular reflection from the coat itself is **not** tinted"。掠射角处透射率按 $T_\mathrm{coat}^{1/\mu^t_i + 1/\mu^t_o}$ 增强（$\mu^t_i = \sqrt{1 − (1−\mu_i^2)/\eta_c^2}$）。
  - ⚠️ **Prism 曾把方向做反**（染了 coat 高光而非基底），本轮已按规范纠正。这是"参数功能"最容易读反的一条：**`coat_color` 不是"清漆自身的高光颜色"，而是"透过清漆看到的基底颜色"**。
  - NPR：不读。
- **`coat_roughness`**（float，默认 0）
  - PBR：清漆层自身锐度。默认 0 = 镜面清漆，破坏它就失去"湿表面"读感。
  - **跨参数功能（极易漏）**：coat 还会**把基底磨粗**——$r'_B = \mathrm{lerp}(r_B, \min(1, (r_B^4 + 2 r_C^4)^{1/4}), C)$（`coat_roughened_base_roughness`，`openpbr_reference.rs:398`）。所以 `coat_roughness` 不只是"第二层高光"，它会**改变基底本身的 GGX α**。NPR 若要接"二阶高光"，这条耦合必须一起考虑，否则基底与高光锐度会互相打架。
  - NPR：不读（NPR 高光锐度只有 `specular_roughness` 一路）。
- **`coat_roughness_anisotropy`**（float [0,1]）
  - PBR：清漆层各向异性，与 base 共用 `openpbr-anisotropy-formula`（`clearcoat.wesl:326`）。
  - NPR：不读。
- **`coat_ior`**（float，默认 1.6；规范硬范围 (0,∞)、Norm [1,3]）
  - PBR：清漆层 F0 与内部反射系数 `K` 的输入——`K_r = 1 − (1 − E_F(η_c))/η_c²`（粗糙/漫反射基底）、`K_s = F(ω_o, η_c)`（光滑基底），`K = lerp(K_s, K_r, r_b)`，`r_b = lerp(r_d, r_m, M)`，`r_d = lerp(1, r, ξ_s·F_s)`、`r_m = r`；`E_F` 用规范闭式拟合 `ln((10893η − 1438.2) / (−774.4η² + 10212η + 1))`（`openpbr_reference.rs:309/338/360`）。
  - **接口耦合（易漏）**：基底/涂层界面的相对 IOR 是 `η_s = lerp(n_b/n_a, n_b/n_c, C)`。当 `n_c > n_b`（coat 比基底更密）时，按此式会出现**伪 TIR（掠射亮环）**，规范建议反转为 `η_cb = n_c/n_b`（Kutz 2021；`coat_specular_ior_ratio`，`openpbr_reference.rs:456`）。即 **`coat_ior > specular_ior` 时这个参数的功能要换一套解释**，否则车漆边缘会多出一圈假亮环。
  - NPR：不读。
- **`coat_darkening`**（float，默认 1）
  - PBR：**能量守恒的关键旋钮**。规范口径是 `lerp(1, Δ, C·δ)`，其中 `Δ = (1 − K)/(1 − E_b·K)`、`K = lerp(K_s, K_r, r_b)`（`modulated_darkening_factor` / `general_darkening_formula`，`openpbr_reference.rs:370/385`）。行为上：清漆会明显压暗其下的漫反射基底（示例 `E_b=0.4, η_c=1.6, r_b≈0.98` 时 `Δ≈0.49`）。`δ = 1` 是纯物理暗化；`δ = 0` 时规范要求给出**恰好反补基底 albedo 的 boost** `B(δ) = lerp(B_0, 1, δ)`——**两端都是"合法材质"**，不是"开了/关了某个 bug"。
  - 规范另有一条**明确假设**：coat "填充基底微孔从而改变基底 Fresnel" 这一物理机制**假定不发生**（coat 只通过上面的公式影响能量），实现不必模拟它。
  - NPR：不读（NPR 直接调色压暗）。
  - ⚠️ 这条曾经是启发式 `1 − C·δ·F(μ_o)`，本轮按规范重写；`coat_adds_glint_at_peak` 测试因此改为用 `δ=0` 隔离镜面高光。
  - 消费点：`clearcoat.rs`（CPU 金标准，1097 行）、`clearcoat.wesl`。

### 3.7 Fuzz 组

- **`fuzz_weight`**（float，默认 0）
  - PBR：绒毛瓣（**v1.1 起启用 Zeltner sheen**，#217）——掠射角的柔光边缘，用于布绒/天鹅绒/毛毯。该层**内部 index-matched、无 Fresnel**，规范给定**固定单位光学厚度**，所以 `fuzz_weight` 是纯覆盖率，不是"厚度"。
  - NPR：不读。NPR 的等效观感由 `rim_*`（`rim_intensity` / `rim_power` / `rim_color`）承担，但那是**视角驱动的边光**，不是绒毛散射。
- **`fuzz_color`**（color3，默认白）
  - PBR：**多次散射后**绒毛反射的染色 albedo。**v1.1 起"可变暗也可变亮"**（#207 允许 fuzz 变暗）——旧口径只允许提亮，接材质时注意这一点。
  - NPR：不读。
- **`fuzz_roughness`**（float，默认 0.5）
  - PBR：纤维朝向集中度——低值→强纤维感（高 sheen 织物），高值→微片近似球形（灰尘感）。
  - NPR：不读。

### 3.8 Thin-film 组

- **`thin_film_weight` / `thin_film_thickness` / `thin_film_ior`**（float×3；`thickness` 单位 **µm**、默认 0.5；`ior` 默认 **1.4**，v1.0 曾是 1.5）
  - PBR：Airy 薄膜干涉——肥皂泡/油膜/蝴蝶翅的虹彩。`weight` 在有膜/无膜 BSDF 间混合；`thickness`（与 IOR 共同）决定条纹间距与色相；`ior` 决定相移与对比度。推荐实现 Belcour & Barla (2017)。膜**不吸收**，只重新分配反射/透射，因此不破坏能量守恒。
  - **规范指定的接入位置 vs Prism 的做法**：规范说膜厚远小于微面尺度且膜面光滑，所以应**直接并入 metal / dielectric-base 微面 BSDF 的 Fresnel 项**，**不新增 slab**；且介电基底时**透射也要产生彩色条纹**，金属基底需把 Fresnel 反推为等效复 IOR（Gulbrandsen 2014）。**Prism 把它做成了 ABI 瓣**（KHR slot 6，`surface.toml:828`）——这是**有意的架构偏差**：换取瓣级裁剪 + per-material 开关，代价是与规范的 slab 位置不同（§11 缺口 8）。
  - v1.1 改 `thin_film_ior` 默认 1.5→1.4 的**动机值得记**：让薄膜在 `specular_ior=1.5`、`coat_ior=1.6` 的默认情形下**可见**而非"隐身"（这是三个版本间唯一的默认值差异）。
  - NPR：三项全不读。二次元的"渐变色高光"是贴图/顶点色手摆的，不用薄膜相位。
  - 消费点：`resolve.rs`（唯一的接入口）、`brdf.wesl`（Airy）。

### 3.9 Emission 组

- **`emission_luminance`**（float，**单位 cd/m²（nits）**，硬范围 [0, ∞)、Norm [0,1000]，默认 0）+ **`emission_color`**（color3，默认白；**v1.1.1 起硬范围为 [0, ∞)³、允许 > 1**，以便接 HDR 贴图——注意 MaterialX 1.1.1 节点的 UI 上界仍写作 `uimax=1,1,1`，**以规范表为准**）
  - PBR：`emissive = emission_color × luminance`（`emission_color = (1,1,1)` 且无 coat/fuzz 时 `luminance` 就是该面的亮度）。零辐亮度的中性面即 `luminance = 0`。参与 bloom/泛光、GI 弹射（作为光源）、曝光。Bevy 桥接按 `luminance = max(R,G,B)`、`color = e / luminance` 因式分解，**乘积精确保留**（`bevy_bridge.rs`）。
  - NPR：**直连**——`add(color, surface.emissive)`（`stylized.rs:242`），即自发光**不经过 ramp**，是唯一"跨越风格化量化"的项（发光体就该是平的亮）。仍参与 bloom 与风格化后处理。
  - 视依赖：invariant → `ShadeChannel::Emission` 页。
  - 回归测试：`resolved_pixel_scales_emission_by_luminance`（曾暴露 CPU 桥接漏乘 luminance、与 WESL twin 不一致的 bug）。

### 3.10 Geometry 组

- **`geometry_opacity`**（float [0,1]，默认 1）
  - PBR：**整个 surface 相对环境介质的 presence weight（线性 alpha）**，乘进 alpha；决定 `blend` 轴的遮罩/透明路径、OIT 权重、透射阴影。规范给出透射阴影的建议公式 `T_pbr = lerp(1, w_dielectric·T_dielectric, α)`、`w_dielectric = 1 − M`——即 **opacity 同时调"有没有这个面"和"阴影透多少"**，这是它与普通 alpha 测试的区别。规范另注明：**非薄壁模式下 α < 1 严格来说不物理**（除非整片移除），只有薄壁模式才给 α 明确物理意义。
  - NPR：**间接**——影响描边的 alpha 处理与排序，但不进 ramp。NPR 的"实心 vs 线稿"更多由 `alpha_cutoff` 与描边通道决定。
  - 消费点：`resolve.rs`、`lighting.rs`、`cloth_advanced.rs`、`dynamic.rs`。
- **`geometry_thin_walled`**（boolean 默认 false，Prism 用 f32 0/1 打包；解码后 Prism 字段 `thin_walled`）
  - PBR：**它不是"渲染质量开关"，而是切换整套 slab 语义的模式位**（规范 thin-walled 节）：假设结构沿基底镜像、基底薄到视为无厚度 2D 薄片，两面外观相同。最简近似 = 忽略背面 coat/fuzz、总把入射光翻转为从上进入。后果逐项：
    - metal 与 glossy-diffuse slab 保持但**双面**；
    - translucent-base 退化为**极薄、纯吸收、不散射**的介电薄片，`transmission_color` 直接给出法向入射透射率（渲染玻璃窗的廉价方案）；
    - subsurface 退化为极薄致密散射薄片：`f^R_diffuse = ½·S(1−g)·f_+`、`f^T_diffuse = ½·S(1+g)·f_-`，`S = subsurface_color`、`g = subsurface_scatter_anisotropy`，且 `E_R + E_T = S ≤ 1`；漫透射 lobe 形状由 `base_diffuse_roughness` 控制；
    - **此模式下 `subsurface_radius` 与 `subsurface_radius_scale` 被完全忽略**（v1.1.1 #258 澄清）——这是"某参数在某模式下功能归零"的**规范明例**，正好印证本文"参数功能取决于管线模式"的口径。
  - NPR：间接。
- **`geometry_normal` / `geometry_tangent` / `geometry_coat_normal` / `geometry_coat_tangent`**（vector3，**shade-time 输入，不是材质参数**）
  - PBR：构造 `ShadingFrame`，是各向异性方向、法线贴图强度、coat 层朝向的载体。
  - NPR：**法线是 ramp 的输入**（`n_dot_l = dot(normal, light)`，`stylized.rs:195`）——所以"手编法线"在 NPR 下是**一等美术手段**：调法线就能调色块形状（Guilty Gear Xrd 式做法，design §14.1 第 5 条）。切线在 NPR 下基本不读。

---

## 4. NPR 专题：OpenPBR 参数被怎么"重新解读"

> **规范显式许可这件事（先破一个误解）**。本节不是"Prism 在钻规范空子"——v1.1.1 新增的 **"Flexibility of implementation"（#248）** 章节原话就是鼓励重解读：
>
> - "As a guideline, we suggest approaching those approximations conceptually like **different level of detail (LOD) variants of an asset**."（把近似视为资产的 LOD 变体）
> - **"A parameter that drastically modifies the appearance of a material should not be discarded just because the corresponding effect is not supported. As an example, in a renderer that doesn't have subsurface scattering, the subsurface color might be used as the albedo of a diffuse BRDF."**
> - **"In the most extreme hypothetical case, the model might even get reduced to as little as a single Lambert BRDF."**
>
> 即：**规范只定义目标外观，把 BSDF 实现与取舍完全交给实现者**；"参数被改作它用"是被明文祝福的路径。§4.2 的建议映射因此不是"非规范做法"，而是规范给出的近似方向——差别只在**由谁来定**：NPR 的取舍是 Prism 的产品决策，不是规范义务。
>
> 另外规范**没有** NPR / 风格化章节，也从不把参数称作 "hint"——`hint` 只出现在 MaterialX 侧属性（`transmission_weight` 带 `hint="transparency"`、`geometry_opacity` 带 `hint="opacity"`），属 MaterialX 语义而非规范核心。

### 4.1 当前直连的三个参数（code-truth）

| OpenPBR 参数 | 在 `evaluate_stylized_direct` 里的角色 | 代码 |
|---|---|---|
| `base_color` | cel ramp 的乘性色调（色块基色） | `stylized.rs:206` |
| `specular_roughness` | 风格高光的锐度（`exponent = mix(MIN, MAX, gloss²)`） | `stylized.rs:213-216` |
| `emission_color` × `emission_luminance` | 直接叠加，不参与量化 | `stylized.rs:242` |

除此之外，`stylized.rs:185-243` 只读 `ShadingFrame.{normal, view}`、`DirectLightSample.{direction, illuminance, visibility}` 和 `StylizedParams`。

**重解读的三条规则**（把物理量当美术量用）：

1. **色 → 色块**：`base_color` 不再参与能量归一化，直接当"没被光照衰减的基色"使用（`× illuminance × ramp × shadow`）。
2. **粗糙度 → 形状**：`specular_roughness` 从"能量分布参数"变成"高光斑大小"，强度被 `specular_intensity` 接管。所以**同一个材质在 PBR 下高光弱、在 NPR 下高光强是完全正常且可预期的**。
3. **自发光 → 直通**：唯一绕过 ramp 的项，保证发光体在风格化下仍然"发光"而不是被量化成色块。

### 4.2 未消费清单 + "如果要接"的建议映射

下表左半是**现状**（NPR 不读），右半是**建议的非规范映射**（路线图，未落地；写在这里是为了让美术/引擎双方有共同词汇，不代表已实现）。

| 参数 | NPR 现状 | 若接入的建议语义 | 优先级 |
|---|---|---|---|
| `base_metalness` | ○ | 二值化金属开关：>0.5 切 MatCap/环境反射近似，否则走 ramp | P2 |
| `specular_weight` | ○ | 映射成 `stylized.specular_intensity` 的逐材质乘子（让美术在材质里控高光有无） | **P1** |
| `specular_color` | ○ | 逐材质高光染色（与 `StylizedParams::specular_color` 相乘） | **P1** |
| `specular_ior` | ○ | 作为 rim 的权重（F0 越高轮廓越亮）或高光阈值偏移 | P2 |
| `specular_roughness_anisotropy` | ○ | 天使环的自动摆位（把切线各向异性转成高光带方向），替代纯手摆 | P2 |
| `transmission_*`（7 项） | ○ | 背后透光近似：`transmission_weight × transmission_color` 当作"背光透出色"，混入 ramp 的暗部 | P2 |
| `subsurface_*`（5 项） | ○ | ramp 假 SSS：用 `subsurface_radius_scale` 的逐通道比给暗部叠一层红光，`subsurface_weight` 控强度 | P2 |
| `coat_*`（6 项） | ○ | `coat_weight` → 第二组 stylized 高光的开关与强度；`coat_roughness` → 第二组锐度 | P3 |
| `fuzz_*`（3 项） | ○ | `fuzz_weight` → 描边内侧的柔光/边缘雾化；`fuzz_color` → 描边色 | P3 |
| `thin_film_*`（3 项） | ○ | 高光沿 N·V 的相位渐变（Spider-Verse 式色分离高光） | P3 |
| `base_diffuse_roughness` | ○ | 无建议（量化后无视觉信息） | — |

> 落地建议的形态：**不要把这些塞进 `StylizedParams`（那是全局的）**，而是给 NPR arm 增加"逐材质调制层"——即让 `evaluate_stylized_direct` 多读几个 `SurfaceSample` 字段。P1 两项（`specular_weight` / `specular_color`）改动最小、收益最直接。

### 4.3 NPR 专属通道（**不吃** OpenPBR 参数）

这些才是 NPR 的主力，全部挂在共享基底上，由 NPR arm 独占消费（design §4.2 的 C 类）：

| 通道 | 数据来源 | 消费点 | 与 OpenPBR 的关系 |
|---|---|---|---|
| **SDF 面部阴影**（`npr_face` 瓣） | 独立单通道 SDF 贴图 + 头部坐标系光方位角 | `face_shadow.rs`（`face_shadow_light_cosines` / `face_shadow_flip_u` / `evaluate_face_shadow`）；WESL 孪生 `brdf.wesl:1219/1238/1245`；参数注入 `shading_resolve.wesl:2560` | **唯一被扩进 über 的 NPR 参数**（`npr_face_softness`，`surface.toml:885`，slot 16，`energy_policy = none`，`view = invariant`）。是 Prism 私有瓣，规范之外 |
| **描边** | depth + normal + **vis-buffer material id 边界** | `outline.rs`（`outline_id_edge:114` / `outline_depth_edge:139` / `outline_normal_edge:176` / `evaluate_outline:218`）、`outline.wesl`（`@compute 8×8`） | **完全不读 OpenPBR 参数**——描边是几何/深度/ID 的产物，`material id 边界` 是虚拟几何基底"白送"的。这是"参数无关"的最典型例子 |
| **光分层打光** | `light_routing.rs` / `light_routing.wesl`（`MAX_LIGHTING_CHANNELS=8` / `MAX_LIGHT_LAYERS=4`，`channel_mask_affects`） | NPR 可给不同"光层"分别设色与强度 | 与参数正交：它管"哪盏灯影响哪层"，不管表面参数 |
| **风格化后处理** | `posterize.rs`(368)／`halftone.rs`(361)／`hatching.rs`(328)／`kuwahara.rs`(291)／`ordered_dither.rs`，各配 WESL 孪生 | NPR 前端独占 pass（`shading/{posterize,halftone,hatching,kuwahara,ordered_dither}/`） | 参数无关（全屏或按材质 ID 作用） |
| **风格化头发（天使环／tobikke）** | `stylized_hair.rs`(435)；WESL 孪生 `stylized_hair.wesl:143 stylized_hair_direct` | 分带 diffuse ramp + **两条错位的 Kajiya-Kay 各向异性高光**（靠发根侧紧白主高光、靠发梢侧更宽染色次高光，Scheuermann 双色调），各自 smoothstep 阈值化 + rim | 与物理毛发 `hair.rs`(358，Marschner R/TT/TRT) **是两条独立线**：NPR 发的"高光位置"是手摆的，OpenPBR 参数基本不参与 |
| **逐区域 illumination 遮罩** | `TextureSemantic::IlluminationMask`（`bevy_bridge.rs`，判别值 `ILLUMINATION_MASK_SEMANTIC`） | visibility raster 依据该遮罩**逐区域选 Lit / Stylized**，"不需要改材质 ABI"（语义原文） | **混合线的参数级机制**，见 §5 |
| **风格化高光／rim** | `StylizedParams` | `stylized.rs:208-240` | 见 4.4 |

**关于 `npr_face`：它只有 `softness` 一个字段**（`surface.toml:901`，0..=1，默认 0.1），没有 ramp 贴图／宽度／颜色——**ramp 走的是纹理语义（`TextureSemantic`），不在瓣字段里**。所以"per-material ramp 贴图"今天不在材质 ABI 内（这就是 §11 缺口 3 的具体形态）。

### 4.4 风格参数实际住在哪里：`StylizedParams`

`stylized.rs:71-100` 的 12 个字段，`Default` = `legacy_toon(4)`（四段色阶、纯 Lambert、硬边、无高光无 rim）：

`bands`（色阶数）、`wrap`（余弦包装，0.5 = half-Lambert）、`ramp_softness`（色阶软度）、`shadow_threshold` / `shadow_softness`（阴影阈值与过渡半宽）、`specular_intensity` / `specular_threshold` / `specular_softness` / `specular_color`（风格高光）、`rim_intensity` / `rim_power` / `rim_color`（边光）。

**架构含义**：
- OpenPBR 参数是 **per-material** 的；NPR 风格参数目前是 **per-view/per-style** 的。二者在同一材质上"分工"：材质给颜色与粗糙度，风格给量化与高光形状。
- 这也是 §4.2 P1 建议（把 `specular_weight`/`specular_color` 逐材质化）的动机：**今天美术无法在材质里决定"这个角色有没有高光"，只能整帧开关**。
- `evaluate_toon_direct`（`stylized.rs:251`）是历史 banded toon lobe 的精确包装（`StylizedParams::legacy_toon`），保留是为了 goldens 逐位稳定。

---

## 5. 混合线：同一参数的两种解读如何在同一帧共存

**机制**（design §7，已落地两侧）：

1. **CPU 分桶**：`classification.rs::classify_material_header` 把每个材质头投影到 9 值 `MaterialShadingClass`——`Water`/`Hair` 由 `render_class` 直绑；`Unlit`/`Npr`(=Stylized)/`Custom` 由 **`illumination` 轴**定；其余 `Lit` 由主导 closure 经 `lit_class_from_closures` 投影。`ShadingWorkPlan::build` 据此把像素分成 bin（同桶同 permutation，wavefront 一致）。
2. **GPU 路由**：`shading_resolve.wesl` 同一 resolve pass、同一套光/影/GI 输入，仅按 `params.shading_class` 切 arm（`shade_principled` / `shade_toon` / `shade_subsurface` / `shade_clearcoat` / `shade_cloth` / `shade_hair` / `shade_water` …）。
3. **逐区域遮罩（像素级混合，不只是逐物体）**：`TextureSemantic::IlluminationMask`（`bevy_bridge.rs`，判别值 `ILLUMINATION_MASK_SEMANTIC`）让 visibility raster **按区域选 Lit 或 Stylized**，且"不需要改材质 ABI"（语义原文）。于是同一材质的同一 draw 里，**脸走 Stylized、衣服走 Lit** 是可行的。**对参数的后果**：同一份 `SurfaceSample` 在同一帧被两条响应函数各读一遍，下面的"两种功能之和"就变成"**同一像素集合按遮罩分区**的两种功能"。

**因此"参数在混合线的功能"= 它在两条线上的功能之和**，只是作用在不同像素上：

| 情形 | 参数行为 | 视觉结果 |
|---|---|---|
| 写实角色（Lit）+ 卡通角色（Stylized）同框 | `base_color` 一边当 albedo、一边当色块基色；`specular_roughness` 一边当 GGX α、一边当高光斑大小 | **阴影方向 / 底光一致**（共享 VSM + GI），风格差异只出现在响应处，接缝不"贴纸" |
| 风格化 PBR 中间态（Fortnite/Valorant 式） | 同一材质开 `illumination = Stylized` + 少瓣 über，同时吃真 GI/Nanite | 参数既做物理基底又做 ramp 调制 |
| per-tile 路由 | 分桶已经按 class 把像素分 bin，混合的前端特化开销落在路由与 permutation 数上 | —— |

**B 类固有降级（必须认，不是 bug）**：NPR 对 GI **只收不发**（防能量爆炸）；RT 反射在 NPR 下退化为风格化 albedo 近似；RT 去噪对 NPR 需单独调锐利分段。这些是"数据喂风格化响应"的本质代价（design §4.2）。

### 5.1 混合的**数量**边界：能同时混几种（code-truth）

"混合"在 Prism 里指三件不同的事，各自容量差很多——**不要用同一个"支持几种混合"含糊过去**：

| 机制 | 粒度 | 能混几种 | 真相源 |
|---|---|---|---|
| 材质内多瓣叠加（closure 轴） | 单个材质 | **10** 个已注册瓣可同时激活 | `pkg/prism_render_material/src/surface.rs:414-432`（各位常量）、`:465 COUNT = 10` |
| 逐材质 illumination 轴 | 材质 | **4**（Lit / Stylized / Unlit / Custom） | `pkg/prism_render_material/src/axis.rs:26-37`、`:46 ALL: [Self; 4]` |
| 分类路由桶 | 全屏 work plan | **9** 类 | `pkg/prism_render_shading/src/classification.rs:7 MAX_SHADING_CLASSES = 9` |
| **逐区域 illumination 遮罩** | 像素区域（贴图） | **只有 2（Lit / Stylized），硬二元** | `classification.rs:716-722` |
| 解耦着色缓存通道 | 物体 texel 页 | **4**（LitDiffuse / StylizedDiffuse / Subsurface / Emission） | `pkg/prism_render_scene/src/shading/texel_cache/key.rs:5-15` |
| 灯光层 / 通道 | 灯光 cluster | **8 通道 × 4 层** | `pkg/prism_render_scene/src/shading/light_routing/abi.rs:31`（=8）、`:34`（=4） |

**逐区域遮罩的两重限制**（`classification.rs:716-722` 原文）：

```rust
let illumination = match header.illumination {
    0 | 1 => match region_illumination {
        0 | 1 => region_illumination,   // 只在 Lit(0) / Stylized(1) 之间切换
        _ => header.illumination,       // 写 2/3（Unlit/Custom）或 u32::MAX ⇒ 当作"无掩码"
    },
    _ => header.illumination,           // 材质级本身是 Unlit/Custom ⇒ 区域遮罩完全无效
};
```

1. **只有两档**：写 `2` / `3`（Unlit / Custom）或 `u32::MAX` 一律被当作"无区域掩码"、**静默回落材质级——不报错**。
2. **前置条件**：材质级 `illumination` 必须是 `Lit` / `Stylized`，否则区域遮罩**整体失效**。

所以"**逐区域混 3 种、4 种**"**当前不支持**。这个机制的设计意图也很明确：它不是通用多路混合，而是"**同一材质按遮罩在 Lit / Stylized 之间分区**"（典型用途：卡通脸 + 写实身体共用一份材质）。

**另一种读法明确不支持：同一像素叠加多种响应。** `illumination` 是**单值轴**，`classify_material_header` 返回单个 `MaterialShadingClass` —— 一个像素最终只走一条响应。因此 Prism 的"混合"是**空间分区**（不同像素走不同管线），**不是逐像素加权混合**。要在同一像素上兼得 PBR 与 NPR，只能用"风格化 PBR 中间态"（同一材质开 Stylized + 少瓣）近似，而不是两条响应加权求和。

**若要把逐区域扩到 3–4 档**：(a) 第一行改动很小——`classification.rs:717` 的 `0 | 1` 放宽到 `0..=3`；(b) 但 Unlit 会丢弃全部光照、Custom 需要 program 句柄，而 `custom_program` 是**材质级**字段、没有逐区域版本，所以这两档的区域语义要先定义；(c) visibility raster 侧写 `metadata.z` 的采样路径（`prism_render_scene/src/material/virtual_texture/feedback.rs:510` 认 `IlluminationMask` 语义）要能产出 `2` / `3`。

**材质内多瓣的采样预算**（与"能混几种"是两件事）：质量档只改采样数、**不裁剪瓣集**——`Low` 全采样 1 个瓣、`Medium` 2 个、`High`/`Ultra` 全部；超出预算走随机前缀采样 + TAA 累积（`pkg/prism_render_shading/src/lobe_sampling.rs:150-155`）。所以"10 个瓣同时开"在任何质量档下都能表达，只是低档下靠时间累积收敛。

---

## 6. Unlit 线与自定义线

- **Unlit（`illumination = Unlit`，`resolve.rs:1271`、`:1522`）**：光照项一律不读。OpenPBR 参数里只有 `base_color`（作为纯 albedo）、`emission_*`、`geometry_opacity` / `alpha_cutoff`（走混合）有功能；`specular_*` / `transmission_*` / `subsurface_*` / `coat_*` / `fuzz_*` / `thin_film_*` **全部无功能**。这是"参数存在但零功能"的合法终态，不是缺口。
- **自定义（`illumination = Custom`，投影 `Custom` class）**：项目注入自己的 WESL closure 与 pass。OpenPBR 参数的功能**由项目定义**——引擎只保证解包与分类正确（即"参数以正确的类型与值到达你的响应函数"）。这也意味着自定义线可以复用 `SurfaceSample` 的全部字段，包括 NPR 专属的 `npr_face_softness`。
- **closure 例外**：`Hair` 不是 über 的瓣，而是专用重 closure（Marschner/Chiang R/TT/TRT + dual scattering）；`Water` / `Cloth` 也有专项实现（`water.rs` / `cloth.rs` / `cloth_advanced.rs`）。这些 shade class **不是 OpenPBR Surface 参数**的消费方，但它们会**读取** core 区的 `base_color` / `specular_roughness` / `specular_ior` 等（见 §2 矩阵的消费点列）——所以对它们而言，"OpenPBR 参数的功能"是**部分复用**。

---

## 7. 解耦着色：哪些参数的"功能"可以跨视角缓存

材质 schema 的 `view` 元数据（`surface.toml` 每个字段都有，如 `glint_weight` 的 `view = "dependent"`、`npr_face_softness` 的 `view = "invariant"`）直接决定该参数能否进 **texel-space shading cache**：

`pkg/prism_render_scene/src/shading/texel_cache/key.rs:5-24`：

```rust
pub enum ShadeChannel {
    LitDiffuse = 0,      // 物理表面的漫反射/辐照度贡献
    StylizedDiffuse = 1, // 风格化表面的 ramp-ready 漫反射贡献   ← NPR 专用
    Subsurface = 2,      // 视无关次表面贡献
    Emission = 3,        // 自发光贡献
}
// ShadePageKey { object_id, material_id, material_epoch, face, mip, channel, page_x, page_y }
```

**这张表就是"同一参数、两条管线"在存储层的体现**：

| 参数组 | `view` | 缓存通道 | 含义 |
|---|---|---|---|
| `base_color` / `base_weight` / `base_metalness` / `base_diffuse_roughness` | invariant | `LitDiffuse` **或** `StylizedDiffuse` | 同一份 albedo，两条线各缓存一份"已量化/未量化"的结果 |
| `subsurface_*` / `emission_*` | invariant | `Subsurface` / `Emission` | 视无关，跨视角复用 |
| `specular_*` / `coat_*` / `fuzz_*` / `thin_film_*` / `transmission_*` / `glint_*` | dependent | 屏幕空间（不进 texel 缓存） | 依赖视线/半程向量，必须逐帧算 |
| `npr_face_softness` | invariant | —（走 SDF 通道） | 翻转软度与视角无关 |

`material_epoch` 进 key 是刻意的：**重新作者化的材质不能误用坐标碰巧相同的旧页**（`key.rs` 注释原文）。

---

## 8. 能量策略、合规门禁与"NPR 自由区"

### 8.1 能量策略（schema `energy_policy`）

schema 每个字段声明 `energy_policy ∈ {coupled, independent, none}`（§17.2 / §17.5.5 契约）：

- `coupled`：该参数**重整形**已有瓣的能量而不新增能量。例：`glint_weight` —— "把离散 NDF 混进已拥有能量预算的镜面瓣，估计器无偏，因此重整形高光而不改变其能量"（`surface.toml` 注释）。
- `independent`：引入独立能量预算（如 `emission_luminance` 的自发光）。
- `none`：不参与能量核算（几何/方向/开关类，如 `glint_roughness`、`npr_face_softness`）。

**PBR 线的功能受此约束**：`specular_weight > 1`、`coat_darkening` 的暗化、`base_weight` 的分配，都要在能量账上自洽。

### 8.2 合规门禁（§4.5 三条硬约束之一）

> 对"只用 OpenPBR 字段、不含任何 Prism 扩展位"的材质，Prism 的 BSDF 响应必须与 OpenPBR 参考实现对拍一致（**算术类 ≤ 1 ULP**）。

**这条门禁只约束 PBR 线**。它意味着：

- 在 PBR 线，**参数的"功能"没有自由度**——必须逐字符合规范（GGX/Fresnel/V-Smith/Oren-Nayar/Airy/扩散/暗化公式全部按规范），本文 §3 里 PBR 列的每条描述都是**规范义务**。
- 在 NPR 线，**参数的"功能"是 Prism 的自由定义**——OpenPBR 规范不含 NPR，且规范自己就许可"参数改作它用 / 模型退化为更简单的 BSDF"（§4 引言的 #248 原文）。因此"`specular_roughness` 在 NPR 下当高光锐度用"这种重解读**不可能违反合规**。这正是 §4.2 建议映射可以放心推进的规范依据。
- 这也是"超集"策略的技术红利：**关掉扩展，Prism 就是标准 OpenPBR；打开 illumination 轴，Prism 获得规范不覆盖的表现力**，两者互不干扰。

### 8.3 行为不在 schema 里

schema 只 own 布局与契约（字段名/类型/位区/打包偏移/特化子字段/消费步骤/默认值/合法性谓词/能量策略/视依赖性），**不表达行为数学**（§17.5.5）。所以本文 §3 的"功能"描述来源是**类型化 hooks + 落地代码**（`resolve.rs` / `stylized.rs` / `brdf.wesl` / `openpbr_reference.rs`），不是 schema 文件。

---

## 9. 能力协商与质量降级

### 9.1 能力需求

所有注册瓣的 `requires_caps = []`（如 `npr_face`，`surface.toml:885` 附近注释明写："Empty today: every registered lobe is pure ALU + the standard bindings"）。**含义**：OpenPBR 参数的功能**不因平台能力而消失**——没有任何参数需要 RT / mesh shader / bindless 才能生效。降级只发生在**质量档**（采样数）与**特性开关**（RT 反射等基底服务），不在参数可用性。

### 9.2 质量档：`ShadingQuality` 与 `LobeSampleBudget`

`quality.rs:43-48`：`Low=0 / Medium=1 / High=2（默认）/ Ultra=3`；`budget()` 给出各昂贵瓣的采样预算（如可分离 SSS 的屏幕空间模糊 taps、预积分漫反射 LUT、扩散剖面径向能量、多重散射 LUT），以纯 `u32` 结构体上传为 pass uniform，孪生 `shaders/lobe_budget.wesl`。

**参数级影响**：需要"采样"才能体现功能的参数（`subsurface_radius*`、`fuzz_roughness` 的纤维分布、`transmission_dispersion_*` 的色散分离）在 `Low` 档被压到最少 taps/退化为单 IOR；**不需要采样的参数（几乎整个 core 区）在任何档位都逐位生效**。

### 9.3 已经修掉的一致性坑

CPU resolve 桥接曾漏乘 `emission_luminance`，与 WESL twin（`shading_resolve.wesl` 里乘了）行为不一致——旧默认（luminance=1）掩盖了它，新默认（luminance=0）当场暴露；已按 WESL 口径修正并加回归测试 `resolved_pixel_scales_emission_by_luminance`。**这条对本文的意义**：参数"功能"的定义以 **WESL 为准**，CPU 是 golden 但必须与 twin 对拍。

---

## 10. code-truth 索引表

**先记三条容易搞错的数量与布局事实**（`surface.toml` / `namespace.rs`）：

- **导入词汇表 = 43 个 OpenPBR 拼写**：41 个逐字同名（`import.rs:255-296`）+ 2 个特殊重命名（`specular_roughness_anisotropy → anisotropy`、`transmission_depth → thickness`）；另有 11 个**仅 `standard_surface` 生效**的别名（`base`／`metalness`／`specular`／`specular_IOR`／`coat`／`subsurface`／`transmission`／`thin_walled` 等，`import.rs:302-318`）。⚠️ `index_of_refraction` 也在这 41 个里，但**它不是 OpenPBR 参数名**（§3.4）；⚠️ Standard Surface 的 emission **故意不做别名**（weight/intensity vs nits 单位语义不同，`import.rs:312-315` 注释）。
- **ABI 布局**：`schema_version = 1`、`abi_version = 6`（v5 引入扩展后的 clearcoat 块，v6 增加 8 word 材质纹理行 + 可选 VT 元数据）、`[core] words = 20`。瓣位段**双密集**：KHR 核心瓣 slot `0..6`、扩展瓣从 `EXTENSION_SLOT_BASE = 16` 起（`NPR_FACE` 从 bit 6 **迁移到 bit 16**）；区域划分为 Specification `0..16` / Prism `16..32` / CrossVendor `32..48` / Studio `48..64`（`namespace.rs:89-126`，`SLOTS_PER_REGION = 16`）；生成的 `SURFACE_LOBE_WORDS_BY_SLOT` 是 `[usize; 17]`（中间补零）。schema 校验器**强制**双密集段，违反即 `NonContiguousRegistrySlots` 错误。
- **schema 只 own 契约、不 own 数学**：字段键是 `name / type / bit_region / pack_slot / consumed_by / default / valid / energy_policy / view`——**没有 `tags` 字段**，所以"NPR 归属"不是靠 tag，而是靠 `namespace = "PRISM"` 与 `[[response_mode]] illumination = "stylized"`（`surface.toml:1154`）表达。行为数学一律在类型化 hooks 与落地代码里（§8.3）。

| 参数 | schema 行 | CPU 字段 / 消费点 | WESL 消费点 | 导入映射名 |
|---|---|---|---|---|
| `base_color` | `surface.toml:33` | `resolve.rs`、`stylized.rs:206` | `brdf.wesl`、`material.wesl` | `base_color` |
| `base_metalness` | `:46` | `resolve.rs`、`clearcoat.rs` | `brdf.wesl` | `base_metalness` |
| `specular_roughness` | `:57` | `resolve.rs`、`stylized.rs:213` | `brdf.wesl` | `specular_roughness` |
| `specular_weight` | `:69` | `resolve.rs`、`lighting.rs` | `brdf.wesl` | `specular_weight` |
| `ambient_occlusion` | `:82` | `resolve.rs`、`environment/mod.rs` | `lighting.wesl` | `ambient_occlusion` |
| `normal_scale` | `:93` | `resolve.rs`、`texture_sample.rs` | `surface.wesl` | `normal_scale` |
| `alpha_cutoff` | `:104` | `alpha` 遮罩路径 | `material.wesl` | `alpha_cutoff` |
| `base_weight` | `:115` | `resolve.rs`、`clearcoat.rs` | `brdf.wesl` | `base_weight` |
| `base_diffuse_roughness` | `:129` | `resolve.rs`、`lighting.rs` | `brdf.wesl` | `base_diffuse_roughness` |
| `specular_color` | `:143` | `resolve.rs` | `brdf.wesl` | `specular_color` |
| `specular_ior` | `:156` | `resolve.rs`、`clearcoat.rs`、`water.rs` | `brdf.wesl:217` | `specular_ior` |
| `geometry_thin_walled` | `:170` | `resolve.rs`、`lighting.rs` | `shading_resolve.wesl` | `geometry_thin_walled` |
| `geometry_opacity` | `:189` | `resolve.rs`、`lighting.rs` | `material.wesl` | `geometry_opacity` |
| `emission_color` / `emission_luminance` | `:231` / `:245`（lobe 218） | `resolve.rs`、`stylized.rs:242` | `shading_resolve.wesl` | `emission_color` / `emission_luminance` |
| `coat_*`（6 项） | lobe 285 | `clearcoat.rs`、`resolve.rs` | `clearcoat.wesl:317-326` | `coat_*` 同名 |
| `anisotropy`（=`specular_roughness_anisotropy`） | lobe 435 | `lighting.rs:899-917` | `brdf.wesl:878-886` | `specular_roughness_anisotropy` |
| `sheen`（=`fuzz_*`） | lobe 494 | `cloth.rs`、`resolve.rs` | `brdf.wesl` | `fuzz_weight/color/roughness` |
| `subsurface_*`（5 项） | lobe 565 | `subsurface.rs`、`resolve.rs` | `brdf.wesl` / SSS 预算 | `subsurface_*` 同名 |
| `transmission_*`（7 项） | lobe 689 | `lighting.rs`（slab）、`resolve.rs` | `brdf.wesl` | `transmission_*` 同名 |
| `thin_film_*`（3 项） | lobe 828 | `resolve.rs` | `brdf.wesl`（Airy） | `thin_film_*` 同名 |
| `npr_face_softness` | lobe 885 | `face_shadow.rs` | `brdf.wesl:1212-1245`、`shading_resolve.wesl:2560` | **无**（Prism 扩展，导入器不认） |
| `glint_*`（4 项） | lobe 939 | `glint.rs`、`resolve.rs` | `brdf.wesl` | **无**（Prism 扩展） |
| `measured_brdf` | lobe 1015 | `measured_brdf.rs` | — | **无**（Prism 扩展） |

**导入层**：`pkg/prism_render_material_authoring/src/import.rs:255-296` 列出 **41 个逐字映射**的 OpenPBR 输入名 + 2 个重命名（共 43 个拼写，见本节开头），`:446` 对不认识的 OpenPBR 字段直接报 `unsupported OpenPBR field '<field>'`——**即"参数是否有功能"的第一道门是导入器是否认这个词**；Prism 扩展（`npr_face` / `glint` / `measured_brdf`）不走这个词汇表，只在 Prism 侧作者化。

---

## 11. 缺口与路线图（诚实清单）

| # | 缺口 | 性质 | 建议 |
|---|---|---|---|
| 1 | **NPR 直连只覆盖 3/40+ 参数** | 能力缺口（非规范缺口） | P1：`specular_weight` → 高光强度、`specular_color` → 高光染色（§4.2）；**施工方案见 §12** |
| 2 | **NPR 风格参数是 per-view/per-style，不是 per-material** | 易用性缺口：美术无法在材质里表达"这个角色有/无高光" | 给 NPR arm 加逐材质调制层（§4.4）；合并规则三选一见 §12.4 |
| 3 | **`npr_face` 是唯一进 über 的 NPR 参数**，其余 NPR 通道（描边/ramp 贴图/light layer）参数不在材质 ABI 里 | 架构现状 | 若要"per-material ramp 贴图"，需参照 `npr_face` 的范式再加瓣（开放词汇表本来就是加法） |
| 4 | `anisotropy` 一词在两处不同义（specular 伸长 vs 体散射相位 `g`） | 检索/沟通风险 | 本文与 §4.4 统一称后者为"相函数各向异性" |
| 5 | 4 个 `geometry_*` 向量字段是 shade-time 输入而非材质参数，容易被误当成"参数" | 文档口径 | 已在 §2.4 / §3.10 标注 |
| 6 | `transmission_depth` 的"路径 = 1 场景单位"是 Prism 显式约定 | 刻意的偏差（已在 §4.4 声明） | 保持；单位口径写在 `SurfaceSample` 文档里 |
| 7 | CPU/WESL 双线一致性靠人工对拍（emission 漏乘是实例） | 工程风险 | 已有 parity 测试（`prism_render_material/tests/schema_parity.rs`）；新参数必须双线同改 |
| 8 | **`thin_film_*` 做成了 ABI 瓣，而规范要求并入 base 微面 BSDF 的 Fresnel 项、不新增 slab** | 有意的架构偏差（换取瓣级裁剪 + per-material 开关） | 保持；但要确认**介电基底的透射彩纹**与**金属基底的复 IOR 反推**（Gulbrandsen 2014）都已覆盖，否则只是"看起来有虹彩" |
| 9 | **Prism 主动纠正了规范的解析 albedo 勘误**（`ρ_ms` 的 `1/π`；照字面实现白炉只到 0.70） | 规范文本缺陷 | 已用 17×9 烘焙表 + `eon_lobe_passes_the_white_furnace` 固化；可向 ASWF 上游反馈 |
| 10 | **NPR 已有 9 个 golden 模块（含 Kuwahara／半调／交叉排线／风格化头发），但 inverted-hull 描边壳、降帧步进（on 2s/3s）、墨线笔刷尚未落地** | 进度缺口（design §14.2–14.3 编号 6–13） | 按 NPR 路线图排期；这些**都不吃 OpenPBR 参数**，所以不影响本文的参数功能口径 |
| 11 | `specular_ior` / `specular_weight` 的 Prism 值域比规范窄（`1..=3` vs (0,∞)；`0..=1` vs [0,∞)） | 刻意的合法性约束 | 若要支持 `specular_weight > 1` 的艺术夸张，需放宽 schema 谓词而非绕过校验 |
| 12 | **`bin_class` 的 lobe 位窗口已满（19/19，最后一个合法位是 bit 18）**，而 Prism 瓣被 schema 强制从 slot 16 起严格连续（`NonContiguousRegistrySlots`，`prism_material_schema/src/lib.rs:730-735`），下一个 Prism 瓣**必须是 slot 19** ⇒ 新瓣位会被 `to_bin_class` 的 `(closure_mask & 0x7_FFFF) << 10` **静默掩掉**，导致 CPU 直接路径与 GPU-driven bin 路径不一致 | **静默风险**：`SpecializationId`（u64，closure_mask 占完整 32 位，位段 8..40）能表达它，`bin_class` 不能；且未找到"lobe 位必须 < 19"的显式断言门禁 | 加新瓣前先把 `BIN_CLASS_BITS` 29 → 32（窗口 19 → 22 位，正好容下 slot 19/20/21 三个新瓣），并补一条位预算断言；更多瓣需换方案 |

**明确不是缺口**：Unlit 线下 specular/transmission/coat/fuzz/thin_film 无功能——这是 `illumination` 轴的正确语义；Ramp/SDF/描边比 PBR 更省算力是设计预期（design §14.4）。

---

## 12. 施工方案：把 OpenPBR 参数接进 NPR（§4.2 的落地路径）

### 12.1 为什么这件事便宜：三个已验证的前提

1. **参数已经在 `SurfaceSample` 里**：`pkg/prism_render_shading/src/lighting.rs:24` 起的结构体已持有全部 40+ 字段——NPR 不是"拿不到"，是**不读**。
2. **响应函数已经拿到 `surface`**：CPU 侧 `pkg/prism_render_shading/src/stylized.rs:185-190` 的 `evaluate_stylized_direct(surface, frame, light, params)`；GPU 侧 `pkg/prism_render_scene/src/shaders/brdf.wesl:1167` 的 `stylized_direct(surface, frame, light, params)`。三个调用点 `shading_resolve.wesl:1583`（方向光）／`:1600`（点光）／`:1609`（面光）**全部在 `shade_toon`（`:1552`）内**，`lit: SurfaceSample` 都在作用域里。
3. **"新增效果型"参数的默认值全部是中性值**（见 12.2）。

**结论**：不需要改 schema、不需要改材质 ABI、不需要加瓣、不需要动分类表、不需要动 `bin_class` 位预算、不需要动 `project_to_openpbr`。改动面 = **两个孪生函数 + 常量 + 测试**。

### 12.2 硬不变量：默认值恒等（向后兼容契约）

**契约**：任何参数取其 OpenPBR 默认值时，NPR 输出必须与今天**逐位相同**。逐类核对（默认值取自 `pkg/prism_material_schema/schema/surface.toml`）：

| 接入角色 | 参数 | 默认值 | 为什么默认恒等 |
|---|---|---|---|
| **门控型**（0 = 效果关闭） | `coat_weight` / `fuzz_weight` / `subsurface_weight` / `transmission_weight` / `thin_film_weight` / `glint_weight` / `measured_weight` | 全为 `0.0` | 整族效果被 `weight == 0` 短路 |
| **门控型** | 上述各族的伴随参数（`coat_*` / `fuzz_*` / `subsurface_*` / `transmission_*` / `thin_film_*`） | 含**非中性**值，如 `subsurface_radius_scale = [1.0,0.5,0.25,1.0]`、`coat_ior = 1.6` | 安全**仅**因为它们被同族 `weight == 0` 门控——**门控是唯一的安全来源**，不能只靠默认值 |
| **乘法型**（1 = 恒等） | `specular_weight` / `ambient_occlusion` / `base_weight` | `1.0` | 直接作乘子即恒等 |
| **不染色型**（白 = 恒等） | `specular_color` / `coat_color` / `fuzz_color` / `transmission_color` / `thin_film` 无此 | `1,1,1,1` | 乘法染色，白即恒等 |
| **加法型**（0 = 不新增） | `base_metalness` / `base_diffuse_roughness` / `anisotropy` / `anisotropy_rotation` / `geometry_thin_walled` / `subsurface_scatter_anisotropy` | `0.0` | 作为"选择另一条曲线"的插值参数，0 = 取现有曲线 |
| **⚠️ 需归一化** | `specular_ior` | **`1.5`（非中性）** | **唯一不能直接用**的参数：必须做归一化映射（如 `f = 1 − 1/η`，1.5 ⇒ 1/3；或显式除以 1.5），否则默认值就会改变输出 |
| **已消费** | `base_color` / `specular_roughness`（=`perceptual_roughness`）/ `emission_*` / `normal_scale` / `alpha_cutoff` / `geometry_opacity` / `npr_face_softness` | — | 已在用（§4.1），不在本次"新接"范围 |

**这就是分阶段接入可以随时合并的理由**：每接一个参数，**老资产渲染结果不变**。反过来说，`specular_ior` 是这条契约上唯一的雷。

### 12.3 施工步骤（五步，每步可独立合并）

**S1 — 先定语义表（一页纸，必须先做）**
"某参数在 NPR 下当什么用"是**产品/美术决策**，不是代码决策（§4 引言已证：规范许可重解读，因此没有任何合规约束会替你做这个决定）。需要美术签字的至少包括：`base_metalness` 在 NPR 下是"金属感 ramp 曲线"还是"忽略"；`transmission_weight` 是"背光透亮"还是"忽略"；`coat_weight` 的二段高光是"叠加"还是"替换"。

**S2 — 改 CPU 金标准**：`pkg/prism_render_shading/src/stylized.rs`
- 新增常量时注意**命名双线不一致**：Rust 侧叫 `MIN_SPECULAR_EXPONENT` / `MAX_SPECULAR_EXPONENT`（`stylized.rs:215`），WESL 侧叫 `MIN_STYLIZED_SPEC_EXPONENT` / `MAX_STYLIZED_SPEC_EXPONENT`（`brdf.wesl:1179`）——**必须成对添加**，否则孪生断裂。
- 按 12.2 的门控/乘法约定读字段，保持 `StylizedParams::default()`（`stylized.rs:103-106`，= `legacy_toon(4)`）与 `legacy_toon`（`:112`）**不动**。

**S3 — 同步 WESL 孪生**：`brdf.wesl:1167 stylized_direct` 逐字节同改
- WESL 侧**刻意不用内建 `mix` / `smoothstep`**（用 `stylized_mix:1112`、`stylized_smoothstep:1119` 对齐 CPU 语义）；新增的插值/软边逻辑**必须走这两个 helper**，否则位模式漂移。
- 文件头注释声明 "Byte-for-byte twin of `evaluate_stylized_direct` in `stylized.rs`"（`brdf.wesl:1165-1166`）——这是必须维持的契约。

**S4 — 测试与门禁**
- **恒等门禁（新参数必配）**：断言"取默认值 ⇒ 与改动前逐位相同"。参考实现：与 `evaluate_toon_direct`（`stylized.rs:251`）对拍，或冻结一份期望值。
- **生效门禁（新参数必配）**：断言"取非默认值 ⇒ 输出确实改变"——防"接了线但被门控掉"的假接线。
- 既有 8 个测试在 `stylized.rs:301-560`，风格照抄：`legacy_params_match_historical_toon_bit_for_bit:302`、`toon_and_pbr_share_light_visibility_contract:334`、`output_is_finite_and_non_negative_across_the_stack:352`、`ramp_softness_smooths_band_edges:401`、`half_lambert_wrap_lifts_the_terminator:432`、`stepped_shadow_hardens_the_penumbra:465`、`stylized_specular_adds_a_bounded_hot_spot:496`、`rim_light_survives_shadow:526`。
- **`output_is_finite_and_non_negative_across_the_stack:352` 是新增参数后最容易破的测试**——新项务必同样保持有限且非负。
- GPU 对拍：`pkg/prism_render_scene/src/shading/resolve/shader_tests.rs`；NPR 水面另见 `pkg/prism_volumetric_gpu/tests/water_shading_npr_parity.rs`。

**S5 — 文档回填**：更新本文 §4.1 的"3 个直连参数"表、§4.2 建议映射与 §11 缺口 1/2。

### 12.4 全局风格参数与逐材质参数的合并规则（必须显式定，否则是隐式行为）

今天风格来自**全局**资源：`shading_resolve.wesl:1571 let stylized = env.stylized;`，上游是 `pkg/prism_render_scene/src/lighting/stylized_config.rs:33`（`struct StylizedLighting`）→ `pkg/prism_render_scene/src/lighting/extract.rs:231 apply_stylized`。新接入的 per-material 参数与它如何合并，三选一：

| 规则 | 含义 | 评价 |
|---|---|---|
| **(a) 乘法门控**（推荐） | 材质参数做 **presence/gain**（默认恒等），全局风格做 **shape**（`bands`/`wrap`/threshold/softness） | 天然满足 12.2 的契约；材质能表达"有/无高光、什么颜色"，但**改不了色阶形状** |
| (b) 非默认即覆盖 | 材质非默认就覆盖全局对应量 | "非默认"判定脆弱（浮点相等比较）；且 ABI 里根本没有 `bands`/`wrap`，覆盖不了形状 |
| (c) 风格档位枚举 | 材质存一个风格 ID 选预设 | 需要 ABI 新字段 ⇒ 即 §11 缺口 3 的加瓣/加通道，**属于另一条路** |

> **边界（必须说清）**：读法 A 能让"**每个 OpenPBR 参数在 NPR 下都有功能**"，但**做不到"每个 NPR 风格量都能逐材质"**——`bands` / `wrap` / `ramp_softness` / `rim_power` 在 OpenPBR 里**没有对应物**，只能住在 `StylizedParams`。要逐材质表达它们，必须回到 §11 缺口 3（加瓣/加通道，并连带支付 `bin_class` 位预算，见 §11 缺口 12）。

### 12.5 明确**不要**动的东西

- `pkg/prism_material_schema/schema/surface.toml` —— 参数已存在，动它就是无谓的 ABI 变更。
- `Illumination` 枚举与 `MaterialShadingClass` 分类表 —— NPR 仍走 `Npr` class（`classification.rs:17-26`），接入参数不改变分类。
- `pkg/prism_render_material/src/axis.rs` 的 `bin_class` 位布局 —— 不新增瓣 ⇒ 不动位预算。
- `project_to_openpbr` 的有损向下投影 —— NPR 读多少**不改变**投影语义。
- `SurfaceSample` 的默认值 —— `lighting.rs` 的 `Default` 镜像 OpenPBR 默认表面，改它会波及 PBR 对拍（`prism_render_material` 的字节一致性门禁）。

### 12.6 缓存与视角相关性（一个容易漏的约束）

解耦着色缓存分页键是 `ShadePageKey { object_id, material_id, material_epoch, face, mip, channel, page_x, page_y }`（`pkg/prism_render_scene/src/shading/texel_cache/key.rs:5-24`），其中 `material_id` + `material_epoch` 已在键里 —— 所以**逐材质化的 NPR 响应不会串页**（`material_epoch` 还保证重作者化材质不会误用旧页）。

但通道枚举只有四项：`ShadeChannel { LitDiffuse = 0, StylizedDiffuse = 1, Subsurface = 2, Emission = 3 }`（同文件 `:5-15`）——**只有视角无关的新增项可以放心进缓存**；rim、风格高光这类视角相关项必须留在逐灯重算路径（今天正是如此）。若新接入的材质依赖项是视角相关的，还要确认它**不会污染 `StylizedDiffuse` 页**。

---

## 附录 A：术语与轴

- **closure 轴**：表面 BSDF 长什么样（über-BSDF 一等 / 有界 slab / 毛发专用）。**OpenPBR 参数属于这条轴。**
- **illumination 轴**：`Lit | Stylized | Unlit | Custom`，决定"怎么解读同一份光照数据"。**PBR / NPR 是这条轴的两个取值。**
- **PBR 线**：`illumination = Lit` + 主导 closure 投影出的 class（Principled/Subsurface/ClearCoat/Cloth/Hair/Water）→ `principled_direct`（BRDF 积分）。
- **NPR 线**：`illumination = Stylized` → `Npr` class → `stylized_direct` / `toon_direct`（ramp 量化 + 风格高光 + rim）。
- **直连 / 间接 / 未消费**：本文的三个状态标记，指该管线的**直接光响应函数**是否读取该字段。
- **C 类专属通道**：只被某条 illumination 线消费的通道（SDF 面阴影、描边、light layer、illumination mask）。
- **white furnace test**：规范的验收口径——albedo 全设 1、去掉光照后，方向反照率必须等于 1。规范要求它对金属基底、介电基底、任意混合、加 coat、加 fuzz、加薄膜、加粗糙度／各向异性／法线贴图／`geometry_opacity` **都**保持通过。Prism 侧由 `eon_lobe_passes_the_white_furnace` 等测试承担（§3.2）。
- **specular_ior_level**：**不是 OpenPBR 参数**（Blender 侧叫法）。OpenPBR 的对应物是 `specular_weight` 对 `specular_ior` 的调制（§3.3）。
- **Flexibility of implementation**：v1.1.1 新增章节（#248），明文允许实现者把参数改作它用、乃至把模型退化成更简单的 BSDF（§4 引言）。本文 §4.2 的全部建议映射都在这一许可范围内。

## 附录 B：与既有文档的交叉索引

| 本文位置 | 互补文档位置 |
|---|---|
| §2 矩阵、§3 详解 | `prism_material_pipeline_refactor_zh.md` §4.4（参数↔字段对账）、§4.5（合规门禁） |
| §1.3 五条线 | `prism_material_pipeline_design_zh.md` §3.1（正交轴）、§7（前端） |
| §4.2 / §4.3 NPR 通道 | 同文档 §4.2（A/B/C 类）、§14（NPR 特性清单） |
| §5 混合 | 同文档 §15（混合五条）、§7（tile 路由） |
| §7 缓存通道 | 同文档 §5（跨切面基底）、`texel_cache/key.rs` |
| §8 能量/合规 | `prism_material_pipeline_refactor_zh.md` §17.2、§17.5.5 |
| §9 降级 | `prism_render_shading/src/quality.rs`、`shaders/lobe_budget.wesl` |
