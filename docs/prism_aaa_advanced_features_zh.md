# Prism 渲染引擎 — 顶级次世代 AAA 高级特性专项设计（PBR / NPR / 混合）

> 本文是 `prism_material_pipeline_design_zh.md`（顶层架构与决策）的**下钻分册**：把 §12–16 的对标矩阵与特性清单展开为**逐特性规格书**——每条给出「借鉴对象 / 算法要点 / 性能预算 / 效果上限 / 模块落点 / 验收口径」。
> **一句话立场**：共享 GPU-driven 基底算一次，PBR / NPR / 自定义三前端并存消费，混合在管线级路由。**三条赛道都是一等公民，都能拿到顶级次世代 AAA 效果**，差异只在「怎么解读同一份光/影/GI/几何数据」的前端响应处。
> **纪律**：只借鉴公开算法与形态，**不本地拉取任何产品源码**（UE 算法已获授权，同样只借形态）。

---

## 0. 阅读地图与非目标

- 本文 = 特性目录 + 预算 + 验收。顶层为什么这么拆见管线文档 §0–§11。
- **非目标**：不在本文重复"为什么放弃 Slang / 为什么共享基底"的论证（见管线 §1、§2）；不做纯 raw-VK 独占特性的跨平台承诺（见 §8.1 三桶）。
- 时间预算基线：**1440p 内部渲染 + 时序上采样到 4K，目标 16.6ms（60fps）/ 高配 8.3ms（120fps）**，桌面独显参考档；集显/移动为降级档。

---

## 1. 共享基底服务（三前端的地基，算一次）

> 这些不是"PBR 的特性"，是**全前端共享的数据服务**。NPR / 混合同样消费，只是响应不同。落点均在 `prism_render_architecture/src/`。

| 服务 | 模块落点 | 产出的数据 | 三前端如何消费 |
|---|---|---|---|
| 虚拟几何 vis-buffer | `virtual_geometry/` | cluster DAG LOD、软光栅微三角、visibility buffer、material id/边界 | 三者同吃；NPR 额外白得 material id 描边边 |
| GPU 场景 / 剔除 | `gpu_scene/` `geometry/` | instance/mesh 表、GPU 剔除、draw 生成 | 全共享 |
| 光照数据 | `lighting/` | clustered 光照剔除、ReSTIR 储层预算、探针/GI 采样 | PBR 积分、NPR ramp 量化、混合共享预算 |
| 虚拟阴影 VSM | `virtual_shadow/` | 虚拟页 + clipmap 深度、residency | PBR 软阴影、NPR 阈值硬阴影+染色、另叠 SDF 面部阴影 |
| 光线场景 | `ray_scene/` | BVH/TLAS、RT 反射/阴影/GI 输入 | PBR 全保真、NPR 降级风格化近似 |
| 时序 / 上采样 | `temporal_upscale/` `motion/` `history/` | motion vector、历史累积、reactive mask、上采样 | 全共享；NPR 锐利分段靠 reactive mask 保护 |
| 透明 | `transparency/` | OIT 路径、HairVisibility 等 | 全共享 |
| 形变 / 子系统 | `deformation/` `hair/` (+ 规划中 `cloth/` `particle/`) | 蒙皮/morph/布料/毛发/顶点动画的形变预算与调度 | 全共享 |
| 材质 ABI | `material/` `abi/` | 正交轴 + 闭包 IR + über-BSDF/有界 slab | 三前端的统一材质表达 |
| 视图族 / 分帧 | `view_family/` `frame_graph/` `paging/` | 多视图、帧图、页驻留 | 全共享 |

**规则**：任何"高级特性"先问一句——它是**基底服务**（放这里，全前端白嫖）还是**前端响应**（放前端，各自解读）。绝不让 NPR/PBR 各自重造 GI/阴影/几何。

---

## 2. 顶级产品对标（借形态，不抄码）

> 比管线 §12 更细：拆出"具体借哪一招 / 借到什么程度 / 不借什么"。

### 2.1 PBR 线（业界已收敛，抄作业到位即 AAA）

| 能力 | 首选对标 | 借什么 | 明确不借 |
|---|---|---|---|
| 虚拟几何 | UE5 Nanite | cluster DAG、软光栅、vis-buffer 形态 | 其具体数据布局/源码 |
| 全局光照 | UE5 Lumen + NVIDIA RTXGI/DDGI | SDF/mesh-card 软 RT + 硬 RT 混合、surface cache、屏幕探针、DDGI 探针 | Lumen 内部实现细节 |
| 多光源/采样 | NVIDIA ReSTIR DI/GI (RTXDI) + NRD | 储层时空重采样、ReBLUR/ReLAX 去噪思路 | — |
| 阴影 | UE5 VSM + RT 阴影 | 虚拟页 + clipmap、按驻留渲染 | — |
| 材质分层 | UE5 Substrate / OpenPBR | über-BSDF + slab 分层思想 → **收敛成有界 slab** | 无界 slab（刻意封顶防 variant 爆炸） |
| 上采样 | UE5 TSR / DLSS / FSR2 / XeSS | 时序上采样形态；接厂商 SDK 为可选后端 | — |
| 参考渲染 | RED Engine Cyberpunk RT Overdrive | ReSTIR GI 路径追踪 + NRD 的离线对拍口径 | — |

### 2.2 NPR 线（无现成招牌管线，最大增量也最高风险）

| 能力 | 首选对标 | 借什么 |
|---|---|---|
| 卡通着色 | miHoYo（原神/星铁）、Arc System Works（GG Xrd / DBFZ） | ramp/阶梯 Blinn、ID map + 顶点色控制、手编法线 |
| 面部阴影 | miHoYo / HoYo 系 | SDF 面部阴影图（独立数据、主光方向阈值切换）+ 阈值硬阴影染色 |
| 高光 | miHoYo / Arc Sys | 天使环各向异性高光带（与真实切线解耦）、MatCap、阶梯高光 |
| 描边 | Arc Sys / miHoYo / Borderlands | inverted-hull 背面挤出（顶点色控宽度）、屏幕空间深度/法线边、material id 边、墨线 |
| view-dependent ramp | Valve TF2（论文） | 半兰伯特 warp + light-warp ramp + rim |
| 影视风格化 | Spider-Verse / Arcane(Fortiche) / Okami | halftone/Ben-Day 点、色差、墨线、降帧步进、水墨/宣纸、Kuwahara 油画 |

### 2.3 混合线（对标"风格化 PBR"与影视混合管线）

| 形态 | 对标 | 借什么 |
|---|---|---|
| 风格化 PBR 中间态 | Fortnite / Valorant / Overwatch / Sea of Thieves | über 少瓣 + Stylized illumination 轻叠加 + 真 GI/Nanite/Lumen |
| 影视级混合 | Spider-Verse / Arcane | 3D 基底 + 2D 手绘 FX/线 + 逐物体降帧步进 |
| 选择性 NPR | 影视混合、卡通角色进实景 | 写实世界里对特定角色/道具开 Stylized，其余 PBR，共享同一光影 |

---

## 3. PBR 前端 AAA 高级特性规格

> 前端 = 延迟 PBR（承载虚拟几何/RT/GI/VSM 的最高保真度驱动方）。逐条：算法 / 预算 / 效果 / 落点。

### 3.1 虚拟几何（Nanite 式）
- **算法**：cluster DAG 多级 LOD + 软光栅化微三角 + visibility buffer 延迟着色；几何与像素解耦。
- **预算**：软光栅带宽是主成本；vis-buffer 一趟，material 解析一趟。目标亿级三角、零 LOD pop。
- **落点**：`virtual_geometry/`（现为 stub，最大出血点，见管线 §16）。
- **验收**：静态相机移动无 LOD 跳变；material id 边界可供 NPR 描边复用。

### 3.2 混合 GI（Lumen 式 + DDGI 兜底）
- **算法**：SDF/mesh-card 软件 RT + 硬件 RT 混合，surface cache 摊薄弹射；无 RT 平台降级 SSGI + DDGI 探针。
- **预算**：surface cache 更新分帧摊销；探针体积内存可调。
- **落点**：`lighting/`（GI 采样与探针）+ `ray_scene/`（硬件 RT 输入）。
- **验收**：动态光照无烘焙下间接光收敛；关 RT 时 SSGI/DDGI 平滑降级不黑。

### 3.3 ReSTIR DI/GI + 去噪
- **算法**：光源储层时空重采样代替全光遍历；ReBLUR/ReLAX 式时空去噪。
- **预算**：储层 + 去噪；`lighting/` 已有 `ReservoirBudget` 与视空间 clustered 剔除可接。
- **落点**：`lighting/`（储层）+ `temporal_upscale/`（时序去噪协同）。
- **验收**：千级动态光下低方差、无萤火虫；去噪不糊化边缘。

### 3.4 虚拟阴影 VSM
- **算法**：虚拟页 + clipmap，只渲染驻留页。
- **预算**：residency 管理（已落）；只渲可见页。
- **落点**：`virtual_shadow/`。
- **验收**：全程一致阴影密度，近景不虚、远景不抖。

### 3.5 有界 slab 多瓣材质（Substrate 收敛版）
- **算法**：über-BSDF 一等 + **有界数量** slab 分层（清漆/车漆/多层皮），封顶防 shader variant 爆炸。
- **预算**：shader 特化桶；封顶是刻意取舍（vs UE 无界）。
- **落点**：`material/` `abi/`（闭包 IR）。
- **验收**：清漆/车漆/多层皮达标；variant 数受控。

### 3.6 SSS（次表面散射）
- **算法**：屏幕空间可分离卷积（burley/diffusion profile）。
- **预算**：compute-可移植桶；屏幕空间一趟。
- **落点**：材质 über 瓣（**不是独立子系统**，见管线 §6.2 陷阱档）。
- **验收**：皮肤/蜡/玉透光自然，边缘不发绿。

### 3.7 RT 反射 / 路径追踪参考
- **算法**：硬件 BVH 反射；路径追踪参考模式用于离线对拍与校准。
- **预算**：RT 桶（部分可测）；Metal RT 较弱、跨平台部分覆盖。
- **落点**：`ray_scene/`。
- **验收**：物理镜面正确；参考模式与实时管线对拍误差可量化。

### 3.8 体积雾 / froxel / 体积云
- **算法**：froxel 体积散射 + 体积云。
- **预算**：froxel 分辨率可调、froxel 内存为主成本。
- **落点**：`lighting/`（体积）+ 基底。
- **验收**：光轴/大气/云层无带状伪影。

### 3.9 时序上采样（TSR/DLSS/FSR/XeSS）
- **算法**：低分辨率内部渲染 + 时序累积上采样到目标分辨率。
- **预算**：依赖 motion vector + reactive mask；可接厂商 SDK 为可选后端。
- **落点**：`temporal_upscale/` `motion/` `history/`。
- **验收**：4K 级清晰度，运动无重影/拖尾。

---

## 4. NPR 前端 AAA 高级特性规格（顶级二次元 / 影视风格化）

> NPR **享有几乎所有基底服务**（见 §1 与管线 §13.1）。这里列的是 NPR **专属响应与专属数据**。整体比 PBR 省算力，贵在美术管线与 TAA 摩擦。

### 4.1 着色（对标 miHoYo / Arc Sys / Valve TF2）
1. **ramp / 阶梯量化光照**：把 ReSTIR/GI 的辐照度用 ramp LUT 量化成分段，而非 BRDF 积分。**省**：一次贴图采样。
2. **ID map + 顶点色控制**：分区控制着色参数（GG Xrd 式），手编法线修正卡通高光形状。
3. **半兰伯特 + light-warp ramp + rim**（TF2 论文）：view-dependent 暖冷过渡 + 边缘光。
4. **NPR 只收不发 GI**：收 GI 做风格化底光，不向外弹射（防能量爆炸），见管线 §4.2。

### 4.2 面部阴影与高光（NPR 专属数据）
5. **SDF 面部阴影图**：独立数据通道，按主光方向阈值切换左右脸阴影，走**专属通道**叠在 VSM 之上。
6. **天使环各向异性高光带**：与真实切线解耦的高光带；MatCap；阶梯 Blinn 高光。
7. **阈值硬阴影 + 染色**：消费同一份 VSM 深度页，用阈值二值化 + 阴影色染色（非物理软阴影）。

### 4.3 描边（对标 Arc Sys / miHoYo / Borderlands）
8. **inverted-hull 背面挤出**：顶点色控宽度的轮廓线（几何法）。
9. **屏幕空间深度/法线边 + material id 边**：后处理法，material id 边界白嫖自虚拟几何 vis-buffer。二者互补。

### 4.4 风格化后处理（对标 Spider-Verse / Arcane / Okami）
10. **Halftone / Ben-Day 点 / 网点**：按亮度控点密度做印刷风阴影。
11. **降帧步进（on 2s/3s）**：角色/特效按 12/8 fps 步进与 60fps 背景混排。**需 motion vector 特判防 TAA 抹掉**。
12. **Kuwahara / 油画滤镜**：保边匀色手绘感。
13. **墨线 / 笔刷 / 水墨**（Okami 宣纸 / Borderlands 墨线）：沿边缘叠笔刷 alpha + 纸纹。
14. **风格化 bloom / 色差 / 分级**：夸张辉光 + 边缘色散 + 风格化 LUT。

### 4.5 NPR 也享有的基底高级特性（勿误读为"只有 PBR 有"）
| 特性 | NPR 享有度 | 形态 |
|---|---|---|
| 虚拟几何/vis-buffer | ✅ 完全 | material id 边界反而白送描边 |
| Lumen 混合 GI | ✅（只收不发） | 收 GI 做风格化底光 |
| ReSTIR DI/GI | ✅ 完全 | 同批光源，ramp 量化解读 |
| VSM 虚拟阴影 | ✅ 数据共享 | 阈值硬阴影+染色，另叠 SDF 面部阴影 |
| 体积雾/云 | ✅ 完全 | 可做风格化染色/阶梯体积 |
| 时序上采样 | ⚠️ 共享有摩擦 | 锐利分段/步进 vs TAA，**必须 reactive mask** |
| RT 反射 | ⚠️ 降级 | 同 BVH，NPR 降为风格化 albedo 近似 |
| 路径追踪参考 | ➖ 偏 PBR | 次级光线 NPR 本就降级，参考主要校准 PBR |
| 有界 slab / SSS | ➖ closure 轴 | 正交轴，NPR 可选叠但非主诉求 |

### 4.6 NPR 性能 / 效果总纲
- **省**：ramp/SDF/描边多为一次贴图采样或一趟后处理，比物理积分省。
- **贵在**：SDF 面图、ID map、per-material ramp 是额外资产；顶点色/手编法线需 DCC 侧配套。
- **头号坑**：与 TAA/上采样的固有摩擦——锐利分段与步进动画和时序累积打架，**reactive mask 是 NPR 时序管线的必修课**（见管线 §5）。

---

## 5. 混合前端 AAA 高级特性规格（管线级混合）

> 混合的价值在**管线级**：同一帧不同像素/物体走不同前端，却共享同一套光/影/GI/RT，所以风格切换处不断裂。这是"各前端各自重算"给不了的。

1. **逐像素 material id + tile 分类路由**：GPU 按 material id 分 tile，路由到延迟 PBR / forward+ / NPR / 自定义前端。**性能**：tile 分类 + 前端 specialization，避免全屏跑所有前端。
2. **共享 GI/阴影跨风格连贯**：NPR 角色与 PBR 场景吃**同一份** Lumen 间接光 + VSM 阴影 → 卡通角色落在写实场景阴影方向/底光一致，不"贴纸感"。
3. **选择性 NPR**：写实世界里对特定角色/道具开 Stylized，其余 PBR。
4. **风格化 PBR 中间态**（Fortnite/Valorant/Overwatch/Sea of Thieves）：über 少瓣 + `illumination=Stylized` 的 ramp/rim 轻叠加 + 真 GI/Nanite/Lumen。**产量最大的商业中间带**，本架构靠正交轴自由组合天然覆盖。
5. **影视级混合**（Spider-Verse/Arcane）：3D 基底 + 2D 手绘 FX/线 + 逐物体降帧步进。

**为什么管线级混合优于单独管线**：单独管线会让 NPR/PBR 各自重造 GI/阴影/RT/上采样——没一条能到 AAA 且风格接缝断裂。共享基底 + 前端分叉是"四者皆顶级 + 可混合 + 多平台"的唯一解。

---

## 6. 帧时间预算（1440p→4K，60fps 目标档）

> 粗预算，用于取舍与验收基线；实际以 GPU profiling 为准（沙盒无 GPU，本表为设计目标非实测）。

| 阶段 | PBR 场景占比 | NPR 场景占比 | 混合场景 | 备注 |
|---|---|---|---|---|
| 虚拟几何 vis-buffer + 剔除 | ~20% | ~20% | ~20% | 三者共享，几何成本一致 |
| GI（Lumen/surface cache 摊销） | ~18% | ~10%（只收不发省） | ~16% | NPR 省在不弹射 |
| ReSTIR + 去噪 | ~15% | ~8%（ramp 量化省积分） | ~13% | 光源储层共享 |
| VSM 阴影 | ~10% | ~10%（+SDF 面部小额） | ~10% | 深度页共享 |
| 前端着色 | ~15%（BRDF 积分） | ~8%（ramp/SDF 省） | ~15%（tile 路由 + 特化） | 混合多出 tile 分类 |
| 体积/雾/云 | ~7% | ~7% | ~7% | 共享 |
| 时序上采样 | ~8% | ~8%（+reactive mask 小额） | ~8% | 共享 |
| 风格化后处理 | ~2% | ~9%（halftone/墨线/步进） | ~6% | NPR 后处理重 |
| 透明/其他 | ~5% | ~5% | ~5% | — |

**结论**：NPR 整体算力**低于** PBR（省在 GI 不弹射 + ramp 量化 + 着色简化），但吃美术管线与 reactive mask 复杂度；混合多出 tile 路由与前端特化，换来风格接缝连贯——**三线皆 AAA 的性价比最优点**。

---

## 7. 效果验收口径（三线一致的"顶级"判据）

- **PBR**：与路径追踪参考模式离线对拍，误差可量化收敛；动态光照无烘焙、无萤火虫、无 LOD pop。
- **NPR**：达顶级二次元观感（原神/GG Xrd 级 ramp/描边/面部阴影）；运动下锐利分段不被 TAA 抹糊（reactive mask 生效）；降帧步进与背景混排不撕裂。
- **混合**：卡通角色置入写实场景，阴影方向/底光/间接光一致，无"贴纸感"；风格切换处像素级路由无接缝。
- **跨平台**：野心效果落 compute-可移植桶，可 CPU golden 对拍；RT 桶部分可测；raw-VK 桶严格隔离（见管线 §8.1）。

---

## 8. 落点与路线图映射

| 特性块 | 主模块 | 现状 | 优先级 |
|---|---|---|---|
| 虚拟几何 | `virtual_geometry/` | stub，最大出血点 | P0（先立几何基底） |
| 光照/GI/ReSTIR | `lighting/` | 有 clustered 剔除 + ReservoirBudget | P0 |
| VSM | `virtual_shadow/` | residency 已落 | P0 |
| 时序/上采样/reactive mask | `temporal_upscale/` `motion/` `history/` | 基础在，reactive mask 是 NPR 头号前置 | P0（NPR 前置） |
| 材质 ABI / 有界 slab / SSS | `material/` `abi/` | 正交轴 + 闭包 IR 已定 | P1 |
| RT 反射/路径追踪参考 | `ray_scene/` | 基础在，Metal RT 弱 | P1（跨平台部分覆盖） |
| 混合 tile 路由 | `material/` + 前端 | 依赖 material id 基底 | P1 |
| NPR 专属响应（ramp/SDF/描边/后处理） | 前端 + 专属数据通道 | 待建 | P1（reactive mask 就绪后） |
| 子系统（毛发/布料/粒子/体积） | `hair/`（进行中）、`cloth/`/`particle/`（待建） | 见各子系统文档 | 并行推进 |

**总原则**：先立 P0 基底（几何/光照/阴影/时序），三前端才有共享数据可消费；NPR 的 reactive mask 属 P0 前置；有界 slab/SSS/RT 属 P1；子系统并行开发。

---

## 附：与管线文档的关系
- 顶层架构 / 决策 / 后端策略 / 三桶可测性 / 代码重构清单：见 `prism_material_pipeline_design_zh.md`。
- 毛发子系统：见 `prism_hair_engine_design_zh.md`。
- 粒子：见 `prism_particle_engine_design_zh.md`。物理：见 `prism_physics_design_zh.md`。
- 本文只负责"三前端各自与共享的高级特性目录 + 预算 + 验收"。
