# Prism 渲染引擎 — 顶级次世代 AAA 高级特性专项设计（v5 / PBR·NPR·混合 全前端 + 子系统完成度 + 2025–2026 天花板增补）

> 本文是 `prism_material_pipeline_design_zh.md`（顶层架构与决策）的**下钻分册**：把 §12–16 的对标矩阵与特性清单展开为**逐特性规格书**——每条给出「借鉴对象 / 算法要点 / 性能预算 / 效果上限 / 模块落点 / 验收口径」。
> **一句话立场**：共享 GPU-driven 基底算一次，PBR / NPR / 自定义三前端并存消费，混合在管线级路由。**三条赛道都是一等公民，都能拿到顶级次世代 AAA 效果**，差异只在「怎么解读同一份光/影/GI/几何数据」的前端响应处。
> **纪律**：只借鉴公开算法与形态，**不本地拉取任何产品源码**（UE 算法已获授权，同样只借形态）。
> **数值红线（硬约束）**：所有高级特性走**纯经典数值路径**（蒙特卡洛/准蒙特卡洛、SH/SG、reservoir 重采样、SDF 步进、时空双边降噪、时空蓝噪声、FFT）。**不引入任何 AI / ML / 神经网络 / LLM 路径**——不做神经降噪、Ray Reconstruction、神经辐射缓存、神经上采样、神经材质压缩。厂商时序上采样 SDK（DLSS/FSR2/XeSS）仅作**可选外部后端**接入，本体默认路径为纯经典 TSR 式时序累积，保证 CPU golden 可对拍、跨平台可移植。
> **v2 本版新增**：在 v1（§1–§8，三前端共享基底 + PBR/NPR/混合逐特性规格）之上，追加 **§6「次世代前沿高级特性全景」**——覆盖几何 / 阴影 / 反射 / 材质 / 体积 / 透明 / 毛发 / 水体 / 后期影视 / 采样降噪 / 上采样抗锯齿 / 性能工程 12 条前沿赛道，对标最新实时 AAA 天花板（UE5.6 / Cyberpunk RT Overdrive / Alan Wake 2 / Portal RTX / Horizon / Nanite Tessellation / MegaLights），逐条给算法要点 + 预算 + 落点 + 验收；并刷新 §2 对标矩阵（§2.3 前沿总表）与 §9 落点路线图（按仓内现状分级）。
> **v3 本版新增**：按 2026-10 仓内实况刷新 §9 分级与文件计数（虚拟几何/GI/水体/体积/采样降噪/后期多项由 🟡 升 ✅，依据各 `pkg/` 子目录真实规模）；新增 **§10「子系统完成度矩阵 + 跨子系统集成 + 子系统 AAA 高级特性」**，把物理（GPU 刚体 TGS / XPBD / MPM / FLIP / 断裂 / 软体 VBD）、音频（HRTF 双耳 / 光线声学 / 空间化）、水体（FFT 海面 / FLIP / 浅水）、毛发、布料、体积六大子系统的现状、缺口、及达到 AAA 所需高级特性逐条展开，**全程纯经典数值，排除所有 AI/ML/LLM 路径**。
> **v4 本版新增**：① 按 2026-10 仓内实况再刷新规模计数（物理 GPU 188 / 音频 core 67 / 毛发 GPU 41 / `ray_scene` 82 文件）；② 新增 **§11「次世代天花板增补（2024–2026 前沿，纯经典数值）」**——补齐当前仍是空白或仅骨架的最前沿赛道：辐射级联（Radiance Cascades, PoE2 / Sannikov）、RT 簇级加速结构（RTX Mega Geometry 形态，让虚拟几何可被硬件 RT）、不透明微贴图（Opacity Micromaps）、解耦/物体空间着色（texture-space shading）、镜面/法线抗锯齿（Toksvig/LEAN）、运行期虚拟纹理（RVT）、虚拟几何蒙皮/植被、输入延迟流水线（Reflex 形态，纯经典）；逐条给借鉴对象 + 算法要点 + 预算 + 落点 + 验收；并把这些增补并入 §9 路线图优先级。**全程纯经典数值，排除一切 AI/ML/神经/LLM 路径**（厂商时序上采样 SDK 仅渲染侧可选外部后端）。
> **v5 本版新增（2026-10）**：① 按仓内实况再刷新规模计数（`prism_render_shading` 344 / `gi/` 62 子系统 · `prism_render_scene`(ray_scene) 456 · `prism_render_architecture` 671 · 虚拟几何 GPU 49 · 体积 GPU 163 · 毛发 GPU 98 · 物理 GPU 263 / core 168 / geometry 33 · 音频 core 79 / spatial 52 文件）；② 新增 **§12「2025–2026 最前沿天花板增补（纯经典数值）」**——补齐 §6/§11 仍未展开、却已是当代影视/实时 AAA 天花板的赛道：**OpenPBR 收敛标准材质 · 光谱渲染/Hero 波长 · 薄膜虹彩（Belcour-Barla）· ACES 2.0/AgX/OCIO 显示变换 + 物理相机自动曝光 · 光树/ReGIR 多光重要性采样 · 世界空间哈希辐照缓存 · 异质介质体积路径追踪（delta/ratio tracking）· 物理天空多重散射 · Sampler Feedback 纹理流送 · DirectStorage/GDeflate GPU 解压流送 · 可见性缓冲延迟纹理化+材质分箱 · 矩不变 OIT · 波动声学/高阶 Ambisonics · 跨平台确定性收敛**；逐条给借鉴对象+算法要点+预算+落点+验收，并给 §12 增补路线图优先级。**全程纯经典数值，排除一切 AI/ML/神经/LLM 推理路径**（厂商时序上采样 SDK 仅渲染侧可选外部后端）。

---

## 0. 阅读地图与非目标

- 本文 = 特性目录 + 预算 + 验收。顶层为什么这么拆见管线文档 §0–§11。
- GI 深水区（surface cache / 多层辐射缓存 / World-Space ReSTIR / ReSTIR PT / 焦散 / 探针体）在 `prism_gi_lumen_design_zh.md`（v4）独立成册，本文 §6.3 反射与 §6.5 体积只给**衔接口径**，不重复其内部规格。
- **非目标**：不在本文重复“为什么放弃 Slang / 为什么共享基底”的论证（见管线 §1、§2）；不做纯 raw-VK 独占特性的跨平台承诺（见 §8.1 三桶）；不做离线烘焙管线（Phase 6 另行）；**不做任何神经/ML 推理路径**（见上数值红线）。
- 时间预算基线：**1440p 内部渲染 + 时序上采样到 4K，目标 16.6ms（60fps）/ 高配 8.3ms（120fps）**，桌面独显参考档；集显/移动为降级档。沙盒无 GPU，所有耗时为**设计目标（design target），非实测**。

---

## 1. 共享基底服务（三前端的地基，算一次）

> 这些不是“PBR 的特性”，是**全前端共享的数据服务**。NPR / 混合同样消费，只是响应不同。落点均在 `prism_render_architecture/src/` 与 `prism_render_shading/src/gi/`。

| 服务 | 模块落点 | 产出的数据 | 三前端如何消费 |
|---|---|---|---|
| 虚拟几何 vis-buffer | `virtual_geometry/` | cluster DAG LOD、软光栅微三角、visibility buffer、material id/边界 | 三者同吃；NPR 额外白得 material id 描边边 |
| GPU 场景 / 剔除 | `gpu_scene/` `geometry/` | instance/mesh 表、GPU 剔除、draw 生成 | 全共享 |
| 光照数据 | `lighting/` + `gi/world_restir/` | clustered 光照剔除、ReSTIR 储层预算、探针/GI 采样 | PBR 积分、NPR ramp 量化、混合共享预算 |
| 虚拟阴影 VSM | `virtual_shadow/` | 虚拟页 + clipmap 深度、residency | PBR 软阴影、NPR 阈值硬阴影+染色、另叠 SDF 面部阴影 |
| 光线场景 | `ray_scene/`（80 文件，软件 BVH 已成规模） | BVH/TLAS、RT 反射/阴影/GI 输入 | PBR 全保真、NPR 降级风格化近似 |
| 全局光照 / 反射 | `gi/`（surface_cache/screen_probe/spec_gi/world_restir/path_reuse/probe_volume/vxgi/global_sdf/caustics 等已落） | surface cache、屏幕探针、SHARC、ReSTIR 储层、反射 | 见 GI 文档 v4 |
| 时序 / 上采样 | `temporal_upscale/` `motion/` `history/` | motion vector、历史累积、reactive mask、上采样 | 全共享；NPR 锐利分段靠 reactive mask 保护 |
| 透明 | `transparency/` + `gi/oit/` | OIT 路径、HairVisibility 等 | 全共享 |
| 形变 / 子系统 | `deformation/` `hair/` `cloth/` `particle/` `water/` `volumetric/` | 蒙皮/morph/布料/毛发/粒子/水/顶点动画的形变预算与调度 | 全共享 |
| 材质 ABI | `material/` `abi/` | 正交轴 + 闭包 IR + über-BSDF/有界 slab | 三前端的统一材质表达 |
| 视图族 / 分帧 | `view_family/` `frame_graph/` `paging/` `work_graph/` | 多视图、帧图、页驻留、GPU 工作图 | 全共享 |

**规则**：任何“高级特性”先问一句——它是**基底服务**（放这里，全前端白嫖）还是**前端响应**（放前端，各自解读）。绝不让 NPR/PBR 各自重造 GI/阴影/几何。

---

## 2. 顶级产品对标（借形态，不抄码）

> 比管线 §12 更细：拆出“具体借哪一招 / 借到什么程度 / 不借什么”。

### 2.1 PBR 线（业界已收敛，抄作业到位即 AAA）

| 能力 | 首选对标 | 借什么 | 明确不借 |
|---|---|---|---|
| 虚拟几何 | UE5 Nanite | cluster DAG、软光栅、vis-buffer 形态 | 其具体数据布局/源码 |
| 全局光照 | UE5 Lumen + NVIDIA RTXGI/DDGI | SDF/mesh-card 软 RT + 硬 RT 混合、surface cache、屏幕探针、DDGI 探针 | Lumen 内部实现细节 |
| 多光源/采样 | NVIDIA ReSTIR DI/GI (RTXDI) + NRD | 储层时空重采样、ReBLUR/ReLAX 去噪思路 | — |
| 阴影 | UE5 VSM + RT 阴影 | 虚拟页 + clipmap、按驻留渲染 | — |
| 材质分层 | UE5 Substrate / OpenPBR | über-BSDF + slab 分层思想 → **收敛成有界 slab** | 无界 slab（刻意封顶防 variant 爆炸） |
| 上采样 | UE5 TSR / DLSS / FSR2 / XeSS | 时序上采样形态；接厂商 SDK 为可选后端 | 神经网络内核（本体走经典时序） |
| 参考渲染 | RED Engine Cyberpunk RT Overdrive | ReSTIR GI 路径追踪 + NRD 的离线对拍口径 | Ray Reconstruction（神经降噪） |

### 2.2 NPR 线（无现成招牌管线，最大增量也最高风险）

| 能力 | 首选对标 | 借什么 |
|---|---|---|
| 卡通着色 | miHoYo（原神/星铁）、Arc System Works（GG Xrd / DBFZ） | ramp/阶梯 Blinn、ID map + 顶点色控制、手编法线 |
| 面部阴影 | miHoYo / HoYo 系 | SDF 面部阴影图（独立数据、主光方向阈值切换）+ 阈值硬阴影染色 |
| 高光 | miHoYo / Arc Sys | 天使环各向异性高光带（与真实切线解耦）、MatCap、阶梯高光 |
| 描边 | Arc Sys / miHoYo / Borderlands | inverted-hull 背面挤出（顶点色控宽度）、屏幕空间深度/法线边、material id 边、墨线笔刷 |

### 2.3 前沿总表（v2 新增，§6 各赛道的对标源，均纯经典数值）

| 前沿赛道 | 首选对标 | 借什么形态 | 明确不借 |
|---|---|---|---|
| 几何置换/细分 | UE5.4 Nanite Tessellation + NVIDIA DMM | cluster 级程序化置换 + 位移微网格 LOD | 其数据布局/源码；神经几何压缩 |
| 多光源阴影 | UE5.5 MegaLights + NVIDIA SMRT | 统一海量光源 reservoir 直接光 + 光追软阴影核 | — |
| 混合反射 | Lumen 反射 + 随机 HiZ-SSR + glossy ReSTIR | 粗糙度分档 SSR→RT 升级 + 镜面储层重用 | 神经反射重建 |
| 能量守恒材质 | OpenPBR / Filament multiscatter GGX | 多散射能量补偿 + 布料 sheen + 车漆 flakes | — |
| 体积云/雾 | Decima/Nubis + Frostbite froxel + Volumetric ReSTIR | 时域重投影 + 蓝噪声步进 + froxel 储层 | 神经体积超分 |
| 透明 OIT | McGuire MBOIT + per-pixel linked list | 加权混合/链表 OIT + 折射 + hair visibility | — |
| 毛发 | Marschner + Zinke 双散射 + UE Groom | 物理发丝 BSDF + 双散射多散射能量补偿 + strand→card LOD | — |
| 水体/海洋 | Tessendorf FFT + Sea of Thieves/AC 海洋 | 频谱 FFT 海面 + 泡沫/湿润/焦散 | — |
| 后期影视 | ACES/AgX + Frostbite 物理镜头 | bokeh DoF/物理 bloom/tile 运动模糊/自动曝光 | 神经风格迁移 |
| 采样/降噪 | STBN(Heitz) + Owen-Sobol + NRD/A-SVGF | 时空蓝噪声 + 低差异 QMC + 时空双边降噪 | 神经降噪 / Ray Reconstruction |
| 上采样/AA | UE5 TSR + VRS | 经典时序上采样 + 可变着色率 | 神经上采样内核（SDK 仅可选后端） |
| 性能工程 | D3D12/Metal Work Graphs + NVIDIA SER + bindless | GPU 自驱调度 + 着色重排 + bindless 堆 | — |

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

## 6. 次世代前沿高级特性全景（v2 新增，纯经典数值 / 无 AI·ML·LLM）

> 本章把“到顶级次世代 AAA 天花板”所需的前沿特性拆成 12 条赛道。每条给**借鉴对象 / 算法要点 / 性能预算 / 效果上限 / 模块落点 / 验收口径**。GI 本体规格在 `prism_gi_lumen_design_zh.md`（v4），这里只给与其它赛道的衔接增量。所有条目均为经典数值，**不含任何神经/ML 内核**。

### 6.1 几何前沿 — 微多边形 + 程序化置换

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 微多边形软光栅 | UE5 Nanite | cluster DAG 连续 LOD + 64 像素级 tri 的 compute 软光栅 + 硬件栅格混合 | 与 draw 数解耦，vis-buffer 一趟；两趟 HiZ 遮挡剔除去背面/被遮 cluster | 像素级几何密度、无 LOD pop | `virtual_geometry/`（12 文件，骨架在，软光栅为最大出血点） | 任意距离无 pop、无裂缝；vis-buffer material id 正确 |
| 程序化置换/细分 | UE5.4 Nanite Tessellation + NVIDIA DMM | cluster 级自适应细分 + 位移贴图；位移微网格（DMM）做 LOD 压缩 | 仅可见 cluster 细分；置换预算随屏占自适应 | 近景置换细节（砖缝/岩面/地形）无需烘高模 | `virtual_geometry/` + `material/`（置换轴） | 置换边界不裂、与 VSM/RT 一致 |
| 两趟遮挡剔除 | UE Nanite HiZ | 上帧 HiZ 剔第一趟，渲染后重建 HiZ 剔第二趟补绘 | 剔除在 GPU，CPU 零往返 | 大遮挡场景 draw 大降 | `virtual_geometry/` + `geometry/` | 无错剔（可见物不丢）、无漏剔开销 |
| 蒙皮体素/SDF 代理 | UE Lumen 动态几何 | 蒙皮网格每帧体素化/SDF 更新，供 GI/DFAO/软阴影 | 低分辨率代理 + 增量更新 | 动态角色参与 GI/遮蔽不漏光 | `ray_scene/` + GI `global_sdf/` | 角色移动间接光/AO 跟随无延迟 |

### 6.2 阴影前沿 — 虚拟页 + 光追软阴影 + 海量光

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| VSM 深化 | UE5 Virtual Shadow Maps | 虚拟页 + clipmap，仅渲驻留页；页缓存复用静态 | residency 已落；增量重绘脏页 | 全程一致阴影密度，近不虚远不抖 | `virtual_shadow/`（7 文件） | 页抖动/缓存失配可量化为零 |
| 光追软阴影 | NVIDIA SMRT / RT Shadows + NRD | 面光锥内分层随机射线 + 时空去噪，物理半影 | RT 桶；少射线 + 去噪补 | 距离相关半影、接触硬+远软 | `ray_scene/` + GI `denoise/` | 半影宽度物理正确、无噪无带状 |
| 接触阴影 | 屏幕空间射线步进 | 深度缓冲短程 ray-march 补 VSM 近景漏接触 | 屏幕空间一趟，低成本 | 脚底/缝隙接触黑边自然 | `gi/distance_field_shadow/` + 屏空 | 无 peter-panning、无自阴影痤疮 |
| 海量光阴影 | UE5.5 MegaLights | 统一 reservoir 直接光，阴影与 GI 共享可见性查询，灯数与成本解耦 | 千级动态光下恒定预算 | 千级带阴影光源无逐灯 shadow map | `lighting/` + `gi/world_restir/` | 千灯低方差、无萤火虫、去噪不糊边 |
| 胶囊/面光软阴影 | 解析胶囊 + LTC 面光 | 角色用胶囊近似软阴影；面光用 LTC 解析 | 解析式，无射线 | 角色自阴影柔和、面光软边 | `gi/capsule_shadow/` `gi/area_light/` | 胶囊贴合骨架、面光能量守恒 |

### 6.3 反射前沿 — 分档混合 + 储层重用

> GI/反射本体见 GI 文档 §9.2；此处给**跨档升级与 glossy 储层**衔接增量。

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 粗糙度分档混合反射 | UE5 Lumen Reflections | 镜面→随机 HiZ-SSR→RT 升级，按粗糙度/置信度路由；SSR miss 落 RT/surface cache | 低粗糙度省 RT、高粗糙度走 SG 近似 | 清晰镜面 + 粗糙面各向异性连续 | `gi/spec_gi/` `gi/reflect/` `gi/planar_reflect/` | 与路径追踪参考对拍误差收敛 |
| 随机 HiZ-SSR | 随机屏空反射 | HiZ 加速步进 + 重要性采样 GGX + 时空去噪 | 屏幕空间，层级深度剔除 | 屏内反射无条纹、接触反射准 | `gi/reflect/` | 边缘淡出/屏外回退无突变 |
| glossy reservoir 重用 | ReSTIR 反射 | 镜面样本 reservoir 时空重用，少射线出低方差 glossy | 与 GI 储层共享基建 | 中粗糙镜面低噪、方向保真 | `gi/spec_gi/` `gi/spec_denoise/` | 1–2 spp glossy 收敛、无拖影 |
| 薄膜干涉/各向异性反射 | Belcour-Barla thin-film | 膜厚色散并入 GGX，各向异性切线帧响应 | 闭包内解析，无额外射线 | 肥皂膜/氧化金属/车漆彩虹 | `material/` + `gi/anisotropy/` | 色散随视角连续、能量守恒 |

### 6.4 材质前沿 — 能量守恒 über-BSDF

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 多散射能量补偿 GGX | Filament / OpenPBR | 高粗糙度多次散射能量回补，防变暗 | 查表/解析，极低 | 粗糙金属不发灰、能量守恒 | `gi/env_brdf/` `material/` | 白炉测试(white furnace)能量≈1 |
| 有界 slab 多瓣 | UE5 Substrate / OpenPBR | über-BSDF 一等 + 有界 slab（清漆/车漆/多层皮）封顶防 variant 爆炸 | shader 特化桶，封顶是刻意取舍 | 清漆/车漆/多层皮达标 | `material/` `abi/` | variant 受控、分层响应正确 |
| 布料 sheen / 车漆 flakes | Estevez-Kulla sheen + flake | sheen 边缘散射 BRDF；车漆双 clearcoat + 金属 flake 法线扰动 | 闭包内解析 | 丝绒/天鹅绒边缘光、车漆颗粒闪 | `gi/cloth/` `gi/clearcoat/` | 掠射边缘光/flake 闪随光移动 |
| SSS 三档 | Burley/可分离 + 屏空 + 路径追踪参考 | 低配可分离卷积、中配屏空、参考走路径追踪 diffusion | 屏幕空间一趟（主路径） | 皮肤/蜡/玉透光自然、边不发绿 | `gi/subsurface/` `gi/eye/` | 三档一致收敛、厚薄过渡自然 |
| 视差遮蔽/triplanar/贴花 | POM + 延迟贴花 | POM 自遮蔽 + 轮廓裁剪；triplanar 无缝；clustered 贴花 | 屏空/延迟一趟 | 砖缝/地形细节、无接缝贴花 | `gi/parallax/` `gi/triplanar/` `gi/decal/` | 掠射无拉伸、贴花混法线正确 |

### 6.5 体积前沿 — froxel + 云 + 储层

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| froxel 参与介质 | Frostbite 体积雾 | 相机对齐 froxel 散射/消光 + 时域累积 | froxel 分辨率可调，内存为主成本 | 体积光轴/高度雾无带状 | `volumetric/` `gi/fog/` `gi/light_shaft/` | 光轴/大气无带状、时域不闪 |
| Volumetric ReSTIR | Lin et al. | froxel 储层重用散射样本，介质获多弹跳间接光 | 复用 GI 储层基建 | 体积雾/云内间接光、方差大降 | `gi/volumetric_gi/` | 体积多弹跳低噪、无闪烁 |
| 光线步进体积云 | Decima Nubis + Schneider | 分形噪声密度场 + 蓝噪声步进 + 时域重投影复用 | 低分辨率 + 时域放大 | 实时演进云、自投影/光散射 | `gi/clouds/` | 时域重投影无鬼影、步进无带状 |
| 大气空中透视 | Hillaire sky LUT | 预计算透射/多散射 LUT + 空中透视体积 | LUT 一次，查表廉价 | 行星级大气、黄昏红移 | `gi/atmosphere/` `gi/sky_lut/` | 日照角连续、无 LUT 接缝 |

### 6.6 透明前沿 — 顺序无关 + 折射

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 加权混合 OIT | McGuire-Bavoil MBOIT | 深度加权累积，一趟近似顺序无关 | 一趟，无排序 | 多层半透明无排序闪烁 | `gi/oit/` `transparency/` | 层叠顺序视觉稳定 |
| 每像素链表 OIT | per-pixel linked list | 片段链表 + 精确排序合成，高配精确 | 带宽/内存重，高配档 | 精确多层透明/玻璃 | `transparency/` | 精确排序、无丢片 |
| 折射 | 屏空厚度折射 | 法线/厚度驱动屏空偏移 + 粗糙度模糊 | 屏空一趟 | 玻璃/液体折射、吸收着色 | `gi/refraction/` `gi/translucency/` | 折射方向物理、吸收 Beer 定律 |
| 发丝可见性 | UE HairVisibility | 发丝深度/覆盖 OIT，接 GI/阴影 | 与毛发子系统共享 | 毛发半透明层叠无排序错 | `transparency/` + `prism_hair_gpu` | 毛发边缘无硬切、覆盖正确 |

### 6.7 毛发/皮毛前沿 — 物理发丝 BSDF + 双散射

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| Marschner 发丝 BSDF | Marschner + Chiang | R/TT/TRT 三瓣物理发丝散射 | 闭包解析 | 高光环/透射辉光物理正确 | `gi/hair_bsdf/` + `prism_hair_gpu` | 高光环随光移动、金/深发色准 |
| 双散射多散射能量补偿 | Zinke-Weyrich dual scattering | 全局多散射近似 + 单散射，能量不丢 | 预计算散射表 + 实时查 | 浅色发/毛发体积通透不发黑 | `gi/hair_bsdf/` | 白炉测试能量守恒、浅发通透 |
| strand→card LOD | UE5 Groom | 近景发丝、远景卡片/网格连续过渡 | LOD 随屏占 | 远近无 pop、发量成本可控 | `prism_hair_gpu` | LOD 过渡无跳变 |
| 深度不透明图 | Deep Opacity Maps | 发丝自阴影分层深度 | 低分辨率分层 | 发丛自阴影柔和 | `prism_hair_gpu` + `virtual_shadow/` | 自阴影无痤疮、分层连续 |

### 6.8 水体/海洋前沿 — FFT 频谱海面

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| FFT 海洋 | Tessendorf + Sea of Thieves/AC | Phillips/JONSWAP 频谱 + IFFT 置换+法线，多级联拼接 | GPU FFT，级联数可调 | 真实海浪谱、远近无重复感 | `water/` | 浪谱统计正确、无平铺重复 |
| 水面反射/折射 | planar + 屏空 + RT 回退 | 近 planar/屏空、远 RT/surface cache | 分档，共享反射基建 | 镜面水/粗糙波面连续 | `water/` + `gi/planar_reflect/` | 反射与场景一致、无漏光 |
| 泡沫/湿润/焦散 | 开放世界水体经验 | Jacobian 泡沫 + 湿表面 BRDF 变暗提亮 + 焦散投射 | 焦散走 GI 焦散路径 | 浪尖泡沫、岸边湿痕、水下焦散 | `water/` + `gi/caustics/` | 泡沫随浪峰、焦散锐利无噪 |

### 6.9 后期/影视前沿 — 物理镜头 + 色调映射

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| bokeh 景深 | Frostbite 物理镜头 | 光圈形状散景 + 近/远场分离 + 聚集 | 半分辨率聚集 + 合成 | 影视散景、无硬边 | `gi/depth_of_field/` `gi/lens/` | 焦外过渡自然、无环状伪影 |
| tile 运动模糊 | McGuire tile MB | tile 最大速度 + 邻域重建 | 低分辨率 tile + 重建 | 快速运动平滑、无条带 | `gi/motion_blur/` `motion/` | 速度边界无撕裂、不糊静物 |
| 物理 bloom | Jimenez 下采样卷积 | 多级降采样高斯/FFT 卷积能量守恒辉光 | mip 金字塔 | 高光溢出自然、无方块 | `gi/light_shaft/` + bloom | 能量守恒、无网格伪影 |
| 自动曝光 + 色调映射 | 直方图测光 + ACES/AgX | GPU 直方图测光 + ACES/AgX 色调 + 局部色调映射 | 直方图一趟 | HDR 场景宽容度、无死黑死白 | `gi/local_tonemap/` `gi/color_grade/` | 明暗适应平滑、肤色不偏 |
| 色差/炫光/暗角/片grain | 物理镜头瑕疵 | 棱镜色散/光晕/渐晕/胶片颗粒，艺术可控 | 后处理叠加 | 影视镜头质感 | `gi/lens/` `gi/film_grain/` | 可独立开关、不过曝 |

### 6.10 采样/降噪前沿 — 低差异 + 时空双边

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 时空蓝噪声 | Heitz/Wolfe STBN | 噪声在时空“蓝”化，感知误差远优于白噪声 | 预生成 STBN 纹理 | 低 spp 感知噪声极低 | `gi/sample/` | 低 spp 下无结构性噪声 |
| Owen-scrambled Sobol | QMC 低差异 | 加扰 Sobol 序列做准蒙特卡洛积分 | 查表廉价 | 积分收敛快于白噪声 | `gi/sample/` `gi/nee/` | 收敛阶优于随机采样 |
| 时空双边降噪 | NRD ReBLUR/ReLAX + A-SVGF | 方差引导时空双边 + 历史 clamp + diffuse/specular 分离 | 多趟 à-trous | 1–2 spp 出收敛画面、不糊边 | `gi/denoise/` `gi/spec_denoise/` `gi/temporal/` | 无鬼影、无边缘糊化、无带状 |
| 萤火虫抑制/去遮挡 | firefly clamp + 历史修复 | 亮度钳位 + 去遮挡历史重建 | 低成本后处理 | 无高亮噪点、运动边缘干净 | `gi/denoise/` `history/` | 无萤火虫、去遮挡无拖尾 |

### 6.11 上采样/抗锯齿前沿 — 经典时序为本体

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| TSR 式时序上采样 | UE5 TSR | 低分辨率内部渲染 + 抖动 + 历史重投影累积放大，**纯经典** | 依赖 motion vector + reactive mask | 4K 级清晰度、运动无重影 | `temporal_upscale/` `motion/` `history/` | 4K 清晰度、无拖尾/鬼影 |
| 厂商 SDK 可选后端 | DLSS/FSR2/XeSS | 仅作外部后端接入，不进 CPU golden 路径 | 运行期切换 | 平台最优上采样 | `temporal_upscale/`（后端抽象） | 开关不影响本体 parity |
| 可变着色率 VRS | DX12/VK VRS | 按内容/速度/中心凹降着色率 | 省着色，边缘保全率 | 外围降率省算力、无可见劣化 | `gi/vrs/` | 感知无损、速度区降率正确 |
| 前向 MSAA | 硬件 MSAA | NPR/透明前向路径的几何抗锯齿兜底 | 前向桶 | 卡通硬边抗锯齿 | 前向前端 | 边缘无阶梯、与 TAA 不冲突 |

### 6.12 性能工程前沿 — GPU 自驱 + 相干

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| GPU Work Graphs | D3D12/Metal Work Graphs | GPU 自驱动态调度射线/卡片/探针，去 CPU 往返 | 负载自均衡 | 长短射线负载不塌陷 | `work_graph/` `frame_graph/` | 调度无 CPU 回环、吞吐稳定 |
| 着色执行重排 SER | NVIDIA SER | 硬件重排提升射线相干 | RT 桶可选 | 发散射线吞吐提升 | `ray_scene/`（能力探测） | 开启后吞吐升、结果不变 |
| bindless/描述符堆 | 现代 bindless | 全场景资源 bindless 寻址，去绑定开销 | 描述符堆常驻 | 海量材质/纹理零绑定切换 | `descriptor_heap/` `shader_package/` | 无绑定瓶颈、无越界 |
| 帧图别名 + async compute | frame graph aliasing | 瞬态资源别名复用 + 计算/图形异步重叠 | 显存与时间双省 | 显存占用降、空泡填满 | `frame_graph/` `virtual_resource/` | 别名无读写冲突、重叠无竞态 |
| mesh shader 卡片捕获 | mesh shader | mesh-shader 直出 Surface Cache 卡片/几何 | 剔除+放大在 GPU | 卡片捕获省几何遍历 | `geometry/` + GI `surface_cache/` | 卡片覆盖正确、无遗漏面 |

---
## 7. 帧时间预算（1440p→4K，60fps 目标档）

> 粗预算，用于取舍与验收基线；实际以 GPU profiling 为准（沙盒无 GPU，本表为设计目标非实测）。v2 的前沿特性大多摊进既有阶段（置换进几何、软阴影进阴影、储层反射进 GI/反射、体积云进体积），不新增独立大头。

| 阶段 | PBR 场景占比 | NPR 场景占比 | 混合场景 | 备注 |
|---|---|---|---|---|
| 虚拟几何 vis-buffer + 剔除（含置换/两趟 HiZ） | ~20% | ~20% | ~20% | 三者共享，几何成本一致 |
| GI（Lumen/surface cache/储层 摊销） | ~18% | ~10%（只收不发省） | ~16% | NPR 省在不弹射 |
| ReSTIR DI/GI + 去噪（含 MegaLights） | ~15% | ~8%（ramp 量化省积分） | ~13% | 光源储层共享 |
| VSM + 光追软阴影/接触阴影 | ~10% | ~10%（+SDF 面部小额） | ~10% | 深度页共享 |
| 反射（分档 SSR/RT + glossy 储层） | ~6% | ~3%（NPR 降级近似） | ~5% | 低粗糙度省 RT |
| 前端着色 | ~12%（BRDF 积分） | ~7%（ramp/SDF 省） | ~13%（tile 路由 + 特化） | 混合多出 tile 分类 |
| 体积/雾/云（froxel + 体积储层 + 云重投影） | ~7% | ~7% | ~7% | 共享 |
| 时序上采样（TSR + VRS） | ~7% | ~7%（+reactive mask 小额） | ~7% | 共享 |
| 后期（DoF/MB/bloom/tonemap/lens） | ~3% | ~3% | ~3% | 影视后期链 |
| 风格化后处理 | ~2% | ~9%（halftone/墨线/步进） | ~6% | NPR 后处理重 |
| 透明/其他 | ~5% | ~5% | ~5% | OIT/折射/发丝 |

**结论**：NPR 整体算力**低于** PBR（省在 GI 不弹射 + ramp 量化 + 着色简化），但吃美术管线与 reactive mask 复杂度；混合多出 tile 路由与前端特化，换来风格接缝连贯——**三线皆 AAA 的性价比最优点**。前沿特性靠**分档降级 + 储层/时域复用 + GPU 自驱调度**把成本压回预算内。

---

## 8. 效果验收口径（三线一致的“顶级”判据）

- **PBR**：与路径追踪参考模式离线对拍，误差可量化收敛；动态光照无烘焙、无萤火虫、无 LOD pop；反射/软阴影/SSS/置换均过各自白炉或 parity 门槛。
- **NPR**：达顶级二次元观感（原神/GG Xrd 级 ramp/描边/面部阴影）；运动下锐利分段不被 TAA 抹糊（reactive mask 生效）；降帧步进与背景混排不撕裂。
- **混合**：卡通角色置入写实场景，阴影方向/底光/间接光一致，无“贴纸感”；风格切换处像素级路由无接缝。
- **前沿特性专项**：多散射能量守恒过白炉测试（能量≈1）；毛发/云/水体通透与演进物理合理；降噪 1–2 spp 收敛无鬼影无糊边；上采样 4K 无拖尾；VRS 感知无损。
- **跨平台**：野心效果落 compute-可移植桶，可 CPU golden 对拍；RT 桶部分可测；raw-VK 桶严格隔离（见管线 §8.1）。**所有本体路径为纯经典数值，可确定性复现与对拍**。

---

## 9. 落点与路线图映射（v3 按仓内实况分级）

> 现状分级：✅ 已成规模（核心路径有真实现，深化/收敛阶段）/ 🟡 骨架在待深化 / ⬜ 待建。基于 2026-10 `pkg/` 实际模块：`prism_render_shading/src/gi/` 下 62 个子系统目录已落（crate 共 344 文件）（world_restir·surface_cache·screen_probe·spec_gi·path_reuse·probe_volume·vxgi·global_sdf·caustics·clouds·atmosphere·sky_lut·denoise·spec_denoise·gtao·subsurface·env_brdf·hair_bsdf·oit·refraction·parallax·triplanar·decal·vrs·local_tonemap·depth_of_field·motion_blur·film_grain·sample 等）；`prism_render_scene`(ray_scene) 456 文件（软件 BVH/TLAS/SDF 查询成规模）；虚拟几何由专用 crate `prism_virtual_geometry_gpu`（49 文件：cluster_cull/cluster_raster/payload_raster/lod_projection/lod_select/page_pool/page_storage/vis_payload_codec/triangle_gradients/frustum_cull 等）承载软光栅主路径。

| 特性块 | 主模块 | 现状 | 优先级 |
|---|---|---|---|
| 虚拟几何软光栅 + 置换 + 两趟剔除 | `prism_virtual_geometry_gpu`（49）+ `virtual_geometry/`（12） | ✅ 软光栅/页池/LOD/vis-codec 主路径已落，置换与两趟 HiZ 深化中 | P0 |
| GI/反射/ReSTIR 全家桶 | `gi/*`（62 子系统）`lighting/` | ✅ 多模块成规模，按 GI 文档 v4 收敛对拍 | P0 |
| VSM + 光追软阴影 + MegaLights | `virtual_shadow/`（7）`shadow/virtual_sm` `gi/distance_field_shadow/` `gi/capsule_shadow/` `gi/area_light/` | 🟡 VSM residency + 多级阴影在，MegaLights 海量光待深化 | P0 |
| 时序/上采样/reactive mask/VRS | `temporal_upscale/`（3）`motion/`（7）`history/`（4）`gi/vrs/` `taa/` `upscale/` | 🟡 时序累积+历史在，reactive mask 是 NPR 头号前置 | P0 |
| 材质 ABI / 有界 slab / 多散射 / SSS | `material/` `abi/` `gi/env_brdf/` `gi/subsurface/` `gi/eye/` `gi/clearcoat/` | 🟡 正交轴+闭包 IR 已定，多散射能量守恒白炉收敛待补 | P1 |
| 反射分档 + glossy 储层 + 薄膜 | `gi/spec_gi/` `gi/reflect/` `gi/planar_reflect/` `gi/anisotropy/` `gi/spec_denoise/` `screen_space/` | ✅ SSR/planar/各向异性/储层去噪在，跨档升级连线待收 | P1 |
| 体积雾/云 + 体积储层 | `volumetric/`（21）`prism_volumetric_gpu`（163）`gi/clouds/` `gi/fog/` `gi/volumetric_gi/` `gi/light_shaft/` `gi/atmosphere/` `gi/sky_lut/` | ✅ froxel/云/大气 LUT 成规模，体积储层时域复用深化中 | P1 |
| 透明 OIT + 折射 + 发丝可见性 | `transparency/` `gi/oit/` `gi/refraction/` `gi/translucency/` | 🟡 OIT/折射在，链表精确档+发丝可见性接线待深化 | P1 |
| 毛发 BSDF + 双散射 + LOD | `gi/hair_bsdf/` `prism_hair_gpu`（98）`hair_chiang/kajiya/marschner` 系 | ✅ 发丝 BSDF + strand 管线成规模，双散射能量/LOD 收敛中 | 并行推进 |
| 水体 FFT + 反射 + 泡沫/焦散 | `water/`（32：fft/flip/pbf/swe/foam/caustics/coupling/underwater/wetness/ocean_lod/breaking/shoreline） `gi/caustics/` | ✅ FFT 海面 + FLIP/PBF/浅水 + 泡沫/湿润/焦散/水下成规模 | P2 |
| 后期影视链（DoF/MB/bloom/tonemap/lens） | `gi/depth_of_field/` `gi/motion_blur/` `gi/local_tonemap/` `gi/lens/` `gi/film_grain/` `bloom/cas/dof/tonemap/vignette` | ✅ 多模块已落，接链与分辨率分档待梳 | P2 |
| 采样/降噪（STBN/Sobol/NRD/A-SVGF） | `gi/sample/` `gi/denoise/` `gi/spec_denoise/` `gi/temporal/` `gi/debanding/` | ✅ 多模块已落，diffuse/spec 分离与 clamp 对齐收敛中 | P1 |
| 性能工程（Work Graphs/SER/bindless/帧图别名） | `work_graph/`（5）`descriptor_heap/` `frame_graph/`（7）`virtual_resource/` `shader_package/` | 🟡 骨架在，GPU 自驱调度/SER 探测深化中 | P1 |
| NPR 专属响应（ramp/SDF/描边/后处理） | `stylized/outline/halftone/hatching/posterize/kuwahara/ordered_dither/face_shadow`（shading 已落）+ reactive mask 通道 | 🟡 后处理算子多数已落，reactive mask 保护链待连 | P1 |
| 混合 tile 路由 | `material/` + `light_routing/classification` + 前端 | 🟡 依赖 material id 基底，tile 分类雏形在 | P1 |

**总原则**：先立 P0 基底（几何/光照/阴影/时序），三前端才有共享数据可消费；NPR 的 reactive mask 属 P0 前置；反射/材质/体积/采样属 P1；水体/后期链属 P2；毛发子系统并行开发。**每条前沿特性落地走管线 §9 的“CPU golden → WESL kernel → 真机 parity”三步，未毕业不作生产默认。**

---

## 10. 子系统完成度矩阵 + 跨子系统集成 + 子系统 AAA 高级特性（v3 新增）

> 用户多轮追问「子系统完成度」。本节把渲染之外但构成 AAA 整机的六大子系统（物理 / 音频 / 水体 / 毛发 / 布料 / 体积）按仓内实况分级，并给**达到顶级次世代 AAA 所需的高级特性清单**（借鉴优先产品，纯经典数值，排除 AI/ML/LLM）。计数为 2026-10 `pkg/` 下 `*.rs` 文件数，用于规模参考（非完成度百分比）。

### 10.1 子系统完成度矩阵

| 子系统 | 主 crate / 规模 | 已落能力（真实模块） | 现状 | 距 AAA 的关键缺口 |
|---|---|---|---|---|
| 物理 | `prism_physics_gpu`（263）`prism_physics_core`（168）`prism_physics_geometry`（33） | GPU 刚体 TGS、XPBD、MPM、FLIP 流体、断裂（fracture）、软体 VBD、broadphase/BVH、narrowphase、CCD、island/sleep、radix/scan、关节 joint、陀螺项 | ✅ 广度已达 AAA 骨架 | 大规模破坏 islands 稳态、SDF/体素碰撞代理接 GI、确定性跨平台 parity 收敛 |
| 音频 | `prism_audio_core`（79）`prism_audio_spatial`（52）`prism_audio_hrtf`（9）`prism_audio_rt`（7）`prism_audio_device`（7） | DSP 节点图、空间化、HRTF 双耳、实时 ring 运行时、设备后端、effects（含 tilt_eq） | ✅ 核心链路成规模 | 光线声学遮挡/反射（ray-traced acoustics）、卷积混响 IR、与物理事件/材质耦合 |
| 水体 | `render_architecture/src/water/`（32） | FFT 频谱海面、FLIP/PBF 粒子流体、浅水 SWE、级联 LOD、泡沫/湿润/焦散/水下、破碎波、海岸线、与刚体 coupling | ✅ 成规模 | 大洋级联与近岸统一 LOD 过渡、GPU 全链路 parity、与体积雾/大气衔接 |
| 毛发 | `prism_hair_gpu`（98）`gi/hair_bsdf/` + shading `hair_*` | 发丝 BSDF（Chiang/Kajiya/Marschner/天使环/matcap）、strand 管线、线覆盖亚像素对拍 | ✅ 成规模 | 双散射多散射能量守恒白炉、strand→card LOD 无跳变、深度不透明图自阴影 |
| 布料 | `prism_render_scene/src/cloth/`（含 aero/backstop/ccd/embed/layers/plasticity/pressure/tearing/vbd/self_ccd/virtual GPU 对拍） | XPBD/VBD 布料、自碰撞 CCD、气动、撕裂、塑性、压强、多层、虚拟布料 | ✅ 成规模（GPU parity 密集） | 与毛发/刚体统一碰撞、风场耦合、LOD 降解 |
| 体积 | `prism_volumetric_gpu`（163）`render_architecture/src/volumetric/`（21）`gi/clouds·fog·atmosphere·sky_lut·volumetric_gi` | froxel 参与介质、光线步进云、大气 LUT、体积 GI、光轴 | ✅ 成规模 | 异质介质多弹跳储层收敛、云时域重投影无鬼影、体积阴影与 VSM 衔接 |

### 10.2 跨子系统集成（AAA 整机的「接缝」才是天花板）

> 单子系统达标不等于整机 AAA；次世代观感差距大量来自**子系统耦合**。以下集成点为纯经典数据流，无 ML。

| 集成点 | 借鉴对象 | 数据流 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|
| 物理→GI 动态几何 | UE Lumen 动态几何 | 刚体/布料/断裂碎块每帧体素化/SDF 增量更新喂 global_sdf / DFAO / 软阴影 | 动态物体参与间接光/遮蔽不漏光 | `prism_physics_gpu` + `gi/global_sdf/` | 碎块移动间接光/AO 跟随无延迟 |
| 物理→音频事件 | 物理驱动程序化音频 | 碰撞冲量/断裂事件触发音源参数（能量→响度/频谱） | 破坏/撞击声随物理强度连续 | `prism_physics_core/events` + `prism_audio_core` | 事件延迟 < 1 帧、无声画错位 |
| 音频→光线声学 | NVIDIA/Steam Audio 几何声学 | 复用渲染 BVH 做声线遮挡/早反射/混响 IR（低频射线预算） | 遮挡闷音、空间混响随几何 | `prism_audio_rt` + `ray_scene/`（共享 BVH） | 遮挡/混响随场景几何、CPU golden 可对拍 |
| 水体↔刚体/布料 | 开放世界水体耦合 | 水面高度场→浮力/拖拽；物体→溅射/尾迹注入水面 | 浮船/涉水/落水物理一致 | `water/coupling` + `prism_physics_gpu` | 浮力守恒、尾迹随速度、无穿插 |
| 毛发/布料↔风场 | 统一风场服务 | 共享风场/湍流场驱动毛发、布料、粒子、植被 | 风下毛发/布料/叶片一致摆动 | `particle/wind_field` + `prism_hair_gpu` + cloth | 同风场多子系统方向一致、无相位错 |
| 水体/云→大气散射 | Hillaire 大气 | 水面/云消光并入空中透视体积，共享 sky LUT | 远海/云海与大气连续、黄昏红移一致 | `water/` `gi/clouds/` + `gi/atmosphere/` | 透视连续、无 LUT 接缝 |

### 10.3 子系统 AAA 高级特性清单（借鉴优先产品，纯经典数值）

> 以下为把各子系统从「成规模」推到「顶级次世代 AAA」的高级特性，逐条给借鉴对象 + 算法要点 + 预算 + 落点 + 验收。

#### 物理
| 能力 | 借鉴对象 | 算法要点 | 预算 | 落点 | 验收 |
|---|---|---|---|---|---|
| 大规模破坏/碎裂 | Chaos / Houdini RBD | 预断裂 Voronoi 碎块 + 岛屿睡眠 + 约束弱化连接 | 岛屿级激活，静态碎块休眠 | `prism_physics_gpu/fracture` + `island` | 千碎块稳态不抖、连接断裂物理合理 |
| GPU 刚体大批次 | PhysX GPU / TGS Soft | TGS 软约束 + GPU broadphase/narrowphase 全驻留 | 万级刚体恒定帧预算 | `prism_physics_gpu/rigid` `contacts` | 万刚体堆叠稳定、无穿透无抖动 |
| 统一粒子流体 | FLIP/PBF (Macklin) | FLIP+PBF 混合，网格压力 + 粒子平流 | GPU 网格可调分辨率 | `prism_physics_gpu/fluid` `mpm` | 不可压约束守恒、无体积丢失 |
| 有限元软体/肌肉 | VBD / Projective Dynamics | VBD 顶点块下降，稳定大步长 | 迭代数可调 | `prism_physics_core/vbd` `soft` | 大形变稳定、能量不爆 |
| 连续碰撞 CCD | 保守前进 CCD | TOI 保守步进防隧穿，高速物体可靠 | 仅高速对象启用 | `prism_physics_core/ccd` | 高速物体无穿透、无卡顿 |

#### 音频
| 能力 | 借鉴对象 | 算法要点 | 预算 | 落点 | 验收 |
|---|---|---|---|---|---|
| 几何光线声学 | Steam Audio / NVIDIA ACE(非ML部分) | 复用渲染 BVH 投声线算遮挡/早反射/后期混响 | 低射线数 + 时域平滑 | `prism_audio_rt` + `ray_scene/` | 遮挡闷音、反射随几何、无爆音 |
| HRTF 双耳 | 公开 HRTF 数据集 + 球谐插值 | 方向 → HRTF 卷积 + SH 平滑插值去离散跳变 | 分块卷积 | `prism_audio_hrtf` | 方位/仰角定位准、转头无咔哒 |
| 卷积混响 | 分区卷积混响 | 空间 IR 分区 FFT 卷积，低延迟长尾 | 分块 FFT | `prism_audio_core/nodes` | 尾混自然、低延迟、无周期伪影 |
| 遮挡/传播 | 门户/房间传播 | 房间-门户图传播衰减与低通 | 图遍历廉价 | `prism_audio_spatial` | 隔墙闷、门缝漏声方向正确 |

#### 水体 / 毛发 / 布料 / 体积（补强项）
| 子系统 | 高级特性 | 借鉴对象 | 要点 | 落点 | 验收 |
|---|---|---|---|---|---|
| 水体 | 级联统一 LOD | Sea of Thieves | 远 FFT 大洋 ↔ 近 FLIP 交互无缝过渡 | `water/ocean_lod` `transition` | 过渡无接缝/无突变 |
| 水体 | 波破碎/飞沫 | AC 海洋 | Jacobian 判破碎 + 飞沫粒子注入 | `water/breaking` `foam` | 浪尖破碎物理、飞沫随浪峰 |
| 毛发 | 双散射能量守恒 | Zinke dual scattering | 全局多散射近似 + 单散射补 | `gi/hair_bsdf/` | 浅发通透、白炉能量≈1 |
| 毛发 | strand→card LOD | UE Groom | 近发丝远卡片连续过渡 | `prism_hair_gpu` | LOD 过渡无 pop |
| 布料 | 统一碰撞场 | — | 布料/毛发/刚体共享 SDF 碰撞代理 | cloth + `physics_gpu/bvh` | 多子系统无穿插 |
| 体积 | 异质介质储层 | Volumetric ReSTIR | froxel 储层重用散射样本多弹跳 | `gi/volumetric_gi/` | 体积内间接光低噪无闪 |
| 体积 | 云时域重投影 | Decima Nubis | 低分辨率 + 蓝噪声步进 + 时域放大 | `gi/clouds/` | 时域无鬼影、步进无带状 |

**§10 验收总纲**：每个子系统高级特性走与渲染一致的「CPU golden → GPU kernel → 真机 parity」三步；跨子系统集成点额外要求**确定性数据流**（物理→GI/音频、水体→物理）可逐位复现对拍；**全程纯经典数值，任何子系统均不引入 AI/ML/LLM 推理路径**（厂商上采样 SDK 仅渲染侧可选后端，与子系统本体无关）。

---

## 11. 次世代天花板增补（v4 新增，2024–2026 前沿，纯经典数值 / 无 AI·ML·LLM）

> §6 已覆盖 12 条主流前沿赛道。本章补齐 §6/§9 里**仍是空白或仅骨架、却已是 2024–2026 实时 AAA 天花板**的增量赛道。每条给**借鉴对象 / 算法要点 / 性能预算 / 效果上限 / 模块落点 / 验收口径**，与前文一致。全部为经典数值算法，**不含任何神经/ML 内核**；厂商上采样 SDK 仅作渲染侧可选外部后端，不进入本体默认路径。

### 11.1 全局光照天花板增补

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 辐射级联 Radiance Cascades | Path of Exile 2 / A. Sannikov | 多级同心辐射探针场：近处高角分辨率短射线、远处低角分辨率长射线，按级联合并，penumbra 条件天然满足；可做 2D/3D | 级联数 × 每级探针数可调；长射线少、短射线密，访存连贯 | 近似无偏软 GI/软阴影，收敛快、噪声极低 | `gi/radiance_cascades/`（待建）+ 复用 `global_sdf/` 步进 | 与路径追踪参考对拍能量收敛；相机移动无级联接缝/无闪 |
| 无限弹射辐射缓存 | DDGI 无限弹射 + SHARC | 探针/哈希缓存把上帧辐照再喂入本帧采样，几何级数逼近多弹射 | 复用现有探针/`surface_cache`，仅加一次反馈采样 | 室内多弹射不发黑、能量充足 | `gi/probe_volume/` `gi/surface_cache/` | 关灯/开灯多帧收敛稳定、无能量爆或塌陷 |
| 世界空间辐照哈希 SHARC | NVIDIA SHARC | 世界空间体素哈希缓存辐照，跨帧跨像素复用，摊薄 RT | 哈希表常驻显存，带宽为主成本 | 大场景 RT 预算摊平、远景间接光稳定 | `gi/world_restir/` + 哈希缓存模块 | 哈希碰撞/失配可量化、复用无漏光 |

### 11.2 几何 / RT 加速结构天花板增补

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 簇级 RT 加速结构 | NVIDIA RTX Mega Geometry（CLAS）| 为虚拟几何 cluster 构建**簇级 BLAS**，增量刷新 LOD 变化的簇，让 Nanite 级几何可被硬件/软件 RT 命中 | 仅重建变化簇，摊销 BVH 刷新；内存为主成本 | 虚拟几何直接参与 RT 反射/阴影/GI，无需代理网格 | `prism_virtual_geometry_gpu` + `ray_scene/`（BLAS 增量） | RT 命中与光栅 vis-buffer 几何一致、刷新无卡顿 |
| 不透明微贴图 OMM | NVIDIA Opacity Micromaps | 把 alpha-test 掩码预烘成微三角不透明/透明/未知三态，RT 遍历跳过多数 any-hit 着色 | 预烘一次，遍历期省 any-hit | 植被/铁丝网/树叶 RT 阴影/反射无 any-hit 爆炸 | `ray_scene/` + `material/`（alpha 掩码烘焙） | 掩码三态正确、RT 结果与光栅 alpha-test 一致 |
| 虚拟几何蒙皮/植被 | UE5.5 Nanite Skinned + 植被 | cluster LOD 支持骨骼蒙皮与实例化植被；蒙皮在簇级做，LOD 随屏占 | 蒙皮簇动态刷新，静态植被走实例 | 角色/森林亿级三角无 pop、可进 RT/GI | `prism_virtual_geometry_gpu` + `geometry/skinning` | 蒙皮簇 LOD 无裂缝、与软阴影/RT 一致 |

### 11.3 着色 / 纹理天花板增补

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 解耦/物体空间着色 | 纹理空间着色 / 解耦着色率 | 在物体 UV/纹理空间着色并跨帧复用，几何采样与着色解耦，可低频着色高频几何 | 着色率可调，摊销重着色；需 shading atlas | 高分辨率/VR/高频边下着色成本大降、时序稳定 | `texture_space_shading/`（待建）+ `frame_graph/` | 复用无过期伪影、运动下无糊/无鬼影 |
| 镜面/法线抗锯齿 | Toksvig / LEAN / Kaplanyan 法线方差 | 由法线贴图/几何法线方差推粗糙度加宽，抑制高光闪烁 | 预计算或在线一次，极低 | 远处金属/高光边无爬行闪烁 | `material/` + `gi/env_brdf/` | 高光闪烁可量化下降、不过度变糊 |
| 运行期虚拟纹理 RVT | UE Runtime Virtual Texturing | 地形/贴花/材质合成烘进虚拟纹理页，运行期按需驻留采样 | 页缓存按需驻留，带宽摊销 | 地形层混合/大量贴花零重复着色 | `virtual_texture/`（待建）+ `gi/decal/` | 页驻留无抖动、合成与直算一致 |

### 11.4 系统 / 延迟天花板增补

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 输入延迟流水线 | NVIDIA Reflex 形态（纯经典）| CPU 排队节流 + 渲染提交对齐 present，削减输入到显示延迟，无帧生成 | 调度逻辑，零画质成本 | 低输入延迟、帧节奏稳定 | `frame_graph/` + 提交调度 | 延迟可量化下降、无卡顿/无撕裂 |
| 自适应分辨率缩放 | 动态分辨率 DRS | 按帧预算在线调内部渲染分辨率，接 TSR 上采样回 4K | 预算驱动，开销可忽略 | 重载不掉帧、画质平滑降解 | `temporal_upscale/` + 预算控制器 | 分辨率切换无跳变、与 TSR 协同无拖影 |

> **§11 验收总纲**：每条增补特性仍走管线「CPU golden → GPU kernel → 真机 parity」三步；辐射级联/簇级 BLAS/OMM 必须与**路径追踪参考模式**或**光栅 vis-buffer**可逐位对拍；解耦着色与 RVT 的复用路径必须有**过期/失配可量化**门槛。全部为经典数值，任何增补均不引入 AI/ML/LLM 推理路径。

### 11.5 增补特性路线图优先级

| 增补特性 | 落点 | 现状 | 优先级 | 依赖 |
|---|---|---|---|---|
| 辐射级联 Radiance Cascades | `gi/radiance_cascades/`（待建） | ⬜ 待建（复用 `global_sdf/` 步进基建） | P1 | 全局 SDF、降噪 |
| 簇级 RT 加速结构（CLAS 形态） | `prism_virtual_geometry_gpu` + `ray_scene/` | 🟡 虚拟几何/软件 BVH 在，簇级 BLAS 增量待建 | P1 | 虚拟几何、ray_scene |
| 不透明微贴图 OMM | `ray_scene/` + `material/` | ⬜ 待建 | P2 | RT 遍历、alpha 掩码 |
| 虚拟几何蒙皮/植被 | `prism_virtual_geometry_gpu` + `geometry/skinning` | 🟡 静态虚拟几何成规模，蒙皮簇待深化 | P1 | 虚拟几何、蒙皮 |
| 解耦/物体空间着色 | `texture_space_shading/`（待建） | ⬜ 待建 | P2 | 帧图、shading atlas |
| 镜面/法线抗锯齿 | `material/` `gi/env_brdf/` | 🟡 env_brdf 在，法线方差链待连 | P1 | 材质闭包 |
| 运行期虚拟纹理 RVT | `virtual_texture/`（待建） | ⬜ 待建 | P2 | 贴花、材质合成 |
| 输入延迟 + 动态分辨率 | `frame_graph/` `temporal_upscale/` | 🟡 帧图/时序在，延迟流水线+DRS 待连 | P1 | 帧图、时序上采样 |

**增补总原则**：§11 增补**不改变 §9 的 P0 基底次序**——先立几何/光照/阴影/时序，再叠增补。辐射级联与簇级 BLAS 是「GI 保真」与「RT 覆盖面」两条最高价值增补，优先于 OMM/RVT/解耦着色这类摊销优化。全部增补遵循数值红线，未毕业不作生产默认。

---

## 12. 2025–2026 最前沿天花板增补（v5 新增，纯经典数值 / 无 AI·ML·LLM）

> §6 覆盖 12 条主流赛道、§11 补 8 条 2024–2026 增量。本章补齐**仍未展开、却已是当代影视级 + 实时 AAA 天花板**的赛道，重点在三块长期被实时引擎忽视、但顶级产品已标配的方向：**物理正确的色彩/相机/材质收敛标准**、**多光与异质介质的无偏保真**、**GPU 自驱的资源流送**。每条给 借鉴对象 / 算法要点 / 性能预算 / 效果上限 / 模块落点 / 验收口径。全部经典数值，**不含任何神经/ML 内核**；厂商上采样 SDK 仅渲染侧可选后端。

### 12.1 材质 / 色彩 / 相机 — 物理正确的收敛标准

> 「顶级次世代 AAA」与「很不错」的差距，近半来自**色彩管线与材质标准的物理正确性**。本节把材质对标从「有界 slab」升级到业界 2024 收敛标准 OpenPBR，并补齐光谱域精度与影视级显示变换。

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| OpenPBR 收敛标准材质 | OpenPBR Surface（Adobe/Autodesk 2024）| 把 §3.5 有界 slab 的瓣参数对齐 OpenPBR 分层语义（base/specular/coat/sheen/fuzz/thin-film/emission），统一能量守恒分层与导入导出；美术参数跨 DCC 一致 | 与现 über-BSDF 同桶，仅参数语义层 | 跨 DCC/离线对拍一致、分层能量不溢出 | `material/` `abi/` `gi/env_brdf/` `gi/clearcoat/` | 导入导出 round-trip 一致、白炉能量≈1、与 Arnold/Cycles OpenPBR 参考对拍 |
| 光谱渲染 / Hero 波长 | Weta Manuka / Arnold spectral | 关键路径（色散/薄膜/荧光）走 Hero wavelength 多波长采样 + 光谱→RGB 上采样（Jakob-Hanika），RGB 反照率升维到光谱 | 仅 dispersive 材质启用，4 波长 hero | 棱镜/钻石色散、薄膜彩虹物理正确、无 RGB 偏色 | `gi/sample/` + `material/`（spectral 轴） | 色散角与参考一致、能量守恒、CPU golden 可对拍 |
| 薄膜虹彩 | Belcour-Barla 2017 | 薄膜干涉解析 BSDF（膜厚→相位→可见光谱干涉），并入 coat 瓣，复用光谱上采样 | 解析式，一次求值 | 肥皂泡/氧化金属/甲虫壳虹彩随视角连续 | `gi/clearcoat/` `material/` | 色相随视角/膜厚物理正确、无带状 |
| ACES 2.0 / AgX / OCIO 显示变换 | ACES 2.0 · AgX（Blender）· OpenColorIO | 可插拔显示变换：ACES 2.0 RRT+ODT / AgX（高光去偏色）/ OCIO 配置；HDR（PQ/HLG）与 SDR 共管线；§6.9 local tonemap 作其前级 | LUT + 解析，极低 | 高动态高饱和不偏色/不断层、HDR 显示正确 | `gi/local_tonemap/` `gi/color_grade/` `gamut_map.rs` | 与参考变换逐像素对拍、HDR/SDR 一致观感、无色域裁剪硬边 |
| 物理相机 + 自动曝光 | 物理相机（EV/光圈/ISO/快门）| 曝光由光圈·快门·ISO 推 EV；直方图/测光自动曝光 + 时域平滑；驱动 §6.9 DoF 光圈与运动模糊快门角 | 直方图一趟 + 平滑 | 明暗适应自然、DoF/MB 与曝光物理一致 | `gi/local_tonemap/` + `exposure.rs` + `gi/depth_of_field/` `gi/motion_blur/` | 曝光响应物理正确、无骤变/呼吸、与 DoF/MB 参数联动一致 |

### 12.2 全局光照保真增补 — 多光 + 异质介质无偏

> §6.2/§6.3 与 GI 文档已覆盖 ReSTIR/MegaLights/反射本体。本节补两类**保真度上限决定项**：成千上万光源的无偏重要性采样，与异质介质（云/烟/浑水）的无偏体积积分。

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 光树 / 多光重要性采样 | Cycles Light Tree · Alan Wake 2 · Estevez-Kulla 2018 | 光源建层级 BVH（含方向锥/能量界），按重要性随机下探选光，为 ReSTIR/MegaLights 提供低方差初始候选 | 层级遍历 O(log N)，CPU/GPU 均可 | 万级光源低方差、无逐灯枚举 | `gi/light/` `gi/nee/` + `gi/world_restir/` | 等样本方差显著低于均匀选光、与参考收敛一致 |
| ReGIR 世界网格储层 | ReGIR（Reservoir Grid Importance Resampling）| 世界空间网格每格缓存光源储层，着色点查格取候选再 RIS，和 light tree/ReSTIR 级联 | 网格储层常驻，低显存 | 大场景多光二次重要性采样、时域稳定 | `gi/world_restir/` `gi/world_space/` | 网格无接缝、与逐点 NEE 对拍收敛、无闪烁 |
| 世界空间哈希辐照缓存 | AMD GI-1.0 / Radiance Cache · 空间哈希辐照 | 世界坐标空间哈希格缓存多弹跳辐照，命中直接取、未命中补算并回填，近似无限弹跳 | 哈希格常驻 + 增量更新 | 廉价多弹跳底光、与 surface cache 互补 | `gi/surface_cache/` `gi/world_space/` `gi/global_sdf/` | 哈希命中率/过期可量化、与路径追踪底光对拍 |
| 异质介质体积路径追踪 | delta tracking / ratio tracking / spectral tracking | 异质介质（云/烟/浑浊水）无偏自由程采样（delta tracking）+ ratio tracking 透射率估计，复用 §6.5 froxel 为重要性引导 | 仅参考/高配档启用，低 spp + 去噪 | 云/烟多弹跳通透、无分层带状、物理透射 | `gi/volumetric_gi/` `gi/clouds/` `gi/fog/` | 透射率无偏、与解析均匀介质对拍、能量守恒 |
| 物理天空多重散射 | Hillaire 2020 多散射 LUT · Wilkie-Hosek 光谱天空 | 大气多重散射 LUT（去单散射偏暗）+ 可选光谱天空模型；接 §10.2 水/云透视 | 预计算 LUT + 一次查表 | 蓝天/黄昏/地平线红移物理正确、晴空不偏暗 | `gi/atmosphere/` `gi/sky_lut/` | 与参考大气对拍、黄昏红移连续、无 LUT 接缝 |

### 12.3 纹理 / 资源流送 — GPU 自驱

> 顶级开放世界的「无限细节」靠的是**流送而非常驻**。§11 的 RVT 是合成侧，本节补**驻留决策与解压**侧——GPU 访问驱动的纹理驻留与 GPU 侧解压流送，是「海量资产零卡顿」的天花板。

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| Sampler Feedback 纹理流送 | DX12 Sampler Feedback Streaming (SFS) | GPU 回报实际采样到的 mip/tile，按真实访问驱动 tiled-resource 页驻留，只载入可见细节 | 反馈缓冲低带宽 + 异步页载 | 纹理显存大降、无 mip pop/无过载 | `virtual_texture/`（待建）+ `descriptor_heap/` + 流送控制器 | 页驻留命中率可量化、无可见 pop、显存不溢出 |
| GPU 解压流送 | DirectStorage · GDeflate | 压缩资产直送 GPU，GPU 侧并行解压（GDeflate），绕开 CPU 解压瓶颈 | GPU 解压占计算小额，I/O 并行 | 高速穿行无加载卡顿、流式贴图/几何 | `frame_graph/` + 资产流送后端 | 解压吞吐达盘速、无主线程阻塞、无瞬时掉帧 |
| 可见性缓冲延迟纹理化 + 材质分箱 | 延迟纹理化（deferred texturing）· material binning | vis-buffer 后按 material id 分箱，整箱同材质 wave 一致着色，解耦几何采样与着色，天然相干 | 分箱一趟 + 相干着色 | 海量材质零绑定切换、着色相干率高 | `prism_render_scene`(ray_scene)/visibility + `classification.rs` + `shader_package/` | 分箱覆盖完整、wave 相干率升、与前向结果一致 |

### 12.4 透明 / 合成增补

| 能力 | 借鉴对象 | 算法要点 | 性能预算 | 效果上限 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 矩不变 OIT (MBOIT) | Moment-Based OIT (Münstermann 2018) | 用幂矩/三角矩重建透射率曲线做顺序无关透明，内存定长、比链表省 | 定长矩缓冲，无链表堆 | 大量半透明层无排序伪影、低显存 | `gi/oit/` `transparency/` | 与精确排序对拍误差有界、无闪烁、内存定长 |
| 随机透明 | stochastic transparency | 随机掩码 + MSAA/时域累积解透明，和 TAA/上采样协同 | 复用 MSAA/时域 | 发丝/树叶/薄纱半透明无重排序 | `gi/oit/` + `taa/` | 收敛后无噪、与参考 alpha 一致、运动无鬼影 |
| 双深度剥离参考 | dual depth peeling | 离线/校准档逐层精确剥离作 OIT 参考基准 | 多趟，仅参考档 | 为实时 OIT 提供可对拍 ground truth | `gi/oit/`（参考路径） | 层序精确、作为 MBOIT/随机透明的 parity 基准 |

### 12.5 子系统天花板增补 — 物理 / 音频（纯经典数值）

> 承 §10.3，补物理与音频两条**与渲染同级的 AAA 天花板**赛道：波动声学与高阶 Ambisonics（音频），跨平台确定性与统一碰撞场（物理）。

| 子系统 | 能力 | 借鉴对象 | 算法要点 | 预算 | 落点 | 验收 |
|---|---|---|---|---|---|---|
| 音频 | 波动声学（低频） | Project Acoustics 形态（ARD/FDTD 预计算）| 离线 ARD/FDTD 预计算场景声场参数（遮挡/混响/衰减），运行期按位置插值；与 §10.3 几何声学分频段互补（低频波动、中高频射线）| 预计算烘焙 + 运行期查表 | `prism_audio_spatial` + 预计算烘焙 | 衍射/闷音低频物理正确、与几何声学衔接无突变 |
| 音频 | 高阶 Ambisonics + 近场 HRTF | HOA（1–3 阶）+ 近场补偿 HRTF | 场景声场编码到 HOA 总线，解码到 HRTF 双耳 + 近场距离补偿，转头/近声源定位稳定 | 总线编解码定长 | `prism_audio_spatial` + `prism_audio_hrtf` | 全向定位稳定、近场响度/视差正确、转头无相位跳 |
| 物理 | 跨平台确定性收敛 | 定点/定序确定性物理 | 固定迭代序 + 稳定求和（Kahan/定序归约）+ 可选定点关键路径，保跨平台逐位复现 | 仅约束求解序约束，开销可忽略 | `prism_physics_gpu/*` `prism_physics_core/*` | 多平台逐位一致、回放确定、与 CPU golden 对拍 |
| 物理 | 统一碰撞场（刚体/布料/毛发/流体）| 统一 SDF/体素碰撞代理 | 刚体/布料/毛发/粒子共享一套 SDF 碰撞代理（复用 `prism_physics_geometry` 网格查询族：sphere/capsule contacts·cast），消除子系统两两穿插 | 共享代理，增量更新 | `prism_physics_geometry` + `prism_physics_gpu/bvh` + cloth/hair | 多子系统无穿插、接触一致、确定性对拍 |

### 12.6 v5 增补路线图优先级

| 增补特性 | 落点 | 现状 | 优先级 | 依赖 |
|---|---|---|---|---|
| OpenPBR 参数语义收敛 | `material/` `abi/` | 🟡 über-BSDF/有界 slab 在，OpenPBR 对齐待连 | P0 | 材质闭包 IR |
| ACES 2.0 / AgX / OCIO + 物理相机 | `gi/local_tonemap/` `gi/color_grade/` `exposure.rs` | 🟡 local tonemap/color grade 在，ACES2/AgX/OCIO 变换待接 | P0 | 后期链、曝光 |
| 光树 / ReGIR 多光重要性采样 | `gi/light/` `gi/nee/` `gi/world_restir/` | 🟡 ReSTIR/world_restir 在，light tree/ReGIR 待建 | P1 | NEE、ReSTIR |
| 世界空间哈希辐照缓存 | `gi/surface_cache/` `gi/world_space/` | 🟡 surface cache 在，哈希辐照缓存待建 | P1 | 全局 SDF、surface cache |
| 光谱渲染 / Hero 波长 + 薄膜虹彩 | `gi/sample/` `material/` `gi/clearcoat/` | ⬜ 待建（仅 dispersive/薄膜启用） | P2 | 采样器、材质 |
| 异质介质体积路径追踪 | `gi/volumetric_gi/` `gi/clouds/` | 🟡 froxel/云在，delta/ratio tracking 参考路径待建 | P1 | 体积、降噪 |
| 物理天空多重散射 | `gi/atmosphere/` `gi/sky_lut/` | 🟡 sky LUT 在，多散射 LUT 待补 | P1 | 大气 |
| Sampler Feedback 流送 + GPU 解压 | `virtual_texture/`（待建）+ `frame_graph/` | ⬜ 待建 | P2 | 虚拟纹理、帧图 |
| 延迟纹理化 + 材质分箱 | ray_scene/visibility + `classification.rs` | 🟡 分类雏形在，分箱延迟着色待连 | P1 | vis-buffer、材质 |
| 矩不变 OIT / 随机透明 | `gi/oit/` `transparency/` `taa/` | 🟡 OIT 在，MBOIT/随机透明待深化 | P2 | 透明、时域 |
| 波动声学 + 高阶 Ambisonics | `prism_audio_spatial` `prism_audio_hrtf` | 🟡 空间化/HRTF 在，波动声学/HOA 待建 | P1 | 音频空间化 |
| 物理确定性 + 统一碰撞场 | `prism_physics_*` `prism_physics_geometry` | 🟡 网格查询族成规模，确定性收敛/统一代理待连 | P1 | 物理几何查询 |

**v5 增补总原则**：§12 **不改变 §9 P0 基底次序**，而是把「已达 AAA 骨架」推向「物理正确的顶级天花板」。最高价值两条为 **OpenPBR 收敛 + ACES2/AgX/OCIO 色彩管线**（决定全画面物理正确性与跨工具一致性，列 P0）与 **光树/ReGIR + 哈希辐照缓存 + 异质介质体积路径追踪**（决定多光与介质保真上限，列 P1）。光谱渲染、SFS/GPU 解压、MBOIT 为按需摊销增量（P2）。全部遵循数值红线：走「CPU golden → GPU kernel → 真机 parity」三步，关键路径必须与**路径追踪参考**或**解析解**可对拍，未毕业不作生产默认；**任何增补均不引入 AI/ML/神经/LLM 推理路径**。

---

## 附：与管线文档的关系
- 顶层架构 / 决策 / 后端策略 / 三桶可测性 / 代码重构清单：见 `prism_material_pipeline_design_zh.md`。
- 全局光照 / 反射 / 采样降噪 深水区：见 `prism_gi_lumen_design_zh.md`（v4）。
- 毛发子系统：见 `prism_hair_engine_design_zh.md`。粒子：见 `prism_particle_engine_design_zh.md`。物理：见 `prism_physics_design_zh.md`。体积：见 `prism_volumetric_engine_design_zh.md`。水体：见 `prism_water_engine_design_zh.md`。布料：见 `prism_cloth_engine_design_zh.md`。音频：见 `prism_audio_engine_design_zh.md`。
- 本文只负责“三前端各自与共享的高级特性目录 + 前沿特性全景 + 预算 + 验收”。
