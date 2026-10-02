# Prism 渲染引擎 — 顶级次世代 AAA 高级特性专项设计（v8 / PBR·NPR·混合 全前端 + 子系统完成度 + 路线图兑现核对 + 真实空白补齐 + v7 绝对天花板 + v8 产品对标高级特性增补与规模三刷 + v9 产品对标高级特性增补与规模四刷 + v10 产品对标高级特性增补与规模五刷 + v11 产品对标高级特性增补与规模六刷 + 顶级 AAA 毕业门禁收口）

> 本文是 `prism_material_pipeline_design_zh.md`（顶层架构与决策）的**下钻分册**：把 §12–16 的对标矩阵与特性清单展开为**逐特性规格书**——每条给出「借鉴对象 / 算法要点 / 性能预算 / 效果上限 / 模块落点 / 验收口径」。
> **一句话立场**：共享 GPU-driven 基底算一次，PBR / NPR / 自定义三前端并存消费，混合在管线级路由。**三条赛道都是一等公民，都能拿到顶级次世代 AAA 效果**，差异只在「怎么解读同一份光/影/GI/几何数据」的前端响应处。
> **纪律**：只借鉴公开算法与形态，**不本地拉取任何产品源码**（UE 算法已获授权，同样只借形态）。
> **数值红线（硬约束）**：所有高级特性走**纯经典数值路径**（蒙特卡洛/准蒙特卡洛、SH/SG、reservoir 重采样、SDF 步进、时空双边降噪、时空蓝噪声、FFT）。**不引入任何 AI / ML / 神经网络 / LLM 路径**——不做神经降噪、Ray Reconstruction、神经辐射缓存、神经上采样、神经材质压缩。厂商时序上采样 SDK（DLSS/FSR2/XeSS）仅作**可选外部后端**接入，本体默认路径为纯经典 TSR 式时序累积，保证 CPU golden 可对拍、跨平台可移植。
> **v2 本版新增**：在 v1（§1–§8，三前端共享基底 + PBR/NPR/混合逐特性规格）之上，追加 **§6「次世代前沿高级特性全景」**——覆盖几何 / 阴影 / 反射 / 材质 / 体积 / 透明 / 毛发 / 水体 / 后期影视 / 采样降噪 / 上采样抗锯齿 / 性能工程 12 条前沿赛道，对标最新实时 AAA 天花板（UE5.6 / Cyberpunk RT Overdrive / Alan Wake 2 / Portal RTX / Horizon / Nanite Tessellation / MegaLights），逐条给算法要点 + 预算 + 落点 + 验收；并刷新 §2 对标矩阵（§2.3 前沿总表）与 §9 落点路线图（按仓内现状分级）。
> **v3 本版新增**：按 2026-10 仓内实况刷新 §9 分级与文件计数（虚拟几何/GI/水体/体积/采样降噪/后期多项由 🟡 升 ✅，依据各 `pkg/` 子目录真实规模）；新增 **§10「子系统完成度矩阵 + 跨子系统集成 + 子系统 AAA 高级特性」**，把物理（GPU 刚体 TGS / XPBD / MPM / FLIP / 断裂 / 软体 VBD）、音频（HRTF 双耳 / 光线声学 / 空间化）、水体（FFT 海面 / FLIP / 浅水）、毛发、布料、体积六大子系统的现状、缺口、及达到 AAA 所需高级特性逐条展开，**全程纯经典数值，排除所有 AI/ML/LLM 路径**。
> **v4 本版新增**：① 按 2026-10 仓内实况再刷新规模计数（物理 GPU 188 / 音频 core 67 / 毛发 GPU 41 / `ray_scene` 82 文件）；② 新增 **§11「次世代天花板增补（2024–2026 前沿，纯经典数值）」**——补齐当前仍是空白或仅骨架的最前沿赛道：辐射级联（Radiance Cascades, PoE2 / Sannikov）、RT 簇级加速结构（RTX Mega Geometry 形态，让虚拟几何可被硬件 RT）、不透明微贴图（Opacity Micromaps）、解耦/物体空间着色（texture-space shading）、镜面/法线抗锯齿（Toksvig/LEAN）、运行期虚拟纹理（RVT）、虚拟几何蒙皮/植被、输入延迟流水线（Reflex 形态，纯经典）；逐条给借鉴对象 + 算法要点 + 预算 + 落点 + 验收；并把这些增补并入 §9 路线图优先级。**全程纯经典数值，排除一切 AI/ML/神经/LLM 路径**（厂商时序上采样 SDK 仅渲染侧可选外部后端）。
> **v5 本版新增（2026-10）**：① 按仓内实况再刷新规模计数（`prism_render_shading` 344 / `gi/` 62 子系统 · `prism_render_scene`(ray_scene) 456 · `prism_render_architecture` 671 · 虚拟几何 GPU 49 · 体积 GPU 163 · 毛发 GPU 98 · 物理 GPU 263 / core 168 / geometry 33 · 音频 core 79 / spatial 52 文件）；② 新增 **§12「2025–2026 最前沿天花板增补（纯经典数值）」**——补齐 §6/§11 仍未展开、却已是当代影视/实时 AAA 天花板的赛道：**OpenPBR 收敛标准材质 · 光谱渲染/Hero 波长 · 薄膜虹彩（Belcour-Barla）· ACES 2.0/AgX/OCIO 显示变换 + 物理相机自动曝光 · 光树/ReGIR 多光重要性采样 · 世界空间哈希辐照缓存 · 异质介质体积路径追踪（delta/ratio tracking）· 物理天空多重散射 · Sampler Feedback 纹理流送 · DirectStorage/GDeflate GPU 解压流送 · 可见性缓冲延迟纹理化+材质分箱 · 矩不变 OIT · 波动声学/高阶 Ambisonics · 跨平台确定性收敛**；逐条给借鉴对象+算法要点+预算+落点+验收，并给 §12 增补路线图优先级。**全程纯经典数值，排除一切 AI/ML/神经/LLM 推理路径**（厂商时序上采样 SDK 仅渲染侧可选外部后端）。
> **v6 本版新增（2026-10）**：① **路线图兑现核对**——按仓内实测，v4/v5 列为 ⬜/🟡 的多条前沿已落为真实模块（LTC 面光 `gi/area_light`、路径重用 `gi/path_reuse`、RT 焦散 `gi/caustics`、光树/ReGIR `gi/light`+`gi/nee`、世界空间哈希辐照缓存 `gi/world_space`、镜面 AA `gi/specular_aa`、DDGI 重定位 `gi/irradiance_volume`、IES `gi/ies_profile`、微遮蔽 `gi/micro`、音频波动声学/HOA/卷积混响 `prism_audio_spatial`+`prism_audio_core/reverb`），逐条给文件证据并升级分级；② **真实空白补齐 + 收敛深化**——新增 **§13**：对**仓内 0 文件实测**的真实空白（稀疏虚拟纹理 RVT/SVT + Sampler Feedback、DirectStorage/GDeflate GPU 解压流送、经典帧生成〈无神经〉、仿射体动力学 ABD）给规格，并把已有骨架（GRIS 全路径重用、LTC 纹理化/线光面光、自适应方差制导采样、RT 焦散自适应光子）推向绝对天花板；给 §13 兑现核对表 + 增补路线图优先级。**全程纯经典数值，排除一切 AI/ML/神经/LLM 推理路径**（厂商时序上采样/帧生成 SDK 仅渲染侧可选外部后端）。
> **v7 本版新增（2026-10）**：① 按「代码有更新」用户反馈，用统一口径 `find src -name '*.rs'` **重算规模并修正 v6 §13.0 偏高计数**（新增权威基线表 §14.0）；② 空白赛道再核验（§14.1，RVT/SVT·DirectStorage·GDeflate·frame_gen·ABD·micromap 仍 0 文件，维持 ⬜；仅毛发 LSS 段切分孪生动工）；③ 新增 **§14** 八条 2024–2026 真·前沿赛道：**随机纹理过滤 STF · 硬件曲线/发丝 RT（LSS）· 微几何硬件 RT（DMM+OMM）· 可编程光栅 WPO/蒙皮虚拟几何 · 分层 GI 融合（屏幕探针+世界缓存）· 整帧 GPU Work Graph（mesh nodes）· 物理 IPC 无穿透摩擦接触+ABD · 预计算波场参数解码声学**；逐条给借鉴对象+算法要点+性能预算+效果上限+落点+验收，并并入 §14.3 优先级阶梯。**全程纯经典数值，排除一切 AI/ML/神经/LLM 推理路径**（厂商时序上采样/帧生成 SDK 仅渲染侧可选外部后端）。
> **v8 本版新增（2026-10）**：按「代码有更新」四刷规模（统一口径），在 v7 §14 八条之上追加 **§15** 六条产品级赛道（MegaLights 无界阴影光 · 可见性缓冲延迟材质 · 动态 BVH 实时重建 · 薄膜干涉光谱色散 · Nanite 置换 tessellation 收口 · 实时光线声学+UTD 衍射），逐条实测分级并并入 §15.2 优先级。**全程纯经典数值，排除一切 AI/ML/神经/LLM 推理路径**。
> **v9 本版新增（2026-10）**：再次按「代码有更新」用统一口径 **四刷规模**（§16.0，新增 `physics_gpu`+14/`volumetric_gpu`+7/`hair_gpu`+4/`render_visibility`+2 与 UI 三 crate 登记），并在 v8 §15 六条之上追加 **§16** 六条产品级赛道：**稀疏虚拟纹理 SVT+GPU 反馈缓冲 · Mesh Shader/meshlet 硬件放大路径 · FFT 频谱海洋+Gerstner+破碎泡沫 · 矩不变 OIT+混合折射 · 延迟贴花 DBuffer+网格贴花 · GPU 蒙皮缓存+WPO 几何馈入 Nanite/RT BLAS**；逐条经关键字实测分级（🟡/⬜）给借鉴对象+算法要点+性能预算+效果上限+落点+验收，并并入 §16.2 优先级（**蒙皮缓存 / SVT 列 P1**，作为 v7/v8 动态世界特性的使能器）。**全程纯经典数值，排除一切 AI/ML/神经/LLM 推理路径**（厂商时序上采样/帧生成 SDK 仅渲染侧可选外部后端）。
> **v10 本版新增（2026-10）**：延续用户「代码有更新 / 借鉴参考优先产品 / 添加高级功能 / 兼顾性能与效果 / 达到顶级次世代 AAA 级别 / 更新优化设计文档」。本版 ① 以统一口径 `find pkg/<crate>/src -name '*.rs' | wc -l` **五刷规模**（§17.0，接续 v9 §16.0 基线，实测反映 `physics_gpu`+15/`volumetric_gpu`+12/`hair_gpu`+5/`audio_core`+5/`render_scene`+2 的子系统孪生增量）；② 在 v9 §16 六条之上追加 **§17** 六条产品级赛道：**统一次表面散射 SSS（Burley 可分离 + 随机游走 BSSRDF）· 可变速率着色 VRS Tier2 内容自适应 · GTAO 弯曲法线环境光遮蔽 + 镜面遮蔽 · 大气天空 + 空中透视 + Nubis 体积云 · 开阔世界流式 VHM 虚拟高度场 + Nanite 植被散布 + World Partition/HLOD · 毛发/体积深阴影图统一半透射自阴影**；逐条经关键字实测分级（🟡/⬜）给借鉴对象 + 算法要点 + 性能预算 + 效果上限 + 落点 + 验收，并并入 §17.2 优先级（**SSS / GTAO 列 P1**，皮肤次表面与接触遮蔽是「近景人物与接触可信度」刚需）。**全程纯经典数值，排除一切 AI/ML/神经/LLM 推理路径**（厂商时序上采样/帧生成 SDK 仅渲染侧可选外部后端）。
> **v11 本版新增（2026-10）**：延续用户「代码有更新 / 借鉴参考优先产品 / 添加高级功能 / 兼顾性能与效果 / 达到顶级次世代 AAA 级别 / 更新优化设计文档」。本版 ① 以统一口径 `find pkg/<crate>/src -name '*.rs' | wc -l` **六刷规模**（§18.0，接续 v10 §17.0 基线，实测反映 `prism_physics_gpu`+3 / `prism_hair_gpu`+2 / `prism_audio_core`+1 的子系统孪生增量，并首次单列 `prism_render_shading/gi/` 子系统已达 **246 文件**）；② 在 v10 §17 六条之上追加 **§18.1** 三条经关键字实测确认为**真实空白 / 仅骨架**的新赛道——**着色器 PSO 预编译与管线缓存（消除运行期编译卡顿，stutter-free）· 可见性缓冲 / RT 光线微分纹理 LOD（ray cone / ray differentials 抗纹理走样）· GPU 驱动粒子渲染整合（GPU 排序 + mesh/ribbon 粒子馈入 VisBuffer/OIT/motion vector + 软粒子深度淡出）**；③ 新增 **§18.2「顶级次世代 AAA 毕业门禁与收口」**——把 §3–§17 已登记的 P1/P1.5 特性收敛为一张**交付就绪度（Definition of Done）+ 收口次序 + 帧级顶级判据**矩阵，定义「怎么证明到达顶级 AAA」而非再堆特性；④ 刷新 §18.3 路线图优先级。**全程纯经典数值，排除一切 AI/ML/神经/LLM 推理路径**（厂商时序上采样 / 帧生成 SDK 仅渲染侧可选外部后端，本体默认经典路径，关键路径与路径追踪参考 / 解析解 / 离线数值求解可对拍）。

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
| 镜面/法线抗锯齿 | `material/` `gi/env_brdf/` `gi/specular_aa/` | 🟡 env_brdf 在；几何镜面 AA（Tokuyoshi–Kaplanyan 2019）CPU golden + SSR repack 已接；主通道 resolve（需常驻法线 G-buffer）与 Toksvig 法线贴图 AA 待连 | P1 | 材质闭包 |
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

## 13. v6 顶级天花板增补与路线图兑现核对（2026-10，纯经典数值 / 无 AI·ML·LLM）

> **本版双目的**：① **兑现核对**——按 2026-10 仓内实测，v4/v5 列为 ⬜/🟡 的多条前沿已落为真实模块（给文件证据，升级分级）；② **补齐真实空白 + 收敛深化**——只对**仓内 0 文件实测**的真实空白赛道新增规格，并把**已有骨架**推向绝对天花板，达顶级次世代 AAA。**全程纯经典数值，排除一切 AI/ML/神经/LLM 推理路径**（厂商时序上采样/帧生成 SDK 仅渲染侧可选外部后端，本体走经典路径，CPU golden 可对拍）。

### 13.0 规模计数刷新（2026-10 实测 `*.rs` 文件数）

| crate / 子系统 | v5 记载 | 当前实测 | 增量说明 |
|---|---|---|---|
| `prism_render_shading`（`gi/` 62 子系统） | 344 | **344** | 内部深化为主，gi 子系统已达 62 |
| `prism_render_scene`（ray_scene） | 456 | **458** | 软件 BVH/TLAS/SDF 查询继续收敛 |
| `prism_render_architecture` | 671 | **680** | 帧图/流送/view_family 深化 |
| `prism_virtual_geometry_gpu` | 49 | **49** | 软光栅主路径稳定 |
| `prism_volumetric_gpu` | 163 | **191** | 体积储层/异质介质路径扩张 |
| `prism_hair_gpu` | 98 | **108** | rt_proxy/hero_wavelengths/双散射深化 |
| `prism_physics_gpu` / `core` / `geometry` | 263 / 168 / 33 | **275 / 183 / 34** | 流体/断裂/VBD/CCD/几何流形深化 |
| `prism_audio_core` / `spatial` | 79 / 52 | **83 / 52** | 卷积混响/房间声学深化 |

### 13.1 v4/v5 路线图兑现核对（已从 ⬜/🟡 升级，附文件证据）

> 用户多轮指出「代码有更新」。下表按**真实目录证据**把 v4/v5 roadmap 中原记 ⬜/🟡 的赛道升级，证明「天花板增补」已大面积从设计落为实现，v6 不再把它们当空白。

| 赛道 | v4/v5 记载 | 当前实况（文件证据） | 升级后 |
|---|---|---|---|
| LTC 解析面光 | 未单列 / 🟡 area_light | `gi/area_light/{ltc_lut.rs, polygon.rs, shapes.rs}` | ✅ 多边形 LTC 已落 |
| ReSTIR PT / 路径重用 | 🟡 path_reuse 待连 | `gi/path_reuse/{path_reservoir.rs, shift_map.rs, vertex.rs}` | ✅ 路径储层+重投影骨架已落 |
| RT 焦散（manifold/光子） | 🟡 caustics | `gi/caustics/{photon.rs, manifold_nee.rs, density_estimate.rs}` | ✅ 光子+流形 NEE+密度估计已落 |
| 光树 / ReGIR 多光重采样 | 🟡→P1 待建 | `gi/light/{light_tree.rs, regir.rs}` + `gi/nee/{mis.rs, ris.rs, light_sampling.rs}` | ✅ 光树+ReGIR+MIS/RIS 已落 |
| 世界空间哈希辐照缓存 | 🟡→P1 待建 | `gi/world_space/{radiance_cache.rs, spherical_gaussian.rs, octahedral.rs, probe_placement.rs, probe_interpolation.rs, visibility.rs}` | ✅ 哈希辐照缓存已落 |
| 镜面/法线抗锯齿（Toksvig/LEAN） | §11 🟡 | `gi/specular_aa/{toksvig.rs, lean.rs, normal_variance.rs}` | ✅ 已落 |
| DDGI 探针重定位 | 🟡 探针兜底 | `gi/irradiance_volume/{ddgi_probe.rs, relocation.rs, visibility.rs}` | ✅ 重定位+可见性已落 |
| IES 光度配光 | 未单列 | `gi/ies_profile/{grid.rs, normalize.rs, symmetry.rs}` | ✅ 已落 |
| 微遮蔽（micro-occlusion） | 未单列 | `gi/micro/micro_occlusion.rs` | ✅ 已落 |
| 波动声学 / 高阶 Ambisonics / 衍射 / 卷积混响 | §12 🟡 待建 | `prism_audio_spatial`（52：`ambisonics/hoa/hoa_decode/hoa_rotation/diffraction/room_modes/portal_graph/scattering/material_library/early_reflections/convex_room/outdoor_propagation` 等）+ `prism_audio_core/reverb/convolver.rs` | ✅ 房间声学/HOA/衍射/卷积混响成规模 |
| 物理软体/流体/断裂/VBD/CCD | §10 🟡 | `prism_physics_core/{soft, fluid, mpm, fracture, vbd, ccd, reduced, island, sleep}` | ✅ 子系统成规模 |

**核对结论**：v4/v5 的「天花板增补」清单已**大面积兑现**；渲染正向、GI、面光、焦散、多光重采样、世界空间缓存、镜面 AA、音频波动声学/HOA、物理软体/流体均已有真实模块。v6 的新增聚焦于**仓内 0 文件实测的真实空白**（§13.2）与**已有骨架推向天花板的收敛深化**（§13.3）。

### 13.2 真实空白赛道（本版新增规格，仓内 0 文件实测 → ⬜）

> 以下赛道经 `find pkg -iname` 实测**当前为 0 文件**（`virtual_texture`/`sampler_feedback`/`directstorage`/`gdeflate`/`frame_gen` 均 0；物理 `abd`/`affine` 0）。它们是达到顶级次世代 AAA 仍缺的最后拼图，逐条给规格。

#### 13.2.1 稀疏虚拟纹理运行期（RVT / SVT）+ Sampler Feedback 流送
- **借鉴对象**：id Tech MegaTexture / UE Runtime Virtual Texture / D3D12 Sampler Feedback + Tiled Resources。
- **算法要点**：全场景纹理内容虚拟化为页表（间接纹理 + 物理页池）；着色器通过 Sampler Feedback 回写**本帧实际被采样的 mip/页**，驱动按需页加载与驻留回收；地形/贴花/程序化材质烘到 RVT 复用，着色与 texel 驻留解耦。
- **预算**：间接纹理一次采样 + 物理页池（可配，典型 256–1024MB）；feedback 回写走 UAV，按 tile 聚合，带宽占比 <2%。
- **效果**：零可见纹理 pop、巨幅世界恒定显存、贴花/地形混合一次烘定复用。
- **落点**：新建 `prism_render_architecture/src/virtual_texture/`（页表/物理页池/feedback 聚合）+ `prism_render_architecture/src/texture_streaming/`（已存在，接驻留调度）+ shading 侧间接采样 helper。
- **验收**：高速移动相机下无 mip/页 pop；物理页池满载时 LRU 回收无抖动；feedback 预算与真机 parity 对拍。

#### 13.2.2 DirectStorage / GDeflate GPU 解压直通流送
- **借鉴对象**：Microsoft DirectStorage 1.2 + GDeflate GPU 解压（NVIDIA RTX IO 形态）。
- **算法要点**：资源以 GDeflate 块压缩落盘，IO 请求批量提交，解压在 GPU compute 完成，绕过 CPU 解压与多余拷贝；与 §13.2.1 RVT 页加载、虚拟几何页池、BVH 流送共用一套异步流送队列。
- **预算**：解压 compute 分帧摊销（预算如 <0.3ms/帧），IO 队列深度可配；无平台支持时回退 CPU 多线程解压路径（功能一致、吞吐降级）。
- **效果**：开放世界瞬时跳转/传送无加载卡顿，流送吞吐数 GB/s 级。
- **落点**：`prism_render_architecture/src/{texture_streaming, paging, memory}/` + 新建 `storage/`（IO 队列抽象 + GDeflate 解压 kernel + CPU 回退）。
- **验收**：传送/跳转无长卡顿；GPU 解压与 CPU 回退逐块 bit 一致（经典可对拍）；队列背压下无死锁。

#### 13.2.3 经典帧生成（运动矢量重投影 + 遮挡修复，无神经，可选）
- **借鉴对象**：AMD FSR3 Frame Generation 的**经典核**（光流/运动矢量 + 遮挡遮罩 + UI 剥离），**不含**其任何神经部分。
- **算法要点**：以相邻两渲染帧 + 稠密运动矢量（已有 `screen_space/motion.rs`/`reconstruct.rs` 可复用）做前后双向重投影生成中间帧；遮挡/去遮挡区按运动一致性检测并以邻域填补；HUD/UI 分层剥离后合成避免鬼影；与 §Reflex 形态延迟流水线协同以抵消插帧延迟。**默认关闭、可选**。
- **预算**：生成一帧 compute（1440p 目标 <2ms），运动矢量复用零额外几何开销；延迟预算显式计入（插帧必然 +0.5–1 帧延迟，用低延迟流水线补偿）。
- **效果**：显示帧率翻倍级平滑度提升，运动流畅；**纯经典、CPU 可对拍参考帧**。
- **落点**：新建 `prism_render_shading/src/frame_gen/`（双向重投影/遮挡修复/UI 剥离）复用 `screen_space/motion.rs`、`history/`、`taa/jitter.rs`；厂商 SDK 作可选外部后端。
- **验收**：快速平移/转身无撕裂/鬼影；UI 无抖动；关闭时零开销；延迟预算符合低延迟档；与经典参考帧误差有界。

#### 13.2.4 仿射体动力学 ABD（刚体-软体统一，物理子系统）
- **借鉴对象**：Affine Body Dynamics（Lan et al. 2022）+ IPC 无穿透接触，统一到现有 VBD/XPBD 求解框架。
- **算法要点**：以每体 12 自由度仿射场代替纯刚体 6 自由度，近刚体用高刚度仿射能量约束，天然统一刚体/近刚体软体/关节；与现有 `physics_core/{vbd, soft, ccd}` 共用一套障碍接触与 CCD，消除刚体与软体两套接触代码的接缝。
- **预算**：每体 12 DOF（刚体 6 DOF 的 2×），接触走已有 CCD/几何查询族；确定性定序求解（与现有确定性收敛约定一致）。
- **效果**：刚体/软体/关节统一表达，大刚度下不抖、接触无穿透、与软体耦合无接缝。
- **落点**：新建 `prism_physics_core/src/abd/`（仿射体 + 能量约束 + 接触耦合），复用 `ccd/`、`collide/`、`prism_physics_geometry` 查询族，接入 `solver/`。
- **验收**：高刚度仿射体退化为刚体行为（与刚体解对拍）；刚-软耦合无穿插；跨平台确定性逐位复现。

### 13.3 前沿收敛深化（已有骨架 → 推向绝对天花板）

> 这些赛道已有真实骨架（§13.1），v6 给出**把骨架推向顶级天花板**的收敛规格。

#### 13.3.1 GRIS 广义储层全路径重用（`path_reuse/` 深化）
- **借鉴对象**：Generalized Resampled Importance Sampling（Lin et al. 2022, SIGGRAPH）/ ReSTIR PT。
- **算法要点**：在现有 `path_reservoir.rs` + `shift_map.rs` 之上落**广义平衡启发式 MIS** 与**无偏雅可比校正**：相邻/历史像素路径经 reconnection shift 重连并以 GRIS Jacobian 加权，时空双重用全路径（含多次漫反射弹射），而非仅末端 DI/GI。
- **预算**：路径储层 + shift 重连（每像素 1–2 候选），与现有去噪协同；偏差校正纯解析。
- **效果**：多弹射间接光低方差无偏，暗角/焦散路径收敛显著快于逐帧 PT。
- **落点**：`gi/path_reuse/{path_reservoir.rs, shift_map.rs, vertex.rs}` + `gi/nee/mis.rs`（MIS 权重）+ `gi/denoise/`。
- **验收**：与离线 PT 参考在等时预算下方差更低且无系统性偏差（白炉/等效能量对拍）。

#### 13.3.2 LTC 面光纹理化 + 线光/管光 + 多散射补偿（`gi/area_light/` 深化）
- **借鉴对象**：Heitz et al. Linearly Transformed Cosines（含纹理化面光、线光/管光解析）。
- **算法要点**：现有 `ltc_lut.rs`/`polygon.rs` 扩到**纹理化多边形光**（LTC 对纹理 mip 的解析积分）、**线光/管光**（LTC 边界积分闭式）、以及与 §多散射 GGX 能量补偿衔接，保高粗糙度面光不丢能量。
- **预算**：LTC LUT 一次采样 + 多边形边积分 O(边数)，线/管光闭式；无额外采样噪声。
- **效果**：霓虹/灯管/屏幕等纹理化/线状光源解析高光，无噪声、无需 RT。
- **落点**：`gi/area_light/{ltc_lut.rs, polygon.rs, shapes.rs}` + `gi/env_brdf/`（多散射补偿）。
- **验收**：与 RT 参考面光高光形状/能量一致；纹理化面光 mip 过渡平滑；高粗糙度能量守恒。

#### 13.3.3 自适应方差制导采样 + 停机准则（与 `gi/denoise`/`gi/temporal` 协同）
- **借鉴对象**：Adaptive sampling（Dammertz / A-SVGF 时域方差）+ ReSTIR confidence weights。
- **算法要点**：以时域方差 + 储层置信度估计每像素收敛度，动态分配 spp 预算，收敛像素早停、噪声像素加采；与现有时空去噪的方差估计复用同一通道。
- **预算**：方差估计复用去噪缓冲，无额外全屏 pass；采样预算有上限封顶，防长尾。
- **效果**：等总预算下边缘/高频区更干净，平坦区省算力，整体方差下降。
- **落点**：`gi/sample/`（预算分配）+ `gi/temporal/` + `gi/denoise/`/`gi/spec_denoise/`（方差源）。
- **验收**：固定总 spp 预算下，相对均匀采样方差更低；无过曝/欠采闪烁；预算上限不被突破。

#### 13.3.4 RT 焦散自适应光子 + manifold NEE 收敛（`gi/caustics/` 深化）
- **借鉴对象**：Specular Manifold Sampling（Zeltner et al.）+ 自适应光子密度估计。
- **算法要点**：在现有 `photon.rs`/`manifold_nee.rs`/`density_estimate.rs` 上落**自适应核半径**（按局部光子密度收缩）与**流形 NEE 连接**（镜面-漫反射-光源链），聚焦水下/玻璃焦散的无噪尖锐光斑。
- **预算**：光子图分帧累积 + 自适应核；流形连接每焦散像素少量牛顿迭代。
- **效果**：水下/玻璃/金属焦散尖锐无噪，与体积/水体子系统衔接。
- **落点**：`gi/caustics/{photon.rs, manifold_nee.rs, density_estimate.rs}` + `water/caustics` + `gi/refraction/`。
- **验收**：焦散光斑与离线光子映射参考形状一致、无能量泄漏；动态光下时域稳定不闪。

### 13.4 v6 增补路线图优先级

| 增补特性 | 落点 | 现状（实测） | 优先级 | 依赖 |
|---|---|---|---|---|
| 稀疏虚拟纹理 RVT/SVT + Sampler Feedback | `virtual_texture/`（0，待建）+ `texture_streaming/` | ⬜ | P1 | 页池、帧图、流送 |
| DirectStorage/GDeflate GPU 解压流送 | `storage/`（待建）+ `paging/memory/` | ⬜ | P1 | 异步流送队列 |
| 经典帧生成（可选，无神经） | `frame_gen/`（待建）复用 `motion/history/taa` | ⬜ | P2 | 运动矢量、低延迟流水线 |
| 仿射体动力学 ABD | `prism_physics_core/abd/`（0，待建） | ⬜ | P2 | VBD/CCD/几何查询族 |
| GRIS 广义储层全路径重用 | `gi/path_reuse/` + `gi/nee/mis.rs` | 🟡 骨架在 | P1 | 路径储层、去噪 |
| LTC 面光纹理化 + 线光/管光 | `gi/area_light/` + `gi/env_brdf/` | 🟡 多边形在 | P1 | 面光、多散射 |
| 自适应方差制导采样 + 停机 | `gi/sample/` + `gi/temporal/denoise/` | 🟡 采样器在 | P1 | 时空方差、储层置信度 |
| RT 焦散自适应光子 + manifold NEE | `gi/caustics/` + `water/caustics` | 🟡 骨架在 | P2 | 光子图、折射、水体 |

**v6 增补总原则**：① v6 **不改变 §9 P0 基底次序**，也不重复 §11/§12 已兑现条目；② 真实空白（§13.2，仓内 0 文件实测）列为新规格，其中 **RVT/SVT + DirectStorage 流送**是开放世界恒定显存与零加载卡顿的最后短板（P1），**ABD 与经典帧生成**为按需增量（P2）；③ 已有骨架（§13.3）走收敛深化，**GRIS 全路径重用 / LTC 纹理化面光 / 自适应采样**价值最高（P1）。全部遵循数值红线：走「CPU golden → GPU kernel → 真机 parity」三步，关键路径必须与**路径追踪参考**或**解析解**可对拍，未毕业不作生产默认；**任何增补均不引入 AI/ML/神经/LLM 推理路径**（厂商时序上采样/帧生成 SDK 仅渲染侧可选外部后端）。

---

## 14. v7 绝对天花板增补与规模再刷新（2026-10，纯经典数值 / 无 AI·ML·神经·LLM）

> **本版双目的**：① 用户再次指出「代码有更新」，按 2026-10 实测 `find pkg/<crate>/src -name '*.rs'` **重算规模并修正 v6 §13.0 的记载**（v6 部分子目录计数偏高，本版以统一口径为准）；② 在 v6 §13 的空白补齐与收敛深化之上，追加 **8 条 2024–2026 真·前沿赛道**——这些是把 Prism 从「功能齐备」推到「顶级次世代 AAA 天花板」仍差的最后增量，覆盖**纹理采样 / 硬件曲线 RT / 微几何 RT / 可编程光栅动画几何 / 分层 GI 融合 / 整帧 GPU 自驱 / 物理无穿透接触 / 波场声学**。逐条给借鉴对象 + 算法要点 + 性能预算 + 效果上限 + 模块落点 + 验收口径。**全程纯经典数值，排除一切 AI/ML/神经/LLM 推理路径**；厂商时序上采样/帧生成 SDK 仅渲染侧可选外部后端，本体默认经典路径，关键路径与路径追踪参考或解析解可对拍。

### 14.0 规模计数再刷新（2026-10 实测，统一口径 `find src -name '*.rs' | wc -l`）

> **口径声明**：以下为 `pkg/<crate>/src` 下 `*.rs` 文件计数（含测试文件），与 v3/v6 偶有的"子目录/行数"混计不同。本表为后续各版的权威基线，修正 v6 §13.0 中偏高的几项（如虚拟几何 GPU、体积 GPU、毛发 GPU）。

| crate / 子系统 | v6 §13.0 记载 | v7 统一口径实测 | 说明 |
|---|---|---|---|
| `prism_render_architecture` | 680 | **680** | 帧图/流送/view_family/ray_scene(118) 持续深化 |
| `prism_render_shading`（含 `gi/` **63 子目录** / 246 文件） | 344 | **344** | GI 子系统已达 63 子目录、`gi/` 下 246 `*.rs` |
| `prism_render_scene`（ray_scene 精确 SDF 族） | 458 | **459** | 新增 `cut_hollow_sphere`/`round_cone`/`solid_angle` 等精确 SDF 图元 |
| `prism_render_visibility`（两阶段 HZB 遮挡） | — | **16** | 本 lane：HZB footprint→projection→pyramid→query 全链 CPU 参考 |
| `prism_render_material` | — | **12** | 闭包 IR / ABI 前端 |
| `prism_virtual_geometry_gpu` | 49 | **25** | 软光栅主路径稳定（v6 计数偏高，统一口径修正） |
| `prism_volumetric_gpu` | 191 | **101** | froxel/储层/异质介质孪生（统一口径修正；近期新增 soft_particle/luminance_hist/sharpen_cas/checkerboard_resolve 孪生） |
| `prism_hair_gpu` | 108 | **59** | Marschner/双散射 + **逐发丝 LSS 段切分 GPU 孪生**（统一口径修正） |
| `prism_physics_gpu` / `core` / `geometry` | 275 / 183 / 34 | **220 / 167 / 34** | 流体/断裂/VBD/CCD/几何流形（统一口径修正） |
| `prism_audio_core` / `spatial` / `rt` / `hrtf` | 83 / 52 | **84 / 52 / 6 / 9** | 卷积混响/HOA/衍射/房间声学；共享 `Fft` 原语已统一消除私有 radix-2 重复 |

**修正说明**：v6 §13.0 对 `virtual_geometry_gpu / volumetric_gpu / hair_gpu / physics_gpu` 的计数高于当前统一口径实测——原因是早期按「孪生对 + WGSL + 测试」混计或含已合并文件。本表改用单一 `*.rs` 文件口径，后续版本一律沿用，避免分级虚高。**结论不变**：GI/面光/焦散/多光/世界空间缓存/镜面 AA/音频波动声学/物理软体流体仍为真实成规模模块，兑现核对结论（v6 §13.1）继续成立。

### 14.1 空白赛道再核验（v6 §13.2 四项 + §11 两项，当前实测仍 0 文件 → 维持 ⬜）

> `find pkg -iname '*<kw>*' -name '*.rs'` 实测：`virtual_texture / sampler_feedback / directstorage / gdeflate / frame_gen / abd / affine_body / opacity_micromap / micromap / nanite_tessellation` **当前均 0 文件**。维持 v6 分级，继续列为最高优先的真实短板。**唯一进展信号**：`prism_hair_gpu` 近期落 `逐发丝 LSS 段计数 / 发丝折线→LSS 段切分 GPU 孪生`——这是 §14.2.2「硬件曲线 RT」赛道的**上游几何前置**已动工（LSS 段数据正在生成），但硬件 BLAS 构建/相交内核仍 0，赛道整体维持 🟡 骨架前。

### 14.2 v7 新增前沿赛道（纯经典数值，逐条规格）

#### 14.2.1 随机纹理过滤 STF（Stochastic Texture Filtering）
- **借鉴对象**：Pharr/Wronski/Hofmann《Filtering After Shading with Stochastic Texture Filtering》(2024) + TAA/TSR 时序累积。
- **算法要点**：对**任意纹理编码**（压缩块、神经无关的程序化、稀疏虚拟页、各向异性 ratio 采样）以**单次随机抽样**代替昂贵的硬件三线性/各向异性过滤，把过滤移到着色之后，由时序累积（TSR/TAA）在时域求期望收敛；用蓝噪声（STBN）+ Owen-scrambled Sobol 抖动降低单帧方差，jitter 与上采样 jitter 共用同一序列保证无相关聚块。
- **预算**：每纹理采样从 N tap 降为 1 tap（省带宽/ALU），代价是单帧方差上升、依赖时序累积收敛；禁用过滤硬件时对 RVT/SVT（§13.2.1）尤其划算。
- **效果**：高分辨率贴图/虚拟纹理页在相同带宽下可过滤任意编码；与各向异性 ratio、mip bias 协同，动态场景静止 2–4 帧即收敛到参考三线性质量。
- **落点**：`gi/sample/`（STBN/Sobol 抖动源）+ `temporal_upscale/`（时序求期望）+ `texture_streaming/` / 未来 `virtual_texture/`（页采样）。
- **验收**：静止相机 ≤4 帧内与硬件三线性 PSNR 差 <0.5 dB；运动下无过滤抖动/闪烁；禁用硬件各向异性时等质量下带宽显著下降；CPU golden 以固定随机序列可逐 texel 对拍。

#### 14.2.2 硬件曲线 / 发丝 RT — 线性扫掠球 LSS（Linear Swept Spheres）
- **借鉴对象**：NVIDIA RTX Mega Geometry 的 **Linear Swept Spheres** 曲线图元 + DXR 曲线/`OptiX` curve primitive + UE Groom RT。
- **算法要点**：发丝折线段编码为**两端半径线性插值的扫掠球**（LSS），构建专用 BLAS，由硬件 RT 直接相交发丝曲线——取代把每根发丝 proxy 成管状三角网格（省几何/BVH 内存一个数量级）；相交后接 Marschner/Zinke 双散射 BSDF 做阴影/反射/GI 可见性。**上游 LSS 段切分已在 `prism_hair_gpu` 落孪生**，本赛道补齐 BLAS 构建 + 相交内核 + 软件 fallback。
- **预算**：LSS BLAS 构建分帧 refit（蒙皮/风动发丝）；相交比管状网格省 BVH 节点与 traversal；无 RT 平台降级为 strand→card LOD + 软件 BVH。
- **效果**：毛发在反射/阴影/GI 中以真实曲线出现（而非卡片近似），自阴影与透射散射物理正确；与水体/体积焦散、RT 反射统一走同一 TLAS。
- **落点**：`prism_hair_gpu`（LSS 段数据已动工）+ `prism_render_architecture/ray_scene/`（BLAS/TLAS 接入）+ `prism_render_scene/raytrace/` + `gi/hair_bsdf/`（散射响应）。
- **验收**：LSS 相交与管状网格参考在发丝剪影/自阴影一致；BVH 内存较三角 proxy 降 ≥5×；蒙皮/风动下 refit 不破时域稳定；软件 fallback 与硬件 parity 可对拍。

#### 14.2.3 微几何硬件 RT — 位移微网格 DMM + 不透明微贴图 OMM
- **借鉴对象**：NVIDIA Displaced Micro-Mesh (DMM) + Opacity Micromap (OMM) + RTX Mega Geometry 簇级 BLAS。
- **算法要点**：**DMM**——把高度/位移烘为基三角上的**微网格位移 LOD**，让 Nanite 式程序化置换几何**可被硬件 RT**（反射/阴影/GI 看到真实置换表面，而非低模），位移在 BLAS 构建期解算、按屏幕投影选微细分级；**OMM**——把 alpha-test 掩码（植被叶片/发卡/铁丝网）烘为**三态微贴图**（全不透明/全透明/未知），RT 相交时对「全透明/全不透明」微三角**跳过 any-hit shader**，仅「未知」区回落 alpha 采样，消除植被 RT 的 any-hit 风暴。
- **预算**：DMM/OMM 烘焙离线或异步；BLAS 内存随微细分上升但 traversal 省；OMM 把植被阴影 any-hit 调用降一个数量级。
- **效果**：置换地形/岩石/树皮在 RT 反射与光追阴影中有真实微起伏；植被/透明掩码 RT 阴影无 any-hit 卡顿、边缘锐利。
- **落点**：`virtual_geometry/`（DMM 与软光栅置换共享高度源）+ `ray_scene/`（micromap BLOB → BLAS）+ `gi/shadow/` `gi/reflect/`（消费）。
- **验收**：DMM RT 命中与软光栅置换剪影一致、无 LOD 跳变；OMM 对 alpha-test 植被阴影较朴素 any-hit 提速且无边缘漏光；micromap 构建确定性可对拍。

#### 14.2.4 可编程光栅 WPO — 动画 / 蒙皮虚拟几何（植被风动 · 角色）
- **借鉴对象**：UE5.4+ Nanite **可编程光栅（World Position Offset / 顶点动画）** + Nanite Skinned Mesh + Nanite Foliage。
- **算法要点**：在虚拟几何软光栅前端插入**可编程顶点阶段**——支持 world-position-offset（风场/顶点动画材质驱动植被摆动）与**蒙皮**（骨骼矩阵调色板 → cluster 顶点变形），使海量角色/植被进入 vis-buffer 零 LOD pop 的高密度管线；WPO/蒙皮后更新**保守包围与运动矢量**，供 HZB 两阶段遮挡（本 lane 已落 CPU 参考）与 TSR 重投影正确剔除/累积。
- **预算**：顶点阶段 ALU 随 WPO 复杂度上升；蒙皮矩阵调色板常驻；运动矢量必须随 WPO 更新（否则 TSR ghosting）。
- **效果**：森林级植被风动、人群级蒙皮角色全部走 Nanite 密度与一致着色；与 VSM（§3.4）协同得到动画几何的一致阴影。
- **落点**：`prism_virtual_geometry_gpu`（可编程顶点/蒙皮前端）+ `prism_render_architecture/motion/`（WPO 运动矢量）+ `prism_render_visibility`（WPO 后保守包围喂两阶段 HZB）+ `deformation/`（蒙皮调色板）。
- **验收**：WPO/蒙皮几何在 HZB 遮挡与 TSR 下无错剔除、无 ghosting；运动矢量与几何位移一致；大规模植被/人群维持零 LOD pop。

#### 14.2.5 分层 GI 融合 — 屏幕探针 + 世界空间辐照缓存（ReSTIR GI 多级）
- **借鉴对象**：UE5 Lumen 的 **Screen Probe Gather + World Radiance Cache** 分层 + ReSTIR GI 时空重用 + 世界空间哈希缓存（§13.1 已落 `gi/world_space`）。
- **算法要点**：近场由**屏幕空间探针**（`gi/screen_probe`）稠密采样 + ReSTIR 时空重用解短程间接光；远场/被遮挡区由**世界空间哈希辐照缓存**（`gi/world_space`）提供低频兜底，二者按射线命中距离与屏幕可见性**加权融合**，消除屏幕空间 GI 的边界漏光与 off-screen 丢失；缓存按相机驻留分帧更新、八面体编码方向辐照。
- **预算**：屏幕探针固定屏幕预算；世界缓存哈希桶按驻留分帧摊销；融合权重一趟全屏。
- **效果**：动态无烘焙下近场细节 + 远场稳定兼得；相机转动/off-screen 间接光不丢、不闪；关 RT 时与 SSGI/DDGI 平滑降级。
- **落点**：`gi/screen_probe/` + `gi/world_space/` + `gi/world_restir/`（融合与时空重用）+ `gi/surface_cache/`（弹射摊薄）。
- **验收**：与路径追踪 GI 参考多弹射辐照收敛一致；屏幕边界/遮挡过渡无漏光；相机运动时域稳定、无缓存爆闪。

#### 14.2.6 整帧 GPU 自驱 — Work Graph（mesh nodes 全管线调度）
- **借鉴对象**：D3D12 **Work Graphs**（含 2024 mesh nodes）+ Metal `MTL4` 命令调度 + NVIDIA 持久线程生产者-消费者。
- **算法要点**：把 visibility→material classify→material shade→lighting 的分发从 CPU 预录命令改为 **GPU 自驱 work graph**：上游节点动态产出下游工作项（如 vis-buffer 分箱后按材质桶 fan-out 着色节点，mesh node 直接产出微三角），消除往返 CPU 的 indirect 分发与过度保守的 worst-case 分配；与 bindless 堆 + SER（§6.12）协同提高着色相干。`prism_render_architecture/work_graph/`（5 文件）已有骨架，本赛道补齐节点图编排与材质桶 fan-out。
- **预算**：work graph 调度省 CPU 录制与 indirect 读回；节点 backing 内存按峰值估算封顶；无硬件 work graph 平台降级为经典 indirect + 多 pass。
- **效果**：GPU-driven 管线端到端自驱，材质分箱着色相干、空桶零开销；超大场景分发不再被 CPU 命令录制瓶颈。
- **落点**：`prism_render_architecture/work_graph/`（节点图）+ `gpu_scene/` + `material/`（材质桶 fan-out）+ `frame_graph/`（降级路径）。
- **验收**：work graph 与经典 indirect 多 pass 结果逐像素一致；空材质桶无 launch 开销；降级路径 parity；节点 backing 不溢出封顶。

#### 14.2.7 物理 — IPC 无穿透摩擦接触 + ABD 刚软统一（纯经典数值）
- **借鉴对象**：Li et al.《Incremental Potential Contact》(2020) 的**保证非穿透 + 精确摩擦**障碍能量 + 《Affine Body Dynamics》(2022) 刚软统一（§13.2.4 已列 ABD 空白）。
- **算法要点**：接触以**对数障碍能量**表达（距离趋零时能量趋无穷，数学上保证无穿透），配 CCD 过滤线搜索步长；摩擦走**半隐式对偶**近似精确库仑摩擦；ABD 用仿射坐标统一表达刚体与近刚软体，与现有 VBD/XPBD 求解器共存，难接触场景（齿轮/堆叠/布-体耦合）切 IPC 分支求鲁棒解。
- **预算**：IPC 牛顿迭代 + CCD 线搜索成本高，仅难接触子集启用；ABD 自由度远低于全 FEM，适合实时近刚；与 island/sleep（已落）协同只激活活跃岛。
- **效果**：堆叠/楔入/薄壳接触零穿透、摩擦物理正确，消除 XPBD 常见的穿透抖动与爆飞；刚软耦合（角色与布/软体）统一稳定。
- **落点**：`prism_physics_core/{abd(待建), ccd, island, sleep, soft, vbd}` + `prism_physics_geometry/`（距离/流形查询）。
- **验收**：难接触基准（斜面摩擦停住、深楔入不穿透、堆叠稳定）与离线 IPC 参考一致；能量无爆增；活跃岛外零成本；确定性可对拍。

#### 14.2.8 音频 — 预计算波场 + 运行期参数解码（波动声学天花板，纯经典数值）
- **借鉴对象**：Raghuvanshi/Snyder《Parametric Wave Field Coding》+ Microsoft Project Acoustics 形态 + 现有 `prism_audio_spatial` 波动声学/HOA/衍射。
- **算法要点**：离线用 **FDTD / ARD 波动求解器**在场景几何上预计算声场，提取**感知参数场**（到达方向、初始能量、混响衰减 RT60、遮挡/衍射量）压成空间网格；运行期按声源-听者位置**三线性插值参数**并驱动现有 HOA 空间化 + 卷积混响（`prism_audio_core/reverb/convolver.rs`）——得到绕角衍射、门洞传播、室内外过渡的物理声场，而运行期零波动求解。**纯经典 DSP**：求解是数值 PDE，解码是参数插值 + 卷积，无任何神经/学习路径。
- **预算**：波场求解离线/烘焙；运行期仅参数网格采样 + 既有卷积混响；参数场按分区流送控内存。
- **效果**：遮挡/衍射/混响随几何物理正确变化（墙后闷、门洞泄声、室内外无缝），超越纯几何光线声学的硬边界。
- **落点**：`prism_audio_spatial/`（参数场解码 + 衍射）+ `prism_audio_core/reverb/`（卷积混响驱动）+ 烘焙工具侧波动求解。
- **验收**：参数解码混响/遮挡与离线波动求解参考在关键听点一致；声源/听者移动时参数插值无突跳；室内外过渡平滑；确定性可对拍。

### 14.3 v7 增补路线图优先级（并入既有阶梯，不改 §9 P0 基底次序）

| 增补特性 | 落点 | 现状（实测） | 优先级 | 依赖 |
|---|---|---|---|---|
| 可编程光栅 WPO/蒙皮虚拟几何 | `virtual_geometry_gpu` + `motion/` + `render_visibility` | 🟡 软光栅主路径在，WPO/蒙皮前端待建 | **P0.5** | 虚拟几何软光栅、运动矢量、两阶段 HZB |
| 硬件曲线 RT（LSS 发丝） | `hair_gpu` + `ray_scene` + `gi/hair_bsdf` | 🟡 LSS 段切分孪生已动工，BLAS/相交 0 | **P1** | RTX 曲线 BLAS、毛发 BSDF、TLAS |
| 微几何 RT（DMM + OMM） | `virtual_geometry/` + `ray_scene/` + `gi/shadow,reflect` | ⬜ micromap 0 文件 | **P1** | DMM/OMM 烘焙、簇级 BLAS |
| 分层 GI 融合（屏幕探针+世界缓存） | `gi/screen_probe` + `gi/world_space` + `gi/world_restir` | 🟡 两侧骨架均在，融合权重待连 | **P1** | 屏幕探针、世界哈希缓存、ReSTIR |
| 随机纹理过滤 STF | `gi/sample` + `temporal_upscale` + `texture_streaming` | 🟡 抖动源/时序在，STF 采样路径待建 | **P1** | STBN/Sobol、TSR、（RVT 协同） |
| 整帧 GPU Work Graph（mesh nodes） | `work_graph/` + `gpu_scene` + `material` | 🟡 骨架 5 文件 | **P2** | work graph 后端、bindless、材质分箱 |
| IPC 无穿透摩擦接触 + ABD | `prism_physics_core/{abd(待建),ccd,soft,vbd}` | ⬜ abd 0 文件，CCD/VBD/soft 在 | **P2** | CCD、几何流形、island/sleep |
| 预计算波场 + 参数解码（音频） | `prism_audio_spatial` + `prism_audio_core/reverb` | 🟡 HOA/衍射/卷积混响在，波场烘焙待建 | **P2** | 离线 FDTD/ARD、HOA、卷积混响 |

**v7 增补总原则**：① **不改 §9 P0 基底次序**，也不重复 §6/§11/§12/§13 已兑现条目；② **WPO/蒙皮虚拟几何列 P0.5**——它是「海量动画几何进 Nanite 密度」的开阔世界刚需，且本 lane 两阶段 HZB CPU 参考已就绪可直接消费其保守包围，依赖最成熟、价值最高；③ **LSS 发丝 RT / 微几何 DMM-OMM / 分层 GI 融合 / STF 列 P1**——前两者让毛发与置换/透明几何进入统一 TLAS 的 RT 保真，后两者补 GI 边界质量与带宽；④ **Work Graph / IPC-ABD / 波场声学列 P2**——价值高但依赖重、按需增量；⑤ 全部遵循数值红线，走「CPU golden → GPU kernel → 真机 parity」三步，关键路径必须与**路径追踪参考 / 解析解 / 离线数值求解**可对拍，未毕业不作生产默认；**任何增补均不引入 AI/ML/神经/LLM 推理路径**（厂商时序上采样/帧生成 SDK 仅渲染侧可选外部后端）。

---

## 15. v8 产品对标高级特性增补与规模三刷（2026-10，纯经典数值 / 无 AI·ML·神经·LLM）

> **本版目的**：用户再次指出「代码有更新」并要求「借鉴参考优先产品、添加高级特性、兼顾性能与效果、达到顶级次世代 AAA 级别、更新优化设计文档」。本版 ① 以统一口径 `find pkg/<crate>/src -name '*.rs' | wc -l` **三刷规模**（2026-10 实测，接续 v7 §14.0 的权威基线）；② 在 v7 §14.2 的 8 条前沿之上，追加 **6 条产品级高级特性赛道**——均为 2023–2026 顶级 AAA 实际落地的「可见天花板」，经关键字实测确认**上游骨架是否已在**，逐条标注 🟡（骨架在/缺整合）或 ⬜（真实空白），避免分级虚高；③ 刷新路线图优先级。**全程纯经典数值**，排除一切 AI/ML/神经/LLM 推理路径；厂商时序上采样 / 帧生成 SDK 仅渲染侧可选外部后端，本体默认经典路径，关键路径与路径追踪参考 / 解析解 / 离线数值求解可对拍。

### 15.0 规模三刷（2026-10 实测，统一口径 `find src -name '*.rs' | wc -l`）

| crate / 子系统 | v7 §14.0 | v8 实测 | 变化与说明 |
|---|---|---|---|
| `prism_render_architecture` | 680 | **680** | 帧图 / 流送 / `work_graph`(5 文件: graph·queue·ratio·indirect·mod) / `ray_scene`(含 4 文件 tessellation 族) 持续深化 |
| `prism_render_shading`（`gi/` 63 子目录 / 246 文件） | 344 | **344** | GI 子系统稳定在 63 子目录、`gi/` 下 246 `*.rs`（screen_probe·world_space·world_restir·probe_volume·path_reuse·area_light·caustics·denoise·spec_denoise·light·nee 全在） |
| `prism_render_scene` | 459 | **459** | 精确 SDF 图元族稳定 |
| `prism_render_visibility`（两阶段 HZB 遮挡） | 16 | **20** | 本 lane 本版新增 **场景级两阶段遮挡解析器 `two_phase_resolve.rs`**（Nanite 式：仅当实例在上一帧 + 当前帧 HZB **双重证明被遮挡**才剔除，复用 `classify_early_hzb`/`resolve_current_hzb` 单源 stage 逻辑；75 测试全绿）。全链：footprint→projection→pyramid→query→build→cull_hzb→occlusion_resolve→two_phase→two_phase_resolve |
| `prism_render_material` | 12 | **12** | 闭包 IR / ABI 前端 |
| `prism_virtual_geometry_gpu` | 25 | **25** | 软光栅主路径稳定 |
| `prism_volumetric_gpu` | 101 | **112** | froxel / 储层 / 异质介质孪生持续新增（soft_particle / luminance_hist / sharpen_cas / checkerboard_resolve 等） |
| `prism_hair_gpu` | 59 | **63** | Marschner / 双散射 + **近/远场散射混合 ramp GPU 孪生** + 逐发丝 LSS 段切分 |
| `prism_physics_gpu` / `prism_physics_core` / `prism_physics_geometry` | 220 / 167 / 34 | **224 / 167 / 34** | 流体 / 断裂 / VBD / CCD / MPM / 几何流形；`physics_core` 子目录 30+（含 fluid·fracture·vbd·soft·mpm·island·sleep·ccd·reduced 等） |
| `prism_audio_core` / `prism_audio_spatial` / `prism_audio_rt` / `prism_audio_hrtf` / `prism_audio_device` | 83 / 52 / 6 / 9 / — | **86 / 52 / 6 / 9 / 7** | 音频已拆为 5 crate；卷积混响 / HOA / 衍射 / 房间声学 / 设备 I/O；共享 `Fft` 原语统一 |

**口径一致性**：本表与 v7 §14.0 同口径（单一 `*.rs` 文件计数，含测试文件），兑现核对结论（v6 §13.1 / v7 §14.1）继续成立。**本版唯一结构性增量**在 `prism_render_visibility`（+4，两阶段遮挡解析闭环）与 `volumetric_gpu`/`hair_gpu`/`physics_gpu` 的孪生增量。

### 15.1 产品级高级特性赛道（v8 新增，逐条实测分级）

> 关键字实测（`find pkg -iname '*<kw>*' -name '*.rs'`）结果直接决定下列分级，杜绝虚高：
> `mega_light/light_bvh=0`、`light_tree=1`、`iridescen/thinfilm=0`、`thin_film=1`、`deferred_material/visibility_buffer=0`、`hploc/ploc/lbvh/blas_refit=0`、`tessellation=4`（CPU 侧 ray_scene 族）、`ray_acoustic=0`、`prism_audio_rt=6`。

#### 15.1.1 MegaLights 式无界阴影光（随机光采样 + 光重要性树 + RT 软阴影 + ReSTIR DI 复用）
- **借鉴对象**：UE 5.5 **MegaLights** 形态 + Estevez-Kulla《Importance Sampling of Many Lights with Adaptive Tree Splitting》(2018) + ReSTIR DI 时空复用。
- **算法要点**：对**成百上千盏带阴影光**不再逐光一张阴影图，而是：① 用**光重要性树 BVH**（已有 `gi/light/light_tree.rs` 为上游）按辐照度 × 立体角自适应切分，每像素**随机抽样 1–few 盏光**；② 对抽中光投**硬件 RT 软阴影**（锥角采样）；③ 用 **ReSTIR DI** 时空储层复用邻域/历史样本把单样本方差压到可用（接 `gi/world_restir`）；④ 遮挡剔除消费本 lane 两阶段 HZB 保守可见集，避免对被遮光做无效 RT。
- **预算**：从「N 盏光 × 全屏阴影」降为「每像素 O(1) 光样本 + ReSTIR 复用」；RT 软阴影按半分辨率 + 时序累积；光树构建每帧增量重排。
- **效果**：开阔世界 / 室内海量点光·聚光·面光全部带接触硬化软阴影，告别「只有主光有阴影」的妥协；夜景霓虹、烛海、科幻舰桥密集光达到电影级。
- **落点**：`gi/light`（光树深化）+ `gi/nee/light_sampling` + `gi/world_restir`（DI 储层）+ `ray_scene`（RT 软阴影）+ `render_visibility`（遮挡驱动）。
- **分级 / 验收**：🟡（light_tree + nee + world_restir 骨架在，缺「随机光采样 × RT 软阴影 × ReSTIR DI」整合管线）。验收：固定 1k 盏光场景与路径追踪多光参考在收敛帧误差 < 阈值；相机静止 ReSTIR 收敛无闪烁；光数翻倍帧时间近似常数。

#### 15.1.2 可见性缓冲延迟材质着色（Visibility-Buffer Deferred Material Shading）
- **借鉴对象**：Nanite **Visibility Buffer + Deferred Material** + Burns/Hunt《The Visibility Buffer》(2013) + 材质分类分箱（material classification tiles）。
- **算法要点**：虚拟几何软/硬光栅只写 **VisBuffer（triangle/cluster/instance ID）**，不在光栅阶段执行材质；随后 ① 按**材质 ID 对像素做屏幕 tile 分类**（间接派发，GPU 自驱）；② 每种材质**只跑一次全屏 compute 着色 pass**，在 pass 内按重心插值现算属性 + 执行闭包 IR（接 `render_material` 12 文件闭包前端）。消除 overdraw 下的重复着色，材质数量与管线状态切换解耦。
- **预算**：着色复杂度 = 可见像素 × 单次闭包，与几何密度 / overdraw 解耦；tile 分箱用前缀和 + 间接 dispatch；bindless 纹理避免描述符爆炸。
- **效果**：Nanite 密度（百万三角）下材质着色成本稳定，支撑复杂 über-BSDF（§6.4）+ 多层 Substrate（§3.5）而不炸管线。
- **落点**：`render_architecture`（VisBuffer / 间接派发）+ `render_material`（闭包 IR 着色核）+ `virtual_geometry_gpu`（ID 写出）。
- **分级 / 验收**：⬜（`visibility_buffer/deferred_material/material_classif` 均 0 文件；闭包 IR 前端在）。验收：与 forward 着色同场景逐像素一致；材质种类翻倍时着色时间不随之线性增长；tile 分类无漏/重着色。

#### 15.1.3 动态场景 BVH 实时重建（HPLOC / PLOC++ + refit 混合，簇级 BLAS 刷新）
- **借鉴对象**：Benthin 等《H-PLOC》(2024) + 《PLOC++》(2022) + Nanite 簇级 BLAS 刷新策略。
- **算法要点**：动态几何每帧 RT 加速结构更新采「**refit 优先、重建兜底**」混合：① 微形变（蒙皮/布料小位移）只 **refit** 现有 BLAS 盒；② 拓扑/大位移触发 **GPU 并行 HPLOC/PLOC++ 重建**（Morton 排序 → 局部近邻合并聚簇 → 自底向上建树，全程 GPU、无 CPU 回读）；③ 簇级（meshlet）BLAS 与虚拟几何 LOD 对齐，只重建受影响簇。TLAS 每帧轻量重排。
- **预算**：refit O(节点数) 远低于重建；HPLOC 单 pass 并行合并，适合每帧预算；按「脏簇」局部刷新摊销。
- **效果**：大规模动态场景（群集动画、破碎、植被风动）保持 RT 反射 / 阴影 / GI 的 BVH 新鲜度，无「RT 结构过期导致鬼影 / 漏光」。
- **落点**：`ray_scene`（BVH 构建 / refit）+ `virtual_geometry_gpu`（簇级 BLAS）+ 与 §14.2.4 WPO 蒙皮几何协同。
- **分级 / 验收**：⬜（`hploc/ploc/lbvh/blas_refit` 均 0 文件；ray_scene 有静态求交基础）。验收：动态场景 RT 结果与逐帧全重建参考一致；refit/重建切换阈值无可见跳变；GPU 构建吞吐达每帧预算内千万图元级。

#### 15.1.4 薄膜干涉 + 光谱色散 iridescence（物理正确的虹彩，深化 `gi/material/thin_film.rs`）
- **借鉴对象**：Belcour & Barla《A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence》(2017) + Hero-Wavelength 光谱上采样 + Substrate 薄膜 slab。
- **算法要点**：在现有 `thin_film.rs` 之上补 ① **Airy 多次反射求和**的薄膜反射率（随膜厚 / 入射角变化的相位干涉），② **Hero-wavelength 光谱采样 + RGB 上采样**解决 RGB 下虹彩偏色，③ 与多瓣 slab（§3.5）**能量守恒耦合**（薄膜 coat 作为可叠加 lobe），④ 色散（阿贝数）可选。全程解析 Fresnel + 相位，纯经典数值。
- **预算**：每样本几次三角 / 复数运算（走 `bevy_math::ops`），比全光谱渲染便宜；Hero-wavelength 单波长 + 时序累积补方差。
- **效果**：肥皂泡、油膜、甲虫鞘翅、车漆珠光、CD 光盘、相机镀膜达到物理级虹彩，随视角连续变化而非贴图伪造。
- **落点**：`gi/material/thin_film`（Airy / 光谱）+ `render_material`（闭包 lobe）+ `particle`/über-BSDF 复用。
- **分级 / 验收**：🟡（`thin_film.rs` 在，缺 Airy 求和 + 光谱上采样 + slab 能量守恒耦合）。验收：与离线光谱薄膜参考在多膜厚 / 多视角一致；能量守恒（反射 + 透射 ≤ 1）；无 RGB 偏色拍频。

#### 15.1.5 自适应 Nanite 置换 tessellation 收口（GPU 簇级细分 + crack-free + VSM/RT 一致）
- **借鉴对象**：UE 5.3 **Nanite Tessellation** + 现有 `ray_scene/{adaptive,patch,displacement,silhouette}_tessellation.rs`（CPU 侧 4 文件）。
- **算法要点**：把现有 CPU 侧 tessellation 族推到 **GPU 簇级运行期细分**：① 屏幕空间误差驱动的**自适应细分因子**；② **crack-free 边界**（相邻 patch 边细分因子取 min / 水密缝合）；③ **位移贴图**沿法线置换，与 §14.2.3 DMM（微网格 RT）**同一位移源**保证光栅与 RT 一致；④ 与 VSM（§3.4）阴影 / §14.2.4 WPO 动画几何协同；⑤ 消费本 lane 两阶段 HZB 做细分前保守剔除。
- **预算**：细分因子按屏幕误差夹紧上限；只对轮廓 / 近景簇高细分；位移在 mesh node / 放大 pass 内就地展开，避免烘死几何。
- **效果**：地形 / 岩壁 / 砖石 / 角色褶皱获得真实轮廓位移（非法线贴图伪高），近景无「贴图平面感」，且 RT 阴影 / 反射与之一致。
- **落点**：`ray_scene`（tessellation 族 → GPU）+ `virtual_geometry_gpu`（簇级细分）+ `gi/shadow`/VSM 协同 + `render_visibility`（剔除）。
- **分级 / 验收**：🟡（CPU 侧 4 文件 tessellation 族在，缺 GPU 簇级 + crack-free + DMM 位移源统一）。验收：相邻 patch 无裂缝；位移光栅与 RT BLAS 轮廓一致；屏幕误差阈值内 LOD 过渡无爆/漏。

#### 15.1.6 实时光线声学 + UTD 边缘衍射（深化 `prism_audio_rt`，与 §14.2.8 波场互补）
- **借鉴对象**：NVIDIA VRWorks Audio / Valve Steam Audio 形态 + UTD（Uniform Theory of Diffraction）边缘衍射 + 现有 `prism_audio_rt`(6 文件)。
- **算法要点**：运行期对**声学网格 path-trace** 声线（镜面反射 + 漫反射散射 + 透射），用 **UTD 对几何边缘求衍射贡献**（绕角泄声的解析加权），把到达方向 / 能量 / 延迟聚合为冲激响应，驱动现有 HOA 空间化 + 卷积混响（`prism_audio_core/reverb`）。与 §14.2.8 预计算波场**互补**：波场擅长低频 / 室内外过渡的烘焙精度，实时光线声学擅长**动态几何 / 可破坏场景**的运行期适应。纯经典几何声学 + DSP，无任何学习路径。
- **预算**：声线数 / 反射阶数按预算夹紧；时序累积 + 空间复用（类 ReSTIR 思路）降方差；活跃声源优先。
- **效果**：动态 / 可破坏场景下遮挡、绕角衍射、回声随几何实时变化（墙倒后声场立刻改变），超越静态烘焙的硬边界。
- **落点**：`prism_audio_rt`（声线 path-trace + UTD）+ `prism_audio_spatial`（HOA）+ `prism_audio_core/reverb`（卷积驱动）。
- **分级 / 验收**：🟡（`prism_audio_rt` 6 文件在，缺运行期 path-trace × UTD × HOA/卷积整合）。验收：与离线声学求解参考在关键听点冲激响应一致；动态几何改变后声场即时更新无突跳；确定性可对拍。

### 15.2 v8 增补路线图优先级（并入既有阶梯，不改 §9 P0 基底次序，不重复 §6/§11/§12/§13/§14 条目）

| 增补特性 | 落点 | 现状（实测） | 优先级 | 依赖 |
|---|---|---|---|---|
| MegaLights 无界阴影光（光树 + RT 软阴影 + ReSTIR DI） | `gi/light` + `gi/world_restir` + `ray_scene` + `render_visibility` | 🟡 light_tree + nee + world_restir 骨架在，整合待建 | **P1** | 光重要性树、RT 软阴影、ReSTIR DI、两阶段 HZB |
| 可见性缓冲延迟材质着色 | `render_architecture`(VisBuffer) + `render_material` + `virtual_geometry_gpu` | ⬜ 0 文件；闭包 IR 前端在 | **P1** | VisBuffer、材质 tile 分类、bindless、间接派发 |
| 动态 BVH 实时重建（HPLOC/PLOC++ + refit） | `ray_scene` + `virtual_geometry_gpu` | ⬜ 0 文件；静态求交基础在 | **P1** | GPU LBVH/PLOC、簇级 BLAS、WPO 蒙皮几何 |
| 薄膜干涉 + 光谱色散 iridescence | `gi/material/thin_film` + `render_material` | 🟡 thin_film.rs 在，Airy/光谱待建 | **P1.5** | Airy 求和、Hero-wavelength 上采样、slab 耦合 |
| Nanite 置换 tessellation 收口（GPU 簇级） | `ray_scene`(tess 族) + `virtual_geometry_gpu` + VSM | 🟡 CPU 侧 4 文件在，GPU/crack-free 待建 | **P1** | 自适应细分、DMM 位移源、WPO、VSM、两阶段 HZB |
| 实时光线声学 + UTD 衍射 | `prism_audio_rt` + `prism_audio_spatial` + `prism_audio_core/reverb` | 🟡 audio_rt 6 文件在，整合待建 | **P2** | 声线 path-trace、UTD、HOA、卷积混响 |

**v8 增补总原则**：① **不改 §9 P0 基底次序**，不重复 §6/§11/§12/§13/§14 已登记条目——本版 6 条均经关键字实测确认为「上游骨架在但产品级整合缺位（🟡）」或「真实空白（⬜）」的**新**赛道；② **MegaLights / 可见性缓冲延迟材质 / 动态 BVH / Nanite 置换收口列 P1**——它们是「海量光 + 海量几何 + 动态场景」三位一体 AAA 开阔世界的直接刚需，且多数上游骨架已就绪、依赖最成熟；③ **薄膜干涉列 P1.5**——效果天花板高、风险低（纯解析），在 `thin_film.rs` 上增量即可；④ **实时光线声学列 P2**——与 §14.2.8 预计算波场互补，价值高但依赖重；⑤ 全部遵循数值红线，走「CPU golden → GPU kernel → 真机 parity」三步，关键路径必须与**路径追踪参考 / 解析解 / 离线数值求解**可对拍，未毕业不作生产默认；**任何增补均不引入 AI/ML/神经/LLM 推理路径**（厂商时序上采样 / 帧生成 SDK 仅渲染侧可选外部后端）。

## 16. v9 产品对标高级特性增补与规模四刷（2026-10，纯经典数值 / 无 AI·ML·神经·LLM）

> **本版目的**：用户再次指出「代码有更新」并要求「借鉴参考优先产品、添加高级特性、兼顾性能与效果、达到顶级次世代 AAA 级别、更新优化设计文档」。本版 ① 以统一口径 `find pkg/<crate>/src -name '*.rs' | wc -l` **四刷规模**（2026-10 实测，接续 v8 §15.0 基线）；② 在 v8 §15.1 的 6 条之上，追加 **6 条产品级高级特性赛道**——均为 2023–2026 顶级 AAA 的「可见天花板」，经关键字实测（`find pkg -iname '*<kw>*' -name '*.rs'`）确认上游骨架是否已在，逐条标注 🟡（骨架在/缺整合）或 ⬜（真实空白），杜绝分级虚高；③ 刷新路线图优先级。**全程纯经典数值**，排除一切 AI/ML/神经/LLM 推理路径；厂商时序上采样 / 帧生成 SDK 仅渲染侧可选外部后端，本体默认经典路径，关键路径与路径追踪参考 / 解析解 / 离线数值求解可对拍。

### 16.0 规模四刷（2026-10 实测，统一口径 `find src -name '*.rs' | wc -l`）

| crate / 子系统 | v8 §15.0 | v9 实测 | 变化与说明 |
|---|---|---|---|
| `prism_render_architecture` | 680 | **680** | 帧图 / 流送 / `work_graph` / `ray_scene` 稳定；贴花·OIT·折射·海洋 LOD 骨架在 `particle/`·`water/` 下 |
| `prism_render_shading`（`gi/` 63 子目录 / 246 文件） | 344 | **344** | GI 子系统稳定在 63 子目录、`gi/` 下 246 `*.rs`（surfel·radiance_cache·photon·world_restir·light_tree·thin_film·specular_aa 全在） |
| `prism_render_scene` | 459 | **459** | 精确 SDF 图元族稳定；本版新增 `shading/light_routing/`(6 文件) + `shading/tonemap/`(6 文件) 光路由与色调映射 ABI 前端（计入既有 459） |
| `prism_render_visibility`（两阶段 HZB 遮挡） | 20 | **22** | 两阶段遮挡闭环持续加固（binning 增量）。全链：footprint→projection→pyramid→query→build→cull_hzb→occlusion_resolve→two_phase→two_phase_resolve |
| `prism_render_material` | 12 | **12** | 闭包 IR / ABI 前端 |
| `prism_virtual_geometry_gpu` | 25 | **25** | 软光栅主路径稳定 |
| `prism_volumetric_gpu` | 112 | **119** | froxel / 储层 / 异质介质孪生持续新增（curl_noise / fluid_diffusion / velocity_dilate / ray_obb·capsule·cylinder / ribbon_geometry / sprite_stretch / frustum_aabb_cull 等 7 新孪生） |
| `prism_hair_gpu` | 63 | **67** | Marschner / 双散射 + 分层样本权重/分配 GPU 孪生 + 逐发丝 keep-ratio 段切分 |
| `prism_physics_gpu` / `prism_physics_core` / `prism_physics_geometry` | 224 / 167 / 34 | **238 / 167 / 34** | 流体 / 断裂 / VBD / CCD / MPM；本版新增 long-range 约束族 + 齿条齿轮(rack-pinion)关节 CPU/GPU 孪生 + 陀螺项 parity（+14） |
| `prism_audio_core` / `prism_audio_spatial` / `prism_audio_rt` / `prism_audio_hrtf` / `prism_audio_device` | 86 / 52 / 6 / 9 / 7 | **88 / 52 / 6 / 9 / 7** | 卷积混响 / HOA / 衍射 / 房间声学 / 设备 I/O；本版新增 ping-pong delay 效果节点 |
| `prism_ui_*`（component / devtools / router） | — | **5 / 4 / 5** | 本版新登记：UI 组件 / 开发者工具 / 路由 crate 骨架（响应式视图 + 信号图 + 编排动画，供编辑器与运行期 HUD；非渲染本体，仅登记规模） |

**口径一致性**：本表与 v7 §14.0 / v8 §15.0 同口径（单一 `*.rs` 文件计数，含测试文件），v6 §13.1 / v7 §14.1 的兑现核对结论继续成立。**本版主要结构性增量**在 `physics_gpu`(+14)、`volumetric_gpu`(+7)、`hair_gpu`(+4)、`render_visibility`(+2) 的孪生/关节/剔除增量，以及 UI 三 crate 的新登记。渲染本体核心（architecture / shading / scene / material / virtual_geometry）保持稳定——**天花板增量在子系统孪生与下文 6 条产品级整合缺口**。

### 16.1 产品级高级特性赛道（v9 新增，逐条实测分级）

> 关键字实测直接决定分级：`virtual_texture/feedback_buffer=0`、`meshlet=0`·`mesh_shader=1`(仅 `hair/mesh_shader_strand.rs`)、`fft_ocean/gerstner=0`·`ocean=2`(authoring + LOD 骨架)、`moment_oit/mboit=0`·`oit=3`(particle+hair+shading 骨架)·`refraction=0`、`decal=1`(`particle/decal.rs`)、`skin_cache/wpo/world_position_offset=0`。

#### 16.1.1 稀疏虚拟纹理 SVT + GPU 反馈缓冲（海量唯一纹理 / 零 pop 流送）
- **借鉴对象**：id Tech **MegaTexture** → UE **Virtual Texturing (RVT/SVT)** + Granite 形态 + DX12 **Sampler Feedback**（硬件级采样命中记录）。
- **算法要点**：把 TB 级唯一纹理切成固定页（128²），GPU 着色时**只采样常驻物理页**；用**反馈缓冲**（Sampler Feedback 或软件 min-mip 记录）收集本帧「真实被采样页 × mip」→ 回读驱动页请求队列 → DirectStorage/GDeflate 异步解压上传 → 更新间接页表（indirection texture）。缺页用父 mip 兜底，杜绝黑块/pop。与 §11 RVT、§12 Sampler Feedback、§13 DirectStorage 条目的关系：本条是**产品级整机整合**（页表 × 反馈 × 解压 × 兜底四件套闭环），上游全为 0 文件。
- **预算**：常驻物理池按显存夹紧（如 4K×4K×若干层）；反馈回读半分辨率 + 隔帧；页上传按带宽预算限流；间接表更新 GPU 端 compute。
- **效果**：开阔世界近景 8K 贴图细节 + 远景无缝 LOD，显存占用与可见页数成正比而非与世界总量成正比；相机瞬移/传送无纹理 pop。
- **落点**：`render_architecture`（流送/帧图 + 页表 + 反馈回读）+ `render_scene/geometry`（采样绑定）+ `render_material`（bindless 页表句柄）。
- **分级 / 验收**：⬜（virtual_texture/feedback_buffer 实测 0 文件，流送帧图在）。验收：固定物理池下任意相机轨迹无缺页黑块；反馈页集与「全驻留参考」逐页一致；带宽超限优雅降级（降 mip 不黑块）。

#### 16.1.2 Mesh Shader / meshlet 硬件放大路径（几何前端二选一后端）
- **借鉴对象**：NVIDIA Mesh Shader / UE Nanite 的硬件 meshlet 回退路径 + `VK_EXT_mesh_shader`；现有 `virtual_geometry_gpu`(软光栅主路径) + `hair/mesh_shader_strand.rs`(发丝骨架)。
- **算法要点**：对**中大三角形簇**走硬件 mesh/task shader 两级放大（task 做簇级锥剔除 + LOD 选择，mesh 产出 meshlet 顶点/图元），与现有**软光栅**（擅长 ≤1px 微三角）构成「大三角硬件、微三角软件」分档后端；统一消费两阶段 HZB 保守可见集与簇级 LOD 误差。与 §7 Work Graph(§14.2.6)互补：Work Graph 是整帧调度，mesh node 是其几何叶子。
- **预算**：meshlet 64 顶点 / 124 图元标准封装；task 剔除按屏幕误差 + 背面 + HZB；大三角走硬件省去软光栅原子争用。
- **效果**：高多边形密度场景在支持 mesh shader 的硬件上吞吐提升，微三角仍由软光栅保精度；跨档位无接缝。
- **落点**：`virtual_geometry_gpu`（硬件 meshlet 后端）+ `render_architecture`（后端选择/Work Graph 叶子）+ `render_visibility`（HZB 消费）。
- **分级 / 验收**：🟡（`hair/mesh_shader_strand.rs` 发丝级骨架在，几何级 meshlet 封装 + task 剔除缺位）。验收：硬件/软件两路栅格化结果逐像素一致（±1 ULP 深度）；大/微三角分档切换无裂缝；无硬件支持时自动回退软光栅。

#### 16.1.3 FFT 频谱海洋 + Gerstner 混合 + 破碎白沫/泡沫（开阔水体天花板）
- **借鉴对象**：Tessendorf FFT 海面 + UE Water / Sea of Thieves / Horizon Forbidden West 水体形态 + 现有 `water/authoring/ocean.rs` + `water/ocean_lod.rs`。
- **算法要点**：① 用 **FFT 频谱**（Phillips/JONSWAP 谱）生成多级联（cascade）位移 + 法线贴图，叠加 **Gerstner 波**补充可控大浪方向性；② 由雅可比行列式判定**波峰折叠/破碎**区生成白沫 mask，驱动泡沫（foam）贴花与粒子；③ 海面 LOD 用现有 `ocean_lod` clipmap/投影网格，远景并入体积雾与平面反射/SSR 混合。纯 FFT + 解析波 + 泡沫经典数值，无学习路径。
- **预算**：FFT 分辨率按级联夹紧（512² 近 / 256² 远）；泡沫 mask 复用雅可比，无额外求解；LOD 投影网格顶点数与屏幕覆盖成正比。
- **效果**：真实海面色散/涌浪/破碎白沫，近景发丝级浪花 + 远景无缝到地平线；与岸线、浮体（接物理）耦合。
- **落点**：`render_scene/water`（谱 + Gerstner + 泡沫）+ `render_architecture/water/ocean_lod`（LOD）+ `volumetric_gpu`（浪花雾/喷溅）。
- **分级 / 验收**：🟡（`ocean.rs` + `ocean_lod.rs` 骨架在，FFT 频谱 + 泡沫 + 多级联缺位）。验收：谱统计量（有效波高/周期）与 JONSWAP 参考一致；破碎区白沫与雅可比阈值一致；LOD 过渡无几何爆/漏；详见 `prism_water_engine_design_zh.md`。

#### 16.1.4 矩不变 OIT (Moment-based OIT) + 屏幕/RT 混合折射（透明天花板）
- **借鉴对象**：Münster **Moment-Based OIT**（Münstermann et al. 2018）+ UE 粗糙折射 + 现有 `particle/oit.rs` · `hair/oit_frontend.rs` · `shading/oit.rs` + `particle/reflect_refract_vec.rs`。
- **算法要点**：① 用**幂矩/三角矩**压缩逐像素透射率函数（4/6/8 矩），单遍重建顺序无关的透明合成，取代重排序/深度剥离；② 折射走**屏幕空间近似（粗糙度相关 mip 模糊 + 厚度）为主、RT 折射为精档**的分档，法线/粗糙度/厚度来自透明 GBuffer；③ 与毛发 OIT 前端、粒子 OIT 共享矩缓冲，统一合成口径。纯矩重建 + 解析折射，无学习路径。
- **预算**：矩缓冲 8 矩约 2×RGBA16F；单几何遍 + 单解析遍；RT 折射仅精档/反射性材质开启、半分辨率 + 时序累积。
- **效果**：烟雾/玻璃/头发/植被叠加无排序瑕疵；粗糙玻璃/水/冰的厚度相关折射与色散，边界无硬切。
- **落点**：`render_shading/oit`（矩重建）+ `render_architecture/particle/oit`+`hair/oit_frontend`（前端消费）+ `ray_scene`（RT 折射精档）。
- **分级 / 验收**：🟡（三处 oit 骨架在，矩压缩/重建 + 分档折射缺位）。验收：矩重建合成与「逐片段精确排序参考」在容差内一致；折射厚度/色散与解析/路径追踪参考对拍；多层透明无 pop/排序闪烁。

#### 16.1.5 延迟贴花 DBuffer + 网格贴花（法线/粗糙度/反照率混合）
- **借鉴对象**：UE **DBuffer Decals** + 网格贴花（mesh decals）形态 + 现有 `particle/decal.rs` 骨架。
- **算法要点**：在 GBuffer 之后、光照之前，把贴花投影到 **DBuffer**（反照率/法线/粗糙度三张），用权重与屏幕法线做**各向异性混合**（避免拉伸）后再统一光照；网格贴花走独立几何遍贴合曲面（弹孔/裂纹/涂鸦）。与可见性缓冲延迟材质（§15.1.2）协同：贴花在材质分箱前写入 DBuffer。纯混合，无学习路径。
- **预算**：DBuffer 三张 RT；贴花按簇/tile 剔除后批量；法线混合用 Reoriented Normal Mapping（RNM）解析式。
- **效果**：海量弹孔/泥渍/路面细节无需烘焙到基础贴图，动态投放且与底材无缝融合；法线正确不翻面。
- **落点**：`render_architecture/particle/decal`（投影/DBuffer）+ `render_scene/geometry`（GBuffer 混合）+ `render_shading`（光照前合成）。
- **分级 / 验收**：🟡（`particle/decal.rs` 骨架在，DBuffer 三通道 + RNM 混合 + 网格贴花缺位）。验收：贴花法线混合与 RNM 解析参考一致；陡坡/边缘无拉伸；贴花顺序/权重合成确定可对拍。

#### 16.1.6 GPU 蒙皮缓存 + WPO 几何馈入 Nanite/RT BLAS（动画几何解锁）
- **借鉴对象**：UE **Skin Cache** + Nanite Skeletal / 可编程光栅 WPO + RT 动态 BLAS refit；补 §7 可编程光栅（§14.2.4）与 §8 动态 BVH（§15.1.3）的**动画几何前置**。
- **算法要点**：① **GPU 蒙皮/变形缓存**把 skinned/morph/WPO 顶点一次性烘到可被复用的 GPU 顶点缓冲（而非每遍重算），供光栅 + RT + 两阶段 HZB + 虚拟几何共享；② WPO（世界位置偏移，风/植被/顶点动画）在缓存阶段求值，输出稳定几何供 **RT BLAS refit/重建**（接 §15.1.3 HPLOC/PLOC++），使动画角色/植被进入硬件 RT 与 MegaLights（§15.1.1）可见集。纯顶点变换 + BLAS 刷新，无学习路径。
- **预算**：蒙皮缓存按可见骨架/植被实例预算；BLAS refit 优先于全重建（形变小用 refit，拓扑变用重建）；缓存复用避免 raster/RT 双算。
- **效果**：动画角色、风吹植被、顶点动画统一进入 RT 反射/软阴影/GI 与海量光，开阔动态世界的「动的东西也有正确间接光/阴影」。
- **落点**：`virtual_geometry_gpu`（蒙皮缓存 + WPO）+ `ray_scene`（BLAS refit/重建）+ `render_visibility`（HZB 消费）+ `gi/world_restir`+`gi/light`（动态几何入可见集）。
- **分级 / 验收**：⬜（skin_cache/wpo 实测 0 文件；静态求交 + 两阶段 HZB 基础在）。验收：蒙皮缓存顶点与 CPU 蒙皮参考逐顶点一致；refit 后 BLAS 包围/求交与全重建一致（容差内）；动画几何在 RT 反射/软阴影中无撕裂/残影。

### 16.2 v9 增补路线图优先级（并入既有阶梯，不改 §9 P0 基底次序，不重复 §6/§11/§12/§13/§14/§15 条目）

| 增补特性 | 落点 | 现状（实测） | 优先级 | 依赖 |
|---|---|---|---|---|
| GPU 蒙皮缓存 + WPO 几何馈入 RT/Nanite | `virtual_geometry_gpu` + `ray_scene` + `render_visibility` | ⬜ 0 文件；静态求交 + 两阶段 HZB 在 | **P1** | 蒙皮/morph/WPO、BLAS refit、动态 BVH(§15.1.3)、两阶段 HZB |
| 稀疏虚拟纹理 SVT + GPU 反馈缓冲 | `render_architecture`(流送/页表) + `render_scene/geometry` + `render_material` | ⬜ 0 文件；流送帧图在 | **P1** | 页表、反馈回读、DirectStorage/GDeflate(§13)、bindless |
| 矩不变 OIT + 混合折射 | `render_shading/oit` + `particle/oit` + `hair/oit_frontend` + `ray_scene` | 🟡 三处 oit 骨架在，矩重建/折射缺 | **P1.5** | 幂矩/三角矩、透明 GBuffer、RT 折射精档 |
| FFT 频谱海洋 + Gerstner + 泡沫 | `render_scene/water` + `render_architecture/water/ocean_lod` + `volumetric_gpu` | 🟡 ocean + ocean_lod 骨架在，FFT/泡沫缺 | **P1.5** | FFT、JONSWAP 谱、雅可比白沫、LOD 投影网格 |
| 延迟贴花 DBuffer + 网格贴花 | `particle/decal` + `render_scene/geometry` + `render_shading` | 🟡 decal.rs 骨架在，DBuffer/RNM 缺 | **P2** | DBuffer 三通道、RNM 法线混合、tile 剔除、VisBuffer 协同 |
| Mesh Shader / meshlet 硬件放大路径 | `virtual_geometry_gpu` + `render_architecture` + `render_visibility` | 🟡 发丝级骨架在，几何级 meshlet 缺 | **P2** | `VK_EXT_mesh_shader`、task 剔除、HZB、软光栅回退 |

**v9 增补总原则**：① **不改 §9 P0 基底次序**，不重复 §6/§11/§12/§13/§14/§15 已登记条目——本版 6 条均经关键字实测确认为「上游骨架在但产品级整合缺位（🟡）」或「真实空白（⬜）」的**新**赛道；② **GPU 蒙皮缓存 / 稀疏虚拟纹理列 P1**——它们是「动态开阔世界」的两大前置刚需：蒙皮缓存让 v7/v8 的 WPO/动态 BVH/MegaLights 真正吃到**动画几何**，SVT 让海量唯一纹理在固定显存下可行，二者是 v7/v8 已列 P1 特性的「使能器」，必须先于其完全落地；③ **矩不变 OIT / FFT 海洋列 P1.5**——效果天花板高、骨架已在，增量风险低；④ **延迟贴花 / Mesh Shader 列 P2**——价值明确但分别依赖 VisBuffer（§15.1.2）与硬件扩展，排在使能器之后；⑤ 全部遵循数值红线，走「CPU golden → GPU kernel → 真机 parity」三步，关键路径必须与**路径追踪参考 / 解析解 / 离线数值求解**可对拍，未毕业不作生产默认；**任何增补均不引入 AI/ML/神经/LLM 推理路径**（厂商时序上采样 / 帧生成 SDK 仅渲染侧可选外部后端）。

---

## 17. v10 产品对标高级特性增补与规模五刷（2026-10，纯经典数值 / 无 AI·ML·神经·LLM）

> **本版目的**：延续用户「代码有更新 / 借鉴参考优先产品 / 添加高级功能 / 兼顾性能与效果 / 达到顶级次世代 AAA 级别 / 更新优化设计文档」的要求。本版 ① 以统一口径 `find pkg/<crate>/src -name '*.rs' | wc -l` **五刷规模**（§17.0，接续 v9 §16.0 基线，实测反映本轮子系统孪生增量）；② 在 v9 §16.1 的 6 条之上，追加 **6 条产品级高级特性赛道**——均为 2016–2026 顶级 AAA 的「可见天花板」，经关键字实测（`find pkg -iname '*<kw>*' -name '*.rs'`）确认上游骨架是否已在，逐条标注 🟡（骨架在 / 缺整合）或 ⬜（真实空白），杜绝分级虚高；③ 刷新路线图优先级。**全程纯经典数值**，排除一切 AI/ML/神经/LLM 推理路径；厂商时序上采样 / 帧生成 SDK 仅渲染侧可选外部后端，本体默认经典路径，关键路径与路径追踪参考 / 解析解 / 离线数值求解可对拍。

### 17.0 规模五刷（2026-10 实测，统一口径 `find pkg/<crate>/src -name '*.rs' | wc -l`）

| crate / 子系统 | v9 §16.0 | v10 实测 | 变化与说明 |
|---|---|---|---|
| `prism_render_architecture` | 680 | **680** | 帧图 / 流送 / `work_graph` / `ray_scene` 稳定；贴花·OIT·折射·海洋 LOD 骨架在 `particle/`·`water/` 下 |
| `prism_render_shading`（`gi/` 子系统） | 344 | **344** | GI 子系统稳定（surfel·radiance_cache·photon·world_restir·light_tree·thin_film·specular_aa 全在）；本版新增条目落点集中在 SSS / GTAO 的 shading 前端 |
| `prism_render_scene` | 459 | **461** | 精确 SDF 图元族 + 光路由 / 色调映射 ABI 前端稳定并持续加固（+2） |
| `prism_render_visibility`（两阶段 HZB 遮挡） | 22 | **22** | 两阶段遮挡闭环稳定：footprint→projection→pyramid→query→build→cull_hzb→occlusion_resolve→two_phase→two_phase_resolve |
| `prism_render_material` | 12 | **12** | 闭包 IR / ABI 前端 |
| `prism_virtual_geometry_gpu` | 25 | **25** | 软光栅主路径稳定 |
| `prism_volumetric_gpu` | 119 | **131** | froxel / 储层 / 异质介质孪生持续新增（+12，为 §17.1.4 天空·空中透视·体积云与 §17.1.6 体积深阴影提供底座） |
| `prism_hair_gpu` | 67 | **72** | Marschner / 双散射 + 分层样本 + 逐发丝段切分 GPU 孪生（+5，deep opacity 已达 6 文件，供 §17.1.6 深阴影统一） |
| `prism_physics_gpu` / `prism_physics_core` / `prism_physics_geometry` | 238 / 167 / 34 | **253 / 167 / 34** | 流体 / 断裂 / VBD / CCD / MPM / 关节族持续新增（gpu +15） |
| `prism_audio_core` / `prism_audio_spatial` / `prism_audio_rt` / `prism_audio_hrtf` / `prism_audio_device` | 88 / 52 / 6 / 9 / 7 | **93 / 52 / 6 / 9 / 7** | 卷积混响 / HOA / 衍射 / 房间声学 / 设备 I/O；core +5（效果节点 / 混响拓扑） |
| `prism_ui_*`（component / devtools / router） | 5 / 4 / 5 | **5 / 4 / 5** | UI 三 crate 骨架稳定（非渲染本体，仅登记规模） |

**口径一致性**：本表与 v7 §14.0 / v8 §15.0 / v9 §16.0 同口径（单一 `*.rs` 文件计数，含测试文件）。**本版主要结构性增量**在 `physics_gpu`(+15)、`volumetric_gpu`(+12)、`hair_gpu`(+5)、`audio_core`(+5)、`render_scene`(+2)。渲染本体核心（architecture / shading / material / virtual_geometry）保持稳定——**天花板增量在子系统孪生与下文 6 条产品级整合缺口**（SSS / GTAO / 大气天空 / 深阴影 / VRS / 开阔世界流式）。

### 17.1 产品级高级特性赛道（v10 新增，逐条实测分级）

> 关键字实测直接决定分级（`find pkg -iname '*<kw>*' -name '*.rs' | wc -l`）：`subsurface=1`·`sss=1`·`bssrdf=0`、`vrs=0`·`shading_rate=0`、`gtao=0`·`ssao=0`·`bent_normal=1`·`specular_occlusion=1`、`sky_atmosphere=0`·`aerial_perspective=0`·`nubis=0`·`volumetric_cloud=0`、`heightfield=2`·`terrain=2`·`foliage=1`·`world_partition=0`·`hlod=0`、`deep_shadow=0`·`deep_opacity=6`·`contact_shadow=3`·`planar_reflection=0`。

#### 17.1.1 统一次表面散射 SSS（Burley 可分离漫射轮廓 + 屏幕空间 + 随机游走 BSSRDF）
- **借鉴对象**：UE **Subsurface Profile**（可分离漫射轮廓）· Frostbite **Separable SSS**（Jimenez）· Christensen-Burley 归一化漫射 · 影视 **random-walk BSSRDF**（Disney / Weta / pbrt 随机游走）。
- **算法要点**：① 实时档走屏幕空间可分离卷积——以 Burley/Christensen 归一化漫射轮廓拟合多层皮肤（R/G/B 分离散射半径），按视图深度 / 世界尺度自适应核宽，先横后纵两趟可分离卷积；② 预积分曲率档（pre-integrated skin，查 `NdotL × 曲率` LUT）作低配 / 远景回退；③ 高保真 / 离线对拍档走随机游走 BSSRDF（diffusion 近似外的真散射，均质 / 弱异质用 delta/ratio tracking），作 golden 参考；④ 薄部位（耳缘 / 鼻翼）叠透射轮廓。全程解析漫射轮廓 + 经典蒙特卡洛随机游走，无学习路径。
- **预算**：屏幕空间两趟可分离 + reactive mask 保护锐利边；核宽随距离收敛；随机游走档仅参考 / 高配。
- **效果**：皮肤、蜡、玉石、树叶、牛奶、大理石的真实通透与接触软化——近景人物可信度（AAA 角色刚需）。
- **落点**：`prism_render_shading`（SSS 轮廓 + 屏幕空间 pass + 预积分 LUT）+ `gi/material`（BSSRDF 随机游走参考）+ `render_scene/shading`（profile ABI 前端，接 light_routing）。
- **分级 / 验收**：🟡（subsurface/sss 各 1 文件骨架在，随机游走 BSSRDF 实测 0）。验收：屏幕空间档与随机游走 BSSRDF 参考在同散射系数下漫射轮廓一致（容差内）；核宽随距离 / 尺度正确收敛无 halo；透射部位能量守恒。

#### 17.1.2 可变速率着色 VRS Tier2 + 内容自适应 shading-rate image + 粗像素合并
- **借鉴对象**：NVIDIA **Adaptive Shading** · Call of Duty / Gears 5 **content-adaptive VRS** · DX12 / VK VRS Tier2（`VK_KHR_fragment_shading_rate`）。
- **算法要点**：① 用上一帧亮度 / 边缘 / 运动分析生成每 tile 着色率图（SRI：1×1 / 1×2 / 2×2 / 2×4 / 4×4），平滑 / 低频 / 外围降率，高频 / 边缘 / 中心保 1×1；② 结合 motion / reactive mask 保护锐利分段（与 NPR reactive 协同）；③ 可叠 per-draw / per-primitive 率；④ 时序累积 + TSR 上采样掩盖降率伪影。纯图像统计（Sobel / 亮度方差）驱动，无学习路径。
- **预算**：SRI 生成为轻量 compute；着色节省目标 15–40% 像素着色工作量，几何 / 深度不受影响。
- **效果**：在不可见处省着色换帧时间 / 功耗，等画质下更高帧率或把预算让给 GI / 反射。
- **落点**：`prism_render_architecture`（SRI 生成 pass + frame_graph 资源）+ `render_scene`（VRS attachment 绑定）+ `temporal_upscale`（reactive 协同）。
- **分级 / 验收**：⬜（vrs/shading_rate 实测 0）。验收：VRS 结果与全率 1×1 参考的 SSIM/FLIP 在阈值内；降率区无可见块状；关闭 VRS 可回退逐像素 golden。

#### 17.1.3 GTAO 水平基环境光遮蔽 + 弯曲法线 + 镜面遮蔽 + 多次反弹补偿
- **借鉴对象**：Activision **GTAO**（Jimenez 2016，地面真值匹配）· Horizon **bent normals** · HBAO+ · 解析多次反弹补偿。
- **算法要点**：① 屏幕空间沿方位角切片做 horizon-based 可见性积分，得 AO + 弯曲法线（可见半球平均方向）；② 用弯曲法线修正漫反射入射方向与 GI / 反射的方向性遮蔽；③ 由粗糙度 + 弯曲锥角解析求镜面遮蔽（specular occlusion），抑制掠射漏光；④ Jimenez 多次反弹解析补偿（避免过暗）；⑤ 时空双边滤波去噪 + 时序累积。纯几何可见性积分，无学习路径。
- **预算**：半分辨率切片 + 双边上采；与 GI 的 near-field 遮蔽互补（远场交给 GI / ReSTIR）。
- **效果**：接触阴影、缝隙变暗、镜面漏光抑制、弯曲法线驱动的方向性间接光——近景接触真实感。
- **落点**：`prism_render_shading/gi`（GTAO + bent normal + spec occlusion）+ `render_scene/shading`（消费弯曲法线 / 遮蔽）。
- **分级 / 验收**：🟡（bent_normal / specular_occlusion 各 1 骨架在，gtao / ssao 实测 0）。验收：GTAO 与路径追踪 AO 参考（cosine 加权可见性）在同半径下一致（容差内）；弯曲法线与半球采样参考方向一致；镜面遮蔽消除掠射漏光。

#### 17.1.4 大气天空 + 空中透视 LUT + Nubis 体积云 + 统一 god-rays
- **借鉴对象**：Hillaire **Sky-Atmosphere**（Frostbite / UE5，多重散射 LUT + 空中透视体）· **Nubis**（Decima / Guerrilla 体积云）· Horizon cloudscapes。
- **算法要点**：① 预计算透射率 LUT + 多重散射 LUT + 天空视图 LUT，运行期查表得解析大气；② 空中透视存入低分辨率 froxel 体（接现有 `volumetric_gpu` froxel），统一近景雾 / 远景大气；③ 体积云用 Worley/Perlin FBM 密度场 + Henyey-Greenstein 相位 + 两级 raymarch（低频塑形 + 高频侵蚀）+ powder/beer 透射，接天气参数；④ god-rays 用同一体积积分（无需单独 radial blur）。全程解析相位 + 经典 raymarch / 查表，无学习路径。
- **预算**：LUT 为小分辨率一次性；云 raymarch 半 / 四分之一分辨率 + 时序累积 + reproject；froxel 复用体积系统。
- **效果**：物理正确昼夜天空、地平线泛红、远山空中透视、可穿越体积云与丁达尔光——开阔世界天空天花板。
- **落点**：`prism_volumetric_gpu`（froxel 空中透视 + 云 raymarch）+ `render_shading`（天空 LUT）+ `render_scene`（天气 / 日照参数 ABI）。
- **分级 / 验收**：⬜（sky_atmosphere / aerial_perspective / nubis / volumetric_cloud 实测 0；froxel 体积基础在）。验收：天空 LUT 与离线大气参考（brute-force 多重散射）一致；云透射 / 散射与参考 raymarch 收敛一致；空中透视与雾在近远景平滑衔接。

#### 17.1.5 开阔世界流式 — 虚拟高度场网格 VHM + Nanite GPU 植被散布 + World Partition / HLOD
- **借鉴对象**：UE5 **Landscape Nanite** / **Virtual Heightfield Mesh** · **World Partition**（One File Per Actor / 数据层）· **HLOD** 代理网格 · Horizon 程序化植被散布。
- **算法要点**：① 高度场作为虚拟几何（VHM）——clipmap / 四叉树 LOD + 运行期虚拟纹理（RVT，接 §16.1.1 SVT）合成地貌材质，连续无缝 LOD；② GPU 植被散布——按密度图 / 坡度 / 高度在 compute 中实例化散布点，接虚拟几何 + GPU 剔除（接 §16.1.6 蒙皮缓存做风动 WPO）；③ World Partition 网格化流式分区 + 距离 / 视锥驱动的 cell 载入 + 数据层；④ HLOD 远景代理合并（簇级代理）降 draw / BLAS 压力。纯空间数据结构 + 流式，无学习路径。
- **预算**：cell 异步流式（接 §13 DirectStorage/GDeflate）；散布 GPU 剔除；HLOD 代理远景替换。
- **效果**：数十平方公里连续地貌 + 密集植被 + 无缝 LOD——开阔世界骨架（配 §16.1.1 SVT / §16.1.6 蒙皮缓存 / §15.1.1 MegaLights）。
- **落点**：`prism_render_architecture`（paging / world_partition + HLOD 代理 + 流式）+ `virtual_geometry_gpu`（VHM + 植被实例）+ `render_scene/geometry`（地貌 RVT 材质）。
- **分级 / 验收**：🟡/⬜（heightfield / terrain 各 2 + foliage 1 骨架在，world_partition / hlod 实测 0）。验收：VHM LOD 过渡无裂缝 / 爆点；植被散布与密度图一致且确定性；cell 流式无可见 pop；HLOD 切换无跳变。

#### 17.1.6 毛发 / 体积深阴影图统一 — Deep Opacity / Deep Shadow Maps + 半透射自阴影
- **借鉴对象**：PDI **Deep Shadow Maps**（Lokovic-Veach）· UE 毛发 **deep opacity map** · Frostbite 体积自阴影 · 影视发丝 / 烟雾透射。
- **算法要点**：① 从光源视角沿深度分层记录累积透射率（deep shadow / deep opacity 多层），毛发、烟雾、云、半透织物共用一套半透射自阴影；② 查询时按接收点深度在层间插值得透射率；③ 与 §16.1.4 OIT / §16.1.1 SVT 阴影页协同；④ 毛发接双散射（接现有 Marschner / dual-scatter），体积接 froxel 散射。纯透射率积分，无学习路径。
- **预算**：分层数自适应（近光密集）；复用 VSM 页驻留；半分辨率 + 时序。
- **效果**：毛发柔和自阴影不发黑、烟雾 / 云层内部透光梯度、半透织物真实透射——半透材质阴影天花板。
- **落点**：`prism_hair_gpu`（deep opacity 深化）+ `prism_volumetric_gpu`（体积深阴影）+ `render_shading/virtual_shadow`（统一深阴影页 + 消费）。
- **分级 / 验收**：🟡（deep_opacity 实测 6 毛发骨架在，deep_shadow 实测 0；体积 / 统一半透射缺）。验收：深阴影透射率与离线光线步进参考一致；毛发自阴影无黑块；体积内部透光梯度与 raymarch 参考收敛一致。

### 17.2 v10 增补路线图优先级（并入既有阶梯，不改 §9 P0 基底次序，不重复 §6/§11/§12/§13/§14/§15/§16 条目）

| 增补特性 | 落点 | 现状（实测） | 优先级 | 依赖 |
|---|---|---|---|---|
| 统一次表面散射 SSS | `render_shading` + `gi/material` + `render_scene/shading` | 🟡 subsurface/sss 各 1 骨架，随机游走 0 | **P1** | Burley 轮廓、屏幕空间可分离卷积、随机游走参考、light_routing |
| GTAO 弯曲法线 + 镜面遮蔽 | `render_shading/gi` + `render_scene/shading` | 🟡 bent_normal/spec_occ 各 1，gtao/ssao 0 | **P1** | 水平基积分、时空双边去噪、GI 近场协同 |
| 大气天空 + 空中透视 + Nubis 体积云 | `volumetric_gpu` + `render_shading` + `render_scene` | ⬜ 相关 0；froxel 基础在 | **P1.5** | LUT 预计算、froxel、解析相位 raymarch、天气参数 |
| 毛发 / 体积深阴影统一自阴影 | `hair_gpu` + `volumetric_gpu` + `render_shading/virtual_shadow` | 🟡 deep_opacity 6 毛发在，deep_shadow 0 | **P1.5** | 分层透射、VSM 页复用、OIT 协同 |
| 可变速率着色 VRS Tier2 | `render_architecture` + `render_scene` + `temporal_upscale` | ⬜ vrs/shading_rate 0 | **P2** | `VK_KHR_fragment_shading_rate`、SRI 生成、reactive mask |
| 开阔世界流式 VHM + 植被 + WorldPartition/HLOD | `render_architecture` + `virtual_geometry_gpu` + `render_scene/geometry` | 🟡/⬜ heightfield/terrain/foliage 骨架在，WP/HLOD 0 | **P2** | VHM、RVT/SVT(§16.1.1)、流式(§13)、蒙皮缓存(§16.1.6) |

**v10 增补总原则**：① **不改 §9 P0 基底次序**，不重复 §6/§11–§16 已登记条目——6 条均经关键字实测确认为**新**赛道（🟡 骨架在整合缺位 / ⬜ 真实空白）；② **SSS / GTAO 列 P1**——皮肤次表面与接触遮蔽 / 弯曲法线是「近景人物与接触可信度」的刚需，骨架已在、增量风险可控、对角色与近景效果提升最直接；③ **大气天空 / 深阴影列 P1.5**——效果天花板高（天空与半透自阴影），依赖的 froxel / VSM 已在，增量风险低；④ **VRS / 开阔世界流式列 P2**——VRS 依赖硬件扩展与 TSR 协同，开阔世界流式依赖 SVT(§16.1.1) / 蒙皮缓存(§16.1.6) / 流式(§13) 使能器，排其后；⑤ 全部走「CPU golden → GPU kernel → 真机 parity」三步，关键路径必须与**路径追踪参考 / 解析解 / 离线数值求解**可对拍，未毕业不作生产默认；**任何增补均不引入 AI/ML/神经/LLM 推理路径**（厂商时序上采样 / 帧生成 SDK 仅渲染侧可选外部后端）。

---

## 18. v11 产品对标高级特性增补与规模六刷 + 顶级 AAA 毕业门禁收口（2026-10，纯经典数值 / 无 AI·ML·神经·LLM）

> **本版目的**：延续用户「代码有更新 / 借鉴参考优先产品 / 添加高级功能 / 兼顾性能与效果 / 达到顶级次世代 AAA 级别 / 更新优化设计文档」。本版 ① 以统一口径六刷规模（§18.0，接续 v10 §17.0 基线）；② 追加 **§18.1** 三条**经关键字实测确认为真实空白 / 仅骨架**的新赛道（杜绝与 §3–§17 已登记条目重复，杜绝分级虚高）；③ 以 **§18.2 毕业门禁**把「顶级 AAA」从「再堆特性」转为「可证明到达」——给 P1/P1.5 特性逐项交付就绪度（Definition of Done）、收口次序与帧级判据；④ 刷新 §18.3 优先级。**全程纯经典数值**，排除一切 AI/ML/神经/LLM 推理路径。

### 18.0 规模六刷（2026-10 实测，统一口径 `find pkg/<crate>/src -name '*.rs' | wc -l`）

| crate / 子系统 | v10 §17.0 | v11 实测 | 变化与说明 |
|---|---|---|---|
| `prism_render_architecture` | 680 | **680** | 帧图 / 流送 / `work_graph` / `ray_scene` 稳定；贴花·OIT·折射·海洋 LOD 骨架在 `particle/`·`water/` 下 |
| `prism_render_shading` | 344 | **344** | 渲染本体着色稳定 |
| └─ `prism_render_shading/gi/`（子系统，本版单列） | — | **246** | GI 子系统已成大规模（surfel·radiance_cache·photon·world_restir·light_tree·thin_film·specular_aa·gtao·ies_profile 全在），占 shading crate 七成以上，是本仓增量最活跃区 |
| `prism_render_scene` | 461 | **461** | 精确 SDF 图元族 + 光路由 / 色调映射 ABI 前端稳定 |
| `prism_render_visibility`（两阶段 HZB 遮挡） | 22 | **22** | 两阶段遮挡闭环稳定（footprint→projection→pyramid→query→build→cull_hzb→occlusion_resolve→two_phase→resolve） |
| `prism_render_material` | 12 | **12** | 闭包 IR / ABI 前端稳定 |
| `prism_virtual_geometry_gpu` | 25 | **25** | 软光栅主路径稳定 |
| `prism_volumetric_gpu` | 131 | **131** | froxel / 储层 / 异质介质孪生稳定（为 §17.1.4 天空·空中透视·体积云底座） |
| `prism_hair_gpu` | 72 | **74** | Marschner / 双散射 + 分层样本 GPU 孪生持续新增（+2，deep opacity ≥6 供 §17.1.6 深阴影统一） |
| `prism_physics_gpu` / `prism_physics_core` | 253 / 167 | **256 / 167** | 流体 / 断裂 / VBD / CCD / MPM / 关节族持续新增（gpu +3；本轮含布料身体代理碰撞 body 内核） |
| `prism_audio_core` / `prism_audio_spatial` | 93 / 52 | **94 / 52** | 卷积混响 / HOA / 衍射 / 房间声学；core +1（效果节点 envelope_follower） |

**口径一致性**：本表与 v7 §14.0 / v8 §15.0 / v9 §16.0 / v10 §17.0 同口径（单一 `*.rs` 文件计数，含测试文件）。**本版主要结构性增量**在 `physics_gpu`(+3)、`hair_gpu`(+2)、`audio_core`(+1) 的子系统孪生，以及**首次单列的 `gi/` 子系统 246 文件**（反映并行舰队本轮集中在 GI 子系统孪生）。渲染本体核心（architecture / shading / scene / material / virtual_geometry / visibility）**保持稳定收敛**——**本版天花板增量在下文 §18.1 三条新缺口 + §18.2 毕业门禁收口**，而非再刷特性清单。

### 18.1 产品级高级特性赛道（v11 新增，逐条实测分级）

> 三条均经 `find pkg -iname '*<kw>*' -name '*.rs'` 实测，并经全文检索确认 §3–§17 未登记为专项条目（避免重复与分级虚高）。

#### 18.1.1 着色器 PSO 预编译 + 管线缓存 — 消除运行期编译卡顿（stutter-free）
- **借鉴对象**：UE5 **PSO Precaching / Bundled PSO Cache** · DOOM/id Tech 全量 PSO 预热 · Steam Deck Fossilize 管线缓存 · D3D12 `ID3D12PipelineLibrary` / Vulkan `VkPipelineCache`。
- **算法要点**：① 把「材质 × 顶点工厂 × pass × 混合/深度状态」的 PSO 置换集合在**加载期 / 关卡切换期并行预编译**，运行期命中缓存零编译；② 置换来源 = `shader_package/permutation.rs` 现有置换枚举 + 运行期收集的「缺失 PSO」反馈清单（下次预热合并）；③ 管线缓存可**磁盘持久化 + 跨会话复用**，带驱动/设备指纹失效校验；④ 无硬件管线库平台降级为后台线程异步编译 + 占位 PSO + 完成前跳过该 draw（不阻塞主线程）。纯工程调度与缓存管理，无任何学习路径。
- **预算**：预热在加载屏 / 后台计算队列，运行期编译时间趋零；缓存磁盘占用按置换数封顶 + LRU 淘汰；预热并行度随核数伸缩。
- **效果**：**首见材质 / 首次特效不卡顿**（AAA 公认的「shader comp stutter」顽疾根除）——顶级次世代 AAA 的帧稳定性刚需。
- **落点**：`prism_render_architecture/shader_package/`（置换枚举 + 预热编排 + 磁盘缓存，现有 registry/manifest/permutation/state 5 文件深化）+ `frame_graph/`（降级异步路径）。
- **分级 / 验收**：⬜（`shader_package/` 置换骨架在，PSO 预热 / 磁盘缓存 / 缺失反馈实测 0）。验收：预热后运行期 PSO 编译次数 = 0（埋点计数）；冷启动 vs 热缓存首帧稳定度对比无 hitch；设备指纹变更后缓存正确失效重建；降级异步路径与同步路径最终像素一致。

#### 18.1.2 可见性缓冲 / RT 光线微分纹理 LOD — ray cone / ray differentials（抗纹理走样）
- **借鉴对象**：Igehy **Ray Differentials** · Akenine-Möller **Texture LOD for RT（ray cones）** · UE Nanite vis-buffer 解析导数 · 影视渲染器的 differential geometry。
- **算法要点**：① 在**可见性缓冲延迟着色**与 **RT 命中着色**里，用相邻像素 / 光线锥角解析推导 UV 的屏幕空间偏导 `(ddx,ddy)`，驱动各向异性 mip 选择——取代光栅硬件自动导数在 vis-buffer / RT 下的缺失；② ray cone 沿路径累积张角（表面曲率 + 粗糙度展宽），反射 / 折射 / 多次弹跳逐段更新锥宽；③ 与 §12 可见性缓冲延迟纹理化 + §16.1.1 SVT 反馈协同（正确 mip → 正确 SVT 页请求）。纯解析微分几何，无学习路径。
- **预算**：每像素 / 每命中几条标量 FMA + 一次各向异性采样；无额外 pass。
- **效果**：vis-buffer / RT 下**纹理不走样、不过采样**（远处砖缝 / 栅栏不闪、不糊），SVT 页请求精确——顶级几何密度下的纹理保真刚需。
- **落点**：`prism_render_scene`（vis-buffer 着色求导）+ `ray_scene`（命中点 ray cone 累积，现有 `ray_cone` 1 文件深化）+ `render_material`（mip 选择接口）+ `render_architecture`（SVT 反馈耦合）。
- **分级 / 验收**：🟡→🟢(CPU golden，端到端软件采样器已贯通)（**v11 本版落地**：`render_material` 新增一条**完整纯经典解析 CPU 金标软件采样器**，覆盖 vis-buffer / RT 下缺失固定功能采样器的全链路 `address → LOD → anisotropy → residency → fetch → filter`：① `texture_lod/`——`TriangleLodConstant`(每三角 texel/world 面积比 Δ) · `RayCone`(RTGems 2019 锥角传播：pinhole/反射/散射逐段展宽) · `RayDifferential`(Igehy 1999 屏幕空间 UV 偏导，各向同性+各向异性，含 `major_axis_uv`) · `cone_mip_level`/`AnisotropicMip` 共享钳位 · `anisotropic_taps`(≤16 taps 均权) · `VirtualTexture` SVT 页驻留反馈(trilinear mip 对→PageRequest)；② `texture_addressing/`——`WrapMode`(Repeat/Clamp/Mirror/Border) UV 级 `address_uv`；③ `texture_sample/`——`resolve_differential`(主可见命中，textureGrad 等价各向异性) / `resolve_cone`(次级光线，各向同性 trilinear) 把上述阶段编排为单一 `SampleResolved`；④ `texture_filter/`——`TexelSource` trait + `bilinear`(逐角 `wrap_texel` 寻址，含 ClampToBorder 边界色) / `trilinear`(mip 对混合) / `filter_resolved`(各向异性 taps 均权积分) / `trilinear_bicubic`(**Catmull-Rom 双三次 trilinear**——两 bicubic mip 按 LOD 分数混合，高质量 C1 放大采样模式消除 bilinear 菱形缝) / `filter_resolved_bicubic`(各向异性 `trilinear_bicubic` taps 均权积分，补齐 cubic 的 resolve→fetch→filter 全链路) / `bicubic_catmull_rom`(**Catmull-Rom（Keys a=-1/2）16-tap 双三次放大滤波**——C1、插值过原纹素、线性斜坡精确重建，用于 lightmap/UI/upsample 预滤；可分离生产路径与独立 4×4 双重求和参考逐位互对拍，另加单位和/权重对称/纹素中心插值/线性重建/常数保持/NaN 全防外部锚点) + `bspline_cubic`/`bspline_cubic_fast`(**三次均匀 B 样条平滑滤波**——非负凸核、恒不过冲/振铃，用于高度场/地形/数据纹理平滑上采样；`bspline_cubic` 可分离 16-tap 参考 + `bspline_cubic_fast` Sigg&Hadwiger 四-bilinear-tap 折叠(GPU 实现同构、复用已测 `bilinear`)，两条独立路径跨全部 wrap 模式逐位互对拍，另加单位和/非负性/对称/常数保持/线性重建/硬台阶不过冲/NaN 全防外部锚点) + `cubic_mitchell`/`mitchell_netravali_weights`(**Mitchell-Netravali (B,C) 参数化三次族**——统一上述两核：权重在 (0,1/2) 精确退化为 Catmull-Rom、在 (1,0) 退化为 B 样条，整采样器在 noise 上跨 wrap 模式与 `bicubic_catmull_rom`/`bspline_cubic` 互对拍；另备推荐均衡 `B=C=1/3` 重采样核常量，单位和/常数保持/NaN 全防)；⑤ `texture_codec/`——AAA 纹理落盘主格式 **块压缩解码**：`color_block`(BC1 式 RGB565+2bit 索引色块) · `alpha_block`(BC4 式单通道 8/6 值调色板+3bit 索引) · `formats`(公共 `decode_bc1`/`decode_bc2`/`decode_bc3`/`decode_bc4`/`decode_bc5`) · `source`(`BcTexelSource`——把 BC 块按 4×4 tile 行主序铺入内存/SVT 页，实现 `TexelSource`，闭合`压缩块→解码→源→过滤→采样` 全链路；逐纹素整块解码的正确性金标，供 GPU 常驻源逐位对拍；格式覆盖已扩至 **BC7(mode 4/5/6) 与 `ETC2_RGB8` 基础模式**——这些含不可解模式的格式在 `BcTexelSource::new` 构造期即逐块校验，遇 BC7 分区或 ETC2 T/H/planar 块返回 `UndecodableBlock` 拒绝错解，故采样期绝不静默回退黑块)，纯整数解码（含 BC2 显式 4-bit alpha，硬边 mask 不振铃）、无 AI/ML、GPU 孪生 ±1 LSB 容差内可对拍；再加 `bc7`(**BC7 单子集无分区模式 4/5/6** 解码——mode 6：7bit+p-bit 端点扩 8bit、4bit 共享索引；mode 5：RGB 7bit(MSB 复制扩 8)+独立 8bit alpha 端点、2bit 色/α 双索引、2bit rotation 通道互换；mode 4：RGB 5bit + 独立 6bit alpha 端点、2bit/3bit 双精度双索引块 + idxMode 位路由色/α 精度、2bit rotation 通道互换；`decode_bc7` 按 mode 分派，分区模式 0-3/7 显式返回 `UnsupportedMode`（拒绝错解）；端点纹素逐位精确；BC7 分区模式 0-3/7 待校验 Khronos 分区/锚点表后再纳入)；再加 `bc6h`(**BC6H 单子集模式 11** HDR 解码——10bit 直接 RGB 端点、4bit 索引(锚点 3bit)、`half_bits_to_f32` 无损整数半精→f32(含次正规/inf/nan)、unsigned `Unquantize`+`(q*31)>>6` 收尾；`decode_bc6h_unsigned`/`decode_bc6h_signed` 均仅识别 mode 11(有/无符号两条路径)，delta 模式 12-14 / 分区 1-10 显式返回 `UnsupportedMode`) 与 `bitio`(LSB-first `BitReader` 共享阅读器，从 bc7 抽取供 bc6h 复用)；再加 `etc2`(**ETC2_RGB8 基础模式(ETC1 兼容)块解码**——`individual`(两 `RGB444` 子块色) / `differential`(单 `RGB555` 基色+带符号 3bit 逐通道 delta) 双基础模式纯整数解码为 `RGBA8`；`etc2_rgb8_mode` 以 diff/flip 位与 `base+delta` 溢出 `0..=31` 分类 T/H/planar 三扩展，`decode_etc2_rgb8` 对扩展显式返回 `UnsupportedMode`(拒绝错解，沿用 BC7 分区模式纪律)；ETC 无插值容差故 GPU 硬解逐纹素位精确，9 条手工校验 KAT；移动端 AAA 落盘基线；再加 **ETC2_RGB8 基础模式(ETC1) 块编码器**(iPACKMAN/ETC1 式快速编码：两 flip 朝向 × {differential, individual} 候选，子块基色取量化均值，8 码字 × 4 索引逐纹素暴力最小平方误差，择总误差最小候选；differential 的 delta 钳制使 `base+delta`∈`0..=31` 保证输出不混入 T/H/planar(解码器必判为 differential 基础模式)，往返实解码器 `decode_etc2_rgb8` 逐位对拍，6 条外部锚点含无损平块/两色竖分往返、基础模式不混叠、公差/有界误差/确定性，commit `c9b1f3a2f`)；T/H/planar + `EAC` alpha 编码器为后续)；⑥ `normal_map/`——切线空间**法线重建 + 细节混合**：`reconstruct`(BC5 式 RG / 旧 DXT5nm 式 AG 双通道→`z=sqrt(max(0,1-x²-y²))` 半球重建，over-unit 退化不出 NaN) · `blend`(**Reoriented Normal Mapping** Barré-Brisebois&Hill 2012 为主，另备 whiteout 偏导 / UDN / linear 三档降级)，直接吃采样器输出的 unorm `(R,G)` 产出单位法线并叠加 mesostructure 细节；满足「平细节→基法线」「平基→细节法线」两条正确性恒等式，输出恒为单位向量，纯解析、无 AI/ML；⑦ `texture_mipgen/`——解码后 `RGBA8` 的**精确盒式 mip 链生成**：`srgb`(IEC 61966-2-1 sRGB↔scene-linear 传递函数，全 256 值 ≤1 LSB 往返、NaN/±inf 全防) · `box_filter`(`Rgba8Image` 行主序层 + `box_downsample` 2×2 盒式归约 + `generate_mip_chain`，`ColorSpace::Srgb` 对 albedo 走线性光平均再编码/alpha 保持整数、`ColorSpace::Linear` 对 normal/rough/AO/height 数据图直接整数四舍五入，遵循 GL `max(1,dim>>1)` 非方 POT 维度=1 携带规则，NPOT 多相盒暂不纳入) · `windowed`(**可分离 Lanczos-2/3 窗 sinc 2× 归约**，`windowed_downsample`/`generate_mip_chain_windowed`，比盒式更能压制 mip 闪烁；sinc·sinc 窗核解析、紧支撑 `|x|<a`、边缘钳位保常数，两趟可分离在 scene-linear 浮点工作缓冲内一次性滤波再编码，核峰/对称/紧支撑/常数保持/值域全验) · `kaiser`(**可分离 Kaiser 窗 sinc 2× 归约**，`kaiser_downsample`/`generate_mip_chain_kaiser`+`KaiserFilter{radius,beta}`，`DirectXTex`/NVTT 级最高质量 mip 滤波：单一形状参数 `beta` 连续权衡主瓣宽度(锐度)↔阻带衰减(振铃)，取代 Lanczos 以瓣数 `a` 固定折中；修正 Bessel `I0` 走 Maclaurin 级数 f64 累加、对参考值校验，窗形/单调非增/`beta=0` 退化为矩形窗/边缘 `w(r)=1/I0(beta)` 非零/常数&值域保持/mip 链结构全验) · `resample_core`(把 Lanczos/Kaiser 共用的 **gamma 正确 scene-linear 工作缓冲重采样内核**——色彩升降、钳边可分离卷积、`sinc`、可归约维规则——抽出共享，各滤波仅声明各自 1-D 核，避免重采样回路重复、保持优秀目录结构) · `alpha_coverage`(**mip 链 alpha-test 覆盖率保持**后处理——Castano/NVTT `setAlphaTestCoverage`、UE/Unity「preserve coverage」：按单调二分求每级 alpha 缩放使其过阈 texel 占比对齐 mip0，杜绝 alpha-test 植被/贴花远处变稀/闪烁；滤波无关、可组合任意归约器；覆盖率上下界/缩放单调性/scale=1 恒等/钳饱和/目标覆盖率逐 texel 复原/整链 0.5 覆盖率恢复全验)，纯整数/解析、无 AI/ML，GPU 计算下采样器逐位对拍金标。退化几何 / 非有限输入 / 掠射角全防 NaN/inf，GPU 孪生可逐位对拍，无任何 AI/ML 路径；crate 共 **281 单测绿**、`cargo check` + `cargo clippy` 均零警告，全部超越函数经 `bevy_math::ops`（nostd-libm）走确定性路径以保证 GPU 逐位对拍，commits `bdad48e13`→`c38fe544f`（新增 `decode_bc2` 显式 4-bit alpha 块解码 + `texture_mipgen` gamma 正确盒式/Lanczos 窗 sinc mip 生成 + 超越函数 libm 确定化 + clippy 清零 + `BC7 mode 4/5/6` 单子集解码 + `decode_bc7` 分派 + `BC6H mode 11` 单子集 HDR 解码(无符号 UF16 + 有符号 SF16，复用同一已校验位布局，仅端点二补码/有符号反量化不同) + 共享 `bitio` 阅读器 + `texture_filter` Catmull-Rom 双三次放大滤波（可分离×直接双重求和互对拍） + 三次 B 样条平滑滤波（直接 16-tap × Sigg&Hadwiger 四-bilinear-tap 折叠互对拍） + Mitchell-Netravali (B,C) 参数化三次族（退化为 CR/B 样条双向对拍） + Kaiser 窗 sinc mip 归约(修正 Bessel I0 级数 + 可调 beta 锐度/振铃折中) + 抽出 `resample_core` 共享 gamma 正确重采样内核（Lanczos/Kaiser 复用，commits `0e0592f9c`→`12b68afa4`）+ `normal_map/mipmap`（**方差保持法线 mip + Toksvig 高光抗锯齿** Toksvig 2005：2×2 法线+粗糙度归约把均值法线因子纹素分歧而**损失的长度**(= 粗糙 mip 再也无法以几何表达的法线方差)烘焙进粗糙度，使高光在各 mip 稳定不闪，对齐 `UE`/Frostbite/`CryEngine`；`average_unit_normals` 均值+归一化 + `power_from_roughness`/`roughness_from_power` GGX α↔Blinn 指数互逆 + `toksvig_factor`/`toksvig_roughness`(只增不减粗糙度) + `reduce_normal_roughness_2x` 法线+粗糙度联合归约；纯解析、无 AI/ML、CPU 金标与 GPU 孪生对拍，commit `41a1aa261`）+ `texture_mipgen/premultiplied`（**预乘 alpha mip 归约** Porter-Duff/Blinn associated alpha：按覆盖率加权颜色、分别平均预乘颜色与覆盖率再反预乘，使全/半透明 texel 不把其 RGB 渗入 alpha 混合边缘，消除镂空植被/贴花/UI/粒子的暗边光晕；sRGB 路径在 scene-linear 预乘，全透明足迹回退到无权均值不出 NaN，不透明图与盒式逐位一致，对齐 `UE`/Unity/`DirectXTex` alpha 纹理 mip 构建；`premultiplied_box_downsample`/`generate_mip_chain_premultiplied`，纯解析、无 AI/ML、GPU 孪生 ±1 LSB 可对拍，commit `7ccb11e0c`）+ `texture_codec/encode/bc1`（**BC1/DXT1 块编码器**——首条纹理*压缩*落盘路径，`decode_bc1` 的逆：PCA 主轴幂迭代（3×3 协方差）拟合主色轴播种端点→投影取极值端点→`RGB565` 量化→最近索引分配→最小二乘端点重拟合（2 轮，保留 SSD 更低者，优于裸 min/max 包围盒拟合）；不透明块强制 4 色模式(`c0>c1`，必要时交换端点并重标索引 0↔1/2↔3)、任一 texel alpha<128 则走 1-bit 穿透 3 色模式(`c0<=c1`，透明 texel→索引 3)，输出逐位往返过 `decode_bc1`；对齐 `squish`/`DirectXTex`/`NVTT`，纯解析/整数、无 AI/ML、GPU 孪生同构可逐位对拍，commit `f093cece9`）+ `texture_codec/encode/bc4`（**BC4/RGTC 单通道块编码器**——第二条纹理*压缩*落盘路径，`decode_bc4` 的逆：min/max 端点播种，同时尝试 8 值模式(`r0>r1`，端点 hi,lo)与 6 值模式(`r0<=r1`，端点 lo,hi + 硬 0/255 槽)，各自最近索引分配后保留 SSD 更低者，逐位往返过 `decode_bc4`；复用 `alpha_block::channel_palette` 调色板构建，对齐 `squish`/`DirectXTex`/`NVTT` 的 RGTC 编码，纯解析/整数、无 AI/ML、GPU 孪生同构可逐位对拍，commit `f191920fb`）+ `texture_codec/encode/{bc3,bc5}`（**BC3/DXT5 与 BC5/RGTC2 块编码器**——复用已落地的 BC1/BC4 两块原语组合而非重新推导：BC3 = 不透明 BC1 色块(强制 alpha=255 锁定四色模式，覆盖率只走独立 alpha 半块)拼接 `encode_bc4` 压缩的 alpha 通道，字节序 alpha 在前色块在后、逐位往返 `decode_bc3`；BC5 = 对 `R`/`G` 两通道各跑一次 `encode_bc4`(法线 `XY`，`Z` 采样时半球重建故不落盘)、逐位往返 `decode_bc5`；对齐 `squish`/`DirectXTex`/`NVTT`，纯整数、无 AI/ML、GPU 孪生同构可逐位对拍，commit `f5499fae3`）+ `texture_codec/encode/bc2`（**BC2/DXT3 显式 alpha 块编码器**——BC3 的锐边 alpha 姊妹格式：16 个纹素 alpha 各量化成最近 4-bit nibble(`round(a*15/255)`，解码端 `nibble*17` 复制还原，17 的倍数含硬 0/255 端点逐位精确、硬掩码边不插值不振铃)小端铺入字节 `[0..8]`、色块复用不透明 `encode_bc1` 铺字节 `[8..16]`、逐位往返 `decode_bc2`；至此纹理*压缩*落盘路径补齐 **BC1/BC2/BC3/BC4/BC5 DXT/RGTC 全家桶**，对齐 `squish`/`DirectXTex`/`NVTT`，纯整数、无 AI/ML、GPU 孪生同构可逐位对拍，commit `912d79a32`）））；再加 `texture_codec/encode/bc7`（**BC7/BPTC mode 6 单子集块编码器**——首条现代 LDR 4 通道（含 alpha）*压缩*落盘路径，也是唯一无需 Khronos 分区/锚点表、可纯解析编码的 BC7 模式：16 个 `RGBA` 纹素 4D 协方差幂迭代取主色轴→投影极值播种端点→每端点量化成 7bit/通道+共享 p-bit（两种奇偶择低误差，扩 8bit 同 `(v<<1)|p`）→16 级 4-bit 权重最近索引分配→锚点规则（texel-0 高位必 0，必要时交换端点并全反转索引）→最小二乘端点重拟合（2 轮保留 SSD 更低者）→LSB-first 位写入器（`decode_bc7_mode6` 位读取器的逆），逐位往返 `decode_bc7_mode6`；分区模式 0-3/7 仍待 Khronos 表或 GPU 硬解 oracle，故本条只补单子集 mode 6；对齐 `DirectXTex`/`NVTT`/`ispc_texcomp`，纯整数/`f64`、无 AI/ML、GPU 孪生同构可逐位对拍，commit `1a2b5643f`）；再加 `texture_codec/encode/bc6h`（**BC6H/BPTC 无符号 mode 11 HDR 块编码器**——首条 HDR 纹理*压缩*落盘路径，`decode_bc6h_mode11_unsigned` 的逆，与 BC7 mode 6 同为无需 Khronos 分区/锚点表的单子集模式：输入为半精度**位模式** `u16` 三通道(BC6H 原生量化-对数存储域，负/inf/NaN 钳到可表示 `[0,0x7BFF]`)→每通道映射到 finish 前中间域 `T=round(h*64/31)`(解码端 `(q*31)>>6` 的逆)→16 个中间目标 3D 协方差幂迭代取主轴→投影极值播种 16bit 端点→量化成 10bit 直接端点(`unquantize_unsigned` 的逆 `q=v/64`，端点饱和 0/0xFFFF)→按**精确 finish 链**(端点逐位重建后 `interp_finish` 真实半精度输出)取最近 16 级 4bit 索引→锚点规则(texel-0 高位必 0，交换端点并全反转索引)→中间域最小二乘端点重拟合(2 轮保留误差更低者)→LSB-first 位写入器，逐位往返 `decode_bc6h_mode11_unsigned`、经 `decode_bc6h_unsigned` 分派识别为 mode 11；delta 模式 12-14 / 分区 1-10 及有符号 profile 待后续；对齐 `DirectXTex`/`NVTT`/`ispc_texcomp`，纯整数/`f64`、无 AI/ML、GPU 孪生同构可逐位对拍，commit `2f3b44b51`）；再加 `texture_codec/encode/bc6h`（**BC6H/BPTC 有符号 mode 11 HDR 块编码器**——补齐 BC6H HDR *压缩*落盘的有符号（`SF16`，如法线/速度/位移等带符号数据图）profile，`decode_bc6h_mode11_signed` 的逆、与无符号 profile 共用 mode-11 位布局（故同样无需 Khronos 分区/锚点表）：半精度**位模式** `u16` 三通道按**符号-幅值**解释（inf/NaN 钳到 ±`0x7BFF`）→每通道映射到 finish 前**有符号**中间域 `T`（保符号，解码端 `(|q|*31)>>5` 符号-幅值 finish 的逆）→16 个有符号中间目标 3D 协方差幂迭代取主轴→投影极值播种→量化成 10bit **二补**端点（`unquantize_signed` 的逆，幅值钳 `[0,511]`）→**在 finish 前有符号 q 域**取最近 16 级 4bit 索引（规避符号-幅值半精度直接比较的符号歧义）→锚点规则（texel-0 高位必 0，交换端点并全反转索引）→对称取整最小二乘端点重拟合（2 轮保留误差更低者）→LSB-first 位写入器（6×10bit 二补端点场），逐位往返 `decode_bc6h_mode11_signed` 并经 `decode_bc6h_signed` 分派识别为 mode 11（正/负 flat、正/负梯度、±簇、确定性、mode 标记、锚点、`finish_signed` 符号-幅值等 9 条新测）；至此 BC6H 单子集 mode 11 有/无符号编解码**双向闭合**，delta 模式 12-14 / 分区 1-10 仍待后续；对齐 `DirectXTex`/`NVTT`/`ispc_texcomp`，纯整数/`f64`、无 AI/ML、GPU 孪生同构可逐位对拍，commit `c069c08a1`）；再加 `texture_codec/encode/bc7`（**BC7/BPTC mode 5 LDR+alpha 单子集块编码器**——补齐现代 LDR 含 alpha *压缩*落盘的第二种无分区模式，适配色/α 需不同插值方向的内容（渐变上的遮罩，mode 6 的单一共享索引无法表达）：逐一试四种**通道 rotation**（解码端插值后把某通道与 alpha 互换；互换为自逆置换，故在互换后的通道空间内拟合）→**颜色**（swap 后 RGB 三通道）3D 协方差幂迭代主轴→投影极值播种→每通道量化成 7bit（MSB 复制扩 8，`expand_rep(v,7)` 的逆，128 码穷举取最近重建）→2bit 最近索引→颜色锚点（texel-0 索引高位必 0，交换端点并全反转 `3-idx`）→3D 最小二乘端点重拟合（3 轮）；**alpha**（swap 后第 4 通道）min/max 直接 8bit 端点播种→2bit 最近索引→独立 alpha 锚点→标量最小二乘重拟合；取颜色+alpha 合计误差最低的 rotation，LSB-first 写入（6×7bit RGB 端点 + 2×8bit alpha 端点 + 2bit rotation + 独立 2bit 色/α 索引块各含 1bit 锚点），逐位往返 `decode_bc7_mode5` 并经 `decode_bc7` 分派识别为 mode 5（flat RGBA、色渐变+平 alpha、平色+alpha 渐变验证独立 alpha 索引、R/A 相关通道经 rotation 复原、texel-0 极值锚点、确定性、mode 标记+分派 7 条新测）；对齐 `DirectXTex`/`NVTT`/`ispc_texcomp`，纯整数/`f64`、无 AI/ML、GPU 孪生同构可逐位对拍，commit `7b05c1ede`）；再加 **BC7/BPTC mode 4 双精度双索引块编码器**（第三种无分区单子集模式：5bit RGB + 独立 6bit alpha 端点、2bit 与 3bit 两个不同精度索引块 + 1bit `idxMode` 路由（色/α 谁拿高精度索引）、四通道 rotation × 两 `idxMode` 共 8 候选择优；颜色（swap 后 RGB）3D 主轴播种→按路由权重表量化成 5bit（MSB 复制扩 8，穷举最近码）→最近索引→色锚点（高位必 0，交换端点+全反转）→3D 最小二乘重拟合，alpha（swap 后第 4 通道）min/max 播种→6bit 量化→独立锚点→标量最小二乘重拟合，取颜色+alpha 合计误差最低者；LSB-first 写入（6×5bit RGB + 2×6bit alpha + 2bit rotation + 1bit idxMode + 2bit 索引块（1bit 锚）+ 3bit 索引块（2bit 锚）），逐位往返 `decode_bc7_mode4` 并经 `decode_bc7` 分派识别 mode 4（flat RGBA、色渐变胜 flat、细 alpha 渐变自动选 `idxMode`=0 由 3bit 索引驱动 alpha、细色+块状 alpha 自动翻到 `idxMode`=1 由 3bit 索引驱动颜色、通道 rotation 复原、texel-0 极值锚点、确定性、mode 标记+分派 8 条新测）；至此 BC7 单子集无分区 mode 4/5/6 编码器全闭合，对齐 `DirectXTex`/`NVTT`/`ispc_texcomp`，纯整数/`f64`、无 AI/ML，commit `c84d67c32`）；再加 **BC4/BC5 `SNORM` 带符号块解码器**（AAA 带符号切线空间落盘：`snorm_block` 把两枚端点字节按二补 `i8` 解读、保留 `0x80`→`-127` 对称化重映射，`r0>r1` 走八值 sevenths、否则六值 fifths + `-127`/`127` 硬终端，`decode_bc4_signed` 出 16 个 `i8`、`decode_bc5_signed` 拆 R/G 双通道出 `[[i8;2];16]`，供物件空间法线/带符号位移/运动矢量通道，插值 ±1 LSB 实现定义故测断序/界/终端不断确值，commit `f4ce6f099`）；再加 **BC4/BC5 `SNORM` 带符号块编码器**（`snorm` 编码器补齐 RGTC 带符号压缩落盘：逐 tile 带符号 `min`/`max` 端点（钳 `-127..=127`）、八值/六值双模式按块平方误差择优、逐纹素最近调色板索引，`encode_bc4_signed` 出 8 字节、`encode_bc5_signed` 双通道出 16 字节，逐位往返 `decode_bc4_signed`/`decode_bc5_signed`，纯整数无 AI/ML，commit `d8eeb8836`）。**仍缺（🟡 DoD-2）**：vis-buffer 延迟着色求导 + RT 命中锥宽累积的 GPU kernel 接线、以及 `TexelSource` 对真实 SVT 常驻页的 GPU 实现（落点 `prism_render_scene`/`ray_scene`/`render_architecture`，并行舰队在途）。验收：与光栅硬件导数参考的 mip 选择逐像素一致（容差内）；掠射 / 远景纹理无摩尔纹、无过采样噪点；反射多弹跳锥宽随曲率 / 粗糙度单调展宽，与解析参考一致。

#### 18.1.3 GPU 驱动粒子渲染整合 — GPU 排序 + mesh/ribbon 粒子 + 软粒子
- **借鉴对象**：UE **Niagara** GPU 粒子渲染侧 · Frostbite GPU 粒子 · 影视体积粒子。本条**只覆盖渲染侧整合**，粒子**模拟**规格见 `prism_particle_engine_design_zh.md`，不重复。
- **算法要点**：① GPU 粒子按视深 **GPU 基数 / 双调排序**后喂透明合成，与 §16.1.4 矩不变 OIT 共享合成缓冲（避免逐帧 CPU 回读排序）；② **mesh 粒子 / ribbon 缎带**走 indirect draw，法线 / motion vector 馈入 VisBuffer 与时序历史（粒子参与 TSR 不拖影）；③ **软粒子（soft particle）**按场景深度做近表面透明度淡出，消除粒子与几何交界硬边；④ 与 §6 体积雾 / §17.1.4 云协同光照。纯排序 + 解析淡出 + 经典合成，无学习路径。
- **预算**：GPU 排序 O(n log n) 片上；mesh/ribbon 走现有 indirect；软粒子一次深度读取；半分辨率可选。
- **效果**：海量 GPU 粒子**正确排序透明、不穿插硬边、参与运动模糊 / TSR**——特效（火 / 烟 / 魔法 / 碎屑）达顶级 AAA 观感。
- **落点**：`render_scene` / `render_architecture` 下 `particle/`（排序 + indirect + 软粒子，现有 `soft_particle` 3 文件骨架深化）+ `render_shading/oit`（合成共享）+ `motion/`（motion vector）。
- **分级 / 验收**：🟡（`soft_particle` 3 文件骨架在，GPU 排序 / mesh 粒子 / motion vector 馈入实测 0）。验收：GPU 排序后透明顺序与 CPU 参考全序一致；软粒子交界无硬边且淡出曲线与解析一致；mesh/ribbon 粒子 motion vector 使 TSR 无拖影；与 OIT 合成口径逐像素一致。

### 18.2 顶级次世代 AAA 毕业门禁与收口（v11 新增，把「顶级」从堆特性转为可证明）

> 本节不新增特性，而是把 §3–§17 已登记的 **P1 / P1.5** 特性收敛为一张**交付就绪度（Definition of Done, DoD）矩阵 + 收口次序 + 帧级顶级判据**。目的：回答「怎么证明到达顶级次世代 AAA」，并把并行舰队的零散落地**锚定到统一毕业标准**上。

#### 18.2.1 单特性毕业判据（Definition of Done，每条 P1/P1.5 适用）
一个特性视为**毕业（✅ 生产默认）**需同时满足：
1. **CPU golden 可对拍**：核心数值路径有 CPU 参考实现，与路径追踪 / 解析解 / 离线数值求解在量化容差内收敛（白炉测试类用能量 ≈1）。
2. **GPU parity**：GPU kernel 与 CPU golden 在真机（有 GPU，允许跨沙盒验证）逐像素 / 逐元素 parity（容差内），含边界用例（空输入 / 溢出 / 跨 workgroup 边界）。
3. **预算达标**：在 1440p→4K / 60fps 目标档内满足本特性 §x 列出的时间 / 显存预算，重载有降级档且降级无崩溃。
4. **时序稳定**：与 TSR / 历史累积协同无鬼影 / 无闪烁 / 无拖尾（动态场景实测）。
5. **纯经典数值红线**：无任何 AI/ML/神经/LLM 推理路径；厂商上采样 SDK 若接入仅为可选外部后端，默认经典路径。
6. **无假实现**：无 `todo!/unimplemented!/占位常量冒充结果`；模块含 `//!` 文档 + `# References` + 防御性 clamp + `#[cfg(test)]`。

#### 18.2.2 P1 / P1.5 特性收口矩阵（锚定毕业判据）

| 特性（出处） | 现状 | 毕业缺口（DoD 未满足项） | 收口判据（帧级可见） |
|---|---|---|---|
| 统一次表面散射 SSS（§17.1.1，P1） | 🟡 骨架 | 随机游走 BSSRDF 参考缺（DoD-1）；屏幕空间可分离卷积 GPU parity 缺（DoD-2） | 近景皮肤 / 蜡 / 玉透光自然，与随机游走参考收敛，无边缘漏光 |
| GTAO 弯曲法线 + 镜面遮蔽（§17.1.2，P1） | 🟡 骨架 | 时空双边去噪 + GI 近场协同未连（DoD-4） | 接触处暗部物理正确、无 halo、时序不闪 |
| 材质多散射能量守恒（§6.251 / §9.373，P1） | 🟡 | 白炉测试能量 ≈1 收敛待补（DoD-1） | 粗糙金属 / 分层材质不发灰，白炉能量守恒 |
| GPU 蒙皮缓存 + WPO 馈入 RT/Nanite（§16.1.6，P1） | ⬜ | 蒙皮缓存 + BLAS refit 全缺（DoD-1/2） | 动画几何在 RT 反射 / 软阴影无撕裂残影 |
| 稀疏虚拟纹理 SVT + 反馈缓冲（§16.1.1，P1） | ⬜ | 页表 / 反馈回读 / 流送全缺（DoD-1/3） | 海量唯一纹理固定显存下无 pop、无糊 |
| 着色器 PSO 预编译（§18.1.1，P1） | ⬜ | 预热 / 磁盘缓存 / 缺失反馈全缺（DoD-3） | 首见材质 / 特效零编译卡顿（埋点编译次数=0） |
| 光线微分纹理 LOD（§18.1.2，P1.5） | 🟡 (CPU golden ✅ 端到端软件采样器 + BC 块解码贯通) | DoD-1 CPU 金标已落地并扩展为完整软件采样器 `address→LOD→aniso→residency→fetch→filter`，再加 `texture_codec` BC1/BC2/BC3/BC4/BC5 + BC7 mode 4/5/6 + BC6H mode 11 + ETC2 RGB8 base-mode(ETC1) 块解码 + `BcTexelSource` 闭合 `压缩块→采样` 全链路、`texture_mipgen` gamma 正确盒式 + Lanczos-2/3 窗 sinc mip 生成（`render_material`，528 测，`bdad48e13`→`55852865d`，含 `normal_map` 法线重建+RNM 细节混合 + BC2 显式 alpha 解码 + BC7 mode 4/5/6 单子集解码 + `decode_bc7` 分派 + BC6H mode 11 单子集 HDR 解码(UF16+SF16 有/无符号) + 共享 bitio 阅读器 + sRGB/linear box/Lanczos/Kaiser 窗 sinc mip 链（共享 `resample_core` 重采样内核） + Gaussian 严格非负平滑 mip 归约（零过冲/振铃，用于粗糙度/高度/覆盖率图的软预滤） + Triangle/Bartlett 三角 mip 归约（无参数线性 B 样条帐篷核，比盒式更软的非负低通） + B 样条平滑三次 trilinear 采样模式（`trilinear_bspline`/`filter_resolved_bspline`，非负零过冲，用于高度场/地形/SDF 数据纹理平滑放大）+ alpha-test 覆盖率保持后处理（Castano/NVTT）+ Catmull-Rom 双三次放大 + B 样条平滑(直接×fast 四-tap 对拍) + Mitchell-Netravali (B,C) 参数化三次族滤波 + 方差保持法线 mip + Toksvig 高光抗锯齿（损失法线方差烘焙进粗糙度）+ 预乘 alpha mip 归约（消除透明边缘暗边光晕）+ **BC1/DXT1 块编码器**（首条纹理*压缩*落盘路径，PCA 主轴+最小二乘端点拟合、不透明 4 色/1-bit 穿透两模式，逐位往返 `decode_bc1`，对齐 squish/DirectXTex/NVTT） + **BC4/RGTC 单通道块编码器**（第二条压缩落盘路径，min/max 端点、8 值/6 值双模式择优、逐位往返 `decode_bc4`） + **BC3/DXT5 + BC5/RGTC2 块编码器**（组合 BC1+BC4 原语：BC3 不透明色块+BC4 alpha、BC5 双 BC4 通道，逐位往返 `decode_bc3`/`decode_bc5`）+ **BC2/DXT3 显式 4-bit alpha 块编码器**（锐边掩码不插值，nibble 量化逐位往返 `decode_bc2`，补齐 BC1/BC2/BC3/BC4/BC5 DXT/RGTC 压缩全家桶）+ **BC7/BPTC mode 6 单子集块编码器**（首条现代 LDR 含 alpha 压缩落盘，4D RGBA PCA 主轴+7bit/p-bit 端点+最小二乘重拟合+锚点规则，逐位往返 `decode_bc7_mode6`，无需分区表）+ **BC6H/BPTC 无符号 mode 11 HDR 块编码器**（首条 HDR 压缩落盘，半精度位模式输入、中间域 PCA 主轴+10bit 直接端点+精确 finish 链最近索引+最小二乘重拟合，逐位往返 `decode_bc6h_mode11_unsigned` 并经 `decode_bc6h_unsigned` 分派，无需分区表）+ **BC6H/BPTC 有符号 mode 11 HDR 块编码器**（补齐 `SF16` 带符号 HDR 压缩落盘，符号-幅值半精度输入、有符号中间域 PCA + 10bit 二补端点 + finish 前有符号 q 域最近索引 + 对称最小二乘重拟合，逐位往返 `decode_bc6h_mode11_signed` 并经 `decode_bc6h_signed` 分派，BC6H mode 11 有/无符号编解码双向闭合）+ **BC7/BPTC mode 5 LDR+alpha 单子集块编码器**（第二种无分区模式：3D RGB PCA 色端点 7bit(MSB 复制) + 独立 8bit alpha 端点、2bit 色/α 双索引、四 rotation 择优、颜色+alpha 各自最小二乘重拟合与 1bit 锚点，逐位往返 `decode_bc7_mode5` 并经 `decode_bc7` 分派）+ **BC7/BPTC mode 4 双精度双索引块编码器**（第三种无分区模式：5bit RGB + 独立 6bit alpha 端点、2bit/3bit 双精度索引块 + 1bit `idxMode` 路由色/α 精度、四 rotation × 两 `idxMode` 择优、颜色与 alpha 各自 MSB 复制量化+最小二乘重拟合+各自锚点，逐位往返 `decode_bc7_mode4` 并经 `decode_bc7` 分派，细 alpha 渐变自动选 `idxMode`=0 / 细色自动选 `idxMode`=1，BC7 单子集 4/5/6 编码器全闭合）+ **ETC2 RGB8 基础模式(ETC1) 块解码器**（移动端 AAA 落盘基线：`individual`/`differential` 双基础模式纯整数解码为 `RGBA8`，T/H/planar 三扩展经 `etc2_rgb8_mode` 分类后显式 `UnsupportedMode` 拒绝错解，9 条手工 KAT，commit `d4fbd56d1`）+ **ETC2 RGB8 基础模式(ETC1) 块编码器**（移动端首条 ETC 压缩落盘：两 flip 朝向 × {differential, individual} 候选、子块量化均值基色 + 8 码字 × 4 索引逐纹素暴力最小平方误差择优、differential delta 钳制保证输出恒为基础模式不混入 T/H/planar，逐位往返 `decode_etc2_rgb8`，6 条外部锚点，commit `c9b1f3a2f`）+ **BC4/BC5 `SNORM` 带符号块解码器**（带符号切线空间落盘：二补 `i8` 端点、`0x80`→`-127` 对称化、八值/六值双模式 + `-127`/`127` 终端，`decode_bc4_signed`/`decode_bc5_signed` 出 `i8`，供物件空间法线/带符号位移，8 条外部锚点，commit `f4ce6f099`）+ **BC4/BC5 `SNORM` 带符号块编码器**（RGTC 带符号压缩落盘：带符号 min/max 端点钳 `-127..=127`、八值/六值择优、逐纹素最近索引，`encode_bc4_signed`/`encode_bc5_signed` 逐位往返带符号解码器，6 条往返锚点，commit `d8eeb8836`）+ **`BcTexelSource` 扩展至 BC7/`ETC2_RGB8`**（含不可解模式的格式在构造期逐块校验，BC7 分区 / ETC2 T/H/planar 块返回 `UndecodableBlock` 拒绝错解，采样期绝不静默回退黑块；BC7 mode6 / ETC2 基础模式经各自编码器往返贯通源→采样链路，commit `260b76836`）+ **Catmull-Rom cubic trilinear + 各向异性采样**（把已验证 `bicubic_catmull_rom` 接入 LOD 感知采样器：`trilinear_bicubic` 两 bicubic mip 按分数 LOD 混合、`filter_resolved_bicubic` 各向异性 cubic taps 均权积分，高质量 C1 采样模式消除 bilinear 放大菱形缝、线性斜坡跨 mip 精确重建，6 条外部锚点，commit `fac0f7f2e`）+ **Gaussian 严格非负平滑 mip 归约**（可分离高斯 2× 缩减建于共享 gamma 正确 `resample_core`，恒不过冲/振铃，替代 Lanczos/Kaiser 的 sinc 旁瓣用于粗糙度/高度/覆盖率图软预滤；`GaussianFilter{radius,sigma}` + `gaussian_downsample` + `generate_mip_chain_gaussian`，纯 f32 无 AI/ML，11 条外部锚点，commit `df8407773`）+ **Triangle/Bartlett 三角 mip 归约**（无参数分段线性帐篷核=Bartlett 窗/一阶 B 样条，建于共享 gamma 正确 `resample_core`，严格非负恒不过冲、比盒式阻带更缓且无形状参数，与 Gaussian 并列用于粗糙度/高度/覆盖率软预滤；`TentFilter{radius}` + `tent_downsample` + `generate_mip_chain_tent`，纯 f32 无 AI/ML，11 条外部锚点，commit `d88e2c45b`）+ **B 样条平滑三次 trilinear + 各向异性采样**（把已验证非负 `bspline_cubic` 接入 LOD 感知采样器：`trilinear_bspline` 两 B 样条 mip 按分数 LOD 混合、`filter_resolved_bspline` 各向异性 B 样条 taps 均权积分，Catmull-Rom cubic 采样模式的零过冲 C2 平滑孪生，用于高度场/地形/SDF/覆盖率数据纹理放大消除 cubic 振铃假脊，线性斜坡跨 mip 精确重建，6 条外部锚点，commit `ccf2662f2`）+ **Mitchell-Netravali (B,C) 参数化三次 trilinear + 各向异性采样**（把已验证 `cubic_mitchell` (B,C) 参数化核接入 LOD 感知采样器：`trilinear_mitchell` 两 Mitchell mip 按分数 LOD 混合、`filter_resolved_mitchell` 各向异性 Mitchell taps 均权积分，(B,C) 从 Catmull-Rom (0,1/2) 到 B 样条 (1,0) 连续可调、均衡 `B=C=1/3` 默认，把 bicubic/bspline 两采样模式统一为单一参数族，线性斜坡跨 mip 精确重建，6 条外部锚点，commit `90b8cf64d`）+ **slope-space 法线强度缩放 + 斜率往返**（AAA 材质图的法线强度/凹凸强度滑杆的物理正确实现：`scale_strength` 在斜率空间按 `normalize(s*nx, s*ny, nz)` 缩放切空间法线细节强度，精确满足 `s=1` 恒等 / `s=0` 退平 / 平法线恒平 / 强度乘法可组合 `scale(scale(n,a),b)==scale(n,a*b)` 四条身份；`normal_to_slope`/`slope_to_normal` 暴露高度场梯度(斜率)往返，供细节分层/位移/派生工作流复用；纯解析 f32 无 AI/ML，9 条外部锚点，commit `868d878ef`）+ **任意目标分辨率可分离图像重采样器**（mip 生成只能逐级减半，AAA 导入/烘焙还需把解码后 `RGBA8` 缩放到*任意*分辨率——贴合 2 次幂图集槽、构建 UI/缩略图层级、统一尺寸不匹配的纹理输入；`texture_resize::resize` 为经典可分离、gamma 正确、归一化权重重采样器(Turkowski 1990)：色彩一次性抬升到场景线性 `f32`、沿 X 再 Y 以可选 `ResizeFilter`(Box/Triangle/CatmullRom/Lanczos3) 滤波、再一次性重编码，缩小时核足迹按逆缩放展宽做抗锯齿低通、放大时保持原生宽度做重建，权重恒归一化、仅取样索引钳到边缘；逐构造对拍：同尺寸 Box==恒等、精确 2× Box 缩小==盒式 mip 归约器 2×2 均值(Linear+Srgb)、整数 Box 放大==最近邻复制、任意尺寸/滤波常数保持、三角斜坡放大单调且端点钳制、重采样与水平镜像可交换；14 条外部锚点，commit `438121db8`）+ **重采样器扩展 Mitchell 与 B 样条重建滤波**（为 `ResizeFilter` 补上两条非插值重建核：Mitchell-Netravali 均衡三次 (B=C=1/3，公认最佳通用重采样核，轻微振铃+轻微模糊的最优折中) 与非负三次 B 样条 (B=1,C=0，恒不过冲/无振铃的平滑重建)；两者复用 `texture_filter` 已验证的通用 (B,C) 核(`mn_kernel` 以 `pub(crate)` 暴露)，无重复闭式转写；逐构造对拍：重采样器自带 Keys 闭式 Catmull-Rom 与通用 (0,1/2) Mitchell 核逐点吻合(两条独立推导)、B 样条严格非负、重建核中心叶 <1(Mitchell 16/18、B 样条 2/3，故需权重归一化)、Box/Triangle/Catmull-Rom/Mitchell/B 样条在整数格点上构成单位划分；4 条外部锚点，commit `1ba1ce0dc`）+ **高度场→切空间法线生成**（AAA 材质创作把灰度高度/凹凸场（雕刻腔隙遮罩、平铺细节、贴花、地形）转为切空间法线贴图的固定功能：`height_to_normal` 以 `CentralDifference`(2-tap `(h[+1]-h[-1])/2`) 或 `Sobel`(3×3 行/列平均，抗单纹素噪声、对平面斜坡精确) 估计高度梯度，再喂入本 crate 已验证的 `slope_to_normal` 斜率转换——故生成器与法线强度滑杆共用同一斜率空间、天然一致；逐纹素世界间距 `texel_world_size`(支持非正方纹素/凹凸尺度) + `strength` 斜率缩放 + `WrapMode` 边界取邻(Repeat/MirroredRepeat 保平铺接缝、其余钳边)；逐构造对拍：平场=朝上、平面斜坡匹配解析斜率、斜坡上 central==Sobel、`strength` 恰等于把 s=1 法线过 `scale_strength` 斜率空间缩放、水平镜像翻转 normal.x、恒单位长、仅沿列变化的场在边界 normal.x 恒 0；纯解析 f32 无 AI/ML，8 测，commit `400d731c0`）+ **原生分辨率可分离高斯模糊**（区别于只能逐级减半的高斯 mip 归约，本单元在*原分辨率*就地模糊——bloom/glare 金字塔、SSAO/软阴影去噪、屏幕空间次表面扩散、覆盖率/遮罩软化、不改尺寸的粗糙度/高度预滤的固定功能原语；`texture_blur::gaussian_blur` 为经典可分离、gamma 正确、归一化高斯(Heckbert 1986)：色彩一次性抬到场景线性、沿 X 再 Y 以采样核 `exp(-k²/2σ²)`(|k|≤`ceil(3σ)`) 卷积、再一次性重编码；`gaussian_weights_1d` 归一化对称非负核；`blur_plane` 单通道 f32 核心 + `WrapMode` 边界(Repeat/MirroredRepeat 保平铺、其余钳边)；逐构造对拍：权重归一/非负/对称、常数图恒等保持、σ≤0 恒等、模糊恒不越界(凸组合零过冲)、**周期余弦的频率响应精确等于核的实 DFT 幅值且零相移**(采样核↔频响的直接链接)、水平镜像在钳边下可交换、秩-1 可分信号分解为各因子 1D 模糊之积；纯解析 f32 无 AI/ML，10 测，commit `2b3bea154`）+ **原生分辨率可分离盒式模糊**（最廉价低通：每轴 `(2r+1)` 窗无权平均，大半径实时模糊主力——SAT/方差阴影预滤、廉价景深/bloom、以及「三次盒式≈高斯」(中心极限)技巧；`texture_blur::box_blur` 以**滑动running-sum**(减左加右)实现每行 `O(width)` 与半径无关、共享 gamma 正确工作缓冲、`WrapMode` 边界(Repeat/MirroredRepeat 保平铺、其余钳边)；逐构造对拍：running-sum 输出与**独立逐像素从头窗求和 oracle**逐点吻合(交叉校验增量减加算术与 wrap 索引)、常数恒等保持、r=0 恒等、恒不越界(凸组合零过冲)、**周期余弦频率响应精确等于归一化闭式 Dirichlet 核且零相移**、水平镜像钳边下可交换；纯解析 f32 无 AI/ML，7 测，commit `820fb5e52`）+ **原生分辨率边缘保持双边模糊**（在空间高斯上再乘一个强度差高斯(Tomasi & Manduchi 1998)——跨强边的 tap 得到极小 range 权重，核坍缩到中心所在的一侧，故平坦区平滑而边缘保锐，是边缘感知去噪(SSAO/阴影/RT-GI)、保细节磨皮/色调、细节分层(local-tone-mapping 基层)的固定功能原语；range 权重数据相关故**不可分离**，每输出纹素在 `|dx|,|dy|≤ceil(3σs)` 窗做完整 2D gather、逐纹素重归一化(非负权和为 1 的凸组合→常数恒保持、恒不越界)；`bilateral_blur_plane` 单通道核心 + `bilateral_blur` 场景线性逐通道、`WrapMode` 边界；逐构造对拍：**σr→∞ 时 range 权→1，滤波器收敛到已验证的可分离高斯 `blur_plane`**(积核的 2D 归一化等于可分离逐轴归一化 `Σ_ij gx·gy=(Σgx)(Σgy)`，故二者浮点容差内吻合——把新代码直接锚到已测高斯)、硬台阶边缘上小 σr 使输出严格比同 σs 高斯更贴近输入(边缘保持)、常数图恒保持、σs≤0 或 σr≤0 恒等、图像 API 常数保持+σ≤0 恒等；纯解析 f32 无 AI/ML，7 测，commit `67787b9bb`）+ **原生分辨率反锐化掩模(unsharp mask)锐化**（最广泛部署的锐化算子：减去低通拷贝提取高频细节、再把 `amount` 倍细节加回原图——`out = src + amount*(src - blur) = (1+amount)*src - amount*blur`，低通复用已验证的可分离 gamma 正确高斯 `blur_plane`，`sigma` 定被增强的细节尺度、`amount≥0` 定强度；是纹理导入/mip 锐化、后期锐化、边缘保持去噪后细节再注入的固定功能原语；区别于模糊，反锐化**非凸组合**——负 `-amount*blur` 瓣正是边缘过冲(halo)的来源，输出可越界(回写 RGBA8 时钳制)；`unsharp_mask_plane` 单通道核心 + `unsharp_mask` 场景线性逐通道；逐构造对拍：**与从已验证高斯重算的闭式 `(1+amount)*src - amount*blur` 逐点吻合**(把本单元直接锚到已测代码)、`amount=0` 或 σ≤0 恒等、高斯精确重现常数与仿射(线性斜坡)故 `src-blur=0` 二者对任意 amount 恒保持(锐化只作用于有曲率/细节处)、硬台阶边缘正 amount 过冲越界(halo)、图像 API 常数保持+amount=0 恒等；纯解析 f32 无 AI/ML，7 测，commit `616488e26`）+ **原生分辨率联合/交叉双边(joint/cross bilateral)滤波**（把双边的边缘停止 range 权与被平滑的数据解耦：核由独立**引导(guide)**信号构建、再施加到数据平面——是引导式边缘感知上采样(Petschnigg 2004 闪光/非闪光对、Kopf 2007 联合双边上采样)、用干净 albedo/normal/depth G-buffer 引导噪声通道的联合去噪、以及结构在另一缓冲的细节迁移的固定功能原语；每输出纹素权重 `w=exp(-(dx²+dy²)/2σs²)·exp(-(guide_c-guide_t)²/2σr²)`、对数据平面加权重归一化(凸组合→数据范围不越界、常数数据对任意引导恒保持)；`joint_bilateral_blur_plane` 单通道核心 + `joint_bilateral_blur` 场景线性逐通道(逐通道引导、尺寸不匹配拒绝)；逐构造对拍：**引导==数据时精确等于已验证的 `bilateral_blur_plane`**(直接锚到已测代码)、**σr→∞ 时引导失效、收敛到可分离高斯 `blur_plane`**、常数数据对任意引导恒保持、输出不越数据范围、σ≤0 或引导长度不匹配恒等、台阶引导驱动核偏离纯高斯(证明引导确实在转向)、图像 API 引导==源时等于纯双边+尺寸不匹配返 None；纯解析 f32 无 AI/ML，8 测，commit `7f2989422`）+ **原生各向异性椭圆加权平均(EWA)过滤**（参考级各向异性采样器(Heckbert 1986/1989；Greene & Heckbert 1986)——硬件各向异性(及本 crate `filter_resolved`)沿长轴取若干均权探针近似像素投影纹理足迹，EWA 则把足迹当成它真实的**椭圆**、在高斯重建权下 gather 椭圆内每一枚纹素，故无 tap 数阶梯、无短轴过糊，是教科书式参考结果；足迹由屏幕空间 UV 偏导 `grad_x=d(u,v)/dx`、`grad_y=d(u,v)/dy`(texels/pixel)张成，经 Heckbert 隐式二次型 `r2=A·s²+B·s·t+C·t²` 定义 `r2<1` 椭圆、A/C 各加单位**重建**项使足迹不塌缩到 ~1 纹素以下(优雅放大)，覆盖纹素按钳位高斯 `exp(-α·r2)-exp(-α)`(α=2，边界 r2=1 处连续归零)加权再重归一化(凸组合→常数保持、输出不越采样范围)，偏心率超 `MAX_ANISOTROPY=16` 时拉长短轴以界定工作量；`ewa_sample_plane` 单通道核心 + `ewa_sample_rgba8` 场景线性逐通道、`WrapMode` 边界、退化足迹回退最近纹素；逐构造对拍：**各向同性足迹 `grad_x=(h,0),grad_y=(0,h)` 坍缩为圆 `r2=(s²+t²)/(h²+1)`、精确等于独立径向高斯 gather**(证明 B=0、A=C 二次型系数正确)、**轴对齐足迹 `(a,0),(0,b)` 给 `r2=s²/(a²+1)+t²/(b²+1)`、匹配独立逐轴求和**(证明包围盒/系数正确)、常数平面恒保持、输出恒不越采样范围、零梯度整数坐标返回该精确纹素(重建项仅留中心 tap)、空/尺寸不匹配返 0、图像 API 常数保持+零梯度整数坐标取最近纹素；纯解析 f32 无 AI/ML、GPU 计算 EWA 浮点容差内可对拍，8 测，commit `fc6fe75ce`）+ **原生可分离灰度形态学(膨胀/腐蚀/开/闭)**（覆盖掩膜与有符号距离场(SDF)的固定功能原语：生长/收缩 alpha-test 或贴花覆盖、闭合针孔/开去斑点、一遍描边(`dilate-src`)、以及 jump-flood/chamfer SDF 扫描的 max/min 半程；平坦结构元(SE)下每 tap 权为 0 故为纯秩滤波——无 gamma、无归一化；方形平坦 SE **可分离**(矩形上的最小值=窗内逐行最小值之最小值)，故与已验证的 `box_blur_plane` 同构地先 X 后 Y 两遍、`WrapMode` 边界；`dilate_plane`(局部 max，**外延** `out≥src`) / `erode_plane`(局部 min，**反外延** `out≤src`) / `open_plane`(先腐后膨，去亮斑、**幂等**) / `close_plane`(先膨后腐，填暗孔)；逐构造对拍：**可分离两遍结果与独立全 2D 方窗扫描逐点吻合**(证明分离本身而非假设)、膨胀与腐蚀**取负对偶** `dilate(x)=-erode(-x)`、**单调** `a≤b⇒dilate(a)≤dilate(b)∧erode(a)≤erode(b)`、`dilate≥src≥erode`、**开运算幂等** `open(open(x))=open(x)` 且 `open≤src≤close`、radius=0 恒等、常数平面恒保持、空/尺寸不匹配回拷；纯解析 f32 无 AI/ML、GPU 计算形态学逐位对拍，8 测，commit `44f7ba53c`）+ **原生可分离图层混合模式(W3C/PDF)+RGBA8 合成器**（纹理创作与运行期贴花/细节分层的固定功能合成原语：细节层 `cs` 与背景 `cb` 经逐通道 `B(cb,cs)→[0,1]` 结合；覆盖 Multiply/Screen/Overlay/Darken/Lighten/ColorDodge/ColorBurn/HardLight/SoftLight/Difference/Exclusion/LinearDodge(加)/LinearBurn 及 Normal 共 14 种**可分离**模式(逐通道独立、纯标量算术，区别于不可分离的色相/饱和度/颜色/明度族)，全部 `[0,1]²→[0,1]` 故合成不越界；`blend_channel` 标量核 + `blend_rgba8` 场景线性逐通道合成器(标量 `opacity` 把混合色线性拉回背景 `out=cb+t·(B-cb)`、背景 alpha 保留、尺寸不匹配返 None)；逐构造对拍(锁定每个解析恒等式)：`multiply(a,1)=a∧multiply(a,0)=0`、`screen(a,0)=a∧screen=1-(1-a)(1-b)`、**`overlay(a,b)=hard_light(b,a)`**(层交换对偶)、`darken=min∧lighten=max`、**`soft_light(a,0.5)=a`**(中性灰恒等)、`color_dodge(a,0)=a∧color_burn(a,1)=a`、`linear_dodge=min(a+b,1)∧linear_burn=max(a+b-1,0)`、`difference/exclusion` 对称且 `difference(a,a)=0`、**全 14 模式在 [0,1]² 网格上恒不越单位区间**、合成器 opacity=0 返背景+尺寸不匹配返 None；soft-light 的 `sqrt` 分支走 `bevy_math::ops`、纯解析 f32 无 AI/ML、GPU 图层混合逐位对拍，11 测，commit `96cca3c76`）+ **原生边缘感知引导上采样(JBU)**（半/四分辨率屏幕空间解(AO/漫反射 GI/软阴影/SSS)在全分辨率引导(深度/法线/亮度)下上采样——纯双线性放大会把低分辨率解跨深度/法线不连续处涂抹、在物件轮廓产生光晕，联合双边上采样(Kopf/Cohen/Lischinski/Uyttendaele 2007)用全分辨率引导转向插值：引导边另一侧的低分辨率 tap 得近零值域权，故上采样信号保全分辨率帧的锐边又在区域内平滑；全分辨率像素 `p` 映射到低分辨率坐标 `p_low`(像素中心对齐 `sx=lw/fw`)、在 `radius=3σs+1` 窗内 gather 低分辨率 `q`，权 `exp(-‖q-p_low‖²/2σs²)·exp(-(G_p-G_up(q))²/2σr²)` 再归一化(凸组合→常数低分辨率恒保持、输出不越低分辨率范围)，`G_up(q)` 为低分辨率 `q` 最近全分辨率位置的引导值；`joint_bilateral_upsample_plane` 单通道核、`WrapMode` 边界、空/尺寸不匹配返空、非正或非有限 σ 回退最近邻；逐构造对拍：微 σs+同分辨率 gather 坍缩到共位纹素→**恒等**、**σr→∞(1e9) 匹配独立 `spatial_only` 空间核**(把同一映射+空间权+归一化复刻到独立代码、值域项强制为 1，锁定映射/权/归一化而非自证)、常数低分辨率对任意引导恒保持、输出恒不越低分辨率范围、硬台阶引导+小 σr 保左侧近 0 右侧近 1(边缘感知)、空/不匹配返空、非正 σ 回退最近邻；纯解析 f32 无 AI/ML、GPU 计算上采样浮点容差内可对拍，7 测，commit `4edb4ec2e`）+ **原生 RGBA8 彩色引导上采样(JBU 彩色后续)**（JBU 平面核的彩色后续：半/四分辨率彩色解(廉价 GI/辐照度彩色缓冲、降采样 bloom/半透层、抽稀贴花图集)在全分辨率单通道引导(深度/亮度/边缘信号)下上采样；四个 RGBA8 通道共用**同一组** spatial×range 权(引导为标量故权与通道无关)——既是物理正确(单一引导同等转向所有通道)又比独立上采样四平面省约 4×(超越函数空间/值域权每像素只算一次复用)，彩色通道在 `ColorSpace` 下升到场景线性(凸组合在线性光完成，匹配 GPU 上采样)、alpha 恒线性，再按色彩空间重编码(Srgb RGB 过 `linear_to_srgb`、alpha 与 Linear 全通道走四舍五入)；`joint_bilateral_upsample_rgba8` 出 `Rgba8Image`、尺寸不匹配引导或零全分辨率边返 None、非正/非有限 σ 回退逐通道最近邻；逐构造对拍：**图像上采样恒等于把四通道升到场景线性平面、各自过独立验证的 `joint_bilateral_upsample_plane` 同参、再重编码**(把共享权 gather 锁死到参考单通道核、彻底防假实现，覆盖 Linear/Srgb × ClampToEdge/Repeat)、常数彩色对任意引导恒保持(±1 LSB 往返)、输出 alpha 不越低分辨率 alpha 范围、空/尺寸不匹配引导返 None、非正 σ 逐纹素回退最近邻复制；纯解析 f32 无 AI/ML、GPU 彩色上采样浮点容差内可对拍，5 测，commit `55852865d`）+ 超越函数 `bevy_math::ops` libm 确定化、clippy 清零）；仍缺 vis-buffer / RT GPU 求导接线 + GPU 常驻页 `TexelSource`（DoD-2） | vis-buffer / RT 下纹理不走样不过采样 |
| 大气天空 + 空中透视 + 体积云（§17.1.4，P1.5） | ⬜ | LUT 预计算 + froxel raymarch 缺（DoD-1/3） | 行星级大气黄昏红移、云透光梯度物理合理 |
| 毛发 / 体积深阴影统一（§17.1.6，P1.5） | 🟡 | 体积深阴影 + 统一半透射缺（DoD-1/2） | 毛发自阴影不发黑、体积内部透光梯度收敛 |
| 矩不变 OIT + 混合折射（§16.1.3，P1.5） | 🟡 | 矩重建 + RT 折射精档缺（DoD-1/2） | 多层透明正确合成、折射物理可信 |
| FFT 频谱海洋 + 泡沫（§16.1.4，P1.5） | 🟡 | FFT + 雅可比白沫 + LOD 缺（DoD-1/3） | 大洋级联到近岸无缝、浪尖破碎泡沫物理 |
| GPU 驱动粒子渲染整合（§18.1.3，P1.5→P2） | 🟡 | GPU 排序 / mesh 粒子 / motion vector 缺（DoD-2/4） | 海量粒子正确排序、软边、参与 TSR 不拖影 |

#### 18.2.3 收口次序（使能器优先，锁死依赖链）
1. **第一梯队（使能器，必须先毕业）**：PSO 预编译（帧稳定性地基）→ GPU 蒙皮缓存 + SVT（动态开阔世界两大前置）→ 光线微分纹理 LOD（vis-buffer / RT / SVT 正确 mip 的共同前提）。
2. **第二梯队（近景角色可信度）**：SSS → GTAO → 材质多散射能量守恒（三者共同决定「近景人物与接触」顶级观感）。
3. **第三梯队（环境天花板）**：大气天空 + 体积云 → FFT 海洋 → 毛发 / 体积深阴影统一（依赖 froxel / VSM / 使能器就位）。
4. **第四梯队（透明 / 特效）**：矩不变 OIT + 折射 → GPU 粒子渲染整合（依赖 OIT 合成缓冲 + 排序）。

#### 18.2.4 帧级「顶级次世代 AAA」判据（整帧验收，非单特性）
- **几何**：虚拟几何微三角密度下无 LOD pop、无裂缝；vis-buffer / RT 纹理无走样（§18.1.2）。
- **光照 / GI**：动态光照无烘焙、无萤火虫、1–2 spp 降噪无鬼影；反射 / 软阴影 / SSS / GI 各过白炉或 parity 门槛。
- **稳定性**：首见材质 / 特效零编译卡顿（§18.1.1）；TSR 4K 无拖尾；DRS 重载降质平滑无跳变。
- **动态世界**：动画几何（蒙皮 / WPO）正确进入 RT / Nanite / MegaLights；海量唯一纹理固定显存无 pop。
- **红线**：全链路纯经典数值可对拍，无任何 AI/ML/神经/LLM 推理路径。

### 18.3 v11 增补路线图优先级（并入既有阶梯，不改 §9 P0 基底次序，不重复 §6/§11–§17 条目）

| 增补特性 | 落点 | 现状（实测） | 优先级 | 依赖 |
|---|---|---|---|---|
| 着色器 PSO 预编译 + 管线缓存（stutter-free） | `render_architecture/shader_package` + `frame_graph` | ⬜ 置换骨架 5 文件在，预热 / 磁盘缓存 0 | **P1** | 置换枚举、缺失 PSO 反馈、设备指纹、异步降级 |
| 可见性缓冲 / RT 光线微分纹理 LOD | `render_scene` + `ray_scene`(`ray_cone`) + `render_material` + `render_architecture` | 🟡 `ray_cone` 1 文件在，vis-buffer/RT 求导 0 | **P1.5** | vis-buffer 延迟着色(§12)、SVT 反馈(§16.1.1)、各向异性采样 |
| GPU 驱动粒子渲染整合（排序 + mesh/ribbon + 软粒子） | `particle/`(`soft_particle`) + `render_shading/oit` + `motion` | 🟡 `soft_particle` 3 文件在，GPU 排序 / mesh 粒子 0 | **P2** | GPU 排序、矩不变 OIT(§16.1.3)、motion vector、粒子模拟(另册) |
| 顶级 AAA 毕业门禁收口（§18.2） | 跨 crate（流程 / 验收，不新增特性） | — 流程规格 | **P0.5（横切）** | 各 P1/P1.5 特性、CPU golden / GPU parity 基建 |

**v11 增补总原则**：① **不改 §9 P0 基底次序**，不重复 §6/§11–§17 已登记条目——§18.1 三条均经关键字实测确认为**新**赛道（🟡 骨架在 / ⬜ 真实空白）；② **PSO 预编译列 P1**——「shader comp stutter」是顶级次世代 AAA 公认帧稳定性顽疾，根除它是「顶级」的必要条件而非锦上添花，且不依赖任何未落地使能器，可独立先行；③ **光线微分纹理 LOD 列 P1.5**——它是 vis-buffer / RT / SVT 三者「正确 mip」的共同前提，效果天花板高、增量小；④ **GPU 粒子渲染整合列 P2**——依赖 OIT 合成缓冲与排序，排在使能器之后，且模拟侧归属粒子专册；⑤ **§18.2 毕业门禁列 P0.5 横切**——它不产特性而定义「怎么算到达顶级」，应最先建立以锚定并行舰队的零散落地；⑥ 全部走「CPU golden → GPU kernel → 真机 parity」三步，关键路径必须与**路径追踪参考 / 解析解 / 离线数值求解**可对拍，未毕业不作生产默认；**任何增补均不引入 AI/ML/神经/LLM 推理路径**（厂商时序上采样 / 帧生成 SDK 仅渲染侧可选外部后端）。

---

## 附：与管线文档的关系
- 顶层架构 / 决策 / 后端策略 / 三桶可测性 / 代码重构清单：见 `prism_material_pipeline_design_zh.md`。
- 全局光照 / 反射 / 采样降噪 深水区：见 `prism_gi_lumen_design_zh.md`（v4）。
- 毛发子系统：见 `prism_hair_engine_design_zh.md`。粒子：见 `prism_particle_engine_design_zh.md`。物理：见 `prism_physics_design_zh.md`。体积：见 `prism_volumetric_engine_design_zh.md`。水体：见 `prism_water_engine_design_zh.md`。布料：见 `prism_cloth_engine_design_zh.md`。音频：见 `prism_audio_engine_design_zh.md`。
- 本文只负责“三前端各自与共享的高级特性目录 + 前沿特性全景 + 预算 + 验收”。
