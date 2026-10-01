# Prism 渲染引擎 — 次世代 AAA 级全局光照（GI）子系统完整设计（v4 / wgpu + WESL）

> 状态：架构提案（Draft，允许破坏性重构）。v3 追加 Spherical Gaussian 辐射、Mesh Distance Fields / DFAO、bent normal、多弹跳镜面、贴花 / 多层 / OIT GI、棋盘重建与 bindless / persistent-thread / 距离场预剔除等工程优化。
> **v4（本版）在 v3 之上继续顶格增补一批最新 AAA 高级功能（全部纯经典数值、无任何 AI/ML/神经网络/LLM 路径）**：World-Space ReSTIR（空间哈希蓄水池跨帧跨视角重用）、Volumetric ReSTIR / froxel 参与介质 GI、焦散（自适应光子抛撒 + 流形 NEE）、Adaptive Probe Volumes（自适应密度免烘探针体 + 天空遮蔽）、ReSTIR PT（完整路径重采样 + shift map）、随机 HiZ-SSR + 镜面 reservoir 重用、微遮蔽 / 腔体 GI、薄膜干涉 / 各向异性 GI 反射、水面 / 湿表面 GI、大气 / 体积云多散射耦合；工程侧增补 Shader Execution Reordering（SER）、Opacity / Displaced Micromaps（OMM/DMM）、GPU Work Graphs / 动态射线生成、mesh-shader 卡片捕获、HiZ 加速剔除等。新增 **Ultra+（影视级实时）** 档位，对标 Lumen HWRT / Cyberpunk RT Overdrive / Alan Wake 2 / Portal RTX / Half-Life 2 RTX 天花板。
> 定位：AAA / 次世代 **动态全局光照 + 统一反射 + 采样/降噪** 引擎（软件光追优先、硬件 Ray Query 可选），实时无烘焙，与 PBR/NPR/自定义/混合四前端正交协同，作为四前端共享的高级光照基底。
> Shader：WESL（naga → WGSL/SPIR-V/Metal/DXIL）；运行时：wgpu（Metal/VK/DX12/WebGPU）。
> 关联文档：\`prism_rendering_architecture_zh.md\`（总，Phase 5 多后端 GI/反射）、\`prism_material_pipeline_design_zh.md\`（材质语义/Surface Cache 烘焙）、\`prism_volumetric_engine_design_zh.md\`（froxel 参与介质 GI 衔接）、\`prism_aaa_advanced_features_zh.md\`（共享高级基底）、\`prism_particle_engine_design_zh.md\`（自发光粒子作为 GI 光源）、\`prism_hair_engine_design_zh.md\`（毛发 GI）。
> 沙盒说明：本机 agent 沙盒屏蔽 Metal，文档内帧预算/带宽/耗时均为**设计目标（design target），非实测**；可验证部分限于 CPU golden 纯函数与 WESL 编译。真机 parity 在沙盒外执行。
> 数值路线：**纯经典数值**（蒙特卡洛/准蒙特卡洛积分 / SH / reservoir 重采样 / SDF 步进 / 时空双边降噪 / 时空蓝噪声），**不含任何神经网络 / AI / ML / LLM 路径**——降噪、上采样、重要性采样全部为确定性经典算法。

---

## 0. 定位：这是什么级别的 GI

完整的**实时动态全局光照 + 统一反射引擎**：无预烘焙光照贴图、无光照探针预计算，几何/材质/光源全动态时都能给出多弹跳间接光、镜面/粗糙反射、折射与体积间接光。目标画质对标当前 AAA 天花板（Lumen / Cyberpunk 2077 RT Overdrive / Alan Wake 2 / Metro Exodus EE），仅在**算法层**借鉴，不复制其代码：

| 产品 / 技术 | 借鉴点（算法层） |
|---|---|
| **UE5 Lumen** | 整体拓扑：Screen Probe + World Radiance Cache + Surface Cache（cards）+ Final Gather + 软/硬双后端；远场 distant scene；半透明/体积 GI；hit-lighting |
| **NVIDIA RTXGI / DDGI** | 辐照体积探针 + Chebyshev 深度可见性（均值/方差）漏光抑制；探针重定位/分类/滞后收敛 |
| **NVIDIA SHARC（空间哈希辐射缓存）** | 世界空间哈希缓存替代纯 voxel，视相关分辨率、免网格管理、多弹跳兜底 |
| **ReSTIR GI / GRIS / ReSTIR PT（Cyberpunk RT Overdrive）** | 时空 reservoir 重采样 + 广义无偏 MIS（GRIS）；1–2 spp 出路径追踪级质量 |
| **NVIDIA RTXDI（ReSTIR DI）** | 海量光源直接光 reservoir，与 GI 共享 BVH/样本 |
| **ReGIR（世界空间光源网格）** | 世界空间光源重要性网格，为 ReSTIR DI 提供初始候选，百万级光源可扩展 |
| **光源树 / Light BVH（Cycles / Moreau-Clarberg）** | 层级重要性采样海量光源与自发光三角，方差随光源数近似恒定 |
| **Radiance Cascades（PoE2 / Sannikov）** | 分级辐射区间：近处高角分辨率、远处高空间分辨率，远场 diffuse 近常数成本 |
| **AMD FidelityFX Brixelizer GI** | 全局稀疏 SDF/brick 场景表示，快速锥步进 |
| **CryEngine SVOGI / Godot SDFGI** | 稀疏 voxel 八叉树 / 级联 SDF 作为低配兜底 GI |
| **Kajiya（Embark，参考实现）** | 世界+屏幕探针混合、辐照度缓存工程折衷经验 |
| **NRD（ReBLUR/ReLAX）/ SVGF / A-SVGF** | 时空联合双边降噪、方差引导、历史 clamp、双边分离 diffuse/specular |
| **时空蓝噪声（STBN，Heitz/Wolfe）+ Owen-scrambled Sobol** | 低差异采样：噪声在时空上"蓝"化，感知误差与收敛远优于白噪声 |
| **XeGTAO / GTAO / HBAO+** | 地平线基环境光遮蔽与镜面遮蔽，作为高频接触细节补充 |
| **Frostbite / Metro Exodus EE** | 混合 diffuse GI + 单弹跳镜面、性能分档与开放世界经验 |
| **Frostbite / SEED PICA PICA** | 混合软/硬光追实时 GI 工程折衷；**Spherical Gaussian** 方向辐射表示提升中粗糙镜面响应 |
| **UE5 Mesh Distance Fields / DFAO** | 每物体有向距离场 + 全局 clipmap 合成；距离场软 AO / 软阴影 / 短程间接遮蔽，无 BVH 也能出低频遮蔽 |
| **UE5 MegaLights** | 统一海量光源随机直接光（reservoir 式），阴影与 GI 共享可见性查询，灯光数量与成本解耦 |
| **UE5 Nanite + Virtual Shadow Maps** | 虚拟几何 cluster 派生 Surface Cache 卡片；VSM 高频阴影与 GI 共享几何/剔除，省重复遍历 |
| **World-Space ReSTIR（NVIDIA，空间哈希蓄水池）** | 把 reservoir 存进世界空间哈希网格，跨帧 / 跨视角 / 多弹跳复用样本，解遮挡与多弹跳收敛显著加速，复用 SHARC 哈希基建 |
| **Volumetric ReSTIR（Lin et al.）** | froxel 体素蓄水池重用参与介质散射样本：体积雾 / 体积光 / 云获得多弹跳间接光，方差大降 |
| **焦散：自适应光子抛撒 + 流形 NEE（Hanika/Jakob）** | 水面 / 玻璃锐利焦散经典数值路径：光子/流形引导下一事件估计 + 屏幕空间焦散累积，无需 ML |
| **Adaptive Probe Volumes（Unity APV / UE 探针体）** | 自适应密度辐照 + 可见性探针体，brick 流式 + 天空遮蔽，作免烘动态兜底与远 / 静态区低频 GI |
| **ReSTIR PT（完整路径重采样，Kettunen/Lin）** | GRIS shift map（重连 / 随机重放）做完整路径时空重用，Ultra 档 1–2 spp 逼近离线路径追踪 |
| **Shader Execution Reordering（SER）+ OMM/DMM（NVIDIA Micro-Maps）** | 硬件 RT 重排提升射线相干；不透明 / 位移微网格加速 alpha-test 植被与置换几何的 GI 遮挡 |
| **D3D12 / Metal GPU Work Graphs** | GPU 自驱动态射线 / 卡片 / 探针调度，去 CPU 往返，长短射线负载自均衡 |
| **Thin-film / Iridescence BRDF（Belcour-Barla）** | 薄膜干涉与各向异性 GGX 并入 SG / traced 反射，GI 反射尊重切线帧与膜厚色散 |
| **Portal RTX / Half-Life 2 RTX（RTX Remix 级路径追踪）** | 全路径追踪 + ReSTIR PT + 焦散 + 体积多散射的实时参照上限（Ultra+ 对标目标） |
| **Decima（Horizon）/ Call of Duty** | clipmap irradiance volume 开放世界探针流式与压缩经验，bake-free 滚动更新 |
| **AMD FidelityFX GI-1.0** | 世界探针 + 屏幕探针轻量实时 diffuse GI，低配兜底参考 |
| **Spherical Harmonics / SG 文献** | diffuse 走 SH L1 省内存、glossy 走少量 SG 瓣保方向性的混合辐射基 |

**不做**：预计算光照贴图烘焙管线（Phase 6 离线渲染另行）；**不引入任何 AI/ML/神经网络/LLM**（含神经降噪、Ray Reconstruction、神经辐射缓存等），全部走经典数值路径。

---

## 1. 设计原则与约束（Prism 范式）

1. **软件光追优先**：默认后端是已验证的 \`ray_scene\` 软件 BVH（真机 Metal parity）。硬件 Ray Query 仅在能力探测通过后作为可选加速路径，遵循现有 graduation gate 模式，默认关闭。
2. **契约驱动 + 真机 parity 孪生**：每个 kernel 三步落地 —— ① CPU golden 参考（纯函数、确定性）→ ② WESL kernel → ③ 真机 Metal parity 测试（逐分量对拍，容差显式）。未毕业不得作为生产默认。
3. **不改 Bevy 源码**：全部在 \`pkg/prism_render_*\` crate 内实现，通过 RenderApp 接入。
4. **可降级矩阵**：任何高级功能都必须有明确的低档回退，且低档也要 parity 验证。
5. **无偏优先，偏差显式**：ReSTIR/时域重用引入的 bias 必须有 ground-truth 参考对拍，偏差项显式建模与验证（GRIS 广义平衡启发式）。
6. **性能硬门禁**：每档有帧预算上限，接入现有帧预算控制器，超预算不得毕业。
7. **采样质量优先**：所有随机路径统一走 STBN + Owen-scrambled Sobol 低差异序列，禁止裸白噪声；采样与降噪协同设计。
8. **能量守恒与稳定性**：firefly 钳制、能量补偿（多散射 BRDF）、时域滞后（hysteresis）作为一等公民，宁可略慢收敛也不闪。

---

## 2. 现状盘点（可复用地基）

| 已有资产 | 位置 | 对应组件 | 成熟度 |
|---|---|---|---|
| SH L1 世界辐照缓存 / 八面体编码 / probe 放置 / probe 插值 | \`shading/gi/world_space\` | World Radiance Cache | CPU golden 有 |
| 软件 BVH：wide BVH / TLAS / 无栈遍历 / ray-offset / ray-cone footprint / motion | \`ray_scene\`（~15K，真机 parity） | Software Ray Tracing | 高，已真机 |
| GPU LBVH 构建（Karras）/ any-hit / closest-hit | \`virtual_geometry_gpu\`、\`physics_gpu/bvh\` | 加速结构 | 高，已真机 |
| 光追 ABI + TLAS + footprint 真机测试 | \`scene/raytrace\` | RT 场景接口 | 中高 |
| froxel 参与介质、大气服务 | \`volumetric\` / \`volumetric_gpu\` | 体积 GI 衔接 | 中 |
| 毛发 curve/segment 几何与着色 | \`hair_gpu\` | 毛发 GI 接收/投射 | 中 |
| 虚拟几何 cluster/卡片 | \`virtual_geometry\` | Surface Cache 图集来源 | 中高 |
| TAA/TSR：motion / history / upscale | \`render_*\` resolve | 时域累积/上采样衔接 | 高 |

**缺口**：场景表示层（Surface Cache + 全局 SDF/哈希）、最终 gather 收敛质量、降噪、反射统一、ReSTIR、采样质量（STBN/QMC）、海量光源（ReGIR/光源树）、性能分档、成体系的高级功能（半透明/折射/体积/毛发/水面/接触阴影）。

---

## 3. 总体架构

分六层（v2 独立出**采样与降噪层**），层间以稳定 ABI 解耦；每层可独立降级与 parity：

\`\`\`
[L0 场景表示]  Mesh SW-BVH(近场精确)
               + 全局稀疏 SDF/Brick(远场锥追, 借 Brixelizer)
               + Surface Cache 参数化卡(命中即取 radiance, 借 Lumen cards)
               + SHARC 空间哈希缓存(视相关, 免网格)
               + Distant Scene(超远场代理 + 高度场)
        │  统一 TraceRay ABI
[L1 追踪层]    软件遍历(默认, ray_scene) | 硬件 Ray Query(可选)
               + ray-cone LOD(已有 footprint) + ray-offset 自交避免
               + ray binning/排序(相干性) + 半分辨率追踪
        │  hit(位置/法线/材质id/radiance查询)
[L2 世界缓存]  World Radiance Cache(SH L1, 已有)
               + DDGI Chebyshev 可见性(抗漏光) + 探针分类/滞后
               + Radiance Cascades 远场分级
               + SHARC 哈希缓存(多弹跳兜底)
        │
[L3 屏幕探针]  自适应屏幕探针(已有骨架)
               + ReSTIR GI reservoir(时域重投影 + 空域邻域, GRIS 无偏)
               + 引导重要性采样(BRDF + 上一帧 radiance + 光源树/ReGIR)
        │
[L4 采样/降噪]  STBN + Owen-Sobol 低差异采样
               + firefly 钳制 + 能量守恒
               + 时空双边降噪(ReBLUR/ReLAX 式, diffuse/specular 分离)
               + 方差引导 + 历史 clamp + 视差重投影
        │
[L5 收敛/输出] Final Gather(探针→逐像素, 三级 fallback)
               + 统一反射(粗糙度分档: SSR近 + traced远 + 缓存兜底 + 平面反射)
               + AO/镜面遮蔽(XeGTAO) 高频补充
               + 与 PBR/NPR resolve 合流, 接 TAA/TSR(已有 motion/history/upscale)
\`\`\`

---

## 4. L0 场景表示层（最大新增，第二深水区）

Lumen 级性能的胜负手在这一层：**不对远处几何做逐三角追踪**。

### 4.1 混合表示
- **近场（默认 ~2–8m 可调）**：直接用已有 Mesh SW-BVH 精确追踪，命中做真实材质查询。
- **远场**：全局**稀疏 SDF / brick 体积**（借鉴 Brixelizer），做便宜的 sphere tracing 锥步进；分辨率随距离衰减。
- **超远场（开放世界）**：**Distant Scene**（借鉴 Lumen）—— 低模代理 + 高度场，给天光/大尺度 bounce 兜底。
- **低配兜底**：级联 SDF / 稀疏 voxel 八叉树 cone tracing（借 Godot SDFGI / CryEngine SVOGI），无 BVH 也能出低频 GI。

### 4.2 Surface Cache（多弹跳免费的核心）
- 为每个物体烘一组低分辨率参数化卡（albedo / normal / emissive / depth / material-id / roughness），命中时**直接查缓存的已着色 radiance，不做二次完整光照**。
- 缓存 radiance 每帧由直接光 + 上一帧间接光增量更新（feedback），实现"多弹跳近似免费"。
- 与 \`virtual_geometry\`（Nanite 式）集成：从虚拟几何的 cluster/卡片派生 surface cache 图集；卡片按屏幕投影面积做优先级流式更新。
- **卡片捕获调度**：按可见性/年龄/材质变化打分，GPU 驱动挑选每帧需重烘的卡片子集（capture budget），避免全量刷新。

### 4.3 SHARC 空间哈希辐射缓存
- 用世界空间**哈希表**（voxel key → radiance）替代/补充纯 voxel 网格：视相关分辨率、无需显式网格分配、天然稀疏。
- 作为二次及以上弹跳的 radiance 兜底源，命中远处时直接查哈希，避免深层递归；哈希项带样本计数做运行平均与自适应分辨率。

### 4.4 动态/蒙皮几何
- 蒙皮/形变物体每帧 BVH **refit**（复用 GPU LBVH 增量重建），surface cache 卡片跟随骨骼变换重投影；两面植被（two-sided foliage）单独走双面 BRDF 与半透传输。

---

### 4.5 Mesh Distance Fields + 距离场遮蔽（DFAO，新增 v3）
- 每个网格载入时生成**有向距离场（MDF）** 体积（低分辨率、压缩存储）；刚体仅用变换实例化，无需每帧重建，蒙皮件走粗代理 MDF。
- 全局**距离场场景**（global distance field，clipmap 合成所有 MDF）为中距离遮蔽提供便宜来源：
  - **距离场软 AO（DFAO）**：沿法线锥对距离场多采样，得到大范围柔和遮蔽，补屏幕空间 AO 的屏幕外盲区。
  - **距离场软阴影**：对主光/关键光做低成本软阴影，喂 GI 直接光项。
  - **短程间接遮蔽**：世界缓存 gather 前用距离场做可见性预剔除，减少漏光与无效射线（见 §12）。
- 定位：介于屏幕空间遮蔽与完整 BVH 之间的"便宜中距离"遮蔽层，低配档可独立作为 AO/软阴影来源。

---

## 5. L1 统一追踪层

- 单一 \`TraceRay(origin, dir, cone, tmax) -> Hit\` 抽象；后端可切：
  - **软件**（默认）：复用 \`ray_scene\` 无栈遍历、wide BVH、ray-offset、ray-cone footprint 做 mip/LOD。
  - **硬件 Ray Query**（可选）：能力探测通过后启用 hit-lighting 精确路径。
- **TLAS 每帧刷新**：复用现有 GPU LBVH 构建（Karras morton+radix）与真机 parity 的 any-hit/closest-hit。
- **锥追与射线追混合**：diffuse 用锥（低频、走 SDF/缓存），镜面用射线（高频、走 BVH）。
- **相干性优化**：ray binning / 方向排序后再遍历，改善 GPU cache/warp 相干；diffuse 探针射线做半分辨率追踪再上采样。

---

## 6. L2 世界辐照缓存

### 6.1 SH L1 probe（保留现有）
现有 voxel SH L1 辐照 probe 直接留用，作为世界缓存基础层。

### 6.2 DDGI Chebyshev 可见性（抗漏光）
- 每 probe 额外存**八面体深度 + 深度²** 图。
- 采样时用 Chebyshev 不等式估可见性上界，对 probe 贡献加权 → 显著抑制墙体漏光。
- **探针分类/滞后**：静止/被遮/自由态分类，无效探针休眠省算力；辐照更新带 hysteresis 抑制闪烁。

### 6.3 Radiance Cascades 远场分级
- 多级 voxel：近级高角分辨率（射线区间短、方向多），远级高空间分辨率（区间长、方向少），级间插值。
- 远场 diffuse 变**近常数成本**，是大场景的关键。

### 6.4 SHARC 兜底（见 §4.3）
二次弹跳命中查哈希缓存，避免深层递归。

### 6.5 探针流式（开放世界）
以相机为中心的 clipmap 式世界探针，随移动滚动更新边缘区块；探针数据分级 LOD，远处降密度。

---

### 6.6 Spherical Gaussian 辐射表示（新增 v3，可选高保真）
- 现有世界缓存用 **SH L1**（4 系数/通道），diffuse 足够但中粗糙/glossy 响应偏糊。
- 新增可选 **Spherical Gaussian（SG）** 瓣集（每探针 2–4 瓣）表示方向性辐射：
  - diffuse 仍走 SH L1（省内存）；中粗糙反射从 SG 瓣求值，保留方向性高光。
  - SG 与 GGX/BRDF 卷积有闭式/近似解，求值便宜，适合实时镜面兜底。
- 内存-质量档位：Low 仅 SH；Medium 加 1–2 SG；High/Ultra 2–4 SG + specular 独立降噪。
- 遵循 CPU golden → WESL → parity：SG 投影 / 求值 / 卷积均为确定性纯函数并配单测。

---

## 7. L3 屏幕探针 + ReSTIR

### 7.1 自适应屏幕探针（扩现有）
- 复用已有放置/插值/八面体骨架；按深度/法线不连续自适应加密。
- 每探针每帧仅射 **1–2 条**重要性采样光线。

### 7.2 ReSTIR GI（少射线高质量核心）
- **时域重用**：重投影上一帧 reservoir，MIS 合并；带样本年龄上限防过度相关。
- **空域重用**：邻域探针 reservoir 重采样（几何相似性加权）。
- **GRIS 广义无偏**：用广义平衡启发式做无偏/低偏重采样，等效数十~上百 spp 收敛，成本约 1–2 spp。
- **无偏验证**：保留 ground-truth 蒙特卡洛参考做 parity，bias/variance 显式验证。

### 7.3 引导重要性采样
- 采样方向 = BRDF 重要性 + 上一帧 radiance 引导 + 光源重要性。
- 光源重要性来自 **光源树 + ReGIR 世界光源网格**：海量光源/自发光三角下方差近恒定。

---

## 8. L4 采样与降噪层（v2 独立层，AAA 收敛质量核心）

### 8.1 低差异采样（STBN + QMC）
- 全局统一 **时空蓝噪声（STBN）掩码 + Owen-scrambled Sobol** 序列驱动所有随机决策（方向、光源、reservoir 抖动）。
- 蓝噪声让残余噪声在时空上"蓝"化，TAA/降噪后感知误差显著低于白噪声，低 spp 也干净。

### 8.2 能量稳定
- **firefly 钳制**：邻域/时域自适应上限 + karis 平均，抑制高方差亮点。
- **能量守恒**：多散射 BRDF 补偿、reservoir 归一化校正，避免整体偏暗/偏亮。

### 8.3 时空降噪（ReBLUR/ReLAX 式）
- 时空联合双边 + 方差引导 + 历史 clamp（抗拖影/闪烁）。
- diffuse/specular **分离降噪**，各自法线-深度-粗糙度-世界位置引导；specular 走 NDF 各向异性核。
- 视差重投影：镜面按虚像位置重投影，避免拉丝。
- 与已有 motion/history/upscale 衔接，接 TAA/TSR；支持棋盘/半分辨率 GI 的时域重建。

---

## 9. L5 收敛 / 反射 / 遮蔽

### 9.1 Final Gather
逐像素三级 fallback：屏幕探针 → 世界缓存(SH+可见性) → distant scene。按法线/深度加权插值，避免接缝。

### 9.2 统一反射（粗糙度分档）
- 极低粗糙：屏幕空间反射（SSR）近场 + traced 远场 fallback + **平面反射**（水面/镜面可选精确路径）。
- 中粗糙：traced 镜面射线（与 diffuse 共 BVH），镜面 ReSTIR 复用。
- 高粗糙：直接查世界缓存 / surface cache（与 diffuse 同源，几乎免费）。
- **多弹跳镜面**：反射命中再查 surface cache，避免"反射里一片黑"。

### 9.3 环境光/镜面遮蔽（高频补充）
- **XeGTAO** 式地平线 AO + 镜面遮蔽（bent normal），补 GI 无法覆盖的接触级高频暗部；与 GI 相乘时做能量协调避免过暗。
- **接触阴影（contact shadows）**：屏幕空间短射线补硬阴影接触细节。

---

### 9.4 Bent Normal 方向遮蔽 + 镜面抗锯齿（新增 v3）
- **Bent normal**：由 XeGTAO / 距离场遮蔽导出"未遮蔽平均方向 + 可见锥角"，用于：
  - diffuse GI 方向性遮蔽：按可见锥加权世界缓存查询，减少平面漏光。
  - 镜面遮蔽：反射锥与可见锥求交，抑制被遮挡方向的虚假高光。
- **镜面抗锯齿**：法线贴图/几何法线分布做 Toksvig / filtered-roughness，把几何高频转为有效粗糙度，消除镜面闪烁，并与 GI 反射 mip-cone 口径一致。

---

## 10. 高级功能（v2 扩充）

| 功能 | 借鉴 | 设计 | 档位 |
|---|---|---|---|
| **海量光源直接光**（ReSTIR DI） | RTXDI | 光源 reservoir 时空重用，与 GI 共享 BVH/阴影射线；万级光源可承 | Medium+ |
| **世界光源网格 + 光源树** | ReGIR / Cycles Light Tree | 世界空间光源重要性网格 + 层级采样，百万级光源/自发光三角方差近恒定 | High |
| **半透明 GI** | Lumen Translucency | 前向透明物体按屏幕探针 + 世界缓存查间接光；OIT 兼容 | Medium+ |
| **折射 GI / 玻璃** | 路径重采样 | 折射射线走 BVH + 世界缓存，玻璃后间接光正确 | High |
| **次表面 GI** | Lumen SSS | SSS 材质走扩散剖面 + 世界缓存低频间接 | High |
| **自发光网格作 GI 光源** | Lumen Emissive | surface cache emissive 通道直接进世界缓存 + 光源树采样 | Medium+ |
| **体积/参与介质 GI + 多散射** | Lumen Volumetric Fog | froxel 每 cell 查世界缓存 → 雾接收间接光；体积多散射近似 | Medium+ |
| **毛发 / 各向异性 GI** | — | 毛发 curve 接收世界缓存低频 GI，投射近似遮蔽；各向异性高光走 specular 降噪 | High |
| **水面 / 海洋反射** | 平面反射 + SSR | 平面反射 + traced 兜底，折射/焦散衔接体积 | High |
| **天光遮蔽 / Sky Occlusion** | DDGI skylight | probe 存天空可见性，天光按遮蔽衰减 | Low+ |
| **远场 / Distant Scene** | Lumen | 低模代理 + 高度场，开放世界大尺度 bounce | High |
| **镜面 ReSTIR** | ReSTIR reflections | 反射射线 reservoir 时空重用，粗糙反射降噪 | High |
| **焦散** | 光子 / ReSTIR PT | 自适应光子/路径重采样，水/玻璃焦散 | Ultra |
| **接触阴影 + XeGTAO** | XeGTAO | 屏幕空间高频遮蔽补 GI 细节 | Low+ |
| **多视图 / VR + 注视点** | — | per-view 探针与 reservoir 独立；注视点降密度省算力 | High |
| **GPU 驱动探针管理** | GPU-driven | probe/卡片分配·回收·重定位全 GPU，CPU 只给策略参数；indirect dispatch | 全档 |
| **动态天气/时间 GI** | — | 全动态无烘焙天然支持昼夜/天气突变，探针滞后收敛平滑过渡 | 全档 |

---

## 10b. 高级功能（v3 追加）

| 功能 | 借鉴 | 设计 | 档位 |
|---|---|---|---|
| **多弹跳镜面 / 反射中的反射** | Lumen hit-lighting | 反射命中再查 surface cache / 世界缓存，有界递归，避免"镜中纯黑"，镜中 GI 一致 | High |
| **各向异性 / 光泽反射** | SG + GGX aniso | SG 瓣 + 各向异性 NDF，拉丝金属/毛发高光方向正确 | High |
| **薄表面 / 双面 GI** | 两面材质 | 薄片（树叶/布/玻璃）正反面分别接收间接光，透射项按厚度衰减 | Medium+ |
| **贴花 GI 交互** | Deferred decals | 贴花写入 surface cache 的 albedo/normal/emissive，GI 自动反映贴花 | Medium |
| **透明排序 GI（OIT）** | WBOIT / per-layer | 多层透明各自查屏幕+世界探针，按权重合成，间接光不丢层 | High |
| **虚拟阴影贴图联动** | UE VSM | VSM 几何/剔除与 GI 共享，高频直接阴影喂 GI 直接光项 | High |
| **棋盘 / 交错渲染 GI** | Checkerboard | GI 以棋盘/交错半分辨率追踪，几何引导时域重建满分辨率 | 全档 |
| **反射探针 / 本地立方体贴图兜底** | 传统 reflection probe | 无 traced 预算时用预捕获/实时薄 cubemap 做镜面兜底，与 traced 平滑混合 | Low+ |
| **Spherical Gaussian 高保真镜面** | Frostbite SG | 中粗糙反射从 SG 求值，兼顾内存与方向性（见 §6.6） | Medium+ |
| **距离场软 AO / 软阴影（DFAO）** | UE DFAO | 全局距离场多采样柔和遮蔽 + 软阴影（见 §4.5） | Low+ |
| **镜面抗锯齿（filtered roughness）** | Toksvig | 法线/几何高频转有效粗糙度，消镜面闪烁（见 §9.4） | 全档 |
| **多层材质 / Clear Coat GI** | 分层 BRDF | 清漆/基底分层各自 specular 响应，GI 反射按层叠加 | High |
| **室内 / Portal 遮蔽稳定** | Portal culling | 室内外过渡用遮蔽/portal 约束世界探针可见性，防室内漏天光 | Medium+ |
| **近场高频接触 GI** | 近程 SSGI | 屏幕空间短程 diffuse bounce 补世界缓存低频，接触处细节更实 | Medium+ |

---

## 10c. 高级功能（v4 顶级增补，对标最新实时 AAA）

> 下列均为**纯经典数值**（蒙特卡洛 / 准蒙特卡洛 / reservoir / SDF / 光子 / 双边滤波 / 解析 BRDF），不含任何神经网络 / AI / ML / LLM 路径。全部遵循 CPU golden → WESL → 真机 parity 三步与分档降级矩阵。

| 功能 | 借鉴 | 设计要点 | 档位 |
|---|---|---|---|
| **World-Space ReSTIR**（空间哈希蓄水池） | NVIDIA WS-ReSTIR | reservoir 存入世界空间哈希网格（复用 SHARC key / `mesh_sdf` 体素化），跨帧 / 跨视角 / 多弹跳样本复用；GRIS 合并保持无偏；解遮挡与弱光区收敛大幅加速 | High |
| **Volumetric ReSTIR / Froxel GI** | Lin et al. / Lumen 体积 | froxel 网格每体素存散射 reservoir，时空重投影重用；体积雾 / 体积光 / 云获得多弹跳间接光；与现有 froxel 参与介质管线对接 | High |
| **焦散（Caustics）** | 自适应光子抛撒 + 流形 NEE | 水面 / 玻璃 / 金属锐利聚焦光：光子从光源经镜面 / 折射界面抛撒到屏幕空间焦散缓冲，或流形引导 NEE 直采；时域累积 + 自适应核；全经典 | High / Ultra |
| **Adaptive Probe Volumes（APV）** | Unity APV / UE 探针体 | 自适应密度辐照 + 可见性探针体，brick 按相机流式；天空遮蔽项防室内漏天光；作免烘动态兜底与远 / 静态区低频 GI，与屏幕探针平滑 LOD 混合 | Medium+ |
| **ReSTIR PT（完整路径重采样）** | Kettunen / Lin GRIS | 完整路径时空重用：shift map 用重连（reconnection）+ 随机重放（random-replay）两类，雅可比显式；Ultra 档 1–2 spp 逼近离线 PT；bias 对 ground-truth parity | Ultra |
| **随机 HiZ-SSR + 镜面 reservoir 重用** | 随机 SSR / ReSTIR 反射 | Hierarchical-Z 加速屏幕空间镜面步进，粗糙度抖动 + reservoir 时空重用降噪；与 traced 远场、反射探针按置信度平滑混合 | Medium+ |
| **微遮蔽 / 腔体 GI（micro-occlusion）** | 法线贴图微 AO / 微 bent normal | 由法线 / 高度贴图导出亚纹素尺度微 bent normal + 腔体 AO，多尺度与 GI mip-cone 口径一致，接触细节更实而不过暗 | Medium+ |
| **薄膜干涉 / 各向异性 GI 反射** | Belcour-Barla 薄膜 + aniso GGX | 薄膜色散与各向异性 NDF 并入 SG / traced 反射求值，GI 反射尊重切线帧（拉丝金属 / 肥皂泡 / 甲虫壳 / CD） | High |
| **水面 / 湿表面 GI** | 水体渲染 + 湿表面模型 | 水：屏幕空间 + 平面反射 + 折射 + 焦散按菲涅尔混合；湿表面降粗糙 / 加深 albedo 反馈 GI 反射；浪沫次表面低频散射 | High |
| **大气 / 体积云 多散射耦合 GI** | 大气多散射 LUT / 体积云 | 天空多散射 LUT 直接喂世界缓存天光项；体积云投射 / 接收低频 GI，与气透视（aerial perspective）耦合，开放世界远景一致 | Medium+ |
| **置换 / 曲面细分 GI 遮挡（DMM）** | Displaced Micro-Maps | 位移微网格直接参与 RT 遮挡与 surface cache 捕获，置换几何的接触遮蔽 / 自阴影正确，无需全细分 BVH | Ultra |
| **植被 / alpha-test GI（OMM）** | Opacity Micro-Maps | 不透明微网格让 alpha-test 树叶 / 栅栏在 RT 中按真实镂空透光，GI 穿叶正确且 any-hit 代价低 | High |

### 10c.1 World-Space ReSTIR 落地要点
- 哈希 key = 量化世界位置 + 法线分桶（复用 SHARC 的视相关分辨率策略），每格一组 reservoir。
- 复用 `screen_probe/restir.rs` 的 `Reservoir<S>` / GRIS `merge`：世界格样本作为**空域候选**喂屏幕 ReSTIR，诱导权重 `m*pdf*w` 保持无偏，`cap_confidence` 控制时域相关。
- 作为屏幕探针 reservoir 的"长期记忆"：相机快速移动 / 新进入视野的像素可立即从世界哈希取历史，避免解遮挡闪烁。

### 10c.2 Volumetric ReSTIR / Froxel GI 落地要点
- froxel 每体素存一个散射 reservoir（相位函数重要性 + 入射 radiance 目标函数），沿视线 ray-march 累积单 + 多散射。
- 时空重投影复用上一帧 froxel（考虑体素运动 / 相机运动），与 §8.3 时空降噪共用方差引导。
- 低档回退：无 reservoir 时退回现有 froxel 单散射 + 世界缓存天光，parity 覆盖两路。

### 10c.3 焦散落地要点
- **光子抛撒路径**：从关键光源发射光子，仅沿镜面 / 折射界面传播，命中漫反射面时抛撒进屏幕空间焦散累积缓冲（带时域滤波 + firefly 钳制）。
- **流形 NEE 路径**（Ultra）：对镜面 / 折射链用流形行走求解可连接路径，直采焦散，方差更低。
- 全程确定性 CPU golden：光子-界面求交、流形雅可比、累积核均为纯函数并配单测；无 ML。

### 10c.4 Ultra+（影视级实时）定位
- 在 Ultra 之上叠加 ReSTIR PT 完整路径重用 + Volumetric ReSTIR + 焦散 + SER/OMM/DMM 硬件加速，作为"实时逼近离线路径追踪"的旗舰档（对标 Portal RTX / Half-Life 2 RTX / Alan Wake 2 PT）。
- 仍受帧预算硬门禁约束（见 §11），超预算不毕业；默认关闭，能力探测 + 场景复杂度自适应开启。

---

## 11. 性能预算与分档矩阵（Metal 优先，design target）

| 档位 | 目标平台 | 屏幕探针 | spp | 世界缓存 | 远场 | 反射 | 高级功能 | GI 帧预算 |
|---|---|---|---|---|---|---|---|---|
| **Low** | M2 集显/移动 | 1/16px | 1 | SH L1 + cascades | voxel/SDF 锥追 | SSR + 缓存 | 天光遮蔽 + XeGTAO | ≤ 2.5 ms |
| **Medium** | M-Pro/Max | 1/8px | 1–2 + ReSTIR | + DDGI 可见性 + SHARC | SDF + cascades | traced 中粗糙 | RTXDI/半透明/自发光/体积 GI | ≤ 4 ms |
| **High** | 桌面独显 | 1/4px | 2 + ReSTIR | 全 | SW-BVH 近场 + distant scene | 镜面 ReSTIR + 平面反射 | + SSS/折射/毛发/水面/远场/VR + 光源树 | ≤ 6–8 ms |
| **Ultra** | 高端独显 | 1/2px | 2–4 + ReSTIR PT | 全 + 硬件 RT hit-lighting | 硬件 RT | 硬件 RT 反射 | + 焦散 + 注视点 + 高质量时空降噪 | ≤ 10–12 ms |
| **Ultra+** | 旗舰独显（影视级实时） | 1/1px | ReSTIR PT 完整路径重用 | 全 + 硬件 RT + SER + OMM/DMM | 硬件 RT + 流形 | 硬件 RT + 镜面 reservoir | + World-Space ReSTIR + Volumetric ReSTIR + 焦散 + 薄膜 / 各向异性 + 水面 / 湿表面 + 大气多散射 | ≤ 16–20 ms |

**性能杠杆（收益排序）**：Surface Cache（多弹跳变便宜）> ReSTIR/GRIS（降 spp）> STBN 采样（同 spp 更干净）> Radiance Cascades（远场常数）> SHARC（免网格兜底）> ray binning/半分辨率（相干+降载）> 硬件 Ray Query（有则加速）。

---

## 12. 性能优化专章（工程手段）

| 手段 | 说明 | 收益 |
|---|---|---|
| **异步计算重叠（async compute）** | GI 追踪/降噪与主 raster/阴影异步并行，填满 GPU 空泡 | 隐藏 20–40% 延迟 |
| **Indirect dispatch / GPU 驱动** | 探针、卡片、reservoir 全用 indirect，CPU 零逐物体开销 | 去 CPU 瓶颈 |
| **Ray binning + 方向排序** | 追踪前按方向/原点分桶，提升 warp/cache 相干 | SW-BVH 遍历提速 |
| **半分辨率追踪 + 时空上采样** | diffuse GI 半/四分之一分辨率追踪，几何引导上采样 | 追踪成本 ×0.25–0.5 |
| **数据打包** | radiance R11G11B10 / RGB9E5、法线八面体、reservoir 位压缩 | 带宽/显存 ×0.5 |
| **帧摊销预算（ray budget）** | 世界探针/卡片更新按帧滚动摊销，恒定每帧成本 | 峰值削平 |
| **subgroup/wave 内在** | 归约、reservoir 合并用 subgroup 操作（wgpu 支持时） | 减少 LDS 往返 |
| **VRS / 可变率 GI** | 平坦区域降 GI 速率，边缘保满 | 省 10–20% |
| **注视点渲染（VR）** | 外围视野降探针密度与 spp | VR 大幅省算力 |
| **capability 探测降级** | 无 subgroup/无硬件 RT 时自动落经典路径 | 全平台可跑 |
| **Bindless / 大资源表** | 场景几何/材质/卡片用 bindless 索引，去绑定切换，支撑海量实例追踪 | 去 draw/bind 开销 |
| **Persistent threads + 工作窃取** | 追踪用常驻线程 + 队列工作窃取，均衡长短射线负载 | 相干性外再提速 |
| **流式异步拷贝** | surface cache / SDF / 探针块用 copy queue 异步上传，追踪不停顿 | 隐藏流式延迟 |
| **距离场预剔除射线** | gather 前用全局距离场剔除确定遮挡方向，少发无效射线 | 减 10–30% 射线 |
| **Clustered / tiled 追踪** | 按屏幕 tile 聚类相似射线共享 BVH 遍历状态与缓存 | 提升 cache 命中 |
| **分级 mip 距离场** | 远场锥步进用 mip 距离场，步长随距离放大 | 远场追踪提速 |

| **Shader Execution Reordering（SER）** | 硬件 RT 命中后按材质 / 着色路径重排线程，恢复发散射线的 warp 相干 | hit-lighting 大幅提速（支持时） |
| **GPU Work Graphs / 动态射线生成** | GPU 自驱生成探针 / 卡片 / 二次射线工作项，去 CPU 往返，长短射线负载自均衡 | 去调度瓶颈、负载均衡 |
| **Opacity / Displaced Micromaps（OMM/DMM）** | 微网格描述镂空 / 位移，RT 遍历按微网格快速剔除 / 命中 | alpha-test / 置换几何追踪大幅省算 |
| **Mesh-shader 卡片捕获** | surface cache 卡片用 mesh/amplification shader 批量捕获，省 draw 调度 | 捕获吞吐提升 |
| **HiZ 加速屏幕空间步进** | 分级 Z 金字塔跳步，SSR / 接触阴影 / SSGI 步进次数骤减 | 屏幕空间 trace 提速 |

所有优化项都遵循 graduation gate：优化前后 parity 一致，性能达标才开启。

---

## 13. 与 Prism 范式对接（模块/文件规划）

新增模块（均走 CPU golden → WESL → 真机 parity 三步）：

\`\`\`
prism_render_shading/src/gi/
  world_space/            # 已有: radiance_cache, octahedral, probe_*
    visibility.rs         # DDGI Chebyshev 可见性 + 探针分类/滞后
    cascades.rs           # Radiance Cascades 远场分级
    sharc.rs              # 空间哈希辐射缓存
    clipmap.rs            # 开放世界探针流式
  screen_probe/
    adaptive.rs           # 屏幕探针自适应加密
    restir.rs             # ReSTIR GI reservoir (时空重用 + GRIS MIS)
    guided_sampling.rs    # 引导重要性采样
  sample/
    stbn.rs               # 时空蓝噪声掩码
    sobol.rs              # Owen-scrambled Sobol 低差异序列
    light_tree.rs         # 光源树层级重要性采样
    regir.rs              # 世界空间光源网格
  scene/
    surface_cache.rs      # Surface Cache 卡片烘焙/调度/更新
    sdf_brick.rs          # 全局稀疏 SDF/brick (远场)
    distant_scene.rs      # 远场代理
    skinned_refit.rs      # 蒙皮/形变几何 BVH refit
  reflect/
    unified_reflect.rs    # 粗糙度分档反射 + 多弹跳
    planar.rs             # 平面反射 (水面/镜面)
  denoise/
    reblur.rs             # 时空双边降噪 (diffuse/specular 分离)
    variance.rs           # 方差引导 + 历史 clamp + firefly 钳制
  occlusion/
    gtao.rs               # XeGTAO AO + 镜面遮蔽 + 接触阴影
  direct/
    restir_di.rs          # RTXDI 式海量光源直接光
  integrate/
    final_gather.rs       # 逐像素三级 fallback 合流

WESL kernels:
  gi_trace.wesl gi_restir.wesl gi_gather.wesl gi_denoise.wesl
  gi_surface_cache.wesl gi_sdf_trace.wesl gi_cascades.wesl
  gi_sharc.wesl gi_restir_di.wesl gi_reflect.wesl gi_planar.wesl
  gi_stbn.wesl gi_sobol.wesl gi_light_tree.wesl gi_regir.wesl
  gi_gtao.wesl gi_firefly.wesl
\`\`\`

接入：RenderApp per-view system，排在 opaque/visibility 之后、resolve 之前；异步计算队列并行；gate 默认关闭。

---

**v3 新增模块文件**（延续 CPU golden → WESL → 真机 parity 三步）：

```
prism_render_shading/src/gi/
  world_space/
    spherical_gaussian.rs   # SG 瓣投影/求值/BRDF 卷积 (§6.6)
  scene/
    mesh_sdf.rs             # 每物体 MDF + 全局距离场合成 (§4.5)
    dfao.rs                 # 距离场软 AO / 软阴影
  occlusion/
    bent_normal.rs          # bent normal 方向遮蔽 + 镜面遮蔽 (§9.4)
    specular_aa.rs          # filtered roughness / Toksvig 镜面抗锯齿
  reflect/
    reflection_probe.rs     # 本地 cubemap / 反射探针兜底
    multi_bounce_spec.rs    # 多弹跳镜面有界递归
  integrate/
    checkerboard.rs         # 棋盘/交错 GI 时域重建
  material/
    decal_gi.rs             # 贴花写入 surface cache
    layered_gi.rs           # 多层 / clear coat GI 反射

  world_restir/
    hash_grid.rs            # 世界空间哈希网格 (复用 SHARC key)
    world_reservoir.rs      # 世界格 reservoir + GRIS 合并到屏幕 ReSTIR
  volumetric_gi/
    froxel_reservoir.rs     # 体素散射 reservoir + 时空重投影
    volumetric_restir.rs    # 参与介质多散射重采样积分
  caustics/
    photon_splat.rs         # 镜面/折射光子抛撒 + 屏幕空间累积
    manifold_nee.rs         # 流形行走下一事件估计 (Ultra)
  probe_volume/
    apv.rs                  # 自适应密度探针体 + brick 流式
    sky_occlusion.rs        # 天空遮蔽项 (防室内漏天光)
  path_reuse/
    restir_pt.rs            # 完整路径重采样
    shift_map.rs            # 重连/随机重放 shift + 雅可比
  reflect/                  # (扩展 v3 reflect/)
    stochastic_ssr.rs       # 随机 HiZ-SSR + 镜面 reservoir 重用
  micro/
    micro_occlusion.rs      # 微 bent normal + 腔体 AO
  material/                 # (扩展 v3 material/)
    thin_film.rs            # 薄膜干涉 + 各向异性 GI 反射
    water_gi.rs             # 水面/湿表面 GI (反射+折射+焦散混合)
  atmosphere/
    multiscatter_gi.rs      # 大气/体积云多散射耦合天光

WESL kernels（新增）：
  gi_sg_eval.wesl gi_mesh_sdf.wesl gi_dfao.wesl gi_bent_normal.wesl
  gi_specular_aa.wesl gi_reflection_probe.wesl gi_checkerboard.wesl
  gi_world_restir.wesl gi_volumetric_restir.wesl gi_photon_splat.wesl
  gi_apv.wesl gi_restir_pt.wesl gi_stochastic_ssr.wesl gi_micro_occlusion.wesl
  gi_thin_film.wesl gi_water.wesl gi_atmosphere_gi.wesl
```

---

## 14. 分阶段路线（按收益/风险排序）

| 阶段 | 内容 | 产出 | 增量估计 |
|---|---|---|---|
| **P1 打通闭环** | 世界缓存 + 屏幕探针 + SW-BVH 单弹跳 diffuse GI，真机 parity | 第一张"能看的间接光" | ~8–12K |
| **P2 采样+抗漏光+降噪** | STBN/Sobol + DDGI 可见性 + ReBLUR 式时空降噪 + firefly 钳制 | 稳定、无漏光/噪点/闪烁 | ~12–18K |
| **P3 性能层** | Surface Cache + SHARC + 多弹跳 + ReSTIR GI(GRIS) | 多弹跳近免费、spp 降到 1–2 | ~15–25K |
| **P4 远场+反射+遮蔽** | Radiance Cascades + SDF/brick + 统一反射 + 平面反射 + XeGTAO | 大场景常数成本 + 全粗糙度反射 + 高频细节 | ~14–22K |
| **P5 高级功能** | RTXDI + 光源树/ReGIR + 半透明/折射/自发光/体积/毛发/水面 GI + 天光遮蔽 | 海量光源、全材质接收 GI | ~14–24K |
| **P6 硬件后端 + Ultra** | 硬件 Ray Query hit-lighting + ReSTIR PT + distant scene + SSS + 焦散 + 注视点 | 桌面/高端画质对标顶级 AAA | ~12–24K |
| **性能优化（贯穿）** | 异步计算 + ray binning + 半分辨率 + 打包 + VRS + 帧摊销 | 各档达帧预算 | ~8–15K |
| **P7 v3 增强** | SG 辐射 + MDF/DFAO + bent normal/镜面AA + 多弹跳镜面 + 贴花/多层/OIT GI + 棋盘重建 + bindless/persistent-thread | 更高镜面保真 + 更稳遮蔽 + 更低抖动 | ~10–18K |

| **P8 v4 顶级增补** | World-Space ReSTIR + Volumetric ReSTIR + 焦散 + APV + ReSTIR PT + 随机HiZ-SSR + 微遮蔽 + 薄膜/各向异性 + 水面/湿表面 + 大气多散射 + SER/OMM/DMM/Work Graphs | 影视级实时上限（Ultra+），逼近离线路径追踪 | ~18–30K |

**合计 ~111–188K 行**（含 surface cache、v2/v3/v4 高级功能与性能优化这类重活；不含高级功能与优化约 ~50–75K）。

---

## 15. 风险与降级矩阵

| 风险 | 缓解 |
|---|---|
| Metal 无稳定硬件 RT | 软件 BVH 为默认（已真机验证），硬件仅可选档 |
| Surface Cache 工作量近 Nanite 级 | P3 才做；P1/P2 先用世界缓存兜多弹跳，不阻塞"能看" |
| ReSTIR/GRIS 时域拖影/偏差 | 保留 ground-truth 参考做 parity，bias 显式验证，历史 clamp + 样本年龄上限 |
| 漏光/闪烁 | DDGI 可见性 + 方差引导 + 时域 clamp + 探针滞后 |
| SHARC 哈希冲突/抖动 | 冲突降级到 voxel，抖动交时域降噪吸收 |
| 海量光源方差爆炸 | 光源树 + ReGIR 世界网格，方差随光源数近恒定 |
| firefly / 能量不守恒 | 自适应钳制 + 多散射补偿 + reservoir 归一化 |
| 半分辨率/VRS 边缘瑕疵 | 几何引导上采样 + 边缘保满速率 |
| 异步计算竞态 | 显式 barrier/资源状态跟踪，parity 覆盖并行/串行两路 |
| 每档性能不达标 | 分档矩阵 + 帧预算控制器硬性 gate，超预算不毕业 |

---

## 16. 验收标准（毕业门槛）

每个子模块毕业需同时满足：
1. **CPU golden 参考**：确定性纯函数，单元测试全绿。
2. **WESL 编译**：naga 验证通过，shader 静态测试绿。
3. **真机 Metal parity**：GPU 输出对 CPU golden 逐分量对拍，容差显式且远紧于 bug 级发散。
4. **无偏验证**：ReSTIR/GRIS/时域路径对 ground-truth（大 spp 蒙特卡洛）收敛，bias 有界。
5. **性能门禁**：目标档位帧预算内，接帧预算控制器，回归场景无退化。
6. **降级验证**：低档回退路径同样 parity 通过。
7. **画质基准**：与离线路径追踪参考图做感知误差（如 FLIP/相对 MSE）对比，达到目标档位阈值。

全部通过前，对应 gate（如 \`gi_diffuse_enabled\` / \`gi_restir_enabled\` / \`gi_hardware_rt_enabled\` / \`gi_light_tree_enabled\`）保持默认关闭。

---

## 17. 一句话总结

**方案 = Lumen 拓扑骨架（Prism 已有一半）+ DDGI 抗漏光 + ReSTIR/GRIS/RTXDI 少射线路径级质量 + 光源树/ReGIR 海量光源 + Radiance Cascades 远场常数成本 + Surface Cache/SHARC 多弹跳免费 + Brixelizer 式 SDF 远场 + STBN/Sobol 低差异采样 + NRD 式经典时空降噪 + SG 高保真镜面 + MDF/DFAO 中距离遮蔽 + bent normal 方向遮蔽 + XeGTAO 高频遮蔽 + 多弹跳镜面 + 异步/半分辨率/VRS/bindless 工程优化，全部套进 Prism 已验证的"软件 BVH + CPU golden + 真机 parity"范式。** 先 P1/P2 拿到"稳定能看的间接光"（~2–3 万行），再 P3–P6 叠加高级功能与性能优化，P7 以 SG 辐射 / MDF-DFAO / bent normal / 多弹跳镜面 / 贴花·多层·OIT GI / 棋盘重建 / bindless·persistent-thread 固顶级次世代 AAA 天花板，P8（v4）再以 World-Space ReSTIR / Volumetric ReSTIR / 焦散 / APV / ReSTIR PT / 随机 HiZ-SSR / 微遮蔽 / 薄膜·各向异性 / 水面·湿表面 / 大气多散射 + SER·OMM/DMM·Work Graphs 推到 **Ultra+ 影视级实时**（对标 Portal RTX / Half-Life 2 RTX / Alan Wake 2 PT，累计 ~11–19 万行）。全程纯经典数值，无任何 AI/ML/神经网络/LLM 路径。
