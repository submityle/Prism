# Prism 渲染引擎 — 水体/流体引擎子系统完整设计（v1 / wgpu + WESL）

> 状态：架构提案（Draft，允许破坏性重构）
> 定位：AAA / 次世代 水体·海洋·自由表面流体引擎，与 PBR/NPR/自定义/混合四前端正交协同
> Shader：WESL（naga → WGSL/SPIR-V/Metal/DXIL）；运行时：wgpu（Metal/VK/DX12/WebGPU）
> 关联文档：`prism_rendering_architecture_zh.md`（总）、`prism_material_pipeline_design_zh.md`（§6.2 子系统注册表：水/海洋=档2 后置留槽、§1「NPR 不与 Water 同级」）、`prism_particle_engine_design_zh.md`（飞沫/泡沫/气泡由 Ember 承接；网格流体互补）、`prism_physics_design_zh.md`（§3 统一 XPBD、§4 多求解器、§11 GPU 持久化、§12 异步流水线）、`prism_hair_engine_design_zh.md` / `prism_cloth_engine_design_zh.md`（对称子系统范式）、`prism_aaa_advanced_features_zh.md`
> 沙盒说明：本机屏蔽 Metal（无 GPU），文档内帧预算/带宽/耗时均为**设计目标（design target），非实测**，如实标注。可验证部分限于 CPU 可计算的纯函数（求解步进、波谱采样、LOD 决策、分桶）。
> 数值路线：纯经典数值（FFT/FLIP/PBF/SWE/波谱），不含任何 AI/ML 内容。

---

## 0. 定位：这是什么级别的水体引擎

是**完整的水体/流体引擎**（海洋波谱 + 自由表面流体 sim + 表面重建 + 水体特殊渲染），不是"给平面贴一张法线流动贴图"的着色贴皮。对标业界成熟产品，仅在**算法层**借鉴，不复制其代码：

| 产品 | 借鉴点（算法层） |
|---|---|
| **UE5 Water（Water System / Waterline）** | 河流/湖泊/海洋样条水体、Gerstner 波叠加、水下后处理、waterline（水线掩膜）、水体深度/边缘泡沫、可行走浅水 |
| **UE5 Niagara Fluids** | GPU 网格流体（2D/3D）、烟/火/液体求解模板、与粒子系统耦合、GPU-driven 全流程 |
| **Crest Ocean（含 HDRP/URP）** | LOD 级联位移贴图（cascade displacement）、动态波注入（dynamic wave sim）、水体深度缓存、焦散/泡沫/浅水混合、屏幕空间水下 |
| **Tessendorf FFT 海洋（影视/游戏通行）** | Phillips/JONSWAP 谱、IFFT 位移+法线+雅可比、多方向多尺度级联叠加 |
| **Sea of Thieves / AC 系船海** | 艺术可控波形、船体交互浪、水花与泡沫尾迹、廉价高质海面着色 |
| **Position-Based Fluids（PBF, Macklin 2013）** | 不可压缩 SPH 的 XPBD 化密度约束、GPU 邻域并行、飞沫/交互液体 |
| **FLIP / APIC（Houdini / Zhu-Bridson / Jiang APIC）** | 粒子-网格混合自由表面流体、低耗散、角动量守恒（APIC）、大规模液体 sim |
| **Shallow Water Equations（SWE）** | 高度场浅水波、河流/洪水/交互涟漪的廉价 2.5D 求解 |
| **屏幕空间流体（Screen-Space Fluids, van der Laan）** | 粒子→深度→双边平滑→法线重建的实时液体表面渲染 |

**判据（承接材质设计 §6.2）**：一等子系统必须拥有**自己的几何 + 自己的 sim + 特殊渲染**。水体三者全占：
- **几何**：动态位移海面网格（clipmap/级联）、SWE 高度场、FLIP/PBF 表面重建网格；
- **sim**：不可压缩自由表面流体（FLIP/APIC/PBF）、浅水方程、海洋波谱 IFFT、动态波注入；
- **特殊渲染**：折射 + 屏幕空间/RT 反射 + 焦散 + 泡沫/飞沫 + 水下体积散射 + 水线掩膜。

因此水体是**一等子系统**（材质 §6.2 将其列为「档2 后置留槽」——是真子系统，但重且非通用，wgpu 竖切 MVP 阶段暂不接线，随基底成熟后跟进）。**关键立场（承接材质 §1）**：**NPR 不与 Water 同级**——Water 是 closure/子系统（有几何+sim），NPR 是 illumination（风格轴）。水面既可 PBR 也可 NPR 着色，两者是正交的两个维度。

---

## 1. 与整体架构的关系（子系统内部同构范式）

承接毛发/布料/粒子的同构范式：**共享基底 + PBR/NPR 分叉响应**。

```
共享基底(PBR/NPR/自定义/混合 都吃):
  水体几何(海面级联位移网格 / SWE高度场 / FLIP-PBF表面重建, 连续LOD 禁硬切换)
  + 统一流体 sim(对齐 physics_core 的 XPBD/网格求解原语)
  + 折射(共享场景颜色/深度) + 屏幕空间反射(共享 HZB/SSR) + RT 反射(共享 ray_scene 代理)
  + 水下体积(接入共享 froxel 体积光照) + 焦散(共享光照注入) + 水线掩膜(共享 prepass)
  + 泡沫/飞沫/气泡(委托 Ember 粒子子系统, 不自建粒子池)
  + motion vector(共享时序服务, 供 TAA/上采样)

分叉响应(C类, 真分家 —— 只在"光照响应"分家, 见 §5):
  PBR:  物理菲涅尔 + 环境/SSR/RT 反射 + Beer-Lambert 吸收/散射 + 次表面透射 + 微表面波法线
  NPR:  ramp 量化水色 + 卡通高光块 + 手绘泡沫线/浪花描边 + 风格化焦散(网点/线条) + 水墨扩散
  自定义/混合: 同一网格多材质槽, 逐区/逐层混合 PBR↔NPR (§5.4)

fallback:  近景 FLIP/PBF 表面重建, 远景海洋级联位移; 极远用高度场/静态法线;
           RT 反射里流体用 SDF/包围代理或降级为 SSR
```

**所有一等子系统内部都长这个样**（共享基底 + PBR/NPR 分叉响应），架构一致性拉满。子系统**只产几何/位移/表面数据**；着色走共享材质 closure、透明/折射走共享服务、约束求解对齐 `prism_physics_core`——**不重写这些**。

与其它子系统的边界（去重，避免重复造轮子）：
- **飞沫/泡沫/气泡/水花** → 委托 **Ember 粒子引擎**（它已有 GPU 粒子池/排序/OIT/着色四等公民）。水体只发射 spawn 事件与 reactive mask。
- **气态体素流体（烟/火/爆炸）** → 归 **Ember 网格流体**（材质文档已定：粒子引擎吸收气态流体）。**本引擎只管液态/自由表面/海洋**。
- **约束/积分原语**（XPBD 密度约束、网格投影） → 对齐 `prism_physics_core` 统一求解器，不另起炉灶。
- **破碎/刚体浮力反作用** → 归 `physics_core`；水体只提供高度/速度场查询接口供浮力使用。

---

## 2. 数据与资产模型

三类水体资产（Body），共享统一 `WaterBody` 抽象，按几何/sim 特性分派求解器：

| 资产类型 | 几何形态 | 求解器 | 典型用途 |
|---|---|---|---|
| **Ocean（海洋）** | 无限级联位移网格（clipmap/投影网格） | 波谱 IFFT + 可选动态波注入 | 开放海、大湖 |
| **Surface（水面/河湖）** | 样条/多边形围成的 2.5D 高度场 | SWE 浅水方程 + Gerstner 叠加 | 河流、湖泊、池塘、洪水 |
| **Volume（体积液体）** | 3D 域内粒子 + 重建网格 | FLIP/APIC 或 PBF（不可压缩） | 泳池溅水、翻涌液体、管道、局部大水 |

统一资产字段（`WaterBodyAsset`）：
- 几何：域包围盒 / 样条边界 / 分辨率（网格 cell 或粒子上限）；
- 物性：密度、粘度、表面张力、静止水位、消光系数（吸收 RGB）、散射系数、折射率（IOR≈1.33）；
- 波形（Ocean/Surface）：谱类型（Phillips/JONSWAP/PM）、风速/风向、级联数与各级尺度、陡度（choppiness）、方向扩散；
- 交互：可注入源（船体/角色/爆炸）、边界条件（反射/吸收/周期）；
- 着色：材质句柄（PBR/NPR/自定义/混合）、泡沫阈值、焦散强度、水下能见度、浅水渐变曲线。

资产以数据驱动（RON/序列化），运行期编译为求解器参数 + WESL specialization key。

---

## 3. 管线阶段（端到端，每帧全链路）

严格 GPU-first、零回读（浮力查询走小批量 readback 或 GPU 侧回写）。extract → prepare → queue 融入 Prism 渲染图：

```
[Extract]   收集活跃 WaterBody 组件 + 交互源(船/角色/冲击) → GPU uniform/instance
[Prepare]   1. 波谱更新: 时间推进相位 → IFFT(位移/法线/雅可比) 级联贴图
            2. 动态波注入: 交互源 → 波纹/涟漪写入动态贴图(Crest 式)
            3. 流体 sim(Volume): FLIP/APIC/PBF compute 步进(见 §7)
            4. 浅水 sim(Surface): SWE 高度场步进(见 §6.3)
            5. 表面重建(Volume): 粒子→密度场→等值面(MC/各向异性/屏幕空间)
[Queue]     6. LOD 选择 + clipmap tile 决策 + 视锥/遮挡剔除
            7. 位移网格生成/更新(indirect draw)
            8. Prepass: 写水体深度/水线掩膜/motion vector
            9. Gbuffer or Forward: 水面 closure 着色(PBR/NPR/混合)
           10. 折射: 采样共享场景颜色缓冲(prepass 已隔离)
           11. 反射: SSR(HZB) → miss 回退 RT/环境探针
           12. 焦散: 光空间雅可比投影 or 光子/RT 焦散注入
           13. 水下体积: froxel 单/多次散射 + god ray + 能见度雾
           14. 泡沫/飞沫: 依 sim 生成 reactive mask → Ember 发射粒子
[Post]     15. 水下后处理: 色偏/模糊/色差, 水线过渡(相机穿越水面)
           16. 时序: motion vector 供 TAA/上采样, 泡沫历史累积
```

关键差异化：**波谱、SWE、FLIP/PBF 均为可独立启用的求解桶**，一个场景可同时存在（远海用波谱、近岸用 SWE 涟漪、局部翻涌用 FLIP），在同一深度/折射/反射服务下统一合成。

---

## 4. 求解器矩阵（多桶，各司其职）

| 求解器 | 维度 | 不可压缩 | 强项 | 代价 | 落点 |
|---|---|---|---|---|---|
| **波谱 IFFT（Tessendorf）** | 2.5D 高度/位移 | 不适用(动画) | 无边海面、真实海浪统计谱、极高性价比 | FFT 每级 O(N²logN) | Ocean 基线 |
| **Gerstner 叠加** | 2.5D | 不适用 | 艺术可控、少波数、廉价、可解析求导 | 波数多则贵 | Ocean/Surface 补充 |
| **SWE 浅水方程** | 2.5D 高度场 | 近似 | 河流/洪水/交互涟漪、廉价 2D 步进 | 浅水假设(不能翻卷) | Surface |
| **PBF（Position-Based Fluids）** | 3D 粒子 | 是(密度约束 XPBD) | 与 XPBD 统一、交互强、稳定 | 邻域搜索、迭代 | Volume（交互液体） |
| **FLIP/APIC** | 3D 粒子+网格 | 是(网格投影) | 大规模低耗散、翻涌飞溅、APIC 保角动量 | 网格压力解(泊松) | Volume（大水/影视级） |

**取舍原则**：能用高度场就不用 3D 粒子；能用波谱就不用求解。**远海=波谱级联，近岸交互=SWE+动态波注入，局部剧烈翻涌=FLIP/PBF**，三者在深度合成阶段无缝拼接（§8 LOD/过渡）。求解迭代/投影原语对齐 `prism_physics_core` 的统一 XPBD 与网格投影内核，避免重写。

---

## 5. PBR / NPR / 自定义 / 混合 分叉响应

**只在"光照响应"分家**，几何/sim/折射/反射/深度/motion 全部共享（§1）。

### 5.1 PBR 水面（物理）
- **菲涅尔**：Schlick 近似（F0≈0.02），掠射高反射；
- **反射**：SSR（HZB 追踪）→ miss 回退 RT 反射（`ray_scene` 代理）→ 再回退环境探针/立方图；
- **折射**：采样折射前场景颜色，按 IOR 与深度做屏幕空间偏移 + 色散（可选）；
- **吸收/散射**：Beer-Lambert 按光程指数衰减（水色随深度变蓝绿）+ 单次散射项；
- **微表面**：波谱法线 + 细节法线 + 雅可比驱动的泡沫遮罩，GGX 高光；
- **次表面透射**：背光浪尖的透光（浪峰散射，Sea of Thieves 式廉价近似）。

### 5.2 NPR 水面（风格化，illumination 轴）
- **ramp 量化水色**：深浅按 ramp LUT 分级（塞尔达/卡通渲染）；
- **卡通高光**：SDF/阈值化的块状高光而非连续 GGX；
- **手绘泡沫**：等值线/边缘检测生成描边泡沫线（对标塞尔达浅滩白边、崩坏卡通水）；
- **风格化焦散**：网点（halftone）/线条化焦散，而非物理光斑；
- **水墨/宣纸**（可选）：Okami 式扩散边缘 + 纸纹（复用材质 §NPR 风格库）；
- **流动线**：沿流速场的手绘流动纹（速度线）。

### 5.3 自定义
- über-closure 暴露 hook：作者可注入自定义 BRDF 瓣/水色映射/泡沫合成，编译进 WESL specialization，不改内核。

### 5.4 混合（管线级/材质级）
- **同一水体多材质槽**：如主体 PBR、岸边过渡带 NPR、特定区域自定义；
- **逐层/逐区混合**：按掩膜（深度、距岸、区域贴图）在 PBR↔NPR 间插值光照响应；
- **正交叠加**：PBR 折射/反射基底 + NPR 描边/网点作为风格层叠加（Arcane 式 3D 上叠手绘）。

**四者皆一等公民**：共享基底服务（虚拟几何/GI/VSM/体积/RT/时序上采样）对 PBR 与 NPR 水面**同等可用**，差异只在最终光照解读方式。

---

## 6. 海洋波谱与浅水（Ocean / Surface）

### 6.1 波谱模型
- 谱：Phillips（经典）/ JONSWAP（风浪成长）/ Pierson-Moskowitz（充分成长海）；
- 初始频谱 H0(k) 由谱密度 + 高斯随机相位生成（一次性）；
- 时间演化 H(k,t) = H0(k)·e^{iω(k)t} + conj(H0(-k))·e^{-iω(k)t}，色散 ω(k)=√(g·k)（深水）；
- IFFT → 高度场；同时 IFFT 求 x/z 位移（choppy/Gerstner 化尖浪）、法线（斜率）、雅可比行列式（泡沫判据：J<阈值 = 波峰折叠 = 白沫）。

### 6.2 级联（cascade）
- 多级不同物理尺度（如 4 级：大浪/中浪/细浪/涟漪）叠加，避免单级平铺重复；
- 每级独立分辨率与世界尺度，采样时按距离加权，远处降级少级（LOD）。

### 6.3 SWE 浅水方程（Surface）
- 高度场 h + 通量（hu, hv），显式/半隐式步进，CFL 稳定；
- 交互涟漪：外力/边界注入（角色/雨滴/船），阻尼衰减；
- 与波谱叠加：近岸浅水项 + 远海波谱位移在高度合成阶段相加。

### 6.4 动态波注入（Crest 式）
- 交互源（船体轨迹、角色涉水、爆炸）写入一张随相机移动的动态位移/法线贴图；
- 与波谱级联叠加，产生船首浪、尾迹、局部扰动，无需全域 3D sim。

---

## 7. 自由表面流体（Volume：FLIP/APIC/PBF）

### 7.1 PBF（默认交互液体）
- SPH 邻域 → 密度约束 C_i = ρ_i/ρ0 − 1，XPBD 迭代投影（对齐 physics_core）；
- 人工压力项（防粒子聚簇）、涡量约束（补耗散涡旋）、XSPH 粘度；
- GPU 邻域：空间哈希/网格分桶（compute），确定性输入序。

### 7.2 FLIP/APIC（大规模/影视级）
- 粒子→网格（P2G）→ 压力泊松解（不可压缩投影）→ 网格→粒子（G2P）；
- APIC 用仿射速度场保角动量，抑制 FLIP 噪声与耗散；
- 压力解：Jacobi/共轭梯度/多重网格（GPU 并行），MAC 网格。

### 7.3 表面重建
- 粒子密度场 → 等值面提取；三条路线：
  - **屏幕空间流体**（默认实时）：粒子 splat 深度 → 双边/窄带平滑 → 法线重建 → 直接着色（van der Laan，最省，视相关）；
  - **各向异性 Marching Cubes**（离线/近景高配）：各向异性核（Yu-Turk）→ 光滑连续网格；
  - **narrow-band SDF**：局部 SDF + MC，兼顾质量与带宽。

### 7.4 耦合
- 与刚体：浮力/拖曳（physics_core 查询水速/水位）；
- 与海面：Volume 边界读取 Ocean 高度做开边界；
- 与泡沫粒子：高速度/高曲率/低雅可比处发射 Ember 飞沫/气泡。

---

## 8. LOD 与大世界（禁 pop 硬切换）

- **海面 clipmap/投影网格**：以相机为中心的同心环网格，随距离降密度；连续 morph（geomorphing）过渡，禁硬切换；
- **波谱级联 LOD**：远处减少叠加级数、降低法线细节，用 mip 采样；
- **求解器过渡带**：近景 FLIP/PBF 表面重建，中景 SWE，远景波谱——在过渡带做高度/法线的距离加权融合，避免接缝；
- **瓦片流（tile paging）**：超大水域按瓦片按需驻留位移/深度缓存，接入共享 `paging`/`texture_streaming`；
- **RT 降级**：RT 反射中流体用 SDF/包围盒代理或直接排除，回退 SSR。

---

## 9. 特殊渲染服务（共享 or 水体专属）

| 服务 | 归属 | 说明 |
|---|---|---|
| 折射 | 共享场景颜色/深度 | prepass 隔离水前场景，屏幕空间偏移采样 |
| 反射 | 共享 SSR(HZB) + RT(ray_scene) + 探针 | 三级回退 |
| 焦散 | 水体注入 + 共享光照 | 光空间雅可比投影（廉价）或 RT/光子焦散（高配）→ 写入光照/贴花 |
| 水下体积 | 共享 froxel 体积 | 单/多次散射、god ray、能见度雾、消光按 Beer-Lambert |
| 水线掩膜 | 共享 prepass | 相机穿越水面的水上/水下分割、边缘过渡 |
| 泡沫/飞沫/气泡 | **委托 Ember 粒子** | 水体只发 reactive mask + spawn 事件 |
| motion vector | 共享时序 | 位移网格顶点速度 + 流速场 → TAA/上采样 |

---

## 10. 与共享基底的接线（复用，不重写）

- 着色 closure / OIT / 透明：材质系统（`transparency`、über-closure）；
- 光照 / 体积 / 阴影：`lighting`（聚类）、froxel 体积、`virtual_shadow`（VSM）、`ray_scene`（RT）；
- 时序：`temporal_upscale`、`motion`、`history`；
- GPU-driven：`gpu_scene`、indirect draw、`work_graph`、HZB 遮挡剔除；
- 资源：`virtual_resource`、`descriptor_heap`、`memory`、`paging`、`texture_streaming`；
- 求解原语：`prism_physics_core`（XPBD 密度约束、网格投影、积分器）。

水体子系统**新增的最小内核**只有：波谱 IFFT、SWE 步进、FLIP/PBF 步进与投影调度、表面重建、级联/clipmap LOD 决策、焦散雅可比投影、水线掩膜生成、reactive mask 发射——其余全部消费共享服务。

---

## 11. Crate 拆分与落地形态

落点 `pkg/prism_render_architecture/src/water/`（照抄 cloth/hair/particle 范式，零依赖、`#![forbid(unsafe_code)]`、`extern crate alloc;`、手写向量数学）。CPU 可验证的纯函数与调度先行；GPU shader（WESL）先落契约签名+脚手架，不作本机编译验证目标（沙盒无 GPU）。

建议文件（每文件单一职责，禁大文件堆一处）：
```
water/
  mod.rs            契约: WaterBody/WaterKind/WaterBudget/句柄, 版本
  spectrum.rs       波谱: Phillips/JONSWAP 谱系数、色散、相位推进(纯函数)
  ocean_lod.rs      clipmap/级联 LOD 决策(阈值→环带→morph 权重, 纯函数)
  swe.rs            浅水方程步进(CFL、通量、注入, 纯函数, 确定性)
  pbf.rs            PBF 密度约束调度(邻域分桶、迭代计划, 对齐 physics_core)
  flip.rs           FLIP/APIC P2G/G2P/投影调度(计划与配额, 纯函数部分)
  reconstruct.rs    表面重建选择与参数(屏幕空间/各向异性 MC/SDF 决策)
  caustics.rs       焦散雅可比投影/强度(纯函数)
  foam.rs           泡沫判据(雅可比阈值)+ 飞沫 reactive mask 发射计划
  waterline.rs      水线掩膜与水上/水下过渡决策
  budget.rs         求解/重建/位移 预算仲裁(与形变预算同构)
  transition.rs     求解器过渡带融合权重(FLIP↔SWE↔波谱, 距离加权)
```

黄金范式对齐：`cloth/mod.rs`（模块文档+手写 Vec3+`EPS`+pub mod 列表）、`hair/lod.rs`（阈值/决策纯函数+分桶+`bin_xxx`）、`virtual_geometry/bins.rs`（per-path Vec 桶、确定性输入序、越界跳过不 panic）。`lib.rs` 按字母序加 `pub mod water;`；`deformation/mod.rs` 的 `DeformationKind` 增 `Water` 变体（破坏性，需同步 schedule 测试）。

---

## 12. 性能预算与效果验收

> 沙盒无 GPU，以下为**设计目标（design target）**，非实测；上线需在真机 Metal/VK/DX12 profile 校准。

**性能预算（1080p→4K，桌面/主机档，设计目标）**：

| 项 | 预算(设计目标) | 降级路径 |
|---|---|---|
| 海洋波谱 IFFT(4 级 256²) | ≤ 0.5–1.0 ms | 降级数/降分辨率/合并级 |
| 位移网格(clipmap) | ≤ 0.5 ms | 降环带密度/减 morph |
| SWE 高度场步进 | ≤ 0.3 ms/域 | 降分辨率/降步频 |
| FLIP/PBF sim(局部 ~10⁵ 粒子) | ≤ 2–4 ms | 降粒子数/降迭代/切屏幕空间重建 |
| 表面重建(屏幕空间) | ≤ 1 ms | 降平滑迭代 |
| 反射(SSR) | ≤ 1 ms | miss 才 RT；否则探针 |
| 焦散(雅可比投影) | ≤ 0.3 ms | 关焦散或降分辨率 |
| 水下体积 | 复用共享 froxel | 降 froxel 分辨率 |

**效果验收口径**：
- 海面在近/中/远三段无平铺重复感、无 LOD 接缝跳变、掠射菲涅尔正确；
- 交互（船/角色）产生可见船首浪+尾迹，涉水涟漪连续无 pop；
- 折射随深度变色（Beer-Lambert）、掠射反射到位、SSR miss 平滑回退无黑边；
- 泡沫出现在波峰折叠（雅可比<阈值）与高速交互处，历史累积不闪烁；
- Volume 液体表面连续（无粒子颗粒感）、翻涌飞溅发射飞沫；
- NPR 模式：水色 ramp 分级、卡通高光块、手绘泡沫线、风格化焦散均可开关；
- 水线过渡（相机穿越水面）平滑，水下色偏/模糊/能见度正确；
- 四前端（PBR/NPR/自定义/混合）在同一水体可切换/混合且各享基底高级特性。

---

## 13. 可测性（CPU 可验证，沙盒内）

纯函数单测（不依赖 GPU）：
- 波谱：色散 ω(k)=√(gk) 单调性、Phillips/JONSWAP 谱非负、相位推进周期性、能量随风速单调；
- SWE：CFL 条件判定、静水保持（无扰动不产生波）、质量守恒（无源封闭域总水量不变，EPS 容差）；
- PBF：密度约束符号、迭代计划确定性、邻域分桶越界跳过不 panic；
- LOD：clipmap 环带阈值单调、morph 权重 [0,1]、过渡带权重和为 1；
- 焦散：雅可比阈值分类、强度非负；
- 泡沫：雅可比<阈值→发射、reactive mask 计划确定性；
- 预算仲裁：配额不超、优先级降序+句柄升序贪心、首 job 防饿死。

门禁（逐文件）：`rustfmt --edition 2024 <文件>` + `cargo clippy -p prism_render_architecture --all-targets -- -D warnings` + `cargo test -p prism_render_architecture`。禁 `cargo fmt --all` / `cargo clippy` 全量。

---

## 14. 落地路线图（M0–M8）

- **M0 地基（串行）**：`water/mod.rs` 契约（`WaterBody`/`WaterKind{Ocean,Surface,Volume}`/`WaterBudget`/句柄）；`lib.rs` 按字母序加 `pub mod water;`；`DeformationKind::Water`（破坏性，同步 schedule 测试）；`budget.rs` 预算仲裁。三关门禁全绿→commit。
- **M1 海洋波谱**：`spectrum.rs`（Phillips/JONSWAP/色散/相位/雅可比判据）纯函数 + 测。
- **M2 海面 LOD**：`ocean_lod.rs`（clipmap 环带/morph/级联加权）纯函数 + 测。
- **M3 浅水**：`swe.rs`（CFL/通量/注入/守恒）纯函数 + 测。
- **M4 泡沫/水线**：`foam.rs` + `waterline.rs`（判据/掩膜/reactive mask 发射计划）+ 测。
- **M5 PBF**：`pbf.rs`（密度约束调度、邻域分桶，对齐 physics_core）+ 测。
- **M6 FLIP/APIC**：`flip.rs`（P2G/G2P/投影计划）+ `reconstruct.rs`（重建选择）+ 测。
- **M7 焦散/过渡**：`caustics.rs` + `transition.rs`（求解器融合权重）+ 测。
- **M8 GPU 接线（真机）**：WESL kernel（IFFT/SWE/PBF/FLIP/重建/焦散）+ 渲染图接线 + 折射/反射/水下体积接入共享服务；真机 profile 校准预算。

**并行拆分建议**（M1–M7 写集互不重叠、只依赖 mod.rs 契约）：A=spectrum B=ocean_lod C=swe D=foam+waterline E=pbf F=flip+reconstruct G=caustics+transition。M0 与 M8 串行。

---

## 15. 风险与开放问题

- **求解器过渡接缝**：FLIP↔SWE↔波谱融合带可能出现高度/法线不连续 → `transition.rs` 距离加权 + 深度对齐，真机调参。
- **折射/反射与 OIT 交互**：水下有透明物体时的排序 → 明确水面在 OIT 中的层次，优先走深度合成而非纯 OIT。
- **FLIP 泊松解成本**：大域压力解是瓶颈 → 多重网格/降分辨率/优先 PBF；影视级留 FLIP 高配桶。
- **确定性/网络**：sim 默认表现层不参与网络裁决；需确定性时固定步长+固定迭代+确定输入序（同 physics_core）。
- **浮力回读**：刚体查询水位/水速的 CPU 同步 → GPU 侧回写 + 小批量 readback，避免逐帧全量回读。
- **沙盒验证边界**：GPU 路径无法本机验证 → 仅承诺 CPU 纯函数正确性，GPU 部分标注为未验证，真机补齐。

---

## 附录：术语
- **IFFT/FFT**：逆/快速傅里叶变换，波谱↔高度场转换。
- **Phillips/JONSWAP/PM 谱**：海浪统计能量谱模型。
- **雅可比行列式（Jacobian）**：位移场折叠判据，<阈值=波峰折叠=白沫。
- **FLIP/PIC/APIC**：粒子-网格混合流体法；APIC 用仿射矩阵保角动量。
- **PBF**：Position-Based Fluids，不可压缩 SPH 的 XPBD 化。
- **SWE**：浅水方程，高度场 2.5D 流体。
- **SPH**：光滑粒子流体动力学。
- **clipmap/geomorphing**：以相机为中心的多级网格 + 连续几何过渡。
- **choppiness**：波陡度（水平位移强度），产生尖浪。
- **Beer-Lambert**：光沿光程指数衰减，决定水色随深度变化。
- **焦散（caustics）**：折射/反射光聚焦形成的亮斑。
- **reactive mask**：驱动泡沫/飞沫粒子发射的信号掩膜。
- **waterline**：相机穿越水面时的水上/水下分割掩膜。
