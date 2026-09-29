# Prism 渲染引擎 — 水体/流体引擎子系统完整设计（v2 / wgpu + WESL）

> 状态：架构提案（Draft，允许破坏性重构）
> 定位：AAA / 次世代 水体·海洋·自由表面流体引擎，与 PBR/NPR/自定义/混合四前端正交协同，**四前端同享全部共享高级基底**（虚拟几何 / Lumen 式混合 GI / ReSTIR DI-GI / VSM / 体积 / RT 反射与焦散 / 路径追踪参考 / 时序上采样）。
> Shader：WESL（naga → WGSL/SPIR-V/Metal/DXIL）；运行时：wgpu（Metal/VK/DX12/WebGPU）
> 关联文档：`prism_rendering_architecture_zh.md`（总）、`prism_material_pipeline_design_zh.md`（§6.2 子系统注册表：水/海洋=档2 后置留槽、§1「NPR 不与 Water 同级」）、`prism_aaa_advanced_features_zh.md`（§3 PBR 高级特性、§NPR 享有基底特性映射）、`prism_particle_engine_design_zh.md`（飞沫/泡沫/气泡由 Ember 承接）、`prism_physics_design_zh.md`（§3 XPBD、§4 多求解器、§11 GPU 持久化、§12 异步）、`prism_hair_engine_design_zh.md` / `prism_cloth_engine_design_zh.md`（对称子系统范式）
> 沙盒说明：本机屏蔽 Metal（无 GPU），文档内帧预算/带宽/耗时均为**设计目标（design target），非实测**，如实标注；可验证部分限于 CPU 可计算纯函数。
> 数值路线：纯经典数值（FFT/FLIP/APIC/PBF/SWE/波谱/光子焦散），不含任何 AI/ML 内容。
> **v2 变更**：新增 §5b 共享高级基底接入矩阵（PBR/NPR/混合同享）、§6b 破碎浪与飞沫喷发、§7b 两向耦合、§9b RT 焦散与光谱色散、§9c 高级水下体积、§9d 动态泡沫平流与持久化、§9e 湿润/岸线/天气耦合系统；深化 §12 性能与效果验收；扩充 §0 产品对标与 §5 四前端分叉。

---

## 0. 定位：这是什么级别的水体引擎

是**完整的水体/流体引擎**（海洋波谱 + 自由表面流体 sim + 表面重建 + 水体特殊渲染 + 湿润/岸线/天气耦合），不是"给平面贴一张法线流动贴图"的着色贴皮。对标业界成熟产品，仅在**算法层**借鉴，不复制其代码：

| 产品 | 借鉴点（算法层，v2 细化） |
|---|---|
| **UE5 Water（Water System / Waterline / Single Layer Water）** | 河流/湖泊/海洋样条水体、Gerstner 波叠加、单层水着色模型、waterline 掩膜、水体深度/边缘泡沫、可行走浅水 |
| **UE5 Niagara Fluids** | GPU 2D/3D 网格流体、GPU-driven 全流程、与粒子系统耦合模板 |
| **NVIDIA WaveWorks / Crest Ocean** | LOD 级联位移贴图、动态波注入（dynamic wave sim）、水体深度缓存、掠射 SSS、浅水混合、屏幕空间水下 |
| **Tessendorf FFT 海洋（影视/游戏通行）** | Phillips/JONSWAP 谱、IFFT 位移+法线+雅可比、多方向多尺度级联 |
| **God of War Ragnarök / Horizon 系** | 主机预算下高质海面 + 交互浪、湿润岸线、掠射透光浪尖、艺术可控波形 |
| **Sea of Thieves / AC / RDR2 系船海** | 艺术可控波、船体交互浪与尾迹、破碎浪白沫、河流流向可视化 |
| **Ghost of Tsushima / 塞尔达 / 原神（NPR 侧）** | 风格化水色 ramp、卡通高光块、手绘浅滩白边、程序化涟漪、水墨扩散 |
| **Position-Based Fluids（PBF, Macklin 2013）** | 不可压缩 SPH 的 XPBD 化密度约束、GPU 邻域并行、涡量约束、交互液体 |
| **FLIP / APIC（Houdini / Zhu-Bridson / Jiang et al.）** | 粒子-网格混合自由表面、低耗散、APIC 保角动量、大规模液体 |
| **Shallow Water Equations（SWE）** | 高度场浅水波、河流/洪水/交互涟漪的廉价 2.5D 求解 |
| **屏幕空间流体（Screen-Space Fluids, van der Laan）** | 粒子→深度→双边平滑→法线重建的实时液体表面 |
| **影视 RT 焦散 / 光子映射** | 折射焦散的光子/RT 投影，高配近景真实光斑 |

**判据（承接材质 §6.2）**：一等子系统必须拥有**自己的几何 + 自己的 sim + 特殊渲染**。水体三者全占：
- **几何**：动态位移海面级联网格（clipmap/投影网格）、SWE 高度场、FLIP/PBF 表面重建网格；
- **sim**：不可压缩自由表面（FLIP/APIC/PBF）、浅水方程、海洋波谱 IFFT、动态波注入、破碎浪、动态泡沫平流；
- **特殊渲染**：折射 + SSR/RT 反射 + RT/光子焦散 + 泡沫/飞沫 + 水下体积散射 + 光谱色散 + 水线掩膜 + 湿润岸线。

因此水体是**一等子系统**（材质 §6.2 归「档2 后置留槽」——真子系统但重且非通用，wgpu 竖切 MVP 暂不接线，随基底成熟后跟进）。**关键立场（承接材质 §1）**：**NPR 不与 Water 同级**——Water 是 closure/子系统（有几何+sim），NPR 是 illumination（风格轴）；水面既可 PBR 也可 NPR，两者正交，且**NPR 水面同享全部共享高级基底**（§5b）。

---

## 1. 与整体架构的关系（子系统内部同构范式）

承接毛发/布料/粒子同构范式：**共享基底 + PBR/NPR 分叉响应**。

```
共享基底(PBR/NPR/自定义/混合 都吃):
  水体几何(海面级联位移网格 / SWE高度场 / FLIP-PBF表面重建, 连续LOD 禁硬切换)
  + 统一流体 sim(对齐 physics_core 的 XPBD/网格投影原语)
  + 折射(共享场景颜色/深度) + 反射(共享 HZB SSR → RT → 探针 三级)
  + 共享高级基底(§5b): 虚拟几何 vis-buffer / Lumen式混合GI / ReSTIR DI-GI / VSM / froxel体积 / 路径追踪参考 / 时序上采样
  + 焦散(光空间雅可比 → RT/光子高配) + 水下体积(接入共享 froxel)
  + 水线掩膜(共享 prepass) + motion vector(共享时序)
  + 泡沫/飞沫/气泡(委托 Ember 粒子子系统, 不自建粒子池)

分叉响应(C类, 真分家 —— 只在"光照响应"分家, 见 §5):
  PBR:  物理菲涅尔 + 环境/SSR/RT 反射 + Beer-Lambert 吸收/散射 + 掠射次表面透射 + 微表面波法线 + 光谱色散
  NPR:  ramp 量化水色 + 卡通高光块 + 手绘泡沫线/浪花描边 + 风格化焦散(网点/线条) + 水墨扩散 + 流动线
  自定义/混合: 同一网格多材质槽, 逐区/逐层混合 PBR↔NPR (§5.4)

fallback:  近景 FLIP/PBF 表面重建, 远景海洋级联位移; 极远高度场/静态法线;
           RT 反射/焦散里流体用 SDF/包围代理或降级 SSR/雅可比投影
```

子系统**只产几何/位移/表面/掩膜数据**；着色走共享 closure、透明/折射走共享服务、约束求解对齐 `prism_physics_core`——**不重写这些**。

与其它子系统的边界（去重）：
- **飞沫/泡沫/气泡/水花粒子** → 委托 **Ember 粒子引擎**（已有 GPU 池/排序/OIT/着色四等公民）；水体只发 spawn 事件 + reactive mask。
- **气态体素流体（烟/火/爆炸）** → 归 **Ember 网格流体**；**本引擎只管液态/自由表面/海洋**。
- **约束/积分原语** → 对齐 `prism_physics_core` 统一 XPBD 与网格投影。
- **破碎/刚体浮力反作用** → `physics_core`；水体提供高度/速度场查询接口（§7b 两向耦合）。

---

## 2. 数据与资产模型

三类水体（Body），共享统一 `WaterBody` 抽象：

| 资产类型 | 几何形态 | 求解器 | 典型用途 |
|---|---|---|---|
| **Ocean（海洋）** | 无限级联位移网格 | 波谱 IFFT + 动态波注入 + 破碎浪 | 开放海、大湖 |
| **Surface（水面/河湖）** | 样条/多边形 2.5D 高度场 | SWE + Gerstner 叠加 | 河流、湖泊、池塘、洪水 |
| **Volume（体积液体）** | 3D 域粒子 + 重建网格 | FLIP/APIC 或 PBF | 溅水、翻涌、管道、局部大水 |

`WaterBodyAsset` 字段：几何（域包围盒/样条/分辨率/粒子上限）、物性（密度/粘度/表面张力/静水位/消光 RGB/散射/IOR≈1.33/色散系数）、波形（谱类型/风速风向/级联数与尺度/陡度/方向扩散/破碎阈值）、交互（可注入源/边界条件）、着色（材质句柄 PBR/NPR/自定义/混合、泡沫阈值与持久化、焦散强度、能见度、浅水与湿润渐变曲线）。数据驱动（RON/序列化），编译为求解器参数 + WESL specialization key。

---

## 3. 管线阶段（端到端，每帧全链路，GPU-first 零回读）

```
[Extract]   活跃 WaterBody + 交互源(船/角色/冲击/雨) → GPU uniform/instance
[Prepare]   1. 波谱更新: 相位推进 → IFFT(位移/法线/雅可比) 级联贴图
            2. 动态波注入: 交互源 → 涟漪写入动态贴图(Crest 式)
            3. 破碎浪检测: 雅可比/陡度 → 白沫源 + 飞沫喷发事件(§6b)
            4. 流体 sim(Volume): FLIP/APIC/PBF compute 步进(§7)
            5. 浅水 sim(Surface): SWE 高度场步进(§6.3)
            6. 动态泡沫平流(§9d): 泡沫密度场 advect + 衰减 + 持久化
            7. 表面重建(Volume): 粒子→密度场→等值面(§7.3)
[Queue]     8. LOD + clipmap tile 决策 + 视锥/遮挡剔除(HZB)
            9. 位移网格生成/更新(indirect draw, 接虚拟几何 §5b)
           10. Prepass: 水体深度/水线掩膜/motion vector/material id
           11. Gbuffer/Forward: 水面 closure 着色(PBR/NPR/混合)
           12. 折射: 采样共享场景颜色(prepass 隔离) + 光谱色散(§9b)
           13. 反射: SSR(HZB) → miss RT(ray_scene) → 探针
           14. GI/光照: 消费 Lumen式混合GI + ReSTIR + VSM(§5b)
           15. 焦散: 光空间雅可比投影 or RT/光子焦散(§9b) → 光照/贴花注入
           16. 水下体积: froxel 单/多次散射 + god ray + 能见度 + 色偏(§9c)
           17. 湿润/岸线: 湿润掩膜衰减 + 反照率变暗 + 积水(§9e)
           18. 泡沫/飞沫: reactive mask → Ember 发射粒子
[Post]     19. 水下后处理: 色偏/模糊/色差/焦散叠加, 水线过渡
           20. 时序: motion vector → TAA/上采样(§5b), 泡沫历史累积
```

差异化：波谱/SWE/FLIP/PBF 为可独立启用的求解桶，一个场景可并存（远海波谱、近岸 SWE、局部 FLIP），在深度合成阶段统一（§8 过渡）。

---

## 4. 求解器矩阵（多桶）

| 求解器 | 维度 | 不可压缩 | 强项 | 代价 | 落点 |
|---|---|---|---|---|---|
| **波谱 IFFT（Tessendorf）** | 2.5D | 动画 | 无边海面、真实统计谱、极高性价比 | O(N²logN)/级 | Ocean 基线 |
| **Gerstner 叠加** | 2.5D | 动画 | 艺术可控、少波数、可解析求导 | 波数多则贵 | Ocean/Surface 补充 |
| **SWE 浅水方程** | 2.5D | 近似 | 河流/洪水/交互涟漪、廉价 | 不能翻卷 | Surface |
| **PBF** | 3D 粒子 | 是(XPBD 密度约束) | 与 XPBD 统一、交互强、稳定 | 邻域搜索/迭代 | Volume 交互 |
| **FLIP/APIC** | 3D 粒子+网格 | 是(网格投影) | 大规模低耗散、翻涌飞溅、APIC 保角动量 | 泊松解 | Volume 影视级 |

**取舍**：能高度场不 3D 粒子，能波谱不求解。远海波谱、近岸 SWE+动态波注入、局部翻涌 FLIP/PBF，深度合成无缝拼接。迭代/投影原语对齐 `prism_physics_core`。

---

## 5. PBR / NPR / 自定义 / 混合 分叉响应（只在光照响应分家）

### 5.1 PBR 水面（物理）
菲涅尔（Schlick，F0≈0.02，掠射高反）；反射（SSR→RT→探针三级）；折射（IOR+深度屏幕空间偏移 + 光谱色散 §9b）；吸收/散射（Beer-Lambert 光程指数衰减 + 单次散射）；微表面（波谱法线+细节法线+雅可比泡沫遮罩，GGX）；掠射次表面透射（浪尖背光透光）。

### 5.2 NPR 水面（风格化，illumination 轴）
ramp 量化水色（塞尔达/卡通）；卡通高光块（SDF/阈值化，非连续 GGX）；手绘泡沫（等值线/边缘检测描边，塞尔达浅滩白边）；风格化焦散（halftone 网点/线条）；水墨/宣纸扩散（Okami，复用材质 NPR 风格库）；沿流速场手绘流动线。**关键：NPR 水面同享 §5b 全部共享高级基底**，差异仅在最终光照解读。

### 5.3 自定义
über-closure hook：作者注入自定义 BRDF 瓣/水色映射/泡沫合成，编译进 WESL specialization，不改内核。

### 5.4 混合（管线级/材质级）
同一水体多材质槽（主体 PBR、岸边 NPR、特定区自定义）；逐层/逐区掩膜（深度/距岸/区域贴图）在 PBR↔NPR 光照响应间插值；正交叠加（PBR 折射反射基底 + NPR 描边/网点风格层，Arcane 式 3D 上叠手绘）。**四者皆一等公民**。

---

## 5b. 共享高级基底接入矩阵（PBR/NPR/混合同享，v2 核心）

承接 `prism_aaa_advanced_features_zh.md`：以下均为**共享基底服务**，水面（无论 PBR 还是 NPR）**只消费不重写**，NPR 与 PBR 同等享有。

| 高级基底 | 归属 crate | 水体如何消费 | PBR 侧 | NPR 侧 |
|---|---|---|---|---|
| **虚拟几何 vis-buffer** | `virtual_geometry/` | 海面级联位移网格接 cluster DAG LOD/软光栅；material id 供描边 | 微三角连续 LOD | material id 边界白送浪线描边 |
| **Lumen 式混合 GI** | `lighting/` | 水面/水下收 GI（SDF/mesh-card 软 RT + 硬 RT 混合 + surface cache + 屏幕探针 + DDGI 兜底） | 物理间接光 | ramp 量化 GI 做风格化底光 |
| **ReSTIR DI/GI** | `lighting/` | 同批光源储层时空重采样 + ReBLUR/ReLAX 去噪 | 多光源直/间接积分 | 同储层，ramp 解读 |
| **VSM 虚拟阴影** | `virtual_shadow/` | 水面阴影/水下阴影消费虚拟页 clipmap 深度 | 物理软阴影 | 阈值硬阴影+染色 |
| **froxel 体积** | `lighting/` 体积 | 水下散射/god ray/雾（§9c） | 物理单/多次散射 | 风格化能见度分层 |
| **RT 反射/路径追踪参考** | `ray_scene/` | SSR miss 回退 RT；路径追踪仅离线对拍校准 | 硬件 BVH 反射 | RT 反射叠风格层 |
| **RT/光子焦散** | `ray_scene/` + 光照注入 | 折射焦散高配（§9b） | 物理光斑 | 网点/线条焦散 |
| **时序上采样** | `temporal_upscale/` `motion/` `history/` | 位移网格顶点速度 + 流速场 → motion vector；泡沫历史累积 | TAA/上采样 | 同管线，稳定描边 |
| **有界 slab 多瓣** | `material/` `abi/` | 水面清漆/薄油膜/浮冰分层用有界 slab（封顶防 variant 爆炸） | 分层菲涅尔 | 分层风格叠加 |

**结论（回应"NPR 是否缺少虚拟几何/GI/ReSTIR/VSM/体积/上采样/RT/路径追踪"）：都不缺。** 这些是共享基底，NPR 与 PBR 同吃，差异只在"光照响应"（§5）。

---

## 6. 海洋波谱与浅水（Ocean / Surface）

### 6.1 波谱模型
谱：Phillips / JONSWAP（风浪成长）/ Pierson-Moskowitz。初始 H0(k) = 谱密度 × 高斯随机相位（一次性）。演化 H(k,t) = H0(k)e^{iω t} + conj(H0(−k))e^{−iω t}，色散 ω(k)=√(g·k)（深水）。IFFT → 高度 + x/z 位移（choppy 尖浪）+ 法线（斜率）+ 雅可比（<阈值=波峰折叠=白沫）。

### 6.2 级联（cascade）
多级物理尺度（如 4 级：大/中/细浪/涟漪）叠加破平铺重复；每级独立分辨率与世界尺度，距离加权，远处降级。

### 6.3 SWE 浅水方程（Surface）
高度 h + 通量 (hu,hv)，显式/半隐式步进，CFL 稳定；交互涟漪外力/边界注入（角色/雨滴/船）+ 阻尼；近岸浅水项与波谱位移在高度合成阶段相加。

### 6.4 动态波注入（Crest 式）
交互源（船体轨迹/涉水/爆炸）写入随相机移动的动态位移/法线贴图，与波谱级联叠加，产生船首浪/尾迹/局部扰动，无需全域 3D sim。

---

## 6b. 破碎浪与飞沫喷发（v2 新增）

- **破碎判据**：波陡度（位移梯度）+ 雅可比 + 局部曲率超阈 → 标记破碎带；
- **白沫源**：破碎带写入泡沫密度场（供 §9d 平流），并按强度触发 Ember 飞沫/雾滴发射；
- **浪尖喷发**：高曲率浪尖发射弧线飞沫粒子（初速沿浪面切向+法向），委托 Ember；
- **岸拍浪**：浅水深度<阈 + 波高>阈 → 岸线白沫带 + 回流泡沫（与 §9e 湿润联动）。

---

## 7. 自由表面流体（Volume：FLIP/APIC/PBF）

### 7.1 PBF（默认交互液体）
SPH 邻域 → 密度约束 C_i=ρ_i/ρ0−1，XPBD 迭代投影（对齐 physics_core）；人工压力（防聚簇）+ 涡量约束（补涡旋）+ XSPH 粘度；GPU 邻域空间哈希/网格分桶，确定性输入序。

### 7.2 FLIP/APIC（大规模/影视级）
P2G → 压力泊松解（不可压缩投影）→ G2P；APIC 仿射速度场保角动量抑制噪声/耗散；压力解 Jacobi/CG/多重网格（GPU 并行），MAC 网格。

### 7.3 表面重建
粒子密度场 → 等值面，三路线：屏幕空间流体（默认实时，粒子 splat 深度→双边/窄带平滑→法线重建，van der Laan）；各向异性 Marching Cubes（离线/近景高配，Yu-Turk 各向异性核）；narrow-band SDF+MC（质量/带宽折中）。

---

## 7b. 两向耦合（Two-Way Coupling，v2 新增）

- **流体→刚体**：水体导出高度/速度/压力场查询接口，`physics_core` 计算浮力（阿基米德按浸没体积）+ 拖曳 + 附加质量；
- **刚体→流体**：刚体浸没体积/速度作为源写回（SWE 高度扰动 / FLIP 边界速度 / 动态波注入），产生船首浪、落水溅射；
- **调度**：耦合在 sim 步内做定点迭代（sub-step），避免穿插抖动；查询走 GPU 侧回写 + 小批量 readback，禁逐帧全量回读。

---

## 8. LOD 与大世界（禁 pop 硬切换）

海面 clipmap/投影网格（相机同心环，随距离降密度，geomorphing 连续 morph）；波谱级联 LOD（远处减叠加级 + mip 法线）；求解器过渡带（近 FLIP/PBF、中 SWE、远波谱，距离加权融合高度/法线避接缝）；瓦片流（超大水域按瓦片驻留位移/深度缓存，接 `paging`/`texture_streaming`）；RT 降级（流体用 SDF/包围代理或排除，回退 SSR）。海面网格接虚拟几何 cluster LOD（§5b）。

---

## 9. 特殊渲染服务

| 服务 | 归属 | 说明 |
|---|---|---|
| 折射 | 共享场景颜色/深度 | prepass 隔离水前场景，屏幕空间偏移 + 光谱色散(§9b) |
| 反射 | 共享 SSR(HZB)+RT(ray_scene)+探针 | 三级回退 |
| 焦散 | 水体注入 + 共享光照/RT | 雅可比投影(廉价) or RT/光子(高配)(§9b) |
| 水下体积 | 共享 froxel | 单/多次散射、god ray、能见度、色偏(§9c) |
| 水线掩膜 | 共享 prepass | 相机穿越水面上/下分割 + 过渡 |
| 泡沫/飞沫/气泡 | 委托 Ember 粒子 | 水体发 reactive mask + spawn |
| motion vector | 共享时序 | 位移顶点速度+流速场 → TAA/上采样 |
| 湿润/岸线 | 水体专属 + 共享贴花 | 湿润掩膜/反照率变暗/积水(§9e) |

---

## 9b. RT 焦散与光谱色散（v2 新增，高配）

- **廉价焦散**：光空间用位移雅可比投影亮度到接收面（贴花/光照贴图），实时；
- **RT 焦散**：从光源沿折射方向 RT 求交接收面聚焦亮斑（近景高配，接 `ray_scene`）；
- **光子焦散**：光子发射→折射→接收面 splat→密度估计（离线/影视对拍）；
- **光谱色散**：折射按波长分 RGB 三条略不同 IOR 采样场景颜色，产生水下彩边（掠射/厚水更明显），可开关控成本。

---

## 9c. 高级水下体积（v2 新增）

- **多次散射**：froxel 中除单次散射外加各向同性多次散射近似项（浑浊水更真实），相位函数 Henyey-Greenstein；
- **god ray（体积光轴）**：光源穿水面被波法线调制 → froxel 内散射累积产生水下光柱；
- **能见度/浑浊度**：消光系数 RGB 按 Beer-Lambert，深度雾化 + 悬浮微粒噪声；
- **色偏**：随深度红→橙→绿→蓝衰减（红光先被吸收），物理正确的水下变色；
- **NPR 水下**：能见度分层 ramp、风格化光柱（线条/网点），同享 froxel 数据。

---

## 9d. 动态泡沫平流与持久化（v2 新增）

- **泡沫密度场**：破碎带/高速交互/雅可比折叠处注入泡沫密度；
- **平流（advection）**：泡沫密度沿表面流速场平流（semi-Lagrangian），随时间衰减，产生流动尾迹泡沫（非静态贴图）；
- **持久化**：岸边/低速区泡沫衰减更慢（回流白沫堆积），历史缓冲累积；
- **渲染**：泡沫作为水面 albedo/粗糙度调制层（PBR）或手绘泡沫线阈值（NPR），并驱动 Ember 飞沫发射密度。

---

## 9e. 湿润 / 岸线 / 天气耦合系统（v2 新增）

- **湿润掩膜**：水位以上一段高度按毛细上升生成湿润带，反照率变暗 + 粗糙度降低 + 法线细节增强（湿沙/湿石），退潮/水位下降后按干燥速率衰减；
- **岸线过渡**：浅水深度渐变（水深→颜色/透明/泡沫），可行走浅水 waterline；
- **积水/水坑**：地形凹陷积水（高度场阈值），接入 SWE 微涟漪；
- **天气耦合**：雨滴 → SWE/动态贴图注入涟漪 + 表面湿润扩散 + 溅射飞沫（Ember）；风 → 波谱风速风向驱动。

---

## 10. 与共享基底的接线（复用，不重写）

着色 closure/OIT/透明：材质系统（`transparency`、über-closure、有界 slab）；光照/体积/阴影/GI：`lighting`（聚类+ReSTIR+Lumen 式 GI）、froxel 体积、`virtual_shadow`（VSM）、`ray_scene`（RT 反射/焦散/路径追踪参考）；时序：`temporal_upscale`/`motion`/`history`；GPU-driven：`gpu_scene`/indirect/`work_graph`/HZB/`virtual_geometry`；资源：`virtual_resource`/`descriptor_heap`/`memory`/`paging`/`texture_streaming`；求解原语：`prism_physics_core`。

**水体新增的最小内核**：波谱 IFFT、SWE 步进、FLIP/PBF 步进与投影调度、表面重建、级联/clipmap LOD、破碎浪检测、动态泡沫平流、焦散雅可比投影、湿润/岸线掩膜、水线掩膜、reactive mask 发射、两向耦合调度——其余全部消费共享服务。

---

## 11. Crate 拆分与落地形态

落点 `pkg/prism_render_architecture/src/water/`（照抄 cloth/hair/particle 范式：零依赖、`#![forbid(unsafe_code)]`、`extern crate alloc;`、手写向量数学、只允许 sqrt）。CPU 可验证纯函数与调度先行；GPU WESL 先落契约签名+脚手架，不作本机编译验证目标。

```
water/
  mod.rs            契约: WaterBody/WaterKind{Ocean,Surface,Volume}/WaterBudget/句柄, 版本
  spectrum.rs       波谱: Phillips/JONSWAP/PM 谱、色散、相位推进(纯函数)
  ocean_lod.rs      clipmap/级联 LOD 决策(阈值→环带→morph, 纯函数)
  swe.rs            浅水方程步进(CFL/通量/注入/守恒, 确定性)
  breaking.rs       破碎浪检测(陡度/雅可比/曲率阈)+ 飞沫喷发计划
  pbf.rs            PBF 密度约束调度(邻域分桶/迭代计划, 对齐 physics_core)
  flip.rs           FLIP/APIC P2G/G2P/投影调度(计划与配额)
  reconstruct.rs    表面重建选择与参数(屏幕空间/各向异性MC/SDF 决策)
  foam.rs           泡沫密度平流/衰减/持久化(纯函数, semi-Lagrangian)
  caustics.rs       焦散雅可比投影/RT-光子强度(纯函数)
  underwater.rs     水下体积参数(多次散射/godray/色偏/能见度 决策)
  wetness.rs        湿润/岸线/积水/天气耦合 掩膜决策
  waterline.rs      水线掩膜与水上/水下过渡
  coupling.rs       两向耦合调度(浮力查询接口/源写回/sub-step)
  dispersion.rs     光谱色散(RGB 分波长 IOR 偏移参数)
  budget.rs         求解/重建/位移 预算仲裁(与形变预算同构)
  transition.rs     求解器过渡带融合权重(FLIP↔SWE↔波谱)
```

黄金范式：`cloth/mod.rs`（模块文档+手写 Vec3+`EPS`+pub mod 列表）、`hair/lod.rs`（阈值/决策纯函数+分桶+`bin_xxx`）、`virtual_geometry/bins.rs`（per-path Vec 桶、确定性输入序、越界跳过不 panic）。`lib.rs` 按字母序加 `pub mod water;`；`deformation/mod.rs` 的 `DeformationKind` 增 `Water` 变体（破坏性，同步 schedule 测试）。

---

## 12. 性能预算与效果验收

> 沙盒无 GPU，以下为**设计目标（design target），非实测**；上线需真机 Metal/VK/DX12 profile 校准。

**性能预算（1080p→4K，桌面/主机档，设计目标）**：

| 项 | 预算(设计目标) | 降级路径 |
|---|---|---|
| 海洋波谱 IFFT(4 级 256²) | ≤ 0.5–1.0 ms | 降级数/降分辨率/合并级 |
| 位移网格(clipmap + 虚拟几何) | ≤ 0.5 ms | 降环带密度/减 morph |
| SWE 高度场步进 | ≤ 0.3 ms/域 | 降分辨率/降步频 |
| FLIP/PBF sim(局部 ~10⁵ 粒子) | ≤ 2–4 ms | 降粒子数/迭代/切屏幕空间重建 |
| 表面重建(屏幕空间) | ≤ 1 ms | 降平滑迭代 |
| 反射 SSR | ≤ 1 ms | miss 才 RT；否则探针 |
| RT 焦散(高配) | ≤ 1–2 ms | 回退雅可比投影 |
| 动态泡沫平流 | ≤ 0.3 ms | 降分辨率/关持久化 |
| 水下体积(多次散射) | 复用共享 froxel | 降 froxel 分辨率/关多次散射 |
| 湿润/岸线掩膜 | ≤ 0.3 ms | 静态湿润带 |

**效果验收口径**：
- 海面近/中/远无平铺重复、无 LOD 接缝跳变、掠射菲涅尔正确、破碎浪出白沫；
- 交互（船/角色/落水）产生船首浪+尾迹+溅射，两向耦合浮力正确，涟漪连续无 pop；
- 折射随深度变色（Beer-Lambert）、光谱色散可见彩边、掠射反射到位、SSR miss 平滑回退无黑边；
- 焦散出现在折射聚焦处（雅可比/RT），水下 god ray 与色偏物理正确、多次散射浑浊感到位；
- 泡沫沿流速平流形成尾迹、岸边回流持久化、历史累积不闪烁；
- Volume 液体表面连续（无颗粒感）、翻涌飞溅发射飞沫；
- 湿润带在水位变化后正确变暗/干燥，雨滴涟漪 + 积水正确；
- NPR 模式：水色 ramp、卡通高光块、手绘泡沫线、风格化焦散/水下均可开关，且**同享虚拟几何/GI/ReSTIR/VSM/体积/上采样/RT**（§5b）；
- 水线过渡（相机穿越）平滑，水下色偏/模糊/能见度正确；
- 四前端（PBR/NPR/自定义/混合）在同一水体可切换/混合且各享基底高级特性。

---

## 13. 可测性（CPU 可验证，沙盒内）

纯函数单测：波谱（色散单调、谱非负、相位周期、能量随风速单调）；SWE（CFL 判定、静水保持、质量守恒 EPS 容差）；PBF（密度约束符号、迭代计划确定性、邻域越界跳过不 panic）；破碎（陡度/雅可比阈值分类确定性）；泡沫平流（衰减单调、密度非负、持久化区衰减更慢）；焦散（雅可比阈值分类、强度非负）；LOD（clipmap 环带单调、morph∈[0,1]、过渡带权重和=1）；色散（RGB 偏移有序）；湿润（毛细高度单调、干燥衰减单调）；预算（配额不超、优先级降序+句柄升序贪心、首 job 防饿死）。

门禁（逐文件）：`rustfmt --edition 2024 <文件>` + `cargo clippy -p prism_render_architecture --all-targets -- -D warnings` + `cargo test -p prism_render_architecture`。禁 `cargo fmt --all` / 全量 clippy。

---

## 14. 落地路线图（M0–M9）

- **M0 地基（串行）**：`water/mod.rs` 契约 + `lib.rs` 加 `pub mod water;` + `DeformationKind::Water`（破坏性，同步 schedule 测试）+ `budget.rs`。三关门禁全绿→commit。
- **M1 海洋波谱**：`spectrum.rs`（Phillips/JONSWAP/PM/色散/相位/雅可比）+ 测。
- **M2 海面 LOD**：`ocean_lod.rs`（clipmap/morph/级联加权）+ 测。
- **M3 浅水**：`swe.rs`（CFL/通量/注入/守恒）+ 测。
- **M4 破碎/泡沫**：`breaking.rs` + `foam.rs`（平流/衰减/持久化）+ 测。
- **M5 水线/湿润**：`waterline.rs` + `wetness.rs`（掩膜/毛细/干燥/天气）+ 测。
- **M6 PBF**：`pbf.rs`（密度约束调度、邻域分桶，对齐 physics_core）+ 测。
- **M7 FLIP/APIC + 重建**：`flip.rs` + `reconstruct.rs` + 测。
- **M8 焦散/色散/水下/过渡/耦合**：`caustics.rs` + `dispersion.rs` + `underwater.rs` + `transition.rs` + `coupling.rs` + 测。
- **M9 GPU 接线（真机）**：WESL kernel（IFFT/SWE/PBF/FLIP/重建/焦散/泡沫平流）+ 渲染图接线 + 折射/反射/GI/体积/RT/时序接入共享服务；真机 profile 校准。

**并行拆分建议**（M1–M8 写集互不重叠、只依赖 mod.rs 契约）：A=spectrum B=ocean_lod C=swe D=breaking+foam E=waterline+wetness F=pbf G=flip+reconstruct H=caustics+dispersion+underwater+transition+coupling。M0 与 M9 串行。

---

## 15. 风险与开放问题

- **求解器过渡接缝**：FLIP↔SWE↔波谱融合带高度/法线不连续 → `transition.rs` 距离加权 + 深度对齐，真机调参。
- **折射/反射与 OIT 交互**：水下透明物体排序 → 明确水面在 OIT 层次，优先深度合成而非纯 OIT。
- **FLIP 泊松解成本**：大域压力解瓶颈 → 多重网格/降分辨率/优先 PBF；影视级留 FLIP 高配桶。
- **RT 焦散/光谱色散成本**：高配才开，默认雅可比投影 + 单 IOR，按平台档分级。
- **两向耦合稳定性**：sub-step 定点迭代避免抖动；回读用 GPU 回写 + 小批量。
- **确定性/网络**：sim 默认表现层不参与裁决；需确定性时固定步长+迭代+输入序（同 physics_core）。
- **沙盒验证边界**：GPU 路径无法本机验证 → 仅承诺 CPU 纯函数正确性，GPU 部分标注未验证，真机补齐。

---

## 附录：术语
- **IFFT/FFT**：逆/快速傅里叶变换。**Phillips/JONSWAP/PM 谱**：海浪统计能量谱。
- **雅可比行列式**：位移折叠判据，<阈值=波峰折叠=白沫。
- **FLIP/PIC/APIC**：粒子-网格混合流体；APIC 用仿射矩阵保角动量。**PBF**：不可压缩 SPH 的 XPBD 化。**SWE**：浅水方程。**SPH**：光滑粒子流体动力学。
- **clipmap/geomorphing**：相机中心多级网格 + 连续几何过渡。**choppiness**：波陡度。
- **Beer-Lambert**：光沿光程指数衰减，决定水色随深度。**焦散**：折射/反射光聚焦亮斑。
- **ReSTIR**：储层时空重采样多光源采样。**VSM**：虚拟阴影贴图。**DDGI**：动态漫反射 GI 探针。**froxel**：视锥体素（体积光照）。
- **Henyey-Greenstein**：各向异性散射相位函数。**semi-Lagrangian**：半拉格朗日平流。
- **两向耦合**：流体与刚体互相施力。**reactive mask**：驱动泡沫/飞沫发射的信号掩膜。**waterline**：相机穿越水面的水上/水下分割掩膜。
