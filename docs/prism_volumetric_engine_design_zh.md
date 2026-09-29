# Prism 渲染引擎 — 体积云/大气体积引擎子系统完整设计（v1 / wgpu + WESL）

> 状态：架构提案（Draft，允许破坏性重构）
> 定位：AAA / 次世代 体积云·大气体积引擎（程序化建模 + 天气演化 sim + raymarching + 多次散射），与 PBR/NPR/自定义/混合四前端正交协同，四前端同享共享高级基底（Lumen 式混合 GI / ReSTIR / VSM 云影 / froxel 体积 / RT 参考 / 时序上采样 / 大气散射服务）。
> Shader：WESL（naga → WGSL/SPIR-V/Metal/DXIL）；运行时：wgpu（Metal/VK/DX12/WebGPU）
> 关联文档：`prism_rendering_architecture_zh.md`（总）、`prism_material_pipeline_design_zh.md`（§6.2 子系统注册表：体积=档1 一等子系统；§6.2 档3「大气=光照数据服务」——大气散射是服务，体积云是子系统）、`prism_aaa_advanced_features_zh.md`（共享高级基底）、`prism_particle_engine_design_zh.md`（气态体素流体 烟/火/爆炸归粒子，本引擎不碰）、`prism_water_engine_design_zh.md` / `prism_hair_engine_design_zh.md` / `prism_cloth_engine_design_zh.md`（对称子系统范式）、`prism_physics_design_zh.md`
> 沙盒说明：本机屏蔽 Metal（无 GPU），文档内帧预算/带宽/耗时均为**设计目标（design target），非实测**，如实标注；可验证部分限于 CPU 可计算纯函数。
> 数值路线：纯经典数值（Perlin-Worley 噪声/raymarch/多次散射近似/大气 LUT），不含任何 AI/ML。

---

## 0. 定位：这是什么级别的体积引擎

是**完整的体积云/大气体积引擎**（程序化云建模 + 天气系统演化 + 光线步进 + 多次散射光照），不是"贴一张天空盒 + skybox 云贴图"的着色贴皮。对标业界成熟产品，仅在**算法层**借鉴，不复制其代码：

| 产品 | 借鉴点（算法层） |
|---|---|
| **UE5 Volumetric Clouds + Sky Atmosphere** | 体积云组件、weather map 驱动、raymarch + cone 阴影采样、与物理天空/aerial perspective 衔接、云影投射 |
| **Horizon Zero Dawn「Nubis」（Schneider/Guerrilla）** | Perlin-Worley 噪声建模、coverage/type/height 梯度、detail erosion、powder 效应、能量守恒 raymarch、多重散射近似 |
| **Frostbite 物理大气 + 体积云（Hillaire）** | 预积分 transmittance/multi-scatter LUT、aerial perspective froxel、能量守恒天空 |
| **Decima / RDR2 天气系统** | 动态天气演化、云层随时间平流、降水/风暴耦合、体积光柱 |
| **影视 Wrenninge / Fong「Production Volume Rendering」** | 多倍频散射（octave scattering）、各向异性相位、体积能量守恒离线对拍 |
| **Guerrilla / Ubisoft cloudscape LOD** | 低分辨率 buffer + 时序重投影上采样（checkerboard/quarter-res）、远景 imposter |
| **吉卜力 / 卡通动画（NPR 侧）** | 分层描边云、ramp 量化受光、程序化风格化云朵形状、手绘高光块 |

**判据（承接材质 §6.2）**：一等子系统必须拥有**自己的几何 + 自己的 sim + 特殊渲染**。体积云三者全占：
- **几何**：3D 密度场 / weather map / 云域包围体（程序化体积，非三角网格）；
- **sim**：天气系统演化、云层平流（风场驱动）、随时间生消、降水/风暴耦合；
- **特殊渲染**：raymarch 光线步进、多次散射（powder/octave scattering）、体积自阴影、god ray、aerial perspective 衔接。

因此体积云是**一等子系统**（材质 §6.2 档1：粒子·毛发·体积·布料sim）。**关键边界**：
- **气态体素流体（烟/火/爆炸）已归 Ember 粒子引擎（网格流体）**，本引擎**不碰**；本引擎专做**天空尺度的体积云与大气体积**。
- **大气散射/雾是共享基底服务**（材质 §6.2 档3「大气=光照数据服务」，不是子系统）；本引擎**消费**大气 transmittance/aerial-perspective LUT，不重写。
- froxel 体积光照是共享服务，体积云**接入不重写**。
- **NPR 不与体积云同级**——体积云是 closure/子系统（有几何+sim），NPR 是 illumination（风格轴）；云既可 PBR 也可 NPR，正交。

---

## 1. 与整体架构的关系（子系统内部同构范式）

承接毛发/布料/粒子/水同构范式：**共享基底 + PBR/NPR 分叉响应**。

```
共享基底(PBR/NPR/自定义/混合 都吃):
  云几何(3D 密度场 / weather map / 分层云域, 连续 LOD 禁硬切换)
  + 天气演化 sim(风场平流 / 生消 / 降水耦合)
  + raymarch 内核(自适应步长 / cone 阴影采样 / 能量守恒)
  + 多次散射(Nubis powder + octave scattering 近似)
  + 共享高级基底(§5b): Lumen式GI(收环境光) / ReSTIR / VSM 云影 / froxel体积 / RT参考 / 时序上采样
  + 大气衔接: transmittance/aerial-perspective LUT(消费共享大气服务)
  + 云影投射(写共享阴影/光照) + god ray(接 froxel 体积)
  + motion vector(共享时序, 供 TAA/重投影上采样)

分叉响应(C类, 真分家 —— 只在"光照响应"分家, 见 §5):
  PBR:  物理 HG 相位 + 单/多次散射 + powder + 银边(silver lining) + 大气衰减
  NPR:  ramp 量化受光 + 卡通分层 + 手绘描边云廓 + 风格化银边块 + 吉卜力式蓬松形状
  自定义/混合: 分层云域多材质槽, 逐层/逐高度混合 PBR↔NPR

fallback:  近景高步数 raymarch, 远景低步数/imposter; 极远天空盒 LUT;
           RT 参考仅离线对拍
```

子系统**只产密度场/受光/散射数据**；着色走共享 closure、大气走共享服务、光照/阴影/GI 走共享基底——**不重写这些**。

---

## 2. 数据与资产模型

分层云域（Layer），共享统一 `CloudLayer` 抽象：

| 云类型 | 高度层 | 建模特征 | 典型用途 |
|---|---|---|---|
| **Cumulus（积云）** | 低层 | 高 coverage 对比、蓬松、detail erosion 强 | 晴天蓬松云、风暴云底 |
| **Stratus/Stratocumulus（层云）** | 低-中层 | 均匀铺展、低对比 | 阴天、雾层顶 |
| **Cirrus（卷云）** | 高层 | 稀薄、拉丝、curl 平流 | 高空丝缕云 |
| **Cumulonimbus（积雨云）** | 贯穿 | 高耸、强演化、降水源 | 雷暴、风暴系统 |

`CloudLayerAsset` 字段：几何（高度区间、云域包围体、密度场分辨率）、建模（coverage、cloud type、height gradient、Perlin-Worley 频率/振幅、detail erosion 强度、curl 平流强度）、物性（消光系数、散射反照率、各向异性 g、密度→高度映射）、天气（风速风向、演化速率、降水阈值）、着色（材质句柄 PBR/NPR/自定义/混合、银边强度、powder 强度、环境光比例）。数据驱动（RON/序列化），编译为建模参数 + WESL specialization key。

**weather map**（2D，随世界或相机平铺）：R=coverage、G=cloud type、B=降水强度、A=风扰动，驱动全局云分布，可由天气系统动态演化。

---

## 3. 管线阶段（端到端，每帧全链路，GPU-first）

```
[Extract]   活跃 CloudLayer + 天气状态 + 光源(太阳/月亮/闪电) → GPU uniform
[Prepare]   1. 天气演化: weather map 平流(风场) + 生消 + 降水更新(§9)
            2. 大气 LUT: 消费共享 transmittance/multi-scatter/aerial-perspective LUT
            3. (可选)密度场烘焙: 分层 3D 噪声 → 低分辨率密度缓存(远景/性能档)
[Queue]     4. LOD: 步数/分辨率 buffer 决策 + 相机相关 cloud imposter(远景)
            5. Raymarch(低分辨率 buffer): 自适应步长采样密度 + cone 阴影 + 能量守恒累积(§6)
            6. 多次散射: powder + octave scattering + 环境光(§7)
            7. 时序重投影: 低分辨率 → 全分辨率上采样(checkerboard/quarter-res + history)(§10)
            8. 云影投射: 光空间 raymarch 生成云影贴图 → 写共享光照/阴影(§12)
            9. god ray: 云缝透光 → froxel 体积内散射(§12)
[Composite] 10. 大气衔接: aerial perspective 按深度混合云与场景(§8)
           11. 着色解读: PBR 物理 / NPR 风格化(§5)
[Post]     12. motion vector → TAA/上采样, 云历史累积; 与主场景深度合成
```

差异化：**低分辨率 raymarch + 时序重投影上采样**是实时体积云的性价比核心；**分层云域**允许多云类共存并在合成阶段统一。

---

## 4. 程序化建模（Perlin-Worley 噪声，Nubis 式）

- **基础形状**：低频 Perlin-Worley 混合噪声（Perlin 连续性 + Worley 团块感）→ 云基底形状；
- **梯度调制**：coverage 梯度（weather map R）控云量、cloud type 梯度控云形（层云↔积云）、height gradient 控云在高度上的分布（底平顶蓬）；
- **细节侵蚀（detail erosion）**：高频 Worley 噪声从云边缘"啃"出蓬松细节（remap 而非叠加，保能量）；
- **curl 噪声平流**：卷云/风扰用 curl noise 做无散度扰动，产生拉丝/涡卷；
- **密度合成**：density = remap(base·coverage, erosion, height) × 消光，采样时按需（不预烘焙全域，除非远景性能档）。

---

## 5. PBR / NPR / 自定义 / 混合 分叉响应（只在光照响应分家）

### 5.1 PBR 云（物理）
Henyey-Greenstein 相位（前向散射，可双瓣 dual-lobe）；单次散射（沿视线 raymarch × 光可见度）；多次散射近似（§7）；powder 效应（暗边→亮的自遮挡糖粉感）；银边 silver lining（掠射太阳的亮边）；大气衰减（消费 transmittance LUT）。

### 5.2 NPR 云（风格化，illumination 轴）
ramp 量化受光（吉卜力/卡通分层）；卡通分层块（阈值化亮/暗区，非连续散射）；手绘描边云廓（云边缘检测 + 墨线）；风格化银边块（阈值银边而非连续）；程序化蓬松形状强化（更圆润 remap）；可选水墨扩散边。**关键：NPR 云同享 §5b 共享高级基底**，差异只在最终受光解读。

### 5.3 自定义
über-closure hook：作者注入自定义相位/受光映射/银边合成，编译进 WESL specialization，不改内核。

### 5.4 混合（分层/分区）
分层云域多材质槽（低层积云 PBR、高层卷云 NPR、特定区自定义）；逐高度/逐区掩膜在 PBR↔NPR 受光响应间插值；正交叠加（PBR 散射基底 + NPR 描边风格层）。**四者皆一等公民**。

---

## 5b. 共享高级基底接入矩阵（PBR/NPR/混合同享）

承接 `prism_aaa_advanced_features_zh.md`：以下为**共享基底服务**，体积云（无论 PBR 还是 NPR）**只消费不重写**，NPR 与 PBR 同等享有。

| 高级基底 | 归属 | 体积云如何消费 | PBR 侧 | NPR 侧 |
|---|---|---|---|---|
| **Lumen 式混合 GI** | `lighting/` | 云收环境天光/地面反照（ambient 项） | 物理环境散射 | ramp 量化环境底光 |
| **ReSTIR DI/GI** | `lighting/` | 多光源（太阳/月/闪电）储层采样 | 多光源散射积分 | 同储层 ramp 解读 |
| **VSM 云影** | `virtual_shadow/` | 云影投射写虚拟页；地面收云影 | 物理软云影 | 阈值硬云影+染色 |
| **froxel 体积** | `lighting/` 体积 | god ray/云缝透光注入 froxel | 物理体积内散射 | 风格化光柱分层 |
| **大气散射 LUT** | 大气服务(档3) | transmittance/multi-scatter/aerial-perspective | 物理天空衔接 | 风格化天色 ramp |
| **RT 参考/路径追踪** | `ray_scene/` | 仅离线对拍校准多次散射 | 参考对拍 | — |
| **时序上采样** | `temporal_upscale/` `motion/` `history/` | 低分辨率 raymarch → 全分辨率重投影 | TAA/上采样 | 同管线稳定描边 |

**结论：NPR 云不缺任何共享高级特性**，差异只在"受光响应"（§5）。（虚拟几何对体积云不适用——云无三角网格，改用密度场+raymarch，属预期。）

---

## 6. Raymarch 内核（能量守恒）

- **自适应步长**：空区大步跳过（用低分辨率 coverage/密度做 empty-space skipping），入云区细步；
- **视线积分**：沿视线累积透射率 T *= exp(−σ·density·ds) 与散射 L += T·scatter·phase·lightVisibility·ds；
- **cone 阴影采样**：向太阳做少量锥形采样估云内自遮挡光可见度（Nubis 6 样本式），避免二次 raymarch；
- **能量守恒**：detail erosion 与散射用 remap/归一化，避免加亮或变暗漂移；
- **早停**：透射率 T<阈值提前终止；低分辨率 buffer 降 raymarch 成本（§10 上采样补回）。

---

## 7. 多次散射（Nubis + octave scattering 近似）

- **powder 效应**：`1 − exp(−2σd)` 式暗边增强，模拟稠密云糖粉状自遮挡（近似替代真多次散射的暗边变亮）；
- **octave scattering（Wrenninge）**：多倍频叠加散射（每 octave 降能量、增各向异性、放宽相位），近似多次散射的柔和铺光；
- **环境光项**：从天空/地面收各向同性环境散射（消费 GI，§5b）；
- **各向异性**：HG 双瓣（前向锐 + 后向柔）匹配真实云相位。

---

## 8. 大气散射衔接（消费共享服务，不重写）

- **transmittance LUT**：太阳光穿大气到云的衰减；
- **multi-scatter LUT**：天空多次散射底光；
- **aerial perspective**：按视距在云与场景间混合大气雾（远云偏蓝/褪色），froxel 化；
- 大气本体是**基底服务**（档3），体积云只**采样 LUT** 做衔接。（注：此处仅衔接，避免与大气服务重复实现。）

---

## 9. 天气系统与动态演化

- **weather map 平流**：按风场 semi-Lagrangian 平流 weather map（coverage/type/降水随时间移动），产生云飘动；
- **生消**：coverage 随天气状态机演化（晴↔多云↔阴↔风暴），平滑插值禁跳变；
- **降水耦合**：积雨云降水强度写入 → 触发雨/雪（与 Ember 粒子 + 水体 §9e 湿润/涟漪联动），闪电作为动态光源（§12）；
- **风场**：全局风向风速驱动平流 + curl 扰动；风暴系统局部增强。

---

## 10. 时序重投影与上采样（实时性价比核心）

- **低分辨率 raymarch**：以 1/4 或 checkerboard 分辨率做 raymarch；
- **时序重投影**：用上一帧结果 + motion vector 重投影补全，逐帧填不同像素（Guerrilla/Ubisoft cloudscape 式）；
- **历史校正**：邻域 clamp / 方差裁剪防拖影，遮挡/视差失效处回退当前帧采样；
- **与 TAA/上采样统一**：接共享 `temporal_upscale`，云的 motion vector 由云平流速度 + 相机运动合成。

---

## 11. LOD 与大世界

- **步数 LOD**：远处降 raymarch 步数、增步长；近景高步数；
- **分辨率 buffer LOD**：远景更低分辨率 buffer；
- **cloud imposter**：极远云用相机相关公告板/预渲染 imposter（禁硬切换，交叉淡入）；
- **密度缓存**：远景可烘焙低分辨率 3D 密度缓存省重复噪声评估；
- **瓦片流**：超大天空按需驻留 weather/密度缓存，接 `paging`/`texture_streaming`。

---

## 12. 光照与阴影

- **主光**：太阳/月亮单次散射 + cone 自遮挡 + 多次散射近似；
- **云影投射**：光空间 raymarch 生成云影 → 写共享阴影/光照（地面/物体收云影，接 VSM §5b）；
- **god ray（体积光柱）**：云缝透光被 froxel 体积散射累积 → 光柱（接共享 froxel）；
- **动态光源**：闪电作为强瞬时点/面光注入云内散射（与 §9 风暴联动）；
- **环境光**：天空/地面各向同性项（消费 GI）。

---

## 13. 与其他子系统的边界（去重）

- **烟/火/爆炸气态流体** → **Ember 粒子引擎（网格流体）**，本引擎不碰；两者仅在"大气 LUT/froxel 体积"共享服务层相遇。
- **雨/雪粒子** → Ember 粒子；本引擎只发降水强度信号（§9）。
- **水体** → 水面可反射云（水体消费云的受光结果做反射），本引擎不管水。
- **大气散射/雾** → 共享基底服务，本引擎消费 LUT 不重写。
- **物理/风场** → 全局风参数可来自 gameplay/天气；本引擎只做云平流。

---

## 14. Crate 拆分与落地形态

落点 `pkg/prism_render_architecture/src/volumetric/`（照抄 cloth/hair/particle/water 范式：零依赖、`#![forbid(unsafe_code)]`、`extern crate alloc;`、手写向量数学、只允许 sqrt/exp 近似需自证或用多项式）。CPU 可验证纯函数与调度先行；GPU WESL 先落契约签名+脚手架，不作本机编译验证目标。

> 注：raymarch/散射涉及 `exp`。crate 若严格只允许 `sqrt`，则 `exp` 用有理/多项式近似封装在 `math.rs` 并加精度单测；GPU 侧用原生 `exp`。

```
volumetric/
  mod.rs          契约: CloudLayer/CloudKind/VolumetricBudget/句柄, 版本
  weather.rs      weather map 平流(semi-Lagrangian)/生消状态机/降水(纯函数, 确定性)
  modeling.rs     Perlin-Worley 建模: coverage/type/height 梯度 + detail erosion remap(纯函数)
  noise.rs        Perlin/Worley/curl 噪声基元(确定性, 可复现种子)
  raymarch.rs     raymarch 调度: 自适应步长/空区跳过/早停 决策(纯函数)
  scatter.rs      相位(HG 双瓣)/powder/octave scattering 权重(纯函数)
  cloud_lod.rs    步数/分辨率/imposter LOD 决策(阈值→桶, bin_xxx, 纯函数)
  temporal.rs     时序重投影/上采样计划(checkerboard/quarter-res, 历史校正参数)
  shadow.rs       云影投射与 god ray 注入计划
  atmosphere.rs   大气 LUT 衔接参数(aerial perspective 混合权重, 只采样不重写)
  math.rs         exp/pow 多项式近似 + Vec3 + EPS(附精度单测)
  budget.rs       raymarch/建模/上采样 预算仲裁(与形变预算同构)
```

黄金范式：`cloth/mod.rs`、`hair/lod.rs`（阈值/分桶/`bin_xxx`）、`virtual_geometry/bins.rs`（确定性输入序、越界跳过不 panic）、`deformation/schedule.rs`（预算仲裁）。`lib.rs` 按字母序加 `pub mod volumetric;`。体积云**无三角网格几何**，故**不接** `DeformationKind`（预期差异，区别于毛发/布料/水）。

---

## 15. 性能预算与效果验收

> 沙盒无 GPU，以下为**设计目标（design target），非实测**；上线需真机 profile 校准。

**性能预算（1080p→4K，桌面/主机档，设计目标）**：

| 项 | 预算(设计目标) | 降级路径 |
|---|---|---|
| 天气演化(weather map 平流) | ≤ 0.2 ms | 降更新频率 |
| Raymarch(1/4 分辨率, 中步数) | ≤ 1.5–3.0 ms | 降步数/降分辨率/增早停 |
| 多次散射(powder+2 octave) | 含在 raymarch | 降 octave 数 |
| 云影 raymarch | ≤ 0.5 ms | 降云影分辨率/频率 |
| 时序上采样(重投影) | ≤ 0.5 ms | checkerboard 降密度 |
| god ray(froxel 注入) | 复用共享 froxel | 降 froxel 分辨率 |
| 远景 imposter | ≤ 0.2 ms | 静态天空盒 |

**效果验收口径**：
- 云近/中/远无重复平铺、无 LOD 硬切换、imposter 交叉淡入平滑；
- 蓬松积云有 detail erosion 边缘、powder 暗边、掠射银边；
- 多次散射柔和铺光、稠密云内不发黑/不过曝（能量守恒）；
- 云随风平流、天气状态平滑演化（晴↔阴↔风暴无跳变）、降水触发雨/雪与地面湿润；
- 云影正确投到地面/物体、云缝 god ray 可见、闪电瞬时照亮云内；
- aerial perspective 使远云偏蓝褪色、与场景大气一致；
- 时序上采样无明显拖影/闪烁（history clamp 生效）；
- NPR 模式：ramp 分层、卡通块、手绘云廓描边、风格化银边均可开关，且**同享 GI/ReSTIR/VSM/froxel/大气/上采样**（§5b）；
- 四前端（PBR/NPR/自定义/混合）在分层云域可切换/混合且各享基底高级特性。

---

## 16. 可测性（CPU 可验证，沙盒内）

纯函数单测：噪声（确定性可复现、值域 [0,1]、Perlin 连续性、Worley 非负）；建模（remap 单调、height gradient 边界、erosion 能量不溢出）；天气（平流质量守恒近似、状态机插值∈[0,1]、降水阈值分类确定性）；raymarch（空区跳过判定、早停阈值、步长单调）；散射（HG 相位归一化、powder∈[0,1]、octave 能量递减）；LOD（步数/分辨率阈值单调、imposter 淡入权重∈[0,1]、桶越界跳过不 panic）；时序（重投影像素模式确定性、history clamp 参数有序）；math（exp/pow 近似精度在容差内，EPS 断言）；预算（配额不超、优先级降序+句柄升序贪心、首 job 防饿死）。

门禁（逐文件）：`rustfmt --edition 2024 <文件>` + `cargo clippy -p prism_render_architecture --all-targets -- -D warnings` + `cargo test -p prism_render_architecture`。禁 `cargo fmt --all` / 全量 clippy。

---

## 17. 落地路线图（M0–M8）

- **M0 地基（串行）**：`volumetric/mod.rs` 契约（`CloudLayer`/`CloudKind{Cumulus,Stratus,Cirrus,Cumulonimbus}`/`VolumetricBudget`/句柄）+ `lib.rs` 加 `pub mod volumetric;` + `math.rs`（exp/pow 近似 + Vec3 + EPS + 精度测）+ `budget.rs`。三关门禁全绿→commit。
- **M1 噪声**：`noise.rs`（Perlin/Worley/curl，确定性可复现）+ 测。
- **M2 建模**：`modeling.rs`（coverage/type/height 梯度 + erosion remap）+ 测。
- **M3 天气**：`weather.rs`（平流/状态机/降水）+ 测。
- **M4 Raymarch**：`raymarch.rs`（自适应步长/空区跳过/早停决策）+ 测。
- **M5 散射**：`scatter.rs`（HG 双瓣/powder/octave）+ 测。
- **M6 LOD/时序**：`cloud_lod.rs` + `temporal.rs`（LOD 桶 + 重投影计划）+ 测。
- **M7 阴影/大气衔接**：`shadow.rs`（云影/god ray 计划）+ `atmosphere.rs`（aerial perspective 权重，只采样）+ 测。
- **M8 GPU 接线（真机）**：WESL kernel（噪声/建模/raymarch/散射/云影/上采样）+ 渲染图接线 + 大气/froxel/VSM/GI/时序接入共享服务；真机 profile 校准。

**并行拆分建议**（M1–M7 写集互不重叠、只依赖 mod.rs + math.rs 契约）：A=noise B=modeling C=weather D=raymarch E=scatter F=cloud_lod+temporal G=shadow+atmosphere。M0 与 M8 串行（M0 的 math.rs 是并行前置）。

---

## 18. 风险与开放问题

- **exp 依赖**：crate 若严格禁 exp → 用多项式/有理近似封 `math.rs` 并精度自证；GPU 侧用原生。此点需先与 crate 约束方确认（否则 raymarch/散射无法纯 CPU 精确验证）。
- **时序重投影拖影**：快速云运动/相机转动 → history clamp + 方差裁剪，失效回退当前帧。
- **多次散射保真 vs 成本**：真多次散射太贵 → powder + octave 近似；影视级留 RT/路径追踪对拍桶。
- **与大气服务边界**：严禁重写大气散射，只采样 LUT；若大气服务未就绪需给临时 LUT stub。
- **确定性/网络**：云为表现层，默认不参与网络裁决；需确定性时固定种子+固定步长+固定重投影模式。
- **沙盒验证边界**：GPU 路径无法本机验证 → 仅承诺 CPU 纯函数正确性，GPU 部分标注未验证，真机补齐。

---

## 附录：术语
- **Perlin-Worley 噪声**：Perlin（连续）+ Worley（团块）混合，云建模基础。**curl noise**：无散度噪声，做拉丝/涡卷平流。
- **coverage / cloud type / height gradient**：云量 / 云形 / 高度分布 三大建模梯度。**detail erosion**：高频噪声侵蚀云边细节。
- **raymarch**：沿光线步进采样体积。**empty-space skipping**：空区大步跳过。
- **Henyey-Greenstein（HG）**：各向异性散射相位函数；双瓣=前向+后向。
- **powder 效应**：稠密云暗边变亮的自遮挡近似。**octave scattering（Wrenninge）**：多倍频叠加近似多次散射。**silver lining（银边）**：掠射太阳的云边亮边。
- **aerial perspective**：随视距的大气雾混合（远物偏蓝褪色）。**transmittance/multi-scatter LUT**：预积分大气衰减/多散射查找表。
- **cloud imposter**：远景公告板/预渲染云代理。**checkerboard/quarter-res**：低分辨率 + 时序重投影上采样。
- **VSM**：虚拟阴影贴图。**froxel**：视锥体素（体积光照）。**god ray**：体积光柱。
