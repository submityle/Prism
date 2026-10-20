# Prism 原生 Light / Shadow / GI / Reflection 生产设计基线 v7.0

> 版本日期：2026-10-20
> 状态：**目标架构、实施顺序与验收基线**。本文不是功能完成声明。源码、测试、真实 GPU 帧图、HDR readback 和性能报告优先于本文中的任何“当前状态”描述。
> 允许破坏性更新：本文明确允许删除旧 Light API、旧 Bevy lighting 入口和双向兼容层；迁移必须按阶段建立可回退的 source-control checkpoint。

## 0. 结论

Prism 当前已经具备 Light 系统的若干核心零件，但还没有形成完整的顶级 AAA 级产品闭环。

已有基础包括：canonical light contract、方向光/点光/聚光 payload、stable slot、generation、增量 dirty range、GPU submission fence、direct-light/GI reference 路径以及部分 app 侧执行接线。尚未闭环的部分包括：完整灯型覆盖、全灯型生产阴影、材质/纹理命中点辐射求值、生产级 ReSTIR DI/GI、world-space 多 bounce GI、完整 SSR/probe 反射、scene-to-present parity、真实硬件性能矩阵和旧入口删除。

因此 v7 采用以下产品定义：

> Light 不是一组组件，也不是一个 GPU buffer；它是从作者语义到场景辐射、可见性、时域历史、质量预算、诊断和最终输出的统一照明产品域。

目标闭环：

```text
Authoring / Import / Runtime adapters
        ↓ validate + normalize
Canonical LightScene + EmitterScene + EnvironmentScene
        ↓ immutable snapshot + SceneDelta
GPU Scene tables + geometry/material/radiance scene
        ↓ view planner
Direct + Shadow + Diffuse GI + Specular + Volume
        ↓ temporal/spatial reconstruction
Scene-linear HDR composition → exposure → display transform → present/offline output
```

## 1. 产品目标与边界

### 1.1 必须达到的产品能力

- 一份 canonical scene 语义覆盖 Rust、场景文件、编辑器、导入器和运行时脚本。
- 支持方向光、点光、聚光、矩形面光、圆盘光、胶囊/线光、天空/环境光和 emissive emitter；未闭环的灯型必须在 capability report 中明确拒绝或降级。
- 直接光、阴影、漫反射 GI、镜面反射、体积项和自发光共享稳定 identity、revision、预算和诊断框架，但不共享错误的 estimator 或历史。
- 低端设备有确定性保底路径，高端设备可启用 stochastic direct、VSM、world trace、硬件 RT、probe/cache 等增强 provider。
- 灯光变化、材质变化、几何变化、相机变化和设备恢复只使受影响的资源/历史失效，不默认全量重建。
- 每一次选择、拒绝、降级、延迟、缓存 miss、页耗尽和历史拒绝都可解释、可回放、可统计。
- CPU oracle、GPU intermediate parity、HDR 图像对比、窗口/离屏输出和性能 P50/P95/P99 分层验收。

### 1.2 明确不承诺

- 不把“4096 灯”“4.3 ms”“60 FPS”写成所有硬件和所有内容的保证；它们只能作为候选 workload/profile，必须绑定硬件、API、分辨率、场景、shader、温度和分位数。
- 不把 CPU reference、shader 编译通过、null backend 或模块存在描述为真实 GPU 产品能力。
- 不把 screen-space GI/SSR 描述为完整世界光照；它们必须有 probe、缓存或 world-space fallback。
- 本阶段不把 Nanite/虚拟几何、毛发、水、云、ML denoiser、离线路径追踪器和移动端专用路径混入 Light 域首个垂直切片。

## 2. 设计公理

1. **语义单源**：LightScene、EmitterScene、Material/RadianceScene 各有唯一 owner；任何 provider 不复制完整语义。
2. **数据先于算法**：先冻结 identity、单位、revision、ABI、失效和预算，再选择 clustered、RIS、ReSTIR、VSM 或 trace backend。
3. **一份语义，多种 recipe**：Exact、stochastic、screen-space、probe、software trace、hardware RT 是执行策略，不是互相独立的用户 API。
4. **计划先于执行**：Pass、资源、预算、能力和 fallback 在 Build/Schedule 阶段决定；执行阶段只消费冻结 plan。
5. **失败可见**：资源不足、灯型不支持、材质不支持、历史失效、shadow page miss、GPU 提交失败都必须产生结构化原因。
6. **增量是默认**：上传、重建、失效和回收按变化量处理；全量 rebuild 只能是显式 fallback 或低频 compaction。
7. **画质与性能分开测量**：不能用帧时间下降证明画质正确，也不能用 variance 下降证明最终图像更好。
8. **兼容层有寿命**：破坏性迁移完成后删除旧入口；不维护长期双向同步。

## 3. 分层架构与所有权

### 3.1 唯一数据流

```text
Authoring / Import / ECS / Script / Network
       ↓ adapter
Canonical Scene Owner
  LightScene / EmitterScene / EnvironmentScene
       ↓ snapshot + delta + journal
RenderApp Extract
       ↓ immutable frame snapshot
GpuScene Reconcile
  stable tables / geometry / material / radiance / shadow cache
       ↓ view-local build
LightingWorkGraph Planner
  candidates / clusters / reservoirs / shadow pages / GI probes / history
       ↓ budget + capability gate
FrameGraph Execute
  shadow → direct → GI → reflection → denoise → composition
       ↓ receipt
HDR target / present / capture / readback
```

### 3.2 crate 边界

| 域 | 唯一权威 | 允许的职责 | 禁止的职责 |
|---|---|---|---|
| 光源语义 | `prism_render_architecture::lighting` + canonical scene owner | 类型、单位、验证、handle、delta、oracle、recipe contract | GPU buffer、RHI submit、ECS 专用状态 |
| GPU 生命周期 | `prism_render_app` / GpuScene | stable slot、ABI payload、dirty range、fence、回收、device recovery | 再定义光源语义或材质规则 |
| 场景/材质/几何 | `prism_render_scene` | primitive、instance、material、texture、surface reconstruction、RadianceQuery | 另建 Light registry |
| 规划与诊断 | architecture lighting planner | capability、budget、quality、PlanDecision、Explain | 直接调用设备 API |
| 帧图执行 | app + render graph | resource dependency、pass recording、submit、receipt | provider 旁路提交 |
| 作者和工具 | editor/import/runtime adapters | schema、检查器、迁移、DebugView、capture/replay | 复制 runtime 为第二权威 |

### 3.3 构建期与执行期不变量

- Build 产物携带 `scene_lineage / sequence / epoch / revision / device_generation`。
- Execute 只能读取同一冻结 snapshot；不得在 pass 中隐式修改 canonical scene。
- 跨帧历史至少携带 `surface_id / instance_generation / material_revision / light_epoch / view_id / render_scale / sample_pattern`。
- 每个 GPU 资源写入都必须对应成功 submission receipt；失败或取消不得推进 acknowledged frontier。
- 任何异步结果都必须验证来源 device generation、queue domain 和 query identity。

## 4. Canonical Light IR

### 4.1 形状分类

| 类别 | 类型 | 主要贡献 | 首次生产闭环 |
|---|---|---|---|
| Analytic emitter | Directional | 直接光、主阴影、环境基准 | M2/M3 |
| Analytic emitter | Point | 局部直接光、局部阴影 | M2/M3 |
| Analytic emitter | Spot | 锥形直接光、局部阴影 | M2/M3 |
| Extended emitter | Rect / Disk | 面积直接光、软阴影、间接采样 | M4 |
| Extended emitter | Capsule / Line | 线光、近似面积采样 | M4 |
| Environment emitter | Sky / Environment map | 天空、环境辐射、低频 GI | M3/M6 |
| Mesh emitter | Emissive triangles / instances | 自发光、emitter proposal、GI | M5/M6 |

形状表示采样测度和可见性语义，不只是 UI 名称。Sky、baked radiance、probe field 不自动伪装成 punctual light；emissive mesh 不展开成若干点光。

### 4.2 共享属性与专属 payload

共享 `LightCommon`：

- stable identity、generation、scene lineage；
- transform、visibility/channel、color、temperature；
- mobility：`Static / Stationary / Dynamic`；
- contribution：direct、diffuse indirect、specular、volume、shadow；
- shadow request、importance、range、revision、source provenance。

形状数据单独存储：方向光方向/角直径，点光位置/强度/源半径，聚光锥角/IES，面光尺寸/法线/双面策略，环境图分布和 mip，emitter 三角面积/法线/radiance。

禁止使用一个覆盖所有灯型的稀疏巨型 struct；GPU ABI 按 ABI-compatible payload 分表，neutral value 必须定义且可测试。

### 4.3 物理与单位语义

| 输入 | Canonical 语义 | 约束 |
|---|---|---|
| Directional | illuminance 或明确的 scene-linear irradiance | 角直径/角半径独立记录 |
| Point / Spot | luminous flux 或 radiant intensity | 明确区分 lumen、candela、W/sr |
| Rect / Disk / Capsule | radiance/luminance + 尺寸 | 面积、法线、单/双面策略独立 |
| Sky / Environment | scene-linear radiance | 记录投影、分布、色域、mip |
| Emissive material | material emission/radiance | 不与 mesh emitter 重复贡献 |
| Color | 线性 RGB + primaries | sRGB 导入必须显式转换 |

曝光 EV、display transform 和 tone mapping 只能作用于 view/composition，不得偷偷改写 light intensity。

### 4.4 作者 API 原则

目标形态：

```rust
let light = lighting.add(Light::spot()
    .candela(1_500.0)
    .outer_angle_degrees(25.0)
    .color_temperature_kelvin(4_000.0)
    .shadow(ShadowRequest::Quality(ShadowQuality::High))?);
```

API 必须提供 builder、序列化 schema、编辑器检查器、导入 adapter 和 `explain_view`。单位必须出现在方法名或新类型中；非 finite、范围错误、色温/RGB 冲突、灯型不支持等返回结构化错误，不能 silent clamp。

`Auto` 只代表让 planner 选择，不代表不可见的魔法默认值；最终 recipe、预算、降级原因和覆盖项必须可导出。

## 5. 稳定身份、增量表与 GPU 生命周期

### 5.1 Identity

CPU/GPU handle 至少由以下维度组成：

```text
(scene_lineage, table_id, slot, generation)
```

缓存和异步结果还必须绑定 `content_revision / relevant_epoch / device_generation / queue_domain`。scene replacement 即使 sequence 恰好相同，也必须因 lineage 不同而失效。generation 防止删除后复用槽位导致 ABA。

### 5.2 Stable table 规则

- shader 读取 `capacity`；空洞是 neutral payload，不得把 `live_len` 当索引上限。
- update 合并为 dirty ranges；低脏比例使用区间上传，高脏比例才升级整表上传。
- remove 先提交 neutral payload；neutral 写入所在 queue 完成后才能回收并增加 generation。
- `UploadPlan` 是不可变快照；plan 创建之后的新修改不得被旧 plan 的 acknowledge 清掉。
- buffer 扩容、容量上限、显式 compaction 必须产生可解释的低频事件，并清理相关 temporal/cache identity。
- device lost/recreate 递增 device generation；CPU canonical snapshot 是唯一重建来源。

### 5.3 Completion proof

裸 `u64` completion value 不能独立证明“来自当前 queue/device”。completion proof 至少包含：

```text
(device_generation, queue_domain, submission_id, completion_value)
```

旧 device 的完成值不得回收新 device 的 slot；超过已知成功提交 frontier 的完成值不得被接受。窗口、离屏和 replay 路径都使用对应真实提交 receipt。

事务顺序固定为：

```text
prepare → stage/write → submit → acknowledge(plan, exact_receipt) → reclaim(completed_proof)
```

## 6. Direct Lighting 与工作生成

### 6.1 Exact path

适用于中低 overlap 或需要强确定性的场景：

- view-space cluster/tile 生成可见灯列表；
- 逐灯 evaluate BRDF、距离衰减、锥体、源尺寸和 visibility；
- cluster overflow 必须分批、显式 fallback 或进入 stochastic path；禁止静默丢灯；
- CPU oracle、GPU intermediate 和 HDR 图像分别对拍。

### 6.2 Stochastic path

适用于高灯密度或面积 emitter：

- 按 tile/surface 产生重要性 proposal；
- 使用 RIS/ReSTIR DI 或等价 resampling；
- 记录 target integral、proposal PDF、support、visibility、Jacobian、weight normalization；
- temporal/spatial reuse 绑定 surface/light/view revision；
- 采样预算固定时报告 variance 与图像误差；固定误差时报告所需样本和时间。

planner 不能只按总灯数切换；还要考虑屏幕覆盖、overlap、材质成本、shadowed-light 比、历史有效率、设备能力和内存 residency。

### 6.3 Light hierarchy

area/emissive emitter 可使用 light tree 或层次分布。叶子必须携带保守 bounds、功率/radiance、方向项和 PDF。delta light 与连续 solid-angle emitter 不得使用同一个未定义测度的 PDF。

## 7. Shadow 系统

### 7.1 Provider 抽象

`ShadowRequest` 只描述作者意图：是否需要、软硬程度、质量、更新频率、可接受成本。planner 选择 provider：

- 常规 shadow map / cascade / clipmap；
- virtual shadow page；
- ray-traced visibility；
- contact/screen refinement；
- static baked/cache。

provider 返回 capability、cost estimate、cache identity、miss/fallback 和 invalidation 结果，不改写 Light 的物理语义。

### 7.2 页池与缓存

虚拟页 identity 至少包含：`light_generation / view_projection_revision / caster_revision / page_coordinate / device_generation`。页池必须有：

- demand request、priority、pinning、eviction；
- 确定性排序和可回放分配；
- pool exhaustion 的显式 fallback；
- dirty caster、light transform、view、quality 变化的细粒度失效；
- miss 数、页占用、渲染时间和 fallback 原因 telemetry。

VSM 不等同于真实面积光软阴影。PCSS、VSM filtering、ray visibility 和多样本面积采样必须分别报告 bias、噪声和成本。

### 7.3 生产顺序

先闭环方向光和一种局部灯阴影，再扩展虚拟页、多灯型软阴影和 RT visibility。所有灯型的“存在”不代表所有 provider 都支持；不支持必须拒绝或降级并解释。

## 8. Geometry / Emitter / RadianceQuery

### 8.1 Scene representation

几何和材质是 GI 命中后的权威来源。命中结果至少能映射：

```text
primitive_id + instance_generation + geometry_revision + material_revision + barycentric
```

BVH reorder 不得破坏该映射。software BVH、SDF 和 hardware RT 是 backend，不是上层 GI 语义。

### 8.2 RadianceQuery

GI/反射命中后，通过 `RadianceQuery` 获取法线、UV、材质参数、纹理、emission、BSDF 和 surface flags。首个生产切片可以只支持不透明、无纹理 Builtin Unlit/Lit 子集，但 unsupported material 必须 reject 或走带诊断的 fallback，不得用默认颜色伪装完整材质。

### 8.3 Emissive emitter

自发光三角以面积、法线、radiance、纹理 revision 和实例 identity 进入 emitter table；面积 × 辐亮度的能量语义、单/双面策略和采样 PDF 必须保留。emissive 更新只脏化受影响的 emitter leaf、GI cache 和历史，不重建无关灯表。

## 9. GI、反射与时域重建

### 9.1 漫反射 GI 信号

| 信号 | 适合覆盖 | 不能承担 |
|---|---|---|
| Probe/volume | 低频稳定、低成本、离屏保底 | 细节级局部遮挡、任意高频 bounce |
| Screen-space trace | 当前视图细节和接触变化 | 屏幕外几何、disocclusion、完整世界真值 |
| World-space trace | 离屏几何和动态遮挡 | 不存在的材质/纹理求值、无限预算 |
| ReSTIR GI/cache | 重要路径复用和有限样本 | 未定义 PDF、身份、bias 时的生产承诺 |

每个信号必须声明 target、PDF/support、bias bound、raw variance、history domain 和 invalidation。不同 estimator 不得仅按 `1 / variance` 任意混合；先证明目标兼容，再定义分层合成或权重。

### 9.2 镜面

按 roughness、材质域和 coverage 组合 SSR、probe split-sum、planar、world trace。SSR 命中失败、屏幕边缘和历史拒绝必须退回 probe/environment/world trace；不能留下黑洞。漫反射 GI、镜面、阴影和体积拥有独立 history/denoiser/误差指标。

### 9.3 Temporal contract

历史至少携带：surface identity、normal、depth、motion、material revision、light/environment epoch、view rect、render scale、sample pattern。以下事件按影响范围 soft/hard invalidate：camera cut、灯影响域变化、材质/几何变化、cell unload/load、profile/render scale 变化、device recovery。

禁止两种错误：

1. 任意小变化触发全局 reset，导致质量和性能不可控；
2. stale history 悄悄参与，导致拖影、漏光或跨场景污染。

## 10. QualityContract、预算与 Explain

### 10.1 质量层级

质量由以下层级合成，优先级必须稳定且可解释：

```text
project defaults → device profile → view/camera → scene override → per-light request
```

设备 capability 决定 provider 是否可用；quality/budget 决定取多少样本、页、历史和分辨率。运行期调档只能在显式限幅内进行，并记录决策。

候选 profile（不是已验证性能承诺）：

| Profile | 目标能力 | GI | Reflection | Direct | Shadow |
|---|---|---|---|---|---|
| H1 | 低预算/无 RT | probe/volume | probe | exact，低 overlap | 主方向光 map |
| H2 | 中端实时 | screen + probe | SSR + probe | exact/stochastic 混合 | 主光 VSM + 局部 map |
| H3 | 高端实时 | screen + world trace | SSR + world trace | stochastic/ReSTIR | 多灯 VSM/RT provider |
| H4 | 离线/高质量 | path/reference | path/reference | 大样本 | reference/RT |

profile 名称只代表能力集合，不代表具体 GPU 型号、分辨率、FPS 或耗时。

### 10.2 预算分区

预算至少分成：

- direct lighting；
- shadow/page rendering；
- GI/radiance trace；
- reflection；
- temporal/spatial reconstruction；
- streaming/cache/temporary；
- reserve。

每个 pass 声明 reads、writes、临时资源、预计成本、质量贡献、fallback、同步要求和可回收区。超预算的 plan 在 execute 前被拒绝或降级，不允许执行完再把超支当成正常行为。

### 10.3 PlanDecision 与 Explain

统一结果：

```text
Selected
Rejected(reason)
Degraded(from, to, reason)
Deferred(until)
```

每条决定附带 capability、budget delta、expected/observed cost、residency、影响 signal、history validity 和 fallback。DebugView 至少覆盖：灯贡献、cluster overflow、shadow page/miss、GI source、reservoir/history validity、variance/error、材质 unsupported、内存 residency。

## 11. 性能、画质与维护门槛

### 11.1 性能测量

性能报告必须固定：硬件类别、GPU 型号、驱动、API、分辨率、内部 render scale、scene hash、shader profile、camera path、seed、warm-up、温度/功耗条件。至少报告：

- CPU extract/build/schedule 的 P50/P95/P99；
- GPU timestamp 的真实 pass execution P50/P95/P99；
- queue overlap、同步等待和 frame graph overhead；
- upload bytes/ranges、full rebuild、allocation、compaction；
- shadow pages、GI rays/samples、history reject、cache miss；
- 峰值显存、常驻显存、临时资源和 residency eviction。

### 11.2 画质与正确性

- 数学：单位、衰减、cone、area PDF、delta/continuous measure、BRDF × cosine、RIS normalization。
- ABI：size、alignment、offset、stride、neutral value、layout hash。
- GPU parity：CPU oracle 与 GPU intermediate 对拍，声明精度和容差。
- HDR 图像：linear radiance error、MSE/PSNR/SSIM/FLIP、能量/flux、ghosting、blur、漏光、接缝。
- 场景矩阵：室内小灯、户外日照、高灯密度、重叠软影、动态遮挡、离屏 GI、镜面 roughness 阶梯、材质/纹理变化、流送、camera cut、device recovery。

正确性 gate 失败时，不得进入性能绿灯统计；N/A 必须写原因和适用范围。

### 11.3 扩展清单

新增灯型必须同时完成：

```text
author schema/import
→ canonical validation
→ payload ABI + neutral value
→ CPU eval/sample/PDF oracle
→ GPU path
→ shadow/provider compatibility
→ budget/cost model
→ DebugMeta/Explain
→ schema migration
→ unit/GPU/image/benchmark tests
```

新增 recipe 必须声明 target estimator、reads/writes、资源 lifetime、capability、fallback、history/invalidation、seed/determinism、budget 和 oracle。只新增算法文件而未接入 planner、RenderGraph、telemetry 和验收矩阵，不算新增 runtime 能力。

## 12. 破坏性迁移方案

### 12.1 删除目标

- 删除旧 Bevy lighting 作为 runtime authority 的入口。
- 删除 `prism_render_scene` 与 `prism_render_app` 的双重 authoritative light registry。
- 删除每帧 dense rebuild 作为默认上传策略。
- 删除无 generation、无 epoch、无 device proof 的异步缓存 key。
- 删除 provider 直接写 GPU buffer 或旁路 queue submit 的路径。
- 删除没有 `PlanDecision` 的静默 fallback。

### 12.2 迁移顺序

| 阶段 | 交付 | 删除动作 |
|---|---|---|
| M0 | canonical IR、单位、ABI、identity、journal | 冻结旧字段，建立迁移表 |
| M1 | create/update/remove、stable table、fence/reclaim、device replay | 删除双重 light registry |
| M2 | directional/point/spot exact direct + HDR 输出 | 删除旧 direct-light 入口 |
| M3 | 主方向光 + 一种局部阴影 provider | 删除无诊断 shadow fallback |
| M4 | Rect/Disk/Capsule、emitter proposal、RIS/ReSTIR DI | 删除面积光伪装 punctual 的转换 |
| M5 | geometry hit → RadianceQuery → material/emission | 删除 triangle-indexed 假材质生产路径 |
| M6 | probe + screen + world trace GI、镜面回退 | 删除 screen-only GI 宣称 |
| M7 | editor、Explain、DebugView、capture/replay、性能矩阵 | 删除迁移兼容层和旧 API |

每一阶段必须保留：源码 checkpoint、CPU oracle、GPU parity、失败回退、诊断输出和明确的“未完成项”。

## 13. 验收矩阵

每条结果使用：

```text
hardware × API × driver × resolution × scene_hash × profile × seed × percentile
```

| 场景 | 必须验证 |
|---|---|
| S1 室内点/聚光 | 单位、衰减、cone、局部阴影、动态遮挡 |
| S2 户外方向光 | cascade/clipmap、天空、曝光、环境不重复注能 |
| S3 高灯密度 | exact/stochastic 切换、overflow、采样误差、缩放曲线 |
| S4 面光 | Rect/Disk PDF、软影 bias/noise、source size 变化 |
| S5 emissive mesh | 面积/radiance、emitter proposal、材质 revision |
| S6 动态材质/角色 | hit identity、material reconstruction、history reject |
| S7 离屏 GI | screen miss、probe/world fallback、无黑洞/漏光 |
| S8 镜面阶梯 | SSR/probe/world transition、roughness、denoise 独立性 |
| S9 流送/恢复 | scene lineage、slot reuse、device generation、replay |
| S10 确定性 | fixed seed、plan replay、CPU/GPU 容差、帧间稳定 |

发布判定：correctness 全部通过；unsupported 能力均显式报告；性能必须同时给出 P50/P95/P99 和原始 capture；任何未完成项不得由设计文档自动转为 Done。

## 14. 当前仓库状态快照

截至 2026-10-20，本工作区可确认的范围如下：

- `prism_render_architecture::lighting` 已有 Light contract、recipe/plan、trace、cache、replay、shadow/GI reference 资产。
- `prism_render_app` 已有原生 light extraction、方向光/点光/聚光 payload 上传、direct/GI 初始执行路径和提交/呈现接线。
- `IncrementalGpuTable` 已接入 canonical directional/punctual handle、dirty byte range、neutral removal、submission acknowledgement 和部分 device recovery 处理；测试和真实物理 GPU 验证仍需完成。
- `prism_render_scene` 仍残留依赖 Bevy Light component 的旧 extraction/plugin 与公开 runtime Light API；新 `prism_render_app` 路径已存在，但旧入口尚未完成删除。因此目前仍处于双路径迁移期，不能称为“Bevy lighting 已完全废弃”。
- 当前真实 direct-light ABI 主要覆盖 directional/point/spot；Rect、Disk、Capsule、Sky、Emissive 尚未形成完整生产 GPU 闭环。
- VSM/虚拟页驻留、全灯型真实阴影、完整材质/纹理 RadianceQuery、生产 ReSTIR DI/GI、world-space 多 bounce、完整 SSR/probe 反射、scene-to-present parity 和性能 H×S 数据仍未完成证明。
- 工作区存在多个并行模块的未提交改动；本文件只新增设计基线，不把其他改动视为本次文档已验证内容。

### 14.1 本次验证记录

- `git diff --check`：Light/GI 相关目标文件检查通过。
- 架构 crate 定向 `cargo test -p prism_render_architecture lighting --lib`：未能进入测试执行阶段。编译被工作区现有 `prism_physics_core` 改动阻断：`world.rs` 调用 `BodyStorage::reset_solver_mass_properties` / `set_solver_mass_properties` 时当前编译视图找不到对应方法，并存在 `Vec3` 与 `f32` 类型不匹配。该失败不证明 Light/GI 改动通过或失败；必须在修复该工作区构建阻断后重跑。
- `prism_render_app` 定向测试、真实物理 GPU 测试、HDR readback 和性能矩阵：本次尚未验证。
- 因上述限制，本文是架构设计与迁移基线，不是已通过构建的实现交付声明。

## 15. 最终决策

已确定：

- canonical scene 单一所有权；
- generation-safe stable identity；
- app 负责 GPU 资源与提交生命周期；
- architecture 负责 renderer-independent contract、recipe、budget 和 oracle；
- screen-space 只是覆盖内信号；
- plan 必须能解释选择、拒绝、降级、延迟和缓存失效；
- Light、Shadow、GI、Reflection 共享身份/预算/诊断框架，但不共享错误的物理 estimator 和历史；
- 允许破坏性删除旧入口，不维护长期双向同步。

必须用 benchmark/spike 决定：

- stable table 与 dense upload 的阈值；
- exact 与 stochastic direct 的切换点；
- 首发 shadow provider 组合；
- software BVH、hardware RT 与 scene representation 的取舍；
- probe density、更新频率、历史长度和 denoise 参数；
- 各硬件/API 的内存、功耗、质量和帧时间 profile。

任何未被代码、测试、真实 GPU 或图像/性能报告证明的内容，都继续标记为目标、候选或未闭环，而不是完成。
