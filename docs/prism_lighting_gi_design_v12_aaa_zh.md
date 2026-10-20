# Prism 次时代 Light / Shadow / GI / Reflection 总体设计 v12

> 日期：2026-10-20
> 文档性质：破坏性升级后的目标架构、实施合同和验收规格。
> 适用范围：Prism 通用实时 3D 渲染器、编辑器预览、离屏 capture、未来 reference renderer。
> 重要声明：本文不是功能完成声明。只有源代码接线、自动化测试、真实物理 GPU 中间结果、HDR 图像和性能原始数据齐备后，才能把某项从“目标”改为“已完成”。

## 0. 最终结论

Prism 缺少的不是一个 `LightPlugin`，而是一条完整的、唯一权威的照明产品链：

```text
作者输入 / 导入 / 运行时命令
        ↓
Canonical Lighting Scene
  Light / Emitter / Environment / Geometry / Material / View
        ↓ validate + normalize + journal + revision
Immutable Snapshot + SceneDelta
        ↓
GPU Scene Reconcile
  stable handle / dirty range / distribution / residency
        ↓
Per-view Planner
  capability / quality / budget / importance / history
        ↓
Lighting Work Graph
  direct → shadow → diffuse GI → reflection → reconstruction
        ↓
Scene-linear HDR Composition
        ↓ exposure → display transform → present / EXR / capture
        ↓
Receipt + Explain + Telemetry + Replay
```

v12 的不可破坏决策：

1. `CanonicalLightingScene` 是光照语义的唯一 CPU 权威。
2. `GpuScene` 是 GPU slot、资源生命周期、提交确认和回收的唯一权威。
3. `LightingWorkGraph` 是按视图选择 provider、生成工作和批准预算的唯一入口。
4. Direct、Shadow、GI、Reflection、Environment、Emitter、Material 各自有清晰 owner，不复制第二份真值。
5. Execute 阶段不能临时换算法、扩大预算、创建作者级灯或静默 fallback。
6. 所有 miss、unsupported、budget rejection、page miss、history reject 都必须可解释、可回放、可观测。
7. Bevy 可以在迁移期作为 ECS、资源和 RHI 适配层，但 Bevy Light、Bevy PBR 和 Bevy RenderGraph 不能继续是 Prism 的照明权威。
8. “AAA”只表示绑定硬件、内容、分辨率、质量档、帧时间和图像误差的一组可复现证据，不是功能名或性能口号。

## 1. 产品目标与范围

### 1.1 首个生产基线

首个生产基线必须支持同一套 canonical scene 被以下消费者使用：

- 实时桌面渲染；
- 编辑器视口和灯光调试视图；
- 离屏 capture、HDR/EXR 输出和回归测试；
- 低能力设备的保底 profile；
- 未来 reference renderer 或离线验证路径。

光照语义覆盖：

| 类别 | 形状/来源 | 首要语义 |
| --- | --- | --- |
| Analytic | Directional、Point、Spot | 直接光和可见性 |
| Extended | Rect、Disk、Sphere、Capsule/Line | 面积、源尺寸、软阴影和采样 |
| Environment | Sky、Environment Map、Atmosphere | 环境辐射和 miss radiance |
| Mesh Emitter | 发光三角形、发光实例、发光簇 | GI、直接采样和材质辐射 |
| External Radiance | lightmap、probe field、capture | 明确标识的外部辐射来源 |

### 1.2 首期非目标

以下能力不能作为 Light API 的隐含前提：

- Nanite 等价的通用虚拟几何；
- 所有透明、毛发、水、云和体积天气的统一高质量 GI；
- 跨厂商一致的硬件 RT 性能；
- 电影级离线路径追踪、路径引导、谱渲染和机器学习降噪；
- 任意硬件、任意场景、任意分辨率下固定的“AAA 帧时间”；
- 为了兼容旧 API 长期保留旧 registry、双写路径和无 owner 的 fallback。

## 2. 参考优秀产品后的结构性取舍

### 2.1 Unreal Engine：分层覆盖和虚拟化阴影

借鉴：screen-space 与 world representation 的互补、GI/反射共享场景查询、按需虚拟阴影页、缓存、streaming 和 scalability profile。

Prism 不复制具体产品边界：Lumen Scene、Distance Field、VSM page 或虚拟几何不是 Prism 的上层语义。上层只表达光源、辐射、可见性、质量、预算和作者意图，representation 由 provider 选择。

### 2.2 Unity Adaptive Probe Volumes：低频辐射的空间自适应

借鉴：分级 brick、几何密度驱动的采样、sky occlusion、scenario、streaming、leak 修复和可视化工具。

Prism 的 `ProbeVolume` 只是低频 radiance provider，不是 Direct、GI、Reflection 的混合垃圾桶。brick residency、validity、scenario、stale read 和 leak risk 必须进入统一的 budget/explain。

### 2.3 NVIDIA ReSTIR / RTXDI：采样库不拥有场景

借鉴：candidate、proposal PDF、reservoir、temporal/spatial reuse，以及让 many-light 成本与灯总数弱相关的思路。

Prism 的边界：ReSTIR/RTXDI 是 recipe/provider，不是 Light API、Material API 或 GPU Scene owner。reservoir 必须保存 source identity、generation、revision、measure、PDF、target、history key 和 confidence；偏差必须登记。

### 2.4 专用时域重建：信号分治

阴影、漫反射 GI、镜面反射、体积和透明的噪声与失效原因不同。禁止万能 history、万能 blur 和万能 clamp。每个 signal 拥有独立的 history key、variance、accept/reject、clamp、重置和预算。

### 2.5 Prism 的取舍表

| 问题 | Prism 方案 |
| --- | --- |
| 少量/低重叠灯 | exact clustered |
| 大量/高重叠灯 | stochastic many-light / ReSTIR provider |
| 大世界阴影 | conventional、virtual page、ray visibility 分层 |
| 低频间接光 | adaptive probe/cache + Environment miss |
| 屏幕内细节 | screen GI / SSR |
| 屏幕外补盲 | world trace、probe、Environment |
| 平台差异 | CapabilityGraph + profile recipe |
| 内容变化 | stable identity + revision + dirty range |
| 质量变化 | BudgetLedger + hysteresis + minimum residency |

## 3. 总体架构：五阶段、六个权威域

### 3.1 五阶段

```text
Build
  validate / normalize / journal / snapshot / delta / invalidation
      ↓
Plan
  capability / quality / budget / importance / history / residency
      ↓
Execute
  reconcile / encode / dispatch / submit / receipt
      ↓
Resolve
  reconstruction / HDR composition / exposure / display transform / present
      ↓
Evidence
  telemetry / explain / debug view / capture / replay / benchmark
```

硬规则：

- Plan 之后 Execute 只能读取不可变 plan 和已确认 revision；
- Execute 不得写 canonical scene，不得扩大预算，不得临时创建 fallback；
- Resolve 不得重新解释作者单位，不得私自 gamma、tone map 或重复曝光；
- Evidence 只能观察和记录，不能修改实际结果以“通过”指标。

### 3.2 六个权威域

```text
LightScene        analytic / extended light
EmitterScene      emissive mesh / primitive / triangle distribution
EnvironmentScene  sky / atmosphere / environment map / miss radiance
GeometryScene     mesh / instance / visibility / BVH or AS representation
MaterialScene     material IR / texture / BSDF / emission authority
ViewScene         camera / output / quality / history / capture policy
```

这些域共享：

```text
SceneIdentity / DomainRevision / DeviceGeneration / Capability / Budget / Explain
```

这些域不共享没有定义过的：

```text
estimator / PDF measure / visibility rule / history / temporal filter / physical unit
```

### 3.3 唯一变更传播图

```text
AuthorCommand / Import / RuntimeMutation
        ↓
Validate + Normalize
        ↓
CanonicalMutation + JournalEntry
        ↓
DomainRevision + SceneDelta
        ↓
AffectedRange / AffectedCell / AffectedHistory
        ↓
GPU Reconcile + Planner Invalidation
        ↓
FrameLightingPlan
```

每个 delta 必须携带：domain、handle、generation、revision、影响范围、失效类型、pending frontier 和 replay 信息。

## 4. Canonical Light Scene 与作者 API

### 4.1 作者表达意图，系统选择实现

示意 API：

```rust
let handle = lighting.add(
    Light::spot()
        .intensity(Candela::new(1_500.0))
        .inner_angle(Degrees::new(18.0))
        .outer_angle(Degrees::new(25.0))
        .ies(profile)
        .shadow(ShadowRequest::Auto)
        .contributes(Contribution::DIRECT | Contribution::SPECULAR)
        .mobility(Mobility::Dynamic)
        .linking(LightLink::Group("hero")),
)?;
```

规则：

- `Candela`、`Lumens`、`Lux`、`Nits`、`EV100` 必须是不同类型或不同方法；
- 裸 `f32` 不能同时承载不同单位；
- 色温、色域、IES、cookie、source extent、light function、barn door、贡献域和 linking 可独立修改；
- NaN、Inf、负值、非法 cone、退化 basis、失效资源句柄必须拒绝；
- 编辑器可以做可恢复 clamp，但必须生成 warning、原值和 canonical 修正值；
- `Auto` 只表示 Planner 选择 provider，不能表示“没有明确结果”；
- mobility 只定义更新、缓存和作者意图，不直接硬编码某种阴影或 GI 算法。

### 4.2 `LightRecord` 逻辑结构

```text
LightRecord {
  identity: LightId + Generation + SourceId
  kind: LightKind
  transform: position / orientation / local_frame
  emission: color / unit / flux / intensity / radiance / temperature
  shape: radius / size / capsule_length / cone / barn_door
  influence: bounds / range / falloff / channel_mask
  visibility: shadow_policy / ray_policy / casts / receives / volume_policy
  contribution: direct / indirect / specular / volume
  sampling: importance_class / distribution_node / source_extent
  mobility: static / stationary / dynamic / streaming
  authoring: source_asset / import_revision / provenance / warnings
  revisions: field_epochs / schema_revision
}
```

### 4.3 九类发光语义

| 类型 | 作者输入 | Direct | Shadow | GI/Emitter |
| --- | --- | --- | --- | --- |
| Directional | lux、角直径 | clustered/exact | clipmap、VSM、ray | 可选 |
| Point | candela/lumen、source radius | clustered/stochastic | cube、octa、ray | 可选 |
| Spot | candela、内外锥、IES | clustered/stochastic | projected、VSM、ray | 可选 |
| Rect | nits/lumens、basis | area exact/stochastic | sampled visibility/ray | 可选 |
| Disk | nits/lumens、半径 | area exact/stochastic | sampled visibility/ray | 可选 |
| Sphere | nits/lumens、半径 | area/stochastic | ray/contact | 可选 |
| Capsule/Line | nits/lumens、端点/半径 | analytic/stochastic | ray/contact/近似 | 可选 |
| Environment | sky/map/atmosphere | environment distribution | sky/world visibility | miss radiance |
| Emissive Mesh | 材质 emission | emitter sampling | sampled/ray | primary emitter |

Rect、Disk、Sphere、Capsule 的 deterministic sample estimator 不得自动命名为生产级面积软阴影。必须完成逐样本 visibility、PDF/measure、CPU oracle、GPU image diff 和高 overlap benchmark 后才能进入生产 profile。

## 5. 单位、颜色和能量语义

### 5.1 Canonical 表达

| 来源 | canonical 表达 | 强制记录 |
| --- | --- | --- |
| Directional | illuminance 或明确的 scene-linear irradiance | 角直径、方向和色域 |
| Point/Spot | luminous flux 或 radiant intensity | lumen/candela 来源单位 |
| Area/Mesh | radiance 或明确的 photometric authoring value | 面积、法线、单双面 |
| Environment | scene-linear radiance | 投影、分布、mip、色域 |
| Emissive material | material emission/radiance | 纹理色彩空间和材质 revision |
| Color | scene-linear RGB | primaries、转换证据 |

曝光、白平衡、tone mapping、gamut mapping 和 transfer function 不进入光照积分。艺术增益必须标记为 `ArtisticControl`，不能伪装成物理输入。

### 5.2 能量和重复注能

- direct、indirect、specular、volume 和 emitter contribution 使用独立 mask；
- 同一 emission 不能同时作为材质命中贡献和独立 punctual light 重复注能；
- 环境项和 probe/cache 不能在同一 support 区域无声明地重复累加；
- exposure 只能在 HDR composition 之后应用一次；
- 所有 lighting provider 只输出 scene-linear radiance；
- 降级只能减少样本、范围、更新频率、分辨率或模型精度，不能增加能量或改变作者单位。

## 6. 稳定身份、字段 epoch 和 GPU Scene

### 6.1 稳定句柄

```text
SceneIdentity { lineage, sequence, epoch }
StableHandle  { table, slot, generation }
ResourceKey   { handle, content_revision, device_generation }
HistoryKey    { signal, surface, material, light_epoch, env_epoch,
                view, extent, scale, sample_pattern, algorithm_revision }
```

slot 不能单独代表身份。所有 command、reservoir、shadow request、debug selection 和 capture replay 都必须验证 generation 与 scene epoch。

### 6.2 字段级 epoch

至少区分：

```text
TransformEpoch
EmissionEpoch
ShapeSamplingEpoch
ShadowPolicyEpoch
VisibilityEpoch
EmitterMaterialEpoch
EnvironmentEpoch
GeometryEpoch
MaterialEpoch
SchemaEpoch
DeviceEpoch
```

禁止用一个全局 `lighting_changed` 布尔值让 cluster、shadow、GI、reflection 和 history 全部失效。

### 6.3 GPU 分表和上传事务

```text
DirectionalTable
PunctualTable
AreaTable
EmitterTable
EnvironmentTable
ShadowSourceTable
LinkingTable
```

所有分表由同一个 `LightingUploadPlan` 统一提交和确认。事务状态机：

```text
read canonical frontier
→ reconcile dirty ranges
→ freeze immutable UploadPlan
→ validate ABI / capacity / device generation
→ stage/write
→ record GPU work
→ submit
→ SubmissionReceipt
→ exact-plan acknowledge
→ completion proof
→ retirement / recycle
```

必须满足：

- 旧 plan 不能确认新 revision；
- skipped frame、surface acquire failure、pre-submit error 不推进 acknowledged frontier；
- present failure 与 GPU submission success 分开记录；
- partial table failure 不能静默确认其他表；
- compaction 必须输出 remap 或 epoch invalidation；
- device recovery 增加 `device_generation`，从 canonical snapshot 重建；
- pool 耗尽必须产生结构化拒绝或降级原因，不能静默吞掉。

### 6.4 ABI

Rust/IDL 是唯一 ABI 源，shader 结构由生成或校验得到。CI 必须检查 size、alignment、offset、stride、layout hash、neutral payload 和 ABI version。零值、魔法数字和“全零代表无效”不能替代结构化 validity。

## 7. Unified Planner：能力、质量、预算和重要性

### 7.1 CapabilityGraph

能力至少包括：

```text
backend/API
shader model
storage/texture limits
subgroup/atomics
indirect dispatch
bindless/residency
ray query / AS
timestamp
async queue 实际并发
格式和压缩支持
```

provider 必须声明 required/preferred capabilities、fallback、资源限制、确定性行为、已知偏差和 unsupported cases。代码能编译不等于设备支持。

### 7.2 QualityProfile

质量档是 Planner 输入，不是编译常量：

| 档位 | 目标定位 | 主要路径 | 典型约束 |
| --- | --- | --- | --- |
| H1 | 集显/低功耗 1080p | probe + environment、少量 exact | 低 rpp、低页数、短历史 |
| H2 | 主流独显 1440p | screen + probe、cluster/stochastic | 中等灯数和 shadow budget |
| H3 | 高端独显 4K | screen + world trace、VSM/高质量 reflection | 高样本、高历史、高驻留 |
| H4 | capture/reference | path/reference provider | 不承诺实时帧时间 |

每档至少定义 internal resolution、DRS range、max visible lights、overlap policy、shadow page budget、GI provider、rays/samples、reflection split、history policy、VRAM ceiling、upload ceiling 和 determinism mode。

### 7.3 BudgetLedger

每个 view 维护三本账：

```text
CPU: extract / build / plan / encode / submit
GPU: direct / shadow / GI / reflection / reconstruction / present
VRAM: persistent / transient / upload / residency / reserve
```

每个 pass 申报 estimated CPU/GPU、transient/persistent/upload bytes、lights/samples/rays/candidates/pages/bricks、quality gain、known bias、fallback 和同步依赖。

超预算顺序固定：

1. 剔除没有屏幕影响的 tile/cluster；
2. 降低远处或低重要性资源的更新频率；
3. 降低 sample/ray/page 或内部比例；
4. 切换更便宜 provider；
5. 局部拒绝并使用明确低频/环境 fallback；
6. 只有项目 policy 明确要求时才整帧失败。

### 7.4 PlanDecision

```rust
enum PlanDecision<T> {
    Selected(T),
    Degraded { from: T, to: T, reason: PlanReason },
    Deferred { until: FrameId, reason: PlanReason },
    Rejected { reason: PlanReason },
}
```

`PlanReason` 必须稳定、可序列化、可在编辑器显示，例如：`MissingCapability`、`TransientBudgetExceeded`、`PersistentMemoryExceeded`、`ShadowPoolExhausted`、`HistoryInvalid`、`MaterialUnsupported`、`SourceDistributionStale`、`DeviceGenerationMismatch`、`PageRequestOverflow`、`MissingDynamicCasterEpoch`。

provider 切换必须有 hysteresis、最小驻留帧数和速率限制，避免 exact/stochastic、SSR/world、map/VSM 每帧抖动。

## 8. Direct Lighting 和 many-light

### 8.1 Exact clustered

```text
view culling
→ tile/cluster light list
→ kind/material bins
→ direct evaluate
→ shadow visibility
→ contribution mask
```

适用于低到中等 overlap、确定性模式、低端 profile 和没有高级 RT 能力的设备。overflow 不得静默截断；必须扩大列表、分批执行、切 stochastic 或显式拒绝低优先级贡献。

### 8.2 Stochastic many-light

```text
source distribution / light tree
→ candidates
→ proposal PDF
→ target evaluation
→ temporal/spatial reuse
→ visibility
→ radiance + variance + confidence
```

reservoir 至少包含 source kind、slot、generation、source revision、measure、proposal PDF、target、weight sum、candidate count、surface identity、view identity、sample pattern、visibility revision、age 和 confidence。

离散 PMF、面积测度、solid-angle PDF 及其转换必须显式。没有 normalization、bias 和 support 声明的随机路径只能是实验 provider。

## 9. Shadow System

### 9.1 生命周期

```text
ShadowRequest
→ capability check
→ receiver demand
→ page/map allocation
→ caster epoch validation
→ depth render
→ visibility binding
→ filter/reconstruct
→ receipt + telemetry
```

depth draw、page table entry 和 GPU shadow record 必须引用同一个 `ShadowAllocationId`。

### 9.2 Provider 矩阵

| 灯型 | 高质量 | 中质量 | 保底 |
| --- | --- | --- | --- |
| Directional | virtual clipmap / ray visibility | cascaded/virtual reduced | conventional cascade |
| Spot | virtual perspective / ray | cached projected map | conventional map |
| Point | octa/cube virtual / ray | cached cube | conventional cube |
| Rect/Disk/Sphere | sampled ray visibility | proxy/map/filter | hard/contact approximation |
| Capsule | ray/contact | analytic approximation | no shadow + Explain |
| Emissive | sampled/ray | probe/cache | no direct visibility + Explain |
| Environment | sky occlusion/world visibility | probe occlusion | environment radiance only |

传统 shadow map 不能伪称真实面积软影；source size/angle 只有真正影响 visibility 时才可声明支持。

### 9.3 VSM / page contract

page key 至少包含：`light_generation`、`projection_revision`、`caster_cell_revision`、`virtual_page_coordinate`、`quality_revision`、`device_generation`。

allocator 必须支持 receiver-driven demand、确定性 priority、physical page pool、pin/evict、static cache、dynamic overlay、coarse fallback、missing-page policy 和 request/hit/miss/allocation/eviction/residency 诊断。

动态 caster 的 transform、skinning、morph、material alpha 或 instance visibility 变化必须推进 caster epoch。没有 epoch 必须返回 `MissingDynamicCasterEpoch`。

## 10. RadianceQuery、Emitter 和材质真值

### 10.1 命中不等于辐射

BVH、AS、SDF 或其他 geometry representation 只返回 hit，不拥有材质真值。命中身份至少包括：

```text
primitive_id
instance_generation
geometry_revision
material_id
material_revision
barycentric/local coordinate
```

命中后必须通过 `RadianceQuery` 获取 normal、tangent、UV、base color、roughness、metalness、IOR、transmission、opacity、emission、BSDF、two-sided、alpha、surface flags、light linking、support level 和 fallback reason。

常量 triangle radiance 只能用于骨架和契约测试，不能作为生产 emissive/GI 材质实现。

### 10.2 Emissive distribution

distribution 以 `area × luminance` 为基础，可按空间、材质重要性、方向和 streaming cell 分层。纹理、材质、实例 transform 变化只更新受影响的 leaf/segment。

同一 emission 必须从同一份输入重算 proposal PDF、solid-angle PDF、visibility 和最终权重。禁止 material hit、emitter sampling、probe bake 各自拥有不同的辐亮度真值。

## 11. GI：三层 coverage 加 Environment miss

### 11.1 固定 coverage 链

```text
screen hit with valid material
        ↓ miss / invalid / unsupported
world trace
        ↓ miss / over budget / unavailable
adaptive probe/cache
        ↓ miss / stale / invalid
EnvironmentScene miss radiance
```

这是 coverage 优先级，不是无约束叠加。多个 estimator 同时贡献时必须定义共同 target、support、PDF、MIS/resampling normalization 和 bias。

### 11.2 Screen GI

适合屏幕内高频细节，必须使用 depth、normal、material identity、thickness 和 disocclusion 判据。screen miss 不能直接返回黑色。步数、rpp、分辨率和重建由 BudgetLedger 决定。

### 11.3 World trace

hit-time 必须回到 `RadianceQuery`；BVH/AS reorder 不能破坏 instance generation；ray miss 必须访问 `EnvironmentScene`；unsupported material 必须产生 support/fallback 诊断。

### 11.4 Adaptive Probe/Cache

probe/brick 至少保存 validity、relocation、classification、sky occlusion、leak risk、scenario、confidence、domain revision 和 residency。cache key 至少包含 cell、normal cone、material class、scene/light/environment revision 和 device generation。

### 11.5 GI 默认生产进入门

必须同时满足：

1. `RadianceQuery` 完整；
2. world miss → `EnvironmentScene`；
3. CPU oracle 与 GPU intermediate 对拍；
4. dynamic geometry、material change、cache stale rejection 通过；
5. reference path 有固定时间和图像误差报告；
6. 低预算档仍有非黑、能量受控、可解释的结果；
7. 有真实 HDR 图像回归，而不只是 buffer 非空。

## 12. Reflection

固定 coverage：

```text
explicit planar override
→ valid SSR/SSSR hit
→ world-space reflection trace
→ local probe with parallax correction
→ global EnvironmentScene
```

roughness 同时影响 ray/cone、mip、sample rate、provider 和 denoiser。SSR 必须做 depth/thickness、normal、material、edge 和 confidence 检查。reflection history 不得与 diffuse GI history 复用。miss 不得默认输出黑色。

## 13. Temporal、重建与 HDR 合成

### 13.1 Per-signal History

```text
HistoryKey {
  signal_kind,
  surface_id,
  instance_generation,
  material_revision,
  light_or_environment_epoch,
  view_id,
  projection_revision,
  render_extent,
  render_scale,
  sample_pattern,
  algorithm_revision,
}
```

camera cut、teleport、projection jump 和 device recovery 直接 hard reset；局部灯、材质、几何和 environment 变化按范围 soft invalidate。

### 13.2 Scene-linear composition

```text
emission
+ direct diffuse/specular
+ indirect diffuse
+ indirect specular
+ participating media
→ pre-exposure inverse
→ exposure
→ display/output transform
→ UI/compositor policy
→ present / EXR
```

所有 lighting pass 输出 scene-linear HDR。任何 pass 不得私自 gamma、tone map 或重复曝光。SDR、HDR10 和 EXR 共享 scene-linear source，但使用不同 output transform。

## 14. 性能、内存和调度

### 14.1 目标档位，不是完成证据

下表是 profile 设计目标，必须由指定硬件、场景哈希、相机路径和质量档实测后才能发布为性能承诺：

| 项目 | H1 | H2 | H3 | H4 |
| --- | ---: | ---: | ---: | ---: |
| 输出基准 | 1080p | 1440p | 4K | reference/capture |
| GI | probe/env | screen + probe | screen + world trace | path/reference |
| 灯光路径 | exact | exact/stochastic | stochastic/ReSTIR | reference |
| 目标 GPU lighting budget | 待测 | 设计目标约 4–6 ms | 待测 | 不承诺 |
| 历史上限 | 8 | 32 | 64 | 按 reference |
| 主要约束 | 带宽/热 | overlap/VRAM | trace/VRAM | 离线时间 |

“4096 灯”“60 FPS”“4.3 ms”必须绑定 visible lights、overlap、shadowed ratio、灯型分布、三角数、材质复杂度、DRS、driver 和 P50/P95/P99；它们不能单独写成产品保证。

### 14.2 三本账和调度

每个 benchmark 输出：CPU/GPU P50/P95/P99/max、VRAM resident/peak、upload bytes、dirty ranges、cluster overflow、shadow requested/resident/missed、GI rays/samples/probe hit rate、reflection hit/history acceptance、PSO miss、queue wait 和 hitch 原因。

Build 生成 compact work list，避免 Execute 遍历无影响资源；async queue 是否有效必须由 capability probe 和 telemetry 证明；DRS 先调内部分辨率和低重要性更新，再切换 provider；所有池在 Build 阶段准入，禁止 Execute 无界扩容。

### 14.3 建议资源上限

H2 的初始设计预算可采用以下数量级作为工程门槛，全部必须通过实际内容校准：

| 资源 | 初始门槛 |
| --- | ---: |
| TLAS/BLAS（约 2M 三角） | ≤130 MB |
| VSM/page shadow pool | ≤96 MB |
| Probe/cache residency | ≤64 MB |
| Light + emitter + distribution | <8 MB |
| IES/cookie/light-function atlas | ≤32 MB |
| 1/2 与 1/4 分辨率 transient | ≤40 MB |

## 15. 易用性与编辑器体验

作者只需要理解：光源类型和单位、范围/形状、贡献域、阴影意图、mobility、质量 profile、预算上限和诊断结果。

必须提供：

- `Indoor`、`Outdoor`、`Stylized`、`Showcase` 等可解释 preset；
- 单位、色域、色温、IES、cookie、source extent 的显式 inspector；
- `explain(scene/view)`，回答为何选择 provider、为何降级、影响哪些资源；
- 灯复杂度、cluster overflow、shadow page、probe residency、GI miss、历史拒绝和每灯成本视图；
- static/indirect bake 的进度、增量、diff、取消和恢复；
- capture/replay 包含 scene hash、light revision、quality、budget、seed、device generation 和失败原因；
- “缺少 canonical scene”“不支持材质”“page pool 满”“history stale”等错误不能只显示为黑屏。

## 16. 易扩展与易维护

### 16.1 新增灯型施工单

新增形状必须同时完成：

1. canonical record 和单位/范围校验；
2. GPU payload、ABI hash 和 stable table；
3. sampler、PDF、evaluate 和 contribution mask；
4. direct provider 接线；
5. shadow provider 或明确 unsupported 矩阵；
6. emitter/GI participation 或明确不参与；
7. planner capability、budget 和 explain；
8. CPU oracle、GPU intermediate、HDR image、failure injection 金样。

新增灯型只能增加数据、sampler 或 provider manifest，不能在每个 pass 增加一套私有 registry 和 if/else 分支。

### 16.2 新增 provider manifest

```text
ProviderManifest {
  id / version / owner
  required_capabilities
  supported_light_kinds
  supported_materials
  input_revisions
  output_signals
  budget_model
  fallback_chain
  known_bias
  determinism_mode
  evidence_requirements
}
```

新增 provider 必须通过 contract test、dependency/layering lint、golden image、failure injection、replay 和 benchmark gate。

### 16.3 禁止的完成方式

禁止以下行为替代真实实现：

- 用 TODO、stub、placeholder 代替生产执行；
- 用 null/software/virtual adapter 冒充 physical GPU；
- 用 shader 编译成功冒充 GPU 执行成功；
- 用 buffer 非空冒充 HDR 场景输出正确；
- 用 CPU oracle 通过冒充 GPU、图像或性能通过；
- 用默认关闭灯、阴影、GI 或有效贡献伪造性能达标；
- 用“模块存在”“函数可调用”“测试数量多”冒充端到端闭环；
- 用静默黑色、全零环境或未解释的常量 radiance 处理 miss。

## 17. 破坏性迁移方案

### 17.1 最终删除

- Bevy Light component 作为 Prism runtime lighting authority 的入口；
- render-scene/render-app 的双重 Light registry 和双写；
- 每帧 dense rebuild 作为默认上传路径；
- 没有 generation/revision/device proof 的异步缓存；
- provider 私自提交 GPU work 或写全局 lighting buffer 的路径；
- unsupported source 导致同帧有效灯全部清空的错误处理；
- world/BVH miss 默认黑色；
- 常量 triangle radiance 作为生产材质/GI 路径；
- 只为兼容旧 API 而保留的 `LightPlugin`、旧 extraction、旧 probe/area registry。

### 17.2 临时适配器

只允许单向转换：

```text
legacy ECS/component/import
        ↓
LegacyAdapter
        ↓
CanonicalLightingScene
```

禁止 canonical 回写 legacy owner，禁止新旧系统互相监听形成环。每个适配器必须有 owner、删除条件、迁移统计和截止 milestone。

### 17.3 迁移顺序

1. 固化 canonical scene、journal、identity、revision、snapshot；
2. 将旧 Bevy light 变成只读导入适配器；
3. 完成 Directional/Point/Spot 的 direct scene-to-HDR vertical slice；
4. 完成一个 shadow provider 的完整端到端 slice；
5. 完成 Rect/Disk/Capsule direct + visibility；
6. 完成 EnvironmentScene、MaterialScene 和 `RadianceQuery`；
7. 完成 GI/reflection fallback 链和 HDR image readback；
8. 删除旧 extraction、registry、plugin、双写测试和无主 fallback；
9. 将 `prism_render_scene` 的 Bevy 依赖收敛为明确适配边界；
10. 默认 profile 不再安装 Bevy Light/PBR authority。

## 18. 里程碑和退出门

| 阶段 | 交付 | 退出门 |
| --- | --- | --- |
| M0 Contract | canonical domain、单位、identity/revision、ABI、PlanDecision、BudgetLedger、capture schema | invalid input、unsupported reason、序列化和 replay schema 通过 |
| M1 GPU Scene | stable table、dirty range、retirement、submission proof、device recovery | slot reuse/ABA、stale proof、partial failure、recovery 注入通过 |
| M2 Direct | Directional/Point/Spot、HDR composition、present/EXR | physical GPU intermediate、HDR readback、CPU/GPU parity 通过 |
| M3 Shadow | 一个 shadow provider、caster epoch、allocation identity | caster/camera 移动、pool exhaustion、缺页、bias、恢复通过 |
| M4 Area | Rect/Disk/Sphere/Capsule sample/pdf/evaluate/visibility | 近场、遮挡、动态 source、高 overlap、图像误差通过 |
| M5 Radiance | MaterialScene、RadianceQuery、Emitter、Environment miss | 动态材质、纹理、实例复用、unsupported material 通过 |
| M6 GI/Reflection | screen/world/probe/env 与 SSR/world/probe/env coverage | disocclusion、roughness、漏光、历史、重复注能通过 |
| M7 Product | inspector、profile、Explain、DebugView、capture/replay、导入迁移 | 常见作者问题可定位，目标场景 P50/P95/P99 可复现 |
| M8 High-end | ReSTIR、HWRT、advanced cache、volume/offline provider | 每个 provider 独立 go/no-go，不以“已接入”退出 |
| M9 Cleanup | 删除 Bevy authority、旧 registry、双写和 placeholder | 依赖图、默认 profile、contract/image/perf gate 全绿 |

## 19. 验收矩阵

### 19.1 场景集

| 场景 | 必验内容 |
| --- | --- |
| S1 室内 Point/Spot | 单位、衰减、cone、IES、局部 shadow、动态遮挡 |
| S2 户外 Directional/Sky | 大范围 shadow、environment、曝光、无重复注能 |
| S3 8→4096 灯 | overlap、overflow、exact/stochastic、VRAM、斜率 |
| S4 Rect/Disk/Sphere/Capsule | sample/pdf、source extent、visibility、近场软影 |
| S5 Emissive-heavy | distribution update、材质纹理、visibility、reservoir identity |
| S6 动态几何/材质 | hit identity、AS reorder、history reject、非黑 miss |
| S7 镜面 roughness 阶梯 | SSR/world/probe 切换、接缝、denoise、parallax |
| S8 探针/缓存 | brick residency、leak risk、scenario、stale rejection |
| S9 大世界流送 | cell load/unload、page/brick/AS residency、lineage |
| S10 失败注入 | OOM、pool full、unsupported、device lost、stale proof |
| S11 透明/薄几何 | 明确支持范围、fallback、无伪完成 |
| S12 确定性 | fixed seed、排序、CPU/GPU 容差、跨帧稳定 |

### 19.2 证据层级

```text
数学单测
→ 数据契约
→ CPU oracle
→ shader/graph recording
→ physical GPU intermediate
→ HDR image
→ failure injection
→ performance P50/P95/P99
→ editor UX / Explain
```

证据状态使用 `PASS`、`FAIL`、`SKIPPED`、`SKIP_UNVERIFIED`、`BLOCKED`。缺少上层证据只能标记 partial，不能用下层通过替代上层通过。

## 20. 当前工作树状态（2026-10-20）

### 20.1 已有基础

- `prism_render_architecture` 已有 lighting contract、recipe/trace/cache/replay、reference 和多种 provider 资产；
- `prism_render_app` 已有 native lighting scene、GPU lighting 表、上传/事务和部分 presentation 相关路径；
- `prism_render_scene` 已有 canonical lighting adapter、cluster、area light、IBL、shadow、virtual shadow、world ReSTIR 等模块；
- Directional、Point、Spot、Rect、Disk、Capsule 已有不同程度的 canonical payload、增量上传或面积光测试基础；
- 已补充 unsupported type、单位、flags、layer mask、spot profile 等显式 rejection；
- Prism 的抽取路径不直接读取 Bevy Light 组件，但当前 app 与 scene 仍有各自独立的 canonical light resource；尚未证明存在唯一 authority 和生产同步桥；
- native lighting validation 已覆盖部分 CPU oracle、graph recording、真实 GPU buffer/readback、failure/recovery 边界。

### 20.2 当前不能宣称完成

- `canonical scene → planner → direct/shadow/GI/reflection → HDR composition → exposure/output → present/EXR` 尚未由稳定的真实 GPU HDR evidence 完整证明；
- EnvironmentScene、emissive distribution、完整 hit-time `RadianceQuery` 和 world miss fallback 仍需闭环；
- Rect/Disk/Sphere/Capsule 不能因为 CPU oracle、固定样本或 ABI parity 就宣称生产级阴影、many-light、GI participation 和图像验收完成；
- VSM 仍需跨帧稳定 slot allocator、完整 dynamic caster epoch producer、全灯型 provider 矩阵和真实 page residency 证据；
- `prism_render_scene` 仍保留 Bevy ECS、RenderApp/RenderGraph/RenderQueue、Bevy PBR/StandardMaterial 等适配依赖；删除 Bevy Light authority 不等于全仓 Prism runtime 迁移完成；
- `prism_render_app` 与 `prism_render_scene` 目前各自维护 canonical light resource；app 的 lighting storage-buffer 输出尚无证据写入其最终 HDR/present 颜色目标；
- Prism opaque 路径仍与 Bevy PBR plugin/material queue 共存，production composition 必须定义谁拥有 opaque 光照写入，避免重复着色或隐式双 authority；
- 当前工作树存在大量并行模块修改，未审查的改动不能自动视为已通过本方案验收。

### 20.3 当前构建阻塞

截至本文本次更新，`cargo check -p prism_render_scene --lib` 首轮发现的 CommandBuffer re-export、water history 可见性、HashSet 推断和 SceneFabricHistory 参数等错误均已修复；随后发现并修复 composition error enum 的 `missing_docs` 硬错误。第二轮复核目前等待共享 Cargo build lock，尚未返回最终状态，因此本 crate 的最终 `cargo check` 仍记为 `BLOCKED/UNVERIFIED`，不能记为 PASS。最近一次完整诊断阶段还报告 87 条 warning；它们横跨多个当前修改模块，需在 build 复核后分类清理，不应与硬错误混为一谈。

这些阻塞属于当前工作树的实现状态，不改变本文的目标架构；但在构建恢复前，所有真实 GPU、HDR 和性能结论都必须保持 `BLOCKED` 或 `PARTIAL`。

## 21. 最终决策清单

1. Prism Light System = `CanonicalLightingScene + GPU Scene + Planner + Provider + HDR Composition + Evidence`。
2. Light、Emitter、Environment、Geometry、Material、View 各自拥有权威数据，通过 identity/revision/capability/budget/explain 协作。
3. Exact、stochastic、shadow map、virtual page、ray、screen、probe、environment 都是 provider，不是作者语义。
4. `RadianceQuery`、稳定身份、事务 proof、BudgetLedger、PlanDecision 和 Explain 是 GI/反射/虚拟阴影的前置条件。
5. 任何 miss、unsupported、budget rejection 都必须有非黑、单调、可解释的 fallback 或明确拒绝。
6. 所有 AAA、实时、高性能结论必须绑定硬件、内容、分辨率、profile、seed 和 P50/P95/P99 证据。
7. 允许破坏性更新：删除旧 Light API、旧 registry、双写、Bevy authority 和无 owner 的兼容层。
8. 只有完成 source audit、contract test、HDR image evidence、GPU/perf evidence 后，才允许更新完成状态。

## 附录 A：核心接口示意

```rust
trait LightingScene {
    fn mutate(&mut self, command: LightingCommand) -> Result<SceneDelta, LightingError>;
    fn snapshot(&self) -> CanonicalLightingSnapshot;
}

trait RadianceQuery {
    fn query(&self, hit: SurfaceHit) -> Result<SurfaceRadiance, QueryError>;
}

trait LightingProvider {
    fn plan(&self, ctx: &LightingPlanContext) -> PlanDecision<ProviderPlan>;
    fn execute(&self, plan: &ProviderPlan, frame: &mut FrameExecution)
        -> ExecutionReceipt;
}

trait ShadowProvider {
    fn plan(&self, request: ShadowRequest, ctx: &ViewContext)
        -> PlanDecision<ShadowPlan>;
}

trait EvidenceSink {
    fn record(&mut self, receipt: FrameReceipt);
    fn capture(&mut self, request: CaptureRequest) -> Result<CaptureId, CaptureError>;
}
```

接口名称可以调整；不可变的是 ownership、identity、revision、budget、fallback、diagnostic、submission proof 和 replay 规则。

## 附录 B：关联文档

- `docs/prism_lighting_gi_design_v3_zh.md`：早期 v3.1 灯型、预算和性能模型来源；若不存在，以本文件为准。
- `docs/prism_lighting_gi_design_v5_implementation_zh.md`：实施规格和当前代码边界。
- `docs/prism_lighting_gi_design_v10_execution_zh.md`：执行版架构。
- `docs/prism_lighting_gi_design_v11_aaa_zh.md`：v11 目标架构。
- `docs/prism_native_lighting_validation_acceptance_zh.md`：验证、真实 GPU、recovery 和 skip/fail-close 口径。

本文件是 v3.1、v5、v10、v11 和当前工作树状态的统一入口。旧文档保留作历史决策和迁移参考；新设计、实现和验收以 v12 为准。
