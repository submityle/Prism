# Prism 原生 AAA 光照系统 v14：生产化架构与实施合同

> 版本：v14.0  
> 日期：2026-10-21  
> 适用范围：Prism 原生实时渲染、编辑器视口、离屏/HDR capture、自动质量控制与可选 reference renderer。  
> 状态：架构设计与实施合同；不是整体完成声明。  
> 迁移策略：允许破坏性 API、ABI、资源布局和生产路径更新；不保留长期旧 Light API、旧 registry、旧 Bevy lighting 兼容层。  
> 源码口径：本文的“当前实现”只指本工作树可定位到的代码和已记录验证；设计目标、预算和路线不等于已实现能力。

## 目录

- [0. 结论](#0-结论)
- [1. 目标与边界](#1-目标与边界)
- [2. 产品原则](#2-产品原则)
- [3. 总体架构](#3-总体架构)
- [4. Canonical 数据模型](#4-canonical-数据模型)
- [5. 灯系统](#5-灯系统)
- [6. 阴影系统](#6-阴影系统)
- [7. GI 与反射](#7-gi-与反射)
- [8. 真正的 HDR 像素链](#8-真正的-hdr-像素链)
- [9. GPU Scene 与资源事务](#9-gpu-scene-与资源事务)
- [10. Per-view Planner 与质量控制](#10-per-view-planner-与质量控制)
- [11. 性能、显存与效果预算](#11-性能显存与效果预算)
- [12. 易用性与诊断](#12-易用性与诊断)
- [13. 扩展性与维护性](#13-扩展性与维护性)
- [14. 破坏性迁移方案](#14-破坏性迁移方案)
- [15. 当前代码状态](#15-当前代码状态)
- [16. 垂直切片与验收门](#16-垂直切片与验收门)
- [17. 实施路线](#17-实施路线)
- [附录 A：核心接口](#附录-a核心接口)
- [附录 B：性能证据包](#附录-b性能证据包)

## 0. 结论

Prism 缺的不是一个简单的 `LightPlugin`，而是一条从作者输入到最终显示像素的完整光照生产链。v14 将光照系统定义为以下七个必须同时成立的部分：

```text
Authoring / Import / Runtime Command
        ↓ 校验、单位归一化、journal
Canonical Scene
  Light | Emitter | Environment | Geometry | Material | View
        ↓ revision、delta、stable identity
GPU Scene Reconcile
  slot/generation | dirty range | residency | submission proof
        ↓ 每视图不可变计划
Lighting Planner
  direct | shadow | GI | reflection | output
        ↓ 图执行
Scene-linear HDR Color Target (Rgba16Float)
        ↓ exposure | display transform
Present / HDR Capture / EXR
        ↓
Explain | Telemetry | Replay | Benchmark Evidence
```

### 0.1 v14 的硬性结论

1. `CanonicalLightingScene` 是光照语义的唯一 CPU 权威。
2. `GpuScene` 是 GPU slot、generation、上传、设备代际、资源回收和提交证明的唯一权威。
3. `LightingPlanner` 是每视图 provider、质量档、预算和降级的唯一入口。
4. Opaque surface 的最终照明只能有一个写入 owner，不能由 mesh pass、lighting compute 和 composite 同时声称最终写入权。
5. Direct、shadow、GI、reflection、environment 和 material 分域拥有自己的数据与历史；共享 identity、revision、capability 和预算，不共享未定义的 estimator 或 history。
6. 最终完成证据必须来自 HDR color target 的真实 readback 或 display transform 的真实输入，storage buffer parity 不能代替像素证据。
7. 不允许通过白色 mesh、identity camera、零光源、黑色环境或静默 skip 把未接通的路径伪装成完成。
8. Bevy 适配器只能作为迁移期单向输入边界；生产照明路径最终不得依赖 `bevy_*`。
9. “AAA”只在指定设备、内容、分辨率、质量档、版本和 P50/P95/P99 证据齐备后成立。

### 0.2 和 v3.1 / v13 的关系

v3.1 提供了灯型、VSM、GI、SSR、预算、验收和工具体验的完整算法愿景；v13 将其收敛为 ownership、provider、预算和证据合同；v14 进一步把“能否进入生产”的判定具体化为：

| 维度 | v14 决策 |
| --- | --- |
| 正确性 | 先打通真实 mesh surface/material 到 HDR target 的垂直切片，再扩展 GI 与 many-light |
| 性能 | 预算在 planner 阶段准入，执行阶段不得临时超配；同时记录关键路径和尾延迟 |
| 效果 | scene-linear 中间值、单次曝光、信号独立历史、明确误差阈值 |
| 易用性 | 作者表达意图，系统解释实际 provider、成本、降级和不支持原因 |
| 扩展性 | 灯型、shadow provider、GI recipe、backend 通过 manifest/contract 接入 |
| 维护性 | 单一 owner、版本化 ABI、显式生命周期、可重放证据、最终移除 Bevy 生产依赖 |

## 1. 目标与边界

### 1.1 最终产品目标

最终系统应覆盖：

- Directional、Point、Spot、Rect、Disk、Sphere、Capsule/Line、Sky、Emissive 的分阶段支持。
- 直接光、动态/缓存阴影、环境照明、低频 GI、屏幕空间 GI、世界表示 GI、SSR、环境反射和可选追迹反射。
- opaque、透明、毛发、水和体积拥有独立 consumer contract；不把 opaque 假设隐式套给其它材质。
- Static、Stationary、Dynamic mobility，且 mobility 只表达更新意图，不硬编码成某种 shadow/GI 算法。
- 运行时、编辑器、离屏 capture 使用相同 canonical scene 语义。
- 自动 LOD、DRS、历史管理、页预算、显存预算、质量解释和故障恢复。

### 1.2 首发生产基线

首个可发布基线只承诺一个能完整证明的垂直切片：

```text
封闭测试房 + opaque mesh + position/normal + material albedo
        ↓
Directional / Point / Spot + Environment
        ↓
直接光 + conventional shadow provider
        ↓
Rgba16Float HDR target
        ↓
曝光 + display transform + present/capture
```

Rect、Disk、Emissive、world GI、SSR、virtual shadow page、ReSTIR 和高端追迹都必须在独立 provider 门通过后加入，不能因为类型或枚举存在就算支持。

### 1.3 非目标

本版本不承诺：

- 任意设备、任意场景下固定 60 FPS。
- 任意灯数下固定画质或固定 GPU 时间。
- 全透明、毛发、水、云和体积的统一高质量 GI。
- 跨 GPU bitwise 一致。
- 仅靠 screen-space 技术实现完整屏外 GI/反射。
- 长期维护新旧两套光照 API、registry 或材质 owner。

## 2. 产品原则

### 2.1 作者意图与实现解耦

作者提交的是强类型意图：单位、形状、颜色、变换、源尺寸、贡献域、mobility、shadow intent 和质量请求。作者不直接选择 GPU recipe，也不直接写 GPU buffer。

```rust
let light = lighting.add(
    Light::spot()
        .intensity(Candela::new(1_500.0))
        .inner_angle(Degrees::new(18.0))
        .outer_angle(Degrees::new(25.0))
        .shadow(ShadowRequest::Auto)
        .mobility(Mobility::Dynamic),
)?;
```

`Auto` 不是静默 fallback。plan resolve 后必须能查询：实际 provider、使用的质量档、预算、降级原因、输入 revision 和预计成本。

### 2.2 正确性优先于算法数量

顺序固定为：

1. 真实几何、法线、材质、视图和 direct 到 HDR target。
2. conventional shadow 与遮挡验证。
3. environment、probe 和 screen-space GI。
4. area/emissive/many-light。
5. virtual shadow、world tracing、ReSTIR 和高端反射。

任何后续优化不得掩盖前一层 owner、资源或输出路径未闭环的问题。

### 2.3 资源上限单调，画质不作虚假单调承诺

质量档切换必须保证：资源上限、样本数、页数、历史长度和分辨率按照冻结规则单调变化；随机估计器的单帧逐像素亮度不要求绝对单调。降级结果必须记录统计误差、时域误差和能量偏差。

### 2.4 所有跨阶段数据都带版本

每个 GPU 输入至少携带：`scene_id`、稳定 handle、generation、domain revision、field epoch、device generation、ABI version 和 producer proof。执行阶段只接受 plan 指定 revision 的资源。

## 3. 总体架构

### 3.1 六个 canonical domain

```text
LightScene        解析灯、面积灯、方向灯、影响范围和贡献域
EmitterScene      发光几何、面积、辐亮度、采样分布
EnvironmentScene  天空、环境贴图、大气、miss radiance
GeometryScene     instance、mesh、可见性、BLAS/TLAS 或软件 BVH 表示
MaterialScene     material IR、纹理、BSDF、opacity、displacement、emission
ViewScene         camera、输出、曝光、质量、历史和 capture policy
```

每个 domain 独立拥有：数据、revision、delta、序列化、错误、dirty range、历史失效规则和 replay 输入。

### 3.2 唯一数据流

```text
Command / Import
  → validate + normalize
  → canonical mutation + journal
  → domain delta + revision
  → GPU reconcile transaction
  → visible surface/material extraction
  → per-view immutable plan
  → graph compile
  → direct/shadow/GI/reflection execute
  → HDR composite
  → output transform/present/capture
  → receipt + telemetry + evidence
```

禁止以下双 owner：

- render-scene 和 render-app 同时接受 Light 写入；
- material pass 和 lighting composite 同时写最终 direct lighting；
- provider 私自创建无预算缓存、提交 command buffer 或吞掉 graph error；
- output 层再次 gamma、曝光或把 display-referred 值写回 scene-linear target。

### 3.3 模块边界

| 层 | 负责 | 不负责 |
| --- | --- | --- |
| contracts/schema | 单位、ID、snapshot/delta、ABI、错误 | device、GPU 分配 |
| canonical store | 作者数据、校验、journal、资产引用 | shader、预算执行 |
| GPU scene | slot、generation、上传、residency、completion | 作者语义、provider 选择 |
| planner | capability、质量、预算、provider、退化 | shader 细节、临时改计划 |
| providers | direct、shadow、GI、reflection 的 graph 工作 | scene 真值、独立曝光 |
| composition/output | HDR 合成、曝光、display transform、capture | 重新选择光照算法 |
| diagnostics | explain、telemetry、replay、benchmark | 改变渲染结果 |

依赖方向固定为：

```text
contracts ← canonical scene ← GPU scene/query ← providers ← composition/output
```

## 4. Canonical 数据模型

### 4.1 稳定身份

```rust
struct StableHandle {
    index: u32,
    generation: u32,
}

struct DomainRevision {
    scene: u64,
    field_epoch: u64,
    device_generation: u64,
}
```

数组 slot 可以复用，但旧 generation 永远不能解析到新对象。删除采用 retirement：等待 GPU completion、移除引用、写入 tombstone、再回收 slot。

### 4.2 强类型物理量

`Lux`、`Lumens`、`Candela`、`Nits`、`Radiance`、`EV100` 使用不同类型。禁止用一个裸 `f32` 在上下文中隐式代表不同物理量。

Photometric 到 RGB/radiometric 的转换必须记录工作色域、参考白和光谱近似。不能声称只凭 lux/candela 无损恢复光谱功率。EV100 只用于曝光控制，不能当作灯能量。

### 4.3 Delta 与失效

每条 delta 至少记录：domain、subject handle、变更字段、旧/新 revision、影响范围、需要上传的资源、需要失效的历史、provider 依赖和 replay key。

典型失效规则：

| 变更 | Direct | Shadow | GI/Reflection |
| --- | --- | --- | --- |
| 强度/颜色 | 更新 radiometry/重要性 | 通常保留深度 | 受影响 radiance/history 标 stale |
| 灯变换/方向 | 更新 bounds/list | 投影范围失效 | 受影响区域重照明 |
| 源尺寸/IES/cookie | 更新采样/PDF | 视投影/opacity 决定 | sampling 与 cache 失效 |
| caster 变形 | visibility 重评估 | 受影响页失效 | 局部 history/cache 失效 |
| material normal/roughness | BSDF 失效 | 通常保留深度 | query/history 失效 |
| camera cut/DRS | 重建 view plan | 重新请求视图页 | 独立 history reset/resample |
| device generation | 全部 GPU handle 失效 | 全部重建 | GPU history/cache 重建 |

## 5. 灯系统

### 5.1 灯型与 provider 矩阵

| 灯型 | Direct | Shadow | GI/Emitter | 首发策略 |
| --- | --- | --- | --- | --- |
| Directional | 必须 | conventional map | environment 关联 | P0 |
| Point | 必须 | cubemap/conventional | 可选采样源 | P0 |
| Spot | 必须 | perspective map | 可选采样源 | P0 |
| Rect/Disk | LTC/area | provider-specific | emitter sampling | P2 |
| Sphere/Capsule | area approximation | contact/trace | emitter sampling | P2 |
| Sky/Environment | miss/IBL | 不作为普通 shadow light | probe/world query | P0 |
| Emissive mesh | material emission | 不复用 analytic shadow | area/many-light | P3 |

“支持”必须拆成 direct、shadow、GI、reflection、材质、平台和质量档矩阵。枚举存在、CPU 函数存在或 provider 注册存在都不能代替生产支持证据。

### 5.2 GPU 表示

解析灯使用公共表加按形状 SoA 表：

```text
LightCommon[]  identity、flags、radiometry、bounds、revision
Directional[]  direction、illuminance、color、shadow link
Punctual[]     position、range、intensity、cone、profile
Area[]         transform、extent、radiance、two-sided、sampling key
LightIndexTree importance / spatial candidates / streaming metadata
```

每帧不重建整个灯表。只上传 dirty ranges，并在提交 receipt 中记录 bytes、range、revision 和 completion token。

### 5.3 直接光计算

opaque direct provider 必须在 shader 中接收真实 world-space position、normal、material albedo/BSDF 和真实 view transform；不得继续使用 clip-space position 或固定白色 fragment 作为生产默认路径。

基础 Lambert 合同：

```text
L_o = f_d * (L_direct + L_environment + L_indirect)
f_d = albedo / π
```

高端 provider 可使用 GGX、LTC、importance sampling 或 reservoir，但必须明确测度、PDF、visibility、bias 和历史规则。

### 5.4 Many-light

灯数增长时同时计入：灯表更新、bounds/cluster、candidate 生成、采样分布、visibility、重建和历史失效。不能只测 fragment loop 并宣称“4096 灯恒定成本”。

## 6. 阴影系统

### 6.1 分层 provider

1. `ConventionalShadowProvider`：首发正确性基线，支持方向光、点光和聚光的明确投影模型。
2. `CachedShadowProvider`：静态 caster、局部更新和稳定页复用。
3. `VirtualPageShadowProvider`：按需页、page table、resident/miss、局部失效和池耗尽策略。
4. `RayVisibilityProvider`：设备 capability 和实测预算准入，不作为所有档位默认依赖。

虚拟页不是所有灯型的通用正确性前置。每种灯型必须说明投影模型、页坐标、过滤方式、bias、失效和缺页行为。

### 6.2 页预算

planner 先分配 shadow budget，再生成 page request。页池满时只能执行已批准的质量策略：降低分辨率、缩短更新范围、关闭低重要性灯影或拒绝该 recipe，并记录结构化 reason。不得静默截断或读取未初始化页。

### 6.3 动态正确性

必须验证：移动 blocker、改变 caster topology、改变 alpha/opacity、改变灯强度、相机移动和 resize 后，相关页正确失效，无关页不被全部清空。

## 7. GI 与反射

### 7.1 GI provider 链

```text
屏幕空间细节
    ↓ miss / 屏外补全
Probe / world representation
    ↓ 高端设备
Software BVH / Hardware RT
```

屏幕空间只能提供屏内覆盖，不能把 miss 隐藏成完整 GI。probe 只负责其声明的低频辐射与场景范围；world tracing 负责更高质量屏外覆盖。三者可以共享 GeometryScene 和 MaterialScene 查询，但不能共享没有定义过的 history 或 estimator。

### 7.2 Probe volume

Probe brick 必须带：cell identity、scenario、irradiance payload、sky visibility、bake/runtime provenance、revision 和 residency。边界、漏光、陈旧 scenario、缺页和动态物体覆盖都要可视化。

### 7.3 Screen GI 与反射

SSR/屏幕 GI 必须声明：步长、最大距离、厚度、半分辨率策略、miss 行为、重投影、历史钳制、法线/深度 disocclusion 和最终 fallback。屏外、遮挡和历史失效不能返回未初始化值。

反射降级顺序：

```text
追迹反射 → SSR + environment/probe → filtered environment → material-defined fallback
```

粗糙度、Fresnel、多散射补偿和 prefilter 必须在同一 scene-linear 能量合同下解释。

### 7.4 Emitter 与采样

发光几何进入 `EmitterScene` 后，面积、法线、辐亮度、双面策略和 PDF 必须统一定义。Delta 光源和连续面积分布不在没有共同测度的情况下强行混入 MIS。

## 8. 真正的 HDR 像素链

### 8.1 唯一生产链

```text
Depth / Surface
  → direct + shadow
  → GI / reflection signal
  → fullscreen raster composite
  → Rgba16Float scene-linear target
  → exposure
  → display transform
  → present / HDR capture / EXR
```

lighting compute 可以写 storage buffer 作为中间结果，但最终 composite 必须写入真正的 HDR color attachment，或者明确使用合法 storage texture 作为最终纹理，并被 display transform/present 消费。

### 8.2 禁止事项

- 用 storage buffer readback 代替最终颜色 target readback。
- 将中间 lighting buffer 直接当作已显示画面。
- 在 direct、GI、composite、display transform 多处重复曝光或 gamma。
- 在没有真实 material/view/light binding 时用零资源默默继续绘制。
- 以白色 mesh 或 identity camera 作为默认生产路径。

### 8.3 HDR 验收

固定场景写入 scene-linear `0.25 / 1 / 4 / 16`，在曝光前读取 HDR target；`4/16` 必须保留，不得提前 clamp 到 `1`。之后用负曝光或 display transform 验证输出变化。必须同时保存 HDR readback、display output 和配置。

## 9. GPU Scene 与资源事务

### 9.1 资源 owner

每种 GPU 资源只有一个 owner：

| 资源 | owner |
| --- | --- |
| frame/view uniform | View/GPU scene owner |
| canonical light buffers | Lighting GPU scene |
| material buffers | Material GPU scene |
| shadow pages | Shadow provider + page allocator |
| GI probes/history | GI provider |
| HDR target | Frame/output owner |
| bind-group cache | render resource owner |
| PSO/shader module | pipeline cache |

provider 通过资源句柄读取，不直接拥有 canonical 数据，也不在 draw 时随意创建长期资源。

### 9.2 Frame/material/light binding

forward-lit group 需要显式声明：

```text
group 0 / binding 0  frame/view uniform
group 0 / binding 1  directional-light storage
group 0 / binding 2  punctual-light storage
group 0 / binding 3  area-light storage
group 0 / binding 4  material uniform
```

绑定缺失必须返回稳定 rejection。material、view、light buffer 的 revision 改变时，bind-group cache 必须按 key 重建或更新，不能复用旧 generation。

### 9.3 提交与恢复

资源更新采用：

```text
prepare → validate → encode → submit → completion → publish committed revision
```

上传失败、device lost、OOM、generation 不匹配时，不发布新的 committed revision。设备恢复后重建所有 device-scoped 资源、PSO、bind groups、history 和 output target，再从 canonical snapshot 重新 reconcile。

## 10. Per-view Planner 与质量控制

### 10.1 质量档是运行时数据

质量档不写死在 shader 常量中。每个 view 解析：设备 capability、分辨率、目标帧时间、显存 ceiling、场景复杂度、用户偏好和 provider manifest。

```text
QualityProfile
  resolution_scale
  max_visible_lights
  direct_samples
  shadow_pages / shadow_resolution
  gi_provider / gi_scale / rays_per_pixel
  reflection_provider / history_length
  transient_vram_ceiling
  deterministic / capture policy
```

### 10.2 准入顺序

1. capability：格式、storage texture、深度、shader feature、queue、RT。
2. correctness：provider ABI、输入资源、材质与灯型支持矩阵。
3. memory：常驻、瞬态、history、页池和 staging。
4. critical path：CPU/GPU 时间、尾延迟和队列等待。
5. quality：误差、噪声、拖影、漏光和屏外覆盖。
6. explain：把实际选择和被拒原因写入 frame plan。

### 10.3 降级顺序

```text
DRS / transient scale
  → GI/reflection resolution and update rate
  → shadow page count / filter quality
  → direct candidate/sample budget
  → provider downgrade
  → reject optional pass with explicit reason
```

降级不得静默改变物理单位、重复注能、读取旧 generation 或把失败变成黑色/白色结果。

## 11. 性能、显存与效果预算

### 11.1 三本账

每帧同时记录：

1. `GPU time ledger`：critical path、pass P50/P95/P99、queue wait、barrier、dispatch/draw。
2. `CPU/update ledger`：extract、delta、sorting、planning、encode、上传 bytes 和分配次数。
3. `Memory ledger`：常驻、瞬态峰值、history、页池、staging、PSO、回收延迟。

只报告一个平均 GPU 时间不算性能证据。

### 11.2 初始目标，不是无条件承诺

以 1440p/60 为默认 benchmark 起点：

| 项目 | 初始目标 |
| --- | ---: |
| CPU render preparation | ≤ 1.0 ms P50 |
| command encode | ≤ 0.5 ms P50 |
| direct + shadow | ≤ 3.0 ms P50 |
| GI + reflection | ≤ 4.0 ms P50 |
| composite/output | ≤ 1.0 ms P50 |
| transient VRAM | 由 profile ceiling 决定，超限必须降级或拒绝 |

目标必须在固定场景、固定设备、固定驱动、固定分辨率和固定 warm-up 规则下测量。不得将目标文字当作当前性能结果。

### 11.3 many-light 性能模型

分别测量灯数、可见 overlap、屏幕覆盖、动态更新比例和材质复杂度：

```text
T_frame = T_scene_update
        + T_visibility
        + T_shadow_update
        + T_direct_candidates
        + T_surface_shading
        + T_GI/reflection
        + T_reconstruction
        + T_output
```

每项都要记录 P50/P95/P99；4096 灯只在与场景、覆盖率、provider、分辨率绑定的 benchmark 中声明。

## 12. 易用性与诊断

### 12.1 作者体验

默认 API 应满足：

```rust
app.add_plugins(LightingPlugin::new(LightingProfile::auto()));
lighting.add_light(Light::point().intensity(Candela::new(800.0)));
let report = lighting.explain(&view);
```

编辑器必须能显示：

- 灯的真实单位、颜色空间、mobility、贡献域和 provider。
- 每个灯的直接光、阴影、GI、反射成本。
- 阴影页占用、probe residency、history confidence、GI miss 和 overflow。
- 当前 profile、实际 provider、降级原因和恢复状态。
- 不支持输入的稳定错误、字段路径、修复建议和原始值/修正值。

### 12.2 必备调试视图

`Light complexity`、`shadow page`、`cluster/candidate`、`surface/material AOV`、`normal/depth/motion`、`GI confidence`、`reflection fallback`、`HDR false color`、`history age` 和 `generation` 都必须可以单独查看。

### 12.3 可重放

问题报告至少保存：源码/patch hash、scene hash、canonical snapshot、command sequence、provider/schema version、quality profile、seed、camera path、GPU/driver、frame plan、telemetry 和输出图像/AOV。

## 13. 扩展性与维护性

### 13.1 新灯型准入清单

新增灯型必须同时提供：

1. canonical schema 与单位合同；
2. CPU reference/oracle；
3. GPU ABI 与布局哈希；
4. bounds/culling/sampling；
5. direct provider；
6. shadow provider 或明确声明不支持；
7. GI/emitter provider 或明确声明不支持；
8. quality/budget item；
9. import/export；
10. golden scene、图像误差、恢复和性能证据。

### 13.2 新 provider 准入清单

provider manifest 至少包含 capability、输入 revision、读写资源、预算、history schema、失败 reason、降级关系、shader/ABI digest 和 benchmark profile。provider 不允许调用另一个 provider 的私有资源或绕过 planner。

### 13.3 CI 规则

- `deny(missing_docs)` 和 ABI layout hash 校验。
- lighting crate 不得新增 `bevy_*` 依赖。
- graph read/write 与 bind-group buffer usage 一致。
- 所有 required GPU job 缺少物理 GPU 时失败，不是 skip pass。
- `git diff --check`、格式、单元、契约、recovery、最终像素 readback、性能证据索引进入发布门。
- 任何 `TODO`、`placeholder`、`stand-in`、静默 fallback 出现在生产光照路径时禁止发布。

## 14. 破坏性迁移方案

### 14.1 迁移目标

最终删除：

- `SpineMeshPipeline` 作为生产默认路径；
- 旧 Light registry 与双写 adapter；
- lighting compute 直接以 storage buffer 作为最终输出的路径；
- `prism_render_scene` 对 `bevy_*` 的生产依赖；
- 依赖 Bevy render graph/PBR/light 的旧生产组合。

### 14.2 阶段

| 阶段 | 操作 | 删除门 |
| --- | --- | --- |
| M0 | 冻结 canonical contracts、ABI、错误和 revision | contract tests 通过 |
| M1 | 接通真实 surface/material/view/light binding | final HDR pixel test 通过 |
| M2 | 默认路径切换到 forward-lit provider | recovery 与材质变化测试通过 |
| M3 | composite 改为 HDR color attachment | display/present/readback 闭环通过 |
| M4 | conventional shadow 进入默认 profile | 遮挡、移动 blocker、resize 通过 |
| M5 | GI/reflection provider 按矩阵接入 | 每 provider 有独立证据 |
| M6 | 清理旧 registry、旧 spine、Bevy render 依赖 | workspace dependency audit 通过 |
| M7 | many-light、virtual page、world trace、reference | 各自 provider gate 通过 |

### 14.3 允许的临时 adapter

迁移期可以保留单向导入 adapter：旧数据 → canonical command。adapter 不得反向写旧 registry，不得参与 shader 执行，不得成为 runtime 默认 owner。新 API 稳定后删除 adapter 和其测试夹具。

## 15. 当前代码状态

本节是 2026-10-21 当前工作树的源码核对，不是整体完成声明。工作树包含大量并行未提交改动，因此结论以符号、调用路径和测试记录为准，不能把 HEAD 当作完整快照。

### 15.1 已形成的切片

| 能力 | 当前状态 | 位置/证据 |
| --- | --- | --- |
| draw bind-group contract | 已有切片 | `pkg/prism_render_app/src/draw.rs` |
| queue/phase binding identity | 已有切片 | `pkg/prism_render_app/src/queue.rs` |
| frame graph bind-group buffer dependency | 已有切片 | `pkg/prism_render_app/src/frame.rs` |
| forward-lit provider-side ABI | 已有切片 | `pkg/prism_render_app/src/forward_lit.rs`、`mesh_pipelines.rs` |
| 真实 position/normal shader contract | 已有 provider 代码 | `forward_lit.rs` |
| Rgba16Float target declaration | 已存在 | `frame.rs` 的 `RENDER_TARGET_FORMAT` |
| 定向 binding/descriptor 测试 | 已记录通过 | 并行切片测试记录 |

这些内容只能证明接口和局部切片存在，不能证明主线程已经创建并绑定真实 frame/material/light resources，也不能证明最终像素链已经闭环。

### 15.2 当前硬缺口

| 优先级 | 缺口 | 当前风险 | 关闭条件 |
| --- | --- | --- | --- |
| P0 | 默认仍是 `MeshPipelines::spine(...)` | 生产 mesh 仍可能走固定白色 stand-in | 默认改为 forward-lit，且缺资源显式拒绝 |
| P0 | forward-lit binding 尚未由真实 owner 创建/绑定 | provider ABI 存在但不执行 | frame/light/material/view bind group 真实进入 draw |
| P0 | lighting composite 仍写 storage buffer | 结果可能不进入最终 HDR target | fullscreen raster 或合法 storage texture 写入 target |
| P0 | material 颜色未贯通 queue → GPU | 改材质可能不改变最终像素 | material revision/cache/readback 测试通过 |
| P0 | view matrix owner 未贯通 | identity view 不能作为完成证据 | camera/resize/DRS 改变真实覆盖 |
| P0 | 物理 GPU 最终像素证据不足 | 无法证明 position/normal/light/HDR/present | `PRISM_REQUIRE_GPU=1` required job 通过 |
| P1 | recovery 重建 forward-lit 资源不足 | device loss 后可能丢失 binding/history | generation/rebuild/readback 通过 |
| P1 | shadow 真实遮挡闭环不足 | 阴影能力不可发布 | blocker、移动、缺页、失效测试通过 |
| P1 | canonical owner 仍需收敛 | render-scene/render-app 可能双写 | 单向 authority 与删除门通过 |
| P1 | Bevy 依赖仍存在 | 最终生产路径不满足退出目标 | workspace dependency/source audit 通过 |

### 15.3 现阶段允许的准确表述

可以称为：

- “照明系统架构和部分 GPU binding/provider 切片已建立”；
- “已有真实 forward-lit provider-side contract”；
- “正在打通真实 HDR 像素闭环”。

不能称为：

- “AAA 光照系统已完成”；
- “最终 HDR 像素链已验证”；
- “Bevy 已移除”；
- “已通过物理 GPU 最终图像验收”。

## 16. 垂直切片与验收门

### 16.1 P0：真实 opaque forward-lit 像素闭环

场景：封闭房间、一个三角形/网格、真实 position/normal、一个 material、Directional/Point/Spot、一个 view。

必须验证：

1. position 和 normal 改变会改变 fragment 结果；
2. albedo 改变会改变最终 HDR 像素；
3. light 强度、颜色、方向、位置和 cone 会改变最终 HDR 像素；
4. camera matrix、resize、depth occlusion 会改变覆盖关系；
5. HDR 值大于 `1` 在曝光前 target 中保留；
6. display transform 读取的确实是最终 HDR target；
7. readback 来源是 target，而不是 lighting storage buffer；
8. 缺 position、normal、material、frame 或 light binding 时稳定失败；
9. device recovery 后 generation 更新、资源重建、最终像素正确；
10. `PRISM_REQUIRE_GPU=1` 时缺 GPU 必须失败。

### 16.2 P1：基础阴影

必须验证：

- blocker 在灯与表面之间产生遮挡；
- blocker 移动后只失效受影响 shadow resources；
- 相机移动、resize、light change 不读取旧 generation；
- 点光六面/聚光投影的边界符合 provider 合同；
- shadow pool 满时有明确降级和 telemetry；
- 关闭 shadow 后能量变化符合声明，不把 fallback 当作错误成功。

### 16.3 P2：GI/反射 provider

每个 provider 独立交付：

- input/output schema；
- CPU reference 或解析 oracle；
- scene-linear image diff；
- motion/disocclusion/history；
- 屏外、缺页、材质不支持和设备不支持；
- P50/P95/P99、VRAM、overflow、miss 和降级证据。

### 16.4 发布门

只有以下条件全部满足，才允许称为某个 profile/provider 组合“已准入”：

```text
正确性 + 最终 HDR 像素 + recovery
        + capability/ABI
        + 显存/时间预算
        + 质量误差/时域稳定
        + 可解释降级
        + 可重放证据
```

整体系统完成还需要：P0–P8 范围的支持矩阵完整、旧生产路径删除、Bevy 生产依赖清零和作者工作流通过。

## 17. 实施路线

### 17.1 推荐顺序

1. 收敛 canonical scene owner、frame/view owner 和 material owner。
2. 建立 forward-lit frame/light/material bind-group cache。
3. queue 阶段把真实 material binding 传入 phase/draw。
4. 默认切换到 forward-lit provider，删除生产路径对 Spine 的依赖。
5. 将 lighting composite 改成真正写 HDR color target。
6. 加入最终 target readback 和物理 GPU required tests。
7. 完成 recovery、resize、generation、stale resource 和 device loss。
8. 接入 conventional shadow 与真实遮挡测试。
9. 以 provider contract 接入 probe/SSR/world GI/reflection。
10. 按数据实施 dirty-range、cluster、page cache、history、DRS 和异步优化。
11. 迁出 `prism_render_scene` 的 `bevy_*` 依赖，清理旧 lighting/PBR/render graph。
12. 建立目标设备矩阵，发布每个 profile 的完整证据包。

### 17.2 每个切片的交付物

每个功能切片必须同时提交：

- 代码和 ABI digest；
- 正确性测试与失败注入；
- 最终图像/AOV 或 readback；
- 性能/显存原始数据；
- provider manifest 和支持矩阵；
- recovery/generation 证据；
- 文档状态更新；
- 无 TODO、placeholder、stand-in 或静默 fallback 的生产路径审计。

## 附录 A：核心接口

```rust
trait CanonicalLightingStore {
    fn apply(&mut self, command: LightingCommand) -> Result<SceneDelta, LightingError>;
    fn snapshot(&self) -> Arc<CanonicalLightingSnapshot>;
}

trait GpuSceneReconciler {
    fn prepare(&mut self, delta: &SceneDelta) -> Result<UploadPlan, GpuSceneError>;
    fn commit(&mut self, receipt: SubmissionReceipt) -> Result<(), GpuSceneError>;
}

trait LightingPlanner {
    fn build(&self, input: &LightingPlanInput) -> Result<FrameLightingPlan, PlanError>;
}

trait LightingProvider {
    fn manifest(&self) -> ProviderManifest;
    fn encode(
        &self,
        plan: &ProviderPlan,
        graph: &mut FrameGraphBuilder,
    ) -> Result<ProviderOutputs, EncodeError>;
}

trait OutputOwner {
    fn compose_hdr(&self, inputs: CompositionInputs) -> Result<HdrTarget, OutputError>;
    fn transform_and_present(&self, target: HdrTarget) -> Result<PresentReceipt, OutputError>;
}
```

不可变的不是这些具体 Rust 名称，而是：唯一 owner、typed units、revision/generation、不可变 plan、预算准入、显式资源绑定、真实 HDR target 和可重放 receipt。

## 附录 B：性能证据包

每个 `provider × profile × backend × device` 组合生成独立证据包：

| 类别 | 内容 |
| --- | --- |
| 环境 | commit/patch hash、OS、GPU、driver、API、编译配置 |
| 输入 | scene/asset hash、snapshot、command sequence、camera、resolution、seed |
| 计划 | provider、ABI、quality、budget、capability、降级原因 |
| 图像 | HDR readback、display output、AOV、reference、误差图 |
| 性能 | warm-up、样本数、CPU/GPU P50/P95/P99/max、critical path、queue wait |
| 内存 | 常驻/瞬态 VRAM、history、页池、staging、峰值和回收延迟 |
| 故障 | OOM、pool full、stale revision、device loss、recovery、缺资源 |
| 可重跑 | 精确命令、配置、必需硬件、输出索引、内容 hash |

缺少物理 GPU、最终 target readback 或原始性能数据时，状态只能是“未验证”，不能自动继承相近设备或相近 provider 的结果。

