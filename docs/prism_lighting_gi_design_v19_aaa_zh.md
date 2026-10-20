# Prism 次时代 AAA 光照 / GI 系统设计 v19

> 版本：v19.0
> 日期：2026-10-21
> 状态：深化后的生产架构方案与破坏性迁移基线；不等于当前仓库已经完成全部能力。
> 适用范围：Light、Direct Lighting、Shadow、GI、Reflection、Temporal、Quality、Diagnostics、HDR Output。
> 兼容策略：允许破坏性更新。旧 Light API、旧 registry、Spine 白色管线和 Bevy lighting adapter 以删除门为准退场。

## 0. 执行结论

Prism 当前真正缺少的不是一个孤立的 LightPlugin，而是从作者意图到最终 HDR 像素的完整辐射生产系统：

~~~text
Light / Material / Geometry Intent
  -> Canonical Scene
  -> GPU Scene + Residency
  -> Per-view Radiance Plan
  -> Depth + Opaque Surface/GBuffer
  -> Candidate + Visibility
  -> Direct + GI + Reflection
  -> Temporal Reconstruction + Denoise
  -> Scene-linear HDR
  -> Exposure + Display Transform + Present
  -> Readback + Golden Image + Telemetry + Replay
~~~

v19 的核心决策：

1. LightScene 是光照语义的唯一权威；GPU light table、cluster、shadow cache 和 GI emitter 都是派生产物。
2. RadiancePlanner 是 provider、质量、预算、资源 reservation、history 和 fallback 的唯一决策入口。
3. 光照系统拆成三条正交链：Direct、Visibility/Shadow、Indirect/GI。它们共享 canonical scene，但不共享隐藏状态。
4. 真实完成的硬边界是：真实 opaque raster 写入 GBuffer，direct/shadow/GI 从同一帧数据读取，并向 target 写入 scene-linear HDR。
5. Executed 只代表 GPU 命令执行；只有 readback、视觉金样、恢复和性能证据齐备，才可标为生产准入。
6. 质量档只能减少成本或降低近似阶数，不能通过 stale resource、无效 ambient 或增加能量掩盖缺失能力。
7. 顶级 AAA 的参考重点不是照抄某个产品，而是吸收其边界：多路径 GI、页化阴影、表面缓存或辐射缓存、时域重建、按粗糙度和重要性分配预算，以及可视化诊断。

## 1. 参考产品后的取舍

### 1.1 吸收的工程模式

| 来源 | 吸收的模式 | Prism 的落法 | 明确不照抄 |
| --- | --- | --- | --- |
| Unreal Lumen | screen trace 先行、scene-wide fallback、GI 与 reflection 分开缩放、surface/radiance cache、软件/硬件 trace 双路径 | ScreenProbe、RadianceCache、TraceVisibility provider；每个 provider 有独立 confidence/history | 不复制 Lumen 私有 scene 表示、Nanite 依赖或 console variable 体系 |
| Unreal Virtual Shadow Maps | 虚拟页、按需请求、页缓存、receiver feedback、确定性分配 | VirtualShadowPageProvider 作为 P2；P0 先保留 conventional shadow 正确性 | 不把 VSM 作为 P0 完成的必要条件 |
| Unity HDRP APV | brick 化 probe、流式驻留、局部体积与场景设置分离 | ProbeBrickScene 和 LightingVolume 独立于 LightScene | 不把 probe 当作所有 GI 和镜面反射的最终真值 |
| NVIDIA RTXDI / ReSTIR | 候选生成、reservoir、时空重采样、应用侧 bridge contract | ManyLightSampler 只拥有 estimator 和 reservoir；BRDF、材质、visibility 仍由 Prism 提供 | 不在 WebGPU-first 关键路径绑定 NVIDIA SDK 或闭源实现 |
| AMD GPUOpen | 专用 shadow/reflection denoiser、variance、tile classification、重建与降噪分离 | 每个 noisy signal 都有自己的 variance、history 和 denoiser contract | 不把 ML denoise 作为基础能力或跨设备硬依赖 |

### 1.2 由参考产品反推出的架构结论

- 高质量不是单一算法，而是 surface representation、visibility、cache、temporal 和 output 的组合系统。
- 多路径之间必须有明确的 fallback 和 confidence，不允许把屏幕空间命中、probe 插值和 trace 命中直接相加。
- 大量灯的核心问题是候选和采样分布，不是把全灯表暴露给每个像素。
- 阴影质量的核心问题是可见性预算和缓存失效，不是无限增加 shadow map 分辨率。
- 任何“自动质量”都必须生成可解释的 immutable plan，并可以在 capture/replay 中重建。

## 2. 目标、非目标和首发边界

### 2.1 产品目标

- 物理单位明确的 directional、point、spot、rect、disk、sky/environment 和 emissive source。
- Opaque PBR 真实像素直射、阴影、间接光、镜面和 HDR 输出闭环。
- WebGPU-first：软件 BVH、probe/radiance cache、screen-space provider 为通用能力；硬件 RT、bindless、mesh shader、work graph 只作为 capability provider。
- 4096 级别灯量不要求每像素遍历全灯；通过 candidate、importance、reservoir 和预算控制扩展斜率。
- 静态、stationary、dynamic 具有明确的缓存、失效和更新成本。
- 质量档在集显、主流独显、高端独显和 reference/capture 场景下可预测降级。
- 作者可以通过意图 API、lighting profile、volume 和 explain 工具控制结果，而不是直接操作 GPU 资源。
- 每条能力都有 CPU oracle、GPU readback、视觉金样、性能快照、recovery 和 replay 证据。

### 2.2 首发生产基线 P0

~~~text
opaque mesh
  -> depth
  -> position / normal / albedo / surface-id / coverage
  -> directional + point + spot direct
  -> conventional shadow
  -> scene-linear HDR
  -> exposure / display transform / present
  -> GPU readback / resize / device recovery
~~~

P0 不承诺 VSM、ReSTIR、world trace、emissive mesh sampling、hardware RT、毛发、水、体积、ML denoise 或离线 renderer。

### 2.3 非目标

- 不把玩法、网络、物理、资产 Cook、编辑器工程管理写进 lighting core。
- 不用灯枚举、registry entry、shader 分支或 per-surface storage sample 冒充完成。
- 不把单个硬件、单个分辨率或单个 demo 的时间写成通用性能承诺。
- 不保留无法删除、默认启用、缺乏迁移规则的兼容层。

## 3. 三平面总体架构

### 3.1 Semantic Plane

语义平面保存作者和运行时真正想表达的内容：

~~~text
LightIntent
SurfaceIntent
GeometryInstanceIntent
EnvironmentIntent
QualityIntent
LightingVolumeIntent
~~~

这个平面不包含 GPU buffer offset、bind group、页表、pipeline id 或 frame-local index。

### 3.2 Scene Plane

场景平面把语义转换成稳定、可验证、可追踪的 canonical scene：

~~~text
CanonicalLight
CanonicalSurface
CanonicalEmitter
CanonicalInstance
CanonicalEnvironment
SceneRevision / FieldEpoch / ChangeJournal
~~~

这一层执行单位换算、颜色空间、范围、mobility、linking、visibility mask、profile 和 schema 校验。

### 3.3 Execution Plane

执行平面根据 view、capability、quality、memory pressure 和历史 GPU timing 构建不可变计划：

~~~text
RadiancePlan
  -> resource reservations
  -> pass graph
  -> candidate build
  -> shadow requests
  -> GI/reflection providers
  -> temporal policies
  -> HDR composite
~~~

Execution plane 只能读取 frozen scene revision。执行期间的变化进入下一帧 ChangeJournal，不能修改当前 plan。

### 3.4 唯一权威域

| 权威域 | owner | 不允许被其他域直接写入的内容 |
| --- | --- | --- |
| GeometryScene | render scene | instance、mesh、bounds、velocity、occlusion identity |
| SurfaceScene | material scene | MaterialIR、texture handle、closure、emission |
| LightScene | lighting architecture | 单位、强度、形状、mobility、linking、revision |
| EmitterScene | emitter provider | emissive primitive、area、radiance、PDF |
| VisibilityScene | shadow/trace provider | shadow map、page、BVH、visibility history |
| RadianceScene | GI provider | probe、brick、cache、residency、confidence |
| ViewScene | view owner | camera、jitter、viewport、DRS、exposure、cut epoch |
| PlanScene | planner | provider、budget、fallback、reservation、pass graph |
| OutputScene | output owner | HDR target、AOV、display transform、capture |
| EvidenceScene | diagnostics | readback、telemetry、golden、replay、benchmark |

## 4. 关键不变量

1. Canonical scene 是 CPU 语义权威，GPU scene 是资源化副本。
2. 所有 GPU handle 都带 index + generation；历史和 cache 还要带 scene/plan/device revision。
3. Opaque pass 是 GBuffer 的唯一 writer；HDR composite 是最终 opaque color 的唯一 owner。
4. Direct、GI、Reflection、Shadow、Exposure 各自拥有 history、variance、confidence 和 invalidation key。
5. provider 输出统一为 scene-linear radiance，不在中间 pass 做 gamma 或 display transform。
6. 缺失能力返回结构化 Rejected 或 Degraded，不能写零并报告成功。
7. 所有 pass 在录制前完成 binding；frame graph 录制阶段禁止临时创建 bind group。
8. 同一 plan_revision 内不允许切换 provider、format、sample count 或 resource generation。
9. 质量降级必须产生 reason、fallback、estimated saving 和 output validity。
10. 任何性能数字必须绑定硬件、驱动、后端、分辨率、场景、灯量、质量档和 power mode。

## 5. Canonical Light 设计

### 5.1 LightIntent 与 CanonicalLight

作者层只表达意图：

~~~rust
LightIntent {
    shape: LightShape,
    photometric: PhotometricIntent,
    color: LightColorIntent,
    transform: TransformIntent,
    mobility: Mobility,
    shadow: ShadowIntent,
    linking: LightLinking,
    contribution: ContributionMask,
    quality: LightQualityIntent,
}
~~~

canonical 层保存已验证值：

~~~rust
CanonicalLight {
    handle: LightHandle,
    stable_slot: StableSlot,
    generation: Generation,
    shape: LightShape,
    unit: PhotometricUnit,
    color_linear: Vec3,
    intensity_si: f32,
    transform: Transform,
    range: Option<f32>,
    mobility: Mobility,
    flags: LightFlags,
    channel_mask: u32,
    payload: LightPayload,
    source_revision: Revision,
}
~~~

CanonicalLight 不保存 tile index、shadow page index、reservoir index 或 shader binding。那些都是 plan-scoped 派生产物。

### 5.2 灯型和能力矩阵

| 灯型 | P0 direct | P0 shadow | P1 GI/emitter | P2 many-light | 备注 |
| --- | --- | --- | --- | --- | --- |
| Directional | 是 | conventional/CSM | sky/environment 关联 | 可采样 | 使用 lux |
| Point | 是 | cube/conventional | 可选 | 是 | 使用 candela/lumen 明确转换 |
| Spot | 是 | perspective/conventional | 可选 | 是 | 外角、内角、IES 单独校验 |
| Rect | 否 | 否 | LTC/emitter | 是 | 面积与方向是稳定语义 |
| Disk | 否 | 否 | LTC/emitter | 是 | 不用点光近似伪装支持 |
| Sky/Environment | 是 | sky occlusion | probe/radiance | 是 | 由环境 source owner 管理 |
| Emissive mesh | 否 | 否 | P2 | 是 | 由材质 emission 进入 emitter scene |
| Sphere/Capsule | 否 | 否 | 近似 | P2 | 需要明确的近似和误差上限 |

新增灯型必须分别声明 direct、shadow、GI、reflection、material、platform、quality profile 支持状态；“加入 enum”永远不算支持。

### 5.3 单位与物理边界

- Directional 使用 lux/irradiance 语义。
- Point/spot 允许 candela 或 lumen，但 canonical 层只保留一种 SI 归一化结果。
- Rect/disk 使用面积、辐亮度或 luminous power 的明确转换，不把 nits 和 candela 混用。
- 颜色在进入 canonical scene 前转换到线性工作空间；色温只是输入便利形式。
- 负强度、NaN、无穷、零面积、反向法线和非法角度必须返回诊断。
- 曝光 EV100 只属于 output/view domain，不属于 LightScene。

### 5.4 Mobility 与缓存语义

| Mobility | direct | shadow | indirect | 更新成本 |
| --- | --- | --- | --- | --- |
| Static | 可烘焙或实时 | 可烘焙/缓存 | probe/lightmap/cache | 运行时低 |
| Stationary | 实时 | 静态 caster 缓存 + 动态 caster | probe/cache + 动态补偿 | 中 |
| Dynamic | 实时 | 每帧或按需 | screen/probe/trace | 高 |

Mobility 不直接选择某个算法，而是给 planner 提供成本和失效提示。最终 provider 仍由 capability、quality 和预算决定。

## 6. GPU 数据、候选和 many-light

### 6.1 Stable slot 与 SoA

~~~text
LightCommon[]
DirectionalPayload[]
PunctualPayload[]
AreaPayload[]
EnvironmentPayload[]
EmitterPayload[]
LightIndexMap[]
~~~

规则：

- 删除灯只产生 tombstone 或空闲 slot，不触发全表重排。
- dirty upload 以 range 为单位；首次创建和 schema 变化才允许整表上传。
- GPU record 使用固定 ABI、显式 padding、layout hash 和 schema digest。
- shader 通过 plan 生成的 candidate list 读取灯，不遍历全灯表。
- candidate overflow 必须有计数、重要性排序、可视化和可解释降级。

### 6.2 Candidate pipeline

~~~text
light bounds / importance
  -> view frustum + depth range cull
  -> tile/cluster binning
  -> channel/linking mask
  -> shadow eligibility
  -> material/view importance
  -> bounded candidate list
  -> direct sampler / reservoir sampler
~~~

P0 使用 tiled/clustered candidate。P1 才允许引入 light tree 或 world-space presampling；P2 才允许 ReSTIR DI/GI 进入生产档。

### 6.3 ReSTIR 边界

reservoir 只负责候选重采样和 estimator 权重；不拥有 material、BRDF、scene geometry、shadow 或 trace backend。最小字段：

~~~text
sample_identity
light_handle + generation
target_surface_identity
weight / pdf / M
age
visibility_state
validity / confidence
~~~

任何 geometry discontinuity、surface-id 变化、camera cut、light revision、device recovery 或 plan revision 变化都可能使 reservoir 失效。

## 7. Surface、Material 和 GBuffer

### 7.1 SurfaceClosure

~~~text
base_color
normal
roughness
metallic
specular
emission
transmission
subsurface
clearcoat
anisotropy
feature_mask
texture_handles
~~~

MaterialIR 负责表面 closure，不负责挑选 Light、Shadow 或 GI provider。未实现的 lobe 必须进入显式近似 profile 或 Rejected。

### 7.2 P0 GBuffer 合同

| attachment | 格式 | 语义 |
| --- | --- | --- |
| position | Rgba32Float | world-space position，P0 便于 debug/readback |
| normal | Rgba16Float | world-space shading normal |
| albedo | Rgba16Float | scene-linear base/albedo |
| surface-id | R32Uint | stable surface identity |
| coverage | R8Unorm | opaque surface validity |
| depth | project depth format | occlusion、receiver、reprojection |

后续可以增加 depth reconstruction、compressed normal 或 packed surface profile，但只能新增 contract version，不能让不同 pass 私自改变语义。

### 7.3 GBuffer 规则

- clear pass 必须清理全部 attachment；未覆盖像素的 coverage 必须为零。
- opaque raster 是五路 GBuffer 的唯一 writer。
- composite 只能读取同一 frame/plan revision 的 GBuffer、direct/GI output 和 HDR target。
- coverage 是像素有效性的唯一来源；不能用 position 是否为零推断 coverage。
- resize、device recovery、pipeline replacement 必须同步更新 texture、view、binding、cache 和 generation。
- 真实 opaque draw、HDR readback 和 resize/recovery 是 M1 退出条件。

### 7.4 禁止的伪完成路径

- 用有限的 world-space sample 映射全屏。
- 用 per-surface storage output 冒充逐像素最终画面。
- 用固定 ambient、白色 mesh、identity camera 或零光源报告 direct 完成。
- 复用 static composite group 作为 target-specific group。
- 在 frame graph 录制期间创建 bind group。
- 读上一帧 stale surface、history、page 或 bind group generation。

## 8. Direct Lighting

### 8.1 P0 shading equation

P0 direct provider 至少支持：

~~~text
Lo_direct = visibility * (diffuse_lambert + GGX_specular) * light_radiance
            + emission
~~~

必须使用真实的 position、normal、albedo、roughness、metallic、view vector、candidate lights 和 visibility。Direct provider 输出 scene-linear radiance，不进行曝光和 gamma。

### 8.2 P0/P1/P2 分层

| 层级 | 能力 |
| --- | --- |
| P0 | directional/point/spot、Lambert、GGX、Fresnel、emission、candidate、conventional shadow |
| P1 | rect/disk LTC、IES、light function、soft radius、stationary cache、contact shadow |
| P2 | ReSTIR DI、emissive mesh、light tree、complex closure、多 bounce direct/indirect coupling |

### 8.3 能量和颜色检查

- 物理单位转换、BRDF 和 visibility 分开测量。
- direct-only、environment-only、emission-only、shadow-only 都要有 AOV。
- 对白色 diffuse、镜面阶梯、金属阶梯、强小灯和高曝光场景测量能量误差。
- 降级不能增加总 radiance；任何 exception 都必须由 profile 明确声明并写入报告。

## 9. Shadow / Visibility

### 9.1 Provider 分层

1. ConventionalShadow：P0 正确性基线，方向光 CSM，点/聚光明确 projection。
2. CachedShadow：按 light/caster/receiver revision 失效，支持 static/stationary。
3. VirtualShadowPage：P2，按需页、receiver feedback、确定性分配和页池预算。
4. ContactShadow：小尺度屏幕空间补偿，不承担远距离正确性。
5. TraceVisibility：可选 software/hardware trace provider。

### 9.2 失效键

~~~text
light transform / radius / intensity revision
caster transform / geometry / alpha revision
receiver depth / clip region revision
quality / page-size / resolution revision
device generation
~~~

任何键变化都要传播到相关 shadow resource；不能因为命中缓存就忽略 generation。

### 9.3 页化阴影预算

页池耗尽的固定降级顺序：

~~~text
远距离 receiver 降页
  -> 次要灯降低 shadow resolution
  -> soft shadow 降为 hard/cheaper filter
  -> 次要灯关闭 shadow
  -> 保留主光和近距离关键 receiver
  -> provider rejected with explain
~~~

每个决定都记录 request count、resident pages、miss、eviction、dirty reason、estimated saving 和最终视觉影响。

## 10. GI 与 Reflection

### 10.1 GI provider 谱系

| provider | 适用 | 优点 | 主要限制 |
| --- | --- | --- | --- |
| Probe/Brick | H1+ | 稳定、低成本、屏外覆盖 | 漏光、低频、流式和边界 |
| Screen Probe/GI | H2+ | 动态响应、细节好 | 屏外缺失、遮挡和拖影 |
| Software Trace | H3 | 屏外几何、动态物 | BVH/距离场更新和吞吐 |
| Hardware Trace | capability | 高质量镜面和可见性 | 平台和成本碎片化 |
| Emissive/Many-light | P2 | 动态发射体和复杂灯 | PDF、候选、噪声和 reservoir |

每个 provider 必须输出：

~~~text
radiance
variance
confidence
normal/depth support
estimator metadata / pdf
scene/material/light revision
producer identity
~~~

GI 合成只能按 confidence、variance 和 estimator 合同组合。probe、screen、trace 不能无条件相加，否则会重复计能量。

### 10.2 Probe/Brick

- brick 是 streaming 和 residency 单位，probe 是采样单位。
- probe 记录 irradiance、visibility/occlusion、normal validity、age、scenario 和 revision。
- probe stuck、穿墙、缺 coverage、边界 interpolation 都要可视化。
- dynamic object 默认使用 probe + screen/trace 补偿，不假定 probe 对每个动态物都完全正确。

### 10.3 Screen provider

- 以 GBuffer depth/normal/position 为输入。
- ray miss 必须进入 fallback，不得把屏幕边缘拉伸成命中。
- normal、depth、surface-id、motion 和 history 都参与 rejection。
- 输出 confidence 和 hit distance；无命中不输出伪零 radiance。

### 10.4 Reflection 降级链

~~~text
environment split-sum
  -> reflection probe
  -> SSR / screen trace
  -> planar special case
  -> software trace
  -> hardware trace
~~~

反射 provider 按 roughness、importance、screen coverage 和 budget 选择路径。粗糙表面允许使用 irradiance/radiance cache 近似；镜面表面才请求专用 trace。SSR 必须有屏外 fade、厚度、法线/深度拒绝和 probe/environment fallback。

## 11. Temporal、Denoise 与 Reconstruction

### 11.1 每信号独立历史

Direct、GI、Reflection、Shadow、Exposure 和 DRS 不共享万能 history。最小 HistoryKey：

~~~text
view_id
signal_kind
resolution
camera_cut_epoch
scene_revision
material_revision
light_revision
producer_revision
device_generation
~~~

历史还要保存 validity、confidence、variance、age 和 producer identity。

### 11.2 Reset 和 clamp

以下事件至少触发对应 signal history reset 或缩短：camera cut、teleport、DRS 切换、surface-id discontinuity、material/light revision 跳变、device recovery、pipeline replacement、scene streaming boundary。

时域重建必须使用 motion、depth、normal、surface identity 和 variance；不能只按屏幕坐标复制上一帧。

### 11.3 ML 边界

ML denoise/upscale 可以作为可选 provider，但不能成为 P0 的 correctness 前提。没有 ML 时，基础 temporal + variance-guided spatial path 仍需闭环并可读回。

## 12. Radiance Planner 与质量系统

### 12.1 Planner 输入

~~~text
DeviceCapability
BackendFeatureSet
ViewFamily
SceneStats
MaterialStats
LightStats
MemoryPressure
GpuTimingHistory
UserQualityIntent
DeterminismMode
CapturePolicy
~~~

### 12.2 Planner 输出

~~~text
plan_revision
surface_path
selected providers
pass graph
cluster/tile resolution
light/shadow/GI/reflection budgets
resource reservations
history policy
fallback chain
degradation reasons
~~~

计划不可变。重新规划必须生成新的 plan_revision，并令旧 plan 的提交、临时资源和 history binding 失效。

### 12.3 质量档

| 档位 | GI | Reflection | Shadow | 候选/采样 | 目标 |
| --- | --- | --- | --- | --- | --- |
| H1 | probe/brick | environment/probe | conventional 主光 | 小候选、低采样 | 集显/低功耗 |
| H2 | probe + screen | SSR + probe | conventional/cached | 中候选、时域 | 主流独显 |
| H3 | screen + software/hardware trace | SSR + trace | virtual/cached/trace | reservoir/高采样 | 高端桌面 |
| H4 | deterministic/reference | reference | reference | 可配置 | 离线/验证 |

### 12.4 单调降级规则

固定顺序：

~~~text
dynamic resolution
  -> provider resolution/sample count
  -> history reuse policy
  -> candidate count / shadow pages
  -> secondary light contribution
  -> optional recipe rejection
~~~

降级不能读取旧 generation、增加能量、改变 light unit 或绕过 visibility。每一步必须记录可解释 reason。

## 13. 性能与显存预算

### 13.1 预算不是宣传数字

任何预算必须写成：

~~~text
hardware + driver + backend + resolution + scene hash
  + light count + material complexity + quality + power mode
~~~

默认只给目标上限和测量方法，不把未经实测的数字写成产品承诺。

### 13.2 H2 预算账本模板

| 区域 | 目标预算 | 测量方式 |
| --- | ---: | --- |
| scene/light delta | 0.20 ms | CPU P50/P95/P99 |
| candidate build | 0.30 ms | GPU timestamp |
| shadow | 0.70 ms | page/draw/timestamp |
| direct | 1.10 ms | GPU timestamp + candidate count |
| GI | 0.80 ms | provider split |
| reflection | 0.45 ms | roughness/trace split |
| temporal/denoise | 0.50 ms | signal split |
| composite/output | 0.25 ms | HDR-to-present split |
| 总计 | 4.30 ms 目标 | P50/P95/P99，非单帧最好值 |

4.30 ms 只是 H2/1440p 的计划起点，必须由真实设备和场景矩阵校准。若 P95 超预算，先触发 planner 降级，再报告拒绝原因。

### 13.3 显存账本

显式统计：GBuffer、HDR intermediate、history、shadow page pool、probe brick、radiance cache、light/emitter/candidate buffer、BVH/scene acceleration、readback staging 和 pipeline cache。

每项记录 reserved、resident、peak、evicted、transient 和 device generation。不能只统计长期 buffer 而遗漏 transient peak。

### 13.4 扩展性指标

- 灯数从 8、64、512、4096 增长时，记录 frame time、candidate count、upload bytes、shadow requests 和 VRAM。
- 目标不是“4096 灯恒定成本”，而是证明 candidate/importance 使成本增长受控，并能在预算不足时解释降级。
- 结果必须给 P50/P95/P99，不接受单次录制的最好值。

## 14. 易用性设计

### 14.1 作者 API

~~~rust
lighting.add_light(
    Light::spot()
        .candela(1_500.0)
        .outer_angle_degrees(25.0)
        .shadow(ShadowIntent::High),
);

lighting.set_profile(LightingProfile::auto());
let report = lighting.explain(&scene, &view);
~~~

作者不需要知道 tile、page、bind group、reservoir 或 GPU slot。需要高级控制时，通过 profile、volume、budget override 和 capture policy 提供，而不是暴露内部资源句柄。

### 14.2 Volume 与区域质量

Lighting volume 只覆盖 quality、provider preference、trace distance、shadow distance、probe density、exposure policy 等 view-local 规则；不直接改变 LightScene 的物理单位和权威数据。

局部 volume 采用 priority、weight、blend distance 和 deterministic ordering。volume 冲突必须能在 explain 中显示来源。

### 14.3 必备诊断视图

- Light complexity、tile/cluster occupancy、candidate overflow。
- Light unit、mobility、linking、shadow eligibility。
- Shadow page ownership、dirty reason、miss、eviction、fallback。
- Position、normal、albedo、surface-id、coverage、depth。
- GI provider、confidence、variance、history age、probe coverage。
- Reflection source、roughness path、screen fade、fallback。
- 每 pass GPU 时间、budget、reservation、rejection reason。
- resource generation、plan revision、device recovery、stale-read 拒绝。

## 15. 易拓展设计

### 15.1 RecipeManifest

~~~text
RecipeManifest {
    id
    requires: CapabilitySet
    inputs: ContractSet
    outputs: RadianceContract
    passes: PassDecl[]
    shader_modules
    quality_profiles
    fallback_chain
    diagnostics
    evidence_requirements
}
~~~

recipe 不允许直接拥有全局 LightScene、OutputScene 或 history；它只能声明输入输出和 provider-specific state。

### 15.2 新灯型准入清单

新增灯型必须完成：

1. Intent、单位、验证、canonical payload。
2. stable slot、ABI、layout hash、schema migration。
3. direct sampler、visibility、shadow eligibility。
4. GI/emitter sampling、PDF、MIS 或 reservoir 适配。
5. quality、budget、fallback、debug view。
6. CPU oracle、shader test、真实 GPU readback。
7. directional/point/roughness/occlusion 视觉金样。
8. resize、device recovery、capture/replay 和性能快照。

缺任何一项，只能标记 experimental 或 rejected，不能进入默认 profile。

### 15.3 后端扩展

后端只实现 capability provider：

~~~text
SoftwareVisibility
HardwareVisibility
ConventionalShadow
VirtualShadowPage
BindlessMaterialAccess
AsyncCompute
TimestampQuery
~~~

planner 根据能力图选择实现；lighting core 不直接判断厂商或 API 名称。

## 16. 易维护设计

### 16.1 crate 边界

~~~text
prism_render_architecture
  Light IR / contract / ABI / revision / diagnostics

prism_render_scene
  ECS adapter / extraction / canonical scene publication

prism_render_app
  target / frame graph / submission / recovery / output

prism_render_shading
  BRDF / material evaluation / CPU oracle

prism_render_driver_*
  capabilities / queue / timestamp / device recovery

lighting internal
  packed tables / dirty upload / candidates / provider state
~~~

约束：lighting core 不反向依赖具体 gameplay，不拥有 ECS 业务语义，不把渲染 API 类型泄漏到 canonical contract。

### 16.2 ABI 与版本

每个跨 CPU/GPU contract 至少包含：

~~~text
schema_version
layout_hash
shader_digest
producer_revision
resource_generation
~~~

主版本变化允许删改字段，但必须同步更新 CPU oracle、shader tests、golden image、readback fixture、replay version 和 migration report。

### 16.3 失效传播

~~~text
Light/Material/Geometry mutation
  -> ChangeJournal
  -> FieldEpoch
  -> derived resource invalidation
  -> new SceneRevision
  -> new RadiancePlan
~~~

失效必须带 effect：rebuild、refresh、drop_history、evict、replan 或 reject。不能只设置一个无来源的 dirty bool。

### 16.4 维护性门禁

- layout hash 不匹配时拒绝 pipeline，而不是使用未定义解释。
- generation 不匹配时拒绝读取，而不是 fallback 到上一帧数据。
- owner 变化必须经过设计审查和迁移记录。
- rg 机械检查旧 API、旧 registry、Spine 白色路径和 Bevy lighting adapter 的生产引用。
- 任何新 provider 都必须有独立 telemetry key 和 debug view。

## 17. 当前仓库实现状态

以下只描述 2026-10-21 工作树可观察到的事实，不是完整系统完成声明：

| 模块 | 已观察事实 | 仍缺的生产证据 |
| --- | --- | --- |
| target.rs | 已有 HDR target、position、normal、albedo、surface-id、coverage、depth 创建/resize/destroy/accessor | 真实 GPU readback、跨后端 recovery |
| lighting_gpu.rs | 已有 target-specific composite bind group、五路 GBuffer/HDR binding、dispatch 和按 target identity 缓存 | 真实 composite pixel、销毁/recovery 全链路 |
| frame.rs | clear、opaque MRT、direct/GI dispatch、target-aware composite、透明/风格/custom phase 已接线 | 真实 opaque draw、HDR readback、视觉金样 |
| forward_lit.rs | 已升级五路 MRT，并输出 surface-id、coverage | 真实 GPU draw readback、材质表绑定闭环 |
| presentation.rs/systems.rs | window/offscreen target 生成 target-aware lighting pass | resize/recovery 与多 target capture |
| Light table/candidate | 已存在部分方向光、点/聚光、面积灯和候选结构 | canonical light -> GPU table -> direct/shadow/GI 消费闭环 |
| GI/旧栈 | 仓库仍有并行迁移和旧依赖 | provider owner 收敛、旧路径删除门 |

结论：当前已接通的是 M1 的结构切片，不是完整 AAA Light/GI 系统。尤其不能把现有 per-surface storage output 直接描述为逐像素最终光照。

## 18. 破坏性迁移路线

| 阶段 | 交付 | 硬退出门 |
| --- | --- | --- |
| M0 | owner、contract、ABI、revision、diagnostics、planner skeleton | layout hash、explain、generation 检查通过 |
| M1 | depth/GBuffer/HDR producer、五路 MRT、target-aware composite | 真实 opaque draw + HDR readback + resize/recovery |
| M2 | canonical Light、stable GPU table、cluster/tile、P0 direct、conventional shadow | directional/point/spot 金样 + P95 性能 |
| M3 | probe/brick、screen GI、独立 GI history、confidence 合成 | 室内/室外/动态物 GI 金样 + fallback |
| M4 | SSR/probe reflection、temporal、denoise、DRS | 屏外 fade、history reset、粗糙度单调性 |
| M5 | cached/VSM、many-light、emissive sampling、自动质量 | page/memory/budget/quality matrix |
| M6 | software trace、可选 hardware trace、reference path | capability、replay、readback 对拍 |
| M7 | 删除旧 registry、Spine、Bevy lighting adapter、旧 API | 零生产引用、迁移报告、CI 物理删除 |

### 18.1 允许删除的对象

只要新 contract 和迁移工具已稳定，可以直接删除：

- 旧 Light API 和旧 lighting registry。
- 旧的 per-surface final-color 假路径。
- 旧的默认 white/spine pipeline。
- Bevy lighting adapter 和重复的 Light owner。
- 不能携带 generation/revision 的 cache record。

### 18.2 不允许删除的对象

在对应 provider 有真实替代和证据前，不删除：

- P0 conventional shadow correctness path。
- P0 scene-linear HDR output。
- readback、capture、replay、telemetry 和 debug AOV。
- device recovery、resize 和 generation validation。

## 19. 验收与证据

### 19.1 功能矩阵

- directional、point、spot direct + conventional shadow。
- static/stationary/dynamic 的 cache 和 invalidation。
- position、normal、albedo、surface-id、coverage 的逐像素 readback。
- HDR readback、exposure、display transform、颜色空间。
- target resize、multi-target、device loss、pipeline replacement。
- candidate overflow、shadow page exhaustion、GI fallback、SSR miss。
- history reset、DRS、plan revision 和 stale-read rejection。

### 19.2 场景矩阵

~~~text
S1  室内白墙 + 单主光
S2  室外方向光 + 远景
S3  8 -> 4096 灯量阶梯
S4  动态物 + 间接光
S5  粗糙度/金属度阶梯
S6  强小灯 + emissive source
S7  probe brick 边界和流式
S8  SSR 屏外缺失和反射 fallback
S9  shadow page miss / memory pressure
S10 device recovery / resize / replay
~~~

### 19.3 性能证据

每个场景至少记录：

~~~text
GPU/CPU P50/P95/P99
VRAM reserved/resident/peak
upload bytes
candidate count/overflow
shadow request/page miss/eviction
GI sample/ray count
history rejection
pipeline/shader miss
DRS events
degradation reasons
~~~

### 19.4 视觉和能量证据

- PBR opaque baseline、室内间接光、室外主光、动态物、镜面阶梯。
- FLIP/PSNR 或项目规定的视觉差异指标。
- 直接光、间接光、镜面、shadow、emission、exposure AOV。
- 漏光、firefly、拖影、闪烁、接缝、黑边和历史污染登记。
- 质量档降级不增加总能量，且输出差异可解释。

### 19.5 Evidence package

~~~text
scene_hash
plan_revision
light/material/geometry revision
device/backend/capability
quality profile
frame plan
telemetry
HDR target readback
AOV/debug views
replay seed
shader ABI digest
golden image digest
~~~

## 20. 完成定义和删除门

只有全部满足时，才能称为 Light/GI 系统完成：

1. 默认生产路径使用真实 surface、view、material、light、visibility 和 HDR target。
2. Direct、Shadow、GI、Reflection 至少各有一个真实 provider 和 readback 证据。
3. 每个 provider 的预算、历史、失效、恢复、拒绝和降级可观测。
4. 质量档单调，不读 stale input，不靠伪 ambient 掩盖缺失。
5. GPU readback、视觉金样、P50/P95/P99、VRAM 和 recovery CI 通过。
6. 旧 Light API、旧 registry、Spine 白色管线和 Bevy lighting adapter 已物理删除。

删除门机械检查：

~~~text
rg confirms zero production references to old Light API
rg confirms zero production references to old lighting registry
rg confirms zero Spine white-pipeline path
rg confirms zero Bevy lighting adapter path
ABI digest matches CPU and shader tests
golden image, readback fixture and replay version migrated
old modules are physically absent from CI build
~~~

## 附录 A：最小 contract 集

~~~text
CanonicalLight
LightGpuRecord
LightCandidateList
ShadowRequest / ShadowResult
RadianceSample
RadianceTarget
HistoryKey / HistoryState
ResourceIdentity
RadiancePlan
PassDecl
TelemetrySnapshot
CaptureBundle
~~~

每个 contract 都要声明 producer、consumer、revision、validity、failure mode、fallback 和 evidence requirement。

## 附录 B：首个真实垂直切片

必须包含：

~~~text
opaque mesh
depth
position / normal / albedo / surface-id / coverage
directional + point + spot canonical light
cluster/tile candidate
one direct provider
one conventional shadow provider
scene-linear Rgba16Float HDR
exposure / display transform
GPU readback
resize / device recovery
~~~

明确不包含：

~~~text
VSM
ReSTIR DI/GI
world trace
emissive mesh sampling
hardware RT
hair / water / volume
ML denoise
offline renderer
~~~

## 附录 C：证据等级

| 等级 | 含义 | 允许表述 |
| --- | --- | --- |
| D0 | 设计/contract 存在 | 设计完成 |
| D1 | CPU 准备或资源创建存在 | 实现中 |
| D2 | GPU 命令可记录并有基础测试 | 可执行切片 |
| D3 | 真实 GPU readback、resize/recovery 通过 | 已验证切片 |
| D4 | 视觉、性能分位、显存、跨后端证据齐全 | 生产准入 |

禁止把 D0、D1 或 D2 写成“AAA 已完成”。

## 附录 D：外部公开参考

- Epic Games: [Lumen Global Illumination and Reflections](https://dev.epicgames.com/documentation/unreal-engine/lumen-global-illumination-and-reflections-in-unreal-engine)
- Epic Games: [Lumen Technical Details](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine)
- Epic Games: [Lumen Performance Guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-performance-guide-for-unreal-engine)
- Epic Games: [Virtual Shadow Maps](https://dev.epicgames.com/documentation/en-us/unreal-engine/virtual-shadow-maps-in-unreal-engine)
- Unity HDRP: [Adaptive Probe Volumes](https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@latest/manual/probe-volumes.html)
- NVIDIA RTXDI: [SDK and samples](https://github.com/NVIDIA-RTX/RTXDI)
- NVIDIA RTXDI: [Integration guide](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Integration.md)
- AMD GPUOpen: [FidelityFX Denoiser](https://gpuopen.com/fidelityfx-denoiser/)

这些资料只用于吸收公开工程模式。Prism 的实现、许可证、性能和兼容性必须以本仓库自己的 contract、测试、readback、金样和 benchmark 为准。

