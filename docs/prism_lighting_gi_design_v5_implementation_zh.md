# Prism 次时代光照 / GI 实施设计 v5.0

> 状态：实施规格，允许破坏性更新。
>
> 版本日期：2026-10-20
>
> 本文用于指导 Prism 光照、阴影、直接光、GI、反射、材质辐射查询和质量控制的实际落地。它不是“已有功能清单”，也不把 CPU 参考实现、shader 文件、单元测试或预算目标写成生产完成证据。

## 0. 结论与范围

### 0.1 结论

Prism 已经存在光照基础，但还缺少一套产品级的统一 light system。当前能力分散在 ECS 光源提取、GPU light buffer、cluster 构建、宿主 `LightingPlan`、软件 trace、GI candidate 和若干 reference provider 中；这些能力不能继续各自维护光源语义和生命周期。

本版本采用以下总原则：

> `LightScene` 是唯一光源权威；`GpuScene` 是唯一 GPU 生命周期权威；`LightingWorkGraph` 是唯一按视图生成工作与选择 provider 的入口；direct、shadow、GI、reflection 和 radiance provider 只通过显式契约组合。

### 0.2 目标

- 面向通用实时内容，而非绑定某个游戏的角色、武器、血量或玩法规则。
- 支持方向光、点光、聚光灯、矩形/圆盘/球/胶囊面光、环境光、网格发光体和烘焙/外部辐射源。
- 支持从兼容档到高端档的同一套数据和诊断路径。
- 允许 exact clustered、stochastic direct、ReSTIR DI、VSM、software trace、hardware RT、probe/cache 和 reference path 共存，但不允许它们各自发明第二套光源表、采样权重或历史规则。
- 在真实视图帧图中证明：光源输入、GPU 上传、阴影/GI 工作、材质求值、scene-linear 合成和最终输出确实连通。
- 让性能、画质、内存、带宽、历史拒绝、缓存失效和降级原因都可以捕获、回放和解释。

### 0.3 非目标

- 不承诺在 WebGPU 上复制某个商业引擎全部底层实现。
- 不把固定灯数、固定采样数或固定帧预算写成所有场景都成立的质量保证。
- 不把 software BVH 标成硬件 RT，不把 CPU reference 标成实时生产 backend。
- 不把显示变换后的颜色用于证明能量守恒。
- 不把所有 mesh emitter 强制展开为普通 punctual light。
- 不在第一阶段同时完成透明、毛发、水体、体积、复杂 alpha、完整纹理材质和所有离线能力；这些能力必须沿用同一契约逐步接入。

## 1. 现状诊断与必须纠正的问题

### 1.1 已有资产

当前仓库已有以下可复用基础：

| 资产 | 当前角色 | v5 处理 |
|---|---|---|
| `prism_render_scene/src/lighting` | 历史 Bevy ECS 光源提取、GPU buffer、cluster 路径 | Bevy 已退出目标架构；不再扩展此入口。能力须迁移到 Prism 自有 scene/RHI/work-graph，完成后删除旧入口及依赖 |
| `prism_render_app/src/lighting.rs` | 宿主 direct/GI 计划与部分执行路径 | 变为统一 work graph 的编译/执行入口 |
| `prism_render_architecture/src/lighting` | handle、epoch、journal、trace、预算等契约 | 收敛并升级为稳定 ABI/生命周期契约 |
| light tree、area light、probe、reference PT | 算法和 oracle | 提取为 provider，不复制光源模型 |
| software BVH / trace | 可移植可验证的 trace backend | 作为 capability fallback，保持 backend 身份真实 |
| VSM/reference shadow | 阴影算法参考 | 接入真实 page pool、request、invalidate 和 receipt |
| OCIO/display transform | 最终颜色变换 | 保持在 scene-linear lighting 之后 |

### 1.2 现有系统不能继续保持的状态

1. ECS 提取链和宿主 `LightingSceneInputs` 同时拥有完整光源表。
2. 每帧清空并重建 light vectors、GPU buffer 或候选表作为默认路径。
3. 用一个 `lighting_changed` 布尔值触发所有 cluster、shadow、GI、reflection 和 history 失效。
4. provider 复制材质、光源或 BVH 数据，造成多个语义来源。
5. 以“模块存在”“shader 可编译”“CPU 测试通过”作为真实 GPU 图完成的证据。
6. cluster overflow、shadow page miss、GI cache miss、history reject 和 fallback 被静默处理。
7. 将环境辐射、mesh emission、baked data 和 punctual light 强行塞入同一种 shadow 逻辑。

### 1.3 当前仓库的已验证边界

当前代码应按三条事实理解，而不是简单归类为“有”或“没有” light system：

1. `prism_render_architecture::lighting::LightScene` 已提供稳定 slot、generation、scene epoch、delta、checkpoint 和 replay 契约，但目前还不是 `prism_render_scene` 与 `prism_render_app` 的共同生产输入。
2. `prism_render_scene/src/lighting` 已有真实 Bevy ECS → GPU buffer → cluster → `shading_resolve` 链路，并且逐像素 resolve 会消费方向光、punctual light、environment 和 cluster 数据。
3. `prism_render_app` 已有真实 RHI compute 的 direct/GI/trace/candidate/composite vertical slice，但其 lighting output 仍主要是 storage buffer；在没有最终 HDR texture、材质/可见性输入和 present readback 证据前，不能声称宿主路径已经完成逐像素场景照明。

因此第一条真正的产品纵切不是再增加一个光照算法，而是证明：

```text
LightScene insert/update/remove
  → scene/app adapter
  → stable GPU upload
  → per-view work
  → material/visibility resolve
  → HDR scene-linear output
  → display/present
  → readback or capture evidence
```

这条边界也决定了后续验收：CPU contract、GPU dispatch、buffer 数值、HDR 像素和窗口 present 必须分层报告，不能互相替代。

## 2. 权威架构

### 2.1 数据流

```text
Authoring / Import / ECS / Runtime / Network adapters
                         │
                         ▼
                 LightScene（唯一 CPU 权威）
                         │ SceneSnapshot + SceneDelta
                         ▼
                 GpuScene（稳定 slot 和资源生命周期）
                         │
                         ▼
       LightingWorkGraph（按 view / capability / budget 编译）
          ┌──────────────┼──────────────┬──────────────┐
          ▼              ▼              ▼              ▼
       Direct         Shadow          GI          Reflection
          └──────────────┴──────────────┴──────────────┘
                         │ provider output tokens
                         ▼
             scene-linear resolve → display → present
```

### 2.2 所有权规则

| 对象 | 唯一 owner | 其他模块只能做什么 |
|---|---|---|
| 光源语义、单位、来源、句柄 | `LightScene` | 读取 immutable snapshot、提交 adapter change |
| GPU slot、payload、fence、回收 | `GpuScene` | 提交 upload plan、读取稳定 handle |
| 每视图可见候选、cluster、reservoir 输入 | `LightingWorkGraph` | 消费 snapshot 和 budget |
| 材质响应、发光、命中点辐射 | `RadianceProvider` / material owner | 通过 `RadianceQuery` 读取 |
| 阴影页与可见性 | Shadow provider | 通过 `VisibilityQuery` 请求 |
| 历史、缓存、reservoir | 对应 signal owner | 使用 epoch、revision 和 validity contract |
| 质量决策 | `QualityResolver` | 读取最终 `ResolvedQualityState` |

禁止 `prism_render_scene` 和 `prism_render_app` 各自维护一套 authoritative light registry。ECS、编辑器、导入器、网络和脚本都是 adapter，不得直接写 GPU buffer。

### 2.3 crate 边界

```text
prism_render_architecture
  Light IR / handles / epochs / deltas / ABI / provider contracts / diagnostics

prism_render_scene
  Bevy ECS adapter / scene extraction / view adapter / change publication

prism_render_app
  RenderApp lifecycle / snapshot freezing / work graph / graph recording / receipts

prism_render_shading
  BRDF / material evaluation / radiance semantics / CPU oracle

prism_render_driver_*
  device capabilities / queue / timestamp / RT / resource limits

prism_light_gpu（可先作为内部模块，稳定后再拆 crate）
  packed ABI / stable slots / dirty upload / compaction / GPU indirection
```

## 3. Light IR 与物理语义

### 3.1 光源分类

```text
AnalyticEmitter
  Directional / Point / Spot

ExtendedEmitter
  Rect / Disk / Sphere / Capsule

EnvironmentEmitter
  Sky / EnvironmentMap / Portal / VolumeEmitter

MeshEmitter
  emissive triangles / instances / clustered emitter groups

ExternalRadianceSource
  baked lightmap / probe field / capture / project-defined radiance field
```

分类表示采样和可见性语义，不是编辑器展示名称。环境源和外部辐射源不自动生成 punctual shadow map；mesh emitter 不自动复制为若干点光。

### 3.2 单位

所有记录必须有 canonical unit，所有导入转换必须保留来源单位和转换证据：

| 类型 | canonical 表达 | 约束 |
|---|---|---|
| Directional | illuminance 或明确约定的 scene-linear irradiance | 记录角直径/角半径 |
| Point / Spot | luminous flux 或 radiant intensity | 不混淆 lumen、candela、W/sr |
| Area / Mesh | radiance 或明确的 photometric authoring value | 面积、法线、单/双面策略独立记录 |
| Environment | scene-linear radiance | 记录 mip、分布、投影和色域 |
| Emissive material | material emission/radiance | 不能与 mesh light 重复贡献 |
| Color | scene-linear RGB + primaries | 导入 sRGB 时显式转换 |

曝光、白平衡、tone mapping、gamut mapping 和 transfer function 不进入光照积分。艺术增益必须标记为 `ArtisticControl`，不能伪装成物理输入。

### 3.3 `LightRecord` 规范

逻辑结构：

```text
LightRecord {
  identity: LightId + Generation + SourceId
  kind: LightKind
  transform: position / orientation / local_frame
  emission: color / unit / flux / intensity / radiance / temperature
  shape: radius / size / capsule_length / cone / barn_door
  influence: bounds / range / falloff / channel_mask
  visibility: shadow_policy / ray_policy / casts / receives / volume_policy
  sampling: importance_class / hierarchy_node / texture_distribution
  mobility: static / stationary / dynamic / streaming
  authoring: source_asset / import_revision / warnings / provenance
  revisions: field_epochs / schema_revision
}
```

`mobility` 只是更新策略提示，不自动决定 provider。静态灯仍可能因相机或 shadow page 变化更新；动态灯可能只更新一个 payload slot。

## 4. 稳定句柄、SceneDelta 与生命周期

### 4.1 句柄

生产句柄必须包含 generation：

```text
LightHandle {
  scene_id: SceneId,
  slot: u32,
  generation: u32,
}
```

`slot` 只用于定位，不能单独代表身份。旧 command、reservoir、shadow request、debug selection 和 capture replay 都必须验证 generation 与 scene epoch。

### 4.2 字段 epoch

至少拆分以下 epoch：

```text
TransformEpoch
EmissionEpoch
ShapeSamplingEpoch
ShadowPolicyEpoch
VisibilityEpoch
EmitterMaterialEpoch
EnvironmentEpoch
SchemaEpoch
DeviceEpoch
```

字段 epoch 用于精准决定 cluster、shadow page、GI cache、reflection history、ray scene 和 reservoir 的失效范围，禁止用一个全局 dirty 标记代替。

### 4.3 `SceneDelta`

```text
SceneDelta {
  sequence: u64,
  frame: u64,
  scene_epoch: u64,
  creates: Vec<CreateRecord>,
  updates: Vec<FieldUpdate>,
  destroys: Vec<LightHandle>,
  hierarchy_updates: Vec<HierarchyUpdate>,
  invalidation: ChangeFields,
}
```

应用规则：

1. 先验证 sequence、generation、单位、范围和 schema。
2. 在事务中应用所有变更；任何失败都不提交半个 delta。
3. 生成新的 immutable snapshot 和精确失效集。
4. 发布 `SceneDeltaReceipt`，包含 accepted/rejected/warnings、revision 和受影响资源数量。
5. delta gap、schema 不兼容或设备恢复时，从 checkpoint + snapshot 完整重建。

### 4.4 生命周期

```text
Alive → Retiring → FencePending → Recycled
```

只有确认所有引用该 slot 的 GPU submission 完成后才允许回收。generation 递增；generation 溢出必须进入不可复用错误，而不是静默回绕。

## 5. GpuScene 与增量上传

### 5.1 分层存储

```text
HandleTable
  handle → stable slot → payload offset / alive / generation

HotPayload
  position / direction / color / intensity / bounds / flags

ColdPayload
  shape / IES / texture / hierarchy / debug metadata

ViewIndirection
  visible candidates / cluster lists / reservoir sources
```

stable handle 与 payload offset 分离，允许后台 compaction 而不改变逻辑身份。

### 5.2 上传策略

优先级：

1. dirty range upload；
2. hot/cold split；
3. 相邻 dirty range 合并；
4. 碎片率、generation 密度或容量阈值超过门槛时后台 compaction；
5. 首次创建、设备恢复、delta gap、ABI/schema 变化时 full rebuild。

每次上传必须记录：bytes、range 数、staging copy、submission token、device epoch、是否 full rebuild、被拒绝的 record 和原因。

### 5.3 ABI

- 使用明确的 WGSL/CPU offset、stride、alignment 和数组容量。
- 不把 Rust `repr(C)` 当成 WGSL 布局证明。
- 不使用跨后端不稳定的隐式 bool/u8/u16 布局。
- payload 带 `abi_version` 和 `payload_size`。
- 超过 GPU 容量时必须分层、降级或显式拒绝，禁止 `.min(capacity)` 静默截断。

## 6. ViewLightingWork 与直接光

### 6.1 工作生成

每个 view 根据以下输入生成 immutable `ViewLightingWork`：

```text
SceneSnapshot
View / camera / projection / jitter
CapabilityProfile
ResolvedQualityState
ResourceBudget
Material / geometry coverage
Previous frame validity
```

输出至少包括：visible light candidates、cluster ranges、overflow records、shadow requests、GI candidate budget、reflection history policy 和 provider selection。

### 6.2 Exact clustered

适用于少量强灯、低噪声需求、兼容档和需要稳定结果的 workload：

```text
frustum/depth cull → tile/cluster lists → exact BRDF resolve
```

要求：

- cluster overflow 不得静默截断；
- 强主光可进入显式保留列表；
- 每个候选可回溯到 generation-safe handle；
- overflow、每像素影响灯数和最坏 cluster 必须可诊断。

### 6.3 Stochastic / ReSTIR direct

适用于大量局部灯、面光和 mesh emitter：

```text
hierarchy proposal → visibility → reservoir
→ temporal/spatial reuse → denoise → resolve
```

Reservoir 必须包含 selected sample、`M/W`、source/target PDF、measure、scene/light revision、age、reuse source、visibility 和 bias/confidence。采样数不等于无偏证明；随机性关闭也不等于 reference。

### 6.4 Light hierarchy

层次节点至少提供：bounds、cone/solid-angle bounds、emission upper bound、子节点 PMF、revision 和 dynamic refit 状态。错误的 upper bound 会造成缺光，因此 provider 必须能输出剪枝原因和保守性诊断。

## 7. 阴影系统

### 7.1 Provider 路由

| workload | 首选 | fallback |
|---|---|---|
| 强方向光 | directional VSM clipmap / RT | CSM / cached cascade / contact |
| 少量强点/聚光 | VSM page / RT | cube/spot shadow / contact |
| 大量小灯 | stochastic visibility | unshadowed importance + contact |
| 面光/mesh emitter | RT visibility / shadow reservoir | VSM approximation / baked |
| 环境光 | bent normal / probe / world GI | ambient fallback |
| 静态内容 | baked/cache | runtime shadow |

路由由 `CapabilityProfile + QualityIntent + Budget + LightRecord` 决定，不能由 shader 隐式决定。

### 7.2 VSM page

```text
PageKey {
  light_handle, clip_level, page_xy, mip, view_family
}

PageState {
  resident, last_used, caster_epoch, light_epoch,
  content_hash, fence, fallback_reason
}
```

必须分开统计 resident memory、page request/table traffic、每帧更新页数和 caster draw/geometry cost。失效来源包括光源变换/形状、caster bounds、skinning/WPO/alpha coverage、材质/layout、camera 新页、streaming 和 device epoch。

### 7.3 软阴影降级

允许 `RT → VSM → contact → hard/unshadowed`，但每次降级必须有 reason、计数、debug view 和恢复条件。不能为了预算静默删除灯或吞掉阴影贡献。

## 8. RayScene 与 RadianceQuery

### 8.1 RayScene 层级

```text
Near triangle scene
Instance/skinned scene
Simplified software BVH / proxy
Far-field HLOD / merged scene
Screen depth/normal/material evidence
```

screen、world、far-field hit 不保证同义；offscreen、透明、alpha masked、camera cut、WPO 和远场 proxy 必须在结果中携带 coverage/confidence。

### 8.2 Query 分离

`TraceQuery` 只负责几何可见性和命中信息：hit/miss、surface id、barycentrics、uv、distance、revision、deadline、alpha policy。

`RadianceQuery` 负责材质、纹理、法线、粗糙度、发光、透明/alpha test 和 outgoing radiance。trace backend 不复制材质参数；材质系统不复制 BVH 所有权。

生产结果必须显式标记：

```text
execution_backend: cpu_reference | software_gpu | hardware_rt
material_proof: supported | partial | rejected
geometry_coverage: full | proxy | screen_only | missing
bias_bound: known | estimated | unknown
variance/confidence: explicit value or unknown
```

## 9. GI 与反射

### 9.1 GI 四层

```text
Evidence    screen / probe / world trace / baked field
Radiance    hit material and emission evaluation
Estimator   path sample / ReSTIR GI / cache query
Reconstruction temporal / spatial / denoise / resolve
```

screen GI、probe、world trace 不能无条件相加。组合前必须证明目标积分、measure、PDF、overlap、bias 和 coverage 兼容；否则使用显式 partition、mutually exclusive routing 或拒绝组合。

### 9.2 ReSTIR GI

Reservoir 必须验证 scene epoch、geometry/material/light/environment epoch、surface identity、normal/depth、camera motion、history age 和 candidate validity。miss 是否计入 `M/N` 必须由 CPU/WGSL 共同契约固定。

### 9.3 Probe / surface cache

每个 cache cell 必须有 bounds、resolution、age、coverage、directional representation、leak policy、confidence、update budget、warm-up、eviction 和 device recovery。cache miss 不得读取未初始化历史值。

### 9.4 反射独立建模

镜面信号拥有独立 history key、roughness range、rejection policy、visibility provider 和 fallback：

```text
screen reflection → planar/probe → world trace → hardware/reference
```

漫反射 GI 的 denoise 和镜面反射的 denoise 不得共享一套不加区分的历史规则。

## 10. 材质、透明、毛发、水体与体积

第一阶段只承诺真实闭环的受限材质能力，并在 receipt 中写明能力范围。建议顺序：

1. unlit/no-texture surface；
2. opaque textured + normal/roughness；
3. alpha masked 与材质 coverage；
4. skinning/WPO；
5. transparent、hair、water、volume 专用 provider。

这些类型不应被当作 opaque 的自动特例。必须提前定义是否进入 shadow/ray scene、alpha policy、history key、refraction/volume semantics 和 fallback。

## 11. 质量、能力与预算

### 11.1 四层决策

```text
CapabilityProfile       后端和设备真实支持
QualityIntent            用户/项目质量意图
ResourceBudget           本帧 CPU/GPU/显存/带宽/IO 预算
ResolvedQualityState     本 view 本帧最终选择
```

只有 `ResolvedQualityState` 能进入 shader/graph。它必须可序列化、可 replay，并写明每个被拒绝 provider 的理由。

### 11.2 推荐档位

| 档位 | direct | shadow | GI/reflection |
|---|---|---|---|
| Compatibility | exact clustered | CSM/contact | AO + probe/低成本 screen |
| Balanced | clustered + limited stochastic | cached VSM/RT mixed | screen + probe + limited world trace |
| High | stochastic/ReSTIR | VSM + RT provider | screen → world trace + cache |
| Ultra | high budget reservoir | high-res VSM/RT | high-quality reconstruction |
| Reference | reference integrator | geometry visibility | offline/reference path |

### 11.3 预算维度

预算不仅是总灯数，还包括：

- 每像素影响灯数 P50/P95/P99；
- shadow pages requested/resident/updated；
- ray/triangle/alpha-test 数量；
- reservoir 数量、history memory 和 rejection；
- light/GPU scene upload bytes；
- cache update、streaming IO、skinning、far-field 更新；
- 多视图、split-screen、reflection probe 和 offline output 倍增。

异步 compute 只有真实 queue/dependency/timestamp 支持时才可计入 overlap，不能简单相加或相减 pass 时间。

## 12. 易用性、诊断与回放

### 12.1 必备 debug view

- LightScene：handle、generation、单位、来源、范围和 epoch；
- light complexity：tile/pixel 影响灯、importance、shadow cost；
- cluster occupancy/overflow；
- shadow page requested/resident/updated/invalidated；
- RayScene coverage、proxy、alpha policy 和 missing geometry；
- GI cache coverage、age、confidence、leak risk；
- reservoir 的 selected sample、`M/W`、PDF、reuse lineage；
- temporal rejection reason；
- fallback chain、quality state、budget exhaustion；
- scene-linear energy/error heatmap。

### 12.2 Per-pixel explain

选中一个像素后，应能导出：

```text
pixel
 → material/surface identity
 → direct candidates and weights
 → shadow provider and visibility
 → GI candidate/reservoir lineage
 → cache/probe/world evidence
 → temporal history decision
 → resolve and display transform
```

### 12.3 Capture/replay

Capture 至少包括 LightScene snapshot、delta checkpoint、GpuScene ABI、device capability/limits、camera/view/jitter/exposure、quality/budget/resolved state、scene/material/ray revisions、random state、provider output metadata 和 device epoch。

Replay 分三类：

1. CPU oracle：明确容差和确定性；
2. same-device GPU：尽可能 bitwise 或固定容差；
3. cross-device：只承诺统计/图像误差，不承诺浮点 bitwise 一致。

## 13. 真实帧图接入

标准帧序列：

1. adapter 提交 authoring changes；
2. `LightScene::apply_changes` 验证并生成 snapshot/delta receipt；
3. render world 固化 immutable snapshot；
4. `GpuScene::prepare` 上传 dirty ranges、处理 retirement 和 device recovery；
5. 每个 view 生成 `ViewLightingWork`；
6. graph 编译 direct/shadow/GI/reflection；
7. providers 写入带 revision/measure/confidence 的 output token；
8. scene-linear resolve；
9. display transform、present/offscreen output；
10. 发布 `LightingFrameReport` 和可选 capture。

Render target 必须支持 HDR scene-linear 中间结果；`Rgba8Unorm` 不能作为高端光照合成的唯一中间目标。最终输出格式与内部积分格式必须在契约上分离。

## 14. 破坏性迁移计划

### M0：契约冻结

- 冻结 Light IR、canonical units、generation handle、SceneDelta、ABI schema；
- 建立 provider registry、capability profile、quality/budget/resolved state；
- 旧路径增加 receipt 和诊断，不再新增平行契约。

### M1：唯一 LightScene

- ECS、runtime、import 都只发布 adapter changes；
- `ExtractedLights` 和 `LightingSceneInputs` 降级为 adapter/snapshot 输入；
- 删除旧 `LightHandle(pub u32)` 作为生产身份。

### M2：GpuScene 增量上传

- stable slot、hot/cold payload、dirty range、fence-safe recycle；
- full rebuild 只保留为 recovery/reference；
- 加入容量拒绝、generation stale 和 delta gap 测试。

### M3：Exact clustered + 一条阴影纵切

- 方向光、点光、聚光灯先闭环；
- 证明真实 graph、scene-linear output、overflow diagnostics 和 capture；
- directional VSM/CSM 或已验证的兼容 shadow provider 作为第一条阴影路径。

### M4：VSM page/cache

- page request、resident、invalidate、evict、warm-up、dynamic caster；
- 分离 page memory、request traffic、update pages 和 caster cost。

### M5：Stochastic direct / ReSTIR DI

- hierarchy、proposal、visibility、reservoir、temporal/spatial reuse；
- 小场景逐像素对拍 exact/reference；
- 大灯场景报告噪声、历史拒绝和 P50/P95/P99。

### M6：RadianceQuery + world GI

- 先完成 unlit/no-texture 限定闭环；
- 再逐步扩展纹理、法线、alpha、skinning；
- software trace 作为真实 fallback，硬件 RT 只在 capability 允许时启用。

### M7：Probe/cache/reflection/productization

- cache coverage、leak control、history、reflection 独立信号；
- capture/replay、debug UI、多视图、大世界、streaming、device loss；
- 建立自动 benchmark 和内容回归场景。

## 15. 验收标准

### 15.1 功能闭环

| ID | 验收项 | 证据 |
|---|---|---|
| F1 | ECS/import/runtime 进入同一 LightScene | snapshot identity + source trace |
| F2 | create/update/destroy | generation、retirement、stale rejection |
| F3 | exact clustered | direct output、overflow、空场景、强主光 |
| F4 | incremental upload | 不变帧无全量上传、delta gap 可恢复 |
| F5 | shadow provider | request/resident/invalidate/evict receipt |
| F6 | trace backend | hit/miss、revision、deadline、fallback 正确 |
| F7 | RadianceQuery | material proof、能力拒绝可解释 |
| F8 | GI estimator | measure/PDF/M/W/zero proposal CPU/GPU 一致 |
| F9 | temporal | cut、resize、exposure、epoch、device recovery |
| F10 | reflection | screen/planar/world 路由和历史有效 |
| F11 | output | scene-linear → display → present/offscreen 一致 |
| F12 | replay | capture 可重放且版本不兼容能明确失败 |

### 15.2 性能与内存

每个基准场景至少报告 CPU extract/plan/upload/prepare、GPU pass 和 critical path 的 P50/P95/P99；light/cluster/page/ray/reservoir/history memory；upload bytes；GPU wait、queue overlap、pipeline stall；overflow、cache miss、page invalidation、history reject 和最终图像误差。

4096 light 只能作为一个受控基准，不得单独作为产品能力结论。必须固定分辨率、内容、每像素影响灯数、投影比例、光源分布、backend、内部渲染比例和质量状态。

### 15.3 画质与能量

测试集至少包括：单强方向光、密集室内灯、多面积光 + glossy/rough surfaces、emissive mesh + indirect bounce、动态 caster、skinned mesh、alpha masked foliage、camera cut、曝光变化、resize、大世界远景以及特殊材质边界。

指标包括 scene-linear relative error、irradiance/radiance consistency、shadow penumbra error、temporal flicker/ghosting、GI leak、missing contribution、over-bright contribution 和 fallback correctness。PSNR/SSIM/FLIP 只能辅助，不能替代视觉验收。

### 15.4 降级单调性

不宣称“低质量必然只是更暗”或其他未经证明的普适性质。必须保证：

- 降级原因、资源变化和恢复条件可见；
- 不读取未初始化或过期资源；
- provider 被拒绝时有 fallback 或显式失败；
- 升档不会使用更旧的历史；
- 资源消耗、噪声和误差在定义好的场景中符合门槛。

## 16. 生产完成定义

光照系统只有同时满足以下条件，才能称为生产闭环：

1. 所有来源进入同一个 LightScene，并能追溯单位、来源和 revision。
2. GPU 光照 pass 通过 handle、epoch、view、provider 和 budget 可解释。
3. exact、stochastic、software trace、hardware RT 有明确 capability/fallback 关系。
4. shadow、GI cache、reservoir、material radiance 和 temporal history 的失效规则一致。
5. 降级不会静默删灯、吞贡献或读取过期资源。
6. 真实场景、真实 graph、真实 backend 有 receipt、capture 和图像回归证据。
7. 性能结论带硬件、分辨率、内容分布、质量状态和 P50/P95/P99。
8. 新增光源、provider 或材质能力不需要复制第二套场景表、采样权重和诊断系统。

## 17. 实现进度与证据（滚动更新）

本节记录代码现状，不把设计目标当成已交付能力。2026-10-20 当前纵切：

| 能力 | 当前实现 | 验证状态 / 边界 |
|---|---|---|
| 唯一 CPU LightScene | `prism_render_architecture::lighting::LightScene`；generation-safe scene lineage、delta、checkpoint/replay | 有契约测试；需要和各 adapter 统一验证 |
| 原生宿主适配 | `prism_render_app::LightingLightScene` → directional/point/spot ABI；对不支持的 class/unit/layer/flags/capacity 明确拒绝 | 转换测试覆盖创建/更新/删除、单位、方向符号和聚光角；仓库测试待跑 |
| 旧 ECS 场景渲染上传（非目标路径） | Bevy `ExtractedLights` / `LightGpuBuffers` 属于待移除的废弃接线，不作为 Prism 光照系统能力或验收结果 | 由原生 `prism_render_app::LightingLightScene` 与 Prism RHI 路径替代；旧模块仍待删除 |
| stable GPU light slots | `IncrementalLightTable<T>` 提供稳定 slot、generation、retirement fence、dirty range/full rebuild policy、upload receipt | 有 10 个单元测试；尚未连接 scene adapter 到真实 buffer/queue lifecycle |
| 原生 direct-light buffer 增量上传 | `LightingGpuState::prepare` 缓存方向光与点/聚光 ABI 内容；未变化时零次上传，稀疏变化合并成 byte-range 写入，密集变化使用整段写入；资源 bundle replay 后强制全量重传 | 上传规划单测已加入；当前 workspace 物理拓扑编译错误阻止 package 测试运行，真实设备上传量与 GPU 时间未测 |
| 原生 light scene 提取缓存 | scene lineage identity + sequence + epoch 三元组识别快照；未变化帧复用 render-world 已派生的光源数组，避免克隆整个 CPU `LightScene` 和重复转换；scene 被替换但序号/epoch 碰巧相同仍会刷新，并触发缓存失效 | 已加入同 epoch 不同 scene lineage 的回归测试；当前测试构建被 workspace 物理模块编译问题阻断 |
| 原生光源适配错误闸门 | canonical `LightScene` 无法无损映射到当前 direct-light ABI 时，`execute_frame` 在任何 lighting GPU 上传/图录制前拒绝该帧；`RenderFrameStats::lighting_input_error` 保留结构化原因 | 已扩展 Rect light 拒绝集成测试，验证不提交帧且能读取错误；测试受 workspace 编译错误影响，尚未执行通过 |
| 实际 HDR/present 闭环 | 当前宿主光照仍未证明进入最终 scene-linear HDR texture 并被 present/readback | 未完成；不宣称生产照明闭环 |

截至记录时，`cargo test -p prism_render_scene lighting::buffers::tests --lib` 编译被不相关的既有工作区错误阻断：`pkg/prism_render_scene/src/shading/lobe_budget/gpu.rs` 调用 `uniform_buffer()` 时缺少 `has_dynamic_offset: bool` 参数。该失败不是灯光测试失败，也不证明灯光测试通过。后续应在修复或隔离该阻断后重跑，并继续逐阶段记录 GPU/capture 证据。

## 附录 A：与 v4 的关系

v4 是架构蓝图；本 v5 增加了：

- 实施阶段和破坏性迁移顺序；
- `LightScene → GpuScene → ViewLightingWork → LightingWorkGraph` 的产品调用链；
- 明确的 ownership、receipt、ABI、容量拒绝和 device recovery 约束；
- 真实帧图、HDR 中间目标、材质能力证明和 capture/replay 要求；
- 将“算法存在”与“真实 backend 已接通”分开验收。

## 附录 B：仓库入口

- v4 架构蓝图：[prism_lighting_gi_design_v4_zh.md](prism_lighting_gi_design_v4_zh.md)
- 既有光照架构：[prism_lighting_gi_architecture_zh.md](prism_lighting_gi_architecture_zh.md)
- GI/Lumen 方向：[prism_gi_lumen_design_zh.md](prism_gi_lumen_design_zh.md)
- 光照契约：[../pkg/prism_render_architecture/src/lighting/contract.rs](../pkg/prism_render_architecture/src/lighting/contract.rs)
- ECS 光源入口：[../pkg/prism_render_scene/src/lighting/extract.rs](../pkg/prism_render_scene/src/lighting/extract.rs)
- 宿主光照入口：[../pkg/prism_render_app/src/lighting.rs](../pkg/prism_render_app/src/lighting.rs)
