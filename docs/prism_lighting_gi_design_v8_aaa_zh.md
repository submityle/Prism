# Prism 原生 Light / Shadow / GI / Reflection 总体设计 v8.0

> 日期：2026-10-20
>
> 定位：面向通用实时 3D 引擎的生产级照明系统设计、破坏性迁移方案与验收基线。
>
> 状态：目标架构，不是功能完成声明。代码、GPU capture、HDR readback、性能原始数据和自动化测试优先于本文。
>
> 兼容策略：允许删除旧 Bevy Light 权威、旧双写路径和长期兼容层；迁移只保留短期、单向、可删除的导入适配器。

## 0. 执行结论

Prism 缺少的不是一个名为 `LightPlugin` 的模块，而是一个完整的照明产品域：从作者输入、物理单位、稳定身份、GPU 场景、工作生成、阴影、GI、反射、时域重建，一直到 HDR 合成、诊断、性能预算和编辑器体验，必须形成唯一数据流和可验收闭环。

v8 的核心决策如下：

1. `LightScene`、`EmitterScene`、`EnvironmentScene` 与 `RadianceScene` 分域拥有数据，但共享 identity、revision、budget、capability 和 explain 协议。
2. 直接光采用 exact clustered/tiled 与 stochastic many-light 双路径，由 planner 根据 overlap、屏幕覆盖、阴影成本和设备能力选择，而不是仅按灯数切换。
3. 阴影采用 provider 模型。低端保留传统 shadow map，中高端使用 virtual shadow pages，高端可选 ray visibility；VSM 不是所有软阴影问题的唯一答案。
4. GI 是 screen、probe/cache、world trace 三层覆盖体系；screen-space 永远不是完整世界真值，trace 命中必须回到材质和纹理权威执行 `RadianceQuery`。
5. 镜面采用 SSR/SSSR、probe/environment、world trace 的单调回退链；任何 miss 不得默认输出黑色。
6. 所有 GPU 表使用 generation-safe stable handle、增量 dirty range、精确 submission proof 和延迟回收；设备恢复只能从 canonical CPU snapshot 重建。
7. 时域复用必须绑定 surface、material、light、environment、view、resolution 和 sample pattern 身份；每种信号拥有独立 history，不建立“万能时域缓存”。
8. 性能目标必须绑定硬件、API、驱动、分辨率、内部比例、场景哈希、相机路径、profile、seed 和 P50/P95/P99；不再给无上下文的“AAA”“4096 灯”“4.3 ms”承诺。
9. 所有不支持、拒绝、降级、延迟、页 miss、历史拒绝和预算超限都必须结构化可解释；有效灯不能因同帧某个不支持灯型而整体失效。
10. 先完成可证明的垂直切片，再扩展算法广度；模块存在、shader 编译或 null backend 通过不等于产品能力完成。

目标主链：

```text
Authoring / Import / Runtime API
        ↓ validate + normalize + journal
Canonical Scene Domains
  Light / Emitter / Environment / Geometry / Material
        ↓ immutable snapshot + SceneDelta
GPU Scene Reconcile
  stable tables / acceleration / distributions / caches
        ↓ per-view planning
Lighting Work Graph
  direct / shadow / diffuse GI / specular / volume
        ↓ reconstruct + compose
Scene-linear HDR → exposure → display transform → present/capture
        ↓
Receipt / Telemetry / Explain / Replay
```

## 1. 产品目标、适用范围与非目标

### 1.1 产品目标

- 支持 Directional、Point、Spot、Rect、Disk、Capsule/Line、Sky/Environment、Emissive Mesh 八类发光语义。
- 支持 `Static / Stationary / Dynamic` 三种 mobility，但 mobility 只表达更新和缓存语义，不直接硬编码某个算法。
- 同一 canonical scene 可被桌面实时、低端保底、编辑器预览、离屏 capture 和未来 reference renderer 消费。
- 支持物理单位、色温、IES、light function、barn doors、source radius/length、light linking、direct/indirect/specular/volume contribution。
- 在低灯重叠时提供确定性 exact direct，在高灯重叠和 emissive-heavy 场景提供受控随机采样。
- 支持动态开放世界的局部更新、流送、设备恢复、历史失效和缓存回收。
- 提供稳定、可解释的质量档；画质降低时不应出现额外能量、闪烁、黑洞或不可追踪的 feature pop。
- 提供编辑器调试视图、性能捕获、离线复现包、CPU oracle、GPU intermediate parity 和 HDR 图像回归。

### 1.2 明确不在首个生产闭环内

- 虚拟几何、毛发、水、云和体积天气的完整产品实现。
- 移动端专用 deferred/forward+ 管线。
- 电影级离线路径追踪器、路径引导和谱渲染。
- 机器学习降噪器；首期使用可跨厂商验证的时空滤波。
- 把所有旧 API 永久保留；迁移完成后删除兼容入口。

### 1.3 “AAA”在本文中的定义

“AAA 级”不是某个算法名，而是以下能力同时成立：

- 大内容规模下行为稳定；
- 画质、性能、显存和延迟可量化；
- 设备能力不足时有单调且可解释的回退；
- 作者不需要理解内部每个 pass 才能得到可靠结果；
- 工程团队能定位错误、复现帧、扩展灯型并维护 ABI；
- 功能声明有真实 GPU、图像和场景矩阵证据。

## 2. 行业方案参考与 Prism 取舍

### 2.1 Unreal Engine：Lumen 与 Virtual Shadow Maps

可借鉴：

- screen trace 优先、可靠 world representation 补盲的分层覆盖思路；
- GI 与 reflection 共用场景表示和调试可视化，但保留不同的采样与重建策略；
- virtualized shadow depth、按需页、缓存和高分辨率局部覆盖；
- scalability profile 与明确的性能目标场景。

不直接复制：

- 不把某一种 surface cache、distance field 或硬件 RT 表示固化为上层语义；
- 不把 VSM 与特定虚拟几何实现绑死；
- 不用“相同功能名”推导性能或画质对等，所有对标使用同硬件、同内容、同分辨率和同指标。

### 2.2 Unity：Adaptive Probe Volumes

可借鉴：

- 基于几何密度的分级 brick；
- per-pixel probe sampling；
- 大世界 streaming、sky occlusion、scenario blending；
- authoring override、泄漏修复和可视化工具。

Prism 的差异：

- ProbeVolume 是一种低频 radiance provider，不是整个 GI 产品；
- 动态直接光、screen detail 和 world trace 不写回为同一份不可辨识的 probe 真值；
- brick residency、scenario、validity 和 leak correction 都进入统一 budget/explain。

### 2.3 NVIDIA ReSTIR / RTXDI

可借鉴：

- 候选生成、target function、proposal PDF、reservoir、temporal/spatial resampling 的严格分层；
- 为多灯和 emissive triangle 提供与灯总数弱相关的采样预算；
- sampling library 通过 renderer callback 深度接入材质、G-buffer、场景和可见性，而不是伪装成即插即用黑盒。

Prism 的边界：

- ReSTIR 是 recipe，不是 Light API、GI API 或场景权威；
- reservoir 必须带 source kind、slot、generation、content revision、PDF measure 和 history identity；
- biased 模式必须登记偏差、能量损失和适用 profile；不能只以“更干净”判定正确。

### 2.4 AMD FidelityFX Denoiser / SSSR

可借鉴：

- 阴影、反射和 GI 使用专用重建器，而不是一套通用 blur；
- tile classification 跳过无工作区域；
- roughness、variance、history length 和 disocclusion 驱动可变成本。

Prism 的取舍：

- denoiser 输入/输出采用引擎自有 contract，第三方实现只是 provider；
- 任何 denoiser 都不得隐藏上游 estimator 的无效 PDF、错误 identity 或能量问题。

### 2.5 总体取舍表

| 领域 | 借鉴 | Prism 采用 | Prism 拒绝 |
|---|---|---|---|
| GI 覆盖 | Lumen 分层追迹 | screen → probe/cache → world trace | screen-only 宣称完整 GI |
| 阴影 | VSM 按需页 | provider + virtual page pool | 所有灯无条件 VSM |
| 低频间接光 | Unity APV | adaptive brick + streaming | 固定全局均匀网格 |
| 多灯 | ReSTIR/RTXDI | exact/stochastic 双路径 | 按总灯数单阈值切换 |
| 降噪 | FidelityFX 专用滤波 | signal-specific reconstruction | 万能 history/blur |
| 设备支持 | 多能力档 | capability-driven recipe | 编译期开关等同产品档位 |

## 3. 设计公理与不可破坏不变量

1. **语义单源**：一个领域只能有一个 authoritative owner。
2. **物理量显式**：单位、色域、测度、PDF 和能量转换不允许藏在 shader magic number 中。
3. **稳定身份优先**：所有跨帧状态先证明 identity，再讨论复用。
4. **Build/Execute 分离**：Build 冻结计划，Execute 只消费计划，不隐式改变 canonical scene。
5. **增量默认**：变化驱动更新；全量重建必须有原因、频率和成本记录。
6. **预算先于执行**：不满足 capability、residency 或预算的 pass 在执行前降级或拒绝。
7. **信号分治**：direct、shadow、diffuse GI、specular、volume 拥有各自 estimator、history 和误差指标。
8. **失败局部化**：单个 emitter 或 provider 失败不能清空同帧其他有效贡献，除非一致性契约明确要求整帧失败。
9. **回退单调**：回退只能减少样本、范围、频率或模型精度；不得凭空增加能量。
10. **证据优先**：功能完成必须同时有代码接线、真实执行、图像结果、诊断和验收记录。

关键跨层不变量：

```text
scene_lineage + sequence + epoch + revision
device_generation + queue_domain + submission_id
surface_id + instance_generation + material_revision
light_slot + light_generation + light_revision
view_id + view_revision + render_scale + sample_pattern
```

任何缓存或异步结果必须声明它依赖其中哪些维度。

## 4. 系统边界、crate 职责与唯一数据流

| 层 | 唯一职责 | 不允许 |
|---|---|---|
| `prism_render_architecture::lighting` | renderer-independent contract、验证、oracle、planner schema | RHI submit、GPU buffer、ECS 专用组件 |
| canonical scene owner | Light/Emitter/Environment 的增删改、journal、snapshot | provider 私有副本成为第二权威 |
| `prism_render_scene` | geometry、instance、material、texture、surface reconstruction、RadianceQuery | 重新定义 Light registry |
| `prism_render_app` | extract、GpuScene、资源生命周期、RenderGraph、queue、receipt | 再定义物理单位和作者语义 |
| editor/import/runtime adapters | author schema、导入、迁移、检查器 | 直接写 GPU ABI |
| debug/telemetry | explain、capture、replay、golden、benchmark | 改变执行结果 |

推荐依赖方向：

```text
lighting_contract
    ↑
canonical_scene ← authoring/import adapters
    ↑
render_scene (geometry/material/radiance)
    ↑
render_app (GPU lifecycle/frame graph/providers)
    ↑
editor/debug/benchmark
```

不要为每个子系统立即拆新 crate。只有满足独立 owner、稳定 API、可独立测试和明确依赖方向时才拆分；否则先在现有 crate 内按模块边界实现，避免“目录架构完整、运行闭环缺失”。

## 5. Canonical Scene 模型

### 5.1 四个权威域

#### LightScene

保存 analytic/extended lights 的作者语义、形状、单位、mobility、贡献开关、shadow request、linking 和 revision。

#### EmitterScene

保存 emissive primitive/triangle/instance 的面积、法线、radiance、纹理与材质 revision、单/双面语义和采样分布身份。

#### EnvironmentScene

保存 sky model、environment map、太阳关联、地平线、atmosphere/sky radiance provider、曝光无关的 scene-linear 辐射和分布 revision。

#### RadianceScene

负责由几何命中恢复材质、纹理、法线、emission、BSDF 和 surface flags。它不是另一份 scene copy，而是 geometry/material owner 暴露的查询接口。

### 5.2 为什么不能把一切塞入 LightTable

- Directional/Point/Spot 是 delta 或局部 analytic emitter。
- Rect/Disk/Capsule 是连续面积或线段测度。
- Environment 通常是球面方向分布。
- Emissive mesh 是面积分布并依赖材质/纹理。
- Probe/cache 是 radiance representation，不是发光体。

它们可以共享 source identity、预算和 explain，但不能共享未定义的 PDF、衰减和可见性公式。

### 5.3 Canonical light common fields

```text
handle: { scene_lineage, table, slot, generation }
kind / mobility / enabled
transform / bounds / range
color_linear / color_space / temperature
physical_intensity + unit
direct / diffuse_indirect / specular / volume weights
shadow_request / source_extent
channel_mask / layer_mask / visibility_mask
importance / priority / update_policy
content_revision / transform_revision / policy_revision
provenance / debug_name
```

形状 payload 独立分表，必须定义 neutral payload、alignment、stride、layout hash 和 CPU/GPU parity。

## 6. 作者 API、导入与易用性

### 6.1 API 原则

```rust
let sun = lighting.add(
    Light::directional()
        .illuminance_lux(100_000.0)
        .angular_diameter_degrees(0.53)
        .mobility(Mobility::Dynamic)
        .shadow(ShadowRequest::Auto),
)?;

let key = lighting.add(
    Light::spot()
        .intensity_candela(1_500.0)
        .outer_angle_degrees(25.0)
        .ies(profile)
        .shadow(ShadowRequest::Quality(ShadowQuality::High)),
)?;
```

- 单位进入方法名或强类型，禁止裸 `intensity: f32` 承担多种含义。
- 非 finite、负值、非法 cone、互斥颜色输入和不支持 modifier 返回结构化错误。
- clamp 只用于可恢复的编辑体验，并必须产生 warning；runtime/cook 可配置为 hard error。
- `Auto` 表示 planner 决定，不表示隐藏结果；编辑器能展示最终 provider、预算和降级链。

### 6.2 导入边界

- glTF、USD、UE 或其他 DCC 的单位和坐标只在 adapter 归一化一次。
- 每次转换保留 provenance、原始值、换算版本和警告。
- 导入器不能绕过 canonical validation，不能直接构造 GPU payload。
- schema migration 采用显式版本和批处理报告；破坏性升级必须能扫描项目并给出修复建议。

### 6.3 面向艺术家的默认体验

- 预设只填充明确参数，不建立第二套行为逻辑。
- 提供 Indoor、Outdoor、Night、Stylized、Showcase 等 profile 起点，但最终结果仍由同一 planner 解释。
- 默认展示“影响范围、阴影成本、GI 贡献、当前 provider、历史有效率”，而不是暴露几十个内部常量。
- 一键 capture 生成场景哈希、灯表 revision、质量档、GPU 信息、预算快照、seed 和关键中间图。

## 7. Stable Identity、增量表与 GPU 生命周期

### 7.1 Handle 与 ABA 防护

```text
LightHandle = { scene_lineage, table_id, slot, generation }
```

删除后 slot 先写 neutral payload，待对应 queue 的完成证明满足后才进入 free list；复用时 generation 增加。旧 handle、旧异步结果和旧 history 必须被拒绝。

### 7.2 UploadPlan 事务

```text
reconcile delta
→ freeze immutable UploadPlan
→ stage/write dirty ranges
→ record passes
→ submit
→ acknowledge(plan, exact SubmissionProof)
→ reclaim(CompletionProof)
```

规则：

- plan 创建后的新变化不能被旧 acknowledge 清掉；
- directional、punctual、area、environment、emitter 等表分别确认，局部失败只保留对应 pending plan；
- skipped frame、surface acquire failure 和 pre-submit error 不得推进 acknowledged frontier；
- present failure 若发生在 GPU 成功提交之后，上传可按真实 receipt 确认，但呈现错误单独报告；
- device recovery 增加 `device_generation`，拒绝旧 plan/proof，并从 canonical snapshot 全量重建。

### 7.3 空洞、容量与 compaction

- shader 遍历上限必须与表布局契约一致，不能把 `live_len` 错当最大 slot。
- neutral holes 保证正确性，但会增加迭代成本；planner 根据 hole ratio、访问成本、内存压力和历史失效代价决定 compaction。
- compaction 是低频显式事件，必须生成 handle remap 或整体 epoch 失效，不得静默改变 identity。
- 固定容量仅可作为某 profile 的受测限制，不可成为无诊断硬编码。

## 8. Direct Lighting 与多灯工作生成

### 8.1 Exact path

适合低至中等 overlap、稳定确定性和低端设备：

1. GPU view culling 生成影响当前 view 的 light bounds。
2. tile/cluster 构建 compact light lists。
3. material shading 按 light kind 分组或 bin，减少 divergence。
4. 每灯执行形状 eval、衰减、BRDF、visibility 和 contribution mask。

overflow 不允许静默截断。可选择：扩大临时池、分批执行、切换 stochastic recipe、降低 shadowed-light 数或显式拒绝低优先级贡献。

### 8.2 Stochastic many-light path

适合高 overlap、面积光和大量 emissive sources：

```text
source distribution / light tree / ReGIR-like grid
→ initial candidates
→ target evaluation
→ temporal resampling
→ spatial resampling
→ final visibility
→ radiance + moments + confidence
```

reservoir 最少携带：

```text
source_kind / slot / generation / source_revision
sample_parameterization / measure / proposal_pdf
target_value / weight_sum / candidate_count
surface_identity / view_identity / sample_pattern
visibility_revision / age / confidence
```

delta light、面积 emitter、球面环境分布必须明确 measure；不能把离散 PMF 与 solid-angle PDF 混用。

### 8.3 Recipe 选择

planner 输入包括：

- visible light count、tile overlap 分布和 projected coverage；
- shadowed-light 比例和 visibility 成本；
- emitter triangle count、分布构建/更新成本；
- material lobe 和 shading cost；
- temporal validity、camera motion 和 disocclusion；
- GPU subgroup、atomics、storage、RT 能力与显存压力；
- 当前 direct-light budget 和最近观测耗时。

切换采用 hysteresis，避免 exact/stochastic 每帧抖动。切换必须清理或转换相关 history，并在 Explain 中记录原因。

## 9. Shadow 系统

### 9.1 作者意图与执行 provider 分离

`ShadowRequest` 只表达是否需要、软硬程度、更新偏好和质量重要性。可选 provider：

- conventional map / cascade / clipmap；
- virtual shadow pages；
- ray-traced visibility；
- contact/screen refinement；
- baked/static cache。

同一个 light 可按 view/profile 选择不同 provider，但选择结果必须稳定、可解释、可捕获。

### 9.2 Virtual Shadow Page contract

页 identity：

```text
light_generation
+ shadow_projection_revision
+ caster_cell_revision
+ virtual_page_coordinate
+ quality_revision
+ device_generation
```

页系统必须具备：

- receiver-driven demand；
- deterministic priority 和 allocation；
- physical pool、page table、pin/evict、dirty tracking；
- static/stationary cache 与 dynamic overlay；
- coarse fallback、missing-page policy 和防闪烁策略；
- page request、render、hit、miss、eviction、fallback telemetry。

### 9.3 各灯型策略

| 类型 | 首选 | 备选 | 关键风险 |
|---|---|---|---|
| Directional | clipmapped virtual pages | cascaded map | 大世界 invalidation、远景稳定性 |
| Spot | perspective virtual/map | conventional map | cone 边界、分辨率浪费 |
| Point | octahedral/cube virtual pages | cubemap map | 六面成本、接缝、页爆炸 |
| Rect/Disk | ray visibility 或多样本近似 | map + source-size filter | VSM 不自动等于真实面积软影 |
| Capsule/Line | ray/analytic approximation | capsule/contact proxy | 近场形状误差 |
| Sky/Environment | occlusion provider | probe sky occlusion | 不能当作单一 shadow map |
| Emissive | sampled visibility | contact/cache | source 数量与噪声 |

### 9.4 退化顺序

退化由质量贡献/成本比驱动，但必须满足项目约束。推荐顺序：降低远距离分辨率 → 降低更新频率 → 关闭低重要性局部灯阴影 → 使用 coarse/contact fallback → 最后拒绝贡献。主方向光、关键角色灯等可由 policy pin 住，但 pin 的预算影响必须可见。

## 10. Geometry、Material 与 RadianceQuery

### 10.1 命中身份

```text
primitive_id
+ instance_generation
+ geometry_revision
+ material_id
+ material_revision
+ barycentric / local coordinates
```

software BVH、hardware RT、SDF 或其他 representation 只负责返回稳定命中，不拥有材质真值。BLAS/TLAS reorder、compaction 和 streaming 不得使旧 primitive index 被错误解释为新表面。

### 10.2 RadianceQuery

统一查询至少返回：

- geometric/shading normal、tangent frame、UV；
- base color、roughness、metalness、IOR、transmission、opacity；
- emission 与其纹理/颜色空间；
- material closure 或受控 Material IR；
- surface flags、light linking、two-sided 和 alpha policy；
- query support level 与 fallback reason。

首个切片可以只支持不透明 Builtin Lit/Unlit，但不支持的材质必须 reject、标记或走明确 fallback，不能用常量三角 radiance 冒充生产材质。

### 10.3 Emissive distribution

- 构建 area × luminance 的基础分布，并允许按空间、方向和材质重要性分层。
- 动态纹理、材质参数、实例变换变化只更新受影响 leaf/distribution segment。
- proposal PDF、最终 solid-angle PDF、visibility 和 MIS/resampling 权重必须能在 CPU oracle 重算。
- 同一 emission 不能既作为材质命中贡献又被错误重复注入为独立点光。

## 11. GI 系统

### 11.1 信号分层

| 层 | 优点 | 局限 | 适用档 |
|---|---|---|---|
| Screen trace | 低延迟、当前视图细节 | 屏幕外缺失、disocclusion | H2+ |
| Probe/radiance cache | 稳定、低频、可流送 | 泄漏、高频与近场不足 | H1+ |
| World trace | 离屏几何、动态遮挡 | 成本高，依赖完整材质命中 | H3+ |
| Reference/path | 质量基准 | 非实时 | 验收/离线 |

### 11.2 Probe Volume

采用 adaptive brick，而非固定全世界均匀网格：

- 几何密集、室内、门户和光照变化快的区域提高密度；
- 空旷远景降低密度并允许 streaming；
- probe validity、relocation、classification、sky visibility 和 leak correction 显式存储；
- day/night 或情景切换使用 scenario identity 与受控 blending；
- brick residency、update queue、dirty region 和 interpolation confidence 进入 telemetry。

probe spacing 不写死为 1m。它由场景尺度、几何复杂度、内存、质量 profile 和 leakage 测试共同决定。

### 11.3 Screen 与 world trace 组合

推荐覆盖顺序：

```text
screen hit with valid material
    → use screen result
screen miss / invalid
    → world trace
world miss / unsupported / over budget
    → probe or environment
```

这是一条分层 fallback，不等同于随意线性混合。若多个 estimator 共同贡献，必须定义共同 target、support、PDF 和 bias；不能只按经验权重相加。

### 11.4 多 bounce 与缓存

- 首个生产闭环优先保证一次间接反弹的材质正确性、miss environment 和时域稳定。
- 多 bounce 可由 radiance cache/probe 近似或 reference path 提供，不应在基础闭环未完成时先堆叠递归次数。
- cache key 包含 position cell、normal cone、material class、scene/light/environment revision 和 device generation。
- cache 更新、读取、淘汰和失效必须可回放；禁止 stale radiance 跨场景污染。

### 11.5 ReSTIR GI 进入条件

只有以下前置完成后才进入生产：

- hit-time RadianceQuery 完整；
- sample position/normal/radiance/PDF 定义明确；
- temporal/spatial identity 与 Jacobian/normalization 正确；
- disocclusion、dynamic geometry 和 material change 的拒绝测试通过；
- 与 reference path 的固定样本、固定时间两类误差报告完成。

## 12. Reflection 系统

### 12.1 分层回退

```text
planar override（显式、少量）
→ SSR/SSSR valid hit
→ world-space reflection trace
→ local reflection probe
→ global environment
```

并非每个像素都从第一层开始；planner 根据 roughness、材质域、屏幕覆盖、ray budget 和 probe validity 分类。

### 12.2 关键契约

- roughness 决定 ray/cone、mip、采样率和 denoiser，不得只作为最后 blur 半径。
- SSR hit 必须通过厚度、深度层级、法线、材质和边界 confidence 验证。
- probe parallax correction、box influence 和 blending 必须有显式 validity。
- world miss 返回 EnvironmentScene，不返回黑色。
- diffuse GI 与 specular 不共享 reservoir/history；同一辐射源可以共享身份和场景查询。

## 13. Temporal Reconstruction 与 Denoising

### 13.1 HistoryKey

```text
signal_kind
+ surface_id / instance_generation
+ material_revision
+ light_or_environment_epoch
+ view_id / projection_revision
+ render_extent / render_scale
+ sample_pattern / algorithm_revision
```

### 13.2 拒绝与限幅

- camera cut、teleport、projection jump：hard reset；
- depth/normal/material discontinuity：pixel/tile reject；
- light influence、emitter distribution、environment 变化：按受影响区域 soft invalidate；
- DRS 改变：重采样或 reset 由 recipe 显式声明；
- device recovery：所有 GPU history hard reset。

每个 signal 记录 history length、accept/reject ratio、variance、clamp rate、disocclusion 和 ghosting proxy。

### 13.3 专用重建器

- soft shadow mask：visibility-aware temporal + edge-preserving spatial；
- glossy reflection：roughness-aware variable rate + hit distance；
- diffuse GI：radiance/moments + normal/depth/material；
- reservoir：先验证 sample identity 和 normalization，再进入图像域重建。

“噪声减少”不自动代表质量提高；必须同时检查 bias、lag、energy、ghosting、detail loss 和响应时间。

## 14. Composition、曝光与颜色管理

统一 scene-linear HDR 合成顺序：

```text
emission
+ direct diffuse/specular
+ indirect diffuse
+ indirect specular/reflection
+ participating media
→ exposure
→ tone/display transform
→ UI/compositor policy
→ present/capture
```

- 任何 lighting pass 不得私自 gamma、tone map 或重复曝光。
- Environment 作为 miss radiance、reflection 和 diffuse source 时必须避免重复注能。
- pre-exposure 若用于数值稳定，必须是可逆的执行细节，并在 capture/oracle 中还原一致语义。
- HDR10/SDR/offline EXR 共用 scene-linear 源，但拥有不同 display/output transform。

## 15. Planner、Capability、Quality 与 Budget

### 15.1 CapabilityGraph

能力不是一个 `supports_rt` 布尔值，至少描述：

- API/backend、shader model、storage/texture limits；
- subgroup、atomics、indirect dispatch、timestamp；
- bindless/descriptor indexing 能力和限制；
- acceleration structure、ray query/pipeline；
- async queue 的实际并发能力；
- transient memory、residency 和 format support。

recipe 声明 required、preferred 与 fallback capability。设备不满足 required 时在 Build 阶段拒绝。

### 15.2 Quality profile

| Profile | Direct | Shadow | Diffuse GI | Specular | 目标设备语义 |
|---|---|---|---|---|---|
| H1 | exact、低 overlap | 主光 conventional | probe | probe/environment | 低预算无 RT |
| H2 | exact/stochastic 混合 | 主光 virtual + 局部 map | screen + probe | SSR + probe | 主流桌面/主机级 |
| H3 | stochastic many-light | virtual + optional ray | screen + world + cache | SSR + world + probe | 高端桌面 |
| H4 | reference sampling | reference visibility | path/reference | path/reference | 离线验收 |

profile 是能力和预算集合，不绑定某个具体 GPU 型号，也不承诺固定 FPS。

### 15.3 Budget ledger

每个计划声明：

```text
estimated_gpu_time / cpu_time
persistent_bytes / transient_bytes / upload_bytes
rays / samples / pages / candidates
history_bytes / cache_residency
quality_gain / fallback / synchronization
```

预算分区：direct、shadow、GI、reflection、reconstruction、streaming/cache、reserve。动态控制器使用历史观测校准估计，但必须有 hysteresis、变化速率限制和最小驻留时间。

### 15.4 PlanDecision

```text
Selected(recipe, budget)
Rejected(reason)
Degraded(from, to, reason)
Deferred(until, reason)
```

reason 使用稳定枚举并附上下文，例如 `UnsupportedLightKind`、`MissingCapability`、`TransientBudgetExceeded`、`ShadowPoolExhausted`、`HistoryInvalid`、`MaterialUnsupported`、`DeviceGenerationMismatch`。

同一场景中某个 Rect 或 Emissive 尚不支持时，必须保留 Directional/Point/Spot 等有效贡献，并把拒绝挂到对应 emitter identity。

## 16. 性能与内存设计

### 16.1 CPU

- ECS/authoring 变化先变成紧凑 `SceneDelta`，不做全场景扫描。
- stable table reconcile 与 dirty-range 合并按变化量扩展。
- provider build 可并行，但最终 plan 排序和资源分配必须确定性。
- shader/PSO 在后台预热；执行线程不阻塞编译未知 variant。
- 统计 P50/P95/P99、最大值和 hitch source，不只记录平均值。

### 16.2 GPU

- GPU-driven culling、binning 和 compact work list，避免按最大容量无条件遍历。
- SoA/分表减少无关字段带宽；热数据与冷诊断数据分离。
- tile classification 跳过无光、无阴影、无反射或 history 已收敛区域。
- half/quarter resolution 只用于可重建信号；边缘、镜面和薄几何需要自适应全分辨率。
- async compute 只在 capture 证明有真实 overlap 且不增加关键路径等待时启用。

### 16.3 内存

使用预算池而非散装分配：

| 池 | 内容 | 压力策略 |
|---|---|---|
| Light/Emitter | stable payload、分布、树 | compact、降低更新频率、拒绝低优先级 source |
| Shadow | page table、physical pages、cache | evict、coarse fallback、关闭低优先级阴影 |
| Probe/Cache | brick、SH/radiance、metadata | stream、降低密度、延迟更新 |
| Trace | AS、ray/hit queue | 降低范围/频率/rpp、切 provider |
| History | reservoirs、moments、radiance | 降低分辨率/历史长度、按 signal 回收 |
| Transient | candidates、bins、scratch | alias、分批、拒绝超预算 plan |

具体 MB 数值必须由受测 workload 得出；v3.1 的 96 MB VSM、64 MB probe 等数字保留为初始实验上限，不作为跨平台承诺。

### 16.4 可复现性能 contract

每份报告必须包含：

```text
hardware + driver + OS + API
resolution + internal scale + output mode
scene hash + asset revision + camera path
profile + recipe set + seed
warmup + sample count + thermal/power state
CPU/GPU P50/P95/P99 + VRAM peak/resident
```

同时记录 light overlap、shadow pages、rays/samples、reservoir reuse、history reject、cache hit、upload bytes、compaction 和 PSO miss。只有这些输入一致时，才能比较 Prism 与其他产品或不同版本。

## 17. Debug、Explain、Capture 与可维护性

### 17.1 Debug views

- light influence、tile/cluster occupancy、overflow；
- source kind、proposal PDF、selected sample、reservoir age/confidence；
- shadow request/page residency/miss/eviction；
- GI source、screen/world/probe coverage、trace miss；
- probe brick level、validity、sky visibility、leak risk；
- reflection source、hit distance、roughness rate；
- history acceptance、variance、clamp、disocclusion；
- unsupported material/light、fallback 和成本热图。

### 17.2 Capture/replay

最小复现包：

```text
canonical scene snapshot hash
scene delta journal
capability/profile/budget
all relevant revisions and device generation
recipe plan + resource sizes
random seeds/sample pattern
shader/package hashes
intermediate target checksums or selected readbacks
```

replay 支持 deterministic validation 模式和 production sampling 模式。前者优先可比性，后者评估真实质量与性能。

### 17.3 维护规则

- ABI 从 Rust/IDL 单一源生成 shader 声明，并在 CI 校验 size/alignment/offset/hash。
- 一个物理公式只有一个 reference 实现；GPU 优化版本与 CPU oracle 对拍。
- 大文件按 owner/pass 拆分，禁止再次形成数千行单体 `lighting_gpu.rs`。
- feature flag 只控制 provider 注册或能力，不允许改变相同 API 的物理含义。
- 每个兼容层都有删除条件、owner 和截止 milestone。
- 无 `TODO`/stub/静默 fallback 的完成声明；未实现能力使用显式 rejection。

## 18. 扩展协议

### 18.1 新增灯型

```text
author schema/import
→ canonical validation and units
→ identity/revision/delta
→ CPU eval/sample/pdf oracle
→ GPU payload + neutral value + ABI test
→ candidate generation/direct recipe
→ shadow provider matrix
→ GI/emitter participation
→ budget/explain/debug
→ GPU/image/performance acceptance
```

任何一项缺失都只能标记为 partial capability。

### 18.2 新增 recipe/provider

必须声明：

- target estimator、support、measure、PDF、bias；
- input/output contract；
- capability 与 fallback；
- resource lifetime、同步和预算；
- history identity 与 invalidation；
- deterministic seed 行为；
- oracle、golden 和 benchmark workload。

第三方 provider 与内置 provider 走相同注册、规划和诊断路径，不允许旁路 queue submit 或私有资源生命周期。

## 19. 破坏性迁移

### 19.1 删除项

- 删除 Bevy Light component 作为 Prism runtime authority 的入口。
- 删除 `prism_render_scene` 与 `prism_render_app` 的双重 Light registry。
- 删除每帧 dense rebuild 作为默认灯表上传方式。
- 删除无 generation/epoch/device proof 的缓存和异步 key。
- 删除 provider 私自提交 GPU 工作或写全局 buffer 的路径。
- 删除 unsupported emitter 导致整帧有效灯全部失效的错误策略。
- 删除 BVH miss 默认黑色和命中使用常量 triangle radiance 的生产路径。
- 删除 screen-only GI/reflection 的完整能力宣称。

### 19.2 迁移原则

- 新旧路径不做长期双向同步。
- adapter 只允许 `old → canonical` 单向输入；canonical 永远不回写旧权威。
- 每完成一个垂直切片即删除对应旧入口和测试，避免两套实现继续漂移。
- 删除前先以 `rg`/编译依赖/运行 capture 证明零生产引用。
- 每个阶段建立 source-control checkpoint、迁移说明和 rollback 边界，但不保留运行时双轨。

## 20. 实施路线与退出标准

### M0：契约与证据基础

交付：canonical domain、单位、identity/revision、ABI codegen、PlanDecision、capability、benchmark schema。

退出：contract/ABI/oracle 测试通过；旧入口被冻结；所有 unsupported 都能解释。

### M1：Stable GPU Scene

交付：create/update/remove、dirty range、neutral retirement、submission/completion proof、device replay、compact work list。

退出：slot reuse/ABA、跨设备 proof、skipped submit、partial table acknowledge、compaction 与恢复测试通过。

### M2：三类基础灯的 scene-to-present 闭环

交付：Directional/Point/Spot exact direct、HDR compose、窗口/离屏一致、per-emitter rejection。

退出：CPU/GPU intermediate、HDR readback、present receipt 和混合有效/无效灯场景通过；删除旧 direct authority。

### M3：生产阴影垂直切片

交付：方向光 + 一种局部灯 shadow provider、page/cache telemetry、fallback。

退出：动态 caster、camera move、pool exhaustion、device recovery、软影 bias/噪声测试通过。

### M4：Area 与 Many-Light

交付：Rect/Disk/Capsule、light tree/distribution、stochastic direct、reservoir identity。

退出：面积/PDF/能量 oracle、exact/reference 对比、动态 source 变化和高 overlap 性能曲线通过。

### M5：Emitter 与 RadianceQuery

交付：emissive distribution、材质/纹理命中、environment miss。

退出：动态材质/纹理、实例复用、AS reorder、unsupported material 和无黑 miss 场景通过。

### M6：GI 与 Reflection 覆盖链

交付：adaptive probes、screen trace、world trace、SSR/probe/environment fallback、signal-specific denoise。

退出：离屏遮挡、disocclusion、roughness 阶梯、漏光、历史响应和能量重复注入测试通过。

### M7：产品化

交付：编辑器检查器、profiles、Explain、DebugView、capture/replay、导入迁移、硬件矩阵。

退出：目标平台 CI、场景矩阵和 P50/P95/P99 报告可复现；删除全部迁移兼容层。

### M8：高端增强

候选：hardware RT provider、ReSTIR GI、advanced radiance cache、volumetric lighting、reference/offline integration。

进入条件：M0–M7 的 identity、材质、诊断和验收基础已经稳定；每项单独做 go/no-go benchmark。

## 21. 验收矩阵

### 21.1 场景集

| ID | 场景 | 核心验证 |
|---|---|---|
| S1 | 室内 Point/Spot | 单位、衰减、cone、局部阴影、动态遮挡 |
| S2 | 户外 Directional/Sky | 远近阴影、环境、曝光、无重复注能 |
| S3 | 8→4096 灯压力 | overlap、exact/stochastic、overflow、斜率、显存 |
| S4 | Rect/Disk/Capsule | 采样/PDF、source size、软影偏差与噪声 |
| S5 | Emissive-heavy | 分布更新、材质纹理、visibility、reservoir identity |
| S6 | 动态几何/材质 | hit identity、AS reorder、history reject |
| S7 | 离屏 GI | screen miss、world/probe fallback、漏光与黑洞 |
| S8 | 镜面 roughness 阶梯 | SSR/world/probe 切换、接缝、denoise |
| S9 | 大世界流送 | brick/page/AS residency、lineage、cell unload/load |
| S10 | 恢复与失败注入 | surface failure、submit/present、device lost、replay |
| S11 | 透明/薄几何边界 | 明确支持范围、fallback、无伪完成 |
| S12 | 确定性 | fixed seed、计划排序、CPU/GPU 容差、跨帧稳定 |

### 21.2 测试层级

1. 数学单测：单位、衰减、cone、LTC/area PDF、BRDF、MIS/reservoir normalization。
2. 数据契约：ABI、neutral、handle generation、revision、proof、serialization migration。
3. CPU oracle：eval/sample/pdf、shadow projection、probe interpolation、environment distribution。
4. GPU intermediate：buffer、candidate list、page table、reservoir、hit/material reconstruction。
5. HDR 图像：linear error、MSE/PSNR/SSIM/FLIP、flux/energy、ghosting、leak、seam。
6. 失败注入：OOM/预算、pool full、unsupported、device recovery、stale proof/history。
7. 性能：CPU/GPU P50/P95/P99、VRAM、upload、rays/pages/samples、hitch。
8. UX：导入报告、检查器、Explain、DebugView、capture/replay 可完成常见定位任务。

### 21.3 4096 灯与 4.3 ms 的正确用法

v3.1 中的指标保留为候选 benchmark，不再是无条件产品承诺：

- `4096 lights` 必须指定可见数、重叠分布、shadowed 比例、source 类型和材质复杂度；
- `≤25% slope` 必须定义相对哪一基线、测量 direct-only 还是完整 frame；
- `4.3 ms@1440p` 必须指定 GPU、内部比例、API、场景、profile、温度、P50/P95/P99；
- 成功标准同时包含图像误差和时域稳定，不能通过关闭有效贡献取得性能绿灯。

## 22. 当前仓库状态快照

截至 2026-10-20，可确认的状态分为“已验证基础”“当前工作树候选”和“未闭环”三类。

### 22.1 已验证基础

- architecture 层已有 Light contract、recipe/plan、trace/cache/replay 和 reference 资产。
- stable table 已具备 slot/generation、neutral retirement、submission/completion proof、device generation/queue domain 和恢复重建语义。
- `prism_render_architecture` 的 incremental table 定向测试已取得 `13 passed; 0 failed`；这证明表生命周期单测通过，不证明整个照明产品完成。

### 22.2 当前工作树候选，尚需完整集成证明

- app 侧已存在 Directional/Point/Spot canonical payload、增量上传和真实 submission receipt 接线。
- directional 与 punctual stable upload 已统一为双表事务：prepare 先完成双侧校验、容量预测和必要的 wait-idle，再改变任一表；ack 先验证两侧 plan、domain、epoch、revision 与 submission，再统一确认。
- upload transaction 已绑定单调 token，迟到的旧 submission 不会确认后续 frame 的 pending plan；compaction 失败保持两张表和两套 canonical-slot map 不变。
- per-emitter rejection、有效灯保留和旧 Bevy lighting authority 删除正在并行迁移。
- 事务边界已补充应用层定向测试；受当前工作树物理模块编译和宿主磁盘容量影响，完整 app 编译、真实 GPU、HDR 输出和精确 diff 审核仍不能作为已完成声明。

### 22.3 关键未闭环

- Rect/Disk/Capsule 尚未统一进入 stable canonical direct/shadow/GI 路径。
- Sky/Environment/IBL 尚未形成唯一 EnvironmentScene 与 miss-radiance 权威。
- Emissive 缺稳定分布、PDF、visibility、source identity 和 ReSTIR 接线。
- BVH 命中仍需完整 hit-time RadianceQuery；常量 triangle radiance 不能作为生产方案。
- world trace miss、probe、environment 和 reflection fallback 尚未形成完整覆盖链。
- VSM/page residency、全灯型阴影、生产 ReSTIR DI/GI、完整 SSR/probe 反射和 scene-to-present parity 尚未证明。
- 尚无匹配硬件/内容/分辨率/质量的完整 P50/P95/P99 与竞品对标报告。

### 22.4 Bevy 废弃审计结论

当前不能把“Bevy 已废弃”作为全仓完成事实：

- `prism_render_app` 与 `prism_render_architecture` 的生产依赖已经不包含 `bevy_light`、`bevy_pbr`、`bevy_render` 或 `bevy_core_pipeline`，原生 LightScene -> GPU upload 链路已独立成立。
- `prism_render_scene` 仍直接依赖 Bevy RenderApp、`bevy_pbr` mesh/PBR pipeline、`StandardMaterial` adapter、`ExtractSchedule` 和 Bevy RenderGraph；它必须被视为 legacy adapter island，而不是 Prism native renderer。
- 根 `bevy` 的默认组合仍会安装 `bevy_render::RenderPlugin`、`bevy_light::LightPlugin`、`bevy_core_pipeline::CorePipelinePlugin` 与 `bevy_pbr::PbrPlugin`。
- 已删除的旧灯光 extraction 文件不应恢复；下一步应先建立 `prism_native_default` 产品组合并将 `prism_render_scene` 从默认组合剥离，再迁移或删除 Bevy PBR render phase。

## 23. 关键风险与控制措施

| 风险 | 表现 | 控制 |
|---|---|---|
| 架构过度设计 | contract 很全但没有输出像素 | 每阶段以 scene-to-present 垂直切片退出 |
| 双权威漂移 | Bevy/scene/app 各有灯表 | 单向 adapter，逐阶段删除旧 owner |
| 随机算法掩盖错误 | 噪声低但 bias/能量错 | reference、PDF audit、固定样本/固定时间双测 |
| VSM 页爆炸 | 点光/植被/动态 caster 导致 miss | 页预算、priority、coarse fallback、telemetry |
| 时域污染 | 拖影、漏光、跨场景历史 | 完整 HistoryKey、局部失效、失败注入 |
| 软件 RT 吞吐不足 | 高分辨率/复杂材质超预算 | probe/cache 保底、分辨率/rpp 限制、HWRT provider |
| 显存碎片与瞬态峰值 | hitch/OOM | 预算池、alias、residency、预执行拒绝 |
| 质量档抖动 | recipe/分辨率频繁跳变 | hysteresis、rate limit、最小驻留时间 |
| API 难用 | 单位混乱、默认值不可解释 | 强类型单位、presets、validator、Explain |
| 测试假阳性 | null/编译通过被当成完成 | 真实 GPU、HDR readback、图像与性能证据 |

## 24. 最终决策记录

已确定：

- Prism 原生 canonical lighting 是唯一 runtime 权威；Bevy Light 只可作为短期导入源并最终删除。
- stable identity、revision、submission proof 和 explain 是所有高级算法的前置基础。
- exact/stochastic、shadow map/VSM/ray、screen/probe/world 都是 planner 选择的 provider/recipe，不是互相割裂的用户 API。
- Environment 与 Emissive 是一等辐射源，但不错误伪装为 punctual light。
- GI/Reflection 命中必须查询真实材质；trace miss 必须走 EnvironmentScene。
- 性能、画质、易用性、扩展性和维护性共同决定发布，不以单一 demo 画面或平均帧时判断。

必须通过 spike/benchmark 决定：

- exact 与 stochastic 的切换模型；
- compact work list、stable holes 与 compaction 阈值；
- 首发 VSM page size/pool、点光投影表示和软影策略；
- probe brick 密度、更新频率、编码和 streaming 粒度；
- software BVH、hardware RT 和混合 representation 的平台组合；
- 各 profile 的样本、历史、内存和动态质量控制范围。

未经代码、真实执行、图像和性能报告证明的内容，必须继续标记为目标、候选或未闭环。

## 附录 A：建议的核心 contract

```rust
struct SceneIdentity {
    lineage: u64,
    sequence: u64,
    epoch: u64,
}

struct StableHandle {
    table: u16,
    slot: u32,
    generation: u32,
}

struct SubmissionProof {
    device_generation: u64,
    queue_domain: u32,
    submission_id: u64,
}

enum PlanDecision<T> {
    Selected(T),
    Rejected(Reason),
    Degraded { from: T, to: T, reason: Reason },
    Deferred { until: u64, reason: Reason },
}

trait LightSampler {
    fn sample(&self, context: &SampleContext, random: [f32; 4]) -> LightSample;
    fn pdf(&self, context: &SampleContext, sample: &LightSample) -> Pdf;
    fn evaluate(&self, context: &SampleContext, sample: &LightSample) -> Radiance;
}

trait RadianceQuery {
    fn query(&self, hit: SurfaceHit) -> Result<SurfaceRadiance, QueryError>;
}

trait ShadowProvider {
    fn plan(&self, request: ShadowRequest, context: &ViewContext) -> PlanDecision<ShadowPlan>;
}
```

接口名称为设计示意，不要求照抄；不变量和所有权必须保留。

## 附录 B：每个 pass 的声明模板

```text
id / version / owner
inputs / outputs / access mode
required and preferred capabilities
scene and history dependencies
estimated CPU/GPU/memory/upload budget
quality contribution and known bias
fallback and rejection reason
async eligibility and synchronization
determinism / seed behavior
telemetry counters
oracle / golden / benchmark IDs
```

## 附录 C：参考资料

- Epic Games, *Lumen Technical Details*, Unreal Engine documentation.
- Epic Games, *Shadowing in Unreal Engine* and Virtual Shadow Maps documentation.
- Unity Technologies, *Introduction to Adaptive Probe Volumes*, Unity 6 documentation.
- Bitterli et al., *Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting*, SIGGRAPH 2020.
- Ouyang et al., *ReSTIR GI: Path Resampling for Real-Time Path Tracing*, HPG 2021.
- NVIDIA RTXDI, *Integration Guide* and shader API documentation.
- AMD GPUOpen, *FidelityFX Denoiser* documentation.

这些资料用于提炼设计模式与风险，不构成 Prism 已达到对应产品成熟度的声明。
