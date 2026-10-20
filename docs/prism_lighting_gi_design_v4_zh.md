# Prism 次时代光照 / GI 设计 v4.0

> 状态：架构设计，允许破坏性更新。本文不代表所有模块已经实现，也不把预算、画质或 UE 对标写成实测结论。
>
> 版本日期：2026-10-20
>
> 范围：Prism 的统一光源、阴影、直接光、GI、反射、材质辐射查询、缓存、质量分档、诊断与迁移架构。

## 0. 执行摘要

### 0.1 结论

Prism 目前并非“完全没有 light 系统”。当前仓库已经存在两条光照输入路径：

1. prism_render_scene 中由 Bevy ECS 提取 DirectionalLight、PointLight、SpotLight、环境光和环境贴图，再写入 GPU light buffers，并参与 clustered resolve。
2. prism_render_app 中由宿主注入方向光、点/聚光灯、surface sample、GI candidate batch 与 ChangeJournal，编译 LightingPlan，执行 direct/GI/composite 的垂直切片。

真正缺少的是产品级、唯一权威的 LightScene：

- ECS、运行时宿主、导入器各自都可能成为光源表事实来源；
- light buffer 仍以每帧清空、重建、上传为主，稳定句柄、脏区和 fence-safe 回收尚未成为统一路径；
- prism_render_scene 与 prism_render_app 的光照语义、生命周期和帧图入口没有完全收敛；
- 许多 GI、阴影、ReSTIR、探针、材质和参考路径已有可复用模块，但还没有被一条完整的场景到屏幕产品管线统一装配；
- “有算法”“有 shader”“有 CPU golden”与“真实场景可用、可退化、可观测、可维护”必须分开验收。

v4 的核心变更是：

> 以 LightScene 作为唯一光源权威，以 GpuScene 作为 GPU 生命周期权威，以 LightingWorkGraph 作为每视图工作生成权威，以多个可替换的 Visibility / Radiance / Estimator provider 组成高端与兼容档的同一产品路径。

### 0.2 破坏性决策

| 领域 | 旧方向 | v4 决策 |
|---|---|---|
| 光源模型 | 八类类型平铺 | 分为解析光源、扩展光源、环境光源、网格发光体、烘焙/外部辐射源 |
| 所有权 | ECS 与 host-fed lighting 可分别维护表 | 只有 LightScene 持有光源权威；ECS/导入/网络/编辑器都只是 adapter |
| 句柄 | LightHandle(u32) 语义不足 | LightId + Generation + SceneEpoch，删除、复用、设备恢复均可验证 |
| 上传 | 全量 rebuild 为默认 | stable slot + generation + dirty range；全量 rebuild 仅用于恢复和验证 fallback |
| 阴影 | 倾向所有灯进入 VSM | 根据形状、距离、质量意图和 backend 选择 VSM、RT、screen contact 或 cached/baked |
| 直接光 | 固定单一算法 | exact clustered、stochastic direct、ReSTIR DI 按 workload 路由 |
| GI | screen GI + probe + world trace 直接相加 | 每条估计器声明 measure、PDF、bias、confidence；只按兼容规则组合 |
| RT | WebGPU-first 等同高端上限 | WebGPU/software 是可移植基线；硬件 RT 是高端能力 |
| 质量 | 固定 H1–H4 recipe | CapabilityProfile + QualityIntent + ResourceBudget + ResolvedQualityState |
| 验收 | 模块存在即视为完成 | 垂直切片、P50/P95/P99、图像误差、能量误差、退化单调性和诊断闭环 |

### 0.3 不承诺

本文不承诺：

- 在 wgpu/WebGPU 上复制 UE、Lumen、VSM 或 MegaLights 的全部底层能力；
- 仅凭 4096 个灯的数量指标保证所有内容场景的画质和噪声；
- 用一个全局归一化因子自动修正所有 GI 能量问题；
- 在同一帧内无代价完成完整几何、材质、纹理、光照、GI、反射、透明、体积和后处理；
- 仅凭 CPU 单测证明 GPU 图、真实场景和最终出图已经完成；
- 将 H4 offline/reference 作为实时质量档位；
- 把 stylized/NPR 艺术控制与物理能量守恒混成同一验收规则。

## 1. 设计原则与外部参考

### 1.1 应该参考优秀产品的什么

UE 的 Lumen、Virtual Shadow Maps 和 MegaLights 值得参考的不是功能名称，而是系统工程：

- Lumen 把 screen trace、可靠的 world tracing、surface cache 和跨帧更新组合起来，说明 GI 必须同时管理几何表示、命中点辐射、缓存覆盖和退化路径。
- Virtual Shadow Maps 通过按需 page、缓存和失效管理控制高分辨率阴影成本；它不是简单提高 shadow map 分辨率。
- MegaLights 把大量动态灯的直接光和阴影转化为重要性采样问题，同时承认固定样本预算会带来噪声、重建和 ghosting 风险。

参考：

- Epic：Lumen Technical Details  
  https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine
- Epic：Virtual Shadow Maps  
  https://dev.epicgames.com/documentation/en-us/unreal-engine/virtual-shadow-maps-in-unreal-engine
- Epic：MegaLights  
  https://dev.epicgames.com/documentation/en-us/unreal-engine/megalights-in-unreal-engine
- NVIDIA Research：ReSTIR  
  https://research.nvidia.com/labs/rtr/publication/2020/Spatiotemporal-reservoir-resampling-for-real-time-ray-tracing-with-dynamic-direct-lighting/

### 1.2 五条硬原则

1. **单一权威**：每个业务域只允许一个 authoritative owner。ECS 不直接拥有 GPU 光源表，GPU pass 不自行发明光源语义。
2. **契约先于算法**：先定义单位、测量空间、生命周期、失效、预算和诊断，再选择 VSM、RT、ReSTIR、probe 或 screen 方法。
3. **可退化但不静默**：质量降级必须有显式原因、计数、可视化和恢复条件。
4. **参考模型与生产路径分离**：CPU golden、简化 integrator 和 replay 用于验证，不伪装成可承担全场景实时渲染的 backend。
5. **性能是分布，不是平均数**：至少报告 P50/P95/P99，并报告最坏单像素灯复杂度、缓存失效、页缺失、历史拒绝和带宽。

### 1.3 物理输入与艺术控制

系统内部区分三种值：

- **物理输入**：带单位、线性色彩、可追溯来源的光强、辐亮度、发光功率、材质反射参数。
- **估计器输出**：带 measure、PDF、bias bound、variance/confidence 的积分估计。
- **艺术控制**：曝光、indirect scale、stylized bands、非物理增益、局部颜色修正。

物理验收在 scene-linear、曝光前完成；显示变换后的像素不能用于证明能量守恒。艺术控制必须标记为非物理操作。

## 2. 当前状态：准确诊断

### 2.1 已有基础

| 能力 | 当前位置 | v4 处理 |
|---|---|---|
| ECS 光源提取 | pkg/prism_render_scene/src/lighting/extract.rs | 保留为 adapter，改为发布 SceneDelta |
| 方向光、点光、聚光灯 | ExtractedLights 与 GPU ABI | 映射到统一 Light IR |
| 环境光 / 环境贴图 SH | extract_environment_probe、probe cache | 作为 environment provider 输入，不视为 punctual light |
| cluster 配置 | ClusterConfig：tile、depth slices、每 cluster 上限 | 保留 exact path，增加 overflow 诊断和 GPU build 路径 |
| 新宿主 LightingPlan | prism_render_app/src/lighting.rs、plugin.rs | 成为调度入口，不再维护第二套 light registry |
| direct/GI CPU contracts | prism_render_architecture/src/lighting | 收敛为 provider contracts |
| LightTree、area light、probe volume | prism_render_shading 与 architecture | 提取算法，统一 measure 和版本 |
| VSM CPU reference | lighting/shadow.rs、shadow/virtual_sm | 接入 ShadowWorkGraph；GPU page pool 另行验收 |
| software BVH / trace contract | ray_scene、lighting/trace.rs | 作为 software trace backend，不能冒充 hardware RT |
| 材质 execution proof | from_untextured_unlit_materials 首片 | 扩展为统一 RadianceQuery，保留显式拒绝 |
| OCIO / display transform | lighting/ocio.rs 与新宿主输出 | 保持在最终显示层，不与光照积分混用 |

### 2.2 主要缺口

1. **两个输入/生命周期中心**：Bevy 场景提取链和新宿主 lighting vertical slice 都能承载光照输入，但没有单一 LightScene owner。
2. **全量重建默认化**：scene extraction 和 light buffers 每帧清空并重建，不适合大规模稳定场景。
3. **数据与执行耦合不足**：光源记录、阴影策略、采样策略、材质辐射、GI cache invalidation 没有统一版本关系。
4. **算法与产品装配不等价**：CPU golden、shader、GPU parity、宿主帧图 receipt、终端输出 parity 必须分别验收。
5. **光源类型与阴影类型混淆**：矩形灯、网格发光体、环境光不应强行当成普通 VSM caster；Sky 和 Emissive 不应默认生成 shadow map。
6. **预算只看总灯数不够**：单像素重叠灯、投影面积、可见概率、阴影页更新、RT scene coverage 和历史稳定性才决定质量。

## 3. 总体架构与责任边界

### 3.1 四层权威模型

    Authoring / Import / Runtime / Network / ECS adapters
                             │
                             ▼
                     LightScene（唯一 CPU 权威）
                             │ SceneDelta + epochs
                             ▼
                    GpuScene / StableSlotRegistry
                             │ per-view work generation
                             ▼
          LightingWorkGraph = direct + shadow + GI + reflection + resolve
                             │ provider outputs
                             ▼
                 Scene-linear composite → display transform → present

各层职责：

- AuthoringAdapter：把编辑器、glTF/USD、ECS、脚本、网络状态转换为 Light IR；不得直接写 GPU buffer。
- LightScene：负责句柄、字段版本、单位归一化、启用状态、空间边界、影响范围、来源追踪和 delta journal。
- GpuScene：负责 stable slot、generation、packed payload、indirection、上传、设备代际恢复和 fence-safe recycling。
- LightingWorkGraph：按 view、camera、quality 和 budget 生成工作；不修改 LightScene，不创建第二张权威灯表。
- providers：实现 visibility、radiance、sampling、cache、denoise 等可替换算法。

### 3.2 建议 crate 责任

| crate | 唯一责任 |
|---|---|
| prism_render_architecture | 与后端无关的 Light IR、delta、预算、采样、失效和诊断契约 |
| prism_render_scene | Bevy/ECS adapter、view adapter、可选 scene extraction，不拥有第二套光源语义 |
| prism_render_app | RenderApp 生命周期、帧图装配、provider 选择、提交和发布 |
| prism_render_shading | BRDF、材质响应、辐射查询的 CPU reference 与共享语义 |
| prism_render_driver_* | 资源、队列、timestamp、RT/VSM 能力和后端限制 |
| prism_light_gpu | packed Light/GpuScene ABI、上传、compaction、view work buffer |
| prism_lighting_debug | debug view、explain、capture/replay、报告导出 |

禁止的反模式：

- prism_render_scene 自己维护一套 light registry，prism_render_app 再维护另一套；
- shader 通过隐式 binding 读取未登记的全局光源数组；
- provider 为方便直接复制 MaterialRecord 或灯光参数，形成第二个模型；
- 通过“未实现但计划选中”的 recipe 进入生产帧图。

### 3.3 标准帧时序

1. 主世界/导入器提交 authoring changes。
2. LightScene::apply_changes 验证单位、句柄、范围和能力，追加 SceneDelta。
3. 根据字段 epoch 计算受影响的 GPU slot、cluster、shadow page、GI cache cell 和 history key。
4. render-world 固化 immutable SceneSnapshot，provider 执行期间不读可变主世界。
5. GpuScene::prepare 应用 delta，完成 upload/compaction/recycle，记录 upload receipt。
6. 每个 view 生成 ViewLightingWork：screen tiles、depth bins、visible candidates、shadow requests、GI rays、history policy。
7. LightingWorkGraph 编译并执行 provider passes。
8. 每个 provider 写出带版本、measure、variance/bias/confidence 的 output token。
9. resolve 只消费兼容 token；不兼容贡献必须显式拒绝或走 declared fallback。
10. 记录 LightingFrameReport，发布诊断和终端输出。

## 4. Light IR：光源、单位与作者体验

### 4.1 光源分类

#### A. AnalyticEmitter

- Directional：无限远方向光，可带角半径/角直径。
- Point：球面点光近似，可选有限 source radius。
- Spot：点光 + 内外锥角 + barn door/光函数。

#### B. ExtendedEmitter

- Rect：有限矩形面光，可带纹理、双面策略和边缘/法线定义。
- Disk：有限圆盘面光。
- Sphere：球体面光或球形体积光近似。
- Capsule：胶囊面光，适合霓虹管和灯带。

#### C. EnvironmentEmitter

- Sky：程序天空、环境贴图、太阳/大气耦合的环境辐射。
- Portal：室内环境采样入口约束。
- VolumeEmitter：体积散射或雾中发光，走 volume provider。

#### D. MeshEmitter

从三角形、实例或材质 emissive 区域生成的发光体集合。必须有离线或运行时 importance hierarchy，不能把每个三角形直接变成独立普通灯。

#### E. BakedOrExternalSource

lightmap、probe cache、外部 capture、离线 path trace、摄影测量或项目自定义辐射场。它们是 radiance source，不强制具有实时 shadow caster 语义。

### 4.2 单位和色彩

所有输入记录都携带 UnitTag，或在 schema 层固定来源转换：

| 输入 | 推荐单位 | 备注 |
|---|---|---|
| 方向光 | illuminance，lux | 来源引擎转换必须有 source tag |
| 点/聚光灯 | luminous flux，lumen；或 radiant intensity，W/sr | 不把 candela 误称为 radiant intensity |
| 面光 | radiance，W/(sr·m²)；或 photometric authoring value | area × radiance 不是一般意义的总 flux；单面 Lambertian 需考虑 π 因子 |
| 发光材质 | scene-linear radiance/emission | 由材质求值，不能复制成独立灯后再重复加一次 |
| 环境贴图 | scene-linear radiance | 采样分布、mip、SH/filtered representation 需标记 |
| 颜色 | scene-linear RGB + primaries | 导入 sRGB/Rec.709/ACEScg 必须显式转换 |

光照内部不保存 display-encoded sRGB。曝光、白平衡、tone mapping、gamut mapping、transfer function 在 display transform 阶段处理。

### 4.3 LightRecord

建议逻辑结构如下，具体 GPU packing 不等同于 Rust repr(C)：

    LightRecord
    ├── identity: LightId, Generation, SourceId
    ├── kind: Analytic | Extended | Environment | Mesh | Baked
    ├── transform: position / orientation / scale / local frame
    ├── emission: color, unit, flux/intensity/radiance, temperature
    ├── shape: radius / size / capsule length / cone / barn door
    ├── influence: world bounds, range, falloff, channel mask
    ├── visibility: shadow policy, ray policy, casts / receives / volumetric
    ├── sampling: importance class, hierarchy node, texture distribution
    ├── mobility: static / stationary / dynamic / streaming
    ├── authoring: source asset, overrides, provenance, warnings
    └── revisions: field epochs and schema revision

mobility 只是更新策略提示，不自动决定所有算法。静态灯也可能需要相机相关 page；动态灯也可能只更新一个局部 slot。

### 4.4 来源转换

- glTF KHR_lights_punctual 只负责 punctual light 来源；不得把其单位直接套到所有光源。
- ECS adapter 以源引擎语义为准，保留 SourceUnitSystem，转换后写入 Prism canonical unit。
- USD/编辑器的 photometric values 记录 exposure、temperature、IES、texture 和 coordinate convention。
- 无法可靠转换的来源进入 ImportWarning，不静默猜测。
- source asset、import revision、conversion digest 写入 LightRecord。

## 5. 句柄、增量更新与 GPU 生命周期

### 5.1 Stable identity

生产句柄不再只用 u32 index：

    LightHandle {
        scene_id: SceneId,
        slot: u32,
        generation: u32,
    }

必要时再附加 source_id、record_revision、scene_epoch。删除流程为：

    Alive → Retiring → FencePending → Recycled

generation 在 recycle 时递增；旧 command buffer、shadow request、reservoir、debug selection 都必须因 generation 不匹配而失效。

### 5.2 SceneDelta

    SceneDelta {
        sequence: u64,
        frame: u64,
        scene_epoch: u64,
        creates: [CreateRecord],
        updates: [FieldUpdate],
        destroys: [LightHandle],
        hierarchy_updates: [SamplingHierarchyUpdate],
        invalidation: ChangeFields,
    }

字段更新至少区分 transform/bounds、emission、shape/sampling、shadow policy、material-emitter association、visibility/channel mask、schema/layout revision。禁止用一个 lighting_changed = true 让所有缓存全失效。

### 5.3 GPU packing

GPU ABI 由 schema 生成或由显式 offset/stride 常量定义：

- 不使用原生 u8/u16/bool 作为跨 WGSL/CPU 的隐式布局字段；
- 明确 storage/uniform alignment、stride、array length、endianness 和 version；
- 每个 payload 带 abi_version 与 payload_size；
- Rust repr(C) 只作为源代码布局，不作为 WGSL 布局证明；
- stable slot table 与 packed payload 分离，允许 compaction 而不改变逻辑 handle；
- 绑定资源有容量和 generation 校验，超过上限走可解释拒绝或分层上传。

推荐分层：

    LightHandle table  → slot → payload offset / alive / generation
    Light payload      → hot fields used by shading
    Light extension    → shape, IES, texture, hierarchy, debug metadata
    View indirection   → visible candidates / per-tile list / reservoir source

### 5.4 上传策略

优先级从高到低：

1. dirty range upload；
2. hot/cold split；
3. 在碎片或 generation 密度低于阈值时后台 compaction；
4. 首次创建、设备恢复、schema 变更、delta gap 时 full rebuild。

每次上传记录 bytes、ranges、staging copies、submission、device generation 和是否 full rebuild。

## 6. View Work Generation 与直接光

### 6.1 Exact clustered path

适合灯数量较少、单 cluster 复杂度低、需要稳定低噪声、移动/兼容档以及强主光。

    LightScene → frustum/depth cull → tile/cluster lists → exact BRDF resolve

要求：

- cluster overflow 不得静默截断；
- overflow 按 view/tile/light class 记录；
- 光源列表使用 stable handle 或可回溯 slot；
- 强主光和少数高影响灯可走显式保留列表。

### 6.2 Stochastic direct path

适合大量局部灯、面光、mesh emitter，或逐灯评估已经超过预算的场景。

    light hierarchy / guiding → proposal sample → visibility
                               → reservoir / estimate
                               → temporal/spatial reuse → denoise → resolve

固定样本预算只控制成本的一部分，不保证 error 上界。质量依赖单像素 lighting complexity、可见概率、proposal 分布和历史稳定性。

### 6.3 Light hierarchy

统一 hierarchy 节点支持 AABB/cone/solid-angle bounds、发光贡献 upper bound、静态/动态 revision、子节点 PMF、混合光源和 view importance。

CPU median-split LightTree 可作为 reference；生产构建需要增量更新、动态灯局部 refit、GPU traversal 或 candidate build，以及可解释的 upper bound，避免错误剪枝造成暗部缺光。

### 6.4 采样契约

每个 sample 必须能回答：

    sample.measure        = discrete | area | solid_angle | projected
    sample.source_pdf     = p(source)
    sample.target_pdf     = contribution density under target measure
    sample.selection_pmf  = discrete selection probability when applicable
    sample.delta_class    = none | point/directional delta | glossy/specular delta
    sample.visibility     = binary / filtered / confidence-weighted
    sample.bias_bound     = known bound or Unknown
    sample.variance       = estimator variance estimate

不同 measure 的转换在 sampler/provider 边界完成，不能在 resolve 阶段靠经验比例修正。Delta light 是否参与组合取决于 estimator、measure 和是否存在可比 proposal，不能用“永远排除 MIS”概括。

## 7. 阴影系统

### 7.1 Provider 选择

| 场景 | 首选 | fallback |
|---|---|---|
| 强方向光、大范围主光 | directional VSM clipmap 或高质量 RT | CSM / cached cascade / contact |
| 少量强点/聚光灯 | VSM page 或 RT | cube/spot shadow、screen contact |
| 大量小局部灯 | stochastic RT / shadow reservoir | unshadowed importance + contact |
| 面光 / mesh emitter | RT visibility / shadow reservoir | VSM approximation、baked occlusion |
| 环境光 / Sky | bent normal、AO、probe/world GI | ambient fallback |
| 静态场景 | baked/cache | runtime shadow |
| 体积光 | volume shadow provider | coarse shadow / unshadowed with warning |

### 7.2 Virtual Shadow Page

明确拆分四个成本：

1. resident pool memory；
2. page requests / page table traffic；
3. pages updated per frame；
4. caster draw/geometry cost。

    PageKey   { light_handle, clip_level, page_xy, mip, view_family }
    PageState { resident, last_used, caster_epoch, light_epoch,
                content_hash, fence }

失效来源包括光源 transform/shape/policy、caster bounds/skinning/deformation/WPO/alpha coverage、receiver/camera 新 page、geometry/material/layout epoch、streaming 和 device generation。

不要把 Virtual Shadow Maps 与 variance shadow maps 混称。page 过滤、边界、PCSS/SMRT 近似、depth layer、warm-up 和 miss fallback 都要单独建模。

### 7.3 软阴影

软阴影质量由 emitter shape、source radius/angle、shadow ray 数、page/depth 样本数、visibility provider 近似、temporal accumulation、denoise 和 penumbra clamp 共同决定。

允许逐级退化为硬阴影、少 ray、低 page mip、RT→VSM、VSM→contact；必须保留 fallback_reason。不能为了预算静默删除灯光或把所有灯变成无阴影。

## 8. Ray Scene、可见性与辐射查询

### 8.1 Ray Scene 分层

    Triangle Scene       高质量近场，支持材质命中求值
    Instance Scene       实例/蒙皮/变形后的更新表示
    Simplified Scene     software BVH / SDF / proxy，兼容档
    Far Field Scene      HLOD/merged proxy，大世界远距离
    Screen Scene         depth/normal/material ID，快速首选和补洞

screen trace、world trace、far-field 命中不能假装完全同义，特别是 offscreen、透明、不可见 caster、camera cut 和 art-directed light function 场景。

### 8.2 TraceQuery

    TraceQuery {
        kind: Shadow | Diffuse | Specular | Occlusion | Volume,
        origin / direction / t_max,
        view_id / scene_epoch / ray_generation,
        required_latency,
        geometry_policy,
        alpha_policy,
        material_policy,
    }

结果分离：

- hit/miss：几何结果；
- visibility：shadow/occlusion 可见性；
- surface_id / barycentrics / uv：命中点重建；
- radiance：由 RadianceQuery 求值，不由 trace backend 偷塞；
- bias_bound 与 variance：分别表达系统误差和统计误差。

### 8.3 RadianceQuery

Radiance provider 负责命中点材质、纹理、法线、粗糙度、发光、透明/alpha test，并生成 outgoing radiance 或 surface record，同时记录 material generation、revision、epoch、execution proof。

Trace backend 不复制材质参数；材质系统不复制 BVH ownership。双方通过 stable identity 和 snapshot epoch 连接。

当前受限的 Unlit/no-texture 材质首片应作为第一条真实闭环扩展，而不能被错误扩展为“完整 Lit/纹理命中求值已经完成”。

## 9. GI：多 provider、单积分语义

### 9.1 四层拆分

1. **Evidence provider**：screen trace、depth/normal、probe、world trace、baked field。
2. **Radiance provider**：命中点材质和发光求值。
3. **Estimator**：path sample、ReSTIR GI、probe interpolation、surface cache query。
4. **Reconstruction/resolve**：temporal、spatial、denoise、confidence、最终组合。

这样可以替换 hardware RT/software BVH，而不复制 GI estimator；也可以替换 probe cache，而不改变 trace ABI。

### 9.2 推荐实时路径

    screen evidence
          │ miss / low confidence
          ▼
    world trace / software BVH / hardware RT
          │
          ▼
    radiance query → candidate estimator / reservoir
          │
          ├── cache lookup / probe / surface cache
          ├── temporal reuse with validity checks
          ├── spatial reuse with normal/depth/material tests
          └── per-signal denoise and resolve

screen GI、probe 和 world trace 不是三份可以无条件相加的“光”。组合前必须声明是否估计同一目标积分、measure/PDF 是否兼容、是否存在 overlapping contribution、confidence 是统计量还是启发式权重、bias 是否可界定，以及是否需要 partition-of-unity 或 mutually exclusive routing。

### 9.3 ReSTIR GI 约束

Reservoir 至少记录：

    selected_sample
    W / M / candidate_count
    source_pdf / target_pdf
    target_measure
    visibility / radiance revision
    age / temporal validity
    spatial source id
    bias class / confidence

零贡献 miss 是否计入 N/M 必须由 estimator 规范固定；CPU 和 WGSL 不得各自推断。重用必须检查 scene epoch、geometry/material/light/environment epochs、view motion、normal/depth、surface identity 和 candidate validity。

### 9.4 Probe / surface cache

必须定义 cell 的 bounds、resolution、validity、age、动态覆盖、方向表示、leak control、visibility confidence、update budget、warm-up、eviction 和 device recovery。任何 cache miss 都要有显式 fallback，不能用未初始化历史值静默填补。

## 10. 反射、透明、体积与色彩

### 10.1 镜面信号独立建模

漫反射 GI 和镜面反射的可见性、历史有效性、BRDF 权重和 denoise 特性不同。v4 不把镜面压成 GI 的一个后处理项。

候选路径：

- SpecularScreenSpace：低成本首选，明确 offscreen 缺失；
- ProbeSplitSum：粗糙镜面的环境近似；
- PlanarReflection：平面水面/镜面专用；
- RaySpecular：硬件/软件 trace；
- BakedReflection：静态或离线内容。

每个信号拥有独立 history key、rejection policy、roughness range 和 fallback。

### 10.2 透明、毛发、水体、体积

这些不是普通 opaque 的自动特例。契约应提前确定：

- transparent 是否进入 shadow/ray scene；
- alpha masked 是否需要材质求值还是 coverage proxy；
- hair 使用 strand、cards 还是 proxy；
- water 的 planar/refraction/volume/caustics 哪些属于本系统；
- volume lighting 是否共享 punctual light sampling；
- 不同 primitive 的 temporal history 如何避免互相污染。

v4 只定义接口和诊断门禁，不把它们全部纳入第一阶段完成目标。

### 10.3 Temporal 与色彩

可共享 motion vectors、history allocation、resource lifetime、exposure compensation 和 rejection counters；direct stochastic、diffuse GI、specular、shadow penumbra、volumetric 的 rejection 策略必须独立。

    HistoryHeader {
        signal_kind,
        view_id,
        scene_epoch,
        signal_revision,
        exposure_revision,
        resolution / jitter,
        camera_cut / teleport,
        previous_frame_token,
    }

scene-linear buffer 保持物理/艺术混合前的明确语义；pre-exposure 若使用，必须进入历史和 reservoir compensation；tone mapping 后的结果不得回灌 scene-linear GI。

## 11. 质量档位、能力与预算

### 11.1 四层配置

    CapabilityProfile       设备/后端真实支持什么
    QualityIntent            用户/项目要什么质量
    ResourceBudget           本帧允许多少 GPU/CPU/显存/带宽/IO
    ResolvedQualityState     本 view 本帧最终选了什么

ResolvedQualityState 必须可序列化、可解释、可 replay；不能让 shader 用一堆 feature flag 自己拼出与 CPU 计划不同的结果。

### 11.2 推荐档位

| 档位 | 直接光 | 阴影 | GI/反射 | 适用 |
|---|---|---|---|---|
| Compatibility | exact clustered | CSM/spot/cube/contact | AO + probe/低成本 screen | 无 RT 或低端 GPU |
| Balanced | clustered + limited stochastic | cached VSM/RT mixed | screen + probe + limited world trace | 主流硬件 |
| High | stochastic/ReSTIR direct | VSM + RT provider | screen → world trace + cache | 高端 GPU |
| Ultra | high budget reservoir | high-res VSM/RT | high-quality reconstruction | 高端桌面 |
| Reference | exact/reference integrator | geometry visibility | path/reference | 验证与离线 |

### 11.3 预算维度

预算必须包括 CPU prepare/plan/upload、GPU critical path、dispatch/draw/ray/sample/reservoir 数量、light/history/page/resident/double/in-flight memory、streaming IO、多视图倍增、skinning、alpha test、far field update。

GPU 预算是估计和验收结果，不是编译期保证。异步 compute 只有真实 backend、队列和依赖允许时才能计入 overlap；不能简单加减异步 pass timestamp 当作总帧时间。

### 11.4 4096 光源验收

必须固定分辨率、内部渲染比例、材质复杂度、视图数量、光源类型和空间分布、每像素影响灯数 P50/P95/P99、shadow casting 比例、硬件/backend，并分别报告 direct-only、shadow-only、full lighting 的 P50/P95/P99 GPU/CPU、显存、带宽、噪声/误差和历史拒绝。

## 12. 易用性、诊断与维护

### 12.1 Debug views

至少提供：

- Light Scene：handle/generation、单位、范围、来源；
- Light Complexity：每 tile/pixel 的影响灯、importance、shadow/RT cost；
- cluster occupancy/overflow；
- shadow page requested/resident/updated/invalidated；
- Ray Scene coverage、BVH primitive、missing geometry、alpha policy；
- GI cache coverage、age、confidence、leak risk；
- reservoir selected sample、M/W、PDF、reuse source、rejection reason；
- temporal rejection reason；
- energy/error heatmap；
- fallback chain 和本帧最终质量档位。

### 12.2 Per-pixel explain

选中一个像素时，应能导出：

    pixel → material/surface id
          → direct light candidates and weights
          → shadow provider and visibility result
          → GI candidate/reservoir lineage
          → cache/probe/world trace evidence
          → temporal history decision
          → final resolve and display transform

### 12.3 Capture / replay

Capture 至少包含 LightScene snapshot、delta journal checkpoint、GpuScene ABI、device capability/limits、view/camera/jitter/exposure/display config、quality/budget/resolved state、scene/material/ray revisions、random state、provider output metadata、device generation。

Replay 分为：

1. CPU oracle replay：追求可重复和明确容差；
2. same-device GPU replay：追求 bitwise 或固定容差；
3. cross-device replay：追求统计/图像误差，不宣称浮点 bitwise 一致。

### 12.4 Provider registry

算法通过 capability-aware registry 注册：

    ProviderDescriptor {
        id / version,
        signal_kind,
        required_caps,
        supported_light_kinds,
        supported_material_classes,
        measure_contract,
        latency_class,
        memory_model,
        fallback_chain,
        debug_views,
        execution_backend,
    }

只有拥有真实 execution backend、receipt 和 parity 的 provider 才能进入生产计划。只有 CPU reference 的模块只能用于验证。

### 12.5 Schema 与生命周期

- Light IR、GPU ABI、capture、debug report 都有独立 schema version；
- 破坏布局时升级 ABI；
- provider version 与 algorithm revision 进入 frame report；
- replay 遇到不兼容版本应失败并说明原因；
- 每种资源声明 owner、producer、consumer、retirement fence、recovery policy；
- device loss 后从 SceneSnapshot + asset registry + journal checkpoint 重建；
- cache 失效与 resource destruction 分离；
- debug/capture 不得改变 production scheduling，除非显式启用 instrumentation。

## 13. 迁移方案

### 13.1 现有模块处理分类

| 现有模块 | 分类 | 处理 |
|---|---|---|
| prism_render_scene ECS extraction | 接口适配 | 保留 Bevy 语义转换，输出 SceneDelta，删除直接拥有 GPU light table 的权威地位 |
| ExtractedLights / LightGpuBuffers | 重写 | 改为 LightSceneSnapshot + stable slot upload；旧结构仅留一阶段 adapter |
| prism_render_app::LightingSceneInputs | 接口适配 | 改为 SceneSnapshot/view input 消费入口，不再作为第二灯表 |
| LightingPlan / BudgetCompiler | 直接复用并扩展 | 增加 provider registry、view work、真实 backend receipt |
| LightHandle(pub u32) | 破坏性替换 | generation-safe handle，提供显式 migration function |
| culling.rs | 算法提取 | 保留 CPU oracle，增加 GPU candidate build 和 overflow contract |
| stochastic.rs / restir_di / restir_gi | 算法提取 | 统一 measure/PDF/reservoir schema，不复制权重规则 |
| gi/light/light_tree.rs | 算法提取 | 作为 hierarchy oracle，生产版增加动态 refit/上传 |
| probe_volume / surface_cache | 接口适配 | 添加 coverage、epoch、confidence、eviction、warm-up |
| shadow/virtual_sm | 算法提取 + GPU 接线 | reference 保留，接入真实 page pool 后才算生产 VSM |
| reference_pt | 直接复用 | Reference mode 和 golden source，不作为实时 H4 |
| lighting_trace_gpu | 接口适配 | 统一 TraceQuery、RayScene revision、RadianceQuery |
| OCIO/display transform | 直接复用 | 保持与 lighting integral 的阶段边界 |

### 13.2 里程碑

#### M0：冻结契约

- 定稿 Light IR、单位、handle、SceneDelta、ABI schema；
- 建立 provider registry 和 ResolvedQualityState；
- 旧路径增加 receipt/diagnostic，不再新增平行契约。

#### M1：唯一 LightScene

- ECS、host、import 三类 adapter 统一发布 delta；
- stable slot、generation、fence-safe recycling；
- exact clustered path 先迁移，保证功能回归。

#### M2：GpuScene 与增量上传

- dirty range、hot/cold split、buffer indirection；
- full rebuild 作为 recovery/reference fallback；
- 建立 upload bytes、range count、stale handle 和 device replay 测试。

#### M3：阴影纵切

- directional/spot VSM page request、cache、invalidation；
- point/spot/area provider 路由；
- page debug、fallback chain、动态 caster 验收。

#### M4：direct stochastic / ReSTIR DI

- hierarchy、proposal、visibility、reservoir、temporal/spatial reuse；
- 与 exact path 在小场景逐像素对拍；
- large light complexity benchmark。

#### M5：GI world trace + RadianceQuery

- 先完成 Unlit/no-texture 限定闭环；
- 再扩展 opaque textured、normal map、alpha test、skinning；
- 每一步都有 material capability 和拒绝原因。

#### M6：probe/cache/reflection

- cache coverage、leak control、history；
- 镜面独立信号；
- screen → world → hardware/reference fallback。

#### M7：产品化和多视图

- capture/replay、debug UI、multi-view、large world、streaming、device loss；
- 内容工具和自动 benchmark；
- 只在同硬件、同分辨率、同质量目标下对标。

## 14. 验收矩阵

### 14.1 功能闭环

| 编号 | 验收项 | 必须证明 |
|---|---|---|
| F1 | ECS/import/runtime 三适配器 | 都发布同一 LightScene，不能出现第二权威表 |
| F2 | create/update/destroy | handle generation、回收、旧命令拒绝正确 |
| F3 | exact clustered | direct BRDF、cluster overflow、空场景、强主光正确 |
| F4 | incremental upload | 不变场景无全量上传；delta gap 可恢复 |
| F5 | VSM/cache | page request、resident、invalidate、evict、warm-up 有真实 receipt |
| F6 | software trace | hit/miss、revision、deadline、device loss、partial result 行为明确 |
| F7 | hardware RT adapter | 能力检测和 fallback 正确，software 不标成 hardware |
| F8 | RadianceQuery | material proof、纹理/alpha/normal capability 显式有效 |
| F9 | GI estimator | PDF/measure/M/W/zero proposal 规则 CPU/GPU 一致 |
| F10 | temporal | camera cut、epoch、exposure、resize、device recovery 正确拒绝历史 |
| F11 | reflection | roughness、screen miss、planar、world trace 路由正确 |
| F12 | output | scene-linear → display transform → present/offscreen 语义一致 |

### 14.2 性能和内存

每个测试场景至少记录 CPU extract/plan/upload/prepare、GPU 每 pass 与关键路径的 P50/P95/P99；lights、clusters、shadow pages、rays、reservoirs；upload bytes、resident/history/page-table memory；GPU wait、queue overlap、pipeline stall；history reject、cache miss、page invalidation、cluster overflow；最终 image error、variance、temporal instability。

### 14.3 画质和能量

reference 场景至少包括单强方向光、密集室内灯、多面积光源 + glossy/rough surfaces、emissive mesh + indirect bounce、室内门户、大世界远景、动态 caster、skinned mesh、alpha masked foliage、camera cut、曝光变化、resize，以及 water/hair/volume/transparent 的边界场景。

指标包括 scene-linear luminance relative error、flux/irradiance/radiance consistency、shadow edge/penumbra error、temporal flicker/ghosting、GI leak、missing/over-bright contribution。PSNR/SSIM/FLIP 只能辅助，不能替代视觉检查。

### 14.4 退化单调性

不承诺“降低质量永远不会变亮”这类普适物理性质。应验收：

- 资源消耗单调下降或有明确例外；
- fallback 不产生未初始化值；
- provider 被拒绝时有可见诊断；
- 升档不会读取更旧结果；
- 参考误差、噪声、稳定性在已定义场景中不超过门槛。

## 15. 待拍板决策与成功标准

### 15.1 待拍板

1. 硬件 RT 是否为 High/Ultra 的必需能力，还是始终保留 software high fallback。
2. LightScene 是独立新 crate，还是先放入 prism_render_architecture::lighting 再拆分。
3. VSM page pool 按 view、light family 还是全局共享。
4. Mesh emitter hierarchy 由 Prism asset pipeline 构建，还是先接受运行时 proxy。
5. 是否允许项目 profile 自定义非物理 lighting multiplier，以及如何在报告中隔离。
6. 透明、毛发、水体、体积是否进入第一版 RadianceQuery。
7. high-end backend 是否需要绕过 wgpu 接入 native Vulkan/Metal/DX12。
8. 是否建立独立 reference render executable 用于固定场景、离线 capture 和 image regression。

### 15.2 成功标准

v4 不是以“新增多少光照算法”作为完成标准，而是满足：

1. 任意光源来源最终都进入同一个 LightScene，并能追溯来源和单位。
2. 任意 GPU 光照 pass 都能通过 handle、epoch、provider、view 和 budget 解释输入。
3. exact、stochastic、software trace、hardware trace 之间有明确 capability/fallback 关系。
4. 光源、阴影页、GI cache、reservoir、材质辐射和 temporal history 的失效规则一致。
5. 质量退化不会静默删灯、吞掉贡献或读取过期资源。
6. 性能结论能复现到硬件、分辨率、内容分布、P50/P95/P99 和 capture。
7. reference、CPU oracle、GPU parity、宿主帧图、终端输出和图像回归分层验收。
8. 新增光源、provider 或材质能力时，不需要复制第二套场景表、采样规则和诊断系统。

## 附录 A：与 v3.1 的关键差异

| v3.1 方向 | v4 调整原因 |
|---|---|
| 八类光源统一进入同一 shadow 逻辑 | 光源形状、环境、mesh emitter 的可见性和采样语义不同 |
| WebGPU-first 作为高端主线 | 可移植性与高端质量上限不同；software fallback 与 hardware RT 必须能力分层 |
| 固定 H1–H4 recipe | recipe 不是能力，需要 capability、quality、budget、resolved state 四层 |
| 32B/60B 等 ABI 目标直接写入方案 | 未经 schema/offset 验证的字节数容易与真实字段、WGSL layout 冲突 |
| screen GI + probes + world trace 混合 | 必须先声明是否同一积分、measure、PDF、bias 和 coverage |
| 所有大灯都用 VSM/soft-shadow | VSM、RT、contact、baked、unshadowed 是 provider 选择 |
| 4096 lights ≤25% 成本 | 必须固定光源分布、每像素复杂度、阴影比例、硬件并测画质 |
| 关闭随机性作为 reference | deterministic sampling 不等同 unbiased/reference |
| 计划选中即视为已实现 | 只有真实 execution backend、receipt 和 parity 才能进入生产计划 |
| 全量灯表每帧上传 | 改为 stable slot + dirty range + fence-safe recycle |

## 附录 B：首条垂直切片

第一条可交付切片不追求覆盖所有光源，而追求闭环完整：

    ECS Directional/Point/Spot
        → LightScene adapter
        → stable handle + delta
        → GpuScene dirty upload
        → exact clustered direct
        → one shadow provider with explicit fallback
        → scene-linear resolve
        → display transform
        → capture + per-pixel explain + CPU oracle

通过后再加入 Rect/mesh emitter、stochastic direct、world trace、GI reservoir、probe/cache 和高质量反射。每一步都必须能回答：输入是否唯一、数据是否新鲜、算法是否正确、GPU 是否真正执行、输出是否可见、性能是否可量化。

## 附录 C：仓库阅读入口

- 现有光照架构基线：[prism_lighting_gi_architecture_zh.md](prism_lighting_gi_architecture_zh.md)
- 渲染系统整体差距审视：[prism_render_vs_ue5_gap_zh.md](prism_render_vs_ue5_gap_zh.md)
- 帧图重构：[refactoring-dev/04-frame-graph_refactor_zh.md](refactoring-dev/04-frame-graph_refactor_zh.md)
- Render Fabric 重构：[refactoring-dev/05-render-fabric_refactor_zh.md](refactoring-dev/05-render-fabric_refactor_zh.md)
- 光照契约：[pkg/prism_render_architecture/src/lighting/contract.rs](../pkg/prism_render_architecture/src/lighting/contract.rs)
- ECS 光源提取：[pkg/prism_render_scene/src/lighting/extract.rs](../pkg/prism_render_scene/src/lighting/extract.rs)
- 新宿主 lighting：[pkg/prism_render_app/src/lighting.rs](../pkg/prism_render_app/src/lighting.rs)

