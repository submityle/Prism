# Prism 次时代 AAA 光照 / GI 系统设计 v20

> 日期：2026-10-21。状态：生产架构与实施验收合同，非完成声明。
>
> 范围：Light、Direct、Shadow、GI、Reflection、Temporal、HDR、预算、工具与恢复。
>
> 技术方向：原生 Prism；不以 Bevy 作为生产依赖。允许破坏性更新。
>
> 前版：[v19](prism_lighting_gi_design_v19_aaa_zh.md) 保留为评审基线；冲突以本版为准。
>
> 本次交付：设计文档；没有因此获得新的运行、GPU readback 或性能达标证据。

阅读导航：[关键升级](#1-相对-v31--v19-的关键升级) · [Light 系统](#6-light-一级系统与-canonical-合同) · [逐像素材质](#7-viewsurface-与逐像素材质) · [性能控制](#17-planner自动质量与性能预算) · [验收](#21-验收体系与需求追踪) · [实施迁移](#22-破坏性实施与删除门)

## 0. 执行决策与完成定义

Prism 需要的不是孤立的 LightPlugin，而是从作者意图到显示输出的完整光照系统。Light 必须成为一级系统，但不能吞并材质、几何、相机、资源管理和显示系统。

```text
Author / Import / Runtime edits
  -> LightScene + MaterialScene + GeometryScene + ViewScene
  -> Freeze snapshot + Identity maps + GPU Scene residency
  -> Immutable RadianceFramePlan
  -> Surface encode / Depth / Motion
  -> Light candidates + Visibility + Cache updates
  -> Direct / Indirect diffuse / Indirect specular
  -> Signal-specific reconstruction
  -> Scene-linear composition + Transparent / Volume
  -> Exposure + Temporal upscale + Display transform
  -> Present / Capture / Readback / Evidence
```

本版固定以下不可绕过的决策：

1. 唯一灯语义模型沿用并完善 `LightRecord / LightScene / LightSceneHandle`，不另造第二份 CanonicalLight。
2. 默认不透明主路径采用 deferred surface encode；opaque raster 不评价灯、不输出最终 lit color。
3. 主路径必须逐像素消费同帧 GBuffer。有限 per-surface 样本表只能做 oracle、诊断或明确的近似缓存。
4. Planner 负责策略、预算和不可变计划；provider 负责其缓存、页、reservoir 与执行，禁止中央 God Object。
5. 多种 GI / reflection 表示通过明确的路径归属或成立的 estimator 合成。confidence 不是避免重复能量的数学证明。
6. 单一帧事务覆盖 scene revision、ABI、device、target、view、history 和资源 generation；不允许混代提交。
7. 高级算法是可替换 provider，不是通用正确性前提；缺失能力必须可诊断，不用假 ambient 掩盖。
8. “AAA”是质量、稳定性、规模、工作流和证据的联合目标，不是算法名称或灯数量标签。

完成必须同时满足：真实场景输入 → 真实相机 → opaque 光栅 → 逐像素材质 → 光照/遮挡 → HDR 像素 → GPU 回读；以及性能、恢复、资源、工具、回归和旧路径删除门。shader 编译、命令提交、模块存在均不足以表示完成。

## 1. 相对 v3.1 / v19 的关键升级

| 主题 | 旧方案风险 | v20 的约束 |
| --- | --- | --- |
| Light 系统 | 有灯型与表，缺公共生命周期 | 初始化、发布、冻结、提取、提交确认、删除、恢复全闭环 |
| 权威模型 | 新术语可能再造 schema | 逐字段映射到现有 LightRecord；派生 GPU 表无写回权 |
| 表面数据 | 五路 GBuffer 缺 roughness / metallic / motion | 明确四 MRT 基线、独立 motion 与逐像素材质闭包 |
| 光照职责 | forward_lit 命名与纯 GBuffer 职责混杂 | SurfaceEncode 和 LightingEvaluate 拆开；透明独立 |
| 合成 | confidence 混合掩盖重复估计 | 路径域分解、互斥 fallback 或已证明 MIS |
| 时域 | 每个 provider 自己猜 motion / exposure | 统一 TemporalInputContract，独立 signal history |
| 性能 | 4096 灯、4.3 ms 被当普遍承诺 | workload envelope、分位数、质量误差与硬件共同约束 |
| 生命周期 | resize / recovery 各自重建 | generation 事务、提交依赖、延迟退休、多 target 隔离 |
| 易用性 | 作者面对大量底层开关 | Intent + Profile + Volume + Explain + 可复现 capture |
| 可维护性 | 功能清单代替完成标准 | Requirement → Contract → Test → Artifact → Exit Gate |

取消下列错误承诺：固定 seed 必然跨设备逐位一致；任何降档都使每像素亮度不增；delta light 永远不能涉及 MIS；任意 provider 混合天然不重复能量；固定页数、brick 尺寸或人周适用于所有设备与内容。

## 2. 参考策略：吸收机制，不宣称竞品等价

| 参考方向 | Prism 采用的设计模式 | 不继承的隐含前提 |
| --- | --- | --- |
| Epic Lumen 技术资料 | 多场景表示、屏幕路径与世界回退、缓存更新预算 | 不复制其私有表示，不依赖 Nanite，不宣称等价效果 |
| Epic Virtual Shadow Maps | receiver demand、页驻留、失效追踪与缓存诊断 | 不假设所有灯形、平台和变形内容同成本 |
| Epic MegaLights | 预算化候选采样、专用降噪、重叠复杂度管理 | 固定采样成本不等于质量不随灯重叠恶化 |
| Unity APV | 烘焙漫反射数据、空间细分、流式 cell、scenario | 不把烘焙 probe 等同全动态 GI 或镜面真值 |
| NVIDIA RTXDI / ReSTIR | 采样桥接、时空重用、current/previous identity | SDK 不拥有 Prism 场景、BRDF、可见性和显示 |
| AMD GPUOpen | 按 signal 降噪、分类、variance 与重投影 | 不用一个万能 history / denoiser 处理所有信号 |
| TSR / FSR 类重建契约 | 上游 motion、reactive、曝光、DRS 共同设计 | 不把 upscaler 当末端任意插入的滤镜 |
| ACES / 平台 HDR 文档 | rendering transform 与 display encoding 分离 | 不以 HDR bool 替代显示能力、色域和参考白 |

上述是架构取舍，不是复刻承诺；官方入口见附录 A。产品版本、SDK license、平台支持和具体参数必须在 provider 准入时锁定来源版本。本版不引用无法核实的“最新版本”功能或竞品耗时作为证据。

## 3. 范围、能力等级与质量档

### 3.1 交付边界

完整系统范围包含八类 canonical light 的语义、direct、conventional / virtual / trace shadow、动态与烘焙 GI、reflection、temporal、HDR、capture 和诊断。每种灯的生产支持必须分别证明发光评价、采样、遮挡与导入；枚举存在不算支持。

阶段只规定集成顺序，不把最终目标缩成首个 demo：

| 层级 | 实施能力 | 必须交付的证据 |
| --- | --- | --- |
| P0 正确性 | 真实 opaque；directional / point / spot；GGX；conventional shadow；显式环境；SDR/HDR 管线合同 | 像素 oracle、回读、相机、材质、resize / recovery |
| P1 完整性 | area / sky / emissive 采样、screen / probe / world GI、hybrid reflection、motion / denoise、透明与体积照明接口 | 各 signal 金样、失效、streaming、跨帧稳定性 |
| P2 规模 | many-light、VSM、radiance / surface cache、增量 bake、自动预算、设备矩阵 | 压力测试、P95/P99、VRAM、质量误差、故障注入 |
| Reference | 共用场景/材质的高采样参考、AOV、checkpoint | 收敛与误差基准，不要求实时 |

质量档不是功能枚举：`Portable / Balanced / High / Reference` 只是 profile 名称；每档绑定内部/输出分辨率、设备 capability、内存、采样、缓存更新、误差容忍和必选 fallback。产品必须公布经过验证的 workload envelope。

### 3.2 特殊内容不被静默忽略

植被/WPO、蒙皮、双面薄片、头发、水、粒子、透明、体积、大世界和 capture 各自声明 surface、motion、shadow、GI representation 能力。尚不支持时 importer / validator 返回结构化诊断，不回退成“已支持的普通 opaque”。

玩法、网络、物理策略和工程管理不进入 lighting core。资产系统提供 cook/stream 服务；照明模块只规定所需数据与接口。

## 4. 当前仓库快照与差距

本节是 2026-10-21 工作树观察，不是提交中全部代码的完成证明。工作树存在其他并行修改；设计工作没有重新运行构建或 GPU 验证。检查时 HEAD 为 `458138d79f52b6a35451a4a3da5f24d54acf7be3`，未提交文件不由此 SHA 固定。

| 路径 / 入口 | 观察 | 实施缺口 |
| --- | --- | --- |
| `pkg/prism_render_architecture/src/lighting/scene.rs` | 已有 canonical record、lineage / slot / generation / identity、字段变更模型 | 统一所有生产消费者并删除 index-only 语义歧义 |
| `pkg/prism_render_app/src/lighting_scene.rs` | canonical 向原生灯表派生并可拒绝不支持输入 | 与旧 scene GPU 派生路径确定唯一 owner |
| `pkg/prism_render_app/src/forward_lit.rs` | 现有工作树文件具备 MRT/frame/material 相关实现 | 纯 surface encode 职责、ABI 和真实绑定闭环仍需证明 |
| `pkg/prism_render_app/src/queue_stage.rs` | queue 入口仍传入空 bind groups | 连接真实 view、material、动态 offset 与 provider layout |
| `pkg/prism_render_app/src/views.rs` | RenderView 不包含完整 current/previous 矩阵 | 从 native camera 提取完整 View ABI，禁止 identity 兜底 |
| `pkg/prism_render_app/src/target.rs` | 检查时为 position / normal / sRGB albedo / surface-id 加 depth；与更早的五平面描述不同 | 切换紧凑逐像素材质合同与 motion，检查 MRT 能力 |
| `pkg/prism_render_app/src/lighting_gpu.rs` | 已有 target cache 与上传确认相关路径 | 不能由此推断 direct/GI 已逐像素且输出正确 |
| `pkg/prism_render_app/src/plugin.rs` | 已注册 lighting runtime / queue 等，也有默认 surface 输入 | 删除生产路径的假输入，接入真实 surface producer |
| `pkg/prism_render_app/src/recovery.rs` | 已重建一部分 target/pipeline/material 资源 | 纳入 lighting、history、缓存与 binding generation 事务 |

源目录中的 ReSTIR、DDGI、SSGI、SSR、VSM、TAA、exposure 等模块，只证明候选实现存在，不证明默认路径集成、原生依赖闭环或最终像素正确。

## 5. 三平面与所有权

### 5.1 平面划分

- Semantic：LightScene、MaterialScene、GeometryScene、ViewScene、ProbeBrickScene、DisplayIntent。保存语义与版本，不持有 GPU handle。
- Planning：RadiancePlanner、CapabilityProfile、QualityController、资源 reservation。生成 immutable plan，不私有化 provider 缓存。
- Execution：GPU Scene、frame graph、surface/lighting/shadow/GI/reflection provider、denoiser、output。执行已批准计划，返回结果与 telemetry。

### 5.2 唯一所有者

| 数据 | 权威 owner | 派生/消费方 | 不允许 |
| --- | --- | --- | --- |
| 灯语义与 revision | LightScene | GPU light producer、sampler、bake | provider 私自改 canonical 强度 |
| 材质闭包 | MaterialScene / Material IR | raster、trace、reference | 三套互不一致 BRDF |
| 实例/几何身份 | GPU Scene 对应的语义域 | raster、BVH、shadow、cache | 局部 slot 当永久实体 |
| 相机与 motion convention | ViewScene / TemporalInputContract | 所有 per-view pass | 每个 provider 自设坐标约定 |
| 策略与预算 | Planner / QualityController | provider | provider 自行升档、无限申请 |
| 页/brick/reservoir 内容 | 相应 provider | 其 pass 与诊断 | Planner 直接修改内部结构 |
| 生命周期/提交退休 | ResourceCoordinator / driver | 所有执行资源 | 单个 provider 私自销毁在途资源 |
| 显示输出 | OutputCoordinator | exposure / display provider | 材质 shader 自做 tone mapping |

这些是逻辑角色，不要求立即拆成同名 crate。先建立可测试接口，再按编译时间、依赖和团队 ownership 拆包。

## 6. Light 一级系统与 canonical 合同

### 6.1 公共入口

统一提供注册系统、创建灯、原子批量编辑、删除、只读查询、capture/replay、validation/explain 的入口。作者 API 可称 `LightIntent`，但它只是构造/编辑命令；持久化权威仍为 `LightRecord`。

生命周期：

```text
Uninitialized -> Registered -> Editable
Editable -- validate + publish --> Published(revision)
Published -- frame freeze --> FrozenSnapshot(frame)
FrozenSnapshot -> Derived -> Uploaded -> Submitted -> Acknowledged
Editable -- remove --> Tombstone -> GPU retirement -> Reclaim
DeviceLoss -> Rehydrate from semantic snapshot -> New generation
```

CPU 编辑可与旧帧执行并行；冻结后的修改进入下一 publication。删除立即阻止新引用，但 slot 重用必须满足 generation/identity 校验；不能因 GPU 在途而阻塞作者线程。

入口错误必须区分：invalid input、unsupported capability、asset pending、capacity pressure、device loss。非有限数、负辐射、非法角度/尺寸和单位冲突不得静默 clamp 成看似正确的灯。

### 6.2 沿用现有字段，不重复建模

| 语义 | 现有模型 | v20 决策 |
| --- | --- | --- |
| 稳定身份 | LightSceneHandle：lineage u64、slot u32、generation u32、identity u64 | 全部保留；identity 表示 incarnation，lineage 防跨场景误用 |
| 类别 | Directional / Point / Spot / Rect / Disk / Capsule / Sky / Emissive | 保留八类；支持矩阵按 provider 声明 |
| 移动性 | Static / Stationary / Dynamic | 保留语义，但修改是否有效由字段 dirty mask 决定 |
| 发光 | LightEmission：linear RGB tint、value、unit、temperature provenance | 保留；温度已应用不能导入时再乘一次 |
| 几何 | LightGeometry | 米；cone half-angle 为弧度；方向沿传播方向 |
| 单位 | Lux / Candela / Lumen / Radiance / Nits / Unitless | 保留显式单位；Unitless 限调试或声明式 artistic 模式 |
| 修改 | LightFieldMask / delta / checkpoint | 作为增量 journal；溢出退全量快照，不丢更新 |
| 算法索引 | 旧 index-only LightHandle | 迁为明确的局部索引类型，不能承担公共身份 |

RGB 不能唯一决定光度量与辐射量换算；转换需要光谱/光效模型或显式约定。模型、版本及源单位必须进入 provenance。lux 适合照度语义，nit 适合亮度语义；面积灯 lumen → radiance 必须计入面积、单双面和角分布。

### 6.3 身份与跨帧映射

GPU 紧凑索引只在 `LightTableEpoch` 内有效。生成 current→previous 与 previous→current 映射；校验 lineage、identity、generation 与相关字段 revision。不存在对应项时显式 invalid，不复用旧 reservoir。

同样规则覆盖 emitter triangle、instance、surface、material producer。拓扑变化不能只靠实例 ID 保历史。空间 cache 的失效按影响域传播，不必因为任意一盏灯变化清空全世界。

### 6.4 Mobility 语义

- Static：允许 bake/cache，但编辑仍必须触发失效；不得认为名字叫 Static 就永远不更新。
- Stationary：可缓存不变部分；移动、旋转或几何变化使相关 shadow/bake 内容失效。
- Dynamic：按字段生成更新；仅改强度不应重建不依赖强度的几何 shadow page。

灯通道、light linking、艺术限制是可选 profile 扩展，不能硬编码游戏玩法通道。

## 7. View、Surface 与逐像素材质

### 7.1 View ABI

View identity 至少包含 target identity/generation、view identity/generation、frame index、cut epoch、projection revision。主 camera、shadow、reflection、capture view 区分 kind，不能共享匿名全局 history。

必须提取 current/previous clip-from-world、world-from-clip、view-from-world、world-from-view、非 jittered projection、camera position、viewport origin/extent、内部/输出分辨率、jitter、depth convention、near/far policy、exposure/pre-exposure 与有效位。

从 `PrismCamera` 和 native transform 提取完整矩阵。仅有 clip matrix 或 world position 不能恢复全部相机状态；不可用 identity camera 作为生产默认。缺相机时返回明确的 no-view，清屏或跳过该 view 并报告原因。

### 7.2 默认 SurfaceProfile：四 MRT，非调试五平面

默认 opaque PBR profile 选择如下；格式还需通过 driver format/attachment capabilities 验证：

| 资源 | 格式 | bytes / pixel | 合同 |
| --- | --- | --- | --- |
| GBase | RGBA8Unorm | 4 | linear baseColor RGB，material AO A；非 sRGB attachment |
| GShading | RGBA16Float | 8 | oct normal RG、perceptual roughness B、metallic A |
| GEmission | RGBA16Float | 8 | scene-linear emission RGB；A 保留并清零 |
| GSurface | R32Uint | 4 | 本帧 surface-table index；0=无 surface |
| Depth | Depth32Float | 4 | reverse-Z，clear=0；depth 和 surface validity 联合判定 |
| Motion | RG16Float | 4 | 单独 motion pass，约定见 §15 |

四个颜色 attachment 合计 24 B/px；加 depth、motion 的单份逻辑存储为 32 B/px。附件 byte limit 的实际计算、format feature、混合整数/浮点 MRT 都必须由后端验证，不能只数附件个数。

不在生产 GBuffer 保存 RGBA32F world position：使用 depth + world-from-clip 重建；大世界采用 frame-relative origin / split origin，禁止把全世界坐标直接塞入低精度 half。

粗糙度、金属度、法线贴图、发光必须在表面点完成实际纹理求值后写入。单个 material ID 无法替代带 UV/derivative 的逐像素求值。AO 只用于声明的近似间接遮蔽，不乘到所有 direct，也不重复施加已被 trace 计算的遮挡。

GSurface 指向 immutable frame surface metadata；内容含 instance/material identity 与 generation、closure family、flags。几何/拓扑变化使跨帧映射失效。ID 不要求每像素唯一，但必须足以联合深度/法线/motion 拒绝错误重用。

Motion 基线采用独立几何 pass、相同 jitter 和 depth-equal 规则；其额外几何成本必须记账。高能力 profile 可扩展 MRT 合并 motion，但不能让基础平台隐式需要六个 attachment。

透明、分层材质、clearcoat、SSS、hair 不强塞进这个四 MRT。通过独立 closure/profile、辅助数据或 forward+ path 声明额外成本。任何新 profile 都必须给出 raster/trace/reference 的一致材质语义。

### 7.3 默认 shading 约定

采用能量受控的 metallic-roughness GGX 基线，明确 roughness 下限、Fresnel、normal/shading-normal correction、单双面与单位约定。漫反射不得未经校正与完整镜面重复分配能量。后续多重散射补偿必须版本化并同步 reference。

Opaque surface encode 不读取灯表、不含 ambient、不调用 evaluate_direct。透明 pass 可执行 forward lighting，但必须消费同一 plan/light table/visibility 合同，不能另建灯世界。

## 8. 帧拓扑与事务

### 8.1 与原生 schedule 对齐

| 阶段 | Owner | 输入 → 输出 | 失败语义 |
| --- | --- | --- | --- |
| Extract | semantic extractors | journal / camera → pending snapshot | 缺 journal 则全量；非法输入拒绝 publication |
| PrepareAssets | residency services | 材质/几何/灯资产 → ready / pending | 显式占位策略；不能冒充资产已驻留 |
| CreateViews | ViewCoordinator | 相机/targets → 完整 view set | 零尺寸暂停；无相机不造 identity view |
| PrepareViews | Planner | frozen revisions / capabilities → draft plan | 不支持则选择声明 fallback 或拒绝 view |
| Queue | draw preparation | visible draws + resolved layouts → bound draws | 缺 binding 则报错，禁止白色管线成功 |
| Prepare | resource coordinator / providers | reservation + uploads → executable plan | 资源不足重新规划，不能半套替换 |
| Phase | graph executor | 同代资源 → recorded / submitted work | 提交失败不确认新 GPU revision |
| Cleanup | retirement / telemetry | completion proof → ack / retire / sample | 未完成 staging 不回收；异步读取不阻塞常态帧 |

表中的逻辑 freeze 必须在 draft plan 产生前完成；Queue 不得依赖稍后才创建的未知 layout。layout 在 pipeline prepare 之前稳定，buffer 内容可在资源准备阶段填充，提交前统一验证。

### 8.2 GPU pass DAG

```text
Scene uploads ──────────┬─> Candidate build ────────────────┐
View + Material + Mesh ─┴─> Depth / Surface ─> Motion ─────┤
                                     ├─> Shadow requests -> Visibility
                                     ├─> GI cache / probe / trace updates
                                     └─> Screen hierarchy / Reflection trace
Candidate + Surface + Visibility ────────> Direct raw -> Direct reconstruction
Surface + GI representations ───────────> Diffuse raw -> GI reconstruction
Surface + Reflection providers ─────────> Specular raw -> Specular reconstruction
Reconstructed signals + emission ───────> Opaque HDR
Opaque HDR + Forward transparent / Volume -> Scene HDR
Scene HDR -> Exposure / Upscale -> Display transform -> Present / Capture
```

某些 cache 更新允许使用上一有效 revision，但必须声明 age、失效域和可见误差；同帧 receiver 的最终 shading 不得误读上帧 GBuffer。SSR 的 source color 明确用 direct+diffuse+emission 的 pre-reflection buffer 或标记的历史；禁止读写同一最终 reflection 输出造成反馈。

### 8.3 Freeze → prepare → submit → acknowledge

`FrameToken` 包含 frame、scene/light/material/geometry revision、device generation、ABI digest；`ViewToken` 另含 target/view/surface-profile/history generation。执行资源必须匹配 token。

1. Freeze：取得不可变语义快照，记录 journal checkpoint；后续编辑属于下一帧。
2. Prepare：reservation、GPU table 派生、上传 staging、pipeline/bind group 检查；不提前推进 acknowledged GPU revision。
3. Record：图验证读写、尺寸、格式、布局、范围与 revision，再录制命令。
4. Submit：成功提交只证明队列接受；保存 submission proof。多队列实现要记录跨队列依赖；单队列是通用基线。
5. Acknowledge：符合上传/执行证明后推进对应水位；host 回收仍等待实际 completion。缺失 proof 不得吞掉 journal。

CPU 准备失败可释放未提交 reservation；GPU 已提交后不可声称 rollback GPU 时间线，只能标记失败输出、保留已消费资源到完成并在下一帧修复。失败 publication 不覆盖上一有效 CPU 快照。

## 9. ABI 与 binding 合同

### 9.1 固定小型 ABI

以下为 v20 新合同，区别于现有工作树布局；实施时必须以机器检查固定，而不是凭 `repr(C)` 推断 WGSL：

| Block | v20 内容 | 逻辑大小 | 校验 |
| --- | --- | --- | --- |
| SurfaceFrameV1 | clip_from_world + previous_clip_from_world，两 mat4 | 128 B | column-major、各字段 offset、matrix stride |
| SurfaceMaterialV1 | base factor vec4 + surface/flags/reserved 的四 u32 | 32 B | 16B 分组，无尾部 vec3 对齐歧义 |
| ViewFrameV1 | 六 mat4 + 八 vec4 的等价 scalar layout | 512 B | 下述字段表、16B shader alignment |
| DispatchHeaderV1 | extent_xy、frame_low、flags 四 u32 | 16 B | integer range、dispatch ceil 边界 |

ViewFrameV1 的六矩阵依次为 current jittered clip-from-world、current world-from-clip、previous jittered clip-from-world、view-from-world、world-from-view、current unjittered clip-from-world。八个 16B 槽依次为 camera relative xyz+near、viewport origin+extent、render+output extent、current+previous jitter、current+previous pre-exposure+far+reserved、frame/view/history/flags 四 u32、origin-high xyz+reserved、origin-low xyz+reserved。更多字段通过新版本或独立 block 扩展，不偷偷挤进 padding。

Light SoA、surface metadata、candidate、reservoir 不在文档凭空固定全套 stride：唯一 schema 生成 Rust/WGSL accessor、offset/stride manifest 和 layout digest。其具体 binary layout 锁定是 M1/M3 准入项；未生成/验证前不得宣称 ABI 完成。

### 9.2 各 pipeline 的组绑定

| Pipeline family | group 0 | group 1 | group 2 | group 3 |
| --- | --- | --- | --- | --- |
| SurfaceEncode | b0 SurfaceFrame；b1 dynamic material block | mesh/instance storage | material textures/samplers | profile extension，默认无 |
| Motion | view + previous transforms | mesh/instance + deformation previous | 必需的 alpha-test textures | 默认无 |
| LightingEvaluate | ViewFrame / dispatch | canonical-derived Light SoA / candidates | GBuffer / metadata / visibility | provider output/history |
| Composite | ViewFrame | reconstructed signal textures | emission / surface mask | HDR output 所需资源 |

这是逻辑 binding map；存储纹理写出等后端差异通过编译 manifest 显式展开。不同 pipeline 的 group 0 layout 不是可互换对象。

Surface producer 不绑定 directional / punctual / area buffer。当前 `ForwardLitFrame` 的 ambient 与 light counts 应随职责拆分删除。MESH 和 MESH_BIN draw contract 都必须声明所需 bind groups 与 dynamic offsets，queue 不能继续提交空列表。

SurfaceMaterialV1 是小型 draw header，不是完整材质。roughness/metallic/normal/AO/emission factors、纹理与采样参数从该 header 指向的 immutable material table 和纹理组读取；完整 material schema 同样生成 layout manifest。缺参数是验证失败，不能以 header 中没有这些字段为理由只写常量材质。

Dynamic uniform stride 为 `align_up(logical_size, device.min_uniform_buffer_offset_alignment)`；offset 必须满足对齐、u32 可表达范围和 buffer bounds。不能把常见的 256B 当跨设备常量；较小的最小对齐约束代表更强兼容能力。

Pipeline provider 暴露 layout identity/digest；binding cache 只能使用该 provider 的 layout。恢复先建立新 layout，再建新 bind groups。验证同时覆盖 size、offset、alignment、array stride、matrix convention、shader reflection 与一次真实 dispatch/readback。

## 10. 资源矩阵、在途帧与显存

| 资源 | 逻辑 owner / 写者 | 生命周期 | resize / loss |
| --- | --- | --- | --- |
| canonical CPU scene / journal | LightScene / author commands | persistent CPU | 不因 device loss 丢失 |
| light SoA + identity maps | LightGpuProducer | persistent + per-frame versions | 重新上传；旧版等待最后使用完成 |
| view/material uniform ring | FrameBindingAllocator | frames-in-flight | 未完成 slot 禁止覆盖 |
| GBuffer / depth / motion / HDR | TargetResources / surface & composite | per-target-generation | 新尺寸整体分配、提交前原子发布 |
| candidate / tile lists | sampler provider | per-view transient | 图生命周期结束后可 alias |
| shadow maps / page pool | shadow provider | cache + frame mapping | 失效/eviction 带 fence，loss 重建 |
| probe bricks / radiance cache | GI provider | CPU asset + GPU cache | residency 重新建立，版本校验 |
| reservoir / denoise history | 相应 signal provider | per-view history ring | cut/profile/尺寸不兼容则 reset |
| pipelines / bind groups | pipeline provider / binding service | device + layout generation | 不跨设备复用 |
| readback staging | CaptureService | asynchronous leased | completion + consumer release 后回收 |

frames-in-flight 是 driver 协商值，不能假定固定 2 或 3。每个动态 ring slot 持有最后 submission；忙时选择扩容、等待或丢弃可选 capture，禁止覆盖仍在读取的数据。

临时资源 alias 只在图证明生命周期不重叠且格式/usage 兼容时允许。History/cache 不与当帧 transient 随意 alias。缓存淘汰后仍可能有在途引用，必须延迟物理回收。

显存账本同时记录 requested / reserved / resident / committed / retired-pending / peak；CPU shadow copy、BVH、stream staging、PSO、capture 亦需单列。planner 的 reservation 不等于真实 resident allocation。

1440p 即 2560×1440：32 B/px 基础单份为 117,964,800 B = 112.5 MiB。它不包含 HDR、history、reservoir、shadow 或驱动对齐。不能直接乘 frames-in-flight 就声称总 VRAM；以真实重叠 live set 和分配器统计为准。

## 11. Direct 与 many-light

### 11.1 正确性基线

每个 covered pixel 从 GBuffer 取得完整 PBR 参数、重建相机相对位置与 view direction，再读取候选、评价 BRDF、visibility 与灯辐射。无效 surface 清零 direct，不用首个 sample 广播全屏。

directional / point / spot 先实现可与 CPU oracle 对照的解析路径。点灯近距离奇点通过有限发光体或明确的 artistic near-field 规则处理；range cutoff 及平滑窗口显式版本化，不把任意 attenuation 当物理定律。

### 11.2 Candidate pipeline

按 view、tile/cluster、空间层级建立候选；区分 directional/global、finite local、environment 与 mesh emitter。采样权重可用几何上界、功率、距离、法线、历史 visibility，但有偏/漏选必须标记并测量。

小候选集可解析求和，大候选集可采用 ReGIR / reservoir provider。threshold 不绑定固定灯数，由 overlap、分布、材质、噪声和成本决定。候选溢出有显式计数和 fallback；禁止静默截掉最亮灯。

### 11.3 Reservoir 合同

保存 sample identity/domain、target/proposal density、累计权重、有效样本数及其饱和策略、age、bias mode、producer revision 和 visibility 状态。重用前检查 light mapping、surface、motion、material、camera cut 与 shadow 失效。

delta 与连续样本的 measure 必须分开处理；离散选择概率和连续立体角/面积 PDF 不能直接比较。是否能使用 MIS 由共同积分域与 measure 决定，不由“delta 永不参与”口号决定。

Reuse 的 Jacobian、clamp、visibility reuse 和 M 上限均属于 estimator version。GPU reference 对照必须覆盖灯新增/删除、拓扑变更、快速移动、暗场孤立亮灯和过曝小面积灯。

### 11.4 可扩展性定义

4096 灯仅是测试规模之一。报告 64/256/1024/4096 总灯、可见灯、每像素有效重叠灯、candidate entropy、sample steal、ray miss、history reject 与图像误差。总数增大但均在视野外，不构成 many-light 质量证明。

## 12. Visibility / Shadow

Visibility 独立于辐射 estimator：返回可见度、hit/distance、representation、revision、validity，不返回伪造补光。

### 12.1 Provider 分层

- Conventional：directional cascade、spot、point cubemap 等，作为可验证基线；area 的 visibility 需明确近似或 sampled trace。
- Virtual shadow：虚拟地址、页表、物理池、request、feedback、dirty caster、priority、eviction、coarse fallback。
- Trace：软件 BVH / 可选硬件 RT，使用共同几何与 opacity/material 规则。
- Contact screen path：仅补近场局部信息，不作为离屏阴影的唯一权威。

每个 light class × shadow provider 形成支持矩阵；不宣称八类灯共享完全相同 VSM 参数。

### 12.2 Page request 与失效

请求包含 light incarnation、projection/clipmap epoch、virtual page、request domain、importance、deadline、caster revision 与 requester budget。domain 至少区分 opaque receiver、coarse、volumetric、translucent、reflection、debug。

灯 transform、caster transform、alpha-test、WPO/PDO、skinning、bounds 和 representation LOD 均可触发相关页面失效；仅改灯色不应重画纯深度页。不能把 bounds 不更新导致的缓存错误归咎于 temporal。

缺页时只能采用声明的 coarse map / trace / 有界保守策略。使用 stale page 必须满足版本和误差规则，记录 age；不得为了性能把已失效页当新页。无阴影 fallback 会改变图像能量，必须在 profile 中显式授权并在 explain 中显示。

### 12.3 阴影质量

bias、normal offset、filter footprint、PCF/PCSS、ray count、penumbra、contact 与透明透射都有独立测试。薄墙、近接触、掠射角、级联边界、快速变形和页池压力不能只靠静态大场景截图验收。

## 13. GI：路径归属、表示与更新

### 13.1 合成数学

表面出射辐射按不重叠目标拆分：

```text
L_out = L_emission + L_direct_diffuse + L_direct_specular
        + L_indirect_diffuse + L_indirect_specular
```

实现中可让 direct 输出完整 direct BRDF；此处的分解用于 AOV 与能量归属。间接项的路径域必须排除已计入 direct 的一次光源连接。环境光直达与环境经反弹也需分开。

在同一间接域内，优先采用“每个采样事件只选一个有效表示”的 fallback：screen valid → world trace / cache → probe / environment。若采用 `sum(a_i * estimate_i)`，必须声明估计同一目标、权重归一与偏差，且 confidence 与样本相关会引入额外偏差；不因权重和为 1 就自动无偏。

MIS 仅用于可比较 measure、已知 proposal 与正确权重的 estimator。probe 插值、screen hit 和 world path 不能凭一个 confidence 就合成出物理证明。艺术混合允许存在，但须标记 Approximate/ArtDirected、给出误差场景与参考图。

### 13.2 表示选择

| 表示 | 擅长 | 限制与 fallback |
| --- | --- | --- |
| screen trace / screen probes | 屏幕内细节、低成本接触与短路径 | 离屏/背面缺失；不能用拉伸边缘填洞 |
| baked probe bricks | 静态漫反射、大世界流式、便宜 fallback | 非动态遮挡/镜面真值；scenario/validity 必须匹配 |
| dynamic DDGI / radiance cache | 低频动态间接、预算更新 | 漏光、滞后、薄壁；需要 relocation/visibility 与失效 |
| surface / texel cache | 可重用命中点材质与辐射 | 覆盖、变形、视角依赖材质和纹理分辨率受限 |
| software world trace | 通用世界可见性、离屏路径 | BVH 构建/更新、alpha、skinning 的预算必须纳入 |
| hardware trace | 高能力设备的精度/吞吐替代 | capability provider；不能改变语义或省掉 fallback |

### 13.3 ProbeBrickScene / Bake

probe 数据包含 bake schema、bake revision、scene geometry/material/light dependency hash、baking set、lighting scenario、streaming cell、sky occlusion revision、relocation/validity、encoding 和 residency。

brick 分辨率、spacing、SH/方向编码、压缩和更新频率由质量/内容 profile 决定，不固定一个尺寸覆盖所有场景。静态光变化使相关 bake 失效，动态灯不得重复叠加已经包含在 bake 中的贡献。

Bake 支持预检、增量 dependency、进度、取消、checkpoint、结果原子发布和场景过期检测。过期 bake 可以作为显式 preview，但不能标 production-valid。Scenario blend 必须限定相同语义布局或有可验证重映射。

### 13.4 更新预算

按可见重要性、变化、age、误差与驻留排序更新；镜头切换允许有限 burst，但受 frame 与 memory hard cap 限制。高速移动不能因为缓存未收敛输出未初始化值；回退到合法粗表示并暴露 warming-up 状态。

## 14. Reflection 与复杂介质

镜面域使用统一粗糙度/BRDF 约定，按 roughness、footprint、screen confidence、scene capability 和预算选择 screen / world / cache / environment。Planar reflection、capture 和 SSR 不可重复添加同一反射域。

reflection raw 输出 radiance、hit distance、roughness、validity、variance、producer；denoiser 使用法线、depth、motion、surface identity。环境 fallback 是真实 environment asset，不是常量 ambient 白光。

低 roughness 对 representation mismatch 极敏感；不得只通过加大 blur 隐藏。测试包含镜面阶梯、凹凸细节、镜后物体、离屏高亮、skinned object 与 camera cut。

透明使用 premultiplied compositing 的明确约定；refraction 单独声明读取哪个 scene color 与厚度模型。体积输出 scattering/transmittance，有自己的 froxel/history 与 shadow request domain。头发/水/SSS 通过专用 closure 和 AOV 扩展，不能因主链完成就自动勾选这些能力。

## 15. TemporalInputContract 与分信号重建

### 15.1 统一输入

TemporalInputContractV1 固定如下语义：

- motion 表示当前像素中心到上一帧对应像素中心的位移；单位为上一帧内部像素，坐标原点左上，x 向右、y 向下。
- motion 从非 jittered 投影计算，不含 jitter；重投影按分别记录的 current/previous extent 与 jitter 映射。不能在 extent 改变时直接相加。
- depth 为该 view 的 reverse-Z device depth，另提供解码参数与 camera-relative 重建；无效/sky 由 validity 判断。
- current/previous surface、instance、material producer identity 与 epoch 可查询；prev deformation 缺失时标 invalid motion。
- scene-linear buffer 可 pre-exposed；保存 current/previous pre-exposure，历史转换使用明确比例，禁止二次曝光。
- 提供 reactive / transparency / emissive-change mask、camera-cut、projection-change、DRS、history reset reason。

静态相机、仅 jitter、匀速移动、物体自运动、skinning 与 DRS 各有符号/尺度 oracle。不能只通过“画面看起来没拖影”判定 motion 正确。

精确定义：`m = (uv_previous_unjittered - uv_current_unjittered) * previous_extent`。重投影先从当前 raster UV 去掉 current jitter 得到非 jittered UV，再加 `m / previous_extent`，最后加 previous jitter / previous extent；viewport origin 单独转换。这样 motion=0 只表示非 jittered UV 不变，不意味着不同 extent 下像素坐标不变。

### 15.2 独立 history

DI reservoir、shadow、diffuse GI、specular、volumetric 与 final upscale 分别保存其 history、variance、age、rejection 和 debug AOV。共享输入合同，不共享匿名颜色历史。

每个 history key 至少包含 view identity/generation、signal、provider/schema、surface profile、相关 epochs、extent。小幅 DRS 只有 provider 证明可重映射时才保留历史；否则 reset，不默认按比例复制。

### 15.3 重建顺序

raw signal → validity/reprojection → temporal estimate → spatial filter / resolve → radiance composition。Final temporal upscale 处理完整 scene color 与透明 reactive mask。每个 denoiser 声明输入/输出分辨率、是否预曝光和 latency。

clamp / firefly rejection / history cap 属于 bias policy，进入 capture。禁止为了漂亮截图隐藏 raw-noise 与历史误差 AOV。

## 16. HDR 与显示输出

DisplayOutputContract 包含 working space/primaries、scene-linear 约定、pre-exposure、参考白 nits、目标 primaries/EOTF、display peak/black policy、swapchain format/color space、display capability revision、metadata policy 与 output transform version。

默认工作空间可先采用 linear Rec.709/sRGB primaries，但宽色域输入的转换/越界处理必须显式；升级工作空间必须更新 bake、材质/纹理、capture 与 reference 的版本。

```text
Scene-linear radiance
  -> pre-exposure aware lighting / histories
  -> exposure (manual or histogram metering)
  -> reconstruction in declared exposed domain
  -> rendering / tone / gamut transform
  -> display encoding
  -> matching swapchain colorspace + present
```

曝光 histogram 在 scene-linear 可还原域测量，剔除 sky/UI 的规则可配置。曝光变化不能污染物理 radiance oracle。UI 可在 display-referred 域按 reference white 合成，不与场景共享任意曝光。

SDR、HDR10 类编码、scRGB 类路径按平台能力显式协商；不把 PQ 输出塞入声明为 SDR 的 swapchain。移动窗口/切屏/显示能力变化提升 display epoch，重建 output pipeline；不得因此无故清空所有世界 GI cache。

capture 明确导出 scene-linear EXR/AOV 或 display-referred 图，两者都带色彩元数据。截图 RGB 一致不代表显示器上的峰值/参考白正确。

## 17. Planner、自动质量与性能预算

### 17.1 Planner 的输入与输出

输入：frozen scene、view、能力、residency、telemetry 窗口、质量目标、显存、功率/热/CPU 压力、capture policy。输出：provider graph、每 view/signal 预算、reservation、fallback、history actions、estimated cost、reason trace。

Provider 可返回 estimate / unsupported / insufficient budget；只有 Planner 能批准更改计划。Provider 不能偷偷超过 hard cap，真实执行超预算需上报下一帧控制。

### 17.2 控制器初始策略

以下参数是可调设计初值，必须经 telemetry replay 和实测校准：

1. 以完成的 GPU samples 为准，滚动窗口至少 120 帧，保存 P50/P95/P99；标记 timestamp 可用性、异常和测量开销。
2. 预算误差 `e = P95 / target - 1`。连续 8 个有效样本 `e > 0.05` 才常态降档；显存硬上限/分配失败可立即触发应急路径。
3. 常态一次只调整一个成本族，降档后至少 30 帧 cooldown；避免同时降 DRS、GI、shadow 导致无法解释质量损失。
4. 升档需要至少 180 帧满足 `P95 < 0.85 * target`，无 residency emergency，且预计额外 live set 可容纳；升档后 120 帧 cooldown。
5. 升/降 provider 有独立 switching penalty 与 warming budget；不能每几帧在 screen/probe/trace 间振荡。
6. 变化优先级按 perceptual loss / saved cost 与稳定 deterministic tie-break；紧急 eviction 先丢可选 capture/debug、过期缓存，再降低可选更新。
7. 固定 telemetry replay 应产生相同 plan 序列；跨 GPU 的浮点图像只要求容差，不要求 seed 相同就 bitwise 一致。

“单调降级”只表示减少申请成本/近似阶数并遵守误差策略，不表示每像素亮度必须非增。减少 shadow、样本或改变 estimator 可使局部像素变亮；验收应看能量归属、误差分布、variance 与时间稳定性。

### 17.3 预算示例，不是成绩

原始 4.3 ms 保留为一个待校准 lighting GPU envelope 示例；不把 CPU 时间混进 GPU 加法。各项是串行等价归属账本，不代表开启 async 后的关键路径可相加。

| GPU 成本族 | 示例上限 ms |
| --- | ---: |
| light candidates / sampling | 0.35 |
| shadow update / visibility | 0.85 |
| direct evaluate / reconstruct | 0.65 |
| GI trace / cache / resolve | 1.10 |
| reflection trace / resolve | 0.65 |
| lighting composite | 0.20 |
| safety / bookkeeping headroom | 0.50 |
| 合计 | 4.30 |

该 envelope 排除 opaque geometry、独立 motion、final upscale、display/present；不能宣传为整帧预算。CPU lighting prepare/record 的 0.8 ms 也只是独立待测目标，不与 GPU 4.3 ms 相加求总帧时。

H2 不能只写“中端显卡”：基准记录必须包含准确 GPU/driver/backend、CPU/RAM、功率、内部 2560×1440 与输出尺寸、场景 hash、质量 profile、热状态和帧样本。设备/内容未锁定前，所有时间格标 Target / Unmeasured。

### 17.4 优化优先顺序

先消除全量提取、重复灯表、全灯逐像素、位置冗余、错误 history reset、无界 cache 和同步 readback；再做 tile classification、half-resolution GI、缓存增量、pass fusion / async。所有优化给出成本变化与质量误差，禁止只给平均帧时。

## 18. Authoring、Import、Bake 与 Explain

### 18.1 作者层

作者选择物理单位灯、profile、lighting volume 与 environment；高级页池/候选/历史参数默认隐藏。Preset 必须包含 schema version、units、平台预算和迁移函数。切 profile 不修改灯强度来伪装质量一致。

默认场景可以显式创建 environment 与 sun，但系统本身不偷偷注入白色 ambient。黑场是合法物理结果；工具可提示“缺少光源或环境”，不能自动补光后仍宣称 oracle 正确。

### 18.2 导入合同

| 来源 | 必须处理 | 无法一一对应时 |
| --- | --- | --- |
| glTF + punctual extension | 源轴/单位、intensity、range、spot angle、node transform | 保留 provenance，明确转换与 unsupported extension |
| USD / Lux | units、exposure、normalize、shape、orientation、texture、linking | 不假定与 Prism emission 数值直接等价 |
| UE 导出数据 | 导出格式/版本、单位、mobility、IES、源半径、阴影/艺术参数 | adapter 显式 loss report，不承诺像素级复现 |

Importer 是外层适配包，不把外部引擎类型写进 canonical core。先验证官方格式语义再发布 mapper；未实现的 importer 在支持表标 planned。非均匀缩放对灯面积/方向的影响必须有规则和测试。

批量导入输出机器可读报告：path、source field、converted value/unit、warning/error、fallback、asset dependency。用户能定位为什么某盏灯被拒绝或效果不同。

### 18.3 Explain 与调试

`explain(frame, view, pixel/light/region)` 返回：输入 revisions、最终 provider、预算/压缩原因、候选/选中样本、visibility 来源、cache age/miss、history rejection、重建与 display transform。查询不能引入常态全屏同步回读。

必需调试视图：surface identity、roughness/metallic/emission、motion、candidate overlap/entropy、reservoir age、shadow page state/dirty reason、GI coverage/probe validity、reflection miss、raw vs denoised、history reset、pre-exposure、HDR clipping/gamut、VRAM 与 pass timings。

## 19. Provider 运行时治理与维护

Manifest 声明 provider identity/version、ABI range、输入输出域、capability、资源上限、history schema、依赖、queue/access、支持的 closure/light/content 与 fallback。注册时验证无环依赖和输出归属冲突。

动态切换流程为 register → validate → warm → atomic select at frame boundary → drain old submissions → retire。热卸载不能销毁在途资源；新 shader layout 改变必须换 digest 和 generation。

provider failure 分为 recoverable reject、GPU validation/device error、timeout 与 host crash。第三方 in-process Rust/native 代码不具备真正崩溃隔离；若要隔离 host crash，必须额外使用进程/受限运行时。文档不能把 `Result` 当安全沙箱。

Core 不依赖 SDK vendor；可选 RT/denoise adapter 不能反向成为原生 driver 的必要依赖。CI 以 dependency graph 检查禁止 Bevy/旧 lighting adapter 进入 production feature closure。测试 fixture 可保留外部对照，但不参与默认包。

维护策略：一个 schema registry、一个 frame transaction、一个资源退休协议、一套 capture/evidence。重复表和兼容代码有删除门；不是无限追加注册表。PR 必须说明新增 live set、pass、variants、history、fallback 和故障测试。

## 20. Resize、DRS、多 target 与恢复

### 20.1 Resize / DRS

target identity 与 extent generation 分离。新 extent/profile 先分配验证，再在帧边界发布；旧资源在最后引用的 submission 完成后退休。零尺寸窗口暂停 surface acquire，不反复创建 0×0 texture。

DRS 改内部 render extent 不必改变 window output extent；相关 GBuffer、dispatch、candidate grid 与 history 必须统一更新。并发 offscreen capture 使用独立 target/view/token，不抢用窗口 history。每个 target 有内存预算与优先级，防 capture 拖垮交互窗口。

正常 resize 不以全设备 wait_idle 作为长期方案；可作为启动期 correctness path，但必须记录 stall 并在 P2 删除常态阻塞。

### 20.2 Device recovery 事务

```text
Lost(old generation)
  -> stop new submissions / quarantine old GPU handles
  -> create replacement device + capability profile
  -> create pipelines/layouts -> resources -> bindings
  -> reupload canonical snapshot / rebuild residency
  -> validate complete replacement set
  -> atomic publish(new generation) -> warm histories/caches
```

| 子系统 | 恢复动作 | 强制校验 |
| --- | --- | --- |
| driver / graph | 新 generation，清除旧 acquire/queue 状态 | 新 capability 可能变弱，必须重新规划 |
| MeshPipelines | 先恢复 provider layout | bind groups 不能来自旧 layout |
| targets | 全套附件与 HDR 重建 | 格式、extent、usage 与新设备匹配 |
| light / material / geometry | 从 CPU 语义重新派生/上传 | revision 一致，不依赖丢失 GPU 内容 |
| shadow / GI caches | 重建或按 CPU asset 驻留 | 旧物理页/brick handle 不可用 |
| history / reservoir | reset 或可验证 CPU checkpoint | 默认禁止跨设备 GPU history 复用 |
| composite / bindings | 按新 layout / target 建立 | cache key 包含所有 generation |
| capture | 中止旧 readback 并报告失败 | 不输出空图当成功 |

恢复任一步失败不发布混合资源集；旧设备已失效时不能“回滚继续用旧 GPU”，只能保留 CPU scene、显示恢复状态并重试。replacement device 再丢失也必须可重入。故障注入覆盖每一步及局部 provider fallback。

## 21. 验收体系与需求追踪

### 21.1 证据等级

| 等级 | 含义 | 不能代替 |
| --- | --- | --- |
| D0 | 设计/源代码路径已定位 | 编译、执行、正确性 |
| D1 | 静态/单元/ABI 测试通过，记录命令与版本 | 真实生产 GPU 像素 |
| D2 | 原生 GPU 路径提交并回读，数值 oracle 合格 | 全场景视觉质量/性能 |
| D3 | 场景金样、时间序列、恢复/压力及平台矩阵通过 | 量产负载预算达标 |
| D4 | 锁定 workload 的质量/性能/资源/可靠性全部达标 | 未测试硬件/内容上的普遍保证 |

本次文档没有把任何条目升级到 D2–D4；当前路径观察最多是 D0。历史其他工作报告的测试也不能自动覆盖本版新合同。

### 21.2 Requirement → Contract → Evidence

下列 test/job/artifact 是应建立的标识，不表示 CI job 已存在。每项初始状态为设计目标，owner 为逻辑角色；实施时登记负责人、commit、设备和结果。

| ID / 需求 | 合同 / owner | Test / CI job（计划） | Artifact | Exit gate |
| --- | --- | --- | --- | --- |
| L01 唯一灯权威 | LightScene / scene owner | light_scene_revision / lighting-contracts | delta+snapshot trace | 无双重派生写者；D1 |
| L02 生命周期身份 | identity map / scene owner | delete_recreate_cross_scene / lighting-contracts | handle/remap dump | 无陈旧引用；D1+D2 |
| S01 真正逐像素 | SurfaceProfile / app owner | textured_pbr_readback / lighting-gpu | GBuffer+HDR+oracle diff | 局部材质变化对应像素变化；D2 |
| V01 真实相机 | ViewFrame / view owner | camera_orbit_projection / lighting-gpu | matrices+image sequence | 透视/正交/cut 正确；D2 |
| A01 ABI / binding | manifests / driver owner | layout_dynamic_offsets / lighting-abi | offsets+shader digest+readback | MESH/MESH_BIN 均验证；D1+D2 |
| D01 Direct | BRDF+units / direct owner | analytic_lights / lighting-reference | CPU/GPU AOV diff | 物理/艺术模式分明；D2 |
| H01 Shadow | visibility/page / shadow owner | thin_wall_wpo_overflow / lighting-scenes | page/visibility+golden | 不读错页，错误可解释；D3 |
| G01 GI 合成 | estimator domain / GI owner | white_furnace_cornell_switch / lighting-reference | per-domain AOV+error | 无重复域，偏差量化；D3 |
| G02 Bake/stream | ProbeBrickScene / GI owner | stale_bake_cell_stream / lighting-streaming | revisions+residency trace | 无过期真值与未初始化采样；D3 |
| R01 Reflection | hybrid specular / reflection owner | roughness_offscreen / lighting-scenes | raw/denoised+miss AOV | fallback 无双计能量；D3 |
| T01 Temporal | TemporalInput / temporal owner | jitter_motion_drs_cut / lighting-temporal | motion oracle+video metrics | 无错误重投影；D3 |
| O01 HDR | DisplayOutput / output owner | exposure_display_switch / lighting-output | EXR+encoded ramp+metadata | 无重复曝光/错误 EOTF；D3 |
| Q01 自动预算 | immutable plan / planner owner | telemetry_replay_pressure / lighting-planner | plan trace+oscillation count | 同输入同计划、受控震荡；D1+D3 |
| M01 资源寿命 | retirement / resource owner | inflight_resize_capture / lighting-stress | allocations+fence trace | 无在途覆写、无持续泄漏；D3 |
| F01 恢复 | recovery transaction / app owner | injected_loss_each_stage / lighting-recovery | generation+failure trace | 不混代、可重入；D3 |
| P01 性能质量 | workload envelope / perf owner | fixed_hardware_scenes / lighting-perf | P50/P95/P99+VRAM+error | 同时满足锁定预算；D4 |
| X01 旧路删除 | native dependency / architecture owner | production_feature_graph / lighting-boundary | deps+symbol/call scan | 无生产 Bevy / 假 surface / 白管线；D1+D2 |

### 21.3 场景 / 设备矩阵

场景至少包含解析灯、白炉、Cornell、多灯重叠、暗室点亮、薄墙、植被/WPO、skinning、巨大世界、室内外切换、镜面离屏、透明粒子、volume、probe streaming、DRS、HDR 切屏、多 target 和持续设备故障。

设备按真实适配器建立 portable integrated、mainstream discrete、high-end discrete，以及实际支持的 Vulkan/Metal/DX12/WebGPU backend。可选 hardware RT 的测试独立；不能用高端 RT 设备替代 portable correctness。

跨设备数值测试使用解释清楚的绝对/相对容差和图像阈值；同设备 replay 可追求更强确定性。随机算法同时报告重复 run 的方差与时间域误差。

### 21.4 Evidence package

每包保存 source commit + dirty diff digest、toolchain、features、shader/ABI hashes、资产与场景 hash、seed/estimator、plan、设备驱动、分辨率/profile、命令、测试时长、原始 timestamp/readback、AOV/EXR、图像/时域误差、VRAM、reset/eviction 和失败日志。

所有阈值在运行前写入 workload manifest。不能测试后挑帧、改阈值或只保留平均值。没有可运行 CI 的平台标 manual/unverified，不伪造绿色完成状态。

## 22. 破坏性实施与删除门

### 22.1 单一生产 owner 的迁移原则

从 M0 开始，canonical LightScene 是唯一语义 owner；原生 `prism_render_app` 派生路径作为目标生产 owner。旧 `prism_render_scene` 灯 GPU 适配路径隔离到非默认测试/对照 feature，不能与新路径共同上传同一生产 light table。

若原生路径尚不能满足已有场景，明确标为迁移中的未完成能力；不启动两个互相兜底的生产 renderer 来掩盖缺口。

| 阶段 | 实施交付 | owner | 删除门 |
| --- | --- | --- | --- |
| M0 合同冻结 | schema/ownership/support matrix、现状 capture | architecture | 禁止新增第二份 canonical/schema |
| M1 Surface 闭环 | real View、四 MRT、材质、motion、dynamic bindings、逐像素 composite | app + material + driver | identity fallback、假 surface、生产白管线删除 |
| M2 Direct+Shadow | 三类解析灯、conventional visibility、HDR回读 | direct + shadow | per-surface sample 广播生产路径删除 |
| M3 Canonical 全灯 | area/sky/emissive、compact maps、GPU ABI、导入 validation | scene + sampler | index-only 公共身份、重复灯表派生删除 |
| M4 GI+Reflection | screen/probe/world fallback、分域合成、时域输入 | GI + reflection + temporal | 伪 ambient、重复间接能量、未版本化 history 删除 |
| M5 规模与工具 | many-light/VSM、stream/bake、planner、explain | providers + tools | 固定全量扫描、无界 cache、常态同步回读删除 |
| M6 稳定性 | 多 target、DRS、device recovery、HDR、故障注入 | app + resource + output | 混 generation、常态 resize wait_idle 删除 |
| M7 量产准入 | 场景/设备矩阵、D4 envelope、旧依赖清理 | perf + architecture | 旧 API/registry/Bevy lighting production closure 删除 |

没有固定人周承诺；按垂直切片和证据门推进。每次提交保持编译/测试边界可说明，避免全仓大替换。文档与 manifest 同步，不在未通过 GPU 回读时把设计目标改为“已实现”。

### 22.2 不保留无终点兼容层

未公开 API 可直接重命名/删除；持久化资产、capture 与 bake 仍需要 schema 版本和迁移/拒绝规则。破坏性更新不等于静默误读旧数据。旧 replay 无法可靠转换时返回 incompatible schema，不能默认填值伪装成功。

提交只包含本任务明确审阅的文件；不覆盖其他工作树修改。恢复/回归失败先回退 provider selection 或修复，不复活已删除的隐式灯/白管线路径。

## 23. 风险、冻结决策与后续校准

本版已冻结：唯一 canonical、native-only 生产边界、opaque deferred 主链、逐像素材质、四 MRT 基线、独立 motion 基线、显式合成、统一时域和 generation 事务。

尚需测量而非凭文档决定：compact GBuffer 精度、motion 合并收益、software BVH 更新成本、VSM page 大小、probe encoding/spacing、many-light 切换阈值、控制窗口、async compute 收益和各设备预算。

| 风险 | 防线 | 何时重新决策 |
| --- | --- | --- |
| 24B MRT / 32B surface 集成本过高 | depth 重建、profile 分离、测真实带宽 | portable workload 超预算 |
| 独立 motion 重复几何 | MRT 高能力变体、统一 deformation 数据 | geometry-heavy 场景证明为瓶颈 |
| screen/probe/trace 切换引起闪烁 | 分域 ownership、warmup、history rejection | 时间序列误差超阈值 |
| reservoir 隐性偏差 | estimator version、raw AOV、reference | 亮度/时间误差不收敛 |
| 动态场景 cache 大量失效 | dirty domain、预算与显式 fallback | P99 / stale age 超 envelope |
| 复杂内容侵蚀核心维护性 | closure/profile/provider manifest | 新内容不能用既有合同描述 |
| HDR 平台差异 | 输出协商与 capability revision | swapchain/显示能力变化 |

重开决策必须携带 matched benchmark 与迁移影响，不以“某 AAA 产品也这么做”作为唯一理由。

## 附录 A. 官方研究入口与采用边界

以下列出可追溯的一手入口；这是后续 provider 验证的来源目录，不是已锁定版本的合规清单。使用 SDK / 代码前单独审查 license、版本和平台约束。

- [Epic：Lumen Technical Details](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine)
- [Epic：Lumen Performance Guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-performance-guide-for-unreal-engine)
- [Epic：Virtual Shadow Maps](https://dev.epicgames.com/documentation/en-us/unreal-engine/virtual-shadow-maps-in-unreal-engine)
- [Epic：MegaLights](https://dev.epicgames.com/documentation/en-us/unreal-engine/megalights-in-unreal-engine)
- [Unity Graphics：Adaptive Probe Volumes](https://github.com/Unity-Technologies/Graphics/blob/master/Packages/com.unity.render-pipelines.high-definition/Documentation~/probevolumes.md)
- [Unity Graphics：Probe Volumes Streaming](https://github.com/Unity-Technologies/Graphics/blob/master/Packages/com.unity.render-pipelines.high-definition/Documentation~/probevolumes-streaming.md)
- [NVIDIA RTXDI：Integration](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Integration.md)
- [NVIDIA Research：ReSTIR Direct Lighting](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct)
- [NVIDIA Research：ReSTIR GI](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing)
- [AMD GPUOpen：FidelityFX Denoiser](https://gpuopen.com/fidelityfx-denoiser/)
- [AMD GPUOpen：FidelityFX SSSR](https://gpuopen.com/fidelityfx-sssr/)
- [AMD GPUOpen：FidelityFX Super Resolution 2](https://gpuopen.com/fidelityfx-superresolution-2/)
- [Epic：Temporal Super Resolution](https://dev.epicgames.com/documentation/en-us/unreal-engine/temporal-super-resolution-in-unreal-engine)
- [Microsoft：Advanced Color / HDR](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range)
- [ACES：Output Transforms](https://docs.acescentral.com/system-components/output-transforms/)

本次访问部分官方页面未取得可核对正文，因此本版不引用其具体版本、数值或逐字陈述。实现阶段必须将实际采用页面/论文/SDK 的 revision 与摘录结论写入 provider evidence；外链失效不能成为无法重现实验的理由。

## 附录 B. 实施评审最小清单

提交任何 lighting provider 前回答：它估计哪个路径域；输入/输出单位与预曝光是什么；身份和 revision 如何校验；新增多少 attachment/pass/history/live set；资源由谁创建和退休；fallback 是否重复能量；cut/resize/loss 如何处理；支持哪些 light/closure；用哪些 oracle/AOV/金样证明；成本在哪个硬件/content envelope 下成立。

上述问题缺答案时，保持 Planned / Experimental，不进入默认生产 profile。
