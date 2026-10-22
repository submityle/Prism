# 05 · Render Fabric 与特性声明

> 状态：Render Fabric 目标规范与实现证据审计（允许破坏性重构，不保留旧 API）；2026-10-21：Particle CPU payload/geometry 有先前定向记录；局部物理桥、native 共享帧/admission/visibility、composition 共享 SceneFabric/runtime，以及 Bevy Hair/Volume/Decal owner、共享帧聚合与 physical GPU admission/import 已接线并定向复验，全场景统一 recorder 与完整验收未闭合
> 上层导航：[`00-overview_refactor_zh.md`](00-overview_refactor_zh.md)
> 范围：六层渲染装配与四个权威的职责切分、三个正交轴的术语收敛（`RenderDomain` / `ResponseProgram` / `ExecutionFamily`）、特性的三层声明（`FeatureDefinition` / `FeatureVariant` / `PassTemplate`）、变体的确定性选择规则、语义 ID 与帧图 ID 的分离（`ResourceKey` / `ResourceId`）、`ResourceUse` 的完整访问描述、服务依赖的两层模型与服务提供者校验、`FeatureCatalog`、`FrameRenderItem` 与域负载、三份并行服务集与两份 `ShadingFrontend` 的删除、跨子系统回边的一帧延迟契约
> 非范围：SurfaceData 布局（→ [`01-surface-abi_refactor_zh.md`](01-surface-abi_refactor_zh.md)）、`ShadingPlan` 与五类契约（→ [`02-response-composition_refactor_zh.md`](02-response-composition_refactor_zh.md)）、PSO 身份与缓存失效（→ [`03-shader-package_refactor_zh.md`](03-shader-package_refactor_zh.md)）、两套帧图编译器的合并裁决与手排 pass 覆盖（→ [`04-frame-graph_refactor_zh.md`](04-frame-graph_refactor_zh.md)）

---

## 0. 一句话

00–04 已经把"**一个 Artifact、一条着色路径**"定案。本篇补的是它的另一半：

> **一份特性声明，一张特性图，零份并行基底。**

迁移前的"共享基底"由**位掩码 + 前端枚举 + 手写 pass 顺序**三件套重复表达。本篇要求把这 N 份收成**一份声明数据**（`FeatureDefinition` / `FeatureVariant` / `PassTemplate` 三层），顺序仍由 04 的唯一帧图编译器推导。当前声明、求解及部分水体链路已有实现；这不等于所有生产构帧都已消费该声明，证据边界见 §1.3 / §15。

### 0.1 边界裁定：本篇不是被排除的"第三个图编译器"

历史版本排除过"新增第三个图编译器 / 通用节点图求值器（`Response Graph`、`RenderProgram`/`RenderPlan`、Blend/Layer 节点）"。当前 00 已重编为 §3「必须保持的架构边界」，不再有旧 §8；上述排除项仍作为**本篇独立保留的设计约束**，不能因引用重编而删除。本篇的存在**不推翻这条否决**，裁定如下：

| 被排除的东西 | 本篇的处置 |
|---|---|
| 第三个图编译器 | **不新增**。排序、barrier、别名、生命周期全部复用 `prism_render_graph`；`prism_render_architecture::frame_graph` 仅兼容 re-export（04 §1.1） |
| 通用节点图求值器 | **不新增**。响应组合仍由 `ShadingPlan` 的单一 sum type 表达（02 §0）；本篇不引入 Blend/Layer 节点 |
| `RenderProgram` / `RenderPlan` | **不新增**。响应身份由 `ProgramRef::{Builtin, Registered}` 表达，内建值是 `BuiltinProgram::{Lit, Npr, Unlit}`，组合归 `ShadingPlan`（02 §0），不引入第三个并列类型 |
| 新的契约 crate | **不新增**。见 §2.2 裁决 |
| "响应图"这个名字 | **禁用**。`docs/prism_ui_loom_design_zh.md:16,126` 已占用于 UI 反应式依赖图。本篇一律称**特性图（Feature Graph）** |
| `FeatureRegistry` 这个名字 | **禁用**。`pkg/prism_render_architecture/src/diagnostics/registry.rs:46` 已有 `pub struct FeatureRegistry`（feature-health 诊断），并在 `diagnostics/mod.rs:47` 公开 re-export。本篇的新注册表叫 **`FeatureCatalog`** |

**判据**：本篇若需要新写一个求值器、新写一个拓扑排序、或新增一个并列的响应程序类型，即为越界。

### 0.2 四个权威，各管一件事

本节是全文最重要的一条边界，先立在最前面：**`prism_material_pcm::ExtensionRegistry` 不得变成整个渲染系统的总注册表。**

| 权威 | 管什么 | **明确不管** |
|---|---|---|
| **Material PCM**（`prism_material_pcm`） | 材质扩展的身份、命名空间、槽位分配、扩展能力协商 | 渲染特性、变体、pass、帧图资源 —— **一律不进 `ExtensionManifest`** |
| **Feature Catalog**（本篇新增） | 渲染特性、变体、pass 模板这三层声明 | 材质参数与瓣（01/02 的领域）、设备能力协商（backend 的领域） |
| **GPU Scene**（`gpu_scene`） | 持久化场景对象：`InstanceRecord` + 增量上传与镜像 | 本帧 transient 资源、域负载 payload（§6） |
| **Frame Graph**（`frame_graph`） | 本帧资源、访问、依赖、barrier、别名、生命周期、调度 | 声明本身 —— 它是声明的**消费者**，不是声明的存放处 |

四者互不越界是可机械检查的（§12 门禁 12/13/14）。

### 0.3 本版修订记录（对照评审发现）

**历史 v7 针对第五轮两条发现的修订**（不是本轮执行记录）：

| # | 发现 | 落点 |
|---|---|---|
| P1-m | `provides` 已移到 `FeatureKind::Service`，但多处公式仍按旧写法把它当**变体字段**（`v.provides`；`provides(chosen[providers[s]])` 里 `chosen[…]` 拿到的是 `FeatureVariant`） | §4.1 相 A 第 4 条、§4.1 相 B 求解伪码（引入 `provides_of(f) = catalog[f].kind.provides`，准入条件改按 `providers[s]` 逐项求并）、§4.1「良基」段、§4.2（明确 `provides` 不在变体上）、§4.6.1 写入校验公式改为 `∀ v ∈ f.variants` 全变体量化、§12 门禁 25（重命名为 `..._by_every_provider_variant`）/ 32、§7 替换表 |
| P2-n | 历史版本的 S1 只覆盖“类型骨架 + 槽位校验”，没有把硬需求闭包、`providers[s]`、`requires_services_extra`、`FeatureSelection::resolve` 纳入同一前置契约 | §9 S1 保留完整求解要求；当前能核实求解实现与门禁 27/29/30/31/32 的测试定义，不能据此声称本轮运行通过或生产接线完成，见 §15 |

**v7 附带的量化收紧**：门禁 25 的量化范围从"提供者（当前选中）变体"改为 `∀ 变体 v ∈ f.variants`——只校验选中变体会让"当前恰好能用"掩盖"其它变体没实现该服务接口"，而设备一旦换能力就会静默选到一个不写该资源的变体。

**v8（2026-10-16 实现子项校正，经本轮限定证据范围）**：本篇此前的“现状”段落记录的是迁移前快照，不能继续作为当前状态使用。当前源码具备以下实现证据：

| 项目 | 当前证据 | 当前结论 |
|---|---|---|
| `overlay_opacity` | CPU `resolve_hybrid_weights` / `compose_mask_blend`（`water/shading/hybrid.rs:109-204`）与 WESL `water_hybrid_weight_contract` / `water_hybrid_overlay_factor`（`water_surface_raster.wesl:1666-1719`）；入口从 `view.viewport.z` 读取（`:1795-1809`） | 已接入实际 WESL 消费路径 |
| CPU/GPU Hybrid 公式 | CPU 与 WESL 都是 clamp 两个归一化区域信号、取 `max` 得 NPR、取补数得 PBR、再 clamp overlay；源码契约测试见 `water_hybrid_golden.rs:469-519` | 公式已统一 |
| 水体入口 | `surface_pass.rs:167-177` 的 `MaskBlend` 选择 `water_surface_fs_mask_blend`；WESL 实现在 `water_surface_raster.wesl:1795-1809` | 已替代旧 `water_surface_fs_hybrid` 入口 |
| `MaterialRenderClass` | `record.rs:41-50` 仅保留七个 blend/raster 值；域状态由 `render_state.rs:95-157` 映射 | 四个旧值已移除，管线状态映射已存在 |
| S4b/S5 子项 | `water_hybrid_golden.rs:276-466` 的八项 CPU 行为 golden 与网格；`water_hybrid_gpu.rs` 的冻结 legacy GPU closure parity；`render_fabric_legacy_gate.rs:11-19,54-91` 的旧符号静态门禁；S5 ABI/cache 定向回归见 §15.2 | 当前公式、入口、枚举、冻结 GPU 输出与 ABI/cache 回归已有证据；仍不能追证“先护栏后删除”的历史时序，也不等于完整 Scene/Fabric/TAA、跨适配器或真实 driver binary 验收完成 |

本篇把“测试定义/静态门禁存在”和“本轮实际执行结果”分开表述：已有运行记录只作为历史证据，当前轮次新执行的命令只按其真实输出记账，不把未成功执行的命令写成通过结果。

**v9（此前文档审计）**：修正 `ProgramRef` / `BuiltinProgram` / `ShadingPlan`、8 主链 + 23 后处理、统一帧图类型路径与 00–04 当前节号；收回“唯一生产装配路径”“S1–S6 全完成”等超出证据的结论。`DeviceFabric::compile` / `CompiledFeatureGraph::prepare_runtime` 的定义和测试不等于生产调用，完整验收仍未闭合。全部原始验收要求、37 条禁止与 32 项门禁保留；该次审计仅编辑文档，没有运行测试。

**v10（2026-10-20，scene 接线审计与边界收口）**：`pkg/prism_render_scene/src/gpu/fabric.rs` 曾经分配独立 buffer、绑定通用 compute shader，再由 `RenderGraphSystems::Render` 单独提交；它是 adapter probe，不消费真实 `FrameRenderItem`、scene-owned imports 或 shading/water recorder，不能作为 Fabric 生产执行证据。本轮已移除该 probe 的初始化、record 与提交；`SceneFabric` 仍只负责从实际 Bevy `RenderDevice` 冻结 capability/feature selection，供既有 shading/water 准入消费。

生产 lowering 对全场景仍缺四个同一真实帧的输入/责任方：（1）Bevy scene frame builder 在 render-world production prepare 中生成真实 `Surface/Bounds` 帧项；native `RenderApp` 另有 Surface/Particle 共享帧与 post-GPU Prepare 刷新（见 v22），但仍缺全部 scene 域 payload 与共享全场景 recorder 消费；（2）覆盖全部 scene 域的精确 `FrameResourceBinding` / `RuntimeImport`；（3）覆盖全部实际 shading/scene pass 的 `RuntimePassBinding`；（4）Bevy 所有的 allocation、history publication、submission 与 completion 生命周期。五个 response plane 已在局部 resident composition recorder 中以 canonical `ResourceKey`、动态模板和真实 Bevy buffer 进入 production `RuntimeImport`；生产 recorder 现使用共享 `SceneFabric` selection 与 `fabric_runtime`，`CompositionFabric` 仅保留为 acceptance wrapper；这关闭局部共享选择/runtime seam，不是全场景 lowering。未以固定 blocker 资源、恒零计数或空 recorder 替代缺失域。v10 审计时移除的 probe 不构成生产闭环；水体 compute 与 composition 两个窄 adapter 也不能替代全场景 lowering。

**v11（2026-10-20，payload/history 契约补强）**：payload 以对象 generation、domain、history stream 与 role 联合判重；当前帧/domain 的访问一次建立索引，避免逐 item 重扫全部 pass，并修正先读历史版本、后写当前版本时的误拒绝。`ResourceHistory` 记录外部已确认完成的精确 persistent binding，核对 physical storage、版本与布局，拒绝缺失、重复及倒序 publication；按单资源前缀显式退休，并保留 publication 水位，禁止退休后重放旧帧。Fabric 单元测试的历史完整记录为 134 项；2026-10-21 当前工作树已扩展并复验为 175/175，范围和命令见 §15.2。该结果仍不是 scene 生产生命周期接线证据。

**v12（2026-10-20，局部生产切片接线）**：`water/dispatch.rs` 的唯一 water recorder 在 `SceneFabric + SceneRhi` 可用时消费 resident `WaterGpuBody` 的真实 buffers、pipelines、bind-groups 和 dispatch schedule；`shading/fabric_executor.rs` 为其创建真实 `FrameResourceBinding` / `RuntimeImport`、调用 `DeviceFabric::compile` 与 `prepare_runtime`，经 RHI queue encode 后加入宿主 `RenderContext`。当时 resident Composition 通过独立 `CompositionFabric` 导入五个 canonical response plane 与 HDR/visibility 资源并提交真实 command buffer。该 v12 记录仅证明当时的两条局部生产切片；此后 composition 已接共享 `SceneFabric` selection/runtime（见 v22），但 TAA/history、全域 `FrameRenderItem`/payload 与其他域仍未进入同一全场景 recorder，完整验收仍未完成。

**v13（2026-10-20，water history 证据校正）**：water compute adapter 现只声明并导入本帧真实可证明的 `WATER_SURFACE_RESOURCE` 与 `WATER_FOAM_RESOURCE`。此前把本帧稍后由 `SprayEmit` 写入的单缓冲 `spray_spawn` 标成 `WATER_SPRAY_RESOURCE @ frame-1`，会把未完成发布的当前帧资源冒充上一帧 history；该伪绑定已撤掉。canonical built-in 中 spray→water 的 `FrameLag(1)` 仍保留为设计与 CPU 编译契约，不代表 production 已闭合：现有 `SprayHistory` 双缓冲保存的是 particle/foam feedback，与 `spray_spawn` 语义不同，不能直接互换。要让 adapter 消费 canonical spray history，仍需统一的 persistent ownership、GPU completion 后 publication、exact import validation、resize/version 迁移与 retirement。

**v14（2026-10-20，active-pass compile 切片）**：`ResourceTemplates::compile_active` 与 `DeviceFabric::compile_active` 支持 recorder 按 frozen `FeatureSelection` 编译指定 pass slice，仅 lowering 活跃 pass 实际访问的资源；拒绝重复/未选中 pass，并要求被省略的依赖由调用方显式列为已完成依赖。`active_compile_uses_frozen_selection_and_explicit_completed_dependencies` 此前定向运行记录为 `1 passed; 0 failed`。本轮复跑未到达测试执行阶段，被并发 `prism_physics_core` 编译错误阻断。该能力关闭的是 active-slice compiler 缺口，不会自动建立共享的全场景 `SceneFabric` selection、完整域 imports/history 或唯一生产 recorder。

**v15（2026-10-20，Bevy Surface 帧项生产提取的历史接线记录）**：`PrismGpuScenePlugin` 在 render-world `PrepareResources` 中、GPU Scene transaction 与 geometry registry 同步后调用 `build_render_fabric_frame_system`，把真实 committed scene instances 降为 `RenderFabricFrame` resource。当时该 builder 只产生 `Surface/Bounds` payload；未被共享全场景 recorder 消费，其他域尚未接入该 builder（native 后续聚合另见 v22）。当次 `prism_render_scene` 定向测试先因磁盘空间耗尽未进入源码编译；清理 target 后，authoring 定向测试又被 `prism_render_architecture::particle::compiler` 的当时编译错误阻断；这仅描述该次失败尝试，不替代 §15.2 的后续记录或最新源码复验。

**v16（2026-10-20，Surface 帧项进入可见性输入）**：`visibility::build_unified_visibility` 不再独立从 `RenderGpuScene::mirror().live_handles()` 枚举对象，改为使用 `RenderFabricFrame.items()` 中已校验并排序的 `scene_handle`；`PrismVisibilityPlugin` 显式声明该系统排在 `build_render_fabric_frame_system` 之后。handle scratch 改为 system-local `Vec` 并跨帧复用已增长容量，避免每帧重新分配同尺寸的临时 vector。这样 Surface 帧项成为当前可见性消费端的真实输入，收敛了“帧提取已生产但无人读取”的一个 seam。builder 先按 handle 排序，再用相邻比较以 O(n) 时间、O(1) 额外空间拒绝重复项，避免排序后再建一棵重复的 `BTreeSet`。此接线尚未通过当前工作树编译/运行验证；visibility 仍自行读取 GPU Scene/material/geometry 生成旧的可见性 work item，并未因此成为共享 `SceneFabric` lowering/recorder，也不覆盖其他域 payload 或 history 生命周期。

**v17（2026-10-20，spray feedback owner/access seam）**：`SprayHistory::feedback_access` 现能为每个实体返回生命周期内不复用的 `FabricHistoryOwner`、本帧 producer feedback buffer、lag-one buffer 候选、网格尺寸和 allocation version；resize 会重置本地 dispatch marker 并递增 allocation version。该 API 尚未接入 `SceneFabricHistory` 的 `enqueue → host submission → completion poll → validate_import`，也未把持久 RHI lease、`WATER_SPRAY_RESOURCE @ frame-1` 的 `RuntimeImport`、versioned retirement 接入 recorder。候选 buffer 的存在或本地 `LagOneHistory` dispatch marker 都不是 GPU completion/publication 证明；此前 `previous_frame_published` 命名已移除以免混淆。此切片当前只有 rustfmt/差异空白检查，没有 Cargo 编译或测试通过证据。

**v18（2026-10-20，spray history completion 接线与性能收敛）**：后续实现已超出 v17 的 access seam：spray adapter 现在把精确的上一帧 binding 通过 `SceneFabricHistory` 入队，并仅在 `validate_import` 认可已完成 publication 时使用；否则清零 lag-one 输入。per-instance owner 单调且不复用，feedback allocation 持有 RHI lease；resize/removal 将旧 buffers/leases 留存到 RenderQueue completion 驱动的 retirement 水位越过对应前缀。`RenderFabricFrame` 每帧重建复用 item-vector 容量；spray removal scan 复用 scratch vectors，并用排序后的实体 key 做 `O((live + resident) log live)` 查询，避免每个驻留对象重复扫描全部实体。新增代码仅有 rustfmt 与 diff whitespace 检查，Cargo 编译/测试仍待锁释放后确认；这些仍是 water/spray 局部接线，不代表 TAA history 或全场景 recorder 闭环。Composition 当前声明的 accesses 均为当帧资源，无 `FrameLag`，因此不要求伪造 composition history publication。

**v19（2026-10-20，TAA history 的真实持久资源与完成门控接线）**：TAA history 不再由 system-local Bevy texture cache 单独表达。`taa/resources.rs` 通过 `SceneRhi` 分配真实双槽 `rgba16float` persistent texture，并为每个 retained view 建立新的 `FabricHistoryOwner`；`ViewTaa` 保存本帧写槽的 exact `FrameResourceBinding`。`taa/dispatch.rs` 只有在真实 resolve dispatch 编码后才 enqueue producer，Finish 阶段仍由宿主 RenderQueue completion hook 发布；下一帧只验证精确的 `(resource, frame, version, physical, layout)` 上一帧 binding，缺失 publication 时显式关闭 history。resize、removal、disable 和 device-table replacement 都通过新的 owner 与 retirement watermark 保留旧 lease 至完成。新增单测覆盖 completion 前后导入、多帧发布、缺失 history、slot/physical identity、owner retirement 与 replacement 隔离；v19 当时只完成 rustfmt/diff whitespace 检查；后续 2026-10-22 main 的 strict TAA GPU 重试已通过 19/19（含新增 lifecycle 7 项，见 §15.2），补齐真实 bind-group/queue completion 与局部 GPU 运行证据。TAA 仍是独立窄切片，尚未进入统一全场景 recorder。

**v20（2026-10-21，Particle runtime snapshot 的 Fabric payload / geometry bridge）**：`prism_particle_render::ParticleFabricFrame::from_runtime_snapshot` 直接消费 `prism_particle_app::ExtractedParticleSnapshot` 中的 emitter ranges 与 live particles，保留实体 index/generation，并为每个 emitter 生成精确 count、为非空 emitter 生成 live-position bounds `ValuePayload`，以及 catalog 中 `PARTICLE_STREAM_RESOURCE`、`PARTICLE_ALIVE_RESOURCE` 的 canonical `ResourcePayload`；非连续/越界 range、实体身份错配、非有限位置与未归属 particle 都会被拒绝。v20 当时原生 `RenderApp` 的 `PrepareResources` 从同一 extracted snapshot 构建 `PreparedParticleFabricFrame`，其严格 consumer 要求 count/bounds/stream/alive 四类 payload 完整、唯一且与模板一致，随后形成 `RendererGeometry::Sprite` draw contract；snapshot 或 consumer 失败时发布带当前 frame number 的空 frame/geometry，不复用旧帧。源 snapshot 缺失时同时清空 particles 与 emitters，避免 stale emitter ranges。`prism_particle_render`、`prism_particle_app` 定向单测分别为 **6/6**、**4/4**；此前阻断测试的 `prism_ecs` `Archetypes::created_count()` 已由 step-inspector **8/8** 复验修复。该 payload 仍是语义资源描述，既不分配物理 GPU handle，也未作 Fabric runtime import/bind-group/record；geometry 也尚未由统一全场景 recorder 消费。

**v21（2026-10-21，非法域路径显式化与当时编译记录）**：Hair/Volume/Decal 的 domain builder、resource contract 和 consumer 不再使用生产 `unreachable!` 作为“已校验”证明；Surface/Particle 误入这些 non-surface API 时返回显式 `UnsupportedDomain` 错误。`fabric_runtime::prepare` 收紧为 crate 内入口，避免 private-interface 警告继续伪装成公共生产 API；该模块现在可对真实 RHI handles 执行 `compile_active → prepare_runtime → record_into/submit`，但仍没有接入统一的 scheduled 全场景生产调度。此前 `cargo check -p prism_render_scene --features legacy-bevy --lib` 曾通过；当次重跑在无关的 `prism_asset` 工作树错误（`pkg/prism_asset/src/scheduler.rs:757,876,903` 的 `Flight` 初始化缺少 `cancellation_waiters` 字段）处停止，域帧定向测试也未产生通过证据；`cancellation_waiters` 仅为历史阻断，当前 `Flight` 定义已无此字段，后续反馈已越过 asset 初始化，不据此判断当前编译状态。Particle 另有 renderer-owned `BufferId` 物理资源桥和本地 graph recorder 连续性校验，但尚未进入该统一 runtime。因而本补记关闭的是错误路径与先前编译验证，不关闭 S4/S6、全域 recorder、TAA completion、跨适配器或性能验收。

**v23（2026-10-21，三域 owner 与 physical resource admission/import）**：Bevy render world 现由 `HairRenderWorldOwner`、`VolumeRenderWorldOwner`、`DecalRenderWorldOwner` 持有稳定 object/history/motion identity；Hair/Decal 发布持久快照，Volume plan 单次消费，并通过 `CollectDomainContributions → BuildFrame` 一次性聚合，producer 删除后不回放旧数据。`fabric_domains::gpu` 将 canonical semantic role 解析为带 table/physical identity 的 `SceneRhi` buffer/texture lease，校验容量、range、alignment、usage、texture extent/dimension/format/mip/sample/default view、exact binding/import coverage 与 physical alias，生成运行时 `ResourceId` import 和 record-ready 借用视图。另有独立的生产 volumetric fog 管线，使用真实 3D froxel textures 与三段 compute pipeline；它尚未接入 SceneFabric 的完整资源/pass/hazard/typed recorder/copy-back 链。默认域测试 **22/22**、owner/producer **3/3**、启用 `legacy-bevy` 的完整 `fabric_domains` 筛选 **33/33**。三域 owner/admission 层级已完成；Hair/Decal 的真实 pipeline/consumer，以及 Volume fog 的统一 SceneFabric 执行与各域 full-frame parity 尚未完成。

**v22（2026-10-21，共享 selection/runtime 与 native 共享帧源码同步）**：`gpu/fabric.rs::init_scene_fabric` 在设备初始化时使用 `composition::scene_fabric(caps)`，将 composition 声明并入 builtin catalog/roots 后一次冻结；生产 `response_composition_pass` 消费共享 `SceneFabric` 与 `RenderFabricFrame`，通过 `fabric_runtime::prepare_with_templates` 的 `compile_active → prepare_runtime` 导入真实 response buffers、HDR/visibility textures 并绑定 pipeline/bind group，encode 后交给 `RenderContext::add_command_buffer`。`CompositionFabric` 不再是生产 ECS resource，仅为 acceptance 使用同一 scene selection 构造器的封装。runtime 对缺失/重复 binding、frame/layout/range、recorder coverage 等作拒绝校验；opaque pipeline/bind-group 的同设备与反射契约仍由调用方保证，不宣称完全自动认证。

原生 `RenderApp` 的 `fabric_frame.rs` 则从 authoritative `CpuRenderScene::live_records` 和 `PreparedParticleFabricFrame` 聚合真实 Surface/Particle。Particle lowering 已移到 `extract_particle_fabric_frame`，一次读取 snapshot；帧身份使用 `RenderRecord.frame_index`（无 clock 的 direct tests 才用 0），source simulation clock 可冻结而 render clock 前进。Extract 顺序为 Particle → 共享帧 → visibility；PrepareResources 内 `(prepare_particle_snapshot, finalize_render_fabric_frame).chain()` 在 GPU Prepare 后保留 Surface、刷新 Particle 角色和保守 bounds，不以 capacity 冒充 exact count/liveness，disabled/pending/recovering 移除 Particle。非法 payload/geometry 或错帧清空整帧且保留错误，finalize 不用部分 GPU 数据修复失败提取。native Queue 早于 Prepare，此刷新不能写成发生在 Queue 前。

`systems.rs::admit_render_fabric_frame` 在 window/offscreen host graph record/submit 之前拒绝缺失、错误或错帧输入；它不覆盖更早 Particle Prepare 的 simulation 提交。`visibility_bridge` 在共享帧缺失/错误/错帧时拒绝所有 camera results，按完整 camera identity、view 参数与带 generation 的 Surface membership 校验，缺失 Surface 时拒绝受影响 camera 的整份 work，不保留局部列表或跨 camera 拼接。Bevy 的 `build_unified_visibility` 仍消费其 Surface 帧项，不能与 native 混域聚合混写。新增接线与相关测试定义已存在，但本次只改文档，不运行 Cargo、不新增通过记录；composition 仍是 compute slice，native 仍未打通全域 SceneFabric import/recorder，S4/S6、32 gates、历史时序及全部未完成验收保留。

**v6 针对第四轮三条发现的修订**：

| # | 发现 | 落点 |
|---|---|---|
| P1-j | `transitive_deps(active_roots)` 无法直接计算：依赖挂在 `FeatureVariant.requires_services`，选变体之前不知道闭包应含 `SSR` 还是 `Probe`；且同一 `ServiceId` 可由多个特性提供而无仲裁规则 | §4 类型定义（需求拆为**特性级硬需求** `FeatureDefinition.requires_services` + **变体级软需求** `requires_services_extra`）、§4.1 相 B（闭包只读硬需求、软需求只作准入、单遍无回溯）、§12 门禁 29 / 31 / 32 |
| P1-k | 依赖矩阵与 `provides` 的类型边界未闭合：任何 `FeatureVariant` 都能声明 `provides`，于是 `requires_services` 可能被非 `Service` 特性满足，边矩阵无法机械执行 | §4 类型定义（**`provides` 移入 `FeatureKind::Service` 载荷**，类型层面排除）、§4.1 相 A 第 5/6 条（提供者唯一 + 类型边界）、§4.6.1（验证公式改写为按 `providers[s]` 唯一映射）、§12 门禁 29 / 30 |
| P2-l | `ValuePayload` 注释允许"指向统一的参数块"，与门禁 6"Value 不承载 buffer / texture"冲突 | §6.3（`Value` 收紧为**定宽、不可变、可 `Copy` 的内联值**；统一参数块必须走 `ResourcePayload` 或显式 `ParameterBlock` 资源）、§12 门禁 6、§11 禁止 13 / 新 37 |

**v6 附带的接口收紧**：同一服务特性的**所有变体必须写入同一组 `ServiceOutput.resources`**——变体之间的差异是"怎么算"，不是"发布成什么"。这既是服务语义本身的要求（消费方不应知道选中了哪个变体），也让 `ServiceOutput` 能安全地待在**特性级**（否则 `provides` 必须随变体变化，类型边界又会破），并由门禁 32 守住。

**v5 针对第三轮三条发现的修订**：

| # | 发现 | 落点 |
|---|---|---|
| P1-h | 求解器没有"激活根集合"，遍历并选择**全部** Service / Subsystem / Response，等于把所有特性变成必选项（场景没有水也要求 `WaterSimulation` 可用；可选服务缺 baseline 会让整台设备初始化失败） | §4.1 相 B（`resolve(active_roots, device_caps, quality)`，只求解根集合的依赖**闭包**；并写死 `active_roots` 的静态来源与冻结规则）、§12 门禁 28 |
| P1-i | 分层依赖规则没有成为静态门禁：未明确禁止 `Service → Subsystem/Response` 与 `Subsystem ↔ Response`；且求解公式里仍留着 `engine_services ∪` 隐式例外，与 §4.6.1"不得开特例"自相矛盾 | §4.1 相 A 第 3 条（**边种类矩阵**，违反即构建期错误）、§4.1 约束 5（删除 `engine_services` 旁路）、§12 门禁 27 |
| P2-f | 域负载强制**每个**字段都是帧图资源（`bounds` / `phase_params` 这类也要 `lifetime` / `size` / `alignment`），会为过门禁制造大量伪资源 | §6.3（改为 `DomainPayload = Resource(ResourcePayload) \| Value(ValuePayload)` 和类型 + 判据表）、§12 门禁 6、§11 禁止 13 / 新 33 |

**v4 针对第二轮五条发现的修订**：

| # | 发现 | 落点 |
|---|---|---|
| P1-f | 变体选择不是全局可定义的算法：依赖"已选变体"却没定义求解顺序；"编译期执行一次"与"按设备选择"冲突 | §4.1（拆为**相 A 静态校验 / 相 B 设备初始化求解**）、§5.1、§12 门禁 22 |
| P1-g | 水体 Hybrid 段落自相矛盾（同处既写 `REGION_ILLUMINATION + ShadingPlan`，又写"`REGION_ILLUMINATION` 不参与"）；且历史计划曾把入口收口与等价验证拆成 S4/S4b/S5 | §7.1（替代物统一为 `ShadingPlan::MaskBlend` + 常驻 `CompositionPass`）、§9（分别列明 S4/S4b/S5 的实现、生产与验收证据；门禁 23 的原始等价要求保留） |
| P2-c | `ServiceOutput.resources` 与提供者 pass 的实际写入之间没有机械约束 | §4.6.1（`resources ⊆ 提供者 pass 的写入集`；**v7 把量化从"提供者变体"收紧为 `∀ v ∈ f.variants`**）、§12 门禁 25 |
| P2-d | `ResourceKey.slot` 与 `FeatureId` 的分配规则不完整：按声明顺序发号会漂移身份 | §4.5.1（**显式常量**；`FeatureCatalog` 只校验唯一性、不发号）、§12 门禁 26 |
| P2-e | 两处过期交叉引用；§5.1"唯一消费者"表述过窄 | §10 / §14（引用修正）、§5.1（改为唯一**声明**权威：消费者只读取**已冻结**的 `FeatureSelection`，不得各自重选） |

**v3 针对第一轮六条发现的落点**：

| # | 发现 | 落点 |
|---|---|---|
| P1-a | `FeatureDefinition.domain` 与服务级/响应级不一致 | §4.0（`kind` + `Option<domain>`） |
| P1-b | `ResourceKey` 不该做成全局大枚举 | §4.5 / §4.5.1（`{ owner, slot }` 命名空间键） |
| P1-c | `queue` 有两个事实源（`PassTemplate` 与 `ResourceUse`） | §4.3（采纳方案 A，删 `ResourceUse.queue`） |
| P1-d | `FeatureVariant` 缺确定性选择规则 | §4.1（`rank` + `RenderQualityTier` + 全序选择函数） |
| P1-e | 服务依赖缺"服务提供者"关系 | §4.6（`provides: &[ServiceOutput]`） |
| P2-a | 删除水体 Hybrid 需要更强验收 | §3.6、§7.1（**撤回"可删"判断**） |
| P2-b | `MaterialRenderClass` 四值不能只做枚举迁移 | §3.1.1、§7.2（建 `→ PipelineState` 映射） |

---

## 1. 历史基线与当前实现证据

本节保留迁移前的代码事实，作为为什么需要 Render Fabric 的审计依据；§1.1 / §1.2 / §1.7 的旧符号位置均是**迁移前快照，不是当前行号**。“当前状态”以明确标注的源码证据及其边界为准。

### 1.1 历史基线：两份同名的 `SharedBaseServices`

| bit | `water::SharedBaseServices`（`COUNT = 9`） | `volumetric::SharedBaseServices`（`COUNT = 7`） |
|---|---|---|
| `1 << 0` | `VIRTUAL_GEOMETRY` | `HYBRID_GI` |
| `1 << 1` | `HYBRID_GI` | `RESTIR` |
| `1 << 2` | `RESTIR` | `VSM_SHADOW` |
| `1 << 3` | `VIRTUAL_SHADOW` | `FROXEL_VOLUME` |
| `1 << 4` | `FROXEL_VOLUME` | `ATMOSPHERE_LUT` |
| `1 << 5` | `RT_REFLECTION` | `RT_REFERENCE` |
| `1 << 6` | `RT_CAUSTICS` | `TEMPORAL_UPSAMPLE` |
| `1 << 7` | `PATH_TRACING_REF` | — |
| `1 << 8` | `TEMPORAL_UPSAMPLE` | — |

- 水体：`pkg/prism_render_architecture/src/water/mod.rs:439-447`（文档注释 + 定义）、`:451-467`（九个常量）、`:486`（`COUNT = 9`）、`:655`（测试 `shared_base_set_algebra`）、`:666`（测试 `every_frontend_shares_the_full_advanced_base`）
- 体积：`pkg/prism_render_architecture/src/volumetric/mod.rs:628`（定义）、`:655`（`COUNT = 7`）、`:802`（测试 `shared_base_set_algebra`，与水体**同名**）、`:812`（第二个测试，亦同名）

这是迁移前的证据：同类型名、同成员名 `ALL`/`NONE`/`COUNT`，语义却完全错位。当前状态已改变：`pkg/prism_render_architecture/tests/render_fabric_legacy_gate.rs:11-19,54-91` 将这些旧符号列为禁止回归项，当前服务闭包由 `render_fabric/builtins.rs` 与 `ServiceSet` 提供（实现证据及验收边界见 §15）。

### 1.2 历史基线：两份同名的 `ShadingFrontend` 与四分叉入口

| 位置 | 定义 |
|---|---|
| `water/mod.rs:523` | `ShadingFrontend { Pbr, Npr, Custom, Hybrid }`，注释自述 "diverge only in the lighting response" |
| `volumetric/mod.rs:694` | `ShadingFrontend { Pbr, Npr, Custom, Hybrid }`，同一段自述的副本 |

两份都提供 `shared_base()`，且**无条件返回 `ALL`**（`water/mod.rs:543-545`、`volumetric/mod.rs:717-719`），配同一句自述：

> Every frontend consumes the full base: `NPR` is *not* a reduced path.

两份测试套件**名字也相同**：`shared_base_set_algebra`、`every_frontend_shares_the_full_advanced_base`。

`ShadingFrontend` 共被 **10 个文件**引用，横跨两个 crate：

```
pkg/prism_render_architecture/src/water/{mod,asset,pipeline,assembly,frame_golden}.rs
pkg/prism_render_architecture/src/water/shading/mod.rs
pkg/prism_render_architecture/src/water/gpu/surface_pass.rs
pkg/prism_render_scene/src/water/{surface_node,surface_pipeline,surface_shading}.rs
```

这是迁移前快照。当前水体选择由 `WaterShadingPlan` 表达：三个 `Single` 程序仍分别进入 `water_surface_fs_pbr` / `_npr` / `_custom`，而组合计划进入 `water_surface_fs_mask_blend`（`pkg/prism_render_architecture/src/water/gpu/surface_pass.rs:167-177`）。当前 WESL 只有 `water_surface_fs_pbr`、`_npr`、`_custom`、`_mask_blend` 四个入口（`pkg/prism_render_scene/src/shaders/water_surface_raster.wesl:1770-1809`），旧 `_hybrid` 不再存在。

### 1.3 当前状态：声明与部分水体消费已存在，窄切片已接入，全场景装配尚未闭合

| 载体 | 位置 | 当前下游 |
|---|---|---|
| 水体服务闭包 | `water/fabric.rs:246-334` | `WaterFabricPlan` 使用 catalog-derived `required_services` 与四个 resident pass |
| MaskBlend 声明与链接检查 | `water/fabric.rs:289-332` | `is_link_complete` 校验 resident pass 与资源声明；本身不建立 driver import/export，也不执行 Fabric runtime |
| 实际水体响应 producer | `prism_render_scene/src/shading/graph.rs:196-222` | 声明 water 写入 `response_left/right/controls/water_coverage`，常驻 Composition 随后消费；属于现有 scene shading 链路 |
| 水体计算的窄 Fabric adapter | `water/dispatch.rs:record_water_fabric_commands`、`shading/fabric_executor.rs:prepare_water_runtime` | 在 `SceneFabric + SceneRhi` 可用时，把 resident water body 的真实 surface/foam buffer、pipeline、bind-group 和 dispatch schedule 经过 `compile → prepare_runtime → queue.encode → RenderContext`；该 adapter 自身只覆盖水体计算，不声明未完成发布的 spray history；response composition 由共享 SceneFabric selection/runtime 下的局部 adapter 覆盖，二者仍未统一覆盖 TAA/history 与全域 fanout |
| 水体入口 | `water/gpu/surface_pass.rs:167-177` | `WaterShadingPlan::MaskBlend` → `water_surface_fs_mask_blend` |
| 旧权威符号 | `tests/render_fabric_legacy_gate.rs:11-19,54-91` | 静态扫描禁止 `SharedBaseServices`、`ShadingFrontend`、`bound_base` 等回归 |

**当前结论**：旧服务类型与入口的收口已有源码证据；catalog / selection 已统一服务声明，但**不能写成已替代所有生产装配路径**。`pkg/prism_render_scene/src/gpu/fabric.rs:13-51` 从实际拥有 scene buffers 的 Bevy `RenderDevice` 读取能力，创建 `DeviceFabric` 并冻结选择；`water/spray_history.rs:156-167,234-250` 消费选择并进行 particle root admission。`shading/plugin.rs:463-515,629-633` 已将同一 `SceneFabric` 的冻结选择接入 shading executor 初始化与准入，但 `build_shading_execution` 仍调用现有 `shading_frame_graph().compile()`；`shading/execution.rs:97-160` 的选择校验不等于通用 Fabric 编译。另一方面，`water/prepare.rs:39-94` 虽把 `SceneFabric` 传给 spray history，仍直接 prepare 水体、创建 GPU buffers 与绑定，未证明由通用 Fabric 编译驱动。

`render_fabric/device.rs:33-48` 已定义 `DeviceFabric::compile`，`render_fabric/runtime.rs:195-200` 已定义 `CompiledFeatureGraph::prepare_runtime`。当前水体计算 adapter 已在 `water/dispatch.rs` 的真实 recorder 中调用 `prepare_water_runtime`，由 `shading/fabric_executor.rs` 绑定 resident surface/foam 资源、创建 runtime recorder，并把 RHI command buffer 加入宿主 `RenderContext`；resident Composition 另有一条局部 recorder 导入 canonical response/HDR/visibility 资源并提交真实 command buffer。这关闭的是两个真实但彼此独立的局部生产切片。water adapter 有意不把当前帧 `spray_spawn` 冒充 `frame-1` 的 `WATER_SPRAY_RESOURCE`；canonical `FrameLag(1)` 只有在统一 history publication/import lifecycle 闭合后才能进入该 production recorder。两条局部切片仍不等于 TAA history / 全域 payload / 其他域的统一 `SceneFabric` 装配，也不等于所有域都取消 host fallback。因此 selection 初始化、局部生产接线、声明链接测试、水体现有响应链路与 **全场景 Fabric compile → prepare_runtime → execute 闭环**必须分开记账。

### 1.4 它的意图是对的，只是载体错了

意图由 `docs/prism_material_pipeline_design_zh.md:18-19` 的两条目标记录在案：

| # | 目标 | 当前实现 |
|---|---|---|
| G3 | 高级特性不留白（GI/RT、虚拟阴影、虚拟几何、时序超分、动态光、粒子、透明、毛发、布料、体积、水面） | `FeatureCatalog` / built-in roots / service closure 是声明与选择证据，不是全部高级特性已接生产、已达质量验收的证明；见 §15 |
| G4 | **NPR 不是降级路径**：NPR 必须与 PBR 享有同一份几何/光照/阴影/时序基底 | 服务需求与共享 lighting context 契约保留，水体两臂已有具体链路；全域共享基底的生产消费仍需独立证据 |

**G3/G4 的意图全部保留，本篇只替换它们的载体。** 任何以"删除 `SharedBaseServices`"为名削弱 G4 的改动都违反本篇 §4.6 与 00 §3 的共享 lighting context / 权威边界约束。

### 1.5 持续设计契约与当前证据边界

| 已存在（复用） | 位置 |
|---|---|
| 资源身份 / 访问种类 / 队列 / 生命周期 | `prism_render_architecture::frame_graph`（`ResourceId`、`AccessKind`、`QueueClass`、`ResourceLifetime`，04 §1.1） |
| 设备能力集与协商 | `prism_render_architecture::backend::capability`（`Capability`、`CapabilitySet`、`WEBGPU_BASELINE`） |
| 材质扩展能力与协商 | `prism_material_pcm`（`Capability`、`CapabilitySet`、`requires_caps`、`negotiate`、`is_lossless`） |
| 响应身份 | `ProgramRef::{Builtin, Registered}` 与 `BuiltinProgram::{Lit, Npr, Unlit}`（`pkg/prism_render_material/src/shading_plan.rs:21-25,56-75`）；Custom 通过注册程序表达 |
| 组合轴 | `ShadingPlan`（02 §0） |
| shader 编译期变体 | `shader_package::permutation`（`PermutationSpace` / `PermutationKey` / `PermutationSelector`） |
| 持久场景与增量 | `prism_render_architecture::gpu_scene`（`InstanceRecord`、`SceneChangeJournal`） |

| 必须保持的设计契约（不等于完整验收已通过） | 说明 |
|---|---|
| 跨子系统资源声明 | 继续通过 `ResourceKey` / `ResourceUse` 表达，不得回退到无类型字符串或隐式依赖 |
| 语义资源身份 | `ResourceKey` 与帧内 `ResourceId` 分离；ID/slot 显式声明且稳定 |
| 顺序与 barrier | 资源 hazard 由 authoritative resource graph 推导，不得以手填顺序替代；当前 `shading/graph.rs` 仍有显式 `depends_on` 语义前置边，不能称为已全部由 `accesses` 自动生成；自动推导与旧顺序/barrier 等价要求仍由门禁 2 保留 |
| 服务与底层资源 | `ServiceOutput` 必须由 provider 的每个变体实际写入，并发布同一资源接口 |
| 变体选择 | 只由 active roots、设备能力和静态质量档决定；结果整体冻结、与注册顺序无关 |
| 跨帧回边 | 水体 spray→foam 等回边必须显式 `FrameLag(1)`，不得靠遍历顺序掩盖环 |

### 1.6 已关闭问题与保留约束

1. **固定数量断言矛盾已收口**：当前 shading graph 为 **8 个主链 pass + 23 个后处理 pass，共 31 个声明**；常驻 Composition 已计入 8，不能重复加一。`shading/graph.rs:79-225`、`post_graph.rs:938,1200` 与 `shading/runtime.rs:393-401` 分别提供声明、23 项 executor ledger 与编译结果数量校验；生产断言读取编译图，不硬编码 31。测试定义见 `graph.rs:285-289` 与 `runtime.rs:461-469`（04 §2 / §3）；scene lib 最近完整记录为 1909/1909，RenderGraph 先前定向记录 85/85 及磁盘不足中止均为历史；2026-10-22 main 全 package 已通过 unit 122/122、doc 1 passed / 1 ignored（见 §15.2）。该历史空间阻断不影响随后独立 target 完成的三域 33/33 复验。
2. **`render_layers` 增量已补齐**：`SceneChangeKind::RenderLayers` 在 `gpu_scene/mod.rs:84-103`，transaction/mirror 接线见 `gpu_scene/transaction.rs:38-68,298-363` 与 `gpu_scene/mirror.rs:298`。
3. **注册表与设备选择已实现**：built-in catalog/roots 见 `render_fabric/builtins.rs:527-552`；设备初始化见 `prism_render_scene/src/gpu/fabric.rs:13-51`，由 `gpu/plugin.rs:25` 注册。此处证明 selection 初始化，不证明 Fabric compile/runtime 已接生产。
4. **水体相位与 draw 目标已统一**：水体 feature 通过 `render_state.rs:101-118` 映射到 `Transparent + HdrTransparent + PremultipliedAlpha + depth_write=false`；因此旧 `MaterialRenderClass::Water → Opaque` 矛盾不再存在。
5. **`overlay_opacity` 已有实际渲染消费方**：WESL `water_surface_raster.wesl:1796-1809` 从 `view.viewport.z` 读取并用于 overlay factor；CPU/WESL 契约及独立 reference 见 `water/shading/hybrid.rs:109-204`、`water_hybrid_golden.rs:378-409`。
6. **CPU/GPU Hybrid 公式已统一**：两侧均 clamp 归一化 shore/shallow 信号，`npr=max(shore, shallow)`，`pbr=1-npr`，overlay opacity clamp；契约测试见 `water_hybrid_golden.rs:469-519`。

### 1.7 历史基线与当前唯一服务权威

| # | 服务集 | 位置 | 形状 |
|---|---|---|---|
| 1 | `water::SharedBaseServices` | `water/mod.rs:439-447` | `u16` 位掩码，9 bit |
| 2 | `volumetric::SharedBaseServices` | `volumetric/mod.rs:628` | `u16` 位掩码，7 bit，位义不同 |
| 3 | `particle::{LightingServiceCaps, LightingServices}` | `particle/shading.rs:152`、`:167` | 两个 `bool` 结构体 |

第三份的文档注释已经把我方要的两层模型写出来了，原文（`particle/shading.rs:146-186`）：

> The shared lighting base services a platform makes available … These gate the optional services in `LightingServices`
>
> Determined by whether the model needs lighting at all: `Unlit` subscribes to nothing, while `PBR` and `NPR` subscribe *identically* — the shading axis never gates which shared services a lit model can consume.

即：**可用性**（`LightingServiceCaps { shadow_maps, global_illumination, ray_tracing }`）与**本帧订阅**（`LightingServices { clustered_lights, shadow_maps, global_illumination, ray_traced }`）是两层，且"lit ⇒ 与 PBR/NPR 选择无关"。

三份复制是迁移前证据。当前 `render_fabric_legacy_gate.rs:11-19,54-91` 防止旧名称回归，`FeatureCatalog` / `ServiceSet` / built-in roots 承担唯一服务声明与选择职责；仍保留的约束是“可用服务”和“本帧订阅”必须分层表达，不能重新引入子系统私有位掩码。

---

## 2. 六层装配与裁决

### 2.1 六层，其中三层已经存在

```
Main World / Assets
        ↓
Scene Delta              ← 已有：SceneChangeJournal（gpu_scene/mod.rs:102）
        ↓
Persistent GPU Scene     ← 已有：InstanceRecord（:73）+ GpuSceneReader / GpuSceneBuffers
        ↓
Frame Render Items       ← 新增：FrameRenderItem + 域负载（§6.2）
        ↓
Feature Assembly         ← 新增：FeatureCatalog 的声明 → 帧图输入（§4 / §5）
        ↓
Frame Graph              ← 已有：统一 prism_render_graph（04 §1.1）
        ├─ Water Simulation
        ├─ Particle Simulation
        ├─ Visibility
        ├─ Lighting / GI
        ├─ Shadow / Reflection
        ├─ Response Resolve
        └─ OIT / Temporal / Post
```

两个结论：

1. 六层里**只有中间两层是新增**。Scene Delta / Persistent GPU Scene / Frame Graph 都已落地，本篇不重建它们。
2. 上面 7 个分组保留为**全域装配与验收范围**。旧版 04 的“9 个声明 / 约 40 个手排”是历史快照，不能当成当前计数；当前 04 §2 / §3 记录 8 主链 + 23 后处理及 executor ledger。31 个 shading 声明并不证明 Water/Particle simulation、全域 `FrameRenderItem` 与 Fabric runtime 已全部接生产，原有 7 组覆盖要求不缩减。

### 2.2 裁决：在已有契约上扩展

| 方案 | 内容 | 裁决 |
|---|---|---|
| **(A) 扩展** | 新增 `FeatureCatalog`（§5）承载 `FeatureDefinition` / `FeatureVariant` / `PassTemplate`；`ResourceKey` / `ResourceUse` 定义在 `prism_render_architecture`（与 `frame_graph` 同 crate）；**`ExtensionManifest` 一个字段都不加** | **选 (A)** |
| (B) 新建 `prism_render_contract` | 新 crate 定义 `RenderItemHeader`、`ResourceId`、`ResponseProgram`、`FeatureDefinition` | **否**——会立刻产生第 3 个 `ResourceId`、第 3 套 `CapabilitySet`、第 2 套注册表，正是 00 §0 要消灭的"并行体系" |
| (C) 维持现状 | 保留三份服务集与两份位掩码 | **否**——位编码已经错位（§1.1），且 G3/G4 无法被机器验证 |

**对原方案的收回**：05 的前一版写"把 `reads` / `writes` / `fallback` 加进 `prism_material_pcm::ExtensionManifest`"。**该写法已撤回**——那正是把 PCM 变成总注册表的路径（§0.2）。`ExtensionManifest` 只保留材质扩展的身份与能力，特性声明归 `FeatureCatalog`。

选 (A) 的额外理由：`prism_render_architecture` **不依赖 wgpu**（`Cargo.toml` 仅在 description 中提及，`prism_render_driver_wgpu` 是**仅用于 sparse-buffer 上传测试**的 dev-dependency），`prism_material_pcm` 是**零 prism 依赖的纯叶子**。这保证了特性声明能在纯 CPU 测试里被断言——这是本仓最值钱的属性，方案 (B) 会把它稀释成两层。

---

## 3. 三个轴

### 3.1 `RenderDomain`：**不复用** `MaterialDomain`

| 项 | 内容 |
|---|---|
| 取值 | `RenderDomain { Surface, Particle, Volume, Hair, Decal }` |
| 定义位置 | `prism_render_architecture`（运行域），**不是** `prism_render_material` |

**为什么不能复用 `MaterialDomain`**（`pkg/prism_render_architecture/src/material/mod.rs:11-17` = `{ Surface, Decal, Volume, PostProcess, LightFunction }`）：

1. 它**没有** `Particle` 与 `Hair`——粒子不是材质作者域，毛发在它这里是渲染类；
2. 它有 `PostProcess` / `LightFunction`——这两个不是"渲染对象域"；
3. **材质作者域 ≠ 实际渲染对象域**，把两者等同会让"粒子该用哪个材质域"变成无解问题。

**当前实现：`MaterialRenderClass` 已完成收敛（本节保留契约与迁移理由）**

迁移前 `MaterialRenderClass` 曾含 `Volume` / `Hair` / `Water` / `Decal` 四项，与 `RenderDomain` 重叠。当前定义（`pkg/prism_render_material/src/record.rs:35-50`）只保留 blend/raster family 的七个值：

| 轴 | 保留取值 |
|---|---|
| `MaterialRenderClass`（blend / raster 族） | `Opaque`、`OpaqueTwoSided`、`Masked`、`MaskedTwoSided`、`Transmissive`、`Transparent`、`Additive`（**7 值**） |
| `RenderDomain`（渲染对象域） | `Surface`、`Particle`、`Volume`、`Hair`、`Decal` |

即 **`Volume` / `Hair` / `Decal` 由 `RenderDomain` 表达，Water 则是 `RenderDomain::Surface + WATER` 特性，不存在 `RenderDomain::Water`**。四个旧 class 值的移除已有源码证据；`MaterialRenderClass::from_u32` 当前只接受 `0..=6`（`record.rs:52-69`），对应的域/特性到管线状态映射由 `pipeline_state_for_material` 提供（`pkg/prism_render_material/src/render_state.rs:95-157`）。

**同一时刻只允许一个轴持有 `Volume` / `Hair` / `Decal`**；当前 enum 与 pipeline mapping 测试定义支撑该契约（§12 门禁 15），本轮不声称已运行测试。

### 3.1.1 移交不是删枚举值：四个值的隐含管线状态必须显式映射

迁移前这四个值**不只是标签**，而是重度接线；旧 `queue` / 分类接线只作为历史审计背景保留，不再作为当前源码定位。当前实现已建立显式映射：

```
RenderDomain  +  MaterialRenderClass（blend/raster 族）  +  FeatureSet(§4)
                            ↓
                     PipelineState
        （相位 / blend / depth / cull / OIT 路径 / 专用 shadow·prepass / render_class_bits）
```

映射表（当前契约与回归基准，由 §12 门禁 24 断言）：

| 历史对象域值 | 当前状态 | 当前证据 |
|---|---|---|
| `Water` | 不再作为 `MaterialRenderClass` 值存在；水体由 `Surface + WATER` 解析为透明 HDR 管线状态 | `render_state.rs:101-118`；水体 draw 描述见 `water/gpu/surface_pass.rs:160-177,202-215` |
| `Hair` | 由 `RenderDomain::Hair` 表达 | `render_state.rs:122-135` |
| `Volume` | 由 `RenderDomain::Volume` 表达 | `render_state.rs:136-145`；全 class 回归见 `pipeline_tests.rs:238-247` |
| `Decal` | 由 `RenderDomain::Decal` 表达 | `render_state.rs:146-155`；全 class 回归见 `pipeline_tests.rs:249-257` |

**两条硬约束**：

1. **域与 blend/raster class 必须可逆解码**。`PipelineBinClass` 用稠密的 `(domain, MaterialRenderClass)` 索引编码，`decode()` 可恢复二者（`render_fabric/pipeline.rs:203-250`）；全域、全七 class 的回归见 `pipeline_tests.rs:102-116`。
2. **不得退回手写位窗口或按声明顺序漂移身份**。当前编码由显式的 `DOMAIN_COUNT × CLASS_COUNT` 与稳定 `index()` 计算（`pipeline.rs:209-215`）；这是持续设计契约。

**当前裁定已执行**：以水体 draw 声明为准，水体落在透明相位，写入 HDR 透明目标、使用预乘 alpha 且不写深度；这由 `pipeline_state_for_material` 的 water 分支（`render_state.rs:101-118`）和 `plan_surface_plan`（`water/gpu/surface_pass.rs:202-215`）共同表达。后续只需保持两处契约一致。

### 3.2 `ResponseProgram`：不新增 enum

`ResponseProgram` 在本文只是“响应程序”的概念名，**不是新增 Rust enum，也不是旧 `Illumination` 的别名**。当前源码以程序引用与组合计划分层：

| 取值 | 来源 |
|---|---|
| `BuiltinProgram { Lit, Npr, Unlit }` | `pkg/prism_render_material/src/shading_plan.rs:21-25` |
| `ProgramRef { Builtin(BuiltinProgram), Registered(ProgramId) }` | 同文件 `:56-62`；Custom 是 `Registered`，不是第四个内建值 |
| `ShadingPlan::{Single, MaskBlend}` | 同文件 `:65-75`；两臂均持有 `ProgramRef`，见 02 §0 |

- `PBR → BuiltinProgram::Lit`、`NPR → BuiltinProgram::Npr`；`Stylized` 仍可出现在 phase / lobe 等其它语义中，不能冒充当前程序枚举值；
- **水体是 `Surface` 域，不是一种 shading frontend**；
- **粒子是 `Particle` 域，不是一套材质管线**；
- **`Hybrid` 是组合语义**，通用模型归 `ShadingPlan::MaskBlend`；当前水体使用专用 typed adapter `WaterShadingPlan`，不能据此把所有水体字段写成直接持有通用 `ShadingPlan`。

### 3.3 `ExecutionFamily`：派生量，不是新轴

```rust
pub enum ExecutionFamily {
    FusedBuiltin,
    SplitResponse,
    Composition,
}
```

唯一 enum 定义在 `pkg/prism_render_architecture/src/shader_package/execution.rs:8-12`，材质侧 `shading_plan.rs:120` 仅 re-export。执行家族由已验证的程序/组合绑定与后端执行选择确定，**不是仅由设备 capability 三个位直接映射，也不新增独立事实源**。

**一处口径纠正**：原方案同时写"`ExecutionFamily = FusedBuiltin / SplitResponse / Composition`"和"Raster / Compute / RT 是执行族"。**后者不是执行族**——它们是执行/派发方式；队列类别则由 `QueueClass { Graphics, Compute, Transfer }` 表达（`pkg/prism_render_graph/src/frame_graph/types.rs:22-29`），两者也不是一一对应。全文不得用"执行族"称呼队列或派发种类。

### 3.4 易混轴：`MaterialExecutionPath`（不得合并）

`pkg/prism_render_architecture/src/material/mod.rs:20-25` 的 `MaterialExecutionPath { FixedPbr, FixedNpr, ClosureTable, DiagnosticFallback }` 是**第三个轴**（"闭包怎么求值"），与 `ExecutionFamily` 语义不同，**禁止合并**。见 §3.5 命名坑。

### 3.5 命名坑（必须避开）

| 名称 | 实际含义 | 易混淆对象 |
|---|---|---|
| `prism_material_pcm::Region`（`namespace.rs`，4 值） | **注册槽位分区** `0..64`（Specification / Prism / CrossVendor / Studio） | 逐像素**区域掩码** |
| 逐像素区域掩码 | `MaterialFeatureFlags::REGION_ILLUMINATION`（`record.rs:125-132`） | 上面的槽位分区 |
| "响应图" | `docs/prism_ui_loom_design_zh.md:16,126` 的 **UI 反应式依赖图** | 本篇的"特性图" |
| `FeatureRegistry` | `prism_render_architecture::diagnostics::FeatureRegistry`（`diagnostics/registry.rs:46`，`BTreeMap<&'static str, FeatureStatus>`） | 本篇的 **`FeatureCatalog`** |
| `CapabilitySet` | **两份同名**：`backend/capability.rs:77`（设备）与 `pcm/capability.rs:84`（材质扩）；前者被 `Capability`（`:32`）与 `WEBGPU_BASELINE`（`:87`）配套，后者被 `ExtensionRegistry::required_caps_of` 返回 | 见下方裁定 |
| `ExecutionFamily` | 执行族（`FusedBuiltin` / `SplitResponse` / `Composition`） | `MaterialExecutionPath`、`QueueClass` |
| `QualityTier` | `pkg/prism_app/src/capability.rs:249` 的 `{ Server, Mobile, Desktop }`，是**宿主/OS 运行档**（由有无显示器派生，`:261-269`） | 本篇的**渲染质量档** `RenderQualityTier` |
| `ResourceKey` | 本篇语义资源身份（`{ owner, slot }`，§4.5） | 一个装下所有子系统资源的**中心枚举**（明确禁止，§4.5.1） |

**`CapabilitySet` 裁定**：`FeatureVariant.required_caps` 一律指 **`backend::capability::CapabilitySet`**（设备能力，它才回答"平台有没有 RT"）。PCM 的那份留在扩展协商内部，两者之间**加显式转换函数**，禁止 `as` 强转。

**`QualityTier` 裁定**：`FeatureVariant.min_quality` **不得**复用 `prism_app::capability::QualityTier`，两个理由——(1) 它的三个取值描述的是宿主形态（无头服务器 / 移动 / 桌面），与"SSR 还是 RT"无关；(2) `prism_app` 在分层上位于 `prism_render_architecture` **之上**，反向依赖会立刻破坏 §2.2 选 (A) 的全部理由。渲染质量档必须是定义在 `prism_render_architecture`（与 `CapabilitySet` 同处）的**新类型** `RenderQualityTier { Low, Medium, High, Ultra }`。

### 3.6 逐像素区域响应选择：**只覆盖二值选择，不覆盖连续混合**

`REGION_ILLUMINATION` 的机制**已经实现**：可见性栅格逐像素采样掩码并输出 Lit/NPR 二值选择子（`pkg/prism_render_scene/src/shaders/visibility_raster.wesl:50-55,131-175,247-251`），shader def 在 `pkg/prism_render_scene/src/shading/raster.rs:187`。这里的 selector 不是 `BuiltinProgram` 本身，也不是连续 mask。

**但它是一个硬选择子**：掩码把每个像素二选一送到 Lit 或 NPR 响应。

**这不足以替代连续混合**。水体 MaskBlend 仍是**连续权重混合**；当前实现位于 `water_surface_raster.wesl:1795-1809`，不是 `REGION_ILLUMINATION` 的二值选择。

**裁定**：

| 情形 | 机制 |
|---|---|
| 逐像素**二值**响应选择（材质里的区域掩码） | `REGION_ILLUMINATION`（已实现，复用） |
| 逐像素**连续**权重混合（水体岸边过渡、手绘叠加） | `ShadingPlan::MaskBlend { left, right, mask, domain }`（02 §3 / §4）+ **常驻 `CompositionPass`**（04 §4.3） |

两者**不可互替**。前一版 §3.6 写的"因此'同一表面逐像素混合两种响应'不需要新机制"**已撤回**。

---

## 4. 特性的三层声明

取代原方案的单一大 `FeatureSpec`。三个类型职责不同，**不得合并**：

```rust
/// 静态定义一个特性：它是谁、属于哪一类、有哪些完整实现。
pub struct FeatureDefinition {
    pub id: FeatureId,
    pub kind: FeatureKind,
    /// **只有子系统特性才有域**；服务与响应程序不属于任何渲染域。
    pub domain: Option<RenderDomain>,
    /// **硬需求**：本特性的**每一个**变体都必须满足的服务。
    /// 闭包计算**只看这一项**（§4.1 相 B），因此它与变体选择无关、无需递归。
    pub requires_services: ServiceSet,
    pub variants: &'static [FeatureVariant],
}

pub enum FeatureKind {
    Subsystem,
    /// **只有 `Service` 携带 `provides`**：`Subsystem` / `Response` 在类型层面
    /// 就没有"提供共享服务"的能力，它们只能通过自己的 pass `accesses` 读/写资源。
    Service { provides: &'static [ServiceOutput] },
    Response,
}

/// 同一特性的一个完整、可独立验证的实现。
pub struct FeatureVariant {
    pub id: VariantId,
    /// 确定性优先级：数值小者优先。用于 §4.1 的选择规则。
    pub rank: u16,
    /// 最低渲染质量档（设备/配置级，**不是每帧测量值**）。见 §4.1 / §3.5。
    pub min_quality: RenderQualityTier,
    pub required_caps: backend::CapabilitySet,   // 见 §3.5 裁定
    /// **软需求**：本变体**额外**需要的服务（超出特性级硬需求的部分）。
    /// 只作为该候选的**准入条件**，**绝不参与闭包计算**——否则闭包依赖变体选择，
    /// 而变体选择又依赖闭包，成为循环定义（§4.1 相 B）。
    pub requires_services_extra: ServiceSet,
    pub passes: &'static [PassTemplate],
}

/// 一个被提供的服务：语义名 + 它落在哪些帧图资源上。
/// **每个 `ServiceId` 全 catalog 只有一个提供者特性**（§4.1 相 A 第 5 条）。
pub struct ServiceOutput {
    pub service: ServiceId,
    pub resources: &'static [ResourceKey],
}

/// 真正参与帧图编译的 pass。
pub struct PassTemplate {
    pub id: PassKey,
    pub accesses: &'static [ResourceUse],
    pub queue: QueueClass,   // 唯一的队列事实源（§4.3）
}
```

**为什么需求被拆成"特性级硬需求 + 变体级软需求"**：`requires_services` 原先挂在 `FeatureVariant` 上，于是 `transitive_deps(active_roots)` **根本无法计算**——`Reflection` 的 `SSR` 变体可能需要 `HiZ`、`Probe` 变体不需要，而选哪个变体又取决于能力与画质档。闭包依赖选择、选择又依赖闭包，循环。拆开之后：

- **闭包**只沿特性级 `requires_services` 走，是**纯静态、与设备无关**的函数，在相 A 就能算出来；
- **变体的额外需求**退化为候选的**准入条件**：不满足就落到下一个 `rank`，**不反向扩张闭包**。

这样既不需要"所有变体需求完全一致"（那会让 `Probe` 变体被迫拉进 `HiZ` 的提供者，重新制造 §4.1 相 B 要消灭的"可选服务变必选"），也不需要带回溯的递归候选求解（指数搜索、且"第一个解"难以断言）。

### 4.0 `kind` 与 `domain`：服务级与响应级**不属于任何域**

前一版把 `domain: RenderDomain` 写成必填，于是"服务级"（`VirtualShadow`）与"响应级"（Lit / NPR / Custom）特性只能**伪造一个 `Surface` 或 `Volume` 域**才能注册。这与 §3.2 自己的裁定（"水体是 `Surface` 域，不是一种 shading frontend"）直接冲突。

| `kind` | `domain` | 例 | 声明档位 |
|---|---|---|---|
| `Subsystem` | `Some(_)` | `WaterSimulation`、`ParticleSimulation`、`VolumeFroxel` | §5.3 子系统级 |
| `Service` | `None` | `VirtualShadow`、`Reflection`、`TemporalResolve` | §5.3 服务级 |
| `Response` | `None` | Lit / NPR / Custom（服务声明名，不是内建程序枚举的三个值） | §5.3 响应级 |

非法组合由两个常量构造器排除：`FeatureDefinition::subsystem(id, domain, variants)` / `::service_or_response(id, kind, variants)`；§12 门禁 19 兜底断言 `kind == Subsystem ⟺ domain.is_some()`。

例（反射特性：一个服务级特性 + 三个完整实现）：

```
Reflection      kind = Service { provides: [ ServiceOutput {
                                    service: Reflection,
                                    resources: [ reflection_result ]   // ← 服务接口，变体无关
                                } ] }
                requires_services = {}            // 特性级硬需求
    ├─ SSR      rank 0  required_caps{ hzb }        extra{ HiZ }   // 软需求，仅准入
    │            passes: 写 reflection_result（屏幕空间光线步进）
    ├─ RT       rank 1  required_caps{ RayQuery }   extra{}
    │            passes: 写 reflection_result（硬件光线查询）
    └─ Probe    rank 2  required_caps{} (baseline)  extra{}
                 passes: 写 reflection_result（反射探针）
```

**关键点：变体之间的差异是"怎么算"，不是"发布成什么"。** 三个变体最终都写**同一个** `reflection_result`，因此：

- 消费方（PBR / NPR / 水体）只认服务接口 `Reflection` 与它的资源，**永远不需要知道今天选中了哪个变体**——这才是"服务"这个词的意义；
- `ServiceOutput` 可以安全地放在**特性级**（`FeatureKind::Service` 的载荷），从而在类型层面挡住 `Subsystem` / `Response`；
- §4.6.1 的"产出资源必须由提供者 pass 写入"变成一条**每变体都要满足**的检查，而不是"按选中变体变化"的检查。

`extra{ HiZ }` 表示"**这个候选**额外需要 `HiZ` 服务"：只在评估 `SSR` 候选时检查，**不会**把 `HiZ` 的提供者特性拉进闭包。因此设备没有 `HiZ` 时 `SSR` 被跳过、落到 `RT` / `Probe`，而**不需要**为"可能用到"预先付出一份 `HiZ` 提供者。

### 4.1 变体选择规则必须**确定、全定义、且拓扑稳定**

前一版列了 `variants` 却没写"多个实现同时可用时怎么选"，也没定义**谁先被求解**。规则分两相，**不得合并**——上一版把两件事都塞进"`FeatureCatalog` 编译期执行一次"，与"选择依赖 `device_caps`"直接冲突（设备能力只在设备初始化时才知道）。

**相 A —— 编译期（`FeatureCatalog` 构建期，一次，与设备无关）**：只做静态校验，**不做任何选择**：

1. **身份**：`FeatureId` / `VariantId` 显式声明且唯一，**不得由注册顺序派生**（§4.5.1）。
2. **槽位**：`(owner, slot)` 组内唯一（§4.5.1）。
3. **服务图**：由**特性级** `requires_services` 与 `FeatureKind::Service { provides }` 构成的有向图**无环**，且每个 `requires_services` 成员**存在至少一个潜在提供者**。**边的种类受分层约束**——违反任一行即构建期错误，而不是留给运行期"碰巧正确"（§12 门禁 27）：

   | 从 \ 到 | `Service` | `Subsystem` | `Response` |
   |---|---|---|---|
   | `Service` | ✅ 允许（须无环） | ❌ 禁止 | ❌ 禁止 |
   | `Subsystem` | ✅ 允许 | ❌ 禁止 | ❌ 禁止 |
   | `Response` | ✅ 允许 | ❌ 禁止 | ❌ 禁止 |

   即：**依赖只能向下指向 `Service`**；`Service` 之间可互相依赖但必须无环；`Service` 不得反向依赖任何根，`Subsystem` 与 `Response` 之间也不得互相依赖。没有这条，相 B 的分层求解会撞上"某个 `Service` 已开始解析，却依赖一个禁止作为其下游的 `Subsystem`"。
4. **资源**：每个 `ServiceOutput.resources` 成员都必须被提供者特性**每一个**变体的某条 `PassTemplate` **声明写入**（§4.6.1）——只校验"当前选中的那个变体写了"是不够的，接口必须对全部变体成立。
5. **服务提供者唯一**：每个 `ServiceId` 在全 catalog 中**恰好有一个**提供者特性。这条不是装饰——没有它，"某个已选变体恰好提供"就不可判定，门禁也无法证明结果唯一。两个特性提供同一 `ServiceId` 是构建期错误（§12 门禁 29）。
6. **`provides` 的类型边界**：`ServiceOutput` **只能**出现在 `FeatureKind::Service` 的载荷里（见 §4 的类型定义）。`Subsystem` / `Response` 在**类型层面**就无法声明 `provides`，因此 `requires_services` 的目标集合与 §4.1 的边矩阵是同一集合，矩阵可被机械执行。若将来确有"子系统提供共享语义服务"的需求，必须**新增明确的 provider 种类与仲裁规则**，不得靠"某个已选变体恰好提供"（§12 门禁 30）。

**相 B —— 设备初始化期（`FeatureSelection::resolve(active_roots, device_caps, quality)`，一次，设备局部）**：确定性求解，**不使用任何注册顺序**，且**只求解根集合的依赖闭包**：

```
resolve(active_roots, device_caps, quality):
    // 闭包只沿**特性级** requires_services（硬需求）走 ⇒ 与设备、与变体选择都无关
    closure = transitive_deps_over_feature_requires(active_roots)
    providers = unique_provider_of_service()        // 相 A 第 5 条：每个 ServiceId 唯一提供者（FeatureId）
    provides_of(f) = catalog[f].kind.provides       // provides 是**特性级**的，只在 FeatureKind::Service 上

    for feature in topo_order(closure)              // 分层 + Service 子图拓扑序
            .tie_break_by(|f| f.id):                // 并列时按 FeatureId 稳定破平局
        chosen[f] = candidates(f)
            .filter(|v| v.required_caps     ⊆ device_caps)
            .filter(|v| v.min_quality       ≤ quality)
            .filter(|v| v.requires_services_extra
                         ⊆ ⋃ { provides_of(providers[s]) | s ∈ v.requires_services_extra })
                                                        // 只作准入条件，不扩张闭包
            .min_by_key(|v| (v.rank, v.id))          // rank 小者优；id 破平局
            else 硬错误（本篇契约；00 §3 同样要求不可解析的程序显式报错）
    return FeatureSelection(chosen)
```

**每一步为什么是可计算的**：

| 量 | 依赖 | 何时确定 |
|---|---|---|
| `closure` | 只读特性级 `requires_services` | 相 A 一次算完，与设备无关 |
| `providers[s]` | 相 A 第 5 条的唯一性 | 相 A 一次算完 |
| `provides_of(providers[s])` | 只读**已解完**的 `Service` **特性**的 `kind.provides`（不是变体字段） | 拓扑序保证；`Service` 之间无环 |
| `requires_services_extra` | 是该候选的**准入条件**，不产生新的图节点 | 求解该特性时即判定，失败就换下一个 `rank` |

**没有回溯搜索**：候选被拒**不会**改变 `closure` 或 `providers`，所以求解是单遍的、线性于闭包规模，**不存在"指数候选尝试"**，也不存在"第一个解"这种难以断言的语义。若一个特性连最低 `rank` 的候选都无法准入，就是硬错误，而不是继续扩张闭包去救它（与 00 §3 的显式错误边界一致）。

**为什么"只求解闭包"是必须的**：前一版遍历 `catalog.of_kind(kind)` 的**全部**特性，等于把每个特性都变成必选项——场景没有水也要求 `WaterSimulation` 有可用变体；某个响应程序当天没被任何材质引用，也必须有 baseline；一个可选服务缺 baseline 就能让**整台设备**的初始化失败。**未被任何已激活根依赖到的特性不参与求解，也不需要可用变体。**

**`active_roots` 的来源与冻结规则**（这是与拓扑稳定契约的接缝，必须写死；00 §3 / 04 §5）：

- 来源是**静态的**：构建档案启用的子系统 + 该档案可引用到的响应程序集合，二者在设备初始化时取并集。**不得**来自每帧可见性、材质计数或任何测量值。
- 该集合在设备生命周期内**不得收缩或扩张**：运行期新出现的材质 / 程序必须落在**已声明**的根集合内；落在集合外是**硬错误**，不是"静默加一个根"——后者等于让帧图拓扑随内容变化，违反拓扑稳定约束。
- 这与 04 §4.3 / §5 要求 `CompositionPass` 常驻是同一个理由：**可流式出现的东西，必须在初始化时就进入常驻声明。**

**为什么这个顺序是良基的**：依赖边只向下指向 `Service`（相 A 第 3 条），而 `Service` 子图无环 ⇒ `closure` 的拓扑序唯一（并列由 `FeatureId` 破平局）。闭包只读**特性级**硬需求，`providers` 由相 A 第 5 条唯一确定，而 `provides_of` 读的是**特性**（`FeatureKind::Service` 载荷）而非某个变体 ⇒ `⋃ { provides_of(providers[s]) }` 在求解的每一步都**已冻结**，结果与声明顺序、注册顺序、遍历顺序**全部无关**，也**不依赖已选中的变体**。

**五条硬约束**：

1. **输入只有 `active_roots` + 设备能力 + 一个静态渲染质量档**，**绝不包含每帧测量值**（帧耗时、可见像素数、预算余额）。否则帧图拓扑会随场景变化，违反拓扑稳定约束（04 §5）。选择对应的执行与固定功能状态必须进入 03 §2 的完整管线身份；不能据此把动态质量参数重新枚举为 shader 置换（03 §2.1）。
2. **`rank` 必须全序**：同一特性内 `rank` 唯一，`(rank, id)` 构成全序 ⇒ 选择是确定性函数，可被断言（§12 门禁 22）。
3. **同时满足多个变体时不允许"同时启用多条链"**。前一版 `fallback: FallbackChain` 的写法已撤回，理由见下。
4. **求解结果整体冻结**：`FeatureSelection` 一旦解析，帧图编译器、PSO / 管线缓存、队列系统只能**读取**它，不得各自重新调用 `resolve`、重新比较 `rank` 或重新选择变体（§5.1）。
5. **不得存在隐式引擎例外**：`Lighting` / `ShadowReceiver` 等核心服务同样由特性声明提供（§4.6.1），求解公式里**不得**出现 `engine_services ∪` 这类旁路。上一版公式与 §4.6.1 的"不得开特例"自相矛盾，该旁路已删除。

例（SSR / RT / Probe 三者同时可用时，`rank` 最小且满足能力者胜出 ⇒ 永远选 `SSR`）。

### 4.2 为什么 `variants` 取代 `fallback` 链

前一版在 `FeatureSpec` 里放 `fallback: FallbackChain`。改为 `variants` 的理由：**每条回退本身就是一个完整、可验证的实现**——它有名字、有能力需求、有自己的 pass 列表，可以单独被断言"缺 X 时图中出现它"。回退链则只能被断言"链上第 k 项被选中"，无法断言实现完整性。

**但 `provides` 不在变体上**：服务接口是**特性级**的（`FeatureKind::Service { provides }`），同一服务特性的所有变体共享同一组 `ServiceOutput.resources`——变体之间的差异是**怎么算**，不是**发布成什么**。否则消费方被迫知道"选中了哪个变体"，`provides` 也就必须随变体变化、类型边界随之破裂（§4.6.1、门禁 32）。

**回退语义仍按前一版裁定**（保留）：

| 语义 | 含义 | 适用 | 验收方式 |
|---|---|---|---|
| **图级** | 缺能力 → 帧图**改选另一个 `FeatureVariant`** | `required_caps` 不可满足时的降级 | 断言"缺 X 时图中不出现 variant P 的 pass、出现 variant Q 的 pass" |
| **像素级** | shader 内 `if` 判断后走另一条采样路径 | 运行期 miss | 断言"两条路径都可达 + 选择子是数据"（00 §4 区域选择两臂门禁语义） |

**裁定：能力缺失走图级（换 variant），运行期 miss 走像素级。** 水体现有的三级反射回退（`water/shading/mod.rs:31-43`，`SSR → RT → Probe`）是**像素级**，保持不动。

### 4.3 `PassTemplate` 只存一份访问列表；`queue` 只有一个事实源

两处都改：

**(a) 访问列表合成一份。** 前一版写 `reads` + `writes` **两个列表**。已改为一份 `accesses`：`AccessKind` 已经能判定读写——

```rust
impl AccessKind {
    pub const fn writes(self) -> bool {
        matches!(self, StorageWrite | ColorAttachment | DepthAttachment | TransferWrite)
    }
}
```

（唯一类型定义为 `pkg/prism_render_graph/src/frame_graph/types.rs`，11 个取值：`SampledRead` / `StorageRead` / `StorageWrite` / `ColorAttachment` / `DepthAttachment` / `IndirectRead` / `TransferRead` / `TransferWrite` / `UniformRead` / `IndexRead` / `DepthAttachmentRead`；architecture 仅 re-export。buffer/texture use、texture namespace、attachment 与写入判定由该类型集中映射。）存两份列表是同一信息写两遍，必然漂移。

**(b) `queue` 只留在 `PassTemplate`（方案 A）。** 前一版同时给了 `PassTemplate.queue` 与 `ResourceUse.queue`，而当前帧图模型是**一个 pass 一个队列**（`PassDescriptor.queue`，`prism_render_graph/src/frame_graph/types.rs:103-112`）——两个事实源必然矛盾。裁决：

| 方案 | 内容 | 裁决 |
|---|---|---|
| **A：一个 pass 只能属于一个队列** | 删除 `ResourceUse.queue`；跨队列操作拆成两个 pass | **选 A** |
| B：一个 pass 可含多队列操作 | 需重构 `PassDescriptor`、`Barrier`、提交模型 | **否**——为未来的跨队列能力提前引入复杂度 |

**放弃的收益要记账**：选 A 放弃的是“同一 pass 内多队列访问及其转移”的表达能力，**不是**否认现有 `queue_transfer` 推导。`Barrier.queue_transfer`（`prism_render_graph/src/frame_graph/types.rs:127-139`）仍由 **pass 之间**的逻辑队列差异派生；它不单独承诺原生多队列提交与 fence。**这是主动保留的边界，不是遗漏**，§10 风险表相应保留。

### 4.4 `ResourceUse`：完整的访问描述

```rust
pub struct ResourceUse {
    pub key: ResourceKey,
    pub access: AccessKind,
    pub scope: ResourceScope,
    pub frame_lag: Option<FrameLag>,  // FrameLag(NonZeroU32)，正历史帧距离
}
```

**为什么必须比 `reads: ResourceSet` 更精确**——当前实现已经把两项语义纳入声明与编译校验：

| 字段 | 当前实现 | 校验与语义 |
|---|---|---|
| `scope`（子资源） | `ResourceScope::Whole` 是当前唯一合法取值 | 类型层面明确声明当前采用整资源粒度；mip/layer 语义不属于本契约 |
| `frame_lag` | `FrameLag(NonZeroU32)` 显式表示正历史帧距离 | `render_fabric::compile` 将其映射到持久资源的历史帧，并禁止建立同帧依赖边；`HistoryBeforeFrameZero`、历史绑定和无环测试覆盖该路径 |

**这两个字段不是装饰性字段**：`scope` 的受限取值与 `frame_lag` 的历史帧编译语义均有类型、实现与测试定义（`render_fabric/types.rs:110-130`、`compile.rs`、`compile_tests.rs`）。这证明 CPU 编译模型，不代表生产 history 已由该编译结果分配、发布或导入。当前帧图明确采用整资源访问模型；若未来引入子资源，必须同步扩展资源版本、barrier 和 alias 校验，不能只增加一个枚举值。

（**实现说明**：前一版此处第三项是 `queue`，已按 §4.3 方案 A 删除；队列归属由 `PassTemplate` 单一声明。）

### 4.5 `ResourceKey` 与 `ResourceId` 必须分开（本节是全篇最重要的修改）

| | `ResourceKey` | `ResourceId` |
|---|---|---|
| 性质 | **稳定语义身份** | **当前帧图的局部编号** |
| 形态 | **命名空间键**（见下），如 `water::RESOURCE_FOAM` | `pub struct ResourceId(pub u32)`（`prism_render_graph/src/frame_graph/types.rs:8-11`） |
| 生命周期 | 永久，跨帧、跨子图不变 | 只在一次编译/执行内有效 |
| 谁持有 | `FeatureDefinition` / `FeatureVariant` / `PassTemplate` 的**静态表** | 编译器、`Barrier`、`ResourceVersion` |

```
ResourceKey  ──▶ FrameGraphBuilder ──▶ ResourceId ──▶ Barrier / Alias / Schedule
   （声明）              （编译）            （本帧）
```

**禁止静态特性声明保存 `frame_graph::ResourceId`。** 理由：`ResourceId` 是当前图的局部编号，把它写进 `&'static` 表意味着声明与某一次编译绑定；换个视图/换条路径就会指错资源，而且这种错误在类型上检查不出来。

**迁移前**只有诊断字符串 `ResourceDescriptor.name: Cow<'static, str>` 作为语义线索；当前该字段仍在 `prism_render_graph/src/frame_graph/types.rs:44-53`，但已有 `render_fabric/types.rs` 的 namespaced `ResourceKey`，不能再写“今天没有”。稳定键把声明身份类型化，字符串只负责诊断；生产路径是否真正消费稳定键另见 §1.3。

#### 4.5.1 `ResourceKey` 必须是命名空间键，**不是全局大枚举**

前一版倾向写成一个中心枚举：

```rust
// ✗ 禁止
pub enum ResourceKey { SceneDepth, WaterFoam, ParticleAliveList, ReflectionResult, HairDensity, … }
```

**问题**：水体、粒子、毛发、布料、体积各自新增资源都要改这个中心枚举 ⇒ `prism_render_architecture` 变成新的全局耦合中心，且它与 §0.2 要建立的"四个权威各管一件事"直接冲突（帧图 crate 不该知道水体有哪些资源）。

**正确形式**——键由 `(owner, slot)` 组成：

```rust
pub enum ResourceOwner {
    Engine,                 // 引擎核心：跨子系统的导入/持久资源
    Feature(FeatureId),     // 某个子系统/服务特性拥有
}

pub struct ResourceKey {
    owner: ResourceOwner,
    slot: u16,
}

impl ResourceKey {
    pub const fn engine(slot: u16) -> Self;
    pub const fn of(owner: FeatureId, slot: u16) -> Self;
}
```

各子系统在自己的模块里声明**自己的**键，新增子系统**不需要修改任何中央枚举**：

```rust
// pkg/prism_render_architecture/src/water/resources.rs
pub const RESOURCE_FOAM: ResourceKey = ResourceKey::of(WATER_SURFACE, 0);

// pkg/prism_render_architecture/src/particle/resources.rs
pub const RESOURCE_ALIVE_LIST: ResourceKey = ResourceKey::of(PARTICLE_STREAM, 0);

// 引擎核心
pub const SCENE_DEPTH: ResourceKey = ResourceKey::engine(0);
```

仍然是**类型化、稳定、可验证**的语义 ID（拼错编译不过、可 `Hash`、可进 `&'static` 表），但把"新增资源"的修改面收敛到拥有它的模块。

**ID 与槽位的分配规则**：`FeatureId` / `VariantId` / `(owner, slot)` 一律**在拥有它的模块里显式声明为常量**（见上例的 `ResourceKey::of(WATER_SURFACE, 0)`），**禁止**按声明顺序、注册顺序或遍历顺序自动发号——否则插入一个资源就会改变其后所有 ID，同时破坏 PSO 缓存、ABI 与持久化身份。`FeatureCatalog` 只做**唯一性与冲突校验**，**不负责发号**（§12 门禁 26）。

PCM 的 `ExtensionRegistry::allocate_slot`（`pcm/registry.rs:191`）是**运行期注册**场景的做法，与本篇的**静态声明**不同，不得照搬；`pcm::namespace::{Region, SLOTS_PER_REGION}`（`namespace.rs:89`、`:103`）提供的是分区**容量**，而本篇的 slot 是作者写下的稳定常量。**`Region` 是"槽位分区"、与逐像素区域掩码无关**（§3.5）。

**门禁**（§12 门禁 16 / 20）：静态声明类型里不得出现 `frame_graph::ResourceId`；每个 `(owner, slot)` 组内唯一，且所有被引用的 `ResourceKey` 都有声明的所有者。

### 4.6 服务依赖与底层资源依赖分两层

**不要把 `reads` 闭包直接当作"PBR 和 NPR 共享高级基底"的证明。** 分两层：

```
ServiceRequirement（语义服务）
    Lighting / ShadowReceiver / Reflection / TemporalReprojection / Transparency
                    │
                    ▼  由服务实现展开
底层帧图资源
    clustered_lights / shadow_pages / ssr_result / history_color
```

即 `NPR requires: Lighting + ShadowReceiver + Reflection + TemporalReprojection`，而**底层具体读到哪些资源是各前端自己的实现细节**。PBR 与 NPR 共享的是**语义服务**（§1.7 第 3 份服务集的注释已经明说这点），不必强迫它们读完全相同的底层资源。

**仍然可机械验证**：`NPR 的 requires_services ⊇ PBR 的 requires_services`。

**历史实现仅平凡满足**：两份 `shared_base()` 无条件返回 `ALL`、`lighting_services()` 对任何 lit 模型返回同一集合，无法证明实际服务消费。当前声明已由 `ServiceSet` 表达，`NPR ⊇ PBR` 作为可证伪的单调不变式保留（§12 门禁 4）；声明关系仍不能替代各域真实生产服务与资源消费的验证。

**统一载体**：`ServiceSet` 是唯一的一份，`water::SharedBaseServices`、`volumetric::SharedBaseServices`、`particle::{LightingServiceCaps, LightingServices}` 三份全部删除（§7）。其中"可用性 vs 订阅"的两层**必须保留**：`FeatureVariant.required_caps`（设备可用性）↔ 原 `LightingServiceCaps`；`FeatureDefinition.requires_services` + `FeatureVariant.requires_services_extra` / `FeatureKind::Service::provides`（本帧订阅与产出）↔ 原 `LightingServices`。

#### 4.6.1 服务需求必须有提供者：`ServiceOutput`

只有 `requires_services` 与 `produces_services` 两个集合时，门禁只能验证**资源访问**（`feature_accesses_are_produced_by_the_graph`），**验证不了"服务语义是否有提供者"**——因为"谁提供 `Lighting` / `ShadowReceiver` / `Reflection`"没有类型可查，图编译器只知道"某个变体声称提供了 Reflection"，不知道它落在哪些资源上。

因此产出侧带资源：

```rust
pub struct ServiceOutput {
    pub service: ServiceId,
    pub resources: &'static [ResourceKey],
}
```

于是有了一条**真正可证伪**、且**结果唯一**的验证：

```
∀ 已选特性 f, ∀ s ∈ f.requires_services:
    ∃ 已选特性 p = providers[s]: s ∈ { o.service | o ∈ p.kind.provides }
∀ 已选变体 v, ∀ s ∈ v.requires_services_extra:
    ∃ 已选特性 p = providers[s]: s ∈ { o.service | o ∈ p.kind.provides }
```

**两处细节**：

- **`providers[s]` 是唯一映射**（相 A 第 5 条）。没有这条，"某个已选变体恰好提供"就不可判定——`provides` 可能同时被两个特性声明，而门禁无法证明结果唯一。因此"同一 `ServiceId` 有两个提供者"是**构建期错误**（§12 门禁 29）。
- **`provides` 只存在于 `FeatureKind::Service`**。`Subsystem` / `Response` 在类型层面没有这个字段，所以上述断言的目标集合与 §4.1 的边矩阵是同一集合，矩阵可被机械执行（§12 门禁 30）。若将来确实需要子系统提供共享语义服务，必须**新增 provider 种类与仲裁规则**，不得靠"某个已选变体恰好提供"。

**引擎核心服务也必须走同一条路**（不得开特例）：`Lighting` / `ShadowReceiver` 由引擎侧的 `FeatureKind::Service` 声明提供（`ResourceOwner::Engine` + 一个 `CoreLighting` 服务级特性），这样规则里**没有例外分支**，也就能被一行断言完全覆盖（§12 门禁 21）。求解公式里因此不再出现 `engine_services ∪`（§4.1 约束 5）。

**但"服务有人提供"还不够**：门禁 21 只证明存在提供者，**证明不了** `ServiceOutput.resources` 真的由该提供者的 pass 写入。不补这一条，`resources` 会退化成纯声明标签——与 `bound_base` 同型的失败（§1.3）。因此再加一条**机械约束**，且必须覆盖提供者特性的**全部变体**（不能只查当前选中的那个）：

```
∀ 服务特性 f（已选）, ∀ 变体 v ∈ f.variants,
∀ 输出 o ∈ f.kind.provides, ∀ 资源 r ∈ o.resources:
    ∃ p ∈ v.passes: r ∈ writes(p.accesses)
```

即 `f.kind.provides` 的每个资源都 ⊆ **该特性每一个变体**自身 pass 的**写入集**（由 `AccessKind::writes()` 判定，§4.3），且同一 `(owner, slot)` 不允许多条 pass 无区分地写同一资源（写者必须可归一到唯一 `ResourceVersion`）。**"每个变体"是必要的**：只校验选中变体会让"当前恰好能用"掩盖"其它变体没实现该接口"，而设备一旦换能力就会静默选到一个不写该资源的变体（§12 门禁 25、32）。这条与门禁 1 形成上下闭合：门禁 1 保证"被写的资源有消费者声明的边"，本条保证"声明的服务产出真的有 pass 在写"。

### 4.7 与 02「五类契约」的边界（不得互替）

| 契约 | 回答的问题 | 归属 |
|---|---|---|
| 02 的五类契约（`ResponseManifest`） | 某个**响应**用到哪些**字段/瓣** | 02 §1 |
| 本篇的 `FeatureVariant` / `PassTemplate` | 某个**特性**用哪些**服务**、读/写哪些**帧图资源** | 05 §4 |

一个面向材质参数与 closure，一个面向服务、帧图资源与设备能力。**禁止**用 `requires_services` / `accesses` 重表瓣集；不新增一层 Lobe/Channel IR 的原始约束在本篇继续有效，与 00 §3 的 closure / ABI 权威边界一致，不再引用已不存在的旧 §8。

### 4.8 与 `shader_package::PermutationSpace` 的关系（必须定义，否则出现第三套变体系统）

`shader_package::permutation`（`PermutationSpace::new(axis_option_counts)` / `index_of` / `key_from_index` / `PermutationSelector::select().finish()`）是**编译期**的混合基数 shader 置换索引；`FeatureVariant` 是**运行时**的实现选择。两者不是一回事，但必然相连。

**设计契约**：`FeatureVariant` 对应的 shader 绑定必须**引用**既有 `PermutationSpace`（或绑定 selector），**不得自己再枚举 shader 变体**。这样“某 variant 需要哪些置换坐标”只由 03 §2 的置换空间回答，保留原始“变体数 = 注册模块数”的单一事实源约束；不把旧等式误标为当前 03 §3.1 的热端内容。当前 `FeatureSelection → FrozenShaderSelection → checked PermutationIndex → PipelineKey / PsoCacheKey / warm set` 的生产消费 seam 已闭合，并拒绝 package map、variant、space/key、pass descriptor 与 stale binding 不一致；见 03 §2.2、§6。仍未完成的是这套冻结映射在统一全场景 `SceneFabric` 每帧装配、真实 driver binary 和跨适配器范围内的端到端覆盖。

---

## 5. `FeatureCatalog`：新建，但不是总注册表

### 5.1 边界

| | 内容 |
|---|---|
| **名字** | `FeatureCatalog`（**不是** `FeatureRegistry`——该名字已被 `diagnostics::FeatureRegistry` 占用，§0.1 / §3.5） |
| **装什么** | `FeatureDefinition` / `FeatureVariant` / `PassTemplate` 三层声明 |
| **不装什么** | 材质扩展（→ `prism_material_pcm::ExtensionRegistry`）、材质参数与瓣（→ 01/02）、设备能力协商（→ `backend::capability`）、本帧资源（→ `frame_graph`） |
| **唯一权威** | **不是"只有帧图编译器能用"**，而是"只有它能**声明**"：帧图编译器、PSO / 管线缓存、队列系统**必须**消费同一份**已解析的 `FeatureSelection`**，且**不得各自重新选择变体**（§4.1 相 B 约束 4、§4.8）；这是设计契约，不是所有生产消费者已接入的结论 |
| **形态** | 静态（`&'static` 表），**非运行期可变结构**（§11 禁止 17） |
| **编译期职责（相 A）** | 身份与槽位唯一性校验（§4.5.1）、服务依赖图无环 + 潜在提供者校验 + **边种类矩阵**（§4.1 相 A 第 3 条）——**不做任何选择** |
| **设备初始化职责（相 B）** | 一次 `FeatureSelection::resolve(active_roots, device_caps, quality)`（§4.1），只求解 `active_roots` 的依赖闭包，结果整体冻结后交给消费者 |

### 5.2 PCM 保持材质扩展权威（**一个字段都不加**）

`prism_material_pcm::ExtensionRegistry`（`registry.rs:79`）今天的能力边界，全部是材质扩展自身的关注点：

| 已有 API | 位置 | 职责 |
|---|---|---|
| `register` / `register_new` / `register_identity_only` | `registry.rs:116` / `:152` / `:174` | 注册扩展身份 |
| `allocate_slot(origin: Namespace)` | `:191` | 命名空间槽位分配 |
| `promote` | `:224` | 稳定性提升状态机 |
| `manifest_at(slot)` / `manifest_by_uid(uid)` / `slot_of_uid(uid)` | `:255` / `:260` / `:265` | 按槽位/uid 查清单 |
| `identity_only()` | `:291` | 仅身份（未载入实现）的扩展 |
| `required_caps_of(slots)` / `required_caps_of_uids` | `:297` / `:315` | 聚合扩展所需的**材质扩能力** |

**以上职责全部保留、全部不扩张。** 特别是：`ExtensionManifest` **不新增** `reads` / `writes` / `passes` / `services` 任何一个字段。跨系统查询"某特性需要什么"一律走 `FeatureCatalog`。

### 5.3 声明的三个档位

| 档位 | `kind` | 谁声明 | 例子 |
|---|---|---|---|
| 子系统级 | `Subsystem`（`domain: Some`） | 每个子系统一条 `FeatureDefinition` | WaterSimulation、ParticleSimulation、VolumeFroxel |
| 服务级 | `Service`（`domain: None`） | 每个共享服务一条 | VirtualShadow、SSR、RayTracedReflection、TemporalResolve |
| 响应级 | `Response`（`domain: None`） | 每个响应程序一条 | built-in feature roots 当前为 Lit / NPR / Custom；它们不是 `BuiltinProgram` 枚举本身，也不等于用户程序注册表 |

旧“Lit 不注册”不能再描述当前实现：内建程序不伪装成用户 `ProgramId`，但仍通过 `ProgramRef::Builtin` 在 `ResponseRegistry::register` 中接受契约与编译绑定校验（`prism_render_material/src/shading_plan.rs:932-938,1015-1036`；02 §0 / §5）。`FeatureCatalog` 不取代响应 registry，更不接管材质 PCM 权威。

**共享基底集合是派生的**：某子系统"消费的全部服务"= 它的 `requires_services` 闭包，**不是**一个手写的 `ALL`。这才是 G3/G4 的机器可读版本。

---

## 6. GPU Scene 与本帧资源分开

### 6.1 持久层**已经是对的**

`pkg/prism_render_architecture/src/gpu_scene/mod.rs:73-81`：

```rust
pub struct InstanceRecord {
    pub current_transform: SceneTransform,
    pub previous_transform: SceneTransform,
    pub bounds: SceneBounds,
    pub geometry: GeometryHandle,
    pub material: SceneMaterialHandle,
    pub render_layers: u32,
    pub flags: u32,
}
```

它**恰好就是**方案中的 `PersistentInstance`（`scene_handle` 即 arena 键），而且：是 `Copy`、含 `previous_transform`（运动矢量所需）、**不含任何本帧 payload**。

**所以 §6 这一节的定位是"防止做错"，不是修缺陷。** 两条硬约束：

1. **禁止**往 `InstanceRecord` 加 `history_key` / `motion_source` / 任何 payload 字段——那会破坏 `Copy`，并把本帧 transient 数据塞进持久记录（正是方案要避免的污染）。
2. **禁止**加第二个 `stable_id`：身份已经是 `SceneHandle` / `GenerationalHandle`（`abi/mod.rs:13-30`），绑定失效也靠它（`pkg/prism_render_scene/src/buffers/bindings.rs:22` 的 `buffer_ids: Option<[BufferId; 4]>`）。

### 6.2 `FrameRenderItem`（新增）

```rust
/// 本帧的渲染对象。逐帧重建，不持久化。
pub struct FrameRenderItem {
    pub scene_handle: SceneHandle,      // 指回持久层，不复制持久数据
    pub domain: RenderDomain,           // §3.1
    pub history_key: u32,               // 时序历史的稳定键
    pub motion_source: MotionSource,    // 运动矢量来源
    pub payload: DomainPayload,         // 见 §6.3
}
```

### 6.3 域负载：**资源负载与值负载必须分开**

负载按域分开，**不塞进一个大结构**。但"每个负载字段都必须是帧图资源"是错的：`bounds`（一个包围盒）、`phase_params`（几个浮点参数）这类字段根本不值得各占一条 GPU buffer / texture，强行要求 `lifetime` / `size` / `alignment` 只会为了过门禁而**制造大量伪资源**。负载是二者的和类型：

```rust
pub enum DomainPayload {
    Resource(ResourcePayload),   // 真正进 GPU pass 的 buffer / texture
    Value(ValuePayload),         // 定宽、不可变的**内联**值（CPU 侧常量）
}

pub struct ResourcePayload {
    pub resource: ResourceKey,          // §4.5
    pub lifetime: ResourceLifetime,     // 复用 frame_graph 的 Imported / Persistent / Transient
    pub size: u64,
    pub alignment: u32,
    pub role: PayloadRole,              // 该资源在本域中的语义角色
}

/// **只能**是定宽、不可变的 CPU 侧内联值。不得持有 buffer / texture，
/// **也不得"指向"某个统一参数块**——那会把一个 GPU 资源伪装成 value 绕过校验。
pub struct ValuePayload {
    pub value: PayloadValue,            // 定宽、不可变、可 Copy
    pub role: PayloadRole,
}
```

**统一参数块怎么办**：若多个对象共享同一份参数缓冲（uniform / storage block），它本身就是**一个 GPU 资源**，必须走 `ResourcePayload`（或一个显式的 `ParameterBlock` 资源键），**不能**由 `ValuePayload` "指向"。`Value` 支的语义是"这个值直接内联在负载里"，**不是**"这个值在别处"。

**判据（字段该进哪一支）**：只有当该字段确实作为 buffer / texture 被某条 pass 的 `accesses` 引用时，它才**必须**是 `Resource`；否则走 `Value`。"方便一起传下去"不是理由——把参数塞进资源要同时付出绑定、生命周期与验证三重成本。

| 负载 | `Resource` 支（带 `lifetime`） | `Value` 支 |
|---|---|---|
| `WaterSurfacePayload`（`RenderDomain::Surface`） | `displacement`、`normal`、`foam`、`velocity`、`waterline`（`Transient`）、`depth`（`Imported`） | 域参数（样式 / 混合权重等） |
| `ParticleStreamPayload`（`RenderDomain::Particle`） | `particle_buffer`、`alive_list`、`indirect_args`（`Transient`） | `bounds`（**整体包围盒值**；若确为逐粒子 buffer 则归 `Resource` 支） |
| `VolumePayload`（`RenderDomain::Volume`） | `density`、`velocity_field`、`emission`（`Transient`） | `phase_params` |

**门禁**：`Resource` 支的每个字段必须有 `lifetime` / `size` / `alignment`，且其 `resource` 必须在图中被引用；`Value` 支必须是**定宽、不可变**的值，**不得**成为"逃过资源校验的旁路"（§12 门禁 6）。前者防"payload 退化成无界垃圾桶"，后者防"把 buffer 藏进 value 让门禁形同虚设"。

### 6.4 `render_layers` 的 delta 接线

`InstanceRecord.render_layers`（`:79`）已经通过独立的 `SceneChangeKind::RenderLayers`（`:84-92`）接入增量日志、transaction 和 mirror。它不并入 `Flags`：`Flags` 保持独立语义，因此变更原因可判定、上传范围可收敛，且不会把无关字段合并到同一 dirty path。

**持续门禁**：`InstanceRecord` 的每个持久字段必须有对应 `SceneChangeKind` 变体；当前可核实 `render_layers` 的 transaction 与 mirror 接线及测试定义（`gpu_scene/tests/transaction.rs:121-175`、`tests/mirror.rs:101-125`）。完整字段覆盖与稳定上传运行证据按 §12 门禁 17 验收；本轮未单独运行该 architecture package gate。

---

## 7. 保留 / 替换 / 删除

### 保留

| 项 | 位置 |
|---|---|
| GPU Scene 稳定句柄与增量上传 | `gpu_scene/{allocator,transaction,upload,mirror}.rs` |
| `SceneChangeKind` / `SceneChange` / `SceneChangeJournal`（Scene Delta 层） | `gpu_scene/mod.rs:84-104` |
| `MaterialResolveBins` 的分箱思想 | `pkg/prism_render_architecture/src/material/resolve.rs` |
| `backend::CapabilitySet`（设备能力，并被裁定为 `required_caps` 的所指） | `pkg/prism_render_architecture/src/backend/capability.rs` |
| PCM 的**全部**既有职责与 API | `prism_material_pcm/src/registry.rs`（§5.2） |
| 三份服务集的**语义**（作为统一 `ServiceSet` 的合并输入） | §1.7 三处 |
| `CouplingPlan` / `WaterSprayBurst` / `SprayPublishStats` | `water/coupling.rs:68`、`water/spray_bridge.rs:50`、`:81` |
| 水体的 Extract / Prepare / Queue | `water/pipeline.rs`、`water/gpu/` |
| 粒子的 CPU 侧纯函数与能力分类 | `pkg/prism_render_architecture/src/particle/` |
| 帧图的资源访问与队列依赖模型 | `prism_render_graph`；architecture 兼容 re-export（04 §1.1） |
| `PermutationSpace` / `PermutationKey` / `PermutationSelector` | `shader_package/permutation.rs` |
| 逐像素区域响应选择（**二值选择**） | `MaterialFeatureFlags::REGION_ILLUMINATION`（§3.6） |
| 水体三级反射回退（`SSR → RT → Probe`，**像素级**） | `water/shading/mod.rs:31-43`（§4.2） |

### 替换

| 旧 | 新 |
|---|---|
| `FeatureSpec`（单一大结构 + `fallback` 链） | `FeatureDefinition` / `FeatureVariant` / `PassTemplate` 三层（§4） |
| `water::SharedBaseServices` / `volumetric::SharedBaseServices` / `particle::{LightingServiceCaps, LightingServices}` | 一份 `ServiceSet`（§4.6） |
| `FeatureVariant.produces_services: ServiceSet` | `FeatureKind::Service { provides: &'static [ServiceOutput] }`（**特性级**，所有变体共享；§4.6.1） |
| `water::ShadingFrontend` / `volumetric::ShadingFrontend` | `RenderDomain`（域）+ `ProgramRef` / `BuiltinProgram`（响应身份）+ `ShadingPlan`（组合）；水体保留专用 `WaterShadingPlan` adapter（§3.2） |
| `MaterialRenderClass` 的 `Volume` / `Hair` / `Water` / `Decal` | Volume / Hair / Decal 归 `RenderDomain`，Water 归 `Surface + WATER`；由 `PipelineState` / `PipelineBinClass` 显式映射（§3.1.1 / §7.2） |
| `ResourceId` 作为声明身份 | `ResourceKey`（**命名空间键**，§4.5 / §4.5.1） |
| 子系统直接判断 GI / VSM / RT / OIT | 声明 `requires_services` / `required_caps`，由帧图解析 |
| 文档层"四档子系统注册表" | `FeatureCatalog`（§5），或删除该文档节（`prism_material_pipeline_design_zh.md:618`） |
| 以手填顺序代替资源依赖 | 由 `accesses` 推导 hazard / barrier；当前显式语义前置边仍存在，不得绕过资源图，自动推导与历史等价要求仍保留（§1.5 / §12 门禁 2） |

### 已完成收口（原删除项）

| 已收口项 | 当前实现证据 |
|---|---|
| 两份 `SharedBaseServices`、两份 `ShadingFrontend`、粒子独立 lighting service 集 | `render_fabric_legacy_gate.rs:11-19,54-91` 对旧符号做静态禁止回归；当前服务闭包由 `render_fabric/builtins.rs` 与 `ServiceSet` 提供 |
| `MaterialRenderClass::{Volume, Hair, Water, Decal}` | 当前 enum 仅有七个 blend/raster 值（`prism_render_material/src/record.rs:41-50`；Render Fabric 轴见 `render_fabric/pipeline.rs:5-17`） |
| `bound_base` / `shared_base` 及 `WaterPrepare.shared_base` | 同一 legacy gate 的源码扫描覆盖 `bound_base`、`shared_base`；当前水体入口与服务接线不再依赖它们 |
| 旧 `water_surface_fs_hybrid` 入口 | 当前 WESL 入口只有 `water_surface_fs_pbr` / `_npr` / `_custom` / `_mask_blend`（`water_surface_raster.wesl:1770-1809`）；选择逻辑见 `surface_pass.rs:160-177` |

> **当前说明**：`water_surface_fs_hybrid` 已不在源码中，但它不是被 `REGION_ILLUMINATION` 这个二值 selector“替代”的；连续 Hybrid 行为由 `ShadingPlan::MaskBlend` + 常驻 `CompositionPass` 承载，见 §7.1。

**明确不删**：`prism_material_pcm::ExtensionRegistry` 及其全部 API —— 它是材质扩展的权威，本篇只阻止它扩张（§0.2 / §5.2）。

**原始验收顺序与当前证据**见 §9；S4b 已有公式、CPU golden 与冻结 legacy GPU 输出证据，S5 已有删除、冻结 selection mapping、ABI/cache 定向回归，但完整 Scene/Fabric/TAA、跨适配器、真实 driver binary 与“先护栏后删除”的历史时序不能据此宣称完成，门禁 23 原始要求保留。

### 7.1 水体 Hybrid 的当前实现：`MaskBlend` + `water_surface_fs_mask_blend`

#### 历史差异与当前收口

| | CPU Hybrid 契约 | 当前 WESL MaskBlend |
|---|---|---|
| 位置 | `pkg/prism_render_architecture/src/water/shading/hybrid.rs:109-204,226-247` | `pkg/prism_render_scene/src/shaders/water_surface_raster.wesl:1666-1809` |
| 权重 | 先把 shore/depth 转为归一化信号，再 `npr=max(shore, shallow)`、`pbr=1-npr`；`overlay_opacity` clamp | `water_hybrid_weight_contract` 对两个归一化信号执行同一公式，并 clamp overlay |
| 输出 | `compose_mask_blend` 先混合 PBR/NPR，再用 reactive overlay 混合 NPR | `water_surface_fs_mask_blend` 写入 left/right/controls/coverage，resident Composition 完成同一两阶段组合 |
| 正交项 | `overlay_opacity` clamp 后参与 `overlay_factor` | 从 `view.viewport.z` 进入 WESL，经过 `water_hybrid_overlay_factor` 消费 |

`hybrid.rs:11` 自述：`The blend weights always sum to one` —— 这是**连续权重混合**，不是二值选择。

#### 裁定

1. **替代物是 `ShadingPlan::MaskBlend` + 常驻 `CompositionPass`，不是 `REGION_ILLUMINATION`**。连续 mask 与二值 selector 仍是两种不同契约。
2. **门禁 23 已有 CPU 测试定义**：`water_hybrid_golden.rs:276-466` 覆盖深度、岸边、泡沫、焦散、反射、透明/overlay、motion vector、reactive mask，并以独立 legacy reference 对比 `compose_mask_blend`；`:447-466` 还覆盖组合网格。它们不是 GPU 像素输出测试，也不证明旧入口删除前护栏已实际运行。
3. **旧入口已收口**：当前水体 `MaskBlend` 直接选择 `water_surface_fs_mask_blend`（`surface_pass.rs:167-177`），旧 `water_surface_fs_hybrid` 不在当前 WESL 入口集合中；`water_hybrid_golden.rs:607-613` 固定检查当前四入口名称。
4. **`overlay_opacity` 已接入 WESL**：`water_surface_raster.wesl:1800-1808` 从 `view.viewport.z` 读取并写入组合控制；CPU/WESL 的权重与 overlay 源码契约已统一。这是参数消费与公式实现证据，不是 GPU 输出等价运行结果。
5. **GPU 证据边界**：`water_hybrid_golden.rs:688-706` 自身只创建 raster、composition、HDR shader modules；真正的输出 readback 与冻结旧 shader closure 对比在 `water_hybrid_gpu.rs`。2026-10-20 Apple M2 / Metal 定向复跑结果见 §15.2；它是当前真实运行记录，但仍不能追证旧入口删除前护栏已经运行。
6. **最新局部输出证据**：`pkg/prism_render_scene/tests/water_hybrid_gpu.rs` 为 **19/19**：10 项当前 production output、8 项冻结 legacy full-frame parity、1 项比较器测试，使用生产 WESL、真实 render/composition/export stages 和 output readback。它证明该 16×16 fixture 的冻结输出等价，但不能扩大为完整 Scene/Fabric/TAA 帧 parity、所有设备覆盖、删除前护栏历史时序或全部门禁验收。

**结论**：当前 scene 水体 surface → resident Composition / HDR 链路已有 `MaskBlend` 语义的生产源码接线，水体用 `WaterShadingPlan` adapter，当前入口为 `water_surface_fs_mask_blend`，旧 `_hybrid` 已删除；水体 compute 与 resident composition 也各有局部 Fabric production slice。两条切片仍不能证明统一全场景 `DeviceFabric::compile / prepare_runtime` 闭环，也不能把 S4b 与完整门禁 23 判为全部验收通过。

### 7.2 `MaterialRenderClass` 四值迁移：已实现子项与持续契约

`Volume` / `Hair` / `Water` / `Decal` 的移除已完成；以下条件作为持续回归契约保留：

1. **域与 class 在 `PipelineBinClass` 中仍可逆恢复**（`render_fabric/pipeline.rs:203-250`），而不是依赖旧 `queue.rs` 的位字段解码。
2. **域的编码走稳定稠密索引**，不得新开手写位窗口或按注册顺序重新编号（`pipeline.rs:209-215`；回归见 `pipeline_tests.rs:102-116`）。
3. **水体相位与目标已统一**：`Surface + WATER` 固定为 `Transparent + HdrTransparent + PremultipliedAlpha + depth_write=false`（`render_state.rs:101-118`；`surface_pass.rs:202-215`），历史相位矛盾已经消除。
4. **ABI 影响必须独立记账**：`render_class` 位于 `GpuMaterialHeader`（`record.rs:150-167`）并参与管线/执行身份；七值顺序或域/特性映射变化属 ABI 级变更。当前 ABI/package/cache stale rejection 已有 §15.2 的定向回归证据；但历史变更是否严格作为独立批次、是否先于删除完成，仍不能由当前测试倒推。

**当前证据**：`pipeline_state_for_material`（`render_state.rs:95-157`）与 `render_fabric/pipeline_tests.rs` 提供全域、七个 class、water feature 和非法域组合的实现及测试定义；门禁 `domain_plus_render_class_plus_features_maps_to_pipeline_state`（§12 门禁 24）持续有效，不据此声称运行验收已通过。

---

## 8. 跨子系统回边的延迟契约

水体与粒子互为上下游：

```
water sim ──SpawnEvent──▶ particle sim
particle spray ──?──────▶ water foam
```

**机制已经存在**，不需要新建：`CouplingPlan`（`water/coupling.rs:68`）、`WaterSprayBurst`（`water/spray_bridge.rs:50`）、`SprayPublishStats`（`:81`）——水体产出的飞沫事件就是 `WaterSprayBurst`，有界两向耦合回读由 `CouplingPlan` 表达。

**设计契约**：不重建耦合算法，而是把这条边**从 water 模块内部上提为帧图边的属性**——写入侧产出到 `Persistent` 资源，读取侧声明正帧距 `FrameLag(1)`。

**理由**：否则特性图存在环，拓扑排序不可定义；框架不允许用“遍历顺序恰好正确”掩盖环（04 §4.1 / §5）。迁移前的模块内回边是这一要求的起因；当前 `render_fabric/builtins.rs:112-116` 已有一帧延迟声明，`compile_tests.rs` 有历史帧绑定与无同帧边的测试定义，但它们不证明生产水体/粒子回读已由 Fabric 的编译与 runtime 导入驱动。

**机制成本与边界**：版本模型包含帧维度（§4.4）；声明、CPU 编译映射与生产 history 的分配/发布/导入必须一致。`RuntimeFeatureGraph::execute` 的单 command buffer auto-sync 不承诺外部 history 发布或原生多队列 fence（`render_fabric/runtime.rs:180-188`）。这些约束继续有效，不能把 API 存在当作生产证据。

**门禁**：`spray_foam_edge_is_one_frame_lagged` —— 断言该边的 `FrameLag == 1` 且图中无环（§12 门禁 7）。

---

## 9. 原始验收顺序与当前证据状态

下表保留完整验收目标与先后约束，不是实际执行日志。源码实现、测试定义、生产调用和运行结果分别记账；不能从某个子项的实现推出 S1–S6 或 S4b 整体验收完成。证据汇总于 §15，定向测试结果仅适用于明确记录的范围。

| # | 切片 | 原始要求（持续有效） | 当前证据状态 |
|---|---|---|---|
| **S1** | **求解契约**（纯新增，**不碰 PCM**） | 在 `prism_render_architecture` 定义 `ResourceOwner` / `ResourceKey` / `ResourceScope` / `ResourceUse` / `ServiceSet` / `ServiceOutput` / `RenderQualityTier`，与 `FeatureDefinition`（`kind` + `Option<domain>`）/ `FeatureKind`（`Service { provides }`）/ `FeatureVariant`（含 `requires_services_extra`）/ `PassTemplate` / `FeatureCatalog` 骨架，并**同时实现求解契约本身**：① 编译期静态校验（唯一 provider 索引 `providers[s]`、槽位与 ID 唯一性、**边种类矩阵**）；② **硬需求闭包**计算（只沿 `FeatureDefinition.requires_services`）；③ `FeatureVariant` 选择器与 `FeatureSelection::resolve`（拓扑序 + `FeatureId` 破平局 + `rank` / `RenderQualityTier` 判定，单遍无回溯，结果整体冻结）；④ 对应静态测试——`active_roots` 闭包隔离、声明顺序打乱后逐项相等、软需求不扩张闭包、重复 provider 报错。门禁 **27 / 29 / 30 / 31 / 32** 在 S1 内即须为绿。**`ExtensionManifest` 一个字段都不加**（§5.2） | 类型、求解实现与对应单元测试已存在，并包含在 2026-10-20 的先前 134 项完整运行记录中；本次收口没有新的通过记录，也不能追证这些门禁在后续迁移前已全部通过 |
| **S2** | **建护栏** | 三份服务集分别与 `ServiceSet` 声明做“旧位掩码/布尔结构 ⇔ 新声明一致”断言；§3.1.1 的相位映射表必须有断言（旧 `prism_render_app/src/tests.rs:849-852` 是迁移前定位，非当前行号） | 当前服务闭包与 pipeline 映射有实现及测试定义；旧新等价护栏的历史运行及先后记录未在本轮核验 |
| **S3** | 顺序推导 | `frame_graph` 由 `accesses` 派生依赖与 `Barrier`；断言派生结果 == 迁移前手写结果；消除固定 pass 数矛盾（旧 5 vs 6 是历史问题，不是当前计数） | 编译 fixture 有顺序/barrier 对比；当前 shading 声明为 **8 + 23 = 31**，数量断言读取编译图。现有显式语义 `depends_on` 不得替代资源 hazard，生产自动推导与历史等价仍需各自证明 |
| **S4** | 一条垂直切片 | 水体声明 `requires_services` / `accesses`，由统一编译器自动排序并完成生产端到端消费；消除 `WaterPrepare.shared_base` / `bound_base`；常驻 `CompositionPass` 承载 `MaskBlend` | 旧字段/入口收口及现有水体 → Composition / HDR 链路已证；water compute 与 resident composition 已各自通过 Fabric compile/prepare/record/submit 接入局部生产切片。composition 已共享 SceneFabric selection/runtime，但两个局部 recorder 尚未统一为覆盖 TAA/history、全域 payload 与其他域的 `SceneFabric` 全场景闭环（§1.3） |
| **S4b** | 水体的行为等价 | **旧入口删除前须有八项 golden 等价护栏**：深度、岸边、泡沫、焦散、反射、透明/overlay、motion vector、reactive mask；保留连续混合、统一 CPU/WESL 公式并消费 `overlay_opacity`，完整门禁 23 持续有效 | 公式、参数接线、八项 CPU golden 与组合网格已证；`water_hybrid_gpu.rs` 的 **19/19** 包含 10 项当前输出、8 项冻结 legacy GPU parity 与 1 项比较器。仍缺删除前护栏运行时序、完整 Scene/Fabric/TAA 帧与跨适配器证据，**完整验收未闭合** |
| **S5** | 删除 + ABI bump | 删除三份服务集、两份 `ShadingFrontend`、`bound_base` / `shared_base`、`MaterialRenderClass` 四值及旧 `_hybrid` 入口；保留七值 class、域/特性到 `PipelineState` 的映射；以**独立 ABI 变更批次**记录入口/class 变化并验证缓存失效 | 删除、映射、冻结 `FeatureSelection → package/permutation → PSO/cache/warm set` seam 及 ABI/cache stale rejection 已有源码与定向运行证据；仍不能追证历史批次顺序，且完整 Scene/Fabric、真实 driver binary 与跨适配器覆盖未完成 |
| **S6** | 扇出 | 端到端切片成立后，将粒子 / 体积 / 毛发 / 贴花全部接入同一声明、选择与生产执行链路；补齐 `render_layers` delta | built-in roots 与 selection、`render_layers` 接线已存在；各域 `FrameRenderItem` / payload / 真实 GPU recorder 的全量生产扇出尚未验收 |

**验收顺序契约**：先建立可断言的护栏（S2），再执行受护栏保护的迁移（S5）；先证明一条生产端到端切片（S4），再验收扇出（S6）。这保留原始顺序要求，但不声称本轮掌握历史已按此执行的证据。S4b/S5 的已实现子项不能抵销未闭合的验收项。

**S5 的独立 ABI 要求不变**：入口名属于 shader package 编译身份，`MaterialRenderClass` 改值属 ABI 级变更（`GpuMaterialHeader.render_class`，`record.rs:150-167`）。ABI/package/cache stale rejection 与冻结 selection mapping 的当前回归已有证据；四分量管线身份与冷热缓存边界参见当前 03 §2 / §3.2 / §5，而非旧 §4 的 DDC 节号。当前结果不能反向证明历史上已将 ABI bump 作为独立、可归因且先于删除的批次执行。

---

## 10. 风险

| 风险 | 处置 |
|---|---|
| **入口重命名 / `render_class` 改值 → PSO 缓存静默复用旧管线** | typed pipeline identity、ABI 版本/哈希及 stale rejection 已有定向回归；历史独立 ABI 批次与先后顺序仍须可追溯，不能由当前测试反推（§7.2 / §9 S5） |
| 删除与建立之间的**零护栏窗口** | 保留“先护栏、后删除”要求；当前 legacy static gate、pipeline/水体 golden 定义可防回归，但不能追证删除前已运行、没有历史空窗 |
| **误删水体 Hybrid**（本版新增） | 公式/overlay、CPU 八项 golden、组合网格、resident Composition 接线及 8 项冻结 legacy GPU parity 已有证据；删除前护栏时序、完整 Scene/Fabric/TAA 帧和跨适配器等价仍未证，禁止用 `REGION_ILLUMINATION` 替代连续混合（§7.1 / 门禁 23） |
| **四值迁移把"对象域"与"管线状态域"一起删掉**（本版新增） | 已由 `RenderDomain`、稠密可逆 `PipelineBinClass` 与 `PipelineState` 映射保留；§12 门禁 24 作为持续回归契约 |
| **变体选择引入每帧输入 → 帧图拓扑随场景变化**（本版新增） | §4.1 约束 1：选择输入只有 `device_caps` + 静态 `RenderQualityTier`；§12 门禁 22 |
| **`ResourceKey` 退化成中心大枚举**（本版新增） | §4.5.1 命名空间键；新增子系统不得改中央枚举；§12 门禁 20 |
| **`FeatureCatalog` 退化成新的总注册表** | §0.2 四权威边界 + §12 门禁 12/13/14：PCM 不得装特性，特性不得装材质参、不得装本帧资源 |
| **`ResourceUse` 的新字段沦为装饰** | §4.4 两条裁定持续有效：当前 `scope` 仅支持 `Whole`，正帧距已进入编译版本/历史绑定模型；生产 history 分配、发布、导入同样必须消费该契约，不能把编译测试当作运行闭环 |
| 把 `ResourceId` 写进静态声明 | §4.5 + §12 门禁 16（类型层面可查） |
| payload 退化为无界垃圾桶 | §12 门禁 6：`Resource` 支每个字段必须有 `lifetime`/`size`/`alignment` 且在图中被引用 |
| **反向：为过门禁制造伪资源**（本版新增） | §6.3 的和类型 + 判据表：`bounds` / `phase_params` 走 `Value` 支；门禁 6 同时断言 `Value` 支不承载 buffer / texture |
| **遍历全部特性使可选服务成为必选**（本版新增） | §4.1 相 B 只求解 `active_roots` 闭包；§12 门禁 28 断言闭包外特性不参与求解、不需要可用变体 |
| **`active_roots` 随场景伸缩**（本版新增） | §4.1 相 B 冻结规则：来源只含构建档案与可引用程序集；集合外出现新根是硬错误而非静默加根，保持 00 §3 / 04 §5 的拓扑稳定边界 |
| **`Service → Subsystem/Response` 反向边绕过分层**（本版新增） | §4.1 相 A 第 3 条的边种类矩阵；§12 门禁 27 静态拒绝 |
| **闭包依赖变体选择**（循环定义）（本版新增） | §4 类型定义把需求拆成特性级硬需求 / 变体级软需求；闭包只读硬需求，软需求只作候选准入（§4.1 相 B、门禁 31） |
| **同一 `ServiceId` 有多个提供者，"谁提供"不可判定**（本版新增） | §4.1 相 A 第 5 条的唯一提供者映射 `providers[s]`；重复声明是构建期错误（门禁 29） |
| **`Subsystem` / `Response` 也声明 `provides`，使边矩阵无法机械执行**（本版新增） | `provides` 移入 `FeatureKind::Service` 载荷（类型层面排除，门禁 30） |
| **变体各自发布不同的服务资源接口**（本版新增） | 服务接口必须变体无关：所有变体写同一组 `ServiceOutput.resources`，否则消费方需要知道选中的变体（门禁 32） |
| **`ValuePayload` 变成"指向参数块"的旁路**（本版新增） | §6.3：`Value` 只有定宽不可变内联值；统一参数块必须走 `ResourcePayload` / `ParameterBlock`（门禁 6、禁止 37） |
| 特性图成环 | §8 一帧延迟 + §12 门禁 7 |
| 以"消灭位掩码"为名削弱 G4（NPR 降级） | §12 门禁 3/4：`ServiceSet` 仍能机械推出原 `ALL`；`NPR ⊇ PBR` 仍是单调不变式 |
| 术语撞名（`Region` / "响应图" / `FeatureRegistry` / 两份 `CapabilitySet` / `QualityTier`） | §3.5 命名坑表；全文只用"特性图"与 `FeatureCatalog` |
| `Volume`/`Hair`/`Decal` 同时留在两个轴 | §3.1 移交 + §12 门禁 15 |
| 六路并发迁移到仍在变的 header | 严格 S1→S6；先定契约，再一条切片，再扇出 |
| 把声明做成运行期结构 | §11 禁止 17；声明与选择必须冻结，遵守 00 §3 权威边界及本篇 §4.8 的单一置换事实源，不再把旧等式误引为当前 03 §3.1 |
| **变体求解依赖声明顺序 / 注册顺序**（本版新增） | §4.1 相 B：依赖图拓扑序 + `FeatureId` 破平局，边种类矩阵禁止反向边与同层互依；§12 门禁 22 用"打乱声明顺序"的同一 catalog 断言逐项相等 |
| **`ServiceOutput.resources` 沦为声明性标签**（本版新增） | §4.6.1 的写入集机械约束：产出资源必须由提供者特性的**每一个**变体的 pass 实际写入（只查选中变体会漏掉换能力后的静默失效）；§12 门禁 25 / 32 |
| **`FeatureId` / `ResourceKey.slot` 按声明顺序发号**（本版新增） | §4.5.1：显式常量、catalog 只校验不发号；插入一个资源不得改变其后所有身份；§12 门禁 26 |

---

## 11. 禁止

1. 新建 `prism_render_contract` 或任何承载 `ResourceId` / `CapabilitySet` 的新 crate（§2.2）。
2. 引入第三套拓扑排序或节点图求值器（本篇 §0.1 保留的排除项；统一编译器边界见 04 §1.1）。
3. 引入 `RenderProgram` / `RenderPlan` / Blend / Layer 节点（本篇 §0.1 独立约束；程序/组合权威见 00 §3）。
4. **往 `prism_material_pcm::ExtensionManifest` 加 `reads` / `writes` / `passes` / `services` 任何一项**（§0.2 / §5.2）——PCM 是材质扩展权威，不得成为总注册表。
5. 使用 `FeatureRegistry` 这个名字承载特性声明（与 `diagnostics::FeatureRegistry` 撞名，§0.1 / §3.5）。
6. 在静态特性声明里保存 `frame_graph::ResourceId`（§4.5）。
7. 用两个列表分别存 `reads` / `writes`（§4.3）。
8. **在 `ResourceUse` 与 `PassTemplate` 同时保留 `queue`**（§4.3 采纳方案 A，一个 pass 一个队列）。
9. 引入装饰性的 `scope` / `frame_lag` 字段而不改造 `Barrier` / `ResourceVersion`（§4.4）。
10. 用 `requires_services` / `accesses` 重表 closure 瓣集（与 02 §1 的五类契约重复，§4.7）。
11. 在 `FeatureVariant` 里重新枚举 shader 变体（应引用 `PermutationSpace`，§4.8）。
12. 往 `InstanceRecord` 加 `history_key` / `motion_source` / payload，或加第二个 `stable_id`（§6.1）。
13. 让 `ResourcePayload` 字段缺 `lifetime` / `size` / `alignment`；把 buffer / texture 放进 `ValuePayload`；或让 `ValuePayload` "指向"一个统一参数块以绕开资源校验（§6.3）。
14. 让 `Volume` / `Hair` / `Decal` 同时留在 `RenderDomain` 与 `MaterialRenderClass`（§3.1）。
15. 用“执行族”称呼 Raster / Compute / RT；它们是派发方式，队列归 `QueueClass`，都不是 `ExecutionFamily` 的响应执行分组（§3.3）。
16. 缺能力时在 shader 内静默兜底（本篇 §4.1 的候选准入与显式错误契约；00 §3 的不可解析程序同样是硬错误）。
17. 把声明变成运行期可变结构（本篇 §5.1 冻结契约与 00 §3 权威边界）。
18. 把带 `MaskBlend` 的材质纳入守恒判定（00 §3 / §4 的 conservation 排除与独立计数约束）。
19. 用本文档排除的名字"响应图"称呼特性图（§0.1）。
20. **给非 `Subsystem` 的特性伪造一个 `RenderDomain`**（§4.0）。
21. **把 `ResourceKey` 做成容纳所有子系统资源的中心枚举**（§4.5.1）。
22. **复用 `prism_app::capability::QualityTier` 作为渲染质量档**（§3.5）。
23. **变体选择读取每帧测量值**（帧耗时、可见像素数、预算余额）而非常量设备能力与画质档（§4.1）。
24. **只删 `MaterialRenderClass` 的四个值而不建立 `→ PipelineState` 映射**（§3.1.1 / §7.2）。
25. **用 `REGION_ILLUMINATION` 替代连续权重混合**（§3.6 / §7.1）。
26. **未建立八项 golden 等价护栏即删除旧水体 Hybrid 入口，或无等价保护地改动当前 `MaskBlend` / `CompositionPass` 契约、恢复 `water_surface_fs_hybrid`**（§7.1 / §9 S4b）。当前入口已删除不构成删除前护栏运行时序的证明，该历史要求不能因此撤销。
27. **按声明顺序、注册顺序或遍历顺序分配 `FeatureId` / `VariantId` / `ResourceKey.slot`**（§4.5.1）。
28. **让 `ServiceOutput.resources` 停留在"声明性标签"**而不被提供者特性的**全部变体**的 pass 实际写入；或只校验当前选中的那个变体（§4.6.1、门禁 25）。
29. **在消费者侧（帧图编译器、PSO / 管线缓存、队列系统）重新调用 `resolve` 或重新比较 `rank`**（§4.1 约束 4、§5.1）。
30. **在求解时遍历 catalog 的全部特性**而不从 `active_roots` 的依赖闭包出发——那等于把每个特性都变成必选项（§4.1 相 B）。
31. **在求解公式里保留 `engine_services` 之类隐式引擎旁路**；核心服务必须由特性声明提供（§4.1 约束 5、§4.6.1）。
32. **让 `active_roots` 依赖每帧可见性 / 材质计数 / 任何测量值**，或在设备生命周期内增删根（§4.1 相 B；00 §3 / 04 §5 的拓扑稳定边界）。
33. **把 `bounds` / `phase_params` 这类小而不可变的参数硬做成帧图资源**只为满足门禁（§6.3）。
34. **把 `requires_services` 挂在 `FeatureVariant` 上作为闭包输入**——闭包会依赖变体选择，而选择又依赖闭包（循环定义，§4.1 相 B）。特性级硬需求与变体级软需求必须分开。
35. **让两个特性提供同一个 `ServiceId`**，或让 `Subsystem` / `Response` 声明 `provides`（后者在类型层面已不可能，§4 类型定义 / §4.1 相 A 第 5、6 条）。
36. **用带回溯的递归候选求解替代单遍求解**——指数搜索，且"第一个解"难以断言（§4.1 相 B）。
37. **让 `ValuePayload` 持有或指向 GPU 参数块**（§6.3）。

---

## 12. 门禁

下列 **32 项全部保留为验收契约**，不是“32 项已通过”清单。文档门禁名不要求与测试函数一一同名；源码实现、测试定义、历史运行记录、生产接线与完整验收是不同证据层级。当前定向运行结果见 §15，尚未完成全部门禁的逐项端到端验收；尤其门禁 23 不能由 CPU golden 或 shader-module 创建扩大为完整 GPU 行为证明。

| # | 门禁 | 证明什么 |
|---|---|---|
| 1 | `feature_accesses_are_produced_by_the_graph` | 水体 surface 的**每个** `ResourceUse.key` 都能在特性图中找到产出 pass。防"声明与图漂移" |
| 2 | `derived_order_matches_hand_written_order` | 迁移期守卫：S3 推导出的顺序与 barrier == 迁移前手写结果 |
| 3 | `shared_base_is_derived_not_declared` | 删除位掩码后，由 `ServiceSet` 仍能**机械推出**原 `ALL` 集合（G3 的回归锁） |
| 4 | `npr_service_requirement_superset_of_pbr` | G4 的回归锁：`NPR.requires_services ⊇ PBR.requires_services`（由服务推导，不是硬编码相等） |
| 5 | `no_second_service_set_type` | 三份旧服务集的生产定义/引用为 0，不把门禁自身的禁止词样本或历史文档算作运行权威。当前 legacy gate 的明确扫描根与自身排除见 `render_fabric_legacy_gate.rs:54-91`；该扫描不等于任意全仓字符串搜索为 0 |
| 6 | `payload_declares_resources_and_values_explicitly` | `Resource` 支每个字段都有 `lifetime` / `size` / `alignment` 且其 `ResourceKey` 在图中被引用；`Value` 支是**定宽、不可变、可 `Copy` 的内联值**，不承载 buffer / texture，**也不"指向"任何参数块**（§6.3） |
| 7 | `spray_foam_edge_is_one_frame_lagged` | 图无环且回边显式延迟（§8） |
| 8 | `single_shading_frontend_definition` | 旧 `ShadingFrontend` 全仓保持 0 份定义；当前四个 typed water fragment entries 可以存在，但不得恢复旧 frontend 或旧 `_hybrid` 入口 |
| 9 | `every_bucket_resolves_to_a_compiled_entry` | 承 00 §3 / §4 的动态 bucket 与已编译 entry 解析约束；桶表变化后仍成立 |
| 10 | `region_selection_materials_compile_both_response_arms` | 承 00 §4 的区域选择两臂门禁语义。**注**：本门禁只证明两臂可编译，**不证明**与旧水体 Hybrid 行为等价（§7.1） |
| 11 | `shading_graph_execution_order_assertion_is_single_sourced` | 数量断言读取编译图，禁止重新写固定 pass 数；当前 8 + 23 的声明、单源断言与测试定义已核实（§1.6）；scene lib 本轮 1909/1909，architecture graph gate 未单独运行 |
| 12 | `extension_manifest_has_no_graph_fields` | 类型层面：`ExtensionManifest` 不含 `reads` / `writes` / `passes` / `services`（§5.2） |
| 13 | `feature_catalog_owns_no_material_params` | `FeatureCatalog` 不引用 01 的参数索引 / 02 的瓣集 / `SurfaceData` 字段（§0.2） |
| 14 | `feature_catalog_owns_no_frame_resources` | `FeatureCatalog` 只出现 `ResourceKey`，不出现 `ResourceId` / `ResourceVersion` / `Barrier`（§0.2 / §4.5） |
| 15 | `render_domain_and_render_class_are_disjoint` | Volume / Hair / Decal 属 `RenderDomain`；Water 属 `Surface + MaterialFeatureFlags::WATER`，不是一个 `RenderDomain::Water`。四个旧对象值不得回到 blend/raster class（§3.1） |
| 16 | `static_declarations_use_semantic_keys_only` | 静态声明类型里不出现 `frame_graph::ResourceId`（§4.5） |
| 17 | `every_persistent_instance_field_has_a_delta_kind` | `InstanceRecord` 每个字段都有对应 `SceneChangeKind`（§6.4） |
| 18 | `required_caps_names_the_device_capability_set` | `FeatureVariant.required_caps` 指 `backend::capability::CapabilitySet`，且有显式转换而非 `as`（§3.5） |
| 19 | `feature_kind_matches_domain_presence` | `kind == Subsystem ⟺ domain.is_some()`；服务级与响应级特性不得携带域（§4.0） |
| 20 | `resource_keys_are_namespaced_and_unique` | 每个 `ResourceKey` 有已声明的 `ResourceOwner`，`(owner, slot)` 唯一，且**未修改任何中央枚举**（§4.5.1） |
| 21 | `every_required_service_has_a_selected_provider` | 特性级 `requires_services` 与变体级 `requires_services_extra` 的每个成员，都由**唯一**提供者特性 `providers[s]`（已选）的 `ServiceOutput` 提供；无例外分支、无 `engine_services` 旁路（§4.1 相 A 第 5 条、§4.6.1） |
| 22 | `variant_selection_is_deterministic_and_registration_order_independent` | 校验**整个 `FeatureSelection`**（不只是单次 `select`）：对同一 `active_roots` 与**打乱声明顺序**的同一 catalog，`resolve(active_roots, device_caps, quality)` 逐项相等；闭包内每个 `requires_services` 都被满足；求解顺序只由「依赖图拓扑序（分层 + `Service` 子图序）+ `FeatureId` 破平局」决定，**不读注册顺序**；**闭包只由特性级硬需求决定**（同一 catalog 下改变变体的 `requires_services_extra` 不得改变闭包成员集合）；输入不含每帧测量值（§4.1） |
| 23 | `water_hybrid_golden_matches_before_entry_removal`（文档门禁） | **原要求不变**：旧入口删除前具备深度、岸边、泡沫、焦散、反射、透明/overlay、motion vector、reactive mask 八项等价护栏，当前组合链路持续保留该语义。`water_hybrid_golden.rs:276-466` 是八项 CPU golden / 独立 reference 网格，`:469-519` 是权重源码契约，`:607-613` 编译四入口，`:616-633` 检查 Composition/HDR 源码。`water_hybrid_gpu.rs` 的 **19/19** 包含 10 项当前 production output、8 项冻结 legacy full-frame parity 与 1 项比较器，已关闭冻结 GPU 输出等价缺口；但不能追证删除前护栏运行时序，也不替代完整 Scene/Fabric/TAA 帧、所有设备或跨适配器验收，因此 gate 仍为 **partial**（§7.1 / §15.2） |
| 24 | `domain_plus_render_class_plus_features_maps_to_pipeline_state` | §3.1.1 的映射表对四个旧值成立，且域仍可从 `bin_class` 恢复（§3.1.1 / §7.2） |
| 25 | `every_service_output_resource_is_written_by_every_provider_variant` | 对每个已选 `FeatureKind::Service` 特性 f：`f.kind.provides` 的每个资源都落在**f 的每一个变体**某条 `PassTemplate` 的写入项（`AccessKind::writes()`）中，且写者可归一到唯一 `ResourceVersion`；只覆盖当前选中变体不算通过（§4.6.1、§12 门禁 32）。否则 `ServiceOutput` 只是声明性标签 |
| 26 | `resource_keys_and_feature_ids_are_explicitly_declared` | `FeatureId` / `VariantId` / `(owner, slot)` 全部显式声明；插入或删除任一声明**不改变**已有 ID 与 slot；`FeatureCatalog` 只校验唯一性、**不按声明顺序发号**（§4.5.1） |
| 27 | `feature_dependency_kinds_are_stratified` | 静态检查边种类矩阵：只允许 `Service→Service`、`Subsystem→Service`、`Response→Service`；出现 `Service→Subsystem` / `Service→Response` / `Subsystem→Subsystem` / `Response→Response` / `Subsystem↔Response` 即构建期错误（§4.1 相 A 第 3 条） |
| 28 | `active_roots_are_static_and_frozen` | `resolve` 只求解 `active_roots` 的依赖闭包（闭包外的特性**不参与求解、不需要可用变体**），且 `active_roots` 只由构建档案与可引用程序集导出、在设备生命周期内不变；集合外出现新根即硬错误（§4.1 相 B） |
| 29 | `every_service_id_has_exactly_one_provider_feature` | 全 catalog 中每个 `ServiceId` 恰好被**一个**特性的 `provides` 声明；0 个（无人提供）与 ≥2 个（需仲裁）都是构建期错误（§4.1 相 A 第 5 条） |
| 30 | `only_service_features_provide_services` | `ServiceOutput` 只出现在 `FeatureKind::Service` 的载荷中；`Subsystem` / `Response` 的声明里不存在 `provides` 字段（类型层面，§4.1 相 A 第 6 条） |
| 31 | `variant_extra_requirements_do_not_grow_the_closure` | 改动任一 `requires_services_extra` 后，`closure` 的成员集合不变；被拒的候选只落到下一个 `rank`，不扩张闭包（§4.1 相 B） |
| 32 | `service_variants_publish_the_same_resource_interface` | 同一服务特性的**所有**变体都写入 `f.kind.provides` 里的**同一组**资源（接口是特性级、变体无关），**且逐个变体都成立**——即门禁 25 的量化范围是 `∀ v ∈ f.variants`，不是"仅当前选中变体"。变体差异只在实现，不在发布接口（§4.1 示例、§4.6.1） |

---

## 13. 非目标

- 不重做 04 的帧图合并裁决（(A) 已定）。
- 不改 01 的 `SurfaceData` 布局。
- 不改 02 的 `ShadingPlan` 形状与五类契约。
- 不改 03 的管线身份四分量。
- **不改 `prism_material_pcm::ExtensionRegistry` 的职责与 API**（§5.2）——只阻止它扩张。
- 不改 GPU Scene 的上传/镜像机制（只补 §6.4 的 delta 种类）。
- 不新建帧图编译器、求值器或图 IR。
- **不改变既有性能策略**（PSO 预热、GPU Scene 批处理等）——本篇已将有界内存、稳定排序、资源生命周期和非阻塞编译作为结构约束，性能策略由对应运行时模块和独立基准维护。
- **不新增跨队列 pass 能力**（§4.3 采纳方案 A，主动放弃）。

---

## 14. 交叉引用

### 本系列

- [`00-overview_refactor_zh.md`](00-overview_refactor_zh.md) §1–§2（Artifact / 响应与组合）、§3（ABI、程序解析、四权威与拓扑边界）、§4（具名门禁语义）、§5.2（系列历史运行记录，不能替代本篇全部验收）。旧 I 编号与旧 §8 已不适用；被排除的第三编译器/求值器等要求独立保留于本篇 §0.1。
- [`01-surface-abi_refactor_zh.md`](01-surface-abi_refactor_zh.md) §2.2（`GpuMaterialHeader`）、§3（身份与稠密索引——本篇 §3.1.1 的域编码必须保持一致，不手写位打包）、§4（`ClosureKind` 分层保词，本篇不删瓣）、§6–§7（门禁及证据）。
- [`02-response-composition_refactor_zh.md`](02-response-composition_refactor_zh.md) §0（`ProgramRef` / `BuiltinProgram` / `ShadingPlan`）、§1（五类契约）、§2（执行分桶）、§3（`MaskBlend` 的 fused/split 与共享 lighting context）、§4（二值选择与连续 mask）、§5（Custom 注册程序）、§6（水体组合链路）。
- [`03-shader-package_refactor_zh.md`](03-shader-package_refactor_zh.md) §2（管线身份四分量）、§2.1（静态轴与动态数据）、§3（热/冷缓存与 DriverLocal 边界）、§4（Direct/bin pass set）、§5（必须保持的约束）。原始单一置换事实源与独立 ABI 批次要求仍见本篇 §4.8 / §9 S5，不误引为当前 §3.1 的内容。
- [`04-frame-graph_refactor_zh.md`](04-frame-graph_refactor_zh.md) §1.1（统一 `prism_render_graph` 与 architecture re-export）、§2（8 主链 + 23 后处理）、§3（生成式后处理与 executor ledger）、§4.1（资源访问为依赖真源）、§4.2（两级调度）、§4.3（常驻 Composition 数据流）、§5（架构约束）。旧 5 vs 6、9 个声明和约 40 个手排计数仅属历史背景，本篇 §2.1 的 7 组全域装配目标仍全部保留。

### 上层设计文档

- `docs/prism_material_pipeline_design_zh.md:18-19`（G3/G4 共享基底目标——**意图保留，载体替换**）、`:106`（旧 `SharedBaseServices` 与 `shared_base()` 的设计记录，不代表当前源码）、`:334-336`（“GPU-driven 基底只算一次”）、`:618`（旧“四档子系统注册表”记录，不作为当前运行权威）
- `docs/prism_aaa_advanced_features_zh.md`（高级特性清单）
- `docs/prism_water_engine_design_zh.md`（Hybrid 与 `overlay_opacity` 的设计承诺——当前实现及证据边界见本篇 §7.1）、`docs/prism_volumetric_engine_design_zh.md`、`docs/prism_particle_engine_design_zh.md`、`docs/prism_hair_engine_design_zh.md`、`docs/prism_cloth_engine_design_zh.md`、`docs/prism_terrain_world_design_zh.md`
- `docs/prism_ui_loom_design_zh.md:16,126`（"响应图"撞名，须避）
- `docs/prism_render_scene_rhi_port_zh.md:38/100/104/107`

### 代码

- 帧图类型（`ResourceId` / `AccessKind` / `AccessKind::writes` / `ResourceAccess` / `PassDescriptor` / `Barrier` / `ResourceVersion` / `QueueClass` / `ResourceLifetime`）：`pkg/prism_render_graph/src/frame_graph/types.rs:8-139`；`pkg/prism_render_architecture/src/frame_graph/mod.rs` 仅兼容 re-export，不再有 architecture 自有 `types.rs`。
- 持久场景与增量：`pkg/prism_render_architecture/src/gpu_scene/mod.rs:73`（`InstanceRecord`）、`:84-104`（`SceneChangeKind` / `SceneChange` / `SceneChangeJournal`）
- 设备能力：`pkg/prism_render_architecture/src/backend/capability.rs:32`（`Capability`）、`:77`（`CapabilitySet`）、`:87`（`WEBGPU_BASELINE`）
- 命名坑：`pkg/prism_render_architecture/src/diagnostics/registry.rs:46`（`FeatureRegistry`，**已被占用**）、`diagnostics/mod.rs:47`（re-export）、`pkg/prism_app/src/capability.rs:249`（`QualityTier`，**是宿主档不是画质档**）
- **历史三份服务集（已收口）**：`water/mod.rs:439`、`volumetric/mod.rs:628`、`particle/shading.rs:152`/`:167`/`:185`；当前唯一服务声明与 roots 见 `render_fabric/builtins.rs`，旧符号由 legacy gate 防回归
- shader 编译期变体：`pkg/prism_render_architecture/src/shader_package/permutation.rs`
- 响应程序与组合：`pkg/prism_render_material/src/shading_plan.rs:21-75` 定义 `BuiltinProgram::{Lit,Npr,Unlit}`、`ProgramRef::{Builtin,Registered}`、`ShadingPlan::{Single,MaskBlend}`；`:120` re-export `ExecutionFamily`，定义真源是 `pkg/prism_render_architecture/src/shader_package/execution.rs:8-12`。不再引用已删除的 `axis.rs / Illumination`。
- 材质 class 与 ABI：`pkg/prism_render_material/src/record.rs:41-69`（七值 class）、`:125-132`（`REGION_ILLUMINATION` / `ILLUMINATION_BLEND`）、`:150-167`（`GpuMaterialHeader`）。
- **当前 `MaterialRenderClass` 与域映射**：七值 blend/raster 轴见 `pkg/prism_render_architecture/src/render_fabric/pipeline.rs:5-17`；稠密可逆 `PipelineBinClass` 见 `:203-250`；完整域/class 状态与回归见 `render_fabric/pipeline.rs:253-408`、`render_fabric/pipeline_tests.rs:102-116,238-257`；材质侧 water 映射见 `pkg/prism_render_material/src/render_state.rs:95-157`
- **当前水体 Hybrid 公式与消费路径**：CPU `resolve_hybrid_weights` / `compose_mask_blend` 见 `pkg/prism_render_architecture/src/water/shading/hybrid.rs:109-204`；WESL 权重、overlay 与 reactive contract 见 `pkg/prism_render_scene/src/shaders/water_surface_raster.wesl:1666-1719`；`water_surface_fs_mask_blend` 从 `view.viewport.z` 消费 `overlay_opacity` 见 `:1795-1809`
- 水体入口与相位：`pkg/prism_render_architecture/src/water/gpu/surface_pass.rs:160-177,202-215`；旧 `_hybrid` 不在当前入口集合中，当前四入口的源码 gate 见 `pkg/prism_render_scene/tests/water_hybrid_golden.rs:607-633`
- 运行域：`pkg/prism_render_architecture/src/material/mod.rs:11-17`（`MaterialDomain`，**不复用**）、`:20-25`（`MaterialExecutionPath`）
- PCM（**保留，不扩张**）：`pkg/prism_material_pcm/src/{manifest,extension_set,registry,negotiate,namespace,capability}.rs`（`SLOTS_PER_REGION` `namespace.rs:103`、`allocate_slot` `registry.rs:191`）
- **旧权威回归门禁**：`pkg/prism_render_architecture/tests/render_fabric_legacy_gate.rs:11-19,54-91` 扫描旧服务集、旧 frontend、`bound_base` / `shared_base` 等符号；当前水体旧入口收口由 `surface_pass.rs:167-177` 与 `water_hybrid_golden.rs:607-633` 共同证明
- shader：`pkg/prism_render_scene/src/shaders/{shading_resolve,visibility_raster,water_surface_raster}.wesl`、`pkg/prism_render_scene/src/shading/{graph,runtime,raster}.rs`
- 回边：`pkg/prism_render_architecture/src/water/{coupling,spray_bridge}.rs`
- 设备选择与 runtime 边界：`pkg/prism_render_scene/src/gpu/fabric.rs:13-51` 实际设备初始化 `SceneFabric`；`pkg/prism_render_architecture/src/render_fabric/device.rs:33-48` 定义 `compile`，`runtime.rs:180-200` 定义 runtime 执行与 prepare seam；水体窄 adapter 的生产调用见 `pkg/prism_render_scene/src/shading/fabric_executor.rs` 与 `water/dispatch.rs`，但测试调用和窄切片都不能当作全场景生产闭环。

## 15. 当前实现证据审计（2026-10-20 更新）

### 15.1 当前可核实的实现与测试定义

以下表格区分源码与生产接线；具体测试执行结果另见 §15.2。除明确指出的 scene 生产调用外，API、fixture、CPU sink 或测试中的 runtime 调用不等于生产接线。

| 条款 | 直接证据 | 能证明的范围 |
|---|---|---|
| `FeatureKind`/domain 类型边界、显式 ID 与 namespaced `ResourceKey` | `render_fabric/tests.rs`：`feature_kind_matches_domain_presence`、`constructors_and_explicit_keys_are_const`、duplicate ID/rank、`resource_keys_are_namespaced_and_unique`；`pipeline_tests.rs` 覆盖五个 domain 与七个 render class | 类型/校验实现及测试定义存在；不声称本轮测试通过 |
| 唯一 provider、hard closure、soft requirements、依赖分层 | `render_fabric/tests.rs`：`every_service_id_has_exactly_one_provider_feature`、`missing_hard_and_soft_providers_are_static_errors`、`feature_dependency_kinds_are_stratified`、hard cycle、`hard_transitive_closure_and_selected_providers`、`variant_extra_requirements_do_not_grow_the_closure` | 静态 catalog 与选择契约有实现和断言，不替代生产资源消费 |
| active roots 冻结与闭包隔离 | `selection.rs:40-89` 的 `FrozenRoots::from_build_profile` / `require_root`；`tests.rs` 的 `active_roots_are_static_and_frozen`、来源/unknown、empty profile 与 unused declaration 测试 | roots 冻结与闭包隔离模型；不代表所有域已提交 GPU 工作 |
| 设备能力/画质驱动的确定性选择与注册顺序无关 | `selection.rs:185-267`；`tests.rs` 的 capability/quality matrix、no candidate、registration order 与 explicit identity 测试；03 §2.2 的冻结 shader binding seam | 选择器与 `FeatureSelection → package/permutation → PipelineKey/PsoCacheKey/warm set` 的生产消费 seam 已闭合；完整 Scene/Fabric 每帧接线、driver binary 与跨适配器覆盖另计 |
| 服务输出、每个 provider variant 写入资源、payload 资源/值校验 | `tests.rs` 的 `every_service_output_resource_is_written_by_every_provider_variant`、`service_variants_publish_the_same_resource_interface`；`compile_tests.rs:648,790,857` 的 layout、payload、sink/barrier 测试 | 声明和 CPU 编译校验；不等于真实全域 payload 已从 scene 提取并绑定 |
| compiler 的 active-pass slice | `compile.rs::ResourceTemplates::compile_active`、`device.rs::DeviceFabric::compile_active`；`compile_tests.rs::active_compile_uses_frozen_selection_and_explicit_completed_dependencies` | API 只编译 frozen selection 中明确指定的 pass 并限制资源 lowering 到这些 pass 的访问；省略依赖必须显式声明已完成。此前定向测试 `1/1`；另一次历史复跑曾被并发 `prism_physics_core` 编译错误阻断，最新源码未在本次复验。不是共享全场景 selection、全域 production imports 或 history 生命周期闭环 |
| 编译计划、帧图版本与一帧延迟 | `device.rs:33-48`；`compile_tests.rs:132,326` 的 authoritative graph 顺序/barrier 比较、history frame binding 测试 | fixture 证明版本/历史映射；water/composition 有局部 production compile 调用，active-pass compiler 也已提供 recorder-owned slice；这些都不证明统一全场景 compile 或 history 发布/导入完成 |
| 既有 RenderGraph 的 runtime lowering | `runtime.rs:180-200` 的 `execute` / `prepare_runtime`；`runtime_tests.rs` 的 recorder/import coverage、plan/hazard projection 测试 | runtime API 与测试调用存在；水体 compute 窄 adapter 已有生产调用，但不能标为全场景生产完成 |
| 水体 compute 与 spray feedback 窄生产接线 | `water/dispatch.rs::record_water_fabric_commands`、`shading/fabric_executor.rs::prepare_water_runtime*` / `prepare_spray_feedback_runtime`、`water/spray_history.rs::SprayHistory::prepare` | water solver 的本帧 surface/foam 与 spray 的 per-instance lag-1 feedback 均以真实 resident resources compile/prepare/import/encode 并进入宿主 command buffer；spray publication/import 由 host-queue completion 验证，resize/removal 的 buffer 与 RHI lease 经 completion retirement 后回收。上述 water/spray 专属运行验收未由本次四份日志提供，不用 composition/TAA 局部通过替代；这仍是独立窄 adapter，不覆盖 response composition/TAA、其他域或全场景唯一 recorder |
| PCM 权威边界与 sidecar bridge | `pcm_tests.rs:132,172,246,481` 的 capability conversion、authored variants、inactive variant enforcement 与 manifest→catalog→selection→CPU sink 测试 | bridge 模型与 fixture；one-chain CPU sink 不是生产 GPU recorder 或整帧闭环 |
| built-in catalog、roots 与实际设备 selection 初始化及局部准入 | `builtins.rs:527-552` 的 roots/catalog；`prism_render_scene/src/gpu/fabric.rs:13-51` 从 Bevy `RenderDevice` 建立 `SceneFabric`；`water/spray_history.rs:156-167,234-250` 读取选择并 admit particle root；`shading/plugin.rs:463-515,629-633` 将同一选择接入 executor 初始化 | selection 初始化与局部准入已有生产调用；shading 仍编译既有图，不等于 Fabric compile/runtime 闭环，五个 subsystem 和三个 response root 的声明不等于所有高级服务已生产实现 |
| pipeline mapping、ABI/cache 与 `render_layers` | `pipeline.rs:203-250`、`pipeline_tests.rs:102-116,238-257`；03 §2.2/§6 的冻结 selection mapping；`gpu_scene` transaction/mirror 的 `RenderLayers` 接线与测试（§6.4） | 域/class 可逆编码、显式状态映射、delta、ABI/cache stale rejection 与冻结 mapping seam 已有源码及定向运行记录；历史独立批次顺序、真实 driver binary 和跨适配器范围另计 |
| 旧服务集、旧 frontend 与旧水体入口收口 | `render_fabric_legacy_gate.rs:11-19,54-91`；`surface_pass.rs:167-177`；`record.rs:41-69` | 旧符号删除和当前替代物已有源码证据；legacy gate 有明确扫描范围，不声称全仓字符串零命中 |
| 水体连续 Hybrid、overlay 与 resident Composition | CPU `hybrid.rs:109-204`；WESL `water_surface_raster.wesl:1666-1719,1795-1809`；`shading/graph.rs:196-222`；`water_hybrid_golden.rs:276-519,607-706`；`water_hybrid_gpu.rs` | 公式/参数消费、具体 scene 链路与冻结历史 shader closure 已证；当前 **19/19** 含 10 项当前输出、8 项冻结 legacy full-frame parity、1 项比较器；仍不构成完整 Scene/Fabric/TAA 帧 parity、删除前护栏历史时序或全场景 Fabric 闭环 |
| 五个 response plane 的身份与真实 buffer binding | `water/fabric.rs` 的 `ResponseCompositionPlane`；`shading/composition.rs` 的 `COMPOSITION_ACCESSES`、`resource_bindings` / `resource_templates` / `record_composition_fabric`；built-in catalog 的 response templates | `ResourceKey`、动态 template、真实 Bevy `BufferId`、zero-job auxiliary allocation、生产 `FrameResourceBinding` 与 `RuntimeImport` 已统一；`response_composition_pass` 已经把真实 Fabric command buffer 交给宿主。其 accesses 是当帧资源、`frame_lag=None`，不需要 history publication；`resident_composition_gpu` 调用与生产入口共用的 recorder，覆盖本地 compile/prepare/record/readback，2026-10-22 strict 3/3 已关闭修订后 harness 的待复跑缺口（§15.2）；当前 composition 已接共享 SceneFabric selection/runtime，但仍不等于全场景 recorder |
| 通用 runtime 与 composition 共享生产切片 | `fabric_runtime.rs::prepare_with_templates`、`shading/composition.rs::scene_fabric` / `response_composition_pass`；`gpu/fabric.rs::init_scene_fabric` | 一次冻结 scene selection，composition 走真实 imports、active-pass compile、prepare、encode/host submit；acceptance wrapper 不是第二份生产 catalog。2026-10-22 composition strict 3/3 与 runtime multipass 9/9（8 hardware + 1 CPU fixture）分别验证局部切片，非全场景 recorder |
| native 共享帧、visibility 与 host admission | `prism_render_app/src/fabric_frame.rs`、`plugin.rs`、`visibility_bridge.rs`、`systems.rs`；`fabric_frame/tests.rs` 与局部 tests | 真实 Surface/Particle 聚合、独立 simulation/render clock、非法输入整帧清空、GPU Prepare 后刷新与 host record/submit 前拒绝源码及测试定义存在；不覆盖更早的 simulation 提交，最新源码无本次测试通过记录 |
| Hair/Volume/Decal owner、共享帧与 physical GPU admission/import | `fabric_domains/owners.rs`、`fabric_domains/consumers.rs`、`fabric_domains/resources.rs`、`fabric_domains/gpu.rs`，以及 `plugin.rs` 的 `CollectDomainContributions → BuildFrame` 接线 | render-world owner 与稳定 identity、快照/单次消费、validated semantic frame、`SceneRhi` physical lease 校验、运行时 import 与 record-ready view 已实现并通过 33 项筛选；不包含域 draw/dispatch recorder 或 full-frame parity。Volume 另有独立真实 volumetric fog pipeline/bind group/dispatch/copy-back，但尚未纳入 SceneFabric recorder；Hair/Decal 仍缺真实生产消费者 |
| shading graph 的数量单源 | `shading/graph.rs:79-225,285-289`、`post_graph.rs:938,1200`、`shading/runtime.rs:393-401,464-469` | 当前 **8 主链 + 23 后处理 = 31**，Composition 已在主链内；生产断言读取编译图，不保留旧 6 pass 假设 |
| 测试侧 32-gate ledger 与全域 fanout | `pkg/prism_render_architecture/tests/fixtures/render_fabric_gate_matrix.tsv`、`tests/render_fabric_acceptance.rs` | 32 个唯一 gate 均有可读源码锚点；五个 subsystem domain、三个 response root、provider/resource fanout 和 builtin shader package identity 均被验收测试覆盖；gate 23 保持 `partial`，不等于完整生产帧验收 |
| 跨适配器门禁 | `pkg/prism_render_scene/tests/cross_adapter_acceptance.rs`、`tests/fixtures/cross_adapter_acceptance.txt` | 每个 distinct physical adapter 都必须通过 storage-compute submit/readback；少于两个时默认 `SKIP_UNVERIFIED`、严格 `PRISM_REQUIRE_GPU=1` 失败；skip 不计入通过 |

### 15.2 本轮检查与历史运行记录的边界

- **2026-10-22 architecture 最新主线程复验与提交**：架构改动提交 `652404b94`；`render_fabric` **200 passed / 0 failed / 0 ignored**（`/private/tmp/prism-fabric-architecture-20261022-precommit.log`），`shader_package` **95 passed / 0 failed / 0 ignored**（`/private/tmp/prism-shader-package-20261022-precommit.log`），`render_fabric_acceptance` **3 passed / 0 failed / 0 ignored**（`/private/tmp/prism-fabric-acceptance-20261022-precommit.log`），main 实际执行，terminal exit 0。200 项包含共享 `Arc<[FrameRenderItem]>` 和一次性稳定域索引，recorder 按域访问不再逐次扫描/复制整帧；包含对象身份、原始 access ordinal、互斥 pass、冻结 shader selection→package/permutation→PSO key 与物理子范围 alias 校验。本文较早的 185 项和 shader-package/acceptance“未复跑”仅代表此前截点，以本记录为准。3 项 acceptance 不等于完整 32 gate 端到端运行，gate 23 保持 `partial`。
- **2026-10-22 domain/runtime 与 RHI pipeline 局部桥接复核**：本次读取既有成功日志，未重跑 GPU。`/private/tmp/prism-fabric-domain-runtime-gpu-20261022.log` 为 **29 passed / 0 failed / 0 ignored**（CPU/GPU 混合）；其中 `authenticated_domain_views_upload_dispatch_and_preserve_subranges` 记录 Apple M2/Metal REAL HARDWARE，关闭 authenticated `from_domain`→上传→dispatch→readback/subrange 的局部验证缺口。`/private/tmp/prism-render-pipeline-bridge-20261022-outside.log` 为 **15 passed / 0 failed / 0 ignored**，包含 imported native render-pipeline identity/generation/lease 与原生绘制。不能把全部用例算作硬件，也不能外推为 Hair/Volume/Decal 的真实语义 producer/consumer、typed graphics 全链或统一全帧通过；较早笼统“from_domain 尚无 GPU 验证”的语句以本段的窄范围修正为准。

- **2026-10-22 architecture 帧容器定向运行记录**（main 实际运行，本次文档同步已读取日志、未复跑）：`cargo test -p prism_render_architecture --lib render_fabric::frame`：**19 passed / 0 failed**，日志 `/private/tmp/prism-fabric-frame-20261022.log`；对应 frame 改动已独立提交 `f8ab51076`。覆盖 owned vector 经 `from_items → into_items` 无重分配（`owned_batch_validates_without_reallocation`）、`replace_domain` 保留其他域/帧号并在测试容量内复用原存储（`domain_refresh_preserves_other_domains_clock_and_allocation`），以及错误时清空所有域、后续合法输入恢复（`rejected_domain_refresh_clears_all_domains_and_allows_recovery`）。此项仅为 architecture CPU 帧容器契约证据，不替代 Bevy/native 生产聚合、GPU/full-frame 或完整生产闭环验收；该 frame 子集不用于推增历史总数；全筛选及 catalog 的实际运行结果另记如下，不以局部通过替代 §15.3/§15.4 的全场景未完成项。
- **architecture 全筛选实际运行结果**：2026-10-22 architecture Render Fabric 局部复验：较早 main 运行的 `/private/tmp/prism-fabric-architecture-20261022.log` 为历史 **176/176**；后续 architecture 代理报告已实际运行 `prism_render_architecture --lib render_fabric` **200/200，terminal exit 0**，包含新增 9 项对象资源 CPU 契约测试，子集不与 185 重复相加。本次文档同步未运行构建，185 的来源是代理运行回报，不将旧 176 日志冒充新结果。该筛选不覆盖整个 `shader_package`、architecture acceptance、GPU 或全场景闭环。
- **RenderGraph 全 package 实际运行结果**：2026-10-22 RenderGraph 全 package 运行补记（main 实际运行并确认 terminal exit 0，本次已读取 `/private/tmp/prism-render-graph-20261022.log`）：`cargo test -p prism_render_graph`（无 `--lib` 限制）unit **122 passed / 0 failed / 0 ignored**；doc **1 passed / 0 failed / 1 ignored**。doc ignored 不计作通过。旧 85/85 与磁盘耗尽中止尝试仅保留为历史，不再代表当前状态；当前 package 验证缺口已关闭，不外推为全场景 GPU、full-frame 或跨适配器通过。
- **public Domain GPU integration（2026-10-22）**：沙盒内 0 adapter，严格模式拒绝；沙盒外以 `PRISM_REQUIRE_GPU=1` 重试 `--test fabric_domain_gpu`，Apple M2 / Metal 上 **1 passed / 0 failed / 0 ignored**，terminal exit 0，日志 `/private/tmp/prism-fabric-domain-gpu-20261022-outside.log`。覆盖公开 API 的设备对齐子区间绑定与 lease 保留；不证明域 shader、draw/dispatch 或全场景渲染。
- **runtime multipass strict GPU 实际运行结果**：2026-10-22 runtime multipass 局部复验：较早 `/private/tmp/prism-fabric-runtime-gpu-20261022-fixed.log` 的 **8/8（7 hardware + 1 CPU fixture）**保留为历史；main 后续实际运行并确认 terminal exit 0，本次核对 `/private/tmp/prism-fabric-object-runtime-gpu-20261022.log`，`PRISM_REQUIRE_GPU=1` 下 `fabric_runtime::tests::multipass_gpu` **9 passed / 0 failed / 0 ignored**，其中 **8 项 Apple M2 / Metal 硬件用例 + 1 项 CPU 编译契约 fixture**。新增 `object_suballocations_upload_and_execute_by_exact_identity` 使用 Surface 对象 fixture，验证 object index/generation/history/role 精确寻址、完整 payload 上传与错误 role/短上传拒绝、缺失/重复绑定和重叠拒绝，以及同一真实 buffer 子分配经多 pass 执行/readback 后互不串写；既有 lease/preflight、录制顺序、push constants 与失败清理仍各限测试范围。不是 Hair/Volume/Decal 语义 shader 或 `SceneFabricObjectResource::from_domain` 桥接 GPU 验证，也不是全帧或跨适配器证据。
- **2026-10-22 resident composition integration 复验**（main 实际运行并确认 terminal exit 0，本次文档同步只读日志、未运行 Cargo）：`/private/tmp/prism-resident-composition-gpu-20261022-aligned.log` 记录 `PRISM_REQUIRE_GPU=1` 下 `resident_composition_gpu` **3 passed / 0 failed / 0 ignored**。GPU producers 经生产共用 recorder 写入 HDR，以及 resident targets/pipeline identity 两项记录 Apple M2 / Metal / IntegratedGpu；另 1 项为缺少 HDR storage support 的 capability 拒绝测试，不把三项全部计作硬件用例。修订后的局部 harness 待复跑缺口已关闭，不证明统一全场景 recorder、完整 Scene/Fabric/TAA 帧、第二 adapter 或全 32 gate 通过。
- **2026-10-22 material package 复验**（main 实际运行并确认 terminal exit 0，本次已读取 `/private/tmp/prism-render-material-package-20261022.log`）：`prism_render_material` unit **888 passed / 0 failed / 0 ignored**、integration **79 passed / 0 failed / 0 ignored**、doc 0 tests。integration 为 freshness 19、header ABI 4、PCM parity 7、S5 ABI/cache 4、schema parity 10、ShadingPlan 33、surface ABI mirror 2，子集不与 79 重复相加。该 package 待复跑缺口已关闭，不表示其他 shading/architecture package、acceptance 或全场景 GPU 端到端验收已复跑。
- **2026-10-22 TAA GPU 首次阻断与重试通过记录**：main 的首次 TAA GPU 日志 `/private/tmp/prism-taa-gpu-20261022.log` 因依赖 `prism_physics_core/src/world.rs:651` 缺少 `prepare_assembly_collider` 报 E0599，未执行测试，仅保留为历史编译阻断；随后重试已成功，不再列为当前待复跑状态。main 实际运行并确认 terminal exit 0，本次文档同步已读取 `/private/tmp/prism-taa-gpu-20261022-retry.log`：`prism_render_scene --lib` 的 `shading::taa::resources::gpu_tests` 筛选在 `PRISM_REQUIRE_GPU=1` 严格模式下，Apple M2 / Metal REAL HARDWARE **19 passed / 0 failed / 0 ignored**，2021 filtered out，0.84s，包含新增 lifecycle 7 项。覆盖真实 bind-group、两帧 publication/import 与 CPU golden、queue completion，以及 resize/removal/device-table replacement 的 retirement、未提交 dispatch 资源保留与 ID 回收。日志中的 stale ID panic 由 `catch_unwind` 捕获，是预期失效断言，最终测试通过。TAA 局部 GPU 验证缺口已关闭；完整 Scene/Fabric/TAA full-frame parity、统一全场景 runtime/recorder 和第二 physical adapter 仍未闭合。
- **2026-10-21 当前工作树 Render Fabric 复验**：`CARGO_INCREMENTAL=0 cargo test --offline -q -p prism_render_architecture --lib render_fabric --jobs 1 --config profile.test.debug=0 -- --test-threads=1`：**175 passed，0 failed**。覆盖当前 contract/catalog/selection、pass alternatives、compiler/runtime、11 种 `AccessKind` lowering 与 imported readonly state fixture；它仍不是 scene 生产 recorder、GPU/full-frame 或 32-gate acceptance 结果。同期 `prism_render_graph --lib` 复跑在依赖归档阶段因 `No space left on device` 停止，未进入测试；RenderGraph **85/85** 只保留为先前运行记录；2026-10-22 后续全 package 已通过 unit 122/122、doc 1 passed / 1 ignored，取代该 package 的待复验状态。
- **2026-10-20 architecture 定向运行记录**：`cargo test -p prism_render_architecture --lib render_fabric -- --nocapture`：**136 passed，0 failed**；`cargo test -p prism_render_architecture --lib shader_package -- --nocapture`：**95 passed，0 failed**。这是本轮已完成的 contract/compiler/runtime 与 shader-package 定向测试，不是 scene 生产链或完整 32-gate GPU 通过记录。历史 `134/0` 是较早计数，不应覆盖本轮 136 项结果。
- **同设备资源桥的实际 GPU 运行**：`PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --lib gpu::rhi::tests -- --nocapture`：沙盒内因无 adapter 失败；随后沙盒外 Apple M2 / Metal **13 passed，0 failed，0 ignored，1834 filtered out**，无 GPU skip。验证 exact host handles、buffer/texture/view 的非 owning table lease、table-local ID/slot reuse、worker/drop、真实 host submission/completion；1 项为 Send/Sync 静态断言，其余使用硬件。资源桥具备导入能力不等于 Fabric 已使用真实场景 imports/recorders。
- **原生导入后端回归**：沙盒外 `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_driver_wgpu --lib native_ -- --nocapture`：**4 passed，0 failed，0 ignored，27 filtered out**。覆盖公开导入 API、buffer 与 texture/view 独立 lease／过期 ID／槽位复用，以及 `rgba32uint` 原生格式与整数绑定；该筛选结果不代表整个后端测试集通过。
- **历史记录**：前版本文记录 2026-10-16 Apple M2 Metal 的 Water Hybrid `17/17`、Resident Composition `3/3`；当前记录仅作为历史证据，不替代本轮结果或完整验收。
- **2026-10-20 水体复跑**：沙盒外 Apple M2 / Metal 上，`water_hybrid_golden` **19/19**，`water_hybrid_gpu` **19/19**；其中 GPU target 包含 10 项当前生产输出、8 项冻结 legacy full-frame parity 和 1 项比较器测试。冻结 closure 使用 commit `41a31ceef24e59ab7f8dda076b2b4c7f5e11f6ec`，容差为 FP32 `3e-5 + 3e-5*abs(reference)`、FP16 `2e-6 + 2e-3*abs(reference)`。
- **GPU 口径**：上述结果证明 16×16 水体 fixture 的 production WESL 与冻结旧 shader closure 输出一致；不证明完整 Scene/Fabric/TAA 帧、所有设备或全场景 Fabric compile/runtime 生产闭环。独立 TAA reactive MRT 的当前接线与 shader/layout 定向测试另有记录，但尚无完整场景 parity 证据。
- **ABI/cache 与 selection mapping 复跑**：S5 material target **4/4** 已包含在 2026-10-22 material integration 79 项中，不重复相加；app `pipeline_cache::worker_tests` **27/27** 仍为先前记录，不据此声称本次复跑。结合 architecture target 的先前完整记录，已覆盖冻结 selection 到真实 package/permutation/PSO identity/cache/warm set 的生产消费 seam 与 stale binding 拒绝。未关闭的是完整 Scene/Fabric 每帧装配、真实 driver binary 和跨适配器覆盖。
- **Render Fabric 测试侧验收**：`cargo test -p prism_render_architecture --test render_fabric_acceptance -- --nocapture` 为既有记录 **3 passed，0 failed**；前者验证 32-gate ledger、完整 built-in fanout 和 gate 23 的 `partial` 标记；`render_fabric` 当前工作树定向运行结果为 **200 passed，0 failed**（2026-10-22 main 实际复验），覆盖 contract/compiler/runtime fixture。水体 compute 与 resident composition 另有真实局部 `DeviceFabric::compile → prepare_runtime → record/submit` 调用，但这些测试与局部切片都不等于全场景生产闭环。
- **当前工作树 scene crate 全量单测复验（2026-10-20）**：在独立 `CARGO_TARGET_DIR`（`/private/tmp/prism-render-refactor.ndgbT0`）中，`CARGO_INCREMENTAL=0 cargo --config profile.test.debug=0 test -p prism_render_scene --lib --jobs 1 -- --nocapture` **1909 passed，0 failed，0 ignored**，完整编译并运行当时的 `prism_render_scene` lib-test target；该验证没有清理共享仓库 `target`。2026-10-21 仅 TAA retirement 测试增加了一条新 owner 导入断言；本轮使用共享 target 复跑该最终测试，**1 passed，0 failed**。完整 lib-test target 尚未在这条最后微调上重跑。此前“依赖产物缺失、scene builder 未编译”的失败记录已被这次全量结果取代；它证明前述源码版本的 crate 级编译和单测，不等于 Fabric 全场景 `compile → prepare_runtime → import → record/submit`、GPU payload 或跨适配器 parity 已完成。
- **域帧 owner、生产聚合与 physical GPU admission/import 进展及验证边界**：`fabric_domains.rs` 已从真实 `HairGroup`、`VolumetricPlan`、`DecalProjector` 构建排序稳定的 CPU `ValuePayload` descriptor、canonical `ResourcePayload` contract 与拒绝伪造/缺失 binding 的 consumer。`HairRenderWorldOwner`、`VolumeRenderWorldOwner`、`DecalRenderWorldOwner` 持有稳定 object/history/motion identity；Hair/Decal 发布持久快照，Volume plan 单次消费。Bevy 初始化贡献/status/owner 资源，公开 `CollectDomainContributions → BuildFrame` 调度锚点，并在生产系统中一次性 drain 同帧贡献后与 committed Surface 合成共享 `RenderFabricFrame`；混合帧、跨 producer 重复 identity/role、非法 payload、错帧 pending 或与 Surface 冲突均 fail closed，producer 停止后不回放旧项。`fabric_domains::gpu` 进一步使用真实 `SceneRhi` buffer/texture lease，验证 table/physical identity、容量/range/alignment/usage、texture layout/default view、exact binding/import coverage 和 physical alias，输出运行时 import 与 record-ready view。独立 target 定向运行：默认域 **22/22**、owner/producer **3/3**、`legacy-bevy fabric_domains` **33/33**；其中 GPU admission 8 项。原生 `RenderApp` 的 Particle 物理桥仍未接入 `SceneFabric` runtime；三域也仍无 pipeline、bind group、draw/dispatch recorder 或 visibility/geometry consumer。这些是 owner/semantic/physical admission 三层证据，不是四域共享 Scene/Fabric GPU 帧或 full-frame parity；两条集成路径不能合并记作一个生产闭环。
- **跨适配器验收**：默认 `cargo test -p prism_render_scene --test cross_adapter_acceptance -- --nocapture` 在沙盒外发现 1 个 physical adapter（`Apple M2 / Metal / IntegratedGpu`）并输出 `SKIP_UNVERIFIED` 后结束；`PRISM_REQUIRE_GPU=1` 同命令以“需要至少两个 distinct physical adapters”失败。该失败是严格门禁的预期结果，不是通过，也不把单 adapter probe 写成跨 adapter parity。
- **已实现子项**：七值 class、旧符号/入口移除、CPU/WESL 公式统一与 overlay 消费已有当前源码证据。这些真实完成项应保留，不回退为未实现；同样不能扩大为 S4b/S5 全部要求完成。

- **2026-10-21 最新接线的尝试记录（用户反馈，非本次复跑）**：较早 native `fabric_frame` 运行 9 项为 **8 passed / 1 failed**，失败项 `render_schedule_publishes_gpu_particle_roles_and_revokes_disabled_routes` 在 `ParticleGpuState::new` 遇到 Null baseline fixture 的 Simulation `LimitExceeded`（`storage_buffers_per_stage requested 9 / limit 8`）。随后 96960 尝试在编译阶段以 E0308/E0599 两个错误停止，未运行测试；该尝试同样仅作历史记录，后续已有 `Features::default` 源码修复反馈，未复验结果。最新反馈的 86559 architecture 构建因 LLVM `No space left on device` 终止，**未运行测试、无新增通过证据**。本次文档同步不运行 Cargo、不清理其他任务正在构建/清理的共享 target；不能把旧错误列为当前源码错误，不能写成 9 passed，也不推定修复后通过。

### 15.3 完整验收仍需的证据

历史 `cancellation_waiters`、96960 编译错误、86559/76177 磁盘耗尽与较早域筛选中止分别记账（见 §15.2），不相互覆盖，也不推断当前源码仍有同一错误。这些尝试都不计作源码断言失败或测试通过；后续相应定向复验已完成 `render_fabric` 200/200（2026-10-22 main 实际复验）与独立 target 的 `legacy-bevy fabric_domains` 33/33，取代相应“待复验”状态。GPU admission 测试使用真实资源表/lease 语义，但未执行 shader、draw/dispatch 或 full-frame 输出，不能扩大为 GPU 闭环。

以下是原始目标的证据缺口，不是取消或缩减目标；生产接线仍在实施，其他代理的进行中变更不提前记为完成：

1. **全场景同一选择的生产调用闭环**：水体 compute 与 resident response composition 已各自有窄切片消费共享 `SceneFabric` selection，其中 composition 已通过通用 runtime 的 `compile_active → prepare_runtime`，并把真实 command buffer 交给宿主；2026-10-22 resident composition strict 3/3 已验证该局部 harness。仍需把 TAA/history、全域 payload 与其他域接入同一选择、真实 imports / pass recorders 和既有 RenderGraph。两个局部 recorder 与 3 项通过不能证明全场景链路。
2. **全域帧数据与资源生命周期**：各域 `FrameRenderItem` / payload 的真实提取与绑定、版本/barrier 消费，以及有 temporal 语义资源的 `FrameLag` history 分配、发布和导入。Surface `RenderFabricFrame` 已从 committed scene 提取并由 visibility 消费；native 另有真实 Surface/Particle 聚合、GPU Prepare 后语义刷新和 host graph admission/visibility 接线（v22），这些不能替代全域 GPU imports/recorder；Particle 的真实 app snapshot 已在原生 `RenderApp` 中降为校验后的 `ParticleFabricFrame`，会给出 count/bounds 与 stream/alive semantic `ResourcePayload`，并由严格 consumer 形成 CPU sprite geometry contract；`ParticleFabricGpuFrame` 进一步把 canonical roles 解析为 renderer-owned `BufferId`，并由本地 simulation/bridge/renderer graph 校验物理 owner 和 recorder 连续性；这仍没有接入 `SceneFabric` runtime 的统一 import/bind-group/record 或共享 recorder。Hair/Volume/Decal 已完成 render-world owner、validated semantic frame 与 physical `SceneRhi` admission/import 三层，定向筛选 33/33；仍没有 pipeline、bind group、visibility/geometry consumer、draw/dispatch recorder 或 full-frame parity。Surface/domain-builder 代码曾随 2026-10-20 `prism_render_scene --lib` 全量编译并通过 1909 项单测；该记录早于本次 seam。Particle 2026-10-21 定向结果为 `prism_particle_render` **6/6**、`prism_particle_app` **4/4**，此前 ECS 阻断已由 step-inspector **8/8** 修复。这些结果不证明统一 Scene/Fabric GPU payload 或共享 recorder 消费。water solver 仍导入本帧 surface/foam；spray feedback 已另行接入 per-instance exact lag-1 bindings、completion-validated import/publication enqueue、缺失历史显式清零，以及 resize/removal 后的 completion-gated retirement，并持有旧 buffer/RHI lease 至安全回收。该 water/spray 接线是局部 adapter，不是统一全域 recorder；TAA/history 与其他 temporal domain 尚未消费统一的全场景 history lifecycle。Composition 当前无 `FrameLag` accesses，不要求引入无消费者的 temporal publication。单 command buffer auto-sync 不替代这些责任或原生多队列 fence 契约。
3. **G3/G4、全量扇出及 32 门禁**：测试侧 ledger 和 built-in fanout 有历史 `3/3` acceptance 记录；当前工作树 Render Fabric contract/runtime 定向运行记录为 `200/200`（2026-10-22 main 实际复验），shader-package `95/95` 与 `prism_render_scene --lib` 最近完整记录 **1909/1909** 仍为历史，并未在最新接线上完整复跑。仍缺 7 组全域装配范围、NPR 不降级、全部静态 roots 的真实生产消费者及完整 32 门禁的端到端运行结果。catalog/fixture 覆盖和 crate 单测不等于高级特性质量或生产帧验收。
4. **完整水体行为等价与护栏时序**：冻结旧 shader closure 的八项 GPU 输出已通过；独立 TAA reactive MRT 已有当前生产接线与 shader/layout 定向测试，但完整 Scene/Fabric/TAA 帧 parity 与旧入口删除前护栏的历史运行记录仍缺失。
5. **ABI 与管线消费**：S5 ABI bump、package/cache stale rejection，以及冻结 `FeatureSelection → shader package / PermutationSpace → PipelineKey / PsoCacheKey / warm set` 的生产消费 seam 已有当前源码与先前运行记录（见 03）；2026-10-22 material package unit 888/integration 79（含 S5 cache 4、ShadingPlan 33）已关闭本 package 复验缺口。仍缺的是该 mapping 在完整 Scene/Fabric 每帧装配、真实 driver binary 与跨适配器范围内的端到端覆盖，不据此声称 architecture shader-package/acceptance 已完整复跑，也不重新选择或另建第二套 identity。
6. **跨适配器 GPU 证据**：本机只有一个 Apple M2/Metal physical adapter；`SKIP_UNVERIFIED` 与严格失败语义已固定，但第二个 physical adapter 上的 probe、water/Scene/TAA full-frame replay 和跨 adapter parity 仍未验证。

**结论**：本文保留已实现子项及所有原始要求、37 条禁止和 32 项门禁。2026-10-20 的 payload/history contract、S5 ABI/cache、冻结水体 GPU parity、水体 compute 窄 Fabric adapter，以及 resident composition 的局部 Fabric recorder；2026-10-21 的通用 runtime、composition 共享 selection/runtime、Particle 局部物理桥与 native 共享帧/finalize/admission/visibility，以及 Hair/Volume/Decal 的 owner/semantic/physical admission 三层，只关闭相应契约/局部生产缺口。当前工作树 Render Fabric 基础契约已复验 200/200（2026-10-22 main 实际复验），三域定向筛选 33/33；但统一 scheduled history publication、全域 fanout、pipeline/bind group/draw-dispatch、全域编译接线、跨适配器和**完整 Render Fabric 生产闭环仍未闭合**。此前构建/测试记录各自仅适用于对应版本。

### 15.4 最终收口验收矩阵 / 未完成项（2026-10-21 文档审计）

**2026-10-22 Hair/Volume/Decal 生产接线边界修正**：Hair 与 Decal 的 catalog 仍使用 **256B persistent STORAGE buffer** 泛化模板，每域仅声明 1 个 compute pass 和 3 个 `StorageWrite`，尚不匹配 strand arrays 或 decal receiver/material ABI；仍需真实资源/pass/access、生产数据、pipeline/bind group、draw/dispatch 与下游消费者。Volume 的 catalog 同样是泛化模板，不能代表现有 volumetric fog：`pkg/prism_render_scene/src/shading/volumetrics` 已有按 view 缓存的真实 3D froxel textures、scatter/integrate/apply pipelines、bind groups、scene depth/color 输入及 copy-back。Volume 真正未完成的是把该生产路径纳入冻结 SceneFabric 的真实纹理布局、pass 依赖/hazard、typed recorder 和宿主 record/submit；scene-color copy-back 尚未作为 Fabric pass。不得用占位 buffer、probe 或伪造输出冒充集成；`from_domain` 的局部 GPU 上传/dispatch/subrange 证据不覆盖 volumetrics 语义或全帧闭环。

**2026-10-22 Volume SceneFabric 接线后续**：上段关于 Volume 尚未接入 SceneFabric pass/recorder/copy-back 的状态已过时。当前 `shading/volumetrics/dispatch.rs` 已将生产 scatter、integrate、apply 与 fog-applied→scene-color transfer copy-back 声明并提交给 frozen SceneFabric/typed recorder；builtin catalog 声明真实纹理维度、格式、usage 与 pass 依赖。契约核对发现并修正 SSR depth 与 HDR scene-color apply 输入的物理 usage 不匹配。真实设备 dispatch/readback、设备 96-byte immediate 能力及 view lifecycle 已由下方 2026-10-23 strict GPU 记录验证；完整 schedule 的 completion-gated retirement 和全场景闭环仍未证明。

**2026-10-23 Volumetrics allocation-owner 检查**：生产 typed recorder 现在显式保留 `record_into` 返回的 `RecordedGraph` 至本地检查，并断言 graph allocations 为空，避免无声丢弃 `#[must_use]` 所有权包；当前资源由 view cache 持有，属于外部 persistent textures。本轮 `legacy-bevy` shader 定向测试 **2/2** 与 architecture volumetrics catalog 契约测试 **1/1** 已通过；前者编译了修改后的 scene lib-test，但不代表真实 recorder/GPU runtime 验收。该断言只防止本路径意外引入未托管的 transient graph allocation，不能代替真实 GPU completion/readback 或全帧生命周期证据。

**2026-10-23 Volumetrics immediate capability gate**：pipeline startup 核对实际 device `IMMEDIATES` 与 `max_immediate_size >= 96`；否则不创建 fog pipelines，并在 PrepareResources 清理 per-view 组件和缓存。新增单测最初的非 `Copy` wgpu limits 所有权错误已修复。当前 `legacy-bevy` scene library check 通过，Volumetrics ABI、shader 与 capability gate 筛选 **14/14**，architecture catalog pass/layout contract **1/1** 通过。测试目标重编因空间耗尽后清理可再生成 target **13.8 GiB**，以低占用配置完成上述验证。真实 GPU limits、SceneFabric dispatch/readback 与生命周期仍未验证。

**2026-10-23 strict Volumetrics GPU harness**：新增真实生产链测试，复用 production Volumetrics pipeline/layout 构造，准备 visibility/SSR/froxel 资源和 bind groups，调用 `volumetrics_pass`，由 shared SceneFabric runtime lowering/typed recorder 编码 scatter→integrate→apply→copy-back，并 readback fog target 与 scene color 检查有限输出、非真空改变及 copy-back 逐像素一致性。`PRISM_REQUIRE_GPU=1` 沙盒运行按设计拒绝无 adapter；沙盒外识别 Apple M2/Metal、`max_immediate_size=256`。设备 fixture 检查发现并纳入 Bevy visibility 所需 `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`。此前一次 strict 重跑在 test binary 前因 target 重编磁盘耗尽停止；之后以可用构建产物恢复严格运行，`shading::volumetrics::gpu_tests::production_scene_fabric_fog_dispatch_and_copy_back` **1 passed / 0 failed**。该项实际覆盖 scatter→integrate→apply→copy-back、fog target/scene-color readback、有限非真空 fog 输出及逐像素 copy-back 一致性，关闭 Volumetrics 单次 dispatch/readback 的硬件验证缺口；不覆盖 resize/disable/missing-input 生命周期、完整多域帧、31-pass parity、第二适配器或全场景闭环。

**2026-10-23 Volumetrics view lifecycle strict GPU**：`production_view_lifecycle_rebuilds_and_retires_froxel_resources` 在 Apple M2/Metal、`PRISM_REQUIRE_GPU=1` 下单独实跑 **1/1**。覆盖 viewport resize 后 texture ID 重建、visibility 缺失时 apply group 缺席且 pass 不录制命令、zero viewport 清理 per-view components/cache、disable 清空 cache。早先合跑被并行 geometry 改动从 `geometry/mod.rs` 移除 `representation` 声明/导出阻断；恢复模块与导出后，严格合跑 `shading::volumetrics::gpu_tests` **2 passed / 0 failed**。lifecycle harness 检查 view/cache 语义，不覆盖完整 schedule 中 GPU 在途资源的 completion-gated retirement。

**2026-10-23 Particle 原生路径 strict GPU 复验**：Apple M2/Metal 上 `PRISM_REQUIRE_GPU=1 WGPU_BACKEND=metal cargo test -p prism_render_app --lib physical_gpu_multioutput_cameras_execute_simulation_and_bridge_once -- --nocapture` 实跑 **1 passed / 0 failed**；另 `particle_fabric_` CPU 筛选 **5/5**。GPU 用例走原生 `RenderApp` 的 `ParticleGpuState → simulation → bridge → per-emitter renderer` 并跨多相机图执行/readback；CPU 筛选覆盖 semantic payload/geometry 与 renderer-owned `BufferId` 角色契约。二者均不能证明 Particle 已导入 Bevy SceneFabric runtime：`prism_render_app` 使用原生 Prism RHI/RenderGraph，而该 runtime 目前使用 Bevy `SceneRhi`，资源表、生产调度与提交尚未统一。完整跨域 import/recorder、共享全帧、以及完整 schedule 的 completion-gated resource retirement 仍未闭合。

**2026-10-23 Particle catalog 生产访问流更正**：此前 builtin Particle catalog 将 bridge stream 与 renderer alive 输出都建模为同一 compute pass 的 `StorageWrite`，不足以表达真实 `simulation → bridge → renderer` 的 SSA/hazard 顺序。当前拆为 `PARTICLE_BRIDGE_PASS`（stream `Automatic/StorageWrite`）和依赖它的 `PARTICLE_RENDER_PASS`（stream `ProducedBy(PARTICLE_BRIDGE_PASS)/StorageRead`、alive `Automatic/StorageWrite`），新增 catalog 契约测试；architecture `render_fabric` 筛选 **202/202** 实跑通过。该修正确保静态依赖模型与已存在的原生生产路径一致，不等于实际原生图已由共享 SceneFabric runtime 调度或 recorder 消费。

**2026-10-23 未发布 API 收口与编译验证**：移除标明 compatibility 的 `FixedPageStreamingManager` type alias、BindlessTextureHeap 的 `release_after()` alias 与 `WATER_SPRAY_RESOURCE` export；所有仓库调用改用 `FixedPageStreamingSource`、`retire()` 和所属域 canonical key `PARTICLE_SPRAY_RESOURCE`。`cargo check -p prism_render_scene --features legacy-bevy --lib` 通过，`prism_render_app --lib feature_mapping::tests` **12/12** 通过。为使该 target 能进入测试阶段，同步修正并行未完成变更中 6 个 surface-view fixture 的 `clear` 字段及 queue-stage fixtures 缺失的 `vt_layout` 参数；没有改变这些生产 API 的行为。

本节矩阵记录原审计截点（当时未运行 Cargo/GPU）及其后明确标注的复验（§15.2）：`render_fabric` 为 **200 passed / 0 failed**（2026-10-22 main 实际复验），`legacy-bevy fabric_domains` 为 **33 passed / 0 failed**；`prism_render_scene --lib` 的 **1909 passed / 0 failed** 仍为此前完整运行记录，不表示最新全库已复跑。2026-10-22 material unit 888/integration 79、RenderGraph unit 122/doc 1 passed / 1 ignored、resident composition strict 3/3、runtime multipass 9/9（8 hardware + 1 CPU fixture）与 TAA strict 19/19 各限日志中的范围，子集不重复相加。这些 crate 级与局部运行证据不改变全场景 Render Fabric `E` 列仍未闭环的结论，不等于 full-frame 或跨适配器 GPU parity。

2026-10-22 transfer recorder 复验补记：修复了 runtime texture handle 必须在 `RuntimePassContext` 阶段按声明 access 解析的问题，并修正 `graphics.rs` 的 `AccessKind` 导入与纹理布局字段引用。独立低调试 target 下 `cargo check -p prism_render_scene --features legacy-bevy --lib --jobs 1` 通过；`cargo test -p prism_render_scene --features legacy-bevy --lib fabric_runtime::graphics::transfer_tests --jobs 1 --config profile.test.debug=0 -- --nocapture` 实际 **1 passed / 0 failed**，覆盖 3D bounds、边缘及整数溢出。随后新增真实 SceneFabric copy/readback GPU 用例 `transfer_gpu_tests::real_scene_fabric_texture_transfer_copies_imported_pixels`；全 `prism_render_scene` test target 的 `cargo check --tests` 已通过，证明测试源码可编译。2026-10-22 该 GPU 用例从已存在的 scene test target 编译并在沙盒外以 `PRISM_REQUIRE_GPU=1 WGPU_BACKEND=metal` 实际运行：识别 Apple M2 / Metal physical adapter，SceneFabric transfer submission completion 后 readback 的目标纹理所有像素与源图案一致，**1 passed / 0 failed**。同一 binary 在沙盒内无 adapter 时会 skip；强制 GPU 则按预期失败 `no adapter`，故不能用 sandbox skip 记录代替硬件结果。Cargo 因临时盘空间不足的重编链接尝试中止，但未影响已存在测试 binary 的沙盒外运行。transfer typed copy 的局部硬件执行/readback 缺口已关闭；它仍不证明全域共享 recorder、全场景 31-pass/TAA parity、跨适配器或完整生产闭环。

| 收口条款 | S 源码 | D 测试定义 | R 本次 | H / G 已有证据 | E 全场景闭环 |
|---|---|---|---|---|---|
| S1：catalog、唯一 provider、硬闭包、active roots、payload/history contract | 类型、选择、payload、history lifecycle 与校验实现存在 | `render_fabric` 单测、compile/runtime fixture、32-gate ledger 已定义 | 当前工作树 `render_fabric` 200/200（2026-10-22 main 实际复验）；acceptance 3/3（2026-10-22 main） | acceptance 3 项为先前记录；gate 23 明确 `partial` | 否；fixture/CPU compiler 不等于生产帧 |
| S4：水体与 resident Composition 局部生产切片 | water compute、spray、composition 各有窄 recorder；composition 已共享 SceneFabric selection/runtime，真实 imports/command buffer 接线存在 | water/composition GPU harness 与 runtime gate 已定义 | 2026-10-22 resident composition strict 3/3（2 硬件 + 1 capability 拒绝）；runtime multipass 9/9（8 hardware + 1 CPU fixture）；历史 scene lib 1909/1909 | composition 的 2 项 Apple M2/Metal 硬件用例覆盖局部 recorder/HDR readback 与 identity，另 1 项为 HDR storage capability 拒绝校验；历史水体 19/19、RHI bridge 13 项各限原范围 | 否；三条路径尚未合并为统一全场景 recorder |
| 帧 producer 与域扇出 | Bevy Surface bounds producer 与 native Surface/Particle 聚合、post-GPU Prepare finalize、visibility/host admission 存在；Particle snapshot 有 semantic stream/alive `ResourcePayload`、严格 CPU geometry consumer，以及 renderer-owned `BufferId` 物理桥和本地 graph recorder 校验；Bevy Hair/Volume/Decal 有稳定 owner、descriptor/resource contract、共享帧聚合、physical GPU admission/import。Volume 的真实 fog pass 已有 SceneFabric typed recorder 接线（scatter/integrate/apply/copy-back） | 域 identity、runtime snapshot、payload/geometry、owner lifecycle、resource identity/layout/usage/import coverage/alias 测试有定义；Volume 与 Particle 均有局部 GPU/lifecycle 验收 | scene lib 1909/1909（早于本 seam）；三域筛选 33/33；Volumetrics capability/ABI/shader 14/14、catalog contract 1/1、production dispatch/readback strict GPU 1/1、lifecycle strict GPU 1/1；Particle semantic CPU 5/5、production multiview strict GPU 1/1（Apple M2/Metal） | Volume 与 Particle 局部 GPU 子集通过；不证明完整渲染 schedule 的队列退休或全场景帧 | 否；Particle 尚未统一 import/recorder；Volume 完整 schedule completion retirement 尚缺；Hair/Decal 缺真实语义 pipeline/consumer；无全域 full-frame 扇出 |
| 对象级 runtime 局部切片 | exact object/generation/history/role 地址、payload upload 与子分配 preflight 存在 | Surface 对象隔离 fixture 已定义 | 2026-10-22 main strict multipass 9/9（8 hardware + 1 CPU fixture），exit 0 | Apple M2/Metal 对象子分配真实 dispatch/readback；历史 8/8（7 hardware + 1 CPU fixture） | 否；该 multipass 子集不包含 `from_domain`；另有桥接局部通过记录，仍不含三域语义消费者或全帧 |
| TAA/history | 真实双槽、exact `(resource, frame, version, physical, layout)`、completion publication/retirement 源码存在 | history、TAA ABI/shader 与 GPU lifecycle gate 已定义 | 2026-10-22 main strict GPU 19/19（含 lifecycle 7 项） | Apple M2/Metal 真实 bind-group、publication/import、queue completion 与资源 retirement 已验证，见 §15.2 | 否；局部 GPU 缺口已关闭，统一 runtime/recorder 与 full-frame parity 仍未闭合 |
| S4b/S5：水体等价、旧入口收口、ABI/cache | 当前入口/class/公式/overlay/selection-to-PSO seam 存在 | 8 项 legacy parity、legacy gate、ABI/cache gate 已定义 | 2026-10-22 material unit 888、integration 79（含 freshness 19、ShadingPlan 33、S5 cache 4）；其他 gates 不推定已复跑 | 历史单 adapter 水体 19/19、旧符号静态 gate 与历史记录；当前 package 通过不能证明删除前时序 | 否；完整 Scene/Fabric/TAA parity 未证 |
| S6、32 gate、跨适配器与性能 | 32-gate fixture、strict/skip adapter probe、部分 fanout contract 存在 | acceptance/adapter gate 已定义；全系统性能验收尚无完整门禁证据 | 无 | 单 adapter；少于两个 adapter 时 `SKIP_UNVERIFIED`，不计通过 | 否；无第二 physical adapter、无全场景 P50/P95/P99 |

仍未验证的证据清单如下：

1. `prism_render_architecture --lib render_fabric` 当前工作树已复跑 200/200（2026-10-22 main 实际复验）；仍需 architecture acceptance 与其他相关 integration targets 的当前完整复验。2026-10-22 `prism_render_graph` 全 package unit 122/122、doc 1 passed / 1 ignored，material package unit 888/integration 79，以及 resident composition strict 3/3 已分别关闭各自待复跑缺口，不再列为未运行；局部通过不覆盖完整 32 gate；
2. TAA 真实 bind-group、队列 completion、resize/removal/device-table replacement 的局部 GPU 运行已由 2026-10-22 strict 19/19 关闭；仍需完整 Scene/Fabric/TAA 图像 parity 与统一全场景 runtime/recorder，不能以局部通过替代；
3. 统一 `SceneFabric` 下跨 Surface、Particle、Volume、Hair、Decal、response root 的 `compile → prepare_runtime → import → record/submit` 生产调用；
4. native Surface/Particle 共享帧、post-GPU Prepare finalize、visibility/admission 与 Particle semantic `ResourcePayload` / CPU geometry consumer、renderer-owned `BufferId` 物理桥和本地 graph recorder 校验源码已存在；本轮复验 `particle_fabric_` CPU **5/5**、Particle production multiview strict GPU **1/1**（Apple M2/Metal）。Bevy Volume/Hair/Decal 的 render-world owner、共享生产帧与 physical GPU admission/import 已通过 33 项筛选。仍需 Particle 接入统一 `SceneFabric` runtime 的 resource import/binding 与共享 recorder 消费；当前 `prism_render_app` 原生 RHI/RenderGraph 与 `prism_render_scene` Bevy `SceneRhi` 是不同设备表/生产路径，不能直接混用 handle。Volume 的 production dispatch/readback 与 per-view lifecycle 已有 Apple M2/Metal strict 用例（各 1/1），但完整 schedule 的 completion-gated resource retirement 和全帧闭环仍未验证；Hair/Decal 仍需真实语义资源布局、producer、pipeline/bind group、draw/dispatch 与下游 visibility/geometry/shading 消费；
5. 当前工作树完整 32 gate 的端到端运行、gate 23 从 `partial` 到闭合的证据，以及 S2/S4/S5 历史先后顺序的可追溯护栏记录；
6. 第二个 distinct physical adapter 上的 probe、water/Scene/TAA replay 与 parity；
7. 全场景输出 parity、显存/资源峰值、队列等待、吞吐和 P50/P95/P99 性能数据。

因此，当前可写成“源码契约和若干局部生产切片已存在，部分历史/单适配器证据已记录”；不可写成“Render Fabric 或整个重构系统已完成”。
