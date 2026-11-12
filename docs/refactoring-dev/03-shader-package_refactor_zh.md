# 03 · Shader Package 与置换治理

> **2026-11-09 目标边界（当前权威）**：Shader package、PSO、队列与缓存的目标消费者是 Prism 原生 prism_render_app、Prism RHI 与 prism_render_graph。Bevy、Bevy RenderApp、prism_render_scene 的 Bevy 集成路径、SceneRhi 与 legacy-bevy 仅为历史/排除项，不是 shader package、driver binary 或验收闭环的实现前提。

## 当前审计结论（2026-11-12，native Prism）

**已确认**：native package identity、permutation/descriptor 校验、PSO/cache key、queue/lifecycle 契约以及 Rust/WESL ABI 检查已有源码和局部测试证据；Metal 非空场景 surface encode/GBuffer 与固定 composite readback 均已通过，composite ABI 已收敛到 Metal 可用的 storage-buffer 预算。它们证明身份和局部 command path 的约束，不证明整个 shader package 已产生正确画面。

**当前缺口**：Metal native 的基础直接 draw 与非空场景 package → bind group → surface-encode → GBuffer 最小路径均已通过；2026-11-12 又通过固定 Unlit package 的 identity → composite → HDR 与 completion-gated resize lifecycle，场景验收确认真实 `draws_queued=1`、`draws_recorded=1` 及 normal/albedo/HDR 输出。仍未完成的是完整 response package 消费、统一全帧 dispatch/draw、未就绪/失败/设备丢失/resize/退休的全矩阵、跨适配器输出与 P50/P95/P99。另一个 `native_full_frame_acceptance` 只使用 `DeferredInstance`/Null backend 和无对象空相机，不能验证真实 driver binary 的完整帧。

本篇的 CPU/null/cache/contract 结果以及 Bevy/SceneRhi 历史记录必须保持范围标注。完整 response package 消费、未就绪/失败/设备丢失/resize/退休、统一全帧 dispatch/draw、跨适配器输出与 P50/P95/P99 仍是未完成项。

> 文档口径：以当前源码、测试和已合入历史为准的实现审计。本文描述当前身份模型、缓存生命周期、队列入口与必须保持的架构约束。
> 上层导航：[`00-overview_refactor_zh.md`](00-overview_refactor_zh.md)
> 相关文档：[`02-response-composition_refactor_zh.md`](02-response-composition_refactor_zh.md)、[`04-frame-graph_refactor_zh.md`](04-frame-graph_refactor_zh.md)

## 1. 当前实现基线

| 事实 | 当前源码 | 结论 |
|---|---|---|
| 热路径身份 | `pkg/prism_render_app/src/pipeline_cache.rs` 的 `PipelineKey` 是 `PipelineIdentity<PipelineState>` | 身份由 `execution_family`、`response_module_id`、`permutation_index`、`typed_pipeline_state` 四个具名分量组成 |
| 身份定义 | `pkg/prism_render_architecture/src/shader_package/execution.rs` | `PipelineIdentity` 是按 pass/module 查询的语义 key，不是运行时回写容器；`ExecutionFamily` 有 3 个变体：`FusedBuiltin`、`SplitResponse`、`Composition` |
| 固定功能状态 | `pkg/prism_render_app/src/pipeline_state.rs` | `PipelineState` 保留完整顶点布局、光栅化、深度/模板、多重采样和颜色目标状态；`to_hash()` 只是 `DriverLocal` 编码，语义比较使用完整结构体 |
| Shader package 句柄 | `pkg/prism_render_architecture/src/shader_package/mod.rs` | `ShaderPackageId` 是进程内驻留的数字句柄；文本只保留在冷端注册表，热 key 不做字符串比较 |
| 冷端 PSO key | `pkg/prism_render_architecture/src/shader_package/pso_cache/mod.rs` | `PsoCacheKey = package + permutation_index + PipelineStateHash`，用于确定性排序、预热、持久化和驻留 |
| 热/冷映射 | `pkg/prism_render_app/src/pipeline_cache.rs` | `cold_key_from_pipeline_key` 将完整 typed state 映射为 hash；`PipelineCache` 保存 typed key 与 cold key 的双向观测关系，并拒绝未注册、冲突和歧义映射 |
| 解析结果 | `pkg/prism_render_app/src/pipeline_cache.rs` | `PipelineResolution` 保留 `Ready`、`Fallback` 两个绑定结果；请求状态另由 `PipelineLookupStatus` 区分 `Ready`、`Fallback`、`Miss` |
| 编译队列 | `pkg/prism_render_app/src/pipeline_cache.rs` | job/result 均使用有界 `sync_channel`；默认 `worker_count = 4`、`queue_capacity = 64`，worker 按配置创建，队列满时保持 fallback 并延后重试 |
| 编译入口 | `pkg/prism_render_app/src/mesh_pipeline.rs`、`pipeline_pass.rs` | `MeshPipelineSpecializer` 从 `MeshPipelineKey` 和 descriptor 生成完整 `PipelineKey`；`PipelinePass` 是 direct/bin 两条路径共用的 pass 描述入口 |
| 回退策略 | `pkg/prism_render_app/src/mesh_pipeline.rs` | `full_lobe_fallback` 使用模块声明的 checked dense fallback index，并保留其余 variant 字段；异步编译期间由 `PipelineResolution::Fallback` 保持可绘制性 |

## 2. 管线身份与置换边界

当前身份可写成：

```text
PipelineKey = {
    execution_family,
    response_module_id,
    permutation_index,
    typed_pipeline_state,
}
```

粒度是 `(pass, module)`，不是“每材质一份 key”。split response 的一个 pass set 由左响应、右响应和组合内核组成；fused 路径则由一个 surface pass 描述。`PipelineIdentity` 直接保留 `ModuleId` 和密集 `PermutationIndex`，因此模块身份、置换索引和固定功能状态不会再借用压缩 class 互相编码。

`PipelineState` 的字段由 descriptor 显式解构得到。新增 RHI 固定功能字段会先触发结构体覆盖检查，再进入相等性、排序、哈希和 canonical bytes；`PipelineStateHash` 不能被当作可逆身份，也不能替代完整状态校验。

### 2.1 静态轴与动态数据

判据是“改变该值是否改变编译产物”：

| 内容 | 归属 |
|---|---|
| 响应模块、执行家族、置换索引 | package/module identity |
| 顶点布局、光栅化、深度/模板、MSAA、attachment targets | typed pipeline state |
| 材质参数、质量降级参数、mip、逐像素区域选择、连续组合遮罩 | uniform、纹理、顶点属性或间接参数 |
| `BlendDomain` 的结构形态 | 契约枚举；当前有 3 个变体：`FullRadiance`、`DirectOnly`、`CustomChannelMask(ChannelMask)` |
| `RenderPhase` | 帧图桶；当前有 4 个值：`Opaque`、`Transparent`、`Stylized`、`Custom` |

质量档不能通过运行时换 PSO 实现；如果只是改变采样预算或参数，应沿用同一置换。`LightingContract` 描述模块编译能力，逐帧灯光裁剪、阴影页就绪和样本值属于数据。

### 2.2 FeatureSelection 到 PSO identity 的冻结映射闭环

这一条链现在已经由冻结选择驱动，而不是在 PSO 请求处重新选择：

```text
FeatureSelection
  -> FrozenShaderSelection
  -> SelectedFeatureShaderBinding
  -> checked PermutationIndex
  -> FeaturePipelineIdentity
  -> PipelineKey + PsoCacheKey
  -> PipelineCache / WarmRuntime / WarmSetPlanner
```

- `FeatureSelection::bind_shader_packages` 要求 package map 与已选 `VariantId` 精确相等；缺少绑定、额外绑定、package/module 不一致、PermutationSpace/key 不一致或 pass descriptor 不一致都会拒绝。
- `PermutationSelector::finish_in(&PermutationSpace)` 与 `ShaderPermutationBinding::checked_index()` 共用现有 `PermutationSpace`，dense index 不能由调用方另行填写或伪造；空 permutation space 只代表唯一的 canonical empty permutation。
- `FrozenShaderSelection` 保存选择结果、完整 package/module/permutation/pass binding 和稳定 digest。`PipelineCache::register_frozen_selection`、`require_frozen_selection_warm`、`resolve_frozen_selection_or_request` 只能从这个冻结对象派生真实 package/permutation/PSO identity，不会重新调用 feature resolver，也不会枚举整个 permutation space。
- app 侧以 `(feature, variant, driver module)` 记录 binding fingerprint；同一 slot 的 package digest、permutation/pass declaration 或 selection generation 变化会返回 `FeatureBindingChanged`，不会静默复用 stale cache。
- warm set 只加入冻结选择对应的 exact `PsoCacheKey`，最终排序仍由 `WarmSetPlanner` 按 priority 和 key 确定性完成；catalog/package/registration order 不改变同一冻结选择的 identity。

这完成的是 `FeatureSelection -> shader package/permutation -> PSO identity/cache/warm set` 的冻结映射与生产调用 seam。完整 Scene/Fabric 的每帧生产接线、跨适配器 GPU parity 和真实 driver binary 覆盖仍属于后续验收，不在本闭环的完成声明内。

## 3. 缓存、持久化与线程边界

### 3.1 热端

`PipelineCache` 负责已创建 `RenderPipelineId` 的内容缓存、pending/waiting/in-flight/failed 集合、fallback、预热请求和结果回收。异步路径由 `PipelineCompileConfig` 限制并发和等待容量；没有可用 worker 时走同步创建路径，避免丢失请求。

### 3.2 冷端

`shader_package/pso_cache` 提供：

- `WarmRuntime`：`Absent → Queued → Compiling → Ready/Failed`，draw 请求仅返回 `Draw` 或 `Skip`；`Ready` 才能绘制，未就绪时跳过，`Absent`/`Failed` 会重新入队并把 miss 回馈给 `WarmSetPlanner`；
- `WarmSetPlanner`、`PackageWarmSpec`：按 package、密集 permutation 和状态生成确定性 warm set；
- `LruPsoCache`：可选字节预算驻留；
- `DeviceFingerprint`：绑定 backend、vendor/device、driver 和 ABI；
- `PersistedPsoCache`：`PPSO` 线格式、版本、设备指纹、ABI hash、按 `PsoCacheKey` 排序的 entry 和尾部校验。

架构层只定义确定性模型和线格式；引擎侧通过 `PipelineCache` 的导入/导出 seam 负责实际 driver binary。`DriverLocal` 与可移植 `asset_refs` 分离：package/program 内容引用可进入构建产物，设备/驱动绑定的 PSO 映射和 blob 只进入 DriverLocal。

## 4. Direct/bin 入口与 pass set

`pkg/prism_render_app/src/pipeline_pass.rs` 的 `PassDescriptor` 是两条绘制路径共用的入口，包含：

- `PassRole`：`Surface`、`ResponseLeft`、`ResponseRight`、`Composition`；
- `ExecutionFamily`、`ModuleId`、`PermutationIndex` 和可选 fallback permutation；
- `RenderPhase`、draw function 和独立的 `PassDescriptorClass`。

`PassDescriptorTable` 只接受两种合法形状：一个普通 surface pass，或左/右两个 split response 加一个 composition pass。组合 pass 必须携带自己的 draw function；split pass set 的组合内核是常驻形态。

bin 路径位于 `pkg/prism_render_app/src/bin_queue.rs`：

1. `DrawBinKey.pipeline_class` 查找 `SceneRegistry::pass_descriptors`；
2. 对描述符表中的每个 pass 生成一个 `MeshPipelineKey`；
3. 每个 pass 绑定自己的 pipeline、phase、draw function，但复用同一个 bin 的 indirect argument range；
4. 因此普通 bin 产生 1 个 work item，split bin 产生 3 个 work item；测试 `split_bin_expands_to_three_work_items_with_one_indirect_range` 固定这一契约；
5. 未知描述符表索引使用 `SceneRegistry::fallback_pass`，增加 `unknown_class` 诊断，不静默丢弃绘制。

排序用的 `PassDescriptorClass` 是独立的 `u8` 粗分类，`255` 保留为耗尽哨兵；它不是 `DrawBinKey.pipeline_class` 的表索引，也不是 `PipelineKey` 的编码。

## 5. 必须保持的架构约束

1. `PipelineKey` 只用于查询和身份匹配，不写回运行时解析结果。
2. 完整 `PipelineState` 参与相等性和 canonical 校验，hash 只用于冷端索引与存储编码。
3. typed key、cold key、module/package 的映射出现未注册、冲突或歧义时显式报错，不猜测反向映射。
4. `asset_refs` 保持后端无关；PSO、driver fingerprint、native pipeline binary 属于 DriverLocal。
5. 异步编译保持有界内存和非阻塞渲染线程；`WarmRuntime` 对未就绪 PSO 仅返回 `Skip`，不提供 placeholder。后端若另选真实已编译的 fallback PSO，仍须以该 PSO 自身的 key 通过就绪门禁后才能绘制。
6. 一个材质的 pass set 由 descriptor table 描述；split 形态固定为两个 response arm 加一个 resident composition pass。
7. `PassDescriptorClass` 不得回绕；表索引、粗分类和 pipeline identity 三者保持不同语义。

## 6. 当前验收证据

本节涉及的 scene、Bevy 或 SceneRhi 记录属于历史/排除路径；native shader package 结论不能由这些记录升级。

| 契约 | 证据 |
|---|---|
| typed state 覆盖 descriptor 固定功能字段、hash/canonical bytes 稳定 | `pkg/prism_render_app/src/pipeline_state_tests.rs` |
| typed key ↔ cold key、package/module 冲突、warm runtime 生命周期、LRU 与 driver binary seam | `pkg/prism_render_app/src/pipeline_worker_tests.rs` |
| FeatureSelection 精确绑定 package、PermutationSpace/selector、pass descriptor，冻结 digest 不受注册顺序影响 | `pkg/prism_render_architecture/src/render_fabric/tests.rs`、`pkg/prism_render_architecture/src/shader_package/permutation.rs` |
| 冻结选择生成真实 `PipelineKey`/`PsoCacheKey`，warm set 只消费 selected permutation，并拒绝同 slot stale binding | `pkg/prism_render_app/src/pipeline_worker_tests.rs`、`pkg/prism_render_app/src/pipeline_cache.rs` |
| direct/bin specialization、full-lobe fallback、prewarm、split bin 三 work item | `pkg/prism_render_app/src/tests.rs` |
| shader package interning、manifest/variant、PSO cache wire model | `pkg/prism_render_architecture/src/shader_package/` 及其单元测试 |
| asset refs 与 DriverLocal 隔离、variant manifest 完整性 | `pkg/prism_asset_build_core/src/shader_package.rs` |
| app 侧 ABI 指纹与持久化缓存失效 | `pipeline_cache.rs` 将生成 Surface layout、material record/class 与 response ABI 纳入 `DeviceFingerprint`；`pipeline_worker_tests.rs` 的 schema/layout/class/response/legacy fingerprint 回归 |
| 当前 shading crate 测试 | `cargo test -p prism_render_scene --lib shading`：844 passed，0 failed（历史记录） |

2026-10-20 历史记录：`render_fabric` 较早 134 项、后续 136 项通过；app `pipeline_cache::worker_tests` **27/27**。当时一次 architecture 重试因并发 `prism_physics_core` 编译错误未执行测试，该阻断仅为历史。2026-10-22 architecture 代理已报告实际运行 `prism_render_architecture --lib render_fabric` **200/200、terminal exit 0**，含新增对象资源 CPU 契约 9 项；同日较早 176/176 保留为历史。仅关闭此 `render_fabric` 筛选的待复跑状态，不声称整个 `shader_package`（95/95 仍为历史）、app worker、architecture acceptance 或真实 driver binary 已复跑，也不关闭完整 Scene/Fabric、跨适配器与性能验收。

### 6.1 2026-11-09 原生 Prism 当前未完成项

当前身份模型、冻结选择和 CPU/cache 契约已有源码与局部记录，但以下 native 证据仍缺：

1. 在 prism_render_app + Prism RHI + prism_render_graph 中完成冻结 FeatureSelection 到真实 pipeline/bind group/draw/dispatch 的全帧消费，覆盖 split pass set 与 composition，不把 Bevy/SceneRhi path 的消费当作替代。
2. 取得真实 driver binary 的编译、加载、失效、回退、设备丢失和重建结果；验证 package digest、typed PipelineState、cold key、DriverLocal cache 在至少两个 physical adapter 上的隔离与复用边界。
3. 对异步编译队列、warm set、fallback/skip、PSO 驻留和资源退休记录内存上限、等待、命中率、编译时延、显存和 P50/P95/P99；当前 worker/CPU 测试不是性能结论。
4. 将 shader package 的 native 消费放进完整 Surface/response/TAA/RenderGraph 帧并取得图像 parity；局部 water、composition 或历史 scene library 数字不构成该证据。

## 7. 历史依据

- `6ab2e626d feat(render): close shader package persistence lifecycle`：引入热端与持久化生命周期、模块/包映射及 worker 测试覆盖。
- `8400fe972 feat: complete runtime and render refactoring implementation`：将渲染运行时、shader package、pass set 和测试证据合入统一实现。
- `5ad60f878 test(render): name refactor acceptance gates`：将关键重构验收固定为命名测试。

本篇结论：当前 shader package 的身份、typed render state、interned package handle、冻结 FeatureSelection 到 package/permutation/PSO identity 的映射闭环、有界编译队列、冷热缓存映射、DriverLocal 边界和 direct/bin/split pass set 已由源码与已标注的运行记录形成对应证据；后续变更必须保持上述语义分层，并继续把完整 Scene/Fabric 接线与跨适配器证据作为独立验收项。

## 8. 最终收口验收矩阵 / 未完成项（2026-10-21 文档审计）


**2026-10-22 最新复验与状态修正**：architecture 改动已提交 `652404b94`。main 对当前架构源码实际复验 `render_fabric` **200 passed / 0 failed / 0 ignored**、`shader_package` **95 passed / 0 failed / 0 ignored**、acceptance **3 passed / 0 failed / 0 ignored**；日志与范围见 05 §15.2。本文较早的 185 项及 shader-package/acceptance“未复跑”描述仅代表此前截点，以本补记为准。acceptance 的 3 项验证 ledger/fanout/partial 标记，不是完整 32 gate 端到端通过。

**局部 GPU 桥接证据修正**：既有日志已记录 `fabric_runtime` 定向筛选 **29 passed**（CPU/GPU 混合，不能全部算硬件用例），其中 authenticated `from_domain` bridge 在 Apple M2/Metal 完成真实上传、dispatch 和子范围隔离；RHI pipeline bridge 筛选 **15 passed**，包含 exact native render-pipeline identity/generation/lease 与原生绘制。这里只关闭相应局部桥接验证，不能关闭 Hair/Volume/Decal 的真实语义渲染、typed graphics 全链、统一 Scene/Fabric/TAA full-frame、第二 adapter 或性能验收。

**2026-10-21 收口补记（历史截点）**：此前 `prism_render_scene` library `cargo check` 曾通过；一次后续尝试曾因 `prism_asset` 的 `Flight` 初始化缺少 `cancellation_waiters` 字段而停止。这是历史阻断，当前 `Flight` 定义已无此字段，后续反馈已越过 asset 初始化；本次文档同步未运行 Cargo，不据此判断当前编译结果。当时没有复跑 shader package/architecture acceptance，也没有增加真实 driver binary 或跨适配器证据；2026-10-22 仅更新下述 `render_fabric` 筛选，其余未完成项保留。另，本轮仅核对 `pkg/prism_render_architecture/src/shader_package/pso_cache/warm_runtime.rs` 源码：`DrawDecision` 已仅有 `Draw`/`Skip`，`request_draw(&PsoCacheKey)` 已无 `allow_placeholder` 参数或 `Placeholder` 分支；未就绪 PSO 一律跳过，真实 fallback PSO 仍须按自身 key 通过就绪门禁。这只修正源码事实，不新增测试通过记录，也不扩大生产接线、GPU 或全场景验收范围；下列未完成验收目标全部保留。

**2026-10-22 局部证据补记**：main 实际复验 `prism_render_architecture --lib render_fabric` **200/200、terminal exit 0**（含新增 9 项对象资源 CPU 契约；旧 **176/176** 为历史，不重复相加）。main 实际运行并确认 exit 0 的 strict multipass 为 **9/9 = 8 项 Apple M2/Metal 硬件 + 1 项 CPU fixture**，本次已核对 `/private/tmp/prism-fabric-object-runtime-gpu-20261022.log`；旧 **8/8 = 7 hardware + 1 CPU fixture** 为历史。新增对象 fixture 覆盖 exact object/generation/history/role 寻址、完整上传/错误 role 拒绝与子分配隔离；不代表 `from_domain` 桥接 GPU、三域语义渲染、整个 shader-package/acceptance 或完整帧验收。

**2026-10-22 Hair/Volume/Decal 生产接线边界修正**：Hair 与 Decal 的 catalog 仍使用 **256B persistent STORAGE buffer** 泛化模板，每域仅声明 1 个 compute pass 和 3 个 `StorageWrite`，尚不匹配 strand arrays 或 decal receiver/material ABI；仍需真实资源/pass/access、生产数据、pipeline/bind group、draw/dispatch 与下游消费者。Volume 的 catalog 同样是泛化模板，不能代表现有 volumetric fog：`pkg/prism_render_scene/src/shading/volumetrics` 已有按 view 缓存的真实 3D froxel textures、scatter/integrate/apply pipelines、bind groups、scene depth/color 输入及 copy-back。Volume 真正未完成的是把该生产路径纳入冻结 SceneFabric 的真实纹理布局、pass 依赖/hazard、typed recorder 和宿主 record/submit；scene-color copy-back 尚未作为 Fabric pass。不得用占位 buffer、probe 或伪造输出冒充集成；`from_domain` 的局部 GPU 上传/dispatch/subrange 证据不覆盖 volumetrics 语义或全帧闭环。

**2026-10-23 Volume 状态更正**：上段关于 Volume 尚未进入 SceneFabric 的描述已过时。真实 scatter/integrate/apply/copy-back pass 已接入 frozen SceneFabric typed recorder，并有 96-byte `IMMEDIATES` 能力门禁；Apple M2/Metal 上 production dispatch/readback 与 lifecycle 两个 strict GPU 用例分别 **1/1** 通过。生命周期覆盖 per-view/cache 行为，不等同于完整 schedule 下的 completion-gated resource retirement；此局部进展不关闭完整帧 PSO 消费验收。Hair/Decal 与全场景缺口不变。

**2026-10-23 Scene frozen-selection ingestion 与 queue PSO 消费接线**：`SceneRegistry::insert_material_with_frozen_selection` 从 `FrozenShaderSelection` 和 host bindings 生成并原子安装 pass set，保留 checked provenance；direct queue 按 material/variant/phase 唯一解析 pass，bin queue 使用其共享 pass table descriptor。`MeshPipelineSpecializer::specialize_frozen_pass` 校验 pass/module/family/permutation 一致性，从 pass metadata + descriptor 完整 `PipelineState` 派生 `FeaturePipelineIdentity`，并将 package/module 与 typed/cold key 注册进既有 `PipelineCache` 后才进入相同 pipeline specialization/cache 路径。冷编译 fallback 不占用或改绑 frozen feature slot：它保留所选 package provenance，单独注册经模块声明校验的 fallback permutation 与其 descriptor state 的 typed/cold key。非 frozen 的 spine pass 继续走普通 specializer 路径。`cargo check --offline -p prism_render_app --lib --jobs 1` exit 0；frozen ingestion/direct-bin pass-table **1/1**、direct/bin 实际 specialization 共享 pipeline **1/1**、frozen selection warm-key 与 stale digest rejection **2/2**、fallback cold-key 独立注册且 selected permutation 保持锁定 **1/1** 均通过。以上 CPU/null backend 测试不代表真实 driver binary、完整帧和跨 adapter 验收，这些仍未完成。

**2026-10-23 frozen bin provenance 防降级复验**：bin queue 现在先读取代表实例的材质身份；若 frozen material 对应的 `pipeline_class` 不等于 registry 内该材质的 pass-table index，则在 unknown-class spine fallback 前硬拒绝；已知 pass table 中任一 frozen pass 缺少 `shader_package` provenance 也硬拒绝。`tests::unknown_class_is_zero_or_loud` 在独立 Cargo target 实跑 **1/1**：普通未知 class 仍保留可诊断 fallback 行为，frozen material 的 foreign table 则触发预期拒绝；连同 `scene::tests::frozen_material_rejects_missing_pass_table_and_package_provenance` **1/1**，关闭这两个 queue seam 的 CPU/null-backend 回归缺口。不是 GPU driver binary、跨 adapter 或完整帧证据。

`S`=源码事实，`D`=测试定义，`R`=本轮运行通过，`H`=已有历史/先前记录，`G`=GPU 证据，`E`=全场景闭环。`prism_render_scene --lib` 1909/1909 为历史完整运行记录；architecture acceptance、driver binary 和跨适配器 GPU 仍未由此覆盖。

| 条款 | S | D | R | H / G | E |
|---|---|---|---|---|---|
| typed `PipelineKey`、完整 `PipelineState`、cold key 与 package handle | 当前四分量身份、typed state、hash/canonical 边界存在 | state/cache/worker gate 已定义 | 历史 scene lib 1909/1909；shader_package 95/95、acceptance 3/3（2026-10-22 main） | app worker 与 architecture 的已有记录 | 否；尚未由全场景帧消费证明 |
| Frozen FeatureSelection → permutation → PSO/cache/warm set | 冻结选择、checked dense index、stale binding rejection 和生产 seam 存在 | selector、mapping、cache worker gate 已定义 | render_fabric 200/200（2026-10-22 main 实跑 exit 0）；shader_package 95/95 已于 2026-10-22 main 复验；scene lib 1909/1909 为历史 | app worker `27/0` 为既有记录 | 否；不含完整 Scene/Fabric 每帧装配 |
| direct/bin/split、fallback、异步队列与 DriverLocal | 代码和测试夹具存在 | pass/cache/queue gate 已定义 | 无 | 仅有历史运行记录；无本次复核 | 否；真实 driver binary 与跨 adapter 未覆盖 |

仍未验证：architecture `shader_package`/acceptance 的当前完整重跑（不含已更新的 `render_fabric` 200/200）；完整 Scene/Fabric 每帧由冻结选择生成并消费真实 PSO 的端到端路径；真实 driver binary、跨适配器 cache/PSO 行为；完整水体/Scene/TAA GPU replay；以及编译队列、显存、缓存命中率和 P50/P95/P99 性能数据。历史 `scene lib 1909/1909`、`shader_package 95/0`、`render_fabric 136/0`，以及本次局部 `render_fabric 200/200`、multipass 9/9 都不能替代全场景闭环。

**2026-11-09 Water 入口同步**：Water 生产路径不再保留独立 recorder/ledger，真实 dispatch 由顶层 coordinator 使用冻结 SceneFabric selection 接收；Water 定向测试 **230/230** 通过。真实 driver binary、跨 adapter PSO/cache、全场景冻结选择消费与完整性能数据仍未验证。

**2026-11-09 原生 composite 修正补记**：Registered response 在 Native composite 的左右臂现在使用独立 program identity；该修正只涉及已生成 shader 的正确性，不等于 driver binary 导入/导出、全场景 PSO 消费、跨适配器 cache 行为或性能验收已完成。
