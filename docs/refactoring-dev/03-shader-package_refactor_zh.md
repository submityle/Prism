# 03 · Shader Package 与置换治理

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

- `WarmRuntime`：`Absent → Queued → Compiling → Ready/Failed`，draw 请求返回 `Draw`、`Placeholder` 或 `Skip` 决策，并把 miss 回馈给 `WarmSetPlanner`；
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
5. 异步编译保持有界内存和非阻塞渲染线程；未就绪 draw 使用明确 fallback/placeholder 语义。
6. 一个材质的 pass set 由 descriptor table 描述；split 形态固定为两个 response arm 加一个 resident composition pass。
7. `PassDescriptorClass` 不得回绕；表索引、粗分类和 pipeline identity 三者保持不同语义。

## 6. 当前验收证据

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

2026-10-20 先前完整运行记录：`cargo test -p prism_render_architecture --lib render_fabric -- --nocapture` 为 **134 passed，0 failed**；`cargo test -p prism_render_app --lib pipeline_cache::worker_tests -- --nocapture` 为 **27 passed，0 failed**。本次收口再次运行 architecture target 时未到达测试执行阶段，被并发 `prism_physics_core` 工作树编译错误阻断（缺失 `index` 模块、跨 crate 私有模块访问及 tearing API 不匹配），因此没有新的通过记录。已有验收覆盖 selector/space 到 checked dense index、精确 package map、registration-order 稳定性、冻结选择到真实 pipeline/cold key/warm set 映射，以及同一 feature/variant/module slot 的 stale binding 拒绝。app 侧 ABI 变更拒绝测试仍保留；映射 API 与生产调用 seam 已接入，并有先前完整运行记录，本次收口未重新获得 architecture target 的通过结果。这不等于完整 Scene/Fabric 每帧接线、跨适配器 GPU parity 或真实 driver binary 覆盖已经完成。

## 7. 历史依据

- `6ab2e626d feat(render): close shader package persistence lifecycle`：引入热端与持久化生命周期、模块/包映射及 worker 测试覆盖。
- `8400fe972 feat: complete runtime and render refactoring implementation`：将渲染运行时、shader package、pass set 和测试证据合入统一实现。
- `5ad60f878 test(render): name refactor acceptance gates`：将关键重构验收固定为命名测试。

本篇结论：当前 shader package 的身份、typed render state、interned package handle、冻结 FeatureSelection 到 package/permutation/PSO identity 的映射闭环、有界编译队列、冷热缓存映射、DriverLocal 边界和 direct/bin/split pass set 已由源码与已标注的运行记录形成对应证据；后续变更必须保持上述语义分层，并继续把完整 Scene/Fabric 接线与跨适配器证据作为独立验收项。
