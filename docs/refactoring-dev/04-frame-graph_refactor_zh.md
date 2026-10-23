# 04 · Frame Graph 与 pass 组织

> 文档口径：以当前源码、测试和已合入历史为准的实现审计。本文描述统一编译内核、shading graph 拓扑、生成式后处理清单和调度边界。
> 上层导航：[`00-overview_refactor_zh.md`](00-overview_refactor_zh.md)
> 相关文档：[`02-response-composition_refactor_zh.md`](02-response-composition_refactor_zh.md)、[`03-shader-package_refactor_zh.md`](03-shader-package_refactor_zh.md)

## 1. 当前模块边界

### 1.1 统一 frame graph 编译器

`prism_render_architecture::frame_graph` 兼容 re-export 已删除；frame graph 的唯一定义与公共入口是 `prism_render_graph::frame_graph`。声明类型位于 `pkg/prism_render_graph/src/frame_graph/`，统一推导内核位于 `pkg/prism_render_graph/src/compile/` 和 `pkg/prism_render_graph/src/graph.rs`。Architecture、Scene、World、Editor 与 Profiler 的调用方应直接依赖并导入此唯一真源。

当前声明式类型的枚举数量为：

| 类型 | 变体数 | 当前变体 |
|---|---:|---|
| `PassKind` | 4 | `Raster`、`Compute`、`Transfer`、`Present` |
| `QueueClass` | 3 | `Graphics`、`Compute`、`Transfer` |
| `ResourceLifetime` | 3 | `Imported`、`Persistent`、`Transient` |
| `AccessKind` | 11 | sampled/storage read/write、uniform/index/indirect read、color/depth attachment write、read-only depth attachment、transfer read/write |
| `RenderGraphSystems` | 4 | `Begin`、`Render`、`Submit`、`Finish` |

`GpuFrameGraphBuilder` 的 pass 只声明资源访问、队列和显式前置依赖；统一编译器负责 SSA 版本、hazard、拓扑顺序、dead-pass culling、队列批次、transient lifetime/alias 和 barrier。`PassKind` 的 setup/execute 两阶段契约以及 `ExecuteContext` 负责把计划转换成 driver command buffer。

### 1.2 使用边界

- `pkg/prism_render_scene/src/shading/graph.rs` 构造 shading graph；
- `pkg/prism_render_scene/src/visibility/graph.rs` 构造 visibility graph；
- `pkg/prism_render_runtime`、`pkg/prism_render_app`、`pkg/prism_render_gpudriven` 直接消费 `prism_render_graph`；
- `prism_render_scene` 当前也直接依赖 `prism_render_graph`，不再通过独立的旧编译器实现重复推导。

ECS 编排和 GPU frame graph 仍是两级：`RenderGraphSystems` 约束 Begin → Render → Submit → Finish 的系统锚点；资源依赖和 pass 顺序由 graph compiler 从声明推导。`RenderPhase` 仍是绘制桶，不是材质模型。

## 2. 当前 shading graph 拓扑

`pkg/prism_render_scene/src/shading/graph.rs::shading_frame_graph()` 的固定主链包含 8 个 pass：

1. `visibility_raster`：图形队列写 visibility ids/metadata；
2. `material_classification_count`：计算队列统计材质分类；
3. `material_classification_prefix`：计算前缀和并写 indirect dispatch；
4. `material_classification_scatter`：生成 compact shading work 和 texel-space jobs；
5. `texel_space_shade_cache`：生成/更新 shade atlas；
6. `shading_resolve`：解析 shading work，写 response left 与 HDR scene color；
7. `water_surface_raster`：图形队列写 water response planes 和 coverage；
8. `response_composition`：常驻计算 pass，消费 imported response planes 并写回 HDR。

之后由 `append_post_process_passes` 接入 23 个生成式后处理声明，因此当前 shading graph 总计 31 个 pass：8 个固定主链 pass 加 23 个 post declarations。`post_exposure` 是后处理段的首项，`post_upscale` 是末项；可选效果通过输入数据产生 no-op，不改变图拓扑。

`response_composition` 固定在 `water_surface_raster` 之后、post chain 之前。左/右 response、controls、auxiliary 和 water coverage 是 imported plane；组合 pass 通过资源访问和 graph dependency 获得明确的 WAW/WAR 顺序，不依赖场景动态安装。

## 3. 生成式后处理与 executor ledger

`pkg/prism_render_scene/src/shading/post_graph.rs` 维护三份相互校验的事实：

1. `POST_PROCESS_DECLARATIONS`：23 个 pass skeleton，统一声明 ABI、bind-group shape、pipeline entry、资源和前置依赖；
2. `POST_EXECUTION_BINDINGS`：23 个唯一 production module 与声明名的对应表；
3. `shading/plugin.rs` 的 executor registration：把每个 binding 映射到实际初始化、资源准备和 pass executor。

后处理顺序为：

```text
response_composition
  → exposure → bloom → dof → motion_blur → ssr → spec_denoise
  → ssgi → taa → posterize → outline → hatching
  → halftone → kuwahara → color_grade → chromatic_aberration
  → vignette → film_grain → cas → gamut_map → ordered_dither
  → lens_flare → tonemap → upscale
```

声明表、binding 表、模块名、registration markers、executor 数量和 graph suffix 由 `validate_post_execution_bindings` 及 `validate_post_execution_registration` 交叉校验。生产 inventory 测试固定 `POST_EXECUTION_BINDINGS.len() == 23`，每个模块必须同时出现在 shading module tree、插件注册和执行 ledger 中。

## 4. 编译与资源约束

### 4.1 资源访问是依赖真源

`AccessKind` 的写访问会产生新的资源版本；编译器据此处理：

- read-after-write、write-after-read、write-after-write；
- queue transfer 和 layout barrier；
- transient 资源的生命周期区间与不重叠 alias；
- imported/persistent 资源不进入 transient alias；
- stable tie-breaking、cycle diagnostics 和 deterministic replay。

手工 `depends_on` 用于表达声明层语义边界，但不能绕过资源访问检查。pass 的执行顺序、barrier 和 queue batch 必须能从同一个 graph declaration 重建。

### 4.2 两级调度保持分离

`RenderGraphSystems` 只约束 ECS 系统的 Begin/Render/Submit/Finish 锚点；它不替代 GPU pass graph。`PassKind`、`QueueClass`、资源版本和 executor plan 属于 GPU graph 层；材质 `RenderPhase` 和 draw queue 属于 render app 层。两者不得互相借用枚举编码。

### 4.3 组合 pass 的数据流

```text
shading_resolve ──▶ response planes
water_surface_raster ──▶ response planes / water coverage
response planes + coverage ──▶ response_composition ──▶ HDR
```

组合 pass 是固定拓扑的一部分。split response 的左臂和右臂共享已解析的阴影/光照输入，组合阶段只消费 imported planes 和 mask/control 数据；场景中的区域选择不会改变 graph pass 数量。

## 5. 当前架构约束

1. 仓库内只保留 `prism_render_graph` 的一套 schedule/barrier/lifetime/alias 推导内核。
2. 不保留 `prism_render_architecture::frame_graph` 旧路径；所有调用直接使用 `prism_render_graph::frame_graph`。
3. shading graph 的 pass 数量、执行顺序和 runtime 校验从同一个 `GpuFrameGraphBuilder` 读取；不得维护第二份固定数字。
4. post declaration、executor binding 和 plugin registration 必须保持一一对应；production inventory 必须覆盖 23 个模块。
5. `response_composition` 必须常驻，并固定在 `water_surface_raster` 之后、post chain 之前。
6. pass 顺序来自资源访问和 graph dependency，不以 Bevy tuple 长度或 `add_systems` 调用分组表达 GPU 拓扑。
7. stylized post 与 PBR/响应路径共享同一 HDR 后处理基底，不建立平行 NPR frame graph。
8. transient alias 只作用于 graph-owned transient resource；imported 和 persistent resource 保持外部/跨帧所有权语义。

## 6. 当前验收证据

| 契约 | 证据 |
|---|---|
| 统一 graph 的 pass、资源、SSA version、hazard、schedule、barrier、alias、lifetime | `pkg/prism_render_graph/src/compile/`、`pkg/prism_render_graph/src/graph.rs`、`pkg/prism_render_graph/text/unit/compiler.rs`、`pkg/prism_render_graph/text/unit/graph.rs` |
| 唯一 frame_graph 公共契约 | `pkg/prism_render_graph/src/frame_graph/` 与 `pkg/prism_render_graph/text/unit/compiler.rs` |
| shading graph 单一 pass-count source、主链顺序、资源 barrier | `pkg/prism_render_scene/src/shading/graph.rs` 中 `declared_pass_count_matches_runtime_assertion` |
| composition 固定位置、imported planes、HDR barrier | `pkg/prism_render_scene/text/unit/shading/composition_graph.rs` 中 `composition_is_resident_and_follows_water_response_producer` |
| composition production recorder 与 GPU 回归 | `shading/composition.rs::response_composition_pass` 使用共享 `SceneFabric` / `RenderFabricFrame`，经 `fabric_runtime::prepare_with_templates` 的 active-pass compile、真实 imports、prepare 与 encode 交给宿主；`text/integration/resident_composition_gpu.rs` 与生产入口共用 recorder。acceptance 的 `CompositionFabric` 仅封装同一 scene selection 构造器，不是第二份生产 ECS catalog。2026-10-22 修订后的 harness 已 strict 3/3，局部待复跑缺口已关闭，不证明共享 Scene/Fabric/TAA 全帧闭环 |
| pass skeleton codegen 的确定性、依赖校验、资源复用和 cycle diagnostics | `pkg/prism_render_graph/src/frame_graph/codegen.rs` 及其单元测试 |
| 23 项 post inventory、声明/binding/executor 对齐和插件 registration | `pkg/prism_render_scene/src/shading/post_graph.rs`、`pkg/prism_render_scene/src/shading/plugin.rs` |
| shading crate 历史运行记录 | `cargo test -p prism_render_scene --lib shading`：844 passed，0 failed；不是本轮复跑结果 |

2026-10-22 帧容器定向运行补记（main 实际运行，本次文档同步已读取日志、未复跑）：`cargo test -p prism_render_architecture --lib render_fabric::frame` 为 **19 passed / 0 failed**，日志 `/private/tmp/prism-fabric-frame-20261022.log`；对应 frame 改动已独立提交 `f8ab51076`。覆盖 owned vector 经 `from_items → into_items` 无重分配、`replace_domain` 保留其他域与帧号并在测试容量内复用原存储，以及错误时清空整帧和后续合法输入恢复。该结果仅证明 architecture CPU 帧容器契约，不证明 Bevy/native 生产聚合、31-pass 录制、GPU 或完整 Scene/Fabric/TAA 帧闭环；该 frame 子集不用于推增历史总数；全筛选及 catalog 的实际运行结果另记如下，不以局部通过替代全场景未完成项。

2026-10-22 architecture Render Fabric 局部复验：较早 main 运行的 `/private/tmp/prism-fabric-architecture-20261022.log` 为历史 **176/176**；后续 architecture 代理报告已实际运行 `prism_render_architecture --lib render_fabric` **200/200，terminal exit 0**，包含新增 9 项对象资源 CPU 契约测试，子集不与 185 重复相加。本次文档同步未运行构建，185 的来源是代理运行回报，不将旧 176 日志冒充新结果。该筛选不覆盖整个 `shader_package`、architecture acceptance、GPU 或全场景闭环。

2026-10-22 RenderGraph 全 package 运行补记（main 实际运行并确认 terminal exit 0，本次已读取 `/private/tmp/prism-render-graph-20261022.log`）：`cargo test -p prism_render_graph`（无 `--lib` 限制）unit **122 passed / 0 failed / 0 ignored**；doc **1 passed / 0 failed / 1 ignored**。doc ignored 不计作通过。旧 85/85 与磁盘耗尽中止尝试仅保留为历史，不再代表当前状态；当前 package 验证缺口已关闭，不外推为全场景 GPU、full-frame 或跨适配器通过。

2026-10-22 public Domain GPU integration：沙盒内 0 adapter，严格模式拒绝；沙盒外以 `PRISM_REQUIRE_GPU=1` 重试 `--test fabric_domain_gpu`，Apple M2 / Metal 上 **1 passed / 0 failed / 0 ignored**，terminal exit 0，日志 `/private/tmp/prism-fabric-domain-gpu-20261022-outside.log`。覆盖公开 API 的设备对齐子区间绑定与 lease 保留；不证明域 shader、draw/dispatch 或全场景渲染。

2026-10-22 runtime multipass 局部复验：较早 `/private/tmp/prism-fabric-runtime-gpu-20261022-fixed.log` 的 **8/8（7 hardware + 1 CPU fixture）**保留为历史；main 后续实际运行并确认 terminal exit 0，本次核对 `/private/tmp/prism-fabric-object-runtime-gpu-20261022.log`，`PRISM_REQUIRE_GPU=1` 下 `fabric_runtime::tests::multipass_gpu` **9 passed / 0 failed / 0 ignored**，其中 **8 项 Apple M2 / Metal 硬件用例 + 1 项 CPU 编译契约 fixture**。新增 `object_suballocations_upload_and_execute_by_exact_identity` 使用 Surface 对象 fixture，验证 object index/generation/history/role 精确寻址、完整 payload 上传与错误 role/短上传拒绝、缺失/重复绑定和重叠拒绝，以及同一真实 buffer 子分配经多 pass 执行/readback 后互不串写；既有 lease/preflight、录制顺序、push constants 与失败清理仍各限测试范围。不是 Hair/Volume/Decal 语义 shader 或 `SceneFabricObjectResource::from_domain` 桥接 GPU 验证，也不是全帧或跨适配器证据。

2026-10-22 resident composition integration 复验（main 实际运行并确认 terminal exit 0，本次文档同步只读日志、未运行 Cargo）：`/private/tmp/prism-resident-composition-gpu-20261022-aligned.log` 记录 `PRISM_REQUIRE_GPU=1` 下 **3 passed / 0 failed / 0 ignored**。GPU producers 经生产共用 recorder 写入 HDR，以及 resident targets/pipeline identity 两项记录 Apple M2 / Metal / IntegratedGpu；另 1 项为 HDR storage capability 拒绝校验，不把三项全部计作硬件用例。修订后的局部 harness 待复跑缺口已关闭，不等于真实 Scene 的全部 31 pass、统一 TAA/history 或全帧 parity 已验证。

2026-10-22 material package 关联证据（main 实际运行并确认 terminal exit 0，本次已读取 `/private/tmp/prism-render-material-package-20261022.log`）：`prism_render_material` unit **888 passed / 0 failed / 0 ignored**、integration **79 passed / 0 failed / 0 ignored**、doc 0 tests。integration 含 freshness 19、header ABI 4、PCM parity 7、S5 ABI/cache 4、schema parity 10、ShadingPlan 33、surface ABI mirror 2，子集不与 79 重复相加。仅关闭 material package 复验缺口，不代表其他 shader/architecture acceptance、graph GPU 或全帧消费已验证。

2026-10-22 TAA GPU 首次阻断与重试通过记录：main 的首次 TAA GPU 日志 `/private/tmp/prism-taa-gpu-20261022.log` 因依赖 `prism_physics_core/src/world.rs:651` 缺少 `prepare_assembly_collider` 报 E0599，未执行测试，仅保留为历史编译阻断；随后重试已成功，不再列为当前待复跑状态。main 实际运行并确认 terminal exit 0，本次文档同步已读取 `/private/tmp/prism-taa-gpu-20261022-retry.log`：`prism_render_scene --lib` 的 `shading::taa::resources::gpu_tests` 筛选在 `PRISM_REQUIRE_GPU=1` 严格模式下，Apple M2 / Metal REAL HARDWARE **19 passed / 0 failed / 0 ignored**，2021 filtered out，0.84s，包含新增 lifecycle 7 项。覆盖真实 bind-group、两帧 publication/import 与 CPU golden、queue completion，以及 resize/removal/device-table replacement 的 retirement、未提交 dispatch 资源保留与 ID 回收。日志中的 stale ID panic 由 `catch_unwind` 捕获，是预期失效断言，最终测试通过。TAA 局部 GPU 验证缺口已关闭；完整 Scene/Fabric/TAA full-frame parity、统一全场景 runtime/recorder 和第二 physical adapter 仍未闭合。

跨适配器行为单独由 `pkg/prism_render_scene/tests/cross_adapter_acceptance.rs` 验收：它枚举 distinct physical adapters，并在每个 adapter 上执行 storage compute、submit、copy-to-staging 和 map/readback。默认 profile 在少于两个 adapter 时输出 `SKIP_UNVERIFIED`；`PRISM_REQUIRE_GPU=1` 将同一条件提升为失败。2026-10-20 沙盒外只发现 `Apple M2 / Metal / IntegratedGpu` 一个 adapter，故严格门禁失败。该测试证明 adapter-local submission/readback，不证明 31-pass frame graph、Scene imports、TAA exact history publication 或 water full-frame parity 已跨 adapter 通过；TAA history 的单适配器局部 GPU 缺口已由 2026-10-22 strict 19/19 关闭，但这不是跨 adapter 或 full-frame 验收。

## 7. 历史依据

- `352d594b1 refactor(render): unify frame graph execution planning`：删除重复的 frame-graph compiler 推导，统一执行规划和 codegen 接口。
- `8400fe972 feat: complete runtime and render refactoring implementation`：合入 shading graph、composition、post inventory、visibility 和 runtime 验收链。
- `5ad60f878 test(render): name refactor acceptance gates`：固定 graph pass count、visibility 和分类验收的命名测试。

本篇结论：当前 frame graph 已由统一 `prism_render_graph` 编译器承载；shading graph 以 8 个主链 pass、23 个生成式后处理 pass 和 resident composition 组成固定拓扑，资源访问、执行 ledger 与测试共同构成验收契约。

## 8. 最终收口验收矩阵 / 未完成项（2026-10-21 文档审计）


**2026-10-22 最新复验与状态修正**：architecture 改动已提交 `652404b94`。main 对当前架构源码实际复验 `render_fabric` **200 passed / 0 failed / 0 ignored**、`shader_package` **95 passed / 0 failed / 0 ignored**、acceptance **3 passed / 0 failed / 0 ignored**；日志与范围见 05 §15.2。本文较早的 185 项及 shader-package/acceptance“未复跑”描述仅代表此前截点，以本补记为准。acceptance 的 3 项验证 ledger/fanout/partial 标记，不是完整 32 gate 端到端通过。

**局部 GPU 桥接证据修正**：既有日志已记录 `fabric_runtime` 定向筛选 **29 passed**（CPU/GPU 混合，不能全部算硬件用例），其中 authenticated `from_domain` bridge 在 Apple M2/Metal 完成真实上传、dispatch 和子范围隔离；RHI pipeline bridge 筛选 **15 passed**，包含 exact native render-pipeline identity/generation/lease 与原生绘制。这里只关闭相应局部桥接验证，不能关闭 Hair/Volume/Decal 的真实语义渲染、typed graphics 全链、统一 Scene/Fabric/TAA full-frame、第二 adapter 或性能验收。

**2026-10-21 收口补记**：此前 Scene/Fabric 相关源码的 `cargo check` 曾通过；`prism_asset` 的 `Flight` 初始化缺少 `cancellation_waiters` 字段是另一次尝试的历史阻断，当前定义已无该字段，后续反馈已越过 asset 初始化。composition 已消费共享 `SceneFabric` selection/runtime 的真实 compute 切片；native `RenderApp` 已有 Surface/Particle 聚合、GPU Prepare 后刷新、host graph record/submit 前 admission，以及 visibility 的共享帧/对象身份校验。Bevy `prism_render_scene` 另已为 Hair/Volume/Decal 增加稳定 render-world owner、`CollectDomainContributions → BuildFrame` 的严格帧号聚合 seam，以及从 canonical semantic role 到 `SceneRhi` physical lease、运行时 `ResourceId` import 和 record-ready view 的 admission/import 层。错帧、冲突、错误 identity/layout/usage/import coverage 或物理 alias 都显式拒绝；一次性 drain 不回放已移除 producer 的旧项。两条集成路径都还没有把这些域合并为 Bevy 31-pass 或原生全场景统一 recorder；pipeline、bind group、draw/dispatch recorder 与通用 runtime 的全场景生产调度仍待闭合。

**2026-10-21 当前工作树补充复验**：`AccessKind` 已扩展为 11 种，新增 `UniformRead`、`IndexRead` 与不产生新 SSA 版本的 `DepthAttachmentRead`；buffer/texture use、texture namespace、attachment 与写入判定均由 `AccessKind` 的集中映射提供。RenderGraph 录制失败路径已为本帧创建的 texture/view 提供 RAII 回收，覆盖创建、recorder callback 与被裁剪 closure 的 panic。该切片先前定向运行记录为 **85/85**；本次尝试复跑 `prism_render_graph --lib` 时在依赖归档阶段因磁盘 `No space left on device` 停止，未进入测试执行，因此不把 85/85 写成当前复跑结果；后续 2026-10-22 全 package 已通过 unit 122/122、doc 1 passed / 1 ignored（见 §6）。同期当前工作树 RenderFabric 定向测试已实际执行并通过 **175/175**；Hair/Volume/Decal 的 `legacy-bevy fabric_domains` 定向筛选 **33/33**。这些只证明 architecture contract/compiler/runtime 与三域 owner/semantic/physical admission，不是 31-pass 全场景 GPU 闭环。

**2026-10-22 Hair/Volume/Decal 生产接线边界修正**：Hair 与 Decal 的 catalog 仍使用 **256B persistent STORAGE buffer** 泛化模板，每域仅声明 1 个 compute pass 和 3 个 `StorageWrite`，尚不匹配 strand arrays 或 decal receiver/material ABI；仍需真实资源/pass/access、生产数据、pipeline/bind group、draw/dispatch 与下游消费者。Volume 的 catalog 同样是泛化模板，不能代表现有 volumetric fog：`pkg/prism_render_scene/src/shading/volumetrics` 已有按 view 缓存的真实 3D froxel textures、scatter/integrate/apply pipelines、bind groups、scene depth/color 输入及 copy-back。Volume 真正未完成的是把该生产路径纳入冻结 SceneFabric 的真实纹理布局、pass 依赖/hazard、typed recorder 和宿主 record/submit；scene-color copy-back 尚未作为 Fabric pass。不得用占位 buffer、probe 或伪造输出冒充集成；`from_domain` 的局部 GPU 上传/dispatch/subrange 证据不覆盖 volumetrics 语义或全帧闭环。

**2026-10-23 Volume 状态更正**：上述“Volume 未接 SceneFabric pass/resource/hazard/copy-back recorder”已过时。真实四段生产路径已通过 frozen SceneFabric typed recorder 接入，并核对真实 texture layout/usage；Apple M2/Metal strict GPU harness **1/1** 已验证生产 dispatch、readback 与 copy-back 像素一致性，设备支持 96-byte immediate 门槛。未完成的是 resize/disable/missing-input 生命周期验收及全帧闭环。

`S`=源码实现，`D`=测试定义，`R`=对应版本运行通过，`H`=已有运行记录，`G`=实际 GPU 证据，`E`=全场景闭环。最近完整 `prism_render_scene --lib` 的历史记录为 1909/1909；当前三域筛选为 33/33。这两项不覆盖 package/integration 或 GPU 验收；2026-10-22 RenderGraph unit 122、doc 1 passed / 1 ignored，material unit 888/integration 79，以及 composition 3、TAA 19、runtime 9（8 hardware + 1 CPU fixture）的局部结果分别见 §6，均不是全场景 draw/dispatch 或 full-frame replay。

| 条款 | S | D | R | H / G | E |
|---|---|---|---|---|---|
| graph compiler、SSA version、hazard、barrier、alias、lifetime | 当前统一编译器、11 种访问映射与录制失败 RAII 回收存在 | compiler/codegen/graph/recording gate 已定义 | RenderFabric 200/200（2026-10-22 main 实际复验）；2026-10-22 RenderGraph 全 package unit 122/122、doc 1 passed / 1 ignored | RenderGraph 85/85、shading `844/0` 等为先前记录 | 否；没有 full-frame 复核 |
| 31-pass shading topology（8 主链 + 23 post） | pass inventory、composition 固定位置和 runtime assertion 存在 | pass-count、order、resource/barrier gate 已定义 | scene lib 1909/1909 | 既有记录只证明相应历史工作树 | 否；未证明完整 Scene/Fabric/TAA 执行 |
| resident Composition 与 imported resources | 共享 SceneFabric selection/runtime 下的生产切片、imports 和 HDR barrier 接线存在 | integration/GPU harness 定义存在 | 2026-10-22 resident composition strict 3 passed / 0 failed / 0 ignored（2 硬件 + 1 capability 拒绝） | 2 项 Apple M2/Metal 硬件用例覆盖局部生产共用 recorder/HDR readback 与 identity；1 项为 HDR storage capability 拒绝校验，不等于整图 GPU 证据 | 否 |
| 域帧 producer 与图接入 | Bevy Hair/Volume/Decal 已由稳定 render-world owner 聚合并具备 physical resource admission/import；Volume 真实 fog 四 pass 已接入 SceneFabric typed recorder；native Particle 有 renderer-owned BufferId 和生产 RenderGraph 路径 | 同帧合并、一次性消费、producer 移除、owner identity/snapshot、错帧/冲突、资源 identity/layout/usage/import coverage/alias 测试已定义 | 默认域 22/22；owner/producer 3/3；`legacy-bevy fabric_domains` 33/33；Volume catalog contract 1/1；Volumetrics dispatch/readback 与 lifecycle strict GPU 合跑 2/2；Particle multiview strict GPU 1/1（Apple M2/Metal） | Volume 与 Particle 各自局部生产 GPU 路径/readback 已通过；域 admission/import 本身不编码 Hair/Decal 渲染 | 否；Particle 尚未接入 Bevy SceneFabric runtime；Volume 缺完整 schedule 下资源 retirement 验收；Hair/Decal 仍缺实际 pipeline/consumer/dispatch；没有 31-pass 全场景消费 |
| 对象级 runtime 局部切片 | exact object/generation/history/role 地址、payload upload 与子分配 preflight 存在 | Surface 对象隔离 fixture 已定义 | 2026-10-22 main strict multipass 9/9（8 hardware + 1 CPU fixture），exit 0 | Apple M2/Metal 对象子分配真实 dispatch/readback；历史 8/8（7 hardware + 1 CPU fixture） | 否；该 multipass 子集不包含 `from_domain`；另有桥接局部通过记录，仍不含三域语义消费者或全帧 |
| TAA/history 与跨 adapter | TAA history 源码接线存在；跨 adapter strict/skip probe 存在 | history/adapter 与 GPU lifecycle gate 已定义 | 2026-10-22 main TAA strict GPU 19/19（含 lifecycle 7 项） | Apple M2/Metal 局部 history GPU 已验证；第二 adapter 未验证 | 否；统一 runtime/recorder 与 exact history full-frame parity 仍未闭合 |

RenderGraph 全 package 的 unit 122/122、doc 1 passed / 1 ignored 已于 2026-10-22 验证；material package unit 888/integration 79 与修订后的 resident composition strict 3/3 待复跑缺口也已关闭，不外推为其他 integration 或 GPU 门禁通过。仍未验证：全 31 pass 在真实 Scene imports、barrier、record/submit 上的完整运行；Hair/Decal 的真实 pipeline、bind group、draw/dispatch recorder 与 visibility/geometry consumer；Volume 的完整 schedule completion-gated retirement（resize/disable/missing-input 的局部生命周期已由 strict GPU 覆盖）；Particle 接入统一 SceneFabric runtime；TAA 与统一全场景 runtime/recorder 的集成（局部 completion-gated history GPU 已通过 19/19）；完整 32 gate 端到端运行（gate 23 仍 partial）及旧入口删除前护栏、S2/S4/S5 历史先后顺序；第二 physical adapter；以及全帧图像 parity、资源峰值、吞吐和 P50/P95/P99 性能数据。库内历史 1909 项、RenderFabric 200 项及局部 composition/runtime/TAA/Volumetrics/Particle 通过都不等于生产闭环完成。

**2026-10-22 indirect command bounds 补记**：RenderGraph 的 `ExecuteContext` 现在在加入 driver command 前按 graph-declared `BufferDesc.size` 校验 `draw_indirect`、indexed draw、multi-draw 的末条记录范围，以及 indirect dispatch。参数 record 大小由 `prism_render_driver` 的公开 `DRAW_INDIRECT_ARGS_SIZE`（16B）、`DRAW_INDEXED_INDIRECT_ARGS_SIZE`（20B）、`DISPATCH_INDIRECT_ARGS_SIZE`（12B）ABI 常量统一提供（ABI 常量与单测已提交 `9e4131964`）。范围乘加采用 checked arithmetic，越界、溢出、偏移未对齐或 multi-draw stride 不合法会标记 pass recording error；闭包返回后统一拒绝，闭包继续追加的命令也不会编码或提交。单条与多条 draw、dispatch 的精确末尾和 overrun、sticky error、未对齐偏移、非法 stride，以及 multi-draw 溢出均有回归覆盖；定向 `indirect_` 筛选 **6/6**，RenderGraph 默认配置 unit **127/127**，`parallel` feature unit **131/131** 均在当前源码实跑通过，doc **1 passed / 1 ignored**。该范围依据 imported/virtual buffer 的图声明尺寸；外部 owner 必须保证声明尺寸与真实 allocation 相符。以上只关闭命令参数 bounds contract，不证明 31-pass 全帧执行或 GPU 图像 parity。

**2026-10-23 VSM resolve 死字段清理（源码切片）**：`ViewResolveVsmPageTable` 的 `slot_count` 副本没有消费者，且其 `dead_code` 豁免明确指向不存在的 “later slice”。当前移除此 resolve/bridge 间冗余字段与示例占位代码；page-table 自身的 `slot_count` 仍保留用于分配、上传及尺寸校验。本切片 `rustfmt --check` 与 `git diff --check` 通过；未运行 Cargo/GPU，故不把它记作类型检查或运行验收，也不改变 31-pass / SceneFabric 全帧未闭合结论。


**2026-10-23 Frame Graph 旧入口删除与 runtime 复验**：移除 `prism_render_architecture::frame_graph` 模块及其重复 compiler 测试，runtime 与 Scene composition 测试改为直接引用 `prism_render_graph::frame_graph::ResourceLifetime`。当前 Rust/TOML 搜索未发现 architecture 旧路径消费者；Frame Graph 定义和公开入口只保留在 `prism_render_graph`。随后 `cargo test -p prism_render_runtime --lib retirement_tests --jobs 1 --config profile.test.debug=0 -- --test-threads=1` 实际运行 **28 passed / 0 failed / 20 filtered out**，覆盖 runtime retirement 与 submission-state 回归。该测试关闭这次 API 迁移的 runtime 定向回归缺口，不代表 render-graph 全 package 或全场景闭环重跑。

**2026-10-23 当前工作树 RenderGraph package 复验**：`cargo test -p prism_render_graph --jobs 1 --config profile.test.debug=0 -- --test-threads=1` 实际 **127 passed / 0 failed / 0 ignored**；doc tests **1 passed / 1 ignored**（exit 0）。较 2026-10-22 的 122 项新增覆盖已包含于当前全 package 运行；ignored doctest 不计为通过。该结果仅覆盖 RenderGraph crate，不代表 31-pass 全场景执行或 GPU 图像 parity。

**2026-10-23 Hair raster strict GPU 子切片复验**：在 Apple M2 / Metal REAL HARDWARE 上，启用 `legacy-bevy` 并设置 `PRISM_REQUIRE_GPU=1` 后，`hair_raster_three_pass_physical_gpu_acceptance` **1 passed / 0 failed / 0 ignored**。该用例从生产 WESL 编译三阶段 raster pipeline，以 production `HairRasterDispatchParams` ABI 执行 depth / deterministic ID resolve / coverage publish，并对遮挡、同深度 tie-break、视口边缘和退化线段进行完整 16×12 vis-buffer readback，与 CPU oracle 比较 depth、ID、coverage 全数组。先前未启用所需 feature 的两次命令均为 **0 tests**，不计验证；严格正确 feature 命令实跑退出码 0。此证据只关闭 Hair raster shader/ABI 的 Metal 子切片验收，不证明 `ResampledGroom → projection → production SceneRhi upload → Fabric recorder/consumer` 接线、资源容量/退休、真实场景或完整帧闭环。当前完整 Hair/Decal/Particle 与 31-pass Scene/Fabric/TAA 验收缺口仍保留。
