# 02 · Shading Plan 与 Composition

> **2026-11-09 目标边界（当前权威）**：响应与组合的目标实现/验收只走 Prism 原生 prism_render_app、Prism RHI 与 prism_render_graph。Bevy、Bevy RenderApp、prism_render_scene 的 Bevy 集成路径、SceneRhi 与 legacy-bevy 是历史/排除项，不是当前 composition recorder、TAA/history 或 GPU parity 的前提。局部历史 harness 不能替代原生全帧证据。

## 当前审计结论（2026-11-11，native Prism）

**已确认**：native response/composition 的 `ShadingPlan`、MaskBlend identity、output contract、lobe-budget ABI 和 composite pipeline 有源码/契约证据；非空场景 GBuffer 及 GBuffer → composite Metal readback 已通过。另有单一 `Unlit` 原生垂直切片经 identity、GBuffer、composite 写入 HDR，中心 readback 为 `[1,1,1,1]`。composite group-2 只消费 canonical header/parameter，Surface encode 解析的纹理/VT 结果通过 GBuffer material planes 进入 composite，已满足 Metal storage-buffer 限制。

**仍未完成**：真实非空 native 材质已经能通过最小 Scene GBuffer 验收，独立 composite contract 也能在 Metal readback；但从多种 response program、material/mesh binding、PSO 到最终 HDR 输出的统一全帧尚未闭合。`native_scene_mesh_draws_the_visible_set_on_a_gpu` 验证的是实际 camera-local target 的 normal/albedo 与透明 HDR clear，composite contract 测试验证的是固定 GBuffer consumer，不证明完整材质场景已经产出最终可见画面。`native_full_frame_acceptance.rs` 使用 `NullAdapter`/`NullDevice`/`NullQueue` 的 `DeferredInstance`，相机没有 renderable object，并明确断言 queued/recorded draw 均为 0；这只是空/null lifecycle fixture。

因此，CPU response golden、组合契约、Bevy/SceneRhi 历史 GPU harness 都不构成当前 native 全帧证据。native 的完整 response program 消费、TAA/history 与 composition 的统一全帧接线、多视图/多域 fan-out、真实 failure/retirement、图像 parity、第二适配器和 P50/P95/P99 仍未完成。

> 本文记录当前响应程序、组合器、执行分桶、输出通道和水体接线。Surface ABI 真源见 [`01-surface-abi_refactor_zh.md`](01-surface-abi_refactor_zh.md)。

## 0. 核心模型

`ShadingPlan` 是单一 sum type：

```rust
Single { program: ProgramRef }
MaskBlend { left, right, mask, domain }
```

`ProgramRef` 有两类：

```text
Builtin(Lit | Npr | Unlit)
Registered(ProgramId)
```

`Single` 选择一个响应程序；`MaskBlend` 选择两条响应臂并声明 mask 来源和合并域。组合器不伪装成响应程序，响应程序也不携带另一条响应臂的组合状态。

## 1. 五类 response contract

每个程序通过 `ResponseManifest` 发布五类契约：

| 契约 | 当前职责 |
|---|---|
| `SurfaceContract` | required / optional / intentionally ignored fields、supported lobes、field mapping |
| `LightingContract` | direct light、shadows、indirect diffuse、reflection、transmission、temporal history |
| `OutputContract` | 产出通道与可混合通道 |
| `ExecutionContract` | fused / split 能力与 cost class |
| `CacheContract` | response / surface auxiliary 的读写和 temporal requirement |

字段覆盖采用三值语义：mapped、intentionally unmapped(reason)、未声明错误。活动瓣必须由程序消费，或明确写入忽略原因；没有静默丢弃路径。

## 2. 执行分桶

旧的固定类别表不是当前执行模型。当前分桶键为：

```text
(execution_family, program_id, render_phase)
```

`ResponseBucketTable` 对键排序、去重并分配 dense bucket id；注册程序由 execution tables 提供，内建程序的 id 为 Lit=0、NPR=1、Unlit=2，注册程序使用 `ProgramId`。因此 bucket 数量随当前程序和执行表变化，不绑定某个固定类别上限。

WESL 没有任意动态函数分派，所以运行时 bucket 只选择编译期存在的有限入口。`validate_compiled_entries` 会逐 bucket 检查对应 entry 存在。

## 3. MaskBlend 的执行路径

### 3.1 fused 与 split

- 两条臂都支持 fused 且组合条件满足时，使用 fused builtin path。
- 任一臂需要独立 package、注册程序或 split-only contract 时，使用 split path。
- split path 为 left、right 和 resident `CompositionPass` 分别保留执行身份，不构造每一对程序的联合 package。
- 两条响应臂共享同一份验证过的 lighting context；共享的 visibility、shadow 和其他基础光照项只读取一次，随后分别调用响应程序。

这一边界让 package 身份、响应解释和结果组合保持独立，也让动态程序集合不会变成材质对的 O(N²) permutation 空间。

### 3.2 ResponseOutput

`ResponseOutput` 显式携带：

- direct；
- indirect diffuse；
- reflection；
- transmission；
- emission；
- alpha；
- auxiliary。

`BlendDomain::FullRadiance`、`DirectOnly` 和 `CustomChannelMask` 分别定义 radiance 通道的选择范围。alpha 使用共享表面 alpha 规则；auxiliary 使用逐字段的 typed merge rule。无效 mask、未声明规则、类型冲突、非有限值和不一致 identity 都是显式错误。

任何 `ShadingPlan::MaskBlend` 都从 conservation / white-furnace suite 中排除，并递增 `skipped_mask_blend`；视觉组合不被伪称为单 BSDF 的物理守恒证明。

## 4. 二值选择与连续 mask

二值区域选择与连续 mask 是两个独立契约：

- `REGION_ILLUMINATION`：visibility metadata 中的 Lit/NPR 二值 selector；
- `ILLUMINATION_BLEND`：16-bit UNORM 连续权重，带严格 sentinel 和 reserved-bit 校验。

连续 mask 的分类结果包含两条已验证响应臂、packed mask metadata、blend channel mask 和 response bucket。单响应计划不能伪装成连续混合；缺失或畸形 metadata 会被拒绝。

`region_selection_materials_execute_both_response_arms` 直接覆盖逐区域选择的两臂可达性；`continuous_blend_emits_two_verified_arms_and_packed_response_meta` 覆盖连续路径的两臂和 metadata。

## 5. Custom 程序

Custom 的身份来自 `ProgramRef::Registered(ProgramId)` 和已验证的 `ResponseManifest` / compiled entry。注册过程检查：

- `ProgramId` 不使用保留的无效值；
- package manifest、source revision、entry point、ABI、variant count 与 execution table 一致；
- package state 为 ready；
- verified artifact digest 与 compiled entry 一致；
- permutation space 与 manifest variant count 一致；
- 同一程序不会重复注册。

stale registered id、package revision/ABI/digest 变化、metadata 已 ready 但缺少 executable validation，都会返回错误，不回退到 Lit/NPR/default。Custom 仍然受同一套 Surface、Lighting、Output、Execution、Cache contract 约束。

## 6. 水体当前组合链路

`WaterShadingPlan` 与通用响应正交：

```text
WaterShadingPlan::MaskBlend {
    left: Pbr,
    right: Npr,
    mask: ShoreAndShallow,
    domain: FullRadiance,
}
    → water_surface_fs_mask_blend
    → response_left / response_right / water_coverage
    → resident response_composition
    → HDR export
```

当前 `water_surface_raster.wesl` 的四个 fragment entry 是：

```text
water_surface_fs_pbr
water_surface_fs_npr
water_surface_fs_custom
water_surface_fs_mask_blend
```

`water_surface_fs_hybrid` 不在当前入口集合中；`hybrid_wesl_compiles_and_preserves_all_four_entries` 同时检查四个现行入口可编译，且已删除的名称不会重新成为 live shader API。当前入口防回归已经有测试，但它不能倒推为“删除旧入口前已运行护栏”。该历史验收必须单独记账。

`water_hybrid_gpu.rs` 另有八个 `water_legacy_gpu_parity_*_full_frame` 用例，分别覆盖深度、岸边、泡沫、焦散、反射、透明/overlay、motion vector 和 reactive mask。每例编译未修改的生产 WESL，绘制 PBR、NPR、MaskBlend 三个生产 fragment entry，并运行生产 caustics、underwater、resident composition 与 HDR export；读回后逐像素比较 16 个 response words、control、coverage、motion MRT、HDR 与 premultiplied color。其 oracle 实际执行 `fixtures/water_legacy_41a31cee` 中冻结自 `41a31ceef24e59ab7f8dda076b2b4c7f5e11f6ec` 的旧 shader closure（raster、renderFX、composition 与 HDR export），分别固定 PBR/NPR endpoint，并在旧 ABI 的 `overlay == shallow` 条件下直接比较历史 Hybrid。当前 ABI 独立 overlay 的组合则使用冻结 endpoint 与测试两阶段公式校验，不把 MaskBlend 自身的 motion 当作 oracle 输入，也不冒充旧 ABI 捕获。这里的 full frame 仅指 16×16 离屏水体 fixture 的全部像素，不是完整 Scene/Fabric/TAA 帧的前后版本图像对比；冻结回放也不能追认旧入口删除前护栏已运行。

另有十个 `water_gpu_*` 定向真实 GPU 输出用例，用于定位当前 ABI 的各个可观察结果。这里的 shore 是 solver foam lane，shallow 是 authored scalar；它们不等同于旧 ABI 的独立 shoreline/bathymetry 绑定。八项全帧 parity 与十项定向输出都需要以 `PRISM_REQUIRE_GPU=1` 实跑通过才可作为真实 GPU 证据；无适配器时的 skip 不构成验收。

## 7. 当前门禁

| 门禁 / 测试 | 当前保证 |
|---|---|
| `generated_builtin_contracts_partition_every_field_and_lobe` | 内建程序的五类契约覆盖 schema 字段和 lobe |
| `lobe_support_is_not_a_boolean_claim` | lobe 支持不是单一布尔断言 |
| `unknown_active_bits_are_rejected` | 未知活动位不被接受 |
| `stale_registered_id_fails_plan_validation_instead_of_falling_back` | stale 程序 id 是硬错误 |
| `registration_failure_is_atomic_and_custom_requires_split` | 注册失败不污染表；Custom 的 split 资格被校验 |
| `region_selection_materials_execute_both_response_arms` | 区域选择两臂均可执行 |
| `response_buckets_are_variable_length_and_deterministic` | 动态 bucket 表可变长且确定性 |
| `work_plan_is_bucket_contiguous_generation_safe_and_capacity_bounded` | work plan 的 bucket 连续性、generation 和容量边界 |
| `continuous_blend_emits_two_verified_arms_and_packed_response_meta` | 连续混合两臂与 metadata 完整 |
| `split_outputs_must_borrow_identical_lighting_context_not_equal_copies` | split 两臂共享同一 lighting context |
| `pair_reads_one_visibility_sample_and_shared_terms_with_two_response_calls` | 共享项只采样一次 |
| `mask_blend_materials_are_excluded_from_conservation_suites_and_counted` | 混合材质机械排除并计数 |
| `production_composition_gpu_matches_cpu_all_domains_alpha_and_auxiliary` | GPU composition 与 CPU 在通道、alpha、auxiliary 上一致；当前 harness 与生产入口共用共享 SceneFabric/runtime 的 recorder，acceptance-only `CompositionFabric` 使用同一 scene selection 构造器，不走独立 test-only composition encoder；此处描述测试定义，不新增运行结果 |
| `hybrid_composition_golden_grid_matches_legacy_reference` | 水体组合网格与历史 reference 一致 |
| `hybrid_runtime_is_wired_through_resident_composition_and_hdr_export` | 验证 resident composition 与 HDR export 的源码接线形态；不证明完整 Scene/Fabric/TAA 闭环 |
| `water_legacy_gpu_parity_*_full_frame`（8 项） | 强制 GPU 时，逐像素验证 16×16 水体 fixture 的冻结历史 shader 输出与独立 overlay 组合公式，不代表完整应用帧 parity 或删除前护栏时序 |
| `water_gpu_*`（10 项） | 强制 GPU 时，验证当前 ABI 的定向生产水体输出 |
| `hybrid_wesl_compiles_and_preserves_all_four_entries` | 四个现行入口可编译，已删除的 `water_surface_fs_hybrid` 不会回归 |

## 8. 当前源码与测试证据

本节涉及的 prism_render_scene、Bevy 或 SceneRhi 记录属于历史/排除路径；响应与组合当前验收只接受 Prism 原生 RHI/RenderGraph 证据。

直接源码位置：

- plan 与 contract：`pkg/prism_render_material/src/shading_plan.rs`；
- execution table 与 package 校验：`pkg/prism_render_material/src/execution.rs`；
- bucket 与 work plan：`pkg/prism_render_shading/src/classification.rs`；
- response output / conservation：`pkg/prism_render_shading/src/response_output.rs` 与 `tests/response_output.rs`；
- 水体计划与 fragment entry：`pkg/prism_render_architecture/src/water/mod.rs`、`water/gpu/surface_pass.rs`、`pkg/prism_render_scene/src/shaders/water_surface_raster.wesl`；
- 水体 golden：`pkg/prism_render_scene/tests/water_hybrid_golden.rs`。
- 水体真实 GPU 输出与整帧 parity：`pkg/prism_render_scene/tests/water_hybrid_gpu.rs`。

历史文档记录（以下三条不是 2026-10-20 本轮复跑结果）：

- `cargo test -p prism_render_material --tests`：相关 shading plan integration 33 项通过，stale id、registered source、两臂执行和连续 mask 测试均通过；
- `cargo test -p prism_render_shading --tests`：`3387 passed; 0 failed; 6 ignored`，response composition workload 1 项通过，response output 27 项通过；
- `cargo test -p prism_render_scene --tests`：`1771 passed; 0 failed`，response composition GPU 2 项、water hybrid golden 17 项及 resident composition 3 项通过；

2026-10-20 当前工作树审计与复跑：

- 补强测试 oracle：reactive/overlay 的 motion 输入改为独立 PBR 输出；增加已删除 Hybrid entry 不得回归的编译源码断言。
- `cargo test -p prism_render_scene --test water_hybrid_golden --test water_hybrid_gpu -- --nocapture`：沙盒外 Apple M2 / Metal，`water_hybrid_golden` **19 passed，0 failed，0 ignored**，`water_hybrid_gpu` **19 passed，0 failed，0 ignored**。GPU 结果包含 10 项当前生产水体输出、8 项 `water_legacy_gpu_parity_*_full_frame` 和 1 项比较器容差测试；未把无适配器 skip 当作通过。
- 八项 legacy GPU parity 现在确实执行冻结的 `41a31ceef24e59ab7f8dda076b2b4c7f5e11f6ec` shader closure，并分别固定 PBR/NPR endpoint；它证明 16×16 水体 fixture 的完整输出，不证明完整 Scene/Fabric/TAA 帧 parity，也不能追认旧入口删除前护栏曾经运行。
- 独立 TAA reactive MRT 已接入当前生产路径：`ViewVisibilityBuffer` 持有独立 `R8Unorm` mask，水体 fragment 作为第三个 MRT 输出，无水/opaque 帧先清零，TAA resolve 通过 binding 5 读取并削减 history blend。2026-10-20/21 的历史记录为 TAA 定向 `6/6`、resolve shader `56/56`、最终 retirement 单测 `1/1`；当时尚未重跑完整 TAA suite。后续 2026-10-22 strict GPU 筛选已通过 `19/19`（见 §10），关闭真实 bind-group、queue completion 与局部退休生命周期的验证缺口，不代表完整 TAA CPU suite、完整场景图像 parity 或跨适配器验收已复跑。
- resident composition 的生产入口现为 `response_composition_pass`：消费 `Res<SceneFabric>` 与 `Res<RenderFabricFrame>`，通过 `fabric_runtime::prepare_with_templates` 的 `compile_active → prepare_runtime`，导入真实 response buffers、HDR/visibility textures，并绑定真实 pipeline 和 bind group，经 `SceneRhi` encode 后交给 `RenderContext::add_command_buffer`。composition 声明在设备初始化冻结 selection 前并入 scene catalog/roots；不是另建生产 selection。`CompositionFabric` 只留作 acceptance wrapper，使用同一 scene selection 构造器；`resident_composition_gpu` 与生产入口共用 `record_composition_fabric_commands`。2026-10-22 修订后的 harness 已在严格模式通过 `3/3`（见 §10），局部待复跑缺口已关闭；共享 selection/runtime 的局部 compute 切片不等于全场景 recorder、history/TAA 或完整帧 parity。
- 当前仍未闭合：`SceneFabricHistory` 的 completion-gated lifecycle 已有真实 host-queue hook 和定向单元测试。water compute adapter 只导入真实本帧 surface/foam；另一个 per-instance spray feedback adapter 已接入精确 `WATER_SPRAY_RESOURCE @ frame-1` publication/import、completion 校验、显式清零 fallback，以及 resize/removal 后保留 RHI lease 的安全 retirement。上述 water/spray 专属运行验收未由本次四份日志提供，不用 composition/TAA 局部通过替代。composition graph 的八个 accesses 均为当帧资源（`frame_lag=None`），没有应当发布的 temporal output，因此不把“composition 未消费 history”列为缺陷；真正待闭合的是 TAA/history、完整 Scene payload、全域 Fabric imports 与共享 production recorder 的整帧链路。
- 仍缺完整 Scene/Fabric/TAA 帧的前后版本 parity，以及删除旧入口前已运行护栏的历史记录。TAA history 的真实双槽、exact binding、completion-gated publication/retirement 已由 2026-10-22 strict GPU `19/19` 局部验证（见 §10）；统一全场景 runtime/recorder、完整场景图像 parity 与跨适配器验收仍未闭合。

本轮复跑口径：`water_hybrid_golden` 为 `19/19`，沙盒外 `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test water_hybrid_gpu -- --nocapture` 为 `19/19`，adapter 为 Apple M2 / Metal。跨适配器门禁单独位于 `tests/cross_adapter_acceptance.rs`；本机只有一个 physical adapter，默认输出 `SKIP_UNVERIFIED`，严格模式失败。故水体 `19/19` 只能证明 16×16 fixture 在一个适配器上的局部 output/parity，不扩张为完整 Scene/TAA 图像 parity、第二适配器通过或 Render Fabric 生产闭环。

### 8.1 2026-11-09 原生 Prism 当前未完成项

响应 contract、bucket、MaskBlend 公式以及局部 composition/TAA 证据不能代替原生全帧闭环。当前仍需：

1. 在 prism_render_app 原生 RHI/RenderGraph 中把 Lit/NPR/Unlit/Registered、fused/split、两臂共享 lighting context、composition 与 HDR export 接入同一生产帧，并证明实际 draw/dispatch、barrier、bind group 和 completion 顺序。
2. 完成 water、reactive mask、TAA/history 与 response planes 的原生多视图隔离、精确 frame/version/layout/physical identity、缺历史清零和退休；历史 Bevy/SceneRhi harness 是排除项。
3. 完成任意注册 response program 的 package/entry/ABI/digest/PSO 真实消费和失败恢复；不以 fallback 或 CPU golden 冒充真实原生 GPU 输出。
4. 取得完整帧前后版本图像 parity、第二 physical adapter、资源/带宽/队列/缓存性能和 P50/P95/P99 证据。当前局部 water/composition/TAA 数字只在各自日志范围内有效。

## 9. Git 历史锚点

- `52d407879 feat(material): add response contracts and execution tables`：引入 response contract 与 execution table。
- `776439ee3 feat(material): validate shading plans against response contracts`：将计划校验接入注册和解析。
- `ce6af1163 feat(material): add custom response production registration`：加入注册程序的生产路径。
- `bdca810a9 feat(render): complete scene execution and water GPU path`：接通 response execution、水体 shader 和 GPU 测试。
- `8400fe972 feat: complete runtime and render refactoring implementation`：汇总 runtime/render 执行路径。
- `8400fe972`（2026-10-16 01:31:43 +08:00）：以 `water_surface_fs_mask_blend` 替换 `water_surface_fs_hybrid`；`a1e33a798`（2026-10-16 18:30:52 +08:00）才加入八项 `water_legacy_gpu_parity_*_full_frame`。故没有可追溯证据证明“删除前护栏已运行”；当前仅能以入口不存在断言防止旧 API 回归，历史时序护栏仍未完成。

当前结论以现有源码、门禁和测试为准；历史设计中的固定 9 类表、未来切片顺序和水体入口保留条件不属于当前契约。

## 10. 最终收口验收矩阵 / 未完成项（2026-10-21 文档审计）


**2026-10-22 最新复验与状态修正**：architecture 改动已提交 `652404b94`。main 对当前架构源码实际复验 `render_fabric` **200 passed / 0 failed / 0 ignored**、`shader_package` **95 passed / 0 failed / 0 ignored**、acceptance **3 passed / 0 failed / 0 ignored**；日志与范围见 05 §15.2。本文较早的 185 项及 shader-package/acceptance“未复跑”描述仅代表此前截点，以本补记为准。acceptance 的 3 项验证 ledger/fanout/partial 标记，不是完整 32 gate 端到端通过。

**局部 GPU 桥接证据修正**：既有日志已记录 `fabric_runtime` 定向筛选 **29 passed**（CPU/GPU 混合，不能全部算硬件用例），其中 authenticated `from_domain` bridge 在 Apple M2/Metal 完成真实上传、dispatch 和子范围隔离；RHI pipeline bridge 筛选 **15 passed**，包含 exact native render-pipeline identity/generation/lease 与原生绘制。这里只关闭相应局部桥接验证，不能关闭 Hair/Volume/Decal 的真实语义渲染、typed graphics 全链、统一 Scene/Fabric/TAA full-frame、第二 adapter 或性能验收。

**2026-10-21 收口补记（历史截点）**：此前 `prism_render_scene` library `cargo check` 曾通过；`prism_asset` 的 `Flight` 初始化缺少 `cancellation_waiters` 字段是后续一次尝试的历史阻断，当前定义已无该字段，后续反馈已越过 asset 初始化。当时文档同步仅核对源码：composition 已接共享 `SceneFabric` selection、当前帧与通用 runtime 的真实 import/encode 切片；当时未运行 Cargo，不新增 recorder 测试通过、TAA GPU completion 或 full-frame parity 证据。后续局部复验更新如下，全场景验收目标保持不变。

**2026-10-22 局部 GPU 复验**：main 实际运行并确认 terminal exit 0，本次文档同步只核对日志、未运行 Cargo。`PRISM_REQUIRE_GPU=1` 下 `shading::taa::resources::gpu_tests::` 在 Apple M2 / Metal 上 **19 passed / 0 failed / 0 ignored**（日志 `/private/tmp/prism-taa-gpu-20261022-retry.log`，含 lifecycle 7 项）。覆盖生产 TAA 双槽的真实 import/bind-group、CPU golden、共享帧时钟、同帧重试与重复编码拒绝、多视图隔离，以及未提交命令在 resize/removal/RHI 表替换后的 completion 和退休回收。较早历史 `fabric_runtime::tests::multipass_gpu` 为 **8 passed / 0 failed / 0 ignored**（日志 `/private/tmp/prism-fabric-runtime-gpu-20261022-fixed.log`），其中 **7 项硬件 GPU + 1 项 CPU 编译契约 fixture**，不是 8 项硬件用例；最新 9/9 见下文；历史 8 项仅证明通用 runtime helper 的局部行为，不是统一 Scene/Fabric/TAA 全帧验收。

**2026-10-22 resident composition integration 复验**：main 实际运行并确认 terminal exit 0，本次已读取 `/private/tmp/prism-resident-composition-gpu-20261022-aligned.log`：`PRISM_REQUIRE_GPU=1` 下 `resident_composition_gpu` **3 passed / 0 failed / 0 ignored**。GPU producers→生产共用 recorder→HDR 及 resident targets/pipeline identity 两项记录 Apple M2 / Metal / IntegratedGpu；另含缺少 HDR storage support 的 capability 拒绝测试，不将三项全部计作硬件用例。修订后的局部 harness 待复跑缺口已关闭，不证明 31-pass 全场景 recorder、统一 TAA/history 或完整帧 parity。

**2026-10-22 material package 复验**：main 实际运行并确认 terminal exit 0，本次已读取 `/private/tmp/prism-render-material-package-20261022.log`：`prism_render_material` unit **888 passed / 0 failed / 0 ignored**，integration **79 passed / 0 failed / 0 ignored**，doc 0 tests。integration 包含 freshness 19、header ABI 4、PCM parity 7、S5 ABI/cache 4、schema parity 10、ShadingPlan 33、surface ABI mirror 2，子集不与 79 重复相加。该 package 的待复跑缺口已关闭；`prism_render_shading`、architecture shader-package/acceptance 与其他 integration targets 不据此声称已复跑。

**2026-10-22 局部证据补记**：main 实际复验 `prism_render_architecture --lib render_fabric` **200/200、terminal exit 0**（含新增 9 项对象资源 CPU 契约；旧 **176/176** 为历史，不重复相加）。main 实际运行并确认 exit 0 的 strict multipass 为 **9/9 = 8 项 Apple M2/Metal 硬件 + 1 项 CPU fixture**，本次已核对 `/private/tmp/prism-fabric-object-runtime-gpu-20261022.log`；旧 **8/8 = 7 hardware + 1 CPU fixture** 为历史。新增对象 fixture 覆盖 exact object/generation/history/role 寻址、完整上传/错误 role 拒绝与子分配隔离；不代表 `from_domain` 桥接 GPU、三域语义渲染、整个 shader-package/acceptance 或完整帧验收。

**2026-10-22 Hair/Volume/Decal 生产接线边界修正**：Hair 与 Decal 的 catalog 仍使用 **256B persistent STORAGE buffer** 泛化模板，每域仅声明 1 个 compute pass 和 3 个 `StorageWrite`，尚不匹配 strand arrays 或 decal receiver/material ABI；仍需真实资源/pass/access、生产数据、pipeline/bind group、draw/dispatch 与下游消费者。Volume 的 catalog 同样是泛化模板，不能代表现有 volumetric fog：`pkg/prism_render_scene/src/shading/volumetrics` 已有按 view 缓存的真实 3D froxel textures、scatter/integrate/apply pipelines、bind groups、scene depth/color 输入及 copy-back。Volume 真正未完成的是把该生产路径纳入冻结 SceneFabric 的真实纹理布局、pass 依赖/hazard、typed recorder 和宿主 record/submit；scene-color copy-back 尚未作为 Fabric pass。不得用占位 buffer、probe 或伪造输出冒充集成；`from_domain` 的局部 GPU 上传/dispatch/subrange 证据不覆盖 volumetrics 语义或全帧闭环。

**2026-10-23 Volume 状态更正**：真实 scatter/integrate/apply/copy-back pass 现已纳入 frozen SceneFabric typed recorder；Apple M2/Metal 上 production dispatch/readback 与 lifecycle 两个 strict GPU 用例分别 **1/1** 通过，设备 `max_immediate_size=256` 高于 96-byte 门槛。生命周期用例覆盖 resize/disable/missing-input 的 per-view/cache 行为；完整 schedule 下的 completion-gated resource retirement 仍待验收。Hair/Decal 和全场景统一 recorder 缺口仍在。

证据标签：`S`=源码完成或接线，`D`=测试定义，`R`=对应版本运行通过，`H`=已有记录，`G`=实际 GPU 证据，`E`=全场景闭环。`prism_render_scene --lib` 的 1909/1909 是此前完整运行记录，不能替代上述 2026-10-22 package/integration 与局部 GPU 复验，也不表示最新全库已复跑。水体历史 `G`、composition 3 项、TAA 19 项及 runtime 9 项各限原范围，不扩张为全帧证据。

| 条款 | S | D | R | H / G | E |
|---|---|---|---|---|---|
| ShadingPlan、五类 contract、bucket 与 stale program 拒绝 | 当前 plan、execution table、动态 bucket 和硬错误路径存在 | response/bucket/contract gate 已定义 | 2026-10-22 material unit 888、integration 79（含 ShadingPlan 33）；历史 scene lib 1909/1909 | 其他 shading integration 历史记录不算本次复跑 | 否；未覆盖完整 Scene/Fabric 每帧 |
| MaskBlend、water Hybrid、overlay、resident Composition | CPU/WESL 公式、四入口、响应平面和共享 SceneFabric/runtime 的局部 composition recorder 存在 | golden、入口、composition 和 output gate 已定义 | 2026-10-22 resident composition strict 3/3（2 硬件 + 1 capability 拒绝）；历史 scene lib 1909/1909 | composition 的 2 项硬件用例为 Apple M2/Metal 局部 recorder；历史水体 golden/GPU 各 19/19 仅为 16×16 fixture 范围 | 否；不是完整应用帧 |
| 对象级 runtime 局部切片 | exact object/generation/history/role 地址、payload upload 与子分配 preflight 存在 | Surface 对象隔离 fixture 已定义 | 2026-10-22 main strict multipass 9/9（8 hardware + 1 CPU fixture），exit 0 | Apple M2/Metal 对象子分配真实 dispatch/readback；历史 8/8（7 hardware + 1 CPU fixture） | 否；该 multipass 子集不包含 `from_domain`；另有桥接局部通过记录，仍不含三域语义消费者或全帧 |
| TAA reactive MRT 与 history | reactive mask 接线存在；TAA history 真实双槽/exact binding/completion 逻辑存在 | ABI/shader/history、生产 GPU 与生命周期回归已定义 | 历史 scene lib 1909/1909；本轮 TAA strict GPU 19/19 | Apple M2/Metal，真实 import/bind-group/queue-completion、双槽 CPU golden 与退休生命周期已局部复验 | 否；不是完整 Scene/Fabric/TAA 帧或第二 adapter |
| Composition 生产 recorder | 生产入口与 acceptance harness 共用 recorder，生产已使用共享 SceneFabric selection/runtime | resident composition GPU 测试定义存在 | 2026-10-22 integration strict 3 passed / 0 failed / 0 ignored（2 硬件 + 1 capability 拒绝） | 2 项 Apple M2/Metal 硬件用例覆盖局部 production recorder/HDR readback 与 identity；1 项为 HDR storage capability 拒绝校验，见上文日志 | 否；共享 selection/runtime 不等于统一全场景 recorder |

material package（unit 888 + integration 79）及修订后的 resident composition strict 3/3 待复跑缺口已关闭。仍未验证：其他 response/shading package、architecture acceptance 与相关 integration targets 的当前完整复验；完整 Scene/Fabric/TAA 前后版本图像 parity、统一 runtime/recorder 与全域生命周期；第二 physical adapter；完整 32 gate 端到端运行（gate 23 仍为 `partial`）；旧入口删除前八项护栏及 S2/S4/S5 历史先后顺序的可追溯记录；以及全场景资源峰值、吞吐和 P50/P95/P99 性能证据。TAA 的真实 resource import、bind-group、queue completion 与退休已取得局部 GPU 证据，但不能据此关闭完整帧验收。历史 scene library `1909/1909`、水体单 adapter `19/19`、本次 composition `3/3`、TAA `19/19` 与 runtime `9/9`（8 hardware + 1 CPU fixture）均不能升级为全系统 composition 完成。

**2026-11-09 原生修正补记**：Native composite 的 Registered MaskBlend 右臂已改为使用 `right_program_id`，并加入左右 program identity 回归断言；TAA host submission/cancel/live-retirement 接线也已补入原生 output driver。通用 Texture/VertexAttr/UV/Region response 消费、完整 split 组合、全帧 parity 与跨适配器证据仍未完成。

**2026-11-09 Water coordinator 同步**：Water simulation、spray history reset/publication 与 displacement history copy 已统一到顶层 SceneFabric coordinator 的单次 host submission；旧 Water recorder 已删除，Water 定向筛选 **230/230** 通过。该局部入口收口不等于 Composition/TAA 完整帧 parity、全域 history lifecycle、第二 adapter 或性能验收完成。
