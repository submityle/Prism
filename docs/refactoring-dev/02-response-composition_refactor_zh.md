# 02 · Shading Plan 与 Composition

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

`water_hybrid_gpu.rs` 另有八个 `water_legacy_gpu_parity_*_full_frame` 用例，分别覆盖深度、岸边、泡沫、焦散、反射、透明/overlay、motion vector 和 reactive mask。每例编译未修改的生产 WESL，绘制 PBR、NPR、MaskBlend 三个生产 fragment entry，并运行生产 caustics、underwater、resident composition 与 HDR export；读回后逐像素比较 16 个 response words、control、coverage、motion MRT、HDR 与 premultiplied color。其 oracle 以独立 PBR/NPR 输出和测试 case 重算 legacy 两臂组合，而不是把 MaskBlend 自身的 motion 作为 reactive/overlay 输入。这里的 full frame 仅指 16×16 离屏水体 fixture 的全部像素：没有运行删除前旧 shader，也不是完整 Scene/Fabric/TAA 帧的前后版本图像对比；当前两端点共同的着色回归仍可能被此公式 oracle 掩盖。

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
| `production_composition_gpu_matches_cpu_all_domains_alpha_and_auxiliary` | GPU composition 与 CPU 在通道、alpha、auxiliary 上一致；当前 harness 经真实 `CompositionFabric` recorder 编译、准备并录制 command buffer，不再走独立 test-only composition encoder |
| `hybrid_composition_golden_grid_matches_legacy_reference` | 水体组合网格与历史 reference 一致 |
| `hybrid_runtime_is_wired_through_resident_composition_and_hdr_export` | 验证 resident composition 与 HDR export 的源码接线形态；不证明完整 Scene/Fabric/TAA 闭环 |
| `water_legacy_gpu_parity_*_full_frame`（8 项） | 强制 GPU 时，逐像素验证 16×16 水体 fixture 与 legacy 组合公式一致，不代表完整应用帧 parity |
| `water_gpu_*`（10 项） | 强制 GPU 时，验证当前 ABI 的定向生产水体输出 |
| `hybrid_wesl_compiles_and_preserves_all_four_entries` | 四个现行入口可编译，已删除的 `water_surface_fs_hybrid` 不会回归 |

## 8. 当前源码与测试证据

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
- 独立 TAA reactive MRT 已接入当前生产路径：`ViewVisibilityBuffer` 持有独立 `R8Unorm` mask，水体 fragment 作为第三个 MRT 输出，无水/opaque 帧先清零，TAA resolve 通过 binding 5 读取并削减 history blend；本轮 `cargo test -p prism_render_scene --lib shading::taa -- --nocapture` 为 `6/6`，`shading::resolve::shader_tests` 当前复跑为 `56/56`。这些测试证明 ABI、shader 编译与接线，不等于完整场景图像 parity 或跨适配器验收。
- resident composition 的生产入口现为 `response_composition_pass`：它通过 `CompositionFabric` 的 `DeviceFabric::compile` → `prepare_runtime`，导入真实 response buffers、HDR/visibility textures、pipeline 和 bind group，经 `SceneRhi` 编码后交给 `RenderContext::add_command_buffer`。`resident_composition_gpu` 现调用与生产入口共用的 `record_composition_fabric_commands`，执行真实 Fabric compile/prepare/record 与 GPU readback；修订后的 harness 尚待本轮运行确认。该局部回归仍不等于共享 `SceneFabric` 的全场景 recorder、history/TAA 或完整帧 parity。
- 当前仍未闭合：`SceneFabricHistory` 的 completion-gated lifecycle 已有真实 host-queue hook 和定向单元测试。water compute adapter 只导入真实本帧 surface/foam；另一个 per-instance spray feedback adapter 已接入精确 `WATER_SPRAY_RESOURCE @ frame-1` publication/import、completion 校验、显式清零 fallback，以及 resize/removal 后保留 RHI lease 的安全 retirement。新增接线尚无 Cargo 编译/测试证据。composition graph 的八个 accesses 均为当帧资源（`frame_lag=None`），没有应当发布的 temporal output，因此不把“composition 未消费 history”列为缺陷；真正待闭合的是 TAA/history、完整 Scene payload、全域 Fabric imports 与共享 production recorder 的整帧链路。
- 仍缺完整 Scene/Fabric/TAA 帧的前后版本 parity，以及删除旧入口前已运行护栏的历史记录。当前 TAA 测试证明 ABI、shader 编译与接线，不等于完整场景图像 parity 或跨适配器验收。

本轮复跑口径：`water_hybrid_golden` 为 `19/19`，沙盒外 `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test water_hybrid_gpu -- --nocapture` 为 `19/19`，adapter 为 Apple M2 / Metal。跨适配器门禁单独位于 `tests/cross_adapter_acceptance.rs`；本机只有一个 physical adapter，默认输出 `SKIP_UNVERIFIED`，严格模式失败。故水体 `19/19` 只能证明 16×16 fixture 在一个适配器上的局部 output/parity，不扩张为完整 Scene/TAA 图像 parity、第二适配器通过或 Render Fabric 生产闭环。

## 9. Git 历史锚点

- `52d407879 feat(material): add response contracts and execution tables`：引入 response contract 与 execution table。
- `776439ee3 feat(material): validate shading plans against response contracts`：将计划校验接入注册和解析。
- `ce6af1163 feat(material): add custom response production registration`：加入注册程序的生产路径。
- `bdca810a9 feat(render): complete scene execution and water GPU path`：接通 response execution、水体 shader 和 GPU 测试。
- `8400fe972 feat: complete runtime and render refactoring implementation`：汇总 runtime/render 执行路径。
- `8400fe972`（2026-10-16 01:31:43 +08:00）：以 `water_surface_fs_mask_blend` 替换 `water_surface_fs_hybrid`；`a1e33a798`（2026-10-16 18:30:52 +08:00）才加入八项 `water_legacy_gpu_parity_*_full_frame`。故没有可追溯证据证明“删除前护栏已运行”；当前仅能以入口不存在断言防止旧 API 回归，历史时序护栏仍未完成。

当前结论以现有源码、门禁和测试为准；历史设计中的固定 9 类表、未来切片顺序和水体入口保留条件不属于当前契约。
