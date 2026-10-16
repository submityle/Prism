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

`water_surface_fs_hybrid` 不在当前入口集合中。当前水体测试直接检查四入口编译、两臂响应写入、coverage、resident composition 的通道合并和 HDR alpha 写回；这说明当前源码没有保留旧入口，但不等于“删除旧入口前的 golden/护栏顺序”已经被证明或可以取消。该历史验收要求仍未闭合，必须与当前入口集合分开记账。

`water_hybrid_golden.rs` 的组合公式覆盖深度、岸边、泡沫、焦散、反射、透明/overlay、motion vector、reactive mask、组合网格、权重边界和 WESL 参考公式；该测试集合不能扩大解释为八项 GPU 输出等价、整帧 parity，或已完成旧入口删除前的运行时护栏时序。另有 `water_hybrid_gpu.rs` 的 10 项真实输出测试，证据范围限于该 integration target 覆盖的局部 production water path/output contract，不替代上述历史护栏要求。

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
| `production_composition_gpu_matches_cpu_all_domains_alpha_and_auxiliary` | GPU composition 与 CPU 在通道、alpha、auxiliary 上一致 |
| `hybrid_composition_golden_grid_matches_legacy_reference` | 水体组合网格与历史 reference 一致 |
| `hybrid_runtime_is_wired_through_resident_composition_and_hdr_export` | 水体生产组合链路完整 |

## 8. 当前源码与测试证据

直接源码位置：

- plan 与 contract：`pkg/prism_render_material/src/shading_plan.rs`；
- execution table 与 package 校验：`pkg/prism_render_material/src/execution.rs`；
- bucket 与 work plan：`pkg/prism_render_shading/src/classification.rs`；
- response output / conservation：`pkg/prism_render_shading/src/response_output.rs` 与 `tests/response_output.rs`；
- 水体计划与 fragment entry：`pkg/prism_render_architecture/src/water/mod.rs`、`water/gpu/surface_pass.rs`、`pkg/prism_render_scene/src/shaders/water_surface_raster.wesl`；
- 水体 golden：`pkg/prism_render_scene/tests/water_hybrid_golden.rs`。

本轮运行结果：

- `cargo test -p prism_render_material --tests`：相关 shading plan integration 33 项通过，stale id、registered source、两臂执行和连续 mask 测试均通过；
- `cargo test -p prism_render_shading --tests`：`3387 passed; 0 failed; 6 ignored`，response composition workload 1 项通过，response output 27 项通过；
- `cargo test -p prism_render_scene --tests`：`1771 passed; 0 failed`，response composition GPU 2 项、water hybrid golden 17 项及 resident composition 3 项通过；
- `water_hybrid_gpu.rs` 已有 10/10 项真实输出测试证据；该数字只表示该 integration target 的十项水体 GPU 输出覆盖，不表示八项 legacy golden、整帧 parity、所有设备或 Fabric 生产闭环已完成；
- 当前宿主 Apple M2 Metal 的 GPU 结果按适配器记录，不扩展为所有设备的性能结论。

## 9. Git 历史锚点

- `52d407879 feat(material): add response contracts and execution tables`：引入 response contract 与 execution table。
- `776439ee3 feat(material): validate shading plans against response contracts`：将计划校验接入注册和解析。
- `ce6af1163 feat(material): add custom response production registration`：加入注册程序的生产路径。
- `bdca810a9 feat(render): complete scene execution and water GPU path`：接通 response execution、水体 shader 和 GPU 测试。
- `8400fe972 feat: complete runtime and render refactoring implementation`：汇总 runtime/render 执行路径。

当前结论以现有源码、门禁和测试为准；历史设计中的固定 9 类表、未来切片顺序和水体入口保留条件不属于当前契约。
