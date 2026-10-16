# 00 · Material Artifact 总体架构

> 本文记录 2026-10-16 工作区中已经由源码、测试和 Git 历史共同确认的材质运行形状。它描述当前契约，不列实施切片或未来迁移顺序。

## 0. 一句话

Prism 的材质链路是：

```text
Authoring Graph（可为空）
    → bake / validate / lower
    → MaterialArtifact
    → GPU Surface Shade
```

`MaterialArtifact` 把表面数据、响应计划、blend/raster 类和可移植资产引用组合在一起。响应程序与组合方式是两个正交概念：`Lit`、`Npr`、`Unlit` 和注册程序负责解释表面；`MaskBlend` 负责合并两个响应结果。

## 1. 当前 Artifact 形状

```text
MaterialArtifact
├── surface       SurfaceData（OpenPBR 核心 + 活动瓣）
├── plan          ShadingPlan
│                 ├── Single { program }
│                 └── MaskBlend { left, right, mask, domain }
├── render_class  七个 blend/raster 值
└── asset_refs    程序身份、内容摘要、静态特征
```

`MaterialRenderClass` 当前只有以下七个值：

```text
Opaque / OpaqueTwoSided / Masked / MaskedTwoSided /
Transmissive / Transparent / Additive
```

它只表达 blend/raster 家族。`Surface`、`Decal`、`Volume`、`PostProcess` 等运行域由 `MaterialDomain` / `RenderDomain` 表达，水体也是 surface domain 的特性，不占用 `MaterialRenderClass` 的枚举值。

执行身份不再由手写 illumination、custom-program 或 specialization 位段拼接。`GpuMaterialHeader` 的末尾使用五个 dense execution binding word，指向 plan、response program、package 和 permutation；`render_class` 与 `render_domain` 仍是显式的运行状态字段。

## 2. 响应与组合

当前内建响应集合为：

| 响应 | 当前表示 |
|---|---|
| Lit | `ShadingPlan::Single { BuiltinProgram::Lit }` |
| NPR | `ShadingPlan::Single { BuiltinProgram::Npr }` |
| Unlit | `ShadingPlan::Single { BuiltinProgram::Unlit }` |
| Custom | `ShadingPlan::Single { ProgramRef::Registered(id) }` |
| Hybrid / 连续混合 | `ShadingPlan::MaskBlend { left, right, mask, domain }` |

`MaskBlend` 不是响应程序，也不加入响应程序注册表。两条臂各自解析为已注册的可执行程序，组合由 resident composition pass 完成；fused 路径用于满足资格的内建组合，split 路径为两条响应臂和组合 pass 保留独立执行身份。

水体是当前 `MaskBlend` 的实际客户：`WaterShadingPlan::hybrid()` 产生 PBR/NPR 两臂，水体 surface pass 选择 `water_surface_fs_mask_blend`，再由 resident composition 与 HDR export 消费响应平面。当前 WESL 入口集合由测试固定为 `water_surface_fs_pbr`、`water_surface_fs_npr`、`water_surface_fs_custom`、`water_surface_fs_mask_blend`；`water_surface_fs_hybrid` 不在当前入口集合中。

## 3. 必须保持的架构边界

1. Surface schema、codegen 生成物、Rust mirror 与 WESL mirror 是同一 ABI 的四个可核验面；布局、槽位、字长和默认值不能在消费者侧另造一份。
2. `ClosureKind` 属于作者图 / IR 语义。`Npr`、`Custom`、`Mix`、`Layer` 等变体仍然存在；lowering 后由 `SurfaceData`、`ShadingPlan` 和执行绑定承载运行时语义。
3. `closure_mask` 是原始授权记录，`active_lobes` / `lobe_mask` 是派生活动瓣集合。执行身份和 packed surface decode 使用派生活动瓣，不使用原始授权位。
4. `MaterialRenderClass` 只负责 blend/raster；响应程序、运行域、执行族、package 和 frame-graph pass 各有独立权威。
5. `ShadingPlan` 的每个程序引用都必须解析为注册且可验证的 package entry。stale `ProgramId`、不匹配的 package revision/ABI/digest、不可执行的 artifact 都是构建或解析错误，不转成默认响应。
6. 动态 bucket key 为 `(execution_family, program_id, render_phase)`。bucket 表可变长且排序稳定；WESL 入口仍是编译期有限表，运行时不做任意函数分派。
7. `MaskBlend` 的两条臂共享同一份经过验证的 lighting context；响应输出按显式 radiance、alpha 和 auxiliary 规则合并。`MaskBlend` 材质从 conservation / white-furnace suite 中机械排除，并单独计数。
8. frame graph 的拓扑、resident composition pass、资源 owner 和 barrier 由图编译器统一管理；区域选择和 mask 数值是数据，不改变图拓扑。
9. Render Fabric 的服务需求、provider 唯一性、active-root 闭包和静态选择由 catalog / selection 契约决定；材质 PCM、FeatureCatalog、GPU Scene、Frame Graph 各自只管理自己的权威数据。

## 4. 当前门禁语义

下列名称对应当前源码中的回归断言或静态门禁，描述的是已经存在的检查：

| 检查 | 当前保证 |
|---|---|
| `committed_*` codegen freshness | schema 与提交的 Rust/WESL 生成物一致，手写派生布局不会通过 |
| Rust/WESL surface/header mirror | 字段顺序、类型、字节数和 dense execution binding 对齐 |
| `active_lobe` / contract validation | 活动瓣必须被程序消费，或带有显式忽略原因 |
| `stale_registered_id_fails_plan_validation_instead_of_falling_back` | stale 注册程序是硬错误 |
| `region_selection_materials_execute_both_response_arms` | 区域选择材质的两条响应臂均可达 |
| `response_buckets_are_variable_length_and_deterministic` | bucket 由执行族、程序身份和 phase 派生，长度可变且确定性排序 |
| `continuous_blend_emits_two_verified_arms_and_packed_response_meta` | 连续 mask 使用独立 metadata 与两臂执行 |
| `mask_blend_materials_are_excluded_from_conservation_suites_and_counted` | 混合材质不伪装成单响应守恒样本，跳过数可见 |
| `hybrid_runtime_is_wired_through_resident_composition_and_hdr_export` | 水体两臂、coverage、组合、HDR export 的生产链路连通 |
| `hybrid_wesl_compiles_and_preserves_all_four_entries` | 当前四个水体入口均存在且可编译 |

## 5. 当前源码与测试证据

### 5.1 直接源码证据

- `pkg/prism_render_material/src/record.rs`：七值 `MaterialRenderClass`、`MaterialDomain`、feature flags、`GpuMaterialHeader`。
- `pkg/prism_render_material/src/shading_plan.rs`：`BuiltinProgram::{Lit,Npr,Unlit}`、`ProgramRef::{Builtin,Registered}`、`ShadingPlan::{Single,MaskBlend}` 及五类 response contract。
- `pkg/prism_render_material/src/execution.rs`：`MaterialExecutionBinding::to_words()` 的五个 dense word 与 package/program/plan 校验。
- `pkg/prism_render_shading/src/classification.rs`：`ResponseBucketKey`、动态 bucket 表、执行族和 phase 选择。
- `pkg/prism_render_architecture/src/water/mod.rs` 与 `water/gpu/surface_pass.rs`：水体 `WaterShadingPlan::MaskBlend` 与 `water_surface_fs_mask_blend`。
- `pkg/prism_render_scene/src/shaders/water_surface_raster.wesl`：四个当前 fragment entry。
- `pkg/prism_render_architecture/src/ray_scene/footprint.rs`、`pkg/prism_render_scene/src/raytrace/footprint_gpu_tests.rs` 与 `pkg/prism_render_scene/src/shaders/ray_footprint.wesl`：CPU authoritative footprint、WESL 实现和严格 GPU parity gate。

### 5.2 已记录的完整验收结果

下表对应 2026-10-16 已保存的完整命令输出；后续未修改相关契约的条目继续作为当前证据。

| 目标 | 结果 |
|---|---|
| `cargo test -p prism_render_material --tests` | 单元 `878 passed; 0 failed`；codegen freshness 11、header ABI 3、PCM parity 7、schema parity 9、shading plan 33、surface ABI mirror 2 全部通过 |
| `cargo test -p prism_render_shading --tests` | `3387 passed; 0 failed; 6 ignored`；response composition workload 1 项通过；response output 27 项通过 |
| `cargo test -p prism_render_architecture --tests` | `11328 passed; 0 failed`；Render Fabric legacy gate、virtual shadow residency、water authoring pipeline 均通过 |
| `cargo test -p prism_render_scene --tests` | `1771 passed; 0 failed`；BRDF、classification、material VT、resident composition、response composition 和 shader 测试均通过；`water_hybrid_golden` 17 项通过 |
| `cargo test -p prism_render_architecture --lib ray_scene::footprint -- --nocapture --test-threads=1` | `16 passed; 0 failed`；CPU footprint golden、边缘输入、单调性和 mip 分桶全部通过 |
| `cargo test -p prism_render_scene --lib ray_footprint_wesl_compiles_standalone -- --nocapture --test-threads=1` | `1 passed; 0 failed`；ray footprint WESL 独立编译通过 |
| `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --lib ray_footprint_matches_cpu_golden_on_device -- --nocapture --test-threads=1` | 2026-10-16 在 Apple M2 Metal（沙盒外）`1 passed; 0 failed`；GPU footprint 与 CPU golden 一致 |

GPU 证据按运行环境记录：2026-10-16 在 Apple M2 Metal 上，response composition、resident composition、water hybrid、BRDF channels、material classification、material VT 和 ray footprint parity 门禁均有通过记录。它们证明该适配器上的 shader/module、CPU/GPU parity 与链路契约，不外推为所有适配器的性能结论或跨适配器性能结论。

## 6. Git 历史锚点

- `0de14a400 refactor surface ABI lobe and mirror gates`：建立 schema/codegen、Surface ABI 与镜像门禁。
- `352d594b1 refactor(render): unify frame graph execution planning`：统一 frame graph 执行规划。
- `bdca810a9 feat(render): complete scene execution and water GPU path`：接通场景执行与水体 GPU 路径。
- `3d066db25 feat(render): harden render fabric contracts`：加入 pipeline mapping、Render Fabric contract 和 selection 检查。
- `8400fe972 feat: complete runtime and render refactoring implementation`：汇总 runtime/render 重构实现。

这些历史提交用于确认结论的演进边界；当前文档中的完成状态以当前源码和本轮测试结果为准。

## 7. 交叉引用

- [`01-surface-abi_refactor_zh.md`](01-surface-abi_refactor_zh.md)：Surface schema、packed layout、header ABI 和活动瓣身份。
- [`02-response-composition_refactor_zh.md`](02-response-composition_refactor_zh.md)：响应程序、动态分桶、MaskBlend、ResponseOutput 和水体组合。
- [`03-shader-package_refactor_zh.md`](03-shader-package_refactor_zh.md)：package、entry、permutation 和 typed pipeline identity。
- [`04-frame-graph_refactor_zh.md`](04-frame-graph_refactor_zh.md)：resident composition、资源依赖、barrier 和固定图拓扑。
- [`05-render-fabric_refactor_zh.md`](05-render-fabric_refactor_zh.md)：RenderDomain、服务 provider、FeatureCatalog 和跨子系统资源边界。
