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
- `pkg/prism_render_scene/src/shading/resources.rs`、`water/surface_draw.rs`、`shading/taa/bind_groups.rs` 与 `shaders/taa_resolve.wesl`：独立 `R8Unorm` TAA reactive MRT 的资源、清零、第三个水体输出和 resolve 消费。

### 5.2 已记录的完整验收结果

下表对应 2026-10-16 已保存的历史完整命令输出；仅对自该次运行后未变化的契约保留回归参考价值，不视为当前工作树整体通过。

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

2026-10-20 当前工作树定向验证：TAA ABI/jitter/shader 测试 `6 passed; 0 failed`；resolve shader 测试当前复跑 `56 passed; 0 failed`；`prism_app` 单元测试 `248 passed; 0 failed`，`alloc_core` `6 passed; 0 failed`；Apple M2 Metal 上 `water_hybrid_gpu` `19 passed; 0 failed`。较早的 `cargo check -p prism_render_scene --lib` 和 Render Fabric 定向复跑被当时并发 `prism_physics_core` 修改阻断（缺失 `topology_bridge` 等模块/类型）；这些历史失败不能替代当前状态的复验。target 曾因磁盘空间耗尽被清理，本轮后续的 MaterialX 测试重新编译了相关 workspace 依赖。上述结果证明对应局部契约和当前适配器输出，不等于完整 Scene/Fabric/TAA 帧 parity、Fabric 生产闭环或跨适配器验收。

本轮测试侧验收（2026-10-20）进一步固定了范围：`render_fabric_acceptance` 为 `3 passed; 0 failed`，其中 fixture 严格列出 32 个唯一 Render Fabric gate，并验证五个 subsystem domain、三个 response root、provider/resource fanout 和 builtin shader package identity；gate 23 明确保持 `partial`。较早的完整 Render Fabric 单测记录为 `134 passed; 0 failed`，后续定向运行记录更新为 `136 passed; 0 failed`（见 §8）。这些是 contract、CPU compiler/runtime fixture 和选择扇出的证据，不是 scene 生产 `compile → prepare_runtime → execute` 闭环。

Material authoring 另有一项已实现并通过定向测试的有限切片：`prism_render_material_authoring` 支持受限的 MaterialX author nodes（`add/subtract/multiply/divide/mix/clamp/convert`）、nodedef defaults、instance overrides，以及 nodegraph implementation 的输入/输出签名、多输出选择、嵌套和循环/类型错误诊断；解析索引在一次批量适配中复用。2026-10-20 `cargo test -p prism_render_material_authoring --test materialx -- --nocapture` **27 passed; 0 failed**，覆盖上述 nodegraph-instance 语义与既有 MaterialX adapter。此前 `cargo test -p prism_render_material_authoring --test evaluator` 记录为 `8 passed; 0 failed`，particle compiler 修复后的定向测试为 `21 passed; 0 failed`。此项是 bake/offline authoring 路径，不是完整 MaterialX 一致性、编辑器集成、任意图求值、完整 nodedef 继承，也不是 Render Fabric 通用图求值器。

局部 production 接线现包括 water compute、spray feedback、resident composition、Surface 帧项、Particle CPU/物理资源桥和 TAA history 源码接线。water solver adapter 导入本帧真实 surface/foam；spray feedback 另有 per-instance adapter，基于稳定 owner 和真实双缓冲 RHI lease 形成精确 `WATER_SPRAY_RESOURCE + FrameLag(1)` binding，录制前 enqueue publication/import，history 未完成时走显式清零，并在 resize/removal 后经 RenderQueue completion retirement 再回收旧 buffer/lease。TAA 现在由 `SceneRhi` 分配真实双槽 persistent texture，resolve dispatch 编码后才 enqueue exact producer，下一帧只接受 completion 已发布的上一帧 binding；resize、removal、device-table replacement 使用新的 owner 并退休旧 lease。water dispatch、spray、composition 与 TAA 仍是窄切片，不是共享全场景 recorder。`RenderFabricFrame` 从 committed Surface instances 提取帧项；原生 `RenderApp` 也会把同一份 `ExtractedParticleSnapshot` 降为经过范围、实体身份和有限位置校验的 `ParticleFabricFrame`。Particle 每个 emitter 现同时发出 count、非空 live-position bounds 的 `ValuePayload`，以及 `PARTICLE_STREAM_RESOURCE`、`PARTICLE_ALIVE_RESOURCE` 的 canonical `ResourcePayload`；`PreparedParticleFabricFrame` 以严格 consumer 将齐全且一致的 payload 降为 sprite geometry。随后 `ParticleFabricGpuFrame` 将这些语义角色解析为 renderer-owned `BufferId`，校验容量、owner 和本地 simulation/bridge/renderer graph recorder 的连续性；这仍不是 `SceneFabric` runtime 的统一 import/bind-group/record，也没有进入共享全场景 recorder。2026-10-20 后续 `cargo test -p prism_render_scene --lib` 已在当时工作树完整编译并通过 1909 项单测；2026-10-21 较早 `prism_particle_render` 与 `prism_particle_app` 定向单测分别为 **6/6**、**4/4**。此前阻断这些目标的 `Archetypes::created_count()` 已修复，`cargo test -p prism_ecs --lib diagnostics::step_inspector::tests -- --test-threads=1` 为 **8/8**；此前一次 `fabric_domains` 定向测试曾在编译阶段被无关的 `prism_asset` 工作树错误阻断（`pkg/prism_asset/src/scheduler.rs:757,876,903` 初始化 `Flight` 时缺少 `cancellation_waiters` 字段），未产生测试通过证据，不能将其表述为代码断言失败或 GPU 验收。

2026-10-21 源码同步补充：composition 声明已在设备冻结前并入 scene catalog/roots，生产 `response_composition_pass` 消费共享 `SceneFabric` / `RenderFabricFrame`，经 `fabric_runtime::prepare_with_templates` 的 `compile_active → prepare_runtime → encode` 将真实资源切片交给宿主；`CompositionFabric` 仅为使用同一 selection 构造器的 acceptance wrapper。原生 `RenderApp` 则已有 `ExtractedRenderFabricFrame` 聚合真实 Surface 与 Particle：Particle 在 Extract 阶段仅 lowering 一次，以 `RenderRecord.frame_index` 标识渲染帧，snapshot 的 simulation clock 独立；GPU Prepare 后 `finalize_render_fabric_frame` 保留 Surface、替换 GPU Particle 语义，非法输入清空整帧，disabled/pending/recovering 移除 Particle。GPU bounds 为保守边界，不把 capacity 当作 exact liveness。visibility 校验共享帧与带 generation 的 Surface 身份；host graph record/submit 前 admission 拒绝缺失、错误或错帧输入，但不覆盖更早的 Particle simulation 提交。这些源码接线不是统一全场景 recorder，也不新增测试通过证据；尝试与未复验状态见 05 §15.2。

跨适配器门禁为 `pkg/prism_render_scene/tests/cross_adapter_acceptance.rs`：默认模式发现少于两个 distinct physical adapter 时必须输出 `SKIP_UNVERIFIED`，该状态不计入验收；`PRISM_REQUIRE_GPU=1` 必须失败。当前沙盒 inventory 为 0 个 physical adapter；沙盒外真实 inventory 为 1 个 `Apple M2 / Metal / IntegratedGpu`，因此严格模式按预期失败。该 probe 只验证每个 adapter 的 storage-compute submit/readback，不冒充水体、Scene 或 TAA full-frame parity；第二个 physical adapter 仍是未验证阻塞。

## 6. Git 历史锚点

- `0de14a400 refactor surface ABI lobe and mirror gates`：建立 schema/codegen、Surface ABI 与镜像门禁。
- `352d594b1 refactor(render): unify frame graph execution planning`：统一 frame graph 执行规划。
- `bdca810a9 feat(render): complete scene execution and water GPU path`：接通场景执行与水体 GPU 路径。
- `3d066db25 feat(render): harden render fabric contracts`：加入 pipeline mapping、Render Fabric contract 和 selection 检查。
- `8400fe972 feat: complete runtime and render refactoring implementation`：汇总 runtime/render 重构实现。

这些历史提交用于确认结论的演进边界；当前完成状态按每条明确标注的源码事实、历史运行记录和本轮结果分别判断。

## 7. 交叉引用

- [`01-surface-abi_refactor_zh.md`](01-surface-abi_refactor_zh.md)：Surface schema、packed layout、header ABI 和活动瓣身份。
- [`02-response-composition_refactor_zh.md`](02-response-composition_refactor_zh.md)：响应程序、动态分桶、MaskBlend、ResponseOutput 和水体组合。
- [`03-shader-package_refactor_zh.md`](03-shader-package_refactor_zh.md)：package、entry、permutation 和 typed pipeline identity。
- [`04-frame-graph_refactor_zh.md`](04-frame-graph_refactor_zh.md)：resident composition、资源依赖、barrier 和固定图拓扑。
- [`05-render-fabric_refactor_zh.md`](05-render-fabric_refactor_zh.md)：RenderDomain、服务 provider、FeatureCatalog 和跨子系统资源边界。

## 8. 最终收口验收矩阵 / 未完成项（2026-10-21 文档审计）

**2026-10-21 收口补记**：本轮修复了 Hair/Volume/Decal domain builder、resource contract 和 consumer 中的生产非法分支：不再以 `unreachable!` 代表“已校验”，非法 Surface/Particle 输入现在返回显式错误；`fabric_runtime::prepare` 也收紧为 crate 内入口。此前 `cargo check -p prism_render_scene --features legacy-bevy --lib` 曾通过；后续一次历史重跑在无关的 `prism_asset` 改动处被 `pkg/prism_asset/src/scheduler.rs:757,876,903` 的 `Flight` 初始化缺少 `cancellation_waiters` 字段阻断。该次域帧定向测试未进入执行阶段，不能记为通过或源码断言失败。当前 `Flight` 定义已无 `cancellation_waiters` 字段，后续反馈已越过 asset 初始化；本次文档同步未运行 Cargo，不把该历史错误列为当前阻断。

本节是对当前源码和已有记录的状态审计，不是新增实现声明。本次六文档同步未运行 Cargo，下列 R/测试数字保留前序对应版本的运行口径，不表示最新 composition/native 接线已复验。原审计未运行 Cargo/GPU；其后续推进及验证记录见本节补记。标签含义：`S`=当前源码中存在的实现或接线，`D`=测试/门禁已定义，`R`=本轮运行通过，`H`=已有历史或先前运行记录，`G`=实际 GPU 证据，`E`=满足全场景生产闭环要求的证据。

| 范围 | S 源码完成 | D 测试定义 | R 本次运行通过 | H 已有记录 | G GPU 证据 | E 全场景闭环 |
|---|---|---|---|---|---|---|
| Surface ABI、响应计划、Shader Package、Frame Graph | 当前 schema/codegen、`ShadingPlan`、冻结 package/permutation/PSO 映射和 31-pass graph 均有源码 | 对应 ABI、response、cache、graph 和 water gate 已定义 | scene lib 1909/1909；architecture package suites 未复跑 | 有 2026-10-16/20 的分包记录，不能当作本轮 package 复跑 | Apple M2/Metal 有水体、组合和局部 shader/parity 记录 | 否；没有完整 Scene/Fabric/TAA 帧证据 |
| Render Fabric contract / selection / payload / history | contract、静态 selection、`FrameRenderItem`、Surface/Particle CPU producer、water/spray/composition/TAA 窄接线存在 | `render_fabric` 单测、32-gate ledger、history 测试和 acceptance fixture 已定义 | 本轮 `prism_render_scene --lib` **1909/1909**；material systems **16/16** | architecture `render_fabric` 136 项、`shader_package` 95 项及 acceptance 3 项为先前记录 | RHI bridge 及水体局部 GPU 记录不能证明 Fabric 全帧 | 否；没有统一 recorder 和全域 imports |
| Particle / Volume / Hair / Decal 扇出 | `RenderApp` 已把 `ExtractedParticleSnapshot` 降为校验后的 `ParticleFabricFrame`，并以 canonical semantic `ResourcePayload` 与严格 consumer 形成 geometry；Particle 另有 renderer-owned `BufferId` 物理资源桥和本地 graph recorder 校验；Hair/Volume/Decal 有 descriptor、resource-contract 与 binding consumer，但尚未合入共享生产帧/recorder | Particle runtime snapshot、物理 owner/容量/recorder 连续性、资源 payload/geometry 清空语义有测试；其他域有部分 fixture | `prism_particle_render` 6/6、`prism_particle_app` 4/4；`prism_ecs` step inspector 8/8；域帧测试未执行 | `fabric_domains` 曾在无关 `prism_asset` 编译错误处停止，仅保留为历史尝试；最新接线未复验 | 无四域 Scene/Fabric GPU 帧证据；Particle 的局部物理桥不等于全场景 Fabric GPU 证据 | 否；Particle 尚未进入统一 SceneFabric import/recorder，Volume/Hair/Decal 与 S6 仍未完成 |
| native 共享帧与 host admission | Surface/Particle 聚合、post-GPU Prepare finalize、visibility 与 record/submit 前 admission 源码已接线 | 真实聚合、冻结 simulation clock、非法输入清空/恢复及调度测试已定义 | 本次未运行 | 历史失败尝试见 05 §15.2，不推定最新结果 | 无新增 GPU 证据 | 否；仍非 SceneFabric 全域 import/recorder |
| TAA history | 真实双槽 persistent texture、exact binding、completion publication/retirement 已接线 | TAA ABI/shader/history 单测已定义 | scene lib 1909/1909 | 无真实 bind-group/queue-completion 或整帧 TAA parity 记录 | 无真实 GPU 证据 | 否；仍是独立窄切片 |
| 跨适配器与性能 | 严格 adapter probe 和 identity/lifetime 代码存在 | `cross_adapter_acceptance` 的 skip/strict-fail 语义已定义 | 无 | 只有单 adapter 记录；少于两个 adapter 时 `SKIP_UNVERIFIED` | Apple M2/Metal 单适配器；第二 physical adapter 未验证 | 否；无跨适配器 full-frame replay，也无全系统 P50/P95/P99 性能证据 |

本轮使用独立 Cargo target 完成 `cargo test -p prism_render_scene --lib -- --test-threads=1`（**1909 passed，0 failed**）；其中 `material::systems` 16 项及 shader/layout/history 等库内测试均通过。另 `cargo test -p prism_render_material --lib handle::tests:: -- --test-threads=1` 为 **6 passed，0 failed**。这些是 crate library 单测，不覆盖 integration targets 或强制真实 GPU 运行。注意：1909 项结果对应最后一次 TAA retirement 测试微调之前的源码；2026-10-21 以共享 target 定向复测该最终测试为 **1 passed，0 failed**，但尚未在微调后的完整源码上重跑 1909 项。同日 `prism_particle_render`、`prism_particle_app` 定向单测分别为 **6/6**、**4/4**，而此前阻断它们的 `pkg/prism_ecs/src/diagnostics/step_inspector.rs` `Archetypes::created_count()` 已由 `prism_ecs` step-inspector **8/8** 复验修复。此前一次 `cargo test -p prism_render_scene --lib fabric_domains -- --test-threads=1` 未到测试执行便被无关的 `prism_asset` 工作树错误阻断（`pkg/prism_asset/src/scheduler.rs:757,876,903` 初始化 `Flight` 时缺少 `cancellation_waiters` 字段），不计为失败断言或通过证据。仍未验证的证据为：TAA 真实 bind-group、queue completion 和 resize/removal GPU 运行；统一 `SceneFabric` 的全域 `compile → prepare_runtime → import → record/submit`；Particle 的 SceneFabric import/bind-group 与统一 recorder 消费，以及 Volume/Hair/Decal 的共享生产帧接入、binding、recorder、visibility/geometry 消费；完整 32 gate 的当前端到端运行（gate 23 仍为 `partial`）；删除旧水体入口前护栏已经运行的历史时序；第二 physical adapter 上的 probe、water/Scene/TAA replay；以及 full-scene 的图像 parity、资源峰值、吞吐和 P50/P95/P99 性能数据。故本系列当前结论是“局部源码与局部证据已形成，完整系统收口未证明”，不能写成全系统完成。
