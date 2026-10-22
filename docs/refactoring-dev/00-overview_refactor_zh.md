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

测试侧验收进一步固定了范围：`render_fabric_acceptance` 的既有记录为 `3 passed; 0 failed`，其中 fixture 严格列出 32 个唯一 Render Fabric gate，并验证五个 subsystem domain、三个 response root、provider/resource fanout 和 builtin shader package identity；gate 23 明确保持 `partial`。2026-10-21 的 Render Fabric 定向单测 `175 passed; 0 failed` 保留为历史；2026-10-22 后续结果为下述 200/200。这些是 contract、CPU compiler/runtime fixture、pass alternatives、访问 lowering 和选择扇出的证据，不是 scene 生产 `compile → prepare_runtime → execute` 闭环。

2026-10-22 帧容器定向运行补记（main 实际运行，本次文档同步已读取日志、未复跑）：`cargo test -p prism_render_architecture --lib render_fabric::frame` 为 **19 passed / 0 failed**，日志 `/private/tmp/prism-fabric-frame-20261022.log`；对应 frame 改动已独立提交 `f8ab51076`。覆盖 owned vector 经 `from_items → into_items` 无重分配、`replace_domain` 保留其他域与帧号并在测试容量内复用原存储，以及错误时清空整帧和后续合法输入恢复。这是 architecture CPU 帧容器契约证据，不替代 Bevy/native 生产聚合、GPU 或完整 Scene/Fabric/TAA 帧验收；该 frame 子集不用于推增历史总数；全筛选及 catalog 的实际运行结果另记如下。

2026-10-22 architecture Render Fabric 局部复验：较早 main 运行的 `/private/tmp/prism-fabric-architecture-20261022.log` 为历史 **176/176**；后续 architecture 代理报告已实际运行 `prism_render_architecture --lib render_fabric` **200/200，terminal exit 0**，包含新增 9 项对象资源 CPU 契约测试，子集不与 185 重复相加。本次文档同步未运行构建，185 的来源是代理运行回报，不将旧 176 日志冒充新结果。该筛选不覆盖整个 `shader_package`、architecture acceptance、GPU 或全场景闭环。

2026-10-22 RenderGraph 全 package 运行补记（main 实际运行并确认 terminal exit 0，本次已读取 `/private/tmp/prism-render-graph-20261022.log`）：`cargo test -p prism_render_graph`（无 `--lib` 限制）unit **122 passed / 0 failed / 0 ignored**；doc **1 passed / 0 failed / 1 ignored**。doc ignored 不计作通过。旧 85/85 与磁盘耗尽中止尝试仅保留为历史，不再代表当前状态；当前 package 验证缺口已关闭，不外推为全场景 GPU、full-frame 或跨适配器通过。

2026-10-22 public Domain GPU integration：沙盒内 0 adapter，严格模式拒绝；沙盒外以 `PRISM_REQUIRE_GPU=1` 重试 `--test fabric_domain_gpu`，Apple M2 / Metal 上 **1 passed / 0 failed / 0 ignored**，terminal exit 0，日志 `/private/tmp/prism-fabric-domain-gpu-20261022-outside.log`。覆盖公开 API 的设备对齐子区间绑定与 lease 保留；不证明域 shader、draw/dispatch 或全场景渲染。

2026-10-22 runtime multipass 局部复验：较早 `/private/tmp/prism-fabric-runtime-gpu-20261022-fixed.log` 的 **8/8（7 hardware + 1 CPU fixture）**保留为历史；main 后续实际运行并确认 terminal exit 0，本次核对 `/private/tmp/prism-fabric-object-runtime-gpu-20261022.log`，`PRISM_REQUIRE_GPU=1` 下 `fabric_runtime::tests::multipass_gpu` **9 passed / 0 failed / 0 ignored**，其中 **8 项 Apple M2 / Metal 硬件用例 + 1 项 CPU 编译契约 fixture**。新增 `object_suballocations_upload_and_execute_by_exact_identity` 使用 Surface 对象 fixture，验证 object index/generation/history/role 精确寻址、完整 payload 上传与错误 role/短上传拒绝、缺失/重复绑定和重叠拒绝，以及同一真实 buffer 子分配经多 pass 执行/readback 后互不串写；既有 lease/preflight、录制顺序、push constants 与失败清理仍各限测试范围。不是 Hair/Volume/Decal 语义 shader 或 `SceneFabricObjectResource::from_domain` 桥接 GPU 验证，也不是全帧或跨适配器证据。

2026-10-22 resident composition integration 复验（main 实际运行并确认 terminal exit 0，本次已读取 `/private/tmp/prism-resident-composition-gpu-20261022-aligned.log`）：`PRISM_REQUIRE_GPU=1` 下 `resident_composition_gpu` **3 passed / 0 failed / 0 ignored**，硬件用例标记 Apple M2 / Metal / IntegratedGpu。覆盖 GPU producers 经生产共用 recorder 写入 HDR、resident targets/pipeline identity 校验及缺少 HDR storage support 的 capability 拒绝。修订后的局部 harness 待复跑缺口已关闭；3 项结果不等于 31-pass 全场景 recorder、完整 Scene/Fabric/TAA 帧、第二 adapter 或全 32 gate 验收。

2026-10-22 material package 复验（main 实际运行并确认 terminal exit 0，本次已读取 `/private/tmp/prism-render-material-package-20261022.log`）：`prism_render_material` unit **888 passed / 0 failed / 0 ignored**，integration 合计 **79 passed / 0 failed / 0 ignored**，其中 codegen freshness 19、header ABI 4、PCM parity 7、S5 ABI/cache 4、schema parity 10、ShadingPlan 33、surface ABI mirror 2；这些子集不与 79 重复相加。doc 为 0 tests。该 package 的待复跑缺口已关闭，不代表 `prism_render_shading`、architecture shader-package 或全场景 GPU 验收已复跑。

2026-10-22 TAA GPU 首次阻断与重试通过记录：main 的首次 TAA GPU 日志 `/private/tmp/prism-taa-gpu-20261022.log` 因依赖 `prism_physics_core/src/world.rs:651` 缺少 `prepare_assembly_collider` 报 E0599，未执行测试，仅保留为历史编译阻断；随后重试已成功，不再列为当前待复跑状态。main 实际运行并确认 terminal exit 0，本次文档同步已读取 `/private/tmp/prism-taa-gpu-20261022-retry.log`：`prism_render_scene --lib` 的 `shading::taa::resources::gpu_tests` 筛选在 `PRISM_REQUIRE_GPU=1` 严格模式下，Apple M2 / Metal REAL HARDWARE **19 passed / 0 failed / 0 ignored**，2021 filtered out，0.84s，包含新增 lifecycle 7 项。覆盖真实 bind-group、两帧 publication/import 与 CPU golden、queue completion，以及 resize/removal/device-table replacement 的 retirement、未提交 dispatch 资源保留与 ID 回收。日志中的 stale ID panic 由 `catch_unwind` 捕获，是预期失效断言，最终测试通过。TAA 局部 GPU 验证缺口已关闭；完整 Scene/Fabric/TAA full-frame parity、统一全场景 runtime/recorder 和第二 physical adapter 仍未闭合。

2026-10-23 Volumetrics strict GPU 复验：在 Apple M2 / Metal 上以 `PRISM_REQUIRE_GPU=1 WGPU_BACKEND=metal` 实际运行 `shading::volumetrics::gpu_tests::production_scene_fabric_fog_dispatch_and_copy_back`，**1 passed / 0 failed**；device `max_immediate_size=256`，满足生产管线的 96-byte capability gate。测试使用生产 Volumetrics pipelines/layout、visibility/SSR/froxel bind groups 与 `volumetrics_pass`，经 shared SceneFabric lowering/typed recorder 提交 scatter→integrate→apply→copy-back，读取 fog target 与 scene-color，校验输出有限、fog 非空且改变画面、copy-back 像素一致。该项关闭 Volumetrics 单次 dispatch/readback 的硬件验证缺口；不覆盖 resize/disable/missing-input 生命周期、完整多域帧、31-pass parity、第二适配器或全场景闭环。

2026-10-23 Volumetrics lifecycle strict GPU：`production_view_lifecycle_rebuilds_and_retires_froxel_resources` 在 Apple M2/Metal、`PRISM_REQUIRE_GPU=1` 下实跑 **1 passed / 0 failed**。覆盖 viewport resize 后 texture ID 重建、visibility missing 时 apply group 缺席且 dispatch 不录制、zero viewport 清除 per-view components/cache、disable 清空 cache。早先 GPU 子集合跑因 `geometry/mod.rs` 移除了新 `representation` 模块声明与导出而编译失败；恢复声明/导出后，strict 合跑 `shading::volumetrics::gpu_tests` **2 passed / 0 failed**。完整 schedule 下 completion-gated 资源退休仍未验证。

2026-10-23 Particle production GPU 复验：`PRISM_REQUIRE_GPU=1 WGPU_BACKEND=metal cargo test -p prism_render_app --lib physical_gpu_multioutput_cameras_execute_simulation_and_bridge_once -- --nocapture` 在 Apple M2/Metal strict 模式 **1 passed / 0 failed**。该用例从 production `ParticleGpuState` 路径执行 GPU simulation、shared bridge 与多相机 renderer，验证相机输出/读回及 bridge 在多视口图中的单次生产语义；它不接入 Bevy `prism_render_scene` SceneFabric runtime，也不证明跨域 imports、全帧 recorder 或 completion-gated retirement。`particle_fabric_` CPU 筛选另为 **5/5**，只验证 payload/geometry/物理 buffer 角色绑定契约。

2026-10-23 Particle Fabric pass contract：builtin catalog 现将 renderer-input stream 声明为 `PARTICLE_BRIDGE_PASS` 的 storage write，由后继 `PARTICLE_RENDER_PASS` 以 `ProducedBy`/storage read 消费，并在该 pass 产出 alive-index buffer；architecture `render_fabric` 定向套件 **202/202** 实跑通过。Bevy `prism_render_scene --features legacy-bevy --lib` check 通过；原生 app `feature_mapping::tests` **12/12** 通过。另移除源码明确标注的未发布别名 `FixedPageStreamingManager`、`release_after()` 与 `WATER_SPRAY_RESOURCE`，调用点改用唯一语义资源 `PARTICLE_SPRAY_RESOURCE`/`retire()`/`FixedPageStreamingSource`。本轮顺带对齐并行变更引入的 `SurfaceViewGraphInput.clear` 与 `prepare_view_bindings.vt_layout` 测试调用。以上是契约、crate check 和 API 收口证据，不证明跨 crate 的同一 runtime/RHI production recorder 已打通。

Material authoring 另有一项已实现并通过定向测试的有限切片：`prism_render_material_authoring` 支持受限的 MaterialX author nodes（`add/subtract/multiply/divide/mix/clamp/convert`）、nodedef defaults、instance overrides，以及 nodegraph implementation 的输入/输出签名、多输出选择、嵌套和循环/类型错误诊断；解析索引在一次批量适配中复用。2026-10-20 `cargo test -p prism_render_material_authoring --test materialx -- --nocapture` **27 passed; 0 failed**，覆盖上述 nodegraph-instance 语义与既有 MaterialX adapter。此前 `cargo test -p prism_render_material_authoring --test evaluator` 记录为 `8 passed; 0 failed`，particle compiler 修复后的定向测试为 `21 passed; 0 failed`。此项是 bake/offline authoring 路径，不是完整 MaterialX 一致性、编辑器集成、任意图求值、完整 nodedef 继承，也不是 Render Fabric 通用图求值器。

局部 production 接线现包括 water compute、spray feedback、resident composition、Surface 帧项、Particle CPU/物理资源桥和 TAA history 源码接线。water solver adapter 导入本帧真实 surface/foam；spray feedback 另有 per-instance adapter，基于稳定 owner 和真实双缓冲 RHI lease 形成精确 `WATER_SPRAY_RESOURCE + FrameLag(1)` binding，录制前 enqueue publication/import，history 未完成时走显式清零，并在 resize/removal 后经 RenderQueue completion retirement 再回收旧 buffer/lease。TAA 现在由 `SceneRhi` 分配真实双槽 persistent texture，resolve dispatch 编码后才 enqueue exact producer，下一帧只接受 completion 已发布的上一帧 binding；resize、removal、device-table replacement 使用新的 owner 并退休旧 lease。water dispatch、spray、composition 与 TAA 仍是窄切片，不是共享全场景 recorder。`RenderFabricFrame` 从 committed Surface instances 提取帧项；原生 `RenderApp` 也会把同一份 `ExtractedParticleSnapshot` 降为经过范围、实体身份和有限位置校验的 `ParticleFabricFrame`。Particle 每个 emitter 现同时发出 count、非空 live-position bounds 的 `ValuePayload`，以及 `PARTICLE_STREAM_RESOURCE`、`PARTICLE_ALIVE_RESOURCE` 的 canonical `ResourcePayload`；`PreparedParticleFabricFrame` 以严格 consumer 将齐全且一致的 payload 降为 sprite geometry。随后 `ParticleFabricGpuFrame` 将这些语义角色解析为 renderer-owned `BufferId`，校验容量、owner 和本地 simulation/bridge/renderer graph recorder 的连续性；这仍不是 `SceneFabric` runtime 的统一 import/bind-group/record，也没有进入共享全场景 recorder。2026-10-20 后续 `cargo test -p prism_render_scene --lib` 已在当时工作树完整编译并通过 1909 项单测；2026-10-21 较早 `prism_particle_render` 与 `prism_particle_app` 定向单测分别为 **6/6**、**4/4**。此前阻断这些目标的 `Archetypes::created_count()` 已修复，`cargo test -p prism_ecs --lib diagnostics::step_inspector::tests -- --test-threads=1` 为 **8/8**；此前一次 `fabric_domains` 定向测试曾在编译阶段被无关的 `prism_asset` 工作树错误阻断（`pkg/prism_asset/src/scheduler.rs:757,876,903` 初始化 `Flight` 时缺少 `cancellation_waiters` 字段），未产生测试通过证据，不能将其表述为代码断言失败或 GPU 验收。

2026-10-21 源码同步补充：composition 声明已在设备冻结前并入 scene catalog/roots，生产 `response_composition_pass` 消费共享 `SceneFabric` / `RenderFabricFrame`，经 `fabric_runtime::prepare_with_templates` 的 `compile_active → prepare_runtime → encode` 将真实资源切片交给宿主；`CompositionFabric` 仅为使用同一 selection 构造器的 acceptance wrapper。原生 `RenderApp` 则已有 `ExtractedRenderFabricFrame` 聚合真实 Surface 与 Particle：Particle 在 Extract 阶段仅 lowering 一次，以 `RenderRecord.frame_index` 标识渲染帧，snapshot 的 simulation clock 独立；GPU Prepare 后 `finalize_render_fabric_frame` 保留 Surface、替换 GPU Particle 语义，非法输入清空整帧，disabled/pending/recovering 移除 Particle。GPU bounds 为保守边界，不把 capacity 当作 exact liveness。visibility 校验共享帧与带 generation 的 Surface 身份；host graph record/submit 前 admission 拒绝缺失、错误或错帧输入，但不覆盖更早的 Particle simulation 提交。这些源码接线不是统一全场景 recorder，也不新增测试通过证据；尝试与未复验状态见 05 §15.2。

Bevy `prism_render_scene` 路径现另有 Hair/Volume/Decal 的生产帧聚合 seam：`HairRenderWorldOwner`、`VolumeRenderWorldOwner`、`DecalRenderWorldOwner` 在公开的 `CollectDomainContributions` 调度集合中按 `FabricDomainAggregationStatus::next_frame()` 发布 `FabricDomainFrame`，`BuildFrame` 随后一次性 drain 并与 committed Surface 合成共享 `RenderFabricFrame`。owner 保持稳定 object/history/motion identity；Hair/Decal 使用持久快照，Volume plan 单次消费，producer 停止发布后不回放旧项。同帧多 producer 可合并；依据 `pkg/prism_render_scene/src/fabric_frame.rs` 的 `publish` / `drain_for_frame`、`build_render_fabric_frame_system` 与 `rebuild`，失败边界须分层描述：

1. domain 贡献预校验拒绝（混合源帧、域批内或跨 producer 重复 identity/role、非法 payload），或 drain 时 pending 帧号不符：丢弃域贡献，以空 additional 重建；仅在 Surface 重建合法时发布 Surface-only，并保留可查询 rejection，accepted items 为 0。
2. 共享最终 rebuild 校验失败（包括域贡献与 Surface 冲突），或 Surface 提取/校验失败：清空整帧，包括旧项和本次已积累的部分项，报告 `SharedFrame` rejection；不重试 Surface-only，也不发布成功前缀。
3. clock exhaustion：不复用最后成功帧，清空 pending 贡献及其锁存 rejection，并清空共享帧；保留耗尽的帧号，报告 `FrameClockExhausted`，accepted items 为 0。

`fabric_domains::gpu` 进一步将 canonical semantic role 接纳为带 table/physical identity 的 `SceneRhi` buffer/texture lease，校验容量、range、alignment、usage、texture layout/default view、exact binding/import coverage 与物理别名，并生成运行时 `ResourceId` import 和 record-ready 借用视图。该层只完成 physical-resource admission/import，不含 pipeline、bind group、draw/dispatch recorder、visibility/geometry consumer 或 full-frame parity。Bevy 路径与原生 `RenderApp` 的 Surface/Particle 路径仍是两条集成路径，不合并记作一个 GPU 闭环；上述失败边界不缩减全场景闭环和完整验收目标。

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


**2026-10-22 最新复验与状态修正**：architecture 改动已提交 `652404b94`。main 对当前架构源码实际复验 `render_fabric` **200 passed / 0 failed / 0 ignored**、`shader_package` **95 passed / 0 failed / 0 ignored**、acceptance **3 passed / 0 failed / 0 ignored**；日志与范围见 05 §15.2。本文较早的 185 项及 shader-package/acceptance“未复跑”描述仅代表此前截点，以本补记为准。acceptance 的 3 项验证 ledger/fanout/partial 标记，不是完整 32 gate 端到端通过。

**局部 GPU 桥接证据修正**：既有日志已记录 `fabric_runtime` 定向筛选 **29 passed**（CPU/GPU 混合，不能全部算硬件用例），其中 authenticated `from_domain` bridge 在 Apple M2/Metal 完成真实上传、dispatch 和子范围隔离；RHI pipeline bridge 筛选 **15 passed**，包含 exact native render-pipeline identity/generation/lease 与原生绘制。这里只关闭相应局部桥接验证，不能关闭 Hair/Volume/Decal 的真实语义渲染、typed graphics 全链、统一 Scene/Fabric/TAA full-frame、第二 adapter 或性能验收。

**2026-10-22 Volume SceneFabric 接线进展（未完成）**：生产 `volumetrics_pass` 现已改为以真实 froxel/depth/scene-color 纹理模板声明 scatter、integrate、apply 与 transfer copy-back，并通过共享 `fabric_runtime::prepare_with_templates`/typed recorder 记录；SceneFabric builtin 纹理 layout 与 SSR depth、visibility HDR scene-color 的真实 usage 对齐。已实跑 `cargo check -p prism_render_scene --lib` 通过，catalog contract 定向测试 **1/1** 通过。该代码尚未完成强制物理 GPU shader/dispatch/copy-back/readback 验收，也未验证宿主设备对 96-byte immediates 的能力、resize 与缺输入生命周期；不得据此将 S6 或全系统标记完成。本轮未清理共享 `target`，亦未提交。

**2026-10-21 收口补记**：本轮修复了 Hair/Volume/Decal domain builder、resource contract 和 consumer 中的生产非法分支，并把三域 render-world owner 接入 Bevy `Render` schedule 的共享帧 contribution/aggregation seam：不再以 `unreachable!` 代表“已校验”，非法 Surface/Particle 输入现在返回显式错误；域贡献使用严格帧号、一次性 drain、冲突拒绝与公开 status。随后补齐 physical GPU resource admission/import seam，对真实 `SceneRhi` lease 执行 semantic-role、identity、容量/range/alignment/usage、texture layout/view、exact import coverage 与 alias 校验，形成运行时 import 和 record-ready view。`fabric_runtime::prepare` 也收紧为 crate 内入口。此前被无关工作树错误或磁盘空间阻断的尝试只保留为历史；当前独立 target 定向复验已完成：默认 `fabric_domains` **22/22**，owner/producer **3/3**，启用 `legacy-bevy` 后包含 GPU admission 的完整筛选 **33/33**。

**2026-10-21 当前工作树补充复验**：`prism_render_architecture --lib render_fabric` 已实际执行 **175/175**；`AccessKind` 现为 11 种并由集中映射降低到 buffer/texture use，RenderGraph 录制失败路径具备本帧 texture/view 的 RAII 回收。`prism_render_graph --lib` 本次复跑在依赖归档阶段因磁盘 `No space left on device` 停止，未进入测试；其 **85/85** 仅保留为先前运行记录；2026-10-22 后续全 package 已通过 unit 122/122、doc 1 passed / 1 ignored（见 §5.2）。上述基础契约及三域 33/33 证据不改变统一 SceneFabric recorder、draw/dispatch、full-frame 和跨适配器仍未闭合的结论。

**2026-10-22 三域/Volume 生产接线边界修正**：Hair 与 Decal 的 catalog 仍使用 **256B persistent STORAGE buffer** 泛化模板，每域仅声明 1 个 compute pass 和 3 个 `StorageWrite`，尚不匹配 strand arrays 或 decal receiver/material ABI；仍需真实资源/pass/access、生产数据、pipeline/bind group、draw/dispatch 与下游消费者。Volume 的该 catalog 同样是泛化占位契约，不能代表现有生产实现：`pkg/prism_render_scene/src/shading/volumetrics` 已有按 view 缓存的真实 3D froxel textures、scatter/integrate/apply pipelines、bind groups、scene depth/color 输入及 copy-back。

**2026-10-23 Volume SceneFabric 接线与生命周期进展**：Volume 的四个真实 pass（scatter、integrate、apply、scene-color copy-back）及动态真实纹理模板已接入 frozen SceneFabric 和 typed recorder，SSR depth/HDR scene-color usage 已对齐，并新增 catalog pass/layout contract 测试。关闭设置或 viewport 缺失/为零时会撤销 per-view volume 与 bind-group 组件。`cargo check -p prism_render_scene --lib`、architecture builtin contract 8/8 已通过；正确启用 `legacy-bevy` 的 scene volumetrics shader 定向测试 **2/2** 通过。真实 GPU runtime/readback 与设备 96-byte `IMMEDIATES` 能力已由 strict dispatch harness 1/1、`max_immediate_size=256` 证实；resize/disable/missing-input 的 per-view/cache 行为又由独立 strict lifecycle harness 1/1 覆盖。仍缺完整渲染 schedule 下的 completion-gated 资源退休与全帧闭环；这不代表 S6 或完整系统完成。

**2026-10-23 Volumetrics 录制所有权修正**：复核发现 typed recorder 调用曾丢弃 `record_into` 返回的 `RecordedGraph`，与其“持有 graph allocation 到 GPU 完成”的契约冲突。当前改为显式接收返回值，并断言此生产路径没有未托管的 transient graph allocations（纹理为外部 view-owned persistent resources）；这修正了生产录制的 allocation-owner 检查，但尚未证明宿主队列完成回调、真实 GPU 执行或像素输出。相关源码变更尚未编译/测试复验，不将静态改动记作 R/G 证据。

**2026-10-23 Volumetrics device capability gate**：pipeline 初始化核对实际 Bevy device 已启用 `IMMEDIATES` 且 `max_immediate_size >= 96`，不足时不创建 Volumetrics pipelines；view resource/bind-group prepare 在 pipeline 缺席时清理 view state 与缓存，避免不支持设备仍分配 froxel 资源或进入 recorder。低占用 `legacy-bevy` scene library check 通过；正确 feature 下 Volumetrics 定向测试 **14/14** 通过，覆盖 ABI/workgroup、96/95-byte 与 feature 能力边界、WESL 编译和参数；architecture catalog pass/layout contract **1/1** 通过。此前 `prism_settings` `ValueShape` 编译错误来自当时构建状态，当前源码已无该符号且本轮 check 通过，不再视为现存 blocker。后续真实设备 limits 与 dispatch/readback 结果见下方 strict GPU 复验；系统生命周期和性能仍未验证。

**2026-10-23 strict Volumetrics harness 历史过程（结果已由后续复验更新）**：新增 `shading/volumetrics/gpu_tests.rs`，严格检查物理 adapter/96-byte immediates；沙盒无 adapter 按预期拒绝。首次 Apple M2/Metal fixture 暴露 visibility `Rg16Float STORAGE_BINDING` 所需的 `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`，已补入严格设备请求。一次后续尝试因磁盘空间在 test binary 启动前停止；GPU 结果现由上文 strict 复验 **1/1** 关闭，不再表述为“尚无 GPU 结果”。

本节是对当前源码和已有记录的状态审计，不是全系统完成声明。除明确更新为 `render_fabric` 200/200、`fabric_domains` 33/33 及 §5.2 的 2026-10-22 frame/runtime multipass/TAA 定向结果、resident composition integration 3/3、material 与 RenderGraph package 结果外，其余 R/测试数字保留前序对应版本的运行口径，不表示 native 或统一全场景接线已复验。本次独立文档同步只核对已有日志、源码及运行回报，未运行 Cargo/GPU；200/200 的实际运行与 terminal exit 0 来自 main 实际复验，其余新增实跑证据按 §5.2 的 main 确认记录。标签含义：`S`=当前源码中存在的实现或接线，`D`=测试/门禁已定义，`R`=本轮运行通过，`H`=已有历史或先前运行记录，`G`=实际 GPU 证据，`E`=满足全场景生产闭环要求的证据。

本矩阵中的 Volumetrics GPU 项现已更新为 Apple M2/Metal strict 1/1（见 2026-10-23 实测补记）；旧段落及旧矩阵中“真实 GPU dispatch/readback 尚缺”的描述均为此前审计截点，不再作为当前缺口。当前 Volume 未完成项仅包括 resize/disable/missing-input 系统生命周期和融入完整多域帧闭环。

| 范围 | S 源码完成 | D 测试定义 | R 本次运行通过 | H 已有记录 | G GPU 证据 | E 全场景闭环 |
|---|---|---|---|---|---|---|
| Surface ABI、响应计划、Shader Package、Frame Graph | 当前 schema/codegen、`ShadingPlan`、冻结 package/permutation/PSO 映射和 31-pass graph 均有源码 | 对应 ABI、response、cache、graph 和 water gate 已定义 | material unit 888、integration 79（含 freshness 19 / ShadingPlan 33 / S5 cache 4）；RenderGraph unit 122、doc 1 passed / 1 ignored；scene lib 历史 1909/1909 | 2026-10-16/20 的其他分包记录仍为历史；未声称 architecture package suites 本轮完整复跑 | Apple M2/Metal resident composition strict 3/3（2 硬件 + 1 capability 拒绝）；水体及局部 shader/parity 记录各限原范围 | 否；没有完整 Scene/Fabric/TAA 帧证据 |
| Render Fabric contract / selection / payload / history | contract、静态 selection、`FrameRenderItem`、Surface/Particle CPU producer、water/spray/composition/TAA 窄接线存在 | `render_fabric` 单测、32-gate ledger、history 测试和 acceptance fixture 已定义 | 当前工作树 architecture `render_fabric` **200/200**（2026-10-22 main 实际复验）；scene lib 最近完整记录 **1909/1909** | `shader_package` 95/95、acceptance 3/3 已于 2026-10-22 main 复验，非完整 32 gate | RHI bridge 及水体局部 GPU 记录不能证明 Fabric 全帧 | 否；没有统一 recorder 和全域 imports |
| Particle / Volume / Hair / Decal 扇出 | `RenderApp` 已把 `ExtractedParticleSnapshot` 降为校验后的 `ParticleFabricFrame`，并以 canonical semantic `ResourcePayload` 与严格 consumer 形成 geometry；Particle 另有 renderer-owned `BufferId` 物理资源桥和本地 graph recorder 校验；Bevy Hair/Volume/Decal 已有稳定 owner、共享生产帧和 physical GPU admission/import。Volume 的真实三维 froxel 与四步 fog pass 已接入 SceneFabric typed recorder | Particle runtime snapshot、物理 owner/容量/recorder 连续性、资源 payload/geometry 清空语义有测试；三域覆盖同帧合并、一次性消费、producer 移除、owner identity/snapshot、非法输入、buffer/texture identity/layout/usage/import coverage/alias 拒绝；Volume 有 catalog 契约及 strict dispatch/lifecycle GPU harness | `prism_particle_render` 6/6、`prism_particle_app` 4/4；`prism_ecs` step inspector 8/8；三域默认测试 **22/22**，owner/producer **3/3**，`legacy-bevy fabric_domains` **33/33**；Volume catalog contract 1/1；Volumetrics capability/ABI/shader 14/14；production SceneFabric dispatch/readback 1/1、lifecycle 1/1（Apple M2/Metal） | Volume GPU 子集已验证生产 fog dispatch/readback、copy-back 像素一致性与 view/cache 生命周期；不证明完整渲染 schedule 的在途资源 retirement。域 GPU admission/import 本身不产生 Hair/Decal draw/dispatch | 不等于四域 Scene/Fabric draw/dispatch 或全场景 GPU 帧 | 否；Particle 尚未进入统一 SceneFabric import/recorder；Volume 缺完整 schedule 下的 completion-gated retirement；Hair/Decal 缺真实 pipeline、consumer 与 draw/dispatch。S6 未完成 |
| Particle / Volume / Hair / Decal 扇出 | `RenderApp` 已把 `ExtractedParticleSnapshot` 降为校验后的 `ParticleFabricFrame`，并以 canonical semantic `ResourcePayload` 与严格 consumer 形成 geometry；Particle 另有 renderer-owned `BufferId` 物理资源桥和本地 graph recorder 校验；Bevy Hair/Volume/Decal 已有稳定 owner、共享生产帧和 physical GPU admission/import。Volume 的真实三维 froxel 与四步 fog pass 已接入 SceneFabric typed recorder | Particle runtime snapshot、物理 owner/容量/recorder 连续性、资源 payload/geometry 清空语义有测试；三域覆盖同帧合并、一次性消费、producer 移除、owner identity/snapshot、非法输入、buffer/texture identity/layout/usage/import coverage/alias 拒绝；Volume 有 catalog 契约及 strict dispatch/lifecycle GPU harness | `prism_particle_render` 6/6、`prism_particle_app` 4/4；`prism_ecs` step inspector 8/8；三域默认测试 **22/22**，owner/producer **3/3**，`legacy-bevy fabric_domains` **33/33**；Volume catalog contract 1/1；Volumetrics capability/ABI/shader 14/14；production dispatch/readback strict GPU 1/1、lifecycle strict GPU 1/1（Apple M2/Metal，分开实跑） | Volume GPU 子集验证生产 dispatch/readback、copy-back 像素一致及 view/cache 生命周期；不证明完整渲染 schedule 的 in-flight resource retirement。域 GPU admission/import 本身不产生 Hair/Decal draw/dispatch | 不等于四域 Scene/Fabric draw/dispatch 或全场景 GPU 帧 | 否；Particle 尚未进入统一 SceneFabric import/recorder；Volume 缺完整 schedule 下 completion-gated retirement；Hair/Decal 缺真实 pipeline、consumer 与 draw/dispatch。S6 未完成 |
| native 共享帧与 host admission | Surface/Particle 聚合、post-GPU Prepare finalize、visibility 与 record/submit 前 admission 源码已接线 | 真实聚合、冻结 simulation clock、非法输入清空/恢复及调度测试已定义 | 本次未运行 | 历史失败尝试见 05 §15.2，不推定最新结果 | 无新增 GPU 证据 | 否；仍非 SceneFabric 全域 import/recorder |
| 对象级 runtime 局部切片 | exact object/generation/history/role 地址、payload upload 与子分配 preflight 存在 | Surface 对象隔离 fixture 已定义 | 2026-10-22 main strict multipass 9/9（8 hardware + 1 CPU fixture），exit 0 | 历史 8/8（7 hardware + 1 CPU fixture） | Apple M2/Metal；对象子分配真实 dispatch/readback | 否；该 multipass 子集不包含 `from_domain`；另有桥接局部通过记录，仍不含三域语义消费者或全帧 |
| TAA history | 真实双槽 persistent texture、exact binding、completion publication/retirement 已接线 | TAA ABI/shader/history 与 GPU lifecycle gate 已定义 | 2026-10-22 main strict GPU 19/19（含 lifecycle 7 项） | 真实 bind-group、两帧 publication/import、queue completion 与 retirement 已验证 | Apple M2/Metal REAL HARDWARE；局部 GPU 缺口已关闭 | 否；统一 runtime/recorder 与 full-frame parity 仍未闭合 |
| 跨适配器与性能 | 严格 adapter probe 和 identity/lifetime 代码存在 | `cross_adapter_acceptance` 的 skip/strict-fail 语义已定义 | 无 | 只有单 adapter 记录；少于两个 adapter 时 `SKIP_UNVERIFIED` | Apple M2/Metal 单适配器；第二 physical adapter 未验证 | 否；无跨适配器 full-frame replay，也无全系统 P50/P95/P99 性能证据 |

本轮使用独立 Cargo target 完成 `cargo test -p prism_render_scene --lib -- --test-threads=1`（**1909 passed，0 failed**）；其中 `material::systems` 16 项及 shader/layout/history 等库内测试均通过。另 `cargo test -p prism_render_material --lib handle::tests:: -- --test-threads=1` 为 **6 passed，0 failed**。这些是 crate library 单测，不覆盖 integration targets 或强制真实 GPU 运行。注意：1909 项结果对应最后一次 TAA retirement 测试微调及本次 Hair/Volume/Decal 接线之前的源码；2026-10-21 以共享 target 定向复测 TAA 最终测试为 **1 passed，0 failed**，但尚未在这些微调后的完整源码上重跑 1909 项。同日 `prism_particle_render`、`prism_particle_app` 定向单测分别为 **6/6**、**4/4**，而此前阻断它们的 `pkg/prism_ecs/src/diagnostics/step_inspector.rs` `Archetypes::created_count()` 已由 `prism_ecs` step-inspector **8/8** 复验修复。此前 `fabric_domains` 被无关工作树错误或空间压力阻断的尝试，现已由独立 target 的 `legacy-bevy fabric_domains` **33/33** 成功复验取代；其中默认域测试 22 项、owner/producer 3 项、GPU admission 8 项。TAA 真实 bind-group、queue completion 与 resize/removal/device-table replacement 的局部 GPU 验证已由 2026-10-22 main strict 19/19 关闭（见 §5.2），不改变上述历史计数。仍未验证的证据为：完整 Scene/Fabric/TAA full-frame parity；统一 `SceneFabric` 的全域 `compile → prepare_runtime → import → record/submit`；Particle 的 SceneFabric import/bind-group 与统一 recorder 消费；Hair/Decal 的真实 production pipeline、bind group、draw/dispatch recorder 与 visibility/geometry consumer；Volume 的 resize/disable/missing-input 系统级 lifecycle；完整 32 gate 的当前端到端运行（gate 23 仍为 `partial`）；删除旧水体入口前护栏已经运行的历史时序；第二 physical adapter 上的 probe、water/Scene/TAA replay；以及 full-scene 的图像 parity、资源峰值、吞吐和 P50/P95/P99 性能数据。故本系列当前结论是“局部源码与局部证据已形成，完整系统收口未证明”，不能写成全系统完成。

**2026-10-22 后续性能与 mesh 子范围复验**：提交 `8a06d28d1` 为 `ReactionGraph` 维护 dirty indegree，避免稀疏 flush 为高入度节点扫描干净依赖边；提交 `8ef5b00c4` 加入规模恢复测试，源码定向 `reaction::tests` 实跑 **12/12**，其中 4096 入度用例确认只遍历 dirty 子图两条 outgoing edge。提交 `166e0061f`、`e31500bf0` 补充 mesh vertex/index 可复用 allocation 的精确上传长度断言：6 个顶点的 72B allocation 缩至 3 个顶点时 draw range 为 36B；6 个 16-bit index 的 12B allocation 缩至 3 个 index 时 binding range 为 6B，底层 allocation ID/capacity 保持复用。当前工作树源码实跑 mesh 子集 **11/11**、mesh graph recorder **1/1**、resident transform **12/12**（含真实适配器像素变化、静帧零上传）、vertex range 回归 **1/1**、index range 回归 **1/1**。上述 range 生产实现仍在未提交的 `prism_render_app` 改动中；该改动链路经 mesh prepare、direct/bin queue、transform 附加 48B stream 至 recorder，属于局部 draw-binding 证据，不等于全域 Fabric recorder 或 full-frame parity。`prism_render_app --lib` 全库测试本次因同一共享 target 有其他 Cargo 构建且磁盘空间仅 116 MiB，未进入测试；本次没有清理活动 target。

**2026-10-23 当前源码 mesh recorder 防护复验**：`mesh_draw::ranges_are_valid` 在录制 pipeline/buffer 命令前校验顶点 buffer 与精确字节长度一一对应且非零；indexed draw 还校验 index length 存在、非零、按 index format 对齐，并保证 direct draw 的 `first + count` 字节范围不越过本次上传长度；非索引 direct draw也拒绝 element-range 整数溢出。Bin draw 仅接受带 index 的 indexed-indirect item，并校验参数记录数非零、offset/stride 的 4 字节对齐、stride 不小于 indexed-indirect ABI、参数范围运算不溢出。Direct/bin 非法输入均 fail-closed。当前源码运行 `CARGO_INCREMENTAL=0 cargo test --offline -p prism_render_app --lib mesh_draw::tests --jobs 1 --config profile.test.debug=0 -- --test-threads=1` **3 passed / 0 failed**，覆盖 malformed/零长度范围不录制命令与 indirect ABI/range 边界。只验证 CPU 侧 draw-range contract，不替代 GPU draw/full-frame parity。同期源码审计确认 `queue_stage::fill_queues` 的 `runtime.wait_idle()` 位于每帧 canonical material/frame arena prepare 前；这些 arena 会被同一帧后的上传和 draw 引用，且 resize/layout rebuild 会销毁 buffer/bind group，因此不能直接移除此 queue-wide barrier。以 completion-gated per-frame arena/bind-group/resource retirement 替换全队列 idle 是性能改进方向，仍须先补齐 owner/fence 生命周期与多帧 in-flight GPU 回归，再做吞吐对比；本轮未改动 barrier。

**2026-10-23 mesh GPU recorder 当前工作树复验**：同一源码版本 `CARGO_INCREMENTAL=0 cargo test --offline -p prism_render_app --lib mesh::tests --jobs 1 --config profile.test.debug=0 -- --test-threads=1` **12 passed / 0 failed**。包含真实 GPU 的 prepared mesh draw、半精度属性偏移和颜色 draw，以及复用 index allocation 时只绑定当前上传长度；另覆盖无 index mesh、重准备释放/复用与无效 mesh 拒绝。结合上条 recorder 的 3/3，这关闭当前 mesh 子集与直接/间接 draw range CPU 守门的定向复验缺口，但不等价于 mesh 全 package、bin GPU、统一 Scene/Fabric 或完整帧 parity 验收。
