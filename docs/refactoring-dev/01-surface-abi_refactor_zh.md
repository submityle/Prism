# 01 · Surface ABI 与数据布局

> **2026-11-09 目标边界（当前权威）**：Surface ABI 的目标消费者是 Prism 原生 prism_render_app、Prism RHI 与 prism_render_graph。Bevy、Bevy RenderApp、prism_render_scene 的 Bevy 集成路径、SceneRhi 与 legacy-bevy 只作为历史/排除项保留；它们不作为 ABI 实现、资源提供者或验收前提。当前未完成项必须按原生提取、绑定、RenderGraph 编译/录制/提交和 GPU 证据重新列出。

> 本文记录当前 Surface ABI 的真源、字段边界、身份语义和可复现门禁。响应程序与组合规则见 [`02-response-composition_refactor_zh.md`](02-response-composition_refactor_zh.md)。

## 1. ABI 真源与职责

Surface ABI 的真源是 `pkg/prism_material_schema/schema/surface.toml`。`prism_material_schema` 的 codegen 从 schema 派生：

- Rust surface layout 与字段目录；
- WESL `material_unpack` 与 mirror struct；
- lobe、channel、response program、cache auxiliary 和 identity extension 表；
- 槽位、字足迹、默认值、闭包关系和 ABI 元数据。

运行时只消费生成结果。`SurfaceData` 是 OpenPBR 核心参数和活动扩展瓣的 packed 表示；`GpuMaterialHeader` 负责运行时记录、渲染状态、参数/纹理 heap 地址、PCM digest、动态通道和 dense execution binding。

## 2. 当前字段分层

### 2.1 SurfaceData

Surface layout 分为：

1. OpenPBR 核心字段；
2. KHR 区域瓣；
3. PRISM 扩展瓣；
4. schema 声明的 channel 与 cache auxiliary。

每个字段的名称、类型、默认值、所在 lobe、slot 和 word footprint 都来自 schema/codegen。packed block 的长度由核心 word 数与实际活动瓣足迹相加得到，`lobe_mask` 用于 GPU/CPU 对称解包。

### 2.2 GpuMaterialHeader

当前 header 明确包含：

- generation / revision；
- `render_class` 与 `render_domain`；
- feature flags 与 `closure_mask`；
- packed surface block 的 `parameter_offset` / `parameter_size`；
- texture / sampler heap 范围；
- active 状态与 material epoch；
- `lobe_mask`；
- PCM `used` / `required` digest；
- `dynamic_wetness`、`dynamic_snow`、`dynamic_damage`、`dynamic_dissolve`；
- 五个 dense execution binding word。

旧的 illumination、custom-program、specialization-low/high 语义字段不在当前 Rust header，也不在 WESL mirror 中。响应身份由 `ShadingPlan` 和执行绑定表表达；自定义程序 id 由 `ProgramRef::Registered(ProgramId)` 进入 plan / program 表。

## 3. 身份与索引

当前有三种不同身份，不互相替代：

| 身份 | 作用 | 权威来源 |
|---|---|---|
| `closure_mask` | 记录作者图授权的 closure 集合 | IR / authoring |
| `lobe_mask` / `active_lobes` | packed surface 实际携带的活动瓣 | lowering + schema lobe table |
| execution binding | 当前 plan、program、package、permutation 的可执行身份 | execution tables + verified package |

`active_lobes` 是 Surface packed decode 和执行身份的输入；`closure_mask` 不是 pipeline key。这样可以保留授权记录，同时避免原始 closure 位与派生 surface 形状不一致。

`MaterialRenderClass` 不属于 Surface lobe identity。当前七个值只表达 blend/raster family；响应 program、execution family、render phase 和 package identity 由各自的表表达。

## 4. ClosureKind 的分层边界

`pkg/prism_render_material/src/ir.rs` 中的 `ClosureKind` 仍保留作者图语义，包括 `Npr`、`Custom`、`Mix`、`Layer` 等变体。它们不直接充当 runtime response program：

```text
Authoring IR / ClosureKind
        ↓ lowering
SurfaceData + ShadingPlan
        ↓ resolve
GPU header + dense execution binding
```

因此 closure 图可以表达作者语义，Surface ABI 可以表达实际 packed 数据，response registry 可以表达可执行程序；三者各自只有一个 owner。

## 5. OpenPBR 导入与区域隔离

- `adapt_openpbr_surface` 是 MaterialX/OpenPBR authoring 的唯一导入入口；它直接返回完整的 `RuntimeSurface`，不再保留少字段兼容视图。
- Standard Surface 别名只在语义一致时映射；未知字段直接返回错误。
- `project_to_openpbr` 是有损向下投影。
- 纯 OpenPBR 子集通过 `pure_openpbr_projects_to_a_byte_identical_fixed_point` 保持字节不动点。
- KHR 区和 PRISM 区的字段、slot、默认值与投影规则由 schema namespace 隔离。

这保证扩展瓣可以增加 Surface 能力，而不会把扩展字段偷偷改写成 OpenPBR 核心字段。

## 6. 当前门禁

以下是当前仓库中可直接复现的 ABI 检查：

| 门禁 | 证明内容 |
|---|---|
| `committed_wesl_matches_generator` | 提交的 WESL 生成物与 codegen 输出一致 |
| `committed_rust_layout_matches_generator` | 提交的 Rust layout 与 codegen 输出一致 |
| `generated_field_directory_covers_exactly_non_padding_schema_fields` | schema 非 padding 字段都有且只有一条生成目录记录 |
| `generated_unpack_covers_every_schema_default_once` | 默认值与 unpack 覆盖关系完整且无重复 |
| `handwritten_surface_has_no_schema_derived_layout_definitions` | 消费端没有另写派生布局定义 |
| `rust_and_wesl_surface_mirrors_match_in_order_and_type` | Rust/WESL mirror 字段顺序与类型一致 |
| `surface_mirrors_cover_exactly_the_schema_fields_and_bytes` | mirror 覆盖字段与字节数一致 |
| `core_and_lobe_word_counts_match_surface_rs` / `typed_surface_struct_offsets_match_schema_pack_slots` | word count、offset、slot 一致 |
| `pure_openpbr_projects_to_a_byte_identical_fixed_point` | 纯 OpenPBR 子集保持字节级 fixed point |
| `every_surface_lobe_has_a_stable_activation_source` | 每个 lobe 都有稳定的活动来源 |
| `header_and_packed_parameters_share_the_derived_lobe_mask` | header 与 packed block 使用同一派生活动瓣集合 |
| `abi7_execution_binding_is_five_dense_words_with_full_permutation_width` | 五个 dense binding word 覆盖完整 permutation 宽度 |

## 7. 当前源码与测试证据

本节涉及的 prism_render_scene 运行记录属于历史/排除路径；Surface ABI 当前验收只接受 Prism 原生 material、RHI 与 RenderGraph 证据。

直接源码位置：

- schema：`pkg/prism_material_schema/schema/surface.toml`；
- codegen：`pkg/prism_material_schema/src/`；
- header / class / domain：`pkg/prism_render_material/src/record.rs`；
- packed surface 与 lobe mask：`pkg/prism_render_material/src/surface.rs`；
- execution binding：`pkg/prism_render_material/src/execution.rs`；
- authoring IR 与 closure lowering：`pkg/prism_render_material/src/ir.rs`。

历史运行记录（非本轮复跑）：`cargo test -p prism_render_material --tests` 曾有单元测试 `878 passed; 0 failed`；`codegen_freshness` 11 项、`material_header_abi` 3 项、`pcm_parity` 7 项、`schema_parity` 9 项、`shading_plan` 33 项、`surface_abi_mirror` 2 项均通过。此前另有 `prism_render_scene --lib` 1909/1909 与 `prism_render_material::handle::tests` 6/6 记录；codegen/integration 后续已于 2026-10-22 随 material 全 package 复验，见 §9。

### 7.1 2026-11-09 原生 Prism 当前未完成项

Surface schema/codegen 与 Rust/WESL mirror 的局部门禁已有记录，但下列原生验收仍未完成：

1. 在 prism_render_app 的真实 extraction、material upload、bind-group 建立和 prism_render_graph record/submit 中消费同一 Surface ABI，并证明字段、lobe mask、execution binding、layout 与资源版本没有在中间层复制或漂移。
2. 覆盖原生 response/composition、water、TAA/history 与多视图的 ABI 消费，完成错误输入清空、resize、device replacement、completion-gated retirement 和真实 readback/parity；历史 Bevy/SceneRhi 集成不计入。
3. 覆盖 Surface ABI 对 shader package、pipeline state、PSO cache、driver binary 和后端格式能力的影响，完成至少两个 physical adapter 的严格结果。
4. 记录与 ABI 相关的带宽、上传量、绑定数量、缓存命中率、显存峰值及 P50/P95/P99；单元测试通过不能替代性能证据。

以下历史运行记录仍保留，但只证明其原始范围；它们不把 Surface ABI 标记为 native 全场景完成。

## 8. Git 历史锚点

- `873d8c407 refactor(material): derive surface ABI layout metadata`：建立 schema 派生布局。
- `0de14a400 refactor surface ABI lobe and mirror gates`：加入生成物 freshness、Rust/WESL mirror 和 lobe ABI 门禁。
- `8400fe972 feat: complete runtime and render refactoring implementation`：将 header、执行绑定和材质运行路径接入整体实现。

当前文档不再把手写 `SpecializationId` 位布局或 illumination 轴当作 ABI 现状；它们只作为历史背景，不是当前数据来源。

## 9. 最终收口验收矩阵 / 未完成项（2026-10-21 文档审计）


**2026-10-22 最新复验与状态修正**：architecture 改动已提交 `652404b94`。main 对当前架构源码实际复验 `render_fabric` **200 passed / 0 failed / 0 ignored**、`shader_package` **95 passed / 0 failed / 0 ignored**、acceptance **3 passed / 0 failed / 0 ignored**；日志与范围见 05 §15.2。本文较早的 185 项及 shader-package/acceptance“未复跑”描述仅代表此前截点，以本补记为准。acceptance 的 3 项验证 ledger/fanout/partial 标记，不是完整 32 gate 端到端通过。

**局部 GPU 桥接证据修正**：既有日志已记录 `fabric_runtime` 定向筛选 **29 passed**（CPU/GPU 混合，不能全部算硬件用例），其中 authenticated `from_domain` bridge 在 Apple M2/Metal 完成真实上传、dispatch 和子范围隔离；RHI pipeline bridge 筛选 **15 passed**，包含 exact native render-pipeline identity/generation/lease 与原生绘制。这里只关闭相应局部桥接验证，不能关闭 Hair/Volume/Decal 的真实语义渲染、typed graphics 全链、统一 Scene/Fabric/TAA full-frame、第二 adapter 或性能验收。

**2026-10-21 收口补记**：此前 `prism_render_scene --features legacy-bevy --lib` 的 `cargo check` 曾通过，确认当时的 Surface/Fabric 依赖可编译；一次后续尝试曾因 `prism_asset` 的 `Flight` 初始化缺少 `cancellation_waiters` 字段而停止，域帧定向测试未进入执行。该错误仅是历史记录：当前 `Flight` 定义已无此字段，后续编译反馈也已越过 asset 初始化；本次文档同步未运行 Cargo，不认定当前仍受该错误阻断。这些记录都不是 Surface ABI package freshness、真实 recorder 或 GPU parity 证据，未完成验收全部保留。

**2026-10-22 package 复验**：`cargo test -p prism_render_material --jobs 1 --config profile.test.debug=0`（无 `--lib` 限制）退出码 0；unit **888/888**，integration **79/79**，doc 0 项。integration 分别为 freshness 19、header ABI 4、PCM parity 7、S5 ABI/cache invalidation 4、schema parity 10、ShadingPlan 33、Surface mirror 2；均为 0 failed / 0 ignored。freshness 包含重新生成 Rust layout、schema contract、WESL unpack 与 classification 后逐字节比较。日志 `/private/tmp/prism-render-material-package-20261022.log`。本 package/freshness 缺口已关闭，不代表真实 GPU 或全帧验收。

**2026-10-22 局部证据补记**：main 实际复验 `prism_render_architecture --lib render_fabric` **200/200、terminal exit 0**（含新增 9 项对象资源 CPU 契约；旧 **176/176** 为历史，不重复相加）。main 实际运行并确认 exit 0 的 strict multipass 为 **9/9 = 8 项 Apple M2/Metal 硬件 + 1 项 CPU fixture**，本次已核对 `/private/tmp/prism-fabric-object-runtime-gpu-20261022.log`；旧 **8/8 = 7 hardware + 1 CPU fixture** 为历史。新增对象 fixture 覆盖 exact object/generation/history/role 寻址、完整上传/错误 role 拒绝与子分配隔离；不代表 `from_domain` 桥接 GPU、三域语义渲染、整个 shader-package/acceptance 或完整帧验收。

**2026-10-22 Hair/Volume/Decal 生产接线边界修正**：Hair 与 Decal 的 catalog 仍使用 **256B persistent STORAGE buffer** 泛化模板，每域仅声明 1 个 compute pass 和 3 个 `StorageWrite`，尚不匹配 strand arrays 或 decal receiver/material ABI；仍需真实资源/pass/access、生产数据、pipeline/bind group、draw/dispatch 与下游消费者。Volume 的 catalog 同样是泛化模板，不能代表现有 volumetric fog：`pkg/prism_render_scene/src/shading/volumetrics` 已有按 view 缓存的真实 3D froxel textures、scatter/integrate/apply pipelines、bind groups、scene depth/color 输入及 copy-back。Volume 真正未完成的是把该生产路径纳入冻结 SceneFabric 的真实纹理布局、pass 依赖/hazard、typed recorder 和宿主 record/submit；scene-color copy-back 尚未作为 Fabric pass。不得用占位 buffer、probe 或伪造输出冒充集成；`from_domain` 的局部 GPU 上传/dispatch/subrange 证据不覆盖 volumetrics 语义或全帧闭环。

**2026-10-23 Volume 状态更正**：上述关于 Volume 尚未进入 SceneFabric 的描述已被后续实现取代。生产 scatter/integrate/apply 与 scene-color copy-back 已使用真实纹理模板和 typed recorder 接入 frozen SceneFabric，且设备启动检查 96-byte `IMMEDIATES` 能力；Apple M2/Metal 上 dispatch/readback 与 lifecycle 两个 strict GPU 用例分别 **1/1** 通过。生命周期测试覆盖 viewport resize、missing visibility、zero viewport 与 disable 的 view/cache 行为；仍缺完整 schedule 下的 completion-gated resource retirement 与完整帧闭环。Hair/Decal 与其余全场景缺口保持不变。

本节区分源码、测试定义和实跑证据。`S`=源码事实，`D`=测试定义，`R`=本轮运行通过，`H`=已有运行记录，`G`=GPU 证据，`E`=全场景闭环。scene lib 1909/1909、allocator 6/6 为此前独立运行记录；当前 material package 结果以上述复验为准。

| 条款 | S 源码完成 | D 测试定义 | R | H / G | E 全场景闭环 |
|---|---|---|---|---|---|
| schema/codegen/layout/header/mirror | schema 真源、生成 Rust/WESL、header 与 dense binding 已存在 | freshness、mirror、slot/word、lobe-mask 和 ABI gate 已定义 | material unit 888/888、integration 79/79，含 freshness 19/19 | 无专门跨适配器 ABI GPU 证据 | 否；未证明完整 Scene/Fabric 消费 |
| identity 分层与 OpenPBR boundary | `closure_mask`、`active_lobes`、execution binding 分层及导入边界存在 | fixed-point、activation、layout 和 stale-binding gate 已定义 | material 全 package 已通过，含 schema parity 10/10、ShadingPlan 33/33 | shading package 与 GPU 证据另计 | 否；未证明所有材质在完整帧中消费同一 ABI |
| ABI 变更后的 package/cache/driver 影响 | 由 03 的 typed identity、ABI fingerprint 和 stale rejection 接线承载 | cache invalidation / pipeline worker gate 已定义 | material S5 ABI/cache invalidation 4/4 | 该 CPU gate 不等于全域 driver binary 验证 | 否 |

仍未验证：跨适配器 ABI/shader 消费；Surface ABI 在统一 Scene/Fabric 每帧提取、绑定、barrier 和真实 recorder 中的端到端使用；以及完整帧图像 parity、资源/性能数据。material 全 package 通过仅关闭分包 CPU/生成物验收，不等于全场景验收完成。

**2026-11-09 状态同步**：Water 已移除独立 Core3d recorder，改由顶层 SceneFabric coordinator 统一提交；Water 定向测试 **230/230** 通过。该接线不构成 Surface ABI 在全场景帧中的完整消费证据；统一多域 payload、barrier、跨适配器和全帧 parity 仍未完成。

**2026-11-09 原生 TAA/response 接线补记**：原生 `prism_render_app` 已在 Surface graph 的提交成功、提交失败和 graph 构建失败路径分别记账或取消 TAA history，并在帧末按 live key 执行 completion-gated 回收；这关闭了宿主未调用 history publication 的具体断点。Registered MaskBlend 左右臂现分别传递各自 program identity。仍未完成的是同一原生帧中的全域 ABI 消费、完整 response/split 语义、跨适配器和全帧图像 parity。

**2026-11-09 白色旧路径清理**：审计确认白色 `SpineMeshPipeline` 只由未发布兼容构造和测试使用，默认插件生产入口一直是 `MeshPipelines::surface_encode`；已删除白色 shader、单色 color target、`MeshPipelines::spine` 及其 reserved-module 安装/恢复分支。自定义 provider 不再被替换成假输出；没有 Surface ABI 的 provider 在 recovery 中显式报错并保持暂停。`ModuleId::SPINE` 仍作为现有原生 Surface pass 的模块标识使用，不代表白色 shader，也未被删除。
