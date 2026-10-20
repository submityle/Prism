# 04 · Frame Graph 与 pass 组织

> 文档口径：以当前源码、测试和已合入历史为准的实现审计。本文描述统一编译内核、shading graph 拓扑、生成式后处理清单和调度边界。
> 上层导航：[`00-overview_refactor_zh.md`](00-overview_refactor_zh.md)
> 相关文档：[`02-response-composition_refactor_zh.md`](02-response-composition_refactor_zh.md)、[`03-shader-package_refactor_zh.md`](03-shader-package_refactor_zh.md)

## 1. 当前模块边界

### 1.1 统一 frame graph 编译器

`pkg/prism_render_architecture/src/frame_graph/mod.rs` 现在只是兼容导出：

```rust
pub use prism_render_graph::frame_graph::*;
```

因此 `prism_render_architecture::frame_graph` 不再维护独立的 builder/types/compiler/schedule 实现。实际声明与编译类型位于 `pkg/prism_render_graph/src/frame_graph/`，统一推导内核位于 `pkg/prism_render_graph/src/compile/` 和 `pkg/prism_render_graph/src/graph.rs`。

当前声明式类型的枚举数量为：

| 类型 | 变体数 | 当前变体 |
|---|---:|---|
| `PassKind` | 4 | `Raster`、`Compute`、`Transfer`、`Present` |
| `QueueClass` | 3 | `Graphics`、`Compute`、`Transfer` |
| `ResourceLifetime` | 3 | `Imported`、`Persistent`、`Transient` |
| `AccessKind` | 8 | sampled/storage read/write、color/depth attachment、indirect read、transfer read/write |
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
2. `prism_render_architecture::frame_graph` 作为兼容导出存在，源码事实以 `prism_render_graph` 为准。
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
| compatibility frame_graph 导出与底层契约 | `pkg/prism_render_architecture/src/frame_graph/mod.rs`、`pkg/prism_render_architecture/src/frame_graph/tests.rs` |
| shading graph 单一 pass-count source、主链顺序、资源 barrier | `pkg/prism_render_scene/src/shading/graph.rs` 中 `declared_pass_count_matches_runtime_assertion` |
| composition 固定位置、imported planes、HDR barrier | `pkg/prism_render_scene/text/unit/shading/composition_graph.rs` 中 `composition_is_resident_and_follows_water_response_producer` |
| composition production recorder 与 GPU 回归 | `shading/composition.rs::response_composition_pass` / `record_composition_fabric_commands`；`text/integration/resident_composition_gpu.rs` 现与生产入口共用 recorder，覆盖真实 imports、Fabric compile/prepare/record、producer → composition → readback 顺序；修订后的 harness 尚待本轮运行。即使局部 GPU 回归通过，也不证明共享 Scene/Fabric/TAA 全帧闭环 |
| pass skeleton codegen 的确定性、依赖校验、资源复用和 cycle diagnostics | `pkg/prism_render_graph/src/frame_graph/codegen.rs` 及其单元测试 |
| 23 项 post inventory、声明/binding/executor 对齐和插件 registration | `pkg/prism_render_scene/src/shading/post_graph.rs`、`pkg/prism_render_scene/src/shading/plugin.rs` |
| shading crate 历史运行记录 | `cargo test -p prism_render_scene --lib shading`：844 passed，0 failed；不是本轮复跑结果 |

跨适配器行为单独由 `pkg/prism_render_scene/tests/cross_adapter_acceptance.rs` 验收：它枚举 distinct physical adapters，并在每个 adapter 上执行 storage compute、submit、copy-to-staging 和 map/readback。默认 profile 在少于两个 adapter 时输出 `SKIP_UNVERIFIED`；`PRISM_REQUIRE_GPU=1` 将同一条件提升为失败。2026-10-20 沙盒外只发现 `Apple M2 / Metal / IntegratedGpu` 一个 adapter，故严格门禁失败。该测试证明 adapter-local submission/readback，不证明 31-pass frame graph、Scene imports、TAA exact history publication 或 water full-frame parity 已跨 adapter 通过；TAA history 本轮仅完成源码接线，尚待 Cargo/GPU 验证。

## 7. 历史依据

- `352d594b1 refactor(render): unify frame graph execution planning`：删除重复的 frame-graph compiler 推导，统一执行规划和 codegen 接口。
- `8400fe972 feat: complete runtime and render refactoring implementation`：合入 shading graph、composition、post inventory、visibility 和 runtime 验收链。
- `5ad60f878 test(render): name refactor acceptance gates`：固定 graph pass count、visibility 和分类验收的命名测试。

本篇结论：当前 frame graph 已由统一 `prism_render_graph` 编译器承载；shading graph 以 8 个主链 pass、23 个生成式后处理 pass 和 resident composition 组成固定拓扑，资源访问、执行 ledger 与测试共同构成验收契约。

## 8. 最终收口验收矩阵 / 未完成项（2026-10-20 文档审计）

`S`=源码实现，`D`=测试定义，`R`=本轮运行通过，`H`=已有运行记录，`G`=实际 GPU 证据，`E`=全场景闭环。本轮 `prism_render_scene --lib` 为 1909/1909；`prism_render_graph` package tests、真实 GPU 与 full-frame replay 未由此覆盖。

| 条款 | S | D | R | H / G | E |
|---|---|---|---|---|---|
| graph compiler、SSA version、hazard、barrier、alias、lifetime | 当前统一编译器与资源依赖实现存在 | compiler/codegen/graph gate 已定义 | scene lib 1909/1909；graph package suite 未运行 | shading `844/0` 等为历史记录 | 否；没有 full-frame 复核 |
| 31-pass shading topology（8 主链 + 23 post） | pass inventory、composition 固定位置和 runtime assertion 存在 | pass-count、order、resource/barrier gate 已定义 | scene lib 1909/1909 | 既有记录只证明相应历史工作树 | 否；未证明完整 Scene/Fabric/TAA 执行 |
| resident Composition 与 imported resources | 生产 recorder、imports 和 HDR barrier 接线存在 | integration/GPU harness 定义存在 | scene lib 1909/1909；integration/GPU harness 未运行 | 水体/组合单 adapter 记录不等于整图 GPU 证据 | 否 |
| TAA/history 与跨 adapter | TAA history 源码接线存在；跨 adapter strict/skip probe 存在 | history/adapter gate 已定义 | 无 | 单 Apple M2/Metal；第二 adapter 未验证 | 否；无 exact history full-frame parity |

仍未验证：`prism_render_graph` package/integration suites；修订后的 resident composition GPU harness；全 31 pass 在真实 Scene imports、barrier、record/submit 上的完整运行；TAA completion-gated history 的真实 GPU 行为；第二 physical adapter；以及全帧图像 parity、资源峰值和 P50/P95/P99 性能数据。库内 1909 项通过及编译器/测试定义的存在不等于生产闭环完成。
