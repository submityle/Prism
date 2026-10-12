# `prism_render_scene` 去 `bevy_render` 的实际依赖面（证据驱动）

> 本文只回答一个问题：**要换掉 `bevy_render`，到底要动多少东西、按什么顺序动。**
> 口径：`pkg/prism_render_scene` 是唯一真有 `bevy_render` 依赖的渲染 crate
> （`docs/prism_render_scene_rhi_port_zh.md` §1）；接线层 ①–⑥ 已完成（同文 §6），
> 本文是 ③「RHI 接入已有渲染系统」的施工图。
>
> 复现：`rg -c bevy_render pkg/prism_render_scene/src -g '*.rs' | awk -F: '{s+=$2} END{print s}'`

## 1. 读数

| 量 | 值 | 说明 |
| --- | --- | --- |
| `use bevy_render…;` 语句 | **318** | 导入点，不是引用点 |
| 含真实导入的文件 | **312 / 319** | `src` 下 `*.rs`，几乎整个 crate |
| 去重后的 bevy 条目 | **150** | 逐条映射见 §3 |
| 最重的三个模块 | `render_resource`(285)、`renderer`(255)、`view`(81) | 导入点数 |

按目录（含导入的文件数）：`shading/` 229、`water/` 17、`visibility/` 11、`lighting/` 10、
`geometry/` 8、`material/` 6、`cloth/` 6，其余 ≤5。

**结论**：这不是「换几行依赖」，而是把 `renderer`（录制面）与 `render_resource`（资源面）
整体迁到 `prism_render_driver` + `prism_render_graph` + `prism_render_app`。好在 Prism 的 RHI
当初就照 wgpu 的词汇建模，多数条目**同名对应、语义等价**（§3），所以这是搬运，不是重设计。

## 2. 依赖面排名（导入点，前 24）

| # | bevy 条目 | 导入点 | # | bevy 条目 | 导入点 |
| --- | --- | --- | --- | --- | --- |
| 1 | `renderer::RenderDevice` | 202 | 13 | `view::ExtractedView` | 48 |
| 2 | `render_resource::PipelineCache` | 119 | 14 | `render_resource::TextureDimension` | 48 |
| 3 | `render_resource::ShaderStages` | 82 | 15 | `render_resource::TextureDescriptor` | 46 |
| 4 | `render_resource::BindGroup` | 80 | 16 | `render_resource::TextureUsages` | 46 |
| 5 | `render_resource::BindGroupEntries` | 75 | 17 | `render_resource::TextureView` | 43 |
| 6 | `render_resource::BindGroupLayout` | 73 | 18 | `render_resource::Buffer` | 41 |
| 7 | `renderer::RenderContext` | 72 | 19 | `render_resource::BufferUsages` | 40 |
| 8 | `render_resource::ComputePassDescriptor` | 65 | 20 | `camera::ExtractedCamera` | 37 |
| 9 | `render_resource::TextureSampleType` | 64 | 21 | `texture::{CachedTexture, TextureCache}` | 30 / 30 |
| 10 | `render_resource::ComputePipelineDescriptor` | 62 | 22 | `renderer::RenderQueue` | 33 |
| 11 | `renderer::ViewQuery` | 61 | 23 | `view::Msaa` | 33 |
| 12 | `render_resource::CachedComputePipelineId` | 62 | 24 | `render_resource::TextureFormat` | 29 |

余下 126 个条目各 ≤26 导入点，含 `render_resource` 的
`Extent3d / Origin3d / SamplerDescriptor / AddressMode / FilterMode / TextureAspect /
TextureViewDescriptor / TexelCopyTextureInfo / MapMode / AtomicPod / RawBufferVec /
BufferInitDescriptor / SpecializedRenderPipelines / ShaderType`，
`mesh::{RenderMesh, RenderMeshBufferInfo, allocator::MeshAllocator}`、`render_asset::RenderAssets`、
`render_phase::{PhaseItem, DrawFunctions, ViewBinnedRenderPhases, BinnedRenderPhase, RenderCommand}`、
`sync_world::{MainEntity, MainEntityHashSet}`、`utils::decompress_vertex_position`、
`maths::affine3_to_square`、`allocator::*`。

## 3. 逐项映射（现状以本仓代码为准，非计划）

**已落且同名可用**（`prism_render_driver` 就是按这套词汇建的，见 `pkg/prism_render_driver/src/lib.rs:49-125`）：
`RenderDevice`、`RenderQueue`、`Buffer`/`BufferDescriptor`/`BufferUsages`/`BufferId`、
`Texture*`（`Descriptor`/`Usages`/`Dimension`/`Format`/`Aspect`/`View`/`ViewDescriptor`）、
`Extent3d`、`Sampler*`、`BindGroup*`、`ShaderStages`、`PipelineLayout*`、
`ColorTargetState`/`ColorWrites`/`PrimitiveState`/`MultisampleState`、`ComputePipelineDescriptor`、
`DrawIndexedIndirectArgs`、`AddressMode`/`FilterMode`、`MapMode`/`TexelCopyTextureInfo`/`Origin3d`。

**已落、但要换层**（同一能力，宿主不同）：

| bevy | Prism | 备注 |
| --- | --- | --- |
| `renderer::RenderContext` | `prism_render_graph::ExecuteContext`（`builder.rs:315`） | pass 内录制面已等价（pipeline、bind group、顶点/索引、`draw*`、`multi_draw_indirect`、`dispatch*`、copy）；**缺**「每 view 一份上下文 + `CurrentView` 绑定」这一层 |
| `render_resource::PipelineCache` | `prism_render_app::{PipelineCache, MeshPipelineSpecializer}` | 内容键去重已有；bevy 的 `PipelineCache<T>` 是按类型分表的泛型，Prism 目前只有 mesh 一族，其余族按同模式补 |
| `render_resource::{ComputePassDescriptor, CachedComputePipelineId, SpecializedRenderPipelines}` | `prism_render_graph` 的 compute pass + `ComputePipelineId` | 图侧有 `set_compute_pipeline`/`dispatch`；「缓存 + 特化」按 `PipelineCache` 同模式补 |
| `texture::{TextureCache, CachedTexture}` | `prism_render_app::{TextureCache, GpuTexture}` | 去重/驱逐已有，键含格式/尺寸/用途 |
| `view::{ExtractedView, RetainedViewEntity}` | `prism_render_architecture::view_family`（`ViewRegistry`/`ViewHandle`） | 族已落；每 view 的 uniform 记录待补 |
| `render_phase::*` | `prism_render_app::{PhaseItem, DrawFunctions, RenderQueues}` | 已有；`BinnedRenderPhase` 对应 bin 队列（已接完） |
| `batching::gpu_preprocessing` | `prism_render_visibility::binning` + `prism_render_app::bin_queue` | 已落：GPUD bin → `IndirectArgs` → `MultiDrawIndirect` |

**缺**（下一步要建的 Prism 能力，按证据排序）：

| 缺的东西 | bevy 处 | 导入点 | 建议落点 |
| --- | --- | --- | --- |
| 每 view 录制上下文 + `CurrentView` | `renderer::{RenderContext, ViewQuery}` | 72 + 61 | `prism_render_app`（`ViewQuery` 进行中） |
| extract 层的实体映射 | ~~`sync_world::MainEntity`~~ | 16+5 | **已落**：`prism_render_app::sync_world`（`MainEntity` + `MainEntityHashSet` + `tag_extracted_entities`，接在 `prism_ecs::extract::extract_components` 的 `(main, render)` 对上） |
| 资源表 `RenderAssets<T>` | ~~`render_asset::RenderAssets`~~ | 11 | **已落**：`prism_render_app::render_asset`（`RenderAsset` trait + `RenderAssets<A>` 表 + `sync_assets`/`drain_and_sync_assets`：Added/Modified 经 `prepare` 写入、Removed 走 `unload` 释放 GPU 对象、`RetryNextUpdate` 保留上一版并下帧重试） |
| mesh 资源与分配器 | `mesh::{RenderMesh, allocator::MeshAllocator}` | 9 + 8 | `prism_render_architecture` 的 geometry 侧 |
| camera 提取 / MSAA 目标状态 | `camera::ExtractedCamera`、`view::Msaa` | 37 / 33 | **场景侧**（渲染层不收相机，见 §6/2）；`Msaa` 属渲染目标状态 |
| `ShaderType`、`AtomicPod`、`RawBufferVec` | `render_resource::*` | 9 / 8 / 8 | 已有 `bytemuck` + `prism_render_architecture::abi`；用到再加，不预先造 |
| `maths::affine3_to_square`、`utils::decompress_vertex_position`、`allocator::*` | 同名模块 | 4 / 3 / 8 | `prism_math` / `prism_render_architecture`，纯函数，随首个使用点搬 |

## 4. 切片顺序（依赖有序，每片一个 commit + CR）

1. **S2a 每 view 录制上下文**：`CurrentView` + `ViewQuery` **已落**（`prism_ecs::system::view_query`
   + `prism_render_app` 再导出；缺 skip 机制，参数是 `Option<ViewQuery<..>>`，见该模块文档）；
   剩：把 `ExecuteContext` 包成 view 作用域的录制面（外层循环按 view 设 `CurrentView`）。
2. **S2b extract + 实体映射**：实体映射与资产管线**已落并接进帧**——`extract_assets::<A>`（走 `App::add_extract` 链，新增多段 extract 能力）把主世界 arena 的 Added/Modified 复制进 `ExtractedAssets<A>`、Removed 只传 id；`prepare_assets::<I, A>` 在新增的 `RenderSystems::PrepareAssets` 阶段跑 `prepare_extracted_assets`，`AssetSyncStats<A>` 记录每帧报告；`register_render_assets::<I, A>` 一次注册一族（= 上游 `RenderAssetPlugin<A>`）。剩：把 `RenderMesh`/`GpuImage` 这两族接上（需 S2c 的 mesh 层）。
3. **S2c mesh 资源层**：`RenderMesh`/`RenderMeshBufferInfo`/`MeshAllocator`。
4. **S2d 资源缝收敛**：逐条目把 `render_resource::*` 换成 RHI 同名类型（§3 第一组，机械但量最大）。
5. **S2e 小工具**：`affine3_to_square`、`decompress_vertex_position`、`allocator::*`。
6. **S2f 子系统翻面**：按目录把 `bevy_ecs`/`bevy_app` 系统搬到 `prism_ecs`/`prism_app`
   （`shading/` 229 文件是主战场，按子目录再拆，可多 agent 并行）。

门禁与 `docs/prism_render_scene_rhi_port_zh.md` §5 相同：每片 `build`/`test`/`clippy` 全绿零告警、
该片 `rg bevy_render <子系统>` 归零、GPU 相关走 `PRISM_REQUIRE_GPU=1` 真机、一次 CR + 修复 + 本地 commit。
