# `prism_render_scene` 去 `bevy_render` 移植方案（只换地基）

> 范围：`pkg` 下唯一真正依赖 `bevy_render` 的 crate 是 `prism_render_scene`（Cargo 依赖 + 496 处引用 / 310 文件 / 7 个 `plugin.rs`）。
> 目标：把它的 **设备/命令/资源/帧编排** 从 `bevy_render` 换到 Prism RHI（`prism_render_driver` + `prism_render_graph` + `prism_render_app`），**不改其材质/着色/渲染逻辑**。
> 其余 pkg crate（`prism_render_visibility`、`prism_render_app`）只在注释里提到 `bevy_render`，无真实依赖，无需改动。

## 1. 为什么必须整体协调，不能零敲碎打

`prism_render_scene` 是**一个编译单元**：把某个文件的 `RenderDevice` 换成 RHI trait，会立刻打断所有调用点的类型一致性。因此移植按「切片」推进，每片内部自洽、可独立编译与回归；片内不再用 bevy_render。

## 2. 符号映射（bevy_render → Prism）

| bevy_render | Prism 目标 |
| --- | --- |
| `renderer::{RenderDevice, RenderQueue}` | `prism_render_driver::{RenderDevice, RenderQueue}`（具体后端 `WgpuDevice/WgpuQueue`） |
| `render_resource::{Buffer, Texture, TextureView, Sampler, BindGroup, BindGroupLayout, ShaderModule}` | `prism_render_driver` 同名类型；`Texture/Buffer` 由「真句柄」改为「generation id + 后端槽表」 |
| `render_resource::{TextureFormat, ShaderStages, BufferUsages, ...}` | `prism_render_driver::{format, flags}` |
| `render_resource::{RenderPipelineDescriptor, ComputePipelineDescriptor, ...}` | `prism_render_driver::pipeline` |
| `render_resource::{PipelineCache, SpecializedMeshPipelines, BindGroupLayoutEntryBuilder}` | 渲染器自建的管线/布局特化层（用 RHI 描述符） |
| `texture::TextureCache` | 渲染器自建的 GPU 资源缓存（键为 RHI id） |
| `view::{ExtractedView, RetainedViewEntity}`、`renderer::ViewQuery` | 渲染器自建的 view 记录 + `prism_ecs` |
| `sync_world::{MainEntity, MainEntityHashSet}` | `prism_render_app` 的 extract 映射（主世界 entity → 渲染记录） |
| `Extract` / `ExtractSchedule` | `prism_render_app` 的 `ExtractFn` |
| `Render` / `RenderApp` / `RenderStartup` / `RenderSystems` | `prism_render_app::{RenderApp sub-app, Render schedule}` |
| `RenderGraph` / `RenderGraphSystems` / `init_gpu_resource` | `prism_render_graph` + 渲染器自建的系统集 |
| `maths::affine` | `prism_math` |
| `utils::decompress_vertex_position` | 渲染器内部 helper（保留在 crate 内） |

## 3. 切片顺序（依赖有序）

- **S1 渲染世界地基**：`plugin.rs`（7 个）——`RenderApp/ExtractSchedule/Render/RenderSystems/RenderStartup` → `prism_render_app` 的 render sub-app + `Render` schedule；`renderer::{RenderDevice, RenderQueue}` → RHI trait。产出：crate 骨架跑在 Prism 上。
- **S2 资源类型**：`render_resource::{Buffer, Texture, TextureView, Sampler, BindGroup, BindGroupLayout, ShaderModule}` → RHI 类型/句柄；`TextureCache`/`PipelineCache` 自建。
- **S3 视图与提取**：`ExtractedView/RetainedViewEntity/ViewQuery/sync_world::MainEntity` → 自建 view 记录 + extract 映射。
- **S4 几何/缓冲**：`buffers/*`、`geometry/*`。
- **S5 光照/阴影**：`lighting/*`、`shading/shadow*`、`shading/ibl`、`shading/ao`。
- **S6 着色/后处理**：`shading/*`（gi/ssr/transparent/taa/volumetrics/…）、`opaque/*`。
- **S7 其余子系统**：`water/*`、`cloth/*`、`raytrace/*`、`visibility/*`、`material/*`。
- **S8 收尾**：删除 `bevy_render` 依赖行，`cargo tree` 断言无 `bevy_render`；同批评估 `bevy_pbr/core_pipeline/shader/mesh/image/camera/light/material/color` 的后续替换。

## 4. 移植前需先补齐的 RHI 能力（证据驱动）

- **提交完成回调**：渲染器的 `GpuCompletionTracker` 依赖 `RenderQueue::on_submitted_work_done`。已补：`prism_render_driver::FrameSync::on_submitted_work_done`（wgpu/null 已实现，真机验证）。
- 后续按 S2–S7 实际需要再补（例如纹理缓存/管线特化若需要额外原语），一律 **先补 RHI、再改渲染器**。

## 5. 门禁

- 每片：`cargo build/test/clippy -p prism_render_scene --all-targets` 全绿、零告警；该片 `rg bevy_render <子系统>` 归零。
- GPU 相关片：真机跑（`PRISM_REQUIRE_GPU=1`，跨沙盒）；沙盒默认看不到 adapter。
- 一次 CR + 修复 + 本地 commit。
