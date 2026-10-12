# `prism_render_scene` 去 `bevy_render` 移植方案（只换地基）

> 范围：`pkg` 下唯一真正依赖 `bevy_render` 的 crate 是 `prism_render_scene`（Cargo 依赖 + 496 处引用 / 310 文件 / 7 个 `plugin.rs`）。
> 目标：把它的 **设备/命令/资源/帧编排** 从 `bevy_render` 换到 Prism RHI（`prism_render_driver` + `prism_render_graph` + `prism_render_app`），**不改其材质/着色/渲染逻辑**。
> 其余 pkg crate（`prism_render_visibility`、`prism_render_app`）只在注释里提到 `bevy_render`，无真实依赖，无需改动。

## 1. 为什么必须整体协调，不能零敲碎打

`prism_render_scene` 是**一个编译单元**：把某个文件的 `RenderDevice` 换成 RHI trait，会立刻打断所有调用点的类型一致性。因此移植按「切片」推进，每片内部自洽、可独立编译与回归；片内不再用 bevy_render。

### 1.1 已证实的硬约束（2026-10-12 勘察）

1. **不存在「小步可编译」的切法**：crate 是单个编译单元，任何设备/资源类型变更都会打断全 crate 的类型一致；shader 侧的 `bevy_render::maths` / `bevy_render::utils` 也依赖生产着色器系统，无法单独抽走。
2. **换 RHI 等于换宿主**：它的渲染世界是 `bevy_ecs + bevy_app + bevy_render::{RenderApp, ExtractSchedule, Render, RenderSystems}` 模型；换 RHI 必须同时换成 `prism_app/prism_ecs` 的 render sub-app（+ `prism_asset` 等）。这是整机架构 port，不是「换依赖行」。
3. **桥接过渡不通**：bevy 的 `RenderQueue` / `RenderAdapter` 是私有字段（只有 `RenderDevice::wgpu_device()` 暴露 device，queue/adapter 没有），拿不到 wgpu 队列复用同一设备；另开 Prism 设备则是**双设备**，跨设备资源不可用，中途必错。

4. **只删 `bevy_render` 一行做不到（2026-10-12 复核）**：`prism_render_scene` 还依赖 `bevy_pbr` 与 `bevy_core_pipeline`，而这两个 crate **自身直接依赖 `bevy_render`**（`crates/bevy_pbr/Cargo.toml:61`、`crates/bevy_core_pipeline/Cargo.toml:32`）。因此即便把 crate 里的 `bevy_render::*` 全部改掉、删掉那行依赖，`bevy_render` 仍会经 `bevy_pbr`/`bevy_core_pipeline` 传递进来。
   **推论**：要让 pkg 不再依赖 `bevy_render`，必须同时替换 `bevy_pbr`（PBR 管线）与 `bevy_core_pipeline`（核心 pass），也就是要先把它们的功能落到 Prism 侧（其余 `bevy_mesh/image/camera/light/material/color/asset` 同理，只是它们不直接拉 `bevy_render`）。这已经是「prism_render_scene 整体脱 Bevy」的范围。

**推论**：只能整机 port。工作模式二选一——

- **A 就地 port**：`prism_render_scene` 红若干轮，期间 `--workspace` 构建不可用，port 完成才转绿。
- **B 隔离 port**：独立 worktree/分支里 port，主工作区始终可编译，完成后合回（推荐）。

### 1.2 接线层必须对齐既有「四前端 + 正交风格轴」设计（PBR / NPR / 混合 / 自定义）

**权威设计**：`prism_material_pipeline_design_zh.md` §1/§7——同一引擎**同时**是顶级 PBR / NPR / 混合 / 自定义引擎，四者皆一等：
共享 GPU-driven 基底（几何/可见性/material id、光/影/GI 数据、跨切面基底）**算一次**，四条**并存前端**各自取数据、叠自己的响应：
延迟 PBR 前端 / forward+ 前端 / NPR 风格化前端 / **自定义前端槽**；风格轴 `illumination = Lit / Stylized / Unlit / Custom` 正交叠在任意前端上；
**混合 = 逐像素 material id + tile 分类路由**（因共享光/影/GI 而连贯，不是各前端各建一套）。

**已 committed 的落地锚点（移植时必须保留语义）**：
- CPU 分桶：`prism_render_shading/src/classification.rs`——`classify_material_header` 投影到 9 值 `MaterialShadingClass`，`ShadingWorkPlan::build` 分出同 permutation 的 bin。
- GPU 路由：`prism_render_scene/src/shaders/shading_resolve.wesl`——9 个 `SHADING_CLASS_*` 常量 + `switch params.shading_class` 分派到 `shade_principled`/`shade_toon`(NPR)/`shade_subsurface`/`shade_clearcoat`/`shade_cloth`/`shade_hair`/`shade_water` 等。
- NPR 描边白送：`prism_render_shading/src/outline.rs::outline_id_edge` 直接吃 vis-buffer 的 material id。

**移植约束**：换 `bevy_render` 时，**不能**把 `MeshPipelineKey` + `StandardMaterial`（PBR-only）当成总体管线键；要保持「逐像素 material id + tile 分类 + 9 类前端分派 + 风格轴」这套语义。`Core3d`/`Opaque3d`/`MeshPipelineKey`/`Core3dSystems` 只是 bevy 的**接入点**，替换时按上文设计做等价实现（Prism graph pass + 模型/前端注册 + `prism_render_visibility::binning` 的分批键）。

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

  **进度**：地基第一部分已落（`d3b39d514`）——新增 `src/gpu/`（`PrismGpu` RHI 设备/队列资源 + `PrismGpuPlugin`，best-effort、无 adapter 时资源缺席），`completion.rs` 已改用它（`FrameSync::on_submitted_work_done`）。crate 仍编译、1602 测试通过。

  **规模实测**：`RenderDevice`/`RenderQueue` 在 crate 内共 **594 处**、`bevy_render::renderer` 相关 287 行 / 256 文件。设备/资源类型的替换是**全局协调一次过**的 pass（改一处即全 crate 类型不一致），只能在一次大 pass 内完成，期间 crate 红。
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
