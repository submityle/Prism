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

**权威设计**：`prism_material_pipeline_refactor_zh.md` §3.5/§8.5——四条线（PBR / NPR / 混合 / 自定义）**是同一条 OpenPBR über 上的正交响应模式，不是四套管线**；运行层**只有一条执行路径，四线共用**：

```
1 解包    prism_unpack_surface（变长 OpenPBR 瓣集）
2 分类    per-tile 9-way（illumination / render_class 分波前）
3 瓣求值  热=spec-constant 置换 ⇔ 冷=全瓣 über + mask 兜底
4 光照源  对象空间缓存命中 → IncomingRadiance ｜ 未命中 → 屏幕空间直接积分（唯一分支点）
5 响应    respond()：PBR 乘 albedo·BRDF ｜ NPR ramp/描边 ｜ 混合按 illumination 遮罩 ｜ 自定义走契约
6 合成    + 视角相关项 → 输出
```

差异**只落在步骤 5 的响应函数**；缓存（步骤 4）/降级（步骤 6）**不改 key、不新增 PSO、不产生并行着色栈**。风格轴 `illumination = Lit(0)/Stylized(1)/Unlit/Custom(3)` 正交叠在任一处；**混合 = 逐像素/逐区域 illumination + per-tile 分类路由**。共享基底另见 `prism_aaa_advanced_features_zh.md` §1、`prism_material_pipeline_design_zh.md` §1/§7。

**已 committed 的落地锚点（移植时必须保留语义）**：
- CPU 分桶：`prism_render_shading/src/classification.rs`——`classify_material_header` 投影到 9 值 `MaterialShadingClass`，`ShadingWorkPlan::build` 分出同 permutation 的 bin。
- GPU 路由：`prism_render_scene/src/shaders/shading_resolve.wesl`——9 个 `SHADING_CLASS_*` 常量 + `switch params.shading_class` 分派到 `shade_principled`/`shade_toon`(NPR)/`shade_subsurface`/`shade_clearcoat`/`shade_cloth`/`shade_hair`/`shade_water` 等。
- NPR 描边白送：`prism_render_shading/src/outline.rs::outline_id_edge` 直接吃 vis-buffer 的 material id。

**移植约束**：换 `bevy_render` 时，必须保住这条**单执行路径**——既不能把 `MeshPipelineKey` + `StandardMaterial`（PBR-only）当总体管线键，也**不能拆成四条管线**（§3.5 明确「线 = 管线；模式 = 响应」）。`Core3d`/`Opaque3d`/`Core3dSystems`/`MeshPipelineKey` 只是 bevy 的**接入点**，替换时按上面的执行模型做等价实现（Prism graph pass + `binning` 分批键 + 单一 `respond()` 分派）。

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

## 6. 接线层方案（六条替换，保住单执行路径）

「接线层」= 把 bevy 的**接入点**换成与「四线共用一条执行路径」（§1.2 / `prism_material_pipeline_refactor_zh.md` §3.5/§8.5）等价的 Prism 接线。**不改 shading 逻辑，不拆管线**。逐条：

| # | bevy 接入点 | Prism 目标 | 现状 |
| --- | --- | --- | --- |
| 1 | `RenderApp` / `ExtractSchedule` / `Render` / `RenderStartup` / `RenderSystems` / `RenderGraph(GraphSystems)` / `init_gpu_resource` | `prism_render_app::{RenderApp, Render}` + extract + **Prism 版 render system sets**（Extract/Prepare/Queue/Phase/Cleanup）+ `prism_render_graph` 承载 pass | host 骨架已有；**system sets 待建** |
| 2 | `view::{ExtractedView,RetainedViewEntity}` / `renderer::ViewQuery` / `schedule::camera_driver` | `prism_render_architecture::view_family::{ViewRegistry,ViewEntry}` + Prism camera-driver set | view registry 已有；**driver set 待建** |
| 3 | `core_3d::{Opaque3d,Opaque3dBatchSetKey,Opaque3dBinKey}` / `Core3d` / `Core3dSystems::{MainPass,PostProcess}` | `prism_render_graph` 的 pass + Prism **phase 标签**（Opaque/Transparent/NPR/Custom）；`RenderCommand<Opaque3d>` → Prism 绘制命令 | pass 图已有；**phase 标签待建** |
| 4 | `MeshPipelineKey` / 分批键 | `prism_render_visibility::binning::DrawBinKey`（**已具备** `pipeline_class`/`pass_mask`）+ 由 **瓣集 + illumination + render_class + flags** 生成的 Prism pipeline key（§8.5：key **只**含步骤 1 瓣集 + 特征 flags，质量档/缓存/降级不进 key） | binning 已有；**key 映射待接** |
| 5 | `queue_material_meshes` / `MeshPipelineSystems` / `RenderMeshInstances` / `StandardMaterial` / `PbrPlugin` | `prism_render_material`（IR/record）+ `prism_render_architecture::gpu_scene::{InstanceRecord,MaterialRecord}` + 间接/bindless 绘制（复用 `prism_render_gpudriven` 的间接多绘） | IR/record 已有；**队列/特化待建** |
| 6 | `render_resource::*` / `renderer::{RenderDevice,RenderQueue}` / `TextureCache` / `PipelineCache` / `SpecializedMeshPipelines` | RHI（`PrismGpu` + `prism_render_driver`）+ Prism 版资源/管线缓存 + mesh 管线特化 | `PrismGpu` 已落；**缓存/特化待建** |

**不变量**：步骤 2 的 per-tile 9-way 分类、步骤 5 的单一 `respond()` 分派、`binning` 的 `pipeline_class` 只是同一个 key 的三种视图——迁移后**仍是那条唯一执行路径**，不按 PBR/NPR/混合/自定义拆 PSO。
