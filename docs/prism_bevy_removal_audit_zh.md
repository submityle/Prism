# Prism Bevy 废弃迁移审计

审计日期：2026-10-21（Asia/Shanghai）
审计范围：workspace 默认生产路径、`pkg/prism_render_scene` 及其与原 Bevy 渲染宿主的边界。
写入范围：本次只修改本文件。没有删除源码、没有批量改动依赖、没有修改边界脚本或其他文件。

## 1. 结论

当前不能声称 Prism 已移除 Bevy。

事实是：Prism 的原生应用/渲染分支已经存在，但 workspace 默认成员仍是根包 `bevy`；`pkg/prism_render_scene` 仍是一个以 Bevy ECS、Bevy App、Bevy RenderApp 和 Bevy Core3d 为宿主的渲染场景包。它同时挂接 Prism 的 RHI、Render Graph、Render Architecture、Material、Shading 和 Visibility 契约，因此当前状态是“原生分支与 Bevy 场景分支并存”，不是“原生分支已经成为唯一生产路径”。

本审计只确认迁移事实与删除条件，不把设计文档、已有算法、GPU parity 测试或 RHI 桥接误记为完成迁移。

本次新增证据表明，原生路径已经落地了两个有限切片：`prism_render_app`
现在把 `SceneRegistry` 的 geometry header/LOD 元数据投影到原生 storage
buffer，并处理脏槽位、删除 tombstone、容量增长和 device generation 全量
重建；`prism_editor_app` 也已经有不依赖 Bevy 的 native scene vertical slice，
可从文档候选进入 indexed geometry 上传、共享 wgpu ID/depth/debug-color pass、
异步点选/框选，再回写完整 selection identity 和 controller revision。
这些事实不等于完整 geometry vertex/index streaming、material/visibility/draw
链、最终像素链或整个编辑器运行时已经替代 Bevy。

## 2. 可复现审计方法

所有基线数据来自当前 checkout，使用以下命令得到：

```sh
cargo metadata --manifest-path Cargo.toml --format-version 1 --no-deps
cargo tree -p prism_render_scene -e normal --depth 2 --offline --no-dedupe
cargo tree -p prism_render_app -e normal --depth 2 --offline --no-dedupe
cargo tree -p prism_editor_app -e normal --depth 2 --offline --no-dedupe
rg -n 'bevy_[A-Za-z0-9_]+|::bevy::|\bbevy::' pkg/prism_render_scene/src --glob '*.rs'
rg -n 'bevy_[A-Za-z0-9_]+|::bevy::|\bbevy::' pkg/prism_editor_app pkg/prism_editor_viewport pkg/prism_editor_world --glob '*.rs' --glob 'Cargo.toml'
```

可重复门禁脚本：

```sh
tools/check_prism_bevy_boundary.sh --report
tools/check_prism_bevy_boundary.sh --strict
```

`--report` 用于当前迁移期间输出事实并返回成功；`--strict` 是迁移完成后的删除门。当前仓库运行 `--strict` 必须失败，因为违规项仍然真实存在。脚本只检查 Prism 生产边界，不会删除文件或修改 manifest。

## 3. 当前 Cargo 基线

### 3.1 Workspace 根与默认生产选择

当前 `Cargo.toml` 同时是 workspace 根和名为 `bevy` 的 package：

| 项目 | 当前事实 | 证据 |
|---|---|---|
| 根包 | `bevy`，版本 `0.20.0-dev` | `Cargo.toml:1-13` |
| workspace 成员 | 259 | `cargo metadata` |
| workspace 默认成员 | 1 | `cargo metadata` |
| 默认成员名称 | `bevy` | `cargo metadata` |
| workspace 内 `bevy*` 包 | 79 | `cargo metadata` |
| 根包默认 feature | `2d`、`3d`、`ui`、`audio` | `Cargo.toml:362-394` |
| 根包正常依赖 | `bevy_internal`、可选 `bevy_dylib` | `Cargo.toml:980-986` |

因此默认 `cargo check`/默认 package 选择仍然围绕 Bevy 根包，而不是 `prism_app + prism_render_app + prism_render_runtime`。根包的默认 3D feature 会启用 `bevy_render`、`bevy_core_pipeline`、`bevy_pbr` 等 Bevy 渲染模块；这与 Prism 原生渲染路径是两套宿主。

### 3.2 `prism_render_scene` 的真实依赖

`pkg/prism_render_scene/Cargo.toml:12-28` 仍声明以下 17 个正常 Bevy 依赖：

```text
bevy_app
bevy_asset
bevy_ecs
bevy_derive
bevy_math
bevy_material
bevy_image
bevy_mesh
bevy_camera
bevy_color
bevy_core_pipeline
bevy_log
bevy_pbr
bevy_platform
bevy_render
bevy_transform
bevy_shader
```

当前 `prism_render_scene` manifest 不再直接声明 `bevy_tasks`。
`text/unit/shading/composition_ecs.rs` 的三个被测系统只执行普通 ECS `Query`
遍历和 GPU 命令记录，不调用 `par_iter`、异步任务或任务池；移除原先仅用于
初始化 Bevy 全局 `ComputeTaskPool` 的测试代码不会改变该切片的调度语义。
因此当前 manifest 的静态计数是 17 条 Bevy 正常依赖；这些依赖仍由其他旧适配器
路径使用，不能据此宣称整个 `prism_render_scene` 已经 Bevy-free。

直接声明已删除 `bevy_light`，但这不等于灯系统脱离 Bevy：`cargo tree -p prism_render_scene -e normal` 显示 `bevy_core_pipeline` 和 `bevy_pbr` 仍分别把 `bevy_light` 带入正常依赖闭包。直接依赖、传递依赖和源码宿主必须同时清零，才可以通过删除门。

当前 `prism_render_scene` 还直接依赖以下 Prism 层：

```text
prism_render_graph
prism_render_architecture
prism_mesh
prism_render_material
prism_render_shading
prism_math
prism_material_pcm
prism_render_visibility
prism_render_driver
prism_render_driver_wgpu
```

这说明它不是纯 Bevy 包，也不是原生 Prism 包；它是一个正在迁移中的混合宿主。

### 3.3 Cargo tree 的关键真实路径

以下是 `cargo tree -e normal --depth 2` 与 `cargo metadata` 的图摘要。箭头表示当前 manifest/解析闭包中的真实依赖，不表示建议架构：

```text
workspace default
└── bevy (root package)
    └── bevy_internal
        ├── Bevy app/ECS/platform/assets/state
        └── 默认 2D/3D/UI/audio feature closure

prism_render_scene
├── bevy_app ──> bevy_ecs, bevy_platform, bevy_reflect, bevy_tasks, bevy_utils
├── bevy_asset ──> bevy_app, bevy_ecs, bevy_reflect, asset macros and loaders
├── bevy_core_pipeline ──> bevy_render, bevy_camera, bevy_light, bevy_shader
├── bevy_pbr ──> bevy_core_pipeline, bevy_render, bevy_light, bevy_material, bevy_mesh
├── bevy_render ──> bevy_app, bevy_ecs, bevy_asset, bevy_camera, bevy_image,
│                   bevy_material, bevy_mesh, bevy_shader, bevy_tasks, bevy_window
├── bevy_camera / bevy_mesh / bevy_image / bevy_material / bevy_transform / ...
└── prism_render_graph + prism_render_architecture + prism_render_driver(+wgpu)
    + prism_mesh + prism_render_material + prism_render_shading
    + prism_render_visibility + prism_math + prism_material_pcm

prism_render_app
├── prism_app ──> prism_ecs + prism_platform + prism_settings + prism_time
├── prism_render_runtime ──> prism_render_driver + prism_render_graph + wgpu backend
├── prism_render_graph + prism_render_architecture + prism_render_driver
├── prism_mesh + prism_image + prism_render_material + prism_render_shading
├── prism_render_visibility + prism_render_gpudriven + prism_particle_app
└── prism_transform + prism_asset + prism_tasks + prism_window

prism_scene
└── prism_asset + prism_ecs (+ optional prism_reflect)
```

`prism_render_scene` 的 direct list 里没有直接写 `bevy_light`，但其真实 transitive path 通过 `bevy_core_pipeline`、`bevy_pbr` 到达 `bevy_light`。它也没有被 `prism_render_app` manifest 反向依赖，因此不能把图上两个相邻的 Prism 包拼成一个已经运行的端到端生产链。

### 3.4 原生 Prism 分支的真实依赖

`pkg/prism_render_app/Cargo.toml:10-48` 的正常依赖全部是 `prism_*`、标准 RHI 依赖或 wgpu backend 依赖，没有 `bevy_*`。它依赖的主要原生层是：

```text
prism_app
prism_ecs
prism_render_runtime
prism_render_driver
prism_render_graph
prism_render_architecture
prism_render_material
prism_render_shading
prism_render_visibility
prism_mesh
prism_image
prism_transform
prism_particle_app
prism_render_gpudriven
```

`pkg/prism_app/Cargo.toml:10-24` 只依赖 `prism_ecs`、`prism_time`、`prism_platform`、`prism_settings` 等 Prism 层；`pkg/prism_scene/Cargo.toml:9-12` 只依赖 `prism_asset`、`prism_ecs`、可选 `prism_reflect`。这两个包已经具备脱 Bevy 的基础，但尚未自动替代 `prism_render_scene` 的全部功能。

### 3.5 编辑器原生包的 Cargo 边界

`pkg/prism_editor_app/Cargo.toml:10-46` 的依赖是 `prism_*`、`wgpu`、窗口和
异步 I/O；`pkg/prism_editor_viewport/Cargo.toml:10-12` 只有 `futures-lite`
和 `wgpu`；`pkg/prism_editor_world/Cargo.toml:10-14` 依赖 Prism editor、
asset、ECS 和 scene 包。当前三个 manifest 都没有 `bevy_*` 依赖，编辑器相关
源码扫描也没有 Bevy import。这证明该编辑器切片的宿主边界是原生的，但不
证明它已覆盖完整运行时场景、材质系统或 Prism 的全部渲染功能。

### 3.6 `prism_render_scene` 当前没有 Cargo 直接消费者

`cargo metadata` 反向检查没有发现任何 workspace package 通过 manifest 直接依赖 `prism_render_scene`。这不代表它没有价值：多个原生包和测试通过相对路径 `include_str!`/文档引用使用其 WESL shader 资产，例如 `prism_render_material_gpu`、`prism_render_material` 和 `prism_material_schema`。迁移时必须先把 shader 资产的 owner 与发布路径独立出来，再处理 crate 删除；不能把“无直接 Cargo 消费者”当成“可以直接删除目录”。

## 4. 真实源码与运行边界

### 4.1 `prism_render_scene` 仍由 Bevy 宿主驱动

关键证据：

- `pkg/prism_render_scene/src/plugin.rs:4-13` 导入 `bevy_app::{App, Plugin}`，并导入 `bevy_render::{ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems}`。
- `pkg/prism_render_scene/src/plugin.rs:62-170` 通过 Bevy `App` 安装插件，取得 `RenderApp` 子应用，在 `ExtractSchedule` 和 `Render` 中注册系统，并使用 Bevy 的 GPU 资源初始化与 `RenderGraphSystems::Finish`。
- `pkg/prism_render_scene/src/shading/plugin.rs:644-750` 继续使用 Bevy `RenderApp`、`ExtractSchedule`、`Core3d`、`Core3dSystems`、`ViewBinnedRenderPhases`、`DrawFunctions` 和 `SpecializedMeshPipelines`。
- `pkg/prism_render_scene/src/gpu/rhi.rs:25-31` 直接使用 `bevy_ecs` 与 `bevy_render` 的 `RenderAdapter`、`RenderDevice`、`RenderQueue`、`Buffer`、`Texture`、`BindGroup` 等类型。
- `pkg/prism_render_scene/src/gpu/context.rs:3-7` 明确写出 Bevy Render 与 Prism RHI 并存，且迁移尚未完成。
- `pkg/prism_render_scene/src/completion.rs:4-8` 仍以 Bevy `RenderQueue` 作为主机提交队列的一部分；`track_host` 的完成语义绑定于 Bevy 提交阶段。

当前源码统计：`pkg/prism_render_scene/src` 有 646 个 Rust 文件，其中 490 个文件包含 `bevy_*`/`bevy::` 词元，匹配 1973 次。这个数字包含注释、文档和测试；它不是迁移工作量的唯一估计，但足以证明该 crate 仍是深度 Bevy 耦合，而非少量 facade。

### 4.2 原生渲染宿主已经存在，但不是 `prism_render_scene` 的宿主

`pkg/prism_render_app/src/plugin.rs:6-40` 使用 `prism_app::App`、`prism_ecs` 的 schedule、`prism_render_runtime::RenderRuntime` 和 `prism_render_graph`。`pkg/prism_render_app/src/plugin.rs:274-330` 创建原生 render sub-app、RHI runtime、render target、material buffers 和 native mesh pipelines。

但 `pkg/prism_render_scene` 没有被接入这个原生 `RenderApp`。当前 native plugin 初始化的是自己的 `LightingGpuState`、`MeshPipelines::spine`、native scene GPU state 和 native geometry GPU state 等路径；这只能证明原生渲染脊柱以及 geometry 元数据上传切片存在，不能证明 `prism_render_scene` 的 geometry/material/lighting/shading/water/cloth/raytrace 已进入 native production path。

### 4.3 Native geometry GPU：已落地的有限范围

这不是“完整 geometry renderer”，而是当前已经可由源码确认的 metadata/LOD
GPU 投影切片：

- `pkg/prism_render_architecture/src/geometry/gpu.rs:15-229` 定义固定 ABI 的
  `GpuGeometryHeader`、`GpuGeometryLod`、固定 `MAX_GEOMETRY_LODS` stride，
  并校验 handle、LOD 数量、screen error 和 LOD base；`GeometryUploadPlanner`
  可按脏索引选择 sparse、range 或 full upload。
- `pkg/prism_render_app/src/scene.rs:133-220` 的 `SceneRegistry` 保留 geometry
  epoch、稳定 capacity 和 dirty index；replace/remove 会追踪变更，删除槽位
  保留给 GPU tombstone。
- `pkg/prism_render_app/src/native_geometry_gpu.rs:128-340` 将 registry 投影为
  header/LOD storage buffers，处理 dirty 增量、删除槽位零值、buffer 扩容、
  device generation 变化时的 full snapshot，并在成功后清理对应 dirty epoch。
  同文件的 null-backend 测试覆盖 tombstone、增量计划和 device-generation
  full snapshot 约束。
- `pkg/prism_render_app/src/plugin.rs:361-374,441-467,598-648` 创建资源，
  把 extraction/prepare 接入 native render schedule，并在 shutdown 处理销毁
  或丢弃。

因此目前可以说 geometry 元数据和 LOD 表已经进入原生 GPU owner；不能说
vertex/index 数据流、mesh asset residency、material binding、visibility、
draw queue、完整 shader pass 或最终 HDR 像素链已经完成。

### 4.4 Editor native scene vertical slice

编辑器切片的当前调用链是：

1. `pkg/prism_editor_app/src/desktop_document.rs:53-127` 发起有界 save/load；
   load 先准备并校验 candidate，再调用 renderer prepare，安装 UI 失败时
   丢弃 renderer candidate，成功后才 commit。
2. `pkg/prism_editor_app/src/desktop_scene.rs:47-92` 使用显式完整 identity、
   indexed positions/indices、camera 和 depth 组成 engine-neutral
   `DesktopSceneDocument`；`SceneDocumentAdapter` 是文档模型到场景描述的
   native 边界。
3. `pkg/prism_editor_app/src/desktop_scene.rs:197-235,1029-1054` 使用共享
   wgpu 设备，将 indexed geometry 上传到 `SceneIdGeometryRenderer`；
   `encode_preview` 同一 encoder 追加 depth-tested ID pass 和 load-preserving
   debug-color pass。源码明确不声称完整 material rendering 或自动 model-to-mesh
   extraction。
4. `pkg/prism_editor_app/src/desktop_scene.rs:781-1008,1059-1146` 通过同一
   viewport/target 进行异步点选和有界 GPU 框选；scene、camera、target、
   geometry、bounds 或 controller revision 过期时拒绝 late result，再把有效
   的完整 selection identity 交给 controller。
5. `pkg/prism_editor_app/src/controller.rs:243-327,438-489,577-617,893-950`
   负责对象注册、selection、多选、反射属性编辑、undo/redo、dirty model
   revision 和 save acknowledgement；这使“文档 -> 场景 -> GPU picking ->
   selection/controller”调用链可被源码追踪。

该切片证明编辑器 viewport/scene 交互可以在无 Bevy 的原生包中闭环；它不
等于完整 editor shell、完整运行时 scene instantiation、材质/光照渲染或
整个 Prism Bevy 边界已经清零。

### 4.5 生产与非生产边界

| 边界 | 当前归属 | 结论 |
|---|---|---|
| workspace 默认 package | 根包 `bevy` | 旧 Bevy 默认生产选择，未迁移 |
| Bevy 示例与官方 `crates/bevy_*` | 根 package 的 examples、workspace 成员与 CI | 非 Prism 原生生产路径；仍占据 workspace 默认构建和大量 CI |
| `prism_render_scene` plugin/shading/visibility/material/geometry | Bevy render sub-app | 迁移中的混合路径，不能标成 native |
| `prism_render_app` + `prism_render_runtime` | Prism App/RHI/RenderGraph | 原生生产候选路径，已有独立测试与 null/wgpu backend |
| `prism_editor_app` + `prism_editor_viewport` + `prism_editor_world` | Prism editor/controller + shared wgpu | 已有无 Bevy 的 native scene vertical slice；不是完整运行时渲染/编辑器系统 |
| `prism_scene`、`prism_render_architecture`、`prism_render_visibility`、`prism_render_material` | 无 Bevy manifest 依赖 | 原生契约和算法层；不能自动代表完整渲染接线已完成 |
| `prism_render_scene` 的 WESL shader 资产 | 被多个测试/材质包相对路径引用 | 资产仍有跨包消费者，必须先完成 owner 迁移 |
| `tools/build-easefunction-graphs`、Bevy examples、Bevy CI 工具 | Bevy 上游工具链 | 非 native 生产路径；应单独处置，不混入场景 crate 删除切片 |

## 5. `pkg/prism_render_scene` 模块处置分类

分类含义：

- **direct reuse**：算法、ABI、shader 或无宿主的数据契约可以直接保留，经过独立 owner 复核后继续使用。
- **adapt**：保留核心语义，但需要改数据入口、资源句柄、调度或后端接口。
- **rewrite**：当前实现依赖 Bevy 宿主、Bevy GPU 类型或 Bevy phase，必须以 Prism contract 重新实现；不是简单改 import。
- **remove**：只服务旧桥接、旧比较路径或迁移后不应继续存在的代码。

| 模块 | 当前职责 | 分类 | 迁移处置 |
|---|---|---|---|
| `buffers` | Bevy GPU buffer 与 bind group 上传 | rewrite | 改为 `prism_render_driver` buffer/lease、`prism_render_graph` resource handle 和 completion retirement |
| `cloth` | cloth authoring、Bevy render-world extraction、compute pipeline/dispatch | adapt | `prism_render_architecture::cloth` 的 solver/kernel/ABI 可 direct reuse；Bevy plugin、resource、dispatch 需 native rewrite |
| `compare` | Bevy ECS scene mirror 与 parity diagnostics | remove | 完成迁移前保留为证据；native parity 进入统一 diagnostics 后删除 Bevy 版本 |
| `completion` | 以 Bevy RenderQueue 绑定提交完成 | rewrite | 统一接入 `prism_render_runtime`/`FrameSync`/generation retirement |
| `consumer` | Bevy SystemParam 读取 scene buffer | rewrite | 改成 native render context、typed resource lease 和 frame graph pass input |
| `diagnostics` | Bevy Resource 形式的上传统计 | adapt | 数据字段可复用，owner 改为 native render frame diagnostics |
| `extract` | main world 到 Bevy render world 的实体提取 | rewrite | 改为 `prism_app` sub-app extract、`prism_ecs` snapshot/command 或 RenderIntent/SceneRegistry 输入 |
| `fabric_domains` | domain payload 到 Render Fabric 的无设备 CPU lowering | direct reuse | 保持在 native architecture/scene adapter 边界，继续由 domain owner 提供身份和 payload |
| `fabric_frame` | retained scene 到每帧 Render Fabric 的 lowering，仍使用 Bevy ECS | adapt | payload/identity 语义可复用，frame owner 改为 native scene frame |
| `geometry` | mesh asset、shading geometry registry、Bevy GPU buffers | rewrite | geometry metadata/LOD ABI、SceneRegistry projection 和 native storage upload 已落地；mesh asset residency、vertex/index streaming、material/visibility/draw binding 仍需 native rewrite |
| `gpu` | Bevy device 上的 RHI bridge/import/export | remove | native runtime 接管设备和资源所有权后删除 `from_bevy`、import lease 和双设备语义 |
| `lighting` | canonical lighting snapshot 到 Bevy GPU light buffers | adapt | canonical ABI/cluster 数学可复用；extract、buffer owner、bind group 和 scheduler 改为 native |
| `material` | Bevy material/asset extraction、texture heap、GPU material resources | rewrite | `prism_render_material`/schema/PCM ABI direct reuse；asset lifecycle、heap、bind group、upload 改为 native |
| `opaque` | Bevy mesh phase、draw function、indirect queue | rewrite | 对接 `prism_render_app` queue/draw/frame graph；不能保留 Bevy phase facade |
| `particle` | particle payload 到 scene/fabric 的桥 | adapt | payload/identity 契约可复用，native particle app 和 render queue 接线替换 Bevy extraction |
| `plugin` | PrismGpuScenePlugin 与 Bevy RenderApp 组装 | rewrite | 拆为 Prism native scene/render plugins，并由 native app 显式组合 |
| `raytrace` | Bevy render-world ray traversal service | adapt | `prism_render_architecture::ray_scene` ABI/CPU golden 可 direct reuse；资源、pipeline、dispatch 和 frame graph native rewrite |
| `scene` | Bevy AssetId/Mesh/Entity 到 retained GPU scene mirror | rewrite | `SceneHandle`、geometry/material revisions 和 snapshot 可复用；entity/asset/GPU owner 改为 `prism_scene`/`prism_asset`/native render scene |
| `shading` | Core3d 驱动的 lighting/GI/reflection/post passes | rewrite | WESL 和算法可分层 direct reuse/adapt；pass scheduling、resource declarations、pipelines、views 和 output 必须 native rewrite |
| `visibility` | Bevy camera/phase/HZB/visibility plugin | adapt | `prism_render_visibility` 的 canonical work generation 可复用；view extraction、GPU resource、draw queue 改 native |
| `water` | Bevy author chain、Core3d surface draw、GPU compute | adapt | `prism_render_architecture::water` kernel/ABI 可 direct reuse；surface draw、history、resource ownership、schedule native rewrite |
| `src/shaders` | WESL/WGSL shader modules | direct reuse | 保留可验证算法和 shader ABI；把 asset owner 从 `prism_render_scene` 拆到 native shader package/registry，禁止继续依赖 Bevy embedded asset API |
| `tests`、`text` | Bevy host tests、GPU parity、fixtures | adapt/remove | CPU/ABI/GPU oracle 迁至 native package；只服务旧 Bevy bridge 的 fixtures 在 native parity 门通过后删除 |

### 5.1 依赖模块处置分类

| 当前 Bevy 模块 | 处置 | Native owner | 删除条件 |
|---|---|---|---|
| `bevy_app` | rewrite | `prism_app` | native plugin、sub-app、schedule、lifecycle 全覆盖并通过集成测试 |
| `bevy_ecs` | rewrite | `prism_ecs` | scene/material/visibility/shading 生产数据不再使用 Bevy ECS 类型 |
| `bevy_render` | rewrite | `prism_render_runtime` + `prism_render_driver` + `prism_render_graph` + `prism_render_app` | 设备、资源、phase、submit、recovery、readback 全部走 native |
| `bevy_core_pipeline` | rewrite | `prism_render_graph` + native render sets/view/output | Core3d pass 顺序和附件依赖有 native graph 证据 |
| `bevy_pbr` | rewrite | `prism_render_material` + `prism_render_shading` | PBR/opaque/transparent/shadow material ABI 与最终像素链由 native 测试证明 |
| `bevy_asset` | adapt | `prism_asset` | scene/material/image 生命周期、revision、streaming 和 shader asset registration 全部 native |
| `bevy_mesh` | adapt | `prism_mesh` | mesh vertex/index/LOD/geometry upload 和 draw binding native |
| `bevy_image` | adapt | `prism_image` | image decode/identity/upload/texture residency native |
| `bevy_camera` | adapt | `prism_camera` + `prism_camera_ecs` | view matrix/frustum/jitter/multiview extract native |
| `bevy_transform` | adapt | `prism_transform` + `prism_transform_ecs` | transform propagation and temporal history native |
| `bevy_math` | direct reuse only where algorithmically verified | `prism_math` | 生产源码不再 import Bevy math；逐算法 parity 已登记 |
| `bevy_color` | adapt | `prism_math`/material/output contracts | HDR/color/output ABI native |
| `bevy_material` | rewrite | `prism_render_material` + `prism_material_schema` | material authoring/runtime/GPU binding no longer uses Bevy types |
| `bevy_shader` | adapt | `prism_shader` + native shader registry | shader composition, embedding and permutation cache native |
| `bevy_platform` | rewrite/adapt | `prism_platform`/`prism_platform_os` | native app/window/backend closure contains no Bevy platform type |
| `bevy_log` | adapt | `tracing`/`prism_diagnostic` | logging is no longer installed through Bevy plugin graph |
| `bevy_derive` | remove from Prism production path | `prism_ecs_macros` and native derives | no Prism production manifest or source imports it |
| `bevy_tasks` | remove from Prism production path | `prism_tasks` | dev/test bridge removed from scene crate |

## 6. 删除门

### Gate A：默认生产路径切换

必须同时满足：

1. workspace 默认成员不再是根包 `bevy`，而是明确的 Prism production launcher/package；
2. 默认 feature 不再自动启用 `2d`、`3d`、`ui`、`audio` 这些 Bevy product profiles；
3. 默认 launcher 的 render call path 可追踪为 `prism_app -> prism_render_app -> prism_render_runtime -> prism_render_graph -> prism_render_driver`；
4. 默认路径不加载 `PrismGpuScenePlugin` 或任何 Bevy `RenderApp` 插件。

### Gate B：`prism_render_scene` 宿主替换

必须满足：

1. `pkg/prism_render_scene/Cargo.toml` 不再有任何 `bevy_*` 正常或开发依赖；
2. `src` 不再 import `bevy_app`、`bevy_ecs`、`bevy_render`、`bevy_core_pipeline`、`bevy_pbr` 或其他 Bevy crate；
3. `RenderApp`、`ExtractSchedule`、`Core3d`、`RenderSystems`、`ViewBinnedRenderPhases`、`DrawFunctions`、`SpecializedMeshPipelines` 等宿主概念均由 native owner 替代，而不是改名 facade；
4. `gpu/rhi.rs` 的 Bevy handle import/export bridge 被删除；所有资源有单一 native owner。

### Gate C：资产与 shader owner 收敛

必须满足：

1. WESL/WGSL shader 有 native registry/package owner；
2. `include_str!`、shader codegen、GPU parity 测试不再要求 `prism_render_scene` 目录作为隐式资产仓；
3. material/mesh/image/scene 的 revision、generation、retirement 语义在 native owner 中可独立构建和测试；
4. 删除 `prism_render_scene` 不会让 `prism_render_material*`、`prism_render_shading`、GPU parity tests 的路径解析失效。

### Gate D：效果与性能证据

必须有 native 路径的真实证据，而非 storage-buffer parity：

- opaque triangle/mesh 最终 HDR attachment 像素；
- direct light、shadow、GI、reflection、material、view/resize 改变最终像素；
- depth occlusion、HDR > 1、display transform、device recovery、generation retirement；
- null backend 记录 path 与 wgpu/Metal 实机 path 的契约一致；
- 记录 P50/P95/P99 的 CPU submit、GPU timestamp 或等价可解释指标，以及峰值显存/资源 resident budget；
- benchmark 必须绑定同一质量配置、分辨率、场景和 backend，不能用单个单测替代产品性能结论。

### Gate E：workspace 清理

最后才允许：

1. 从 workspace members 移除被证明没有生产或测试消费者的 Bevy crate；
2. 删除根 `bevy` package、旧 examples/CI 专属配置或将其移入隔离 legacy workspace；
3. `cargo metadata` 的生产闭包、`cargo tree` 和 source scan 同时通过；
4. 全 workspace `cargo check/test/clippy` 与 native GPU acceptance 通过；
5. 文档、脚本、锁文件、资产路径和 CI 不再把 Bevy 作为默认生产入口。

## 7. 精确下一步

按以下顺序执行，避免在仍有双宿主时批量删除：

1. **冻结生产入口**：新增一个明确的 Prism production launcher/package，先让默认构建能选中它；保留根 Bevy package 仅作为隔离 legacy workspace 成员，直到所有引用迁出。
2. **迁移 shader owner**：把 `pkg/prism_render_scene/src/shaders` 的共享 WESL/WGSL 资产迁到 native shader package 或明确的 asset registry，更新 `prism_render_material`、`prism_material_schema`、`prism_render_material_gpu` 的路径引用并保留 ABI parity。
3. **补完 native scene substrate**：geometry metadata/LOD registry projection 已存在；仍需完成 material revision、view snapshot、GPU resource lease、vertex/index residency、queue/draw binding 和 completion retirement，并以调用链和最终像素证据验收。
4. **按宿主边界迁移 `extract -> geometry -> material -> visibility -> opaque`**：每个切片先有 native CPU contract，再接 native render graph，最后移除对应的 Bevy adapter；每片都通过 `cargo tree`、source scan、null backend 和真实 wgpu 像素测试。
5. **迁移 lighting/shadow/GI/reflection/shading**：复用 `prism_render_architecture` 和 shader 算法，重写 Bevy Core3d/phase/resource owner；将最终 HDR color attachment 作为验收对象。
6. **迁移 water/cloth/raytrace/particle**：保留已验证 kernel/ABI，替换 Bevy plugin、resource、dispatch、history 和 recovery；完成后把 parity 测试移入 native crates。
7. **切换默认路径**：默认 launcher 只组装 `prism_app`、`prism_render_app` 和 native domain plugins，运行 `tools/check_prism_bevy_boundary.sh --strict`。
8. **清理 legacy workspace**：只有 Gate A-E 全部满足后，才移除 Bevy workspace member、旧根 package 和不再需要的官方 examples/tools；每个删除批次单独 commit。

## 8. Commit 与 CI 约束

本审计变更应单独提交，本次 commit 内容只包含：

```text
docs/prism_bevy_removal_audit_zh.md
```

迁移实现阶段建议把脚本加入 native boundary job：

```yaml
- name: Audit Prism Bevy boundary
  run: tools/check_prism_bevy_boundary.sh --strict
```

在 Gate A 之前 CI 不应把 `--strict` 标成已通过；可以运行 `--report` 产出当前基线。严格门禁启用后，以下任一变化都必须失败：Prism production manifest 新增 `bevy_*` 依赖、生产 source root 新增 Bevy import、workspace 默认成员回到 Bevy package。

## 9. 当前审计结论的边界

本文件没有声称：

- Bevy 已经从 workspace 移除；
- `prism_render_scene` 已经成为 native Prism crate；
- `prism_render_app` 已经接管 `prism_render_scene` 的全部渲染功能；
- native geometry GPU 已经完成 vertex/index streaming、material binding、visibility、draw 或最终像素链；
- editor native scene vertical slice 已经成为完整运行时场景实例化或完整编辑器系统；
- Direct/Shadow/GI/Reflection/HDR 的完整最终像素链已完成；
- 全 workspace 或物理 GPU 生产验收已经通过。

本文件确认的新增有限事实是：native geometry metadata/LOD GPU projection 已
接入原生 render schedule，且 editor native scene vertical slice 已在无 Bevy
编辑器包中形成文档、indexed geometry、ID/depth/debug pass、异步选择和
controller revision 的可追踪调用链；这些事实不会缩短仍然存在的 Bevy 删除门。

这些结论必须等删除门对应的源码、Cargo 图、运行时调用链和最终像素证据同时存在后再更新。
