# Prism RHI 阶段一（M0/M1/M2）落地设计与进度

> 面向 `prism_render_driver`（RHI 门面）阶段一「wgpu 单后端打底」的可执行设计与进度记录。
> 设计总纲与分层见 `prism_render_driver_design_zh.md` §20–§22；本文只写**阶段一怎么落地、落到哪**。
> 所有 `pkg` / `os-backends` crate 不含任何 Unreal Engine / Unity 源码或衍生代码。

## 1. 里程碑与验收（对齐 `prism_render_driver_design_zh.md` §22）

| 里程碑 | 内容 | 验收 |
| --- | --- | --- |
| **M0** | Instance/Adapter/Device/Queue/**Surface+Swapchain** 门面 + `Backend` 抽象 + wgpu 后端 + **Null 后端** → 画出三角形（接 window/shader） | 离屏三角形像素级通过；窗口帧可 acquire/present |
| **M1** | 完整资源/视图/采样器 + 多线程录制 + **拷贝** + 自动屏障 + **frames-in-flight** → 带纹理 PBR 物体 | 拷贝命令 GPU 对拍；帧在飞有上界；PBR 物体出图 |
| **M2** | `prism_render_graph` + 材质/场景真实路径压在 wgpu 上；bindless / multi-draw-indirect-count / compute 剔除按 wgpu 能力上 → 千物体 GPU-driven 基准 | 千物体场景 + GPU-driven draw 开销基准 |

## 2. 后端切换架构（关键决策）

后端切换**不发生在 runtime**，而发生在 RHI 的 bring-up 门面；上层只碰 trait：

- `prism_render_driver::Instance`（`src/instance.rs`）：`enumerate_adapters` / `request_adapter[_for_surface]` / `create_device` / `create_surface`。关联类型 `Adapter/Device/Queue/Surface` 承载后端具体类型。
- `prism_render_driver::Adapter`：`info()` + `capabilities()`。
- `prism_render_driver::Surface`（`src/surface.rs`）：`configure` → `acquire` → `image` → `present`；acquire 到的交换链图由后端注册成普通 `TextureId`/`TextureViewId`，图按外部纹理 import。
- 窗口句柄走 `raw-window-handle`（仅句柄 ABI，不含窗口系统）：`create_surface(Arc<W>, &device, &queue)`，`W: HasWindowHandle + HasDisplayHandle`。
- `prism_render_runtime::RenderRuntime<I: Instance>` **对后端泛型**：加后端 = 多一个 `Instance` 实现 + feature，runtime/graph/scene 一行不改（`prism_render_driver_design_zh.md` §28）。`RenderRuntime::<WgpuInstance>::wgpu(extent)` 只是默认后端便捷构造。

阶段二的原生后端（Vulkan/Metal/D3D12）是同一个 `Instance` 门面的新实现，条件触发（§22 阶段二）。

## 3. 已落地（committed）

- **M0 门面**：`Instance`/`Adapter`/`Surface`/`SurfaceConfig`/`SurfaceCapabilities`/`SurfaceImage`/`SurfaceError`/`DeviceDescriptor`/`DeviceError`/`DeviceType`/`PowerPreference`（`pkg/prism_render_driver/src/{instance.rs,surface.rs}`）。
- **wgpu 后端**：`WgpuAdapter`、`impl Instance for WgpuInstance`、拥有型 `WgpuSurface`（`'static`，`Arc<Window>`）、交换链图像注册/释放钩子（`DeviceInner::register_texture*`）。
- **Null 后端**：`os-backends/prism_render_driver_null`（单一 CPU/Noop adapter；generation 校验的 id 生命周期；surface 全流程）。
- **runtime**：`RenderRuntime<I: Instance>` 后端无关，`headless` + `wgpu` 便捷构造。
- **M0 验收**：`rhi_graph_draws_a_triangle` 在真实 GPU 上离屏画出三角形并回读中心红、四角黑。
- **M1 命令/同步**：RHI 补齐 `CopyCommand`（4 变体）+ `Pass::Transfer` + `push_transfer_pass` + `SetPushConstants`；`prism_render_graph` 把 Transfer pass 录进命令缓冲；`FrameSync`（可等待提交）落在 RHI，runtime 有界 frames-in-flight。
- **M1 多线程录制**：`prism_render_graph` 加 `parallel` feature，pass 在 worker 池并行录制后按 `plan.order` 确定性装配，与串行结果逐字节相等（可选 feature，默认 no_std 不变）。
- **M1 资源/采样器验收**：`tests/textured_quad.rs` —— 上传纹理 → sampler + bind group → 采样着色 → 离屏回读，4096 像素**精确等值**断言。
- **M0 窗口呈现**：runtime 的 `render_frame` 走 `acquire → import 交换链 → 构图 → execute → present`；`tools/render_triangle` 是真窗口 demo（真机 8s 无错运行）。
- **M2 起步**：`prism_render_driver` 补 `DrawIndexedIndirect` + `MultiDrawIndirect`（multi-draw-indirect-count）；新 crate `pkg/prism_render_gpudriven` 做 compute 视锥剔除 → GPU 写间接参数 → indirect 绘制，GPU 结果与 CPU 参考集**完全一致**。
- **M2 场景 + 基准**：`prism_render_gpudriven` 经 `prism_render_graph` 真绘制 ~1000 物体；`benches/benches/prism_gpu_cull` 实测——GPU-driven 逐帧 CPU 提交 ~290 ns 且与物体数无关，朴素路径 10000 物体 58 µs（**232×**）。顺手修出「剔除原子游标未重置」真 bug。
- **M2 bindless**：RHI 加 descriptor-array（buffer/texture/sampler/storage-texture 四类）+ wgpu `binding_array` 翻译，四类均真机**精确值**验证；按 wgpu 30 实际能力门控。
- **集成**：`pkg/prism_render_app` —— runtime 接进 `prism_app` 的 render sub-app（extract + `Render` schedule；GPU 状态只在 render world，`forbid(unsafe_code)`）。
- **CR 修复**：① `prism_math::Frustum::from_view_proj` 原用 OpenGL `[-1,1]` 近平面，与 Prism 自身 `[0,1]` 投影不符——改为 `[0,1]` 默认 + 新增 `_gl`。② `Surface::discard` 补齐：帧已 acquire 但图编译失败时不再泄漏交换链 id。

## 4. 后续切片（依赖有序）

> 范围约定：当前只做 **RHI 接入**。已有材质/渲染系统（`prism_render_material` / `prism_render_scene` / `prism_render_shading` / `prism_render_architecture`）不替换、不重建。

1. **自动屏障可观测**：`StateTracker` 已算屏障；wgpu 阶段为 no-op，补 `validation` 档把推导屏障与后端追踪对拍。
2. **`prism_render_app` 接真窗口**：把 render sub-app 的离屏目标换成 `prism_window_winit` 的 surface。
3. **RHI 接入已有渲染系统**：把现役 `prism_render_scene` 的设备/命令面收敛到 RHI（只换地基，不改其材质/逻辑）。

## 5. 验证门禁

- 每个 crate 独立 `cargo build/test/clippy --all-targets` 全绿、零告警、可回退。
- GPU 测试用「无 adapter 即 skip」模式，保证 GPU-less CI 也通过；设 `PRISM_REQUIRE_GPU=1` 时缺 adapter **直接失败**，避免「以为跑了其实全跳过」（沙盒默认看不到 adapter，真机验证须跨沙盒 + 该环境变量）。
- 画面/拷贝正确性走像素与字节级对拍（`WgpuQueue::read_texture_rgba8`）。
- 切片粒度：一次 CR + 修复 + 本地 commit。
