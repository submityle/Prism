# GPU / 跨沙盒验证记录

验证日期：2026-10-15。

本记录覆盖测试层的真实 WGPU/Metal 入口、RenderFabric 定向测试、response composition，以及水体 `MaskBlend` raster 的设备级能力门禁。2026-10-15 的跨沙盒运行已拿到 Apple M2 Metal adapter：response composition 严格 GPU 对拍通过；水体 raster 因 `max_bind_groups=8` 小于所需 10，严格门禁按设计失败。没有修改 `docs/refactoring-dev/`。

## 2026-10-15 水体 Hybrid 门禁增量

`PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test water_hybrid_golden` 在真实 Apple M2 Metal adapter 上完成 adapter/device 获取，但本机报告 `max_bind_groups=8`，而完整水面 raster 绑定需要 10 组，因此普通门禁输出明确 `SKIP`；严格门禁输出明确失败。这不是 GPU shader-module 通过，也没有把能力不足记为通过。

## 测试入口与 gating

- RenderFabric：`cargo test -p prism_render_architecture render_fabric --lib --offline -- --nocapture`。
- UI WGPU parity：`cargo test -p prism_ui_render_backend --features gpu --test gpu_parity --offline -- --nocapture`；默认无设备时显式打印 skip，`PRISM_REQUIRE_GPU=1` 时无可用设备会失败。
- Dock GPU parity：`cargo test -p prism_ui_render_backend --features gpu --test dock_gpu_parity --offline -- --nocapture`；当前测试本来就要求真实 adapter，无 adapter 会失败。
- Response composition：`PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test response_composition_gpu --offline -- --nocapture`。
- Resident composition：`PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test resident_composition_gpu --offline -- --nocapture`。

response composition 测试在获取 adapter 后检查 compute workgroup 与 storage-buffer 容量；resident composition 额外检查 `Rgba16Float` 的 `STORAGE_BINDING` 和 `Rgba32Uint` 的 sampled/render-attachment usage。设备创建失败同样按 gating 处理：普通模式记录 skip，强制模式失败。新增纯门禁测试验证缺失 HDR storage 或不足 workgroup 时会在分配前拒绝。

## 真实输出

| 命令 | 结果 |
|---|---|
| `cargo test -p prism_render_architecture render_fabric --lib --offline -- --nocapture` | **73 passed / 0 failed / 11,200 filtered**；当前工作树已包含并运行 RenderFabric PCM bridge 测试；存在 3 个既有 warning。 |
| `cargo test -p prism_ui_render_backend --features gpu --test gpu_parity --offline -- --nocapture` | **7 passed / 0 failed**；7 个用例均输出 `no usable WGPU device, skipping`，因此这是 CPU 侧测试通过，不是 GPU 对拍通过。 |
| `PRISM_REQUIRE_GPU=1 cargo test -p prism_ui_render_backend --features gpu --test gpu_parity --offline -- --nocapture` | **0 passed / 7 failed**；每个用例报告 `could not acquire a usable WGPU device`。 |
| `cargo test -p prism_ui_render_backend --features gpu --test dock_gpu_parity --offline -- --nocapture` | **0 passed / 1 failed**；报告 `dock GPU parity requires a real wgpu adapter; refusing to skip validation`。 |
| `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_gpudriven --lib --offline -- --nocapture` | **11 passed / 7 failed**；7 个 GPU 用例真实报告 `PRISM_REQUIRE_GPU is set but no adapter was found`。 |
| `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test response_composition_gpu --test water_hybrid_golden --offline -- --nocapture` | response composition **2 passed / 0 failed**，真实 Apple M2 Metal adapter/device 获取成功，1320 cases / 14520 auxiliary jobs GPU readback 通过；water Hybrid **14 passed / 1 failed**，唯一失败为真实能力门禁：`max_bind_groups=8`，所需 10。 |

跨沙盒主机本轮有 Apple M2 Metal adapter，response composition 已有真实 GPU readback 对拍通过；水体 Hybrid 因绑定组上限不足保留严格失败证据。resident composition 未在本轮命令中执行，不作通过声明。

## 写集边界

本次新增/修改仅限：

- `pkg/prism_ui_render_backend/tests/gpu_parity.rs`
- `pkg/prism_render_scene/tests/response_composition_gpu.rs`
- `pkg/prism_render_scene/text/integration/resident_composition_gpu.rs`
- `pkg/prism_render_scene/text/unit/shading/composition_ecs.rs`
- `docs/gpu_cross_sandbox_implementation_zh.md`

resident composition 仍需在目标设备上以 `PRISM_REQUIRE_GPU=1` 单独执行；水体 raster 需要降低绑定组数量或提供 `max_bind_groups >= 10` 的设备后再重跑。上述能力失败保留为失败证据，不用 skip 或伪造输出替代。
