# GPU / 跨沙盒验证记录

验证日期：2026-10-15。

本记录覆盖测试层的真实 WGPU/Metal 入口、RenderFabric 定向测试、response composition，以及水体 `MaskBlend` raster 的设备级能力门禁。2026-10-15 的水体门禁已修复一个真实 shader ABI 错误，并将低 `max_bind_groups` 设备明确分类为 skip；没有修改 `docs/refactoring-dev/`。

## 2026-10-15 水体 Hybrid 门禁增量

`PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test water_hybrid_golden` 首次在真实设备上暴露 `SurfaceSample.geometry_opacity` 不存在的字段访问，已删除错误写入。重新运行后，真实适配器和设备请求成功，但本机适配器报告 `max_bind_groups=8`，而完整水面 raster 绑定需要 10 组，因此普通门禁输出明确 `SKIP`；这不是 GPU shader-module 通过。若设置 `PRISM_REQUIRE_GPU=1`，能力不足会按设计直接失败。

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
| `cargo test -p prism_render_architecture render_fabric --lib --offline -- --nocapture` | **70 passed / 0 failed / 11,195 filtered**；当前工作树已包含并运行 RenderFabric PCM bridge 测试；存在 3 个既有 warning。 |
| `cargo test -p prism_ui_render_backend --features gpu --test gpu_parity --offline -- --nocapture` | **7 passed / 0 failed**；7 个用例均输出 `no usable WGPU device, skipping`，因此这是 CPU 侧测试通过，不是 GPU 对拍通过。 |
| `PRISM_REQUIRE_GPU=1 cargo test -p prism_ui_render_backend --features gpu --test gpu_parity --offline -- --nocapture` | **0 passed / 7 failed**；每个用例报告 `could not acquire a usable WGPU device`。 |
| `cargo test -p prism_ui_render_backend --features gpu --test dock_gpu_parity --offline -- --nocapture` | **0 passed / 1 failed**；报告 `dock GPU parity requires a real wgpu adapter; refusing to skip validation`。 |
| `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_gpudriven --lib --offline -- --nocapture` | **11 passed / 7 failed**；7 个 GPU 用例真实报告 `PRISM_REQUIRE_GPU is set but no adapter was found`。 |
| `cargo test -p prism_render_scene --test response_composition_gpu --offline -- --nocapture` | **2 passed / 0 failed**；CPU packing 通过，GPU parity 在 adapter 获取阶段显式 skip，真实输出为 `No suitable graphics adapter found; noop not requested, vulkan support not compiled in, metal found no adapters, dx12 support not compiled in, gl not requested, webgpu not requested`。这不是 GPU 对拍通过。 |
| `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test response_composition_gpu --offline -- --nocapture` | **1 passed / 1 failed**；CPU packing 通过，GPU parity 按严格能力门禁失败，真实输出为 `required GPU adapter unavailable: No suitable graphics adapter found; noop not requested, vulkan support not compiled in, metal found no adapters, dx12 support not compiled in, gl not requested, webgpu not requested`。 |
| `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test resident_composition_gpu --offline -- --nocapture` | 同上，未进入 GPU 初始化，被同一现有编译错误阻断。 |

当前隔离环境没有可用 WGPU adapter；没有可宣称的 Metal 实机对拍、response composition readback 或 resident HDR 输出。response composition 的普通模式只保留显式 skip，严格模式保留真实失败，不能把默认通过解释为 GPU 已运行。

## 写集边界

本次新增/修改仅限：

- `pkg/prism_ui_render_backend/tests/gpu_parity.rs`
- `pkg/prism_render_scene/tests/response_composition_gpu.rs`
- `pkg/prism_render_scene/text/integration/resident_composition_gpu.rs`
- `pkg/prism_render_scene/text/unit/shading/composition_ecs.rs`
- `docs/gpu_cross_sandbox_implementation_zh.md`

未覆盖的 response/resident 命令需要先由上游修复 `classification.rs` 与当前生成的 `ResponseProgramLayout` 不一致，再在带 Metal/WGPU adapter 的主机上以 `PRISM_REQUIRE_GPU=1` 重跑。上述 compile failure 和 no-adapter failure 都保留为失败证据，不用 skip 或伪造输出替代。
