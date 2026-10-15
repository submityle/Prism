# GPU / 跨沙盒验证记录

验证日期：2026-10-15。

本记录覆盖测试层的真实 WGPU/Metal 入口、RenderFabric 定向测试、response composition，以及水体 `MaskBlend` raster 的设备级能力门禁。2026-10-15 的跨沙盒运行已拿到 Apple M2 Metal adapter：response composition 与水体 raster 严格 GPU 对拍均通过。没有修改 `docs/refactoring-dev/`。

## 2026-10-15 水体 Hybrid 门禁增量

`PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test water_hybrid_golden` 在真实 Apple M2 Metal adapter 上完成 adapter/device 获取；当前水面 raster 使用 7 个 bind groups，本机 `max_bind_groups=8`，严格门禁输出明确 `PASS`。

## 2026-10-15 World proxy GPU twin

独立 crate `pkg/prism_world_gpu` 的真实 GPU 门禁：

```text
PRISM_REQUIRE_GPU=1 WGPU_BACKEND=metal cargo test \
  --manifest-path pkg/prism_world_gpu/Cargo.toml --features gpu \
  --test wgsl_compile --test gpu_parity -- --nocapture
```

结果：WGSL parse/validation `1 passed`；Apple M2 Metal adapter 上 compute pipeline 创建、proxy SoA 上传、visibility/frustum/range/LOD dispatch、MAP_READ readback 与 CPU golden 逐 proxy 四字对拍 `1 passed`。adapter evidence：`Metal` / `IntegratedGpu` / `Apple M2`，无 optional required feature。

该测试证明的是 proxy visibility/cull/LOD 合同和 shader/readback 闭环，不宣称完整 world renderer、GPU PCG、流送直传或性能指标；性能仍需独立 benchmark 记录上传字节、GPU pass 时间、显存和 P50/P95/P99。

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
| `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test response_composition_gpu --test water_hybrid_golden --offline -- --nocapture` | response composition **2 passed / 0 failed**，真实 Apple M2 Metal adapter/device 获取成功，1320 cases / 14520 auxiliary jobs GPU readback 通过（host submit+readback 6.654458 ms）；water Hybrid **15 passed / 0 failed**，真实 adapter/device 与 shader module gate 通过。 |

跨沙盒主机本轮有 Apple M2 Metal adapter，response composition 与水体 Hybrid 均有真实 GPU readback/shader gate 通过证据。resident composition 未在本轮命令中执行，不作通过声明。

## 写集边界

本次新增/修改仅限：

- `pkg/prism_ui_render_backend/tests/gpu_parity.rs`
- `pkg/prism_render_scene/tests/response_composition_gpu.rs`
- `pkg/prism_render_scene/text/integration/resident_composition_gpu.rs`
- `pkg/prism_render_scene/text/unit/shading/composition_ecs.rs`
- `docs/gpu_cross_sandbox_implementation_zh.md`

resident composition 仍需在目标设备上以 `PRISM_REQUIRE_GPU=1` 单独执行；当前水体 raster 的 7-group layout 已满足 Apple M2 Metal 的 `max_bind_groups=8`，后续设备验证应继续保留 `PASS/SKIP/FAIL` 的严格分类。

## 2026-10-15 材质响应与 8-word execution plan

`material_classification_gpu` 已在 Apple M2 Metal 上通过真实 compute dispatch、prefix/scatter readback 与 CPU plan parity：2/2 passed；本轮覆盖 logical/physical stride、68 个桶、140 个可接受 work item，以及 stale/foreign-owner/overflow 诊断。

`brdf_channels_gpu` 已在同一真实适配器上编译并执行 504 个完整 `SurfaceSample` 案例、16,632 个 RGB 分量和 guard-row 检查。CPU/WESL 结果的逐位/单 ULP 差异不作为跨后端硬约束：Metal 的 FMA contraction 与设备超越函数会造成低位差异；当前生产门禁固定为每个有限分量最多 32 ULP，并保留逐分量 bit-pattern 诊断。实际最差值为 24 ULP，因此该门禁通过；fixture 仍会在发现超预算时失败，不会把异常降级为 skip。
