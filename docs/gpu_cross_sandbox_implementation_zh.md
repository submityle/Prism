# GPU / 跨沙盒验证记录

验证日期：2026-10-16（保留 2026-10-15 的跨沙盒证据）。

本记录覆盖测试层的真实 WGPU/Metal 入口、RenderFabric 定向测试、response composition，以及水体 `MaskBlend` raster 的设备级能力门禁。2026-10-15 的跨沙盒运行已拿到 Apple M2 Metal adapter：response composition 与水体 raster 严格 GPU 对拍均通过。没有修改 `docs/refactoring-dev/`。

本轮新增的完整水体输出 target 为 `pkg/prism_render_scene/tests/water_hybrid_gpu.rs`。
它覆盖生产 raster、render-FX、resident composition、HDR export 和 readback；但本机
本轮没有可用硬件 adapter，首次完整构建还受到磁盘空间不足阻断，所以该 target
目前只能标记为“待跨沙盒执行”，不能沿用旧的 shader-module gate 结果冒充输出通过。

本轮同时补齐 RHI `Rgba32Uint` 的格式编号、wgpu 转换、Null 生命周期和硬件 readback
测试入口。`prism_render_driver` 88 项、Null backend 3 项、wgpu backend 28 项通过；
`tests/rgba32uint.rs` 的 2 个真实输出用例在当前主机因无 adapter 均明确 `SKIPPED`，
因此尚未形成本机真实 GPU readback 证据，不能把 CPU/Null 结果当作硬件通过。

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

## 2026-10-15 Terrain draw GPU parity

独立的 `pkg/prism_world_gpu/tests/terrain_draw.rs` 在同一 Apple M2 Metal
适配器上完成真实 indexed terrain draw 验证：`1 passed / 0 failed`。测试
同时覆盖完整网格上传、局部 dirty patch 上传、index buffer draw、readback
以及局部更新后与完整重建的逐像素结果比较。

本次 fixture 的可复核证据为：完整 vertex upload `2592 bytes`，局部 patch
upload `800 bytes`；局部上传路径的 GPU 输出与完整上传路径逐像素一致。
该结果证明的是 terrain mesh 的索引/局部更新/GPU draw 合同，不把一个小型
fixture 的耗时外推为通用性能指标；正式性能仍需按内容规模记录 GPU 时间、
显存和 P50/P95/P99。

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
| `CARGO_TARGET_DIR=/private/tmp/prism-render-scene-target PRISM_REQUIRE_GPU=1 WGPU_BACKEND=metal cargo test -p prism_render_scene --test resident_composition_gpu --offline -- --nocapture` | **3 passed / 0 failed**；Apple M2 Metal，HDR storage capability gate、resident target/pipeline identity 与 GPU producer→resident composition→HDR readback 全部通过。 |
| `PRISM_REQUIRE_GPU=1 WGPU_BACKEND=metal cargo test --manifest-path pkg/prism_world_gpu/Cargo.toml --features gpu --test terrain_draw -- --nocapture` | **1 passed / 0 failed**；Apple M2 Metal，完整上传 `2592 bytes`、局部 patch `800 bytes`，GPU draw readback 与完整重建逐像素一致。 |

## 2026-10-16 本机阻断证据

| 命令 | 结果 |
|---|---|
| `env CARGO_TARGET_DIR=/private/tmp/prism-render-target cargo test -j 1 -p prism_render_architecture --lib water::frame_golden::whole_frame_plan_matches_the_golden_digest -- --nocapture` | **1 passed / 0 failed**；当前 canonical frame-plan digest 为 `0x2d726174148032c7`。 |
| `env CARGO_TARGET_DIR=/private/tmp/prism-render-target cargo test -j 1 -p prism_render_scene --test water_hybrid_gpu` | **未进入测试**；编译阶段 `No space left on device (os error 28)`。 |
| `PRISM_REQUIRE_GPU=1 cargo test -p prism_world_gpu --features gpu --test gpu_parity` | **0 passed / 1 failed**；adapter 获取失败，Metal 无可用 adapter；严格 GPU 门禁正确拒绝。 |

跨沙盒主机本轮有 Apple M2 Metal adapter，response composition、水体 Hybrid、resident composition 与 terrain draw 均有真实 GPU readback/shader gate 通过证据。

## GPU 验证写集

本次新增/修改仅限：

- `pkg/prism_ui_render_backend/tests/gpu_parity.rs`
- `pkg/prism_render_scene/tests/response_composition_gpu.rs`
- `pkg/prism_render_scene/text/integration/resident_composition_gpu.rs`
- `pkg/prism_render_scene/text/unit/shading/composition_ecs.rs`
- `docs/gpu_cross_sandbox_implementation_zh.md`

resident composition 已在目标 Apple M2 Metal 设备上以 `PRISM_REQUIRE_GPU=1` 单独执行并通过；当前水体 raster 的 7-group layout 满足该设备的 `max_bind_groups=8`，GPU 门禁统一保留 `PASS/SKIP/FAIL` 严格分类。

## 2026-10-15 几何 LOD 与 draw-range 接线

RenderScene 的几何 LOD 生命周期、独立 prepared mesh 映射、每 LOD buffer class、visibility/draw-bin、opaque/transparent/raster/shadow/VSM 消费，以及 indexed/non-indexed draw range fail-closed 校验已完成接线。对应单元测试与 RenderScene 全量测试通过；无效或陈旧几何元数据只会跳过 draw，不会提交越界 GPU 命令。

## 2026-10-15 材质响应与 8-word execution plan

`material_classification_gpu` 已在 Apple M2 Metal 上通过真实 compute dispatch、prefix/scatter readback 与 CPU plan parity：2/2 passed；本轮覆盖 logical/physical stride、68 个桶、140 个可接受 work item，以及 stale/foreign-owner/overflow 诊断。

`brdf_channels_gpu` 已在同一真实适配器上编译并执行 504 个完整 `SurfaceSample` 案例、16,632 个 RGB 分量和 guard-row 检查。CPU/WESL 结果的逐位/单 ULP 差异不作为跨后端硬约束：Metal 的 FMA contraction 与设备超越函数会造成低位差异；当前生产门禁固定为每个有限分量最多 32 ULP，并保留逐分量 bit-pattern 诊断。实际最差值为 24 ULP，因此该门禁通过；fixture 仍会在发现超预算时失败，不会把异常降级为 skip。
