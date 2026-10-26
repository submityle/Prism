# Render Fabric / Scene / TAA / Response 性能验收切片（2026-10-26）

## 入口审计

当前工作区已有以下相关入口：

- `pkg/prism_render_scene/tests/response_composition_gpu.rs`：生产 response composition shader 的 GPU 对拍，并输出小规模 host submit + readback 分位数。
- `pkg/prism_render_scene/src/shading/taa/resources.rs` 引入的 `gpu_tests.rs`：生产 TAA prepare、bind group、dispatch、history completion/retirement 的严格 GPU 用例。
- `pkg/prism_render_scene/src/composition.rs` 中的 `CompositionFabric::new()`：冻结共享 Scene Fabric selection，包含内建场景、response composition 和 TAA 声明。
- `pkg/prism_render_scene/src/fabric_runtime.rs`：Render Fabric 的 compile / prepare / typed recorder / encode 入口；其完整宿主提交仍由 Render World 持有。

新增独立目标：

```text
prism_render_scene --test render_fabric_scene_taa_response_performance
```

源码：[render_fabric_scene_taa_response_performance.rs](/Users/wangkailong/Documents/GitHub/Prism/pkg/prism_render_scene/tests/render_fabric_scene_taa_response_performance.rs)

## 测量范围

本切片使用真实生产 shader/pipeline，命令链为：

```text
response_composition
  → response_to_hdr
  → taa_resolve
  → TAA history ping-pong
  → HDR history texture + auxiliary readback
```

它是当前无需修改 readback 迁移和 Decal 实现即可运行的最大公共 GPU 帧切片，不冒充完整 31-pass 应用帧。几何、光照、Scene object import 和宿主 Render World recorder 不在此目标内；因此该结果不能升级为完整 Scene/Fabric/TAA 全帧性能结论。

测试每帧真实执行：

- response left/auxiliary seed 到工作资源的 GPU copy；
- production response radiance 和 typed auxiliary 两个 compute entry；
- production `response_to_hdr.wgsl`；
- production `taa_resolve.wesl` 编译后的 `resolve_taa`，包括 20-byte immediate ABI、motion/reactive 输入和双 history slot；
- 最终 history texture 与 auxiliary buffer 的真实 GPU copy/readback。

默认尺寸为 `640×360`，可用环境变量提升到目标分辨率：

```text
PRISM_REQUIRE_GPU=1 \
PRISM_RENDER_FRAME_BENCH_WIDTH=1920 \
PRISM_RENDER_FRAME_BENCH_HEIGHT=1080 \
PRISM_RENDER_FRAME_BENCH_WARMUP=3 \
PRISM_RENDER_FRAME_BENCH_SAMPLES=31 \
cargo test -p prism_render_scene --features legacy-bevy \
  --test render_fabric_scene_taa_response_performance -- --nocapture
```

## 指标口径

测试不写入预置性能数字，所有结果来自本次实际运行：

- `host_submit_ms`：`queue.submit` 调用本身的 host wall-clock 时间。
- `host_completion_ms`：提交后到 `device.poll(wait_indefinitely())` 返回的 wall-clock 时间。
- `readback_ms`：完成确认后，HDR history 与 auxiliary staging buffer map/read/unmap 的 wall-clock 时间。
- `total_ms`：从 command encoding 开始到两类 readback 完成的 wall-clock 时间。
- `gpu_chain_ms`：若 physical adapter 支持 `TIMESTAMP_QUERY`，由 GPU timestamp 包围 response composition 到 TAA resolve 的实际设备时间；不支持时明确打印 `unavailable`，不以 host 时间替代。
- `logical_resource_peak_mib`：测试显式创建的 buffer/texture payload 和 staging payload 的逻辑字节总和，包含 row padding；不是驱动 VRAM telemetry，也不包含隐式 heap、压缩、tile memory 或 staging allocator 的额外开销。
- `checksum`：每个样本对最终 history HDR readback 与 auxiliary readback 的确定性字节校验；样本间必须一致且非零，避免只测到空命令链。

所有 host/GPU 时间均输出 `P50/P95/P99`。队列深度固定为 1，每个样本等待完成后才开始下一样本，避免把未完成帧的排队时间错误算成吞吐。

当物理适配器不可用时，测试仍会先输出 Render Fabric selection、workload 配置以及带/不带
`TIMESTAMP_QUERY` 的逻辑资源计划值，然后输出结构化阻塞状态；此时不会输出或推导任何
P50/P95/P99。该逻辑资源值是可复现的替代指标，不是 GPU/VRAM telemetry。

## 验收边界

通过该目标只能证明：在当前适配器、当前分辨率和当前 workload 下，production response → HDR → TAA 计算链真实可提交、可完成、可读回，并有可复现的分位数和逻辑资源计量。

仍不能由该目标证明：31-pass 全场景 recorder、完整 Scene object payload、完整几何/光照 producer、完整跨域 SceneFabric、真实 VRAM 峰值、第二 physical adapter parity，或完整最终图像 parity。没有适配器、缺少 TAA 所需格式/IMMEDIATES，或 `PRISM_REQUIRE_GPU=1` 下缺少 physical GPU 时，测试失败/严格拒绝，而不是把 skip 当成性能通过。

## 本轮实际验证（2026-10-26）

已完成：

- 本轮新增切片的初始专用 target `cargo check --offline` 通过，真实 GPU target 也曾完成编译并进入测试；随后为验证无适配器结构化输出而清理了该 package 的构建产物，重新编译被工作区既有 `pkg/prism_render_scene/src/particle/fabric.rs` 的语法错误和 `SceneFabricRuntimeError: Clone` 错误阻塞。
- `git diff --check` 通过。
- 强制真实 GPU 命令已执行，但当前主机未发现可用 Metal physical adapter，测试精确失败：

```text
PRISM_REQUIRE_GPU: no adapter: No suitable graphics adapter found; noop not requested, vulkan support not compiled in, metal found no adapters, dx12 support not compiled in, gl not requested, webgpu not requested
```

因此本轮没有可发布的真实 `host_submit_ms`、`host_completion_ms`、`readback_ms`、
`total_ms` 或 `gpu_chain_ms` P50/P95/P99，也没有写入占位数字。当前可复现结果是：

- 代码与 target 编译通过；
- 物理 GPU 验收被上述 adapter 阻塞；
- 无适配器模式会输出 `logical_resource_plan_mib_without_timestamps`、
  `logical_resource_plan_mib_with_timestamps`，并明确标记
  `status=blocked_gpu_metrics_unavailable`；
- 当前全包重新编译还需先修复上述既有 `particle/fabric.rs` 阻塞；这不是本切片的性能数据，也不应被解释为 GPU 失败的替代原因；
- 获得可用 Metal/Vulkan/DX12 physical adapter 后，仍需重新运行强制命令并记录真实
  P50/P95/P99、checksum、timestamp 能力和最终逻辑资源峰值。
