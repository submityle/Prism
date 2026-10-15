# Virtual Texture Feedback 闭环状态

本闭环从可见性/外部反馈输入开始，经 `feedback decode -> readback -> residency streamer -> persistent page table -> atlas/render binding` 进入下一帧采样。

## 已完成

- `prism_virtual_texture_gpu` 提供 GPU feedback decode、GPU page lookup/resolve 和可复用 feedback readback；这些 GPU API 的主路径返回 `Result`，不会把设备丢失或 readback 失败伪装成空结果。
- `GpuPageTableStorage` 保留持久化 GPU buffer；`upload_resizing` 使用新 buffer 完成分配和上传后再替换旧 buffer，扩容失败不会破坏当前表。
- `GpuContext::try_real_headless` 只接受 wgpu 报告为 integrated/discrete 的真实 adapter；CPU/software adapter 不会伪装成 GPU 通过 gate。
- readback 的 map、poll、mapped-range、device-lost 错误均可传播；设备丢失后 frame pipeline 自动回到 CPU。
- `VirtualTexturePipeline` 统一 CPU fallback 和 GPU decode/residency/page-table 上传，CPU streamer 仍是确定性 residency source of truth。
- parity、malformed feedback、persistent table expansion（包括扩容后旧表内容保留）和真实 adapter gate 测试已覆盖；WGSL 解析/验证和 CPU golden 始终在无 GPU 环境运行。无真实 GPU 时，real-GPU parity 只显式 skip，不计作 GPU 通过。

## 语义边界

- wgpu adapter gate 证明的是 integrated/discrete adapter 上的 compute/readback 执行，不等同于所有平台的硬件 sampler-feedback 原生格式支持。
- render_scene 的可见性 attachment readback 仍由 `prism_render_scene` 的 RenderQueue/RenderDevice 负责；本 crate 接收外部 feedback buffer 或 CPU grid，不伪造 raster feedback。
- page-table storage 的扩容是逻辑表上传前的 replacement allocation；旧 storage 在 replacement 成功前仍有效。设备完全丢失后不能复活原 wgpu device，必须重建 `GpuContext`，再重新上传 CPU streamer 当前表。
- `PRISM_REQUIRE_GPU` 只用于硬件门禁：缺少 adapter/device，或 wgpu 选择了 CPU/software adapter，都会失败；默认模式才允许无真实 GPU 的测试显式 skip。

## 验证

```text
CARGO_TARGET_DIR=/private/tmp/prism-vt-target cargo test --locked --offline -p prism_virtual_texture_gpu --all-targets -- --nocapture
PRISM_REQUIRE_GPU=1 CARGO_TARGET_DIR=/private/tmp/prism-vt-gpu-target \
  cargo test --locked --offline -p prism_virtual_texture_gpu --all-targets
```

第一条命令验证无 GPU 模式：设备无 adapter 时只对 real-GPU 测试输出明确 skip，设备无关的单元测试和 WGSL gate 仍必须通过。第二条命令是硬件 CI 门禁；它必须在真实 integrated/discrete GPU 上运行，不能用 software adapter 冒充通过。本机当前没有可用的真实 adapter，因此只能确认无 GPU 行为，不能宣称 GPU parity 已通过。
