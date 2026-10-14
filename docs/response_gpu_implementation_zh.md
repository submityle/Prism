# Response GPU 合成验证

验证日期：2026-10-15。

`prism_render_scene::response_gpu::ResponseCompositionPipelines` 创建并复用生产 compute 管线。
GPU ABI 为 64 字节的五路 RGB + alpha、16 字节的控制项，以及独立 64 字节的类型化辅助项。
名称保留在 CPU；GPU 不进行字符串查找。合成覆盖 left，使用两份辐射平面，不增加第三份结果平面。
未选择的辐射通道保持 left；alpha 采用显式规则，整数辅助值不插值。

上传准备共享 CPU 的输入校验，不克隆两份 ResponseOutput、不求一次无用的 CPU 辐射合成。
辅助字段规则在准备时校验，GPU 仅接收紧凑定宽标签和载荷。

## 实测

普通门禁命令：`cargo test -p prism_render_scene --test response_composition_gpu --offline -- --nocapture`。
结果为 **2 passed / 0 failed**：CPU 侧非法输入/packing 测试通过；response GPU parity 测试在 adapter 获取阶段明确 skip。真实 stderr 为：
`No suitable graphics adapter found; noop not requested, vulkan support not compiled in, metal found no adapters, dx12 support not compiled in, gl not requested, webgpu not requested`。

严格门禁命令：`PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test response_composition_gpu --offline -- --nocapture`。
结果为 **1 passed / 1 failed**：同一 CPU packing 测试通过，GPU parity 测试在 adapter 不可用时按设计失败，panic 信息为：
`required GPU adapter unavailable: No suitable graphics adapter found; noop not requested, vulkan support not compiled in, metal found no adapters, dx12 support not compiled in, gl not requested, webgpu not requested`。

因此本机本轮没有可宣称的 Metal/WGPU response readback 或 GPU 对拍结果；默认通过仅表示显式 capability gate 允许跳过，不是伪造的 GPU 通过。测试仍覆盖全部 31 个非空自定义辐射掩码、FullRadiance、DirectOnly、5 种 alpha 规则和 8 个遮罩值的 CPU-side job 构建，以及非法输入/辅助输出计数与类型错误拒绝行为。真实 GPU readback 需在存在合适 adapter 的主机上重新运行严格命令。
