# Response GPU 合成验证

验证日期：2026-10-15。

`prism_render_scene::response_gpu::ResponseCompositionPipelines` 创建并复用生产 compute 管线。
GPU ABI 为 64 字节的五路 RGB + alpha、16 字节的控制项，以及独立 64 字节的类型化辅助项。
名称保留在 CPU；GPU 不进行字符串查找。合成覆盖 left，使用两份辐射平面，不增加第三份结果平面。
未选择的辐射通道保持 left；alpha 采用显式规则，整数辅助值不插值。

上传准备共享 CPU 的输入校验，不克隆两份 ResponseOutput、不求一次无用的 CPU 辐射合成。
辅助字段规则在准备时校验，GPU 仅接收紧凑定宽标签和载荷。

## 实测

跨沙盒严格门禁命令：`PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test response_composition_gpu --offline -- --nocapture`。
结果为 **2 passed / 0 failed**：CPU packing 与 production GPU parity 均通过；真实 adapter 为 Apple M2、backend 为 Metal，GPU readback 覆盖 1320 cases / 14520 auxiliary jobs，提交与读回耗时约 2.75 ms。

因此本轮已有可宣称的 Metal response readback 与 GPU 对拍结果。测试仍覆盖全部 31 个非空自定义辐射掩码、FullRadiance、DirectOnly、5 种 alpha 规则和 8 个遮罩值的 CPU-side job 构建，以及非法输入/辅助输出计数与类型错误拒绝行为。水体 Hybrid 的独立严格门禁也已通过：当前 raster 使用 7 个 bind groups，Apple M2 Metal 的 `max_bind_groups=8` 足够；两类结果仍分别记录，不混写。
