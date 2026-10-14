# Response GPU 合成验证

验证日期：2026-10-14。

`prism_render_scene::response_gpu::ResponseCompositionPipelines` 创建并复用生产 compute 管线。
GPU ABI 为 64 字节的五路 RGB + alpha、16 字节的控制项，以及独立 64 字节的类型化辅助项。
名称保留在 CPU；GPU 不进行字符串查找。合成覆盖 left，使用两份辐射平面，不增加第三份结果平面。
未选择的辐射通道保持 left；alpha 采用显式规则，整数辅助值不插值。

上传准备共享 CPU 的输入校验，不克隆两份 ResponseOutput、不求一次无用的 CPU 辐射合成。
辅助字段规则在准备时校验，GPU 仅接收紧凑定宽标签和载荷。

## 实测

命令：`PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --test response_composition_gpu -- --nocapture`。
在 Apple M2 / Metal 上 2 项测试通过，无跳过。
覆盖全部 31 个非空自定义辐射掩码、FullRadiance、DirectOnly，5 种 alpha 规则和 8 个遮罩值：
1,320 项辐射输出、14,520 项辅助作业与权威 CPU 合成逐项对照。
同时覆盖非法输入、辅助输出计数与类型错误的拒绝行为。

该批次两份辐射平面共 168,960 字节，最新复验提交到读回的主机墙钟为 4.025083 ms。
这个值包含驱动提交、同步、映射和读回，不是 GPU timestamp，也不是逐帧性能或完整场景基准。
单独的内核对照证明合成语义与 ABI；完整运行链路的证据记录在帧图实现文档。
