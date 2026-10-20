# 编辑器真实 GPU 纹理预览

更新日期：2026-10-21。实现位于 `prism_editor_profiler::gpu_preview`，由 `gpu` feature 启用。它读取应用实际 wgpu 纹理，不从资源描述或 pass 名称生成预览图。

## 捕获与身份

宿主用 `TexturePreviewBinding::new` 选择权威 `GraphCaptureReport` 中的 pass/resource；不存在的访问被拒绝。生产器拥有 command encoder，宿主先记录真实 pass，再调用 `copy_texture`，之后可继续记录覆盖同一纹理的 pass。复制位于同一命令流中的确切位置，所以后续覆盖不会改变已捕获像素。

生产器验证真实纹理的格式、完整尺寸、维度、mip 数和采样数与捕获描述相符。宿主仍须从实际执行器的资源表选择正确纹理，并在选定 pass 后调用复制；相同描述不能证明两个纹理是同一资源。当前 API 是这一接线边界，不能据此宣称引擎所有生产 pass 已自动接入。

结果保留原始报告的 `Arc` 身份、pass/resource、mip/layer、格式和尺寸，全部只读。相同 frame 数字的另一份报告不能通过 `matches`。每个 handle 在进程内唯一，跨生产器、已消费或已取消的 handle 被拒绝。

## 像素与显示

支持单采样 D2 及数组层的 RGBA8/BGRA8 unorm/sRGB、RGBA16Float、RGBA32Float、R32Float、R32Uint、Depth32Float；要求真实 `COPY_SRC`。压缩格式、多采样和其它格式显式报错。按所选 mip/layer 复制，去除 256 字节 staging 行填充后发布原始字节。

`pixel` 返回实际数值，sRGB 颜色转为线性光；f64 输出保留全部 u32 ID 和源 f32 的精度。半浮点转换使用整数位规范化，避免大图逐通道求幂。`rgba8` 用显式通道、数值范围、曝光和可选 Reinhard 转换产生 sRGB straight-alpha 图像。NaN/Inf 显示为洋红诊断；非法曝光、通道、非有限或溢出的范围被拒绝。它是诊断显示转换，不能代替完整编辑器 HDR 输出或 OCIO 流程。

## 预算与取消

同时限制占用槽位、每张纹理的填充字节、合计填充字节和像素数，并检查设备 `max_buffer_size`。所有检查先于 GPU buffer 创建与 copy 录制。默认 3 槽、每图 64 MiB、合计 128 MiB、4M 像素；配置硬上限分别为 8 槽、256 MiB、512 MiB 和 16M 像素。CPU 原始输出由同一尺寸约束；显示输出每像素 4 字节。调用者须为已取走并保留的结果另设缓存预算。

取消未提交录制立即释放 encoder 和 staging；取消已提交读回只隔离结果，继续占用槽位和字节额度，直到非阻塞 `poll` 观察到 map 回调并 unmap。取消不会提前复用 GPU 仍在访问的资源。完成和失败均保留 handle，错误不返回替代图。

## 验证

`cargo test --offline -p prism_editor_profiler --features gpu-tests,render-graph --test gpu_texture_preview -- --nocapture` 必须获得真实非 CPU GPU，否则失败，不能静默跳过。2026-10-21 在沙盒外 Apple M2 / Metal 通过：真实 render pass 后复制再覆盖、非对齐行、HDR/depth、半浮点次正规/边界值、BGRA/sRGB、数组层/mip、精确报告身份、过期 handle、取消 quarantine、整数 ID 精度、非有限像素和像素/单图/总量预算拒绝。`cargo clippy --offline -p prism_editor_profiler --all-targets --all-features --no-deps -- -D warnings` 通过；依赖包仍有不在此检查范围内的告警。测试用真实编译帧图生成权威报告；它不替代生产执行器挂载与原生窗口显示验收。
