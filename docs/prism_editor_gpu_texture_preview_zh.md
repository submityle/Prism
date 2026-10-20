# 编辑器真实 GPU 纹理预览

更新日期：2026-10-21。实现位于 `prism_editor_profiler::gpu_preview`，由 `gpu` feature 启用。它读取应用实际 wgpu 纹理，不从资源描述或 pass 名称生成预览图。

## 捕获与身份

宿主用 `TexturePreviewBinding::new` 选择权威 `GraphCaptureReport` 中的 pass/resource；不存在的访问被拒绝。生产器拥有 command encoder，宿主先记录真实 pass，再调用 `copy_texture`，之后可继续记录覆盖同一纹理的 pass。复制位于同一命令流中的确切位置，所以后续覆盖不会改变已捕获像素。

生产器验证真实纹理的格式、完整尺寸、维度、mip 数和采样数与捕获描述相符。直接使用此 API 时，宿主仍须从实际执行器的资源表选择正确纹理，并在选定 pass 后调用复制；相同描述不能证明两个纹理是同一资源。Fabric 的生产录制接入见下一节。

结果保留原始报告的 `Arc` 身份、pass/resource、mip/layer、格式和尺寸，全部只读。相同 frame 数字的另一份报告不能通过 `matches`。每个 handle 在进程内唯一，跨生产器、已消费或已取消的 handle 被拒绝。

## 生产帧图录制接入

`runtime-preview` feature 启用 `runtime_preview::RuntimePreviewFrame`。`record` 接收实际 `RuntimeFeatureGraph` 和其 `WgpuDevice`/`WgpuQueue`，先完成权威捕获验证，再执行一次真实 graph recording。报告、SSA、资源映射和待提交命令来自同一个 prepared frame；外部纹理从精确 import ID 解析，帧内纹理从实际 graph allocation table 解析，调用者不再提供一个仅描述相同的替代纹理。映射处理 Fabric 原始 buffer/texture 混排和底层纹理优先的归一化顺序。

使用当前帧 `binding(pass, resource)` 选择资源后，`submit` 按真实执行计划中的 pass 位置移动拆分 command passes。现有 wgpu bridge 分别编码 prefix/suffix，以一次 native queue submission 提交 `prefix → texture copy → suffix`；不复制命令、不重跑 recorder、不在 CPU 等待完成。不能匹配当前报告、缺少执行位置/物理资源、格式/子资源不受支持或预算不足时，返回原本未提交的 frame，可重试或丢弃，没有部分帧提交。

`RuntimePreviewSubmission::poll` 非阻塞返回一次真实结果，`is_complete` 独立表示整个后半帧已完成并释放图内分配。取消只抑制像素发布，完整图仍会执行。队列完成回调独立持有资源所有权，因此丢弃 UI handle 也不会提前回收 GPU 正在使用的 ID。未提交 frame 的 drop、编码异常及完成回调都有 RAII 资源释放。Persistent/history 和其它 imports 继续由原宿主管理；Fabric 要求它们有实际外部 binding，这条路径不创建第二份持久资源所有者。宿主必须继续 poll 同一设备以驱动映射和完成回调。

每个 frame 拥有独立生产器；宿主还须限制跨帧保留的 frame/submission 数量及 staging 总量，不能通过反复创建 frame 绕过应用总预算。这是可选的诊断录制入口；引擎需在其实际提交入口使用它，原生窗口的所有生产 pass 尚未自动改道到此 API。

`RuntimePreviewSubmission` 公开实际复制的 `mip()`/`layer()` 与仍占用的 `occupied_slots()`/`allocated_bytes()`；`TexturePreviewProducer::limits()` 返回真实 admission 配置。跨帧 admission 可以据此核对实际 subresource 与 padded staging 数量，取消后的额度仍须等待读回排空后回收。

## 像素与显示

支持单采样 D2 及数组层的 RGBA8/BGRA8 unorm/sRGB、RGBA16Float、RGBA32Float、R32Float、R32Uint、Depth32Float；要求真实 `COPY_SRC`。压缩格式、多采样和其它格式显式报错。按所选 mip/layer 复制，去除 256 字节 staging 行填充后发布原始字节。

`pixel` 返回实际数值，sRGB 颜色转为线性光；f64 输出保留全部 u32 ID 和源 f32 的精度。半浮点转换使用整数位规范化，避免大图逐通道求幂。`rgba8` 用显式通道、数值范围、曝光和可选 Reinhard 转换产生 sRGB straight-alpha 图像。NaN/Inf 显示为洋红诊断；非法曝光、通道、非有限或溢出的范围被拒绝。它是诊断显示转换，不能代替完整编辑器 HDR 输出或 OCIO 流程。

## 预算与取消

同时限制占用槽位、每张纹理的填充字节、合计填充字节和像素数，并检查设备 `max_buffer_size`。所有检查先于 GPU buffer 创建与 copy 录制。默认 3 槽、每图 64 MiB、合计 128 MiB、4M 像素；配置硬上限分别为 8 槽、256 MiB、512 MiB 和 16M 像素。CPU 原始输出由同一尺寸约束；显示输出每像素 4 字节。调用者须为已取走并保留的结果另设缓存预算。

取消未提交录制立即释放 encoder 和 staging；取消已提交读回只隔离结果，继续占用槽位和字节额度，直到非阻塞 `poll` 观察到 map 回调并 unmap。取消不会提前复用 GPU 仍在访问的资源。完成和失败均保留 handle，错误不返回替代图。

## 验证

`cargo test --offline -p prism_editor_profiler --features gpu-tests,render-graph --test gpu_texture_preview -- --nocapture` 必须获得真实非 CPU GPU，否则失败，不能静默跳过。2026-10-21 在沙盒外 Apple M2 / Metal 通过：真实 render pass 后复制再覆盖、非对齐行、HDR/depth、半浮点次正规/边界值、BGRA/sRGB、数组层/mip、精确报告身份、过期 handle、取消 quarantine、整数 ID 精度、非有限像素和像素/单图/总量预算拒绝。

同日执行 `cargo test --offline -p prism_editor_profiler --features runtime-preview,gpu-tests --all-targets -- --nocapture`，26 项通过、零失败/零跳过，包括真实 timestamp、纹理和 Fabric 录制硬件测试。新增硬件验收通过真实两次 buffer→texture transfer 验证红色/绿色两个 pass 时点、buffer/texture 原始顺序、transient/imported/persistent import、每 recorder 恰执行一次、独立报告拒绝、预算拒绝后同帧重试、取消、丢弃 pending 和丢弃未提交录制。原生窗口显示及引擎全量入口仍须分别验收。

同日 session 79648 在沙盒外以相同命令重跑，新增 subresource 验收后共 27 项通过、零失败/零忽略。真实 Fabric 分配和 imported 纹理均为 6×4×2、2 mips；两个 transfer pass 仅写 mip 1/layer 1，分别捕获 3×2 红色/绿色真实像素。mip 2 或 layer 2 的拒绝返回原始未提交帧，随后同帧重试成功，recorder 总计仍只调用三次。submission 与结果均保留 mip/layer=1；512 字节 padded staging 和一个槽位在完成消费后归零。日志为 `/private/tmp/prism-editor-runtime-subresources-gpu.log`。

同日 `cargo clippy --offline -p prism_editor_profiler --all-targets --all-features --no-deps -- -D warnings` 通过。新增 getter/subresource 回归后，session 46011 再次以 profiler/timeline/world 联合命令和相同 targets/features/strict 参数通过。该范围不包含依赖包的 lint；依赖仍有既有告警。
