# 原生编辑器图像 GPU 合成

对应编辑器设计 §11 的实际缩略图显示边界。内容 worker 已产出真实 PNG 像素，本模块负责将这些像素上传并叠加到宿主的同一张渲染目标。它不创建窗口、扫描目录或等待缩略图 worker；原生内容网格、布局、请求调度与实际应用挂接由 UI/app 完成。

## 调用

公开模块为 `prism_editor_viewport::image`。使用宿主现有的 wgpu 30 device/queue 创建 `ImageRenderer::new(device, queue, format, limits)`；不能混用其他 device 的 queue、encoder 或 texture。允许单样本 RGBA/BGRA8 的线性与 sRGB attachment、RGBA16Float 与 RGB10A2。后两者仅表示可合成到该格式，不表示 HDR 显示映射或色域管理已完成。

1. 每个资产修订完成解码后，调用 `upload(width, height, rgba8)` 一次，获取不透明 `ImageHandle`。输入是紧密排列、直通透明度的 sRGB RGBA8；调用后可释放 CPU 像素。
2. 根据 Loom 的可见图像节点布局及 DPI，形成 `ImageQuad`：物理像素的 left/top/width/height、同坐标系 clip 和 0..=1 opacity。按绘制顺序调用 `prepare([target_width, target_height], quads)`；布局/可见性没有变化时保留上一份准备结果即可。
3. 宿主完成底层背景/面板绘制后，调用 `encode(&mut encoder, &texture)`，最后由宿主统一 submit。合成使用 Load 保留已有像素，不提交、不读回、不分配全尺寸中间 attachment。宿主负责使其他需要遮盖图像的前景在其后绘制。
4. 资产修订改变或缓存淘汰时 `remove(handle)`；项目关闭/设备重建时 `clear()`。重新创建 device 后应重新创建 renderer 和上传图片。

encode 接收 texture 而不是任意 view，并内部构造 mip 0 的单层视图，避免 wgpu 30 无法查询 view descriptor 而误接受其他 mip/格式。渲染目标尺寸、格式、维度、层数、样本数和 RENDER_ATTACHMENT usage 均检查；不符合要求则返回 TargetMismatch。宿主提供匹配 device 是资源契约，不能通过这些元数据推断其他 device 的归属。

## 图像质量

GPU 纹理为 Rgba8UnormSrgb，textureLoad 在采样时解码传递函数。shader 对四个样本先在**线性光**中乘透明度，再双线性插值，最后应用整体 opacity，并以预乘混合叠加。透明像素中隐藏的 RGB 不会泄漏成彩边；写入 sRGB attachment 时由 GPU 编码，写入线性 attachment 时保持线性值。clip 根据物理 fragment 中心判断，保留分数像素边界，不改变源 UV。

缩小到目标尺寸的面积滤波由 `ThumbnailService` 完成；本层负责位置/小范围尺寸变化的双线性采样，没有生成 mip 链。宿主应按显示尺寸请求缩略图，避免把大源图反复强缩小后声称具有面积滤波质量。输入方向保持左上为原点。

## 预算、缓存与失败

默认最多 512 张图、64 MiB 逻辑 RGBA8 像素、单边 512、每帧 512 个输入 quad；GPU 对齐、driver 分配和已提交命令持有的旧资源不计入逻辑字节。计数/字节/设备纹理边长及整数乘法溢出在分配前拒绝。每个 renderer 的 handle 带私有 owner 身份，删除、清空及其他 renderer 的 handle 均失效，ID 不复用。

geometry 使用一个保留顶点缓冲，按几何增长且不超过 quad/设备预算；相同顶点数据不再次 queue upload，`geometry_uploads()` 可观察。prepare 先验证整批，再上传；非法 geometry、超预算或过期 handle 保留上一成功帧。移除成功帧引用的图片会撤销该帧。零尺寸/全透明 quad 不形成 draw。GPU 资源创建/上传捕获 validation、internal 和 OOM 错误。

queue 写入不是未提交 encoder 的独立快照：必须先提交已编码帧，再改变 prepare。多个帧可以按正常队列顺序在途；此接口不声称可以缓存旧 command buffer 后反复改变保留 buffer。提交后 GPU 资源生命周期由 wgpu 持有，clear/remove 不等待 GPU。

## 已执行验证

2026-10-20 在 Apple M2 / Metal 上，`PRISM_REQUIRE_GPU=1 cargo test --offline -p prism_editor_viewport --all-features` 通过 42 项测试，含新增 5 项图像硬件测试与原有拾取/geometry 回归，无跳过。沙盒内无法访问 Metal adapter 的失败不计入覆盖，成功结果来自授权的沙盒外执行。

图像测试实际提交 draw 后只在测试端读回像素，覆盖：源方向、sRGB→线性和 sRGB 往返、透明颜色无彩边、线性插值中点、绘制顺序、opacity、分数 clip、背景保留、空帧、resize、布局不变零再次上传、非法批次保留旧帧、全部预算、u32 极值拒绝、移除/clear/跨 renderer 围栏、错误目标与全部六种 attachment 格式。生产绘制路径无读回。相应 strict Clippy、包内格式及定向 diff 检查随提交执行。
