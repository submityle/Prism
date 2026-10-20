# E9 实际 CPU / GPU profiler producer

对应编辑器设计 §9。`prism_editor_profiler` 保留已有 `Frame`、`FrameStore`、帧图、内存与可视日志模型，新增实际采集 producer，不依赖 Bevy。此切片提供 app 可调用的采集边界；app/UI 挂接、Tracy 桥接、时钟校准与分析界面由各自切片负责。

## CPU 接入

默认构建无外部依赖。`capture::CpuProducer::new(ProfilerLimits, max_active_frames)` 建立单调 `Instant` 原点；`begin_frame(FrameId, sequence)` 返回不透明 `CaptureHandle`。帧内 `begin_span(handle, name, lane)` / `end_span(SpanHandle)` 使用真实 `Instant` 样本，按 LIFO 形成包含关系。一个帧对应一个跨度栈；不同线程可使用独立 producer。

```rust
use prism_editor_profiler::{FrameId, FrameStore, ProfilerLimits};
use prism_editor_profiler::capture::CpuProducer;

let limits = ProfilerLimits::default();
let mut cpu = CpuProducer::new(limits, 2)?;
let mut store = FrameStore::new(limits);
let frame = cpu.begin_frame(FrameId::from_raw(1), 1)?;
let span = cpu.begin_span(frame, "prepare render", "cpu/main")?;
// 应用执行实际准备工作。
cpu.end_span(span)?;
store.insert(cpu.finish_frame(frame)?)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`finish_frame` 仅在全部跨度结束后成功，直接返回已有 `Frame`；它不隐式驱逐历史帧。调用方管理有界 `FrameStore`，满时处理 `LimitExceeded`（例如轮换 store）。`cancel` / `cancel_all` 释放未完成 CPU 数据。帧时间戳是 producer 原点后的纳秒，跨度相对帧起点。

## GPU 接入

启用 `gpu` feature。app 请求 device 时必须显式启用 `gpu::REQUIRED_TIMESTAMP_FEATURES`（`TIMESTAMP_QUERY`）；只检查 adapter 宣称支持还不够。`GpuProducer::new(&device, &queue, limits, config)` 要求同设备 queue，缺少 enabled feature 或有效 timestamp period 时返回 `UnsupportedTimestamps`，没有 CPU 时间或人工 timestamp 降级。不要求 M2 Metal 不支持的 `TIMESTAMP_QUERY_INSIDE_ENCODERS`。

渲染循环调用顺序：

1. `begin_frame(id, sequence)` 创建录制代际，返回 `CaptureHandle`。
2. `begin_render_pass(handle, name, &RenderPassDescriptor)` / `begin_compute_pass(handle, name, &ComputePassDescriptor)` 返回实际 wgpu pass，自动注入 pass 起始/结束 query；应用照常设置 pipeline、资源、draw / dispatch，drop pass 时结束。GPU span 为平铺实际 pass 区间，GPU frame 时长为这些 pass 的时间包络，不声称测量排队延迟或整个 CPU 渲染循环。已有 timestamp_writes 的 descriptor 会被明确拒绝，避免覆盖调用方 query。
3. `submit_frame(handle)` 执行 query resolve、copy 到 MAP_READ staging buffer、提交 encoder，并启动 `map_async`。没有任何采集 pass 的空帧被拒绝。返回真正 `wgpu::SubmissionIndex`，应用可据此追踪提交。应用不再对这个 encoder 额外 submit。
4. 后续渲染循环调用非阻塞 `poll()`，返回最多 slot 数量的 `GpuCaptureResult { handle, frame: Result<Frame, CaptureError> }`。将成功帧插入 GPU `FrameStore`，向 UI 显式暴露失败。不会在后台制造无限结果队列。

只需将现有 viewport 的 begin_render_pass 换成 producer 方法，pass 内部 draw 代码保持原调用方式。`encoder(handle)` 可借用同一 encoder 来添加 copy 或未采集工作；这部分工作不单独生成 span。借用遵守 wgpu 独占约束，调用方不可对 pass 使用 `forget_lifetime` 绕过该约束。producer 持有 device/queue 克隆，未另建渲染设备。硬件可能省略没有 draw/dispatch 的空 pass query；全零查询对返回 `InvalidTimestamp`，不把它伪装成有效耗时。

GPU 原始 ticks 乘以 queue 的 `get_timestamp_period()` 转纳秒；跨度端点先相对最早 pass 起始 tick 相减，再分别舍入。反向 tick、溢出与不合法区间会显式报错，不估造时间。GPU `Frame.timestamp_ns` 处于 GPU 时钟域，CPU `Instant` 处于另一时钟域；两个 store 可通过调用方分配的 frame ID/sequence 配对，但未经校准不能声称绝对对齐，GPU 延迟读回也不等同于渲染 CPU 耗时。

## 有界、取消与代际围栏

CPU 未完成帧由 `max_active_frames` 限制，跨度数、深度、文本字节由 `ProfilerLimits` 限制；错误输入在保留字符串前拒绝。GPU `GpuCaptureConfig` 同时限制在录制和在途 slot 总数与每帧 pass 数。每个 slot 持有一个 query set、一个 resolve buffer 和一个 readback buffer；每帧最多 `2 * max_spans_per_frame` 个 query，不能超过 wgpu 4096 query 上限，也不能超过模型跨度上限。每个 buffer 为 query 数乘 8 字节，池满返回 `AtCapacity`，绝不隐式等待/扩容。

Metal 对 query reset 为 no-op，空 pass 可能遗留旧 sample；因此每个新 capture generation 更换 query set，只有旧回调排空后才复用 slot/buffer。这个硬件回归已有实际测试，保证空 pass 不会把上一帧的真实 timestamp 冒充本帧。query set 不形成无界历史集合，最多固定 slot 数量的活跃集合（替换瞬间多一个）。

每次 `begin_frame` 分配进程内不重复的 generation handle，耗尽返回错误；跨度 handle 也绑定相应 generation。旧帧、旧跨度、跨 producer 与已消费句柄都被拒绝。recording encoder 由 producer 独占，录制阶段取消会丢弃 encoder，避免调用方保存旧 command buffer 后写入复用 query 资源。

已经提交的取消只标记失效：slot 和容量为 1 的回调 channel 保留到真实 callback 到达。`poll` 排空旧 callback、unmap 后释放 slot，取消帧永不输出。这样即使持续取消也无法绕过在途上限；调用方仍须 poll 以恢复容量。drop 会 unmap 未结束读回，迟到 callback 只持有有限 sender，无访问 producer 的能力。设备丢失/poll 失败显式返回 `GpuPoll`，调用方可销毁重建 producer；不宣称旧任务完成。

## 验证

```text
cargo test -p prism_editor_profiler --no-default-features
PRISM_REQUIRE_GPU=1 cargo test -p prism_editor_profiler --features gpu-tests --test gpu_capture -- --nocapture
cargo clippy -p prism_editor_profiler --all-targets --all-features -- -D warnings
```

`gpu-tests` 为硬件验收 feature，始终要求真实原生 GPU；没有 adapter、缺能力或软件 CPU adapter 都失败，没有 skip 分支。测试实际执行 compute（逐项校验输出）及 render draw，记录 adapter、submission、period 与真实读回帧/跨度耗时，验证非零计时、`FrameStore` 插入、取消前后围栏、slot 复用与空 pass 不读旧 sample、无 feature device 的显式拒绝。不使用人工 timestamp 代替硬件验收。

共享工作区不修改全局 `Cargo.lock` 时，可在专用临时目录复制同一 manifest，链接当前 crate 的 `src` 和 `tests`，用独立 lock/target 执行相同测试；构建配置 `CARGO_BUILD_JOBS=2`、`CARGO_PROFILE_DEV_DEBUG=0`、`CARGO_PROFILE_TEST_DEBUG=0`、`CARGO_INCREMENTAL=0`。只清理该次专用 target。

2026-10-20 实际验收：独立 manifest 与 crate manifest 完全相同，源码/测试链接到本仓库。无默认 feature 的 session `53486` 通过 11 项测试；跨沙盒强制 GPU 的 session `56322` 通过相同 11 项及 1 项硬件测试，adapter 为 Apple M2 / Metal，timestamp period 为 1ns，真实 submission index 为 1。该次 GPU pass 包络为 479375ns，compute 为 228833ns，render draw 为 40125ns；这些单次结果只用于证明真实 query/readback 通路，不是性能基准。计算输出逐项校验通过；取消、跨 producer 句柄、slot 复用、空 pass 代际隔离、cancel_all、在途 drop 和未启用 feature 的拒绝检查全部通过。session `81219` 的 all-targets/all-features Clippy `-D warnings` 通过，rustfmt 与 diff whitespace 检查通过。验收后已清理该次自有 target。
