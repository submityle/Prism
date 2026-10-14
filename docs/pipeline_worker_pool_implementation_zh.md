# 03-S4：有界 Pipeline 编译池

记录日期：2026-10-14。范围只限 `pipeline_cache` 内部后台编译调度；不修改 `PipelineKey` 现有字段、构造 API、mesh_pipeline、shader package 冷层、03-S2/S3。

## 后续正式复验（2026-10-14）

恢复任务后重新执行 `cargo test -p prism_render_app --lib`，得到 **146 passed / 0 failed**；其中六项 worker 专项、既有异步背压/fallback、prewarm、direct/bin 与 WGPU worker 测试均报告通过。
`cargo test -p prism_render_app --lib pipeline_cache -- --test-threads=1` 得到 **18 passed / 0 failed**，复核单线程调度下饱和/取消/错误/shutdown 回归。
新增的 canonical-state 测试使 state+worker 补充独立 harness 从原 14 项变为 **15 passed / 0 failed**。
下面「验证」段保留初次 S4 交付时的阻塞历史，不代表当前仍失败。
本次未修改已交付 worker 实现或 `PipelineKey` API；没有对持久化 PSO 导入或 GPU 组合 pass 作完成声明。

## 实际接线与 API

`PipelineCompileConfig { worker_count, queue_capacity }` 显式控制并行编译数和额外等待描述符数；默认 4/64。`worker_count == 0` 和总容量算术溢出直接拒绝。新增 `PipelineCache::with_compiler_config(compiler, config)`、`with_worker_config(device, config)`；既有 `with_compiler` / `with_worker` 使用默认配置，`PipelineKey::new` / `from_descriptor` 不变。

`PipelineCache::compiling_len()` 返回当前执行中的编译调用数；`in_flight_len()` 返回已交 worker 的任务数（包括尚未 pump 的完成结果）；`waiting_len()` 返回本地有界 FIFO；`pending_len()` 为全部已接纳且未完成请求数；`compile_config()` 返回配置。总接纳上限是 `worker_count + queue_capacity`。worker 共用有界 `sync_channel`，receive 锁只覆盖 `recv`，编译本身可并行；结果也使用有界 channel。派发使用 `try_send`，render thread 不会等待满队列。

满载时 `resolve_or_request` 仍立即返回正确 fallback，不保留新 key/descriptor；下一次相同 draw 请求重新尝试。`prewarm` 走同一入口，因此超量预热项也需要调用方以后再次提交，而非保证首次 batch 全部完成。等待中的同 key 请求不会重复加入；release 移除等待项，clear 保留已经派发的占用直至 pump 清理，取消后再次请求可复用尚在编译的同 key 工作。同步 get_or_create 可越过后台任务，晚到结果被销毁，等待项直接移除。

RHI `PipelineCompiler::compile` 仍返回 ID 而不是 `Result`；池用 `catch_unwind` 把编译 panic 记为失败、释放 slot 并保持 worker 可用。失败 key 继续使用 fallback；既有同步路径可清除失败并重试。`shutdown(device)` 先关闭请求、持续 drain 结果直到 worker 退出，再销毁结果及 cache 对象；避免有界结果 channel 满时 join 死锁。单纯 Drop 会先断开结果接收端再 join，保证不死锁，但 Drop 无设备引用，无法对尚未 pump 的 GPU 对象作确定性销毁；有后台工作的显式 teardown 应调用 `shutdown(device)`。若外部编译器永不返回，任何同步 shutdown/Drop 都只能等待，无法强制取消后端调用。

## 验证

独立 harness 直接编译工作区最新 `pipeline_cache.rs`、`pipeline_state.rs` 与各自内联测试，链接现有 `target/debug/deps` 依赖产物：**14 passed / 0 failed**。其中本轮六项 worker 回归覆盖：2 worker 真并行与计数、1 槽等待和饱和非阻塞/重试、key 去重、编译 panic 隔离/后续 worker 复用、满结果 channel 的 shutdown 和 Drop、取消/clear 后复用、同步抢先编译与等待项移除、零 worker 配置拒绝。既有 S1 的 8 项也在同一次运行通过。harness 路径 `/private/tmp/prism-pipeline-state-s1.0cq7ed/pipeline-worker-tests`；它并非全工作区重新编译。

按要求实际尝试 `cargo test -p prism_render_app --lib pipeline_state` 和 `cargo test -p prism_render_app --lib`，分别在并行改动的 `prism_render_graph/src/compile/schedule.rs:152` E0308 与 `prism_render_architecture/src/render_fabric/topology.rs:58` E0004 阶段阻断；随后再尝试 `cargo test -p prism_render_app --lib pipeline`，阻断点变为 `prism_render_graph/src/graph.rs:218` 的 E0061（调用 `PassBuilder::new` 少 `GraphLimits`）。这些文件本轮未修改；故不能声称 app lib / wgpu 全包测试通过。`rustfmt` 及本轮缓存文件 `git diff --check` 通过。并行主线修复后应重跑 `cargo test -p prism_render_app --lib`，重点复核既有异步 fallback、5 项背压、prewarm 与 WGPU worker 测试。

本轮无 commit/push，未修改 `docs/refactoring-dev/`。
