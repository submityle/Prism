# Prism 实际 frame graph 采集与诊断回放

本切片位于 prism_editor_profiler，无 Bevy 依赖。默认提供有界报告、二进制序列化与诊断回放；可选 render-graph feature 直接依赖当前 prism_render_architecture 和 prism_render_graph。CPU/GPU producer 的原有 features 保持独立。

## 权威来源与接入点

prism_render_architecture::frame_graph 是 prism_render_graph 的兼容导出；SSA、hazard、调度、别名计算只有后者一个权威实现。适配器没有重新编译图或另写 hazard solver。

两个入口均已从 profiler 根导出（仅 render-graph feature）：

~~~rust,ignore
pub fn capture_feature_graph(
    graph: &prism_render_architecture::render_fabric::CompiledFeatureGraph,
    inputs: GraphCaptureInputs<'_>,
    limits: GraphCaptureLimits,
) -> Result<GraphCaptureReport, ProfilerError>;

pub fn capture_runtime_graph(
    graph: &prism_render_architecture::render_fabric::RuntimeFeatureGraph,
    inputs: GraphCaptureInputs<'_>,
    limits: GraphCaptureLimits,
) -> Result<GraphCaptureReport, ProfilerError>;
~~~

renderer/app 所需的外部 hook 是：在真实 CompiledFeatureGraph::prepare_runtime 得到 RuntimeFeatureGraph 后、其 record/execute/into_prepared 消费前，调用 capture_runtime_graph(&runtime, inputs, limits)，将返回的报告交给 editor。仅借用，不执行 recorder、不分配 GPU 对象、不提交队列。消费该 runtime 后无法再借用，调用方应事先保留报告。

本切片只实现 adapter 和生产构图测试，尚未安装 renderer/app/native UI hook。测试构造的是真实 catalog、ResourceTemplates、CompiledFeatureGraph，再实际 prepare_runtime 到生产 RenderGraph；这证明适配路径可用，不能据此声称已采集运行中的编辑器帧。后续 native 挂载无需修改 UI/content/timeline 的共享 root 来使用本 API；消费方的 profiler Cargo 依赖需启用 features = ["render-graph"]。

## 报告契约

frame 来自 compiled graph；source 分别为 Declaration 和 PreparedRuntime。passes/resources 保持原始表顺序，索引就是报告 ID；order 保存实际编译执行顺序。每个 pass 保存语义 PassKey、FeatureId、VariantId、名称、queue、声明顺序 accesses；每个 access 保存真实 input/output SSA version 和 read/write 类型。

dependencies 保留显式 depends_on 及编译器 logical_barriers 两种来源，按完整字段排序去重。一个 pass 对可能同时存在显式边及若干资源 hazard；replay 到旧 FrameGraph 时只合并相同 pass 对。没有资源读写的显式边也被保留。适配器把编译器 texture-first access ordinal 映射回声明顺序，并通过实际 compiled accesses 建立 resource namespace 映射；不把 imported-first 的 runtime 局部索引误当全局 ResourceId。

资源记录 ResourceKey owner/slot、绑定 frame、descriptor name/lifetime/size/alignment、buffer usage/bytes 或 texture extent/dimension/format/mips/samples/usage。格式为驱动枚举诊断名称，不反解成可执行 GPU 类型。报告不保留物理 GPU handle，不承担资源或命令回放。

logical_bytes 是 descriptor 逻辑字节数的 checked sum；alias_heap_bytes 是该编译产品的临时别名堆地址范围（含对齐空隙）。Declaration 使用拓扑前端的 footprint，PreparedRuntime 使用实际 backend plan footprint，两者允许不同。它们都不能解释为实测驻留显存或驱动分配量。

GraphCaptureInputs 可携带 FrameBudgetLedger，保存其 upload/readback/relocation bytes 和 acceleration_structure_builds 四项 consumed/limit；另可传 alias_heap_budget。类别稳定排序，consumed > limit 才越界；0 预算有效。ledger 自身采用饱和计数，报告如实保存该快照，不能恢复已饱和前的精确消费量。调用方需在同一帧采样 ledger。

## 时钟与关键路径

timings 是以原始 pass index 标识的 GraphPassTiming，统一指定 Cpu 或 Gpu 时钟来源；重复/越界索引、存在 timing 却无来源均拒绝。调用方必须将同一帧的真实测量匹配到正确 pass。GPU 异步读回未完成时保持 duration_ns = None；adapter 不制造 timestamp。

critical_path 仅在非空图每个 pass 都有实测时长时返回 Some(CriticalPath)；不完整或空图返回 None。它是依赖 DAG 的最大加权路径，稳定 tie-breaker 使用既有 FrameGraph；算术溢出报错。它不包含未采集的队列等待，也不等于帧墙钟耗时。CPU/GPU 时长不可混加。

replay(limits) 生成现有 FrameGraph，用于 topology UI；未测量节点的显示权重暂用零，调用者应以报告的 duration_ns/critical_path 判断测量状态。不要把 replay().critical_path() 的零权重解释为真实 GPU 时间。报告、各模型以及 capture 函数均从 profiler 根或 graph_capture 模块导出。

## 有界、确定性与格式

GraphCaptureLimits 独立限制 passes/resources/总 accesses/dependencies/budgets/单文本字节/wire 总字节。采集先检查表规模和文本，成功返回前检查完整编码大小；错误不返回部分报告、不修改源图。默认限制为 4096/8192/65536/65536/32/4096/8MiB。报告不自带无界历史，app 应用自己的帧数/总字节保留策略。

to_bytes 采用两遍编码：先计算大小和检查 wire 上限，再分配精确输出。V1 header 为 PRFG + 版本字节 1 + 三个零字节；整数均 little-endian u64，UTF-8 字符串长度前缀，不含平台地址、HashMap 遍历顺序或采集墙钟。标识、枚举和布尔窄化均检查。from_bytes 在分配集合前同时检查声明长度、配置上限及剩余输入；严格拒绝截断、尾随字节、未知版本/tag、非法 UTF-8、非规范排序、重复 ID/边/预算、非法引用、循环和 checked 算术溢出。诊断名称保留原字符串，相同报告产生完全相同字节。

回放是诊断模型重建：验证并解码 GraphCaptureReport，再 replay/critical_path/预算判断。没有 shader/资源内容/driver handles，因此不能恢复 GPU 命令执行，亦不宣称 GPU frame replay。

## 验证

~~~text
cargo test -p prism_editor_profiler --features render-graph
cargo clippy -p prism_editor_profiler --no-deps --all-targets --features render-graph -- -D warnings
~~~

实际 workspace 验证（离线、单 job、debug=0、incremental=0）：17 项单元测试及 6 项真实构图集成测试通过；上述限定 profiler 自身的严格 Clippy 通过。包含依赖的严格 Clippy 被范围外 prism_math/src/ops.rs:1 的 doc_markdown 警告阻断，本切片未修改该依赖。

真实生产构图测试覆盖混合 buffer/texture、不同 queue 的 barrier、SSA、显式无资源依赖、资源描述、CPU Instant 测量及预算；恶意输入回归覆盖截断/位翻转/超界长度、非法引用/枚举、重复/循环、计时缺失与溢出。GPU producer 的物理 GPU query/readback 回归由独立 gpu-tests feature 负责；本任务的图构造测试不宣称提交过 GPU 命令。
