# 帧图声明校验、版本身份与顺序一致性

日期：2026-10-16。以当前源码为接口依据；分开记录各范围已执行结果、在途验证与生产集成边界。

## 三种声明输入，一份权威编译结果

`prism_render_graph/src/frame_graph/builder.rs` 的声明 frontend 委托既有 RenderGraph/compile 内核，不另建调度算法。

- `compile()` 先按显式依赖确定 lowering 顺序，再生成隐式输入版本与 hazard；合法前向 pass 引用不因声明数组顺序误判成资源环。
- `compile_with_inputs(&[Vec<u32>])` 为每条访问给定输入版本。行列使用原始 PassId 和声明 access ordinal，不是执行位置或 texture-first normalized ordinal；版本必须在该访问 lower 时已经存在。
- `compile_with_sources(&[Vec<AccessInput>])` 支持 `Version(n)` 和 `ProducedBy { pass, access }`。producer 必须是另一 pass 对同一资源的确切写访问，其前置关系补入共享控制图。
- `dependency_order()` 只取得共享内核的控制顺序，不 lower 资源、不生成 SSA/hazards/barriers/alias；不能把它记作另一轮完整资源编译。

两种显式入口都允许保留旧输入；以旧版本为输入写入仍由 checked builder 分配下一输出版本，调用者不能指定输出版本。权威结果的 `CompiledAccess` 分别保存 input/output；RAW/WAR/WAW、barriers、queue waits 和最终顺序均从同一内核派生。无依赖约束时使用稳定破平局。

## 分层校验及身份保持

| 层 | 校验与边界 |
|---|---|
| Graph declaration admission | pass/resource/access/dependency 容量在插表前检查；compile 检查资源/依赖引用、循环、输入形状与版本；symbolic producer 错误带原始 pass/access/producer 身份 |
| Fabric Catalog | 检查全部 variant；拒绝同 pass 重复写、两个 Initial 写分叉、相同 ProducedBy 分叉、多 writer 的 Automatic、self/history producer、重复/self control 依赖；显式非分叉链允许 |
| Fabric selected compile | 校验冻结选择来源、选中 producer 是否写同资源、控制前置是否选中、transient 读来源及跨 feature 写链；把 ResourceInput 转为 authority inputs，不猜输出版本 |
| Fabric runtime projection | 以语义资源、texture/buffer namespace、原始 pass/access ordinal 与输入输出版本比较声明/执行计划；barrier 的源/目标及重复访问 multiplicity 均保留 |
| runtime submission owner | prepare 与 GPU record/submit 分离；真实 completion callback 与最后 owner Drop 共同决定回收，值耗尽/失败 wait 不提前分配 GPU 资源或丢失 token |
| Scene execution registration | 精确声明、phase/group/owner/liveness/连续性准入；post executor 按位置比较 `system_type()`，debug 名仅作诊断，不以同名替代真实函数绑定 |

`PassTemplate.depends_on` 与资源 producer 前置共同保留到最终声明及可执行图，不能以“最终 depends_on 为空”描述当前实现。control order 也不能把模糊多 writer 自动变成合法写链。

Fabric 合法链例如 `A: Initial → B: ProducedBy(A) → C: ProducedBy(B)`；任意 read 可明确引用中间输出。历史 `FrameLag(n)` 使用 Persistent 的独立绝对帧 Imported 资源、SSA 0，不作为同帧 ProducedBy。content version 为外部 base 加权威 SSA，经过容量与溢出检查。

Scene 的 production 顺序为 resolve→water→response composition→post；相关 post 子链是 SSR→GI/denoise→SSGI→TAA。资源 access、producer 与实际 executor 均受检，而不是只校验注释或表面排序。实际绑定证据与启动顺序见 `docs/frame_graph_implementation_zh.md`。

## prepare/record 与 completion 的验证边界

`RenderGraph::prepare` 编译一次并冻结 `PreparedGraph`，不创建 GPU 资源、不运行执行 callback；`PreparedGraph::record` 复用计划，产生持有 commands、plan、allocations 的 `RecordedGraph`，不重新编译或 submit。裸 allocation owner 必须在实际完成后显式 release；其 Drop 不销毁资源，imports 不属于它。

runtime `execute_owned` 返回 `FrameSubmission`，由实际提交队列的 completion callback 保留一个 clone。调用方 early drop 不提前销毁；callback 确认完成且最后 owner Drop 后，才释放自有资源。显式 clone 可保留跨 completion 的 history，发布方仍须覆盖所有后续 imported 使用。owner 固定创建 device 与 `device_generation()`，旧代次 ID/迟到 callback 不迁移到恢复后的设备；callback 丢弃或失败 wait 不是完成证明。

同一设备代次允许发放一次 `u64::MAX`，之后 `CompletionValueExhausted`，不环绕、不复用。耗尽在 pacing dequeue/wait 前拒绝；compile 失败在触碰 pacer 前返回；pacing wait 失败恢复最老 token 到队首，保持在途顺序/数量及提交值。上述拒绝均不创建 GPU 资源、不运行录制 callback、不提交，不代表编译过程没有 CPU 分配。`wait_idle` 仍走真实 queue-wide barrier，若该 barrier 失败，按原顺序保留未确认 token 供重试；若成功则覆盖此前单 token 未确认状态。

## 可定位回归及其断言范围

`pkg/prism_render_graph/text/unit/compiler.rs` 中的回归包括：

- `declaration_versions_follow_dependencies_before_deriving_hazards`：显式依赖先于隐式版本推导。
- `declaration_inputs_retained_read_and_stale_write`：旧输入读写及 authority 输出。
- `declaration_inputs_reject_shape_mismatches_with_original_pass_ids`：原始 ID 的形状诊断。
- `declaration_sources_use_exact_write_access_and_keep_older_inputs`：symbolic 精确写访问及旧来源保持。
- `declaration_sources_reject_invalid_producer_identities`：错误 pass/access/resource/write 身份。
- `declaration_sources_and_control_edges_share_original_cycle_diagnostics`：symbolic/control 共用循环诊断。

`render_fabric/tests.rs` 包含上述 Catalog 正负例，并在 active、能力/质量不合格、root 闭包外状态下检查静态规则；`compile_tests.rs` 包含 `explicit_writer_chain_keeps_initial_and_intermediate_read_provenance`、`symbolic_mixed_writer_chain_preserves_admission_ids_and_numeric_plan` 等版本/计划回归。列出测试表明源码断言范围，执行结果另见下表，不从单个测试名推断全目标完成。

runtime `retirement_tests.rs` 的 13 项 controlled 回归覆盖 early drop、clone、runtime teardown、callback 丢弃、失败 wait、代次隔离、溢出和编译失败。具体拒绝路径由 `failed_pacing_waits_restore_the_oldest_token_before_allocation_record_or_submit`、`failed_idle_wait_keeps_unconfirmed_tokens_in_original_order_for_retry`、`completion_values_never_wrap_or_reuse_the_final_value`、`compile_failure_has_no_retirement_token_allocation_recording_or_submission` 断言；它们不是物理 GPU 测试。

`submission_gpu_tests.rs` 的 `execute_owned_gpu_clone_preserves_buffers_through_completion_and_readback` 与 `execute_owned_gpu_early_drop_completes_copies_and_retires_callback_resources` 在同一真实设备执行 upload → graph-owned Transient → graph-owned Persistent → readback。非零 payload、callback、保留 clone/early drop 与最后 owner 释放后 backend 槽位复用及 generation 加一均有检查；只证明 transfer/readback/owner 表项回收，不证明驱动 heap 归还或完整 Scene 帧。`PRISM_REQUIRE_GPU=1` 下无硬件不能 skip；完成前的确定性保活另由 controlled 测试验证。

## 验证检查点与生产边界

以下为主线已执行并反馈的结果；本次仅核对源码并编辑文档，未运行 cargo。

| 范围 | 执行结果与边界 |
|---|---|
| graph `--lib --all-features` | 最新重跑 78 passed，零 warning |
| architecture Render Fabric | 本轮 122 passed / 0 failed |
| runtime `PRISM_REQUIRE_GPU=1`、`--lib -- --test-threads=1` | 32 passed / 0 failed，含 13 项 controlled retirement 与 2 项真实 GPU 测试 |
| runtime 两项 GPU `--nocapture` 复跑 | 2 passed；adapter 输出 Apple M2 / Metal / Hardware |
| Scene `specular_` / plugin / Apple M2 startup | 较早检查点分别为 14 / 13 / 1 通过 |
| `cargo check -p prism_render_app --all-targets` 与 owner caller 回归 | check 已通过，有既存 warnings；修复 3 处新增 `RecordedGraph` unused 后，`mesh_draw_records_into_a_graph_pass` 1 项及 `records_a_multi_draw` 2 项均通过 |
| gpudriven 严格 GPU `--lib -- --nocapture --test-threads=1` | 18 passed；7 条路径输出 Apple M2 / Metal 硬件证据 |

旧批次不是当前总数，不将 Graph 的零 warning 外推至其他范围或全 workspace。app 三项 caller 回归使用仅捕获命令、不提交 GPU 的 CapturingQueue，因此录制后立即 release；不将其描述为 GPU 验证。gpudriven 记录 18 项通过与 7 条路径的硬件证据，不将全部测试都归为 GPU 路径。真实 startup 不代表完整设备帧；runtime 的两项 GPU 结果也不代表 Scene 整帧、compute dispatch、图像 golden 或性能验收。app/gpudriven 检查点同样不能证明通用 Fabric 的整帧生产闭环。

单内核、版本/依赖校验及 Scene 专用 executor 接入已有实现；通用 Fabric 的 Scene 原生 record/submit/completion/history 整帧生产接线仍未完成。单操作 admission 原子性不等于 setup closure 全事务，逻辑别名/队列元数据不等于物理 heap alias/多队列设备验收。

本次仅修改指定 tracked 文档，不改代码、不改 `docs/refactoring-dev/`、不 cargo、不提交。
