# Render Fabric：冻结选择、资源版本链与运行时接入

日期：2026-10-16。工作区：`/Users/wangkailong/Documents/GitHub/Prism`。

本文以当前源码描述接口，以主线已执行的结果记录验证检查点。基础设施、Scene 的专用生产接入与通用 Fabric 的整帧生产接线分别陈述；不再以历史 S1/S2/S3 切片或临时 harness 代表当前状态。

## 当前范围与架构边界

`pkg/prism_render_architecture/src/render_fabric/` 包含 catalog、冻结求解、内建资源模板、PCM 桥、声明编译、payload 与 runtime lowering。`Cargo.toml` 直接依赖 `prism_material_pcm`，`mod.rs` 已公开 PCM 桥，不再是仅靠独立 harness 加载的源码。

- `FeatureId / VariantId / PassKey / ServiceId` 是显式稳定身份；`ResourceKey = (ResourceOwner, slot)`，不由注册顺序发号。
- 访问只声明于 `accesses`；reads/writes 视图由它派生。队列归 pass 所有，写入性质使用权威 `AccessKind::writes()`。
- 只有 Service 发布接口；provider 索引保持唯一。这是“每个服务一个 provider”的约束，**不是每个资源全局只有一个 writer**。
- 静态构建档案冻结根、硬依赖闭包和设备 variant。求解按能力、质量、服务与稳定 ID 确定结果；软需求不扩根、不触发回溯，运行期不做第二次选择。
- 编译时校验冻结选择的完整源声明；复用相同 variant 不能掩盖 feature 种类、硬依赖、服务接口或候选集改变。
- 内建水体、体积、粒子、毛发、贴花模板及 provider 各 variant 的输出写集属于同一资源契约；登记模板本身不等于这些域均已走通用 Fabric 提交整帧。

源码入口：`types.rs`、`catalog.rs`、`selection.rs`、`topology.rs`、`builtins.rs`、`compile.rs`（均相对上述目录）。

## 显式 ResourceInput 链与控制依赖

| 输入 | 当前语义 |
|---|---|
| `Automatic` | read 消费唯一选中 writer；无 writer 时使用外部初始内容；write 从初始内容开始。多 writer 不能靠声明顺序或控制顺序消歧 |
| `Initial` | 明确消费 SSA 0，包括外部历史帧初始内容 |
| `ProducedBy(PassKey)` | 消费另一选中 pass 对同一当前帧资源的输出；可用于 read，也可作为后续 write 的输入 |

合法写链例如 `A: Initial → B: ProducedBy(A) → C: ProducedBy(B)`。读取者可以明确保留 A 或 B 的旧输出，不会被重新绑定为本帧最后一个 writer。每个版本只能有一个后继 writer；两个 Initial 写或两个消费同一 ProducedBy 的写均是分叉，附加 `depends_on` 不能使其合法。

`PassTemplate.depends_on` 是真实的额外控制依赖。编译适配把它与资源 producer 前置条件合并、映射到实际 PassId，**保留到最终 descriptor 与 runtime graph**，不是生成 SSA 后清空。无共享资源的控制边也必须生效。

### 校验责任

- Catalog 检查全部 feature、全部 variant，包括能力/质量不满足或 root 闭包外的声明：同 pass 重复写、多 writer 的 Automatic/分叉、self producer、history producer、重复/self control dependency，以及服务每个 variant 的完整输出写集。
- Catalog 的局部形状检查不冒充选中图检查。`ResourceTemplates::compile` 校验 producer 是否确实选中、是否写同一资源、控制前置 pass 是否存在；拒绝 transient 无来源读取及选中图中的跨 feature 分叉。
- 正数 `FrameLag(n)` 映射为 `frame - n` 的独立 Imported 地址，仅 Persistent 模板可用。历史读使用初始版本，不建立同帧 ProducedBy 回边，历史写被拒绝。
- 资源布局、usage、类型、逻辑 footprint、对齐与受检算术由模板/绑定校验承担。外部绑定只接受实际需要的非 transient 地址；重复、多余、缺失、物理重叠及类型/范围错误均有具体错误。

对应回归位于 `tests.rs`、`compile_tests.rs`：既有 catalog 正负例覆盖 inactive variant；`explicit_writer_chain_keeps_initial_and_intermediate_read_provenance`、`missing_wrong_and_unselected_producers_fail_before_recording` 等直接检查版本来源。测试存在与某轮执行通过是两件事，结果边界见末节。

## authority 版本与单内核 lowering

当前 `compile.rs` 的流程：

1. 用同一 `GpuFrameGraphBuilder::dependency_order()` 取得控制 admission 顺序；此调用不 lower 资源、不生成 SSA/hazard/alias。
2. 以 `AccessInput::Version(0)` 或 `ProducedBy { pass, access }` 引用声明中确切的 producer 写访问；pass/access 使用原始声明身份。
3. 一次完整 `compile_with_sources` 由 checked graph builder 分配输出版本，再由唯一编译内核生成 schedule、hazards、barriers、alias 与 queue batches。Fabric 不猜测或自行发放 SSA 输出版本。
4. 将 authority 的 texture-first access 表投影回原始访问次序，保留重复访问及来源，不按资源去重。

`BoundResourceUse` 分别保存 `input_ssa_version / input_content_version / input_producer` 和消费/产出侧 `ssa_version / content_version / producer`。write 的后者指向 authority 输出，read 指向输入；外部初始及历史内容为 SSA 0。content version 是 external base 加 authority SSA 的受检结果，不绕过 `GraphLimits.versions`。

`FrameRenderItem / DomainPayload` 继续复用 SceneHandle、MotionSource 及实际选中访问做 layout/domain 校验；Value 仅为定宽 Copy 数值，不另开指针/参数块旁路。`FeatureGraphSink` 按权威顺序消费原 PassSpec 与对应 barrier，错误立即传播；逻辑 sink 测试不等于 GPU submit。

## runtime 接缝及所有权

当前 `runtime.rs` 的 `prepare_runtime` 对 recorder/import 做 exact coverage 校验。PassKey 索引加顺序 callback 槽的 coverage 复杂度为 `O((P + R) log P)`，保留确定的 unexpected/duplicate/missing 错误，不再每 recorder 线性扫全部 pass。

2026-10-16，Scene adapter 的 `gpu/fabric.rs` 已将 `DeviceFabric::compile` 移到 `RenderStartup`：启动期创建固定 probe 资源、绑定表和 `CompiledFeatureGraph`，Render 帧只克隆已冻结的声明并执行 `prepare_runtime → record → Bevy 原生提交`。`SceneFabricDiagnostics` 记录 compile/prepare 次数、缓存命中、录制帧数及 compile/prepare 错误；因此不会再把每帧 compile 当作正常路径。这里仍是固定 frame=1 的 adapter probe：每帧 prepare 仍会创建一次 move-only `PreparedGraph`，并未解决动态 FrameLag、真实 Scene payload、completion/history retirement 或 pass-specific production recorders。

runtime 保留控制依赖，从每次访问的 `input_ssa_version` 取 handle，并核对写出的 authority 版本。`verify_plan` 对 order/liveness/position/queue batches、逐访问输入输出版本和 logical barriers 做语义投影核对：texture/buffer namespace 分离；源/目标 pass 与原始 access ordinal、重复访问的 multiplicity 均保留，不能只比较资源索引或 barrier 数量。

`RuntimeFeatureGraph::record` 对接 graph 的 `PreparedGraph / RecordedGraph / GraphAllocations`：`RenderGraph::prepare` 编译一次并冻结计划，不创建 GPU 资源、不运行执行 callback；`PreparedGraph::record` 使用同一计划，不再编译、不 submit。裸 `RecordedGraph` 持有 commands、plan 与 allocation owner，可用 `into_parts` 交给外部提交方；graph/Fabric 的 `execute` 是 record 后提交的便捷入口。裸 owner 需要在确认完成后显式 release，单纯 Drop 不销毁 GPU 资源；imports 始终由外部持有。

`prism_render_runtime::RenderRuntime::execute_owned` 则返回 `FrameSubmission`：实际提交队列的 completion callback 保留一个 clone，调用方 early drop 不会提前销毁资源；实际 callback 已运行且最后一个 owner 释放后，才在创建资源的 device 上回收自有资源。显式 clone 可将 Transient/Persistent 保留到 completion 之后；history 发布方须一直保留 owner，直到后续所有 imported 使用均完成。`device_generation()` 标明创建代次，旧 owner 不把旧资源 ID 或迟到 callback 迁移到恢复后的设备。callback 被丢弃或 wait 失败均不构成完成证明，submit 也不自动发布 history。

completion value 在同一设备代次内递增：`u64::MAX` 可发放一次，之后返回 `CompletionValueExhausted`，不环绕、不复用；饱和的 `next_completion_value()` 只是提示，不承诺下一提交成功。编译失败不动 pacer；值耗尽发生在 pacing wait 前；pacing wait 失败会把最老 token 恢复队首。这些拒绝路径均不创建 GPU 资源、不运行录制 callback、不发出新提交，也不推进提交值。`wait_idle` 仍执行真实 queue-wide barrier；若该 barrier 失败，按原顺序保留未确认 token 供重试。详细接口和回归范围见 `docs/frame_graph_implementation_zh.md`。

## PCM 与 Scene 的生产接入界线

PCM 使用真实 `ExtensionRegistry / ExtensionManifest / CapabilitySet`；静态 UID→FeatureId/kind/version sidecar 不以 slot/显示名/注册顺序代替身份。Provider→Service、ResponseMode/Composite→Response；能力没有设备词汇等价物时返回明确错误，不按位强转或静默丢弃要求。

Scene 已有实际接入：`gpu/fabric.rs::init_scene_fabric` 从拥有 Scene buffers 的 Bevy RenderDevice 冻结内建根与选择；`shading/plugin.rs` 显式在其后初始化 shading execution。shading/water 用冻结 variant 准入、真实 executor 和 WaterSurface 外部执行站点，不由可见水体或 extracted items 动态增删根。

Scene 的当前生产执行仍是专用 shading 声明图、Core3d driver 与各 per-view resource owner。其 SSR→GI/denoise→SSGI→TAA 实际调用和资源见 `docs/frame_graph_implementation_zh.md`。本轮新增的 compile-once adapter 只证明冻结声明可以进入真实 Scene RHI 的 runtime prepare/record/submit 接缝，**不表示通用 ResourceTemplates→prepare_runtime→record→Scene 原生提交与完成回收的整条生产路径已经完成**。

## 已执行检查点与本次核对范围

以下为主线已执行并反馈的检查点；本次仅核对源码与更新文档，没有运行 cargo。

| 范围 | 主线反馈的执行结果与边界 |
|---|---|
| graph `--lib --all-features` | 最新重跑 78 passed，零 warning |
| architecture Render Fabric | 本轮 122 passed / 0 failed |
| runtime 严格 `PRISM_REQUIRE_GPU=1`、`--lib -- --test-threads=1` | 32 passed / 0 failed，包含 13 项 controlled retirement 与 2 项真实 GPU 测试 |
| runtime 两项 GPU `--nocapture` 复跑 | 2 passed；实际 adapter 输出 Apple M2 / Metal / Hardware |
| Scene `specular_` / plugin 定向测试 | 较早检查点分别为 14 / 13 通过 |
| Apple M2 真实设备 startup | 较早检查点 1 通过 |
| `cargo check -p prism_render_app --all-targets` 与 owner caller 回归 | check 已通过，有既存 warnings；修复 3 处新增 `RecordedGraph` unused 后，`mesh_draw_records_into_a_graph_pass` 1 项及 `records_a_multi_draw` 2 项均通过 |
| gpudriven 严格 GPU `--lib -- --nocapture --test-threads=1` | 18 passed；7 条路径输出 Apple M2 / Metal 硬件证据 |

runtime GPU 证据是同一真实设备上的 transfer 链、readback、completion callback、owner clone/early drop 与 backend 资源表项回收；通过槽位复用及 generation 递增验证旧表项已移除，不代表驱动 heap 已归还。controlled retirement 测试补充可控的完成前保活、失败 wait、代次隔离和溢出路径，不能冒充 GPU 执行。app 三项 caller 回归仅捕获命令、不提交 GPU，因此录制后立即 release，不计作 GPU 验证；gpudriven 的 18 项也不全部描述为硬件执行路径。上述结果不等于完整 Scene 帧、通用 Fabric 整帧生产接线、图像 golden、物理多队列或性能验收。逻辑 footprint 不代表后端 heap 放置大小，绑定元数据校验不替代真实 device allocation/format 限制。

本次只更新指定 tracked 文档，不改生产代码、不改 `docs/refactoring-dev/`、不提交。
