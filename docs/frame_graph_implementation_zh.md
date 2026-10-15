# Frame Graph 单内核迁移实跑记录

> 版本：0.20.0-dev；适用引擎：Prism；日期：2026-10-14。
> 对应规范：`docs/refactoring-dev/04-frame-graph_refactor_zh.md` §2、§3.4、§5 门禁 1/2、§7 S1→S2a→S2b。
> 状态：S1 / S2a / S2b / S3 / S4 / S5 均已 **SHIPPED**。S5 的 23 个声明式后处理节点均已绑定到实际 Core3d executor；复杂效果以有序 executor 链表达，并由启动期与运行期账本校验覆盖、唯一性和顺序。

## 已交付接口与边界

- 唯一 SSA 依赖、RAW/WAR/WAW hazard、确定性拓扑排序、生命周期、checked first-fit 分配内核位于 `prism_render_graph::compile::kernel`。可执行 RenderGraph 与声明 facade 均委托它；queue batch 统一由 `compile::schedule::queue_batches` 产生。
- `prism_render_architecture::frame_graph` 只重导出 `prism_render_graph::frame_graph`。architecture 的旧 builder/compiler/schedule/transient/types 实现删除，旧测试保留并迁移错误 API。
- scene 的 shading / visibility 图直接使用 `prism_render_graph`。本阶段保持既有 6 / 3 pass 拓扑、资源版本、9 条具名 shading barrier 与队列批语义。
- runtime 用 `CompiledGpuFrameGraph::declared_pass_count` 校验执行顺序，不再各自维护 5/6 常量。真实 ECS 系统连续执行两次的回归覆盖此路径，也覆盖已有 readback/native-barycentrics 状态保存。
- 可执行 `ExecutionPlan` 增加 `logical_barriers` / `queue_batches`，既有 driver `StateTracker` 仅负责具体 layout、子资源以及 imported 入口/出口状态的 lowering；不再承担另一份 SSA 依赖求解。逻辑队列批不等于后端已支持多队列物理提交。
- `GraphLimits` 与 `try_*` admission 在缩窄 ID、增长表、修改版本或调用 pass setup 之前校验。越界访问拒绝后，不改 usage/version/access table；非法未来版本拒绝，合法旧版本仍可使用。
- checked admission 的原子性是单操作级，不是整个用户 setup closure 的事务回滚。句柄目前没有 graph identity；另一个 graph 的同范围 ID 无法区分。没有宣称覆盖这两个边界。
- `CompileError::Cycle { unscheduled, blocked, cycle, names }` 提供实际闭合路径与名称；下游 blocked pass 不伪装成环成员。错误类型由 Copy 调整为 Clone；旧 `CyclicDependency` 匹配须迁移到 Cycle。
- 旧版句柄的 overwrite 使用 output-1 的物理前驱建立 WAW/WAR，不漏掉迟声明的旧版本读者；同 pass 不产生自 hazard。含旧版本读者的可执行真实环已有回归。
- allocator 生命周期端点包含使用 pass，只在 `last < first` 时复用。任意/零 alignment 及地址运算经过校验；heap 字节是地址范围而非 live-byte 之和。可执行 frontend 保留旧 256 字节 footprint padding；声明 frontend 保留 812 字节 golden。
- 删除未使用的声明 normalize 与 `TransientAllocation::plan`，不留 dead-code 备用算法。可执行 `schedule::normalize` 是仍被调用的 SSA 输入适配器，不是第二份 solver。

## 精确写集（本切片）

路径均相对仓库根。没有修改父线程 gpu_scene / material / response / app 实现，没有手工改根 Cargo.lock，没有 commit/push。

### 修改

```text
pkg/prism_render_architecture/Cargo.toml
pkg/prism_render_architecture/src/frame_graph/mod.rs
pkg/prism_render_architecture/src/frame_graph/tests.rs
pkg/prism_render_graph/src/builder.rs
pkg/prism_render_graph/src/compile/alias.rs
pkg/prism_render_graph/src/compile/barrier.rs
pkg/prism_render_graph/src/compile/lifetime.rs
pkg/prism_render_graph/src/compile/mod.rs
pkg/prism_render_graph/src/compile/schedule.rs
pkg/prism_render_graph/src/graph.rs
pkg/prism_render_graph/src/lib.rs
pkg/prism_render_graph/src/pass.rs
pkg/prism_render_graph/src/plan.rs
pkg/prism_render_graph/src/systems.rs
pkg/prism_render_scene/Cargo.toml
pkg/prism_render_scene/src/shading/graph.rs
pkg/prism_render_scene/src/shading/runtime.rs
pkg/prism_render_scene/src/visibility/graph.rs
```

两个 Cargo.toml 均仅增加本切片的 `prism_render_graph` path 依赖，其余并发变更不归本切片。

### 删除

```text
pkg/prism_render_architecture/src/frame_graph/builder.rs
pkg/prism_render_architecture/src/frame_graph/compiler.rs
pkg/prism_render_architecture/src/frame_graph/compiler/schedule.rs
pkg/prism_render_architecture/src/frame_graph/transient.rs
pkg/prism_render_architecture/src/frame_graph/types.rs
pkg/prism_render_graph/src/tests.rs
```

最后一项是原 1053 行测试迁移到 `text/unit/graph.rs`，不是删除测试或放宽门禁。

### 新增

```text
pkg/prism_render_graph/src/capacity.rs
pkg/prism_render_graph/src/compile/kernel.rs
pkg/prism_render_graph/src/frame_graph.rs
pkg/prism_render_graph/src/frame_graph/builder.rs
pkg/prism_render_graph/src/frame_graph/compiler.rs
pkg/prism_render_graph/src/frame_graph/transient.rs
pkg/prism_render_graph/src/frame_graph/types.rs
pkg/prism_render_graph/text/unit/compiler.rs
pkg/prism_render_graph/text/unit/graph.rs
pkg/prism_render_graph/text/unit/systems.rs
pkg/prism_render_graph/text/unit/transient.rs
docs/frame_graph_implementation_zh.md
```

建议集成顺序：graph 权威内核/API/原测试迁移一起；architecture 重导出及 API 测试迁移一起；scene 依赖/两个图/runtime 及本记录一起。不能只提交删除项或漏掉新增 kernel/capacity/facade 文件。

## 实跑验证

以下均在此工作区运行，2026-10-14 实际退出码为 0；计数是本次并发工作区快照，不推断后续父侧新增测试数量。

| 命令 | 结果 |
|---|---|
| `cargo test -q -p prism_render_graph --all-features` | 38/38；doctest 1 通过、1 既有 ignored；文档落地前再次复跑同样通过 |
| `cargo test -q -p prism_render_graph --no-default-features` | 35/35；doctest 1 通过、1 既有 ignored |
| `cargo clippy -p prism_render_graph --all-targets --all-features -- -D warnings -D missing-docs` | 通过，零告警 |
| `RUSTDOCFLAGS='-D warnings' cargo doc -p prism_render_graph --no-deps --all-features` | 通过，零告警；修正两处旧 CommandBuffer 链接后重跑 |
| `cargo test -q -p prism_render_architecture --lib` | 11241/11241，无告警输出 |
| `cargo test -q -p prism_render_architecture --lib gpu_scene` | 26/26；11215 filtered，无告警输出 |
| `cargo test -q -p prism_render_scene --lib` | 1719/1719；存在范围外既有告警，不能称全链零告警 |
| `cargo test -q -p prism_render_runtime -p prism_render_gpudriven -p prism_render_app --lib` | runtime 12/12、gpudriven 18/18、app 146/146；有范围外告警 |

局部 rustfmt 使用 edition 2024 与 `skip_children=true`，只处理上述已修改/新增 Rust 文件；未运行 workspace 全量格式化。范围内 `git diff --check` 通过。

scene 的范围外告警包括 lobe_sampling/virtual_texture/spec_denoise/texel_cache 的 qualification/unused/dead-method、world_restir 的 Mat4 deprecated、area_light 的 unmet lint；其他消费者还有 asset streaming missing-docs、app 测试 qualification/unused 和 page residency dead-method。本切片不越权修这些文件。graph 本身 missing-docs、严格 Clippy 与 rustdoc 门禁通过。

新增 compiler 10 个回归覆盖：自然 u32/配置容量边界且不分配巨表；setup 前拒绝；失败访问的原子性；非法资源/自依赖/version；两 frontend 的 order/hazard/alias/queue 一致；迟声明旧版本读者；stale overwrite；真实可执行环与 blocked 下游；同 pass/fanout 确定性；812 golden/地址洞/非幂次 alignment/overflow。architecture 原 8 个测试和 graph 原测试保留。

未运行完整 workspace tests/Clippy；未做 GPU 物理 alias、多队列 submit、帧时间或无分配基准。本记录不把纯编译器一致性测试当成这些性能/设备证据。

## 多维质量评估

| 维度 | 状态与取舍 |
|---|---|
| 性能 | 部分：只保留一个 solver；BTreeSet 保证稳定排序，编译阶段仍分配，尚无帧预算/零分配证明 |
| 效果 | 旧 topology/barrier/version/alias golden 保持；Composition 与后处理声明/执行账本已接入，设备图像质量仍由对应 GPU/golden 测试给出证据 |
| 易用性 | 部分：旧 facade 仍可用且委托权威；新增 checked API，错误迁移与 panic 前置条件有文档 |
| 易维护性 | 满足本切片：重复内核删除；graph 原测试收口 text；scene/architecture 既有测试布局没有扩大迁移 |
| 易拓展性 | 逻辑 SSA 与 driver lowering 分离；跨 graph 句柄身份与物理多队列保持独立接口，不改变本图的确定性编译结果 |
| 易迭代性 | 满足：S1–S2b 独立集成切片，写集与接口迁移明确；不混入父 response 实现 |
| 健壮性 | 满足本切片：容量先验、checked 算术、Result、稳定顺序、具体 cycle 诊断均有回归；closure 全事务不在范围 |
| 安全 | 满足本切片：无新增 unsafe；非法索引在表 lookup 前拒绝；未宣称任意 foreign graph 句柄可检出 |

## 已验证范围与协作接口

- S3 Composition 已常驻于 `shading_frame_graph()`：`response_composition` 消费四个 imported response plane，并在 `shading_resolve` 后、后处理前写回 `hdr_scene_color`；其资源所有权仍由各 producer 管理。
- S4 codegen 已由 `post_graph.rs` 的声明表统一产出 ABI、bind-group、pipeline entry、资源访问和依赖边；声明错误在图变更前拒绝。
- S5 已完成全量接入：`POST_PROCESS_DECLARATIONS` 的 23 个声明与 `POST_EXECUTION_BINDINGS` 一一对应。曝光、SSR、SSGI、spec denoise 等多 dispatch 效果记录完整 executor 链；账本拒绝缺失、重复或错序绑定。具体系统仍按 Bevy Core3d 的分组限制注册，但不再存在未消费的声明式 pass。

## S5 执行账本

`ShadingFrameGraph` 同时保存编译产物与 post executor 账本。启动时校验声明名称、图尾顺序、executor 非空以及 executor 唯一性；`prepare_shading_work` 每帧保留同一账本身份，避免运行期悄然换表。账本是稳定的诊断/验证接口，不把 Rust 系统类型名暴露为 ABI，也不在每帧重新分配或排序。

这条校验不冒充 GPU 设备效果验证：设备能力、shader 编译、真实帧时间和图像质量仍由各专项 GPU/golden 门禁负责；本文件只确认声明图与实际 executor 装配一致。
- 水体 Fabric 与 particle frame-lag/versioning 属于独立的 architecture/scene 切片，分别维护自身的资源提供者与验证报告；本图只消费它们最终导入的资源句柄。
