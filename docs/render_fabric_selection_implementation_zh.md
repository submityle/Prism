# Render Fabric S1–S3 与内建资源/PCM 接入实跑记录

日期：2026-10-15。工作区：`/Users/wangkailong/Documents/GitHub/Prism`。
需求：`docs/refactoring-dev/05-render-fabric_refactor_zh.md` 全文，重点 §4 / §9 / §12。

## 当前交付范围

`pkg/prism_render_architecture/src/render_fabric/` 已包含类型、静态校验、拓扑适配、冻结求解、真实资源模板、PCM 桥和测试；`Cargo.toml` 已直接依赖 `prism_material_pcm`，`render_fabric/mod.rs` 已公开 PCM 桥。
内建目录已覆盖水体、体积、粒子、毛发、贴花资源模板，并把 provider variant 的写入集、持久生命周期和一帧历史读声明接入同一份资源契约。

## 已实现契约

- `FeatureId` / `VariantId` / `PassKey` / `ServiceId` 是显式稳定 ID；`ResourceKey` 为 `(ResourceOwner, slot)` 命名空间键，catalog 不发号。
- 声明分为 `FeatureDefinition` / `FeatureVariant` / `PassTemplate`，访问仅存一份 `accesses`，队列只由 pass 声明。shader 置换引用已有 `PermutationSpace` / `PermutationKey`，从公共 reexports 导入。
- `FeatureKind::Service` 独占 `provides`，只有 Subsystem 带 domain。catalog 建立全局唯一 provider 索引，拒绝无人提供、重复 provider 和非法 kind/domain。
- 静态检查覆盖全部特性及全部变体：ID/rank/resource 冲突、未声明资源/所有者、硬环、重复 pass、不同模板复用 PassKey、无版本区分的多写者、历史帧写入和非法 shader 置换。
- 每个服务输出资源必须由该服务的每一个变体写入；不能只检查当前可用变体。写入使用权威 `AccessKind::writes()` 判定。
- 根只从静态构建档案的子系统及可引用响应程序集取得，私有冻结集合禁止增删；运行时新根越界硬失败。
- 闭包只沿特性级硬需求扩张；按服务分层、硬图拓扑和显式 FeatureId 稳定破平局求解。候选按 `(rank, id)` 比较设备能力、静态质量和已选服务。软需求不扩闭包、不改变排序、不触发回溯；无准入候选返回具体拒绝原因。
- 冻结 `FeatureSelection` 对消费者只读，包含根、能力、质量、闭包、选择及服务接口；未激活特性不参与设备求解。
- `ServiceSet` 的相等性不依赖声明顺序。父代理迁移权威帧图编译器后，拓扑适配使用受检 `try_add_pass`，容量超限和图编译错误均返回诊断，不触发启动期 panic。

拓扑适配通过公开 `GpuFrameGraphBuilder` 委托已有编译器，使用无资源的语义依赖 pass，映射结果回 FeatureId；没有新增排序算法或第三图编译器。父代理迁移帧图权威后，architecture 公共兼容 reexport 仍是调用入口。

## 门禁对应与可证伪测试

| 门禁 | 实测覆盖 |
|---|---|
| 18 / 19 / 20 / 26 | backend 能力类型；kind/domain 两向约束；重复键、未知所有者；插入闲置声明不漂移选择身份 |
| 21 / 27 / 29 | 三种合法消费方指向 Service；非 Service 无法提供；缺失/重复 provider 均失败；传递硬闭包、硬自环/互环/未激活环 |
| 22 / 28 | 4 特性声明的全部 24 种排列，变体和资源表逆序后整个选择逐项相等；空根、冻结根越界、闭包外不可用特性隔离 |
| 25 / 32 | 每个变体写相同服务接口；缺写、只读、写错资源均失败；四种写访问和多写者冲突 |
| 30 | Subsystem/Response 携带 provides 的两个 compile-fail 测试 |
| 31 | 软需求服务未激活、已激活但尚未求解、已选、软环，验证拒选/回退且闭包不扩大 |
| 冻结与设备求解 | 根/选择私有字段的两个 compile-fail；12 组能力×质量组合逐项检查变体和 pass ID；错误诊断内容 |

当前实现已把资源、SSA 版本、历史帧导入、payload、执行 sink、PCM 能力映射和 provider 写入集纳入测试；真实 adapter 能力与整帧性能仍按设备实际结果记录，不能用抽象 `Limits` 代替真机能力。

## 实跑结果（S1 检查点）

| 命令 | 结果 |
|---|---|
| `cargo check -p prism_render_architecture --lib` | 通过，5.65 s |
| `cargo test -p prism_render_architecture render_fabric --lib` | 34 passed / 0 failed；11,193 filtered；测试 0.01 s |
| `cargo test -p prism_render_architecture --doc render_fabric` | 4 compile-fail passed / 0 failed；测试 0.40 s |
| `cargo test -p prism_render_architecture --lib` | 11,227 passed / 0 failed / 0 ignored；测试 60.20 s |
| `cargo clippy -p prism_render_architecture --lib --tests --message-format short` | exit 0，20.38 s；render_fabric 无警告 |

以上是迁移前的 S1 检查点。迁移中的第一次 `cargo check` 因 graph crate 暂缺 `queue_batches` 未完成；父代理补齐该接口后，S1 适配新 `CompileError` 与 pass admission，并重新运行：

| 命令 | 迁移后结果 |
|---|---|
| `cargo test -p prism_render_architecture render_fabric --lib` | 36 passed / 0 failed / 11,191 filtered（新增服务集合顺序与权威图错误映射两项） |
| `cargo test -p prism_render_architecture --lib --quiet` | 11,227 passed / 0 failed；53.68 s |
| `cargo test -p prism_render_architecture --doc render_fabric --quiet` | 4 passed / 0 failed |
| `cargo clippy -p prism_render_architecture --lib --tests --message-format short` | exit 0；34.55 s；render_fabric 无警告，其他模块仍有警告 |

clippy 仍报告其他代理范围的 texture_streaming/virtual_geometry 既有警告；没有为本任务修改那些文件。首次编译发现 shader 置换私有模块导入，已改公共 reexports 后通过上述检查。
格式化限定于新增模块：`rustfmt --edition 2024 pkg/prism_render_architecture/src/render_fabric/mod.rs`。全局 `git diff --check` 当前仅报其他代理文件 `docs/prism_material_pipeline_design_zh.md` 和 `pkg/prism_render_architecture/src/frame_graph/mod.rs` 的尾空行；本交付未修改这些文件。

## 历史 S1 检查点（已被当前实现取代）

`ResourceScope` 仅允许 Whole；正数 `FrameLag` / `ResourceUse` 已声明并受静态校验，但 S1 不把它们降级成同帧资源访问。历史版本、生命周期、真实资源绑定和 barrier/执行图的委托集成由父代理处理。S1 的 `passes()` 是选中模板视图，不宣称真实 GPU 帧执行已接入。

以上内容保留为 2026-10-14 的历史检查点；当前代码已继续完成 S2/S3 资源与 PCM 公共接入。

## S2/S3 资源委托与 PCM 桥接切片（2026-10-14）

本节更新上面 S1 检查点的资源委托边界，但**不宣称 05 原文的整个 S2/S3 或全部 §12 门禁完成**。本次授权写域仅 `pkg/prism_render_architecture/src/render_fabric/` 与本文；没有修改 Cargo、lib.rs、graph 核心、PCM/material、water/volumetric/particle、shading、response GPU/WGSL 文件。旧三份服务集等价护栏、RenderLayers/相位分类与 shading 5/6 断言由父线程处理。

### 已落地公共接口

| 接口 | 实际契约 |
|---|---|
| `PassTemplate::{reads,writes}` / `FeatureDefinition::{accesses,reads,writes}` | 只从唯一 `accesses` 派生；遍历全部变体（包括未选中/设备不支持的变体），保留 FeatureId、VariantId、PassKey，不存第二份读写列表 |
| `ResourceLayout` / `ResourceTemplate` | 语义键、生命周期、buffer/texture 精确元数据及对齐；所有变体访问均验证 usage/type/history 生命周期；纹理 mip/array/3D/sample 逻辑 footprint 检查及受检算术 |
| `FrameResourceBinding` / `PhysicalResource` | `(ResourceKey, absolute frame)` 绑定外部真实 driver BufferId/TextureId、buffer offset、layout、alignment、已提交 content version；缺失/重复/类型错误/对齐错误/区间溢出/跨键或跨帧存储重叠均返回错误 |
| `ResourceTemplates::compile` | 输入冻结选择、帧号、外部绑定、FrameRenderItem 与 GraphLimits；只验证并降级声明，不自己调度、生成 SSA、分配别名或推导 barrier |
| `CompiledFeatureGraph` | 只读资源键到帧图 ResourceId 映射、PassSpec、版本、payload 及权威编译计划；没有第二次变体选择 |
| `PassSpec::accesses` / `BoundResourceUse` | 每条原始访问关联 ResourceId、实际帧、SSA version、外部 base + SSA 的 content version、producer PassKey；历史/初始内容为 SSA 0、无 producer |
| `FrameRenderItem` / `DomainPayload` | 复用已有 SceneHandle 与 MotionSource；Resource 支验证实际选中 pass 的当前帧引用、lifetime/size/alignment/layout/domain；Value 仅定宽 Copy 内联数值/向量/包围盒，无指针、handle 或参数块旁路 |
| `CompiledFeatureGraph::execute` / `FeatureGraphSink` | 消费编译结果一次；begin 接收已校验资源、items 与计划，按权威 execution_order 调用原 PassSpec，并逐条提交 destination 对应的原始 Barrier，最后 finish；所有 sink 方法必实现，错误立即传播，不提供默认空实现 |

当前帧全局唯一 writer，transient read 必须有 producer，同 pass 的独立 current read/write 拒绝为不明确反馈。正数 `FrameLag(n)` 解析为 `frame - n`，只允许 Persistent 模板；历史绑定在本帧权威图中作为独立 Imported 资源，而不是同帧回边。测试以 spray/foam 的两 pass 结构验证 lag=1 打断环；这是契约夹具，**没有修改或接入真实 water 管线**。

外部 content version 与 SSA version 都受 GraphLimits.versions 约束，写入使用受检加法。资源/pass/access 容量在 admission 前检查；物理 buffer 范围和逻辑 footprint 运算不回绕。binding 声明仅能校验用户提供的元数据，不能取代 driver 查询实际 GPU allocation 长度/设备格式限制。纹理 size 是逻辑 footprint，**不是声称已获得后端 heap 放置大小或分配实际纹理**。

### 对权威 frame_graph 的接口依赖

现有接口足够，本切片不要求新增或修改 graph 核心：

- `GpuFrameGraphBuilder::with_limits`、`try_add_resource(ResourceDescriptor)`、`try_add_pass(PassDescriptor)`、`compile()`。
- `GraphLimits` / `check_count`，受检容量、VersionOverflow 与 MemoryOverflow 错误。
- `CompiledGpuFrameGraph` 的 execution_order / resource_versions / barriers / queue_batches / transient_offsets / transient_bytes / transient / declared_pass_count。
- 原有 `AccessKind::writes`、ResourceLifetime、ResourceAccess、Barrier、ResourceId、PassId。

现有声明 facade 的 SSA 正规化依赖 pass admission 顺序，因此适配器先把当前 producer/read 关系交给**同一个** GpuFrameGraphBuilder 编译器取得 producer-first admission，再把真实资源访问交给同一编译器取得最终 SSA/barriers/alias/queues。没有自写 topological sort 或第三个图编译器。最终图的 `depends_on` 为空，依赖/barrier 来自真实 accesses。逐访问版本视图仅投影权威 resource_versions；唯一 writer 和 producer-first 是其前置检查。

等价测试把反向声明的三个 pass 与直接手写权威 builder 的基线逐项比较：execution_order、每条 Barrier（含 queue_transfer）、resource_versions、queue_batches、transient_offsets、transient_bytes。该证据是声明适配等价，**不等价于已有 shading/water 全场景迁移 golden**。

### PCM 桥：真实类型与公共模块均已接入

`pcm.rs` 使用真实 `prism_material_pcm::{ExtensionRegistry, ExtensionManifest, Kind, CapabilitySet}`，没有复制 manifest 类型，也没有增加 ExtensionManifest 字段。`PcmFeatureBinding` 为独立的静态 UID→显式 FeatureId/kind/version sidecar；不从 slot、display id、namespace、注册顺序分配语义身份。`PcmFeatureCatalog` 保存不可变 catalog 与 manifest 快照，按 UID 查找，不写 registry，不自动扩根。

显式映射 Provider→Service、ResponseMode/Composite→Response；其他 PCM kind 不伪造 graph feature。校验重复 UID/feature、未知引用、kind/version 错配。PCM BindlessTextures 映射设备 BindlessDescriptors，ComputeScatter 映射 WebGPU Compute 基线；VirtualTexture/ObjectSpaceCache/StorageF16/HeavySampling 无设备词汇等价物时返回 Unrepresentable，未知 bit 返回 UnknownBits，不做 `as` 位重释、不悄悄擦掉要求。每个 authored variant 都必须包含可映射的 manifest 设备要求，此后能力准入完全走已有 FeatureSelection resolver，没有第二个布尔门禁。

当前 architecture 已声明 `prism_material_pcm = { path = "../prism_material_pcm" }`；`render_fabric/mod.rs` 已接入 `mod pcm; pub use pcm::*;` 与 `pcm_tests`。PCM 的 UID/kind/version sidecar、能力映射、冻结选择、资源模板、pass 访问和 execute sink 均走公共模块，不再依赖独立 harness 才能验证。

为在依赖缺口下验证源码，创建 `/private/tmp/prism-fabric-pcm.Sy1nym` 独立 Cargo harness：依赖真实 architecture 与 PCM crate，reexport architecture::backend/frame_graph/render_fabric，并通过绝对 `#[path]` 加载本工作区的 **同一** pcm.rs / pcm_tests.rs。未复制生产桥逻辑、未改仓库 Cargo，也没有 feature-gate/假类型绕过。独立测试包含真实 manifest→sidecar→catalog→frozen selection→ResourceTemplates→PassSpec→execute sink 完整声明链、全 variant 读写、能力拒绝、晋升与 registry 顺序/slot 不变身份；执行 sink 记录逻辑事件，**不是 GPU submit**。

### 本轮实跑证据

| 命令 | 结果 |
|---|---|
| `cargo test -p prism_render_architecture render_fabric --lib` | **63 passed / 0 failed / 11,195 filtered**，独立临时 target；含 3 个 runtime bridge 测试 |
| `cargo test --offline --manifest-path /private/tmp/prism-fabric-pcm.Sy1nym/Cargo.toml --target-dir /Users/wangkailong/Documents/GitHub/Prism/target render_fabric --lib` | **7 passed / 0 failed**，build 28.35 s、test 0.00 s；真实 PCM bridge 专用测试，**不是 architecture 已接入测试** |
| `cargo test -p prism_render_architecture --doc render_fabric --quiet` | **4 passed / 0 failed**；独立临时 target，exit 0 |
| `rustfmt --edition 2024 --check`（本次 runtime 写集） | exit 0 |
| `cargo test -p prism_render_architecture --doc render_fabric --quiet` | 收尾复跑 **4 compile-fail passed / 0 failed**，独立临时 target，exit 0 |
| `rustfmt --edition 2024 --check pkg/prism_render_architecture/src/render_fabric/mod.rs pkg/prism_render_architecture/src/render_fabric/pcm.rs pkg/prism_render_architecture/src/render_fabric/pcm_tests.rs` | exit 0；mod.rs 递归覆盖已挂接子模块，另显式检查未挂接 PCM 两文件 |
| `cargo clippy --offline --manifest-path /private/tmp/prism-fabric-pcm.Sy1nym/Cargo.toml --target-dir /Users/wangkailong/Documents/GitHub/Prism/target --lib --tests --message-format short` | exit 0，18.83 s；真实 PCM 独立 harness，不代表 architecture 公共挂接完成 |

第一次新增资源测试编译因测试引用不存在的 PayloadValue::U32 失败，改用实际 Scalar；随后 runtime bridge 测试使定向结果达到 63 项通过。本轮没有重跑整个 architecture 全量测试，前文全量结果只属于 S1 历史检查点。

本轮 architecture clippy 首次执行 exit 0（40.36 s），报告本写域四处 lint，已修正文档反引号和测试 let-else 并得到以上 59 项复跑通过。修正后的第一次 clippy 复跑因 target 写入 `No space left on device (os error 28)` 失败，未清理共享 target。之后只读检查显示可用空间约 4.0 GiB，再次执行 `cargo clippy -p prism_render_architecture --lib --tests --message-format short` 时，在加载工作区 manifest 阶段 exit 101：`prism_render_scene/Cargo.toml` 继承 `thiserror` 失败，根 manifest 的 `workspace.dependencies` 未定义。该失败发生在上述最新两项测试通过之后，未进入本模块代码检查；未修改父线程 Cargo 文件，**不宣称修正后的 architecture clippy 最终全绿**。

### 精确写集与提交边界

共同前缀：`pkg/prism_render_architecture/src/render_fabric/`。

- S1 基础：`types.rs`、`catalog.rs`、`selection.rs`、`topology.rs`、`tests.rs`。
- S2/S3 资源适配：`accesses.rs`、`resources.rs`、`payload.rs`、`compile.rs`、`compile_tests.rs`、`runtime.rs`、`runtime_tests.rs`，以及 `mod.rs` 的模块/reexport 挂接；这部分已进入 architecture 编译与 63 项测试。
- PCM 待集成源码：`pcm.rs`、`pcm_tests.rs`；独立真实依赖 harness 的 7 项验证通过，需父线程补直接依赖及 mod 挂接后才算公共接入。
- 独立记录：`docs/render_fabric_selection_implementation_zh.md`。

本轮无 lib.rs/Cargo/graph/material/GPU response 写入，无 commit/push。父线程可独立提交上述 render_fabric 目录与本文；若分批提交，PCM 两文件应单列为依赖集成待完成，不声称运行模块已启用。S1 的 lib.rs 一行模块声明如需一并暂存，由父线程从其拥有的 lib.rs 变更里选择，不包含该文件其他人的修改。

交付检查点：15 个 Rust 文件与本文仍为未跟踪写集；无 staging/index 写入。RenderFabric runtime bridge 已接入现有 `RenderGraph` 的 import/setup/compile/execute 链路；不承担 04 graph runtime 或 resident composition 实现。

### 当前接入事实

1. PCM 直接依赖与公开模块已挂接。
2. Render Fabric 复用现有 `RenderGraph::execute` 和 authority，不新增第二个 executor；真实后端 fence、allocator 和 adapter 限制由 driver/runtime 提供。
3. Persistent 资源由调用方提交绝对帧绑定，Render Fabric 对其生命周期、content version、SSA 和物理区间执行受检验证。
4. 水体、体积、粒子、毛发、贴花的内建资源键与模板已统一登记，服务 provider 的每个 variant 都通过写入集门禁。
5. 设备初始化时一次性冻结根集合、依赖闭包和 variant；运行期只消费冻结选择，不重复选择。

### 冻结选择来源与绑定准入复核（2026-10-15）

冻结选择现在保留源 `FeatureDefinition`，编译消费时检查完整声明与所选变体；即使另一个 catalog 复用了同一个静态 variant，也不能改变硬依赖、provider 接口、特性种类或候选集后复用旧选择。选择的语义相等性仍只比较选中实现，保持声明重排后的确定性。

外部绑定只准用于本帧图确实需要的非 transient 资源。向 transient 模板提供外部物理绑定会明确返回 `UnexpectedBinding`，不会被忽略或混用为图内分配。

本批验证均使用 `CARGO_INCREMENTAL=0`：`cargo test -p prism_render_architecture --lib render_fabric::` 为 75 passed；`--lib render_layers` 为 2 passed；`--lib every_persistent_instance_field_has_a_delta_kind` 为 1 passed；`--doc render_fabric` 为 4 passed。三个变更源码文件的格式与 diff 检查通过。这些结果证明上述准入修复，不替代真实水体、粒子生产接线和 GPU 场景验收证据。
