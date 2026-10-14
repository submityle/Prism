# 03-S1：具名 PipelineState 与热缓存固定功能身份

记录日期：2026-10-14。已完整阅读 `docs/refactoring-dev/03-shader-package_refactor_zh.md`。

## 后续集成复验（2026-10-14）

以下为本次恢复任务后的最新实测，正文末尾的编译阻塞是初次 S1 交付时的历史记录，不再是当前状态。
父线程已适配 mesh 调用点并修复并行编译阻塞；本次运行 `cargo test -p prism_render_app --lib` 为 **146 passed / 0 failed**。
`cargo test -p prism_render_app --lib pipeline_cache -- --test-threads=1` 为 **18 passed / 0 failed**。
本次没有覆盖父线程的 mesh 改动，也没有修改 material/architecture/schema/codegen 或无关 page_pool 文件。

新增 `PipelineState::canonical_bytes() -> Vec<u8>`：返回与 `to_hash()` 完全同源的 schema-v1 字节流，为后续持久化导入时比较完整固定功能状态作准备。
`to_hash()` 仍不分配字节缓冲，既有 golden `0xc5dc594b5a5a8cb1` 不变。
新增第 9 项 state 测试 `canonical_state_bytes_match_hash_and_cover_every_fixed_function_field`，独立对字节计算 FNV 并逐字段变异验证字节/hash 一致。
含六项 worker 测试的补充独立 harness 单线程复验为 **15 passed / 0 failed**。
这项准备工作不等于 PPSO 导入或 S2/S3 完成；当前 `PipelineKey` 字段与构造 API 均未改变，S2 依赖共享身份类型与 checked dense permutation 接缝。

## 范围与当前状态

本次只实现 03-S1：固定功能具名状态、稳定 `to_hash()`，以及热 PipelineCache 实际使用完整状态查询。
保留 S1 期间原有 `class` / `shader_id`；未实现 S2 四分量身份/package interning，也未实现 S3 冷热统一、S4 worker pool 或 S5–S7。
初次交付时源码及专项测试已落地、独立测试通过，cargo 集成验收被下述两个外部调用点/并行测试问题阻塞；最新复验以本文开头为准。

没有改动 mesh_pipeline、architecture、共享 lib.rs、其他 owner 的实现或 refactoring-dev 文档；没有 commit/push。

## 修改文件

- `pkg/prism_render_app/src/pipeline_cache.rs`：完整状态 key、descriptor 一致性验证、非 Copy key 所需 clone。
- `pkg/prism_render_app/src/pipeline_state.rs`：具名状态、descriptor 提取、稳定编码。
- `pkg/prism_render_app/src/pipeline_state_tests.rs`：8 项专项测试。
- `pkg/prism_render_app/src/tests.rs`：仅迁移四项既有 pipeline-cache/prewarm 测试，不迁移 mesh-specializer 测试。
- 本文：实现事实、验证结果与父 agent 集成要求。

## 真正用于热缓存的状态

`PipelineKey.typed_pipeline_state: PipelineState` 直接参与派生的 Eq / Ord / Hash。
`ContentCache`、pending / waiting / in_flight / canceled / failed 传递或比较完整 key；不是只存模型，也不是以 u64 state hash 代替语义身份。
因此即使 `to_hash()` 碰撞，热缓存也不会因碰撞合并不同固定功能管线。

状态覆盖当前 RHI descriptor 的全部固定功能字段：

- vertex_buffers：有序 buffer 列表、stride、step mode、有序 attributes 的 format/offset/location。
- rasterizer：topology、strip index format、front face、可选 cull mode、polygon mode、unclipped depth。
- depth_stencil：可选附件、format、depth write/compare、前后面的 stencil compare/fail/depth_fail/pass、read/write masks、depth bias constant/slope/clamp。
- multisample：count、完整 u64 mask、alpha-to-coverage。
- color_targets：所有有序 target 槽，包括禁用槽、format、可选 blend、RGB/alpha 的 src/dst/op、write mask。

没有 fragment stage 与存在 fragment 但 targets 为空明确不同。Option 的存在性、列表长度、顺序都参与身份和编码。
depth bias 的 f32 使用 `FloatBits(u32)` 精确保留 bits；正负零与 NaN payload 不归一化，Eq/Ord 保持自反和全序。

label、shader module 后端句柄、entry point、pipeline layout 后端句柄不属于固定功能状态，不编码进去。
S1 中调用者必须使 `shader_id` 表达 shader/entry points 与兼容 pipeline layout 的程序身份；这部分仍是调用者契约，不声称解决 S2 程序身份。

`get_or_create` 和 `resolve_or_request` 在命中查询或其他状态变更之前验证 key 与 descriptor 的固定功能一致性；后者同时验证 fallback。
prewarm 经过同一个验证入口。错误 key 使用普通 `assert_eq!` 明确拒绝，包括 release 构建；不会修补或回写 key。

## 稳定编码

`PipelineState::to_hash() -> prism_render_architecture::shader_package::PipelineStateHash`。
使用 FNV-1a 64-bit，初值 `0xcbf29ce484222325`，乘数 `0x100000001b3`，前缀 `PSTATE`，schema version 1。
版本、标量、枚举 tag、长度均按 LE u64 编码；bool/Option tag 为 0/1；i32 符号扩展为 i64 后转 u64；f32 bits 按 u32 零扩展。
枚举使用显式稳定 tag，而不是 RHI 原始 discriminant 或标准库 DefaultHasher；不含地址、运行时 handle 或平台字节序。
提取与编码均使用无 `..` 的穷尽结构体解构，新加字段未处理会编译失败。未知未来 RHI enum 值明确 panic，须先补状态 schema。
既有 tag 不得重编号；改变线编码须更新版本与 golden fixture。此步骤没有接持久化文件或 PPSO 读写。

golden descriptor 的 53 个 LE u64 wire words（不含 ASCII 前缀）独立计算与 Rust 测试均得到 `0xc5dc594b5a5a8cb1`。
64-bit 编码仍可能碰撞，不作为碰撞无损身份或密码学保证。

## S1 API 与父 agent 迁移

```rust
PipelineKey::new(class: u32, shader_id: u64, typed_pipeline_state: PipelineState)
PipelineKey::from_descriptor(class: u32, shader_id: u64, descriptor: &RenderPipelineDescriptor)
PipelineState::from_descriptor(descriptor: &RenderPipelineDescriptor)
PipelineState::to_hash(&self) -> PipelineStateHash
```

旧七参数构造器删除：它无法描述 provider 的完整固定功能，不能做无损兼容适配。
`PipelineKey` 从 Copy 改为 Clone；完整 state 及 buffers/targets 使用 Vec。
`hash_vertex_layouts` 暂保留供既有调用方诊断，但不再参与热 key 身份。
状态模块当前位于 `pipeline_cache::pipeline_state`，通过拥有的 cache 文件挂载；共享 lib.rs 未改。
父 agent 应在公共根/prelude 导出 PipelineState，必要时导出 pipeline_state 模块以便命名嵌套状态类型。

父 agent 的 mesh 迁移必须：

1. 先执行 `with_mesh_layout(provider_descriptor, layout)`，完成全部 provider 覆盖。
2. exact、fallback、prewarm 各自从实际最终 descriptor 提取 key。
3. 保存原始完整 key 用于 release，不能仅凭 MeshPipelineKey + layout 推算 provider 的 bias/blend/stencil 等；需要 key 索引或修改 release 接口。
4. 按需 clone 非 Copy key。后续四分量身份替换 class/shader_id 时，保留本次完整 typed state 比较契约。

本切片不替父 agent 修改 mesh_pipeline。S2 接口可在 architecture 中独立定义，再由 cache owner 接入；本记录不冒充四分量已实现。

## 验证结果

新增专项测试：

- `state_struct_fields_are_all_hashed`：逐固定功能字段及容器边界变异，分别检查 state、热 key、hash 的变化。
- `stable_pipeline_state_hash_golden_and_float_bits`：golden、稳定 clone、正负零、NaN payload。
- `pipeline_state_preserves_layout_order_and_optional_stage_boundaries`：buffer/attribute 顺序、空 fragment 与无 fragment。
- `descriptor_runtime_handles_and_labels_are_not_fixed_function_state`：固定功能与程序/后端句柄边界。
- `pipeline_cache_uses_complete_descriptor_state_for_lookup`：NullDevice 实际创建 ID、去重、变异不混用、完整 key release/refcount。
- `pipeline_key_is_used_for_lookup_and_not_for_writeback`：调用者 key 不变，缓存命中与 fallback 不接受错配 descriptor。
- `pipeline_cache_rejects_stale_state_before_cold_requests_and_prewarm`：冷查询/prewarm 拒绝错误状态，未插入 cache/pending/waiting。
- `async_pipeline_cache_tracks_fixed_state_in_pending_and_fallback`：相同 class/shader 的不同 bias 分成两个后台工作项，得到不同 ready ID；编译器收到的 state 与 descriptor 相符。

使用 rustc 临时 harness 直接加载工作区最新 pipeline_cache/state/test 源码，以已有依赖 rlib 链接，NullDevice 路径 **8 passed / 0 failed**。
harness 位于 `/private/tmp/prism-pipeline-state-s1.0cq7ed/`，只是补充隔离验证；已有依赖产物不是本轮整包新编译证明。

实际运行 `cargo test -p prism_render_app --lib pipeline_state`：失败，最新一次编译只剩两个错误：

- `mesh_pipeline.rs:49`：E0061，仍调用删除的七参数 `PipelineKey::new`；已报告父 agent，未越权修改。
- `render_asset/page_pool_tests.rs:190`：E0283，`(0u8..16).collect()` 无法推断 `Into<Arc<[u8]>>` 的目标；属于并行 owner 测试，未修改。

首次还遇到并行 render_fabric 缺文件/私有导入错误，后续这些错误已消失。缓存测试的非 Copy 编译错误已修复。
已有四项缓存/prewarm测试已迁移，但由于以上整包编译阻塞，本轮不能声称它们已经在 cargo 下运行通过。
父 agent 集成后需重跑 `cargo test -p prism_render_app --lib pipeline_state`、`cargo test -p prism_render_app --lib pipeline_cache`、`cargo test -p prism_render_app --lib pipeline_prewarm`，并检查 mesh/direct/bin 回归。
本轮没有执行真实 GPU 或驱动持久化验证，没有改 worker 数量/队列模型，也没有冷热统一完成声明。
