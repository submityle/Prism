# 03-S2/S3/S5/S6/S7：运行时接线记录

日期：2026-10-14。完整阅读 01、03 规范；不得把后续设计当作已完成实现。本记录不修改 `docs/refactoring-dev/`。

## 当前真实状态

S1 完整 typed fixed-function state 与 S4 有界编译池已经接入实际 PipelineCache，正式 app lib 复验 **146 passed / 0 failed**。
本轮仅新增 `PipelineState::canonical_bytes()` 和一项完整字段变异回归；hash-only 路径没有增加字节缓冲分配，既有线编码和 golden 不变。
S2 尚未换形：当前 key 仍有 `class`/`shader_id`；app 的 mesh/scene/bin 仍消费旧 SpecializationId。
因此 S3/S5/S6/S7 均未完成，不声称已有冷热缓存、验证模块缓存、split GPU 执行或粗分类门禁。
03 强制验收顺序是 S1 → S2 →（S3 ∥ S4）→ S5 → S6 → S7；尚未越过 S2 接缝进入后续切片。

## 已确认可消费的 artifact 接口

主线 architecture 已提供私有构造的 `VerifiedShaderArtifact`，通过正式 Naga SPIR-V parse + Validator 校验；app 将只消费其 `manifest()`、`spirv()`、`digest()`。
`ShaderPackageRegistry::register_verified()` 原子登记代码和 manifest，`verified_artifact()` 可区别正式验证产物与 metadata-only 登记。
包是否 Ready 和 ABI 是否匹配必须另外检查；存在 artifact 不等于当前设备可执行。
app 不修改 artifact、registry、material CompiledEntry 或其 tests；不在 app 提供绕过校验的构造入口。

## S2 所需共享接缝（待主线提供）

- architecture 单一定义 `ModuleId`、`ExecutionFamily`、`PermutationIndex`，支持 Copy/Eq/Ord/Hash；material reexport ExecutionFamily，不能重复定义。
- `ModuleId` 是 DriverLocal 数值句柄，不是可移植 artifact 字段；分配及内容更新不能静默复用仍驻留的旧身份。
- `PermutationIndex` 使用 dense u64；声明驱动的轴、checked encode/decode、checked 总大小与全瓣 fallback 必须拒绝 overflow，不得保留旧 u32 瓣窗口或另加手工位打包。
- material 暴露 ShadingPlan、派生活动瓣/静态能力与 checked permutation 构建接缝；动态 mask、质量档与数值参数不作为静态轴。
- typed package/permutation GPU header 由 material/authoring owner 迁移，app 不写 header ABI 或 schema/codegen。

当前 architecture `PermutationSpace::index_of` 使用 saturating multiply + 普通 add，`total_permutations` 饱和，不能直接作为无碰撞 dense 身份保证；这里仅记录需要修复的上游接缝，没有越权修改。
`ShaderPackageId` 当前仍是冷 String；计划由 app DriverLocal 表在注册/解析时一次 interning，热 key 与热排序只比较数值句柄。此策略无需改 architecture mod.rs。

## app 接线方案（尚未实现）

S2：PipelineKey 只保留 family/module/permutation/typed state；MeshPipelineKey 删除 class+shader_id，descriptor 最终固定功能仍从实际最终 descriptor 提取。
所有 sync/async/fallback/prewarm/release 使用相同身份，不把 runtime shader handle 或解析状态写回 key。

S3：通过 app compiler seam 接 fingerprint、WarmRuntime 与 PersistedPsoCache；导入前校验内容 digest、family、permutation 与完整 canonical state，不能只信 u64 hash。
RHI 当前没有 PSO binary import/export 接口，默认后端应明确 Unsupported，不能把 GPU id 或 WGSL 源码冒充持久化 driver blob。
冷热容量必须由真实 residency/引用计数约束；WarmRuntime 目前没有 forget/invalidate、LruPsoCache 没有 remove/pin，需先决定上游补接口还是 app 生命周期适配，不能只更新模型账本而留下已销毁的 Ready。

S5：验证后的 response module 创建/复用实际 RHI shader module；Composition kernel 独立常驻，不能为 Custom 材质对生成 O(N²) 联合包。
模块缓存 key 绑定 code digest/ABI/入口与后端身份，销毁顺序应先释放依赖 PSO 再销毁模块。

S6：app 拥有 ResolvedMaterialPasses / PassDescriptor 表；每项携带 family、module、permutation、phase、pass role 和显式 fallback。
SceneRegistry 登记构建期解析的 pass 集，direct 与 bin 查询同表；bin 的 u32 pipeline_class 仅表示描述符表索引，不再解码 SpecializationId。
split 一组多项共享几何/间接参数范围；真实 composition 的输入资源、调度依赖与常驻 pass 要接帧图 owner 接缝，三个普通 surface draw 不能冒充组合执行。
越界必须响亮诊断或有完整身份的 direct fallback，不可静默少画。

S7：描述符表 u32 下标与 u8 粗排序类严格分离；255 作为满额门禁，不能截断/回绕表索引。
app 可以构建粗分类；visibility crate 的字段/API 重命名需要另行 owner 授权，本轮不写该 crate。

## 精确写域与冲突边界

本轮已改源码：`pkg/prism_render_app/src/pipeline_state.rs`、`pipeline_state_tests.rs`。
本轮文档：本文，以及 `docs/pipeline_state_implementation_zh.md`、`docs/pipeline_worker_pool_implementation_zh.md`。

后续已授权 app 写域：`pipeline_cache.rs`、`pipeline_state*`、`pipeline_worker_tests.rs`、`mesh_pipeline.rs`、`mesh_pipelines.rs`、`scene.rs`、`queue.rs`、`bin_queue.rs`、`queue_stage.rs` 及相关 app 测试；`lib.rs/startup.rs/systems.rs` 仅必要接线。
不写 architecture/material/authoring/schema/codegen/graph，也不覆盖 page_pool 等无关改动。父线程已有 mesh 修复保持原样，后续身份迁移从当前内容精确修改。没有 commit/push。

## 本轮实测

- `cargo test -p prism_render_app --lib`：146 passed / 0 failed。
- `cargo test -p prism_render_app --lib pipeline_cache -- --test-threads=1`：18 passed / 0 failed。
- 补充 rustc harness（工作区最新 cache/state/worker 源码 + 既有依赖 rlib）：15 passed / 0 failed，仅作隔离验证。
- 指定 state 两文件 rustfmt 成功，canonical 与既有 hash golden 一致。

旧次运行的 render_fabric 缺文件阻塞已在本次正式复验消失。本轮未验证真实 persisted binary 导入、shader artifact GPU 执行或 split composition 结果；这些是后续切片验收，不计入上述已通过项。
