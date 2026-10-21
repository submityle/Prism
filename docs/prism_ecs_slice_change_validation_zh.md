# ECS column slice 变更屏障验证记录

## 1. 范围与结论

本文记录 2026-10-21 当前工作树对 column-slice change barrier / public extension 契约的验证结果。它只覆盖本轮 slice barrier 实现和相关测试，不是整个 ECS 或 Prism 系统的完成报告。

已关闭的验证范围：外部 ColumnSliceData 扩展、顺序/并行 slice 交付、三层变更屏障、未消费尾部、panic 后交付集合、结构迁移、SparseSet 对照、tick 回绕、observer/reaction、journal gap、restore、批量记录，以及最大切片大小的算术回归。

仍未由本文证明：任意第三方 unsafe slice 实现、所有线程交错、完整 World tick 物理穿越 u32 回绕、GPU/NUMA/跨 crate 性能、生产规模和 wall-clock 性能门槛。本文不等于 ECS 全面完成；主设计文档中的 dirty-query 严格变化量复杂度、真实 GPU/NUMA、跨机网络以及完整编辑器/音频工作流仍保持未完成或未验收状态。

## 2. 当前调用路径

| 环节 | 当前事实 | 源码入口 |
| --- | --- | --- |
| 公开扩展 | query 导出 ColumnSliceData、ColumnSliceChanges；外部实现可命名变更上下文 | pkg/prism_ecs/src/query/mod.rs、query/slice.rs |
| 顺序交付 | simd_iter_mut 校验能力，QuerySliceIter::next 限制在当前物理 chunk | pkg/prism_ecs/src/query/state.rs、query/slice_iter.rs |
| 并行交付 | par_chunks_mut 按物理 chunk 建立 batch，run_chunk_batch 调用回调 | pkg/prism_ecs/src/query/par.rs |
| 可变屏障 | 暴露 slice 前写 changed tick、发布 chunk summary，并记录范围实体 | pkg/prism_ecs/src/query/slice.rs |
| 批量记录 | record_batch 对非空批次只获取一次锁；非激活组件不保留事件 | pkg/prism_ecs/src/change_index.rs |
| 下游收集 | observer flush 收集组件变化后结算 reactions | pkg/prism_ecs/src/change_observers.rs、world_reactions.rs |

## 3. 关键不变量

1. ColumnSliceData 是 unsafe 扩展契约；组件 term 必须 table-backed，sparse/shared 在构造 slice 前拒绝。
2. state、archetype、变更上下文必须来自同一 live World；非空范围必须位于单一物理 chunk。
3. tuple 各列必须描述同一实体范围且长度一致；可变 slice 不得重叠或越过生命周期。
4. 可变 slice 在暴露 storage 前必须发布 row tick、chunk summary 和精确实体索引；只读 term 不发布变更。
5. 裸 &mut [T] 的保守变更集合是实际交付给调用方的实体集合；未消费的后续 slice 不应被标脏。
6. panic 不回滚已写入值；已交付 slice 仍属于变更集合；同一 World/task pool 随后必须可复用。
7. DirtyChunk 是粗粒度候选窗口；Changed、observer 和 Reaction 仍需交付精确实体集合。
8. 比较窗口为 (last_run, this_run]，使用 wrapping age；gap 和 restore 必须保守恢复而不能漏报。

## 4. 测试映射与结果

### column_slice_contract

文件：pkg/prism_ecs/tests/column_slice_contract.rs。无 multi_thread 运行 3 项；启用后另有 2 项，全部通过。覆盖外部 term 转发、顺序 noop、顺序 panic、1/4 workers 并行 noop、并行 panic 和未写尾部。

### parallel_change_barriers

文件：pkg/prism_ecs/tests/parallel_change_barriers.rs。multi_thread 下 6 项全部通过。覆盖 table/sparse 串行对照、结构迁移和 swap-remove、串行/并行 Schedule、row panic 后复用、chunk panic 后恢复、tick 回绕。未启用 feature 时的 0 项不计入证据。

### reaction_scale_recovery

文件：pkg/prism_ecs/tests/reaction_scale_recovery.rs。4 项全部通过。覆盖 16,384 entities 的 observer/reaction 规模场景、SparseSet 历史 replay、70,000 次写入造成的 journal gap 恢复、重复 restore 和同 tick 新写入。断言操作计数，不是 wall-clock benchmark。

### 其他回归

ChangeTracker 批量记录与逐行记录结果等价；空批次和未激活组件无事件/dirty revision；超过保留上限时 gap 与下一 tick 恢复通过。顺序 slice 覆盖 K = usize::MAX 跨多个物理 chunk 的算术回归；修复前复现 overflow，修复后通过。

## 5. 已执行命令与结果

本轮使用 Rust 1.96.1 / Cargo 1.96.1，采用 --locked --offline，任务专属 target 为 /private/tmp/prism-ecs-slice-verify.oBXPt8，并设置 CARGO_BUILD_JOBS=2、CARGO_INCREMENTAL=0、关闭测试/开发调试符号。验证结束后已删除该专属 target，释放约 2.6 GiB。

- no-default-features + std 的 column/reaction 集成测试：3 + 4 = 7 项通过。
- multi_thread 下三个集成目标：5 + 6 + 4 = 15 项通过。
- 默认 prism_ecs：库 936 项通过，集成测试和 doctests 通过。
- all-features prism_ecs：库 1,286 项通过，集成测试和 doctests 通过。
- cargo check --no-default-features：通过。

这些结果来自包含其他未提交改动的共享工作树；它们证明当时工作树的上述契约，不等价于一个干净提交的独立重建，也不覆盖其他 crate 的全量验证。

## 6. 限制

本记录没有证明：任意第三方 unsafe ColumnSliceData 实现；所有线程交错、完整结构变更竞态和完整 World tick 物理回绕矩阵；strict O(change) 的端到端 dirty-query 到 observer 到 reaction 性能；真实 GPU adapter、GPU SDF sampling/ECS 集成、NUMA node-local page 和受控 P50/P95/P99；跨机网络、NAT/ICE/relay、生产恢复，以及完整编辑器、场景和音频联合工作流。

因此，本文件的通过只表示列切片变更屏障契约在列出的配置和测试场景下通过；整套 ECS 设计的剩余范围仍以 docs/prism_ecs_design_zh.md 第 22 节及第 24.6 节矩阵为准。
