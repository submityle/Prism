# prism_state_tree_ecs

StateTree ST1 的最小 ECS 装配 crate。它让共享、不可变 `Plan` 可作为资产被多个运行时实例引用；每个实体组件持有独立且长期存活的 core `Instance`。core `Guard` 要求 `Send + Sync`，所以实例满足 ECS Component 的线程安全边界；每个 tick 直接运行同一 runtime。

典型调用顺序：

1. 编译 `TreeDef` 得到 `Plan` 并包装成 `StateTreeAsset`。
2. 通过 `queue_spawn` 在 ECS `CommandQueue` 中预约一个实例实体，然后应用命令。
3. 调用 `tick_entity` 做单实体初始进入，或调用 `tick_all` 让 ECS query 收集并按 Entity 顺序执行所有实例；之后可通过 `World::get_mut::<StateTreeInstance>` 调用 `request_event`。
4. 下一个 `tick_entity` 才消费事件并应用迁移。

带守卫的迁移可在实例上调用 `set_guard(GuardId, guard)` 安装实现 `Guard + Send + Sync` 的纯谓词；未安装的 guard id 按 core 语义拒绝该迁移。Guard 属于逐实例运行时状态，不会写回共享资产 `Plan`。

迁移属于 core runtime 的数据状态更新，不在迁移回调中改 ECS 结构。调用方需在自己的独占 tick 边界显式调用 `tick_entity` 或 `tick_all`。当前提供 `ValidatedContext`/`validate_context` 作为显式 Schema/Context 校验边界；`tick_all` 已负责 StateTree 实例的 ECS gather，但不负责 Context gather。完整自动 Schedule、typed binding、guard 安装/保留、observer 通知与 asset server 集成尚未实现，见 `docs/version/v0.1.0.md`。
