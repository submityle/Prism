# prism_state_tree_ecs

StateTree ST1 的最小 ECS 装配 crate。它让共享、不可变 `Plan` 可作为资产被多个运行时实例引用；每个实体组件持有独立且长期存活的 core `Instance`。core `Guard` 要求 `Send + Sync`，所以实例满足 ECS Component 的线程安全边界；每个 tick 直接运行同一 runtime。

典型调用顺序：

1. 编译 `TreeDef` 得到 `Plan` 并包装成 `StateTreeAsset`。
2. 通过 `queue_spawn` 在 ECS `CommandQueue` 中预约一个实例实体，然后应用命令。
3. 调用 `tick_entity` 做单实体初始进入，或调用 `tick_all` 让 ECS query 收集并按 Entity 顺序执行所有实例；也可以把 `install_tick_system` 注册到宿主 `Schedule`，每次 schedule run 自动执行一次同一个 gather/tick 路径，并把结果写入 `StateTreeTickReport`；之后可通过 `World::get_mut::<StateTreeInstance>` 调用 `request_event`。
4. 下一个 `tick_entity` 才消费事件并应用迁移。

带守卫的迁移可在实例上调用 `set_guard(GuardId, guard)` 安装实现 `Guard + Send + Sync` 的纯谓词；未安装的 guard id 按 core 语义拒绝该迁移。Guard 属于逐实例运行时状态，不会写回共享资产 `Plan`。

迁移属于 core runtime 的数据状态更新，不在迁移回调中改 ECS 结构。`install_tick_system` 先 rotation 双缓冲生命周期事件，再接通确定性 Schedule 驱动与结构化的最后一次批量报告，并把确定性 Entity 顺序下的生命周期事件发布到 `Events<StateTreeLifecycleNotification>`；事件保留标准两次 update grace window。`StateTreeInstance::drain_lifecycle_events` 仍可供显式 tick 调用者直接消费。`EcsSchema`/`gather_context` 现在提供显式、typed、借用 World 的上下文 gather 和结构化缺失错误。当前接通的是带 payload 的双缓冲 Event stream，不是 `prism_ecs` 无载荷 marker observer；typed binding、自动 Context gather 或 asset server 集成仍未接通。`ValidatedContext`/`validate_context` 仍是显式 Schema/Context 校验边界，见 `docs/version/v0.1.0.md`。
