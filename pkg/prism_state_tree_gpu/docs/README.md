# prism_state_tree_gpu

`prism_state_tree_gpu` 是 StateTree 批量 Evaluator 的真实设备 `wgpu` 适配器。它只处理满足 `GpuBatchContract` 的纯数据计算：固定列 SoA 输入、Q16 定点评分、固定宽度输出、PlanRevision 校验、CPU golden parity 和结构化错误回退。

适配器不拥有 StateTree 活跃配置，也不在 GPU 上修改 ECS 结构。设备结果只有在序号、Entity、Evaluator、PlanRevision、状态编码和 CPU golden 全部一致时才被接受；否则调用方继续使用 ECS CPU reference。

验证：

- `cargo test -p prism_state_tree_gpu --offline -- --nocapture`
- `PRISM_REQUIRE_GPU=1 cargo test -p prism_state_tree_gpu --test real_device_parity -- --nocapture`
- `PRISM_REQUIRE_GPU=1 cargo test -p prism_state_tree_gpu --test scale_report -- --nocapture`
- Apple M2 / Metal：8 行及 signed/saturated parity 通过；4,096 行端到端报告 P50 `3,075,166 ns`、P95 `5,436,500 ns`、P99 `5,836,333 ns`。
- 默认模式在无 native adapter 时集成测试安全跳过，并输出明确原因；设置 `PRISM_REQUIRE_GPU=1` 后无设备会直接失败，因此不会把 skip 计为 GPU 证据。
