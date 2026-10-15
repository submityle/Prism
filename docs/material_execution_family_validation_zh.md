# 材质执行族登记校验

`MaterialExecutionTables::register_plan` 保留响应注册表已经选定的执行族，不再把所有 `MaskBlend` 强制改成组合路径。

合法关系：

| 计划 | FusedBuiltin | SplitResponse | Composition |
| --- | --- | --- | --- |
| 单个内建程序 | 接受 | 接受 | 拒绝 |
| 单个注册程序 | 拒绝 | 接受 | 拒绝 |
| 两个内建程序混合 | 接受 | 拒绝 | 接受 |
| 含注册程序的混合 | 拒绝 | 拒绝 | 接受 |

混合计划的 `Composition` 表示两臂独立执行后的组合，不要求联合 package。`SplitResponse` 表示单臂响应执行，不能冒充整个混合计划。登记校验只证明执行描述合法，实际 shader 路由与输出读回由各自的执行测试验证。

验证命令：

```sh
CARGO_INCREMENTAL=0 cargo test -p prism_render_material --lib execution::tests --no-default-features
```

2026-10-15：5 项通过，包含内建混合、合法 `Registered(0)`、非法执行族拒绝和 GPU 描述执行族保留。
