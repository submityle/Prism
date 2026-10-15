# 帧图声明校验与顺序一致性

2026-10-15。本次修复使用现有权威编译内核，不引入第二套调度算法。

声明图首先按显式依赖取得顺序，再生成隐式资源版本及 hazard。合法的前向 pass 引用因此不会因声明数组顺序被误判为资源环；SSA、barrier 与队列等待仍从同一编译结果派生。未受约束的声明保持确定性破平局顺序。

声明准入在注册前完整检查 pass、资源、访问数及依赖容量；失败不残留半注册资源或 pass。循环、非法依赖、队列等待与容量失败均有回归覆盖。

本批使用 `CARGO_INCREMENTAL=0` 验证：

- `cargo test -p prism_render_graph --features parallel`：45 项单元测试通过；1 项 doctest 通过，1 项原有 doctest ignored。
- `cargo test -p prism_render_architecture --lib frame_graph`：8 项通过。
- 五个变更源码文件的 `git diff --check` 通过。

这些结果验证声明编译及其 architecture 适配，不代表完整设备 pipeline 或全部渲染场景的 GPU 验收。
