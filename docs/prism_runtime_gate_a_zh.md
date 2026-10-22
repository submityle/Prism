# Prism Runtime Gate A：无 Bevy 的 headless native launcher

本切片把 Gate A 的第一个可独立验证入口落到 `pkg/prism_runtime`。它不是
Bevy 根 package 的别名，也不依赖 `bevy_*` crate，而是由以下真实组件组成：

```text
prism_runtime
  -> prism_app::App
  -> prism_render_app::PrismRenderPlugin
  -> prism_render_driver_null::NullInstance
```

`run_headless` 创建 `App`，以 `RenderAppConfig` 配置 offscreen target，安装
null RHI 实例，然后执行有界的 `App::update` 循环。每次 update 都经过主世界
到 render world 的 `RenderIntent` 抽取和 render schedule；返回的
`HeadlessFrameReport` 直接复制 `RenderFrameStats` 与最终 `RenderRecord`，因此
调用方可以验证抽取帧数、提交帧数、失败帧数和跳过帧数，而不是由 launcher 猜测
或填充成功结果。

Gate A 直接执行 PrismRenderPlugin 注册的完整 RenderStartup 路径。粒子桥接的
异步 build queue 在 render world 创建时注册，因此 startup 不需要 launcher 侧绕过、
替换或伪造。核心 render startup、抽取、frame schedule、frame graph 记录和 null
queue 提交均由生产插件路径执行；本切片用 null backend 验证调度合同，不声称已经
完成硬件 GPU 粒子效果验证。

根 workspace 的 `default-members` 现在只包含 `pkg/prism_runtime`。根 package
仍然保留，完整 workspace 成员也仍然保留；该设置只改变不带 `-p` 的默认 Cargo
选择，使 Gate A 的默认命令落到无 Bevy 的 launcher。

验证命令：

```text
cargo check -p prism_runtime
cargo test -p prism_runtime
cargo run -p prism_runtime
```

`cargo test -p prism_runtime` 中的 launcher 测试使用 null backend，并要求
两个 frame 均完成抽取与提交，同时要求失败和跳过计数为零。硬件 GPU 验证不属于
本切片；null backend 验证的是相同 Prism app/render/RHI 合同的无设备执行路径。

## 当前工作树复核（2026-10-23）

在当前工作树上，使用独立临时 target、单并发、关闭增量编译和调试信息执行：

```text
CARGO_TARGET_DIR=/private/tmp/prism-runtime-check-20261023 \
CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 \
CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 \
cargo test --offline --locked -p prism_runtime --lib
```

结果为 **1 passed / 0 failed**。此前 `prism_runtime` 的编译检查也以相同独立 target 通过；本次测试结束后已删除临时 target。该结果只证明 runtime 库测试与无设备 Gate A 入口可编译，不扩展为全 workspace、物理 GPU 或整帧生产渲染验收。
