# Prism App / RenderApp 本机验收记录

更新时间：2026-10-16。本文只记录当前工作树在本机可复现的 App 与 RenderApp 证据；不把设计目标、被跳过的硬件测试或单机样本扩展成跨平台结论。`prism_replication`、`prism_asset` 不在本次修改范围内。

## 当前已覆盖

- `prism_app`：无头 Runner、固定步/生命周期、子应用提取与流水线、回滚相关单测均可在 crate 隔离条件下执行。
- `prism_render_app`：null backend 下的 plugin → extract → render schedule → frame graph → submit 闭环，以及资源、队列、管线、恢复和 transform resident 测试均可执行。
- App runner 对拍：同一确定性内容在两个 headless runner 实例中运行，比较生命周期计数、固定步数、overstep 和 trace digest。
- CPU pipeline benchmark：比较 serial、dedicated render 和 task lane；只测合成 CPU 工作负载与调度/流水线指标，不代表 GPU、窗口 present 或游戏内容性能。

## 本机命令与结果

### App 单元/集成测试

```text
CARGO_TARGET_DIR=/private/tmp/prism-app-acceptance \
CARGO_BUILD_JOBS=1 \
cargo test -p prism_app --lib
```

结果：`343 passed, 0 failed`。

### RenderApp 软件/null 路径

```text
CARGO_TARGET_DIR=/private/tmp/prism-render-app-acceptance \
CARGO_BUILD_JOBS=1 \
cargo test -p prism_render_app --lib
```

结果：`228 passed, 0 failed, 0 ignored`。该命令默认允许硬件门禁在没有 adapter 时返回，因此不能单独证明真实 GPU 路径。

### RenderApp 强制真实 GPU 门禁

```text
PRISM_REQUIRE_GPU=1 \
CARGO_TARGET_DIR=/private/tmp/prism-render-app-acceptance \
CARGO_BUILD_JOBS=1 \
cargo test -p prism_render_app --lib
```

在沙盒内运行时，16 个硬件门禁因无法访问 adapter 而失败；这证明沙盒不能作为本机 GPU 验收环境。允许沙盒外访问本机图形设备后，Metal/wgpu 路径实际启动，第一次完整门禁为 `227 passed, 1 failed`。修正 recovery 测试中错误的 `frames_before` 采样边界后，硬件门禁仍在 `recovery::hardware_tests::gpu_recovery_rebuilds_resources_and_restores_pixels` 的 material buffer 字节等价断言失败（`hardware_tests.rs:344`）。因此当前真实 GPU recovery 不能宣称完成。

### Headless runner parity

```text
CARGO_TARGET_DIR=/private/tmp/prism-app-acceptance \
CARGO_BUILD_JOBS=1 \
cargo run -p prism_app --features winit,determinism \
  --example runner_parity -- --headless-only
```

结果：`frames=128 fixed_steps=177 final_hash=e679c6d9b125aadd trace_hash=49b156c395f0513f ... parity=true`。

### CPU pipeline benchmark

```text
CARGO_TARGET_DIR=/private/tmp/prism-app-acceptance \
CARGO_BUILD_JOBS=1 \
cargo bench -p prism_app --bench runtime_pipeline \
  --features multi_thread,pipelined-tasks
```

本次机器样本：macOS/aarch64，8 logical workers，240 samples，合成 workload 为每 system `415584` iterations。输出验证 `output_parity=true`、`frame_sequence=true`；dedicated/task lane 的相邻帧 overlap ratio 分别为 `0.997031` / `0.995642`，serial 为 `0`。schedule scaling 的 P50（1/2/4/8 workers）为 `4029209/3100333/2097083/1409292 ns`。benchmark 明确输出 `performance_thresholds=false`，这些数值仅用于当前机器和当前构造负载的回归参考。

## 尚未闭环的缺口

1. `real_window_presentation_acceptance_entry` 仍是 `#[ignore]`，需要真实 OS window 和 display compositor；本次没有伪造窗口验收。
2. `runner_parity` 本次只运行 `--headless-only`，没有执行 Winit 窗口分支。
3. 真实 Metal/wgpu 的普通绘制、材质、mesh、transform 等硬件用例可在沙盒外启动，但 device-loss recovery 的 material buffer 恢复字节对等仍失败；需继续修复后再把 GPU recovery 设为通过。
4. benchmark 没有跨机器、长时间、真实游戏场景、present latency、功耗或 GPU 帧时间门槛；不能据此声称“超过 UE”或达到固定 FPS。
5. 全仓测试不作为本记录门禁；`prism_replication` 的 `security.rs` 重复导入问题由 Dalton 处理，本次未修改或绕过它。

