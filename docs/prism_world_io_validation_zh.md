# World IO 独立验证记录

日期：2026-10-15。验证源码基线：world IO 调度修复工作树（随后由独立 commit 固化）。

本文保留最初独立 provider 验证快照。同日后续的有界容量等待、异步 coordinator 到 `SceneHost` 接线、唤醒回归及 release archive 测量见 `prism_world_async_runtime_zh.md`；下文“尚未形成链路”只描述本次历史快照。

本次只验证 `prism_world_io` 调度与 threaded provider，验证后仅整理相关文档，不混入其他任务的暂存内容。

## 环境与命令

工作目录为 Prism 仓库根目录，编译器为 `rustc 1.96.1 (31fca3adb 2026-06-26)`。该 crate 没有第三方依赖，因此直接使用 rustc，未运行 Cargo、未使用任何 Cargo target，也未申请 GPU。

本次使用 `mktemp -d /private/tmp/prism-world-io-check.XXXXXX` 创建的目录为 `/private/tmp/prism-world-io-check.xDZTSS`。以下为实际执行命令；复现时应重新创建独占目录并替换输出路径。

```sh
rustc --test --edition=2024 --cfg 'feature="std"' -D warnings \
  pkg/prism_world_io/src/lib.rs -C debuginfo=0 -C opt-level=0 \
  -o /private/tmp/prism-world-io-check.xDZTSS/io_tests

/private/tmp/prism-world-io-check.xDZTSS/io_tests --test-threads=2

rustc --crate-name prism_world_io --crate-type lib --edition=2024 \
  -D warnings --emit=metadata pkg/prism_world_io/src/lib.rs \
  -o /private/tmp/prism-world-io-check.xDZTSS/io_no_std.rmeta

for prism_io_round in {1..100}; do
  /private/tmp/prism-world-io-check.xDZTSS/io_tests \
    threaded_file_tests:: --test-threads=8 --quiet || exit 1
done
```

最后的重复命令使用 zsh；每轮运行同一组 16 项测试，不是新增 1,600 种测试场景。

## 结果

- std 测试编译和 no_std 元数据编译均通过 `-D warnings`。
- 全部 30 项单元测试通过，0 失败、0 忽略，其中 threaded provider 回归 16 项，新增活动读取 round-robin 与请求 ID 耗尽 fail-closed 回归各 1 项。
- threaded provider 回归以 8 个测试线程重复 100 轮，1,600 次测试执行全部通过。它增加调度交错覆盖，不等同于模型检查或无竞态证明。
- 对本批六个 IO/terrain 源码文件执行 `rustfmt --edition 2024 --config skip_children=true --check` 通过；这是格式检查，不是 terrain 编译证据。

回归覆盖准入与参数检查、精确范围/短读/缺失文件、ACK 停驻与 worker 复用、queued/running/ready 取消、重复 shutdown、last-owner drop、waker 替换/注册时序，以及唤醒与读取回调 panic 后的恢复。

默认 Cargo feature 是 `std`；本次 no_std 编译没有传入该 cfg，只验证禁用 std 后的库元数据构建，不包含 std 文件 provider 或目标平台链接。

## 验证边界与清理

独立 rustc 检查不执行 Cargo feature 解析、workspace lint 配置、clippy、doctest 或依赖集成。此次没有测量吞吐、帧耗时、P50/P95/P99、RSS 或 GPU 指标，也未执行 terrain 测试。统一实现记录中的其他任务测试结果应按其各自命令和执行记录解释，不能归因于本次运行。

固定 worker provider 与 crate 内 scheduler 公平轮询已有上述测试证据，但尚不表示 `FileCellProvider`、world coordinator 和 `prism_tasks` 已形成非阻塞端到端链路；provider 临时背压重试和 runtime adapter 仍需单独验证。请求 ID 已验证不回绕、不复用。

测试二进制与元数据合计约 1.8 MiB；完成后只删除本次独占目录中的这两个可再生成文件并移除空目录，不删除或清理共享 target。本次记录不构成完整 world 系统验收。
