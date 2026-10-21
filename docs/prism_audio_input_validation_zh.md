# Prism 输入设备闭环验证记录

日期：2026-10-21

本文只记录 `prism_audio_device` 输入控制面的可验证边界；纯逻辑测试、默认跳过的 smoke test 与真实硬件验收分开统计。

## 已验证

- `CpalInputCatalog`：输入端点按稳定 CPAL ID 排序；首次成功轮询只建立 baseline；后续轮询报告 added/removed/default-route delta；枚举失败不提交半成品 baseline。路由通知是显式 polling，不是 OS push notification。
- lifecycle：paused open、`start`、`stop`、重复 start/stop 的状态错误；`Stopped -> Running -> Recovering -> Running/Failed`。每个 stream generation 使用独立 fault latch，状态迁移不依赖 event-log 是否成功入队或是否已消费。
- recovery：固定 ID 与 follow-default 均只在 reopen 时解析；健康 stream 不会因 catalog polling 自动迁移。route lookup、配置、build、play 失败共用固定 3 次预算，耗尽立即进入 terminal `Failed`；recovering 时 stop 会退休旧 stream 并取消恢复意图。旧 generation 的 fault 不会污染新 generation。
- event-log：固定 64 个槽位；满时丢弃新诊断记录并累计 `dropped`。`sequence` 是唯一发号票据，不是时间戳；并发 producer 的入队顺序不保证等于票据顺序。诊断日志不承担状态 replay。捕获环溢出事件只交付一次。
- RT 边界：输入 callback 只做 native PCM 转换、整帧入 ring、原子 fault/error 记录和有界诊断入队；不枚举、不 reopen、不分配、不阻塞。sink 必须保持单一活跃 producer，否则并发 per-sample push 可能交错帧。

## 可复现命令与结果

在低磁盘条件下使用 `CARGO_INCREMENTAL=0` 与测试 debug 信息关闭配置，并串行执行：

```text
CARGO_INCREMENTAL=0 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo test -j1 -p prism_audio_device --lib --tests --no-fail-fast
```

结果：33 个 library tests、11 个 integration tests 通过。`physical_device.rs` 与 `physical_input_device.rs` 在未设置 `PRISM_AUDIO_REQUIRE_DEVICE=1` 时是明确 skip；这种 `ok` 不构成硬件证据。

特性检查：

```text
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 cargo check -j1 -p prism_audio_device --no-default-features
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 cargo check -j1 -p prism_audio_device --all-features
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 cargo check -j1 -p prism_audio_ecs --no-default-features
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 cargo check -j1 -p prism_audio_ecs --all-features
```

结果：设备 crate 两项和 ECS crate 两项均通过。`cargo fmt -p prism_audio_device -p prism_audio_ecs -- --check` 通过。

## 严格硬件证据与未完成项

严格命令：

```text
PRISM_AUDIO_REQUIRE_DEVICE=1 CARGO_INCREMENTAL=0 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo test -j1 -p prism_audio_device --test physical_input_device -- --nocapture
```

当前受限执行环境中的原始结果：在输入枚举阶段失败，错误为：

```text
input enumeration is required: device configuration query failed: An error unknown to the coreaudio-rs API occurred, OSStatus: 560947818
```

随后请求沙箱外复核时，审批服务两次返回 `stream disconnected before completion: The service is busy`，命令未执行。因此当前没有可宣称的真实 callback frame、拔插恢复或默认路由切换证据；该错误也不能单独证明主机没有麦克风。

仍未完成的真实主机 gate：

1. 真实输入 endpoint 的 enumerate/open/play 与 callback frame arrival。
2. 真实 unplug/replug 后的 CPAL error callback、reopen、generation 轮换和 bounded recovery。
3. 真实默认输入切换/设备消失/设备恢复，以及跨平台后端矩阵。
4. OS 原生 push notification；当前契约是显式 polling。
5. recovery 后的时间戳、跨 generation 样本标记、无缝 gap compensation 与 sample-rate conversion；当前仅保留 ring 中已有样本并重新协商 endpoint。
6. `prism_audio_ecs` 尚未接入输入 catalog/capture/recovery；输入控制面仍归 `prism_audio_device`。

输出设备 recovery 的 fake 测试、默认 skip 的物理测试和输入纯状态测试都不能替代上述真实输入硬件 gate。
