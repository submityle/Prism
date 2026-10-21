# World Asset-Container 异步 Cell IO

本切片把已验证的 `AssetContainerCellProvider` 接入现有 `CellIo` 异步合同，新增
`AssetContainerCellIo`。它不替换同步 provider；需要同步兼容路径的调用继续使用
`AssetContainerCellProvider`，需要 frame-driven 非阻塞轮询的调用将 adapter 传给
`CellFrameCoordinator::with_io`。

## 接线

```rust
let io = AssetContainerCellIo::new(
    provider,
    named_threads.clone(),
    AssetContainerCellIoLimits::new(4, 64 * 1024 * 1024, 2)?,
);
let mut coordinator = CellFrameCoordinator::with_io(io);
```

`NamedThreads` 由宿主创建并持有；adapter 只向已有 `ThreadCategory::Io` lane
派发工作，不创建逐请求线程，也不负责宿主 lane 的最终 teardown。provider 在 worker
上执行完整的 route、range、header 和 checksum 校验，终态再通过 `CellIo` poll
交给 coordinator，成功 payload 才能进入 `SceneHost`。

## 准入、取消与优先级

- `max_in_flight` 限制所有已接受但尚未消费终态的请求数量。
- `max_declared_bytes` 限制这些请求的 `CellManifest::io_bytes` 总和；成功、失败和
  取消终态都要被消费后才释放配额。
- `max_active` 限制同时派发到共享 IO lane 的 provider 读取数；队列按 live
  `PriorityCell`、请求优先级和提交序号稳定排序。
- 队列中的取消立即变为 `Cancelled`，不会执行 provider read；已派发读取使用原子
  cancellation flag，晚到成功被抑制为取消，底层同步 read 不声称可抢占。
- poll 无终态时注册 waker；worker 完成后在锁外唤醒。adapter 不把 payload 复制到
  第二个完成队列，终态记录持有的声明字节配额直到调用方消费。

## 验证

```text
CARGO_TARGET_DIR=/private/tmp/prism-world-asset-async-20261021 \
cargo test --offline -p prism_world_runtime \
  --features asset_container,tasks --lib
```

当前验证结果：定向 adapter **7 passed**；runtime `asset_container,tasks` 全部
**143 passed**。覆盖 bounded count/bytes、waker、NamedThreads IO lane、队列取消、
派发后取消、provider error identity 和 live priority ordering。

## 明确边界

该 adapter 仍在 host worker 上运行同步的 validated container read；本切片没有把
`ThreadedCellIo`/`ReactorCellIo` 的 range provider 替换成 asset-container provider，
没有实现 mmap schema 结构体视图、GPU direct transfer、GPU-side decode、物理 IO
抢占或底层同步读取中断。`BlockDeflate` 批次内的 decode-once 优化属于
asset-container range assembly 切片，不等同于 GPU 解压。
