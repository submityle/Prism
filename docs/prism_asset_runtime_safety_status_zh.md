# Prism Asset 运行时安全状态

> 验证日期：2026-10-21。本文只记录 `prism_asset` 的加载、重载、回收与并发安全切片，不代表 `Prism Asset Build` 或完整资产系统已经交付。

## 本批已完成

- 普通异步加载以共享 flight 去重；单个 joiner 取消只终止自身等待，最后一个消费者离开后才把 flight 转入取消。
- 取消与提交通过同一 lifecycle gate 线性化：取消先到则候选不得发布；提交先到则发布可以完成，迟到取消只停止 waiter。
- 排队且尚未开始的工作可短路；同步 source read 与同步 loader decode 不支持执行中强杀，只在阶段边界观察协作取消。
- reload 共用的读取会先完成 drain，再由本地 flight 在 decode 或 commit 前丢弃候选，不会取消其他 revision 的消费者。
- 已取消但仍在清理的同路径 flight 保留 draining tombstone；新请求显式得到 `Cancelling`，避免旧 worker 与新身份并行发布。
- soft batch admission 对新增 root 执行全有或全无的容量检查，并保持同路径去重与类型冲突诊断。
- 失败重试使用新 allocation identity；旧失败句柄及诊断保持可观察，不把 arena 的 Failed 状态伪装回 Loading。
- storage replacement、失败、回收和 server staged candidate 的 payload 均延迟到相关 mutex 释放后析构；waker clone、wake 与 Drop 也不在 scheduler 锁内执行。
- eager removal 按依赖闭包 fixpoint 回收，不依赖单次遍历顺序。
- 同步 load 与异步 reload 的 provisional label/source reservation 都有回滚；factory panic 不再绕过 journal。
- reload factory panic 保留旧 root revision 与既有 label，撤销新依赖和新 label 的 intern、catalog、retained、owner、graph 与 storage 状态。
- panic payload、未发布 candidate 和替换下来的旧 payload 在 server mutex 与 reload epoch gate 外析构；回归覆盖析构重入和单析构 panic。
- 普通首次异步加载会先完整规划 primary 与全部 labeled dependencies，再启动 child I/O：按声明顺序稳定展开、按 `AssetPath` 去重、合并兼容类型与 ancestry，并在调度前拒绝 ancestry cycle 或同路径具体类型冲突。
- 普通首次异步依赖使用 scheduler 容量约束下的增量 ready-wave admission；已准入 siblings 被并发轮询，后声明依赖可以先完成，但最终仍按 primary/label 原始槽位顺序回填。
- async reload 会按 revision-local BFS frontier 先规划整波 dependency source：按首次声明顺序去重，在任何 child I/O 前拒绝同路径类型冲突或 candidate closure 越界。
- reload dependency closure 同样使用 scheduler 容量约束下的增量 ready-wave；siblings 可并发读取并乱序完成，但 decoded outputs 按稳定 planner 槽位归约，完整 candidate 最终仍只原子提交一次。
- 每个唯一 child flight 都继承 parent 的动态优先级；跨 root wait-edge 的 cycle detection 与插边在同一 scheduler 临界区完成，guard 同时校验 parent/child flight identity，避免旧 generation 清理新 generation 的边。
- 任一 child 失败或 parent 取消时，普通首次异步依赖批次立即 detach 全部未完成 tickets，不等待被阻塞 sibling 的 source I/O 排空；共享 child 仅在最后一个消费者离开后取消。
- 任一 reload child 失败或 reload ticket 取消时，当前 wave 会立即 drop 其余 waiter futures；已经进入同步 source read 的 detached 工作仍自行 drain，但不会阻塞 ticket 返回或发布部分 candidate。

## 精确边界

- 当前取消是协作式的阶段边界取消，不是 CPU 抢占、线程终止或 OS I/O 取消。
- `panic=unwind` 下执行 panic 隔离与回滚；`panic=abort`、OOM、进程终止和同一次清理中的多重析构 panic 不在保证范围内。
- 当前 loom 只覆盖抽取出的关键状态机小模型；尚未形成完整 scheduler、AssetServer、source、loader 与 storage 的穷举模型。
- 普通首次异步加载与 async reload dependency closure 均已覆盖有界 ready-wave；loader decode 仍是同步调用，不支持执行中抢占，且尚未形成 decode/upload 分阶段作业与更大规模交错验证。
- 本批证明的是有界进程内运行时安全，不是完整 EDL、GPU 页流送、远程发行或构建农场能力。

## 验证证据

使用独立 target 目录、关闭 incremental 与测试 debuginfo、单作业执行：

```text
cargo test -j 1 --locked -p prism_asset server_panic_drop_tests --quiet
15 passed; 0 failed

cargo test -j 1 --locked -p prism_asset scheduler_cancellation_tests --quiet
27 passed; 0 failed

cargo test -j 1 --locked -p prism_asset --test async_labeled_reload --quiet
34 passed; 0 failed

cargo test -p prism_asset --test scheduler_wait_cycles -- --nocapture
5 passed; 0 failed

cargo test -p prism_asset --lib scheduler::range_tests -- --nocapture
6 passed; 0 failed

cargo test -j 1 --locked -p prism_asset --lib --quiet
383 passed; 0 failed

cargo test -j 1 --locked -p prism_asset --all-targets --quiet
383 + 4 + 34 + 4 + 6 + 1 + 7 + 9 + 14 + 5 passed; 0 failed
```

## 后续未完成项

1. 完整 EDL：阶段化可抢占 decode/upload 作业，以及跨 source/loader/storage 的更大状态空间验证。
2. 完整 labeled revision 生命周期：大规模交错、promotion/retirement 与 watcher 重载压力测试。
3. 压缩容器到 VT/VG 的 feedback、解码、上传、page table、consumer 与 eviction 生产闭环。
4. Prefetch、DirectStorage 形态队列、上传时间预算和真实平台 I/O 后端。
5. 通用材质、场景与编辑器 Bake 消费，以及真实 GPU sampling 验收。
6. 多格式导入、平台 Cook、VT/HLOD/World Partition、动画和音频分块。
7. 动态插件准入、签名、隔离、远程 worker 与 OS 资源限制。
8. 通用字段级失效、远程分层 DDC/CAS、Build Farm 与百万资产 Registry。
9. 完整编辑器 UX，以及 CDN、下载、DLC、加密、密钥托管、远程信任根和发行回滚。
10. fuzz、soak、跨平台、确定性与 P50/P95/P99 性能证据。
