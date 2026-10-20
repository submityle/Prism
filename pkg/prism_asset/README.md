# prism_asset

`prism_asset` 是 Prism 的运行时资产内核，负责稳定身份、代际句柄、类型化存储、依赖关系、加载器、AssetServer、热重载和有界流送原语。

当前已落地的运行时切片包括：

- `no_std + alloc` 的 `Asset`、`AssetId`、`Handle`、`Assets`、依赖图与事件；
- `std` 下的同步 AssetServer、VFS、锁外异步加载/重载和 I/O/CPU 调度；
- `LoadPriority`、SoftHandle 批次预算和重复请求合并；
- 类型化 Placeholder/Failure fallback 与显式驻留字节预算；
- `Immediate`、`KeepAlive`、`Lru`、`Pinned` 驻留策略和确定性淘汰；
- 固定页 CPU residency、range assembly 和外部 feedback admission。

离线 Cook、DDC/CAS、发行容器、完整 EDL、GPU 解码和远程分发属于相邻 crate；本 crate 不伪装提供这些能力。

设计与状态依据见 [`docs/prism_asset_design_zh.md`](../../docs/prism_asset_design_zh.md)。
