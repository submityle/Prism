# prism_asset 运行时资产内核设计

## 状态

本文件是 crate 级入口；引擎级完整规格位于 [`../../../docs/prism_asset_design_zh.md`](../../../docs/prism_asset_design_zh.md)。

当前实现覆盖稳定身份、代际存储、句柄生命周期、依赖图、同步/有限异步加载、热重载、SoftHandle 解析、调度优先级、fallback 和驻留缓存。完整 EDL、GPU feedback/page-table、GPU 解码、远程分发与发行安全链仍为 PLANNED，必须以引擎级文档中的实施状态和测试证据为准。

## 契约边界

- fallback 是展示替代值，不改变真实 `LoadState`，也不伪造 ready 事件；
- retain cache 只接受调用方声明的 payload 字节预算，不声称精确限制整个进程 heap；
- scheduler 优先级作用于排队的 I/O/CPU 工作，不能抢占已经运行的同步读取或解码；
- range streaming 负责有界读取与驻留状态，不等价于 DirectStorage 或 GPU-direct I/O。
