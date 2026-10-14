# 03：Shader Package 与管线运行时接线记录

日期：2026-10-15。本文记录当前源码和现场测试事实，不修改
`docs/refactoring-dev/` 附件规范，也不把设计文字当成实现证据。

## 当前实现状态

03 的核心身份和运行时骨架已经落地：

- `PipelineIdentity<PipelineState>` 提供
  `execution_family`、`response_module_id`、`permutation_index`、
  `typed_pipeline_state` 四个身份分量；固定功能状态由完整结构体比较，
  `to_hash()` 只用于紧凑编码和线格式。
- `ShaderPackageId` 在冷边界驻留字符串，`ModuleInterner` 为驱动局部模块
  分配数值 `ModuleId`；模块身份绑定 package、代码摘要、入口和 binding ABI。
- `PermutationIndex` 是 checked dense `u64`。`try_new` 和 `new` 都拒绝零轴及
  乘积溢出；质量、mask 和普通数值参数不进入静态置换身份。
- `PipelineCache` 使用有界等待队列和多 worker 编译池；队列满时立即保留
  fallback，不在渲染线程阻塞；panic 会释放编译槽并记录失败。
- `ResponseModuleCache` 已实际创建/复用 RHI shader module，composition kernel
  作为独立常驻模块；custom response 不允许伪装成 fused builtin。
- `PassDescriptorTable`、split 三 pass 形状、direct/bin 共用 pass 查询和未知
  descriptor 的可绘制 fallback 已接入 app；一个 split bin 可以产出多个 work item。
- app 的 `PassDescriptorClass` 已拒绝 `255`，不把描述符表 `u32` 下标写入排序
  键的 `u8` 粗分类。

## 冷层模型与真实接线边界

architecture 提供 `PsoCacheKey`、`WarmSetPlanner`、`WarmRuntime`、
`LruPsoCache`、`DeviceFingerprint` 和 `PersistedPsoCache`。这些接口已经有确定性
生命周期、指纹校验、校验和、LRU 字节预算和 miss feedback 测试。

当前 RHI 没有 driver PSO binary 的 import/export 接口，因此代码不会把
`RenderPipelineId`、WGSL 源码或 pipeline hash 冒充可持久化二进制。持久化格式只
负责可验证的 wire model；真实后端提供 binary seam 后才能恢复 driver blob。
在此限制下，app 的热缓存、异步编译、fallback 和模块驻留是真实运行时路径；冷层
模型的磁盘读写仍由后端消费者负责，不能把模型测试写成跨会话 GPU 命中证明。

## 身份与数据分工

重新编译才需要进入 package/permutation/state identity：响应模块、静态能力、
固定功能状态和声明驱动的置换轴属于身份；材质参数、质量档、连续 mask、光照
选择值、缓存命中状态属于运行时数据。所有同步、异步、fallback、prewarm 和
release 路径都使用同一份 typed identity，运行时句柄不会回写 key。

`asset_refs` 仍是可移植 artifact 的字段；模块句柄、设备指纹、PSO 映射和驱动
二进制只能属于 DriverLocal。这样可以避免把设备/驱动特征泄漏进跨机器 DDC 或
Manifest。

## 验证记录

- `cargo test -q -p prism_render_architecture --lib`：11,273 passed。
- `cargo test -q -p prism_render_app --lib`：149 passed。
- `cargo test -q -p prism_render_scene --lib`：1,729 passed。
- `cargo test -q -p prism_render_architecture shader_package --lib`：86 passed。
- `cargo fmt --all -- --check`：通过。
- `PermutationSpace` 严格构造与置换回归：10 passed。

这些是 CPU/契约层证据，不等同于真实 GPU adapter 通过。GPU 测试必须报告实际
adapter、限制和是否执行了 dispatch；设备能力不足或无 adapter 时只能记为跳过，
不能改写成 GPU 成功。

## 后续接线的验收原则

真实后端 binary seam、完整主 render pass 的硬件 parity、split composition 的
资源依赖执行和跨会话 driver-cache 命中，必须各自提供可运行证据后才可标为完成。
这几项不能用 CPU golden、Naga 编译或 metadata-only artifact 登记替代。
