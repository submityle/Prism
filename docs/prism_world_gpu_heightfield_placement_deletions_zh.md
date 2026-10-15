# GPU 高度场 placement 完整键删除

验收日期：2026-10-16。本文件记录 `prism_world_gpu` 中独立的
`heightfield placement deletion` GPU 切片，不表示整个 World 系统或产品运行时已经完成。

## 已交付范围

`GpuHeightfieldPlacementDeletions` 自己拥有一个 `GpuHeightfieldPlacement` 上游对象，
先执行二维 XZ placement、精确高度投影和作者场，再在 GPU 上对最终 placement compact
结果执行完整键删除。删除 pass 不重写上游 `records`、`surface_samples` 或 `attributes`；
只产生新的 `compact_indices()` 和 `indirect_args()`。生产路径不把候选点读回 CPU，也不在
CPU 上逐点重建或筛选结果。

删除键是 `HeightfieldPlacementKey` 的完整身份，固定为 14 个 `u32`、56 字节：

```text
placement_version
source.kernel_version
source.seed (u64)
source.version
source.salt (u64)
source.grid_spacing_mm
source.grid.x (i64)
source.grid.y (i64)
source.grid.z (i64)
```

键集合沿用 CPU `HeightfieldPlacementDeletionSet` 的升序、去重结果，因此 shader 可以对
有序键执行二分查找。签名包含完整 `HeightfieldPlacementCacheKey`、完整删除集合、CPU/GPU
算法版本、键 ABI 版本和 draw args；不存在于当前输出的删除键也保留为缓存依赖。

## GPU 资源与生命周期

构造函数为：

```text
GpuHeightfieldPlacementDeletions::new(ctx, placement, capacity)
```

`placement` 必须由同一 `GpuContext` 创建；不同逻辑 device 即使底层适配器名称相同也会
返回 `DeviceMismatch`。固定资源包括：

- 56 字节乘 `max_keys` 的 deletion key buffer；
- 4 字节乘 placement capacity 的 deletion compact buffer；
- 128 字节参数 uniform；
- 16 字节 `[vertex_count, instance_count, first_vertex, first_instance]` indirect buffer；
- 一次性保留的 host key packing scratch。

构造前检查总 buffer budget、每个 storage/uniform/buffer size、binding 数、workgroup 大小、
workgroup storage 和 dispatch 上限。零容量仍分配一个不可寻址物理槽，但逻辑容量仍为零。
请求写入前检查 placement 上游的所有工作/覆盖/容量条件、删除键容量、输入槽预算、键预算、
二分比较预算和 scratch 预算；任何失败都不提交 queue upload，也保留上一次成功请求。

`write_request` 成功后必须先提交当前 encoder，再写入同一对象的下一请求。`encode` 会：

1. 编码 placement 上游四个 compute 阶段；
2. 复制上游 indirect draw args；
3. 清零 deletion indirect 的 instance count；
4. 以 64-lane workgroup 对上游 compact 前缀做完整键二分查找和 workgroup scan；
5. 输出新的 compact 前缀和 deletion indirect count。

删除 shader 只使用 placement record 的 source grid `words[0..5]` 加请求 namespace 组装完整
键。所有 barrier 路径包含尾部无效 lane，避免越界 lane 提前退出造成未定义同步。

## 验证

静态检查、WGSL/Naga ABI 检查和 placement deletion 物理 GPU 测试分别运行：

```bash
cargo check -p prism_world_gpu --features terrain_scatter --tests
cargo test -p prism_world_gpu --features terrain_scatter --test wgsl_compile --no-fail-fast
WGPU_BACKEND=metal cargo test -p prism_world_gpu --features terrain_scatter --test heightfield_placement_deletion -- --nocapture
```

物理 GPU 测试要求真实 `IntegratedGpu` 或 `DiscreteGpu`，不会把无 adapter、软件 adapter
或 CPU 路径静默当作通过。测试对拍 CPU placement key 集合，并覆盖精确删除、重复键去重、
不存在键、空删除集合、全量删除、负坐标完整 i64 键、indirect count 和 compact 输出。

## 明确边界

本切片没有修改系统设计总文档、CPU placement deletion 合约或其他 GPU deletion 模块，
也没有接入 runtime persistence writer/reader、scene/editor command、网络复制、跨 GPU
adapter 迁移或产品 renderer draw graph。没有新增 benchmark，因此不宣称吞吐、P99、功耗或
与 UE/其他引擎的性能等价；本切片的 GPU parity 也不等于整个 World 系统完成。
