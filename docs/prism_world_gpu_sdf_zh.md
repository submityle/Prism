# Prism World GPU 稀疏 SDF brick lookup

本切片把 `prism_world_residency::SparseSdfBrickGrid` 的只读整数采样接到
`prism_world_gpu` 的真实 `wgpu` compute 路径。CPU sparse grid 仍是唯一数据源；GPU
只持有有界、可丢弃的 sorted brick 视图，不形成第二份世界真相。

## 契约

`GpuSdfLookup::new(context, max_bricks, max_queries, brick_size, mip_levels)` 一次性
分配固定容量资源。上传前检查 grid 的 brick edge/mip 几何、brick 数、sample 数和
设备限制；失败不会更新已上传的 brick count、参数或 GPU buffer。

每个 brick 的完整键为 `[x:i32, y:i32, z:i32, mip:i32]`，按 `BrickCoord` 的
`BTreeMap` 顺序排列。shader 对键执行二分查找，使用数学 floor division 处理负坐标，
sample 仍保持 `z-major, y-next, x-fastest` 的 CPU 布局。两个 `i16` sample 打包到
一个 `u32`，GPU 解包后以 `i32` 输出。

结果 ABI 为四个 `i32`：`[status, value, 0, 0]`。`status=1` 是已驻留的量化值，
`status=0` 是 `Unknown`；缺 brick 和不存在的 mip 数据都不会被伪造为零距离或空场。

固定资源的逻辑字节数为：

```text
16
+ 16 * max(max_bricks, 1)       # keys
+ 16 * max(max_bricks, 1)       # metadata
+ 4  * max(max_bricks, 1) * ceil(brick_size^3 / 2)  # packed samples
+ 16 * max(max_queries, 1)      # queries
+ 16 * max(max_queries, 1)      # results
```

每个 query 的工作量为 `O(log B)`，跨 query 只做固定 1D dispatch；没有 CPU 逐 query
回读。`evaluate_with_readback` 只用于严格 parity harness，生产侧应在同一 encoder
中调用 `encode` 并由消费者直接使用 GPU 结果。

## 验证

CPU 与 ABI 闸门：

```bash
cargo test --offline --manifest-path pkg/prism_world_residency/Cargo.toml --lib
cargo test --offline --manifest-path pkg/prism_world_gpu/Cargo.toml --features gpu --test wgsl_compile
cargo clippy --offline --manifest-path pkg/prism_world_gpu/Cargo.toml --features gpu --all-targets --no-deps -- -D warnings
```

物理设备 parity：

```bash
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 \
  cargo test --offline --manifest-path pkg/prism_world_gpu/Cargo.toml \
  --features gpu --test sdf_gpu -- --nocapture --test-threads=1
```

2026-10-16 在 Apple M2/Metal 上两项测试通过，覆盖负坐标、跨 brick 查询、未知
brick、未知 mip、GPU pipeline 创建、dispatch、packed sample 解包和失败后旧上传
状态保持。当前受限沙盒没有 Metal adapter；该探测失败不被计入通过，真实设备证据
来自跨沙盒 Metal 运行。

## 边界

本切片交付的是 GPU brick upload/lookup contract。它尚未把 SDF 接入产品 renderer
graph、physics broad phase、GI cone tracing、nav 查询或 runtime clipmap 驱逐调度；
这些消费者应复用同一 `Unknown` 语义和 CPU grid/clipmap 的版本化输入，不能复制另一份
持久世界状态，也不能把缺 brick 当成空场。
