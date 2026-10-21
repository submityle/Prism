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

## 固定槽 toroidal clipmap

`GpuSdfToroidalLookup::new(context, config, brick_size, max_queries,
max_upload_bytes)` 使用 `SdfToroidalClipmap` 的 X/Z 方形窗口和每 mip 精确 Y 平面，
分配固定 modulo slot；mip 是独立 slot 平面。查询按负坐标数学 floor division 得到完整
`[x,y,z,mip]` key，并在读取样本前比较四个字段，因此窗口滚动后物理槽被复用时，旧 key
不能 alias 成新 brick。超出当前 Y 平面、缺失 brick、无效 metadata 和不存在的 mip
统一返回 `Unknown`。

`update_center(grid, center)` 先在 CPU staging 中校验并准备完整双向 delta、上传字节
预算与目标 planner 状态；失败不改变已提交 center、slot metadata 或可见 brick count。
成功后只写发生变化的 key、metadata 与已驻留 brick 的 packed samples。调用者对当前窗口内
的样本编辑/删除，必须通过 `upload_bricks(grid, coords)` 显式刷新；缺失的 brick 会清除
metadata 并使相应槽查询为 `Unknown`。重复坐标、窗口外坐标与超出本次上传预算会在 GPU
queue write 前拒绝。CPU `SparseSdfBrickGrid` 仍是权威数据；此 API 不会监听 grid mutation。

令 `S = 2 × ring_radius + 1`、`L = mip_levels`、`C = max(L × S², 1)`、
`Q = max(max_queries, 1)`、`W = ceil(brick_size³ / 2)`，固定逻辑 buffer 总量为：

```text
16                                    # params
+ 16 × C                              # complete keys
+ 16 × C                              # metadata
+ 4 × C × W                           # packed samples
+ 16 × Q                              # queries
+ 16 × Q                              # results
```

`max_upload_bytes` 单独限制每次中心滚动或显式 dirty 更新所写入的 key、metadata 和
resident sample 字节；clipmap config 的 `max_update_bytes` 约束 CPU delta 规划，两者
不是同一个预算，也都不代表 allocator/RSS 或物理显存峰值。query 是固定槽直接寻址，不做
排序表二分；CPU 侧仍需准备 staging 数据，消费者可在同一 command encoder 内 encode，
生产路径不要求逐 query readback。

2026-10-21 Apple M2/Metal/IntegratedGpu 真实设备回归 2 项通过，包含负坐标与多 mip parity、
小步 ring roll、旧槽 key alias 防护、dirty upload、删除失效及预算/输入错误原子性；WGSL
parse/validation 6 项、`prism_world_gpu --features gpu --lib` 123 项通过。该验证证明固定槽
lookup 的设备 dispatch/readback，不表示已接入产品 runtime 调度或多消费者。

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

本切片交付的是 GPU brick upload/lookup contract，以及由 CPU clipmap delta 驱动的固定槽
toroidal lookup。它尚未把 SDF 接入产品 renderer graph、physics broad phase、GI cone tracing、
nav 查询或产品 runtime clipmap 驱逐/重烘调度；
这些消费者应复用同一 `Unknown` 语义和 CPU grid/clipmap 的版本化输入，不能复制另一份
持久世界状态，也不能把缺 brick 当成空场。
