# World GPU 整数散布生成

日期：2026-10-15

## 契约与权威边界

`prism_world_gpu/gpu` 的 `GpuScatter` 是 `prism_world_residency::scatter` revision 1 的 GPU twin，直接复用 `ScatterConfig`、`ScatterRegion`、`ScatterCacheKey` 与 `estimate_scatter_work`，不另建规则/持久化真相。CPU kernel revision 改变会触发编译期门禁，必须显式复审 GPU 实现，不能以新 cache key 输出旧算法样本。

它在全局三维整数毫米网格每格产生一个 hash-jitter 候选；可选 Matérn type-II 式严格最小距离筛选会重建未请求邻格的原始候选，不递归筛选邻居。此分布不是 Poisson 输入，不保证蓝噪声、最大填充或表面投影。完整 candidate identity 为 namespace 加有符号全局 grid，不是 32/64 位 hash、dense slot 或持久对象 ID。任意切分的半开位置范围唯一归属，不把请求边界当成筛选边界。

WGSL 只使用 `u32` 及 low/high pair 实现 64 位 PCG、wrapping 混合、移位、整数乘法和有符号比较，不要求 shader-i64、浮点位宽扩展或任何 optional device feature。jitter 使用与 CPU 相同的 multiply-high。CPU 预计算全局合法 grid 两端与 `i64::MIN/MAX` 的 Euclidean remainder；GPU 在乘法 wrapping 后按 edge-grid/remainder 判定合法位置，不将越界点夹回世界。距离平方采用精确整数余量比较，严格小于半径才排斥；等于半径允许保留。

## 固定资源与生命周期

```rust
let mut scatter = GpuScatter::new(&context, capacity, vertex_count, buffer_budget)?;
scatter.write_request(config, region, candidate_work_budget)?;
scatter.encode(&mut encoder)?;
// 后续 compute/draw 可读取 scatter.records()/compact_indices()/indirect_args()。
// 先提交本次 encoder，再向同一对象上传/encode 后续请求；不要求 CPU wait。
```

构造时检查 GPU ABI、逻辑 buffer 预算、storage/uniform binding 数/大小、workgroup 大小/共享内存以及二维 dispatch 限制；先校验，后分配。所持逻辑 buffer 字节为 `68×max(capacity,1)+208`：64 字节/primary record、4 字节/compact slot、192 字节 uniform、16 字节 indirect。它不是物理显存、分配器/RSS 峰值或驱动内部资源测量。

`write_request` 先复用 CPU 预检，按 primary grid 数验证固定容量，并按最坏 `N×(2×ceil(radius/spacing)+1)^3` 验证显式候选工作预算。无筛选为 N。容量约束的是所有 primary slot，不是未知的最终 survivor 数；与 CPU 的按实际接受数检查 output budget 不同，GPU 使用保守固定资源上界，不依赖回读后决定是否截断。错误保留上一请求，且不排队任何写入。相同 uniform 位模式不重传；变化只写 192 字节，不生成/上传每实例列。

`encode` 清零 indirect 的 accepted count 后记录生成/筛选/compact。空请求只清 count，不 dispatch。encode 不分配 GPU 资源、不生成 CPU 点、不上传、不 submit/wait/map/readback。每个非空 workgroup 只预留一次全局 compact 区间。二维 workgroup 编号线性化，单轴达到设备上限后可继续第二维；不改变 source slot 顺序。

正常 wgpu 资源 OOM/device loss 仍由宿主处理，不属于返回的可恢复 preflight 错误。队列写入不占据 encoder 中的位置，因此必须先提交当前请求，再上传后续请求；多个不同请求不能在同一对象上仅 encode 后最后一起 submit。

## 可直接消费的 GPU ABI

records 为 64 字节/slot，16 个 little-endian `u32`：

| word | 含义 |
|---|---|
| 0..6 | 有符号 grid X/Y/Z，各 low/high pair |
| 6..12 | 有符号绝对毫米位置 X/Y/Z，各 low/high pair |
| 12 | accepted，0 或 1 |
| 13..15 | 无符号 priority low/high |
| 15 | 零 padding |

primary source slot 为全局 grid X/Y/Z 词典序，Z 最快。越出 i64 世界的候选位置规范化为零、accepted 为零；其他拒绝记录也不能当作样本消费。完整 namespace 从 `cache_key` 获取。

compact 输出唯一 source-slot 索引，跨 workgroup 顺序不保证。只有 indirect `[vertex_count, accepted_count, 0, 0]` 的 accepted_count 前缀有效。records 只有 `candidate_count()` 前缀有效；复用较小/空请求不清尾部。后续 renderer 或 compute 必须经 source slot 访问 record，不可将 compact instance index 当作原始 slot，也不可将 compact 顺序用作稳定身份。

本 kernel 不规定资产种类、地表投影、坡度/高度过滤或 mesh/material；它也不自动写入另一条 CPU `ProxyRegistry`。这些消费/失效策略应在同一 scene 规则派生链上接入，不能用生产回读重建一份权威场景。

## 复现入口

```sh
cargo test --offline -p prism_world_gpu --features gpu --lib -j1
cargo test --offline -p prism_world_gpu --features gpu --test wgsl_compile -j1
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -p prism_world_gpu --features gpu --test scatter_generation -j1 -- --nocapture
cargo clippy --offline -p prism_world_gpu --features gpu --all-targets -j1 -- -D warnings
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo bench --offline -p prism_world_gpu --features gpu --bench scatter_generation -j1
```

基准只测 encode+finish 和 submit+wait 两段主机墙钟，不是 GPU timestamp 或产品帧时；8 次预热、64 次样本、nearest-rank P50/P95/P99。计时外执行首次 192 字节上传及 CPU 精确集合/位置、compact 唯一性、indirect 验证。无距离筛选的 10k/100k/1m 档位与 10k 三维 halo 筛选分开，不把无筛选吞吐推广成密集邻域筛选性能。fixture 为一维排列的三维 jitter 体素，不代表真实地表分布/画质。

### 本轮验证范围

独立 `/private/tmp` workspace harness 以 path dependency 编译真实 world crates，直接注册仓库原始 lib/tests/bench 源码，避免 GPU crate 的 terrain/mesh dev-dependency 扩大到其他并行任务正在改动的 asset/ECS 闭包；无 mock、源码副本或跳过设备。严格 Metal 全套 104 项通过、0 失败/忽略：库 82、scatter 5、density 8、resident 6、parity 1、WGSL 2。no-default 原始库源码另有 73 项通过。harness all-target clippy 和正常 workspace GPU lib clippy 均通过 `-D warnings`；不宣称正常 workspace 全 target 或整仓库已通过。

scatter 回归逐槽比较 grid/position/priority 与完整 CPU 成员集合，覆盖 u32 spacing 极值、负坐标及超出 f64 精确整数范围、i64 两端各八角、严格半径/未请求 blocker、任意分区与逆序请求、0/1/63/64/65/128/129 尾部、无 host wait 连续提交、重复 encode 清 count、固定资源及失败原子性。真实 `draw_indirect` 直接读取 records/compact 并校验 64×64 输出像素。二维 dispatch 与填充矩形超出 2^32 invocation 的拒绝用 synthetic device limit 单测覆盖；受默认 storage binding 大小限制，本次真实设备未跨二维边界。

### 本机 release 实测

2026-10-15，Apple M2 / Metal / IntegratedGpu，未启用 optional device features；其他并行任务运行时测量，保留尾部数据，不作为无干扰 A/B 或跨硬件承诺。每行计时后验证完整 CPU 输出、每个接受 slot 的 grid/位置、compact 唯一性与 indirect count；所有行通过。

| primary / 半径 mm | accepted | 逻辑 buffer 字节 | encode+finish P50/P95/P99 µs | submit+wait P50/P95/P99 µs |
|---|---:|---:|---|---|
| 10,000 / 0 | 10,000 | 680,208 | 52.375 / 111.083 / 1129.875 | 466.667 / 1190.792 / 1900.209 |
| 100,000 / 0 | 100,000 | 6,800,208 | 101.875 / 146.291 / 198.750 | 1076.084 / 1239.791 / 1386.084 |
| 1,000,000 / 0 | 1,000,000 | 68,000,208 | 130.416 / 196.583 / 394.708 | 3572.834 / 4616.916 / 4775.042 |
| 10,000 / 17 | 2,736 | 680,208 | 108.375 / 156.541 / 369.542 | 1519.625 / 1720.250 / 2515.084 |

每个 fixture 首次上传 192 字节，计时内上传 0、回读 0、绘制 0。最小距离档遍历三维 halo，包含请求外原始候选，即使 fixture 的 primary 只沿 X 排列。半径/spacing 大时工作量立方增长，工作预算预检不可绕过；此档结果不保证其他比例的成本。

## 不扩大的结论

候选生成/筛选不等于作者密度场、刷子编辑、地表/SDF 避让、风/交互、HZB、分层 GPU 代理、产品 scene/render graph 全链路，或设计中的蓝噪声与百万散布产品性能验收。此前密度 coverage/dither 是已有 proxy 的显示策略，本 kernel 则产生新的派生候选；两者没有被当作同一能力。
