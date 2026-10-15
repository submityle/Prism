# World GPU 作者密度与种类场

日期：2026-10-15

## 实现与能力边界

`prism_world_gpu/gpu` 的 `GpuScatterField` 组合既有 `GpuScatter` 与作者场过滤 pass。前者生成全局毫米整数候选并作可选最小距离筛选，后者在 GPU 上完成 density/species 采样、独立 rank 选取、compact 与 indirect count；生产路径不在 CPU 上逐点生成、采样或读回。配方复用 `prism_world_residency::ScatterDensityField`，不是另一套 scene 内容或编辑器真相。

CPU revision 1 的噪声、三通道 domain warp、平方距离笔刷、Replace/Add 取整、species 阈值及 rank 规则见 `prism_world_scatter_field_zh.md`。GPU 完整复用这些位模式，无浮点位置、shader-i64 或其他 optional feature。编译期同时固定源 scatter 与 field 算法版本，不能只更新 cache key 而保留旧 shader 算法。rank 与 source slot、请求范围和密度无关；固定 key/seed 的密度选择单调，0/65535 端点精确。

WGSL 用 u32 low/high pair 实现 PCG、signed Euclidean lattice 与 64/32 除法，保留高于 2^31 分母需要的进位；brush 的 `ratio*65535` 用 96-bit 乘积比较生成精确 16-bit 商。warp 先计算原坐标的 floor quotient/remainder，再把有符号 offset 加到 remainder 并修正格坐标，因此跨 i64 边缘时不夹断或绕回位置；只有虚拟格 hash 的明确 2^64 周期。世界两端的刷子差值在缩窄/平方前检查，三轴平方从剩余半径预算中扣减，避免溢出。

该 pass 需要 **5 个 storage binding 和 1 个 uniform binding**，不属于四 storage binding 的 downlevel baseline。原 `GpuContext::try_headless()` 保持四 binding 默认值；调用者必须显式请求：

```rust
let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
    max_storage_buffers_per_shader_stage: 5,
    ..wgpu::Limits::downlevel_defaults()
})?.ok_or("physical GPU required")?;
let mut scatter = GpuScatterField::new(&ctx, capacity, max_brushes, vertex_count, buffer_budget)?;
scatter.write_request(config, region, source_work_budget, &field, field_work_budget)?;
scatter.encode(&mut encoder)?;
// 同一 encoder 的后续 compute/draw 可直接消费下述 buffers。
```

不支持所需 limit 时 device acquisition 或构造器显式失败，不偷偷降级为 CPU、忽略刷子或改变配方。

## 输出 ABI 与消费规则

| buffer | 每 source slot / 固定载荷 | 语义 |
|---|---|---|
| `records()` | 64 字节，16×u32 | 与 `GpuScatter` 完全相同；grid XYZ pairs、position XYZ pairs、**源** accepted、priority pair、padding |
| `attributes()` | 16 字节，4×u32 | UNORM16 density、species、**场筛选后** accepted、独立 u32 rank |
| `compact_indices()` | 4 字节×capacity | 最终 accepted source slot 的唯一前缀；跨 workgroup 顺序不保证 |
| `indirect_args()` | 16 字节 | `[vertex_count, final_accepted_count, 0, 0]` |

`candidate_count()` 是有效 primary slot 前缀，不能当作最终实例数。源拒绝的 slot 属性四字全部为零；场拒绝但源接受的 slot 仍保留有效 sample 与 rank。源记录保持不变，**不能仅凭 records 的源 accepted flag 绘制**；下游必须用本对象的 compact/indirect，再以 source slot 同时索引 records 与 attributes。超出有效 prefix 或 indirect count 的尾部未定义，不得读取其历史值推导成员。

源 pass 后记录 field pass，pass 边界确保读到当前源记录。field 每个 64-lane workgroup 做 inclusive scan，一次原子预留输出区间；包括尾部与矩形 dispatch padding 的全部 invocation 都经过 barrier。每次 encode 先清最终 count，空请求也不保留旧实例数。

## 固定容量、预算与队列生命周期

令 `N=max(capacity,1)`、`M=max(max_brushes,1)`，全部 owned GPU buffer 逻辑字节为：

```text
88*N + 64*M + 352
```

包含源 kernel 的 `68*N+208`、field attributes/compact `20*N`、128 字节 field uniform、64*M 笔刷和16字节最终 indirect。零容量仍保留合法 sentinel buffer。容量覆盖所有 primary slot，不按低密度/源幸存数缩减；全部笔刷必须放入固定容量，不能截断后当成功。构造器先检查复合预算、各 buffer/device limit 与 binding 数，再分配；host 另一次 fallible 预留 `max_brushes` 个64字节刷子 scratch record，实际容量可由 `brush_scratch_bytes()` 查询。逻辑字节不含 pipeline/driver 内存、物理 VRAM、Vec 元数据或 RSS。

`write_request` 在任何 queue write 前检查源配置/范围/源工作量/容量、笔刷容量、`primary_candidates*field.work_per_sample()` checked 乘法与场工作预算以及 dispatch limit。返回的可恢复错误保留完整旧 source/field 请求与输出资源；设备丢失、driver OOM 等仍属于 wgpu 设备错误，不在此原子预检承诺内。

上传为变更驱动：源请求变化最多192字节；field uniform 变化128字节；规范笔刷列表变化64×brush_count字节。相同请求/描述不上传，不变笔刷不随源范围变化重传；空笔刷列表不写 scratch buffer。完整 cache key 包含源完整描述、field 算法版本与不可变配方，inactive 参数也参与失效；caller version 变化可只改变 key 而不改变 shader 数值。配方 clone O(1)，完整比较最坏 O(B)，不是固定 O(1) 的摘要比较。

`encode` 不分配 owned buffer/scratch、不上传、不 submit/wait/map/readback；驱动内部命令和 queue upload 临时存储不在 owned-allocation 承诺内。调用者必须先提交当前 encoder，再更新或 encode 下一个请求；按此顺序连续提交不需要 host wait。不能先记录多个不同请求，随后才一起提交并指望 uniform/brush 自动保留旧版本。

场逻辑预算是保守的 `primary*(1+B+8*noise_enabled+24*warp_enabled)`，不是 GPU 指令数或毫秒限额。源拒绝 slot 会跳过实际 field/rank 计算；每个源幸存者仍线性遍历 B 个刷子，无运行时笔刷空间索引。作者场只作 thinning，不补洞、不重新判断之前的源阻挡关系。

## 验证与复现

```sh
cargo test --offline -p prism_world_gpu --features gpu --test wgsl_compile
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -p prism_world_gpu --features gpu --test scatter_field -j1 -- --nocapture
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo bench --offline -p prism_world_gpu --features gpu --bench scatter_field -j1
```

设备回归严格拒绝没有 adapter/device、CPU adapter、与 `WGPU_BACKEND` 不符的后端，不跳过或伪造 GPU 通过。测试专用 readback 对比 CPU 的完整源成员/grid/position/priority 与每个 source slot 的 sample/rank/final accepted，同时检查 compact 的唯一性、集合与 indirect count；生产 API 没有此读回循环。

2026-10-15，Apple M2 / Metal / IntegratedGpu，无 optional feature，显式五 storage binding：作者场 7 项设备回归通过、0 失败/忽略。覆盖密度端点和完整 candidate namespace、signed/world-edge/u32 极值 noise/warp、精确笔刷衰减/取整/species/稳定排序、最大半径与世界角点差值、尾部/空请求/资源复用、无 host wait 连续提交、预检失败保留旧请求；最后一项在同一 encoder 内以实际 GPU records/attributes/compact/indirect 绘制离屏像素，不由 CPU 构造 draw payload。固定源种子 9 确保半径 2mm 的测试同时包含 source 拒绝、场拒绝与保留点；没有削弱混合集合断言。

完整定向回归 113 项通过、0 失败/忽略：库 84、scatter 5、field 7、density 8、resident 6、parity 1、WGSL 2。no-default 原始库 73 项通过。harness all-target Clippy 与正常 workspace GPU lib Clippy 均通过 `-D warnings`。测试使用 `/private/tmp` 独立 Cargo harness 引用仓库原始源码和 path 依赖，避开 GPU 包 dev-dependency 经 terrain/native Mesh 拉入的无关引擎构建；不是源码副本或替代实现。完整 workspace/多后端未据此验收。二维 dispatch 算术由源 host synthetic-limit 单测覆盖，新 field 测试没有跨实机二维 dispatch 阈值；两个 shader 的矩形尾部与编号防回绕仍分别保留。

## 限定真实 GPU 性能

2026-10-15，Apple M2 / Metal / IntegratedGpu，release，`CARGO_INCREMENTAL=0`，上述原始源码 harness，每行 8 次预热、64 个样本，nearest-rank P50/P95/P99。两个计时窗口均为 host wall clock：`encode+finish` 排除 encoder 创建，`submit+wait` 包含真实队列提交与 native device 等待；均不含 pipeline/buffer 创建、上传、读回、CPU 校验或 draw。初始 queue writes 在预热前提交并等待完成。每次等待/map callback 均有 30 秒硬失败限时，不重试/跳过慢样本。

全部 576 次派发均在计时外与预计算完整 CPU oracle 比对，覆盖每个 source record 的 16 字、每个属性的 4 字、compact 唯一集合和 indirect count。digest 按 source slot 数值顺序生成，不依赖非确定的跨 workgroup compact 次序；它是复现指纹，不是无碰撞身份。

| primary | workload | 最终接受 | encode+finish P50/P95/P99 (µs) | submit+wait P50/P95/P99 (µs) | digest |
|---:|---|---:|---|---|---|
| 10,000 | constant | 4,979 | 19.542 / 22.000 / 24.333 | 521.834 / 539.750 / 547.750 | `5d836e946813393f` |
| 10,000 | noise | 5,083 | 20.250 / 30.125 / 36.500 | 593.792 / 638.166 / 671.416 | `a47ca4005b6defe8` |
| 10,000 | warp | 5,064 | 18.750 / 26.375 / 30.875 | 577.541 / 670.542 / 722.208 | `3d5920f1401954f9` |
| 10,000 | warp+32 brushes | 4,647 | 20.000 / 27.125 / 33.583 | 1066.833 / 1085.584 / 1135.625 | `c580b4405adc6478` |
| 100,000 | constant | 50,038 | 33.584 / 53.917 / 66.625 | 911.333 / 965.750 / 1211.250 | `49b7fa16e4caabd7` |
| 100,000 | noise | 50,909 | 32.375 / 42.625 / 62.833 | 1138.167 / 1193.167 / 1446.500 | `741b77dd7c493cfe` |
| 100,000 | warp | 50,589 | 32.750 / 44.333 / 50.334 | 2195.833 / 5701.417 / 5876.500 | `0315bfe43e2d4987` |
| 100,000 | warp+32 brushes | 46,631 | 28.375 / 46.208 / 57.791 | 6437.792 / 6715.792 / 7302.500 | `e4c1474479449f67` |

三维 fixture 分别是 20×20×25 和 40×50×50 个完整 jittered voxel，spacing=50mm、源 radius=0，所有 primary 都源接受，不是一维条带或半数源点提前拒绝的场成本。源 seed=`0123456789abcdef`、version=1、salt=22；场 seed=`fedcba9876543210`、version=1、base_density=32768、base_species=7。noise 的 cell=311mm、amplitude=24000；warp 的 cell=127mm、amplitude=201mm。32 个同心球均覆盖全部点，center=[0,0,0]、inner=50+ID mm、outer=4000mm、opacity=40000；ID 为 8 的倍数时 Replace(12000+ID×1000)，其余偶数 Add(1000)、奇数 Add(-1000)，species=ID+1。输入逆序，配方按 ID 规范排序，采样实际包含径向衰减与有符号取整。

| primary | brushes | owned GPU buffer 逻辑字节 | host brush scratch 字节 | 首次 request 上传字节 | 验证专用 staging 字节 |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 0 | 880,416 | 0 | 320 | 840,016 |
| 10,000 | 32 | 882,400 | 2,048 | 2,368 | 840,016 |
| 100,000 | 0 | 8,800,416 | 0 | 320 | 8,400,016 |
| 100,000 | 32 | 8,802,400 | 2,048 | 2,368 | 8,400,016 |

构造时另上传两个 indirect 各16字节，因此初始总上传为352/2400字节。验证 staging 只为基准检查而建，不属于生产 pass 的 owned buffer。GPU 逻辑字节不是物理 VRAM，host scratch 不是 RSS。当前 source pass 仍产生自身 compact/count（field 只读其 records），计时包括这部分冗余；未以独立 source-only A/B 证明优化收益。100k warp 的尾延迟也必须保留，不能只挑 P50。此数据不是 GPU timestamp、隔离设备执行时间、产品帧时、蓝噪声/植被画质验收或与历史 CPU fixture 的严格加速比。

## 产品边界

这是可调用的整数 GPU 作者场，不是完整产品散布。尚不包含 scene schema/编辑器工具、笔刷空间索引、地表投影/坡度/高度/SDF 过滤、朝向/缩放生成、风/交互、持久 override 删除、蓝噪声频谱、HZB、impostor/HLOD 或产品 renderer graph 接线。既有 resident coverage/dither 是另一层显示策略；本 pass 的 density 是生成选取参数，不自动变成 raster alpha。局部 parity、像素测试和限定基准不能冒充这些系统或完整画质/帧时目标完成。
