# GPU 高度场 placement 完整键删除

首次验收日期：2026-10-16；最新复验：2026-10-22。本文件记录 `prism_world_gpu` 中独立的
`heightfield placement deletion` GPU 切片，不表示整个 World 系统或产品运行时已经完成。

## 独立交付提交

实现链由以下独立提交组成：

- `1ebd0674c`：精确二维 heightfield placement GPU 管线；
- `aab96181b`：56-byte 完整键 placement deletion GPU 管线及端到端测试；
- `a53b4fa76`：删除交付的资源、失败事务和测试边界加固；
- `231a9643b`：同步 placement/deletion 验收文档。
- `07fdaac3f`：补齐 admission、零容量和失败事务原子性测试。
- `47ff0cecb`：记录此前的 placement deletion 复验结果。
- `0b93f55ca`：RenderGraph deletion copy、count reset、filter 与 draw lowering。

本文件只记录本切片的实现与验收边界；不把这些提交扩展解释为其他 World 模块或整个
产品运行时的完成证明。

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

历史交付验收通过以下检查：

- `rustfmt --check`：通过；
- `cargo check --offline --manifest-path pkg/prism_world_gpu/Cargo.toml --features terrain_scatter --tests`：通过；
- `cargo clippy --offline --manifest-path pkg/prism_world_gpu/Cargo.toml --features terrain_scatter --all-targets --no-deps -- -D warnings`：通过；
- WGSL/Naga ABI 测试：8/8 通过；
- Metal 真机 placement deletion：3/3 通过，设备为 Apple M2 IntegratedGpu。

可复现命令：

```bash
rustfmt --check pkg/prism_world_gpu/src/heightfield_placement_deletion.rs pkg/prism_world_gpu/tests/heightfield_placement_deletion.rs
cargo check --offline --manifest-path pkg/prism_world_gpu/Cargo.toml --features terrain_scatter --tests
cargo clippy --offline --manifest-path pkg/prism_world_gpu/Cargo.toml --features terrain_scatter --all-targets --no-deps -- -D warnings
cargo test --offline --manifest-path pkg/prism_world_gpu/Cargo.toml --features terrain_scatter --test wgsl_compile -- --nocapture
CARGO_TARGET_DIR=/private/tmp/prism-world-heightfield-placement-deletion-gpu WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline --manifest-path pkg/prism_world_gpu/Cargo.toml --features terrain_scatter --test heightfield_placement_deletion -- --nocapture --test-threads=1
```

物理 GPU 测试要求真实 `IntegratedGpu` 或 `DiscreteGpu`，不会把无 adapter、软件 adapter
或 CPU 路径静默当作通过。历史 Metal 真机验收在 Apple M2 上 3 个测试全部通过。2026-10-16
本轮复验的 rustfmt、`cargo check`、Clippy 和 WGSL/Naga 8/8 均通过；同一 Metal 命令因当前
运行环境返回 `metal found no adapters`，未获得新的物理 GPU 结果，因此不把本轮记为 GPU 通过。
测试对拍 CPU placement key 集合，并覆盖精确删除、重复键去重、不存在键、空删除集合、全量删除、
负坐标完整 i64 键、indirect count 和 compact 输出；另覆盖 admission 失败时旧请求、缓存和上传
统计保持不变，以及零容量构造约束。

## 明确边界

本切片没有修改系统设计总文档、CPU placement deletion 合约或其他 GPU deletion 模块，
也没有接入 runtime persistence writer/reader、scene/editor command、网络复制、跨 GPU
adapter 迁移或产品 renderer draw graph。没有新增 benchmark，因此不宣称吞吐、P99、功耗或
与 UE/其他引擎的性能等价；本切片的 GPU parity 也不等于整个 World 系统完成。

## 2026-10-16 收口复验

本次收口使用隔离的 `CARGO_TARGET_DIR=/private/tmp/prism-world-heightfield-placement-deletion-gpu`，
未触碰工作树中其他模块的已有改动。当前环境的定向结果为：rustfmt 通过，离线 `cargo check`
通过，Clippy（`-D warnings`）通过，WGSL/Naga 测试 8/8 通过。物理 GPU 命令已实际执行，
但当前运行环境没有 Metal adapter，3 个测试均在设备获取阶段返回
`metal found no adapters`；因此本次没有新增物理 GPU 通过证据，历史 Apple M2 3/3 记录仍单独
按历史结果保留。

## 2026-10-21 RenderGraph lowering 补充

`prism_world_gpu::build_placement_render_graph` 的可选 deletion 分支将图顺序声明为：placement
finalize → 复制 16-byte indirect 参数 → 清零输出参数的 `instance_count` → 完整键 deletion
compute → draw。清零是必需步骤：deletion shader 以 atomic add 累加保留实例数，若沿用复制来的
上游 count，结果会重复计数。`PlacementDeletionWork` 显式接收清零 compute stage；其单 binding
WGSL ABI 可由 `PLACEMENT_INDIRECT_COUNT_RESET_WGSL` 获取。

bridge 对 deletion 固定 ABI 做最小容量检查（128-byte params、至少一个 56-byte key 槽、候选容量
对应的 64-byte records 与 4-byte compact slots、16-byte indirect 参数），并校验 draw indirect
offset 的对齐与完整参数范围。Null RHI 回归验证 copy 源/目标、copy → reset → filter 顺序、dispatch
命令与 draw 使用 deletion indirect buffer。

draw 的 native bind group 是 opaque RHI ID；RenderGraph 可以声明 filtered compact 的读依赖和
barrier，但无法反射或强制 bind group 内具体 buffer。因此调用方必须使用 deletion compact buffer
构造 draw bind group。此 lowering 尚不是产品 renderer 注册/录制/提交/retirement 或 scene/ECS 消费
链路的完成证明，也没有新增物理 GPU 或端到端性能证据。

本次复验（2026-10-21）：`cargo test -p prism_world_gpu --features gpu --lib --offline`
通过（124/124），`cargo test -p prism_world_runtime --lib --offline` 通过（183/183），
`cargo test -p prism_world_gpu --features gpu --test wgsl_compile --offline` 通过（7/7），
相关 Rust 文件 `rustfmt --check` 与切片 `git diff --check` 通过。GPU feature 编译成功；
验证包含 Null RHI RenderGraph 录制测试，不包含物理 GPU 执行或端到端 renderer 性能测试。

## 2026-10-22 ABI、物理别名与 native snapshot 加固

本次在已有 lowering 上补齐以下可执行校验，错误在创建图资源之前返回：

- 上游 11 个 buffer 与 deletion 4 个 buffer 的最小 ABI 大小；包括 192-byte source uniform、
  128-byte project/field/deletion uniform、每候选 64-byte records/samples、16-byte attributes、
  4-byte compact，以及完整的 `56 * max(key_count, 1)` 删除键存储。
- 上游 reset 与 deletion count reset 必须各为 `[1, 1, 1]`，避免重复 workgroup 写入竞争。
- `PlacementDeletionRequest` 提供与已上传 uniform 一致的候选数、键数及二维 workgroup 元数据。
  deletion dispatch 必须覆盖候选数、Z 为 1、XY 与元数据相同，且整个含尾部 lane 的矩形不超过
  `2^32` 次调用，防止索引回绕；乘法也做溢出检查。
- 当前固定 indirect ABI 只写 record 0，因此 draw offset 必须为 0；对齐但指向其他 record
  的 offset 同样被拒绝。
- 以物理 `BufferId` 检查同 stream 和跨 stream 别名：任何涉及写入的重名 allocation 被拒绝。
  只读共享必须具有相同物理大小与导入状态，并在图中复用同一 handle，避免隐式重复导入。
  固定每 stream 最多 15 个绑定，检查复杂度为 `O(S log S)`，不扫描 GPU 内容。

`GpuHeightfieldPlacementDeletions::registration_snapshot()` 返回完整上游 snapshot、deletion 的
四个 native buffer、请求元数据、singleton count-reset pipeline/bind group，以及非空请求的
filter stage。元数据直接取自 `write_request` 的同一份已上传请求；snapshot 和直接 `encode`
共用 generation、候选数和 draw args 校验。reset pipeline/bind group 在构造时建立，不在每次
snapshot 时创建；snapshot 仅克隆 native handle，不分配或复制 GPU buffer。

snapshot 的执行顺序为 upstream reset/active stages → copy indirect 16 bytes → count-reset
compute → optional filter。零候选请求没有 active stage，但仍执行 reset/copy/count-reset。
直接 `encode` 保留 `clear_buffer` 路径；两条路径均通过 GPU readback 与 CPU 完整键结果对拍。

snapshot 不冻结队列内存，也不替代资源管理：调用方必须在向同一 producer 写入下一请求前提交
当前请求的所有命令，只在同一 device 上注册 handle，并保留 snapshot/注册直至 GPU 使用完成。
raw RHI 调用方必须保证 `PlacementDeletionRequest` 对应 bind group 的实际 uniform 内容。
bridge 不能反射 opaque bind group，也不能识别同一 native buffer 被注册为不同 `BufferId`
的情况；不能将本次元数据/别名校验解释为对任意调用方资源来源的认证。

### 本次验证结果

```bash
cargo test -p prism_world_runtime --lib --offline -j 2
cargo test -p prism_world_gpu --features gpu --lib --offline -j 2
cargo test -p prism_world_gpu --features terrain_scatter --lib --offline -j 2
cargo test -p prism_world_gpu --features terrain_scatter --test wgsl_compile --offline -j 2
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test -p prism_world_gpu --features terrain_scatter --test heightfield_placement_deletion --offline -j 2 -- --nocapture --test-threads=1
```

结果分别为 runtime 183/183、GPU feature lib 132/132、terrain feature lib 148/148、WGSL 11/11、
Metal 实卡 3/3。新增 bridge 回归覆盖全部 15 个 buffer 的欠容量、完整键容量、dispatch
覆盖/stride/Z/回绕边界、固定 draw offset、重复 reset、同 stream/跨 stream 的 writable 别名矩阵
和只读共享导入。实卡为 Apple M2 `IntegratedGpu`，严格禁止无 adapter 或软件 fallback 静默通过。
本次沙盒内获取 Metal adapter 失败后，获准在沙盒外执行并通过；实卡测试将 dispatch 上限约束为
2，实际运行带无效尾部 lane 的 `[2, 2, 1]`，分别编码 native snapshot 链和直接链，再与 CPU
对拍精确删除、空删除集、全量删除及空请求。相关 Rust 文件定向 rustfmt 与 `git diff --check` 通过。

`cargo clippy -p prism_world_gpu --features terrain_scatter --lib --offline --no-deps -j 2 -- -D warnings`
未通过：已有 project/field/finalize 辅助函数的 3 条参数过多告警、`gpu.rs` 的 2 条大错误枚举告警，
以及 `planet_placement.rs` 的 1 条冗余转换告警。未修改这些其他路径，也未通过关闭 lint 掩盖结果。
本次新增 alias error 的大枚举告警已通过缩小错误布局消除。

这些结果证明 native compute 链和图声明/录制分别通过，并不证明真实 RHI RenderGraph 的
注册、GPU submit、draw bind group 与 retirement 整条链已集成；scene/ECS、产品 renderer、
持久化消费和性能/画质 benchmark 仍不在本切片交付范围内。
