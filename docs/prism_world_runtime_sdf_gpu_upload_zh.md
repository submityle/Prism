# World runtime SDF GPU clipmap 上传

本切片在可选 `gpu_proxy` feature 下提供 `WorldSdfGpuClipmap`，把 runtime dirty 调度器和
`GpuSdfToroidalLookup` 固定槽 ring 作为一个派生缓存 owner 创建及维护。CPU
`SparseSdfBrickGrid` 仍是样本源，Scene/override 权威不转移到 GPU。本对象不写存档。

## 事务与调用顺序

1. `new` 验证 runtime 配置，再以同一 clipmap 几何构造 GPU ring。没有接收外部已滚动
   planner 的接口，也不暴露 scheduler/GPU 的可变引用，避免两份窗口状态分叉。
2. `update_center(grid, center)` 先生成完整 CPU plan，验证窗口、dirty 数量/字节及版本，
   再调用 GPU 上传。GPU backend 在所有返回错误点之前预检和 staging；成功后 runtime
   直接发布已分配的 plan，不再执行可能失败的版本递增或分配。新进入窗口的坐标已经上传
   样本或 Unknown，不重复加入后续上传队列；保留区的编辑继续 dirty，离开区的 dirty 丢弃。
3. resident 样本编辑/删除后，调用方显式 `mark_dirty`。off-window 编辑不必标记，未来进入
   窗口时读取最新 CPU grid。适配器不读取或清空 CPU grid 的 dirty 集合，其他消费者可独立
   使用它。`insert_brick` 本身不等于标记所有消费者的 dirty。
4. `flush_dirty(grid, max_bricks, max_bytes)` 只处理一个同时满足数量和字节限制的有序
   suffix。上传成功才截断该 suffix；失败保留 dirty 和 revision，可重试。之前成功的 batch
   不因后续 batch 失败而回滚。重复编辑仍推进 revision，不误清除新编辑。
5. `submit_queries` 在初始化且无 dirty 时才接受查询。它拥有传入的同设备 encoder，依次
   编码 lookup 和 consumer 回调，再在 owner queue 上立即 submit。因此后续修改不会覆盖
   尚未提交的旧查询。consumer 可以编码 GPU 消费或 readback copy，不能自行 submit 或
   改写 ring 资源；返回的 buffer 不是跨提交不变的快照，需在该回调内消费或复制。

`update_center`/`flush_dirty` 的成功确认仅表示 queue writes 已 staging，并不是 GPU 已完成。
查询提交不等待 GPU，也不执行 CPU readback；产品通过自己的 completion 生命周期管理
结果。异步设备丢失、wgpu validation panic、进程 OOM 不属于可恢复 `Result` 的原子性承诺，
设备恢复时重建派生 owner。CPU 源修改必须由调用方正确标脏；本接口不自动观察对象变化，
也不定义异步 bake 完成协议。

## 预算、效果与性能

每 brick 上传包括 16-byte 完整 key、16-byte metadata；存在样本时另外收费
`4 * ceil(brick_size^3 / 2)` bytes，缺失样本只写 32 bytes，并产生显式 Unknown，不把未知
空间伪造为可通行或空白。奇数样本数包含 word padding。batch 字节上限为本次参数与构造
时 `max_upload_bytes` 的较小值；构造参数也限制单次 center roll，不是全帧累计上限。
第一个 dirty brick 无法容纳则返回明确错误而非无进度成功；空 dirty flush 可以使用零预算。

同中心 update 在检查 grid 几何后 O(1) 返回，不复制 planner、不扫描 dirty、不上传。
dirty batch 选择只读取最多 `max_bricks` 项，直接借用排序 suffix，无坐标副本或确认 scratch；
成功确认是 `Vec::truncate`，避免逐帧小批处理反复移动整个剩余前缀。GPU backend 仍需按 slot
排序和分配 staging，不能把适配器的零坐标副本解释为整条上传链路零分配。suffix 顺序确定，
没有对持续重标高排序坐标的工作负载承诺公平性；宿主应保证足够上传预算或控制编辑负载。

center roll 保留现有 planner/dirty 事务成本及保守容量预检，即使 entered 会在上传后立即
确认，CPU plan 仍要求可容纳其临时 dirty。固定 ring buffer 字节可通过
`allocated_buffer_bytes` 观测；这不包括驱动、queue staging、CPU grid 或其他 owner。没有
基于逻辑字节预算宣称实际显存峰值、P99、产品帧时、跨设备性能或与其他引擎等价。

## 验收范围

CPU oracle 测试验证保留/进入/离开生命周期、无重复边缘上传、最新样本、奇数样本对齐、
删除 Unknown、grid/数量/字节拒绝、后端错误后的状态保留及重试、CPU dirty 所有权和静止
中心 no-op。scheduler 专项回归另外验证版本耗尽及容量失败在上传回调前拒绝，以及 sorted
suffix 仅在成功后确认。真实 GPU 测试入口为 runtime 的 `sdf_clipmap_gpu`，严格要求物理
adapter，不能把无设备或软件 adapter 静默记成通过。

## 本轮验证（2026-10-21）

独立 Cargo target：

```bash
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
export CARGO_TARGET_DIR=/private/tmp/prism-world-runtime-sdf-upload-verify-20261021
cargo test --offline -j1 -p prism_world_runtime --lib
cargo test --offline -j1 -p prism_world_runtime --features gpu_proxy --lib
cargo check --offline -j1 -p prism_world_runtime --no-default-features --lib
cargo clippy --offline -j1 -p prism_world_runtime --lib --no-deps -- -D warnings
cargo clippy --offline -j1 -p prism_world_runtime --features gpu_proxy --lib --test sdf_clipmap_gpu --no-deps -- -D warnings
WGPU_BACKEND=metal PRISM_REQUIRE_GPU=1 cargo test --offline -j1 -p prism_world_runtime --features gpu_proxy --test sdf_clipmap_gpu -- --nocapture --test-threads=1
```

结果：runtime 默认 lib 134 passed；`gpu_proxy` lib 154 passed、7 ignored；no-default
lib check、runtime lib 严格 Clippy 通过。真实 GPU 集成在 Apple M2 / Metal /
IntegratedGpu 上 4/4 通过（测试严格要求硬件 adapter；沙盒内复跑因没有 Metal adapter
而失败，用户授权的跨沙盒复跑成功）。GPU 集成 strict Clippy 首次发现
`err_expect` 测试 lint，已改为 `expect_err`，最终 lint 结果以修正后复跑为准。

共享工作树 `cargo test -p prism_world_gpu --features gpu --lib` 曾被其他 agent 的
`prism_asset` scheduler 工作树内容阻断：仍有构造点引用已移除的 `Flight.consumers`
字段。本切片没有修改该文件。另以已提交基线 `d7eeb443c` 加本切片 runtime 文件建立
临时双 crate workspace，可隔离确认已提交依赖加本切片的 GPU crate 组合；此证据不能替代
共享工作树 GPU crate 全量测试，也不覆盖其他 agent 的未提交内容。

## 产品边界

本切片交付同步 runtime→GPU ring 上传与同提交消费契约，不是整个设计 §17 的完成证明。
相机/焦点系统自动驱动、对象 dirty 传播及 bake 编排、真实 physics solver/GI/renderer/nav
消费者、跨消费者帧预算、设备恢复和产品性能验收仍须对应端到端证据。无需改变权威状态
即可丢弃并重新创建该派生缓存。
