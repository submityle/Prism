# Prism World Runtime SDF Clipmap 调度

本切片把 `prism_world_residency::SdfToroidalClipmap` 接入 runtime 的事务边界。
CPU residency planner 仍是几何权威；runtime 只保存当前窗口的 dirty brick 调度状态，
不复制 SDF 样本，也不直接创建 GPU 资源。

## 契约

`WorldSdfClipmapScheduler` 使用两阶段中心滚动：`plan_center` 先复制 planner 并计算完整
entered/exited delta，`commit` 在 scheduler 身份及 revision 均匹配时一次发布。plan 保留
owner 的 `Arc` 身份，拒绝来自其他 scheduler 的同版本 plan，也不依赖可复用的裸地址。
返回错误或拒绝旧 plan 不会改变当前 center、dirty 列表或 revision。

新进入窗口的坐标自动进入 dirty 列表；离开窗口的坐标从 dirty 列表移除。调用方在完成
SDF 烘焙或 GPU 上传后用 `clear_dirty_at` 确认成功，提交的版本应为工作开始时捕获的
`revision()`；过期完成不会清除后续编辑。`clear_dirty` 只适用于调用方已确保没有异步
版本竞争的同步路径。编辑或仿真影响当前窗口时用 `mark_dirty` 重新排队。所有显式
mark/clear 操作完整验证当前窗口，不能把窗口外坐标误标为可用。

dirty 坐标按 `BrickCoord` 稳定排序、去重；集合成员是幂等的，但每个有效非空 mark 请求
都会推进 revision，即使全部坐标已经标脏。否则旧异步完成可能清掉新编辑。空 mark、未
移除任何成员的 clear、空 take、同中心滚动不推进 revision。版本耗尽时拒绝变更，保留
原状态；同中心 no-op 仍可提交。全局 revision 保守地使其他并行完成失效，不是逐 brick
版本确认协议；调用方应重新获取 dirty 和版本后重试。

`take_dirty` 直接移交列表所有权并推进 revision，不是完成确认队列；处理失败时调用方
必须对仍在窗口内的坐标重新 mark。需要失败后自动保留 pending 的工作流应读取
`dirty_bricks()`，成功后再调用 `clear_dirty_at`。

## 预算与性能边界

配置分别限制 clipmap delta、dirty 坐标数量/逻辑字节，以及单次 mark 请求的原始数量/
逻辑字节。`WorldSdfClipmapConfig::new` 默认让 mark 限制与 dirty 限制一致，可用
`with_mark_request_budget` 独立配置。mark 请求的重复项在去重前照常计费，所有请求预算
在复制输入前检查，最终 dirty 预算在发布前检查。公开配置字面量在 scheduler 创建时
再次验证，不能绕过零字节预算和 clipmap 几何约束。

设原 dirty 数为 D，mark 原始输入为 K、去重后为 U。mark 使用排序去重、二分计数和
有序合并，时间为 `O(K log K + U log(D+1) + D + U)`，不逐元素执行 `Vec::insert`。
逻辑坐标载荷峰值包括旧 dirty、新 dirty、原始请求副本，分别受上述预算约束；不是
allocator 容量、进程 RSS 或所有并行 plan 的总峰值上限。clear 不复制请求，只申请 D
字节的逻辑移除表，重复输入不会放大 scratch；其时间仍为 `O(K log(D+1) + D)`，完整
坐标验证另为 O(K)，没有对任意长 clear 输入承诺恒定耗时。

center plan 仍扫描保留 dirty 并排序合并，包含同中心 no-op 的状态复制，不能视为 O(1)
静止帧接口；调用方可先比较 `center()` 避免无变化时构造 plan。每个 plan 独立持有
prospective dirty/delta，aggregate in-flight 数量/字节由宿主管理。`Vec` 预留使用可失败
分配；owner 的 `Arc::new` 仍遵循标准 allocator 的 OOM 行为，未承诺所有分配失败均返回
可恢复错误。这里没有产品帧时、GPU 时间或物理显存性能结论。

## 验证

```bash
# 每条命令使用相同的独立 target，关闭增量和 debug 信息以限制磁盘占用。
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
export CARGO_TARGET_DIR=/private/tmp/prism-world-runtime-sdf-clipmap-verify-20261021
cargo test --offline -j 1 -p prism_world_runtime --lib
cargo check --offline -j 1 -p prism_world_runtime --no-default-features --lib
cargo test --offline -j 1 -p prism_world_runtime --no-default-features --lib
cargo clippy --offline -j 1 -p prism_world_runtime --lib --no-deps -- -D warnings
cargo test --offline -j 1 -p prism_world_runtime --features asset_container,tasks --lib
cargo test --offline -j 1 -p prism_world_gpu --features gpu --lib
```

单元测试覆盖：首次窗口事务发布、entered/exited dirty 生命周期、旧/外来 plan 拒绝、
重复编辑拒绝旧完成、dirty 和原始请求独立数量/字节限制、非法坐标、零预算、版本溢出、
同中心 no-op、100,000 项重复 clear、16,641 坐标窗口的逆序重复 mark 合并。内存上限和
复杂度来自源码检查，不把结果正确性测试当作 allocation profiler 或性能基准。

2026-10-21 正常共享工作树（基于 HEAD `75c72e9d9`）的独立 target 复验：

| 检查 | 结果与范围 |
|---|---|
| 默认 runtime lib | 123 passed，0 failed/ignored；含本片 14 项专项回归 |
| runtime no-default lib `cargo check` | 通过，验证 no_std 库构建；不等于该配置测试通过 |
| runtime lib Clippy `--no-deps -- -D warnings` | 通过，仅本 crate lib 的严格 lint |
| runtime no-default lib `cargo test` | 编译失败：既有 `hlod_runtime.rs:302–303` 测试缺少 `alloc::vec` 宏导入；未修改该模块 |
| runtime `asset_container,tasks` lib | 依赖 `prism_asset/src/scheduler.rs:806/944/971` 的 `AtomicUsize` 未导入及 `Flight.consumers` 字段不一致，6 个编译错误阻断 |
| GPU feature lib | 同一资产依赖编译错误阻断；本轮没有真实设备 dispatch，也未把历史 GPU 通过记录算成本轮结果 |
| 定向 rustfmt / diff whitespace | 通过；未格式化或提交其他 agent 的文件 |

静态独立复审未发现可确认的新逻辑、预算绕过或失败原子性缺陷；复审提出的新增成员
revision 溢出分支已补回归。测试依赖当时共享工作树内容，不代表干净 checkout 或整个
workspace 通过。本片源码无 `todo!`、`unimplemented!` 或占位实现，产品集成边界如下。

## 边界

本切片不声明已完成 SDF 样本烘焙、GPU buffer 上传、相机系统接线、physics solver、GI、
renderer 或导航消费者接入。调用方仍必须把 `entered`/`exited` 转换为实际 bake/upload/retire
操作，并在成功完成后清除对应 dirty 坐标；失败应保留 dirty 以便重试。
