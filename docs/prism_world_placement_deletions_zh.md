# 二维贴地放置的完整键删除与持久恢复

日期：2026-10-16。入口为 `prism_world_terrain/scatter` 的
`HeightfieldPlacementDeletionSet`、`filter_heightfield_placement`，以及显式启用
`prism_world_runtime/placement` 后的 `PlacementDeletionBinding` 和 `PPLD` codec。
默认 runtime 不增加 terrain/GPU 依赖；启用 placement 仍支持 `no_std + alloc`、
禁止 unsafe，不需要 mesh、ECS、文件系统或渲染设备。文件示例另需 `std`。

## 身份与权威

一项持久删除由 scene 作者节点 `PersistentId`、调用方 schema 的 `FieldId` 和完整
`HeightfieldPlacementKey` 定位。placement key 包含 placement 算法版本和完整源
`ScatterCandidateKey`，其中 grid Y 必须为 0。它不是原三维散布 key、投影坐标、
GPU slot、实体 ID 或截断 hash。旧 `PSDL` payload 被明确拒绝，不能默默转换。

构造及编辑逐键检查 placement/source 版本、零间距和源坐标重建溢出；不查询地形，
也不要求键属于当前请求、地形覆盖或最终输出。删除快照可提前覆盖尚未加载的点。
terrain、offset、最小距离和作者场变化不改变同一个源候选身份，旧删除因而继续
生效；seed/version/salt/spacing/grid XZ 或作者 namespace 改变则不会错误相互命中。
语义迁移必须由项目显式定义，基座不做邻近位置匹配。

删除集合只保存规范升序唯一完整键，使用不可变 `Arc<Vec<Key>>` 共享。
`new` 先检查输入去重前数量和实际 capacity，再逐键验证、原地排序去重。
`with_deleted` 验证 key 后，无变化返回 O(1) clone；实际编辑产生受预算约束的新
Vec，保留旧快照。完整集合及算法版本都是额外缓存依赖，不以 Arc 地址、集合 hash
或当前 removed 数代替结构相等。

## 原地过滤与内存

`filter_heightfield_placement` 先检查 points/attributes 长度，包括空输入、空删除
等路径；两列错位明确报错。输入按值消费，失败不返还该对象。成功时同步稳定压紧
两列，保留投影位置、精确三角样本、作者属性、源缓存和全部上游 work counters。
两 Vec 指针及 capacity 不变，全部删除也不缩容。没有重新投影、采样、生成或补点，
不破坏已有最小距离和任意 XZ 分区不变性。

terrain 返回 `DeletionFilteredHeightfieldPlacement { source, deletions, removed }`。
runtime 的 `PersistedHeightfieldPlacementOutput` 另保留完整
`PlacementDeletionSnapshot`：作者节点、field、完整集合、deletion 算法版本。
消费者必须将它与上游 `HeightfieldPlacementCacheKey` 一起比较；当前请求外的删除键
或不同绑定，即使产生相同点，也不是相同缓存。源缓存本身不会被伪装成过滤后缓存。

令 P 为输入放置行数，D 为删除键数：

- 构造 O(D log D)，实际单键编辑 O(D) 拷贝；clone 为 O(1)。
- 先检查列对齐；空删除 O(1) 返回，非空删除 O(P log(D+1))，O(1) 额外空间，
  不分配 points/attributes。
- `max_keys` 限制构造去重前数及实际编辑后数；`max_key_bytes` 限制实际
  `Vec::capacity() * size_of::<HeightfieldPlacementKey>()`，包括空 Vec 的 spare capacity。
- wire 长度与解码 Rust key 容量各自预算，分配前检查所需容量，fallible reserve 后
  再检查实际 capacity。所有计数与长度计算 checked；Vec reservation 错误可恢复。
- Arc 创建使用分配器普通 OOM 行为，不承诺 recoverable OOM。预算不含 Vec/Arc 头、
  分配器元数据、地形/场快照、新旧删除集合共存、保留历史和进程 RSS。
- 点变少不代表释放缓冲内存。Timeline/state/log/inverse 会持有整字段 Bytes，
  各次历史与 clone 成本需另计；这是低频全快照接缝，不是增量 tombstone 数据库。

## `PPLD` revision 1 格式

所有字段 little-endian，绝不直接序列化 Rust struct 布局。头部固定 16 字节：

| 绝对 offset | 字节数 | 内容 |
|---:|---:|---|
| 0 | 4 | ASCII `PPLD` |
| 4 | 4 | codec revision u32，当前为 1 |
| 8 | 8 | key 数量 u64 |
| 16 | `56 * D` | 完整键记录 |

每条完整键固定 56 字节：

| 记录内 offset | 字节数 | 内容 |
|---:|---:|---|
| 0 | 4 | placement_version u32 |
| 4 | 4 | source.kernel_version u32 |
| 8 | 8 | source.seed u64 |
| 16 | 4 | source.version u32 |
| 20 | 8 | source.salt u64 |
| 28 | 4 | source.grid_spacing_mm u32 |
| 32 | 8 | source.grid.x i64 |
| 40 | 8 | source.grid.y i64，必须为 0 |
| 48 | 8 | source.grid.z i64 |

按完整 Rust 数值结构 Ord 严格升序，XYZ 为有符号比较，不按编码字节字典序。
decode 拒绝乱序、重复、截断、尾随字节、未知 codec/placement/source 版本、无效源键
和超预算，任何错误都不能降级成空集合让已删除点复活。

独立空集合编码仍为 16B；binding 对空集合生成 `OverrideValue::Delete`，无 wire
分配，因此允许 wire 预算为 0。正常读缺字段或 Delete 才表示空集合；其它字段类型
返回错误。格式不另加 checksum/认证，日志/快照外层提供其既有完整性保护；合法
payload 替换不能靠字段验证检测，不能将它宣称为数字签名。

codec revision、placement identity revision 和 deletion-filter revision 分开管理；
任一未知身份版本都不自动迁移。即使当前均为 1，也不把它们合并成同一概念。

## 日志、撤销与合并

绑定只生成既有 `OverrideDelta`、读取 `WorldState`，不创建第二份内容数据库。
Timeline commit 从真实前态捕获 inverse；清空删除、undo、redo 都沿用原事务语义。
重新进入世界时可丢弃所有派生 points/attributes，再按原配方生成、读取恢复字段过滤。

此处仍是字段级 LWW：并发提交的整集合取胜者覆盖整字段，不是逐键 union、OR-set
或 remove-wins。并发删除不同点可能使落败一方的删除丢失；项目需串行编辑或提供
明确的逐键协议。该局部接缝不声称完成了多写者删除协议。

`examples/placement_deletions.rs` 在真实非平面 heightfield 上生成 400 点及作者
属性，丢弃派生缓冲，执行删除/undo/redo，各阶段真实关闭并重开 FileJournal、回放、
重新生成并逐行核对两列与完整缓存依赖。示例先 clone Timeline staging，落盘同步成功
后才发布内存状态；这是以全量 clone 换取明确失败发布边界，不是产品增量事务或
跨资源崩溃原子操作。append 失败时停止发布、销毁 poisoned 句柄并从磁盘恢复，
不能假定失败等于磁盘没有完整记录。示例保留进程内 undo 栈，回放只重建 WorldState，
不承诺重启后的可编辑历史。临时目录独占创建，结束只清理自己的日志与空目录。

## 验证复现

使用独立 `--target-dir /private/tmp/prism-placement-deletion-verify-PTYB5U`，
并设置 `CARGO_INCREMENTAL=0`、`-j1`；不复用或清理并行任务的 target：

```text
cargo test --offline --locked -j1 -p prism_world_terrain --no-default-features --features scatter
cargo test --offline --locked -j1 -p prism_world_terrain --no-default-features
cargo test --offline --locked -j1 -p prism_world_runtime --features placement
cargo test --offline --locked -j1 -p prism_world_runtime --no-default-features --features placement
cargo clippy --offline --locked -j1 -p prism_world_terrain --no-default-features --features scatter --all-targets -- -D warnings
cargo clippy --offline --locked -j1 -p prism_world_runtime --features placement --all-targets -- -D warnings
cargo clippy --offline --locked -j1 -p prism_world_runtime --no-default-features --features placement --all-targets -- -D warnings
cargo run --offline --locked -j1 -p prism_world_runtime --features placement --example placement_deletions
cargo bench --offline --locked -j1 -p prism_world_runtime --features placement --bench placement_deletions
```

本切片只交付 CPU 放置删除及权威字段持久恢复，不是 GPU 二维贴地放置、风/交互、
产品 scene decoder/render graph 接线、蓝噪声质量保证或整个 World 系统验收。
runtime 对完整 World Partition 的边界由 `WorldPartitionCapability::require` 明确表达：
当前接受 cell streaming、manifest reload、eviction、trace replay 和 engine-neutral Data Layers admission；Data Layers
只覆盖 cell payload 的 unloaded/loaded/activated 状态、多 layer 交集、确定性排序、驻留驱逐、错误原子性与重新加载，
不覆盖 ECS partition、terrain、GPU、render 或项目层的 layer payload 解析；
HLOD、OFPA、Level Instance、跨 cell 引用、world-origin rebasing、多人 handoff、
跨资源崩溃原子性和完整 World Partition 均返回
`WorldPartitionCapabilityError::Unsupported { capability }`，不会静默降级或伪装已实现。

## 定向验收与测量口径

2026-10-16，正常workspace的定向验证（独立target、offline/locked）：

- terrain no-default：78项库测试；scatter：132项库测试、26项二维放置集成、
  11项索引地形组合集成，全部通过、无失败/忽略。新增删除库回归23项。
- runtime默认79项；启用placement 104项；no-default+placement 55项，全部通过。
  新增持久恢复25项（其中真实文件日志1项仅std）。
- terrain no-default/scatter及runtime默认+placement/no-default+placement四种配置
  all-target Clippy `-D warnings`通过；定向rustfmt、diff whitespace检查通过。
- 完整文件示例实际运行通过：400个重新生成的放置及属性，删除/undo/redo=1/0/1，
  3条持久事务重开回放；不是只验证内存mock或复用旧输出。

release主机墙钟：Apple M2、macOS27.0.1、aarch64-apple-darwin、rustc1.96.1。
真实65²非平面ramp和species刷子，10000/100000整格候选、全密度、水平半径0、
offset7mm；删除0%、按源ordinal每4行1项的25%、100%。每档每操作8预热+64测量，
nearest-rank P50/P95/P99为第32/61/64个样本。六档每轮864次计时调用，全部在
计时外逐完整行核对独立ordinal+BTreeSet oracle、cache/counters及两Vec指针/容量；
每次codec解码后另做一次完整过滤对照。不是只检查数量或抽样hash。

filter不含源生成、clone、集合构造、codec、外部oracle和结果销毁；codec独立计
encode+decode，含wire/key分配及内部身份验证，不含外部核对/解码集过滤/销毁。
不使用GPU timestamp，不宣称零分配实测、峰值内存/RSS、产品帧时或无干扰A/B。

实测两输出Vec总capacity为1840000/18400000B（point176B、attribute8B），
删除全部也不释放；共享heightfield样本8450B和brush载荷、源副本、oracle、
Arc/Vec头、分配器元数据另计。每键Rust capacity64B而wire56B：
10k的0/25/100%集合capacity为0/160000/640000B，wire为16/140016/560016B；
100k分别为0/1600000/6400000B和16/1400016/5600016B。
解码capacity与原集合相同，实测wire Vec capacity与wire长度相同；
codec同时持有原集合、wire和新解码集合，不能把单列预算误称峰值。

最终代码的完整结果（单位µs，每格P50/P95/P99）：

| 输入 | 删除 | 原地filter | encode+decode |
|---:|---:|---:|---:|
| 10000 | 0% | 0.042 / 0.042 / 0.042 | 0.083 / 0.084 / 2.667 |
| 10000 | 25% | 755.667 / 765.833 / 797.083 | 110.458 / 111.500 / 137.375 |
| 10000 | 100% | 831.666 / 966.041 / 1134.125 | 443.625 / 461.292 / 475.125 |
| 100000 | 0% | 0.042 / 0.167 / 0.250 | 0.292 / 0.417 / 7.208 |
| 100000 | 25% | 10103.084 / 13572.792 / 19537.500 | 1233.209 / 2192.375 / 2230.084 |
| 100000 | 100% | 12872.542 / 32825.208 / 115402.916 | 5681.417 / 8748.792 / 9216.750 |

空删除在列对齐检查后O(1)返回，不遍历行；其亚微秒样本接近计时分辨率，
不能据此承诺42ns稳定开销。非空档位仍为有界CPU全量过滤而非每帧增量更新；
特别保留100k全删除P99=115.403ms，不隐藏尾延迟，也不推导实时帧预算承诺。

首轮代码尚未加入空删除快速返回，同样全量校验通过，保留原始结果供追踪：

| 输入 | 删除 | 原地filter | encode+decode |
|---:|---:|---:|---:|
| 10000 | 0% | 12.667 / 29.375 / 59.250 | 0.125 / 0.834 / 4.458 |
| 10000 | 25% | 762.458 / 786.500 / 1209.375 | 110.792 / 111.708 / 112.167 |
| 10000 | 100% | 832.042 / 833.959 / 838.250 | 443.459 / 446.292 / 449.791 |
| 100000 | 0% | 124.917 / 125.625 / 126.125 | 0.292 / 0.541 / 20.375 |
| 100000 | 25% | 9206.833 / 9545.000 / 11650.791 | 1073.125 / 1134.334 / 1260.083 |
| 100000 | 100% | 9617.958 / 10651.292 / 11356.666 | 4261.625 / 4765.000 / 4828.375 |

两轮共1728次计时调用及其全部解码后过滤均通过。第二轮非空尾延迟更高，
未定位其主机调度/系统负载成因；这不是无干扰A/B，不能挑选首轮较小值作为最终验收。
