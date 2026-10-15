# 可再生散布的完整键持久删除

日期：2026-10-16。实现位于 `prism_world_residency::ScatterDeletionSet` 与
`prism_world_runtime::ScatterDeletionBinding`；核心保持 `no_std + alloc`、无 unsafe。
这是源散布/作者场到既有 override 日志的接缝，并提供地形结果过滤适配；不是完整产品散布验收。
GPU完整键删除及最终compact/indirect消费见`prism_world_gpu_scatter_deletions_zh.md`。

## 身份、权威与恢复

一条删除的身份是 **scene 拥有的作者散布节点 PersistentId + 完整
ScatterCandidateKey**。调用方为该节点选择自己的 `FieldId`，在此字段存整组删除键；
field 也属于运行时快照/缓存绑定。候选键不是 PersistentId，不截成 u128、
不使用碰撞摘要、不持久化 dense slot、GPU compact 序号或实体 ID。

完整键包含 kernel revision、seed、recipe version、salt、grid spacing 与全局
有符号 i64 XYZ 格坐标。修改这些身份参数会得到不同候选；不会自动把旧删除迁移到
新配方。修改最小距离、请求分区、作者场或地形不改变源键。业务若希望“相似位置也
保持删除”，必须显式定义迁移规则，不能默默按邻近位置匹配。

`ScatterDeletionSet` 保存排序去重的不可变 `Arc<Vec<Key>>`。`new` 先检查原始
数量和容量预算、逐键 `rebuild` 验证，再原地排序去重；重复键不能掩盖超预算输入。
`with_deleted(key, bool, budget)` 返回新集合，旧快照不变；无实际变化时仍验证 key，
然后 O(1) Arc clone，不重新对旧分配套用新预算。实际编辑分配新 Vec，预算只约束
新结果，不是新旧快照共存的峰值。

`ScatterDeletionBinding::delta` 为非空集合生成 `OverrideValue::Bytes`，空集合
生成 `Delete`，再由 `Timeline::commit` 捕获字段真实旧值作为 inverse。不要自己
假定旧值为空或拼一份伪 inverse。正常 read 中缺字段/已删除字段表示空集合；字段
类型错误、损坏数据、未知 revision、无效 key 或超预算都返回错误，不能当空集合
继续绘制而让已删除对象复活。清空集合可撤销；undo/redo 也是需持久写入的事务。

驱逐时丢弃派生点缓冲，重新进入时按原配方生成，再读取恢复的权威字段过滤。
存档不保留派生 points/attributes。测试真实关闭并重开 `FileJournal`，回放后重新
生成源，再验证删除、撤销、重做分别产生相应结果。

## 规范编码 revision 1

所有整数 little-endian，禁止直接序列化 Rust struct 布局。头部固定 16B：

| 绝对字节 offset | 长度 | 内容 |
|---:|---:|---|
| 0 | 4 | ASCII `PSDL` |
| 4 | 4 | 格式 revision，当前 1 |
| 8 | 8 | key 数量 u64 |
| 16 | `52 × D` | 严格递增的完整 key 记录 |

每条记录固定 52B：

| 记录内 offset | 长度 | 内容 |
|---:|---:|---|
| 0 | 4 | kernel_version u32 |
| 4 | 8 | seed u64 |
| 12 | 4 | recipe version u32 |
| 16 | 8 | salt u64 |
| 24 | 4 | grid_spacing_mm u32 |
| 28 | 8 | grid.x i64 |
| 36 | 8 | grid.y i64 |
| 44 | 8 | grid.z i64 |

排序采用 `ScatterCandidateKey` 的 Rust 结构数值 Ord（字段按上表顺序，XYZ
有符号比较），不是编码字节的字典序。decode 不替调用方规范化重复/乱序；严格
拒绝非 canonical 输入、截断和尾随数据。先检查 wire 字节预算、头部、checked
count/总长度、key 数量与 Rust 载荷预算，再 fallible reserve；实际 capacity
再次检查后解码，最后完整验证键。空集合的独立编码仍是 16B；空 binding delta
无需编码、即使 wire 预算为 0 也可移除字段。wire 版本与 kernel 版本各自校验。

这是存放于既有持久化 codec 内的字段格式，不额外提供 checksum；日志/快照的
既有封装负责其完整性。有效的其他 canonical 字节可以代表另一组合法键，不能把
字段校验当作认证或数字签名。单 bit 变异回归要求“明确拒绝，或重新编码完全相同
且所有键有效”，没有宣称能检测所有合法数据替换。

## CPU 过滤、缓存与资源

`filter_source` 对已有 `ScatterOutput` 原地稳定 retain；`filter_field` 先拒绝
points/attributes 数量不一致，然后同步原地压紧两列。位置、完整 key、源 cache
key、工作计数、两列指针及容量保持原语义，仅移除命中行。不补点、不重新选择
源候选，故保留源最小距离与分区独立性。删除集合可能包含当前请求之外或先前已被
密度场剔除的键；完整集合仍是缓存依赖，不能只记录当前 `removed` 或幸存输出。

residency 返回 `DeletionFilteredScatter` / `DeletionFilteredFieldScatter`，保留
完整删除集合。runtime 的 `PersistedScatterOutput` / `PersistedFieldScatterOutput`
另保留完整 `ScatterDeletionSnapshot`（作者节点、field、全部键）。内层源 cache
key **只描述上游生成**；下游缓存必须同时比较 snapshot。即使当前 points 完全
相同，不同请求外删除键或不同作者绑定也不等价。比较是完整结构比较，不以摘要
或 Arc 地址代替正确性。

启用`prism_world_terrain/scatter`后，`filter_heightfield_scatter`接收已有
`HeightfieldScatterOutput`，先拒绝points/attributes不对齐，再稳定原地压紧两列。
返回`DeletionFilteredHeightfieldScatter { source, deletions, removed }`，保留完整
地形/作者场/源cache key、上游工作计数和两列指针/容量；完整集合包括请求外和已被
上游剔除的键，始终为额外cache依赖。它只删不投影/补点，不重新烘焙或验证手工
构造的上游cache，不引入runtime/persist依赖。传入source按值消费，错误也不返回
原对象；misalignment包括空删除和单列为空的情况。其O(P log(D+1))过滤不分配
points/attributes，删除快照仅Arc clone；P为输入幸存行数。

2026-10-16正常workspace、offline/locked、独立target验证：terrain no-default
78项、no-default+scatter 109项全部通过，含新增16项删除wrapper测试，两配置
all-target Clippy `-D warnings`通过。新增测试覆盖公开bake/生成链、完整缓存所有
payload/version及请求外键、重新烘焙快照、分区/最小距离/位置、两列顺序与capacity、
全部移除、空输入和误对齐；不据此声称mesh feature或全workspace当前通过。

令 D 为删除数、N 为输入 points：

- 构建最坏 O(D log D)，每键 rebuild；单次 membership O(log D)。
- 实际单键编辑需 O(D) 拷贝；clone O(1)。过滤 O(N log D)、不创建新 point/attr Vec。
- codec 编码 O(D)；decode 逐键检查 canonical 后仍复用统一构造器验证/排序，
  不再分配第二份 key Vec，不承诺任意排序实现的线性最坏界。
- `max_keys` 限制输入/编辑后的 key 数量；`max_key_bytes` 使用实际
  `capacity × size_of::<ScatterCandidateKey>()`，不使用 wire 的 52B。当前
  aarch64 构建 Key 为 56B；编码长度为 `16 + 52D`，wire 预算独立。
- Vec reservation 可恢复报错；Arc 创建沿用分配器普通 OOM 行为。预算不含
  Arc/Vec 头、分配器元数据、保留历史、旧/新快照并存或 RSS。源码点缓冲的原
  容量不会因删除而缩小，也不能把输出长度变短算作实际释放内存。
- Timeline / log / WorldState / inverse 持有的整字段 Bytes 有拷贝及累计历史成本。
  这是低频编辑的全快照桥接，不是百万删除的增量 tombstone 数据库；调用方需管理
  历史、checkpoint 和保留窗口，codec 单字段预算不约束累计历史内存或日志大小。

## 合并与提交边界

该字段使用既有 **字段级 LWW**：两端各自从旧快照删除不同候选后提交完整集合，
获胜事务替换整个字段，不自动合并集合，可能使另一端的删除丢失/复活。这里没有
逐键 OR-set、因果 remove-wins 或多写者 union 承诺。需要并发编辑时，由项目
序列化这类编辑或另行提供符合其撤销语义的逐键协议；不能把 LWW 标成已完成该协议。

`Timeline::commit` 只更新内存；`FileJournal::append` 才执行日志写入和同步。
先 commit 再 append 并非崩溃原子双状态事务，append 失败不能继续发布该内存结果。
示例 `examples/scatter_deletions.rs` 为完整演示，clone Timeline 后在 staging
上 commit，append 成功后才替换已发布 Timeline。该做法以整份 clone 成本换取
清楚的失败发布边界，不是产品增量事务算法。磁盘成功后进程退出可重放恢复；写入
失败/poison 时停止发布，销毁句柄并从磁盘恢复后再决定重试，不能假定失败必定代表
磁盘完全没写入。落盘与内存替换不是跨资源原子指令；若使用已有 snapshot，仍需
遵守对应 sequence 与日志 suffix 的恢复规则。

示例保留进程内的 Timeline 撤销栈，重开日志后对比恢复的 WorldState；它不声称从
日志重建了重启后仍可编辑的 undo/redo 历史。已写入的撤销/重做事务本身可以回放。

运行示例使用自己创建的临时目录，验证后清理自己的日志；它不修改项目存档。
FieldId、节点身份、编辑授权、事务 ID/actor/clock 由调用方提供或分配，基座不
硬编码树木/武器/生命值、固定通道或多人权限。

## 验证与性能证据

2026-10-16 macOS 27.0.1、aarch64、rustc 1.96.1，独立 target：

```text
cargo test --offline -p prism_world_residency --no-default-features -j1
cargo test --offline -p prism_world_runtime -j1
cargo test --offline -p prism_world_runtime --no-default-features -j1
cargo test --offline -p prism_world_runtime --features proxy_host -j1
cargo clippy --offline -p prism_world_runtime --all-targets -j1 -- -D warnings
cargo clippy --offline -p prism_world_runtime --no-default-features --all-targets -j1 -- -D warnings
cargo run --offline -p prism_world_runtime --example scatter_deletions -j1
cargo bench --offline --locked -p prism_world_runtime --bench scatter_deletions -j1
```

这些命令均应附加本批独立 `--target-dir /private/tmp/prism-world-sector-build-EFoWO8`，
避免共享构建目录。residency 为 91 单测 + 1 doctest；runtime 默认 78、no-default
30、proxy_host 98 单测，在并行流送修改之前均 0 失败/忽略。全 feature 在最后三项
回归加入之前曾有 115 项通过。随后默认/no-default 的 all-target 严格 lint 和完整
示例也通过：51 个重生成点，删除/undo/redo 分别移除 1/0/1 点、3 条事务回放成功。
随后曾被并行流送修改中尚未匹配的 `StreamAction::Reload` 阻断；该阻塞已修复。
2026-10-16 使用同一独立 target、`--offline --locked -j1 --lib` 重跑，runtime
默认 **79** 项及全 feature **119** 项全部通过，0 失败/忽略。较早的 feature/lint
结果保留其历史范围，不扩大为当前整个脏工作区或所有配置通过。

回归覆盖 full key 全字段、namespace、任意分区、距离不变、不补点、属性对齐、
spare capacity、无操作 Arc clone、预算/布局溢出、所有截断前缀/尾随字节、
未知格式/kernel、非 canonical、单 bit 变异、真实 inverse 的 undo/redo、snapshot
codec + suffix replay、字段 LWW 非 union，以及真实文件关闭/重开后的重新生成。

release 基准每项 8 预热 + 64 样本，nearest-rank P50/P95/P99 的 1-based
索引为 32/61/64；输入无距离筛选，按原序号每四个删除一个。以下单位 µs：

| 输入 N | 操作 | P50 | P95 | P99 |
|---:|---|---:|---:|---:|
| 10,000 | Set.filter_source | 791.917 | 797.167 | 922.708 |
| 10,000 | encode + decode | 106.000 | 109.209 | 113.917 |
| 100,000 | Set.filter_source | 9758.500 | 9875.125 | 11031.166 |
| 100,000 | encode + decode | 1083.584 | 1092.292 | 1252.833 |

| 输入→输出 | D | key Vec 容量载荷 B | wire B | 原/过滤后 point Vec 容量 B |
|---|---:|---:|---:|---:|
| 10,000→7,500 | 2,500 | 140,000 | 130,016 | 800,000 / 800,000 |
| 100,000→75,000 | 25,000 | 1,400,000 | 1,300,016 | 8,000,000 / 8,000,000 |

过滤计时只含对预生成 buffer 的过滤，不含生成、clone、构造集合、codec、校验和
析构；codec 单独计时，含自身分配/键验证，不含结果销毁。每个预热/样本在计时外
完整验证 key/位置/顺序/元数据及指针/容量。原/解码 key Vec 容量相同。数据是
主机墙钟、非 GPU timestamp、产品帧时、RSS 或无干扰 A/B；10 万点 CPU 路径的
约 10ms 不应被标为 GPU-first 大规模每帧散布已达标。

## 尚须独立交付

产品 scene/ECS/renderer 接线和逐键并发协议仍需独立交付。地形专用wrapper已按
上文交付，GPU删除pass的独立设备与性能证据见`prism_world_gpu_scatter_deletions_zh.md`，
不以既有源/作者场/地形测试替代。这里的真实文件回放同样不是完整产品存档UI或
网络会话验收；GPU只消费CPU恢复的完整集合，不拥有第二份持久权威状态。
