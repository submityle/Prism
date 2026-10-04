# Prism Utils 顶级次世代 AAA 级容器 / 工具 / 分配器基础库设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **数据结构（SmallVec/ArrayVec/竞技场）+ 句柄表（generational slotmap）+ 稀疏集/位集 + 稳定哈希 + 字符串驻留（FName）+ 分配器抽象 + 并发容器** 底层工具库设计。它是 `bevy_utils` / `bevy_ptr` 的自研替代，是 ECS 存储、资产句柄、调度器、诊断缓冲共同站立的「数据结构地基」。
> 借形态不抄码。借鉴：
> - **Rust 生态**：`hashbrown`（SwissTable）、`smallvec`/`arrayvec`、`slotmap`/`slab`（generational index）、`bumpalo`（竞技场）、`bitvec`、`crossbeam`（无锁队列/epoch）
> - **引擎级容器**：Unreal `TArray`/`TMap`/`TSet`/`FName`（字符串驻留）、EASTL（嵌入式友好容器）、Unity DOTS `NativeArray`/`NativeHashMap`（原生内存 + job 安全）
> - **高性能库**：Abseil（SwissTable 原型）、folly（F14/并发）、DOD 稀疏集（EnTT/flecs 的 sparse-set）
> 本文为纯经典数据结构 / 内存管理路线，**不含任何 AI/ML 内容**。

- 版本: v0.2（核心 M0–M6 已落地并验证；§24 高级增补 24.1–24.7 已全部交付；v0.1→v0.2 新增第 24 章「AAA 高级功能增补」：作用域分配器栈与竞技场组合/无锁进阶(hazard pointer·RCU·分片并发哈希)/内存安全加固(canary·poison·debug 校验)/热冷分离与 SoA 自动布局/可重定位内存映射容器(offset 指针)/内容寻址去重缓存/确定性进阶(有序并发·可复现哈希)；24.1–24.7 已全部交付，详见各小节与 §24.8）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: **无 Prism 上游依赖**（与 `prism_math` 并列为依赖图根）；仅经典 crate 级 `hashbrown`(可选内置自有实现)、`libm`；SIMD 哈希/查找走 `core::arch`
- 层级定位: L1 地基（被 `prism_ecs`(存储/原型) / `prism_asset`(句柄) / `prism_diagnostic`(环形缓冲/驻留) / `prism_reflect` / `prism_tasks` 等广泛依赖）
- 明确约束: 核心 `no_std + alloc`，部分容器 `no_std` 零 alloc（栈上 ArrayVec/位集）；`std`（并发原语）/ `concurrent`（无锁容器）/ `determinism`（确定迭代序）/ `serde` / `bytemuck`（POD）/ `alloc-track`（分配埋点）为 feature；**不依赖任何 `bevy_*` crate**

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier / feature）
4. 分层架构
5. 核心容器：SmallVec / ArrayVec / 竞技场（Arena / Bump）
6. 句柄表：Slotmap（generational index）
7. 稀疏集 SparseSet / 位集 BitSet
8. 稳定哈希与 HashMap / HashSet 门面
9. 字符串驻留 / 稳定 ID（FName 形态）
10. 分配器抽象（Allocator trait / 池 / 帧分配器）
11. 并发容器（接 tasks：MPMC 队列 / 并发哈希）
12. no_std + alloc 友好与可移植
13. 确定性容器（确定迭代序，接四方确定性）
14. 序列化 / POD 友好（接 reflect / bytemuck）
15. 与 ECS / asset / tasks / diagnostic 集成
16. 可观测性（容器统计 / 内存剖析）
17. 高级功能增补（AAA）
18. 性能工程
19. 易用性与 Bevy 迁移策略
20. crate 分层与模块布局
21. 契约、不变量与版本化
22. 路线图（M0–M6）与基准即规格
23. 诚实边界与风险
24. AAA 高级功能增补（v0.2）

---

## 1. 设计哲学与目标

引擎的性能，80% 决定在**数据怎么摆、怎么取、怎么分配**上。一个 `HashMap` 的探测策略、一个句柄表的失效检测、一次分配走不走池，被 ECS/资产/调度器亿万次复用，直接决定缓存命中率与帧时间。`prism_utils` 的使命是：提供一套**缓存友好、分配可控、可确定、`no_std` 可移植**的底层数据结构与内存工具，让所有上层 crate 不必各造轮子，且共享同一套性能/确定性保证。

**一句话定位**：`prism_utils` 是 Prism 的「数据结构地基」——SmallVec/竞技场省分配，generational slotmap 做安全句柄，稀疏集/位集服务 ECS，SwissTable 哈希求缓存命中，字符串驻留做稳定 ID，分配器抽象把内存握在手里；核心 `no_std`，确定性档保证迭代序可复现。

四条总目标（按权重）：

1. **性能**：缓存友好布局；SwissTable SIMD 探测；栈上小容量优化省堆分配；池/帧分配器摊薄 malloc；无锁并发容器。
2. **效果（能力）**：句柄表 / 稀疏集 / 位集 / 驻留 / 分配器 / 并发容器齐全，覆盖 ECS/资产/调度/诊断全需求。
3. **易用**：API 贴近 Rust std + 常用 crate（`Vec`/`HashMap` 直觉），`prelude` 一把带入；兼容 `bevy_utils` 常用名。
4. **可移植 + 档位化 + 确定性**：核心 `no_std`；并发/确定性/序列化按 feature；确定性档保证哈希迭代序可复现。

---

## 2. 参考产品取舍

| 来源 | 吸收 | 规避 |
|---|---|---|
| hashbrown / Abseil SwissTable | SIMD 分组探测、高负载因子、缓存友好，Rust 事实标准 | 默认随机种子导致迭代序不定（确定性档需固定种子/有序变体） |
| smallvec / arrayvec | 栈上小容量优化、`no_std`、省堆分配 | spill 到堆的边界成本；需明确容量选择 |
| slotmap / slab | generational index 防悬垂、O(1) 增删查、密集存储 | 版本号回绕处理；外部存 key 的生命周期约定 |
| bumpalo | 竞技场批量分配、一次性释放、极快 | 不单独释放单元素；生命周期绑 arena |
| bitvec / 稀疏集(EnTT) | 位集紧凑、稀疏集 O(1) 且可密集遍历（ECS 核心） | 稀疏集内存换速度；大 ID 空间稀疏浪费 |
| crossbeam / folly | 无锁 MPMC 队列、epoch 回收、并发哈希 | 无锁实现正确性难、ABA/回收复杂；仅在 `concurrent` 档启用 |
| Unreal `FName` | 全局字符串驻留成 32-bit ID、比较/哈希 O(1) | 全局表并发/生命周期；需分域驻留避免无界增长 |

**取舍结论**：以 Rust std + hashbrown/smallvec/slotmap 的成熟形态为骨架（最省学习与迁移成本），补引擎特需的三件：**generational slotmap 句柄表**（资产/实体句柄的安全根基）、**稀疏集**（ECS 存储核心）、**字符串驻留 FName**（稳定 ID）；再叠**分配器抽象 + 帧分配器**（把内存握在手里）与**确定性有序容器**（联机/回滚）。无锁容器仅在 `concurrent` 档谨慎启用。

---

## 3. 档位化（capability / quality tier / feature）

- **分配档（alloc tier）**：`no_std`（栈上容器：ArrayVec/固定位集，零堆）/ `alloc`（堆容器：SmallVec/HashMap/竞技场）/ `std`（并发原语依赖）。
- **并发档（concurrency tier）**：`single`（无并发，最简最快）/ `concurrent`（无锁 MPMC/并发哈希/epoch 回收，接 tasks）。
- **功能 feature**：`std`、`concurrent`、`determinism`（确定迭代序/固定哈希种子）、`serde`、`bytemuck`（POD 容器直传）、`alloc-track`（分配埋点，接 diagnostic）、`simd-hash`（SwissTable SIMD 探测）。

裁剪示例：确定性服务器 `["alloc","determinism"]`（有序容器、固定种子、无无锁非确定）；客户端 `["std","concurrent","simd-hash","bytemuck"]`；嵌入式/确定性仿真核 `["no_std"]`（仅栈上容器）。

---

## 4. 分层架构

```
  L2 消费方   prism_ecs（原型/稀疏集/位集）· prism_asset（句柄表）·
  (本 crate 之上) prism_diagnostic（环形缓冲/驻留）· tasks（并发队列）· reflect
  ───────────────────────────────────────────────────────────────
  L1 容器门面  SmallVec · ArrayVec · HashMap/Set · SparseSet · BitSet ·
              SlotMap(句柄) · Interner(FName) · RingBuffer · 并发容器
  ───────────────────────────────────────────────────────────────
  L0 内存基元  Allocator trait · Bump/Arena · Pool · FrameAllocator ·
              原子/epoch 回收 · POD/对齐 · no_std 原语
```

关键：**容器门面不绑死全局分配器**——每个容器可带 `A: Allocator`，让 ECS/帧临时数据走池/帧分配器，长生命周期走全局；这是「把内存握在手里」的根本，也是降低 malloc 抖动与碎片的关键。

---

## 5. 核心容器：SmallVec / ArrayVec / 竞技场（Arena / Bump）

```rust
pub struct SmallVec<T, const N: usize>;   // N 以内栈上，溢出转堆
pub struct ArrayVec<T, const N: usize>;   // 固定容量，纯栈，no_std 零 alloc
pub struct Bump;                          // 竞技场：批量分配、一次性 reset
pub struct Arena<T>;                      // 同类型竞技场 + 可选 generational 索引
```

- **SmallVec**：小集合（组件列表、子节点、临时结果）栈上存，避免高频小分配（ECS/transform/查询大量用）。
- **ArrayVec**：已知上界的固定容量，纯栈、`no_std`，用于确定性/嵌入式热路径。
- **Bump 竞技场**：一帧内大量临时对象（命令缓冲、临时图节点）批量分配，帧末一次 `reset`（O(1) 释放），零碎片。
- **零拷贝取出**：竞技场可产 `&'arena T`，生命周期绑 arena，避免复制。

---

## 6. 句柄表：Slotmap（generational index）

```rust
pub struct Key { index: u32, generation: u32 }   // 外部持有的稳定句柄
pub struct SlotMap<T> { /* 密集存储 + 空闲链 + 代数 */ }
impl<T> SlotMap<T> {
    fn insert(&mut self, v: T) -> Key;
    fn get(&self, k: Key) -> Option<&T>;          // 代数不匹配 => None（防悬垂）
    fn remove(&mut self, k: Key) -> Option<T>;
}
```

- **防悬垂**：删除后槽位代数 +1，旧 Key 再访问代数不匹配返回 `None`——AAA 引擎资产/实体句柄的安全根基。
- **密集存储**：值密集排列（可选 `DenseSlotMap`）支持缓存友好遍历；稀疏索引层做 Key→槽位映射。
- **用途**：`prism_asset` 资产句柄、`prism_ecs` 实体 ID 候选、资源管理器、任意「发出去的 ID 必须能安全失效」的场景。
- **变体**：`SecondaryMap`（并行附属数据）、`SlotMap`/`DenseSlotMap`/`HopSlotMap` 按遍历 vs 增删权衡。

---

## 7. 稀疏集 SparseSet / 位集 BitSet

- **SparseSet**：`sparse[id] -> dense_idx` + `dense[]` 密集值，O(1) 增删查 **且** 可密集缓存友好遍历——ECS 组件存储与查询的核心数据结构（EnTT/flecs 形态）。
- **BitSet**：紧凑位集（固定 `no_std` + 动态 `alloc`），位运算求交并差（ECS archetype 匹配、查询过滤、脏标记集）。
- **位运算批处理**：`iter_ones` 用 `trailing_zeros` 快速遍历置位；SIMD 位运算批量求交（query 匹配热路径）。
- **分层位集**（hierarchical bitset）：大 ID 空间下用两级位集加速「找下一个置位」，供稀疏大世界实体集。

---

## 8. 稳定哈希与 HashMap / HashSet 门面

- **SwissTable 门面**：`HashMap`/`HashSet` 基于 SwissTable（SIMD 分组探测、高负载因子、缓存友好），默认 Rust 生态等价性能。
- **哈希器选择**：`FxHash`（快、非抗碰撞，内部用）/ `aHash`（抗碰撞，默认）/ 固定种子哈希（确定性档）。小整数 key 用恒等哈希。
- **确定性档**：`determinism` 下用固定种子 + 有序迭代变体（`IndexMap` 形态保插入序），保证跨运行迭代序一致（联机/回滚/序列化稳定）。
- **稳定哈希**：类型/字符串的**跨运行稳定哈希**（非进程随机种子），供资产 ID、类型 ID（接 `prism_reflect`）、网络协议版本。

---

## 9. 字符串驻留 / 稳定 ID（FName 形态）

- **Interner**：字符串驻留成 32/64-bit `Istr` ID，比较/哈希/拷贝 O(1)（Unreal `FName` 形态），用于标签、资产路径、system 名、诊断字符串。
- **分域驻留**：按用途分独立驻留表（标签域、路径域、调试域），避免单一全局表无界增长与并发热点。
- **双向映射**：ID→字符串用于诊断/序列化反查；字符串→ID 用于解析输入。
- **并发驻留**（`concurrent`）：多线程驻留用分片锁/无锁插入；确定性档下驻留顺序不影响逻辑（仅做 ID 身份，不参与跨机比较）或采用内容哈希 ID。

---

## 10. 分配器抽象（Allocator trait / 池 / 帧分配器）

```rust
pub trait Allocator { fn alloc(&self, layout: Layout) -> *mut u8; fn dealloc(&self, p: *mut u8, layout: Layout); }
pub struct PoolAllocator<const SIZE: usize>;  // 固定块池，O(1) 分配/回收，零碎片
pub struct FrameAllocator;                    // 线性帧分配，帧末 reset
pub struct TrackingAllocator<A>;              // 包装任意分配器 + 分配埋点
```

- **容器可带分配器**：`SmallVec<T, N, A>`、`HashMap<K, V, S, A>` 可指定分配器，让临时数据走帧/池分配器，长期数据走全局。
- **池分配器**：同尺寸对象（粒子、节点、事件）固定块池，O(1) 分配回收、零碎片、缓存友好。
- **帧分配器**：每帧临时数据线性分配、帧末整体 reset（O(1)），对标主机引擎的 scratch/double-buffer 线性堆。
- **分配埋点**（`alloc-track`）：`TrackingAllocator` 按标签/调用栈归类分配，喂 `prism_diagnostic` 内存火焰图。

---

## 11. 并发容器（`concurrent` feature，接 tasks）

- **MPMC 队列**：无锁多生产多消费队列，供 `prism_tasks` work-stealing 的任务队列与跨线程消息。
- **并发哈希**：分片/无锁并发 HashMap，供资产缓存、类型注册的多线程读多写少场景。
- **epoch / 延迟回收**：无锁结构的安全内存回收（crossbeam-epoch 形态），避免 use-after-free/ABA。
- **SPSC 环形缓冲**：单生产单消费无锁环形，供诊断线程本地缓冲（`prism_diagnostic` §13）。
- **边界**：无锁结构正确性难、确定性档慎用（归并序须确定，接 tasks §确定归并）；默认 `single` 档用普通容器 + 外部同步。

---

## 12. no_std + alloc 友好与可移植

- **分层 no_std**：纯栈容器（ArrayVec/固定位集）无需 `alloc`；堆容器需 `alloc`；并发原语需 `std`（或平台原子抽象，接 `prism_platform`）。
- **原子抽象**：`no_std` 下原子操作经 `core::sync::atomic`；无原子平台由 `prism_platform` 兜底。
- **可移植**：核心数据结构在服务器/客户端/嵌入式/WASM 一致编译；SIMD 探测有标量回退。

---

## 13. 确定性容器（`determinism` feature，接四方确定性）

四方确定性契约（ECS 序 + tasks 归并 + time 定点 + transform 定点）要求**容器迭代序跨运行跨平台一致**，否则同输入不同序 → 不同结果 → 联机 desync：

- **有序哈希**：`determinism` 下 HashMap 用插入序迭代（`IndexMap` 形态）或固定种子，杜绝随机种子导致的迭代序抖动。
- **确定遍历**：SparseSet/SlotMap 的遍历序固定（按 dense 索引），不依赖内存地址/哈希随机。
- **确定分配**：确定性仿真核的分配走帧/池分配器（分配序确定），不依赖全局 malloc 地址。
- **稳定 ID**：驻留 ID 在确定性场景用内容哈希而非插入序，保证跨机身份一致。

---

## 14. 序列化 / POD 友好（接 reflect / bytemuck）

- **POD 容器**（`bytemuck`）：元素为 `Pod` 的 `Vec`/`ArrayVec` 可整块 `cast_slice` 直传 GPU/磁盘，供 ECS 组件列、`prism_math` 向量数组。
- **serde 支持**（`serde`）：容器可序列化（确定性档下序列化序稳定），供场景/存档。
- **反射桥**（接 `prism_reflect`）：容器类型注册反射，支持动态字段遍历/脚本访问。
- **稳定布局**：POD 容器内存布局版本化，跨版本存档/网络需迁移时显式处理。

---

## 15. 与 ECS / asset / tasks / diagnostic 集成

- **prism_ecs**：SparseSet/BitSet 是原型存储与查询匹配的核心；SmallVec 存组件/子节点列表；帧分配器供命令缓冲；稳定类型哈希做组件 ID。
- **prism_asset**：SlotMap generational 句柄是资产句柄的安全根基；并发哈希做资产缓存；驻留做资产路径 ID。
- **prism_tasks**：MPMC 队列是 work-stealing 任务队列；epoch 回收供无锁结构；帧分配器供 job 临时数据。
- **prism_diagnostic**：SPSC 环形缓冲是埋点线程本地缓冲；驻留做静态字符串 ID；TrackingAllocator 做内存剖析。
- **prism_reflect**：稳定类型哈希/驻留做 TypeId；容器反射支持。

---

## 16. 可观测性（容器统计 / 内存剖析）

- **容器统计**（接 `prism_diagnostic`）：HashMap 负载因子/探测长度、SmallVec spill 率、池占用、竞技场高水位，入计数器。
- **分配剖析**（`alloc-track`）：TrackingAllocator 按标签归类，产内存火焰图 + 峰值/泄漏定位。
- **退化检测**：哈希探测过长（碰撞攻击/坏哈希）、SmallVec 频繁 spill（容量选错）报警。
- **碎片监控**：池/帧分配器碎片率与 reset 频率可见。

---

## 17. 高级功能增补（AAA）

- **分层位集加速遍历**：两级/三级位集（summary bits）让「稀疏大 ID 空间找下一个置位」从 O(n) 降到近 O(置位数)，供大世界稀疏实体集与查询。
- **拷贝写时共享（CoW）容器**：不可变共享 + 写时复制，供资产/配置的多读者零拷贝共享。
- **侵入式链表 / 空闲链**：对象自带 next 指针的侵入式结构，零额外分配，供池空闲链、LRU 缓存。
- **稳定内容寻址 ID**：内容哈希（经典如 xxHash/blake 形态非 ML）做资产/数据块去重与缓存 key，供烘焙/VFS。
- **SoA 容器包装**：`SoaVec<(A,B,C)>` 自动拆成多列存储，缓存友好 + SIMD 友好（接 `prism_math` SoA 批处理、ECS 列存）。
- **范围分配器 / Buddy / TLSF**：通用低碎片实时分配器，供 GPU 显存子分配、流送缓冲管理（接 RHI）。

所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 18. 性能工程

- **栈上优先**：小集合 SmallVec/ArrayVec 栈上，消灭高频小分配（分配是帧时间隐形杀手）。
- **缓存友好布局**：密集存储（DenseSlotMap/SparseSet dense）顺序遍历；SwissTable 分组探测减少 cache miss。
- **SIMD 探测/位运算**：SwissTable SIMD 匹配、位集 SIMD 求交，热路径矢量化。
- **分配摊薄**：池/帧分配器把 malloc 从每对象降到每帧一次 reset。
- **内联与单态化**：泛型容器单态化 + 热方法内联，等价手写。
- **预留与批量**：`reserve`/批量插入避免反复扩容；扩容按 1.5×/2× 权衡拷贝 vs 浪费。

---

## 19. 易用性与 Bevy 迁移策略

- **API 贴 std**：`push/get/insert/iter` 等与 `Vec`/`HashMap` 同形，学习成本近零。
- **`bevy_utils` 兼容层**（`compat-bevy`）：`HashMap`/`HashSet`（bevy 的 aHash 别名）、`StableHashMap` 等映射到 Prism 类型，迁移改 `use`。
- **prelude**：`use prism_utils::prelude::*;` 带入常用容器 + 句柄类型 + 分配器。
- **合理默认**：默认哈希器、默认 SmallVec 容量有工程经验默认值，用户不配也能用好。
- **类型别名**：`EntityMap`/`AssetMap` 等语义别名（在消费 crate 定义），提升可读性。

---

## 20. crate 分层与模块布局

```
pkg/prism_utils/
  src/
    lib.rs            # re-export + prelude
    small_vec.rs      # SmallVec
    array_vec.rs      # ArrayVec（no_std 栈上）
    arena.rs          # Bump / Arena 竞技场
    slotmap.rs        # SlotMap/DenseSlotMap/SecondaryMap（generational）
    sparse_set.rs     # SparseSet（ECS 核心）
    bitset.rs         # BitSet + 分层位集
    hash/             # 哈希
      map.rs set.rs hashers.rs stable.rs   # SwissTable 门面 + FxHash/aHash + 稳定哈希
    intern.rs         # 字符串驻留 FName
    alloc/            # 分配器
      allocator.rs pool.rs frame.rs tracking.rs buddy.rs
    concurrent/       # 并发容器（concurrent feature）
      mpmc.rs spsc.rs concurrent_map.rs epoch.rs
    soa.rs            # SoA 容器包装
    cow.rs            # 写时复制容器
    ring_buffer.rs    # 环形缓冲（供 diagnostic）
    prelude.rs
  features = ["std","concurrent","determinism","serde","bytemuck",
             "alloc-track","simd-hash","compat-bevy"]
```

依赖：**无 Prism 上游**（与 `prism_math` 并列为根）；可选 `hashbrown`。**不碰任何 `bevy_*`。**

---

## 21. 契约、不变量与版本化

- **句柄安全**：SlotMap 的 Key 代数不匹配必返回 `None`，绝不悬垂访问；代数回绕有明确处理约定。
- **确定性契约**：`determinism` 档下所有容器迭代序跨运行跨平台一致（固定种子/插入序/dense 序），是四方确定性的容器底座。
- **分配器契约**：容器指定的分配器生命周期必须覆盖容器；帧分配器 reset 前所有借用必须结束（借用检查 + 文档）。
- **POD 布局**：`bytemuck` 容器内存布局版本化，GPU/磁盘直传依赖它。
- **稳定哈希**：稳定哈希/内容 ID 的算法与种子版本化，跨版本资产/网络兼容需迁移。
- **版本化**：容器序列化格式、句柄 Key 布局、驻留 ID 域、稳定哈希算法均为版本化契约。

---

## 22. 路线图（M0–M6）与基准即规格

- **M0 核心容器**：SmallVec/ArrayVec/Bump + HashMap/Set（SwissTable）+ prelude → 单测 + vs std 基准。**是 ECS/asset M0 的前置**。
- **M1 句柄与稀疏**：SlotMap(generational) + SparseSet + BitSet → 正确性（代数失效、稀疏遍历）+ 基准（增删查）。
- **M2 分配器**：Allocator trait + Pool + FrameAllocator + 容器带分配器 → 基准（池 vs malloc、帧 reset）。
- **M3 驻留 + 稳定哈希**：Interner(FName) + 稳定/内容哈希 → 正确性（跨运行稳定）+ 基准（驻留/比较 O(1)）。
- **M4 确定性档**：有序哈希 + 确定遍历 + 确定分配 → 跨运行跨平台迭代序位级一致验证。
- **M5 并发容器**：MPMC/SPSC/并发哈希 + epoch 回收 → 正确性（压力测试/loom）+ 基准（吞吐/扩展）。
- **M6 高级 + 迁移**：分层位集 + SoA + CoW + buddy/TLSF + `compat-bevy` → 大世界/GPU 子分配验证 + 迁移。

**基准即规格**：容器增删查吞吐 vs std、SmallVec spill 率、SwissTable 负载因子/探测长度、句柄失效正确性、池/帧分配 vs malloc 加速比、确定性迭代序一致、并发容器扩展比。核心价值在 **M0（全引擎前置）+ M1（ECS 存储底座）+ M4（确定性底座）**。

---

## 23. 诚实边界与风险

- M0–M6 核心路线图**已全部落地并通过验证**：实现 + 单测（53 项 lib 测试全绿）+ 基准，`cargo clippy --all-targets` 零告警、`cargo test` 零失败。状态随代码演进；§24「AAA 高级功能增补」24.1–24.7 已全部交付（见 §24.8），部分子项随消费方接线深化。
- **高风险项**：
  1. **无锁容器正确性（M5）**：MPMC/并发哈希/epoch 回收是公认难写对的代码，ABA/内存序/回收时机错一处即偶发崩溃或数据损坏；必须 loom/压力测试 + 保守默认（`single` 档），非必要不上无锁。
  2. **确定性迭代序（M4）**：标准 SwissTable 随机种子天然破坏确定性；确定性档必须切有序变体/固定种子，且全引擎约定「确定性路径禁用非确定容器」，否则一处漏网即 desync。
  3. **分配器生命周期（M2）**：容器带分配器时，帧分配器 reset 早于容器释放会悬垂；借用关系复杂，需严格 API 约束 + 文档，否则 UB。
  4. **驻留表无界增长（M3）**：全局字符串驻留若无分域/回收，运行时间长了无界膨胀；需分域 + 生命周期策略。
  5. **句柄代数回绕（M1）**：32-bit 代数在高频增删下可能回绕导致误匹配；需足够位宽或回绕检测，极端场景需 64-bit Key。
  6. **SmallVec 容量误选（M0）**：N 选太小频繁 spill 反而更慢，选太大浪费栈/内存；需按实测调默认值 + spill 率监控。
- **与既有文档关系**：本 crate 与 `prism_math_design_zh.md` 并列为依赖图根；SparseSet/BitSet/帧分配器是 `prism_ecs_design_zh.md` 存储与调度的底座；SlotMap 句柄供 `prism_asset`；环形缓冲/驻留供 `prism_diagnostic_design_zh.md`；MPMC 队列供 `prism_tasks_design_zh.md`；确定性容器接 ECS 序 / tasks 归并 / time 定点 / transform 定点的「四方确定性」；稳定类型哈希供 `prism_reflect_design_zh.md`。整体组件缺口见 `prism_engine_component_gap_zh.md`。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

## 24. AAA 高级功能增补（v0.2）

本章补齐顶级数据结构 / 内存库在真实 AAA 项目里缺一不可的能力。均 feature/档位门控，默认不付成本；与前文的 SmallVec/竞技场/slotmap/SwissTable 内核互补。

### 24.1 作用域分配器栈与竞技场组合 —— ✅ 部分已交付（`alloc_::scope::ScopeStack`）

单一帧分配器不够：引擎需要**嵌套生命周期**的内存域：

| 作用域 | 生命周期 | 用途 |
|---|---|---|
| 帧（Frame） | 1 帧，帧末 reset | 命令缓冲、临时数组 |
| 子帧（Scope） | 一个阶段/系统内 | 分块中间结果 |
| 持久池（Pool） | 跨帧，显式释放 | 对象实例、组件 |
| 双缓冲（DoubleBuffer） | 跨 2 帧 | 需被下帧读的数据 |

- **分配器栈**：进入作用域 push 一个 bump 竞技场，退出 pop 整体 reset（零逐对象析构开销），嵌套安全。 **（✅ 已交付：`alloc_::scope::ScopeStack` + RAII `Scope` guard + `ScopeMark`——`scope()` 捕获游标、guard drop/`rewind` O(1) 回退、`high_water` 水位、rewind 永不前移游标；实现 `Allocator` trait，可注入任意容器。）**
- **分配器注入**：容器泛型携带分配器参数，同一 `Vec` 可绑帧/池/全局分配器，调用点决定内存域。
- 供 `prism_tasks` 每 worker 帧分配器、ECS 命令缓冲、渲染每帧瞬态数据。

### 24.2 无锁进阶：hazard pointer / RCU / 分片并发哈希 —— ✅ 已交付（`concurrent::Rcu` / `concurrent::Collector` / `concurrent::ConcurrentHashMap`）

§11 并发容器之上的高并发工业级实现：

- **hazard pointer / epoch 回收**：安全回收被并发读者引用的节点，解决 ABA 与 use-after-free（crossbeam-epoch 形态）。 **（✅ 已交付：`concurrent::Collector`/`Guard`——三桶 epoch 回收，`defer` 延迟释放、全局 epoch 推进后才真正回收，`concurrent::TreiberStack` 为无锁对拍样例。）**
- **RCU（读多写少）**：读者零锁零等待，写者复制更新，供全局只读表（类型注册、资产索引）的高频读。 **（✅ 已交付：`concurrent::Rcu<T>` + `RcuGuard`——`read()` 单次 `AtomicPtr` 原子载入 + epoch pin，无 CAS 无自旋；`store()`/`update(f)` 复制整值后原子交换发布，旧版本经 epoch `defer` 在无读者可见后回收；读者持有的快照跨写入保持一致。）**
- **分片并发哈希**：按 key 哈希分片降争用（Java `ConcurrentHashMap`/folly F14 形态），供资产缓存、实体映射的多线程访问。 **（✅ 已交付：`concurrent::ConcurrentHashMap`——按 key 哈希分片、每分片读写锁，不相交 key 跨核扩展。）**
- **保守默认**：默认单线程/加锁实现，无锁仅在 `concurrent` 档 + 实测需要时启用（见 §23 风险）。

**交付状态**：§24.2 的三件套随 M5 `concurrent` 档交付，位于 `prism_utils` 的独立 `concurrent` 模块（`concurrent/epoch.rs` 的 epoch 回收、`concurrent/hash.rs` 的分片并发哈希随 M5 先行落地；本次补齐 `concurrent/rcu.rs` 的 `Rcu<T>`/`RcuGuard`）。`lib.rs` 以 `pub mod concurrent` + `pub use concurrent::{Rcu, RcuGuard, …}` 导出并进 `prelude`。`Rcu` 复用本章 §24.2 的 epoch 回收域（`pin_with`/`Guard::defer`）保证读者借出的 `&T` 在 pin 期间不被回收——读路径只有一次 `AtomicPtr` acquire 载入，写路径 `store`/`update` 用整值复制 + `swap`/`compare_exchange` 发布、旧版本延迟释放；全程保守内存序，`unsafe` 仅限原子指针 `Box::into_raw`/`from_raw` 搬运与跨线程回收，每处均带 `#[expect(unsafe_code, reason=…)]` + `// SAFETY:` 注释，与 `TreiberStack`/`epoch` 既有做法一致。专项单测覆盖初值读取、`store`/`update` 发布、`load_cloned`、持有快照跨写入一致（RCU 语义）、`Default`/`Debug`、空/单元素边界，以及多线程：读者永不观测撕裂/被回收值、并发 `update` 串行化无丢失、超版本「恰好回收一次」无泄漏无双重释放、共享回收域跨 cell（见 `src/concurrent/rcu_tests.rs`，11 项）。

### 24.3 内存安全加固（canary / poison / debug 校验） —— ✅ 已交付（`guard`）

- **分配 canary**：debug 档在分配块前后插哨兵字节，释放时校验，捕获缓冲区溢出写。 **（✅ 已交付：`guard::GuardedBuffer` 在 payload 前后各插一段 canary redzone（默认 16 字节/侧、特征字节 `0xFD`），`write_at`/`read_at` 对 payload 做边界检查，`validate()`/`free()` 复校前后 canary——前段被改报 `Underflow`、后段被改报 `Overflow`；另有 `unsafe` 分配层 `alloc_::guard::GuardedAllocator` 做等价检查。）**
- **释放毒化（poison）**：释放后填充特征字节 + 可选保护页（接 platform §9），把 use-after-free 变成即时崩溃。 **（✅ 已交付：`guard::GuardedBuffer::free()` 先校 canary 再用毒化字节（默认 `0xDD`）覆写整个 payload 并置「已释放」位，之后任何 `payload`/`write_at`/`read_at`/`validate` 均返回 `UseAfterFree`。保护页需真实 `MMU` 系统调用，属 platform §9，纯安全容器无法 fault，故改为「下次校验即报告」语义。）**
- **双重释放 / 越界检测**：池/slotmap 校验代数与状态，双释放/悬垂句柄即断言（接 `prism_diagnostic` §11）。 **（✅ 已交付：`guard::GuardedPool<T, D>` 为分代 slab——每槽带 generation，复用时自增（跳 0）；`try_get`/`try_remove` 区分出悬垂句柄 `DanglingHandle`（槽从未分配）、`DoubleFree`（本句柄已释放该槽）、`UseAfterFree`（槽已被回收复用）；`get`/`remove` 为断言版，panic 文案含 `use-after-free`/`double free`/`dangling handle` 便于定位。）**
- **分配栈追踪**：接 `alloc-track`，每分配记调用栈，供 diagnostic §24.3 泄漏/碎片可视化。 **（PLANNED：调用栈捕获需 `alloc-track` 后端，属 `prism_diagnostic`，不在本内核 crate 实现。）**

**交付状态**：canary / poison / 双重释放 / 悬垂句柄 / use-after-free 检测已随本构建交付，位于 `prism_utils` 的独立 `guard` 模块（`guard/canary.rs` 的 `GuardedBuffer` + `guard/pool.rs` 的 `GuardedPool`/`GuardHandle`，共享 `guard/mod.rs` 的 `GuardError`/`GuardConfig`）。两者为**纯安全**实现：全部检测用安全的边界检查、canary 字节比较、代数比较与存活标志完成，**零 `unsafe`**、无一处本地 lint 覆盖，满足工作区 `unsafe_code` deny；它们是裸内存版 `alloc_::guard::GuardedAllocator`（分配层、含 `unsafe` 指针算术）在容器层的安全互补物。no_std+alloc 兼容（`extern crate alloc`）。`lib.rs` 以 `pub mod guard` + `pub use guard::{GuardConfig, GuardError, GuardHandle, GuardedBuffer, GuardedPool}` 导出并进 `prelude`。专项单测以影子 `Vec<u8>`（buffer）与影子存活句柄表（pool）对拍，覆盖越界读写、canary 上/下溢、释放毒化、双重释放、use-after-free、悬垂句柄、分代复用、空/单元素边界，以及伪随机生命周期对拍与断言版 `#[should_panic]`（见 `src/guard/tests.rs`，21 项）。保护页属 platform §9、分配栈追踪属 `prism_diagnostic` §24.3，仍为 PLANNED。

### 24.4 热 / 冷数据分离与 SoA 自动布局 —— ✅ 已交付（`layout`）

缓存效率的关键是「只把热字段塞进缓存行」：

- **热冷分离容器**：常访问字段（位置/变换）与冷字段（名称/调试信息）分列存储，遍历热路径不污染缓存。 **（✅ 已交付：`layout::HotCold<H, C>`——热字段元组 `H` 与冷字段元组 `C` 各存一套独立 `SoaVec`，按行索引锁步；`push`/`swap_remove`/`clear` 两半严格等长，热批处理 `hot_columns()` 只触热内存、绝不载入冷字节，冷字段仍 O(1) 可取。）**
- **SoA 自动布局**：`SoaVec<(A,B,C)>`（§17）编译期拆列 + SIMD 对齐（接 `prism_math` SoA、platform 缓存行探测），供 ECS 列存与批处理。 **（✅ 已交付：`layout::LayoutPlan`/`GroupLayout`/`ColumnShape` + `ColumnShapes` trait——确定性计算每列尺寸/对齐/步长、分组推荐缓存对齐、热工作集宽度与 `fits_cache_line`，`stride_for(lane_align)` 给出 SIMD 车道对齐步长；`align_up` 为纯 const 对齐取整，全程安全无 `unsafe`。）**
- **AoSoA 混合**：块内 SoA、块间 AoS，兼顾 SIMD 与局部性，供粒子/物理大批量。 **（PLANNED：当前 `LayoutPlan::lane_bytes_for` 已给出 AoSoA 块尺寸所需的车道步长，块内 SoA/块间 AoS 的物理块容器待粒子/物理批量落地。）**

**交付状态**：热冷分离容器与 SoA 自动布局已随本构建交付，位于 `prism_utils` 的独立 `layout` 模块（`layout/hotcold.rs` + `layout/plan.rs`，`lib.rs` 以 `pub mod layout` + `pub use` 导出，并进 `prelude`）。`HotCold` 复用 §24.4/§17 的 derive-free `Soa` 列存，纯安全代码、零 `unsafe`、no_std+alloc 兼容（`extern crate alloc`，与既有模块一致）；`LayoutPlan` 为纯确定性布局描述，不分配被存数据。专项单测覆盖热/冷列独立增删一致、索引访问、切片迭代顺序、布局对齐/步长、空/单元素边界（见 `src/tests_hotcold.rs`，15 项）。AoSoA 混合物理块仍为 PLANNED。

### 24.5 可重定位 / 内存映射友好容器（offset 指针）—— ✅ 已交付（`reloc`）

资产直接 mmap 后用、免反序列化，是 AAA 加载速度的关键：

- **offset 指针**：容器内部用相对偏移而非绝对指针，整块可 memcpy/mmap 到任意地址仍有效（Unreal `TArray` 序列化、`FlatBuffers` 形态）。 **（✅ 已交付：`reloc::OffsetPtr<T>`/`reloc::OffsetSlice<T>`——以 `i32` **自相对**偏移（相对字段自身地址）而非绝对指针寻址，`OffsetPtr` 以偏移 `0` 表示 NULL，`OffsetSlice` 为 `i32` 偏移 + `u32` 长度；整块 `memcpy` 或内嵌进更大 blob 后，偏移语义不变，无需任何重定位修补。）**
- **可重定位容器**：`RelocVec`/`RelocMap` 支持「烘焙成连续 `blob` → 运行期 mmap 零解析直接用」（接 platform §6 mmap、`prism_asset` 烘焙）。 **（✅ 已交付：`reloc::RelocVec<T>`/`reloc::RelocMap<K, V>` 将元素序列化为**带魔数头的连续字节 `blob`**（`RelocVec` 魔数 `0x3143_5652`、头 16 字节；`RelocMap` 魔数 `0x3150_4D52`、头 28 字节、键有序存储供二分查找），对应 `RelocVecView`/`RelocMapView` 以**纯安全边界检查**在原地（memcpy 到任意地址后）零拷贝解析，所有读取均先验证魔数、长度与元素边界，越界/坏魔数/尺寸不符一律返回 `RelocError`。）**
- **POD 布局版本化**：`blob` 布局以魔数区分，GPU/磁盘直传。 **（✅ 已交付：`reloc::Reloc` trait 为 POD 标量（`u8`..`u64`/`i8`..`i64`/`f32`/`f64`/`bool`，经 `impl_reloc_le!` 覆盖）定义**固定字节宽 + 小端（LE）**的 `encode`/`decode`，跨大小端平台位级一致；容器头的魔数即最小布局版本标记。嵌套变长图与 bytemuck 整体零拷贝转型属更上层能力，见诚实边界。）**

**交付状态**：可重定位 offset 指针容器已随本构建交付，位于 `prism_utils` 的独立 `reloc` 模块（`reloc/mod.rs` 的 `RelocError`/`Reloc` trait/`impl_reloc_le!`，`reloc/offset.rs` 的 `OffsetPtr`/`OffsetSlice`，`reloc/vec.rs` 的 `RelocVec`/`RelocVecView`，`reloc/map.rs` 的 `RelocMap`/`RelocMapView`）。**纯安全**实现：全部序列化/解析用安全的字节切片边界检查、LE 定宽编解码与 `u32`/`i32` 偏移算术完成，`#![forbid(unsafe_code)]`、零 `unsafe`、无一处本地 lint 覆盖；no_std 风格 + alloc 兼容（各文件 `extern crate alloc`，与既有模块一致）。`lib.rs` 以 `pub use reloc::{OffsetPtr, OffsetSlice, Reloc, RelocError, RelocMap, RelocMapView, RelocVec, RelocVecView}` 导出并进 `prelude`。专项单测以手算 LE 字节序列为 oracle 对拍，覆盖标量编解码位级向量、`RelocVec`/`RelocMap` 烘焙→解析往返、整块 `memcpy` 到新缓冲后仍有效、内嵌进更大 blob 后按偏移解析、`RelocMap` 二分查找命中/未命中、坏魔数/截断/尺寸不符的 corruption 检测、空/单元素边界（见 `src/reloc/tests.rs`，20 项）。

### 24.6 内容寻址与去重缓存 —— ✅ 已交付（`intern::InternCache`）

- **内容寻址驻留**：对数据块算经典内容哈希（xxHash/BLAKE 形态，非 ML）做 key，相同内容只存一份，供资产/网格/纹理去重（接烘焙）。 **（✅ 已交付：`InternCache<T, D>` 以本 crate 的 `hash::stable_hash`（FNV-1a，非加密、非 ML）做桶 key，哈希仅选桶、再以完整 `Eq` 确认，碰撞绝不别名；相同内容只 `push` 一份。）**
- **去重缓存门面**：`InternCache<T>` 自动合并等价对象，返回稳定句柄，省内存 + 使相等比较降为句柄比较（O(1)）。 **（✅ 已交付：`intern(value)` 幂等——等价内容恒返回同一个 `Interned<T, D>` 句柄；句柄是域标记的 `u32`，`Copy`、`size_of == size_of::<u32>()`，两句柄相等当且仅当寻址同一份内容，相等比较由深比较降为 `u32` 比较；另有 `get`/`contains` 不插入、`intern_ref` 缺失才克隆、`resolve`/`try_resolve` 回解。）**
- **分域驻留**：字符串/类型/资产 ID 分域驻留，避免全局表无界增长（见 §23 风险），生命周期按域回收。 **（✅ 已交付：`D` 为不可居住域标记（`domain::{Tag,Path,Debug,Type,Asset,Mesh,Texture}`），不同域句柄是不同类型、编译期拒绝混用；每域一张独立缓存，`clear()` 整域回收、使旧句柄失效、索引从 0 重新开始——无全局无界表。）**

**交付状态**：内容寻址去重缓存已随本构建交付，位于 `prism_utils` 的独立 `intern` 模块（原字符串驻留 `intern.rs` 重构为 `intern/string.rs`，新增 `intern/cache.rs` 的 `InternCache`/`Interned`、与字符串驻留共享 `intern/domain.rs` 的域标记，`intern/mod.rs` 汇总）。纯安全代码、零 `unsafe`、no_std+alloc 兼容（`extern crate alloc`），底层复用 `Vec<T>` + `HashMap<u64, Vec<u32>>` 桶索引。`lib.rs` 以 `pub use intern::{InternCache, Interned, …}` 导出并进 `prelude`。专项单测以线性扫描 `Vec<T>` 为 oracle 对拍，覆盖去重、句柄等价⇔内容等价、`resolve` 往返、`get`/`contains` 不插入、`intern_ref`、4000 次伪随机混合序对拍、插入序迭代、`clear` 回收、`content_hash` 稳定、域类型隔离、句柄 `u32` 大小与 `Copy`（见 `src/intern/cache_tests.rs`，11 项）。

### 24.7 确定性进阶（有序并发 / 可复现哈希）—— ✅ 已交付（`det`）

- **有序并发容器**：`concurrent` + `determinism` 双档下，并发写入最终以确定序归并（接 tasks §24.7 确定归并），保证回放一致。 **（✅ 已交付：`det::DeterministicMerge<K, V>` 收集任意顺序的 `(K, V)` 贡献，`into_sorted()` 按键稳定排序给出**与贡献/交错顺序无关**的确定结果，`into_reduced(combine)` 按键归约；`concurrent` 档下的 `det::ConcurrentMerge<K, V>`（内部 std `Mutex`）允许多线程并发 `contribute`，最终归并结果位级等同单线程 oracle，保证回放一致。）**
- **可复现哈希**：固定种子 + 固定迭代序，跨运行跨平台位级一致，是四方确定性的哈希底座（接 ECS/replication）。 **（✅ 已交付：`det::mix64`（`splitmix64` finalizer，`const fn`）+ `det::OrderedHashCombiner`/`det::UnorderedHashCombiner` + `det::reproducible_hash_ordered`/`det::reproducible_hash_unordered`——有序组合器对序列敏感、无序组合器用可交换的异或折叠使**集合哈希与元素顺序无关**，全程无浮点、无地址依赖、无 `std::hash` 随机种子，跨运行跨平台位级一致。）**
- **确定性分配**：确定性档下分配地址无关的 ID/序，容器状态可跨机复现（调试 desync）。 **（PLANNED：`DeterministicMerge` 的键序结果已地址无关；地址无关 ID 分配器待确定性档整体落地。）**

**交付状态**：确定性进阶（有序并发合并 + 可复现哈希）已随本构建交付，位于 `prism_utils` 的独立 `det` 模块（`det/hash.rs` 的 `mix64`/`OrderedHashCombiner`/`UnorderedHashCombiner`/`reproducible_hash_ordered`/`reproducible_hash_unordered`，`det/merge.rs` 的 `DeterministicMerge` 与 `concurrent` 档下的 `ConcurrentMerge`）。**纯安全**实现：哈希为非加密、非 ML 的 `splitmix64` 风格整数混合（固定常量 + 固定折叠序），合并为稳定排序 + 归约，`#![forbid(unsafe_code)]`、零 `unsafe`；no_std 风格 + alloc 兼容（各文件 `extern crate alloc`，`ConcurrentMerge` 仅在 `concurrent` 档启用、用 std `Mutex`）。`lib.rs` 以 `pub use det::{mix64, reproducible_hash_ordered, reproducible_hash_unordered, DeterministicMerge, OrderedHashCombiner, UnorderedHashCombiner}`（+ `#[cfg(feature = "concurrent")] ConcurrentMerge`）导出并进 `prelude`。专项单测以**冻结的已知哈希向量**（手算/独立 oracle 复核，如 `mix64(1)=0x5692161d100b05e5`、`ordered([1,2,3])=0xee3314c644c036cd`、`unordered([1,2,3])==unordered([3,2,1])=0xb8c190a947434478`）对拍，覆盖跨调用稳定、空/单元素、有序敏感性、无序置换不变性、`DeterministicMerge` 乱序贡献→有序结果、归约求和、以及 `concurrent` 档下 8 线程并发贡献等同单线程 oracle 且多次独立运行一致（见 `src/det/tests.rs`，17 项，其中 2 项 `concurrent`-gated）。

### 24.8 诚实边界

**24.1 作用域分配器已交付**（`alloc_::scope::ScopeStack`），**24.2 无锁进阶已交付**（`concurrent::Collector`/`Guard` epoch 回收 + `concurrent::Rcu` 读多写少 RCU + `concurrent::ConcurrentHashMap` 分片并发哈希；均在 `concurrent` 档下，默认关闭），**24.3 内存安全加固已交付**（纯安全容器层 `guard::GuardedBuffer`/`guard::GuardedPool` + 既有裸内存分配层 `alloc_::guard::GuardedAllocator`；保护页属 platform §9、分配栈追踪属 `prism_diagnostic` §24.3，仍为 PLANNED），**24.4 热冷分离 + SoA 自动布局已交付**（`layout::HotCold` + `layout::LayoutPlan`；其中 AoSoA 混合物理块仍为 PLANNED），**24.5 可重定位 offset 指针容器已交付**（`reloc::OffsetPtr`/`OffsetSlice` + `reloc::RelocVec`/`RelocMap` + `reloc::Reloc` LE 编解码；仅支持定宽 POD 标量与单层 blob 的 memcpy/重定位后有效，**嵌套变长图/字符串池、运行期 mmap 系统调用属 platform §6**、bytemuck 整体零拷贝转型属 §14，仍为 PLANNED），**24.7 确定性进阶已交付**（`det::DeterministicMerge` 确定序归并 + `concurrent` 档 `det::ConcurrentMerge` + `det::reproducible_hash_ordered`/`reproducible_hash_unordered` 可复现哈希；**哈希为非加密、非 ML 稳定哈希**，`into_reduced` 的 `combine` 需可交换且可结合、`UnorderedHashCombiner` 的无序不变性仅对去重后集合成立，地址无关 ID 分配器待确定性档整体落地，仍为 PLANNED）；其余为 PLANNED。**24.1 作用域分配器 + 24.4 热冷/SoA 布局**是 ECS/tasks/渲染最先依赖的能力，建议随 M2/M3 优先落地；24.5 可重定位容器已交付（`reloc`），后续随 `prism_asset` 烘焙对接；24.6 内容寻址去重已交付（`intern::InternCache`）；24.2 无锁进阶已随 M5 落地（RCU 本次补齐），24.7 确定性进阶已随 M5 落地（`det`）。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。
