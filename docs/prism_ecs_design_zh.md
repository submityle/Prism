# Prism ECS 顶级次世代 AAA 级自研 ECS 内核设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的数据导向、cache 友好、chunk 并行、关系驱动、确定性可回滚的 **顶级次世代 AAA 级** ECS 内核设计。
> 借形态不抄码。借鉴：
> - **人体工学**：Bevy ECS（derive / system 参数推导 / tick 变更检测 / required components / run condition / observer）
> - **性能存储**：Unity DOTS（16KiB chunk / chunk 版本号 / IJobChunk / shared component / blob）、EnTT（sparse-set 极速增删 / owning group 完美打包 / snapshot / signal）
> - **关系模型**：flecs v4（fragmenting relations / pair `(Relation, Target)` / 通配 / 传递 / prefab `IsA` / observer / meta 反射）
> - **大规模仿真**：Unreal MassEntity（海量轻实体 processor）、Star Citizen（64-bit 浮点原点重定位）、UE5 World Partition（cell 流送 / HLOD / 数据层）
> - **作业系统**：Naughty Dog / DOOM（fiber job graph，细粒度依赖 + 工作窃取）
> - **确定性网络**：Photon Quantum / GGPO / Overwatch（确定性仿真 + 回滚/预测 + 状态哈希去同步）
> - **GPU 驱动**：Horizon / Insomniac（GPU 常驻实例列直传、脏块增量上传）
> - **反应式/数据真相**：Our Machinery（The Truth）+ SolidJS（细粒度无毛刺反应）
> 本文为纯经典数据结构 + 调度路线，不含任何 AI/ML 内容。

- 版本: v0.4（核心内核 M0–M5 与第 23 章增补已在 `pkg/prism_ecs/` 落地实现，第 24 章跨 crate 对齐、M6 迁移与大规模/硬件验证仍为 PLANNED；v0.2→v0.3 新增第 23 章「AAA 高级功能增补」：实体禁用/关系删除策略/排他关系/调度歧义检测与单步/渲染提取管线/作用域命令/层级 despawn/处理器 LOD；v0.3→v0.4 新增第 24 章「跨 crate 契约对齐」：调度执行底座对齐 tasks QoS 车道+time 帧预算 / Observer 作为空间派生更新总线 / 反射驱动动态组件·快照·网络增量 / 确定性链路四方对齐）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: `prism_math`（glam SIMD）、`prism_tasks`（work-stealing + fiber 作业图，`std`/`multi_thread`）、`prism_reflect`（可选，序列化/反射/脚本桥）
- 明确约束: 核心 `no_std + alloc`；`std` / `multi_thread` / `serialize` / `reflect` / `trace` / `determinism` / `simd` / `partition` 为 feature；**不依赖任何 `bevy_*` crate**

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier）
4. 分层架构
5. 核心数据模型（Entity / Component / Archetype / Chunk / Table）
6. 存储模型：Table / SparseSet / SharedComponent / OwningGroup 四态
7. 查询系统（Query / Filter / Iter / Par / 脏块访问器）
8. 系统与调度（System / Schedule / 冲突图执行器 / Fiber 作业图）
9. 命令与结构性变更（Commands / 并行 ECB / 排序回放）
10. 变更检测与反应式内核（双层 tick + chunk 版本 + push 反应图）
11. 关系系统（flecs 式 fragmenting relations / 传递 / 通配 / 层级）
12. Observer 与生命周期 Hook
13. 大规模与大世界（World Partition 流送 / 实体 LOD/休眠 / 64-bit 浮点原点）
14. 确定性仿真与回滚/预测网络
15. GPU 驱动（常驻列直传 / 脏块增量上传）
16. 其他高级功能（required components / 动态组件 / prefab 继承 / blob / snapshot / 反射桥 / 多 World）
17. 性能工程（SIMD / 竞技场 / NUMA / 预热 / 内存回收）
18. 易用性与 Bevy 迁移策略
19. crate 分层与模块布局
20. 契约、不变量与版本化
21. 路线图（M0–M6）与基准即规格
22. 诚实边界与风险
23. AAA 高级功能增补（v0.3：禁用/关系删除策略/排他/歧义检测/提取管线/作用域命令/处理器 LOD）
24. 跨 crate 契约对齐（v0.4：与 tasks/time/reflect/transform 升级后的新契约对齐）

---

## 1. 设计哲学与目标

ECS 是整个 Prism 脱离 Bevy 工程的"命门"——渲染、物理、音频算法层基本只依赖 `bevy_math`（= glam 封装），深耦合集中在少数 ECS 对接层（`prism_render_scene / material / visibility`、`prism_bevy`、`prism_audio_bevy`、`prism_ui_ecs`）。把 ECS 自研化并对外保持近 Bevy 的 API，就能让这些对接层以"改 import"级别的成本迁移。

**一句话定位**：`prism_ecs` 是 Prism 的"权威仿真内核"——百万~千万级实体、每帧数千~数万结构变更、GPU 驱动提取下，迭代/调度/变更检测/命令回放/内存成本正比于**变化量**而非总量；多核近线性扩展；并原生支撑大世界流送、确定性回滚网络、GPU 常驻。

四条总目标（按权重）：

1. **性能（成本 ∝ 变化量）**：chunk SoA + SIMD 迭代、chunk 版本号整块跳过、owning group 完美打包、fiber 作业图 system 内并行、竞技场/NUMA 内存、空闲数据零成本。
2. **效果（规模能力）**：大世界 cell 流送、实体 LOD/休眠、64-bit 浮点原点、确定性回滚网络、一等关系图谱、GPU 常驻列直传。
3. **易用**：derive + 自动 system 参数推导，与 `bevy_ecs` 近乎一致的手感；高级能力默认关闭、按档位开启。
4. **可移植 + 档位化**：核心 `no_std + alloc`；同一内核经 capability / quality tier / feature 三重门控，从移动端缩放到高端桌面 AAA。

非目标：不追求 100% 兼容 Bevy 每个边角 API；不做 C ABI（脚本层后续另走 FFI/WASM）；跨平台浮点严格一致仅在 `determinism` 档位 + 定点路径下保证。

---

## 2. 参考产品取舍

| 产品 | 吸收 | 规避 |
|---|---|---|
| Bevy ECS | derive、system 参数推导、tick 变更检测、Commands、States、run condition、required components、observer | archetype 碎片化、少量运行时开销、无确定性/回滚一等公民 |
| Unity DOTS | 16KiB chunk、chunk 版本号、IJobChunk 并行、shared component 批次、blob 资产、确定性 | C#/Burst 绑定、API 啰嗦 |
| EnTT | sparse-set 极速增删、owning group 完美打包、snapshot、signal | C++、手动管理 |
| flecs v4 | fragmenting relations、pair、通配/传递查询、层级、prefab `IsA`、meta 反射 | C ABI 风格、宏不友好 |
| Unreal MassEntity | 海量轻实体、processor/fragment 分离、LOD 化更新 | 与 UObject 耦合、专有 |
| Star Citizen | 64-bit 浮点原点重定位（局部坐标 + cell 偏移） | 专有 |
| UE5 World Partition | cell 流送、HLOD、数据层、运行时加载 | 与 UE 资产系统耦合 |
| Naughty Dog / DOOM | fiber job graph（原子计数依赖 + 工作窃取 + 近零同步点） | 平台专有调度器 |
| Photon Quantum / GGPO | 确定性仿真 + 回滚/预测 + 状态哈希去同步 | 闭源网络栈 |
| Horizon / Insomniac | GPU 常驻实例列、脏块增量上传、GPU 驱动剔除 | 主机专有 |
| Our Machinery (Truth) + SolidJS | push 变更推送、细粒度无毛刺反应图 | 专有/前端语境 |

综合：**Bevy 人体工学 + Unity chunk 性能 + flecs 关系模型** 为三支柱，叠加 **fiber 作业图 + 大世界流送 + 确定性回滚 + GPU 常驻 + 反应式内核** 的 AAA 能力层，全部走 feature/档位门控。

---

## 3. 档位化（capability / quality tier）

同一内核经三重门控缩放，默认档位行为稳定，高级档单独验证：

| 维度 | 说明 | 示例 |
|---|---|---|
| **capability** | 运行时探测的硬件能力 | 线程数、SIMD 宽度、是否支持 GPU 常驻映射 |
| **quality tier** | 内容/场景规模档 | mobile / console / high-end-desktop（实体上限、chunk 大小、LOD 距离、流送半径） |
| **feature flag** | 编译期裁剪 | `simd` / `multi_thread` / `partition` / `determinism` / `reflect` |

目标：移动端只付基础 chunk 迭代成本；高端桌面叠加 fiber 作业图 + 大世界流送 + GPU 常驻 + 回滚网络。

---

## 4. 分层架构

```
L5  App / Plugin / 固定阶段          (prism_app)
L4  Schedule / Executor / Fiber 作业图  冲突图并行 + system 内 chunk 子作业
L3  System / Query / Command / Observer  参数推导、查询、延迟变更、事件响应
L2  World / Archetype / Relation / 反应图  实体、原型、关系、push 反应
L1  Storage: Table/Chunk / SparseSet / Shared / OwningGroup
L0  Entity / Component / Chunk 竞技场      代际索引、注册表、16KiB 块、arena
```

依赖方向严格向下；L0–L2 可 `no_std + alloc`，L4 的多线程/fiber 执行器需 `std` + `prism_tasks`。

---

## 5. 核心数据模型

### 5.1 Entity（代际索引）

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity { index: u32, generation: NonZeroU32 }
```

- 分配器维护 `free-list` + `meta: Vec<EntityMeta>`；`EntityMeta = { generation, location: {ArchetypeId, ChunkIndex, Row} }`。
- 回收后 generation 自增，悬垂访问因 generation 不匹配安全返回 `None`（流送卸载依赖此语义，见 §13）。
- generation 0 保留（`NonZeroU32` 不变量）：回收时 generation `wrapping_add(1)` 并跳过 0；entity **index 0 为正常可用槽**（分配器不保留 index 0）。确定性档位下分配走稳定序（见 §14）。

### 5.2 Component 注册表

```rust
pub struct ComponentId(u32);
pub struct ComponentInfo {
    id: ComponentId,
    name: Cow<'static, str>,         // 动态组件可运行时命名
    layout: Layout,
    storage: StorageType,            // Table | SparseSet | Shared | OwningGroup 成员
    align_simd: u8,                  // 热组件按 SIMD 宽对齐
    gpu_resident: bool,              // §15 常驻列
    drop: Option<unsafe fn(*mut u8)>,
    hooks: ComponentHooks,
    required: SmallVec<ComponentId>, // §16.1
}
```

### 5.3 Archetype 与 Chunk

- **Archetype** = 唯一组件集合（含关系 pair），持有一组 **Chunk**。
- **Chunk** = 固定 16KiB 块，列式 SoA；每列按 SIMD 宽对齐。chunk 槽内嵌 `change_version` + 可选 `ChunkMeta`（AABB / 可见位 / LOD / 批次键 / cell 归属）。
- `N`（每 chunk 容量）= `floor((16KiB − 元数据) / 单实体列字节和)`。

> **现状诚实注记（chunk 物理布局）**：`pkg/prism_ecs` 现已落地**物理分块存储 + chunk-aware fetch**。一个 archetype 按 `rows_per_chunk` 把行切成 chunk，每个 chunk 是**一个物理块**，块内按列式 SoA 依次存放所有列在该 row 窗口的切片（块级分配/回收、按 64B 对齐、增长只追加新块，故**已有块地址稳定**，无 realloc 搬移）；行 `r` 落在块 `r / rows_per_chunk` 的第 `r % rows_per_chunk` 槽，块 `c` 与逻辑 chunk `c` 一一对应。**逐行 fetch 不经过块表**：`QueryData::init_fetch_for_chunk` + `TableFetch` 在每个 chunk 入口解析一次块基址与 change-version cell，逐行 `fetch` 退化为 `base + (row - first_row) * size` 的纯偏移；`QueryIter` / `fold` / `par_for_each(_mut)` 按 chunk 推进，单实体 `get` 按行所属 chunk 解析。切片驱动（`simd_iter` / `par_chunks`）把每个切片夹在**单块**内（跨块切片在入口被拒），SIMD 核拿到的仍是连续 SoA run。
>
> **DOTS 式单块 chunk（已落地）**：一个 chunk = **一个物理块**，块头依次是 `[ 每列的 change_version cell | ChunkMeta | 各列的 SoA 切片 ]`，全部在**同一个分配**里（design §5.3「chunk 槽内嵌 change_version + ChunkMeta」）。`Table` 持有块、`Column` 是块内视图、块级分配/回收；`Table` 在 `allocate`/`swap_remove_row` 时按需分配/释放整块，并挂接/摘除各列的切片与版本 cell。有测试断言"**块数 == chunk 数**（而非 chunk × 列数）"，以及"change-version cell 与 `ChunkMeta` 的地址都落在该块分配范围内"。逐行的块基址、change-version cell 由 chunk-aware fetch 在每 chunk 解析一次。§22 高风险项第 1 条至此完成（剩余规模项只有大页/NUMA 与真机压测）。
>
> 注：曾直接先换块存储再谈 fetch，逐行 fetch 每行多一次块指针间接，实测可变串行 6.6 → ~8.7 ns/实体、`fold` 优势从 ~4.5× 塌到 ~1.2×；因此改为**先 chunk-aware fetch、再块存储**。两步落地后热路径回到基线（见 §17.1），且块存储带来稳定地址与块级回收。

- 结构性变更 = archetype 间搬迁（swap-remove + append），经并行 ECB 批量化（§9）。

---

## 6. 存储模型：四态

组件在 derive 时声明存储类型（默认 Table）：

```rust
#[derive(Component)] struct Position(Vec3);                        // Table（默认）
#[derive(Component)] #[component(storage = "sparse")] struct Selected;   // SparseSet
#[derive(Component)] #[component(storage = "shared")]
struct RenderBatchKey { mesh: AssetId, material: AssetId }         // Shared：按值聚簇
```

- **Table/Chunk**（默认）：列式 chunk。迭代最快、可 SIMD、chunk 版本号粗跳过、可按 chunk 并行。
- **SparseSet**：`sparse: Vec<u32>` + `dense: Vec<(Entity, T)>`。增删不搬迁 archetype，适合频繁 toggle 的 tag/临时组件。
- **SharedComponent**（Unity 式）：值去重存一份并按值分裂 archetype，天然形成渲染/物理批次 key，直接喂 `prism_render_scene` 的 instancing/合批。
- **OwningGroup**（EnTT 式）：把超热查询声明为 owning group，结构变更时维持完美打包连续段，迭代退化为线性扫描（零分支）。单一拥有者约束，冲突在初始化期报错。

查询对四态透明；fetch 层按存储类型单态化不同取数路径。

---

## 7. 查询系统

```rust
Query<&T> / Query<&mut T> / Query<Option<&T>>
Query<(Entity, Has<T>)>                     // 在场探针：yield bool，不排除原型、不读值、不注册 access
Query<AnyOf<(&A, &B)>>                      // 至少一个在场：yield (Option<&A>, Option<&B>)，全缺的行不迭代
Query<(Entity, &A, &B)>
Query<&A, (With<B>, Without<C>)>
Query<&A, Or<(Changed<B>, Added<C>)>>
Query<Relations<ChildOf>>                 // §11
```

- **archetype 匹配缓存**：首次构建匹配列表；archetype 新增时增量更新，杜绝每帧全表扫描。
- **查询健全性（world 身份 + 结构世代 + 缓存同步）**：`QueryState` 记录构造时的 [`WorldId`]，**所有**安全入口——`iter`/`iter_mut`/`get`/`get_mut`/`single`/`single_mut`/`get_many`/`get_many_mut`/`iter_many` 与并行入口 `par_for_each(_mut)`/`par_chunks(_mut)`——先校验传入 `World` 的身份，跨 World 复用**立即 panic**而非把另一世界的 dense 列错误重解释为本查询的类型（设计 §7/§20）。系统层同样绑定：`FunctionSystem::initialize` 记录 `WorldId`，`run`/`run_unsafe`/`apply_deferred` 校验，杜绝「在 A 上 initialize、在 B 上 run」的跨 World 复用（`system/function.rs`）。archetype 注册表被**整体替换**（snapshot `restore`）时递增 `World::archetype_generation`，使所有匹配缓存失效重建，杜绝陈旧 `ArchetypeId` 指向已释放/重编号列的隐患（设计 §10/§14）。匹配缓存由自旋锁保护（`query/matched.rs`），`QueryState` 的 `Send + Sync` 由锁保证而非不安全的 `UnsafeCell` 约定——多线程共享同一 `&QueryState` 并发 `iter` 不再有数据竞争（并发回归测试见 `query/cache_concurrency_tests.rs`）；匹配扫描**不在持锁期间**调用用户可实现的 `D::matches`/`F::matches`（避免重入死锁，且中途 panic 不会留下半更新缓存），结果在短临界区内原子提交。`&mut Query` 的 `IntoIterator` 绑定到 `&mut` 借用生命周期 `'i`（而非 world 生存期 `'w`），安全代码无法同时取得两个别名可变迭代器（compile-fail 守护见 `system/query_param.rs`）。
- **`Has<T>` 在场探针**：yield `bool`——实体是否持有 `T`，不排除任何 archetype、不读组件值、**不注册任何 access**（故可与同组件 `&mut T` 共存，`Query<(&mut T, Has<T>)>` 可靠无别名冲突）。统一覆盖 table / sparse / shared 三态存储（§6）；是 `Option<&T>` 的只测在场版（后者借值，前者只报在场）。
- **`AnyOf<(..)>` 并集查询**：元组 `(A, B)` 的 `||` 对偶——archetype **至少一个**元素在场即匹配，yield 各元素的 `Option`，且仅当该行真有某元素在场才迭代（永不迭代全 `None` 行）。元素可为 `&T`/`&mut T`/`Ref<T>`，统一覆盖 table/sparse/shared；注册元素访问的**并集**（与 `Option<&T>` 同样保守），故仍与写同组件的项冲突、自冲突集（如 `AnyOf<(&mut T, &T)>`）被访问检查拒绝。用于「同一职责但以不同类型存储的组件」。
- **随机访问与基数约束**：`get`/`get_mut` 按实体 `O(1)` 直达（实体 location → 原型门 `D::matches && F::matches` → 逐行门 `F::filter_fetch`/稀疏在场 → `fetch`），失败返 `QueryEntityError::{NoSuchEntity, QueryDoesNotMatch, AliasedMutability}`；`contains` 是其布尔对偶。`single`/`single_mut` 断言「**恰好一个**匹配」——取首行、见到第二行即拒（单命中不停在首行之后、多命中不停在第二行之后），失败返 `QuerySingleError::{NoEntities, MultipleEntities}`（携带 `type_name::<D>()` 供诊断）。两种错误均实现 `Display` + `core::error::Error`（可 `?` 上抛）；`Query` 系统参数同形提供上述全部入口（`get`/`get_mut`/`contains`/`single`/`single_mut`），与静态迭代共享同一「原型门 → 逐行门」契约。
- **批量随机访问**：`get_many::<N>` / `get_many_mut::<N>` 按固定 id 列表解析，按列表序返回 `[Item; N]`，首个失败短路；`get_many_mut` 先查重、重复即 `QueryEntityError::AliasedMutability`（避免同一实体被 `&mut` 两次别名，且在构造任何 `&mut` 之前拒绝）。`iter_many(list)`（只读）与 `iter_many_mut(list)`（可变）是流式对偶，逐 id 产出 `Result<Item, QueryEntityError>`（`size_hint` 精确）；只读路径允许重复 id 各产出一次共享引用，可变路径**拒绝重复 id**（同一次迭代内再遇即 `AliasedMutability`，故不会对同一行给出两个 `&mut`）。`QueryManyIter::matched()` 跳过非匹配项。用于层级子节点 / 选择集 / 复制增量 / 回放命令等「已有 id 列表、无需原型扫描」的场景；`Query` 系统参数同形提供全部入口。
- **查询状态便捷访问**：`QueryState` 提供只读的 `contains`/`count`/`is_empty`/`for_each` 与可变对偶 `for_each_mut`（与 `Query` 系统参数同形），免去手写 `iter(world).count()` 之类；两者都走 `QueryIter::fold` 批量化快路径（§17.1）。`QueryIter::size_hint` 返回剩余行的**紧上界**（剩余原型行数之和，逐行门全通过时即精确），使 `collect`/`Vec` 一次定容。
- **成对/组合迭代**：`QueryState::iter_combinations::<K>(world)` 产出匹配行的**无序 K 子集**（Bevy `iter_combinations` 的只读半边），按行序**字典序**、确定可重复；`K=2` 即宽相等/约束求解要的候选对，`K=1` 退化为普通迭代，`K=0` 或 `K>匹配数` 产出为空。限定 `D: ReadOnlyQueryData`——一次 yield 交出 K 个 item，若允许 `&mut` 项，同一实体可同时出现在两个存活组合里而自别名；成对写入请用另一查询或命令缓冲。匹配行收集复用普通迭代器的逐行门（门逻辑只有一份，经 `QueryIter::next_with_entity` 暴露），代价 `O(匹配行 + K·C(匹配行, K))`。
- **系统 / raw 入口的 world 契约（补强）**：`System::validate_world` 钩子在 `System::run` **递增 world tick 之前**校验（并行执行器也在分配 change 窗口前校验），所以被拒绝的跨 world 调用不留副作用；为**不同 world 重新 `initialize`** 会重置 `last_run`，避免用旧世界的 tick 下界漏掉新世界的既有变更。所有 raw `*_from_ptr` 入口（含 `par_*_from_ptr`）的 safety 契约写明「world 必须是构造该 state 的 world」且迭代期间禁止并发结构修改（`World::restore`）。
- **脏块访问器**（核心）：`QueryState::dirty_chunks(world, last_run, this_run)` 与系统参数面 `Query::dirty_chunks()`（自动用本 system 的 `(last_run, this_run]` 窗口）仅遍历 `chunk.change_version` 落在窗口内的块，把提取/剔除/宽相从 O(匹配实体) 降到 O(脏块)；结果是逐行 `Changed<T>` 的**正确超集**，绝不漏行。
- **SIMD 批量核 + chunk 切片迭代**（已落地）：`simd` feature 提供 `add_assign`/`scale_assign`/`axpy_assign`/`sum` 等列批量核（`core::arch` SSE2/NEON，标量路径作为 golden 对拍）。`QueryState::simd_iter::<K>(&world)` / `simd_iter_mut::<K>(&mut World)` 顺序产出**chunk 对齐的类型化列切片**（每片 ≤ K 行、尾片可短）——切片本身就是内核输入（组件列是 `f32` 时即 `&[f32]`），不需要再打包一层；`par_chunks`/`par_chunks_mut` 是同一机制的并行版（§8.3）。数据项须 `ColumnSliceData`、过滤器须 `ArchetypalFilter`，两者都在入口处**一次性断言**（稀疏数据/逐行过滤器明确 panic，不静默降级）；`ColumnSliceData`/`ArchetypalFilter` 与 `multi_thread` 解耦，顺序切片迭代在基础构建即可用。
- **par_iter**：按 chunk 切分投递到 `prism_tasks`（见 §8），work-stealing 负载均衡。
- **disjoint 查询**：同 system 多个**不冲突**查询可共存（各自声明访问集，冲突在 `init_state` 期即 panic）。**`ParamSet`（让同 system 内可冲突查询逐个访问）仍为 PLANNED**：其成员类型要求 `D: 'static`（`SystemParam::State: 'static`），嵌套省略寿命的 `Query<&mut T>` 会被拒绝，需要先把 `QueryData` 的生命周期从类型参数中移除（Bevy 的 `QueryData for &mut T` 形态）或引入等价的 `IterQueryData` 拆分后才能提供符合人体工学的签名。

---

## 8. 系统与调度

### 8.1 System 参数推导

```rust
fn sys(q: Query<(&mut A, &B)>, res: Res<Config>, mut ev: EventWriter<Hit>, mut cmd: Commands) { ... }
```

每个 `SystemParam` 暴露 `(读/写 ComponentId / ResourceId)` 访问集，供执行器冲突分析。

### 8.2 冲突图执行器（system 间并行）

- 按访问集自动构建可并行 DAG（Bevy 式），无需手动锁。
- 固定阶段 `First → PreUpdate → Update → FixedUpdate → PostUpdate → Last`，可插 `SystemSet`。
- 排序/条件/状态：`before/after/chain`、`run_if(cond)`、`States` 状态机。

### 8.3 Fiber 作业图（system 内并行，Naughty Dog/DOOM）

- 在 `prism_tasks` 之上建**细粒度任务图**：原子计数器依赖 + 工作窃取 + fiber（或 async-task 近似）。
- 单个重 system（物理求解、蒙皮、剔除、提取）内部按 chunk 拆子作业并行，同步点近零。
- 与冲突图执行器融合：执行器调度 system，fiber 图调度 system 内子作业。暴露 `JobHandle` 作为 `SystemParam`。

```rust
fn skinning(mut q: Query<(&mut SkinnedMesh, &Pose)>, jobs: JobGraph) {
    jobs.par_chunks(&mut q, |chunk| { /* 每块一子作业，SIMD 蒙皮 */ });
}
```

### 8.4 独占与确定性

- `exclusive system` 需 `&mut World` 的在独占阶段跑（结构合并、snapshot、流送）。
- `determinism` 档位：稳定 system 排序、稳定实体/原型分配、固定步长 `FixedUpdate`、确定性归约。

---

## 9. 命令与结构性变更

- **Commands / 并行 ECB**：每线程独立命令缓冲，sync point 按**确定键排序**合并回放，合并同原型迁移，降原型图抖动与同步点数量。
- **结构变更批量化**：同一实体多次 insert/remove 合并为一次 archetype 搬迁。

```rust
fn spawn_bullets(mut cmd: Commands, q: Query<&Transform, With<Gun>>) {
    for t in &q { cmd.spawn((Bullet, *t, Velocity(FORWARD))); }
}
```

---

## 10. 变更检测与反应式内核

- **双层变更检测**：
  - *chunk 版本号*（粗）：`Changed<T>` 先比对 chunk 版本，整块未变直接跳过——空闲 chunk 零成本。
  - *per-component tick*（细）：精确判定单实体是否在 `[last_run, now)` 内被写，支撑 `Added<T>`/`Changed<T>`。
  - 相等保护写：避免无谓 tick 推进（沿用 `prism_ui_ecs` 现有桥接思路，防 ECS→signal→ECS 往返震荡）。
- **push 反应图**（Our Machinery Truth + SolidJS）：从"每帧轮询"升级为"**变更推送驱动**"。组件写入产生细粒度变更事件，下游派生/system 只在依赖变化时执行（去重、无毛刺）。与 observer/hook 统一为一套反应内核，对接 `prism_ui_reactive`。默认与轮询并存，按档位开启。价值：把"成本 ∝ 变化量"从提取推广到全部派生计算。

---

## 11. 关系系统（flecs 式）

关系是本设计相对 Bevy 的最大能力增强，统一表达层级、挂点、装备、技能目标、所属等。

```rust
world.entity(child).add_relation::<ChildOf>(parent);
world.entity(sword).add_relation::<EquippedBy>(hero);

Query<(Entity, Relations<ChildOf>)>
q.iter_targets::<ChildOf>(e)
world.query_pair::<ChildOf>(Wildcard)       // (ChildOf, *)
world.query_transitive::<LocatedIn>(room)   // 传递闭包
```

- pair 作为 archetype 的一部分（**fragmenting**，低基数）或走旁路索引（**non-fragmenting**，高基数关系）。
- **传递 / 通配 / 缓存**：传递关系闭包、通配符查询、关系索引缓存。
- **层级**：`ChildOf` 内建层级缓存 + 变换传播（供 `prism_transform`）。
- **级联删除**：parent 删除时按策略级联/孤立化。

---

## 12. Observer 与生命周期 Hook

- **组件 Hook**（低层、同步、随结构变更触发）：`on_add / on_insert / on_replace / on_remove`，用于自动注册/释放 GPU 资源、维护派生索引。
- **Observer**（高层、事件驱动，可冒泡）：

```rust
world.observe::<OnAdd, Mesh3d>(|t, mut cmd| { /* 自动补 GpuMeshHandle */ });
world.observe::<OnRemove, RelationFilter<ChildOf>>(|t| { /* 级联清理 */ });
```

- 支持针对组件、关系、自定义事件触发；可沿 `ChildOf` 向上冒泡（UI/游戏事件）。
- **变更观察总线（§24.2）**：`World::observe_changed::<T>(cb)` 订阅**变更检测**（`Changed<T>` 语义）而非结构变更。`World::flush_changed_observers()` 每个观察者每次 flush **至多回调一次**，携带**合并去重、按实体索引升序**的变更集合，观察者按注册序触发，保证确定序回放；`Schedule::run` / `run_parallel` 在每次运行末尾自动 flush（也可手动控制节奏）。回调内对同组件的写入会被安排到**下一次** flush（窗口先推进再回调），不会被本次窗口吞掉。

---

## 13. 大规模与大世界

### 13.1 World Partition cell 流送（UE5）

- 世界网格化为 cell，按视点/兴趣源流入流出；实体随 cell 加载/卸载，卸载时序列化到磁盘（对接 `prism_scene`/snapshot）。
- 悬垂引用靠**稳定实体 ID + 弱句柄解析**（卸载后 generation 失配安全失败）。
- `partition` feature；`WorldPartitionCell` 组件 + cell 调度集管理。**需大场景验证。**

### 13.2 实体 LOD / 休眠（Unreal MassEntity）

- 远处实体降更新频率（每 N 帧 tick）或塌缩为代理（impostor/统计体）。
- 休眠（dormant）：静止实体移出活跃调度集，事件唤醒。

### 13.3 64-bit 浮点原点重定位（Star Citizen）

- 以玩家/相机为局部原点，按 cell 存相对坐标 + cell 偏移；渲染/物理用局部 f32，消除远距抖动；双精度仅存 cell 原点。
- 新增 `GridCell` / `FloatingOrigin` 组件与变换传播适配，与 `prism_transform` 协同。

---

## 14. 确定性仿真与回滚/预测网络

- **确定性世界**：冻结迭代序（稳定键）+ 可选定点数运算路径；同输入→同输出。
- **快照/增量**：世界快照 + delta（EnTT snapshot 形态，chunk 增量编码控内存），支撑回滚到任意已确认帧并重放输入。
- **预测-回滚**（Quantum/GGPO）：客户端预测本地输入，收权威帧后回滚重放；**逐帧状态哈希校验去同步**。
- **双跑发散定位**（`determinism` feature，见 `world/snapshot/determinism.rs`）：逐帧哈希只告诉你*发散了*，不告诉*何时/何处*。审计器两段式定位——`FrameHashLog::first_divergence` 比对两次运行的逐帧状态哈希，返回**首个发散 tick**（哈希不符 / tick 节奏漂移 / 一方提前结束）；拿到坏帧后对两次运行各`snapshot()` 并调 `locate_divergence`，按与 `state_hash` 相同的确定性折叠序（tick 游标→分配器存活→实体表→各列 holder/变更 tick/值字节→资源）返回**首个发散的 `(entity, component)` 坐标**。它是 `structurally_eq` 的定位版：`locate_divergence(a,b).is_none() == a.structurally_eq(b)`。
- 多 World：主仿真 World + 预测 World 并存，回滚时从权威快照重放输入。
- `determinism` feature + 固定步长；对接后续 `prism_net` 网络模块。
- 验证：同输入双跑逐帧哈希一致；回滚 N 帧后状态与无回滚一致。**浮点跨平台一致为高风险，需定点路径或严格 flag。**

---

## 15. GPU 驱动

- 标记"GPU 常驻"组件列（如 `GpuInstance`），列数据持久映射到 GPU 缓冲，按**脏块增量上传**，CPU 不再逐实体打包。
- 内核提供脏块 + 列指针访问器（增能），实际上传在渲染侧（`prism_render_scene` / `prism_render_driver`）。
- 对接 GPU 驱动剔除/合批（§6 SharedComponent 批次键 + §13 cell 可见位）。**需真实设备验证，CPU 单测只验脏块产出正确。**

---

## 16. 其他高级功能

### 16.1 Required Components

```rust
#[derive(Component)] #[require(Transform, GlobalTransform, Visibility)]
struct Mesh3d(Handle<Mesh>);
```

插入自动补齐缺失的必需组件（可带默认构造器），消除 bundle 样板。

### 16.2 动态 / 运行时组件 + 热重载 + 实时调参

按 `name + layout + storage` 在运行时注册非 Rust 类型组件，服务脚本层、编辑器、数据驱动；配合 blob/raw 列存储与反射读写；支持热重载与实时调参。

### 16.3 Prefab + 继承（`IsA`）

prefab 作为实体模板；实例通过 `IsA` 关系继承其组件并可逐字段覆盖；配合 `prism_reflect` 做场景序列化。

### 16.4 Blob 资产

不可变、可共享、可被多实体引用的大块数据（碰撞网格、曲线、动画片段），按引用计数 + 内容寻址存储。

### 16.5 Snapshot / 多 World / 反射桥

- `world.snapshot()` / `restore()`：结构化（可差量）快照，服务回滚网络、`prism_ui_timetravel` 时间旅行、编辑器撤销。
- 反射桥（`prism_reflect`）：序列化、编辑器 inspector、脚本访问统一走反射，组件无需各自手写 serde。**已落地**（`reflect` feature，见 `src/reflect_bridge.rs` 与 §24.3）：`register_reflect_component::<T>()` 后可按 `StableTypeId` 动态 `get_reflect` / `get_reflect_mut` / `insert_reflect` / `remove_reflect`，并做 dense `ComponentId` ↔ `StableTypeId` 双向映射；组件仍住在类型化列里，注册表只存单态化操作，不装箱活值。**以反射为基座的序列化/复制也已落地**（`serialize` feature）：字段级 diff 走 `diff_reflect` / `apply_reflect_patch`（`Patch::Unchanged` 是真 no-op），按策略收发走 `replicated_diff_reflect`，实体二进制存档走 `save/load_reflect_archive`，跨版本读取走 `SchemaRegistry` + `Migration` 链——均见 §24.3。

### 16.6 ECS 检视器 + 系统火焰图 + 时间旅行

内核暴露诊断接口（原型/chunk 占用、system 耗时、变更量、关系图谱），供编辑器检视器 + 火焰图 + 时间旅行调试（对接 `prism_ui_devtools` / `prism_ui_inspector` / 远程协议）。

火焰图有两条互补的诊断面：`diagnostics/profiler.rs` 把 span 折叠成**自时间树**（flame graph）；`trace` feature（`diagnostics/trace/`）是其**对偶**——保留**有序事件流**（显式时间戳 + 并行 track），经 `to_chrome_json` 导出 Chrome Trace Event Format JSON，可在 `chrome://tracing` / Perfetto 离线查看。两者共用 `SystemInstrument` 捕获钩子，`TraceRecorder` 实时捕获层 `std` 门控。

### 16.7 Events

双缓冲事件（`EventReader/EventWriter`，跨帧）+ observer 式即时事件（同帧响应）两套并存。

---

## 17. 性能工程

- **列对齐 + SIMD**：热组件列按 SIMD 宽对齐 + 分块向量化（变换传播/粒子/蒙皮 4–16 宽）；标量 vs SIMD 逐值对拍。
- **owning group**：超热查询完美打包，迭代线性零分支。
- **命令排序回放**：并行录制 + 确定键排序批量应用，合并同原型迁移。
- **fiber 作业图**：重 system 内 chunk 子作业并行，近零同步点，多核近线性。
- **内存**：原型预热**已落地**为 `World::prewarm::<B>(entities) -> ArchetypeId`——按 bundle（**含 required components**）解析/创建目标 archetype，并预留每个表列、实体列与 chunk 版本窗口的容量；**不生成实体、不跑 hook/observer/构造器**，因此可以安全地在关卡加载时把首帧会 burst 的 bundle 先热起来（有测试断言预热后同 bundle 的 spawn 落在同一 archetype，且第二次预热只会扩大容量）。共享组件 bundle 会**明确 panic**：共享组件的 archetype 由 interned 值决定，光凭 bundle 类型无法解析（文档写明替代做法）。实体/原型空槽回收已落地（free list + 空原型复用）。**大页竞技场分配已落地**（`vm` feature，见 `storage/arena.rs`）：`ChunkArena` 从 `prism_platform` 的 VM 层预留**大块页对齐（平台支持时大页/巨页支撑）**的 segment 并 bump 切块，释放的块进 free list 复用，不再每 chunk 一次全局分配（无 `vm`/无 VM 后端时回退逐块全局分配，行为等价）。**NUMA**：当前是**first-touch 就近**——块由首次写它的线程所在节点物理支撑（经典默认策略，天然把热点 chunk 留在流入它的 worker 本地）；**显式绑定某节点（mbind/set_mempolicy）仍为 PLANNED**，因为 `prism_platform` 尚未暴露该接口，文档如实说明而非假装绑定。
- **GPU 常驻列直传**：脏块增量上传，省 CPU 打包。
- **成本 ∝ 变化量**：脏块访问器 + push 反应图把增量路径设为默认，禁止全表扫描成为常态。

### 17.1 实测（本机，release，2026-10）

量具：[`pkg/prism_ecs/tests/scale.rs`]，显式运行 `cargo test -p prism_ecs --release --features multi_thread --test scale -- --ignored --nocapture`。每档取 3 次中最快一次；测试**断言数值正确**、只**打印**耗时（不设墙钟阈值，避免在任意机器上变成 flaky）。

| 场景（100 万实体，`(Pos, Vel)` 两个 f32×3 列） | 耗时 | 单实体 | 吞吐 |
|---|---|---|---|
| spawn 100 万 | 178 ms | 178 ns | 5.6 M/s |
| 串行 `(&Vel, &mut Pos)` 传播（`QueryIter`） | 6.6 ms | 6.6 ns | 151 M/s |
| `par_for_each_mut`，1 线程 | 4.09 ms | 4.09 ns | 245 M/s |
| `par_for_each_mut`，2 线程 | 2.72 ms | 2.72 ns | 367 M/s（1.50×） |
| `par_for_each_mut`，4 线程 | 2.40 ms | 2.40 ns | 417 M/s（1.70×） |
| `par_for_each_mut`，8 线程 | 2.20 ms | 2.20 ns | 455 M/s（1.86×） |

只读/可变 × 串行/批量的对照（`scale_serial_vs_batch_diagnosis`，3 次重复稳定）：只读串行 3.9 ns vs `par(1)` 1.70 ns（**2.3×**）；可变串行 6.6 ns vs `par(1)` 4.0 ns（**1.63×**）。

**批量生成（`World::spawn_batch`）**：同机同场景 `spawn_batch` 稳定在 **50–78 ns/实体**，而逐个 `spawn` 在负载下于 **252–576 ns/实体** 波动 → **3.4–7.4×**。差异说明逐实体开销的大头确实是**每实体的 archetype 解析**（bundle id 收集 + required 展开 + 存储分类 + `get_or_create`）——批量把这件事只做一次，而 hooks / observers / required 默认值仍逐实体执行（有测试专门断言这三件事没有被"优化掉"）。含共享组件的 bundle 会回退到逐实体路径（共享值决定 archetype，无法一次解析），行为有测试覆盖。

**「成本 ∝ 变化量」实测（1M 行，改 1% = 10k 行）**：对同一个 100 万实体世界，把「逐行 `Changed<T>` 扫描」与「`dirty_chunks` 粗粒度遍历」放在一起量，并**分别测变化聚集与散布**——因为 chunk 粒度只在变化聚集时才真正省事：

| 变化分布 | 逐行 `Changed<T>` 扫描 | `dirty_chunks` 粗遍历 | 比值 | 粗层覆盖行数 |
|---|---|---|---|---|
| 聚集（改前 1 万行） | 3.8–5.8 ms | **9.6–10.5 µs** | **390–550×** | 10,230（≈改动量） |
| 散布（每 100 行改 1 行） | 3.0–3.7 ms | 21–71 µs | 43–142× | **1,000,000（每个 chunk 都脏）** |

诚实结论：粗粒度遍历本身在两种分布下都是**微秒级**（它是 O(chunk 数) = O(N / rows_per_chunk)，与改动量无关），所以相对逐行扫描永远有 40× 以上优势；但**它报出的覆盖行数**才是后续精炼的成本：聚集时覆盖 ≈ 改动量（这正是设计假设的场景，成本真的 ∝ 变化量），散布时覆盖整个世界——此时粗层仍能快速定位，但按行精炼会退化成全量工作。这条实测把设计的假设边界写清楚了，而不是笼统宣称"永远 ∝ 变化量"。

**SIMD 列批量核实测（`simd` feature，`query/simd.rs`，Neon 后端）**：把列核与「LLVM 已自动向量化的标量 `for` 循环」正面对比（`scale_simd_kernels_vs_scalar`，同一台 aarch64 机器，两种工作集各测 3 次取最快）：

| 核（ns/元素） | 1K 元素（4 KiB，L1 常驻） | 1M 元素（4 MiB，流式） |
|---|---|---|
| `add_assign` 向量 / 标量 | 0.12 / 0.08（**0.67×**） | 0.17–0.44 / 0.13–0.16（**0.30–0.89×**） |
| `scale_assign` 向量 | 0.16–0.33 | 0.13–0.17 |
| `axpy_assign` 向量 | 0.08–0.33 | 0.15–0.18 |
| `sum` 向量 / 标量 | 0.20–0.37 / 0.90–1.26（**3.5–4.4×**） | 0.24–0.30 / 1.00–1.30（**4.3×**） |

诚实结论（也是设计取舍的证据）：**元素级核（`add/scale/axpy`）没有跑赢**——LLVM 已把标量 `for` 循环自动向量化成 NEON，手写 intrinsic 在两种工作集下都没有正收益，`add_assign` 甚至稳定慢于标量（0.67×）；**真正赚到的是水平归约 `sum`**，稳定 **3.5–4.4×**——因为标量 `acc += v` 是**串行依赖链**（每步都要等上一步），向量路径把 4 条 lane 并行累加、只在收尾做一次归约，打掉了这条链。含义：显式 SIMD 的价值在**打破依赖链的归约/扫描/前缀和**，而非逐元素算术——后者交给编译器自动向量化即可，手写属多余。这条把「SIMD 迭代」的设计收益收敛到了真正有收益的核上，而不是笼统宣称"向量化一定更快"。

**实测结论（诚实）**：

- 百万级单线程迭代已在 **~6.6 ns/实体**量级，规模性不再只是推断。
- 多核**非线性**（早前 8 线程约 1.86×）。本机为共享开发机（测量期间有其它构建/agent 并发），且该场景是**内存带宽受限**的 24 B/实体读写。**已提供可复现的扩展曲线量具** `scale_parallel_speedup_curve`（`tests/scale.rs`）：对 [1,2,4,8,主机并行度] 各取 **9 次**采样，输出**中位数 ns/实体**、**均值 ± 95% 置信区间**、相对 1 worker 的 **speedup** 与 **parallel efficiency**（speedup / 线程数），并在表头打印"需独占空闲机器"的注意事项——它在共享机上会把宽置信区间与悲观 speedup 直接暴露出来，而不是掩盖。跑法：`cargo test -p prism_ecs --release --features multi_thread --test scale -- --ignored --nocapture scale_parallel_speedup_curve`。共享机上测到的曲线（1→2→4→8 线程 speedup ≈ 1.56×/1.90×/0.46×，8 线程点 CI 极宽）**不是可信曲线**，只说明共享机的干扰有多大；可信的近线性曲线仍需**独占机器 + NUMA/绑核**，仍属 PLANNED。
- **已修复的量化优化**：初测发现串行 `QueryIter` 比「`par_for_each_mut` + 1 线程」慢约 2.1×（可变）。定位为 `QueryIter::next` 的逐行开销：每行都要做两次 `Option` 判别 + 切片边界检查。改为在**已证明的不变量**（进入一个 archetype 时同时设置 `row_len` 与两个 fetch 游标，故 `row < row_len` ⇒ 游标已初始化且 `row < entities.len()`）下用 `get_unchecked`/`unwrap_unchecked` 去掉这两个分支，并给 `next` 加 `inline(always)`；随后实测可变 **9.05 → 6.6 ns/实体（1.37×）**、只读 **5.02 → 3.9 ns/实体（1.29×）**，全量测试无回归。
- **批量消费者快路径（`fold` 重写，原 PLANNED 现落地）+ 分块后的再测**：`for` 循环必须走 `Iterator::next`；给 `QueryIter` 实现了批量化的 `Iterator::fold` 重写（chunk 内紧凑内循环、实体列与 fetch 游标提到局部变量），`for_each`/`sum`/`count`/显式 `fold` 全部改走它，行序与访问集合与 `next` 完全一致（排序 / 计数 / 写回三条测试守护）。**连续列存时代**实测只读 `iter.for_each` 比 `for-next` 快约 **4–5×**（1.1–1.5 vs 4.5–6.6 ns/实体）。**改为物理分块列存 + chunk-aware fetch 后这条优势收窄到 ~1.1×**（只读 `iter.for_each` 3.5 vs `for-next` 3.9；可变 6.7 vs 6.9 ns/实体）：逐行成本此时由 chunk 局部 fetch 主导，迭代器协议不再是瓶颈，两条路径趋同——`fold` 快路径仍保留（它优化的是协议开销，存储/迭代策略变化时自动让位），但文档不再宣称固定倍数。绝对吞吐与基线相当（下表为最近一次实测）。

诚实边界：CPU 单测只能证机制正确；NUMA / GPU 常驻 / 独占机的多核近线性等规模指标仍需真实硬件与大场景压测，仍标注 PLANNED。

---

## 18. 易用性与 Bevy 迁移策略

对外 API 刻意贴近 `bevy_ecs`：`#[derive(Component/Bundle/Resource/Event)]`、`Query<...>`、`Res/ResMut`、`Commands`、`App::add_systems`、`States`、`run_if`、`observe`。

迁移路径：
1. 提供 `prism_ecs::prelude` 与 bevy_ecs 近同名导出。
2. 把 `prism_ui_ecs`（当前依赖 `bevy_ecs`）切到 `prism_ecs`，用其现有测试对拍验证。
3. 渲染/音频对接层（`prism_render_scene/material/visibility`、`prism_bevy`、`prism_audio_bevy`）逐个切换，保留公共 API，内部换内核。

---

## 19. crate 分层与模块布局

```
pkg/prism_ecs/                     # no_std + alloc 内核
  src/
    entity.rs                      # 代际索引、分配器、meta
    component/                     # 注册表、hooks、required、动态注册
    archetype.rs chunk.rs table.rs sparse.rs shared.rs owning_group.rs storage.rs arena.rs
    query/   { fetch, filter, iter, par, simd, dirty, state(cache) }
    system/  { param, function, exclusive, into_system }
    schedule/{ graph, set, condition, state, executor, fiber_job, phase }
    world.rs command.rs event.rs observer.rs relation.rs reaction.rs
    change.rs bundle.rs resource.rs prefab.rs blob.rs snapshot.rs
    partition/{ cell, lod, dormant, floating_origin }
    gpu_resident.rs diagnostics.rs reflect_bridge.rs
  features = ["std","multi_thread","simd","serialize","reflect","trace","determinism","partition","gpu_resident","vm"]

pkg/prism_ecs_macros/              # derive: Component/Bundle/SystemSet/Resource/Event/SystemParam/Relation 已实现
```

> **feature 现状诚实注记**：上方 feature 清单为**目标形态**。`pkg/prism_ecs/Cargo.toml` 当前**已落地**的 feature 为 `std` / `multi_thread` / `simd` / `partition` / `gpu_resident` / `determinism` / `trace` / **`reflect`** / **`serialize`** / **`vm`**（均有真实 `cfg` 门控代码与测试；`vm` 见 `storage/arena.rs`：VM 竞技场把 chunk 块从 `prism_platform` 的页对齐/大页 segment 里 bump 切出并回收，无 `vm` 时回退逐块全局分配。`determinism` 见 `world/snapshot/determinism.rs`，`trace` 见 `diagnostics/trace/`）。`reflect` 已按 §16.5 / §24.3 接入 `prism_reflect`：`src/reflect_bridge.rs` 提供 `World::register_reflect_component::<T>()`（登记 `StableTypeId → 单态化操作` 表）与 `get_reflect` / `get_reflect_mut` / `insert_reflect` / `remove_reflect` / `reflect_component_id` / `reflect_stable_id` / `reflect_entity`，热路径为一次哈希探测 + 一次间接调用，不把活组件装箱（`T: Reflect + FromReflect + Clone`；`get_reflect_mut` 走 `Mut`，因此会像直接 `get_mut` 一样标记 changed）。`serialize` 已落地为反射驱动的实体二进制存档（`src/reflect_archive.rs`，信封与语义见 §24.3）：`save_reflect_archive` 产出**字节确定**的存档，`load_reflect_archive` 同 id 存活则覆盖、缺失则新开实体并返回落地实体表。**跨版本读取已接通**：记录里带写入时的具体类型名，加载经 `World::reflect_schemas_mut()` 的 `SchemaRegistry` 解析到写入版本、按该版 `TypeInfo` 解码、跑 `Migration` 链升级后再落到今天的类型（见 §24.3）。`trace` 已落地为「结构化事件流 + Chrome/Perfetto 时间线导出」——作为 §16.6 火焰图 profiler 的对偶（profiler 折叠自时间树，trace 保留有序事件流），共用 `SystemInstrument` 捕获钩子、core `no_std + alloc`、`TraceRecorder` 实时捕获层 `std` 门控，非薄壳。

依赖：仅 `prism_math`、`prism_tasks`（std）、`prism_reflect`（可选）、`prism_platform`（可选，`vm` feature 的 VM 竞技场）。**不碰任何 `bevy_*`。**

---

## 20. 契约、不变量与版本化

- **下游 API 契约优先**：对接层公共 API 是红线；内部实现随便改，公共 API 改动必须同步迁移依赖方，保持全仓绿。
- **版本化新增契约（以实际导出名为准）**：`storage::{ChunkVersions, TARGET_CHUNK_BYTES, rows_per_chunk}`（每 chunk 一条变更版本）、`storage::ChunkMeta` + `storage::ChunkBounds`（每块元数据：AABB / 可见位 / LOD / 批次键 / cell 归属 + 结构 epoch，经 `Archetype::table().chunk_metas()` 与 `World::chunk_meta(_mut)` 暴露）、`query::DirtyChunk` + `QueryState::dirty_chunks`（草案名 `ChunkDirtyIndex`）、`WorldSnapshot` / `SnapshotDelta` / `SnapshotRing`（草案名 `EcsSnapshot`/`EcsDelta`）、`OwningGroupId`、`WorldPartitionCell`、`GridCell`/`FloatingOrigin`、`NodeId` / `ReactionGraph`（草案名 `ReactionHandle`）、`JobHandle`（来自 `prism_tasks`）、`WorldSnapshot::state_hash()`（草案名 `StateHash`，实际是逐帧 `u64` 哈希）、`GpuResidentColumn`。列出实际名是有意的：这一节是给别人照抄的契约面，旧稿的草案名与代码不符会让读者找不到对应类型。
- **不变量**：owning group 单一拥有者；chunk 版本单调递增；确定性稳定键；快照 roundtrip 等价；子世界实体迁移保 generation；Entity generation 为 `NonZeroU32`（保留 0、回收自增、回绕跳 0），index 0 为正常可用槽。

---

## 21. 路线图（M0–M6）与基准即规格

- **M0 内核**：代际 Entity + 分配器；Component 注册表；Archetype + Chunk 列存 + arena；derive(Component/Bundle)；基础 `Query` + `With/Without/Option`；`spawn/get/despawn`；Commands。→ 单线程可编译可测最小闭环 + S0 微基准。
- **M1 调度**：system 参数推导、冲突图执行器、Resources、Events、States/RunCondition/SystemSet。
- **M2 性能核心**：双层变更检测 + 脏块访问器、SparseSet、SIMD 列、owning group、查询缓存、命令排序回放。
- **M3 作业**：fiber 作业图 + system 内 chunk 并行；多线程执行器。
- **M4 关系/反应**：fragmenting relations + 传递/通配/层级；observer/hook；push 反应图；required components；prefab/`IsA`；blob。
- **M5 规模/网络/GPU**：World Partition 流送 + 实体 LOD/休眠 + 64-bit 浮点原点；snapshot/delta + 确定性 + 回滚/预测；GPU 常驻列直传。
- **M6 迁移/工具**：bevy_ecs 兼容 prelude；切换 `prism_ui_ecs` 与渲染/音频对接层对拍；ECS 检视器 + 火焰图 + 时间旅行。

**基准即规格**：每里程碑以微基准红绿为完成判据，性能回归阻断合入；SIMD 逐值等价；确定性双跑哈希一致；chunk 脏块 ∝ 变化量。核心价值集中在 **M2（chunk/脏块）+ M3（fiber）+ M5（流送/确定性网络/GPU 常驻）**。

---

## 22. 诚实边界与风险

- 本文含设计规格与现状：**M0–M5 内核及第 23 章增补已在 `pkg/prism_ecs/` 落地实现**（代际 Entity / 原型 chunk 列存 + **物理分块存储**（**DOTS 式单块 chunk**：一个 row 窗口 = 一个物理块，块头 = 每列 change_version cell + `ChunkMeta`，其后是所有列的 SoA 切片，`Table` 持有块、`Column` 是块内视图，块级分配回收 / 稳定地址 / 块局部切片 / 块内嵌元数据）+ **chunk-aware fetch**（每 chunk 解析一次块基址）/ 四态存储 / 查询+脏块访问器 / 调度+冲突图执行器 / fiber 作业图 / 关系+Observer+Hook / push 反应图 / 变更观察总线 / 规范层级关系 ChildOf（作用域命令 + 层级 despawn）/ 反射桥（动态访问 + 字段级 diff/复制策略 + 实体二进制存档 + 跨版本 schema 迁移）/ 提取契约（`ExtractResource`/`ExtractComponent` + 运行器）/ 快照+回滚 / 分区流送 / GPU 常驻列 / 原型预热 / 诊断接口均有实现与测试）；**仍为 PLANNED 的是** M6 迁移对接、第 24 章跨 crate 契约对齐（§24.1 已落地（lane→优先级派发 + 帧预算门控）；§24.2 的 ECS 侧变更观察总线已落地、跨 crate 接线待续；§24.3 已落地动态访问 / 字段级 diff 复制 / 实体存档 / 跨版本 schema 迁移；§24.4 已有四环对拍夹具，见 §24.4/§24.5）、§23.5 的渲染侧接线与流水线并行（ECS 侧提取契约已落地）、**显式 NUMA 绑定（`mbind`/`set_mempolicy`；当前为 first-touch 就近，大页竞技场已落地）**，以及独占机多核近线性 / GPU 真机常驻 / 跨平台浮点确定性等需硬件压测的指标（百万级单线程迭代已实测，见 §17.1）。
- 规模与多核扩展性、NUMA、GPU 常驻指标依赖真实硬件压测，单测不能替代。
- **高风险项**：
  1. **chunk 变更版本 + 脏块访问器（M2）**：触存储/原型/变更检测三核心，最复杂，须充分基准 + 等价测试，严禁一次性大改。
  2. **fiber 作业图（M3）**：并发正确性要害（数据竞争/借用），须 `UnsafeWorldCell` 访问证明 + 压测；可先 async-task 近似再上 fiber。
  3. **确定性/回滚（M5）**：浮点跨平台一致难，定点路径或严格 flag + 逐帧哈希；回滚内存靠 chunk 增量快照控制。
  4. **大世界流送（M5）**：卸载/重载实体引用悬垂，靠稳定 ID + 弱句柄解析。
  5. **关系 fragmenting vs non-fragmenting**：高基数关系勿 fragment（原型爆炸），走旁路索引。
  6. **冲突图执行器（M1）**：正确性要害，改动最保守。
- **已定位并修复的偶发失败（`JobDag` 双派发竞态，原待跟进项）**：全量测试此前偶发 `test result: FAILED`，一次在 `--features multi_thread`、一次在 `--all-features`（约 80 次复跑零复现，故长期挂在"待跟进"）。本轮先在 `--features multi_thread` 下把它钉到具体用例：`system::job_dag_tests::edge_provides_happens_before_for_values` 与 `sub_job_panic_propagates`，panic 位置统一是 `job_dag.rs` 的 `JobDag: node dispatched more than once`（后者还把真实的 `boom in sub-job` 掩盖掉，令 `should_panic` 失配）。**根因不在 `prism_tasks`，而在 `JobDag::dispatch` 的派发循环本身**：多线程路径用 `state.pending.iter()` **实时读取原子 pending** 来判定"谁是根"，但 `scope` 会在 driver 仍在遍历时就把刚 spawn 的根跑完（工作窃取 / 调用线程参与 join），该根把某个 join 节点的 pending 递减到 0——driver 随后遍历到该 join、读到 pending 已经 0，于是**再派发一次**，`.take()` 第二次返回 `None`。**修复**：根集合在移动进原子计数**之前**就按**初始入度** `indeg == 0` 固化成 `roots`，driver 只派发这些真根；其余节点一律由"最后完成的前驱"那次 `fetch_sub(AcqRel)` 观测到 `== 1` 时派发（同一原子计数对每个节点只可能有一个 `== 1` 观察者，故绝不重复）。**回归证据**：新增 `join_under_racing_roots_dispatches_each_node_once`（400 次双根 join，断言每个节点恰好跑一次）——修复前**稳定 3/3 复现** `node dispatched more than once`，修复后 40 次模块复跑 + 20 次 `multi_thread` 全量 + 20 次 `all-features` 全量共 80 次**零失败**。
- 与既有 `prism_ecs_refactor_plan_zh.md`（原地改写 bevy_ecs 的 Fork-and-Own 方案）为两条候选路线：本文是"从零自研（greenfield chunked-archetype）"，前者是"原地演化"。二者可择一，或先 Fork-and-Own 启动、再按本文 chunk/关系/作业/流送/确定性设计逐步替换内部实现，最终形态一致。

---

## 23. AAA 高级功能增补（v0.3）

本章补齐顶级 ECS 常被忽视、但在真实 AAA 项目里缺一不可的能力，均 feature/档位门控，默认不付成本。

### 23.1 实体禁用 / 切换（Disabled）

内置 `Disabled` 标记关系（flecs 形态）：禁用的实体默认从查询中剔除，但不释放存储，可零成本重新启用。用于对象池、编辑器隐藏、关卡数据层开关、休眠实体。查询可用 `.include_disabled()` 显式纳入。与 §13.2 实体 LOD/休眠协同：休眠=降频更新，禁用=彻底跳过。

### 23.2 关系删除策略（OnDelete / OnDeleteTarget）

借 flecs 的清理策略：当关系的目标实体被销毁时，按策略级联处理，避免悬垂关系：

| 策略 | 行为 |
|---|---|
| `Remove`（默认） | 仅从持有者移除该关系对 |
| `Delete` | 级联销毁持有者（如 `ChildOf` 父死子亡） |
| `Panic` | 调试档断言，捕获非法删除 |

策略在关系类型注册时声明；大世界流送卸载 cell（§13.1）时据此一致地清理跨 cell 关系。

### 23.3 排他关系与原型不变量（Exclusive）

标记为 `Exclusive` 的关系每个实体至多持有一个目标（如 `ChildOf`、`DockedTo`）：插入新对自动替换旧对，保证层级/状态机不变量，省去用户手写"先移除再插入"。配合 §11 的 fragmenting 开关：排他 + 低基数走 fragment（享原型加速），高基数走旁路索引（防原型爆炸）。

### 23.4 调度歧义检测与单步调试

- **歧义检测（ambiguity）**：构图时静态分析两 system 存在"同资源读写冲突且无显式顺序"时报告歧义集合，CI 可将其设为硬门禁，杜绝非确定执行序引入的隐性 bug。
- **系统单步（stepping）**：调试档支持按 system 粒度单步执行 Schedule，配合 §16.6 检视器逐步观察 World 变化，复现时序相关缺陷。

### 23.5 渲染提取管线（Pipelined Extract）—— 解耦渲染与仿真

为解决 §2 中 `prism_render_scene` 的最重接缝：定义**提取阶段**与**渲染 World**。仿真 World 每帧经只读 `Extract` system 把可见数据抽取进独立渲染 World；渲染可与下一帧仿真**流水线并行**（Bevy pipelined rendering 形态）。

- 契约：渲染侧只依赖稳定的"提取 trait"（`ExtractComponent`/`ExtractResource`），不反向依赖仿真内部布局。
- 收益：仿真/渲染双 World 隔离，使 `prism_render_scene` 脱 Bevy 时只需实现提取 trait，而非耦合整套存储；也为 §15 GPU 常驻直传提供自然边界。

**ECS 侧已落地（提取契约 + 运行器）**：`pkg/prism_ecs/src/extract.rs` 提供文档承诺的两个稳定 trait——`ExtractResource`（主世界资源 → 次世界资源：`extract(&self, &World) -> Option<Out>`，返回 `None` 即本帧不覆盖，次世界保留上一帧值）与 `ExtractComponent`（`type QueryData: ReadOnlyQueryData` / `type QueryFilter` / `extract(item) -> Option<Out>`；提取器自己声明读哪些组件，次世界代码里**从不出现仿真组件名**）；配套运行器 `extract_resource::<R>(main, secondary) -> bool` 与 `extract_components::<C>(main, secondary) -> Vec<(Entity, Entity)>`。契约是**单向只读**：运行器只以 `&World` 读主世界、查询数据一律 `ReadOnlyQueryData`，所以提取既不会改写仿真状态，也不会被变更检测当成写（有专门用例断言 `Changed` 不因提取而触发）；匹配行沿用普通迭代器，`Added`/`Changed`/稀疏在场门一个不少。次世界实体由**它自己的分配器**新建，运行器返回 `(主实体, 次实体)` 映射，调用方按需存成 `RenderEntity` 式标记——不强行让两个分配器 id 相同。接入点已经现成：`prism_app` 的 `App::set_extract(sub_app, FnMut(&mut World, &mut World))` 正是本契约的形状（其文档写明"更丰富的 trait 驱动 `ExtractComponent` 管线尚未在 `prism_ecs` 提供"），现在可直接由两个运行器组合而成。**仍未做**：流水线并行的双缓冲渲染世界（M3 Inc2）与渲染侧实际改接——属渲染/App 侧工作。

### 23.6 作用域命令队列与层级 despawn

- **作用域命令**：`commands.entity(e).with_children(|b| ...)` 式分层构建，子作业命令归属父作用域，批量应用时按作用域确定序回放（配合 §9 并行 ECB）。
- **层级 despawn**：`despawn_recursive` 沿 `ChildOf`（排他 + `OnDelete=Delete`，见 23.2/23.3）一致级联，保证不漏不悬垂；与快照/回滚（§16.5）协同保持结构一致。

**已落地（`hierarchy.rs` + `command/`）**：引擎现在有**规范的层级关系** `ChildOf`（`Relation::KIND`：排他 = 一个父节点；`on_delete_target = Delete` = 父死子亡，走 §23.2 级联）。两项操作建立在其上——`EntityCommands::with_children(|children| { children.spawn(bundle); ... })` 是**作用域命令**（子实体在同一个嵌套 builder 里"生成 + 挂到父节点"，队列按书写顺序回放、且保证 spawn 先于关系添加，因此确定序可复现；`ChildBuilder::parent()` 可读出作用域父节点），`EntityCommands::add_child(existing)` 用于把已有实体挂进来（`ChildOf` 排他，重复挂载会**搬家**而不是双亲）。`World::despawn_recursive(entity)` 深度优先收集子树后**由深到浅**删除（先删子再删父，hook/observer 沿途总能看到父节点仍存在），返回**子树规模**——不是"`despawn` 返回 true 的次数"，因为 Delete 级联可能在中途先删掉某个成员，计数不能依赖这种交错；遍历带 visited 集合，畸形环也能终止；对已死实体的调用返回 `0`。这些入口按需自动注册 `ChildOf`（`World::register_child_of()` 幂等），因此不用层级的世界不付注册成本。

### 23.7 处理器 LOD（MassEntity 形态）

把"海量轻实体"的更新组织为**处理器（processor）**：按距离/重要度分档，不同档位以不同频率、不同精度批量跑（如远处 AI 10Hz 粗更新、近处 60Hz 精更新）。处理器消费 chunk 批次（§6）、受 §13.2 LOD 调度，使 CPU 成本随"活跃近处实体数"而非"总实体数"增长。

### 23.8 诚实边界

本章 23.1/23.2/23.3（实体禁用 / 关系删除策略 / 排他关系，见 `relation.rs`）、23.4（歧义检测与单步，见 `schedule/{ambiguity,stepping}.rs`）、23.6（作用域命令 / 层级 despawn，见 `command/`）、23.7（处理器 LOD，见 `partition/processor.rs`）均已落地实现并带测试；**仅 23.5 渲染提取管线仍为 PLANNED**（随 M3 渲染重接前置落地，对解耦收益最高）。

---

## 24. 跨 crate 契约对齐（v0.4）

`prism_tasks`/`prism_time`/`prism_reflect`/`prism_transform` 升级到 v0.2 后新增了若干跨 crate 能力。本章把它们**回接进 ECS**，明确「哪部分真相归谁」，避免两处各实现一套、契约漂移。本章不引入 ECS 新机制，只做对齐与引用。

### 24.1 调度执行底座对齐（tasks QoS 车道 + time 帧预算）

ECS 冲突图执行器（§8.2）与 fiber 作业图（§8.3）**不自建线程池/优先级**，而是派发到 `prism_tasks`：

- system 与 system 内子作业按关键度映射到 tasks 的**优先级车道**（见 tasks §24.1）：关键路径（物理/提取/传播）走 `Critical`，常规 gameplay 走 `Normal`，异步预计算走 `Background`。
- 执行受 `prism_time` **帧预算**（time §24.4）约束：预算耗尽时 `Background` system 顺延下帧，关键 system 必达。**已落地（ECS 侧）**：`Schedule::run_budgeted(world, budget)` 与 `Schedule::run_parallel_budgeted(pool, world, budget)` 在单线程与并行执行器上实现该门控——`Critical`/`Normal` 无论预算始终执行，预算耗尽后所有 `Background` 系统**本帧跳过并计数**（下帧预算允许时再跑）；`SystemLane` → `prism_tasks::Priority` 的映射与 `PriorityGroup` 派发也已接入（`schedule/{lane,executor,parallel_executor,graph}.rs`，带单线程/并行两条预算测试）。诚实边界：预算是墙钟门控，故只用于可延后的背景档；`determinism` 档仍走不带预算的固定执行序。
- **单一真相**：并行与调度策略实现在 tasks；时钟/预算在 time；ECS 只声明「访问集 + 顺序 + 车道标注」。确定性并行归并序由 tasks §24.7 保证（见 §24.4）。

### 24.2 Observer 作为跨系统空间 / 派生更新总线

ECS §12 Observer 是其他 crate 派生更新的统一入口，避免轮询：

- `prism_transform` 的 `OnChanged<GlobalTransform>`（transform §24.4）、空间加速结构增量同步（transform §24.5）均经 ECS Observer 触发。
- ECS 保证 Observer **批量合并**（本帧多次变化只回调一次）与**确定序回放**（配合 §14 确定性），使空间索引/音频重定位/阴影失效的触发可预测、可回滚。

**ECS 侧已落地**：`World::observe_changed::<T>(cb)` / `World::flush_changed_observers()`（见 `change_observers.rs`）实现上述两条保证——按帧合并（同一实体多次变化只投递一次）与确定序（观察者按注册序、变更集合按实体索引升序）；回调经 `ChangeContext` 拿到 `&mut World` 与合并后的变更集合，可安全做结构变更，且回调自身的写入顺延到下一次 flush，天然避免「回调触发自己」的抖动。`Schedule::run`/`run_parallel` 末尾自动 flush。「跨 crate 的 `prism_transform` 触发点接线」仍待 transform 侧 v0.2 契约落地后接入。

### 24.3 反射驱动的动态组件 / 快照 / 网络增量

ECS §16 的动态组件、snapshot、prefab 继承、反射桥**以 `prism_reflect` v0.2 为类型基座**：

- 动态组件布局/访问走 reflect §24.1 静态 `TypeInfo` + §24.2 访问器缓存（零注册、热路径 O(1)）。
- snapshot/存档走 reflect §24.3 二进制零拷贝；跨版本读取走 schema 迁移。
- 字段级网络增量（配合 `prism_replication`）走 reflect §24.4 diff + §24.5 字段级复制 + ECS §10 变更检测；`StableTypeId` 为跨会话/跨机组件标识。

**ECS 侧已落地**：`reflect` feature 下的 `src/reflect_bridge.rs` 把「按类型名动态读写组件」变成真实可用的 API——`register_reflect_component::<T>()` 登记 `StableTypeId → {get, get_mut, apply, remove}` 单态化函数指针表，随后 `get_reflect` / `get_reflect_mut` / `insert_reflect` / `remove_reflect` 全部只认 `&dyn Reflect` 与 `StableTypeId`，调用点不再出现 Rust 类型名（编辑器检视器、场景装载、存档迁移、复制层正是这种用法）。`insert_reflect` 同时接受**具体值**（`downcast` 快路径）与 **`Dynamic*` 树**（存档解码/`reflect_clone` 的产物，经 `FromReflect` 还原）；`StableTypeId` 由类型路径派生，因此可安全写进存档/网络流；`reflect_component_id` / `reflect_stable_id` 提供「世界内 dense id ↔ 稳定 id」双向映射。**字段级增量也已接上第一段**：`diff_reflect(entity, id, authoritative)` 直接产出 `prism_reflect::Patch`（只含变化字段），`apply_reflect_patch(entity, id, &patch)` 把补丁写回活组件；`Patch::Unchanged` 是**真 no-op**（既不写也不标 changed），所以已经同步的接收端不会因「空补丁」反复触发 `Changed<T>`/脏块。**二进制存档也已接上**（`serialize` feature，见本节末与 §24.3）：实体经 `save_reflect_archive` / `load_reflect_archive` 往返回读，值走 `prism_reflect` 的二进制格式（含其反序列化限额），跨版本读取由 `SchemaRegistry` + `Migration` 链承担。

**复制策略一层**：`replicated_diff_reflect(entity, id, authoritative, &ReplicationPlan)` 在 `diff_reflect` 之上叠加 `prism_reflect::ReplicationPlan` 的字段许可——补丁只覆盖策略选中的字段（例如服务端专有字段永不下发），再交给 `apply_reflect_patch` 落地，构成「按策略只发差异字段」的完整收发回路。

**存档一段已落地（`serialize` feature）**：`src/reflect_archive.rs` 提供 `World::save_reflect_archive(entities) -> Result<Vec<u8>, _>` 与 `load_reflect_archive(&[u8]) -> Result<Vec<Entity>, _>`。信封 = `"PECA" | 版本 | 实体数 | { 实体bits, 组件数, [类型名, StableTypeId, blob_len, prism_reflect::to_binary blob] }`；组件按 `StableTypeId` 升序写出，所以同一世界产出**字节确定**的存档（可哈希/可 diff/可按内容缓存）。回读语义明确：同 id 仍存活则原地覆盖，缺失则新开实体，返回按存档序的落地实体表供上层建映射。坏 magic / 版本不符 / 截断 / 实体位非法 / 名字与 id 不符 / 目标世界未注册该类型 / 值解码失败，都返回带原因的 `ReflectArchiveError`（复用 `prism_reflect` 的二进制格式与其反序列化限额）。

**跨版本读取已落地（schema 迁移）**：每条组件记录都带**写入时的具体类型名**（与 `prism_reflect::DynamicScene` 的取法一致），因此存档能扛住类型演进：加载时用 `World::reflect_schemas_mut()` 里的 `SchemaRegistry` 把「写入方的具体类型名」解析成 `(logical, 写入版本)`，**按那一版的 `TypeInfo` 解码**（不是按今天的形状），再跑 `Migration` 的 `vN → vN+1` 链把 `DynamicStruct` 升到当前版本，最后才按当前版本的类型名找到 `ComponentId` 并用 `FromReflect` 落到今天的具体类型上。没有注册 schema 的类型走「按当前形状直接解码」的原路。测试用真实的两个版本结构（`HealthV1{hp}` → `HealthV2{hp,shield}`）+ 一条补字段的迁移，验证旧存档读进新构建后 `hp` 保留、新字段取默认值；未注册映射的旧形状则明确报 `UnknownType` 而不是猜。

### 24.4 确定性链路四方对齐

ECS 确定性仿真（§14）不是孤立的，必须与三个地基 crate 的确定性档**同时成立**，否则链路任一环破坏即整体失确定：

| 环节 | 契约来源 | 要求 |
|---|---|---|
| 系统/实体序 | ECS §8.4 / §14 | 稳定 system 排序、稳定实体/原型分配 |
| 并行归并 | tasks §24.7 | 确定归并树，结果与线程数/窃取序无关 |
| 时间步长 | time §10 / §24.8 | 有理/定点 `fixed_dt`、整数 tick、确定性审计 |
| 空间传播 | transform §12 | 定点层级传播，位级一致 |

四者共享同一句契约：**同输入 → 位等价输出**。`determinism` 档联调时用 time §24.1 录制回放 + §24.8 审计双跑定位首个发散 tick。

**已落地（可执行对拍夹具）**：`pkg/prism_ecs/tests/determinism_link.rs`（需 `--features multi_thread`）把四环串成一条真实链路——`prism_time::TickClock`（1/60 **有理**步长 + 整数 tick）驱动 `prism_ecs::Schedule` 在**并行执行器**上运行，每 tick 取 `WorldSnapshot::state_hash()`；并行归并走 `prism_tasks::TaskPool::deterministic_ordered` 做**非交换**折叠；空间传播走 `prism_transform::propagate_fixed` + `hash_globals`（整数定点、链式层级）。同一场景在 **1 与 8 worker** 下逐 tick 的四项值全部**位等价**；开启 `determinism` feature 时再把逐 tick 哈希喂给 `FrameHashLog::first_divergence`，断言无发散、失败时直接点名首个发散 tick。诚实边界：这证明的是**该场景下四环同时成立**，不等于「任意场景都确定」。

### 24.5 诚实边界

本章为跨 crate 职责对齐说明：**§24.1 调度执行底座已落地**（冲突图执行器把并行波次派发到 `prism_tasks::TaskPool`，`SystemLane` → `Priority` 经 `PriorityGroup` 分优先级派发，且帧预算门控 `Background` 系统见 `schedule/{graph,config,lane,executor,parallel_executor}.rs`）；**§24.2 的 ECS 侧已落地**（变更观察总线 `change_observers.rs`：`observe_changed` / `flush_changed_observers`，按帧合并 + 确定序 + `Schedule::run` 自动 flush；跨 crate 的 transform 触发点仍待接）；**§24.3 的 ECS 侧已落地**（`reflect` feature + `reflect_bridge.rs` 按 `StableTypeId` 动态读写组件，`serialize` feature + `reflect_archive.rs` 落字段级 diff / 复制策略 / 实体二进制存档 / 跨版本 schema 迁移，见 §24.3）；**§24.4 已有可执行的四环对拍夹具**（`tests/determinism_link.rs`，见 §24.4），但「任意场景确定」仍需各 crate 的确定性档全覆盖。本章不引入 ECS 新机制，仅固定跨 crate 职责边界与引用。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。
