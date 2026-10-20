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

- 版本: v0.4（M0–M5、第 23/24 章与 M6 已有多项内核及跨 crate 实现；不以整章标签代替验收。v0.2→v0.3 新增第 23 章「AAA 高级功能增补」；v0.3→v0.4 新增第 24 章「跨 crate 契约对齐」。当前实现边界与剩余目标见 §22、§23.8、§24.5）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: `prism_math`（glam SIMD）、`prism_tasks`（work-stealing + fiber 作业图，`std`/`multi_thread`）、`prism_reflect`（可选，序列化/反射/脚本桥）
- 明确约束: 核心 `no_std + alloc`；`std` / `multi_thread` / `serialize` / `reflect` / `trace` / `determinism` / `simd` / `partition` 为 feature；**不依赖任何 `bevy_*` crate**

> **现状核查口径（2026-10-21；当前工作树含其他未提交改动）**：本文同时保留完整设计目标与当前实现边界。「已实现」与「已验证」分开记录；提交历史中的测试入口或历史验收记录不等于跨机、NAT、真实 GPU 或性能目标完成。此前修复 dirty-query 保守 fallback 后未推进 column-journal cursor、导致旧写入跨窗口重报的问题；后续 column journal k 路归并与 dirty-chunk 候选小集去重优化已在当前源码。shared-binding eviction 后缀清理快路径后的默认 `cargo test -p prism_ecs` 全包通过：库 890 项、集成测试和 doctests 均通过；all-features 全包通过：库 1240 项、集成测试和 doctests 均通过；`cargo check -p prism_ecs --no-default-features` 通过。为适应共享机器磁盘压力，这轮测试关闭 debuginfo/incremental，使用独立临时 target。定向回归覆盖 eviction-event 探测命中、缓存扫描分支未命中、shared posting 后缀与非后缀淘汰。默认/all-features 测试不代表其他 crate、GPU、跨机或受控性能目标通过，也不证明并行中的其他系统改动已并入 ECS 交付。§17.1 与其他历史修复段落中的测量值/通过次数仍是既有记录。
>
> **历史补验记录（同日）**：新增高基数回归后，以隔离 target 重跑默认整包：当时库 876 项通过，集成测试/doctests 通过；scale 3 项通过、3 项忽略。该历史计数已被后续改动覆盖，当前验证口径见上方现状核查。
>
> **journal 溢出补验（同日历史记录）**：新增 `journal_overflow_falls_back_without_missing_chunks_or_replaying_them`，以超过 8,192 条的真实 World 写事件验证 retained-journal gap 后 indexed fallback 找回全部 1,024 个 chunk，下一前向窗口不重报。当时定向测试与默认/all-features 整包均通过；该次库测试分别为 877/1227 项，后续验证已覆盖此计数。

> **本次 main 历史同步依据**：UDP/ACK/rebind 与 localhost 跨进程验收：`39bd3806a`、`0bf59f55a`、`d6000586b`、`c7201323e`、`1290d101b`；NUMA acceptance：`9390b7a5e`、`07a3d5b9c`；audio/source-route、设备 callback lifecycle 与 event-log：`ea2ed8026`、`cd03b14c7`、`3e503dd38`、`94d763d70`；GPU resident/wgpu gate 与 benchmark：`22aebe26`、`a288cf12e`、`c7201323e`、`3a8120874`、`0e7e67259`；CPU SDF clipmap planner：`7e99fa30f`、`90ff7a89a`；shader package binding 与水体 parity：`a1e33a798`；heightfield/planetary placement 与 deletion：`13428ef38`、`2afc018fb`。

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

ECS 是整个 Prism 脱离 Bevy 工程的"命门"。迁移目标是把算法层与 ECS 对接层解耦，并保持近 Bevy 的使用体验；迁移不保证只需「改 import」，还要核对调度、生命周期、提取和回滚语义。当前 `prism_ui_ecs`、`prism_render_app`、`prism_scene` 与 `prism_audio_ecs` 已有原生 `prism_ecs` 链路；`prism_audio_bevy` 已收敛为仅重导出 `prism_audio_ecs` 的兼容 facade，不再维护第二套 Bevy 音频实现。仍不能把代码迁移完成等同于应用工作流和联合验收完成（见 §21/§22）。

**一句话定位（目标）**：`prism_ecs` 是 Prism 的"权威仿真内核"——面向百万~千万级实体、每帧数千~数万结构变更与 GPU 驱动提取，追求增量路径成本随**变化量**增长、多核近线性扩展，并原生支撑大世界流送、确定性回滚网络、GPU 常驻。**这不是所有路径已达成的复杂度/吞吐保证**：全量查询仍处理匹配数据；Table chunk 的正常写入发现已具备变化源索引，但查询匹配与下游反应/上传仍有各自成本，见 §7/§10/§17/§22。

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
| **capability** | host 显式提供、与编译 feature 交叉检查的能力 | 可用 worker 数、SIMD 宽度、GPU 常驻映射、VM；不从档位名猜硬件 |
| **quality tier** | 项目可自定义的内容/场景规模 profile | mobile / console / high-end-desktop / custom（显式实体准入预算、chunk 批处理策略、LOD 与流送参数） |
| **feature flag** | 编译期裁剪 | `simd` / `multi_thread` / `partition` / `determinism` / `reflect` |

目标：移动端只付基础 chunk 迭代成本；高端桌面叠加 fiber 作业图 + 大世界流送 + GPU 常驻 + 回滚网络。

**当前链路**：`pkg/prism_ecs/src/policy.rs` 的 `EcsPolicy::resolve` 校验显式 `EcsCapabilities`/`EcsProfile`；`pkg/prism_app/src/ecs_policy.rs` 的 `configure_ecs_policy` / `apply_ecs_policy` 可配置/替换主 App 的 schedule/QoS 池。`ecs_policy/partition.rs` 消费现有 LOD/partition 实现，由 host 提供位置、帧号并完成加载，不固定业务组件或自动选视点。实体预算仅约束 `check_entity_admission` / `try_spawn_with_policy` 等**显式准入入口**，不是所有 `World`/`Commands`/hook 分配的全局硬限额；独立 sub-app/渲染池也不自动纳入主池预算。chunk 策略必须使用表的真实 `rows_per_chunk`，不能凭 profile 改变现有物理布局或宣称实现任意 chunk 大小。能力声明不等于成功分配大页/GPU 映射或 NUMA 绑定。

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
> **DOTS 式单块 chunk（已有实现）**：一个 chunk = **一个物理块**，块头依次是 `[ 每列的 change_version cell | ChunkMeta | 各列的 SoA 切片 ]`，全部在**同一个分配**里（design §5.3「chunk 槽内嵌 change_version + ChunkMeta」）。`Table` 持有块、`Column` 是块内视图、块级分配/回收；`Table` 在 `allocate`/`swap_remove_row` 时按需分配/释放整块，并挂接/摘除各列的切片与版本 cell。有测试断言"**块数 == chunk 数**（而非 chunk × 列数）"，以及"change-version cell 与 `ChunkMeta` 的地址都落在该块分配范围内"。逐行的块基址、change-version cell 由 chunk-aware fetch 在每 chunk 解析一次。这完成了物理布局机制，不代表 §22 风险项全部验收：写屏障的并发/回绕正确性、脏块发现复杂度及受控规模验证仍需分别检查；VM arena 现状见 §17。
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

- **archetype 匹配缓存**：首次构建匹配列表；archetype 新增时增量更新，杜绝每帧全表扫描。迭代器持有不可变 COW `Arc<Vec<ArchetypeId>>` 快照：缓存稳定时获取快照为 O(1)，不再每次复制全部匹配 ID；无 reader 持有快照时，新增匹配 archetype 复用 `Vec` 摊还扩容，避免每次增长都复制整份 ID 列表；存在 reader 时 `Arc::make_mut` 会复制以保留旧快照不可变。restore 或动态布局重载后的结构世代变化会重建缓存。
- **查询健全性（world 身份 + 结构世代 + 缓存同步）**：`QueryState` 记录构造时的 [`WorldId`]，**所有**安全入口——`iter`/`iter_mut`/`get`/`get_mut`/`single`/`single_mut`/`get_many`/`get_many_mut`/`iter_many` 与并行入口 `par_for_each(_mut)`/`par_chunks(_mut)`——先校验传入 `World` 的身份，跨 World 复用**立即 panic**而非把另一世界的 dense 列错误重解释为本查询的类型（设计 §7/§20）。系统层同样绑定：`FunctionSystem::initialize` 记录 `WorldId`，`run`/`run_unsafe`/`apply_deferred` 校验，杜绝「在 A 上 initialize、在 B 上 run」的跨 World 复用（`system/function.rs`）。archetype 注册表被**整体替换**（snapshot `restore`）时递增 `World::archetype_generation`，使所有匹配缓存失效重建，杜绝陈旧 `ArchetypeId` 指向已释放/重编号列的隐患（设计 §10/§14）。匹配缓存由自旋锁保护（`query/matched.rs`），`QueryState` 的 `Send + Sync` 由锁保证而非不安全的 `UnsafeCell` 约定——多线程共享同一 `&QueryState` 并发 `iter` 不再有数据竞争（并发回归测试见 `query/cache_concurrency_tests.rs`）；匹配扫描**不在持锁期间**调用用户可实现的 `D::matches`/`F::matches`（避免重入死锁，且中途 panic 不会留下半更新缓存），结果在短临界区内原子提交。`&mut Query` 的 `IntoIterator` 绑定到 `&mut` 借用生命周期 `'i`（而非 world 生存期 `'w`），安全代码无法同时取得两个别名可变迭代器（compile-fail 守护见 `system/query_param.rs`）。
- **`Has<T>` 在场探针**：yield `bool`——实体是否持有 `T`，不排除任何 archetype、不读组件值、**不注册任何 access**（故可与同组件 `&mut T` 共存，`Query<(&mut T, Has<T>)>` 可靠无别名冲突）。统一覆盖 table / sparse / shared 三态存储（§6）；是 `Option<&T>` 的只测在场版（后者借值，前者只报在场）。
- **`AnyOf<(..)>` 并集查询**：元组 `(A, B)` 的 `||` 对偶——archetype **至少一个**元素在场即匹配，yield 各元素的 `Option`，且仅当该行真有某元素在场才迭代（永不迭代全 `None` 行）。元素可为 `&T`/`&mut T`/`Ref<T>`，统一覆盖 table/sparse/shared；注册元素访问的**并集**（与 `Option<&T>` 同样保守），故仍与写同组件的项冲突、自冲突集（如 `AnyOf<(&mut T, &T)>`）被访问检查拒绝。用于「同一职责但以不同类型存储的组件」。
- **随机访问与基数约束**：`get`/`get_mut` 按实体 `O(1)` 直达（实体 location → 原型门 `D::matches && F::matches` → 逐行门 `F::filter_fetch`/稀疏在场 → `fetch`），失败返 `QueryEntityError::{NoSuchEntity, QueryDoesNotMatch, AliasedMutability}`；`contains` 是其布尔对偶。`single`/`single_mut` 断言「**恰好一个**匹配」——取首行、见到第二行即拒（单命中不停在首行之后、多命中不停在第二行之后），失败返 `QuerySingleError::{NoEntities, MultipleEntities}`（携带 `type_name::<D>()` 供诊断）。两种错误均实现 `Display` + `core::error::Error`（可 `?` 上抛）；`Query` 系统参数同形提供上述全部入口（`get`/`get_mut`/`contains`/`single`/`single_mut`），与静态迭代共享同一「原型门 → 逐行门」契约。
- **批量随机访问**：`get_many::<N>` / `get_many_mut::<N>` 按固定 id 列表解析，按列表序返回 `[Item; N]`，首个失败短路；`get_many_mut` 先查重、重复即 `QueryEntityError::AliasedMutability`（避免同一实体被 `&mut` 两次别名，且在构造任何 `&mut` 之前拒绝）。`iter_many(list)`（只读）与 `iter_many_mut(list)`（可变）是流式对偶，逐 id 产出 `Result<Item, QueryEntityError>`（`size_hint` 精确）；只读路径允许重复 id 各产出一次共享引用，可变路径**拒绝重复 id**（同一次迭代内再遇即 `AliasedMutability`，故不会对同一行给出两个 `&mut`）。`QueryManyIter::matched()` 跳过非匹配项。用于层级子节点 / 选择集 / 复制增量 / 回放命令等「已有 id 列表、无需原型扫描」的场景；`Query` 系统参数同形提供全部入口。
- **查询状态便捷访问**：`QueryState` 提供只读的 `contains`/`count`/`is_empty`/`for_each` 与可变对偶 `for_each_mut`（与 `Query` 系统参数同形），免去手写 `iter(world).count()` 之类；两者都走 `QueryIter::fold` 批量化快路径（§17.1）。`QueryIter::size_hint` 返回剩余行的**紧上界**（剩余原型行数之和，逐行门全通过时即精确），使 `collect`/`Vec` 一次定容。
- **成对/组合迭代**：`QueryState::iter_combinations::<K>(world)` 产出匹配行的**无序 K 子集**（Bevy `iter_combinations` 的只读半边），按行序**字典序**、确定可重复；`K=2` 即宽相等/约束求解要的候选对，`K=1` 退化为普通迭代，`K=0` 或 `K>匹配数` 产出为空。限定 `D: ReadOnlyQueryData`——一次 yield 交出 K 个 item，若允许 `&mut` 项，同一实体可同时出现在两个存活组合里而自别名；成对写入请用另一查询或命令缓冲。匹配行收集复用普通迭代器的逐行门（门逻辑只有一份，经 `QueryIter::next_with_entity` 暴露），代价 `O(匹配行 + K·C(匹配行, K))`。
- **系统 / raw 入口的 world 契约（补强）**：`System::validate_world` 钩子在 `System::run` **递增 world tick 之前**校验（并行执行器也在分配 change 窗口前校验），所以被拒绝的跨 world 调用不留副作用；为**不同 world 重新 `initialize`** 会重置 `last_run`，避免用旧世界的 tick 下界漏掉新世界的既有变更。所有 raw `*_from_ptr` 入口（含 `par_*_from_ptr`）的 safety 契约写明「world 必须是构造该 state 的 world」且迭代期间禁止并发结构修改（`World::restore`）。
- **脏块访问器**（核心）：`QueryState::dirty_chunks(world, last_run, this_run)` 与系统参数面 `Query::dirty_chunks()`（自动用本 system 的 `(last_run, this_run]` 窗口）返回版本命中的 Table 行窗口。新增 `dirty_chunks_with_stats()` / `DirtyChunkReport` 提供可测的候选列数、列候选叶数、合并候选数、过滤拒绝数、摘要读取数与输出覆盖行数；它们是复杂度/回归量具，不是墙钟承诺。多列路径对每个访问列使用自己的层级 chunk 版本索引，再对已排序候选叶做多路合并，避免把全部候选重新全局排序；来源数不超过 4 时使用无堆分配的线性快路径，更高来源数使用最小堆 k 路归并，合并成本为 `O(U log C)`（`C` 为来源数，`U` 为来源项数；低来源数线性路径是固定上界），过滤器的 `filter_chunk` 在合并后保守拒绝，故 `Changed<T>` / `Or` / AND 组合不会把 clean chunk 当成可交付窗口。SparseSet 的写入、插入和移除另外通过每个 `QueryState` 持有的有界 change-index cursor 投递到实体当前行，再与稀疏集合的 tick 检查合并；query 激活后的前向窗口不扫描整个 live set；早于 query 激活边界、重放或与上次请求重叠的窗口不消费 forward cursor，而执行一次有界 live-set fallback。`DirtyChunkStats::sparse_index_events`、`sparse_fallback_scans` 与 `sparse_entities_checked` 是该路径的成本量具；没有活跃 query/observer 时不会保留事件历史。总发现成本仍包括各列索引实际读取的摘要数 `S`、候选项合并、匹配 archetype 遍历及 `O(E_s)` 稀疏变更事件/集合检查；实现**没有证明**严格 `O(change)`，`S`、候选数与匹配 archetype/chunk 规模仍可能随世界规模增长。粗窗口允许多报，不替代逐行过滤与 SparseSet 在场检查；写屏障/版本回绕不变量必须保持，不能用「粗层是超集」掩盖这些正确性要求。
- **SIMD 批量核 + chunk 切片迭代**（已落地）：`simd` feature 提供 `add_assign`/`scale_assign`/`axpy_assign`/`sum` 等列批量核（`core::arch` SSE2/NEON，标量路径作为 golden 对拍）。`QueryState::simd_iter::<K>(&world)` / `simd_iter_mut::<K>(&mut World)` 顺序产出**chunk 对齐的类型化列切片**（每片 ≤ K 行、尾片可短）——切片本身就是内核输入（组件列是 `f32` 时即 `&[f32]`），不需要再打包一层；`par_chunks`/`par_chunks_mut` 是同一机制的并行版（§8.3）。数据项须 `ColumnSliceData`、过滤器须 `ArchetypalFilter`，两者都在入口处**一次性断言**（稀疏数据/逐行过滤器明确 panic，不静默降级）；`ColumnSliceData`/`ArchetypalFilter` 与 `multi_thread` 解耦，顺序切片迭代在基础构建即可用。
- **par_iter**：按 chunk 切分投递到 `prism_tasks`（见 §8），work-stealing 负载均衡。
- **disjoint 查询与 `ParamSet`**：同 system 多个**不冲突**查询可共存，普通参数的访问冲突在初始化时拒绝。`system/param_set.rs` 已实现 0–12 元组及嵌套 `ParamSet`，支持 `ParamSet<(Query<&mut T>, Query<&T>)>` 的普通函数签名。`p0()`…`p11()` **独占借用整个 set**，惰性 fetch，返回项的 world/state 两个寿命都缩短到该次借用，禁止同时保留冲突项或经 consuming API 逃逸；成员内部仍须无冲突，对外及调度的静态访问集为**所有成员的并集**（即使未 fetch），deferred apply/discard 按元组顺序。derive `SystemParam` 的结构体成员描述符需显式 `'static`，不是活值借用。运行时回归见 `tests/param_set.rs`，同时访问/嵌套/寿命逃逸的拒绝用例为模块内 `compile_fail` doctest；本次未重跑。

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

### 8.5 System panic 隔离（已实现）

- 默认 `SystemPanicPolicy::FailFast` 保持 panic 传播。`std` 构建可向 World 插入 `SystemPanicPolicy::DisableAndContinue`：系统体、延迟命令或私有 run condition 发生 unwind 时，永久停用对应节点并继续执行后续系统；共享 set condition 失败后永久关闭该条件。
- 串行、并行、预算执行和单步执行共用隔离策略；并行执行等待当前 wave 全部任务结束，再按调度顺序记录失败和清理。`SystemFailures` 提供有界、UTF-8 安全的失败记录，`StepReport::Failed` 标明单步失败。
- 隔离不回滚已发生的直接写入或已执行命令。失败系统通过 `System::discard_deferred` / `SystemParam::discard` 丢弃未执行命令，并仅回收其自身尚未放入 archetype 的 spawn 预留；其他队列的预留和已成功 spawn 的空实体均保留。自定义缓冲参数需实现 `discard` 回收自己拥有的预留。
- 初始化错误、非法调度图、跨 World 误用仍然 fail-fast；`panic = "abort"` 无法隔离。清理回调及析构函数应避免再次 panic。labeled schedule 的暂时移出使用作用域守卫，普通 unwind 后仍归还原注册表。
- 回归位于 `pkg/prism_ecs/tests/panic_isolation.rs`，覆盖失败停用、条件组合、延迟回放中断、预留回收、generation 复用、并行同 wave 成功保留，以及 labeled schedule 的 unwind 恢复。
- 此切片不保证任意构造、析构或结构写入中途 panic 后 World 仍可继续使用；存储层的完整 unwind 安全需要独立完善，`catch_unwind` 本身不恢复这些不变量。

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
  - *chunk 版本号*（粗）：正常写入先通过 tick→chunk 变化源索引定位窗口内候选，再用 chunk 摘要/代际校验保持保守正确性；**跳过 payload 不等于端到端零成本**，仍需枚举匹配 archetype/访问列并合并候选，rebuild/restore/日志缺口还可回退 interval tree。普通查询各迭代入口的 skip/写屏障一致性另需回归验证。
  - *per-component tick*（细）：精确判定单实体是否在 `(last_run, this_run]` 内被写，支撑 `Added<T>`/`Changed<T>`。
  - *restore 日志边界*：snapshot restore 会推进 change-index epoch、清空旧批次，并将已注册 change observer 的 pending batch 与 cursor 重置到恢复后的 tick 基线；恢复前时间线的变更不会重放，而恢复后同 tick 的首次写入仍可被观察。该边界由 restore + observer 回归测试覆盖。
  - 相等保护写：避免无谓 tick 推进（沿用 `prism_ui_ecs` 现有桥接思路，防 ECS→signal→ECS 往返震荡）。
- **push 反应图**（Our Machinery Truth + SolidJS）：`reaction.rs` 已有去重、拓扑顺序的派生 DAG；`world_reactions.rs` 的 `reaction_source::<T>` / `add_reaction` 将组件变更批次绑定到 World，先收齐源批次再执行 dirty 子图，effect 写入下次 flush 才可见。panic 后保留未完成批次供重试，不回滚已发生的用户副作用。**当前是「候选索引 + 源日志复核 + 下游 push」**：`change_observers.rs` 在 flush 时先由脏组件摘要和活动索引选择候选 observer，已消除空闲帧、无关组件写入和历史移除槽造成的全 observer-slot 扫描；候选内部仍须通过对应组件日志与实体 tick 收集、复核变化源，并非每次组件写入都直接入全局事件队列，不能宣称端到端 O(变化量)。`prism_ui_ecs/src/schedule.rs` 已接 signal queue：ECS→signal 消费观察批次；signal→ECS 仅处理 `SignalChangeQueue` 通知的 two-way bindings，干净帧不遍历绑定、不读 signal 值；通知处理 panic 后未确认项保留待重试。剩余工作是降低高基数候选内部日志复核、大批次回调与下游消费成本，不能因候选索引、UI push 与下游 DAG 为增量路径就宣称整个反应图端到端 O(变化量)。

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

- 目标：标记"GPU 常驻"组件列（如 `GpuInstance`），维护 GPU 列镜像并按**脏块增量上传**，减少 CPU 逐实体打包；不要求所有后端都能把普通 ECS 分配直接持久映射给 GPU。
- 内核提供脏块 + 列指针访问器；`pkg/prism_render_app/src/ecs_columns.rs` 已有实际提取消费，`plugin.rs` 安装该提取链，设备上传与同步由渲染侧负责，不是「渲染侧完全待接」。`pkg/prism_render_app/tests/ecs_columns.rs` 已有 resident-column 回读、局部脏列上传、设备丢失恢复与旧缓冲释放的验收入口（`22aebe26b`）；`os-backends/prism_render_driver_wgpu/tests/gpu.rs` 已要求 physical GPU evidence，缺少真实 adapter 时明确 `SKIPPED`，`PRISM_REQUIRE_GPU=1` 时失败（`a288cf12e`）。这些是实现与验收入口，不是本次重跑或当前主机真实 GPU 通过证明。
- 对接 GPU 驱动剔除/合批（§6 SharedComponent 批次键 + §13 cell 可见位）。**CPU 侧已有 chunk 粒度机制**：`Table::iter_chunks()` 产出行窗口及 `ChunkMeta`，`GpuBatchBuilder::build_from_table(table)` 按 `batch_key`/`visible` 生成 `GpuBatchPlan`；`slot` 对应 GPU 列镜像的实体行号。`Table::column_upload_spans(id, last_upload, now, out)` 检查每列各 chunk 的 `change_version`，产出 `(buffer_offset, src, len)`，渲染侧按 span 复制。**发现仍为 O(该列 chunk 总数)，复制量才随脏窗口的 payload 增长**；不可把总成本写成仅 O(脏 chunk 数)。`World::recompute_chunk_bounds::<T>` / `Table::recompute_chunk_bounds` 按投影计算 AABB，以结构 epoch + 变更版本决定是否重扫该窗口；干净帧仍逐窗口判定。`benches/benches/prism_virtual_texture/main.rs`（`c7201323e`，并由 `3a8120874` 扩展）已有 CPU feedback/residency 与真实 adapter 条件下的 page-table lookup、feedback-decode readback benchmark；无真实 adapter 时是 `SKIPPED`，严格模式失败。当前仍不能宣称物理 GPU 已通过、GPU mapping/sync 生命周期、上传收益、端到端帧性能或 GPU SDF 已完成；本轮未重跑这些入口。

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

**当前已有对外活值链路**（`pkg/prism_ecs/src/world/dynamic_reload.rs`）：`World::{register_dynamic_component, insert_dynamic_component, get_dynamic_component, get_dynamic_component_mut, remove_dynamic_component, reload_dynamic_component}` 按 `ComponentId` 操作真实 Table/SparseSet 值；注册 hooks/required 有显式入口。`DynamicComponentLayout` 的 unsafe ABI 描述符由调用方保证 layout、Send/Sync、drop/clone/hash glue 一致且代码仍存活；raw 类型重解释仍是 unsafe，不会由反射自动证明。插入先 stage initializer/required，遵循 tick 与生命周期语义；可变指针访问标记 changed，单纯只读不标记。

**迁移事务边界**：先为所有受影响原型/稀疏活值转换并分配，再发布新布局；转换 `Err`、未初始化输出或转换 panic 时，按 ABI 契约不修改源值，丢弃 staged 值，旧值/布局/tick/schema 保持不变。成功保留 added tick，更新 changed tick；非目标列、共享绑定、位置和稀疏顺序保留。即便同 layout 的值迁移也递增 schema 与结构 generation，使 Query 缓存失效，旧快照不可 restore；旧快照保留原 glue，仍可安全读取/析构（schema guard 见 `world/snapshot/{capture,restore,delta}.rs`）。旧值析构在**发布之后**：其 panic 不撤销新 World；分配 abort 或 unwind 中再次析构 panic 不承诺恢复。普通插入/删除的用户 hook 副作用也不属于事务回滚。

**明确不支持的边界**：live reload 拒绝 Rust 类型组件、Shared、带 hook 或被旧 schema required 构造器引用的组件，不更换 storage 策略、不发伪造生命周期事件。动态 ABI 与反射注册是两条契约：尚不等于脚本 VM/编辑器自动编译并迁移任意类型，也不支持任意 dylib 卸载后继续使用旧 glue。回归入口 `world/dynamic_reload_tests.rs` 覆盖多原型/非目标列、Sparse、失败与析构 panic、同布局快照、高对齐/ZST、hook/required 等限制；这里只核对测试源码存在，未重跑。

### 16.3 Prefab + 继承（`IsA`）

prefab 作为实体模板；实例通过 `IsA` 关系继承其组件并可逐字段覆盖；配合 `prism_reflect` 做场景序列化。

**当前实现**：`prefab.rs` 提供实例/祖先关系及组件级 `get_inherited::<T>`；`reflect` 下的 `prefab_reflect.rs` 提供 `get_inherited_reflect`、字段 override/清除、组件移除标记与 reset。按广度优先、直接父插入序解析，近层显式字段获胜；只存覆盖策略，不把继承默认值缓存成实例值，因此下次解析可见父级修改。无 base 的字段策略可 dormant，重叠路径拒绝，路径/类型失配返回错误，祖先须存活且无环。**普通 `get`/Query 不会自动获得字段合成值**，`get_inherited::<T>` 仍是组件级解析，消费者需显式选择反射解析或上层物化。

`serialize` 的 `reflect_archive.rs` 已用 PECA V3 保存闭合祖先图、父优先序与字段策略（包括 dormant/removal），保留 V2 组件存档读取；提供实体映射和显式引用 remap，不猜测整数是 Entity。覆盖策略要求精确 schema layout，组件 migration **不会自动推断字段所有权的迁移**；任意 hook/panic 下加载不承诺全 World 事务。测试入口 `tests/{prefab_inheritance,prefab_reflect,prefab_archive}.rs`；场景编辑器全流程与全部旧资产兼容仍需单独验收。

### 16.4 Blob 资产

不可变、可共享、可被多实体引用的大块数据（碰撞网格、曲线、动画片段），由
`prism_ecs::blob::BlobStore` 提供进程内闭环：

- `insert_bytes` / `try_insert_bytes` 按确定性的内容哈希分桶，并对候选执行完整字节比较；哈希碰撞不会导致错误别名。
- `BlobHandle` 是 slot index + generation 的轻量句柄；槽位回收会递增 generation，旧句柄的 `get`、`retain`、`release` 均安全失败。
- `insert_bytes` 与 `retain` 都增加引用；`try_release` 返回本次是否回收了存储，引用归零才从内容索引移除、释放载荷并回收槽位。
- `BlobId` 只表达“哈希 + 长度”的内容身份；它不是无碰撞证明，跨哈希桶的最终身份仍由完整字节比较确认。`find` / `contains` 不改变引用计数，适合只读查询。
- 需要错误分支的调用方使用 `try_insert_bytes`、`try_retain`、`try_release`；旧的布尔/句柄入口保留作便捷 API。引用计数溢出、槽位索引溢出和陈旧句柄不得静默回绕。
- `insert_pod` / `get_pod` 是 `unsafe` 便利层：调用者必须保证完整对象表示对 `T` 有效；任意原始字节应使用 `get`，不可把它们用于 `bool`、引用、`NonZero*` 或枚举等有额外有效性约束的类型。

物理列的 `storage::BlobVec` 不是 Blob 资产索引，而是 ECS 类型擦除列的块存储。它保留稳定块地址和 borrowed-table ownership；`try_reserve`、`try_attach_block`、`try_detach_block` 将容量溢出、借用列欠配和非法块生命周期提升为可检查错误，避免只在 debug 构建中依赖断言。

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
- **fiber 作业图**：重 system 内 chunk 子作业并行；减少同步开销与多核近线性是性能目标，不是所有工作负载的保证。
- **内存**：原型预热**已落地**为 `World::prewarm::<B>(entities) -> ArchetypeId`——按 bundle（**含 required components**）解析/创建目标 archetype，并预留每个表列、实体列与 chunk 版本窗口的容量；**不生成实体、不跑 hook/observer/构造器**，因此可以在加载时预热预计 burst 的 bundle（有对应回归源码）。共享组件 bundle 会**明确 panic**：共享值参与 archetype 身份，单凭 bundle 类型不能解析。实体/原型空槽回收已有 free list + 空原型复用。**大页竞技场分配已有实现**（`vm` feature，`storage/arena.rs`）：`ChunkArena` 从 `prism_platform` VM 层预留页对齐 segment 并 bump 切块，支持时尝试大页/巨页，释放块复用；无相应后端时回退逐块全局分配。NUMA/亲和已有真实 OS acceptance 入口：`prism_platform_os` topology/affinity readback、worker binding outcome 与 arena placement fallback 由 `pkg/prism_tasks/tests/numa_acceptance.rs` 验收（`9390b7a5e`、`07a3d5b9c`）；默认缺能力输出 `SKIPPED`，`PRISM_REQUIRE_NUMA=1` 时失败，arena 对 requested node 的 `Bound`/`Fallback` 结果有明确断言。它不等于 ECS arena 已证明 node-local pages，也不等于 Linux strict success、拓扑感知调度或 NUMA 性能已验证。
- **GPU 常驻列直传**：脏块增量上传，省 CPU 打包。
- **成本 ∝ 变化量（目标）**：Table chunk 的正常写入源发现已推进为 tick→chunk 索引，存储层窗口查询接近严格变化量；但匹配 archetype/访问列枚举、多列合并、观察注册项、绑定信号及下游执行/上传仍有独立成本。须分别量测发现、精炼、执行与上传，不能把这一切片等同于全链路零扫描。

### 17.1 实测（本机，release，2026-10）

**以下为保留的历史测量记录，不是本次文档核查重跑结果，也不是当前未提交实现的验收。** 不同记录跨越存储/迭代器版本；缺少与当前 checkout 对齐的提交、硬件/功耗、绑核/NUMA、背景负载与场景配置，以及受控 P50/P95/P99 和跨平台对拍证据。量具存在不等于性能回归已在 CI 阻断，也不应把历史倍数推广到任意机器/分布。

**当前 checkout 定向复测（2026-10-20，release，Darwin arm64；`multi_thread` feature，仅运行 `scale_dirty_chunks_vs_changed_scan_1m`）**：在同一个百万行 world 中修改 1%（10,000 行），连续启动 4 次独立测试进程。聚集修改时 `Changed<T>` 单次扫描为 **25.3–26.3 µs**，dirty-chunk 为 **6.6–17.1 µs**，覆盖 **10,230 行**（约 **1.5–3.9×**）；散布修改时扫描为 **1.187–1.447 ms**，dirty-chunk 为 **75.9–82.4 µs**，覆盖 **1,000,000 行**（约 **15.1–17.6×**）。这是当前优化后完整 accessor 对照，不是只有 journal discovery 的量测；测试每次只采一个样本/场景、未控制 CPU 负载/绑核/NUMA，聚集路径波动明显，不能据此设性能保证、CI 门槛或 P95/P99，也未拆出写入成本和候选归并各阶段。

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

诚实结论：历史样本中的粗粒度遍历为**微秒级**，记录的比值最低约 43×，不构成「永远高于 40×」保证；这些样本也早于当前变化源索引，不能当作本切片的验收。当前 Table chunk 正常写入的存储发现为 O(log B + B_w + K)，但 query 仍需匹配 archetype/访问列和多列合并，粗窗口精炼还取决于**覆盖行数**：聚集时接近改动量，散布时可覆盖全世界。因此总成本不能宣称端到端严格仅 ∝ 变化量；后续 `Changed` 迭代优化还可能改变对照基线。

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

迁移状态与剩余路径：
1. `prism_ecs::prelude` 与 Bevy ECS 常用 API 的近同名导出已实现；这提供常见调用形态，不承诺与 `bevy_ecs` 的完整 API/语义二进制兼容。
2. `prism_ui_ecs` 已直接依赖 `prism_ecs`，不再是待迁移项；`prism_scene` 与 `prism_render_app` 也已接入原生内核。仍须在真实应用中完成旧工程/场景资产加载、编辑、保存、运行时更新与回写的端到端对拍，不能由 crate 依赖或单测代替。
3. 音频已拆分为原生 `prism_audio_ecs` 与仅作兼容重导出的 `prism_audio_bevy` facade，不维护第二套 Bevy 音频实现；局部验证已有 30 个 unit tests、22 个 integration tests、4 个 facade integration tests 的历史记录。可用物理设备上的启停/恢复、音频资产导入/解码、播放头绑定、场景生命周期与完整应用工作流仍需联合验收。其他渲染/音频对接必须继续按公共 API、生命周期、调度和失败恢复逐项核验。

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
- **版本化新增契约（以实际导出名为准）**：`storage::{ChunkVersions, TARGET_CHUNK_BYTES, rows_per_chunk}`（每 chunk 一条变更版本）、`storage::ChunkMeta` + `storage::ChunkBounds`（每块元数据：AABB / 可见位 / LOD / 批次键 / cell 归属 + 结构 epoch，经 `Archetype::table().chunk_metas()` 与 `World::chunk_meta(_mut)` 暴露）、`query::DirtyChunk` + `QueryState::dirty_chunks`（草案名 `ChunkDirtyIndex`）、`WorldSnapshot` / `SnapshotDelta` / `SnapshotRing`（草案名 `EcsSnapshot`/`EcsDelta`）、`OwningGroupId`、`WorldPartitionCell`、`GridCell`/`FloatingOrigin`、`NodeId` / `ReactionGraph`（草案名 `ReactionHandle`）、`JobHandle`（来自 `prism_tasks`）、`WorldSnapshot::state_hash()`（草案名 `StateHash`，实际是逐帧 `u64` 哈希）、`GpuResidentColumn`（CPU 侧）、`storage::ChunkInfo` + `Table::iter_chunks()`（每 chunk 的行范围 + 内嵌 `ChunkMeta`，供 GPU 剔除/上传与合批消费）、`GpuBatchBuilder::build_from_table`、`storage::ColumnUploadSpan` + `Table::column_upload_spans()`（每列按 `change_version` 产出脏 chunk 的 `(buffer_offset, src, len)` 上传列表）、`ChunkMeta::{set_bounds, bounds_epoch, bounds_version}` + `World::recompute_chunk_bounds` / `Table::recompute_chunk_bounds`（按结构 epoch + change_version 增量产出的每 chunk AABB）、`culling::frustum_cull_aabb` + `World::cull_chunks` / `Table::cull_chunks`（视锥剔除写 `visible` 标记）、端到端 CPU 剔除流水线（`recompute_chunk_bounds` → `cull_chunks` → `build_from_table`）。列出实际名是有意的：这一节是给别人照抄的契约面，旧稿的草案名与代码不符会让读者找不到对应类型。
- **不变量**：owning group 单一拥有者；chunk 版本单调递增；确定性稳定键；快照 roundtrip 等价；子世界实体迁移保 generation；Entity generation 为 `NonZeroU32`（保留 0、回收自增、回绕跳 0），index 0 为正常可用槽。

---

## 21. 路线图（M0–M6）与基准即规格

- **M0 内核**：代际 Entity + 分配器；Component 注册表；Archetype + Chunk 列存 + arena；derive(Component/Bundle)；基础 `Query` + `With/Without/Option`；`spawn/get/despawn`；Commands。→ 单线程可编译可测最小闭环 + S0 微基准。
- **M1 调度**：system 参数推导、冲突图执行器、Resources、Events、States/RunCondition/SystemSet。
- **M2 性能核心**：双层变更检测 + 脏块访问器、SparseSet、SIMD 列、owning group、查询缓存、命令排序回放。
- **M3 作业**：fiber 作业图 + system 内 chunk 并行；多线程执行器。
- **M4 关系/反应**：fragmenting relations + 传递/通配/层级；observer/hook；push 反应图；required components；prefab/`IsA`；blob。
- **M5 规模/网络/GPU**：World Partition 流送 + 实体 LOD/休眠 + 64-bit 浮点原点；snapshot/delta + 确定性 + 回滚/预测；GPU 常驻列直传。
- **M6 迁移/工具（完整目标保留）**：bevy_ecs 兼容 prelude；UI、渲染、场景、音频对接层迁移与对拍；ECS 检视器 + 火焰图 + 时间旅行。当前 `prism_ui_ecs`、`prism_render_app`、`prism_scene` 已使用原生 ECS，`prism_audio_bevy` 已改为重导出 `prism_audio_ecs` 的兼容 facade；inspector/timetravel 有可选 `ecs` 接入；场景已有 materialize、嵌套实例、override replay 与 partition 接线，不能列作全未实现。**仍需完整应用/旧场景资产及编辑流程迁移验收、场景与音频联合工作流对拍，以及音频资产导入/解码、播放头绑定和场景生命周期接线**。

**基准即规格（验收目标）**：每里程碑须有正确性与受控性能基线，性能回归阻断合入；SIMD 逐值等价，确定性双跑哈希一致，并区分 chunk 发现总量与脏 payload 量。当前 `scale` 量具打印耗时不设墙钟阈值，不等于这些门槛已全部落实。核心价值仍集中在 **M2（chunk/脏块）+ M3（fiber）+ M5（流送/确定性网络/GPU 常驻）**；原目标不因标注边界而缩减。

---

## 22. 诚实边界与风险

- **本轮已验收增量**：提交 `78ac705a4` 与 `6d28cee97` 完成 shared-value archetype posting 回收及 QueryState tombstone 压缩；128 轮多 table-set churn 回归验证 posting、缓存压缩和旧 Arc 快照有效。随后增加 eviction ID 日志/QueryState 游标，并按积压事件量与缓存长度自适应选择探测路径；shared-binding eviction 后缀快路径后的默认整包库 890 项、all-features 整包库 1240 项、no-default-features check 均通过，默认/all-features 的集成测试与 doctests 也通过。新增扫描命中分支回归覆盖 eviction 积压前端/末端命中，并保留旧 Arc 内容。
- **Archetype tombstone 重对象释放**：永久 eviction 在稳定 ID 槽位上保留组件/关系轻量元数据，但将原 `Table` 替换为空表壳，释放旧列、chunk block、table 容量与 shared-value `Arc`；查询快照继续按稳定 `ArchetypeId` 存活并在刷新时过滤 evicted ID。回归验证空表无行/块/列、shared binding 已释放、旧快照仍有效，并覆盖 snapshot capture/restore。当前默认 890 项、all-features 1240 项整包及 no-default check 均通过。仍未实现 archetype registry 槽位压缩/ID 复用；eviction 日志及每个 tombstone 的轻量元数据仍随历史 churn 增长。
- **shared-binding eviction 索引与 posting 清理快路径**：`Archetypes::by_shared_binding` 按 `(ComponentId, SharedValueId)` 保存稳定有序 posting；淘汰候选仅从目标 pair 选取，并逐个重建 canonical key 删除 `by_key` 项。被淘汰 IDs 按 posting 分组：若待删 ID 与有序 posting 的尾部完全相等则直接 `truncate`，否则使用原有稳定过滤，因此不扫描常见的“新建后立即释放”后缀；非后缀淘汰仍为 O(posting 长度)。共同 table-component / shared-binding posting 均应用该规则。128 轮 churn 回归覆盖多 table-set 变体、posting 顺序、活跃 BatchKey 保留及 Layer 共绑定清理；非后缀回归覆盖先创建旧 archetype、后创建仍活跃 archetype、再淘汰旧项。当前 release 定向量具每个基数取 7 个样本、每样本 1,024 次淘汰：释放 tombstone 重对象后，后缀 churn 在 64/256/1,024/4,096 个并存值下约 2.46/2.65/2.40/2.33 μs/淘汰；非后缀量具 `tests/scale.rs::scale_shared_binding_eviction_non_tail`（每样本先建 1,024 个目标，再保留对应数量的较新 posting 项）约 1.20/1.31/1.54/2.20 μs/淘汰。更早的后缀快路径前后数据为历史共享机样本，不能与本次运行直接比较。当前数据仍是共享机器诊断值、非受控性能门槛；非后缀过滤成本随 posting 长度增长，需继续评估并复测。

- **当前源码包含的机制（不等于整章验收）**：物理单块 chunk / chunk-aware fetch / 四态存储 / ParamSet / 调度与作业图 / fragmenting relations / observer、hook 与反应图 / 快照回滚 / partition 与 LOD / 动态活值及布局迁移 / prefab 字段 override 与 V3 存档 / VM arena 均有实现入口。跨 crate 也已有 App 可选流水线与渲染提取、显式 policy 配置、transform observer 与 BVH/grid 消费、原生 UI 桥和场景物化，以及策略驱动的 ECS 网络回滚/帧哈希审计；不能再统一标为 PLANNED。各项安全契约与限制见对应章节；有源码入口、局部测试或 acceptance gate 仍不能替代对应平台和规模的完整验收。
- **尚未实现或未充分验收的关键范围**：Table chunk 正常写入的脏块源发现已不再是 O(匹配 chunk 总数)，并已补上 SparseSet 变化实体到当前 archetype 行窗口的 fallback；query 的匹配 archetype 遍历、候选物化与日志缺口/replay fallback 尚未统一为严格变化量，脏块多来源合并现有低来源数线性快路径和高来源数堆归并，但仍须在高基数真实 query 上基准。dirty-query journal 路径按 archetype 聚合 journal 与 SparseSet 候选，再做一次查询/过滤匹配；小候选集线性去重，大集合哈希去重并排序唯一 chunk。保守 tick-index fallback 现按访问的 table component 倒排索引生成 archetype 候选，并与 SparseSet 变化实体所在 archetype 做有序多路归并；少来源走线性路径，多来源用堆归并，避免先拼接全部 posting 再整体排序。对 `Option<&T>` 这类匹配大量无关 archetype 的查询，不再遍历完整既有匹配列表后逐列探测，匹配缓存仍只增量补测尚未检查的新 archetype。倒排表存储量随可查询的 archetype-component 关系数增长，共享值变体各自占 posting；shared value 最后引用释放后永久 evict 的空 archetype 会从 posting 回收，但稳定 ArchetypeId 与 registry tombstone 仍保留。本轮已释放 tombstone 持有的原 Table 列/chunk 分配与 shared-value `Arc`，但 Archetype registry 槽位、组件/关系轻量元数据、空 Table 壳及 QueryState eviction 日志仍保留，压缩或复用稳定 ID 尚未实现。候选查找成本随被访问列 posting 总量和来源数增长；非后缀 posting 清理当前仍为 O(posting 长度)，共享机器诊断基准见本节上一条。dirty-query 候选发现优化仍不能据此宣称端到端严格 `O(change)`；interval-index 读取、日志缺口重建和输出成本仍存在。测试覆盖重复写入 4096 次只输出一个 chunk、48 来源归并及候选工作计数。change observer 已有按组件/实体/tick 的写入索引，覆盖 `Mut<T>`、spawn 和 insert，并以实体 tick 复核作为安全 fallback；flush 先原子取得本窗口写过的组件摘要，再通过按组件维护的活动 observer 索引选择候选，空闲帧和无关组件写入均不扫描全部 observer 或历史空槽。panic 后未完成批次另有有序 pending 索引，注册/删除仍保持确定投递序；snapshot restore 会清理旧 pending、推进 index epoch 并重建 cursor 基线。仍须补高基数端到端量具、并发写屏障/回绕及完整结构变更矩阵，不能据此宣称整个 Query→Observer→Reaction 链路严格 `O(change)`。transform 不再每帧枚举/排序全局关系边：`prism_transform::ecs` 使用按 `ChildOf` 类型的有界关系日志和缓存拓扑，静态帧的零全局边扫描/排序有 `tests/ecs_relation_cache.rs` 回归断言；`prism_transform_ecs::TransformSpatialPlugin` 已固定 propagation→observer flush→BVH/grid sync 顺序，并覆盖 spawn/move/despawn/restore。历史缺口/restore 仍需重建基线，chunk 摘要、关系源与下游空间消费者的总成本也尚未证明严格随变化量增长。M6 的音频 native ECS/facade、source-route/scene adapter/full-chain 有实现和局部测试（依据 `ea2ed8026`、`cd03b14c7`），但完整应用/旧资产/editor 联合验收仍未完成；bounded memory/file `EventLogBackend` 与 replay bridge 已实现（`3e503dd38`），且 `ReplicationServerPlugin::with_event_log` / `ServerReplication::new_durable` 已接入 authority journal；插件集成测试覆盖文件 journal 重启恢复、ambiguous I/O 重试和多个 append/sync/publication/checkpoint 边界的子进程 SIGKILL。仍缺真实断电/设备故障与目标文件系统、平台的耐久性证明，跨机 HA/复制和生产恢复运维流程也未验收。NUMA acceptance gate 已实现，但 Linux strict success、ECS arena local-page proof、拓扑感知调度和 NUMA 性能仍未验证。`prism_world_residency` 当前是 CPU deterministic SDF clipmap planner（`7e99fa30f`、`90ff7a89a`），不是 GPU SDF sampling 或 ECS GPU 集成。当前 checkout 的受控多核扩展、端到端 GPU 常驻/同步收益、跨平台浮点确定性及 P50/P95/P99 证据仍待补齐；历史量具输出不代替这些验收。
- **QueryState tombstone 压缩的性能边界**：已将 registry-wide revision 改为 append-only eviction ID 日志及每个 QueryState 的消费游标；refresh 自适应比较“待处理 eviction 数 × 匹配 ID 二分探测成本”与缓存长度，选择事件探测或扫描缓存 tombstone 标记；只有命中该 query 时才执行整表 COW 压缩。事件日志与为稳定 ID 保留的 archetype tombstone 同阶增长；命中时仍须压缩该 QueryState 全部匹配 ID。回归覆盖：64 轮无关 eviction 积压保持旧 Arc 共享；16 个匹配 archetype 中单个淘汰走事件探测并压缩；16 个匹配 archetype、4 个待处理淘汰（含匹配项在首/末）走缓存扫描并压缩，旧 Arc 内容不变。日志 cursor 以 usize 索引，避免有限宽 revision 回绕；结构 generation 替换时重置 cursor。当前默认/all-features 整包测试与 no-default 检查均通过；仍需在多 warmed query + 高 churn 下量测事件探测、命中压缩及日志保留成本，稳定 ID registry 槽与轻量 tombstone 元数据仍未回收。
- **音频设备状态修订（`94d763d70`）**：上一段关于真实 device callback 尚未完成的描述已过时；CPAL 播放 callback 的启停、断线/流错误处理、有限恢复和 runtime generation 重建已有实现及 fake backend 生命周期测试。仍缺可用物理设备 strict 验收（本机无默认输出端点）、输入设备 lifecycle/枚举/默认路由通知，以及完整应用/旧资产/editor 联合验收。
- **column journal 与 dirty-query 性能状态**：按组件 revision / structural-event 索引、`journal_records_scanned` 及 observer/scheduler 规模回归已在 `cdbca0b03` 提交；column revision k 路堆归并和小集/高基数候选去重均有回归。当前默认 `cargo test -p prism_ecs` 整包库 886 项、all-features 整包库 1236 项通过，两种配置的集成测试和 doctests 均通过；`cargo check -p prism_ecs --no-default-features` 通过。scale 默认 3 项通过/3 项忽略、all-features 3 项通过/8 项忽略。此前回归覆盖 1,024 distinct chunks 和 8,192 条 journal 保留上限溢出后 fallback。百万行 dirty-chunk 对照用例在 release 下额外运行 4 次，当前机器的单样本区间见 §17.1。2026-10-20 新增 portable complexity gate 场景：4,096 个真实 chunk、131,072 行中对 1,024 个不同 chunk 各写 4 次；Release 定向测试通过，记录 journal 选择数、候选标记数、候选原型检查数、去重后候选数、层级摘要读取数和唯一输出 chunk 数，实测分别为 4,096 / 4,096 / 1 / 1,024 / 0 / 1,024。2026-10-21 又新增 64 个无关 archetype + 1 个目标 archetype 的前向窗口 gate，确认一次目标写入只读取 1 条 journal 记录、检查 1 个候选 archetype、测试 0 个 chunk summary；定向测试 1/1 通过。此 gate 证明候选定位不随无关 archetype 数增长，不是 wall-clock 性能保证。该 portable baseline 只门禁这些确定性工作计数，墙钟耗时仍为诊断值，不是受控硬件性能验收。受控基准、性能阈值和更多变化分布仍待完成。`candidate_chunk_marks` 覆盖写入候选处理量。候选去重期望 O(K)，唯一候选排序 O(U log U)，不是严格最坏情况界；journal 写入侧索引也有维护成本，仍须用受控基准比较收益。
- **已收口的增量查询切片**：`cb3a45e7b` 让正常前向窗口优先消费保留的 column journal；后续按 archetype 聚合候选、增加高来源数堆归并。本轮又修复保守 fallback 覆盖当前 tick 后未推进 journal cursor、导致已覆盖写入在下一前向窗口重报的问题；回归见 `tests/scale.rs::perf_contract_dirty_output_tracks_touched_chunks_not_world_rows`。首次请求、回退/重放、日志缺口和历史窗口仍保守走既有索引路径。当前默认整包库 886 项、all-features 整包库 1236 项通过，两种配置的集成测试与 doctests 均通过，no-default-features 检查通过。以上证明限定回归正确，不证明 query 端到端严格 `O(change)`、全部并发写屏障或所有 feature 组合已经完成。
- **并行安全收口**：chunk 元数据并发访问、写屏障与版本回绕仍需集中验证。真实并行 query→Table→ChunkChange→ColumnJournal 回归覆盖 4 worker/512 行；高基数回归覆盖 1,024 个 distinct chunks；`journal_overflow_falls_back_without_missing_chunks_or_replaying_them` 覆盖超过 8,192 条事件后的 indexed fallback 和下一窗口 cursor。三者均纳入当前 all-features 整包。独立原子 chunk summary 与 ColumnJournal 并发测试也仍在。以上不证明所有线程交错、结构变更竞态、版本回绕组合或并发压力矩阵。系统 panic 隔离亦不回滚已发生的普通写入/用户外部副作用，不能泛化为任意操作的事务回滚。
- **chunk cache 计数器耗尽**：`ChunkChange` 写入 revision 与 `ChunkMeta` structural epoch 现饱和于 `u64::MAX`，不回绕成可能复用的旧值；bounds 缓存把饱和值视为始终过期，避免计数耗尽后错误复用。定向回归覆盖饱和边界；这不替代 `Tick(u32)` 的 wraparound 矩阵，也不证明所有并发交错正确。
- **`Tick(u32)` dirty-query / change-index 回绕对拍**：`tests/changed_chunk_filter.rs::dirty_query_matches_row_oracle_for_ticks_straddling_u32_wrap` 将真实 Table 行的 changed tick 放在 `MAX-1`、`MAX`、`0`、`1` 与 `MAX_CHANGE_AGE`，并比较 8×8 边界窗口下的 `Changed<T>` 查询结果和逐行 tick oracle；该集成测试于 2026-10-21 重跑 6 项全通过。`query::dirty_tests::sparse_dirty_fallback_matches_tick_oracle_across_wrap` 对拍 SparseSet fallback 与 row tick oracle，明确覆盖 live-set fallback 而非 indexed query 路径；`change_index::revision_tests::collect_delivers_indexed_batches_across_tick_wrap` 覆盖 `MAX-1 → MAX → 0 → 1` 连续窗口的 indexed batch 收集。2026-10-21 默认配置 `cargo test -p prism_ecs --lib` 883 项全通过，包含这两项回归和 64 个无关 archetype 的候选定位 gate。仍不代表完整 `World` tick 周期、SparseSet indexed query 全部回绕交错或并发结构变更矩阵验收。
- **高风险项**：
  1. **chunk 变更版本 + 脏块访问器（M2）**：触存储/原型/变更检测三核心，最复杂，须充分基准 + 等价测试，严禁一次性大改。
  2. **fiber 作业图（M3）**：并发正确性要害（数据竞争/借用），须 `UnsafeWorldCell` 访问证明 + 压测；可先 async-task 近似再上 fiber。
  3. **确定性/回滚（M5）**：浮点跨平台一致难，定点路径或严格 flag + 逐帧哈希；回滚内存靠 chunk 增量快照控制。
  4. **大世界流送（M5）**：卸载/重载实体引用悬垂，靠稳定 ID + 弱句柄解析。
  5. **关系 fragmenting vs non-fragmenting**：高基数关系勿 fragment（原型爆炸），走旁路索引。
  6. **冲突图执行器（M1）**：正确性要害，改动最保守。
- **历史修复记录（`JobDag` 双派发竞态，非本次重跑）**：此前全量测试偶发 `test result: FAILED`，后定位到 `system::job_dag_tests::edge_provides_happens_before_for_values` 与 `sub_job_panic_propagates` 的 `JobDag: node dispatched more than once`。根因是 driver 用实时原子 pending 判定根时，已完成的根会把 join 节点递减到 0，导致该 join 被 driver 和前驱重复派发。源码修复为按**初始入度**固定根集合，其余节点仅由最后完成的前驱派发。原记录保留：`join_under_racing_roots_dispatches_each_node_once` 修复前 3/3 复现，修复后 40 次模块 + 20 次 `multi_thread` 全量 + 20 次 `all-features` 全量零失败。该历史记录不证明当前共享工作树全部改动已验证，本次未执行上述命令。
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

**ECS 侧已有提取契约 + 运行器**：`pkg/prism_ecs/src/extract.rs` 提供 `ExtractResource`（`extract(&self, &World) -> Option<Out>`，`None` 保留次世界旧值）与 `ExtractComponent`（`QueryData: ReadOnlyQueryData`、filter 与单行转换），以及 `extract_resource::<R>` / `extract_components::<C>`。这两个运行器只以 `&World` 读取主世界，类型上限制查询只读；匹配仍遵循普通查询的 filter 与 Sparse 在场规则。次世界实体用自己的分配器生成，返回主/次实体映射，不假设两边 id 一致。**App 通用回调是另一层契约**：`set_extract` / `add_extract` 接受 `FnMut(&mut World, &mut World)`，允许主线程 bookkeeping；主世界权威只读在这里不是 Rust 签名强制，不能把 typed runner 的只读证明泛化给任意回调。

**App 与渲染侧也已有实际接线**：`pkg/prism_app/src/pipelined.rs` 在 `pipelined` feature 下提供可选执行器，`App::enable_pipelined_rendering` 显式开启；默认仍串行。次世界 N 帧执行与主世界 N+1 帧更新重叠，然后 join 前帧次世界、在主线程执行 extract 屏障，再派发新次帧；不存在渲染线程同时读主世界。可用常驻专用线程，或 `pipelined-tasks` 的 `prism_tasks` 池；`sync_sub_apps` 收回工作线程持有的次世界，worker panic 交回主线程传播。`pkg/prism_render_app/src/plugin.rs` 已追加 frame record、published transforms、GPU columns 和 render assets 的提取链。**边界**：这是所有权移交 + 屏障的流水线，不是任意 World 的通用双缓冲/自动增量复制；安装渲染插件不等于开启流水线，已有接线也不证明所有渲染功能、GPU 后端或延迟/吞吐目标已验收。

### 23.6 作用域命令队列与层级 despawn

- **作用域命令**：`commands.entity(e).with_children(|b| ...)` 式分层构建，子作业命令归属父作用域，批量应用时按作用域确定序回放（配合 §9 并行 ECB）。
- **层级 despawn**：`despawn_recursive` 沿 `ChildOf`（排他 + `OnDelete=Delete`，见 23.2/23.3）一致级联，保证不漏不悬垂；与快照/回滚（§16.5）协同保持结构一致。

**已落地（`hierarchy.rs` + `command/`）**：引擎现在有**规范的层级关系** `ChildOf`（`Relation::KIND`：排他 = 一个父节点；`on_delete_target = Delete` = 父死子亡，走 §23.2 级联）。两项操作建立在其上——`EntityCommands::with_children(|children| { children.spawn(bundle); ... })` 是**作用域命令**（子实体在同一个嵌套 builder 里"生成 + 挂到父节点"，队列按书写顺序回放、且保证 spawn 先于关系添加，因此确定序可复现；`ChildBuilder::parent()` 可读出作用域父节点），`EntityCommands::add_child(existing)` 用于把已有实体挂进来（`ChildOf` 排他，重复挂载会**搬家**而不是双亲）。`World::despawn_recursive(entity)` 深度优先收集子树后**由深到浅**删除（先删子再删父，hook/observer 沿途总能看到父节点仍存在），返回**子树规模**——不是"`despawn` 返回 true 的次数"，因为 Delete 级联可能在中途先删掉某个成员，计数不能依赖这种交错；遍历带 visited 集合，畸形环也能终止；对已死实体的调用返回 `0`。这些入口按需自动注册 `ChildOf`（`World::register_child_of()` 幂等），因此不用层级的世界不付注册成本。

### 23.7 处理器 LOD（MassEntity 形态）

把"海量轻实体"的更新组织为**处理器（processor）**：按距离/重要度分档，不同档位以不同频率、不同精度批量跑（如远处 AI 10Hz 粗更新、近处 60Hz 精更新）。处理器消费 chunk 批次（§6）、受 §13.2 LOD 调度，使 CPU 成本随"活跃近处实体数"而非"总实体数"增长。

### 23.8 诚实边界

本章 23.1/23.2/23.3（实体禁用 / 关系删除策略 / 排他关系，见 `relation.rs`）、23.4（歧义检测与单步，见 `schedule/{ambiguity,stepping}.rs`）、23.6（作用域命令 / 层级 despawn，见 `command/`）、23.7（处理器 LOD，见 `partition/processor.rs`）都有实现与回归源码；**23.5 也已有 ECS runner、App 可选流水线和原生渲染提取接线，不再整体标为 PLANNED**。仍须按功能/feature 组合验证失败清理、同步与真实后端收益；本次未重跑这些测试，不能据源码存在宣布整章完成。

---

## 24. 跨 crate 契约对齐（v0.4）

`prism_tasks`/`prism_time`/`prism_reflect`/`prism_transform` 升级到 v0.2 后新增了若干跨 crate 能力。本章把它们**回接进 ECS**，明确「哪部分真相归谁」，避免两处各实现一套、契约漂移。本章不引入 ECS 新机制，只做对齐与引用。

### 24.1 调度执行底座对齐（tasks QoS 车道 + time 帧预算）

ECS 冲突图执行器（§8.2）与 fiber 作业图（§8.3）**不自建线程池/优先级**，而是派发到 `prism_tasks`：

- system 与 system 内子作业按关键度映射到 tasks 的**优先级车道**（见 tasks §24.1）：关键路径（物理/提取/传播）走 `Critical`，常规 gameplay 走 `Normal`，异步预计算走 `Background`。
- 执行受 `prism_time` **帧预算**（time §24.4）约束：预算耗尽时 `Background` system 顺延下帧，关键 system 必达。**已落地（ECS 侧）**：`Schedule::run_budgeted(world, budget)` 与 `Schedule::run_parallel_budgeted(pool, world, budget)` 在单线程与并行执行器上实现该门控——`Critical`/`Normal` 无论预算始终执行，预算耗尽后所有 `Background` 系统**本帧跳过并计数**（下帧预算允许时再跑）；`SystemLane` → `prism_tasks::Priority` 的映射与 `PriorityGroup` 派发也已接入（`schedule/{lane,executor,parallel_executor,graph}.rs`，带单线程/并行两条预算测试）。诚实边界：预算是墙钟门控，故只用于可延后的背景档；`determinism` 档仍走不带预算的固定执行序。
- **单一真相**：并行与调度策略实现在 tasks；时钟/预算在 time；ECS 只声明「访问集 + 顺序 + 车道标注」。确定性并行归并序由 tasks §24.7 保证（见 §24.4）。

### 24.2 Observer 作为跨系统空间 / 派生更新总线

ECS §12 Observer 是其他 crate 派生更新的统一入口，减少消费者自行全量重建。change tracker 现已在每次索引写入时维护去重的脏组件摘要，flush 原子取得该摘要，并通过 `active_by_component` 与独立的 panic-retry pending 索引只访问本窗口候选 observer；因此空闲帧、无关组件写入和历史移除槽不会触发全 observer 扫描。候选 observer 内部仍须读取对应组件日志并以实体 tick 复核，restore、日志缺口与下游消费者也各有恢复成本，所以这不是整个反应链路严格 `O(change)` 的证明：

- `pkg/prism_transform/src/ecs/` 已通过生命周期 observer、`observe_changed::<GlobalTransform/LocalBounds>` 维护派生状态及 `drain_spatial_updates` 的 Insert/Update/Remove 流，保留完整代际 Entity；由 host 安装并按 writer→传播→flush→消费的顺序运行。
- ECS 变更总线按 flush 窗口合并实体并确定序投递，回调写入留给后续 flush；不自动撤销消费者已发生的外部副作用。音频重定位/阴影失效是该契约的应用目标，不因 transform 接线存在就视为全部消费者已实现。

**已有真实空间消费者**：`pkg/prism_transform_ecs/src/lib.rs` 的 `sync_spatial_indexes` 将一次 drain 分发给已安装的 BVH（physics `DynamicBvh`）与 grid（replication `AoiGrid`）后端；grid 是 AABB 中心点的 XZ 兴趣查询，不是体积重叠查询。非零 spatial margin 代表最后提交的 fat bound，不保证每个中间姿态都更新索引。restore 通过世界身份/结构 generation 失效并协调重建，跨 world 清旧代理；初次接入/restore 可能全量同步，普通帧消费待更新集。传播使用按关系类型的有界变更日志与缓存父子拓扑；静态帧不枚举或排序全局关系边（`tests/ecs_relation_cache.rs` 有工作量断言）。变化发现仍需检查层级 chunk 摘要、关系日志和受影响子树；日志缺口/restore 会重建基线。因此已有增量机制不等于全链路严格 O(变化量)，也不等于当前运行硬件上的性能验收。

**原生 UI 同样已有消费路径**：`pkg/prism_ui_ecs/src/` 已使用 `prism_ecs`；ECS→signal 通过变更/生命周期观察增量拉取，初始绑定和 restore 刷新基线；signal→ECS 通过 `SignalChangeQueue` 收集通知，仅读取被通知的 two-way binding，干净帧不遍历 binding、不读取或克隆 signal 值。push 过程中新增的通知留待下一轮；panic 时未确认的当前项及余下批次可重试，已完成项不回滚。`schedule.rs` 以 non-send bridge + 独占 pull/push 系统接入 `LoomSyncSet` 顺序，并有 panic 时桥所有权恢复逻辑。inspector/timetravel 的可选 `ecs` feature 不等于整个编辑器/旧工程工作流已完成迁移。

### 24.3 反射驱动的动态组件 / 快照 / 网络增量

ECS §16 与 `prism_reflect` 的契约须区分类型化反射桥、raw 动态 ABI 和原生 snapshot，不能统一描述成所有活值都由反射托管：

- 反射桥须显式 `register_reflect_component::<T>`，用 `TypeInfo` / `StableTypeId` 与单态化访问器；不是「所有组件零注册」。§16.2 raw 动态布局/热迁移则由调用方承担 unsafe ABI 与 glue 生命周期，不依赖反射自动证明布局安全。
- 原生 `WorldSnapshot` 使用注册的 clone/hash/drop glue 与存储/schema 契约；反射存档另走二进制解码、`FromReflect` 与 schema migration。不能一概宣称二者反射零拷贝。
- 字段 diff/复制许可使用 `prism_reflect`；`prism_replication` 的 wire schema、transport 和 ECS 转换由应用策略明确提供。`StableTypeId` 便于跨会话标识，但仍须双方 schema 兼容，不自动获得网络互通。

**ECS 侧已落地**：`reflect` feature 下的 `src/reflect_bridge.rs` 把「按类型名动态读写组件」变成真实可用的 API——`register_reflect_component::<T>()` 登记 `StableTypeId → {get, get_mut, apply, remove}` 单态化函数指针表，随后 `get_reflect` / `get_reflect_mut` / `insert_reflect` / `remove_reflect` 全部只认 `&dyn Reflect` 与 `StableTypeId`，调用点不再出现 Rust 类型名（编辑器检视器、场景装载、存档迁移、复制层正是这种用法）。`insert_reflect` 同时接受**具体值**（`downcast` 快路径）与 **`Dynamic*` 树**（存档解码/`reflect_clone` 的产物，经 `FromReflect` 还原）；`StableTypeId` 由类型路径派生，因此可安全写进存档/网络流；`reflect_component_id` / `reflect_stable_id` 提供「世界内 dense id ↔ 稳定 id」双向映射。**字段级增量也已接上第一段**：`diff_reflect(entity, id, authoritative)` 直接产出 `prism_reflect::Patch`（只含变化字段），`apply_reflect_patch(entity, id, &patch)` 把补丁写回活组件；`Patch::Unchanged` 是**真 no-op**（既不写也不标 changed），所以已经同步的接收端不会因「空补丁」反复触发 `Changed<T>`/脏块。**二进制存档也已接上**（`serialize` feature，见本节末与 §24.3）：实体经 `save_reflect_archive` / `load_reflect_archive` 往返回读，值走 `prism_reflect` 的二进制格式（含其反序列化限额），跨版本读取由 `SchemaRegistry` + `Migration` 链承担。

**复制策略一层**：`replicated_diff_reflect(entity, id, authoritative, &ReplicationPlan)` 在 diff 之上按字段许可裁剪，再由 `apply_reflect_patch` 本地落地；这是补丁生成/应用契约，**不自动包含网络编码、可靠通道、权限、实体映射或所有组件复制注册**。`pkg/prism_replication/src/plugin/ecs.rs` 的 `EcsReplicationPlugin` 已接入主 `App`：`First` pump 网络、固定循环前 reconcile、每个 `FixedFirst` tick 通过 `App::update()` 提交 input 前 snapshot/history、预测并发送，再应用 ECS 输入；权威帧回到同一 `App::update()` 的固定循环前完成 rollback/replay。提交顺序是事务性的：snapshot/history、prediction、send 任一步拒绝都会恢复本次分支并保留待处理输入。具体 wire 记录如何映射 ECS 由 `EcsReplicationStrategy` 负责，不硬编码业务组件。

**存档已有 V3 prefab 语义（`serialize` feature）**：`src/reflect_archive.rs` 的 `save/load_reflect_archive` 使用 `PECA` 信封，组件记录包含具体类型名、`StableTypeId` 与二进制值；V3 还保存请求根及其闭合祖先图、父优先序、组件移除/字段覆盖策略与 schema-layout 校验信息，保留空祖先与纯实例。重复根去重，死亡实体、悬空祖先、环和非法路径返回错误。V2 组件存档仍可读，V1 拒绝。默认复用加载前同 id 存活实体，否则生成新实体；选项可强制 fresh，完整映射用于重映射 `IsA`，应用组件里的实体引用须显式 callback，不能猜整数就是 Entity。V3 替换落地实体的**反射组件集合、出向 IsA 和覆盖策略**，保留非反射组件；V2 保持旧 merge 语义。解析、迁移、图/schema 和 `FromReflect` 预检先于目标写入，但任意用户 hook/panic 不在全局事务保证内。

**Prefab 字段继承与存档边界**：`prefab_reflect.rs` 的显式 resolver 按祖先优先序合成值，普通 Query 不自动合成；无 base 的策略可 dormant。组件值可沿 schema migration 链升级，但字段所有权策略必须匹配 canonical schema layout，不能由组件迁移自动推断新字段路径/所有权。对应源码回归入口为 `tests/prefab_{inheritance,reflect,archive}.rs`；本次未运行。

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

**已有可执行对拍夹具，非本次验证结果**：`pkg/prism_ecs/tests/determinism_link.rs`（需 `--features multi_thread`）把 `TickClock` 的 1/60 有理步长/整数 tick、并行 `Schedule` 快照哈希、tasks 非交换确定归并、transform 整数定点层级传播串联；测试源码对比 1 与 8 worker，`determinism` feature 下用 `FrameHashLog::first_divergence` 定位。即使该夹具通过也只覆盖这个场景，不能证明任意浮点业务、SIMD/平台组合都位等价；本次未重跑。

**本次端到端切片验证**：`prism_replication` 的 ECS 测试使用 secure in-process transport、固定 `10ms` `ManualDelta`/fixed timestep，并只调用 `client_app.update()` / `server_app.update()`；没有手动补跑 `FixedFirst`。`ecs_e2e_is_structurally_equal_for_one_and_eight_workers` 在 `determinism,ecs,multi_thread` 下比较 1 与 8 worker 的最终 `WorldSnapshot`、每个 retained history snapshot 的 `structurally_eq`，hash 只作辅助对照。`failed_ecs_commit_leaves_session_history_and_input_untouched` 验证未注册 snapshot component 时 World/history cursor、prediction、input timeline 和待处理输入均保持可重试状态；已有 replay-failure 测试继续覆盖权威回滚失败清理。该证据只覆盖单进程/in-process、注册了 clone/equality/hash glue 的测试 World；不等价于真实网络、跨平台硬件/浮点位级证明，也不覆盖未注册 snapshot glue 的业务 World。

`pkg/prism_replication/tests/udp_acceptance.rs` 另有独立真实 loopback UDP 验收：fault proxy 丢弃首个 data 与反向 ACK，验证 ARQ 重传、顺序和 exactly-once；另一个测试关闭旧 socket，经显式 ticket 提交 peer rebind，验证候选端点在提交前被拒、稳定 peer id 保持、重连后双向收发恢复（`39bd3806a`、`0bf59f55a`、`c7201323e`、`d6000586b`）。`1290d101b` 又增加独立 server/client 子进程的 localhost 验收，覆盖稳定 peer、握手、业务数据、确定性重传、双向 ACK 清空、超时和子进程回收；它证明单机跨进程，不是跨机、公网、NAT/ICE、relay、真实网络故障或生产迁移证据。

**网络审计也已有实际链路**：`pkg/prism_replication/src/plugin/audit.rs` 的 `FrameHashAudit` 审计有限保留窗口内的 **input 前 ECS `WorldSnapshot::state_hash()`**，不是 wire 记录哈希；`plugin/ecs.rs` 从 `SnapshotHistory` 同步，回滚修正后重新发布，client/server 通过可靠事件通道发送/接收帧哈希。单调序号区分修正与旧包，pending 合并并在 backpressure 后重试；重复冲突、未来/过期窗口与未送达过期均有显式结果。比较只给出**保留且已知共同 tick 内最早发散**，缺包/过期不代表相同，也不能证明没有更早发散。组件值仅覆盖已注册的语义 hash glue，clone-only 不提供语义哈希、资源须显式注册；双方 schema、实体 id、ticks 与存储布局须一致，哈希相等仍是概率性证据。回归入口 `plugin/audit/tests.rs` 与 `plugin/ecs/tests/audit_tests.rs` 已有源码，未在本次运行。

### 24.5 诚实边界

本章职责对齐已有真实跨 crate 消费，不能再概括成「仅 ECS 侧实现」：§24.1 tasks lane/预算，§24.2 observer 候选索引、transform 空间消费者与原生 UI 桥，§24.3 动态 ABI/反射桥/字段策略/V3 prefab 存档，§24.4 原生 App 网络回滚与有限窗口审计均见上述入口；§23.5 App/渲染流水线也已有源码。observer 候选索引已完成默认/all-features 全量库测试，transform App 插件已完成四组 feature 测试；其他机制中仍有仅存在源码或历史验收入口、未在当前平台重跑的部分，不能据此宣称所有 feature/平台/失败路径均已完成。

完整目标继续保留：脏块的正常写入源发现已推进到 tick→chunk 索引，change observer 也已推进到脏组件→候选 observer 索引，但 query 端到端、rebuild/restore/fallback、候选内部日志复核与 UI 初始/restore 基线刷新尚非统一的严格变化量成本；关系日志缺口或 restore 仍需重建拓扑。音频 native ECS/source-route/scene adapter/full-chain 与 bounded event-log provider 已有实现，但可用物理设备启停/恢复、完整应用/旧资产/editor 工作流、durable replication recovery 及跨进程/跨机持久化仍待验收。NUMA acceptance gate 已有，但真实 Linux strict pass、ECS arena local-page proof、拓扑感知调度和 NUMA 性能仍未证明。GPU resident-column/physical-adapter gate、virtual-texture benchmark harness 与 CPU SDF planner 已有入口，但真实 GPU 通过、同步/收益压测和 GPU SDF sampling/ECS 集成仍未验证。真实 loopback UDP 的丢包、ACK、顺序/确传与 peer rebind 已有独立验收入口，但不覆盖跨机/NAT/ICE/relay/生产迁移。有限窗口/注册范围的哈希相等不能替代整个仿真链路正确性证明；主 App 接线、固定 tick、预测/和解/回滚、失败清理和 1/8 worker structural equality 仅有上述限定证据。本章仍用于固定职责与引用，而非将原范围缩减为现有机制清单。实现仅借鉴公开架构形态与经典数值，不引入 Unreal Engine / Unity 源码。

### 24.6 当前 `main` 的实现/验证/未验证矩阵

音频设备生命周期切片（`94d763d70`）：`prism_audio_device` 提供后端启停/事件契约与 CPAL 输出实现；`prism_audio_ecs` 在控制线程处理断线/流错误、有限重开、格式变化与 runtime generation 替换，callback 不承担恢复工作。fake backend 生命周期与 ECS 接线测试通过，CPAL all-features 编译通过。严格物理设备测试因当前 CoreAudio 无默认输出端点未能通过；输入设备、设备枚举和系统默认路由变更通知尚未实现，应用/旧资产联合音频工作流也未验收。

| 范围 | 已实现 | 已验证或已有验收入口 | 仍未验证/不可宣称 |
|---|---|---|---|
| ECS dirty query / archetype cache | 前向窗口 column journal；按组件 revision / 结构事件过滤；revision 源 k 路堆归并；journal/SparseSet 候选按 archetype 聚合并在线去重，结果稳定排序；fallback 通过 table-component→archetype 有序 posting 与 SparseSet 当前归属构造候选，再按来源线性/堆多路归并；不遍历完整既有匹配列表；匹配缓存只增量补测新 archetype，并以 COW `Arc<Vec<_>>` 共享稳定快照；fallback 后在当前 tick 建立 journal 基线；shared-value eviction 从 postings 移除永久淘汰 archetype；QueryState 使用 eviction ID 日志游标并自适应选择事件探测/缓存扫描，只对命中缓存的淘汰做压缩 | 当前源码默认/all-features `cargo test -p prism_ecs`：库分别 890/1240 项通过，集成测试和 doctests 均通过；`cargo check -p prism_ecs --no-default-features` 通过；验证使用独立临时 target。回归覆盖 eviction 快照、旧 Arc 保持、Table/chunk/column 与 shared Arc 重对象释放，以及 shared posting 后缀/非后缀淘汰；fallback 单 posting/六列重叠、restore 顺序与 Table tick-wrap 等也有测试 | query 全链路严格 `O(change)`、非后缀淘汰受控性能门槛、完整 `World` tick 周期及 SparseSet indexed query 的全回绕窗口、并发写屏障/结构变化矩阵和忽略的大规模场景；稳定 ArchetypeId 槽、轻量 tombstone 元数据、空 Table 壳和事件日志仍保留，registry 压缩/ID 复用未实现 |
| ECS change observer | 写入侧维护去重脏组件摘要；按组件活动索引选择候选；独立 pending 索引保存 panic 重试批次；按 id 删除不再线性扫描；restore 清理旧 pending 并重建 cursor 基线 | fast-path 13 项；覆盖空闲/无关组件零 slot visit、2048 历史槽 churn、注册删除顺序、回调中增删、panic retry、World replacement、table/Sparse 写入和 generation reuse；最新默认/all-features 整包分别 890/1240 项通过（与 ECS cache 行当前源码验证结果一致） | Query→Observer→Reaction 端到端严格 `O(change)`、并发写屏障/回绕、完整结构变更矩阵与高基数性能 |
| UDP / ACK / rebind | 有限可靠 UDP、ACK 重传、候选端点与稳定 peer rebind | 真实 loopback fault-proxy 测试覆盖 data/ACK 丢包、顺序/exactly-once 与旧 socket 关闭后的 rebind；独立 server/client 子进程 localhost 验收覆盖握手、业务数据、确定性重传、双向 ACK 清空与稳定 peer（`1290d101b`，本轮定向复跑 1/1 通过） | 跨机、公网、NAT/ICE/relay、真实网络故障、HA 与生产迁移 |
| NUMA / affinity | OS topology/affinity readback、worker outcome、显式 arena placement/fallback 语义；策略层拓扑模型 | `numa_acceptance` 入口；arena 默认测试 5 项、`vm` 测试 12 项、`--no-default-features` check 已通过；默认 acceptance 缺能力时 `SKIPPED`，`PRISM_REQUIRE_NUMA=1` 严格失败 | 当前机器 Linux strict pass、ECS node-local pages 的真实物理证明、拓扑感知调度收益与 NUMA 性能 |
| audio / event-log | audio source-route/scene adapter/full-chain；CPAL callback 启停、断线/流错误处理、有限恢复与 runtime generation 重建；bounded memory/file event-log 与 replay bridge；replication authority journal 插件接线 | 音频默认/no-default 测试共 98 项通过、CPAL all-features 编译检查通过；物理设备 strict gate 因当前 CoreAudio 无默认输出端点而失败；durable plugin/file tests 覆盖 append/sync/publication/checkpoint 故障注入、重启恢复与子进程 SIGKILL 边界（本轮未重跑） | 可用物理设备上的真实启停/恢复；输入设备 lifecycle/枚举/默认路由通知；完整应用/旧资产/editor 联合流程；真实断电/介质/文件系统耐久性与跨平台证明；跨机 HA/复制和生产恢复运维 |
| GPU resident / benchmark / SDF | CPU resident-column extraction；physical-GPU gate；virtual-texture CPU/GPU benchmark harness（`3a8120874`、`0e7e67259`）；CPU deterministic SDF planner | resident-column readback、wgpu hardware-evidence、benchmark 入口均存在，缺设备明确 skip/strict fail | 当前真实 adapter 通过、GPU sync/upload benefit、性能数字、GPU SDF sampling/ECS 集成 |
| Render Fabric / placement parity | shader package/module/permutation/pass identity binding；缓存失效校验；水体整帧 parity 入口；heightfield/planetary placement deletion wrapper 与 WGSL | shader binding 与 placement 代码已提交；水体 parity 8 项、world GPU/runtime 定向测试已有通过记录 | Scene Fabric 真实 payload/prepare/record/submit 闭环、当前机器物理 GPU 通过、跨后端 parity、GPU 上传收益与 P50/P95/P99 仍未证明 |
