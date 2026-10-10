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
Query<(Entity, &A, &B)>
Query<&A, (With<B>, Without<C>)>
Query<&A, Or<(Changed<B>, Added<C>)>>
Query<Relations<ChildOf>>                 // §11
```

- **archetype 匹配缓存**：首次构建匹配列表；archetype 新增时增量更新，杜绝每帧全表扫描。
- **脏块访问器**（核心）：`q.dirty_chunks()` 仅遍历 `chunk.change_version > last_run` 的块，把提取/剔除/宽相从 O(匹配实体) 降到 O(脏块)。
- **SIMD 迭代**：`q.simd_iter::<8>()` 按 chunk 分块向量化（`simd` feature，基于 `core::simd`），标量路径作为 golden 对拍。
- **par_iter**：按 chunk 切分投递到 `prism_tasks`（见 §8），work-stealing 负载均衡。
- **disjoint 查询**：同 system 多个不冲突查询共存，`ParamSet` 处理潜在别名。

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
- 反射桥（`prism_reflect`）：序列化、编辑器 inspector、脚本访问统一走反射，组件无需各自手写 serde。

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
- **内存**：每原型列用大页竞技场分配，NUMA 感知就近分配，原型预热避免首帧抖动，实体/原型空槽回收。
- **GPU 常驻列直传**：脏块增量上传，省 CPU 打包。
- **成本 ∝ 变化量**：脏块访问器 + push 反应图把增量路径设为默认，禁止全表扫描成为常态。

诚实边界：CPU 单测只能证机制正确；百万级实体 / 多核近线性 / NUMA / GPU 常驻等规模指标需真实硬件与大场景压测，均标注 PLANNED。

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
  features = ["std","multi_thread","simd","serialize","reflect","trace","determinism","partition","gpu_resident"]

pkg/prism_ecs_macros/              # derive: Component/Bundle/SystemSet/Resource/Event/SystemParam/Relation 已实现
```

> **feature 现状诚实注记**：上方 feature 清单为**目标形态**。`pkg/prism_ecs/Cargo.toml` 当前**已落地**的 feature 为 `std` / `multi_thread` / `simd` / `partition` / `gpu_resident` / `determinism` / `trace`（均有真实 `cfg` 门控代码与测试，`determinism` 见 `world/snapshot/determinism.rs`，`trace` 见 `diagnostics/trace/`）。`serialize` / `reflect` 仍为 **PLANNED feature 名**（尚未在 Cargo.toml 落地）：待 `prism_reflect` 跨 crate 桥接（§16.5 / §24.3）就绪后接入。`trace` 已落地为「结构化事件流 + Chrome/Perfetto 时间线导出」——作为 §16.6 火焰图 profiler 的对偶（profiler 折叠自时间树，trace 保留有序事件流），共用 `SystemInstrument` 捕获钩子、core `no_std + alloc`、`TraceRecorder` 实时捕获层 `std` 门控，非薄壳。`reflect_bridge.rs` 同为 PLANNED（源码树暂未落地，随反射桥一并补）。

依赖：仅 `prism_math`、`prism_tasks`（std）、`prism_reflect`（可选）。**不碰任何 `bevy_*`。**

---

## 20. 契约、不变量与版本化

- **下游 API 契约优先**：对接层公共 API 是红线；内部实现随便改，公共 API 改动必须同步迁移依赖方，保持全仓绿。
- **版本化新增契约**：`ChunkMeta`、`ChunkDirtyIndex`、`EcsSnapshot`/`EcsDelta`、`OwningGroupId`、`WorldPartitionCell`、`GridCell`/`FloatingOrigin`、`ReactionHandle`、`JobHandle`、`StateHash`、`GpuResidentColumn`。
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

- 本文含设计规格与现状：**M0–M5 内核及第 23 章增补已在 `pkg/prism_ecs/` 落地实现**（代际 Entity / 原型 chunk 列存 / 四态存储 / 查询+脏块访问器 / 调度+冲突图执行器 / fiber 作业图 / 关系+Observer+Hook / push 反应图 / 快照+回滚 / 分区流送 / GPU 常驻列 / 诊断接口均有实现与测试）；**仍为 PLANNED 的是** M6 迁移对接、第 24 章跨 crate 契约对齐（§24.1 已部分接入 `prism_tasks`）、§23.5 渲染提取管线，以及百万级实体 / 多核近线性 / NUMA / GPU 真机常驻 / 跨平台浮点确定性等需硬件压测的指标。
- 规模与多核扩展性、NUMA、GPU 常驻指标依赖真实硬件压测，单测不能替代。
- **高风险项**：
  1. **chunk 变更版本 + 脏块访问器（M2）**：触存储/原型/变更检测三核心，最复杂，须充分基准 + 等价测试，严禁一次性大改。
  2. **fiber 作业图（M3）**：并发正确性要害（数据竞争/借用），须 `UnsafeWorldCell` 访问证明 + 压测；可先 async-task 近似再上 fiber。
  3. **确定性/回滚（M5）**：浮点跨平台一致难，定点路径或严格 flag + 逐帧哈希；回滚内存靠 chunk 增量快照控制。
  4. **大世界流送（M5）**：卸载/重载实体引用悬垂，靠稳定 ID + 弱句柄解析。
  5. **关系 fragmenting vs non-fragmenting**：高基数关系勿 fragment（原型爆炸），走旁路索引。
  6. **冲突图执行器（M1）**：正确性要害，改动最保守。
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

### 23.6 作用域命令队列与层级 despawn

- **作用域命令**：`commands.entity(e).with_children(|b| ...)` 式分层构建，子作业命令归属父作用域，批量应用时按作用域确定序回放（配合 §9 并行 ECB）。
- **层级 despawn**：`despawn_recursive` 沿 `ChildOf`（排他 + `OnDelete=Delete`，见 23.2/23.3）一致级联，保证不漏不悬垂；与快照/回滚（§16.5）协同保持结构一致。

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
- 执行受 `prism_time` **帧预算**（time §24.4）约束：预算耗尽时 `Background` system 顺延下帧，关键 system 必达。
- **单一真相**：并行与调度策略实现在 tasks；时钟/预算在 time；ECS 只声明「访问集 + 顺序 + 车道标注」。确定性并行归并序由 tasks §24.7 保证（见 §24.4）。

### 24.2 Observer 作为跨系统空间 / 派生更新总线

ECS §12 Observer 是其他 crate 派生更新的统一入口，避免轮询：

- `prism_transform` 的 `OnChanged<GlobalTransform>`（transform §24.4）、空间加速结构增量同步（transform §24.5）均经 ECS Observer 触发。
- ECS 保证 Observer **批量合并**（本帧多次变化只回调一次）与**确定序回放**（配合 §14 确定性），使空间索引/音频重定位/阴影失效的触发可预测、可回滚。

### 24.3 反射驱动的动态组件 / 快照 / 网络增量

ECS §16 的动态组件、snapshot、prefab 继承、反射桥**以 `prism_reflect` v0.2 为类型基座**：

- 动态组件布局/访问走 reflect §24.1 静态 `TypeInfo` + §24.2 访问器缓存（零注册、热路径 O(1)）。
- snapshot/存档走 reflect §24.3 二进制零拷贝；跨版本读取走 schema 迁移。
- 字段级网络增量（配合 `prism_replication`）走 reflect §24.4 diff + §24.5 字段级复制 + ECS §10 变更检测；`StableTypeId` 为跨会话/跨机组件标识。

### 24.4 确定性链路四方对齐

ECS 确定性仿真（§14）不是孤立的，必须与三个地基 crate 的确定性档**同时成立**，否则链路任一环破坏即整体失确定：

| 环节 | 契约来源 | 要求 |
|---|---|---|
| 系统/实体序 | ECS §8.4 / §14 | 稳定 system 排序、稳定实体/原型分配 |
| 并行归并 | tasks §24.7 | 确定归并树，结果与线程数/窃取序无关 |
| 时间步长 | time §10 / §24.8 | 有理/定点 `fixed_dt`、整数 tick、确定性审计 |
| 空间传播 | transform §12 | 定点层级传播，位级一致 |

四者共享同一句契约：**同输入 → 位等价输出**。`determinism` 档联调时用 time §24.1 录制回放 + §24.8 审计双跑定位首个发散 tick。

### 24.5 诚实边界

本章为跨 crate 职责对齐说明：**§24.1 调度执行底座已部分接入**（冲突图执行器把并行波次派发到 `prism_tasks::TaskPool`、lane→优先级映射见 `schedule/{graph,config,lane}.rs`）；**§24.2 / §24.3 / §24.4 仍为 PLANNED**（分别随 M4 Observer 层、reflect/M2 资产、M5 确定性/网络落地）。本章不引入 ECS 新机制，仅固定跨 crate 职责边界与引用。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

