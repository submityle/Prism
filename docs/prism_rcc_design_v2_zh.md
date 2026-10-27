# Prism RCC v2：Reactive Causal Core

> Prism 的下一代可视化脚本与可编程运行时统一设计。RCC（Reactive Causal Core）不是另一套脚本语言，而是把图、Rust 节点、WASM Component 和持久化流程统一为一种可增量求值、可确定性提交、可跨版本迁移的因果程序。

- 版本：v2.0
- 状态：正式设计，契约完整，可按里程碑实现
- 适用范围：`pkg/prism_visual_scripting`、`pkg/prism_wasm`、编辑器图作者化、Gameplay 流程、UGC/WASM 沙箱
- 不引入：独立 `prism_script` crate、新脚本语言、未定义的后端语义

## 1. 范式定义

传统脚本是按时间执行的命令序列：运行时维护程序计数器，优化是减少指令，热更新是替换实例。RCC 改变语义单位：

> 作者声明数据依赖与效应约束；运行时在当前输入版本上求解满足约束的执行计划。

RCC 程序由五类实体组成：

1. **Cell**：纯数据方程，`value = f(inputs, state)`，可按需求值和记忆化。
2. **Effect**：带能力、幂等、预算和确定性等级的外部效应。
3. **StateCell**：宿主拥有、可快照、可迁移的跨帧状态。
4. **Trigger**：事件、输入、定时器、资源版本变化等需求根。
5. **Constraint**：`After`、`Exclusive`、`Barrier`、`OncePerEpoch` 等偏序约束。

执行顺序不再是作者强制写出的全序。纯 Cell 没有时间顺序；Effect 只有必要的偏序；运行时用稳定排序将偏序线性化，并把不可比且互不冲突的 Effect 合并到并行组。

### 1.1 不变量

- 纯 Cell 的结果只由输入值、StateCell 版本和 tracked read 组成。
- 每个外部 Effect 都有稳定幂等键，重复恢复不会重复写外部世界。
- 所有跨帧状态都在宿主 shadow state 中，载体线性内存不是唯一真相。
- `Exact` 程序在所有允许执行形态下产生相同输出与 Effect journal。
- 热更新只替换新 epoch；在途调用继续持有旧 epoch，发布过程不打断帧。
- 所有运行时预算都有可解释结果：执行、memo 命中、延迟、降级或拒绝。

## 2. 产品借鉴与取舍

| 来源 | 采用机制 | RCC 的改造 |
|---|---|---|
| Unreal Blueprint | exec/data 心智、Pure/Impure、热点原生化 | exec 线编译为偏序约束；原生化成为可验证 Tier，不产生第二套语义 |
| UEFN Verse | 持久状态、回滚、能力边界 | StateCell + revision journal + capability row |
| Houdini | 依赖图、脏节点传播、按需 cook | 动态 DepLog、需求拉取、early cutoff |
| Niagara | 数据导向 VM、SoA 批处理 | Entity-parametric fragment 自动进入批量 Tier |
| Temporal | 确定性重放、外部效应日志、Saga | Effect journal、幂等键、drain barrier |
| Erlang/OTP | 双版本热加载、状态交接 | stable ID 对齐状态，三阶段 prepare/drain/publish |
| Salsa/rust-analyzer | revision、memo、动态依赖、backdating | Cell 依赖验证与早停 |
| Excel/Solid signals | 细粒度重算、依赖传播、glitch-free | revision ledger + 拓扑提交 |
| wasmtime Component Model | typed ABI、沙箱、预实例化、AOT | `prism_wasm` 作为 T4 执行底座 |
| Unity DOTS/Burst | SoA、chunk、特化 | 仅对实体参数化 fragment 批处理，不重写 Prism ECS |

不复制这些产品的对象模型、编辑器协议或历史缺陷。尤其不采用 Blueprint 二进制资产不可合并、nativization 语义漂移、Visual Scripting 黑盒性能等做法。

## 3. 系统边界

```text
作者层：Graph / Rust Node / 任意 WASM Component / LLM Patch
          │
          ▼
语义层：RCC（Cell、Effect、State、Trigger、Constraint）
          │
          ├── T0 Slot VM（解释/调试）
          ├── T1 Fused Kernel（超指令）
          ├── T2 Specialized Kernel（类型特化 + deopt）
          ├── T3 AOT/JIT（可信热图）
          ├── T4 prism_wasm（UGC/第三方/LLM）
          └── T5 SoA Batch（实体批处理）
          │
          ▼
宿主：prism_reflect / prism_ecs / prism_tasks / prism_diagnostic / prism_asset
```

RCC 进入现有 `prism_visual_scripting`，不建立 `prism_script`。现有 `Graph → ExecutionPlan → SlotProgram` 管线保留：Graph 先生成 RCC，再按目标生成 `SlotProgram` 或其他 LowerPlan。现有 `NodeDefinition`、`Purity`、`EffectSpec`、`EffectClass`、`Vm`、`DurableStore`、`Saga`、`Await` 均作为兼容基础。

`prism_wasm` 只提供 Component 编译、沙箱、Policy、Loader、缓存、HotSwap、Canary、Attestation 和时间旅行基础设施；它不依赖 RCC、Graph 或 Prism ECS。

## 4. RCC 数据模型

```rust
pub struct Rcc {
    pub ir_version: u16,
    pub cells: Arena<Cell>,
    pub effects: Arena<Effect>,
    pub triggers: Vec<Trigger>,
    pub constraints: Vec<Constraint>,
    pub states: Arena<StateCell>,
    pub manifest: Manifest,
}

pub struct Cell {
    pub stable: StableId,
    pub def: DefId,
    pub inputs: SmallVec<[CellRef; 6]>,
    pub slots: SlotRange,
    pub deps: DepPolicy,       // Dynamic（默认）或 Fixed（静态保守上界）
    pub memo: MemoPolicy,      // Pure、Idempotent、Never
    pub det: DetLevel,         // Exact、Seeded、Statistical
}

pub struct Effect {
    pub stable: StableId,
    pub spec: EffectSpec,
    pub caps: CapBits,
    pub idem: IdemKey,
    pub order: OrderKey,
    pub barrier: BarrierKind,
}

pub enum Constraint {
    After { a: EffectId, b: EffectId },
    Exclusive { cap: CapId, effects: SmallVec<[EffectId; 4]> },
    Barrier { effect: EffectId, kind: DrainKind },
    OncePerEpoch { effect: EffectId },
}
```

`StableId = hash(def_contract_digest, author_anchor, canonical_structural_path)`。复制节点生成新的 author anchor；移动、改显示名称、调整布局不改变 StableId。StableId 是 memo、状态、journal、热更新和语义 diff 的同一把钥匙。

### 4.1 动态依赖

静态 `ReadSet` 只能作为能力准入、并行冲突保守上界和 CI 审计依据。正确性依赖运行时 `DepLog`：所有世界、资源、输入、时钟、随机读取必须通过 `ReadCtx`，每次读取记录 `(DepKey, version)`。

```rust
pub struct DepLog {
    pub entries: SmallVec<[(DepKey, u32); 8]>,
}

pub struct MemoEntry {
    pub key: CellKey,
    pub deps: DepList,
    pub value: Value,
    pub verified_at: Revision,
    pub epoch: u32,
}
```

依赖校验使用 revision ledger。任一依赖版本变化，Cell 失效；所有依赖未变化则直接命中。重新计算后若输出与旧值相等，执行 early cutoff：只更新 `verified_at`，不使下游失效。大对象使用 Arc identity 或 digest 比较，避免深拷贝。

### 4.2 Effect row

```rust
pub struct EffectRow {
    pub caps: CapBits,
    pub class: EffectClass,
    pub det: DetLevel,
    pub memo: MemoPolicy,
    pub quality: QualityTier,
    pub budget: BudgetClass,
}
```

能力命名空间包括 `ecs.*`、`gameplay.*`、`anim.*`、`audio.*`、`ui.*`、`nav.*`、`net.*`、`debug.*`。Capability 在模块准入时解析为位图；调用时只做无锁位测试。第三方和 UGC 只能由 `prism_wasm` 沙箱承载。

## 5. 执行语义

### 5.1 帧协议

```text
Trigger.fire
  → demanded roots
  → pull Cell 闭包
  → DepLog / memo / early cutoff
  → 生成 Effect 实例
  → 求解偏序约束
  → 并行执行合法 group
  → 按稳定线性化顺序 commit journal
  → revision + 1
```

Cell 是需求拉取，不被无需求的帧更新唤醒。Effect 是提交边界；外部世界只有在 journal commit 后可观察。

### 5.2 约束求解

1. 合并数据依赖、`After` 和 `Exclusive` 约束生成 DAG。
2. 拓扑环立即生成结构化诊断，定位具体节点和约束边。
3. 使用 `(OrderKey, StableId)` 稳定优先队列生成确定性线性化。
4. 不可比、无共享 Exclusive 能力的 Effect 形成并行组。
5. `Exact` 禁止并行浮点归约；`Seeded` 使用 stable/revision/draw 派生随机流。

```rust
pub struct Linearization {
    pub groups: Vec<ParallelGroup>,
    pub commit_order: Vec<EffectId>,
}
```

### 5.3 Effect journal

每次 Effect 的 journal 记录：`epoch`、`stable`、`idem_key`、输入摘要、能力、资源用量、输出摘要、错误、重试和提交序号。恢复时先查 `idem_key`；已提交返回旧结果，未提交按 `EffectClass` 与 `IdempotencyPolicy` 决定重试或补偿。

现有 `EffectSpec` 的 `ExternalWrite` 必须为幂等；`Saga` 处理跨不可逆屏障的补偿；`DurableStore` 作为 journal 的持久化实现。

## 6. 编译与执行形态

### 6.1 Lowering 接口

```rust
pub trait Lowering {
    fn id(&self) -> LoweringId;
    fn admissible(&self, fragment: &Fragment, target: &TargetCtx) -> bool;
    fn estimate(&self, fragment: &Fragment, target: &TargetCtx) -> Cost;
    fn lower(&self, fragment: &Fragment) -> Result<LowerPlan, LowerError>;
}
```

Fragment 在 Effect barrier、信任边界、SoA 参数化边界处分割。代价模型：

```text
cost = setup
     + calls_per_frame × dispatch
     + elements × per_element
     + boundary
     + memory_pressure
```

### 6.2 Tier 契约

| Tier | 执行形式 | 目标场景 | 约束 |
|---|---|---|---|
| T0 | 现有 Slot VM | 调试、冷启动、无 JIT 平台 | 语义基准 |
| T1 | Fused 超指令 | 纯链、扇出 1 | 无外部 Effect |
| T2 | 类型特化 kernel | 高频标量 Cell | guard 失败 deopt |
| T3 | Cranelift AOT/JIT | 签名第一方热点 | 平台允许 JIT |
| T4 | `prism_wasm` Component | UGC、第三方、LLM | 沙箱、Policy、Fuel、Deadline |
| T5 | SoA batch | Entity-parametric fragment | 固定 lane、可确定归约 |

T4 直接复用 `prism_wasm::SandboxEngine`、Component digest 缓存、`InstancePre`、Policy、HotSwap、Canary 和 typed call。RCC 只负责调用语义、Effect journal、shadow state 和因果 trace。

### 6.3 特化和 deopt

只对短小纯 Cell 生成类型特化。每个 Cell 最多保留两个观测类型快路径；guard 失败转泛型路径，失败计数超过阈值后永久去特化。特化计划只改变计算形态，不改变 Effect journal 和错误语义。

## 7. 性能设计

### 7.1 内存布局

- Cell 热元数据采用 SoA；StableId、DefId、slot offset、memo index 分列存放。
- Value 使用 16 字节 tagged representation；标量内联，字符串/列表/JSON 使用 Arc。
- DepLog 使用线程局部 arena，默认 8 个依赖 inline，溢出才分配。
- Memo 表分 64 个 shard，开放寻址，key 为 128-bit 指纹；默认 4 MiB，按 epoch + LRU 回收。
- 热循环禁止全局锁与临时堆分配；帧 arena 在提交后重置。

### 7.2 并行

- Pure Cell 在 worker lane 执行；世界写 Effect 主线程提交。
- 并行组内部可 work-stealing；commit 仍按稳定序列化。
- Entity-parametric fragment 按 32/64 实体批处理，掩码表示缺失实体。
- 单次 WASM 调用不得用于逐实体热循环；使用批量 Component 入口，或在 T2/T5 处理。

### 7.3 性能验收目标

| 指标 | 目标 |
|---|---:|
| memo 校验（8 个依赖以内） | ≤ 15 ns/Cell |
| T1 dispatch | ≤ 3 ns/Cell |
| T2 标量快路径 | ≤ 2 ns/Cell |
| 10,000 Cell、1% 脏闭包 | ≤ 10 µs/帧 |
| WASM POD typed call | ≤ 1.2 µs/调用 |
| 1,000 Effect 约束求解 | ≤ 80 µs |
| 1,000 Cell 热更发布 P99 | ≤ 2 ms |
| 编辑到预览 P95 | ≤ 80 ms |
| Memo GC 单次暂停 | ≤ 0.5 ms |

所有目标以 `prism_diagnostic` 微基准、固定 golden RCC 语料和差分执行器验证。指标是验收契约，不是未经测量的现状声明。

## 8. 效果与质量

### 8.1 表现力

RCC 能统一驱动 Gameplay、动画、音频、UI、导航、网络和调试能力。每种能力都是 reflect 注册的 Effect，不暴露裸 `World`、裸指针或未审计的 WASM import。

### 8.2 表现质量

每个视觉/音频 Effect 声明 `QualityTier` 与 fallback fragment。平台能力不足、帧预算紧张或网络降级时，切换到预声明 fallback，不静默丢弃整段逻辑。`Exact` 逻辑不允许质量降级；`Statistical` 逻辑允许自适应降级。

### 8.3 可解释调试

```rust
pub enum ExplainReason {
    MemoHit { verified_at: Revision },
    DependencyChanged { key: DepKey },
    NotDemanded,
    EffectBarrier,
    BudgetDeferred,
    QualityFallback,
}

pub struct ExplainReport {
    pub ran: bool,
    pub reason: ExplainReason,
    pub cost: CostBreakdown,
    pub dependencies: DepList,
    pub effects: Vec<EffectTrace>,
}
```

编辑器提供因果火焰图、why-not-ran、live wire value、revision 时间轴、Effect journal 差分和热更 RCC diff。关闭诊断时不进入热路径。

## 9. 热更新与版本

热更新采用 `prepare → drain → publish`：

1. **prepare**：新 Graph 解析、RCC diff、变更 Fragment 编译、能力与成本验证。
2. **drain**：停止旧 epoch 新 Effect；在途调用到最近 Drain barrier；失败则取消发布。
3. **publish**：原子切换 `Arc<Rcc>`，新调用读取新 epoch；旧调用持有旧 epoch 租约。

StableId 相同且 contract digest/config/输入结构未变的 Cell 保留 memo 和 StateCell。变更 Cell 只失效后代闭包；新增 Cell 在新 epoch 首次被需求时计算；移除的未完成 Effect 由 Saga 补偿。

RCC 规范序列化按 StableId、约束 kind、Map key 排序，浮点采用位级编码。结果可进入 Git，支持 Cell 级语义 diff 和三方合并，不使用不可解释的二进制图资产作为唯一真相。

## 10. 安全与平台

- `prism_wasm` 是唯一不可信代码载体；Native 仅接受签名第一方代码。
- WASM 模块在安装阶段校验 digest、签名、Capability 和平台策略。
- `Fuel` 防止指令失控，epoch deadline 防止长时间阻塞，ResourceCaps 限制线性内存与宿主资源。
- attestation 记录模块 digest、输入输出摘要和资源消耗；RCC journal 记录语义层 Effect，两者互补。
- 主机/iOS 禁止运行时 JIT：T4 使用解释形态或认证包内 AOT；逐帧热点使用 T2/T5。
- 不把客户端脚本当安全根；权威状态与外部写入只接受服务器 Effect journal。

## 11. 可扩展性与维护

### 11.1 扩展点

- `#[script_cell]` / `#[script_effect]` 宏：生成 NodeDefinition、EffectRow、reflect schema 和作者元数据。
- `Lowering` trait：新增 GPU、远程、DSP 等执行形态不修改 RCC 语义。
- Capability namespace：新增宿主能力只增加 manifest、reflect API 和 executor。
- Node package：由 `def_digest`、schema version、能力清单和签名组成。
- IR 使用 N-2 兼容窗口；迁移失败拒绝加载并返回结构化诊断。

### 11.2 维护性不变量

1. 所有 Tier 的 `Exact` 输出与 Effect journal 完全一致。
2. 所有真实读取都记录在 DepLog。
3. 外部写入只有 journal commit 后才可观察。
4. 载体内存清空后，StateCell + journal 可以恢复。
5. 约束求解不依赖线程执行顺序。
6. 预算超限一定产生解释结果。
7. RCC 公共类型不暴露后端专有语义。
8. 每项性能优化必须通过差分 golden corpus。
9. 旧 IR 要么在 N-2 窗口加载，要么给出完整迁移诊断。
10. 任何 Capability 扩展都必须带权限测试和审计 schema。

### 11.3 测试矩阵

- 语义：Graph/RCC/T0/T1/T2/T4 差分执行。
- 依赖：隐藏读取、版本变化、early cutoff、循环依赖和 backdating。
- Effect：幂等重试、崩溃恢复、Saga 补偿、journal 损坏、顺序一致性。
- 热更：prepare 失败、drain 超时、在途旧 epoch、状态 schema 迁移、回滚。
- 沙箱：Policy、Fuel、Deadline、内存、WASI、Component digest、typed call、HotSwap、Canary。
- 性能：固定实体数量、固定 dirty ratio、固定 Effect 数量、固定平台档，避免基准漂移。
- 工具：规范序列化、语义 diff、三方合并、why-not-ran、time-travel。

## 12. 模块落地

在 `pkg/prism_visual_scripting/src/` 增加以下模块族：

```text
rcc/
  mod.rs          RCC 公共类型与版本
  build.rs         Graph → RCC
  verify.rs        类型、能力、约束、确定性校验
  diff.rs          StableId 图差分
  manifest.rs      能力、签名、schema、成本清单
memo/
  table.rs         分片 memo 表
  dependency.rs    DepLog 与 revision ledger
  cutoff.rs        early cutoff/backdating
  gc.rs            epoch/LRU 回收
execution/
  demand.rs        demanded root 与 pull 求值
  constraints.rs   Effect 偏序求解
  journal.rs       Effect commit/replay
lower/
  fused.rs         T1
  specialized.rs   T2 guard/deopt
  wasm.rs          T4 prism_wasm adapter
  batch.rs         T5 SoA
explain/
  report.rs        explain/why-not-ran
  trace.rs         因果 trace 与时间轴
```

`prism_wasm` 保持领域无关，包名和目录正式使用 `prism_wasm`。`prism_visual_scripting` 通过可选依赖使用它；WASM feature 关闭时默认构建不产生 WASM 依赖税。

## 13. 完成路线

1. **RCC 基础**：Graph 生成 RCC，T0 执行，revision、DepLog、memo、explain 完整可用。
2. **效应求解**：Constraint DAG、稳定线性化、parallel group、journal 和 Saga 接通。
3. **增量优化**：early cutoff、backdating、依赖索引、预算和质量 fallback。
4. **执行升级**：T1 融合、T2 类型特化/deopt、T5 SoA。
5. **版本连续性**：规范序列化、StableId diff、prepare/drain/publish、StateCell 迁移。
6. **WASM 接入**：T4 适配 `prism_wasm` typed Component；保留 JSON 兼容路径；批量入口和 attestation/journal 对齐。
7. **产品闭环**：编辑器因果火焰图、why-not-ran、time-travel、语义 diff、LLM 约束 patch。

每一阶段都以对应的类型契约、测试矩阵和性能目标验收；不以未定义的功能名、空模块或注释标记代替实现边界。

## 14. 结论

RCC 把 Prism 脚本从“图上的指令序列”升级为“可验证的因果程序”：作者表达依赖和约束，运行时求解时间；Cell 只在因果闭包变化时重算；Effect 以幂等 journal 提交；WASM 由 `prism_wasm` 提供可信执行；热更以 StableId 保留状态；性能由 lowering 与成本模型自动选择；调试以因果 trace 解释每一次运行与不运行。

这不是把多个成熟产品的零件拼接，而是把它们共同解决的核心问题重新统一：**游戏脚本的真正运行单位不是一条指令，而是一组在版本、能力、时间和外部效应约束下可重新求解的因果闭包。**
