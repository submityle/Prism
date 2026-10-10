# Prism Visual Scripting 顶级次世代 AAA 级可视化蓝图系统设计方案

> 面向 Prism（脱离 Bevy 的独立引擎）的 **图编排 + 扁平 IR + 多后端执行器 + 持久化执行 + 可热替换 + LLM 可作者化** 的可视化脚本（Blueprint 对标）内核设计。它是 `prism_script_design_zh.md` 中 `backend-graph` 后端的正式落地：**同一张图既能编译成贴着原生 System 调用的扁平指令链在热路径边缘低开销运行，也能把子图折叠成 WASM Component / 原生 dylib，还能承载事件驱动的持久化长流程（存档即恢复、崩溃可续、轨迹可重放）**。目标：**连线即逻辑、改图即见效、热点不被解释拖垮、长流程不丢状态、机器（LLM）也能可靠生成与自修复图**。
>
> 借形态不抄码。借鉴：
> - **可视化蓝图编译**：Unreal **Blueprint**（exec 白线 + data 彩线、Pure/Impure 节点、K2 节点图编译为 Blueprint 字节码、Event Graph vs Function、**Blueprint Nativization** 把热图转原生）、Unreal **MetaSound**（节点图一次编译为 `IOperator` 算子链、音频线程零分配求值）、**Niagara**（数据导向 VM、SoA 批量、CPU/GPU 双目标）、**PCG Graph**（数据流程程化内容）
> - **数据流图与惰性求值**：Houdini **VOP/VEX**（可视节点编译为 VEX SIMD 字节码、SOP 依赖图增量重算）、Blender **Geometry Nodes**（惰性 field 求值 + 多线程 + 匿名属性）、Unreal/Unity **Shader Graph / Amplify**（DAG 编译为 HLSL）、TouchDesigner / Max/MSP / Pure Data（实时数据流）
> - **节点式工作流 / 自动化**：**n8n** / **Node-RED** / Zapier（触发器节点、分支、重试、可视化调试）
> - **持久化执行 / 事件溯源**：**Temporal** / Azure **Durable Functions** / AWS **Step Functions**（确定性重放、Saga 补偿、定时器/信号、人审批关口、子工作流）、Erlang/OTP（双版本并存 + 状态迁移热加载）
> - **LLM / Agent 编排**：**LangGraph** / Dify / Flowise（流式 token、human-in-the-loop、子图/子 agent、工具调用图）
> - **编译器 IR 优化**：sea-of-nodes（V8 TurboFan / HotSpot C2）的 GVN/CSE/DCE/常量折叠心智，用于 Pure 子图折叠与热路径融合
> - **WASM 组件化沙箱**：wasmtime **Component Model + WASI p2**、内容寻址 + `InstancePre` 缓存 + 原子指针热插拔 + 执行回执（本仓 `uwu_wasm` 为后端参考实现）
> - **引擎内脚本内核**：`uwu_visual_script`（本设计的架构基线：Graph→ExecutionPlan→SlotProgram→VM、Pure/Impure、slot 扁平数组、Effect/Saga/Actor/Await、流式输出、WASM Effect 适配）
>
> 本文聚焦「图作为编译目标 + 可持久化运行时 + 作者/机器双友好」的经典工程路线；不复刻 UE/Unity/Houdini 的对象模型或源码，只借其**节点语义边界、编译形态与调度心智**；核心抽象 `no_std + alloc` 友好，后端执行器需 `std`；**不依赖任何 `bevy_*` crate**。

- 版本: v0.1（设计阶段，未进入编码）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 架构基线: `uwu_visual_script`（独立仓，引擎内可视化脚本引擎内核，提供 Graph 模型 / 三层 IR / slot VM / Effect / Saga / Actor / Await / 流式输出 / 可恢复执行 / WASM Effect 适配）、`uwu_wasm`（独立仓，wasmtime Component 沙箱，提供内容寻址 / `InstancePre` 缓存 / 热插拔 / 能力策略 / 回执 / 时间旅行）
- 关键依赖（均为 `pkg/` 下的 prism 原生 crate）: `prism_reflect`（§12 函数反射 / 路径访问 / `schema` 版本化迁移——**宿主 API 绑定、节点元数据、图状态迁移的唯一真相层**）、`prism_ecs`（Commands/ECB、exclusive system `&mut World`、observer/reaction、`world::snapshot` 回滚底座——**图节点访问世界的受控边界**）、`prism_tasks`（work-stealing + fiber 作业图：off-thread 图编译 / 批量子图并行 / 主线程亲和）、`prism_platform`（`vm` 页级内存服务 WASM 线性内存与 slot 池、`security`/`capability` 能力模型、高精度计时服务 deadline、`dynlib` 服务原生节点后端）、`prism_app`（Plugin/Schedule/`cvar`/`settings`/`fixed`/`determinism`）、`prism_asset`（图资产加载 / mtime 监视 / 热重载管线）、`prism_diagnostic`（`budget`/`hitch`/`profiler`/`replay`/`telemetry`）、`prism_time`（固定步 / 录制回放 / 确定性）、`prism_ui` + `prism_ui_*`（蓝图编辑器画布、inspector、timetravel、devtools、hotreload 的 UI 底座）
- 可选后端依赖: `wasmtime` + `wasmtime-wasi`（节点级/子图级 WASM Component，门控 feature `node-wasm`，参考 `uwu_wasm`）、`cranelift`（可选热图 JIT，feature `jit`）
- 层级定位: `prism_script` §「后端矩阵」中 `backend-graph` 的正式落地；架在 `prism_reflect`（类型/函数真相层）与 `prism_ecs`（世界）之上，**图不进内核仿真内循环**；对上服务 `prism_game_creator`（创作工作流）与 gameplay 可编程性接缝
- 相关文档: `prism_script_design_zh.md`（后端矩阵 / 宿主绑定 / 热更新内核 / 调度接缝——**本文复用其 `ScriptHost`/`BindingRegistry`/`HotSwap`/`WorldProxy` 并补齐图特有的编译与编辑产品化**）、`prism_reflect_design_zh.md`（§12 函数反射 / §24.6 编辑器属性桥 / schema 迁移）、`prism_ecs_design_zh.md`（Commands / exclusive / snapshot）、`prism_tasks_design_zh.md`（并行 / fiber / 主线程亲和）、`prism_diagnostic_design_zh.md`（预算 / hitch / profiler / 回放）、`prism_editor_framework_design_zh.md` + `prism_ui_*`（编辑器画布与调试 HUD）、`prism_gameplay_design_zh.md`（§18 可编程性接缝、§26 Game Features 热插拔）、`prism_sequencer`/`prism_animation_engine_design_zh.md`（图驱动时间轴/状态机的消费者）、`prism_network_design_zh.md` / `prism_anticheat_design_zh.md`（确定性 / 回放 / 权威校验接缝）
- 明确约束: 核心抽象（Graph 模型 / 三层 IR / 注册表骨架 / slot VM 骨架）`no_std + alloc` 友好；异步 VM / Effect / WASM / 编辑器需 `std`；`graph-sync` / `graph-async` / `durable` / `node-wasm` / `node-native` / `jit` / `hot-reload` / `determinism` / `editor` / `llm-authoring` / `trace` 均为 feature，发行版可只带所需能力；默认构建**零图脚本税**（未启用任何图后端时编译期移除）；图**永不下探内核热路径**（ECS 存储内循环、物理 solver、渲染 RG 永远走原生字段）

---

## 目录
1. 设计哲学
2. 现状基线与差距（`uwu_visual_script` 复用面 / 缺口）
3. 分层架构：作者层 / 编译层 / 运行层 / 宿主集成层
4. 图模型：Pin / Node / Edge / Variable / Graph / Control
5. 三层 IR 与编译管线：Graph → ExecutionPlan → SlotProgram
6. 节点模型与注册表：NodeDefinition / Purity / RunnerKind / 元数据
7. 多后端节点执行体：Rust / Async / Effect / WASM / Native / RPC
8. 执行引擎：Slot VM / Workflow VM / 调度器 / 可恢复
9. 持久化与可靠性：Durable / Actor / Saga / Await / 事件溯源
10. 效果层：图能驱动引擎的什么（ECS / Gameplay / 时间轴 / 动画 / 音频 / UI / 网络 / LLM）
11. 性能：IR 优化 / Slot 布局 / 批量 SoA / 热图 AOT·JIT / 零拷贝 ABI
12. 确定性、网络与回放
13. 安全与沙箱：能力白名单 / 预算 / 污点 / WASM 边界
14. 可观测性与调试：时间旅行 / 断点 / 数据监视 / 热重载 HUD
15. LLM 友好性：节点元数据·Schema 导出 / 图↔自然语言 / 编译报错自修复 / Agent 原语
16. 易用性分层与编辑器模型
17. 公共 API 草案
18. Crate 拆分与落地形态
19. 路线图（M0–M7）
20. 关键扩展点、风险与非目标

---

## 1. 设计哲学

1. **图是编译目标，不是解释玩具。** 核心铁律借 Blueprint Nativization / MetaSound / VEX：编辑态 `Graph` 一律 lower 成扁平的 `SlotProgram`（`SlotId` 直接当数组下标，VM 主循环零 HashMap、无递归驱动），热点子图可进一步折叠为单个 WASM Component 或原生函数。**逐节点解释只在冷路径与调试相位出现。**
2. **exec/data 双流 + Pure/Impure 二分是第一性区分。** exec 控制流决定「何时跑」，data 数据流决定「要什么」；Pure 节点 demand-driven（可缓存、可折叠、可池化、可并行），Impure 节点沿 exec 流推进（对应 `CallPure` / `CallImpure`）。这是整套优化与调度的地基。
3. **持久化执行是一等公民。** 借 Temporal/Durable Functions：VM 在每个 block 边界生成可序列化 `VmState`，长流程（任务链、对话、建造流程、剧情）支持 checkpoint→存档→跨进程恢复；副作用经声明式 Effect journaled，重放确定、幂等去重、Saga 逆序补偿。**游戏里的"等玩家 3 天后回来继续"与 Agent 的"等审批"是同一套原语。**
4. **多后端对编译器与 VM 透明。** 借 `uwu_visual_script` 的 `NodeRunner` trait 对象：同一张图可混用 Rust 闭包节点、Async 节点、声明式 Effect、WASM 节点、原生 dylib 节点、未来的 RPC 节点。换后端不改图、不改编译器。
5. **性能分级，诚实标注。** 原生节点（零开销、不沙箱）→ 图内联/Pure 折叠（低解释开销）→ WASM 节点（沙箱、可控开销）→ 异步/Effect（含调度与持久化税）。每档开销与隔离级别明示，作者按场景选；热点可一键"折叠为原生/WASM"。
6. **机器可作者化（LLM-friendly）是硬指标，不是附赠。** 节点自带结构化元数据（描述/分类/schema/示例），可导出机器可读目录；`Graph` 是纯 serde JSON，LLM 可直接吐；编译期静态校验返回**结构化错误**，闭环喂回 LLM 自修复；图可反向摘要为自然语言。**人连线与机器生成走同一条编译/校验管线。**
7. **改完即见效、错了说得清、崩图不崩引擎。** 热重载 < 一帧可感知（内容寻址 + 原子指针翻转 + 状态迁移）；每次节点调用是微沙箱边界（燃料/时限/内存/能力 + trap 捕获 + 超预算熔断）；图错误降级为"这帧这张图不生效 + 带节点定位的诊断"，不是进程崩溃。
8. **图不碰内核热路径。** 与引擎分层铁律一致：图只调用稳定的 gameplay/服务/数据 API（经 `prism_reflect` 绑定表与 `prism_ecs` 受控代理），内层仿真循环永远走原生字段。

---

## 2. 现状基线与差距

### 2.1 已有基础（`uwu_visual_script` 直接复用，不重复造轮子）
- **图模型（编辑/磁盘态已就绪）**：`Graph { nodes, edges, variables, entries, controls }`、`Pin { name, dir, ty, default }`、`Node { id, def, title, config }`、`Edge { from, to }`、`NodeDefRef { id, version }`——完整的可序列化编辑器画布格式，天然适配编辑器与 LLM。
- **三层 IR + 编译管线（核心已就绪）**：`Graph`（源）→ `ExecutionPlan`（可序列化计划，`def_id` 字符串化、不带 trait object、可缓存/签名/远程执行）→ `SlotProgram`（instantiate 后带 `Arc<NodeDefinition>` 的最终程序）。`compile::{plan, instantiate, lower, validate}` 已分层。
- **扁平 slot 运行期（已就绪）**：`Instr::{LoadConst, Move, LoadVar, StoreVar, CallPure, CallImpure, ...}`、`SlotProgram { slots_count, blocks, defs, vars, entries, registry_digest }`、`HALT` sentinel、每 impure 节点一个 Block、Pure 节点反向折叠（`pure_emitted` 缓存）。
- **静态校验（已就绪）**：方向 / exec-data 一致性 / 类型兼容 / Wildcard 拒绝 / Pure 子图无环。
- **值与类型系统（已就绪）**：`ValueType::{Bool,I32,I64,F32,F64,String,Json,List,Wildcard,Exec}`、`Value`（小标量内联、大对象 `Arc`）、隐式转换（`accepts`/`coerce`，`I32→F64`、`*→String`、`Json` 接纳任意）——**已对齐 `wasmtime::component::Val` 形态，便于 WASM 节点桥接**。
- **双轨执行体（已就绪）**：`NodeRunner`（sync）/ `AsyncNodeRunner`（async，可 `.await`）/ `RunnerKind::{Sync,Async,Effect}`；同步 VM 零运行时依赖，异步 VM 两种都能跑；`FnRunner` 闭包适配器。
- **执行环境注入（已就绪）**：`ExecutionEnv` 可挂 `PermissionGate` / `BudgetMeter`（维度含 `steps`/`tokens`/`money_usd`）/ `TraceSink` / `NodeMiddleware` / `DeterminismGate` / `EffectExecutor` / `ExecutionClock` / `ExecutionEventSink` / `CancellationToken` / `ChunkTx`。
- **流式输出（已就绪）**：`Chunk::{Delta(Value), Progress{ratio,message}, Final(Value)}` 可序列化、可跨进程随事件流走（RPC/回放/预热）。
- **持久化与可靠性原语（已就绪）**：可恢复 VM（block 边界 `VmState`，`execution_id` 绑定防串线）、多图 `Call`/`Return` 调用帧、有界并行 `Fork`（all/race join + 并发上限）、`Transition` 状态迁移、声明式 `Effect`（capability/幂等/重试/超时 + `MemoizingEffectExecutor` 完整性校验与 single-flight 去重）、`WorkflowVmState`（task/call frame/join/队列/step budget 可恢复）、Durable `ActorState`（keyed 状态 + 幂等 signal mailbox + fencing lease）、`SagaState`（已提交副作用逆序补偿）、`AwaitCondition::{External,Timer,Signal,Approval,ChildRun}`（可序列化 `Suspension` 恢复闭环）。
- **事件审计 / 回放（已就绪）**：`ExecutionEventSink` + `InMemoryEventLog`（单调序号、`events_since` 增量读取）+ `ExecutionReplay`（`ReplayDivergence`/`ReplayReport`）。
- **窗口流处理（已就绪）**：`WindowedStream`（tumbling/sliding/watermark/allowed lateness/去重/背压）+ deterministic reducers（count/sum/avg/min/max/distinct，绑定 source digest、幂等重放）。
- **身份锁定（已就绪）**：`ExecutionPlan` 绑定 `NodeRegistry` manifest digest，实例化拒绝不同节点快照；`VmState`/`WorkflowVmState` 固定 `execution_id`。
- **WASM Effect 适配（已就绪）**：`node-wasm` feature 下 `WasmEffectExecutor` 经 JSON Component ABI（`operation: func(string)->string`）接入 `uwu_wasm::Sandbox` 的 fuel/deadline/策略/attestation。
- **`uwu_wasm`（WASM 后端参考实现）**：wasmtime Component Model + WASI p2、内容寻址 SHA-256、`InstancePre` 缓存、mtime 轮询 + 原子指针热插拔、Capability/Fuel/Deadline/内存页策略、金丝雀路由 + 自愈、执行回执、时间旅行（snapshot/rewind/diff/replay）。

### 2.2 关键缺口（本设计填补）
`uwu_visual_script` 是**引擎无关的内核**，`prism_script` 只给了 `backend-graph` 三行方针。要达到次世代 AAA + LLM 友好，需补：
- **节点级 WASM / 原生执行体**：当前只有 Effect 级 WASM（`WasmEffectExecutor`）；缺 `WasmRunner: NodeRunner`（`Value↔component::Val` 直接编组，README 标注"待补"）与 `NativeRunner`（经 `prism_platform::dynlib`）。
- **节点元数据与机器可读目录**：`NodeDefinition` 只有 `id/purity/inputs/outputs/runner`，**无 description/category/schema/示例**；LLM 作者化、编辑器节点面板、文档生成都缺真相源。
- **图↔自然语言 / 自修复闭环**：无"NL→Graph"生成辅助、无"Graph→NL"摘要器、无"生成→`compile()` 报错→回灌"的产品化 loop。
- **引擎集成接缝**：无图↔ECS 受控世界访问（查询代理 / 命令缓冲 / 确定性相位），无图↔`prism_reflect` 绑定表的节点自动生成，无图作为 `prism_asset` 资产的热重载接入，无 Prism 调度（主线程亲和 / 帧预算熔断）接缝。
- **编译优化**：常量折叠 + DCE、`TypedCallSite`（消除运行期 HashMap）、Pure cluster 跨 block 共享、Store 池化、热图 AOT 融合 / 可选 cranelift JIT、批量 SoA 子图（Niagara 式数据并行）均在设计计划中但未落地。
- **编辑器产品化**：无蓝图画布 UI、无节点级断点 / 数据监视 / 单步 / time-travel 的交互层（底座在 `prism_ui_*` 已有，需接）。
- **零拷贝 WASM ABI**：当前 WASM 走 JSON 编组（握手期可用，热路径有税）；缺基于 Component Model 类型化参数的零/低拷贝通道。

---

## 3. 分层架构

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ L0 作者层 (Author)                                                          │
│   蓝图编辑器画布 (prism_ui + prism_editor_framework)                         │
│   LLM 作者化 (llm-authoring: NL→Graph / Graph→NL / 自修复 loop)              │
│   节点面板 ← 注册表元数据 / Schema 导出                                       │
│      │ 产出：Graph (serde JSON，编辑/磁盘态)                                  │
├──────────────────────────────────────────────────────────────────────────┤
│ L1 编译层 (Compile) —— off-thread，prism_tasks                               │
│   validate  静态校验（方向/类型/exec-data/Wildcard/Pure 无环）                │
│   lower     Graph → ExecutionPlan（Pure 折叠 / 常量折叠 / DCE / slot 分配）    │
│   optimize  TypedCallSite / Pure cluster 共享 / 热图 AOT 标记                 │
│   instantiate ExecutionPlan → SlotProgram（绑定 Arc<NodeDefinition>）        │
│      │ 产出：SlotProgram（in-proc）+ ExecutionPlan（可缓存/签名/远程）         │
├──────────────────────────────────────────────────────────────────────────┤
│ L2 运行层 (Runtime)                                                         │
│   Slot VM        同步/异步主循环（slot 数组 / 零 HashMap / step budget）       │
│   Workflow VM    多图调用帧 / Fork-Join / Transition / 可恢复                 │
│   Scheduler      FairScheduler（admit / 并发上限 / 全局 step budget）          │
│   Checkpoint     VmState / WorkflowVmState（JSON 序列化恢复）                  │
├──────────────────────────────────────────────────────────────────────────┤
│ L3 宿主集成层 (Host Integration)                                            │
│   BindingRegistry  ← prism_reflect 函数反射（宿主 API → 节点，零手写胶水）     │
│   WorldProxy       ← prism_ecs（只读相位查询 / Commands/ECB / observer 边界）  │
│   EffectExecutor   本地服务 / RPC worker / uwu_wasm::Sandbox                 │
│   节点后端         Rust 闭包 / Async / WASM(uwu_wasm) / Native(dynlib) / RPC   │
│   能力/预算/污点/trace/事件  ← prism_platform / prism_diagnostic             │
└──────────────────────────────────────────────────────────────────────────┘
```

分层铁律：
- **L0→L1 单向**：编辑器与 LLM 只产出 `Graph`，不碰运行期；编译一律 off-thread（`prism_tasks`），编译失败不影响已运行图。
- **L1 对后端透明**：编译器只认 `NodeDefinition` 的 pin 契约与 purity，不认后端是 Rust/WASM/Native。
- **L2 不进内核内循环**：VM 跑在 gameplay/服务相位；需要触碰世界时经 L3 的 `WorldProxy`（受控、确定性可控、延迟结构性变更走 ECB）。
- **L3 是唯一引擎接缝**：宿主 API 全部经 `prism_reflect` 绑定表暴露为节点，不为每个 API 手写节点。

---

## 4. 图模型

沿用 `uwu_visual_script::model`，并补 Prism 侧的元数据与控制语义。

### 4.1 核心类型（基线，已就绪）
```rust
pub type NodeId = u32;  pub type PinIndex = u16;

pub enum PinDir { In, Out }

pub struct Pin { pub name: String, pub dir: PinDir, pub ty: ValueType, pub default: Option<Value> }
// is_exec() == matches!(ty, ValueType::Exec)

pub struct NodeDefRef { pub id: String, pub version: Option<String> }  // 指向注册表

pub struct Node { pub id: NodeId, pub def: NodeDefRef, pub title: Option<String>,
                  pub config: HashMap<String, Value> }  // 不走 pin 的字面量配置

pub struct Endpoint { pub node: NodeId, pub pin: PinIndex }
pub struct Edge { pub from: Endpoint, pub to: Endpoint }

pub struct Variable { pub name: String, pub ty: ValueType, pub default: Option<Value> }

pub struct Graph {
    pub name: String,
    pub nodes: Vec<Node>, pub edges: Vec<Edge>,
    pub variables: Vec<Variable>,
    pub entries: Vec<NodeId>,                 // 事件入口（无 exec 入边的 Impure event 自动识别）
    pub controls: BTreeMap<NodeId, ControlNode>, // 结构化控制语义（多图 Workflow VM）
}
```

### 4.2 结构化控制语义（已就绪）
```rust
pub enum ControlNode {
    FrameEntry,                                   // 调用帧入口
    Return,                                       // 返回
    Call   { target: GraphCall },                 // 跨图调用（真实调用帧，可恢复）
    Fork   { branches: Vec<GraphCall>, policy: JoinPolicy, max_concurrency: u16 }, // all/race
    Transition { state: String },                 // 状态机迁移
}
```

### 4.3 Prism 扩展（本设计新增，向后兼容 `#[serde(default)]`）
- **图种类 `GraphKind`**：`Function`（纯函数子图，无 event）/ `Event`（事件图，有 entry）/ `StateMachine`（状态图，`controls` 以 `Transition` 为主，对标 Bolt State Graph / 动画状态机）/ `Workflow`（长流程持久化图，默认开 `durable`）。决定默认校验规则与调度策略。
- **节点注释 / 分组 / 折叠**：`Node.title` 之外新增可选 `comment`、`group_id`（折叠子图=可复用 Function 的内联视图，对标 Blueprint Collapse Nodes / Macro）。
- **引脚元数据**：`Pin` 增可选 `label` / `tooltip` / `advanced`（高级引脚默认折叠）/ `editor_widget`（inspector 控件提示：slider/color/asset-ref/enum）。
- **确定性标注 `det: Option<DetClass>`**：`Deterministic` / `NonDeterministic`（随机/时间/IO），供 §12 回放与 §13 污点使用。
- **热路径标注 `hot: bool`**：标记子图参与 §11 AOT 融合。

> 兼容性原则：编辑态字段一律 `#[serde(default)]`，老图可无损加载；运行期 `ExecutionPlan` 只固化编译必需字段，装饰性元数据不进程序，保证 plan 稳定可签名。


### 4.4 图资源文件类型与资产管线（本设计新增）

> 回答"节点图是不是有个单独资源文件类型"：**是**。图是 Prism 的一等资产，有独立扩展名、容器格式、稳定 GUID 与依赖图，经 `prism_asset` 导入/热重载——与网格、材质、动画资产同一套管线。

**三类文件（源 / 编译产物 / 运行实例，职责分离）：**

| 扩展名 | 角色 | 格式 | 谁产出 / 谁消费 | 入版本控制 |
|---|---|---|---|---|
| `.prismgraph` | **图源资产**（编辑态真相） | 文本 JSON/RON（`Graph` + 元信息头） | 编辑器 / LLM 产出；编译器消费 | ✅ 入库（可读、可 diff、可 code review） |
| `.prismgraph.plan` | **编译产物缓存**（`ExecutionPlan`） | 二进制/紧凑，内容寻址 digest | 编译器产出；VM/远程/WASM 执行器消费 | ❌ 派生物，进 `prism_asset` 缓存，可重建 |
| （内存/存档内）`VmState` | **运行实例快照**（§9 持久化执行） | serde（JSON/紧凑），绑 `execution_id` | VM 产出；存档/恢复/回放消费 | ❌ 存档数据，非资产 |

**`.prismgraph` 容器头（在 `Graph` 之上包一层资产信封）：**
```rust
pub struct GraphAsset {
    pub magic: [u8; 4],              // "PRMG" 魔数
    pub format_version: u16,         // 容器格式版本（迁移用）
    pub guid: AssetGuid,            // 稳定资产 ID（跨改名/移动不变，引用用它而非路径）
    pub kind: GraphKind,            // Function/Event/StateMachine/Workflow（§4.3）
    pub schema_version: u32,        // 图语义版本（prism_reflect 迁移锚点，§9）
    pub registry_digest: [u8; 32],  // 作者时节点注册表指纹（校验/漂移检测，§15.2）
    pub dependencies: GraphDeps,     // 依赖图：子图 guid / 资产 guid / 节点类型集
    pub editor_meta: EditorMeta,     // 画布布局/视口/折叠组（仅编辑器用，可剥离）
    pub graph: Graph,                // §4.1 本体
}

pub struct GraphDeps {
    pub subgraphs: Vec<AssetGuid>,   // Call/Fork 引用的子图（§4.2），构成资产依赖 DAG
    pub assets: Vec<AssetGuid>,      // config 里引用的 mesh/material/sound/数据表等
    pub node_types: Vec<NodeTypeId>, // 用到的节点类型（准入检查 + 缺失诊断）
}
```

**资产管线关键点（接 `prism_asset`）：**
- **GUID 引用而非路径**：子图/资产引用一律用稳定 `AssetGuid`,改名/移动文件不断链(对标 UE `.uasset` / Unity `.meta` GUID)。路径仅人读索引。
- **导入 = 编译**：`prism_asset` 导入器把 `.prismgraph` 编译为 `.prismgraph.plan`（内容寻址缓存）；源不变则命中缓存零重编译。编译错误(§5.4)作为导入诊断报出,不阻塞其他资产。
- **依赖驱动的热重载（§14）**：mtime/事件监视 `.prismgraph` → off-thread 重编译 → digest 比对 → 原子指针翻转；**子图改动经依赖 DAG 反向失效**父图(对标 Houdini SOP 依赖重算)。运行中实例按 §9 schema 迁移或安全边界重启。
- **编辑元数据可剥离**：`EditorMeta`（画布坐标/折叠组）打包进 `.prismgraph` 便于协作,但 `ExecutionPlan` 不含它——发行版可剥离,保证运行产物精简稳定可签名。
- **烘焙/发行形态**：发行包只带 `.prismgraph.plan`(甚至 §11 AOT 折叠后的 WASM/原生),不带 `.prismgraph` 源与 `EditorMeta`,减小包体并避免逻辑源泄露。
- **LLM 友好对齐（§15）**：`.prismgraph` 的 `graph` 段是纯 serde，LLM 直接读写；`registry_digest` + `node_types` 让"LLM 看到的目录"与"图实际依赖"可交叉校验,杜绝幻觉连法入库。

> 一句话:`.prismgraph`(源,入库可 diff)→ `.prismgraph.plan`(编译缓存,派生)→ `VmState`(运行快照,存档)——三态分离,GUID 依赖图驱动热重载,与引擎其他资产同一套 `prism_asset` 管线。
---

## 5. 三层 IR 与编译管线

### 5.1 管线总览（基线 + Prism 优化）
```text
Graph ──validate──► Graph' ──lower──► ExecutionPlan ──instantiate──► SlotProgram
  (serde)            (校验通过)        (可序列化/缓存/签名)            (in-proc/VM 直解)
                                   ▲ Pure 折叠 / 常量折叠 / DCE
                                   ▲ slot 分配(图着色) / TypedCallSite
                                   ▲ 热图 AOT 标记 / 批量 SoA 划分
```
- **ExecutionPlan**：纯数据（`def_id` 字符串、无 trait object），可缓存、签名、走远程/WASM 执行器；绑定 `registry_digest`。
- **SlotProgram**：instantiate 后带 `Arc<NodeDefinition>`，VM 直接解释；`to_plan()` 可无损投影回 `ExecutionPlan`。

### 5.2 Lower 规则（基线，已就绪）
- **每个 Impure 节点 → 一个 Block**；exec 跳转 = Block 间 `BlockId` 跳转，无递归调用栈。
- **Pure 节点反向折叠**：对每个 impure block 反向递归 emit `CallPure`，`pure_emitted` 缓存消除同 block 内重复求值。
- **Slot 扁平数组**：`SlotId` 直接做下标；编译期完成全部 slot 分配。
- **Wildcard 必须编译期解析**：MVP 直接拒绝未消解 Wildcard，避免运行期类型协商。

### 5.3 Prism 新增编译 Pass（§11 详述性能）
1. **常量折叠（Const Fold）**：Pure 子图中全常量输入的节点在编译期直接求值为 `LoadConst`（借 sea-of-nodes 常量传播）。约束：仅标注 `Deterministic` 且无 config 外部依赖的 Pure 节点可折叠。
2. **死代码消除（DCE）**：无 exec 可达、或输出未被任何活节点消费的 Pure 节点整体删除。
3. **Pure cluster 跨 block 共享**：同一 Pure 子表达式被多个 impure block 消费时，提升为共享 slot（对标 GVN/CSE），避免每 block 重算。需副作用自由 + 确定性保证。
4. **TypedCallSite**：把运行期按名/按类型查找的 pin 映射预编译成定长 slot range + 类型 tag，消除 VM 内 HashMap 查找。
5. **Slot 着色（Liveness + 图着色）**：按活跃区间复用 slot，降低 `slots_count` 与 cache footprint；Pure 临时值优先复用。
6. **热图划分**：`hot` 子图标记 AOT 融合单元；`Deterministic` 的数据并行 Pure 子图标记为批量 SoA 候选。
7. **批量窗口（Batch Lowering，可选）**：对"同一 Pure 子图作用于 N 个实体/粒子"场景，lower 成 SoA 批处理内核（Niagara/VEX 心智），单次调用处理一批。

### 5.4 编译错误模型（结构化，供 §15 LLM 自修复）
```rust
pub enum VsError {
    UnknownDef(String),                 // NodeDefRef 在注册表找不到
    Type { node: NodeId, pin: PinIndex, expected: ValueType, found: ValueType, reason: TypeErrReason },
    Cycle { nodes: Vec<NodeId> },       // Pure 子图自环
    Dangling { node: NodeId, pin: PinIndex },  // 必填 in-pin 未连线且无 default
    WildcardUnresolved { node: NodeId, pin: PinIndex },
    RegistryDigestMismatch { expected: [u8;32], found: [u8;32] },
    Runtime(String),                    // VM 运行期（含 step budget 耗尽）
    AsyncRunnerInSyncVm { node: NodeId },
    Cancelled,
}
```
- 编译期错误一律在 `compile()` 返回且**携带 `NodeId`/`PinIndex` 定位**（编辑器高亮 + LLM 定位修复的前提）；运行期错误从 `vm.run_*` 返回。所有路径以 `VsResult<T>` 暴露。

---

## 6. 节点模型与注册表

### 6.1 NodeDefinition（基线 + Prism 元数据扩展）
```rust
// 基线（已就绪）
pub enum Purity { Pure, Impure }

pub struct NodeDefinition {
    pub id: String,
    pub purity: Purity,
    pub inputs: Vec<Pin>,
    pub outputs: Vec<Pin>,
    pub runner: RunnerKind,
    // ── Prism 新增：机器/人双可读元数据（见 §15）──
    pub meta: NodeMeta,
}

pub struct NodeMeta {
    pub display_name: String,          // "Add (Float)"
    pub category: String,              // "Math/Arithmetic"，驱动编辑器面板树
    pub summary: String,               // 一句话用途（LLM/tooltip）
    pub doc: Option<String>,           // 长文档（Markdown）
    pub keywords: Vec<String>,         // 搜索/LLM 召回
    pub examples: Vec<NodeExample>,    // 输入→输出样例（LLM few-shot / 文档 / 回归测试）
    pub det: DetClass,                 // Deterministic / NonDeterministic
    pub stability: Stability,          // Stable / Experimental / Deprecated{since,replacement}
    pub cost_hint: CostHint,           // Trivial / Cheap / Expensive / IO —— 调度与预算提示
}
```
- `contract_digest()`（已就绪）对 ABI（id + pin 契约 + Effect spec）取稳定 SHA-256，Runner 实现通过 **definition id / registry manifest 绑定**，不依赖 trait object 地址。
- `meta` **不进 `ExecutionPlan`**（装饰性，避免污染可签名计划）；它进 §15 的"节点目录"导出与编辑器面板。

### 6.2 ExecNext（已就绪）
```rust
pub enum ExecNext {
    Pin(String),                 // 走指定 exec 出针
    End,                         // 终止此分支
    Error(String),               // 结构化错误分支（命名 error exec pin 接管）
    Suspend(crate::Suspension),  // 节点边界持久化 continuation（持久化执行闭环）
}
```

### 6.3 注册表（NodeLibrary / NodeRegistry）
- `NodeLibrary`（`dashmap`）注册 `NodeDefinition`，`manifest digest` 锁定整体快照。
- **Prism 新增来源**：
  1. **内置节点**（`builtin`：event.begin / flow.branch / math.* / cmp.* / debug.print / var.* 等，已就绪）。
  2. **反射桥接节点（零手写）**：`#[graph_node]` 过程宏 + `prism_reflect::DynamicFunction`，把宿主 Rust 函数/方法自动注册为节点，pin 由反射签名推导、`meta` 由 doc 注释与属性推导。**这是节点规模化的关键：宿主 API 改签名→节点自动更新。**
  3. **图即节点（Collapse/Macro）**：任意 `Function` 图可注册为可复用节点（内联或 `Call`）。
  4. **WASM/Native 节点**：从 `uwu_wasm` 组件 / `dynlib` 导出注册（§7）。

### 6.4 节点实例配置（已就绪）
- `Node.config`（不走 pin 的常量）进入 `ExecutionPlan`，编译期**规范化键顺序**（保证 plan digest 稳定），运行期经 `InvokeCtx.config` 按实例传递。

---

## 7. 多后端节点执行体

`RunnerKind` 是后端分发点，编译器与 VM 对其透明（借 `uwu_visual_script` trait 对象设计）。

| 后端 | RunnerKind | 隔离 | 开销 | 热更新 | 适用 |
|---|---|---|---|---|---|
| Rust 闭包 | `Sync(Arc<dyn NodeRunner>)` | 无 | 零 | 随 dylib/编译 | 第一方内置、Pure 数学/逻辑 |
| 异步 | `Async(Arc<dyn AsyncNodeRunner>)` | 无 | 调度税 | 同上 | `.await` IO、LLM 调用、长操作 |
| 声明式副作用 | `Effect(EffectSpec)` | 执行器定 | 幂等/重试/持久化税 | 换执行器 | 外部 API、tool-call、不确定 IO |
| WASM 节点（新增） | `Wasm(WasmNodeSpec)` | 强（沙箱） | 可控（实例池/零拷贝 ABI） | 内容寻址原子热插拔 | 第三方/UGC/mod、不可信逻辑 |
| 原生节点（新增） | `Native(NativeNodeSpec)` | 无 | 零 | dylib 热重载 | 第一方热点、需零开销 |
| RPC（预留） | `Rpc(RpcSpec)` | 进程/网络 | 网络税 | 换端点 | 跨进程/跨机服务 |

### 7.1 节点级 WasmRunner（本设计落地 `uwu_visual_script` 的"待补"）
```rust
pub struct WasmRunner {
    sandbox: Arc<uwu_wasm::Sandbox>,
    component: String,   // 内容寻址摘要/名
    export: String,      // 组件导出函数
}

impl NodeRunner for WasmRunner {
    fn invoke(&self, inputs: &[Value], outputs: &mut [Value], cx: &mut InvokeCtx<'_>)
        -> VsResult<ExecNext>
    {
        // 1. Value -> wasmtime::component::Val（类型化，零/低拷贝；大对象经线性内存句柄）
        // 2. sandbox.call_typed::<Params, Returns>(&component, &export, params)
        //    —— fuel/deadline/能力/内存页由 uwu_wasm::Policy 管；回执可验签
        // 3. component::Val -> Value 写回 outputs
        Ok(ExecNext::End)
    }
}
```
- **类型化 ABI（非 JSON）**：`ValueType` 已对齐 `component::Val`，标量直传、`String`/`List`/`Json` 经线性内存 + 句柄（`prism_platform::vm` 页池），**消除 §2.2 的 JSON 编组税**。
- **异步变体 `AsyncWasmRunner`**：`spawn_blocking` 包 `call_typed`，或接 wasmtime async + fuel-based yield，支持长运行组件配合取消。
- 与现有 `WasmEffectExecutor`（Effect 级）并存：Effect 级用于"声明式副作用 + 持久化去重"，节点级用于"纯/快的沙箱计算"。

### 7.2 原生节点 NativeRunner
- 经 `prism_platform::dynlib` 加载第一方 dylib 导出，ABI 由 `#[graph_node]` 宏生成的稳定 C-ABI thunk 固定；热重载走 §14 双版本并存 + 原子指针（借 `prism_script` HotSwap）。**硬约束：Native 不接受非第一方源**（§13）。

### 7.3 流式与取消（已就绪，贯穿所有后端）
- 节点经 `InvokeCtx.chunk_tx` 推 `Chunk::{Delta,Progress,Final}`；长操作定期查 `cx.cancel.is_cancelled()`。VM 主循环每 block 自检取消与 step budget。

---

## 8. 执行引擎

### 8.1 Slot VM（基线，已就绪 + Prism 调优）
- **主循环**：按 `BlockId` 取 Block → 顺序执行 `Instr` → 节点 `invoke` 返回 `ExecNext` 决定下一个 `BlockId`（`HALT = u32::MAX` 终止）。`SlotId` 直接数组下标，**零 HashMap、无递归**。
- **step budget**（已就绪）：默认 1,000,000，`with_step_budget` 覆盖（0=不限），每 block 消耗一格，防死循环；配合 §13 预算熔断。
- **同步 vs 异步**：同步 `run_entry`/`run_all` 遇 `RunnerKind::Async` 返回 `AsyncRunnerInSyncVm`；异步 `run_*_async` 两种 runner 都能跑。可渐进把存量同步节点替换为 async。
- **Prism 调优**：slot 存储用 arena（`prism_platform::vm` 页池）；Pure 临时 Store 池化复用（§11 P0）；热 block 的 `TypedCallSite` 消除查找。

### 8.2 Workflow VM（多图，已就绪）
- `Call`/`Return` 生成真实调用帧（参数/返回/调用栈可恢复）；`Fork` 有界并行（deterministic all/race join + 并发上限 + 失败分支隔离）；`Transition` 产出结构化状态迁移结果由上层状态机持久化。
- `WorkflowVmState` 保存 task / call frame / join / 队列 / 全局 step budget，恢复时校验全部引用。

### 8.3 调度器（已就绪 + Prism 接缝）
- `FairScheduler`（`AdmitResult` / 并发上限 / 全局 step budget / `SchedulerSnapshot`）。
- **Prism 接缝**：图运行作为 `prism_tasks` 作业提交——
  - **主线程亲和**：触碰非 `Send` 宿主资源或需 `&mut World`（exclusive）的图钉在主线程。
  - **批量并行**：无共享可变状态的多图实例（如 N 个 AI 行为树实例）走 work-stealing。
  - **帧预算熔断**：单帧图总步数/时限超 `prism_diagnostic::budget` 阈值则本帧让出、下帧续跑（可恢复 VM 天然支持）。

### 8.4 可恢复执行（已就绪）
- block 边界生成 `VmState`（JSON 可序列化），从下一 block 恢复；`execution_id` 绑定拒绝跨运行串线。长流程（§9）由此获得"存档即恢复"。

---

## 9. 持久化与可靠性

> 这是本设计相对普通蓝图系统的**次世代差异点**：把 Temporal/Durable Functions 的持久化执行语义带进引擎，让"长流程"成为图的一等场景。

### 9.1 Durable 执行（已就绪原语 + Prism 存储）
- `DurableStore`（文件/内存 CAS）+ `RunCoordinator`（`RunRecord`/`RunStatus`/`VersionedCheckpoint`）管理长流程生命周期；接 Prism 存档系统（`prism_asset`/平台存储）。
- 适用：建造/研究/生产链计时、剧情分支状态、玩家离线挂机、Agent 多步任务。

### 9.2 Durable Actor（已就绪）
- `ActorState`（keyed 状态 + 幂等 signal mailbox + in-flight 恢复 + fencing lease），任意 `ActorStore` 持久化。适用：每玩家/每 NPC/每会话的有状态实体，signal 幂等保证"同一指令投递多次只生效一次"。

### 9.3 Saga 补偿（已就绪）
- `SagaState` 只记已提交副作用，失败后逆序生成稳定 compensation `EffectRequest`，逐步 checkpoint、恢复、补偿失败隔离。适用：多步交易/发奖/跨服务操作的回滚。

### 9.4 类型化等待（已就绪）
- `AwaitCondition::{External, Timer, Signal, Approval, ChildRun}` + 可序列化 `Suspension`。
  - `Timer`：定时唤醒（冷却/倒计时/延迟奖励）。
  - `Signal`：等外部事件（玩家交互、网络消息）。
  - `Approval`：human-in-the-loop（GM 审批、Agent 高风险操作确认）。
  - `ChildRun`：等子图/子 agent 完成（分形编排）。
- `InMemoryAwaitBroker` 为默认；生产接持久 broker + Prism 事件总线。

### 9.5 事件溯源与审计（已就绪）
- `ExecutionEventSink` 记录节点边界 / Effect 请求·失败·完成；`InMemoryEventLog` 单调序号 + `events_since` 增量读取；`ExecutionReplay` 做重放投影与 `ReplayDivergence` 检测。供 §12 回放、§14 time-travel、反作弊对账。

---

## 10. 效果层：图能驱动引擎的什么

> 图的价值 = 能驱动多少引擎能力 × 多安全地驱动。所有引擎能力经 §6.3 反射桥接节点暴露，不手写。

- **ECS 世界（经 `WorldProxy`，§3 L3）**：
  - *只读相位*：查询组件、读资源、空间查询（走 `prism_ecs` 只读视图/快照，确定性安全）。
  - *变更相位*：生成/销毁实体、增删组件、改字段——**一律经 Commands/ECB 延迟到同步点**（确定键排序回放），图内不直接 `&mut` 存储。
  - *exclusive 图*：需 `&mut World` 的图标记主线程亲和，走 `prism_ecs` exclusive system。
  - *observer/reaction*：图可作为 observer 回调体，响应组件变更。
- **Gameplay 事件**：事件入口节点（`event.begin` 及派生）绑定 gameplay 事件（OnBeginPlay / OnHit / OnInteract / 自定义 GameplayEvent），对标 Blueprint Event Graph。
- **时间轴 / Sequencer**：图驱动过场、技能 montage、关卡脚本；`Timer`/`Await` 原语做时间编排；与 `prism_sequencer` 轨道互转。
- **动画状态机**：`StateMachine` 图（`Transition` 控制节点）直接作为动画状态图/行为树的作者层，接 `prism_animation_engine`。
- **音频**：触发/参数化音频事件（接 `prism_audio_engine`，节点为参数绑定与 cue 触发）。
- **UI**：图驱动 UI 事件与数据绑定（接 `prism_ui_*` 的 reactive/store），HUD 逻辑、对话系统。
- **网络**：图标注 `Server`/`Client`/`Multicast` 作用域（对标 Blueprint RPC 语义），经 `prism_network` 派发；确定性图参与 lockstep（§12）。
- **LLM / Agent**：异步节点调 LLM（流式 token 经 `Chunk::Delta`）、tool-call 经 Effect、多 agent 经 `Fork`/`ChildRun`、审批经 `Approval`、token/成本经 `BudgetMeter`。**游戏内 NPC"思考"与编辑器内"AI 助手"是同一套图运行时。**
- **流式产出（贯穿）**：任何节点可推 `Delta`/`Progress`/`Final`，驱动进度条、打字机效果、实时预览。

---

## 11. 性能

> 目标：图在 gameplay 相位的开销可预算、可诚实标注；热点可一键降级为原生/WASM/批量；默认零图脚本税。

### 11.1 优化路线（基线计划 P0/P1/P2 + Prism 深化）

**P0（消除运行期查找与重复求值）**
- **TypedCallSite**：pin 映射编译期固化为 slot range + 类型 tag，VM 内零 HashMap（§5.3）。
- **Pure cluster 跨 block 共享**：共享子表达式提升为共享 slot（GVN/CSE 心智），避免每 block 重算。
- **Store 池化**：Pure 节点临时 Store 从页池 arena 分配、按 block 复用，零堆分配热路径。

**P1（减少工作量与提高并行）**
- **常量折叠 + DCE**：编译期求值全常量 Pure 子图、删除不可达/未消费节点（§5.3）。
- **副作用感知并行**：`Fork` + 调度器把无依赖的 impure 分支并行（`prism_tasks` work-stealing），data 依赖图定序。
- **Slot 内联 + Resource handle**：大对象（String/List/Json/Asset）传句柄而非拷贝；WASM 走线性内存句柄。
- **变量短路 / Latent 异步调度**：热变量读写短路到寄存器式 slot；`.await` 节点挂起不占主线程。

**P2（热点原生化与确定性）**
- **热图 AOT 融合**：标 `hot` 的子图编译为**单个 WASM Component 或原生函数**（对标 Blueprint Nativization / MetaSound 一次编译算子链），消除逐节点 dispatch。
- **可选 cranelift JIT（feature `jit`）**：把 `SlotProgram` 热 block JIT 成原生代码（Pure 子图最先受益，无副作用边界清晰）。诚实标注：JIT 带编译 warmup 与内存税，仅热点 opt-in。
- **确定性回放 + Snapshot 懒序列化**：§12；checkpoint 仅在边界懒序列化变更集，避免每步全量快照。

### 11.2 Prism 新增：批量 SoA 数据并行（Niagara/VEX 心智）
- 对"同一 Pure 子图作用于 N 个实体/粒子/样本"的场景（AI 感知批量、粒子逻辑、程序化网格），lower 成 **SoA 批处理内核**：输入输出按列存，单次调用处理一批，内层可 SIMD（经 `prism_math` 向量化）。
- 调度：批量内核交 `prism_tasks` 分块并行；GPU 卸载为长期扩展点（图标注 `gpu` → 子图转 compute kernel，远期，不在 M0–M7）。

### 11.3 WASM 实例化与调用成本压制（借 `uwu_wasm`）
- `InstancePre` 缓存（按 engine+digest）+ **pooling allocator** + 实例池：热路径只做 `instantiate_pre + call`。
- 默认**关 attestation/fuel 旁路**于非安全敏感第一方组件（可配），安全敏感/UGC 组件开满策略。
- 零拷贝类型化 ABI（§7.1）替代 JSON，大对象走线性内存句柄。

### 11.4 性能目标（基准，M4/M6 对齐，`prism_diagnostic` 微基准守门）
- Pure 密集图（数学/逻辑，折叠后）：**解释开销 < 等价手写 Rust 的 1.5×**（slot VM 分支预测友好）。
- Impure 图单节点 dispatch：**< 50 ns**（TypedCallSite，无 HashMap）。
- 热图 AOT/JIT 后：**接近原生**（dispatch 消除）。
- WASM 节点调用（池化 + 类型化 ABI）：**< 1 µs 握手**（标量），大对象线性拷贝带宽受限。
- 每帧图预算熔断粒度：block 级（可恢复），**不产生跨帧 hitch**。

---

## 12. 确定性、网络与回放

- **确定性档（opt-in，feature `determinism`）**：VM 主循环本就无非确定（slot 数组、固定定序）；非确定来源收敛到标 `NonDeterministic` 的节点（随机/时间/IO）。确定性图要求：定点/固定种子随机节点、`prism_time` 固定步时钟、IO 经 journaled Effect。
- **事件溯源重放**：`ExecutionEventSink` + `ExecutionReplay` 记录每次节点边界与 Effect 请求/响应；重放时注入同一响应流，`ReplayDivergence` 检测与权威态分叉——服务录像、网络回滚、反作弊对账复用。
- **网络语义**：节点/图标注 `Server`/`Client`/`Multicast`/`Authority`，经 `prism_network` 派发；lockstep 场景只允许确定性图参与仿真，非确定图降级为表现层。
- **与 `DeterminismGate`（已就绪）**：对普通 Rust runner 做准入检查（Effect 因已 journaled 而豁免），防止确定性相位引入隐藏非确定。

---

## 13. 安全与沙箱

- **能力白名单**：`PermissionGate::check_permission(action, scope)`（已就绪）+ `prism_platform::capability`。图只能调用其能力集内的节点；UGC/mod 图默认最小能力。
- **资源预算**：`BudgetMeter::consume_budget(dimension, amount)`（已就绪），维度 `steps`/`tokens`/`money_usd`/`alloc_bytes`/`io_ops`；超预算熔断（节点返回 `Error` 或 VM `Runtime` 熔断）。
- **WASM 边界（UGC/mod）**：`uwu_wasm::Policy`（fuel/deadline/memory_pages/capability 白名单）+ 内容寻址准入 + 执行回执验签 + 可选 eBPF 双链交叉。崩/trap 捕获降级为"这张图这帧不生效"。
- **污点传播（`TaintPolicy`/`TaintProvider`，已就绪）**：标记不可信输入（玩家文本、网络数据、LLM 输出），阻止其未经净化流入敏感 Effect（如执行命令、写存档）。对 LLM 节点尤为关键（prompt injection 防线）。
- **Native 硬约束**：原生节点零沙箱 → **仅第一方签名源**可注册 Native；第三方/UGC 强制走 WASM。
- **身份锁定（已就绪）**：`ExecutionPlan` 绑定 `registry_digest`，`VmState` 绑定 `execution_id`，拒绝节点快照替换与跨运行串线。

---

## 14. 可观测性与调试

- **时间旅行（借 `uwu_wasm::TimeTravelSession`）**：snapshot / rewind / diff / replay。图层面：在任意 block 边界打点，倒带重放，对比两次运行的 slot/变量差分——定位"为什么这次走了 false 分支"。
- **节点级断点 / 单步 / 数据监视**：VM 可恢复特性天然支持单步（每 block 暂停）；编辑器在节点/引脚挂断点与 watch，实时显示流经引脚的 `Value` 与 `Chunk` 流。
- **事件审计 HUD**：`ExecutionEvent` 流投影为时间线（节点进入/退出、Effect 请求/完成/失败），接 `prism_diagnostic::telemetry` 与 `prism_ui_timetravel`。
- **热路径预算归因**：`prism_diagnostic::hitch`/`profiler` 把图开销归因到具体节点/子图，标红超预算节点。
- **热重载（<一帧可感知）**：图作为 `prism_asset` 资产，mtime 监视 → off-thread 重编译 → 内容寻址比对 → 原子指针翻转；运行中实例按 §9 状态迁移（schema 版本化经 `prism_reflect`）或在安全边界重启。编译失败保留旧版本、HUD 报错不中断。
- **REPL / live 调参**：编辑器内即时改 `Node.config`/变量 default，无需重编译整图（config 已是 plan 的规范化部分，支持细粒度失效）。

---

## 15. LLM 友好性：节点元数据·Schema 导出 / 图↔自然语言 / 编译报错自修复 / Agent 原语

> **这是本设计的硬指标，不是附赠功能。** 用户明确问过"LLM 友好吗"——回答是：基线 `uwu_visual_script` 已经打好三分之二地基（`Graph` 是纯 serde JSON、`VsError` 是结构化错误、slot VM 可单步可重放），缺的那一块（节点自描述元数据 + 机器可读目录 + 自修复闭环 + Agent 作者化原语）由本章补齐。目标：**人连线与机器生成走同一条编译/校验/执行管线，LLM 既能"读图"也能"写图"还能"改图"，且每一步都有结构化反馈可闭环。**

### 15.1 为什么这套架构天然对 LLM 友好（基线既有优势）

1. **图即纯数据（serde JSON / RON）。** `Graph{nodes, edges, variables, entry}` 全程 serde 可序列化，无隐藏指针、无宿主回调句柄。LLM 直接吐一段 JSON 就是一张合法图的候选，无需生成宿主语言代码再编译。对标 n8n / LangGraph 的 JSON 工作流，但多了强类型 pin 与编译期校验。
2. **声明式语义、无控制流歧义。** exec/data 双流 + Pure/Impure 把"何时跑"和"要什么"显式连线，没有隐式求值顺序；LLM 不必推断执行序，只需连对 exec 白线与 data 彩线。
3. **结构化错误天然可闭环。** 编译/校验失败返回带 `node_id` / `pin` / `expected_type` / `found_type` / `error_code` 的结构化 `VsError`，不是自由文本堆栈——可直接序列化回喂给 LLM 做定点修复。
4. **可单步 / 可重放 / 可快照。** VM 每 block 边界可暂停并导出 `VmState`，Agent 能"观测中间态→决定下一步"，天然适配 ReAct / plan-execute 式自省循环。

### 15.2 节点元数据与机器可读目录（补齐 §6.1 `NodeMeta`）

LLM 要"写对图"的前提是"读懂有哪些积木、每块怎么接"。为此 `NodeDefinition` 在 §6.1 基础上强制挂 `NodeMeta`，并由 `prism_reflect` 的 schema 能力生成**机器可读节点目录**（Node Catalog）：

```rust
/// 节点自描述元数据：人读 + 机器读的唯一真相（由 prism_reflect schema 派生/校验）
pub struct NodeMeta {
    pub id: NodeTypeId,              // 稳定全限定名，如 "gameplay.ability.apply_damage"
    pub title: String,              // 人类标题
    pub summary: String,            // 一句话用途（LLM 检索锚点）
    pub docs: String,              // 多行语义/副作用/前置条件说明
    pub category: NodePath,         // 分层目录 gameplay/ability/...
    pub keywords: Vec<String>,      // 同义词/检索扩展（"伤害"/"扣血"/"damage"）
    pub purity: Purity,             // Pure / Impure（决定可否缓存折叠）
    pub runner: RunnerKind,         // Sync/Async/Effect（决定调度与开销档）
    pub determinism: Determinism,   // Deterministic / NonDeterministic（随机/时间/IO）
    pub inputs:  Vec<PinSchema>,    // 每个入 pin 的类型/默认/约束/说明
    pub outputs: Vec<PinSchema>,    // 每个出 pin 的类型/说明
    pub config_schema: JsonSchema,  // Node.config 的 JSON Schema（prism_reflect 导出）
    pub capabilities: CapabilitySet,// 调用所需能力（UGC 图准入检查）
    pub side_effects: Vec<EffectTag>,// 声明式副作用标签（写世界/网络/存档/LLM...）
    pub examples: Vec<GraphFragment>,// 典型连法片段（few-shot 素材）
    pub cost_hint: CostClass,       // 开销档：Native/Inline/Wasm/Async（LLM 预算感知）
    pub stability: Stability,       // Stable/Experimental/Deprecated(+迁移指引)
}
```

- **导出格式**：`prism_script catalog export` 产出 ① 完整 JSON（CI 工件、RAG 索引源）② 精简 JSON Schema（约束 LLM 结构化输出，见 §15.3）③ 分页 Markdown（人读 + 作为系统提示的节点手册）。目录内容寻址（digest），与 `ExecutionPlan.registry_digest` 对齐，保证"LLM 看到的目录"与"运行时注册表"同一真相。
- **检索友好**：`summary + keywords + category` 支撑向量/关键词混合检索——大型目录（上千节点）下，LLM 不塞全量，只按任务检索 top-k 节点 schema 进上下文（对标 MCP tool 列表的分级暴露）。
- **示例即 few-shot**：`examples` 是经编译校验过的真实图片段，既做文档也做 LLM 的 in-context 范例，避免幻觉连法。

### 15.3 LLM 作者化管线（feature `llm-authoring`）：生成→约束→校验→自修复闭环

```
任务(NL) ──检索─▶ 相关 NodeMeta/Schema(top-k)
          └─▶ LLM 结构化输出(受 JSON Schema 约束) ──▶ GraphDraft(JSON)
                    ▲                                       │
                    │              ┌─────────── 编译/校验 ──┘
            结构化 Repair 提示      ▼
            (VsError + 定位 + 建议)  通过? ──是─▶ 干跑/影子执行(§15.4) ──▶ 入库/提审
                    └─────否──────┘  (最多 N 轮，预算熔断)
```

1. **结构化约束生成**：用 §15.2 导出的 JSON Schema 做 LLM 结构化输出约束（function-calling / JSON mode / grammar 约束），从源头压制非法节点名、错 pin、缺字段。
2. **编译即校验**：生成的 `GraphDraft` 走与人类作者**完全相同**的编译管线（§5）：类型检查、pin 兼容、exec 可达性、Pure 无副作用、能力准入、循环/悬空边检测。无特殊通道。
3. **自修复 loop**：失败时把结构化 `VsError`（含 `node_id`/`pin`/`expected`/`found`/`code`/`hint`）+ 相关节点 schema 回喂 LLM，定点修复；每轮受 `BudgetMeter`（tokens/money/轮数）熔断，防止死循环烧钱（§13 预算直接复用）。
4. **可解释产物**：成功的图附带 LLM 生成的"意图注释"（节点/分组 `comment` 字段），人工复核时读得懂机器为什么这么连。

### 15.4 图 → 自然语言（反向摘要与审计）

- **图摘要**：遍历 exec 主干 + 关键 data 依赖，按 `NodeMeta.summary` 生成结构化自然语言描述（"当玩家进入区域→若血量<30%→播放警报音并广播给队友"），供：① 人工 code review ② diff 两版图的语义变更 ③ 给另一 LLM 做"这图干了啥"的上下文。
- **语义 diff**：两张图 lower 到规范化 IR 后做结构 diff（节点增删、连线改动、config 变更），再由 LLM 翻成"这次改动把……"——比逐行 JSON diff 对人/机都友好，接 §14 时间旅行。
- **污点与安全解释**：对含 LLM 节点或不可信输入的图，摘要显式标注污点流向与净化点（§13 `TaintPolicy`），让审计者一眼看到 prompt-injection 防线在哪。

### 15.5 Agent 作者化原语（图作为 Agent 运行时）

本系统不仅"被 LLM 生成"，还能"运行 LLM/Agent 工作流"——持久化执行层（§9/§10）直接就是 Agent 编排底座，对标 LangGraph/Dify：

- **LLM 节点**：`llm.chat` / `llm.structured` / `llm.tool_call` 作为 `RunnerKind::Async` + 流式 `Chunk::{Delta,Progress,Final}`，token 增量经 §10 流出到 UI/对话框；大 prompt/响应走线性内存句柄（§11）不拷贝。
- **工具调用图**：工具即普通节点（Rust/WASM/RPC runner），LLM 的 `tool_call` 输出经 data 流路由到对应工具节点——工具目录与 §15.2 节点目录同源，**一次定义，人连/机调/Agent 用三处复用**。
- **human-in-the-loop**：复用 §9 `Await::Approval`——Agent 流程停在审批关口持久化等待，人在编辑器/游戏内批准后恢复，天然支持"等玩家决策 3 天"。
- **子 Agent / 子图**：`Await::ChildRun` 派生子图（子 Agent），父图挂起等结果；配合 Saga 补偿做"子任务失败→回滚已执行步骤"。
- **记忆与状态**：Agent 的对话/工作记忆就是图变量 + `VmState`，随 checkpoint 持久化；回放（§12）用于复盘"Agent 当时为什么这么决策"。
- **污点与预算护栏**：LLM 输出默认打污点（§13），未经净化不得流入敏感 Effect；tokens/money 走 `BudgetMeter` 维度，超预算熔断。**这是把 Agent 跑在游戏引擎里而不炸预算、不被注入的关键。**

### 15.6 LLM 友好性目标（可验收）

- 给定任务 NL + top-k 节点 schema，LLM 一次生成图的**首轮编译通过率**与**≤3 轮自修复通过率**作为 `llm-authoring` 的回归指标（CI 用固定任务集跑分）。
- 节点目录导出为**稳定内容寻址工件**，LLM 看到的即运行时真相，杜绝"文档与实现漂移"导致的幻觉连法。
- 任意合法图可**双向转换**：图→NL 摘要、NL→图草案，二者经同一 IR 对齐校验。

---

## 16. 易用性分层与编辑器模型

> 一套系统要同时服务"策划连逻辑""Gameplay 程序写系统""TA 搭数据流""UGC 玩家做 mod""LLM 生成图"——用**分层体验**而非一刀切，底座全是同一张图、同一编译管线。

### 16.1 作者分层（L0–L5，能力递增、心智连续）

- **L0 预设/模板**：成品图模板 + 暴露少量参数（`cvar`/变量 default），策划只调数值。零连线。
- **L1 连线作者**：画布拖节点连 exec/data 线——Blueprint 式主体验。pin 类型着色、不兼容连线即时拒绝、Pure 节点折叠显示。
- **L2 子图/函数/宏**：抽子图为可复用 `Function`/`Macro` 节点（对标 BP 函数/宏库、MetaSound graph 复用），参数化、带 `NodeMeta` 进目录。
- **L3 数据流/批量**：VEX/Niagara 心智——对集合/粒子/实体批量跑 Pure 子图（§11.2 SoA 内核），TA 友好。
- **L4 代码节点**：内嵌 Rust（第一方 Native）/ WASM（UGC）/ 表达式 DSL 节点，程序员在图里下沉到代码而不离开画布。
- **L5 机器作者**：LLM 生成/修改图（§15），人在同一画布复核 diff。

### 16.2 编辑器模型（接 `prism_ui` / `prism_ui_*`）

- **画布**：节点/引脚/连线渲染、框选、对齐、分组 `comment`、小地图——`prism_ui` 自定义绘制 + `prism_editor_framework` 停靠面板。
- **Inspector**：选中节点编辑 `config`（由 `config_schema` 驱动自动生成属性控件，复用 `prism_reflect` §24.6 编辑器属性桥）、变量表、图元信息。
- **实时调试叠层**：§14 的断点/单步/数据监视/预算红标直接画在画布上；连线上流动的 `Value`/`Chunk` 可视化（对标 n8n 执行高亮）。
- **时间旅行面板**：接 `prism_ui_timetravel`，倒带/对比/定位分支（§14）。
- **热重载 HUD**：接 `prism_ui_hotreload`，保存即重编译、编译错误浮层带节点定位（§14），不中断运行实例。
- **搜索即生成**：节点搜索框整合 §15.2 目录检索 + 可选"描述一下要干啥"的 NL→图草案入口（L5 入口藏在同一搜索框，渐进暴露）。

### 16.3 防错与引导（降低心智门槛）

- **类型/能力即时反馈**：错连当场拒绝 + 悬浮解释（"输出是 `Entity`，该入口要 `Transform`——要插 `GetTransform` 吗？"）。
- **副作用可视**：Impure/带 Effect 节点配色/角标区分，一眼看出"这节点会改世界/发网络/花钱"。
- **渐进披露**：默认藏高级 pin（可选输入、流控），需要时展开——避免新手被满屏引脚淹没。

---

## 17. 公共 API 草案

> 面向三类消费者：① 引擎集成方（把图挂进 schedule / 资产系统）② 节点作者（注册节点后端）③ 作者化工具（编辑器 / LLM 管线）。以下为意图草案，非最终签名。

```rust
// ── 17.1 节点注册（节点作者）────────────────────────────────
pub trait NodeRunner: Send + Sync {
    fn meta(&self) -> &NodeMeta;
    fn kind(&self) -> RunnerKind;              // Sync / Async / Effect
    fn run(&self, cx: &mut NodeCtx<'_>) -> NodeOutcome; // slot 读写经 cx
}

pub struct NodeRegistry { /* 内容寻址，产 registry_digest */ }
impl NodeRegistry {
    pub fn register<R: NodeRunner + 'static>(&mut self, runner: R) -> Result<(), RegError>;
    pub fn register_wasm(&mut self, component: WasmComponentRef, meta: NodeMeta) -> Result<(), RegError>;
    pub fn digest(&self) -> RegistryDigest;
    pub fn export_catalog(&self, fmt: CatalogFormat) -> Catalog; // §15.2
}

// ── 17.2 编译（引擎 / 工具）─────────────────────────────────
pub fn compile(graph: &Graph, reg: &NodeRegistry, opts: CompileOpts)
    -> Result<ExecutionPlan, Vec<VsError>>;        // 结构化错误 → §15 自修复

pub fn plan_to_program(plan: &ExecutionPlan) -> SlotProgram; // 扁平 IR

// ── 17.3 执行（引擎集成）───────────────────────────────────
pub struct Vm { /* slot 寄存器 + 可恢复状态 */ }
impl Vm {
    pub fn start(program: &SlotProgram, args: Args) -> Vm;
    pub fn step(&mut self, world: &mut dyn HostWorld) -> StepOutcome;  // 单 block
    pub fn run_to_yield(&mut self, world: &mut dyn HostWorld, budget: Budget) -> RunOutcome;
    pub fn snapshot(&self) -> VmState;                 // 持久化 §9
    pub fn resume(state: VmState, program: &SlotProgram) -> Result<Vm, ResumeError>;
}

// 宿主世界边界：图通过受控接口访问 prism_ecs（Commands/ECB/snapshot）
pub trait HostWorld { /* spawn/insert/query/send_event/schedule_effect... */ }

// ── 17.4 持久化执行（Durable）──────────────────────────────
pub trait ExecutionStore {              // 存档/恢复/事件溯源 §9/§12
    fn checkpoint(&self, id: ExecutionId, state: &VmState) -> Result<(), StoreError>;
    fn load(&self, id: ExecutionId) -> Result<Option<VmState>, StoreError>;
    fn append_event(&self, id: ExecutionId, ev: &ExecutionEvent) -> Result<(), StoreError>;
}
pub trait EffectExecutor { fn execute(&self, req: EffectRequest) -> EffectFuture; } // §10

// ── 17.5 LLM 作者化（feature llm-authoring）─────────────────
pub struct Authoring<'a> { catalog: &'a Catalog, reg: &'a NodeRegistry }
impl<'a> Authoring<'a> {
    pub fn draft_from_nl(&self, task: &str, retr: &Retriever, llm: &dyn LlmClient) -> GraphDraft;
    pub fn repair(&self, draft: GraphDraft, errs: &[VsError], llm: &dyn LlmClient) -> GraphDraft;
    pub fn summarize(&self, graph: &Graph, llm: &dyn LlmClient) -> String; // 图→NL §15.4
}
```

- **稳定性分层**：`compile`/`Vm`/`NodeRunner`/`NodeMeta` 为稳定核心；`Authoring`/`jit`/`determinism` 为门控扩展。
- **错误即数据**：所有 `*Error` 均 serde 可序列化、带 code + 定位，服务 §15 自修复与 §14 诊断 HUD。

---

## 18. Crate 拆分与落地形态

> 对标 `prism_script` 的 feature 门控原则：**默认零图脚本税**（不启用图后端时编译期全移除），发行版按需裁剪。

| crate | 职责 | 关键 feature | `no_std` |
|---|---|---|---|
| `prism_script_graph_core` | Graph 模型 / 三层 IR / 注册表骨架 / `NodeMeta` / 结构化错误 | — | ✅ `no_std + alloc` |
| `prism_script_graph_compile` | 编译管线 / 类型检查 / IR 优化（折叠/DCE/CSE）/ lower | — | ✅ |
| `prism_script_graph_vm` | slot VM / 可恢复执行 / 调度 | `graph-sync` / `graph-async` | 核心 ✅，async 需 `std` |
| `prism_script_graph_durable` | 持久化 / Actor / Saga / Await / 事件溯源 | `durable` | `std` |
| `prism_script_graph_wasm` | 节点级/子图级 WASM Component 后端（参考 `uwu_wasm`） | `node-wasm` | `std` |
| `prism_script_graph_native` | 原生 dylib 节点后端（`prism_platform::dynlib`） | `node-native` | `std` |
| `prism_script_graph_jit` | 热图 cranelift JIT | `jit` | `std` |
| `prism_script_graph_llm` | LLM 作者化 / 目录导出 / 自修复 / 图↔NL | `llm-authoring` | `std` |
| `prism_script_graph_editor` | 编辑器画布 / inspector / 调试叠层（接 `prism_ui_*`） | `editor` | `std` |
| `prism_script_graph_runtime` | 引擎集成：资产加载 / 热重载 / schedule 挂点 / HUD | `hot-reload` / `trace` | `std` |

- **feature 组合示例**：服务器确定性仿真 = `graph-sync + durable + determinism`（无 editor/llm/wasm）；UGC 客户端 = `graph-async + node-wasm + hot-reload`（强沙箱）；创作工作站 = 全开。
- **依赖方向**：core ← compile ← vm ← {durable, wasm, native, jit} ← runtime；editor/llm 旁挂，不进运行时关键路径。

---

## 19. 路线图（M0–M7）

- **M0 骨架对齐**：移植 `uwu_visual_script` 的 Graph/IR/slot VM 骨架为 `prism_script_graph_core/compile/vm`，接 `prism_reflect` 类型真相层、`prism_ecs` 世界边界。产出：Pure/Impure 同步图可编译可跑，`prism_diagnostic` 微基准接入。
- **M1 节点模型与目录**：`NodeMeta` + 注册表内容寻址 + `prism_reflect` schema 导出目录（§15.2）。产出：节点手册 JSON/MD 自动生成，Inspector 属性自动生成。
- **M2 异步与效果层**：`RunnerKind::Async` + 声明式 Effect + 流式 `Chunk`（§10），接 `prism_tasks` 调度。产出：异步节点、流式输出、主线程亲和。
- **M3 持久化执行**：`VmState` 快照/恢复 + `ExecutionStore` + Actor/Saga/Await（§9），事件溯源骨架。产出：长流程存档恢复、崩溃续跑。
- **M4 性能优化相位**：IR 折叠/DCE/CSE、slot 布局优化、批量 SoA 内核（§11.2）、热图 AOT 融合；达成 §11.4 基准。
- **M5 WASM/Native 节点后端**：`node-wasm`（参考 `uwu_wasm`：`InstancePre` 缓存/pooling/Policy/回执）+ `node-native`（签名源 dylib）。产出：UGC 沙箱节点、第一方原生热点节点，补齐 README 标注的节点级 `WasmRunner` 缺口。
- **M6 确定性/网络/回放 + 可选 JIT**：`determinism` 档 + 事件溯源重放 + `prism_network` 接缝（§12）；可选 cranelift `jit` 热块原生化。产出：lockstep 可用、服务录像回放、反作弊对账。
- **M7 LLM 作者化 + 编辑器完善**：`llm-authoring` 生成→校验→自修复闭环 + 图↔NL（§15）；编辑器时间旅行/热重载 HUD（§16）全量。产出：首轮/≤3 轮自修复通过率回归上线，L0–L5 分层体验完整。

---

## 20. 关键扩展点、风险与非目标

### 20.1 扩展点
- **自定义 `NodeRunner` 后端**：RPC 节点（跨进程/跨机微服务图）、GPU compute 节点（§11.2 远期）、脚本语言桥（Lua/JS 节点）均为新 runner，不动编译器/VM。
- **自定义 Effect / EffectExecutor**：接新宿主能力（新网络协议、新持久化后端、新 LLM provider）只加 Effect 类型 + executor。
- **自定义 IR pass**：编译优化（新折叠规则、领域特定 lower）作为 pass 插入 §5 管线。
- **目录检索后端**：§15.2 catalog 可接任意向量库/关键词引擎做 LLM 检索。

### 20.2 风险与缓解
- **JIT 复杂度与收益**：cranelift 带 warmup/内存税——设为 opt-in、仅热点、诚实标注；AOT WASM 折叠作为更保守的默认热点方案。
- **持久化 schema 漂移**：长流程存档跨版本恢复难——`VmState` 绑 `registry_digest` + `prism_reflect` schema 版本化迁移（§9），迁移失败在安全边界重启而非静默错乱。
- **LLM 幻觉连法**：靠结构化 Schema 约束 + 同一编译校验管线兜底（§15.3），非法图根本编不过；自修复轮数/预算熔断防死循环。
- **WASM 调用开销**：池化 + 类型化 ABI + 大对象句柄压制到 <1µs 握手（§11.3）；仍不够则 AOT 折叠为单 Component。
- **图被滥用进热路径**：架构铁律——图永不下探 ECS 内循环/物理 solver/渲染 RG（§元信息约束），`prism_diagnostic` 预算归因在 CI 守门。

### 20.3 非目标
- 不复刻 UE/Unity/Houdini 的对象模型或源码，只借节点语义/编译形态/调度心智。
- 不做"全引擎皆可视化编程"——图是 gameplay/流程/数据流/Agent 编排层，不替代系统级 Rust 代码。
- 不在本设计内落地 GPU compute 卸载、分布式图跨机调度（列为远期扩展点）。
- 不提供运行时任意代码执行通道——第三方/UGC 强制 WASM 沙箱，Native 仅签名源。

---

> **一句话总结**：Prism 可视化脚本以「**图是编译目标、持久化是一等公民、机器与人同管线**」为骨——编辑态的强类型节点图被 lower 成零 HashMap 的扁平 slot IR 在热路径边缘低开销运行，热点可折叠为 WASM/原生消除解释，长流程经声明式 Effect 与可序列化 `VmState` 做到存档即恢复、崩溃可续、轨迹可重放，而节点自描述元数据 + 结构化错误 + 自修复闭环让 LLM 既能可靠读图、写图、改图，也能把 Agent 工作流直接跑在引擎里——**连线即逻辑、改图即见效、热点不被解释拖垮、长流程不丢状态、机器也能可靠作者化**。
