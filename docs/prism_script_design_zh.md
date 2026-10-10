# Prism Script 顶级次世代 AAA 级热更新脚本系统设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **单载体、可热更新、可沙箱、可确定性回放** 脚本运行时内核设计。它不是「再造一门语言」，也不是「并列四种后端」，而是把「运行时可编程性」收敛为一条清晰的主干：**WASM 组件作为唯一的运行时热更脚本载体 ＋ 反射驱动的宿主 API 绑定层 ＋ 内容寻址热替换 ＋ 状态迁移 ＋ 灰度回滚 ＋ 预算化沙箱调度**。目标：**改脚本不重启、崩脚本不崩引擎、热路径不被脚本拖垮、改完即见效**。
>
> **单载体立场（本文的核心收敛）**：运行时热更的脚本代码**只有 WASM 一种载体**。可视化图脚本是另一层作者形态，**完整归属 `prism_visual_scripting_design_zh.md`**，本文只在绑定层与之共享 host API，不重写它；第一方原生代码的开发期热重载是**引擎机制**，归 `prism_app_design_zh.md` §24.2，**不是脚本载体**。不再保留「Native / Graph / WASM / Rhai 四类载体菜单」，也**不含 Rhai 等嵌入式解释语言**。
>
> 借形态不抄码。借鉴：
> - **可下发字节码 + 原生热点**：Unreal **Blueprint**（字节码跨平台解释、随 pak 下发、可热改）＋ nativized/C++（随包编译、不热更）——本文的 WASM 解释 ≈ Blueprint 字节码，AOT/随包原生 ≈ nativized
> - **能力安全脚本**：UEFN **Verse**（可热更新、能力安全、强类型）、Roblox **Luau**（沙箱化 + 渐进类型）、Garry's Mod / WoW Lua modding（能力白名单）
> - **WASM 作为可信边界 ABI**：**Fastly Compute / Extism**（WASM 插件 ABI）、wasmtime **Component Model + WASI p2**（typed import/export、内容寻址 + `InstancePre` 缓存 + 原子指针热插拔，本仓 `uwu_wasm` 为后端参考实现）
> - **托管运行时热重载**：Unity C# **Domain Reload / ScriptableObject 热重载**、**Erlang/OTP 热代码加载**（双版本并存 + `code_change` 回调迁移状态）
> - **可下发内容包**：Unreal **Game Feature Plugin**（code + content 可插拔单元）/ `.pak` 挂载、UEFN **Verse 包**
> - **AOT 预编译**：wasmtime `Engine::precompile_module` / `Component::serialize` → `.cwasm`（cook 期预编译，跳过运行期 Cranelift，冷启动快）
> - **确定性 / 回放**：lockstep + 定点量化 + 事件溯源重放（联机回滚 / 录像 / 反作弊对账）
>
> 本文为纯经典运行时 / 编译 / 沙箱 / 调度路线，**不含任何 AI/ML 内容**；不复刻 UE/Unity/Roblox 的对象模型或源码，只借其**可编程性边界划分与热更新形态**；**不依赖任何 `bevy_*` crate**。

- 版本: v0.2（设计阶段，未进入编码；相对 v0.1 收敛为单一 WASM 载体）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖（均为 `pkg/` 下的 prism 原生 crate）: `prism_reflect`（函数反射 §12 / 路径访问 / `schema` 版本化迁移 / 脚本桥 §24.6——**宿主 API 绑定与状态迁移的唯一真相层**）、`prism_ecs`（Commands/ECB、exclusive system `&mut World`、observer/reaction、`world::snapshot` 回滚底座）、`prism_tasks`（work-stealing + fiber 作业图：off-thread 编译 / 批量脚本并行 / 主线程亲和）、`prism_platform`（`vm` 虚拟内存页级底座服务 WASM 线性内存与池化、`security`/`capability` 能力模型、高精度计时）、`prism_app`（Plugin/Schedule/`cvar`/`settings`/`platform_tier`/`fixed`/`determinism`；**第一方代码热重载 §24.2**）、`prism_asset`（脚本资产加载 / **容器打包与挂载 `.ucas`+`MountTable` §13、热重载与依赖失效 §12、patch/DLC/CDN §23.4/§23.5**，脚本包分发全部复用它）、`prism_diagnostic`（`budget`/`hitch`/`profiler`/`replay`/`telemetry`）、`prism_time`（固定步 / 录制回放 / 确定性）
- 可选后端依赖: `wasmtime` + `wasmtime-wasi`（WASM Component Model，门控 feature `backend-wasm`，参考本仓 `uwu_wasm`）；主机/iOS 认证档可选 `wasmi`（纯解释器，运行时热更兜底，feature `backend-wasm-interp`）
- 层级定位: Gameplay 文档 §18「可编程性：脚本与可视化蓝图」中**脚本运行时**部分的正式落地；架在 `prism_reflect`（类型真相层）与 `prism_ecs`（世界）之上，**不进内核热路径**
- 相关文档: `prism_visual_scripting_design_zh.md`（**图脚本的全部内容——图模型 / 三层 IR / 节点执行 / 图编辑器**，本文不重写，仅共享绑定层）、`prism_app_design_zh.md`（§24.2 第一方代码/模块热重载、`platform_tier`、确定性）、`prism_asset_design_zh.md`（§12 热重载与依赖失效、§13 打包容器 `.ucas`/`MountTable`、§23.4/§23.5 CDN/patch/DLC）、`prism_reflect_design_zh.md`（§12 函数反射 / §24.6 脚本编辑器属性桥 / schema 版本化）、`prism_ecs_design_zh.md`（Commands / exclusive system / snapshot）、`prism_tasks_design_zh.md`（并行 / fiber / 主线程亲和）、`prism_platform_design_zh.md`（vm / security / capability）、`prism_diagnostic_design_zh.md`（预算 / hitch / profiler / 回放）、`prism_gameplay_design_zh.md`（§13 数据驱动、§26 Game Features 运行时插拔）、`prism_game_creator_design_zh.md`（§24 迭代/PIE、§26 内容包、`prism_gc_ugc` UGC/Mod 消费者）、`prism_network_design_zh.md` / `prism_anticheat_design_zh.md`（确定性 / 回滚 / 权威校验接缝）
- 明确约束: 核心抽象 `no_std + alloc` 友好（绑定表 / 句柄 / 调度骨架）；WASM 执行器需 `std`；`backend-wasm` / `backend-wasm-interp` / `hot-reload` / `determinism` / `sandbox` / `trace` 均为 feature；默认构建**零脚本税**（未启用后端时编译期移除）

---

## 目录
1. 设计哲学
2. 现状基线与差距：复用什么、本文只做什么
3. 分层架构
4. 单一载体：为什么只用 WASM（图脚本 / 原生码的边界）
5. 核心抽象：Host / Module / Context / Binding / Handle
6. 宿主 API 绑定层：反射驱动、稳定 ABI、零手写胶水（本文独占）
7. ECS 接入模型：世界访问、命令缓冲、查询代理
8. WASM 执行底座：编译 / 实例化 / 调用（本文独占深化）
9. 热更新机制：内容寻址 + 原子替换 + 零中断（复用 app/asset + AOT 缓存）
10. 状态迁移与版本化：复用 reflect schema，改脚本不丢状态
11. 性能设计（速度）
12. 安全与沙箱（零信任：三档能力门控）
13. 调度与执行模型（并行 / 帧预算 / 主线程亲和）
14. 效果层：脚本能驱动什么
15. 确定性、网络与反作弊接缝
16. 可观测性与调试（时间旅行 / 热路径预算）
17. 易用性分层与默认体验
18. 公共 API 草案
19. Crate 拆分与落地形态
20. 分发与打包：Prism 脚本包（复用 asset 容器 + code chunk 独占）
21. 平台执行矩阵：解释 / JIT / AOT × PC / 主机 / 移动 × 认证铁律
22. 路线图（M0–M5）
23. 关键扩展点、风险与非目标

---

## 1. 设计哲学

1. **可编程性是一层绑定，不是一门语言；运行时热更只有一个载体。** 真相在 `prism_reflect`：类型、函数、路径、schema 版本都从反射来。运行时热更脚本统一走 **WASM 组件**——跨语言、强沙箱、可热替换、可下发。不再并列「原生 / 图 / WASM / 解释器」四后端：图脚本是**另一层作者形态**（§4，归 VS 文档），原生代码热重载是**引擎开发期机制**（§4，归 app §24.2）。
2. **热更新是一等公民，而非事后补丁。** 从第一天就按「双版本并存 + 原子指针翻转 + 旧调用零中断 + 状态迁移回调 + 灰度/回滚」设计（借 Erlang/OTP 与 `uwu_wasm` 热插拔）。
3. **不重造已有的轮子，只做脚本层增量。** 容器打包、挂载、CDN/patch、资产热重载与依赖失效、第一方代码热重载、Game Feature 运行时插拔——**全在兄弟文档已设计**（§2 清单）。本文只做它们没覆盖的脚本层：**WASM 执行底座 + 反射绑定层 + 沙箱门控 + code chunk 交接 + 脚本层确定性/预算调度接缝**。
4. **脚本不碰内核热路径。** 分层铁律：脚本只调用 Layer 3/4 的稳定 API（gameplay / 服务 / 数据），内层仿真循环（ECS 存储、物理 solver、渲染 RG）永远走原生字段，反射与脚本边界不下探（与 `prism_reflect` §「边界使用原则」一致）。
5. **崩脚本不崩引擎。** 每次脚本调用是一个「微沙箱边界」：资源上限（燃料 / 时限 / 内存页）、trap 捕获、超预算熔断、坏版本自愈回退。脚本错误降级为「这一帧这个脚本不生效 + 诊断」，不是进程崩溃。
6. **改完即见效、错了说得清。** 易用性是可量化目标：热重载 < 一帧可感知、错误带源码位置与调用栈、作者层一行接入、编辑器内 live 调参。
7. **诚实标注平台铁律。** 「下发并执行代码」在主机（PS/Xbox/Switch）与 iOS 上受认证约束。本文不假装没有这堵墙：PC/Android 走 JIT（又快又热）、cook 期 AOT 随签名包（快但随包）、主机/iOS 运行时热更兜底走解释器（可热但慢）——按平台显式选形态（§21）。

---

## 2. 现状基线与差距：复用什么、本文只做什么

### 2.1 已有基建（必须引用而非重写）
下表的能力**在兄弟文档里已经设计好**。本文**只引用、不重写**；脚本层仅在交接点（验签 / 实例化 / 注册）接上去。

| 能力 | 已有归属 | 本文如何复用 |
|---|---|---|
| 第一方代码 / 模块开发期热重载（dylib 卸旧符号 → 载新 → 重接 Schedule） | `prism_app` §24.2 | §4/§9 一句指针；本文不做原生码热替换 |
| 资产热重载 + 依赖失效传播 | `prism_asset` §12 | §9 脚本字节作为资产，复用其 mtime 监视与失效通知 |
| 打包 / 容器 `.ucas` / `MountTable` / 优先级叠挂 | `prism_asset` §13 | §20 脚本包 = 一个 `.ucas`，挂载走同一路径 |
| patch / DLC / 内容目录 / 远程 CDN | `prism_asset` §23.4/§23.5 | §20 脚本热更 = 高优先级容器叠挂，不自造分发 |
| 数据驱动热重载 + Game Feature 运行时插拔 | `prism_gameplay` §13/§26 | §14 脚本模块可作为 Feature 逻辑载体随之插拔 |
| PIE / 迭代 / UGC 包 / Mod 装配 | `prism_game_creator` §24/§26、`prism_gc_ugc` | §17 作者工作流上层消费者 |
| 函数反射 / 路径访问 / schema 版本化迁移 / 脚本属性桥 | `prism_reflect` §12/§24.6 | §6/§10 绑定与状态迁移的唯一真相层 |
| 图脚本：图模型 / 三层 IR / 节点执行 / 图编辑器 | `prism_visual_scripting` 全文 | §4 一句指针；本文只在绑定层与之共享 host API |
| 回滚底座 / 快照 / Commands / 并行 ECB | `prism_ecs` | §7 世界访问与回滚相位复用 |
| 并行 / fiber / 主线程亲和 / 帧预算 | `prism_tasks` | §13 调度完全架在其上，不自建线程池 |
| 预算 / hitch / profiler / 回放 / 时间旅行 | `prism_diagnostic` | §16 脚本诊断复用 |
| WASM 后端参考实现（wasmtime CM + 内容寻址 + 热插拔） | 独立仓 `uwu_wasm` | §8 `backend-wasm` 的直接蓝本 |

### 2.2 本文真正独占的增量（填补上述都没有的脚本层）
Gameplay §18 只给了「Rust 一等公民 / 可选图脚本 / 可选 WASM」的三行方针，**没有**成体系的脚本运行时。以下**没有任何兄弟文档覆盖**，是本文独占并深化的：

1. **WASM 执行底座**（§8）：`compile → instantiate_pre → call` 全链路、`InstancePre`/pooling allocator、fuel/deadline/内存页预算、AOT 预编译——借 `uwu_wasm` 并对其默认配置做热路径调优。
2. **反射驱动的宿主 API 绑定层**（§6）：`#[script_api]` 派生宏 → `BindingRegistry`，建于 `prism_reflect` derive 之上。reflect 本身没写这层，这是脚本独有；**图脚本（VS 文档）也消费这同一张绑定表**。
3. **WASM 沙箱能力门控**（§12）：Trusted / Sandboxed / Fortified 三档随签名信任根映射。
4. **code chunk 交接协议**（§20）：字节 → 验签 → 实例化 → 注册进 `ScriptHost`，与 asset loader 协议同构但代码专属。
5. **脚本层确定性 / 回放 / 预算调度接缝**（§13/§15）：架在 `prism_tasks` / `prism_diagnostic` 上的脚本专属车道与熔断。

> 相对 v0.1 的删改：**删除** Rhai 后端与「四类载体」矩阵；**移出** 图脚本编译/求值（归 VS 文档）；**降级** Native dylib 为一句边界说明（归 app §24.2）；**保留深化** 上述 5 项脚本层独占增量。

---

## 3. 分层架构

```
L5  作者 / 工具层   编辑器 Inspector 绑定、REPL/live 调参、热重载 HUD
                     （图脚本编辑器在 VS 文档；本文只提供绑定表 → 补全/签名提示）
──────────────────────────────────────────────────────────────────────
L4  WASM 执行器      WasmBackend（wasmtime Component Model + WASI p2）
                     —— 持有编译缓存（content-addressed）/ 实例池 / 策略
                     —— 主机/iOS 认证档可切 wasmi 解释器实现同一 trait
──────────────────────────────────────────────────────────────────────
L3  脚本内核        ScriptHost（注册表 / 分发 / 生命周期）
                     HotSwap（内容寻址 + 原子指针 + 灰度 + 自愈）
                     StateMigrator（schema 版本化状态迁移，建于 reflect）
                     Scheduler 接缝（prism_tasks：off-thread 编译 / 批量 / 亲和 / 预算）
──────────────────────────────────────────────────────────────────────
L2  绑定与访问      BindingRegistry（反射驱动，host fn / 组件读写 / 资源 / 事件）
                     ↑ 图脚本（VS 文档）与 WASM 共享此表，是唯一的跨形态交点
                     WorldProxy（Commands / 查询代理 / snapshot 只读视图）
                     Policy（capability 白名单 + 资源上限 + 确定性档）
──────────────────────────────────────────────────────────────────────
L1  真相与底座      prism_reflect（类型/函数/路径/schema）  prism_ecs（世界）
                     prism_platform（vm/security）  prism_tasks（并行）
```

**铁律**：L4 WASM 执行器只能通过 L2 的 `WorldProxy` + `BindingRegistry` 访问世界，拿不到 `&mut World` 原始引用，拿不到宿主裸内存。脚本永远跑在 L3/L4，内核仿真热路径（L1 以下）永不进脚本。

---

## 4. 单一载体：为什么只用 WASM（图脚本 / 原生码的边界）

**运行时热更脚本 = WASM 组件，唯一载体。** 这不是「四选一后保留一个」，而是对「哪种东西才是运行时可下发、可热替换、可沙箱的代码」的收敛结论。另两样东西不是被删，而是**本来就不属于本文**：

| 东西 | 它是什么 | 归属 | 本文如何处理 |
|---|---|---|---|
| **WASM 组件** | 运行时可下发、强沙箱、跨语言、可热替换的**脚本代码** | **本文**（§5–§21） | 唯一载体，全文深化 |
| **图脚本（Blueprint 对标）** | 可视化**作者形态**，有自己的图模型 / 三层 IR / 节点执行引擎 / 图编辑器 | `prism_visual_scripting_design_zh.md` 全文 | 只在 §6 绑定层共享同一张 host API 表；执行与编译一概不在本文 |
| **第一方原生代码（dylib）** | 随签名安装包发货的**引擎/玩法编译码**；开发期可热重载 | `prism_app_design_zh.md` §24.2 | 不是脚本载体；§9 一句指针，不做原生码热替换 |

### 4.1 为什么是 WASM（对标 UEFN Verse / Fastly / Blueprint 字节码）
- **跨语言**：任何能编到 WASM Component 的语言（Rust / C / C++ / AssemblyScript / Zig …）都能写脚本/mod——这是跨语言诉求的正解，而非自造一门语言或绑死 Lua。
- **强沙箱**：线性内存隔离 + 能力白名单 + 燃料/时限/内存页上限，是接受第三方/玩家代码的唯一安全选项（§12）。原生 dylib 无内存沙箱，**结构上就不能**当 UGC 载体。
- **可热替换**：内容寻址 + `InstancePre` 缓存 + 原子指针翻转，天然零中断热更（§9）。
- **可下发 + 跨平台一致**：PC/Android 走 JIT，主机/iOS 走 AOT（随包）或解释器（热更兜底），一份 `.wasm` 真相源按平台 cook（§21）——正是 UE Blueprint「字节码到处跑、热点烘原生」智慧的现代同构。

### 4.2 为什么图脚本不在本文
图脚本是**给策划/设计师的可视化作者层**，它的价值在节点模型、类型化 pin、图编辑器、图→IR 编译——这些在 VS 文档已成体系。它与本文的唯一交点是：**图编译后调用的 host API，与 WASM 调用的是同一张 `BindingRegistry`**（§6）。所以「写一次 Rust 的 `#[script_api]`，图脚本节点面板和 WASM import 同时可见」——这是本文绑定层带来的红利，但图的编译/执行不该在脚本运行时文档里重写。

### 4.3 为什么原生 dylib 不是脚本载体
第一方原生代码的开发期热重载（改 gameplay crate → 增量编译 → 卸旧符号载新符号 → 重接 Schedule、持久 state 外置）是 **`prism_app` §24.2** 的引擎机制，面向第一方工程师的本地迭代，**不经下发、不受沙箱、不接受 UGC 来源**。把它叫「脚本后端」会模糊「可信第一方编译码」与「可下发沙箱脚本」的安全边界。因此本文只在 §9 留一句指针，§12 明确「原生码无内存沙箱，不接受非第一方源」。

---

## 5. 核心抽象：Host / Module / Context / Binding / Handle

```rust
/// 脚本内核门面：一个 App/World 通常持有一个。
pub struct ScriptHost {
    backend:  WasmBackend,              // 唯一后端（JIT 或 wasmi 解释，按平台）
    bindings: Arc<BindingRegistry>,     // 反射驱动的宿主 API 表（与图脚本共享）
    hotswap:  HotSwap,                  // 内容寻址 + 原子指针热替换
    migrator: StateMigrator,            // schema 版本化状态迁移（建于 reflect）
    policy:   PolicyTable,              // per-模块能力/资源策略
    scheduler: ScriptScheduler,         // prism_tasks 接缝
    diag:     ScriptDiagnostics,        // 预算/hitch/trace/time-travel
}

/// 一个「逻辑脚本模块」：name 稳定，digest 随内容变。
pub struct ScriptModule {
    pub name: Interned,        // 逻辑名（热替换时不变）
    pub digest: [u8; 32],      // 内容寻址 SHA-256（缓存键 + 审计 ID）
    pub schema_version: u32,   // 状态 schema 版本（迁移用）
    pub exports: ExportTable,  // 导出入口（type-checked 签名）
}

/// 每次调用的上下文：世界代理 + 预算 + 确定性种子。
pub struct ScriptContext<'w> {
    pub world: WorldProxy<'w>,    // 受控世界访问（§7）
    pub budget: CallBudget,       // 燃料/时限/内存页（§12）
    pub rng: DeterministicRng,    // 确定性随机（§15）
    pub frame: FrameInfo,         // tick / dt / 固定步标记
}

/// 执行后端 trait：WASM（JIT）与 wasmi（解释）各实现一次，便于按平台切换；
/// 同时作为「图脚本执行体在 VS 文档实现同一消费接口」的概念对齐点，但本文只落地 WASM。
pub trait ScriptBackend: Send + Sync {
    fn compile(&self, src: ModuleSource, bindings: &BindingRegistry) -> Result<CompiledModule>;
    fn instantiate(&self, m: &CompiledModule, policy: &Policy) -> Result<Instance>;
    fn call(&self, inst: &Instance, entry: ExportId, ctx: &mut ScriptContext,
            args: &[Value]) -> Result<CallOutcome>;
    fn snapshot_state(&self, inst: &Instance) -> Result<StateBlob>;   // 热替换/迁移用
    fn restore_state(&self, inst: &Instance, blob: &StateBlob) -> Result<()>;
}
```

- **`Interned name` vs `digest`**：热替换 ≡ 把 `name → digest` 的指针翻到新版本；旧 `Instance` 被在途调用持有，自然延命（借 `uwu_wasm` 原子指针心智）。
- **`Handle` 零拷贝句柄**：脚本持有的实体/资产/资源都是不透明 `Handle`（generational id），不是裸指针——安全、可校验、可审计。脚本永远拿不到宿主裸内存。
- **为什么留 `ScriptBackend` trait 而非写死 WASM**：唯一目的是让「同平台 JIT 实现」与「主机/iOS 解释器实现」共用上层内核（§21），**不是**为了未来再塞回四后端。

---

## 6. 宿主 API 绑定层：反射驱动、稳定 ABI、零手写胶水（本文独占）

**核心主张：宿主 API 不手写 FFI 胶水，全部从 `prism_reflect` 函数反射生成。**（建于 reflect §12 + §24.6 脚本桥；这是 reflect 本身没写、脚本独有的一层）

```rust
// 第一方只需在 Rust 侧标注一次：
#[script_api(category = "gameplay", caps = ["spawn", "damage"])]
pub fn apply_damage(world: &mut WorldProxy, target: EntityHandle, amount: f32) { ... }

#[script_api(category = "query", caps = ["read_transform"])]
pub fn get_position(world: &WorldProxy, e: EntityHandle) -> Vec3 { ... }
```

- `#[script_api]` 派生宏（建于 reflect 的 derive 期静态 `TypeInfo` 之上，§24.1）在编译期把函数签名注册进 `BindingRegistry`：名字、参数/返回类型（走 reflect `TypeInfo`）、能力标签、文档字符串、确定性标记。
- **稳定 ABI**：绑定表是「语义 ABI」而非内存布局 ABI。WASM 侧用 Component Model 的 typed import/export（`(params)->results`）映射；**图脚本侧（VS 文档）节点 pin 类型从同一 `TypeInfo` 来**。新增一个 host API，WASM import 与图节点面板同时可见。
- **零拷贝边界（WASM）**：POD 类型（`Vec3`/`Quat`/句柄/定长结构）走 reflect §24.3 的二进制零拷贝布局，经 WASM 线性内存直接读写，避免逐字段 lower/lift；复杂类型才走 `DynamicStruct`。
- **能力门控**：每个绑定带 `caps`，调用时与模块 `Policy` 白名单比对，不在白名单的 host call 直接拒绝（§12）。
- **文档即提示**：reflect 的文档元数据 + 类型信息直接喂给编辑器补全、REPL 签名提示（图节点面板由 VS 文档消费同一数据）——**易用性来自真相层，不额外维护**。

> 与「每个脚本 API 手写 Lua/WASM 胶水」的传统做法相比，这里是「写一次 Rust 函数 + 一个属性宏」，WASM 与图脚本自动可见。新增一个玩法 API 的边际成本 ≈ 写一个普通 Rust 函数。

---

## 7. ECS 接入模型：世界访问、命令缓冲、查询代理

脚本**不直接**持有 `&mut World`。它拿到的是 `WorldProxy`——一个受策略约束的世界视图：

```rust
pub struct WorldProxy<'w> {
    reads:  QueryView<'w>,     // 只读查询（快照或借用，取决于调度相位）
    cmds:   ScriptCommands,    // 延迟结构性变更（复用 prism_ecs 并行 ECB）
    res:    ResourceView<'w>,  // 受能力门控的资源读写
    events: EventBridge<'w>,   // 受控事件读写
}
```

三种访问相位（与 `prism_ecs` 调度一致）：

1. **只读并行相位（默认，最快）**：脚本在普通 system 里跑，`WorldProxy` 只给只读查询 + 命令缓冲。多个脚本可**并行**执行（§13），结构性变更（spawn/despawn/insert）写进 per-thread ECB，在 sync point 按**确定键排序**合并回放（复用 `prism_ecs` §「Commands / 并行 ECB」）。→ 无数据竞争、确定性、可并行。
2. **独占相位（opt-in，需 `exclusive` 能力）**：脚本声明需要 `&mut World`，调度器把它放进独占阶段串行跑（复用 ECS exclusive system）。用于必须立即可见结构变更的少数场景。
3. **快照只读相位（回放/网络/调试）**：`WorldProxy` 绑定一个 `WorldSnapshot`（`prism_ecs::world::snapshot`），脚本看到的是冻结世界，用于确定性重放、服务器权威校验、time-travel（§15/§16）。

- **查询代理**：脚本声明的查询由宿主在注册期解析为 ECS `Query`，脚本侧只拿迭代句柄，不暴露原型/chunk 内部。
- **变更检测**：复用 ECS tick 变更检测，脚本可订阅「某组件变了」而非每帧全扫（对标 observer/reaction）。
- **背压**：ECB 有上限，脚本一帧内 spawn 超限触发熔断 + 诊断，防止失控脚本撑爆世界。

---

## 8. WASM 执行底座：编译 / 实例化 / 调用（本文独占深化）

这是 reflect/asset/app 都没有、脚本层独占的部分。直接以 `uwu_wasm`（wasmtime 37 + Component Model + WASI p2）为蓝本，并对其默认配置做热路径调优。

### 8.1 全链路
```
ModuleSource(字节/.wasm 或 .cwasm)
   │
   ├─ 内容寻址：SHA-256(digest) ── 命中 CompiledModule 缓存 ─▶ 跳过编译
   │
   ▼ 未命中
compile：
   ├─ JIT 路径（PC/Android）：wasmtime + Cranelift(Speed) 编译 Component
   └─ AOT 路径（cook 期已 precompile）：Component::deserialize(.cwasm)（跳过 Cranelift）
   │
   ▼
InstancePre（每沙箱/每策略缓存，继承 uwu_wasm 的 InstancePre 缓存 + OnceCell 去重）
   │
   ▼
instantiate_pre + call（热路径只做这两步）
```

### 8.2 对 `uwu_wasm` 默认配置的热路径调优
`uwu_wasm` 默认有 4 个热路径固定开销；`backend-wasm` 落地时**默认关闭/旁路**，按需开启：

| 开销项 | `uwu_wasm` 默认 | Prism `backend-wasm` 策略 |
|---|---|---|
| 每调用新建 `Store` + instantiate | 恒定付出 | **开启 pooling allocator**（`InstanceAllocationStrategy::Pooling`，复用线性内存映射） |
| 无 pooling allocator | on-demand mmap/munmap | **预留实例池**（走 `prism_platform::vm` 页级 reserve/commit），按 `platform_tier` 配池大小 |
| 无条件 `format!("{args:?}")` + HMAC 回执 | 每调用付出 | **attestation 默认关闭**，仅 Fortified 审计档开启；fast path 不格式化不签名 |
| fuel + epoch + attestation 全开 | 每调用付出 | **可信（第一方签名）脚本关 fuel**（只留 epoch deadline 兜底）；UGC 才开 fuel |

### 8.3 AOT 预编译（editor cook → `.cwasm`）
这是对「editor 打包时把 WASM 预编译、pkg 里直接带 AOT 产物」诉求的落地，也是对其**边界的诚实处理**：

- **cook 期**：editor 对每个 `.wasm` 按**目标平台**调用 `Engine::precompile_module` / `Component::serialize`，产出内容寻址 `.cwasm`，缓存键 = `digest + wasmtime 版本 + target triple`（三者任一变即失效）。
- **运行期**：引擎 `Component::deserialize(.cwasm)`，**跳过 Cranelift**，冷启动与首帧墙显著降低。
- **可移植真相源**：pkg 同时保留可移植 `.wasm` 作为真相源；`.cwasm` 不跨 wasmtime 版本/CPU target 移植，引擎升级时按平台重 cook。
- **诚实边界（关键）**：`.cwasm` = **原生机器码**。
  - **PC / Android**：`deserialize` 并执行机器码是允许的，所以 AOT 既能加速打包脚本的冷启动，**也能当运行时热更载体**（新 `.cwasm` deserialize + 原子指针翻转）。
  - **主机 / iOS**：运行时下发并执行机器码 = 等同下发 native dll = **认证禁止**（wasmtime deserialize 走自己的 mmap+PROT_EXEC，不走平台签名路径；iOS 无 JIT entitlement）。因此 `.cwasm` 在主机/iOS **只能随签名安装包或官方认证补丁走**，**不能当运行时热更载体**；真·运行时热更任意/UGC 代码必须退回解释器（§21）。

### 8.4 其他底座要点
- **零拷贝 ABI**：POD 走线性内存直读（§6），避免逐字段 lower/lift。
- **batch 调用**：对「同一脚本、多实体」用 `call_many`，把 digest 查表/锁/策略常量摊薄到整批（继承 `uwu_wasm::call_typed_many`），一个 `spawn_blocking` 跑一批。
- **编译在 off-thread**：编译/反序列化走 `prism_tasks` 的 `spawn_blocking` 车道，不占帧 worker。

---

## 9. 热更新机制：内容寻址 + 原子替换 + 零中断（复用 app/asset + AOT 缓存）

脚本热更内核（借 `uwu_wasm::HotSwap` + Erlang/OTP 双版本并存）。**源变更监视、容器叠挂、分发全部复用 asset**（§12/§13/§23），本文只做「字节就绪之后」的编译 + 替换 + 迁移：

```
源变更 (asset mtime §12 / 编辑器保存 / asset 高优先级容器叠挂下发 §23.5)
   │
   ▼
Loader 读字节 → SHA-256 digest ──(digest 未变)──▶ 跳过
   │ (digest 变了)
   ▼
off-thread 编译 / .cwasm 反序列化 (prism_tasks spawn_blocking)   ← 不卡主线程/帧
   │ 编译失败 ──▶ 保留旧版本 + 诊断，绝不中断在途
   ▼
编译成功 → 装入 CompiledModule 缓存 (content-addressed)
   │
   ▼
【灰度门】按策略：直接切 / 金丝雀按权重 / 影子双跑比对
   │
   ▼
状态迁移 (StateMigrator §10)：旧实例 state → 新 schema
   │
   ▼
原子指针翻转：name → new digest
   │   旧 Instance 被在途调用持有，最后一个调用结束后释放（零中断）
   ▼
自愈监视：新版本错误率 > 阈值 → pin_version 回滚到上一稳定 digest
```

要点：
- **内容寻址去重**：相同字节只编译一次（跨模块/跨租户），digest 既是缓存键也是审计 ID（借 `uwu_wasm`）。
- **编译在 off-thread**：主线程只做「翻指针」这一个原子操作，热替换对帧的影响 ≈ 一次原子写（目标 < 一帧不可感知）。
- **编译失败安全**：新版本编译/校验失败 → 保留当前版本，旧调用完全不受影响，错误进诊断面板。
- **灰度 / 影子**：金丝雀按流量权重或哈希分发（借 `uwu_wasm::CanaryRouter`）；影子模式让新旧版本双跑、比对输出，验证热更正确性而不影响玩家。
- **自愈回滚**：错误率/超预算率超阈值自动 `pin_version` 回退（借 `SelfHealing`），并通知作者。
- **AOT 缓存段**：命中 `.cwasm`（§8.3）时「编译」退化为反序列化，PC/Android 热替换更快；主机/iOS 的 `.cwasm` 不走运行时下发（§21）。
- **第一方原生码热重载不在此**：改引擎/玩法 crate 的热重载是 `prism_app` §24.2 的 dylib 机制（卸旧符号→载新→重接 Schedule、state 外置），**与脚本热更是两条独立路径**，本文不重写。

---

## 10. 状态迁移与版本化：复用 reflect schema，改脚本不丢状态

热更新最难的不是「换代码」，是「换代码时保住运行时状态」。本设计把它建在 **`prism_reflect` schema 版本化迁移** 之上，**不自造迁移框架**：

- 每个脚本模块声明 `schema_version`。脚本的持久状态是一个 **`Reflect` 结构**（而非 WASM 实例私有内存），因此可被 reflect 序列化/反序列化、diff/patch、版本迁移。
- 热替换时 `StateMigrator`：
  1. `snapshot_state(old_inst)` → `StateBlob`（reflect 序列化）。
  2. 若 `old.schema_version == new.schema_version`：直接 `restore_state`。
  3. 若版本不同：走 reflect **版本化迁移链**（字段增删改名有迁移规则），或调用脚本显式导出的 `migrate(old) -> new` 回调（Erlang/OTP `code_change` 心智）。
  4. 迁移失败 → 保留旧版本 + 诊断（不丢档、不崩）。
- **ECS 组件热替换**：脚本定义的运行时组件（经 reflect 动态类型）版本变更时，复用 reflect `apply`/部分 patch 做字段级迁移，不整表重建（对标 Unity 域重载后的字段保留，但更细粒度）。
- **不可迁移的东西**：原生裸内存、文件句柄、网络连接——这些不进脚本持久 state，由宿主侧资源管理，脚本只持 `Handle`，热替换后句柄仍有效。

> 这解决了「`uwu_wasm` 每次调用新建 Store、调用间无持久状态」的空白：状态持久化在**宿主侧的 reflect state 块**，而非 WASM 实例内，所以热替换与无状态执行可以共存。

---

## 11. 性能设计（速度）

性能是硬指标，分三条战线：**单次调用开销**、**吞吐/并行**、**热路径隔离**。

### 11.1 单次调用开销（WASM 后端，承接 §8 调优）
开启 §8.2 调优（pooling allocator / `InstancePre` 缓存 / 旁路 attestation / 可信脚本关 fuel）后：
- 纯计算脚本 per-call 可压到微秒级（取决于脚本体量），相对手写 Rust 的开销主要来自实例化与边界封送，可控。
- `.cwasm` AOT（§8.3）在 PC/Android 省掉运行期 Cranelift，冷启动/首帧墙显著降低。

> **口径声明**：本节 pooling allocator / `InstancePre` / 微秒级数字均为 **PC/服务器 JIT 口径**（wasmtime + Cranelift）。主机/iOS 受认证政策走解释器时，纯计算 per-call 要**重标一个量级**（慢 10–50×），执行形态与选型见 §21。

### 11.2 吞吐 / 并行
- 只读相位脚本在 `prism_tasks` work-stealing 池上**并行**跑（§13），N 个实体的同脚本调用走 `parallel_for` / `call_many`。
- off-thread 编译/热替换不占帧预算。
- NUMA/混合核感知由 `prism_tasks` 承载，脚本池可绑特定线程类。

### 11.3 热路径隔离（分层铁律的性能兑现）
- 脚本只在 Layer 3/4 的 gameplay/服务相位跑，**物理 solver / 渲染 RG / ECS 存储内循环永不进脚本**。
- 反射只在边界用（绑定封送），内层仿真走原生字段（与 reflect §「边界使用原则」一致）。
- 每脚本每帧有**预算**（§16）：超预算当帧熔断 + 下帧降频/告警，防止一个坏脚本拖垮帧率。
- **逐帧性能热点怎么办**：WASM 不够时，热点逻辑应下沉为**第一方原生 system**（随签名包发货，走 app §24.2 的开发期热重载迭代），而非让脚本进内核热路径——这与 UE「Blueprint 字节码 + nativized 热点」同构。

---

## 12. 安全与沙箱（零信任：三档能力门控）

沙箱档位随**签名信任根**走（§20.4 验签决定信任档），不再随「后端类型」走：

| 档 | 适用 | 隔离手段 |
|---|---|---|
| **Trusted（第一方签名 WASM）** | 自家代码 | 可关 fuel（留 epoch 兜底）、可授更多能力、可走 AOT/JIT；仍是 WASM 内存沙箱 |
| **Sandboxed（WASM 默认，第三方/玩家 mod/UGC）** | UGC | 能力白名单 + 燃料上限 + 时限(epoch) + 线性内存页上限 + 表元素上限 + 无歧义 host 边界 |
| **Fortified（不可信 UGC + 审计）** | 公开 UGC 市场 | 在 Sandboxed 基础上开 attestation 执行回执（模块摘要 + 入/出参摘要 + 资源用量，可审计/可申诉）、内容白名单（digest 允许列表） |

- **能力白名单（Capability）**：借 `prism_platform::capability` + 绑定 `caps` 标签。脚本只能调用其策略允许的 host 函数；文件/网络/进程能力默认全关，UGC 脚本拿不到。
- **资源上限**：fuel（指令预算）、deadline（epoch 中断，防死循环）、memory_pages、table_elements、输出字节上限、ECB 条目上限。超限 → trap → 当帧该脚本失效 + 诊断，不影响他者。
- **确定性边界即安全边界**：UGC 脚本强制走确定性档（§15），禁用 wall-clock/真随机/非确定 host 调用——既保回放一致，也堵了侧信道。
- **trap 收敛**：WASM trap 统一收敛为 `ScriptError`，带模块名 + digest + 入口 + 源码位置。
- **与反作弊接缝**：权威服务器对脚本产生的「提议」走 `prism_anticheat` 的快路径硬拒绝 + 深路径对账（§15），脚本本身不被信任为安全根。

> ⚠️ **安全提示**：第一方原生 dylib（app §24.2）**无内存沙箱**，等同第一方信任代码。任何接受第三方/玩家代码的场景**必须**走 WASM Sandboxed/Fortified 档——这是硬性约束。Loader 层**拒绝**把来自 UGC/网络下发源的字节当原生码加载。

---

## 13. 调度与执行模型（并行 / 帧预算 / 主线程亲和）

脚本调度完全架在 `prism_tasks` 上，不自建线程池：

- **三类执行车道**：
  1. **并行车道**（默认）：只读相位脚本在 work-stealing 池并行跑，`parallel_for` 覆盖「同脚本 × 多实体」。
  2. **主线程亲和车道**：声明触碰非 Send 资源的脚本走主线程类，串行但不跨线程（复用主线程亲和）。
  3. **阻塞车道**：编译 / 大状态迁移 / 磁盘加载走 `spawn_blocking`，不占帧 worker。
- **帧预算调度**（复用 `prism_tasks` 帧预算 + `prism_diagnostic::budget`）：每类脚本有 per-frame 时间预算，超预算的低优先级脚本**让渡到下一帧**（结构化、可取消），保主循环节奏。
- **结构化并发与取消**：热替换/场景切换时，用取消令牌优雅中止在途脚本任务，不泄漏、不半死状态。
- **确定性调度档**（opt-in）：固定步 + 有序执行（按确定键排序），服务 lockstep/回放（§15），默认走高性能非确定路径。

---

## 14. 效果层：脚本能驱动什么

脚本的「效果」= 它能通过绑定表触达的玩法/表现系统。按 Gameplay 文档的分层，脚本只调 Layer 3/4 稳定 API，但覆盖面很广：

- **Gameplay 逻辑**：GAS 能力/效果触发、GameplayTag 操作、游戏流程状态推进、Subsystem 服务调用、消息总线收发（gameplay §9/§8/§11/§12）。
- **实体编排**：spawn/despawn prefab、组件读写、关系操作（经 WorldProxy + ECB）。
- **序列与编排**：驱动 Timeline/Playable（gameplay §15），技能连招、过场脚本。
- **AI**：行为树节点 / 黑板读写 / 自定义决策（gameplay §14）。
- **表现接缝（受控、不进渲染热路径）**：触发 VFX、音频事件、运镜、UI 数据绑定——脚本发「意图事件」，由原生系统执行表现，脚本不碰 GPU/音频 DSP 热路径。
- **数据驱动**：读写 GameDataAsset / DataTable（gameplay §13），配合热重载做「改数据/改逻辑都不重启」的完整迭代闭环。
- **Game Features 运行时插拔**（gameplay §26 旗舰）：脚本模块可作为一个 Game Feature 的逻辑载体，随 Feature 热插拔——`prism_gc_ugc` 的 Mod 包正是这样装配（game_creator §26）。

> 效果的边界由**能力白名单**精确裁剪：UGC 脚本可能只拿到「生成受限 prefab / 改分 / 放置实体」，拿不到文件/网络/反射任意写（game_creator §20 最小权限）。

---

## 15. 确定性、网络与反作弊接缝

- **确定性档**（feature `determinism`，复用 `prism_app::determinism` + `prism_time::determinism`）：
  - 固定步调度、有序执行、定点数值可选、确定性 RNG（`ScriptContext.rng` 以 tick+entity 派生种子）。
  - 禁用非确定 host 调用（wall-clock、真随机、线程时序依赖），WASM 侧天然可裁剪这些 WASI 能力。
  - 相同输入 → 相同输出，服务 lockstep 联机、录像回放、服务器权威重算。
- **网络（接缝归 `prism_network`）**：
  - 脚本产生的状态变更按组件级复制；客户端预测 + 回滚复用 `prism_ecs::world::snapshot`——回滚时脚本在**快照只读相位**（§7）重放。
  - 热更新下发：服务器可把新脚本 digest 下发客户端，内容寻址保证一致性，灰度/回滚统一走 §9。
- **反作弊（接缝归 `prism_anticheat`）**：
  - 客户端脚本一律**不是安全根**。权威服务器对脚本「提议」走快路径硬拒绝（运动学/可达性/资源守恒）+ 深路径确定性重放对账。
  - UGC 脚本的执行回执（§12 Fortified）可喂给反作弊的 append-only 审计流做交叉验证。

---

## 16. 可观测性与调试（时间旅行 / 热路径预算）

调试体验是「改完即见效、错了说得清」的后半句，复用 `prism_diagnostic`：

- **预算与 hitch 归因**：每脚本每帧耗时/燃料/分配计入 `prism_diagnostic::budget`；超预算或造成 hitch 的脚本被 `hitch` 归因点名（模块名 + digest + 入口）。
- **Profiler 集成**：脚本调用进 `profiler` span 树，与原生系统同一张火焰图，一眼看出脚本热点。
- **时间旅行调试**（借 `uwu_wasm` time-travel + `prism_diagnostic::replay`）：快照/倒带/差分/重放脚本执行；配合 WorldSnapshot 做「这一帧这个脚本为什么这么做」的回溯。
- **Live 调参**（编辑器内，L5）：对所有脚本提供「不改源、调常量/参数即时生效」的 live tuning（对标 UE 控制台）。图脚本的交互式调试在 VS 文档。
- **错误诊断**：`ScriptError` 带源码位置（source map：WASM DWARF）、调用栈、入参摘要、修复提示。编译错误直接定位到源行。
- **热重载 HUD**：显示每个模块的当前 digest、版本、灰度状态、错误率、上次热替换时间——作者对「现在跑的是哪版」一目了然。

---

## 17. 易用性分层与默认体验

易用性是可量化目标，分两类用户（图脚本的策划体验在 VS 文档）：

**第一方工程师（Rust / host API）**
- 一个属性宏暴露 API：`#[script_api]`，WASM import 与图节点面板自动可见（§6）。
- 写 mod/逻辑：Rust 编到 WASM Component，`host.watch` 即热替换，状态保留（§10）。目标：改一行逻辑到看到效果 < 数秒（PC）。
- 性能热点：下沉为第一方原生 system（随包，走 app §24.2 迭代），不进脚本热路径。

**玩家 / UGC 作者（WASM）**
- 一行接入宿主：`ScriptHost::spawn_module(name, source)`，默认 Sandboxed 策略（安全默认）。
- 跨语言：任何能编到 WASM Component 的语言（Rust/C/C++/AssemblyScript/…）都能写 mod。
- 错误友好：trap/编译错误带位置与提示，不会把游戏搞崩。

**默认体验（Convention over Configuration）**
- `ScriptPlugin::default()` 一行注册到 `prism_app`：自动接 asset 监视（§12）、默认 Sandboxed 策略、默认调度车道。
- 发行版默认**零脚本税**：不启用 `backend-wasm` 时整个 `prism_script` 编译期移除，无运行时开销。
- 按 `platform_tier`（prism_app）自动选执行形态：PC/Android JIT、主机/iOS 走随包 AOT 或解释器（§21），作者无感。

---

## 18. 公共 API 草案

```rust
// —— 一行接入 ——
app.add_plugin(ScriptPlugin::default());   // 自动接 asset 监视 + 默认 Sandboxed 策略 + 调度

// —— 注册宿主 API（第一方，一次标注，WASM + 图脚本可见）——
#[script_api(category = "gameplay", caps = ["spawn", "damage"])]
pub fn apply_damage(w: &mut WorldProxy, target: EntityHandle, amount: f32) { /* ... */ }

// —— 加载/热更一个 WASM 模块 ——
let m = host.load_module(ModuleSpec {
    name: "enemy_ai",
    source: Source::File("mods/enemy_ai.wasm".into()),  // 或 .cwasm（AOT）/ 内存 / 网络下发
    policy: Policy::sandboxed()                           // 安全默认
        .fuel(2_000_000)
        .deadline(Duration::from_millis(2))
        .memory_pages(64)
        .allow_caps(["spawn", "read_transform"]),
    schema_version: 3,
})?;

host.watch(&m);   // asset mtime/下发自动热替换（内容寻址 + 状态迁移 + 灰度）

// —— 批量调用（同脚本 × 多实体，摊薄常量开销）——
host.call_many("enemy_ai", "tick", &entities, &mut ctx)?;

// —— 灰度 / 回滚 ——
host.canary("enemy_ai", /*new*/ digest_v4, /*weight*/ 10);  // 10% 流量试新版
host.pin("enemy_ai", /*stable*/ digest_v3);                 // 自愈或手动回退

// —— 诊断 ——
let report = host.diagnostics("enemy_ai");  // 当前 digest / 版本 / 错误率 / per-frame 预算 / hitch
```

设计意图：**单载体（WASM）、安全默认（sandboxed）、热更内建（watch 即可）、性能可调（policy + call_many + AOT）、灰度回滚一等公民**。

---

## 19. Crate 拆分与落地形态

```
pkg/
├── prism_script            # L3 内核：ScriptHost / HotSwap / StateMigrator / Scheduler 接缝
│                           #   核心 no_std+alloc 友好；WASM 后端经 feature 门控
├── prism_script_macros     # #[script_api] 派生宏（建于 prism_reflect derive 之上）
├── prism_script_bind       # L2：BindingRegistry / WorldProxy / Policy（反射驱动，图脚本共享）
└── prism_script_wasm       # L4：WasmBackend（wasmtime Component Model + WASI p2，借 uwu_wasm）
                            #   feature backend-wasm-interp 时内含 wasmi 解释实现（主机/iOS）
```

- feature 门控：`backend-wasm` / `backend-wasm-interp`（主机/iOS 解释器）/ `hot-reload` / `sandbox` / `determinism` / `trace`。
- **没有 `prism_script_graph`**：图脚本整套在 `prism_visual_scripting` 的 crate 里，只经 `prism_script_bind` 的 `BindingRegistry` 与本文交点。
- **没有 `prism_script_rhai`**：已删除。
- 第一方原生码热重载在 `prism_app`（§24.2），不在本套 crate。
- `prism_script` 不依赖任何 `bevy_*` crate；WASM 后端依赖 `wasmtime`，仅在 `backend-wasm` 下编入；WASM 线性内存池经 `prism_platform::vm`。
- **脚本包分发**（§20）复用 `prism_asset` 的容器 `.ucas` / `MountTable` / 补丁叠挂 / CDN，不自造格式；`prism_script` 只新增 code chunk 的验签 / 实例化与包清单校验。

---

## 20. 分发与打包：Prism 脚本包（复用 asset 容器 + code chunk 独占）

**铁律：复用资产容器，不自造格式。** 容器、挂载、优先级叠挂、补丁/DLC、CDN、完整性校验**全在 `prism_asset`**（§13/§23/§24）。本文只做它没有的两件代码专属事：**code chunk 的验签 + 实例化**、**包清单（manifest/签名/caps/WIT）校验**。

### 20.1 脚本包 = 一个 `.ucas` 容器（多两类块）
```
Prism 脚本包（一个 .ucas 容器，asset §13）
├─ manifest           id / version / schema_version / 目标平台 / caps 声明 /
│                      WIT 接口版本 / 签名 + 信任根指纹
├─ bindings.wit       接口契约（宿主 API 的 WIT，版本协商用）
├─ code/*             code chunk（新 chunk kind）：WASM 模块 和/或 cook 期 .cwasm
└─ content/*          普通资产 chunk（复用 prism_asset，纯脚本包可空）
```
- **纯脚本包**：只含 `code/` + 清单（补丁逻辑、玩法规则热更、纯逻辑 mod）。
- **混合包**：`code/` + `content/`（对标 Game Feature Plugin——「一张新关卡 + 它的玩法代码 + 它的资产」整体插拔）。
- 两种形态在 asset 里只是「容器里 chunk 组成不同」，挂载/卸载/覆盖/热重载**走同一条路径**——复用容器的直接红利。

### 20.2 信任：内容哈希 ≠ 代码签名（本文独占）
asset 的内容哈希（§24）只回答「字节完整吗」（防损坏/篡改）。但「下发即执行」的**代码**需要高一个量级信任：**谁签的、能不能跑**。脚本包因此叠加**密码学签名 + 信任根**（这是 asset 没有、脚本独占的一层）：
- **信任根分级**：第一方根（引擎/发行商）、UGC 市场根（平台审核过）、本地开发根（dev 自签，仅开发构建接受）。
- **挂载即验签**：验签结果决定该包 code chunk 的**信任档**（映射 §12 Trusted/Sandboxed/Fortified）——第一方根 → 可授 Trusted；UGC 市场根 → Sandboxed/Fortified；**未签名 / 验签失败 → 拒绝执行**，或策略允许时强制降 Sandboxed 并剥夺敏感能力。

### 20.3 code chunk 交接：资产运字节，脚本管执行（本文独占）
- **`prism_asset`**：只负责把 code chunk 字节**运到**（定位/解压/完整性/挂载覆盖），不理解它是可执行代码。
- **`prism_script`**：拿到字节后做**验签 → 编译/反序列化（§8，按 §21 平台形态）→ 注册进 `ScriptHost`**。
- 二者经 `CodeChunkLoader` 交接：asset 发「code chunk 就绪」事件，script 订阅并走 §9 热更内核装入——与普通资产 loader 协议同构，不破坏 asset 分层。

### 20.4 WIT 版本协商 + 能力校验
挂载时宿主做两项校验，任一不过则拒绝（或走迁移）：
1. **能力校验**：包清单 `caps` ⊆ 宿主对该信任根授予的能力集（§12）。
2. **WIT 版本协商**：包针对 `bindings.wit` 某版本编译，宿主当前 WIT 按 semver 比对——兼容直接挂、minor 落后可挂、major 不兼容则拒绝或走 §10 迁移链。

### 20.5 挂载生命周期（五相）
```
Mount    ── 经 prism_asset MountTable 挂容器（按优先级，asset §13）
Verify   ── 完整性(asset §24) + 代码签名(§20.2) + WIT 协商(§20.4) + caps 校验(§12)
Register ── code chunk 导出注册进 ScriptHost；content chunk 进资产 catalog
Activate ── 进调度车道(§13) / 触发 observer / Game Feature 挂接(gameplay §26)
   … 运行中：热更(§9) / 补丁覆盖(asset §23.5) / 状态迁移(§10) / 自愈回滚
Deactivate ── 退出调度、停触发，保留已注册（可快速重启）
Unmount  ── MountTable 卸容器；在途调用自然延命后释放（§9 原子指针心智）
```
分层映射：挂载/容器/补丁/CDN → `prism_asset`；验签/导出注册/调度 → `prism_script`；Game Feature 插拔 → `prism_gameplay` §26；WIT/状态迁移 → `prism_reflect`。

### 20.6 热更 / 补丁 / 回滚（复用已有机制）
- **热更**：改一个包 → 新容器高优先级叠挂（asset §23.5）→ code chunk digest 变 → 走 §9 off-thread 编译 + 原子替换 + §10 状态迁移。
- **补丁**：只发增量容器，不重发全量；混合包的 content 补丁与 code 补丁同机制。
- **整包回滚**：unmount 补丁容器即回到旧版本；叠加 §9 `pin_version` 做逻辑级回退——包级 + 版本级两级回滚互补。

---

## 21. 平台执行矩阵：解释 / JIT / AOT × PC / 主机 / 移动 × 认证铁律

> ⚠️ **make-or-break 铁律**：能被**下发并执行**的代码，在主机（PS/Xbox/Switch）与 iOS 上受**代码签名/认证**政策约束。这不是技术选择，是平台政策——设计必须正面处理，否则「WASM 热更」在主机/iOS 上根本上不了架。

### 21.1 WASM 的三种执行形态
| 形态 | 实现 | 机器码何时生成 | 速度 | 可运行时热更 | 主机/iOS 认证 |
|---|---|---|---|---|---|
| **解释** | wasmi / wasm3 | 从不（纯解释字节码） | 慢（约 native 的 1/10–1/50） | ✅ | ✅ 过（无动态机器码） |
| **JIT** | wasmtime + Cranelift | 运行期动态生成 | 快（接近 native） | ✅ | ❌ 禁（W^X / 禁 JIT 政策） |
| **AOT** | wasmtime `.cwasm`（§8.3） | cook / 打包期 | 快（≈ native） | ⚠️ 仅 PC/Android | ⚠️ 仅随签名包 / 官方认证补丁 |

关键：**AOT 产物（`.cwasm`）就是机器码**。PC/Android 上「运行时 deserialize 新 `.cwasm` + 原子翻转」是合法热更；但主机/iOS 上「运行时下发 `.cwasm` 并执行」等同「下发 native dll」，同样被认证禁止——AOT 在主机/iOS 只能随安装包或官方认证补丁走，**不能当运行时热更载体**。

### 21.2 平台 × 形态选型
| 平台 | 默认热路径形态 | 运行时热更形态 | 说明 |
|---|---|---|---|
| **PC / 服务器** | JIT（§11.1 口径）或随包 AOT | JIT / 新 `.cwasm` deserialize | 最舒服：又快又能运行时热更 |
| **Android** | JIT（无强制 W^X）或 AOT | JIT / AOT | 接近 PC |
| **主机 / iOS** | cook 期 **AOT**（随认证包，快） | **解释**（wasmi，慢但可热更） | 鱼与熊掌不可兼得：要快则随认证包、要热则解释 |

### 21.3 与 §11 性能口径的关系（诚实修正）
- §11.1 的 pooling allocator / `InstancePre` / 微秒级数字，**全部是 PC/服务器 JIT 口径**。
- 主机/iOS 走解释器时纯计算 per-call 慢 10–50×。因此在主机/iOS 上：
  - **逐帧热路径逻辑**优先用**第一方原生 system（随认证包，app §24.2 迭代）**——它是「随包的原生代码」，不受下发限制；
  - **WASM 留给**：非热路径逻辑、工具/配置脚本、以及**必须运行时热更的 UGC/mod**（接受解释器的慢，换「主机上也能下发第三方代码且不崩引擎」）。

### 21.4 对标 UE：这正是 Blueprint 的智慧
UE 的做法与此同构：**Blueprint 字节码到处用解释执行**（跨平台一致、随 pak 下发、可热改）；**性能热点烘成 nativized/C++**（随包编译、不热更）。Prism 的映射：**WASM 解释 ≈ Blueprint 字节码**（可下发、可热更、慢）；**AOT / 第一方原生 ≈ nativized**（随包、快、不热更）。两条路都在，按平台与是否热路径自动选。

### 21.5 作者无感 + 诚实边界
- **作者无感**：包清单声明目标平台；cook/打包期按平台产出对应执行形态（PC 带可 JIT 的 `.wasm` + 可选 `.cwasm`、主机带 AOT 或解释模块），运行时按 `platform_tier`（prism_app）自动选。作者写一次，平台差异由管线吸收。
- **诚实边界**：**主机上「下发第三方代码并原生速度执行」在当前认证体系下不可能**——要么解释器（慢）、要么随官方认证补丁（非真·运行时热更）。这是平台政策的硬墙，任何引擎（含 UE/Unity）都绕不过；本设计选择「主机 UGC 走解释器、第一方热点随认证包」把墙的代价**显式化**，而非假装没有墙。

---

## 22. 路线图（M0–M5）

- **M0 内核骨架**：`ScriptHost` + `BindingRegistry`（反射驱动）+ `#[script_api]` 宏 + `WorldProxy`（只读相位 + ECB）。**先接 WASM 最小闭环**（wasmtime Component Model，load/instantiate/call 跑通）。→ 可编译可测的最小闭环 + 微基准。
- **M1 WASM 底座调优**：移植 `uwu_wasm` 的内容寻址 + `InstancePre` 缓存 + 策略；**默认开 pooling allocator、关 attestation、可信脚本旁路 fuel**（§8.2）；接 `prism_platform::vm` 实例池。
- **M2 热更新内核**：HotSwap（内容寻址 + off-thread 编译 + 原子指针）+ asset mtime 监视接入（§12）+ 编译失败安全 + 灰度/自愈回滚。
- **M3 状态迁移 + AOT**：`StateMigrator`（reflect schema 版本化）+ 组件级字段迁移 + `migrate` 回调；cook 期 `.cwasm` 预编译 + 内容寻址缓存（§8.3）。
- **M4 调度 / 性能 / 安全**：`prism_tasks` 并行车道 + 主线程亲和 + 帧预算熔断 + `call_many` + 零拷贝 ABI；能力白名单 + 资源上限 + 三档沙箱 + 确定性档。基准对齐 §11。
- **M5 分发 / 平台 / 工具**：脚本包验签 + code chunk 交接（§20）+ WIT 协商；主机/iOS wasmi 解释器后端 + 平台形态自动选（§21）；live 调参 + time-travel + 热重载 HUD（L5）。

---

## 23. 关键扩展点、风险与非目标

**扩展点**
- `ScriptBackend` trait：JIT 与解释器实现共用上层内核（§5）；**不用于再塞回多语言后端**。
- `Loader` trait：File/Memory/OCI/网络下发可插拔（借 `uwu_wasm::ChainLoader`）。
- `Policy` / `Capability`：可定义新能力标签与资源维度。
- `StateMigrator` 迁移链：schema 迁移规则可注册（建于 reflect）。

**风险与缓解**
- *WASM 实例化成本*：靠 pooling allocator + 实例池 + `InstancePre` 缓存 + AOT 压制（§8/§11）；仍不达标则该逻辑下沉第一方原生。
- *状态迁移复杂度*：schema 剧变时提供显式 `migrate` 回调兜底；迁移失败保留旧版本不丢档。
- *平台认证铁律*：主机/iOS 禁 JIT，WASM 运行时热更只能解释器（慢）或随认证包 AOT（不热）；缓解：主机逐帧热路径走第一方原生，WASM 承载可热更的非热路径/UGC（§21）。
- *AOT 可移植性*：`.cwasm` 不跨 wasmtime 版本/CPU target；缓解：保留 `.wasm` 真相源 + 按平台 cook + 引擎升级缓存失效（§8.3）。
- *脚本包信任*：内容哈希只防损坏/篡改，不等于代码可信；下发代码强制密码学签名 + 信任根验签决定信任档，未签/验签失败拒执行或强制 Sandboxed（§20.2）。

**非目标**（与引擎分层一致）
- 不自造脚本语言、不自造 JIT（WASM 用 wasmtime）；**不并列多后端**（运行时热更只 WASM 一种载体）。
- **不重写图脚本**（归 `prism_visual_scripting`）、**不做第一方原生码热重载**（归 `prism_app` §24.2）、**不自造打包/分发格式**（复用 `prism_asset`）。
- 不让脚本进内核热路径（物理/渲染/ECS 存储内循环）。
- 不承担分布式调度 / 跨节点复制（归上层）；不把客户端脚本当安全根（归 `prism_anticheat`）。

---

> 一句话总结：**`prism_script` 把「运行时可编程性」收敛为「一张反射驱动的绑定表 + 唯一的 WASM 执行底座 + 内容寻址热替换 + reflect 状态迁移 + 预算化沙箱调度」。** 图脚本归 VS 文档、原生码热重载归 app、打包分发归 asset——本文只做它们都没有的脚本运行时层，并用平台执行矩阵把「主机/iOS 认证墙」的代价显式化。这正是次世代 AAA 对「热更新脚本系统」既强大又诚实的要求。
