# Prism Tasks 顶级次世代 AAA 级作业系统 / 并行运行时设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **work-stealing 线程池 + fiber 作业图 + 结构化并行 + 异步执行器 + 确定性调度** 内核设计。它是 `bevy_tasks` 的自研替代，是 `prism_ecs`（system 内并行 / fiber 作业图）与 `prism_app`（子应用流水线）共同的并行地基。
> 借形态不抄码。借鉴：
> - **fiber 作业图**：Naughty Dog《Parallelizing the Naughty Dog Engine Using Fibers》（Gyrling）、DOOM 2016 作业系统（原子计数依赖 + 工作窃取 + 近零同步点 + fiber 等待不阻塞 worker）
> - **work-stealing**：Cilk（理论）、Chase-Lev 无锁双端队列、Rust rayon（join/scope/parallel_for）
> - **安全并行作业**：Unity C# Job System + Burst（IJob/IJobParallelFor、依赖句柄、安全系统）
> - **任务图/命名线程**：Unreal TaskGraph（具名线程 + 任务依赖 + 后继）、Intel TBB（task_group / parallel_for / flow graph）
> - **M:N 调度**：Go runtime（goroutine + GMP 工作窃取）、async-executor
> - **异步**：Rust `futures` / async executor（与作业图共存，统一一个线程池）
> 本文为纯经典并发/调度路线，**不含任何 AI/ML 内容**。

- 版本: v0.2（核心 M0–M6 已落地并验证；§24 高级增补仍为设计阶段；v0.1→v0.2 新增第 24 章「AAA 高级功能增补」：优先级/QoS 车道/帧预算调度/主线程亲和与线程类分离/结构化并发与取消/并行原语(parallel_for/reduce/scan)/异步 I/O 桥/NUMA 与混合核拓扑感知/确定性并行/背压与死锁预防；其中 24.1/24.2/24.3/24.4/24.7/24.8 已落地，余为 PLANNED）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: `prism_math`（可选 SIMD 并行原语）、`prism_platform`（线程/亲和/NUMA/高精度计时）、`prism_diagnostic`（可选 trace）
- 层级定位: ECS 文档 L4「Schedule/Executor/Fiber 作业图」的执行底座；App 文档 L3「运行时服务」
- 明确约束: 无锁数据结构核心可 `no_std + alloc`；**线程池/fiber/执行器需 `std` + 平台线程 API**；`multi_thread` / `fibers` / `async` / `numa` / `determinism` / `trace` 为 feature；**不依赖任何 `bevy_*` crate**

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier）
4. 分层架构
5. 核心模型：Worker / Job / Counter / Fiber
6. 工作窃取调度器（Chase-Lev 双端队列 + 窃取策略）
7. 作业图与依赖（原子计数 + 后继 + fork-join）
8. Fiber：等待不阻塞 worker（stackful 协程切换）
9. 结构化并行 API（scope / join / parallel_for / reduce / scan）
10. 异步执行器（futures 与作业图共存）
11. 命名线程与亲和（main / render / io / async-compute）
12. 内存：每 worker 竞技场 + 作业分配器 + 栈池
13. 确定性调度（固定序 / 可回放）
14. NUMA 与大核小核（亲和 / 就近分配 / 负载均衡）
15. 与 ECS / App 的集成（system 内并行 / 流水线 / 提取）
16. 可观测性（作业 trace / 窃取率 / 占用率火焰图）
17. 高级功能增补
18. 性能工程
19. 易用性与 Bevy 迁移策略
20. crate 分层与模块布局
21. 契约、不变量与版本化
22. 路线图（M0–M6）与基准即规格
23. 诚实边界与风险
24. AAA 高级功能增补（v0.2）

---

## 1. 设计哲学与目标

`prism_tasks` 是整个引擎的「并行心脏」：所有子系统（ECS system 内 chunk 并行、渲染 extract、物理 substep、资产解码、流送 I/O）都往同一个线程池投作业，由统一调度器做工作窃取负载均衡。关键创新点来自 Naughty Dog/DOOM：用 **fiber** 让「等待依赖」的作业把 worker 让给别的作业，而不是阻塞线程——于是**几乎没有空转的核**，也**几乎没有全局同步点**。

**一句话定位**：`prism_tasks` 是 Prism 的「单一线程池 + fiber 作业图 + 原子计数依赖」的并行运行时——细粒度作业、工作窃取、等待即切换、近零同步点、多核近线性扩展；并原生支撑 ECS 的 chunk 子作业、App 的子应用流水线、以及 `determinism` 档的固定序可回放。

四条总目标（按权重）：

1. **性能**：Chase-Lev 无锁窃取、fiber 等待不阻塞核、每 worker 竞技场零分配投递、cache-line 填充防伪共享、近线性多核扩展。
2. **效果（规模能力）**：fiber 作业图支撑数万细粒度作业/帧；统一线程池让 CPU/async-compute/IO 协同不打架；确定性档支撑回滚网络。
3. **易用**：`scope`/`join`/`parallel_for` 结构化 API 像写顺序代码；底层 fiber/窃取对用户透明；高级能力默认关闭、按档位开启。
4. **可移植 + 档位化**：无锁结构核心可 `no_std`；同一调度器从单核移动端（直跑）缩放到 32 核桌面（fiber + NUMA）。

非目标：不做分布式/多机调度；不做抢占式调度（协作式 fiber 切换）；不绑定具体 async 运行时语义（提供最小 executor，兼容 `Future`）。

---

## 2. 参考产品取舍

| 产品 | 吸收 | 规避 |
|---|---|---|
| Naughty Dog Fibers | fiber 作业图、等待切换不阻塞 worker、原子计数依赖、近零同步点 | 平台专有 fiber API（经 `prism_platform` 抽象） |
| DOOM 2016 | 细粒度作业、工作窃取、作业优先级、主线程也当 worker | 主机专有调度器 |
| Cilk / Chase-Lev | 工作窃取理论、无锁双端队列（owner push/pop + 窃取 steal） | —— |
| rayon | join/scope/parallel_for/reduce 结构化 API、递归分治 | 纯 CPU 计算语境、无 fiber/无 async 统一 |
| Unity Job System + Burst | IJob/IJobParallelFor、依赖句柄、作业安全系统（读写冲突检测） | C#/Burst 绑定、API 啰嗦 |
| Unreal TaskGraph | 具名线程（渲染/音频/游戏）、任务后继、任务图 | 与 UObject 耦合、专有 |
| Intel TBB | task_group、parallel_for、flow graph、可组合并行 | C++、重 |
| Go runtime | M:N 调度、GMP 工作窃取、goroutine 轻量 | 抢占式 + GC |
| Rust futures | 最小 async executor、`Future` 兼容 | 过度泛化的运行时 |

综合：**Naughty Dog fiber 作业图 + Chase-Lev 工作窃取 + rayon 结构化 API + Unity 作业安全** 为四支柱，叠加 **统一线程池（CPU/async-compute/IO）+ 命名线程 + 确定性调度** 的 AAA 能力层，全部走 feature/档位门控。

---

## 3. 档位化（capability / quality tier）

| 维度 | 说明 | 示例 |
|---|---|---|
| **capability** | 运行时探测 | 逻辑核数、大核/小核拓扑、NUMA 节点数、是否支持 fiber/用户态上下文切换 |
| **quality tier** | 并行形态档 | single（单核直跑，无线程）/ mobile（少 worker + 省电）/ desktop（满核 + fiber + NUMA） |
| **feature flag** | 编译期裁剪 | `multi_thread` / `fibers` / `async` / `numa` / `determinism` / `trace` |

目标：`single` 档作业直接同步执行（零线程开销，便于调试/确定性）；桌面档开满 worker + fiber + NUMA 亲和。**同一份作业代码在三档行为一致，只是调度方式不同。**

---

## 4. 分层架构

```
L5  集成     ECS system 内 chunk 子作业 / App 子应用流水线 / 子系统并行入口
L4  API      scope / join / parallel_for / reduce / spawn / async block
L3  调度     Scheduler：worker 线程、fiber 池、窃取、优先级、命名线程
L2  图       Job / Counter（原子计数依赖）/ 后继 / fork-join
L1  结构     Chase-Lev 双端队列 / MPMC 注入队 / 每 worker 竞技场 / 栈池
L0  平台     prism_platform：线程、亲和、NUMA、fiber 上下文切换、计时
```

依赖严格向下；L0–L2 的无锁结构可 `no_std + alloc`，L3 的 worker/fiber 需 `std` + 平台线程 API。

---

## 5. 核心模型：Worker / Job / Counter / Fiber

```rust
pub struct Job {
    func: JobFn,              // 作业体（FnOnce-ish，经竞技场分配）
    dependency: *Counter,     // 入口依赖：为 0 才可运行
    completion: *Counter,     // 完成时递减此计数（唤醒后继）
    priority: Priority,       // High/Normal/Low
    affinity: Affinity,       // Any / Named(main|render|io)
}

pub struct Counter(AtomicU32);   // 剩余未完成作业数；归零=依赖满足/wait 返回

pub struct Fiber {               // stackful 协程：承载一个作业的执行上下文
    stack: StackHandle,          // 来自栈池
    context: PlatformContext,    // 寄存器/栈指针快照
}
```

- **Worker**：绑定一个 OS 线程（可绑核），持有一个本地 Chase-Lev 队列；空了就去偷别人的。
- **Job**：细粒度工作单元；入口 `Counter` 归零才可调度，完成时递减它所挂的 `Counter`。
- **Counter**：既是「等待句柄」（`wait(counter)` 阻塞到归零——但靠 fiber 切换不阻塞 worker），也是「依赖门」。
- **Fiber**：作业跑在 fiber 上；作业内 `wait` 时，fiber 连同其栈被挂起放到等待集，worker 立刻捡下一个 fiber 跑别的作业。

---

## 6. 工作窃取调度器

- **本地双端队列（Chase-Lev）**：owner 从**底部** push/pop（LIFO，cache 热、无锁快路径）；小偷从**顶部** steal（FIFO，拿最老的大作业，减少冲突）。
- **窃取策略**：本地空 → 随机选受害者偷 → 连续偷失败则指数退避 + park（省电，避免忙等烧 CPU）；新作业注入时 unpark 睡着的 worker。
- **注入队**：外部（非 worker 线程，如 OS 回调）投作业走 MPMC 注入队，worker 周期性抽取。
- **主线程即 worker**：主线程在 `wait` 根计数时也参与跑作业（DOOM 形态），不浪费主核。
- **优先级**：High 作业优先调度（如本帧渲染提交链）；Low（如预取/日志）让路。

---

## 7. 作业图与依赖

```rust
let counter = Counter::new();
scheduler.spawn_batch(&jobs, &counter);   // 投一批作业，共享完成计数
scheduler.wait(&counter);                 // 等它们全完成（当前 fiber 挂起，不阻塞 worker）

// 链式依赖：B 的入口依赖 = A 的完成计数
let a = scheduler.spawn(job_a);
let b = scheduler.spawn_after(a, job_b);  // a 完成后 b 才可运行（后继唤醒）
```

- **原子计数依赖**：无需锁、无需条件变量；完成即 `fetch_sub`，归零则把后继/等待者重新入队。
- **fork-join**：`spawn_batch` + `wait` 即 fork-join；可嵌套（作业内再 fork），形成作业图。
- **后继（continuation）**：作业完成自动触发后继作业，省去用户手写轮询。
- **作业安全（借 Unity）**：`debug` 档记录每作业声明的读写资源集，运行时检测并发读写冲突并报告（配合 ECS 冲突图，杜绝数据竞争）。

---

## 8. Fiber：等待不阻塞 worker（核心创新）

```
worker 跑 fiber F1 → F1 内 wait(counter)（counter 未归零）
  → 保存 F1 上下文到等待集（挂到 counter 上）
  → worker 不阻塞，从队列捡 fiber F2 继续跑
  → 别处作业使 counter 归零 → F1 被标记可恢复 → 某 worker 捡起 F1 从 wait 处续跑
```

- **栈池**：预分配固定数量 fiber 栈（如 128 个大栈 + 128 个小栈），避免运行时分栈开销；栈大小按作业类型分档。
- **上下文切换**：经 `prism_platform` 抽象（Windows Fibers / `ucontext`/`makecontext` / 内联汇编 swapcontext），纳秒级。
- **收益**：等待依赖的作业不占核；几乎无线程阻塞；全局同步点趋近于零——这是多核近线性扩展的关键。
- **`fibers` 档关闭时**：`wait` 退化为「worker 继续跑队列里其它作业直到 counter 归零」的忙帮工（无栈切换），正确但扩展性略差；`single` 档则同步直跑。

---

## 9. 结构化并行 API（rayon 形态）

对用户隐藏 fiber/窃取，像写顺序代码：

```rust
tasks::scope(|s| {                 // 结构化作用域：退出前保证内部作业全完成
    s.spawn(|| heavy_a());
    s.spawn(|| heavy_b());
});                                // join 点

let (x, y) = tasks::join(|| f(), || g());         // 二分 fork-join

tasks::parallel_for(&mut data, GRAIN, |chunk| {   // 数据并行，自动按粒度切块 + 窃取
    for item in chunk { process(item); }
});

let sum = tasks::reduce(&data, 0, |a, b| a + b);  // 并行归约
let scan = tasks::prefix_sum(&data);              // 并行前缀和
```

- **粒度（grain size）**：按 §3 capability 自适应（太细调度开销大，太粗负载不均）。
- **结构化并发**：`scope` 保证作用域退出前子作业全部完成，杜绝悬垂作业/借用越界（Rust 生命周期在编译期护栏）。
- **递归分治**：`parallel_for` 内部二分 + 窃取，天然负载均衡。

---

## 10. 异步执行器（与作业图共存）

统一一个线程池同时跑「作业图」和「`async` 任务」，避免两套运行时抢核：

```rust
let handle = tasks::spawn_async(async {
    let bytes = asset_io.read("mesh.bin").await;   // IO 等待时 worker 去跑别的
    decode(bytes)
});
let mesh = tasks::block_on(handle);                // 或在作业图里 wait
```

- **`Future` 兼容**：实现最小 executor + waker，把 `async` 任务当作业投到同一池。
- **IO 友好**：`async` 适合 IO 密集（资产加载/流送/网络），作业图适合 CPU 密集（ECS/物理/剔除）；二者共享窃取负载均衡。
- **桥接**：`async` 任务可 `await` 一个作业图 `Counter`；作业可 `wait` 一个 async 结果——两套模型互通。

---

## 11. 命名线程与亲和

有些工作必须在特定线程（Unreal TaskGraph 形态）：

| 命名线程 | 用途 |
|---|---|
| `Main` | 窗口事件、平台回调、必须主线程的 API |
| `Render` | GPU 命令录制/提交（多数后端要求单线程提交） |
| `IO` | 阻塞式文件/网络（不占 CPU worker，避免拖慢计算池） |
| `AsyncCompute` | 低优先级后台计算（流送预处理、烘焙） |

- 作业可声明 `Affinity::Named(Render)`，调度器只在该线程跑它。
- CPU worker 池独立于 IO/Render 线程，互不饥饿。

---

## 12. 内存：每 worker 竞技场 + 作业分配器 + 栈池

- **每 worker 帧竞技场**：作业体/闭包/临时数据从 worker 本地 bump 分配器出，无锁、cache 热；帧末整体 reset（零析构成本）。
- **栈池**：fiber 栈预分配复用（§8），分大/小两档。
- **伪共享防护**：Counter、队列头尾指针等高频原子字段 cache-line（64/128B）填充对齐。
- **零分配快路径**：`spawn`/`parallel_for` 常见路径不走全局堆。

---

## 13. 确定性调度（`determinism` 档）

回滚网络/录制重放要求「同输入 → 同结果」，但工作窃取天然乱序。方案：

- **结果确定 ≠ 执行序确定**：只要作业间无数据竞争（靠 §7 作业安全 + ECS 冲突图保证），乱序执行结果仍一致——这是默认保证。
- **严格确定档**：`determinism` 下关闭窃取随机性，worker 按固定分配策略取作业；归约/parallel_for 用**确定性树形归约**（固定结合顺序），消除浮点加法非结合性导致的抖动。
- **可回放**：记录作业图结构 + 固定分配种子，可逐帧重放调度，复现并发相关 bug。

---

## 14. NUMA 与大核小核

- **NUMA 就近分配**：每 worker 竞技场/栈在其所在 NUMA 节点分配；窃取优先同节点，跨节点窃取加惩罚（减少远程内存访问）。
- **大核/小核（混合架构）**：High 优先级/延迟敏感作业绑大核；Low/后台作业放小核；按 capability 探测拓扑。
- **亲和**：worker 绑核减少迁移抖动与 cache 失效。

---

## 15. 与 ECS / App 的集成

- **ECS system 内并行**（ECS §8.3）：一个重 system 把它的 chunk 批次拆成 `parallel_for` 子作业投到本池；system 间并行由 ECS 冲突图决定，system 内并行由本调度器承载。
- **App 子应用流水线**（App §9）：仿真子应用与渲染子应用在不同 worker 组/错帧重叠，靠作业图 Counter 做跨帧依赖。
- **提取**：extract 作业声明只读 main world + 写 render world，经作业安全校验无冲突。
- **统一池**：ECS、物理 substep、资产解码、流送 IO 全投同一池，由窃取做全局负载均衡，避免「物理线程空转而渲染线程过载」。

---

## 16. 可观测性

- **作业 trace**：每作业 begin/end span（名称/线程/时长/等待时长），导出 chrome-trace / tracy，接 `prism_diagnostic`。
- **窃取率/占用率**：每 worker 的运行/窃取/空闲/park 时间占比，定位负载不均。
- **火焰图**：作业图可视化（依赖链、关键路径），找并行度瓶颈（接 ECS §16.6 系统火焰图、App §16 阶段火焰图）。
- **栈池/竞技场水位**：fiber 栈占用峰值、竞技场高水位，调容量。

---

## 17. 高级功能增补

- **优先级继承**：高优作业等待的 Counter 上挂着的前置作业被提权，避免优先级反转。
- **作业取消**：`CancelToken` 协作式取消（如相机切走后取消该区域流送解码），作业在检查点自愿退出。
- **节流/配额**：IO/AsyncCompute 池设并发配额，防后台作业挤占前台（如流送解码限 2 并发）。
- **周期作业**：注册每帧/每固定步自动投递的作业（配合 App tick 组）。
- **批量投递**：`spawn_batch` 一次投一批共享 Counter，摊薄调度开销（海量粒子/实体 LOD 处理器场景）。
- **panic 传播**：worker 上作业 panic 被捕获并经 Counter/scope 传回等待点，不静默吞掉也不崩整池。
- **自适应粒度**：`parallel_for` 按实测调度开销/负载在线微调 grain，兼顾细粒度负载均衡与低开销。

---

## 18. 性能工程

- **无锁快路径**：本地 push/pop 无 CAS 竞争；仅窃取与注入走原子。
- **fiber 等待不阻塞核**：§8，几乎无线程阻塞，近零同步点。
- **退避 + park**：空闲 worker 指数退避后 park，省电防忙等烧核。
- **cache-line 填充**：防伪共享。
- **每 worker 竞技场**：零锁分配，帧末批量 reset。
- **主线程即 worker**：不浪费主核。
- **近线性扩展**：目标在无依赖密集作业下随核数近线性加速。

诚实边界：窃取率、近线性扩展、fiber 切换开销、NUMA 收益等均需真实多核硬件 + 代表性负载压测，标注 PLANNED；CPU 单测只能证机制正确（依赖满足、无死锁、结果确定）。

---

## 19. 易用性与 Bevy 迁移策略

对外 API 覆盖 `bevy_tasks` 常用面：`TaskPool`/`ComputeTaskPool`/`AsyncComputeTaskPool`/`IoTaskPool`、`scope`、`spawn`、`block_on`、`par_iter`（对应 `parallel_for`）。

迁移路径：
1. 提供 `prism_tasks::prelude` 与 bevy_tasks 近同名导出。
2. ECS 的 `par_iter` / system 内并行底层由本调度器承载（替换 bevy_tasks）。
3. 资产/流送的 `AsyncComputeTaskPool`/`IoTaskPool` 用法原样切换，内部换统一池 + 命名线程。

---

## 20. crate 分层与模块布局

```
pkg/prism_tasks/
  src/
    worker.rs              # Worker 线程、本地队列、park/unpark
    deque.rs               # Chase-Lev 无锁双端队列
    inject.rs              # MPMC 注入队
    scheduler.rs           # 调度核心：窃取、优先级、命名线程分发
    job.rs counter.rs      # Job、Counter（原子计数依赖/等待句柄）
    fiber/                 # 栈池、上下文切换（经 prism_platform）、等待集
    scope.rs join.rs       # 结构化并行作用域、fork-join
    parallel.rs            # parallel_for / reduce / prefix_sum / 自适应粒度
    async_exec.rs          # 最小 Future executor + waker + 桥接 Counter
    affinity.rs numa.rs    # 命名线程、亲和、NUMA 就近、大小核
    arena.rs               # 每 worker 帧竞技场 bump 分配器
    determinism.rs         # 固定分配 + 确定性树形归约 + 可回放
    cancel.rs throttle.rs  # 取消令牌、并发配额/节流
    diagnostics.rs         # 作业 trace / 窃取率 / 火焰图数据
  features = ["std","multi_thread","fibers","async","numa","determinism","trace"]
```

依赖：`prism_platform`（线程/亲和/NUMA/fiber 上下文/计时），可选 `prism_math`/`prism_diagnostic`。**不碰任何 `bevy_*`。**

---

## 21. 契约、不变量与版本化

- **Counter 单调归零**：完成只递减，不复用已归零 Counter（复用需显式 reset）。
- **结构化作用域闭合**：`scope`/`join` 退出前内部作业必全完成（Rust 生命周期护栏 + 运行时保证）。
- **窃取不破坏结果确定性**：前提是作业间无数据竞争（作业安全 + ECS 冲突图保证）；违反即报告。
- **命名线程唯一性**：`Affinity::Named(Render)` 作业只在该线程跑，不被窃取到别处。
- **版本化契约**：`Counter`、`JobHandle`、`ScopeHandle`、`Priority`、`Affinity`、`CancelToken`、确定性种子格式。

---

## 22. 路线图（M0–M6）与基准即规格

- **M0 线程池**：Worker + Chase-Lev 队列 + 注入队 + `spawn`/`wait(Counter)` + `single` 档同步回退 → 跑通 fork-join + 单测（无死锁/依赖满足）。
- **M1 结构化并行**：`scope`/`join`/`parallel_for`/`reduce`/`prefix_sum` + 自适应粒度 + 主线程即 worker。
- **M2 Fiber**：栈池 + 上下文切换 + 等待集 + `wait` 挂起不阻塞 worker（`fibers` 档）；关档忙帮工回退。
- **M3 异步 + 命名线程**：最小 Future executor + waker + Counter 桥接；Main/Render/IO/AsyncCompute 命名线程。
- **M4 内存/亲和/NUMA**：每 worker 竞技场、栈池调优、worker 绑核、NUMA 就近 + 跨节点窃取惩罚、大小核。
- **M5 确定性 + 高级**：确定性树形归约 + 固定分配 + 可回放；优先级继承、取消、节流、panic 传播。
- **M6 集成/工具**：ECS `par_iter` / system 内并行底层切到本池；App 子应用流水线接线；作业 trace / 窃取率 / 火焰图；bevy_tasks 兼容 prelude。

**基准即规格**：每里程碑以微基准红绿为完成判据——fork-join 吞吐、`parallel_for` 随核数加速比、fiber 切换 ns 级开销、窃取率/占用率、确定性双跑结果位等价。核心价值集中在 **M2（fiber）+ M4（内存/NUMA）+ M5（确定性）**。

---

## 23. 诚实边界与风险

- M0–M6 核心路线图**已全部落地并通过验证**：实现 + 单测（117 项 lib 测试全绿）+ 基准，`cargo clippy --all-targets` 零告警、`cargo test` 零失败。状态随代码演进；§24「AAA 高级功能增补」仍为 PLANNED，按本文优先级随消费方接线落地。
- **高风险项**：
  1. **Fiber 上下文切换（M2）**：平台相关 + `unsafe` 要害（栈切换/寄存器保存），跨 OS 行为差异大，须经 `prism_platform` 严格抽象 + 大量压测；可先上「忙帮工」回退跑通语义，再上真 fiber。这是能否复刻 Naughty Dog「等待不阻塞核」的决定性一步。
  2. **无锁双端队列（M0）**：Chase-Lev 的内存序（Acquire/Release/SeqCst）极易错，须 loom/模型检查 + 高并发压测；错一个 fence 就是偶发数据损坏。
  3. **作业安全/数据竞争（M1/M6）**：默认结果确定性依赖「无数据竞争」前提；须与 ECS 冲突图联调，debug 档运行时冲突检测必须到位，否则乱序执行会偶发不确定。
  4. **确定性归约（M5）**：浮点加法非结合，并行归约默认结果会抖；确定档必须固定结合树形归约，有性能代价，需权衡。
  5. **优先级反转/饥饿**：IO/后台池配额与优先级继承须联调，防前台作业被后台饿死或被低优持锁阻塞。
  6. **栈池容量**：fiber 栈数不足会死锁（所有 worker 的 fiber 都在 wait 且无空栈恢复）；须按最大并发等待深度留足 + 水位告警。
- **与既有文档关系**：本层是 ECS 文档 §8.3 fiber 作业图与 App 文档 §9 流水线的**共同执行底座**，三者 Counter/作业语义须保持契约一致；ECS system 内并行、App 子应用并行都投到本统一池。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 24. AAA 高级功能增补（v0.2）

本章补齐顶级作业系统常被忽视、却在真实 AAA 项目里缺一不可的能力。均 feature/档位门控，默认不付成本；与前文的 work-stealing + fiber 内核互补。

### 24.1 优先级 / QoS 车道与帧预算调度 —— ✅ 已交付（`qos` 模块）

单一队列无法区分「必须本帧完成」与「可拖到空闲」的作业。引入**优先级车道**：

| 车道 | 用途 | 调度策略 |
|---|---|---|
| `Critical` | 本帧渲染/物理关键路径 | 抢占式，优先窃取 |
| `Normal` | 常规 gameplay 系统 | 公平窃取 |
| `Background` | 流送/烘焙/预计算 | 仅空闲 worker 消费，可跨帧 |
| `Deadline(t)` | 带截止时刻 | 按 EDF（最早截止优先）排序 |

配合 `prism_time` 帧预算（见 time §24.5）：调度器感知「本帧剩余毫秒」，预算耗尽时把未开工的 `Background` 顺延下帧，杜绝背景作业挤爆帧时间。

**交付状态**：`pkg/prism_tasks/src/qos/`（`lane.rs` 纯决策核心 + `mod.rs` 线程池门面）已落地。核心 `LaneQueues<T>`（五条 FIFO 车道按 `Priority` 判别子索引，`core::array::from_fn` 初始化）给出**纯确定性、无时钟、无线程**的调度决策：四条前台车道（`Low`..=`Critical`）为本帧必跑的承诺工作，按车道从高到低、车道内 FIFO 依次放行；唯一的 `Background` 车道按**帧预算**门控——`next_step(&mut remaining_nanos)` 仅当背景队头估算成本 `est_nanos` 仍能装进剩余余量时放行并扣减，否则返回 `DeferredBackground` 把未开工的背景作业**顺延下帧**。门控用的 `remaining_nanos` 正是 `prism_diagnostic` 帧预算台账导出的 `remaining_background_nanos`（本帧前台实测后、帧目标下的背景余量），两侧语义严格对齐：帧紧时背景让路。门面 `FrameScheduler`（`TaskPool::frame_scheduler`）薄封装既有 `TaskPool`：被放行的作业经 `spawn`/`wait` 扇出到真实工作窃取池（含 help-on-wait 不死锁与单线程内联回退），`run_frame` 返回确定性的 `FrameRunReport`（前台跑数/背景跑数/顺延数/消耗纳秒/剩余余量），其数值只取决于队列内容与预算、与 worker 完成先后无关。`no_std`+`alloc` 兼容（核心仅用 `core`+`alloc::collections::VecDeque`）。**9 单测绿**：车道优先级+车道内 FIFO 对拍独立串行 oracle、预算耗尽背景顺延（队头阻塞式）、预算充足全放行、零预算放行零成本背景而顺延有成本者、零预算不阻塞前台、空队列 `Idle`、`peek_plan` 预测 `next_step`、`FrameScheduler` 多线程下前台必跑+背景门控+顺延作业跨帧补跑、单线程回退内联执行；`cargo clippy -p prism_tasks --all-targets` 零告警（含 `doc_markdown`）。诚实边界：单个估算成本大于整帧余量的背景作业会一直顺延到余量足够的帧（队头阻塞，不拆分、不抢跑），这是刻意的「不炸帧时间」契约；`est_nanos` 为调用方估算，非实测。

### 24.2 主线程亲和与线程类分离 —— ✅ 已交付（`thread_class` 模块）

部分工作**只能在特定线程**跑（GPU 提交、窗口/输入事件、某些平台 API）：

- `main_thread_only` 作业投递到主线程专用队列，worker 不碰。
- **线程类分离**：compute 池（吃满 CPU）与 I/O 池（多为阻塞等待）分开，避免阻塞 I/O 饿死计算 worker（对标 Unreal `TaskGraph` 的命名线程 + 独立 I/O 线程）。
- fiber 等待（见前文）解决「compute 作业内部等依赖」；线程类分离解决「阻塞型工作不占 compute 核」。

**交付状态**：`pkg/prism_tasks/src/thread_class/`（`route.rs` 纯决策核心 + `mod.rs` 执行门面）已落地。核心 `ClassRouter<T>`（三条 FIFO 车道按 `WorkClass` 判别子索引，`core::array::from_fn` 初始化）给出**纯确定性、无时钟、无线程**的分类路由：`route(WorkClass)->ExecLane` 是唯一路由表（`Compute→ComputePool`、`Io→IoPool`、`MainThread→MainQueue`），`next_step(&mut LaneBudget)` 按固定服务序（compute→io→main，车道内 FIFO）放行，compute/I-O 按 `LaneBudget` 的每波额度准入（`usize::MAX` 为无界）、main 车道恒准入（入队无并发成本），额度耗尽的 compute/I-O 原地**顺延下一波**并由 `Idle` 报回 `compute_deferred`/`io_deferred`。三类车道彼此独立，故作业绝不跨类：把 `io_slots` 设为 `0`（I-O 车道饱和/停滞）时 **compute 与 main 仍全量放行、零 I-O**——这正是「阻塞 I-O 不饿死 compute」的确定性表达。门面 `ThreadClassPool`（`TaskPool::thread_class_pool`）薄封装既有设施**复用而非重写**：`submit_compute`/`submit_io`/`submit_main` 入队，`dispatch` 按核心决策把 compute 作业 `spawn` 到既有工作窃取 `TaskPool`、I-O 作业投递到既有 `NamedThreads` 的 `Io` 阻塞线程（不占 compute worker）、main 作业入 `NamedThreads` 主队列仅由 `pump_main` 在主线程抽水执行；`wait` 经 `TaskPool::wait`（help-on-wait 不死锁）阻塞至本波 compute+I-O 完成，`DispatchReport` 的计数只取决于队列内容与预算、与执行先后无关。单线程回退 `ThreadClassPool::inline`（单线程池经 `thread_class_pool` 自动选用）compute+I-O 按 `dispatch` 内联、依车道抽干序执行，main 仍入内部队列候 `pump_main`，无任何 OS 线程、完全确定。`no_std`+`alloc` 兼容（核心仅用 `core`+`alloc::collections::VecDeque`）。**12 单测绿**：路由表全覆盖+索引序、无界抽干三类 FIFO 对拍独立串行 oracle、I-O 饱和不饿死 compute/main（隔离）、compute 额度精确/超额顺延（边界）、空路由 `Idle`、`admits` 额度语义+main 恒准入、`peek_plan` 预测 `next_step`、顺延作业跨波补跑、门面多线程 compute 入池·I-O 离池·main 候泵+`wait` 同步、门面 `io_slots=0` 隔离（compute 跑而 I-O 顺延再放行）、门面每波额度钳位三波抽干、单线程回退内联按序执行；`cargo clippy -p prism_tasks --all-targets` 零告警（含 `doc_markdown`）。诚实边界：`LaneBudget` 的每波额度是准入计数（面向帧级分波投递的背压），非实时并发信号量，门面不感知 I-O/compute 执行线程的实时空闲度；真正的平台异步 I-O（io_uring/IOCP/kqueue）事件唤醒属 §24.5，随 `prism_asset` 落地；主线程抽水 `pump_main` 需调用方在主线程按帧驱动。

### 24.3 结构化并发与取消 —— ✅ 已交付（`scope_concurrency` 模块）

作用域任务：父作用域退出前自动 join 所有子作业，杜绝悬挂/泄漏（对标 Rust `std::thread::scope`、Kotlin structured concurrency）：

```rust
scope(|s| {
    s.spawn(|| chunk_a());
    s.spawn(|| chunk_b());
}); // 退出此处前两子作业必完成
```

- **取消令牌**：长作业（寻路、烘焙）可携带 `CancelToken`，关卡切换/玩家离开时协作式取消，释放资源。
- 取消是**协作式**（作业在检查点自查），非强杀，保证状态一致。

**交付状态**：`pkg/prism_tasks/src/scope_concurrency/`（`tree.rs` 纯决策核心 + `mod.rs` 执行门面）已落地。核心 `ScopeTree`（根 `NodeId(0)`，节点状态 `Pending`/`Running`/`Joined`/`Cancelled` 带 `is_terminal`）给出**纯确定性、无时钟、无线程**的作用域任务树语义：`spawn_child` 建父子边（父已取消则子**出生即取消**）、`set_running`/`set_joined` 尊重终态不回退、`cancel_subtree` 以**前序遍历**返回被取消的节点集合（跳过已 `Joined`/`Cancelled` 的分支），`depth`/`pending_descendants`/`subtree_preorder` 等查询均为纯函数，直接对拍独立串行镜像 oracle（取消传播的集合与顺序、出生即取消、兄弟隔离、终态单调）。门面 `StructuredScope`/`TaskPool::structured_scope` 薄封装既有 `TaskPool::scope`**复用而非重写**其生命周期擦除与 join 栅栏（本模块**不引入任何新 `unsafe`**），只叠加 `CancelToken` 树与「准入/跳过/完成」记账：作用域从父 `CancelToken` 派生子令牌（父取消级联子、父已取消则子出生即取消），每个 `spawn` 的子任务领到作用域令牌、在自身检查点协作式自查；若任务将启动时令牌已取消则**整体跳过**（计入 `ScopeOutcome::skipped`）而非跑起来再退出，否则跑到完成（计入 `completed`），两种情形都在作用域返回前被 join。`ScopeOutcome` 的 `spawned == completed + skipped` 恒成立，与 worker 完成先后无关。继承 `TaskPool::scope` 的 help-on-wait（不死锁）与单线程内联回退。`no_std`+`alloc` 兼容（核心仅用 `core`+`alloc`，门面用 `alloc::sync::Arc` 共享记账计数）。**10 单测绿**：取消子树集合+前序顺序对拍 oracle、出生即取消、终态不回退、兄弟隔离、`spawn` 形状对镜像、作用域 join 全部并计数、预取消作用域跳过每个任务、`cancel` 级联子令牌、父令牌取消抵达作用域（协作式跳过）、单线程回退内联执行；`cargo clippy -p prism_tasks --all-targets` 零告警（含 `doc_markdown`）。诚实边界：取消是**协作式**——已在飞行中的任务只在其下一个检查点（`is_cancelled`/`check`）观察到取消并自愿退出，本模块不强杀线程；取消的*传播*与跳过记账是确定的，但真实 OS 线程何时抵达检查点取决于调度，不由确定性核心感知。

### 24.4 数据并行原语（parallel_for / reduce / scan / join）—— ✅ 已交付（`par_cost` 模块）

高层门面，隐藏手工分块：

```rust
tasks.par_for(0..n, GRAIN, |i| process(i));        // 自动分块 + 粒度自适应
let sum = tasks.par_reduce(items, 0, |a,b| a+b);   // 并行归并（确定序，见 24.8）
tasks.join(|| left(), || right());                 // 分治二叉 join
```

- 粒度自适应：运行期按单元耗时标定 chunk 大小，平衡调度开销与负载均衡。
- 供 `prism_transform` 分块传播（见 transform §8）、ECS 并行查询、物理 island 求解直接复用。

**交付状态**：核心 `parallel_for`/`reduce`/`prefix_sum`/`join` 早在 M1/M5/M6 已落地（按 len+worker 数静态选 grain）。本次补齐 §24.4 真正的增量——**运行期按单元耗时标定 grain**：`pkg/prism_tasks/src/par_cost.rs` 的 `CostModel`（`target_chunk_nanos`/`min_grain`/`probe_items`/`oversubscription`，纯函数 `grain(len,workers,cost_per_item_nanos)`：成本下界摊销调度开销 ⊓ 负载均衡上界保证 `workers×oversubscription` 个块 ⊓ 硬下限 ⊓ 不超 len）与门面 `TaskPool::par_for_calibrated`/`par_reduce_calibrated`（先把真实工作的前缀 probe 内联计时估单元成本，再按标定 grain 经 `scope` 扇出，每索引恰好访问一次）。标定读时钟→块数随运行变动,故 `par_reduce_calibrated` 对非结合 combine **非位复现**（位复现走 §24.7 `deterministic_reduce`,文档已明示边界）。9 单测绿（grain 三段式钳位/廉价摊到下限/昂贵细分/不超 len/builder 钳 1/for 全索引覆盖/空·小区间/reduce 对拍串行和/小区间走 probe 结果）,clippy 零告警。

### 24.5 异步 I/O 桥（async/await 集成）

把 `async` future 跑在作业池上，I/O 等待不占 compute 核：

- 平台异步 I/O（io_uring / IOCP / kqueue）事件就绪后唤醒对应作业，与计算流水线重叠。
- 供 `prism_asset` 异步加载/热重载：加载作业发 I/O 请求 → fiber/future 让出 → 数据到达续跑解析，全程不阻塞 worker。

### 24.6 NUMA 与混合核拓扑感知

- **NUMA**：worker 绑核，作业优先在产生数据的 NUMA 节点消费，减少跨节点内存访问（服务器/工作站大核场景）。
- **混合核（P-core/E-core）**：感知 Intel/ARM big.LITTLE 拓扑，`Critical` 车道优先派给性能核，`Background` 派给能效核；移动端省电。
- 拓扑探测由 `prism_platform` 提供，缺失时退化为均匀池。

### 24.7 确定性并行（可回放）—— ✅ 已交付（`deterministic_replay` 模块）

回滚网络/录像要求并行结果**与线程数/窃取时序无关**：

- 并行 `reduce`/`scan` 走**确定归并树**（固定结合顺序），而非到达序累加。
- 结果写回按实体/索引有序，不依赖 worker 完成先后。
- 与 ECS §确定性、`prism_replication` 契约一致：同输入 → 同输出（位等价），无论几核。

**交付状态**：`pkg/prism_tasks/src/deterministic_replay/`（`order.rs` 纯决策核心 + `mod.rs` 线程池门面）已落地，模块文件头 `#![forbid(unsafe_code)]`。核心抓住一个诚实事实：真正不确定的只有**顺序敏感归并**（非交换 fold：序列拼接/哈希/`wrapping_mul` 链）**各任务结果被提交进累加器的先后**——哪个 worker 先完成、提交以何序落地，取决于 OS 调度与窃取时序，是工作窃取池里的真实竞态。模块把该竞态**可复现而不改变结果**：`SeedStream`（`splitmix64`，`for_task(seed,i)` 为纯函数，worker 与串行 oracle 对同一任务索引算出同值）合成「同种子 → 同输入」的工作负载；`ExecutionOrder`（种子 + `0..len` 的**提交序排列**，`validate` 校验为真排列）捕获该次提交序，可序列化为文本（`prism-order v1 seed=.. len=..` + 每行一索引）与紧凑二进制（`PROR` 魔数 + 版本 + LE 种子/长度/索引），往返无损；`fold_in_order` 是**纯串行 oracle**——按给定序重算每任务值并折叠，replay 必须逐位复现它。门面在既有 `TaskPool` 上**复用而非重写**：`record_ordered(seed,len,compute,init,combine)` 经 `TaskPool::scope`（help-on-wait join 栅栏 + 单线程内联回退）并行算值、在一把 `Mutex` 下提交（到达锁的顺序即真实完成序，逐一 append 索引并折叠），返回 `ReplayOutcome{value,order}`；`replay_ordered(&order,…)` 校验排列后并行把每任务值算入各自槽位、再在调用线程**严格按记录序**折叠，结果与录制 run **位等价**，与本次 replay 的 worker 数无关；`deterministic_ordered(seed,len,…)` 直接按**规范序 `0..len`** 折叠，是「同种子 → 同结果、与核数无关」的免录制确定性原语。门面不新增任何并发原语、不引入 `unsafe`。`no_std`+`alloc` 兼容（`order.rs` 仅用 `core`+`alloc`；门面合理使用 `std::sync::Mutex`）。**19 单测绿**：排列校验/非法序拒绝（重复·越界）、文本/二进制往返、畸形输入拒绝（截断·坏头·坏项·长度不符·非排列）、`SeedStream` 可复现且按索引寻址、`fold_in_order` 顺序敏感性（正序≠逆序）、**录制值对拍其自身序的串行 oracle**（2000 任务 8 worker）、**replay 在 1/2/4/8-worker 与单线程回退上逐位复现录制值**、序列化往返后 replay 仍复现、**规范序折叠跨 1/2/4/8/单线程 worker 数不变**对拍 oracle、规范序≠逆序 replay、异种子异结果、重复录制各自对拍 oracle 并 replay 复现、空负载往返、单线程录制序即规范序；`cargo clippy -p prism_tasks --all-targets` 零告警。诚实边界：本模块把并行的**结果排序**规整为确定——它**不复现底层线程的原始窃取/调度时序**（那是 OS 不可控的非确定输入），只捕获并复现「提交序对结果的影响」；`compute` **必须是任务索引（与种子）的纯函数**，否则 replay 不保证复现（这是契约，非实现缺陷）；`record_ordered` 的提交点是一把全局 `Mutex`（刻意串行化 commit 以取真实到达序），其并行性在于**提交前的 compute**，故适配「并行映射 + 顺序敏感归并」这类负载；对**交换律**归并，`deterministic_ordered` 的结果本就与序无关，record/replay 主要服务于「要复现某一次具体非确定交织以供调试/回放」的场景。

### 24.8 背压、死锁预防与健康监测 —— ✅ 已交付（`health` 模块）

- **背压**：队列深度超阈值时拒绝/降级 `Background` 投递，防内存爆。
- **死锁预防**：作业图在提交时静态检测环依赖；fiber 等待链深度限幅，防栈耗尽。
- **健康监测**：worker 饥饿/长作业/窃取失败率导出到 `prism_diagnostic`，`trace` 档出每作业 span（tracy/chrome-trace）与依赖图可视化。

**交付状态**：`pkg/prism_tasks/src/health/`（`backpressure.rs`/`wait_graph.rs`/`monitor.rs` 三个纯决策核心 + `mod.rs` 执行门面）已落地。**背压**：`QueueBackpressure`（`BackpressureLimits::new` 钳位 `low ≤ high ≤ cap`、`cap ≥ 1`）以**双水位滞回**给出纯确定性准入——深度达 `high` 起进入 shedding、降到 `low` 才解除（滞回防抖），shedding 期降级丢弃 `Background` 投递（`Deferred`），满 `cap` 一律 `Rejected`，前台优先级不受降级影响；`offer`/`complete`/`complete_many`/`headroom`/`peek` 均为纯函数，对拍逐步重放 oracle（含 LCG 随机流多配置）。**死锁预防**：`WaitGraph`（任务/资源等待图）在**提交依赖时**用可达性检测**成环即拒绝**（`try_add_dependency` 先 `creates_cycle` 探测、确认加边才落图），`find_cycle` 用三色迭代 DFS 还原具体环，`longest_chain`/`resulting_chain_depth` 对等待链**深度限幅**（防 fiber 等待链栈耗尽），错误以 `DeadlockError::{Cycle, ChainTooDeep}` 精确报出；对拍暴力可达性 oracle、自环、链限、删边。**健康监测**：`LatencyHistogram`（升序去重桶 + 溢出桶，`percentile_nanos` 用 nearest-rank + ceil 整数除、`max_nanos`）给确定性延迟分位、`StarvationDetector`（每 worker 连续空闲计数 vs 阈值）检测**饥饿**、`StealStats`（`failure_per_mille`）给**窃取失败率**，`PoolHealthMonitor` 聚合为确定性 `HealthReport`（峰值队列深度/饥饿 worker 数/窃取失败千分比/p50·p99·最长作业延迟/样本计数）；分位单调、样本计数等对拍 nearest-rank oracle（含随机流）。门面 `BackpressureQueue`（`TaskPool::backpressure_queue`）把准入决策接到真实 `TaskPool`：`offer` 经背压门控准入后入队、`run` 扇出 `dispatch`+`wait`+drain；`HealthProbe`（`Arc<Mutex<PoolHealthMonitor>>`，`Clone`）在 worker 上以 `std::time::Instant` 采**真实**作业墙钟延迟喂进直方图。`no_std`+`alloc` 兼容（三个核心仅用 `core`+`alloc`；`HealthProbe`/`BackpressureQueue` 门面合理使用 `std`）。**16 单测绿**：背压准入/滞回/逐步重放 oracle（含随机）、满 cap 拒绝、等待图成环拒绝对拍暴力可达性、`find_cycle` 还原环、自环、链深限幅、删边、直方图分位对拍 nearest-rank（含随机）、空·乱序桶、饥饿检测、窃取失败率、`backpressure_queue` 与 `health_probe` 门面；`cargo clippy -p prism_tasks --all-targets` 零告警（含 `doc_markdown`）。诚实边界：死锁*预防*是**提交期静态环检测 + 链深限幅**（确定性），不等于运行期无锁争用的实时死锁消解；`QueueBackpressure` 的水位是队列深度账（面向内存爆与过载的准入背压），非实时并发信号量；`HealthProbe::measure` 采的墙钟延迟取决于真实 OS 调度（唯一的非确定输入），但其后一切派生指标（分位、最长作业、饥饿计数、窃取率）全是确定函数；worker 的真实空闲须由调用方喂 `record_idle`/`record_busy`，确定性核心不自行感知线程实时空闲。

### 24.9 诚实边界

**24.1 优先级 / QoS 车道 + 帧预算调度**（`qos` 模块）、**24.2 主线程亲和与线程类分离**（`thread_class` 模块）、**24.3 结构化并发与取消**（`scope_concurrency` 模块）、**24.4 并行原语标定**（`par_cost` 模块）与 **24.8 背压、死锁预防与健康监测**（`health` 模块）已落地（见各节「交付状态」），是其他 crate（transform/ECS/物理）最先依赖的能力；本章其余小节（24.5 异步 I/O、24.6 NUMA/混合核）仍为 PLANNED 设计目标，24.7 确定性并行（可回放）已落地（见该节「交付状态」）。24.1 的帧预算门控与 `prism_diagnostic` 的 `remaining_background_nanos` 严格对齐（背景作业据可用余量决定是否本帧执行）；24.2 线程类分离（`thread_class` 模块：compute 入工作窃取池、I-O 走 `NamedThreads` 离池阻塞线程、main-only 候 `pump_main` 主线程抽水，`io_slots=0` 时 compute/main 不受阻）已落地，纯 `ClassRouter` 决策核心对拍独立 oracle、门面复用既有 `TaskPool`/`NamedThreads`；真正的平台异步 I-O 唤醒属 §24.5 随 `prism_asset`；24.5 异步 I/O 随 `prism_asset` 落地；24.6 NUMA/混合核随 M5 落地。24.7 确定性并行（`deterministic_replay` 模块：`ExecutionOrder` 捕获提交序排列 + `fold_in_order` 纯串行 oracle + 门面 `record_ordered`/`replay_ordered`/`deterministic_ordered` 复用 `TaskPool::scope`）已落地，诚实边界为只规整并行**结果排序**的确定性、不复现底层窃取/调度原始时序，且要求 `compute` 为任务索引的纯函数。24.3 结构化并发（`scope_concurrency` 模块：纯 `ScopeTree` 决策核心对拍独立 oracle、`StructuredScope` 门面复用 `TaskPool::scope` 的 join 栅栏 + `CancelToken` 树做协作式取消与跳过记账）已落地，诚实边界为协作式取消依赖真实线程何时抵达检查点。24.8 背压/死锁预防/健康监测（`health` 模块：`QueueBackpressure` 双水位滞回准入、`WaitGraph` 提交期环检测 + 链深限幅、`PoolHealthMonitor` 确定性指标）已落地，诚实边界为 `HealthProbe` 墙钟延迟采样取决于真实 OS 调度（唯一非确定输入，其后派生指标全确定）、死锁预防为提交期静态检测而非运行期实时消解。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

