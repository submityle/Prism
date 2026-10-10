# Prism Diagnostic 顶级次世代 AAA 级诊断 / 日志 / 可观测性设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **结构化日志 + CPU/GPU 作用域计时 + 帧捕获时间线 + 计数器/统计/直方图 + 断言契约 + 崩溃捕获 + 远程实时观测** 统一可观测性内核设计。它是 `bevy_log` / `bevy_diagnostic` 的自研替代，也是全引擎「看得见自己在干什么、慢在哪、崩在哪」的唯一真相面。
> 借形态不抄码。借鉴：
> - **结构化日志 + Span 订阅模型**：Rust `tracing` / `tracing-subscriber`（Span/Event/Field/Layer）
> - **帧级性能剖析**：Tracy、Optick、RAD Telemetry、Unreal Insights、Superluminal（帧时间线 / 作用域 / 锁竞争 / GPU 轨道）
> - **跨工具 trace 格式**：Chrome Trace Event / Perfetto（可用现成 UI 消费）
> - **崩溃捕获**：Breakpad / Crashpad 形态 minidump + 符号化
> - **帧统计/HUD**：Unreal `stat` 命令族、Unity Profiler 计数器
> 本文为纯经典可观测性 / 计时 / 日志路线，**不含任何 AI/ML 内容**。

- 版本: v0.2（核心 M0–M6 已落地并验证；§24 高级增补 24.1–24.7 已全部交付代码+单测；v0.1→v0.2 新增第 24 章「AAA 高级功能增补」：实时性能预算与自动回归告警/CPU·GPU 统一时间线跨队列关联/内存分配追踪与泄漏碎片可视化/确定性回放与 trace 对拍(反 desync)/统计采样剖析器(低开销)/分布式多实例聚合观测/发行版遥测与隐私脱敏；其中 24.1–24.7 已全部交付代码+单测，零告警零失败）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: `prism_utils`（无锁环形缓冲 / 字符串驻留 / 位集）、`prism_platform`（高精度时钟 / 线程 ID / 文件写出 / minidump）、`prism_time`（帧序号 / 时间线对齐）；可选 `prism_render_driver`（GPU timestamp query）、`prism_tasks`（job 时间线）
- 层级定位: L1 地基（被所有上层 crate 以「埋点」方式依赖；是横切关注点，不反向依赖业务 crate）
- 明确约束: 核心 `no_std + alloc`（环形缓冲 / 事件编码）；`std`（文件/socket/线程名）/ `gpu`（GPU 计时）/ `tracy`（Tracy 协议）/ `chrome-trace` / `crash`（minidump）/ `remote`（实时面板）为 feature；release 下埋点可编译期归零；**不依赖任何 `bevy_*` crate**

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier / feature）
4. 分层架构
5. 核心模型：Event / Span / Scope / Counter / Marker
6. 结构化日志（分级 / 字段 / 过滤 / 订阅者）
7. CPU 作用域计时（Span / Scope / 层级火焰）
8. GPU 计时（timestamp query，接 RHI）
9. 帧捕获与时间线（Tracy / Chrome Trace / Perfetto 导出）
10. 计数器 / 统计 / 直方图（帧统计 / 内存 / 绘制调用）
11. 断言与契约检查（debug / release 分级）
12. 崩溃捕获与 minidump
13. 低开销设计（无锁环形缓冲 / 线程本地 / 采样）
14. 与 ECS / App / tasks / render 集成
15. 远程 / 实时观测
16. 可观测性的自观测（开销预算）
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

一个 AAA 引擎每帧跑成百上千个 system/job、几千次绘制调用、几 GB 内存流动。**看不见就没法优化，复现不了就没法修 bug**。`prism_diagnostic` 的使命是：用**可编译期归零的低开销埋点**，把「谁在跑、跑了多久、用了多少、为什么崩」变成一条可查询、可导出、可实时观测的真相流——开发期全开求洞察，发布期几乎零成本。

**一句话定位**：`prism_diagnostic` 是 Prism 的「可观测性内核」——结构化日志（出了什么事）、作用域计时（花了多少时间）、计数器（用了多少资源）、崩溃捕获（为什么挂了）四合一；埋点 API 极简（一行宏），后端可插拔（Tracy / Chrome Trace / 自有面板 / 文件）；release 下未启用的档位编译期消失。

四条总目标（按权重）：

1. **性能**：埋点热路径无锁、线程本地、每事件 < 几十纳秒；release 未开档位编译期归零（零指令、零字节）。
2. **效果（能力）**：日志 + CPU/GPU 计时 + 计数器 + 崩溃 + 帧捕获 + 远程面板，覆盖 AAA 全套观测需求。
3. **易用**：`info!()`、`span!()`、`profile_scope!()`、`counter!()` 一行埋点；现成 UI（Tracy/Perfetto）直接消费，不必自造轮子。
4. **可移植 + 档位化**：核心 `no_std`；文件/socket/GPU/崩溃按 feature 裁剪；服务器/客户端/发布包各取所需。

---

## 2. 参考产品取舍

| 来源 | 吸收 | 规避 |
|---|---|---|
| Rust `tracing` | Span/Event/Field 结构化模型、`Layer` 可组合订阅者、编译期级别过滤 | 纯日志向、无原生 GPU 计时/帧捕获/崩溃；性能埋点需额外层 |
| Tracy | 极低开销帧级作用域、锁竞争/上下文切换、实时连接、现成强大 UI | 协议私有需适配；嵌入需运行时常驻 |
| Optick / Superluminal | 轻量作用域宏、采样剖析、直觉火焰图 | 平台/商业约束；仅借作用域与采样形态 |
| Chrome Trace / Perfetto | 开放 JSON/proto 格式、浏览器即可看、跨语言 | 非实时（导出后看）；大 trace 文件膨胀 |
| Breakpad / Crashpad | 跨平台 minidump、符号化、崩溃上报管线 | 集成复杂、平台相关；仅借 minidump 形态 |
| Unreal `stat` / Unity Profiler | 帧统计 HUD、分类计数器、层级统计命令 | 与各自引擎耦合；仅借计数器分类与 HUD 思路 |

**取舍结论**：以 `tracing` 的「Span/Event/Field + 可插拔 Layer」为**统一数据模型**（日志与性能埋点同源），用 Tracy 的「极低开销作用域 + 实时连接」为**性能后端标杆**，导出走 Chrome Trace/Perfetto 让用户用现成 UI，崩溃走 minidump 形态。一套埋点，多个后端，开发期实时、离线可导出、发布期可归零。

---

## 3. 档位化（capability / quality tier / feature）

- **埋点档（instrumentation tier）**：`off`（release 默认，编译期归零）/ `lite`（仅关键 Span + 帧统计，可随发布包）/ `full`（全量作用域 + GPU + 计数器直方图，开发期）。
- **后端档（sink tier）**：`fmt`（控制台/文件格式化）/ `tracy`（实时）/ `chrome`（导出 JSON）/ `remote`（自有面板）/ `null`（丢弃）；可多后端并存。
- **功能 feature**：`std`、`gpu`、`tracy`、`chrome-trace`、`perfetto`、`crash`（minidump）、`remote`、`hud`（屏上统计覆盖层）、`alloc-track`（分配剖析）。
- **编译期级别**：`max_level_info` 等常量在编译期裁掉更低级别日志（零运行时成本）。

裁剪示例：发布客户端 `["std","crash","hud"]`（仅崩溃上报 + 帧率 HUD，日志编译期砍到 warn）；开发客户端 `["std","gpu","tracy","chrome-trace","alloc-track"]`；专用服务器 `["std","chrome-trace","remote"]`（无 GPU/HUD）。

---

## 4. 分层架构

```
  ┌──────────────────────────────────────────────────────────┐
  │ 埋点 API（宏）: info! span! profile_scope! counter! assert │  ← 上层只碰这层
  ├──────────────────────────────────────────────────────────┤
  │ 核心数据模型: Event / Span / Scope / Counter / Marker      │
  │              + 线程本地环形缓冲（无锁写入）                 │
  ├──────────────────────────────────────────────────────────┤
  │ 收集器 Collector: 聚合/采样/级别过滤/帧边界切分            │
  ├───────────┬───────────┬───────────┬───────────┬──────────┤
  │ fmt sink  │ tracy sink│chrome sink│remote sink│crash sink│  ← 可插拔后端
  └───────────┴───────────┴───────────┴───────────┴──────────┘
        │           │           │           │
     文件/控制台  Tracy UI   .json/.perfetto 自有面板   .dmp
```

关键：**上层只写宏，不知后端存在**；后端通过注册组合（像 tracing 的 Layer）。埋点写入走**线程本地无锁环形缓冲**，收集器在帧边界或后台线程 drain，绝不在热路径做 I/O 或加锁。

---

## 5. 核心模型：Event / Span / Scope / Counter / Marker

```rust
pub enum Level { Error, Warn, Info, Debug, Trace }

// 瞬时事件（日志一行、一个标记）
pub struct Event<'a> { pub level: Level, pub target: &'a str, pub fields: Fields<'a>, pub ts: u64 }

// 有时长的区间（system 执行、函数作用域）—— RAII guard 自动闭合
pub struct Span { id: SpanId, name: &'static str, parent: Option<SpanId> }
pub struct ScopeGuard; // Drop 时记录结束时间戳

// 数值采样（帧率、绘制调用数、内存字节、队列深度）
pub struct Counter { name: &'static str, kind: CounterKind /* Gauge/Sum/Histogram */ }

// 时间线标注（帧开始、关卡加载、GC）
pub struct Marker { name: &'static str, color: u32, ts: u64 }
```

设计约定（固定契约）：
- **统一时间基**：所有时间戳取自 `prism_platform` 高精度单调时钟，跨线程可对齐到同一时间线（接 `prism_time` 帧序号）。
- **Span 层级**：Span 有父子关系，天然组成火焰图；跨线程 Span 用 thread id + flow 事件连接（job 跨线程续）。
- **字段结构化**：日志字段是 `key=value`（可查询/可过滤），而非拼成一行文本——支持后期按字段聚合。
- **RAII 计时**：`profile_scope!("name")` 返回 guard，离开作用域自动记结束，不会漏配对。

---

## 6. 结构化日志（分级 / 字段 / 过滤 / 订阅者）

- **宏**：`error! / warn! / info! / debug! / trace!`，支持结构化字段 `info!(entity = ?id, dt, "spawned")`。
- **编译期裁剪**：`max_level` 常量在编译期砍掉低级别调用（release 下 `trace!/debug!` 零成本）。
- **运行期过滤**：按 `target`（模块路径）+ level 的过滤指令（`prism_render=debug,prism_physics=warn` 环境变量/配置），运行时可改。
- **订阅者（Layer）**：格式化输出（控制台带色/JSON 行/文件滚动）、字段聚合、限流（same-message rate limit 防刷屏）。
- **与 panic/错误集成**：`panic` 钩子记 Error 事件 + 栈；错误类型可携带 Span 上下文。

---

## 7. CPU 作用域计时（Span / Scope / 层级火焰）

- **作用域宏**：`profile_scope!("Physics::Step")` RAII 计时；函数级 `#[profile]` 属性宏（可选）。
- **层级火焰图**：Span 父子 + 每线程一条轨道，直接喂 Tracy/Chrome Trace 画火焰图。
- **采样 vs 埋点**：埋点（精确区间）为主；可选统计采样剖析（周期性抓调用栈，`platform` 提供栈回溯）补未埋点热点。
- **锁/等待可视化**：对 `prism_tasks` 的 job 等待、锁获取埋 flow 事件，可视化竞争与 stall（Tracy 形态）。

---

## 8. GPU 计时（`gpu` feature，接 RHI）

- **GPU 作用域**：`gpu_scope!(encoder, "ShadowPass")` 在命令缓冲插 timestamp query 对，经 `prism_render_driver` 读回。
- **CPU↔GPU 时间线对齐**：标定 GPU 时间戳到 CPU 时间基（周期性 calibration），GPU 轨道与 CPU 轨道同屏对齐（找 CPU 等 GPU 的气泡）。
- **管线统计**：可选 pipeline statistics query（顶点数、片元数、裁剪数），供渲染瓶颈定位。
- **跨帧延迟**：timestamp 读回有 N 帧延迟，收集器维护环形关联帧号，不阻塞当前帧。

---

## 9. 帧捕获与时间线（Tracy / Chrome Trace / Perfetto 导出）

- **帧边界**：`frame_mark!()` 切分帧，时间线按帧对齐；可「捕获最近 N 帧」dump 到文件。
- **Tracy 实时**（`tracy` feature）：嵌入 Tracy 协议 server，Tracy 客户端实时连上看火焰/统计/内存/锁。
- **Chrome Trace / Perfetto**（`chrome-trace`/`perfetto`）：导出 `.json`/proto，浏览器 `chrome://tracing` 或 Perfetto UI 打开。
- **按需捕获**：平时 lite 档低开销运行，触发（热键/命令/卡顿检测）时切 full 并 dump 近窗口，避免全程 full 的开销。

---

## 10. 计数器 / 统计 / 直方图（帧统计 / 内存 / 绘制调用）

- **计数器类型**：`Gauge`（瞬时值，如 FPS/内存占用）、`Sum`（累加，如本帧绘制调用）、`Histogram`（分布，如帧时间 p50/p95/p99）。
- **分类**：按子系统分组（Render/Physics/Audio/ECS/Asset）的统计树，`stat render` 形态查询。
- **帧统计内建**：帧时间、FPS、CPU/GPU 各 pass 时长、绘制调用、三角形数、实体数、drawcall、内存分池占用（接 `alloc-track`）。
- **HUD**（`hud` feature）：屏上覆盖层实时显示关键计数器 + 帧时间曲线（供性能回归肉眼巡检）。

---

## 11. 断言与契约检查（debug / release 分级）

- **分级断言**：`debug_assert!`（仅 debug）、`check!`（release 也在但可配置为 warn/abort）、`verify!`（求值一定执行，仅断言可剥离）。
- **契约宏**：前置/后置/不变量断言，失败时记 Error 事件 + 栈 + 相关字段，而非裸 panic。
- **软失败模式**：发布包可配置「断言失败记录并继续」而非崩溃（AAA 游戏常用，避免一个边角 bug 顶掉玩家）。
- **与崩溃捕获联动**：断言失败可触发 §12 minidump，带上契约上下文。

---

## 12. 崩溃捕获与 minidump（`crash` feature）

- **信号/异常钩子**：捕获 SIGSEGV/SIGABRT/SEH，写出 minidump（经 `prism_platform` 平台 API）。
- **上下文附加**：崩溃时附最近 N 帧的 Span/日志环形缓冲快照、引擎版本、关卡/场景、显卡/驱动信息。
- **符号化**：离线用符号文件解析堆栈；dump 文件可本地或上报（上报走业务层，本 crate 只产 dump）。
- **安全**：崩溃处理器只用预分配缓冲 + 异步信号安全调用，不在崩溃路径分配/加锁。

---

## 13. 低开销设计（无锁环形缓冲 / 线程本地 / 采样）

- **线程本地写入**：每线程一个无锁环形缓冲（`prism_utils` 提供），埋点仅「写一条紧凑编码事件 + 推进指针」，无锁无 I/O。
- **后台 drain**：收集器线程周期性 drain 各线程缓冲，做聚合/级别过滤/喂后端；热路径与 I/O 完全解耦。
- **紧凑编码**：事件按紧凑二进制编码（字符串驻留成 ID，`prism_utils` 字符串池），省带宽省缓存。
- **编译期归零**：`off` 档下宏展开为空，无任何残留指令/字节；lite/full 由 feature + 常量级别控制。
- **开销预算**：full 档目标埋点 < 1% 帧时间；lite 档 < 0.1%；超预算由 §16 自观测报警。

---

## 14. 与 ECS / App / tasks / render 集成

- **prism_app**：主循环每阶段自动 `frame_mark` + 阶段 Span；`App` 启动时注册后端（配置驱动）。
- **prism_ecs**：每个 system 执行自动包 Span（system 名 + 实体数），直接产出 system 火焰图（接 ECS §调度）；变更检测/原型迁移计数入计数器。
- **prism_tasks**：每个 job 一个 Span + 跨线程 flow 连接，可视化 work-stealing 负载均衡与 stall（tasks §时间线）。
- **prism_render**：每 pass 一个 GPU scope；drawcall/三角形/VRAM 入计数器；渲染图节点时间线。
- **prism_asset**：加载/烘焙/热重载事件入日志 + 耗时 Span，定位资产卡顿。

---

## 15. 远程 / 实时观测（`remote` feature）

- **实时协议**：引擎开 socket，自有面板/CLI 连上实时拉日志流 + 计数器 + 帧时间线（无需停机）。
- **远程命令**：面板可下发命令（切埋点档、触发帧捕获、改日志过滤、dump 内存），运行时调参。
- **多实例**：专用服务器/多客户端场景下，面板聚合多实例观测（分布式压测可观测）。
- **安全**：远程端口默认本地回环 + 鉴权；发布包默认关闭（避免信息泄露/被滥用）。

---

## 16. 可观测性的自观测（开销预算）

- **自计时**：诊断系统自身的 drain/编码/后端写出耗时入专门计数器，确保「观测开销」本身可见。
- **预算报警**：埋点开销超帧预算阈值时降档（full→lite）并记 warn，避免观测拖垮被观测对象。
- **缓冲溢出计数**：环形缓冲满丢弃事件时计数（而非阻塞），丢弃率可见；提示提高缓冲或降采样。
- **后端背压**：后端（文件/socket）慢时丢旧而非阻塞热路径；背压事件可见。

---

## 17. 高级功能增补（AAA）

- **因果追踪（flow / causality）**：跨线程/跨帧的异步任务用 flow 事件串成因果链（请求→加载→上传→渲染），定位异步延迟。
- **内存剖析（`alloc-track`）**：hook 分配器（接 `prism_utils` allocator trait）按调用栈/标签归类分配，产出内存火焰图 + 泄漏/峰值定位。
- **卡顿检测器（hitch detector）**：帧时间超阈值自动触发近窗口捕获 + 高频采样栈，专抓偶发卡顿（AAA 最难调的问题）。
- **确定性回放标记**：确定性档下记录每帧状态哈希 + 输入序列（接 `prism_replication`），desync 时定位首个分叉帧。
- **GPU crash 调试**：GPU 挂起/设备丢失时读 breadcrumb（命令缓冲进度标记，接 RHI），定位挂在哪个 pass（对标 Aftermath/DRED 形态）。
- **统计录制与回归**：帧统计序列可录制为基线，CI 对比检测性能回归（接「基准即规格」）。

所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 18. 性能工程

- **热路径零 I/O 零锁**：埋点只写线程本地环形缓冲；一切 I/O/聚合在后台线程。
- **字符串驻留**：静态字符串编译期驻留成 ID，事件只存 ID + 时间戳 + 少量字段，极致紧凑。
- **分支预测友好**：级别检查用 likely/unlikely 提示 + 编译期常量裁剪，未命中埋点近零成本。
- **缓存友好缓冲**：环形缓冲按 cache line 对齐；批量 drain 顺序读。
- **采样降频**：高频事件（每顶点级别）禁止埋点，仅作用域/帧级埋点；需要时用统计采样而非全埋。

---

## 19. 易用性与 Bevy 迁移策略

- **宏对齐 tracing**：`info!/warn!/span!` 与 `tracing` 同形，Bevy（本就用 tracing）迁移基本是改 `use`。
- **`bevy_diagnostic` 兼容层**（`compat-bevy`）：`DiagnosticsStore`/`FrameTimeDiagnosticsPlugin` 形态 API 映射到 Prism 计数器。
- **一行接入**：`App` 加 `DiagnosticPlugin::dev()`（全开）或 `::release()`（仅崩溃+HUD），后端自动注册。
- **现成 UI**：导出 Chrome Trace/Perfetto 用浏览器看，连 Tracy 看实时，无需学自有工具。
- **约定优于配置**：system/job/pass 自动埋点，用户零手工埋点即得火焰图；手工埋点仅为细化。

---

## 20. crate 分层与模块布局

```
pkg/prism_diagnostic/
  src/
    lib.rs              # 宏导出 + prelude + DiagnosticPlugin
    model.rs            # Event/Span/Scope/Counter/Marker + Level
    macros.rs           # info!/span!/profile_scope!/counter!/check!
    buffer.rs           # 线程本地无锁环形缓冲（依赖 prism_utils）
    collector.rs        # 聚合/采样/级别过滤/帧切分
    log/                # 结构化日志
      fmt.rs filter.rs
    cpu_timing.rs       # CPU 作用域 + 火焰
    gpu_timing.rs       # GPU timestamp query（gpu feature，接 RHI）
    counters.rs         # Gauge/Sum/Histogram + 帧统计
    assert.rs           # 分级断言/契约
    crash.rs            # minidump（crash feature，经 platform）
    sinks/              # 后端
      tracy.rs chrome.rs perfetto.rs remote.rs file.rs
    hud.rs              # 屏上统计覆盖（hud feature）
    alloc_track.rs      # 分配剖析（alloc-track feature）
    self_obs.rs         # 自观测/开销预算
    prelude.rs
  features = ["std","gpu","tracy","chrome-trace","perfetto","crash",
             "remote","hud","alloc-track","compat-bevy"]
```

依赖：`prism_utils` + `prism_platform` + `prism_time`；可选 `prism_render_driver`（GPU 计时）。**不碰任何 `bevy_*`。**

---

## 21. 契约、不变量与版本化

- **横切不反依赖**：诊断是横切关注点，只被依赖、不反向依赖任何业务 crate（避免循环依赖）。
- **埋点可归零**：`off` 档下埋点必须编译期完全消失（无指令/无字节），这是发布包契约。
- **时间基单一**：所有时间戳同一单调时钟源，可跨线程/CPU-GPU 对齐。
- **热路径无阻塞**：埋点永不阻塞、永不 I/O、永不分配（崩溃路径只用预分配）；缓冲满则丢弃并计数，绝不阻塞仿真。
- **后端可插拔**：新增后端不改埋点 API；trace 导出格式（Chrome/Perfetto）为版本化契约。
- **版本化**：事件编码格式、trace 导出 schema、minidump 附加数据布局、远程协议均版本化。

---

## 22. 路线图（M0–M6）与基准即规格

- **M0 日志骨架**：Event/Level + 宏 + fmt 后端（控制台/文件）+ 编译期级别裁剪 → 单测（过滤/字段）。
- **M1 作用域计时**：Span/Scope RAII + 线程本地环形缓冲 + Chrome Trace 导出 → 浏览器看到火焰图。
- **M2 计数器 + 帧统计**：Gauge/Sum/Histogram + 帧时间/drawcall 内建统计 + HUD → 屏上实时统计。
- **M3 ECS/tasks 集成**：system/job 自动埋点 + flow 跨线程连接 → system 火焰图 + 负载可视化。
- **M4 GPU 计时**：timestamp query + CPU/GPU 时间线对齐（接 RHI）→ 找到 CPU 等 GPU 气泡。
- **M5 Tracy + 远程**：Tracy 实时 + 远程面板/命令 → 实时观测 + 运行时调参。
- **M6 崩溃 + 内存剖析 + 卡顿**：minidump + alloc-track + hitch detector + 确定性回放标记 → 偶发崩溃/卡顿/泄漏可定位。

**基准即规格**：埋点单事件开销（ns）、full/lite 档帧开销占比、环形缓冲吞吐与丢弃率、trace 导出大小、GPU 计时精度、崩溃捕获成功率。核心价值在 **M1（火焰图）+ M3（ECS/tasks 集成）+ M6（崩溃/卡顿定位）**。

---

## 23. 诚实边界与风险

- M0–M6 核心路线图**已全部落地并通过验证**：实现 + 单测（55 项 lib 测试全绿）+ 基准，`cargo clippy --all-targets` 零告警、`cargo test` 零失败。状态随代码演进；§24「AAA 高级功能增补」24.1–24.7 已全部交付（见 §24.8），按本文优先级随消费方接线深化。
- **高风险项**：
  1. **埋点开销失控（M1/M2）**：埋点太细（每顶点/每实体）会反噬帧预算，观测拖垮被观测；必须严守「仅作用域/帧级埋点 + 热路径零 I/O」，并由 §16 自观测兜底。
  2. **CPU/GPU 时间对齐（M4）**：GPU 时间戳域与 CPU 不同、读回有延迟、驱动差异大；标定不准则火焰图误导人；需多驱动验证。
  3. **崩溃路径安全（M6）**：崩溃处理器里分配/加锁会二次崩溃；必须严格异步信号安全 + 预分配；跨平台（SEH vs signal）实现差异大。
  4. **Tracy/格式版本漂移（M5）**：Tracy 协议、Chrome/Perfetto schema 会升级；需版本协商与兼容层，否则 UI 打不开 trace。
  5. **多线程事件乱序（M3）**：跨线程时间线需严格单调时钟 + flow 连接，时钟漂移会让因果链错乱；依赖 `prism_platform` 时钟质量。
  6. **发布包信息泄露（M5）**：远程端口/详细日志若在发布包默认开启，可能泄露内部信息或被滥用；默认关闭 + 鉴权是硬约束。
- **与既有文档关系**：环形缓冲/字符串驻留依赖 `prism_utils_design_zh.md`；时钟/线程/minidump 依赖 `prism_platform_design_zh.md`；帧序号/时间线对齐接 `prism_time_design_zh.md`；system/job 自动埋点接 `prism_ecs_design_zh.md` 调度与 `prism_tasks_design_zh.md`；GPU 计时接 `prism_render_driver`（RHI）；确定性回放标记接 `prism_replication`。整体组件缺口见 `prism_engine_component_gap_zh.md`。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

## 24. AAA 高级功能增补（v0.2）

本章补齐顶级可观测性系统在真实 AAA 项目里缺一不可、却常被最先砍掉的能力。均 feature/档位门控，默认不付成本；与前文的无锁环形缓冲 + Span/Counter 内核互补。

### 24.1 实时性能预算与自动回归告警 —— ✅ 已交付（`budget` 模块）

AAA 项目的帧时间是**契约**（60/120Hz 必达），不能靠事后看 trace：

- **预算声明**：各子系统声明帧预算（渲染 ≤8ms、物理 ≤2ms、gameplay ≤3ms…），运行期对比实测，超支即红标 + 事件。
- **自动回归检测**：CI/夜构里把每帧关键 Span 的 p50/p99 存基线，新提交超阈值（如 +5%）自动告警并定位到提交，杜绝「性能慢性退化」。
- **预算接 tasks 调度**：实时帧预算喂给 `prism_tasks`（见 tasks §24.1 帧预算调度），背景作业据此顺延，形成「观测→调度」闭环。
- **热点自动归因**：火焰图顶部帧按耗时自动排序 + diff 上一基线，直接指向回归函数。

**交付状态**：已落地 `pkg/prism_diagnostic/src/budget.rs`（`no_std` + `alloc`，纯 `core` 算术、无自带时钟、确定性）。`BudgetRegistry`：各子系统 `declare(category, budget_nanos)` 声明帧预算（60/120Hz 预设 `fps_60`/`fps_120`），`evaluate`/`evaluate_frame` 对比实测 → 超支 `BudgetStatus{over_budget, overspend_nanos, utilization}` 红标；`FrameBudgetReport{total_measured_nanos, over_frame, remaining_background_nanos}` 把帧内剩余余量喂给 `prism_tasks` 帧预算调度（tasks §24.1，纯 `u64` 读数，无依赖边）。`RegressionTracker`：存 `Baseline{p50,p99}`（`from_samples` 最近秩分位，与 `hitch` 同法），`check(key, p50, p99, commit)` 超阈值（默认 +5%，带噪声下限滤除微 span）→ `RegressionAlert`（归因到提交），不自动晋升基线。`hotspot_diff(current, baseline, threshold)`：按自耗时降序（火焰图顶部）稳定排序 + 逐名 diff，新出现热点与超阈值项标 `is_regression`。10 单测全绿、`cargo clippy --all-targets` 零告警。
**优先级**：高（帧时间契约地基；预算→调度闭环的观测侧，CI 回归告警防性能慢性退化）。

### 24.2 CPU·GPU 统一时间线与跨队列关联 —— ✅ 已交付（`gpu` 模块：时间基对齐 + 统一时间线 + 跨队列关联 + 提交→执行延迟 + GPU 气泡 + Chrome/Tracy 导出）

单看 CPU 或 GPU 时间线无法发现「CPU 提交早、GPU 却因依赖空等」：

- **统一时间轴**：CPU Span 与 GPU timestamp（见 §8）投影到同一时间线，可视化「提交→执行」延迟与 GPU 气泡。
- **跨队列关联 ID**：一次 draw/dispatch 带关联 token，串起 CPU 录制 → 队列提交 → GPU 执行 → 呈现（present），端到端延迟一屏可见。
- **PIX/RenderDoc/Tracy 对齐**：导出格式与主流 GPU 剖析器时间基对齐，便于交叉验证（接 RHI 的 debug marker）。

**交付状态（`gpu` feature，纯 `core`/`alloc` 整数、无 `unsafe`、后端中立）**：
- **时间基对齐**（`gpu::calibration`）：`GpuClockCalibration` 以 `timestamp_period`（ns/tick）+ 周期性 `(cpu_now, gpu_now)` 校准样本拟合 CPU↔GPU 仿射映射（`AffineFit`），`project_span` 把 GPU tick 区间投影到 CPU ns 轴，使 CPU/GPU 落在同一时间线。
- **N 帧回读关联环**（`gpu::ring`）：`GpuReadbackRing` 把 issue（录制期）与 resolve（数帧后回读）解耦，`ready_queries` 给出可回读帧，`resolve` 产出带关联 token 的 `GpuSpan`。
- **统一时间线**（`gpu::timeline`）：`UnifiedTimeline` 把 CPU `SpanRecord` 与投影后 GPU 区间按 `(start, 轨道种类, 轨道 id, label)` 确定性合并为 `TimelineEntry`（`TimelineTrack::Cpu{thread_id}` / `Gpu{queue}`）；`CorrelationId` 串起一次操作的录制→提交→执行→呈现各阶段，`end_to_end_latency` 给端到端延迟。
- **提交→执行延迟与 GPU 气泡**（`gpu::latency`，本次新增独立模块）：`correlation_breakdown` 把关联链拆为 CPU 提交侧与 GPU 执行侧，直接读出 `submit_to_execute_nanos`（CPU 提交完 → GPU 开始执行的空等，饱和到 0 表示流水线重叠无可观测停顿）、`gpu_active_nanos`、`total_nanos`；`gpu_bubbles` 以运行覆盖端扫描单队列相邻执行区间，报告气泡（空闲缝隙，正确处理重叠/嵌套不误报），`gpu_idle_nanos` 给队列总空闲。
- **主流剖析器对齐**（`gpu::timeline` + `trace::chrome` + `profiler::tracy`）：`export_chrome_with_gpu` 把每条 GPU 队列作为独立 `tid`（`GPU_TRACK_TID_BASE + queue`）与 CPU 轨道对齐导出 Chrome/Perfetto；`tracy` feature 的 `TracyMessage::GpuZone` 产出 Tracy 兼容 GPU 区事件，便于与 PIX/RenderDoc/Tracy 交叉验证。
- **验证**：`cargo test -p prism_diagnostic --features gpu` 全绿（含 `gpu::latency` 8 项 + `tests_gpu_timeline` 3 项手算 oracle：全流程 submit→execute 延迟、气泡检测、Chrome 导出对齐）；`cargo clippy -p prism_diagnostic --all-targets --features gpu` 与默认特征均零告警；无桩。

### 24.3 内存分配追踪与泄漏/碎片可视化 —— ✅ 已交付（`alloc_track` + `mem` 模块：泄漏对账/碎片/预算守卫）

- **带标签分配**：接 `prism_utils` 分配器埋点（utils §10 alloc-track），每次分配记录调用栈 + 类别标签（资产/渲染/gameplay），按类别出内存占用树。
- **泄漏检测**：作用域/帧边界对账未释放分配，关卡切换后应归零的池若残留即报泄漏 + 分配栈。
- **碎片可视化**：虚拟内存/池占用图谱，暴露碎片与大块空洞，指导分配器调参（接 platform §9 内存信息）。
- **预算守卫**：类别内存超预算触发事件，防发行版 OOM。

**交付状态**：热路径核心 `alloc_track`（带标签 `GlobalAlloc`、精确 live/peak 字节、per-tag 累计）已交付，并新增 `LiveTrackingAllocator`——在零开销无头部 `TrackingAllocator` 之外附加 opt-in 的 per-allocation 头部，按分配基址盖上归属标签，`dealloc` 读头部回减对应标签的 live 计数，交付精确 per-tag **live** 字节（§24.3 泄漏归属信号；`TagStat::live_bytes` 填充）；无头部分配器仍是默认零成本路径、其 `live_bytes` 恒为 0；本次在其之上补齐帧/作用域边界的对账与守卫层，置于独立 `mem` 子模块（`src/mem/`，纯 `core`/`alloc` 算术、无 `unsafe`、与 `alloc-track` feature 无关，始终编译）：
- 泄漏检测（`mem::leak`）：`LeakCheckpoint` 在作用域/帧开闭边界各取一次 live 字节/分配快照，`reconcile` 出带符号残差 `LeakReport`（正=泄漏、负=过度释放、零=归零平衡）；`alloc-track` 开启时可直接从 `AllocSnapshot` 构造。
- 预算守卫（`mem::budget`）：`MemBudgetRegistry` 按类别（资产/渲染/gameplay）声明 live 字节上限，`evaluate`/`evaluate_all` 出 `MemBudgetReport`（逐类别红标 `over_budget`、超额字节、聚合与 `offenders()`），声明顺序稳定以便 diff/HUD；类别名可与 `alloc_track::tag_report` 的标签名对应取实测。
- 碎片可视化（`mem::fragmentation`）：`analyze_fragmentation(capacity, &[Span])` 归一化（排序/合并/钳位）占用区间后，出 `FragmentationReport`（used/free 字节、空闲 run 数、最大连续空闲 run、`fragmentation_ratio`、`can_fit`）；`occupancy_map(capacity, occupied, buckets)` 出每桶 0..=100 占用百分比热条。
专项测试 `tests_mem.rs`（12 用例，手算 oracle 对拍）全绿；`LiveTrackingAllocator` 另有 5 个 inline 单元测试（对齐/边界、per-tag live 往返、无标签不污染、交错标签独立、realloc 调整 live）直接驱动 `alloc`/`dealloc`（无需装为 `#[global_allocator]`，爆炸半径受限）全绿，`cargo clippy -p prism_diagnostic --features alloc-track --all-targets` 与默认 features 均零告警。

### 24.4 确定性回放与 trace 对拍（反 desync）✅ 已交付（`determinism` 模块）

回滚网络/录像/确定性模拟最难调的就是「两次运行结果不一样」：

- **确定性 trace**：确定性档下录制每帧关键状态哈希（ECS 世界哈希、物理状态哈希），两次运行逐帧对拍，首个分叉帧即 desync 根因。
- **输入+随机种子录制**：录制输入流 + 种子，支持精确回放复现偶发 bug（对标 Overwatch/格斗游戏的回放调试）。
- **跨平台位对拍**：Win/macOS/Linux 三端跑同输入，对拍世界哈希，验证四方确定性（接 ECS 序/tasks 归并/time 定点/transform 定点）。

**交付状态**：§24.4 三要素已交付，置于独立 `determinism` 子模块（`src/determinism/`，纯 `core`/`alloc` 定点整数运算、无 `unsafe`、`no_std`+`alloc`，始终编译）：
- 确定性 trace（`determinism::hash` + `determinism::trace`）：稳定 64 位 `FNV`-1a 哈希器 `StateHasher`（固定字段顺序折叠、整数小端、浮点按位、无 per-process 种子，附 `const` 的 `fnv1a_64`），`DeterminismTrace` 逐帧记录 `FrameHash`（帧号+状态哈希）。
- 输入+随机种子录制流（`determinism::record`）：`InputRecorder` 逐帧录制 `FrameInput`（帧号+输入哈希+种子），`to_trace`/`digest` 把录制流折叠成可对拍的 trace/摘要，`InputReplay` 游标把同一输入+种子序列精确回放复现（偶发 bug 重放）。
- 两条 trace 逐帧对拍（`determinism::compare`）：返回首个分叉帧 `TraceDiff`——`Identical`/`Diverged{frame,left,right}`/`LengthMismatch{matched,left_len,right_len}`，按位置且按值匹配（丢帧/改帧号不会被静默重对齐）。
与 `prism_time` 的 `multiworld::audit` 审计器概念对齐（同一 `FNV`-1a 常量、同一 identical/diverged/length-mismatch 对拍形态）但**不产生依赖边**：这是 diagnostic 层面向任意子系统的通用 trace 对拍，与已有 `replay` 标签标记流互补（后者是带标签的标记流，本模块是逐帧状态哈希流）。专项测试 `tests_determinism.rs`（25 用例，含 `FNV`-1a 官方向量钉死 + 手算结构 oracle）全绿，`cargo clippy -p prism_diagnostic --all-targets` 零告警。
已知边界：各帧的「关键状态哈希」内容（ECS 世界/物理状态怎么折叠）由调用方在边界处用 `StateHasher` 喂入；跨 Win/macOS/Linux 三端位对拍需上层接 ECS 序/time 定点后对接本模块，本层只提供确定性整数对拍原语，不自带模拟或 RNG。

### 24.5 统计采样剖析器（低开销）—— ✅ 已交付（`sampling` 模块）

插桩 Span 覆盖不到第三方库与未插桩热点：

- **采样剖析**：定时中断采当前调用栈（接 platform 高精度时钟 + 栈回溯），统计式还原热点，开销 <1%，可在发行版开。
- **插桩 + 采样融合**：插桩给精确关键 Span，采样补全全局分布，两者在同一火焰图叠加。
- **按线程/车道分面**：区分 compute/I-O/主线程（接 tasks §24.2 线程类分离）的热点分布。

**交付状态**：`sampling` 模块（`symbol`/`sample`/`fold`/`facet`/`fusion` 五个子文件 + `mod`）落地确定性采样数据模型与离线统计还原：

- 符号驻留（`symbol`）：`SymbolTable` 首见即稳定 `FrameId` 驻留栈帧名，`?<id>` 兜底未解析帧。
- 采样缓冲（`sample`）：`StackSample`（车道 `LaneKind`/线程 id/时间戳/权重/栈帧序列）+ `SamplingProfiler`（采样间隔钳为 `>=1`，`record_named`/加权记录/总权重）。
- 热点折叠（`fold`）：`flat_profile` 自身命中（叶）/包含（每样本去重）折叠、`call_tree` 自顶向下（`TopDown`）/自底向上（`BottomUp`）调用树（合成根 + 子节点按包含降序再按 `FrameId` 定序）、`collapsed_stacks` 输出 Brendan Gregg 折叠格式（`BTreeMap` 字典序）。
- 车道/线程分面（`facet`）：`facet_by_lane`（规范车道序）/ `facet_by_thread`（线程 id 升序）统计热点分布。
- 插桩 + 采样融合（`fusion`）：`fuse` 把 `flat_profile` 与 `InstrumentedSpan` 叠加成 `FusedProfile`，`agreement_ratio`（采样包含 / 插桩包含）+ `disagreements(tol)` 定位两源背离的作用域。

纯 `core`/`alloc` 整数、无 `unsafe`、始终编译。专项测试 `tests_sampling.rs`（14 用例，固定四样本缓冲手算 oracle 对拍：驻留稳定性、flat 自身/包含计数与行序、折叠栈渲染、调用树结构与深度、车道/线程分面、融合比与背离集）全绿，`cargo clippy -p prism_diagnostic --all-targets` 零告警。
已知边界：真实的定时中断采样 + 调用栈回溯 + `<1%` 运行时开销由上层 `prism_platform`（高精度时钟 + 栈回溯）接线触发，本层只拥有确定性的采样数据模型与离线统计还原（flat/tree/collapsed/分面/融合），全部 oracle 对拍、不自带时钟或中断。

### 24.6 分布式 / 多实例聚合观测 —— ✅ 已交付（`aggregate` 模块）

专用服务器/大世界多进程需要**集群级**视角：

- **多实例聚合**：N 个服务器实例/客户端的指标汇聚到一处，出集群帧时间分布、异常实例定位。
- **分布式 trace 关联**：一次跨进程请求（客户端→服务器→DB）用关联 ID 串成分布式 span（OpenTelemetry 形态）。
- **抽样上报**：高频指标按采样率上报，控制带宽；异常实例自动提采样。

**交付状态**：`aggregate` 模块（`instance`/`cluster`/`dtrace`/`sampling_rate` 四个子文件 + `mod`）落地确定性集群聚合、分布式关联与采样率策略：

- 单实例汇总（`instance`）：`InstanceFrameReport` 收原始帧时间，`InstanceSummary::from_samples` 用全 crate 一致的最近秩分位（`percentile_nearest_rank`）压成 count/min/max/sum/p50/p90/p99/p999 紧凑摘要；实例可报原始样本或边缘预压摘要。
- 集群聚合 + 异常定位（`cluster`）：`ClusterAggregator` 把 N 份报告池化成 `ClusterFrametimeReport`（全舰队 p50/p99/p999 分布），用**中位 p99 x 因子 + 中位绝对偏差（MAD）次级门**鲁棒定位异常实例——用中位数（非均值）避免一台已坏实例抬高基线掩盖第二台，`min_median_nanos` 地板避免健康快舰队误报，异常按最坏优先排序。
- 分布式 trace 关联（`dtrace`）：`TraceAssembler` 按 `TraceId` 分组、按 `SpanId` 父子链接把任意乱序到达的 `DistributedSpan` 装成 `AssembledTrace` 森林（OpenTelemetry 形态），导出 `CriticalPath`（最长时长根到叶链）与 per-service 延迟归因（`ServiceLatency`）；父未到达的 span 升为 orphan 根而非丢弃，重复 `SpanId` 保留首个。
- 自适应采样上报（`sampling_rate`）：`SampleRate`（`1/N` 上报，`should_report(seq)`/`expected_reports(captured)` 确定性带宽估计）+ `SamplingController` 把集群报告转成 per-instance `SamplingDecision`——健康实例走基线稀疏率、被判异常实例自动提到满分辨率（boost 永不低于基线），`estimate_fleet_reports` 估全舰队带宽。

纯 `core`/`alloc` 整数、无 `unsafe`、始终编译。专项测试 `tests_aggregate.rs`（23 用例，独立 oracle 对拍：最近秩分位、池化分布、MAD 单/双异常与误报抑制、trace 树/关键路径/服务归因/orphan/去重、采样率上报模式与计数、boost 与基线覆盖）全绿，`cargo clippy -p prism_diagnostic --all-targets` 零告警。
已知边界：网络传输 / RPC（把报告与 span 在进程间搬运）属上层接线，本层只拥有确定性的集群聚合（池化分位、鲁棒中位 + MAD 异常定位）、分布式 trace 关联/树装配、采样率策略数据结构，全部离线 oracle 对拍。

### 24.7 发行版遥测与隐私脱敏 —— ✅ 已交付（`telemetry` 模块）

- **分级遥测**：开发全量、发行版仅关键指标（崩溃率、帧时间 p99、内存峰值），可远程调级。
- **隐私/合规**：遥测与崩溃转储（§12）默认脱敏（去 PII、去绝对路径、去内存明文），需用户同意方上传，策略在本层统一（接 platform §16 只负责产 dump）。
- **符号化离线化**：崩溃堆栈用稳定堆栈哈希分桶聚类，上传最小化，符号在后端离线还原。

**交付状态**：§24.7 的脱敏 + 事件构建数据模型已交付，置于独立 `telemetry` 子模块（`src/telemetry/`：`redact`/`event`/`sampling` + `mod`，纯 `core`/`alloc` 整数、无 `unsafe`、无时钟、无 RNG，始终编译）：

- 隐私脱敏（`telemetry::redact`）：`redact_user_path` 剥离绝对家目录的用户名段（`/Users/<name>/`→`/Users/redacted/`、`/home/<name>/`、Windows `\Users\<name>\`，支持内嵌于长串/多次出现/非 ASCII 用户名，按 ASCII 字节边界扫描保证 UTF-8 安全）；`hash_identifier` 复用 crate 稳定 `FNV`-1a（`determinism::fnv1a_64`）把标识符单向哈希成 `h:<16hex>` 稳定非加密 token；`truncate_str` 按 `char` 截断并追加省略号标记（不切裂多字节）；`RedactionPolicy` 把字段白名单/黑名单过滤与「丢弃/哈希/路径剥离+截断」per-value 变换组合（优先级 deny>allow>hash>keep）。
- 事件构建（`telemetry::event`）：`EventSchema` 把字段分级为必采/可选/禁采（未声明字段默认禁采 = 严格白名单，`allow_unknown` 可放宽）；`build_event` 产出 `RedactedEvent`（字段按 key 排序、禁采/被丢字段入 `dropped`、缺失必采字段入 `missing_required` 而非伪造），`canonical` 确定性可逆序列化（`\`/`|`/`=` 转义）+ 64 位 `digest`；`EventAggregator` 按 digest 把同签名事件折叠成聚合计数（`ranked` 按计数降序定序）。
- 确定性采样（`telemetry::sampling`）：`SampleRatio`（keep/out_of，`always`/`never`/`one_in`/`ratio` 构造并钳定不变量，`admits`/`expected_keep`/`fraction`）对「事件名加盐的 key 哈希桶」做决策——绝不用真随机，相同 key 在相同事件名下恒落同一桶；`TelemetrySampler` 带默认率 + per-event 覆盖率（发行版可崩溃全采、帧统计稀疏采），事件名加盐使各事件流采样相互独立。

确定性 + 隐私是本层双契约：相同输入事件恒产生相同脱敏/序列化/采样结果，且敏感原文（用户名、原始标识符）在输出中可证缺席。专项测试 `tests_telemetry.rs`（38 用例，固定输入手算 oracle 对拍：`FNV`-1a 钉死的哈希/桶/digest 向量、路径剥离各形态、UTF-8 截断、策略优先级、分级构建与缺失必采、转义、聚合排名、采样钳定/边界/确定性）全绿，`cargo clippy -p prism_diagnostic --all-targets` 零告警。
已知边界：真实网络上传 / 后端摄取与用户同意获取属上层接线（`prism_platform` §16 产原始 dump 并拥有同意弹窗），崩溃堆栈的离线符号化在后端完成；本层只拥有确定性的脱敏 + 事件构建 + 采样数据模型，全部离线 oracle 对拍。

### 24.8 诚实边界

**24.1 性能预算 + 自动回归告警已交付**（`budget` 模块：预算声明/帧级红标/调度余量/p50-p99 回归告警/热点 diff 归因）。**24.3 内存追踪已交付**（`alloc_track` 热路径核心 + `mem` 对账/守卫层：泄漏带符号残差对账、per-类别预算红标、碎片 run/占用图谱分析）；已知边界：per-tag **live** 字节已由 opt-in 的 `LiveTrackingAllocator`（头部盖标签）交付；默认零开销 `TrackingAllocator` 仍只出 per-tag 累计字节（其 live 恒 0），泄漏/碎片层的输入快照与占用列表由调用方在边界处采集。**24.4 确定性回放与 trace 对拍已交付**（`determinism` 模块：稳定 `FNV`-1a `StateHasher`、`DeterminismTrace` 逐帧状态哈希、`InputRecorder`/`InputReplay` 输入+种子录制回放、`compare` 返回首个分叉帧 `Identical`/`Diverged`/`LengthMismatch`；纯 `core`/`alloc` 整数、无 `unsafe`、始终编译）；已知边界：各帧关键状态哈希的内容由调用方用 `StateHasher` 折叠喂入，跨三端位对拍需上层接 ECS 序/time 定点后对接，本层只提供确定性整数对拍原语、不自带模拟或 RNG，与 `prism_time::multiworld::audit` 概念对齐但无依赖边。**24.5 统计采样剖析器已交付**（`sampling` 模块：符号驻留、`StackSample`/`SamplingProfiler` 采样数据模型、`flat_profile` 自身/包含折叠、`call_tree` 自顶向下/自底向上调用树、`collapsed_stacks` 折叠格式、车道/线程分面、插桩 + 采样融合与背离定位；纯 `core`/`alloc` 整数、无 `unsafe`、始终编译）；已知边界：真实定时中断采样 + 栈回溯 + `<1%` 开销由上层 `prism_platform` 接线触发，本层只拥有确定性采样数据模型与离线统计还原、不自带时钟或中断。**24.6 分布式 / 多实例聚合观测已交付**（`aggregate` 模块：`InstanceSummary` 单实例汇总、`ClusterAggregator` 池化分布 + 中位 p99 x 因子 + MAD 鲁棒异常定位、`TraceAssembler` 分布式 span 树装配/关键路径/服务归因/orphan 处理、`SampleRate`/`SamplingController` 自适应采样上报与异常自动提采样；纯 `core`/`alloc` 整数、无 `unsafe`、始终编译）；已知边界：网络传输 / RPC 属上层接线，本层只拥有确定性集群聚合、分布式 trace 关联/树装配、采样率策略数据结构。**24.7 发行版遥测与隐私脱敏已交付**（`telemetry` 模块：`redact_user_path` 用户名路径剥离 / `hash_identifier` 稳定 `FNV`-1a 单向哈希 / `truncate_str` UTF-8 安全截断 / `RedactionPolicy` 白黑名单+per-value 变换、`EventSchema` 必采/可选/禁采分级 + `build_event` 确定性 `canonical`/`digest` + `EventAggregator` 聚合计数、`SampleRatio`/`TelemetrySampler` 事件名加盐 key 哈希的确定性采样；纯 `core`/`alloc` 整数、无 `unsafe`、无时钟、无 RNG、始终编译）；已知边界：真实网络上传 / 后端摄取与用户同意获取属上层接线（`prism_platform` §16 产 dump 并拥有同意弹窗），崩溃堆栈离线符号化在后端完成，本层只拥有确定性脱敏 + 事件构建 + 采样数据模型。**24.2 CPU·GPU 统一时间线已交付**（`gpu` 模块：时间基仿射对齐、N 帧回读关联环、统一确定性时间线、跨队列关联 token 端到端延迟、提交→执行延迟拆解、GPU 气泡/队列空闲扫描、Chrome/Perfetto + Tracy 导出对齐；纯 `core`/`alloc` 整数、无 `unsafe`、后端中立、始终可编译）；已知边界：真实 GPU timestamp 查询的采集精度与多驱动语义校验（D3D12/Vulkan/Metal 查询语义、`timestamp_period` 怪癖、disjoint-query、队列时钟域）由上层 RHI/render 后端接入本摄取缝（issue/resolve + 周期校准样本）后落地，本层只拥有确定性对齐数学、回读关联与时间线投影/分析数据模型，不触碰 GPU、不伪造 GPU 行为。其余为 PLANNED 设计目标，无代码。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。
