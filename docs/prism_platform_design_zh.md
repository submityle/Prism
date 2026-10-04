# Prism Platform 顶级次世代 AAA 级平台抽象层（PAL / HAL）设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **操作系统抽象层**：文件系统 / 路径、高精度时钟、线程与同步原语、虚拟内存 / 页 / 对齐分配、动态库加载、进程 / 环境 / 命令行、CPU 特性探测、原子 / 内存屏障、崩溃捕获。它是整个引擎踩在不同 OS（Windows / macOS / Linux / iOS / Android / 主机 / Web）上时的**唯一地面**，让上层 crate 写一次、处处可跑。
> 借形态不抄码。借鉴：
> - **Rust 生态**：`std`（`fs`/`time`/`thread`/`sync`/`process`）、`libc`、`parking_lot`（高效锁）、`libloading`（动态库）、`rustix`（直系统调用）、`sysinfo`
> - **引擎级 PAL**：Unreal `FPlatformProcess` / `FPlatformTime` / `FPlatformMemory` / `FPlatformMisc` / `FGenericPlatform*`、sokol（极简跨平台）、SDL（设备 / 窗口 / 线程抽象）、PhysicsFS / BGFX 平台层
> - **系统/底层**：mimalloc / jemalloc（虚拟内存与大页策略）、Breakpad / Crashpad（minidump）、hwloc（拓扑 / NUMA）
> 本文为纯经典系统编程路线，**不含任何 AI/ML 内容**。

- 版本: v0.2（核心 M0–M6 已落地并验证；§24 高级增补仍为设计阶段；v0.1→v0.2 新增第 24 章「AAA 高级功能增补」：高级异步 I/O(io_uring·IOCP·批量提交)/虚拟内存进阶(稀疏堆·按需提交·GPU 共享内存)/混合核·NUMA·能耗感知调度钩子/高级崩溃观测(跨进程 Crashpad·稳定堆栈哈希分桶)/安全加固探测(ASLR·DEP·CFG·代码签名)/平台能力数据库与降级矩阵/Web·主机后端进阶；均为 PLANNED，无代码）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: **无 Prism 上游依赖或仅依赖 `prism_utils`**（与 `prism_math` / `prism_utils` 同处依赖图根级）；底层经典 crate `libc` / `rustix` / `windows-sys` / `parking_lot`(可选) / `libloading`(可选)
- 层级定位: L1 地基根级（被 `prism_time`(时钟) / `prism_tasks`(线程) / `prism_app`(主循环/命令行) / `prism_asset`(文件/VFS) / `prism_diagnostic`(minidump) / `prism_math`(SIMD 探测) / `prism_script`(动态库热重载) 直接依赖）
- 明确约束: 核心 `no_std + alloc` 可编译（纯计算/原子/CPU 探测路径），OS 服务（文件/线程/动态库/进程）需 `std` 或平台后端 feature；每个 OS 一个后端模块，统一 trait 门面；**不依赖任何 `bevy_*` crate**

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / tier / feature）
4. 分层架构
5. 核心模型：Platform trait 与能力探测
6. 文件系统与路径（VFS 友好）
7. 高精度时钟（接 time）
8. 线程与同步原语（接 tasks）
9. 虚拟内存 / 页 / 大页 / 对齐分配（接 utils）
10. 动态库加载（热重载，接 script）
11. 进程 / 环境 / 命令行 / 标准流
12. CPU 特性探测与拓扑（接 math / tasks）
13. 原子与内存屏障抽象（no_std）
14. 平台差异矩阵（Win/macOS/Linux/移动/主机/Web）
15. 与 app / window / tasks / asset 集成
16. 崩溃捕获与 minidump（供 diagnostic）
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

引擎最终要落在真实的操作系统上：读文件要走某个 OS 的 `open`，取时间要调某个 OS 的高精度计数器，开线程要调某个 OS 的 `pthread_create` / `CreateThread`。如果每个上层 crate 各写各的 `#[cfg(target_os)]`，代码会碎成一地、移植成本爆炸、Bug 散落各处。`prism_platform` 的使命是：**把所有「与 OS 打交道」的动作收敛到一层**，对上提供一套干净、统一、可测试的门面，对下为每个平台写一个后端，让「换平台 = 换后端」而非「改全引擎」。

**一句话定位**：`prism_platform` 是 Prism 的「操作系统地面」——文件、时间、线程、内存、动态库、进程、CPU 探测、崩溃捕获，全部一个门面多后端；上层 crate 永不直接碰 `libc` 或 `winapi`，只调 Prism PAL；核心尽量 `no_std`，OS 服务按平台后端 feature 开。

四条总目标（按权重）：

1. **可移植**：Windows / macOS / Linux / iOS / Android / 主机 / Web 一套上层代码，差异全锁在后端；新增平台 = 实现一组 trait。
2. **性能**：高精度时钟零系统调用开销（RDTSC/mach/QPC 校准）；锁走 `parking_lot` 级快路径；虚拟内存按页/大页直管；减少不必要的 OS 边界穿越。
3. **效果（能力）**：文件/时钟/线程/虚存/动态库/进程/CPU 探测/原子/崩溃捕获齐全，覆盖引擎所有 OS 需求。
4. **易用 + 可测**：门面 API 直觉贴近 std；提供 `mock` 后端做无 OS 单测；能力探测让上层优雅降级。

---

## 2. 参考产品取舍

| 来源 | 借鉴什么 | Prism 取舍 |
|---|---|---|
| Rust `std` | `fs`/`time`/`thread`/`sync`/`process` 的成熟 API 形态 | 门面向 std 看齐；但补齐 std 没有的大页/亲和性/高精度 TSC/动态库统一抽象 |
| `libc` / `rustix` / `windows-sys` | 直系统调用绑定 | 后端用它们；上层不可见 |
| Unreal `FPlatform*` 家族 | PAL 分类（Process/Time/Memory/Misc/Atomics/TLS）与「Generic + 平台覆写」模式 | 采用 trait 默认实现 + 平台后端覆写的结构 |
| sokol / SDL | 极简跨平台最小集、能力探测 | 借「薄门面 + 能力位」思路，但不绑定其窗口/渲染 |
| `parking_lot` | 自适应快路径锁、无毒化 | 作为同步原语后端选项 |
| `libloading` | 跨平台动态库加载 | 作为动态库后端选项，封装成热重载友好门面 |
| mimalloc/jemalloc | 虚拟内存/大页/区段管理策略 | 借虚存管理形态，供 `prism_utils` 分配器与 GPU 子分配 |
| Breakpad/Crashpad | minidump 捕获与越进程转储 | 借形态做 `platform::crash`，为 `prism_diagnostic` §12 供料 |
| hwloc | CPU 拓扑 / NUMA / 核心亲和性 | 借拓扑模型供 `prism_tasks` 线程池绑核 |

**不做**：不做窗口/输入（归 `prism_window`/`prism_input`）；不做图形 API（归 `prism_render_driver`）；不做网络协议栈（仅暴露 socket 原语给上层网络 crate）。

---

## 3. 档位化（capability / tier / feature）

- **能力位 `PlatformCaps`**：运行时位标志，表达「本平台是否支持 大页 / 核心亲和性 / 高精度 TSC / 动态库卸载 / minidump / 内存保护页 / 虚拟内存预留」等；上层据此优雅降级（如 Web 无大页则回退普通分配）。
- **编译期 feature**：`std`（OS 服务总开关）、`backend-win` / `backend-posix` / `backend-apple` / `backend-android` / `backend-web`（每平台后端）、`crash`（minidump）、`dynlib`（动态库）、`affinity`（绑核）、`hugepages`（大页）、`parking-lot`（高效锁后端）、`mock`（测试后端）。
- **质量/行为档**：时钟档（`coarse` 省电 / `precise` 高精度 TSC 校准）；分配档（`page` 原始页 / `virtual` 预留+提交）；锁档（`std-sync` / `parking-lot` / `spin`）。
- **原则**：核心计算（原子、CPU 探测、时间换算）在 `no_std` 下可用；一旦触碰文件/线程/进程/动态库即需 `std` 或平台后端。

---

## 4. 分层架构

```
          上层 crate（time / tasks / app / asset / diagnostic / math / script）
                                   │  仅依赖门面 trait
          ┌────────────────────────┴────────────────────────┐
          │              prism_platform 门面层                 │
          │  Platform / Fs / Clock / Thread / Vmem /          │
          │  DynLib / Process / Cpu / Atomics / Crash         │
          └────────────────────────┬────────────────────────┘
            ┌──────────┬───────────┼───────────┬──────────┐
         backend-win  backend-apple backend-posix backend-android backend-web  mock
         (QPC/VirtualAlloc) (mach/vm) (libc/rustix) (NDK)   (wasm/emscripten) (测试)
```

- **门面层**：一组 trait + 中立数据类型（`PathBuf` 风格路径、`Instant`/`Duration` 风格时间），不含任何 OS 代码。
- **后端层**：每平台一个模块，`#[cfg(target_os)]` 选择；实现门面 trait。
- **mock 后端**：内存文件系统 + 可编程时钟 + 可控线程，供上层单测脱离真实 OS。

---

## 5. 核心模型：Platform trait 与能力探测

- **`Platform`**：顶层入口，`Platform::current()` 返回当前平台实现，暴露各子系统（`fs()`/`clock()`/`threads()`/`vmem()`/`cpu()` …）。
- **`PlatformInfo`**：OS 名称/版本、架构（x86_64/aarch64/wasm32）、页大小、逻辑/物理核心数、是否移动/主机/Web。
- **`PlatformCaps`**：运行时能力位（见 §3），上层据此分支。
- **初始化契约**：引擎启动时 `prism_app` 第一步建立 `Platform`，所有子系统从它派生；禁止上层 crate 自行 `std::fs`/`libc` 调用（lint/评审约束）。

---

## 6. 文件系统与路径（VFS 友好）

- **中立路径类型**：统一正斜杠 + 规范化，隐藏 Windows `\` 与盘符差异；提供 `Path`/`PathBuf` 门面。
- **基础文件操作**：open/read/write/seek/metadata/`read_dir`/create_dir/remove/rename，async 友好（返回可被 `prism_tasks` 驱动的句柄或走阻塞线程池）。
- **内存映射 mmap**：大资产零拷贝映射（接 `prism_asset` 流送）；平台无 mmap（Web）时回退普通读。
- **文件监视 watch**：inotify/FSEvents/ReadDirectoryChangesW 统一门面，供热重载（资产/脚本/shader）。
- **标准目录**：可执行目录、用户数据、临时、缓存目录的平台正确解析。
- **VFS 分层点**：本 crate 只提供「真实 OS 文件」；虚拟文件系统（pak/打包/覆盖挂载）由 `prism_asset` 在其上搭建，但复用这里的 mmap/watch。

---

## 7. 高精度时钟（接 time）

- **单调时钟 `Instant`**：底层走 QPC（Win）/ `mach_absolute_time`（Apple）/ `clock_gettime(MONOTONIC)`（Linux），保证不回退，供帧计时。
- **TSC 快路径**：可选 RDTSC + 校准（频率标定 + 失配回退 OS 时钟），把取时间降到数纳秒，供高频剖析打点。
- **墙钟 `SystemTime`**：UTC/本地，供日志时间戳（不用于游戏逻辑）。
- **高分辨率睡眠/让出**：精确 sleep（Win 高精度定时器、`nanosleep`），供帧节流（接 `prism_time` 的帧步长控制）。
- **契约**：`prism_time` 的「定点步长 + alpha 插值」建立在本 crate 的单调时钟之上；逻辑步进用定点累加器，不直接用墙钟 → 支撑四方确定性的 time 一维。

---

## 8. 线程与同步原语（接 tasks）

- **线程创建/命名/优先级**：统一门面；线程命名供剖析器显示（接 `prism_diagnostic`）。
- **TLS（线程本地存储）**：跨平台 TLS 门面，供 `prism_tasks` 的每线程 worker 状态、帧分配器。
- **核心亲和性 affinity**：把 worker 绑定到指定逻辑核（接 §12 拓扑），减少迁移抖动，供 `prism_tasks` NUMA/大小核调度。
- **同步原语**：Mutex / RwLock（可走 `parking_lot`）、Condvar、Semaphore、Barrier、一次性初始化 Once；低争用快路径、无毒化语义。
- **futex/park 抽象**：底层线程挂起/唤醒原语，供 `prism_tasks` 的 work-stealing 调度器实现高效休眠-唤醒。
- **契约**：`prism_tasks` 不自建线程/锁，全部经由本 crate；绑核 + futex 是其性能底座。

---

## 9. 虚拟内存 / 页 / 大页 / 对齐分配（接 utils）

- **页级原语**：`reserve`（预留地址空间不提交）/ `commit`（提交物理页）/ `decommit` / `release`，对应 `VirtualAlloc` / `mmap(PROT_NONE)+mprotect`。
- **大页 hugepages**：可选 2MB/1GB 大页，降低 TLB miss，供大型连续缓冲（GPU 上传、流送环）。
- **对齐分配**：按缓存行/页/SIMD 对齐分配，供 `prism_utils` 的池/帧分配器与 `prism_math` SoA。
- **内存保护**：保护页（guard page）做栈溢出/越界检测（debug 档）；可执行页用于 JIT（如脚本后端，谨慎）。
- **内存信息**：物理/可用内存、本进程占用（供预算与 OOM 预警，接 `prism_diagnostic`）。
- **契约**：`prism_utils` 的分配器站在这里的页级原语上；引擎的内存预算系统用这里的内存信息做上限。

---

## 10. 动态库加载（热重载，接 script）

- **统一门面**：load/get_symbol/unload，封装 `LoadLibrary`/`dlopen`/`libloading`，隐藏符号修饰差异。
- **热重载支持**：加载到临时副本 + 版本化句柄 + 可卸载，供 `prism_script`（如原生 gameplay 模块）热替换；卸载安全由上层保证（无悬垂函数指针）。
- **能力位**：部分平台（iOS/主机/Web）禁用或限制动态加载 → `PlatformCaps` 标注，上层静态链接回退。
- **契约**：热重载是「开发期能力」，发行版可编译期关闭（`dynlib` feature off），全静态链接。

---

## 11. 进程 / 环境 / 命令行 / 标准流

- **命令行解析原料**：拿到原始 argv（含 Windows 宽字符正确解码），供 `prism_app` 的 CLI/配置层解析。
- **环境变量**：读/写门面，供配置覆盖（如 `PRISM_LOG`、`PRISM_RENDER_BACKEND`）。
- **子进程**：spawn/wait/pipe，供工具链（asset_bake / shader 编译外部进程）。
- **标准流**：stdout/stderr 门面（含 Windows 控制台 UTF-8 / 颜色），供 `prism_diagnostic` 日志落地。
- **退出/信号**：优雅退出钩子、信号/异常捕获入口（接 §16 崩溃捕获）。

---

## 12. CPU 特性探测与拓扑（接 math / tasks）

- **指令集探测**：运行时探测 SSE/AVX/AVX2/AVX-512 / NEON / SVE，供 `prism_math` 选择 SIMD 实现路径（dispatch）。
- **缓存行大小 / 页大小**：供 `prism_utils` 容器对齐与伪共享规避。
- **拓扑**：逻辑/物理核心、超线程、大小核（P/E core）、NUMA 节点，供 `prism_tasks` 线程池规模与绑核策略（接 §8 affinity）。
- **时间戳频率**：TSC 频率标定（接 §7）。
- **契约**：`prism_math` 的 SIMD dispatch 与 `prism_tasks` 的线程池尺寸均由本探测驱动；探测结果缓存一次、全局只读。

---

## 13. 原子与内存屏障抽象（no_std）

- **原子门面**：封装 `core::sync::atomic`，补齐引擎常用的原子计数、标志、无锁结构原料（供 `prism_utils` 并发容器、`prism_tasks` 调度器）。
- **内存屏障**：acquire/release/seq_cst 语义中立化，必要时提供平台专用 fence（极少用，默认走标准模型）。
- **CPU pause/yield**：自旋等待时的 `_mm_pause`/`YIELD`，降低自旋能耗与争用。
- **no_std 可用**：本章能力不依赖 OS，`no_std` 下即可用，是「核心计算层」可移植的保证之一。

---

## 14. 平台差异矩阵（Win/macOS/Linux/移动/主机/Web）

| 能力 | Windows | macOS/iOS | Linux/Android | 主机(抽象) | Web(wasm) |
|---|---|---|---|---|---|
| 文件系统 | ✅ Win32 | ✅ POSIX | ✅ POSIX | ✅ 定制 | 🟡 虚拟/受限 |
| mmap | ✅ | ✅ | ✅ | ✅ | ❌ 回退读 |
| 高精度时钟 | QPC/TSC | mach/TSC | MONOTONIC/TSC | 定制 | performance.now |
| 线程 | ✅ | ✅ | ✅ | ✅ | 🟡 Web Worker/SAB |
| 绑核 affinity | ✅ | 🟡 受限 | ✅ | ✅ | ❌ |
| 大页 | ✅ | 🟡 | ✅ | ✅ | ❌ |
| 虚拟内存预留 | ✅ | ✅ | ✅ | ✅ | 🟡 线性内存增长 |
| 动态库 | ✅ | 🟡(iOS 禁) | ✅ | 🟡 | ❌ |
| 子进程 | ✅ | ✅ | ✅ | 🟡 | ❌ |
| minidump | ✅ | ✅(mach) | ✅ | 定制 | ❌ |

- **原则**：能力缺失处由 `PlatformCaps` 运行时标注 + 上层优雅降级；Web/主机后端为可插拔抽象，细节随 SDK（本文不含任何受限 SDK 代码）。

---

## 15. 与 app / window / tasks / asset 集成

- **`prism_app`**：启动即建 `Platform`；命令行/环境变量/退出钩子/主循环节流（高精度 sleep）全走本 crate。
- **`prism_window` / `prism_input`**：窗口系统需要的进程/线程/时间原语由本 crate 供给（窗口与输入设备本身不在此）。
- **`prism_tasks`**：线程创建/绑核/futex/TLS/CPU 拓扑是其调度器全部底座。
- **`prism_asset`**：文件/mmap/watch 是其 VFS 与热重载底座。
- **`prism_time`**：单调时钟是其定点步长底座。
- **`prism_diagnostic`**：stdout/stderr、线程命名、内存信息、minidump 全来自本 crate。
- **`prism_math`**：SIMD dispatch 依赖本 crate 的 CPU 特性探测。

---

## 16. 崩溃捕获与 minidump（供 diagnostic）

- **异常/信号钩子**：安装 SEH（Win）/ `signal`+`sigaction`（POSIX）/ mach 异常端口（Apple）处理器，捕获段错误/非法指令/浮点异常。
- **minidump 生成**：崩溃时转储线程栈 + 寄存器 + 模块列表 + 关键内存，产出可供离线符号化的 dump（Breakpad/Crashpad 形态）。
- **越进程转储**：可选独立 handler 进程（更可靠，崩溃进程本身不可信），默认进程内轻量实现。
- **交接 diagnostic**：生成的 dump 路径/摘要交给 `prism_diagnostic`（§12 崩溃转储）落地与上报；本 crate 只负责「捕获+产出 dump」，不负责 UI/上报策略。
- **安全约束**：转储内容可能含敏感内存，默认不自动上传；路径/脱敏策略由 diagnostic 层决定。

---

## 17. 高级功能增补（AAA）

- **TSC 高频剖析时钟**：校准后的 RDTSC 让每帧数万次剖析打点近乎零成本，是 AAA 级细粒度 profiler（接 `prism_diagnostic`）的时间底座。
- **大页 + 虚拟内存区段管理**：为流送环、GPU 上传堆、ECS 大型列存提供低 TLB-miss 的连续地址空间，媲美引擎级自定义分配。
- **NUMA/大小核感知拓扑**：把 `prism_tasks` 的 worker 智能分布到正确核心（P-core 跑关键路径、E-core 跑后台），对标主机/高端 PC 调度。
- **保护页越界检测**：debug 档用 guard page 把越界写变成即时崩溃+精确栈，极大缩短内存 bug 定位时间。
- **可卸载热重载模块**：开发期原生 gameplay/工具模块热替换，迭代秒级（接 `prism_script`）。
- **跨进程崩溃转储**：独立 handler 保证崩溃现场可靠捕获，达到商业引擎的崩溃可观测水平。

所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 18. 性能工程

- **消除 OS 边界穿越**：高精度时钟走 TSC、自旋让出走 pause、锁走 parking_lot 快路径，热路径尽量不进内核。
- **页级直管**：reserve/commit 分离，避免大缓冲一次性物理提交；大页降 TLB miss。
- **绑核减抖动**：worker 固定核心，避免 OS 调度迁移毁掉缓存热度。
- **零拷贝 I/O**：mmap 大资产直接映射，省一次用户态拷贝（接流送）。
- **探测缓存**：CPU 特性/拓扑/页大小启动探测一次，全程只读，热路径零成本分支。
- **薄门面**：门面层尽量 `#[inline]` 透传到后端，不引入额外间接开销。

---

## 19. 易用性与 Bevy 迁移策略

- **贴 std 直觉**：`fs`/`time`/`thread` 门面与 `std` 同形，熟悉 Rust 即会用。
- **`Platform::current()` 一把入口**：所有子系统从单一入口派生，无需到处 `#[cfg]`。
- **mock 后端做单测**：上层逻辑用内存文件系统 + 可编程时钟测试，CI 无需真实 OS 行为。
- **优雅降级样板**：`if caps.contains(HUGEPAGES) { … } else { … }` 的标准范式，缺能力不崩。
- **Bevy 迁移**：Bevy 直接用 `std`/`winit` 散落各处；迁移时把 `std::fs`/`std::time`/`std::thread` 调用改为 `prism_platform` 门面，集中可控、可换后端、可测试。

---

## 20. crate 分层与模块布局

```
pkg/prism_platform/
  src/
    lib.rs            # re-export + prelude + Platform::current()
    platform.rs       # Platform / PlatformInfo / PlatformCaps
    fs/               # 文件系统
      file.rs dir.rs path.rs mmap.rs watch.rs dirs.rs
    clock.rs          # 单调/墙钟/TSC/高精度 sleep
    thread/           # 线程与同步
      spawn.rs tls.rs affinity.rs sync.rs park.rs
    vmem.rs           # 页/大页/对齐/保护/内存信息
    dynlib.rs         # 动态库加载（热重载）
    process.rs        # argv/env/子进程/标准流/退出
    cpu.rs            # 指令集/缓存/拓扑/TSC 频率
    atomics.rs        # 原子/屏障/pause（no_std）
    crash/            # 崩溃捕获 + minidump
      handler.rs minidump.rs
    backend/
      win.rs apple.rs posix.rs android.rs web.rs mock.rs
    prelude.rs
  features = ["std","backend-win","backend-apple","backend-posix",
             "backend-android","backend-web","crash","dynlib",
             "affinity","hugepages","parking-lot","mock"]
```

依赖：**无 Prism 上游**（与 `prism_math` / `prism_utils` 并列为根；可选依赖 `prism_utils` 的容器）；底层经典 `libc`/`rustix`/`windows-sys`/`parking_lot`/`libloading`。**不碰任何 `bevy_*`。**

---

## 21. 契约、不变量与版本化

- **单一入口不变量**：全引擎只有一个 `Platform` 实例；上层 crate 禁止直接 `std::fs`/`libc` 调用（评审 + 可选 lint 约束）。
- **单调时钟不回退**：`Instant` 保证单调不减，否则 time/剖析逻辑错乱；TSC 失配必回退 OS 时钟。
- **能力探测只读稳定**：`PlatformCaps`/CPU 探测启动后不变，上层可缓存分支判断。
- **页级原语配对**：reserve/commit/decommit/release 的生命周期配对不变量；错配即 UB，由门面 + 文档约束。
- **动态库卸载安全**：卸载前上层必须保证无悬垂函数指针/静态引用（契约在 `prism_script` 侧落实）。
- **版本化**：路径规范化规则、minidump 格式、能力位布局、后端 ABI 均为版本化契约；跨版本需迁移说明。

---

## 22. 路线图（M0–M6）与基准即规格

- **M0 核心无 OS 层**：`atomics` + `cpu`（指令集/拓扑）+ `clock`（单调+TSC）+ `Platform/Caps` 骨架 → `no_std` 可编译 + 探测正确性。**是 math SIMD dispatch / tasks 线程池尺寸的前置**。
- **M1 文件系统**：file/dir/path/标准目录（三桌面平台）→ 正确性（跨平台路径）+ 基准（吞吐）。**是 asset M0 前置**。
- **M2 线程与同步**：spawn/TLS/affinity/sync/park（三桌面平台）→ 正确性 + 基准（锁争用、绑核加速）。**是 tasks M0 前置**。
- **M3 虚拟内存**：reserve/commit/大页/对齐/保护页/内存信息 → 基准（页管理 vs malloc、大页 TLB 收益）。**是 utils 分配器前置**。
- **M4 mmap + watch + 动态库**：零拷贝映射、文件监视、热重载 → 热重载闭环（接 asset/script）。
- **M5 进程/环境/标准流 + 墙钟**：argv/env/子进程/stdout → 接 app CLI 与 diagnostic 日志。
- **M6 崩溃捕获 + Web/移动后端 + mock**：minidump（三桌面）+ 移动/Web 后端 + mock 后端 → 崩溃闭环（接 diagnostic）+ 跨平台 CI。

**基准即规格**：高精度时钟取时成本（ns）、单调性验证、锁争用吞吐、绑核缓存命中改善、页/大页分配 vs malloc、mmap 零拷贝加速、CPU 探测正确性、minidump 可符号化。核心价值在 **M0（math/tasks 探测前置）+ M1/M2/M3（asset/tasks/utils 三大地基前置）**。

---

## 23. 诚实边界与风险

- M0–M6 核心路线图**已全部落地并通过验证**：实现 + 单测（56 项 lib 测试全绿）+ 基准，`cargo clippy --all-targets` 零告警、`cargo test` 零失败。状态随代码演进；§24「AAA 高级功能增补」仍为 PLANNED，按本文优先级随消费方接线落地。
- **高风险项**：
  1. **跨平台后端维护成本（全程）**：每个 OS 一个后端，Win/Apple/Linux/Android/Web/主机差异巨大，门面抽象漏一处即到处 `#[cfg]` 反噬；必须严守「差异锁后端、门面保纯净」，并以 mock + 真机 CI 多平台持续验证。
  2. **崩溃捕获正确性（M6）**：信号/SEH/mach 异常处理器本身在崩溃上下文中运行，可用操作极受限（不能分配、不能加锁），写错即二次崩溃丢现场；需严格异步信号安全 + 可选越进程 handler。
  3. **TSC 可靠性（M0）**：TSC 频率随平台/节能/多核不一定稳定（非不变 TSC、核间漂移），误用即计时错乱；必须频率标定 + 失配检测 + 回退 OS 单调时钟。
  4. **动态库卸载悬垂（M4）**：热重载卸载旧库时若仍有函数指针/静态在用即崩溃；卸载安全契约复杂，发行版应可编译期关闭。
  5. **虚拟内存生命周期（M3）**：reserve/commit/release 错配、提前 release 正在用的区段即 UB；门面需强约束 + debug 保护页兜底。
  6. **Web/主机能力缺口（M6）**：wasm 无 mmap/绑核/动态库/子进程，主机受 SDK 限制；大量能力需优雅降级，上层若未按 `PlatformCaps` 分支则在这些平台崩溃。
- **与既有文档关系**：本 crate 与 `prism_math_design_zh.md` / `prism_utils_design_zh.md` 并列为依赖图根；高精度时钟是 `prism_time_design_zh.md` 定点步长底座；线程/绑核/futex/拓扑是 `prism_tasks_design_zh.md` 调度器底座；文件/mmap/watch 是 `prism_asset` VFS 底座；CPU 指令集探测驱动 `prism_math_design_zh.md` 的 SIMD dispatch；minidump 供 `prism_diagnostic_design_zh.md` §12 崩溃转储；命令行/环境/退出钩子/主循环节流接 `prism_app_design_zh.md`；页级原语供 `prism_utils_design_zh.md` 分配器。四方确定性中的 time 一维（单调时钟）由本 crate 托底。整体组件缺口见 `prism_engine_component_gap_zh.md`。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

## 24. AAA 高级功能增补（v0.2）

本章补齐顶级平台抽象层在真实 AAA 项目里缺一不可的能力。均 feature/能力位门控，默认不付成本；与前文的文件/时钟/线程/虚存/动态库内核互补。

### 24.1 高级异步 I/O（io_uring / IOCP / 批量提交）

AAA 的开放世界靠高吞吐流送喂饱 GPU，同步读文件远远不够：

- **统一异步 I/O 门面**：底层接 Linux `io_uring` / Windows IOCP / Apple kqueue，暴露「提交队列 + 完成队列」模型，一次系统调用批量提交 N 个读请求。
- **零拷贝 + 对齐直读**：绕过页缓存的 direct I/O + 页对齐缓冲（接 §9 虚存），大资产直落目标内存。
- **优先级 I/O**：关键流送（镜头前方区块）高优先，预取低优先，接 `prism_tasks` §24.5 异步 I/O 桥。
- **I/O 带宽预算**：限速 + 背压，防流送挤垮磁盘/网络，供 `prism_asset` 流送调度。

### 24.2 虚拟内存进阶（稀疏堆 / 按需提交 / GPU 共享内存）

- **稀疏虚拟堆**：预留巨大地址空间（如 TB 级 world），仅对访问到的页按需 commit（commit-on-fault），供大世界/虚拟纹理稀疏驻留。
- **环形流送堆**：虚拟地址镜像映射（magic ring buffer），环形缓冲跨界读写无需分支，供音频/流送环。
- **GPU 共享 / 可见内存**：抽象 host-visible / upload / readback 内存域（接 `prism_render_driver` RHI），统一 CPU-GPU 内存桥。
- **内存域标签**：虚存区段带类别标签，供 `prism_diagnostic` §24.3 内存图谱按域可视化。

### 24.3 混合核 / NUMA / 能耗感知调度钩子

- **拓扑即数据**：导出完整 CPU 拓扑（P/E-core、SMT、NUMA 节点、缓存共享域，接 §12），供 `prism_tasks` 构建绑核策略。
- **能耗/热感知**：读取平台能耗状态/热节流信号（移动/主机关键），提供钩子让调度器在过热时降背景负载、让关键路径上性能核。
- **QoS 类映射**：把引擎车道（Critical/Background，tasks §24.1）映射到 OS QoS 类（Apple QoS、Win Quality of Service），OS 据此派核与调频。
- **省电模式**：移动端前后台切换、低电量时的降频/降帧钩子（接 `prism_time` 帧节流）。

### 24.4 高级崩溃观测（跨进程 Crashpad / 稳定堆栈哈希分桶）

§16 之上的工业级崩溃可观测：

- **跨进程 handler**：独立 Crashpad 形态 handler 进程，崩溃进程不可信也能可靠转储（主崩溃进程只触发，转储由旁进程做）。
- **稳定堆栈哈希分桶**：对崩溃调用栈规整后算稳定哈希，相同根因的崩溃自动聚类，供 `prism_diagnostic` §24.7 最小化上报。
- **可控转储粒度**：mini / full dump 可选，含/不含堆内存，平衡体积与可调试性；默认脱敏（去明文）。
- **符号化离线化**：转储只带模块+偏移，符号在后端离线还原，发行版不带符号表。

### 24.5 安全加固探测（ASLR / DEP / CFG / 代码签名）

- **缓解措施探测/启用**：探测并（可）启用 ASLR、DEP/NX、Control Flow Guard、stack canary，发行版默认开。
- **代码签名 / 完整性校验**：探测二进制签名、校验动态库签名（接 §10），拒载未签名模块（反注入/反作弊基础设施，非作弊本身）。
- **沙箱能力探测**：探测平台沙箱/权限模型（移动/主机/Web），上层据能力位调整文件/网络访问。
- **安全约束**：本层只做防御性加固与探测，不含任何绕过/攻击/反调试对抗代码。

### 24.6 平台能力数据库与降级矩阵

- **能力数据库**：把 §3 `PlatformCaps` 扩展成「能力 → 平台 → 支持度/替代路径」结构化数据，上层查询即得降级方案。
- **声明式降级**：`require(HUGEPAGES).or_fallback(normal_alloc)` 式 API，缺能力自动走回退，杜绝「某平台漏判直接崩」。
- **特性门控报告**：启动期输出本平台能力报告（接 `prism_diagnostic`），一眼看清哪些 AAA 路径在当前平台降级。

### 24.7 Web / 主机后端进阶

- **Web 多线程**：基于 SharedArrayBuffer + Web Worker 的线程池（接 §8），wasm 原子，使 `prism_tasks` 在浏览器可并行；无 SAB 时退单线程。
- **Web 存储/网络 I/O**：IndexedDB/fetch 适配 §6 文件门面与 §24.1 异步 I/O，使资产流送在 Web 可用。
- **主机内存域**：主机统一内存/独立显存域抽象（接 §24.2 GPU 共享内存），细节随 SDK（本文不含任何受限 SDK 代码）。
- **主机 I/O**：主机高速存储 API 适配 §24.1 异步 I/O 门面。

### 24.8 诚实边界

本章全部为 PLANNED 设计目标，无代码。**24.1 异步 I/O + 24.2 虚存进阶**是 `prism_asset` 流送与 RHI 最先依赖的能力，建议随 M3/M4 优先落地；24.3 调度钩子随 `prism_tasks` M5 落地；24.4 跨进程崩溃随 M6 崩溃闭环落地；24.5 安全加固随发行加固落地；24.6 能力数据库贯穿始终（降级正确性的保证）；24.7 Web/主机后端随对应平台接线落地。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。
