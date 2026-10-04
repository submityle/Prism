# Prism Time 顶级次世代 AAA 级时间 / 时钟 / 固定步长设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **三时钟（Real/Virtual/Fixed）+ 固定步长积分 + 时间膨胀 + 计时器 + 确定性时钟 + 网络时间同步** 内核设计。它是 `bevy_time` 的自研替代，是 `prism_app` 主循环、`prism_ecs` 固定步、物理/动画/网络的时间基座。
> 借形态不抄码。借鉴：
> - **三时钟分离**：Bevy `Time<Real/Virtual/Fixed>`、Timer/Stopwatch
> - **时间膨胀/缩放**：Unity（timeScale/unscaledTime/fixedDeltaTime）、Unreal（time dilation、`FApp::DeltaTime`）
> - **固定步长积分**：Glenn Fiedler「Fix Your Timestep」（累加器 + 插值 alpha + 死亡螺旋熔断）
> - **高精度时钟**：平台单调时钟（`QueryPerformanceCounter`/`mach_absolute_time`/`clock_gettime`）
> - **网络时间**：GGPO/Quantum 的 tick 对齐、NTP 式时钟同步、插值延迟缓冲
> 本文为纯经典时间/积分路线，**不含任何 AI/ML 内容**。

- 版本: v0.2（核心 M0–M6 已落地并验证；§24 高级增补仍为设计阶段；v0.1→v0.2 新增第 24 章「AAA 高级功能增补」：输入+时间录制回放/帧节奏与低延迟(VRR/Reflex 形态)/帧预算驱动自适应质量/长会话高精度与漂移修正/游戏内定时调度器/挂起恢复与后台暂停/多世界时间域隔离/确定性时间审计；均为 PLANNED，无代码）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: `prism_platform`（单调高精度时钟），可选 `prism_math`（有理/定点时间）、`prism_diagnostic`
- 层级定位: ECS 文档 L3「运行时服务」；App 文档 §8 固定步长的时钟供给方
- 明确约束: 核心 `no_std + alloc`（时钟源经平台注入）；`std` / `determinism` / `network` / `trace` 为 feature；**不依赖任何 `bevy_*` crate**

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier）
4. 分层架构
5. 核心模型：Clock / Instant / Duration / Time<Kind>
6. 三时钟：Real / Virtual / Fixed
7. 固定步长积分（累加器 / 插值 alpha / 死亡螺旋熔断）
8. 时间膨胀 / 暂停 / 慢动作 / 子弹时间
9. 计时器与秒表（Timer / Stopwatch / 冷却 / 节流）
10. 确定性时钟（有理/定点步长 / 可回放）
11. 网络时间同步（server tick / 时钟偏移 / 插值延迟缓冲）
12. 平滑 delta（抖动平滑 / 最大 delta 钳制）
13. 局部时间域（per-world / per-system 时间缩放）
14. 时间轴 / 序列器时间源（过场 / 动画）
15. 与 App / ECS / 物理 / 动画 / 网络集成
16. 可观测性（帧预算 / 时间统计）
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

时间看似简单，却是**最易出 bug 的地基**：帧率无关移动、暂停不冻物理、慢动作不乱动画、低 tickrate 不抖、网络不漂移、回滚可重放——全系于一套严谨的时间模型。`prism_time` 把「墙钟 / 可控游戏时间 / 确定性固定步」三者**显式分离**，让每个子系统取用正确的时钟，杜绝「用错 delta」类顽疾。

**一句话定位**：`prism_time` 是 Prism 的「权威时间基座」——Real/Virtual/Fixed 三时钟分离 + 固定步长确定性心跳 + 时间膨胀 + 高精度单调计时 + 网络时钟同步；供 App 主循环驱动、供物理/网络吃固定步、供表现层吃插值 alpha。

四条总目标（按权重）：

1. **性能**：时钟推进 O(1) 原子/标量运算；高精度单调时钟缓存频率；无堆分配热路径。
2. **效果（能力）**：固定步长支撑确定性物理/回滚网络；时间膨胀支撑子弹时间/暂停；插值 alpha 消抖；网络时钟同步抗漂移。
3. **易用**：`time.delta_secs()` 一把梭；暂停/缩放改一个字段全局生效；与 `bevy_time` 近乎一致。
4. **可移植 + 档位化**：核心 `no_std`（时钟源注入）；确定性/网络按 feature 裁剪。

非目标：不做日历/时区（那是 gameplay 的事，可另建）；不保证墙钟跨机绝对一致（只在 `determinism` 档保证固定步逻辑一致）。

---

## 2. 参考产品取舍

| 产品 | 吸收 | 规避 |
|---|---|---|
| Bevy Time | `Time<Real/Virtual/Fixed>` 三分、Timer/Stopwatch、泛型时钟上下文 | —— |
| Unity | timeScale/unscaledTime/fixedDeltaTime 清晰分层 | C#、maximumDeltaTime 隐式默认坑 |
| Unreal | time dilation、`FApp` delta、暂停语义 | 专有 |
| Fix Your Timestep | 累加器固定步 + 插值 alpha + 死亡螺旋熔断 | —— |
| GGPO/Quantum | tick 对齐、确定性步、回滚时间 | 闭源网络栈 |
| 平台单调时钟 | 高精度单调、频率缓存、单调不回退 | 平台 API 差异（经 prism_platform 隔离） |

综合：**Bevy 三时钟 + Unity 分层清晰 + Fix-Your-Timestep 确定心跳 + GGPO 网络 tick** 四支柱，叠加 **时间膨胀 + 平滑 delta + 网络时钟同步 + 局部时间域** 的 AAA 能力层。

---

## 3. 档位化（capability / quality tier）

| 维度 | 说明 | 示例 |
|---|---|---|
| **capability** | 环境 | 高精度计时器可用性、单调时钟分辨率 |
| **quality tier** | 形态 | server（权威定 tick）/ mobile（省电可变步）/ desktop（固定步 + 插值） |
| **feature flag** | 裁剪 | `determinism` / `network` / `trace` |

目标：server 档只需固定步心跳 + 网络时钟；桌面叠加插值 alpha + 平滑 delta；移动端省电可变步。

---

## 4. 分层架构

```
L4  集成     物理吃 Fixed / 表现吃 alpha / 网络吃 tick / 动画吃 Virtual
L3  计时器   Timer / Stopwatch / 冷却 / 节流 / 时间轴
L2  时钟语义 Time<Real> / Time<Virtual>（膨胀/暂停）/ Time<Fixed>（累加器）
L1  时间源   单调 Instant / Duration / 平滑 / 钳制 / 网络偏移
L0  平台     prism_platform：单调高精度时钟、频率
```

依赖严格向下；L0 时钟源经平台注入，L1–L3 可 `no_std + alloc`。

---

## 5. 核心模型

```rust
pub struct Instant(u64);          // 单调 tick（平台频率换算），不回退
pub struct Duration { secs: u64, nanos: u32 }

pub struct Time<T: TimeKind> {
    delta: Duration,              // 本次推进步长
    elapsed: Duration,            // 累计
    delta_secs: f32, delta_secs_f64: f64,
    context: T,                   // Real/Virtual/Fixed 各自上下文
}

pub trait TimeKind { /* Real / Virtual / Fixed */ }
```

- 泛型 `Time<T>` 复用同一套 delta/elapsed 访问器（Bevy 形态），`T` 决定推进语义。
- `f32`（够用、快）与 `f64`（长时程精度）两版 delta 都提供，避免长会话 `f32` 累积误差。

---

## 6. 三时钟

| 时钟 | 语义 | 谁用 |
|---|---|---|
| `Time<Real>` | 墙钟，单调，不受暂停/缩放影响 | profiler、网络 RTT、UI 动画、超时 |
| `Time<Virtual>` | 游戏时间：可暂停、可缩放（膨胀） | gameplay、动画、粒子、相机 |
| `Time<Fixed>` | 确定性固定步长（由 Virtual 累加驱动） | 物理、网络仿真、确定性 gameplay |

默认 `Time`（无泛型）在 `Update` 指向 Virtual、在 `FixedMain` 指向 Fixed——与 App §7 阶段对应，用户多数时候 `time.delta_secs()` 即正确。

---

## 7. 固定步长积分

与 App §8 共用同一累加器（时钟侧实现，App 侧驱动）：

```
virtual_delta = real_delta * time_scale            // 受暂停(=0)/缩放影响
virtual_delta = min(virtual_delta, max_delta)      // 死亡螺旋熔断
Time<Virtual>.advance(virtual_delta)
accumulator += virtual_delta
while accumulator >= fixed_dt {                    // 跑 0..N 次 FixedMain
    Time<Fixed>.advance(fixed_dt)
    accumulator -= fixed_dt
    if substeps++ > max_substeps { break }         // 熔断
}
Time<Fixed>.overstep_fraction = accumulator / fixed_dt   // = 插值 alpha
```

- `alpha`（overstep fraction）供表现层对「上一固定态↔当前固定态」插值（§12 + transform 插值），消除低 tickrate 视觉抖动。
- 暂停 = `time_scale = 0`：Virtual/Fixed 冻结，Real 照走（UI/网络仍活）。

---

## 8. 时间膨胀 / 暂停 / 慢动作

- **`time_scale`**：Virtual 相对 Real 的倍率。0=暂停、0.1=子弹时间、2.0=快进。
- **分层缩放**：全局 scale × 局部域 scale（§13），如全局正常但某特效域慢放。
- **平滑过渡**：scale 变化可带缓动，避免瞬间突变（子弹时间入场/退场）。
- **不受影响集**：UI 动画、profiler、网络超时走 Real，暂停时仍响应。

---

## 9. 计时器与秒表

- **Timer**：倒计时/循环（`finished()`/`just_finished()`/`times_finished_this_tick()`），吃指定时钟（Virtual 默认，可选 Real）。
- **Stopwatch**：累计计时，可暂停/重置。
- **冷却/节流**：技能 CD、输入节流、周期触发，建在 Timer 上。
- 全部以 `tick(delta)` 推进，确定性友好（不读全局状态）。

---

## 10. 确定性时钟

`determinism` 档：固定步长必须**精确可复现**——用有理数（分子/分母，如 1/60）或定点表示 `fixed_dt`，避免 `f32` 1/60 的二进制不精确累积漂移。elapsed 用整数 tick 计数（`tick * fixed_dt`）而非浮点累加。回滚时时钟可随 World 快照一并还原到任意历史 tick，重放到当前（接 ECS §14 / App §15）。

---

## 11. 网络时间同步

`network` 档：

- **server tick 权威**：固定 tickrate（如 60Hz），客户端对齐到 server tick 编号。
- **时钟偏移估计**：类 NTP 往返测量估 client↔server 时钟偏移 + RTT，平滑收敛，抗抖动。
- **插值延迟缓冲**：远端实体状态缓冲若干 tick（如 100ms），在「过去时间」平滑插值渲染（对抗抖动/丢包），本地预测走当前 tick。
- 与回滚（ECS §14）协同：预测用当前 tick，确认用 server tick 回滚重放。

---

## 12. 平滑 delta

- **抖动平滑**：对 real_delta 做滑动平均/中值，供相机/平滑移动用，消除帧时间毛刺（但物理仍吃原始固定步，不被平滑污染）。
- **最大 delta 钳制**：`max_delta`（如 0.25s）防断点调试/卡顿后一帧巨 delta 把物体瞬移穿墙（也是死亡螺旋熔断的一环）。

---

## 13. 局部时间域

- **per-world**：大世界分区的不同 World 可各自时间缩放（如暂停某后台模拟 World）。
- **per-system/per-entity**：局部慢放域（某 Boss 技能让局部时间变慢），通过域缩放因子注入对应 system/查询。
- 域缩放叠加在全局 scale 之上，确定性档下域因子也须可复现。

---

## 14. 时间轴 / 序列器时间源

为过场动画/Timeline/动画播放提供可寻址时间源：可 seek（跳到 t）、可缩放、可反向、可循环；与 `Time<Virtual>` 解耦（过场可独立于游戏时间推进），服务 `prism_anim_runtime` 与编辑器序列器。

---

## 15. 与 App / ECS / 物理 / 动画 / 网络集成

- **App**（§7/§8）：App 主循环每帧推进 Real→Virtual，驱动累加器跑 FixedMain，把 alpha 暴露给表现。
- **ECS**：时钟作为资源注入；`Time<Fixed>` 在 FixedMain 调度、`Time<Virtual>` 在 Update。
- **物理**：吃 `Time<Fixed>`，确定性 substep。
- **动画/粒子**：吃 `Time<Virtual>`（随暂停/慢放），表现插值吃 alpha。
- **网络**：吃 server tick + 时钟同步 + 插值延迟缓冲。

---

## 16. 可观测性

- **帧预算**：本帧墙钟耗时 vs 目标帧时间，超预算告警（接 App §16 帧统计）。
- **时间统计**：FixedMain substep 数、time_scale、累加器水位、网络时钟偏移/RTT。
- **时间 trace**：各时钟 delta 曲线，定位卡顿/抖动来源。

---

## 17. 高级功能增补

- **可变 fixed_dt**：按 quality tier/网络 tickrate 配置固定步频率（移动 30Hz / 桌面 60Hz / 竞技 128Hz）。
- **重放确定步**：录制每固定步输入 + 周期快照，逐 tick 重放（接 App §15）。
- **帧步进调试**：暂停下按帧/按固定步单步推进（配合 ECS §23.4 system 单步）。
- **时间事件**：到点触发事件（闹钟/定时生成），确定性档下按 tick 触发。
- **长时程精度**：elapsed 用 `f64`/整数 tick，避免长会话 `f32` 累积误差。
- **自适应 time_scale 过渡**：子弹时间进出带缓动曲线。

---

## 18. 性能工程

- **O(1) 推进**：每帧时钟推进是标量运算，无分配、无锁（单线程主循环推进，读为只读资源）。
- **频率缓存**：平台时钟频率启动期查一次，之后纯乘法换算。
- **f32/f64 双版**：热路径用 f32，长时程/确定性用 f64/整数 tick。
- **无全局可变状态**：Timer `tick(delta)` 显式传入，便于并行只读与确定性。

诚实边界：网络时钟同步收敛性、插值缓冲手感需真实网络条件测试；确定性步长跨平台一致需定点路径验证，标注 PLANNED。

---

## 19. 易用性与 Bevy 迁移策略

对外 API 贴近 `bevy_time`：`Time`、`Time<Real/Virtual/Fixed>`、`time.delta_secs()`/`elapsed_secs()`、`Timer`/`Stopwatch`、`time.set_relative_speed()`（time_scale）、`Fixed::overstep_fraction()`（alpha）。

迁移：
1. `prism_time::prelude` 近同名导出。
2. App 主循环改用本 crate 推进三时钟。
3. 物理/动画/网络按时钟种类取用（多数只换 import）。

---

## 20. crate 分层与模块布局

```
pkg/prism_time/
  src/
    instant.rs duration.rs     # 单调 Instant、Duration、平台频率换算
    time.rs                    # Time<T> 泛型、delta/elapsed 访问器
    real.rs virtual_.rs fixed.rs   # 三时钟语义
    fixed_loop.rs              # 累加器 + alpha + 死亡螺旋熔断（App 驱动）
    scale.rs                   # time_scale、暂停、分层/局部缩放、缓动过渡
    timer.rs stopwatch.rs      # Timer/Stopwatch/冷却/节流
    smooth.rs clamp.rs         # 平滑 delta、最大 delta 钳制
    determinism.rs             # 有理/定点步长、整数 tick、重放还原（determinism 档）
    network.rs                 # server tick、时钟偏移估计、插值延迟缓冲（network 档）
    timeline.rs                # 可寻址时间源（过场/序列器）
    diagnostics.rs             # 帧预算、时间统计、trace
  features = ["std","determinism","network","trace"]
```

依赖：`prism_platform`（单调时钟），可选 `prism_math`/`prism_diagnostic`。**不碰任何 `bevy_*`。**

---

## 21. 契约、不变量与版本化

- **单调不回退**：`Instant` 严格单调；`Time<Real>` elapsed 只增。
- **暂停语义**：`time_scale=0` 冻结 Virtual/Fixed，Real 照走。
- **固定步精确**：`determinism` 档 `fixed_dt` 用有理/定点，elapsed=整数 tick × dt。
- **alpha 范围**：`overstep_fraction ∈ [0,1)`。
- **版本化契约**：`Time<Kind>` 字段、`fixed_dt`/`max_delta`/`max_substeps` 配置、网络 tick/偏移格式、Timer 行为。

---

## 22. 路线图（M0–M6）与基准即规格

- **M0 时间源**：平台单调 Instant/Duration + 频率换算 + `Time<Real>` → 单测（单调/换算正确）。
- **M1 三时钟**：`Time<Virtual>`（scale/暂停）+ `Time<Fixed>` + 默认 Time 上下文切换。
- **M2 固定步长**：累加器 + alpha + 死亡螺旋熔断 + 最大 delta 钳制（与 App §8 对接）。
- **M3 计时器**：Timer/Stopwatch/冷却/节流 + 平滑 delta。
- **M4 确定性**：有理/定点步长、整数 tick、重放还原。
- **M5 网络 + 局部域**：server tick、时钟偏移、插值延迟缓冲、per-world/局部缩放、缓动过渡。
- **M6 集成/工具**：App/ECS/物理/动画/网络接线、时间轴时间源、帧步进调试、帧预算/统计；bevy_time 兼容 prelude。

**基准即规格**：时钟推进开销、jitter 输入下固定步数正确、alpha 单调性、确定性双跑 tick/elapsed 位等价、网络时钟收敛曲线。核心价值在 **M2（固定步）+ M4（确定性）+ M5（网络）**。

---

## 23. 诚实边界与风险

- M0–M6 核心路线图**已全部落地并通过验证**：实现 + 单测（117 项 lib 测试全绿）+ 基准，`cargo clippy --all-targets` 零告警、`cargo test` 零失败。状态随代码演进；§24「AAA 高级功能增补」仍为 PLANNED，按本文优先级随消费方接线落地。
- **高风险项**：
  1. **死亡螺旋熔断参数（M2）**：`max_delta`/`max_substeps` 错配会在卡顿时表现为慢放或穿墙；须按内容压测标定，并与 App §8 保持单一真相（避免两处各算一套累加器）。
  2. **确定性步长（M4）**：`f32` 的 1/60 不精确会长时程漂移；必须有理/定点 + 整数 tick，且跨平台一致要定点数学路径验证。
  3. **网络时钟同步（M5）**：偏移估计抖动/突变会导致远端实体抖动或瞬移；平滑收敛策略与插值缓冲长度需真实弱网测试。
  4. **f32 累积误差**：长会话 elapsed 用 f32 会丢精度（动画/计时错位）；须 f64/整数 tick 兜底。
  5. **暂停语义边界**：哪些系统吃 Real（不冻）哪些吃 Virtual（冻）须全局约定清楚，错配会出现「暂停时 UI 卡死」或「暂停时物理仍动」。
- **与既有文档关系**：本 crate 的固定步累加器与 App §8 是**同一机制的两侧**（时钟实现 + 主循环驱动），须单一真相；`alpha` 供 transform 插值（§12）与表现层；确定性步长与 ECS §14、网络 tick 与 `prism_replication` 契约一致。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 24. AAA 高级功能增补（v0.2）

本章补齐顶级时间系统常被忽视、却在真实 AAA 项目里缺一不可的能力。均 feature/档位门控，默认不付成本；与前文三时钟、固定步、时间膨胀、确定性时钟、网络同步互补。

### 24.1 输入 + 时间录制回放

把每帧的输入与时间推进记录成轨道,可逐帧回放,用于 bug 复现、回归测试、过场录制:

- 录制 `(tick, fixed_dt, 输入快照)`;回放时**用录制的 delta 喂时钟**,不读墙钟,保证严格复现。
- 与 §确定性时钟 + ECS 回滚联动:同输入 + 同步长 → 同结果(位等价),是确定性调试的闭环工具。
- 供自动化测试:录一段玩法,CI 回放断言状态哈希一致。

### 24.2 帧节奏与低延迟（Frame Pacing / Reflex 形态）

稳定帧时间比高平均帧率更重要（抖动比低帧率更伤手感）：

- **帧节奏**:平滑 present 间隔,避免「微卡顿」(micro-stutter);对标主机 frame pacing。
- **低延迟渲染提交**:延迟采样输入到尽可能靠近模拟开始(late-latching),压缩 input-to-photon 延迟(对标 NVIDIA Reflex / AMD Anti-Lag 的「按需开工」理念——不提前排队过多帧)。
- 与 `prism_app` 主循环、RHI present 协同:时间系统提供「下一 present 预估时刻」供提交对齐。

### 24.3 可变刷新率（VRR）感知

G-Sync/FreeSync/VRR 下显示刷新非固定:

- 时间系统感知显示器可变刷新窗口,把 present 调度到最优时刻,配合帧节奏(§24.2)。
- 固定步仿真(§7)与可变显示解耦,表现层插值 alpha(§12)吸收刷新抖动。

### 24.4 帧预算驱动的自适应质量

时间系统把「本帧用了多少/还剩多少毫秒」作为**反馈信号**喂给质量调节：

- 导出帧预算与各阶段耗时(接 §16 可观测性、`prism_profiler`)。
- 子系统据此动态调节:动态分辨率、LOD 偏置、阴影级联数、粒子上限、`prism_tasks` 的 `Background` 车道顺延(tasks §24.1)。
- 闭环:超预算→降质保帧率;富余→回升画质,平滑迟滞避免振荡。

### 24.5 长会话高精度与漂移修正

24/7 服务器、长流程单机会话的时间精度:

- `elapsed` 用 f64/整数 tick 累加,杜绝 f32 长时程精度丢失(动画/计时错位)。
- 单调时钟溢出/回绕处理(平台计数器位宽有限)。
- 可选对墙钟做缓慢漂移修正(服务器长跑与 NTP 对齐),但不破坏单调与固定步确定性。

### 24.6 游戏内定时调度器

统一的「在 T 时刻/经过 D 后/每隔 P」回调调度(高于裸 Timer):

```rust
time.schedule_after(Duration::from_secs(3), |w| spawn_wave(w));
time.schedule_every(Duration::from_millis(500), |w| tick_regen(w));
```

- 走 Virtual 时钟(受暂停/缩放影响)或 Real 时钟(UI/网络心跳)可选。
- 支持取消句柄、合并、确定性排序(回放一致)。供 gameplay 冷却、刷怪、buff 到期。

### 24.7 挂起 / 恢复与后台暂停

进程挂起（移动端切后台、主机休眠、窗口最小化）后恢复：

- 恢复时**不把挂起时长算成一帧巨 delta**（否则物理穿墙/动画跳跃）——钳制或丢弃该帧 delta（接 §12 最大 delta 钳制）。
- 可配置后台行为:暂停 Virtual(游戏冻结)、Real 照走(网络心跳不断)、或低频后台更新。
- 平台挂起/恢复事件由 `prism_platform`/`prism_app` 转发给时间系统。

### 24.8 多世界时间域隔离与确定性审计

- **多世界**:每个 World（主世界、编辑器预览、服务器子应用）持独立时间上下文,互不干扰(接 App §子应用)。
- **确定性审计**(`determinism` + `trace` 档):记录每 tick 的 `dt`/累加器/子步数,双跑比对定位首个发散 tick,是确定性回归的诊断利器。

### 24.9 诚实边界

本章全部为 PLANNED 设计目标,无代码。**24.1 录制回放 + 24.8 确定性审计**是确定性系统(物理/网络)的调试基石,建议随 M4 落地;24.2 帧节奏 + 24.3 VRR + 24.7 挂起恢复随 M2 主循环/平台接线落地;24.4 自适应质量随 `prism_profiler`/渲染反馈落地;24.5 长会话精度贯穿始终;24.6 调度器随 gameplay 层落地。帧节奏/低延迟须真实硬件(含主机/VRR 显示器/移动端)验证,设计阶段无法断言数值。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码;仅借鉴公开架构形态与经典数值。

