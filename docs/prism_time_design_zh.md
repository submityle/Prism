# Prism Time 顶级次世代 AAA 级时间 / 时钟 / 固定步长设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **三时钟（Real/Virtual/Fixed）+ 固定步长积分 + 时间膨胀 + 计时器 + 确定性时钟 + 网络时间同步** 内核设计。它是 `bevy_time` 的自研替代，是 `prism_app` 主循环、`prism_ecs` 固定步、物理/动画/网络的时间基座。
> 借形态不抄码。借鉴：
> - **三时钟分离**：Bevy `Time<Real/Virtual/Fixed>`、Timer/Stopwatch
> - **时间膨胀/缩放**：Unity（timeScale/unscaledTime/fixedDeltaTime）、Unreal（time dilation、`FApp::DeltaTime`）
> - **固定步长积分**：Glenn Fiedler「Fix Your Timestep」（累加器 + 插值 alpha + 死亡螺旋熔断）
> - **高精度时钟**：平台单调时钟（`QueryPerformanceCounter`/`mach_absolute_time`/`clock_gettime`）
> - **网络时间**：GGPO/Quantum 的 tick 对齐、NTP 式时钟同步、插值延迟缓冲
> 本文为纯经典时间/积分路线，**不含任何 AI/ML 内容**。

- 版本: v0.2（核心 M0–M6 已落地并验证；§24 高级增补：24.1 录制回放 + 24.4 帧预算自适应质量 + 24.5 长会话漂移修正 + 24.6 游戏内定时调度器 + 24.8 多世界/确定性审计已落地，24.7 挂起/恢复已落地纯 CPU 可做部分（时间戳层确定性修正，OS 信号接线仍 PLANNED），24.2 帧节奏与低延迟 + 24.3 VRR 已落地纯 CPU 调度数据模型（`pacing`）；v0.1→v0.2 新增第 24 章「AAA 高级功能增补」：输入+时间录制回放/帧节奏与低延迟(VRR/Reflex 形态)/帧预算驱动自适应质量/长会话高精度与漂移修正/游戏内定时调度器/挂起恢复与后台暂停/多世界时间域隔离/确定性时间审计）
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

- M0–M6 核心路线图**已全部落地并通过验证**：实现 + 单测（lib 测试全绿，含 §24.1/§24.8 新增 20 项）+ 基准，`cargo clippy --all-targets` 零告警、`cargo test` 零失败。状态随代码演进；§24「AAA 高级功能增补」中 24.1/24.4/24.5/24.6/24.8 已落地、24.7 已落地纯 CPU 可做部分（OS 挂起信号接线仍 PLANNED），24.2 帧节奏/低延迟 + 24.3 VRR 已落地纯 CPU 调度数据模型（`pacing`，真实 present 时间戳采集与 RHI 提交/显示 VRR 范围读取属 `prism_app`/平台接线）。
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

### 24.1 输入 + 时间录制回放 ✅ 已交付（`recording`）

把每帧的输入与时间推进记录成轨道,可逐帧回放,用于 bug 复现、回归测试、过场录制:

- 录制 `(tick, fixed_dt, 输入快照)`;回放时**用录制的 delta 喂时钟**,不读墙钟,保证严格复现。
- 与 §确定性时钟 + ECS 回滚联动:同输入 + 同步长 → 同结果(位等价),是确定性调试的闭环工具。
- 供自动化测试:录一段玩法,CI 回放断言状态哈希一致。

**交付状态（已落地）**：`pkg/prism_time/src/recording/`（`mod.rs` 共享数据 + `record.rs` 录制态 + `replay.rs` 回放态）。
- `RecordedFrame<I>` 记录每帧 `(dt, 输入快照, 随机种子)`；`Recording<I>` 为有序确定性时间线（纯数据、`Clone`、可被两条独立回放复用做双跑比对）。
- **两态逐帧投喂**：`Recorder<I>` 为录制态（`record_frame` 逐帧追加 → `finish()` 产出 `Recording`）；`Player<I>` 为回放态（`next_frame()` 逐帧投喂，取回放 `dt` 喂时钟而非读墙钟，`seek`/`reset` 支持逐帧拖拽调试）。
- **严格复现**：回放用录制 delta 驱动 `TickClock`，与原跑逐帧 tick/累加器位等价；同种子经确定性 PRNG 复现同一随机流；配合 `multiworld` 审计可录一段玩法、CI 回放断言状态哈希逐帧一致（闭环）。
- 纯整数/`Duration` 运算，`no_std + alloc`（时间线用 `alloc::Vec`，热路径无额外分配）。单测见 `tests_record_replay.rs`（录→放逐帧一致、种子复现、时钟逐帧对齐、拖拽、空录制与边界、闭环哈希一致）。

### 24.2 帧节奏与低延迟（Frame Pacing / Reflex 形态） ✅ 已交付（`pacing`）

稳定帧时间比高平均帧率更重要（抖动比低帧率更伤手感）：

- **帧节奏**:平滑 present 间隔,避免「微卡顿」(micro-stutter);对标主机 frame pacing。
- **低延迟渲染提交**:延迟采样输入到尽可能靠近模拟开始(late-latching),压缩 input-to-photon 延迟(对标 NVIDIA Reflex / AMD Anti-Lag 的「按需开工」理念——不提前排队过多帧)。
- 与 `prism_app` 主循环、RHI present 协同:时间系统提供「下一 present 预估时刻」供提交对齐。

**交付状态（已落地）**：`pkg/prism_time/src/pacing/`（`mod.rs` 共享整数 EMA + `pacer.rs` 帧节奏/低延迟 + `vrr.rs` VRR）。均为纯 CPU、确定性、整数运算的**调度数据模型**，不读墙钟、不碰 OS/显示/RHI。
- **帧节奏锁相（`FramePacer`，§24.2）**：喂入调用方给的 present 时间戳流，维持一条按 `target_interval_ns` 稳步推进的**预测 present 时刻**（`next_present_ns`）。小抖动被吸收（预测不随单帧抖动漂移，杜绝微卡顿外溢到提交时序），仅当偏差超过 `resync_threshold_ns`（默认半帧）或时间戳非单调才**硬重锚**到真实时刻；另有独立整数 EMA 跟踪观测间隔（`smoothed_interval_ns`）供诊断/动态节奏。导出 `jitter_ns`/`max_jitter_ns`/`resyncs` 统计。对标主机 frame pacing。
- **低延迟开工门（`ReflexGate`，§24.2）**：编码「按需开工、不预排过多帧」（NVIDIA Reflex / AMD Anti-Lag **形态**）。给定 present 截止、CPU+GPU 工作量估计、在飞帧数，决策 `Begin`（已到 just-in-time）/`Wait{wait_ns}`（late-latch，推迟开工使输入采样尽量贴近模拟开始，压缩 input-to-photon）/`QueueFull`（队列已满，等 present 排空）。`latest_begin_ns = deadline − work − safety_margin`（饱和）。
- **「下一 present 预估时刻」**：由 `FramePacer::next_present_ns()` 提供，供 `prism_app` 主循环与 RHI present 提交对齐（§24.2 协同点）。
- 纯整数、确定性：同一输入序列跨运行产出位相同调度，可从确定性回放驱动。`no_std + alloc`、无 `unsafe`。单测见 `tests_pacing.rs`（首帧只锚定、稳态零抖动零重锚、小抖动吸收不动预测、大偏差/非单调硬重锚、`from_hz`、`reset`、双跑一致；Reflex 队列满/late-latch 等待/紧截止立即开工/零上限钳 1）。

### 24.3 可变刷新率（VRR）感知 ✅ 已交付（`pacing`）

G-Sync/FreeSync/VRR 下显示刷新非固定:

- 时间系统感知显示器可变刷新窗口,把 present 调度到最优时刻,配合帧节奏(§24.2)。
- 固定步仿真(§7)与可变显示解耦,表现层插值 alpha(§12)吸收刷新抖动。

**交付状态（已落地）**：同 §24.2 的 `pacing` 模块（`vrr.rs`）。
- **刷新窗口模型（`VrrWindow`）**：以最快/最慢允许 present 间隔表示 G-Sync/FreeSync/Adaptive-Sync 窗口，`from_hz(min_hz,max_hz)` 构造（最快刷新→最短间隔），构造器钳制使窗口恒非空且有序。
- **`classify(desired)` 把目标间隔映射进窗口**：窗口内 → `InRange`（原样可变显示）；快于窗口 → `ClampedFast`（钳到最短间隔，不能快过面板最大刷新）；慢于窗口 → `Lfc{multiplier,sub_interval}`（低帧率补偿 LFC：取最小整数倍 ≥2 使子间隔回落窗口内，复制帧维持显示刷新）。
- **`earliest_present_ns(ready,last)`**：present 不得快于面板最大刷新，故钳到 `last + min_interval`；配合 §24.2 帧节奏把 present 调度到最优时刻。
- **仿真/显示解耦**：固定步仿真（§7）按确定性步长推进，VRR 只影响表现层 present 时刻，表现层插值 alpha（§12）吸收刷新抖动。
- 纯整数、确定性、`no_std + alloc`、无 `unsafe`。单测见 `tests_pacing.rs`（Hz 边界与排序、窗口内透传、过快钳制、慢帧 LFC x2/极慢 x3 且子间隔回落窗口、`earliest_present` 强制最大刷新、退化窗口钳制、`effective_interval` 访问器）。

### 24.4 帧预算驱动的自适应质量 ✅ 已交付（`adaptive_quality`）

时间系统把「本帧用了多少/还剩多少毫秒」作为**反馈信号**喂给质量调节：

- 导出帧预算与各阶段耗时(接 §16 可观测性、`prism_profiler`)。
- 子系统据此动态调节:动态分辨率、LOD 偏置、阴影级联数、粒子上限、`prism_tasks` 的 `Background` 车道顺延(tasks §24.1)。
- 闭环:超预算→降质保帧率;富余→回升画质,平滑迟滞避免振荡。

**交付状态（已落地）**：`pkg/prism_time/src/adaptive_quality.rs`（纯 CPU 控制器，不碰渲染）。
- **分级质量 + 迟滞死区**：`AdaptiveQualityController` 在 `min_level..=max_level` 整数档位上闭环。每帧按整数 ppm 计算利用率 `util_ppm = frame_ns × 1_000_000 / budget_ns`（`utilization_ppm` 公开，`budget=0` 返回 0），仅当 `util ≥ downgrade_ppm`（默认 100%）记为超预算、`util ≤ upgrade_ppm`（默认 80%）记为富余，二者之间为**中性死区**不动——上下阈分离杜绝单一设定点附近的振荡。
- **超预算快降、富余缓升**：降档只需 `downgrade_patience`（默认 2）连续超预算帧即触发以快速保帧率；升档需 `upgrade_patience`（默认 30）连续富余帧，只有持续有余量才回升画质。中性帧清零两侧连击计数（要求严格连续信号）。
- **冷却期防抖**：任意调整后进入 `cooldown_frames`（默认 8）帧冷却，期间抑制常规升降，钳住 ping-pong。
- **尖峰逃生档**：`util ≥ severe_ppm`（默认 150%）一帧即一次跌 `severe_step`（默认 2）档，**绕过 patience 与冷却**——卡顿/死亡螺旋的快速 bail-out（已在底则钳到 `min_level`）。
- **配置自洽钳制**：构造器 `clamped` 强制 `max_level ≥ min_level`、`upgrade_ppm ≤ downgrade_ppm ≤ severe_ppm`、step/patience `≥ 1`，非法组合不会把控制律带进不一致态。决策以 `QualityAdjustment{Hold|Downgrade{from,to}|Upgrade{from,to}}` 返回（带 `changed`/`delta`/`to_level`）。
- 全程整数/ppm 定点运算，无浮点、不读墙钟：同一 `(frame,budget)` 序列跨运行产出位相同的档位轨迹，可从确定性回放驱动。`no_std + alloc`、无 `unsafe`。单测见 `tests_adaptive_quality.rs`（利用率精确值、patience 快降/缓升、死区保持与连击清零、尖峰多级跌档且绕冷却、底则钳制、冷却窗口精确计数、配置钳制、`set_level`/`reset`、双跑一致等）。

### 24.5 长会话高精度与漂移修正 ✅ 已交付（`drift`）

24/7 服务器、长流程单机会话的时间精度:

- `elapsed` 用 f64/整数 tick 累加,杜绝 f32 长时程精度丢失(动画/计时错位)。
- 单调时钟溢出/回绕处理(平台计数器位宽有限)。
- 可选对墙钟做缓慢漂移修正(服务器长跑与 NTP 对齐),但不破坏单调与固定步确定性。

**交付状态（已落地）**：`pkg/prism_time/src/drift/`（`mod.rs` + `monotonic.rs` 单调基准 + `corrector.rs` 漂移修正）。
- **整数 tick 权威，杜绝 f32 漂移**：`MonotonicBaseline` 把定频、定位宽的硬件计数器累加为无界 `u128` tick 总量作为权威 elapsed，`Duration`/`f64` 视图均由整数 tick 派生——长会话不积累 `f32`/`f64` 精度误差（动画/计时不错位）。
- **回绕安全**：`update(raw)` 以 `(raw.wrapping_sub(last)) & mask` 处理固定位宽计数器溢出回绕（`width_bits` 1..=64，掩码截断），只要每个回绕周期至少轮询一次即可重建真实前进量；回绕周期对任何现实频率/位宽都远长于一帧。
- **有界 slew 漂移修正，不破坏单调**：`DriftCorrector` 以 `residual = reference - corrected` 跟踪待吸收偏移，每步 `slew = clamp(residual, ±max_slew)`，`max_slew = real_delta × ppm / 1e6`；slew 上限严格 < 1e6 ppm（默认 500 ppm），故修正量恒小于真实 delta，corrected 时间**严格单调、绝不跳变**，保全固定步确定性，多帧后 residual 收敛至零。
- **会话边界硬跳**：`resync(reference)` 为首次同步 / 挂起恢复等真实不连续点提供一次显式、可报告的阶跃（返回有符号跳变 ns），由调用方当作会话边界而非逐帧修正处理。
- 全程确定性整数运算（`u128`/`i128` 纳秒，无浮点进修正路径），同一 delta 与参考样本序列跨运行产出位相同的修正时间线；`no_std + alloc`、无 `unsafe`。单测见 `tests_drift.rs`（回绕重建、整数权威无漂移、slew 严格单调收敛、ppm 边界、`resync` 阶跃、双跑一致等）。

### 24.6 游戏内定时调度器 ✅ 已交付（`scheduler`）

统一的「在 T 时刻/经过 D 后/每隔 P」回调调度(高于裸 Timer):

```rust
time.schedule_after(Duration::from_secs(3), |w| spawn_wave(w));
time.schedule_every(Duration::from_millis(500), |w| tick_regen(w));
```

- 走 Virtual 时钟(受暂停/缩放影响)或 Real 时钟(UI/网络心跳)可选。
- 支持取消句柄、合并、确定性排序(回放一致)。供 gameplay 冷却、刷怪、buff 到期。

**交付状态（已落地）**：`pkg/prism_time/src/scheduler/`（`mod.rs` + `handle.rs` 句柄/事件 + `queue.rs` 最小堆队列）。
- **延迟 / 定点 / 周期调度**：`schedule_after(delay, payload)` / `schedule_at(when, payload)` / `schedule_every(period, payload)`（及 `schedule_every_from(first, period, payload)` 分离首帧与周期），周期为零会 panic（零周期无法前进）。
- **确定性时间轮进**：时间以精确整数纳秒（`u128`）累加，不读墙钟、调度路径无浮点；`advance(delta, out)` 推进并按序追加 `Fired<T>` 事件流，`advance_collect` 为便捷分配版。
- **确定性排序**：事件按 `(fire_time, 插入 seq)` 定序——同刻事件按调度顺序触发，故两次以相同顺序调度相同 timer 的运行产出逐字节一致的 `Fired` 流（§24.6 要求的回放一致）。
- **取消 / 重调度（句柄稳定）**：`TimerHandle` 跨 `reschedule` 保持有效；`cancel` 升 generation 退役槽位，`reschedule` 升 epoch 作废旧堆项——二者 O(1) + 懒清理；槽位经 free list 复用并以 generation 区分，陈旧句柄不会复活。
- **大 delta 追帧与尖峰防护**：周期 timer 靠最小堆自然补齐跨越的多个周期；`max_fires_per_advance`（默认 4096）为单次 advance 的触发上限，防止极小周期在巨 delta 下无界 spin，余量由后续 `advance`/`drain_due` 续排。
- **诚实边界（payload 而非闭包）**：本层不存 `FnMut` 闭包（闭包无法确定性 `Clone` 做双跑比对，也无法在无分配器 `no_std` 内核落地），而是拥有时间内核能确定性拥有的部分——按精确触发时间定序、携带用户 payload 的优先队列；由 payload 映射回 gameplay 动作（刷怪 / buff 到期 / 冷却）属 gameplay 层接线（见 §24.9）。走 Virtual（受暂停/缩放）或 Real（UI/心跳）时钟由调用方选择喂哪条 delta。
- `no_std + alloc`（堆为 `alloc::collections::BinaryHeap`）、无 `unsafe`。单测见 `tests_scheduler.rs`（准点单发、同刻按序、大 delta 时序、周期追帧、取消、重调度保句柄、槽位复用不复活陈旧句柄、触发上限分批、双跑一致、零周期 panic 等）。

### 24.7 挂起 / 恢复与后台暂停 ✅ 已部分交付（`suspend`，CPU 可做部分）

进程挂起（移动端切后台、主机休眠、窗口最小化）后恢复：

- 恢复时**不把挂起时长算成一帧巨 delta**（否则物理穿墙/动画跳跃）——钳制或丢弃该帧 delta（接 §12 最大 delta 钳制）。
- 可配置后台行为:暂停 Virtual(游戏冻结)、Real 照走(网络心跳不断)、或低频后台更新。
- 平台挂起/恢复事件由 `prism_platform`/`prism_app` 转发给时间系统。

**交付状态（已落地：纯 CPU 可做部分）**：`pkg/prism_time/src/suspend.rs`（对调用方喂入的时间戳做确定性修正，不调用任何 OS 挂起 API）。
- **挂起期 elapsed 不计入**：`SuspendableClock` 消费单调递增的墙钟时间戳（`Duration`，自任意 epoch 起），**只在运行区间**把墙钟差折进模拟 elapsed；`suspend(at)` 先把挂起瞬刻前的运行部分计入再停表，`resume(at)` 以 resume 瞬刻为新基准重开——`[suspend, resume]` 这段挂起墙钟差对 elapsed/tick 贡献为零，故恢复瞬间**不产生巨 delta 跳变**（避免物理穿墙/动画跳跃）。
- **elapsed/tick 确定性修正**：权威 elapsed 为整数纳秒（`u128`），`elapsed()`/`elapsed_nanos()`/`ticks(tick)=⌊elapsed/tick⌋` 均由其派生；挂起/恢复的修正体现在同一权威量上，tick 视图与 elapsed 恒一致。
- **可选最大追帧钳制**：`with_max_delta`（接 §12 最大 delta 钳制）把任一次运行步长的折入上限钳住，超额部分丢弃并计入 `total_clamped`——既吸收普通卡顿，也兜底任何异常长步；`total_suspended` 报告累计被排除的挂起时长。挂起态下喂入的时间戳被忽略（`advance_to` 返回零），非单调回退时间戳饱和为零 delta。
- 全程整数纳秒运算，内部不读墙钟、无浮点：同一时间戳序列跨运行产出位相同的模拟时间线。`no_std + alloc`、无 `unsafe`。单测见 `tests_suspend.rs`（运行累加、挂起恢复排除间隙且无尖峰、首观测前挂起、重复挂起/游离恢复幂等、最大 delta 钳制与丢弃计量、零钳制冻结、`reset` 保钳、双跑一致等）。
- **诚实边界（PLANNED）**：真实 OS 挂起/恢复信号的接线（移动端切后台、主机休眠、窗口最小化）与后台策略选择（冻结 Virtual / Real 照走心跳 / 低频后台更新）由 `prism_platform`/`prism_app` 读取并转发给本时钟——该事件源接线仍为 **PLANNED**；本层只对调用方喂入的时间戳做确定性修正，不触碰任何 OS API 或系统时钟。

### 24.8 多世界时间域隔离与确定性审计 ✅ 已交付（`multiworld`）

- **多世界**:每个 World（主世界、编辑器预览、服务器子应用）持独立时间上下文,互不干扰(接 App §子应用)。
- **确定性审计**(`determinism` + `trace` 档):记录每 tick 的 `dt`/累加器/子步数,双跑比对定位首个发散 tick,是确定性回归的诊断利器。

**交付状态（已落地）**：`pkg/prism_time/src/multiworld/`（`mod.rs` + `time_domain.rs` 多世界 + `audit.rs` 确定性审计）。
- **多世界隔离**：`WorldTimeDomain` 为每个 World 持独立的 `scale`/`pause`/累加器与确定性 tick（内置 `TickClock`）；推进一个世界只改该世界状态——暂停编辑器预览世界不影响主世界运行，各世界 `scale` 相互独立。`WorldSet` 以同一真实 delta 推进多个世界，各自套用自己的策略。
- **确定性审计**：`StateHasher`（64 位 FNV-1a，无随机种子、跨平台确定）把每帧关键状态（tick/累加器/步长/scale 位/暂停位）哈希为摘要；`AuditTrail` 逐帧记录；`compare_trails` 双跑比对，`AuditDiff` 报告 `Identical` / 首个发散帧 `Diverged{frame,left,right}` / 等长不符 `LengthMismatch`——首个分叉即定位。
- 纯整数/`Duration`/`f64`-bit 确定性运算，`no_std + alloc`。单测见 `tests_multiworld.rs`（暂停隔离、scale 独立、`WorldSet` 独立推进、审计双跑一致、首个发散定位、长度不符、哈希确定性、重置与边界、精确步长零漂移）。

### 24.9 诚实边界

本章 **24.1 录制回放、24.4 帧预算自适应质量、24.5 长会话漂移修正、24.6 游戏内定时调度器、24.8 多世界时间域隔离/确定性审计已落地，24.7 挂起/恢复已落地纯 CPU 可做部分**（见各节「交付状态」，`recording` / `adaptive_quality` / `drift` / `scheduler` / `multiworld` / `suspend` 六模块，`cargo test -p prism_time` 全绿、`cargo clippy --all-targets` 零告警）；24.2 帧节奏 + 24.3 VRR 已落地纯 CPU 调度数据模型（`pacing`）。**24.1 录制回放 + 24.8 确定性审计**是确定性系统(物理/网络)的调试基石,已随 M4 确定性路线落地;**24.4 自适应质量**已交付纯 CPU 控制器（档位决策），真实 per-stage 耗时采集(`prism_profiler`/§16)与档位→渲染设置映射属消费方接线;**24.7 挂起/恢复**已交付时间戳层确定性修正（挂起期不计入 elapsed/tick、可选最大追帧钳制），真实 OS 挂起/恢复信号接线仍为 PLANNED,随平台接线落地;24.2 帧节奏 + 24.3 VRR 已交付纯 CPU 调度模型（`FramePacer`/`ReflexGate`/`VrrWindow`），真实 present 时间戳采集、显示 VRR 范围查询与 RHI 提交属 M2 主循环/平台接线。

**24.4 / 24.5 / 24.6 / 24.7 的诚实边界（硬件 / 接线归属）**：本层只提供**确定性算法 / 控制决策数据模型**,不碰任何真实硬件、渲染或墙钟。
- **24.4 自适应质量**：`AdaptiveQualityController` 只产出整数质量档位与升降决策；真实 per-stage 帧耗时采集(`prism_profiler`/§16 可观测性)与把档位映射到动态分辨率/LOD/阴影级联/粒子上限等渲染设置,属渲染/gameplay 层接线,控制器自身不分配、不读时钟。
- **24.7 挂起/恢复**：`SuspendableClock` 只对调用方喂入的单调时间戳做确定性修正(挂起期不计入 elapsed/tick、可选最大追帧钳制);真实 OS 挂起/恢复事件源与后台策略(冻结 Virtual / Real 照走 / 低频后台)由 `prism_platform`/`prism_app` 转发——该接线为 **PLANNED**,本层不调用任何 OS 挂起 API。
- **真实 OS 单调时钟 / 平台计数器频率与位宽**由 `prism_platform` 读取并注入；`MonotonicBaseline` 只对调用方喂入的原始计数值做回绕安全累加,位宽(`width_bits`)/频率(`ticks_per_sec`)为调用方声明的参数,本模块不探测硬件。
- **NTP / 权威服务器参考样本**由 net 层(往返估计)或平台授时提供；`DriftCorrector` 只消费调用方给出的参考 `Duration`/偏移,执行有界 slew 修正,不发起网络请求、不读系统墙钟。真实网络条件下的收敛手感须联网实测,本层仅保证算法层面的单调性与确定性。
- **调度器 payload→动作映射、Virtual/Real 时钟选择、World 调用**属 gameplay 层接线；`Scheduler` 不存闭包、不调用 World,只按精确触发时间定序派发携带 payload 的 `Fired` 事件(理由见 §24.6「诚实边界」)。喂哪条时钟的 delta 由调用方决定。

帧节奏/低延迟须真实硬件(含主机/VRR 显示器/移动端)验证,设计阶段无法断言数值。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码;仅借鉴公开架构形态与经典数值。

