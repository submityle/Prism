# Prism 输入系统 — 破坏性重构设计（设备输入内核 · 确定性时间轴 · Action 映射层 · 手柄触觉/陀螺 · 无障碍 · 并发·零分配 · golden 对拍）

> 状态：架构提案（Draft，**允许破坏性重构，不保留旧 API**）
> 面向版本：Prism 下一代运行底座（Bevy fork，`pkg/` workspace）
> 覆盖范围：`pkg/prism_input`（设备输入内核）+ 新建 `pkg/prism_input_map`（Action 映射层），以及它们与 OS 后端、UI 输入层（`prism_ui_input`）的边界契约。
> 本文依据：对 `pkg/prism_input/src/*.rs` 的逐文件静态阅读（以代码为准，行号见 §1）。实测 `prism_input` 约 1389 行；`prism_input_map` 目前**不存在**，为本文新建目标。
> 拆分说明：本文由原「窗口 & 输入」合并设计文档拆分而来，与**窗口系统**文档正交。窗口内核/后端的 ABI、三态同步、Surface 句柄、HDR/VRR/多屏请见窗口文档；本文只讲**输入**。两份文档共享同一套后端 ABI 范式、单调时间轴与 golden 对拍地基（§4、§5、§11）。
> 核心命题：**设备输入内核保持纯粹、确定、`no_std + alloc`、零依赖、无 `unsafe`；把"平台脏活"全部挡在一条显式 ABI 契约之外。重构把内核升级为能承载 AAA 需求（确定性时间轴/录制重放/lockstep/触觉/陀螺/Action 映射）的权威状态模型 + 一条命令出·事件入的后端边界。**
> 时间轴命题：**输入事件携带单调时间戳，内核以确定性时间轴驱动——同一事件流在任意机器上重放得到字节级一致的状态。这是 rollback 网络同步、回归测试、输入延迟测量的共同地基。**
> 分层命题：**设备级原始输入（`prism_input`）与语义级 Action 映射（新 `prism_input_map`）正交分层；UI 路由（`prism_ui_input`）在其上。三者不互相塌缩成一个"大输入系统"，各自是可独立测试的纯函数内核。**
> 并发命题：**后端线程与模拟线程之间只经两条无锁单生产者单消费者（SPSC）队列通信——事件入、命令出；内核状态只被模拟线程拥有（single-owner），不加锁、不共享可变态。**
> 零分配命题：**稳态每帧热路径零堆分配：事件是 `Copy` POD 走环形缓冲（§5.5），状态容器（`BTreeSet`/`BTreeMap`）跨帧复用、每帧只做 O(变更量) 的边沿清理。分配只发生在热插拔设备/加载绑定档案等冷路径。**
> 可测命题：**确定性由 golden 对拍门禁锁定（§11）：录制一段 `InputEventEnvelope` 流 → 重放两次 → 状态快照字节级一致，且跨平台（x86/ARM）一致。确定性不是口号，是 CI 红线。**
> 关联文档：窗口系统重构文档、`prism_platform_design_zh.md`（后端/平台契约）、`prism_app_design_zh.md`、`prism_window_winit_design_zh.md`（winit 后端，同时产出窗口与输入事件）。

---

## 0. 设计目标与非目标

### 目标（硬约束）
- **顶级次世代 AAA 质量**：手柄触觉/陀螺/自适应扳机（DualSense）、Action 语义映射、无障碍重映射、高轮询率原始输入（1k–8k Hz）、录制重放/lockstep——皆一等公民。
- **内核纯粹性不动摇**：`prism_input` 继续是 **纯、确定、`no_std + alloc`、零依赖、无 `unsafe`** 的类型系统（现状已满足，见 §1）。重构只扩词汇表与契约，不往内核里塞 winit/SDL/OS syscall。
- **一条后端 ABI 契约**：内核与 OS 后端（winit/SDL3/原生/主机/无头）之间只有一条显式、可版本化的"命令出·事件入"边界。后端可替换、可多实现并存、可字节级录制回放。
- **确定性时间轴**：所有输入事件携带单调时间戳；内核状态推进是输入流的纯函数。重放/lockstep/回归对拍共用同一地基。
- **语义正交**：设备输入 ↔ Action 映射 ↔ UI 路由 三层正交，各自纯函数内核，不互相吞并。
- **易用**：游戏逻辑面向 **Action**（`Jump`/`Move`/`Fire`），而非 `KeyCode::Space`；美术/策划在数据层配绑定与设备档案，不改代码。
- **性能确定**：热路径零分配、`Copy` 事件、`BTree*` 确定性迭代（现状已是）、每帧边沿清理 O(变更量)。高轮询率鼠标与原始输入不丢样本。

### 非目标（主动不做）
- **不把平台后端并入内核**：winit/SDL/OS syscall/手柄 HID 永远在独立后端 crate；内核对其零依赖（现状 Cargo.toml 已零依赖）。
- **不做"大一统输入系统"**：拒绝把设备输入、Action 映射、UI 手势塌缩进一个巨型 crate。分层正交是性能与可测性的前提。
- **不引入运行时反射驱动的事件分发**：事件是封闭 `enum`（`InputEvent`），`Copy` 且可穷举匹配。
- **不在内核实现触觉波形/音频/呈现时序等副作用**：输出命令由后端执行，不回灌模拟态（§8.4）。

---

## 1. 现状基线（`prism_input` 代码实测，作为重构起点）

| 事实 | 位置 | 说明 | 本次处理 |
|---|---|---|---|
| 零依赖 + `no_std + alloc` + 无 `unsafe`，edition 2024 | `Cargo.toml`、`lib.rs` | 后端翻译平台→`InputEvent` 流，内核转可查询状态 | **保持**；升级为带时间戳的确定性时间轴（§5） |
| `ButtonInput<T>` 三有序集 `pressed/just_pressed/just_released`（`BTreeSet`）+ `press/release/pressed/just_pressed/just_released/clear/clear_all` | `button.rs:21`（`BTreeSet:22`） | 键盘/鼠标/手柄按键统一模型，**有序→确定性迭代** | **保持此基石**；`clear`（每帧清边沿）纳入时间轴 tick（§5.2） |
| `Axis<T>` = `BTreeMap<T,f32>`，`MIN=-1/MAX=1`，`get`（钳位）/`get_unclamped`/set/remove | `axis.rs:14`（`MIN/MAX`） | 连续轴存储，确定性 | 保留；Action 映射层消费它（§7） |
| `InputEvent`（`Copy` enum）：`Keyboard/MouseButton/MouseMotion/MouseWheel/Touch/GamepadButton/GamepadAxis/GamepadConnection` | `event.rs:34` | 统一原始事件；按到达序 | **破坏性**：引入 `InputEventEnvelope{timestamp,device_id,InputEvent}`（§5.2） |
| `ButtonState{Pressed,Released}` + `is_pressed` | `event.rs:14` | 数字输入状态 | 保留 |
| `KeyCode`（`#[non_exhaustive]`，W3C UI Events 物理码位）+ `is_modifier`；`ModifiersState{shift,control,alt,super}`；`KeyboardInput{key_code,state,repeat}` | `keyboard.rs:12/18/131/151` | 物理键位（布局无关）+ 逻辑修饰态 | 保留物理码位；**补逻辑字符层**（IME/文本输入，§5.4） |
| `MouseButton`/`MouseMotion`/`MouseWheel` + 单位枚举 | `mouse.rs` | 鼠标按钮/相对位移/滚轮 | 保留；相对 delta 全累加（§4.6） |
| `TouchInput`/`Touches` 相位状态机 | `touch.rs` | 多点触控 | 保留；相位边沿不可合并（§4.6） |
| `GamepadButton`/`GamepadAxis`/`GamepadConnection`，`AxisSettings`（死区/活区/迟滞）+ `filter`、`radial_deadzone`、`GamepadSettings`、`sqrt_f32` | `gamepad.rs:133/166` | 手柄"只进"输入 + 死区数学（自带，不依赖 libm） | **破坏性**：手柄变**双向**（输出命令 §8）；`AxisSettings` 被 Action 层复用（§7） |

**一句话**：`prism_input` 现状是一个干净的"设备输入内核"——`ButtonInput`/`Axis`/`Touches` 三件套 + 自带死区数学，纯确定零依赖、`BTree*` 有序迭代（rollback/回归的一半地基）。缺口是**时间戳 + 纯函数状态推进（时间轴）、逻辑字符层（IME）、手柄双向输出、Action 语义层**尚未进入词汇表，且后端边界是注释约定而非正式契约。

---

## 2. 对标优秀项目（取其形，不抄其码）

| 项目 | 借鉴点 | 落地到本文 |
|---|---|---|
| **Unreal Enhanced Input** | Action/MappingContext/Modifier/Trigger 分层，上下文优先级与 consume | §7 Action 映射层的核心模型 |
| **Unity Input System** | Action Map 切换、设备无关 Action、PlayerInput 玩家槽 | §7.4 上下文栈、§8.2 玩家槽配对 |
| **Steam Input** | 绑定重映射、按设备切 glyph 键帽图标 | §7.5 glyph 查询 API |
| **GGPO / rollback netcode** | 确定性模拟 + 快照/回滚/重算 | §5.5 环形缓冲·快照·回滚、§5.6 固定步长累加器 |
| **Windows.Gaming.Input / GameInput** | 统一手柄/触觉/扳机/陀螺/电量、原始输入高轮询率 | §8 手柄输出能力词汇表 + 原始输入 |
| **DualSense（PS5）** | 自适应扳机阻力、HD 触觉、陀螺仪/加速度计、触摸板 | §8.1 自适应扳机 + 双电机 + 陀螺词汇表 |
| **W3C UI Events / Pointer Events** | 物理 `code` vs 逻辑 `key`、Pointer 统一 | §5.4 逻辑字符分层、§5.3 Pointer 统一（远期） |

**一句话**：输入侧对标 Enhanced Input/Unity/Steam 的**Action 分层**、GGPO 的**确定性时间轴**、GameInput/DualSense 的**手柄输出能力**。

---

## 3. 分层架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│ 应用 / 游戏逻辑                                                        │
│   面向 Action（Jump/Move/Fire）                                        │
└───────────────┬─────────────────────────────┬────────────────────────┘
                │ 查询 Action 态                │
┌───────────────▼───────────────┐              │
│ UI 路由层  prism_ui_input       │              │
│  手势/焦点/命中测试/速度         │              │
│  （独立，消费 Action 或原始）    │              │
└───────────────┬───────────────┘              │
                │                               │
┌───────────────▼───────────────┐              │
│ Action 映射层 prism_input_map  │（新）         │
│  InputAction/MappingContext    │              │
│  Modifier/Trigger/优先级栈      │              │
│  设备无关语义，数据驱动绑定      │              │
└───────────────┬───────────────┘              │
                │ 查询原始态                      │
┌───────────────▼───────────────────────────────▼──────────────────────┐
│ 设备输入内核 prism_input（重构）                                       │
│  ButtonInput<T> / Axis<T> / Touches                                   │
│  确定性时间轴（时间戳事件 InputEventEnvelope，纯函数 tick）             │
│  手柄输出能力（触觉/扳机/陀螺 GamepadOutput）                          │
└───────────────┬───────────────────────────────▲──────────────────────┘
                │ 命令出 GamepadOutput             │ 事件入 InputEventEnvelope
┌───────────────▼───────────────────────────────┴──────────────────────┐
│ OS 后端（独立 crate，可多实现并存）winit / SDL3 / 原生 / 主机 / 无头   │
│  平台消息 → Envelope；执行输出命令；触觉 / 原始输入 / HID              │
└───────────────────────────────────────────────────────────────────────┘
```

### 核心不变量
- **内核纯粹**：`prism_input`/`prism_input_map` 皆 `no_std + alloc`、零依赖、无 `unsafe`、确定。平台脏活只在后端 crate。
- **一条边界**：内核↔后端只有"命令出（`GamepadOutput`）+ 事件入（`InputEventEnvelope`）"两个方向；后端可换、可并存、可录制回放。
- **三层正交**：设备输入 → Action 映射 → UI 路由，单向依赖，各自纯函数内核，互不吞并。
- **时间轴统一**：输入事件携带单调时间戳（与窗口事件同一时间基准）；状态推进是事件流纯函数——重放=lockstep=回归对拍共用地基。
- **降级不降精度**：弱设备/弱平台降的是**后端能力探测结果**（无触觉就空实现），内核模型与语义**不塌缩**。

---

## 4. 输入后端 ABI 契约

> 窗口与输入共享同一后端 ABI 范式（信封入、命令出、能力探测、SPSC 线程边界）。本节只给**输入侧**类型；范式的完整论证见窗口/平台文档。

### 4.1 事件入 / 命令出 / 能力探测
```
// 事件入：后端 → 内核
pub struct InputEventEnvelope {
    pub timestamp: Instant,   // 单调时间戳（子帧精度，支撑高轮询率）
    pub device: DeviceId,     // 破坏性：区分多设备（双手柄/多鼠标）
    pub event: InputEvent,    // 现有 Copy enum，保持
}

// 命令出：内核/应用 → 后端（手柄双向，§8）
pub enum GamepadOutput {
    Rumble { gamepad: GamepadId, low_freq: f32, high_freq: f32, duration: Duration },
    TriggerFeedback { gamepad: GamepadId, trigger: TriggerSide, mode: AdaptiveTrigger },
    LedColor { gamepad: GamepadId, rgb: [u8;3] },
    Haptic { gamepad: GamepadId, pattern: HapticPattern },
}

// 能力探测：后端 → 内核（一次性/热插拔刷新）
pub struct InputBackendCapabilities {
    pub raw_input: bool,          // 高轮询率原始鼠标
    pub max_polling_hz: u32,
    pub rumble: bool,
    pub adaptive_trigger: bool,
    pub gyro: bool,
    pub touchpad: bool,
}
```

### 4.2 后端契约的三条铁律
1. **后端无状态权威**：可查询输入态唯一真相在内核 `InputState`；后端只把 OS 原始输入封装为 `InputEventEnvelope` 回灌，并异步执行 `GamepadOutput`。
2. **事件不丢序不丢时戳**：后端按 OS 到达序封装，附单调时间戳；内核假定事件流有序。时间戳非单调时由后端钳为单调（回退时 clamp）。
3. **能力透明**：后端启动即上报 `InputBackendCapabilities`；无能力即空实现（无触觉→no-op，无陀螺→不产 Motion），**不 panic、不假设**。

### 4.3 线程与所有权模型
```
 主线程 / OS 事件泵（后端拥有）           模拟线程（内核拥有）
 ┌─────────────────────────┐            ┌──────────────────────────┐
 │ winit/SDL event loop     │  events →  │ InputState / ActionState   │
 │ 翻译 OS msg → *Envelope   │ (SPSC ring)│ tick()：纯函数推进         │
 │ 执行 GamepadOutput        │ ← commands │ 产出 GamepadOutput         │
 └─────────────────────────┘  (SPSC ring)└──────────────────────────┘
```
- **单一所有权**：`InputState`/`ActionState` 只被模拟线程 `&mut` 持有——**无 `Arc<Mutex>`、无 `RwLock`、无共享可变态**。消除锁竞争与优先级反转。
- **两条 SPSC 环形队列**：事件入（后端→模拟）、命令出（模拟→后端）。`Copy` POD 事件天然 `Send`，无需深拷贝/装箱。
- **帧栅栏（frame fence）**：模拟线程每 tick 开始时**一次性排空**事件环（drain-to-batch），保证一帧内看到的输入是一个确定切片（§5.2 `tick(now, &[Envelope])`）。
- **背压**：环满时后端合并/丢弃**可合并**事件（§4.6），绝不阻塞 OS 事件泵。不可合并事件（按键边沿/`Touch` 相位/连接）永不丢；环按最坏情况预分配容量。
- **主机/无头**：无独立 UI 线程时退化为单线程直连（队列变成同线程 drain），ABI 不变。

### 4.4 `DeviceId` 多设备
破坏性引入 `DeviceId` 区分多鼠标/多键盘/多手柄（分屏多人、绘图笔 + 鼠标并用）。内核按 `DeviceId` 聚合或分流状态，玩家槽（§8.2）把 `GamepadId` 映射到稳定逻辑槽位。

### 4.6 事件批处理与合并（coalescing）
一帧内 OS 可能产出成百上千条输入事件（1k–8k Hz 鼠标、拖拽连发）。后端在入环前做**保序合并**：

| 事件 | 合并策略 | 理由 |
|---|---|---|
| `MouseMotion`（相对 delta） | **累加**不丢（§5.2） | 视角需要全部相对位移之和 |
| `MouseWheel` | 同单位累加 | 滚动量可叠加 |
| `GamepadAxis`（绝对值） | 同轴保留**最后一条** | 当前摇杆位即可 |
| 按键/按钮边沿、`Touch` 相位、连接事件 | **永不合并** | 边沿语义不可丢（按下→抬起必须成对） |

合并在后端线程完成，模拟线程拿到的已是"每帧最小充分事件集"。相对 delta 的累加保证**高轮询率零精度损失**。

---

## 5. 输入确定性：时间戳时间轴 / 录制重放 / lockstep

### 5.1 问题（现状）
现状 `InputEvent`（`event.rs:34`）是 `Copy` 但**无时间戳、无设备 id**。`ButtonInput`/`Axis`/`Touches` 已用 `BTree*` 保证**确定性迭代**（现状最大优点）——这正是 rollback/lockstep 需要的一半地基。缺的另一半是**时间戳 + 纯函数状态推进**。

### 5.2 方案：带时间戳信封 + tick 纯函数
```
pub struct InputEventEnvelope {
    pub timestamp: Instant,
    pub device: DeviceId,
    pub event: InputEvent,
}

// 内核推进：确定性纯函数
impl InputState {
    pub fn tick(&mut self, now: Instant, events: &[InputEventEnvelope]);
    //   1) clear 上帧 just_pressed/just_released（button.rs 已有 clear）
    //   2) 按 timestamp 升序 apply 事件（BTree 确定性）
    //   3) 固定 tick 下，同一 events 切片 → 同一状态（可对拍）
}
```

- **高轮询率**（1k–8k Hz 鼠标）：一个模拟帧内多条 `MouseMotion`，各带子帧时间戳；可**全部累加**（视角）或**按需重采样**（§6 late-latch）。不丢样本。
- **录制重放**：序列化 `InputEventEnvelope` 流即可字节级重放（事件是 `Copy` POD）。
- **lockstep/rollback**：确定性 `tick` + `BTree*` 有序状态 ⇒ 可保存/回滚 `InputState` 快照，重算到当前帧。这是 GGPO 式网络同步的内核地基。

### 5.3 Pointer 统一（远期）
鼠标/触控/笔共性抽象（W3C Pointer Events 式）作为**可选上层视图**，不塌缩底层设备事件（`MouseButton`/`TouchInput` 保留）。P2 远期。

### 5.4 逻辑字符 / IME（文本输入）
现状 `KeyCode`（`keyboard.rs:18`）是**物理码位**（布局无关，适合游戏绑定）。文本输入需**逻辑字符层**：破坏性新增 `InputEvent::Text{ ... }` + IME 组合事件（`ImePreedit/ImeCommit`），由后端产出。物理层（游戏绑定）与逻辑层（文本/UI）正交，不互相替代。

### 5.5 环形缓冲 · 快照 · 回滚（rollback 深化）
- **事件环形缓冲**：`InputRing`（固定容量，`Copy` POD）保存最近 `N` 帧的 `InputEventEnvelope`。稳态零分配：写指针绕回覆盖最旧。容量按 `最大回滚帧数 × 每帧最坏事件数` 预分配。
- **状态快照**：`InputState` 全 `Copy`/`Clone` 且无堆外引用，`snapshot()` 是一次浅拷贝（`BTreeSet`/`BTreeMap` 的 clone 成本 = O(当前按下键数)，通常个位数）。按**定帧频率**存快照（如每确认帧一个），不是每帧。
- **回滚重算**：收到迟到的远端输入后，`restore(snapshot_at(f))` → 用环里 `[f, now]` 的事件 `tick` 重放到当前帧。成本 = `回滚帧数 × 每帧 tick`，因 tick 纯函数且零分配，可预测、可预算。
- **为何此前就成立**：现状 `BTree*` 有序状态保证重放顺序确定；本节只是把"顺序确定"升级为"可保存/可回放的确定时间线"。这是 GGPO/rollback 的内核地基，**不依赖网络层**——网络层只负责搬运 `InputEventEnvelope`。

### 5.6 固定步长累加器（determinism 深化）
- 模拟以**固定 tick**（如 1/60 s）推进；渲染可变帧率。累加器 `acc += frame_dt`，当 `acc >= TICK` 时跑一次 `tick(now, drain)` 并 `acc -= TICK`，`now` 用整数 ns 推进（**不累积浮点误差**）。
- 一个渲染帧内可能 0 次或多次模拟 tick；每次 tick 从环里取**截至该 tick 时间戳**的事件切片，保证不同帧率下同一录制产生同一模拟序列。
- 渲染插值用 `alpha = acc / TICK`（仅用于显示平滑，不回灌模拟态），确定性不受影响。

---

## 6. 低延迟与晚采样（input-to-photon 的输入侧）

> 帧节奏器 `FramePacer` 的 present 侧数学在**窗口文档**；本节只给输入侧参与延迟链的部分。

- **晚采样（late-latch）**：模拟前最后一刻再读输入时间轴，用环形缓冲（§5.5）里**截至该时刻**的事件推进状态，削一整帧输入延迟。`t_sample` 尽量贴近 `t_sim_begin`。
- **延迟标记**：时间轴在关键相位打单调时间戳，构成可测延迟链 `t_sample → t_sim_begin/t_sim_end → t_submit → t_present`，`photon_latency ≈ t_present - t_sample` 逐帧入直方图（对接 `prism_diagnostic`），暴露 p50/p99。
- 内核只提供"打点 + 查询"API，不实现厂商 Reflex/Anti-Lag SDK；后端把低延迟提示映射到厂商扩展。

---

## 7. Action 映射层 `prism_input_map`（新 crate）

### 7.1 为何独立成层
现状 `prism_input` 只有**设备级原始输入**。游戏逻辑写 `keyboard.pressed(KeyCode::Space)` 会把"跳跃"硬绑到物理键，无法重绑、无法跨设备（手柄南键）、无法做无障碍。对标 UE Enhanced Input / Unity Input System：引入 **Action 语义层**，与设备层正交。

### 7.2 核心模型
```
pub struct InputAction { pub id: ActionId, pub value_kind: ValueKind } // Button/Axis1D/Axis2D
pub struct Binding { pub source: InputSource, pub modifiers: Vec<Modifier>, pub trigger: Trigger }
pub enum InputSource { Key(KeyCode), Mouse(MouseButton), MouseAxis(..), Pad(GamepadButton), PadAxis(GamepadAxis), Touch(..) }
pub enum Modifier { Deadzone(AxisSettings), Invert, Scale(f32), ResponseCurve(Curve), SwizzleAxis, Negate }
pub enum Trigger { Pressed, Released, Hold{secs:f32}, Tap, DoubleTap, Chord(Vec<InputSource>), Down }
pub struct MappingContext { pub priority: i32, pub bindings: BTreeMap<ActionId, Vec<Binding>> }
pub struct PlayerSlot(pub u8); // ↔ GamepadId 配对
```

### 7.3 求值：原始态 → Action 态（纯函数）
```
impl ActionState {
    pub fn evaluate(&mut self,
                    contexts: &[MappingContext],   // 按 priority 降序
                    raw: &InputState,               // §5 设备输入
                    now: Instant);
    // 高优先级上下文可"消费"输入，屏蔽低优先级（UE 式 consume）
}
pub fn action_value(&self, id: ActionId) -> ActionValue; // bool / f32 / Vec2
```
- 复用现有 `AxisSettings::filter`（`gamepad.rs:133`，死区/活区）与 `radial_deadzone` 作为 `Modifier::Deadzone`——**不重造死区数学**。
- `Modifier`/`Trigger` 皆纯函数、确定、`no_std`。

### 7.4 上下文优先级栈（Action Map 切换）
- `MappingContext` 带 `priority`：如 `Menu`(100) > `Vehicle`(50) > `OnFoot`(10)。进菜单 push 高优先级上下文，自动屏蔽行走绑定。
- 玩家槽（§8.2）各自持上下文栈，分屏多人各自独立。

### 7.5 提示 / glyph（易用）
`ActionId → 当前绑定 → 设备相应 glyph`（Steam Input 式），UI 显示"按 [X] 跳跃"随当前设备自动切键帽图标。此为查询 API，不含资源，由上层提供 glyph 图集。

### 7.6 求值管线与缓存（性能深化）
`evaluate` 每帧 O(激活绑定数)，用三步流水并尽量短路：
1. **上下文折叠**：按 `priority` 降序把激活的 `MappingContext` 折叠成一张 `ActionId → 生效绑定` 的扁平表（`BTreeMap`，确定序）。上下文集不变时此表**缓存复用**（脏标记触发重建），稳态零分配。
2. **源采样**：每条 `Binding` 从 `InputState`（§5）读原始值 → 过 `Modifier` 链（死区/反转/曲线/swizzle）。复用 `AxisSettings::filter`。
3. **消费短路**：高优先级上下文命中后标记"消费"该源，`BTreeSet<InputSource>` 记录已消费集，低优先级遇到即短路跳过（UE consume 式）。
- 复杂度：`O(Σ 激活绑定)`，与总绑定库无关；折叠表让每帧只遍历当前上下文栈里真正激活的绑定。
- 全程 `Copy` 值类型、无 `dyn`、无堆分配（折叠表跨帧复用）。

### 7.7 触发消歧（Chord / Tap / Hold 优先级）
同一物理源可能同时参与多种触发，需确定的消歧规则（否则同输入在不同机器判定不同）：

| 冲突 | 规则 | 依据 |
|---|---|---|
| `Chord` vs 组成单键 | 和弦绑定**优先消费**其所有组成源；单键绑定只在该源未被任何激活和弦占用时触发 | 避免"按 Ctrl+S 时也触发了 S" |
| `Tap` vs `DoubleTap` | 第一次抬起后进入双击窗口（时间轴毫秒阈值 §5.6）；窗口内第二次按下 → `DoubleTap` 并抑制 `Tap`；窗口超时 → 补发 `Tap` | 时间确定、可重放 |
| `Pressed`(瞬发) vs `Hold` | `Pressed` 在按下沿立即触发；`Hold` 在持续 `secs` 后触发；二者可共存 | 作者可选语义 |
- 所有时间阈值走统一时间轴（整数 ns），**消歧结果随录制可字节级重放**——消歧本身也在 golden 对拍范围（§11）。

---

## 8. 手柄深化：触觉 / 自适应扳机 / 陀螺 / 电量 / 热插拔

### 8.1 输出能力（破坏性：手柄从"只进"变"双向"）
现状手柄只有输入（`gamepad.rs`）。新增**输出命令**（内核→后端，见 §4.1 `GamepadOutput`）：
```
pub enum AdaptiveTrigger { Off, Resistance{start:u8,force:u8}, Weapon{start:u8,end:u8,force:u8}, Vibration{..} }
pub enum HapticPriority { Ambient, Gameplay, Critical }
```
- 弱手柄/无触觉：后端空实现（能力探测 `InputBackendCapabilities`），**不报错**。

### 8.2 热插拔 + 玩家槽配对
- `GamepadConnection{Connected,Disconnected}`（`gamepad.rs`）已在词汇表。补**玩家槽**：`PlayerSlot(u8)` ↔ `GamepadId` 配对表（Unity PlayerInput 式），分屏多人/手柄断连重连保持槽位。
- 新设备接入默认分配空槽或等待"按任意键加入"（配对策略，上层）。

### 8.3 设备档案（数据驱动）
- `GamepadSettings`（`gamepad.rs:166`，死区/迟滞/每轴每键覆盖）升级为**可序列化设备档案**，按手柄型号（SDL_GameControllerDB 式 GUID）加载默认死区/映射。
- 陀螺/加速度计：新增 `GamepadAxis` 扩展或独立 `MotionInput{gyro:[f32;3], accel:[f32;3]}`（陀螺瞄准）。保持 `Axis<T>` 确定性存储。
- 电量/音频/触摸板：`GamepadInfo{battery:Option<f32>, has_touchpad:bool, has_gyro:bool, guid:[u8;16]}` 查询 API。

### 8.4 输出调度（async 触觉队列）
触觉/扳机/LED 是**带时长的异步效果**，不能每帧阻塞等待 OS。模型：
- 内核产出 `GamepadOutput` 命令入**命令环**（§4.3 的命令出队列），后端在其线程异步执行，模拟线程不阻塞。
- **优先级与抢占**：同一 `gamepad` 上新效果按 `HapticPriority` 抢占/混合（如"受击强震"抢占"引擎低频轰鸣"）。避免效果互相覆盖导致"手感丢失"。
- **时长与去重**：命令带 `duration` 与可选 `effect_id`；相同 `effect_id` 的重复请求**替换**而非叠加，防止连发把电机顶满。
- **断连安全**：`GamepadConnection::Disconnected` 到达时后端清空该手柄待执行输出，内核侧 `GamepadOutput` 变 no-op。
- 确定性边界：输出是**副作用**，不回灌模拟态，故不进 golden 对拍（§11 只对输入→状态确定性，不对震动波形）。

---

## 9. 无障碍（Accessibility）

| 能力 | 机制 | 层 |
|---|---|---|
| 全量重映射 | Action 层数据驱动绑定（§7.2），玩家可改并持久化 | `prism_input_map` |
| Hold ↔ Toggle 互换 | `Trigger::Hold` ↔ 自动 toggle 包装器 | `prism_input_map` |
| 粘滞键 / 慢速键 | Modifier 时间阈值（基于时间轴 §5） | `prism_input_map` |
| 死区/灵敏度自定义 | 复用 `AxisSettings`/`GamepadSettings`（§1） | `prism_input` + 档案 |
| 单手模式 / 和弦拆分 | `Trigger::Chord` 可拆为顺序触发 | `prism_input_map` |
| 输入提示 glyph | §7.5 随设备切图标 | 上层 |

无障碍是 Action 层的**自然副产品**——因为逻辑面向语义而非物理键，重映射/替代触发零成本接入。

---

## 10. 并发 · 内存 · 零分配（工程化深化）

### 10.1 所有权与线程（总览见 §4.3）
- 内核单一所有权（模拟线程 `&mut`），仅两条 SPSC 环通信，`#![forbid(unsafe_code)]`。无锁、无 unsafe、无优先级反转。

### 10.2 内存布局与零分配
- 事件 `Copy` POD 走 `InputRing`（§5.5）；状态容器 `BTreeSet`/`BTreeMap` 跨帧复用，`press/release/clear`（`button.rs`）复用已分配节点。每帧 `clear` 只清 `just_*` 边沿集，O(本帧边沿数)。
- **热路径分配预算 = 0**：稳态每帧（排空事件环 → tick → 求值 Action）不触发堆分配。分配只在冷路径：设备热插拔、加载/重建绑定折叠表（§7.6）、首次见到新按键/新触点。
- **可选 SoA**：若 profiling 显示 `BTreeMap<T,f32>`（`axis.rs`）在超多轴下成瓶颈，可为固定枚举轴（`GamepadAxis` 仅 6 个）改用定长数组视图，保留 `BTree*` 作稀疏/动态集的后备。默认不改——确定性与简单性优先。

### 10.3 `no_std` 与跨平台数值确定
- 内核维持 `no_std + alloc`、零依赖，手柄自带 `sqrt_f32`（`gamepad.rs`），不依赖 `libm`。
- 时间全整数 ns（§5.6）：**消除浮点在不同 CPU/编译器下的舍入差异**，是 x86/ARM golden 一致（§11）的前提。
- CI 门禁：`cargo tree` 断言内核依赖集为空；`#![forbid(unsafe_code)]` 锁 `unsafe`。

---

## 11. 确定性测试与 golden harness

### 11.1 对拍门禁（CI 红线）
```
录制：真实后端跑一段 → 落盘 Vec<InputEventEnvelope>（JSON/二进制）
重放 A：空状态 → tick 全程 → snapshot_A
重放 B：空状态 → tick 全程 → snapshot_B
断言：snapshot_A == snapshot_B（字节级）           // 自洽确定
跨平台：x86 CI 与 ARM CI 各自重放 → snapshot 相等    // 跨架构确定
```
- 覆盖范围：设备输入→`InputState`、Action 求值→`ActionState`、触发消歧（§7.7）。
- **不覆盖**：触觉波形等副作用（§8.4）——它们不回灌模拟态。

### 11.2 回归语料库（golden corpus）
| 语料 | 验证点 |
|---|---|
| 高轮询率鼠标甩狙（8k Hz 一帧多 motion） | 累加零丢失（§4.6/§5.2），视角终值确定 |
| 和弦 + 单键冲突（Ctrl+S / S） | 消歧确定（§7.7） |
| 双击 vs 单击边界（阈值 ±1ns） | 时间轴消歧稳定（§5.6） |
| 手柄热插拔重连 | 玩家槽保持（§8.2），断连输出清空（§8.4） |
| 回滚 8 帧重算 | `restore+replay` 结果 == 不回滚直算（§5.5） |

### 11.3 性质测试 / fuzz
- **性质**：任意事件流，`tick` 幂等于"排空同一切片"；`just_pressed ⊆ pressed`；`clear` 后 `just_* 为空且 pressed 不变`（对 `button.rs` 不变量）。
- **fuzz**：随机 `InputEventEnvelope` 流喂内核，断言不 panic、无溢出、`Axis::get` 恒在 `[MIN,MAX]`（`axis.rs`）。
- **模型对拍**：Action 层对一个朴素参考实现（线性扫描全绑定）对拍折叠表实现（§7.6），保证优化不改语义。

---

## 12. 性能 / 效果 / 易用 权衡总表

| 维度 | 决策 | 性能 | 效果 | 易用 |
|---|---|---|---|---|
| 内核零依赖 `no_std` | 保持 | 极佳：无 libm、热路径零分配 | — | 后端作者负担略增 |
| `Copy` 封闭 enum 事件 | 保持 + 加信封（§4/§5） | 极佳：POD、可穷举匹配、无 `dyn` | — | 匹配穷举、易重放 |
| `BTree*` 确定性状态 | 保持 | 良：有序迭代 O(log n) | 确定性是 rollback/回归前提 | — |
| 时间戳时间轴 | 新增（§5） | 良：每事件 +16B 时戳 | 高：子帧延迟、重放、lockstep | 录制回放一等公民 |
| Action 映射层 | 新 crate（§7） | 良：每帧求值 O(绑定数)，可缓存 | 高：跨设备/重绑/无障碍 | 极佳：逻辑面向语义 |
| 手柄双向输出 | 破坏性（§8） | 良：输出命令异步 | 极高：触觉/自适应扳机/陀螺 | 空实现兜底 |
| 无锁 SPSC 线程模型 | 新增（§4.3/§10.1） | 极佳：无锁、无优先级反转 | — | 后端作者按队列对接 |
| 事件合并 coalescing | 新增（§4.6） | 极佳：每帧最小充分事件集 | 高：高轮询率零丢失 | 对上层透明 |
| 环形缓冲 + 快照回滚 | 新增（§5.5） | 良：回滚成本 = 帧数×纯 tick，可预算 | 极高：rollback 网络同步 | 网络层只搬运信封 |
| 固定步长累加器 | 新增（§5.6） | 良：整数 ns，无浮点漂移 | 高：帧率无关的确定模拟 | 渲染插值不污染模拟 |
| golden 对拍门禁 | 新增（§11） | — | 确定性成 CI 红线 | 回归可自动化 |

---

## 13. 路线图（输入侧）

| 阶段 | 范围 | 产出 | 验收 |
|---|---|---|---|
| **P0 契约奠基** | 输入信封 + 时间轴 tick（§5.2）、`DeviceId`、SPSC 事件/命令环 + 合并（§4.3/§4.6）、golden harness（§11） | 破坏性改 `InputEvent`→`InputEventEnvelope`；`InputState::tick`；`InputRing` | 单元：事件流→状态确定性对拍（录两次重放字节级一致，x86/ARM 一致） |
| **P1 Action 层** | `prism_input_map`：Action/Context/Modifier/Trigger + 优先级栈 + 折叠表缓存（§7） | 新 crate | 重绑/上下文切换/和弦/长按/消歧用例测试（§7.7）；复用 `AxisSettings` 死区；模型对拍（§11.3） |
| **P1.2 回滚地基** | 快照/回滚/固定步长累加器（§5.5/§5.6） | `snapshot/restore` + `InputRing` 容量规划 | 8 帧回滚重算 == 直算（golden §11.2） |
| **P1.5 手柄双向** | `GamepadOutput`（§8.1）、热插拔 + 玩家槽（§8.2）、设备档案（§8.3） | 输出命令 + 配对表 | DualSense 触觉/自适应扳机真机；断连重连保槽 |
| **P2.5 文本/IME** | 逻辑字符层 + IME 组合事件（§5.4） | `InputEvent::Text`/`Ime*` | 中日韩 IME 组合、UI 文本框 |
| **P3 远期** | Pointer 统一（§5.3）、SDL3/原生/主机/无头后端、Steam Input glyph | 多后端并存 | 后端可切换、事件流跨后端一致 |

> 输入 P0 的信封 + 时间轴必须与窗口 P0 在**同一单调时间基准**上落地，winit 后端（P0.5）同时产出窗口与输入信封。

---

## 14. 风险 → 高性能高效果解法表（输入侧）

| 风险 | 后果 | 解法 | 性能/效果权衡 |
|---|---|---|---|
| 高轮询率鼠标丢样本 | 视角抖动/延迟 | 子帧时间戳 + 一帧内多 `MouseMotion` 全累加（§5.2） | 累加 O(样本数)，无丢失 |
| 重放不一致 | 回归/网络同步失效 | `tick` 纯函数 + `BTree*` 有序 + 时间戳升序 apply（§5.2） | 确定性是硬保证 |
| 手柄无触觉 | 调用崩溃 | 后端空实现兜底（§8.1） | 零成本 no-op |
| Action 层每帧求值开销 | CPU 热 | 绑定按 `ActionId` 预索引（`BTreeMap`），上下文优先级短路消费（§7.3/7.4） | O(激活绑定数)，可缓存 |
| 时间戳单调性被 OS 破坏 | 时间轴错乱 | 内核只接受单调 `Instant`，后端负责钳为单调 | 后端薄层处理 |
| 事件环溢出（事件风暴） | 丢关键事件 | 可合并事件在后端合并（§4.6），不可合并事件永不丢；环按最坏情况预分配；绝不阻塞 OS 泵 | 背压由合并吸收 |
| 回滚重算成本不可控 | 卡顿 | tick 纯函数零分配，回滚成本 = 回滚帧数 × 固定 tick，可静态预算（§5.5） | 可预测、可封顶 |
| 触觉效果互相覆盖 | 手感丢失 | 输出按 `HapticPriority` 抢占/分层，`effect_id` 去重替换（§8.4） | 异步不阻塞模拟 |
| 触发消歧跨平台不一致 | 同输入不同判定 | 消歧全走整数时间轴阈值，纳入 golden 对拍（§7.7/§11.2） | 确定性覆盖消歧 |
| 共享可变态引入锁 | 竞争/反转 | 内核单一所有权（§4.3/§10.1），仅 SPSC 环通信，`#![forbid(unsafe_code)]` | 无锁、无 unsafe |

---

## 15. 实现进度（Implementation Status）

> 本节仅记录内核落地进度，不改动上文设计论证；签名以代码为准。

### 15.1 P0 契约奠基 — ABI 信封层（已落地）
`prism_input` 内核已新增一条显式 ABI 信封层，按本文 §4.1 / §5.2 的"事件入 / 命令出 / 能力探测 / 单调时间轴 / 帧栅栏 drain-to-batch"范式实现，类型独立定义在 input crate 内，**不跨 crate 依赖窗口内核**。保持 `no_std + alloc`、零依赖、无 `unsafe`、确定性约束不变。

落地要点与对草案（附录 A.1）的差异说明：

- **单调时间轴用整数纳秒而非 `Instant`**：草案示意为 `pub timestamp: Instant`，实现改为 `MonotonicTimestamp(pub u64)`（纳秒整数）。理由：录制重放 / lockstep / 跨架构 golden 对拍要求时间戳**字节级稳定**，`std::time::Instant` 不可序列化、不确定、且与 `no_std` 冲突。语义与窗口侧单调时间基准对齐，但类型在本 crate 内独立定义。提供 `ZERO` / `from_nanos` / `as_nanos` / `saturating_duration_since`（时钟回退饱和到 0，不 panic）。
- **新增 `InputEventSequence(pub u64)`**：后端入口全局单调序号，为同一时间戳下的多事件（1k–8k Hz 高轮询率）提供稳定定序裂项，是确定性排序的第二把钥匙（草案 §5.2 的"按 timestamp 升序 apply"在同戳并发时需要它）。`next()` 在 `u64::MAX` 处饱和不回绕。
- **`InputDeviceId(pub u64)`**：对应草案的 `DeviceId`（§4.4），区分多鼠标 / 多键盘 / 多手柄 / 笔。与 `GamepadId`（逻辑槽位）正交。
- **`PlatformInputStamp { timestamp, sequence, source }`**：手动实现 `Ord` / `PartialOrd`，**仅按 `(timestamp, sequence)` 排序**，`source` 不参与定序（保证只按时间轴稳定排序）；`source` 为 `InputSource { Keyboard, Mouse, Touch, Gamepad, Pen, Other }` 的粗粒度路由 / 诊断标签。
- **`InputEventEnvelope { stamp, device, event }`**：在既有 `InputEvent`（`Copy` enum，未改动本体）之上包裹，保持 `Copy`（热路径零装箱）。因 `InputEvent` 携带 `f32` 载荷只能 `PartialEq`，故信封不实现 `Eq`/`Ord`；排序通过 `stamp_order` 比较器或 `registry` 工具完成。
- **`InputBackendCapabilities`（能力探测，§4.1）**：**刻意不与窗口侧 `WindowCapabilities` 同名**，避免并行演进时命名碰撞。字段覆盖文档 §4.1（`raw_motion`/`max_polling_hz`/`rumble`/`adaptive_trigger`/`gyro`/`touchpad`）并补充设备可用性与手柄数量上限（`keyboard`/`mouse`/`touch`/`gamepad`/`max_gamepads`）。提供 `NONE`/`none()`（无头后端 / 降级起点）与 `supports_force_feedback()`/`has_gamepad_output()` 便捷查询。"降级不降模型"：无能力即 `false`/`0`，不 panic。
- **`registry` 帧栅栏（§4.3 drain-to-batch）**：提供纯函数 `sort_envelopes(&mut [InputEventEnvelope])`（**稳定**排序，按 `(timestamp, sequence)`）与 `InputEnvelopeQueue`（累积缓冲 + `drain_sorted()` / `apply_sorted(f)`，排空后保留容量以维持稳态零分配）。`apply_sorted` 与窗口侧 `WindowRegistry::apply_sorted` 对称：一条确定性入口把有序事件喂给调用方自有状态，内核无需知道状态形状。此处尚未落地完整 `InputState::tick`（草案 §5.2，需状态容器 + 边沿清理，属后续里程碑），当前先交付确定性排序 / 批处理地基。

### 15.2 测试
各新增模块内联 `#[cfg(test)] mod tests`，覆盖：时间戳单调 / 饱和、`PlatformInputStamp` 仅按 `(timestamp, sequence)` 定序（source 不影响）、信封 `Copy` 语义、多设备区分、能力结构构造 / 降级、稳定排序确定性（不同输入排列得同一结果）、`InputEnvelopeQueue` 排空保序 / 保容量。`cargo test -p prism_input --offline` 全绿（46 passed）；`cargo build -p prism_input --no-default-features --offline`（no_std）通过；`cargo clippy` 无告警。

### 15.3 未决 / 待主 agent 决策
- `PlatformInputStamp` 的 `Ord` 与 `Eq` 刻意不一致（`Eq` 比较含 `source`，`Ord` 不含）：这是为"仅按时间轴稳定排序"的刻意取舍，`stamp` 不作为 `BTree*` 键使用，故安全；若后续要把 `stamp` 当有序容器键，需重新评估。
- `InputBackendCapabilities` 字段名 `raw_motion` 对应草案 §4.1 的 `raw_input`（语义同：高轮询率原始相对位移）；如需与窗口 / 平台文档统一命名可再调整。
- 完整 `InputState::tick` / `InputRing` / 快照回滚（§5.5）、`GamepadOutput` 命令类型（§4.1/§8）、IME/Text 变体（§5.4）尚未实现，属后续里程碑；本次仅奠定信封 + 时间轴 + 能力 + 帧栅栏地基。

---


## 附录 A. 数据结构草案（示意，非最终签名）

### A.1 设备输入（`prism_input`，破坏性）
```
pub struct InputEventEnvelope { pub timestamp: Instant, pub device: DeviceId, pub event: InputEvent }
pub struct InputState { /* ButtonInput<KeyCode>, ButtonInput<MouseButton>, Axis<GamepadAxis>, Touches ... */ }
impl InputState { pub fn tick(&mut self, now: Instant, events: &[InputEventEnvelope]); }
// InputEvent 增变体：Text{...}, ImePreedit{...}, ImeCommit{...}, Motion(MotionInput)
pub struct MotionInput { pub gyro: [f32;3], pub accel: [f32;3] }
pub enum GamepadOutput { Rumble{..}, TriggerFeedback{..}, LedColor{..}, Haptic{..} }
pub enum AdaptiveTrigger { Off, Resistance{start:u8,force:u8}, Weapon{start:u8,end:u8,force:u8}, Vibration{..} }
pub struct GamepadInfo { pub battery: Option<f32>, pub has_touchpad: bool, pub has_gyro: bool, pub guid: [u8;16] }
pub struct InputRing { /* 固定容量 Copy POD 环，稳态零分配；容量 = 回滚帧 × 每帧最坏事件 */ }
impl InputState { pub fn snapshot(&self) -> InputState; pub fn restore(&mut self, snap: &InputState); }
pub enum HapticPriority { Ambient, Gameplay, Critical }
```

### A.2 Action 映射（`prism_input_map`，新）
```
pub struct InputAction { pub id: ActionId, pub value_kind: ValueKind }       // Button/Axis1D/Axis2D
pub struct Binding { pub source: InputSource, pub modifiers: Vec<Modifier>, pub trigger: Trigger }
pub struct MappingContext { pub priority: i32, pub bindings: BTreeMap<ActionId, Vec<Binding>> }
pub struct ActionState { /* evaluate(contexts, raw, now) -> action_value(id) */ }
pub enum Modifier { Deadzone(AxisSettings), Invert, Scale(f32), ResponseCurve(Curve), SwizzleAxis, Negate }
pub enum Trigger { Pressed, Released, Down, Hold{secs:f32}, Tap, DoubleTap, Chord(Vec<InputSource>) }
pub struct PlayerSlot(pub u8); // ↔ GamepadId 配对
```

---

## 附录 B. 术语速查

| 术语 | 含义 |
|---|---|
| 内核（kernel） | 纯、确定、`no_std + alloc`、零依赖、无 `unsafe` 的类型/状态系统（`prism_input`/`prism_input_map`） |
| 后端（backend） | 独立 crate，翻译 OS 消息→信封、执行输出命令、上报能力（winit/SDL3/原生/主机/无头） |
| 信封（Envelope） | 事件 + `DeviceId` + 单调时间戳的包装，用于路由与重放 |
| 命令（Command） | 内核/应用→后端的指令（`GamepadOutput`） |
| 能力探测（Capabilities） | 后端启动上报的平台能力集合，驱动降级路径 |
| 时间轴（timeline） | 单调时间戳 + 纯函数 `tick`，重放/lockstep/延迟测量的共同地基 |
| Action | 设备无关的语义输入（Jump/Move/Fire），与物理键正交 |
| MappingContext | 带优先级的绑定集合，支持上下文切换与消费屏蔽（UE Mapping Context 式） |
| Modifier / Trigger | Action 绑定的值变换（死区/反转/曲线）与触发条件（长按/连击/和弦） |
| 玩家槽（PlayerSlot） | 分屏多人中 `GamepadId` 的稳定逻辑槽位 |
| 降级不降模型 | 弱平台降的是后端能力结果（无触觉→no-op），内核语义/词汇表不塌缩 |
| SPSC 环 | 单生产者单消费者无锁环形队列；后端↔模拟线程唯一通信方式（§4.3） |
| 事件合并（coalescing） | 后端在入环前把可合并事件（相对 delta）累加、绝对值取最后一条（§4.6） |
| 帧栅栏 drain-to-batch | 模拟每 tick 一次性排空事件环，得到一帧确定事件切片（§4.3/§5.2） |
| InputRing / 快照 / 回滚 | 环形缓冲存近 N 帧事件 + 定帧快照，迟到输入触发 restore+replay（§5.5） |
| 固定步长累加器 | 整数 ns 累加驱动定频模拟，帧率无关、无浮点漂移（§5.6） |
| 折叠表 | Action 层把激活上下文按优先级压成的 `ActionId→生效绑定` 缓存表（§7.6） |
| late-latch 晚采样 | 模拟前最后一刻读输入时间轴，削一帧延迟（§6） |
| golden 对拍 | 录制事件流重放两次 + 跨架构，状态快照须字节级一致（CI 红线，§11） |
