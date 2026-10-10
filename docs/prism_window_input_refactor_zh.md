# Prism 窗口 & 输入系统 — 破坏性重构设计（v2 / 内核·后端 ABI 契约 · 确定性输入时间轴 · Action 映射层 · 多窗口·HDR·VRR · 低延迟呈现 · 手柄触觉·陀螺 · 无障碍 · 并发·零分配·golden 对拍深化版）

> 状态：架构提案（Draft，**允许破坏性重构，不保留旧 API**）
> 面向版本：Prism 下一代运行底座（Bevy fork，`pkg/` workspace）
> 覆盖范围：`pkg/prism_window`（窗口内核）+ `pkg/prism_input`（输入内核），以及它们与 OS 后端、渲染呈现层、UI 输入层（`prism_ui_input`）的边界契约。
> 本文依据：对 `pkg/prism_window/src/*.rs`、`pkg/prism_input/src/*.rs` 的逐文件静态阅读（以代码为准，行号见 §1）。
> 核心命题：**内核保持纯粹、确定、`no_std + alloc`、零依赖、无 `unsafe`；把"平台脏活"全部挡在一条显式 ABI 契约之外。重构不是把平台代码塞进内核，而是把内核升级成一个能承载 AAA 需求（多窗口/HDR/VRR/低延迟/触觉/陀螺/Action 映射/录制重放/lockstep）的权威状态模型 + 一条命令出·事件入的后端边界。**
> 时间轴命题：**输入与窗口事件携带单调时间戳，内核以确定性时间轴驱动——同一事件流在任意机器上重放得到字节级一致的状态。这是 rollback 网络同步、回归测试、输入延迟测量的共同地基，而不是三套各自为政的机制。**
> 分层命题：**设备级原始输入（`prism_input`）与语义级 Action 映射（新 `prism_input_map`）正交分层；UI 路由（`prism_ui_input`）在其上。三者不互相塌缩成一个"大输入系统"，各自是可独立测试的纯函数内核。**
> 并发命题：**后端线程与模拟线程之间只经两条无锁单生产者单消费者（SPSC）队列通信——事件入、命令出；内核状态只被模拟线程拥有（single-owner），不加锁、不共享可变态。线程边界即 ABI 边界（§4.5）。**
> 零分配命题：**稳态每帧热路径零堆分配：事件是 `Copy` POD 走环形缓冲（§7.5），状态容器（`BTreeSet`/`BTreeMap`）跨帧复用、每帧只做 O(变更量) 的边沿清理。分配只发生在创建窗口/热插拔设备/加载绑定档案等冷路径。**
> 可测命题：**确定性由 golden 对拍门禁锁定（§12）：录制一段 `*Envelope` 流 → 重放两次 → 状态快照字节级一致，且跨平台（x86/ARM）一致。确定性不是口号，是 CI 红线。**
> v2 深化：本版在 v1 骨架上细化**并发与所有权模型（§4.5）**、**事件批处理合并（§4.6）**、**帧节奏数学与 Reflex 标记（§6）**、**环形缓冲·快照回滚·固定步长累加器（§7.5/7.6）**、**Action 求值管线·缓存·触发消歧（§8.6/8.7）**、**手柄输出调度（§9.4）**，并新增 **§11 并发·内存·零分配** 与 **§13 确定性测试与 golden harness**。
> 最后更新：2026-10-08

---

## 0. 设计目标与非目标

### 目标（硬约束）
- **顶级次世代 AAA 质量**：多窗口、多显示器、HDR10/scRGB、VRR（G-Sync/FreeSync）、混合 DPI、低延迟呈现（Reflex 式）、手柄触觉/陀螺/自适应扳机（DualSense）、Action 语义映射、无障碍重映射——皆一等公民。
- **内核纯粹性不动摇**：`prism_window` / `prism_input` 继续是 **纯、确定、`no_std + alloc`、零依赖、无 `unsafe`** 的类型系统（现状已满足，见 §1）。重构只扩词汇表与契约，不往内核里塞 winit/SDL/OS syscall。
- **一条后端 ABI 契约**：内核与 OS 后端（winit/SDL3/原生/主机/无头）之间只有一条显式、可版本化的"命令出·事件入"边界。后端可替换、可多实现并存、可字节级录制回放。
- **确定性时间轴**：所有输入与窗口事件携带单调时间戳；内核状态推进是输入流的纯函数。重放/lockstep/回归对拍共用同一地基。
- **语义正交**：设备输入 ↔ Action 映射 ↔ UI 路由 三层正交，各自纯函数内核，不互相吞并。
- **易用**：游戏逻辑面向 **Action**（`Jump`/`Move`/`Fire`），而非 `KeyCode::Space`；美术/策划在数据层配绑定与设备档案，不改代码。
- **性能确定**：热路径零分配、`Copy` 事件、`BTree*` 确定性迭代（现状已是，见 §1）、每帧边沿清理 O(变更量)。高轮询率鼠标（1k–8k Hz）与原始输入不丢样本。

### 非目标（主动不做）
- **不把平台后端并入内核**：winit/SDL/`raw_window_handle`/OS syscall 永远在独立后端 crate；内核对其零依赖（现状 Cargo.toml 已零依赖，§1）。
- **不做"大一统输入系统"**：拒绝把设备输入、Action 映射、UI 手势塌缩进一个巨型 crate。分层正交是性能与可测性的前提。
- **不引入运行时反射驱动的事件分发**：事件是封闭 `enum`（`WindowEvent`/`InputEvent`），`Copy` 且可穷举匹配，不走 `dyn Any` 动态派发。
- **不在内核里做浮点非确定运算**：几何转换已避开 `std` 浮点内建（无 `round`/`floor`/`libm`，§1）；手柄已自带 `no_std` `sqrt_f32`。重构保持此不变量，禁止为"方便"引回 libm。
- **不保留旧 API**：现 `WindowEvent` 不带 window id、`InputEvent` 不带时间戳——这些将破坏性修改（§4、§7），不做兼容垫片。

---

## 1. 现状基线（代码实测，作为重构起点）

### 1.1 `prism_window`（窗口内核）

| 事实 | 位置 | 说明 | 本次处理 |
|---|---|---|---|
| 零依赖 + `no_std + alloc` + 无 `unsafe`，edition 2024 | `Cargo.toml`；`lib.rs:20` `#![cfg_attr(not(std), no_std)]` | 后端（winit/SDL/native）在独立 crate，翻译平台消息→`WindowEvent`，驱动 `Window::apply`，回读 `WindowAttributes` | **保持**；把这条非正式约定升级为正式 ABI（§4） |
| `WindowId(pub u64)` | `window.rs:13` | 窗口标识 | 保留；成为后端 ABI 的路由键（§4） |
| `WindowAttributes{title,resolution,mode,present_mode,resizable,decorations,transparent,visible,resize_constraints,window_level,composite_alpha_mode,cursor}` + builder | `window.rs:21`、`new:70`、`with_present_mode:94` | 期望态（desired state）；后端据此 realize | 扩字段：HDR/色域、VRR 偏好、父窗口、`raw_window_handle` 契约（§5） |
| `Window{attributes,position,focused,minimized,maximized,occluded,close_requested,cursor_inside,physical_cursor_position}` + `apply(WindowEvent)->bool` | `window.rs:106`、`apply:262` | 运行态；`apply` 返回是否变化 | **保持**此"事件入→状态变更"核心；`apply` 增时间戳入参（§7） |
| `WindowEvent`（`Copy` enum）：`Resized/ScaleFactorChanged{scale_factor_milli:u32}/Moved/CloseRequested/Destroyed/Focused/CursorMoved/CursorEntered/CursorLeft/Occluded/ThemeChanged/Minimized/Maximized/Restored` | `event.rs:16`、`ScaleFactorChanged:21` | **window id 不在事件内**，由后端在旁路携带 | **破坏性**：引入 `WindowEventEnvelope{WindowId,timestamp,WindowEvent}`（§4）；多窗口必需 |
| `PresentMode{AutoVsync,AutoNoVsync,Fifo,FifoRelaxed,Immediate,Mailbox}` + `is_vsync` | `mode.rs:34` | 呈现模式，含延迟/撕裂语义注释 | 扩：低延迟/Reflex 式帧节奏策略（§6），VRR 自适应 |
| `WindowMode / WindowLevel / CompositeAlphaMode / WindowTheme` | `mode.rs:6/67/80/96` | 窗口模式/层级/合成 alpha/主题 | 保留；`WindowMode` 增独占全屏+VideoMode 绑定（§5） |
| `WindowResolution{physical,scale_factor:f32}` 默认 1280×720@1.0；`WindowResizeConstraints{min/max}` + `clamp` | `resolution.rs` | 分辨率+缩放+约束 | 保留；缩放改用 `scale_factor_milli` 整数以消非确定浮点（§5） |
| `PhysicalSize/LogicalSize/Position`、`sane_scale`、`to_logical/to_physical`，避开 `round/floor/libm` | `geometry.rs` | 确定性几何 | **保持此不变量**；是跨平台确定性的根基 |
| `Monitor{id,...,refresh_rate_millihertz:Option}`、`VideoMode{size,bit_depth:u16,refresh_rate_millihertz:u32}`、`best_video_mode` | `monitor.rs:10/14/34/58` | 显示器/视频模式；刷新率毫赫兹整数 | 扩：HDR 元数据（峰值亮度/色域/EOTF）、刷新率范围（VRR）、工作区（§5） |
| `CursorGrabMode{None,Confined,Locked}`、`CursorIcon`（~34 CSS 式）、`CursorOptions` | `cursor.rs:...` | 光标抓取/图标/选项 | 保留；grab/锁定与原始鼠标 delta 联动（§7） |

**一句话**：`prism_window` 现状是一个干净的 "M0 窗口内核"——期望态（`WindowAttributes`）+ 运行态（`Window`）+ 事件（`WindowEvent`）三件套，纯确定零依赖。缺口是 **AAA 显示能力（HDR/VRR/多窗口路由/低延迟）尚未进入词汇表**，且后端边界是注释约定而非正式契约。

### 1.2 `prism_input`（输入内核）

| 事实 | 位置 | 说明 | 本次处理 |
|---|---|---|---|
| 零依赖 + `no_std + alloc` + 无 `unsafe`，edition 2024 | `Cargo.toml`、`lib.rs` | 后端翻译平台→`InputEvent` 流，内核转可查询状态 | **保持**；升级为带时间戳的确定性时间轴（§7） |
| `ButtonInput<T>` 三有序集 `pressed/just_pressed/just_released`（`BTreeSet`）+ `press/release/pressed/just_pressed/just_released/clear/clear_all` | `button.rs:21`（`BTreeSet:22`） | 键盘/鼠标/手柄按键统一模型，**有序→确定性迭代** | **保持此基石**；`clear`（每帧清边沿）纳入时间轴 tick（§7） |
| `Axis<T>` = `BTreeMap<T,f32>`，`MIN=-1/MAX=1`，`get`（钳位）/`get_unclamped`/set/remove | `axis.rs:14`（`MIN/MAX`） | 连续轴存储，确定性 | 保留；Action 映射层消费它（§8） |
| `InputEvent`（`Copy` enum）：`Keyboard/MouseButton/MouseMotion/MouseWheel/Touch/GamepadButton/GamepadAxis/GamepadConnection` | `event.rs:34` | 统一原始事件；按到达序 | **破坏性**：引入 `InputEventEnvelope{timestamp,device_id,InputEvent}`（§7） |
| `ButtonState{Pressed,Released}` + `is_pressed` | `event.rs:14` | 数字输入状态 | 保留 |
| `KeyCode`（`#[non_exhaustive]`，W3C UI Events 物理码位）+ `is_modifier`；`ModifiersState{shift,control,alt,super}`；`KeyboardInput{key_code,state,repeat}` | `keyboard.rs:12/18/131/151` | 物理键位（布局无关）+ 逻辑修饰态 | 保留物理码位；**补逻辑字符层**（IME/文本输入，§7.4） |
| `MouseButton{Left,Right,Middle,Back,Forward,Other(u16)}`；`MouseMotion{delta_x,delta_y}`（原始 delta，供抓取/视角）；`MouseScrollUnit{Line,Pixel}`；`MouseWheel` | `mouse.rs` | 鼠标按键/相对移动/滚轮 | 保留；原始 delta 与高轮询率/子帧时间戳联动（§7.2） |
| `GamepadId(u32)`；`GamepadButton`（南/东/北/西、肩/扳机、Select/Start/guide、摇杆按下、dpad…）；`GamepadAxis{LeftStickX/Y,RightStickX/Y,LeftZ,RightZ}`；`GamepadConnection{Connected,Disconnected}` | `gamepad.rs` | 手柄输入模型 | **扩输出能力**：触觉/自适应扳机/陀螺/电量/音频（§9）——现为纯输入 |
| `AxisSettings{deadzone=0.1,livezone=0.95,threshold=0.01}` + `filter`；`ButtonSettings{press=0.75,release=0.65}` 迟滞；`GamepadSettings` 默认+每轴/每键覆盖；`radial_deadzone(x,y,dz)` 2D 摇杆 | `gamepad.rs:82/92/133/158/166` | 死区/活区/迟滞/径向死区 | 保留；并入设备档案系统（§9.3） |
| 自带 `no_std` `sqrt_f32`/`libm_hypot`（位技巧 + 牛顿迭代，**无 libm 依赖**） | `gamepad.rs`（§摘要） | 确定性径向死区所需 | **保持此不变量**；禁止为便利引回 libm |
| `TouchPhase{Started,Moved,Ended,Canceled}`、`TouchInput`、`Touch{position/start/previous,delta/distance_from_start}`、`Touches`（`BTreeMap` 升序 id 确定性多点追踪） | `touch.rs:14/61/101`（`BTreeMap:10`） | 多点触控逐帧追踪 | 保留；触点时间戳纳入时间轴（§7） |

**一句话**：`prism_input` 现状是一个确定性的 "M0 输入内核"——泛型 `ButtonInput<T>`/`Axis<T>` + 设备词汇表 + `BTree*` 有序状态。三大缺口：**(1) 事件无时间戳/设备 id → 无法做高保真重放与子帧延迟测量；(2) 只有设备级原始输入，缺语义 Action 层；(3) 手柄只进不出（无触觉/陀螺/电量输出通道）。**

### 1.3 生态上下文（实测）
- `prism_ui_input` 是**独立的更高层 crate**（dispatch/gesture/swipe/hit_test/velocity/focus/multitap），负责 UI 层输入路由，与设备级 `prism_input` 不同层。重构后它消费 Action 层或原始层，但不被并入。
- `pkg/` 中**当前无任何 crate 依赖 `prism_window`/`prism_input`**（grep 其 Cargo.toml 消费者为零）——它们是全新/独立内核，**无存量后端 crate**。这给了破坏性重构最大的自由度：没有下游兼容包袱。
- `prism_platform` / `prism_platform_os` 存在（topology/qos/power/security/sysctl），是 OS 级能力，不是窗口系统；窗口后端可复用其能力探测，但不与窗口内核耦合。

---

## 2. 对标优秀项目（取其形，不抄其码）

| 项目 / 规范 | 借鉴点 | Prism 取舍 |
|---|---|---|
| **winit** | 跨平台事件循环抽象、`WindowId`、`raw_window_handle` 契约、scale factor 事件化 | 借其**事件入/属性回读**形态（现状已同构）；不把 winit 并入内核，作为**一个后端实现**，内核定义 ABI 让 winit/SDL/原生并存 |
| **SDL3** | 统一手柄数据库（SDL_GameControllerDB）、触觉 API、显示器/HDR 查询、`SDL_HINT` 可调 | 借其**手柄映射数据库 + 触觉输出通道 + HDR 查询**；档案以数据驱动（§9.3） |
| **GLFW** | 极简窗口/上下文、显示器枚举、gamma ramp | 借其**显示器/视频模式**模型（现状已同构，§1.1 monitor.rs） |
| **UE5 Enhanced Input** | Action/Mapping Context/Modifier/Trigger 分层、运行时上下文优先级、设备无关语义 | **核心借鉴**：Action 映射层（§8）正是其形——`InputAction` + `MappingContext`（优先级栈）+ `Modifier`（死区/反转/响应曲线）+ `Trigger`（按下/长按/连击/和弦） |
| **Unity Input System** | Action Map 切换、Control Scheme、设备热插拔与玩家配对（PlayerInput）、绑定重映射运行时 | 借其**玩家槽配对 + 运行时重映射 + Action Map 切换**（§8.4、§9.2、§10） |
| **Steam Input** | 完全数据驱动的绑定（玩家可改）、动作集、glyph 查询、跨手柄抽象 | 借其**玩家可重绑 + glyph/提示抽象**（§10 无障碍，§8.5 提示） |
| **GGPO / rollback（格斗游戏）** | 确定性输入、输入预测与回滚、lockstep 帧同步 | **核心借鉴**：确定性时间轴（§7）——`BTree*` 有序状态（现状已是）+ 时间戳事件 + 纯函数状态推进 = rollback 地基 |
| **NVIDIA Reflex / Anti-Lag** | 低延迟渲染队列、输入到光子（input-to-photon）测量、帧节奏 | 借其**低延迟呈现策略 + 延迟标记**（§6），内核侧提供 present 策略与时间戳钩子 |
| **Windows.Gaming.Input / GameInput** | 统一手柄/触觉/扳机/陀螺/电量、原始输入高轮询率 | 借其**手柄输出能力词汇表 + 原始输入**（§9） |
| **DualSense（PS5）** | 自适应扳机阻力、HD 触觉、陀螺仪/加速度计、触摸板、扬声器 | 借其**自适应扳机 + 双电机+ 陀螺**词汇表（§9.1） |
| **W3C UI Events / Pointer Events** | 物理 `code` vs 逻辑 `key`、Pointer 统一（鼠标/触控/笔） | 借其**物理码位（现状 KeyCode 已是）+ 逻辑字符分层**（§7.4）、Pointer 统一（§7.3 远期） |

**一句话**：窗口侧对标 winit/SDL3/GLFW 的**后端边界与显示能力**；输入侧对标 Enhanced Input/Unity/Steam 的**Action 分层**与 GGPO 的**确定性时间轴**，以及 GameInput/DualSense 的**手柄输出能力**。

---

## 3. 分层架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│ 应用 / 游戏逻辑                                                        │
│   面向 Action（Jump/Move/Fire），面向语义窗口事件（Resized/Focus）     │
└───────────────┬─────────────────────────────┬────────────────────────┘
                │ 查询 Action 态                │ 查询窗口运行态
┌───────────────▼───────────────┐  ┌──────────▼────────────────────────┐
│ UI 路由层  prism_ui_input       │  │ 窗口内核  prism_window（重构）      │
│  手势/焦点/命中测试/速度         │  │  期望态 WindowAttributes(+HDR/VRR) │
│  （独立，消费 Action 或原始）    │  │  运行态 Window.apply(Envelope)     │
└───────────────┬───────────────┘  │  显示 Monitor/VideoMode(+HDR 元数据)│
                │                   │  呈现 PresentMode(+低延迟策略)      │
┌───────────────▼───────────────┐  └──────────┬────────────────────────┘
│ Action 映射层 prism_input_map  │             │ 命令出 WindowCommand
│  （新）InputAction/MappingCtx  │             │ 事件入 WindowEventEnvelope
│  Modifier/Trigger/优先级栈      │             │
│  设备无关语义，数据驱动绑定      │  ┌──────────▼────────────────────────┐
└───────────────┬───────────────┘  │ OS 后端（独立 crate，可多实现并存） │
                │ 查询原始态          │  winit / SDL3 / 原生 / 主机 / 无头   │
┌───────────────▼───────────────┐  │  平台消息 → Envelope；执行命令       │
│ 设备输入内核 prism_input（重构）│◄─┤  raw_window_handle / 触觉 / 原始输入 │
│  ButtonInput<T>/Axis<T>/Touches │  └─────────────────────────────────────┘
│  确定性时间轴（时间戳事件）      │             ▲
│  手柄输出能力（触觉/扳机/陀螺）  ├─────────────┘ 输出命令 GamepadOutput
└────────────────────────────────┘
```

### 核心不变量
- **内核纯粹**：`prism_window`/`prism_input`/`prism_input_map` 皆 `no_std + alloc`、零依赖、无 `unsafe`、确定。平台脏活只在后端 crate。
- **一条边界**：内核↔后端只有"命令出（`WindowCommand`/`GamepadOutput`）+ 事件入（`*Envelope`）"两个方向；后端可换、可并存、可录制回放。
- **三层正交**：设备输入 → Action 映射 → UI 路由，单向依赖，各自纯函数内核，互不吞并。
- **时间轴统一**：窗口与输入事件共用单调时间戳；状态推进是事件流纯函数——重放=lockstep=回归对拍共用地基。
- **降级不降精度**：弱设备/弱平台降的是**后端能力探测结果**（无 HDR 就走 sRGB，无触觉就空实现），内核模型与语义**不塌缩**。

---

## 4. 窗口内核·后端 ABI 契约（破坏性重构核心）

### 4.1 问题（现状）
现状 `WindowEvent`（`event.rs:16`）**不含 window id**，由后端"在旁路携带"（§1.1）。单窗口尚可，多窗口（编辑器多视口、副屏、工具窗）下无法把事件路由到正确 `Window`。且事件无时间戳，无法做延迟测量与重放。

### 4.2 方案：显式信封 + 命令通道
破坏性引入三个契约类型（后端 crate 实现者唯一需要对接的 API 面）：

```
// 事件入：后端 → 内核
pub struct WindowEventEnvelope {
    pub window: WindowId,        // 破坏性：id 进入信封，不再旁路
    pub timestamp: Instant,      // 单调时间戳（§7 统一时间轴）
    pub event: WindowEvent,      // 现有 Copy enum，保持
}

// 命令出：内核/应用 → 后端
pub enum WindowCommand {
    Create(WindowId, WindowAttributes),
    Apply(WindowId, WindowAttributesDelta), // 增量，避免整块 realize
    SetPresentMode(WindowId, PresentMode),
    SetCursor(WindowId, CursorOptions),
    RequestRedraw(WindowId),
    SetHdr(WindowId, HdrConfig),             // §5
    Destroy(WindowId),
}

// 能力探测：后端 → 内核（一次性/热插拔刷新）
pub struct BackendCapabilities {
    pub multi_window: bool,
    pub hdr: HdrSupport,          // None / HDR10 / scRGB ...
    pub vrr: bool,
    pub present_modes: &'static [PresentMode],
    pub raw_input: bool,          // 高轮询率原始鼠标
    pub max_polling_hz: u32,
}
```

- `Window::apply` 破坏性改签名为 `apply(&mut self, env: WindowEventEnvelope) -> bool`（或 `apply_event(ts, event)`）；内核只处理属于自己 `WindowId` 的信封。
- 多窗口由上层 `WindowRegistry: BTreeMap<WindowId, Window>`（确定性迭代）持有，按 `env.window` 分发。
- `WindowAttributesDelta`：只带变更字段，后端据此做最小 realize（避免每帧整块对比）。

### 4.3 后端契约的三条铁律
1. **后端无状态权威**：期望态唯一真相在内核 `WindowAttributes`；后端只"实现它"并把 OS 实际变化回灌为 `WindowEventEnvelope`（如 WM 拒绝了尺寸请求）。
2. **事件不丢序不丢时戳**：后端按 OS 到达序封装，附单调时间戳；内核假定事件流有序。
3. **能力透明**：后端启动即上报 `BackendCapabilities`；内核据此选择降级路径（如无 HDR → `HdrConfig` 被忽略，走 sRGB），**不 panic、不假设**。

### 4.4 `raw_window_handle` 契约
渲染层（wgpu）需要原生句柄。约定：窗口内核**不依赖** `raw_window_handle`（保持零依赖），而由后端 crate 实现 `HasWindowHandle`/`HasDisplayHandle` 并通过 `WindowId → handle` 查询暴露给渲染层。内核只持有 `WindowId`，句柄解析是后端职责。

### 4.5 线程与所有权模型（并发深化）
AAA 引擎里 OS 事件循环（必须跑在主线程/UI 线程，如 macOS `NSApp`、Win32 消息泵）与游戏模拟线程是**两条线程**。ABI 边界即线程边界：

```
 主线程 / OS 事件泵（后端拥有）           模拟线程（内核拥有）
 ┌─────────────────────────┐            ┌──────────────────────────┐
 │ winit/SDL event loop     │  events →  │ WindowRegistry / InputState│
 │ 翻译 OS msg → *Envelope   │ (SPSC ring)│ tick()：纯函数推进         │
 │ 执行 WindowCommand/Output │ ← commands │ 产出 WindowCommand/Output  │
 └─────────────────────────┘  (SPSC ring)└──────────────────────────┘
```

- **单一所有权**：`WindowRegistry`/`InputState`/`ActionState` 只被模拟线程 `&mut` 持有——**无 `Arc<Mutex>`、无 `RwLock`、无共享可变态**。消除锁竞争与优先级反转。
- **两条 SPSC 环形队列**：事件入（后端→模拟）、命令出（模拟→后端）。`Copy` POD 事件天然 `Send`，无需深拷贝/装箱。
- **帧栅栏（frame fence）**：模拟线程每 tick 开始时**一次性排空**事件环（drain-to-batch），保证一帧内看到的输入是一个确定的切片（§7.2 `tick(now, &[Envelope])`）。
- **背压**：环满时后端合并/丢弃**可合并**事件（§4.6），绝不阻塞 OS 事件泵（阻塞会导致系统判定"无响应"）。不可合并事件（`CloseRequested`/`Destroyed`/按键边沿）永不丢弃，环按最坏情况预分配容量。
- **主机/无头**：无独立 UI 线程时退化为单线程直连（队列变成同线程 drain），ABI 不变。

### 4.6 事件批处理与合并（coalescing，性能深化）
一帧内 OS 可能产出成百上千条事件（1k–8k Hz 鼠标、拖拽 resize 连发）。后端在入环前做**保序合并**：

| 事件 | 合并策略 | 理由 |
|---|---|---|
| `Resized` / `Moved` | 同窗口只保留**最后一条** | 中间尺寸无意义，最终态才需 realize |
| `ScaleFactorChanged` | 同窗口保留最后一条 | 同上 |
| `CursorMoved`（绝对位置） | 保留最后一条 | UI 命中测试只关心当前位 |
| `MouseMotion`（相对 delta） | **累加**不丢（§7.2） | 视角需要全部相对位移之和 |
| `MouseWheel` | 同单位累加 | 滚动量可叠加 |
| 按键/按钮边沿、`Touch` 相位、连接事件 | **永不合并** | 边沿语义不可丢（按下→抬起必须成对） |

合并在后端线程完成，模拟线程拿到的已是"每帧最小充分事件集"。相对 delta 的累加保证**高轮询率零精度损失**（§7.2 的另一面）。

---

## 5. 多窗口 / 多显示器 / HDR / VRR / 混合 DPI

### 5.1 多窗口
- `WindowRegistry`（上层，`BTreeMap<WindowId,Window>`）确定性持有多窗口。
- `WindowAttributes` 增 `parent: Option<WindowId>`（工具窗/模态/子视口）。
- 焦点/遮挡/最小化均已是 per-`Window` 运行态（`window.rs:106`），天然支持多窗口；只需信封路由（§4.2）即可。

### 5.2 多显示器 + 混合 DPI
- `Monitor`（`monitor.rs:34`）已有刷新率；补 `work_area`（排除任务栏）、`position`（虚拟桌面坐标，用于跨屏拖拽）、`scale_factor_milli`。
- **混合 DPI**：窗口跨屏时后端发 `ScaleFactorChanged`（`event.rs:21`，已是 `scale_factor_milli:u32` 整数——**保持整数以避免非确定浮点**）；几何层 `to_logical/to_physical`（`geometry.rs`）据此换算。重构统一**所有缩放用毫整数**（milli），消除 `scale_factor:f32`（`resolution.rs`）带来的平台浮点差异。

### 5.3 HDR / 色域
显示能力进入词汇表：

```
pub struct HdrMetadata {        // Monitor 查询所得
    pub max_luminance_nits: u16,
    pub min_luminance_milli_nits: u32,
    pub max_full_frame_nits: u16,
    pub color_gamut: ColorGamut, // sRGB / DisplayP3 / Rec2020
    pub eotf: Eotf,              // sRGB / PQ(ST2084) / HLG
}
pub struct HdrConfig {          // WindowAttributes 期望态
    pub enabled: bool,
    pub format: HdrFormat,       // HDR10 / scRGB(fp16) / None
    pub reference_white_nits: u16,
}
```

- `Monitor` 增 `hdr: Option<HdrMetadata>`；无 HDR 时为 `None`，渲染层回退 sRGB（降级不降模型）。
- `CompositeAlphaMode`（`mode.rs:80`）与 HDR swapchain 格式协同由后端 realize。

### 5.4 VRR（可变刷新率 / G-Sync / FreeSync）
- `VideoMode`（`monitor.rs:14`）补刷新率**范围**：`refresh_rate_range_millihertz: Option<(u32,u32)>`。
- `WindowAttributes` 增 `vrr: VrrMode{Disabled, Enabled, AdaptiveWithCap(u32 /*max fps*/)}`。
- 呈现策略（§6）在 VRR 下优先 `Mailbox`/`Immediate` 并由帧节奏器控制提交时刻。

---

## 6. 低延迟呈现与帧节奏（Reflex 式）

现状 `PresentMode`（`mode.rs:34`）已覆盖 vsync/no-vsync/mailbox/immediate 且带延迟注释。重构补**主动的延迟管理**（内核提供策略，后端+渲染层执行）：

| 层级 | 现状 | 重构补充 |
|---|---|---|
| 呈现模式 | `PresentMode` 6 态 + `is_vsync` | 保持；增 `LowLatencyHint{None,On,OnPlusBoost}`（Reflex 式）供后端映射到 DXGI/VK present wait |
| 帧节奏 | 无 | `FramePacer`（上层）：根据显示器刷新率（`refresh_rate_millihertz`）+ VRR 范围计算提交时刻，降低"输入到光子"延迟 |
| 延迟标记 | 无 | 时间轴（§7）在"输入采样→模拟→渲染提交→呈现"打单调时间戳，输出 input-to-photon 直方图（诊断对接 `prism_diagnostic`） |
| 输入采样时刻 | 每帧轮询 | 支持**晚采样（late-latch）**：模拟前最后一刻再读输入时间轴，削一帧延迟 |

- 内核职责：提供**时间戳 + present 策略词汇表**；不自己调 OS present API（那是后端）。
- 低延迟与 VRR/HDR 正交：三者皆为 `WindowAttributes`/`BackendCapabilities` 字段，组合由后端 realize。

### 6.1 帧节奏数学（FramePacer）
目标是让 CPU 提交时刻尽量贴近 GPU 需要帧的时刻，削掉"排队等待"那段延迟。

- **固定刷新（无 VRR）**：设刷新周期 `T = 1e9 / refresh_rate_millihertz * 1000`（ns）。`FramePacer` 维护滑动窗口估计 CPU 帧时 `c` 与 GPU 帧时 `g`，令模拟起点 = `下一个 vblank - max(c,g) - 安全余量`。晚于此则跳过本次呈现（避免排队堆积）。
- **VRR 范围内**：帧率落在 `refresh_rate_range_millihertz`（§5.4）区间时直接按真实帧时提交，`FramePacer` 只做**上限钳制**（`AdaptiveWithCap`，防撕裂带外闪烁与功耗失控）。
- **输出**：每帧产出 `pacing_decision`（提交/跳过/等待 ns），对接 `prism_diagnostic` 画帧时间直方图。
- 纯数学、整数 ns、确定；不碰 OS，节奏结果由后端在 present 时执行。

### 6.2 Reflex 式延迟标记（input-to-photon）
时间轴（§7）在关键相位打单调时间戳，构成一条可测量的延迟链：

| 标记点 | 含义 | 用途 |
|---|---|---|
| `t_sample` | 本帧晚采样读输入时间轴的时刻 | 链路起点 |
| `t_sim_begin` / `t_sim_end` | 模拟开始/结束 | CPU 模拟耗时 |
| `t_submit` | 渲染命令提交 | CPU→GPU 边界 |
| `t_present` | 后端 present 返回/vblank 回灌 | 链路终点 |

- `photon_latency ≈ t_present - t_sample`，逐帧入直方图，暴露 p50/p99。
- **晚采样（late-latch）**：`t_sample` 尽量贴近 `t_sim_begin`，用环形缓冲（§7.5）里**截至该时刻**的事件推进状态，削一整帧输入延迟。
- 内核只提供"打点 + 查询"API，不实现 Reflex SDK；后端若支持厂商低延迟扩展（NVIDIA Reflex / AMD Anti-Lag），把 `LowLatencyHint` 映射过去。

---

## 7. 输入确定性：时间戳时间轴 / 录制重放 / lockstep

### 7.1 问题（现状）
现状 `InputEvent`（`event.rs:34`）是 `Copy` 但**无时间戳、无设备 id**。`ButtonInput`/`Axis`/`Touches` 已用 `BTree*` 保证**确定性迭代**（现状最大优点，§1.2）——这正是 rollback/lockstep 需要的一半地基。缺的另一半是**时间戳 + 纯函数状态推进**。

### 7.2 方案：带时间戳信封 + tick 纯函数
```
pub struct InputEventEnvelope {
    pub timestamp: Instant,      // 单调时间戳（子帧精度，支撑高轮询率）
    pub device: DeviceId,        // 破坏性：区分多设备（双手柄/多鼠标）
    pub event: InputEvent,       // 现有 Copy enum，保持
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

### 7.3 Pointer 统一（远期）
鼠标/触控/笔共性抽象（W3C Pointer Events 式）作为**可选上层视图**，不塌缩底层设备事件（`MouseButton`/`TouchInput` 保留）。P2 远期。

### 7.4 逻辑字符 / IME（文本输入）
现状 `KeyCode`（`keyboard.rs:18`）是**物理码位**（布局无关，适合游戏绑定）。文本输入需**逻辑字符层**：破坏性新增 `InputEvent::Text{ ... }` + IME 组合事件（`ImePreedit/ImeCommit`），由后端产出。物理层（游戏绑定）与逻辑层（文本/UI）正交，不互相替代。

### 7.5 环形缓冲 · 快照 · 回滚（rollback 深化）
- **事件环形缓冲**：`InputRing`（固定容量，`Copy` POD）保存最近 `N` 帧的 `InputEventEnvelope`。稳态零分配：写指针绕回覆盖最旧。容量按 `最大回滚帧数 × 每帧最坏事件数` 预分配（例如 8 帧回滚 × 高轮询率上限）。
- **状态快照**：`InputState` 全 `Copy`/`Clone` 且无堆外引用，`snapshot()` 是一次浅拷贝（`BTreeSet`/`BTreeMap` 的 clone 成本 = O(当前按下键数)，通常个位数）。按**定帧频率**存快照（如每确认帧一个），不是每帧。
- **回滚重算**：收到迟到的远端输入后，`restore(snapshot_at(f))` → 用环里 `[f, now]` 的事件 `tick` 重放到当前帧。成本 = `回滚帧数 × 每帧 tick`，因 tick 是纯函数且零分配，可预测、可预算。
- **为何此前就成立**：现状 `BTree*` 有序状态（§1.2）保证重放顺序确定；本节只是把"顺序确定"升级为"可保存/可回放的确定时间线"。这是 GGPO/rollback（§2）的内核地基，**不依赖网络层**——网络层只负责搬运 `InputEventEnvelope`。

### 7.6 固定步长累加器（determinism 深化）
- 模拟以**固定 tick**（如 1/60 s）推进；渲染可变帧率。累加器 `acc += frame_dt`，当 `acc >= TICK` 时跑一次 `tick(now, drain)` 并 `acc -= TICK`，`now` 用整数 ns 推进（**不累积浮点误差**）。
- 一个渲染帧内可能 0 次或多次模拟 tick；每次 tick 从环里取**截至该 tick 时间戳**的事件切片，保证不同帧率下同一录制产生同一模拟序列。
- 渲染插值用 `alpha = acc / TICK`（仅用于显示平滑，不回灌模拟态），确定性不受影响。

---

## 8. Action 映射层 `prism_input_map`（新 crate）

### 8.1 为何独立成层
现状 `prism_input` 只有**设备级原始输入**。游戏逻辑写 `keyboard.pressed(KeyCode::Space)` 会把"跳跃"硬绑到物理键，无法重绑、无法跨设备（手柄南键）、无法做无障碍。对标 UE Enhanced Input / Unity Input System：引入 **Action 语义层**，与设备层正交。

### 8.2 核心模型
```
pub struct InputAction { pub id: ActionId, pub value_kind: ValueKind } // Button/Axis1D/Axis2D
pub struct Binding { pub source: InputSource, pub modifiers: Vec<Modifier>, pub trigger: Trigger }
pub enum InputSource { Key(KeyCode), Mouse(MouseButton), MouseAxis(..), Pad(GamepadButton), PadAxis(GamepadAxis), Touch(..) }
pub enum Modifier { Deadzone(AxisSettings), Invert, Scale(f32), ResponseCurve(Curve), SwizzleAxis, Negate }
pub enum Trigger { Pressed, Released, Hold{secs:f32}, Tap, DoubleTap, Chord(Vec<InputSource>), Down }
pub struct MappingContext { pub priority: i32, pub bindings: BTreeMap<ActionId, Vec<Binding>> }
```

### 8.3 求值：原始态 → Action 态（纯函数）
```
impl ActionState {
    pub fn evaluate(&mut self,
                    contexts: &[MappingContext],   // 按 priority 降序
                    raw: &InputState,               // §7 设备输入
                    now: Instant);
    // 高优先级上下文可"消费"输入，屏蔽低优先级（UE 式 consume）
}
pub fn action_value(&self, id: ActionId) -> ActionValue; // bool / f32 / Vec2
```

- 复用现有 `AxisSettings::filter`（`gamepad.rs:133`，死区/活区）与 `radial_deadzone` 作为 `Modifier::Deadzone`——**不重造死区数学**。
- `Modifier`/`Trigger` 皆纯函数、确定、`no_std`。

### 8.4 上下文优先级栈（Action Map 切换）
- `MappingContext` 带 `priority`：如 `Menu`(100) > `Vehicle`(50) > `OnFoot`(10)。进菜单 push 高优先级上下文，自动屏蔽行走绑定（UE Mapping Context / Unity Action Map 式）。
- 玩家槽（§9.2）各自持上下文栈，分屏多人各自独立。

### 8.5 提示 / glyph（易用）
`ActionId → 当前绑定 → 设备相应 glyph`（Steam Input 式），UI 显示"按 [X] 跳跃"随当前设备自动切键帽图标。此为查询 API，不含资源，由上层提供 glyph 图集。

### 8.6 求值管线与缓存（性能深化）
`evaluate` 每帧 O(激活绑定数)，用三步流水并尽量短路：

1. **上下文折叠**：按 `priority` 降序把激活的 `MappingContext` 折叠成一张 `ActionId → 生效绑定` 的扁平表（`BTreeMap`，确定序）。上下文集不变时此表**缓存复用**（脏标记触发重建），稳态零分配。
2. **源采样**：每条 `Binding` 从 `InputState`（§7）读原始值 → 过 `Modifier` 链（死区/反转/曲线/swizzle）。复用 `AxisSettings::filter`（`gamepad.rs:133`）与 `radial_deadzone`，不重造数学。
3. **触发判定 + 消费**：`Trigger` 产出 `ActionValue`；高优先级上下文可标记"消费"该源，`BTreeSet<InputSource>` 记录已消费集，低优先级遇到即短路跳过（UE consume 式）。
- 复杂度：`O(Σ 激活绑定)`，与总绑定库无关；折叠表让每帧只遍历"当前上下文栈里真正激活的"绑定。
- 全程 `Copy` 值类型、无 `dyn`、无堆分配（折叠表跨帧复用）。

### 8.7 触发消歧（Chord / Tap / Hold 优先级）
同一物理源可能同时参与"单击"与"双击"/"短按"与"长按"/"单键"与"和弦"，需确定的消歧规则（否则同输入在不同机器判定不同）：

| 冲突 | 规则 | 依据 |
|---|---|---|
| `Chord` vs 组成单键 | 和弦绑定**优先消费**其所有组成源；单键绑定只在该源未被任何激活和弦占用时触发 | 避免"按 Ctrl+S 时也触发了 S" |
| `Tap` vs `DoubleTap` | 第一次抬起后进入 `双击窗口`（时间轴毫秒阈值 §7.6）；窗口内第二次按下 → `DoubleTap` 并抑制 `Tap`；窗口超时 → 补发 `Tap` | 时间确定、可重放 |
| `Pressed`(瞬发) vs `Hold` | `Pressed` 在按下沿立即触发；`Hold` 在持续 `secs` 后触发；二者可共存（点按=动作A，长按=动作B 需作者显式用 `Pressed` 的"短按补发"或改绑） | 作者可选语义 |
- 所有时间阈值走统一时间轴（整数 ns），**消歧结果随录制可字节级重放**——消歧本身也在 golden 对拍范围（§12）。

---

## 9. 手柄深化：触觉 / 自适应扳机 / 陀螺 / 电量 / 热插拔

### 9.1 输出能力（破坏性：手柄从"只进"变"双向"）
现状手柄只有输入（`gamepad.rs`）。新增**输出命令**（内核→后端）：
```
pub enum GamepadOutput {
    Rumble { gamepad: GamepadId, low_freq: f32, high_freq: f32, duration: Duration }, // 双电机
    TriggerFeedback { gamepad: GamepadId, trigger: Trigger, mode: AdaptiveTrigger },   // DualSense 自适应扳机
    LedColor { gamepad: GamepadId, rgb: [u8;3] },
    Haptic { gamepad: GamepadId, pattern: HapticPattern },                             // HD 触觉
}
pub enum AdaptiveTrigger { Off, Resistance{start:u8,force:u8}, Weapon{start:u8,end:u8,force:u8}, Vibration{..} }
```
- 弱手柄/无触觉：后端空实现（能力探测 `BackendCapabilities`），**不报错**。

### 9.2 热插拔 + 玩家槽配对
- `GamepadConnection{Connected,Disconnected}`（`gamepad.rs`）已在词汇表。补**玩家槽**：`PlayerSlot(u8)` ↔ `GamepadId` 配对表（Unity PlayerInput 式），分屏多人/手柄断连重连保持槽位。
- 新设备接入默认分配空槽或等待"按任意键加入"（配对策略，上层）。

### 9.3 设备档案（数据驱动）
- `GamepadSettings`（`gamepad.rs:166`，死区/迟滞/每轴每键覆盖）升级为**可序列化设备档案**，按手柄型号（SDL_GameControllerDB 式 GUID）加载默认死区/映射。
- 陀螺/加速度计：新增 `GamepadAxis` 扩展或独立 `MotionInput{gyro:[f32;3], accel:[f32;3]}`（陀螺瞄准）。保持 `Axis<T>` 确定性存储。
- 电量/音频/触摸板：`GamepadInfo{battery:Option<f32>, has_touchpad:bool, ...}` 查询 API。

### 9.4 输出调度（async 触觉队列，性能/效果深化）
触觉/扳机/LED 是**带时长的异步效果**，不能每帧阻塞等待 OS。模型：
- 内核产出 `GamepadOutput` 命令入**命令环**（§4.5 的命令出队列），后端在其线程异步执行，模拟线程不阻塞。
- **优先级与抢占**：同一 `gamepad` 上新效果按 `HapticPriority` 抢占/混合（如"受击强震"抢占"引擎低频轰鸣"；或双电机分配给不同效果层）。避免效果互相覆盖导致"手感丢失"。
- **时长与去重**：命令带 `duration` 与可选 `effect_id`；相同 `effect_id` 的重复请求**替换**而非叠加，防止连发把电机顶满。
- **断连安全**：`GamepadConnection::Disconnected`（`gamepad.rs`）到达时后端清空该手柄的待执行输出，内核侧 `GamepadOutput` 变 no-op（§9.1 的空实现兜底与此一致）。
- 确定性边界：输出是**副作用**，不回灌模拟态，故不进 golden 对拍（§12 只对输入→状态确定性，不对震动波形）。

---

## 10. 无障碍（Accessibility）

| 能力 | 机制 | 层 |
|---|---|---|
| 全量重映射 | Action 层数据驱动绑定（§8.2），玩家可改并持久化 | `prism_input_map` |
| Hold ↔ Toggle 互换 | `Trigger::Hold` ↔ 自动 toggle 包装器 | `prism_input_map` |
| 粘滞键 / 慢速键 | Modifier 时间阈值（基于时间轴 §7） | `prism_input_map` |
| 死区/灵敏度自定义 | 复用 `AxisSettings`/`GamepadSettings`（§1.2） | `prism_input` + 档案 |
| 单手模式 / 和弦拆分 | `Trigger::Chord` 可拆为顺序触发 | `prism_input_map` |
| 输入提示 glyph | §8.5 随设备切图标 | 上层 |

无障碍是 Action 层的**自然副产品**——因为逻辑面向语义而非物理键，重映射/替代触发零成本接入。

---

## 11. 并发 · 内存 · 零分配（工程化深化）

### 11.1 所有权与线程（总览见 §4.5）
| 组件 | 拥有者 | 并发语义 |
|---|---|---|
| OS 事件泵 / `*Backend` | 主线程 | 翻译 + realize；不碰内核态 |
| 事件入环 `InputRing`/`WindowEventRing` | 后端写、模拟读 | 无锁 SPSC，`Copy` POD |
| 命令出环 `WindowCommand`/`GamepadOutput` | 模拟写、后端读 | 无锁 SPSC |
| `WindowRegistry`/`InputState`/`ActionState` | 模拟线程独占 | 单一 `&mut`，无锁 |
| 设备档案 / 绑定表（§8/§9.3） | 加载期构建，运行期只读 | 冷路径分配，热路径只读 |

无共享可变态 ⇒ 无数据竞争、无锁、无优先级反转。多窗口/多手柄只是容器里的多条目，不引入额外线程。

### 11.2 内存布局与零分配
- **事件 = `Copy` POD**：`WindowEvent`（`event.rs:16`）/`InputEvent`（`event.rs:34`）已是 `Copy`。信封加 `Instant`(8–16B)+id，仍是栈值，走环形缓冲无堆分配。
- **状态容器跨帧复用**：`ButtonInput`（`button.rs:21`）三 `BTreeSet` 与 `Axis`/`Touches` 的 `BTreeMap` 只在首次按下/新触点时按需扩节点；稳态 `press/release/clear`（`button.rs`）复用已分配节点。每帧 `clear` 只清 `just_*` 边沿集，O(本帧边沿数)。
- **热路径分配预算 = 0**：稳态每帧（排空事件环 → tick → 求值 Action）不触发堆分配。分配只在冷路径：建窗口、设备热插拔、加载/重建绑定折叠表（§8.6）、首次见到新按键/新触点。
- **可选 SoA**：若 profiling 显示 `BTreeMap<T,f32>`（`axis.rs`）在超多轴下成瓶颈，可为固定枚举轴（`GamepadAxis` 仅 6 个）改用定长数组视图，保留 `BTree*` 作稀疏/动态集的后备。默认不改——确定性与简单性优先。

### 11.3 `no_std` 与跨平台数值确定
- 内核维持 `no_std + alloc`、零依赖（§1 两份 Cargo.toml 实测），几何避 `round/floor/libm`（`geometry.rs`），手柄自带 `sqrt_f32`（`gamepad.rs`）。
- 缩放全整数 milli（§5.2）、时间全整数 ns（§7.6）：**消除浮点在不同 CPU/编译器下的舍入差异**，是 x86/ARM golden 一致（§13）的前提。
- CI 门禁：`cargo tree` 断言内核依赖集为空；`#![forbid(unsafe_code)]` 锁 `unsafe`。

---

## 12. 确定性测试与 golden harness（可测性深化）

### 12.1 对拍门禁（CI 红线）
```
录制：真实后端跑一段 → 落盘 Vec<InputEventEnvelope> + Vec<WindowEventEnvelope>（JSON/二进制）
重放 A：空状态 → tick 全程 → snapshot_A
重放 B：空状态 → tick 全程 → snapshot_B
断言：snapshot_A == snapshot_B（字节级）           // 自洽确定
跨平台：x86 CI 与 ARM CI 各自重放 → snapshot 相等    // 跨架构确定
```
- 覆盖范围：设备输入→`InputState`、Action 求值→`ActionState`、触发消歧（§8.7）、窗口事件→`Window` 运行态。
- **不覆盖**：触觉波形/呈现时序等副作用（§9.4）——它们不回灌模拟态。

### 12.2 回归语料库（golden corpus）
| 语料 | 验证点 |
|---|---|
| 高轮询率鼠标甩狙（8k Hz 一帧多 motion） | 累加零丢失（§4.6/§7.2），视角终值确定 |
| 和弦 + 单键冲突（Ctrl+S / S） | 消歧确定（§8.7） |
| 双击 vs 单击边界（阈值 ±1ns） | 时间轴消歧稳定（§7.6） |
| 跨屏拖拽（混合 DPI，scale 连变） | `ScaleFactorChanged` 合并 + 整数换算确定（§4.6/§5.2） |
| 手柄热插拔重连 | 玩家槽保持（§9.2），断连输出清空（§9.4） |
| 回滚 8 帧重算 | `restore+replay` 结果 == 不回滚直算（§7.5） |

### 12.3 性质测试 / fuzz
- **性质**：任意事件流，`tick` 幂等于"排空同一切片"；`just_pressed ⊆ pressed`；`clear` 后 `just_* 为空且 pressed 不变`（对 `button.rs` 不变量）。
- **fuzz**：随机 `*Envelope` 流喂内核，断言不 panic、无溢出、`Axis::get` 恒在 `[MIN,MAX]`（`axis.rs`）。
- **模型对拍**：Action 层对一个朴素参考实现（线性扫描全绑定）对拍折叠表实现（§8.6），保证优化不改语义。

---

## 13. 性能 / 效果 / 易用 权衡总表

| 维度 | 决策 | 性能 | 效果 | 易用 |
|---|---|---|---|---|
| 内核零依赖 `no_std` | 保持（§0） | 极佳：无 libm、热路径零分配 | — | 后端作者负担略增（需实现 ABI） |
| `Copy` 封闭 enum 事件 | 保持 + 加信封（§4/§7） | 极佳：POD、可穷举匹配、无 `dyn` | — | 匹配穷举、易重放 |
| `BTree*` 确定性状态 | 保持（§1） | 良：有序迭代 O(log n)；换 hash 更快但失确定性 | 确定性是 rollback/回归前提 | — |
| 时间戳时间轴 | 新增（§7） | 良：每事件 +16B 时戳 | 高：子帧延迟、重放、lockstep | 录制回放一等公民 |
| Action 映射层 | 新 crate（§8） | 良：每帧求值 O(绑定数)，可缓存 | 高：跨设备/重绑/无障碍 | 极佳：逻辑面向语义 |
| 多窗口信封路由 | 破坏性（§4） | 良：`BTreeMap` 分发 | 高：编辑器多视口/副屏 | 清晰 |
| HDR/VRR/低延迟 | 词汇表 + 能力探测（§5/§6） | 良：策略在后端执行 | 极高：次世代显示 | 降级透明（无能力即忽略） |
| 手柄双向输出 | 破坏性（§9） | 良：输出命令异步 | 极高：触觉/自适应扳机/陀螺 | 空实现兜底 |
| 缩放全整数 milli | 破坏性（§5.2） | 良 | 跨平台确定 | 消除浮点差异 |
| 无锁 SPSC 线程模型 | 新增（§4.5/§11.1） | 极佳：无锁、无优先级反转 | — | 后端作者按队列对接 |
| 事件合并 coalescing | 新增（§4.6） | 极佳：每帧最小充分事件集 | 高：高轮询率零丢失 | 对上层透明 |
| 环形缓冲 + 快照回滚 | 新增（§7.5） | 良：回滚成本 = 帧数×纯 tick，可预算 | 极高：rollback 网络同步 | 网络层只搬运信封 |
| 固定步长累加器 | 新增（§7.6） | 良：整数 ns，无浮点漂移 | 高：帧率无关的确定模拟 | 渲染插值不污染模拟 |
| golden 对拍门禁 | 新增（§12） | — | 确定性成 CI 红线 | 回归可自动化 |

---

## 14. 路线图

| 阶段 | 范围 | 产出 | 验收 |
|---|---|---|---|
| **P0 契约奠基** | 窗口信封（§4.2）、输入信封+时间轴 tick（§7.2）、缩放全整数（§5.2）、SPSC 事件/命令环 + 合并（§4.5/§4.6）、golden harness（§12） | 破坏性改 `WindowEvent::apply`/`InputEvent`；`WindowRegistry`/`InputState::tick`；`InputRing`/`WindowEventRing` | 单元：事件流→状态确定性对拍（录两次重放字节级一致，x86/ARM 一致） |
| **P0.5 首个后端** | winit 后端 crate（独立），实现 `WindowCommand`/`*Envelope`/`BackendCapabilities` + `raw_window_handle` | `prism_window_winit` | 真机开窗、多窗口路由、resize/scale 事件回灌 |
| **P1 Action 层** | `prism_input_map`：Action/Context/Modifier/Trigger + 优先级栈 + 折叠表缓存（§8） | 新 crate | 重绑/上下文切换/和弦/长按/消歧用例测试（§8.7）；复用 `AxisSettings` 死区；模型对拍（§12.3） |
| **P1.2 回滚地基** | 快照/回滚/固定步长累加器（§7.5/§7.6） | `snapshot/restore` + `InputRing` 容量规划 | 8 帧回滚重算 == 直算（golden §12.2） |
| **P1.5 手柄双向** | `GamepadOutput`（§9.1）、热插拔+玩家槽（§9.2）、设备档案（§9.3） | 输出命令 + 配对表 | DualSense 触觉/自适应扳机真机；断连重连保槽 |
| **P2 次世代显示** | HDR 元数据/配置（§5.3）、VRR（§5.4）、低延迟策略+帧节奏（§6） | `HdrMetadata`/`HdrConfig`/`VrrMode`/`LowLatencyHint`/`FramePacer` | HDR10 swapchain、VRR 范围内自适应、input-to-photon 直方图 |
| **P2.5 文本/IME** | 逻辑字符层 + IME 组合事件（§7.4） | `InputEvent::Text`/`Ime*` | 中日韩 IME 组合、UI 文本框 |
| **P3 远期** | Pointer 统一（§7.3）、SDL3/原生/主机/无头后端、Steam Input glyph | 多后端并存 | 后端可切换、事件流跨后端一致 |

---

## 15. 风险 → 高性能高效果解法表

| 风险 | 后果 | 解法 | 性能/效果权衡 |
|---|---|---|---|
| 多窗口事件误路由 | 事件打到错窗口 | 信封强制带 `WindowId`（§4.2），`BTreeMap` 分发，无旁路态 | O(log n) 分发，确定 |
| 高轮询率鼠标丢样本 | 视角抖动/延迟 | 子帧时间戳 + 一帧内多 `MouseMotion` 全累加（§7.2） | 累加 O(样本数)，无丢失 |
| 重放不一致 | 回归/网络同步失效 | `tick` 纯函数 + `BTree*` 有序 + 时间戳升序 apply（§7.2） | 确定性是硬保证 |
| 浮点缩放跨平台漂移 | 几何像素差 | 缩放全用 `scale_factor_milli:u32`（§5.2），几何避 round/floor（现状保持） | 整数确定，无 libm |
| HDR/VRR 平台不支持 | panic/黑屏 | `BackendCapabilities` 能力探测，无能力即忽略配置回退 sRGB/固定刷新（§4.3/§5.3） | 降级不降模型 |
| 手柄无触觉 | 调用崩溃 | 后端空实现兜底（§9.1） | 零成本 no-op |
| Action 层每帧求值开销 | CPU 热 | 绑定按 `ActionId` 预索引（`BTreeMap`），上下文优先级短路消费（§8.3/8.4） | O(激活绑定数)，可缓存 |
| 后端把状态权威抢走 | 期望态/实际态脱节 | 铁律：期望态唯一真相在内核，后端只 realize + 回灌（§4.3） | 单一数据源 |
| 内核被平台依赖污染 | 失去确定/`no_std` | ABI 把平台挡在后端 crate；内核 Cargo.toml 维持零依赖（CI 门禁） | 不变量锁定 |
| 时间戳单调性被 OS 破坏 | 时间轴错乱 | 内核只接受单调 `Instant`，后端负责钳为单调（回退时 clamp） | 后端薄层处理 |
| 事件环溢出（事件风暴） | 丢关键事件 | 可合并事件在后端合并（§4.6），不可合并事件永不丢；环按最坏情况预分配；绝不阻塞 OS 泵 | 背压由合并吸收 |
| 回滚重算成本不可控 | 卡顿 | tick 纯函数零分配，回滚成本 = 回滚帧数 × 固定 tick，可静态预算（§7.5） | 可预测、可封顶 |
| 触觉效果互相覆盖 | 手感丢失 | 输出按 `HapticPriority` 抢占/分层，`effect_id` 去重替换（§9.4） | 异步不阻塞模拟 |
| 触发消歧跨平台不一致 | 同输入不同判定 | 消歧全走整数时间轴阈值，纳入 golden 对拍（§8.7/§12.2） | 确定性覆盖消歧 |
| 共享可变态引入锁 | 竞争/反转 | 内核单一所有权（§4.5/§11.1），仅 SPSC 环通信，`#![forbid(unsafe_code)]` | 无锁、无 unsafe |

---

## 附录 A. 数据结构草案（示意，非最终签名）

### A.1 窗口（`prism_window`，破坏性）
```
pub struct WindowEventEnvelope { pub window: WindowId, pub timestamp: Instant, pub event: WindowEvent }
pub enum WindowCommand { Create(WindowId, WindowAttributes), Apply(WindowId, WindowAttributesDelta),
                         SetPresentMode(WindowId, PresentMode), SetCursor(WindowId, CursorOptions),
                         SetHdr(WindowId, HdrConfig), RequestRedraw(WindowId), Destroy(WindowId) }
pub struct BackendCapabilities { pub multi_window: bool, pub hdr: HdrSupport, pub vrr: bool,
                                 pub present_modes: &'static [PresentMode], pub raw_input: bool, pub max_polling_hz: u32 }
// WindowAttributes 增字段：parent: Option<WindowId>, hdr: HdrConfig, vrr: VrrMode
// Monitor 增字段：hdr: Option<HdrMetadata>, work_area: PhysicalRect, position: PhysicalPosition
// VideoMode 增字段：refresh_rate_range_millihertz: Option<(u32,u32)>
pub struct HdrMetadata { pub max_luminance_nits: u16, pub min_luminance_milli_nits: u32,
                         pub max_full_frame_nits: u16, pub color_gamut: ColorGamut, pub eotf: Eotf }
pub struct HdrConfig { pub enabled: bool, pub format: HdrFormat, pub reference_white_nits: u16 }
pub enum VrrMode { Disabled, Enabled, AdaptiveWithCap(u32) }
pub enum LowLatencyHint { None, On, OnPlusBoost }
```

### A.2 输入（`prism_input`，破坏性）
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

### A.3 Action 映射（`prism_input_map`，新）
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
| 内核（kernel） | 纯、确定、`no_std + alloc`、零依赖、无 `unsafe` 的类型/状态系统（`prism_window`/`prism_input`/`prism_input_map`） |
| 后端（backend） | 独立 crate，翻译 OS 消息→信封、执行命令、上报能力（winit/SDL3/原生/主机/无头） |
| 信封（Envelope） | 事件 + `WindowId`/`DeviceId` + 单调时间戳的包装，用于路由与重放 |
| 命令（Command） | 内核/应用→后端的 realize 指令（`WindowCommand`/`GamepadOutput`） |
| 能力探测（Capabilities） | 后端启动上报的平台能力集合，驱动降级路径 |
| 时间轴（timeline） | 单调时间戳 + 纯函数 `tick`，重放/lockstep/延迟测量的共同地基 |
| Action | 设备无关的语义输入（Jump/Move/Fire），与物理键正交 |
| MappingContext | 带优先级的绑定集合，支持上下文切换与消费屏蔽（UE Mapping Context 式） |
| Modifier / Trigger | Action 绑定的值变换（死区/反转/曲线）与触发条件（长按/连击/和弦） |
| 玩家槽（PlayerSlot） | 分屏多人中 `GamepadId` 的稳定逻辑槽位 |
| VRR | 可变刷新率（G-Sync/FreeSync），显示器刷新率随帧率浮动 |
| HDR 元数据 | 显示器峰值亮度/色域/EOTF，驱动 HDR swapchain 与色调映射 |
| 降级不降模型 | 弱平台降的是后端能力结果（无 HDR→sRGB），内核语义/词汇表不塌缩 |
| SPSC 环 | 单生产者单消费者无锁环形队列；后端↔模拟线程唯一通信方式（§4.5） |
| 事件合并（coalescing） | 后端在入环前把可合并事件（resize/绝对位置）压成最后一条、相对 delta 累加（§4.6） |
| 帧栅栏 drain-to-batch | 模拟每 tick 一次性排空事件环，得到一帧确定事件切片（§4.5/§7.2） |
| InputRing / 快照 / 回滚 | 环形缓冲存近 N 帧事件 + 定帧快照，迟到输入触发 restore+replay（§7.5） |
| 固定步长累加器 | 整数 ns 累加驱动定频模拟，帧率无关、无浮点漂移（§7.6） |
| 折叠表 | Action 层把激活上下文按优先级压成的 `ActionId→生效绑定` 缓存表（§8.6） |
| FramePacer | 根据刷新率/VRR 计算提交时刻的帧节奏器，降 input-to-photon（§6.1） |
| late-latch 晚采样 | 模拟前最后一刻读输入时间轴，削一帧延迟（§6.2） |
| golden 对拍 | 录制事件流重放两次 + 跨架构，状态快照须字节级一致（CI 红线，§12） |
