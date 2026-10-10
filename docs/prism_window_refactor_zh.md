# Prism 窗口系统 — 破坏性重构设计（内核·后端 ABI 契约 · 多窗口·HDR·VRR·混合 DPI · 低延迟呈现 · 并发·零分配·golden 对拍）

> 状态：架构提案（Draft，**允许破坏性重构，不保留旧 API**）
> 面向版本：Prism 下一代运行底座（Bevy fork，`pkg/` workspace）
> 覆盖范围：`pkg/prism_window`（窗口内核），以及它与 OS 后端（winit/SDL3/原生/主机/无头）、渲染呈现层（wgpu/`raw_window_handle`）的边界契约。
> 本文依据：对 `pkg/prism_window/src/*.rs` 的逐文件静态阅读（以代码为准，行号见 §1）。实测 `prism_window` 约 1250 行（含 `tests.rs` 265 行）。
> 拆分说明：本文由原「窗口 & 输入」合并设计文档拆分而来，与**输入系统**文档正交。设备输入内核、Action 映射层、确定性输入时间轴、手柄输出、无障碍请见 `docs/prism_input_refactor_zh.md`；本文只讲**窗口**。两份文档共享同一套后端 ABI 范式（命令出·事件入·能力探测）、单调时间轴与 golden 对拍地基。
> 核心命题：**窗口内核保持纯粹、确定、`no_std + alloc`、零依赖、无 `unsafe`；把"平台脏活"全部挡在一条显式 ABI 契约之外。重构不是把平台代码塞进内核，而是把内核升级成一个能承载 AAA 需求（多窗口/HDR/VRR/混合 DPI/低延迟呈现）的权威状态模型 + 一条命令出·事件入的后端边界。**
> 时间轴命题：**窗口事件携带单调时间戳并归属明确的 `WindowId`——同一事件流在任意机器上重放得到字节级一致的窗口运行态。这是回归测试、输入到光子（input-to-photon）延迟测量的共同地基，与输入文档共用同一条时间轴。**
> 并发命题：**后端线程（OS 事件泵）与模拟线程之间只经两条无锁单生产者单消费者（SPSC）队列通信——事件入、命令出；窗口注册表只被模拟线程拥有（single-owner），不加锁、不共享可变态。线程边界即 ABI 边界（§4.5）。**
> 零分配命题：**稳态每帧热路径零堆分配：窗口事件是 `Copy` POD 走环形缓冲，状态容器（`BTreeMap`）跨帧复用。分配只发生在创建/销毁窗口、热插拔显示器等冷路径。**
> 可测命题：**确定性由 golden 对拍门禁锁定（§7）：录制一段 `WindowEventEnvelope` 流 → 重放两次 → 窗口运行态快照字节级一致，且跨平台（x86/ARM）一致。**
> 关联文档：`docs/prism_input_refactor_zh.md`（输入系统）、`docs/prism_window_winit_design_zh.md`（winit 后端落地）、`docs/prism_platform_design_zh.md`（平台/后端共享能力）、`docs/prism_app_design_zh.md`（运行底座）、`docs/prism_render_driver_design_zh.md`（呈现层）。
> 最后更新：2026-10-10

---

## 0. 设计目标与非目标

### 目标（硬约束）
- **顶级次世代 AAA 质量**：多窗口、多显示器、HDR10/scRGB、VRR（G-Sync/FreeSync）、混合 DPI、低延迟呈现（Reflex 式帧节奏）——皆一等公民。
- **内核纯粹性不动摇**：`prism_window` 继续是 **纯、确定、`no_std + alloc`、零依赖、无 `unsafe`** 的类型系统（现状已满足，见 §1）。重构只扩词汇表与契约，不往内核里塞 winit/SDL/OS syscall。
- **一条后端 ABI 契约**：内核与 OS 后端之间只有一条显式、可版本化的"命令出·事件入"边界。后端可替换、可多实现并存、可字节级录制回放。
- **确定性时间轴**：所有窗口事件携带单调时间戳并归属 `WindowId`；内核状态推进是事件流的纯函数。重放/回归对拍/延迟测量共用同一地基（与输入文档一致）。
- **易用**：应用面向语义窗口事件（`Resized`/`Focused`/`ScaleFactorChanged`）与期望态属性（`WindowAttributes`），不碰平台句柄。
- **性能确定**：热路径零分配、`Copy` 事件、`BTreeMap` 确定性迭代（现状已是，见 §1）。拖拽 resize / 跨屏 DPI 连变等事件风暴由后端合并吸收（§4.6）。

### 非目标（主动不做）
- **不把平台后端并入内核**：winit/SDL/`raw_window_handle`/OS syscall 永远在独立后端 crate；内核对其零依赖（现状 Cargo.toml 已零依赖，§1）。
- **不引入运行时反射驱动的事件分发**：窗口事件是封闭 `enum`（`WindowEvent`），`Copy` 且可穷举匹配，不走 `dyn Any` 动态派发。
- **不在内核里做浮点非确定运算**：几何转换已避开 `std` 浮点内建（无 `round`/`floor`/`libm`，§1）。重构保持此不变量，缩放全用整数毫值（milli）。
- **不保留旧 API**：现 `WindowEvent` 不带 window id——这将破坏性修改（§4），不做兼容垫片。
- **不在窗口内核里处理设备输入/Action/触觉**：输入系统独立成册（见 `prism_input_refactor_zh.md`）；窗口内核只处理窗口与显示语义。

---

## 1. 现状基线（代码实测，作为重构起点）

### 1.1 `prism_window`（窗口内核）

| 事实 | 位置 | 说明 | 本次处理 |
|---|---|---|---|
| 零依赖 + `no_std + alloc` + 无 `unsafe`，edition 2024 | `Cargo.toml`；`lib.rs:20` `#![cfg_attr(not(std), no_std)]` | 后端（winit/SDL/native）在独立 crate，翻译平台消息→`WindowEvent`，驱动 `Window::apply`，回读 `WindowAttributes` | **保持**；把这条非正式约定升级为正式 ABI（§4） |
| `WindowId(pub u64)` | `window.rs:13` | 窗口标识 | 保留；成为后端 ABI 的路由键（§4） |
| `WindowAttributes{title,resolution,mode,present_mode,resizable,decorations,transparent,visible,resize_constraints,window_level,composite_alpha_mode,cursor}` + builder | `window.rs:21`、`new:70`、`with_present_mode:94` | 期望态（desired state）；后端据此 realize | 扩字段：HDR/色域、VRR 偏好、父窗口、`raw_window_handle` 契约（§5） |
| `Window{attributes,position,focused,minimized,maximized,occluded,close_requested,cursor_inside,physical_cursor_position}` + `apply(WindowEvent)->bool` | `window.rs:106`、`apply:262` | 运行态；`apply` 返回是否变化 | **保持**此"事件入→状态变更"核心；`apply` 增时间戳入参（§4.2） |
| `WindowEvent`（`Copy` enum）：`Resized/ScaleFactorChanged{scale_factor_milli:u32}/Moved/CloseRequested/Destroyed/Focused/CursorMoved/CursorEntered/CursorLeft/Occluded/ThemeChanged/Minimized/Maximized/Restored` | `event.rs:16`、`ScaleFactorChanged:21` | **window id 不在事件内**，由后端在旁路携带 | **破坏性**：引入 `WindowEventEnvelope{WindowId,timestamp,WindowEvent}`（§4）；多窗口必需 |
| `PresentMode{AutoVsync,AutoNoVsync,Fifo,FifoRelaxed,Immediate,Mailbox}` + `is_vsync` | `mode.rs:34` | 呈现模式，含延迟/撕裂语义注释 | 扩：低延迟/Reflex 式帧节奏策略（§6），VRR 自适应 |
| `WindowMode / WindowLevel / CompositeAlphaMode / WindowTheme` | `mode.rs:6/67/80/96` | 窗口模式/层级/合成 alpha/主题 | 保留；`WindowMode` 增独占全屏+VideoMode 绑定（§5） |
| `WindowResolution{physical,scale_factor:f32}` 默认 1280×720@1.0；`WindowResizeConstraints{min/max}` + `clamp` | `resolution.rs` | 分辨率+缩放+约束 | 保留；缩放改用 `scale_factor_milli` 整数以消非确定浮点（§5） |
| `PhysicalSize/LogicalSize/Position`、`sane_scale`、`to_logical/to_physical`，避开 `round/floor/libm` | `geometry.rs` | 确定性几何 | **保持此不变量**；是跨平台确定性的根基 |
| `Monitor`/`VideoMode`（刷新率等） | `monitor.rs:14/34` | 显示器/视频模式枚举 | 扩：`work_area`、虚拟桌面 `position`、`scale_factor_milli`、HDR 元数据、VRR 刷新率范围（§5） |
| `CursorOptions`/`CursorIcon` 等光标词汇表 | `cursor.rs` | 光标外观/可见/锁定 | 保留；成为 `WindowCommand::SetCursor` 的载荷（§4.2） |

**一句话**：`prism_window` 现状是一个确定性的 "M0 窗口内核"——`WindowAttributes` 期望态 + `Window::apply` 事件驱动运行态 + 确定性几何 + 显示器枚举。三大缺口：**(1) 事件无 window id/时间戳 → 无法多窗口路由与重放；(2) 缺 HDR/VRR/混合 DPI/低延迟呈现词汇表；(3) 内核↔后端边界是非正式约定，未形成可版本化 ABI。**

### 1.2 生态上下文（实测）
- `pkg/` 中**当前无任何 crate 依赖 `prism_window`**（grep 其 Cargo.toml 消费者为零）——它是全新/独立内核，**无存量后端 crate**。这给了破坏性重构最大的自由度：没有下游兼容包袱。
- `pkg/prism_app` 已存在（约 9300 行实现 + 7900 行测试），已**预留但尚未实现** `WinitRunner`——窗口后端落地的首个消费者（见 `prism_window_winit_design_zh.md`）。
- `prism_platform` / `prism_platform_os` 存在（topology/qos/power/security/sysctl），是 OS 级能力，不是窗口系统；窗口后端可复用其能力探测，但不与窗口内核耦合。
- 输入内核 `prism_input` 与 UI 输入层 `prism_ui_input` 独立于窗口内核（见输入文档）；窗口后端（如 winit）通常同时产出窗口事件与输入事件，但二者走各自的信封与状态模型，不塌缩成一个"大 IO 系统"。

---

## 2. 对标优秀项目（取其形，不抄其码）

| 项目 / 规范 | 借鉴点 | Prism 取舍 |
|---|---|---|
| **winit** | 跨平台事件循环抽象、`WindowId`、`raw_window_handle` 契约、scale factor 事件化 | 借其**事件入/属性回读**形态（现状已同构）；不把 winit 并入内核，作为**一个后端实现**，内核定义 ABI 让 winit/SDL/原生并存 |
| **SDL3** | 显示器/HDR 查询、`SDL_HINT` 可调、多窗口 | 借其**显示器/HDR 查询**与多窗口模型；能力以数据驱动探测（§4.3/§5.3） |
| **GLFW** | 极简窗口/上下文、显示器枚举、gamma ramp | 借其**显示器/视频模式**模型（现状已同构，§1.1 `monitor.rs`） |
| **NVIDIA Reflex / AMD Anti-Lag** | 低延迟渲染队列、输入到光子（input-to-photon）测量、帧节奏 | 借其**低延迟呈现策略 + 延迟标记**（§6），内核侧提供 present 策略与时间戳钩子 |
| **Windows DXGI / Vulkan present** | present wait、独占全屏、VRR tearing 控制 | 借其**呈现模式与 present wait 语义**，由 `PresentMode` + `LowLatencyHint` 映射到后端 |
| **DisplayP3 / Rec2020 / ST2084(PQ) / HLG** | HDR 色域与 EOTF 标准 | 借其**HDR 元数据词汇表**（§5.3），内核只描述能力与期望态，realize 由后端 |

**一句话**：窗口侧对标 winit/SDL3/GLFW 的**后端边界与显示能力**，对标 Reflex/DXGI 的**低延迟呈现策略**；把平台能力压成一张可版本化的 ABI 词汇表 + 能力探测，内核保持确定与零依赖。

---

## 3. 分层架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│ 应用 / 游戏逻辑                                                        │
│   查询窗口运行态（Resized/Focus/Occluded），下发期望态（标题/分辨率）   │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │ 查询窗口运行态 / 下发 WindowAttributes
                      ┌──────────▼────────────────────────┐
                      │ 窗口内核  prism_window（重构）      │
                      │  期望态 WindowAttributes(+HDR/VRR) │
                      │  运行态 Window.apply(Envelope)     │
                      │  显示 Monitor/VideoMode(+HDR 元数据)│
                      │  呈现 PresentMode(+低延迟策略)      │
                      └──────────┬────────────────────────┘
                                 │ 命令出 WindowCommand
                                 │ 事件入 WindowEventEnvelope
                      ┌──────────▼────────────────────────┐
                      │ OS 后端（独立 crate，可多实现并存） │
                      │  winit / SDL3 / 原生 / 主机 / 无头   │
                      │  平台消息 → Envelope；执行命令       │
                      │  raw_window_handle 解析 / 能力探测   │
                      └──────────┬────────────────────────┘
                                 │ 原生句柄
                      ┌──────────▼────────────────────────┐
                      │ 渲染呈现层（wgpu / Surface）        │
                      └─────────────────────────────────────┘
```

> 输入系统（`prism_input` / `prism_input_map` / `prism_ui_input`）与窗口内核**正交分层**：同一 OS 后端既产出 `WindowEventEnvelope` 也产出 `InputEventEnvelope`，但两条事件流进入各自的状态模型。输入分层详见 `prism_input_refactor_zh.md`。

### 核心不变量
- **内核纯粹**：`prism_window` 为 `no_std + alloc`、零依赖、无 `unsafe`、确定。平台脏活只在后端 crate。
- **一条边界**：内核↔后端只有"命令出（`WindowCommand`）+ 事件入（`WindowEventEnvelope`）"两个方向；后端可换、可并存、可录制回放。
- **时间轴统一**：窗口事件携带单调时间戳；状态推进是事件流纯函数——重放=回归对拍=延迟测量共用地基（与输入文档同一条时间轴）。
- **降级不降精度**：弱平台降的是**后端能力探测结果**（无 HDR 就走 sRGB，无 VRR 就固定刷新），内核模型与语义**不塌缩**。
- **缩放全整数**：所有 scale factor 用毫整数（`scale_factor_milli:u32`），消除平台浮点差异（§5.2）。

---

## 4. 窗口内核·后端 ABI 契约（破坏性重构核心）

### 4.1 问题（现状）
现状 `WindowEvent`（`event.rs:16`）**不含 window id**，由后端"在旁路携带"（§1.1）。单窗口尚可，多窗口（编辑器多视口、副屏、工具窗）下无法把事件路由到正确 `Window`。且事件无时间戳，无法做延迟测量与重放。

### 4.2 方案：显式信封 + 命令通道
破坏性引入一组契约类型（后端 crate 实现者唯一需要对接的 API 面）：事件信封、命令信封 + 命令结果通道、能力探测。

```
// 事件入：后端 → 内核
pub struct PlatformEventStamp {
    pub timestamp: Instant,      // 单调时间戳（统一时间轴，与输入共用）
    pub sequence: u64,           // 后端入口全局单调序号，同一时间戳下稳定排序
    pub source: EventSource,     // 事件来源（窗口/输入/设备…）
}
pub struct WindowEventEnvelope {
    pub stamp: PlatformEventStamp, // 破坏性：时戳+序号归入 stamp（对齐 winit §5.4）
    pub window: WindowId,          // 破坏性：id 进入信封，不再旁路
    pub event: WindowEvent,        // 现有 Copy enum，保持
}

// 命令出：内核/应用 → 后端（带信封，可确认、可版本化）
pub struct WindowCommandEnvelope {
    pub sequence: u64,           // 命令单调序号，与结果通道配对
    pub issued_at: Instant,      // 发令时刻（单调时间轴）
    pub target: WindowId,        // 目标窗口（Create 时为新分配 id）
    pub command: WindowCommand,
}

// 细粒度命令词汇表由后端 ABI 文档（winit §5.2）拥有；此处列代表性类别：
pub enum WindowCommand {
    Create(WindowAttributes),
    Destroy,
    SetTitle(/* Arc<str>：冷路径可分配 */),
    RequestInnerSize(PhysicalSize),          // OS 可调整/拒绝，结果经通道回报
    SetPresentMode(PresentMode),
    SetCursor(CursorOptions),
    SetHdr(HdrConfig),                       // §5
    RequestRedraw,
    // …SetVisible/SetDecorations/SetMode/SetIme/RequestFocus 等见 winit §5.2
}

// 命令结果：后端 → 内核（desired≠realized 时回报真实结果与原因）
pub struct WindowCommandResult {
    pub sequence: u64,           // 对应命令信封的 sequence
    pub window: Option<WindowId>,
    pub status: CommandStatus,
    pub realized_delta: RealizedWindowDelta, // OS 实际落地的差量
}

pub enum CommandStatus {
    Applied,                     // 已落地
    AcceptedPending,             // 已受理，待 OS 事件回报 Realized
    NoChange,                    // 与当前态一致，无操作
    Adjusted(AdjustmentReason),  // 平台调整（如 144Hz→120Hz、exclusive→borderless）
    Unsupported(Capability),     // 平台不支持该能力
    Denied(PlatformDenial),      // 平台拒绝（如需用户手势、前台激活策略）
    Failed(WindowBackendError),  // 执行失败
}

// 能力探测：后端 → 内核（一次性/热插拔刷新）
pub struct WindowCapabilities {
    pub multi_window: bool,
    pub hdr: HdrSupport,          // None / HDR10 / scRGB ...
    pub vrr: bool,
    pub present_modes: &'static [PresentMode],
    pub raw_input: bool,          // 高轮询率原始鼠标（输入侧消费）
    pub max_polling_hz: u32,
}
```

- `Window::apply` 破坏性改签名为 `apply(&mut self, env: WindowEventEnvelope) -> bool`（或 `apply_event(ts, event)`）；内核只处理属于自己 `WindowId` 的信封。
- 多窗口由上层 `WindowRegistry: BTreeMap<WindowId, Window>`（确定性迭代）持有，按 `env.window` 分发。
- `WindowAttributesDelta`：只带变更字段，后端据此做最小 realize（避免每帧整块对比）。

### 4.3 三态状态模型与后端契约
期望态与实际态必须分离，否则命令会与 OS 回写事件形成反馈环。沿用后端 ABI 权威文档（`prism_window_winit_design_zh.md` §9）的三态模型：

```
Desired   内核/应用希望的配置（意图的唯一真相，内核 WindowAttributes 持有）
Applied   后端最后已调用/受理的请求快照（后端跟踪，不进入内核纯状态）
Realized  OS 当前回报的真实状态（由 WindowEventEnvelope 回灌）
```

例：请求 1600×900 → Desired 立即变 1600×900；后端调 `request_inner_size` 后更新 Applied；OS 发 `Resized(1598×900)` 后才更新 Realized。三态一致即判定该请求**收敛**；不一致且平台允许重试时走有限状态机，不每帧重复 syscall（反馈抑制细则见 winit §9.3）。

差量按约束顺序 realize（结构属性→min/max→mode/monitor→尺寸/位置→可见性/焦点→cursor/IME→redraw；同批冲突命令先折叠，`Destroy` 覆盖其后普通命令）——完整 7 步顺序由 winit §9.2 拥有，本文不复制。

**后端契约三条铁律：**
1. **意图唯一真相在内核**：Desired 的唯一真相是内核 `WindowAttributes`；后端不得改写 Desired，只负责向 OS realize、跟踪 Applied，并把 OS 实际变化回灌为 `WindowEventEnvelope`（更新 Realized），同时经 `WindowCommandResult` 回报命令结果与原因（如 WM 调整/拒绝了尺寸请求）。
2. **事件不丢序不丢时戳**：后端按 OS 到达序封装，附 `PlatformEventStamp`（单调时间戳 + 全局单调 `sequence`）；内核按 (timestamp, sequence) 稳定排序，假定事件流有序。
3. **能力透明**：后端启动即上报 `WindowCapabilities`；内核据此选择降级路径（如无 HDR → `HdrConfig` 被 `Unsupported` 回报并走 sRGB），**不 panic、不假设**。

### 4.4 `raw_window_handle` 契约
渲染层（wgpu）需要原生句柄。约定：窗口内核**不依赖** `raw_window_handle`（保持零依赖），而由后端 crate 实现 `HasWindowHandle`/`HasDisplayHandle` 并通过 `WindowId → handle` 查询暴露给渲染层。内核只持有 `WindowId`，句柄解析是后端职责。

### 4.5 线程与所有权模型（并发深化）
AAA 引擎里 OS 事件循环（必须跑在主线程/UI 线程，如 macOS `NSApp`、Win32 消息泵）与游戏模拟线程是**两条线程**。ABI 边界即线程边界：

```
 主线程 / OS 事件泵（后端拥有）           模拟线程（内核拥有）
 ┌─────────────────────────┐            ┌──────────────────────────┐
 │ winit/SDL event loop     │  events →  │ WindowRegistry            │
 │ 翻译 OS msg → *Envelope   │ (SPSC ring)│ apply()：纯函数推进        │
 │ 执行 WindowCommand        │ ← commands │ 产出 WindowCommand         │
 └─────────────────────────┘  (SPSC ring)└──────────────────────────┘
```

- **单一所有权**：`WindowRegistry` 只被模拟线程 `&mut` 持有——**无 `Arc<Mutex>`、无 `RwLock`、无共享可变态**。消除锁竞争与优先级反转。
- **两条 SPSC 环形队列**：事件入（后端→模拟）、命令出（模拟→后端）。`Copy` POD 事件天然 `Send`，无需深拷贝/装箱。
- **帧栅栏（frame fence）**：模拟线程每 tick 开始时**一次性排空**事件环（drain-to-batch），保证一帧内看到的窗口事件是一个确定的切片。
- **背压**：环满时后端合并/丢弃**可合并**事件（§4.6），绝不阻塞 OS 事件泵（阻塞会导致系统判定"无响应"）。不可合并事件（`CloseRequested`/`Destroyed`）永不丢弃，环按最坏情况预分配容量。
- **主机/无头**：无独立 UI 线程时退化为单线程直连（队列变成同线程 drain），ABI 不变。

> 该线程/所有权模型与输入侧共用同一对 SPSC 环范式；OS 后端把窗口与输入事件分别入各自的环（见输入文档 §4.3）。

### 4.6 事件批处理与合并（coalescing，性能深化）
一帧内 OS 可能产出成百上千条事件（拖拽 resize 连发、跨屏 DPI 连变）。后端在入环前做**保序合并**：

| 事件 | 合并策略 | 理由 |
|---|---|---|
| `Resized` / `Moved` | 同窗口只保留**最后一条** | 中间尺寸无意义，最终态才需 realize |
| `ScaleFactorChanged` | 同窗口保留最后一条 | 同上 |
| `CursorMoved`（绝对位置） | 保留最后一条 | UI 命中测试只关心当前位 |
| 焦点/遮挡/最小化等状态翻转 | 保留**最后一条**（终态） | 中间抖动无意义 |
| `CloseRequested` / `Destroyed` | **永不合并** | 生命周期边沿不可丢 |

合并在后端线程完成，模拟线程拿到的已是"每帧最小充分事件集"。

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
| 延迟标记 | 无 | 统一时间轴在"输入采样→模拟→渲染提交→呈现"打单调时间戳，输出 input-to-photon 直方图 |

### 6.1 帧节奏数学（FramePacer）
- 输入：显示器刷新周期 `T_refresh = 1e9 / refresh_rate_millihertz * 1000`（ns 整数），VRR 范围 `(min,max)`，目标 present 模式。
- 固定刷新（Fifo）：目标提交时刻 = 下一 vblank − 预估 GPU 耗时 − 安全余量，使"提交后尽快呈现"而非"提前一帧排队"。
- VRR：帧率落在 `(min,max)` 内时无需 vsync 等待，`FramePacer` 把目标帧时长钳在范围内，避免跌出 VRR 窗口导致撕裂/卡顿回退。
- 内核只提供"刷新率/VRR 范围 → 目标提交时刻"的纯函数策略；真正的 present wait 由后端执行。

### 6.2 延迟标记（input-to-photon，呈现侧）
统一时间轴（与输入文档共用）在关键节点打单调时间戳：

| 时间戳 | 位置 | 含义 |
|---|---|---|
| `t_sample` | 模拟前最后一刻采样输入时间轴的时刻 | 链路起点（输入侧，见输入文档） |
| `t_sim_begin` / `t_sim_end` | 模拟开始/结束 | CPU 模拟耗时 |
| `t_submit` | 渲染命令提交 | CPU→GPU 边界 |
| `t_present` | 后端 present 返回 / vblank 回灌 | 链路终点 |

- `photon_latency ≈ t_present - t_sample`，逐帧入直方图，暴露 p50/p99。
- 窗口/呈现侧负责 `t_submit`/`t_present` 打点与 present 策略；`t_sample` 的"晚采样（late-latch）"属于输入侧（见 `prism_input_refactor_zh.md` §6）。
- 内核只提供"打点 + 查询"API，不实现 Reflex SDK；后端若支持厂商低延迟扩展（NVIDIA Reflex / AMD Anti-Lag），把 `LowLatencyHint` 映射过去。

---

## 7. 确定性测试与 golden harness（窗口侧）

### 7.1 对拍门禁（CI 红线）
```
录制：真实后端跑一段 → 落盘 Vec<WindowEventEnvelope>（JSON/二进制）
重放 A：空 WindowRegistry → apply 全程 → snapshot_A
重放 B：空 WindowRegistry → apply 全程 → snapshot_B
断言：snapshot_A == snapshot_B（字节级）           // 自洽确定
跨平台：x86 CI 与 ARM CI 各自重放 → snapshot 相等    // 跨架构确定
```
- 覆盖范围：窗口事件→`Window` 运行态（尺寸/位置/焦点/遮挡/缩放/生命周期）、多窗口信封路由、整数缩放换算。
- **不覆盖**：真实 GPU present 时序、HDR 色调映射视觉结果（这些是后端/渲染层职责，不进内核对拍）。

### 7.2 回归语料库（golden corpus）

| 场景 | 断言 |
|---|---|
| 跨屏拖拽（混合 DPI，scale 连变） | `ScaleFactorChanged` 合并 + 整数换算确定（§4.6/§5.2） |
| 拖拽 resize 风暴 | `Resized` 合并到终态，`WindowResizeConstraints::clamp` 确定 |
| 多窗口焦点/遮挡交替 | 信封按 `WindowId` 路由，各 `Window` 运行态独立确定（§4.2） |
| 窗口创建→销毁生命周期 | `CloseRequested`/`Destroyed` 永不合并，边沿成对（§4.6） |
| 全屏⇄窗口模式切换 | `WindowMode` + `VideoMode` 绑定确定 |

### 7.3 性质测试 / fuzz
- **性质**：`apply` 幂等于"排空同一事件切片"；`to_logical(to_physical(x)) ≈ x`（在整数缩放下无漂移）；`clamp` 后尺寸恒在 `[min,max]`。
- **fuzz**：随机 `WindowEventEnvelope` 流喂内核，断言不 panic、无溢出、几何换算不越界。

---

## 8. 性能 / 效果 / 易用 权衡总表

| 维度 | 决策 | 性能 | 效果 | 易用 |
|---|---|---|---|---|
| 内核零依赖 `no_std` | 保持（§0） | 极佳：无 libm、热路径零分配 | — | 后端作者负担略增（需实现 ABI） |
| `Copy` 封闭 enum 事件 | 保持 + 加信封（§4） | 极佳：POD、可穷举匹配、无 `dyn` | — | 匹配穷举、易重放 |
| `BTreeMap` 确定性注册表 | 新增 `WindowRegistry`（§4.2） | 良：有序迭代 O(log n) | 确定性是重放/回归前提 | — |
| 时间戳时间轴 | 新增（§4.2） | 良：每事件 +16B 时戳 | 高：延迟测量、重放 | 录制回放一等公民 |
| 多窗口信封路由 | 破坏性（§4） | 良：`BTreeMap` 分发 | 高：编辑器多视口/副屏 | 清晰 |
| HDR/VRR/低延迟 | 词汇表 + 能力探测（§5/§6） | 良：策略在后端执行 | 极高：次世代显示 | 降级透明（无能力即忽略） |
| 缩放全整数 milli | 破坏性（§5.2） | 良 | 跨平台确定 | 消除浮点差异 |
| 无锁 SPSC 线程模型 | 新增（§4.5） | 极佳：无锁、无优先级反转 | — | 后端作者按队列对接 |
| 事件合并 coalescing | 新增（§4.6） | 极佳：每帧最小充分事件集 | 高：事件风暴零阻塞 | 对上层透明 |
| golden 对拍门禁 | 新增（§7） | — | 确定性成 CI 红线 | 回归可自动化 |

---

## 9. 路线图

| 阶段 | 范围 | 产出 | 验收 |
|---|---|---|---|
| **P0 契约奠基** | 窗口信封（§4.2）、缩放全整数（§5.2）、SPSC 事件/命令环 + 合并（§4.5/§4.6）、golden harness（§7） | 破坏性改 `Window::apply` 签名；`WindowRegistry`；`WindowEventRing` | 单元：窗口事件流→运行态确定性对拍（录两次重放字节级一致，x86/ARM 一致） |
| **P0.5 首个后端** | winit 后端 crate（独立），实现 `WindowCommand`/`WindowEventEnvelope`/`WindowCapabilities` + `raw_window_handle` | `prism_window_winit`（见专文） | 真机开窗、多窗口路由、resize/scale 事件回灌 |
| **P1 多窗口 + 显示器** | `WindowRegistry` 多条目、`parent`、`Monitor` 扩展（work_area/position/scale_milli）（§5.1/§5.2） | 多窗口 API + 显示器枚举扩展 | 编辑器多视口/副屏、跨屏拖拽混合 DPI 确定 |
| **P2 次世代显示** | HDR 元数据/配置（§5.3）、VRR（§5.4）、低延迟策略+帧节奏（§6） | `HdrMetadata`/`HdrConfig`/`VrrMode`/`LowLatencyHint`/`FramePacer` | HDR10 swapchain、VRR 范围内自适应、input-to-photon 直方图 |
| **P3 远期** | SDL3/原生/主机/无头后端并存；独占全屏 + VideoMode 绑定 | 多后端可切换 | 后端可切换、窗口事件流跨后端一致 |

> 输入系统路线图（时间轴 tick、Action 层、回滚、手柄双向等）见 `prism_input_refactor_zh.md`。两者在 P0 共享"信封 + SPSC 环 + golden"地基，P0.5 共享同一个 winit 后端。

---

## 10. 风险 → 高性能高效果解法表

| 风险 | 后果 | 解法 | 性能/效果权衡 |
|---|---|---|---|
| 多窗口事件误路由 | 事件打到错窗口 | 信封强制带 `WindowId`（§4.2），`BTreeMap` 分发，无旁路态 | O(log n) 分发，确定 |
| 重放不一致 | 回归/延迟测量失效 | `apply` 纯函数 + `BTreeMap` 有序 + (时间戳, sequence) 升序 apply（§4.2） | 确定性是硬保证 |
| 浮点缩放跨平台漂移 | 几何像素差 | 缩放全用 `scale_factor_milli:u32`（§5.2），几何避 round/floor（现状保持） | 整数确定，无 libm |
| HDR/VRR 平台不支持 | panic/黑屏 | `WindowCapabilities` 能力探测，无能力即忽略配置回退 sRGB/固定刷新（§4.3/§5.3） | 降级不降模型 |
| 后端把 Desired 权威抢走 | 意图/实际态脱节 | 三态：Desired 唯一真相在内核，后端只跟踪 Applied + 向 OS realize，Realized 由事件回灌，三态一致即收敛（§4.3） | 单一意图源 |
| 内核被平台依赖污染 | 失去确定/`no_std` | ABI 把平台挡在后端 crate；内核 Cargo.toml 维持零依赖（CI 门禁） | 不变量锁定 |
| 时间戳单调性被 OS 破坏 | 时间轴错乱 | 内核只接受单调 `Instant`，后端负责钳为单调（回退时 clamp） | 后端薄层处理 |
| 事件环溢出（resize 风暴） | 丢关键事件 | 可合并事件在后端合并（§4.6），生命周期事件永不丢；环按最坏情况预分配；绝不阻塞 OS 泵 | 背压由合并吸收 |
| 共享可变态引入锁 | 竞争/反转 | 内核单一所有权（§4.5），仅 SPSC 环通信，`#![forbid(unsafe_code)]` | 无锁、无 unsafe |
| present 提交时机不当 | input-to-photon 高 | `FramePacer` 按刷新率/VRR 计算提交时刻（§6.1），`LowLatencyHint` 映射厂商扩展 | 策略在后端，内核纯函数 |

---

## 附录 A. 数据结构草案（示意，非最终签名）

### A.1 窗口（`prism_window`，破坏性）
```
pub struct PlatformEventStamp { pub timestamp: Instant, pub sequence: u64, pub source: EventSource }
pub struct WindowEventEnvelope { pub stamp: PlatformEventStamp, pub window: WindowId, pub event: WindowEvent }
pub struct WindowCommandEnvelope { pub sequence: u64, pub issued_at: Instant, pub target: WindowId, pub command: WindowCommand }
pub enum WindowCommand { Create(WindowAttributes), Destroy, SetTitle(/*Arc<str>*/), RequestInnerSize(PhysicalSize),
                         SetPresentMode(PresentMode), SetCursor(CursorOptions), SetHdr(HdrConfig), RequestRedraw /* 细粒度词汇表见 winit §5.2 */ }
pub struct WindowCommandResult { pub sequence: u64, pub window: Option<WindowId>, pub status: CommandStatus, pub realized_delta: RealizedWindowDelta }
pub enum CommandStatus { Applied, AcceptedPending, NoChange, Adjusted(AdjustmentReason), Unsupported(Capability), Denied(PlatformDenial), Failed(WindowBackendError) }
pub struct WindowCapabilities { pub multi_window: bool, pub hdr: HdrSupport, pub vrr: bool,
                                pub present_modes: &'static [PresentMode], pub raw_input: bool, pub max_polling_hz: u32 }
// WindowAttributes 增字段：parent: Option<WindowId>, hdr: HdrConfig, vrr: VrrMode
// Monitor 增字段：hdr: Option<HdrMetadata>, work_area: PhysicalRect, position: PhysicalPosition, scale_factor_milli: u32
// VideoMode 增字段：refresh_rate_range_millihertz: Option<(u32,u32)>
pub struct HdrMetadata { pub max_luminance_nits: u16, pub min_luminance_milli_nits: u32,
                         pub max_full_frame_nits: u16, pub color_gamut: ColorGamut, pub eotf: Eotf }
pub struct HdrConfig { pub enabled: bool, pub format: HdrFormat, pub reference_white_nits: u16 }
pub enum VrrMode { Disabled, Enabled, AdaptiveWithCap(u32) }
pub enum LowLatencyHint { None, On, OnPlusBoost }
pub struct WindowRegistry { /* BTreeMap<WindowId, Window>，确定性迭代，模拟线程独占 */ }
```

> 输入侧数据结构（`InputEventEnvelope`/`InputState`/`GamepadOutput`/`prism_input_map`）见 `prism_input_refactor_zh.md` 附录 A。

---

## 附录 B. 术语速查

| 术语 | 含义 |
|---|---|
| 内核（kernel） | 纯、确定、`no_std + alloc`、零依赖、无 `unsafe` 的类型/状态系统（本文指 `prism_window`） |
| 后端（backend） | 独立 crate，翻译 OS 消息→信封、执行命令、上报能力（winit/SDL3/原生/主机/无头） |
| 信封（Envelope） | 事件 + `WindowId` + `PlatformEventStamp`（单调时间戳 + 全局序号）的包装，用于路由与重放 |
| 命令结果（CommandResult） | 后端对每条命令的回执：`CommandStatus`（Applied/Adjusted/Unsupported/Denied…）+ 实际落地差量（§4.2） |
| 三态（Desired/Applied/Realized） | 意图在内核（Desired）、后端已受理快照（Applied）、OS 真实回报（Realized）；三态一致即收敛（§4.3，细则 winit §9） |
| 命令（Command） | 内核/应用→后端的 realize 指令（`WindowCommand`） |
| 能力探测（Capabilities） | 后端启动上报的平台能力集合，驱动降级路径 |
| 时间轴（timeline） | 单调时间戳 + 纯函数 `apply`，重放/延迟测量的共同地基（与输入共用） |
| VRR | 可变刷新率（G-Sync/FreeSync），显示器刷新率随帧率浮动 |
| HDR 元数据 | 显示器峰值亮度/色域/EOTF，驱动 HDR swapchain 与色调映射 |
| 降级不降模型 | 弱平台降的是后端能力结果（无 HDR→sRGB），内核语义/词汇表不塌缩 |
| SPSC 环 | 单生产者单消费者无锁环形队列；后端↔模拟线程唯一通信方式（§4.5） |
| 事件合并（coalescing） | 后端在入环前把可合并事件（resize/绝对位置/状态翻转）压成最后一条（§4.6） |
| 帧栅栏 drain-to-batch | 模拟每 tick 一次性排空事件环，得到一帧确定事件切片（§4.5） |
| FramePacer | 根据刷新率/VRR 计算提交时刻的帧节奏器，降 input-to-photon（§6.1） |
| golden 对拍 | 录制事件流重放两次 + 跨架构，状态快照须字节级一致（CI 红线，§7） |

---

## 实现进度（持续更新）

> 本节记录落地状态，与设计正文解耦；每完成一部分追加一次。

### Phase 0 — 内核 ABI + winit 后端骨架（已落地）

- **`prism_window` 内核**：几何/事件/命令/信封/能力词汇表完整，`no_std + alloc`、零依赖、无 `unsafe`、确定性时间戳（`MonotonicTimestamp`）。单测 44 passed、clippy 0 告警。
  - 修复 `PlatformEventStamp`/`EventSource` 的 `Ord`/`Eq` 契约 bug：改 `#[derive(Ord, PartialOrd)]`（字段序 timestamp→sequence→source），`order_key()` 不受 source 影响（`5c9fde44a`）。
- **`pkg/prism_window_winit` 新后端 crate（winit 0.30）已诞生并编译通过**（`71148ded0` + 工作区成员注册 `ebf037e9f`）。模块化目录，禁单文件堆砌：
  | 模块 | 职责 | 对应设计 |
  | --- | --- | --- |
  | `convert.rs` | winit `dpi` ↔ 内核几何的唯一转换边界（含 scale milli-u32、cursor 半进位） | — |
  | `id.rs` | `winit::WindowId ↔ prism WindowId` 双向映射，两侧原子一致 | — |
  | `translate.rs` | winit `WindowEvent` → 内核 `Copy` 事件纯函数；输入设备事件归 `prism_input` | §窗口/输入切分 |
  | `sync.rs` | **三态 Desired/Applied/Realized** 核心：`diff()`/`mark_applied()`/`observe()`，反馈抑制 + clamp 不重发 | §9.3 / §9.4 |
  | `batch.rs` | 每帧 `(window,kind)` last-wins 合并（resize/move/cursor/scale），离散事件保序 | §10.3 |
  | `command.rs` | 命令/枚举 → winit setter 全量映射 + `build_attributes` + `apply` | §命令 realize |
  | `runner.rs` | `WinitBackend` + `WinitRunner`：持 `EventLoop`/live window/每窗 `WindowSync`，`ApplicationHandler` 驱动帧循环（translate→observe→coalesce→host→diff→apply→mark_applied） | §9 / §10 / §14 |
  | `error.rs` | 可恢复 `BackendError`（不 panic 退化） | §19 |
- **验证**：`cargo test -p prism_window_winit --offline` = **39 passed**；`cargo clippy -p prism_window_winit --offline` = **0 告警**。真实 event loop 需显示服务器，沙箱内只编译 + 纯逻辑单测。

### 工程量（实测代码行）

| 范围 | 代码行（含测试） | 其中非测试（约） |
| --- | --- | --- |
| `prism_window` 内核（已存在） | ~2,604 | — |
| `prism_window_winit` 后端（本次完成） | **1,716** | ~1,230 |

> winit 后端本次新增 ~1,716 行（代码 + 内联单测），其中非测试逻辑约 1,230 行。

### 下一步

- `prism_app` 的 `WinitRunner` 集成（后端首个消费者）。
- Surface 句柄桥接（raw-window-handle 0.6）：runner 已持 `Arc<winit::Window>`，向渲染暴露 `window_handle()`/`display_handle()`。
- exclusive fullscreen 的 video-mode 枚举（当前回退 `Borderless`）、present/HDR/IME 的 surface 层接线。
