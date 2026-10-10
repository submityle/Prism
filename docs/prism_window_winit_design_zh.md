# Prism Window Winit：顶级次世代 AAA 窗口后端设计

> **产品/模块名：`prism_window_winit`**。它是 `prism_window` 的 winit 平台后端，也是窗口客户端连接 `prism_app`、`prism_input`、渲染 Surface 与操作系统事件循环的适配层。
>
> 本文只设计后端边界和集成方案，不把平台逻辑塞回 `prism_window` 内核。`prism_window` 继续保持后端无关、确定性、`no_std + alloc`、无 `unsafe`；`prism_window_winit` 使用 `std + winit`，承接必须在操作系统主线程执行的窗口操作和事件循环。

- 文档状态：架构提案（Draft，允许破坏性调整）
- 目标：Windows、macOS、Linux、Web，以及 winit 能可靠覆盖的移动平台
- 当前基线：仓库已有 `pkg/prism_window` 纯内核；`pkg/prism_app` 已预留但尚未实现窗口化 `WinitRunner`；仓库可参考 `crates/bevy_winit` 的成熟形态，但不复制其 ECS 耦合与线程局部全局状态
- 推荐依赖基线：`winit 0.30`、`raw-window-handle 0.6`；实际落地时统一由 workspace 锁定精确版本
- 关联文档：`docs/prism_window_refactor_zh.md`、`docs/prism_input_refactor_zh.md`、`docs/prism_app_design_zh.md`、`docs/prism_render_driver_design_zh.md`、`docs/prism_platform_design_zh.md`

---

## 1. 定位与职责

### 1.1 一句话定位

**`prism_window_winit` 把 winit 的命令式、平台相关、主线程事件循环，转换为 Prism 的稳定窗口命令、带时间戳事件和生命周期协议。**

```text
                         Prism engine
┌───────────────────────────────────────────────────────────────────┐
│ prism_app        prism_window       prism_input       render/RHI  │
│ lifecycle/frame  desired+realized   device events     surfaces    │
└─────────┬──────────────┬─────────────────┬──────────────┬──────────┘
          │              │                 │              │
          └──────────────┴──── backend ABI ┴──────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ prism_window_winit        │
                    │ runner · adapter · bridge │
                    └─────────────┬─────────────┘
                                  │ winit 0.30
                    ┌─────────────▼─────────────┐
                    │ Win32/Cocoa/X11/Wayland/  │
                    │ Web/iOS/Android           │
                    └───────────────────────────┘
```

### 1.2 它负责什么

1. 创建、修改和销毁原生窗口。
2. 驱动 `winit::event_loop::EventLoop` 与 `ApplicationHandler`。
3. 将 winit 窗口、输入、设备和生命周期事件翻译为 Prism 稳定事件。
4. 将 Prism 的窗口意图差量同步为 winit 调用。
5. 维护 `Prism WindowId ↔ winit WindowId ↔ native window` 映射。
6. 提供受生命周期约束的 raw display/window handle，供 RHI 创建 Surface。
7. 枚举显示器和视频模式，报告平台真实能力与实际结果。
8. 处理 DPI、IME、光标、拖放、触摸板手势和 Web canvas 等平台细节。
9. 将操作系统 suspend/resume、内存压力和退出信号接入 `prism_app`。
10. 为连续、反应式、低功耗、暂停和按需渲染提供事件循环策略。
11. 可选接入 AccessKit，但不拥有 UI 的无障碍语义树。
12. 提供诊断、事件录制、故障降级和跨平台一致性测试入口。

### 1.3 它不负责什么

- 不定义游戏使用的窗口数据模型；该职责属于 `prism_window`。
- 不保存权威窗口配置；后端只保存 OS 对象、映射和同步缓存。
- 不定义 Action Mapping；物理输入事件进入 `prism_input`，语义映射属于 `prism_input_map`。
- 不创建 wgpu/Vulkan/Metal/DX12 Surface；它只安全提供创建 Surface 所需句柄和生命周期通知。
- 不选择 swapchain 格式、Present Mode、HDR 色彩空间或帧延迟；这些属于 RHI/渲染呈现层。
- 不拥有游戏主循环策略；它实现 `prism_app` 定义的窗口 Runner 契约。
- 不在后端中运行 gameplay、UI 或渲染业务逻辑。
- 不承诺 winit 无法跨平台提供的能力，例如所有平台统一的独占全屏、精确 present timestamp 或全局 raw input。

### 1.4 设计成功标准

- **效果**：HDR/VRR、多显示器、混合 DPI、多窗口、IME 和高精度输入不因后端抽象而降级；最终呈现能力由窗口、RHI 和显示设备共同确认。
- **性能**：事件翻译和命令同步不成为帧时间热点，空闲工具不持续耗电，高轮询率输入不丢样。
- **易用**：默认插件即可获得正确窗口循环，复杂平台差异通过 capability、fallback 和结构化诊断解释。
- **易扩展**：统一 ABI 可承载 winit、SDL、原生主机、headless 和 replay 后端。
- **易维护**：平台差异集中、生命周期显式、转换器可穷举测试、winit 升级流程可重复。

- 更换 SDL、原生主机 SDK 或测试后端时，`prism_window`、游戏逻辑和大部分渲染代码不变。
- 稳态事件翻译零堆分配，窗口属性无变化时零系统调用。
- 8 kHz 鼠标原始运动不丢失；常规窗口 resize/move 不形成无界事件洪水。
- 多窗口创建、Surface 初始化、销毁和 GPU fence 回收顺序可证明正确。
- DPI、IME、焦点、最小化、暂停/恢复和显示器热插拔具有明确状态机。
- 同一规范化事件流可录制、重放并驱动 `prism_window`/`prism_input` 得到相同状态。
- 后端失败返回结构化诊断和能力降级，不因平台拒绝某个请求而 panic。

---

## 2. 参考优秀产品与取舍

### 2.1 Bevy `bevy_winit`

借鉴：

- Plugin 安装窗口 Runner。
- winit 事件到引擎窗口/输入事件的完整转换。
- 多窗口映射、显示器实体、光标和 AccessKit 集成。
- focused/unfocused 的连续或反应式更新策略。
- 原生句柄供渲染 Surface 使用。

Prism 改进：

- 不让后端直接依赖具体 ECS 查询和组件变更检测。
- 不使用公开线程局部全局表持有窗口真相。
- 不把 `winit::WindowEvent` 原样复制后再分发到引擎业务层。
- 通过显式 Command/Event ABI 支持 SDL、原生和回放后端。
- desired state、realized state、capability 和 operation result 分离，避免“写入字段即假定 OS 已接受”。
- Window/Event 与 Input/Event 分流，但共享统一时间戳和序列号。

### 2.2 Unreal Engine Application / Slate

借鉴：

- 平台 Application 层与游戏/Slate 语义分离。
- 多窗口、焦点、捕获、IME、拖放和显示器工作区的统一适配。
- foreground/background 生命周期和窗口激活状态驱动节流。
- 低延迟输入采样、事件泵与渲染提交的明确阶段。

Prism 不采用：

- 不把完整 UI 框架、消息路由和窗口后端绑成一个巨型模块。
- 不在后端内构建自己的平台小部件系统。

### 2.3 SDL3

借鉴：

- 稳定的引擎侧事件类型和平台能力查询。
- 明确区分 window ID、device ID、display ID。
- 高 DPI、游戏手柄、触摸与平台生命周期的统一事件面。

取舍：

- `prism_window_winit` 不追求模拟 SDL 全部 API；通用能力进入 Prism ABI，winit 专属能力保留在可选扩展查询中。
- SDL 可成为未来 `prism_window_sdl`，验证 ABI 的后端可替换性。

### 2.4 Chrome/Android/iOS 响应式生命周期

借鉴：

- 可见、隐藏、暂停、恢复、Surface 丢失是正常状态，不视作异常。
- Web 和移动端由宿主事件循环驱动，不能假定传统无限 `run()` 循环。
- 后台节流、资源释放和恢复重建必须由状态机处理。

### 2.5 竞技游戏低延迟实践

借鉴：

- 原始输入尽早采集，模拟尽量靠近提交。
- frame pacing 与 Present Mode/RHI 协同，而不是仅使用 `ControlFlow::Poll` 忙等。
- 记录 Input Sample、Simulation Start、Render Submit、Present 的时间标记。

诚实边界：winit 提供事件循环和窗口，但不是 NVIDIA Reflex、DXGI waitable swapchain、Metal presented time 或 Vulkan present timing 的替代品。低延迟闭环必须跨 `prism_window_winit`、`prism_app` 和 RHI 完成。

---

## 3. 与现有 Prism 模块的关系

### 3.1 与 `prism_window`

```text
prism_window
  纯数据、状态机、WindowId、MonitorId、WindowAttributes、WindowEvent
                   ▲ command/event contract
                   │
prism_window_winit
  原生对象、事件循环、转换、能力探测、执行结果
```

`prism_window` 是权威领域模型；`prism_window_winit` 是可替换执行器。后端不得让 winit 类型泄漏到 `prism_window` 公共 API。

当前 `prism_window` 已实现基础 `Window`、`WindowAttributes`、Monitor、VideoMode 和 `Copy WindowEvent`。为承载生产级后端，建议后续补充：

- `WindowCommand` 与 command sequence。
- desired/realized state 或 operation acknowledgement。
- `WindowEventEnvelope` 的 `WindowId + Timestamp + Sequence`。
- `WindowCapabilities` 与 unsupported/degraded 结果。
- Surface 生命周期事件。
- IME、文件拖放、display changed 等跨后端语义；输入事件仍放 `prism_input`。

这些是内核协议扩展，不意味着内核依赖 winit。

### 3.2 与 `prism_app`

`prism_app` 定义 Runner 和 App 生命周期；`prism_window_winit` 提供实现：

```text
prism_app::Runner contract
              ▲
              │ implements
prism_window_winit::WinitRunner
              │ owns
              ▼
winit EventLoop + ApplicationHandler
```

`WinitRunner` 负责决定什么时候调用：

- `app.resume()` / `app.suspend()`。
- `app.process_platform_events(batch)`。
- `app.update()`。
- `app.render()` 或请求 render sub-app 工作。
- `app.request_exit()`。

Runner 不应反向成为 `prism_app` 的核心依赖。无头服务器和测试不编译 winit。

### 3.3 与 `prism_input`

窗口事件和输入设备事件必须分流：

```text
winit WindowEvent::Resized ───────▶ prism_window::WindowEvent
winit WindowEvent::KeyboardInput ─▶ prism_input::InputEvent
winit DeviceEvent::MouseMotion ───▶ prism_input::RawDeviceEvent
winit WindowEvent::Ime ────────────▶ prism_input::TextInputEvent
winit WindowEvent::Touch ──────────▶ prism_input::TouchEvent
```

共同点是：都由同一后端时钟域加 `PlatformEventStamp`，都保留每设备/每窗口身份与顺序。`prism_window_winit` 只做无损规范化，不做按键绑定、加速度、死区、手势识别或 UI 命中测试。

### 3.4 与渲染/RHI

```text
Native window ready
  → HandleLease available
  → RHI creates Surface
  → Surface configured by renderer
  → Render/Present
  → Window destroy requested
  → stop new frames
  → wait/retire in-flight GPU work
  → drop Surface
  → release HandleLease
  → destroy native window
```

后端提供 `raw-window-handle`，RHI 承担：

- Surface 格式与 alpha mode。
- swapchain image count。
- vsync/present mode。
- HDR/scRGB/PQ 色彩空间与元数据。
- VRR、tearing、frame latency。
- Surface lost/outdated 的重配。
- present timestamp 和 GPU fence。

`WindowAttributes.present_mode` 是用户意图；真正是否支持以及最终选择，由 RHI 回报 realized presentation state。winit 后端不能伪造成功。

### 3.5 与 UI 和无障碍

- UI 系统生成语义树、焦点和 Action。
- 可选 AccessKit 适配层把语义树提交给原生辅助技术。
- winit 后端提供 Window 与 AccessKit 的宿主连接，并把原生 action request 送回 UI。
- 无障碍语义节点不存入 `prism_window_winit`，避免后端与某个 UI 实现绑定。

### 3.6 与 `prism_platform`

`prism_platform` 提供单调时钟、线程、进程和平台能力底座；`prism_window_winit` 负责 winit 专属平台接入。重复能力的权威来源必须唯一：时间戳使用 `prism_platform` 时钟域；winit 的事件到达时间在入口立即采样并映射到该时钟域。

---

## 4. crate 与模块设计

### 4.1 crate

```text
pkg/prism_window_winit/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── plugin.rs             # 装配与 feature 检查
    ├── runner.rs             # EventLoop/ApplicationHandler
    ├── state.rs              # WinitBackendState，主线程所有
    ├── registry.rs           # Prism/Winit ID 映射
    ├── command.rs            # WindowCommand 执行与结果
    ├── event.rs              # 事件入口、批处理与队列
    ├── convert/
    │   ├── window.rs
    │   ├── keyboard.rs
    │   ├── pointer.rs
    │   ├── touch.rs
    │   ├── ime.rs
    │   ├── monitor.rs
    │   └── cursor.rs
    ├── monitor.rs            # 显示器快照与热插拔差分
    ├── handle.rs             # raw handle lease
    ├── pacing.rs             # ControlFlow/Redraw 策略
    ├── accessibility.rs      # 可选 AccessKit 宿主桥
    ├── diagnostics.rs
    ├── recording.rs          # 可选规范事件录制
    └── platform/
        ├── macos.rs
        ├── windows.rs
        ├── linux.rs
        ├── web.rs
        ├── ios.rs
        └── android.rs
```

### 4.2 Feature 建议

```toml
[features]
default = ["x11", "wayland"] # Linux 默认策略需由发行产品确认
x11 = ["winit/x11"]
wayland = ["winit/wayland"]
web = ["winit/web-sys"]
accessibility = ["dep:accesskit", "dep:accesskit_winit"]
serde = ["prism_window/serde", "prism_input/serde"]
recording = ["serde"]
trace = ["dep:tracing"]
custom_cursor = []
platform-extensions = []
```

规则：

- 不用 `cfg` 把业务语义散落在通用模块；平台差异集中在 `platform/` 和 capability 层。
- X11/Wayland 是互相独立 feature，可同时编译并由 winit 运行时选择。
- 无障碍、录制等能力可裁剪，基础窗口路径保持最小依赖。
- 版本由 workspace 统一锁定；不要在多个后端 crate 中各自选择 winit/raw-handle 版本。

### 4.3 依赖方向

```text
prism_window_winit
  ├─ prism_window
  ├─ prism_input
  ├─ prism_app
  ├─ prism_platform
  ├─ prism_diagnostic (optional but recommended)
  ├─ winit
  ├─ raw-window-handle
  └─ accesskit_winit (optional)
```

禁止依赖完整 renderer、游戏 UI 或 asset 系统。自定义光标图像由上层转成后端描述或通过小型接口提交，不能因此让窗口后端依赖整个资产图。

---

## 5. 后端 ABI：命令出、事件入

### 5.1 为什么需要显式 ABI

如果后端直接观察 ECS `Changed<Window>` 并立即调用 winit：

- 创建/失败/OS 拒绝难以表达。
- 同一字段被 OS 事件回写时容易形成反馈循环。
- 无法录制和重放。
- 非 ECS 前端、测试和未来 SDL 后端难以复用。
- 多线程所有权和时序隐含在系统顺序中。

因此定义稳定的 command/event 协议。

### 5.2 窗口命令草案

```rust
pub struct WindowCommandEnvelope {
    pub sequence: CommandSequence,
    pub issued_at: MonotonicTimestamp,
    pub target: WindowTarget,
    pub command: WindowCommand,
}

pub enum WindowCommand {
    Create(CreateWindowRequest),
    Destroy,
    SetTitle(Arc<str>),
    SetVisible(bool),
    SetOuterPosition(PhysicalPosition),
    RequestInnerSize(PhysicalSize),
    SetMinInnerSize(Option<PhysicalSize>),
    SetMaxInnerSize(Option<PhysicalSize>),
    SetResizable(bool),
    SetDecorations(bool),
    SetLevel(WindowLevel),
    SetMode(WindowModeRequest),
    SetTheme(Option<WindowTheme>),
    SetCursor(CursorCommand),
    SetIme(ImeCommand),
    RequestFocus,
    RequestAttention(AttentionKind),
    RequestRedraw,
    BeginDragWindow,
    BeginResizeDrag(ResizeDirection),
    SetContentProtected(bool),
}
```

冷路径命令可以拥有 `Arc<str>` 等分配；热路径事件保持 POD/小对象。不是所有命令在所有平台都支持，结果必须被确认。

### 5.3 命令结果

```rust
pub struct WindowCommandResult {
    pub sequence: CommandSequence,
    pub window: Option<WindowId>,
    pub status: CommandStatus,
    pub realized_delta: RealizedWindowDelta,
}

pub enum CommandStatus {
    Applied,
    AcceptedPending,
    NoChange,
    Adjusted(AdjustmentReason),
    Unsupported(Capability),
    Denied(PlatformDenial),
    Failed(WindowBackendError),
}
```

例如请求 `3840×2160 exclusive fullscreen 144 Hz`，OS 可能给出 120 Hz、borderless 或拒绝。内核不能把 desired 值直接当 realized 值；后端回报真实结果和原因。

### 5.4 事件 Envelope

```rust
pub struct PlatformEventStamp {
    pub timestamp: MonotonicTimestamp,
    pub sequence: PlatformEventSequence,
    pub source: EventSource,
}

pub struct WindowEventEnvelope {
    pub stamp: PlatformEventStamp,
    pub window: WindowId,
    pub event: prism_window::WindowEvent,
}
```

`sequence` 是后端入口处的全局单调序号，用于同一时间戳下稳定排序。时间戳表示**引擎收到平台事件的时刻**，不是硬件采样时刻；平台若提供可信硬件时间，可作为附加字段保留，不能与到达时间混为一谈。

### 5.5 批次

```rust
pub struct PlatformEventBatch<'a> {
    pub lifecycle: &'a [LifecycleEventEnvelope],
    pub windows: &'a [WindowEventEnvelope],
    pub input: &'a [InputEventEnvelope],
    pub raw_devices: &'a [RawDeviceEventEnvelope],
    pub backend: &'a [BackendDiagnosticEvent],
}
```

批次按全局 sequence 提供合并视图，分片仅用于消费者快速筛选。调用 `app.update()` 前冻结本轮批次，更新过程中到达的 user event 进入下一轮，避免可重入修改。

### 5.6 ABI 版本化

- Prism 内部 Rust API 随 workspace 一起编译，不需要人为制造 C ABI。
- 录制文件、跨进程输入代理或远程测试使用独立版本化 wire schema。
- 未知事件使用 length-delimited 跳过或明确拒绝，不能反序列化为默认值。
- 事件语义变化需要升级 protocol version；只增加可选字段可以保持兼容。

---

## 6. 所有权与线程模型

### 6.1 基本约束

在 macOS、iOS、Web 等平台，事件循环和窗口创建必须位于宿主/主线程。winit 原生 Window 也可能不是任意线程可安全操作。因此：

> **`EventLoop`、`ActiveEventLoop`、winit Window registry 和平台适配状态由窗口主线程唯一拥有。**

不要通过 Mutex 把 winit Window 暴露给任意系统调用；“类型上能 Send/Sync”不代表平台语义允许无序跨线程调用。

### 6.2 默认集成模式

```text
OS main thread:
  winit callback
    → timestamp + translate + append event batch
    → drain window commands
    → decide whether App update is due
    → app.update / dispatch render work
    → request_redraw / set_control_flow
```

优点：行为简单，平台兼容最好，没有额外一帧消息延迟。CPU 仿真/渲染内部仍可由 `prism_tasks` 多线程并行。

### 6.3 可选流水线模式

高端客户端可以让主线程只拥有平台泵和少量编排，仿真/渲染子应用在工作线程执行：

```text
Main/window thread                        Simulation/render workers
  collect events ───── bounded SPSC ───▶ consume batch / update
  execute commands ◀── bounded SPSC ──── emit window commands
  native callbacks                        produce render packet
```

约束：

- 不是所有 App 插件都允许离开主线程；启用前进行 capability/trait 检查。
- 队列有界，不能在输入洪水时无限增长。
- 窗口关闭、Surface 销毁和 App Exit 需要握手，不能直接跨线程 drop。
- 默认先交付集成模式；只有基准证明拆线程有收益时再启用流水线模式。

### 6.4 唤醒

后台任务通过 `EventLoopProxy<PrismUserEvent>` 唤醒：

```rust
pub enum PrismUserEvent {
    Wake(WakeReason),
    WindowCommandsAvailable,
    RenderCompleted(WindowId),
    AccessibilityActionPending,
    ExitRequested,
}
```

高频唤醒必须合并：原子 `wake_pending` 从 false→true 时才发送一次 proxy event，窗口线程处理后清零，避免任务完成风暴淹没事件循环。

### 6.5 无全局可变状态

所有后端状态封装在 `WinitApplication`：

```rust
struct WinitApplication {
    app: PrismApp,
    windows: WinitWindowRegistry,
    monitors: MonitorRegistry,
    commands: WindowCommandReceiver,
    events: EventBatchBuilder,
    scheduler: LoopScheduler,
    lifecycle: BackendLifecycle,
    diagnostics: WinitDiagnostics,
}
```

测试可以创建独立 adapter/translator，不依赖 thread-local singleton。确因 winit 生命周期不可构造的部分，通过窄 trait 和 fake 实现测试。

---

## 7. EventLoop 与 App 生命周期

### 7.1 winit 0.30 状态机

使用 `ApplicationHandler<PrismUserEvent>`：

```text
New
  → Resumed
  → Running
  ↔ Suspended
  → Exiting
  → Terminated
```

事件处理入口包括 `resumed`、`suspended`、`window_event`、`device_event`、`user_event`、`about_to_wait` 和 `exiting`。Prism 不假定首次 `resumed` 只发生一次。

### 7.2 `resumed`

1. 将生命周期推进到 `Resuming`。
2. 刷新 monitor snapshot。
3. 创建等待中的 native windows。
4. 发布 `NativeWindowCreated` 和 handle lease 可用事件。
5. 让 RHI 创建/恢复 Surface。
6. 只有最低运行条件满足后发布 `AppResumed`。
7. 请求首帧 redraw。

移动/Web 恢复时 Surface 可能需要重新创建，即使逻辑 WindowId 不变。

### 7.3 `suspended`

1. 停止安排新 present。
2. 发布 `AppSuspending`，允许 RHI 释放短生命周期 Surface。
3. 保存或保留逻辑 Window 状态。
4. 降低后台工作优先级，暂停不允许在后台运行的任务。
5. 清空容易形成“幽灵按键”的 pressed 状态，发合成 focus-lost/release 事件。
6. 进入 `Suspended`，等待新 `resumed`。

不能在 suspend 回调里执行无界等待或同步保存整个游戏；平台可能只给很短时间。

### 7.4 `about_to_wait`

它是提交批次、安排下一次唤醒和选择 `ControlFlow` 的关键点，但不是唯一更新入口：

- 若有 Immediate work、连续模式或 redraw，运行/安排一帧。
- 若仅有事件变化，按 Reactive 策略运行更新。
- 若无工作，根据最近 deadline 使用 `Wait` 或 `WaitUntil`。
- 在进入等待前排空命令，避免窗口命令因无后续事件而永久延迟。

### 7.5 退出

退出分为“请求”和“提交”：

```text
OS close request
  → WindowEvent::CloseRequested
  → game/UI may accept or cancel
  → WindowCommand::Destroy or AppExit accepted
  → stop scheduling window
  → retire Surface/GPU work
  → native window destroyed
  → WindowDestroyed confirmed
  → if exit policy matches, event_loop.exit()
```

不得把 `CloseRequested` 直接等同于销毁，主窗口可弹保存确认，工具窗口可独立关闭。

### 7.6 panic 与致命错误

- Runner 顶层记录最后平台事件、窗口状态和 backend phase。
- 可恢复的窗口创建失败只影响该请求。
- EventLoop 创建失败、关键主窗口无法创建等根据 `StartupPolicy` 转为 AppExit 和结构化错误。
- 不尝试在 panic unwinding 中执行复杂 GPU/窗口调用；使用最小化 shutdown path。

---

## 8. Window Registry 与生命周期

### 8.1 ID 映射

```rust
struct WinitWindowEntry {
    prism_id: prism_window::WindowId,
    winit_id: winit::window::WindowId,
    window: Arc<winit::window::Window>,
    desired_revision: u64,
    applied_snapshot: AppliedWindowSnapshot,
    lifecycle: NativeWindowLifecycle,
    surface_leases: u32,
}
```

映射必须双向 O(1)。`WindowId` 在进程生命周期内不复用，或使用 generation 防止迟到事件命中新窗口。

### 8.2 生命周期

```text
Declared
  → CreateQueued
  → NativeCreated
  → SurfacePending
  → Ready
  → CloseRequested
  → RetiringSurface
  → DestroyQueued
  → Destroyed
  → Tombstone
```

Tombstone 短期保留 winit ID 和 generation，用于识别迟到事件并输出低级诊断，而不是误路由。

### 8.3 创建窗口

- 只能在 `ActiveEventLoop` 可用时执行；此前请求进入队列。
- 创建请求先规范化并验证 min/max size、透明、父窗口、平台扩展。
- 后端返回真实 scale factor、inner/outer size、position、theme、focus 和 monitor。
- `NativeCreated` 早于 `Ready`；只有 Surface/关键附件就绪后才进入 Ready。
- 创建多个窗口时限制单轮数量，避免编辑器恢复布局造成长时间事件泵阻塞。

### 8.4 销毁窗口

Rust drop 顺序不是完整的 Surface 协议。后端需要 `SurfaceLease` 或 renderer acknowledgement：

- 销毁请求后停止生成新 render requests。
- RHI 释放或退休 Surface。
- 所有 handle lease 归零后 drop native window。
- 设置超时只用于诊断，不能在仍有可能访问句柄时强制销毁。
- App 强制退出时走平台允许的最小安全路径，由 OS 回收剩余资源。

### 8.5 父子窗口与工具窗口

编辑器需要主窗口、Viewport、浮动 Inspector、模态对话框。通用模型应支持：

- owner/parent relationship。
- modality policy。
- taskbar visibility。
- transient/tool window intent。
- always-on-top level。
- close cascade policy。

winit/平台不支持的能力返回 `Unsupported/Adjusted`；平台扩展放在受控 feature 后，不污染通用核心。

---

## 9. Desired、Applied 与 Realized 状态同步

### 9.1 三态模型

```text
Desired  用户/引擎希望的配置
Applied  后端最后已调用/接受的请求快照
Realized OS 当前报告的真实状态
```

例如用户请求 1600×900：

- Desired 立即变为 1600×900。
- 后端调用 `request_inner_size` 后更新 Applied。
- OS 发 `Resized(1598×900)` 后才更新 Realized。

这避免命令反馈循环和平台调整被覆盖。

### 9.2 差量应用顺序

一批属性变化按约束顺序执行：

1. parent/level/decorations/resizable 等结构属性。
2. min/max constraints。
3. mode/monitor/video mode。
4. inner size/outer position/maximized/minimized。
5. visibility/focus/attention。
6. cursor/IME。
7. redraw。

同一批次中的冲突命令先折叠，例如连续三个标题更新只执行最后一个；`Destroy` 覆盖其后普通属性命令。

### 9.3 反馈抑制

- 每个命令有 sequence 和 desired revision。
- OS 事件永远进入 Realized，不因“看起来由我们触发”而完全丢弃。
- 若 Realized 与 Desired 一致，标记请求收敛。
- 若不一致且平台允许重试，使用有限状态机，不在每帧重复 syscall。
- 连续拒绝后进入 Degraded，直到 desired revision 再次变化。

### 9.4 平台拒绝与调整

所有调整可观察：

```text
Requested exclusive fullscreen → Adjusted to borderless
Requested 0×0 → clamped to 1×1
Requested cursor lock → Denied until user gesture (Web)
Requested focus → Denied by foreground activation policy
Requested HDR → Unsupported by RHI/display
```

不应刷屏；相同 window/capability/reason 在状态不变时只报告一次，并在诊断面板保持当前降级状态。

---

## 10. 事件转换与批处理

### 10.1 转换原则

1. winit 类型不越过 adapter 层。
2. 保留可用信息，不把 unknown 静默映射为普通值。
3. 物理键与逻辑键并存；文本输入来自 IME，不由 KeyCode 猜测。
4. 单位必须显式：physical pixel、logical point、line scroll、pixel scroll。
5. 事件入口立即加 timestamp/sequence/window/device identity。
6. 平台合成事件必须标记 `synthetic`，使输入系统能统一处理焦点丢失。

### 10.2 不可合并事件

以下事件必须保持顺序：

- key/button press/release。
- text/IME preedit/commit。
- touch begin/end/cancel。
- close/destroy/focus/lifecycle。
- file drop。
- pointer enter/leave。
- monitor add/remove。

### 10.3 可合并事件

在同一 pump batch 内，且没有顺序屏障时：

- window resize 保留最后尺寸，并记录 coalesced count。
- window move 保留最后位置。
- cursor absolute move可保留最后位置；原始 mouse delta 不能简单只保留最后一个，应该累加并可选保留高精度样本。
- scale-factor change 是屏障，不能跨过它合并坐标事件。
- mouse wheel 可以按相同单位/phase 累加；不同 phase 不合并。

编辑器 resize 时可能需要中间事件做实时布局，但渲染 Surface 重配应只使用稳定最新尺寸，并经过 debounce/zero-size 策略。

### 10.4 队列过载策略

队列必须有界。优先级：

```text
绝不丢：lifecycle、destroy、close、key/button edge、touch edge、IME commit
可合并：resize、move、cursor absolute、wheel、raw delta
可丢且计数：重复 redraw、诊断采样、冗余 hover motion
```

接近容量时先合并，不允许覆盖不可丢事件。真正容量耗尽时：

- 设置 sticky overflow flag。
- 尽最大努力插入 `InputStateInvalidated`。
- 下一帧清空 pressed 状态并请求设备状态重同步。
- 输出一次高严重度诊断及丢弃计数。

### 10.5 每帧零分配

- `EventBatchBuilder` 使用预分配 `Vec`/ring buffers，跨帧复用容量。
- 文本、路径等可变长数据进入 frame arena，以小 handle 放在事件中。
- 常规移动/按键/resize 是定长 POD。
- batch 消费完成后统一 reset arena，不逐事件释放。
- 容量按观测高水位缓慢增长，有硬上限与诊断；不因单次恶意输入永久膨胀。

---

## 11. 输入、IME、指针与手势边界

### 11.1 键盘

每个键事件保留：

- physical key/code：适合游戏绑定，跟键盘布局无关。
- logical key：适合快捷键和可访问性。
- key location：左/右、数字区。
- repeat。
- synthetic。
- native scan code（可选诊断扩展，不作为跨平台逻辑基础）。
- modifiers snapshot。

焦点丢失时，后端/输入内核合成释放所有仍按下的键和按钮，防止 Alt-Tab 后持续移动。合成策略必须有测试且标记来源。

### 11.2 文本与 IME

IME 是状态机，不是字符事件：

```text
Disabled → Enabled
Enabled → Preedit(text, cursor/selection)
        → Commit(text)
        → PreeditChanged/Cancelled
        → Disabled
```

- UI 指定 IME allowed 与候选框逻辑区域。
- 后端转换到窗口 physical/logical 坐标并调用 winit。
- preedit 与 commit 使用 frame arena 字符串，保持 Unicode 原样。
- 不从 `KeyboardInput` 生成文本；死键、组合输入、中文/日文/韩文必须走 IME。
- 密码框的日志、录制和诊断必须遮蔽文本内容。

### 11.3 指针

区分：

- Window-relative absolute pointer position。
- Device raw relative motion。
- Cursor grab/confine/lock。
- Cursor visible/icon/custom cursor。
- Pointer pressure/tilt（若来源支持，进入扩展事件）。

游戏相机优先消费 raw delta；UI 消费 absolute pointer。平台不支持 raw motion 或 lock 时能力降级，不能把 absolute delta 冒充 raw input。

### 11.4 滚轮与触摸板

- LineDelta 与 PixelDelta 保留原始单位，后续由 UI/Input Profile 归一化。
- phase（start/move/end/momentum）若平台可用应保留。
- pinch、rotate、pan、double tap 等进入 `prism_input` 手势候选流，手势识别策略不在窗口后端中硬编码。
- 高分辨率触摸板不能量化到整数 notch。

### 11.5 触摸与笔

保留 touch/device ID、phase、position、pressure、force、tool type 和可用的 tilt。touch cancel 在 suspend、focus loss 和 Surface loss 时必须正确生成。多点触控的同批次顺序使用全局 sequence。

### 11.6 文件拖放

窗口后端输出 `HoveredFile/DropFile/HoverCancelled` 语义，路径视为不可信输入：

- 不自动读取文件。
- Web 使用受限 handle/blob token 而非伪本地路径。
- 编辑器决定导入，游戏决定是否接受。
- 路径字符串进入 arena，避免热路径事件膨胀。

---

## 12. DPI、坐标与多显示器

### 12.1 坐标空间

至少区分：

```text
Desktop Physical      虚拟桌面像素坐标，可为负
Window Outer Physical 含装饰窗口矩形
Client Physical       可渲染区域像素
Window Logical        DPI 缩放后的逻辑点
UI Logical            UI scale 后坐标
Render Extent         Surface 实际纹理尺寸
```

任何 API 名称和结构都必须带空间/单位，禁止裸 `Vec2 size`。

### 12.2 ScaleFactorChanged

正确流程：

1. 收到新 scale factor 和 OS 建议尺寸。
2. 建立一个带 sequence 的原子 DPI transition。
3. 更新 monitor/realized scale。
4. 规范化新的 client physical size。
5. 发布一个不可拆开的 `ScaleFactorChanged` 事件。
6. UI 重建 scale-dependent layout/font atlas request。
7. RHI 在非零尺寸时重配 Surface。
8. 随后坐标事件均使用新 scale。

不能先发 resize、下一帧再发 scale，造成一帧 UI/鼠标错位。

### 12.3 用户缩放与后端缩放

最终 UI scale 可为：

```text
ui_scale = backend_dpi_scale × accessibility_scale × user_scale
```

winit 后端只报告 backend DPI；可访问性和用户缩放属于 UI/项目设置。将三者混成一个字段会破坏窗口尺寸转换和跨设备一致性。

### 12.4 显示器身份

winit monitor handle 未必提供跨重启稳定 ID。因此 `MonitorId` 的语义应是**本进程热插拔 epoch 内稳定**，不要承诺跨开机稳定。可选 `MonitorFingerprint` 由名称、位置、尺寸、EDID-like 可用信息构成，仅用于布局恢复提示，冲突时需匹配评分和用户确认。

### 12.5 热插拔

winit 对所有平台的 monitor-added/removed 通知能力并不完全一致。策略：

- 在 resume、window move/scale change、fullscreen request 和周期性低频检查时刷新 snapshot。
- 对 snapshot 做稳定差分，产生 added/changed/removed。
- 目标显示器移除时退出 exclusive fullscreen，迁移到 primary/nearest monitor，并报告调整。
- 不在每帧完整枚举显示器。

### 12.6 视频模式选择

选择采用可解释评分：

1. 显式 MonitorId。
2. HDR/bit depth 硬约束。
3. 分辨率距离。
4. refresh rate 与目标帧率匹配。
5. 更高 refresh 作为平局项。

结果记录候选和评分，便于诊断“为什么没有进入 144 Hz”。实际 exclusive 支持受平台/winit 限制，失败应可靠退回 borderless，而非循环重试。

---

## 13. 光标与捕获

### 13.1 状态模型

```text
DesiredCursorState
  visible / icon / custom / grab / hit_test / position request
RealizedCursorState
  actual grab / visibility / last error / capability
```

光标 lock/confine 常受用户手势、焦点和平台安全策略限制。Web 必须在用户 gesture 中请求 pointer lock；后端用 `AcceptedPending` 表示等待浏览器确认。

### 13.2 自定义光标

- 上层提交已解码 RGBA、尺寸、hotspot 和 scale variants，不让后端依赖 asset loader。
- 后端在冷路径创建 winit custom cursor 并缓存内容哈希。
- 限制最大尺寸和内存，验证 hotspot。
- DPI/monitor 变化时选择合适 scale variant。
- 创建失败回退到标准 cursor，并保留诊断。

### 13.3 捕获丢失

focus loss、suspend、modal dialog 和 OS policy 都可能解除 grab。后端必须更新 Realized 并发 `CursorGrabChanged`；若 Desired 仍要求捕获，只能在合法时机有限重试。不能假定一次 `set_cursor_grab` 永久有效。

### 13.4 游戏与编辑器策略

- 游戏：点击 viewport 后 lock，Esc 释放，焦点恢复需重新确认。
- 编辑器：多个 viewport 独立拥有 capture token，只有一个为 active capture owner。
- capture ownership 由上层仲裁；后端只执行最终窗口命令，避免多个系统互相抢光标。

---

## 14. Surface 句柄与渲染握手

### 14.1 Raw handle 不是普通长期资源

raw window/display handle 只在 native window 存活时有效。设计 `NativeHandleLease`：

```rust
pub trait NativeWindowHandleProvider {
    fn acquire(
        &self,
        window: WindowId,
    ) -> Result<NativeHandleLease, HandleError>;
}
```

Lease 内部持有 `Arc<winit::window::Window>` 或等价生命周期锚点，实现 `HasWindowHandle + HasDisplayHandle`。公共 API 不暴露 registry 的可变引用。

### 14.2 Surface 状态机

```text
NoNativeWindow
  → HandleAvailable
  → CreatingSurface
  → SurfaceReady
  ↔ SurfaceOutdated/Resizing
  → SurfaceLost
  → Recreating
  → Retiring
  → NoSurface
```

window `0×0`（最小化或 Web 隐藏）时不 configure Surface，不等同于销毁；保留逻辑资源，在恢复非零尺寸后重配。

### 14.3 Resize storm

- Window realized size 每个事件都更新，保证输入/UI 坐标准确。
- RHI Surface configure 可在同一 batch 合并，只使用最后非零尺寸。
- 交互 resize 时渲染可选择拉伸最后一帧、低频重配或实时重配，由 Render Resize Policy 决定。
- 后端不自行 sleep/debounce 导致事件延迟；它提供 batch/coalescing hint。

### 14.4 HDR

HDR 是跨层能力：

```text
Monitor reports potential HDR
+ OS HDR state/platform extension
+ adapter/surface format support
+ renderer color pipeline
+ swapchain color space/metadata
= realized HDR presentation
```

winit 不能单独保证 HDR。`prism_window_winit` 报告显示器和平台可观测信息；RHI 决定格式/色彩空间并回报 `HdrPresentationState`。切换显示器或 OS HDR 时，Surface 可能重建，色调映射和 UI 白点同步更新。

### 14.5 VRR 与 Present Mode

同样由多层协作：

- 后端报告窗口焦点、fullscreen/monitor/refresh 信息。
- RHI 查询 tearing、FIFO/Mailbox/Immediate 和平台扩展。
- frame pacing 根据 realized present mode/VRR range 调整 deadline。
- UI 显示“Requested vs Realized”，避免设置页撒谎。

---

## 15. Frame Pacing、低延迟与功耗

### 15.1 Update Policy

```rust
pub enum LoopMode {
    Continuous,
    Reactive { max_wait: Duration },
    OnDemand,
    Suspended,
}

pub struct WindowLoopPolicy {
    pub focused: LoopMode,
    pub unfocused: LoopMode,
    pub occluded: LoopMode,
    pub minimized: LoopMode,
    pub battery_saver: Option<LoopMode>,
}
```

默认建议：

- 游戏前台：Continuous，但由 Frame Pacer 控制，不无界忙轮询。
- 游戏失焦：低频 Reactive，联网/音频按 App Policy 单独继续。
- 编辑器有交互：Continuous；静止面板：OnDemand/Reactive。
- 最小化/全部遮挡：停止 present，仿真按产品需求降频。
- 移动 suspend：Suspended。

### 15.2 `ControlFlow` 选择

- 有立即工作：`Poll` 或 request_redraw，但只在必要窗口。
- 有未来 deadline：`WaitUntil(deadline)`。
- 完全空闲：`Wait`。
- deadline 由 `prism_app`/frame pacer 提供，winit 后端只是执行者。
- 避免“每个 window request_redraw 导致一帧多次 App update”；全局 update 和 per-window render request 分开。

### 15.3 多窗口 redraw

```text
App Update N (once)
  → dirty window set {A, C}
  → render A and C as policy permits
  → clean B does not acquire/present
```

编辑器多个 3D viewport 可有不同刷新率和 visibility。每窗口维护 `RedrawReason` 位集：ContentChanged、Animation、Resize、Expose、SurfaceRecovered、Screenshot、ExternalRequest。重复原因合并。

### 15.4 低延迟标记

统一时间线：

```text
OS event arrival
Input batch sealed
Simulation start/end
Render extract
Render submit
Surface acquire/present
Present observed (if RHI/platform supports)
```

这些 timestamp 使用同一映射时钟或记录时钟变换，输出 p50/p95/p99：input-to-sim、input-to-submit、submit-to-present、端到端估算。无 present feedback 时明确标记“估算”，不能报告伪精确数字。

### 15.5 Reflex/低延迟模式

- winit backend 提供事件泵、唤醒和 deadline 执行。
- `prism_app` 将模拟尽量延后到预计提交前。
- RHI 控制 frame queue depth、waitable object、present timing/Reflex 扩展。
- Prism LowLatencyPolicy 协调三者，按 capability 降级。
- 不能靠 `ControlFlow::Poll` 代替低延迟技术；它只会增加 CPU 和功耗。

### 15.6 时间预算

窗口后端自身每帧目标：

- 无事件、无命令：接近零工作。
- 常规 60–240 Hz 游戏：事件转换/命令同步 p99 < 0.1 ms。
- 8 kHz raw mouse：批处理 p99 < 0.2 ms，不丢样本。
- resize storm：主线程转换 p99 < 0.3 ms，Surface configure 不计入后端预算。

目标需在实际平台校准，作为基准门槛而非绝对硬件承诺。

---

## 16. 多窗口与编辑器级能力

### 16.1 窗口角色

建议上层声明 role：

```text
GameMain
EditorMain
Viewport
Tool
Popup
Modal
Splash
Presentation
```

Role 不是平台窗口类型，但用于默认策略：退出、taskbar、always-on-top、resize、Surface、刷新率、无障碍 root 和保存布局。

### 16.2 退出策略

```rust
pub enum ExitPolicy {
    ExplicitOnly,
    OnPrimaryWindowClosed,
    OnLastWindowClosed,
    OnAllRequiredWindowsClosed,
}
```

后端只报告关闭/销毁；`prism_app` 根据策略决定退出。工具窗口关闭不能默认杀死整个编辑器。

### 16.3 布局持久化

保存逻辑窗口布局，而非生硬保存物理像素：

- monitor fingerprint + normalized work-area position。
- logical size。
- maximized/fullscreen intent。
- DPI 与保存时间的 monitor bounds。

恢复时先匹配显示器，再 clamp 到当前 work area，保证标题栏可见；显示器缺失时迁移到 primary。布局持久化属于编辑器/应用层，后端提供必要快照和恢复命令。

### 16.4 窗口数量治理

- 创建窗口是冷路径，可分帧处理。
- 设置软/硬窗口数量上限，避免插件失控。
- 不可见窗口不默认持续 acquire Surface。
- 长期隐藏工具窗口可根据策略退休 Surface，保留逻辑窗口。
- 每窗口诊断资源：事件率、present rate、Surface bytes、focus/capture 状态。

### 16.5 嵌入与外部宿主

未来可支持由外部宿主提供 native window：DCC 插件、编辑器嵌入、测试 harness。该模式不能强塞进普通 `CreateWindowRequest`；使用明确 `HostedWindowDescriptor`、所有权（Borrowed/Owned）与销毁协议。winit 对外部窗口嵌入能力有限，列为平台扩展并明确支持矩阵。

---

## 17. 无障碍设计

### 17.1 分层

```text
prism_ui_accessibility  语义节点、角色、状态、焦点、action
           ↓ TreeUpdate / ActionRequest
prism_window_winit::accessibility  AccessKit window adapter
           ↓
OS screen reader / switch control / automation
```

窗口后端只管理 per-window adapter 的激活、更新和事件转发。

### 17.2 多窗口

每个可访问窗口有独立 root 和 focus。浮动工具、模态窗口打开/关闭时更新 active tree 与 focus，不把所有 UI 节点塞到主窗口 root。

### 17.3 性能

- 语义树增量更新，不每帧重发完整树。
- 不活跃/未请求无障碍时不构建后端更新，但 UI 语义真相仍可存在。
- Action request 使用有界队列和 wake coalescing。
- 大型编辑器树按 dirty subtree 更新，基准覆盖 10 万节点但每帧少量变化。

### 17.4 正确性

窗口 title、focus、scale 和 bounds 变化要同步到语义 root。辅助技术 Action 进入 UI 命令流，不能直接在 winit callback 中修改 UI/ECS，避免可重入。

---

## 18. 平台能力矩阵与特殊策略

### 18.1 Windows

重点：

- foreground focus 请求可能被 OS 拒绝。
- per-monitor DPI 与跨屏 resize。
- exclusive/borderless、tearing、HDR、waitable swapchain 要与 DX12/Vulkan RHI 协同。
- raw mouse、IME、窗口拖拽和 resize modal loop 的事件节奏。
- content protection、drag/drop 等功能按 capability 报告。

需要真机测试多显示器、混合 100/150/200% DPI、HDR 开关、Alt-Tab、远程桌面和显示器断开。

### 18.2 macOS

重点：

- EventLoop 和窗口必须在主线程。
- Retina scale、Spaces fullscreen、窗口 tabbing/activation。
- Live resize、occlusion、Cmd 快捷键、IME 和 trackpad gesture。
- Metal Surface 生命周期和 App Nap/电源策略。
- exclusive fullscreen 语义与 Windows 不同，应诚实映射/降级。

### 18.3 Linux X11

重点：

- Window manager 行为差异大，position/focus/always-on-top 可能只是 hint。
- raw input、IME、剪贴板/拖放由不同协议组合。
- 独占全屏和刷新率选择能力不一致。
- CI 覆盖至少 GNOME/KDE 和 Xvfb 基础路径；性能/输入需真显示环境。

### 18.4 Linux Wayland

重点：

- 客户端通常不能控制绝对窗口位置。
- pointer lock/confine 依赖协议支持。
- fractional scaling 和不同 compositor 行为。
- 全局 raw input/窗口操作受安全模型限制。
- 不能把 X11 能力假装为 Wayland 可用；返回 Unsupported 并提供合理体验降级。

### 18.5 Web

重点：

- EventLoop 被浏览器宿主，不能阻塞。
- Canvas 创建/选择、CSS size 与 backing buffer physical size 分离。
- ResizeObserver、devicePixelRatio 变化和页面 visibility。
- pointer lock/fullscreen/clipboard 等需要 user gesture。
- file drop 返回 browser file handle/blob。
- 页面隐藏时 requestAnimationFrame 停止或大幅降频。
- WebGPU Surface/Canvas 重配由 renderer 处理。

提供 `CanvasBinding`：创建 canvas、绑定指定 DOM selector、或使用外部 canvas。DOM 对象不能进入平台无关内核。

### 18.6 iOS/Android

重点：

- 单/少窗口、Surface 可反复丢失和恢复。
- safe area、orientation、virtual keyboard、touch/gesture。
- app suspend/resume、memory warning、thermal state。
- 后台运行权限受限。
- winit 未完整覆盖的能力通过平台 extension 注入，核心语义仍走统一生命周期事件。

### 18.7 支持级别

每项能力标注：

```text
Tier 1  CI + 真机持续验证，发布支持
Tier 2  CI/定期真机验证，实验支持
Tier 3  可编译或社区维护，不作发布保证
Unsupported 明确不支持
```

文档和运行时查询使用同一 capability 数据源，避免宣传与实际行为不一致。

---

## 19. 错误、诊断与可观测性

### 19.1 错误分类

```rust
pub enum WindowBackendError {
    EventLoopCreation { source: ErrorId },
    WindowCreation { request: WindowRequestId, source: ErrorId },
    InvalidAttributes { field: AttributeId, reason: ValidationError },
    OperationDenied { capability: Capability, reason: PlatformDenial },
    Unsupported { capability: Capability },
    HandleUnavailable { window: WindowId, lifecycle: NativeWindowLifecycle },
    RegistryInvariant { code: InvariantCode },
    QueueOverflow { queue: QueueKind, dropped: u64 },
    Platform { code: PlatformErrorCode },
}
```

错误不直接保存不可序列化的巨大第三方对象；保留稳定错误码、摘要和可选 source chain。

### 19.2 日志降噪

- Unsupported/Denied 首次报告 Warning，状态不变不重复。
- resize/move/focus 等高频事件默认只做 trace counter，不逐条 info。
- 创建、销毁、Surface lifecycle 和 monitor topology change 记录结构化 info。
- 队列 overflow、ID invariant、迟到事件误路由风险为 error。
- debug 模式可对指定 WindowId 开启事件采样。

### 19.3 指标

全局：

- event loop wakeups/sec、empty wake ratio。
- update/redraw 次数和合并率。
- Poll/Wait/WaitUntil 时间占比。
- command/event queue depth/high-water/overflow。
- translator CPU time。
- focused/unfocused/occluded/minimized 时长。

每窗口：

- WindowEvent/InputEvent rate。
- resize/move coalesced count。
- redraw requested/rendered/presented count。
- native lifecycle、Surface state、DPI、monitor。
- requested vs realized attributes。
- cursor/IME/focus/capture 状态。

### 19.4 调试面板

提供 Window Backend Inspector：

- ID 映射和 generation。
- Desired/Applied/Realized 三态 diff。
- 当前 capability/降级原因。
- 最近 N 条规范事件和命令结果。
- Surface lease、生命周期和 pending destroy。
- monitor topology 与视频模式选择评分。
- frame pacing/wakeup 时间线。

该工具对跨平台“只在用户机器出现”的窗口问题非常关键。

### 19.5 事件录制

可选录制规范化后的 Prism events，而非直接序列化 winit enum：

- 带 protocol version、platform metadata、clock info。
- 可选择遮蔽文本、路径和窗口标题。
- 支持实时环形 flight recorder，只在崩溃/错误时保存最近数秒。
- 重放驱动 fake backend，不需要真实 OS 窗口。
- 原始高频运动可按测试目的选择完整样本或聚合轨迹。

---

## 20. 性能与内存设计

### 20.1 热路径

热路径包括 `window_event/device_event` 转换、队列追加、批次提交和 redraw 决策。原则：

- 不做哈希字符串解析。
- WindowId 映射 O(1)。
- 定长事件内联。
- Vec/ring capacity 跨帧复用。
- 不逐事件锁 Mutex。
- 不逐事件触发 App update 或 wake。
- 不在 callback 中创建 GPU 资源、读取资产或执行复杂 UI。

### 20.2 冷路径

允许分配：窗口创建、title/custom cursor、monitor enumeration、AccessKit tree setup、录制启动。冷路径也必须有限制，尤其 monitor video modes 和拖放路径不能无界信任平台输入。

### 20.3 Registry

窗口数量通常很少，优先清晰与正确；可使用双 HashMap。若确定性诊断输出需要稳定顺序，快照时按 Prism WindowId 排序，不应为此让热查找退化。

### 20.4 批量系统调用

winit API 多为逐窗口调用，无法真正 syscall batching；后端通过 command coalescing 减少调用：

- 相同属性只执行最后值。
- 与 Applied 相同不调用。
- Destroy 覆盖未执行属性。
- resize constraints 一组规范化后执行。
- redraw 使用 dirty bit，重复请求合并。

### 20.5 高轮询率输入

8 kHz 鼠标每帧可产生数十到上百样本：

- raw delta 使用专用小型 ring/segment，不与字符串事件共用昂贵容器。
- gameplay 可请求 summed delta；竞技/研究模式可请求 sample sequence。
- 保持 button edge 与 delta 的相对 sequence。
- 输入录制可采用 delta encoding 和块压缩，不影响 live path。

### 20.6 空闲功耗

工具和启动器的质量不仅是峰值 FPS，也包括空闲功耗：

- 无 dirty window 时不 redraw。
- animation timer 统一成下个 deadline，不开大量独立 wake timer。
- 后台/遮挡状态降低 tick 和 present。
- wake reason 指标识别“谁阻止休眠”。
- CI 设空闲 60 秒 wakeup 数和 CPU 使用回归测试。

---

## 21. 易用性设计

### 21.1 最小启动

普通游戏只需：

```rust
App::new()
    .add_plugins(PrismDefaultPlugins)
    .run();
```

默认插件根据平台创建一个主窗口，安装 `WinitRunner`，接通输入和 Surface 请求。高级用户可以在创建前提供 `WindowAttributes` 和 `WindowLoopPolicy`。

### 21.2 默认策略

- 首窗居中、合理逻辑尺寸、可调整大小。
- AutoVsync 是用户意图，RHI 选择真实支持模式。
- 失焦降频但不擅自暂停联网 gameplay。
- DPI 自动、UI 用户缩放独立。
- CloseRequested 走 App Exit Policy。
- 透明、独占全屏、content protection 等非通用能力默认关闭。

### 21.3 不暴露 winit 类型

常规用户只使用 `prism_window`/`prism_input` 类型。确需平台集成时，提供显式 escape hatch：

```rust
with_winit_window(window_id, |window: &winit::window::Window| { ... })
```

该 API：

- 只在窗口线程合法阶段调用。
- 不允许闭包或引用逃逸。
- 标为高级、平台相关、不保证可替换后端。
- 通过 command/user event 排队，而不是任意线程同步调用。

### 21.4 Capability-first API

用户先查询能力或提交可降级请求：

```rust
WindowModeRequest {
    preferred: Exclusive { mode },
    fallback: Borderless,
}
```

结果返回 realized state。相比让用户写平台 `cfg`，这更容易扩展和维护。

### 21.5 快速故障定位

`prism asset doctor` 对应内容系统；窗口侧提供：

```text
prism window doctor
  OS/session/display server
  winit backend
  monitors/video modes/DPI
  raw input/pointer lock
  surface handle availability
  HDR/VRR handoff summary
  accessibility adapter
  event loop wake diagnostics
```

CLI/诊断输出可复制为脱敏报告，用于 QA 和用户支持。

---

## 22. 易扩展与易维护

### 22.1 后端可替换

公共协议必须能由以下实现：

```text
prism_window_winit    桌面/Web/部分移动
prism_window_sdl      可选替代
prism_window_native_* 主机 SDK/特殊平台
prism_window_headless 测试/服务器
prism_window_replay   事件重放
```

如果新增 SDL 后端必须修改 gameplay 或 `prism_window` 大量类型，说明 ABI 仍泄漏 winit 语义。

### 22.2 平台扩展

使用 namespaced extension：

```rust
WindowExtensionCommand::Windows(...)
WindowExtensionCommand::MacOS(...)
WindowExtensionCommand::Web(...)
```

只有确实无法抽象且有产品价值的能力才进入扩展。先判断能否提升为通用 capability；扩展不得改变 WindowId、事件顺序和销毁协议。

### 22.3 转换器纯函数化

键码、按钮、theme、resize direction 等转换尽可能写成纯函数，输入第三方 enum、输出 Prism enum，并有穷举测试。winit 升级时 compiler 能提示新增 variant；unknown/unsupported 走明确 fallback，不使用宽泛 `_ => default` 吞掉信息。

### 22.4 winit 升级流程

1. 单独分支升级依赖。
2. 编译所有平台 target。
3. 审阅 upstream changelog 的事件/生命周期语义变化。
4. 更新 converters exhaustive mapping。
5. 跑规范事件 Golden、UI/IME/DPI 测试。
6. Tier 1 平台真机 smoke。
7. 比较 event rate、wakeups、CPU、延迟基线。
8. 更新兼容矩阵与已知降级。

不得只因“能编译”就认为后端升级完成。

### 22.5 版本边界

- `prism_window_winit` 与 winit 版本紧密绑定，可随 workspace 内部演进。
- Prism backend ABI、recording schema 和 renderer handle handshake 独立版本化。
- winit 原始 enum 不进入存档/网络/插件稳定协议。
- 平台 extension 标注实验/稳定级别。

### 22.6 文档唯一真相

- `prism_window` 文档定义平台无关窗口语义。
- 本文定义 winit 实现与集成。
- `prism_window_refactor_zh.md` 定义窗口整体架构与确定性时间轴；`prism_input_refactor_zh.md` 定义输入整体架构。
- 渲染文档定义 Surface、HDR、VRR 和 Present。
- 重叠部分只保留职责摘要和链接，详细规则由单一文档拥有，避免长期漂移。

---

## 23. 测试与验证

### 23.1 单元测试

- 所有 winit→Prism enum converters。
- physical/logical/DPI 转换边界。
- desired/applied/realized diff 和 command coalescing。
- fullscreen video mode 评分。
- redraw/loop policy 决策。
- close/exit policy。
- WindowId generation 与 tombstone。
- queue overflow 和不可丢事件策略。

### 23.2 属性测试

- 任意合法 command 序列不会违反 lifecycle invariant。
- min/max/size 规范化总产生合法请求。
- 相同输入 snapshot/diff 产生相同命令。
- 事件合并后最终 Window realized state 与未合并流一致。
- ID 双向映射增删保持互逆。

### 23.3 模型测试

用简化 OS model 穷举：

- create/destroy 与 Surface lease 交错。
- close request 被取消/接受。
- suspend/resume 与创建请求交错。
- focus loss 与 pressed keys。
- command pending 时 OS 主动 resize。
- monitor remove 与 fullscreen。

若引入跨线程队列，对 wake coalescing、取消和 shutdown 使用 loom/等价并发模型测试。

### 23.4 Fake 后端

抽象窄 `NativeWindowOps` 供测试：

- 可配置成功、调整、拒绝、延迟确认。
- 模拟 resize storm、DPI change、迟到事件。
- 模拟 Surface lease 长时间不归还。
- 模拟事件循环 suspend/resume。

不要试图在普通单测中构造完整真实 `ActiveEventLoop`。

### 23.5 事件 Golden

固定场景：

- 启动→创建→resize→focus→close。
- 跨 100%/200% DPI 显示器移动。
- IME preedit/commit/cancel。
- Alt-Tab 与按键释放。
- pointer lock 获得/丢失。
- 多点触摸取消。
- Web visibility/suspend/resume。

记录规范化 Prism event stream，双跑比较 sequence、状态快照和合并结果。平台 Golden 可不同，但同平台同输入必须稳定。

### 23.6 真机矩阵

Tier 1 建议：

- Windows 11：单屏、多屏混合 DPI、HDR、VRR。
- macOS 当前支持版本：Retina、外接屏、Spaces fullscreen、trackpad/IME。
- Ubuntu/Fedora：Wayland GNOME、KDE；X11 至少一个主流桌面。
- Chrome/Edge/Safari：Canvas resize、DPI、pointer lock、IME、visibility。

移动端若列 Tier 1，再增加真实 iOS/Android 生命周期和键盘/旋转矩阵。虚拟机只能覆盖基础路径，不能替代 HDR/VRR/raw input/真实 DPI 验证。

### 23.7 性能基准

- 1M converter calls。
- 8 kHz mouse 60 秒，无丢样和稳定内存。
- 每秒 1000 resize/move 事件的合并率和 CPU。
- 64 窗口 command diff（压力测试，不是普通产品目标）。
- 空闲 60 秒 wakeup/CPU。
- Reactive 模式后台任务唤醒延迟。
- 录制开/关额外成本。

### 23.8 故障注入

- native window create 失败。
- focus/cursor/fullscreen 被拒绝。
- monitor 在 fullscreen 切换时拔出。
- Surface create/resize/lost。
- event queue overflow。
- worker 发命令后退出。
- App panic/强制退出。
- resume 后 raw handle 变化。

验收标准：无悬垂句柄、无死锁、无 feedback syscall storm，保留结构化诊断并尽可能继续运行。

---

## 24. 不变量

以下作为代码审查和测试硬约束：

1. winit Window 只由窗口主线程所有和操作。
2. winit 类型不进入 `prism_window`、gameplay 或稳定录制协议。
3. 每个活跃 winit ID 最多映射一个 `(Prism WindowId, generation)`。
4. Destroyed 后的迟到事件不能作用于新窗口。
5. native window 的销毁晚于所有 Surface/handle lease 释放。
6. `CloseRequested` 不自动等于 `Destroyed`。
7. Desired 不等于 Realized；所有平台调整可观察。
8. 输入 edge、IME commit 和 lifecycle 事件不可被普通合并策略丢弃。
9. scale change 与关联尺寸更新对消费者原子可见。
10. 文本输入只来自文本/IME 通道，不由物理键猜测。
11. 后端空闲时不产生无理由持续 redraw/wakeup。
12. suspend/resume 可以重复发生，且逻辑 WindowId 不因 Surface 重建而必然变化。
13. 同一 batch 内事件有全局稳定 sequence。
14. callback 中不执行资产加载、GPU 重建或复杂业务。
15. 后端失败不 panic，除非检测到内部不变量破坏且继续运行会导致内存/句柄不安全。

---

## 25. API 草案

### 25.1 Plugin 与 Runner

```rust
pub struct WinitPlugin {
    pub runner: WinitRunnerSettings,
    pub windows: WindowBackendSettings,
    pub accessibility: AccessibilityBackendSettings,
}

pub struct WinitRunnerSettings {
    pub loop_policy: WindowLoopPolicy,
    pub event_capacity: EventCapacity,
    pub startup_policy: StartupPolicy,
    pub threading: WinitThreadingMode,
}

pub enum WinitThreadingMode {
    IntegratedMainThread,
    #[cfg(feature = "pipelined-runner")]
    Pipelined,
}
```

Plugin：

- 注册 `WindowBackend` capability。
- 安装窗口 command sink/event source。
- 安装 `WinitRunner`。
- 验证没有第二个窗口 Runner/后端争夺 App。
- 不在 `build()` 阶段提前创建必须依赖 `ActiveEventLoop` 的原生窗口。

### 25.2 后端服务

```rust
pub trait WindowBackend: Send + Sync {
    fn capabilities(&self) -> WindowBackendCapabilities;
    fn submit(&self, command: WindowCommandEnvelope) -> Result<(), SubmitError>;
    fn wake(&self, reason: WakeReason) -> Result<(), WakeError>;
}
```

此 handle 是线程安全的命令提交端，不暴露 native Window。真实命令由主线程 drain。

### 25.3 创建请求

```rust
pub struct CreateWindowRequest {
    pub request_id: WindowRequestId,
    pub proposed_id: WindowId,
    pub role: WindowRole,
    pub attributes: WindowAttributes,
    pub parent: Option<WindowId>,
    pub surface: SurfaceIntent,
    pub fallback: WindowFallbackPolicy,
}
```

`SurfaceIntent::Required/Optional/None` 允许游戏窗口、工具窗口和纯原生对话框拥有不同 ready 条件。

### 25.4 生命周期事件

```rust
pub enum NativeWindowEvent {
    Created {
        window: WindowId,
        capabilities: WindowCapabilities,
        realized: RealizedWindowState,
    },
    Ready { window: WindowId },
    CommandResult(WindowCommandResult),
    SurfaceHandleChanged { window: WindowId, epoch: HandleEpoch },
    CloseRequested { window: WindowId },
    Destroyed { window: WindowId },
}
```

`HandleEpoch` 变化通知 RHI 旧 handle 不再用于新 Surface 创建；旧 Lease 生命周期仍保证已在途操作安全。

### 25.5 监视器查询

```rust
pub trait MonitorQuery {
    fn snapshot(&self) -> Arc<MonitorSnapshot>;
    fn refresh(&self) -> RefreshRequestId;
}
```

查询返回不可变 snapshot，避免读者持锁遍历 native handles。刷新由窗口线程执行并通过事件提交新 revision。

---

## 26. 实施路线

### M0：最小桌面闭环

- 新建 `prism_window_winit` crate。
- `ApplicationHandler` + `WinitRunner`。
- 单窗口创建/销毁、resize、focus、scale、close。
- WindowId registry。
- raw handle lease + wgpu/现有 RHI Surface smoke test。
- 键盘、鼠标按钮/移动/滚轮基础转换。
- Windows/macOS 当前开发机真机测试。

**验收**：打开窗口、稳定渲染、resize/最小化/恢复、关闭无崩溃；`prism_app` 不再把 WinitRunner 标记为缺失。

### M1：命令/事件 ABI 与多窗口

- 完整 Command Envelope/Result。
- Desired/Applied/Realized 三态。
- 多窗口、monitor snapshot、DPI 原子 transition。
- command coalescing、batch event、零分配热路径。
- Fake backend、record/replay Golden。

**验收**：编辑器主窗口 + 两个 viewport；窗口独立关闭/resize/Surface 生命周期正确。

### M2：完整输入与桌面体验

- raw mouse、IME、触摸、手势、拖放、自定义光标。
- focus loss 合成释放、capture ownership。
- AccessKit 多窗口桥。
- X11/Wayland 能力差异与诊断。
- Window Backend Inspector。

### M3：AAA 呈现协作

- RHI realized present/HDR/VRR handshake。
- frame pacing deadline、per-window redraw、occlusion/minimize 策略。
- 低延迟时间线和诊断。
- resize storm、Surface lost/recovery 完整联测。
- 混合 DPI/HDR/VRR 真机矩阵。

### M4：Web 与移动生命周期

- CanvasBinding、ResizeObserver、pointer lock/fullscreen gesture。
- Web visibility/IME/drop。
- iOS/Android resume/suspend、orientation、safe area、virtual keyboard。
- Surface handle epoch/recreation。

平台达到 Tier 1 前不得只凭可编译宣称生产支持。

### M5：规模化与替代后端验证

- Pipelined runner（仅在实测有收益后）。
- 外部宿主窗口实验能力。
- 用 `prism_window_headless/replay` 完整实现同一 ABI。
- 选择一个最小 SDL/fake adapter 验证 winit 语义未泄漏。
- 空闲功耗、8 kHz 输入和 64 窗口压力门禁。

### 路线纪律

- 每个里程碑都必须有真实窗口 + Surface + 输入的纵向闭环。
- M0/M1 不为未来主机 SDK过度抽象；只冻结已经被两个实现或 replay 测试证明必要的接口。
- HDR/VRR 不在没有 RHI 实际回报前标记完成。
- 移动/Web 不以桌面事件循环假设强行适配。

---

## 27. 性能与质量验收表

| 领域 | 指标 | 初始门槛 |
|---|---|---:|
| 空闲 | OnDemand 无变化窗口 redraw | 0 次/秒（除平台必要事件） |
| 空闲 | desktop app CPU | 接近宿主事件等待基线 |
| 事件 | 常规 batch 转换 p99 | < 0.1 ms |
| 输入 | 8 kHz raw mouse | 60 秒零 edge 丢失、零 overflow |
| 输入 | raw batch 处理 p99 | < 0.2 ms/帧 |
| resize | 1000 events/s | realized 正确，Surface 重配显著合并 |
| 命令 | 未变属性 | 0 次 winit 调用 |
| 唤醒 | 重复 proxy wake | 单 pending 周期合并为 1 次 |
| 内存 | 稳态 30 分钟 | event arena/queue 无持续增长 |
| 多窗口 | 创建/销毁 1000 轮 | 无 ID 错配、handle/Surface 泄漏 |
| DPI | 混合 DPI 跨屏 | UI、指针、Surface 同 transition 一致 |
| 生命周期 | suspend/resume 1000 轮模型测试 | 无非法状态、旧 handle 使用 |
| 延迟 | event arrival→batch sealed | p99 < 1 ms（不含 OS 调度） |
| 功耗 | 全部窗口遮挡/最小化 | present 停止，tick 符合策略 |
| 可访问性 | 小增量更新 | 不全量重发语义树 |
| 正确性 | Tier 1 event Golden | 100% 通过 |

门槛按硬件和平台基线建立方差带；不能用虚拟机成绩替代低延迟、HDR、VRR、混合 DPI 和真实输入设备结果。

---

## 28. 风险与取舍

| 风险 | 后果 | 应对 |
|---|---|---|
| winit 类型泄漏 | 无法替换后端，测试困难 | 显式 Command/Event ABI，escape hatch 隔离 |
| 主线程约束被忽略 | macOS/mobile 崩溃或未定义行为 | Window registry 和 ops 单线程所有 |
| ECS change detection 直接驱动 OS | 反馈循环、重复 syscall | Desired/Applied/Realized 三态与 revision |
| 事件全部逐条保留 | resize/move 洪水、卡顿 | 语义感知批处理和屏障 |
| 合并过度 | 丢按键/IME/高频输入 | 不可丢分类、raw 专用 ring、overflow 恢复 |
| 把 PresentMode 当 winit 能力 | 设置显示成功但实际无效 | RHI realized presentation handshake |
| native window 先于 Surface 销毁 | 悬垂 raw handle/GPU 错误 | Handle lease + retire state machine |
| 每帧 Poll | 高 CPU/功耗且不等于低延迟 | deadline pacing、WaitUntil、RHI 协同 |
| 平台 `cfg` 扩散 | 维护困难、行为漂移 | platform module + capability-first API |
| MonitorId 被误认为跨重启稳定 | 布局恢复到错误屏幕 | 进程稳定 ID + fingerprint 匹配 |
| 第三方事件日志泄露文本/路径 | 隐私风险 | recording 脱敏与敏感通道策略 |
| 只测能开窗口 | DPI/IME/暂停/多屏问题进入发布 | Tier 1 真机矩阵 + Golden + 故障注入 |
| 后端承担 UI/渲染职责 | 巨型耦合模块 | 维护清晰边界与共享握手协议 |

---

## 29. 最终决策摘要

`prism_window_winit` 应被定义为：

> **Prism 的 winit 窗口后端和窗口化 App Runner：拥有原生事件循环与 Window 对象，执行窗口命令，规范化平台事件，提供安全 Surface 句柄，并与输入、生命周期、呈现和无障碍系统通过显式协议协作。**

最终架构：

```text
prism_window
  平台无关窗口领域模型与状态

prism_window_winit
  winit 后端、事件循环、转换、主线程窗口所有权

prism_app
  App 生命周期、更新调度、Runner 契约、退出策略

prism_input
  原始设备与文本/触摸输入状态

prism_render / RHI
  Surface、swapchain、HDR、VRR、Present 和 GPU 生命周期

prism_ui_accessibility
  语义树；winit/accesskit 仅作宿主桥
```

必须坚持的十条原则：

1. `prism_window` 保持纯内核，winit 永不进入其公共领域模型。
2. 后端采用显式“命令出、事件入”契约，不以隐式 ECS 同步作为唯一协议。
3. 原生窗口和 EventLoop 由窗口主线程唯一拥有。
4. Desired、Applied、Realized 分离，平台拒绝和降级完全可观察。
5. Window、Surface 与 raw handle 通过租约和生命周期状态机安全销毁。
6. 输入边沿、IME 与生命周期不丢；高频运动批量化且稳态零分配。
7. DPI、坐标和多显示器使用显式单位和原子 transition。
8. HDR、VRR、Present 与低延迟是跨后端/RHI/App 的协作能力，不由 winit 独占。
9. 连续游戏、反应式工具、后台节流和移动暂停由统一 Loop Policy 驱动。
10. 真机矩阵、事件 Golden、故障注入、性能与空闲功耗共同构成发布门禁。

优秀的窗口后端不应让开发者频繁意识到它的存在：它在普通路径上默认正确、低成本，在复杂多窗口和跨平台路径上行为可解释，在平台能力不足时诚实降级，并为渲染、输入和 UI 提供稳定、可维护、可替换的地基。
