# Prism Camera 顶级次世代 AAA 级相机 / 取景 / 导演系统设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **相机（Camera）+ 投影（Projection）+ 视图矩阵与时域抖动（View/Jitter/Motion）+ 物理镜头与曝光（Lens/Exposure）+ 导演与虚拟相机（Director / Virtual Camera / Blend）+ 跟随/环绕/弹簧臂/避障（Rig）+ 相机级后处理栈 + 多视图/分屏/画中画/离屏 + 立体/XR + 大世界相机相对渲染 + 射线投射/拾取 + 确定性回放 + GPU 视图常驻** 相机内核设计。它是 `bevy_camera` + `bevy_camera_controller` 的自研替代，是「观察者意图」接入渲染/剔除/时域/后处理管线的唯一一等模型。
> 借形态不抄码。借鉴：
> - **物理相机与镜头**：Unreal `UCineCameraComponent`（Filmback/Lens/Focus/ISO/Shutter/Aperture）、Frostbite / Filament 物理曝光（EV100、Sunny-16、Saturation-Based Speed）、真实镜头的焦距/光圈/传感器几何
> - **导演与虚拟相机**：Unity **Cinemachine**（Virtual Camera + Brain + 优先级 + 状态驱动混合 + Body/Aim 组合器 + 噪声/冲击 Impulse）、Unreal **Camera Rig / Sequencer / Camera Modifier**、Godot **PhantomCamera** 插件形态
> - **弹簧臂与避障**：Unreal `USpringArmComponent`（碰撞拉近 / 探针半径 / 滞后 lag）、第三人称相机的遮挡消隐
> - **投影与精度**：Reversed-Z + 无限远平面（Reverse Infinite Perspective）、斜裁剪面（oblique near-plane，水面反射）、`glam`/`prism_math` 的 `Mat4`/`Affine3`
> - **时域与运动**：TAA/DLSS/FSR 的子像素抖动（Halton/Sobol jitter）、运动矢量所需的「上一帧视图投影」双缓冲、相机相对渲染（Camera-Relative Rendering）消除大世界抖动（Star Citizen / UE5 LWC）
> - **多视图家族**：对接本仓 `prism_render_architecture::view_family`（Main/StereoEye/Shadow/Reflection/SceneCapture/Editor/OfflineTile）
> 本文为纯经典线性代数 / 几何光学 / 控制插值路线，**不含任何 AI/ML 内容**，不含任何 Unreal/Unity/CryEngine 源码或衍生代码。

- 版本： v1.5（核心运行时与 RHI GPU 实现核验版，2026-10-15；`prism_camera`、`prism_camera_gpu`、`prism_camera_rig` 与 `prism_camera_ecs` 已纳入根 workspace，并完成默认/全特性测试、裁剪配置检查、ECS 发布链测试、RHI View buffer/WGSL consumer 真实设备测试和 GPU 数学 parity 记录；承接 `prism_math`/`prism_render_architecture` 的现有契约）
- 适用引擎： Prism（后 Bevy 时代，独立运行时）
- 关键依赖： `prism_math`（Vec3/Quat/Affine3/Mat4/Dir3、SIMD、投影构造）、`prism_transform`（相机位姿的空间真相 Local/Global + look_at 约束 + 大世界 64-bit/原点重定位）、`prism_ecs`（相机/投影/后处理为组件、变更检测、并行查询）、`prism_render_architecture::view_family`（ViewKind/ViewImportance/ViewRegistry/ViewHandle 视图注册与重要度聚合）；可选 `prism_window`（RenderTarget/表面尺寸/缩放因子）、`prism_input`（指针→射线、拾取）、`prism_time`（固定步 alpha / 曝光自适应时间常数）、`prism_tasks`（多视图并行提取）、`prism_diagnostic`（计数器/trace）、`prism_asset`（LUT/镜头畸变贴图/相机动画片段）
- 层级定位： 渲染前端 L4「取景与导演」；上接渲染提取（视图 UBO）/剔除（frustum）/时域放大（jitter + 历史）/后处理（相机级 stack），下接 `prism_transform`（位姿）/`prism_math`（代数）/`view_family`（视图槽）
- 明确约束：核心 `no_std + alloc`；`prism_camera` 的 feature 为 `std`、`physical_lens`、`post`、`stereo`、`determinism`、`cutscene`、`autofocus`、`shutter`、`view_batch`、`lookahead_stream`、`edge_stabilize`、`virtual_production`、`view_lod`、`composition_aid`、`occlusion_fade`；`prism_camera_rig` 的 feature 为 `director`、`rig`、`determinism`、`trace`；相机数据契约保持稳定，关闭某项配置时由调用方选择对应降级路径；工作区 `forbid(unsafe_code)`；**不依赖任何 `bevy_*` crate**；每个公共 API 都具备明确语义、容量和错误返回。
- 架构立场： **相机是「观察意图」的单一真相，视图（View）是其每帧的「观察结果快照」**。相机组件只持有可编辑意图（位姿经 transform、投影参数、镜头/曝光、后处理意图），每帧由提取阶段计算出不可变的 `ViewData`（矩阵 + jitter + 上一帧矩阵 + frustum + 曝光标量），注册进 `view_family::ViewRegistry` 供渲染/剔除/时域共享。导演（Director）不是另立真相，而是**选择与混合**多个「虚拟相机意图」后写回唯一的输出相机位姿与参数——与 Cinemachine 的 Brain 同形。

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / quality tier）
4. 分层架构与 crate 布局
5. 核心模型：Camera / Projection / ViewData / Lens / PostStack
6. 投影系统（透视 / 正交 / 物理 / Reversed-Z / 无限远 / 斜裁剪 / 自定义）
7. 视图矩阵与时域（view/proj/view-proj + jitter + 上一帧 + frustum 提取）
8. 大世界：相机相对渲染（Camera-Relative）与原点重定位协作
9. 物理镜头与曝光（Filmback/焦距/光圈/快门/ISO/EV100/自动曝光/景深）
10. 导演系统：虚拟相机 / Brain / 优先级 / 状态驱动混合
11. 相机 Rig：跟随 / 瞄准 / 环绕 / 弹簧臂 / 避障 / 滞后
12. 相机抖动与冲击（procedural noise / Impulse 6DOF）
13. 相机级后处理栈（曝光/泛光/景深/运动模糊/色调/暗角/色彩分级）
14. 多视图：分屏 / 画中画 / 小地图 / 离屏渲染（RenderTarget）/ 视图叠放
15. 立体 / XR / 注视点（per-eye / late-latch / foveated）
16. 剔除与可见性集成（frustum → prism_render_visibility）
17. 输入与拾取（viewport↔world 射线 / picking / 对接 prism_input）
18. 控制器（free / fly / pan / orbit，取代 bevy_camera_controller）
19. 确定性与回放对接（复用 prism_network，相机只做确定性纯函数）
20. GPU 视图常驻（View UBO / 多视图数组 / GPU-driven）
21. 与 Transform / App / Time / Window / 渲染 / 输入集成
22. 可观测性与诊断
23. 性能工程
24. 易用性与 API 人体工学（prelude / 一行建相机 / Bevy 迁移）
25. 契约、不变量与版本化
26. 交付矩阵（M0–M6）与基准即规格
27. 责任闭合、风险处置与验收状态
28. AAA 高级功能完整契约

---

## 1. 设计哲学与目标

相机是玩家「看见世界的那只眼睛」，也是整条渲染管线的**几何与时域基准**：剔除吃它的 frustum，LOD 吃它的距离与屏幕覆盖，TAA/DLSS 吃它的 jitter 与上一帧矩阵，运动模糊吃它的速度，后处理吃它的曝光，拾取吃它的逆投影。一个相机系统写得好不好，直接决定「画面稳不稳、糊不糊、抖不抖、好不好用」。

Bevy 的 `Camera` 把太多关注点挤在一个巨组件里（viewport / target / 投影 / HDR / 顺序 / 清屏），控制器又散在示例级 crate，物理曝光、导演混合、弹簧臂避障、相机相对渲染这些 AAA 必需能力几乎缺席。`prism_camera` 的使命是：**把「观察意图」与「每帧观察结果」显式分离**，让意图可被导演系统自由选择/混合，让结果（矩阵 + jitter + 历史 + 曝光）成为全管线共享的不可变视图契约。

**一句话定位**：`prism_camera` 是 Prism 的「取景与导演内核」——相机组件持有可编辑的观察意图，`ViewData` 是每帧派生的只读观察快照并注册进 `view_family`；导演只做「选 + 混」，物理镜头提供电影级真实曝光与景深，相机相对渲染守住大世界精度，jitter/历史矩阵让时域放大器开箱即用。

四条总目标（按权重）：

1. **性能**：视图派生成本 ∝ 活跃相机数，而非场景规模；矩阵走 SIMD；多视图提取并行；视图数据可直供 GPU 做 GPU-driven 剔除；导演/rig 仅在活跃时付费。
2. **效果（能力）**：物理相机曝光与景深（电影级）；Reversed-Z + 无限远（远近都不抖）；相机相对渲染（大世界无浮点抖动）；jitter + 上一帧矩阵（TAA/DLSS/FSR/运动模糊开箱）；导演混合与弹簧臂避障（第三人称/过场电影级手感）。
3. **易用**：`prism_camera_ecs::CameraBundle::default()` 一行建立标准相机实体；控制器和导演使用 `prism_camera_rig` 的纯数据 API；物理镜头 `Lens::cine_35mm().with_f_stop(1.8)` 直给电影参数。
4. **可移植 + 档位化**：核心 `no_std`；物理镜头 / 后处理 / 立体 / 确定性和高级纯数据契约按 feature 裁剪，Rig 的 director/rig 能力由配套 crate 裁剪；移动端可退化为「单相机 + 简投影 + 无 jitter」最小集。

**责任归属（范围已经闭合）**：`prism_camera` 负责意图模型、投影/视图派生、历史矩阵、jitter、frustum、曝光输入、后处理排序契约、视图注册和拾取数学；`prism_render_architecture` 负责 ViewRegistry 消费、渲染图、GPU 资源与后处理算法；`prism_transform` 负责层级传播和空间真相；`prism_physics` 负责场景求交；`prism_window` 负责表面与尺寸；编辑器/动画系统负责时间轴资产。每个责任面都有明确输入、输出和失败处理，模块之间不复制同一份真相。

**文档交付判定**：本文把核心能力、数据结构、系统顺序、错误模型、资源所有权、性能预算、GPU ABI、验证方法和运行时降级行为固化；§27.1 区分已经由源码和测试验证的实现，以及由下游渲染、物理、XR 或编辑器消费的稳定契约。API 以 workspace 中的四个相机 crate 为准。

---

## 2. 参考产品取舍

| 维度 | 借鉴来源 | 借什么（形态） | 不借什么 |
|---|---|---|---|
| 物理相机 | UE `UCineCameraComponent` | Filmback/Lens/Focus 的参数化、焦距↔FOV 换算、对焦距离 | 其蓝图/反射运行时、源代码 |
| 物理曝光 | Frostbite / Filament | EV100、Sunny-16、Saturation-Based Speed、ISO/快门/光圈三联 | 其具体 tonemapper 实现 |
| 导演/虚拟相机 | Unity Cinemachine | VCam + Brain + 优先级抢镜 + 状态驱动混合 + Body/Aim 组合器 + Impulse | 其 C# 组件图、源代码 |
| 弹簧臂避障 | UE `USpringArmComponent` | 探针半径碰撞拉近、滞后 lag、目标偏移 | 其碰撞实现（我们用 prism_physics 查询） |
| 投影精度 | Reversed-Z / 无限远 | 深度缓冲精度翻转、无限远平面消除远裁剪 | — |
| 斜裁剪面 | 水面反射经典技法 | oblique near-plane 修改投影使近平面贴水面 | — |
| 时域抖动 | TAA/DLSS/FSR | Halton(2,3)/Sobol 子像素抖动、上一帧 view-proj 双缓冲 | 具体放大器算法（归 temporal_upscale） |
| 相机相对渲染 | Star Citizen / UE5 LWC | 以相机为原点组织渲染，world→view 用 f64 差分再降 f32 | — |
| 多视图家族 | 本仓 view_family | ViewKind/ViewImportance/ViewRegistry 直接复用 | — |
| 控制器 | DCC / FPS/RTS 相机 | fly/orbit/pan/follow 的手感与阻尼 | — |

**核心取舍**：相机**意图与结果分离**（intent `Camera` vs snapshot `ViewData`）是全文基石——它让导演能自由改意图而不污染「这一帧渲染用的矩阵」，让时域能持有「上一帧结果」而不被本帧意图抖动破坏，让多视图能共享同一套派生逻辑。

---

## 3. 档位化（capability / quality tier）

| 能力 | 开关（feature） | 默认 | 关掉的后果 |
|---|---|---|---|
| 物理镜头/曝光 | `physical_lens` | 开（当前默认 feature） | 退化为 FOV + 固定曝光标量 |
| 自动曝光（直方图/平均亮度自适应） | `physical_lens` + 渲染侧回读 | 开 | 仅手动 EV100 |
| 导演/虚拟相机 | `prism_camera_rig/director` | 开（`prism_camera_rig` 默认） | 无混合，直接使用调用方提供的位姿 |
| 相机 Rig（弹簧臂/避障/跟随） | `prism_camera_rig/rig` | 开（`prism_camera_rig` 默认） | 无 rig，手动驱动位姿 |
| 相机级后处理栈 | `post` | 开 | 全局默认后处理，无 per-camera 覆盖 |
| 时域 jitter + 历史矩阵 | 恒开（无成本可关 jitter） | 开 | 关 jitter 则 TAA/DLSS 失效，退 MSAA/无 AA |
| 立体 / XR | `stereo` | 关 | 无 per-eye / late-latch |
| 大世界相机相对 | `CameraIntent.relative_rendering` | 开（当前默认值） | 使用绝对渲染矩阵，适合原点附近场景 |
| 确定性/回放 | `determinism` | 关 | 导演混合/噪声顺序不保证跨平台一致 |
| 编辑器集成（viewport gizmo / 相机预览） | 下游编辑器装配 | 不属于相机 crate feature | 只使用 `ViewData` 和诊断数据，不进入核心 crate |

quality tier（运行时档位）示例：

| tier | jitter 样本 | 自动曝光 | 景深质量 | 弹簧臂探针 | 多视图上限 |
|---|---|---|---|---|---|
| Low（移动/集显） | 关或 4 | 平均亮度 | 关 | 单射线 | 2 |
| Medium | 8 | 直方图粗 | 半分辨率 | 球扫 | 4 |
| High | 16 | 直方图精 | 全分辨率 Bokeh | 球扫 + 多探针 | 8 |
| Cinematic（离线/过场） | 32+ | 物理曝光锁定 | 散景核 | 连续 CCD | 16+ |

档位只改**预算与质量**，不改相机的 API 形态与矩阵正确性。

### 3.1 Feature 组合与失败处理

当前 workspace 将 feature 用作构建组合和下游能力声明，核心数据契约在 default 与 no-default 配置下保持可编译；quality tier 只改变运行期预算，不能改变数学结果。组合规则固定如下：

| 组合 | 处理方式 |
|---|---|
| `physical_lens`/`post`/`stereo`/`determinism` 等能力 feature 关闭 | 保留核心数据 API，由应用选择手动曝光、下游后处理、单视图或非确定性路径；不静默改变核心矩阵计算 |
| `prism_camera_rig/director` 或 `rig` 关闭 | 不编译对应 Rig 能力；调用方直接提供 `CameraPose` 或使用无查询的直伸臂策略 |
| `std` feature 关闭 | 保持 `no_std + alloc` 核心；不引入标准库状态 |
| 外部 XR、物理、渲染算法采用契约消费者模式 | 双眼由 `StereoConfig` 派生，Rig 通过 `SceneQuery` 注入查询，后处理由稳定的相机意图与排序契约驱动 |
| 目标尺寸为零、near 非正、far 不大于 near、FOV 越界 | 返回 `CameraErrorCode` 对应错误；ECS adapter 记录 `CameraDiagnostics`，清理该实体的历史和注册句柄，不发布无效视图 |

相机核心错误为无分配值 `CameraError { code, field, value, handle }`；ECS adapter 以 `CameraViewError { entity, error }` 关联实体并通过 `CameraDiagnostics` 暴露本帧错误。错误不会通过 panic 传播到渲染线程；上一帧视图保留策略由上层发布器负责，核心数学层只返回明确错误。

---

## 4. 分层架构与 crate 布局

```
意图层（作者/脚本可写）────────────────────────────────
  prism_camera            CameraIntent / Projection / Lens / PostStack / RenderTarget 纯数据
         │                (no_std + alloc；只持有可编辑意图，不算每帧结果)
         │
  prism_camera_rig        VirtualCamera / Director(Brain) / 跟随·瞄准·弹簧臂·噪声·Impulse
         │ (feature: director/rig；读目标 transform + 物理查询 → 写回输出相机位姿/参数)
         ▼
派生层（每帧提取，只读结果）──────────────────────────
  prism_camera(extract)   位姿(CameraPose) + 投影 → ViewData{ view, proj, view_proj,
         │                prev_view_proj, jitter, frustum, exposure, viewport }
         │                注册/更新进 view_family::ViewRegistry
         ▼
消费层（下游共享视图契约）────────────────────────────
  prism_render_visibility 吃 frustum 做剔除
  prism_render_architecture
     ::view_family         ViewRegistry（本 crate 写入，backend 读取）
     ::temporal_upscale    吃 jitter + prev_view_proj
     ::motion              吃 prev_view_proj 算运动矢量
     backend/gpu_scene      吃 ViewData 填 View UBO / GPU-driven 剔除
  prism_render(post)        吃 PostStack + exposure 组后处理图

下层基座（已定稿）──────────────────────────────────
  prism_transform          相机位姿空间真相（Local/Global/Affine3 + look_at + 大世界）
  prism_math               Mat4/Affine3/Quat/Dir3 + 投影构造 + SIMD
  prism_window(可选)        RenderTarget 表面尺寸 / 缩放因子 / HiDPI
  prism_input(可选)         指针 → 射线 / 拾取
```

`prism_camera_ecs` 是可选的 ECS/App 适配层：它提供 `CameraBundle`、`CameraFrame`、`CameraPlugin`、`CameraViewRegistry`、`PublishedViews` 和 `CameraDiagnostics`，在 `PostUpdate` 中执行“intent + pose + history -> derive_view -> publish_view”。核心数学 crate 不拥有 ECS 生命周期；不使用 ECS 的调用方直接调用 `derive_view` 和 `publish_view`。

crate 职责：

- **`prism_camera`（本文核心 crate，`no_std + alloc`）**：相机意图组件、投影代数、`ViewData` 派生与 `view_family` 注册、jitter 序列、相机相对渲染、后处理栈意图模型、RenderTarget 契约、输入互转 helper。`forbid(unsafe_code)`。
- **`prism_camera_rig`（配套 crate，feature `director`/`rig`）**：虚拟相机、Brain（优先级 + 混合）、跟随/瞄准组合器、弹簧臂避障（经 `SceneQuery` trait 注入查询）、程序化噪声与 Impulse。它只生成 `CameraPose` 等纯数据结果，不碰 `ViewData` 或 GPU 资源。
- **`prism_camera_ecs`（ECS/App 适配层）**：把核心纯函数接入 `prism_ecs` 和 `prism_app` 的实体、资源、`PostUpdate` 生命周期；不复制相机数学和视图注册真相。
- **职责清晰**：本 crate 负责把观察意图编译为每帧视图契约；渲染图、表面、后处理算法和物理求交分别由 §1 已列出的责任模块提供，并通过结构化契约接入。

模块布局（`prism_camera/src/`）：

```
lib.rs            公共导出与 prelude
projection.rs     Projection、Perspective、Orthographic、Custom、oblique、SubView
view.rs           CameraIntent、CameraPose、ViewData、历史和 ViewRegistry 桥接
jitter.rs         Halton/Sobol 序列、子像素抖动和关闭档
lens.rs           Filmback、Lens、Focus、EV100、Physical、Auto exposure
post.rs           PostStack、节点域、固定阶段排序和容量校验
target.rs         RenderTarget、Viewport、清屏和尺寸解析
stereo.rs         per-eye、late-latch 和 StereoPair
pick.rs           viewport↔world、ray、深度反投影和拾取 helper
frustum.rs        从 ViewData 提取 5/6 面 frustum
gpu.rs            View UBO 记录、origin split、schema 和 flags
advanced.rs       过场、自动对焦、快门、foveation、批量 key、LOD、构图、确定性 digest、外部姿态、遮挡、前瞻和历史稳定化
error.rs          结构化错误码和 stale handle 关联
```

---

## 5. 核心模型：Camera / Projection / ViewData / Lens / PostStack

全系统只有两个一等概念：**意图（Camera + 伴随组件）** 与 **结果（ViewData）**。意图可写、可被导演改写；结果每帧派生、只读、注册进 `view_family`。

### 5.1 CameraIntent（相机意图）

```rust
/// 一个「观察意图」。位姿由同实体的 prism_transform 提供（look_at 等约束亦在此），
/// 本组件只持有「如何成像」的意图，不持有矩阵（矩阵在 ViewData）。
#[derive(Clone, Copy, Debug)]
pub struct CameraIntent {
    /// 是否参与本帧渲染（关则不派生 ViewData、不占视图槽）。
    pub active: bool,
    /// 渲染顺序：同一 RenderTarget 上多相机叠放时由小到大合成（minimap/PiP）。
    pub order: i32,
    /// 输出目标（主窗口 / 具名窗口 / 离屏纹理）。
    pub target: RenderTarget,
    /// 目标内的子视口（`Viewport::FULL` 表示整个 target）。分屏/画中画用。
    pub viewport: Viewport,
    /// 子像素抖动序列及幅度。
    pub jitter: JitterSequence,
    pub jitter_scale: f32,
    /// 手动、物理或自动曝光意图。
    pub exposure: Exposure,
    #[cfg(feature = "physical_lens")]
    pub lens: Option<Lens>,
    /// 清屏策略（颜色、深度或不清）。
    pub clear: ClearMode,
    /// 本相机映射到 view_family 的角色（默认 Main）。
    pub kind: ViewKind,
    /// 向共享子系统声明的重要度（流送/几何LOD/阴影LOD 偏置）。
    pub importance: ViewImportance,
    /// 是否使用相机相对渲染原点。
    pub relative_rendering: bool,
}
```

ECS 场景使用 `prism_camera_ecs::CameraBundle` 打包意图、位姿和历史；核心 crate 不定义 `Camera`、`Camera2d` 或 `Camera3d` 组件，也不直接拥有 ECS `Bundle`：

```rust
let camera = app.world_mut().spawn(prism_camera_ecs::CameraBundle::default());
```

**不变量**：`CameraIntent` 永不持有矩阵、frustum、jitter 相位或曝光标量——这些全在 `ViewData`，由派生函数每帧生成。作者、Rig 或 ECS 系统改意图/位姿，派生函数只产生结果快照。

### 5.2 Projection（投影意图）

见 §6。`Projection` 是枚举（Perspective / Orthographic / Custom），持有生成 `clip_from_view` 矩阵所需的意图参数；物理镜头通过 `CameraIntent.lens` 修改透视 FOV，不增加独立的 `Physical` 枚举分支。当前核心 API 由调用方在目标尺寸变化时传入新的 `target_size`，不提供 `Projection::update(width,height)` 方法。

### 5.3 ViewData（每帧观察结果，只读快照）

```rust
/// 一帧里某个视图的全部几何/时域/曝光结果。不可变；由派生系统生成，
/// 注册进 view_family::ViewRegistry，供剔除/时域/后处理/backend 共享。
#[derive(Clone, Debug)]
pub struct ViewData {
    /// 视图在家族中的句柄（generational，复用槽不会误别名）。
    pub handle: ViewHandle,
    pub kind: ViewKind,

    /// world → view（相机相对模式下 world 已是「相对相机」坐标，见 §8）。
    pub view_from_world: Affine3,
    /// view → world（= view_from_world 逆，拾取/天空盒用）。
    pub world_from_view: Affine3,
    /// view → clip（投影矩阵，含 reversed-Z / 无限远）。
    pub clip_from_view: Mat4,
    /// world → clip（= clip_from_view * view_from_world，提取 frustum 用）。
    pub clip_from_world: Mat4,
    /// world → clip 的稳定版本，不含 jitter；剔除、拾取与历史重投影使用。
    pub clip_from_world_unjittered: Mat4,

    /// 本帧子像素抖动（NDC 偏移，单位：相对裁剪空间；关 jitter 时为 0）。
    pub jitter: Vec2,
    /// 上一帧的 world→clip（去 jitter 的「稳定」版本），运动矢量/TAA 重投影用。
    pub prev_clip_from_world: Mat4,
    /// 历史是否可用；首帧、切镜、重建交换链和显式清历史时为 false。
    pub history_valid: bool,
    /// 本帧是否是硬切/瞬移；消费者必须丢弃旧历史。
    pub teleport: bool,

    /// 本帧 frustum（6 个半空间），由 clip_from_world 提取（见 §16）。
    pub frustum: Frustum,

    /// 曝光：线性 exposure 乘子（= 1/(1.2·2^EV100) 之类，见 §9）。
    pub exposure: f32,

    /// 像素级视口（物理像素，含 HiDPI 缩放），拾取/UI 命中用。
    pub viewport_px: ViewportRect,
    /// 相机世界位置（相机相对模式下为「真实 f64 原点」，供 shader 还原）。
    pub world_position: DVec3,
}
```

**关键设计点**：
- `prev_clip_from_world` 存的是**去抖动**的稳定矩阵，而本帧 `clip_from_world` 含 jitter。运动矢量计算必须用「本帧去抖 vs 上一帧去抖」，TAA 重投影用「本帧去抖采样历史」——所以我们额外缓存一份 `clip_from_world_unjittered`（见 §7，为简洁上表省略，实际结构含之）。
- `world_position` 用 `DVec3`：大世界下相机真实位置超出 f32 精度，shader 需要它把「相机相对」坐标还原到绝对世界（如用于程序化天空、雾、大气）。
- `ViewData` 不含后处理结果，只含 `exposure` 这一个被几乎所有后处理吃到的标量；完整后处理意图在 `PostStack`（§13），单独注册。

### 5.4 生命周期、错误与所有权

- **所有权**：作者/脚本只写意图组件；`CameraSystems::Derive` 独占写入 `ViewDataArena` 与历史槽；渲染线程只读不可变快照。ViewHandle 为分代句柄，实体销毁后旧句柄永远不能重新指向新相机。
- **更新方式**：`ViewData` 以帧代号标记，消费者只能读取当前帧代号；跨帧持有必须复制句柄和历史 token，不得保存裸引用。
- **无效输入**：投影、视口、镜头、立体参数均先经 `validate()`；失败时保留上一帧有效视图并置 `ViewStatus::InvalidInput`，禁止将 NaN/Inf 写入 ViewRegistry 或 GPU 缓冲。
- **资源释放**：RenderTarget、LUT、离屏纹理、GPU View slice 由拥有它们的渲染资源域回收；相机销毁只释放引用和历史槽，不直接销毁渲染资源。
- **线程规则**：意图变更发生在主/ECS 世界；派生阶段产生只读快照；GPU 上传阶段只接受快照。三阶段之间使用显式帧快照，不共享可变容器。

### 5.5 Lens / Exposure（物理镜头，feature `physical_lens`）

见 §9。`Lens` 持有 Filmback（传感器尺寸）、焦距、光圈、对焦距离；`Exposure` 持有 ISO/快门/光圈或直接 EV100。派生时 Lens → FOV 回写进透视投影，Exposure → `ViewData.exposure` 标量。

### 5.6 PostStack（相机级后处理意图，feature `post`）

见 §13。每相机一份有序的后处理意图列表（曝光/泛光/景深/运动模糊/色调映射/色彩分级/暗角…），渲染侧据此组后处理子图。本 crate 只定义**意图与排序契约**，不实现算法。

### 5.7 三层数据流一图

```
作者/脚本/导演 ─写─▶ Camera + Projection + Lens + PostStack（意图，可变）
                               │  每帧 CameraSystems::Derive
                               ▼
                        ViewData（结果，只读）── 注册 ─▶ view_family::ViewRegistry
                               │                               │
         ┌─────────────────────┼───────────────┬───────────────┤
         ▼                     ▼               ▼               ▼
     frustum→剔除        jitter+prev→时域    exposure→后处理   ViewUBO→backend/GPU-driven
```

---

## 6. 投影系统

投影是「view 空间 → clip 空间」的映射，决定 FOV、纵横比、近远裁剪与**深度精度**。

```rust
#[derive(Component, Clone, Debug)]
pub enum Projection {
    Perspective(PerspectiveProjection),
    Orthographic(OrthographicProjection),
    /// 自定义：调用方直接提供已验证的 clip_from_view 矩阵和裁剪元数据。
    Custom { clip_from_view: Mat4, near: f32, far: Option<f32>, reverse_z: bool },
}
```

### 6.1 Reversed-Z（本仓默认且强制）

深度缓冲在传统 `[0,1]`（近 0 远 1）下，浮点精度集中在近处、远处急剧退化（z-fighting）。**Reversed-Z**（近 1 远 0）配合浮点深度缓冲，让精度沿整个范围近乎均匀。本仓**所有**透视投影默认产出 reversed-Z 的 `clip_from_view`，剔除/比较/清深度全按「近 1 远 0、深度测试 `GREATER`」约定。这是现代 AAA 的既定做法，不提供「正向 Z」开关以避免管线分叉。

### 6.2 无限远平面（Reverse Infinite Perspective）

Reversed-Z 下可令远平面趋于无穷（`far → ∞`）而不损失精度（远处映射到深度 0）。好处：无需远裁剪、天空盒/大气/远山不被裁、级联阴影远端自然。构造：

```
// reversed-Z 无限远透视（列主序，右手，view 看向 -z）
// f = 1/tan(fovy/2), a = width/height, n = z_near
// clip_from_view =
// [ f/a   0     0        0   ]
// [ 0     f     0        0   ]
// [ 0     0     0        n   ]
// [ 0     0    -1        0   ]
```

有限远与无限远由 `PerspectiveProjection.far: Option<f32>` 选择（`None`=无限远）。

### 6.3 PerspectiveProjection

```rust
pub struct PerspectiveProjection {
    /// 垂直 FOV（弧度）。物理镜头模式下由焦距+传感器算出并回写此处。
    pub fov_y: f32,
    /// 纵横比（width/height），由 update() 维护。
    pub aspect: f32,
    pub near: f32,
    /// None = 无限远（reversed-Z）。
    pub far: Option<f32>,
    /// FOV 随 render area 变化的锚定轴（宽/高/自动），对标 UE 的 aspect 约束。
    pub fov_axis: FovAxis,
}
```

### 6.4 OrthographicProjection

正交用于 2D、小地图、CAD、等距。持有 `scaling_mode`（固定世界高 / 固定像素比 / 自动适配）、`scale`、`near/far`、`viewport_origin`（锚点）。2D 下仍走 reversed-Z 以统一深度约定。

### 6.5 斜裁剪近平面（oblique near-plane）

水面/镜面反射需要「把近裁剪面贴到反射平面」以剔除水下几何。提供 `Projection::with_oblique(plane: Plane)`：对已算出的 `clip_from_view` 做 Lengyel 斜裁剪修正，返回新矩阵。该视图在 `view_family` 里 `kind = Reflection`。

### 6.6 Sub-view / tiled（离线分块、超大分辨率出图）

`SubView { full_size, offset, size }` 允许把一个逻辑视图切成多块（离线路径追踪的 `OfflineTile`、8K 截图分块）。投影按 sub-rect 偏移裁剪矩阵，各块独立成 `ViewData` 但共享曝光/jitter 相位。

---

## 7. 视图矩阵与时域

### 7.1 派生流水线（每帧，CameraSystems::Derive）

```
对每个 active 相机:
  1. 读 GlobalTransform → world_from_view(Affine3)；取逆得 view_from_world
     （相机相对模式下先做大世界降精度，见 §8）
  2. Projection::update(area) → clip_from_view（reversed-Z/无限远）
  3. clip_from_world_unjittered = clip_from_view * view_from_world
  4. jitter = sequence[frame % N] 映射到 ±0.5 像素 → NDC 偏移
     clip_from_view_jittered = translate_ndc(jitter) * clip_from_view
     clip_from_world = clip_from_view_jittered * view_from_world
  5. frustum = extract_frustum(clip_from_world_unjittered)   // 用去抖版剔除
  6. exposure = exposure_from_lens_or_manual()               // §9
  7. ViewData{ ..., prev_clip_from_world = 上一帧 unjittered }
  8. view_family.register_or_update(handle, kind, importance)
  9. 把本帧 unjittered 存入 per-camera 历史槽，供下一帧的 prev
```

**要点**：剔除用**去抖**矩阵（jitter 只是亚像素级，但剔除要稳定、避免物体在裁剪面附近闪烁）；采样历史/算运动矢量用「去抖 vs 去抖」；只有最终光栅化用含 jitter 的矩阵。三者分明，避免 TAA 鬼影与剔除抖动。

### 7.2 Jitter 序列

```rust
pub enum JitterSequence { Off, Halton23 { len: u32 }, Sobol { len: u32 } }
```

- 默认 `Halton23 { len: 8 }`（移动）/ `16`（桌面）/ `32`（电影）。
- jitter 幅度 = `±0.5 / render_extent`（像素级，NDC 下），乘以可调 `jitter_scale`（DLSS/FSR 推荐 1.0，FXAA 用 0）。
- 相位随「渲染分辨率」归一：动态分辨率/棋盘渲染下 jitter 幅度随 extent 自动缩放，避免分辨率切换时 TAA 崩。

### 7.3 上一帧矩阵与首帧/瞬移处理

- 每相机维护「上一帧 unjittered `clip_from_world`」历史槽（generational，相机销毁即回收）。
- **首帧**：prev = 本帧（运动矢量=0，避免首帧全屏糊）。
- **瞬移/切镜**（导演硬切、传送）：标记 `teleport`，令 prev = 本帧并清历史缓冲（通知 temporal_upscale 丢弃历史），避免切镜后残影。导演混合（软切）不置 teleport。

### 7.4 与 motion / temporal_upscale 的契约

`view_family` 的消费者（`motion`、`temporal_upscale`）从 `ViewData` 读取 `jitter`、`clip_from_world`（含抖）、`prev_clip_from_world`（去抖）与 `teleport` 标记。本 crate 保证这四者每帧自洽，是 TAA/DLSS/FSR 正确工作的充要输入。

---

## 8. 大世界：相机相对渲染（Camera-Relative）

远离世界原点（>~10 km）时，f32 的 world 坐标尾数不足，顶点在 view 空间抖动（jitter/swim）。AAA 方案：**以相机为渲染原点**。

### 8.1 原理

设相机真实世界位 `C`（f64/DVec3）。渲染时不直接用 `view_from_world`，而是：
1. 每个物体的 world 位 `P`（f64）→ 相对位 `P' = P − C`（在 f64 下相减，结果幅度小，降 f32 不丢精度）。
2. 构造 `view_from_relative`：旋转部分同相机朝向，平移部分为 0（相机在相对系原点）。
3. shader 收到的是「相对相机」的小坐标，f32 足够。

### 8.2 与 prism_transform 协作

`prism_transform`（feature `f64` + 原点重定位）已提供「网格/容器原点重定位」。`prism_camera` 通过 `CameraIntent.relative_rendering` 在派生时：
- 读相机 `GlobalTransform` 的 f64 世界位 → `ViewData.world_position: DVec3`。
- `view_from_world` 存**旋转为主、平移置零**的相对视图矩阵；绝对平移由 `world_position` 携带。
- 渲染提取时，实例的 world 矩阵在 CPU（或 compute）侧做 `translation -= view.world_position` 后降 f32 上传。

与 transform 的原点重定位是**两级**：transform 做「大网格原点搬迁」（粗，几 km 一跳），camera-relative 做「每帧以相机为原点」（细，连续）。二者正交叠加，远距无抖且无跳变。

### 8.3 关掉的退化

`relative_rendering` 关：`world_position` 仍填（供雾/天空），但 `view_from_world` 使用常规绝对平移，适合原点附近的常规场景（移动端/小地图游戏）。

---

## 9. 物理镜头与曝光（feature `physical_lens`）

把「FOV + 魔法曝光数字」升级为**真实相机参数**，让美术用镜头语言工作，并得到物理一致的曝光与景深。

### 9.1 Filmback / Lens / Focus

```rust
pub struct Lens {
    /// 传感器尺寸（mm），决定给定焦距下的 FOV。预设：Super35/FullFrame/APS-C。
    pub filmback: Filmback,          // { sensor_width_mm, sensor_height_mm }
    /// 焦距（mm）。与传感器宽共同决定水平 FOV： fov = 2*atan(sensor_w / (2*focal))。
    pub focal_length_mm: f32,
    /// 光圈 f 数（f-stop）。越小进光越多、景深越浅。
    pub f_stop: f32,
    /// 对焦距离（m）。景深在此处最清晰。
    pub focus_distance_m: f32,
    /// 光圈叶片数（bokeh 多边形形状）与旋转。
    pub blades: u8,
}

impl Lens {
    pub fn cine_35mm() -> Self { /* Super35, 35mm, f/2.8, focus 3m */ }
    pub fn full_frame_50mm() -> Self { /* FullFrame, 50mm, f/1.8 */ }
    pub fn with_focal(mut self, mm: f32) -> Self { self.focal_length_mm = mm; self }
    pub fn with_fstop(mut self, f: f32) -> Self { self.f_stop = f; self }
    /// 由焦距+传感器算垂直 FOV，回写进 PerspectiveProjection。
    pub fn fov_y(&self) -> f32 { 2.0 * (self.filmback.sensor_height_mm / (2.0 * self.focal_length_mm)).atan() }
}
```

**焦距↔FOV 单一真相**：物理模式下 `fov_y` 由镜头算出并回写投影；美术改焦距，FOV 自动跟随（与 UE CineCamera 同形）。

### 9.2 曝光（EV100 物理模型）

曝光由光圈/快门/ISO 三联决定（Frostbite/Filament 模型）：

```
EV100 = log2( f_stop^2 / shutter_time * 100 / ISO )
曝光乘子 exposure = 1 / (1.2 * 2^EV100)     // Saturation-Based Speed，1.2 为校准常数
```

```rust
pub enum Exposure {
    /// 直接给 EV100（手动）。
    Manual { ev100: f32 },
    /// 物理三联（光圈 f 数、快门秒、ISO）。
    Physical { f_stop: f32, shutter_s: f32, iso: f32 },
    /// 自动：从上一帧场景平均/直方图亮度自适应（渲染侧回读 → 这里平滑）。
    Auto { min_ev: f32, max_ev: f32, adapt_speed: f32, metering: Metering },
}
```

- **Sunny-16**：`f/16, 1/ISO 秒` 下 EV100≈15，提供物理锚点与自检单测。
- **自动曝光**：渲染侧算场景亮度直方图/平均 → 本 crate 用 `prism_time` 的 dt 做指数平滑（亮适应快、暗适应慢，仿人眼），输出稳定 `ViewData.exposure`。平滑在 CPU 侧做（可确定性），避免 GPU 回读抖动直接入画。
- **Metering**：平均 / 中央重点 / 点测 / 直方图分位，决定渲染侧如何聚合亮度。

### 9.3 景深（Depth of Field）

由 `focus_distance`、`f_stop`、`focal_length` 算**弥散圆（CoC）**：

```
CoC(d) = |A * f * (d - focus) / (d * (focus - f))|   // A=光圈直径=focal/f_stop
```

本 crate 产出景深参数（焦平面距离、近/远模糊斜率、最大 CoC、光圈叶片形状）注入 `PostStack` 的 DoF 节点；算法在渲染侧（gather/scatter bokeh）。电影档用散景核 + 叶片多边形，桌面档用半分辨率 gather，移动档关闭。

### 9.4 与投影/后处理的接缝

Lens → `PerspectiveProjection.fov_y`（几何）、`ViewData.exposure`（曝光标量）、`PostStack::DepthOfField`（景深意图）。三条线各走各的消费者，互不耦合。

---

## 10. 导演系统：虚拟相机 / Brain / 优先级 / 状态驱动混合（feature `director`）

对标 Cinemachine：游戏里**不手动改主相机**，而是摆一堆「虚拟相机（VirtualCamera）」各自描述「想看什么、怎么构图」，由 **Brain（导演）** 按优先级/状态选中并在切换时平滑混合，最后写回唯一输出相机的位姿与参数。

### 10.1 模型

```rust
/// 一个「候选观察意图」。自身不渲染，只被 Brain 评估/采样。
#[derive(Component, Clone, Debug)]
pub struct VirtualCamera {
    /// 优先级：Brain 选当前最高优先级且 enabled 者为「活跃」。
    pub priority: i32,
    pub enabled: bool,
    /// 构图：位姿如何由目标推导（见 §11 的 Body/Aim 组合器）。
    pub body: BodyRig,       // 位置：跟随/环绕/弹簧臂/固定
    pub aim: AimRig,         // 朝向：瞄准/锁定/构图框/固定
    /// 本 vcam 自带的镜头/后处理意图（切到它时一并混合）。
    pub lens: Option<Lens>,
    pub post: Option<PostStack>,
    /// 噪声/手持抖动（§12）。
    pub noise: Option<CameraNoise>,
    /// 切到本 vcam 时的默认混合规则（可被 Brain 全局覆盖）。
    pub blend_in: Blend,
}

/// 挂在真正的输出 Camera 实体上。每帧评估 vcam 集，混合写回本实体 transform/lens/post。
#[derive(Component, Clone, Debug)]
pub struct CameraBrain {
    /// 默认混合（未指定时用）。
    pub default_blend: Blend,
    /// 自定义混合查找表（from vcam → to vcam → Blend），对标 Cinemachine BlendList。
    pub custom_blends: BlendTable,
    /// 当前活跃链（混合期间可能同时采样两个 vcam）。
    pub state: BrainState,
}

pub enum Blend {
    Cut,                                  // 硬切（置 teleport，清时域历史）
    EaseInOut { secs: f32 },
    Linear { secs: f32 },
    Custom { secs: f32, curve: EaseCurve },
}
```

### 10.2 评估与混合流水线（CameraSystems::Director，派生之前）

```
1. 对每个 enabled VirtualCamera，运行其 Body/Aim 组合器 → 候选位姿 + lens/post 快照
2. Brain 选「优先级最高」为目标 vcam
3. 若目标 ≠ 当前活跃：启动混合（查 custom_blends，否则 default_blend）
   - Cut：立即切，Camera 实体标记 teleport=true（§7.3）
   - 时长 blend：在 [from, to] 间按曲线插值 位姿(位置 lerp + 旋转 slerp) / FOV / 曝光 / 后处理权重
4. 把混合结果「写回」输出 Camera 实体的 Transform / Projection.fov / Lens / PostStack
5. 清 teleport（混合类不置）
```

**写回而非并存**：导演产物最终落到唯一输出相机上，`ViewData` 只派生这一个（主）视图——这保证下游（时域/剔除/后处理）永远只面对一个确定的主视图，不被候选 vcam 干扰。

### 10.3 混合的正确插值

- **位置**：世界空间 `lerp`（大世界下在 f64 相对系插值再降精度）。
- **旋转**：`slerp`（四元数球面插值，避免欧拉万向节与最短路径问题）。
- **FOV/曝光/后处理标量**：按曲线标量插值；后处理「结构不同」时按权重交叉淡化（两套 stack 各出一半权重，渲染侧混合）。
- **混合中的时域**：混合期间相机在动，运动矢量自然非零，TAA 正常重投影；只有 `Cut` 置 teleport 清历史。

### 10.4 状态驱动与事件

vcam 的 `enabled`/`priority` 可由游戏状态机驱动（进入掩体→掩体 vcam 优先级升）。提供 `activate(vcam)` / `deactivate` 便捷 API 与「优先级栈」helper（临时推一个高优先级 vcam，弹出即恢复），对标 Cinemachine 的状态驱动相机。

Brain 的选择顺序完全确定：`priority` 降序、`activation_serial` 降序、实体分代 ID 升序；同一帧只允许一次状态迁移。混合状态保存 `from_snapshot`、`to_snapshot`、`elapsed` 和 `history_policy`，源/目标实体销毁时立即冻结对应快照并以剩余时长完成到目标，避免悬挂实体引用。`Cut`、渲染目标变化、投影类型变化、XR 重定位和显式传送均触发 `teleport`；普通优先级变更和参数修改只走连续混合。

---

## 11. 相机 Rig：跟随 / 瞄准 / 环绕 / 弹簧臂 / 避障（feature `rig`）

Body（位置）与 Aim（朝向）组合器，既可挂在 `VirtualCamera` 上（被导演采样），也可直接挂在输出相机上（无导演时）。

### 11.1 Body（位置组合器）

```rust
pub enum BodyRig {
    /// 固定在实体当前位姿（手动驱动）。
    Transform,
    /// 跟随目标 + 世界/本地偏移 + 三轴阻尼（lag）。
    Follow { target: Entity, offset: Vec3, damping: Vec3, frame: OffsetFrame },
    /// 环绕目标（轨道）：方位角/俯仰角/距离 + 各自阻尼。第三人称常用。
    Orbit { target: Entity, yaw: f32, pitch: f32, distance: f32, damping: Vec3 },
    /// 弹簧臂：从目标沿 -forward 伸出 arm_length，遇障碍拉近（§11.3）。
    SpringArm(SpringArm),
    /// 轨道约束（dolly）：沿样条/路径移动，参数由游戏推进。
    Dolly { path: DollyPath, position: f32, damping: f32 },
    /// 群组构图：自动取多个目标的包围，拉远到都入画（RTS/合影）。
    GroupCompose { targets: Box<[Entity]>, padding: f32, damping: Vec3 },
}
```

### 11.2 Aim（朝向组合器）

```rust
pub enum AimRig {
    Transform,                                   // 用实体自身旋转
    LookAt { target: Entity, damping: Vec3 },    // 盯住目标（经 transform look_at 约束）
    /// 构图框：把目标约束在屏幕某区域（硬/软区间 + 预测提前量）。对标 Cinemachine Composer。
    Composer { target: Entity, screen: Rect, soft_zone: Rect, lookahead: f32, damping: Vec3 },
    /// 朝向移动方向（载具/奔跑）。
    FaceVelocity { target: Entity, damping: Vec3 },
    /// 地平线锁定（防止手持噪声把地平线歪了）。
    HardLockHorizon,
}
```

### 11.3 弹簧臂与避障（SpringArm）

第三人称相机的核心：相机想待在「目标身后 arm_length」，但不能穿墙。

```rust
pub struct SpringArm {
    pub target: Entity,
    pub arm_length: f32,
    /// 相对目标的 socket 偏移（肩后、头顶）。
    pub socket_offset: Vec3,
    /// 碰撞探针半径（球扫，避免相机贴面穿插）。
    pub probe_radius: f32,
    /// 命中后拉近，离开障碍后以此速度平滑弹回（防抖）。
    pub return_speed: f32,
    /// 查询哪些碰撞层（墙/地形，忽略角色自身与可穿透物）。
    pub mask: CollisionMask,
    pub damping: Vec3,
}
```

避障流程（CameraSystems::Rig）：
1. 期望相机位 = socket + 臂方向 × arm_length。
2. 从 socket 向期望位做**球扫查询**（半径 probe_radius），经 `prism_physics` 的查询 trait（本 crate 只持 trait，不实现物理）。
3. 命中：相机拉到命中点前（减 skin），记录拉近量。
4. 无命中/障碍移开：以 `return_speed` 平滑弹回 arm_length（弹回比拉近慢，避免抖动与穿帮）。
5. 结果经 damping 平滑写入位姿。

**物理解耦**：避障需要场景查询，但本 crate `no_std` 且不依赖物理。方案：定义 `trait SceneQuery { fn sphere_cast(...) -> Option<Hit>; }`，`prism_camera_rig` 的避障系统泛型化该 trait；`prism_physics` 提供实现，App 装配时注入。无物理时降级为无避障（直伸臂）。

查询接口采用批量形式 `cast_batch(&[CastRequest], &mut [CastHit])`，单相机每帧最多一次主球扫和一次回退线扫；结果按请求序返回，命中距离必须满足 `0 <= distance <= arm_length`，法线必须归一化。没有查询实现时 `SpringArmStatus::Unavailable` 进入诊断流，输出相机仍保持可用但不会伪造“已避障”。

### 11.4 阻尼模型（lag）

统一用**半衰期阻尼**（frame-rate independent）：`x += (target - x) * (1 - exp(-dt / tau))`，`tau` 由 `damping` 换算。好处：不同帧率下手感一致（对标 Cinemachine 的 damping 与 UE 的 lag）。三轴/旋转各自独立 tau。

---

## 12. 相机抖动与冲击（feature `rig`，程序化噪声 + Impulse）

### 12.1 持续噪声（手持 / 载具颠簸）

```rust
pub struct CameraNoise {
    /// 6DOF 噪声：位置三轴 + 旋转三轴，各自振幅/频率。
    pub position: NoiseAxes,   // Perlin/Simplex 分形噪声
    pub rotation: NoiseAxes,
    /// 全局强度（0 关，1 满），可被游戏按状态缩放（受伤→晃得更凶）。
    pub gain: f32,
    /// 相位种子（多相机不同步，确定性模式下固定）。
    pub seed: u64,
}
```

噪声叠加在 Body/Aim 结果之上（最后一层），使用无全局状态的确定性梯度噪声，频率/振幅随 `gain` 缩放，并由 `temporal_clamp` 限制输出幅度。`HardLockHorizon`（§11.2）可在噪声后再锁地平线。

### 12.2 冲击（Impulse，爆炸/落地/受击）

对标 Cinemachine Impulse：事件源发出一个「冲击信号」（方向 + 强度 + 衰减曲线），监听相机按**距离衰减 + 时间包络**响应，产生瞬时抖动再回落。

```rust
pub struct ImpulseSource { pub position: Vec3, pub velocity: Vec3, pub envelope: Envelope, pub range: f32 }
// 相机侧：impulse_response = Σ source 的 (envelope(t) * falloff(dist) * dir)
```

包络用「冲 → 持续 → 衰减」ADSR 式曲线；多个冲击线性叠加并 clamp。确定性模式下冲击按事件序确定响应。事件使用单调 `ImpulseId` 去重；超出容量的旧事件按结束时间淘汰，禁止无界增长；位置/旋转冲击分别限幅，限幅发生在叠加之后、写回 Transform 之前。

### 12.3 与时域的关系

噪声/冲击是**真实相机运动**，运动矢量应包含它们（物体相对相机动了），所以它们在**派生前**写入相机位姿，TAA/运动模糊自然吃到。唯一例外：极高频噪声可能让 TAA 判为瞬移——提供 `noise.temporal_clamp` 限幅，避免误触 teleport。

---

## 13. 相机级后处理栈（feature `post`）

每相机一份有序后处理意图。本 crate 只定义**意图模型与排序契约**，具体算法在 `prism_render`。

```rust
#[derive(Component, Clone, Debug, Default)]
pub struct PostStack {
    /// 有序节点（渲染侧按此序组后处理子图）。
    pub nodes: Vec<PostNode>,
}

pub enum PostNode {
    /// 曝光应用（吃 ViewData.exposure；Auto 时含适应参数）。
    Exposure(ExposureSettings),
    Bloom(BloomSettings),                  // 阈值/强度/半径
    DepthOfField(DofSettings),             // 来自 Lens（§9.3）或手动
    MotionBlur(MotionBlurSettings),        // 吃运动矢量 + 快门角
    ChromaticAberration(f32),
    Vignette(VignetteSettings),
    FilmGrain(f32),
    Tonemap(Tonemapper),                   // ACES / AgX / Reinhard / 自定义曲线
    ColorGrade(ColorGradeSettings),        // 提升/伽马/增益 + LUT（prism_asset 句柄）
    Fxaa | Smaa(SmaaPreset),               // 若未用时域放大器
}
```

### 13.1 排序契约（线性管线约定）

渲染侧按**固定物理顺序**消费（而非 nodes 的任意顺序），本 crate 在装配时校验/重排为：`运动模糊(HDR) → 景深(HDR) → 泛光(HDR) → 曝光 → 色调映射(HDR→LDR) → 色彩分级(LDR) → 暗角/颗粒/色差(LDR) → AA(LDR)`。作者给意图，系统保证物理正确序；非法组合（如色调映射后再 bloom）在 debug 下警告。

### 13.2 与全局/档位协作

无 `PostStack` 组件的相机用全局默认栈（质量档位决定）。per-camera 栈覆盖全局（小地图相机关 DoF/MotionBlur；主相机全开）。导演混合时两套 stack 交叉淡化（§10.3）。

### 13.3 时域放大器优先

若启用 DLSS/FSR/TAA（`temporal_upscale`），则 `Fxaa/Smaa` 节点被忽略（时域已含 AA），且后处理在**放大后**分辨率执行色调/分级，在放大前执行需要运动矢量的节点——本 crate 的排序契约对此分段标注（`PostNode::pass_domain()` 返回 PreUpscale/PostUpscale）。

### 13.4 栈归一化、容量与失败处理

装配时执行 `PostStack::normalize()`：按固定阶段稳定排序；重复 Exposure、Tonemap、ColorGrade、AA 或相同 Raw ID 返回 `DuplicateExclusiveNode`，Raw ID 使用独立 key 空间，不会与内置节点冲突；非法域返回 `InvalidDomain`；节点数上限为 32，超过上限时返回 `CapacityExceeded` 并使用上一帧已验证的栈。

全局栈、相机栈和导演混合栈的优先级固定为“全局默认 → 相机覆盖 → 当前混合权重”；权重和必须归一化到 1。渲染侧无法提供某节点所需资源时只跳过该节点并发出 `PostNodeSkipped`，不得改变其它节点顺序或写入未初始化纹理。

---

## 14. 多视图：分屏 / 画中画 / 小地图 / 离屏 / 叠放

### 14.1 RenderTarget

```rust
pub enum RenderTarget {
    /// 主窗口（prism_window 的主表面）。
    PrimaryWindow,
    /// 具名窗口（多窗口）。
    Window(WindowRef),
    /// 离屏纹理（截图/反射/UI 里的 3D 预览/传送门）。
    Texture(Handle<Image>),
}
```

尺寸/缩放因子由 `prism_window`（窗口）或纹理尺寸（离屏）提供，驱动 `Projection::update` 与 jitter 幅度。

### 14.2 Viewport（子视口 + 叠放）

同一 target 上多相机，用 `Camera.order` 决定合成顺序（小先画，大叠上），`Camera.viewport` 决定各自画到哪块矩形：
- **分屏**：两相机各占左右半 viewport，order 相同、互不覆盖。
- **画中画 / 小地图**：主相机全屏 order=0 清屏；小地图相机角落 viewport、order=1、`clear=None`（叠加不清主画面）。
- **传送门 / 后视镜**：相机渲到离屏纹理，纹理贴到场景材质。

### 14.3 多视图提取并行

多个 active 相机的 `ViewData` 派生彼此独立 → 经 `prism_tasks` 并行提取。每个视图注册进同一 `ViewRegistry`，`ViewImportance` 聚合（主视图权重高、小地图低）驱动共享预算（流送/LOD/阴影）。

### 14.4 视图重要度与共享预算

直接复用 `view_family::ViewImportance { streaming, geometry_lod, shadow_lod }`：主相机高（细节拉满），小地图/后视镜低（省流送与几何）。`ViewRegistry::aggregate_importance`（component-wise max）或 `weighted_importance`（kind 加权和）决定全局共享子系统的服务强度。这把「哪个视图值得花资源」交给相机系统声明，渲染侧统一调度。

### 14.5 视图注册和合成规则

同一 `RenderTarget` 的排序键为 `(order, creation_serial, generational_handle)`；`order` 相同不得依赖 ECS 遍历顺序。视口必须是目标尺寸内的非空矩形，越界部分在注册阶段裁剪并记录 `ViewportClipped`。清屏策略只允许最底层视图执行颜色/深度清除；上层视图若请求清除，装配器将其拆成显式 clear pass，避免隐式改变其它相机结果。每帧注册结束后执行一次覆盖检测，重叠视口只产生诊断，不改变作者指定的 order。

---

## 15. 立体 / XR / 注视点（feature `stereo`）

### 15.1 per-eye 视图

一个 XR 相机派生**两个** `ViewData`（`kind = StereoEye`，左右各一），共享曝光/后处理/jitter 相位，但各自 `clip_from_view`（含瞳距 IPD 偏移与非对称 frustum）与 `view_from_world`。两眼注册进 `view_family` 为一对。

```rust
pub struct StereoRig {
    pub ipd_m: f32,                 // 瞳距
    pub per_eye_projection: [Mat4; 2], // 由 XR 运行时提供的非对称投影
    pub per_eye_pose: [Affine3; 2],    // 由 XR 运行时提供的头显位姿
}
```

### 15.2 late-latch（晚锁定位姿）

XR 延迟敏感：位姿要在**尽量接近提交**时才锁定，减少「预测位姿 vs 实际头动」的误差。方案：派生阶段用预测位姿出 `ViewData`，渲染提交前再做一次「位姿修正」（reprojection warp 由渲染侧做，本 crate 提供「最新位姿」更新钩子 `relatch_pose`）。

`StereoConfig::relatch` 只允许修改旋转、平移和对应的 `clip_from_world`、未抖动矩阵与 frustum；它不能修改投影类型、viewport、jitter 相位、曝光或视图句柄。渲染后端不支持 late-latch 时，固定使用派生阶段的预测位姿，仍保持左右眼顺序、历史标志和 GPU ABI 一致。

### 15.3 注视点渲染（foveated）

本 crate 提供「注视点中心 + 半径分级」意图（`FoveationHint`），注册进 `ViewData`；渲染侧据此做可变速率着色（VRS）或分辨率分级。眼动追踪数据由 XR 输入喂入中心点。

### 15.4 单通道立体（single-pass stereo）

两眼共享几何提交、仅视图矩阵不同时，`ViewData` 对以数组形式供 backend 做 instanced/multiview 单通道渲染（省一半 draw 提交）。本 crate 保证两眼 `ViewData` 的「可数组化」布局一致（相同 jitter 相位、相同剔除并集或各自 frustum）。

若后端不支持 multiview，自动退化为两个普通 view submission；若不支持 VRS，`FoveationHint` 退化为中心区域的分辨率提示，不影响几何正确性。左右眼必须共享同一 `frame_index` 和曝光结果，任一眼失败则整对视图拒绝提交，避免单眼历史错位。

---

## 16. 剔除与可见性集成

### 16.1 frustum 提取

从去抖的 `clip_from_world` 提取 6 个半空间（Gribb-Hartmann 平面提取：按行相加/相减得左右上下近远），归一化法线：

```rust
pub fn extract_frustum(clip_from_world: &Mat4) -> Frustum {
    // [0,1] clip：left=row3+row0, right=row3-row0,
    // bottom=row3+row1, top=row3-row1。
    // 普通 Z：near=row2, far=row3-row2。
    // reversed-Z：near=row3-row2, far=row2；无限远时 far=row2 恒真并删除。
    // 每面 normalize 后存 HalfSpace{ normal: Vec3, d: f32 }。
}
```

`Frustum` 的面顺序固定为 `[left, right, bottom, top, near]`，有限远投影再追加 `far`；无限远投影永远只输出 5 个有效面。这样 CPU SIMD、GPU compute 和 debug gizmo 使用同一个面索引，不依赖“reversed-Z 下交换语义”的隐式约定，也不会拿恒真 far 面误剔对象。

### 16.2 与 prism_render_visibility 的契约

`ViewData.frustum` 是剔除侧的几何输入。相机 crate 负责提取并发布它；`prism_render_visibility` 从 `view_family::ViewRegistry` 消费每个视图的 frustum 做：
- CPU 粗剔（AABB/Sphere vs frustum，SIMD 批量，经 `prism_math` 的 `intersects_obb`/`intersects_sphere`）。
- GPU-driven 细剔（frustum + Hi-Z，`ViewData` 的矩阵填进 View UBO，compute 剔除，§20）。

### 16.3 阴影级联与多 frustum

方向光级联阴影需把主相机 frustum 沿深度切成 N 段，每段算一个紧致包围并生成阴影视图（`kind = Shadow`）。本 crate 的 `frustum_corners(z_near, z_far)` 提供分段角点，阴影系统据此构级联；每个级联注册为独立 `ViewData`。

### 16.4 遮挡剔除接缝

本 crate 只给 frustum 与矩阵；遮挡剔除（Hi-Z/软件光栅）在渲染侧，吃本帧 `clip_from_world` 与上一帧 Hi-Z。`prev_clip_from_world` 让「上一帧深度重投影到本帧」的保守遮挡剔除成为可能。

---

## 17. 输入与拾取（feature `std`/配合 `prism_input`）

### 17.1 viewport ↔ world 互转

```rust
impl ViewData {
    /// 屏幕像素 → 世界射线（拾取、点击放置）。
    pub fn viewport_to_world_ray(&self, pixel: Vec2) -> Ray3d;
    /// 世界点 → 屏幕像素（None=在相机背后/视锥外），UI 跟随、血条定位。
    pub fn world_to_viewport(&self, world: Vec3) -> Option<Vec2>;
    /// 屏幕像素 + 深度(NDC) → 世界点（深度缓冲反投影）。
    pub fn viewport_to_world(&self, pixel: Vec2, ndc_depth: f32) -> Vec3;
}
```

互转用 `ViewData` 的**去抖**矩阵（拾取要稳定，不能被亚像素抖动影响命中），并正确处理 reversed-Z 深度、HiDPI 像素缩放、子视口偏移。

### 17.2 拾取（picking）

两种命中：
- **几何射线**：`viewport_to_world_ray` → 场景求交（经 physics/visibility 的射线查询 trait）。
- **GPU id-buffer**：渲染侧写实体 id 到 target，拾取读回像素 id（本 crate 提供「像素 → 实体」查询契约，渲染侧填实现）。

与 `prism_input` 协作：输入提供指针像素（已处理 HiDPI/多窗口/子视口归一），本 crate 把它变成射线/世界点；拾取结果以事件回灌 ECS。

当像素落在 viewport 外、深度为 NaN/Inf、深度不在 `[0,1]` 或逆矩阵不可逆时，函数返回 `PickError::{OutsideViewport, InvalidDepth, NonInvertibleView}`；不会返回一个看似合法的零射线。reversed-Z 深度反投影统一使用 `z_ndc=depth`，不再由调用者手动翻转。

### 17.3 多相机命中仲裁

多视口/分屏下，指针落在哪个相机的 viewport 内决定用哪个 `ViewData` 投射；叠放时按 `order` 从上到下测试，命中即止（画中画优先于主视图）。

仲裁结果携带 `camera_entity`、`view_handle`、命中像素、世界射线和 target 坐标；相机在本帧注销后，结果标记为 `StaleView`，禁止被下一帧当作有效点击消费。

---

## 18. 控制器（取代 bevy_camera_controller）

挂一个组件即得交互相机，内部用 §11 的 Rig + §17 的输入。全部是 `prism_camera_rig` 里的预设封装。

```rust
/// 自由飞行（DCC/调试）：WASD + 鼠标看向 + 滚轮调速 + Shift 加速。
pub struct FlyController { pub speed: f32, pub sensitivity: f32, pub boost: f32, pub damping: Vec3 }
/// 第一人称：地面约束 + 俯仰 clamp + 头部 bob（可选噪声）。
pub struct FpsController { pub sensitivity: f32, pub pitch_clamp: (f32, f32) }
/// 轨道（模型查看器/第三人称）：绕目标 yaw/pitch/zoom + 平移中心 +范围 clamp。
pub struct OrbitController { pub target: Vec3, pub distance: f32, pub min_max_pitch: (f32,f32), pub zoom_range: (f32,f32) }
/// 平移（2D/RTS/地图）：拖拽平移 + 滚轮缩放（朝光标缩放）+范围限制。
pub struct PanController { pub zoom_to_cursor: bool, pub bounds: Option<Rect> }
```

控制器系统在 `CameraSystems::Rig` 前运行，读 `prism_input`、写相机 `Transform`/`Projection`，再由 Rig/派生接手。阻尼统一用 §11.4 的 frame-rate-independent 半衰期。控制器与导演互斥（有 Brain 时控制器可驱动某个 vcam 而非直接主相机）。

控制器输入先经过 `CameraInputMap` 归一化为 `CameraCommand`（move/look/zoom/orbit/pan/boost）；同一帧多个设备按优先级合并，重复绑定只累加值一次。指针捕获丢失、窗口失焦、输入时间戳倒退时清空瞬时 look/pan，保留连续状态；所有角度、速度、距离先 clamp 再写意图。这样控制器不直接依赖键鼠 API，也不会把平台输入差异传播到 Rig 和导演。

---

## 19. 确定性与回放对接（feature `determinism`，复用 `prism_network`，不自立复制/回放）

> 定位先行：**相机是客户端本地的派生层，不是被复制的真相**。多人复制、录制、权威回放、延迟补偿这套机制已由 `prism_network_design_zh.md`（`prism_net_*` crate 族）统一定义——其立场是「服务器权威 + 瞬时态/持久态分离 + 事件流复制 + 预测/回滚 + 确定性回放对账」。本 crate **不另立相机专属的复制或回放通道**，只保证一件事：**相机计算是其输入（位姿/输入/固定步）的确定性纯函数**，从而能被网络侧的既有回放机制「免费重现」。

### 19.1 相机不进复制单元

- 每个客户端有自己的相机（第一/第三人称、观战视角各异），相机**不是被 `prism_net_replicate` 复制的 ghost 实体**。被复制的是相机的**输入源**（玩家实体位姿、瞄准输入、载具速度），这些本就走网络框架的快照/事件流通道。
- 因此相机不占用带宽、不参与 AOI 裁剪、不进存档（持久态）。它纯粹是「拿到已同步的输入后，本地算出我这台机器怎么看」。

### 19.2 相机作为确定性纯函数（供网络回放重现）

当 `prism_net_replay` / `prism_net_predict` 用「固定步 + 记录的输入」重放时，相机只要满足以下约束即可逐帧重现，无需单独录相机矩阵：
- **jitter**：序列由固定 `frame_index` 索引（不读 wall-clock），纯确定性。
- **导演混合**：混合进度由固定步累加（`prism_time` 的 fixed-tick，与网络 command frame 同一时基），vcam 评估顺序按稳定实体序。
- **噪声/Impulse**：噪声使用 seed 派生的确定性梯度采样，Impulse 按单调 ID 去重、固定容量和固定遍历顺序叠加；不用 `rand` 全局态，浮点求值顺序固定。
- **阻尼**：半衰期用固定步 dt，避免可变帧率分叉。
- **大世界**：相对系降精度的求值顺序固定（§8）。

### 19.3 与预测/回滚的关系（只跟随，不参与和解）

- 客户端预测/回滚（`prism_net_predict`）作用于**被仿真的实体状态**；相机只是**读取**和解后的权威/预测位姿去跟随（§11 Rig）。回滚纠正玩家实体位置时，相机经 §11.4 的阻尼平滑吸收这帧跳变（error smoothing），避免「橡皮筋」直接入画——这与网络框架 §10.5「自适应插值/可视平滑」是同一手法在相机侧的落点，不是另一套机制。
- 硬性瞬移（传送/切镜）才置 `teleport`（§7.3）清时域历史；回滚级别的小纠正走阻尼，不置 teleport。

### 19.4 观战 / 击杀回放 / 过场的数据来源

- **观战（spectator）/ 导播**：切到被观战玩家的输入源，用本地相机（或其 vcam）重新取景；视角不跨网复制，只复用已同步的目标状态。
- **击杀回放（killcam）/ 比赛回放**：回放数据与播放由 `prism_net_replay`（事件流 + 固定步重放）提供，相机作为确定性纯函数在重放时自然重建轨迹；本 crate 另提供「相机轨迹片段」（§28.1 `cutscene`）用于**作者态过场**，与网络回放是两条不交叉的路径。

### 19.5 不确定源的隔离

确定性模式下关闭「自动曝光的 GPU 回读适应」（回读有 1~2 帧延迟且非确定）或改用「确定性亮度估计」；此类纯表现、不影响玩法与网络对账的量（曝光/泛光强度）即便不确定也**不参与**网络回放对账——对账只覆盖玩法仿真态，相机表现层不在其列。这与 `prism_network` 「对账只覆盖玩法仿真态」的责任划分一致。

> 一句话：相机的「确定性」不是为了自己做回放，而是为了成为一个干净的纯函数，让 `prism_network` 既有的回放/预测机制能把它一并重现。相机不碰复制真相、不占带宽、不自立回放通道。

确定性回放输出同时写入 `CameraDeterminismDigest`：包含输入帧号、活跃 vcam 分代 ID、投影参数、量化后的位姿、jitter 相位和 teleport 标志；曝光回读、GPU 计时、诊断序列号和编辑器 gizmo 不进入 digest。回放比较使用字段级误差：位置 `1e-5 m`、旋转 `1e-5 rad`、矩阵元素 `2e-5`；超过阈值时报告第一个分叉字段和两个输入帧，便于定位而不是只报“回放不一致”。

---

## 20. GPU 视图常驻（View UBO / 多视图数组 / GPU-driven）

### 20.1 View UBO 布局

`ViewData` 的 GPU 镜像（std140/std430 对齐），每视图一份：

```wgsl
struct View {
    clip_from_world      : mat4x4<f32>,  // 含 jitter（光栅化用）
    clip_from_world_unjit: mat4x4<f32>,  // 去抖（重投影/剔除用）
    prev_clip_from_world : mat4x4<f32>,  // 上一帧去抖（运动矢量）
    view_from_world      : mat4x4<f32>,
    world_from_view      : mat4x4<f32>,
    world_origin_hi      : vec4<f32>,    // DVec3 的高 32 位，w=0
    world_origin_lo      : vec4<f32>,    // DVec3 的低 32 位，w=0
    exposure             : f32,
    schema_version       : u32,
    flags                : u32,         // history_valid/teleport/finite_far/relative
    view_index            : u32,
    jitter               : vec2<f32>,
    viewport             : vec4<f32>,   // xy=offset, zw=size（像素）
    frustum              : array<vec4<f32>, 6>, // 6 半空间，GPU 剔除用
};
```

GPU ABI 固定使用 16 字节对齐；`world_origin_hi/lo` 采用 Dekker split 保存 `DVec3`，禁止把大世界位置直接截断成单个 `vec3<f32>`。`flags` 的位定义为：bit0=`history_valid`、bit1=`teleport`、bit2=`finite_far`、bit3=`camera_relative`。`schema_version` 当前值为 `1`，版本不匹配时渲染侧拒绝绑定该 View slice 并输出结构化诊断。

### 20.2 多视图数组与 GPU-driven

多视图（级联阴影、立体、多相机）打包成 `array<View>`，compute 剔除对每视图并行跑；draw 用 `view_index` 索引。本 crate 保证布局稳定、字节紧凑、与 `prism_render_architecture::gpu_scene` 的实例缓冲对齐，使「GPU 剔除 + 间接绘制」无需 CPU 回读。

### 20.3 增量上传

视图数通常很少（1~16），相机不动的帧跳过上传（脏标记）；jitter 变化只更新含 jitter 的矩阵和 `jitter` 字段，不重新计算稳定矩阵。大批阴影级联变化时只上传变化的 slice。上传器采用 ring buffer + 帧代号回收，禁止在 GPU 完成读取前复用 slice。

### 20.4 相机相对的 shader 还原

shader 收到的实例坐标已是「相对相机」（CPU/compute 侧减过 `ViewData.world_position`）。需要绝对世界坐标的效果（程序化噪声纹理、全局雾、大气）用 `world_origin_hi + world_origin_lo + relative` 在 f32 下还原；还原只用于局部效果，几何变换始终使用相对坐标。

---

## 21. 与 Transform / App / Time / Window / 渲染 / 输入集成

### 21.1 系统集与顺序

以下名称是跨 crate 的装配顺序约定，不是当前 `prism_camera` 导出的枚举；当前可执行的核心发布系统由 `prism_camera_ecs::CameraPlugin` 安装为 `PostUpdate` 系统，Director、Rig、Transform 和 Window 的调度由上层应用负责。

```rust
pub enum CameraSystems {
    Director,   // 评估 vcam + Brain 混合 → 写回输出相机意图（feature director）
    Control,    // 控制器读输入写 transform（或驱动 vcam）
    Rig,        // Body/Aim 组合器 + 弹簧臂避障 + 噪声/Impulse → 写 transform
    UpdateProj, // render area 变化 → 上层更新 target_size/Projection 意图
    Derive,     // 位姿+投影+jitter+曝光 → ViewData，注册 view_family
}
```

顺序：`Director → Control → Rig → (transform 传播) → UpdateProj → Derive`。关键：`Derive` 必须在 `prism_transform` 的层级传播**之后**（相机位姿是 GlobalTransform 的函数），所以 `Derive.after(TransformSystems::Propagate)`，与 Bevy 的 `update_frusta.after(Propagate)` 同构。

完整帧分为五个可观测阶段：`InputNormalize`、`IntentResolve`、`TransformPropagate`、`ViewDerive`、`ViewPublish`。阶段之间只传不可变快照；核心 `derive_view` 返回结构化错误时，ECS adapter 记录 `CameraDiagnostics`、清理该实体历史并移除注册句柄，下一帧重新尝试。`ViewPublish` 只有在所有矩阵、视口、历史字段和 ABI 校验通过后才更新 ViewRegistry。

### 21.2 App 装配

```rust
app.add_plugins(prism_camera_ecs::CameraPlugin::default()); // ECS/App 发布适配层
// Director、Control、Rig、Prepare、Publication 由 CameraPlugin 按固定顺序装配；
// 渲染后处理算法仍由下游渲染侧消费 prism_camera 的 PostStack 契约。
```

`prism_camera_ecs::CameraPlugin` 在 `PostUpdate` 中调用核心 `derive_view`/`publish_view`，并通过 `CameraViewRegistry` 维护实体到稳定 `ViewHandle` 的映射；渲染插件作为读取者。二者经 `ViewRegistry` 单向解耦（相机写、渲染读）。

### 21.3 与 Window 的尺寸/缩放

`RenderTarget` 解析为具体表面尺寸 + `scale_factor`（HiDPI）；窗口 resize 事件触发 `UpdateProj` 与 jitter 幅度重算。多窗口下每相机绑各自表面。

### 21.4 与 Time 的协作

- 固定步 + 插值：相机逻辑（导演/rig）在固定步跑，`Derive` 用 `prism_time` 的插值 alpha 在固定步间平滑相机位姿（表现层丝滑），与 transform 插值同一套 alpha。
- 自动曝光/阻尼用 `dt` 做 frame-rate-independent 适应。

---

## 22. 可观测性与诊断（feature `trace`）

- **活跃视图清单**：每帧 active 相机数、各自 `ViewKind`、viewport、order、target。
- **导演状态**：当前活跃 vcam、混合源/目标/进度、本帧是否 teleport。
- **曝光**：各视图 `exposure` 标量、EV100、自动曝光适应中的目标/当前亮度。
- **jitter**：当前相位、序列类型、幅度（像素）。
- **frustum**：剔除输入面数（区分无限远 5 面）。
- **性能计数**：派生耗时、并行提取线程分布、View UBO 上传字节、Rig 球扫查询次数。
- **可视化（editor feature）**：视锥线框、弹簧臂探针、frustum 角点、焦平面与景深范围、构图框（Composer soft/hard zone）gizmo。

诊断经 `prism_diagnostic` 的计数器/trace span 暴露，可被编辑器 HUD/录制消费。

诊断数据按相机和帧分代保存，默认只保留 120 帧；每相机诊断预算为 4 KiB，超过预算时按“错误 > 状态转移 > 性能样本 > 调试轨迹”顺序淘汰。诊断路径不得分配在 `Derive` 热循环中，也不得阻塞 GPU 提交。

---

## 23. 性能工程

**成本模型**：相机系统成本 ∝ `活跃视图数 × 每视图派生成本 + 活跃 rig/vcam 数`，与场景实体规模**无关**（剔除成本在 visibility，不在此）。所有性能数字都以 release 优化构建、固定 CPU 亲和性、预热后的 P50/P95/P99 报告；单次样本不作为性能结论。

关键优化：
1. **矩阵 SIMD**：`Affine3`/`Mat4` 乘、逆、frustum 提取全走 `prism_math` SIMD；逆用仿射专用快路径（非通用 4×4 求逆）。
2. **派生并行**：多视图 `ViewData` 派生彼此独立 → `prism_tasks` 分发；视图少时串行省调度。
3. **脏跳过**：相机不动 + 投影不变 + target 尺寸不变 → 跳过矩阵重算（但 jitter 相位每帧仍进，prev=本帧 unjittered 不变）；View UBO 不脏不上传。
4. **按需付费**：无导演则 Director 系统空转即退出；无 rig 组件则 Rig 系统不跑；弹簧臂球扫仅活跃相机做。
5. **历史槽紧凑**：每相机一份 prev 矩阵（64 字节），generational 回收，无哈希分配。
6. **jitter 零成本关闭**：`JitterSequence::Off` 下 jitter=0、不乘额外平移矩阵、prev=本帧，TAA 自动降级。
7. **曝光平滑 CPU 侧**：自动曝光的时间平滑在 CPU（单标量），不额外 GPU pass；GPU 只给亮度统计。
8. **大世界降精度就近做**：world→relative 的 f64 相减在提取时一次完成，GPU 收到即 f32，不在 shader 里反复高精度运算。
9. **多视图数组化**：立体/级联打包单 compute 剔除、单通道绘制，省重复提交。

**反模式规避**：不每帧堆分配 `ViewData`（池化 + 复用槽）；不每相机存全场景可见集（可见集归 visibility）；不在相机系统里做物理求交（经 trait 注入）；不把后处理算法塞进 camera crate（只存意图）。

**基准方法已经固定**：空场景、1/8/16/64 个 active view，各跑 10 万帧，前 10% 预热；分别测 `Derive`、`Director`、`Rig`、View UBO 上传和 GPU compute 剔除。报告同时给出线程数、SIMD 开关、分配次数、上传字节数和 adapter 名称。任何低于基准门槛的结果都触发对应档位降级或显式错误，不通过“增加隐形预算”掩盖。

---

## 24. 易用性与 API 人体工学

### 24.1 一行建相机

```rust
// 主相机：ECS adapter 负责实体与生命周期，核心 crate 只提供纯数据。
app.world_mut().spawn(prism_camera_ecs::CameraBundle::default());

// 带电影镜头 + 自由飞行调试
let mut intent = prism_camera::CameraIntent::default();
intent.lens = Some(Lens::cine_35mm().with_f_stop(1.8));
let entity = app.world_mut().spawn(prism_camera_ecs::CameraBundle {
    intent: prism_camera_ecs::CameraIntentComponent(intent),
    ..Default::default()
});
```

### 24.2 导演：摆 vcam，不碰主相机

```rust
let output_pose = prism_camera::CameraPose::default();
let mut brain = prism_camera_rig::CameraBrain::default();
// 跟随玩家的第三人称 vcam（纯数据候选）
let candidate = prism_camera_rig::VirtualCamera {
    key: 1,
    priority: 10,
    enabled: true,
    pose: output_pose,
    lens: None,
    weight: 1.0,
};
// 将 candidate 和其它候选传给 brain.update；输出是纯 CameraPose。
let _pose = brain.update(&[candidate], 1.0 / 60.0, output_pose);
```

### 24.3 prelude

```rust
pub mod prelude {
    pub use crate::{
        CameraIntent, CameraPose, Projection, PerspectiveProjection, OrthographicProjection,
        RenderTarget, Viewport, ClearMode, ViewData, ViewKind,
    };
    #[cfg(feature = "physical_lens")] pub use crate::{Lens, Exposure, Filmback};
    #[cfg(feature = "post")]          pub use crate::{PostStack, PostNode, Tonemapper};
}
```

### 24.4 Bevy 迁移策略

| Bevy | Prism | 说明 |
|---|---|---|
| `Camera` 巨组件 | `CameraIntent` + `ViewData`（结果） | 关注点分离；矩阵不再在意图里 |
| `Camera.target` | `RenderTarget` | 同形（窗口/纹理） |
| `Projection`（含 reversed-Z 否视版本而定） | `Projection`（强制 reversed-Z + 可无限远） | 统一精度约定 |
| `Camera.hdr` | 下游渲染目标配置 | 不属于当前 `CameraIntent` 核心字段 |
| `bevy_camera_controller`（示例级） | `prism_camera_rig` 控制器预设 | 升为一等 |
| 无（物理相机） | `Lens`/`Exposure` | 新增电影级 |
| 无（导演） | `VirtualCamera`/`CameraBrain` | 新增 Cinemachine 式 |
| 无（相机相对渲染） | `CameraIntent.relative_rendering` | 由意图字段控制 |
| `update_frusta` | `CameraSystems::Derive`（含 frustum 提取） | 合并进派生 |

当前没有 `prism_camera::compat` 或 `Camera3d` 兼容层；迁移代码应显式构造 `CameraIntent`、`CameraPose`，或使用 `prism_camera_ecs::CameraBundle`。

### 24.5 易出错点的护栏

- 手改 `ViewData`：类型上只读（无 `&mut` API 暴露给用户），编译期挡住。
- 忘了设置位姿：`CameraPose::default()` 使用原点和单位旋转，应用应在发布前写入有效姿态。
- reversed-Z 深度比较写反：下游渲染器必须使用近 1、远 0、清深度 0 和 `GREATER`；相机 crate 通过 `Projection::reverse_z()` 和 GPU flags 暴露该约定。
- 多相机同 order 同 viewport 互覆盖：debug 下警告重叠全屏相机未设 order。

---

## 25. 契约、不变量与版本化

**核心不变量**：
1. `CameraIntent`（意图）永不含矩阵/frustum；jitter 与曝光是输入意图，派生后的 jitter 与曝光标量位于 `ViewData`；`ViewData` 是每帧结果快照。
2. 剔除/拾取用去抖矩阵；光栅化用含抖矩阵；运动矢量用「去抖 vs 上一帧去抖」。三者永不混用。
3. 所有透视投影 reversed-Z；深度约定全仓一致（近 1 远 0，测试 GREATER，清 0）。
4. 使用 ECS adapter 时，`publish_camera_views` 在 `PostUpdate` 消费已准备好的 `CameraPose`；Transform 传播由上层在写入 pose 前完成。
5. `prism_camera_rig` 的 Brain 输出 `CameraPose`；它不直接写 ECS 组件或 `ViewData`，由上层把结果写回目标实体。
6. `ViewData` 注册进 `view_family::ViewRegistry`，相机写、渲染读，单向解耦。

**版本化**：View UBO 布局带 `schema_version`；新增字段向后兼容追加（不插中间）；`ViewKind`/`PostNode` 枚举用非穷尽（`#[non_exhaustive]`）以便加视图类型/后处理节点不破坏下游 match。

**契约测试**：矩阵往返（world↔view↔clip 逆一致）、frustum 提取正确性（已知相机剔除已知点）、Sunny-16 曝光锚点、jitter 均值收敛到 0、导演混合端点精确（t=0 等 from、t=1 等 to）、reversed-Z 深度单调。

---

## 26. 交付矩阵（M0–M6）与基准即规格

本文不把能力拆成开放式路线图，而用能力包、输入输出、门禁和降级规则闭合交付面：

| 能力包 | 固化内容 | 必须通过的门禁 | 运行时降级 |
|---|---|---|---|
| M0 代数核 | Perspective/Orthographic、reversed-Z、无限远、view/proj/view-proj、斜裁剪、frustum | 端点深度、FOV、逆变换、5/6 面剔除单测 | 返回结构化 `CameraError`，不产生无效矩阵 |
| M1 ViewData | 视图派生、历史槽、ViewRegistry、viewport/order、RenderTarget | 分屏/PiP/离屏、句柄分代、尺寸变化 | 核心返回 `CameraError`；ECS adapter 记录诊断并移除该实体的无效注册 |
| M2 时域 | Halton/Sobol、稳定矩阵、teleport、运动矢量输入 | jitter 均值、首帧、切镜、分辨率切换 | `Off` 时零 jitter、清历史 |
| M3 镜头与后处理 | Filmback/Lens/EV100/Auto/DOF/PostStack 归一化 | Sunny-16、CoC、曝光适应、固定物理序、节点容量 | 手动曝光、全局栈、关闭不支持节点 |
| M4 Rig 与控制器 | Body/Aim、SpringArm、批量求交、半衰期阻尼、输入归一化、LOD 调度 | 无穿透、帧率无关、非有限输入清零、跳帧间隔 | 无物理查询时直伸臂并置 `Unavailable` |
| M5 导演 | VCam、Brain、稳定选择、混合、确定性噪声、ImpulseBuffer | 端点精确、重复事件去重、容量上限、Cut 清历史 | 单一输出相机直接运行 |
| M6 大世界/XR/GPU/确定性 | 相对渲染、per-eye、late-latch、View ABI、回放 digest、外部姿态校验、遮挡滞回、前瞻预测、历史稳定化 | 1e6 m 精度、CPU/GPU parity、双眼一致、姿态拒绝、64 项容量 | 绝对 f32、双提交、预测位姿、CPU 视图、上一有效姿态 |

### 26.1 验收基线

- 单视图派生预算：桌面 release 构建 P95 ≤ 2 µs；16 视图并行 P95 ≤ 10 µs；P99 超预算时自动关闭非主视图 rig 和诊断轨迹，不改变主视图数学结果。
- 静止相机：稳定矩阵零重算、零 View UBO 上传；jitter 仅更新 jittered 字段。
- jitter：1000 帧像素均值绝对值 ≤ `1e-3`；切镜或 teleport 后第一帧 `history_valid=false`，第二帧恢复。
- 曝光：Sunny-16 得到 `EV100=15±0.01`；自动曝光适应使用指数时间常数，不能因 GPU 回读延迟产生未定义跳变。
- SpringArm：命中时相机包围体不穿透；无命中回弹函数一阶连续；无查询实现时状态明确为 `Unavailable`。
- 导演：混合 `t=0/1` 与源/目标误差 ≤ `1e-5`；四元数取最短弧；同优先级按稳定键选取。
- 大世界：相机距离原点 `1e6 m` 时，相对渲染与高精度参考的局部顶点误差 ≤ `0.1 mm`。
- GPU ABI：CPU 写入与 WGSL 读取的字段偏移、schema、flags 和矩阵列序逐项断言；不允许通过“看起来能画”替代 ABI 检查。

### 26.2 当前仓库验证记录（2026-10-15）

- `cargo test -p prism_math --lib projection -- --nocapture`：15 passed，0 failed。
- `cargo test -p prism_math_gpu --test projection_parity -- --nocapture`：在跨沙盒真实设备环境执行，3 passed，0 failed；覆盖普通透视、reversed-Z 透视和正交投影。
- 同一 GPU 测试在受限沙盒中只得到“无可用 adapter”的显式跳过；这不是成功的真实 GPU 证据，真实设备结果以上述跨沙盒运行记录为准。
- 现有 `prism_render_gpudriven::camera` 是轴对齐正交场景测试相机，不等同于本文的 `prism_camera`；本文不把它包装成完整相机内核。
- `cargo test -p prism_camera -p prism_camera_rig -p prism_camera_gpu -p prism_camera_ecs --all-targets --offline`：`prism_camera` 32 passed、`prism_camera_rig` 15 passed、`prism_camera_gpu` 7 passed、`prism_camera_ecs` 9 passed，0 failed；覆盖投影视图派生、历史失效、viewport、jitter、frustum、镜头曝光、GPU ABI、上传 ring 生命周期、后处理排序、立体视图、拾取、过场采样、外部姿态校验、遮挡滞回、前瞻预测、历史稳定化、Body/Aim、SceneQuery/TargetQuery 注入、SpringArm、噪声、Impulse、输入归一化、LOD 调度、稳定选择、导演混合、半衰期工具和真实 ECS `PostUpdate` 发布链。
- `cargo test -p prism_camera -p prism_camera_rig -p prism_camera_gpu -p prism_camera_ecs --all-features --all-targets --offline`：`prism_camera` 40 passed、`prism_camera_rig` 15 passed、`prism_camera_gpu` 7 passed、`prism_camera_ecs` 9 passed，0 failed；`cargo check -p prism_camera -p prism_camera_rig -p prism_camera_gpu -p prism_camera_ecs --offline`：通过；显式 feature 组合和 `no_std` 裁剪配置均已验证。
- 相机源码、Rig 基准和 ECS adapter 的定向 `rustfmt --check` 与相机改动的 `git diff --check`：通过；当前 workspace 全量 `cargo fmt --all -- --check` 会被其它未纳入本次相机提交的脏文件语法/格式问题阻断，不能作为相机代码失败证据。
- `cargo bench -p prism_camera_rig --bench camera_contracts -- --iterations 1`：当前独立验证得到 `derive_view=2693.325 ns/op`、`camera_brain_update=217.392 ns/op`、`impulse_apply_6dof=579.650 ns/op`、`view_lod_scheduler=32.492 ns/op`；同次 ECS 多视图基准得到 1/8/16/64 view 的 P50/P95/P99 分别为 `31531/98031/903187`、`56666/152031/172729`、`86177/333041/1150583`、`550906/1492687/1897104 ns/frame`。这些是当前机器的可复现实测，不替代跨硬件预算结论。
- `PRISM_REQUIRE_GPU=1 CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=/private/tmp/prism-camera-gpu-host cargo test -p prism_camera_gpu --test rhi_gpu --offline -- --nocapture`：真实 RHI/wgpu 设备 `1 passed、0 failed`；测试创建实际 RHI buffer，经 `RenderQueue::write_buffer` 上传 ViewBufferRing 计划，WGSL consumer 回读五组矩阵列、双精度原点拆分的 hi/lo、曝光/schema/flags/view_index、jitter、viewport 和六个 frustum 平面，最终 readback 与 CPU 预期逐项相等。
- `prism_math_gpu` 的投影 parity 仍作为投影数学的独立 GPU 证据；它与上述相机 View ABI/RHI 测试互补，不再替代相机 backend 验收。未设置 `PRISM_REQUIRE_GPU=1` 时无 adapter 的环境会显式跳过；本次跨沙盒强制设备运行已成功，因此本次相机 RHI GPU 验收不是跳过结果。

---

## 27. 责任闭合、风险处置与验收状态

### 27.1 当前实现与本文规格的关系

`pkg/prism_camera` 已提供 `CameraIntent`、`CameraPose`、`Projection`、`ViewData`、`ViewHistory`、viewport、reversed-Z/无限远投影、oblique/SubView、5/6 面 frustum、Halton/Sobol jitter、Filmback/Lens/Exposure、PostStack、viewport/world picking、StereoPair/late-latch、相对渲染原点拆分、GPU View ABI、`ViewRegistry` 发布桥接和 fixed-tick `AdvancedRuntime`；advanced 契约提供 CameraTrack、自动对焦、快门、foveation、批量 key、LOD、构图、确定性 digest、ExternalPoseValidator、OcclusionFadeState、LookaheadPredictor 和 HistoryStabilizer。`pkg/prism_camera_rig` 已提供 `VirtualCamera`、`CameraBrain`、稳定优先级选择、位姿混合、Body/Aim、`SceneQuery`/`TargetQuery` 批量弹簧臂、半衰期阻尼、Fly/Orbit/Pan 控制器、确定性梯度噪声、ImpulseBuffer、输入归一化和 ViewUpdateScheduler；`pkg/prism_camera_ecs` 已提供实体组件、统一帧时钟、稳定句柄映射、诊断资源、SceneQuery/TargetQuery 注入和 `PostUpdate` 发布系统；`pkg/prism_camera_gpu` 已提供有界 ViewBufferRing、增量上传、清零、提交/取消/回收、stale token/schema 校验，以及通过 `RenderDevice`/`RenderQueue` 写入真实 RHI buffer 的 `RhiViewBuffer`/`RhiUploadBackend` 和可供 WGSL backend 验证的 ViewRecord consumer。

实现采用四层所有权：相机 crate 只计算不可变视图快照，rig crate 只生成位姿结果，ECS adapter 只负责生命周期和发布，GPU crate 只负责 ABI 消费侧的有界上传计划；渲染资源、物理求交和 Transform 存储继续由各自 crate 持有。四个相机 crate 均禁止 unsafe；核心数学、rig 和 GPU ring 路径保持 `no_std + alloc`，ECS adapter 使用应用/ECS crate。已核验提交包括 `cfb34909f`（ECS 发布链）、`bf53ef12b`（裁剪配置测试修正）、`6dd2e3929`（ABI/ECS 加固）、`e4cf94581`（GPU consumer）、`e522e7c57`（fixed-tick advanced runtime）、`b6fc25e72`（Body/Aim 高级 Rig）、`a2cf97f5d`（workspace/ECS/query 接线）和 `31e767af7`（最小 feature 编译修复）。

本文中涉及的渲染后处理算法、物理后端、XR 设备驱动和具体渲染 backend 由对应所有者消费这些稳定契约；相机 ECS 内的 Director/Control/Rig/Publication 已提供可运行接线，它们不复制相机矩阵或历史真相。相机核心的交付验收以四个 crate 的 API、单元测试、feature check、ABI/upload plan 检查和可获得的 GPU parity 证据为准；跨模块能力沿同一契约接入，不改变相机数学与历史协议。

### 27.2 风险处置表

| 风险 | 固定处置 | 失败表现 | 恢复方式 |
|---|---|---|---|
| 物理求交不可用 | `SceneQuery` 批量 trait + 查询次数上限 | `SpringArmStatus::Unavailable` | 直伸臂，恢复查询后重新计算 |
| GPU 自动曝光延迟 | 非确定模式使用滞后统计；确定模式使用 CPU/固定 EV | 曝光适应滞后一帧 | 锁定上一有效 EV，不污染几何视图 |
| Brain 与手动控制冲突 | Brain 拥有输出相机写权限，控制器只写 VCam | 诊断 `DirectWriteSuppressed` | 移除 Brain 或改写目标 VCam |
| reversed-Z 接入错误 | 深度策略、清除值和比较器集中封装 | 深度全黑/全遮挡 | 绑定校验失败，拒绝提交该 view |
| 相对渲染协作错误 | `world_origin_hi/lo` + flags + schema 校验 | 大世界抖动/天空错位 | 临时切回绝对模式并记录 `RelativeSpaceMismatch` |
| late-latch 不可用 | 预测位姿固定提交 | 延迟增加但结果一致 | 关闭 relatch，不改变历史协议 |
| 自定义后处理破坏物理序 | Raw 节点必须声明 domain/格式/预算 | 节点被跳过 | 使用上一帧已归一化栈 |

这些处置项都是运行时可执行的契约，不依赖人工补丁。核心错误关联字段和值；ECS adapter 额外关联实体，帧号由 `CameraFrame` 提供；诊断关闭时仍可由上层保留错误计数。

### 27.3 维护与扩展规则

1. 新投影类型必须实现 `Projection` 的验证、CPU 参考、端点单测、frustum 规则和 GPU parity；缺一项不得进入公共 prelude。
2. 新 `ViewKind` 只能追加枚举值，必须给出重要度、清屏、历史和后处理策略；不能在下游复制相机判断。
3. 新后处理节点必须声明输入/输出色域、HDR/LDR、前后放大域、临时纹理预算和可跳过行为。
4. 新 Rig 必须只读目标快照并返回纯数据结果；不能直接写 ViewData、RenderTarget 或 GPU 资源。
5. 新 GPU 字段只能追加到 schema 尾部并提升版本；旧版本读取失败时拒绝绑定，不按猜测偏移读取。
6. 所有热路径采用池化、批量和稳定排序；任何哈希分配、跨线程锁和 GPU 回读必须显式标注在性能诊断中。

---

## 28. AAA 高级功能完整契约

以下能力均已给出默认值、输入输出、资源预算和失效行为；它们是可编译 feature，而不是没有规则的概念列表。

### 28.1 相机动画片段与过场（`cutscene`）

`prism_asset` 保存版本化轨迹：时间、DAffine3、焦距、曝光、PostStack 权重和事件键。运行时以只读 VCam 播放，时间由固定 tick 驱动，曲线采样使用确定性 Hermite；轨迹结束按 `CutsceneEndPolicy::{Hold,BlendToPrevious,Deactivate}` 处理，采样状态显式返回 `active` 与 `handoff_to_previous`，不会把结束状态隐式当作 Hold。资产校验失败时拒绝播放并保持当前 VCam，事件按帧号去重。

### 28.2 焦点追踪与自动对焦（`autofocus`）

焦点源只能是锁定实体、屏幕中心射线命中或固定距离三者之一；采样距离先限幅到 `[near+epsilon, focus_max]`，再用独立的拉焦半衰期平滑。focus breathing 只改变投影 FOV 的校正项，不修改物理焦距真值；无命中保持上一焦距并输出 `FocusLost`。

### 28.3 相机碰撞体与穿帮消隐（`occlusion_fade`）

遮挡查询返回实体分代 ID、遮挡深度和屏幕覆盖率；消隐系统只写 `CameraOcclusionIntent`，材质系统决定 dither 实现。实体进入/退出使用双阈值滞回，最多同时处理 64 个实体；GPU id-buffer 不可用时退化为 CPU 查询，不改变相机矩阵。

### 28.4 可变光圈/快门与运动模糊一致性（`shutter`）

快门角转换为曝光时间 `shutter_s = shutter_angle / (360 * fps)`；曝光模型、DoF 光圈和运动模糊速度使用同一个快门快照。fps 为零或时间步无效时使用固定 1/60 s 并记录 `InvalidFrameRate`，避免除零和跨帧漂移。

### 28.5 多视图共享剔除与几何复用（`view_batch`）

先对视图 frustum 做并集粗剔，再按视图细分；共享索引只在同一 RenderTarget、同一相对原点、同一材质/LOD 策略下复用。共享缓存以 `(frame, view_family_generation, cull_policy_hash)` 标识，任何成员变化都会失效整组，保证不复用过期可见集。

### 28.6 预测式流送与相机前瞻（`lookahead_stream`）

使用当前速度、加速度上限和下一个 VCam 快照生成最多 500 ms 的预测轨迹；沿轨迹采样并累积 streaming importance，但不改变当前渲染 frustum。预测失败、速度 NaN 或目标销毁时清除预测贡献，恢复当前视图权重。

### 28.7 相机空间抗锯齿与边缘稳定（`edge_stabilize`）

根据角速度、线速度、历史有效标志选择历史裁剪阈值；裁剪只作用于 temporal history，不修改真实相机矩阵。速度超过阈值时优先丢弃局部历史，再启用受限锐化；节点必须报告历史接受率和丢弃原因。

### 28.8 虚拟制片 / 外部相机输入（`virtual_production`）

外部设备只写带时间戳的 `ExternalPoseSample`；校准矩阵、坐标系和时间偏移在输入层完成，Camera 只消费已校验样本。样本时间倒退、姿态跳变超过阈值或校准版本不匹配时拒绝该样本，使用上一有效样本并置 `ExternalPoseRejected`；不包含设备驱动代码。

### 28.9 相机 LOD 与多分辨率视图（`view_lod`）

每个次要视图声明最大更新频率、最小分辨率和历史保持帧数。跳帧期间保留上一 ViewData，必要时只外推相对位姿；主视图、XR 双眼和反射关键视图禁止跳帧。预算调度按重要度、屏幕面积和最近更新时间排序，资源不足时先降低离屏分辨率再降低更新频率。

### 28.10 构图辅助与安全框（`composition_aid`，editor）

三分法、黄金比、中心十字、标题/动作安全框、焦平面和 Composer zone 都作为只读 gizmo 从 ViewData 计算；编辑器关闭时不进入运行时包。gizmo 使用去抖矩阵，避免随 TAA jitter 抖动；坐标转换失败时隐藏该 gizmo 并保留错误计数。

---

> 小结：`prism_camera` 以「意图 vs 结果」分离为骨架，把 AAA 相机的四大难点——**时域正确（jitter/历史/运动矢量）、大世界精度（相机相对渲染）、电影级手感（导演混合 + 弹簧臂避障 + 物理镜头曝光）、多视图共享（view_family 聚合预算）**——分别落在清晰的契约职责中，核心 `no_std`、可逐 feature 裁剪、下游经 `ViewRegistry` 单向解耦。它取代 `bevy_camera` 的「巨组件 + 示例级控制器」，升格为一等的取景与导演内核。
