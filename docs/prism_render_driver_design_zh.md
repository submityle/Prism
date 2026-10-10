# Prism Render Driver 顶级次世代 AAA 级图形硬件抽象层（RHI）设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **RHI（Render Hardware Interface，图形硬件接口）**：把 Vulkan / D3D12 / Metal / WebGPU 等现代显式图形 API 的差异收敛到一层统一、显式、低开销的门面。它是整个渲染体系（render_graph / material / shader / scene）唯一踩在 GPU 上的地面，让渲染代码写一次、跨 Windows / macOS / Linux / 移动 / 主机 / Web 运行。
> 借形态不抄码。借鉴：
> - **现代跨平台抽象**：wgpu / WebGPU（可移植显式 API 范型）、The-Forge（AAA 多后端 RHI）、NVRHI（NVIDIA 显式 RHI）、Diligent Engine、sokol_gfx（极简）、bgfx（广兼容）
> - **引擎级 RHI**：Unreal `FDynamicRHI` / `FRHICommandList` / `RDG`、Unity SRP `CommandBuffer` / `GraphicsBuffer`、寒霜 / id Tech 的显式后端抽象
> - **底层显式 API**：Vulkan、D3D12、Metal、WebGPU（三/四大后端的屏障/描述符/队列/PSO 模型）
> 本文为纯经典图形系统编程路线，**不含任何 AI/ML 内容**（不涉及 DLSS/XeSS 等 ML 超分；仅提供其所需的 RHI 原语）。

- 版本: v0.5（**编码阶段进行中**：阶段一「共享层 + 阶段1 门面」已实现、测试并提交；高级功能与原生后端仍为 PLANNED）
  - v0.1→v0.2：新增第 24 章「AAA 高级功能增补·深化」（GPU-Driven 完全体 / 多 GPU / 设备丢失恢复 / 稀疏常驻 / GPU 工作图与时间线 / 高级光栅 / 管线库与着色器 ABI）。
  - v0.2→v0.3（本次重构升级）：① 路线图改为「wgpu 先行 · 原生按需扩展（可能永不触发）」并全文对齐里程碑口径；② Web 结论定稿为「wgpu/wasm 直通浏览器 WebGPU，不写 web-sys 后端、不做 WebGL 回退」；③ 门面策略定稿为「借 WebGPU 对象模型 · 拒绝其隐式语义」，删除 `compat-wgpu` 垫片概念；④ 第 1 章新增「七条不可妥协设计原则」；⑤ 第 2 章补入 gfx-hal 失败教训等反面教材与关键取舍；⑥ 第 18 章细化极致性能工程与性能反模式；⑦ 新增第 25 章「错误与结果模型」、第 26 章「线程模型与 Send/Sync 契约」、第 27 章「测试、一致性与黄金图像」。均为 PLANNED，无代码。
  - v0.3→v0.4（本次深化升级）：① 第 4 章补入**门面/后端边界的核心 API 速写**（`RenderBackend`/`Device`/句柄 trait 示意），落地「加后端=实现一组 trait」；② 第 5 章深化 Adapter 选择策略与设备特性协商；③ 第 7 章补入**统一绑定模型跨后端映射表**与 bindless 三档；④ 第 10 章补入**自动屏障状态求解模型**与拆分屏障示意；⑤ 第 14 章深化分配器策略（TLSF/buddy、专用分配、defrag、residency）；⑥ 新增第 28 章「可扩展性与稳定性契约」（加后端/加能力/trait 版本化）、第 29 章「性能 KPI 目标（量化基准）」。API 速写均为**示意、非最终**，仍无代码。
  - v0.4→v0.5（**进入编码**）：`prism_render_driver` 的**完整共享层**（同步状态/资源跟踪 tracker、slotmap 句柄表、PSO/采样器/布局缓存、TLSF/buddy/ring 子分配器、延迟回收、显存预算）与**阶段1 门面骨架**已落地为真实代码——6222 LOC、78 单元测试、`cargo clippy`/`fmt` 零告警，已提交（commit `be94783`）。本节以上 API 速写部分从「示意」转为「与已提交代码对齐」；§24 高级功能、原生 Vulkan/D3D12/Metal 后端仍为 PLANNED（阶段二按需，§22）。
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: `prism_platform`（窗口 surface/动态库/虚存/GPU 可见内存）、`prism_math`（矩阵/向量/打包）、`prism_utils`（句柄表/竞技场/位集）、`prism_diagnostic`（GPU 计时/调试标记）；底层后端经典 crate `ash`(Vulkan)/`windows`(D3D12)/`metal`(Metal)/`wgpu`(可选包装后端)
- 层级定位: L4 表现层根（被 `prism_render_graph` / `prism_shader` / `prism_material` / `prism_render_scene` / `prism_ui` 直接依赖；向下只依赖 L1 地基）
- 明确约束: 显式、无全局状态、线程安全命令记录；核心跨后端语义统一，后端差异锁在后端模块；`std` 必需（GPU 驱动交互）；**不依赖任何 `bevy_*` crate**（含 `bevy_render`/`wgpu` 仅作为可选后端被门面隔离）

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 档位化（capability / tier / feature）
4. 分层架构（RHI 门面 + 多后端）
5. 核心模型：Instance / Adapter / Device / Queue / Surface
6. 资源模型：Buffer / Texture / Sampler / 视图
7. 绑定模型：BindGroup / DescriptorSet / 根签名
8. 管线与着色器（PSO / ShaderModule / 管线缓存）
9. 命令记录与提交（CommandEncoder / 多线程录制）
10. 同步模型：资源状态 / Barrier / Fence / Semaphore
11. 交换链与呈现（Surface / Swapchain / 帧节奏）
12. 渲染通道与附件（RenderPass / 动态渲染）
13. 多队列：异步计算与传输
14. 显存管理与子分配（接 platform / utils）
15. 后端矩阵（Vulkan / D3D12 / Metal / WebGPU / Null）
16. 与 render_graph / shader / asset / diagnostic 集成
17. 高级功能增补（AAA）
18. 性能工程
19. 易用性与 Bevy / wgpu 迁移策略
20. crate 分层与模块布局
21. 契约、不变量与版本化
22. 路线图（wgpu 先行 · 原生按需扩展）与基准即规格
23. 诚实边界与风险
24. AAA 高级功能增补·深化（v0.2）
25. 错误与结果模型（Result 边界 · 创建期前移）
26. 线程模型与 Send / Sync 契约
27. 测试、一致性与黄金图像（质量门禁）
28. 可扩展性与稳定性契约（加后端 / 加能力 / trait 版本化）
29. 性能 KPI 目标（量化基准）

---

## 1. 设计哲学与目标

现代图形 API（Vulkan/D3D12/Metal/WebGPU）是**显式**的：资源状态、内存、同步、描述符全要手动管，换来极致性能与可预测性，代价是极度复杂且四家互不兼容。如果渲染器直接写某一家 API，就被钉死在一个平台；如果到处 `#[cfg]`，代码会碎成四份。`prism_render_driver` 的使命是：**把四大显式 API 的共性抽象成一套统一、显式、低开销的门面**，对上给渲染图/材质/着色器一套干净接口，对下为每家 API 写一个后端，让「换 GPU 平台 = 换后端」而非「重写渲染器」。

**一句话定位**：`prism_render_driver` 是 Prism 的「GPU 地面」——Device/Queue/资源/绑定/管线/命令/同步/交换链全部一个显式门面多后端；渲染器永不直接碰 Vulkan 或 Metal，只调 RHI；语义向 WebGPU/现代显式 API 看齐，保留手动屏障与多线程录制以求 AAA 级性能。

四条总目标（按权重）：

1. **性能**：显式低开销（薄门面透传）；多线程命令录制；手动/自动屏障最小化；PSO/描述符缓存；多队列重叠（图形/计算/传输）；GPU 驱动内存子分配。
2. **效果（能力）**：覆盖现代渲染全需求——compute、indirect/multi-draw、bindless、mesh shader、光追（可选）、时间戳查询、多视图（VR）。
3. **可移植**：Vulkan/D3D12/Metal/WebGPU 一套上层代码，差异锁后端；新增后端 = 实现一组 trait。
4. **易用 + 可测**：门面向 WebGPU 直觉看齐（比裸 Vulkan 友好得多）；提供 Null/软件后端做无 GPU 单测与 CI；能力探测让上层优雅降级。

**七条不可妥协的设计原则**（贯穿全文，评审任何 API 以此为尺）：

1. **显式优先、自动兜底**：默认路径自动正确（屏障/状态/回收全自动），但每个自动机制都留「手动逃生舱」，热路径可完全接管——不是「要么全自动要么全手动」，而是「自动是默认，手动随时可降维介入」。
2. **零成本抽象（zero-cost）**：门面是编译期薄层，`#[inline]` 直落后端，热路径无 vtable、无隐藏分配、无运行时反射；不用的能力不付代价（feature + `GpuCaps` 双门控）。
3. **错误前移**：能在创建期/录制期报的错绝不留到提交期/运行期（PSO 格式不符、绑定布局不匹配一律 `Result` 返回而非驱动崩，见 §25）。
4. **句柄而非指针**：所有 GPU 对象 generational 句柄化，杜绝悬垂与跨后端指针泄漏；底层原生对象后端私有（见 §21）。
5. **后端差异锁后端**：上层代码零 `#[cfg]`；坐标系/格式/描述符/队列族差异全部在后端模块内归一（见 §23）。
6. **可测在先**：Null 后端 + 黄金图像对拍 + 校验层是一等公民而非事后补丁，无 GPU 也能跑逻辑单测（见 §27）。
7. **按需演进、拒绝投机（YAGNI）**：先用 wgpu 单后端把渲染体系立起来，原生后端/高级能力「撞墙才做」；每个特性由真实消费者（render_graph/material）拉动，不预先投机实现（见 §22）。

---

## 2. 参考产品取舍

| 来源 | 借鉴什么 | Prism 取舍 |
|---|---|---|
| WebGPU / wgpu | 可移植显式 API 的「恰到好处」抽象：BindGroup/PSO/Encoder/队列模型 | 门面语义以此为基线；但补齐 bindless、手动屏障、多队列等 AAA 能力 |
| The-Forge | AAA 多后端 RHI 的工程组织、descriptor/root signature 抽象 | 借多后端分层与资源绑定模型 |
| NVRHI / Diligent | 显式 RHI 的状态追踪、自动 vs 手动屏障双模 | 采「可自动可手动」的屏障策略 |
| Unreal `FRHICommandList`/RDG | 命令列表 + 渲染依赖图分离、并行录制 | RHI 只做命令；依赖图归 `prism_render_graph`（见 §16） |
| Vulkan | 屏障/队列族/描述符/内存类型的最显式模型 | 作为能力上界与首要后端 |
| D3D12 | root signature、resource state、fence 模型 | 后端映射参考 |
| Metal | argument buffer（bindless 友好）、heap、命令队列 | 后端映射参考（Apple 平台首选） |
| sokol_gfx / bgfx | 极简门面与广兼容的降级思路 | 借「能力位 + 降级」理念，不绑其抽象层级 |
| VMA（Vulkan Memory Allocator） | GPU 显存子分配策略 | 借形态做 §14 显存分配器（接 platform/utils） |
| **gfx-hal（已废弃，反面教材）** | Rust 社区「通用 Vulkan 式全量 HAL」的尝试与**失败教训** | **引以为戒**：过度贴 Vulkan 全量语义、维护成本压垮项目，最终整个 Rust 图形生态让位给 wgpu——印证本文「借对象模型 + wgpu 先行、原生按需」而非再造一个全量 HAL |
| rafx（Rust 引擎 RHI） | api/framework/assets 三层分层、着色器反射驱动绑定与资源生命周期 | 借反射驱动 `BindGroupLayout` 自动生成（§7/§16）与分层职责切分 |
| Granite（Themaister，开源 Vulkan 引擎） | 「自动屏障 + render graph 自动瞬态别名」的工业级落地 | 借自动屏障 + 瞬态别名落地经验（§10/§14 与 `prism_render_graph`） |
| id Tech / 寒霜 FrameGraph | FrameGraph 概念起源、瞬态资源与自动屏障推导 | 印证「RHI 只给原语、帧编排归图」的分层（§16） |
| NRI（NVIDIA）/ RPS（AMD）| 现代多后端薄 RHI 与 render-pipeline-shaders 的职责边界 | 佐证「薄门面 + 外置依赖图」的切分正确性 |

**关键取舍一句话**：**借 WebGPU 的对象模型（已被验证可映射三大原生 API），但不接管渲染编排、不绑最低公分母语义，也不重走 gfx-hal 的全量 HAL 老路**；wgpu 先行把体系立起来，原生后端撞墙才补。

**不做**：不做高层渲染逻辑（PBR/光照/后处理归 `prism_render_*`）；不做帧依赖编排（归 `prism_render_graph`）；不做着色器编译（归 `prism_shader`，RHI 只吃编译好的字节码/模块）；不做窗口创建（归 `prism_window`，RHI 只接 surface 句柄）。

---

## 3. 档位化（capability / tier / feature）

- **能力位 `GpuCaps`**：运行时探测设备支持——bindless/descriptor indexing、mesh shader、光追（ray query/pipeline）、可变速率着色 VRS、timestamp query、多视图、indirect count、64-bit 原子、UMA（统一内存）、异步计算队列数等；上层据此选渲染路径或降级。
- **特性档 `FeatureTier`**：`baseline`（WebGPU 级最低集，保证全平台）→ `desktop`（桌面显式后端全能力）→ `cutting_edge`（mesh shader/光追/bindless 全开）。渲染器按档选管线。
- **编译期 feature**：`backend-wgpu`(首发唯一后端，wasm 上直通浏览器 WebGPU) / `backend-null`(测试) / `backend-vulkan` / `backend-d3d12` / `backend-metal`(后三者为阶段二按需扩展，见 §22)、`raytracing`、`mesh-shader`、`bindless`、`validation`（调试层）、`gpu-timing`（接 diagnostic）。
- **原则**：`baseline` 档保证 WebGPU 能表达的一切跨所有后端可用；AAA 高级能力（§17）全部 `GpuCaps` 门控，缺失即降级，绝不假设存在。

---

## 4. 分层架构（RHI 门面 + 多后端）

```
     render_graph / shader / material / render_scene / ui （上层渲染）
                              │  仅依赖 RHI trait + 句柄
     ┌────────────────────────┴────────────────────────┐
     │                prism_render_driver 门面层          │
     │  Device/Queue/Buffer/Texture/BindGroup/Pipeline/  │
     │  Encoder/Barrier/Fence/Swapchain  (全部句柄化)     │
     └────────────────────────┬────────────────────────┘
       ┌──────────┬───────────┼───────────┬──────────┐
   backend-wgpu   backend-null  backend-vulkan backend-d3d12 backend-metal
   (wgpu;首发)    (测试/离屏)   (ash)         (windows)    (metal)
   └ 桌面/移动/Web 全跑它 ┘   └──── 阶段二按需扩展（撞墙才做，见 §22）────┘
          └──────── 共享：显存分配器 / 状态追踪 / 缓存 ────────┘
```

- **门面层**：全部资源**句柄化**（generational handle，接 `prism_utils` slotmap），不暴露后端原生对象指针；命令通过 `CommandEncoder` 录制。
- **后端层**：每 API 一个模块，实现门面 trait；原生对象存后端侧资源表，句柄→原生对象映射。
- **共享层**：显存子分配、资源状态追踪、PSO/描述符缓存等后端无关逻辑复用。
- **Null 后端**：记录调用但不触 GPU，供上层逻辑单测与 CI（无 GPU 环境）。

**门面/后端边界速写**（示意，非最终 API）——上层只见 trait + 句柄，永不见原生对象；加新后端 = 实现这组 trait，上层零改动（见 §28）：

```rust
// 句柄：generational，Copy + Send + Sync，后端私有原生对象藏在后端资源表里
pub struct BufferHandle(RawHandle);
pub struct TextureHandle(RawHandle);
pub struct PipelineHandle(RawHandle);
// BindGroupHandle / SamplerHandle / BindGroupLayoutHandle ...

/// 每个后端（wgpu/vulkan/metal/d3d12/null）实现这一组 trait
pub trait RenderBackend: Send + Sync {
    type Device: Device;
    fn enumerate_adapters(&self) -> Vec<AdapterInfo>;              // §5
    fn create_device(&self, adapter: AdapterId, req: &RequestedCaps)
        -> Result<Self::Device, RhiError>;                        // 特性协商，§5/§25
}

pub trait Device: Send + Sync {                                   // §26 线程契约
    fn caps(&self) -> &GpuCaps;                                   // §3 能力门控
    fn create_buffer(&self, d: &BufferDesc)   -> Result<BufferHandle,   RhiError>;
    fn create_texture(&self, d: &TextureDesc) -> Result<TextureHandle,  RhiError>;
    fn create_pipeline(&self, d: &PipelineDesc)-> Result<PipelineHandle, RhiError>; // 错误前移 §8/§25
    fn create_encoder(&self, q: QueueKind) -> CommandEncoder;     // Send + !Sync，§9/§26
    fn destroy(&self, h: AnyHandle);                              // 延迟回收，非立即 free，§10/§14
}

pub trait CommandEncoder: Send {                                 // 单线程录制，§9
    fn barrier(&mut self, b: &[Barrier]);        // 手动逃生舱；默认由 render_graph 自动插，§10
    fn draw_indirect_count(&mut self, /* ... */);// GPU-driven，§17
    fn finish(self) -> CommandBuffer;            // Send，回提交线程排序 submit
}
```

---

## 5. 核心模型：Instance / Adapter / Device / Queue / Surface

- **`Instance`**：RHI 入口，选后端 + 开调试层；枚举 `Adapter`。
- **`Adapter`**：物理 GPU，暴露 `GpuCaps` + 内存信息 + 队列族；供多 GPU 选择（独显/核显）。
- **`Device`**：逻辑设备，资源创建的根；线程安全（可多线程创建资源与录制命令）。
- **`Queue`**：提交命令的通道，分 `Graphics` / `Compute` / `Transfer` 能力位；多队列见 §13。
- **`Surface`**：由 `prism_window` 提供原生窗口句柄创建（见 §11），RHI 不碰窗口系统本身。
- **Adapter 选择策略**：默认按「可用显存 + 能力档 + 是否独显 + 是否可呈现到目标 surface」加权打分自动选最优；可被用户显式指定（强制独显/核显/指定 PCI id，供多 GPU 工作站与笔记本双显卡）。打分策略可被上层覆盖（接 §28 可扩展点）。
- **设备特性协商（negotiation）**：`create_device` 吃 `RequestedCaps`（required / preferred 两级）——required 缺失即 `Err(Unsupported)` 启动期失败（而非运行期崩）；preferred 缺失则静默降级并在 `GpuCaps` 如实反映。杜绝「假设能力存在」（原则 2/7）。
- **`GpuCaps` 是只读事实源**：设备建成后 `caps()` 冻结，上层所有高级路径分支以它为唯一依据（§3）；绝不运行期试探性调用后 catch 错误。
- **初始化契约**：`prism_app` 建 `prism_platform` → `prism_window` 建 surface → RHI `Instance`→选 `Adapter`→协商 `Device`→`Swapchain`，顺序固定；每步失败返回 `RhiError`（§25），仅「无任何可用 Adapter」panic。

---

## 6. 资源模型：Buffer / Texture / Sampler / 视图

- **Buffer**：usage 位（vertex/index/uniform/storage/indirect/copy）+ 内存域（device-local/host-visible/upload/readback，接 platform §24.2 GPU 共享内存）。
- **Texture**：维度/格式/mip/array/采样数；格式能力按 `GpuCaps` 探测（并非所有格式处处可用）。
- **TextureView / BufferView**：子资源视图（mip 范围、array 层、格式重解释），绑定与附件用视图而非原始资源。
- **Sampler**：过滤/寻址/各向异性/比较采样；各向异性上限按能力。
- **句柄化**：所有资源返回 `BufferHandle`/`TextureHandle`（generational，接 utils），释放走延迟回收（见 §10/§14），杜绝 GPU 仍在用时 CPU 释放。
- **格式门面**：统一 `Format` 枚举，映射各后端格式；不支持格式在创建期即报错（而非运行期崩）。

---

## 7. 绑定模型：BindGroup / DescriptorSet / 根签名

- **BindGroup / BindGroupLayout**（WebGPU 形态）：把资源成组绑定，布局与数据分离，复用布局降开销。
- **根签名 / PipelineLayout**：声明管线可见的 BindGroup 槽位 + push/root constants。
- **Bindless（`bindless` 档）**：大描述符数组 + 运行期索引（Vulkan descriptor indexing / D3D12 unbounded / Metal argument buffer），供海量材质/纹理一次绑定、draw 内按索引取，大幅降绑定开销——现代 AAA 的关键能力。
- **Push/Root Constants**：小而高频的参数走常量，免 buffer 更新开销。
- **动态偏移**：uniform buffer 动态偏移，一个 buffer 服务多 draw。
- **缓存**：BindGroupLayout/PipelineLayout 内容寻址缓存（接 utils §24.6），相同布局只建一次。

**统一绑定模型 → 各后端的落地映射**（门面概念一套，后端差异锁在模块内，上层零感知）：

| 门面概念 | WebGPU/wgpu | Vulkan | D3D12 | Metal |
|---|---|---|---|---|
| `BindGroupLayout` | `BindGroupLayout` | `VkDescriptorSetLayout` | 描述符表 range（root signature 片段） | argument buffer 布局 |
| `BindGroup`（一组资源） | `BindGroup` | `VkDescriptorSet` | 描述符堆内连续区间 + 表指针 | `MTLArgumentBuffer` 实例 |
| `PipelineLayout`/根签名 | `PipelineLayout` | `VkPipelineLayout` + push const | `ID3D12RootSignature` | `MTLArgumentEncoder` 组合 |
| Push/Root Constants | 无（用 small uniform） | push constants | root constants | `setBytes`（内联常量） |
| 动态偏移 | dynamic offset | `dynamicOffset` | root CBV / 偏移 | buffer offset |
| Bindless 数组 | 有限（`binding_array`） | descriptor indexing | unbounded table | argument buffer + 资源堆 |

> 映射策略：门面只定义「布局 + 组 + 偏移」三件事；D3D12 的 root signature 由门面从 `PipelineLayout` **自动推导并缓存**（descriptor table vs root descriptor 的权衡由后端启发式决定，热点 BindGroup 优先放 root），Metal 的 argument buffer encoder 同理按布局内容寻址缓存。后端差异不上浮到上层 API。

**Bindless 三档（`GpuCaps` 门控，缺档逐级降级，接 §3）**：
- **Tier 0 · 无 bindless**：传统 per-draw 绑定（WebGPU 下限 / 老移动）。海量材质走「排序 + 批次切换」降绑定次数。
- **Tier 1 · 部分 bindless**：有限大小的纹理/采样器数组（WebGPU `binding_array`、部分移动）。够做纹理图集式索引，不够全场景一次绑定。
- **Tier 2 · 完全 bindless**：unbounded 描述符数组 + 运行期索引（Vulkan descriptor indexing / D3D12 unbounded / Metal argument buffer）。全场景材质/纹理一次绑定、draw 内按索引取，是 GPU-Driven（§17）与 indirect 海量绘制的前提。

---

## 8. 管线与着色器（PSO / ShaderModule / 管线缓存）

- **ShaderModule**：吃 `prism_shader` 产出的后端字节码（SPIR-V / DXIL / MSL / WGSL，见 §16），RHI 不编译着色器源码。
- **GraphicsPipeline（PSO）**：固化状态（着色器+顶点布局+光栅+混合+深度+附件格式）为不可变对象，对标 Vulkan/D3D12 PSO，消除 draw 时状态切换开销。
- **ComputePipeline**：计算着色器 PSO。
- **管线缓存 PipelineCache**：PSO 编译结果磁盘持久化（接 platform §6 文件），二次启动免重编，缩短加载；内容寻址 key（着色器哈希+状态哈希）。
- **异步 PSO 编译**：PSO 编译投递到 `prism_tasks`（后台车道），避免卡顿；未就绪时用占位/跳过，防首帧卡死（PSO stutter 是 AAA 顽疾，见 §17）。
- **特化常量**：编译期常量特化 PSO 变体（接 shader 变体系统）。

---

## 9. 命令记录与提交（CommandEncoder / 多线程录制）

- **CommandEncoder**：录制命令到命令缓冲；设计为**可多线程并行录制**（每线程一个 encoder，接 `prism_tasks` worker），录完合并提交——AAA 并行渲染的核心。
- **二级命令缓冲 / Bundle**：预录可复用命令序列（WebGPU render bundle / Vulkan secondary / D3D12 bundle），静态物体一次录制多帧复用。
- **Draw/Dispatch**：draw、draw_indexed、draw_indirect、multi_draw_indirect(_count)、dispatch、dispatch_indirect。
- **拷贝命令**：buffer↔buffer、buffer↔texture、texture↔texture、mip 生成。
- **调试标记**：push/pop debug group + 插入 marker（接 `prism_diagnostic` §24.2 GPU 时间线、RenderDoc/PIX）。
- **提交**：encoder → CommandBuffer → `Queue::submit`，返回可等待的提交标记（接 §10 fence）。

---

## 10. 同步模型：资源状态 / Barrier / Fence / Semaphore

显式 API 最难的部分，RHI 的核心价值所在：

- **资源状态转换 Barrier（一句话）**：GPU 侧的「交通信号」——资源在不同用途间（渲染目标↔采样↔拷贝源）切换时，既做状态/布局转换，又下内存可见性+执行屏障，保证「上一步写完、缓存刷新」后下一步才读，否则乱序并行会踩踏出随机闪烁/损坏/驱动崩。
- **双模策略**：
  - **自动屏障**：RHI 跟踪资源状态，`render_graph` 不写屏障，由 RHI/图自动插入（易用，默认）。
  - **手动屏障**：高级用户显式下屏障，极致控制（AAA 热路径）。
- **Fence（CPU-GPU 同步）**：CPU 等 GPU 完成（帧资源回收、readback），接帧飞行（frames-in-flight）。
- **Semaphore（GPU-GPU / 队列间）**：跨队列（图形↔计算↔present）同步，供异步计算重叠（见 §13）。
- **延迟资源回收**：资源释放挂到「N 帧后」安全点，GPU 用完才真正释放（接 §14），杜绝 use-after-free。
- **契约**：屏障错误是 GPU 渲染最隐蔽的 bug（表现为随机闪烁/损坏/驱动崩），`validation` 档必须开校验层交叉验证。

**自动屏障状态求解模型**（默认路径的核心，借 Granite / FrameGraph 工业经验，§2/§16）：
- **每资源跟踪当前状态**：RHI 为每个 buffer/texture（可细到 subresource：mip × array slice）维护「当前访问状态」= 用途（render-target / sampled / storage / copy-src/dst / present）+ 队列归属 + 可见性阶段。
- **声明式读写 → 自动推导转换**：`render_graph` 的每个 pass 只声明「我读谁、写谁、作什么用」；求解器比对资源「期望状态 vs 当前状态」，差异处自动生成最小屏障集合，并把访问阶段精确到 pipeline stage（避免过宽屏障拖慢）。
- **屏障去重与合批**：同一提交内对同一资源的多次转换合并，相邻 pass 的屏障聚成一次 `pipelineBarrier`（减少 GPU 停顿点）；只读→只读不插屏障。
- **跨队列所有权转移**：资源在图形↔计算↔传输队列间迁移时，自动插 release/acquire 对（Vulkan queue family ownership）+ timeline semaphore 依赖（§13/§24.5）。
- **瞬态资源别名屏障**：接 §14 placed resources——别名复用同块显存的瞬态资源，在复用边界自动插 aliasing barrier，防前一资源的残留访问踩踏后一资源。

**拆分屏障 split barrier（极致优化逃生舱，示意·非最终 API）**——把屏障的「开始」与「结束」拆开，让 GPU 在等待期间继续干无关活，填满气泡：

```rust
// 一次性屏障：阻塞到转换完成（默认、易用）
encoder.barrier(&[Barrier::texture(tex, State::RenderTarget, State::Sampled)]);

// 拆分屏障：begin 后插入无关工作，end 处才真正等待 —— 隐藏转换延迟
let token = encoder.barrier_begin(&[Barrier::texture(tex, State::RenderTarget, State::Sampled)]);
encoder.dispatch(/* 与该资源无关的 compute，填 GPU 气泡 */);
encoder.barrier_end(token);   // 到此才要求转换完成
```

> 默认全自动（上层不写一行屏障）；手动屏障 + 拆分屏障是 AAA 热路径的「逃生舱」（原则 1），自动求解结果可被手动覆盖，二者在 §27 模糊测试里交叉对拍一致。

---

## 11. 交换链与呈现（Surface / Swapchain / 帧节奏）

- **Swapchain**：从 `Surface` 创建，管理呈现图像队列；格式/色域/present mode（FIFO/Mailbox/Immediate）可选。
- **呈现模式**：垂直同步（FIFO）、低延迟（Mailbox）、不同步（Immediate）；HDR 色域（接 §17）。
- **帧节奏 frames-in-flight**：N 帧 CPU-GPU 流水并行（通常 2–3），每帧独立命令缓冲/同步对象，CPU 不空等 GPU。
- **重建**：窗口 resize / 最小化 / 设备丢失时 swapchain 重建（接 `prism_window` 事件）。
- **present**：acquire→渲染→present 的标准循环，与 `prism_time` 帧步进、`prism_app` 主循环对接。
- **多窗口**：多 surface/swapchain 支持（编辑器多视口）。

---

## 12. 渲染通道与附件（RenderPass / 动态渲染）

- **RenderPass**：声明颜色/深度附件 + load/store 操作（clear/load/dontcare），对 tile-based GPU（移动/Apple）至关重要（省带宽）。
- **动态渲染 Dynamic Rendering**（Vulkan 1.3 / D3D12 / Metal）：免预声明 renderpass 对象，直接开始渲染，简化 API；RHI 门面优先暴露动态渲染，兼容传统 renderpass。
- **MSAA / resolve**：多重采样附件 + resolve 到单采样。
- **多视图 multiview**：一次 draw 渲染多视图（VR 双眼 / CSM 级联），接 §17。
- **tile 内存 / subpass**（移动）：subpass 让后处理在 tile 内完成不回主存，RHI 暴露给移动渲染路径。
- **契约**：附件格式必须与 PSO 声明一致（见 §8），否则创建期报错。

---

## 13. 多队列：异步计算与传输

- **异步计算 Async Compute**：计算队列与图形队列并行，把 compute 工作（光照剔除、后处理、粒子）塞进图形队列的 GPU 气泡，提升占用率——AAA 性能关键。
- **传输队列 Transfer**：专用 DMA 队列做上传/流送，与渲染并行不抢图形队列（接 §14 流送、asset）。
- **队列间同步**：semaphore 跨队列依赖（见 §10），资源跨队列所有权转移（Vulkan queue family ownership）。
- **能力降级**：无独立计算/传输队列的设备（部分移动/WebGPU）回退单队列串行，由 `GpuCaps` 门控。

---

## 14. 显存管理与子分配（接 platform / utils）

- **显存子分配器**（VMA 形态）：从驱动大块显存里子分配，避免每资源一次驱动 alloc（驱动分配次数有硬上限且慢）。
- **内存域**：device-local（显存）、host-visible（上传）、readback、UMA（统一内存，移动/Apple 一块到底）；接 `prism_platform` §24.2 GPU 共享内存。
- **放置资源 placed resources**：资源别名复用同块显存（接 `prism_render_graph` 瞬态资源别名，省显存）。
- **流送**：ring buffer 上传堆 + 异步传输队列（接 §13、asset 流送、platform §24.1 异步 I/O）。
- **预算与碎片**：显存预算追踪 + 碎片监控（接 `prism_diagnostic`），超预算降纹理 mip/分辨率。
- **延迟回收**：接 §10，资源 N 帧后回收，分配器据飞行帧安全复用。

**分配器策略深化**（VMA / D3D12MA 形态，接 utils §24.6）：
- **分级分配算法**：小/中块用 **TLSF（Two-Level Segregated Fit）** 做 O(1) 分配/释放且低碎片；大块按 2 次幂走 **buddy** 便于别名对齐；两者之上是「驱动大块（如 256 MB heap）→ 子分配」两级结构，驱动级 alloc 次数压到最低（驱动分配有硬上限且慢）。
- **专用分配 dedicated allocation**：大附件（全屏 MSAA target / 大体积纹理）与跨队列共享资源走独立 `VkMemoryDedicatedAllocation` / D3D12 committed resource，避免大资源卡在共享堆里拖累子分配、也利于驱动做压缩优化。
- **内存类型自动选型**：按 usage + 访问模式自动选内存域（device-local / host-visible-coherent / host-cached-readback / UMA）；上传路径优先 `BAR`/`ReBAR` 可见显存，无则回退 staging + 传输队列拷贝（§13）。
- **碎片整理 defrag**：后台增量搬迁（compaction）回收碎片——仅搬可重定位资源（句柄间接，原生指针不外露，§4 使搬迁对上层透明），受飞行帧 fence 门控，分摊到多帧避免卡顿。
- **常驻管理 residency**：显存预算追踪 + 冷热分级，超预算时按 LRU/优先级把低频资源逐出到系统内存（D3D12 `Evict`/`MakeResident`、Vulkan 内存优先级），接 §24.4 稀疏/部分常驻与 asset 流送；预算与碎片指标进 `prism_diagnostic`，超预算降纹理 mip/分辨率（§18）。
- **瞬态别名 placed**：接 `prism_render_graph` 帧内生命周期不重叠的瞬态资源别名复用同块显存（省显存峰值），别名边界的 aliasing barrier 由自动求解器插入（§10）。

---

## 15. 后端矩阵（Vulkan / D3D12 / Metal / WebGPU / Null）

| 能力 | Vulkan | D3D12 | Metal | WebGPU | Null |
|---|---|---|---|---|---|
| 平台 | Win/Linux/Android | Win/Xbox | macOS/iOS | 浏览器/wasm | 全(测试) |
| 显式屏障 | ✅ 最细 | ✅ | 🟡(自动多) | 🟡(自动) | n/a |
| Bindless | ✅ desc indexing | ✅ unbounded | ✅ argument buffer | 🟡 有限 | n/a |
| Mesh shader | ✅ | ✅ | ✅(Apple) | ❌ | n/a |
| 光追 | ✅ KHR | ✅ DXR | ✅(Apple) | ❌ | n/a |
| 异步计算 | ✅ | ✅ | ✅ | 🟡 | n/a |
| 多队列传输 | ✅ | ✅ | ✅ | ❌ | n/a |
| 动态渲染 | ✅ 1.3 | ✅ | ✅ | 🟡 | n/a |
| 调试层 | validation | debug layer | GPU frame capture | 浏览器 | 记录校验 |

> 读表须知：本表是**能力地图**，描述各后端目标平台与硬件能力，不是「首发要写几个后端」。首发只做 `backend-wgpu` 一个（见下），其余列是阶段二按需扩展的参照（§22）。

- **首发=单一 wgpu 后端**：阶段一只上 `backend-wgpu`，桌面（Win/Linux/macOS）、移动、Web **全部跑同一个 wgpu 后端**；CI 用 Null。原生 Vulkan/D3D12/Metal 是阶段二「撞到 wgpu 天花板才触发」的条件扩展（见 §22），可能永不触发。
- **Web 结论**：Web 唯一后端是 **wgpu/wasm**（wgpu 在 wasm 上直通浏览器 WebGPU），**不写 web-sys 手搓 WebGPU 后端**（相对 wgpu 零性能收益、纯重复劳动），**不做 WebGL 回退**。表中「WebGPU」列即描述此「wgpu-on-wasm」目标的能力画像。
- **Null 后端是什么**：一个**实现完整 RHI 接口但不碰任何真实 GPU 的空实现**——所有命令记录/提交都是 no-op，资源是占位句柄。用途是无显卡环境下跑渲染逻辑单测与 CI、做 headless 校验，以及给门面接口当「永远编得过」的参照实现。
- **屏障（Barrier）是什么**：显式 GPU 的同步原语，干两件事——(1) 把资源从一种用途状态/布局切到另一种（如渲染目标→被采样）；(2) 下执行+内存屏障，确保前一步 GPU 写完、缓存刷新可见，后一步才读，防止乱序并行踩踏导致随机闪烁/损坏。RHI 默认自动插屏障（见 §10），热路径可手动精调。

---

## 16. 与 render_graph / shader / asset / diagnostic 集成

- **`prism_render_graph`**：在 RHI 之上做帧依赖图——声明 pass + 资源读写，自动算屏障、瞬态资源别名（接 §14 placed）、并行录制（接 §9）。RHI 只提供原语，不管编排。
- **`prism_shader`**：产出各后端字节码（SPIR-V/DXIL/MSL）+ 反射信息（绑定布局），RHI 吃字节码建 ShaderModule/PSO（见 §8）；反射驱动自动 BindGroupLayout 生成。
- **`prism_asset`**：纹理/网格/材质资产经流送上传到 GPU buffer/texture（接 §14 流送、platform 异步 I/O）；GPU 资源句柄挂在资产句柄上。
- **`prism_diagnostic`**：GPU timestamp query（§9 marker）喂 diagnostic §8 GPU 计时 / §24.2 CPU·GPU 统一时间线；显存预算接内存图谱。
- **`prism_math`**：矩阵/向量/打包格式（法线压缩、颜色打包）供顶点/常量数据准备。
- **`prism_window`**：提供 surface 原生句柄与 resize 事件（见 §11）。

---

## 17. 高级功能增补（AAA）

- **Bindless 全局资源表**：海量材质/纹理一次绑定、draw 内索引取，配合 GPU-driven 渲染（§见下）大幅降 CPU 绑定开销，现代 AAA 标配。
- **GPU-Driven 渲染原语**：indirect/multi-draw-indirect-count + compute 剔除，让 GPU 自己决定画什么，CPU 只提交一次——支撑海量物体（接 render_graph/visibility）。
- **Mesh / Task Shader**（`mesh-shader` 档）：绕过传统顶点管线，GPU 端几何生成/剔除/LOD，支撑 Nanite 式海量几何（RHI 提供原语，不含其具体算法）。
- **硬件光线追踪**（`raytracing` 档）：加速结构（BLAS/TLAS）构建 + ray query / 光追管线，供反射/阴影/GI（接 GI 文档；RHI 只提供原语）。
- **可变速率着色 VRS**：按屏幕区域/内容降着色率省算力，供性能/移动。
- **PSO 预编译与缓存**：异步编译（§8）+ 磁盘缓存 + 预热，消除「PSO stutter」这一 AAA 顽疾。
- **多视图渲染**：VR 双眼 / 级联阴影一次 draw 多视图（§12），省一半 draw 开销。
- **HDR / 宽色域呈现**：HDR10 / scRGB 交换链，接色调映射。
- **GPU 时间线信号量 / 可计时同步**：细粒度 GPU 事件，供精密帧内重叠调度。

所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 18. 性能工程

目标：帧时间从 CPU-bound 推向 GPU-bound，再把 GPU 占用率拉满。按「CPU 提交侧 / GPU 执行侧 / 显存与带宽」三轴组织，每条都对应 §22 可量化基准。

**CPU 提交侧（降 draw-call 开销、吃满核心）**：
- **薄门面透传**：门面 `#[inline]` 直落后端，热路径无 vtable/无隐藏分配；句柄映射用密集表 O(1)（接 utils slotmap）。
- **多线程录制**：命令并行录制（§9）吃满核心，接 `prism_tasks` work-stealing；录制扩展比（1→N 核加速比）是硬指标。
- **bindless + GPU-driven**：海量材质/纹理一次绑定、draw 内索引取（§7/§17），配合 multi-draw-indirect-count 把「每物体一次 CPU 提交」降到「一帧一次提交」。
- **命令复用 bundle**：静态物体预录命令序列多帧复用（§9），免每帧重录。
- **PSO/描述符缓存**：内容寻址去重消除重复编译与布局重建（§7/§8）；缓存用分片锁/无锁避免成为并行录制瓶颈（§26）。
- **状态分桶与排序**：按 PSO/BindGroup 排序 draw，最小化 GPU 侧状态切换（降 pipeline flush）。

**GPU 执行侧（减停顿、提占用率）**：
- **屏障最小化 + 拆分屏障（split barrier）**：状态追踪合并冗余屏障、批量下发，并用拆分屏障把「转换开始/结束」拉开，让转换与无关工作重叠，减少 GPU 流水停顿（§10）。
- **多队列重叠**：异步计算/传输填图形队列的 GPU 气泡（§13），timeline semaphore 精排重叠点（§24.5）。
- **异步 PSO 编译 + 预热**：编译投后台车道，预热集消除 PSO stutter（§8/§24.7），首帧不卡。
- **瞬态资源别名**：render graph 瞬态资源复用同块显存（§14 placed），省显存且提升缓存局部性。
- **VRS / mesh shader / 多视图**：按内容降着色率、GPU 端几何剔除、一次 draw 多视图（§17），均 `GpuCaps` 门控。

**显存与带宽**：
- **显存子分配**：摊薄驱动分配开销（§14，驱动 alloc 次数有硬上限且慢），放置资源省显存。
- **零拷贝流送**：direct/异步 I/O + 持久映射（persistent-mapped）上传 ring（接 platform §24.1），资产直达 GPU 免中转拷贝。
- **tile 内存 / subpass**：移动端后处理在 tile 内完成不回主存（§12），省带宽（带宽是移动/Apple 的首要瓶颈）。
- **采样器反馈驱动流送**：GPU 告诉 CPU 实际采了哪些 tile，精准 page-in（§24.4/§24.6），省显存省带宽。

**性能反模式（评审一票否决）**：
- 热路径返回 `Result` 或做 `validation` 校验（校验只在开发期档位，见 §25）。
- 每资源一次驱动 `alloc`、每帧重建 PSO/BindGroupLayout、每 draw 全量重绑。
- 自动屏障对热路径一刀切，不给手动逃生舱（违反原则 1，见 §1/§10）。
- 单线程串行录制整帧、提交侧成为并行录制的锁瓶颈。
- 为规避「某后端慢」而在上层加 `#[cfg]`（差异必须锁后端，见 §23）。

---

## 19. 易用性与 Bevy / wgpu 迁移策略

> 核心取舍：**借 WebGPU 的对象模型，拒绝 WebGPU 的隐式语义。** 门面长得像 wgpu（映射稳、好读），但语义是显式的（手动屏障/显式内存/多队列是一等公民）。不追求对外部用户的「熟悉感承诺」，也不提供 wgpu 源码级兼容垫片——Prism 是脱 Bevy 的自研引擎，无需迁就外部代码库。

- **门面借 WebGPU 对象模型、语义做显式**：Device / Queue / BindGroup / BindGroupLayout / Encoder / PSO 的**名词与粗结构**借鉴 WebGPU/wgpu——这套分解已被证明能映射到 Vulkan/D3D12/Metal，借它省掉「从零推导可移植显式抽象」这一最大设计风险。但**语义一步到位做显式**：手动屏障、显式内存域、多队列同步、timeline semaphore 都是一等公民，而非 WebGPU 式隐式封装之上的补丁。只借名词形态，不借它为浏览器而设的「安全最低公分母」语义，避免 AAA 热路径永远在跟基座拧。
- **不承诺源码级兼容（无 `compat-wgpu`）**：不提供 wgpu 源码 drop-in 垫片。真正的迁移成本在**架构**（RenderApp / extract-prepare-queue / RenderGraph），不在 wgpu 调用；垫片只接得住最简单的裸调用，却会把 RHI 塑形回 wgpu、堵死手动屏障/多队列这些自研它的理由，得不偿失。
- **wgpu 是首发唯一后端（可能长期保留）**：阶段一 `backend-wgpu` 承载整条渲染体系（见 §22），**不是「临时点亮三角形」的占位，而是真正把引擎立起来的地面**。显式门面压在 wgpu 后端上时，手动屏障等显式调用退化为 no-op / hint（wgpu 本就自动追踪状态），不影响正确性；原生后端只在实测撞到 wgpu 天花板时按需补、可能永不触发。此点与「门面是否贴 wgpu」正交——API 做显式，照样能用 wgpu 当后端。
- **抽象早验证（纸面优先）**：门面抽象定稿前，拿 **Vulkan 做一次纸面映射**确认每个原语可落原生（见 §22 阶段一「抽象防锁死」），而非只靠宽容的 wgpu 后端——wgpu 太包容，会掩盖映射盲区，拖到真正写原生后端时才暴露则返工巨大。
- **自动屏障默认**：默认走自动屏障（§10），新手不碰同步也能跑对；需要极致性能再切手动。
- **Null 后端开发**：无 GPU 也能跑渲染逻辑单测（CI 友好）。
- **验证层一键开**：`validation` 档开校验，开发期即时捕获屏障/绑定/格式错误 + 可读报错。
- **prelude**：`use prism_render_driver::prelude::*;` 带入核心类型与句柄。
- **Bevy 迁移是架构 port，不是垫片**：Bevy 渲染绑死 wgpu + 散落的 render app；迁移以**架构重写**为主——`bevy_render` 的 RenderGraph → `prism_render_graph`、extract/prepare/queue → 自有帧编排、wgpu 资源句柄 → RHI generational 句柄。wgpu 调用本身是最好搬的部分，按需手工 port 即可；收敛到显式 RHI 门面后，后端可换、可测、可上 AAA 能力。

---

## 20. crate 分层与模块布局

```
pkg/prism_render_driver/
  src/
    lib.rs            # re-export + prelude
    instance.rs       # Instance / Adapter / GpuCaps / FeatureTier
    device.rs         # Device / Queue
    resource/         # 资源
      buffer.rs texture.rs sampler.rs view.rs format.rs
    binding/          # 绑定
      bind_group.rs layout.rs bindless.rs
    pipeline/         # 管线
      graphics.rs compute.rs cache.rs shader_module.rs
    command/          # 命令
      encoder.rs render_pass.rs copy.rs bundle.rs debug.rs
    sync.rs           # 资源状态/barrier/fence/semaphore/延迟回收
    swapchain.rs      # surface/swapchain/present/帧节奏
    memory/           # 显存
      allocator.rs heap.rs upload.rs budget.rs
    queue_graph.rs    # 多队列/异步计算/传输
    raytracing.rs     # 光追原语（raytracing 档）
    mesh_shader.rs    # mesh/task shader（mesh-shader 档）
    backend/
      wgpu/ null/            # 首发（阶段一）
      vulkan/ d3d12/ metal/     # 阶段二按需扩展（撞墙才做，见 §22）
    prelude.rs
  features = ["backend-wgpu","backend-null",          # 首发默认
             "backend-vulkan","backend-d3d12","backend-metal", # 阶段二按需
             "raytracing","mesh-shader","bindless","validation",
             "gpu-timing"]
  # 注：Web 由 backend-wgpu 在 wasm 上直通浏览器 WebGPU，不单列 backend-webgpu。
```

依赖：L1 地基 `prism_platform`/`prism_math`/`prism_utils`/`prism_diagnostic`；后端经典 `ash`/`windows`/`metal`/`wgpu`。**不碰任何 `bevy_*`**（`wgpu` 仅作可选后端被门面隔离）。

---

## 21. 契约、不变量与版本化

- **句柄安全**：资源句柄 generational（接 utils），GPU 仍在用时释放走延迟回收（§10/§14），绝不悬垂。
- **资源状态正确性**：自动屏障模式下 RHI 保证状态转换正确；手动模式下由调用方负责，`validation` 档交叉校验。
- **PSO 兼容性**：PSO 的附件格式/顶点布局/绑定布局必须与 renderpass/bindgroup/shader 一致，不一致创建期报错（非运行期崩）。
- **帧飞行不变量**：每飞行帧独立命令缓冲/同步对象/上传堆；回收必在对应 fence 完成后。
- **能力门控**：所有 AAA 能力（§17）用前必查 `GpuCaps`，缺失即降级，禁止无条件假设存在。
- **线程安全**：Device 资源创建与命令录制线程安全；单个 Encoder 不跨线程。
- **版本化**：PSO 缓存格式、字节码 ABI（与 shader 约定）、句柄布局、Format 枚举、显存堆布局均为版本化契约；驱动/后端版本探测与最低要求明确。

---

## 22. 路线图（wgpu 先行 · 原生按需扩展）与基准即规格

> 策略总纲：**先只做 wgpu 一个后端，把整条渲染体系在它上面跑通；原生 Vulkan/Metal/D3D12 不是硬日程，而是「撞到 wgpu 天花板才触发」的条件里程碑。** 门面从第一天起按显式语义设计（§19），wgpu 后端把手动屏障等显式调用当 no-op，后端走 `Backend` trait——加原生后端是纯增量，上层渲染器一行不改。

### 阶段一：wgpu 单后端打底（必做，核心价值所在）

- **M0 门面骨架 + Null/wgpu 后端**：Instance/Adapter/Device/Queue/Buffer/Texture/BindGroup/PSO/Encoder/Swapchain 门面（**语义显式**）+ `Backend` trait + Null 后端 + wgpu 后端 → **画出三角形**（接 window/shader）。对齐 gap 文档。
- **M1 资源与命令完备**：完整资源/视图/采样器 + 多线程录制 + 拷贝 + 自动屏障 + frames-in-flight → 渲染一个带纹理 PBR 物体（全程 wgpu 后端）。
- **M2 渲染体系在 wgpu 上跑通**：`prism_render_graph` + 材质/场景的真实渲染路径压在 wgpu 后端上跑起来，让**真实需求**把特性拉进 RHI；bindless / multi-draw-indirect-count / compute 剔除按 wgpu 现有能力能上多少上多少 → 千物体场景 + GPU-driven draw 开销基准。**这是脱 Bevy 后渲染器真正立起来的点，而非"脱 wgpu"。**
- **抽象防锁死**：抽象定稿前做一次 **Vulkan 纸面映射**，确认每个门面原语都能落到原生，排掉"只被 wgpu 兜底掩盖"的盲区——保证后续加原生是增量而非重写。

### 阶段二：原生后端（条件触发，撞墙才做，可能永不触发）

> 触发条件 = wgpu 的能力/控制粒度**实测**挡住了某个具体诉求（手动屏障榨性能、wgpu 尚缺的某能力、某平台 wgpu 表现不达标），而非到点就做。每个原生后端用前先问"是哪个 pass/基准非它不可"。

- **N-VK（条件）Vulkan 原生后端**：覆盖 Win/Linux/Android + 显存子分配 + 手动屏障 → 与 wgpu 后端**对拍一致性** + 性能增量基准。触发前，Windows/Linux/Android 一律用 wgpu。
- **N-MT（条件）Metal 原生后端**：Apple + 异步计算/传输队列 + UMA 内存域 → Apple 平台性能增量 + 异步计算收益基准。触发前 Apple 用 wgpu。
- **N-DX（条件）D3D12 原生后端**：仅 Win/Xbox 强需时才做（Windows 平时走 Vulkan 或 wgpu）+ 异步 PSO 编译 + 磁盘缓存 → PSO stutter 消除 + 加载时间基准。
- **高级能力（条件）**：mesh shader / 光追原语 / VRS / 多视图——哪个渲染技术真要用才加，严格 `GpuCaps` 门控，缺失即降级（见 §3/§24）。
- **Web**：始终 wgpu/wasm，**WebGPU 唯一后端，不做 WebGL 回退**；无需单独原生后端。

**基准即规格**：三角形/千物体帧时间、多线程录制扩展比、屏障合并前后 GPU 停顿、异步计算占用率提升、bindless/GPU-driven 的 CPU 提交降幅、显存子分配 vs 驱动分配、PSO 缓存命中与 stutter 消除、原生 vs wgpu 后端的画面位/感知一致与性能增量。核心价值在 **M0（点亮渲染，全渲染体系前置）+ M2（渲染体系在 wgpu 上立起来，脱 Bevy 的真正落点）**；原生后端是性能触顶后的可选增量，不是立项前提。

---

## 23. 诚实边界与风险

- 本文为设计规格，**当前无代码**；阶段一（M0–M2，wgpu 单后端）与阶段二（原生按需扩展，见 §22）均为 PLANNED。本 crate 是**整个引擎缺口影响面最大**的一块（gap 文档已标「关键」），也是工作量与风险最高的一块。
- **高风险项**：
  1. **自研原生后端工作量巨大（阶段二，条件触发）**：每个原生 Vulkan/D3D12/Metal 后端都是数万行显式代码，屏障/描述符/内存/同步四处是深坑；**务实路线是阶段一只做 wgpu 一个后端把整条渲染体系跑通，原生后端只在实测撞到 wgpu 天花板时按需补**（见 §22），可能永不触发。切忌立项即自研多后端拖垮全局进度——这是本 crate 最大的「假性必做」陷阱。
  2. **同步/屏障正确性（阶段一全程）**：资源状态/屏障是显式 API 最隐蔽的 bug，错一处即随机闪烁/画面损坏/驱动崩且难复现；自动屏障追踪逻辑复杂，必须 `validation` 档 + 跨后端对拍 + GPU 捕获工具交叉验证。
  3. **跨后端语义差异（全程）**：四家在描述符模型、内存类型、队列族、坐标系（Y 翻转/深度范围/NDC）、格式支持上差异巨大，门面抽象漏一处即某后端画面错；需严格的后端一致性测试套件（同输入对拍画面）。
  4. **显存管理与碎片（阶段一 M2 / 原生后端更甚）**：子分配器写不好会碎片化导致 OOM；放置资源别名错误导致数据踩踏；需 VMA 级成熟策略 + 预算监控。
  5. **PSO stutter（渲染体系成型后）**：PSO 首次编译卡顿是 AAA 顽疾，异步编译 + 缓存 + 预热缺一不可，且预热集合难覆盖全；需与 shader 变体系统深度配合。
  6. **AAA 能力的平台不均（高级能力落地时）**：mesh shader/光追/bindless 在 WebGPU/部分移动不支持，渲染器必须为每条高级路径备降级路径，否则这些平台直接黑屏；`GpuCaps` 门控 + 降级矩阵是硬要求。
  7. **着色器字节码链路（依赖 shader）**：RHI 吃 `prism_shader` 的后端字节码，二者 ABI/反射契约必须严丝合缝，否则 PSO 建不起来；两 crate 需协同演进。
- **与既有文档关系**：本 crate 向下依赖 `prism_platform_design_zh.md`（surface/GPU 可见内存/异步 I/O/动态库）、`prism_math_design_zh.md`（打包/矩阵）、`prism_utils_design_zh.md`（句柄表/可重定位容器/内容寻址缓存）、`prism_diagnostic_design_zh.md`（GPU 计时/统一时间线/显存图谱）；向上被 `prism_render_graph`（编排/屏障/瞬态别名）、`prism_shader`（字节码/反射）、`prism_material`、`prism_render_scene`、`prism_ui` 依赖；GPU-driven/mesh/光追原语供 `prism_gi_lumen_design_zh.md` 等渲染特性文档。整体组件缺口与分层见 `prism_engine_component_gap_zh.md`。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

## 24. AAA 高级功能增补·深化（v0.2）

本章在 §17 基础上**深化**顶级 RHI 的工业级能力——这些是「3A 画面规模与健壮性」真正的分水岭，而非 demo 级功能。均 `GpuCaps`/feature 门控，默认不付成本，缺失即走 §17 的基础路径降级；与前文显式门面内核互补。

### 24.1 GPU-Driven 渲染完全体（meshlet / 两阶段 HZB 遮挡剔除 / GPU 场景常驻）

§17 的 indirect/multi-draw 只是入口，真正支撑海量几何需要**全链路上 GPU**：

- **GPU 场景常驻**：实例变换/包围盒/材质索引常驻 GPU buffer（接 bindless §7），CPU 只增量更新脏项，不每帧重传整场景。
- **meshlet 管线**：网格切成 meshlet（64/128 三角形簇），compute/mesh shader 做簇级视锥 + 背面 + 遮挡剔除，支撑 Nanite 式海量三角形（RHI 提供 meshlet draw 原语，不含其具体 LOD 算法）。
- **两阶段遮挡剔除（HZB）**：第一阶段用上帧深度建层级 Z 缓冲（HZB）粗剔，渲染可见物生成本帧 HZB，第二阶段补剔上帧误判——消除遮挡闪烁，经典 two-phase occlusion culling。
- **indirect count 全家桶**：multi-draw-indirect-count + dispatch-indirect 让「画多少」由 GPU compute 决定，CPU 零参与。
- 供 `prism_render_graph`/visibility 搭 GPU-driven 主路径；CPU 从「每物体一次 draw call」解放到「每帧几次 indirect」。

### 24.2 显式多 GPU 与分帧 / 分块渲染

高端工作站 / 云渲染需要榨干多卡：

- **显式多设备**：枚举并独立驱动多个 `Device`（Vulkan device group / D3D12 linked adapters / 跨厂商独立设备）。
- **AFR / SFR**：交替帧渲染（AFR，各卡轮流出整帧）或分块渲染（SFR，各卡画屏幕一块），RHI 提供跨设备资源拷贝 + 同步原语。
- **跨设备传输**：显式 P2P 拷贝/共享堆 + 跨设备 semaphore；无 P2P 时经 host 中转（降级）。
- **能力门控**：绝大多数消费场景单卡，mGPU 为 `multi-gpu` 档，默认关闭。

### 24.3 设备丢失恢复与健壮性（TDR / 无缝重建）

AAA 发行版不能因驱动超时（TDR）/ 设备移除就崩溃退出：

- **device lost 检测**：submit/present 返回设备丢失即进入恢复流程，而非 panic。
- **无缝重建**：销毁旧 Device 下全部 GPU 资源 → 重建 Device/Swapchain → 从 CPU 侧权威数据（资产/场景）重上传 GPU 资源 → 续渲染，玩家最多见一次短黑屏而非崩溃。
- **资源重建契约**：GPU 资源句柄保持稳定（接 utils generational），底层原生对象重建后重新绑定，上层无感。
- **驱动崩溃取证**：丢失时抓 GPU 崩溃信息（Vulkan device fault / DRED on D3D12）喂 `prism_diagnostic`，定位是哪个 draw/资源触发。

### 24.4 稀疏 / 平铺资源与常驻管理（虚拟纹理支撑）

- **稀疏资源 sparse / tiled resources**：巨型纹理/缓冲只提交访问到的 tile（接 platform §24.2 稀疏虚拟堆），支撑**虚拟纹理**（terrain/世界的 TB 级纹理只驻留可见部分）。
- **常驻管理 residency**：显存超预算时把冷资源 evict 到系统内存/磁盘，访问时按需 page-in（接 §14 预算、asset 流送、platform 异步 I/O）。
- **部分常驻纹理 PRT**：mip 尾部常驻、高清 mip 按需，供流送纹理系统。
- **反馈驱动**：配合 §24.6 采样器反馈，GPU 告诉 CPU「实际采了哪些 tile」，精准流送。

### 24.5 GPU 工作图与时间线细粒度调度

- **时间线信号量 timeline semaphore**：单调递增计数的同步对象，替代大量二元 semaphore/fence，表达复杂帧内依赖更简洁高效（Vulkan timeline / D3D12 fence 原生支持）。
- **GPU 工作图 work graphs**（`work-graphs` 档）：GPU 自己派生后续工作（生产者-消费者在 GPU 端展开），进一步减少 CPU 往返——最前沿的 GPU-driven 形态（D3D12 Work Graphs / 等价扩展）。
- **细粒度帧内重叠**：用时间线信号量精确编排图形/计算/传输三队列（§13）的重叠点，最大化 GPU 占用率。
- **条件渲染 / 谓词**：predication 让 GPU 据查询结果跳过工作（遮挡查询驱动的条件 draw）。

### 24.6 高级光栅特性（ROV / 可编程混合 / 采样器反馈 / 原子）

- **光栅顺序视图 ROV**：保证同像素片元按提交序串行访问，供 OIT（顺序无关透明）、可编程混合、decal 累积。
- **采样器反馈 sampler feedback**：GPU 记录「实际采样了纹理哪些区域/mip」，回读驱动精准纹理流送（接 §24.4），省显存省带宽。
- **片元着色器原子 / UAV 原子**：像素级原子操作，供 OIT、直方图、GPU 粒子计数。
- **可变速率着色进阶**：§17 VRS 之上的 per-primitive / 基于内容的着色率，接眼动追踪（VR 注视点渲染）。
- **保守光栅化**：供体素化、GI、碰撞烘焙的保守覆盖。

### 24.7 管线库与着色器 ABI 深化

彻底消除 §17 提到的 PSO stutter，并稳固与 `prism_shader` 的契约：

- **管线库 pipeline libraries**：PSO 按着色器阶段/状态拆成可组合库单元，链接期组合，大幅减少变体编译量（Vulkan graphics pipeline library / D3D12）。
- **着色器 ABI 契约**：与 `prism_shader` 约定统一的绑定号/寄存器空间/push constant 布局/spec constant 映射，使同一份着色器源跨后端产出可互换的 BindGroupLayout（反射驱动，见 §16）。
- **特化常量深化**：运行期 spec constant 特化 PSO（分支消除、循环展开），免为每组合预编译独立 PSO。
- **根签名 / argument buffer 统一**：把 D3D12 root signature、Vulkan descriptor layout、Metal argument buffer 统一到一个声明式布局，一处声明多后端映射。
- **PSO 预热集采集**：开发/首次运行采集真实 PSO 组合，打包进发行版预热（接 platform 文件缓存），玩家零 stutter。

### 24.8 诚实边界

本章全部为 PLANNED 设计目标，无代码，且**深度与风险均高于 §17**。落地按「真实消费者拉动 + 能力撞墙触发」而非固定日程（对齐 §22 阶段模型）：**24.1 GPU-Driven 完全体**是 AAA 画面规模的核心，待 `prism_render_graph`/visibility 真正需要海量几何时拉入；**24.3 设备丢失恢复**是发行版健壮性硬指标，**必须随第一个原生后端同步建立**（恢复逻辑后补极难，wgpu 阶段先把句柄稳定性与重上传链路设计到位）；24.5 时间线信号量随多队列（§13）落地，24.7 管线库与着色器 ABI 随 PSO 缓存成型 + `prism_shader` 协同落地；24.4 稀疏/常驻、24.6 采样器反馈随虚拟纹理/流送系统落地；work graphs 为最前沿、仅 `cutting_edge` 档试验；24.2 多 GPU 为小众场景，最后考虑。所有能力严格 `GpuCaps` 门控，缺失即降级到 §17 基础路径——否则对应平台黑屏。与 `prism_render_graph`（GPU-driven 编排/HZB/瞬态别名）、`prism_shader`（ABI/管线库/spec constant）、`prism_asset`（稀疏流送）、`prism_diagnostic`（设备故障取证）的协同契约必须同步演进。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 25. 错误与结果模型（Result 边界 · 创建期前移）

Rust RHI 必须明确「什么用 `Result`、什么用 `panic`、什么留给 `validation` 档」——否则要么到处 `unwrap` 脆弱，要么热路径被 `Result` 拖慢。贯彻原则 3「错误前移」。

- **三类错误分层**：
  - **可恢复、运行期可发生 → `Result<T, RhiError>`**：设备丢失（§24.3）、OOM、swapchain 过期（out-of-date）、present 失败、PSO 异步编译未就绪。调用方必须处理。
  - **程序错误、开发期应消灭 → `debug_assert` + `validation` 档校验**，release 下为契约式未定义行为：手动屏障遗漏、绑定布局不匹配、句柄跨 `Device` 误用。不进 release 热路径成本。
  - **不可恢复、环境根本不满足 → `panic`（仅启动期）**：无任何可用 `Adapter`、请求的后端 feature 未编译。
- **错误前移到创建期**：PSO 的附件格式/顶点布局/`BindGroupLayout` 不一致 → **创建期** `Result::Err` 并带可读上下文（哪个字段、期望 vs 实际），绝不留到 draw 时驱动崩（见 §8/§21）。
- **`RhiError` 结构**：分类枚举（`DeviceLost` / `OutOfMemory{device|host}` / `SurfaceOutOfDate` / `Validation{msg}` / `Unsupported{cap}` / `BackendError{raw}`），`#[non_exhaustive]` 便于演进；携带后端原始码便于取证（接 §24.3 DRED / device fault）。
- **热路径零成本**：draw/dispatch/下屏障等每帧百万级调用**不返回 `Result`**（录制期不逐调用校验，错误交 `validation` 档在开发期捕获）；只有「可能真失败」的边界（`submit`/`present`/`create_*`/`map`）返回 `Result`。
- **设备丢失传播**：`submit`/`present` 返回 `DeviceLost` 时上层进恢复流程（§24.3）而非 panic——发行版健壮性硬指标。

---

## 26. 线程模型与 Send / Sync 契约

多线程命令录制（§9）是 AAA 并行渲染核心，但「什么能跨线程、什么不能」必须钉死在类型系统里，而非文档口头约定。贯彻原则 2/6。

- **`Device`：`Send + Sync`**：资源创建线程安全，可多线程并发 `create_*`（内部按需分片锁/无锁，后端保证）。资源句柄 `Copy + Send + Sync`，自由跨线程传递。
- **`Queue`：`Send + Sync`，`submit` 串行化**：提交顺序影响 GPU 执行序，`submit` 内部串行；跨队列依赖走 timeline semaphore（§24.5）而非锁。
- **`CommandEncoder`：`Send + !Sync`**：可 move 到某 worker 线程录制，但**单个 encoder 不被多线程共享**（录制独占）。每 worker 一个 encoder。
- **`CommandBuffer`：`Send`**：录好的命令缓冲 move 回提交线程排序提交。
- **并行录制模型**：`render_graph` 把一帧 pass 切 job（接 `prism_tasks` work-stealing），每 job 独立 encoder 并行录制 → 按图拓扑序汇总 `submit`；录制扩展比是 §22/§27 基准。
- **内部可变性策略**：PSO/`BindGroupLayout`/allocator 等共享缓存用无锁或分片锁，避免成为并行录制的串行瓶颈；句柄表（utils slotmap）读多写少，用读写锁或 epoch 回收。
- **帧飞行隔离**：每飞行帧（§11）独立 encoder 池 / 上传堆 / 同步对象，天然免跨帧数据竞争；回收必在对应 fence 完成后（§10）。

---

## 27. 测试、一致性与黄金图像（质量门禁）

跨后端 RHI 最大隐患是「同一份代码在不同后端画面不一致」（§23 风险 3）。质量不靠人眼巡检，必须工程化门禁。贯彻原则 6「可测在先」。

- **Null 后端逻辑单测**：无 GPU 环境（CI）跑完整录制/提交/状态追踪逻辑——句柄生命周期、自动屏障插入序、延迟回收时机、能力门控分支全部可断言（§15）。CI 必过。
- **黄金图像对拍（golden image）**：固定场景在每后端渲染 → 与基准图按**感知差异阈值**（SSIM/ΔE，而非逐位相等，因浮点/光栅规则跨后端有合法微差）比对，超阈值即 CI 失败。坐标系/深度范围/NDC 差异（§23）在此第一时间暴露。
- **后端对拍（cross-backend diff）**：wgpu 后端与原生后端（就绪后）同输入对拍，锁死「加原生不改画面」契约（§22）。
- **屏障/状态模糊测试**：随机命令序列喂自动屏障求解器，`validation` 档 + GPU 校验层交叉验证无状态错误，并与手动屏障结果对拍一致。
- **能力矩阵测试**：用 `GpuCaps` 伪装（关掉某能力）跑降级路径，确保每条高级路径的降级分支（§3 档位）真能跑而非黑屏。
- **性能回归门禁**：§22 基准指标（帧时间/录制扩展比/屏障停顿/占用率）纳入 CI 趋势监控，回归超阈值告警——性能是规格，不是玄学。
- **验证层分级**：`validation` 档开发期全开（屏障/绑定/格式/生命周期），release 零成本关闭；崩溃取证（DRED/device fault，§24.3）始终可一键开。

所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 28. 可扩展性与稳定性契约（加后端 / 加能力 / trait 版本化）

RHI 是 L4 表现层地基，被 render_graph/material/shader/scene 直接依赖——它的接口稳定性决定整个渲染体系的演进成本。本章把「如何扩展而不震荡上层」钉成契约（贯彻原则 2/5/7）。

- **加后端 = 实现一组 trait，上层零改动**：新增原生后端（阶段二 Vulkan/D3D12/Metal，§22）只需在后端 crate 实现 §4 的 `RenderBackend`/`Device`/`CommandEncoder` 等 trait + 句柄映射表，经编译期 feature（`backend-*`）接入。门面层与上层代码不改一行；新后端的差异（屏障粒度、绑定映射、队列模型）全部锁在后端模块内（§4/§7/§10 的映射表就是落地点）。新后端上线即进 §27 跨后端对拍，锁死「加后端不改画面」。
- **加能力 = 新增 `GpuCaps` 位 + 门控路径，缺省安全降级**：新硬件特性（如新一代 VRS/光追扩展）走「新增能力位 → 高级路径 `if caps.xxx` 门控 → 提供降级分支」三步，绝不改既有调用的默认行为；老后端/老硬件因能力位为 false 自动走降级，不被破坏（原则 2「缺失即降级」）。
- **trait/接口版本化**：面向上层的核心枚举（`RhiError`/`Format`/`Barrier` 状态等）一律 `#[non_exhaustive]`，新增变体不构成破坏性变更；接口演进遵循 semver，破坏性改动集中到大版本并提供迁移说明（接 §21 契约与不变量）。
- **可覆盖的策略点（policy hooks）**：Adapter 打分（§5）、D3D12 root signature 推导启发式（§7）、分配器选型（§14）、自动屏障合批策略（§10）等都是「有合理默认 + 可被上层覆盖」的扩展点，供特定项目做针对性调优而不 fork 门面。
- **稳定性门禁**：公共 API 变更需过 §27 的 Null 后端逻辑单测 + 跨后端对拍 + 性能回归三道门禁；API 文档标注稳定性等级（stable / experimental），`experimental` 项不保证跨小版本兼容，供新能力在稳定前迭代。
- **弃用流程**：接口弃用先标 `#[deprecated]` 保留至少一个小版本周期并给替代项，再在大版本移除——给上层 crate 留迁移窗口。

## 29. 性能 KPI 目标（量化基准）

性能是**规格而非玄学**（原则 4）：下列指标纳入 §27 性能回归门禁与 §22「基准即规格」，CI 持续追踪，回归超阈值即告警。数字为**设计阶段目标**（PLANNED），首个真实后端跑通后以实测校准，不是硬承诺。

| 维度 | 指标 | 目标 | 说明 |
|---|---|---|---|
| CPU 录制 | 单 draw 录制开销 | < 300 ns（典型 PSO/BindGroup 已缓存） | 热路径无 `Result`、无堆分配（§25） |
| CPU 录制 | 并行录制扩展比 | ≥ 0.8 × 核数（8 核达 6.4×） | work-stealing 多 encoder（§9/§26），缓存无锁/分片锁 |
| CPU 提交 | 每帧门面侧 CPU 开销 | < 1 ms（万级 draw，bindless 路径） | GPU-Driven/indirect 把 draw 下沉 GPU（§17） |
| 屏障 | 自动屏障 vs 手工最优的停顿差 | < 5% | 求解器合批 + 拆分屏障（§10）逼近手调 |
| 内存 | 子分配额外开销 | < 2% 显存浪费、驱动 alloc 次数 < 资源数的 1% | TLSF/buddy 两级（§14） |
| 内存 | 瞬态别名省显存 | 峰值显存 ↓ 20–40%（重后处理链场景） | placed + render_graph 别名（§14/§16） |
| PSO | 首帧卡顿（stutter） | 稳定帧无 > 2 ms 的 PSO 编译卡顿 | 异步 PSO 编译 + 管线缓存持久化（§8） |
| 启动 | 二次启动 PSO 重编 | ≈ 0（命中磁盘缓存） | 内容寻址 PipelineCache（§8） |
| 占用率 | 异步计算重叠收益 | 关键帧 GPU 占用率 ↑ 10–25% | compute 填图形队列气泡（§13） |
| 跨后端 | 黄金图像感知差异 | SSIM ≥ 0.995 / ΔE 阈内 | 浮点/光栅合法微差，非逐位（§27） |
| 一致性 | 加原生后端后画面偏差 | 对拍阈内无回归 | 「加后端不改画面」契约（§22/§28） |

- **测量方法**：固定场景 + 固定输入在 CI 跑，CPU 侧用 `prism_diagnostic` 采样，GPU 侧用 timestamp query（§24.2）；每后端各自基线，趋势比对而非绝对值跨机对比。
- **分级目标**：`baseline` 档保「不卡、正确」优先；`cutting_edge` 档追上表极致数字。移动/Web（wgpu/wasm）单独一套更宽松基线（受浏览器/tile GPU 约束），不与桌面同表强比。

