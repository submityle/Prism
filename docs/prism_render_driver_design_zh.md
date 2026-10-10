# Prism Render Driver 顶级次世代 AAA 级图形硬件抽象层（RHI）设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **RHI（Render Hardware Interface，图形硬件接口）**：把 Vulkan / D3D12 / Metal / WebGPU 等现代显式图形 API 的差异收敛到一层统一、显式、低开销的门面。它是整个渲染体系（render_graph / material / shader / scene）唯一踩在 GPU 上的地面，让渲染代码写一次、跨 Windows / macOS / Linux / 移动 / 主机 / Web 运行。
> 借形态不抄码。借鉴：
> - **现代跨平台抽象**：wgpu / WebGPU（可移植显式 API 范型）、The-Forge（AAA 多后端 RHI）、NVRHI（NVIDIA 显式 RHI）、Diligent Engine、sokol_gfx（极简）、bgfx（广兼容）
> - **引擎级 RHI**：Unreal `FDynamicRHI` / `FRHICommandList` / `RDG`、Unity SRP `CommandBuffer` / `GraphicsBuffer`、寒霜 / id Tech 的显式后端抽象
> - **底层显式 API**：Vulkan、D3D12、Metal、WebGPU（三/四大后端的屏障/描述符/队列/PSO 模型）
> 本文为纯经典图形系统编程路线，**不含任何 AI/ML 内容**（不涉及 DLSS/XeSS 等 ML 超分；仅提供其所需的 RHI 原语）。

- 版本: v0.2（设计阶段，未进入编码；v0.1→v0.2 新增第 24 章「AAA 高级功能增补·深化」：GPU-Driven 渲染完全体(meshlet·两阶段 HZB 遮挡剔除·GPU 场景常驻)/显式多 GPU 与分帧分块渲染/设备丢失恢复与健壮性(TDR·无缝重建)/稀疏平铺资源与常驻管理(虚拟纹理支撑)/GPU 工作图与时间线细粒度调度/高级光栅特性(ROV·可编程混合·采样器反馈·原子)/管线库与着色器 ABI 深化；均为 PLANNED，无代码）
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
22. 路线图（M0–M6）与基准即规格
23. 诚实边界与风险
24. AAA 高级功能增补·深化（v0.2）

---

## 1. 设计哲学与目标

现代图形 API（Vulkan/D3D12/Metal/WebGPU）是**显式**的：资源状态、内存、同步、描述符全要手动管，换来极致性能与可预测性，代价是极度复杂且四家互不兼容。如果渲染器直接写某一家 API，就被钉死在一个平台；如果到处 `#[cfg]`，代码会碎成四份。`prism_render_driver` 的使命是：**把四大显式 API 的共性抽象成一套统一、显式、低开销的门面**，对上给渲染图/材质/着色器一套干净接口，对下为每家 API 写一个后端，让「换 GPU 平台 = 换后端」而非「重写渲染器」。

**一句话定位**：`prism_render_driver` 是 Prism 的「GPU 地面」——Device/Queue/资源/绑定/管线/命令/同步/交换链全部一个显式门面多后端；渲染器永不直接碰 Vulkan 或 Metal，只调 RHI；语义向 WebGPU/现代显式 API 看齐，保留手动屏障与多线程录制以求 AAA 级性能。

四条总目标（按权重）：

1. **性能**：显式低开销（薄门面透传）；多线程命令录制；手动/自动屏障最小化；PSO/描述符缓存；多队列重叠（图形/计算/传输）；GPU 驱动内存子分配。
2. **效果（能力）**：覆盖现代渲染全需求——compute、indirect/multi-draw、bindless、mesh shader、光追（可选）、时间戳查询、多视图（VR）。
3. **可移植**：Vulkan/D3D12/Metal/WebGPU 一套上层代码，差异锁后端；新增后端 = 实现一组 trait。
4. **易用 + 可测**：门面向 WebGPU 直觉看齐（比裸 Vulkan 友好得多）；提供 Null/软件后端做无 GPU 单测与 CI；能力探测让上层优雅降级。

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

**不做**：不做高层渲染逻辑（PBR/光照/后处理归 `prism_render_*`）；不做帧依赖编排（归 `prism_render_graph`）；不做着色器编译（归 `prism_shader`，RHI 只吃编译好的字节码/模块）；不做窗口创建（归 `prism_window`，RHI 只接 surface 句柄）。

---

## 3. 档位化（capability / tier / feature）

- **能力位 `GpuCaps`**：运行时探测设备支持——bindless/descriptor indexing、mesh shader、光追（ray query/pipeline）、可变速率着色 VRS、timestamp query、多视图、indirect count、64-bit 原子、UMA（统一内存）、异步计算队列数等；上层据此选渲染路径或降级。
- **特性档 `FeatureTier`**：`baseline`（WebGPU 级最低集，保证全平台）→ `desktop`（桌面显式后端全能力）→ `cutting_edge`（mesh shader/光追/bindless 全开）。渲染器按档选管线。
- **编译期 feature**：`backend-vulkan` / `backend-d3d12` / `backend-metal` / `backend-webgpu` / `backend-wgpu`(包装) / `backend-null`(测试)、`raytracing`、`mesh-shader`、`bindless`、`validation`（调试层）、`gpu-timing`（接 diagnostic）、`compat-wgpu`（迁移兼容层）。
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
   backend-vulkan backend-d3d12 backend-metal backend-webgpu  backend-null
     (ash)        (windows)    (metal)      (浏览器)      (测试/离屏)
          └──────── 共享：显存分配器 / 状态追踪 / 缓存 ────────┘
```

- **门面层**：全部资源**句柄化**（generational handle，接 `prism_utils` slotmap），不暴露后端原生对象指针；命令通过 `CommandEncoder` 录制。
- **后端层**：每 API 一个模块，实现门面 trait；原生对象存后端侧资源表，句柄→原生对象映射。
- **共享层**：显存子分配、资源状态追踪、PSO/描述符缓存等后端无关逻辑复用。
- **Null 后端**：记录调用但不触 GPU，供上层逻辑单测与 CI（无 GPU 环境）。

---

## 5. 核心模型：Instance / Adapter / Device / Queue / Surface

- **`Instance`**：RHI 入口，选后端 + 开调试层；枚举 `Adapter`。
- **`Adapter`**：物理 GPU，暴露 `GpuCaps` + 内存信息 + 队列族；供多 GPU 选择（独显/核显）。
- **`Device`**：逻辑设备，资源创建的根；线程安全（可多线程创建资源与录制命令）。
- **`Queue`**：提交命令的通道，分 `Graphics` / `Compute` / `Transfer` 能力位；多队列见 §13。
- **`Surface`**：由 `prism_window` 提供原生窗口句柄创建（见 §11），RHI 不碰窗口系统本身。
- **初始化契约**：`prism_app` 建 `prism_platform` → `prism_window` 建 surface → RHI `Instance`→`Adapter`→`Device`→`Swapchain`，顺序固定。

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

- **资源状态转换 Barrier**：资源在不同用途间（渲染目标↔采样↔拷贝源）需状态/布局转换 + 内存可见性屏障。
- **双模策略**：
  - **自动屏障**：RHI 跟踪资源状态，`render_graph` 不写屏障，由 RHI/图自动插入（易用，默认）。
  - **手动屏障**：高级用户显式下屏障，极致控制（AAA 热路径）。
- **Fence（CPU-GPU 同步）**：CPU 等 GPU 完成（帧资源回收、readback），接帧飞行（frames-in-flight）。
- **Semaphore（GPU-GPU / 队列间）**：跨队列（图形↔计算↔present）同步，供异步计算重叠（见 §13）。
- **延迟资源回收**：资源释放挂到「N 帧后」安全点，GPU 用完才真正释放（接 §14），杜绝 use-after-free。
- **契约**：屏障错误是 GPU 渲染最隐蔽的 bug（表现为随机闪烁/损坏/驱动崩），`validation` 档必须开校验层交叉验证。

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

- **首发建议**：桌面优先 Vulkan（覆盖 Win/Linux/Android）+ Metal（Apple）；Win 可加 D3D12；Web 用 WebGPU；CI 用 Null。
- **过渡策略**：M0/M1 可先用 `backend-wgpu`（包 wgpu）快速点亮三角形，待热点明确再按平台替换为自研显式后端（见 §23 与 gap 文档建议）。

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

- **薄门面透传**：门面 `#[inline]` 直落后端，不加额外间接层；句柄映射用密集表 O(1)。
- **多线程录制**：命令并行录制（§9）吃满核心，接 `prism_tasks`，消除单线程录制瓶颈。
- **PSO/描述符缓存**：消除重复编译与绑定布局重建；内容寻址去重。
- **屏障最小化**：状态追踪合并冗余屏障，批量下发（减少 GPU 流水停顿）。
- **多队列重叠**：异步计算/传输填 GPU 气泡（§13），提升占用率。
- **显存子分配**：摊薄驱动分配开销（§14），放置资源省显存。
- **零拷贝流送**：direct/异步 I/O + 上传 ring（接 platform §24.1），资产直达 GPU。
- **bindless/GPU-driven**：降 CPU 提交开销，让帧时间从 CPU-bound 转向 GPU-bound。

---

## 19. 易用性与 Bevy / wgpu 迁移策略

- **门面贴 WebGPU 直觉**：Device/Queue/BindGroup/Encoder/PSO 语义与 wgpu/WebGPU 一致，熟悉 wgpu 即会用；但保留手动屏障/多队列的「逃生舱」做 AAA 优化。
- **`compat-wgpu` 兼容层**：映射 wgpu 常用类型/调用，Bevy/wgpu 代码迁移主要改 `use` + 少量 API 调整。
- **自动屏障默认**：默认走自动屏障（§10），新手不碰同步也能跑对；需要极致性能再切手动。
- **Null 后端开发**：无 GPU 也能跑渲染逻辑单测（CI 友好）。
- **验证层一键开**：`validation` 档开校验，开发期即时捕获屏障/绑定/格式错误 + 可读报错。
- **prelude**：`use prism_render_driver::prelude::*;` 带入核心类型与句柄。
- **Bevy 迁移**：Bevy 渲染绑死 wgpu + 散落的 render app；迁移时把 wgpu 调用收敛到 RHI 门面，渲染逻辑改依赖 RHI trait + render_graph，后端可换、可测、可上 AAA 能力。

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
      vulkan/ d3d12/ metal/ webgpu/ wgpu_wrap/ null/
    prelude.rs
  features = ["backend-vulkan","backend-d3d12","backend-metal",
             "backend-webgpu","backend-wgpu","backend-null",
             "raytracing","mesh-shader","bindless","validation",
             "gpu-timing","compat-wgpu"]
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

## 22. 路线图（M0–M6）与基准即规格

- **M0 门面骨架 + Null/wgpu 后端**：Instance/Adapter/Device/Queue/Buffer/Texture/BindGroup/PSO/Encoder/Swapchain 门面 + Null 后端 + wgpu 包装后端 → **画出三角形**（接 window/shader）。对齐 gap 文档 M2。
- **M1 资源与命令完备**：完整资源/视图/采样器 + 多线程录制 + 拷贝 + 自动屏障 + frames-in-flight → 渲染一个带纹理 PBR 物体。
- **M2 Vulkan 原生后端**：自研 Vulkan 后端（覆盖 Win/Linux/Android）+ 显存子分配 + 手动屏障 → 与 wgpu 后端对拍一致性 + 性能基准。
- **M3 Metal 后端 + 多队列**：Metal 后端（Apple）+ 异步计算/传输队列 + UMA 内存域 → 跨后端画面一致 + 异步计算收益基准。
- **M4 bindless + GPU-driven**：bindless 资源表 + multi-draw-indirect-count + compute 剔除 → 海量物体 draw 开销基准（CPU 提交降幅）。
- **M5 D3D12 后端 + PSO 缓存**：D3D12 后端（Win/Xbox 向）+ 异步 PSO 编译 + 磁盘缓存 → 消除 PSO stutter 验证 + 加载时间基准。
- **M6 高级能力 + Web**：mesh shader + 光追原语 + VRS + 多视图 + WebGPU 后端 → 高级特性 demo + Web 跑通 + `compat-wgpu` 迁移。

**基准即规格**：三角形/千物体帧时间、多线程录制扩展比、屏障合并前后 GPU 停顿、异步计算占用率提升、bindless/GPU-driven 的 CPU 提交降幅、显存子分配 vs 驱动分配、PSO 缓存命中与 stutter 消除、跨后端画面位/感知一致。核心价值在 **M0（点亮渲染，全渲染体系前置）+ M2（自研 Vulkan，脱 wgpu 第一步）+ M4（bindless/GPU-driven，AAA 性能分水岭）**。

---

## 23. 诚实边界与风险

- 本文为设计规格，**当前无代码**；M0–M6 均为 PLANNED。本 crate 是**整个引擎缺口影响面最大**的一块（gap 文档已标「关键」），也是工作量与风险最高的一块。
- **高风险项**：
  1. **自研多后端工作量巨大（M2–M6）**：Vulkan/D3D12/Metal/WebGPU 四后端各是数万行显式代码，屏障/描述符/内存/同步四处是深坑；**务实路线是 M0/M1 先用 wgpu 后端快速点亮渲染体系，待热点与瓶颈明确再按平台逐个替换为自研显式后端**（见 gap §的建议），切忌一上来自研四后端拖垮全局进度。
  2. **同步/屏障正确性（M1–M2）**：资源状态/屏障是显式 API 最隐蔽的 bug，错一处即随机闪烁/画面损坏/驱动崩且难复现；自动屏障追踪逻辑复杂，必须 `validation` 档 + 跨后端对拍 + GPU 捕获工具交叉验证。
  3. **跨后端语义差异（全程）**：四家在描述符模型、内存类型、队列族、坐标系（Y 翻转/深度范围/NDC）、格式支持上差异巨大，门面抽象漏一处即某后端画面错；需严格的后端一致性测试套件（同输入对拍画面）。
  4. **显存管理与碎片（M2）**：子分配器写不好会碎片化导致 OOM；放置资源别名错误导致数据踩踏；需 VMA 级成熟策略 + 预算监控。
  5. **PSO stutter（M5）**：PSO 首次编译卡顿是 AAA 顽疾，异步编译 + 缓存 + 预热缺一不可，且预热集合难覆盖全；需与 shader 变体系统深度配合。
  6. **AAA 能力的平台不均（M6）**：mesh shader/光追/bindless 在 WebGPU/部分移动不支持，渲染器必须为每条高级路径备降级路径，否则这些平台直接黑屏；`GpuCaps` 门控 + 降级矩阵是硬要求。
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

本章全部为 PLANNED 设计目标，无代码，且**深度与风险均高于 §17**。落地建议：**24.1 GPU-Driven 完全体**是 AAA 画面规模的核心，随 M4 优先；**24.3 设备丢失恢复**是发行版健壮性硬指标，应随 M2 自研后端同步建立（恢复逻辑后补极难）；24.7 管线库与着色器 ABI 随 M5 PSO 缓存 + `prism_shader` 协同落地；24.4 稀疏/常驻、24.6 采样器反馈随虚拟纹理/流送系统落地；24.5 时间线信号量随 M3 多队列落地，work graphs 为最前沿、仅 `cutting_edge` 档试验；24.2 多 GPU 为小众场景，最后考虑。所有能力严格 `GpuCaps` 门控，缺失即降级到 §17 基础路径——否则对应平台黑屏。与 `prism_render_graph`（GPU-driven 编排/HZB/瞬态别名）、`prism_shader`（ABI/管线库/spec constant）、`prism_asset`（稀疏流送）、`prism_diagnostic`（设备故障取证）的协同契约必须同步演进。所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。
