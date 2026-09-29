# Prism 渲染引擎 — 材质与管线完整架构设计（v2 / wgpu + WESL 版，放弃 Slang）

> 状态：架构提案（Draft，允许破坏性重构）
> 面向版本：Bevy/Prism 下一代渲染底座
> Shader 语言：**WESL**（现状，经 naga → WGSL/SPIR-V/Metal/DXIL）—— **放弃引入 Slang**（决策记录见 §2）
> 运行时后端：**wgpu**（已在用，经 Bevy）—— 跨平台的**运行时 API 层**抽象，Metal/Vulkan/D3D12/WebGPU 全覆盖
> 跨平台结论（2026-09-29 定案）：跨平台**只需一层**——运行时层 wgpu 已达成；着色器层维持 WESL，**不再引入 Slang 作为独立语言层**。原"Vulkan 优先"的真实目标是跨平台，已由 wgpu 满足。详见 §0.5 / §2。
> 本文依据：对 `pkg/` 下渲染 crate 的静态阅读（`wc -l` 实测）+ 架构讨论收敛结论；未运行示例/测试/基准
> 关联文档：`docs/prism_rendering_architecture_zh.md`（较早 draft，本文在其之上收敛材质/管线/子系统决策）
> 最后更新：2026-09-29（本版新增：§12 顶级产品对标矩阵、§13 PBR / §13.1 NPR 高级特性享有对照 / §14 NPR / §15 混合 三线 AAA 次世代高级特性清单、§16 性能/效果/易用性权衡总表；上版核心变更：放弃 Slang，收敛到 wgpu + WESL 单语言层）

---

## 0. 设计目标与非目标

### 目标（硬约束）

- 同一引擎**同时**是顶级 AAA 的 PBR / NPR / 混合 / 自定义引擎；四者皆一等公民，无谁是底座、无谁是补丁。赛道由具体项目/场景选择，不由引擎替用户选。
- **跨平台**（硬目标）：运行时经 **wgpu** 覆盖 Metal/Vulkan/D3D12/WebGPU/主机/移动；着色器经 **WESL → naga** 落到 WGSL/SPIR-V/Metal/DXIL。原文"Vulkan 优先"修订为此目标，见 §0.5。**不引入 Slang**（见 §2）。
- 吃满高级特性：GI/RT、虚拟阴影（VSM）、虚拟几何、时序上采样、动态光照、粒子、透明、毛发、布料、体积。

### 非目标（主动不做，避免撞墙）

- 不做完整无界 Substrate（性能重、撞弱平台、对"混合"无用——混合靠逐像素多材质，不靠单材质无限瓣）。
- 不追求 RT 里对 NPR 的物理正确（NPR 在次级光线里本就是降级的，见 §10）。
- 不追求"物理外观一致"的 PBR/NPR 统一（这是错误目标，见 §4）。

---

## 0.5 后端策略修订：跨平台的正确拆法（2026-09-29）

原文写"Vulkan 优先"，事后澄清：**真实意图是跨平台，Vulkan 只是当时误选的载体**。裸 Vulkan 并非天然跨平台——苹果平台无原生 VK（只能靠 MoltenVK 转译，功能是子集），Web 平台 VK 完全覆盖不到。以原生 VK 作跨平台地基，反而把苹果/Web 变成短板。

**跨平台需要两层各自独立的抽象，Prism 一层已经有了：**

| 抽象层 | 作用 | 正解 | Prism 现状 |
|---|---|---|---|
| **运行时 API 层** | 统一 Metal/VK/D3D12/WebGPU | **wgpu** | ✅ 已在用（Bevy→wgpu），Metal/VK/DX12/**Web** 天然全覆盖，比裸 VK 更跨平台 |
| **着色器语言层** | 一份 shader 编到各后端 | **WESL → naga**（现状即够） | ✅ 现有 68 个 `.wesl` 经 naga 落 WGSL/SPIR-V/Metal/DXIL，wgpu 全后端消费；曾拟引入 Slang，**已放弃**（§2） |

**修订结论（2026-09-29 定案）**：

- 运行时跨平台**已由 wgpu 达成**，不需要、也不应该为原生 VK 折腾（那是跨平台的倒退）。
- 着色器层**维持 WESL**：WESL 经 naga 已能落 WGSL/SPIR-V/Metal/DXIL，wgpu 在各后端消费——"一份代码编全平台"这条**现有链路已经具备**，无需再叠一层 Slang。曾把 WESL 判为"跨平台缺口"，复评后认定该缺口**本不存在**（naga 已补齐目标面；RT 也吃 WGSL）。详见 §2 决策记录。
- 因此本文所有"Vulkan 优先/VK 先"应读作 **"wgpu（已有运行时）+ WESL（已有语言层）"**。**不存在独立的原生 Vulkan 后端**——只有一个 wgpu 后端；超出 `WebGPU` baseline 的能力（bindless / mesh shader / ray query / 多重间接 count / RT）都是 **opt-in 的 wgpu 扩展 feature**（wgpu 内部映射到底层 Vulkan/Metal/D3D12 扩展）。adapter 报告支持哪些 feature，device 启用请求的子集，上层特性声明所需 capability。**能力协商是三档，不是二档**：① wgpu baseline（全平台）；② wgpu 类型化扩展 feature（`wgpu/vulkan` 等映射到底层 VK/Metal/D3D12 扩展，adapter 给不出则走 **fallback**，绝不切后端）；③ **`raw_vulkan_init` HAL 逃生舱**——对 wgpu 连类型化 feature 都没暴露的裸 VK 扩展，Bevy 上游 `raw_vulkan_init`（`crates/bevy_render` `raw_vulkan_init = ["wgpu/vulkan"]`）经 `wgpu::hal::vulkan::Instance::init_with_callback` + `Adapter::open_with_callback` → `create_device_from_hal` 注入 instance/device 创建回调，在**同一个 wgpu 设备**上启用额外 VK 扩展（能力记进 `AdditionalVulkanFeatures`，用时 `adapter.as_hal::<Vulkan>()` 下探裸句柄）。这**不是第二个原生后端**，也不是通用兜底——它是 Vulkan-only、丢跨平台+丢 CPU golden 的最窄一条路，非 VK 平台（Metal/Web）直接 fallback 回普通 wgpu。三桶判据与"什么该跌进逃生舱"见 §8.1。
- 开发机是 Apple Silicon macOS，Metal（经 wgpu）本就在跑，且能就地测 Metal + Web 两条跨平台线——是验证跨平台最合适的机器，而非障碍。

---

## 1. 顶层架构：共享 GPU-driven 基底 + 多并存前端 + 正交风格轴

```
                        ┌─────────────────────────────────────────────┐
                        │   资产 / 材质图 (node graph → closure IR)     │
                        └───────────────┬─────────────────────────────┘
                                        │ normalize → closure IR
        ┌───────────────────────────────┼───────────────────────────────┐
        │                    支柱一：材质 ABI (正交轴)                     │
        │   domain × closure(IR) × illumination × blend + specialization │
        └───────────────────────────────┼───────────────────────────────┘
                                        │
   ┌────────────────────────────────────┼────────────────────────────────────┐
   │              共 享 G P U - d r i v e n  基 底 (算一次, 全前端消费)          │
   │  支柱四: 可变形几何/模拟(蒙皮父级 → 布料/毛发/粒子/体积/水 sim) → gpu_scene │
   │  → 可见性(vis-buffer / virtual_geometry) → material id                   │
   │  支柱二: 光照/阴影"数据"服务(光源SoA+cluster / VSM页 / RT阴影反射 /       │
   │          GI·IBL·GTAO irradiance / BVH)  ——只出数据, 不出响应             │
   │  支柱三: 跨切面基底(motion+reactive mask / OIT / RT去噪 / capability)     │
   └────────────────────────────────────┼────────────────────────────────────┘
                                        │  material id + tile 分类路由
   ┌──────────────┬──────────────────────┼───────────────────┬────────────────┐
   │ 延迟 PBR 前端 │  forward+ 前端        │  NPR 风格化前端    │  自定义前端槽   │
   │ (虚拟几何/RT/ │ (一等:透明+粒子+可选  │ (描边/ramp/SDF面阴 │ (项目注入)      │
   │  GI/VSM)      │  NPR 承载)           │  影/风格化高光/post)│                │
   └──────────────┴──────────────────────┴───────────────────┴────────────────┘
        正交风格轴 illumination = Lit / Stylized / Unlit / Custom (叠在任意前端)
        混合 = 逐像素 material id, tile 分类到对应前端, 因共享光/影/GI而连贯
```

**一句话**：基底装满、算一次；每个前端从基底取自己要的数据、叠自己独有的响应和通道。这是"四者皆顶级 + 可混合 + 多平台"的唯一解——单独管线会让每条前端各自重造高级特性，没一条能到 AAA。

---

## 2. 决策记录：为什么放弃 Slang（本版核心变更）

> **结论（2026-09-29 定案）**：**不引入 Slang**。着色器层维持 WESL（经 naga → WGSL/SPIR-V/Metal/DXIL），跨平台交由 wgpu 运行时层。曾有的 `prism_render_slang`（885 行）、`prism_render_slang_abi`（1166 行）两个 crate **已删除**（commit `211f15988`，删除前实测零外部依赖）；文档命名与注释残留也已在 commit `abb661911` 清理完毕，当前全仓 `rg -i slang` 仅命中本决策记录（§2）。

### 2.1 曾经的四条理由，逐条失效

引入 Slang 的原始动机是四个"硬需求"。复评后逐条塌掉：

| 原动机 | 曾以为只有 Slang 能做 | 复评：为什么 WESL/现状已够 |
|---|---|---|
| 闭包 IR 要被延迟 pass **和** RT hit shader 共用 | Slang `interface`+泛型一份定义两处实例化 | WESL 有 `import` 能共享 closure 模块；wgpu 的 RT（ray query）本就吃 WGSL，同一份 WESL 编出的 WGSL 两处都能 import |
| 后端中立、真跨平台（含苹果/Web） | 同源 → WGSL/SPIR-V/DXIL/Metal/CPU | **naga 已做 WGSL→SPIR-V/Metal/DXIL**；运行时 wgpu 覆盖 Metal/VK/DX12/Web。目标面**本就齐**，不是缺口 |
| CPU golden reference 不漂移 | Slang CPU host target 从同一份 shader 出 CPU 参考 | **这是唯一 Slang 独占项**，但见 §2.2——它买的东西现在并不存在真实需求 |
| specialization 排列编译 | link-time specialization + 类型参数 | WGSL `override` 常量 + WESL 条件编译能近似；组合管理是构建脚本问题，非语言问题 |

四条里，前两条**本就不缺**，第四条**可替代**，真正 Slang 独占的只剩第三条"CPU 同源 golden"。

### 2.2 唯一独占项也不值：CPU golden 只是测试脚手架，不是真实路径

Slang 的净价值最终收敛为一件事：**把"shader 与 CPU golden 两份手写孪生"塌成单源**。但这条价值的前提是"存在一条需要长期对齐的真实 CPU 渲染路径"——**实测不存在**：

- 现状 **0 个 `.slang` / 68 个 `.wesl` / 0 个 `.wgsl`**，功能全在 WESL 上跑，golden 全在（20+ 个 `#[cfg(test)]` 手写 Rust 参考，如 `ssr/abi.rs:14` 注释 "agree with the CPU golden"）。
- 这些 CPU 参考**只是单元测试的对数脚手架**，不是引擎运行时会走的 CPU 渲染后端。它们数量有限、变更不频繁，"人肉对齐"的漂移成本**远低于**引入一整套 Slang 构建子系统的成本。
- 换句话说：Slang 花大代价消除的是一个**当前很小、且只在测试里存在**的漂移风险，收益/成本严重倒挂。

### 2.3 成本侧（放弃 Slang 直接省掉）

- slangc 进工具链、`build.rs` reflection→Rust codegen、variant 缓存 = **建一整套 shader 构建子系统**的一次性成本。
- 本机网络受限，**slangc 装不上**（需 `require_escalated` 联网拉二进制），直接卡住迭代。
- 现有 naga/WESL 对齐测试要全部改造成"Slang CPU target 对齐"，是纯迁移开销、零功能增量。

### 2.4 放弃后的着色器层长相（维持现状即架构）

- **语言 = WESL**：`pkg/prism_render_scene/src/shaders/*.wesl`，靠 WESL `import` 做 closure 复用，靠预处理/`override` 常量做特化。
- **ABI = 手写 `#[repr(C)]` + 校验测试**：`GpuMaterialHeader`/`GpuSurfaceParameters` 保持手写，但用**结构哈希驱动的 `MATERIAL_ABI_VERSION`** + CI 对齐测试兜住漂移（不需要反射生成器）。
- **跨平台落地 = naga + wgpu**：一份 WESL → naga 编各后端 → wgpu 运行时消费，Metal/VK/DX12/Web 全覆盖。
- 若将来 compute 核心真出现"大量 shader/CPU 孪生难维护"的痛点，再评估——但那是**可维护性投资**，不是功能/跨平台 blocker，且届时 Slang 也非唯一选项（可先考虑 naga 侧工具或宏生成）。

---

## 3. 支柱一：材质 ABI（破坏性重构）

### 3.1 正交轴取代互斥枚举

删掉 `MaterialShadingModel`（`pkg/prism_render_material/src/record.rs`，9 值单标量互斥枚举）这个**性能+易用性双病根**。改为四条正交轴：

```
Material = {
    domain:        Surface | Decal | Volume | PostProcess       // 保留, 现有已对
    closure:       ClosureGraph (IR, 见 3.2)                     // 取代 shading_model
    illumination:  Lit | Stylized | Unlit | Custom              // 新增, 正交风格轴
    blend:         Opaque | Masked | Transmissive | Transparent | Additive
}
specialization_id: u64   // 由上面轴的合法排列特化产出
```

- **`illumination` 是"风格轴"**，正交于 closure。`Stylized` 不再挤在 shading_model 里，而是"用什么方式解读同一份光照数据"（BRDF 积分 vs ramp 量化）。这解决了一个关键直觉——**NPR 不该和 Water 同级**：Water 是 closure/子系统，NPR 是 illumination 轴。
- `MaterialRenderClass` 里的 `NprOpaque/NprTransparent/CustomOpaque/CustomTransparent` **全删**——它们是 `blend × illumination` 的笛卡尔积被错误地拍平进一个枚举。拆回两条正交轴后自然消失。

### 3.2 闭包 IR：über-BSDF 一等 + 有界 slab 可选

`pkg/prism_render_material/src/ir.rs` 的 `ClosureKind` 保留大部分，但语义改变：

- **地基 = OpenPBR 锚定的 über-BSDF 一等 closure**。metallic/roughness/clearcoat/sheen/subsurface/transmission/anisotropy 全是它的**参数/瓣**，不是独立 shading model。`Cloth sheen`、`ClearCoat` 降级为 über 的瓣（不再是顶层 model）。
- **有界 slab 栈（封顶 3–4 瓣）作可选升级**：`Layer`/`Mix` 节点保留，但**编译期强制封顶**。超过上限 → validation error，不进无界 Substrate。
- **`normalize()` 必须改**：现在它把闭包压成单 `shading_model`（`ir.rs` 里 `if kind==Npr { shading_model = Npr }`）——**这是把正交信息毁掉的元凶**。新 `normalize` 产出 `ClosureGraph`（保留多瓣结构）+ `illumination` 轴 + `specialization_id`，**不再塌缩**。
- **毛发是真例外**：`ClosureKind::Hair` = 专用重 closure（Marschner/Chiang R/TT/TRT + dual-scattering），**不是** über 的瓣。NPR 头发 = `illumination=Stylized` 下的各向异性高光带（可与物理切线解耦，做"天使环"），是另一条 closure 实现。见 §6.3。

### 3.3 GPU 数据结构：ABI v4 变长打包堆（对标 UE Substrate 每像素预算裁剪）

> 本节据实描述已落地代码：`pkg/prism_render_material/src/record.rs`（`MATERIAL_ABI_VERSION = 4`）+ `surface.rs` + WESL 孪生 `pkg/prism_render_scene/src/shaders/material_unpack.wesl`。设计动机（拆胖结构）保留，但形态已从"定长数组下标 + 定长 blob"演进为**变长字堆 + LobeMask 解码**。

**动机（未变）**：旧 `GpuSurfaceParameters` 是 18 字段胖结构，每像素无差别携带 clearcoat/sheen/subsurface/transmission 等它可能永远用不到的瓣——既是性能坑（每像素都为不存在的瓣付带宽/寄存器），又是建模谎言（暗示每个面都有全部瓣）。ABI v4 把它拆成 **OpenPBR 锚定的紧凑 über 核心 + 按瓣可选 blob**，只为真正存在的瓣付代价。

**`GpuMaterialHeader`（`#[repr(C)]` + `bytemuck::Pod`，20 个 u32 字段）**：旧 `shading_model` 字段已删除，风格由 `illumination` 承载，编译期 permutation 身份由 `specialization_low`/`specialization_high`（切分的 `SpecializationId`）承载。关键字段：

- `parameter_offset`：**变长字堆里本材质打包 surface block 的 u32 字地址**（不再是定长 `GpuSurfaceParameters` 数组的元素下标）——因为块是变长的，场景只打包 über 核心 + 存在的瓣。
- `parameter_size`：该块字节长度 = `(12 + present_lobes*4) * 4`（`packed_size_bytes()`）。
- `lobe_mask`：本材质携带哪些可选 über 瓣（`LobeMask` 位），驱动打包参数解码——shader 在 `parameter_offset` 读 `parameter_size/4` 个字，用此 mask 把核心+存在瓣展开回完整 surface。
- `closure_graph_offset`：指向序列化闭包 IR，供 RT/延迟消费；`closure_mask` 保留（RT/分类用）。
- 其余：generation/revision/illumination/render_class/feature_flags/texture_offset/texture_count/sampler_offset/sampler_count/custom_program/active/material_epoch_{low,high}；`specialization()` 由 low|high 重组回 u64。

**变长打包核心（`surface.rs`）**：

- `SURFACE_CORE_WORDS = 12`：`GpuSurfaceCore`（base_color[4]/metallic/perceptual_roughness/reflectance/ambient_occlusion/normal_scale/alpha_cutoff + 2 pad），每个 lit/stylized/unlit 面都带的 OpenPBR 基座，**不含** clearcoat/sheen/subsurface/transmission/anisotropy。
- `SURFACE_LOBE_WORDS = 4`：每个瓣 blob 统一 16 字节量子，打包/解包按每个 set bit 定步长。
- `LobeMask`（6 个瓣，`COUNT = 6`）：`EMISSION=1<<0` / `CLEARCOAT=1<<1` / `ANISOTROPY=1<<2` / `SHEEN=1<<3` / `SUBSURFACE=1<<4` / `TRANSMISSION=1<<5`。瓣结构分别为 `GpuEmissionLobe`(emissive[4]) / `GpuClearCoatLobe` / `GpuAnisotropyLobe` / `GpuSheenLobe` / `GpuSubsurfaceLobe` / `GpuTransmissionLobe`(transmission/thickness/ior/dispersion)。
- `SurfaceParameterBlock::pack()`：先写 12 字核心，再**按 canonical 低位优先顺序只写存在的瓣**；`unpack(lobe_mask, words)` 逆向读回，缺席瓣解码为各自中性 `Default`（如 IOR 回到 1.5，绝不泄漏陈旧字段）。`packed_len_words() = CORE + present*LOBE`。
- **示例**：纯电介质（无瓣）= 12 字，而非旧的 24 字（18 字段 f32 对齐）。

**WESL 孪生（字节精确镜像）**：`material_unpack.wesl::prism_unpack_surface` 是 `SurfaceParameterBlock::unpack` 的字节精确 GPU 镜像——同样的核心 12 字 + canonical 低位优先瓣顺序。CPU 打包 / GPU 解包必须逐字节一致，否则材质错读。

**漂移守护（未变）**：结构手写 `#[repr(C)]`，由 `MATERIAL_ABI_VERSION`（当前 = 4）+ 对齐测试兜漂移（改结构 → 版本 bump → 测试红）。不引入反射生成器（见 §2）。回归测试已覆盖：`size_of::<GpuSurfaceCore>() == SURFACE_CORE_WORDS*4`、逐瓣子集 round-trip、错误长度 `unpack` 拒绝（`SurfaceUnpackError::WrongLength`）。

**与 §3.4 的连接**：§3.4 的 `specialization_id` 打包（illumination `0..8` / closure_mask `8..40` / render_class `40..48`）落进本结构的 `specialization_{low,high}`；`closure_mask`（哪些 closure 瓣存在）与 `lobe_mask`（哪些 über 瓣 blob 被打包）配合，GPU 侧据此选 permutation 并只读存在的瓣 blob。

### 3.4 有界 slab 闭包图：对标 UE5 Substrate 的深化规格

本节把 §3.2 的"有界 slab"从口号钉成可实现规格，直接映射到已落地的 `pkg/prism_render_material/src/ir.rs` / `axis.rs`（非纸面设计，代码已在）。

**数据模型**：材质是一张 `MaterialGraph`（DAG，节点 `MaterialNode`），`normalize()` 编译成 `NormalizedMaterial{ closure_mask, illumination, specialization_id, .. }`。闭包由三类节点组成：

- `MaterialNode::Closure{kind: ClosureKind, inputs}` —— 单个 BSDF 瓣 / slab。
- `ClosureKind::Mix` —— **水平混合**（coverage/weight 在两个子闭包间 lerp），对标 Substrate *Horizontal Mix*。用于"锈斑覆盖金属"、"泥点盖车漆"这类同一像素多材质按面积占比混合。
- `ClosureKind::Layer` —— **垂直镀层**（coat-over-base，按能量守恒把透射 throughput 传给底层），对标 Substrate *Vertical Layer / coat*。用于清漆、薄膜、湿膜、脏污叠层。

**`ClosureKind` 14 值 → `closure_mask` 位分配**（`closure_mask |= 1 << (kind as u32)`，`#[repr(u32)]` 判别式即位号；`specialization_id` 里 `closure_mask` 占 bit `8..40`）：

| bit | ClosureKind | 归属 | 说明 |
|---|---|---|---|
| 0 | Diffuse | über 瓣 | 朗伯/OpenPBR base |
| 1 | Conductor | über 瓣 | 金属 GGX |
| 2 | Dielectric | über 瓣 | 电介质镜面 |
| 3 | ClearCoat | über 瓣 | 车漆清漆（可由 Layer 承载，亦保留瓣位） |
| 4 | Sheen | über 瓣 | 布料绒毛边缘 |
| 5 | Subsurface | über 瓣 | 皮肤/蜡/叶片 SSS |
| 6 | Transmission | über 瓣 | 玻璃/透射 |
| 7 | Emission | über 瓣 | 自发光 |
| 8 | Hair | **专用重 closure** | Marschner/Chiang R/TT/TRT + dual-scatter，**不进 slab**（见 §6.3） |
| 9 | Volume | 域闭包 | 参与介质 |
| 10 | Npr | 风格标记 | 推 `illumination=Stylized`（正交轴，非塌缩） |
| 11 | Custom | 风格标记 | 推 `illumination=Custom` |
| 12 | Mix | 混合算子 | 水平，计入 slab 深度 |
| 13 | Layer | 混合算子 | 垂直，计入 slab 深度 |

- **über-BSDF = 单 slab 的多瓣参数化**（bit 0–7 的组合），对标 OpenPBR surface / Substrate 单 slab。metallic/roughness/anisotropy 是它的参数，不占独立 slab 深度。
- **只有 `Mix`/`Layer` 计入 slab 深度**。`slab_depth()` 在 DAG 上做记忆化最长链求解，`> MAX_CLOSURE_SLAB_DEPTH(=4)` → `MaterialValidationError`，永不进入无界 Substrate 树。

**封顶 4 的取舍（对齐"性能/稳定优先"）**：UE5 Substrate 是无界 slab 树 + 运行时 slab 预算裁剪；我们取其 slab 语义与 Mix/Layer 算子，但**编译期封顶 depth=4**。4 层足以覆盖 AAA 主力场景（base + clearcoat 车漆、湿表面镀膜、薄膜干涉、2–3 层污渍/锈蚀叠加）。以"拒绝无界"换取：固定每像素寄存器预算、permutation 可静态特化、着色器无递归、VGPR 占用可预测——即 objective 的性能与稳定性硬约束。需要第 5 层的极端资产按"降一层近似 + 美术告警"处理，不放开上限。

**`specialization_id` 打包**（`axis.rs`，LSB→MSB）：bit `0..8`=illumination、`8..40`=closure_mask、`40..48`=render_class(blend/domain 族)。`low()/high()` 切成两个 `u32` 落进 `GpuMaterialHeader`，GPU 侧据此选 permutation 与读哪些瓣 blob（§3.3）。

**normalize 不变式**（回归测试守）：① 单输出锥；② slab 深度 ≤ 4；③ `illumination` 由 `Npr`/`Custom` closure 推出正交轴值而**非**塌缩成单 model；④ 改任一 GPU 结构 → 哈希变 → `MATERIAL_ABI_VERSION` bump → 对齐测试红（§3.3）。

---

## 4. 支柱二：光照/阴影 数据服务 + 各前端响应（比材质更关键）

**核心原则：共享的是数据/加速结构，不是响应。** "一致"的正确定义 = **同一批授权光源驱动，外观按各自风格解读**（NPR 本就该长得不一样），不是物理外观相同。

### 4.1 基底共享（算一次，全前端消费）

**核心：这些是与「响应风格无关」的数据/加速结构，PBR/NPR/混合/自定义前端都读同一份，各自解读（§4.2）。** 下表钉到已落地的 `pkg/prism_render_shading/src/` 模块（非纸面），对齐 UE 的 clustered-forward+ / VSM clipmap / Lumen 世界空间 GI 形态：

| 基底服务 | 落地模块 | 关键类型 / 契约 | 对标 |
|---|---|---|---|
| **光源 SoA** | scene 侧光源缓冲 | 光源打包为 SoA，cluster/RT/GI 共用同一份索引空间 | UE `FLightSceneInfo` SoA |
| **Cluster 剔光（froxel）** | `cluster/{grid,assign,bounds}.rs` | `ClusterGrid`（`[x,y,z]` 维度 + `z_slice` 对数深度切片 + `linear_index`）；`assign_lights_to_clusters(grid, view_from_world, projection, lights, cfg)` → `ClusterLightAssignment{offsets_and_counts, light_indices}`（每 froxel `[offset,count]` 表 + 扁平升序光索引）；`ClusterAssignmentConfig{max_lights_per_cluster=256, intensity_cutoff=0.01}` | UE clustered-forward+ 光剔 |
| **VSM 深度页** | `shadow/virtual_sm/{clipmap,page_table,allocator,invalidation,request,slot,receiver_gen}.rs` | `VirtualShadowMap` 帧驱动 `drive_frame(FrameInput{light,camera_light_space,receivers,caster_movements})` → `FrameResult{resident,to_render,evicted,invalidation,budget,windows}`；`ClipmapConfig` 分级、`VirtualPageTable` 驻留、`PhysicalPageAllocator` 预算驱逐；`BudgetStats`(requested/hits/misses/allocations/evictions/over_budget/hit_rate)。**相机移动只做 clipmap window 快照、绝不失效页**；caster 移动脏其扫过的页 | UE5 VSM（clipmap + 页驻留 + 预算裁剪） |
| **世界空间 GI（辐照缓存）** | `gi/world_space/{probe_placement,probe_interpolation,octahedral,radiance_cache}.rs` | 探针放置 + `InterpolationConfig`/`ProbeNeighbor` 邻域插值 + 八面体编码 + `radiance_cache` | Lumen 世界空间辐射缓存/探针 |
| **屏幕空间 GI（fallback）** | `screen_space/gi.rs` | `SsgiParams` / `build_hemisphere_ray` / `gather_indirect_diffuse` → `SsgiGather`；无 RT/探针不足时的降级路径 | Lumen SSGI 兜底 |
| **RT 阴影·反射结果** | `screen_space/*`（SS 兜底）+ RT 桶（§8.1） | RT closure 着色可 CPU golden、traversal/BVH 不可（§8.1 三桶） | UE Lumen/RT 反射 |
| **IBL / 环境** | `environment/{cubemap,prefilter,brdf_lut}.rs` | 预滤环境 + split-sum BRDF LUT | UE 反射捕获 + split-sum |
| **AO** | `ao/{mod,temporal,denoise}.rs` | 空间 AO + 时序累积 + 去噪 | GTAO/时序 AO |
| **光通道 / light layer 路由** | `light_routing.rs` | `LightingChannelMask` / `LightLayerMask` / `LightRouting::contributes_to_layer` / `cull_lights_by_channel`——**这是 NPR 「分层打光」的共享地基**（§4.2 C 类专属响应的底座） | UE lighting channels |

**为什么这样拆算一次**：光剔（froxel）、阴影页驻留、GI 辐照、IBL、AO 全是**视图/场景函数、与像素着色风格无关**——算一次喂所有前端最省。风格差异只发生在「怎么响应这些数据」（§4.2），不在「数据本身」。VSM 的「相机动不失效页」「按 receiver 请求 + 预算驱逐」是把阴影带宽与场景规模解耦的关键，NPR 硬阴影同样白嫖这份页缓存（只是采样时换阈值/染色）。

### 4.2 各前端响应分家（A/B/C 三类）

| 基底数据 | A 类·同能力不同实现 | B 类·能用但固有降级 | C 类·各自专属 |
|---|---|---|---|
| 光源+cluster | PBR: BRDF 积分 / NPR: ramp 量化 | — | NPR: light layer 分层打光 |
| VSM 深度页 | PBR: PCF 软阴影 / NPR: 阈值硬阴影+染色 | — | NPR: **SDF 面部阴影**(独立数据源叠加) |
| GI/IBL | PBR: 物理环境光 / NPR: 收进来做风格化底光 | **NPR 对 GI 只收不发**(防能量爆炸) | — |
| RT 反射 | PBR: 物理反射 | RT 里 **NPR 降级为风格化 albedo 近似** | — |
| RT 去噪 | 共享去噪器 | **NPR 需单独调锐利分段**参数 | — |

- **A 类占大头**——"同能力不同实现"在这里成立。
- **B 类必须认**：RT×NPR 是"数据喂风格化响应"，不是物理追 NPR，换任何设计绕不掉。
- **NPR 专属通道**（SDF 面阴影 / shadow ramp / light layer）挂基底，**只 NPR 前端消费**。

### 4.3 WESL/WGSL 表达（钉到已落地着色器）

抽象的 `ILightResponse.shade(ctx, LightSample, Closure)` 在 WESL 里**不是一个 trait，而是"共享数据 import + 按 shading class 分派响应函数"**。下表钉到 `pkg/prism_render_scene/src/shaders/` 现存文件（非纸面），与 Rust 侧 `prism_render_shading` 一一对孪生（CPU golden 逐像素对齐）：

| 层 | WESL 落地 | 关键符号 | 消费方 | Rust 孪生 |
|---|---|---|---|---|
| **共享光源/阴影数据** | `lighting.wesl` | `LightSample`/`sample_directional_light`/`sample_punctual_light`/`sh_irradiance`/`env_brdf_approx` | 全前端 | `lighting.rs`/`punctual.rs` |
| **共享 surface/frame 桥** | `surface.wesl`/`brdf.wesl`/`material_unpack.wesl` | `SurfaceSample`/`ShadingFrame`/`DirectLightSample`/`to_direct_sample` | 全前端 | `surface.rs`/`resolve.rs` |
| **A 类响应·PBR** | `brdf.wesl` | `principled_direct`（+ GGX/anisotropic/fresnel/vis-smith） | Principled/Subsurface/ClearCoat/Cloth/Hair/Water 桶 | `resolve.rs`/`punctual.rs` |
| **A 类响应·NPR** | `brdf.wesl` | `stylized_direct`/`stylized_ramp`/`stylized_shadow`/`toon_direct`（legacy 兼容） | NPR 桶 | `stylized.rs`(`evaluate_stylized_direct`) |
| **C 类·描边（NPR 专属）** | `outline.wesl` | `outline_id_edge`/`outline_depth_edge`/`outline_normal_edge`/`evaluate_outline` + `outline_main`(@compute 8×8) | NPR 前端独占 | `outline.rs` |
| **C 类·光分层地基** | `light_routing.wesl` | `channel_mask_affects`/`light_routing_contributes_to_layer`/`cull_lights_by_channel_word`（`MAX_LIGHTING_CHANNELS=8`/`MAX_LIGHT_LAYERS=4`） | NPR 分层打光 + 全前端剔光 | `light_routing.rs` |
| **分派器** | `shading_resolve.wesl` | `switch params.shading_class`（9 arm，`SHADING_CLASS_PRINCIPLED..CUSTOM`）；`class_counts`/`class_offsets` 按桶分 bin（wavefront 一致） | —— | `classification.rs`(`classify_material_header`/`ShadingWorkPlan`) |

**前端切换 = 换 `switch` arm 里的响应函数，光源/阴影/surface 数据结构零改动**：`shading_resolve.wesl` 的每个 class arm 对同一份 `to_direct_sample(sample)`（来自共享 `lighting.wesl`）分别喂 `principled_direct`（PBR arm）或 `stylized_direct`（NPR arm）。9 值 `SHADING_CLASS_*` 常量是 §9-item3 正交轴派生投影的 WESL 端镜像，`class_offsets/class_counts` 保证同桶像素同 permutation（对齐 UE wavefront/tile 分类着色）。NPR 专属通道（`outline.wesl`/`light_routing.wesl` 分层）挂在共享基底上、只被 NPR arm 消费——正是 §4.2 C 类"各自专属"的着色器实体。

---

## 5. 支柱三：跨切面基底（解一次，全前端共享）

**这些是"多个前端/子系统都要，但谁都不该各造一份"的公共设施。** 下表钉到已落地 `pkg/prism_render_shading/src/` 模块并诚实标注落地/待建，对齐 FSR2/UE 的对应形态：

| 基底设施 | 落地模块 | 关键符号 / 契约 | 状态 | 对标 |
|---|---|---|---|---|
| **motion vector** | `prism_render_shading/src/screen_space/{motion,temporal}.rs`（着色侧）＋ `prism_render_architecture/src/motion/`（已 committed 一等模块，`lib.rs` 挂载：`encode`/`dilation`/`disocclusion`/`reproject`/`tiles`） | `motion_vector`/`MotionSample`/`project_world_to_screen`/`reproject_prev_uv_motion`；架构侧 `VelocityEncoding`（snorm16 速度量化 encode/decode）、`EncodedMotion`（velocity+masks 紧凑 texel）、`encode_sample`（透明像素自动置 `TRANSPARENT` flag） | ✅ 已落地（速度 G-buffer 有正式 wire 格式 + tile/膨胀/去遮挡/重投影全套） | UE velocity G-buffer + FSR2 dilated depth/velocity |
| **TAA / 时序上采样** | `taa/{jitter,resolve}.rs`、`upscale/{history,robust,reproject}.rs` | `taa_jitter`/`halton`/`resolve_taa`（YCoCg 邻域裁剪 + tonemap 权重）；`HistoryLock`/`update_lock`/`clip_history_neighbourhood`/`disocclusion_history_weight`（history lock 遇 disocclusion 融化 + 薄特征保护） | ✅ 已落地（FSR2 式） | FSR2 / UE TSR |
| **reactive / stencil mask** | 生产端：`transparency/routing.rs`（`writes_reactive_mask`）、`particle/shading.rs`（`reactive_mask`/`temporally_unstable`）、`motion/mod.rs`（`MotionSample.reactive`）＋ **已 committed 的正式 wire 格式** `motion/encode.rs`：`PackedMasks(u32)` 把 reactive/transparency/confidence（各 unorm8）＋ flags（`DISOCCLUDED`/`TRANSPARENT`/`STREAMING_REVEAL`）打进一个 32-bit 通道，随 `EncodedMotion` 与 snorm16 速度同存速度 G-buffer；消费端：两处 `taa/`（`prism_render_scene/src/shading/taa/` GPU dispatch 节点、`prism_render_shading/src/taa/` 算法层） | 粒子/透明/NPR/毛发标"别被 TAA 吃掉"的响应权重，喂给 `resolve_taa`/`disocclusion_history_weight` 做逐像素放宽 | 🟡 **生产端已收口（含正式打包 ABI）、消费端仍待接**：reactive 权重的产出与打包格式均已 committed，**仍缺把 `PackedMasks.reactive()` 读进 `taa/` resolve 的那一步**（实测两处 `taa/` committed 源零 reactive 引用） | FSR2 reactive/transparency mask / UE responsive AA |
| **OIT** | `oit.rs` | `OitFragment`/`oit_weight(view_depth, alpha)`/`OitAccumulation`/`composite_transparency`（Weighted-Blended OIT，深度加权） | ✅ 已落地（WBOIT，公共设施非子系统） | McGuire-Bavoil WBOIT |
| **RT / SS 去噪** | `screen_space/gi_denoise.rs`、`ao/{temporal,denoise}.rs` | `denoise_ssgi`/`SsgiDenoiseConfig`/`denoise_ssgi_pixel`；`reproject_prev_uv_gtao` | ✅ SS 侧已落地（RT 去噪共享此器，NPR 单独调参见 §4.2 B 类） | 时空联合去噪 |
| **能力分层 / fallback** | `prism_render_architecture/src/backend/` | 见 §8 三档协商 + fallback 矩阵 | ✅ 已落地 | UE RHI feature level |

**为什么这些必须"解一次"**：motion/OIT/去噪/TAA history 都是**逐像素跨前端共享的时序或合成资源**，任一前端各造一份就会算法漂移 + 显存翻倍 + 互相打架（尤其 TAA×粒子/透明）。**当前真·出血点进一步收敛为 reactive mask 的消费端接线**——motion（含 `motion/encode.rs` 的 `PackedMasks` 正式打包 ABI）、FSR2 式 history-lock、以及透明/粒子/motion 的 reactive **生产端**均已 committed 落地（连 wire 格式都定了），缺的只剩把已产出的 `PackedMasks.reactive()` 权重读进 `taa/` resolve（`resolve_taa`/`disocclusion_history_weight`）的那一小步（§11 粒子竖切收尾会逼出这条线；该接线归 TAA/上采样 lane，且 `taa/` 正被并发施工，见 §9 第 10 项）。OIT/去噪已是可复用公共设施，不得被误建成并列子系统。

---

## 6. 支柱四：可变形几何 / 模拟阶段 + 子系统注册表

### 6.1 蒙皮父级（所有形变子系统的根）

```
可变形几何阶段 (骨骼蒙皮 / morph / 形变输出)
   ├── 布料 sim (XPBD)        → 形变顶点
   ├── 毛发 sim (guide strand)→ 形变顶点
   ├── 粒子 sim (GPU)         → 实例/条带
   └── 普通蒙皮角色           → 形变顶点
          ↓ 全部汇入 gpu_scene → 可见性 → 各前端
```

落 `prism_physics_core / prism_physics_geometry`。**任何角色都要蒙皮**，所以这是 wgpu 竖切（§11）绕不过的第一个真子系统。

**`prism_physics_core` 现状远超“一句 XPBD”（已落地 ~24k 行，milestone 化）**——它就是“可变形几何/模拟阶段”的实体，下表钉到真实模块树 + 里程碑注释，对齐 UE Chaos / Houdini / 前沿论文：

| 求解器 / 子模块 | 落地模块 | 方法 | 里程碑 | 覆盖形变类 | 对标 |
|---|---|---|---|---|---|
| **统一软体 / 布料 / 绳 / 毛发** | `soft/{body,build,constraint,particle,solver}`（~3.1k 行）+ `solver/xpbd` | substep XPBD，“一切=粒子+约束”：布料=三角网、绳/毛发=1D 约束链 | M4 | 布料·毛发 guide·绳 | UE Chaos Cloth / PBD |
| **刚体** | `solver/xpbd/{rigid,contact_constraint,joint_constraint,island_solve,sleep_solve,velocity_solve,parallel_solve}`（~3.6k 行）+ `island`/`sleep`/`joint`/`ccd` | 子步位置动力学 + 顺应接触/静摩擦 + island 并行 + CCD sweep | M-rigid | 刚体角色/道具 | PhysX/Chaos rigid |
| **VBD 软体** | `vbd/{body,element,solver,system}`（~1k 行） | Vertex Block Descent（Chen 2024 SIGGRAPH），块坐标下降解隐式欧拉能量，**无条件稳定** | M7 | 高刚度软体/厚布 | 前沿 VBD |
| **MLS-MPM** | `mpm/{constitutive,grid,svd,transfer,weights,particle,solver,expf}`（~1.9k 行） | 物质点 P2G/G2P + 本构（弹/塑/雪/沙），SVD 形变梯度 | M5.5 | 雪·沙·泥·可碎 | Disney/Houdini MPM |
| **FLIP/APIC 液体** | `fluid/{mac_grid,pressure,transfer,particle,solver}`（~1.7k 行） | MAC 交错网格 + 压力泊松投影 + 标记粒子平流（自由表面） | M5.5 | 液体（液体引擎子系统底座） | FLIP/APIC 流体 |
| **降阶模态软体** | `reduced/{modes,subspace,integrate}`（~1k 行） | 低频振动模态子空间 `u=U·q`，千自由度→个位数 | M7 | 海量廉价软体 | 模态/子空间动力学 |
| **公共层** | `collide`/`collider`/`constraint`/`dynamics/integrator`/`pipeline`/`driver`/`island`/`sleep`/`state`/`snapshot`/`cache`/`query`/`lod`/`command`/`events`/`config` | 接触/碰撞/积分/管线/快照/LOD/命令流——各求解器共享 | —— | 全部 | Chaos 求解框架 |

**含义修正（2026-09-29 复核代码后校准，勿再误读为「已共享」）**：上表准确描述 `prism_physics_core` 这一 ~24k 行**独立统一模拟核**的内部结构，但必须诚实指出当前 committed 代码里的**两栈并存 + 未接线**现状——

- **实测依赖图**：`prism_render_architecture/Cargo.toml` **不依赖** `prism_physics_core`（`rg prism_physics_core pkg/prism_render_architecture/Cargo.toml` 无命中）；全仓仅 workspace 根 `Cargo.toml` 与 `benches/Cargo.toml` 引用它。即 physics_core 目前**尚未被渲染管线消费**。
- **渲染侧子系统各自自持 sim**：`prism_render_architecture/src/` 下 `cloth/`（`dynamics.rs::solve_cloth`+`vbd.rs`+`ccd.rs`+`tearing.rs`+`pressure.rs`+`collision.rs`+`sleep.rs`+`lod.rs`）、`hair/`（§6.3 的 XPBD/VBD solver）、`particle/`（`simulation.rs`+`emitter.rs`+`stages.rs`）、`water/`（`flip.rs`+`pbf.rs`+`swe.rs`+`spectrum.rs`）、`volumetric/`（`raymarch.rs`+`avsm.rs`+`multiscatter.rs`）都**就地实现了自己的求解器**，并未调用 physics_core。
- **设计意图 vs 现状的裂缝**：`cloth/coupling.rs` 的注释明确「authoritative rigid-body integrator lives in `prism_physics_core`，render-side module must not reimplement the solver」——即**意图**是 physics_core 当权威刚体/软体积分器、渲染侧只做接触的渲染半边。但**现状**是渲染侧自持了完整 sim，physics_core 未接线，形成两套并行模拟栈。**这是一条真实的待收敛重构线（非本文档 lane 可独改，须由 physics/architecture owner agent 决策接线方向）**。

**因此子系统边界的准确表述是**：每个一等子系统（粒子/毛发/布料/体积/水）当前**自持几何生产 + sim + 特殊渲染**三件套（§6.2 判据据此成立）；physics_core 是**平行的权威模拟核候选**，其与渲染侧的接线（复用 vs 保持渲染侧轻量代理 + physics_core 当权威）是**未定案的架构决策**，不应在设计文档里预先断言为「已共享」。§6.3“共享基底 + 分叉响应”仍成立，但那讲的是**渲染响应侧**（PBR/NPR 分家、跨切面服务共享），**不等于 sim 核已统一**。

### 6.2 子系统注册表（四档，钉死）

> 现状校准（2026-09-29，实测 `pkg/prism_render_architecture/src/` 模块树 + 行数）：档 1 五个一等子系统**均已落地**，且原列「档 2 后置留槽」的**水/海洋已提升为一等子系统**（`water/` 25 文件 ~9.2k 行，含 flip/pbf/swe/spectrum/foam/caustics/dispersion/coupling + GPU WESL 计算内核）。真正仍留槽的只剩植被/贴花。

| 档 | 成员 | 判据 | 落地现状 |
|---|---|---|---|
| **0 地基** | 可变形几何(蒙皮/morph) | 所有形变子系统父级 | 竖切第一子系统 |
| **1 一等子系统** | 粒子(`particle/` 19 文件 ~15.4k 行)·毛发(`hair/` 20 模块 ~7.8k 行)·布料(`cloth/` 23 文件 ~12.1k 行)·体积(`volumetric/` 25 文件 ~11.2k 行)·**水/海洋**(`water/` 25 文件 ~9.2k 行) | 自己的几何+sim+特殊渲染 | **均已落地**（各自持 sim，见 §6.1；粒子最先，带起 reactive mask） |
| **2 后置留槽** | 植被·贴花 | 是子系统但重/非通用 | 未建（真留槽） |
| **3 陷阱·别做子系统** | SSS/sheen/清漆(=closure)、大气(=光照数据服务)、OIT/motion(=基底服务) | 无几何无sim / 是公共设施 | — |

**判据**：需要自己的「几何+sim+特殊渲染(透射/OIT/RT代理)」才是子系统；只是着色变化 → closure；大家都消费 → 基底服务。**布料要拆**：sim 是子系统、sheen 是 über 一个瓣。**液体不单列子系统**——液体 sim 由 `water/`（FLIP/PBF/SWE 自由表面）+ `prism_physics_core` 的 `fluid/`(FLIP/APIC)/`mpm/` 承载（§6.1），渲染半边归 water 子系统 + 透明/OIT 基底服务，符合判据。

### 6.3 毛发 mini 支柱（子系统内部同构范例，已落地为 7.8k 行毛发引擎）

```
共享基底(PBR/NPR 都吃): strand几何(连续LOD→card, 禁硬切换) + guide-strand sim
                       + 共享 deep-transmittance 自阴影服务 + 插进共享 OIT
分叉响应(C类, 真分家):  PBR: Marschner/Chiang R/TT/TRT + dual-scattering
                       NPR: 风格化各向异性高光带 + ramp + 阴影偏移(可与切线解耦=天使环)
fallback:              strand 高配, card 基线; RT 反射里毛发用 proxy 或排除
```

**这不是纸面 mini 支柱**——它已落地为 `pkg/prism_render_architecture/src/hair/`（20 模块、~7819 行的毛发子系统）+ `pkg/prism_render_shading/src/hair*.rs`（4 套响应模型），恰好逐条印证「共享基底 + PBR/NPR 分叉响应」这套子系统同构范式。下表把范式钉到 committed 代码，对齐 UE Groom / Chiang-Marschner / Deep Opacity Maps：

| 范式层 | 落地模块 | 关键符号 / 契约 | 对标 |
|---|---|---|---|
| **模拟（子系统私有 sim）** | `hair/solver.rs`、`hair/dynamics.rs`、`hair/self_collision.rs`、`hair/sdf_collision.rs`、`hair/collision.rs`、`hair/wind.rs`、`hair/sleep.rs` | `HairSolverKind{Xpbd,Vbd}` + `SolverSelection::choose`（按 authored stretch stiffness 单阈值择解：软/中走 XPBD、硬定型走 VBD）；`simulate_strand_vbd`；SDF/解析碰撞 `push_out_of_field`/`resolve_strand_collisions`；`WindField`/`apply_wind`；`GroomSleepState`（motion-energy 阈值休眠省算） | UE Chaos 毛发 XPBD + VBD 定型 / Houdini Vellum |
| **几何生产（连续 LOD，禁硬切换）** | `hair/lod.rs`、`hair/transition.rs`、`hair/interpolation.rs`、`hair/ribbon.rs`、`hair/frames.rs` | `HairLodTier{Strands→ReducedStrands→Cards→Mesh}`（`is_strand_based`/`coarseness`/`coarser_of`）；`HairLodTransition`（`is_cross_fading`/`proxy_alpha` 交叉淡入，杜绝 LOD 跳变）；guide→render `interpolation`；`RibbonMesh` card/ribbon 生成；`build_strand_frames`（RMF 相干帧供 ribbon 定向） | UE Groom guide→strand 插值 + card LOD |
| **光栅路由（软/硬分派）** | `hair/raster.rs` | `HairRasterPath{SubpixelSoftware,ThickHardware,Culled}`（细 strand 走 compute 软光栅解析累进 vis-buffer 覆盖、近粗 strand/card 走硬件三角、背面/亚可见/零长剔除）；`DEFAULT_HAIR_SOFTWARE_WIDTH_PX`/`DEFAULT_HAIR_MIN_COVERAGE` | UE Nanite 毛发软光栅 / vis-buffer |
| **共享自阴影服务** | `hair/deep_transmittance.rs`、`hair/deep_opacity_layout.rs` | `build_deep_opacity`/`DeepOpacityLayers`/`sample_transmittance`；`DeepOpacityMap`/`build_deep_opacity_map`/`map_transmittance`（分层深度不透明累积，PBR/NPR 前端共吃同一透射服务，不各造一份） | UE Deep Opacity Maps / Yuksel-Keyser |
| **RT 代理策略** | `hair/rt_proxy.rs` | `RtReflectionRole`/`RtProxyPolicy`（`participates_in_rt`/`traces_strands`：RT 反射里毛发按 proxy 或排除，兑现 §10「RT×毛发降级」） | UE RT 毛发 proxy |
| **导入 / 分组** | `hair/groom.rs`、`hair/groom_import.rs`、`hair/mod.rs` | `step_groom`/`GroomStepConfig` 每帧驱动；`HairGroupHandle`；groom 资产导入 | UE Groom asset |
| **PBR 响应（真分家）** | `prism_render_shading/src/hair_chiang.rs`、`hair_kajiya.rs`、`hair_fiber.rs`、`hair.rs` | `evaluate_hair_chiang_direct`（Chiang R/TT/TRT + dual-scattering）、`evaluate_hair_kajiya_direct`（Kajiya-Kay 廉价路径）、`evaluate_hair_fiber_direct`（fiber-level）、`evaluate_hair_direct`（统一入口，配 CPU golden） | Chiang(Disney) / Marschner / Kajiya-Kay |
| **NPR 响应（真分家）** | `prism_render_shading/src/stylized_hair.rs` | `StylizedHairParams`/`evaluate_stylized_hair_direct`（风格化各向异性高光带 + ramp + 可与切线解耦的阴影偏移＝天使环） | miHoYo / Arc System Works 毛发 |

**注意共享分界（对齐 §6.1 校准后的现状）**：毛发 sim 落在 `prism_render_architecture/src/hair/`（子系统私有几何+动力学，`solver.rs` 自持 XPBD/VBD），与 `prism_physics_core` 的通用 soft/XPBD 是「同族算法、不同落点」。**关键澄清**：如 §6.1 复核所述，渲染侧子系统（含毛发）当前**均自持 sim、未接线 physics_core**（`prism_render_architecture` 不依赖该 crate）；physics_core 是平行的权威模拟核候选。毛发因 strand 拓扑特化 + deep-opacity/软光栅深度耦合，即便将来接线也很可能**保持子系统就地 sim**（strand 特化远离通用软体）。这条边界在 §6.2 判据下自洽（子系统 = 自己的几何+sim+特殊渲染）。

**所有一等子系统内部都应长这个样**（共享基底 + PBR/NPR 分叉响应 + 连续 LOD + 软/硬光栅路由 + 共享跨切面服务）——毛发是已落地的同构样板，布料/粒子/液体/体积按此范式对齐，架构一致性拉满。

---

## 7. 前端（并存，皆一等）

- **延迟 PBR 前端**：承载虚拟几何(Nanite 级)/RT/GI/VSM。虚拟几何与 forward+ 互斥，**这才是逼向延迟的真原因，不是 PBR 本身**。
- **forward+ 前端（一等公民）**：透明**强制** forward，所以这条路本就必建；NPR 复用它、粒子挂它，**不是额外成本**。
- **NPR 风格化前端**：描边 / ramp / SDF 面阴影 / 风格化高光 / rim / 风格化 post。**vis-buffer 的 material id 边界白送描边**。
- **自定义前端槽**：项目注入 `illumination=Custom`，走自定义 WESL closure + 自定义 pass。
- **混合 = 逐像素 material id + tile 分类路由**：GPU 把像素按 material id 分 tile，路由到对应前端。因共享光/影/GI 而连贯——这是"管线级混合"的具体机制，也是它比"各前端各算高级特性"优越的地方。

**落地锚点（对齐 committed 代码）**："逐像素 material id + tile 分类路由"不是设想，已两侧落地——
- **CPU 分桶**：`prism_render_shading/src/classification.rs` 的 `classify_material_header` 把每个材质头投影到 9 值 `MaterialShadingClass`（Water/Hair 由 `render_class` 直绑；Unlit/Npr(=Stylized)/Custom 由 `illumination` 定；其余 Lit 由主导 closure 经 `lit_class_from_closures` 投影 Hair/Subsurface/ClearCoat/Cloth/Principled），`ShadingWorkPlan::build` 据此把像素分成 `ShadingWorkItem` bin（同桶同 permutation，wavefront 一致）。
- **GPU 路由**：`prism_render_scene/src/shaders/shading_resolve.wesl` 里 9 个 `SHADING_CLASS_*`(PRINCIPLED=0 … CUSTOM=8) 常量 + `switch params.shading_class` 的 9 条 case，分派到 `shade_principled`/`shade_toon`(NPR)/`shade_subsurface`/`shade_clearcoat`/`shade_cloth`/`shade_hair`/`shade_water` 等前端着色函数——**同一 resolve pass、同一套光/影/GI 输入，仅着色分支按桶切换**，这正是「管线级混合」优于「各前端各建一套高级特性」之处。
- **描边白送**：NPR 前端的 material-id 边界描边直接消费 vis-buffer 的 id（`outline.rs::outline_id_edge`，§9 第 4 项），无需额外 id pass。

---

## 8. 后端抽象（破坏性重构）

> 修订（2026-09-29 定案）：跨平台的**运行时抽象已由 wgpu 提供**（Bevy 现状）。本节是在 wgpu 之上的**能力协商层**，**不是**要绕开 wgpu 自己写多后端，也**不存在第二个原生 Vulkan 后端可供升级**。收敛为**单一 wgpu 后端**：能力 = `WebGPU` baseline + 按需启用的 wgpu 扩展 feature；给不出的能力走 fallback，绝不切后端。见 §0.5。

- 删 `BackendMode{VulkanFirst,WgpuCompatibility}` 二值枚举、删 `BackendId{Wgpu,NativeVulkan}` 两后端模型、删 `VulkanTier/BackendTier` 分级 → **单 wgpu 后端 + wgpu 扩展 feature 协商**。
- 能力用 `Capability` 位集表达（`Compute`=baseline，其余 `BindlessDescriptors/IndirectDrawCount/MeshShading/RayQuery/RayTracingPipeline` 皆为 opt-in 扩展 feature，文档标注对应 wgpu feature 名，如 `EXPERIMENTAL_MESH_SHADER`）。`negotiate_features(adapter_supported, requested)` 产出 `EnabledFeatures{enabled, unavailable}`：`enabled = WEBGPU_BASELINE ∪ (adapter ∩ requested)`，`unavailable = requested − adapter`（请求了但 adapter 不支持的扩展，驱动 fallback）。
- **fallback 矩阵**：每个高级特性声明所需 capability + 降级路径（无 RT → SSR/SSGI；无 mesh shader → 传统 index draw；无 bindless → 描述符表）。
- "一份 shader、多平台"由 **WESL → naga** 达成（一份 WESL 编到 WGSL/SPIR-V/Metal/DXIL）；跨平台差异收敛为"补 capability 分支"，运行时由 wgpu 落到 Metal/VK/D3D12/WebGPU/Web。

### 8.1 可测性三桶边界（2026-09-29）

`raw_vulkan_init`（Bevy 上游 feature）本质是**在同一个 wgpu 设备上、经 wgpu 的 Vulkan HAL 回调启用 wgpu 类型化 feature 未暴露的额外 VK 扩展**——它**不是第二个后端**，非 Vulkan 平台（Metal/Web）直接 fallback 回普通 wgpu。它是"扩展注入路径而非独立后端"，且是这套扩展协商里最窄、最危险的一条：**不是"补齐特性"的万能补丁**——它用**两个设计承诺换一个平台特性**：跌进这条路的特性同时丢掉「跨平台」（裸 VK 在 Mac/Web 没有）和「CPU golden 可测」（写不出同源 CPU 参考）。据此把高级特性分三桶：

| 桶 | 覆盖路径 | 跨平台 | CPU golden 可测（测试脚手架） |
|---|---|---|---|
| compute-可移植（VSM/GI/OIT/粒子/froxel 体积/时序上采样/毛发布料 sim） | wgpu compute（含 EXPERIMENTAL） | ✅ | ✅ 数组进数组出，手写 Rust 参考可逐像素对数 |
| RT | wgpu ray query/pipeline（主 VK/DX12，**Metal RT 弱**） | 部分 | 半（**closure 着色可 golden；traversal/驱动 BVH 不可**） |
| raw-VK-only（SPARSE/TILED、SHADING_RATE/VRS、WORK_GRAPH，实测 wgpu 30=缺） | 仅 `raw_vulkan_init` | ❌ | ❌（驱动/硬件黑盒，无数值输出可对） |

**为什么 raw-VK 桶"没 cpu"**：CPU golden 成立的前提是算法能用同源 shader 编到 CPU target 逐像素复现。compute 效果本质数组进数组出，能复现；而 RT 遍历/驱动建 BVH、sparse 页驻留、VRS 采样率、work graph 调度都是**驱动/硬件黑盒，没有可对齐的数值输出**——要 golden 就得手写第二份实现，那本身就是漂移，违背 golden 初衷。

**结论**：`raw_vulkan_init` 必须被关进"明确非可移植、明确无 CPU golden 覆盖"的小笼子，单独配 GPU-only 抓帧 diff 的验证策略，**不得当通用兜底污染主线**。要不要为某个 raw-VK 特性付这笔"双失"代价，取决于它是否为核心野心——**大多数野心效果是 compute，落在第一桶，根本用不到这个逃生舱**。

**实装现状（对齐 committed 代码，勿误读为已接线）**：第三桶 `raw_vulkan_init` / hal 注入路径在 Prism 的 `pkg/prism_render_architecture/src/backend/` 里**零代码**——`capability.rs::Capability` 只枚举第一、二桶的 6 个能力位（`Compute` baseline + 5 个类型化 wgpu 扩展：`BindlessDescriptors`/`IndirectDrawCount`/`MeshShading`/`RayQuery`/`RayTracingPipeline`），**没有 SPARSE/TILED、SHADING_RATE/VRS、WORK_GRAPH 的能力位，也没有 `as_hal`/`create_device_from_hal`/`AdditionalVulkanFeatures` 任何 hal 下探**（实测 `git grep raw_vulkan|as_hal|create_device_from_hal|AdditionalVulkan pkg/` 零命中）。即：能力协商层已 committed 落地的是**第一、二桶**（baseline + 类型化扩展 fallback，§9 第 5 项）；第三桶是**记录在案但刻意不实装**的逃生舱设计——仅当某个 raw-VK 野心真的立项时才经 Bevy/wgpu 上游 hal 回调接入，届时须同时接受「丢跨平台 + 丢 CPU golden」并单配抓帧 diff 验证。当前无此需求，故 backend crate 不含任何 hal 代码，符合「大多数野心是 compute、用不到逃生舱」的判断。

---

## 9. 代码层破坏性重构清单（可编译可测每步）

1. **（已完成）** `pkg/prism_render_material/src/record.rs`：`MaterialShadingModel` 已删除（仅注释保留"former"字样）；`MaterialRenderClass` 已去 `Npr*/Custom*`（保留 Opaque/OpaqueTwoSided/Masked/MaskedTwoSided/Transmissive/Transparent/Additive/Volume/Hair/Water/Decal）；`GpuMaterialHeader` 已去 `shading_model`、加 `illumination`/`closure_graph_offset`/`specialization_{low,high}`/`lobe_mask`；`GpuSurfaceParameters` 已拆为 `GpuSurfaceCore`（12 字）+ 按 `LobeMask` 可选瓣 blob（`surface.rs`，见 §3.3，ABI v4）。
2. **（已完成）** `pkg/prism_render_material/src/ir.rs`：`normalize()` 不再塌缩成单 `shading_model`，产出 `NormalizedMaterial{ closure_mask, illumination, specialization_id }`；`slab_depth()` 在 DAG 上记忆化最长链求解，`> MAX_CLOSURE_SLAB_DEPTH(=4)` → `MaterialValidationError::ClosureSlabTooDeep`（Mix/Layer 封顶校验，见 §3.4）。
3. **（已完成，口径修正：择更优方案）** `pkg/prism_render_shading/src/classification.rs`：原计划"9 桶 → 按 specialization_id 动态分桶"，实施时择更务实方案——**保留 9 值 `MaterialShadingClass` 枚举**（Principled/Unlit/Subsurface/ClearCoat/Cloth/Hair/Water/Npr/Custom），但它不再是存储字段而是**正交轴的派生投影**：`classify_material_header` 据 `illumination`+`closure_mask`+`render_class` 现算桶（Water/Hair render_class 直绑，Unlit/Stylized/Custom 由 illumination 定，Lit 由主导 closure 经 `lit_class_from_closures` 投影）。保留固定枚举让着色 pass 保持 wavefront 一致（同桶像素同 permutation），材质侧则完全正交。`ShadingWorkPlan::build` 按桶分 bin 产出 `ShadingWorkItem`。
4. **（已完成）** `pkg/prism_render_shading/src/resolve.rs`：`evaluate_toon_direct` 已从"一个特例分支"提升为 `ILightResponse(Stylized)`（`evaluate_stylized_direct`，旧 toon 作为 `StylizedParams::legacy_toon` 兼容层）；描边/ramp/SDF 面阴影/rim 分别落到 `stylized.rs`（`evaluate_stylized_direct`）、`outline.rs`（`outline_id_edge`/`outline_depth_edge`/`outline_normal_edge`/`evaluate_outline`）、`face_shadow.rs`（`face_shadow_light_cosines`/`evaluate_face_shadow`）。
5. **（已完成）** `pkg/prism_render_architecture/src/backend/`：删两后端模型（`BackendId::NativeVulkan`/`BackendTier`/`registry`），收敛为单一 wgpu 后端——`capability.rs`（`Capability` 位集 + `WEBGPU_BASELINE` + `FeatureRequirement`）、`negotiation.rs`（`negotiate_features` → `EnabledFeatures`）、`fake.rs`（`FakeBackend` 仅持有 `enabled` capability 集）、`RenderBackend` trait 仅留 `capabilities()`/`supports()`/`wait_idle_for_shutdown()`。
6. **（已完成）删除 `prism_render_slang` + `prism_render_slang_abi`**（commit `211f15988`，共 ~2051 行，删除前零外部依赖）。放弃 Slang（见 §2）。
7. **（口径修正）NPR 前端已就地落在 `prism_render_shading`**：原计划的独立 crate `prism_render_npr` 被就地实现取代——`evaluate_stylized_direct`（ramp/stepped shadow/stylized specular/rim）、`outline.rs`（material-id/深度/法线三路描边）、`face_shadow.rs`（SDF 面阴影）均已在着色 crate 内落地并配 CPU golden；着色 WESL、ABI 手写 `#[repr(C)]`（§2.4）。抽取为独立 `prism_render_npr` crate 降级为可选后续项（当前不做，避免与在建 NPR 着色工作冲突）。
8. 着色器继续用 **WESL**；ABI 用手写 `#[repr(C)]` + 哈希版本 + 对齐测试兜漂移（§2.4）。不做 `.slang` 迁移。
9. **（待建 · backend lane，非材质 ABI 重构）** `pkg/prism_render_architecture/src/virtual_geometry/`：CPU 决策层 10 文件已 committed（cull/lod/page_table/raster_path/pipeline/hierarchy/bins/page_request/frame + mod，确定性、后端无关、可单测）；`mod.rs` 边界声明明确「GPU vis-buffer 软/硬光栅、物理页存储、流式 I/O 均在 backend 待建」。这是当前功能层最大出血点，需 GPU 环境落地 + 抓帧 diff 验证（本沙盒无 GPU 不可验），归渲染后端子系统 lane 推进。
10. **（待建 · TAA/上采样 lane，接线项）** reactive mask 消费端未接进两处 `taa/`（`pkg/prism_render_scene/src/shading/taa/` GPU dispatch 节点 + `pkg/prism_render_shading/src/taa/` 算法层）resolve：生产端已 committed 收口——除 transparency/particle/motion 各自标注 reactive 写入外，`motion/encode.rs` 已给出正式打包 ABI（`PackedMasks` 把 reactive/transparency/confidence unorm8 + flags 塞进 32-bit，随 `EncodedMotion` 存速度 G-buffer）；但实测两处 `taa/` committed 源均零 reactive 引用——history 锁定尚未按 mask 收紧、粒子/透明高频区仍走全局时序权重。属 TAA/时序上采样 lane 的接线收口（`prism_render_shading/src/taa/{mod,resolve}.rs` 正被并发施工，接线可能在途），非材质 ABI 重构。

> **重构收敛状态（2026-09-29）**：材质 / ABI / 后端 / Slang / NPR 侧的破坏性重构（第 1–8 项）已**全部完成并 committed**（逐条经 committed 代码复核，见各项内联证据）。剩余第 9–10 项为**跨 lane 的 GPU 后端落地 / 接线收口**，由对应子系统 agent 在其 lane 推进；本材质设计文档只如实登记状态，不越界代改并发 agent 的后端代码。

---

## 10. 诚实的固有摩擦与风险（不糊）

- **B 类降级不可消除**：RT×NPR、NPR 对 GI 只收不发、RT 反射里 NPR 降级 albedo、去噪器 NPR 单独调参。
- **NPR 一等公民无成熟范式**：PBR 侧有 UE/业界抄作业，**NPR 一等前端这条基本得自己趟**——这是全设计风险最高、参考最少的部分（顶级二次元 NPR 大作多为自研或魔改引擎，不用 UE 招牌管线）。
- **已放弃 Slang（2026-09-29 定案，见 §2）**：其唯一独占价值（shader/CPU-golden 单源）建立在"存在需长期对齐的真实 CPU 路径"上，而实测 CPU 参考只是测试脚手架、漂移面很小；成本却是一整套 shader 构建子系统 + 联网装 slangc（本机装不上）。收益/成本倒挂，直接砍掉。着色器维持 WESL，跨平台交 wgpu。将来若 compute 核心 shader/CPU 孪生真的维护痛，再评估（且 Slang 非唯一选项）。
- **strand 毛发是 AAA 最重特性之一**：card 基线务实、strand 高配可选，别一上来就 strand。
- **真正的 blocker 不在材质模型**（业界已收敛），也不在运行时后端（**wgpu 已提供跨平台运行时**），也不在着色器语言（**WESL 已够，Slang 已放弃**，见 §2）。真·出血点已随并发施工收敛为两条：**① vis-buffer 基底只完成了 CPU 决策层，GPU 光栅后端未落地（功能层面最大出血点）+ ② reactive mask 消费端未接进 TAA resolve**。原先「大量效果仍是桩、未上 frame graph」已大幅缓解——outline/halftone/kuwahara/hatching/ssgi/world_space_gi/virtual_shadow 等已成全套 Core3d dispatch 节点上图（见 §11 第 1 步）。`virtual_geometry` 已从早期 22 行 stub 演进为 ~1968 行、10 文件的 CPU 侧决策层（cull/lod/page_table/raster_path/pipeline/hierarchy/bins/page_request/frame，确定性、后端无关、可单测），但**物理页存储、vis-buffer 软/硬光栅、流式 I/O 仍在 backend 待建**（见 mod.rs 边界声明）。历史上先误判为"Vulkan 后端"、再误判为"Slang 工具链"，实为**桩接管线 + vis-buffer GPU 后端**。

### 与 UE 的关系（定位参考）

- **对齐部分**（PBR 骨架，业界共识）：GPU-driven 基底 + 后端中立（≈ UE RHI）、延迟为主 + 虚拟几何逼向延迟、透明走 forward、材质从互斥枚举转向可组合（≈ UE Substrate 的方向）。
- **主动分歧**：① 把 NPR 从补丁抬成一等前端（UE 不做）；② 把 Substrate 从近乎无界收敛成**有界 slab** 以吃满多平台。
- 结论：PBR 骨架"是同一标准"，NPR 一等 + 材质封顶策略"刻意不同"，且不同得有道理。

---

## 11. 落地路线图（跨平台 = wgpu 运行时 + WESL 着色器，每步可编译+测试绿）

**破除分析瘫痪的关键：先一条极薄端到端竖切，让竖切反过来钉死 ABI。**
**跨平台策略（2026-09-29 定案）：竖切跑在 wgpu 上（Metal/Web 就地可测），不等原生 Vulkan；着色器用 WESL（经 naga 编各后端）。已放弃 Slang（§2）。**
**优先级：真·第一优先是把已有桩效果接上 frame graph（0 行新语言层依赖）+ 补 vis-buffer 的 GPU 光栅后端（CPU 决策层 `virtual_geometry` 已落地 ~1968 行，缺物理页存储/软硬光栅/流式 I/O），不是任何 shader 语言工作。**

1. **（大部已完成）桩接管线（纯 WESL）**：把桩效果接成活管线——**outline 已落地**（`pkg/prism_render_scene/src/shading/outline/`：abi/settings/pipeline/bind_groups/resources/dispatch 共 ~767 行，Core3d compute pass 消费 vis-buffer + SSR 深度/法线，输出拷回 `scene_color`）；**halftone/kuwahara/hatching 亦已落地**（各自成 abi/pipeline/dispatch/settings 全套 Core3d 节点）。**剩余**：`light_routing` 目前只有 `mod.rs`（数据服务逻辑 + WESL + 测试），尚未包装成独立 Core3d dispatch 节点——若需作为可视 pass 呈现则补一个 dispatch，否则维持数据服务被上层消费即可。
2. **极薄竖切**：wgpu → 蒙皮(支柱四地基) → vis-buffer → material id → **一个 PBR 延迟着色 + 一个 NPR forward 着色**，点亮光照/阴影**数据服务最小版**（一盏方向光 + 一张 VSM）。
3. **材质 ABI 破坏性重构**（§9 第 1–3 步），用竖切验证正交轴 + specialization；ABI 用手写 `#[repr(C)]` + 哈希版本 + 对齐测试。
4. **粒子子系统**：优先，因为它带起 reactive mask（§5 最该早做的基底）。
5. **NPR 前端（已就地落在 `prism_render_shading`）**：描边（material id 边界白送，`outline.rs`）+ ramp/rim/SDF 面阴影（`stylized.rs`/`face_shadow.rs`）+ `evaluate_stylized_direct`；着色 WESL，CPU golden 已随附。独立 `prism_render_npr` crate 抽取为可选后续项，非阻塞。
6. **frame graph 装配节点**，逐节点在 wgpu 上实装（超 baseline 能力经 wgpu 扩展 feature 按需启用，adapter 给不出则走 fallback；无独立原生后端）。
7. **（已完成）清理 Slang 残留**：两 crate 已于 commit `211f15988` 从 workspace 删除；文档命名/注释残留已于 commit `abb661911` 清理，`rg -i slang` 现仅命中 §2 决策记录（§2 / §9.6）。
8. 毛发(card)/布料/froxel 体积按 §6.2 优先级跟进；水/植被/贴花后置。

---

## 12. 顶级产品对标矩阵（借鉴算法/形态，不抄代码）

> 三条赛道各有其"抄作业对象"。PBR 侧业界已收敛、可大量对标 UE5/主机大作；NPR 侧参考分散在二次元与影视风格化各家自研引擎；混合侧对标"风格化 PBR"与影视混合管线。**只借鉴公开算法与形态，不本地拉取任何产品源码。**

| 能力域 | PBR 对标（借什么） | NPR 对标（借什么） | 混合 / 风格化对标 |
|---|---|---|---|
| 虚拟几何 | UE5 **Nanite**：cluster DAG LOD + 软光栅微三角 + vis-buffer | 同基底复用（NPR 也吃 vis-buffer，material id 边界白送描边） | Fortnite（风格化外观 + 全 Nanite） |
| 全局光照 | UE5 **Lumen**（SDF/mesh-card 软件 RT + 硬件 RT 混合 + surface cache + 屏幕探针）、**RTXGI/DDGI** 探针 | miHoYo：GI 只收不发做**风格化底光**（防能量爆炸） | Fortnite/Valorant：风格化材质吃真 GI |
| 多光源 / 采样 | NVIDIA **ReSTIR DI/GI**（RTXDI）、**NRD**（ReBLUR/ReLAX）去噪 | NPR: **light layer 分层打光** + 主光方向驱动 | 混合场景同一 ReSTIR 预算共享 |
| 阴影 | UE5 **VSM**（虚拟页 + clipmap）、RT 阴影 | miHoYo/HoYo：**SDF 面部阴影**（独立数据、方向阈值）+ 阈值硬阴影染色 | 共享 VSM 深度页，响应分家 |
| 材质分层 | UE5 **Substrate**（无界 slab）→ 本引擎收敛成**有界 slab** | Arc Sys **Guilty Gear Xrd**：ID map + 顶点色控制、手编法线 | 风格化 PBR = über 少瓣 + Stylized illumination |
| 描边 | —（PBR 通常无描边） | Arc Sys：**inverted-hull 背面挤出**（顶点色控宽度）；miHoYo：屏幕空间深度/法线边 + material id 边；Borderlands：墨线 | 混合按物体开关描边 |
| 高光 | OpenPBR/UE über-BSDF | miHoYo/Arc：**天使环各向异性高光带**（与真实切线解耦）、MatCap、阶梯 Blinn | Team Fortress 2（Valve 论文：view-dependent + light-warp ramp + rim） |
| 后处理风格 | ACES tonemap + 物理 bloom/DoF/motion blur | Spider-Verse：**halftone/Ben-Day 点**、色差、墨线；Okami：水墨/宣纸；Kuwahara 油画 | Arcane（Fortiche）：3D 上叠 2D 手绘 FX |
| 时间步进 | TSR/DLSS/FSR/XeSS 连续时序 | Spider-Verse/Arcane：**降帧步进（on 2s/3s）**风格化 | 混合可逐物体设步进节拍 |
| 上采样 | UE5 **TSR** / DLSS / FSR2 / XeSS | 复用同一时序上采样（NPR 需 reactive mask 保锐利分段） | 同基底 |
| 参考渲染 | RED Engine **Cyberpunk RT Overdrive**（ReSTIR GI 路径追踪 + NRD）、离线路径追踪对拍 | — | — |

**借鉴纪律**：PBR 线"抄作业"到位即达 AAA；NPR 线**没有一套现成招牌管线**（顶级二次元多为自研/魔改），本设计把散落各家的算法收进"共享基底 + Stylized 前端响应"范式，这是最大增量也是最高风险（见 §10）。

---

## 13. PBR 前端 AAA 次世代高级特性清单（含性能 / 效果取舍）

> 前端 = 延迟 PBR（承载虚拟几何/RT/GI/VSM）。以下均挂 §1 共享基底，前端只取数据、叠响应。

> **关键澄清（勿误读）**：本节标题写"PBR 前端"，指的是这些高级特性由**延迟 PBR 前端以最高保真度驱动/承载**，**不是说 NPR 没有**。§1 的铁律是"基底算一次、全前端消费"——虚拟几何/GI/ReSTIR/VSM/体积/上采样等几乎全是**共享基底服务（支柱一~四）**，NPR 前端同样取用，差异只在"怎么解读这份数据"的**响应**处（§4 的 A/B/C 分类）。下面 §13.1 逐条给出 NPR 的享有程度。

### 13.1 NPR 是否享有这些高级特性？（逐条对照 §4 A/B/C）

| 特性 | NPR 享有？ | 形态 / 为什么 |
|---|---|---|
| 虚拟几何 / vis-buffer | ✅ 完全共享 | 同一份 vis-buffer；**material id 边界还给 NPR 白送描边**，NPR 反而额外获益 |
| Lumen 混合 GI | ✅ 共享（B 类降级消费） | NPR **只收不发**——收 GI 做风格化底光，不向外弹射（防能量爆炸），见 §4.2 |
| ReSTIR DI/GI | ✅ 完全共享 | 同一批授权光源与储层；NPR 用 **ramp 量化**解读，PBR 用 BRDF 积分（§4.2 A 类） |
| VSM 虚拟阴影 | ✅ 共享数据 | 同一份深度页；NPR 用**阈值硬阴影 + 染色**，另叠 **SDF 面部阴影**专属通道（§4.2 C 类） |
| 体积雾 / froxel / 体积云 | ✅ 完全共享 | 基底服务；NPR 还能做风格化体积（染色/阶梯） |
| 时序上采样（TSR/DLSS/FSR） | ⚠️ 共享但有摩擦 | 同一上采样器；NPR 的锐利分段 + 降帧步进与 TAA 打架，**必须 reactive mask 保护**（§5，头号坑） |
| RT 反射 | ⚠️ 共享但降级 | 同一 BVH；RT 里 **NPR 降级为风格化 albedo 近似**（B 类，物理追 NPR 无意义） |
| 路径追踪参考 | ➖ 偏 PBR | 次级光线里 NPR 本就降级（§0 非目标）；参考模式主要用于校准 PBR |
| 有界 slab（多瓣材质） | ➖ closure 轴，正交 | slab 是 **closure 轴**、NPR 是 **illumination 轴**，二者正交——NPR 可叠 slab，但通常用 ramp 响应而非多瓣 |
| SSS 次表面散射 | ➖ 偏 PBR（可选） | SSS 是 über 一个瓣；NPR 皮肤多用 **ramp 假 SSS**（更省更可控），需要时也能吃真 SSS |

**一句话**：**几乎全部高级特性 NPR 都享有**——基底数据是共享的，绝大多数是"同能力不同实现（A 类）"或"能用但固有降级（B 类）"；只有 slab/SSS/路径追踪这几项偏 PBR，且原因是它们本质属 closure 轴或物理正确性（对 NPR 无意义），**不是 NPR 被架构排除**。这正是"共享基底 + 前端分叉"范式的价值：NPR 不用自己重造这些高级特性，白嫖基底、只写风格化响应。


1. **虚拟几何（Nanite 级）**：cluster DAG 连续 LOD + 软件光栅（compute 画微三角，避硬件光栅小三角浪费）+ vis-buffer 延迟着色。**性能**：几何吞吐与屏幕像素解耦，海量 instance 近似常数级；代价是软光栅 compute 占用 + vis-buffer 带宽。**效果**：零 LOD 突变、亿级三角。CPU 侧决策层已落 `virtual_geometry`（~1968 行：cluster DAG cut/连续 LOD/页驻留/软硬光栅分类/帧计划），**GPU vis-buffer 光栅后端仍待建**（§11 真·出血点）。
2. **Lumen 式混合 GI**：近场 SDF/mesh-card 软件 RT + 有硬件 RT 时切硬件；surface cache 缓存表面辐照度；屏幕空间探针 final gather。**性能**：surface cache + 探针把每像素多次弹射摊薄到缓存更新；**效果**：动态间接光/软反射，无需烘焙。无 RT 平台降级 SSGI/SSR + 探针。落 §4.1 基底 + 前端 final gather。
3. **ReSTIR DI/GI**：储层时空重采样，几千动态光 + GI 一次积分。`lighting/mod.rs` 已有 `ReservoirBudget`，culling clustered 已就位（commit `d7dfa5fc1`）可接候选层。**性能**：把"每像素遍历所有光"降到"重采样少量储层"；**效果**：多光源无偏低方差。
4. **虚拟阴影 VSM**：16k 虚拟页 + directional clipmap，只渲驻留页。`virtual_shadow` 已有 residency loop（集成测试绿）。**性能**：只画可见页；**效果**：接触硬阴影到远景一致密度。
5. **有界 Substrate slab**：über-BSDF 一等 + 封顶 3–4 瓣的 layer/mix（§3.2）。**性能**：封顶避免无界 Substrate 的 shader 爆炸与弱平台撞墙；**效果**：清漆/车漆/多层皮肤够用。
6. **次表面散射 SSS**：预积分皮肤（Penner）+ 可分离 SSS（Jimenez）/ Burley 归一化扩散。**性能**：屏幕空间可分离卷积 << 真体积；**效果**：皮肤/蜡/玉。属 über 瓣，非子系统（§6.2 档 3）。
7. **RT / SSR 反射**：有 RT 走硬件反射 + NRD 去噪；否则 SSR + 屏外探针兜底。**效果**：镜面/湿地/金属。
8. **体积雾 + froxel + 体积云**：froxel 光照体（compute 可移植桶，§8.1）。**性能**：froxel 分辨率可调；**效果**：光轴/大气/云层。
9. **时序上采样**：TSR/DLSS/FSR2/XeSS，配 motion vector + reactive mask（§5，最该早做的基底）。**性能**：内部低分辨率 + 时序重建；**效果**：4K 级清晰度。
10. **路径追踪参考模式**：ReSTIR GI + NRD 的离线对拍模式（对标 Cyberpunk RT Overdrive），用于校准实时管线偏差。RT 桶（§8.1），不可 CPU golden。

---

## 14. NPR 前端 AAA 次世代高级特性清单（顶级二次元 / 风格化）

> 前端 = NPR 风格化前端（描边/ramp/SDF 面阴影/风格化高光/rim/风格化 post）。核心立场：**NPR 是 illumination=Stylized 轴 + 专属通道**，不是独立管线；与 PBR 共享几何/光照数据/VSM/GI，只在"怎么解读光照数据"处分家（§4）。

### 14.1 着色（对标 miHoYo / Arc System Works / Valve TF2）

1. **Ramp 量化漫反射**：per-material shadow ramp 贴图（明→暗分段），NdotL 采样 ramp 而非积分。**性能**：一次贴图采样 << BRDF 积分；**效果**：干净二分/三分调子（原神/星穹铁道）。
2. **SDF 面部阴影**：独立单通道 SDF 图存"该像素在光转到某角度时进阴影"的阈值，主光方位角驱动——解决传统法线阴影在脸上"脏"的老问题。**独立数据源叠加**（§4.2 C 类专属通道）。**效果**：鼻侧/刘海阴影随光平滑扫过，永远干净。
3. **风格化高光（天使环）**：各向异性高光带**可与真实切线解耦**，手动摆位做发丝"天使环"；MatCap 补金属/宝石；阶梯化 Blinn。**效果**：二次元发/眼高光。
4. **Rim / fresnel 边光**：视角驱动描亮轮廓，可按光方向偏置。**效果**：角色与背景分离。
5. **手编法线 + 顶点色控制**（对标 Guilty Gear Xrd）：平滑法线稳定描边与高光走向；顶点色/ID map 驱动描边宽度、区域高光开关、"假"高光摆位。**效果**：手绘级可控性。

### 14.2 描边（对标 Arc Sys / miHoYo / Borderlands）

6. **Inverted-hull 背面挤出**：沿法线外扩背面成描边壳，顶点色控每处宽度、随距离/FOV 补偿。**性能**：一趟额外 draw；**效果**：稳定粗描边（卡通/格斗）。
7. **屏幕空间边缘检测**：深度 + 法线 + **vis-buffer material id 边界（白送）** 提边。**性能**：一趟后处理 compute；**效果**：内部结构线、材质分界线。
8. **混合描边**：外轮廓走 inverted-hull、内部线走屏幕空间，二者互补。

### 14.3 风格化后处理（对标 Spider-Verse / Arcane / Okami）

9. **Halftone / Ben-Day 点 / 网点**：按亮度控点密度做印刷风阴影（蜘蛛侠）。
10. **降帧步进（on 2s/3s）**：角色动画/特效按 12/8 fps 步进，与 60fps 背景混排（蜘蛛侠/Arcane 影视感）。**实现**：逐物体设步进节拍，需 motion vector 特判防 TAA 抹掉。
11. **Kuwahara / 油画滤镜**：保边匀色做手绘油画感。
12. **墨线 / 笔刷 / 水墨**（Okami 宣纸、Borderlands 墨线）：沿边缘叠笔刷 alpha、纸纹叠加。
13. **风格化 bloom / 色差 / 分级**：夸张辉光 + 边缘色散 + 风格化 LUT。

### 14.4 NPR 性能 / 效果总纲

- **省**：ramp/SDF/描边多为一次贴图采样或一趟后处理，**比物理积分更省**；NPR 天然是"廉价但要美术精调"。
- **贵在带宽与美术管线**：SDF 面图、ID map、per-material ramp 是额外资产；顶点色/手编法线需 DCC 侧配套。
- **与 TAA/上采样的固有摩擦**（§5）：锐利分段与步进动画和时序累积打架，必须 reactive mask 保护——这是 NPR 上时序管线的头号坑。

---

## 15. 混合前端高级特性（管线级混合，非叠加各算）

> 混合的价值在**管线级**：同一帧不同像素/物体走不同前端，却共享同一套光/影/GI/RT——所以风格切换处不断裂。这是"各前端各自重算高级特性"给不了的。

1. **逐像素 material id + tile 分类路由**（§7）：GPU 把像素按 material id 分 tile，路由到延迟 PBR / forward+ / NPR / 自定义前端。**性能**：tile 分类 + 前端 specialization，避免全屏跑所有前端。
2. **共享 GI / 阴影跨风格连贯**：NPR 角色与 PBR 场景吃**同一份** Lumen 间接光 + VSM 阴影 → 卡通角色落在写实场景里阴影方向/底光一致，不"贴纸感"。
3. **选择性 NPR**：写实世界里对特定角色/道具开 Stylized，其余 PBR（对标影视混合、卡通角色进实景）。
4. **风格化 PBR 中间态**（对标 Fortnite / Valorant / Overwatch / Sea of Thieves）：über-BSDF 少瓣 + `illumination=Stylized` 的 ramp/rim 轻叠加 + 真 GI/Nanite/Lumen——既非纯物理也非纯 toon，是产量最大的商业中间带。本架构天然覆盖（正交轴自由组合）。
5. **影视级混合**（对标 Spider-Verse/Arcane）：3D 基底 + 2D 手绘 FX/线 + 降帧步进，逐物体调节拍。

**为什么管线级混合优于单独管线**：单独管线会让 NPR/PBR 各自重造 GI、阴影、RT、上采样——没一条能到 AAA 且风格接缝断裂。共享基底 + 前端分叉是"四者皆顶级 + 可混合 + 多平台"的唯一解（§1 一句话）。

---

## 16. 性能 / 效果 / 易用性权衡总表

| 特性 | 桶（§8.1） | 性能要点 | 效果上限 | 易用性 / 风险 |
|---|---|---|---|---|
| 虚拟几何 Nanite | compute-可移植 | 几何与像素解耦；软光栅 + vis-buffer 带宽 | 亿级三角、零 LOD pop | CPU 决策层已落地（~1968 行），缺 GPU 光栅后端，最大出血点 |
| Lumen 混合 GI | compute + RT | surface cache 摊薄弹射；RT 加速可选 | 动态无烘焙 GI | 高复杂度；无 RT 降级 SSGI |
| ReSTIR DI/GI | compute + RT | 储层重采样代替全光遍历 | 千级动态光低方差 | 需去噪配套；基底已就位 |
| VSM | compute-可移植 | 只渲驻留页 | 全程一致密度阴影 | residency 已落 |
| 有界 slab | shader 特化 | 封顶防 variant 爆炸 | 清漆/车漆/多层皮 | 封顶是刻意取舍（vs UE 无界） |
| SSS | compute-可移植 | 屏幕空间可分离卷积 | 皮肤/蜡/玉 | über 瓣，非子系统 |
| 体积雾/云 | compute-可移植 | froxel 分辨率可调 | 光轴/大气/云 | froxel 内存 |
| 时序上采样 | compute-可移植 | 低分辨率内部渲染 | 4K 级 | 依赖 motion+reactive mask |
| NPR ramp/SDF/描边 | compute-可移植 | 贴图采样/单趟后处理，**比 PBR 省** | 顶级二次元 | 美术资产 + DCC 管线成本高；无成熟范式（最高风险） |
| 降帧步进/halftone/墨线 | compute-可移植 | 后处理开销小 | 影视风格化 | 与 TAA 摩擦，需 reactive mask |
| 混合 tile 路由 | compute-可移植 | tile 分类 + 前端特化 | 风格无缝共存 | 依赖 material id 基底 |
| RT 反射/路径追踪参考 | RT | 硬件 BVH；Metal RT 弱 | 物理镜面/离线对拍 | 跨平台部分覆盖，不可 CPU golden |
| SPARSE/VRS/work graph | raw-VK-only | 平台特化收益 | 特定场景增益 | **双失**（无跨平台 + 无 CPU golden），关小笼子（§8.1） |

**总原则**：野心效果绝大多数落 compute-可移植桶（跨平台 + 可 CPU golden 对拍），RT 桶部分可测，raw-VK 桶严格隔离。NPR 整体比 PBR 省算力但吃美术管线；混合的开销主要在 tile 路由与前端特化，换来风格接缝连贯——这是三线皆 AAA 的性价比最优点。

---

## 附录 A：术语与轴速查

- **前端 (frontend)**：延迟 PBR / forward+ / NPR / 自定义。决定"怎么组织 pass 与着色"。
- **illumination 轴**：Lit / Stylized / Unlit / Custom。决定"用什么方式解读光照数据"。正交于前端与 closure。
- **closure**：über-BSDF（一等）/ 有界 slab / 毛发专用。决定"表面 BSDF 长什么样"。
- **子系统**：有自己几何+sim+特殊渲染的横切模块（毛发/粒子/体积/水/布料sim）。
- **基底服务**：大家都消费的公共设施（OIT/motion+mask/透射/去噪/光照数据）。**不是**子系统。
