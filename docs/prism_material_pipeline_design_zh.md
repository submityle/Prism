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
- 因此本文所有"Vulkan 优先/VK 先"应读作 **"wgpu（已有运行时）+ WESL（已有语言层）"**。**不存在独立的原生 Vulkan 后端**——只有一个 wgpu 后端；超出 `WebGPU` baseline 的能力（bindless / mesh shader / ray query / 多重间接 count / RT）都是 **opt-in 的 wgpu 扩展 feature**（wgpu 内部映射到底层 Vulkan/Metal/D3D12 扩展）。adapter 报告支持哪些 feature，device 启用请求的子集，上层特性声明所需 capability；adapter 给不出的 capability 走 **fallback**，而不是切后端。
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
   │  支柱四: 可变形几何/模拟(蒙皮父级 → 布料/毛发/粒子 sim) → gpu_scene       │
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

### 3.3 GPU 数据结构（拆胖结构）

- `GpuSurfaceParameters`（18 字段胖结构，每像素全带）拆成 **über 核心参数（紧凑）+ 按瓣可选 blob**。specialization variant 决定读哪些 blob，不再无脑塞 18 字段。
- `GpuMaterialHeader`：`shading_model` 字段删除；新增 `illumination`、`closure_graph_offset`、`specialization_id`。`closure_mask` 保留（RT/分类用）。
- 这些结构**手写 `#[repr(C)]`**，但由**结构哈希驱动 `MATERIAL_ABI_VERSION`** + CI 对齐测试兜漂移（改结构 → 哈希变 → 版本自动 bump → 测试红）。不引入反射生成器（见 §2）。

---

## 4. 支柱二：光照/阴影 数据服务 + 各前端响应（比材质更关键）

**核心原则：共享的是数据/加速结构，不是响应。** "一致"的正确定义 = **同一批授权光源驱动，外观按各自风格解读**（NPR 本就该长得不一样），不是物理外观相同。

### 4.1 基底共享（算一次）

光源 SoA + cluster 剔光 / VSM 深度页 / RT 阴影·反射结果 / GI·IBL·GTAO irradiance / BVH。

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

### 4.3 WESL/WGSL 表达

`ILightResponse.shade(ShadingCtx, LightSample, Closure)` 在 WESL 里落成一组按前端选择的着色函数（Lit/Stylized 各一份），经 `import` 共享同一份光源/阴影数据结构。前端切换 = 换着色函数（编译期特化或运行时分支），光照数据零改动。

---

## 5. 支柱三：跨切面基底（解一次，全前端共享）

- **motion vector + reactive/stencil mask**：粒子/透明/NPR/毛发**全和 TAA 打架**，这是**最该早做**的基底。粒子子系统会逼你先把它立起来（§11）。
- **OIT**：透明 + 粒子 + 毛发共同消费的公共设施（**不是**并列子系统）。
- **RT 去噪**：共享去噪器，每前端调参（NPR 见 §4.2 B 类）。
- **能力分层 / fallback**：见 §8。

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

### 6.2 子系统注册表（四档，钉死）

| 档 | 成员 | 判据 | 落地优先级 |
|---|---|---|---|
| **0 地基** | 可变形几何(蒙皮/morph) | 所有形变子系统父级 | **最先** |
| **1 一等子系统** | 粒子·毛发·体积·布料sim | 自己的几何+sim+特殊渲染 | 粒子最先(带起 reactive mask) |
| **2 后置留槽** | 水/海洋·植被·贴花 | 是子系统但重/非通用 | 竖切不碰 |
| **3 陷阱·别做子系统** | SSS/sheen/清漆(=closure)、大气(=光照数据服务)、OIT/motion(=基底服务) | 无几何无sim / 是公共设施 | — |

**判据**：需要自己的「几何+sim+特殊渲染(透射/OIT/RT代理)」才是子系统；只是着色变化 → closure；大家都消费 → 基底服务。**布料要拆**：sim 是子系统、sheen 是 über 一个瓣。

### 6.3 毛发 mini 支柱（子系统内部同构范例）

```
共享基底(PBR/NPR 都吃): strand几何(连续LOD→card, 禁硬切换) + guide-strand sim
                       + 共享 deep-transmittance 自阴影服务 + 插进共享 OIT
分叉响应(C类, 真分家):  PBR: Marschner/Chiang R/TT/TRT + dual-scattering
                       NPR: 风格化各向异性高光带 + ramp + 阴影偏移(可与切线解耦=天使环)
fallback:              strand 高配, card 基线; RT 反射里毛发用 proxy 或排除
```

**所有一等子系统内部都长这个样**（共享基底 + PBR/NPR 分叉响应），架构一致性拉满。

---

## 7. 前端（并存，皆一等）

- **延迟 PBR 前端**：承载虚拟几何(Nanite 级)/RT/GI/VSM。虚拟几何与 forward+ 互斥，**这才是逼向延迟的真原因，不是 PBR 本身**。
- **forward+ 前端（一等公民）**：透明**强制** forward，所以这条路本就必建；NPR 复用它、粒子挂它，**不是额外成本**。
- **NPR 风格化前端**：描边 / ramp / SDF 面阴影 / 风格化高光 / rim / 风格化 post。**vis-buffer 的 material id 边界白送描边**。
- **自定义前端槽**：项目注入 `illumination=Custom`，走自定义 WESL closure + 自定义 pass。
- **混合 = 逐像素 material id + tile 分类路由**：GPU 把像素按 material id 分 tile，路由到对应前端。因共享光/影/GI 而连贯——这是"管线级混合"的具体机制，也是它比"各前端各算高级特性"优越的地方。

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

---

## 9. 代码层破坏性重构清单（可编译可测每步）

1. `pkg/prism_render_material/src/record.rs`：删 `MaterialShadingModel`；`MaterialRenderClass` 删 `Npr*/Custom*`；`GpuMaterialHeader` 去 `shading_model`、加 `illumination`/`closure_graph_offset`/`specialization_id`；`GpuSurfaceParameters` 拆核心+瓣 blob。
2. `pkg/prism_render_material/src/ir.rs`：`normalize()` 停止塌缩成单 `shading_model`，产出 `ClosureGraph`+`illumination`+`specialization_id`；`Layer/Mix` 加封顶校验。
3. `pkg/prism_render_shading/src/classification.rs`：`MaterialShadingClass` 固定 9 桶 → 按 `specialization_id`/tile 动态分桶；`classify_material_header` 重写。
4. **（已完成）** `pkg/prism_render_shading/src/resolve.rs`：`evaluate_toon_direct` 已从"一个特例分支"提升为 `ILightResponse(Stylized)`（`evaluate_stylized_direct`，旧 toon 作为 `StylizedParams::legacy_toon` 兼容层）；描边/ramp/SDF 面阴影/rim 分别落到 `stylized.rs`（`evaluate_stylized_direct`）、`outline.rs`（`outline_id_edge`/`outline_depth_edge`/`outline_normal_edge`/`evaluate_outline`）、`face_shadow.rs`（`face_shadow_light_cosines`/`evaluate_face_shadow`）。
5. **（已完成）** `pkg/prism_render_architecture/src/backend/`：删两后端模型（`BackendId::NativeVulkan`/`BackendTier`/`registry`），收敛为单一 wgpu 后端——`capability.rs`（`Capability` 位集 + `WEBGPU_BASELINE` + `FeatureRequirement`）、`negotiation.rs`（`negotiate_features` → `EnabledFeatures`）、`fake.rs`（`FakeBackend` 仅持有 `enabled` capability 集）、`RenderBackend` trait 仅留 `capabilities()`/`supports()`/`wait_idle_for_shutdown()`。
6. **（已完成）删除 `prism_render_slang` + `prism_render_slang_abi`**（commit `211f15988`，共 ~2051 行，删除前零外部依赖）。放弃 Slang（见 §2）。
7. **（口径修正）NPR 前端已就地落在 `prism_render_shading`**：原计划的独立 crate `prism_render_npr` 被就地实现取代——`evaluate_stylized_direct`（ramp/stepped shadow/stylized specular/rim）、`outline.rs`（material-id/深度/法线三路描边）、`face_shadow.rs`（SDF 面阴影）均已在着色 crate 内落地并配 CPU golden；着色 WESL、ABI 手写 `#[repr(C)]`（§2.4）。抽取为独立 `prism_render_npr` crate 降级为可选后续项（当前不做，避免与在建 NPR 着色工作冲突）。
8. 着色器继续用 **WESL**；ABI 用手写 `#[repr(C)]` + 哈希版本 + 对齐测试兜漂移（§2.4）。不做 `.slang` 迁移。

---

## 10. 诚实的固有摩擦与风险（不糊）

- **B 类降级不可消除**：RT×NPR、NPR 对 GI 只收不发、RT 反射里 NPR 降级 albedo、去噪器 NPR 单独调参。
- **NPR 一等公民无成熟范式**：PBR 侧有 UE/业界抄作业，**NPR 一等前端这条基本得自己趟**——这是全设计风险最高、参考最少的部分（顶级二次元 NPR 大作多为自研或魔改引擎，不用 UE 招牌管线）。
- **已放弃 Slang（2026-09-29 定案，见 §2）**：其唯一独占价值（shader/CPU-golden 单源）建立在"存在需长期对齐的真实 CPU 路径"上，而实测 CPU 参考只是测试脚手架、漂移面很小；成本却是一整套 shader 构建子系统 + 联网装 slangc（本机装不上）。收益/成本倒挂，直接砍掉。着色器维持 WESL，跨平台交 wgpu。将来若 compute 核心 shader/CPU 孪生真的维护痛，再评估（且 Slang 非唯一选项）。
- **strand 毛发是 AAA 最重特性之一**：card 基线务实、strand 高配可选，别一上来就 strand。
- **真正的 blocker 不在材质模型**（业界已收敛），也不在运行时后端（**wgpu 已提供跨平台运行时**），也不在着色器语言（**WESL 已够，Slang 已放弃**，见 §2）。真·出血点是 **① 大量效果仍是桩、未上 frame graph（功能层面的最大出血点）+ ② 还不存在的 vis-buffer 基底（`virtual_geometry` 仅 22 行 stub）**。历史上先误判为"Vulkan 后端"、再误判为"Slang 工具链"，实为**桩接管线 + vis-buffer**。

### 与 UE 的关系（定位参考）

- **对齐部分**（PBR 骨架，业界共识）：GPU-driven 基底 + 后端中立（≈ UE RHI）、延迟为主 + 虚拟几何逼向延迟、透明走 forward、材质从互斥枚举转向可组合（≈ UE Substrate 的方向）。
- **主动分歧**：① 把 NPR 从补丁抬成一等前端（UE 不做）；② 把 Substrate 从近乎无界收敛成**有界 slab** 以吃满多平台。
- 结论：PBR 骨架"是同一标准"，NPR 一等 + 材质封顶策略"刻意不同"，且不同得有道理。

---

## 11. 落地路线图（跨平台 = wgpu 运行时 + WESL 着色器，每步可编译+测试绿）

**破除分析瘫痪的关键：先一条极薄端到端竖切，让竖切反过来钉死 ABI。**
**跨平台策略（2026-09-29 定案）：竖切跑在 wgpu 上（Metal/Web 就地可测），不等原生 Vulkan；着色器用 WESL（经 naga 编各后端）。已放弃 Slang（§2）。**
**优先级：真·第一优先是把已有桩效果接上 frame graph（0 行新语言层依赖）+ 补 vis-buffer 基底（`virtual_geometry` 仅 22 行 stub），不是任何 shader 语言工作。**

1. **桩接管线（第一优先，纯 WESL）**：把已有桩效果接成活管线，**outline 先行**（消费端 composite 已活、零新基建、ROI 最高）——抄 SSR 先例：params ABI → per-view line target → compute pipeline → Core3d node → composite mix。之后 light_routing → halftone/kuwahara。
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


1. **虚拟几何（Nanite 级）**：cluster DAG 连续 LOD + 软件光栅（compute 画微三角，避硬件光栅小三角浪费）+ vis-buffer 延迟着色。**性能**：几何吞吐与屏幕像素解耦，海量 instance 近似常数级；代价是软光栅 compute 占用 + vis-buffer 带宽。**效果**：零 LOD 突变、亿级三角。落 `virtual_geometry`（当前仅 stub，§11 真·出血点）。
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
| 虚拟几何 Nanite | compute-可移植 | 几何与像素解耦；软光栅 + vis-buffer 带宽 | 亿级三角、零 LOD pop | 需重写 `virtual_geometry`（现 stub），最大出血点 |
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
