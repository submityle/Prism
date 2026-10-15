# Prism 渲染引擎 — 材质与管线完整架构设计（v3 / 与实现对齐版）

> **状态**：架构基准（Architecture Baseline）。允许破坏性重构；本文同时描述「实现已经成立的部分」与「架构已定、尚未落地的部分」，两者分列，不混写。
> **版本**：v3 · 2026-10-15。上一版 v2 内部标注 2026-09-30，其材质 ABI、闭包图、前端与调度描述已与实现不符；本次同步补上 `ShadingPlan` / `FeatureCatalog` / 水体 GPU 能力门禁的现状。
> **代码基准**：`pkg/` 工作区实测，HEAD = `679c476b8`（2026-10-15）。凡标注 `code-truth` 的论断均给出 `文件:行`，可直接复核；文件级索引见 **附录 C**。
> **后端**：运行时 API 层 = **wgpu**；着色语言层 = **WESL → naga → WGSL / SPIR-V / MSL / DXIL**。没有独立的原生 Vulkan 后端（§0.5）。
> **数值红线**：不引入任何神经 / ML 推理路径；布局逐字节相等、算术 ≤1 ULP（§3.5、§13.5），运行时只有一条执行路径（§1.1）。
> **本文不重复**：材质侧的破坏性重构细节 → `docs/prism_material_pipeline_refactor_zh.md`（v10）；OpenPBR 参数逐条功能对照 → `docs/prism_openpbr_pipeline_function_zh.md`；完成度核验 → `docs/prism_material_pipeline_refactor_verification_zh.md`；四层收口方案 → `docs/refactoring-dev/00-overview_refactor_zh.md`。

## 0. 设计目标与非目标

### 0.1 目标（硬约束）

| # | 目标 | 当前落点（code-truth） |
|---|---|---|
| G1 | **PBR / NPR / 自定义响应与 MaskBlend 组合同时是一等公民**，不是"一条主线 + 特例" | 响应由 `ResponseProgram::{Pbr,Npr,Custom}` 表达，组合由 `WaterShadingPlan::MaskBlend` 表达（`pkg/prism_render_architecture/src/water/mod.rs`） |
| G2 | **跨平台**：macOS / Windows / Linux / Web 用同一套代码与同一套着色器 | 单层 wgpu + WESL；能力以 `Capability` 位集建模（§0.5、§8） |
| G3 | **高级特性不留白**：GI/RT、虚拟阴影、虚拟几何、时序超分、动态光、粒子、透明、毛发、布料、体积、水面 | 共享服务由 `FeatureCatalog` 的硬需求闭包派生；水体 9 项、体积 7 项服务 ID 由 `render_fabric::builtin_catalog` 统一声明 |
| G4 | **NPR 不是降级路径**：NPR 必须与 PBR 享有同一份几何/光照/阴影/时序基底 | `ResponseProgram::Npr` 与 PBR 使用同一个水体服务闭包；闭包来自 `FeatureDefinition.requires_services`，不由响应枚举缩减 |
| G5 | **单一事实源**：布局/掩码/默认值/键不允许多处手写 | `pkg/prism_material_schema/schema/surface.toml` → `prism-material-codegen` → Rust 布局 + WESL 解包器（§3.5） |
| G6 | **不设全局硬顶**：瓣数、纹理槽、闭包层数不用常数封顶 | 闭包 slab 上限与纹理槽上限已删除（§2.3）；`SURFACE_MAX_WORDS = 104` 是解码缓冲尺寸，不是"最多能表达多少" |
| G7 | **能力协商确定性降级**：缺能力时行为可预测，不静默串台 | 载入期 `used` / `required` 摘要 + 硬错误（§3.6）；运行时能力位集 `Capability`（§8.1） |
| G8 | **可测试性可分层**：跨平台可测的部分与不可测的部分显式分桶 | 三桶模型（§8.2）：compute 可移植 / RT / raw-VK-only |

### 0.2 非目标（主动不做）

1. **不做无界 Substrate 式闭包图**：作者层可以任意分层/混合，但**运行时**只携带一个定宽参数块与一个位掩码，不做运行期图遍历（§2.2）。
2. **不为 NPR 承诺 RT 物理正确性**：NPR 只"收"共享光照/阴影结构，不"发"新的物理光照；在路径追踪参考里 NPR 退化为风格化反照率（§4.2）。
3. **不强制 PBR 与 NPR 外观统一**：风格化是正交轴（`Illumination`），不是"PBR 的近似"。
4. **不引入神经渲染 / ML 推理**：全部为经典数值方法。
5. **不做离线烘焙管线**：另有专文（Phase 6）。
6. **不做"每个前端一套管线"**：不接受为 NPR/自定义复制几何、阴影、GI、时序代码。
7. **不做 raw Vulkan 独占特性的跨平台承诺**：见 §8.2 第三桶——用了就放弃跨平台与 CPU golden，属于显式逃生舱。

## 0.5 后端策略：单一 wgpu + 单一着色语言，能力三档协商

**结论（2026-09-29 修正，之后未变）**：本项目的目标是**跨平台**，不是 "Vulkan 优先"。早期文档里的"Vulkan 优先 / Vulkan 专用后端"来自一个错误假设——以为 macOS 原生支持 Vulkan。macOS 没有原生 Vulkan，只能经 MoltenVK 在 Metal 之上翻译，因此 Vulkan 不应被当作首要或唯一后端。正确抽象是**两个彼此独立的层**：

- **运行时 API 层**：wgpu。所有平台同一份提交代码。
- **着色语言层**：WESL（单一源码）→ naga → 各目标（WGSL / SPIR-V / MSL / DXIL）。

在此之上，能力协商分三档：

| 档 | 内容 | 缺能力时的行为 | 现状（code-truth） |
|---|---|---|---|
| ① 基线 | WebGPU 保证的 compute + storage buffer | 永远可用 | `CapabilitySet::WEBGPU_BASELINE = Capability::Compute.bit()`（`pkg/prism_render_architecture/src/backend/capability.rs:87`） |
| ② 类型化扩展 | bindless、multi-draw indirect count、mesh shade、ray query、RT pipeline —— 都是 **wgpu 扩展 feature**，由适配器决定是否可用 | **不换后端**，按声明降级；缺 `required` 能力 → 载入期硬错误（§3.6） | `Capability` 六值 + `FeatureRequirement` 校验（`backend/capability.rs:31-48`、`backend/negotiation.rs`） |
| ③ 原生逃生舱 | Bevy `raw_vulkan_init`：`wgpu::hal::vulkan` 直连（`create_device_from_hal`，能力记录到 `AdditionalVulkanFeatures`） | 只在明确需要 Vulkan 独占能力时启用；**代价是失去跨平台与 CPU golden** | Prism 侧**未接线**：`pkg/prism_render_architecture/src/backend/` 只有 `capability.rs` / `negotiation.rs` / `fake.rs` / `mod.rs`，`pkg/` 下 `raw_vulkan_init` / `create_device_from_hal` / `AdditionalVulkanFeatures` **0 命中**。它依赖的 Bevy 能力在本仓 fork 里**存在且可达**（`crates/bevy_render/src/renderer/raw_vulkan_init.rs:71,123,138`，由 `bevy_anti_alias` 的 DLSS 路径消费，feature 非默认开启 `crates/bevy_render/Cargo.toml:26`、根 `Cargo.toml:613`） |

第③档刻意只保留契约、不落代码：一旦落地，它就是"第二个后端"，会同时破坏 G2（跨平台）与 §8.2 的可测性分桶。任何使用它的特性必须登记在 §8.2 第三桶里。

## 1. 顶层架构

### 1.1 三层，只有一个运行时执行路径

```
作者层 (authoring)      OpenPBR / MaterialX / 节点图 / archetype 模板
                        └─ pkg/prism_render_material_authoring
                                   │  分层解析：任意层数 → 纹理空间混合（无 N 层硬顶）
烘焙层 (bake)           └─ pkg/prism_render_material (bake) + pkg/prism_material_schema
                                   │  产出：一个定宽 über 参数块 + 位掩码 + 特化键
运行层 (runtime)        ┌─ 解包一次 (material_unpack.wesl，由 schema 生成)
                        ├─ 分类一次 (per-pixel / per-tile material id → pipeline)
                        └─ 响应分家 (PBR / NPR / 混合 / 自定义 = 同一个基底的四种解读)
```

三层的边界就是**信息损失点**，必须显式：

- 作者层是**开放**的：可以有任意多层、任意混合、任意自定义节点。
- 烘焙层的产出是**定宽**的：一个参数块（§3.4）+ 一个瓣位掩码（§3.3）+ 一个特化键（§3.1）。开放性的代价在烘焙期一次付清。
- 运行层**不做图遍历**、不做运行期分层：`record.rs:129-132` 明确写下 *"The runtime is graph-free … there is no serialized closure IR"*。

### 1.2 三个正交维度（替代 v2 的互斥枚举）

v2 把"PBR / NPR / 自定义"写成一个互斥的 `MaterialShadingModel` 枚举。这必然膨胀成 `NprOpaque / NprMasked / NprTransmissive …` 的组合爆炸，且让"透明 NPR"无法表达。现在的模型是**三个彼此独立的维度**：

| 维度 | 类型 | 取值 | 状态（code-truth） |
|---|---|---|---|
| **照明轴** | `Illumination` (`#[repr(u32)]`) | `Lit=0` / `Stylized=1` / `Unlit=2` / `Custom=3` | `pkg/prism_render_material/src/axis.rs:24-37,46` |
| **渲染类**（混合/光栅路由） | `MaterialRenderClass` | `Opaque, OpaqueTwoSided, Masked, MaskedTwoSided, Transmissive, Transparent, Additive, Volume, Hair, Water, Decal`（**11 值**） | `pkg/prism_render_material/src/record.rs:51-64`，反查 `from_u32` `record.rs:72` |
| **响应与组合计划** | `ResponseProgram` + `WaterShadingPlan` | `Pbr` / `Npr` / `Custom`；`Single` / `MaskBlend` | `pkg/prism_render_architecture/src/water/mod.rs` |

`MaterialShadingModel` **已删除**。`Npr*` / `Custom*` 渲染类也**已从枚举里删除**——`record.rs:45-49` 的注释给出理由：它们是"混合 × 照明"两个正交轴被压平后的产物，压平后就无法表达"NPR + 透明"这类合法组合。风格属于照明轴，不属于渲染类。

推论（也是设计契约）：
- "NPR 材质" = `Illumination::Stylized` + 任意 `MaterialRenderClass`；
- "混合计划" = 同一像素对两条响应臂做连续 `MaskBlend`，由常驻组合 pass 或 fused 内建入口执行；二值区域选择仍由 `Illumination` 掩码路由（§4.3、§7.2）；
- 特化键里**同时**携带这三轴，因此不存在"NPR 版本缺少某个渲染类"的缺口。

### 1.3 共享的是数据与加速结构，不是响应

最容易走错的一步是把"共享基底"理解成"共享着色代码"。共享的正确对象是**数据、加速结构、时序历史与可见性**——它们对所有前端逐字节相同；差异只允许出现在"拿到光照之后怎么解读"。当前实现把它写成了**可编译检查的数据**：

| 共享基底服务（9 项） | 含义 | 位 |
|---|---|---|
| `VIRTUAL_GEOMETRY` | 虚拟几何 cluster LOD / 软光栅 | 1<<0 |
| `HYBRID_GI` | Lumen 式混合全局光照 | 1<<1 |
| `RESTIR` | ReSTIR 直接/间接光重采样 | 1<<2 |
| `VIRTUAL_SHADOW` | 虚拟阴影贴图 VSM | 1<<3 |
| `FROXEL_VOLUME` | Froxel 体积（水下散射、god ray） | 1<<4 |
| `RT_REFLECTION` | 光追反射（SSR miss 回退） | 1<<5 |
| `RT_CAUSTICS` | 光追/光子焦散 | 1<<6 |
| `PATH_TRACING_REF` | 离线路径追踪参考（标定用） | 1<<7 |
| `TEMPORAL_UPSAMPLE` | 时序超分 / 抗锯齿历史 | 1<<8 |

服务闭包由 `pkg/prism_render_architecture/src/render_fabric/builtins.rs` 的 `FeatureCatalog` 派生：水体根需要 9 个稳定服务 ID，体积根需要另一组全局不冲突的 7 个服务 ID；`FeatureSelection::resolve` 为根集合冻结 provider 选择。资源/访问/生命周期仍由选中的 pass 模板和权威帧图拥有，内建目录不伪造尚未绑定的资源写入。

于是"**NPR 是不是漏了某个高级特性**"这个问题有编译期答案：不会漏，因为 PBR/NPR/Custom 响应与 `MaskBlend` 计划都从同一个水体根闭包读取服务集合。

### 1.4 与"多条管线"的差别

| 做法 | 几何/阴影/GI/时序 | 着色 | 结论 |
|---|---|---|---|
| 多条管线 | 复制多份 | 各写一套 | 不接受：NPR 永远落后、特性同步成本随响应数增长 |
| 一条管线 + 特例分支 | 1 份 | if/else 散落 | 不接受：不可特化、分支成本常驻 |
| **一条执行路径 + 正交响应**（本文） | **1 份** | **同一入口，按特化键选择响应函数** | 采用 |

### 1.5 crate 地图（`pkg/`，2026-10-14 实测）

| 层 | crate（`src` 行数） | 职责 |
|---|---|---|
| 材质 | `prism_render_material` (40894) | 运行时 ABI：`record.rs` / `surface.rs` / `axis.rs` / `ir.rs` / 纹理编解码族 / `bake` / `openpbr` |
| 材质 | `prism_material_schema` (2431) | **单一事实源**：`schema/surface.toml` + `prism-material-codegen`（Rust 布局 & WESL 生成） |
| 材质 | `prism_material_pcm` (3319) | **PCM**：扩展点统一注册表（9 类 Kind）+ 载入期协商 |
| 材质 | `prism_render_material_authoring` (1895) | 作者层图的解析 / 求值 / 导入导出 |
| 材质 | `prism_render_material_gpu` (431) | 压缩纹理硬解 oracle（BC/ETC2/ASTC 的 GPU 真值），**不是** 材质头 ABI |
| 着色 | `prism_render_shading` (131510) | 材质求值、分类、NPR/风格化、GI(257 文件)、阴影(19)、AO、后处理族 |
| 场景 | `prism_render_scene` (152228) | GPU Scene 与渲染集成（`shading/` 438 文件为最大子目录） |
| 地基 | `prism_render_architecture` (401871) | 后端无关契约：frame graph、PSO 缓存、可见性、虚拟资源、各子系统契约 |
| 地基 | `prism_render_app` / `prism_render_driver` / `prism_render_graph` / `prism_render_gpudriven` / `prism_render_visibility` / `prism_render_runtime` | 应用装配、RHI 驱动、帧图、GPU-driven 提交、可见性、运行时 |
| 子系统 | `prism_hair_gpu` (28382) / `prism_volumetric_gpu` (347165) / `prism_virtual_geometry_gpu` (7453) / `prism_virtual_texture_gpu` (1329) / `prism_water_render` (1476) | 各自带几何+模拟+专用渲染 |
| 物理 | `prism_physics_core` (129271) / `prism_physics_gpu` (97521) / `prism_physics_geometry` (9148) | 引擎无关物理内核 + GPU 求解；**注意**其依赖接入现状见 §6.4 |
| 已删除 | `prism_render_slang`、`prism_render_slang_abi` | 见 §2.1 |

## 2. 决策记录

### 2.1 放弃 Slang（2026-09-29 定案）

**决策**：不引入 Slang；着色语言层只用 WESL + naga。

- `prism_render_slang`（约 885 行）与 `prism_render_slang_abi`（约 1166 行）**已删除**（commit `211f15988`），零外部依赖；`pkg/` 下 `rg -i slang` 无命中，残留只在文档里。
- 四个当初的动机全部塌缩：反射信息可由手写 `#[repr(C)]` + 测试 + schema 覆盖；变体缓存不需要第三方编译器；工具链（`slangc`）在受限网络机器上不可得。
- 唯一"Slang 独占"的价值是"着色器与 CPU golden 单一源码"。但**真实的 CPU 渲染路径并不存在**，CPU 侧只有 `#[cfg(test)]` 骨架，因此这个价值为零。
- 替代方案已落地：WESL 手写 + schema 生成 + CI 对齐测试（§3.5）+ naga + wgpu。

### 2.2 运行时不做闭包图（commit `f5172023c`）

**决策**：作者层保留闭包图，运行时**不序列化**图。

- `normalize()` 保留图（`pkg/prism_render_material/src/ir.rs:63-137`），供校验/烘焙/诊断；
- 运行头里没有 `closure_graph_offset`，`pkg/` 下 0 命中；运行时只留 `closure_mask`（`record.rs:129-132` 注释为准）；
- 理由：运行期图遍历意味着不确定的寄存器压力、无法定长的特化键、以及每个后端各自一套遍历器。

### 2.3 删除闭包 slab 与纹理槽硬上限（commit `1e9d1812b`）

**决策**：`MAX_CLOSURE_SLAB_DEPTH`、`slab_depth()`、`MaterialValidationError::ClosureSlabTooDeep`、纹理槽上限全部删除。

- 旧模型"最多 3–4 瓣"是一个**会静默改变外观**的封顶；超出即报错或截断，与 G6 冲突。
- 现在：瓣由 `LobeMask` 位表达（10 个已定义瓣，`surface.rs:411-465`），参数块按"每个在位的瓣"变长打包（§3.4），openpbr 侧由 `project_to_openpbr`（`openpbr.rs:118`）给出"投影后丢了哪些瓣"（`DroppedLobe`，`openpbr.rs:43`）而不是硬失败。
- 残留的注释级提及（`pkg/prism_render_scene/src/material/buffers.rs:33`、`pkg/prism_render_material/src/tests.rs:83,200`）属于文档债。

### 2.4 单一事实源：`schema/surface.toml` + codegen（G5）

**决策**：布局/掩码/默认值/取值范围/能量策略/可见性，只在 `pkg/prism_material_schema/schema/surface.toml` 声明一次，由 `prism-material-codegen` 生成 Rust 常量与 WESL 解包器。

- 生成物 `pkg/prism_render_material/src/surface_layout.rs` 首行即 `// DO NOT EDIT — generated by 'prism-material-codegen' …`；
- 再生成命令：`cargo run -p prism_material_schema --bin prism-material-codegen -- write`；
- 漂移门禁：`codegen_freshness` 测试 + `pkg/prism_render_material/tests/schema_parity.rs`（校验 `abi_version` 一致）；
- 存在的理由：v2 时代同一份布局被手写在**六处**（Rust `surface.rs`/`record.rs`、WESL `material_unpack`/`material_sample`、CPU golden、文档表格），任何一处漏改都表现为"渲染结果莫名不对"。

### 2.5 PCM：九类扩展点统一注册（"超集"的完整模型）

**决策**：新增一个扩展点不再各自发明机制，而是登记进 **Prism Capability Model**（`pkg/prism_material_pcm`）。

- 统一了此前九个各自封闭的机制：加瓣 / 加参数通道 / 加响应模式 / 传输重映射 / 缓存辅助 / 数据提供者 / 降级策略 / 作者节点 / 合成 pass。
- `Kind` 九值：`Lobe, Channel, ResponseMode, Transfer, CacheAux, Provider, Downgrade, AuthorNode, Composite`（`pkg/prism_material_pcm/src/namespace.rs:213-232`）；
- `Region { Specification=0, Prism=1, CrossVendor=2, Studio=3 }`，每区 16 槽（`namespace.rs:89-96,112`）；
- **性能红线**：全部治理（命名空间解析、协商、used/required 集合、位/区分配）在烘焙/载入期完成然后**蒸发**；运行时只有单态化的 WESL + 头部里定宽的 64 槽 / 4 字摘要（`pkg/prism_material_pcm/src/lib.rs:75-79`），热路径无分发、无查表。

### 2.6 键的正交性（I2）与两种键视图（I5）

**决策**：特化键只由「瓣集合 + 特性标志」决定；`bin_class` 与 `SpecializationId` 是同一个键的两种视图。

- `SpecializationId(u64)`：照明 `0..8` / 闭包 `8..40` / 渲染类 `40..48`（`axis.rs:102-116`）；
- `bin_class`：29 位投影，供 GPU-driven 路径使用（`axis.rs:90,162-166`），`from_bin_class` 保证往返（`axis.rs:183-192`）；
- 质量档、缓存命中、降级**永远不进键**（I2），否则同一材质会因距离/画质档而重编译 PSO；
- 由于两种视图同源，GPU-driven bin 路径与 CPU direct 路径必须命中同一个 PSO（I5）。
- ⚠ 已知缺口：`bin_class` 的瓣窗口是**手写的 19 位**，而瓣词汇表已到 bit 18；第 4 个扩展瓣（bit 19）会被静默丢弃。收口方案：**取消位打包身份，改用密集索引** —— 见 `docs/refactoring-dev/01-surface-abi_refactor_zh.md` §3 与 `docs/refactoring-dev/03-shader-package_refactor_zh.md` §2。

## 3. 支柱一：材质 ABI 与特化键（code-truth）

> 本节描述**运行时**契约。作者层与烘焙层的开放性见 §1.1；材质侧的完整重构方案见 `docs/prism_material_pipeline_refactor_zh.md`。

### 3.1 特化键：`SpecializationId(u64)` 三轴 + `bin_class` 投影

位布局（LSB → MSB），`pkg/prism_render_material/src/axis.rs:98-116`：

| 位段 | 字段 | 语义 |
|---|---|---|
| `0..8` | `illumination` | 照明轴：`Lit=0` / `Stylized=1` / `Unlit=2` / `Custom=3`（`axis.rs:24-37`，名称 `"lit"/"stylized"/"unlit"/"custom"` `axis.rs:55`） |
| `8..40` | `closure_mask` | **活跃瓣集合**（不是作者原始位），32 位 |
| `40..48` | `render_class` | `MaterialRenderClass`（11 值，§3.8） |

- `new(illumination, closure_mask, render_class)`（`axis.rs:111`）、`illumination()/closure_mask()/render_class_bits()`、`low()/high()`（`axis.rs:139-146`，供 64 位头拆成两个 u32）。
- **GPU-driven 投影**：`bin_class(self) -> u32`，29 位（`BIN_CLASS_BITS = 29`，`axis.rs:90`），布局 `照明 0..2 | 渲染类 2..10 | 活跃瓣集合 10..29`（`axis.rs:162-166`）；`from_bin_class() -> Option<Self>` 为精确逆映射（`axis.rs:183-192`），因此 **GPU bin 路径与 CPU direct 路径落同一个 PSO**（不变式 I5）。
- **已知缺口（必须修）**：`bin_class` 的瓣窗口是**手写 19 位**（`0..29` 减去 10 位前缀），而瓣词汇表已到 `MEASURED_BRDF = 1<<18`；再增加第 4 个扩展瓣（bit 19）会被静默截断，"丢一个瓣"不报错。且 `bin_class` 是 u32 共享预算（`2(照明) + 8(渲染类) + W`），W=22 即到 u32 极限、W=23 溢出 —— 最多再加 3 个瓣。收口方案：身份改用可枚举的密集索引 `(ShaderPackageId, PermutationIndex, PipelineStateHash)`，见 `docs/refactoring-dev/01-surface-abi_refactor_zh.md` §3 与 `docs/refactoring-dev/03-shader-package_refactor_zh.md` §2。
- **不变式 I2**：键只由「瓣集合 + 特性标志」决定。质量档、缓存命中、距离降级**不得**进入键，否则同一个材质会因画质档/距离而重新编译 PSO。

### 3.2 闭包 IR：作者层保图，运行层只留掩码

- `ClosureKind` **14 值**（`pkg/prism_render_material/src/ir.rs:15-32`）：`Diffuse, Conductor, Dielectric, ClearCoat, Sheen, Subsurface, Transmission, Emission, Hair, Volume, Npr, Custom, Mix, Layer`；掩码位 = `1 << kind as u32`（`ir.rs:93`）。
- `normalize()` 产出 `NormalizedMaterial { nodes, output, features, render_class, illumination, closure_mask, specialization_id }`（`ir.rs:63-137`）——**保留图**，供校验/烘焙/诊断。
- 运行时**不序列化**该图：`GpuMaterialHeader` 没有 `closure_graph_offset`，`pkg/` 下 0 命中（§2.2，commit `f5172023c`）。运行时只保留 `closure_mask` + 变长参数块。
- 无 `MAX_CLOSURE_SLAB_DEPTH` / `slab_depth()` / `ClosureSlabTooDeep`（§2.3，commit `1e9d1812b`）。

### 3.3 运行时头部 `GpuMaterialHeader`（ABI v6，27 × u32 = 108 B）

`pub const MATERIAL_ABI_VERSION: u32 = 6;`（`pkg/prism_render_material/src/record.rs:18`）
`#[repr(C)] #[derive(…, bytemuck::Pod, bytemuck::Zeroable)] pub struct GpuMaterialHeader`（`record.rs:133-218`），字段顺序即布局：

| 组 | 字段（顺序） | 说明 |
|---|---|---|
| 身份 | `generation`, `revision` | 代际 + 修订（内容寻址 / 增量更新用） |
| 分类 | `illumination`, `render_class`, `feature_flags`, `closure_mask`, `lobe_mask` | 与 `SpecializationId` 三轴对应；`lobe_mask` 是**表面瓣**集合（§3.4），与 `closure_mask`（闭包种类）分属两层 |
| 参数堆 | `parameter_offset`(u32 字偏移), `parameter_size`(字节), `texture_offset`, `texture_count`, `sampler_offset`, `sampler_count`, `custom_program` | 变长参数块在共享堆中的位置；解码必须配合 `lobe_mask` |
| 活跃 | `is_active` | 拼写带 `is_`：WGSL 保留字 `active`；由 `tests/material_header_abi.rs` 钉住 |
| 纪元 | `material_epoch_low`, `material_epoch_high` | 材质图/烘焙纪元，用于缓存失效判定 |
| 键 | `specialization_low`, `specialization_high` | `SpecializationId` 的 64 位拆分（§3.1） |
| PCM 摘要 | `extensions_used_low/high`, `extensions_required_low/high` | 4 字 = 64 槽定宽摘要（`HEADER_DIGEST_SLOTS = 64`、`HEADER_DIGEST_WORDS = 4`，`pkg/prism_material_pcm/src/lib.rs:75-79`）；纯 OpenPBR 材质全 0 |
| 动态通道 | `dynamic_wetness`, `dynamic_snow`, `dynamic_damage`, `dynamic_dissolve` | f32 位模式；**头部里唯一允许运行时改写**的字，`0.0` 表示精确恒等 |

- 尺寸以关系式断言而非字面量：`tests/material_header_abi.rs:107-108` 校验 `rust.len() * 4 == size_of::<GpuMaterialHeader>()`。
- `DynamicChannels`（`record.rs:240-276`）：`NONE` 常量、`to_words()/from_words()`、`sanitize()`（非有限值 → 0，其余夹到 `[0,1]`）；`NO_VIRTUAL_TEXTURE = u32::MAX`、`PORTABLE_TEXTURE_ATLAS = u32::MAX-1`、`FALLBACK_MATERIAL_HANDLE = {index:0,generation:0}`（`record.rs:23-29`）。
- **没有**结构哈希机制：版本由手写常量 + 测试 + schema 的 `abi_version` 三者共同维护（v2 文档里"结构哈希驱动版本"只是提案）。

### 3.4 表面参数块：20 字核心 + 按瓣变长

核心常量**全部由 schema 生成**（`pkg/prism_render_material/src/surface_layout.rs`，首行 `DO NOT EDIT`）：

| 常量 | 值 | 含义 |
|---|---|---|
| `SURFACE_CORE_WORDS` | **20** | OpenPBR 核心参数（元数据）字数 |
| `SURFACE_LOBE_WORDS` | **16** | 最宽瓣的尺寸；注释明确 *"never a packing stride"* |
| `SURFACE_LOBE_WORDS_BY_SLOT` | `[usize; 19]` = `[8,16,4,8,16,16,4, 0×9, 4,4,4]` | 逐槽字数；两段密集区（KHR 从 0，扩展从 `EXTENSION_SLOT_BASE`），中间留保留段 |
| `SURFACE_LOBE_COUNT` | **10** | 已定义瓣数 |
| `SURFACE_MAX_WORDS` | **104** | 解码缓冲上界 = 核心 + 全瓣；WESL 侧 `array<u32, 104>` |

瓣注册表（`surface_layout.rs:73-184`，`LobeLayout { id, uid, mask_const, registry_slot, words, namespace, stability, requires_caps, energy_policy }`），**低位优先进位**：

| 槽 | 瓣 | `LobeMask` 位 | 字数 | 命名空间 | 能量策略 |
|---|---|---|---|---|---|
| 0 | emission | `1<<0` | 8 | `org.khronos.openpbr.emission` (KHR) | independent |
| 1 | clearcoat | `1<<1` | 16 | `org.khronos.openpbr.coat` (KHR) | coupled |
| 2 | anisotropy | `1<<2` | 4 | `org.khronos.openpbr.specular_anisotropy` | — |
| 3 | sheen | `1<<3` | 8 | KHR | — |
| 4 | subsurface | `1<<4` | 16 | KHR | — |
| 5 | transmission | `1<<5` | 16 | KHR | — |
| 6 | thin_film | `1<<6` | 4 | KHR | — |
| 16 | npr_face | `1<<16` | 4 | Prism 扩展 | — |
| 17 | glint | `1<<17` | 4 | Prism 扩展 | — |
| 18 | measured_brdf | `1<<18` | 4 | Prism 扩展 | — |

- 区划分隔：`KHR_REGION`（7 个规范瓣，`pkg/prism_render_material/src/surface.rs:444`）与 `EXTENSION_REGION`（3 个 Prism 瓣，`surface.rs:460`）；辅助函数 `canonical_region()` / `extension_region()` / `is_pure_openpbr()` / `region_of(bit)`（`surface.rs:510-533`）。`LobeMask::COUNT = 10`（`surface.rs:465`）。位 7..15 留白，给规范后续增瓣。
- 打包：`SurfaceParameterBlock`（`surface.rs:577`），`pack() -> Vec<u32>`（`surface.rs:843`）、`unpack(lobe_mask, words) -> Result<Self, SurfaceUnpackError>`（`surface.rs:856`）；长度公式 `packed_len_words = SURFACE_CORE_WORDS + Σ(在位瓣的 BY_SLOT 字数)`，**变长步长**（不是 20 + 10×16）。
- 门禁测试：`size_of::<GpuSurfaceCore>() == SURFACE_CORE_WORDS * 4`（`surface.rs:1073`）、空块长度 == 核心（`:1101-1103`）、逐瓣 `words <= SURFACE_LOBE_WORDS`（`:984-1001`）、`LobeMask` 与生成表一致（`layout_matches_generated_table`）。

### 3.5 生成式布局：单一事实源 `schema/surface.toml` → Rust + WESL

`pkg/prism_material_schema/schema/surface.toml`（1255 行）声明：`schema_version = 1`、`abi_version = 6`（`surface.toml:18,23`）、`[core] words = 20`；每个字段带 `name / type / bit_region(KHR|extension) / pack_slot / consumed_by(["unpack","classify","sample","golden"]) / default / valid / energy_policy / view`。例：`base_color: vec4f` slot 0 默认 `[0.8,0.8,0.8,1.0]`；`base_metalness` slot 4；`specular_roughness` slot 5 默认 `0.3`；`specular_weight` slot 6 默认 `1.0`（取代旧 `reflectance` 代理）。

生成物两件，都带 `DO NOT EDIT`：

| 生成物 | 位置 | 校验 |
|---|---|---|
| Rust 布局常量与注册表 | `pkg/prism_render_material/src/surface_layout.rs`（445 行） | `codegen_freshness` 测试 + `surface.rs::layout_matches_generated_table` |
| WESL 解包器 | `pkg/prism_render_scene/src/shaders/material_unpack.wesl`（`prism_unpack_surface`） | `codegen_freshness`；与 Rust `unpack()` 逐字对齐 |

- 再生成命令：`cargo run -p prism_material_schema --bin prism-material-codegen -- write`
- ABI 版本一致性门禁：`pkg/prism_render_material/tests/schema_parity.rs:19`（TOML `abi_version` ⇄ `MATERIAL_ABI_VERSION`）。
- 除瓣以外，schema 还生成：`SURFACE_CHANNELS`（4，槽 19–22）、`SURFACE_RESPONSE_MODES`（3，槽 23–25）、`SURFACE_CACHE_AUX`（1，槽 26）、`SURFACE_IDENTITY_EXTENSIONS`（5）、`SURFACE_LOBE_CLOSURES`（3）。
- 阶段：S1 = 声明并与手写布局做 parity 检查；**S2（当前）** = 成为唯一生成器（Rust 常量 + WESL 解包器），手写副本消失。
- **文档债（需修）**：`surface.toml:31` 的注释仍写 `SURFACE_CORE_WORDS = 12`，而同文件 `[core] words = 20`；v2 文档的"12 字核心"很可能就是从这里抄错的。

### 3.6 PCM：九类扩展点的统一注册与载入期协商

`pkg/prism_material_pcm`（3319 行 / 10 文件）= **Prism Capability Model**。它把此前**九个各自封闭**的机制收进一份声明式能力契约（加瓣 / 加参数通道 / 加响应模式 / 传输重映射 / 缓存辅助 / 数据提供者 / 降级策略 / 作者节点 / 合成 pass）。

| 概念 | 定义 | 位置 |
|---|---|---|
| `Kind`（九类扩展点） | `Lobe, Channel, ResponseMode, Transfer, CacheAux, Provider, Downgrade, AuthorNode, Composite` | `namespace.rs:213-232` |
| `Region` + 槽 | `Specification=0, Prism=1, CrossVendor=2, Studio=3`，每区 `SLOTS_PER_REGION = 16` | `namespace.rs:89-96,112` |
| `ExtensionManifest` | 不可变身份（`uid/registry_slot/origin`）+ 可变归属（`namespace/vendor/stability`）+ 派生显示 id | `manifest.rs:238` |
| `ExtensionRegistry` | 表结构 + 命名空间分区槽分配器 + 提升状态机 + 可机器校验的字段谓词 | `registry.rs:79` |
| 协商 | 载入期握手：缺 `required` ⇒ **硬错误**；缺 `used` ⇒ **优雅降级/投影** | `capability.rs` / `negotiate.rs` |
| `ExtensionSet` | 定宽 used/required 摘要（进头部 4 字 / 64 槽） | `lib.rs:75-79` |

- **性能红线**：命名空间解析、协商、位/区分配全部在烘焙/载入期完成**然后蒸发**；运行时只有单态化 WESL + 头部里定宽摘要，热路径无分发、无查表。
- **缺口（code-truth）**：`Composite` 这一 Kind 目前把合成 pass 折进"烘焙期置换键**或** pass 图"（`hooks.rs:38-42`），但**没有运行时 pass 注册扩展点**，也没有 frame-graph 代码把 PCM 当作 pass 来源；PCM 目前只被 `prism_render_scene/src/shading/classification_gpu.rs:42` 与 `prism_render_shading/src/classification.rs` 消费。该缺口由 §9 的四层收口覆盖。

### 3.7 OpenPBR 合规投影（把"超集"变成可判定谓词）

- `project_to_openpbr()`（`pkg/prism_render_material/src/openpbr.rs:118`）把任意材质投影到纯 OpenPBR；`OpenPbrProjection`（`:65`）给出投影结果，`DroppedLobe`（`:43`）逐条列出被丢掉的瓣（而不是静默截断）。
- 判定三值：`OpenPbrConformance { Conformant, NotApplicable, Breached }`（`openpbr.rs:162`）。
- 规范区与扩展区的隔离由 §3.4 的 `KHR_REGION` / `EXTENSION_REGION` 强制（不变式 I4）：规范区内容必须符合规范，扩展位必须落在扩展区，二者不混用。
- OpenPBR 每个参数在 PBR / NPR / MaskBlend / 自定义响应里的功能，逐条见 `docs/prism_openpbr_pipeline_function_zh.md`。

### 3.8 渲染类、域与特性标志

- `MaterialDomain`：`Surface`(默认) / `Decal` / `Volume` / `PostProcess`（`record.rs:33-41`）。
- `MaterialRenderClass`：11 值（§1.2），`from_u32` 反查（`record.rs:72-88`）。
- `MaterialFeatureFlags`（`record.rs:90-110`）：

| 位 | 名称 | 语义 |
|---|---|---|
| `1<<0` | `DOUBLE_SIDED` | 双面 |
| `1<<1` | `ALPHA_MASK` | 掩码透明 |
| `1<<2` | `TRANSMISSION` | 透射 |
| `1<<3` | `NORMAL_MAP` | 法线贴图存在 |
| `1<<4` | `EMISSIVE` | 自发光存在 |
| `1<<5` | `COMPLEX_CLOSURE` | 多瓣/复杂闭包（触发更贵的路径） |
| `1<<6` | `FACE_SHADOW` | 面部 SDF 阴影（NPR） |
| `1<<7` | `REGION_ILLUMINATION` | **逐像素照明区域掩码**：可见性光栅采样后写进可见性元数据的备用字，分类按区域各自分箱（§4.3），**不改变材质 ABI** |

### 3.9 架构侧的材质执行契约（与 GPU 布局区分）

`pkg/prism_render_architecture/src/material/`（3 文件 / 551 行）持有一份 **CPU 侧**材质执行/分箱契约：`MaterialDomain`（`mod.rs:11`）、`MaterialExecutionPath`（`mod.rs:20`）、`MaterialRecord`（`mod.rs:28`）、`MaterialRegistry`（`registry.rs:32`）、`MaterialResolveBins` 与 `RESOLVE_PASS_ORDER: [MaterialExecutionPath; 4]`（`resolve.rs:42,103`）。

它**不定义 GPU 布局**——布局的唯一事实源是 §3.4/§3.5。二者关系：架构侧决定"材质进哪个解析批次/哪条执行路径"，材质 crate 决定"这批数据在 GPU 内存里长什么样"。

## 4. 支柱二：光照 / 阴影数据服务与响应计划

### 4.1 共享的是数据与加速结构，不是响应

GPU-driven 基底只计算一次：光源集合与图元、阴影图/虚拟阴影页、探针与环境、速度场、可见性缓冲、簇划分。**响应函数**（怎么把这份数据变成像素颜色）由 `ResponseProgram` 分叉；**组合**由 `WaterShadingPlan::MaskBlend` 表达，组合不产生新的响应程序。

水体共享服务的 9 个语义身份，以及体积共享服务的 7 个语义身份，统一由 `render_fabric::builtin_catalog` 声明并由 provider 闭包校验；不再存在按子系统复制的位掩码事实源。

服务身份使用显式 `ServiceId` 与集合类型 `ServiceSet`，不是旧式位掩码：水体使用 `ServiceId(1..=9)`，体积使用 `ServiceId(101..=107)`；`FeatureDefinition::requires_services` 表示特性级硬需求，`FeatureVariant::requires_services_extra` 只参与候选准入，不扩张闭包。provider 由 `FeatureKind::Service { provides }` 唯一声明，所有 variant 必须发布同一组 `ServiceOutput.resources`，并由静态门禁校验每个资源确实被每个 provider variant 写入。

### 4.2 三档响应分类（保留 v2 的分类法，补上代码落点）

| 档 | 含义 | 实例 | 代码落点 |
|---|---|---|---|
| **A** | 同能力，实现不同 | 直射光积分、阴影采样、GI 收集、时序累积 | PBR：`prism_render_shading/src/lighting.rs:369` `evaluate_principled_direct`；NPR：`stylized.rs:185` `evaluate_stylized_direct`（同一 `DirectLightSample` 输入） |
| **B** | 能用，但本质退化 | NPR 在 RT/GI 里只取反照率；体积内的 NPR 取平均色 | NPR 收共享探针，但不向 GI 发新能量（见 §4.5 缺口） |
| **C** | 每条线自己的 | 光照通道/图层（NPR）、面部 SDF 阴影、描边 | `light_routing.rs`（`MAX_LIGHTING_CHANNELS=8` `:29`、`MAX_LIGHT_LAYERS=4` `:34`）、`face_shadow.rs:144`、`outline.rs:218` |

**把 v2 的一句话钉死到实现上**："共享光照/阴影**数据**，不共享**响应**"——`SurfaceSample`（40+ 字段，`lighting.rs:24-111`）是共享的输入契约，响应函数是分叉点。

### 4.3 逐像素照明区域（`REGION_ILLUMINATION`）

同一材质可以在同一帧里按区域走不同照明轴：可见性光栅采样一个逐像素的 Lit/Stylized 掩码，写进可见性元数据的备用字，分类器按区域各自分箱（`MaterialFeatureFlags::REGION_ILLUMINATION = 1<<7`，`record.rs`）——**不改变材质 ABI**，也不新增渲染类。

### 4.4 光照 / 阴影数据服务的当前版图（code-truth）

| 服务 | 位置 | 关键入口 |
|---|---|---|
| 光簇 | `prism_render_shading/src/cluster/` | `assign_lights_to_clusters`（`assign.rs`）、`ClusterGrid`、`ClusterAabb` |
| 直射光源类型 | `resolve.rs` / `punctual.rs:28` | `DirectionalLight :99`、`LightingEnvironment :120`、`resolve_pixel :370` |
| 面积光 | `gi/area_light/` | `ltc_lut`、`polygon`、`representative`、`resolve` |
| 多重重要性 / RIS | `gi/nee/`、`gi/spec_gi/` | `light_sampling.rs:57`（`LightSample`）、`mis`、`ris`、`spec_gi/reservoir` |
| IBL / 环境 | `environment/`（顶层） | `brdf_lut`、`cubemap`、`prefilter.rs`、`evaluate_image_based_light :198`、`SphericalHarmonicsL2 :67` |
| 阴影 | `shadow/` | `csm.rs`（`compute_cascade_matrices`）、`atlas`、`bias`、`filter`、`point/spot/directional` |
| 虚拟阴影 | `shadow/virtual_sm/` | `drive_frame :203`、`drive_frame_with_requests :230`、`page_table.rs:118`、`allocator.rs:76` |
| AO | `ao/`（顶层） | `gtao_pixel :115`、`compute_gtao :240`、`denoise_gtao` |
| 屏幕空间 GI/SSR | `screen_space/`（顶层） | `march_hierarchical`、`trace_screen_space_reflection :88`、`screen_space/gi.rs:231` |
| 世界空间 GI / ReSTIR | `gi/world_space/`、`world_restir*.wesl` | `probe_placement`、`radiance_cache`、`spherical_gaussian`、`visibility` |
| 降噪 | `gi/denoise/` | `reblur`、`variance`、`firefly` |
| 大气 / 天空 | `gi/atmosphere`、`sky_*_lut.wesl` | `sky_transmittance_lut` / `sky_multiscatter_lut` / `sky_view_lut` |

### 4.5 已知缺口（诚实清单）

1. **NPR 只消费 3 个 OpenPBR 参数**（`base_color` `stylized.rs:206`、`perceptual_roughness` `:213`、`emissive` `:242`）——40+ 个其余参数对 NPR 不可见。收口方案：NPR 作为**完整 OpenPBR Surface 上的一条响应**直接读取瓣与字段（不新增 `NprParams`、不重表为 closure），覆盖率由**三值断言**守住（`mapped` / `intentionally_unmapped(reason)` / 未声明 = 构建期报错），并先产出「字段 × 响应程序」覆盖矩阵。见 `docs/refactoring-dev/02-response-composition_refactor_zh.md` §2。注意：早期的两个方案 ——「逐参数映射表」与 `npr = stylize(pbr_lite)` 通道变换 —— **均已排除**，理由见 `docs/refactoring-dev/00-overview_refactor_zh.md` §8。
2. **class 表与 `shading_resolve.wesl` 的 switch 之间没有机器门禁**：唯一的 Rust↔WESL 同步测试只读 `material_classification.wesl` 的 `CLASS_*`（`classification.rs:995-1046`）；`shading_resolve.wesl:131-139` 的 `SHADING_CLASS_*` 与 9 个 `case` 标签由人工保持。
3. **WESL 侧多一个 `default:` 臂**（`shading_resolve.wesl:2588` → `shade_principled`），而 Rust 侧刻意保持 9 臂无通配（`resolve.rs:750-759`）——违反不变式 I6 的精神，目前靠分类阶段的提前丢弃掩盖（`material_classification.wesl:153`）。
4. **reactive mask 的 GPU 侧未接**：CPU 三个入口已闭合（§5.3），但 `taa_resolve.wesl`(172 行) 与 `upscale_reconstruct.wesl`(388 行) 中 `reactive` 出现 **0** 次。

## 5. 支柱三：跨切面基底

### 5.1 运动 / 速度编码（消费方契约）

`pkg/prism_render_scene/src/motion/encode.rs`：`VelocityEncoding`（`:26`）、`PackedMasks(u32)`（`:123`，位 `DISOCCLUDED` / `TRANSPARENT` / `STREAMING_REVEAL`，`:182-188`）、`EncodedMotion`（`:174`）、`encode_sample`（`:193`）、去遮挡分类（`:215`）；膨胀在 `dilation.rs:75,296`（`VelocityField` / `TileVelocityField`）。

### 5.2 时序超分（arch 与 shading 双驻）

- 架构侧：`prism_render_architecture/src/temporal_upscale/`（9 文件 / 2588 行）`JitterSequence`、`LockState`（`lock.rs:332`）、`NeighborhoodStats`（`neighborhood.rs:304`）、`ResolveOutput`。
- 着色侧：`prism_render_shading/src/{taa,upscale}/`（1927 行）`resolve_taa`（`taa/resolve.rs:103`）、`upscale_pixel`（`upscale/mod.rs:161`）、`apply_rcas`（`upscale/sharpen.rs`）。
- 双驻是**尚未收口**的事实，不是设计意图（§9 第三层）。

### 5.3 反应掩码（reactive mask）：CPU 已闭合，GPU 待接

| 入口 | 位置 | 公式 |
|---|---|---|
| TAA 解析 | `taa/resolve.rs:137` | `effective_blend = history_blend · (1 − reactive)` |
| 稳健超分 | `upscale/robust.rs:119` | 文档式：`base · (1 − d) · (1 − reactive)` |
| 超分主路径 | `upscale/mod.rs:206-207` | `history_weight = (1 − current_weight) · (1 − reactive)` |

三个 commit（`e4790e0c6` / `1b1c61080` / `d1c9827c4`）各只改一个文件；测试 `reactive_zero_is_bit_for_bit_the_shipping_resolve`（`upscale/mod.rs:543`）等保证 reactive=0 时逐位不变。

### 5.4 OIT / 透明

三处并存（§9 待收口）：`prism_render_shading/src/oit.rs`（`OitFragment :19`、`oit_weight :52`、`OitAccumulation :68`、`composite_transparency :155`）、架构侧 `transparency/`（2538 行）、粒子 `particle/oit.rs`；另有第二族矩 OIT（`gi/oit/mod.rs:155` `MomentOit`）。

### 5.5 降噪与能力分层

- 降噪：`gi/denoise/{reblur,variance,firefly}`（时域重投影 + 方差引导空间滤波）。
- 能力分层：`Capability` 六位（§8.1），`CapabilitySet::WEBGPU_BASELINE` 是**必须**满足的下限。

## 6. 支柱四：可变形几何 / 模拟与子系统

### 6.1 关于"子系统注册表"的更正

v2 文档 §6.2 描述的"四档子系统注册表"是**文档产物，不是代码**：全仓库 0 命中 `SubsystemTier` / `SubsystemRegistry` / `SubsystemKind`。实际存在的是**按目录划分的子系统 crate/模块 + 一份共享基底位集（§4.1）**。子系统之所以是子系统，判据仍成立：**自有几何 + 自有模拟 + 自有专用渲染阶段**。

### 6.2 子系统现状（LOC = `pkg/**/src/*.rs`）

| 子系统 | 位置（`prism_render_architecture/src/` 或独立 crate） | 规模 | GPU 孪生 |
|---|---|---|---|
| 粒子 | `particle/`（263 个 `mod`） | 265 文件 / 165 790 行 | — |
| 毛发 | `hair/` | 77 / 35 913 | `prism_hair_gpu` 89 / 28 382（87 `.wesl`） |
| 布料 | `cloth/` | 33 / 18 038 | —（`cloth_*.wesl` 在 `prism_render_scene`） |
| 水面 | `water/` | 78 / 31 515 | `prism_water_render` 11 / 1710（真机 wgpu 光栅 + 回读） |
| 体积 | `volumetric/`（自持模拟，不依赖 physics_core） | 25 / 12 862 | `prism_volumetric_gpu` 639 / 347 165（80 `.wesl`，规模异常，含 600+ `Gpu*` 类型） |
| 虚拟几何 | `virtual_geometry/` | 16 / 4924 | `prism_virtual_geometry_gpu` 25 / 7453（24 `.wesl`） |
| 虚拟纹理 | — | — | `prism_virtual_texture_gpu` 5 / 1329（3 `.wgsl`） |
| 光线场景 / RT | `ray_scene/` | 137 / 72 339 | — |
| 路径追踪参考 | `reference_pt/` | 42 / 15 657 | — |
| 形变 / 几何 | `deformation/`、`geometry/` | 2 / 321、3 / 674 | — |

**共享模拟核心**：`pkg/prism_physics_core`（411 文件 / 129 271 行）是 `prism_render_architecture` 的**硬依赖、非可选**（`Cargo.toml:13`）；26 个文件导入它（布料 17、毛发 6、水面 2、粒子 1；体积 0）。**未接线**：`pkg/prism_physics_gpu`（368 文件 / 97 521 行 / 110 `.wgsl`）目前**没有任何消费方 crate**，未接入渲染图——这是独立的待接线项，不是子系统设计问题。

### 6.3 毛发：同构 mini 支柱（保留 v2 的示范意义）

毛发拥有自己的求解器、LOD、光栅路径与 RT 代理：`HairSolverKind`（`hair/solver.rs:40`）、`HairLodTier`（`hair/mod.rs:143`）、`HairLodTransition`（`transition.rs:33`）、`HairRasterPath`（`raster.rs:63`）、`build_deep_opacity` / `sample_transmittance`（`deep_transmittance.rs:95,161`）、`RtReflectionRole` / `RtProxyPolicy`（`rt_proxy.rs:28,69`）、`step_groom`（`groom.rs:71`）。它照样消费共享基底（GI/阴影/时序），因此是"子系统 ≠ 另一条管线"的样板。

### 6.4 着色语言与验证现实

`pkg/` 下 WESL 共 **400** 个文件：`prism_render_scene` 175、`prism_hair_gpu` 87、`prism_volumetric_gpu` 80、`prism_render_architecture` 29、`prism_virtual_geometry_gpu` 24、`prism_render_shading` 5。另有 `.wgsl` 约 130 个（`prism_physics_gpu` 110、`motion_gpu` 5、`prism_render_gpudriven` 4 等）——**v2 头部"68 .wesl / 0 .wgsl"已不成立**。

验证分层是本项目最重要的工程现实：GPU 孪生 crate 依赖 `prism_render_architecture`（CPU golden 在后者），因此**同一个算法可以 CPU 逐位对照 GPU**。虚拟几何的 CPU golden 在 `virtual_geometry/software_raster.rs`（64 位深度/负载、reversed-Z、top-left 规则），GPU 孪生把同一份 WESL 编进 `prism_virtual_geometry_gpu/src/raster.rs:3,30`，并以 `gpu_parity.rs:84` 做对拍；无适配器时 `GpuContext::try_headless` 静默跳过。这是 §8.2 三桶模型的实践基础。

## 7. 响应计划：多种响应如何在一次绘制里共存

### 7.1 现行机制（code-truth）

1. **分箱**：`ShadingWorkPlan { offsets: [u32; 9], counts: [u32; 9], work: Vec<ShadingWorkItem> }`（`classification.rs:45-60`）把像素按 9 个渲染类分箱；GPU 侧同构：`material_classification.wesl` 的 `material_class`(:87) → `prefix_classes`(:163) → `scatter_work`(:178)。
2. **派发**：`resolve.rs:760-771` 九臂显式 `match`（`Unlit / Npr / Principled / Subsurface / ClearCoat / Cloth / Hair / Water / Custom`），无通配；WESL 孪生 `shading_resolve.wesl:2524`。
3. **渲染阶段**：`prism_render_app/src/sets.rs:126` 的 `ORDER = [Opaque, Transparent, Stylized, Custom]`（上限 4）——**这就是 v2 所称"逐像素 material id + tile 路由"的落地形态**：不是新管线，是同一批 pass 上的分箱 + 四阶段排序。
4. **执行路径契约**：架构侧 `MaterialExecutionPath` + `RESOLVE_PASS_ORDER: [MaterialExecutionPath; 4]`（`prism_render_architecture/src/material/resolve.rs:42,103`）把"材质进哪个解析批次"固定下来。
5. **子系统级声明**：`ResponseProgram` 选择响应，`WaterShadingPlan` 选择 `Single` 或 `MaskBlend`；二者与共享服务闭包解耦。

### 7.2 混合计划（MaskBlend）不是第四种响应程序

`MaskBlend` = **同一像素上多响应共存**：它有独立的组合契约和常驻组合 pass，但不新增响应程序、材质 ABI 或几何路径。连续遮罩与二值区域选择是两种不同语义，不能互相替代。

## 8. 后端抽象与可测性分桶

### 8.1 能力模型（唯一的事实源）

`pkg/prism_render_architecture/src/backend/capability.rs`（298 行）：

| 项 | 定义 | 位置 |
|---|---|---|
| `Capability`（6 值） | `Compute=0`（wgpu 基线，必定可用）、`BindlessDescriptors=1`、`IndirectDrawCount=2`、`MeshShading=3`、`RayQuery=4`、`RayTracingPipeline=5` | `:31-49`，`ALL` `:52` |
| `CapabilitySet(u32)` | `EMPTY` `:81`、`WEBGPU_BASELINE = Compute` `:87`、`from_slice` `:91`、`contains_all` / `union` / `intersection` / `difference` / `missing_from` `:103-160` | `:77` |
| `FeatureRequirement` | `validate() -> Result<(), MissingCapabilities>` | `:167,188,201` |
| 协商 | `negotiate_features`：`enabled = WEBGPU_BASELINE ∪ (adapter ∩ requested)`；`unavailable = requested − adapter`；`EnabledFeatures { enabled, unavailable }` | `negotiation.rs:26,63` |
| 后端接口 | `trait RenderBackend { capabilities, wait_idle_for_shutdown, supports }` | `backend/mod.rs:46-56` |
| 测试替身 | `FakeBackend::baseline()` / `with_all_extensions()` / `is_shut_down()` | `fake.rs:20-137` |

**确定性与降级**：扩展能力缺失时**丢位**（`unavailable`），换更便宜的路径，**绝不切换后端**（`negotiation.rs`）。全仓库 0 命中 `BackendId` / `BackendTier` / `VulkanTier`——没有"多后端分层"这回事。

### 8.2 可测性三桶

| 桶 | 内容 | 验证手段 | 代码现实 |
|---|---|---|---|
| **1. compute 可移植** | AO、GI 探针、降噪、时序、体积、粒子、水面、虚拟几何、虚拟阴影 | CPU golden ↔ GPU 对拍（同 crate 依赖图：孪生依赖 `prism_render_architecture`） | 绝大多数；`prism_virtual_geometry_gpu/src/gpu_parity.rs:84` 是范式 |
| **2. RT** | 反射/阴影/GI 的 RT 路径、TLAS 遍历 | 能力位门控 + 软件参考（`reference_pt/`、`ray_scene/`） | `Capability::RayQuery` / `RayTracingPipeline` |
| **3. raw-VK-only** | 只有裸 Vulkan 才有的特性（如部分 DLSS/DXR 交互） | **无 CI**；用即放弃跨平台与 CPU golden | `pkg/` 内 **0** 命中；唯一入口在 Bevy 侧：`raw_vulkan_init`（`crates/bevy_render`，`create_device_from_hal::<Vulkan>`、`AdditionalVulkanFeatures`） |

**请求侧的能力声明**：`prism_render_scene/src/primitives/support.rs`（206 行）`GpuSceneSupport::try_new`（`:33`）——核心位 0..9（`:76` `CORE_BITS = 10`）、16 项扩展位（`:106-131`）、`REQUESTABLE.extend`（`:139`）。这是"缺能力 ⇒ 确定性降级"在场景侧的入口。

## 9. 破坏性变更台账与四层收口

### 9.1 已完成（v2 描述已作废，此处只列结果）

| 变更 | 证据 |
|---|---|
| 放弃 Slang，删除 `prism_render_slang` / `prism_render_slang_abi` | 全仓库 `slang` 0 命中（仅历史决策记录） |
| 删除运行时闭包图（`closure_graph_offset` / WESL 图遍历） | commit `f5172023c`；头部字段与 §2.2 现状一致 |
| 删除闭包 slab 深度上限与纹理槽硬顶 | commit `1e9d1812b`（P1-c） |
| 删除互斥枚举 `MaterialShadingModel`，改为轴组合 | `axis.rs`；`MaterialRenderClass` 不再含 `Npr*` / `Custom*`（`record.rs:49-64`） |
| 材质布局改为生成式（S2：Rust + WESL 双生成 + freshness 门禁） | §3.5 |
| 材质扩展点九类统一进 PCM 注册表 | commit `3090956db`（P2-j）；§3.6 |
| reactive mask CPU 三入口闭合 | §5.3 |

### 9.2 待收口（四层，方案见 `docs/refactoring-dev/00`–`04`）

| 层 | 病根（同一模式：正确抽象已存在且被测试保护，却不是唯一入口） | 现状证据 | 收口判据 |
|---|---|---|---|
| ① Surface ABI / 数据布局 | `PermutationSpace`（可枚举混合基数空间，`permutation.rs:1-75`）**已存在**；但材质身份仍是手写位打包的 `bin_class` —— 19 位窗口使第 4 个扩展瓣（bit 19）被静默丢弃，且 u32 是共享预算（`2 + 8 + W`，W=22 即极限）**最多再加 3 个瓣** | `axis.rs:162-192` | 身份改用 `(ShaderPackageId, PermutationIndex, PipelineStateHash)` 密集索引；瓣数与响应程序数**不设位宽上限**（`docs/refactoring-dev/01-surface-abi_refactor_zh.md` §3、`03-shader-package_refactor_zh.md` §2） |
| ② 着色响应 | 9 类表已是注册驱动（`ShadingClassTable::from_registry` `classification.rs:453`、loud error `:170-217`）；但 9 臂 `match` 与 WESL switch 仍手工同步，NPR 只吃 3 个参数 | §4.5 缺口 1–3 | 加一个响应模式 = 1 行 PCM 注册 + 1 次 codegen，Rust/WESL 自动同步；缺实现则构建失败 |
| ③ 管线缓存 | `PsoCacheKey { package, permutation_index, state: PipelineStateHash }`（`shader_package/pso_cache/mod.rs:61`）、`WarmRuntime`（`warm_runtime.rs:120`）、`DeviceFingerprint`（`fingerprint.rs:41`）齐备；`prism_render_app::PipelineKey`（`pipeline_cache.rs:47`）**没有 state 维度**，blend/depth/cull 折进 `class` 一个 u32，启动时 0 预热 | `pipeline_cache.rs:43-45,79,509` | 预热后冷编译 = 0（`warm_runtime.rs:35-37`）；每个影响 PSO 身份的状态都是具名键字段 |
| ④ 调度 | 声明式 `GpuFrameGraphBuilder`（`frame_graph/builder.rs:6-29`）只覆盖 **9 个** pass（`shading/graph.rs:76,91,104,115,129,139`、`visibility/graph.rs:98,145,160`）；其余约 69 个 `*_pass` 系统手工排程（`prism_render_scene/src/shading/plugin.rs`，1655 行，41 个 `.add_systems`） | §9.2 行的证据 | 加一个 pass = 1 个 `PassDescriptor` + 1 个 WESL + 1 个 settings 结构，不在 `plugin.rs` 手写顺序 |

**同一场景里并存三套"图"概念**（v2 未澄清）：`prism_render_graph`（**被执行**：`RenderGraph` / `PassNode` / `PassFlags` / `ExecutionPlan`，经 `add_raster_pass` 等组装，作者是 `prism_render_app/src/frame.rs:121,155` 与 `prism_render_gpudriven/src/scene_draw.rs:407,421,456`，合计 ≤8 个 pass）、架构侧 `frame_graph`（**只用于校验/诊断**：编译结果仅被 `debug_assert` 与诊断读取）、以及 Bevy 的 `RenderGraph` 调度标签。收口时需要一张能力并集表，而不是再造第四套。

**已知缺陷（已确认）**：`prism_render_scene/src/shading/runtime.rs:386` 断言编译序长度 == 5，而 `shading/graph.rs` 注册了 6 个 pass（其自带测试断言 6，`:188`）；编译器不做裁剪（`frame_graph/compiler.rs:42-62`，实测 `validate → dependencies → topological_order`，无裁剪步骤），`prepare_shading_work`（`plugin.rs:626`）在 debug 构建下**必然 panic**。收口方向见 `docs/refactoring-dev/04-frame-graph_refactor_zh.md` §2-P5（pass 计数必须有单一事实源）。

## 10. 诚实的固有摩擦与风险

1. **B 档降级不可移除**。NPR 在 RT/GI 里只取反照率是**设计选择**，不是缺陷或欠账；写清它，才能避免下游误以为"再努力一点就能物理正确"。
2. **没有成熟的"一等 NPR"范式**。产业界多数 NPR 是后处理或独立管线；本项目的赌注是"同一 über + 不同响应函数"。第一条线（§4.5）就是这个赌注尚未兑现的证据：NPR 目前只吃 3 个 OpenPBR 参数。
3. **ALU 上涨是可预期的代价**。要 NPR 消费全部通道，瓣必须声明 `requires_caps`，采样预算按画质档缩放，且**画质档不进置换键**（I2）。
4. **置换爆炸的治理是工程问题，不是架构问题**：内容哈希去重 + archetype 聚类 + 按需异步编译 + 全瓣 über 回退，**不设全局硬顶**（`refactor_zh` §8.4）。
5. **同一能力在多处并存**是当前最大的结构性风险：两套 PSO 缓存（§9.2 ③）、两套帧图（§9.2 ④）、TAA/upscale 双驻（§5.2）、OIT 三驻（§5.4）。收口必须按"一个模型 + 一个键 + 两个层级"推进，禁止再造第三套。
6. **Bevy 耦合边界**：迁移后每个 pass 仍是 ECS 系统，只把**顺序**改成声明式；`plugin.rs` 的 20 元组上限是当前被迫的 workaround 来源。
7. **验证边界**：(a) 真机 GPU 对拍需要 `PRISM_REQUIRE_GPU=1` 且主机提供真实 adapter；2026-10-15 Apple M2 Metal 已通过 response composition 严格 readback 与水面 raster 严格 shader/device gate，水面 raster 采用 7 个 bind groups，满足设备 `max_bind_groups=8`；其余未逐项运行的 GPU target 仍不能写成通过；(b) `shading_resolve.wesl` 的有限 shading-class switch 已由 `shading_resolve_switch_covers_every_registered_shading_class` 机械门禁覆盖；(c) `visibility_raster.wesl` 已纳入 `tests/material_header_abi.rs` 的六份 `MaterialHeader` 镜像清单，字段顺序与 Rust ABI 逐字校验。
8. **生成物门禁已恢复为绿色**：2026-10-15 执行 `cargo run -p prism_material_schema --bin prism-material-codegen --offline -- write` 重新生成 `surface_layout.rs` 与 `material_unpack.wesl`；`codegen_freshness` 全部通过，生成物不再依赖手工格式修补（`prism_material_pipeline_refactor_verification_zh.md:38,50-67`）。

## 11. 落地路线图（状态以 2026-10-15 工作树为准）

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0-a | 多散射 GGX 能量补偿 + furnace 一致 | ✅ CPU 线；白炉配置仍有 1 处缺 `metal(0.1)` |
| P0-b | 高光抗锯齿（法线方差 Toksvig/LEAN） | ◐ 消费端已落，生产端 `normal_variance` 仍恒 0（SSR 除外） |
| P1-a | 逐像素纹理 LOD / VT 采样路径 | ◐ 采样路径已接线；缺"喂真实 VT 行 + 头部 `vt_page_offset/count`" |
| P1-b | 头部/分类 WESL 镜像逐字段同步 | ✅（`tests/material_header_abi.rs`） |
| P1-c | 删闭包 slab 与纹理槽硬顶 | ✅（`1e9d1812b`） |
| P1-d | 置换治理（去重/按需编译/über 回退） | ✅ 引擎侧 |
| P1-e | 贵瓣采样数按画质档缩放 | ◐ CPU 全链落地；GPU `lobe_budget.wesl` 仅测试引用 |
| P1-f | über 参数块规范化到 OpenPBR 瓣序 | ✅（含默认值 8 项对齐） |
| P1-g | 自描述反射瓣 schema | ◐ S1+S2 完成；S3/S4（`material_classification.wesl` 与 `SpecializationId` 打包仍手写）未做 |
| P1-h | PCM 地基（九类 kind + 64 槽摘要） | ✅ |
| P2-c/d/g/j | 开放瓣词汇（glint / measured BRDF / thin-film）、超集合规门禁 + 投影导出器、动态参数通道、响应模式注册化 | ✅ |
| P2-a/b/e/h/i | 作者前端与 archetype、节点图烘焙到 VT、`respond_custom`、原理化分层、全 kind manifest + MaterialX nodedef 往返 | ◐ 部分（`respond_custom` 缺失；`coat_darkening` 在 über 路径未消费；`author_node` 零实例） |
| P3-a..f | 解耦着色（texel-space 缓存）、视角无关项解耦、失效摊销、光谱提升、随机单瓣采样、内存压缩 | ◐ 地基与独立 slice 已落，生产闭环未接 |
| **四层收口** | ① Surface ABI / 数据布局 → ② Response 与 Composition → ③ Shader Package 与 Pipeline → ④ 帧图与 pass 调度（总览 00） | 方案已定（`docs/refactoring-dev/00`–`04`），**未实施**；建议第一批：① 的布局常量改 codegen 派生（门禁 1）+ ④ 的 S5（挑 `posterize` 做声明化可行性验证）——两者互不依赖 |

## 12. 顶级产品对标矩阵（只取算法与形态，不取源码）

| 对标 | 借鉴什么 | 本项目落点 | 状态 |
|---|---|---|---|
| UE Nanite | 虚拟几何 + 簇剔除 + 软件光栅 | `virtual_geometry/` + `prism_virtual_geometry_gpu`（CPU golden ↔ GPU 对拍） | ◐ |
| UE Lumen / DDGI | 探针辐照度 + 屏幕空间回退 | `gi/world_space/` + `ddgi_*.wesl` + `ssgi.wesl` | ◐ |
| ReSTIR | 蓄水池时空复用 | `world_restir_*.wesl`（inject/fill/resolve/reuse/spatial） | ◐ |
| 虚拟阴影图（VSM） | 页表 + 按需驻留 | `shadow/virtual_sm/`（`drive_frame` / `page_table` / `allocator`） | ✅ CPU 线 |
| UE Substrate | 分层 BSDF 的表达力 | PCM 瓣 + 开放 über（但**不做**运行期图遍历，§2.2） | ✅ 表达 / ✅ 运行时受限 |
| GGF Xrd / miHoYo | 多通道 NPR + 面部 SDF | `light_routing.rs` + `face_shadow.rs` | ✅ |
| Spider-Verse / Okami | 描边与风格化后处理 | `outline.rs` + 17 个风格化后处理模块（§14.3） | ✅ |
| Kuwahara / 有序抖动 | 绘画感 | `kuwahara.rs` / `ordered_dither.rs` | ✅ |
| TSR / FSR2 | 时序超分 + 历史钳制 | `taa/` + `upscale/`（含 RCAS） | ✅ CPU 线 |
| Cyberpunk RT Overdrive | 全 RT 路径 | `ray_scene/` + `reference_pt/`（参考实现） | ◐ |

## 13. PBR 前端：次世代高级特性清单

13.1 **表面**：OpenPBR 全参数（`SurfaceSample` 40+ 字段）+ 10 瓣（emission / clearcoat / anisotropy / sheen / subsurface / transmission / thin_film / npr_face / glint / measured_brdf）+ 色散 + 薄膜虹彩 + 涂层级联 + glint + 实测 BRDF。
13.2 **光照/阴影**：光簇、LTC 面积光、CSM + 图集、点/聚光阴影、虚拟阴影页、接触阴影、DDGI 探针、ReSTIR、SSGI/SSR、spec GI、环境 BRDF LUT 与预滤波、天空 LUT 三重（transmittance / multiscatter / view）、体积云与体积雾、光轴。
13.3 **几何/材质**：虚拟几何（软件光栅 golden）+ 虚拟纹理（逐像素 LOD 解析求导）+ 形变/蒙皮 + 毛发/布料/水面/粒子子系统。
13.4 **时序/后处理**：TAA + 历史钳制、稳健超分 + RCAS、reactive mask（CPU 闭合）、运动模糊、景深、泛光、镜头光晕、色度、色调映射（Reinhard / ACES）、色彩分级、色域映射、曝光、暗角、CAS、耀斑抑制。
13.5 **确定性**：布局逐字节相等、算术 ≤1 ULP、白炉能量守恒（1.00±2%）。

## 14. NPR 前端：清单与现状

14.1 **着色**：ramp（`stylized_ramp`）/ 阴影（`stylized_shadow`）/ 高光斑 / 视角 Fresnel 边缘（`evaluate_stylized_direct`，`stylized.rs:185`），WESL 孪生在 `brdf.wesl:1095-1167` 与 `shading_resolve.wesl:1552`（`shade_toon`）。**当前只读 `base_color` / `perceptual_roughness` / `emissive` 三个参数**。
14.2 **描边**：三条边（id / depth / normal，`outline.rs:114,139,176`）+ `evaluate_outline:218`，WESL `outline.wesl`。
14.3 **风格化后处理**（全部为真实模块，不是规划）：`posterize.rs`、`halftone.rs`、`hatching.rs`、`kuwahara.rs`、`ordered_dither.rs`、`chromatic_aberration.rs`、`film_grain.rs`、`color_grade.rs`（+ PBR 共享的 `bloom/dof/lens_flare/motion_blur/vignette/cas/exposure/tonemap/gamut_map`）。
14.4 **NPR 专属数据**：光照通道 8 / 光照图层 4（`light_routing.rs`）、面部 SDF（`face_shadow.rs`）、风格化毛发（`stylized_hair.rs` 的 band 与天使环）、matcap（`hair_matcap.rs`）。
14.5 **缺口**：无逐材质风格化（风格量住在全局 `StylizedParams`，`stylized.rs:71-100`）；无 `stylize` 规则类型；40+ 参数未消费；`SHADING_CLASS_*` 与 9 臂无 codegen 门禁。

## 15. MaskBlend 组合计划

混合**不是**第四套着色器：它是 (§4.3) 的逐像素照明区域掩码 + 9 类分箱的联合表达，已落地（`material_classification.wesl:87,104-106,152`；`classification.rs:686,716-722`；`visibility_raster.wesl:47,127`）。同一帧内 PBR 区域与 NPR 区域共用几何、阴影、GI、时序基底，只切换响应函数。

## 16. 性能 / 效果 / 易用性权衡总表

| 决策 | 性能 | 效果 | 易用性 |
|---|---|---|---|
| 单一 über + 9 类分箱 + 常驻组合 pass | 一次几何/阴影/时序；分类是 O(n) 前缀和 | 多响应可同屏 | 作者只面对一份材质 |
| 无全局硬顶 + 置换治理 | 冷编译成本可控（去重 + 按需 + 回退） | 表达力不塌缩 | 需要 archetype 引导 |
| 生成式布局（schema → Rust + WESL） | 零运行时开销 | 手写漂移被门禁挡住 | 加字段 = 改 TOML + codegen |
| PCM 载入期协商 | 热路径零分发 | 缺失能力可预测降级 | 扩展点有统一登记 |
| 声明式帧图（当前 9/69） | 屏障/别名/队列可自动推导 | 顺序成为契约 | pass 骨架约 680 行 → 目标 1 描述符 |
| 解耦视角无关着色（P3-a） | 摊销重光照 | 画质取决于缓存分辨率 | 需要失效策略 |

## 附录 A. 术语与轴速查

| 术语 | 含义 | 权威位置 |
|---|---|---|
| 响应程序 / 组合计划 | 响应函数：PBR / NPR / Custom；组合：Single / MaskBlend | `ResponseProgram` / `WaterShadingPlan` |
| 照明轴 `Illumination` | 正交轴：`Lit=0 / Stylized=1 / Unlit=2 / Custom=3` | `axis.rs:24-37` |
| 瓣（lobe） | 一个注册在案的表面闭包/BSDF 片段 | `SURFACE_LOBES`（`surface_layout.rs:73`） |
| 通道（channel） | **两种含义，勿混**：① PCM `Kind::Channel` = 可逐帧调制的材质参数通道（动态四轴 wetness/snow/damage/dissolve）；② 着色缓存通道 `ShadeChannel{LitDiffuse, StylizedDiffuse, Subsurface, …}`（`shading/texel_cache/key.rs:5-24`），由字段 `view=invariant|dependent` 元数据选择 | `record.rs:240-276`；`key.rs:5-24` |
| PCM | **Prism Capability Model**（能力基座）：全管线九类扩展点的统一注册与载入期协商 | §3.6；`pkg/prism_material_pcm` |
| 九类扩展点 | lobe / channel / response_mode / transfer / cache_aux / provider / downgrade / author_node / composite | `namespace.rs:213-232` |
| `SpecializationId` | 64 位特化身份：illumination 0..8、closure_mask 8..40、render_class 40..48、48..64 保留 | `axis.rs:98-116` |
| `bin_class` | 29 位紧凑投影：illumination 2b、render_class 8b、活跃瓣集 17b(10..29) | `axis.rs:90,162-192` |
| `PermutationSpace` | 混合基数可枚举置换空间（目标态：以密集索引取代手写位打包） | `shader_package/permutation.rs:1-75` |
| `LobeMask` | 10 位瓣掩码（KHR 区 7 + 扩展区 3） | `surface.rs:411-465` |
| KHR / EXTENSION 双密集段 | 规范瓣槽 0..6，Prism 扩展槽自 `EXTENSION_SLOT_BASE=16` 起，bits 7..15 预留给未来规范瓣 | §3.4；`refactor_zh` §4.5 约束② |
| `DerivationClass` | `OfflineShipped / OfflineEditorOnly / RuntimeDerived / DriverLocal`（PSO 缓存属 DriverLocal，永不进 DDC） | `prism_derived_data_contract_zh.md:89-94`；`prism_asset_build_design_zh.md:146-156` |
| 一条执行路径 / 开放 über / 无全局硬顶 / 无塌缩降级 / 响应与组合正交 / 传递无关缓存 / 统一执行模型 | 七条 headline 不变量 | `docs/refactoring-dev/00-overview_refactor_zh.md` |
| 补瓣 / 向下投影 / identity-only / header 常驻 | 加法式扩展瓣 / `project_to_openpbr` 有损投影 / 只在烘焙或 pass 期解析、零头部槽 / 占用 64 槽摘要的 kind | §3.6、§3.7 |
| `StylizedParams` / `LobeSampleBudget` / `ShadingQuality` | 全局风格量 / 瓣采样预算 / 画质档 | `stylized.rs:70`；`quality.rs:109,43` |

## 附录 B. v2 → v3 差异台账

| # | v2 说法 | v3（code-truth） | 证据 |
|---|---|---|---|
| 1 | `MATERIAL_ABI_VERSION = 4` | **6** | `record.rs:18`；`surface.toml:23`；`schema_parity.rs:19` |
| 2 | 头部含 `closure_graph_offset`，运行期按需遍历闭包图 | 头部 27×u32 = **108 B**，**无**闭包图字段；运行时 graph-free | `record.rs:135-218`；commit `f5172023c` |
| 3 | 闭包 slab 深度上限 4 + `ClosureSlabTooDeep`；纹理槽硬顶 | 全部删除 | commit `1e9d1812b`；`tests.rs:83,200` 仅存历史注释 |
| 4 | `SURFACE_CORE_WORDS = 12`、`SURFACE_LOBE_WORDS = 4`、`LobeMask::COUNT = 6` | **20 / 16 / 10**；`BY_SLOT=[8,16,4,8,16,16,4,0×9,4,4,4]`（19 项） | `surface_layout.rs:11,20,28-34` |
| 5 | 以互斥枚举 `MaterialShadingModel` 为机制；含 `Npr*` / `Custom*` 渲染类 | 照明轴 × 瓣集正交组合；渲染类 11 值且不含 Npr*/Custom* | `axis.rs`；`record.rs:49-64` |
| 6 | 结构哈希驱动 ABI 版本 | 不存在结构哈希；手写常量 + 测试 + schema `abi_version` | 本文件 §3.5 |
| 7 | 质量档/缓存命中参与置换键 | 禁止：键 = 瓣集 + 特征 flags（不变式 I2） | `refactor_zh:568` |
| 8 | NPR 作为独立管线/crate（`prism_render_npr` 计划） | 原地实现在 `prism_render_shading`，无独立 crate | §14；`stylized.rs:185` |
| 9 | `evaluate_toon_direct` 被重命名 | 新函数 `evaluate_stylized_direct` **追加**；旧名保留为一行包装 | `stylized.rs:185,251` |
| 10 | `StylizedParams.legacy_toon` 是字段 | 是 `const fn` 预设构造器 | `stylized.rs:112` |
| 11 | §6.2 子系统注册表 | 已由 `FeatureCatalog` 与 `render_fabric::builtin_catalog` 提供机器可检验目录；服务闭包与 provider 唯一性有测试 | `pkg/prism_render_architecture/src/render_fabric/` |
| 12 | `prism_physics_core` ≈ 24k 行、未被渲染依赖 | 129 271 行 / 411 文件，且是 `prism_render_architecture` 的硬依赖 | §6.2；`Cargo.toml:13` |
| 13 | "68 个 .wesl / 0 .wgsl" | **400** 个 `.wesl` + 约 130 个 `.wgsl` | §6.4 |
| 14 | 约 40 个 pass 手排在 `plugin.rs`（1655 行） | 手排点在 `pkg/prism_render_scene/src/shading/plugin.rs`（1655 行）；`prism_render_scene/src/plugin.rs` 仅 165 行；声明式帧图覆盖 9 个 pass，全场景约 69 个 `*_pass` | §9.2 ④ |
| 15 | 虚拟纹理未接线 | 采样路径已接线，缺真实 VT 行与头部 `vt_page_offset/count` | `bindings.rs:174,195`；`material_sample.wesl:264` |
| 16 | 逐区域照明"未完成" | 已落地（CPU + WESL + 可见性光栅） | `classification.rs:686,716-722`；`visibility_raster.wesl:47,127` |
| 17 | reactive mask 三入口闭合（v2 自述已完成） | CPU 闭合成立，但 GPU 孪生 0 命中 `reactive` | §5.3、§4.5 缺口 4 |
| 18 | 头部尺寸在文内两处自相矛盾 | 统一为 27w/108B，并给出字节链 20w→19w→+4 PCM→+4 动态 | §3.3 |

## 附录 C. 代码索引（crate → 角色 → 锚点）

| crate / 目录 | 角色 | 关键锚点 |
|---|---|---|
| `pkg/prism_render_material` | 材质 ABI：身份、闭包 IR、紧凑记录、校验 | `record.rs:18,135-218`；`axis.rs:90,162-192`；`surface.rs:411-465,577`；`ir.rs`；`openpbr.rs:118` |
| `pkg/prism_material_schema` | 单一事实源 + 生成器 | `schema/surface.toml`（`abi_version=6`、`[core] words=20`）；`src/codegen.rs`；`--bin prism-material-codegen` |
| `pkg/prism_material_pcm` | 能力基座（九类 kind、四区 × 16 槽、载入期协商） | `namespace.rs:89-96,213-232`；`registry.rs:79`；`manifest.rs:238`；`negotiate.rs` |
| `pkg/prism_render_material_authoring` | 作者层（archetype/节点图，1895 行） | `authoring`：语义常量 `FACE_SHADOW_SDF_SEMANTIC` 等 |
| `pkg/prism_render_material_gpu` | 块压缩纹理硬件解码 oracle（**不是**材质 ABI crate） | `BlockOracle::try_new` |
| `pkg/prism_render_architecture` | 共享基底：能力协商、帧图、着色包/PSO 缓存、GPU 场景、描述符堆、分页、子系统 | §8.1、§9.2；`shader_package/pso_cache/mod.rs:61`；`frame_graph/builder.rs:6-29`；`gpu_scene/`；`descriptor_heap/` |
| `pkg/prism_render_shading` | 响应函数与光照/阴影/后处理算法（131 510 行） | `classification.rs:433,453,673`；`resolve.rs:760-771`；`lighting.rs:24,369`；`stylized.rs:185`；`gi/`、`shadow/virtual_sm/`、`ao/` |
| `pkg/prism_render_scene` | 场景编排、pass、WESL 主体（175 个 `.wesl`） | `shading/plugin.rs`（1655 行）；`shading/graph.rs:76-139`；`motion/encode.rs:26`；`shaders/material_unpack.wesl` |
| `pkg/prism_render_app` | 管线缓存、帧编排、Bevy 集成 | `pipeline_cache.rs:47,509`；`frame.rs:121,155`；`sets.rs:126` |
| `pkg/prism_render_graph` / `prism_render_driver` | 被执行帧图 / RHI 契约与缓存 | `graph.rs:162-202`；`pass.rs:63,90`；`driver` 的 `ContentCache` |
| `pkg/prism_render_gpudriven` | GPU 驱动绘制与间接参数 | `scene_draw.rs:407,421,456` |
| `pkg/prism_virtual_geometry_gpu` / `prism_virtual_texture_gpu` / `prism_hair_gpu` / `prism_volumetric_gpu` / `prism_water_render` | GPU 孪生与真机验证 | `prism_virtual_geometry_gpu/src/raster.rs:3,30`、`gpu_parity.rs:84`；`prism_water_render/src/ocean.rs:34-37` |
| `pkg/prism_physics_core` / `prism_physics_gpu` | 共享模拟核心 / **未接线**的 GPU 物理 | `prism_render_architecture/Cargo.toml:13`；`prism_physics_gpu`（110 `.wgsl`，无消费方） |

---

**变更历史**：v1（初稿）→ v2（2026-09-30，wgpu + WESL 版，含 ABI v4 与闭包图描述）→ **v3（2026-10-14，与实现对齐：ABI v6、生成式布局、PCM、注册驱动分类、四层收口台账）**。
