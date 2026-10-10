# Prism 次世代 AAA 引擎 — 组件缺口全景与脱 Bevy 蓝图

> 本文回答三件事：**(1) 要成为完全独立的顶级次世代 AAA 引擎，还缺哪些组件；(2) 现有 `pkg/` 已覆盖到什么程度、哪些仍深耦合 Bevy；(3) 按什么顺序脱离 Bevy。**
> 借形态不抄码。对标 Unreal Engine 5 / Unity DOTS / Godot 4 / 寒霜(Frostbite) / 决意(Decima) / 寒霜级大世界管线，仅取架构形态与经典数值，**不含任何 Unreal/Unity 源码或衍生代码，不含 AI/ML 内容**。
>
> - 版本: v0.1（现状盘点 + 缺口清单 + 脱 Bevy 关键路径；除已存在的 `pkg/` crate 外，所有"待建"项均为 PLANNED，无代码）
> - 配套文档: `prism_ecs_design_zh.md`（ECS 内核）、`prism_bevy_refactor_plan_zh.md`（62 crate 处置总表）、`prism_aaa_advanced_features_zh.md`（渲染高级特性）、各子系统 `*_design_zh.md`
> - 口径: "Have"=`pkg/` 已有实现；"Coupled"=已有实现但深度依赖 `bevy_*`，脱 Bevy 需换地基；"Missing"=尚无对应 crate，需自建

---

## 0. 结论先行（TL;DR）

`pkg/` 的**算法与效果层已非常完整**（渲染/物理/音频/UI 四大件均达 AAA 形态），但它们都"长"在 Bevy 的**运行时地基**上。要完全脱离 Bevy，缺口不在"效果"，而在**地基（foundation）+ 平台 I/O + gameplay 中间层 + 工具链**这四类"看不见但缺一不可"的组件。

**最关键的 8 个地基 crate（脱 Bevy 的命门，全部 Missing）**：

| 优先级 | crate | 作用 | 替代的 Bevy |
|---|---|---|---|
| P0 | `prism_math` | SIMD 数学（glam 封装的自有门面，设计见 `prism_math_design_zh.md`） | `bevy_math` |
| P0 | `prism_ecs` | 权威仿真内核（见专文） | `bevy_ecs` |
| P0 | `prism_tasks` | work-stealing + fiber 作业图（设计见 `prism_tasks_design_zh.md`） | `bevy_tasks` |
| P0 | `prism_app` | App/Plugin/Schedule/固定阶段/主循环（设计见 `prism_app_design_zh.md`） | `bevy_app` |
| P0 | `prism_reflect` | 反射/序列化/类型注册 | `bevy_reflect` |
| P1 | `prism_asset` | 资产 ID/加载器/依赖图/热重载/烘焙 | `bevy_asset` |
| P1 | `prism_render_driver`(RHI) | 后端无关图形抽象（Vulkan/Metal/DX12/WebGPU，设计见 `prism_render_driver_design_zh.md`） | `bevy_render` 底层 |
| P1 | `prism_window` + `prism_input` | 窗口/事件循环/输入设备 | `bevy_winit` + `bevy_input` |

搞定这 8 个，`pkg` 里只有 7 个深耦合 crate（见 §2）需要"换地基"重接；其余 ~50 个只依赖 `bevy_math`，换成 `prism_math` 即近零成本迁移。

---

## 1. 现状盘点：`pkg/` 已覆盖的能力

`pkg/` 共 59 个 crate，效果层高度完整：

- **渲染（13 crate）**: 虚拟几何(Nanite 式)、ReSTIR DI/GI、虚拟阴影 VSM、体积雾/云、毛发/皮毛、水体 FFT、材质 über-BSDF、可见性/剔除、架构外壳。
- **物理（4 crate）**: XPBD、软体、流体、MPM、断裂、几何、GPU 求解。
- **音频（15 crate）**: 核心/空间/HRTF/音乐/程序化/重采样/实时/母线治理/输出/设备。
- **UI（30 crate）**: 布局/样式/文本/动画/反应式/路由/状态/虚拟化/工作台/无障碍/热重载/时间旅行/检视器等——其中 29/30 **完全不碰 Bevy**。
- **GPU 子系统**: `prism_virtual_geometry_gpu`、`prism_volumetric_gpu`、`prism_hair_gpu`、`prism_physics_gpu`、`prism_render_material_gpu`。

> 换言之：**"引擎做什么好看/好玩"已经齐了；"引擎如何自立门户地跑起来"还缺地基。**

---

## 2. 现状盘点：仍深耦合 Bevy 的 7 个 crate

脱 Bevy 时需要"换地基并重接"的，只有这 7 个（其余只需把 `bevy_math` → `prism_math`）：

| crate | 依赖的 bevy_* | 换地基工作量 |
|---|---|---|
| `prism_render_scene` | app/asset/camera/color/core_pipeline/derive/ecs/image/light/material/mesh/pbr/platform/render/shader/transform（15+） | **最重**。渲染世界提取、相机、光照、材质、网格、渲染图全在此接缝 |
| `prism_render_material` | asset/color/image/material/pbr | 材质资产与 PBR 定义重接到 `prism_asset`/自有材质模型 |
| `prism_render_visibility` | camera/render/shape/transform | 剔除输入（相机/变换/包围体）重接 |
| `prism_bevy` | app/camera/ecs/shape/transform | **本就是 Bevy 桥接层**，脱 Bevy 后整体被 `prism_app` 取代/废弃 |
| `prism_audio_bevy` | app/ecs/transform | 音频世界的 ECS 对接层，重接到 `prism_ecs` |
| `prism_audio_device` | bevy_audio | 设备后端重接到自有 `prism_audio_output`/平台音频 |
| `prism_ui_ecs` | ecs | 仅依赖 `bevy_ecs`，换 `prism_ecs` 即可 |

---

## 3. AAA 引擎组件全景矩阵（缺口清单）

状态：✅ Have ／ 🟡 Coupled（有实现但绑 Bevy）／ ❌ Missing（需自建）。

### 3.1 地基层 Foundation（脱 Bevy 命门，几乎全缺）

| 组件 | 状态 | 说明 / 对标 |
|---|---|---|
| 数学 `prism_math` | ❌ | glam/SIMD 门面，向量/矩阵/四元数/变换/曲线/定点（确定性档）。**设计见 `prism_math_design_zh.md`** |
| ECS 内核 `prism_ecs` | ❌ | 见 `prism_ecs_design_zh.md`；chunk SoA + 关系 + 回滚 |
| 任务系统 `prism_tasks` | ❌ | work-stealing + fiber job graph（Naughty Dog/DOOM 形态）。**设计见 `prism_tasks_design_zh.md`** |
| App/插件 `prism_app` | ❌ | App/Plugin/Schedule/固定与可变时间步/主循环/状态机。**设计见 `prism_app_design_zh.md`** |
| 反射 `prism_reflect` | ❌ | 类型注册/序列化/动态字段/脚本桥（flecs meta/Bevy reflect 形态）。**设计见 `prism_reflect_design_zh.md`** |
| 时间 `prism_time` | ❌ | 固定步长、插值 alpha、时间缩放、确定性时钟。**设计见 `prism_time_design_zh.md`** |
| 诊断/日志 `prism_diagnostic` | ❌ | tracing、帧统计、计数器、断言、崩溃转储。**设计见 `prism_diagnostic_design_zh.md`** |
| 容器/工具 `prism_utils` | ❌ | 稳定哈希、SmallVec/竞技场/句柄表/位集、`no_std` 友好。**设计见 `prism_utils_design_zh.md`** |
| 平台抽象 `prism_platform` | ❌ | OS 差异、文件系统、时钟、线程、动态库。**设计见 `prism_platform_design_zh.md`** |

### 3.2 平台 I/O 层（部分可薄封装现成库）

| 组件 | 状态 | 说明 |
|---|---|---|
| 窗口 `prism_window` | ❌ | winit 薄封装或自研；多窗口/DPI/全屏/HDR 输出 |
| 输入 `prism_input` | ❌ | 键鼠/手柄/触控/手势，动作映射（action map） |
| 图形 RHI `prism_render_driver` | ❌ | **关键**。wgpu 或自研 Vulkan/Metal/DX12 后端抽象；队列/屏障/描述符/管线缓存。**设计见 `prism_render_driver_design_zh.md`** |
| 渲染图 `prism_render_graph` | ❌ | pass/资源别名/瞬态分配/自动屏障（frame graph 形态）。**设计见 `prism_render_graph_design_zh.md`** |
| 着色器工具 `prism_shader` | ❌ | WGSL/HLSL 预处理、反射、变体、热重载、离线编译 |
| 音频后端 `prism_audio_backend` | 🟡 | 现为 `prism_audio_device`→`bevy_audio`；需换 CPAL/平台音频 |
| 网络传输 `prism_net_transport` | ❌ | UDP/可靠通道/QUIC；回滚网络的管道 |
| 存储/VFS `prism_vfs` | ❌ | 虚拟文件系统、pak/打包、按需流送 I/O |

### 3.3 资产与内容管线

| 组件 | 状态 | 说明 |
|---|---|---|
| 资产内核 `prism_asset` | ❌ | 句柄/异步加载/依赖图/热重载/引用计数/生命周期 |
| 导入器 `prism_asset_import` | ❌ | glTF/FBX/USD、PNG/KTX2/basis、WAV/OGG、字体 |
| 烘焙/离线 `prism_asset_bake` | ❌ | 平台相关压缩、mesh 优化、虚拟几何/虚拟纹理预处理 |
| 虚拟纹理流送 | 🟡 | 渲染侧有，但缺独立于 Bevy 的流送 I/O 调度 |
| 材质系统 | 🟡 | `prism_render_material` 绑 `bevy_material/pbr`，需自有材质图/编译 |

### 3.4 场景与世界

| 组件 | 状态 | 说明 |
|---|---|---|
| 变换/层级 `prism_transform` | ❌ | 局部/全局变换传播（配合 ECS 关系）。**设计见 `prism_transform_design_zh.md`** |
| 场景序列化 `prism_scene` | ❌ | 场景/预制体(prefab)的保存/加载/实例化/覆盖 |
| 世界分区流送 | 🟡 | 设计见 `prism_world_system_design_zh.md`；运行时流送缺地基 |
| 大世界原点重定位 | ❌ | 64-bit 浮点 + cell 偏移（Star Citizen 形态），需 ECS 支撑 |
| HLOD/数据层 | ❌ | 分层 LOD、数据层开关、运行时加载 |

### 3.5 Gameplay 中间层（几乎全缺）

| 组件 | 状态 | 说明 |
|---|---|---|
| 动画运行时 `prism_anim_runtime` | 🟡 | 设计见 `prism_animation_engine_design_zh.md`；缺脱 Bevy 的骨骼/蒙皮/状态机/IK/混合运行时 |
| Gameplay 框架 `prism_gameplay` | 🟡 | 设计见 `prism_gameplay_design_zh.md`；缺实现（能力系统/属性/标签/事件） |
| 游戏创作套件 `prism_gc_*` | ⬜ | 设计见 `prism_game_creator_design_zh.md`；建于 `prism_gameplay` 之上的游戏模板/内容系统/创作工作流（Lyra 等价层），全 PLANNED，无代码 |
| 脚本 `prism_script` | ❌ | Rust 热重载 / WASM / Lua 绑定，经 `prism_reflect` 暴露 |
| AI 导航 `prism_navigation` | ❌ | navmesh 生成/寻路/避障/群体 |
| 行为 `prism_behavior` | ❌ | 行为树/状态机/黑板/效用 AI（经典算法，无 ML） |
| 网络复制 `prism_replication` | ❌ | 确定性回滚/预测（Quantum/GGPO 形态）+ 快照插值 |
| 粒子/VFX 运行时 | ✅/🟡 | `prism_particle_engine_design_zh.md` + GPU 粒子；对接层待重接 |

### 3.6 工具链与可观测性

| 组件 | 状态 | 说明 |
|---|---|---|
| 编辑器框架 `prism_editor` | 🟡 | 设计见 `prism_editor_framework_design_zh.md`；UI 层(`prism_ui_*`)已就绪，缺编辑器运行时与世界检视 |
| ECS 检视器/火焰图 | ❌ | 实体/组件检视、system 火焰图、时间旅行（UI 侧 `prism_ui_timetravel` 可复用） |
| 性能分析 `prism_profiler` | ❌ | CPU/GPU 计时、tracy/chrome-trace 导出、帧捕获 |
| 构建/CI | 🟡 | 仓库有 Cargo workspace；缺引擎级打包/平台导出流水线 |

---

## 4. 依赖分层（脱 Bevy 后的目标形态）

```
L7  工具链   prism_editor / profiler / 检视器
L6  Gameplay prism_gameplay / anim_runtime / navigation / behavior / replication / script
L5  内容     prism_scene / asset / asset_import / asset_bake / vfs
L4  表现     prism_render_scene* / render_graph / render_driver(RHI) / shader / audio / ui
L3  仿真     prism_ecs / transform / physics / 世界分区
L2  运行时   prism_app / tasks / time / input / window
L1  地基     prism_math / reflect / diagnostic / utils / platform
L0  外部     wgpu/winit/cpal 等（经 L2/L4 薄封装隔离，可替换）
```

规则：依赖严格向下；`pkg` 现有效果 crate 落在 L3/L4，只要地基 L1/L2 就位即可逐个"抽掉 bevy_*、换 prism_*"。

---

## 5. 脱 Bevy 关键路径（里程碑）

与 `prism_bevy_refactor_plan_zh.md` 的"62 crate 处置"互补：那份讲"每个 bevy crate 怎么处置"，本节讲"自建地基的推进顺序"。

- **M0 地基可编译**：`prism_math` + `prism_ecs`(骨架) + `prism_tasks` + `prism_app` 最小主循环跑通空 World + Schedule。
- **M1 反射与资产**：`prism_reflect` + `prism_asset` + `prism_time`/`diagnostic`/`utils`，支撑序列化与热重载。
- **M2 平台与 RHI**：`prism_window` + `prism_input` + `prism_render_driver`(先包 wgpu) + `prism_render_graph` + `prism_shader`，画出三角形。
- **M3 渲染重接**：`prism_render_scene/material/visibility` 抽掉 `bevy_*`，接 `prism_ecs`/`prism_asset`/RHI；点亮一个 PBR 场景。
- **M4 其余解耦**：`prism_audio_bevy/device`、`prism_ui_ecs` 换 `prism_ecs`；~50 个 `bevy_math` 用户批量换 `prism_math`。
- **M5 Gameplay**：`prism_transform`/`scene`、动画运行时、gameplay 框架、脚本、导航、复制。
- **M6 工具链**：编辑器运行时、ECS 检视器/火焰图、profiler、平台打包导出。

每个里程碑以"基准即规格"验收（帧时间预算见 `prism_aaa_advanced_features_zh.md` §7）。

---

## 6. 风险与诚实边界

- 除 `pkg/` 已存在 crate 外，本文所有"待建"组件**均无代码**，为 PLANNED 设计目标。
- RHI 自研（Vulkan/Metal/DX12）工作量巨大；**建议 L0 先包 `wgpu` 以 RHI 门面隔离**，待热点明确再按平台替换，避免过早自研拖垮进度。
- 确定性回滚网络要求跨平台浮点一致，仅在 `determinism` 档位 + 定点/软件数学路径下保证。
- 最大单点风险是 `prism_render_scene` 的渲染世界提取接缝；建议最先为它定义稳定的"提取 trait"，让渲染与 ECS 解耦。
- 所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。
