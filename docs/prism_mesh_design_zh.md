# Prism Mesh 系统设计：顶级次世代 AAA 级几何数据模型与运行时几何管线

> 状态：设计规格与逐项验证记录（v2）；实现结论以 §26 的证据为准
> 面向版本：Prism `0.20-dev` 当前仓库契约
> 目标读者：渲染、资产工具链、平台、动画/物理与性能工程人员
> 本文依据：当前仓库代码、CPU 单测、RenderApp 真实 GPU 像素读回、蒙皮/虚拟几何/GDeflate/OMM GPU twin 与 Mesh CPU 基准；预算表与设备实测严格分栏，不用预算或 Null/CPU 回退替代真实 GPU 结论
> 关联文档：`prism_rendering_architecture_zh.md`（全局渲染架构）、`prism_asset_design_zh.md`（资产内核）、`prism_asset_build_design_zh.md`（构建/烘焙）、`prism_material_pipeline_refactor_zh.md`（材质与着色 ABI）、`prism_scene_design_zh.md`（场景与实例）、`prism_aaa_advanced_features_zh.md`（AAA 特性总表）
> 最后更新：2026-10-22

## 0. 阅读地图与本文责任范围

本文只回答一个问题：**Prism 的“网格（Mesh）”从美术授权到 GPU 光栅、再到光追与物理派生，应该是一套什么样的系统**。它把散落在多个 crate 里的能力（`prism_mesh` 的数据模型、`prism_render_architecture::virtual_geometry` 的 Cluster/LOD/流送、`geometry` 的稳定几何表、`deformation` 的蒙皮缓存、`prism_micromap` 的 OMM、`prism_gdeflate_gpu` 的 GPU 解压）收敛成**一个模型、一个身份体系、三个产物面（运行时网格 / 虚拟几何页 / 派生几何）**。

本文**不**重新定义：GPU Scene 的实例身份（见渲染架构 §5）、材质着色模型与参数 ABI（见材质管线重构文档）、资产构建 DAG 与 DDC（见资产构建文档）、场景/Prefab 语义（见场景文档）。这些文档是本文的上游约束；本文只规定它们与几何相交的那一层契约。

全文分为四块：

- **§1–§4 目标与对标**：要达到什么、不做什么、借谁的形态、分几档平台。
- **§5–§9 数据模型与形变**：一件 Mesh 的授权模型、顶点流与量化、几何变体、蒙皮/Morph/布料。
- **§10–§17 运行时几何管线**：LOD、虚拟几何、GPU 驻留与上传、可见性与提交、材质 ABI、光追/物理派生、压缩、构建集成。
- **§18–§27 工程兑现**：性能预算、测试、维护与演进纪律、公共 API、实施矩阵、ADR、风险、验收矩阵。

## 1. 摘要

一句话定位：**Mesh 是“几何的授权真相”，渲染器只是它的一种视图。** Prism 用同一个几何模型同时服务标准 draw、GPU-driven 实例化、虚拟几何、蒙皮/形变、光追、物理与编辑器预览，而不是让每条路径各自持有一份网格真相。

六条核心结论：

1. **一份授权数据 + 一次构建 + 多个产物面。** `Mesh`（authoring/IR）是唯一真相；`prism_asset_build_core` 把它派生成运行时网格、虚拟几何页、碰撞几何、光追几何、缩略图等制品。任何运行时路径不得反向修改授权数据。
2. **数据半个 crate，GPU 半个 crate。** `pkg/prism_mesh` 只持有与设备无关的几何数据（属性、索引、拓扑、包围体、校验、打包），`#![forbid(unsafe_code)]`、无 ECS、无 GPU 句柄；GPU 形态（缓冲、上传、生命周期）归 `prism_render_app`/`prism_render_scene`。职责分离直接保证可测试性。
3. **顶点词表属于 RHI，不属于 Mesh。** 属性格式、缓冲布局、拓扑、索引宽度直接复用 `prism_render_driver` 的类型，避免两套枚举漂移（现状已如此，本文将其固化为契约）。
4. **顶点流分流 + 域相关量化。** 位置/蒙皮/形变/材质属性按访问模式拆流，深度与阴影 pass 只绑定它真正需要的流；量化策略按几何变体（静态 vs 蒙皮 vs 虚拟几何）分级，而不是全局一刀切。
5. **虚拟几何是主线，不是可选彩蛋。** 静态高模默认走离线 Cluster 构建 + 连续误差 LOD + 分页流送 + 软件/硬件双光栅；标准 mesh 路径在小网格、动态几何、低端档与非虚拟化场景下长期共存，两者共享同一套 Cluster cull 输入与材质解析契约。
6. **基准即规格，证据与设计分栏。** 每项性能/画质目标都写成可测量的门槛；实现证据、运行策略和能力降级在同一矩阵中记录，不用宣传语替代证据。

## 2. 目标、非目标与硬约束

### 2.1 效果目标

- **任意复杂度静态几何可达**：高模无需手工离散 LOD；导入即自动生成分页 Cluster 资产与独立 QEM LOD 资产集合，屏幕上百万级可见三角形仍保持稳定帧时。
- **无 LOD 突变的连续过渡**：屏幕空间误差驱动选择，配合滞回与 dither，消除传统 LOD 切换的跳变与闪烁。
- **形变几何质量不退化**：蒙皮角色、Morph、布料、顶点动画在深度、阴影、速度、光追各 pass 中共享同一形变结果，不出现“主视图对、阴影错”的错位。
- **光追与光栅几何一致**：同一件资产的 BLAS、OMM（alpha 覆盖）、LOD 选择与光栅可见性来自同一几何身份，不产生两套互相矛盾的轮廓。
- **叙事级细节预算**：单场景几何页预算、Cluster 预算、蒙皮顶点预算可配置、可观测、可解释降级。

### 2.2 性能目标

- **CPU 提交成本与可见工作量而非场景总量成正比**：百万级 Cluster 场景下 CPU 每帧只提交增量与策略参数。
- **带宽受控**：位置在虚拟几何路径下按 Cluster 局部量化到 16 bit；索引按授权宽度保留；页压缩使磁盘与 PCIe 传输量显著低于未压缩形态。
- **驻留可预测**：几何页驻留占用有明确上限，超限时按优先级驱逐而不是 OOM；首帧墙（first-frame wall）优先于远景完整度。
- **上传不阻塞帧**：异步拷贝队列 + 环形 staging，上传预算按帧分配，避免瞬时大块分配造成卡顿。
- **形变一次、多 pass 复用**：蒙皮/形变结果写入形变缓存，深度、阴影、速度、光追 refit 共享，避免重复计算。

### 2.3 易维护 / 易扩展目标

- **稳定契约**：几何数据的公共 API 承诺语义（属性语义、索引宽度、包围体含义），不承诺单一物理缓冲布局；后端可重新打包。
- **注册表而非分支**：新增顶点语义、拓扑种类、形变器、LOD 选择器、压缩器通过注册项扩展，不改 `Mesh` 核心类型、不改既有匹配分支。
- **显式版本化**：所有落盘几何制品（虚拟几何页、meshlet、压缩流）必须带 `magic + schema_version + build_settings_hash`，旧缓存可检测并重建，禁止静默错读。
- **可解释性**：任何“换了一条路径 / 降了一级 LOD / 丢了一页”的决定都能在诊断视图中说明原因。
- **弃用纪律**：原生 Prism RHI 统一使用 `pkg/prism_mesh`；`crates/bevy_mesh` 仅位于 Bevy 接入与兼容路径，转换规则与替代条件见 §20、§22。

### 2.4 非目标

- 当前通用路径不启用硬件曲面细分（Hull/Domain/Geometry Shader）作为常规几何路径；细分作为离线构建步骤或显式能力项。
- 不要求动态拓扑（每帧重建连接关系）进入虚拟几何路径；这类几何走程序化/标准路径，虚拟几何只接受“拓扑在构建期固定”的几何。
- 可变形几何走标准 GPU skinning；Deformable Cluster 作为独立 profile，启用时仍复用同一 Cluster、LOD、驻留与可见性契约。
- 不承诺跨平台逐像素一致；承诺语义一致、质量档位可解释、误差有界。
- 不用单一超级顶点格式覆盖一切；格式矩阵按变体分层，宁可有多个明确格式，不要一个“万能壳”。

### 2.5 硬约束

- `pkg/prism_mesh` 必须保持设备无关：不链接图形后端、不持有 GPU 句柄、不依赖 ECS。
- 顶点词表必须与 `prism_render_driver` 保持一致：`VertexFormat`、`VertexBufferLayout`、`PrimitiveTopology`、`IndexFormat`、`VertexStepMode` 是唯一来源。
- 必须遵守设备限制：`prism_render_driver::Limits` 的 `max_vertex_buffers`、`max_vertex_attributes`、`max_storage_buffer_binding_size` 等是几何布局的硬上界（`baseline()` 为 8 个顶点缓冲 / 16 个属性 / 128 MiB storage，`desktop()` 为 16 / 32 / 2 GiB）。
- 不得引入 UE/Unity 源码或派生代码；对标只借形态，算法使用公开经典方法。
- AI/机器学习方法不得进入几何核心路径（与仓库其他模块的既有约束一致）。

## 3. 参考产品对标与取舍

### 3.1 对标表

| 产品 / 技术 | 借鉴的形态 | 明确不照搬的部分 |
|---|---|---|
| UE5 Nanite（虚拟几何） | 离线 Cluster + 层级选择、屏幕空间误差 LOD、Cluster 剔除、软件/硬件双光栅、Visibility Buffer、分页流送、按 Cluster 局部量化 | 单一大 DAG 的整体实现细节；Prism v1 使用可验证的 Cluster 资产 + 独立 LOD 资产集合，保留显式 schema 与可插拔光栅后端 |
| UE5 StaticMesh / SkeletalMesh / Geometry Script | 资产与派生几何分离；同一资产派生渲染/碰撞/导航/光追几何；编辑期脚本化几何 | 不在运行时依赖编辑器脚本层 |
| Unity Mesh / DOTS / Entities Graphics | 顶点流分流与紧凑格式；批处理友好的共享布局；SRP Batcher 式的“材质键 → 状态缓存” | 不采用托管运行时与 GC 依赖 |
| meshoptimizer / Draco / MeshOpt 编码 | 索引重排提升缓存局部性；顶点 cache-aware 编码；属性量化与边拆分 | 只借算法，实现独立且可被 GPU twin 校验 |
| id Tech / Frostbite 的虚拟几何路线 | Compute 软件光栅到 Visibility Buffer；按需页请求与预算 | 不做与特定主机强绑定的固定管线假设 |
| Metro Exodus / 现代主机引擎的 Mesh Shader 路径 | Mesh Shader 生成可见 Cluster 与间接参数 | Prism 不把高层算法绑定在 Mesh Shader 上，保留 Compute/SW 回退 |
| DirectStorage / RTX IO / GDeflate | GPU 直通解压与分块 I/O；容器化流送 | 解压内核与容器格式独立实现 |
| UE5 Virtual Shadow Map / OMM | 几何与阴影/光追的协同；alpha 覆盖微图 | 只是几何侧契约，阴影算法归别文档 |
| USD / glTF / Alembic | 授权数据分层（几何/骨架/动画/材质分开）+ 稳定子资产标识 | 不把 USD/glTF 顶层结构直接当作引擎运行时 ABI |
| Simplygon 类自动 LOD 工具 | 自动简化、误差报告、材质槽隔离 | 简化算法自研并与 Golden 指标挂钩 |

### 3.2 借什么、不借什么

**借形态**：固定 Cluster 分区 + 独立 QEM LOD 资产集合 + 屏幕空间误差，是当前实现采用的可验证形态；分流顶点 + 域量化是带宽优化的成熟做法；一份授权数据派生多产物是 AAA 内容流水线的共识。单一大 DAG 只作为外部技术对标，不作为 Prism 当前实现声明。

**不借结论**：不做“必须 Mesh Shader”“必须某家 GPU”“必须单一万能格式”的强假设。Prism 的高层算法与光栅后端解耦（§11.4），能力探测与档位选择（§4）保证同一场景语义在兼容档也能跑。

## 4. 档位化（capability / quality tier）

几何路径按设备能力与项目质量档选择，而不是按操作系统硬编码。档位是**策略输入**，不是分支散落点。

| 档位 | 目标设备 | 静态几何路径 | 形变路径 | 光追几何 | 量化策略 |
|---|---|---|---|---|---|
| `Compatible` | 老桌面、部分移动、WebGPU | 标准 GPU-driven indexed draw；手工/自动离散 LOD | GPU skinning（4 影响） | BLAS（低 LOD）+ SDF 兜底 | 半精度法线/UV，位置 32 bit |
| `Standard` | 主流桌面、现代主机 | 虚拟几何 Compute 软件光栅路径 | GPU skinning（4–8 影响）+ morph | BLAS + OMM | 虚拟几何位置 16 bit 局部量化 |
| `Ultra` | 高端桌面、当代主机 | 虚拟几何 + HW Mesh Shader 路径 | Deformable Cluster profile + 形变缓存 | BLAS refit + OMM + RT LOD | 全量化 + 页压缩 + GPU 直通解压 |
| `Offline` | 参考路径追踪 / 电影 | 全精度几何，不做运行时量化 | 全精度形变，缓存到磁盘 | 完整 BVH + OMM | 仅无损压缩，确定性优先 |

档位选择由 `prism_render_driver::DeviceCapabilities` + 项目 profile + 运行时微基准共同决定；档位切换必须保持**场景语义不变**（同一实例、同一材质、同一可见性意图），只改变实现路径与精度。

## 5. 分层架构与 crate 职责

### 5.1 六层

```text
① Authoring / IR        美术工具、FBX/USD/glTF、程序化生成、编辑器脚本
        │  授权数据 + 稳定子资产标识
        ▼
② Mesh Asset (prism_mesh)   属性/索引/拓扑/包围体/校验/内容哈希
        │  （设备无关的数据半边，可单测、可断言）
        ▼
③ Build & Bake (prism_asset_build_core)  IR → 运行时网格 / 虚拟几何页 / 碰撞 / 光追 / 预览
        │  显式 schema 版本 + build_settings_hash，落 CAS/DDC
        ▼
④ GPU Residency (prism_render_app / prism_render_scene)
        │  缓冲类、上传、子分配、驻留状态机、页表
        ▼
⑤ Geometry Consumers（互不重复真相）
        │  标准 draw / GPU-driven 实例 / 虚拟几何 / 光追 / 物理 / 编辑器预览
        ▼
⑥ Deformation (prism_mesh::deformation + prism_render_app::DeformationCache)
           蒙皮/Morph/布料/顶点动画 → 形变缓存 → 多 pass 复用
```

层级依赖只向下。任何消费者都不得重新解包授权数据，只能消费构建期产物与其运行时句柄。

### 5.2 crate / 模块职责布局

```text
pkg/prism_mesh                      设备无关几何数据（现状已落地，本文扩展其职责）
  ├─ attribute.rs    属性语义注册表 + 类型化取值
  ├─ indices.rs      索引（保留授权宽度）
  ├─ topology.rs     拓扑种类与图元数计算
  ├─ bounds.rs       AABB / 球 / 每 LOD 与每 Cluster 包围体
  ├─ streams.rs      顶点流划分与交错策略
  ├─ primitives/     内置图元生成（立方、球、胶囊、网格、地形片）
  ├─ validate.rs     完整校验规则集
  ├─ morph.rs        Morph target 数据（设备无关）
  ├─ skin.rs         蒙皮权重数据（设备无关）
  ├─ deformation.rs  批量 Morph/LBS/修正 CPU reference、输入校验与共享 storage ABI
  └─ hash.rs         确定性内容哈希（DDC 键）

pkg/prism_asset_build_core
  ├─ virtual_geometry_builder.rs    静态 Cluster 化与分页
  ├─ virtual_geometry_lod.rs        QEM LOD 链构建
  ├─ virtual_geometry_payload.rs    Cluster 局部量化 payload
  └─ virtual_geometry.rs            版本化制品校验与编码（§17）
pkg/prism_render_architecture
  ├─ geometry/        稳定几何表（GeometryRecord / GeometryLodRecord / ABI 校验）
  ├─ virtual_geometry/ Cluster cull、屏幕空间误差 LOD、页表、光栅路径、软件光栅、cut 选择
  ├─ deformation/     DeformationHandle / Kind / Budget，形变调度
  └─ gpu_scene/       实例身份与上传（几何只引用其句柄）
pkg/prism_render_scene/src/geometry   Bevy 接入层；几何算法消费 `prism_mesh`，`mesh_adapter` 只负责源资产转换
pkg/prism_math_gpu/src/skinning       GPU 蒙皮内核（CPU golden → GPU twin 校验）
pkg/prism_render_app/src/mesh/deformation.rs  原生 RHI 批量计算、持久资源、跨帧双缓冲
pkg/prism_micromap(_gpu)              OMM 烘焙与孪生
pkg/prism_virtual_geometry_gpu        vis-buffer 软件光栅 GPU 孪生（元/逐像素校验）
pkg/prism_gdeflate_gpu                页解压的去交错 GPU 孪生
crates/bevy_mesh                      Bevy 接入与兼容路径；原生 Prism RHI 以 prism_mesh 为准（§20）
```

### 5.3 当前实现与验证基线（2026-10-15）

- `prism_mesh::Mesh` 已提供属性、索引、拓扑、单流布局、分流布局、字节打包、包围体、校验、Morph、Skin、SubMesh 与确定性内容哈希；`#![forbid(unsafe_code)]`、无 ECS、无 GPU 句柄。
- `VirtualGeometryVersion=1` 已提供 Cluster 资产契约、量化 payload 编解码、层级/包围体校验、页表、驻留页存储、LOD 选择、滞回、速度预取、Cluster cull、RasterPath 分类、软件光栅与逐帧计划组合；构建侧同时提供固定 Cluster 分区和独立 QEM LOD 资产集合。
- 静态 LOD 使用属性感知 endpoint-QEM，保留幸存授权顶点的全部属性位模式、材质分区、开放边与重合拆分缝；输出请求/实际三角形数、源内容哈希、版本化配方哈希及包含位置量化的采样误差包络。策略允许按语义注册连续代价、分类约束或仅保留，不以新增格式硬编码新的简化分支。
- `GeometryTable` 已提供 generation 句柄、代际回收、LOD 驻留回退与 ABI 检查；`RenderMesh` 已把分流数据接入 queue、frame graph、draw slot 与固定页 GPU 池。
- 多消费者选择已收敛到 `prism_render_architecture::geometry::{MeshConsumerBinding, MeshConsumer, MeshRepresentation}`；`prism_render_app::mesh::consumer` 将静态 RenderMesh、共享形变 current/previous 输出和虚拟几何页绑定为同一份 GPU source contract。`prism_render_scene::geometry::consumer` 已作为 Bevy 场景适配桥接，主视图、Visibility、透明前向、阴影深度与虚拟阴影 caster 在触碰 `RenderMesh`/allocator 前统一经过 `MainView` 或 `Shadow` contract；缺页只使用显式 fallback，Collision 与 Derived Geometry 必须使用带 schema 的版本化派生制品。
- `prism_render_scene::geometry::shading` 已迁移到 `prism_mesh::Mesh`；Bevy 主世界 Mesh 只在 `geometry::mesh_adapter` 中转换一次，几何解析、拓扑处理与单测不再依赖 Bevy Mesh 数据结构。
- CPU 证据包括 `prism_mesh`、`prism_render_architecture`、资产流式化与渲染资产页池测试；GPU twin 证据覆盖虚拟几何计算内核与 GDeflate 去交错内核。RenderApp、虚拟几何、GDeflate、OMM 测试使用 `PRISM_REQUIRE_GPU=1`；数学蒙皮使用 `PRISM_MATH_GPU_REQUIRED=1`。强制运行时无适配器即失败，Null RHI 不计为 GPU 通过。
- `prism_mesh::deformation` 与 `prism_render_app::DeformationCache` 已验证批量 Morph → 4/8 影响 LBS → 外部修正，包含 50,000 顶点 CPU/GPU 对照、历史失效与直接顶点绘制像素读回；该证据不替代全部运行时阴影/深度/RT 消费者的集成验收。
- 本文把已验证契约、运行时降级、性能门槛和演进规则统一成可执行规格；代码实现以这些契约和测试作为唯一判定来源。

本次 2026-10-15 复核在当前工作树重新执行了 Mesh 核心证据：
`cargo test -p prism_mesh --lib --tests` 为 34 个库测试加 14 个集成测试全部通过；
`cargo test -p prism_asset_build_core --lib` 为 136 项全部通过，其中包含 12 项静态 LOD 定向测试；
Mesh、静态 LOD、RenderApp Mesh、RenderScene 几何适配和几何表相关 Rust 源文件的 `rustfmt --check` 全部通过。
真实 GPU 复核在沙盒外执行 `PRISM_REQUIRE_GPU=1 cargo test -p prism_virtual_geometry_gpu --test cluster_vertex_pull_parity -- --nocapture`，
1 项通过，确认 CLST v3 顶点拉取与 `Unorm8x4` 属性解码经过真实 Metal render pass 并完成颜色读回。
RenderApp Mesh 过滤集已在当前工作树源码快照的隔离验证副本中执行：
`CARGO_BUILD_JOBS=1 PRISM_REQUIRE_GPU=1 cargo test -p prism_render_app mesh:: --lib -- --nocapture`
共 20 项全部通过，包含真实 GPU 绘制、half 属性、批量形变和 current/previous 历史读回。
当前主工作树的完整 RenderApp 编译仍被并行 ECS 列同步改动在
`pkg/prism_render_app/src/ecs_columns.rs:451` 读取不存在的 `ChunkInfo::incarnation` 所阻断；隔离副本只将
该非 Mesh 字段读取替换为稳定的临时 chunk index，未修改 Mesh 生产代码，也未将该验证副本改动提交。

## 6. 核心数据模型：一件 Mesh 是什么

### 6.1 三层身份

几何必须区分三层身份，混用会产生悬挂引用与缓存失效错误：

| 层 | 类型 | 生命周期 | 用途 |
|---|---|---|---|
| 资产身份 | `AssetId`（稳定 GUID，见资产文档） | 跨会话、跨平台稳定 | DDC 键、引用、热重载 |
| 运行时几何身份 | `GeometryHandle { index, generation }`（见 `prism_render_architecture::gpu_scene::GeometryHandle`） | 会话内、按帧代际回收 | GPU Scene 表、间接 draw、可见性 |
| 页 / Cluster 身份 | `GeometryPageKey` / `ClusterKey`（`VIRTUAL_GEOMETRY_VERSION=1` 定义） | 驻留期间 | 流送、页表、cut 选择 |

约束：

- GPU 侧不得长期保存 ECS `Entity`；实例到几何的映射只存在于 bridge。
- 销毁后槽位进入延迟回收，至少等待 `frames_in_flight + async_compute_lag`。
- GPU 反馈（页请求、cut 结果）必须带 generation 或 frame epoch；CPU 忽略过期反馈。

### 6.2 授权数据模型（Authoring / IR）

授权态 `Mesh` 存“美术写了什么”，不是“GPU 怎么读”：

```text
MeshIR
├─ id               稳定子资产标识（在同一源文件内可寻址）
├─ topology         图元拓扑
├─ attributes[]     语义 + 类型化取值（每组一个元素/顶点）
├─ indices?         可选索引（保留授权宽度）
├─ submeshes[]      材质槽 + 索引区间（+ Cluster 区间，构建后填充）
├─ bounds           构建期计算的局部 AABB / 球
├─ skinning?        骨架引用 + 每顶点影响（索引 + 权重）
├─ morph_targets[]  命名增量流（稀疏或稠密）
└─ source_meta      源文件、节点路径、单位、坐标系、导入设置哈希
```

运行时 `Mesh`（现状 `pkg/prism_mesh::Mesh`）是 IR 的**已校验、可打包子集**：属性按 shader location 排序、格式已校验、索引已校验、包围体可按需派生。IR 的更多字段（skin/morph/submesh/source_meta）作为同一 crate 内的独立类型存在，不塞进 `Mesh` 本体，避免“一个类型承担所有变体”。

### 6.3 顶点属性注册表与格式矩阵

属性由**注册项**声明，而不是散落的字符串常量。注册项至少包含：语义名、语义类别（位置/法线/切线/UV/颜色/蒙皮/形变/自定义）、默认格式、shader location 分配策略、是否参与量化、是否参与包围体计算。

现状 `MeshVertexAttribute { name, format, shader_location }` 的三元组是正确的形状；本文补齐的是**注册表与格式矩阵**：

| 语义类别 | 推荐语义名 | 授权类型 | `Compatible`/`Standard` 出货格式 | `Ultra` 出货格式 | 字节/顶点 |
|---|---|---|---|---|---|
| 位置 | `Vertex_Position` | `[f32;3]` | `Float32x3` | CLST payload：局部 `Unorm16x3` 编码；标准/蒙皮：`Float32x3` | 12 / 6（payload） |
| 法线 | `Vertex_Normal` | `[f32;3]` | `Float32x3` / 显式 `Snorm8x4` | 标准网格与 CLST v3 保留原始格式和值 | 12 / 4 |
| 切线 | `Vertex_Tangent` | `[f32;4]` | `Float32x4` / 显式 `Snorm8x4` | 标准网格保留 | 16 / 4 |
| UV | `Vertex_Uv0..3` | `[f32;2]` | `Float32x2` / `Float16x2` | 标准网格保留 | 8 / 4 |
| 顶点色 | `Vertex_Color` | `[f32;4]` | `Unorm8x4` / `Snorm8x4` | `Unorm8x4` | 4 |
| 骨骼索引 | `Vertex_JointIndex` | `[u32;4]` | `Uint32x4`，索引不得按归一化颜色解释 | 同左；批量 compute 使用每影响 `u32` | 16（四索引） |
| 骨骼权重 | `Vertex_JointWeight` | `[f32;4]` | `Unorm8x4`（归一化） | `Unorm8x4` | 4 |
| Morph 增量 | `MorphTarget` | `[f32;3]` | compute CSR storage：`xyz + target_index` | 同左，独立于顶点属性数 | 16 / 非零贡献 |
| 自定义 | `Vertex_Custom0..N` | 项目定义 | 由注册项声明 | 同左 | 视格式 |

约束：

- 属性数量必须落在 `Limits::max_vertex_attributes` 内（`baseline` 16、`desktop` 32）；顶点缓冲数不得超 `max_vertex_buffers`（8 / 16）。
- 同一语义在整个项目内只有一个格式定义；变体（如法线的两种编码）只能是不同**流类别**，不能是同一语义的两种隐式含义。
- 标准顶点输入只使用 `prism_render_driver::VertexFormat` 的现有变体。`Unorm16x3` 在本文专指 CLST 局部位置编码，不是 RHI 顶点格式；八面体、纹理 Morph 等其他编码必须有独立实现/制品证据，不能从格式名称推断存在。

### 6.4 拓扑

- 运行时主路径：`TriangleList`、`TriangleStrip`、`LineList`、`LineStrip`、`PointList`（与 `prism_render_driver::PrimitiveTopology` 对齐）。
- 授权态可包含 `Quad`/`Polygon`/`Patch`，但必须在构建期三角化后才能进入运行时；运行时不存在四边形或曲面细分拓扑。
- `Topology` 负责图元数计算：`element_count`、是否能非索引化绘制、是否需要三角形内 barycentric 支持。
- **负缩放与手性**：负行列式变换会翻转绕序；绕序翻转在光栅路径与背面剔除中统一处理（`virtual_geometry::software_raster` 已有 winding swap 逻辑），不在数据模型里“修正”顶点顺序。

### 6.5 子网格与材质槽

```rust
pub struct SubMesh {
    pub material_slot: u16,
    pub index_range: core::ops::Range<u32>,
    pub cluster_range: Option<core::ops::Range<u32>>, // 虚拟几何构建后填充
}
```

- 材质槽是几何到材质的**唯一**连接点；一个 Mesh 可有多槽（多材质），但槽数应有上限并计入诊断。
- 虚拟几何路径下，材质槽是 Cluster 划分的**硬约束**：正在被简化合并的三角形不得跨材质槽（§11.1）。
- 同一 Mesh 的不同槽共享同一顶点缓冲，避免重复顶点；索引区间不相交。

### 6.6 索引

- 保留授权宽度（`Indices::Uint16 | Uint32`），理由已写在 `indices.rs`：16 bit 索引的带宽是 32 bit 的一半，“为安全起见加宽”是隐性成本。
- 构建期做 cache-aware 重排（meshoptimizer 形态）以提升后变换缓存命中率；重排必须保持子网格索引区间与绕序语义。
- 16 bit 与 32 bit 可在同一资产的不同 LOD 中混合；`GeometryLodRecord` 用 `primitive_kind`、`first_element`、`base_vertex`、`element_count` 和 `vertex_count` 描述逻辑 draw 区间，具体索引格式由运行时缓冲类元数据携带。
- 非索引化几何（`NonIndexed`）在 `GeometryPrimitiveKind` 中已表达；虚拟几何路径一律索引化。

### 6.7 包围体

一个 Mesh 需要多种包围体，用途不同：

| 包围体 | 计算位置 | 用途 | 现状 |
|---|---|---|---|
| 局部 AABB | `Mesh::bounds()` | 通用裁剪、流送、编辑器框选 | ✅ `MeshBounds` |
| 包围球 | 由 AABB 派生（`MeshBounds::radius`） | 廉价裁剪、Cluster cull、光追宽相 | ✅ 派生 |
| 每 LOD AABB | 构建期 | LOD 级裁剪与阴影页请求 | LOD 记录与构建产物字段 |
| 每 Cluster AABB + 法线锥 | 构建期（`virtual_geometry`） | Cluster cull（视锥 + 背面锥） | cull 与资产校验契约 |
| 有向包围盒（OBB） | 构建期策略项 | 细长物体（剑、杆）的裁剪质量 | profile 开关，未启用时使用 AABB |

规则：非有限位置在计算时跳过而不是污染整个盒（现状 `MeshBounds::from_positions` 已如此）；空网格返回 `None` 而不是全零盒。

### 6.8 校验规则集（完整清单）

渲染前必须通过的校验（现状 `Mesh::validate` 覆盖 1–6、9；Skin/Morph 与构建诊断规则由各自模块或构建阶段负责）：

1. 至少一个属性（否则没有顶点阶段可绑定）。
2. 每个属性的元素数与网格顶点数一致。
3. 每个取值的存储格式与该属性的声明格式一致。
4. 索引（若有）最大值小于顶点数。
5. 索引数量与拓扑匹配（三角形列表为 3 的倍数等）。
6. 子网格索引区间不越界且互不重叠。
7. 蒙皮：骨骼索引小于骨架骨骼数；权重和为有限值（不要求严格为 1，构建期归一化并报告偏差）。
8. Morph：目标数与权重数一致；稀疏增量索引在范围内且严格递增。
9. 顶点数受 `u32` 索引能力约束（> 2^32-1 顶点为内容错误）。
10. 位置中非有限值被隔离（不致命，但计入构建诊断 `non_finite_positions`）；运行时 Mesh 校验拒绝 Float32 与 Float16 的 NaN/Inf 位模式。
11. 属性总量与流划分满足 `Limits`。
12. 退化三角形（零面积、重复索引）在授权态允许存在，但构建期必须报告并可移除；运行时校验不因退化而失败（脏数据不应让整个网格不可上传）。

`MeshError` 现已包含属性格式/数量、索引范围、拓扑数量、子网格范围重叠与 primitive 对齐、以及超过 `u32` ABI 能力的计数错误；构建期诊断与运行期致命错误仍按各自构建模块分级，避免把内容警告当成上传失败。

### 6.9 内容哈希与确定性

- 每个几何产物计算**确定性内容哈希**：相同授权数据 + 相同构建设置 → 相同哈希。哈希覆盖：属性语义与格式、顶点数据（量化前）、索引、拓扑、子网格划分、构建设置、schema 版本。
- 哈希是 DDC/CAS 键（见资产构建文档 §10）；不参与哈希的字段（如源文件路径、时间戳）必须显式排除。
- 浮点比较用位模式（`to_bits`）而非 `==`，保证跨平台一致；量化在哈希之后（哈希授权精度，量化属于构建设置）。
- 确定性纪律：属性遍历顺序固定（现状按 shader location 的 `BTreeMap` 顺序，已是确定性的），量化与简化算法禁止使用会随线程数变化的归约顺序。

## 7. 顶点流布局与量化策略

### 7.1 单流 vs 分流

`Mesh::vertex_layout()` 保留单流、交错、level 0 兼容布局；`stream_layouts()` 与 `RenderMesh` 提供确定性的多流布局和实际字节打包。本文统一规定两者的选择策略：

| 流 | 内容 | 访问模式 | 是否必需 |
|---|---|---|---|
| Stream 0 `position` | 位置（+ 可选 Cluster ID） | 每个 pass（深度、阴影、速度、可见性）都读 | 必需 |
| Stream 1 `shading` | 法线/切线/UV/顶点色/自定义 | 仅着色 pass 读 | 视材质 |
| Stream 2 `skinning` | 骨骼索引 + 权重 | 仅形变 pass 读 | 蒙皮几何 |
| Stream 3 `morph` | Morph 增量（稠密模式） | 仅形变 pass 读 | 有 Morph 的几何 |

设计原则：

- **深度/阴影/可见性 pass 只绑定 Stream 0**（虚拟几何另有 Cluster 自有缓冲），这是分流的最大收益来源。
- 分流数量不得超过 `Limits::max_vertex_buffers`；当属性较少时，允许“位置 + 着色”两流或回退到单流交错（策略由构建期 profile 决定，运行时不动态重组）。
- 分流结果写入构建产物的显式 schema；运行时不得隐式假设交错。
- `Mesh::vertex_layout()` 保持单流语义不变（向后兼容），分流布局由 `streams.rs` 生成 `Vec<VertexStreamLayout>`，每项包含 `VertexStreamKind`、stride 与属性偏移。

### 7.2 属性 → 出货格式（策略表）

量化策略按几何变体分级，核心原则：**误差必须在屏幕空间可解释，且可被构建期报告**。

| 变体 | 位置 | 法线/切线 | UV | 备注 |
|---|---|---|---|---|
| 静态标准网格 | `Float32x3` | `Snorm8x4` / 八面体 | `Float16x2` | 兼容性优先，量化可选 |
| 静态虚拟几何 CLST v3 | Cluster 空间 `Unorm16x3`（局部量化） | 保留授权格式和值 | 保留授权格式和值 | 直接 Cluster 构建保留全部 12 种 RHI 属性格式、语义与 shader location；解码兼容 v2 |
| 蒙皮网格 | `Float32x3` | `Snorm8x4` | `Float16x2` | 位置不量化：形变需要精度且每帧被 GPU 写 |
| Morph 顶点 | `Float32x3`（目标空间） | 由法线重建 | 同基网格 | 增量幅度小，可选 `Float16x3` |
| 地形 | clipmap/heightfield 专用 | 由高度差分 | 无 | 不走通用网格量化 |
| 毛发/草 | 程序化生成 | 程序化 | 程序化 | 不落静态页 |

说明：`Unorm16x3`、`Float16x3` 等编码不属于 `prism_render_driver::VertexFormat` 顶点输入；CLST v3 以 Cluster AABB 对位置执行 `u16` 归一化编码，其他属性按 Cluster 局部顶点映射无损采集，不做隐式格式转换。标准网格和 CLST v3 使用 RHI 当前的 `Float32/x2/x3/x4`、`Float16x2/4`、`Uint32/x2/x4`、`Sint32`、`Unorm8x4`、`Snorm8x4`。`Float16x2/x4` 的实际 stride 为 4/8 字节；CPU 布局及 Metal 实际绘制共同锁定该值。

#### 7.2.1 CLST 版本化字节布局

- v2 的 `encode_cluster_payload` 继续输出 position-only 兼容字节；`encode_cluster_payload_with_attributes` 与直接 Cluster 构建输出 v3，统一 decoder 接受两者。
- 两版几何布局相同：34 字节固定头（magic/version/counts/min/max），每顶点 6 字节位置，每索引 2 字节。v3 在几何末尾补零到 4 字节对齐，再写 `position_location:u32 / attribute_count:u16 / reserved:u16`。
- 每条属性按 shader location 严格升序写入 `name_len:u16 / format_tag:u16 / location:u32 / data_len:u32`、UTF-8 语义名、4 字节对齐补零、紧凑 LE 顶点数据。总属性数含位置最多 32，语义名 1–255 字节，禁止控制字符、重复语义或重复 location。
- 持久化格式标签独立于 Rust 枚举：`Float32/x2/x3/x4=0/1/2/3`，`Float16x2/x4=4/5`，`Uint32/x2/x4=6/7/8`，`Sint32=9`，`Unorm8x4=10`，`Snorm8x4=11`。未知标签、NaN/Inf（含 half）、错误长度、非零 padding/reserved、乱序和尾随字节均返回类型化错误。
- 分簇预算包含属性头、语义名与全部属性值，追加三角形前以 O(1) 计算实际大小；单三角形仍超页时拒绝构建。设置哈希绑定 CLST schema 与 domain tag，v2 旧缓存不会命中 v3 构建。

### 7.3 量化与误差界

- **位置量化**基于 Cluster/页的局部包围盒，把位置映射到归一化整数；解码用页头部的 `scale/offset`。误差上界 = `局部extent / (2^bits - 1)`，必须并入该 Cluster 的 `object_error`，使 LOD 选择能感知量化误差。
- **法线/切线/UV**由标准网格分流布局和 CLST v3 属性表按授权格式存储；直接分簇保留原值，不重采样。
- **UV** 当前紧凑格式是 `Float16x2`，不执行基于 UV 包围盒的隐式归一化；Cluster 局部映射保留接缝两侧的独立顶点。
- **极端位置**量化与 CPU 重建使用 f64 中间运算，误差上界包含重建舍入；若平方误差不能表示为 f32，报告正无穷而不是零或伪造可通过预算的有限值。
- 量化是构建期设置的一部分；改变量化必须改变 `build_settings_hash` 并触发重建，禁止运行时“就地量化”。

### 7.4 缓冲对齐与 GPU 限制

- 每个顶点流偏移必须满足设备要求的对齐（通常 4 字节，某些格式要求更高）；构建期按 `min(alignment, 设备能力)` 对齐。
- 索引缓冲偏移对齐 2（u16）/ 4（u32）字节。
- 虚拟几何页的大小是固定值（建议 64 KiB 或 128 KiB，见 §11.2），必须与 `prism_virtual_geometry_gpu` 的页 ABI 对齐。
- 大网格（> 2^16 顶点）必须使用 32 bit 索引；构建期自动选择索引宽度并在 `GeometryLodRecord` 中显式记录。
- 规范禁止依赖 `base_vertex` 隐式合并多个网格到同一缓冲（会破坏按资产回收）；`base_vertex` 只用于同一资产内的子网格偏移。

## 8. 几何变体

| 变体 | 拓扑 | 形变 | 路径 | 特殊约束 |
|---|---|---|---|---|
| 静态网格 | 固定 | 无 | 标准 draw 或虚拟几何 | 虚拟几何要求拓扑构建期固定 |
| 实例化网格 | 固定 | 无 | GPU-driven 实例 + 共享几何 | 几何共享，仅实例 transform 变化 |
| 蒙皮网格 | 固定 | 每帧 GPU skinning | 标准路径 + 形变缓存 | 骨骼影响数有档位上限 |
| Morph 网格 | 固定 | Morph 权重混合 | 与蒙皮复合 | 大量目标走 delta 纹理 |
| 布料 | 固定（拓扑不变） | 每帧解算 | 布料解算 → 形变缓存 | 与蒙皮按 §9.3 固定顺序叠加 |
| 顶点动画（VAT） | 固定 | 采样动画纹理 | GPU 采样 | 内存小，精度受纹理格式限制 |
| 程序化 / 粒子 / 水 | 程序化 | 每帧生成 | Compute/Mesh Shader | 不进入静态几何页 |
| 地形 | clipmap | LOD 网格重建 | 专用路径 | 与虚拟几何的二选一由编辑需求决定 |
| 毛发/草 | Guide/Strand/Card | 插值 + 物理 | 程序化 | 不掉进静态页 |
| 变形（破面/破碎） | 构建期固定 | 可选 | 虚拟几何兼容（拓扑不变） | 破碎产物在构建期生成 |

自动路径选择器（构建期 + 运行期两段）：构建期决定“这个资产是否生成虚拟几何产物”；运行期决定“这个实例走哪条光栅路径”。判据是三角形数、屏幕占比、是否形变、设备能力与项目策略，**不是**“所有 Mesh 一刀切”。

## 9. 形变管线

### 9.1 蒙皮

- 批量入口：设备无关 `DeformationMesh::new`，原生 GPU `DeformationCache::submit_pose`。profile 显式选择 4/8 影响；超限报错，不静默截断。数学库的 `prism_math_gpu::skinning` 是另一个算法 twin 入口，不冒充持久运行时缓存。
- 内核为线性混合蒙皮（LBS），不把双四元数的数学 twin 等同于当前批量算法。位置使用三行仿射矩阵；法线始终使用对应逆转置，支持非均匀缩放与反射，奇异/非有限矩阵在提交前拒绝。
- 权重在静态准备时归一化并去掉零项；无影响或和为零的顶点保留 **Morph 后的绑定姿态**。带符号 Morph 权重有效，蒙皮权重必须非负。
- 有限值仍可能在乘加中溢出：静态准备记录位置/法线/逐目标的绝对包络，动态验证按目标、骨骼和修正计算保守范围。超过 `f32::MAX / 4` 的中间包络返回 `ArithmeticRange`，不钳制、不污染上一帧缓冲；该策略可保守拒绝最终会相消的极值姿态。

### 9.2 Morph target / blend shape

- Dense/Sparse 授权目标在准备时转置为逐顶点 CSR storage；零增量不存储，不分配 `顶点数 × 目标数` 的稠密表，也不占 shader location。
- 每项贡献为 `xyz` 位移 + `u32 target_index`（16 字节），每帧只上传目标权重；每个顶点遍历自己的非零贡献。
- 当前 `MorphDelta` 只表达位置增量。法线由基法线逆转置蒙皮后归一化，可叠加显式外部法线修正；该接口不隐含自动重算法线或 Morph 法线导数。

### 9.3 复合顺序

固定顺序，跨 pass 一致：

```text
bind pose → morph targets（按权重累加）→ skinning（骨骼变换）→ 布料/物理修正（若有）→ 输出
```

顺序必须写进着色/形变契约文档；任何变体（如“先蒙皮后 morph”）必须显式声明，不得隐式。

### 9.4 形变缓存与 motion vector

- `DeformationCache` 以拥有设备的 `Arc` 保存全部资源，满足 `Send + Sync + 'static`；静态输入只上传一次，每个非空姿态一次 compute dispatch，无生产读回，Drop 显式释放资源。创建前检查 storage/uniform/bind-group/dispatch 限制。
- 两个输出均为 `STORAGE | VERTEX | COPY_SRC`，每顶点 32 字节，位置/法线位于 offset 0/16；`DeformationOutput::vertex_layout` 返回实际布局，不能误用标准 split-position 的 12 字节步长。
- frame 必须严格递增；首帧、跳帧或 `invalidate_history` 使 `previous` 别名 `current` 且 `history_valid=false`，连续有效帧才承接历史。非法输入在任何 queue write 前拒绝，停用外部修正会显式清零。
- 输出为 mesh-local：速度消费者必须分别乘当前/上一帧实例矩阵；历史无效时不计算形变速度。消费者须在同一有序 queue 的下一次更新前提交，缓存只保留两个姿态，不保存任意多帧。
- 16×16 真实 GPU 像素测试已证明两个输出可直接作为顶点流参与位移/法线/速度着色，不经 CPU 重打包；完整引擎的主视图、深度、阴影及 RT refit 接线按各消费者单独核验。
- `deformation::{DeformationBudget, DeformationKind}` 是预算与分类入口；每秒形变顶点数、BLAS refit 次数是硬预算。
- 形变缓存的分配与普通顶点缓冲分属不同缓冲类（见 §12），避免与静态几何争夺分配器。

### 9.5 Deformable Cluster（Nanite skinning 形态）

`Ultra` Deformable Cluster profile：让蒙皮高模也能走虚拟几何，做法是把骨骼影响“烘焙进 Cluster”：

- Cluster 携带影响它的骨骼集合（通常 ≤ 64 骨骼/Cluster）与原位姿态。
- 运行时用 Compute 逐 Cluster 重建顶点位置（局部），再走同一套 Cluster cull 与光栅路径。
- 代价是每帧几何计算与内存上升；策略只对单角色、次要与近景层开启，LOD 以“变形误差”而非仅几何误差驱动。

## 10. LOD 系统

Prism v1 有两条互补的 LOD 机制，二者共享误差语义：

- **离散 LOD 链**（标准路径、蒙皮、程序化几何）：构建期生成 LOD0..N，运行时按屏幕空间误差选择。
- **Cluster 层级 cut**（已序列化层级资产）：层级节点上逐区域选择，允许同一物体不同区域处在不同细化级别；构建侧的固定 Cluster 资产把每个分区作为独立 root，独立 QEM LOD 资产集合提供跨分辨率的整体替换。

### 10.1 屏幕空间误差

统一公式（已在渲染架构 §6.3 提出，此处作为几何侧契约固化）：

```text
projected_error = object_error * projection_scale / max(view_depth, near_epsilon)
refine if projected_error > threshold_pixels
```

- `object_error` 对离散 LOD 来自简化误差，对层级 Cluster 来自节点误差（父节点误差不得小于子节点）；固定 Cluster root 的误差为零，因为它只表示原始分区，不冒充简化代理。
- 阈值以**像素**为单位可配置，并与抗锯齿/上采样配合（TAA/TSR 下可用略高阈值）。
- `virtual_geometry::lod` 已实现滞回与速度缩放预取；离散 LOD 链复用相同参数语义。

### 10.2 反跳变

- **滞回**：进入与退出细化级别的阈值不同，避免阈值抖动。
- **Dither 过渡**：LOD 切换时对相邻两级做屏幕空间抖动混合，消除“突然变细/变粗”。
- **速度偏置**：高速相机预取更细一级，避免运动中出现欠细化。
- **时间预算**：每帧限制 cut 选择与新页请求量，保证帧时稳定。

### 10.3 多视图与次要视图

每个视图（主视图、阴影、反射、光追、编辑器缩略图）有自己的误差权重与预算：

- 阴影视图通常用更低精度（阴影误差在接收面上被隐藏），但**不得**与主视图共用错误的空间映射。
- 光追视图的 LOD 由光线微分与初级命中距离决定（§15）。
- 编辑器缩略图用固定低 LOD 并缓存，不参与运行时预算。

### 10.4 HLOD 与代理

- 远景用 HLOD 代理（合并网格 + 合并材质），由场景/世界系统驱动（见场景文档 §24.11）。
- HLOD 的几何仍是标准网格产物，走虚拟几何或标准路径由构建期决定。
- HLOD 切换点由屏幕空间误差与像素预算共同决定，不允许仅按固定距离。

### 10.5 回退与降级

- 请求的 LOD 页未驻留时：渲染最近已驻留的**祖先**，禁止出现洞（`virtual_geometry::page_table` 与 `geometry::GeometryRecord::resolve_lod` 已体现该语义）。
- 离散 LOD 链缺失时：回退到 fallback mesh（构建期必产），再回退到 AABB 占位。
- 任何降级必须可诊断（§19）。

## 11. 虚拟几何：从 Meshlet 到自动几何系统

### 11.1 离线构建

导入即自动生成虚拟几何产物，无需美术手工做离散 LOD（与资产构建文档 §12.2 对齐）：

```text
含属性 Mesh        → 完整校验（索引、拓扑、有限值、SubMesh 覆盖）
                   → 按材质 SubMesh 独立分区，解析属性策略与拆分缝
                   → 从前一层执行 endpoint-QEM，保留原顶点及属性
                   → 每个 LOD 独立 Cluster 化（≤128 三角形 / ≤64 顶点）
                   → Cluster AABB 内 `u16x3` 位置量化 + 局部索引 + CLST v3 属性
                   → 按确定性 Cluster 顺序打包固定大小页
                   → BVH 采样误差、量化误差包络、版本化身份与交叉校验
```

硬约束：

- **材质槽不可跨**：简化不得合并不同材质槽的三角形。
- **属性不静默丢失**：分簇和 LOD 均保留全部 12 种 typed 属性、语义与 shader location；`VertexAttributeValues::gather` 按 source remap 精确保留位模式、重复及顺序。endpoint-QEM 不生成插值顶点，因此没有未定义的自定义整数/分类属性插值。
- **属性策略**：浮点、half、normalized 属性默认连续代价，整数默认分类约束；`StaticLodPolicy` 可逐语义覆盖为 `Continuous { weight }`、`Categorical` 或 `PreserveOnly`。连续代价是候选排序启发，不是属性误差上界。未知/重复语义、位置覆盖、负值/非有限权重返回类型化错误；tangent handedness 始终作为分类约束。
- **拓扑与缝**：固定开放边、非流形顶点、相同位置的拆分顶点（含正负零同位置），执行 link condition、重复面与翻面检查。不能达到目标时保留拓扑并如实记录 `achieved_triangles`，不把请求值伪装为实际值；退化面清理后无表面则报错。
- **确定性与缓存身份**：同一输入、设置、策略与非递增目标序列生成相同资产身份、页顺序与编码字节。源内容、源资产 ID、构建设置、完整目标序列、解析后的属性策略与 `STATIC_LOD_ALGORITHM_VERSION=2` 进入配方哈希；层级派生页身份发生截断碰撞即报错。
- **误差定义**：`measured_vertex_error` 是原始授权顶点到该层未量化三角面的最大单向采样距离，使用确定性 f64 最近三角形 BVH 求值并向上舍入；不是连续表面或双向 Hausdorff 上界。`vertex_error` 将该值与该层最大 Cluster 位置量化位移相加，再取非递减前缀包络供 LOD 选择。
- **量化与元数据**：固定分区叶节点的 `self_error` 使用共享 `cluster_position_error`，不再写零；量化误差包括半步长与 f32 重建舍入。无法表达的平方误差保留为正无穷，不能当成零预算；距离使用 f64 直接求根。LOD 集合对身份、实际计数、有限性和单调性严格校验，不通过 clamp 掩盖坏元数据。
- **性能结构**：QEM 使用局部邻接、双向候选、版本化堆和 one-ring 重评分；误差 BVH 使用中位分割、近节点优先、AABB 剪枝与可复用遍历栈，避免对每个源顶点扫描全部目标三角形。
- **GPU twin 对齐**：Cluster 布局、页大小、量化编码必须与 `prism_virtual_geometry_gpu` 的 ABI 一致。

### 11.2 资产格式

`pkg/prism_render_architecture::virtual_geometry::VirtualGeometryAsset` 是可分页资产的 v1 编解码契约（渲染架构 §6.2 的字段落地）：

```text
VirtualGeometryAsset
├─ Header
│  ├─ magic / schema_version / build_settings_hash / content_hash
│  ├─ bounds / material_slot_count
│  └─ root_page_ids / fallback_mesh_ref
├─ HierarchyPages       层级节点、误差、包围体、法线锥、子页引用
├─ GeometryPages
│  ├─ 量化位置（Cluster 局部）
│  ├─ 局部三角形索引（`u16`）
│  ├─ CLST v3 属性语义、格式、shader location 与局部顶点值
│  ├─ Cluster 描述符（顶点/三角形计数、payload 范围、材质槽）
│  └─ Cluster 包围体 + 法线锥
├─ DependencyTable      页之间的父子/依赖关系（用于预取）
└─ StreamingMetadata    页大小、页计数、优先级与 locality 提示
```

- 生产档页大小固定为 64 KiB，元数据按 1024 字节粒度对齐；格式验证同时接受不小于 1 KiB 的二次幂页大小，以便测试和受限设备复用同一 ABI。`VirtualGeometryAsset` 与 GPU twin 共享页大小契约。
- 页存储可压缩（zstd/GDeflate 形态），运行时经 `prism_gdeflate_gpu` 直通解压或 CPU 解压。
- 必须使用显式 `schema_version` 与 `build_settings_hash`；旧缓存可检测并重建，禁止静默错读。

当前代码已对 header、层级节点、页排序/大小、cluster 范围、非空 cluster、cluster/node page identity、层级/Cluster AABB 包含关系、材质槽、payload 引用和确定性编解码做校验；CLST v2/v3 codec 由运行时架构 crate 统一持有，构建器通过兼容导出复用。页面进入 `VirtualGeometryPageStream` 前校验 magic/version、精确 framing、bounds、索引、属性表及 descriptor 计数，错误不会污染 resident storage。直接构建的属性保留、endpoint-QEM 实际简化及精确属性 remap、量化误差与压缩页逐字节一致性分别有正反测试；页字节一致性不能外推为 shader 属性解码或最终材质着色已验证。

### 11.3 运行时 cut 与剔除

已有决策层：`virtual_geometry::{hierarchy, cull, bins, pipeline, frame}`。契约如下：

```text
每视图：
  cut 选择（屏幕空间误差 + 滞回 + 速度预取）→ 可见 Cluster 集合
      → 视锥剔除（Cluster AABB）
      → 背面锥剔除（法线锥）
      → 遮挡剔除（HiZ / 两级）
      → 分箱（bins，供间接光栅参数）
      → 逐 Cluster 光栅路径分类（raster_path）
```

- 2026-09-22 基线中，“标准 Mesh Visibility Raster 已接入独立 `Visibility3d` binned phase”，消费 CPU reference visible work；虚拟几何通过同一份 `GeometryFramePlan` 输出 cut、raster bins 与 page requests，后端以此生成 indirect stream（渲染架构 §17.1）。
- Cluster cull 必须无界安全：任意大的 cut 输入不得导致 CPU 内存爆炸；用固定容量批次 + 溢出回退。

### 11.4 两条光栅后端

高层算法不绑定 Mesh Shader：

| 后端 | 适用 | 说明 |
|---|---|---|
| HW Mesh Shader | `Ultra`、支持 Task/Mesh Shader 且基准更优 | Task 做 Cluster cull/压缩，Mesh 输出微三角形 |
| Compute + 软件光栅 | `Standard`、Metal/移动/无 Mesh Shader 平台 | Compute 生成可见 Cluster 与间接参数；微三角形软件光栅写 Visibility Buffer |

两条后端必须产出**相同语义**的 Visibility Buffer（相同编码、相同深度约定、相同 winding 处理），由 GPU twin 逐元/逐像素校验覆盖规则、绕序、top-left 填充与 reversed-Z 编码（`prism_virtual_geometry_gpu` 已具备该范式）。

### 11.5 Visibility Buffer 与材质解析

```text
visibility.x = scene_instance_index
visibility.y = cluster_or_primitive_id
depth        = hardware depth（reversed-Z）
barycentrics = 可重建 / 原生输出 / 额外紧凑目标
```

优点：几何可见性与材质复杂度解耦；大量材质不增加几何 draw；PBR/NPR 共享可见性；可做材质分类与 tile dispatch。

约束：

- visibility 采用 single-sample；MSAA 视图必须显式撤销 phase item，不做有损 MSAA resolve（渲染架构 §17.1 已记录该决定）。
- 透明、折射、部分 alpha-tested 材质保持独立 forward/fragment 路径。
- 空白像素用全一 invalid sentinel，避免被解析为合法索引 0。

### 11.6 流送与驻留

- 页请求由可见性 + 预取共同驱动；`virtual_geometry::page_request` 做跨 Cluster 的按帧合并。
- 页表（`page_table`）负责驻留与预算驱逐；预算来自统一虚拟资源系统（渲染架构 §8）。
- 优先级：主视图近景 > 主视图远景 > 阴影 > 反射 > 预取；相机速度提升预取层级。
- 防抖动：页到达后延迟生效（避免 pop），或以 dither 过渡；缺页时渲染祖先。
- I/O 路径：资产容器（pak/IoStore 形态）→ 分块读 → GPU 直通解压（可选）→ 上传；与资产文档 §11 的流送管理器共用预算与反馈。

### 11.7 与标准 mesh 的共存与回退

- `Compatible` 档没有虚拟几何时，构建期仍可产出离散 LOD 链；运行时选择器不因路径不同而改变场景语义。
- 同一场景内允许虚拟几何与标准网格混合（角色/毛发/粒子/地形走标准或程序化路径）。
- 回退链：虚拟几何页 → fallback mesh → 低 LOD 离散网格 → AABB 占位。每一级都必须可见且可诊断。

## 12. GPU 表示、驻留与上传

几何数据在 GPU 侧分四类缓冲，回收策略不同：

| 缓冲类 | 内容 | 生命周期 | 回收 |
|---|---|---|---|
| `Static` | 标准网格的顶点/索引、Cluster 页 | 常驻（除非驱逐） | 引用计数 + 预算驱逐 |
| `Streaming` | 虚拟几何页 | 会话内按需驻留 | 页表驱逐、代际延迟回收 |
| `Dynamic` | 形变输出、程序化几何 | 每帧或会话内反复写 | 环形/双缓冲复用 |
| `Readback` | GPU 可见性统计、页请求 | 短时 | 帧后回收 |

设计要点：

- **子分配**：页与运行时网格按固定粒度从 slab 分配，避免每资产一个大 buffer 造成的碎片与对齐浪费；分配器必须能报告碎片率。
- **上传不阻塞帧**：异步拷贝队列 + 环形 staging，上传预算按帧分配；大资产分帧上传。
- **首帧墙优先**：加载时先上传“首帧必需集”（fallback + 近景 LOD），远景页随流送补齐。
- **预算与超配**：显存超配时按优先级驱逐并在诊断中记录“因预算未驻留”；禁止 OOM。
- **设备丢失恢复**：几何缓冲可在设备恢复后重建；静态几何可从 CPU 侧或磁盘制品重新上传，`prism_render_scene` 已有设备恢复路径可复用。
- **对齐**：页与缓冲对齐必须满足 RHI 要求，并对齐到 GPU twin 假设的固定粒度。

## 13. 可见性、提交与 GPU Scene 集成

- 实例身份、变换、材质引用归 GPU Scene（渲染架构 §5）；几何只消费 `GeometryHandle` 与 GPU Scene 索引。
- 标准路径：CPU 侧可见性只提交增量；GPU-driven 实例剔除与间接 draw 归 `prism_render_gpudriven`。
- 虚拟几何路径：Cluster cull 在 GPU（或 CPU 参考实现用于校验），产出间接参数供两后端消费。
- 两条路径必须共享同一套“可见 work”语义（渲染架构 §17.1：`Visibility3d` binned phase 消费可见 work），避免各自维护剔除结果。
- 遮挡：两级 HiZ（上一帧最小/最大深度）用于保守剔除；虚拟几何与标准网格共用同一 HZB 生成。
- 提交侧禁止每 Cluster/每实例一次 CPU draw call；一律间接。

### 13.1 统一 Mesh consumer contract

Mesh 表示不能由主视图、深度、阴影、速度、光追和派生几何各自重新选择。统一入口已经落在两个层次：

| 层 | 实现入口 | 责任 |
|---|---|---|
| 设备无关选择 | `prism_render_architecture::geometry::consumer` | 校验 `GeometryHandle`、revision、current/previous history、页缺失 fallback 与派生 artifact schema；输出 `MeshConsumerSelection` |
| RHI 绑定 | `prism_render_app::mesh::consumer` | 将 `RenderMesh`、`DeformationOutput`、`GeometryPageKey` 解析为静态、形变或虚拟 Cluster GPU source，并保持 current/previous/fallback 的身份一致 |

规则如下：

- `MainView`、`Depth`、`Shadow`、`RayTracing` 共享同一个 current representation；不能在某个 pass 中私自换成另一份 Mesh。
- `MotionVector` 始终返回 current/previous 成对表示；首帧、跳帧或 teleport 时 previous 等同 current 且 `history_valid=false`，由速度 pass 输出零速度。
- 虚拟几何页未驻留时，渲染消费者只绑定 `fallback`，并将 `used_fallback=true` 传递给诊断；不得返回空 draw 或未校验页。
- `Collision` 与 `DerivedGeometry` 不直接消费虚拟 Cluster；它们必须消费 `MeshDerivedArtifact { key, kind, schema }`，用制品身份和 schema 防止碰撞、SDF、导航、OMM 等派生数据错配。
- 新增消费者只能扩展 `MeshConsumer` 与对应选择测试；GPU 后端不得复制一套独立的 Mesh 选择逻辑。

### 13.2 Bevy 场景适配与 LOD 资产生命周期（已实现）

`prism_render_scene` 已把上述设备无关契约接入 Bevy `RenderApp`。`PrismGpuSceneLodSet`
是实体级的显式 LOD 输入：实体的 `Mesh3d` 是 level 0，组件中的 `PrismGpuSceneLod`
按声明顺序成为 level 1 及之后的独立准备资产，并携带 `screen_error`。提取系统把这些
资产复制到 render world，GPU Scene 只保存 `GeometryHandle { index, generation }` 到
base asset/LOD asset 的稳定映射；它不把 Mesh 顶点数据复制进场景表。

运行时同步由 `RenderGeometryRegistry` 完成：每个驻留 LOD 都有自己的真实 allocator
draw range、vertex/index buffer class 和 prepared asset 映射。资产重载会保留仍有效的
LOD 映射并刷新对应范围；LOD asset 被 `AssetEvent::Unused` 或 `AssetEvent::Removed`
回收时，只移除该 LOD，不破坏 base geometry record；base asset 回收则连同整条 LOD
链一起移除。句柄与资产的双向绑定遇到冲突会拒绝写入，调用侧跳过本次实例更新并增加
`GpuSceneDiagnostics::geometry_binding_failures`，避免旧实例悬挂到错误几何。

CPU visibility 选出的 resident LOD 会携带精确 prepared asset 进入 Opaque direct/indirect、
Visibility Raster 与 Transparent 阶段；Shadow depth 和 VSM caster 使用相同的
`Shadow` consumer contract，并以 level 0 作为稳定请求入口，由 `GeometryRecord::resolve_lod`
解析到实际驻留 fallback。所有 draw 在取 allocator slice 前都重新校验 buffer class、
vertex/index range、`base_vertex` 与 `vertex_count`，越界或溢出直接跳过 draw，保证失败
是 fail-closed 而不是提交非法 GPU 参数。

对应入口与测试位于：

- `pkg/prism_render_scene/src/extract/components.rs`、`extract/systems.rs`、`extract/lifecycle.rs`；
- `pkg/prism_render_scene/src/scene.rs`、`geometry/runtime.rs`、`geometry/systems.rs`；
- `pkg/prism_render_scene/src/geometry/consumer.rs` 与各 draw/queue consumer；
- `pkg/prism_render_scene/src/geometry/draw_range.rs` 及 geometry/scene/lifecycle 单测。

Shadow depth 与 VSM caster 已纳入同一 Mesh consumer contract：两条路径都从稳定的
`GeometryHandle`/LOD 选择取得 prepared asset，并在 allocator slice、buffer class、索引/顶点
范围和 `base_vertex` 进入 GPU draw 前执行 fail-closed 校验。VSM 的页表读回、同帧 bridge、物理
atlas caster-depth 与 shading resolve 绑定已经由 `prism_render_scene::shading::plugin` 串成同一
条执行链；coarsest clip-window depth slab 是当前稳定契约，用于避免每帧 caster 扫描和页驻留
变化导致投影 ABI 失配。

这部分完成性结论由源代码、CPU/编译测试和已有 GPU twin 证据共同组成；完整场景 Golden 图、
跨设备画质对拍和性能预算仍按 §18 的固定采集协议单独判定，不用单元测试或设计预算替代真实
捕获物。

## 14. 材质 / 着色 ABI 与 PSO 复用

几何与材质的相交面只有一个：**顶点布局 + 属性语义 + 图元拓扑**。为保证 PSO 可缓存且不爆炸：

- PSO 键至少包含：顶点布局哈希（属性格式/偏移/分流）、着色模型 ID、渲染状态（深度/混合/剔除/绕序）、目标格式、MSAA、可用特性集。
- 顶点布局哈希必须与 `Mesh::vertex_layout()` 的确定性输出一致（同属性集 → 同字节布局，现状已保证交替共享）。
- 材质域可声明“需要哪些属性”；几何只提供材质声明的属性，未声明的属性不污染布局。
- 虚拟几何路径的材质解析发生在 visibility 之后，因此 PSO 键不含“每个 Cluster 的几何变体”，只含材质域与可见性编码。
- 顶点属性上限、流上限由 `Limits` 决定；构建期若超限必须报错，不得运行时静默丢弃属性。

## 15. 光追与物理派生几何

### 15.1 光追几何

- BLAS 从 **LOD0/最高精度** 三角形构建；透明与 alpha-tested 几何另用 OMM（`prism_micromap` + `prism_micromap_gpu`）剔除不需要 any-hit 的微三角形。
- 形变几何（蒙皮/布料）每帧 refit BLAS 或重建动态 BLAS；refit 次数受 `DeformationBudget::blas_refits_per_frame` 约束。
- 光追 LOD 由光线微分决定：近距离全精度、远距离低 LOD 或 SDF 代理；切换不得产生轮廓跳变（与主视图 LOD 一致性检查）。
- 虚拟几何与光追共享几何身份；页驻留影响光追可用精度，缺页时使用 fallback（构建期保证 fallback 覆盖全轮廓）。
- 支持硬件 Ray Query 的平台优先硬件路径；否则回退软件 BVH（渲染架构 §10）。

### 15.2 物理与派生几何

物理/导航/阴影/光追是**独立产物**，可以有不同的精度与拓扑：

| 产物 | 来源 | 精度 | 说明 |
|---|---|---|---|
| 碰撞网格 | Mesh IR → 凸分解/简化 | 低（凸包/简化网格） | 不共享渲染顶点缓冲 |
| 导航网格 | 场景 + 静态几何 | 低 | 世界级，不是单资产 |
| 阴影几何 | LOD 子集 | 低 | 复用渲染几何，不另存 |
| 光追几何 | LOD0 + OMM | 高 | 可与虚拟几何页共享 |
| SDF | 构建期体素化 | 中 | 用于软件追踪与 GI 代理 |

规则：所有派生几何在 `prism_asset_build_core` 的 Artifact DAG 中显式表达依赖（资产构建文档 §12.2），不以“渲染时临时转换”替代构建。

运行时绑定由 `MeshDerivedArtifact` 传递制品 key、kind 和 schema。绑定校验失败时返回结构化错误，不允许把当前渲染 LOD、虚拟 Cluster 或过期 revision 当作碰撞/导航/SDF/OMM 输入；这保证派生消费者与渲染消费者共享几何 revision，但保持各自产物的精度和生命周期。

## 16. 压缩与带宽

- **页压缩**：虚拟几何页存储时压缩（zstd/GDeflate 形态）；解压可 CPU 或 GPU 直通（`prism_gdeflate_gpu` 的既有范式：并行去交错在 GPU，串行 inflate 在 CPU，或反之，以实测为准）。
- **顶点/索引编码**：cache-aware 索引重排、顶点差分编码、属性量化（§7.2）三管齐下；目标是显著降低索引带宽与后变换缓存缺失。
- **带宽预算**：每档位给定“每帧几何读取字节数 / 三角形”的上界；超限必须触发 LOD 收紧或页预算下调，而不是掉帧。
- **磁盘与 PCIe**：大页压缩降低 I/O；直通解压避免 CPU 往返；容器分块（资产文档 §10.5）与流送请求粒度对齐。
- **确定性**：压缩必须无损可逆（相对其输入）；量化误差在解压前已计入 `object_error`，不因压缩引入新误差。

## 17. 资产构建与烘焙集成

几何构建不是“一个转换函数”，而是资产构建 DAG 中的一组 Processor（资产构建文档 §6、§12.2）：

```text
Source(FBX/USD/glTF) → Scene Source IR → Mesh IR
                                        ├─ Render Mesh（标准 LOD 链）
                                        ├─ Meshlets → VG Pages（虚拟几何）
                                        ├─ Collision Mesh（碰撞/导航）
                                        ├─ RT Geometry（BLAS 输入 + OMM 覆盖）
                                        ├─ Preview Mesh（编辑器/缩略图）
                                        └─ SDF（可选）
```

契约：

- 每个产物有 `ArtifactDescriptor`（schema 版本、build_settings_hash、content_hash、依赖集），可独立缓存与失效。
- Processor 声明 `estimated_peak_memory`；大型几何必须流式/分块处理，避免多节点同时形成内存峰值。
- 增量：改材质只重建依赖材质槽的子产物；改几何则整链重建；源码哈希用于精确失效。
- 验证与门禁：结构化诊断（错误/警告/信息）、预算聚合、Golden 指标（包围盒、轮廓、Hausdorff/屏幕误差、材质槽约束、蒙皮结果）。
- 失败处理：构建失败不产出半成品（原子写入）；负缓存避免反复重试已知坏输入。
- 依赖外部工具时（简化、凸分解）通过外部进程/远程 worker 运行，保持确定性与可复现（§18）。

与编辑器的交界：

- Import Inspector 暴露 Mesh 相关设置（法线/切线生成、LOD 策略、量化、Cluster 上限、碰撞简化），默认值必须“开箱可用”。
- 一键重导入不丢用户覆盖；设置哈希变化触发精确失效。
- 编辑器预览使用同一运行时网格格式（低 LOD），避免“编辑器里对、运行时不对”。

## 18. 性能预算与基准即规格

以下数字是产品级回归预算，不是当前设备已经达到的性能结论。一次完整场景性能样本必须同时记录设备标识、后端、驱动、分辨率、质量档、场景种子、冷启动/热启动状态和 P50/P95/P99，并写入 `target/benchmarks/mesh/<date>/`。固定场景为：`100 万 Cluster 静态场景`、`2000 万三角形近景高模`、`单角色 5 万顶点蒙皮`、`1 万实例共享几何`。

| 指标 | 目标（`Standard`，1080p） | 目标（`Ultra`，4K） | 说明 |
|---|---|---|---|
| CPU 几何提交 | ≤ 1.5 ms | ≤ 2.5 ms | 与可见实例/Cluster 数相关，与场景总量无关 |
| Cluster 剔除吞吐 | ≥ 50M Cluster/s（GPU） | ≥ 150M Cluster/s（GPU） | 两级遮挡下的保守剔除 |
| 静态近景几何 GPU | ≤ 8 ms | ≤ 12 ms | 含 cull + 光栅 + 材质解析 |
| 蒙皮 5 万顶点 | ≤ 0.6 ms | ≤ 1.0 ms | 形变一次、多 pass 复用 |
| 几何显存驻留 | ≤ 1.5 GB | ≤ 4 GB | 超限按优先级驱逐 |
| 页上传带宽 | 峰值 ≤ 300 MB/s | 峰值 ≤ 800 MB/s | 分帧上传，避免单帧大块 |
| 缺页可见 pop | 0（渲染祖先） | 0 | 允许 LOD 变粗，不允许洞 |
| PSO 数量 | 项目内可枚举、有上限告警 | 同左 | 顶点布局哈希驱动复用 |

测量纪律：

- CPU 基准（`pkg/prism_asset_build_core/benches/mesh_virtual_geometry.rs`）+ GPU 时间戳 + 帧捕获三层证据共同组成一次有效样本；CPU 构建与 LOD 选择由该基准给出，页解码、上传、GPU cull、光栅、材质解析和回收由运行时捕获分别计时。
- 每次运行给出 P50/P95/P99、峰值 VRAM、上传字节、resident/page miss、PSO 创建数、设备丢失次数和功耗采样；跨设备比较只比较同分辨率、同档位、同场景种子。
- 画质对标协议与缺陷等级沿用渲染架构文档 §15.5/§15.6，几何侧补充“轮廓误差/屏幕误差/LOD 跳变”三项。
- 性能结论必须绑定基准 artifact；画质结论必须绑定 Golden 图与误差报告，文档中的预算数字本身不替代实测。

### 18.1 2026-10-15 当前实测

实测设备为 Apple M2（10 核 GPU）、Metal、macOS 27.0.1（26A434），工具链为 `rustc 1.96.1` / `cargo 1.96.1`。`mesh_virtual_geometry` CPU 基准使用 8192 个独立三角形，生成 391 个 Cluster、4 页；另用带法线、UV、切线的 32×32 网格测 LOD 构建，请求与实际数量均为 `[2048, 1024, 512]`。每组采集 24 个样本，LOD 构建预热后测量并逐次比较资产确定性，对比时间不计入构建时间。日志为 `target/benchmarks/mesh/2026-10-15/attribute_lod_cpu.log`：

| CPU 指标 | P50 | P95 | P99 |
|---|---:|---:|---:|
| 固定 Cluster 构建 | 2,094,500 ns | 2,578,334 ns | 2,770,625 ns |
| 属性感知三层 LOD 构建 | 13,552,292 ns | 63,611,542 ns | 77,742,375 ns |
| LOD 选择 ×64 | 875 ns | 1,041 ns | 1,125 ns |

该次运行与其他编译/测试并发，尾部耗时包含系统争用，不能作为隔离环境性能门禁或版本间加速结论；选择计时明确为 64 次，不是单次调用。这组数字只证明当前资产构建、真实属性网格简化与 CPU LOD 选择的测量结果。真实 GPU 测试证明 GPU 路径执行正确，但本轮不产生 100 万 Cluster/1080p/4K 完整场景的 GPU timestamp、峰值 VRAM、上传带宽、缺页、功耗或帧时间数据，因此不据此宣称达到上表预算。

## 19. 诊断、测试与验证

### 19.1 必备诊断

- 每资产：顶点/三角形数、属性与流、索引宽度、LOD 链、Cluster 数、页数、量化误差、构建耗时与内存峰值。
- 每帧：可见 Cluster 数、剔除通过率（视锥/锥/遮挡）、cut 分布、缺页次数、驱逐次数、上传字节、形变顶点数、PSO 切换次数。
- 降级原因：每条“为什么走了这条路径 / 为什么用了这个 LOD / 为什么缺页”都可枚举。

### 19.2 Debug View

- 运行时统一使用 `GpuSceneDebugView` 作为调试视图入口，并与材质 ID / 对象 ID 视图共存（渲染架构 §15.2）。
- Mesh 诊断协议保留三角形密度/超采样率、Cluster 轮廓、LOD、Cut、页驻留、法线锥与材质槽字段；验收时以实际诊断输出或捕获物为证据，不把字段清单本身视为已生成的 Debug View。

### 19.3 测试金字塔

测试分成“当前完成性证据”和“产品级持续回归协议”，两者不得混写：

1. **当前 CPU 契约证据**：`cargo test -p prism_mesh` 的 34 项库单测、14 项集成测试通过，`cargo test -p prism_asset_build_core --lib` 的 136 项 library tests 通过；定向 `virtual_geometry_lod` 测试 12 项通过。覆盖属性/索引/拓扑、包围体、哈希、分流布局、批量形变、溢出拒绝、CLST v2/v3、12 种格式精确 gather、截断/损坏载荷、含属性分簇、压缩页、endpoint-QEM 实际简化、材质/缝/分类约束、版本化身份、f64 BVH 与跨尺度量化误差。
2. **真实 GPU 证据**：2026-10-15 沙盒外验证中，RenderApp Mesh 过滤集 20 项全部通过，其中标准与 half2/half4/附加颜色属性的 8×8 绘制、批量形变对照、current/previous 的 16×16 绘制实际执行 GPU 工作；数学蒙皮强制 parity 4 项通过；虚拟几何、GDeflate、OMM 强制 suites 通过。当前主工作树的完整 RenderApp 编译被非 Mesh 的 `ecs_columns.rs:451` API 不匹配阻断，20 项 Mesh 过滤集已在仅隔离该非 Mesh 编译问题的当前源码副本中完成验证；不把临时副本改动计入产品实现。CLST v3 整页 Metal storage roundtrip 逐字节一致只证明压缩/驻留/读取；`prism_virtual_geometry_gpu --test cluster_vertex_pull_parity` 已在真实 render pass 中从原始 CLST v3 字节解码位置与 `Unorm8x4` 属性并读回颜色，补足 shader 属性解码证据，但不扩写为完整场景材质 Golden 图结论。
3. **产品级图像回归协议**：标准网格、虚拟几何、蒙皮、Morph、LOD 过渡按质量档采集 Golden 图和 TAA 稳定性帧；只有生成捕获物并通过误差阈值时才形成图像质量结论。
4. **产品级压力协议**：百万级 Cluster、极端顶点数、极端尺度、非有限输入、退化/越界索引和超预算场景按固定种子执行；结果必须绑定测试日志与捕获物。
5. **确定性协议**：同输入多次构建哈希一致、不同线程数结果一致；跨平台差异必须由版本化量化规则解释并记录。
6. **回归门槛**：LOD 跳变、缺页 pop、轮廓误差、PSO 数量和显存超配分别设置断言与诊断字段，不用单一“测试通过”覆盖所有维度。

本批同时执行 `cargo test -p prism_render_architecture --lib`：11,303 项通过；这是跨渲染架构的回归快照，不把全部测试数计为新增 Mesh 用例。

### 19.4 故障注入矩阵

| 注入 | 期望 |
|---|---|
| 索引越界 | 构建期错误；运行时不崩溃 |
| 位置含 NaN/Inf | 构建期返回 `MeshError::NonFiniteAttributeValue`，不生成资产 |
| 骨骼权重全零 | 绑定姿态，不产生 NaN |
| 页数据截断/损坏 | 检测 schema/hash，回退 fallback mesh，不崩溃 |
| 页请求洪峰 | 固定容量合并，溢出走预算降级 |
| 设备丢失 | 几何重建，场景语义不变 |
| 极端缩放（1e-5 / 1e5） | 包围体与误差不溢出 |

### 19.5 2026-10-22 补充复核

当前工作树原样执行 `cargo test -p prism_mesh --lib --tests`：34 项库单测、14 项集成测试全部通过。该结果只确认 `prism_mesh` 数据模型、变形参考实现及其集成契约；不覆盖资产构建、RenderApp、RenderScene、真实 GPU、产品级图像回归或完整场景性能门槛。尝试执行 `cargo test -p prism_render_app mesh:: --lib -- --nocapture` 时，因编译产物写满磁盘（`No space left on device`）而未进入测试阶段，因此本次不记作 RenderApp 通过或失败。对应构建产物已清理；其余 Mesh 验收仍以 §26 的逐项证据为准。

## 20. 易维护性与演进规则

### 20.1 稳定契约

- `prism_mesh` 的公共承诺：属性语义与格式、索引宽度、拓扑、包围体含义、确定性布局输出。实现可重新打包（交错/分流/量化）而不改语义。
- `prism_render_architecture::geometry` 的 `GEOMETRY_ABI_VERSION` 与 `virtual_geometry::VIRTUAL_GEOMETRY_VERSION` 是显式契约版本；破坏性改动必须升版本并提供迁移。
- 落盘几何制品（虚拟几何页、meshlet、压缩流）必须版本化并可在构建期重建。

### 20.2 扩展点（注册而非分支）

| 扩展需求 | 扩展方式 | 禁止 |
|---|---|---|
| 新增顶点语义 | 属性注册表加注册项 + 格式 | 在 `Mesh` 里加 `match` 分支 |
| 新增拓扑 | 拓扑注册项 + 图元数规则 | 使用者自行算图元数 |
| 新增形变器（如布料） | `DeformationKind` + 形变调度注册 | 在蒙皮内核里塞特例 |
| 新增 LOD 选择器 | 误差函数策略 + 预算参数 | 硬编码距离阈值 |
| 新增压缩/量化 | 量化 profile + 版本化 schema | 运行时“就地量化” |
| 新增光栅后端 | 实现同一 Visibility Buffer 契约 | 后端私有几何语义 |

### 20.3 弃用与迁移

- `prism_mesh` 是 Prism 原生资产与几何算法的唯一真相；`prism_render_app`、`prism_asset_build_core` 和虚拟几何路径直接消费它。
- `prism_render_scene` 是 Bevy 接入层：ECS 组件、`Assets<Mesh>`、`RenderAssets<RenderMesh>` 和 Bevy PBR 专用接口继续使用 Bevy 类型，几何算法通过 `geometry::mesh_adapter` 转为 `prism_mesh` 后执行。
- Bevy 接入层的转换必须集中在适配器并有契约测试；禁止在剔除、拓扑、量化、打包或着色几何算法中重新读取 `bevy_mesh` 数据。
- 当 Bevy 接入层被 `prism_app`/原生 RHI 管线替代后，可删除适配器及其 Bevy Mesh 资产接缝；原生几何契约无需改变。
- API 弃用遵循仓库策略：先标记、后告警、再删除；每个阶段有迁移说明。

### 20.4 代码组织纪律

- `prism_mesh` 保持 `#![forbid(unsafe_code)]`、无设备依赖、无 ECS；任何需要 GPU 的逻辑都不进这个 crate。
- 单文件职责单一（现状 `mesh.rs`/`attribute.rs`/`indices.rs`/`bounds.rs`/`error.rs` 的划分正确）；新增能力开新文件而不是膨胀 `mesh.rs`。
- 数值代码有 Golden 与误差界；算法改动必须同时改测试。
- 文档与代码同源：格式矩阵由 RHI 枚举测试约束，档位由能力 profile 约束，CPU 预算由 Mesh 基准约束，GPU 预算由 GPU timestamp 参数约束；变更必须在同一提交更新对应文档与测试。

## 21. 公共 API 契约

以下签名与当前仓库导出 API 对齐；字段语义、错误处理和版本规则构成稳定契约，物理缓冲布局仍由渲染后端管理。

### 21.1 数据半边（`prism_mesh`）

```rust
// 属性注册：语义 + 格式 + 位置
pub struct MeshVertexAttribute {
    pub name: &'static str,
    pub format: VertexFormat,
    pub shader_location: u32,
}

pub enum VertexAttributeValues {
    Float32(Vec<f32>),
    Float32x2(Vec<[f32; 2]>),
    Float32x3(Vec<[f32; 3]>),
    Float32x4(Vec<[f32; 4]>),
    Float16x2(Vec<[u16; 2]>),
    Float16x4(Vec<[u16; 4]>),
    Uint32(Vec<u32>),
    Uint32x2(Vec<[u32; 2]>),
    Uint32x4(Vec<[u32; 4]>),
    Sint32(Vec<i32>),
    Unorm8x4(Vec<[u8; 4]>),
    Snorm8x4(Vec<[i8; 4]>),
}

impl Mesh {
    pub fn new(topology: PrimitiveTopology) -> Self;
    pub fn insert_attribute(&mut self, attr: MeshVertexAttribute, values: VertexAttributeValues)
        -> Result<(), MeshError>;
    pub fn indexed(self, indices: Indices) -> Self;
    pub fn with_submeshes(&mut self, submeshes: Vec<SubMesh>);
    pub fn vertex_layout(&self) -> Option<VertexBufferLayout>;      // 单流（兼容）
    pub fn stream_layouts(&self, profile: &StreamLayoutProfile) -> Vec<VertexStreamLayout>;
    pub fn vertex_buffer_data(&self) -> Vec<u8>;
    pub fn bounds(&self) -> Option<MeshBounds>;
    pub fn validate(&self) -> Result<(), MeshError>;
}

pub struct SubMesh {
    pub material_slot: u16,
    pub index_range: Range<u32>,
    pub cluster_range: Option<Range<u32>>,
}
pub struct VertexStreamLayout {
    pub kind: VertexStreamKind,
    pub array_stride: u64,
    pub attributes: Vec<StreamAttribute>,
}

pub fn content_hash(mesh: &Mesh) -> MeshContentHash;
```

### 21.2 运行时几何表（`prism_render_architecture::geometry`）

```rust
pub struct GeometryRecord {
    pub handle: GeometryHandle,
    pub revision: u32,
    pub vertex_buffer_class: u32,
    pub index_buffer_class: u32,
    pub primitive_start: u32,
    pub primitive_count: u32,
    pub lods: Vec<GeometryLodRecord>,
}

impl GeometryRecord {
    pub fn resolve_lod(&self, requested: u32) -> Option<&GeometryLodRecord>;
}
```

### 21.3 虚拟几何（`prism_render_architecture::virtual_geometry`）

```rust
pub fn plan_frame(
    hierarchy: &ClusterHierarchy,
    view: FrameView<'_>,
    raster: RasterConfig<'_>,
) -> GeometryFramePlan;
pub fn cluster_cull(
    frustum: &Frustum,
    bounds: &SceneBounds,
    occlusion: Option<OcclusionProbe>,
) -> CullVerdict;
pub fn select_lod(
    levels: &[LodLevel],
    projection: LodProjection,
    view_distance: f32,
    closing_speed: f32,
    policy: GeometryLodPolicy,
    previous: Option<u32>,
) -> Option<LodSelection>;
pub fn select_raster_path(
    stats: ClusterRasterStats,
    capability: RasterCapability,
    software_pixel_threshold: f32,
) -> GeometryRasterPath;
```

构建侧 `prism_asset_build_core` 提供 `build_static_virtual_geometry_lod_chain` 默认入口与 `build_static_virtual_geometry_lod_chain_with_policy(asset_id, mesh, settings, targets, &StaticLodPolicy)` 策略入口。`StaticVirtualGeometryLod` 同时提供 `target_triangles`、`achieved_triangles`、`source_content_hash`、`chain_build_hash`、`measured_vertex_error`、`vertex_error` 和独立分页资产；`StaticVirtualGeometryLodSet::new` 验证元数据后向共享选择器提供显示/预取资产。策略只控制候选代价与分类约束，不能解除开放边/拆分缝/材质与拓扑保护。

### 21.4 形变（设备无关输入、原生 RHI 缓存与共享调度）

`prism_mesh::deformation` 导出 `DeformationMesh::new(positions, normals, targets, skin, max_influences)`、`DeformationPose { morph_weights, joints, corrections }`、`JointTransform`、`DeformationCorrection` 与 `DeformedVertex`。`validate_pose` 在任何 GPU 写入之前校验数量、数值、矩阵与保守算术包络；`evaluate` 提供批量 CPU reference。四/八影响 profile 均拒绝超量输入，不截断。

`prism_render_app` 在 crate 根公开 `DeformationSource`、`DeformationCache`、`DeformationCacheError`、`DeformationOutput`。`DeformationSource` 一次上传 immutable authored streams、创建绑定布局和 compute pipeline；同一设备上的多个实例通过 `Arc<DeformationSource>` 共享这些资源，每个实例只拥有 pose、correction、双输出和 bind group。`DeformationCache::new` 是单实例便捷入口，`from_source` 用于显式复用：

```rust
impl DeformationCache {
    pub fn new(device: Arc<dyn RenderDevice + Send + Sync>, queue: &dyn RenderQueue,
        mesh: DeformationMesh) -> Result<Self, DeformationCacheError>;
    pub fn from_source(source: Arc<DeformationSource>, queue: &dyn RenderQueue)
        -> Result<Self, DeformationCacheError>;
    pub fn submit_pose(&mut self, queue: &dyn RenderQueue, frame: u64,
        pose: DeformationPose<'_>) -> Result<DeformationOutput, DeformationCacheError>;
    pub fn output(&self) -> Option<DeformationOutput>;
    pub fn invalidate_history(&mut self);
    pub fn allocated_bytes(&self) -> u64;
}
```

缓存拥有 `Arc` 共享 source 与 GPU 资源，满足 `Send + Sync + 'static`；source 的 immutable buffer、绑定布局和 pipeline 只创建/上传一次，实例 Drop 不会提前释放仍被其他实例引用的资源。输出 32 字节 stride，位置 offset 0、法线 offset 16；`current` 与 `previous` 都可作为 vertex/storage 输入。双缓冲要求所有消费者在同一有序队列的下一次姿态更新前提交；没有无限历史保留。失败不写 buffer、不推进 frame。当前测试已验证两个实例共享 source、各自拥有独立输出与绑定资源，并验证 source 最后释放时才释放 immutable GPU 资源。

共享预算调度继续使用 `prism_render_architecture::deformation`：

```rust
pub struct DeformationHandle(pub u32);
pub enum DeformationKind { Skinning, Morph, Cloth, VertexAnimation, Hair, Particle, Water }
pub struct DeformationBudget { pub vertices_per_frame: u32, pub blas_refits_per_frame: u32 }
```

## 22. 实施与验收矩阵

路线已改为可重复执行的能力矩阵；每一行都有代码入口、验证入口和运行时回退，不用时间线措辞表示完成状态。

| 能力包 | 代码入口 | 验证入口 | 运行时行为 |
|---|---|---|---|
| Mesh 数据契约 | `pkg/prism_mesh` | `cargo test -p prism_mesh` | 校验失败拒绝上传并返回结构化错误 |
| 分流与字节打包 | `prism_mesh::streams`、`RenderMesh` | `pkg/prism_render_app/src/mesh/tests.rs` | 超过设备流/属性预算时选择单流 profile 或拒绝不兼容内容 |
| 几何表与 LOD 驻留 | `prism_render_architecture::geometry` | geometry table tests | 请求 LOD 未驻留时选择已驻留祖先或 fallback |
| Bevy 场景 Mesh/LOD 接入 | `PrismGpuSceneLodSet`、`RenderGpuScene`、`RenderGeometryRegistry`、`geometry::consumer` | `CARGO_BUILD_JOBS=1 cargo check -p prism_render_scene --lib`；场景适配器/生命周期/geometry 单测 | base 与独立 LOD asset 稳定绑定；LOD 回收不破坏 base；冲突绑定拒绝并记录诊断；所有场景 draw fail-closed |
| 虚拟几何决策层 | `virtual_geometry::{hierarchy,lod,cull,frame}` | module tests + GPU parity suites | 以 `GeometryFramePlan` 统一 cut、bins 与 page requests |
| 属性感知静态 LOD | `virtual_geometry_lod`、`mesh_decimation::vertex_preserving` | LOD/拓扑/typed gather/BVH/量化测试及 attributed-grid 基准 | 精确 source remap，策略可注册；拓扑阻止简化时报告实际数量；不合法策略或误差拒绝构建 |
| 多消费者 Mesh 绑定 | `geometry::consumer`、`mesh::consumer`、`render_scene::geometry::consumer` | architecture geometry tests + `cargo test -p prism_render_app mesh:: --lib` + 场景适配器单测 | 主视图/Visibility/透明/阴影/VSM 在实际 draw 前统一经过 contract；Motion 使用 current/previous；缺页使用显式 fallback；派生消费者要求版本化 artifact |
| 虚拟几何 GPU twin | `prism_virtual_geometry_gpu` | `cargo test -p prism_virtual_geometry_gpu` | 适配器缺少所需特性时回退 CPU/标准网格路径 |
| 页驻留与 GPU 池 | `prism_asset::streaming`、`prism_render_app::render_asset` | page residency/page pool tests | 预算不足驱逐低优先级页，祖先覆盖画面 |
| GDeflate GPU twin | `prism_gdeflate_gpu` | `cargo test -p prism_gdeflate_gpu` | GPU 解码不可用时使用 CPU 解码，保持同一 payload ABI |
| 批量形变与缓存 | `prism_mesh::deformation`、`prism_render_app::DeformationCache`、共享调度 | 10 项 CPU reference 测试、4 项原生缓存测试（其中 2 项真实 GPU） | CSR Morph → 4/8 LBS → 外部修正；跨帧双缓冲；输入错误保留上一帧，跳帧/teleport 失效历史 |
| OMM/光追派生 | `prism_micromap(_gpu)` 与资产 DAG | OMM parity/capture tests | 不具备 RT/OMM 能力时使用标准阴影/光栅几何 |
| 性能回归 | `benches/`、GPU timestamps、capture artifacts | §18 protocol | 超预算触发档位降级、预取收紧或页驱逐 |

每次能力变更必须同时更新契约测试、诊断字段、回退路径和本文件矩阵；这四项缺一则变更不具备可合并状态。

## 23. 第一个可合并切片

第一个可合并切片为“契约、对齐与上传闭环”：

1. 新增 `pkg/prism_mesh/src/topology.rs`：拓扑种类与图元数计算。✅
2. Mesh 属性语义唯一性与浮点有限性校验：重复语义、NaN/Inf 均在上传前拒绝，并保留精确诊断位置。✅
3. 新增静态契约模块：`submesh.rs`、`limits.rs`、`morph.rs`、`skin.rs`，并接入 `Mesh::validate()`。✅
4. 扩展 `VertexAttributeValues`：补 `Float16x2/4`、`Uint32`、`Sint32`、`Snorm8x4`，与 `VertexFormat` 一一对齐，并加“格式矩阵全覆盖”单测。✅
5. 新增 `pkg/prism_mesh/src/hash.rs`：确定性内容哈希（位模式、固定遍历序）。✅
6. 新增 `streams.rs`：位置/着色/蒙皮/形变分流布局与实际字节打包，并接入 `RenderMesh` 多 buffer、queue、frame graph 和 draw slot。✅
7. 修复 `MeshBounds` 对有限极值坐标的派生运算溢出，并加入极值回归测试；无法由 `f32` 表示的范围/半径按契约饱和至有限最大值。✅
8. 批量形变共享 ABI、CPU reference 与原生 RHI 双缓冲缓存：50,000 顶点 CPU/GPU 对照，Morph-only、四/八影响、非均匀缩放、修正清零、资源释放、非法输入不改历史、current/previous 直接绘制像素断言。✅
9. 属性感知 endpoint-QEM、精确 typed gather、材质/缝/拓扑保护、版本化 LOD 配方身份、BVH 采样误差与 CLST 量化误差包络。✅

理由：设备无关工作可独立单测，GPU 上传闭环验证真实 buffer 的写入与替换语义；极值数值测试确保有限输入不会因中间运算变成 NaN/Inf。

## 24. 关键决策记录（ADR）

| # | 决策 | 理由 | 替代方案与代价 |
|---|---|---|---|
| A1 | 属性按 shader location 排序、单一语义单一格式 | 保证布局字节确定性、PSO 复用 | 允许自由顺序 → 布局哈希抖动 |
| A2 | 保留授权索引宽度 | 16 bit 索引带宽减半 | 统一 u32 → 无谓带宽成本 |
| A3 | `prism_mesh` 保持设备无关、`forbid(unsafe_code)` | 可单测、可复用、职责清晰 | 混入 GPU 逻辑 → 无法在无设备环境测试 |
| A4 | 顶点词表复用 RHI 类型 | 避免两套枚举漂移 | 自定义枚举 → 需要维护转换与校验 |
| A5 | 虚拟几何从 meshlet 原型演进并保留显式 schema 版本 | 可缓存、可重建、可回退 | 无版本 → 静默错读 |
| A6 | 静态高模默认自动生成虚拟几何，无需手工 LOD | 美术效率与一致性 | 手工 LOD → 成本高、易漂移 |
| A7 | CPU golden → GPU twin 逐元一致作为验收手段 | 既有仓库范式，可证明移植正确 | 只跑编译 → 无法证明数值正确 |
| A8 | 材质槽是 Cluster 划分硬约束 | 避免材质槽跨 Cluster 导致解析错误 | 允许跨槽 → visibility 解析歧义 |
| A9 | 形变一次写入缓存、多 pass 复用 | 降低成本、保证多 pass 一致 | 每 pass 重算 → 成本高且可能不一致 |
| A10 | 降级必须可诊断，缺页渲染祖先 | 不出现洞、可解释 | 缺页不画 → 画面破损 |

## 25. 风险控制与能力矩阵

### 25.1 风险

| 风险 | 影响 | 对策 |
|---|---|---|
| 虚拟几何实现复杂度高 | 帧时间与内容制作成本上升 | CPU 决策层、Compute twin、标准网格 fallback 共用同一 ABI |
| 平台差异（Metal/移动/Web 无 Mesh Shader） | 路径分裂 | 双后端同一 Visibility Buffer 契约；能力档位选择 |
| 蒙皮/Morph 与虚拟几何冲突 | 动态几何受限 | 动态几何走标准路径；Deformable Cluster 使用独立 profile |
| 页流送抖动导致 pop | 画质抖动 | 祖先渲染 + 预取 + dither；缺页计数进门槛 |
| 量化误差累积到轮廓 | 画质下降 | 误差并入 `object_error`；构建期报告与门禁 |
| PSO 爆炸 | 首帧卡顿 | 顶点布局哈希 + 材质域声明；数量告警 |
| 显存超配 | 掉帧/崩溃 | 预算驱动驱逐；超配可诊断；禁止 OOM |
| 与 `bevy_mesh` 双轨维护 | 维护成本 | 原生 RHI 只依赖 `prism_mesh`；Bevy 接入层集中适配并以契约测试锁定 |

### 25.2 已验证能力矩阵

| 能力 | 当前证据 | 失败/降级语义 |
|---|---|---|
| Mesh 数据与批量形变 reference | `cargo test -p prism_mesh`：34 项库单测、14 项集成测试通过 | 返回结构化错误，不创建 GPU 表示；不截断影响列表 |
| 分流布局、实际字节打包与标准 draw | Apple M2/Metal 上标准和 half2/half4/附加颜色属性的 8×8 GPU 绘制通过；流打包复用一次临时分配 | 选择兼容单流 profile 或拒绝超出 `Limits` 的布局 |
| 虚拟几何构建 | `prism_asset_build_core --lib` 当前 136 项通过；`virtual_geometry_lod` 定向 12 项通过；v3 属性保留、endpoint-QEM、精确 remap、BVH/量化误差与反例均有测试 | 拓扑保护导致目标不可达时记录实际数量；坏策略/身份/非有限误差拒绝构建 |
| 原生批量形变与历史 | 已保存的沙盒外 GPU 验证中 RenderApp Mesh 过滤集 20 项通过；批量对照与 current/previous 光栅像素测试实际执行真实 GPU | pose/限制校验失败保留资源与历史；首帧/跳帧/teleport previous 别名 current；Drop 释放全部自有资源 |
| 固定页 GPU 池事务 | `prism_render_app` page pool tests 通过 | 事务失败回滚，保留旧 resident range |
| 虚拟几何 CPU 决策 | `prism_render_architecture` tests 通过 | 空/损坏层级生成空计划并记录诊断 |
| 虚拟几何、GDeflate、OMM GPU twin | Apple M2/Metal 上以 `PRISM_REQUIRE_GPU=1` 强制执行对应 suites 并通过 | 缺少适配器/特性时保持 CPU reference 与标准路径 |
| 蒙皮 GPU twin | `prism_math_gpu --test skinning_parity` 在 Apple M2/Metal 上实际运行 4 项并通过 | 无适配器且强制 GPU 时直接失败，不记录为 parity 通过 |
| 强制真实 GPU 探测 | 沙盒内因没有适配器而按预期失败；沙盒外检测到 Metal 适配器并执行 GPU 工作 | 无适配器的结果只记录环境缺失，不计为 GPU 通过 |
| 设备丢失与预算超限 | page pool、asset residency 状态机测试通过 | 延迟释放、驱逐、重上传，禁止悬挂句柄与 OOM |
| CPU 性能样本 | §18.1 记录 8192 三角形/391 Cluster/4 页与属性网格 `[2048,1024,512]` 实际简化，24 样本 P50/P95/P99 | 并发负载样本；选择计时为 ×64，不外推为完整场景 GPU 性能 |
| 完整场景性能门槛 | §18 固定场景、计时字段和 P50/P95/P99 协议 | 只有 GPU timestamp、VRAM、上传、缺页与功耗捕获齐全时才判定预算 |

## 26. 最终验收矩阵：标准与本轮证据

| 维度 | 产品级验收标准 | 本轮证据与结论 |
|---|---|---|
| 数据模型 | 属性/拓扑/索引/子网格/包围体/哈希有正反单测；格式矩阵覆盖 RHI 变体 | `prism_mesh` 34 项库单测与 14 项集成测试通过；该维度形成契约证据 |
| 布局与标准路径 | 同属性集产生确定布局，上传字节与 draw 布局一致 | CPU 契约与 Metal 标准/半精度分流 8×8 像素测试通过；不等同于全档位 Golden 图集 |
| 虚拟几何 | 固定 Cluster、CLST v3、QEM LOD、CPU 计划、页 ABI、LOD/cull/raster GPU twin 一致 | `prism_asset_build_core --lib` 136 项、`virtual_geometry_lod` 定向 12 项通过；属性感知简化保留精确源顶点与全部 typed 属性；原始 CLST v3 vertex-pulling render pass 已通过颜色读回，整页 IO 一致不再被误写成属性 shader 着色证据，但仍不替代完整场景 Golden 图 |
| 形变 | Skin/Morph、缓存、预算与深度/阴影/速度/RT 消费一致 | 批量 Morph/LBS/修正的 CPU/GPU 对照（含 50,000 顶点）与 16×16 current/previous 直接绘制通过；数学蒙皮 parity 4 项通过。全部引擎 pass/RT 消费接线和跨档位 Golden 图不由这些独立测试证明 |
| 多消费者绑定 | 主视图、深度、阴影、Motion Vector、RT、Collision、Derived Geometry 使用统一身份与明确回退 | `geometry::consumer`、`prism_render_app::mesh::consumer` 与 `prism_render_scene::geometry::consumer` 契约测试；场景侧实际主视图、Visibility、透明、阴影和 VSM draw 在取 allocator slice 前统一校验 `MainView`/`Shadow`。形变历史、RT/Collision/Derived 的具体制品仍按各自输入协议选择，不得伪装为静态场景 draw 已覆盖 |
| 光追派生 | OMM 与派生几何使用版本化资产契约；无 RT 能力时选择光栅路径 | OMM 强制 GPU parity 通过；不把它扩张为完整 RT 场景画质结论 |
| 带宽与性能 | §18 固定场景输出 P50/P95/P99、GPU timestamp、VRAM、上传、缺页与功耗字段 | 已记录 CPU 构建/LOD P50/P95/P99；完整场景 GPU 字段无本轮 artifact，因此预算表保持回归门槛而非达标声明 |
| 维护性 | 扩展点遵守 §20.2，schema/ABI 显式版本化，错误类型稳定 | CLST v3/v2 兼容、稳定格式标签、缓存失效、精确 framing 与类型化错误由代码和测试锁定 |
| 诊断 | 每条降级可解释，诊断字段可导出 | 通用 `GpuSceneDebugView` 与状态机诊断作为现有入口；Mesh 专用视图只有产生实际捕获物时才计入图像证据 |

## 27. 结论

Prism 的 Mesh 系统只有一个真相源和三个产物面：**设备无关的授权/运行时数据**（`prism_mesh`）、**构建期派生的运行时网格、Cluster 分区资产与 QEM LOD 资产集合**（`prism_asset_build_core`）、**多消费者的 GPU 表示**（`prism_render_app`/`prism_render_scene`/`prism_render_architecture`）。系统通过分流顶点、域量化、层级误差选择、统一 Visibility Buffer 语义、预算驱动驻留和可解释回退，维持性能、质量、扩展性与维护性的同一套契约。

文档中的所有能力都以 §22 的代码入口、§26 的验收项和 §18 的证据格式判定；实现、测试、诊断与回退必须在同一提交中同步更新。这样 Mesh 不会因为增加新格式、新后端或新形变器而分裂出第二份真相。
