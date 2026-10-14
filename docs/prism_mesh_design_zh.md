# Prism Mesh 系统设计：顶级次世代 AAA 级几何数据模型与运行时几何管线

> 状态：完整设计与验证版（v2）
> 面向版本：Prism `0.20-dev` 当前仓库契约
> 目标读者：渲染、资产工具链、平台、动画/物理与性能工程人员
> 本文依据：当前仓库代码、CPU 单测、RenderApp 资产上传测试、虚拟几何 GPU twin 与 GDeflate GPU twin 的验证结果；性能门槛采用可复现基准协议记录，设备实测写入同一验证记录
> 关联文档：`prism_rendering_architecture_zh.md`（全局渲染架构）、`prism_asset_design_zh.md`（资产内核）、`prism_asset_build_design_zh.md`（构建/烘焙）、`prism_material_pipeline_refactor_zh.md`（材质与着色 ABI）、`prism_scene_design_zh.md`（场景与实例）、`prism_aaa_advanced_features_zh.md`（AAA 特性总表）
> 最后更新：2026-10-14

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

1. **一份授权数据 + 一次构建 + 多个产物面。** `Mesh`（authoring/IR）是唯一真相；`prism_asset_build` 把它派生成运行时网格、虚拟几何页、碰撞几何、光追几何、缩略图等制品。任何运行时路径不得反向修改授权数据。
2. **数据半个 crate，GPU 半个 crate。** `pkg/prism_mesh` 只持有与设备无关的几何数据（属性、索引、拓扑、包围体、校验、打包），`#![forbid(unsafe_code)]`、无 ECS、无 GPU 句柄；GPU 形态（缓冲、上传、生命周期）归 `prism_render_app`/`prism_render_scene`。职责分离直接保证可测试性。
3. **顶点词表属于 RHI，不属于 Mesh。** 属性格式、缓冲布局、拓扑、索引宽度直接复用 `prism_render_driver` 的类型，避免两套枚举漂移（现状已如此，本文将其固化为契约）。
4. **顶点流分流 + 域相关量化。** 位置/蒙皮/形变/材质属性按访问模式拆流，深度与阴影 pass 只绑定它真正需要的流；量化策略按几何变体（静态 vs 蒙皮 vs 虚拟几何）分级，而不是全局一刀切。
5. **虚拟几何是主线，不是可选彩蛋。** 静态高模默认走离线 Cluster 构建 + 连续误差 LOD + 分页流送 + 软件/硬件双光栅；标准 mesh 路径在小网格、动态几何、低端档与非虚拟化场景下长期共存，两者共享同一套 Cluster cull 输入与材质解析契约。
6. **基准即规格，证据与设计分栏。** 每项性能/画质目标都写成可测量的门槛；实现证据、运行策略和能力降级在同一矩阵中记录，不用宣传语替代证据。

## 2. 目标、非目标与硬约束

### 2.1 效果目标

- **任意复杂度静态几何可达**：高模无需手工离散 LOD；导入即自动生成 Cluster/DAG 分页结构，屏幕上百万级可见三角形仍保持稳定帧时。
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
- **弃用纪律**：`crates/bevy_mesh` 仅服务旧示例与旧兼容路径；新 RHI 代码统一使用 `pkg/prism_mesh`，迁移与删除条件见 §20、§22。

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
| UE5 Nanite（虚拟几何） | 离线 Cluster + DAG、屏幕空间误差 LOD、Cluster 剔除、软件/硬件双光栅、Visibility Buffer、分页流送、按 Cluster 局部量化 | 单一大 DAG 的整体实现细节；Prism 保持显式 schema 与可插拔光栅后端 |
| UE5 StaticMesh / SkeletalMesh / Geometry Script | 资产与派生几何分离；同一资产派生渲染/碰撞/导航/光追几何；编辑期脚本化几何 | 不在运行时依赖编辑器脚本层 |
| Unity Mesh / DOTS / Entities Graphics | 顶点流分流与紧凑格式；批处理友好的共享布局；SRP Batcher 式的“材质键 → 状态缓存” | 不采用托管运行时与 GC 依赖 |
| meshoptimizer / Draco / MeshOpt 编码 | 索引重排提升缓存局部性；顶点 cache-aware 编码；属性量化与边拆分 | 只借算法，实现独立且可被 GPU twin 校验 |
| id Tech / Frostbite 的虚拟几何路线 | Compute 软件光栅到 Visibility Buffer；按需页请求与预算 | 不做与特定主机强绑定的固定管线假设 |
| Metro Exodus / 现代主机引擎的 Mesh Shader 路径 | Mesh Shader 生成可见 Cluster 与间接参数 | Prism 不把高层算法绑定在 Mesh Shader 上，保留 Compute/SW 回退 |
| DirectStorage / RTX IO / GDeflate | GPU 直通解压与分块 I/O；容器化流送 | 解压内核与容器格式独立实现 |
| UE5 Virtual Shadow Map / OMM | 几何与阴影/光追的协同；alpha 覆盖微图 | 只是几何侧契约，阴影算法归别文档 |
| USD / glTF / Alembic | 授权数据分层（几何/骨架/动画/材质分开）+ 稳定子资产标识 | 不把 USD/glTF 顶层结构直接当作引擎运行时 ABI |
| Simplygon 类自动 LOD 工具 | 自动简化、轮廓与误差报告、硬边/UV seam 保护 | 简化算法自研并与 Golden 指标挂钩 |

### 3.2 借什么、不借什么

**借形态**：Cluster + DAG + 屏幕空间误差是当前公开信息里对“任意复杂度几何”最可验证的形态；分流顶点 + 域量化是带宽优化的成熟做法；一份授权数据派生多产物是 AAA 内容流水线的共识。

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
③ Build & Bake (prism_asset_build)  IR → 运行时网格 / 虚拟几何页 / 碰撞 / 光追 / 预览
        │  显式 schema 版本 + build_settings_hash，落 CAS/DDC
        ▼
④ GPU Residency (prism_render_app / prism_render_scene)
        │  缓冲类、上传、子分配、驻留状态机、页表
        ▼
⑤ Geometry Consumers（互不重复真相）
        │  标准 draw / GPU-driven 实例 / 虚拟几何 / 光追 / 物理 / 编辑器预览
        ▼
⑥ Deformation (prism_render_architecture::deformation + prism_math_gpu)
           蒙皮/Morph/布料/顶点动画 → 形变缓存 → 多 pass 复用
```

层级依赖只向下。任何消费者都不得重新解包授权数据，只能消费构建期产物与其运行时句柄。

### 5.2 crate / 模块布局（目标态）

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
  └─ hash.rs         确定性内容哈希（DDC 键）

pkg/prism_asset_build/src/mesh      IR 修复/优化/简化/Cluster 化/量化/分页（§17）
pkg/prism_render_architecture
  ├─ geometry/        稳定几何表（GeometryRecord / GeometryLodRecord / ABI 校验）
  ├─ virtual_geometry/ Cluster cull、屏幕空间误差 LOD、页表、光栅路径、软件光栅、cut 选择
  ├─ deformation/     DeformationHandle / Kind / Budget，形变调度
  └─ gpu_scene/       实例身份与上传（几何只引用其句柄）
pkg/prism_render_scene/src/geometry   几何资产生命周期与 consumer API（ECS 侧）
pkg/prism_math_gpu/src/skinning       GPU 蒙皮内核（CPU golden → GPU twin 校验）
pkg/prism_micromap(_gpu)              OMM 烘焙与孪生
pkg/prism_virtual_geometry_gpu        vis-buffer 软件光栅 GPU 孪生（元/逐像素校验）
pkg/prism_gdeflate_gpu                页解压的去交错 GPU 孪生
crates/bevy_mesh                      仅保留给旧示例与兼容路径；RHI 路径以 prism_mesh 为准（§20）
```

### 5.3 当前实现与验证基线（2026-10-14）

- `prism_mesh::Mesh` 已提供属性、索引、拓扑、单流布局、分流布局、字节打包、包围体、校验、Morph、Skin、SubMesh 与确定性内容哈希；`#![forbid(unsafe_code)]`、无 ECS、无 GPU 句柄。
- `VirtualGeometryVersion=1` 已提供 Cluster 资产契约、量化 payload 编解码、层级/包围体校验、页表、驻留页存储、LOD 选择、滞回、速度预取、Cluster cull、RasterPath 分类、软件光栅与逐帧计划组合。
- `GeometryTable` 已提供 generation 句柄、代际回收、LOD 驻留回退与 ABI 检查；`RenderMesh` 已把分流数据接入 queue、frame graph、draw slot 与固定页 GPU 池。
- CPU 证据包括 `prism_mesh`、`prism_render_architecture`、资产流式化与渲染资产页池测试；GPU twin 证据覆盖虚拟几何计算内核与 GDeflate 去交错内核。真实设备测试使用 `PRISM_REQUIRE_GPU=1` 强制适配器存在，未暴露适配器时直接失败而不把 Null RHI 计为 GPU 通过。
- 本文把已验证契约、运行时降级、性能门槛和演进规则统一成可执行规格；代码实现以这些契约和测试作为唯一判定来源。

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
| 位置 | `Vertex_Position` | `[f32;3]` | `Float32x3` | 虚拟几何：`Snorm16x3` 级局部量化；蒙皮：`Float32x3` | 12 / 6 |
| 法线 | `Vertex_Normal` | `[f32;3]` | `Snorm8x4` 或八面体 `Snorm16x2` | 八面体 `Snorm16x2` | 4 |
| 切线 | `Vertex_Tangent` | `[f32;4]` | `Snorm8x4`（handedness 入 w） | 由法线 + 符号位重建 | 4 / 1 |
| UV | `Vertex_Uv0..3` | `[f32;2]` | `Float16x2` | `Unorm16x2`（按 UV 包围盒归一） | 4 |
| 顶点色 | `Vertex_Color` | `[f32;4]` | `Unorm8x4` / `Snorm8x4` | `Unorm8x4` | 4 |
| 骨骼索引 | `Vertex_JointIndex` | `[u32;4]` | `Unorm8x4`（≤256 骨/批）或 `Uint16x4` | `Unorm8x4` | 4 / 8 |
| 骨骼权重 | `Vertex_JointWeight` | `[f32;4]` | `Unorm8x4`（归一化） | `Unorm8x4` | 4 |
| Morph 增量 | （纹理化） | `[f32;3]` | 稠密：属性；大量目标：delta 纹理 | delta 纹理 | 0（纹理） |
| 自定义 | `Vertex_Custom0..N` | 项目定义 | 由注册项声明 | 同左 | 视格式 |

约束：

- 属性数量必须落在 `Limits::max_vertex_attributes` 内（`baseline` 16、`desktop` 32）；顶点缓冲数不得超 `max_vertex_buffers`（8 / 16）。
- 同一语义在整个项目内只有一个格式定义；变体（如法线的两种编码）只能是不同**流类别**，不能是同一语义的两种隐式含义。
- 格式矩阵与 `prism_render_driver::VertexFormat` 的 12 个现有变体一一对齐；未进入 RHI 的 `Unorm16x2`、`Snorm16x2/3` 等编码只存在于虚拟几何 payload，不伪装成顶点输入格式。

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
| 静态虚拟几何 | Cluster 空间 `Snorm16x3`（局部量化） | 八面体 `Snorm16x2` | `Unorm16x2` | 与 Nanite 形态一致；量化误差进 DAG 误差 |
| 蒙皮网格 | `Float32x3` | `Snorm8x4` | `Float16x2` | 位置不量化：形变需要精度且每帧被 GPU 写 |
| Morph 顶点 | `Float32x3`（目标空间） | 由法线重建 | 同基网格 | 增量幅度小，可选 `Float16x3` |
| 地形 | clipmap/heightfield 专用 | 由高度差分 | 无 | 不走通用网格量化 |
| 毛发/草 | 程序化生成 | 程序化 | 程序化 | 不落静态页 |

说明：表中 `Snorm16x2/x3`、`Unorm16x2`、`Float16x3` 等编码不属于 `prism_render_driver::VertexFormat` 顶点输入；虚拟几何 payload 直接使用自己的量化编码，标准网格使用 RHI 当前的 `Float32/x2/x3/x4`、`Float16x2/4`、`Uint32/x2/x4`、`Sint32`、`Unorm8x4`、`Snorm8x4`，并在构建 profile 中记录实际格式。

### 7.3 量化与误差界

- **位置量化**基于 Cluster/页的局部包围盒，把位置映射到归一化整数；解码用页头部的 `scale/offset`。误差上界 = `局部extent / (2^bits - 1)`，必须并入该 Cluster 的 `object_error`，使 LOD 选择能感知量化误差。
- **法线**用八面体映射到两个有符号 16 bit（或 8 bit）；误差以最大角误差（度）报告，屏幕空间阈值由材质类型决定（法线贴图会掩盖部分误差）。
- **切线**优先不存满 4 分量：法线 + 符号位重建切线（保证 UV 连续）或每 Cluster 存切线空间基。
- **UV** 按 UV 包围盒归一化到 `Unorm16`；若 UV 超出 [0,1]（平铺 UV），退化为 `Float16x2` 并在序列化中标记。
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

- 默认路径：标准 GPU skinning，4 个影响（`Compatible`）到 8 个影响（`Standard`/`Ultra`）。
- 内核：`prism_math_gpu::skinning`（CPU golden + GPU 孪生校验范式）。
- 法线变换使用骨骼旋转（或用逆矩阵转置，按骨骼是否均匀缩放决定），构建期标记骨架是否存在非均匀缩放以选择正确路径。
- 权重在构建期归一化；和为零的顶点必须退化为绑定姿态（而不是变成 NaN）。

### 9.2 Morph target / blend shape

- 少量目标：每目标一个顶点属性流，权重作为 uniform。
- 大量目标（面部、纠正形变）：**delta 纹理**形态——每目标一层纹理，`(xyz)` 存位移；顶点阶段按权重采样累加。这避免属性数量爆炸。
- 稀疏增量：只存变化的顶点及其索引，配合间接索引表；构建期报告稀疏率。
- 法线 Morph：默认不存（用基法线 + 可选切线空间修正），可选“重算法线”标志由材质选择。

### 9.3 复合顺序

固定顺序，跨 pass 一致：

```text
bind pose → morph targets（按权重累加）→ skinning（骨骼变换）→ 布料/物理修正（若有）→ 输出
```

顺序必须写进着色/形变契约文档；任何变体（如“先蒙皮后 morph”）必须显式声明，不得隐式。

### 9.4 形变缓存与 motion vector

- 形变结果写入**形变缓存**（double-buffered：当前帧 + 上一帧），供主视图、阴影、速度、可见性、光追 refit 复用。
- 速度 = 当前形变位置 − 上一帧形变位置（世界空间）；缺上一帧时退化为实例位移。
- `deformation::{DeformationBudget, DeformationKind}` 是预算与分类入口；每秒形变顶点数、BLAS refit 次数是硬预算。
- 形变缓存的分配与普通顶点缓冲分属不同缓冲类（见 §12），避免与静态几何争夺分配器。

### 9.5 Deformable Cluster（Nanite skinning 形态）

`Ultra` Deformable Cluster profile：让蒙皮高模也能走虚拟几何，做法是把骨骼影响“烘焙进 Cluster”：

- Cluster 携带影响它的骨骼集合（通常 ≤ 64 骨骼/Cluster）与原位姿态。
- 运行时用 Compute 逐 Cluster 重建顶点位置（局部），再走同一套 Cluster cull 与光栅路径。
- 代价是每帧几何计算与内存上升；策略只对单角色、次要与近景层开启，LOD 以“变形误差”而非仅几何误差驱动。

## 10. LOD 系统

Prism 有两条互补的 LOD 机制，二者共享误差语义：

- **离散 LOD 链**（标准路径、蒙皮、程序化几何）：构建期生成 LOD0..N，运行时按屏幕空间误差选择。
- **连续虚拟几何 cut**（静态高模）：DAG 上的“割”，允许同一物体不同区域处在不同细化级别。

### 10.1 屏幕空间误差

统一公式（已在渲染架构 §6.3 提出，此处作为几何侧契约固化）：

```text
projected_error = object_error * projection_scale / max(view_depth, near_epsilon)
refine if projected_error > threshold_pixels
```

- `object_error` 对离散 LOD 来自简化误差，对虚拟几何来自 DAG 节点误差（父节点误差不得小于子节点）。
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
Mesh IR → 修复/校验 → 顶点/索引优化
        → LOD/简化（可选，用于 fallback 与远景）
        → Cluster 化（≤128 三角形 / ≤64 顶点）
        → 分组 + 相邻图
        → 逐组 50% 简化（锁定接缝、材质槽、UV seam、硬边）
        → 递归构建 DAG/BVH8（父误差 ≥ 子误差）
        → 量化（Cluster 局部位置）+ 法线/切线/UV 编码
        → 按空间邻近与父子访问相关性打包固定大小页
        → 生成 fallback mesh
```

硬约束：

- **材质槽不可跨**：简化不得合并不同材质槽的三角形。
- **蒙皮/UV 接缝锁定**：锁定顶点在简化中不移动，避免 UV 拉伸与蒙皮裂缝。
- **退化处理**：零面积/重复顶点在 Cluster 化前移除并计数。
- **误差单调**：DAG 父节点误差不得小于子节点，违反则构建失败（不是警告）。
- **GPU twin 对齐**：Cluster 布局、页大小、量化编码必须与 `prism_virtual_geometry_gpu` 的 ABI 一致。

### 11.2 资产格式

`pkg/prism_render_architecture::virtual_geometry::VirtualGeometryAsset` 是可分页资产的 v1 编解码契约（渲染架构 §6.2 的字段落地）：

```text
VirtualGeometryAsset
├─ Header
│  ├─ magic / schema_version / build_settings_hash / content_hash
│  ├─ bounds / material_slot_count
│  └─ root_page_ids / fallback_mesh_ref
├─ HierarchyPages       DAG 节点、误差、包围体、法线锥、子页引用
├─ GeometryPages
│  ├─ 量化位置（Cluster 局部）
│  ├─ 法线/切线（八面体）/ UV / 颜色 / 自定义
│  ├─ Cluster 描述符（三角形索引 8 bit + 顶点索引、材质范围）
│  └─ Cluster 包围体 + 法线锥
├─ DependencyTable      页之间的父子/依赖关系（用于预取）
└─ StreamingMetadata    页大小、页计数、优先级与 locality 提示
```

- 页大小固定（建议 64 KiB 起，1024 字节粒度的元数据对齐；具体值在 M2 定档并与 GPU twin 锁定）。
- 页存储可压缩（zstd/GDeflate 形态），运行时经 `prism_gdeflate_gpu` 直通解压或 CPU 解压。
- 必须使用显式 `schema_version` 与 `build_settings_hash`；旧缓存可检测并重建，禁止静默错读。

当前代码已对 header、层级节点、页排序/大小、cluster 范围、非空 cluster、cluster/node page identity、层级/Cluster AABB 包含关系、材质槽、payload 引用和确定性编解码做校验；CLST v2 量化 payload 已由运行时架构 crate 统一持有，构建器通过兼容导出复用同一 codec；页面进入 `VirtualGeometryPageStream` 前会校验 magic/version、精确 framing、bounds、索引以及 descriptor 的顶点/三角形计数，错误不会污染 resident storage。资产构建器的多级简化 DAG 和 `prism_gdeflate_gpu` 的逐字节解压校验均属于资产产物的强制验证规则。

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

规则：所有派生几何在 `prism_asset_build` 的 Artifact DAG 中显式表达依赖（资产构建文档 §12.2），不以“渲染时临时转换”替代构建。

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

以下是可执行预算门槛。每次基准运行同时记录设备标识、后端、驱动、分辨率、质量档、场景种子、冷启动/热启动状态和 P50/P95/P99；预算值用于回归判定，实测样本写入 `target/benchmarks/mesh/<date>/`。固定场景为：`100 万 Cluster 静态场景`、`2000 万三角形近景高模`、`单角色 5 万顶点蒙皮`、`1 万实例共享几何`。

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

- Criterion 基准（`benches/`）+ GPU 时间戳 + 帧捕获三层证据共同组成一次有效样本；CPU 提交、页解码、上传、GPU cull、光栅、材质解析和回收分别计时。
- 每次运行给出 P50/P95/P99、峰值 VRAM、上传字节、resident/page miss、PSO 创建数、设备丢失次数和功耗采样；跨设备比较只比较同分辨率、同档位、同场景种子。
- 画质对标协议与缺陷等级沿用渲染架构文档 §15.5/§15.6，几何侧补充“轮廓误差/屏幕误差/LOD 跳变”三项。
- 性能结论必须绑定基准 artifact；画质结论必须绑定 Golden 图与误差报告，文档中的预算数字本身不替代实测。

## 19. 诊断、测试与验证

### 19.1 必备诊断

- 每资产：顶点/三角形数、属性与流、索引宽度、LOD 链、Cluster 数、页数、量化误差、构建耗时与内存峰值。
- 每帧：可见 Cluster 数、剔除通过率（视锥/锥/遮挡）、cut 分布、缺页次数、驱逐次数、上传字节、形变顶点数、PSO 切换次数。
- 降级原因：每条“为什么走了这条路径 / 为什么用了这个 LOD / 为什么缺页”都可枚举。

### 19.2 Debug View

- 三角形密度/超采样率热图、Cluster 轮廓、LOD 级别着色、Cut 可视化、页驻留热图、法线锥可视化、材质槽着色。
- 与材质 ID / 对象 ID 视图共存（渲染架构 §15.2）。

### 19.3 测试金字塔

1. **单元测试（CPU golden）**：属性插值/合并、索引校验、包围体、量化/反量化误差界、Cluster 划分、DAG 误差单调、页打包与解包、子网格区间、Morph/Skin 数据结构；`pkg/prism_mesh/src/tests.rs` 与虚拟几何模块测试覆盖这些契约。
2. **GPU 孪生一致性**：软件光栅、LOD 选择、vis word 编解码、投影、OMM、GDeflate 去交错 —— 与 CPU 参考逐元/逐像素比对（`prism_virtual_geometry_gpu`、`prism_micromap_gpu`、`prism_gdeflate_gpu` 已建立范式）。
3. **图像测试**：标准网格、虚拟几何、蒙皮、Morph、LOD 过渡在不同档位的 Golden 图；含 TAA 稳定性帧。
4. **压力测试**：百万级 Cluster、极端顶点数、极端尺度、非有限输入、恶意簇（NaN/退化/越界索引）、超预算场景。
5. **确定性测试**：同输入多次构建哈希一致；不同线程数构建结果一致；跨平台位模式一致（允许量化差异的部分显式标注）。
6. **回归门槛**：LOD 跳变、缺页 pop、轮廓误差、PSO 爆炸、显存超配都必须有专用断言。

### 19.4 故障注入矩阵

| 注入 | 期望 |
|---|---|
| 索引越界 | 构建期错误；运行时不崩溃 |
| 位置含 NaN/Inf | 跳过该顶点，包围体不受污染，诊断计数 |
| 骨骼权重全零 | 绑定姿态，不产生 NaN |
| 页数据截断/损坏 | 检测 schema/hash，回退 fallback mesh，不崩溃 |
| 页请求洪峰 | 固定容量合并，溢出走预算降级 |
| 设备丢失 | 几何重建，场景语义不变 |
| 极端缩放（1e-5 / 1e5） | 包围体与误差不溢出 |

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

- `crates/bevy_mesh` 只保留给旧示例与 `bevy_pbr` 兼容路径；新代码一律用 `prism_mesh`。
- `prism_mesh` 与 `bevy_mesh` 之间**不保留**格式转换桥；迁移通过重导入（构建期）完成，避免长期维护两套格式。
- 删除条件：当（a）示例与 `bevy_pbr` 不再引用 `bevy_mesh`，且（b）RHI 路径的几何消费全部走 `prism_mesh` + 虚拟几何，即可移除旧兼容路径。
- API 弃用遵循仓库策略：先标记、后告警、再删除；每个阶段有迁移说明。

### 20.4 代码组织纪律

- `prism_mesh` 保持 `#![forbid(unsafe_code)]`、无设备依赖、无 ECS；任何需要 GPU 的逻辑都不进这个 crate。
- 单文件职责单一（现状 `mesh.rs`/`attribute.rs`/`indices.rs`/`bounds.rs`/`error.rs` 的划分正确）；新增能力开新文件而不是膨胀 `mesh.rs`。
- 数值代码有 Golden 与误差界；算法改动必须同时改测试。
- 文档与代码同源：格式矩阵由 RHI 枚举测试约束，档位由能力 profile 约束，预算由 Criterion/GPU timestamp 参数约束；变更必须在同一提交更新对应文档与测试。

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

pub struct SubMesh { pub material_slot: u16, pub index_range: Range<u32> }
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

### 21.4 形变（`prism_render_architecture::deformation`）

```rust
pub struct DeformationHandle(pub u32);
pub enum DeformationKind { Skinning, Morph, Cloth, VertexAnimation, Hair, Particle, Water }
pub struct DeformationBudget { pub vertices_per_frame: u32, pub blas_refits_per_frame: u32 }
```

## 22. 实施与验收矩阵

路线已改为可重复执行的能力矩阵；每一行都有代码入口、验证入口和运行时回退，不用日期或“下一阶段”表示完成状态。

| 能力包 | 代码入口 | 验证入口 | 运行时行为 |
|---|---|---|---|
| Mesh 数据契约 | `pkg/prism_mesh` | `cargo test -p prism_mesh` | 校验失败拒绝上传并返回结构化错误 |
| 分流与字节打包 | `prism_mesh::streams`、`RenderMesh` | `pkg/prism_render_app/src/mesh/tests.rs` | 超过设备流/属性预算时选择单流 profile 或拒绝不兼容内容 |
| 几何表与 LOD 驻留 | `prism_render_architecture::geometry` | geometry table tests | 请求 LOD 未驻留时选择已驻留祖先或 fallback |
| 虚拟几何决策层 | `virtual_geometry::{hierarchy,lod,cull,frame}` | module tests + GPU parity suites | 以 `GeometryFramePlan` 统一 cut、bins 与 page requests |
| 虚拟几何 GPU twin | `prism_virtual_geometry_gpu` | `cargo test -p prism_virtual_geometry_gpu` | 适配器缺少所需特性时回退 CPU/标准网格路径 |
| 页驻留与 GPU 池 | `prism_asset::streaming`、`prism_render_app::render_asset` | page residency/page pool tests | 预算不足驱逐低优先级页，祖先覆盖画面 |
| GDeflate GPU twin | `prism_gdeflate_gpu` | `cargo test -p prism_gdeflate_gpu` | GPU 解码不可用时使用 CPU 解码，保持同一 payload ABI |
| 形变契约 | `prism_render_architecture::deformation`、`prism_math_gpu::skinning` | CPU golden/GPU twin tests | 动态几何走标准路径，复用形变预算与缓存接口 |
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
| 与 `bevy_mesh` 双轨维护 | 维护成本 | 新 RHI 代码只依赖 `prism_mesh`；旧路径通过重导入保持兼容 |

### 25.2 已验证能力矩阵

| 能力 | 当前证据 | 失败/降级语义 |
|---|---|---|
| Mesh 属性、拓扑、索引、SubMesh、包围体、哈希 | `cargo test -p prism_mesh` 通过 | 返回 `MeshError`，不创建 GPU 表示 |
| 分流布局与实际字节打包 | RenderMesh mesh tests 与渲染资产测试通过 | 选择兼容单流 profile 或拒绝超出 `Limits` 的布局 |
| 固定页 GPU 池事务 | `prism_render_app` page pool tests 通过 | 事务失败回滚，保留旧 resident range |
| 虚拟几何 CPU 决策 | `prism_render_architecture` tests 通过 | 空/损坏层级生成空计划并记录诊断 |
| 虚拟几何 GPU twin | `cargo test -p prism_virtual_geometry_gpu` 通过 | 缺少适配器/特性时保持 CPU reference 与标准路径 |
| GDeflate GPU twin | `cargo test -p prism_gdeflate_gpu` 通过 | GPU 解码不可用时使用 CPU 解码 |
| 强制真实 GPU 探测 | `PRISM_REQUIRE_GPU=1` 使无适配器环境直接失败 | 结果不计为 GPU 通过，CI 记录适配器缺失 |
| 设备丢失与预算超限 | page pool、asset residency 状态机测试通过 | 延迟释放、驱逐、重上传，禁止悬挂句柄与 OOM |
| 性能门槛 | §18 的固定场景、计时字段和 P50/P95/P99 协议 | 超预算触发质量档位、LOD、预取和驻留策略调整 |

## 26. 最终验收矩阵

| 维度 | 验收标准 | 证据 |
|---|---|---|
| 数据模型 | 属性/拓扑/索引/子网格/包围体/哈希全部有单测；格式矩阵全覆盖 RHI 变体 | `cargo test -p prism_mesh` |
| 布局确定性 | 同属性集产生字节一致布局，哈希稳定 | 单测 + 哈希快照 |
| 校验 | §6.8 全部规则有正/反用例；坏数据不崩溃 | 单测 + 故障注入 |
| 标准路径 | `Compatible` 档标准网格 Golden 图通过 | 图像测试 |
| 虚拟几何 | CPU 计划、页 ABI、LOD/cull/raster GPU twin 逐项一致；缺页使用祖先/fallback | 模块测试 + `cargo test -p prism_virtual_geometry_gpu` |
| 形变 | Skin/Morph 数据结构、预算和 GPU twin 契约一致；动态路径回退标准网格 | `prism_mesh`/`prism_math_gpu` golden 与 render tests |
| 光追 | OMM 与派生几何使用版本化资产契约；不具备 RT 时选择光栅 fallback | OMM parity/capture tests |
| 带宽/性能 | §18 场景协议输出 P50/P95/P99、GPU 时间戳、VRAM、上传与缺页字段 | Criterion + GPU timestamp artifact |
| 维护性 | 扩展点按 §20.2 注册方式落地；无分支式扩展 | 代码评审 + 契约测试 |
| 诊断 | 每条降级可解释 | Debug View + 诊断输出 |

## 27. 结论

Prism 的 Mesh 系统只有一个真相源和三个产物面：**设备无关的授权/运行时数据**（`prism_mesh`）、**构建期派生的运行时网格与虚拟几何页**（`prism_asset_build`）、**多消费者的 GPU 表示**（`prism_render_app`/`prism_render_scene`/`prism_render_architecture`）。系统通过分流顶点、域量化、Cluster/DAG 误差选择、统一 Visibility Buffer 语义、预算驱动驻留和可解释回退，维持性能、质量、扩展性与维护性的同一套契约。

文档中的所有能力都以 §22 的代码入口、§26 的验收项和 §18 的证据格式判定；实现、测试、诊断与回退必须在同一提交中同步更新。这样 Mesh 不会因为增加新格式、新后端或新形变器而分裂出第二份真相。
