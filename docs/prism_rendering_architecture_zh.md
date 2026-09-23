# Prism 下一代渲染架构设计

> 状态：架构提案（Draft）  
> 面向版本：Bevy/Prism `0.20-dev` 及后续演进  
> 目标读者：渲染、资产、平台、工具链和性能工程人员  
> 本文依据：对当前仓库代码的静态阅读；未运行示例、测试或性能基准  
> 最后更新：2026-09-21

## 1. 摘要

Prism 的目标不是复制某个商业引擎的功能列表，而是建立一套可以持续演进的统一渲染底座：同一个 GPU Scene、同一套几何与材质标识、同一套可见性结果和同一套资源驻留系统，同时服务 PBR、NPR、自定义渲染、实时全局光照、虚拟阴影与离线路径追踪。

建议的核心设计可以概括为五点：

1. **统一场景真相源**：所有实时与离线渲染器消费 `GpuScene`，禁止各功能重复维护完整的实例、材质和光源镜像。
2. **GPU 驱动可见性**：实例、Cluster/Meshlet、LOD、遮挡、阴影页请求和光追实例筛选尽量在 GPU 上完成，CPU 只提交增量变更和策略参数。
3. **虚拟化资源统一调度**：几何页、纹理页、阴影页和光照缓存页共用预算、反馈、优先级、驻留与驱逐框架。
4. **策略与机制分离**：PBR、NPR、项目自定义着色是策略；Render Graph、GPU Scene、可见性、资源生命周期是机制。策略可以替换，机制必须稳定。
5. **能力分层而非最低公分母**：桌面高端、桌面兼容、移动、Web 各自选择最优路径，同时保持相同的场景语义和近似画面意图。

本文将目标系统命名为 **Prism Render Architecture（PRA）**。它不是一次重写，而是一组可以逐阶段落地、每阶段都有独立收益和回退路径的演进方案。

## 2. 目标、非目标与约束

### 2.1 产品目标

PRA 最终应支持：

- **UE 5.x 同等级画质**：在约定的高端桌面参考平台、相同源资产、分辨率、相机、光照和效果设置下，Prism 的最终帧必须达到与 UE 5.x 同一质量等级；该目标是发布门槛，而不是宣传性愿景。
- 标准 PBR：金属度/粗糙度、清漆、各向异性、次表面、透射、薄膜、织物等可组合 BSDF。
- NPR：卡通分层、描边、艺术化高光、Ramp、Hatching、材质 ID/对象 ID 驱动效果。
- 自定义渲染：自定义 Pass、自定义可见性分类、自定义材质域、自定义后处理和 Compute 管线。
- 类 Nanite 的虚拟几何：离线自动 Cluster 构建、连续误差 LOD、GPU 选择、按需流送、软件/硬件光栅路径。
- 统一 GPU Scene：稳定句柄、结构化数据表、逐字段脏标记、增量上传、GPU 端散写和延迟回收。
- 类 Lumen 的动态 GI/反射：软件追踪优先的广覆盖路径、硬件 Ray Query 增强路径、屏幕空间补充、时空复用与可伸缩质量。
- 类虚拟阴影：按页分配、接收者驱动请求、缓存复用、局部失效、按需渲染和传统阴影回退。
- 离线渲染预留：共享场景/材质/灯光语义，独立高质量积分器，可扩展 AOV 和确定性输出。
- 跨平台自动适配：按特性和性能基准选择渲染档位，而不是只按操作系统硬编码。
- 高性能与稳定性：明确 CPU/GPU/显存预算、无界资源保护、可诊断降级、可复现性能测试。

### 2.2 架构目标

- 功能之间共享数据，避免 Meshlet、阴影、GI、光追各自重复提取世界。
- 大多数实体变化的成本与“变化数量”成正比，而不是与“场景总量”成正比。
- Render Graph 中的资源依赖可被验证，Pass 可被替换、插入、裁剪和异步调度。
- 热路径数据以 SoA 或可按需读取的紧凑布局存在；用户侧 ECS 组件不直接定义 GPU ABI。
- 不把 NVIDIA、AMD、Apple 或单一 API 的专属能力写入高层场景接口。
- 实验功能必须通过 capability、quality tier 和 feature flag 三重门控。

### 2.3 非目标

- 不要求每个开发阶段都立即达到 UE 5.x 同等级画质；但高端档正式发布时必须通过本文的画质对标门槛，中间版本必须明确标记缺失项。
- 不保证所有平台输出像素完全一致；保证语义一致、质量档位可解释。
- 不用单一超级 Shader 覆盖一切。PBR/NPR/透明/体积/离线域允许不同执行模型。
- 不要求动态拓扑几何在第一版进入虚拟几何路径；蒙皮和变形需要专门路径。
- 不在没有公开基准、画质基线和目标硬件的情况下宣称“性能超过 UE”。

### 2.4 成功定义

“设计更先进”和“性能更好”必须转化为可测指标：

- 100 万静态实例中移动 1,000 个实例时，CPU 提取和上传成本近似 O(1,000)。
- 相机静止且场景不变时，几何、阴影和 GI 缓存写入趋近于零。
- 视锥外、遮挡和低贡献内容不产生材质着色成本。
- 资源超预算时质量平滑下降，不崩溃、不无限分配、不出现长期抖动。
- 同一场景可在高端桌面、兼容桌面、移动/Web 档位启动，并给出清晰的实际启用能力报告。
- 关键 Pass 具备 GPU 时间、工作量、缓存命中、溢出和降级原因指标。
- 高端参考平台上的标准对标场景通过几何、材质、光照、阴影、反射、体积、抗锯齿、后处理和时域稳定性九类画质验收；不得以更模糊、更闪烁或缺失效果换取所谓性能领先。

### 2.5 UE 同等级画质的定义

“一样好”不意味着逐像素复制 UE，也不以单张挑选截图判断。它表示在盲测和客观测试中，Prism 不存在稳定、显著、可感知的整体质量落差：

- **静态帧等价**：相同资产和灯光意图下，材质能量、间接光、阴影接触、反射完整度、几何细节和色彩层次处于同一等级。
- **运动画面等价**：相机移动、物体运动、遮挡显露、灯光变化时，没有更明显的闪烁、重影、拖尾、LOD 跳变、阴影抖动和 GI 泵动。
- **内容覆盖等价**：室内、室外、大世界、角色、植被、透明、发丝、体积、水体和高动态范围场景都有可用于生产的路径，不能只在静态建筑场景中达标。
- **创作上限等价**：艺术家可控制材质、灯光、曝光、调色、体积和镜头效果；不能只靠固定自动算法得到偶然好看的画面。
- **参考模式等价**：提供比实时模式更高质量的 Reference/Offline 模式，用于发现实时近似中的偏差。

画质对标使用当前项目锁定的 UE 5.x 基准版本。升级基准版本时必须记录 UE 项目设置、插件、硬件、驱动和所有可伸缩性参数，禁止只写“Epic/Cinematic”而遗漏具体配置。

## 3. 当前代码基线

当前仓库不是从零开始。以下能力可以保留并演进。

### 3.1 已有优势

#### Render World 与调度

`bevy_render` 已将渲染放入独立 `RenderApp`，并通过 `ExtractSchedule` 从主世界提取数据。`RenderSystems` 已区分资产准备、网格准备、视图创建、队列、排序、资源准备和提交等阶段。这为并行渲染和数据生命周期隔离提供了良好基础。

相关代码：

- `crates/bevy_render/src/lib.rs`
- `crates/bevy_extract/src/sync_world.rs`
- `crates/bevy_render/src/renderer/mod.rs`

#### GPU 驱动批处理与遮挡

`batching/gpu_preprocessing.rs` 已包含 GPU preprocessing、间接绘制参数、工作项缓冲、前后帧实例输入和 GPU culling 支持判断。它证明现有架构能够承载 GPU 驱动路径。

相关代码：

- `crates/bevy_render/src/batching/gpu_preprocessing.rs`
- `crates/bevy_render/src/occlusion_culling/`

#### 稀疏 GPU 更新

`SparseBufferVec` 和 `update_sparse_buffers` 已支持通过更新列表在 GPU 上 scatter，适合扩展为 GPU Scene 的增量字段更新基础设施。

相关代码：

- `crates/bevy_render/src/render_resource/sparse_buffer_vec.rs`
- `crates/bevy_render/src/render_resource/sparse_buffer_update.wesl`

#### Meshlet、BVH 与自动 LOD

现有 Meshlet 路径已经包含：

- 离线 Mesh 到 Meshlet 转换；
- 多级简化与误差传播；
- BVH8；
- Cluster 级视锥、遮挡和 LOD 选择；
- 持久 GPU 几何缓冲；
- visibility buffer；
- 大量几何单次/少量绘制；
- Vulkan 与 Metal 路径。

这已经是“类 Nanite”的重要原型，但还缺少真正的虚拟页流送、统一 GPU Scene、广泛材质支持、动态几何策略和完整跨平台回退。

相关代码：

- `crates/bevy_pbr/src/meshlet/`
- `crates/bevy_pbr/src/meshlet/asset.rs`
- `crates/bevy_pbr/src/meshlet/from_mesh.rs`

#### 可扩展材质

`Material`、`MaterialExtension` 和 `ExtendedMaterial` 已允许用户替换或扩展顶点、片元、prepass、deferred 和 meshlet shader。它适合作为迁移入口，但长期需要从“每种 Rust 类型生成管线”演进为“材质域 + 着色模型 + 参数记录”的稳定 ABI。

相关代码：

- `crates/bevy_pbr/src/material.rs`
- `crates/bevy_pbr/src/extended_material.rs`
- `crates/bevy_material/`

#### Solari 实时光追与参考路径追踪

`bevy_solari` 已有 Ray Query 场景、BLAS/TLAS、实时直接/间接光、ReSTIR 实验路径、world cache、反射测试以及非实时累积路径追踪器。场景变换提取已经使用 changed query，并直接更新实例数据。

相关代码：

- `crates/bevy_solari/src/scene/`
- `crates/bevy_solari/src/realtime/`
- `crates/bevy_solari/src/pathtracer/`

#### 跨平台后端

渲染器基于 wgpu，原生后端包含 Vulkan、Metal、DX12，并提供 WebGPU/GLES feature。已有能力查询和插件启用检查，可扩展成系统化 capability profile。

相关代码：

- `crates/bevy_render/src/settings.rs`
- `crates/bevy_render/Cargo.toml`

### 3.2 关键缺口

| 领域 | 当前状态 | 主要缺口 |
|---|---|---|
| 场景表示 | Render World 中各功能各自提取/维护数据 | 缺少跨 Raster、Shadow、GI、RT 共用的稳定 GPU Scene ABI |
| Meshlet | 连续 LOD、BVH、GPU culling 原型较强 | 缺几何分页/流送、动态预算、通用材质与完整回退 |
| 阴影 | 传统点光/聚光/级联方向光阴影 | 缺虚拟页表、局部失效、缓存复用和统一页预算 |
| GI | Solari 实验性 Ray Query + world cache | 硬件要求高，缺软件追踪路径和成熟降噪/回退矩阵 |
| 材质 | Rust 泛型材质与 shader specialization | 容易管线组合爆炸，PBR/NPR/RT/离线语义未统一 |
| 自定义渲染 | 可插入系统和 Pass，底层 API 可用 | 缺稳定扩展契约、资源声明和兼容性校验 |
| 离线渲染 | 验证用途 Pathtracer | 缺生产级采样、AOV、检查点、确定性和批处理接口 |
| 跨平台 | wgpu 能力查询与部分 feature gate | 缺自动档位选择、性能探测、质量策略和统一报告 |
| 性能治理 | 有 diagnostic 基础 | 缺帧预算控制器、统一缓存指标、回归场景与硬性门槛 |

### 3.3 不应直接延续的模式

- 不应让 Solari、阴影、Meshlet 和标准 Mesh 分别持有不兼容的“实例真相”。
- 不应将 `StandardMaterial` 的 CPU 结构直接当作永久 GPU/RT/离线材质契约。
- 不应通过无限增加 pipeline specialization key 实现所有功能组合。
- 不应在运行时同步执行昂贵 Meshlet/LOD 构建。
- 不应把“支持某个 WgpuFeatures”直接等同于“该路径在此设备上更快”。

## 4. 总体架构

```text
Main World / Assets / Editor
          │  Change Journal（创建、销毁、字段级变更）
          ▼
Render Scene Bridge
          │  分配稳定句柄、合并写入、版本检查
          ▼
┌──────────────────────── Unified GPU Scene ────────────────────────┐
│ Instance │ Geometry │ Material │ Light │ Probe │ Decal │ Volume │
│ Stable ID│ SoA data │ Bindless │ Bounds│ Links │ Flags │ Epoch  │
└───────────┬─────────────────┬──────────────────┬─────────────────┘
            │                 │                  │
            ▼                 ▼                  ▼
   GPU Visibility       Virtual Resource     Ray Scene Builder
 Instance/Cluster/LOD   Geometry/Texture/    BLAS/TLAS or
 Occlusion/Work Graph   Shadow/GI Pages      Software BVH/SDF
            │                 │                  │
            └────────────┬────┴──────────┬───────┘
                         ▼               ▼
                  Frame/Render Graph   Offline Graph
                  Raster/PBR/NPR/GI    Path Trace/AOV
                         │               │
                         └───────┬───────┘
                                 ▼
                          Post/Compose/Present
```

### 4.1 分层

建议将渲染系统分成六层：

1. **Scene Contract**：主世界到渲染世界的语义与增量协议。
2. **GPU Scene**：稳定句柄、GPU 表、生命周期和上传。
3. **Virtualization**：几何/纹理/阴影/GI 页与预算系统。
4. **Visibility & Lighting**：可见性、光源筛选、GI/反射、阴影请求。
5. **Shading & Render Graph**：PBR/NPR/自定义着色和 Pass 编排。
6. **Platform & Quality**：能力探测、自动档位、预算控制、回退。

层级依赖只能向下。离线渲染与实时渲染都依赖 Scene Contract 和材质语义，但离线积分器不依赖实时 GBuffer 布局。

### 4.2 建议 crate 边界

新能力应拆分为小 crate，避免继续放大 `bevy_pbr`：

```text
bevy_render
  ├─ backend/device/render graph/pipeline/resource primitives
  └─ 不拥有 PBR 语义

bevy_render_scene            新增
  ├─ stable handles / change journal / GPU Scene
  ├─ instance, geometry, material, light tables
  └─ frame snapshot / epoch reclamation

bevy_render_virtual          新增
  ├─ page allocator / residency / feedback / budgets
  └─ geometry, texture, shadow, radiance cache clients

bevy_visibility             可从 bevy_render 拆出
  ├─ instance/cluster culling / HZB / indirect command generation
  └─ visibility buffer contract

bevy_geometry_virtual       从 meshlet 演进
  ├─ offline builder / asset format / streaming
  └─ HW mesh shader + compute/software raster backends

bevy_material               演进
  ├─ material domain / shading model registry / parameter ABI
  └─ PBR, NPR and custom material compilation

bevy_lighting               新增或扩展 bevy_light
  ├─ clustered lights / shadow policy / direct lighting
  └─ shared light sampling ABI

bevy_virtual_shadow         新增
  └─ virtual page request, cache, invalidation and rendering

bevy_gi                     从 Solari realtime 演进
  ├─ screen/software/hardware tracing
  ├─ reservoir / radiance cache / denoising
  └─ reflection composition

bevy_offline_render         从 Solari pathtracer 演进
  └─ path tracer / AOV / batch / checkpoint / reference renderer

bevy_pbr
  └─ 标准 PBR 策略与用户友好组件，不再承担全部底层机制
```

初期不必立即物理拆 crate，可先在现有 crate 中按以上边界建 module，待 API 稳定后迁移。

## 5. 统一 GPU Scene

GPU Scene 是本设计的最高优先级。没有它，后续功能会重复提取、重复分配、重复上传，最终无法保持一致性。

### 5.1 身份模型

GPU 侧不得直接长期保存 ECS `Entity`。建议使用 64 位分代句柄：

```rust
#[repr(C)]
pub struct SceneHandle {
    pub index: u32,
    pub generation: u32,
}
```

- `index` 定位 GPU 表槽位。
- `generation` 防止销毁后复用槽位产生悬挂引用。
- ECS Entity 到 SceneHandle 的映射只存在于 bridge。
- 销毁后槽位进入延迟回收队列，至少等待 `frames_in_flight + async_compute_lag`。
- GPU 生成的反馈必须带 generation 或 frame epoch，CPU 忽略过期反馈。

### 5.2 数据布局

实例热数据使用 SoA，减少只读取 bounds 或 transform 时的带宽：

```text
GpuScene
├─ instance_transform_current[]
├─ instance_transform_previous[]
├─ instance_bounds[]
├─ instance_geometry_handle[]
├─ instance_material_table[]
├─ instance_flags[]
├─ instance_layer_mask[]
├─ instance_custom_data_offset[]
├─ geometry_records[]
├─ material_records[]
├─ light_records[]
└─ indirection/version tables[]
```

不要把所有字段放进一个巨大的 AoS `GpuInstance`。阴影、深度、GI、光追和主着色读取集合不同，SoA 可以减少无效流量。对于总是共同读取的 16 字节字段可局部 AoSoA 打包。

### 5.3 Change Journal

主世界以变更日志而非“复制所有可见实体”驱动渲染：

```rust
pub enum SceneChange {
    CreateInstance { handle: SceneHandle, descriptor: InstanceDescriptor },
    DestroyInstance { handle: SceneHandle },
    UpdateTransform { handle: SceneHandle, current: Affine3A, previous: Affine3A },
    UpdateBounds { handle: SceneHandle, bounds: SphereAabb },
    UpdateGeometry { handle: SceneHandle, geometry: GeometryHandle },
    UpdateMaterial { handle: SceneHandle, slot: u16, material: MaterialHandle },
    UpdateFlags { handle: SceneHandle, mask: u32, value: u32 },
}
```

实现原则：

- 同一帧对同一句柄同一字段的多次写入在 CPU 侧合并，只保留最终值。
- 创建后立即销毁可折叠为空操作。
- 结构变化与数值变化分队列；结构变化需要资源生命周期处理，数值变化走快速 scatter。
- 大块连续变化用 queue write/copy，小量离散变化复用 `SparseBufferVec` 的 GPU scatter。
- 上传环形缓冲必须有每帧字节上限；超限时按可见性和重要性排队，不能无界增长。

### 5.4 帧快照与一致性

建议引入不可变 `GpuSceneSnapshot(frame_epoch)`：

- 帧开始冻结当前 scene table 版本。
- Raster、Shadow、GI、RT 在同一帧读取相同版本。
- 异步计算读取 N 帧快照时，资源由 epoch reclamation 保活。
- 对 transform 保留 current/previous，运动矢量与时域算法不再由各功能各自猜测上一帧。
- Camera cut、world origin rebasing 和大规模 teleport 写入显式 epoch flag。

### 5.5 GPU Scene 更新流程

```text
ECS change detection
  → SceneChange 合并
  → 句柄分配/释放与引用验证
  → 连续上传 + 稀疏更新任务
  → RenderGraph::SceneUpload
  → GPU scatter / table patch
  → bounds refit / dirty hierarchy propagation
  → 发布 GpuSceneSnapshot
```

### 5.6 与当前代码的迁移关系

- 用现有 `SyncToRenderWorld`/`MainEntity` 建立初始映射，不立刻重写世界同步。
- 将 `BatchedInstanceBuffers` 的实例输入逐步改为引用 `SceneHandle`。
- 将 Solari 的独立 transform/material 镜像改为读取 GPU Scene，并只维护 RT 专属 acceleration structure metadata。
- 将 Meshlet `InstanceManager` 的实例 uniform 改为 GPU Scene 索引。
- 将标准 Mesh 与 Meshlet 路径统一到相同 instance/material/light handle。

### 5.7 公共 API 草案

```rust
pub trait RenderSceneComponent {
    type GpuValue: ShaderType + Pod;

    fn field_mask() -> SceneFieldMask;
    fn encode(&self, context: &SceneEncodeContext) -> Self::GpuValue;
}

pub struct GpuScenePlugin {
    pub capacities: SceneCapacities,
    pub upload_budget: UploadBudget,
    pub validation: SceneValidation,
}

pub struct SceneCapacities {
    pub initial_instances: u32,
    pub max_instances: u32,
    pub initial_materials: u32,
    pub max_materials: u32,
}
```

公共 API 应承诺句柄语义和字段含义，不承诺单一物理 Buffer；后端可重新打包。

## 6. 虚拟几何：从 Meshlet 原型到自动化几何系统

### 6.1 资产构建

导入器默认自动生成虚拟几何资产，无需美术手工制作离散 LOD：

1. 验证拓扑、去除退化三角形、规范 attribute stream。
2. 根据材质边界、硬边、UV seam 和骨骼影响划分初始 Cluster。
3. 构建相邻图，将 Cluster 组成 group。
4. 对 group 进行约 50% 简化，锁定边界，计算 object-space error。
5. 递归构建 DAG/BVH8，父节点误差不得小于子节点。
6. 量化位置、编码法线/切线、压缩索引和 attribute page。
7. 按空间邻近和父子访问相关性打包固定大小页。
8. 生成 fallback mesh，确保不支持虚拟几何的平台仍可显示。

当前 `from_mesh.rs` 已覆盖 2–5 的重要部分，应保留算法并把输出扩展为可分页格式。

### 6.2 资产格式

建议从单体 `MeshletMesh` 扩展为：

```text
VirtualGeometryAsset
├─ Header
│  ├─ magic/version/content_hash/build_settings_hash
│  ├─ bounds/material_slot_count
│  └─ root_page_ids/fallback_mesh
├─ HierarchyPages
├─ GeometryPages
│  ├─ compressed positions
│  ├─ normals/tangents/uv/color/custom attributes
│  ├─ cluster descriptors
│  └─ local material ranges
├─ DependencyTable
└─ StreamingMetadata
```

必须使用显式 schema version 和 build settings hash。旧缓存可检测并重建，不允许静默错读。

### 6.3 运行时 LOD

LOD 选择基于屏幕空间误差而不是距离阈值：

```text
projected_error = object_error * projection_scale / max(view_depth, near_epsilon)
refine if projected_error > threshold_pixels
```

还需加入：

- 速度偏置：高速相机预取更细一级父/子页。
- 滞回：进入和退出阈值不同，避免边界闪烁。
- 时间预算：每帧限制 hierarchy traversal 与新页请求量。
- 重要性：主视图、阴影、反射和次要视图使用不同权重。
- 驻留约束：子页不可用时渲染最近已驻留祖先，禁止出现洞。

### 6.4 两条光栅后端

高层算法不绑定 Mesh Shader：

- **Mesh Shader 路径**：Task/Mesh Shader 可用且实际基准更优时使用。
- **Compute + Indirect/Software Raster 路径**：用 Compute 生成可见 Cluster 和间接参数；微三角形可选择软件光栅到 visibility buffer。

这能覆盖 Metal/Vulkan/DX12 的差异，并避免 Web/移动因为没有 Mesh Shader 而失去整个架构。

### 6.5 Visibility Buffer

推荐虚拟几何写入紧凑 visibility buffer：

```text
visibility.x = scene_instance_index
visibility.y = cluster_or_primitive_id
depth        = hardware depth
barycentrics = 可重建、原生输出或额外紧凑目标
```

随后只对可见像素执行材质着色。优点：

- 几何可见性与材质复杂度解耦；
- 大量材质不会直接增加几何 draw call；
- PBR/NPR 可共享可见性；
- 可以做材质分类与 tile dispatch；
- 深度、阴影请求、运动向量可重用稳定 primitive identity。

透明、折射和部分 alpha-tested 材质保持独立 forward/fragment path。第一阶段 alpha mask 可在 visibility pass 使用简化 opacity shader。

### 6.6 动态几何策略

| 几何类型 | 首选路径 | 说明 |
|---|---|---|
| 静态高模 | 虚拟几何 | 完整分页、连续 LOD、Cluster culling |
| 刚体移动高模 | 虚拟几何实例 | 几何共享，只更新实例 transform |
| 蒙皮角色 | 标准 mesh + GPU skinning；后续 deformable cluster | 第一版避免每帧重建完整层级 |
| 地形 | 专用 clipmap/virtual heightfield 或虚拟几何 | 根据编辑和碰撞需求选择 |
| 毛发/草 | 程序化 Mesh Shader/Compute | 不强制转换为静态页 |
| 粒子 | Compute 生成 | 使用 GPU Scene light/layer/material 语义 |
| 小低模 | 标准 GPU-driven mesh | 避免虚拟几何固定开销 |

自动选择器应基于三角形数、屏幕占比、变形类型和设备能力，而不是所有 Mesh 一刀切。

## 7. 材质、PBR、NPR 与自定义渲染

### 7.1 材质域

建议建立稳定 `MaterialDomain`：

```rust
pub enum MaterialDomain {
    Surface,
    Decal,
    PostProcess,
    Volume,
    LightFunction,
    Ui,
}

pub enum SurfaceBlendMode {
    Opaque,
    Masked,
    Translucent,
    Additive,
    Modulate,
}
```

材质记录由固定 header + 参数块组成：

```text
MaterialRecord
├─ shading_model_id
├─ domain/blend/feature flags
├─ parameter_block_offset
├─ texture_handle_table_offset
├─ sampler_handle_table_offset
└─ custom_shader_entry_id
```

### 7.2 着色模型注册表

PBR 和 NPR 通过注册表并列存在：

```rust
pub trait ShadingModel: Send + Sync + 'static {
    const ID: ShadingModelId;
    const REQUIRED_INPUTS: MaterialInputMask;

    fn realtime_shader() -> ShaderRef;
    fn shadow_opacity_shader() -> Option<ShaderRef>;
    fn ray_hit_shader() -> Option<ShaderRef>;
    fn offline_closure() -> Option<OfflineClosureId>;
}
```

内建模型建议包括：

- `StandardPbr`
- `ClearCoatPbr`
- `SubsurfacePbr`
- `HairPbr`
- `ToonLit`
- `Unlit`
- `CustomSurface`

### 7.3 避免管线爆炸

组合爆炸是长期性能和稳定性的主要风险。采用三级策略：

1. 高频结构差异使用少量静态 permutation，例如 depth-only、opaque、masked。
2. 材质功能使用 bit mask 和统一参数 ABI，尽量运行时分支或 tile 分类。
3. 自定义模型按 shading model ID 将可见像素压缩到工作队列，再间接 dispatch 对应 Compute shader。

不要为“每个材质资产 × 每个灯光模式 × 每个调试视图”生成独立 pipeline。

### 7.4 PBR 质量基线

实时 PBR 至少统一以下约定：

- 线性工作空间和明确色域/显示变换；
- 多重散射能量补偿；
- IBL 预过滤与 DFG；
- 法线映射和几何法线的一致能量处理；
- 面积光 LTC 或采样路径；
- clearcoat/anisotropy/transmission 的一致直接光与间接光语义；
- 与离线路径追踪器共享参数解释和参考测试。

每种 BSDF 都需要 furnace test、白炉能量检查、粗糙度 sweep 和实时/离线差异图。

### 7.5 UE 同等级材质质量要求

要达到 UE 同等级画质，基础金属度/粗糙度材质远远不够。高端档必须覆盖：

| 材质能力 | 必须达到的结果 |
|---|---|
| 多层表面 | Base + clearcoat、灰尘、湿润、薄膜等层之间能量守恒，法线可独立混合 |
| 各向异性 | 拉丝金属在直接光、IBL、反射和离线模式中方向一致 |
| 次表面 | 皮肤、蜡、叶片具有可控 profile、厚度和透射，屏幕边缘不明显漏光 |
| 透射与折射 | 薄表面、实体介质、吸收、色散预留和粗糙透射具有一致语义 |
| 毛发 | 双高光、方位粗糙度、透射与多重散射；与专用毛发几何和阴影协同 |
| 布料 | Sheen、纤维方向和 grazing response 可控 |
| 地形/植被 | 多层混合、虚拟纹理、距离宏观变化、风动法线和双面 foliage shading |
| Decal | DBuffer/材质属性混合、法线/粗糙度/颜色独立通道和稳定排序 |
| 高质量纹理 | 各向异性过滤、正确 mip、BC/ASTC、虚拟纹理及稳定 streaming |

着色质量还必须包括：

- 高精度切线空间与 MikkTSpace 兼容导入；
- specular anti-aliasing，抑制高频法线和微几何导致的亮点闪烁；
- normal/roughness filtering，保证缩小时能量稳定；
- 光照单位、曝光、相机和环境亮度使用明确的物理单位；
- 路径追踪参考结果与实时结果共享 BSDF 参数定义；
- 材质编译失败时使用明确诊断材质，不能悄悄降级为错误外观。

### 7.6 NPR

NPR 不应通过破坏 PBR 管线硬塞入特殊分支。建议：

- `ToonLit` 读取统一的 direct/indirect light descriptors，但自行量化响应。
- Ramp 使用 bindless 纹理句柄，支持亮度、N·L、N·H 或艺术家自定义坐标。
- 描边拆成屏幕空间边缘和几何轮廓两种策略，可组合。
- visibility buffer 提供 instance/material/primitive ID，供边缘和风格化后处理使用。
- 允许 `CustomSurface` 输出自定义 feature buffer，但必须声明格式和预算。
- GI 可选择物理辐照度、量化后辐照度或完全禁用。

### 7.7 自定义 Pass API

自定义渲染必须声明资源和调度契约：

```rust
pub trait RenderFeaturePlugin {
    fn register_scene_fields(&self, registry: &mut SceneFieldRegistry);
    fn register_material_models(&self, registry: &mut MaterialRegistry);
    fn build_graph(&self, graph: &mut RenderGraphBuilder, caps: &RenderCapabilities);
}
```

Graph Pass 应声明：

- 读取/写入的逻辑资源；
- queue 类型（graphics/compute/copy）；
- 是否允许 async compute；
- 是否需要主视图、阴影视图或离线 sample context；
- capability predicate；
- fallback pass；
- transient resource 描述；
- debug name、预算类别和时间戳范围。

严禁插件依赖内部 Pass 的偶然执行顺序；依赖必须由资源或显式 edge 表达。

## 8. 虚拟资源系统

几何、阴影、纹理和 GI 虽然页内容不同，但生命周期相同：请求、分配、填充、使用、统计、驱逐。因此应共享机制。

### 8.1 核心接口

```rust
pub trait VirtualResourceClient {
    type VirtualKey: Pod + Ord;
    type PhysicalPage;

    fn classify_request(&self, key: Self::VirtualKey) -> RequestPriority;
    fn populate_page(&mut self, key: Self::VirtualKey, page: Self::PhysicalPage);
    fn invalidate(&mut self, event: InvalidationEvent);
}
```

统一系统提供：

- 物理页池；
- 多级页表；
- GPU feedback 去重与压缩；
- CPU/GPU 双侧预算；
- 优先级队列；
- LRU-K/clock 驱逐；
- pin、prefetch、grace period；
- 每客户端最低保障和全局压力仲裁；
- residency、fault、eviction、thrash 指标。

### 8.2 优先级

建议优先级评分：

```text
priority = view_importance
         × projected_coverage
         × temporal_urgency
         × content_importance
         × miss_penalty
         ÷ estimated_fill_cost
```

主相机可见几何通常高于远处阴影页；接近屏幕中心且高速靠近的内容应提前预取。所有权重可由 quality profile 调整。

### 8.3 防抖动

- 页至少驻留 N 帧后才可驱逐。
- 同一页连续 fault 提升长期优先级。
- 预算缩减分批进行，不在单帧大量驱逐。
- 统计工作集超过预算时主动提高 LOD/阴影误差阈值，而不是持续 fault。
- Camera cut 采用独立预热策略，暂时降低高阶效果采样。

## 9. 虚拟阴影

### 9.1 目标

- 高分辨率近景阴影；
- 大世界和大量局部光；
- 静态区域跨帧复用；
- 只渲染真正被接收者采样的页；
- 动态物体只失效相交页，而非整张 shadow map；
- 不支持路径回退到 CSM/atlas shadow map。

### 9.2 数据结构

```text
VirtualShadowSystem
├─ PerLightVirtualAddressSpace
│  ├─ directional clipmaps
│  ├─ spot projection
│  └─ point light cube/octahedral mapping
├─ PageTable
├─ PhysicalDepthPagePool
├─ PageMetadata
│  ├─ owner light / virtual coord / mip
│  ├─ last used / last rendered / caster epoch
│  └─ static/dynamic flags
├─ RequestBuffer
└─ InvalidationBuffer
```

### 9.3 帧流程

```text
主视图深度/visibility
  → 按接收者重建世界位置
  → 选择影响光源与所需 shadow mip
  → 写虚拟页请求
  → GPU 排序、去重、压缩
  → 分配物理页并更新页表
  → 构建每页 caster list
  → 仅渲染 dirty/new pages
  → shading 时查询页表
  → missing page 使用父 mip 或传统低分辨率 fallback
```

### 9.4 失效模型

为避免“任意物体移动导致整盏灯缓存清空”：

- GPU Scene 为 caster 保存 bounds 与 `shadow_caster_epoch`。
- 变换变化产生旧 bounds 和新 bounds 的 swept volume。
- 对受影响灯光，将 swept volume 投影到虚拟地址空间，标记相交页 dirty。
- 材质 opacity/shadow bias 改变也更新 caster epoch。
- 静态与动态 caster 可分层：静态页长期缓存，动态层单独合成或更频繁更新。
- 方向光 clipmap 滚动时复用仍在覆盖范围内的物理页。

### 9.5 过滤

第一版建议 PCF/PCSS-compatible depth pages；后续可加入 EVSM moments，但需谨慎处理漏光。过滤半径影响所需邻页，页请求阶段必须扩张 footprint。缺邻页时逐级回退，不能采样未映射内存。

### 9.6 与几何系统协同

- 阴影视图复用 GPU Scene 与 Cluster hierarchy。
- 阴影 LOD 使用 light-space projected error，可比主视图更粗。
- 页级 caster list 只保存 SceneHandle/Cluster range，不复制几何。
- 虚拟几何页缺失时使用祖先 Cluster，保证阴影不出现洞。

### 9.7 UE 同等级阴影要求

- 近景接触阴影锐利且稳定，远景按光源角尺寸自然变软；
- 方向光覆盖大世界时没有明显 cascade 边界、游泳和分辨率断层；
- 点光、聚光、面积光和发光几何的阴影策略一致且可诊断；
- 植被、masked material、毛发、薄片和双面材质具有正确透射/遮蔽语义；
- 相机静止时缓存结果完全稳定，物体移动只更新相关区域；
- 缺页时使用父页或低分辨率结果，禁止突然无阴影；
- contact shadow 只能补充微小接触细节，不能用于掩盖主阴影系统缺失。

## 10. 动态 GI 与反射

目标不是照搬 Lumen 的内部实现，而是提供相同问题域的多后端解决方案。

### 10.1 统一接口，多个追踪后端

```rust
pub enum TraceBackend {
    ScreenSpace,
    SoftwareBvh,
    HardwareRayQuery,
}
```

每个像素/探针可组合后端：

1. 屏幕空间命中优先，成本最低且细节高。
2. 屏幕外或遮挡失败进入软件 BVH/SDF/卡片表示。
3. 高端档位使用硬件 Ray Query 提升精度，尤其镜面反射和薄几何。
4. 天空/环境图作为最终 miss。

不能让整个 GI 功能依赖 `EXPERIMENTAL_RAY_QUERY`；否则无法实现真正跨平台。

### 10.2 场景追踪表示

建议分层：

- **Near field**：Cluster BVH 或硬件 BLAS，保留几何细节。
- **Far field**：简化代理、card/surface cache 或稀疏距离场。
- **Emissive/light data**：统一 LightTable 与 emissive triangle alias table。
- **Material hit data**：从 MaterialRecord 获取简化 BRDF/opacity/emission。

软件 BVH 可以复用虚拟几何 hierarchy 和驻留页，但必须有稳定低精代理，以免未驻留页造成漏光。

### 10.3 直接光

Solari 已有 ReSTIR 实验基础。演进方向：

- 统一 analytical light、emissive mesh 和环境光的 sample descriptor。
- 初始候选、时间复用、空间复用分别可开关。
- reservoir 带 scene/light epoch；光源或材质重大变化时拒绝旧样本。
- visibility 可按质量档位延迟或减少射线。
- 小光源数量少时自动选择 clustered exact evaluation，避免 ReSTIR 固定开销。

### 10.4 间接光与辐照缓存

建议从 Solari world cache 演进为 camera-relative sparse radiance cache：

- 多级空间 LOD；
- cell key 与世界原点重定位兼容；
- cell 保存方向性辐射而非单标量；
- 新 cell、高方差 cell、光照变化区域优先更新；
- 对历史样本做法线、深度、材质和 scene epoch 验证；
- 设定每帧软目标与硬上限；
- 可见像素从 cache 插值，必要时发出补充射线。

为了减少漏光，应使用表面感知 key（位置 + 法线/表面 ID）或对 thin wall 场景使用更严格权重。

### 10.5 反射

- 粗糙反射优先查询 radiance cache，减少射线数。
- 低粗糙镜面使用屏幕空间 + 硬件/软件 ray hit。
- 命中后按材质重要性决定完整材质求值或简化 closure。
- 透明/多层介质作为独立高级档位，不阻塞第一版。
- 输出 confidence/hit distance，交给统一 denoiser/compositor。

### 10.6 降噪与时域稳定

不能把 DLSS Ray Reconstruction 当作唯一方案。需要引擎自带 vendor-neutral 路径：

- motion vector、depth、normal、roughness、material/instance ID；
- disocclusion detection；
- temporal accumulation with variance clipping；
- a-trous/wavelet 或 recurrent-free spatial filter；
- specular 与 diffuse 分离历史；
- camera cut 和大规模 scene epoch reset；
- 可选接入 DLSS-RR 等平台插件。

### 10.7 UE 同等级光照与反射要求

高端画质档应同时具备以下结果，而不是只实现“有 GI”：

- 室内多次反弹能够稳定抬升暗部，门窗和细缝不过度漏光；
- 天空光、方向光、局部光、发光材质和环境图参与一致的直接/间接照明；
- 动态灯光、开门、移动遮挡物和发光物变化能在有限延迟内传播，不长期保留旧光照；
- 漫反射 GI 在相机运动时无明显低频泵动，薄墙和角落不过度糊成一片；
- 镜面反射覆盖屏幕外信息，并在粗糙度提高时平滑过渡到辐照/反射缓存；
- 反射中保持重要材质、阴影、天空、雾和发光信息，不出现与主视图完全不同的世界；
- 透明、水体和多层介质至少有明确的高质量专用路径及降级说明；
- 高端硬件允许提高反弹数、射线预算、cache resolution 和更新率，不被兼容档上限束缚。

画质优先模式不得通过过度时域累积掩盖采样不足。静止截图、匀速移动、快速旋转、突然显露和动态光变化必须分别验收。

## 11. Render Graph 2.0

现有 Render Graph/调度可演进为“声明式帧图 + 持久 GPU Scene”。

### 11.1 逻辑资源

Pass 引用逻辑资源 ID，而非直接持有临时 Texture：

```rust
let depth = graph.create_texture("main_depth", depth_desc);
graph.add_pass("visibility", |pass| {
    pass.read(gpu_scene);
    pass.write(depth);
    pass.write(visibility);
    pass.queue(QueueClass::Graphics);
});
```

构建器可执行：

- 生命周期分析与 transient aliasing；
- barrier/transition 推导；
- queue ownership transfer；
- dead pass elimination；
- capability-based pass replacement；
- 资源格式兼容验证；
- debug capture 中输出最终 DAG。

### 11.2 推荐帧图

```text
SceneUpload/Scatter
  ├─ VirtualResourceFeedbackResolve (上一帧)
  ├─ GPU Scene bounds/refit
  └─ BLAS/TLAS incremental update
        ↓
InstanceCull → ClusterCull/LOD → Visibility/Depth → HZB
                                      │              │
                                      │              └→ late occlusion pass
                                      ├→ Shadow page requests → Shadow page renders
                                      ├→ Material classification → PBR/NPR shading
                                      ├→ GI/Reflection trace + cache update
                                      └→ Motion/depth/ID auxiliaries
                                                        ↓
Transparency/Volume → Denoise → Post Process → UI → Present
```

### 11.3 异步计算

适合 async compute 的候选：

- 上一帧 virtual feedback resolve；
- 部分 culling 与 indirect command 构建；
- radiance cache update；
- denoise；
- 非关键 BLAS compaction。

是否异步必须由平台 profile 和实测决定。移动端 unified queue 或带宽受限设备可能因并发更慢。

### 11.4 Render Feature 扩展点

稳定扩展点只放在语义边界：

- `AfterSceneUpload`
- `BeforeVisibility`
- `AfterVisibility`
- `BeforeOpaqueShading`
- `AfterOpaqueShading`
- `BeforeTransparency`
- `BeforeTonemap`
- `AfterTonemap`
- `OfflineIntegrator`

扩展点并非固定顺序字符串，而是预注册 graph slot/资源契约。内部可重排，只要契约不变。

## 12. 离线渲染预留

“预留离线渲染”不能只保留一个 TODO。实时架构从第一天就必须保证场景和材质语义可被高质量积分器读取。

### 12.1 共享与隔离

共享：

- SceneHandle、实例变换、几何、材质参数、灯光、相机；
- BSDF 参数定义、纹理色彩空间、单位；
- 虚拟资源读取接口；
- AOV 中的 object/material ID。

隔离：

- 实时 GBuffer/visibility 格式；
- 时域历史和实时降噪假设；
- 实时近似 BRDF 的具体代码；
- 实时帧预算和离线 sample scheduler。

### 12.2 OfflineClosure

每个实时 shading model 应明确映射到离线 closure：

```rust
pub trait OfflineMaterial {
    fn closures(&self, hit: &SurfaceHit, out: &mut ClosureSet);
    fn emission(&self, hit: &SurfaceHit) -> Spectrum;
    fn opacity(&self, hit: &SurfaceHit) -> f32;
}
```

无法映射的 custom shader 必须：

- 提供 offline closure；或
- 使用 bake/export hook；或
- 明确标记为离线不支持，并输出诊断材质；

绝不能静默渲染成错误材质。

### 12.3 生产级路径追踪路线

从当前验证 pathtracer 逐步加入：

- next-event estimation + MIS；
- emissive triangle/light tree sampling；
- Russian roulette；
- spectral-ready 接口（首版可 RGB 实现）；
- transparent/volume；
- adaptive sampling；
- tile/checkpoint/resume；
- deterministic seed；
- AOV：beauty、albedo、normal、depth、motion、direct、indirect、emission、object/material ID；
- EXR/高动态范围输出；
- headless batch API；
- CPU reference backend 可选，用于 GPU 差异定位。

## 13. 跨平台自动适配

### 13.1 能力档位

建议不是“高/中/低”一个维度，而是独立 capability axes：

```rust
pub struct RenderCapabilities {
    pub binding_model: BindingTier,
    pub geometry: GeometryTier,
    pub ray_tracing: RayTracingTier,
    pub shadows: ShadowTier,
    pub gi: GiTier,
    pub async_compute: AsyncComputeTier,
    pub memory: MemoryClass,
}
```

推荐预设：

| Profile | 几何 | 阴影 | GI/反射 | 材质绑定 | 典型平台 |
|---|---|---|---|---|---|
| Ultra RT | 虚拟几何 + Mesh/Compute | 虚拟阴影 | HW Ray Query + cache | bindless | 高端 Vulkan/DX12/Metal |
| High | 虚拟几何 Compute | 虚拟阴影 | Software BVH + screen | bindless/large arrays | 主流桌面/主机级 |
| Compatible | GPU-driven standard mesh | CSM/atlas | SSAO/SSGI/probes | bounded arrays | 老桌面、部分移动 |
| Mobile | LOD mesh + indirect optional | compact atlas | probes + screen | limited bindings | iOS/Android |
| Web | standard mesh | atlas | baked/probes/SSAO | WebGPU limits | 浏览器 |

注意：Metal 是否支持某 API 不代表所有 Apple GPU 都应启用最高档；仍需内存和性能评分。

### 13.2 选择流程

```text
静态能力查询
  → 驱动/设备 denylist 与 workaround
  → 启动时微基准（可缓存）
  → 显存/统一内存预算
  → 选择 profile
  → 用户配置覆盖
  → 运行时 frame budget controller 小范围调节
```

微基准只测试关键分歧：bindless access、indirect throughput、Mesh Shader 与 Compute 路径、Ray Query、async overlap。结果以 adapter/driver/app version 为 key 缓存。

### 13.3 运行时自适应

按 0.5–2 秒滑动窗口控制，不逐帧剧烈切换：

- 动态分辨率；
- 虚拟几何误差阈值；
- 阴影页预算与最大 mip；
- GI 射线数、cache updates；
- 反射 roughness cutoff；
- 后处理质量。

使用滞回和每次只改一个主要旋钮，便于归因。用户可以锁定质量，编辑器和截图模式可禁用动态调节。

### 13.4 能力报告

启动时生成结构化报告：

```text
Adapter: ...
Selected profile: High
Virtual geometry: Compute path (Mesh Shader slower in cached benchmark)
Virtual shadow: enabled, 2048 physical pages
GI: Software BVH + screen trace
Hardware RT: unavailable (missing ...)
Fallbacks: translucency uses forward; skinned meshes use standard mesh path
```

报告可通过 diagnostic API 和 UI 查看，避免“为什么功能没开”的猜测。

## 14. 性能与稳定性设计

### 14.1 帧预算

每个系统必须有软/硬预算：

| 类别 | 软预算行为 | 硬上限行为 |
|---|---|---|
| Scene upload | 延后低优先级更新 | 强制分批并报告 latency |
| Geometry pages | 提高 LOD error | 渲染驻留祖先 |
| Shadow pages | 降低远处分辨率 | 使用父页/低分辨率 fallback |
| GI updates | 降低更新率/射线数 | 仅复用有效历史或环境光 |
| Pipeline compile | 后台异步 | 使用 fallback pipeline |
| Transient memory | alias/reduce resolution | 禁用可选 Pass |

### 14.2 无界输入保护

- 所有 GPU append buffer 都必须有 capacity 和 overflow counter。
- 溢出不能越界写；使用 clamp + 下一帧扩容/降级。
- 页请求先 GPU 去重再 readback，readback 字节有上限。
- shader permutation 数量设定构建期阈值。
- bindless 表索引检查可在 debug/validation profile 打开。
- 资产声明尺寸与解压尺寸都要校验，防止异常资产耗尽内存。

### 14.3 帧间稳定

- LOD、页驱逐、动态分辨率全部使用滞回。
- TAA/GI history 依赖稳定 SceneHandle + primitive ID。
- shader/pipeline 未就绪时使用稳定 fallback，不随机缺材质。
- world origin rebasing 必须统一通知 GPU Scene、GI cache、shadow cache 和 RT AS。
- 设备丢失后可从 CPU asset/scene journal 重建，不依赖只存在于 GPU 的唯一数据。

### 14.4 线程与锁

- Change Journal 使用 per-thread chunk，帧末批量 merge。
- 资产构建、shader compile、page IO、解压分离任务池并限流。
- Render World 热路径避免全局 Mutex。
- GPU handle allocator 可批量分配 range，减少原子争用。
- 不在 render graph 执行期间修改 graph topology；使用下一帧事务提交。

### 14.5 显存治理

总预算按 adapter budget 动态计算，并保留安全余量：

```text
available_budget
  ├─ persistent scene tables
  ├─ virtual geometry pool
  ├─ texture pool
  ├─ shadow pool
  ├─ GI/cache pool
  ├─ RT acceleration structures
  ├─ transient graph heap
  └─ emergency reserve
```

预算仲裁器每秒评估一次压力；不要让每个子系统独立认为自己拥有全部显存。

## 15. 诊断、测试与基准

### 15.1 必备指标

统一暴露：

- GPU Scene：总槽位、活跃数、create/destroy/update 数、上传字节、scatter 数、延迟队列。
- Visibility：输入实例/Cluster、视锥剔除、遮挡剔除、LOD 接受、溢出。
- Geometry：resident/fault/evict pages、祖先 fallback、IO/解压字节。
- Shadow：requested/unique/allocated/rendered/reused pages、dirty 原因、缺页。
- GI：rays、hit backend 比例、reservoir reuse、cache cells、updates、variance、history rejection。
- RT：BLAS build/refit/compact、TLAS instances、AS memory。
- Material：可见 shading model、tile/dispatch 数、fallback pipeline、compile latency。
- Graph：每 Pass GPU/CPU 时间、barrier、transient peak、async overlap。

### 15.2 Debug View

- SceneHandle/instance ID/material ID；
- Cluster、LOD level、projected error；
- geometry residency 与 fault；
- shadow virtual page/mip/cache age/dirty；
- GI trace backend、hit distance、cache LOD、history confidence；
- overdraw、material complexity、bandwidth estimate；
- fallback 原因和 profile。

### 15.3 自动化测试

#### 单元测试

- 分代句柄复用与过期拒绝；
- change merge；
- page allocator/eviction；
- LOD 误差单调性；
- capability profile 决策；
- material ABI 编解码。

#### GPU 测试

- sparse scatter 正确性；
- culling 与 CPU reference 对比；
- visibility primitive reconstruction；
- virtual page table 翻译；
- shadow invalidation 覆盖；
- software/hardware trace 的容差比较。

#### 图像测试

- PBR 参数 sweep；
- NPR ramp/outline；
- 静态与运动阴影；
- GI disocclusion/camera cut；
- 不同 profile 的容差基线；
- realtime 与 offline reference 差异热图。

#### 压力测试

- 100 万实例，随机 0.1%/1%/10% 变化；
- 数万材质但少量屏幕可见；
- 快速穿越超大场景引发 streaming pressure；
- 大量移动 caster；
- 页池刻意缩小验证降级；
- append buffer 溢出和设备丢失恢复。

### 15.4 性能比较规范

若要声称优于 UE 或其他引擎，必须公开：

- 相同源资产、分辨率、画质目标、相机路径；
- 相同动态光和阴影范围；
- 冷缓存与热缓存；
- CPU/GPU 型号、驱动、操作系统、功耗模式；
- 平均、P50、P95、P99 帧时间；
- CPU render thread、GPU 各 Pass、显存峰值；
- 是否包含 shader/PSO compilation 和 streaming stutter；
- 图像差异或人工画质审查。

只比较 FPS 不足以支撑架构结论。

### 15.5 UE 画质对标协议

建立版本控制下的 `render_quality_suite`，所有场景同时保存 Prism 与 UE 的锁定配置。测试至少包括：

| 场景 | 主要检查项 |
|---|---|
| Cornell/材质球阵列 | BRDF 能量、粗糙度、金属、clearcoat、曝光 |
| 室内公寓 | 多反弹 GI、薄墙漏光、局部光、镜面反射 |
| 阳光室外城市 | 虚拟几何、远近阴影、天空/大气、快速移动 |
| 森林与草地 | masked foliage、双面透射、风动、阴影稳定、LOD |
| 数字人近景 | 皮肤、眼睛、毛发、微表面、景深、运动 |
| 夜景霓虹 | 发光材质、反射、曝光、bloom、噪声和 ghosting |
| 水体与玻璃 | 透明、折射、吸收、反射、介质排序 |
| 大世界穿越 | streaming、阴影/GI cache、时域稳定和卡顿 |
| 极端动态场景 | camera cut、灯光切换、物体显露、历史拒绝 |

每个场景输出：

- 固定曝光的 HDR 原始线性图；
- 最终显示变换后的 SDR/HDR 图；
- 静态图以及固定路径视频；
- depth、normal、albedo、roughness、motion、direct、indirect、reflection、shadow 等 AOV；
- 与 Prism 离线路径追踪 reference 的差异图；
- GPU 时间、显存峰值和 streaming 状态，防止通过未记录的超额预算换画质。

验收采用三层门槛：

1. **物理与数值测试**：白炉、能量守恒、色彩/曝光、AOV 和 reference error 必须通过各自阈值。
2. **时域测试**：固定视频路径分析闪烁、重影、拖尾、亮度方差、阴影跳变和 LOD popping；静态图通过但视频失败仍算失败。
3. **双盲主观测试**：渲染工程师、技术美术和普通观察者随机观看 A/B；若 UE 被稳定偏好，记录原因并阻止高端画质档发布。

建议初始发布门槛：所有 blocker 场景无严重缺陷；95% 的关键镜头在专家审查中达到“同等级或更好”；剩余镜头只能存在有登记、有负责人、有截止版本的非阻断差异。具体 SSIM/FLIP 阈值必须按场景和 AOV 标定，不能使用一个全局数值替代视觉审查。

### 15.6 画质优先级和缺陷等级

画质问题按发布影响分级：

- **Q0 阻断**：漏光、反射缺失、阴影大面积消失、曝光/色彩错误、严重重影、几何洞或持续闪烁。
- **Q1 高优先级**：可重复察觉的材质能量差异、局部噪声、边缘不稳、细节显著低于 UE。
- **Q2 一般**：需要专业观察或局部放大才能发现，且不影响创作意图。
- **Q3 改进项**：风格或默认参数差异，可由艺术控制修正。

高端档不允许存在 Q0；标准对标场景中 Q1 必须清零。兼容档可以降级，但 UI 和 capability report 必须明确说明关闭或降低的效果。

## 16. 分阶段实施路线

### Phase 0：基线与守门（2–4 周）

交付：

- 固定的渲染 benchmark 场景集；
- GPU/CPU/显存指标命名规范；
- capability dump；
- 锁定 UE 基准版本、参考项目、CVar/Scalability 配置、相机路径和资产导入设置；
- `render_quality_suite` 的第一批 HDR、视频、AOV 和双盲审查基线；
- 当前 Meshlet、标准 mesh、Solari、阴影的基线报告；
- CI 中的功能图像测试框架。

退出条件：后续改动能量化比较，性能回归可定位到 Pass，并能重复生成同配置的 Prism/UE 画质对照结果。

### Phase 1：GPU Scene MVP（6–10 周）

交付：

- SceneHandle/allocator/generation/epoch；
- transform、bounds、geometry、material、flags 的 SoA 表；
- change journal 合并与 sparse update；
- 标准 opaque mesh 使用 SceneHandle；
- debug view 和完整指标；
- 保留旧路径 feature flag 回退。

退出条件：百万实例小比例更新的 CPU/上传成本随变更数量增长；图像与旧路径一致。

### Phase 2：统一可见性与材质分类（8–12 周）

交付：

- 标准 mesh 与 Meshlet 共用实例 culling 输入；
- visibility buffer contract；
- opaque PBR compute/tiled shading 原型；
- MaterialRecord 和 shading model registry；
- ToonLit/NPR 示例；
- custom render feature API v0。

退出条件：PBR/NPR 共享可见性；大量材质场景不退化为大量 draw call。

当前进度（2026-09-23）：Material ABI 核心和 RenderApp 接入已经完成，包括版本化三表 ABI、`StandardMaterial` 自动映射、分代回收、稀疏上传、设备恢复、fallback slot、bind group、WESL generation 校验以及 GPU Scene material handle 自动关联。统一可见性的 RenderApp 基线也已接入：多视图稳定身份、camera cut/history epoch、frustum/layer/material/LOD 分类、稳定 work/range GPU buffer、FrameGraph 声明、设备恢复和 consumer API 已闭环。当前实现刻意标记为 CPU deterministic reference（诊断中的 `cpu_reference_frames`），尚未虚构 GPU compute 已完成；GPU compute culling/compaction/indirect、previous/current HZB early/late cull、visibility buffer 和 opaque PBR/NPR consumer 尚未达到本阶段退出条件，因此 Phase 2 仍为进行中。

Opaque bootstrap 已改为按统一 visibility work stream 入队，而不是把 Bevy `RenderVisibleEntities` 当作绘制真值；后者仅用于删除旧 phase item。shader 同时绑定 Material ABI，并在 generation 不匹配时解析到 fallback slot。当前 shaded 模式读取 ABI base color，仍不是完整 direct/indirect lighting、IBL、shadow、normal/texture sampling 或 NPR resolve。

### Phase 3：虚拟几何流送（10–16 周）

交付：

- 分页资产格式与离线 builder；
- virtual resource core；
- geometry feedback、residency、祖先 fallback；
- Compute 和 Mesh Shader 后端选择；
- 自动 import pipeline；
- 快速相机与预算压力测试。

退出条件：工作集超过页池时稳定降级，无洞、无越界、无持续 thrash。

### Phase 4：虚拟阴影（10–16 周）

交付：

- 方向光 clipmap；
- spot/point 虚拟地址；
- receiver-driven request；
- page caster list；
- 局部失效与静态缓存；
- CSM/atlas fallback。

退出条件：静止场景阴影页重绘接近零；移动 caster 只更新局部；缺页平滑回退。

### Phase 5：多后端 GI/反射（持续 16+ 周）

交付顺序：

1. 将 Solari 改为消费统一 GPU Scene/MaterialRecord。
2. vendor-neutral denoiser。
3. screen + software trace backend。
4. radiance cache 稳定性和 invalidation。
5. HW Ray Query 增强与自动选择。
6. emissive sampling/ReSTIR 成熟化。

退出条件：无硬件 RT 设备具备可用动态 GI 档位；高端设备可自动增强；camera cut/动态物体稳定。

### Phase 6：离线渲染产品化（可与 Phase 5 并行）

交付：

- OfflineClosure；
- MIS/path tracing；
- AOV/EXR/headless；
- checkpoint/determinism；
- 实时/离线材质一致性测试。

退出条件：离线图可作为 PBR/GI 参考真值，也可用于可重复批量输出。

## 17. 第一个可合并切片

不要从“新建所有 crate”开始。建议第一个 PR 只做以下闭环：

1. 在 `bevy_render` 内新增实验性 `gpu_scene` module。
2. 实现 `SceneHandle`、分代 allocator、延迟回收。
3. 实现 transform/bounds 的 `SparseBufferVec` 表。
4. 从 ECS Changed query 生成合并 journal。
5. 让标准 3D opaque mesh 的 GPU preprocessing 输入引用 scene index。
6. 增加统计：active instances、updated transforms、uploaded bytes、stale handles。
7. 增加句柄生命周期单元测试和 10 万实例/1% 移动 benchmark。
8. 通过 feature flag 保留当前输入路径，便于 A/B。

这个切片能验证最关键的假设：稳定 GPU 索引和增量更新是否能无破坏地接入现有 batching。它不依赖虚拟阴影或 GI 完成。

### 17.1 当前 `pkg/` 落地状态（2026-09-22）

第一步已按“不修改 Bevy 源码”的约束落在两个独立 package：

- `prism_render_architecture::gpu_scene`：分代句柄、GPU completion 延迟回收、原子事务、CPU authoritative mirror、previous transform 和上传规划；
- `prism_render_scene`：opt-in ECS 提取、SoA GPU 表、共享 bind group、设备恢复、诊断、几何资产生命周期和 consumer API；
- `PrismGpuSceneOpaquePlugin`：独立标准 opaque consumer，使用 scene index + generation 直接读取 GPU Scene，在 Shader 中拒绝 inactive/stale handle，不修改 `bevy_pbr`；
- `GpuSceneMode::{Disabled, Enabled, Compare}`：Disabled 是实时 kill switch；Enabled/Compare 保留逐项 A/B 的入口。Compare 输出 scene/queue/parity 诊断，不在同一 color/depth target 重复绘制两条路径；像素 A/B 由独立 view/capture 工具完成。Opaque consumer 仍显式安装，避免在材质等价前静默替换 Bevy PBR。
- `benches/prism_gpu_scene`：固定 10 万实例、1% transform 更新的 Criterion 基准；2026-09-22 当前开发机首次 release 测量为 `596–775 µs`（中位估计 `682 µs`），用于后续回归对比，不作为跨机器绝对承诺。

当前 opaque consumer 是“底座闭环”而不是 UE 同画质材质系统：它证明 Mesh draw 已经真实消费 GPU Scene，而不是继续通过 `MeshInputUniform` 获取 transform。完整 StandardMaterial/PBR、阴影、motion vector、skinning/morph、masked/transparent 将作为后续 consumer 依次接入统一 Material ABI。

## 18. 关键决策记录（ADR 候选）

后续应为以下问题分别建立 ADR：

1. GPU Scene 使用 SoA 还是 AoSoA，以及 ABI 稳定范围。
2. SceneHandle 位宽、generation 和跨帧回收策略。
3. Visibility buffer 的 ID/barycentric 编码。
4. 虚拟页大小和多资源共享 allocator 的边界。
5. 虚拟几何使用树还是 DAG，以及资产兼容策略。
6. MaterialRecord 的 bindless/limited-binding 双布局。
7. 自定义 shading model 是动态分发还是 pipeline 分类。
8. 软件 GI 追踪表示选择：Cluster BVH、SDF、cards 或混合。
9. RT acceleration structure 是否只保存当前驻留几何。
10. Render Graph transient allocator 与多 queue ownership 模型。
11. 离线 closure 的语言/IR 和自定义 shader 映射策略。

每个 ADR 必须包含：目标、备选、数据、平台差异、迁移成本、回退和复审日期。

## 19. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| GPU Scene 一次迁移过大 | 长期分支、难合并 | 按标准 opaque → Meshlet → Shadow → Solari 逐消费者迁移 |
| Bindless 平台差异 | Web/移动无法共用布局 | MaterialHandle 语义统一，物理绑定布局按 profile 实现 |
| Meshlet 材质限制 | 高模不能使用常见材质 | visibility/material resolve 解耦；保留 masked/transparent fallback |
| GI 依赖实验 Ray Query | 覆盖面小 | 软件 BVH + screen 为基础，HWRT 为增强 |
| 虚拟资源互相抢预算 | 抖动、缺页 | 全局预算仲裁、客户端最低保障、工作集压力反馈 |
| 管线组合爆炸 | 卡顿、缓存膨胀 | shading model 分类、少量静态 permutation、异步 fallback |
| 时域算法重影 | 画面不稳定 | 稳定 ID、epoch、严格 history validation、debug confidence |
| 异步计算负收益 | 帧时间更差 | capability + 微基准决定，默认可关闭 |
| 宣称“超过 UE”缺乏证据 | 技术目标失真 | 以公开 benchmark/画质/稳定性标准验收 |
| 多年大重写 | 无中间产品价值 | 每个 Phase 可独立发布、旧路径共存且设删除门槛 |

## 20. 删除旧路径的门槛

旧路径只能在以下条件全部满足后删除：

- 新路径覆盖其支持平台或有明确替代路径；
- 主要示例和生态扩展已迁移；
- 性能在至少三类目标设备不回退，或回退有明确画质收益并经批准；
- 图像测试、压力测试、设备丢失恢复通过；
- 至少一个发布周期提供弃用警告；
- 文档包含迁移指南和能力差异。

## 21. 最终验收矩阵

| 能力 | 功能验收 | 性能验收 | 稳定性验收 | 回退验收 |
|---|---|---|---|---|
| GPU Scene | 多消费者共享稳定句柄 | 小比例更新近似 O(changes) | stale handle 不读错对象 | CPU/旧实例路径 |
| 虚拟几何 | 自动 LOD、无洞流送 | 高密度场景受益 | 超预算不崩溃/不闪洞 | fallback mesh |
| PBR/NPR | 同场景并存、自定义模型 | 材质数不线性增加 draw | fallback shader 可见诊断 | 标准 forward/deferred |
| 虚拟阴影 | 局部页请求和失效 | 静止场景高复用 | 缺页有父级结果 | CSM/atlas |
| GI/反射 | 屏幕/软件/硬件后端 | 达到档位帧预算 | camera cut 和动态场景稳定 | probes/SSGI/env |
| 离线渲染 | BSDF、AOV、批处理 | 可扩展到长任务 | checkpoint/确定性 | 诊断不支持材质 |
| 跨平台 | 自动 profile | 路径选择有实测依据 | denylist/设备丢失恢复 | 最低兼容 profile |

高端画质档还有一个覆盖所有行的总门槛：必须通过第 15.5 节的 UE 对标协议，且标准场景不存在 Q0/Q1 画质问题。任何子系统的性能达标都不能抵消该门槛失败。

## 22. 结论

Prism 当前最有价值的不是堆叠更多独立高级功能，而是把已有的 Meshlet、GPU preprocessing、SparseBuffer、Material 扩展和 Solari 收敛到统一的数据与资源架构。实施顺序必须是：

```text
可测基线
  → 统一 GPU Scene
  → 统一可见性与材质语义
  → 虚拟资源与几何流送
  → 虚拟阴影
  → 多后端动态 GI/反射
  → 生产级离线渲染
```

这样做的优势不是“功能名比 UE 更多”，而是数据只维护一次、变化只上传一次、可见性只求一次、缓存由一个预算系统治理，并且高端路径与兼容路径共享场景语义。在此基础上，高端档以“UE 5.x 同等级画质”为不可跳过的发布门槛，性能优化不得依靠可感知的画质退化。这才是可验证、可跨平台、可长期维护，也最有机会在同等画质下取得领先性能的架构基础。
