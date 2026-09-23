# Prism × Unreal Engine：Vulkan 内部移植总体方案

> 状态：总体实施方案（Draft）  
> 日期：2026-09-22  
> Prism 基线：当前工作区 `0.20.0-dev`  
> UE 源码位置：<https://github.com/EpicGames/UnrealEngine>（仅远程访问，不拉入 Prism 工作区）  
> 已核对远程基线：`release` 当前为 UE 5.8.2，commit `16d75d84714512edfb744e1fd0a59e9c74d57873`  
> 目标平台：现代桌面 Vulkan 1.3  
> 发布范围：内部自用  
> 质量目标：锁定 UE 基准版本后，在约定场景中达到 UE 同等级画质

## 1. 结论

整体方案采用“**Prism 主架构 + UE 算法实现移植 + Vulkan 专用后端**”，而不是把 Unreal Renderer、RDG、RHI、UObject 和 SceneProxy 整体搬入 Prism。

保留 Prism 的：

- ECS、Asset、Main World/Render World；
- Render Schedule/Render Graph；
- wgpu 基础资源与兼容渲染；
- Meshlet/BVH8/Visibility Buffer 基础；
- Solari BLAS/TLAS、Ray Query、ReSTIR 和 World Cache 基础；
- Rust 公共 API、生命周期与诊断系统。

选择性移植 UE 的：

- HLSL Shader 算法和公共 GPU primitives；
- TSR；
- Virtual Shadow Maps；
- Nanite 的分页、流送、Cluster 选择和光栅关键算法；
- Lumen 的 Screen Probe、Screen Trace、Surface/Radiance Cache、反射与降噪关键算法；
- Local Exposure、Bloom、DOF、Motion Blur、色彩与镜头效果中独立的计算内核；
- 经过授权且对 Vulkan 有价值的离线资产构建算法。

重写所有 Prism 集成层：场景提取、稳定句柄、资源生命周期、Render Graph Pass、Vulkan/WGPU 资源桥、材质 ABI、线程调度和公共 API。

预计投入：12–16 名熟悉 Vulkan/UE Renderer/Rust 的工程师，约 24–34 个月达到广泛场景 UE 同等级画质；6–10 个月形成第一批高端画质成果，14–20 个月达到内部 Alpha。

## 2. 前提与边界

### 2.1 授权门槛

实施前由授权负责人形成一页书面确认，至少覆盖：

- 允许把 UE 源码和 Shader 用于独立的内部引擎；
- 允许修改、翻译、编译和在内部目标机器部署；
- 允许由哪些员工、承包商和 CI 机器访问；
- 是否允许生成和保留 SPIR-V、符号、反射元数据及中间文件；
- 是否允许使用 UE 第三方目录中的组件；
- 内部构建物、日志、Crash dump 和 Shader 源码的保存要求；
- Prism 开源仓库、内部派生仓库与最终产品之间的隔离规则。

本方案假定以上均已获准，但不把“UE 仓库可访问”自动等同于每个第三方依赖都可移植。

### 2.2 仓库隔离

UE 源码不进入当前 Prism 工作区，也不加入 Prism Git history。建议三个逻辑区域：

```text
Prism public-compatible workspace
├─ crates/                    不含 UE 派生源代码
├─ docs/                      可公开的架构文档
└─ interfaces/                中性 ABI/trait

Prism licensed internal overlay
├─ ue_shader_adapter/
├─ ue_ported_algorithms/
├─ manifests/
└─ provenance/

Remote UE source
└─ github.com/EpicGames/UnrealEngine
```

当前会话不下载、不 clone、不 archive UE 仓库。若未来需要编译 UE 派生代码，应由内部 CI 在隔离、受控的工作区读取授权源，生成的产物通过 manifest 接入 Prism；不得复制到公开工作区。

### 2.3 目标 Vulkan 能力

最低档建议锁定：

- Vulkan 1.3；
- `VK_KHR_dynamic_rendering`；
- `VK_KHR_synchronization2`；
- `VK_KHR_timeline_semaphore`；
- descriptor indexing；
- buffer device address；
- draw indirect count；
- shader subgroup；
- 16-bit storage/arithmetic（按 Shader 需求）；
- scalar block layout。

高端档增加：

- `VK_KHR_acceleration_structure`；
- `VK_KHR_ray_query`；
- `VK_EXT_mesh_shader`；
- 可选 sparse residency；
- NVIDIA/AMD 厂商扩展只可存在于 adapter 层。

首批认证硬件建议：NVIDIA RTX 30/40/50 与 AMD RDNA2/3/4。Intel、移动 Vulkan 和旧 GPU 延后，不作为首版发布阻断项。

## 3. UE 远程源码访问方式

### 3.1 本次可见性

已通过用户登录的 GitHub 会话确认私有仓库可访问，且全程只使用 GitHub Web 页面阅读，没有 clone、ZIP 下载、`git fetch` 或其他本地落地操作。当前 `release` 分支顶部为 **UE 5.8.2**，完整 commit 为 `16d75d84714512edfb744e1fd0a59e9c74d57873`。本文后续路径核对以该 commit 为远程调查基线；实际实施应使用 commit URL，而不是可移动的 `release` URL。

已验证下列核心区域确实存在：

- `Engine/Source/Runtime/Renderer/Private/GPUScene.cpp`、`GPUScene.h`；
- `Engine/Source/Runtime/Renderer/Private/InstanceCulling/`；
- `Engine/Source/Runtime/Renderer/Private/Nanite/`；
- `Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/`；
- `Engine/Source/Runtime/Renderer/Private/Lumen/`；
- `Engine/Source/Runtime/Renderer/Private/PostProcess/`；
- `Engine/Source/Runtime/Renderer/Private/RayTracing/`；
- `Engine/Source/Runtime/Renderer/Private/Substrate/`；
- `Engine/Source/Runtime/Renderer/Private/MegaLights/`；
- `Engine/Source/Runtime/Renderer/Private/PathTracing.cpp`；
- `Engine/Source/Runtime/Renderer/Private/ScreenSpaceDenoise.cpp`；
- `Engine/Source/Runtime/Renderer/Private/GlobalDistanceField.cpp`；
- `Engine/Source/Runtime/VulkanRHI/`；
- `Engine/Shaders/Private/Lumen/`、`Nanite/`、`VirtualShadowMaps/`、`TemporalSuperResolution/`、`PathTracing/`。

这也说明最初总体方向成立，但 UE 5.8.2 还新增/强化了 `MegaLights`、`MaterialCache`、`StochasticLighting`、`Substrate`、`StateStream` 等相关模块。若最终要求“UE 5.8.2 全面同等级画质”，这些模块必须进入第二轮依赖调查，不能只考察传统的 Nanite/Lumen/VSM/TSR 四项。

### 3.2 不落地源码的分析流程

在具备浏览权限后，使用 GitHub Web UI 完成：

1. 锁定 UE commit SHA/tag，不跟随移动分支。
2. 使用仓库代码搜索定位模块、Shader entry 和依赖。
3. 为目标文件记录 URL、commit、blob SHA、许可分类、输入输出、依赖和移植方式。
4. 只在内部 overlay 中保存必要的派生实现；Prism 仓库仅保存中性接口和 manifest ID。
5. 每个移植 PR 都带 provenance record，禁止“记不清来源”的代码进入。

建议 manifest：

```toml
id = "tsr.history_update"
ue_commit = "<locked-sha>"
source_urls = ["https://github.com/EpicGames/UnrealEngine/blob/<sha>/..."]
source_blobs = ["<blob-sha>"]
license_scope = "internal-authorized"
port_mode = "shader-port" # reference | translated | adapted
prism_owner = "temporal-team"
validation_scene = "quality/temporal/thin_geometry"
```

## 4. 总体架构

```text
Bevy ECS / Assets / Editor
            │
            ▼
Prism Scene Bridge ─── Change Journal / Stable SceneHandle
            │
            ▼
Unified GPU Scene
   ├─ Instance/Transform/Bounds
   ├─ Geometry/Cluster/Page
   ├─ Material/Texture/Sampler
   ├─ Light/Decal/Volume
   └─ Epoch/History Identity
            │
            ├───────────────┬──────────────────┐
            ▼               ▼                  ▼
    GPU Visibility    Virtual Resource     RT Scene
    Cluster/LOD/HZB   Geometry/Shadow/     BLAS/TLAS
    Visibility IDs    Surface Cache Pages  Ray Query
            │               │                  │
            └───────────────┴────────┬─────────┘
                                     ▼
                       Prism Declarative Render Graph
                                     │
             ┌───────────────────────┼──────────────────────┐
             ▼                       ▼                      ▼
     Prism Native Passes     UE-derived GPU kernels    Custom/NPR Passes
     WESL/WGSL/SPIR-V        HLSL → DXC → SPIR-V       WESL/HLSL/SPIR-V
             │                       │                      │
             └───────────────────────┴──────────────────────┘
                                     ▼
                  Vulkan Advanced Device/Resource Layer
                                     ▼
                         Vulkan 1.3 Driver / GPU
```

原则：UE-derived kernel 永远不能直接读取 ECS 或控制 Render World；它只能读取 Prism 定义的稳定 GPU ABI。Prism 负责资源和生命周期，UE 算法负责指定 Pass 内的计算。

## 5. 移植与重写矩阵

| UE 子系统 | 处理方式 | Prism 接入点 | 理由 |
|---|---|---|---|
| RHI | 不移植 | `bevy_render::renderer` / Vulkan adapter | 与 UE 平台抽象和对象生命周期高度耦合 |
| RDG | 不移植，实现等价能力 | Prism Render Graph 2.0 | 保持 ECS/Render Schedule 原生调度 |
| SceneProxy/PrimitiveSceneInfo | 不移植 | Unified GPU Scene | 避免在 Prism 内重建 UObject 世界 |
| GPU Scene | 参考布局和算法，CPU 集成重写 | `bevy_render::gpu_scene` | 数据思想可复用，更新来源完全不同 |
| Shader Compiler/USH | 实现必要兼容子集 | `bevy_shader` licensed HLSL frontend | 可最大化保留 Shader 算法，避免模拟全部 UE 工具链 |
| GPU primitives | 选择性移植 | `bevy_render::gpu_algorithms` | Scan/sort/compact 等边界清楚、复用高 |
| TSR | Shader/常量/流程选择性移植，集成重写 | `bevy_anti_alias::temporal_upscale` | 独立且对动态画质收益高 |
| VSM | GPU 内核和策略选择性移植，场景/页资源重写 | `bevy_virtual_shadow` | 强依赖 GPU Scene，但算法边界可定义 |
| Nanite | 算法融合，不整体替换现有 Meshlet | `bevy_pbr::meshlet` → virtual geometry | Prism 已有 BVH8/LOD/visibility 基础 |
| Lumen | 分阶段选择性移植 | Solari → `bevy_gi` | 依赖面最大，最后接入最安全 |
| Path Tracer | 参考/选择性移植积分器 | `bevy_offline_render` | 应共享 Prism material closure |
| Post Process | 独立 Pass 选择性移植 | `bevy_post_process` | 输入输出明确，适合早期带来观感提升 |
| Editor/Cooker/DDC | 不整体移植 | Prism Asset Pipeline | 数据格式和工具环境不同 |

## 6. UE 源码调查清单

按锁定 commit 调查以下已验证区域：

```text
Engine/Source/Runtime/Renderer/
Engine/Source/Runtime/Renderer/Private/InstanceCulling/
Engine/Source/Runtime/Renderer/Private/MaterialCache/
Engine/Source/Runtime/Renderer/Private/MegaLights/
Engine/Source/Runtime/Renderer/Private/StochasticLighting/
Engine/Source/Runtime/Renderer/Private/Substrate/
Engine/Source/Runtime/RenderCore/
Engine/Source/Runtime/RHI/
Engine/Source/Runtime/VulkanRHI/
Engine/Source/Developer/ShaderCompilerCommon/
Engine/Source/Developer/DerivedDataCache/
Engine/Shaders/Private/
Engine/Shaders/Private/TemporalSuperResolution/
Engine/Shaders/Private/VirtualShadowMaps/
Engine/Shaders/Private/Nanite/
Engine/Shaders/Private/Lumen/
Engine/Shaders/Private/PathTracing/
Engine/Shaders/Shared/
```

远程固定入口：

- [UE 5.8.2 固定 commit](https://github.com/EpicGames/UnrealEngine/tree/16d75d84714512edfb744e1fd0a59e9c74d57873)
- [Renderer/Private](https://github.com/EpicGames/UnrealEngine/tree/16d75d84714512edfb744e1fd0a59e9c74d57873/Engine/Source/Runtime/Renderer/Private)
- [Engine/Shaders/Private](https://github.com/EpicGames/UnrealEngine/tree/16d75d84714512edfb744e1fd0a59e9c74d57873/Engine/Shaders/Private)
- [VulkanRHI](https://github.com/EpicGames/UnrealEngine/tree/16d75d84714512edfb744e1fd0a59e9c74d57873/Engine/Source/Runtime/VulkanRHI)

每个系统先产出依赖图，而不是立即翻译文件：

```text
Public control/CVar
→ CPU setup / scene state
→ RDG pass sequence
→ shader entry points/permutations
→ parameter structs
→ persistent resources/history
→ transient resources
→ platform feature conditions
→ debug/visualization passes
```

调查交付物只记录移植所需的接口、行为、依赖和来源，不把整段远程源码复制进本仓库文档。

## 7. Vulkan 与 Shader 策略

### 7.1 两层后端

保持 wgpu 管理普通路径，在能力不够时使用 Vulkan advanced layer：

```text
RenderDevice API
├─ WgpuDevicePath
│  ├─ 普通 Buffer/Texture/Pipeline
│  ├─ 标准 Render/Compute Pass
│  └─ 兼容和调试路径
└─ VulkanAdvancedPath
   ├─ HLSL SPIR-V pipeline
   ├─ Ray Query/AS
   ├─ Mesh Shader
   ├─ Device Address
   ├─ Sparse resources（可选）
   └─ 细粒度同步/alias（确有需要时）
```

先尝试经 wgpu 创建 SPIR-V pipeline。当前仓库 `Shader::from_spirv` 和 `PipelineCache` 已有 SPIR-V feature 路径，可作为入口；Raw Vulkan 仅用于 wgpu 未暴露或语义不足的能力。

### 7.2 资源所有权

禁止 wgpu 和 Raw Vulkan 在不知情的情况下同时操作资源。每个资源必须属于以下一种模式：

- `WgpuOwned`：只通过 wgpu 使用；
- `VulkanOwned`：只通过 advanced layer 使用；
- `Interop`：拥有显式状态机、queue family、layout、timeline value 和释放规则。

Interop 资源在图中声明 acquire/release pass；禁止在业务 Shader 模块中手写所有权转换。

### 7.3 HLSL 管线

建议实现离线/增量编译工具，而不是运行时完整模拟 UE ShaderCompileWorker：

```text
Licensed HLSL/USH subset
→ controlled preprocessor/include map
→ permutation manifest
→ DXC (Vulkan target environment)
→ SPIR-V validation/optimization
→ reflection extraction
→ Prism binding ABI validation
→ signed internal shader package
→ PipelineCache
```

必须确定并自动验证：

- DX layout/scalar layout；
- row-major/column-major 矩阵约定；
- bool、half、结构体 padding；
- register/space 到 set/binding 映射；
- push constants 与 uniform buffer；
- bindless descriptor arrays；
- wave/subgroup 大小假设；
- texture gather、atomic、derivative 语义；
- reversed-Z、clip-space Y 和 depth range；
- specialization/permutation hash。

### 7.4 不实现“完整 UE HLSL 环境”

每次只支持目标模块实际需要的 include、宏和 parameter metadata。若某个 UE helper 会拖入庞大依赖，提取算法语义后用 Prism helper 重写。禁止形成一个无法升级、无法理解的半套 UE Shader 编译器。

## 8. Prism 基础改造

### 8.1 GPU Scene（第一优先级）

修改：

- `crates/bevy_extract/src/sync_world.rs`
- `crates/bevy_render/src/lib.rs`
- `crates/bevy_render/src/batching/gpu_preprocessing.rs`
- `crates/bevy_render/src/render_resource/sparse_buffer_vec.rs`
- `crates/bevy_pbr/src/meshlet/instance_manager.rs`
- `crates/bevy_solari/src/scene/extract.rs`

新增内部稳定接口：

```text
crates/bevy_render/src/gpu_scene/
├─ handle.rs
├─ allocator.rs
├─ journal.rs
├─ instance_table.rs
├─ geometry_table.rs
├─ material_table.rs
├─ light_table.rs
├─ upload.rs
├─ snapshot.rs
└─ diagnostics.rs
```

GPU Scene 的 ABI 由 Prism 定义，不复制 UE 对象布局。UE Shader 的输入通过 adapter 转换或在移植时改为 Prism ABI。

### 8.2 Render Graph 2.0

需要支持：

- 逻辑资源读写声明；
- transient resource lifetime/alias；
- graphics/compute/copy queue；
- barrier 与 layout 推导；
- persistent history resource；
- pass permutation/capability predicate；
- dead pass elimination；
- GPU event/timestamp；
- graph dump；
- WgpuOwned/VulkanOwned/Interop 资源状态。

UE RDG pass sequence 被翻译为 Prism graph pass，不保留 UE RDG builder API。

### 8.3 Material ABI

建立统一 `MaterialRecord`，供标准 raster、Meshlet、VSM、Lumen/Solari 和 Path Tracer 使用。`StandardMaterial` 只是用户资产格式，不再是所有内部模块的硬依赖。

当前落地状态（2026-09-23）：

- `pkg/prism_render_material` 已提供版本化 CPU/GPU ABI、分代句柄、revision/epoch、固定纹理语义表、PBR/NPR/custom closure IR、`StandardMaterial` lowering 和稀疏 dirty row；
- `pkg/prism_render_scene::material` 已把资产事件接入 RenderApp，完成增量 sparse scatter upload、GPU completion 后回收、设备恢复重建、只读 consumer API、三表 bind group 和 shader 侧 generation 校验契约；
- slot 0 是永久 fallback material。异步加载时实例使用该槽；删除或 stale generation 由消费 shader 校验后解析到该槽，不读已回收材质；
- `MeshMaterial3d<StandardMaterial>` 会自动解析为 GPU Scene 的 material handle；`PrismGpuSceneEntity::material` 保留为显式 override，不再要求普通使用者手工分配句柄；
- GPU Scene 与 Material Registry 仍保持两个所有权域：GPU Scene 只存 `(index, generation)`，不复制资产对象或 Bevy/UE 材质布局；
- 本阶段不修改 `crates/` 下 Bevy 源码。后续 visibility、opaque、VSM、GI、ray/offline consumer 只依赖公开 ABI/bindings。

这里的“完成 Material ABI”指身份、数据、上传、恢复与消费契约闭环，不等于 UE 同画质已经完成。当前 opaque consumer 仍需改为消费统一 visible work，并用该 ABI 执行完整 PBR/NPR resolve；纹理 descriptor residency、virtual texture 和离线高精度 closure 仍属于后续 consumer/virtual-resource 工作。

统一可见性 RenderApp 基线已经落在 `pkg/prism_render_scene::visibility`：它从 GPU Scene、Material ABI 和 Bevy 多视图状态构建共享 work stream，维护 stable view handle、camera cut/history epoch、LOD hysteresis、overflow-safe capacity、诊断、GPU work/range buffer 和 FrameGraph 资源访问声明。第一条真实 GPU compute dispatch 已接线：通过 `PendingCommandBuffers` 进入统一提交，读取 GPU Scene/view ABI/Material header，完成 generation validation、render layer/frustum culling、fallback material resolution、projected-size LOD classification、pass classification、bounded atomic compaction，并写独立 parity work/counter/range/overflow，不与当前 CPU consumer buffer 形成写冲突。counter 与 work 使用异步 map readback，按视图和 scene identity 做无序集合 parity，记录 match/mismatch/drop/failure；它不同步等待 GPU，并支持关闭。此处 LOD 先冻结 work ABI，真实 geometry LOD/residency table 与 hysteresis 仍待接线。`gpu_compute_dispatches` 只在真实编码 dispatch 时递增；CPU deterministic reference 仍明确计入 `cpu_reference_frames` 并暂任消费者真值，直到 GPU LOD residency、sort/binning、indirect 与 HZB 完成后再切换，避免把半条 GPU 路径误称为完全 GPU-driven。

标准 opaque bootstrap 已从统一 work stream 取 opaque work，并绑定 Material ABI 三表；Bevy visibility list 只负责清理被移出的旧 phase item。该路径已验证 instance/material generation fallback 和 base-color 数据消费，但完整 PBR 光照、masked/NPR/custom 分类管线以及 GPU indirect draw 仍是下一层实现，不在文档中提前标为完成。

## 9. TSR 移植包

### 9.1 为什么先做

TSR 相对独立，能最早改善细线、植被、高光、远景和动态分辨率，是验证 HLSL→SPIR-V 和 Render Graph 映射的最佳首个 UE 子系统。

### 9.2 移植内容

- jitter/reprojection；
- velocity dilation；
- disocclusion/history rejection；
- history sample/accumulation；
- responsive/reactive mask；
- translucency/composition mask；
- exposure compensation；
- spatial reconstruction/resolve；
- sharpening；
- debug visualization。

### 9.3 Prism 接入

```text
crates/bevy_anti_alias/src/temporal_upscale/
├─ mod.rs
├─ settings.rs
├─ history.rs
├─ prepare.rs
├─ graph.rs
├─ reactive_mask.rs
├─ diagnostics.rs
└─ licensed_shader_bridge.rs
```

前置条件：所有 opaque/masked/skinned/meshlet 内容有正确 motion vectors；camera cut、FOV change、resolution change 和 exposure discontinuity 都能显式 reset/adjust history。

### 9.4 验收

- 静止亚像素细节；
- 平移、旋转、快速推进；
- 遮挡显露；
- 粒子和透明；
- 高亮 emissive；
- 50%–100% render scale；
- 与锁定 UE TSR 的固定视频 A/B。

## 10. Virtual Shadow Maps 移植包

### 10.1 移植内容

- virtual address/page table；
- directional clipmap；
- page request/mark/compact；
- physical page allocation/free list；
- page cache and invalidation；
- per-page caster command generation；
- page render；
- projection/sampling/filtering；
- cache/debug visualization。

### 10.2 重写内容

- Light ECS 提取；
- SceneHandle/caster bounds；
- 标准 Mesh 与 Meshlet 的 shadow command；
- Render Graph 资源；
- Vulkan image/buffer 生命周期；
- quality profile 和 fallback。

建议新增 `crates/bevy_virtual_shadow/`，传统 CSM/atlas 保留到新路径完全验收。

### 10.3 验收

- 太阳光大世界；
- 点光/聚光大量局部灯；
- 静止场景页重绘趋近零；
- 单个 caster 移动只局部失效；
- 植被/masked/hair；
- 页池压力和父页 fallback；
- 不出现越界、黑页、长时间缺阴影或 cascade-like 跳变。

## 11. Nanite 与现有 Meshlet 融合包

### 11.1 保留 Prism

- `MeshletMesh` 构建入口；
- 分组简化和误差传播；
- BVH8；
- Cluster culling；
- Visibility Buffer；
- Persistent GPU buffer 基础；
- 现有 PBR material resolve。

### 11.2 引入 UE 思路/实现

- hierarchy/cluster page layout；
- page dependency and root residency；
- GPU streaming feedback；
- request priority、dedup、readback；
- resident ancestor fallback；
- occlusion two-pass；
- persistent cluster culling；
- hardware/software raster 策略；
- material classification/resolve 可借鉴部分算法。

不复制 UE Nanite UObject/SceneProxy/Streaming Manager 外壳，而是在 `bevy_pbr::meshlet` 现有模块上演进。

### 11.3 资产转换

离线构建器必须版本化，并记录 UE-derived 算法版本。生成的内部资产不得混入公开 release artifacts，除非授权明确允许。

### 11.4 验收

- 数十亿源三角形场景的稳定工作集；
- 快速移动无洞；
- LOD 误差和时间稳定；
- geometry page fault/eviction 可控；
- masked foliage 后续接入；
- 小低模自动绕过虚拟几何固定开销。

## 12. Lumen 与 Solari 融合包

### 12.1 目标结构

不是同时保留完整 Lumen Scene 与 Solari Scene，而是形成统一 `bevy_gi`：

```text
Unified GPU Scene
├─ Screen Representation
├─ Software Trace Representation
├─ Hardware BLAS/TLAS (Solari)
├─ Surface Cache / Radiance Cache
└─ Unified Light/Material Sampling

bevy_gi
├─ screen_trace/
├─ screen_probe_gather/
├─ surface_cache/
├─ radiance_cache/
├─ reflections/
├─ hardware_rt/
├─ software_trace/
├─ restir/
└─ denoise/
```

### 12.2 分步移植

1. Solari 改为消费 GPU Scene/MaterialRecord。
2. 移植 Screen Trace 与 Scene Color/Depth hierarchy 接口。
3. 建立 Prism Surface Cache page system。
4. 移植 Screen Probe Gather 和时空滤波。
5. 复用 Solari Ray Query 作为 HWRT fallback/增强。
6. 增加 software trace representation。
7. 移植反射、radiance cache 和远场策略。
8. 将 ReSTIR、emissive sampling 和 direct lighting 统一。
9. vendor-neutral denoiser 为基线，DLSS-RR 仅作可选增强。

### 12.3 风险

Lumen 是依赖最广的模块。GPU Scene、Material ABI、TSR history、virtual resource system 未稳定前不得整体开工，否则会持续返工。

### 12.4 验收

- 室内多反弹；
- 薄墙和小开口漏光；
- 动态灯/门/物体；
- emissive lighting；
- rough/specular reflections；
- camera cut/fast motion；
- cache pressure；
- HWRT 开关前后语义一致；
- UE 固定视频 A/B 不存在 Q0/Q1 差距。

## 13. PBR、NPR 与专用材质

UE 同等级画质还需要材质和内容路径，不能只移植四个旗舰系统：

- Standard PBR、clearcoat、anisotropy、subsurface、transmission；
- skin/eye/hair/cloth/foliage/water；
- decal 和 layered material；
- specular AA/normal roughness filtering；
- virtual texture；
- NPR `ToonLit`、ramp、outline、hatching；
- offline closure。

UE Shader 可提供公式和高质量实现，但所有路径必须落到 Prism `MaterialRecord`，不得长期维持“UE Material 参数”和“Prism Material 参数”双体系。

## 14. 后处理与色彩

按依赖从低到高选择性移植：

1. Local Exposure；
2. Bloom；
3. Motion Blur；
4. Depth of Field；
5. Lens/flare/grain；
6. Display mapping/HDR output；
7. Volumetric composition integration。

Prism 保留 Camera、Exposure、Tonemapping 和 Post Process API；UE 算法在 adapter 中读取等价参数。建立锁定的线性工作空间、白点、曝光、pre-exposure、SDR/HDR 输出契约，否则 A/B 没有意义。

## 15. 离线渲染

Solari Pathtracer 演进为参考与生产离线渲染器：

- 统一 OfflineClosure；
- NEE/MIS；
- emissive triangle/light tree；
- Russian roulette；
- AOV/EXR；
- checkpoint/deterministic seed；
- headless/tile；
- transparent/volume/hair。

UE Path Tracer 中边界清楚的采样、MIS、材质和降噪算法可选择性移植，但场景和材质输入统一使用 Prism ABI。

## 16. 工作流与治理

### 16.1 Port Manifest

每个移植单元必须有：

- UE commit 和源 URL/blob；
- 授权分类；
- 原始模块 owner；
- Prism owner；
- port mode；
- 修改摘要；
- 依赖列表；
- 对应自动化测试；
- 更新 UE 版本时的复核状态。

### 16.2 代码标记

内部 overlay 中 UE-derived 文件使用统一 header 和 SPDX 风格内部标记；Prism 公共侧只能出现中性接口、算法无关测试输入和 manifest ID。CI 增加扫描：禁止授权目录文件、UE 特有标记和 manifest 标注的派生文件进入公开 artifact。

### 16.3 升级策略

不要持续追 UE main。锁定一个 UE release/commit 至少 12 个月：

- 第一次完整移植建立 baseline；
- 安全/驱动阻断修复按 patch cherry-pick；
- 下一次大版本升级先生成 source diff 和 shader permutation diff；
- 每个 port package 独立选择升级，禁止整树同步。

### 16.4 代码审查

每个移植 PR 至少需要：

- 一名 UE 子系统 owner；
- 一名 Prism 对应模块 owner；
- 一名 Vulkan/同步审查者（涉及 GPU 资源时）；
- 自动画质与 GPU validation 结果；
- provenance manifest 校验。

## 17. 分阶段里程碑

### M0：授权、基线、远程调查（第 0–2 月）

- 锁定 UE SHA 和目标 CVar；
- GitHub Web 远程访问打通；
- provenance/manifest/隔离 CI；
- UE/Prism 对标场景；
- Vulkan feature matrix；
- HLSL/SPIR-V 小型 proof-of-concept。

退出：不拉取 UE 仓库也能定位锁定版本文件、记录来源并编译一个授权 Shader 测试包。

### M1：基础底座（第 1–6 月）

- GPU Scene MVP；
- MaterialRecord v0；
- Render Graph resource state；
- HLSL/USH 必要子集；
- SPIR-V reflection/binding validation；
- Wgpu/Raw Vulkan ownership model；
- GPU primitives。

退出：标准 Mesh 与 Meshlet 引用稳定 SceneHandle；稀疏更新与百万实例 benchmark 通过。

### M2：TSR 与画面管线（第 4–10 月）

- motion vector 完整性；
- TSR 全流程；
- reactive/transparency masks；
- local exposure 与 selected post passes；
- HDR/tonemap 对标。

退出：固定动态视频中细节稳定性接近锁定 UE 基线，无主要 ghosting blocker。

### M3：虚拟阴影（第 6–14 月）

- VSM directional/local light；
- page cache/invalidation；
- standard mesh/meshlet caster；
- masked foliage；
- traditional shadow fallback。

退出：大世界和局部光场景达到同等级阴影质量，静止缓存稳定。

### M4：虚拟几何（第 6–16 月）

- paged asset；
- feedback/residency/streaming；
- resident ancestor；
- Vulkan mesh/compute/software raster 选择；
- material resolve 扩展。

退出：快速穿越和显存压力无洞、无持续 thrash，几何细节达到对标标准。

### M5：Lumen/Solari 融合 Alpha（第 10–24 月）

- unified RT scene；
- screen trace/probe gather；
- surface/radiance cache；
- HWRT；
- reflections；
- denoise/history validation。

退出：核心室内/室外/动态灯场景无 Q0，内部项目可使用。

### M6：内容覆盖与生产化（第 18–30 月）

- skin/eye/hair/foliage/water/cloud/volume；
- virtual texture；
- PSO/shader package；
- crash/device lost/memory pressure；
- asset build farm；
- diagnostics/editor views。

退出：内部 Alpha/Beta 项目连续使用，重大缺陷和卡顿可诊断。

### M7：UE 同等级发布门槛（第 24–34 月）

- 全部对标场景 Q0/Q1 清零；
- 专家/普通用户双盲 A/B；
- P95/P99 frame time；
- NVIDIA/AMD 多型号驱动矩阵；
- long-run stability；
- 画质相同时的性能优化。

## 18. 团队配置

推荐 14 人核心团队：

| 方向 | 人数 | 职责 |
|---|---:|---|
| 架构/GPU Scene/Render Graph | 3 | 稳定 ABI、资源、同步、调度 |
| Shader/Vulkan Toolchain | 2 | USH subset、DXC、SPIR-V、reflection、pipeline |
| TSR/Post/Color | 2 | 时域、曝光、最终成像 |
| Virtual Geometry | 2 | builder、streaming、culling/raster |
| Virtual Shadow | 2 | page/cache/invalidation/filtering |
| GI/Reflection | 3 | Solari/Lumen、cache、RT、denoise |

专用材质、工具、自动化和性能由各方向共同承担；若要成熟编辑器和资产生产体验，再增加 3–5 名工具/技术美术/QA 工程师。

## 19. 工作量与时间

| 交付 | 预计时间（12–16 人） |
|---|---:|
| HLSL→SPIR-V + GPU Scene 基础 | 4–6 个月 |
| TSR/色彩/首批高端画面 | 6–10 个月 |
| VSM + Virtual Geometry Alpha | 10–16 个月 |
| Lumen/Solari 融合 Alpha | 16–24 个月 |
| 真实项目生产可用 | 18–26 个月 |
| 广泛场景 UE 同等级 | 24–34 个月 |
| 完整工具与长期稳定 | 28–40 个月 |

最大不确定性不是 Shader 翻译，而是：UE 版本与权限可见性、Lumen 的依赖规模、wgpu/Raw Vulkan interop、动态内容的时域稳定和资产生产流程。

## 20. 依赖与并行关系

```text
授权/远程调查 ──→ HLSL-SPIR-V Toolchain ─────────────┐
        │                                             │
        └→ GPU Scene → Material ABI → Unified Visibility
                        │               │              │
                        │               ├→ TSR ────────┤
                        │               ├→ VSM ────────┤
                        │               └→ Nanite融合 ─┤
                        │                              │
                        └→ Solari Scene重构 ───────────┤
                                                       ▼
                                            Lumen/Solari融合
                                                       │
                              专用材质/体积/水体/云 ←──┤
                                                       ▼
                                            UE画质验收与优化
```

TSR、VSM 和 virtual geometry 可在 GPU Scene/Toolchain 稳定后并行。Lumen/Solari 必须等待 Material ABI、history identity 和 virtual resource core 至少达到 v0.8。

## 21. 测试与发布门槛

### 21.1 每个 Port Package

- CPU reference 或确定性输入输出测试；
- SPIR-V validation；
- descriptor/layout reflection 对比；
- Vulkan validation layer 无错误；
- NVIDIA/AMD Shader 结果比较；
- GPU buffer overflow/capacity 测试；
- Shader permutation coverage；
- 来源 manifest 校验。

### 21.2 系统验收

- HDR 原始帧、显示输出、AOV；
- 固定相机视频；
- 静态与动态场景；
- camera cut/disocclusion；
- streaming/memory pressure；
- P50/P95/P99；
- 显存峰值和 page thrashing；
- 离线 reference 差异；
- 锁定 UE 版本双盲 A/B。

高端档禁止用更模糊、更多重影、更短阴影距离、缺失反射或更低几何精度换取性能。标准对标场景 Q0/Q1 未清零时不得标记“UE 同等级”。

## 22. 主要风险与止损

| 风险 | 预警 | 止损策略 |
|---|---|---|
| 远程 UE 权限不稳定 | 404/人员离组无法访问 | 锁定访问名单、SHA、manifest；不启动依赖不明的移植 |
| 模拟完整 UE 框架 | adapter 代码持续增长 | 禁止移植 RHI/RDG/SceneProxy；只保留窄 ABI |
| wgpu/Raw Vulkan 同步错误 | validation/device lost | 资源单一 owner；Interop 显式 acquire/release |
| Shader ABI 不一致 | NVIDIA 可用、AMD 错图 | 自动 layout reflection + 双厂商 CI |
| Nanite 整体替换现有 Meshlet | 长期分支不可合并 | 以现有 Meshlet 为宿主逐功能引入 |
| Lumen 提前开工 | 每次底座变化都重写 | 强制依赖门槛和接口冻结 |
| UE 升级追赶失控 | 每月大量 port diff | 锁定 12 个月基线，package 独立升级 |
| 授权代码泄漏 | 出现在公开 artifact | overlay、访问控制、CI 内容扫描、产物白名单 |
| 只追静态截图 | 动态画面重影/闪烁 | 视频和时域指标为发布 blocker |

## 23. 前 90 天执行清单

### 第 1–2 周

- 指定 UE locked SHA、目标 Scalability/CVar 和 Vulkan GPU；
- 打通授权 GitHub Web 访问；
- 建立 port manifest、访问控制和内部 overlay；
- 冻结 8–10 个 UE/Prism 对标场景。

### 第 3–6 周

- 调查 TSR、VSM、Nanite、Lumen 的 pass/Shader/parameter 依赖图；
- DXC→SPIR-V 编译一个独立 Compute kernel；
- 验证 `bevy_shader` SPIR-V feature 与 PipelineCache 接入；
- 完成 HLSL/Rust layout golden test；
- GPU Scene SceneHandle/allocator/journal 原型。

### 第 7–10 周

- 标准 Mesh transform/bounds/material table 接入 GPU Scene；
- Render Graph 加入 persistent history 和资源状态原型；
- TSR 第一组 pass 在 Prism 跑通离线测试输入；
- 建立 NVIDIA/AMD Vulkan validation CI。

### 第 11–13 周

- Meshlet InstanceManager 改用 SceneHandle；
- TSR reprojection/history rejection 形成可视 Demo；
- VSM page table/addressing 离线单测；
- 完成一次从 source manifest 到内部 shader package 的可审计流水线。

90 天退出条件：不是“画面已追平 UE”，而是移植通道、隔离、GPU Scene 和第一个 UE-derived Shader package 都已验证，后续团队可以安全并行。

## 24. 需要尽快冻结的决策

1. UE 具体版本与 commit SHA。
2. 目标 GPU/驱动和是否要求 AMD 首发同步达标。
3. 是否允许内部 CI 临时 checkout UE；如果仍坚持完全不落地，只能人工 Web 阅读并手工维护派生实现，速度会明显下降。
4. UE-derived HLSL 是在内部 overlay 保存源文件，还是只向 Prism 构建发布签名 SPIR-V package。
5. wgpu 与 Raw Vulkan 的边界，特别是 AS、Mesh Shader、Sparse Resource。
6. Prism 公共仓库是否继续保持 MIT/Apache compatible clean tree。
7. UE 画质对标的具体 CVar、分辨率、TSR scale、Lumen/VSM/Nanite 设置。

## 25. 方案复审后的升级与优化

结合当前 Prism 代码和已确认的 UE 5.8.2 目录，原方案需要做以下升级。它们不是附加功能，而是减少返工、避免双后端失控并真正达到 UE 画质的结构性调整。

### 25.1 从“双层后端”升级为“单一资源所有者 + Vulkan 扩展接口”

原方案中的 `WgpuDevicePath + VulkanAdvancedPath` 如果各自创建和管理 Buffer、Image、Descriptor、Command Buffer，会形成两个资源管理器。当前 `raw_vulkan_init.rs` 只提供 Vulkan instance/device 创建回调和额外 feature 标记，并不是成熟的资源互操作层。直接扩展成两套后端会带来：

- image layout 和 queue ownership 无法可靠追踪；
- wgpu 内部 barrier 与 Raw Vulkan barrier 重复或缺失；
- 资源销毁和 frames-in-flight 保活不一致；
- validation error/device lost 难以归因；
- transient aliasing 几乎无法安全实现。

升级决策：**同一运行模式只能有一个资源与提交所有者**。

```text
推荐主模式：Vulkan-first
Prism Render API
  → Prism Vulkan Backend（唯一 owner）
     ├─ Buffer/Image/AS/Descriptor
     ├─ Command/Barrier/Queue/Timeline
     ├─ Pipeline/Shader package
     └─ Swapchain

兼容模式：Wgpu
Prism Render API
  → 现有 wgpu Backend（独立运行，不与 Vulkan 资源互操作）
```

如果初期必须继续使用 wgpu，则高级功能只能使用 wgpu 已安全暴露的 Vulkan 能力；不要在同一帧绕过 wgpu 操作其私有资源。仅在验证一个完整 buffer/image acquire-release 原型、通过 validation 和多帧销毁测试后，才允许极少量显式 interop。

这会增加约 2–4 个月底层工作，但能显著减少 VSM、Nanite、Lumen 阶段的同步返工。对于 Vulkan-only、长期追求顶级性能的目标，建议尽早确定 Vulkan-first，而不是逐步堆积 escape hatch。

### 25.2 保留 ECS 驱动，将当前 RenderGraph 明确定义为调度表，并新增真正的 Frame Graph

当前 `bevy_render::renderer::RenderGraph` 实际是一个 ECS `ScheduleLabel`，核心只有 `Begin/Render/Submit/Finish` 阶段；它不是 UE RDG 意义上的资源图。不能在现有名称下假定已经拥有：

- 逻辑资源 SSA/version；
- subresource read/write；
- 自动 barrier/layout；
- transient lifetime/aliasing；
- queue dependency；
- dead-pass elimination；
- persistent history import/export。

这不意味着 Frame Graph 不能使用 ECS，也不意味着要放弃 Bevy 的 Render World、System、Query、Resource 和 Plugin。两者解决的问题不同：

| 系统 | 负责内容 |
|---|---|
| ECS / Render Schedule | CPU System 顺序与并行、组件和资源借用、场景提取、插件组合、按场景决定本帧启用哪些 Pass |
| GPU Frame Graph | GPU Buffer/Texture 的 subresource 读写、Vulkan stage/access/layout、Barrier、Queue 依赖、Transient Alias、跨帧 History |

ECS 的 `Res`/`ResMut` 只能说明 CPU 侧 Rust Resource 的借用关系，不能表达一个 Texture 是作为 Depth Attachment 写入还是 Sampled Texture 读取，也不能表达 mip、layer、aspect、Shader stage、Image Layout 和 Vulkan Queue Ownership。因此 ECS 调度不能代替 GPU 资源依赖编译；反过来，Frame Graph 也不应代替 ECS 的场景和插件体系。

升级方案：由 ECS 驱动 Frame Graph。保留现有 schedule 作为帧图构建与执行容器，将 Builder、CompiledGraph 和 Diagnostics 暴露为 Render World Resource，由普通 ECS System 注册 Pass：

```text
Main World ECS
    ↓ Extract
Render World ECS
    ↓
Render Schedule
├─ Prepare               ECS Systems 准备场景、视图和 GPU Scene
├─ FrameGraphBuild       ECS Systems 向 Builder 注册本帧 Pass
├─ FrameGraphCompile     编译资源版本、依赖、Barrier、Alias 和 Queue batch
├─ FrameGraphExecute     可并行录制并执行 Command Buffer
├─ Submit
└─ Cleanup               回收单帧 DAG，保留声明为 persistent 的资源
```

典型插件仍使用 ECS System 构建图：

```rust
fn build_visibility_graph(
    gpu_scene: Res<GpuScene>,
    views: Query<&ExtractedView>,
    mut graph: ResMut<GpuFrameGraphBuilder>,
) {
    let depth = graph.create_texture("main_depth", depth_descriptor());
    let visibility = graph.create_texture("visibility", visibility_descriptor());

    graph.add_compute_pass("instance_cull", |pass| {
        pass.read_buffer(gpu_scene.instances(), BufferAccess::ShaderRead);
        pass.write_buffer("visible_instances", BufferAccess::ShaderWrite);
    });

    graph.add_graphics_pass("visibility", |pass| {
        pass.read_buffer("visible_instances", BufferAccess::IndirectRead);
        pass.write_depth(depth);
        pass.write_color(visibility);
    });
}
```

这里 ECS 决定有哪些视图、灯光、Feature 和 Pass；Frame Graph Compiler 负责 GPU 级正确性与优化。Pass 执行函数可以继续使用经过约束的 `SystemParam` 或预提取参数，但所有影响 Vulkan 同步的 GPU 访问必须在 Pass 声明中显式出现。

单帧 Pass 和逻辑资源原则上也可以表示为 ECS Entity，但不推荐作为默认实现：Frame Graph 是每帧创建和销毁的短生命周期 DAG，紧凑 Arena/Index 更适合拓扑排序、资源版本化和生命周期分析。建议边界为：

- 长生命周期场景、视图、Feature、配置和插件使用 ECS；
- 单帧 Pass DAG 使用紧凑 `GpuFrameGraphBuilder`；
- Builder、CompiledGraph、Diagnostics 作为 ECS Resource；
- ECS System 负责构建、编译触发和执行驱动。

为避免与现有 `RenderGraph` 名称冲突，新增类型暂命名为 `GpuFrameGraph`、`GpuFrameGraphBuilder` 和 `CompiledGpuFrameGraph`。现有 `RenderGraph` 保留为 ECS Schedule，迁移稳定后再决定是否重命名为 `RenderSchedule`，避免一次性破坏插件 API。

`FrameGraphCompile` 必须生成可打印的 pass DAG、资源版本、barrier、queue batch、transient heap 和峰值显存。TSR 只需 persistent history；VSM/Nanite 需要 persistent virtual pools；Lumen 还需要跨帧 cache。三类资源不能混用生命周期规则。

当前 `pkg/prism_render_architecture/frame_graph` 已完成第一版可执行编译内核：显式资源生命周期、RAW/WAR/WAW 依赖、确定性拓扑序、资源版本、跨队列 barrier 标记、queue batch/wait、按实际使用区间进行 transient alias，并包含非法引用、环、读后写和别名测试。它不取代 ECS；ECS 仍负责构图与驱动。Vulkan command recorder、image subresource/layout 和真实 semaphore/timeline 翻译属于 Vulkan backend 接入阶段。

在 Frame Graph v1 完成前，只允许移植独立 Compute kernel，不允许大规模翻译 UE RDG pass sequence。

### 25.3 Shader 方案升级为“构建期 Shader Package”，不在运行时兼容 UE

当前 Prism 已有 `shader_format_spirv`、`Shader::from_spirv` 和 `PipelineCache` 入口，这是可复用优势。但当前 SPIR-V 路径：

- 不解析 WESL import；
- `Shader::from_spirv` 默认关闭 shader validation；
- 没有 UE parameter struct/layout codegen；
- 没有完整 permutation manifest；
- 没有可持久化、可追溯的 Vulkan pipeline cache 包。

因此升级为离线构建包：

```text
授权 UE HLSL/USH（隔离构建环境）
→ 目标模块专用 preprocess adapter
→ DXC pinned version
→ SPIR-V Tools validate/optimize pinned version
→ reflection + ABI codegen
→ permutation pruning
→ content-addressed shader package
→ 签名 manifest
→ Prism 运行时只加载 SPIR-V + reflection metadata
```

运行时不读取 UE 源码、不模拟 ShaderCompileWorker、不动态解析任意 USH。开发热重载由隔离构建服务生成新 package。所有工具必须固定精确版本，package key 至少包含 UE SHA、源 blob、DXC/SPIR-V Tools 版本、defines、entry、target GPU tier 和 Prism ABI version。

### 25.4 新增 ABI codegen，禁止手写跨语言布局

UE Shader 移植最常见的隐性错误不是算法，而是结构体 padding、矩阵方向、bit field、descriptor space 和 bool/half 布局。应新增单一 schema：

```text
render_abi/*.schema
  ├─ 生成 Rust repr(C)/Pod 类型
  ├─ 生成 HLSL include
  ├─ 生成 WESL/WGSL struct
  ├─ 生成 SPIR-V reflection expectation
  └─ 生成 offset/size golden tests
```

GPU Scene、Material、View、Virtual Page、Reservoir、History、Ray Hit 都必须由 schema 生成。手写结构只能存在于私有 Shader 局部变量，不能跨 CPU/GPU 或跨 Shader package 边界。

### 25.5 在 VSM 和虚拟几何之前加入统一 Virtual Resource Core

原里程碑让 VSM 与 Virtual Geometry 较早并行，但两者都需要相同的 page allocator、feedback、dedup、readback、budget 和 eviction。如果分别实现，Lumen Surface Cache 到来时还会出现第三套。

应在 M1 后新增 **M1.5 Virtual Resource Core**：

```text
bevy_render_virtual/
├─ virtual_address.rs
├─ physical_pool.rs
├─ page_table.rs
├─ feedback.rs
├─ gpu_dedup.rs
├─ request_priority.rs
├─ residency.rs
├─ eviction.rs
├─ upload_scheduler.rs
├─ budget.rs
└─ diagnostics.rs
```

客户端仅实现页面内容和失效规则：

- Virtual Geometry：geometry/hierarchy page；
- VSM：depth page；
- Lumen：surface/radiance page；
- Virtual Texture：texture page。

共享机制不代表共享同一物理池；不同格式保留独立 pool，但共用调度、预算和 feedback 协议。

### 25.6 不直接演进现有 Meshlet 资产格式，建立 Virtual Geometry v1

现有 Meshlet 已有 BVH8、连续 LOD 和 visibility buffer，适合验证算法。但当前格式是整资产持久上传，材质受限，实例仍存在全量遍历 TODO。若在 `MeshletMesh` v3 上直接增加 UE 风格分页，容易背负兼容包袱。

优化方案：

- 冻结 `MeshletMesh` 为 bootstrap/兼容路径；
- 新建 `VirtualGeometryAsset v1`；
- 离线转换器可复用现有简化/BVH 算法，也可接入 UE-derived builder；
- 两种资产共享 SceneHandle、MaterialRecord 和 visibility resolve；
- 新格式稳定后再决定是否弃用旧格式。

这样能并行比较“现有 BVH8 builder”和“UE-derived hierarchy/page builder”，用画质、构建时间、压缩率和运行时 residency 数据决定，而不是提前绑定错误格式。

### 25.7 Lumen/Solari 从“融合”升级为“职责拆分后替换”

简单把两套算法融合容易保留重复的 scene、world cache、history 和 denoiser。更清晰的策略是：

```text
保留 Solari
├─ Vulkan Ray Query/BLAS/TLAS 基础
├─ reference/path tracing
├─ 已有 light sampling 中可复用部分
└─ 过渡期验证

由 UE-derived GI pipeline 逐步替换
├─ Screen Trace
├─ Screen Probe Gather
├─ Surface Cache
├─ Radiance Cache
├─ Reflections
└─ Denoise/history
```

最终只保留一个实时 GI pipeline。Solari 的 world cache、Lumen cache 与新 Prism cache 不允许三套长期并存。每完成一条等价路径就进行 A/B 并删除过渡实现，避免维护成本永久叠加。

### 25.8 MaterialRecord 升级为 Material IR/Closure 系统

UE 5.8.2 已有 `Substrate`、`MaterialCache`，说明仅靠固定 `shading_model_id + parameter block` 很难覆盖多层材质、Lumen hit lighting、Path Tracing 和自定义 NPR。

建议两层表示：

```text
Material Asset/Graph
→ Material IR（离线规范化、裁剪、常量折叠）
→ Runtime MaterialRecord
   ├─ fast-path fixed records（常见 PBR/NPR）
   ├─ closure bytecode/table（复杂 layered material）
   └─ ray/offline callable metadata
```

常见材质保持固定布局高性能路径；复杂 Substrate-like 材质才进入 closure 路径。这样既能达到内容上限，又不会让所有像素为通用材质解释器付费。

### 25.9 将 MegaLights 和 Stochastic Lighting 纳入正式路线

若基准是 UE 5.8.2，只有 Lumen/VSM 并不足以覆盖大量动态光源场景。已验证远程 Renderer 中存在 `MegaLights` 和 `StochasticLighting`。建议新增 M5.5：

- 统一 LightTable 和 emissive light sampling；
- light candidate generation；
- reservoir/time/space reuse；
- ray traced visibility；
- 与 VSM 的自动成本选择；
- 与 Lumen direct lighting 的去重；
- 对少光源场景自动绕过随机采样固定成本。

MegaLights 不应在第一年抢占 GPU Scene/TSR/VSM 资源，但必须在“UE 5.8.2 同等级”最终门槛前完成。

### 25.10 Pipeline Cache 升级为三层缓存

当前 `PipelineCache` 可异步创建 pipeline，但顶级 Vulkan 项目还需要：

1. **Shader Package Cache**：SPIR-V 与 reflection，按内容寻址。
2. **Pipeline Recipe Cache**：descriptor/layout/render state/permutation 的稳定 hash。
3. **Vulkan Driver Cache**：按 vendor/device/driver UUID 保存 VkPipelineCache 数据。

运行时缺失 pipeline 时使用明确 fallback，后台编译并记录 hitch；编辑器/构建农场收集实际使用 recipe，生成项目级预热包。任何一次超过帧预算的 pipeline compile 都进入 telemetry。

### 25.11 增加显存驻留与帧节奏控制器

顶级平均 FPS 不等于稳定性能。新增全局控制器：

- 获取 Vulkan budget/usage；
- 为 geometry、texture、VSM、Lumen、AS、transient 分配软硬预算；
- 统一 emergency reserve；
- page fault、upload、BLAS build、shader compile 有每帧工作上限；
- 以 P95/P99 frame time 而非只看平均值调节；
- 相机切换采用预热/暂时降采样，不允许单帧资源风暴；
- 所有 append/feedback buffer 有 overflow counter 和安全 fallback。

### 25.12 调整里程碑顺序

优化后的主路径：

```text
M0  远程源码调查、授权治理、UE质量基线
M1  Vulkan-first Backend + GPU Scene + ABI schema + Frame Graph v1
M1.5 Virtual Resource Core + Shader Package/Pipeline Cache
M2  TSR + Motion Vector + Exposure/Color
M3  VSM
M4  Virtual Geometry v1（旧Meshlet作为bootstrap）
M5  Material IR + Solari RT Scene统一 + Lumen实时GI
M5.5 MegaLights/Stochastic Lighting
M6  Hair/Skin/Eye/Foliage/Water/Cloud/Volume/Virtual Texture
M7  Offline reference、质量缺陷清零和性能优化
```

并行关系保持，但接口冻结门槛更严格：

- TSR 需要 View/History ABI 和 motion vector，不必等待完整 virtual resource；
- VSM/Virtual Geometry 必须等待 Virtual Resource Core；
- Lumen 必须等待 GPU Scene、Material IR、Frame Graph、Virtual Resource 和稳定 history identity；
- MegaLights 必须等待统一 LightTable、Ray Scene 和 reservoir primitives。

### 25.13 修正工期预期

Vulkan-first Backend、Frame Graph、ABI codegen 和 Virtual Resource Core 增加前期投入，但减少后期返工。更新后的估计：

| 交付 | 12–16 人日历时间 |
|---|---:|
| 架构/Shader/Vulkan 技术验证 | 3–5 个月 |
| GPU Scene + Frame Graph + ABI | 6–9 个月 |
| TSR 高质量版本 | 8–12 个月 |
| VSM + Virtual Geometry Alpha | 14–20 个月 |
| Lumen/Material IR 内部 Alpha | 22–30 个月 |
| 含 MegaLights 和复杂内容的生产版本 | 28–38 个月 |
| 广泛 UE 5.8.2 同等级画质 | 32–44 个月 |

若坚持 wgpu 与 Raw Vulkan 混合拥有资源、长期只通过 GitHub 页面手工读取源码、或同时追踪 `ue5-main`，上述工期还会显著增加。此前 24–34 个月可以作为理想条件下的激进目标，但不宜作为正式承诺。

### 25.14 第一批实际代码改动重新收敛

第一批不应同时修改 Nanite、VSM 和 Lumen。建议四个小闭环：

1. `render_abi`：Rust/HLSL/WESL layout schema 与 golden tests。
2. `gpu_scene`：SceneHandle、Transform/Bounds/Material 索引、change journal、sparse upload。
3. `frame_graph`：一个 Compute pass、一个 Graphics pass、自动 barrier 和 persistent history import。
4. `shader_package`：一个授权 Compute Shader 经 DXC→SPIR-V→reflection→PipelineCache 的完整链路。

四个闭环均通过 NVIDIA/AMD Vulkan validation 后，再启动 TSR vertical slice。这比同时铺开大型功能更容易确认底层方向正确。

## 26. 第二轮架构复审：继续升级项

第一轮复审解决了 Vulkan owner、ECS 驱动 Frame Graph、ABI、Virtual Resource、Material IR 和大型功能顺序。继续检查当前 Prism 的透明、资产、色彩、动态几何与诊断代码后，还需要补齐以下部分。这些项目决定引擎能否从“高级静态场景 Demo”走向真实项目。

### 26.1 统一 Bindless/Descriptor Heap，而不是继续按材质 Slab 扩展

当前 Bevy bindless 设计主要围绕 material bind group/slab，文档也注明普通 bindless buffer 支持有限。Vulkan-only 高端路径应升级为全局资源句柄模型：

```text
GpuResourceHandle(index, generation, class)
├─ sampled image heap
├─ storage image heap
├─ sampler heap
├─ uniform/storage buffer address table
└─ acceleration structure table
```

要求：

- `VK_EXT_descriptor_indexing`/descriptor buffer 能力按实测选择；
- stable handle 与 physical descriptor slot 分离，便于压缩和迁移；
- generation 防止槽位复用后读到错误资源；
- update-after-bind、partially-bound 和 non-uniform access 统一封装；
- descriptor 回收同样使用 frame epoch；
- MaterialRecord、GPU Scene、VSM、Lumen 和自定义渲染共用句柄语义；
- debug 模式对越界、错误 class、stale handle 和未驻留资源可视化。

不要让每个 UE-derived 模块携带自己的 descriptor layout。Shader adapter 必须把其资源映射到 Prism heap ABI。

### 26.2 新增统一 History Registry 与失效协议

当前 TAA、Solari、PreviousGlobalTransform 和各效果分别管理上一帧状态。TSR、VSM、Lumen、反射、云、体积、自动曝光都会依赖历史；若每个模块单独判断 camera cut，会产生不同步重影。

新增：

```text
HistoryRegistry
├─ ViewHistoryId + generation
├─ SceneEpoch / LightingEpoch / MaterialEpoch
├─ current/previous resolution and jitter
├─ current/previous exposure and pre-exposure
├─ camera cut / teleport / origin rebase
├─ feature version
└─ per-history validity mask
```

所有时域 Pass 声明自己依赖哪些 epoch。失效不是简单 `reset: bool`，而是原因掩码：camera cut、resolution change、FOV change、large transform、material change、light change、streaming reveal、shader version change。这样可以只拒绝真正失效的历史，减少画质闪断。

### 26.3 建立统一 Motion/Disocclusion Contract

TSR 不能只依赖现有 opaque motion vector。必须定义整个引擎的运动数据契约：

- camera、rigid、skinned、morph、vertex animation、world position offset；
- virtual geometry LOD 切换后的稳定 primitive identity；
- masked foliage/wind；
- particle/ribbon；
- transparent/refraction；
- decals 与 surface deformation；
- newly spawned、destroyed 和 teleport 标记；
- previous transform/skin/morph 数据何时保留与回收。

输出不仅是 2D velocity，还包括 disocclusion/reprojection confidence、reactive mask、transparency/composition mask 和 stable surface ID。GPU Scene 负责 identity，具体几何路径负责生成正确运动。

### 26.4 透明渲染升级为独立体系

当前存在排序透明、实验性 OIT 和屏幕空间 transmission，但 UE 同等级画质需要按内容选择路径：

| 内容 | 首选路径 |
|---|---|
| 常规 alpha blend | GPU tile/bin + back-to-front 或近似 OIT |
| 大量粒子 | Weighted/Moment OIT，可按项目选择 |
| 高质量玻璃 | 分层深度、折射、吸收和反射专用路径 |
| 水体 | Single-layer water 专用 shading/composite |
| 毛发 | Hair visibility/deep opacity/专用 resolve |
| 烟雾云 | Volumetric integration，不进入普通透明排序 |

透明系统必须参与 TSR reactive mask、Lumen/反射、雾、DOF、Motion Blur 和曝光。不要试图用一种 OIT 覆盖玻璃、水、毛发和粒子。现有 OIT 保留为一个策略，不作为统一答案。

### 26.5 动态几何与角色 GPU 管线提前

虚拟几何不能代表角色。当前已有 skinning、morph target 和动态 bounds 基础，但生产级路径还需：

- Compute skinning/deformation cache；
- current/previous deformed vertex 数据共享；
- bone palette/morph sparse update；
- skin cache 供 raster、VSM、RT BLAS 和 motion vector 共用；
- deformation bounds GPU reduction；
- BLAS refit/rebuild 策略；
- hair groom/strand/card；
- cloth/geometry cache；
- 动态角色的 LOD 和 streaming。

将 `DeformationGraph/SkinCache` 移到 M3–M4，而不是等 M6。否则 TSR、VSM、Lumen 在静态场景通过后，会因角色路径缺失整体返工。

### 26.6 资产系统升级为 Cooker + 内容寻址 DDC

现有 `AssetLoader/AssetSaver`、KTX2/DDS、压缩纹理和示例 mip generator 是良好基础，但高端渲染需要正式离线流水线：

```text
Source Asset
→ importer normalization
→ dependency graph
→ target recipe
→ processors（mesh/material/texture/RT/virtual pages）
→ content-addressed DDC
→ platform package/chunks
→ runtime async IO
```

DDC key 必须包含源内容 hash、所有依赖 hash、processor version、配置、目标 GPU tier、UE-derived builder version 和 Prism ABI。纹理处理按语义区分 Color、Normal、Roughness/Metal/AO、Height、HDR、Mask；不能用通用颜色 mip 处理全部纹理。

### 26.7 增加完整 Texture Streaming 与 Virtual Texture

当前支持 mip/压缩格式并不等于支持运行时纹理流送。需要：

- mip-tail/root residency；
- GPU feedback 或可见性驱动的 desired mip；
- async IO、解压、staging 和 copy queue；
- texture budget/eviction；
- streaming priority、prefetch 和 camera cut；
- sparse image 或 atlas/page-table 两种后端评估；
- terrain/large material 的 virtual texture；
- VSM/Nanite/Lumen 与 texture streaming 共享预算仲裁；
- 缺页时稳定的父 mip，不得出现黑纹理。

Virtual Texture 应复用 `VirtualResourceCore` 的调度协议，但物理 tile pool 独立。

### 26.8 统一 Scene Visibility 与 Multi-view

GPU culling 结果不应只服务主相机。建立 `ViewFamily`：主视图、双眼、阴影页、反射捕获、Scene Capture、编辑器视图和离线 tile 共享：

- stable view handle；
- view mask；
- coarse instance visibility；
- per-view cluster refine；
- visibility reuse policy；
- LOD importance 和 streaming weight；
- history ownership。

阴影、反射和 GI 不能简单复用主视图最终可见列表，因为屏幕外内容仍会贡献；但可以复用 coarse hierarchy 和场景索引。

### 26.9 大世界坐标与空间分区前置

大世界不仅是浮点精度问题。统一支持：

- double/sector-based CPU world coordinates；
- camera-relative GPU float；
- world origin rebasing epoch；
- GPU Scene spatial pages；
- cell streaming 与引用保活；
- VSM clipmap、Lumen cache、RT AS 和虚拟几何在重定位时一致失效/重映射；
- 跨 cell 灯光、阴影 caster 和 GI 影响范围；
- deterministic replay。

该协议必须在 GPU Scene v1 冻结前确定，否则实例 ABI 和 cache key 会被迫重写。

### 26.10 色彩系统升级为 Scene-linear → Display Pipeline

当前有 HDR 中间目标、Exposure、Auto Exposure、Color Grading 和多个 tonemapper，但 window 代码仍明确把真正 HDR 输出留作未来工作。UE 同等级最终成像需要：

```text
Texture input color space
→ scene-linear working space
→ pre-exposure
→ lighting/composition
→ local exposure
→ scene-referred grading
→ display transform
→ output gamut/EOTF
→ SDR / scRGB / HDR10(PQ)
```

补充：

- 明确工作色域（不能把“LinearRgba”默认等同于完整色彩管理）；
- 每张纹理携带 input color space/transfer metadata；
- pre-exposure 由 View ABI 统一；
- Local Exposure 与 TSR/Lumen 历史一致；
- SDR/HDR 白点、paper white、peak nits 和 gamut mapping；
- 3D LUT/可选 OCIO 接口；
- Screenshot/AOV 可保存未 tonemap scene-linear 数据；
- 不同显示器输出不改变光照计算。

### 26.11 将地形、植被、水和体积列为架构客户端

它们不应在最后作为普通材质补丁：

- 地形需要 clipmap/virtual heightfield、RV/VT、洞、地表混合和大范围阴影/GI；
- 植被需要 instance hierarchy、wind deformation、masked AA、two-sided transmission 和 cluster streaming；
- 水需要网格/FFT、single-layer shading、反射/折射、underwater medium 和 caustics 预留；
- 云和雾需要 froxel/volume cache、temporal reprojection、光照与阴影；
- sparse volume texture 需要单独 streaming client。

这些客户端必须复用 GPU Scene handle、HistoryRegistry、VirtualResourceCore、FrameGraph 和 quality budget，而不是各自建立管理器。

### 26.12 增加 Render Feature SDK 与 ABI 版本化

要支持自定义渲染而不让内部架构僵化，需要定义稳定层级：

```text
Level 1: Material/Shading Model plugin
Level 2: FrameGraph pass plugin
Level 3: GPU Scene field extension
Level 4: Vulkan backend extension（内部受控）
```

每级声明 capability、资源访问、ABI version、history dependency 和 fallback。公开的 Rust 类型不暴露 UE 派生私有结构；授权 Shader package 通过 manifest ID 注册。插件不允许保存 transient graph handle 到下一帧。

### 26.13 GPU 错误隔离、恢复与 Crash 诊断

高端 Shader 数量上升后，device lost 和 GPU hang 不可避免。需要：

- Vulkan validation/robustness debug profile；
- descriptor/buffer bounds instrumentation；
- pass breadcrumb 和 marker；
- GPU timeout 前后的 graph dump；
- shader package/permutation hash；
- NVIDIA Aftermath、AMD GPU crash 工具等可选 adapter；
- pipeline/feature quarantine；
- device lost 后从 Asset + GPU Scene journal 重建；
- 崩溃日志不泄露授权 Shader 源码，只记录可审计 package ID。

### 26.14 可重复渲染、Capture/Replay 与差分调试

仅靠截图难以调试跨帧算法。新增 Render Capture：

- 锁定随机种子、时间、动态分辨率和 streaming；
- 保存本帧 GPU Scene journal、View ABI、资源描述、Pass 参数和 package ID；
- 支持 headless replay；
- Pass-by-pass AOV/dump；
- Prism/UE 相同相机路径和灯光事件脚本；
- 自动二分第一个产生显著差异的 Pass；
- 性能 capture 与画质 capture 分离，避免 instrumentation 污染计时。

授权限制下不保存 UE 源码，只保存允许的输入和内部产物标识。

### 26.15 质量档位从 Feature Toggle 升级为预算求解器

不能只定义 Low/Medium/High 和一组开关。建立 `QualityBudgetController`：

- 输入：目标帧时间、分辨率、显存、功耗/温度、场景压力；
- 输出：render scale、TSR 模式、geometry error、shadow page/mip、GI rays/cache updates、reflection cutoff、volume samples；
- 每个旋钮有画质损失模型和成本估计；
- 使用滞回、冷却时间和一次只改变少数旋钮；
- cinematic/reference 模式禁用自适应；
- 改动记录到 capture，确保性能可复现。

优先保持运动稳定和关键阴影，最后才降低核心材质正确性。

### 26.16 CPU 数据路径继续优化

统一 GPU Scene 后仍需避免 CPU 成为瓶颈：

- per-thread Change Journal，帧末 radix/sort 合并；
- structural change 与 value update 分离；
- upload arena/ring 与 copy batch；
- generation handle 批量分配和 epoch reclamation；
- scene spatial cell 的批量 create/destroy；
- material/texture descriptor 更新去重；
- Render World 查询仅处理 changed/removed，不做全量扫描；
- 遥测 O(changes) 是否退化为 O(scene)。

当前 Meshlet 每帧全实例迭代 TODO、部分 decal clear/rebuild 和各模块独立提取都应列为迁移债务，而不是把这些模式带入新架构。

### 26.17 更新后的模块依赖

```text
Vulkan Backend / ABI / FrameGraph
          │
          ├→ Global Descriptor Heap
          ├→ GPU Scene + Large World + ViewFamily
          ├→ HistoryRegistry + Motion Contract
          ├→ Shader/Pipeline Package
          └→ VirtualResourceCore + Texture Streaming
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
             TSR        VSM     Virtual Geometry
              │          │           │
              └──────────┼───────────┘
                         ▼
               Deformation/Skin Cache
                         │
               Material IR / Transparency
                         │
          Lumen / MegaLights / Reflections
                         │
       Terrain/Foliage/Water/Cloud/Volumes
                         │
          Display Pipeline / Offline Reference
```

### 26.18 更新后的交付优先级

| 优先级 | 必须完成 | 原因 |
|---|---|---|
| P0 | Vulkan owner、ABI、FrameGraph、GPU Scene、History、Descriptor Heap | 后续所有模块共同底座 |
| P1 | Shader package、Virtual Resource、Motion、TSR、VSM | 最早形成稳定高端画面 |
| P2 | Virtual Geometry、Texture Streaming、Deformation Cache、Material IR | 支撑真实内容规模与角色 |
| P3 | Lumen、MegaLights、Transparency、专用材质 | 达到 UE 核心动态画质 |
| P4 | Terrain/Water/Cloud/Volume、HDR Display、Offline | 补齐广泛场景与最终成像 |
| P5 | SDK、Capture/Replay、Recovery、Quality Controller | 生产效率和长期稳定性 |

其中 Capture/Replay 和 GPU crash breadcrumb 虽列为生产能力，最小版本必须在 P0–P1 就开始接入，否则后期难以定位跨帧错误。

## 27. 第三轮架构复审：性能上限与工程效率

前两轮已覆盖主要功能和生产能力。本轮聚焦容易在大型场景中成为隐性瓶颈的 Shader 组合、Command Submission、内存碎片、光追更新和工程验证。

### 27.1 建立 Shader/Pipeline Permutation Budget

当前 PBR、Prepass、Deferred、Meshlet、Fog 等模块通过大量 `PipelineKey + shader_defs + TypeId` 组合生成专门管线。加入 UE-derived Shader、Material IR、TSR、VSM 和 Lumen 后，如果继续无限组合，PSO 数量、构建时间和运行时 hitch 会指数增长。

每个 Feature 必须提交 permutation manifest：

```text
PermutationDimension
├─ StaticRequired     真正改变资源布局或阶段的少数维度
├─ RuntimeBranch      低成本功能分支
├─ Specialization     数值/工作组等 Vulkan specialization constant
└─ WorkClassification 先分类，再间接 dispatch 对应 shader
```

设硬门槛：每个 package 的理论组合数、实际构建数、项目使用数、磁盘体积、首次加载和 warmup 时间。CI 阻止无解释的组合增长。常见材质通过 tile/material classification 合并；不把 tonemap、fog、shadow filter 等所有视图选项乘进每个材质 PSO。

### 27.2 GPU-driven Work Graph/Indirect Scheduler

统一可见性之后，不应继续由 CPU 为每个 Phase 构建完整 draw list。新增 GPU Work Scheduler：

- persistent work queue；
- instance/cluster/material/light 分类；
- prefix sum、radix sort、compaction；
- indirect draw/dispatch count；
- overflow-safe append buffers；
- work stealing/load balancing；
- early/late occlusion 两阶段；
- 统计 occupancy、empty dispatch 和队列饱和度。

Vulkan 路径先使用 indirect draw/dispatch；若目标硬件和驱动对 device-generated commands/Shader Enqueue 等扩展成熟，再通过 capability adapter 追加，不能把实验扩展写入高层 API。

### 27.3 Multi-queue Submission Planner

Frame Graph 不应只生成 barrier，还要形成 submission plan：

```text
Transfer Queue   asset/page upload ───────────────┐
Compute Queue    cull/cache/denoise ──────────────┼→ timeline dependencies
Graphics Queue   visibility/shadow/shading ───────┘
```

要求：

- Timeline Semaphore 统一帧值；
- Queue Family ownership 自动生成；
- async compute 只在实测 overlap 有收益时开启；
- 小 Pass 合并，避免 submit/encoder 开销；
- 单独提交 BLAS、Meshlet write、Slab update 等现有路径逐步迁入 Frame Graph；
- 禁止业务模块直接 `queue.submit`；
- submission 数、wait bubble、queue idle 和 overlap 纳入诊断。

当前部分 Meshlet、BLAS、Slab 和纹理上传会自行创建 encoder/submit；这些需要列入迁移清单，否则 Frame Graph 无法看到完整依赖。

### 27.4 统一 Readback/Feedback Pipeline

VSM、Virtual Geometry、VT、Lumen、统计和 picking 都需要 GPU→CPU feedback。不能每个模块独立 map buffer。建立：

- 每帧一个或少量 readback arena；
- ticket + typed range；
- copy/transfer queue 批处理；
- N 帧延迟和 deadline；
- 压缩/去重后才 readback；
- 最大字节预算；
- timeout/drop policy；
- capture 模式与 runtime 模式分离。

GPU 决策不应同步等待 readback；反馈只能影响后续帧。必须记录 readback latency 和 stale feedback rate。

### 27.5 GPU 内存分配器与在线碎片治理

现有 Slab、Mesh allocator、Meshlet persistent range allocator 可复用经验，但顶级 Vulkan 后端需要统一分配层：

- device-local/upload/readback/transient/AS 专属 heap；
- buddy/TLSF/slab 按资源大小选择；
- dedicated allocation 门槛；
- buffer device address 稳定性；
- alias heap 由 Frame Graph 管理；
- fragmentation、largest free block 和 committed/used 指标；
- 预算压力下增量 compact/relocate；
- relocation 通过 indirection handle 修补，不暴露物理地址给上层；
- AS scratch/transient scratch 跨 Pass 复用。

任何在线整理都有每帧字节预算，不能为“降低碎片”制造帧尖峰。

### 27.6 Ray Tracing Scene 分级更新

当前 Solari 已有 BLAS build/compaction 和 TLAS 增量基础，但 BLAS refit 仍是明确待办。升级为策略系统：

| 几何 | 策略 |
|---|---|
| 静态 | Build → Compact → 长期复用 |
| 刚体实例 | 仅 TLAS transform update |
| 蒙皮/形变 | Refit 或按启发式 Rebuild |
| Virtual Geometry | coarse proxy BLAS + 可选 resident detail BLAS |
| 毛发 | curve/triangle 专用 AS 策略 |
| 粒子 | 默认不进入 AS，必要时低精代理 |

建立 build/refit/rebuild 成本模型、每帧 AS budget、scratch pool、compaction queue 和 lifetime epoch。上一帧 TLAS 引用的 BLAS 必须保活；HistoryRegistry 与 AS epoch 协同，避免时域算法采样已失效实例。

### 27.7 Sampler Feedback、Residency 与 Ray LOD 统一

光追 Shader 当前仍有固定 mip 采样 TODO。Raster、Ray Query、Surface Cache 和 Offline hit 需要统一纹理 LOD 语义：

- ray cone/texture footprint；
- anisotropic/roughness-aware LOD；
- hit distance 和 cone spread；
- texture residency fallback；
- normal/roughness mip filtering；
- alpha-tested ray hit 的稳定 opacity mip；
- Surface Cache capture 与主视图一致的材质 LOD。

否则主视图清晰、反射和 GI 中材质模糊或闪烁，无法达到同等级画质。

### 27.8 GPU Scene 数据局部性与热冷分层

SoA 只是开始。根据访问频率拆分：

```text
Hot
├─ bounds / flags / transform index / geometry index
├─ current transform
└─ visibility/material classification key

Warm
├─ previous transform / custom data
├─ lightmap/decal/layer metadata
└─ RT instance metadata

Cold
├─ editor/debug/name/source asset
└─ rarely used extension fields
```

热表按 16/32 字节对齐并支持 wave-coalesced access；可见性 Pass 不读取材质大块数据。空间 cell 可对实例索引排序，但 SceneHandle 保持稳定，使用 indirection 区分逻辑 ID 与物理排列。通过真实 GPU counter 决定布局，不凭直觉过早压缩。

### 27.9 ECS→GPU Scene 事务与确定性

Change Journal 增加事务语义：

- 同帧 create/update/destroy 折叠；
- parent/child transform 更新排序；
- asset dependency 未就绪时使用版本化 placeholder；
- scene transaction 有 sequence/epoch；
- 多线程 journal merge 输出确定顺序；
- replay 使用同一事务流；
- GPU feedback 只能引用已发布 snapshot；
- 失败事务回滚或延迟整批发布，不能出现几何和材质半更新。

这对编辑器热重载、大规模 cell streaming 和网络同步场景尤其重要。

### 27.10 资源格式与带宽预算

每个 Pass 除时间外还声明带宽类别和格式理由。统一评估：

- visibility ID/barycentric 编码位数；
- depth/normal/motion/history 格式；
- R11G11B10、FP16、packed normal 的适用范围；
- VSM depth page 格式；
- radiance/reservoir cache 压缩；
- UAV/attachment 压缩损失；
- resolve/copy 次数；
- tiled GPU cache friendliness。

建立每帧读写字节估算与实测 counter。画质数据需要高精度时保留 FP16/FP32，但不能默认所有中间目标都用 RGBA16F。格式变化必须通过 reference image 与时域测试。

### 27.11 Shader 数值稳定与跨厂商一致性

UE-derived HLSL 转 SPIR-V 后，需规定：

- fast-math/precise policy；
- NaN/Inf sanitization 边界；
- denormal/flush-to-zero 假设；
- subgroup size control 或 size-independent algorithm；
- wave intrinsic fallback；
- atomic ordering；
- FP16 使用白名单；
- NVIDIA/AMD golden tolerance；
- Debug Shader 中越界/非法值计数。

时域缓存一旦写入 NaN，可能污染数百帧；所有 history/cache write 在 debug 和 shipping-safe 边界都应有有限值保护。

### 27.12 Feature 故障域和逐级回退

高端功能不能只有“加载成功/黑屏”。为每个模块定义 health state：

```text
Unavailable → Initializing → Warming → Active
                         ↘ Degraded → Quarantined
```

- Shader/PSO 未就绪使用明确 fallback；
- VSM 页池故障回退传统 shadow；
- Lumen history/cache 无效回退 probe/environment；
- Virtual Geometry 缺页渲染 resident root/fallback mesh；
- TSR 故障回退 TAA/native resolution；
- 单 Feature 可隔离禁用，不必重建设备；
- 所有降级原因进入 capability report/capture。

### 27.13 Benchmark 扩展为微基准 + 场景基准 + Soak

三层基准缺一不可：

- **微基准**：descriptor access、scatter、sort、indirect、mesh shader、ray query、async overlap、copy bandwidth；
- **场景基准**：室内、城市、森林、角色、夜景、水体、大世界；
- **Soak**：2–8 小时 camera path、热重载、cell streaming、分辨率切换和显存压力。

记录平均、P50/P95/P99/P99.9、shader hitch、page fault、allocation peak、device lost、画质 AOV。微基准负责路径选择，不能替代真实场景结论。

### 27.14 建立 Feature Flag、实验通道和删除期限

每个新旧路径都有：

- compile feature；
- runtime capability；
- quality/profile selection；
- kill switch；
- telemetry ownership；
- fallback；
- graduation criteria；
- 旧路径删除版本。

实验实现达到毕业门槛后必须合并或删除，不能永久同时维护 Meshlet/Virtual Geometry、多套 GI、多套 Shadow 和多套 Material ABI。

### 27.15 API 稳定面最小化

在架构尚未成熟时，只稳定语义，不稳定物理布局：

- 稳定 `SceneHandle/MaterialHandle/ViewHandle` 行为；
- 稳定 Feature 注册、资源访问声明和错误模型；
- 不公开 Buffer offset、descriptor set、Vulkan handle 和具体 GBuffer 格式；
- internal crate 可快速迭代；
- 用户 Shader 通过版本化 semantic bindings，而非硬编码 binding number；
- 每次 ABI bump 提供 package reject 与迁移诊断。

这能避免为了早期 API 兼容牺牲后续性能布局。

### 27.16 更新后的前置门槛

在启动大型功能前新增硬性门槛：

| 大型功能 | 前置条件 |
|---|---|
| TSR | HistoryRegistry、Motion Contract、Shader Package、Display/Exposure ABI |
| VSM | GPU Scene Bounds/Epoch、VirtualResourceCore、FrameGraph、Indirect Scheduler |
| Virtual Geometry | VirtualResourceCore、GPU feedback、Geometry Asset v1、Allocator |
| Lumen | Material IR、History、Texture Ray LOD、RT Scene、VirtualResourceCore |
| MegaLights | LightTable、Reservoir primitives、RT visibility、VSM integration |
| Hair/Water/Volume | Transparency architecture、History、专用 visibility/composite contract |

不满足前置条件时只允许做离线算法实验，不能建立临时生产集成层。

## 28. 最终推荐

实施主线固定为：

```text
远程 UE 调查与来源治理
→ HLSL/SPIR-V 窄兼容层 + Unified GPU Scene
→ TSR
→ VSM 与现有 Meshlet/Nanite 融合
→ Solari 场景统一
→ Lumen/Solari 融合
→ 专用材质、体积、水体、云
→ 离线参考与 UE 同等级画质验收
```

最重要的边界是：**移植算法，不移植 UE 世界；移植 Shader，不移植 RHI；复用成熟质量，不建立第二套场景和资源生命周期。** 这样才能真正获得授权源码带来的时间收益，同时保留 Prism 更轻、更数据驱动、适合统一 GPU Scene 的架构优势。
