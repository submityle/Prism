# Prism 渲染系统对照 UE5 的差距审视（详细）

> 基线：`main@fad551316`，审阅范围 `pkg/prism_*`（渲染相关 42 个 crate）+ `crates/bevy_*`（62 个 crate）+ `os-backends/` + `docs/prism_rendering_architecture_zh.md`、`docs/prism_gi_lumen_design_zh.md`、`docs/prism_rhi_m0_m2_plan_zh.md`、`docs/prism_render_scene_rhi_port_zh.md`。
>
> 对照基准：UE5 5.4–5.5 的渲染架构（Nanite / Lumen / Virtual Shadow Maps / TSR / Substrate / MegaLights / World Partition / RDG / PSO Precaching / Path Tracer）。
>
> 本文只回答一个问题：**Prism 的渲染系统，哪些地方还不够好、为什么不够好、离 "UE5 同等级" 还差什么。**

**目录**

- [0. 结论先行](#0-结论先行)
- [1. 判定口径：怎么算"有"](#1-判定口径怎么算有)
- [2. Prism 渲染系统的真实拓扑](#2-prism-渲染系统的真实拓扑)
- [3. 根基层：RHI / 帧图 / GPU Scene / 描述符 / 驻留](#3-根基层rhi--帧图--gpu-scene--描述符--驻留)
- [4. 几何与虚拟几何（对照 Nanite）](#4-几何与虚拟几何对照-nanite)
- [5. 可见性与遮挡剔除（对照 InstanceCulling + 两趟 HZB）](#5-可见性与遮挡剔除对照-ue-的-instanceculling--hzb-两趟)
- [6. 阴影（对照 Virtual Shadow Maps）](#6-阴影对照-virtual-shadow-maps)
- [7. 动态 GI 与反射（对照 Lumen）](#7-动态-gi-与反射对照-lumen)
- [8. 直接光与海量光源（对照 MegaLights）](#8-直接光与海量光源对照-clustered-lighting--megalights)
- [9. 材质与着色（对照 Substrate）](#9-材质与着色对照-substrate)
- [10. 纹理与流送（对照 VT / SVT / RVT）](#10-纹理与流送对照-vt--svt--rvt)
- [11. 后处理 / AA / 上采样（对照 TSR）](#11-后处理--aa--上采样对照-tsr--dlssfsrxess)
- [12. 专用管线：毛发 / 布料 / 水体 / 地形 / 粒子 / 大气 / 植被](#12-专用管线毛发--布料--水体--地形--粒子--大气--植被)
- [13. 光追 / 路径追踪 / 离线验证](#13-光追--路径追踪--离线验证对照-ue-ray-tracing--path-tracer)
- [14. 工具链 / 着色器编译 / 诊断](#14-工具链--着色器编译--诊断对照-shadercompileworker--pso-precaching--stat-gpu)
- [15. 平台与可伸缩性](#15-平台与可伸缩性对照-scalabilities--主机--移动--vr)
- [16. 内容管线与工具](#16-内容管线与工具对照-editor--ddc--cooking)
- [17. 验收矩阵的可执行性](#17-验收矩阵的可执行性pra-21-现在一条都跑不了)
- [18. 建议的优先级路线](#18-建议的优先级路线)
- [19. 什么才算"够好"](#19-什么才算够好)
- [附录 A：证据复核命令](#附录-a证据复核命令)

---

## 0. 结论先行

### 0.1 一句话

Prism 的渲染问题**不是功能清单短**，而是三条互不装配的技术栈各自把自己修到了"内核级完备、系统级空缺"：能做到的机制很多，能证明的很少，能端到端跑起来的一条都没有达到 UE 级画质门槛。

### 0.2 十条最要命的差距（按严重度排序）

| # | 差距 | 性质 | 影响面 |
|---|---|---|---|
| 1 | **没有单一可运行的渲染基线**：`prism_render_scene`(162K 行/178 shader)、`prism_render_app`(32K 行) 的仓库内消费者数均为 **0**，唯一装配新 RHI 的可执行文件是 `tools/render_triangle` | 工程/架构 | 一切画质承诺都无法度量、无法回归、无法交付 |
| 2 | **实际出图的仍是 bevy 栈**，它没有 Nanite/VSM/Lumen/TSR/Substrate，且能力上限由 wgpu 决定 | 能力 | 与 UE5 的对比实质上是"Bevy wgpu 渲染器 vs UE5" |
| 3 | **RHI 是 wgpu 门面**，没有 mesh shader、work graphs、sparse residency、bindless tier-2、console 后端、PSO 预缓存控制 | 底座 | 直接否决 Nanite/VSM/SVT 的原生实现形式，只能软件模拟 |
| 4 | **虚拟几何缺"生产端"**：有 25 个 GPU 内核（cluster cull / LOD projection / DAG cut / 软件光栅 / vis buffer），但没有离线 cluster DAG 构建（简化、误差度量、watertight、UV/normal 保持） | 几何 | Nanite 的难点在构建器，不在光栅器 |
| 5 | **Lumen 等价物只到"片元级"**：有 SSGI/SSR/spec-GI/world-ReSTIR/surface-cache/DDGI 的 shader 与 pass，但没有 far-field 表示、没有 multi-bounce 能量验证、没有 leak 抑制的系统性工程 | GI | 静止截图能看、运动与暗部会崩 |
| 6 | **VSM 有 page table/receiver，但没有 GPU 侧失效与常驻**：caster epoch、swept volume 失效仍停留在设计文档 | 阴影 | 动态场景一移动就退化成 CSM 或全量重绘 |
| 7 | **无 PSO 预缓存与 shader permutation cooking**：只有 `shader_package`（SPIR-V 制品校验）与运行时 pipeline cache | 工具链 | 首帧/首次遭遇材质必卡，主机平台不可接受 |
| 8 | **无编辑器/材质图/DDC/自动 LOD/HLOD 构建**：画质只能靠手写测试场景验证 | 内容管线 | "UE5 同等级画质"这个门槛在流程上无法被证明 |
| 9 | **能力分层与降级策略未闭环**：`quality` 只有 `AdaptiveQualityController` 契约，没有"缺页→父 mip→低分辨率"的逐级回退链被验证过 | 鲁棒性 | 高端档与兼容档的边界全靠设计假设 |
| 10 | **诊断与性能可观测性弱**：无 GPU pass 级计时、无 buffer visualization、无 GPU crash dump 等价物 | 可维护性 | 性能回归只能靠外部工具逐帧抓 |

另有三个没排进前十、但同样属于"整块缺失"的空洞：**GPU 粒子渲染/模拟前端完全不存在**（`pkg` 下无任何粒子渲染 crate，bevy 侧也未集成 hanabi，对照 Niagara）；**Substrate 式分层材质缺位**（`Layer`/`Mix` 退化为一个 feature 位，见 §9.3）；**毛发没有 groom 导入/绑定/LOD 管线**（87 个 shader 是孤岛，见 §12）。

### 0.3 一个必须说清楚的判断

`docs/prism_rendering_architecture_zh.md` 把"UE5.x 同等级画质"写成发布门槛。按当前代码状态，这个门槛**不是差 30%，而是差一个完整的"产品化闭环"**：机制（mechanism）已经写了很多，产品（product）——可运行、可度量、可退化、可被内容驱动——几乎为零。详见 §2 与 §17。

---

## 1. 判定口径：怎么算"有"

本文不按"文件存在"计分，按**可执行证据**分五档。每一档都给出可复核的取证方式。

| 档位 | 定义 | 取证方式 | 含义 |
|---|---|---|---|
| **E4 产品级** | 端到端出图 + 自动回归 + 降级策略 + 平台覆盖 | example/benchmark + CI 图像回归 + 多后端跑通 | 可以拿来和 UE 比画质 |
| **E3 可运行基线** | 有真实 app/example 装配，能出图 | `examples/` 或 `tools/` 入口 | 能看见、能量 |
| **E2 GPU 内核级** | 独立 crate 的 GPU kernel + 真机 parity 测试，但无消费者 | `*_gpu` crate + `PRISM_REQUIRE_GPU` 测试 + 0 消费者 | 内核是对的，系统不存在 |
| **E1 CPU 契约级** | 只有 CPU golden/契约/ABI，无 GPU 执行 | 无 shader 的 `.rs` 单测 | 只是规格 |
| **E0 设计级** | 只有 `.md` | docs | 只是意图 |
| **缺失** | 无 | — | — |

**这份文档里绝大多数"Prism 的 UE 级特性"落在 E1/E2**，而 UE 的对应特性落在 E4。这个落差本身就是结论。

---

## 2. Prism 渲染系统的真实拓扑

### 2.1 三层并行栈

```text
┌─ 层 A：bevy 栈（唯一实际出图）────────────────────────────┐
│ crates/bevy_render 35,303 行  ← wgpu 封装，RenderApp/RDG-lite │
│ crates/bevy_pbr    40,936 行  ← clustered forward+ / CSM     │
│ crates/bevy_core_pipeline 7,851 行 ← TAA/SMAA/FXAA/CAS/bloom │
│ crates/bevy_solari 4,977 行   ← ReSTIR DI/GI + PT（实验）     │
│ crates/bevy_anti_alias 2,903 行 ← 含 DLSS 绑定                │
│ 共 62 个 crate，根 Cargo.toml 就是 Bevy 0.20.0-dev 门面       │
└──────────────────────────────────────────────────────────────┘
┌─ 层 B：prism_render_scene（Prism 版渲染器，但不被装配）────┐
│ 162,762 行 / 178 个 .wesl                                    │
│ SSGI / SSR(HZB) / spec-GI(ReSTIR) / world-space GI / DDGI /   │
│ surface cache / world ReSTIR / VSM(page_mark/page_table/atlas)│
│ / TAA / upscale(RCAS+reconstruct) / volumetrics / clouds /    │
│ hair(14) / cloth(13) / water(16) / particle / GTAO ...        │
│ 依赖 bevy_render + bevy_pbr + bevy_core_pipeline（S1 起在往   │
│ PrismGpu 上迁）                                               │
│ 消费者数：0                                                  │
└──────────────────────────────────────────────────────────────┘
┌─ 层 C：新 RHI + 契约 + 独立 GPU 内核岛（只有测试与 bench）──┐
│ prism_render_architecture 415,346 行 / 861 rs / 29 wesl      │
│ prism_volumetric_gpu 347,150 行 / 80 shader                  │
│ prism_render_shading 133,372 行（CPU golden 为主）           │
│ prism_render_material 45,734 行、prism_hair_gpu 28,382 行/87 │
│ prism_render_driver 7,538、visibility 6,682、graph 6,057      │
│ prism_render_runtime 4,204、gpudriven 3,447、vtex_gpu 2,311   │
│ virtual_geometry_gpu 7,822（25 shader）、water 1,476、        │
│ micromap/dmm/radiance_cascades/ray_reorder（合计 ~3.9K）      │
│ 消费者：prism_render_app=0，render_scene=0，     …           │
└──────────────────────────────────────────────────────────────┘
```

合计渲染相关代码约 **120 万行 Rust + 400 个 shader**，其中能被一个真实应用装配起来的部分是 **层 A**。

### 2.2 消费者事实表（可复核）

用 `grep -rl "^<crate> *=" --include=Cargo.toml .`（排除自身）统计。行数为 `pkg/<crate>/src` 下 `.rs` 之和（不含 tests/benches），shader 数为 crate 内 `.wesl`/`.wgsl` 文件数：

| crate | 行数 | shader | 消费者 |
|---|---:|---:|---:|
| `prism_render_scene` | 162,762 | 178 | **0** |
| `prism_render_app` | 32,125 | 2 | **0** |
| `prism_volumetric_gpu` | 347,150 | 80 | **0** |
| `prism_hair_gpu` | 28,382 | 87 | **0** |
| `prism_virtual_geometry_gpu` | 7,822 | 25 | **0** |
| `prism_water_render` | 1,476 | 2 | **0** |
| `prism_micromap_gpu` | 667 | 2 | **0** |
| `prism_dmm_gpu` | 738 | 4 | **0** |
| `prism_ray_reorder_gpu` | 691 | 2 | **0** |
| `prism_radiance_cascades_gpu` | 782 | 3 | **0** |
| `prism_virtual_texture_gpu` | 2,311 | 3 | 1（benches） |
| `prism_render_gpudriven` | 3,447 | 4 | 1（benches） |
| `prism_render_shading` | 133,372 | 5 | 2 |
| `prism_render_visibility` | 6,682 | 0 | 2 |

仓库里唯一调用新 RHI 的可执行文件：

```text
tools/render_triangle/src/main.rs:21: use prism_render_runtime::WgpuRuntime;
```

也就是说：**`prism_render_scene` 里那 178 个 shader（SSGI/SSR/VSM/spec-GI/水面/毛发/布料）从来没有被任何应用、示例或工具装载过。** 它们只在自己的集成测试里被编译、被单 pass 执行。

> 更准确地说：`prism_render_scene` 的设计形态是 **Bevy render sub-app 上的附加层**，而不是替代 Renderer（`pkg/prism_render_scene/src/plugin.rs:34`；实体通过 `PrismGpuScenePlugin::entity()` 选择走"保留式 GPU Scene"或"legacy renderer-only 路径"）。所以"0 消费者"的准确含义是：**没有任何应用/示例 `add_plugins(PrismGpuScenePlugin)`**，插件与它的 178 个 shader 因此从未被真实帧驱动过。这个设计选择让"接线"的工程量比换掉 bevy 小，但也正是它让整条栈长期处于"可编译、不可见"的状态。

### 2.3 这个拓扑为什么本身是最大的"不够好"

1. **画质不可度量。** "UE5 同等级"是视觉结论，需要同一套场景、同一套后处理、同一套相机路径出图对比。现在三条栈各自有各自的坐标系，连"当前画质是多少"都答不出来。
2. **回归会分叉。** 同一个效果（例如 SSR）在 `bevy_pbr`、`prism_render_scene`、`prism_render_architecture` 里有三份语义/三份 shader。改一处，另外两处不跟，最终没人知道哪份是真相。
3. **优化无从下手。** UE 的性能工作建立在"单一帧图 + pass 级计时"上；Prism 的 pass 分散在三个栈，没有统一的 GPU timing 采集，profile 只能在单个 crate 的测试里做微基准。
4. **人力被摊薄。** 120 万行里，真正决定"像不像 UE"的部分（Lumen 的降噪收敛、VSM 的缓存失效、Nanite 的构建器）恰好是每个栈都做了一半的部分。

> 这不是"代码写得不好"，而是**投入方式的问题**：先横向铺满 40 个系统，再纵向打通任何一条。UE 的顺序相反。

---

## 3. 根基层：RHI / 帧图 / GPU Scene / 描述符 / 驻留

### 3.1 wgpu 天花板就是 Prism 的天花板

`os-backends/` 只有两个后端：`prism_render_driver_wgpu` 与 `prism_render_driver_null`。所有 GPU 执行最终都落到 wgpu（Vulkan/Metal/DX12/GLES 的抽象层）。对照 UE RHI 的直接后果：

| UE 机制 | 依赖的底层能力 | wgpu 现状 | Prism 后果 |
|---|---|---|---|
| Nanite 硬件光栅 | mesh shader / 可编程光栅 | 无（WGSL 无 mesh/task shader） | 只能 compute 软件光栅，带宽与三角形吞吐受限 |
| Nanite/WPO 的程序化位移 | 顶点/网格阶段任意调度 | 受限 | 需在 compute 里重写 |
| VSM 页表常驻 | sparse texture / 显存驻留 | 无 sparse residency | 只能自建 atlas + 手动分页 |
| SVT/RVT | sparse texture + 硬件 page table | 无 | 只能自建页表 + indirection，采样成本上升 |
| Lumen/Empire 的 work graphs | work graphs (D3D12) | 无 | 只能多 dispatch 模拟，调度开销与并行度差 |
| PSO 预缓存 | 驱动级 PSO cache 控制、离线编译 | 仅 wgpu pipeline cache（黑盒） | 无法做 UE 的"录制-烘焙-预热"闭环 |
| 多队列/异步 compute | 显式多队列 + fence | 部分（wgpu 抽象掉了 queue 细节） | 异步 compute overlap 不可控 |
| Bindless 全量 | descriptor heap tier 2 / bindless arrays | 有 binding arrays，但上限与兼容性差 | 大场景描述符管理需分批，draw 融合受限 |
| Console | PS5/XSX/Switch 原生 RHI | 无 | 主机平台不可达 |
| GPU crash dump / marker | 平台调试 API | 无（有部分 marker） | 崩溃定位靠猜 |

**结论：** 只要 RHI 停留在 wgpu 门面，UE 的若干核心机制在 Prism 里只能是"模拟实现"，而模拟实现的性能/质量天花板没有任何数据支撑。`docs/prism_rendering_architecture_zh.md` 自称后续要做原生 Vulkan 后端，但当前 `prism_render_driver` 的 trait 设计（`RenderDevice`/`RenderQueue`/描述符集）是围绕 wgpu 能力建模的，届时要么大改 trait，要么保留 wgpu 语义。

### 3.2 Prism RHI 契约 vs UE RHI

| 维度 | UE RHI | Prism 现状 | 差距 |
|---|---|---|---|
| 后端 | D3D12/Vulkan/Metal + 主机 | **只有 wgpu + Null**，`backend/mod.rs:3-4` 自述 "exactly one backend ... no second, native backend to escalate to" | 上限由 wgpu 决定 |
| 资源状态与屏障 | 部分自动（RDG）+ 手动兜底 | `prism_render_graph` 五阶段编译含 barrier 求解（`compile/barrier.rs:42`） | 方向对，但只覆盖单队列、无跨队列所有权转移 |
| Transient 资源 | RDG 全权管理 + **内存别名** | 别名**只算不做**：`AliasPlan::heap_size` 无执行端消费者，`realize()` 给每个瞬态各自 `create_texture`（`graph.rs:1152-1190`、`compile/alias.rs:10-45`） | 显存收益 = 0 |
| 队列批处理 | RDG 多队列提交 | `queue_batches` 只进元数据，执行端 `queue.submit(&[单个 command buffer])`（`plan.rs:250`、`graph.rs:1052,1082`） | 分批无收益 |
| 异步 compute | 有 fence + overlap 度量 | `PassFlags::ASYNC_COMPUTE` 只映射成 `QueueClass::Compute` 标签；runtime 全程只有一对 device/queue（`runtime/lib.rs:268,344-353`） | 完全缺失 |
| Pass 合并 | NativeRenderPass / TBDR tile | `MERGE_CANDIDATE` 只出现在 Debug 输出，每 pass 各自 `push_render_pass`（`pass.rs:105,164`） | 移动端无法用 TBDR |
| 并行 Pass 设置 | RDG 多线程 setup | `parallel` feature **无任何 crate 启用**；实现是每帧 `std::thread::scope` 临时线程（`parallel.rs:64-83`） | 默认单线程 |
| 编译缓存 | 结构稳定时不重编译 | 每帧重编译（`compile/mod.rs:49`） | CPU 成本固定 |
| GPU 计时 | per-pass timestamp + stat GPU | 弱 | 无法定位热点 pass |
| 描述符 | 全 bindless（UE5 已全面 bindless） | `descriptor_heap` 契约**无消费者**；生产用固定长度 binding array（`MAX_BINDLESS_TEXTURES=4096`，`texture_heap.rs:243-249`） | 堆容量启动后固定、不能动态增长/压缩 |
| 命令录制 | 多线程 RHI command list | 有渲染线程（`pipelined_rendering.rs:181-183`），无独立 RHI 提交线程 | 同步原语只有"整队 wait + 帧令牌" |

### 3.3 最刺眼的落差：RDG 编译器有了，RDG 运行时没有

Prism 的帧图**编译器**是整套代码里质量最高的部分之一：SSA 版本化 + RAW/WAR/WAW 求解 + 五阶段（cull→schedule→lifetime→alias→barrier），`compile/` 目录真实可读。

但执行端把这些成果全部丢掉：

```text
别名计划   → 只算不分配        显存峰值不降
队列批次   → 只做元数据        单队列单次 submit
异步 compute → 标签            overlap = 0
pass 合并  → Debug 字符串      subpass/TBDR = 0
并行录制   → feature 默认关闭   大场景 CPU 提交不缩短
编译缓存   → 每帧重编译        帧时间基线抬高
```

对应的 UE RDG 收益（显存峰值降 30–60%、async overlap、多队列 copy）**兑现率 ≈ 0**。同时 bevy 基线的 `RenderGraph` 已经退化成空标签：

```rust
// crates/bevy_render/src/renderer/mod.rs:48
pub struct RenderGraph;
```

也就是说**现役渲染路径根本没有帧图**，pass 顺序靠 system set 顺序，同步靠 wgpu 自动屏障。三条栈里"有图但不执行"和"无图但有画面"各占一头，中间那层（图生效的运行时）不存在。

### 3.4 GPU Scene vs UE 的 GPUScene

UE 的 `FGPUScene` 是全局唯一真相源：primitive/instance/light/reflection capture 走同一条上传通道，带 `FInstanceCullingContext` 与 GPU 端增量更新。

Prism 有**四份**：`prism_render_architecture/src/gpu_scene`（契约，`SceneHandle{index, generation}`、change journal、epoch 回收）、`prism_render_scene/src/scene.rs`（`RenderGpuScene::apply_transaction/bind_entity`，实际使用）、`prism_render_app`（从 `prism_ecs::gpu_resident` 取列布局）、以及 bevy 基线自己的 `AtomicSparseBufferVec` 路径（`crates/bevy_render/src/render_resource/sparse_buffer_vec.rs:195-251`——这条反而是活的）。

**不够好的点：**

- **无 GPU 端 spawn/delete**：全仓 grep `gpu_spawn` 零命中，事务只能 CPU 发起。UE 的 GPU Scene 支持 GPU 端创建销毁（Niagara、GPU 驱动实例化都依赖它）。
- **增量上传退化**：自研 `SparseBufferVec` 只做整 buffer `write_buffer`（`gpu_scene/sparse_buffer.rs:1-60`），`UploadPlanner` 的 SparseScatter/ContiguousRanges/FullRewrite/Deferred 策略只有类型（`upload.rs`）。活路径靠 bevy 的 `AtomicSparseBufferVec` + compute delta job。
- **承载面窄**：UE 的 GPUScene 还管 light grid、reflection capture、decal、Niagara VF、morph target、骨骼矩阵；Prism 契约只覆盖 instance/geometry/material/light/probe/decal，**morph/skin/particle 未进同一张表**，后续每个特性都得再开一条上传路径。
- **字段级 dirty 未实现**：`apply_transaction` 是实体级，不是 UE 的字段级 patch。

### 3.5 虚拟资源与驻留：三套页系统互不相通

PRA §8 定义了漂亮的 `VirtualResourceClient`（几何/纹理/阴影/GI 共用页生命周期，优先级 = 六因子乘积）。实际：

- `architecture/virtual_resource`（registry/scheduler/state/budget，1,284 行）**零消费者**。
- `architecture/paging/streaming.rs` 是完整的 Nanite 式优先级流送器（ResidencyTable/PagePool/PageStorage/Reconcile），但只被虚拟几何流适配器与对拍测试驱动。
- `texture_streaming` 有独立的 `PhysicalPagePool` + feedback。
- `render_app/src/render_asset/page_streaming.rs` 又有一个 `FixedPagePool`。
- VSM 的驻留在 `render_scene/shading/virtual_shadow` 是第四套。

**UE：一个 streaming manager 管所有虚拟资源，统一预算、统一抢占。Prism：四套页表各自为政，预算算几份（`MemoryBudget`/`HeapStats`/`FrameWorkBudget` 都是 CPU 记账），设备显存查询 pending。** 这不是实现细节差异，是 thrash 抑制与预算仲裁能力的有无。

### 3.6 PSO 与着色器编译：app 路径意外地好，场景路径没接上

这是少数**Prism 做得比预期好**的地方：

- `prism_render_app/src/pipeline_cache.rs:401-481,847-969`：真实后台编译线程池（有界通道）+ LRU + 磁盘持久化 + warm set 就绪度。
- 驱动层内容寻址缓存（PSO/layout/sampler，引用计数 + BTreeMap 确定性）：`prism_render_driver/src/cache.rs:1-30`。
- 驱动层子分配器 TLSF/Buddy/Ring 都在（`suballoc.rs:136,432,548,620`），只是帧图不用。

**但仍缺 UE 的闭环：** warm set 没有和场景资产/关卡绑定（UE 是"编辑器里录制 → 烘焙 → 进包 → 首帧零卡顿"），场景侧仍走 bevy `PipelineCache` 的 `AsyncComputeTaskPool`。所以实际游戏的首帧卡顿问题在 Prism 上并没有被解决，只是被推迟到"哪条路径先被采用"。

---

## 4. 几何与虚拟几何（对照 Nanite）

### 4.1 Nanite 真正难的地方

外界常把 Nanite 等同于"GPU 软件光栅 + 虚拟几何"，但 UE 的实现里，**真正决定成败的是构建器**：

1. **切簇**：把网格切成 ≤128 三角形的 cluster，控制顶点复用与边界顶点数。
2. **逐级简化**：每层把 4 个 cluster 合成 group，简化到约 50%，重复到只剩一个根。
3. **误差度量**：每个 cluster 记录相对父层的几何误差（世界空间），运行时用投影误差选 cut。
4. **边界锁定（watertight）**：跨 cluster 的边界顶点必须焊接，否则简化后出现裂缝。
5. **属性保持**：UV seam、硬边法线、切线、顶点色都不能在简化中丢失。
6. **DAG 而非树**：一个 cluster 允许多个父，从而支持更优的 cut 与共享（UE 用 group 概念 + 多父）。
7. **DDC**：构建结果进 DDC，否则每次改模型都要重跑数分钟。
8. **运行时**：持久线程调度、cluster BVH 遍历、两趟遮挡（先 compute 只写深度，再硬件光栅大三角形 + 软件光栅小三角形）、64 位 visibility buffer、material resolve、页流送、与 VSM/Lumen 的 caster 复用。

### 4.2 Prism 现状

`pkg/prism_render_architecture/src/virtual_geometry`（CPU golden）+ `pkg/prism_virtual_geometry_gpu`（25 shader，26 个真机 parity 测试）合计约 2 万行，覆盖：

| 环节 | 状态 | 证据 |
|---|---|---|
| cluster 数据契约 | **真实可用**：`CLST` magic/v3、局部 u16 量化、32 个顶点属性 | `cluster_payload.rs:17-21`、`cluster_payload/attributes.rs:13` |
| 三角形上限 | 128（7 bit 打包进 payload） | `software_raster.rs:170`、`:395` |
| 层次结构 | 有，但是**树**（first_child/child_count 连续区间、单父），注释自称 DAG | `hierarchy.rs:1`、`:36-42` |
| 误差 | 只消费不计算：`self_error` 由外部资产提供，运行时只做投影误差（平方形式，免 sqrt） | `hierarchy.rs:34-42`、`lod.rs` |
| 运行时 cut | 自顶向下、单调误差、子树叶包围 + GPU 孪生 `select_cut.wesl` | `hierarchy.rs`、`shaders/select_cut.wesl` |
| 软件光栅 | CPU 边函数扫描线 + GPU vis buffer 软光栅，有 parity | `software_raster.rs`、`vis_buffer_software_raster.wesl` |
| 页表/驻留/驱逐/流送 | CPU 侧成体系 | `paging/mod.rs:53,146`、`page_pool.rs:128`、`streaming.rs:98` |
| GPU 选簇/页驻留 pass | 存在，但**零下游依赖** | 根 `Cargo.toml:116` 仅 workspace 成员 |

### 4.3 逐条差距

| 差距 | 说明 | 严重度 |
|---|---|---|
| **无 cluster 构建器** | 仓库没有切簇、没有简化、没有误差计算。注释明确 builder 在模块之外：`virtual_geometry/asset.rs:3`。Nanite 数据链断在源头，现有全部簇/页/LOD 代码只能吃外部烘焙数据 | 致命 |
| **DAG 退化为树** | `hierarchy.rs` 是单父树。UE 的多父 DAG 是 cut 质量与几何共享的关键；树结构下简化误差只能单调不增，cut 空间小一截 | 高 |
| **误差度量未自证** | `self_error` 外部输入，意味着"简化质量"从未被自己的构建器验证；UE 的误差是构建器算出来的 | 高 |
| **硬件光栅缺失** | 只有策略分类器（`raster_path.rs:41`、`mod.rs:121-127` 的 MeshShader 枚举），没有 mesh shader 实现。UE 的大三角形走硬件光栅，小三角形走软件光栅，两者混合是性能关键 | 高 |
| **无持久线程/无 cluster 级 BVH 遍历** | GPU 侧是逐 dispatch 的 cull/select，没有 UE 那种一个 persistent kernel 里做 instance→cluster 层级遍历 | 高 |
| **未接入最终画面** | 25 个 shader、页流送、cut 全在"孪生验证"状态，渲染器一行不调用 | 致命 |
| **只有不透明三角形** | payload 无材质/alpha 域，vis buffer 载荷是 7 bit 三角形 id（`software_raster.rs:170-181`）。Masked 材质、双面、透明、植被全无 | 高 |
| **无形变簇** | `virtual_geometry/` 全目录 skin/deform/morph 零命中。UE 5.5 已有 Nanite 骨骼网格（实验） | 中 |
| **无 Nanite 光追** | 见 §13，连 BLAS/TLAS 构建都没有 | 高 |
| **未与 VSM/Lumen 复用 caster** | VSM caster 是逐实例 replay 普通几何（`atlas/raster.rs:105-143`），而不是复用簇层级 | 高 |

### 4.4 被低估的"看不见的工程"

即使构建器写出来，还有一批 UE 花了数年打磨、文档里一句话带过的细节：

- **镜像/对称网格**：简化时两边必须一致，否则 LOD 切换时左右不对称。
- **UV seam 与硬边**：seam 上的顶点不能焊，但几何必须封闭；这两件事冲突时 UE 有专门的 seam 处理。
- **顶点属性压缩**：位置/法线/切线/UV 的量化误差要在屏幕空间可见阈值以下。
- **构建时间**：UE 用 DDC + 多线程 + 增量；一个 3A 场景的 Nanite 构建是小时级。
- **退化情况**：非流形边、零面积三角形、超大三角形、负缩放实例。

Prism 的 `prism_asset_build_core` 有构建框架，但没有针对虚拟几何的构建管线，这些细节全部为零。

### 4.5 结论

虚拟几何是 Prism **最接近 UE 形态、但离产品最远**的子系统：内核与数据结构做得比预期好（payload/量化/parity 测试都在），缺的恰好是最难、最不显眼的两块——**构建器**与**接线**。

---

## 5. 可见性与遮挡剔除（对照 UE 的 InstanceCulling + HZB 两趟）

### 5.1 Prism 现状：这一块意外地扎实

`prism_render_scene/src/visibility` 实现了完整的两阶段遮挡流水：

```text
前帧 HZB 分类（EARLY=1 / LATE_RETEST=2 / LATE_VISIBLE=4）
  → early prepass（仅通过前帧 HZB 的候选）
  → 重建本帧深度金字塔
  → 本帧复测（late retest）
  → late compact（只接受"前帧被挡且本帧通过"）
```

- 阶段位语义正确：`hzb_visibility.wesl:75-80`、`hzb_late_compact.wesl:55`（`& 6u != 6u` 丢弃）。
- 保守性处理到位：camera cut / history 无效时按未遮挡放行；reverse-Z 取 min：`hzb_visibility.wesl:63-64`。
- early 流被 stage 位门控：`visibility.wesl:213`。
- GPU 写间接绘制参数真实存在：`visibility.wesl:244`、`:262-276`。
- HZB 金字塔复用 bevy 的 `ViewDepthPyramid`（`hzb.rs:2,250-285`）。

### 5.2 差距

| 差距 | UE 做法 | Prism | 影响 |
|---|---|---|---|
| **遮挡测试的 footprint** | 矩形 4 角 min/max 归约，保守覆盖整个屏幕矩形 | 单 mip 4 角点最小深度（`hzb_visibility.wesl`） | 边缘漏剔除或误剔除，大物体在近处尤其明显 |
| **候选枚举** | GPU 端 instance culling（GPUScene 遍历） | CPU 每帧双循环投影全部实例 bounds 上传（`hzb_gpu.rs:185`、`:28-40`） | 万级实例时 CPU 成本线性，且与 UE 的 GPU-driven 目标矛盾 |
| **分箱与 LOD 选择** | GPU 端 `FInstanceCullingContext` | CPU（`visibility/systems.rs:34,349`） | 同上 |
| **multi-draw-indirect-count** | 有（GPU 写 count） | 无：draw 数量用 CPU 算的 `command_capacity` 上界，空槽每帧写 0（`opaque/indirect.rs:136-142`） | 上限即实际 dispatch 数，GPU 空转 |
| **阴影视图复用** | 主视图/阴影视图共享 culling 与 GPUScene | 阴影 caster 不做任何剔除，全部 scene instance 入队；**每 draw 一个 render pass**（`shading/shadow/depth_pass.rs:101,207`） | caster 数线性放大 draw/pass 数，是最先撞墙的瓶颈 |
| **meshlet 路径** | Nanite 的 GPU-driven | bevy `MeshletPlugin` 在 pkg 下 0 处注册，`from_mesh` 离线转换被 feature 门控（`crates/bevy_pbr/src/meshlet/mod.rs:1`、根 `Cargo.toml:886-890`） | 现役路径完全没有虚拟几何 |
| **HZB 的 late 阶段是否真的跑满** | UE 有两趟 + 软/硬光栅混合 | Prism 的 late retest 存在但依赖 `OcclusionCulling` 组件；bevy 遮挡插件只产 HZB 数据不剔除（`crates/bevy_render/src/occlusion_culling/mod.rs:21,44-48`） | 默认路径可能退化为单趟 |

### 5.3 判断

可见性系统是 Prism 里**设计最接近 UE、实现最完整**的一块。但它把"GPU-driven"做成了"CPU 决策 + GPU 校验"，这与 PRA §6/§7 宣称的"GPU 驱动可见性"不一致。**要么改目标，要么改实现**，现在两边都说不过去。

---

## 6. 阴影（对照 Virtual Shadow Maps）

### 6.1 Prism 现状

三层并存：

- **bevy 基线**：PSSM 级联 + PCF + PCSS（含立方体 PCSS），**无 VSM**（`crates/bevy_pbr` grep `virtual_shadow` 为空）。
- **`prism_render_scene/shading/shadow`**：自研 PSSM + PCF + PCSS + 点光立方距离图，`shadow.wesl` 919 行，完整（`shadow.wesl:10-14,44-46,82-167`）。
- **VSM**：约 4,441 行，三阶段 `page_mark → 分配 → PCF 采样`，是四个"UE 级特性"里落地最实的。

VSM 的页标记是**真 GPU pass**：每帧 clear request bitmap + `atomicOr` 标记 + 2D 网格 dispatch（`page_mark/dispatch.rs:39,64,83`、`shaders/vsm_page_mark.wesl`）。

### 6.2 差距

| 差距 | 证据 | 与 UE 的差别 |
|---|---|---|
| **页分配在 CPU，且每帧 GPU→CPU 回读页表** | `virtual_shadow/page_table/readback.rs:1-12`（260 行） | UE 的页表常驻 GPU，分配/驱逐全在 GPU 侧，延迟为零；Prism 延迟一帧且占用回读带宽 |
| **无动态 caster 失效** | `page_table/resources.rs:104-105` 明确写 "not driven yet" | UE 用 caster epoch + swept volume 只失效相交页；Prism 只能全量或 LRU |
| **只服务主方向光** | `shaders/shading_resolve.wesl:388-404,420-422`、`resolve/bind_groups.rs:263` | UE 的 VSM 覆盖方向光/点光/聚光/面光 |
| **无 SMRT（屏幕空间近场补充）** | grep `smrt` 为空 | UE 用 SMRT 补近场接触与高频细节，VSM 本身分辨率可以更低 |
| **PFB 半径固定 1 texel** | `virtual_shadow/settings.rs:50-64`（`pcf_radius: 1`） | UE 有完整页内过滤 + 多级 PCF + 页边界处理 |
| **caster 不虚拟化** | `atlas/raster.rs:105-143` 逐实例 replay 普通几何 | UE 的 Nanite caster 走簇层级，页级剔除 |
| **默认关闭** | `shading/runtime.rs:280`（`enable_virtual_shadow` 默认 false） | 默认画质仍是 CSM，VSM 是"可选实验" |
| **无 RT 阴影后端** | §13 | UE 有硬件光追阴影替代路径 |
| **无 alpha/OMM 正确遮蔽** | `prism_micromap` 只被自身 Cargo.toml 引用 | UE 用 OMM/微图处理 alpha 阴影 |

### 6.3 判断

VSM 的**骨架是对的**（页标记、页表、atlas、receiver 都有），坏在"把 GPU-driven 的东西留在 CPU 侧"：回读页表 + CPU 分配 + 无增量失效，等于把一个为动态场景设计的系统用成了"每帧重建的静态缓存"。这三个问题不解决，VSM 的收益不会超过 CSM 太多。

---

## 7. 动态 GI 与反射（对照 Lumen）

### 7.1 最核心的发现：Prism 的 GI 全是"屏幕空间重采样"

逐个系统看语义（不是看 shader 是否存在）：

| 系统 | 实际做法 | 证据 |
|---|---|---|
| DDGI 探针 | trace_ray 是**屏幕空间深度 march**（投影到 NDC 读 `scene_depth`），不是世界/BVH 追踪 | `ddgi_probe_update.wesl:212-250` |
| World-Space GI | **完全不投射射线**：把 tile 当帧已着色辐射采进 L1 球谐探针；`cell_size` 注释自称 "forwarded for parity" | `world_space_gi_probe_update.wesl:1-14, :41, :57-60` |
| Surface Cache | **每屏幕 tile 一个 surfel** 的屏幕辐射 EMA 缓存，不是 Lumen 的参数化卡片图集，无 off-screen 表面 | `surface_cache/mod.rs:12-19` |
| SSGI | 单次弹射，只取当帧屏幕已着色色 | `ssgi.wesl:1-12`；`ssgi/mod.rs:5-9` |
| 软件 SDF 追踪 | 存在，但是**CPU golden**，注释写 "the CPU golden reference for ... the DXR/Vulkan BLAS would run inside each brick's box" | `ray_scene/sdf_brick.rs:1-28` |
| 全局 SDF 砖格/锥追踪 | 1,462 行 CPU，无 GPU twin、无消费者 | `prism_render_shading/src/gi/global_sdf/{brick_grid.rs:529行, cone_trace.rs:535行, merge.rs:398行}` |
| 硬件 Ray Query | 只有内核 + ABI 常量，自述沙盒无 GPU 故 "cannot be proven by a bit-exact parity test" | `ray_scene/hardware_ray_query.rs:1-25, :36-38` |
| Radiance Cascades | **零依赖者孤儿 crate** | `pkg/prism_radiance_cascades_gpu`（根 `Cargo.toml:108` 仅成员） |
| APV/探针网格/SH 辐照度 | 1,569 行 CPU 契约 | `gi/probe_volume/{sh_irradiance.rs:497行, sky_occlusion.rs:391行, probe_grid.rs:349行}` |

**结论：Prism 的 GI 语义等价于"增强型 SSGI"。** 屏幕外没有任何信息、没有多次弹射、没有世界空间光线。这与 Lumen 的差距是**架构性的**，不是调参能追上的。

### 7.2 ReSTIR 与降噪：部分真实

- **ReSTIR DI 是真的**：`world_restir_{visible_points,inject,seed,fill,composite}.wesl` 共 2,499 行，SHARC 空间哈希 reservoir 结构完整。
- 但**种子只来自屏幕 tile 中心像素，seed 内没有任何光源可见性测试**（`world_restir_seed.wesl:1-10`，`grep shadow|visibility` 只命中一条注释）。UE 的 ReSTIR DI / MegaLights 的核心是共享可见性，缺了它，阴影只能靠 GBuffer 猜。
- **降噪是真的**：SSGI 降噪、镜面 `history_clamp/reproject/spatial`、SSR temporal 都有 GPU twin。
- **ReSTIR GI / ReGIR / 随机聚类光照**共 6,562 行 CPU golden，shaders 目录无对应 twin（`arch/lighting/{restir_gi.rs:1090行, regir.rs:709行, restir_gi_resolve.rs:999行, stochastic.rs:490行}`）。
- **镜面 GI 缺 trace pass**：模块自述 "Pipeline wiring (trace reservoirs, resolve, MIS) arrives in the following blocks"（`spec_gi/mod.rs:3-9`），现有只有 lobe/reservoir/reuse/spatial/brdf_mis/composite。
- **所有 GI 子系统默认关闭**：`world_space_gi/settings.rs:59`、`world_restir/settings.rs:93`、`ddgi/settings.rs:112`、`surface_cache/settings.rs:76` 全是 `enabled: false`。

### 7.3 反射：SSR 最完整，其余是契约

- **SSR 完整**：visibility-buffer 重建输入 → 反投影光线沿 reverse-Z HZB 最近深度金字塔 march → 命中重投影上一帧颜色 → 置信度混合预滤波 IBL（`ssr.wesl` 458 行、`ssr/hzb.rs:1-8`）。对比 bevy 的 SSR 只是线性步进+二分求根（`crates/bevy_pbr/src/ssr/raymarch.wesl:92-128`）——Prism 这一块**确实领先 bevy 基线**。
- 反射探针/球谐：bevy 侧有烘焙 light probe（环境图 + 辐照度体，`bevy_pbr/src/light_probe`）。
- Prism 侧 `reflection_probe`(402 行)、`stochastic_ssr`(791)、`planar_reflect`(1,385) 全是 CPU golden。
- 无统一反射仲裁（SSR ↔ 探针 ↔ RT 三段降级），SSR 只与 IBL 混合。

### 7.4 光追：服务存在，结构不存在

- GPU 软件 BVH 遍历服务完整且**有真机对拍**：`tlas_traverse` 712 行 / `ray_traverse` 372 / `ray_footprint` 136。
- 但模块级 `allow(dead_code)`，自述 "its first render-graph consumer ... is not yet wired"（`raytrace/mod.rs:69,73-75`）。
- **全仓没有任何 `create_blas`/`create_tlas`**：`AccelerationStructure` 只作为内存/描述符堆的资源类枚举出现（`memory/mod.rs:31`、`descriptor_heap/heap.rs:239,244`）。即便硬件 ray query 内核已写好，也**没有可查询的加速结构**。
- `reference_pt` 是纯 CPU 离线 oracle（42 文件、0 shader），不是在渲染器里可切换的参考模式。
- `bevy_solari`（官方 RT GI + PT）在仓库里但默认 feature 关闭，只有两个 example 使用（`crates/bevy_internal/Cargo.toml:283`、根 `Cargo.toml:1767-1784`）。

### 7.5 体积：雾最弱

- **froxel 雾**：scatter→integrate→apply 三 pass，但 **visibility 硬编码 1.0**（`volumetrics.wesl:267-268`，注释原文 "No shadow map in the standalone fog path"），且无时间重投影（`shading/volumetrics/dispatch.rs:1-20`）。
- **体积云反而完整**：1,110 行，光空间 AVSM 云阴影 march + 多散射 LUT + HG/Draine 双瓣（`volumetric_clouds.wesl:26-28, :576, :588, :645`）。
- 云的时间重投影/棋盘/1/4 分辨率重建只有 CPU 契约（`arch/volumetric/temporal.rs:1-20`）。
- bevy 基线雾会采样方向光阴影视图（`volumetric_fog.wesl:2-10,32-34`、`render.rs:468`），但只是屏空间 raymarch 单次散射，无 froxel、无多散射 LUT。

### 7.6 判断

GI 是 Prism 与 UE 差距**最大且最难补**的一块，原因不是算力，而是**缺少世界空间光线**：

```text
Lumen = 世界表示（SDF/卡片/BVH）
      × 追踪（screen + world + HW）
      × 缓存（surface cache + radiance cache）
      × 降噪（时空）
      × 多弹射
```

Prism 已有第 4 项（降噪）与第 2 项的屏幕部分，第 1、3、5 项或为 CPU 契约、或为屏幕近似、或完全缺失。这就是为什么代码量很大但一谈画质就露怯。

---

## 8. 直接光与海量光源（对照 clustered lighting + MegaLights）

### 8.1 UE 的做法

- **5.0–5.4**：clustered deferred lighting + light grid，per-light shadow map；forward+/deferred 混合材质。
- **5.5 MegaLights（实验）**：每像素对上千光源做**随机采样**，阴影按需追踪（VSM 或 RT），时空复用（ReSTIR 式）+ 降噪。核心不是"快"，而是**把光源数量与阴影数量解耦**。
- 配套：rect light、IES profile、light function、light channel、light grid 的 GPU 构建。

### 8.2 Prism 现状

- **bevy 基线**：clustered forward+，**GPU 集群分配**（`crates/bevy_pbr/src/cluster`），每光源一套 shadow。
- **`prism_render_scene/lighting/cluster`**：CPU golden 集群分配（`build.rs:1-25` 引 `prism_render_shading::assign_lights_to_clusters`），再打包成三个 GPU 资源。**没有 GPU 侧 light culling**。
- **`shading/world_restir`**：ReSTIR DI 的 SHARC 结构完整，但 seed 只取屏幕 tile 中心像素、**无光源可见性测试**，且默认关闭（§7.2）。
- **光源类型**：点光/方向光/聚光。grep `rect_light`、`ies`、`light_function`、`light_channel` 全空。
- **探针**：`lighting/probe.rs:1-9` 只从 `EnvironmentMapLight` 解码，不追踪场景。

### 8.3 差距

| 差距 | 影响 |
|---|---|
| **无 MegaLights 等价物** | 上千动态光源时，Prism 只能"每光源一套 shadow map"，O(lights) 直接崩；UE 5.5 已能把 1000+ 光源当常规场景 |
| **集群分配在 CPU** | 每帧 O(lights × clusters) 的 CPU 打包；而 bevy 基线本来就是 GPU 分配——render_scene 在这里**相对 bevy 是回退** |
| **光源类型窄** | 无矩形光（室内 GI 的关键）、无 IES、无 light function、无 light channel；UE 的这些直接决定美术表达 |
| **阴影与光源数量强耦合** | 没有"按需可见性"路径，阴影成本随光源线性增长（§6 又放大了这一点） |
| **无光照函数/投影贴图** | 风格化与电影化布光手段缺失 |

### 8.4 判断

海量光源是 UE 5.5 之后**新的画质/规模门槛**（城市夜景、大量发光体、竞技场）。Prism 在这条线上只有"CPU 集群 + 每光源阴影"，属于 5.0 时代形态，且实现位置比 bevy 还靠 CPU。

---

## 9. 材质与着色（对照 Substrate）

### 9.1 UE Substrate 的模型

- 材质 = **BSDF 算子图**（slab）：每个 slab 有 coverage 权重，运行时逐像素求值；层与层可以任意混合、遮蔽、透射。
- 表达力：三层清漆叠半透明基材、程序化遮罩混合三种 BSDF、体材质 slab —— 都能直接表达。
- 代价：运行时分支多、寄存器压力大，UE 用 slab 编译 + 动态分支 + 限制最大 slab 数来控制。
- 配套：材质图编辑器、Material Layers、参数集合、材质实例、shader permutation 烹饪。

### 9.2 Prism 现状：一个"更工程化、但表达力更窄"的模型

Prism 的材质模型不是 Substrate 的算子图，而是 **OpenPBR 超集 + 声明式 schema + 编译期单态化**：

- **OpenPBR 超集且可机检**：`prism_render_material/src/openpbr.rs:1-13`，`openpbr_conformance` 做**字节恒等**校验（`:162`），`project_to_openpbr` 返回被丢弃的瓣列表（`:118`、`DroppedLobe` `:43`）。
- **10 个瓣 + 位区隔离**：KHR 位区 0–6、EXT 位区 16/17/18（`surface_layout.rs:36,65,76,80`），`SCHEMA_VERSION=2`、`ABI_VERSION=8`、变长瓣打包（每瓣 4/8/16 字，`SURFACE_MAX_WORDS=104`，`:11,13,33,39`）。
- **BRDF 数值是真的**：`brdf.wesl` 1,503 行，含多重散射能量补偿、EON/Oren-Nayar、各向异性 GGX+Smith、色散、薄壁、glint（`:593,627,663,731,459,386,842`）。
- **PCM（Prism Capability Model）**：把所有材质扩展点（瓣、参数通道、响应模式、降级策略、组合 pass…）收进**一个声明式注册表**，且**只在烘焙/加载期运行**，运行时只剩单态化的 WESL + 定宽 `u64` 摘要（`prism_material_pcm/src/lib.rs:1-24`）。这是对 UE 的 permutation 爆炸的一种回答。
- **动态通道**：wetness/snow/damage/dissolve 被移出排列空间，运行时只重上传 4 个 word（`material.wesl:25-33`）。

### 9.3 差距

| 差距 | 证据 | 性质 |
|---|---|---|
| **无 Substrate 式 slab/层混合** | `ClosureKind` 只有 `Layer`/`Mix` 两个枚举值，normalize 把它们折成一个 feature 位就丢弃语义（`prism_render_material/src/ir.rs:18-33,132-134`） | 模型级：无法表达"三层清漆叠半透明基材"这类材质 |
| **材质分层只有 CPU 烘焙** | `bake.rs:254 bake_layer_stack`（LayerParams/LayerStack），coverage 加权写进记录/纹素，运行时无逐层混合 | 表达力 + 编辑体验 |
| **材质图不运行时求值** | `ir.rs:100 normalize` 是编译期死节点裁剪 + mask 派生，落到定长参数块；authoring evaluator 自承"非 MaterialX 运行时"，纹理/程序化节点直接 `UnsafeConnection`（`evaluator.rs:1-7,11`） | 材质编辑器不可能在这种 IR 上做实时预览 |
| **无排列预算/上限** | 只有 u64 溢出拒绝，无 variant 数上限、无按需回退（`shader_package/permutation.rs:57`） | 大规模内容下排列会失控 |
| **无材质实例/参数集合** | 未见 MaterialInstance/ParameterCollection 等价物 | 美术工作流缺失 |
| **无 DBuffer/延迟贴花** | bevy 有 forward+clustered decal，Prism 侧无 | 与 Substrate 的层混合配套缺失 |

### 9.4 但 Prism 有一处真正领先的地方

**vis buffer → 材质分类 → 间接 dispatch → megakernel resolve** 是**真实 GPU compute 链**，不是纸面契约：

- `visibility.wesl`（272 行）+ `visibility_raster.wesl` 产 vis buffer；
- `shading/classification_gpu.rs:26`（`WORKGROUP_SIZE=64`）、`:99` 建 pipeline、`:141-153` 排 classify_count/prefix_classes/scatter_work；
- resolve 侧每个 shading class 一次 `dispatch_workgroups_indirect`（`shading/resolve/dispatch.rs:1-20`）；
- megakernel 内容真实：簇光照、VSM 阴影、IBL(SH+prefilter)、LTC 面光（`shading_resolve.wesl:573,639,870,911,1096`）；
- 分类结果与 CPU 计划在真实 wgpu 设备上逐项对账（tests/integration/material_classification_gpu.rs:633-724）。

这就是 UE Nanite material resolve 的同构架构。**它写对了，只是没有应用跑它。**

---

## 10. 纹理与流送（对照 VT / SVT / RVT）

### 10.1 UE 的做法

- **SVT / VT / RVT** 三件套 + lightmap VT；page table 常驻 GPU；feedback 在 GPU 端去重；**离线 BCn/ASTC 压缩 + 页对齐**；硬件稀疏驻留；RVT 用于地形/贴花/大面积材质混合。
- 流送与驻留统一进 streaming manager，和 Nanite 页、VSM 页共享预算。

### 10.2 Prism 现状

- `prism_virtual_texture_gpu`：1,718 行、3 个 shader（`page_lookup` / `page_resolve` / `feedback_decode`），页表 + feedback，**feedback 回读在 CPU**（`src/feedback/readback.rs` 152 行）——与 VSM 页表同样的反模式。
- `prism_image`：只有解码/上传路径，**没有 BCn/ASTC 编码器**（grep `bc7`/`bcenc`/`Compressor` 全空）。
- 无 RVT、无 SVT；`render_app` 里还有第三套 `FixedPagePool`（§3.5）。
- bevy 基线**完全没有虚拟纹理**。

### 10.3 差距

| 差距 | 影响 |
|---|---|
| **无离线压缩管线** | UE 的 VT 页是 BC 压缩的；Prism 若走未压缩页，显存与带宽是 UE 的数倍，Nanite/VSM 省下来的预算会被吃回去 |
| **无 RVT** | 地形、贴花、大面积程序化混合只能传统 splatting，层数一多就爆带宽 |
| **无 SVT** | 无法流送超大单张贴图（地标、海报、扫描资产） |
| **feedback 回读 CPU** | 缺页响应延迟一帧以上，快速转头时明显掉页 |
| **无硬件稀疏页** | wgpu 无 sparse residency，indirection 采样要多一次依赖纹理查询，采样成本高于 UE |
| **三套页池不共享预算** | thrash 时互相抢占，没有全局优先级 |

### 10.4 判断

纹理是"不显眼但决定 3A 体量"的地方。Prism 有页表与 feedback 的骨架，缺的是**压缩管线**（内容侧）与**GPU 常驻页表**（运行时侧）。没有压缩，虚拟纹理的意义（省显存）就先被自己的未压缩页抵消。

---

## 11. 后处理 / AA / 上采样（对照 TSR + DLSS/FSR/XeSS）

### 11.1 现状：铺得很宽

- **bevy 基线后处理齐全**：bloom / DoF / motion blur / 自动曝光直方图 / effect stack；10 种 tonemapper（含 ACES/AgX/PBR-Neutral/TonyMcMapface），`crates/bevy_render/src/view/mod.rs:283-330`、`crates/bevy_core_pipeline/src/tonemapping/mod.rs:220-246`。
- **Prism 侧 shader 覆盖**（`prism_render_scene/src/shaders`）：`bloom`、`dof`、`motion_blur`、`exposure`、`color_grade`、`film_grain`、`gamut_map`、`chromatic_aberration`、`lens_flare`、`vignette`、`kuwahara`、`halftone`、`hatching`、`posterize`、`ordered_dither`、`glint`、`outline`、`thin_film`、`specular_aa`、`subsurface`、`sky_*` LUT，外加 `upscale_rcas` / `upscale_reconstruct`。
- **Prism 侧 tonemap 的 CPU 参考实现**同样到位（ACEScg 矩阵 + RRT/ODT fit、AgX 全链，`prism_render_shading/src/tonemap.rs:133,240,247`）。
- **运动矢量真实生成**（有/无两套 view layout + prepass flag，`crates/bevy_pbr/src/prepass/mod.rs:253-254,552-557`）。

### 11.2 但时域这一层是空的

| 差距 | 证据 | 影响 |
|---|---|---|
| **TSR 级时域超分没接线** | `upscale_reconstruct.wesl:275` 是 `@compute(8,8,1)`，含 lanczos2（`:130`）、variance box（`:178`）、depth disocclusion（`:202`）、lock update（`:240`）——但零消费者 | 4K 上采样方案等于没有 |
| **基线只有双线性 blit** | `crates/bevy_core_pipeline/src/upscaling/mod.rs:38`（`ViewUpscalingPipeline` 只是个 blit） | 低分辨率渲染必糊 |
| **TAA 是教科书基础版** | YCoCg 邻域 AABB clip + blend 0.1/0.015；无深度导数 disocclusion、无 LWC、无随机光照重投影（`taa.wesl:10-13,58`） | 移动相机时的残影/闪烁是"看得出是自研"的第一触点 |
| **无 FSR / XeSS** | 全仓只出现在注释里（`upscale/mod.rs:1`、`motion/mod.rs:6`） | 非 DLSS 硬件上无高质量上采样选项 |
| **DLSS 默认关且限 raw Vulkan** | `crates/bevy_anti_alias/Cargo.toml:16,38`、`default_plugins.rs:41-42` | 主力路径用不上 |
| **无局部曝光** | `architecture/display/mod.rs:41` 只有一个 `local_exposure: bool` 字段 | 室内外反差大的场景会比 UE 明显更"平" |
| **色彩管理只有 Rec709** | `display/transfer.rs:1-11`（自述 "intentionally performs no transcendental math"）、`:41 REC709`、`:54 REC2020` | HDR/宽色域输出、OCIO 工作流缺失 |
| **OIT 只有逐像素链表** | `crates/bevy_core_pipeline/src/oit/mod.rs:44`、`:36 TODO`（未做 WBOIT/Moment/depth peeling） | 半透明排序问题在复杂场景暴露 |

### 11.3 判断

后处理是典型的"数量掩盖质量"：Prism 有 20+ 个后处理 shader，但**决定画面稳定性与清晰度的两件事——时域抗锯齿质量与时域超分——恰好是最弱的两件**。UE 的画面"稳"，很大程度是 TSR 的功劳；Prism 在这条线上等于 5 年前的水平。

---

## 12. 专用管线：毛发 / 布料 / 水体 / 地形 / 粒子 / 大气 / 植被

| 子系统 | UE 形态 | Prism 现状 | 差距性质 |
|---|---|---|---|
| **毛发** | Groom：导入/绑定/LOD/物理/渲染/RT 一体；strand-based shading、dual scattering、deep opacity、与 Nanite/VSM 集成 | `prism_hair_gpu` 28,382 行 / 87 shader：dual scattering、deep opacity、adaptive transmittance、cosserat 物理、cluster cull、DFTL 等，**0 消费者** | 内核很全，但无导入/绑定/LOD/集成；且不在任何渲染器里 |
| **布料** | Chaos Cloth：与骨骼网格绑定、碰撞、风、LOD、GPU 求解 | `prism_render_scene/cloth`：VBD/CCD/backstop/aero + GPU parity 测试 | 物理与求解到位，但 cloth 的**渲染**（各向异性、绒感、透光）没有独立 BSDF 校准 |
| **水体** | Water：river/ocean/underwater、与 Lumen/阴影/雾耦合、FFT + 泡沫 + 折射 | `prism_render_scene/water`：FFT ocean（`fft_upload.rs`）+ FLIP + surface 渲染 / VSM / SSR，16 shader | 单体完整，缺"水体作为场景元素"的集成（反射探针、GI、体积雾交互） |
| **地形** | Landscape：分层材质、RVT、LOD、碰撞、植被散布 | `prism_world_terrain`：`no_std`，只有 bake/coordinates/cube_sphere/height 契约，**无渲染** | 只有数据与烘焙，没有 Landscape 的渲染形态 |
| **粒子** | Niagara：GPU 模拟 + 渲染 + 碰撞 + 事件 + 与光照/GI 交互 | **pkg 下没有任何粒子渲染 crate**；bevy 侧未集成 hanabi | 完全缺失（最大的功能空洞之一） |
| **大气** | Sky Atmosphere：Bruneton 模型、LUT、与体积云/雾耦合 | bevy `bevy_pbr/atmosphere`（LUT 式） | 有基础，但与本仓库的体积云/雾是两套散射模型，未统一 |
| **植被** | Foliage + Nanite foliage + 风 + LOD dither | 无专用系统；Nanite 也不支持植被 | 缺失 |
| **贴花** | Deferred Decal + DBuffer + RVT 贴花 | bevy `bevy_pbr/decal`（forward + clustered） | 有基础；Prism 侧无 DBuffer |
| **透明/OIT** | 排序 + OIT + 折射 + 体积透明 | `architecture/transparency/moment_oit`（CPU 契约）、`render_scene/shading/transparent`（1,380 行，forward + composite） | 有骨架；Moments OIT 未上 GPU |

**共同问题：这些子系统几乎全是 island。** UE 的这些系统是"内容工具 + 运行时 + 物理 + 光照"四位一体；Prism 是把每个系统的算法内核做到 80 分，然后放进一个没有宿主的 crate 里。毛发没有 groom 导入就没有内容，水体没有场景集成就不反射 GI，粒子根本不存在。

---

## 13. 光追 / 路径追踪 / 离线验证（对照 UE Ray Tracing + Path Tracer）

### 13.1 UE 的形态

- 硬件光追阴影/GI/AO/反射是**产品路径**（不是实验开关）：BLAS/TLAS 增量构建、RTPSO、SER（5.4）、OMM（近年）、与 VSM/Lumen 的统一调度。
- Path Tracer 是**编辑器内的参考渲染器**：用来验证 GI 收敛、材质能量守恒、降噪偏差，且可对同一场景出对照图。
- Nanite 有专门的 RT 代理网格（简化版，用于加速结构）。

### 13.2 Prism 现状

- **软件 BVH 遍历服务完整且有真机对拍**：`tlas_traverse` 712 行 / `ray_traverse` 372 / `ray_footprint` 136，但模块 `allow(dead_code)`，自述 "its first render-graph consumer ... is not yet wired"（`raytrace/mod.rs:69,73-75`）。
- **加速结构构建完全缺失**：全仓无 `create_blas`/`create_tlas`，`AccelerationStructure` 只作为内存/描述符堆的资源类枚举（`memory/mod.rs:31`、`descriptor_heap/heap.rs:239,244`）。硬件 ray query 内核（`hardware_ray_query.rs`）**没有可查询的结构**。
- **配套 GPU 岛都没接线**：`prism_micromap_gpu`、`prism_dmm_gpu`、`prism_ray_reorder_gpu` 合计约 2.1K 行，消费者全为 0。
- **bevy_solari**（官方 RT GI + ReSTIR DI/GI + 世界缓存 + DLSS-RR + PT）在仓库里，但默认 feature 关闭，只有两个 example（`crates/bevy_internal/Cargo.toml:283`、根 `Cargo.toml:1767-1784`）。
- **`reference_pt` 是纯 CPU 离线 oracle**（42 文件、0 shader），不是在渲染器里可切换的参考模式（`reference_pt/mod.rs:1-8`）。

### 13.3 差距

| 差距 | 后果 |
|---|---|
| 无 BLAS/TLAS 构建 | 所有 RT 路径（阴影/GI/反射/OMM/微图）都不可用，即便 shader 内核已写好 |
| 无 Nanite RT 代理 | 即使补上加速结构，Nanite 几何也无法进 RT |
| 无 SER / Work Graphs | 光追路径的调度效率与 UE 5.4+ 差一个数量级（wgpu 也不支持） |
| 无引擎内 Path Tracer | 没有 ground truth → GI 收敛、能量守恒、降噪偏差都无法验证（这正是 §7 那些近似实现无法被证伪的原因） |
| 无离线渲染 | UE 有 Movie Render Queue + 离线 GI 路径；Prism 的 `offline` 只有设计 |

### 13.4 判断

光追是 Prism 里**"看起来有、实际为零"**的典型：有 712 行的 TLAS 遍历、有 ray query 常量、有 micro-map/DMM/ray-reorder 三个 GPU 岛，但**没有加速结构**，所以整条链在物理上不可运行。更关键的是缺少引擎内 Path Tracer —— 没有参照物，前面章节里所有 GI 近似都无法被判定"够不够好"。

---

## 14. 工具链 / 着色器编译 / 诊断（对照 ShaderCompileWorker + PSO Precaching + stat GPU）

### 14.1 现状

| 能力 | UE | Prism | 证据 |
|---|---|---|---|
| 离线着色器编译 | ShaderCompileWorker + DXC + DDC | **无**：`pkg/*` 里 `dxc/glslang/shaderc/naga-cli` 零命中 | 全仓只在 bevy 的 Cargo.toml 出现 |
| SPIR-V 产物 | 编译产物进包 | `shader_package` **只有元数据没有产物**（自述 "build-time HLSL→SPIR-V package metadata"） | `shader_package/mod.rs:1-2`、`variant_manifest.rs:1-20` |
| 文本预处理 | — | `prism_shader` 是 1,517 行的 C 风格 `#ifdef` 组合器，**零消费者**，与 WESL 的 `@if(...)` 并行不接 | `prism_shader/src/preprocess.rs:7,131` |
| PSO 预缓存 | 录制→烘焙→进包→首帧零卡 | `pso_cache` 全套（指纹/预热计划/LRU/持久化/逐 draw 准入）是**纯 CPU 模型**，自述编译/线程/磁盘 I/O 留给"不存在的 engine-side consumer" | `pso_cache/mod.rs:1-13`、`warm_runtime.rs:120,83` |
| 驱动级持久缓存 | `wgpu::PipelineCache` 等价物 | **现役 bevy 路径无**：`crates/bevy_render` 全域无 `create_pipeline_cache`；`prism_render_app/src/pipeline_cache.rs` 有真有磁盘持久化，但它在零消费者的第三条栈上 | `render_resource/pipeline_cache.rs`（1,090 行，只有 `CachedPipelineState`） |
| 排列预算 | 排列规模可管 | 只有 u64 溢出拒绝，无 variant 上限 | `shader_package/permutation.rs:57` |
| 热重载 | 有 | 有（AssetEvent<Shader> → 重编） | `pipeline_cache.rs:950-953` |
| GPU pass 计时 | stat GPU + 逐 pass timestamp | 弱：无统一 per-pass GPU 计时 | — |
| 缓冲可视化 | Buffer Visualization（几十种视图） | 无 | — |
| GPU 崩溃诊断 | Aftermath/GPU crash dump | 无 | — |
| 帧捕获 | RenderDoc 集成 + 内部 capture | capture 只有头部 + ABI 校验，无消费者 | `architecture/src/capture/mod.rs:44-64` |

### 14.2 净后果

1. **冷启动 stutter 在两条栈上都没解决**：bevy 侧无 `PipelineCache`，Prism 侧的 warm set 没有任何代码去驱动真实编译。
2. **shader permutation 失控风险**：没有预算，内容一多，编译时间与包体都不可控。
3. **性能回归不可归因**：没有 pass 级 GPU 计时，就只能靠外部工具逐帧抓；UE 的 `stat GPU` 是团队级能力。
4. **没有"看中间缓冲"的能力**：调 GI/阴影时无法直接看 SSAO buffer、VSM 页表、vis buffer、SH 探针，这会把每个问题的定位时间乘以 5–10。

---

## 15. 平台与可伸缩性（对照 scalabilities + 主机 + 移动 + VR）

| 维度 | UE | Prism | 判断 |
|---|---|---|---|
| 主机 | PS5/XSX/Switch 原生 RHI | 无（wgpu 抽象层之外无入口） | 商业 3A 的硬门槛 |
| 原生 API | D3D12/Vulkan/Metal 全量 | 无，`backend/mod.rs:3-4` 明说只有一个后端 | 见 §3.1 |
| 移动 / TBDR | 专门的 mobile renderer + subpass + tile | 无 pass 合并（§3.3），TBDR 无法利用 | 移动端性能会差一档 |
| VR / 多视图 | Instanced stereo / multiview | 无 | 缺失 |
| 动态分辨率 | 有（配合 TSR） | 有 `AdaptiveQualityController` 契约；无时域超分配合（§11） | 半成品 |
| 显存预算 | 显式预算 + defrag + OOM 降级 | 只有 CPU 记账（`MemoryBudget`/`HeapStats`/`FrameWorkBudget`），设备查询 pending | 大场景 OOM 风险不可控 |
| 帧节奏/VRR | 有 | 无 | 缺失 |
| 热/功耗 | 有（移动） | 无 | 缺失 |

**判断：** "能力分层 + 自适应质量"在设计文档里写得很完整（PRA §13），但落地依赖两件没做的事：**时域超分**（低分辨率渲染的前提）与**统一驻留预算**（内存压力的仲裁者）。没有这两件，scaler 无法真正"降档而不降观感"。

---

## 16. 内容管线与工具（对照 Editor + DDC + Cooking）

### 16.1 差距清单

| 能力 | UE | Prism |
|---|---|---|
| 编辑器 | 完整（关卡/材质/蓝图/Niagara/Sequencer） | 无（只有 `bevy_dev_tools` 与 `prism_ui_component_kit` 的组件库；无编辑器 crate） |
| 材质编辑器 | 图形化 + 实时预览 | 无；`material_authoring` 的 evaluator **明确不是运行时**，纹理/程序化节点报 `UnsafeConnection`（`evaluator.rs:1-7,11`） |
| Nanite 构建 | 切簇/简化/误差/DDC | **无构建器**（§4.3） |
| 自动 LOD | 有 | 无（`prism_mesh` 有 LOD 基础设施，但无自动生成） |
| HLOD | 有（World Partition + HLOD builder） | 无（只有 ECS 侧 `hlod_pyramid` 诊断） |
| 光照烘焙 | Lightmass + GPU Lightmass（可选） | 无离线烘焙；GI 全运行时（且 §7 那些运行时 GI 还是屏幕近似） |
| 世界分区 | World Partition + streaming | `prism_world_stream/persist/residency/io` 有契约层，无渲染器侧集成 |
| DDC | 有 | 无 |
| 资产烹饪 | 有（平台相关打包） | `prism_asset_build_core` 有构建框架，无渲染资产（VT 页/压缩纹理/簇数据/PSO warm set）的烹饪 |
| 离线渲染 | Movie Render Queue / Path Tracer | `offline` 只有设计文档 |

### 16.2 这条差距为什么致命

"UE5 同等级画质"是**内容驱动的结论**：同样的技术，在 UE 里靠材质图、LOD 链、HLOD、光照烘焙、世界分区把 100 平方公里的场景压进 60fps；Prism 没有这些工具，就意味着：

1. **无法制备能体现技术水平的场景**——每个 demo 都得手写代码生成几何与材质。
2. **无法验证"同级"**——没有对照场景，就没有对照结论。
3. **无法交付**——即使运行时做对了，美术也接不上手。

这就是为什么 §17 的验收矩阵在当前状态下**不可执行**。

---

## 17. 验收矩阵的可执行性：PRA §21 现在一条都跑不了

`docs/prism_rendering_architecture_zh.md` §21 列了一组验收项（几何/阴影/GI/材质/性能/平台）。按 §1 的档位逐条映射后：

| 验收类别 | 设计中的判据 | 当前可执行性 | 原因 |
|---|---|---|---|
| 虚拟几何 | 簇 cut 正确、页驻留、软/硬光栅混合、每帧吞吐 | **不可执行**：无构建器、无宿主、无吞吐数据 | §4.3 |
| 可见性 | 两趟 HZB 保守性、逐视图复用 | 部分可执行（单 crate 测试），无整帧 | §5.2 |
| 阴影 | VSM 页命中率、失效正确性、与 CSM 的画质差 | 不可执行：无动态失效、默认关闭 | §6.2 |
| GI | 与 Path Tracer 对照的收敛/能量守恒/泄漏 | **不可执行**：无引擎内 PT、无世界空间追踪 | §7、§13 |
| 材质 | OpenPBR 一致性、vis buffer 分类正确性 | 部分可执行（字节级 conformance、真机分类对账） | §9 |
| 性能 | 帧预算、显存峰值、首帧卡顿 | 不可执行：无 pass 计时、别名不分配、无 PSO 预热 | §3.3、§14 |
| 平台 | 主机/移动/VR 覆盖 | 不可执行：无后端 | §15 |

**结论：验收矩阵不是"没达标"，而是"没有可运行的判据"。** 这比达标率低更严重——它意味着团队无法知道自己离目标多远。

---

## 18. 建议的优先级路线

排序原则：**先让一件事可度量，再让画质可增长**。不建议同时推进三条栈。

### P0（0–3 个月）：把一条栈跑成"一天"

1. **选 `prism_render_scene` 作为产品栈并接线**：写一个 `apps/prism_viewer`（或 example），装配 `PrismGpuScenePlugin` + vis buffer + classification + resolve + 后处理 + present，跑通 `scene → frame`。
   - 注意：它已自述是"Bevy render sub-app 上的附加层"（`plugin.rs:34`），先以插件形态接，**不要**先做 RHI 移植（S1–S8 那条线会拖死进度）。
2. **建立画质/性能基线**：3–5 个 golden scene（室内/室外/毛发/水/大量光源）+ 图像 diff + 逐 pass GPU 计时 + 显存统计。
3. **冻结/归档**另外两条栈里不参与本栈的部分（尤其是 `prism_render_architecture` 的纯契约模块），避免继续摊薄。

### P1（3–9 个月）：补齐"看得见的画质"

4. **接世界空间追踪服务**（`raytrace/*` 已有真机对拍）+ **BLAS/TLAS 构建**。这一步同时解锁：Lumen 式 GI、RT 阴影、RT 反射、Path Tracer 参考模式。
5. **VSM 上 GPU**：页表常驻 + caster epoch 失效 + 多光源页（§6.2 的三条）。
6. **时域超分上线**：`upscale_reconstruct.wesl` 接线 + TAA 升级（disocclusion/LWC）；同步接 FSR（开源、wgpu 可行）。
7. **纹理压缩管线**（BCn/ASTC 离线编码）+ page 格式对齐（§10.3）。

### P2（9–18 个月）：规模与内容

8. **Nanite 构建器**（切簇/简化/误差/水密/属性保持）+ 页流送接入画面。
9. **GPU 常驻页表与统一 resident manager**（几何/纹理/阴影三套合一）。
10. **PSO 预热闭环 + 离线 shader 编译**（§14）。
11. **编辑器与材质图**（至少一个能实时预览的材质编辑前端）。
12. **海量光源**（MegaLights 等价物）。

### 明确不建议现在做的

- 原生 Vulkan/D3D12 后端（在单栈基线跑通前，投入产出比极低）。
- Work Graphs / SER / mesh shader（wgpu 不支持，先接受软件模拟）。
- 主机移植（同上）。

---

## 19. 什么才算"够好"

把 UE5 的标准压缩成四句话，Prism 当前的差距就是这四句里的空位：

1. **可运行**：存在一个应用，把 scene → visibility → material → lighting → post → present 全链跑起来，且能换场景、能量性能。→ **当前 0 条栈满足**（§2）。
2. **可度量**：同一场景、同一相机、同一后处理，出图可与 UE 对照；性能可归因到 pass。→ **当前无基线**（§3.3、§14）。
3. **可退化**：缺页→父 mip、超预算→降 LOD、无 RT 硬件→软件追踪、低端机→降档，每一级都被验证过。→ **当前只有设计**（§10、§15）。
4. **可内容驱动**：美术能用编辑器、材质图、LOD/HLOD/烘焙管线把场景做出来。→ **当前无工具链**（§16）。

### 最后一句话

Prism 的渲染系统**不是"功能不够多"，而是"功能太多、系统太少"**：120 万行代码里，UE 级机制几乎都有对应物（有的还做得相当扎实，如 vis-buffer 分类链、HZB 两趟、SSR/HZB、ReSTIR DI 的 SHARC 结构、OpenPBR 字节级一致性），但缺少把它们组织成一个**可运行、可度量、可退化、可内容驱动**的产品的那一层。补上这一层，画质差距会从"架构性"迅速收敛到"调优性"；补不上，继续加代码只会把这个差距藏得更深。

---

## 附录 A：证据复核命令

> 说明：文中行数为 `.rs` 文件 `wc -l` 之和（含测试与注释），shader 数为 `.wesl`/`.wgsl` 文件数。全部命令在仓库根执行。

**1. 消费者计数（本文 §2.2 表的来源）**

```bash
for c in prism_render_scene prism_render_app prism_virtual_geometry_gpu \
         prism_volumetric_gpu prism_hair_gpu prism_water_render \
         prism_virtual_texture_gpu prism_radiance_cascades_gpu \
         prism_micromap_gpu prism_dmm_gpu prism_ray_reorder_gpu \
         prism_render_shading prism_render_visibility prism_render_gpudriven; do
  n=$(grep -rl "^$c *=" --include=Cargo.toml . | grep -v "^./pkg/$c/" | wc -l | tr -d ' ')
  echo "$c consumers=$n"
done
```

**2. 唯一装配新 RHI 的可执行**

```bash
grep -rn "prism_render_runtime::WgpuRuntime" --include=*.rs .
# 期望：仅 tools/render_triangle/src/main.rs:21
```

**3. 渲染代码规模**

```bash
for c in prism_render_architecture prism_render_scene prism_volumetric_gpu \
         prism_render_shading prism_hair_gpu prism_render_material \
         prism_virtual_geometry_gpu prism_render_driver prism_render_graph; do
  printf "%-32s %8s lines  %3s shaders\n" "$c" \
    "$(cat pkg/$c/src/**/*.rs pkg/$c/src/*.rs 2>/dev/null | wc -l | tr -d ' ')" \
    "$(ls pkg/$c/src/shaders/*.wesl pkg/$c/shaders/*.wesl 2>/dev/null | wc -l | tr -d ' ')"
done
```

**4. 关键证据点（可直接跳转）**

| 论点 | 位置 |
|---|---|
| bevy 帧图退化为空标签 | `crates/bevy_render/src/renderer/mod.rs:48` |
| 别名只算不做 | `pkg/prism_render_graph/src/graph.rs:1152-1190`、`src/plan.rs:181` |
| 单队列单次 submit | `pkg/prism_render_graph/src/graph.rs:1052,1082` |
| 异步 compute 只是标签 | `pkg/prism_render_graph/src/compile/schedule.rs:68` |
| 并行录制默认关闭 | `pkg/prism_render_graph/Cargo.toml:18`、`src/parallel.rs:64-83` |
| 无原生后端 | `pkg/prism_render_architecture/src/backend/mod.rs:3-4` |
| 虚拟几何无构建器 | `pkg/prism_render_architecture/src/virtual_geometry/asset.rs:3` |
| 虚拟几何 GPU 管线零消费者 | 根 `Cargo.toml:116`（仅成员） |
| 硬件光栅只有枚举 | `pkg/prism_render_architecture/src/virtual_geometry/mod.rs:121-127` |
| 遮挡 footprint 单 mip 四角 | `pkg/prism_render_scene/src/shaders/hzb_visibility.wesl` |
| VSM 页表 CPU 回读 | `pkg/prism_render_scene/src/shading/virtual_shadow/page_table/readback.rs:1-12` |
| VSM caster 失效未驱动 | `.../virtual_shadow/page_table/resources.rs:104-105` |
| VSM 默认关闭 | `pkg/prism_render_scene/src/shading/runtime.rs:280` |
| 阴影视图不剔除、每 draw 一 pass | `pkg/prism_render_scene/src/shading/shadow/depth_pass.rs:101,207` |
| DDGI trace 是屏幕 march | `pkg/prism_render_scene/src/shaders/ddgi_probe_update.wesl:212-250` |
| World-Space GI 不投射射线 | `.../shaders/world_space_gi_probe_update.wesl:1-14,41,57-60` |
| Surface Cache 是屏幕 tile surfel | `pkg/prism_render_scene/src/shading/surface_cache/mod.rs:12-19` |
| 体积雾不采样阴影 | `.../shaders/volumetrics.wesl:267-268` |
| 光追服务未接线 | `pkg/prism_render_scene/src/raytrace/mod.rs:69,73-75` |
| 无 BLAS/TLAS 构建 | 全仓 `grep -rn "create_blas\|create_tlas"` 零命中 |
| 材质无 slab/层混合 | `pkg/prism_render_material/src/ir.rs:18-33,132-134` |
| OpenPBR 字节级一致性 | `pkg/prism_render_material/src/openpbr.rs:118,162` |
| 分类链真实 GPU 实现 | `pkg/prism_render_scene/src/shading/classification_gpu.rs:26,99,141-153` |
| 时域超分内核未接线 | `pkg/prism_render_scene/src/shaders/upscale_reconstruct.wesl:275` |
| 基线只有双线性上采样 | `crates/bevy_core_pipeline/src/upscaling/mod.rs:38` |
| 无持久 PSO 缓存 | `crates/bevy_render/src/render_resource/pipeline_cache.rs`（无 `create_pipeline_cache`） |
| PSO 预缓存只是 CPU 模型 | `pkg/prism_render_architecture/src/shader_package/pso_cache/mod.rs:1-13` |
| 无离线着色器编译器 | `grep -rn "dxc\|glslang\|shaderc" pkg/` 零命中 |
| 无局部曝光 | `pkg/prism_render_architecture/src/display/mod.rs:41` |
| 色彩管理仅 Rec709 | `pkg/prism_render_architecture/src/display/transfer.rs:1-11,41,54` |

**5. 一项口径限制**

本文的"E4 产品级"判定要求"端到端出图 + 自动回归 + 降级 + 平台覆盖"。仓库中唯一端到端出图的路径是 vendored Bevy 的示例与 `tools/render_triangle`，二者都不包含 Nanite/VSM/Lumen 级特性；因此所有"Prism 自有栈已实现"的条目，最高只能记为 E2。
