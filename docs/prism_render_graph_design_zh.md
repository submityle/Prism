# Prism Render Graph 顶级次世代 AAA 级帧渲染依赖图设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **Render Graph（渲染图 / 帧依赖图，又称 Frame Graph）**：把「一帧要画什么」描述成一张 **pass（通道）× resource（资源）** 的有向无环图，由框架自动完成**依赖排序、无用通道剔除、瞬态资源分配与内存别名复用、资源状态转换与屏障插入、多队列（异步计算/传输）调度、并行命令录制、原生 RenderPass/subpass 合并与 TBDR tile 内存优化、编译缓存与增量重编译**。它坐在 `prism_render_driver`（RHI）之上、各渲染特性（material / scene / lighting / post）之下，是渲染器的「编排大脑」：渲染代码只声明「我要读谁、写谁」，何时转换、何时同步、显存怎么复用全交给图。
> 借形态不抄码。借鉴：
> - **帧图开创**：Frostbite FrameGraph（Yuriy O'Donnell, GDC 2017，瞬态资源 + 自动别名 + setup/execute 分离的范式奠基）
> - **引擎级渲染依赖图**：Unreal `RDG`（Render Dependency Graph：pass flag、RDG buffer/texture、异步计算 fence、parallel translate、pass 剔除、transient allocator）、Unity SRP RenderGraph（HDRP pass/资源虚拟化、Native RenderPass/NRP 合并、RenderGraph compiler 两代编译器）
> - **开源实现**：Granite（Themaister 的 Vulkan render graph：屏障/别名/信号量全自动、subpass 合并，附完整博客系列）、bevy `render_graph`（节点/边/slot/子图形态）、The-Forge / Diligent 的帧级编排、O3DE/Atom 的 Frame Graph、Wicked Engine 的 rendergraph
> - **工业界实践**：id Tech 7（Doom Eternal，GPU 驱动 + 显式 barrier 预算）、Decima（Horizon，frame graph + async compute 实践）、Call of Duty / Destiny 的 frame graph 分享、AMD FidelityFX/RMV（Radeon Memory Visualizer 的别名可视化思路）
> - **底层能力**：Vulkan render pass / dynamic rendering + `VK_KHR_dynamic_rendering_local_read`、`VK_KHR_synchronization2`、timeline semaphore；D3D12 enhanced barriers / split barrier / resource state；Metal `tile shader` / imageblock / memoryless / programmable blending
> 本文为纯经典图形系统编程路线，**不含任何 AI/ML 内容**（不涉及任何学习型调度；排序/别名/屏障/合并均为确定性经典算法）。

- 版本: v0.3（**编码阶段进行中**：核心编译器已实现、测试并提交；高级功能仍为 PLANNED。v0.1→v0.2 做了一次**重构深化**：拆分并深化依赖/剔除、生命周期/别名、屏障、多队列、并行录制、TBDR 六大核心算法章；新增「§17 极致性能工程」「§19 正确性·不变量·验证」「§20 易用性·可维护性·扩展性」三大横切章；高级功能章扩充 work graph / mesh-task pass / 光追加速结构构建 pass / 多 GPU 分帧 / 资源驻留流送编排；参考产品从 7 项扩至 15+ 项并逐项落到具体机制；路线图扩至 M0–M8。v0.2→v0.3（**进入编码**）：核心编译管线落地为真实代码——剔除（Kahn 拓扑 + 确定性 tie-break）、SSA 调度、生命周期区间分析、瞬态显存别名（复用 driver 的 TLSF 子分配器）、屏障规划、附件 load/store 推导，`#![no_std]` core+alloc，含 record-only mock 后端与 10 项编译/执行不变量测试，`cargo clippy`/`fmt` 零告警，已提交。）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: `prism_render_driver`（RHI：Device/Queue/资源/屏障/Fence/Semaphore/placed 子分配）、`prism_math`（视口/尺寸/打包）、`prism_utils`（句柄表/竞技场/位集/拓扑排序/区间树）、`prism_diagnostic`（GPU 计时/pass 标记/图可视化/内存图谱）、`prism_tasks`（并行录制与并行 setup/compile，§24.7 确定性归并）
- 层级定位: L4 表现层（向下只依赖 `prism_render_driver` 与 L1 地基；向上被 `prism_render_scene` / `prism_material` / `prism_lighting` / `prism_post` / `prism_ui` 使用）
- 明确约束: 声明式、无全局可变状态、构图与执行分离；一帧一张图、可完全丢弃重建；屏障与别名由框架推导且可被显式覆盖；`std` 必需；**不依赖任何 `bevy_*` crate**（`bevy_render::render_graph` 仅作形态参考，不链接）

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍（深化：逐项落到机制）
3. 核心概念：Pass / Resource / Handle / State / Graph
4. 构图 API：setup/execute 分离与 builder（深化：类型化、访问推导、人机工程）
5. 资源虚拟化：瞬态 / 导入 / 持久（深化：尺寸与用途推导、外部同步）
6. 依赖推导与无用通道剔除（深化：SSA 写版本、hazard 分类、剔除算法）
7. 资源生命周期分析与内存别名（深化：区间计算、堆分类、分配算法、别名屏障、碎片与去碎片）
8. 屏障与状态转换自动插入（深化：状态模型、访问域、批合并、split barrier、冗余消除、sync2/enhanced barriers）
9. 多队列：异步计算与传输调度（深化：timeline 信号量、关键路径列表调度、所有权转移、重叠度量）
10. 并行命令录制（深化：二级命令缓冲、parallel translate、确定性归并、负载均衡）
11. 原生 RenderPass / subpass 合并与 TBDR tile 内存优化（深化：合并判据、三后端映射、load/store/resolve 推导）
12. 执行模型：编译 → 分配 → 执行 → 回收
13. Blackboard 与跨 pass 数据流
14. 图可视化与调试（接 diagnostic）
15. 与 render_driver / shader / material / scene / streaming 集成
16. 持久化资源与跨帧状态（history buffer）
17. 极致性能工程（新增：编译缓存、增量重编译、零分配、SoA、并行编译、plan 复用）
18. 高级功能增补（AAA·深化）
19. 正确性、不变量与验证（新增：形式化不变量、别名证明、验证层、差分/属性/模糊测试）
20. 易用性、可维护性、扩展性（新增：API 工效、错误诊断、模块边界、可插拔编译器、自定义扩展点）
21. crate 分层与模块布局
22. 契约、不变量与版本化
23. 路线图（M0–M8）与基准即规格
24. 诚实边界与风险

---

## 1. 设计哲学与目标

现代渲染一帧由几十到上百个 pass 组成（depth prepass、GBuffer、阴影、SSAO、光照剔除、光照、透明、体积、SSR、后处理、UI……），每个 pass 读若干纹理/缓冲、写若干纹理/缓冲。在显式 API（Vulkan/D3D12/Metal）下，必须手动管：资源状态转换（barrier）、队列间同步（semaphore）、瞬态显存的分配与复用。如果让每个渲染特性各自硬编码这些，一帧的编排就会碎成几十处互相耦合的手工同步，改一个 pass 顺序就可能漏一个屏障 → 画面错误或 GPU 挂起；加一个 pass 就要重算一遍别人占了哪块显存。

`prism_render_graph` 的使命：**让渲染特性只声明「读谁、写谁」，把「何时转换、何时同步、显存怎么复用、哪些 pass 其实没人用、哪些 pass 能合进一个 tile pass」全部交给图自动推导**。构图（declare）与执行（execute）彻底分离——构图阶段只记录意图、不碰 GPU；编译阶段做依赖排序 / 剔除 / 生命周期 / 别名 / 屏障 / 合并；执行阶段才真正录命令。

**一句话定位**：`prism_render_graph` 是 Prism 的「帧编排大脑」——输入一组声明式 pass，输出一份被剔除、排序、别名、加屏障、分队列、可并行录制、可合并进 tile 的最优**不可变执行计划**；渲染特性永不手写屏障与瞬态显存，只描述数据依赖。

### 1.1 五条设计原则（贯穿全文）

1. **机制与策略分离**：图提供「排序 / 别名 / 屏障 / 调度 / 合并」等**机制**；具体画什么（PBR/NPR/后处理算法）是**策略**，归 `prism_render_*`。机制稳定、策略可换（呼应 `prism_rendering_architecture_zh.md` 的 PRA 第 4 原则）。
2. **正确先于快**：别名重叠、漏屏障、跨队列竞态是「静默画面错误 + 偶现挂起」，排查成本极高。所有优化必须在**编译期可证正确**或**可一键关闭回退到安全路径**之后才启用（§19）。
3. **声明即契约**：execute 只能碰 setup 声明过的资源；越权、读未写的黑板键、显式屏障不覆盖推导集，debug 下全部 panic 并指名道姓（§20.2）。
4. **确定性可复现**：相同图结构 + 相同输入 ⇒ 相同执行序、相同别名布局、相同提交序（接 `prism_tasks` §24.7 确定性归并）。对回放、录像、golden test、确定性联机至关重要。
5. **成本随变化、不随规模**：结构稳定帧复用上帧 `ExecutionPlan`，编译从「每帧算」降为「结构变才算」；热路径零堆分配（竞技场）。编译开销与「本帧改了多少」成正比，而非与「图多大」成正比（§17）。

### 1.2 四维总目标（按权重，可测）

1. **性能**：瞬态资源内存别名（峰值显存可降 30–60%）；无用 pass/资源自动剔除；屏障合并与最小化（split barrier）；异步计算与传输重叠；并行录制跑满核心；结构稳定帧编译耗时趋近 0。
2. **效果地基（正确）**：依赖推导零遗漏——读写关系唯一决定执行序与屏障；别名资源生命周期不重叠由编译期证明；跨队列同步由框架插信号量；TBDR 合并仅在可证等价处启用。
3. **易用 + 可维护 + 可扩展**：一个 pass = 一个 setup 闭包 + 一个 execute 闭包；资源不用预分配；编译器本身是一串可插拔 pass；自定义 pass 类型/资源类型/调度策略不改框架核心。
4. **可测 + 可视**：图可 dump 成 DOT/JSON；Null 后端下对剔除/排序/别名/屏障做 golden 断言；别名显存时间轴、异步重叠率、屏障数全部可视化并进 CI 门禁。

### 1.3 非目标

- 不做具体渲染算法（PBR/阴影/GI/后处理归 `prism_render_*`）。
- 不做 GPU 命令原语与屏障的后端实现（归 `prism_render_driver`，图只调 RHI 门面）。
- 不做着色器编译与反射（归 `prism_shader`，图只吃编译好的 PSO + 反射出的绑定布局）。
- 不做场景裁剪与可见性（归 `prism_render_scene`/`prism_render_visibility`，图只把其产出的 draw list / indirect buffer 当外部输入 `read`）。
- 不保证各后端像素完全一致；保证**语义一致**、优化在不可证等价处安全退化。
- 不含任何学习型调度；一切排序/别名/合并均为确定性经典算法。

---

## 2. 参考产品取舍（逐项落到机制）

| 来源 | 借鉴什么（具体机制） | Prism 取舍与落点 |
|---|---|---|
| **Frostbite FrameGraph**（GDC 2017） | 瞬态资源 + 生命周期自动别名 + setup/execute 两阶段分离的开创范式；virtual resource handle | 作为核心范式基线（§4/§5/§7）：虚拟资源 + 区间别名 + 两阶段闭包 |
| **Unreal RDG** | `ERDGPassFlags`（Raster/Compute/AsyncCompute/Copy/NeverCull）、RDG buffer/texture 句柄、异步计算 fence、pass 剔除、`FRDGBuilder` setup/execute lambda、parallel translate（并行把 pass 翻译成 RHI 命令）、transient resource allocator | 借 pass flag + 句柄 + 异步 fence + parallel translate 形态（§4/§9/§10）；transient allocator 思路进 §7 |
| **Unity SRP RenderGraph** | pass/资源虚拟化、HDRP 资源池与 import、**Native RenderPass（NRP）自动合并**、RenderGraph compiler 从 v1（贪心）到 v2（更强合并/别名）的演进 | 借资源池 + import 接口（§5）、NRP 合并进 §11、两代编译器演进思路进 §17 |
| **Granite**（Themaister） | Vulkan 下屏障/别名/信号量全自动、render pass + subpass 合并、per-resource 状态追踪、博客公开的完整实现细节 | 作为屏障推导（§8）与 subpass 合并（§11）的主要实现参考（公开形态，不抄码） |
| **bevy `render_graph`** | 节点/边/slot、子图（sub-graph）组织、view node 复用 | 借子图与节点复用理念（§4 子图），**不绑其命令式 slot 运行时**（Prism 走声明式自动连边） |
| **The-Forge / Diligent** | 帧级编排、resource barrier 自动/手动双模、多后端一致语义 | 借「可自动可手动屏障」理念（§8 可覆盖） |
| **O3DE/Atom Frame Graph、Wicked Engine rendergraph** | 工业级/开源级的 pass 组织、资源 attachment 声明、异步 compute 实践 | 作为 API 工效（§4/§20）与异步调度（§9）的旁证参考 |
| **id Tech 7（Doom Eternal）** | GPU 驱动 + 严格 barrier 预算 + 低 draw 调用 + 高 async compute 占用 | 作为 §9 异步重叠率、§17 屏障预算基准的目标画像 |
| **Decima（Horizon）/ Call of Duty / Destiny** | frame graph + async compute 的工业落地、跨队列同步实践 | 作为 §9 列表调度策略与跨队列安全阀的实践参照 |
| **AMD RMV / FidelityFX** | Radeon Memory Visualizer 的显存别名时间轴可视化、资源驻留观测 | 作为 §14 别名可视化与 §7 别名正确性验证的工具形态 |
| **Vulkan render pass / dynamic rendering + `local_read`** | subpass + input attachment；dynamic rendering 的 `VK_KHR_dynamic_rendering_local_read` 做 tile 内读 | §11 TBDR 合并的首要后端映射目标（新旧两条路径都支持） |
| **`VK_KHR_synchronization2` / D3D12 enhanced barriers** | 更细的 stage/access mask、split barrier、layout 与 access 解耦 | §8 屏障推导的能力上界与建模基准 |
| **Metal tile shader / imageblock / memoryless** | tile 内存显式编程、`memoryless` 瞬态附件（不回显存）、programmable blending | §11 Apple 平台 tile 优化的后端映射目标 |
| **timeline semaphore（Vulkan/D3D12 fence）** | 单调递增时间线、跨队列/跨帧细粒度同步 | §9 跨队列与 §16 跨帧同步的统一原语 |

**不做**：不做具体渲染算法（PBR/阴影/后处理归 `prism_render_*`，图只提供编排）；不做 GPU 命令原语（归 `prism_render_driver`，图只调 RHI）；不做着色器编译（归 `prism_shader`）；不做场景裁剪与可见性（归 `prism_render_scene`/`prism_render_visibility`，图只吃其产出的 draw list）。

> 所有 Prism crate 不含任何 Unreal Engine / Unity / Frostbite 源码或衍生代码；仅借鉴公开架构形态、公开演讲/博客中的经典算法与数值。

---

## 3. 核心概念：Pass / Resource / Handle / State / Graph

- **Graph（图）**：一帧的全部编排，一次性构建、编译、执行、丢弃（或**结构池化复用**，见 §17）。
- **Pass（通道）**：图的节点。按能力分四类——`Raster`（有渲染附件的光栅通道）、`Compute`（计算）、`Transfer`（拷贝/清除/mip 生成/blit）、`AccelBuild`（光追加速结构 BLAS/TLAS 构建，§18）。另有 `Present`（呈现，恒有副作用不剔除）。每个 pass 带一个 setup（声明依赖）与一个 execute（录命令），以及一组 **PassFlags**（`AsyncCompute` / `NeverCull` / `Transfer` / `MergeCandidate` / `SideEffect`，借 UE `ERDGPassFlags`）。
- **Resource（资源）**：图的「虚拟」资源，分 `Texture` / `Buffer`（后续可加 `AccelStruct`）。编译前只是「描述 + 句柄」，不占显存；编译后被分配/别名到真实 RHI 资源。
- **Handle（句柄）**：`TextureHandle` / `BufferHandle`，强类型、带**写版本**（见 §6）。setup 返回句柄，execute 用句柄向图索取真实 RHI 资源。句柄是 `prism_utils` 的 generational handle（index + 版本），越界/过期访问编译期或 debug 期可查。
- **ResourceDesc**：纹理（尺寸/格式/mip/采样数/数组层/用途位）或缓冲（大小/用途位）的描述；尺寸可相对交换链（§5）。
- **Access（访问）**：pass 对资源的意图——读 / 写 / 读写 / 附件（color/depth/resolve/input）；访问类型 + 着色阶段决定屏障的 src/dst **状态与访问域**（§8）。
- **ResourceState（资源状态）**：资源在时间轴上的 `(layout, access, stage)` 三元组建模（对齐 Vulkan sync2 / D3D12 enhanced barriers 的解耦模型，§8）。

**资源三态（生命周期类别）**：
- **Transient（瞬态）**——仅本帧存在，由图分配与别名（§7 别名的唯一对象）。
- **Imported（导入）**——外部（交换链 backbuffer、持久 history、常驻资产纹理、streaming 页）注入图，图只管状态转换不管分配。
- **Persistent（持久）**——由图分配但跨帧保留（§16），参与状态转换但**不参与瞬态别名**。

**三类 pass × 三态资源** 的笛卡尔积构成图的完整语义空间；编译器的每个阶段（§6–§11）都在这个空间上做确定性变换。

---
## 4. 构图 API：setup/execute 分离与 builder（深化）

构图的核心是「两阶段闭包」：setup 只声明意图（可并行、可缓存、不碰 GPU），execute 延迟到编译后真正录命令。形态（PLANNED，示意）：

```rust
// 伪代码：形态示意，非最终 API，均为 PLANNED
struct GBufferData { albedo: TextureHandle, normal: TextureHandle, depth: TextureHandle }

let gb = graph.add_raster_pass("gbuffer", PassFlags::empty(), |b: &mut PassBuilder| {
    let albedo = b.create_color_attachment("albedo", TextureDesc::rgba8_screen());
    let normal = b.create_color_attachment("normal", TextureDesc::rgb10a2_screen());
    let depth  = b.create_depth_attachment("depth",  TextureDesc::depth32_screen());
    b.read_external(scene.visible_meshes);     // 外部 draw list（CPU 侧，仅建依赖序）
    b.read(scene.gpu_draw_args, Stage::DRAW_INDIRECT); // GPU indirect buffer，建屏障依赖
    GBufferData { albedo, normal, depth }      // 把句柄传给 execute
}, |data: &GBufferData, ctx: &mut ExecuteContext| {
    let mut rp = ctx.begin_raster(&[data.albedo, data.normal], Some(data.depth));
    rp.set_pipeline(ctx.pso(PSO_GBUFFER));
    rp.draw_indirect(ctx.buffer(scene.gpu_draw_args));
});
```

### 4.1 要点与契约

- **setup 返回的 `PassData`** 携带句柄，被框架存起来在 execute 时按不可变引用回灌，避免闭包捕获可变借用、也避免 execute 捕获 setup 环境造成生命周期纠缠。
- **create/read/write/import** 全在 builder 上登记，框架据此建依赖边与资源生命周期；一次登记产生「一条访问记录」= `(resource, access, stage, 可选附件槽位)`。
- **声明即契约**：execute 内只能访问 setup 声明过的资源；越权访问 debug 下 panic 并打印 `pass 名 + 资源名 + 访问类型`（§20.2）。
- **setup 必须无副作用且幂等**：这是「可并行跑 setup」「可缓存复用」的前提（§17）。setup 内禁止碰 GPU、禁止读全局可变状态。

### 4.2 访问推导（减少手写 read/write）

易用性的关键优化：**从 PSO 的绑定反射自动并集出部分 access**。`prism_shader` 产出的 PSO 带绑定布局（哪个 set/binding 是 sampled / storage / UAV）。setup 时 `b.bind_pipeline(pso)` 可让框架据反射自动把「这个 pass 会 sampled-read 绑定点 X 的纹理」登记为依赖，用户只需补充「哪个图句柄绑到哪个点」。附件（color/depth）仍显式声明（它决定 render pass 结构，不能靠反射）。此特性可关（`access-inference` feature），关掉则纯手写，语义等价。

### 4.3 类型化与人机工程

- **强类型 `PassData`**：每个 pass 自定义结构体承载句柄，execute 拿到的是具名字段而非裸索引，重构友好、IDE 可补全。
- **`TextureDesc` 构造器族**：`rgba8_screen()` / `depth32_screen()` / `half_res(fmt)` / `fixed(w,h,fmt)`，常见组合一行搞定。
- **附件声明糖**：`create_color_attachment` 隐含 `COLOR_ATTACHMENT` 用途位 + 写访问 + 附件槽位，省三处手写。
- **子图（sub-graph）**：把一组 pass 封装成可复用、可参数化单元（如「阴影级联子图」「cubemap 6 面子图」），`graph.instantiate(sub_graph, params)` 插入主图（借 bevy sub-graph 形态）。子图内部句柄不泄漏到外部，只通过显式 in/out 端口连接（§18 可重入子图）。

---

## 5. 资源虚拟化：瞬态 / 导入 / 持久（深化）

瞬态资源是帧图的灵魂：pass 声明「我要一张半分辨率 RG16F」，图在编译期根据**所有瞬态资源的生命周期**决定它们真实占用哪块显存、能否与别的瞬态资源**共用同一块**（生命周期不重叠即可别名）。

### 5.1 三种引入方式

- **创建（瞬态）**：`b.create_texture(desc)` / `b.create_buffer(desc)` 返回瞬态句柄；不立即分配，参与 §7 别名。
- **导入**：`graph.import_texture(rhi_tex, current_state)` 把交换链 backbuffer、上一帧 history、常驻资产、streaming 页注入；图负责从 `current_state` 转到首个使用所需状态，并在帧末转回（或交给呈现 / 交回导入者）。导入资源**不参与别名**（图不拥有其显存）。
- **持久创建**：`graph.create_persistent(desc)` 由图分配但跨帧保留（§16），参与状态转换不参与别名。

### 5.2 尺寸推导

`TextureDesc` 支持相对尺寸：`Screen`、`Screen/2`、`Screen*scale`、`Fixed(w,h)`、`RelativeTo(other_handle)`。编译期按当前交换链解析为绝对像素。相对尺寸让「半分辨率 SSR」在交换链 resize 时自动跟随，无需手动重算。解析在编译最前置步骤完成（供后续别名用绝对字节数算区间）。

### 5.3 用途位与格式推导

- **用途位（TextureUsage/BufferUsage）**：图从「谁读谁写、作为什么附件、哪个阶段用」自动并集出 RHI 用途位（sampled / storage / color-attachment / depth-attachment / copy-src/dst / indirect / vertex / index / uniform）。用户无需手填，杜绝「忘了加 STORAGE 位导致后端创建失败」这类错配。
- **格式校验**：附件格式与 PSO 的 render target 格式在编译期比对，不匹配即报错（而非运行期后端崩）。
- **用途位过宽的代价**：某些后端用途位越多，资源创建的内部开销越大、可用压缩越少。图默认按「实际用到的最小并集」，不无脑全开。

### 5.4 外部同步契约

导入资源的 `current_state` 由导入者**如实声明**；图据此插首个屏障。若导入者谎报状态（例如实际还在被上一次提交写），属契约违反——图在 validation feature 下用一次 debug fence + 状态影子表尽力检测，但根本保证靠调用方（§19）。

---

## 6. 依赖推导与无用通道剔除（深化）

### 6.1 SSA 写版本法

资源的读写关系唯一决定执行序。写版本法（SSA 风格）——每次写一个资源，其句柄「写版本 +1」，产生一个新的「资源版本节点」；读某版本即建一条从「产生该版本的 pass」到「本 pass」的依赖边。

- **版本即不可变快照**：`TextureHandle` 携带 `(resource_id, version)`。读 `v0`、写出 `v1` 是两个不同的句柄值；下游读 `v1` 自然连到写它的 pass，读 `v0` 连到更早的 pass。这天然消解「读旧值还是新值」的歧义。
- **三类 hazard 全覆盖**：
  - **RAW（Read-after-Write）**：读某版本 → 依赖产生它的写 pass。
  - **WAR（Write-after-Read）**：新写要等旧读完成 → 图在「写 vN+1」与「所有读 vN 的 pass」间建反依赖边（否则新写会覆盖还没读完的内容）。
  - **WAW（Write-after-Write）**：连续写同一资源 → 版本链天然串行（除非写的是不相交子区域，见 6.3）。
- **无隐式全局顺序**：两个互不依赖的 pass 被判定为可并行/可乱序 —— 这是 §9 异步计算与 §10 并行录制的信息来源。拓扑排序在满足所有依赖边的前提下，可选多种合法线性序，调度器据关键路径择优（§9）。

### 6.2 剔除算法（反向引用计数）

从「最终被消费的资源」（交换链 backbuffer、被导出的 history、回读到 CPU 的 buffer）反向追踪：

1. 每个资源版本 ref count = 「读它的 pass 数」；每个 pass 的「有效输出数」= 「其写出的、ref>0 的资源版本数」。
2. 工作队列初始化为「有效输出数 == 0 且无副作用」的 pass（死 pass 候选）。
3. 弹出一个死 pass → 剔除它 → 把它读的每个资源版本 ref count 减一 → 若某上游资源版本 ref 归零且其生产 pass 的有效输出也归零且无副作用，则该上游 pass 入队。连锁传播直到队列空。
4. **`SideEffect` / `NeverCull` / `Present` pass 永不剔除**（写 backbuffer、CPU 回读、external signal、显式标注的调试 pass）。

效果：调试期临时加的 pass、被开关关掉的特性分支、「算了但没人用」的中间结果，无需手动摘除——没人消费就自动不执行、不分配显存。剔除结果可 golden 断言（§19）。

### 6.3 细粒度依赖（降低假串行）

默认按「整资源」建依赖。两处可细化以提并行度（可选、保守）：

- **子区域写**：对 buffer 的不相交区间 `[a,b)` 与 `[c,d)` 的写，若区间不交则不建 WAW 边（需 pass 显式声明写区间；不声明则保守按整资源串行）。
- **mip/层级粒度**：对纹理不同 mip / array layer 的读写可分别追踪状态（§8 的 subresource 状态表），避免「写 mip0 挡住读 mip3」。

细粒度是把双刃剑（追踪成本 + 复杂度），默认关，按资源 opt-in。

---
## 7. 资源生命周期分析与内存别名（深化）

别名是帧图峰值显存降 30–60% 的核心来源，也是最危险的优化（算错 → 两个活资源踩同块显存 → 难查画面错误）。本章把它拆成四步，每步可证、可视、可关。

### 7.1 生命周期区间计算

编译期在**最终执行序**上为每个瞬态资源算出区间 `[first_use, last_use]`（以 pass 的拓扑序号为坐标）。

- `first_use` = 第一个创建/写它的 pass 序号；`last_use` = 最后一个读/写它的 pass 序号。
- **异步队列的坐标统一**：多队列下，`last_use` 必须取「所有队列上最后一次使用」的全局同步点，而非单队列序号——否则异步队列还在用的资源会被图形队列的新资源别名覆盖。图把跨队列 use 投影到统一时间线（signal/wait 点）再算区间（与 §9 耦合）。
- **持久/导入资源不算区间**（不参与别名）。

### 7.2 堆/类型兼容性分类

别名只在**兼容类别**内进行（接 `prism_render_driver` §14 的 placed 子分配器与内存类型）：

- 颜色附件、深度/模板附件、存储纹理、普通缓冲可能属不同内存类型 / 不同对齐族 / 不同 heap tier（部分平台 RT/DS 要专用 heap）。
- 兼容键 = `(memory_type, alignment_class, 是否 RT/DS 专用, 是否 memoryless 候选)`。只有兼容键相同的资源进同一个别名池。
- Metal/UMA 平台兼容面更宽；分离显存桌面平台更窄。兼容性由 RHI 的 `GpuCaps` + 内存类型查询回答，图不硬编码。

### 7.3 别名分配算法（两档）

把「给每个资源在物理堆上定 `(offset, size)`，使区间不重叠的资源可共享字节」建模为经典问题。提供两档：

- **档一·线性扫描（默认，快）**：按 `first_use` 升序遍历资源，维护「空闲物理块」池（offset+size 的空洞表）。资源到来时取一个 ≥ 其 size 的空闲块（best-fit 减碎片）；资源在 `last_use` 后把块归还池。O(n log n)，适合每帧跑。借 Frostbite / UE transient allocator 形态。
- **档二·区间图着色（更省，慢）**：把「区间重叠」建成冲突图，近似图着色求最小总显存；适合「结构稳定、只在首次或 resize 时算一次」的离线/缓存场景（§17 编译缓存命中后不重算）。借 Unity RenderGraph v2 更强合并的思路。

**偏置策略**：
- **贪心按 size 降序放置大资源**（大块先定位，减少无法填入的碎片）。
- **对齐**：每个 `(offset)` 向资源对齐要求取整；跨资源保守取堆最大对齐，避免 overlap 判定踩对齐空洞。

### 7.4 别名屏障（aliasing barrier）

复用同一块物理显存的两个资源切换时，必须插 **aliasing barrier**（内容失效语义：旧资源内容在新资源看来是 undefined）：

- 框架自动在 `last_use(A)` → `first_use(B)` 的边界、且 A 与 B 落在同一物理块时插入。
- aliasing barrier 要正确表达「A 的最后写必须对 B 的首次使用可见地完成」+「B 的 layout 从 undefined 初始化」，映射到 Vulkan 的 `VK_IMAGE_LAYOUT_UNDEFINED` 过渡 / D3D12 aliasing barrier / Metal heap aliasing。
- **别名引入的假依赖**：A、B 本可并行（无数据依赖），但共享物理块后 B 必须等 A 用完 → 被串行化。这是别名的隐藏代价（见 7.5）。

### 7.5 别名 vs 并行度权衡（可调旋钮）

- **过度别名制造假依赖**：显存省了，但 GPU 并行度掉了（尤其异步队列）。
- **别名预算旋钮**：`AliasPolicy::{ MemoryFirst, ParallelFirst, Auto }`。
  - `MemoryFirst`：显存紧张时激进别名（移动端 / 低显存桌面默认）。
  - `ParallelFirst`：显存充裕时放宽别名以保并行度（高端桌面默认）。
  - `Auto`：按当前显存占用自适应——占用越高越激进（接 RHI 显存预算查询 + diagnostic 内存图谱）。
- **跨队列别名默认禁止**：异步队列资源与图形队列资源别名同块需额外同步、极易引入竞态，默认关，显式 opt-in（§9 安全阀）。

### 7.6 碎片与去碎片

- 结构稳定帧的别名布局缓存复用（§17），不每帧重算，天然无碎片抖动。
- 结构变化导致空洞累积时，重编译触发一次 best-fit 重排（compaction），把布局重算成紧凑形态。
- 去碎片只动「图拥有的瞬态堆」，不动导入/持久资源。

### 7.7 可观测与可关

- **别名显存时间轴**：导出「哪块物理块在哪些 pass 区间被哪个资源占用」（§14，形态借 AMD RMV）。
- **一键关闭**：`AliasPolicy::Off` 让每个瞬态资源独占显存——用于定位「疑似别名导致的画面错误」，排除法秒级验证。

---

## 8. 屏障与状态转换自动插入（深化）

每个资源（及其 subresource：mip/层/面）在时间轴上经历一串状态。框架沿执行序为每条「状态变化」生成 RHI 屏障。本章对齐 `VK_KHR_synchronization2` / D3D12 enhanced barriers 的**解耦模型**。

### 8.1 状态模型（三元组）

资源状态 = `(layout, access_mask, stage_mask)`：
- **layout**：图像布局（ColorAttachment / DepthAttachment / ShaderReadOnly / General(storage) / TransferSrc/Dst / Present）。缓冲无 layout，只有 access/stage。
- **access_mask**：读/写访问类型（ShaderRead / ShaderWrite / ColorWrite / DepthWrite / TransferRead/Write / IndirectRead …）。
- **stage_mask**：产生/消费该访问的管线阶段（VS/FS/CS/Transfer/DrawIndirect/EarlyZ/LateZ …）。

把 layout 与 access/stage 解耦，正是 sync2 / enhanced barriers 的进步点——能表达「layout 不变但要同步 access」或「只 execution barrier 不 memory barrier」，屏障更精确、气泡更小。

### 8.2 状态推导

由 pass 的 access 类型 + 绑定阶段决定目标状态：写附件→`ColorAttachment`/`DepthAttachment` + 对应 stage；sampled 读→`ShaderReadOnly` + 读它的着色阶段；storage→`General` + ShaderWrite/Read；indirect 读→access=IndirectRead + stage=DrawIndirect；拷贝→`TransferSrc/Dst`。访问阶段尽量取**精确阶段**（只在 FS 读就不要上 `ALL_GRAPHICS`），减少过度同步。

### 8.3 屏障批合并（batching）

同一提交点上多资源的状态变化合并为**一次** `pipeline barrier` / `barrier group` 调用（减少 API 开销与驱动端重复 flush）。合并规则：同一「同步点」（两个 pass 之间的边界）上收集所有待转换资源，按 src/dst stage 并集成尽量少的 barrier group。

### 8.4 Split barrier（分离屏障）

在「资源状态不再被需要」处发出 begin，在「下次使用」前发出 end，让 GPU 在两点之间隐藏转换/刷新延迟：

- 适用场景：A 写完某纹理后，隔了好几个 pass 下游才读它 → begin 紧跟 A，end 紧贴下游读，中间 GPU 可继续干别的。
- 接 RHI 的 split barrier 能力（Vulkan event / D3D12 split barrier）；后端无此能力则安全退化为单屏障（语义等价，仅不省气泡）。

### 8.5 冗余屏障消除与读合并

- **冗余消除**：若资源已处于目标状态（上一个屏障已转到位、中间无改变状态的访问），跳过本次屏障。
- **读合并（read-combining）**：连续多个「只读同一状态」的 pass 之间不需要屏障；图把一段「同状态只读区间」合并，仅在进入/离开该状态时各一个屏障。
- **初始/终末状态**：瞬态资源首次使用从 `Undefined`（可丢弃旧内容，省一次 flush）；导入资源从声明的 `current_state`；帧末把导入/持久资源转回约定状态或 `Present`。

### 8.6 队列族所有权转移

跨队列（图形↔计算↔传输）使用同一资源时，Vulkan 需 queue family ownership transfer（release + acquire 一对屏障），D3D12/Metal 走对应状态/同步。框架在 §9 的 signal/wait 点配套插入这对屏障，用户无感。

### 8.7 可覆盖（显式屏障逃生门）

极端优化场景允许 pass 显式声明屏障，框架让路并仅**校验一致性**：显式屏障必须覆盖框架推导的最小集（覆盖所有真实状态变化），否则 debug panic 并打印缺的那条。既保留手动榨性能的能力，又用契约防「手动漏屏障」这个最经典的 bug。

---

## 9. 多队列：异步计算与传输调度（深化）

依赖图的「可并行」信息（§6）天然支持把无依赖的 compute/transfer pass 丢到异步队列，与图形队列重叠。

### 9.1 队列分配

- pass 可标 `queue_hint = Graphics / AsyncCompute / Transfer`；框架在不违反依赖前提下尽量满足。
- 资源上传/下载/mip 生成类 pass 优先走传输队列；重 compute（SSAO、光照剔除、GI 探针更新、粒子模拟）优先投异步计算队列。
- 队列数与能力由 RHI `GpuCaps` 回答（有几条 async compute / transfer 队列）；无独立队列的平台安全退化为单队列（语义等价）。

### 9.2 跨队列同步（timeline semaphore）

当队列 A 的 pass 产出被队列 B 的 pass 消费：框架在 A 末插 signal(value=t)、B 前插 wait(value=t)，统一用 **timeline semaphore**（单调递增值，一个信号量表达多点同步，省去管理一堆 binary semaphore）。接 RHI §10 / §13。

### 9.3 关键路径列表调度

经典 **list scheduling（表调度）**：
- 为每个 pass 算「到终点的最长路径」（关键路径长度）作为优先级。
- 把长 compute（SSAO、光照剔除）**尽早**投到异步队列，与阴影/几何的图形队列并行；避免「所有东西挤在图形队列尾部」造成的尾部串行。
- 目标画像：对齐 id Tech 7 / Decima 的高 async compute 占用率。
- 调度是确定性的（相同图 → 相同分配），保证可复现。

### 9.4 重叠度量

- 每个异步 pass 记录「实际与图形队列重叠了多少 µs」（接 §14 GPU 计时 timestamp）。
- 「异步收益」统计进性能面板与 CI 基准：如果某 pass 投了异步却几乎没重叠（被同步点卡住），面板会标红，提示调度或依赖有问题。

### 9.5 安全阀

- **跨队列别名默认禁止**（见 7.5）：异步队列资源与图形队列资源别名同块需额外同步，极易竞态，默认关、显式 opt-in、且 opt-in 后强制插保守同步。
- **强制单队列安全档**：`QueuePolicy::ForceSingle` 一键把所有 pass 压回图形队列——用于排查「疑似跨队列同步」导致的挂起/竞态，排除法定位。
- **死锁预防**：跨队列 wait 图在编译期做环检测（接 `prism_tasks` §24.8 的 WaitGraph 思路），有环即报错而非运行期挂 GPU。

---
## 10. 并行命令录制（深化，接 tasks）

执行阶段，互不依赖的 pass 的 execute 闭包可在 `prism_tasks` 的 worker 上并行录制到各自的命令缓冲，最后按执行序提交。借 UE 的 **parallel translate**（并行把 pass 翻译成 RHI 命令）形态。

### 10.1 录制粒度

- 每个 pass（或一组被合并进同一 render pass 的相邻 raster pass，见 §11）对应一个**二级命令缓冲 / 并行 render pass 编码器**（Vulkan secondary command buffer、D3D12 bundle/多 command list、Metal parallel render command encoder）。
- 一组 raster subpass 合并链共享一个 render pass 实例，内部各 subpass 可并行录到 secondary buffer。

### 10.2 确定性归并（关键）

并行录制的命令缓冲，其**提交顺序**严格按编译期确定的执行序归并（接 `prism_tasks` §24.7 确定性归并保证提交序逐帧一致），而非「谁先录完谁先提交」。这对回放、录像、确定性联机、golden 调试至关重要——录制可乱序并行，提交必须确定有序。

### 10.3 负载均衡

- 按 pass 的历史 GPU/CPU 录制耗时（接 §14 diagnostic 的 per-pass 计时环形历史）估算录制成本，用 `prism_tasks` 的标定 grain（§24.4）把 pass 均摊到 worker，避免「一个超重 pass（如几万 draw 的 GBuffer）拖尾、其他 worker 空转」。
- 超重 pass 可进一步按 draw 范围切成多段并行录到同一 render pass 的多个 secondary buffer（pass 内并行）。

### 10.3 线程安全

- RHI 命令录制本身线程安全（`prism_render_driver` §9 保证：每个 encoder 独立、不共享可变状态）。
- 图只需保证各 pass 的 execute 不共享可变录制状态、不跨 pass 捕获可变借用——这由 §4 的 `PassData` 回灌 + setup 无副作用契约保证。

### 10.4 可关

`parallel-record` feature 可关，退化为单线程按执行序串行录制（语义等价，仅慢）——用于排查「疑似并行录制」导致的问题，或在单核/wasm 环境。

---

## 11. 原生 RenderPass / subpass 合并与 TBDR tile 内存优化（深化）

移动与 Apple GPU 是 TBDR（Tile-Based Deferred Rendering），把渲染目标切 tile 常驻 on-chip。若连续 pass 对同一组附件做「写→读同像素」，可合并进一个 render pass 的多个 subpass，中间结果留在 tile 内存，省掉往返显存的带宽（延迟渲染 GBuffer→光照是经典场景，移动端收益巨大）。桌面侧则对应 Unity 的 **Native RenderPass（NRP）** 合并以减少 render pass 切换开销。

### 11.1 合并判据

相邻 raster pass 可合并进一个 render pass 当且仅当：
- 附件集合兼容（分辨率、采样数、附件格式一致）；
- 后一 pass 对前一输出的读是**「同像素」input attachment**（`texelFetch` 当前片元位置），而非任意坐标采样（任意采样要求结果已落显存，不能留 tile）；
- 中间没有改变附件内容的异步/跨队列访问；
- 不跨越会打断 tile 的操作（如 resolve 到外部后又读）。

### 11.2 三后端映射

- **Vulkan（旧）**：subpass + input attachment + `by_region` 依赖。
- **Vulkan（新）**：dynamic rendering + `VK_KHR_dynamic_rendering_local_read`（在 dynamic rendering 下做 tile 内读，摆脱老 render pass 对象的样板）。图按 `GpuCaps` 选新旧路径。
- **Metal**：`tile shader` / imageblock / programmable blending；瞬态且无人读的附件标 `memoryless`（根本不分配显存，纯 tile）。
- **D3D12 / 桌面无 tile 概念**：退化为普通 pass + 屏障（语义等价，仅不省带宽）；仍可受益于 NRP 式的 render pass 合并减少状态切换。

### 11.3 load/store/resolve 自动推导

图据资源生命周期自动决定附件的：
- **load**：`clear`（首次写且需清）/ `load`（要保留旧内容）/ `dontcare`（瞬态首用，省读带宽）。
- **store**：`store`（后续有人读）/ `dontcare`（瞬态且后续无人读，移动端省写带宽巨大）/ `resolve`（MSAA 解析到单采样目标）。
- Metal 的 `memoryless` 附件由「瞬态 + 全程只在合并链内 tile 读写 + 从不落显存」自动识别。

### 11.4 自动 + 可提示 + 可关

- 框架自动识别可合并链；也接受 pass 显式 `merge_hint` / `merge_barrier` 处理识别不到或要强制的情形。
- `subpass-merge` feature 可关，全部退化为独立 render pass + 屏障（语义等价）——合并是纯带宽/切换优化，关掉只影响性能不影响正确性，便于隔离后端差异问题。

---

## 12. 执行模型：编译 → 分配 → 执行 → 回收

一帧流水线（PLANNED）：

1. **构图（record）**：调用各 pass 的 setup，收集 pass/资源/依赖/附件声明。setup 无副作用，可并行跑（接 tasks）。
2. **编译（compile）**：剔除无用 pass（§6）→ 拓扑排序 + 队列分配（§6/§9）→ 生命周期分析 + 别名（§7）→ 屏障计划（§8）→ subpass/NRP 合并（§11）→ 产出一份**不可变** `ExecutionPlan`。编译器本身是一串可插拔的 compiler pass（§20.4）。
3. **分配（realize）**：按别名结果从 RHI 的 placed 子分配器申请物理块，绑定给瞬态资源；导入资源登记当前状态；持久资源沿用上帧物理块。
4. **执行（execute）**：按计划（可并行录制 §10）调各 pass 的 execute 录命令，插屏障/信号量/aliasing barrier，分队列提交（确定性归并顺序）。
5. **呈现 + 回收（present/recycle）**：呈现 backbuffer；归还瞬态物理块到池；持久/history 资源留存并 ping-pong 轮换（§16）；`ExecutionPlan` 结构可池化复用（§17 增量重编译）。

`ExecutionPlan` 一旦编出即**不可变**：它是「这一帧确切怎么跑」的唯一真相，execute 阶段只读不改。可变性全集中在 record+compile，执行期无决策——这是确定性与可并行的基础。

---

## 13. Blackboard 与跨 pass 数据流

Blackboard（黑板）是图级的类型化键值表，解决「上游 pass 产出的句柄要给下游 pass 用」的传参问题，避免手工层层透传：

- `blackboard.set::<GBuffer>(GBuffer { albedo, normal, depth })` 由 gbuffer pass 写入；
- `let gb = blackboard.get::<GBuffer>()` 由光照 pass 读出，拿到句柄建依赖；
- **类型即键**（`TypeId`），编译期类型安全；缺键在 debug 下 panic（契约：读黑板前该键必须被某上游 pass 写过），并打印缺的类型名。

细节：
- 黑板只存**图句柄与小型元数据**，不存 GPU 资源本体；跨帧数据走 §16 持久资源 + history，不走黑板（黑板每帧清空）。
- **作用域黑板**：子图可有私有黑板层（子图内键不污染主图命名空间），通过显式 in/out 端口与外层交换（§4.3 / §18 子图）。

---

## 14. 图可视化与调试（接 diagnostic）

渲染图是「声明式」的，调试器必须把隐式的编译决策显式呈现出来，否则声明式反而比命令式更难排错。本章全部接 `prism_diagnostic_design_zh.md` 的 GPU 计时 / 统一时间线 / 显存图谱，不自建一套。

### 14.1 结构导出（DOT / JSON）

- **DOT**（`dot-export` feature）：导出剔除前后两张图——节点为 pass（按队列着色：graphics/compute/transfer），边为资源依赖（标 RAW/WAR/WAW 与版本号），虚线为别名共享同块显存的资源对。一眼看出假串行、未剔除的死 pass、意外的跨队列边。
- **JSON**（`serde` feature）：导出完整 `ExecutionPlan`——pass 执行序、队列分配、每个瞬态资源的别名堆偏移/区间、每条屏障的 before/after 状态三元组、subpass 合并链。供外部工具链与 CI 快照比对（§19 差分测试直接 diff 这份 JSON）。
- 两种导出都带 **schema 版本号**，格式演进走弃用期（§22）。

### 14.2 GPU 计时与时间线

- 每个 pass 自动插 GPU timestamp query（接 diagnostic 的统一时间线），execute 后回读，产出 per-pass GPU 耗时；按队列分轨渲染成甘特图，异步计算与图形的**实际重叠区间**直接可见（印证 §9.4 的重叠度量不是纸面值）。
- Debug marker：每个 pass 自动 push/pop 调试组（`VK_EXT_debug_utils` / PIX / Metal 的 `pushDebugGroup`），名字即 pass 名——RenderDoc/PIX/Xcode 抓帧时层级与图结构一一对应，不用猜。
- CPU 侧 record/compile/realize 各阶段耗时打到 diagnostic 时间线，和 GPU 轨对齐，CPU-bound 还是 GPU-bound 一目了然。

### 14.3 别名时间轴（借 AMD RMV 形态）

借鉴 AMD Radeon Memory Visualizer 的形态：横轴为 pass 执行序，纵轴为别名堆的物理偏移，每个瞬态资源画成一个占用矩形（活跃区间 × 偏移范围）。重叠矩形=潜在踩踏（编译期本应证明不重叠，可视化是最后一道人眼校验），空白=碎片。配合 §7.7 的别名开关，一键关掉别名后此图退化为「无重叠满铺」，用于对拍定位别名 bug。

### 14.4 Null 后端 golden 断言

- `backend-null`（继承 Driver 的 Null 后端）不触碰真实 GPU，但完整跑 record→compile，产出确定性的 `ExecutionPlan`。CI 把执行序 / 剔除结果 / 别名布局 / 屏障计划序列化成 golden 快照，任何编译器改动若改变输出即红——**编译器行为被钉死为规格**（§19）。
- 断言覆盖：剔除正确性（死 pass 必被删、活 pass 必保留）、拓扑序合法性（所有依赖边被尊重）、别名不重叠（区间两两不交）、屏障覆盖（每条状态变化有对应屏障）。

### 14.5 实时内省 HUD

接 diagnostic 的 overlay：运行时叠加显示本帧 pass 数（剔除前/后）、峰值瞬态显存与别名节省比、屏障数、异步重叠率、record/compile 墙钟。性能回退时第一眼就知道是哪个维度崩了，不用离线抓帧。

---

## 15. 与 render_driver / shader / material / scene / streaming 集成

Render Graph 是编排层，自己不产生渲染内容，所有「画什么」都从上游来，「怎么落到 GPU」全交给下游 Driver。本章定清五个边界。

### 15.1 向下：render_driver（RHI）

- **唯一的 GPU 出口**：Graph 不直接碰任何后端 API，只调 Driver 的门面。`backend_bridge` 模块是适配层——把编译产物翻译成 Driver 调用：别名结果 → `placed` 子分配器申请（Driver §14）；屏障计划 → Driver 的 barrier/状态转换 API（Driver §10）；多队列提交 → Driver 的 Queue + timeline semaphore（Driver §13）；subpass 合并 → Driver 的 RenderPass/动态渲染（Driver §12）。
- **能力门控**：Graph 的高级编排（异步计算、subpass 合并、GPU-driven）用前查 Driver 的 `GpuCaps`，缺失即退化（单队列 / 独立 pass / CPU 提交），绝不假设存在（Driver §3 档位化）。
- **帧飞行**：Graph 每飞行帧一份瞬态资源池与命令缓冲，回收必在对应 fence 后（Driver §21 帧飞行不变量），Graph 不自管 fence，复用 Driver 的延迟回收。

### 15.2 向上：scene / material / shader

- **输入契约**：scene 产出 draw list（可见性剔除后的批次 + 实例数据），material 产出 PSO 句柄 + bindgroup，shader 产出字节码/反射。Graph 把这些当**不透明外部输入**接入某个 raster/compute pass 的 execute 闭包里，自己不解析其语义——Graph 只关心「这个 pass 读哪些资源、写哪些资源」，不关心 draw 了什么。
- **PSO 兼容性**由 Driver 创建期保证（Driver §21），Graph 只负责在正确的 render pass 上下文里 bind 与 draw，附件格式一致性由 Graph 的附件声明 + Driver 校验共同兜。
- **bindless 协同**：material 用 bindless 时，Graph 的屏障推导需知道「这个 pass 可能访问整张 bindless 堆」，走 §18 的 bindless 保守屏障策略。

### 15.3 横向：streaming（资源流送）

- streaming 决定某纹理/mesh 当前是否驻留显存。Graph 的 import 资源可能指向一个**部分驻留**的流送资源，Graph 需从 streaming 查询当前 mip 可用级别，execute 时按实际驻留级别采样（降级而非崩）。
- 显存预算压力下，streaming 可要求 Graph 让出瞬态显存（§18 显存预算降级），Graph 响应：缩小别名堆 / 降瞬态分辨率 / 砍可选 pass。二者通过 Driver 的 budget 监控（Driver §14）协调，不直接耦合。

### 15.4 边界原则

Graph 不拥有任何长生命周期 GPU 资源的「内容」——瞬态资源的物理块来自 Driver 子分配器且每帧归还；import/持久资源的内容由上游拥有，Graph 只持句柄与状态。这保证 Graph 可被完整 reset 而不泄漏，也让多个 Graph 实例（主视图 / 阴影 / 反射探针）能共享同一 Driver 而互不干扰（§18 子图实例化）。

---

## 16. 持久化资源与跨帧状态（history buffer）

TAA、SSR、运动模糊、自适应曝光、GI 时间累积等现代渲染技术都要「读上一帧的某个结果」，渲染图必须一等公民地支持跨帧资源，而不是让用户在图外手搓双缓冲。

### 16.1 history 环（ping-pong）

- `graph.create_history::<T>(desc, n)` 声明一个 n 槽（通常 2，GI 累积可能更多）的历史资源环。本帧写 slot `frame % n`，读 slot `(frame-1) % n`，框架自动轮换，用户只写「读历史/写当前」语义，不管哪个物理槽。
- history 资源是**持久资源**（§5 第三类）：物理块跨帧留存，不进瞬态别名池，不每帧归还。Graph 为其登记跨帧状态，首帧（无历史）走契约化的 fallback（清零 / 读当前 / 用户指定），避免采样未初始化显存。

### 16.2 分辨率/格式变更失效

- 窗口 resize、动态分辨率缩放、后端切换时，history 的尺寸/格式失效。Graph 检测到 desc 变更即**失效并重建**该 history 环，重置为首帧 fallback 状态——绝不拿旧分辨率的历史采样新分辨率（会花屏）。动态分辨率下 history 与当前帧分辨率可能不一致，提供「历史重投影」的尺寸元数据给上游（Graph 不做重投影，只给信息）。

### 16.3 跨帧同步

- 读上一帧 history 需确保上一帧对它的写已完成。Graph 用 Driver 的 timeline semaphore 跨帧等待（Driver §10/§13），而非粗暴全设备同步——只等那一个资源的写 fence，不阻塞无关队列。
- 异步时间线场景（§18）：某些跨帧 pass（如异步 GI 累积）可能跨多帧时间线运行，Graph 为其维护独立的跨帧依赖链，不塞进单帧编译。

### 16.4 跨帧状态即契约

history 的槽数、fallback 策略、失效条件都是**显式契约**，序列化进 `ExecutionPlan` 并可被 §14 可视化。跨帧 bug（读错槽 / 用脏历史 / 漏失效）是最难查的一类，必须编译期把这些决策钉死、运行期可观测。

---

## 17. 极致性能工程（编译期零开销 · 执行期零决策）

渲染图的 CPU 成本集中在「每帧重新编译」。一个复杂帧有数百 pass、上千资源，若每帧全量跑剔除+排序+生命周期+别名+屏障，CPU 可能先于 GPU 成为瓶颈。本章把编译成本压到「结构稳定帧趋近 0」。总纲：**可变性全锁在 record+compile，execute 期零决策；稳定结构免重编译；全程零堆分配。**

### 17.1 编译缓存 + 结构哈希

- 一帧的拓扑结构（pass 集合、依赖边、资源 desc）在相机移动、光源不变时逐帧**几乎不变**，变的只是 uniform 数据与 draw 内容。对图结构算一个**结构哈希**（pass 列表 + 边 + 资源 desc + 后端能力，不含逐帧数据），命中则直接复用上帧的 `ExecutionPlan`（执行序/别名布局/屏障计划全部复用），跳过整个 compile 阶段。
- 哈希用内容寻址（接 `prism_utils` 内容寻址缓存），稳定帧 compile 退化为一次哈希比对 + 缓存查找，耗时趋近 0。缓存按 LRU 保留最近 N 种结构（应对菜单/游戏/过场几套固定帧图切换）。

### 17.2 增量重编译

- 结构部分变化（加了一个可选 pass、某资源改了尺寸）时，不全量重编译：以依赖图为单位做**脏传播**——只有受影响的子图重跑对应编译器 pass，未变部分复用缓存结果。例如仅某 pass 的别名区间变化，只重跑 alias，不重跑 cull/schedule。
- 增量的正确性由 §19 验证层兜底：增量结果必须与全量重编译 bit-一致（CI 用 `backend-null` golden 对拍增量 vs 全量），否则增量逻辑有 bug。

### 17.3 零分配竞技场

- compile 的所有中间结构（生命周期区间表、别名候选、屏障候选、拓扑序缓冲）分配在**帧竞技场**（接 `prism_utils` 的 bump/arena allocator），帧末整体 reset，期间零 `malloc`/`free`。
- `ExecutionPlan` 本身也在可复用的 plan 池里（§17.5），不每帧新建。目标：稳定帧的 record+compile 全程零堆分配，GC 压力/分配抖动归零。

### 17.4 SoA pass 存储与缓存友好

- pass/resource 不用 `Vec<Box<dyn Pass>>` 这种指针追逐布局，而是 **SoA**（Structure of Arrays）：pass 的类型/队列/依赖/附件分列存储，编译器扫描时顺序访问，cache 命中率高。热路径（拓扑排序、生命周期扫描、屏障推导）全是对连续数组的线性扫描，无指针跳转。
- 句柄用 generational index（接 utils），不是裸指针，既安全又紧凑；资源表是密集数组 + free list，支持 O(1) 分配/回收。

### 17.5 并行 setup / compile 与 plan 池化

- **并行 setup**：各 pass 的 setup 无副作用（§4 契约），可在 `prism_tasks` 上并行跑，结果确定性归并（接 tasks §24.7），数百 pass 的声明收集不串行。
- **并行 compile**：编译器 pass 内部可并行的子任务（如多条独立依赖链的生命周期分析）切分到 tasks，关键路径调度（§9.3）本身也可并行化。
- **plan 池化**：`ExecutionPlan` 结构（含内部各数组容量）跨帧复用，命中缓存时直接拿旧 plan，未命中时在旧 plan 的已分配容量上原地重填，避免反复扩容。

### 17.6 屏障预算与批量化

- 屏障推导的产物按 §8.3 批合并后，还要控制「屏障本身的 CPU 记录成本」：屏障计划用紧凑的 SoA 数组表示，execute 期一次性 flush 一批，而非逐条调用。为屏障数设软预算，超预算触发 §14 诊断告警（通常意味着别名过激或 pass 粒度过细）。

### 17.7 bindless 减屏障

- 传统每资源每状态一条屏障，bindless 场景（material 用大描述符堆）可用「堆级保守屏障」替代海量细粒度屏障——整堆一次状态转换，换取屏障数量数量级下降（代价是保守同步，牺牲一点重叠）。此策略 feature 可选，默认在 bindless pass 上启用，非 bindless 路径仍走细粒度（§8）。

### 17.8 性能即 CI 门禁

稳定帧 compile 耗时、每帧堆分配字节数、屏障数、峰值瞬态显存、record/compile 并行加速比全部进 CI 基准（§23 基准即规格），回退即红。「零开销」不是口号而是被钉死的量化门禁。

---

## 18. 高级功能增补（AAA·深化）

本章是把渲染图从「能跑」推到「AAA 完全体」的能力集合。每一项都严格 `GpuCaps` 门控（经 Driver §3），缺失即降级，绝不无条件假设存在；每一项都是 feature 可关，关掉只影响性能/效果上限，不破坏正确性。

### 18.1 GPU-Driven 渲染完全体

- 渲染图原生编排 GPU-driven 管线：compute 剔除（视锥/遮挡/背面）→ 产出 indirect draw 参数与计数 → `multi_draw_indirect_count` 一次提交数万 draw。Graph 把「compute 写 indirect buffer」与「graphics 读 indirect buffer draw」之间的依赖/屏障自动插好（这是最易漏的 RAW）。
- **两阶段 HZB 遮挡剔除**：第一阶段用上帧深度剔大部分 → 画 → 构建本帧 HZB → 第二阶段补画被误剔的。Graph 把 HZB 构建（mip 链 compute）、两趟 draw、history 深度（§16）编排成一张正确的子图，依赖关系对用户透明。

### 18.2 Work Graph / GPU 工作图

- 支持 GPU work graph（D3D12 work graphs / 等价扩展）：GPU 自己派生工作、动态调度，CPU 只提交根节点。Graph 为这类 pass 提供「GPU 侧动态展开」的编排语义——CPU 侧依赖图在该节点变成不透明的 GPU 子图，Graph 负责其输入/输出资源的屏障与状态，内部调度交给 GPU。能力缺失时降级为 CPU 侧 indirect dispatch 链。

### 18.3 Mesh / Task Shader Pass

- 一等支持 mesh/task shader pass：无传统顶点装配，meshlet 数据作为资源输入，Graph 按 mesh pipeline 组织 draw。与 §18.1 的 GPU 剔除协同——task shader 做 meshlet 级剔除，Graph 编排 meshlet 可见性 buffer 的生产/消费依赖。缺 mesh-shader 档则降级为传统 GPU-driven indirect。

### 18.4 光追 BLAS / TLAS 构建 Pass

- 光追加速结构的构建/刷新是一类特殊 pass：BLAS（几何级，静态物体缓存、动态物体每帧 refit）、TLAS（实例级，每帧重建）。Graph 把 AS 构建编排为 compute-like pass，自动处理「AS 构建完成 → 光追 dispatch 读 AS」的依赖与屏障（AS 有专门的 build/trace 状态，接 Driver §24.1 光追原语）。AS 内存走持久资源 + 增量更新，不进瞬态别名。

### 18.5 VRS（可变着色率）

- 可变着色率作为 pass 级/区域级属性：Graph 允许某 raster pass 绑定一张 VRS image（由上游 compute 按内容/运动/中心凹产出），把 VRS image 的生产/消费依赖编排进图。缺 VRS 档则忽略（全速率着色，仅损性能不损正确）。

### 18.6 Multiview / VR

- 多视图（VR 双目、cubemap 一次画六面、级联阴影一次画多层）作为 pass 的 view-count 属性：一个 pass 声明 N 视图，Graph 为其附件按数组层组织、按视图广播 draw，一趟几何提交渲染多视图。缺 multiview 则退化为 N 趟独立 pass（语义等价，仅多提交开销）。VR 还需与跨帧重投影（§16）、低延迟呈现（Driver §11 帧节奏）协同。

### 18.7 条件执行

- pass 可声明「条件执行」：某 GPU 侧计数/预测为 0 时整个 pass 跳过（如可见实例为 0 的 draw、无粒子的模拟）。用 Driver 的 predication/conditional rendering 原语，避免 CPU 回读 stall。Graph 把条件 pass 的资源依赖按「可能执行」保守处理（屏障照插），但 GPU 侧真跳过时零开销。

### 18.8 异步时间线跨帧 Pass

- 某些工作不必锁定在单帧内：异步 GI 累积、probe 更新、BVH 增量重建可在独立 GPU 时间线上跨多帧低优先级推进，结果 ready 时再被主帧消费。Graph 为这类 pass 维护独立时间线与跨帧依赖链（接 §16.3），不塞进单帧关键路径——用低优先异步队列填 GPU 空闲，提升占用率。

### 18.9 资源驻留 / 流送编排 与 显存预算降级

- Graph 与 streaming（§15.3）协同编排资源驻留：某 pass 需要尚未驻留的资源时，Graph 可延迟该 pass 到资源 ready（可选 pass）或走降级路径（必需 pass 用低 mip）。
- **显存预算降级**：Driver 的 budget 监控（Driver §14）报告显存压力时，Graph 分级降级——先缩别名堆、再降瞬态分辨率、再砍可选 pass（后处理特效）、最后降主渲染分辨率。降级策略是显式配置的优先级列表，可观测（§14 HUD）。

### 18.10 多 GPU 分帧 / 分块

- 支持显式多 GPU（接 Driver §24.2）：AFR（交替帧）或 SFR（分块）。Graph 为多 GPU 编排跨设备资源拷贝与同步（哪些资源要跨卡传、何时同步），这是极少数场景的高级能力，默认关闭。

### 18.11 可重入子图实例化

- 渲染图支持**子图**（subgraph）作为可参数化、可复用的编排单元：阴影级联、反射探针、平面反射、画中画都是「同一套 pass 以不同视图/分辨率/目标跑多次」。
- 子图是**实例化**而非拷贝：定义一次（如「阴影子图」），用不同参数（视图矩阵、分辨率、目标 atlas 区域）实例化多份，每份有私有作用域黑板（§13），但共享结构哈希（§17.1，N 个阴影级联只算一次结构、复用别名布局）。
- 子图可嵌套，有显式 in/out 端口与外层交换资源句柄，内部命名空间隔离，不污染主图。这是把「几十个相似 pass」压成「一个子图 × N 实例」的关键，既减重复代码又减编译量。

---

## 19. 正确性、不变量与验证

渲染图把「屏障/别名/同步」这些最易出随机 bug 的东西自动化了——自动化的回报是「写对一次，处处正确」，代价是「编译器一旦有 bug，错误以难以复现的画面损坏形式散落全局」。所以验证不是附加项，而是本 crate 的生命线。

### 19.1 形式化不变量（编译期/运行期双重保证）

渲染图承诺并强制以下不变量（部分编译期证明，部分运行期 `validation` 档校验）：

1. **声明即全集**：pass 的 execute 只能访问其 setup 声明过的资源；越权访问 = 契约违反，`validation` 档运行期拦截并报「哪个 pass 访问了未声明的哪个资源」。
2. **依赖完备**：每条 read-after-write 必有一条依赖边，不存在「隐式顺序巧合正确」；编译期由 SSA 写版本法（§6.1）保证推全。
3. **别名不重叠**：被别名到同一物理块的任意两个资源，其生命周期区间两两不相交——**编译期证明**（区间相交检测），而非运行期撞见才知道。
4. **屏障覆盖**：每一次资源状态变化都有对应屏障覆盖；用户的显式屏障（§8.7）必须是推导最小集的超集，少覆盖即编译期报错。
5. **确定性**：相同图结构 + 相同输入 → 逐 bit 相同的执行序、别名布局、提交顺序（接 tasks 确定性归并 §24.7），这是可复现调试与 golden 测试的前提。

### 19.2 别名不重叠证明

别名是性能收益最大、也最危险的优化。证明路径：生命周期分析（§7.1）为每个瞬态资源算出 `[first_use, last_use]` 的 pass 序区间 → 别名分配（§7.3）只把区间不相交的资源放同块 → 分配后做一次全局校验：遍历每个物理块上的所有资源对，断言区间两两不交。任何相交即编译失败（而非运行期踩踏），错误信息指出「资源 A（区间 x）与资源 B（区间 y）被别名到同块但区间相交」。§14.3 的别名时间轴是这个证明的人眼可视化。

### 19.3 Validation 层（可关的运行期交叉校验）

`validation` feature（默认 debug 开、release 关）在运行期交叉校验编译器输出：
- 资源访问合规（不变量 1）；
- 每个 pass execute 前，其读资源确已被上游写过（不读未初始化）；
- 屏障实际转换的 before 状态与 Graph 追踪的当前状态一致（抓状态追踪 bug）；
- 跨队列资源在所有权转移点确有对应的 release/acquire 屏障对（抓异步同步漏插）。
与 Driver 的 `validation` 档 + Vulkan validation layer 叠加，形成「Graph 层 → RHI 层 → 驱动层」三重校验网。

### 19.4 差分 / 属性 / 模糊测试调度器

- **差分测试**：同一张图在不同配置下跑，断言画面/状态等价——`alias on` vs `alias off`、`async-compute on` vs 单队列、`subpass-merge on` vs 独立 pass、全量编译 vs 增量编译（§17.2）。任何差异即某项优化破坏了语义等价（这些优化全部承诺「仅影响性能不影响正确」）。
- **属性测试**：随机生成合法图（随机 pass/资源/依赖），断言不变量 1–5 恒成立，无论结构多复杂。
- **模糊测试调度器**：在合法执行序空间内随机扰动调度（§9.3 列表调度的 tie-break、并行录制的归并顺序），断言结果不变——抓出「意外依赖于某个特定调度顺序」的隐藏 bug。
- 全部压在 `backend-null` 上跑，无需真实 GPU，CI 可大规模并行跑。

### 19.5 Golden 快照即规格

§14.4 的 Null 后端 golden 快照（执行序/剔除/别名/屏障的序列化）是编译器行为的**可执行规格**：任何改动若改变输出，CI 红，强制 reviewer 确认「这是预期的行为变更」还是「引入了 bug」。编译器的每一个决策都被钉死、可审计。

---

## 20. 易用性、可维护性、扩展性

一个强大但难用、难改、难扩的渲染图等于没用——它会被渲染工程师绕过去手搓。本章保证「好用、好维护、好扩展」是一等目标。

### 20.1 API 工效（易用性）

- **最小认知负担**：写一个 pass 只需 setup（声明读写）+ execute（录命令）两个闭包，不碰屏障/别名/同步/队列——这些框架全包。新手能画出正确的 pass，专家能用逃生门（§8.7 显式屏障、§11.4 合并提示）精调。
- **类型化句柄**（§4.3）：`Handle<Texture>` vs `Handle<Buffer>` 编译期区分，接错类型编译不过；访问推导（§4.2）从 execute 里的用法反推 read/write，减少手写声明的遗漏。
- **内置 pass 范例库**：fullscreen blit、clear、copy、mip 生成、常见后处理骨架作为开箱即用的 pass，照着改即可。

### 20.2 错误诊断（易用性 × 可维护性）

- 契约违反的报错**指名道姓**：不是「依赖错误」而是「pass `Lighting` 读资源 `GBufferNormal`，但无任何上游 pass 写过它（最近的写者候选：无）」。缺黑板键报缺的类型名（§13）。别名相交报具体的两个资源与区间（§19.2）。
- 报错在**编译期**而非运行期崩溃，带足够上下文定位到具体 pass/资源/行。配合 §14 可视化，「图画错了」变成可诊断而非玄学。

### 20.3 从 Bevy 迁移映射（可维护性）

Prism 是 Bevy 的 fork，渲染工程师的肌肉记忆来自 `bevy_render` 的 `RenderGraph`。提供明确映射：Bevy 的 `Node` → Prism 的 pass（setup/execute 分离）；Bevy 的 `SlotInfo`/`SlotType` 边 → Prism 的类型化句柄 + 读写声明；Bevy 的手动 `add_node_edge` 连边 → Prism 的自动依赖推导（§6，不用手连）；Bevy 的 `RenderGraphContext` 取资源 → Prism 的 blackboard（§13）。迁移文档逐 API 对照，并标注「Prism 自动化了 Bevy 需手做的 X」。

### 20.4 可插拔编译器 Pass（扩展性核心）

编译器本身是一条**可插拔的 compiler pass 流水线**（§12 compile 阶段）：cull → schedule → lifetime → alias → barrier → subpass → plan，每一级实现统一的 `CompilerPass` trait（输入上一级产物，输出下一级输入）。扩展点：
- 可**替换**某级实现（如换一个更激进的别名策略、换一个针对特定后端的调度器）而不动其余；
- 可**插入**自定义级（如插一个「调试染色」pass 给所有 pass 加 debug marker、插一个「预算裁剪」pass）；
- 可**关闭**某级（关 alias/subpass 退化为安全路径，§7.7/§11.4），用于隔离问题。
这让编译器成为开放架构而非黑盒——研究新调度/别名算法时直接替一级对拍，不改框架。

### 20.5 自定义 Pass / 资源 / 调度策略扩展点

- **自定义 pass 类型**：除内置 Raster/Compute/Transfer，可注册自定义 pass 类别（如光追 pass、AS 构建 pass），框架按其声明的资源访问纳入依赖/屏障推导。
- **自定义资源类型**：除 texture/buffer，可引入新资源类（如加速结构、稀疏瓦片池），实现状态模型接口即可参与屏障推导。
- **自定义调度策略**：§20.4 的可插拔调度器允许业务按场景定制（如 VR 优先低延迟、移动端优先省带宽）。
- 扩展点全部走 trait + feature，不改核心，符合「开放扩展、封闭修改」。

### 20.6 测试策略（可维护性）

分层测试：单元（各编译器 pass 独立正确性）→ golden（Null 后端端到端快照，§19.5）→ 差分/属性/模糊（§19.4）→ 真实后端画面对拍（接 Driver 的黄金图像门禁）。重构编译器时，golden + 差分测试是「没改坏」的安全网，这是本 crate 敢于持续优化的底气。

---

## 21. crate 分层与模块布局

```
pkg/prism_render_graph/
├── Cargo.toml
│     # features:
│     #   alias            内存别名（可关，退化为满铺，§7.7）
│     #   async-compute    异步计算/传输队列（可关，退化单队列，§9.5）
│     #   subpass-merge    TBDR subpass 合并（可关，退化独立 pass，§11.4）
│     #   compile-cache    编译缓存 + 增量重编译（§17.1/17.2）
│     #   bindless-barrier bindless 堆级保守屏障（§17.7）
│     #   gpu-driven       GPU-driven / indirect 编排（§18.1）
│     #   mesh-shader      mesh/task shader pass（§18.3，门控）
│     #   raytracing       BLAS/TLAS 构建 pass（§18.4，门控）
│     #   multiview        多视图/VR（§18.6，门控）
│     #   validation       运行期交叉校验层（默认 debug 开，§19.3）
│     #   dot-export       DOT 结构导出（§14.1）
│     #   serde            ExecutionPlan JSON 导出（§14.1）
│     #   gpu-timing       per-pass GPU 计时（接 diagnostic，§14.2）
├── src/
│   ├── lib.rs                 # 门面导出 + prelude：RenderGraph / PassBuilder / Handle
│   ├── graph.rs               # 图容器：pass/resource 登记、构图入口、子图实例化（§18.11）
│   ├── pass.rs                # Pass 定义：Raster/Compute/Transfer/自定义；setup/execute
│   ├── builder.rs             # PassBuilder：create/read/write/import、附件声明、访问推导（§4）
│   ├── resource.rs            # 虚拟资源 desc、版本、尺寸/用途推导（§5）
│   ├── handle.rs              # 强类型 generational 句柄与版本（§4.3，接 utils）
│   ├── blackboard.rs          # 类型化黑板 + 作用域黑板（§13）
│   ├── compile/               # 可插拔编译器 pass 流水线（§12/§20.4）
│   │   ├── mod.rs             # CompilerPass trait + 流水线编排 + 并行 compile（§17.5）
│   │   ├── cull.rs            # 无用 pass 剔除（§6.2）
│   │   ├── schedule.rs        # 拓扑排序 + 队列分配 + 关键路径列表调度（§6/§9.3）
│   │   ├── lifetime.rs        # 生命周期区间分析（§7.1）
│   │   ├── alias.rs           # 内存别名分配 + 不重叠证明（§7.3/§19.2）
│   │   ├── barrier.rs         # 屏障/状态转换计划 + 批合并 + split（§8）
│   │   ├── subpass.rs         # TBDR subpass 合并 + load/store/resolve 推导（§11）
│   │   ├── cache.rs           # 结构哈希 + 编译缓存 + 增量重编译（§17.1/17.2）
│   │   └── plan.rs            # 不可变 ExecutionPlan + plan 池化（§12/§17.5）
│   ├── execute.rs             # 执行器：realize/record/submit/present/recycle（§12）
│   ├── parallel.rs            # 并行录制 + 确定性归并（接 tasks，§10）
│   ├── history.rs             # 持久资源与 history 环 + 失效/重建（§16）
│   ├── validation.rs          # 运行期交叉校验层（§19.3，validation 档）
│   ├── viz.rs                 # DOT/JSON 导出、GPU 计时、别名时间轴、HUD（§14）
│   ├── backend_bridge.rs      # 向 render_driver 的资源/屏障/提交适配（§15.1）
│   └── arena.rs               # 帧竞技场 / 零分配编译缓冲（§17.3，接 utils）
└── tests/
    ├── golden/                # Null 后端 golden 快照：剔除/序/别名/屏障（§14.4/§19.5）
    ├── diff/                  # 差分测试：alias/async/subpass/增量 on-vs-off（§19.4）
    └── property/              # 属性 + 模糊测试：随机图断言不变量（§19.4）
```

- **依赖方向**：仅向下依赖 `prism_render_driver`（RHI）+ L1 地基（`prism_math` 尺寸/视口、`prism_utils` 句柄表/竞技场/位集/拓扑排序/内容寻址缓存、`prism_diagnostic` 计时/标记/可视化、`prism_tasks` 并行+确定性归并）；**无任何 `bevy_*`**。
- **feature 门原则**：所有性能/高级 feature 可关，关掉退化为语义等价的安全路径（别名→满铺、异步→单队列、合并→独立 pass、增量→全量），只影响性能/效果上限，不影响正确性——这是「敢优化」的结构保证。

---

## 22. 契约、不变量与版本化

**对上游（scene/material/shader）契约**：
- 输入 draw list / PSO / bindgroup 为不透明句柄，Graph 不解析其内部语义；格式/版本由上游与 Driver 共同保证。
- pass 的 setup 必须声明全部资源访问（声明即全集，§19 不变量 1）；execute 必须只在声明范围内操作。

**对下游（render_driver）契约**：
- Graph 只经 Driver 门面访问 GPU，别名经 `placed` 子分配器、屏障经 barrier API、多队列经 timeline semaphore（§15.1）。
- 所有高级编排用前查 `GpuCaps`，缺失即降级（Driver §3）。

**内部不变量**：§19.1 的五条（声明即全集 / 依赖完备 / 别名不重叠 / 屏障覆盖 / 确定性），编译期证明 + 运行期 `validation` 校验。

**版本化**：
- `ExecutionPlan` 的序列化格式、黑板键集语义、DOT/JSON 导出 schema 均带版本号，演进走弃用期。
- `CompilerPass` trait（§20.4）、自定义 pass/资源扩展点接口为版本化的扩展契约，破坏性变更需 major bump。
- 对 Driver 的屏障/提交接口、对 scene 的 draw list 格式作为跨 crate 契约集中声明并版本化，变更双方协同。

---

## 23. 路线图（M0–M8）与基准即规格

> 节奏与 Driver 咬合：Graph 的 M0 可在 Driver Null 后端先行（纸面编译正确）；别名/屏障/多队列等需 Driver 对应能力就位（Driver §22 的 M1/M2）。核心价值落点是 **M2（别名省显存）+ M4（异步+并行）+ M6（编译缓存近零开销）**。

- **M0 骨架**：Pass/Resource/Handle/Builder + 单队列、无别名、全屏障兜底、串行录制；Null 后端跑通「声明→编译→执行」。基准：编译+执行无 panic，golden 执行序快照。
- **M1 剔除 + 依赖**：SSA 写版本法依赖推导（§6.1）、无用 pass 剔除（§6.2）、细粒度依赖（§6.3）；基准：golden 剔除/排序快照，假串行率。
- **M2 别名**：生命周期分析（§7.1）+ 两档别名分配（§7.3）+ aliasing barrier（§7.4）+ 不重叠证明（§19.2）；基准：峰值瞬态显存 vs 无别名的下降比，别名时间轴可视化。
- **M3 屏障优化**：三元组状态模型（§8.1）+ 批合并（§8.3）+ split barrier（§8.4）+ 冗余消除（§8.5）；基准：屏障数、GPU 气泡（接 diagnostic GPU 计时）。
- **M4 异步 + 并行**：异步计算/传输队列（§9）+ 跨队列 timeline semaphore + 关键路径列表调度 + 并行录制确定性归并（§10）；基准：异步重叠率、录制墙钟并行加速比。
- **M5 TBDR + history**：subpass 合并（§11）+ load/store/dontcare/memoryless 推导 + 持久/history 环（§16）；基准：移动端带宽、TAA 可跑、history 失效正确性。
- **M6 编译缓存 + GPU-Driven**：结构哈希 + 编译缓存 + 增量重编译（§17.1/17.2）+ 零分配竞技场（§17.3）+ indirect/count 编排（§18.1）；基准：稳定帧 compile 耗时趋近 0、每帧堆分配字节数、GPU-driven CPU 提交降幅。
- **M7 子图 + 高级能力**：可重入子图实例化（§18.11）+ 多视图（§18.6）+ VRS（§18.5）+ 两阶段 HZB（§18.1）+ 条件执行（§18.7）；基准：N 级联阴影的结构复用率、多视图一趟 vs N 趟开销。
- **M8 前沿编排**：mesh/task shader pass（§18.3）+ 光追 BLAS/TLAS 构建 pass（§18.4）+ work graph（§18.2）+ 异步时间线跨帧 pass（§18.8）+ 显存预算降级（§18.9）；基准：各能力门控降级路径正确性、GPU 占用率提升。

**基准即规格**：每里程碑的峰值瞬态显存、稳定帧 compile 耗时、每帧堆分配、屏障数、异步重叠率、并行录制加速比、剔除率全部写进 CI 门禁（接 diagnostic），回退即红。差分测试（§19.4）保证每项优化开/关的语义等价也是门禁。

---

## 24. 诚实边界与风险

本文为设计规格，**当前无代码**；M0–M8 均为 PLANNED。本 crate 是 Driver（最大缺口）之后的紧邻缺口，价值在于「把现代渲染技术的编排复杂度一次性封装、处处复用」，但其正确性风险高度集中在自动化的屏障/别名/同步上。

**高风险项**：
1. **别名正确性（最高危）**：生命周期算错 → 两个「活着」的资源踩同块显存 → 随机、难复现的画面损坏。缓解：编译期不重叠证明（§19.2）+ Null 后端 golden（§14.4）+ 别名时间轴可视化（§14.3）+ `alias` feature 一键关闭定位问题 + 差分测试 alias on-vs-off。
2. **跨队列同步**：异步计算的 timeline 信号量 / 队列族所有权转移漏插 → GPU 挂起或竞态。缓解：框架统一插同步、默认禁止跨队列别名、`async-compute` 可关退化单队列安全档、validation 层校验所有权转移对（§19.3）。
3. **编译开销成为 CPU 瓶颈**：复杂帧每帧全量编译（数百 pass）可能先于 GPU 撞墙。缓解：§17 编译缓存 + 增量重编译 + 零分配，结构稳定帧近零；但增量重编译自身的正确性是新风险，用全量-增量差分测试（§17.2/§19.4）兜底。
4. **TBDR 合并的后端差异**：subpass / input attachment / tile shader 在 Vulkan/Metal/D3D12 语义不完全对齐，桌面「假 tile」与移动真 tile 行为有别。缓解：合并仅在可证等价处启用，`subpass-merge` 可关安全退化为独立 pass + 屏障。
5. **声明式学习曲线 + 调试心智**：从命令式手连边转到声明读写、把屏障/别名交给框架，渲染工程师需适应；出错时「框架替我做的决策」不透明。缓解：§20 迁移映射 + 指名道姓的编译期报错 + §14 可视化把隐式决策全显式化 + 逃生门（显式屏障/合并提示）保留专家控制权。
6. **高级能力的平台不均**：mesh shader / 光追 / work graph / VRS / multiview 在 WebGPU 与部分移动端缺失，每条高级编排路径必须有降级路径，否则这些平台黑屏。缓解：全程 `GpuCaps` 门控 + feature 可关 + 每能力明确降级目标（§18 逐项已标）。
7. **跨帧状态 bug**：history 读错槽 / 用脏历史 / 漏失效（分辨率变更）→ 花屏且难复现。缓解：history 槽轮换与失效由框架统一管（§16）、决策序列化进 ExecutionPlan 可观测、首帧 fallback 契约化。

**与既有文档关系**：
- 向下依赖 `prism_render_driver_design_zh.md`（RHI：§6 资源 / §10 同步屏障 / §12 RenderPass / §13 多队列 / §14 显存子分配 / §16 集成章与本文对接；高级能力对应 Driver §24 的 GPU-driven/光追/work graph/VRS）。
- 并行录制依赖 `prism_tasks_design_zh.md`（§24.7 确定性归并保证提交序逐帧一致、§24.8 WaitGraph、§24.4 标定 grain）。
- 地基依赖 `prism_math_design_zh.md`（尺寸/视口/视图矩阵）、`prism_utils_design_zh.md`（句柄表/竞技场/位集/拓扑排序/内容寻址缓存）、`prism_diagnostic_design_zh.md`（GPU 计时 / pass debug marker / 统一时间线 / 图可视化 / 显存图谱）。
- 向上服务 `prism_render_scene` / `prism_material` / `prism_shader` / `prism_ui` 与 streaming（见各自 `*_design_zh.md`）；draw list / PSO / 材质批次 / 流送驻留作为外部输入接入。
- 全景与分层见 `prism_engine_component_gap_zh.md`（render_graph 为 render_driver 之后的紧邻缺口），整体渲染路径见 `prism_rendering_architecture_zh.md`。

所有 Prism crate 不含任何 Unreal Engine / Unity / Frostbite 源码或衍生代码；仅借鉴公开架构形态与经典数值。
