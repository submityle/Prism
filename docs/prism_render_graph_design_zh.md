# Prism Render Graph 顶级次世代 AAA 级帧渲染依赖图设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **Render Graph（渲染图 / 帧依赖图，又称 Frame Graph）**：把「一帧要画什么」描述成一张 **pass（通道）× resource（资源）** 的有向无环图，由框架自动完成**依赖排序、无用通道剔除、瞬态资源分配与内存别名复用、资源状态转换与屏障插入、多队列（异步计算/传输）调度、并行命令录制**。它坐在 `prism_render_driver`（RHI）之上、各渲染特性（material / scene / lighting / post）之下，是渲染器的「编排大脑」：渲染代码只声明「我要读谁、写谁」，何时转换、何时同步、显存怎么复用全交给图。
> 借形态不抄码。借鉴：
> - **帧图开创**：Frostbite FrameGraph（Yuriy O'Donnell, GDC 2017，瞬态资源 + 自动别名的范式奠基）
> - **引擎级渲染依赖图**：Unreal `RDG`（Render Dependency Graph，setup/execute 分离、异步计算、RDG 资源）、Unity SRP RenderGraph（HDRP 的 pass/资源虚拟化）
> - **开源实现**：Granite（Themaister 的 Vulkan render graph，屏障/别名/信号量自动化）、bevy `render_graph`（节点/边/slot 形态）、The-Forge / Diligent 的帧级编排
> - **移动/TBDR**：Metal / Vulkan 的 subpass merge、tile memory、`load/store/resolve` 语义（面向移动与 Apple GPU 的 on-chip 优化）
> 本文为纯经典图形系统编程路线，**不含任何 AI/ML 内容**（不涉及任何学习型调度；排序/别名/屏障均为确定性经典算法）。

- 版本: v0.1（设计阶段，未进入编码；均为 PLANNED，无代码）
- 适用引擎: Prism（后 Bevy 时代，独立运行时）
- 关键依赖: `prism_render_driver`（RHI：Device/Queue/资源/屏障/Fence/Semaphore）、`prism_math`（视口/尺寸/打包）、`prism_utils`（句柄表/竞技场/位集/拓扑排序）、`prism_diagnostic`（GPU 计时/pass 标记/图可视化）、`prism_tasks`（并行录制与 setup 并行）
- 层级定位: L4 表现层（向下只依赖 `prism_render_driver` 与 L1 地基；向上被 `prism_render_scene` / `prism_material` / `prism_lighting` / `prism_post` / `prism_ui` 使用）
- 明确约束: 声明式、无全局可变状态、构图与执行分离；一帧一张图、可完全丢弃重建；屏障与别名由框架推导且可被显式覆盖；`std` 必需；**不依赖任何 `bevy_*` crate**（`bevy_render::render_graph` 仅作形态参考，不链接）

---

## 目录

1. 设计哲学与目标
2. 参考产品取舍
3. 核心概念：Pass / Resource / Handle / Graph
4. 构图 API：setup/execute 分离与 builder
5. 资源虚拟化：瞬态资源与导入资源
6. 依赖推导与无用通道剔除
7. 资源生命周期分析与内存别名
8. 屏障与状态转换自动插入
9. 多队列：异步计算与传输调度
10. 并行命令录制（接 tasks）
11. 子通道合并与 TBDR tile 内存优化
12. 执行模型：编译 → 执行 → 回收
13. Blackboard 与跨 pass 数据流
14. 图可视化与调试（接 diagnostic）
15. 与 render_driver / shader / material / scene 集成
16. 持久化资源与跨帧状态（history buffer）
17. 高级功能增补（AAA）
18. 性能工程
19. 易用性与 Bevy 迁移策略
20. crate 分层与模块布局
21. 契约、不变量与版本化
22. 路线图（M0–M6）与基准即规格
23. 诚实边界与风险

---

## 1. 设计哲学与目标

现代渲染一帧由几十到上百个 pass 组成（depth prepass、GBuffer、阴影、SSAO、光照、透明、后处理、UI……），每个 pass 读若干纹理/缓冲、写若干纹理/缓冲。在显式 API（Vulkan/D3D12/Metal）下，必须手动管：资源状态转换（barrier）、队列间同步（semaphore）、瞬态显存的分配与复用。如果让每个渲染特性各自硬编码这些，一帧的编排就会碎成几十处互相耦合的手工同步，改一个 pass 顺序就可能漏一个屏障 → 画面错误或 GPU 挂起。

`prism_render_graph` 的使命：**让渲染特性只声明「读谁、写谁」，把「何时转换、何时同步、显存怎么复用、哪些 pass 其实没人用」全部交给图自动推导**。构图（declare）与执行（execute）彻底分离——构图阶段只记录意图、不碰 GPU；编译阶段做依赖排序 / 剔除 / 生命周期 / 别名 / 屏障；执行阶段才真正录命令。

**一句话定位**：`prism_render_graph` 是 Prism 的「帧编排大脑」——输入一组声明式 pass，输出一份被剔除、排序、别名、加屏障、分队列、可并行录制的最优执行计划；渲染特性永不手写屏障与瞬态显存，只描述数据依赖。

四条总目标（按权重）：

1. **性能**：瞬态资源内存别名（峰值显存可降 30–60%）；无用 pass/资源自动剔除；屏障合并与最小化（split barrier）；异步计算与传输重叠；并行录制跑满核心。
2. **正确（效果地基）**：依赖推导零遗漏——读写关系唯一决定执行序与屏障；别名资源的生命周期不重叠由编译期保证；跨队列同步由框架插信号量。
3. **易用**：一个 pass = 一个 setup 闭包（声明读写、拿句柄）+ 一个 execute 闭包（拿 RHI 命令录制）；资源不用预分配，图替你算尺寸与生命周期。
4. **可测 + 可视**：图可 dump 成 DOT/JSON，渲染调试器看依赖/别名/屏障；Null 后端下可做无 GPU 的编译正确性单测（剔除/排序/别名/屏障计划全部可断言）。

---

## 2. 参考产品取舍

| 来源 | 借鉴什么 | Prism 取舍 |
|---|---|---|
| Frostbite FrameGraph（GDC 2017） | 瞬态资源 + 自动别名 + setup/execute 分离的开创范式 | 作为核心范式基线：虚拟资源 + 生命周期别名 |
| Unreal RDG | setup/execute lambda、RDG 资源句柄、异步计算、pass 剔除 | 借 builder + 句柄 + 异步计算调度形态 |
| Unity SRP RenderGraph | pass/资源虚拟化、HDRP 的资源池与 import 机制 | 借资源池与导入外部资源的接口形态 |
| Granite（Themaister） | Vulkan 下屏障/别名/信号量全自动、subpass 合并 | 借屏障推导与 TBDR subpass 合并实现思路 |
| bevy `render_graph` | 节点/边/slot、子图（sub-graph）组织 | 借子图与节点复用理念，不绑其 slot 运行时 |
| Vulkan render pass / Metal tile | load/store/resolve、subpass、tile memory | 作为 TBDR 优化的后端映射目标 |
| D3D12 resource state / split barrier | 显式状态与分离屏障 | 作为屏障推导的能力上界 |

**不做**：不做具体渲染算法（PBR/阴影/后处理归 `prism_render_*`，图只提供编排）；不做 GPU 命令原语（归 `prism_render_driver`，图只调 RHI）；不做着色器编译（归 `prism_shader`）；不做场景裁剪与可见性（归 `prism_render_scene`/`prism_render_visibility`，图只吃其产出的 draw list）。

---

## 3. 核心概念：Pass / Resource / Handle / Graph

- **Graph（图）**：一帧的全部编排，一次性构建、编译、执行、丢弃（或池化复用结构）。
- **Pass（通道）**：图的节点。分三类——`Raster`（有渲染附件的光栅通道）、`Compute`（计算）、`Transfer`（拷贝/清除/mip 生成）。每个 pass 带一个 setup（声明依赖）与一个 execute（录命令）。
- **Resource（资源）**：图的「虚拟」资源，分 `Texture` / `Buffer`。编译前只是「描述 + 句柄」，不占显存；编译后被分配/别名到真实 RHI 资源。
- **Handle（句柄）**：`TextureHandle` / `BufferHandle`，强类型、带版本（见 §6 写版本）。setup 返回句柄，execute 用句柄向图索取真实 RHI 资源。
- **ResourceDesc**：纹理（尺寸/格式/mip/采样数/用途位）或缓冲（大小/用途位）的描述；尺寸可相对交换链（如「半分辨率」）。
- **Access（访问）**：pass 对资源的意图——读 / 写 / 读写 / 作为附件（color/depth/resolve）；访问类型决定屏障的 src/dst 状态与排序边。

资源三态：**Transient（瞬态）**——仅本帧存在，由图分配与别名；**Imported（导入）**——外部（交换链 backbuffer、持久 history、常驻资产纹理）注入图，图只管状态转换不管分配；**Persistent（持久）**——由图分配但跨帧保留（见 §16）。

---

## 4. 构图 API：setup/execute 分离与 builder

构图的核心是「两阶段闭包」：setup 只声明意图（并行、可缓存、不碰 GPU），execute 延迟到编译后真正录命令。形态（PLANNED，示意）：

```rust
// 伪代码：形态示意，非最终 API，均为 PLANNED
let gbuffer = graph.add_raster_pass("gbuffer", |b: &mut PassBuilder| {
    let albedo = b.create_color_attachment("albedo", TextureDesc::rgba8_screen());
    let normal = b.create_color_attachment("normal", TextureDesc::rgb10a2_screen());
    let depth  = b.create_depth_attachment("depth",  TextureDesc::depth32_screen());
    b.read(scene.visible_meshes);          // 外部 draw list（CPU 侧，仅建依赖）
    PassData { albedo, normal, depth }     // 把句柄传给 execute
}, |data, ctx: &mut ExecuteContext| {
    // execute：拿到真实 RHI 资源与命令编码器
    let mut rp = ctx.begin_raster(&[data.albedo, data.normal], Some(data.depth));
    rp.set_pipeline(ctx.pso(PSO_GBUFFER));
    rp.draw_indirect(ctx.buffer(scene.draw_args));
});
```

要点：
- **setup 返回的 `PassData`** 携带句柄，被框架存起来在 execute 时回灌，避免闭包捕获可变借用。
- **create/read/write** 在 builder 上登记，框架据此建依赖边与资源生命周期。
- **声明即契约**：execute 内只能访问 setup 声明过的资源；越权访问在 debug 下 panic（契约校验）。
- **子图（sub-graph）**：可把一组 pass 封装成可复用单元（如「阴影子图」），参数化后插入主图（借 bevy sub-graph 形态）。

---

## 5. 资源虚拟化：瞬态资源与导入资源

瞬态资源是帧图的灵魂：pass 声明「我要一张半分辨率 RG16F」，图在编译期根据**所有瞬态资源的生命周期**决定它们真实占用哪块显存、能否与别的瞬态资源**共用同一块**（生命周期不重叠即可别名）。

- **创建**：`b.create_texture(desc)` / `b.create_buffer(desc)` 返回瞬态句柄；不立即分配。
- **导入**：`graph.import_texture(rhi_tex, current_state)` 把交换链 backbuffer、上一帧 history、常驻资产等注入；图负责把它从 `current_state` 转到首个使用所需状态，并在帧末转回（或交给呈现）。
- **尺寸推导**：`TextureDesc` 支持相对尺寸（`Screen`、`Screen/2`、`Fixed(w,h)`），编译期按当前交换链解析为绝对像素。
- **用途位推导**：图从「谁读谁写、作为什么附件」自动并集出 RHI 的 `TextureUsage`（采样/存储/颜色附件/深度附件/拷贝），无需手填，减少错配。

---

## 6. 依赖推导与无用通道剔除

**依赖推导**：资源的读写关系唯一决定执行序。写版本法（SSA 风格）——每次写一个资源，其句柄「版本 +1」；读某版本即建一条从「产生该版本的 pass」到「本 pass」的依赖边。于是：

- Write-after-Read / Write-after-Write / Read-after-Write 全部由版本边表达，框架据此拓扑排序。
- 不存在隐式全局顺序：两个互不依赖的 pass 可被判定为可并行/可乱序（为异步计算与并行录制开门）。

**无用通道剔除**：从「最终被消费的资源」（交换链 backbuffer、被导出的 history）反向追踪引用计数：

1. 初始化每个资源的 ref count = 「读它的 pass 数」；
2. 把「写了无人读的资源」的 pass 标记为候选死通道；
3. 反向传播：一个 pass 若其所有输出都无人消费且无副作用，则剔除，并把它读的资源 ref count 减一，可能连锁剔除更多上游 pass；
4. 带「副作用标记」（如写 backbuffer、回读到 CPU、external signal）的 pass 永不剔除。

效果：调试期临时加的 pass、被开关关掉的特性分支，无需手动摘除——没人消费就自动不执行、不分配显存。

---

## 7. 资源生命周期分析与内存别名

编译期对每个瞬态资源算出 **[first_use, last_use]** 的 pass 区间（按最终执行序）。两个资源**区间不重叠**且**兼容（同堆/同对齐族）**，即可别名到同一块物理显存。

- **别名算法**：按 first_use 排序，维护「空闲物理块」池（类似寄存器分配 / 线性扫描）；资源在 last_use 后归还其物理块供后续资源复用。
- **堆/类型匹配**：颜色附件、深度附件、存储纹理、普通缓冲可能属不同内存类型；别名只在兼容类别内进行（接 `prism_render_driver` §14 的子分配器与内存类型）。
- **别名屏障**：复用同一块显存的两个资源切换时，需插 **aliasing barrier**（内容失效语义）；框架自动在 last_use(A)→first_use(B) 边界插入。
- **收益与开销**：别名显著降峰值显存；但过度别名会制造「假依赖」（本可并行的 pass 因共享物理块被串行化）。框架提供「别名预算」旋钮：显存紧张时激进别名，显存充裕时放宽以保并行度（见 §18）。

---

## 8. 屏障与状态转换自动插入

每个资源在时间轴上经历一串状态（如 `Undefined → ColorAttachment → ShaderRead → Present`）。框架沿执行序为每条「状态变化」生成 RHI 屏障：

- **状态推导**：由 pass 的 access 类型决定目标状态（写附件→`ColorAttachment`/`DepthAttachment`；读采样→`ShaderRead`；storage→`UnorderedAccess`；拷贝→`CopySrc/Dst`）。
- **屏障合并**：同一提交点上多资源的屏障合并为一次 `pipeline barrier` 调用（减少 API 开销）。
- **Split barrier（分离屏障）**：在「状态不再被需要」处发出 begin，在「下次使用」前发出 end，让 GPU 在两点间隐藏转换延迟（接 RHI 的 split barrier 能力，无则退化为单屏障）。
- **队列族所有权转移**：跨队列（图形↔计算↔传输）使用同一资源时，插入 queue family ownership transfer（Vulkan）/状态转换（D3D12），并配合 §9 的信号量。
- **可覆盖**：极端优化场景允许 pass 显式声明屏障，框架让路并仅校验一致性（契约：显式屏障必须覆盖框架推导的最小集，否则 debug panic）。

---

## 9. 多队列：异步计算与传输调度

依赖图的「可并行」信息（§6）天然支持把无依赖的 compute pass 丢到异步计算队列，与图形队列重叠：

- **队列分配**：pass 可标 `queue_hint = AsyncCompute / Transfer / Graphics`；框架在不违反依赖的前提下尽量满足 hint；资源上传/下载类 pass 优先走传输队列。
- **跨队列同步**：当队列 A 的 pass 产出被队列 B 的 pass 消费，框架在 A 末插 signal、B 前插 wait（timeline/binary semaphore，接 RHI §10）。
- **调度策略**：经典列表调度（list scheduling）——按关键路径优先，把长 compute（如 SSAO、光照剔除）尽早投到异步队列，与阴影/几何的图形队列并行；避免「所有东西挤在图形队列尾部」。
- **安全阀**：跨队列别名的物理块需额外同步；框架默认禁止「异步队列资源与图形队列资源别名同一块」除非显式开启（避免难查的竞态）。

---

## 10. 并行命令录制（接 tasks）

执行阶段，互不依赖的 pass 的 execute 闭包可在 `prism_tasks` 的 worker 上并行录制到各自的 `CommandBuffer`，最后按执行序提交：

- **录制粒度**：每个 pass（或一组相邻 raster pass 共享 render pass）一个二级命令缓冲 / command list bundle。
- **确定性归并**：并行录制的命令缓冲，其**提交顺序**严格按编译期确定的执行序（接 `prism_tasks` §24.7 确定性归并），保证跨运行逐帧一致（对回放/录像/确定性调试至关重要）。
- **负载均衡**：按 pass 的历史 GPU/CPU 耗时（接 §14 diagnostic）估算录制成本，均摊到 worker，避免「一个超重 pass 拖尾」。
- **线程安全**：RHI 命令录制本身线程安全（`prism_render_driver` §9 保证）；图只需保证各 pass 不共享可变录制状态。

---

## 11. 子通道合并与 TBDR tile 内存优化

移动与 Apple GPU 是 TBDR（Tile-Based Deferred Rendering），把渲染目标切 tile 常驻 on-chip。若连续 pass 对同一组附件做「写→读同像素」，可合并进一个 render pass 的多个 subpass，中间结果留在 tile 内存，省掉往返显存的带宽（延迟渲染的 GBuffer→光照是经典场景）。

- **合并条件**：相邻 raster pass、附件集合兼容、后一 pass 对前一输出的读是「同像素 input attachment」（非任意采样）。
- **实现**：Vulkan 用 subpass + input attachment；Metal 用 `tile shader`/programmable blending/imageblock；桌面显式 API 无 tile 概念则退化为普通 pass + 屏障（语义等价，仅不省带宽）。
- **自动 + 可提示**：框架尝试自动识别可合并链；也接受 pass 显式 `merge_hint` 以处理识别不到的情形。
- **load/store/resolve**：图据资源生命周期自动决定附件的 `load`（clear/load/dontcare）与 `store`（store/dontcare/resolve）——瞬态且后续无人读的附件用 `dontcare` 省带宽（移动端收益巨大）。

---

## 12. 执行模型：编译 → 执行 → 回收

一帧流水线（PLANNED）：

1. **构图（record）**：调用各 pass 的 setup，收集 pass/资源/依赖。可并行跑 setup（接 tasks），因 setup 无副作用。
2. **编译（compile）**：剔除无用 pass（§6）→ 拓扑排序 + 队列分配（§6/§9）→ 生命周期分析 + 别名（§7）→ 屏障计划（§8）→ subpass 合并（§11）。产出一份不可变 `ExecutionPlan`。
3. **分配（realize）**：按别名结果从显存池申请物理块，绑定给瞬态资源；导入资源登记当前状态。
4. **执行（execute）**：按计划（可并行录制 §10）调各 pass 的 execute 录命令，插屏障/信号量，分队列提交。
5. **呈现 + 回收（present/recycle）**：呈现 backbuffer；归还瞬态物理块到池；持久资源留存（§16）；`ExecutionPlan` 可结构复用（见 §18 增量重编译）。

**编译缓存**：若图结构（pass 集 + 依赖 + 资源 desc）与上帧一致（常态），复用上帧的 `ExecutionPlan`，只刷新动态数据（draw args、uniform），跳过排序/别名/屏障推导——编译从「每帧算」降为「结构变才算」。

---

## 13. Blackboard 与跨 pass 数据流

Blackboard（黑板）是图级的类型化键值表，解决「上游 pass 产出的句柄要给下游 pass 用」的传参问题，避免手工层层透传：

- `blackboard.set::<GBuffer>(GBuffer { albedo, normal, depth })` 由 gbuffer pass 写入；
- `let gb = blackboard.get::<GBuffer>()` 由光照 pass 读出，拿到句柄建依赖；
- 类型即键（`TypeId`），编译期类型安全；缺键在 debug 下 panic（契约：读黑板前该键必须被某上游 pass 写过）。

黑板只存**图句柄与小型元数据**，不存 GPU 资源本体；跨帧数据走 §16 持久资源 + history 机制，不走黑板。

---

## 14. 图可视化与调试（接 diagnostic）

- **DOT/JSON 导出**：把编译后的图（pass、资源、依赖边、队列、别名组、屏障）导出，渲染调试器或 Graphviz 可视化——一眼看出「谁依赖谁、哪块显存被谁别名、哪里插了屏障」。
- **GPU 计时**：每个 pass 自动包一对 timestamp query（接 `prism_diagnostic` 的 GPU 计时），帧末汇总每 pass 的 GPU 耗时，驱动 §10 负载均衡与性能面板。
- **调试标记**：每个 pass 自动插 debug marker/label（接 RHI 调试标记），RenderDoc/PIX/Xcode 抓帧时 pass 名清晰可见。
- **别名可视化**：显存时间轴图（哪块物理块在哪些 pass 区间被哪个资源占用），直接验证 §7 别名正确性与收益。
- **编译断言**：Null 后端下，可对「剔除结果 / 执行序 / 别名组 / 屏障计划」做快照断言（golden test），任何编排逻辑回归立即被 CI 抓到。

---

## 15. 与 render_driver / shader / material / scene 集成

- **向下（render_driver / RHI）**：图是 RHI 的唯一「帧级大客户」——所有资源分配走 RHI 的显存子分配器（§7 别名复用 RHI 堆），所有屏障/信号量/提交走 RHI 命令接口。图不碰具体后端。
- **shader**：pass 的 execute 用 `prism_shader` 产出的 PSO/绑定布局；图不编译着色器，只在 setup 时据 PSO 的资源绑定推导部分 access（可选，减少手写 read/write）。
- **material**：材质系统把一个 draw 批次喂给 raster pass 的 execute；图负责该 pass 的附件/依赖，不关心材质内部。
- **scene / visibility**：场景裁剪产出的 draw list / indirect buffer 作为「外部输入」被 pass `read`（建依赖，确保 GPU 剔除 compute pass 在 draw pass 前完成）。
- **ui / post**：UI 与后处理各是一组 pass（常含 subpass 合并链），插在主图末尾，向 backbuffer 写最终像素。

契约见 §21：图对 RHI 只用「资源 + 屏障 + 提交」接口，对上只收「声明式 pass」，中间全自动。

---

## 16. 持久化资源与跨帧状态（history buffer）

TAA、SSR、运动模糊、时域降噪等需要「上一帧的某纹理」。图提供持久资源与双缓冲 history：

- **持久资源**：`graph.create_persistent(desc)` 分配跨帧保留的资源，图负责其每帧的状态转换，但不参与瞬态别名（生命周期跨帧，不可被别名覆盖）。
- **History 环**：`graph.history::<T>(n)` 维护 n 帧环形缓冲，本帧写 `[cur]`、读 `[cur-1]`；图自动做 ping-pong 轮换与首帧（无历史）降级（返回「无效/清零」标记供 pass 处理）。
- **分辨率变更**：交换链尺寸变化时，持久/history 资源自动重建并标记「历史失效」（避免拿错尺寸的旧帧）。
- **跨帧依赖的屏障**：读上一帧 history 的 pass 与写本帧 history 的 pass 之间，图保证正确的帧间状态与（必要时）帧间 fence。

---

## 17. 高级功能增补（AAA）

- **GPU-Driven 编排**：支持 pass 以 `indirect` / `multi_draw_indirect_count` 执行，draw 数量由上游 GPU 剔除 compute pass 写入的 count buffer 决定；图把「剔除 compute → draw」的依赖与屏障自动串好（接 RHI 的 indirect 能力）。
- **可变着色率（VRS）附件**：pass 可声明一张 shading rate 附件作为输入，图管理其状态与生命周期；为注视点渲染/性能分级提供编排支持（算法本身不含 ML，仅经典速率图）。
- **多视图 / VR**：pass 支持 `view_mask`（多视图单 pass 渲双眼），图把多视图附件数组纳入生命周期与别名分析。
- **条件执行**：pass 可挂 `condition`（如「仅当该特性开启」「仅当上游 occlusion query 命中」），编译期或执行期裁剪，与 §6 剔除协同。
- **异步时间线**：长耗时 compute（如大 GI 探针烘焙的运行时更新片段）可跨多帧的异步时间线推进，图提供「跨帧 pass」原语与进度令牌（纯调度，不含学习型内容）。
- **瞬态显存预算与降级**：显存紧张时，图可按优先级丢弃「可选质量 pass」（如半分辨率 → 更低、或跳过某后处理），并上报降级事件（接 diagnostic）。
- **可重入子图与实例化**：阴影级联、立方体贴图 6 面、多相机分屏等可用「参数化子图」实例化多份，共享结构、各自资源，减少重复构图开销。

> 以上均为 PLANNED 设计形态；所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。

---

## 18. 性能工程

- **编译缓存 / 增量重编译**：图结构稳定时复用 `ExecutionPlan`（§12）；仅资源 desc/尺寸变化时做局部重算（别名 + 屏障），pass 集变化才全量重编译。
- **零分配热路径**：构图与编译用竞技场/帧分配器（接 `prism_utils`），一帧的临时结构帧末整体回收，避免逐 pass 堆分配。
- **别名 vs 并行度权衡**：§7 的别名预算旋钮；提供「显存优先 / 并行优先」两档策略与自动模式（按当前显存占用自适应）。
- **屏障最小化**：合并 + split barrier（§8）；避免「全局屏障」兜底，按资源精确转换。
- **异步计算收益守恒**：list scheduling（§9）把 compute 塞进图形队列的「气泡」；提供「异步收益」统计（异步 pass 实际与图形重叠了多少 µs）。
- **基准即规格**：编译耗时、峰值显存、屏障数、异步重叠率全部进 §22 基准门禁。

---

## 19. 易用性与 Bevy 迁移策略

- **心智模型**：Bevy 的 `RenderGraph` 是「节点 + slot + 边」命令式连线；Prism 用「setup 声明读写 + 自动推导」声明式——用户不再手连边、手填 slot，只说依赖。
- **迁移映射**：Bevy node → Prism pass；node 的 input/output slot → setup 的 read/create；手工 `add_node_edge` → 自动依赖推导（删除）；`get_render_graph` 子图 → Prism sub-graph。
- **渐进接入**：迁移期可用「导入资源」把现有手工管理的纹理注入图，逐 pass 搬迁，不必一次性重写整条渲染管线。
- **防呆**：越权访问未声明资源、读未写过的黑板键、显式屏障不覆盖推导集，debug 下全部 panic 并给出 pass 名与资源名，定位成本低。
- **文档即示例**：每个内置 pass（gbuffer/shadow/light/post）作为可读范例，用户照猫画虎加自己的 pass。

---

## 20. crate 分层与模块布局

```
pkg/prism_render_graph/
├── Cargo.toml                 # features: alias, async-compute, subpass-merge, dot-export, serde
├── src/
│   ├── lib.rs                 # 门面导出：RenderGraph / PassBuilder / Handle
│   ├── graph.rs               # 图容器：pass/resource 登记、构图入口
│   ├── pass.rs                # Pass 定义：Raster/Compute/Transfer、setup/execute
│   ├── builder.rs             # PassBuilder：create/read/write/import、附件声明
│   ├── resource.rs            # 虚拟资源 desc、句柄、版本、用途推导
│   ├── handle.rs              # 强类型句柄与版本
│   ├── blackboard.rs          # 类型化黑板（§13）
│   ├── compile/
│   │   ├── cull.rs            # 无用 pass 剔除（§6）
│   │   ├── schedule.rs        # 拓扑排序 + 队列分配（§6/§9）
│   │   ├── lifetime.rs        # 生命周期分析（§7）
│   │   ├── alias.rs           # 内存别名分配（§7）
│   │   ├── barrier.rs         # 屏障/状态转换计划（§8）
│   │   ├── subpass.rs         # TBDR subpass 合并（§11）
│   │   └── plan.rs            # 不可变 ExecutionPlan + 编译缓存（§12/§18）
│   ├── execute.rs             # 执行器：realize/record/submit（§12）
│   ├── parallel.rs            # 并行录制（接 tasks，§10）
│   ├── history.rs             # 持久资源与 history 环（§16）
│   ├── viz.rs                 # DOT/JSON 导出与别名时间轴（§14）
│   └── backend_bridge.rs      # 向 render_driver 的资源/屏障/提交适配（§15）
└── tests/                     # Null 后端编译正确性 golden test（剔除/序/别名/屏障）
```

- 依赖方向：仅向下依赖 `prism_render_driver` + L1（math/utils/diagnostic/tasks）；**无任何 `bevy_*`**。
- feature 门：`async-compute` / `subpass-merge` 可关（退化为单队列/普通 pass，语义等价）；`dot-export` / `serde` 仅调试与工具链用。

---

## 21. 契约、不变量与版本化

**不变量（编译期/运行期保证）**：
1. **声明即全集**：execute 只能访问 setup 声明的资源；越权 = 契约违反。
2. **依赖完备**：所有 read-after-write 必有依赖边；无「隐式顺序」。
3. **别名不重叠**：被别名到同块显存的资源，生命周期区间两两不重叠（编译期证明）。
4. **屏障覆盖**：每条状态变化必有对应屏障；显式屏障必须覆盖推导最小集。
5. **确定性**：相同图结构 + 相同输入 → 相同执行序、相同别名布局、相同提交顺序（接 tasks 确定性归并）。

**版本化**：`ExecutionPlan` 格式与黑板键集随 crate 语义版本演进；DOT/JSON 导出带 schema 版本；跨 crate 契约（对 RHI 的屏障/提交接口、对 scene 的 draw list 格式）集中声明并版本化，变更走弃用期。

---

## 22. 路线图（M0–M6）与基准即规格

- **M0 骨架**：Pass/Resource/Handle/Builder + 单队列、无别名、全屏障兜底、串行录制；Null 后端跑通「声明→执行」。基准：编译+执行无 panic，golden 执行序。
- **M1 剔除 + 依赖**：写版本法依赖推导、无用 pass 剔除；golden 剔除/排序快照。
- **M2 别名**：生命周期分析 + 线性扫描别名 + aliasing barrier；基准：峰值显存 vs 无别名的下降比。
- **M3 屏障优化**：屏障合并 + split barrier + 用途/状态精确推导；基准：屏障数、GPU 气泡。
- **M4 异步 + 并行**：异步计算队列 + 跨队列信号量 + 并行录制（接 tasks）；基准：异步重叠率、录制墙钟。
- **M5 TBDR + history**：subpass 合并 + load/store/dontcare + 持久/history 环；基准：移动端带宽、TAA 可跑。
- **M6 编译缓存 + GPU-Driven**：增量重编译 + indirect/多视图/VRS 编排；基准：稳定帧编译耗时趋近 0。

**基准即规格**：每里程碑的峰值显存、编译耗时、屏障数、异步重叠率、并行录制加速比写进 CI 门禁，回退即红。

---

## 23. 诚实边界与风险

本文为设计规格，当前无代码；M0–M6 均为 PLANNED。

**高风险项**：
1. **别名正确性**：生命周期算错 → 两个「活着」的资源踩同块显存 → 难查的画面错误。缓解：编译期不重叠证明 + Null 后端 golden + 别名时间轴可视化 + 可一键关闭别名定位问题。
2. **跨队列同步**：异步计算的信号量/所有权转移漏插 → GPU 挂起或竞态。缓解：框架统一插同步、默认禁止跨队列别名、提供「强制单队列」安全档。
3. **编译开销**：复杂帧每帧全量编译可能成为 CPU 瓶颈。缓解：§12/§18 编译缓存 + 增量重编译，结构稳定帧近零开销。
4. **TBDR 合并的后端差异**：subpass/input attachment/tile shader 在各后端语义不完全对齐。缓解：合并仅在可证等价处启用，桌面后端安全退化为普通 pass。
5. **声明式的学习曲线**：从命令式连边转到声明读写需要适应。缓解：§19 迁移映射 + 内置 pass 范例 + 强契约报错。

**与既有文档关系**：
- 向下依赖 `prism_render_driver_design_zh.md`（RHI：§6 资源 / §10 同步 / §13 多队列 / §14 显存子分配 / §16 集成章与本文对接）。
- 并行录制依赖 `prism_tasks_design_zh.md`（§24.7 确定性归并保证提交序逐帧一致）。
- 地基依赖 `prism_math_design_zh.md`（尺寸/视口）、`prism_utils_design_zh.md`（句柄表/竞技场/位集/拓扑排序）、`prism_diagnostic_design_zh.md`（GPU 计时/pass 标记/图可视化）。
- 向上服务 `prism_render_scene` / `prism_material` / `prism_shader` / `prism_ui`（见各自 `*_design_zh.md`）；draw list / PSO / 材质批次作为外部输入接入。
- 全景与里程碑见 `prism_engine_component_gap_zh.md`（render_graph 为 render_driver 之后的紧邻缺口）。

所有 Prism crate 不含任何 Unreal Engine / Unity 源码或衍生代码；仅借鉴公开架构形态与经典数值。
