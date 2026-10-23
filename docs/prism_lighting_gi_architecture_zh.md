# Prism 光照与 GI 架构演进方案（v2.6 · 提案）

> 状态：架构基线 + 确定性 direct-light 与初始 GI reservoir GPU 垂直切片已实施；文中所有帧预算、带宽、周期数仍为**设计目标**，非实测。
> 修订：v2.2 修正 v2.1 的 8 处问题（1 处误差模型错误、1 处混合统计前提缺失、2 处契约设计缺陷、4 处未定义或低估的验收），逐条见 §0.5；v2.3 修正 v2.2 的 6 处（误差不等式方向、variance 的测量阶段、Conservative 权重的粒度、submission/输出的状态机、S1 与决策 #4 的矛盾、P7 的范围蔓延），逐条见 §0.6；v2.4 修正 v2.3 的 4 处（上置信乘子的形式、Conservative 的标量与向量不一致、QP 成本的未验证假设、验收矩阵缺分位与规则自相矛盾），逐条见 §0.7；v2.5 修正 v2.4 的 4 处（Trace 契约未闭合、α 边界的伪精确断言、退路 ③ 与单权重混合矛盾、验收矩阵的 H/S 未成列），逐条见 §0.8。v2.6 补全**镜面反射**这条被压成一行的纵切：recipe 从 13 扩到 16（新增 `SpecularScreenSpace` / `ProbeSplitSum` / `PlanarReflection`）、新增 §5.3.3 镜面信号契约、§18.2 新增 S7 场景类与 4 条镜面门槛、§17 新增 2 项待拍板决策，逐条见 §0.10；并做了一轮**维护性自检**，拆掉本轮自己引入的 3 个回归（§0.11）。**2026-10-20 实施状态更新**：trace-hit→GPU candidate build 已覆盖独立 hit/miss oracle 对拍；GPU 宿主图同帧 trace→candidate→GI→composite 已与 CPU GI oracle 对拍通过。辐射输入现按 BVH 稳定 primitive ID 对齐，支持不可变共享表与 revision 感知上传缓存。另有一个严格受限的材质权威首片：`Surface + Opaque/OpaqueTwoSided + 无纹理 + 无动态通道 + Builtin Unlit` 可经统一材质执行表求值并携带 handle/generation/revision/epoch/execution proof；它尚不是自动场景 producer，也不覆盖 Lit、纹理或命中点重建。该证据仍不覆盖完整材质/纹理求值、LightingPlan trace 准入、scene 输出 parity 或 P7a 完整验收。**本文的工期与验收仍是待验证的假设，不是承诺。**
> 基线：`main`（2026-10-19）；本次实施状态核对于 2026-10-20。当前环境已验证 null backend graph 记录、真实 wgpu adapter 的 shader/pipeline 创建、一帧 dispatch 录制、GPU buffer readback 与 CPU oracle 对拍。本文不把设计目标、完整物理 recipe 或最终画质验收写成已完成事实。
> 范围：`pkg/` 内光照、阴影、全局光照、虚拟流送、增量更新的**架构**演进；不含 `crates/bevy_*` 旧栈改造，也不含具体 GI 算法推导。
> 与既有文档的关系：**不是** GI 算法设计（那是 `prism_gi_lumen_design_zh.md` v4 的职责），而是让 16 个光照/GI/反射技术能共存、可预算、可维护、可扩展的**架构层**方案。
> 关联文档：`prism_gi_lumen_design_zh.md`（GI 子系统设计 v4）、`prism_rendering_architecture_zh.md`（总架构与分档表）、`prism_material_pipeline_refactor_zh.md`（四前端唯一执行路径）、`prism_render_scene_rhi_port_zh.md`（bevy 轨移植切片）、`prism_derived_data_contract_zh.md`（B1 / E1 派生数据登记）。
> 数值路线：**纯经典数值**（蒙特卡洛 / reservoir 重采样 / SH / SDF 步进 / 时空降噪），不含任何 AI / ML / 神经网络路径。

### 0.9 实施落点（2026-10-23 增量更新）

**2026-10-23 实施状态增量**：`WorldRestir` 已登记为 architecture executable provider，provider 复用 `restir_gi` 的 GI sample reservoir、确定性初始候选流和 finalized radiance CPU reference；有效候选输入现在可通过 `LightingPlan` 准入。recipe backend 定向测试 9/9 通过。当前磁盘耗尽中断了 planner 重建验证及强制物理 GPU 宿主图测试，因此**不宣称** trace→candidate→GI→composite 当前端到端验收通过；原有材质/纹理闭环、scene/present parity 和性能矩阵仍未完成。

本轮重构把提案中最容易再次分叉的边界落成了可编译代码，并接入新宿主的真实帧生命周期：

| 设计项 | 实施位置 | 结果 |
|---|---|---|
| `FieldEpoch` / `ChangeJournal` / checkpoint | `pkg/prism_render_architecture/src/lighting/contract.rs` | 字段级 WAL 记录、单调 epoch、显式快照 |
| `TraceQuery` / `RayBackend` / `TraceSubmission` | 同上 | 槽位稳定映射、generation、deadline、`Stale`/`Expired`/取消状态 |
| `Contrib` / `BlendOperator` | 同上 | target 校验，bias 与 variance 分列，禁止有 bias 的无偏混合 |
| 13 个 recipe 白名单与成本表（v2.6 的 3 条镜面 recipe 仍是提案，尚未进入枚举） | `pkg/prism_render_architecture/src/lighting/contract.rs` | `BudgetCompiler` 按能力、确定性、真实 backend 注册和 GPU/带宽/驻留/IO 预算准入并生成拒绝原因；architecture backend 注册 `DirectStochasticRt`、`DirectStochasticVsm`、`SpecializedDeterministic`、`WorldRestir`；GI provider 为 CPU deterministic reference，app GPU pass 仍须独立验收 |
| `CacheEffect` / `ExecPolicy` | 同上 | 缓存失效与执行策略不再混入图资源声明 |
| render-world 计划入口 | `pkg/prism_render_app/src/lighting.rs`、`plugin.rs`、`stats.rs` | 每个抽取帧编译一次，统计选中/拒绝数量，`explain()` 可追溯；保留字段 epoch 与失效计数 |
| 未实现 recipe 准入门禁 | `pkg/prism_render_architecture/src/lighting/contract.rs` | 只有已注册真实执行后端的 recipe 才能进入 `LightingPlan`；其余 recipe 以 `unavailable:<recipe>` 明确拒绝，不再出现“计划选中但没有执行”的假闭环 |
| 计划输入配置与抽取 | `pkg/prism_render_app/src/config.rs`、`lighting.rs`、`plugin.rs` | host 可注入 `SceneStats`、`FrameResourceBudget`、`PlanCaps`、`Tier`，并通过 `LightingSceneInputs`/`ChangeJournal` 跨主世界→render-world 传递快照 |
| wgpu 外部 buffer 注册 | `os-backends/prism_render_driver_wgpu/src/device.rs` | 修复 `DeviceInner` 自身字段的错误访问，恢复后端编译路径 |
| GPU 执行垂直切片（非完整场景闭环） | `pkg/prism_render_app/src/lighting_gpu.rs`、`frame.rs`、`systems.rs`、`tests.rs` | `LightingPlan` 驱动 direct、GI reservoir 与 composite；当 trace 场景输入存在时，GPU candidate-build 按 surface/ray 范围消费 hit、triangle、ray、surface 与 host outgoing-radiance，写入 GI pass 同一 candidate buffer；独立物理 GPU 测试覆盖 hit/miss 并与 CPU candidate oracle 对拍。普通 host-candidate 路径及已有多 surface reservoir GPU 对拍仍保留为测试/输入分支；这不是场景材质求值或完整 GI 闭环 |
| 宿主 trace→candidate→GI 帧图接线 | `lighting_trace_gpu.rs`、`systems.rs`、`frame.rs`、`presentation.rs` | 软件 BVH trace 与独立 candidate-build pass 接入 window/offscreen 同一宿主 RenderGraph；候选 pass 显式依赖 trace pass，读取 hit 写后版本并写入 GI 读取的 candidate buffer；按 surface stable ID/frame 生成 ray。`WorldRestir` 已注册并可通过 `LightingPlan` 候选准入；历史端到端 GPU parity 仍需在当前 checkout 强制重验。BVH 上传按场景缓冲身份缓存，场景或 device generation 改变时重传 |
| 初始 GI RIS 数学与 GPU candidate 构建 | `lighting.rs`、`lighting_gpu.rs`、`frame.rs`、`tests.rs`、`lighting_trace_gpu.rs` | CPU oracle 与 WGSL 均将 Lambert 被积量纳入 `1/π`，Reservoir 归一化以提议总数（含零贡献/miss）为 N；ray ABI 保留 cosine-weighted solid-angle PDF 与稳定 selection variate，hit ABI 保留 packed triangle index。trace-hit→候选 WGSL 使用 triangle-indexed host outgoing-radiance 生成 per-surface candidate；真实物理 GPU hit/miss 与 CPU candidate oracle 对拍通过。radiance 仍是外部输入，尚未从场景材质/纹理求值 |
| 材质权威辐射首片（受限） | `pkg/prism_render_app/src/lighting.rs`、`pkg/prism_render_shading/src/resolve.rs` | host 可将稳定 primitive 关联到权威 `MaterialRecord` 与已验证 `MaterialExecutionBinding`，仅对 `Surface + Opaque/OpaqueTwoSided + 无纹理 + 无动态通道 + Builtin Unlit` 复用统一 Unlit 响应求值，并保留 material handle/generation/revision/epoch/execution proof；不支持的 domain/class/plan/program 显式拒绝。该入口尚未成为自动 scene producer，packed hit 也未携带实例行、UV、平滑法线/切线，因此不能扩张为 Lit/纹理材质闭环 |

本节只记录已经存在的调用链，不把设计目标或未实测的帧预算写成性能结论。

#### 已完成

- `FieldEpoch`、`ChangeJournal`、checkpoint、`TraceQuery`、`TraceSubmission`、`Contrib`、`BlendOperator`、recipe 白名单（当前 `RecipeId::ALL` 为 13 条；v2.6 新增的 3 条镜面 recipe 尚未进入源码）和 `BudgetCompiler` 已落成并接入新宿主的计划编译链。
- **独立验收审计（2026-10-23 更新）**：`RecipeId::ALL` 当前为 13 项；`recipe_backend::registrations()` 是 planner 唯一 backend 白名单，注册 `DirectStochasticRt`、`DirectStochasticVsm`、`SpecializedDeterministic`、`WorldRestir`。`WorldRestir` provider 复用 `restir_gi` GI sample reservoir，提供 deterministic initial candidate stream/reference radiance；候选来源有效时 planner 允许 GI 进入 `LightingPlan`。强制物理 GPU host graph parity 因磁盘耗尽尚未重验，不把 provider 注册等同 GPU 闭环完成。
- `BudgetCompiler` 已增加 backend 注册准入门禁；未注册 recipe 一律以结构化 `Rejection::Unavailable(recipe)` 拒绝，且拒绝发生在 candidate、variant 和 budget admission 之前。当前四个 architecture providers 均有 CPU/reference 执行；`SpecializedDeterministic` 另有 app GPU 执行路径；`WorldRestir` 有 GI reservoir deterministic provider，GPU host graph parity 单列验收。
- `LightingSceneInputs` 已改为真实输入契约：surface batch、directional lights、point/spot lights、按 surface 对齐的 GI candidate batches；抽取到 render-world 后参与 GPU 上传。新 API 不保留单 surface 兼容入口。
- `lighting/recipe_backend.rs` 已补充 DirectStochasticRT 与 DirectStochasticVSM 的可执行 CPU RIS/VSM provider、确定性 twin、shadow-slot 校验、registration/cost 一致性和 bitwise 稳定性测试；这些 backend 尚未接入新宿主的真实 GPU graph，因此文档不把它们写成 GPU 完成项。
- `lighting/trace.rs` 已接入架构模块树：`SwBvhBackend`、稳定 BVH、异步 submit/poll/cancel、部分结果、deadline、generation、device-loss 和 slot 重叠回收均有确定性行为测试；这完成的是 CPU trace backend 与调度契约，不是硬件 RayQuery 或完整 GPU trace graph。
- `prism_render_app/src/lighting_trace_gpu.rs` 已加入 GPU 软件 BVH trace 执行器与 candidate-build kernel：消费冻结 packed node/triangle ABI，验证输入边界与 BVH 可达性，bounded-stack 遍历输出稳定 hit/miss 槽位；8-triangle 多层 BVH 的 65-ray（跨 workgroup、双分支命中、nearest-hit、miss）通过 CPU oracle 对拍。旧 `encode_and_submit` 旁路已删除；window/offscreen 宿主图声明 trace 与 candidate pass，candidate 读取 hit 写后 SSA 版本并显式依赖 trace。准备阶段校验 surface-major ray 数、按 stable primitive ID 完整覆盖且无重复的辐射表、有限非负值、candidate/radiance/surface/uniform 绑定容量及设备 limits，拒绝非法输入而不触发容量断言。GPU 上传按 packed triangle 顺序整理；不可变 BVH/辐射表身份及 revision 不变时复用排序结果和上传，设备资源 bundle replay 后重新上传。候选 kernel 计算几何法线和 solid-angle RIS proposal；独立 hit/miss GPU readback 与 CPU oracle 对拍通过。`WorldRestir` 已注册且当前 planner 源码现允许有效 GI candidate admission，但 planner 重建验证和强制物理 GPU trace→candidate→GI→composite parity 尚待当前状态重验。**未完成边界**：host outgoing-radiance 仍由外部输入而非场景材质/纹理系统生成，present 终端输出 parity 与大场景性能标定也未通过。
- `LightingTraceSceneInput::from_untextured_unlit_materials` 已建立第一条不复制材质模型的权威求值入口：验证材质 handle、domain、render class、动态通道、纹理为空、作者计划与执行表分类一致，只接受单一 Builtin Unlit；实际颜色复用主 resolve 的 `evaluate_unlit_response`，并保存 material generation/revision、snapshot epoch 与五字执行绑定证明。测试覆盖 BVH 重排后的 primitive-radiance 对齐、candidate radiance、epoch 变化失效，以及纹理、动态通道、Lit、计划错配和 MaskBlend 的显式拒绝。该入口仍由 host 显式构造，不是自动场景提取/生产链。
- `lighting/shadow.rs` 已接入架构模块树：shadow slot 选择、VSM 页请求合并、优先级、预算裁剪、物理页分配/驱逐、caster epoch 失效和 RayTrace/VSM/Cascade fallback 均已实现并测试；尚未成为宿主真实阴影 pass。
- `lighting/cache.rs`、`resolve.rs`、`replay.rs` 已接入架构模块树：有界 reservoir/sample cache、软/硬失效、temporal/spatial reuse、raw/accumulated/post variance、三档 denoise/resolve、multi-surface/multi-view BRDF resolve、WAL/checkpoint/snapshot/restore 和 device-generation replay 均已有 CPU reference 与边界测试。
- `lighting/virtual.rs` 已接入架构模块树：cell 到 shadow/GI/cache page 绑定、请求合并与优先级、有界驻留/淘汰、epoch 失效、显式 fallback 和设备恢复 snapshot/restore 已有 CPU reference 与边界测试；尚未连接真实 GPU page pool/streaming IO。
- `lighting/metrics.rs` 已接入架构模块树：H1–H4 × S1–S7、P50/P95/P99/P99.9、GPU/CPU 时间、带宽、驻留、IO fault、variance、bias、temporal instability、PSNR、能量误差与 N/A 规则可生成确定性报告；S7 已登记镜面通量误差、fallback 单调性、SSR seam 与 FLIP 指标，并有稳定序列化名和报告门禁测试。测试样本与阈值仅验证报告逻辑，不是实测结果或架构验收门槛；H/S 矩阵仍需硬件实测校准。
- `lighting/ocio.rs` 已接入架构模块树：ACES、AgX、linear+sRGB、曝光、白平衡、gamut、transfer function、版本化序列化、digest、配置校验和 CPU reference transform 均已完成。`RenderAppConfig.display_transform` 是窗口与 offscreen/headless 共用的配置来源；8×8 WGSL compute pass 采样线性 `Rgba8Unorm` 帧目标并写入持久 display-encoded `Rgba8Unorm` 纹理。窗口 transform 与 surface present 在同一 RenderGraph/队列提交，present 消费变换纹理；headless/offscreen transform 也已加入宿主帧图，并随场景图通过同一次 `execute_tracked` 提交。公开 `OffscreenFrameOutput` 发布最近一次成功提交的终端纹理、extent、frame index、submission 与 device generation，供离线消费者安排后续 GPU 使用或 readback；resize 会先等待 GPU idle，设备代际替换会使旧输出失效并重建 transform 资源。现有真实 GPU texture readback 的 36 组 tone mapping × gamut × transfer CPU oracle 对拍已通过。新增 offscreen 物理 GPU parity 已通过：先执行实际宿主 graph，再向 seeded target 写入多组非零线性 RGB/alpha 值，并在独立的相同 transform graph 中与 CPU reference 对拍；这验证了宿主 graph 可执行及同一 transform 的非零像素数值，但不表示非零样本直接经过了完整 scene host graph。编码边界：帧目标 UNORM 数值是线性场景值；OCIO 输出按配置 transfer function 编码；sRGB attachment 的 present shader 先 decode、再由 attachment 硬件 encode 一次，非 sRGB attachment 直接写入已编码值。未完成：真实 swapchain 像素 readback、显示器色彩管理，以及跨多设备/多窗口的显示配置策略验收。
- `cargo test -p prism_render_architecture --lib lighting --quiet` 于 2026-10-22 在当前 checkout 实际执行 **288 passed; 0 failed; 0 ignored**（退出码 0，耗时 47.74s）。该结果只证明所覆盖的架构/CPU reference 测试通过，不等同于完整渲染效果或 GPU 性能验收。
- `WorldRestir` 的当前 GPU 实例已是真实单跳 deterministic RIS 初始 reservoir，并与 CPU oracle 使用同一计算语义：solid-angle source/target density 权重、稳定选择随机数、Lambert `albedo / π`、入射余弦、visibility 与 GI composite；估计量分母按提交提议数计数，零目标权重提议不会被从 N 中删除。候选 storage 按 surface-range 紧凑打包；GI WGSL 以 runtime-sized candidate storage 按实际 binding 长度寻址，避免固定数组长度截断 batch。host-candidate 多 surface（65 surface、跨 workgroup、不同 surface 候选）有真实 GPU oracle parity。新增 trace candidate kernel 的物理 GPU hit/miss oracle parity 已通过，但**真实材质辐射求值、宿主全链输出 parity、multi-bounce 与时空复用**仍未完成。
- 持久 GPU bundle、uniform/storage buffers、bind group、direct/GI/composite compute pipeline、render graph resource declaration、batch dispatch 和 device-loss replay 已闭合；每帧会清零完整输出容量，空 surface batch 不生成 lighting compute pass。
- CPU direct/GI oracle 已通过单测；null backend graph 记录已覆盖 direct/GI/composite 以及禁用/空 batch 分支；真实 wgpu adapter 的 readback 对拍覆盖 65 个 surface 的 direct/GI/composite、跨 workgroup 结果及输出容量尾部清零。该证据不代表完整屏幕/视图 parity或独立 trace 输入的正确性。

#### 未完成

- 相对于 v2.6 提案目标的 16 个 recipe，当前源码白名单只有 13 项，architecture backend 注册 4 项（`WorldRestir` 与 direct providers 提供 CPU/reference，`SpecializedDeterministic` 另有宿主 GPU pass）；因此仍有 **12 个提案 recipe 没有注册 backend**，其中当前枚举内有 9 个以 `Unavailable` 拒绝，另有 3 个镜面 recipe 尚未进入枚举。未注册项包括确定性 CSM、DDGI/brick GI、screen probe/ReSTIR、screen-space GI、specular ReSTIR、sparse SDF GI、radiance cascades、path-traced offline 及提案镜面族。这些仍是架构契约、CPU/reference 资产或旧实现资产，不得按“已完成”计。
- 当前 direct recipe 已支持最多 1024 个独立 surface sample 的 GPU batch，但仍不是完整屏幕/视图 lighting resolve；尚未接入完整材质 BRDF、可见性场景查询、阴影、反射、透明、后处理和多视图输出；其中**镜面反射的缺口最大**——v2.6 定义的 4 条 specular recipe（`SpecularRestir` / `SpecularScreenSpace` / `ProbeSplitSum` / `PlanarReflection`）没有一条接上 GPU 执行，`post_ssr_trace` 与 `spec_gi_*.wesl` 资产既未转正也未纳入 `LightingPlan`。
- 当前 GI GPU 路径仍是单跳 deterministic 初始 reservoir。存在 trace scene 时，trace hit 已经由独立 GPU candidate-build kernel 转为按 surface 分组的 RIS proposals，并通过宿主图 candidate buffer 进入 GI pass；**这不等于完整材质辐射求值已完成**：当前只有 host 显式构造的无纹理、不透明 Builtin Unlit 子集经过材质权威链，candidate kernel 仍查 triangle-indexed outgoing-radiance，packed triangle 不携带 material identity、instance row、UV、平滑法线或切线，Lit/纹理/动态材质无法在命中点准确重建和求值，自动 scene producer 也尚不存在。candidate kernel 的 hit/miss 物理 GPU 对拍不等于宿主 graph 的 GI/composite/present 端到端输出 parity。按 surface stable ID/frame/sample 可复现的余弦加权 raygen 已接到 app trace 输入。架构层 `RayScheduler`、cache、temporal/spatial reuse、resolve、denoise、variance、virtual residency 和 replay CPU reference 尚未绑定到完整 GPU 场景更新与 multi-bounce 执行链。
- DebugMeta、结构化 EXPLAIN 拒绝原因、shader variant gate、DebugView 消费链和 plugin sync 均已落地；OCIO 配置契约、共享配置入口、window/offscreen texture graph transform、公开 offscreen 输出帧契约及窗口 present 消费链已实现。null backend graph 与公开输出契约测试覆盖宿主接线；非零像素物理 GPU offscreen parity 已通过，覆盖实际宿主 graph 执行和 seeded target 上相同 transform graph 的非零值 CPU 对拍两段，但非零值未直接注入完整 scene host graph。完整 P8 收敛验收、真实 swapchain readback 与显示器色彩管理仍未完成。
- 需要继续完成 cache/page pool/virtual streaming 的真实 GPU 资源驻留、packed-BVH trace kernel 的 `LightingPlan` 准入、场景材质/纹理到 triangle-indexed outgoing-radiance 的真实求值与失效契约、GPU shadow pass、multi-view resolve 的宿主与 GPU graph 接线、完整 BRDF/可见性/反射/透明/后处理链，以及 raw/accumulated/post variance、bias、denoise 的实测标定。metrics 契约虽已具备，H/S 矩阵仍没有完整硬件实测数据。
- P7a 的新宿主 GI 路径已恢复 `WorldRestir` architecture registration，源码准入逻辑不再因 backend 缺失而将 candidate source 强制置为 `Disabled`；但本轮 planner rebuild 和强制物理 GPU parity 因系统磁盘耗尽未能完成，故仍不能声称 trace→GI→composite 强制验收通过。仍未证明 present 终端输出及完整场景帧 parity，也未完成场景材质辐射求值、与冻结 `prism_render_scene` 输出 parity，以及 H1–H4 × S1–S7 最终性能/画质矩阵，因此 **P7a 整体验收未完成**。

因此当前结论是：**架构契约、计划入口、结构化诊断契约、确定性 direct-light 与初始 GI 多-surface GPU/CPU 对拍已完成；完整 lighting/GI 重构尚未完成。** 后续实现必须继续沿用同一 `LightingPlan` 和新输入契约，不保留旧 API，也不能在子系统内新增旁路开关。

---

## 目录

- [0. 摘要与变更日志](#0-摘要与变更日志)
- [1. 审计基线与问题诊断](#1-审计基线与问题诊断)
- [2. 设计原则](#2-设计原则)
- [3. 架构](#3-架构)
- [4. 七个契约](#4-七个契约)
- [5. 算子代数与 recipe 库](#5-算子代数与-recipe-库)
- [6. 规划与预算](#6-规划与预算)
- [7. 增量与失效](#7-增量与失效)
- [8. 执行模型与性能](#8-执行模型与性能)
- [9. 精度、流送与保真度](#9-精度流送与保真度)
- [10. 效果工程](#10-效果工程)
- [11. 易用层](#11-易用层)
- [12. 易维护](#12-易维护)
- [13. 易拓展](#13-易拓展)
- [14. 借鉴总表](#14-借鉴总表)
- [15. 迁移阶段（不承诺单点工期）](#15-迁移阶段不承诺单点工期)
- [16. 风险登记册](#16-风险登记册)
- [17. 决策清单](#17-决策清单)
- [18. 验收矩阵（硬件 × 场景 × 分位）](#18-验收矩阵硬件--场景--分位)
- [附录 A：审计证据索引](#附录-a审计证据索引)
- [附录 B：术语表](#附录-b术语表)
- [附录 C：误差模型与降档（形式化）](#附录-c误差模型与降档形式化)

---

## 0. 摘要与变更日志

### 0.1 一句话

把当前「N 个子系统各自发射线、各自存缓存、各自 24 个 `enabled: false` 开关」的形态，收敛为 **一条执行路径 + 三条纵切链 + 七个契约 + 十六个 recipe**：技术继续增加，但耦合面不再增加。

### 0.2 规模

| 项 | 数值 |
|---|---|
| 主线阶段 | **P0–P8（9 阶段，P7 拆为 P7a/P7b）+ 3 个前置 spike**；不承诺单点工期（见 §15） |
| 烘焙轨道 | 并行，不占主线 |
| recipe 数 | **16**（4 直接光 + 8 漫反射间接光 + 4 镜面反射）；其中 13 条已在白名单，3 条镜面 recipe 于 v2.6 新增、待实现 |
| 契约数 | **7**（v2.2 硬上限 1000 行；v2.1 的 900 行在拆出 `CacheEffect` / `ExecPolicy` 与补全 backend 生命周期后不再现实） |
| 不变量 | **7**（全部进 CI） |
| 补丁数 | **26** |
| crate 数 | **141 → ~101**（归档 39 个零引用 crate）；这是 **P0 的记账动作**，不作 P7 验收（§0.6 第 6 条） |

### 0.3 版本演进

| 版本 | 变更 |
|---|---|
| v1 | 初版：六层分层、五契约、预算二轴、oracle 三层、10 recipe |
| v2 | 六层 + **三条纵切链**；契约 5→7；预算 **4 轴 + 双预算**；oracle 3→4 层（+L4 序列）；新增**执行模型**与**易用层**；新增借鉴对照表 |
| v2.1 | 吸收 **26 条补丁**：14 条来自图形/引擎产品，12 条来自跨行业工程方法（见 §0.4） |
| **v2.2** | 修正 8 处问题：误差模型（bias ≠ variance）、混合的统计前提、`PassDecl` 万能化、`RayBackend` 生命周期缺失、逐帧单调不可计算、DebugView 成本低估、工期不可靠、验收不可度量（逐条见 §0.5） |
| v2.3 | 修正 6 处：`MSE ≈ …` 改为**上界**并给成立条件、variance 按采样/时域/降噪三级分别测量、Conservative 权重的粒度与求解、submission↔output 的稳定映射与状态机、S1 spike 与决策 #4 解耦（允许 mock backend）、P7 拆为「GI 垂直切片边界」与「全仓迁移（范围外）」（逐条见 §0.6） |
| v2.4 | 修正 4 处：上置信乘子改 `c_{1−α}(n) = (n−1)/χ²_α(n−1)`（原 `(1+κ)(1−α)` 在 α 大时反而缩小，不是上界）、Conservative 的标量/向量不一致（明确 A 逐通道 / B 单权重 + 逐通道和 两方案，权重用闭式 + 主动集而非通用 QP）、QP 成本从"可忽略"改为**待 spike 实测**、验收矩阵补齐 `类别` + `分位` 两列并与 §18.5 规则 2 自洽（逐条见 §0.7） |
| **v2.5（本文）** | 修正 4 处：`TraceSubmission` 补全 `slot_base`/`n`/`generation`/`deadline_frame` 且 `submit` 收 `deadline_frame`、`poll` 收 `generation`+`frame`，新增 `Stale` 终态（10 态）；`c < 1` 的边界改为"只采用 `α ≤ 0.1`"不再给伪精确阈值；退路 ③ 改为单权重 `w_i ∝ 1/Σ_c v_{c,i}`（A 的逐通道解只用于报告）；§18.3 增显式 `H`/`S` 两列 + 两条读表规则（`N/A ≠ pass`、组合剪枝），§18.4 声明 `N/A` 为有效声明（逐条见 §0.8） |
| **v2.6（本文）** | 补全镜面反射：`SpecularRestir` 从「间接光 9 条里的一行」拆为**独立的镜面族 4 条**（+ `SpecularScreenSpace` / `ProbeSplitSum` / `PlanarReflection`），间接光表 9→8；新增 §5.3.3 镜面信号契约（降级链 / 重投影域 / 半分辨率接缝责任 / 能量项）；§18.2 新增 S7 场景类、§18.3 新增 4 条镜面门槛；§6.4 驱逐顺序接入镜面降级链；§16 新增 R33/R34；§17 新增决策 #16/#17（逐条见 §0.10）；另做维护性自检，把 4 条镜面 recipe 收敛到 **1 个 backend**、降级链收敛到 **1 处真值源**（§0.11） |

### 0.4 补丁索引（26 条，一条不缺）

| # | 补丁 | 落点 | 来源 |
|---|---|---|---|
| 1 | `PassDecl` 瘦身（只描述图）+ `ExecPolicy`（split/join、VRS）+ transient 生命周期 | §4 / §8.2 | UE RDG + VRS |
| 2 | 降噪：每信号独立实例 + validation 策略 | §8.3 | NVIDIA NRD |
| 3 | DDGI → brick 自适应 + 页化（与流送合流） | §9.3 | Unity APV |
| 4 | 新 recipe `SparseSdfGi` | §5.3 | AMD Brixelizer GI |
| 5 | 直接光 Estimator 支持 RIS，与 GI 共用 `ReservoirCore` | §5.2 / §8.3 | RTXDI / GAR |
| 6 | DebugView 契约驱动自动注册（+ 显式 `DebugMeta` 声明，非零工作） | §11.1 | Unity Rendering Debugger |
| 7 | 页池 key 挂 world partition cell | §9.1 | UE World Partition |
| 8 | `BackendCaps`（静态）+ `BackendRuntime`（动态）+ `submit/poll/cancel` 句柄 | §4 | Godot RD / GLES3 |
| 9 | `RenderFeaturePlugin` 补 sync 阶段 | §13 | USD Hydra |
| 10 | 增量状态显式 checkpoint 序列化 | §7.2 | Arnold / Renderman |
| 11 | CI 增加 shader variant 数量门禁 | §12 | Filament / UE 变体收敛 |
| 12 | 分帧摊销调度器 | §6.5 | Decima |
| 13 | `PathTracedOffline` 升为长期质量锚 | §5.3 / §10 | Cyberpunk RT Overdrive |
| 14 | ADR 补 NPR 确定性的外部实证 | §10.6 | Genshin / GGXrd |
| 15 | `BudgetCompiler` = cost model + 统计 + plan regression；EXPLAIN | §6.1 / §6.8 | 数据库查询优化器 |
| 16 | 场景统计量作为规划输入 | §6.2 | cardinality estimation |
| 17 | `c_k` 系数构建期标定 | §6.1 | ML autotuning |
| 18 | 红绿失效传播 + pass 显式声明 `destroys` | §7.1 | rustc/salsa + LLVM |
| 19 | `ChangeJournal` 采 WAL 语义 + 快照 | §7.2 | 数据库 WAL / MVCC |
| 20 | 摊销重算改 level-based 调度 | §6.5 | EDA incremental STA |
| 21 | 预算升级为分区式（闲置可回收） | §6.3 | ARINC 653 |
| 22 | 每链 requests/limits + 驱逐顺序 + 准入控制 | §6.4 | Kubernetes QoS |
| 23 | 降档改测量闭环（BBR 式）+ 限幅阻尼 | §6.6 | TCP BBR |
| 24 | 标准 SSE 公式（用于选级）+ `bias` 独立通道；任何时刻出图 | §9.2 / §9.4 | 3D Tiles + VTK |
| 25 | 序列级不退化判定（取代逐帧单调断言） | §2.2 / §18 | Boddy & Dean |
| 26 | `QualityContract` 分层叠加 + 冲突检测 | §6.7 | USD composition |

### 0.5 v2.1 → v2.2 勘误（8 条）

> 这一节保留错误本身，而不是悄悄改掉。v2.1 中 **3 条是错误/缺陷**，**5 条是未定义或低估**。

| # | v2.1 的说法 | 性质 | v2.2 修正 |
|---|---|---|---|
| 1 | `proxy_error` / `sse` 流入 `variance`，把代理误差变成「可降档的方差」 | **错误**（概念混淆） | bias 与 variance **分列**：bias 来自表示/代理，降噪平均不掉；variance 来自采样，可被时空重建降低。降档用 `MSE ≤ variance + bias_bound²`，两项**不共享预算、不互相抵扣**（§9.4、附录 C） |
| 2 | 不变量 7「新帧不得比旧帧差」 | 未定义（不可计算） | 改为**序列级不退化**：固定测试序列的 P50/P95 判定；运行期断言仅限 `history_valid && !disoccluded` 区域，并允许有界振荡（§2.2、§18） |
| 3 | `Σw=1` 且 `w ∝ 1/var` 即能量守恒 + 无偏 | **错误**（缺统计前提） | `Σw=1` 只保证凸组合，不保证物理能量守恒；`1/var` 权重是随机量，比值估计有偏；不同 estimator 可能**积分目标不同**。改为双模式 `Unbiased`（同 target + MIS pdf）与 `Conservative`（带 bias 上界），能量守恒作为**独立 flux 验收项**（§5.4） |
| 4 | `RayBackend` 只有能力表 + 同步 `trace` | 不足 | 拆 `BackendCaps`（静态）+ `BackendRuntime`（在飞/队列/延迟/错误/驻留）+ `submit/poll/cancel` 句柄，支持异步、部分结果、取消、失败（§4） |
| 5 | `PassDecl` 同时承载资源依赖、缓存破坏、异步调度、VRS、预算 | **设计缺陷**（万能契约） | 拆为 `PassDecl`（只描述图）/ `CacheEffect`（缓存效果）/ `ExecPolicy`（异步、VRS、预算），由编译器关联校验（§4、§12） |
| 6 | 「新增 recipe 零 UI 工作」 | 偏乐观 | 结构与数值可自动注册；但每 recipe 需声明 `DebugMeta`（通道语义/色阶/范围/时间窗），缺省降级为原始值直显。修正为「零 UI 代码 + 每 recipe 约 10 行显示元数据」（§11.1） |
| 7 | 串行 35.5 周 / 并行后 ~30 周 | 不可靠 | 覆盖 trace、虚拟流送、DDGI 页化、降噪、增量、离线路径、host 迁移、crate 归档，**没有团队规模、同类系统历史速度、代码量校准**。改为「3 个前置 spike + P0–P8 阶段」，**不承诺单点工期**（§15） |
| 8 | 「256 灯斜率 ≈0」「hitch = 0」「thrash = 0」 | 不可度量 | 改为 **硬件类 × 场景类 × P50/P95/P99** 的验收矩阵；禁用绝对零门槛，hitch/thrash 改为统计上限（§18） |

**这 8 条对方案本身的影响**：架构骨架（一执行路径 + 三链 + 七契约 + recipe 白名单）不变；变的是**误差如何记账**、**契约如何切分**、**什么算通过**。第 7、8 条性质最重——一个没有可信工期与可执行门槛的方案，不能被当作工程计划审查。

### 0.6 v2.2 → v2.3 勘误（6 条）

| # | v2.2 的说法 | 性质 | v2.3 修正 |
|---|---|---|---|
| 1 | `MSE ≈ variance + bias_bound²` | **符号错**：把上界写成等式 | 真值关系是 `MSE = Var + bias²`；`bias_bound` 是 `abs(bias)` 的上界，故只能写 **`MSE ≤ Var + bias_bound²`**。另列 5 条成立条件（同 target、bound 真为上界、方差来自独立样本、只在降噪前定义、bound 需报紧度）（§9.4、附录 C） |
| 2 | 「降噪降低 variance」可作验收 | 不充分（会掩盖重影与系统性错误） | variance 分三级：`σ²_raw`（单帧原始采样）/ `σ²_acc`（时域累积后按**有效样本数**算）/ `σ²_post`（降噪后只能对照参考测）。降噪是 variance→bias 的交换，验收改看 `MSE_post` / FLIP / ghost / blur（§9.5） |
| 3 | Conservative 模式「由 bias_bound 定权」 | **不可计算**（粒度未定义） | 粒度定为**逐像素 × 逐通道 × 逐贡献**，运行时用可分解形式 `b_i = g_i(p)·β_i(material)`；权重由小规模 QP 求解（§5.4.1） |
| 4 | `submit/poll/cancel` 已有，映射与失败语义未定 | 缺规格 | 输出按**调用方 query 槽位**索引（不是完成顺序），每槽带 `generation`；给出完整状态转移表，含设备丢失 / 超时 / 部分结果过期（§4.1） |
| 5 | S1 接 `DirectStochasticRT`，但决策 #4「先 SW BVH / 同时 RayQuery」未定 | **自相矛盾** | S1 只验**契约形状**：允许 `MockBackend`（确定性合成命中）+ **SW BVH 真实后端**，不要求 RayQuery。`DirectStochasticRT` 在 S1 中实例化于 SW BVH——后端是参数，这正是 `TraceQuery` 的意义。决策 #4 不阻塞 S1（§15.1） |
| 6 | P7「host 无 `bevy_render`、141 → ~101」 | **范围蔓延**（会把 GI 重构扩成全仓迁移） | 拆为 **P7a GI 垂直切片**（唯一范围内的迁移：只要求 GI 路径上的 crate 无 `bevy_*`）与 **P7b 全仓迁移（明确范围外，可能永不做）**；"141 → ~101" 降级为 P0 的记账动作，不再作 P7 验收（§15.2） |

**这 6 条不改骨架，改的是"看起来严谨、实际不可算"的三处**：误差是不等式不是等式、variance 有测量阶段、权重有粒度。

### 0.7 v2.3 → v2.4 勘误（4 条）

| # | v2.3 的说法 | 性质 | v2.4 修正 |
|---|---|---|---|
| 1 | 用估计方差时 `MSE ≤ Var_est·(1+κ)(1−α) + bias_bound²` | **符号/单调性错**：`(1−α)` 随 α 增大而变小，α 大时该式可**小于** `Var_est`，不可能是上界；κ 也无定义 | 改为 `MSE ≤ c_{1−α}(n_eff)·Var_est + bias_bound²`，`c = (n−1)/χ²_α(n−1) ≥ 1`；给出 χ² 依据、Wilson–Hilferty 免查表实现、`α ≤ 0.1` 约束、参考值表，并点名 **firefly 会让 s² 系统性低估**（§9.4、附录 C） |
| 2 | `bias_bound: Vec3`，但 Conservative QP 用标量 `bᵀw` | **维度不一致**（界算不出来） | 明确两方案并用：**A 逐通道独立**（用于报告的 `Vec3` 界）、**B 单权重 + 逐通道和**（用于实际混合，因为一个像素只能有一套 `w`）；合并成 `M = diag(Σ_c v_c) + Σ_c b_c b_cᵀ`；通道相关性**显式声明不建模**（§5.4.1） |
| 3 | 「逐像素 QP 在 ALU 上可忽略」 | **未验证的性能假设** | 改为闭式 `w* = M⁻¹1/(1ᵀM⁻¹1)` + 主动集截断（n≤4，≤2 次重归一），给纸面成本 100–200 ALU/px 但标为**待 spike 实测**（目标 ≤0.15ms@1080p）；**禁止**在主 composite 路径放通用 QP 迭代求解器；给三条退路（查表 / 固定 2 次迭代 / 退 `w∝1/v`）（§5.4.1） |
| 4 | §18.3 只写阈值不写分位，而 §18.5 规则 2 说"缺分位视为未定义" | **文档自相矛盾** | §18.3 增 `类别`（分布型/布尔型/确定性标量）与 `分位` 两列，13 行逐行补齐；§18.5 重写为 6 条，其中规则 3 规定"类别决定分位要求"，布尔型 100% 降为其特例，消除旧规则 2 与 3 的冲突（§18.3、§18.5） |

**这 4 条的性质与前两轮不同**：§0.5、§0.6 修的是**概念**（bias≠variance、不等式方向），§0.7 修的是**形式**——公式写反、维度对不上、假设未验、两节互相打架。骨架与 recipe 白名单不变；变的是"公式能不能算、算出来是不是界、门槛能不能被机器判定"。

仍不解决的三处（已知，不假装解决）：

1. `g_i(p)` / `β_i(material)` 的**具体系数**只有形式没有标定——需要 S3 用实测偏差回归出来，否则 `bias_bound` 仍是符号。
2. `c_{1−α}(n)` 依赖 `n_eff`，而 `n_eff` 本身是近似（重投影样本不独立）——理论界会**偏乐观**，所以验收必须同时报"理论界 / 实测界"的松紧比（§18.3 混合偏置紧度行）。
3. 上界解**系统性偏保守**（`b_i` 是上界 ⇒ 权重重分配到低 bias 项 ⇒ 可能过度降权），紧度依赖 S3 标定质量。

### 0.8 v2.4 → v2.5 勘误（4 条）

| # | v2.4 的说法 | 性质 | v2.5 修正 |
|---|---|---|---|
| 1 | `TraceSubmission { token, submitted }`，`submit` 无 `deadline_frame`，`poll` 无 `generation`/`frame` | **契约未闭合**（状态机实际不可实现） | 补 `slot_base`/`n`/`generation`/`deadline_frame`；`submit(..., slot_base, generation, deadline_frame)`；`poll(token, generation, frame, out)`；新增 `Stale` 终态并写明它是迟到结果的**唯一显式出口**；状态机由 9 态补到 **10 态**，与 `PollState` 的 7 个返回值一一对应（§4、§4.1） |
| 2 | 「`α > 0.5` 时 `c < 1`，不再是上界」 | **伪精确断言**：`c < 1` 的精确条件是 `χ²_α(n−1) > n−1`，阈值随自由度在约 (0.5, 0.61] 间变化 | 删掉通用边界论断，改为**只采用 `α ≤ 0.1`，其他范围不采用**；同时保留精确条件作注（§9.4、附录 C） |
| 3 | 退路 ③「降为 A 的逐通道 `w ∝ 1/v`」 | **与单权重混合矛盾**（一个像素只能有一套 `w`） | 改为 **B 在 `b = 0` 时的单权重解** `w_i ∝ 1/Σ_c v_{c,i}`；明确 **A 的逐通道解只用于报告的 `Vec3` 界，不能用于混合**（§5.4.1、R32、决策 #15） |
| 4 | §18.3 的 `H`/`S` 只藏在"门槛形式"文字里 | **规则与表格不一致**：规则 2 要求四列齐全，表里只有两列 | §18.3 增显式 `H`、`S` 两列（共 7 列），13 行逐行填；新增两条读表规则——**`N/A ≠ pass`**（不适用必须报 `N/A` 并列原因，如 H1 的混合偏置行）与**组合剪枝**（`H × S` 是剪枝结果，其余组合报 `uncovered` 并进 ADR）；§18.4 声明其 `H = S = N/A` 是有效声明（§18.3、§18.4） |

**这一轮的性质又变了**：§0.5/§0.6 修概念，§0.7 修形式，§0.8 修的是**闭合性**——契约的输入是否完整（缺一个入参，状态机就有不可达转移）、断言的适用域是否精确（给一个不成立的"精确边界"比不给更糟）、退路是否与主路径同一语义、规则与表格是否描述同一件事。

### 0.10 v2.5 → v2.6 勘误（5 条）

| # | v2.5 的问题 | 性质 | v2.6 修正 |
|---|---|---|---|
| 1 | 「反射」全文只有 7 个孤立词元（`TraceKind::Specular` 一个变体、`SpecularRestir` 一行 recipe、§8.3 一行降噪参数、§5.4.1 一个"是否近似纯镜面"系数、§11.2 一个示例、§0.9 一句"尚未接入"、附录 A 两处资产名），没有任何一节承担它 | **覆盖假象**（写了，但不构成可验收单元） | 镜面从"标签"升为**独立族**：§5.3 拆出 4 条 recipe + §5.3.3 信号契约；漫反射表 9→8 行（`SpecularRestir` 迁出），总数 13→16 |
| 2 | 白名单里没有任何一条负责镜面的 **fallback**，而 `SpecularRestir` 在 H1（无 RT）上整条不可用 ⇒ **镜面在 H1 上没有 recipe** | **可执行性缺口**（比 §18.3 已承认的"H1 混合偏置无定义"更严重） | 新增 `ProbeSplitSum`（split-sum 环境探针）作为 fallback 终点并进白名单；§5.3.3 规定降级链 `RT → SSR → probe → ambient` 单调且可解释 |
| 3 | 镜面的重投影域（沿反射向量的虚拟相机）与漫反射（surface 重投影）不同，§8.3 只写"重连移位校验"未定义域；半分辨率上采样接缝也无契约 | **契约缺失**（history valid 判据无从实现） | §5.3.3 显式定义重投影域、半分辨率接缝责任与能量项；§8.3 的 Specular 行改为引用该契约，不再自带参数 |
| 4 | 镜面能量是 Fresnel + 多次散射项，与 §18.3 的"全白单灯通量"不是同一个量；§18.3 的 S6 专项（半透明 / 毛发 / 水）不含镜面 ⇒ **镜面能量守恒无人验收** | **验收缺口** | §18.2 新增 **S7 镜面反射**（粗糙度阶梯 × 平面镜 × 水面 × 无 RT 档 fallback）；§18.3 新增镜面能量守恒、降级单调性、SSR 接缝、画质覆盖 4 条 |
| 5 | 反射资产（`post_ssr_trace`、`post_spec_gi_reservoir_prior`、`spec_gi_*.wesl`）在 §3.4 的「净删 / 转正」两个名单里**都不在**；§17 的 15 项决策里 **0 项**涉及反射 | **资产悬空 + 决策盲区**（白名单一旦冻结就再拆不开） | §3.4 新增镜面转正行；§16 新增 R33（无 RT 档镜面塌陷）与 R34（半分辨率接缝被当降噪误差）；§17 新增 #16（specular 首期是否含 SSR）与 #17（`PlanarReflection` 是否进首期） |

**这一轮的性质是「补层」而不是「纠错」**：§0.5–§0.8 修的都是已经写下的内容，v2.6 修的是**从来没被写下的那一层**——一个拥有独立 fallback 链、独立重投影域、独立能量项的信号，此前只占一行表格。判据也很朴素：**一条技术若不能在 §5.3、§8.3、§18.3、§17 四处同时找到自己的行，它就没有被这份文档覆盖。**

### 0.11 v2.6 自身的维护性自检（3 条）

补一层新信号，最容易在「易维护 / 易拓展」上倒退。自查后发现 v2.6 初稿确实引入了 3 个回归，已当场拆掉，对应约束写进 §12 / §13 / §18.4。

| # | v2.6 初稿的回归 | 为什么会发生 | 约束（已写入） |
|---|---|---|---|
| 1 | 4 条镜面 recipe 各自需要一个 `RecipeBackend` 实现 | 把「4 条不同的降级级」当成「4 种不同的算法」——它们在算子代数里只是同一个 Rep 的 `(target, sampler, estimator)` 取了不同值 | **只允许一个 `SpecularRep`**：前三条共用，`PlanarReflection` 除外（决策 #17 已排除首期）。新增第 5 条镜面 recipe = 白名单 1 行 + 1 个参数组合，**0 个新 backend、0 处 if/else**（§5.3.2 末段、§13 新行、§18.4 新行） |
| 2 | 降级链 `RT → SSR → probe → ambient` 若同时写死在 `DEFAULT_FALLBACKS`（补丁 1）、EXPLAIN 文案、§8.3 表、§18.3 门槛里，就是 4 处真相源 | 链是「跨契约的秩序」，天然倾向于在每个引用点重述一遍 | **链的顺序只在 `lighting::constants::SPECULAR_FALLBACK` 定义一次**，其余四处只引用不复制；新增一条链必须同时登记 §18.3（否则该链没有 `H`/`S` 验收，等于没验收）。§12 新增「降级链登记」行 |
| 3 | `PlanarReflection` 会新增一个独立镜像渲染 pass | 它是镜面族里唯一**不共享既有 kernel** 的：新 RT + 新带宽项 + 新 variant | 不进首期（决策 #17）；将来单独立项时必须同时过 §12「shader variant 只许降」与 §8「预算新增」两道门，而不是当作一条 recipe 悄悄接进来 |

**自检判据**：新增一条 recipe，若「白名单行 + 参数取值」之外还要改第三处代码，它就不满足 §18.4「新 GI 技术改动文件数 ≤3」；若它还需要一个新的 `RecipeBackend` 实现，就不满足 §13「0 处 if/else」。v2.6 两条都成立——把 4 条 recipe 压成 1 个 backend 的理由不是"代码少写"，而是**多写的那三份会各自演化**。

---

## 1. 审计基线与问题诊断

### 1.1 现状量化（只读审计）

| 事实 | 数值 / 位置 |
|---|---|
| `pkg/` crate 数 | 141 个目录（140 个 workspace 成员） |
| 无任何仓库内消费者的 crate | 39 个（渲染侧 10 个） |
| CPU golden 层 | `pkg/prism_render_shading/src/gi/` 62 个子模块、2,278 个 `#[test]` |
| bevy 轨完整实现 | `pkg/prism_render_scene`：162,611 行 `.rs`、1,874 个 `#[test]`、178 个 WESL，**0 个仓库内依赖者** |
| 新宿主 | `pkg/prism_render_app`：不依赖任何 `bevy_*`，但 `src/frame.rs` 只做 raster pass，**无光照/GI pass** |
| 默认关闭的开关 | `pkg/prism_render_scene/src` 中 **24 处 `enabled: false`** |
| 可复用的"死资产" | `virtual_geometry_gpu` 7,822 行/0 测试、`virtual_texture_gpu` 2,311/18、`dmm_gpu` 738/0、`micromap_gpu` 667/0、`ray_reorder_gpu` 691/2、`radiance_cascades_gpu`（0 消费者），合计 **12,229 行 / 20 测试** |
| 移植切片 | `prism_render_scene_rhi_port_zh.md`：S1 渲染世界 → S2 资源 → S3 视图 → S4 几何 → S5 光照/阴影 → S6 着色/后处理 → S7 water/cloth/raytrace → S8 收尾 |

### 1.2 六病症

| 病症 | 证据 | 后果 |
|---|---|---|
| **一个算法最多 5 份实现** | golden（`prism_render_shading`）→ 手写 WESL → GPU 孪生（仅 `gi/surface_cache/gpu`）→ parity harness | 改一处要动五处，靠逐行对拍粘住 |
| **pass 注册两套** | 声明式 `shading/graph.rs:80/133/185` + 各子系统 `dispatch.rs` 命令式 | 顺序错误只能靠人记 |
| **布尔组合爆炸** | 24 处 `enabled: false` + CPU 组合快照 | 2^24 不可穷举、不可测 |
| **无共用追踪/缓存抽象** | 6 个 GI 子系统各发各的 ray、各存各的 cache | 预算不可控、复用不可审计 |
| **迁移是水平重写** | 162,611 行 `prism_render_scene` 无消费者，按 S1–S8 逐层搬 | 越写越沉，永远"快完成" |
| **系统不可见** | 无 GI / 探针 / 页 / 预算的可视化 | **不可维护——这是根本原因** |

> 关键判断：前五条是**现象**，第六条是**病因**。一个你看不见内部状态的 GI 系统不是"难调"，而是"不可维护"。

---

## 2. 设计原则

### 2.1 三条公理

1. **所有光传输 = 采样 + 重建。** 技术差异只在 `(Rep, Sampler, Estimator, Reuse, Reconstruct)` 五元组。UE5 Lumen 随机化了间接光、MegaLights 随机化了直接光，两者共同验证这条公理。
2. **成本常数化靠"压缩脏数据"，不是靠少做 pass。** 空转的 lane、未命中的 ray、无效的页，比 pass 数量更耗性能。
3. **不可见的系统不可维护。** 因此调试可视化是**维护前提**，不是加分项——这条直接推出 §11 是必要设计，而非可选功能。

### 2.2 七条不变量（全部进 CI）

1. **单一实现源**：GPU/WESL 是唯一实现，CPU 只做 oracle。
2. **单执行路径**：材质 `respond()` 唯一；质量档不进管线键。
3. **唯一射线预算出口**：`RayScheduler`；子系统禁止自行发射线。
4. **禁新增布尔门控**：新能力必须是 recipe 或 capability。
5. **每个 recipe 必有 deterministic twin**（NPR / 回放 / 对拍 / 无 GPU 可用）。
6. **增量的唯一实现源**：`ChangeJournal`（字段级）+ `FieldEpoch`。
7. **任何时刻出图 + 序列级不退化**：页缺失 / 未收敛时必须有可用输出；质量门槛按**固定测试序列**的 P50/P95 判定（§18），**不做逐帧单调断言**——时空降噪、重投影、镜头运动、disocclusion 下逐帧单调性通常不成立（v2.1 的「新帧不得比旧帧差」不可计算，见 §0.5 第 2 条）。运行期只允许在 `history_valid && !disoccluded` 区域内做有滞回的单帧断言。

---

## 3. 架构

### 3.1 六层（横切）

```
层 6  Platform & Quality   capability profile / QualityContract / 预算编译器 / 降档
层 5  Shading & Graph      respond() 单路径 / PassDecl / oracle hooks / DebugView
层 4  Lighting             recipe 白名单 / 算子 / BlendOperator / 阴影槽位
层 3  Virtualization       页池 / 页表 / feedback / 预算仲裁 / 保真度 / cell 绑定
层 2  GPU Scene            稳定句柄 / LightTable / BVH / 字段 epoch
层 1  Scene Contract       ChangeJournal / epoch reclamation / 快照
```

### 3.2 三条纵切链（唯一入口 / 唯一出口）

```
预算链   LightingPlan ──→ BudgetCompiler ──→ PassDecl.budget ──→ FrameResourceBudget ──→ 指标
         (唯一入口)                                                  (唯一出口)

增量链   ChangeJournal ──→ FieldEpoch ──→ 红绿失效传播 ──→ 局部重算 ──→ 等价性断言
         (唯一入口)                                        (唯一出口)

追踪链   TraceQuery ──→ RayScheduler ──→ RayBackend×3 ──→ RayHit{bias,variance} ──→ 误差预算(MSE) ──→ 降档
         (唯一入口)                                                          (唯一出口)
```

**每条链只有一个入口、一个出口，中间禁止旁路。** 这是"易维护"的结构性保证——不靠文档约束，靠类型系统。当前病灶恰好是每条链都有 N 个入口（各子系统自己 extract、自己判断脏、自己发 ray）。

### 3.3 crate 拓扑

| crate | 职责 | 状态 | 估计规模 |
|---|---|---|---|
| `prism_render_abi` | 光照 / 材质 / GI / 变更的纯数据 ABI（**无 GPU 依赖**） | 新 | ~2.5K |
| `prism_render_driver` | RHI | 已有 | — |
| `prism_render_graph` | PassDecl + 编译器（barrier / aliasing / 合并 / async split） | 已有，扩 | — |
| `prism_render_trace` | TraceQuery + RayScheduler + 3 后端 + 执行模型 | 新 | ~8K |
| `prism_render_virtual` | 页池 + 页表 + feedback + 预算 + `RepProvider` + cell | 新（吃 5 个死 crate） | ~12K |
| `prism_render_cache` | Reservoir + SampleCache + 统一时空重建 | 新 | ~5K |
| `prism_render_gi` | 算子 + 16 recipe（含 4 镜面）+ 镜面信号契约 + BlendOperator + 阴影槽位 | 新 | ~18K |
| `prism_render_debug` | DebugView（自动注册）+ EXPLAIN + 报告 | **新** | ~4K |
| `prism_render_shading` | 响应函数 + 后处理（GI 迁出，−40%） | 已有，收缩 | — |
| `prism_render_host` | host + shim（现 `prism_render_app` 演进） | 已有，演进 | — |
| `prism_render_architecture` | 冻结为 oracle / 参考 | 冻结 | 不增长 |
| `prism_render_scene` | 移植后归档 | 冻结 | 不增长 |

> `prism_render_debug` 必须是**第一类 crate**：若调试是"有空再加"，公理 3 落不了地。

### 3.4 净删 / 降格

| 动作 | 对象 | 收益 |
|---|---|---|
| 删 | PSSM 级联管理 + PCSS/PCF 权重（`shaders/shadow.wesl`） | 三份 filter → 一个 `FilterKind` 枚举 |
| 删 | 点光 cube 距离图（golden 1,697 行） | −1 套阴影实现 |
| 删 | 逐灯 shadow map 分配与缓存复用 | O(灯数) → O(采样数) |
| 删 | `light_routing` 通道门控（Rust 299 行 + WESL 2,979 行 ABI） | 门控 → 采样分布 |
| 删 | 24 处 `enabled: false` + CPU 组合快照 | 组合爆炸消失 |
| **降格** | VSM 页表 | 从"第三条阴影主线"→ `DirectStochasticVSM` 的实现细节（**不删**） |
| 归档 | 39 个零引用 crate（渲染侧 10 个） | 141 → ~101 |
| **转正** | `radiance_cascades_gpu` / `ray_reorder_gpu` / `virtual_geometry_gpu` 等 | 12,229 行死资产变底座 |
| **转正** | 镜面资产：`post_ssr_trace`（`post_graph.rs:215`）/ `post_spec_gi_reservoir_prior`（`:263`）/ `spec_gi_*.wesl` | 4 条 specular recipe 的底座；此前在「净删」与「转正」两个名单里都不在（§0.10 第 5 条） |

---

## 4. 七个契约

> 硬约束：总量 **≤1000 行**（v2.1 的 900 行在拆出 `CacheEffect` / `ExecPolicy`、补全 backend 生命周期后不再现实；上限上调一次，此后不再上调）。超了就是抽象失败——这条判断标准比任何设计描述都有效。

```rust
// ① ABI（prism_render_abi）
pub struct FieldEpoch {
    pub geometry: u32,
    pub material: u32,
    pub lighting: u32,
    pub env: u32,
    pub layout: u32,      // cell 卸载 / 拓扑变更
}
pub struct ChangeJournal { /* 字段级变更；WAL 语义：先记账后改状态 */ }
pub struct JournalSnapshot { /* 显式序列化，可回放 / 可复现 */ }

// ② PassDecl（prism_render_graph）——只描述图本身
pub struct PassDecl {
    pub id: PassId,
    pub reads: Vec<ResRef>,
    pub writes: Vec<ResRef>,
    pub queue: QueueKind,
    pub transient: Option<Lifetime>,      // transient 资源生命周期区间
    pub aliasing: AliasingPolicy,
}

// ②b CacheEffect（prism_render_cache，独立注册表）
pub struct CacheEffect {
    pub pass: PassId,
    pub destroys: Vec<CacheTag>,          // 本 pass 破坏哪些缓存（显式，禁止"人记得"）
    pub preserves: Vec<CacheTag>,
}

// ②c ExecPolicy（prism_render_runtime，执行与预算关注点）
pub struct ExecPolicy {
    pub pass: PassId,
    pub async_split: AsyncPolicy,         // split / join
    pub persistent: bool,                 // 常驻线程（见 §8.2）
    pub shading_rate: Option<TileMask>,   // VRS：按 tile 的着色密度
    pub budget: Option<BudgetHint>,
}

// ③ Trace（prism_render_trace）
pub enum TraceKind { Shadow, Diffuse, Specular, Volumetric, Occlusion }

// 静态能力：可查询，不随时间变化
pub struct BackendCaps {
    pub kinds: BitSet,
    pub max_t_max: f32,
    pub alpha: bool,
    pub is_hw: bool,
    pub exec: ExecModel,          // Sync | Async{depth} | Persistent{threads}
    pub latency: LatencyClass,    // SameFrame | Plus1Frame | PlusNFrame(n)
    pub max_batch: u32,
}

// 动态状态：每帧可查，降档与准入的输入
pub struct BackendRuntime {
    pub inflight: u32,
    pub queue_depth: u32,
    pub last_completion_frame: u64,
    pub error: BackendError,      // Ok | DeviceLost | OutOfMemory | KindUnavailable(kind)
    pub residency: ResidencyNeed, // 需要的页池 / 显存类别
}

// 提交即返回句柄：支持异步、部分结果、取消、失败
// 槽位区间 [slot_base, slot_base+n) 由调用方声明；结果写到 slot_base+i，
// **不按完成顺序**（禁止 compaction，见 §4.1）
pub struct TraceSubmission {
    pub token: SubmissionToken,     // 带 backend epoch；设备重建后旧 token 永久失效
    pub slot_base: u32,             // 本批首个槽位（= 调用方 query 数组下标）
    pub n: u32,                     // 本批槽位数（= 本次 query 数）
    pub generation: u64,            // 槽位世代；复用同一 slot 区间时 +1
    pub submitted_frame: u64,
    pub deadline_frame: u64,        // 超过即 Expired，不再等待
}

pub enum PollState {
    Pending,
    Partial { ready: SlotMask },    // 哪些槽就绪；未就绪槽必须是 None
    Ready,
    Cancelled { acked: bool },      // ack 前缓冲不可复用（§4.1 规则 1）
    Expired,                        // 已过 deadline_frame
    Failed(BackendError),           // 含 DeviceLost：本批全部作废
    Stale,                          // generation / token epoch 失配：结果一律丢弃
}

pub trait RayBackend {
    fn capabilities(&self) -> BackendCaps;
    fn runtime(&self) -> BackendRuntime;
    /// deadline_frame 必传：由 latency_class 推出（§4.1 规则 3），后端不得自行放宽
    fn submit(&mut self, q: &[TraceQuery], slot_base: u32, generation: u64,
              deadline_frame: u64) -> Result<TraceSubmission, BackendError>;
    /// 必须传当前 generation 与当前 frame：二者用于判定 Stale / Expired
    fn poll(&mut self, token: SubmissionToken, generation: u64, frame: u64,
            out: &mut [Option<RayHit>]) -> PollState;
    fn cancel(&mut self, token: SubmissionToken, generation: u64) -> CancelAck;
}

// 契约闭合性（v2.4 之前缺了 4 个必要输入，状态机实际不可实现）：
// 1. `slot_base`/`n` 必须记在 submission 里：poll 不再传 base，避免两次调用给不同 base 而错位。
// 2. `submit` 必须带 `deadline_frame`：超时判定在提交时冻结，否则后端可无限延迟。
// 3. `poll` 必须带 `generation`：这是"部分结果过期"的唯一判定输入（槽位复用后旧结果必须作废）。
// 4. `poll` 必须带 `frame`：这是 `Expired` 的唯一判定输入（不能靠后端"自觉"上报超时）。
// 5. `Stale` 是迟到结果的**显式出口**；缺它就只能靠 `out[i] == None` 猜，不可测、不可断言。

pub struct RayHit {
    pub radiance: Vec3,
    pub depth: f32,
    pub material: MaterialId,
    pub bias_bound: f32,     // 表示/代理带来的系统误差上界（**不可**被降噪平均掉）
    pub variance: f32,       // 采样带来的统计方差（可被时空重建降低）
}

// ④ RepProvider（prism_render_virtual）
pub struct RepSpec { pub kind: RepKind, pub max_sse: f32 }

pub struct RepReadiness {
    pub fidelity: Fidelity,
    pub bias_bound: f32,            // 该表示的几何/辐射误差**上界**（bias，不是 variance）
    pub sse: f32,                   // bias 的空间度量，用于选级
    pub missing_pages: u32,
    pub fallback_available: bool,   // 任何时刻出图
}
pub trait RepProvider {
    fn readiness(&self, spec: &RepSpec) -> RepReadiness;
    fn fallback(&self, spec: &RepSpec) -> Option<Rep>;
}

// ⑤ ShadowSlot（prism_render_gi）
pub enum ShadowMethod { RayTrace, VirtualShadowMap, CascadeMap }

pub struct ShadowCostModel {
    pub gpu_ms: f32,
    pub bandwidth: u64,
    pub resident_bytes: u64,
    pub io_ops: u32,
}
pub struct ShadowSlot {
    pub method: ShadowMethod,
    pub rep: RepSpec,
    pub filter: FilterKind,
    pub cost: ShadowCostModel,
}
pub fn sample_visibility(slot: &ShadowSlot, q: &TraceQuery) -> Visibility;

// ⑥ Cache + Blend（prism_render_cache）
pub trait SampleCache {
    fn fetch(&self, k: CacheKey) -> Option<CacheSlot>;
    fn commit(&mut self, k: CacheKey, v: CacheSlot);
    fn invalidate(&mut self, e: InvalidationEvent);
    fn evict(&mut self, b: Budget);
    fn quality(&self) -> f32;        // anytime：当前解的置信度
}

// 混合的输入必须是三元组，而不是裸 radiance
pub struct Contrib {
    pub value: Vec3,
    pub variance: f32,               // 统计方差（可被时空重建降低）
    pub bias_bound: f32,             // 系统误差上界（不可被平均，见 §9.4）
    pub target: TargetId,            // 积分目标：bounce 数 / BRDF / 域；混合前必须一致
    pub weight_semantics: WeightKind,// MIS{pdf} | Ratio{exo_var} | AdHoc
}
pub struct BlendOut { pub value: Vec3, pub bias_bound: f32, pub variance: f32 }

// 前提检查内建：不许把不同 target 或带 bias 的贡献静默平均（见 §5.4）
pub fn blend(cs: &[Contrib], policy: BlendPolicy) -> BlendOut;

// ⑦ Plan（prism_render_runtime）
pub struct SceneStats {
    pub lights: u32,
    pub punctual_max: u32,
    pub npr_materials: u32,
    pub proxy_page_ratio: f32,
    pub tri_density: f32,
}
pub struct FrameResourceBudget {
    pub gpu_ms: f32,
    pub bandwidth: u64,
    pub resident_bytes: u64,
    pub io_ops: u32,
    pub shading_rate_budget: f32,
}
pub struct BudgetPartition { /* 每条链的硬时间/带宽/IO 窗，闲置可回收 */ }

pub struct LightingPlan {
    pub tier: Tier,
    pub recipes: Vec<RecipeId>,
    pub partition: BudgetPartition,
    pub budget: FrameResourceBudget,
    pub rejected: Vec<Rejection>,   // EXPLAIN 数据
}
pub fn explain(plan: &LightingPlan) -> String;
```

**误差必须双通道传递**：`RayHit.bias_bound` / `RepReadiness.bias_bound` 进入 `Contrib.bias_bound`，与 `variance` **分列**。把 bias 混进 variance 是错的——降噪会平均掉 variance，但**平均不掉 bias**（v2.1 的原始错误，见 §0.5 与附录 C）。

**PassDecl 不得膨胀**：缓存效果、执行策略、预算是**三个独立注册表**（②b / ②c），由编译器关联校验。任一结构体同时承载"图依赖 + 缓存效果 + 执行预算"，就会重新退化成万能契约（v2.1 的缺陷 5）。

### 4.1 Trace 的提交、映射与状态机（v2.3 新增）

**稳定映射**（不定这条，异步路径会静默错位）：

- 调用方持有的数组下标 = **query 槽位** `QuerySlot(u32)`，**不是**完成顺序。
- `submit(&[TraceQuery], slot_base, generation, deadline_frame)` 的槽位区间 `[slot_base, slot_base+n)` 由调用方分配；返回的 `TraceSubmission` 完整记录 `token / slot_base / n / generation / submitted_frame / deadline_frame`（见 §4 的 `TraceSubmission` 定义）。
- `poll(token, generation, frame, out)` 语义：`out[i]` 只对应 `slot_base + i`；**未就绪位置必须是 `None`，不得压缩前移**（禁止 compaction——一旦压缩，错位无法被检测）。
- `generation` 与 `frame` 是 `poll` 的**必需入参**，不是后端"应该自己知道"的状态：`generation` 判 `Stale`，`frame` 判 `Expired`；两者缺一，状态机就有不可达或歧义转移（§0.8 第 1 条）。
- 每槽带 `generation`：调用方对同一槽位重新提交时 `generation += 1`。迟到的旧结果因 `generation` 失配而丢弃，这是"部分结果过期"的唯一防线。

**状态机**（每个 submission 一个，不是每个 query 一个）：

| 状态 | 进入条件 | 允许迁移 | `poll` 返回 |
|---|---|---|---|
| `Created` | submit 之前 | → Submitted / Cancelled | 不可 poll |
| `Submitted` | `submit()` 返回 | → Pending / Ready（同步后端）/ Failed | Pending |
| `Pending` | 已入队、无结果 | → Partial / Ready / Cancelled / Failed / Expired | Pending |
| `Partial` | 部分槽位已写 | → Partial / Ready / Cancelled / Failed / Expired | Partial（合格槽位可见） |
| `Ready` | 全部槽位已写 | → Consumed | Ready |
| `Consumed` | 调用方取走 | （终态） | — |
| `Cancelled` | `cancel()` 且已 ack | （终态） | Cancelled（结果全部作废，含已写槽位） |
| `Failed` | 后端错误 | （终态） | Failed(`BackendError`) |
| `Expired` | 超过 `deadline_frame` | （终态） | Expired（仅可取 age ≤ `max_age` 且 generation 匹配的槽位） |
| `Stale` | `generation` 失配，或 token 的 backend epoch 失效 | （终态） | Stale（结果全部丢弃；**不降档、不重试**，由调用方下一帧重新提交） |

> 上表共 **10 态**，与 `PollState` 的 7 个返回值一一对应（`Pending`/`Partial`/`Ready`/`Cancelled`/`Expired`/`Failed`/`Stale`）；`Created`/`Consumed` 是调用方侧状态，后端不可见。

**失败与恢复规则**：

1. **取消**：`cancel()` 后后端不得再写该 submission 的目标缓冲，需等 ack（或本帧结束）才能复用缓冲；ack 前 `poll` 返回 `Pending`，ack 后返回 `Cancelled`。
2. **设备丢失**：所有 inflight → `Failed(DeviceLost)`；调用方走"降档不崩"路径（回退 raster 或用上一帧结果）。设备重建后**旧 token 永久失效**（token 带 epoch），禁止复用。
3. **超时**：`deadline_frame = submit_frame + latency_class 允许帧数 + 1`；超时转 `Expired` 且不再等待，该帧按"缺结果"降档。
4. **部分结果过期**：`Partial` / `Expired` 下只有满足 `slot.generation == 当前 generation` **且** `frame − written_frame ≤ max_age(latency_class)` 的槽位可用；其余按未命中处理，下一帧重新提交。
5. **同槽位重复提交**：新 submission 生效，旧 submission 若仍 inflight 则被隐式取消——两边都写是不允许的。

**契约一致性测试（不依赖硬件）**：`MockBackend` 与 `SwBvhBackend` 必须通过同一组行为测试（槽位映射、乱序完成、取消、超时、设备丢失、generation 失配）。这是 S1 的组成部分。

---

## 5. 算子代数与 recipe 库

### 5.1 五元组与编译期剪枝

```rust
pub trait Rep         { fn accept(&self, q: &Query) -> bool; fn cost(&self) -> CostCoeff; }
pub trait Sampler     { fn samples(&self, budget: u32) -> SampleIter; }
pub trait Estimator   { fn estimate(&self, s: &Sample) -> Contrib;
                        fn variance(&self) -> f32; fn cost(&self) -> CostCoeff; }
pub trait Reuse       { fn merge(&mut self, c: Contrib); fn age(&self) -> u32; }
pub trait Reconstruct { fn apply(&self, raw: &[Contrib]) -> Vec3; }
```

组合不是 2^N，而是**三条剪枝规则**：

| 规则 | 条件 | 动作 |
|---|---|---|
| `R1` 结构不兼容 | `Rep::produces ≠ Estimator::accepts` | 剪 |
| `R2` 后端不满足 | 需要 RT 但 `BackendCaps` 不支持 | 剪（走降级链） |
| `R3` 确定性冲突 | 随机采样 × (NPR / 半透明 / 毛发 / 水) | 剪（走 deterministic twin） |

> `R3` 把"UE MegaLights 不支持头发/水/半透明"这类**产品注释**变成**可执行约束**。

### 5.2 直接光（4）

| recipe | Rep | Estimator | Reuse | 状态 |
|---|---|---|---|---|
| `DirectStochasticRT` | BVH + LightTable | 固定 N ray MC / 可选 RIS | temporal + guiding | 缺失 |
| `DirectStochasticVSM` | VSM 页 + LightTable | 同上，阴影走 VSM | temporal | 半成品 |
| `DirectDeterministicCSM` | 解析 | 逐灯解析 + PSSM/PCSS | 无 | 现状主路径 |
| `SpecializedDeterministic` | 专用 | hair / water / translucency / NPR | 各自 | 已有但分散 |

### 5.3 间接光与镜面（12：8 漫反射 + 4 镜面）

> v2.5 及以前，镜面只有 `SpecularRestir` 一行，混在 9 条漫反射 recipe 里。**这两族不能共表**：它们的 fallback 终点、重投影域、降噪历史长度、能量公式都不同（§5.3.3）。v2.6 拆成两个表。

### 5.3.1 漫反射间接光（8）

| recipe | Rep | Sampler | Estimator | 状态 |
|---|---|---|---|---|
| `DdgiProbes` | **brick 自适应 + 页化** | Fibonacci 半球 | 2-moment depth + irradiance | 已有 → 改造 |
| `ScreenProbeGi` | 16px tile | octahedral snap | SH L1 | 已有（默认关） |
| `SurfaceCacheSurfel` | surfel | 相机对齐 | EMA confidence | 已有 |
| `WorldRestir` | 空间哈希 | SHARC 分箱 | RIS / GRIS | 已有 |
| `SsgiScreenSpace` | HZB | 余弦加权 | 1-spp | 已有 |
| `SparseSdfGi` | **稀疏 SDF brick** | 锥追 | 确定性积分 | **新**（Medium/Low） |
| `RadianceCascades` | 级联 | 确定性扇形 | 无历史依赖 | 死资产转正 |
| `PathTracedOffline` | 完整 BVH | 自适应 | 累积 | **长期质量锚** |

**关键结构性质**：`SparseSdfGi` 与 `RadianceCascades` 是**无历史依赖**的确定性 GI。它们同时承担三个角色：增量软失效的兜底、NPR 场景唯一可用的 GI、回放/对拍的确定性来源。这条性质**任何单一技术都不具备，只有矩阵能表达**。

### 5.3.2 镜面反射（4，v2.6 新增）

| recipe | Rep | Sampler | Estimator | 状态 |
|---|---|---|---|---|
| `SpecularRestir` | 方向 | GGX lobe | ReSTIR + 重连移位 | 已有（此前在间接光表里） |
| `SpecularScreenSpace` | G-buffer + HZB | GGX lobe | 屏幕 march（`post_ssr_trace`） | 资产已有，未接入 plan |
| `ProbeSplitSum` | prefiltered env cube（mip = roughness）+ DFG LUT | GGX 的确定性等价 | split-sum | v2.6 提案项；当前源代码未进入 `RecipeId::ALL`，未接入新宿主 |
| `PlanarReflection` | 平面镜像相机 | 确定性 | 一次镜像渲染 | 无 |

**这 4 条不是并列选项，是一条单调降级链**：`RT（SpecularRestir） → SSR（SpecularScreenSpace） → probe（ProbeSplitSum） → ambient`。每一级只在前一级不可用时被选中；选择必须可由 EXPLAIN 解释（§6.8）。**H1（无 RT）档只有后两级可用**，这是 §18.3 与 R33 的直接来源。

**共享核心（§0.11 第 1 条）**：除 `PlanarReflection` 外，前三条共用**一个** `SpecularRep` backend、同一套 `TraceQuery` / `ReservoirCore` 与同一组 `BackendCaps`，差异只在 `(target, sampler, estimator)` 三元——`SpecularRestir` 是 GGX lobe + ReSTIR，`SpecularScreenSpace` 只是把 Rep 换成 G-buffer + HZB 的确定性 march，`ProbeSplitSum` 把它换成 mip 选择 + DFG LUT。**「4 条 recipe」不等于「4 个实现」**；新增第 5 条镜面 recipe 只加白名单一行（§13、§18.4）。

### 5.3.3 镜面信号契约（v2.6 新增）

镜面与漫反射共用三条纵切链（预算 / 增量 / 追踪），但**信号契约不能共用**。以下四项必须显式约定，否则每个实现会各自解释：

| 项 | 约定 | 为什么不能沿用漫反射 |
|---|---|---|
| **降级链** | `RT → SSR → probe → ambient`，单调；降档只许沿链后退，不许跨级（无 RT 且无 SSR ⇒ 直接 probe），且每步写 EXPLAIN 原因 | 漫反射终点是 SH L2 / irradiance probe；镜面终点是 split-sum prefiltered env。两条 fallback 的 eviction / promotion 语义不同，§7.3 的三级失效只覆盖了漫反射 |
| **重投影域** | 反射向量域：以命中点（或无限远方向）为锚，用 `prev_view · cur_view⁻¹` 变换反射方向再查 history；surface 重投影**仅**用于未命中/屏外 fallback 分支 | 镜面命中点在屏外时表面没动、反射内容却变了；沿用 surface 重投影会把移动光源的反射糊成静态 |
| **分辨率与接缝** | 半分辨率 trace + 深度引导上采样；**接缝责任在 upsampler**——命中/未命中边界必须按法线-深度连续性过滤，不许交给降噪收尾 | 漫反射全分辨率、16–32 帧长历史；镜面 4–8 帧短历史，接缝先于噪声出现（R34） |
| **能量项** | Fresnel + 多次散射（`brdf.wesl` 的 `multiscatter_direct`）；验收是对高 spp 参考的**通量比**，不是"σ→0 逼近白" | 漫反射靠 albedo 能量守恒；镜面 Fresnel 随视角变化，用同一把尺会把掠射角误差当噪声 |

> 这四行是 §18.3 镜面门槛、§8.3 Specular 降噪行、§6.4 驱逐顺序的**唯一共同引用点**。任何 recipe 自带一套参数而不引用本节，即视为契约破坏。

### 5.4 BlendOperator（含混合前提）

**v2.1 的错误表述**：「`Σwᵢ = 1` 且 `wᵢ ∝ 1/varᵢ` 即能量守恒 + 无偏」。两条都不成立：

- `Σw = 1` 只保证**凸组合**（不放大不缩小），不保证物理能量守恒；
- `w ∝ 1/var` 里的 `var` 是**估计值**，是随机量，比值估计本身有偏；
- 不同 estimator 可能**积分目标不同**（bounce 数 / BRDF / 定义域不同），混合后不是"无偏估计"，而是"两个不同量混在一起"。

**两类混合前提（必须先判定）**：

| 模式 | 前提 | 权重来源 | 结果性质 |
|---|---|---|---|
| `Unbiased` | 所有贡献 `target` 相同；每项带 pdf | MIS / balance heuristic（由 pdf 算，**不用**估计方差） | 无偏（可验证：对高 spp 参考收敛） |
| `Conservative` | 至少一项 `bias_bound > 0`（缓存 / 代理 / 降噪历史） | 由 `bias_bound` 定权，且只许低 bias 者主导 | **有偏但界可知**：误差 ≤ Σwᵢ·biasᵢ |

```
scene_color = base + Σᵢ wᵢ · albedo · (contribᵢ − ambient)
前提 0：所有 contribᵢ.target 相同，否则拒绝混合（编译期）
前提 1：Unbiased 模式 → 权重由 pdf 导出，且 Σwᵢ = 1
前提 2：Conservative 模式 → 输出 bias_bound = Σwᵢ·biasᵢ，随值一起外传
前提 3：能量守恒是**独立 flux 验收项**，不由加权公式推出
兜底：全部不可用 → base + ambient       （永不黑屏）
```

**结论性约束**：
1. `target` 不同的贡献**不许**直接混合，只能先换算（降弹跳 / 降精度 / 换域）。
2. 缓存与代理贡献一律进 `Conservative` 模式，其 `bias_bound` 必须外传——否则漏光会变成静默错误。
3. GI **替换** ambient 而非相加，这个既有做法（`world_space_gi/composite.rs`、`ddgi/composite.rs`）被保留；但"替换"只保证不重复计能量，**不等于**替换量本身物理正确，后者要靠 flux 测试。
4. 混合输出仍是 `{value, bias_bound, variance}` 三元组，继续流向降档（§9.4）。

### 5.4.1 Conservative 权重：粒度与求解（v2.4 重写）

**粒度约定**（v2.2 只说"由 `bias_bound` 定权"，不可实现）：

| 维度 | 约定 | 存储 / 计算 |
|---|---|---|
| 逐像素 | 是。`bias` 依赖几何代理偏差与局部辐射梯度 | 在 composite pass 内逐像素算 |
| 逐通道 | 是。`bias_bound: Vec3`，RGB 分别有界（色度误差与亮度误差不同量级） | 侧缓冲 `rgb16f`，或由标量 × 通道系数展开 |
| 逐贡献 | 是。每个 `Contrib` 一项，`n ≤ 4` | 与混合同 pass，无额外 pass |
| 逐材质 | 只作为**系数** `β_i(material)`（albedo / 粗糙度 / 是否近似纯镜面），不是独立标量场 | 常数表 + 材质 flag |

可分解形式（避免为每贡献存一张 full-res 三通道图）：

```
b_i(p) = g_i(p) · β_i(material)      ← b_i ∈ ℝ³（逐通道）
  g_i(p)  : 几何/表示/缓存项（缺页比例、代理偏移 × 辐射梯度、cone 半角 × 梯度、缓存年龄）
  β_i(material): 材质放大系数（albedo 上限、粗糙度、透明度）
```

**标量与向量的不一致（v2.3 的缺陷）**：`bias_bound` 是 `Vec3`，但 v2.3 的 QP 写的是标量 `bᵀw`。两处必须二选一并写明：

| 方案 | 目标函数 | 性质 | 采用 |
|---|---|---|---|
| **A. 逐通道独立**（默认） | 对 `c ∈ {R,G,B}` 各一个 QP，`w` 可不同 | 每通道界**精确**；不处理通道相关 | ✅ 用于**报告的 `bias_bound`** |
| **B. 单一权重 + 逐通道和** | `min_w Σ_c [wᵀdiag(v_c)w + (b_cᵀw)²]` | 一次求解、一套 `w`；界略松但仍是上界 | ✅ 用于**实际混合** |
| C. 标量最坏相关 | `b_i := ‖b_i‖` 且假设全相关 1 | 是 B 的上界（更松），但可退化成单标量 QP | 仅作退化参考 |

**同时实现 A 与 B 的原因**：混合只能用**一套** `w`（一个像素不能按通道给三种权重），所以求解用 B；但报告出去的 `bias_bound` 是 `Vec3`，需要用 A 的逐通道界才有意义。B 的目标可合并成一个二次型：

```
M = diag( Σ_c v_c ) + Σ_c b_c·b_cᵀ            （n ≤ 4，秩 ≤ 4 的小矩阵）
min_w  wᵀ M w      s.t.  1ᵀw = 1,  w ≥ 0
```

**求解：闭式 + 主动集，不引入通用 QP 求解器**

1. 等式约束解：`w* = M⁻¹1 / (1ᵀM⁻¹1)`（一次 Cholesky 解，不需矩阵求逆）。
2. 非负约束：`n ≤ 4` 时可枚举 ≤16 个活动集；实用做法是「截断负分量 → 重新归一化 → 重复 ≤2 次」，与精确解在 `n ≤ 4` 上差异可忽略（需在 spike 里验）。
3. 通道相关性**不建模**：A 逐通道忽略它，B 通过逐通道和吸收；若将来要联合界，需显式给相关矩阵，当前不做。

**性能假设必须标为待验证（v2.3 的错误：直接写"ALU 上可忽略"）**：

- 解析成本估计：一次 4×4 Cholesky ≈ 30–40 次乘加 + 截断循环 ≈ 100–200 ALU/像素；在 1080p 全屏 composite 上属"可见但不大"的量级。
- **但这只是纸面估计**，必须由 spike 实测（pass 级 GPU timestamp 对比 composite 有/无 blend）：目标 ≤ 0.15ms @1080p（1080p 帧预算的 ~1%）。
- 若超预算，退路按优先级：①量化比值 `b_c/√v_c` 构建期打表 + 运行时三线性查表；②固定迭代 2 次的近似解；③降为 **B 在 `b = 0` 时的单权重解** `w_i ∝ 1/Σ_c v_{c,i}`（丢弃 bias 权重、接受更松的界，但**仍可用于实际混合**）。
- **退路 ③ 不能写成「A 的逐通道 `w ∝ 1/v`」**（v2.3 的错误）：A 是三套权重，一个像素只能混合一次、只有一套权重，逐通道解只能用于**报告**的 `Vec3` 界，不能用于混合。
- **禁止**在主 composite 路径里放入通用 QP 迭代求解器（迭代次数不定 = 帧时间不可预测）。

**重要性质**：由于 `b_i` 是上界，解出的目标值也是 MSE 的**上界**——保守方向正确，但会过度降权。因此必须同时报紧度（§9.4 条件 5），否则"安全"会退化成"什么都不用"。若所有 `b_i = 0`，B 退化为经典单权重 `w_i ∝ 1/Σ_c v_{c,i}`（此时才与 v2.1 的直觉一致；注意仍不是逐通道三套权重）。

---

## 6. 规划与预算

> 本节吸收数据库查询优化器、Kubernetes、ARINC 653、TCP BBR 的成熟方法。前提：**所有行业方案都必须先过「16ms 硬截止」这一道筛**。

### 6.1 成本方程与构建期标定

```
frame_ms ≈ Σ_pixels Σ_k N_k(p) · c_k + overhead
c_k      = c_sample + P_vis · c_trace + c_reconstruct

约束：Σ_k N_k(p) ≤ N_max(p)                      （per-pixel ray budget）
      Σ ≤ FrameResourceBudget                     （四轴）
```

`c_k` **构建期离线标定成表**（benchmark 生成），运行时只查表。这是"可预测性能"的全部依据：因为每个 estimator 都是固定采样数形态，`c_k` 可标定，预算可在运行前算出来。

### 6.2 规划输入

`SceneStats`（灯数 / 单类灯上限 / NPR 材质数 / 代理页比例 / 三角密度）+ 设备 profile → `BudgetCompiler`。

> **没有统计量的 plan 是赌博**——这是查询优化器给的最大教训（cardinality 估错的代价远大于算法本身差）。

### 6.3 分区预算（ARINC 653 式）

| 分区 | 硬窗 | 闲置处理 |
|---|---|---|
| Trace（shadow / GI / spec ray） | gpu_ms 窗 + 带宽窗 | 可被 paging 回收 |
| Paging（页传输） | IO 窗 | 可被 trace 回收 |
| Shading（raster 解析） | gpu_ms 窗 | 固定 |

分区而非优先级仲裁的理由：**最坏情况可静态推理**（这正是航电分区要的确定性）。分区必须 work-conserving（闲置可回收），否则浪费。

### 6.4 每链 limits + 驱逐顺序 + 准入（K8s QoS 式）

- 每链有 `requests`（保证）/ `limits`（上限）。
- 超预算时按**固定顺序驱逐**：先降 spec 采样 → 再沿 §5.3.3 的降级链退 spec 的 Rep 档（`RT → SSR → probe`）→ 再降 GI 采样 → 最后降 `DdgiProbes` 页精度。
- **准入控制**：新页请求在 IO 窗满时被拒绝 / 排队，而不是拖慢全局。
- 驱逐必须可预测，不允许抖动：页至少驻留 N 帧。

### 6.5 摊销调度（Decima + EDA incremental STA）

`FieldEpoch` 变更 + `SceneStats` → 摊销器决定本帧更新谁。

重算顺序用 **level-based**（依赖层级）：重要性只决定**优先级**，不决定**正确顺序**。这是 EDA 增量 STA 的核心经验。

### 6.6 测量闭环（TCP BBR 式）

上一帧实测 `c_k` 与分区用量 → 下一帧预算修正。

**必须阻尼 + 限幅**（增益 ≤ 0.5，单帧变化 ≤ 25%），否则抖动比静态表更差；退化路径是回退静态表。

### 6.7 QualityContract 分层叠加（USD composition 式）

```
平台默认 ⊂ 设备 profile ⊂ 场景 ⊂ 用户
```

叠加必须做**冲突检测**（不允许"两种配置都生效"）。这是 UE "scalability + device profile + PP volume" 的正式化版本。

### 6.8 EXPLAIN 与 plan regression

- `explain(plan)`：选了什么、拒绝了什么（R1/R2/R3）、为什么降档、分区占用。
- **plan regression test**：新增 recipe / 调整系数不许让既有场景的 plan 变差。

---

## 7. 增量与失效

### 7.1 红绿失效传播（rustc/salsa + LLVM）

```rust
// pass 侧：destroys: Vec<CacheTag>        （已进 PassDecl）
// 变更侧：ChangeJournal 记录 (field, old_epoch)
// 框架：从根标记 red → 验证 green → 只重算 "red 且未 green" 的节点
```

关键：**"这个 pass 破坏什么"必须显式声明**（LLVM preserved analyses 的教训）。现状是"人记得清哪些缓存"，这正是增量 bug 之源。

### 7.2 WAL + 快照 + 版本

| 机制 | 作用 | 来源 |
|---|---|---|
| WAL 语义 | 先记账后改状态 → 异常可回放 | 数据库 |
| `JournalSnapshot` | 显式序列化整个采样状态 | 影视 checkpoint |
| MVCC 式版本 | 每个 history slot 标注它基于哪个 `FieldEpoch`，过期即 soft-invalidate | 数据库 MVCC |

> 增量 bug 若不可回放，就不可复现，也就不可修。

### 7.3 失效三级

| 级 | 条件 | 动作 |
|---|---|---|
| hard | 几何 / 材质拓扑变更 | 清空重建 |
| soft | 光照 / 环境变更 | 降 confidence，不清空（避免跳变） |
| warm | 预热 / 烘焙 | 低 confidence 入池，不当真值 |

### 7.4 等价性验收（EDA LEC 式）

**增量结果 vs 强制全失效结果**：
- 确定性路径：**bit-exact**；
- 随机路径：PSNR + 能量差双门槛。

这是形式等价检查（LEC）的软件版。

---

## 8. 执行模型与性能

### 8.1 执行模型（公理 2）

| 机制 | 作用 | 落点 |
|---|---|---|
| Work compaction | 压实活跃 lane，消灭空转 | `prism_render_trace` |
| Ray reorder | 按命中材质 / 方向分桶，提高遍历相干性 | 复用 `ray_reorder_gpu`（691 行/2 测试） |
| Persistent thread | 免 dispatch 与 tail effect | trace 后端统一 |
| Async compute | shadow / GI ray 与 raster 重叠 | `PassDecl.async_split` |

RayQuery 路径上 compaction + reorder 通常能拿回 **15–30%** 的空转开销——远大于"少做两个 pass"。

### 8.2 带宽与 VRS

- fp16 全链路；irradiance 用 octahedral 2×u16；radiance cache 用 BC6H；bindless 描述符；pass aliasing。
- **VRS = 第五旋钮**：GI / 阴影采样密度按 tile 重要性（距离 / 运动 / 焦点区）分配，是低端档最便宜的一档性能来源。

### 8.3 降噪：每信号独立（NRD 式）

| 层 | 历史 | 实例 | ALU 预算 |
|---|---|---|---|
| Direct（随机直接光） | 4–8 帧 + 重投影校验 | RELAX 式 | 15–25% |
| GI（漫反射） | 16–32 帧 + 方差衰减 | RELAX 长历史 | 15–25% |
| Specular | 4–8 帧 + **反射向量域**重投影（§5.3.3） | SIGMA 式半分辨率 + 深度引导上采样 | 15–25% |

**每信号独立实例 + 独立 validation 策略**。一个万能降噪吃三种信号是这个行业最常见的失败点。

> 镜面行的三处参数（重投影域、半分辨率接缝责任、能量项）**不在本表定义**，一律引用 §5.3.3——本表只给历史长度与 ALU 预算。

### 8.4 可伸缩性矩阵

| 档 | recipe | 采样 | 分辨率 | Rep 保真度 |
|---|---|---|---|---|
| Ultra | 全部 + 2 bounce | 8–16 | 100% | 全驻留 |
| High | 全 stochastic + 1 bounce | 4–8 | 100% | 页级 |
| Medium | 4 个（+VSM 阴影） | 2–4 | 75% | 页级 + 代理 |
| Low / 移动 | `CascadeMap` + `SparseSdfGi` / `ScreenProbe`；**镜面固定 `ProbeSplitSum`**（无 RT 档） | 1 | 50% | 代理为主 |

移动端按**可持续 ms（热预算）**而非峰值 ms 设预算。

### 8.5 测量项（门槛见 §18）

pass 级 GPU timestamp、`pages_rebuilt/frame`、`invalidate_count/frame`、分区闲置率、在飞 ray 数、队列深度。

**这里只规定"测什么"，不规定"等于多少"**：所有门槛按 **硬件类 × 场景类 × 分位**（P50/P95/P99）在 §18 给出，不使用平均值，也不使用绝对零门槛（v2.1 的 `hitch = 0` 属不可度量，见 §0.5 第 8 条）。

---

## 9. 精度、流送与保真度

### 9.1 页池

| 机制 | 内容 |
|---|---|
| key 空间 | 挂 **world partition cell**：cell 卸载 = **批量失效**（1 个事件，不是 N 次逐页失效） |
| 驻留量 | **working set 模型**（Denning）：按最近 Δ 帧访问集定 `resident_bytes`，不拍数 |
| 驱逐 | K8s QoS 式顺序 + 页至少驻留 N 帧（防抖动） |
| 预热 | checkpoint 预热，低 confidence 入池 |

### 9.2 保真度与 SSE

```
SSE = geometricError × screenHeight / (2 · distance · tan(fov/2))
```

`RepReadiness.sse ≤ RepSpec.max_sse` → 该表示可用；否则换更高保真页 / 走 `fallback()`。

**任何时刻出图**：`fallback_available == false` 的 `RepProvider` 实现不允许注册。

### 9.3 DDGI 的 brick 化

规则网格 → **自适应 brick + 页化** → 与 `prism_render_virtual` 合流。

> 这一步把「DDGI」与「虚拟流送」从两件事变成**一件事**：探针的驻留 / 精度 / 更新全部走页池，不再有第二套内存管理。这是 v2.1 里结构性收益最大的一处。

### 9.4 误差模型：bias 与 variance 分列

v2.1 把代理误差（`proxy_error` / `sse`）塞进 `variance`，是**概念错误**：

- **variance** 来自采样不足，可被时空重建**平均掉**（多帧累积、降噪都会降低它）；
- **bias** 来自表示 / 代理 / 缓存滞后 / 过宽的锥，**平均不掉**——累积多少帧都还是同一个错误方向。

两者对降档的作用完全不同：

| 通道 | 来源 | 可降低手段 | 降档动作 |
|---|---|---|---|
| `variance` | 采样数不足 | 时域累积、降噪、加 spp | 增加采样 / 换更长历史的重建 |
| `bias_bound` | 代理几何、缺页、缓存过期、cone 过宽 | **只能**升级表示或保守兜底 | 提页优先级 / 换高保真 `Rep` / 走 `fallback()` / 收窄 cone |

误差预算是**上界**，不是等式（推导见附录 C）：

```
真值关系（等式）：  MSE = Var[Î] + bias²
只知道上界时：      MSE ≤ Var[Î] + bias_bound²                 ← 真定理
用估计方差时：      MSE ≤ c_{1−α}(n)·Var_est + bias_bound²      ← 以概率 ≥ 1−α 成立
```

**`c_{1−α}(n)` 的定义**（v2.3 写的 `Var_est·(1+κ)(1−α)` 是错的：它随 `α` 增大而**变小**，不可能是上界；且 `(1−α)` 与"置信"无关）：

若 `Var_est = s²` 是 `n` 个独立同分布样本的**无偏**样本方差，则 `(n−1)·s²/σ² ~ χ²(n−1)`，于是

```
c_{1−α}(n) = (n−1) / χ²_α(n−1)        ← 上置信限乘子（≥1）
```

`χ²_α(ν)` 是自由度 `ν` 的 χ² 分布的 **α 下分位点**（`P(χ² ≤ χ²_α) = α`），`1−α` 为置信水平。

| 事项 | 约定 |
|---|---|
| 取值范围 | **只采用 `α ≤ 0.1`（置信 ≥ 90%），其他范围一律不采用**。`c → 1` 当 `n` 增大。注：`c < 1` 的精确条件是 `χ²_α(n−1) > n−1`（即 `α > 1 − F_{n−1}(n−1)`），该阈值随自由度变化（`n−1 ≥ 3` 时落在约 (0.5, 0.61]），**不是固定的 `α > 0.5`**——v2.3 的写法既不精确也不必要，本版不再给通用边界，只约束可用域 |
| 免查表实现 | Wilson–Hilferty：`χ²_α(ν) ≈ ν·(1 − 2/(9ν) + z_α·√(2/(9ν)))³`，`z_α` 为标准正态 α 分位（`α<0.5` 时取负）；误差 <1%，可在 CI 与 shader 共用 |
| 参考值（α = 0.05） | n=8 → c≈1.71；n=32 → c≈1.28；n=128 → c≈1.13；n=512 → c≈1.06 |
| 用哪个 n | 用**有效样本数** `n_eff`（按 §9.5 的时域折损），不是名义帧数 |

**该上界成立的前提**（与下面"五个条件"并列，缺一条则 `c` 无效）：

1. 样本**独立**；时域重投影后并不独立，故 `n_eff` 已折损，但 χ² 前提被弱化——必须在验收里报"理论界 vs 实测界"的松紧比。
2. 单样本估计量**近似正态**。重尾（firefly）时 χ² 前提破裂、`s²` 系统性低估 → **必须先做 firefly 抑制再估方差**，或改用 bootstrap 分位；这条是渲染里最容易踩的失效点。
3. `Var_est` 必须是**降噪前**的原始估计（降噪后无法反推，见 §9.5）。
4. `w` 的求解不改变上述分布（通道间相关性已由 §5.4.1 的逐通道独立处理）。

**禁止**：任何"打折式"乘子（`σ²·(1−α)`、`σ²/c`、`σ²·(1+κ)(1−α)`）。上界只能**放大**方差，不能缩小。

- `Var[Î]` 项进 §6 的**采样预算**；`bias_bound²` 项进 §9 的**保真度预算**（页请求 / 表示升级）。
- **两项不共享预算，也不互相抵扣。**

**这不等式成立的五个条件**（缺一条就退回"不算数"）：

1. `bias_bound` 确实是该像素、该通道、该材质、该帧的 $|\mathbb{E}[\hat I]-I|$ 上界——不是全局平均或经验常量；
2. `Var[Î]` 由**独立样本**估计；用时间上相关的历史样本会**低估**方差；
3. `I` 与 recipe 的积分目标一致（同弹跳数 / 同定义域）；否则"bias"混进了定义差异；
4. 该分解**只在降噪前定义**。降噪是非线性算子，`MSE_post` 必须直接对照参考测，不能由前两项推出；
5. `bias_bound` 必须报告**紧度**（预测 vs 实测偏差 ≤ 2×，见 §18.3），否则上界过松会把降档过度触发。

**两项不共享预算，也不互相抵扣。** 由此修正 v2.1 的断言：代理误差**不能**"变成噪点"——它只能变成"更小的 bias"或"明确的兜底"；会变成噪点的只有 variance。

### 9.5 variance 的三级测量（v2.3 新增）

**"降噪降低了 variance" 是一个几乎恒真的命题，因此它不能当验收条件**——任何模糊都会降 variance，包括把重影和系统性错误一起糊进去。必须分级测：

| 量 | 定义 | 怎么测 | 用途 |
|---|---|---|---|
| `σ²_raw` | 单帧、单次采样的估计器方差 | 同一像素多帧**不相关**抖动（重置历史）取样本方差 | 采样预算是否够（§6） |
| `σ²_acc` | 时域累积后的方差 | 按**有效样本数** `n_eff`（非帧数）估计：`σ²_raw / n_eff`，`n_eff` 折扣由重投影置信度定 | 历史长度是否有效（§8.3） |
| `σ²_post` | 降噪后的输出方差 | **无法从单次输出估出**：只能用参考序列（高 spp 或离线 PT）对照，或两次不同抖动/不同降噪器实例的差 | 只作诊断，不作验收 |

**降噪的本质是 variance → bias 的交换**，所以验收必须直接看误差本身：

| 伪影 | 指标 | 为什么不能用方差代替 |
|---|---|---|
| 重影（ghosting） | 运动/镜头切/遮挡变化后 `k` 帧内与无历史参考的差 | 重影是**时域 bias**，方差可能同时下降 |
| 模糊（blurring） | 与参考的高频能量比（MTF 式），按频带分 | 模糊是**空间 bias**，同样能降 variance |
| 系统性偏移 | `MSE_post` 或 FLIP 对参考 | 唯一能同时抓到两种 bias 的量 |

**结论**：§18 的效果门槛一律用 `MSE_post` / FLIP / ghost / blur 四项，**不接受"variance 下降"作为通过条件**。

---

## 10. 效果工程

| 项 | 做法 | 验收 |
|---|---|---|
| 能量守恒 | GI 替换（非相加）ambient；`Σw=1` 只保证不放大不缩小，**不是**守恒 | **flux conservation 测试**（全白场景单灯通量守恒） |
| 多次弹射 | 1 bounce 默认 / 2 bounce High / RadianceCache Ultra | 对 path-traced oracle |
| 漏光抑制 | near / far field + SDF proxy + normal offset + 页级法线钳制 | 指定场景集 leak 面积占比 ≤ 阈值(H)（不用"零漏光"） |
| 降噪分层 | §8.3 三档独立 | 无重影 / 无拖尾 |
| 感知度量 | **FLIP 主门槛** + 能量差硬断言 + PSNR 仅作回归 | 双门槛 |
| 色调映射 | OCIO 式配置层（ACES / AgX 为 config，不动管线） | 换 tonemap 不改 pass 图 |
| 质量锚 | 所有 recipe 朝同一 `PathTracedOffline` oracle 收敛 | 感知误差 ≤ 阈值 |

### 10.1 NPR 确定性（硬约束）

NPR（三渲二）的明暗阶跃依赖**逐帧确定性**：随机采样 + 降噪会把阶跃抹成渐变、把描边糊掉。这是外部实证（Genshin / GGXrd 的工程实践），不是本方案推测。

因此：`R3` 剪枝 + deterministic twin 是**硬约束**，不是可选项。ADR 需记录此判断与实证来源。

---

## 11. 易用层

### 11.1 DebugView：契约驱动自动注册

**不维护手写视图清单。** 凡契约字段自动进面板：

`PageState / RepReadiness.sse / Contrib / ReservoirAge / BudgetUsage / PartitionIdle / Rejection / FieldEpoch`

由此自动得到：页驻留热图、Ray 可视化、方差热图、**保真度热图**、预算超支高亮、Reservoir 年龄、失效溯源、阴影槽位、**Contributor 分解**、NPR 分屏、能量误差图、fault 时间线。

**成本修正（v2.2）**：自动注册只覆盖**结构与数值**（枚举、范围、标量场）。"能看"和"可用"是两件事——可用的可视化还需要 recipe 声明显示语义：

```rust
pub struct DebugMeta {
    pub channel: ChannelSemantics,   // radiance / irradiance / visibility / age / mask
    pub range: (f32, f32),           // 色阶定义域
    pub scale: Scale,                // linear / log / perceptual
    pub temporal_window: u32,        // 时间轴聚合窗口
    pub combine_with: &'static [DebugView],
}
```

`DebugMeta` 可缺省，缺省即按原始值直显（通常不可读）。**结论：从"零 UI 工作"修正为"零 UI 代码，但每 recipe 约 10 行显示元数据"**——缺省不阻塞接入，只降低可诊断性。这条修正消除的是"接入成本被低估"，而不是结构收益本身。

### 11.2 EXPLAIN

```
tier     = High
recipes  = [DirectStochasticVSM, DdgiProbes, SurfaceCacheSurfel]
拒绝     : DirectStochasticRT  ← R2（无 RT capability）
           SpecularRestir     ← R3（场景含 3 个 NPR 材质）
降档     : gpu_ms 12.4 > 11.0 → 采样 8 → 4
分区     : trace 6.2/7.0   paging 1.1/2.0   shading 4.5/5.0
```

UE 靠 cvars + 人肉排查，Prism 靠结构化诊断——这是最便宜、回报最高的差异化能力。

### 11.3 三合一报告

每次提交产出 markdown artifact：正确性（L2/L4 等价性 + FLIP）+ 性能（p99 / 预算 / 分区闲置 / hitch）+ 画质（FLIP 分布）。

---

## 12. 易维护

| 项 | 做法 |
|---|---|
| 契约版本化 | `abi/v1`；破坏性变更走 ADR + 一次性切换 |
| 依赖边界 lint | `abi` 不依赖 gpu；`trace` 不依赖 `gi`；`cache` 不依赖 `virtual` 之外 |
| 测试金字塔 | L1 算子 / L2 接口（≤300 对拍）/ L3 帧 / L4 序列（增量 / 流送 / 别名） |
| **shader variant 门禁** | variant 数量进 CI，只许降不许升（无 ADR 不得例外） |
| **契约膨胀防护** | `PassDecl` / `CacheEffect` / `ExecPolicy` 三者物理分离；任一结构体不得同时承载"图依赖 + 缓存效果 + 执行预算"（v2.1 缺陷 5） |
| **镜面参数唯一源** | 镜面族的降级链 / 重投影域 / 半分辨率接缝 / 能量项只在 §5.3.3 定义；任何 recipe 自带 `enable_*`、`roughness_cutoff`、`history_len` 视为契约破坏（延续 §5.4.1「参数唯一源」原则） |
| **降级链登记** | 每条降级链（漫反射 / 镜面 / 阴影）的顺序只在 `constants` 定义一次；EXPLAIN、§8.3、§18.3 只引用不复制。新增一条链 = 同时登记 §18.3，否则该链无验收 |
| shader 覆盖 | 每个 WESL 必须被 oracle 覆盖，或显式标注"视觉专有" |
| ADR 制度 | 设计讨论不进主文档 |

---

## 13. 易拓展

| 扩展点 | 机制 | 保证 |
|---|---|---|
| 新 GI 技术 | 注册式 recipe（不 fork） | 0 处 if/else |
| 新追踪后端 | `RayBackend` + `BackendCaps` | 0 个调用点改动 |
| 新页客户端 | `VirtualResourceClient`（`prism_rendering_architecture_zh.md` §8） | 复用池 / 表 / 预算 |
| 新镜面 recipe | 同一 `SpecularRep` 的 `(target, sampler, estimator)` 参数组合；planar 除外（新 pass，须 ADR） | 白名单 +1 行、0 个新 `RecipeBackend`、0 处 if/else |
| 新材质响应 | `render_callback` 表项 | 不产生新 PSO |
| 引擎外插件 | `RenderFeaturePlugin`：**delegate（能力）/ index（场景索引）/ sync（增量同步）三段** | 不改 graph 编译器 |
| 幂等性 | 新增 recipe 不得改变既有 recipe 数值 | 回归断言 |

---

## 14. 借鉴总表

### 14.1 图形 / 引擎产品

| 来源 | 借鉴 | 不借鉴 / 落点 |
|---|---|---|
| UE5 Lumen | 多后端单 gather + 显式 fallback、Surface Cache、分层 radiance cache | 强依赖 HWRT；无 NPR → 补 deterministic twin |
| UE5 MegaLights | 随机直接光、per-light 阴影可插拔、ray guiding、专用降噪 | 不支持方向光 / 头发 / 水 / 半透明 → 变成 `R3` 剪枝 |
| UE5 VSM | 页表 + caster epoch + swept volume 局部失效 | per-light 开销 → 过四轴门控；本方案中降格 |
| RTXGI / DDGI | 探针 + 重定位 + 分类 | 单后端无屏幕空间融合 |
| ReSTIR / GRIS / SHARC | **估计器与复用解耦**（矩阵的理论基础） | 单技术当万能 |
| Radiance Cascades | 确定性、无历史依赖 | 内存开销 |
| Frostbite / CryEngine | temporal reprojection 工程细节、稳定降噪 | 体素分辨率 / 内存 |
| id Tech | wave compaction / ray reorder | 专用硬件假设 |
| RE Engine / Northlight | screen-space + fallback 的务实路线 | — |
| Unity HDRP | 可视化调试与文档体验 | 管线碎片化 |
| Nanite | 页级 LOD + 自动 fallback + 代理误差 | 绑定自研格式 |
| Metro Exodus RT GI | 单技术做到极致的对照系 | 硬绑 RT |

### 14.2 工程层产品

| 来源 | 借鉴 | 落点 |
|---|---|---|
| NVIDIA NRD | 每信号独立降噪 + history validation + reference 模式 | §8.3 |
| UE RDG | pass 只能靠资源表达依赖、自动 barrier、transient aliasing、split/join | §4 |
| VRS（Intel / UE / AMD） | 按 tile 分配着色密度 | §8.2 |
| Decima | 分帧摊销 + 更新时间账本 | §6.5 |
| Unity APV | 探针 brick 自适应 + 逐像素 + 流送 | §9.3 |
| AMD Brixelizer GI | 稀疏 SDF brick GI（无 RT 友好） | §5.3 |
| RTXDI / GAR | 光源重采样（质量上限高于固定 N ray） | §5.2 |
| Blender EEVEE Next | 同量级重构的公开案例；"预览即最终"；旧后端兼容层拖死新后端 | §7 + P0 冻结依据 |
| Filament | 单一材质格式、variant 数当 CI 指标、干净的 FrameGraph | §12 |
| USD Hydra | delegate / render index / sync 三段 | §13 |
| Godot RD / GLES3 | 双后端共享抽象 + 能力表 | §4 |
| UE World Partition | 流送单元绑世界分区 cell | §9.1 |

### 14.3 跨行业

| 来源 | 借鉴 | 落点 |
|---|---|---|
| 数据库查询优化器 | cost model + 统计 + EXPLAIN + plan regression | §6.1 / §6.8 |
| cardinality estimation | 场景统计量作为规划输入 | §6.2 |
| ML autotuning（Triton / TVM） | 构建期搜表、运行时零搜索 | §6.1 |
| rustc query system / salsa | 红绿失效传播 | §7.1 |
| LLVM pass manager | preserved analyses / 显式失效声明 | §4 |
| 数据库 WAL / MVCC / checkpoint | 先记账后改状态 + 快照 + 版本一致性 | §7.2 |
| EDA incremental STA | level-based 重算调度 | §6.5 |
| EDA technology mapping | recipe 库 + 特性表 + 约束满足 | §5 |
| EDA LEC | 增量 vs 全失效的等价性检查 | §7.4 |
| ARINC 653 | 时间 / 空间分区预算 | §6.3 |
| Kubernetes QoS / eviction / admission | requests / limits、驱逐顺序、准入控制 | §6.4 |
| TCP BBR | 测量闭环 + pacing | §6.6 |
| OS working set（Denning） | 驻留量与 thrash 判定 | §9.1 |
| GIS 3D Tiles（Cesium） | geometricError → screen-space error | §9.2 |
| 医学影像 / VTK out-of-core | 渐进加载 + 必须始终出图 | §9.2 |
| anytime algorithms（Boddy & Dean） | 随时可中断、**序列级**质量不退化 | §2.2 不变量 7 |
| USD composition arcs | 分层叠加 + 冲突检测 | §6.7 |
| OCIO | 配置与实现分离 | §10 |
| 影视 image regression | 参考图 + 容差分区 + 失败图存档 | §12 |
| 音频 voice stealing | 固定复音预算 + 按优先级偷声 + 平滑过渡 | §6.4（偷声必须渐进） |

### 14.4 反面教材（明确不抄）

| 来源 | 为什么 |
|---|---|
| Metro Exodus 纯 RT GI | 硬绑 RT，与 4 档可伸缩冲突 |
| Naughty Dog / lightmap 主路径 | 与 E1「实时无烘焙」冲突（`prism_derived_data_contract_zh.md:100`） |
| 神经 GI / DLSS Ray Reconstruction | 不可审计；最多 Ultra 可选，不进主干 |
| CryEngine LPV | 漏光 + 分辨率问题已无竞争力 |
| UE 的 cvars 海洋 | 易用反面：一百个 cvar = 没人知道哪组是对的 |
| Unity SRP 三套 | 易维护反面：正是"一个算法五份实现"的放大版 |
| 数据库的最终一致 | 一帧必须自洽 |
| EDA / 影视的小时级验证 | CI 必须分钟级（故 oracle 必须分层） |
| OS / K8s 的吞吐优先 | 本方案优化 p99 帧时间，不是均值 |
| 航电的允许浪费 | 帧渲染要求满负载（故分区必须 work-conserving） |
| 学术 anytime 的严格单调假设 | 降噪 / 重投影会破坏单调 → 用断言兜 |

---

## 15. 迁移阶段（不承诺单点工期）

> v2.1 给的「串行 35.5 周 / 并行后 ~30 周」是**路线图**，不是工期：这份方案没有团队规模、没有同类系统的历史速度数据、也没有代码量校准，任何周数都是编的（见 §0.5 第 7 条）。v2.2 改为「**前置 spike + 阶段 + 里程碑**」，并且**把 spike 放在承诺之前**——先证伪，再排期。

### 15.1 三个前置 spike（先证伪）

| spike | 要回答的问题 | 通过标准 | 不通过则 |
|---|---|---|---|
| **S1 契约可行性** | 7 个契约能否在 ≤1000 行内覆盖 16 个 recipe；**只验契约形状，不验后端物理** | 3 个异构 recipe（`DdgiProbes` / `SsgiScreenSpace` / `DirectStochasticRT`）接入时**不改契约**；后端用 `MockBackend` + `SwBvhBackend`（**不要求 RayQuery**，决策 #4 不阻塞本 spike）；两者通过 §4.1 的同一组行为测试 | 契约定型推迟，后续阶段不能开工 |
| **S2 增量正确性** | 红绿传播在**动态依赖图**上是否可维护 | 一组变更序列下，增量结果 == 全失效结果（确定性路径 bit-exact；随机路径 PSNR + 能量双门槛） | 退回粗粒度 epoch，放弃细粒度增量 |
| **S3 混合统计** | `Unbiased` + `Conservative` 双模式能否同时给「无偏」与「有界」 | flux 测试与高 spp 收敛测试**同时**通过 | 放弃统一 Blend，改为按载体分区合成 |

### 15.2 阶段与里程碑

| 阶段 | 里程碑（可验收） | 依赖 | 规模不确定性 |
|---|---|---|---|
| **P0 冻结清点** | 39 crate 归属表 + 24 个布尔 → recipe 映射 + 零引用 CI 上线 | — | **低**（纯清点） |
| **P1 契约** | S1 通过；7 契约冻结；7 不变量进 CI | P0 | **高**（S1 不过要重做） |
| **P2 Oracle** | L2 ≤300 对拍 + L3 帧级 + FLIP 基线 + 参考图体系 | P1 | 中 |
| **P3 追踪与阴影槽位** | 至少 2 个真实后端（`SwBvhBackend` + 其一）行为一致；阴影槽位 2–3 个；灯数成本曲线**可测**（RayQuery 是否进首期见决策 #4，不阻塞本阶段） | P1 | **高** |
| **P4 虚拟流送** | 页池 + cell 绑定 + `RepProvider`；任何时刻出图 | P1 | **高**（可与 P3 并行） |
| **P5 缓存与增量** | S2 通过；降噪三档；WAL / 快照 / 回放 | P3, P4 | **高** |
| **P6 算子化** | 16 recipe（含 4 镜面）+ 镜面信号契约 + Blend（S3 通过）+ 分区预算 + 摊销 | P5 | 中 |
| **P7a GI 垂直切片**（范围内） | GI 路径上的 crate（`abi` / `trace` / `cache` / `virtual` / `gi` / `graph` / `driver` / `host` / `debug`）无 `bevy_*`；一条端到端链路（场景 → trace → GI recipe → composite → present）在新宿主上跑通，并与冻结的 `prism_render_scene` 输出做 parity；边界外子系统（water / hair / cloth / particle / NPR / postfx）**保持冻结** | P6 | 中 |
| **P7b 全仓迁移**（**范围外**，单独立项） | 141 → ~101 的归档、全仓脱 bevy、其余子系统迁移 | — | 不承诺，可能永不做 |
| **P8 收敛项** | EXPLAIN / variant 门禁 / plugin sync / OCIO 配置层 | P7a | **低**（可整段砍） |

**唯一可给的量化**：P0 是**天级**；P1–P2 是**月级**；P3–P7a 各为「数月」，彼此量级相近；P8 可无限期后置。P7b **不计入**这条路线（范围外、单独立项）。

**要真实工期，必须先用 S1–S3 的实测速度外推**——把 spike 的吞吐当作唯一可信输入，而不是先写一个总周数再往里塞任务。

**并行轨道 B1 烘焙**：warm start + 离线参考图，不占主线，按 `prism_derived_data_contract_zh.md:65` 的 B1 登记。

---

## 16. 风险登记册

| 组 | 风险 |
|---|---|
| 地基 | R1 契约抽象失败（>1000 行）· R2 oracle 不足 · R3 recipe 白名单过僵 · R4 RT 后端分裂 · R5 NPR twin 工作量 · R6 大面积重写失控 · R7 宿主层遗漏 |
| 流送 | R8 页池与既有资产冲突 · R9 thrash · R10 feedback 延迟 · R11 与增量耦合 · R12 cell 卸载撕裂 |
| 烘焙 | R13 warm start 混入错误数据 · R14 增量 bug 不可复现 · R15 接缝 · R16 Schema 漂移 · R17 验收成本 |
| 执行 | R18 执行模型复杂度 · R19 调试视图维护反噬（已由补丁 6 消除）· R20 FLIP 门槛不稳 |
| 新增 | **R21** 红绿算法在动态依赖图上的正确性（缓：先粗粒度 + 断言） |
| | **R22** 分区预算闲置率过高（缓：分区须 work-conserving） |
| | **R23** `SparseSdfGi` 内存 / 质量不达标（缓：首期只覆盖 Medium / Low） |
| | **R24** APV 式 DDGI 复杂度超预期（缓：先 brick 后页化，两步走） |
| | **R25** 测量闭环抖动（缓：阻尼 / 限幅，退化回静态表） |
| | **R26 范围膨胀**（26 补丁本身是最大风险；缓：P8 可整段砍掉） |
| | **R27 bias / variance 混用**（症状：降噪后画面依旧错，且无法定位是采样不足还是表示太粗；缓：双通道 + `bias_bound` 全程外传 + 附录 C 的 `Var` 与 `bias²` 分开测） |
| | **R28 验收不可执行**（症状：绝对零门槛让 CI 恒红，随后被人静默放宽；缓：硬件 × 场景 × 分位矩阵，禁用绝对零，只有布尔类事实允许 100%） |
| | **R29 范围蔓延**（症状：GI 垂直切片被扩写成"全仓脱 bevy + 141→101"的迁移，工期与风险重新失控；缓：P7a / P7b 硬边界，"141→101" 只是 P0 记账、不作 P7 验收） |
| | **R30 降噪掩盖 bias**（症状：用 "variance 下降" 证明降噪有效，实际把重影/模糊糊进去了；缓：只认 `MSE_post` / FLIP / ghost / blur 四项，§9.5） |
| | **R31 方差估计在重尾下失效**（症状：firefly 让 `s²` 系统性低估、`c_{1−α}` 也救不回来，界看着成立实际被违反；缓：先 firefly 抑制再估方差、验收报"理论界 / 实测界"松紧比、必要时改用 bootstrap 分位，§9.4） |
| | **R32 权重求解超预算**（症状：`c` 与 QP 闭式的实测成本吃掉 composite 预算，或为省成本退化成 `w∝1/v` 又把 bias 权重丢掉；缓：S2 起就接 pass 级 GPU timestamp 对照，超预算按查表 → 固定 2 次迭代 → 退单权重 `1/Σᶜv` 三级退路，§5.4.1） |
| | **R33 无 RT 档的镜面塌陷**（症状：H1 上 `SpecularRestir` / `SpecularScreenSpace` 全不可用，只剩 `ProbeSplitSum`，粗糙度 0.0–0.2 的锐利反射消失且无 EXPLAIN 记录；缓：§5.3.3 要求降级链单调且可解释，§18.3 新增"降级单调性"布尔门槛，S7 必测 H1） |
| | **R34 半分辨率接缝被当降噪误差**（症状：SSR 命中/未命中边界的跳变被计入 `MSE_post`，靠调大降噪半径"解决"，实际是 upsampler 责任缺失；缓：§5.3.3 把接缝责任显式划给 upsampler，§18.3 单列 seam 门槛，禁止用降噪验收替代） |

---

## 17. 决策清单（17 项，待拍板）

| # | 决策 | 建议 | 状态 |
|---|---|---|---|
| 1 | 单一实现源（GPU/WESL 唯一） | 接受 | 待定 |
| 2 | 冻结 + 归档 ~39 个零引用 crate | 授权 | 待定 |
| 3 | recipe 白名单（禁运行时任意组合） | 接受 | 待定 |
| 4 | RT 首期：只 SW BVH / 同时 RayQuery | 先 SW BVH（**不阻塞 S1/P3**，见 §15.1） | 待定 |
| 5 | NPR deterministic twin 是否硬约束 | 是 | 待定 |
| 6 | 是否允许重写 `shading_resolve.wesl`（2,938 行） | 允许，拆表驱动 | 待定 |
| 7 | 流送是否一等公民 | 是 | 待定 |
| 8 | 阴影槽位首期几个 | 大世界 3 个 / 中小 2 个 | 待定 |
| 9 | 烘焙首期产出 | oracle + warm start 双产出 | 待定 |
| 10 | 分区预算 vs 优先级仲裁 | 分区（可静态推理） | 待定 |
| 11 | `SparseSdfGi` 是否进首期 | 进，但只覆盖 Medium / Low | 待定 |
| 12 | spike 是否允许 mock backend | 允许，但**必须另有 ≥1 个真实后端**（SW BVH）通过同一组行为测试 | 待定 |
| 13 | 垂直切片边界 | 只覆盖 GI 路径（P7a）；其余子系统冻结，全仓迁移另立项（P7b，范围外） | 待定 |
| 14 | 置信水平 `α` 与 `c_{1−α}(n)` 的落点 | `α = 0.05`；`c` 按 `n_eff` 分档**构建期打表**，shader 只查表插值（不在热路径算 χ²） | 待定 |
| 15 | 权重求解的预算与退路阈值 | 目标 ≤0.15ms@1080p；超则查表 → 固定 2 次迭代 → 退单权重 `w_i ∝ 1/Σ_c v_{c,i}`（三级退路，§5.4.1） | 待定 |
| 16 | specular 首期是否含屏幕空间 SSR | **含**——H1 无 RT，`ProbeSplitSum` 单靠 split-sum 覆盖不了镜面/水面的锐利反射；`post_ssr_trace` 资产已存在，接入成本低于新写 | 待定 |
| 17 | `PlanarReflection` 是否进首期 | **不进**——水面/镜子用 `SpecularScreenSpace` + probe 兜底；planar 作为 R3 确定性路径的后续项（规避 R33）；维护性理由见 §0.11 第 3 条 | 待定 |

---

## 18. 验收矩阵（硬件 × 场景 × 分位）

> v2.1 的 `≈0` / `= 0` 无法执行：既不能判红，也不能判绿。v2.2 的所有门槛必须带三元组 **硬件类 × 场景类 × 分位数**，并禁用绝对零门槛（见 §0.5 第 8 条）。

### 18.1 硬件类（H）

| 代号 | 硬件 | 特征 |
|---|---|---|
| H1 | 集显 / 移动 SoC | 无 RT、带宽敏感、热受限 |
| H2 | 笔电独显 | 有 RT 但功耗受限 |
| H3 | 桌面独显 | 有 RT，主目标平台 |
| H4 | HW-RT 高阶 / 主机级 | 允许 RayQuery 作主路径 |

### 18.2 场景类（S）

| 代号 | 场景 | 测什么 |
|---|---|---|
| S1 | 静态室内，少灯（≤8） | 基线、收敛、能量守恒 |
| S2 | 动态物体 + 镜头运动 | 重投影、disocclusion、拖影 |
| S3 | 海量光源（8 → 256） | 成本是否与灯数解耦 |
| S4 | 大世界流送（页 fault 密集） | hitch、thrash、任何时刻出图 |
| S5 | 重 NPR / 三渲二 | 确定性、阶跃保真、逐帧稳定 |
| S6 | 半透明 / 毛发 / 水 | 专用确定性路径不被随机化污染 |
| **S7** | 镜面反射：粗糙度阶梯 0.0 / 0.1 / 0.3 / 0.6 / 0.9 + 平面镜 + 水面 + 无 RT 档 fallback | 降级链单调性、SSR 接缝、Fresnel 能量 |

### 18.3 性能与效果门槛（H / S / 类别 / 分位 四列显式）

四列缺一即为未定义（§18.5 规则 2）。`H` / `S` 是**显式列**而不是藏在备注里——机器解析（CI 表驱动）与人工评审读同一份表。`类别` 只有三种：**分布型**（逐帧/逐像素/逐事件有分布）、**布尔型**（有/无）、**确定性标量**（每场景一个确定值）。

| 维度 | 指标 | H | S | 类别 | 分位 | 门槛形式 |
|---|---|---|---|---|---|---|
| 性能 | 帧时间 | H1–H4 | S1–S7 | 分布型 | `P50` / `P95` / `P99` | `P50 ≤ B`，`P95 ≤ 1.10·B`，`P99 ≤ 1.25·B`（B = 该 H 档预算） |
| 性能 | 灯数成本斜率 | H1–H4 | S3 | 分布型 | `P50`（另报 `P95`） | `(t256@P50 − t8@P50) / t8@P50 ≤ 阈值(H)`；H1/H2 允许更高 |
| 性能 | hitch | H1–H4 | S4（含 S3） | 分布型 | `P99.9` + `max` | 10 分钟内 `P99.9 ≤ 2·B`、`max ≤ 4·B`、`> 4·B` 帧数 = 0 |
| 流送 | thrash | H1–H4 | S4 | 分布型 | `P95`（10 分钟窗口 fault 率） | `P95 ≤ 阈值(H)`；连续 fault 提权后必须收敛 |
| 效果 | 能量守恒 | H1–H4 | S1（+S6 专项） | 确定性标量 | `max`（S1 场景集） | 全白单灯通量误差 `max ≤ 0.5%`，**且** `Unbiased` 模式对高 spp 参考无系统偏差 |
| 效果 | 镜面能量守恒 | H1–H4 | S7 | 确定性标量 | 粗糙度阶梯每档的 `max` | 对高 spp 参考的**通量比**误差 `max ≤ 0.5%`（与 S1 同尺度）；H1 只对 `ProbeSplitSum` 单独验收，**不因缺 RT 而报 `N/A`** |
| 效果 | 镜面降级单调性 | H1–H4 | S7 | 布尔型 | `100%` | 沿 §5.3.3 降级链后退时输出单调，且每步在 EXPLAIN 给出降级原因；任何跨级跳变即判失败 |
| 效果 | SSR 接缝 seam | H2–H4 | S7 | 分布型 | `P95`（命中/未命中边界像素集） | 边界两侧亮度跳变 `P95 ≤ 阈值(H)`；**不得**用 `MSE_post` 或降噪验收替代 |
| 效果 | 画质 | H1–H4 | S1, S2, S7 | 分布型 | `P50` / `P95` | FLIP：`P50 ≤ f1`、`P95 ≤ f2`（`f1/f2` 每 H 档单列） |
| 效果 | 混合偏置紧度 | H2–H4 | S2 | 分布型 | `P95`（另报 `P50`） | `bias_bound` 预测 / 实测偏差的 `P95 ≤ 2×`。**H1 不适用**：无 traced 贡献 ⇒ `bias_bound ≡ 0`，该行在 H1 上无定义而非"通过" |
| 精度 | 增量等价 | H1–H4 | S1, S2 | 确定性标量 | 最差帧（`min` PSNR） | 确定性路径 bit-exact；随机路径 `min PSNR ≥ X` 且能量差 ≤ Y |
| 精度 | 任何时刻出图 | H1–H4 | S4 | 布尔型 | `100%` | 100% 帧有可显示输出，无空洞/黑块 |
| 效果 | 降噪后误差 `MSE_post` | H1–H4 | S2 | 分布型 | 像素 `P50/P95`、帧 `P95` | `@P50 ≤ 阈值(H)`、`@P95 ≤ 2×阈值(H)`；**唯一**能同时抓重影与模糊 |
| 效果 | 重影 ghost | H1–H4 | S2 | 分布型 | `P95`（切镜/遮挡变化事件集）+ `max` | `k` 帧内与无历史参考的差 `P95 ≤ 阈值(H)`、`max` 不得出现可见拖影 |
| 效果 | 模糊 blur | H1–H4 | S2 | 分布型 | `P50` / `P95`（频带 × 像素） | 与参考的高频能量比 `P50 ≥ 阈值(H)` |
| 效果 | NPR 保真 | H1–H4 | S5 | 分布型 | 帧 `P50/P95` + 逐帧差 `max` | PSNR `P50 ≥ 阈值(H)`；逐帧差 `max ≤ ε`（分屏对拍） |

**两条读表规则**（避免"没跑就是通过"）：

1. **不适用 ≠ 通过**：某行在某 `H`/`S` 上不适用时，CI 必须报 `N/A` 并在报告里列出原因（如 H1 的混合偏置行：无 traced 贡献）。把 `N/A` 混同 `pass` 是本矩阵最容易作弊的地方。
2. **组合覆盖面**：上表给的 `H × S` 是**笛卡尔积的剪枝结果**，不是全组合。CI 跑的是"每行 × 其声明的 H/S 集合"，其余组合报 `uncovered` 并列进 §18.5 规则 4 的 ADR。

### 18.4 与硬件无关的门槛

> 本节条目按定义与 `H`/`S` 无关（CI、审计、代码度量类），因此 `H = S = N/A` 且这是**有效声明**而不是遗漏；它们不进第 18.3 节的 H × S 矩阵，判定方式见"验证"列，`阶段`列给出最晚生效阶段。

| 维度 | 指标 | 门槛 | 验证 | 阶段 |
|---|---|---|---|---|
| 易开发 | 新 GI 技术改动文件数 | ≤3 | 实测接入 | P6 |
| | 布尔开关数 | 不增长 | CI grep | 全程 |
| | shader variant 数 | 不增长 | CI | P8 |
| 易拓展 | 新后端改动点 | 1 组 trait / 0 调用点 | 实测接入 | P3 |
| 易开发 | 新 recipe 接入 | 白名单 1 行 + 1 个参数组合；0 个新 backend 实现 | 实测接入 | P6 |
| 易维护 | 降级链真值源 | 1 处（`constants`），其余引用 | CI grep（禁复制链序列） | P6 |
| 易维护 | 实现份数 | 1 + oracle | 审计 | P7a |
| | 对拍数 | ≤300 | CI | P2 |
| | 零引用 crate | 0 | CI | P0（记账动作，非迁移） |
| 易用 | DebugView 覆盖 | 契约字段 100% 可见（含 `DebugMeta` 缺省降级） | 自动注册审计 | P6 |
| | EXPLAIN | 可解释拒绝与降档 | 手工评审 | P8 |
| 安全 | 降档 | 资源不足时不丢帧不崩，且降档可回退 | 故障注入 | P5 |

### 18.5 门槛的校准规则

1. 阈值（`B`、`阈值(H)`、`f1/f2`、`X/Y`、`ε`）**先由实测标定再进 CI**，不允许凭空写数（这正是 v2.1 的错误来源）。
2. 每条门槛必须写清 **H、S、类别、分位**四项；**缺任一项的条目视为未定义**。
   - v2.3 之前在 §18.3 只写了阈值、未写分位，与此条自相矛盾；v2.4 已给 §18.3 补齐 `类别` 与 `分位` 两列。
3. **类别决定分位要求，不是可选项**：
   - **分布型**（逐帧 / 逐像素 / 逐事件有分布的量）→ 必须给分位；默认 `P50 + P95`，尾部敏感项（帧时间、hitch、thrash）另加 `P99/P99.9` 或 `max`。
   - **布尔型**（崩溃、空洞、黑屏、缺页失败）→ 只能写 `100%`。
   - **确定性标量**（每场景/每测试一个确定值）→ 必须写**定义域内最坏值**（`max` / `min`），不允许只报均值（能量守恒、bit-exact、最差帧 PSNR）。
4. 布尔型允许 100% 是第 3 条的特例，不再单列（旧规则 3 与旧规则 2 的冲突由此消除）。
5. 门槛收紧需 ADR；放宽需附实测数据 + 失败图存档。
6. **降噪类验收不得用「variance 下降」作为通过条件**：那是恒真命题，且会掩盖重影与系统性偏差（§9.5、§0.6 第 2 条）。

---

## 附录 A：审计证据索引

行号为本次只读审计（`main @ fad551316`）时的快照，随代码演进可能漂移。

| 结论 | 证据 |
|---|---|
| CPU golden 层的自述定位 | `pkg/prism_render_shading/src/gi/mod.rs:1-6` |
| OpenPBR über 直接光 golden | `pkg/prism_render_shading/src/lighting.rs`（`SurfaceSample:24`、`evaluate_principled_direct:368`、`linear_furnace_response:923`） |
| 声明式 pass 图 | `pkg/prism_render_scene/src/shading/graph.rs:80`（`visibility_raster`）、`:133`（`texel_space_shade_cache`）、`:185`（`shading_resolve`） |
| 后处理 pass 声明 | `pkg/prism_render_scene/src/shading/post_graph.rs:215`（`post_ssr_trace`）、`:245`（`post_ssgi_base`）、`:263`（`post_spec_gi_reservoir_prior`） |
| 镜面反射资产（v2.6 处置：转正） | `post_graph.rs:215`（`post_ssr_trace`）、`:263`（`post_spec_gi_reservoir_prior`）、`spec_gi_*.wesl`；此前既不在 §3.4 净删名单也不在转正名单（§0.10 第 5 条） |
| 光照模块只做输入、数学在 golden/shader | `pkg/prism_render_scene/src/lighting/mod.rs:1-15` |
| 环境探针 → SH 投影 | `pkg/prism_render_scene/src/lighting/probe.rs:1-16`（`project_image_to_sh:85`、`EnvironmentProbeCache:96`） |
| 光照 ABI 的 WESL 孪生 | `pkg/prism_render_scene/src/shaders/lighting.wesl`（`sample_punctual_light:139`、`sh_irradiance:208`、`env_brdf_approx:219`） |
| 阴影三套实现并存 | `pkg/prism_render_scene/src/shaders/shadow.wesl`（PSSM + PCSS + 点光 cube）、`vsm_*.wesl` |
| GI 各后端的 shader 事实 | `ddgi_probe_update.wesl`、`ssgi.wesl`、`world_restir_*.wesl`、`surface_cache_*.wesl`、`spec_gi_*.wesl` |
| 无 lightmap 管线 | `pkg/prism_render_scene/src/opaque/queue.rs:242`（`lightmap_slab: None`） |
| 主路径"实时无烘焙" | `docs/prism_derived_data_contract_zh.md:100`（E1） |
| 可选烘焙缓存登记 | `docs/prism_derived_data_contract_zh.md:65`（B1，`OfflineShipped`） |
| 分档与 Compatible 档 GI | `docs/prism_rendering_architecture_zh.md:1035`、`:1452` |
| 虚拟资源客户端与页池设计（已声明未实现） | `docs/prism_rendering_architecture_zh.md` §8 |
| 四前端唯一执行路径 | `docs/prism_material_pipeline_refactor_zh.md`（步骤 4：缓存命中 → `IncomingRadiance`；未命中 → 屏幕空间直接积分） |
| bevy 轨移植切片 | `docs/prism_render_scene_rhi_port_zh.md`（S1–S8） |
| 新宿主已有 direct + multi-surface initial-GI GPU compute slice；OCIO 已进入 window 与 offscreen/headless RenderGraph，headless 终端输出通过 `OffscreenFrameOutput` 发布；offscreen 物理 GPU parity 已通过（宿主 graph 执行后，以 seeded target 对相同 transform graph 的多组非零 RGB/alpha 做 CPU 对拍，非零值未直接注入完整 scene host graph）；剩余 14 个 recipe 与 per-surface trace 尚未闭合 | `pkg/prism_render_app/src/lighting_gpu.rs`、`pkg/prism_render_app/src/display_transform_gpu.rs`、`pkg/prism_render_app/src/presentation.rs`、`pkg/prism_render_app/src/frame.rs`、`pkg/prism_render_app/src/systems.rs` |

---

## 附录 B：术语表

| 术语 | 含义 |
|---|---|
| **recipe** | 一个可命名的 `(Rep, Sampler, Estimator, Reuse, Reconstruct)` 组合实现，白名单制 |
| **Rep** | 场景/辐射的表示（BVH、SDF brick、探针网格、surfel、级联…） |
| **契约** | 跨 crate 的纯数据接口；总量受 1000 行硬约束 |
| **纵切链** | 贯穿多层的单一机制通道（预算 / 增量 / 追踪），唯一入口 + 唯一出口 |
| **L1–L4 oracle** | 算子级 / 接口级 / 帧级 / 序列级测试基准 |
| **SSE** | screen-space error，屏幕空间误差；是 **bias** 的空间度量，用于选级，不是 variance |
| **anytime** | 随时可中断、任何时刻返回当前最优解的算法性质 |
| **EXPLAIN** | 结构化输出"为什么选这些 recipe、拒绝/降档原因" |
| **B1 / E1** | `prism_derived_data_contract_zh.md` 中的烘焙 GI 缓存条目与实时无烘焙条目 |
| **σ²_raw / σ²_acc / σ²_post** | 单帧原始采样方差 / 时域累积后方差（按有效样本数）/ 降噪后方差（仅可对照参考测） |
| **垂直切片（P7a）** | 只覆盖 GI 路径的端到端可运行链路；其余子系统冻结，全仓迁移（P7b）明确范围外 |

---

## 附录 C：误差模型与降档（形式化）

设某像素某 recipe 的贡献估计为 $\hat I$，真值为 $I$：

$$\mathbb{E}\big[(\hat I - I)^2\big] = \underbrace{\operatorname{Var}[\hat I]}_{\text{可被时空重建降低}} + \underbrace{(\mathbb{E}[\hat I] - I)^2}_{\text{bias}^2,\ \text{平均不掉}}$$

因 `bias_bound` 是 $|\mathbb{E}[\hat I]-I|$ 的上界，且 `variance` 本身由有限样本估计，**能写出的只有不等式**：

```
MSE ≤ Var[Î] + bias_bound²                    （真上界，需 §9.4 的 5 条成立条件）
MSE ≤ c_{1−α}(n_eff)·Var_est + bias_bound²     （用估计方差时的置信形式，概率 ≥ 1−α）
c_{1−α}(n) = (n−1) / χ²_α(n−1) ≥ 1            （α ≤ 0.1 时；Wilson–Hilferty 免查表）
```

**上置信乘子的三条禁令**（v2.3 曾写 `·(1+κ)(1−α)`，是错的）：

1. **只能放大**：`c ≥ 1`。任何 `·(1−α)`、`·1/c` 形式的"打折"都让不等式反向，不成立。
2. **`α` 是失效概率、不是置信水平**：`α` 越小界越松。**只允许 `α ≤ 0.1`**，此范围外不采用（`c < 1` 的精确阈值随自由度变化，不能写成 `α > 0.5`）。
3. **`n` 必须是有效独立样本数**：时域累积的名义帧数会高估 `n`，`c` 因此偏小（偏乐观）。


**推论**：

1. **降噪只能处理第一项。** 把 bias 记进 variance 会导致「降噪之后仍然错」，且无法归因是采样不够还是表示太粗——这正是 v2.1 的错误（§0.5 第 1 条）。
2. **预算按边际收益分配。** `variance` 项加采样（§6），`bias²` 项加保真度（§9）；两项不可互抵，否则会用「多采几帧」去掩盖「表示太粗」。
3. **混合只在 `bias = 0` 时有无偏语义。** `Σw = 1` 本身只是凸组合；`Conservative` 模式下误差界为 $\sum_i w_i \cdot \mathrm{bias}_i$（§5.4）。
4. **验收必须能分别测两项，且 variance 要分阶段测。** `σ²_raw` 用不相关抖动样本、`σ²_acc` 按有效样本数折算、`σ²_post` 只能对照参考序列；`bias²` 用与离线 path-traced 参考的期望差。只测其一、或只报"降噪后 variance 变小"的门槛都是不完整的（§9.5）。
5. **上界松紧决定可用性。** 上界越松，降档越保守、画质越差；因此 `bias_bound` 必须报紧度（§9.4 条件 5、§18.3），否则"安全"会退化为"什么都不用"。

**降档的形式化定义**（把「降档」从形容词变成判据）：

```
keep(recipe r) ⟺ Var_r ≤ V_budget(tier) ∧ bias_r ≤ Bias_budget(tier)
否则 按  ∂(MSE)/∂cost 最小 的顺序降级：加采样 → 降历史长度 → 降 Rep 保真 → 换 recipe
```
