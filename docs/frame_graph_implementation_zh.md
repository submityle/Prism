# Frame Graph：单内核基础设施与 Scene 生产接入

日期：2026-10-16。工作区：`/Users/wangkailong/Documents/GitHub/Prism`。

单内核、受检声明与执行接口已有实现；Scene shading/water/post 通过专用 shading 图接入真实生产系统。本文记录源码契约、各范围实际检查点与证据边界；专用图的生产接入不等于通用 Fabric 已接管整帧。

## 唯一 authority 与受检入口

- `prism_render_graph::compile::kernel` 统一 SSA、RAW/WAR/WAW、确定性拓扑、生命周期与 checked first-fit；queue batches 由 `compile::schedule::queue_batches` 派生。
- 可执行 `RenderGraph` 与声明 `frame_graph::GpuFrameGraphBuilder` 共用该内核；`prism_render_architecture::frame_graph` 只重导出，不维护第二份 compiler。
- `GraphLimits`、`try_*` admission 在缩窄 ID、增长表、修改版本或运行 pass setup 前检查。拒绝访问不改该操作的 usage/version/access table；这是单操作原子性，不是用户 setup closure 的全事务回滚。
- `ExecutionPlan` 暴露逐访问 `CompiledAccess { resource, input_version, output_version }`、logical barriers、order/liveness/position 与 queue batches。`LogicalResource::Texture / Buffer` 保持各自 namespace，不以同数值索引冒充同一资源。
- driver StateTracker lower 具体 layout、子资源及 imported 入口/出口状态，不重新求解 SSA。logical queue batches 不等于物理多队列提交。
- `Cycle` 提供闭合路径、名称及 blocked 下游；旧版本 overwrite 仍按物理前驱建立 WAW/WAR，不丢迟声明旧版本读者。生命周期区间包含端点，仅 `last < first` 可复用。

`frame_graph/builder.rs` 的三种资源输入入口共用权威输出：

| 接口 | 输入与版本责任 |
|---|---|
| `compile()` | 先按显式控制依赖规范化 lowering 顺序，再产生隐式资源输入 |
| `compile_with_inputs()` | 每个原始 pass/access 提供数值输入版本；输入必须在 lowering 时存在，允许读旧版本或以旧输入写入；输出仍由 checked builder 分配 |
| `compile_with_sources()` | 接受数值输入或 `AccessInput::ProducedBy { pass, access }`；引用另一 pass 对同资源的确切写，补入 producer 前置条件；不由调用者猜输出版本 |

`dependency_order()` 仅请求共享内核的控制顺序，不进行完整资源编译。Fabric 使用它安排 admission 后做一次完整 symbolic 编译，并在最终 descriptor/runtime 中保留 `depends_on`；不再要求全局唯一 writer，也不把最终控制边清空。详细 ResourceInput 链见 `docs/render_fabric_selection_implementation_zh.md`。

### 原始 PassId、显式 SSA 与控制依赖

`graph.rs::compile_declarations` 的 producer-first lowering 使用临时 pass 索引；当前实现在线性还原 pass 表与依赖的**原始声明 ID**后，才调用唯一的 `graph.compile()`。因此 `order` 的元素、`alive`/`position_by_pass`/`accesses` 的行、hazard 两端及 cycle/blocked/name 诊断都保持原始 PassId，而不是临时 lowering 下标。`position_by_pass` 的值、queue batch 的 `pass_range` 则是编译后的执行位置，不能当作 PassId 使用。这不是第二次排序，也不是编译后逐字段修补。

`compile_with_inputs(&[Vec<u32>])` 与 `compile_with_sources(&[Vec<AccessInput>])` 的行、列分别对应原始 pass 与原声明 access ordinal；`AccessInput::Version(u32)` 指定既存版本，`ProducedBy { pass, access }` 指向另一原始 pass 对同资源的确切写访问。写输出只能由 checked builder 分配，Fabric 直接消费权威 `CompiledAccess`，不按执行次序或写次数另算 SSA。`ExecutionPlan::accesses` 保留被裁剪 pass 的行，行内是 **textures first、随后 buffers**，与 logical barrier 的 access ordinal 一致，不等同于混合资源声明的列顺序。

显式旧输入允许读 retained version，也允许以旧输入写入，但输入在 lowering 时必须已存在；旧版本不是额外物理资源，覆盖仍受 RAW/WAR/WAW 约束。shape mismatch、未来输入、非法 producer、版本容量与环均返回受检错误。可执行图的公开 `try_add_pass_with_dependencies` 保留显式控制依赖，并在运行 setup 前拒绝非法依赖。

`text/unit/compiler.rs` 的反向依赖、buffer/texture 混合、纯控制 fan-in、cycle 原始 ID、旧输入、容量与 symbolic source 回归覆盖上述契约；其中 `declaration_sources_match_numeric_authority_metadata_with_reverse_ids` 对照数值输入与 symbolic 输入的权威元数据，`executable_control_dependencies_preserve_edges_and_batch_waits` 检查控制边及 batch 等待。Fabric 当前路径是一轮轻量 `dependency_order()` 加一轮完整 `compile_with_sources()`，不再先完整编译空访问图、再以隐式写版本编译后重编显式输入图。

## Scene startup 与真实 executor 准入

`gpu/fabric.rs::init_scene_fabric` 读取**实际拥有 Scene 资源的 Bevy RenderDevice** 特性，以静态 builtin roots 和质量档冻结 DeviceFabric，不使用另建的 PrismGpu 设备替代。根不随 extracted/visible bodies 变化。

`shading/plugin.rs` 在 `RenderStartup` 注册 `init_shading_execution.after(init_scene_fabric)`，使 Commands 写入的 SceneFabric 在消费者前可见；`build_shading_execution` 调用 `shading_frame_graph().compile()`、校验实际 bindings，再创建 execution plan。water plugin 注册真实 `ExternalExecutionSite::WaterSurface`；缺失站点或冻结 variant 不匹配会拒绝启动，而非补空 executor。

当前 `shading/graph.rs` 是 8 个基础声明加 `post_graph.rs` 的 23 个 post 声明，共 31 个声明，不是旧 6-pass shading 切片。编译表中的 declared count/order/liveness 为权威，不另存固定 5/6 执行数量。

Core3d 的实际边界是：

`PreResolve → Resolve → WaterSurface → Response → Post`

pre-resolve/resolve driver 在 MainPass 前；water 的 Draw 位于既有外部 schedule；response driver 在 MainPass 与 WaterSurface Draw 后，post driver 在 response 后。driver 只投影编译顺序到这些 phase，不另外排序。

`shading/execution.rs` 检查：

- declaration/binding 精确对应，编译表长度、order、position 与 liveness 一致。
- 所有声明（含被裁剪项）的 `SelectedVariantRequirement { feature, variant }` 都向同一 frozen DeviceFabric 验证 root 准入与 variant 身份；仅在 dependency closure 中出现的服务不自动获得 root 执行许可。
- Systems group 有且仅有一个非空 owner；GroupMember 必须指向该 owner。
- group 不跨 phase（含被裁剪 group）、无部分存活、存活顺序连续且 owner 最先执行；拒绝 backwards phase 顺序及生产表中未分配 phase 的声明。
- WaterSurface 只能绑定对应声明与外部站点；driver 遍历外部项时直接跳过，由既有 water schedule 执行，不把登记或遍历记作 GPU 工作完成。

四个 driver 的范围来自**编译存活 order**的 phase 连续区间，不再依赖可能被裁剪的固定声明端点。pre-resolve driver 在执行任何范围前无条件 `begin_frame()` 清空 `executed_groups`，即使整个 pre phase 为空也重置；其余 driver 共享本次状态，owner 的 systems 只运行一次，member 先于 owner 会报错。该状态随 Core3d 的 view 执行重置，不充当所有 view 共享的曝光帧时钟。

回归 `four_drivers_reset_groups_and_preserve_the_external_boundary_across_frames`、`culling_old_range_endpoints_does_not_skip_surviving_phase_work`、`empty_pre_phase_still_resets_later_groups_every_frame` 覆盖跨帧跳过与裁剪端点缺陷；另有 phase 内 compiler 顺序、空 phase、跨 phase group 及外部站点错绑的拒绝测试。

`shading/plugin.rs::validate_post_execution_registration` 不只比对声明名或 debug 名：按位置核对系统数量及 `system.system_type()` 与真实 `boxed_post_executor` 的类型身份，debug names 关闭也能拒绝错绑。名称只用于诊断。该校验不宣称证明任意外来 compiled plan 的来源；调用方仍必须编译同一声明表。

## Response / Water 的资源边界

`shading/graph.rs` 中 resolve 写 left response、HDR，以及 SSR 使用的 env-specular/spec-weight。water 读取 resolve 后的 HDR/depth，写 left/right/controls/water-coverage。常驻 `response_composition` 在 water 后访问 response planes、visibility 与 coverage，写 left、auxiliary 及 `hdr_scene_color`，然后进入后处理。

因此 composition 不是“resolve 后只消费四个 imported planes”的旧模型，也不是先完成 composition 再让 water 生产 response。资源仍由相应 per-view owner 管理；声明图不伪造固定纹理分配大小。

`text/unit/shading/composition_ecs.rs` 有 3 项回归：两项 CPU capability gate 检查 workgroup 限制及 motion storage 的 feature/usage；一项 `resident_composition_ecs_prepares_executes_and_invalidates_targets` 在真实设备分支执行资源 prepare、resident composition、Bevy `PendingCommandBuffers` 原生提交与 HDR 像素读回，并检查重复消费、resize 与旧帧 publication 拒绝。fixture 请求 `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`，用生产纹理允许的 `RENDER_ATTACHMENT` 初始化 visibility，再通过 `TEXTURE_BINDING` 读取 HDR 到读回目标，不假设生产纹理具有 `COPY_DST`/`COPY_SRC`。

这 3 项已包含在主线最新 shading **886 passed / 0 failed** 中，不另加 3。真实设备用例在缺少 adapter、capability 或设备创建失败时默认可跳过，设置 `PRISM_REQUIRE_GPU=1` 才强制失败；因此 broad 通过计数本身不证明该 GPU 分支在严格模式执行，也不证明整帧 Fabric 接管。

## 曝光：两 pipeline 联合就绪与共享帧时钟

此前可复现的两个缺陷是：histogram 自身 pipeline/bind group 就绪即能累加，而 average 尚未就绪，导致 histogram 缺少同帧 reduce/clear；average 内的 `Local<Option<Instant>>` 按 view 调用推进，同一帧两个 view 可分别得到 16 ms 与 1 ms 的适应时间。

当前 `exposure/frame.rs::ExposurePassPair` 是 histogram 与 average 共用的录制门禁：曝光启用、extent 两维非零，且 **histogram pipeline、average pipeline、该 view 的 histogram bind group、average bind group 四项全部就绪**，任一缺失两 pass 均不录制。两处生产 pass 都查询这四项，而不是只在测试 helper 中检查。per-view `ViewExposureBuffers` 继续拥有真实持久 histogram/state buffer；build 累加、resolve 归约并清零，state 保留跨帧适应值。

实际调度保持 `response_composition → post_exposure → post_bloom`；`post_exposure` 的既有 Systems 顺序仍是 `exposure_histogram_pass → exposure_average_pass`，分别录制自己的 compute dispatch。此修复没有新增图声明、编译器或替代 recorder，也没有改变后续 SSR/GI/SSGI 的既有位置。

既有 `resources.rs::prepare_exposure_buffers` 在 `RenderSystems::PrepareResources` 每 render frame 运行一次，在 view 循环前推进 `ExposureFrameClock`，用 Commands 发布不可变的 `ExposureFrameTime` 快照，供 PrepareBindGroups/Render 之后的所有 view 共享。首帧使用 `1/60 s`，后续用单调时间差并钳制到 `[0, 1] s`；无 view、曝光关闭或 pipeline 等待时仍推进。average 只读该资源，不再按 view 维护或推进时间；沿用既有 plugin 注册即可。

`frame.rs` 的 7 项 CPU 回归覆盖四项 readiness 的 16 种组合、pipeline warmup、关闭/空 extent、同帧多 view 同 delta、无 dispatch 时持续推进、快照冻结及长/逆向时间钳制；`resources.rs::all_views_read_the_same_render_world_time_resource` 补充 ECS Resource 共享语义。该 7 项曾以独立 `rustc --test` 全部通过，相关模块现也纳入主线 broad shading 886 项通过结果。这些是录制准入与时钟语义证据，不作为曝光 GPU histogram 读回或图像质量证据。

## SSR → GI/denoise → SSGI → TAA：实际资源与调用

`POST_PROCESS_DECLARATIONS` 的相关顺序是 `post_ssr → post_spec_denoise → post_ssgi → post_taa`；稳定语义 ID 不因顺序变化重新编号。前置 exposure/bloom/DoF/motion-blur 仍按声明链运行。

| 生产者 → 消费者 | 精确资源与实际绑定证据 |
|---|---|
| `shading_resolve → post_ssr` | resolve 写 `post_ssr_env_specular / post_ssr_spec_weight`；`ssr/composite.rs` 绑定 visibility 的 env-specular/spec-weight 读视图；SSR 不冒充这两个资源的 writer |
| `post_ssr → spec_gi_reuse` | `spec_gi/bind_groups.rs::prepare_spec_gi_reuse_bind_groups` 实际绑定 `scene_depth_sampled / normal_roughness_view / ssr_out_view`，对应 `post_ssr_depth / post_ssr_normal_roughness / post_ssr_trace` |
| `post_ssr → spec_denoise` | `spec_denoise/bind_groups.rs` 绑定 `ssr_hit_view / normal_roughness_view / scene_depth_sampled`，含 `post_ssr_hit_distance`；这些是当帧输入，不是上一帧替代物 |
| GI/denoise → specular composite | reuse 读 prior reservoir、写 current reservoir 与 `post_spec_gi_resolved`；降噪产生 `post_spec_filtered`；`spec_gi/composite.rs` 优先读 filtered，否则读 resolved，并读 SSR `color_mip_view(0)`（`post_ssr_color_mips`）及 resolve 的 env-specular，写 scene_color |
| `post_spec_denoise → post_ssgi` | `ssgi/composite.rs` 先从 `visibility.scene_color_view()` 复制到 `post_ssgi_base`，再读 base、denoised GI、albedo/ambient 写 scene_color；复制必须晚于 specular composite |
| `post_ssgi → post_taa` | `taa/bind_groups.rs` 绑定最终 scene_color、`taa.read_view()`、`taa.write_view()` 与 motion vectors；TAA 消费 SSGI 后的当帧 HDR |

真实 executor 链由 `post_graph.rs` 列表、`plugin.rs::boxed_post_executor` 和生产 Systems bindings 共同确定：

- SSR：`ssr_prepass_pass → ssr_hzb_pass → ssr_repack_pass → ssr_color_mips_pass → ssr_trace_pass → ssr_reconstruct_pass → ssr_temporal_pass → ssr_composite_pass`。
- GI/denoise：`spec_gi_reuse_pass → spec_gi_spatial_pass → spec_denoise_reproject_pass → spec_denoise_history_clamp_pass → spec_denoise_spatial_pass → spec_gi_composite_pass`。
- SSGI：`ssgi_trace_pass → ssgi_denoise_pass → ssgi_composite_pass`；其后 `taa_resolve_pass`。

`validate_specular_resource_contract` 逐项检查 SSR 当帧写→GI/denoise 读、resolve exports 的真实 writer、HDR 消费及三条控制前置边；仅有顺序注释不能替代 access 声明。`graph.rs::specular_pipeline_graph_tracks_real_raw_provenance` 检查具体 RAW 资源/端点；`post_graph.rs` 的负例删除输入、伪造/删除 resolve writer、删除 producer-order 边时必须拒绝；plugin 回归覆盖缺失、重复、错序与错类型 executor。

prior/current reservoirs、specular/TAA history 是 Persistent；GI resolved、filtered、SSGI base 等帧内中间量是 Transient。per-view ping-pong 历史与 generic Fabric 的绝对帧导入是各自明确的所有权契约，不能只因同名“history”就认定已统一生产提交。

## PreparedGraph / RecordedGraph 与裸 allocation owner

当前 `prism_render_graph/src/graph.rs` 提供：

- `RenderGraph::prepare` 编译一次并冻结 ExecutionPlan，不创建 GPU 资源、不运行执行 callback；`PreparedGraph::record` 消费同一计划，不再编译、不 submit。编译过程仍可分配 CPU 计划数据，不能把此边界写成全程零分配。
- `RecordedGraph` 持有有序 commands、plan 与 `GraphAllocations`；`into_parts` 可转交外部提交方，`execute` 是 record 后单队列 submit 的便捷入口。
- allocation owner 必须保留到实际 GPU completion，再用创建资源的 device 释放；`release_transients` 保留 Persistent，`release` 释放全部自有资源，imports 不归它销毁。
- 裸 `RecordedGraph / GraphAllocations` 的 Drop 不销毁 GPU 资源，漏掉显式 release 会留至 device teardown；取消帧须先确保相关 commands 不会再提交。此裸接口与下述 completion-owned runtime 接口不同，不能把两者都描述成必须手工 release。

## runtime completion owner、代次与拒绝路径

`prism_render_runtime/src/lib.rs::execute_owned` 先 prepare，再在 pacer 锁内完成容量等待、record、completion value 分配与真实提交，不经假 queue 或再次编译；返回 `submission.rs::FrameSubmission`。`execute_tracked / execute` 复用该路径，分别返回完成值或 `()`。

- 真实 queue 的 completion callback 保留一个 owner clone；调用方提前 Drop 不会授权提前销毁。callback 标记完成，释放它持有的 clone；只有完成已确认且最后一个 owner Drop 时，才释放 graph-owned allocations。
- 显式 clone 共享资源 owner，可在 completion 之后保留 Transient 与 Persistent。发布 Persistent history 时，必须保留 owner 直到之后所有 imported GPU 使用均已完成；不会因一次 submit 自动发布 history，imports 也不由该 owner 销毁。
- owner 保留创建资源的 device `Arc`；`device_generation()` 表示创建代次，`completion_value()` 只在该代次内解释。runtime 恢复后，旧 owner 仍在旧设备回收，旧 ID 不可导入新设备，迟到旧 callback 不推进新代次完成状态。
- callback 被后端丢弃不等于 callback 已执行；shutdown wait 失败不授权销毁。未确认完成的自有资源留给创建设备 teardown，而不是对可能仍在使用的资源发出 destroy。

拒绝路径保持如下可重试边界：

| 路径 | 当前行为 |
|---|---|
| compile 失败 | 在触碰 pacer 前返回；不取得 retirement token、不创建 GPU 资源、不运行执行 callback、不提交 |
| completion value 耗尽 | `MAX - 1` 后允许一次 `u64::MAX`；再次提交返回 `CompletionValueExhausted`，在 dequeue/wait 前拒绝，不环绕、不复用、不推进值，也不创建 GPU 资源、record 或 submit |
| pacing wait 未确认 | `CompletionUnconfirmed`；最老 token 恢复队首，保持在途顺序/数量及提交值；不创建 GPU 资源、record 或 submit，重试先等同一 token |
| `wait_idle` 未确认 | 逐 token 等待后仍执行 queue-wide barrier；该 barrier 失败时，将未确认 token 按原顺序留供重试；barrier 成功则覆盖此前单 token 未确认状态 |

`next_completion_value()` 在 MAX 饱和，仅是规划提示；实际 execute 用受检加法拒绝耗尽。设备恢复以新的 generation 和完成值状态开始，不在原代次复用值。

`retirement_tests.rs` 的 13 项 controlled 测试覆盖 owner early drop/clone、runtime drop、callback 丢弃、失败 wait、恢复代次及溢出。其中 `failed_pacing_waits_restore_the_oldest_token_before_allocation_record_or_submit`、`failed_idle_wait_keeps_unconfirmed_tokens_in_original_order_for_retry`、`completion_values_never_wrap_or_reuse_the_final_value`、`compile_failure_has_no_retirement_token_allocation_recording_or_submission` 直接断言上述拒绝语义；受控后端计数与 token 顺序断言不充当 GPU 证据。

`submission_gpu_tests.rs` 的两项真实设备回归为 `execute_owned_gpu_clone_preserves_buffers_through_completion_and_readback` 和 `execute_owned_gpu_early_drop_completes_copies_and_retires_callback_resources`。同一 runtime/device 上，非零 payload 经外部 upload → graph-owned Transient → graph-owned Persistent → 外部 readback 的实际 transfer；检查 readback、真实 callback、clone 保活与 early drop 后回收。释放最后 owner 后，重分配复用旧槽且 generation 加一，验证 backend 表旧 ID 不再存活；不宣称 OS/driver heap 已归还。无硬件时默认可 skip，但 `PRISM_REQUIRE_GPU=1` 下必须失败；设备创建错误始终失败。early-drop GPU 用例不假设快速硬件尚未完成，完成前保活由 controlled 测试补充。

当前 Scene 证据对应 `SceneFabric` 冻结选择准入、`shading_frame_graph().compile()`、真实 BoxedSystem 与 Bevy 原生资源/提交路径。它不等同于同一 frozen selection 下 `DeviceFabric::compile → prepare_runtime → 真实 RHI 资源与 pass callbacks → Bevy 原生提交 → completion/history 回收` 的整帧集成证据；runtime 的独立 buffer GPU 验证也只证明其自身路径。FeatureDefinition/FeatureVariant/PassTemplate 仍是 Fabric 的单一声明权威，不能把专用 shading binding 表描述为另一套 Fabric 声明或完整替代编译链。

## 已执行检查点与证据限制

以下是资源迁移前已执行的检查点；不代表下节最终资源切片已重新构建或通过测试：

| 范围 | 已执行结果 |
|---|---|
| graph `--lib --all-features` | 最新重跑 78 passed，零 warning |
| architecture Render Fabric | 本轮 122 passed / 0 failed |
| `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_runtime --lib -- --test-threads=1` | 32 passed / 0 failed；含 13 项 controlled retirement、2 项真实 GPU 测试 |
| runtime 两项 GPU `--nocapture` 复跑 | 2 passed；adapter 证据 Apple M2 / Metal / Hardware |
| Scene broad shading（最新主线反馈） | **886 passed / 0 failed**，包含曝光 readiness/clock 回归及 composition ECS 3 项；不是 886 + 3 |
| Scene `specular_` / plugin / Apple M2 startup | 较早检查点分别为 14 / 13 / 1 通过 |
| `cargo check -p prism_render_app --all-targets` 与 owner caller 回归 | check 已通过，有既存 warnings；修复 3 处新增 `RecordedGraph` unused 后，`mesh_draw_records_into_a_graph_pass` 1 项及 `records_a_multi_draw` 2 项均通过 |
| gpudriven 严格 GPU `--lib -- --nocapture --test-threads=1` | 18 passed；7 条路径输出 Apple M2 / Metal 硬件证据 |

Graph/Fabric/runtime/Scene/app/gpudriven 结果按上述范围记录，不推断全 workspace 通过或其他范围零 warning；app 三项 caller 回归仅捕获命令、不提交 GPU，因此录制后立即 release，不计作 GPU 验证。Apple M2 startup 对应设备初始化、冻结选择与 executor 准入；runtime GPU 对应 transfer/readback/completion/owner 表项回收；gpudriven 的 18 项通过中记录了 7 条路径的硬件证据，不据此把全部测试都归为 GPU 路径。shading 886 是测试汇总，composition 的条件 GPU 分支与曝光 CPU/ECS 回归按各自语义解释；本轮没有据该汇总新增严格 GPU 运行结论。这些结果均不等于完整 Scene 帧 dispatch、通用 Fabric 整帧生产接线、图像质量或性能验收。旧“本机无 adapter”的环境陈述不再作为当前结论。

契约与推断限制：跨 graph 同范围 handle 的 graph identity 不在当前 handle 校验保证内；单操作 admission 不提供 closure 全事务；逻辑 alias/footprint 不证明 GPU heap 物理 alias；logical queue 不证明多队列 fence/submit；P50/P95/P99、零分配及图像 golden 均需对应实际测量，不能从编译器单测推导。

## Scene visibility 资源切片与本轮验证状态

本轮最后的生产改动限定在 Scene 资源接线，不扩展 driver 格式或通用 Fabric 编译链：

- `gpu/rhi.rs` 从同一 Bevy adapter/device/queue 建立 `SceneRhi`，保留原生提交权。buffer、texture、view 的 move-only lease 保留创建表；Drop 仅移除表项，不显式销毁仍由原生 clone/commands 持有的资源。`RenderStartup` 先建立该桥，再初始化 `SceneFabric`。
- `ViewVisibilityBuffer` 的 HDR、SSR 两个导出、motion、SSGI 两个导出及 world-ReSTIR 导出，共七张纹理经 RHI 分配，并把同一原生纹理交给已有 pass。等尺寸同表复用；尺寸或表变化重建；关闭、MSAA、缺失/零尺寸 viewport 与实体删除释放相应 owner。这里不提供通用 FrameLag 发布或 GPU completion 管理器。
- ID 与 metadata 两张纹理维持精确的原生 `Rgba32Uint` 分配和 shader ABI，不经 TextureCache。当前 RHI `TextureFormat` 没有该格式；这两张附件不是 RHI 资源，不能写成九张纹理都完成 RHI 迁移，也没有用其他格式替代。
- 新增资源复用、同尺寸多视图隔离、resize、无效 viewport、MSAA、开关、换表与销毁测试，并在 resident composition 测试中检查 HDR 的原生 allocation identity。bridge 定义九项测试，其中八项依赖 GPU；这些新增/修改测试没有取得本轮通过结果。

资源迁移后的最后一次 Cargo 尝试在编译阶段以 exit 101 结束：暴露了 RHI 缺少 `Rgba32Uint`、测试误用不存在的 RHI `MAP_READ`、以及 fixture 缺少 `Deref` 导入。当前源码已分别改为保留两张原生整数附件、使用已有 COPY_SRC 与原生 staging readback、补齐导入；这是静态修正，不是重新编译通过的证据。

2026-10-16 因磁盘空间告急，用户要求停止新的 Cargo 构建，仅收尾、白名单提交及轻量 Git 检查。因此上表的 shading 886 与 composition ECS 3 项仅属于迁移前检查点；当前 fixture 定义四项 ECS 测试，不记为四项通过。本轮格式化成功且工作区 `git diff --check` 通过；不据此推断类型检查、严格 GPU 或性能验收通过。

## 原始文档目标尚未闭环的验收项

1. 同一 frozen selection 的 Fabric compile → prepare_runtime → 真实 imports/recorders → Scene 原生整帧提交及 completion/history 回收。
2. 全域 FrameRenderItem/payload 提取、版本/barrier 消费，以及 FrameLag 分配、发布与导入。
3. G3/G4、七组全域装配、全部静态 roots 的生产消费者、NPR 同等基础能力及全部 32 门禁的可追溯运行证据。
4. 水体八项真实 GPU 输出、整帧 reference 等价，以及旧入口删除前护栏的历史运行证据；新增 GPU harness 源码不算通过结果。
5. 独立 ABI bump 与 package/cache 失效验证，以及同一 selection 到 shader package/PermutationSpace 的生产消费。
6. 本节最终资源切片的重新编译、四项 ECS 与 bridge GPU 回归，以及原目标的 P50/P95/P99、质量和整帧性能测量。

`docs/refactoring-dev/` 按用户要求不进入提交。已有 push 尝试因 HTTPS GitHub 凭据不可用失败，未通过改写 remote 或凭据配置绕过；本地 commit 与远端 push 的状态分别报告。上述原始要求没有被删除、降级或标记为全部完成。
