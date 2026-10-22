# Prism 材质重构文档 — 完成项 / 未完成项 核验报告

- 核验对象：`docs/prism_material_pipeline_refactor_zh.md`（v10，1772 行）
- 初始核验时间：2026-10-20；后续增量记录至 2026-10-22（各节按日期标注）；仓库 `/Users/wangkailong/Documents/GitHub/Prism`，核验基线为当时记录的 `HEAD`（工作树快照见 §2.2）
- 核验范围（用户口径）：**只核验文档中的完成项与未完成项**，不实现任何新功能
- 方法：只读代码核对（含 6 路并行只读子代理逐项对抗核验）+ 本轮真跑部分门禁
- 裁决档位：**成立** / **部分成立** / **不成立** / **未执行（无法判定）**

> 文档 §13.a 自称口径：*"只记已落地到代码并带测试门禁的项；实现以仓库代码为准，与本文冲突时以代码为准。"* 本报告按此口径执行，并把"代码存在"与"有门禁守护"分开判定。

## 0.1 2026-10-20 工作树增量（原静态核对；后续运行证据见 §0.2 / §2.1.1）

- **材质 VT 的 source 裁剪已补上实现**：`MaterialVtMaterialSourceQueries` 维护已发布材质实际引用的 image 集合；bootstrap demand 与 VT producer 只从该集合读取 source，避免未被材质引用的 image 抢占有限 page capacity。后续 scene lib 全量测试已编译此生产代码；完整 Material-VT GPU 生命周期与主 render pass readback 仍未验收。
- **P1-e GPU budget 已有生产消费者**：`LobeSampleBudgetGpu` 在 `PrepareResources` 上传 32-byte pass uniform，resolve 的贵瓣选择器同时受 dispatch ceiling 与上传的 `max_full_lobes` 限制；`multiscatter_dir_samples` / `multiscatter_march_samples` 也已进入天空多散射 LUT 的真实 GPU dispatch 参数。2026-10-21 复核确认 SSS separable taps、SSS preintegration steps、translucency profile samples 已驱动生产 direct SSS evaluator 内的动态积分循环，定向 CPU/WESL/真机 GPU 证据见 §0.5。完整 resolve 全帧 readback、各质量档画质/成本验收仍未闭合，不能把小型 wrapper 门禁等同于全场景生产验收。
- **编译阻塞已解除**：工作树曾在 Bevy bridge 移除后请求已删除的 `bevy` feature，并出现 scene 编译错误；后续材质提取 target **16/16** 与 scene lib target **1909/1909** 已通过。它们不替代 Material-VT/P1-e 的独立 GPU 门禁。

## 0.2 2026-10-20 后续复验

- 干净 `target` 后，`CARGO_INCREMENTAL=0 cargo --config profile.test.debug=0 test --message-format short -p prism_render_scene --lib --jobs 1 material::systems::tests -- --nocapture`：**16 passed / 0 failed**（全程重编 4 分 47 秒）。测试目标成功编译并执行；此前记录的 12 个 scene 编译错误已不再阻塞该定向 target。
- 首次复验暴露两处测试夹具把 `MaterialRegistry` 容量设为 0/1，违反 GPU scene allocator“slot 0 保留，至少另有一个 live slot”的构造约束。夹具改为合法容量，并通过预占唯一 live slot 保留容量失败/重试场景；回归与同模块 16 项一起通过。
- 同一编译还发现虚拟阴影 readback 在 `prism_math::Vec2` 与 Bevy `glam::Vec2` 边界间缺显式转换；转换已补齐。上述结果仅证明当前 `prism_render_scene` lib-test target 可编译及材质提取定向测试通过，不代表整个 workspace、GPU readback 或材质主路径图像验收完成。

## 0.3 2026-10-21 天空多散射预算 GPU 对拍

- 严格命令 `PRISM_REQUIRE_GPU=1 CARGO_INCREMENTAL=0 cargo --config profile.test.debug=0 test -p prism_render_scene --test sky_multiscatter_budget_gpu --jobs 1 -- --nocapture` 在 Apple M2 / Metal 上 **1 passed / 0 failed**。测试直接编译生产 `sky_multiscatter_lut.wesl`，对 High 与 0.25 缩放预算各执行 4×4 LUT，并逐 texel 与 CPU `bake_multiscatter_lut` reference 比较；RGBA16F readback 容差为 `max(0.003, 2.5% * abs(reference))`。
- 扩展覆盖发现并修正 GPU LUT 的 U/V 轴次序错误：生产 shader 原先把高度写在 X、太阳余弦写在 Y，与 CPU/采样契约的 U=太阳余弦、V=高度相反。多 texel GPU/CPU 对拍现已通过；证据限于该 LUT kernel 和两组采样预算，不代表全场景天空生产 pass、SSS/透射 GPU budget 或全帧画质/成本验收完成。

## 0.4 2026-10-21 P1-a Material-VT publication 原子性切片

- portable residency 构建遇到某个已引用 image 尚无 texture slot 时，现按 source 跳过，不再让整个批次提前返回；该 source 不进入 active publication，material row 继续保留 authored texture index/sampler fallback。回归同时断言其他已有 slot 的 source 仍能产生 page record、atlas upload 与 publication identity。
- page-table active slice 获得 completion proof 后，生产路径在同帧重新发布引用受影响 image 的 material rows，使 row 中的 VT source revision/geometry 与 active publication 一起可见；失败的 row 会重新记录 source change，等待后续 extraction 重试。
- 定向结果：`material::vt_producer::tests` **8 passed / 0 failed**，`material::` **91 passed / 0 failed**，`material_virtual_texture_wesl` **1 passed / 0 failed**。严格 GPU `material_vt_gpu_parity` 为 **0 passed / 3 failed**：当前运行环境找不到可用 graphics adapter，三项均在 adapter 请求阶段失败，未发生 dispatch/readback；因此本切片不声明 GPU 或跨设备完成。

## 0.5 2026-10-21 P1-e direct SSS 预算生产切片

### 生产消费与 layout 审计

- 当前工作树的生产接线已存在，本切片仅补充门禁并修正文档，不重复改写生产算法。`subsurface.wesl::subsurface_direct` 通过 `lb_uploaded()` 读取 `LobeSampleBudgetGpu`；`brdf.wesl::principled_direct_channels_with` 也读取同一 uniform，生产 `shading_resolve.wesl` 的 principled 与独立 SSS 路径均调用这些 evaluator。
- 三字段不是只读回 uniform 或改一个输出倍率：`brdf_sss_budget_quality` 的 separable 积分执行 `sss_separable_taps` 次循环；`brdf_sss_preintegrated_channel` 按 `sss_preintegration_steps` 对每个 RGB 通道积分；`brdf_sss_profile_energy` 按 `translucency_profile_samples` 积分，结果进入背光透射的归一化。门禁逐函数锁定 runtime count、循环上限及预算传参，GPU 则实际执行这些生产函数并检查输出变化。本轮测试分别将 High 的 **13 / 160 / 48** 独立降为 **3 / 16 / 8**，其他预算保持不变。
- **group 7 binding 9 与全部当前实际消费该预算的生产 pipeline layout 相容**：resolve 的 `area_light_layout_entries()` 是固定十项，binding 9 为 COMPUTE-visible、无动态偏移的 uniform；host/WESL 均为 8×u32 = 32 bytes。group 7 的顺序绑定第十项为 `lobe_budget.buffer()`，dispatch 固定绑定 group 0..7。初始化的 bindless/non-bindless 两种 variant、registered-response 热重链、area-light/shade-cache 启用与 disabled fallback 共用同一八组 layout。预算更新写入稳定 buffer，不因质量变化换 allocation 或新增 pipeline permutation。
- 全部生产 WESL/WGSL 调用点搜索表明：当前读取该 uniform 的路径只有上述 resolve。water raster、cloth、hair、clearcoat 等导入公共 BRDF 类型/helpers，但不调用读取预算的 integrator，不需给这些无关 layout 强加 binding 9；天空 LUT 使用自己的 dispatch 参数。此结论不等于“所有 renderer pipeline 都有 group 7 binding 9”，也不等于全部 resolve PSO 已在所有设备上完成硬件创建/dispatch。
- 本切片的范围是 **direct SSS 的 angular/separable quadrature、Penner preintegration quadrature、translucency radial-energy normalization**；不是新增屏幕空间 SSS 图像卷积 pass，也不是独立 preintegration LUT 预烘焙。

### 本轮实际运行结果

- 低调试信息、单 job、关闭增量缓存编译当前 scene lib-test 成功。首次普通运行的两个 GPU 用例因沙盒无 adapter 而跳过，不算 GPU 通过；随后严格模式在沙盒外成功访问 **Apple M2 / Metal / IntegratedGpu**。
- 严格 `shading::lobe_budget::`：**10 passed / 0 failed**，包含 CPU ABI/policy 3 项、CPU 三字段行为 1 项、生产调用链/循环及 layout variant 守护 2 项、WESL helper/生产 SSS 链接 2 项、真实 GPU dispatch/readback 2 项。没有用 skip 冒充严格通过。
- SSS GPU 门禁直接链接生产 `brdf.wesl + subsurface.wesl`，对四组预算分别执行前光与背光求值，24 个 RGB 输出通道均与 CPU `evaluate_subsurface_direct_with_budget` 对拍，绝对误差门限 `2e-4`。分别降低 taps、steps、profile samples 的红通道差约为 `4.55e-4`、`3.48e-5`、`0.1181`，分别超过 `1e-5`、`1e-5`、`1e-3` 的变化门限。preintegration fixture 选在 terminator 附近以显露积分误差，未放宽断言阈值。
- 贵瓣选择 GPU 门禁同时通过：High 与 0.5 缩放预算使真实 selector 的 full evaluations 从 **2 降至 1**，并改变 readback 的 lobe mixture；它不是 GPU timing 测量。
- `resolve_auxiliary_layout_matches_shader_bindings` 与 `resolve_shader_and_dispatch_stay_within_eight_groups`：**2 passed / 0 failed**，覆盖实际 host binding 类型/顺序、WESL 声明、dispatch 绑定与八组上限；新增 variant 门禁锁定初始化与热重链复用相同 descriptor。
- `shading_resolve_wesl_compiles_and_resolves_all_imports`：**1 passed / 0 failed**，实际包含普通 bindless/non-bindless、空 registered table、有限 registered-response 的 bindless/non-bindless 链接。该测试验证完整生产 WESL 链接，不是完整 resolve GPU dispatch。

本轮使用的命令（后三条复用第一条生成的测试二进制，避免重复编译共享 target）：

```sh
CARGO_INCREMENTAL=0 cargo --config profile.test.debug=0 test -p prism_render_scene --lib --jobs 1 shading::lobe_budget::shader_tests:: -- --nocapture
PRISM_REQUIRE_GPU=1 target/debug/deps/prism_render_scene-52519585a36c04ae shading::lobe_budget:: --nocapture
target/debug/deps/prism_render_scene-52519585a36c04ae shading::resolve::pipeline::layout_tests:: --nocapture
target/debug/deps/prism_render_scene-52519585a36c04ae shading_resolve_wesl_compiles_and_resolves_all_imports --nocapture
```

二进制 hash 随构建变化；重建复跑可将第二条改为 `PRISM_REQUIRE_GPU=1 CARGO_INCREMENTAL=0 cargo --config profile.test.debug=0 test -p prism_render_scene --lib --jobs 1 shading::lobe_budget:: -- --nocapture`。严格 GPU 命令必须在能访问物理 GPU 的环境执行。未清理共享 target、未全仓格式化、未提交；本切片写集仅 `shading/lobe_budget/shader_tests.rs` 与本文档。

**剩余边界**：本切片关闭“三个 SSS/透射字段没有生产 GPU 消费与定向数值门禁”的缺口，但不将整个 P1-e 标为完成。完整 `shading_resolve` 全帧 dispatch/readback、各质量档图像误差与 GPU timing/成本回归、跨设备/后端仍未验收；§0.3 天空 LUT 的历史真机结果未在本轮重跑。

## 0.6 2026-10-21 材质 schema / ABI / authoring 干净构建复验

- 当前工作树下 `cargo test -p prism_material_schema --lib --jobs 1`：**32 passed / 0 failed**。
- `cargo run -p prism_material_schema --bin prism-material-codegen -- check`：**4 个生成产品均最新**。
- `cargo test -p prism_render_material --test material_header_abi --jobs 1`：**4 passed / 0 failed**，覆盖 Rust/WESL header 镜像、schema 动态字段顺序与 ABI8 execution binding。
- `cargo test -p prism_render_material_authoring --all-targets --jobs 1`：**39 passed / 0 failed**（archetypes 2、evaluator 8、MaterialX 27、procedural 2；lib target 0 项）。该结果只证明当前已实现的受限 MaterialX/nodedef/procedural 子集；unsupported 节点仍明确 fail-closed，不表示完整节点集、完整 procedural bake 或编辑器 authoring 已完成。
- `cargo test -p prism_render_material --all-targets --jobs 1` 未进入测试执行：lib-test harness 链接时 LLVM 报 `No space left on device`。这是环境容量失败，不是测试断言失败；因此 schema parity 等其他 integration targets 在本次未验收。
- `prism_render_scene` 材质 VT、resolve、response composition、TAA、生产 GI 材质命中与完整全帧 readback **本轮未运行/未验收**。局部 ABI/schema 成功不能上推为实时生产帧闭环；path-traced reference 也仍是独立 oracle，不能替代生产闭环。
- 为释放本仓库构建空间执行了 `cargo clean`，最多一次成功回收 **4.3 GiB**；未删除源码。最终 `git diff --check` 通过。

## 0.7 2026-10-21 Material-VT GPU 门禁迁移状态复核

- 本轮尝试执行 `material_vt_gpu_parity` 时，Cargo 明确提示该 integration target 需要 `prism_render_scene/legacy-bevy`；不带该 feature 的命令未进入编译或测试。该 target 仍使用 Bevy `RenderDevice`、`RenderQueue`、`MaterialBindGroup` 与 `MaterialTextureArrays` 搭建测试环境，因此当前它不是 Bevy-free Prism driver 的生产路径门禁。
- 随后以 `--features legacy-bevy` 在物理 GPU 环境启动严格复验；干净构建在编译 `bevy_pbr` 时因 `No space left on device` 中止，未生成测试二进制，也未发生 GPU adapter 请求、dispatch 或 readback。构建结束后确认共享 target 无活动 Cargo/rustc 使用者，按项目要求执行 `cargo clean`，回收 **196.9 MiB**。
- **结论**：既有 Material-VT GPU parity 结果仍仅是旧 adapter 下的局部 shader/布局门禁；本轮没有新增 GPU 通过证据。将该验证迁移到 Bevy-free 生产 Material-VT bindings/driver，并连接实际 publication 与主 resolve，是仍需完成的 P0 工作，不能通过继续启用 `legacy-bevy` 宣称收口。

## 0.8 2026-10-21 默认材质 ABI GPU 门禁的帧目标阻塞

- 严格执行 `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_app --lib material::tests::the_abi_lands_byte_exact_on_a_gpu -- --nocapture` 时，物理 GPU 上首次发现 lighting composite 的 HDR 输出纹理缺少 `STORAGE_BINDING`；`RenderTarget` 创建 usage 已补齐该用途，并在 `target::tests::composite_gbuffer_planes_have_contract_formats_and_unique_resources` 中加入契约断言，定向测试 **1 passed / 0 failed**。
- 同一严格 GPU 测试再次推进到帧 command encoder 后，暴露更早的生产帧门禁：颜色附件总量 **45 bytes/sample** 超过设备限制 **32**。测试在材质 header/parameter heap GPU readback 前失败，故本轮**不构成材质 ABI GPU 通过证据**。
- 后续需先压缩/重组默认 mesh GBuffer 的生产 attachment layout（并保持 surface identity、coverage、线性 albedo 与 world-position 的语义和精度），再拆分不必要的全平面 clear attachment 集合；随后重跑严格 GPU readback 和主帧 golden parity。不能仅以 null-device usage 断言宣称 production GPU 闭环完成。

## 0.9 2026-10-22 默认 mesh GBuffer/composite GPU 边界门禁

- 修正 `lighting_gpu.rs::physical_gpu_surface_encode_gbuffer_readback_and_composite_close_the_contract` 的 sentinel 边界期望：第 4 个像素的 `surface_id == u32::MAX` 是无效 sentinel，production composite 应输出透明零值；高位但非 sentinel 的 `0xf123_4567` 仍是有效覆盖像素，输出黑色且 alpha=1。原测试把 index 4 也要求为 alpha=1，和生产 shader 的 sentinel 语义矛盾。
- 首次沙盒内严格运行因该环境无 GPU adapter 而无法验证；随后跨沙盒 Apple Metal 严格运行同一集成门禁 **1 passed / 0 failed**。测试真实覆盖默认 `surface_encode` draw、GBuffer readback、lighting composite 与边界像素，不是仅编译或 shader link-test。
- 此门禁关闭的是默认 surface encode/GBuffer/composite 的当前设备切片；它仍不覆盖 canonical 注册 response、材质纹理进入默认完整 resolve 的全部语义、全帧 CPU golden parity 或 Metal 外设备，故 **P0-material 仍未完成**。不得将此结果表述为完整默认材质着色路径验收。

## 0. 2026-10-15 增量复核

本报告原始结论保留为 2026-10-14 历史快照；本次重构增量已单独完成并验证：

- `water` 定向单元测试：670 passed。
- `render_fabric` 定向测试：73 passed，覆盖 provider 唯一性、硬需求闭包、变体写入契约、静态根集合与确定性选择。
- `prism_render_scene --test water_hybrid_golden`：**15 passed / 0 failed**；CPU/WESL 门禁、`MaskBlend` 组合、overlay、motion vector、reactive mask、四个入口及真实 GPU gate 全部通过。
- 跨沙盒真实 GPU 门禁：response composition 在 Apple M2 Metal adapter 上 **2 passed / 0 failed**，1320 cases / 14520 auxiliary jobs readback 通过；水面 raster 同一真实 adapter/device 上 **15 passed / 0 failed**。当前水面 raster 使用 7 个 bind groups，满足该设备 `max_bind_groups=8`，`PRISM_REQUIRE_GPU=1` 明确报告 `PASS`。
- `ResponseProgram` 现在只包含 PBR/NPR/Custom；Hybrid 由 `WaterShadingPlan::MaskBlend` 表达。共享服务由 `FeatureCatalog` 闭包推导，旧的子系统级共享位掩码不再是事实源。

---

## 1. 总体结论（7 条）

1. **文档主体可信。** 大部分 ✅ 项与代码相符：P1-b、P1-c、P1-d（引擎侧）、P1-f2、P1-h、P2-c（两个开放瓣）、P2-f、P2-g、P2-j 均逐条找到落地代码与门禁。
2. **初始文档快照有 4 处低估实现**：P2-e 区域 illumination、P3-b 入射辐照度语义、P1-g S1–S4、coat_darkening 均已在 §3/§5 裁决并由后续状态更正；它们不应继续进入当前未完成清单。
3. **MaterialX nodedef**：解析、导出与结构往返测试已有；未完成的是 nodedef ↔ `author_node` 的语义映射与 PCM 实例，而非基础 nodedef 往返。
4. **P2-h 原理化分层 BSDF 已有局部 CPU/GPU/reference 对照证据**：coat/base 能量耦合闭式、零反照率再分配项，以及 `coat_darkening` 对 reflective base 与 transmission substrate 的统一消费均已落地。path-traced reference 已接入 `CoatLayer`、clearcoat conductor/diffuse 与 `Bsdf` 分发；2026-10-20 定向 `reference_pt::clearcoat` **13 passed**。受限 Lambert fixture 有 CPU parity；§0.8 另记录 production `shading_resolve` GPU readback 与 clearcoat reference 对拍 **1 passed**。这些结果不等于完整材质组合或 path-traced 场景图像 parity；粗糙 diffuse 等未支持输入仍由 adapter fail-closed。
5. **计数类偏差普遍存在**（都是历史快照，不影响结论、但影响可信度）：门禁数字、header 字数、烘表常量数、PRISM 占槽数与目录行数、`bake::tests` 项数。
6. **历史快照有过编号/状态冲突及过期失败记录**；当前勘误保留在 §5，现行状态以 §3、§4、§7 为准。
7. **当前工作树的生成物门禁已恢复绿色**：`prism-material-codegen write` 已重新生成 `surface_layout.rs` 与 `material_unpack.wesl`，`codegen_freshness` 全部通过；剩余设备能力失败仅属于需要真实 GPU 特性的独立门禁。

---

## 2. 本轮真跑门禁 vs 文档快照

### 2.1 实跑结果

下表以本轮最终复验为准；旧数字保留在“文档快照”列，只用于说明历史漂移。

| 包 / target | 文档快照 | 本轮实测 | 结论 |
|---|---|---|---|
| `prism_material_schema` | 19 | **20 passed** | ⚠️ 快照过时 |
| `prism_material_pcm` | 28 | **28 passed** | ✅ 一致 |
| `prism_render_material` lib | 851 | **870 passed** | ⚠️ 快照过时 |
| `prism_render_material` `codegen_freshness` | 文档称绿 | **8 passed / 0 failed** | ✅ 当前绿色 |
| `prism_render_material` `material_header_abi` | 1 | **2 passed**（旧快照；当前源码 MIRRORS 已扩为 6 项） | ⚠️ 快照过时 |
| `prism_render_material` `pcm_parity` | 7 | **7 passed** | ✅ |
| `prism_render_material` `schema_parity` | 6 | **7 passed** | ⚠️ 快照过时 |
| `prism_render_material_authoring` | 23（18/3/2） | **23 passed（18/3/2）** | ✅ 一致 |
| `prism_render_app` lib | 114 | **124 passed** | ⚠️ 快照过时 |
| `prism_render_shading` lib | 3377 | **3379 passed / 6 ignored** | ✅ 一致 |
| `prism_render_scene` lib | 1602 | **1729 passed / 0 failed** | ⚠️ 快照过时 |
| response composition GPU | 旧快照未执行 | **2 passed / 0 failed**；1320 cases、14520 auxiliary jobs readback | ✅ Apple M2 Metal |
| water Hybrid GPU 能力门禁 | 旧快照未执行 | **15 passed / 0 failed**；Apple M2 Metal，7 groups ≤ `max_bind_groups=8` | ✅ 严格 GPU 通过 |

`prism_render_app` 另有 3 条无关 warning：unused import `Instance`、`PrepareContextError`、`prepare_extracted_assets_with_context`。

### 2.1.1 2026-10-20 当前工作树全量复验

在清理本仓库 `target`（移除 2.4 GiB 旧构建产物）后，执行：

```text
CARGO_INCREMENTAL=0 cargo --config profile.test.debug=0 test -p prism_render_scene --lib --jobs 1 -- --nocapture
```

结果：**1909 passed / 0 failed / 0 ignored**，目标 crate 从当前工作树完整编译并执行成功。该结果覆盖 `prism_render_scene` lib 单测，包括此前报告的 8 个失败项；它替代此前“scene 全量测试未复验/8 项失败”的历史状态，但不代表 GPU readback、全场景跨适配器 parity 或全 workspace 测试通过。构建仍产生 65 条 warning，须按来源分类处理，不能把测试通过等同于 warning-free。

同日 reference clearcoat 定向测试记录仍为 **13 passed / 0 failed**；它验证 path-traced reference 局部 BSDF，不是实时 GPU 与 reference 的图像级 parity。

较早的 2026-10-20 增量尝试曾因当时工作树编译错误而未执行材质提取 target，并记录过并发 build lock；该状态已由 §0.2 的材质提取 **16/16** 与 §2.1.1 的 scene lib **1909/1909** 后续复验取代。path-traced clearcoat 定向测试 **13/13** 亦在 §2.1.1 记载。旧尝试保留为历史，不再代表当前验收状态。

### 2.2 已修复的历史红门禁与当前状态

历史失败原文：

```
tests/codegen_freshness.rs:42:5
assertion left == right failed: surface_layout.rs is stale;
regenerate with `cargo run -p prism_material_schema --bin prism-material-codegen -- write`
```

当时的差异只有 10 insertions / 12 deletions，把生成器输出的多行形态

```rust
pub const SURFACE_CACHE_AUX: [CacheAuxLayout; 1] = [
    CacheAuxLayout {
```

折叠成 `= [CacheAuxLayout {` 单行起手。修复生成器的长数组、长字符串、空映射数组和响应结构体排版策略后，重新运行 codegen `write`；当前 `codegen_freshness` 已为 **8 passed / 0 failed**。该历史红门禁已关闭，水体设备能力失败不是生成物问题。

---

## 3. 逐项裁决表

### 3.1 已完成项（✅ 成立）

| 项 | 裁决 | 关键证据 |
|---|---|---|
| **ABI v6** | ✅ | `MATERIAL_ABI_VERSION: u32 = 6`（`pkg/prism_render_material/src/record.rs:18`）；schema `abi_version = 6`（`pkg/prism_material_schema/schema/surface.toml:23`）；`pkg/prism_render_material/tests/schema_parity.rs:19` 断言二者相等 |
| **破坏性删除** | ✅ | 全仓 grep `MAX_MATERIAL_TEXTURES` / `MAX_CLOSURE_SLAB_DEPTH` / `MAX_LOBE` / `TooManyTextures` / `ClosureSlabTooDeep` **零定义**，仅剩 3 条历史注释（`pkg/prism_render_scene/src/material/buffers.rs:33`、`pkg/prism_render_material/src/tests.rs:83`、`:200`）；`ir.rs:75` 只余 `normalize`，无深度检查 |
| **P1-b header/分类对齐** | ✅ | 六处 WESL 镜像（含 `visibility_raster.wesl` 与生成的 `material_classification.wesl`）逐 (名,类型) 比对 + `rust.len()*4 == size_of::<GpuMaterialHeader>()`（`pkg/prism_render_material/tests/material_header_abi.rs:94-145`）；先前记录的第六镜像盲区已关闭，须以当前 `MIRRORS` 六项为准 |
| **P1-c 删硬顶 + 纹理变长行堆** | ✅ | 闭包 slab 深度与纹理槽上限全删；`buffers.rs:46` `texture_heap: ParameterHeap`、`:114` `texture_heap.alloc`、`:160` `header(offset, texture_offset, epoch)`；WESL `shading_resolve.wesl:2317` |
| **P1-d 置换治理（引擎侧）** | ✅ | `pipeline_cache.rs:289` `MAX_IN_FLIGHT_COMPILES = 4`、`:472` 背压循环；`mesh_pipeline.rs:70` `full_lobe_fallback`、`:204/:250` 调用；`systems.rs:73` `pump_pipeline_compiles`、`plugin.rs:202` 挂在 `PrepareResources`；门禁 `tests.rs:987`、`:1256` |
| **P1-f2 OpenPBR 默认值 8 项** | ✅ | 四处一致：schema `surface.toml:40/63/318/331/405/631/239/255` ↔ `record.rs:412/416/430-434/441/413-414` ↔ `surface.rs:109-110/155-163/244` ↔ `lighting.rs:236/240/253/259/268/275-277`；emission 桥接漏乘回归测试 `resolve.rs:871`（断言 `[4,2,1]`，`:888-894`） |
| **P1-h PCM 地基（九类 kind）** | ✅ | `hooks.rs:173` `hooks()` 覆盖九类 `Kind`；`Stage`（`:48-53`）推导 Twin 义务（`:68-73`）；`header_resident()`（`:193`）= lobe/channel/response_mode/cache_aux；identity-only 走 `ExtensionManifest::new_identity_only`（`manifest.rs:280`）且 `registry_slot() == None`（`:367`）；`EnergyPolicy` + `roll_up`（`:170-213`）+ `EnergyPolicyOutsideLobe`（`:498-503`）；`promote` 对 identity-only 原地重写（`registry.rs:224`，回归 `tests.rs:215`） |
| **P2-c 开放瓣全链（glint / measured BRDF）** | ✅ | 瓣位 `surface.rs:429` GLINT=1<<17、`:432` MEASURED_BRDF=1<<18、`:465` `LobeMask::COUNT = 10`；glint 用 `4π·sin²(hw/2)`（`glint.rs:203-215`，注释明写取代 `1−cos`）与 WESL `glint.wesl:120-129` 同；measured BRDF Rusinkiewicz + sqrt 轴 warp + 三线性（`measured_brdf.rs:538/588-590/1010`）；`bin_class` 已扩到 19 bit（`axis.rs:162-166`，`BIN_CLASS_BITS = 29` @ `:90`） |
| **P2-d thin_film** | ✅ | `lighting.rs:180-187/275-277` 默认 0/0/1.3；权重 0 逐位恒等（`lighting.rs:443-455` / `brdf.wesl:862-873`）；µm→nm 双线（`resolve.rs:324-326` ↔ `shading_resolve.wesl:2359-2361`） |
| **P2-f 超集三条约束 + 能量** | ✅ | 独立 oracle 与 `lighting.rs`/`clearcoat.rs` 零共享（`openpbr_reference.rs:1-18`，仅 import core/bevy_math `:37-39`）；`project_to_openpbr`（`pkg/prism_render_material/src/openpbr.rs:118`）；位区隔离门禁 `surface.rs:1011`、`tests/schema_parity.rs:297`；fuzz 层 `cloth.rs:99-100` 基底乘 `1 − sheen·E_fuzz`；subsurface 闭式 `subsurface.rs:119-120` `e_wrap = (1+2w)/(1+w)²` |
| **P2-g 运行时动态参数通道** | ✅ | `record.rs:211-217` 四字段；`to_words` 非有限→0 其余钳 `[0,1]`（`:269-276`、`:298-304`）；`set_dynamic_channels` 幂等且唯一不走 publish（`registry.rs:144-166`），`publish` 重置 NONE（`:68-70`）；调制序 `dynamic.rs:113-166`，`a*(1-t)+b*t`（`:196-198`）；dissolve 覆盖度被 `opaque.wesl:202` 与 `transparent.wesl:188` 消费 |
| **P2-i 全 kind manifest + codegen** | ✅（编辑器类除外） | codegen 产出 `SURFACE_LOBES`/`SURFACE_LOBE_CLOSURES`/`SURFACE_CHANNELS`/`SURFACE_RESPONSE_MODES`/`SURFACE_CACHE_AUX`/`SURFACE_IDENTITY_EXTENSIONS` 六表 + 5 个 Layout（`codegen.rs:490/524/569/590/616/646/668/695/721/751`）；8 个有实例的 kind 全部在 `surface.toml` 声明；槽位碰撞机检 `prism_material_schema/src/lib.rs:683/769/784/807`（`DuplicateHeaderSlot`） |
| **P2-j response-mode 分类注册化** | ✅ | `classification.rs:453` `from_registry` 从注册表派生类集合与顺序；无实现即报错 `UnhandledResponseMode`（`:177/223/423`）；穷举平价门禁 `:1206`；WESL `CLASS_*` 一一对应门禁 `:995-1046`；`class_count() = max(id)+1`（`:509`） |
| **P2-a 作者前端首片** | ✅（首片） | `ir.rs:97` `Document`；`parse.rs:215-233` 17 类白名单；`import.rs:146` `adapt_openpbr_surface` + `AdapterReport`（`:20-30`）；`evaluate_connection`（`lib.rs:25`）、运算符 `evaluator.rs:119-185`、环检测 `:51-57`、维度/有限值 `:214-266/:227-233`、不支持节点 → `UnsafeConnection`（`:178-184`）；八类 archetype `archetype.rs:14-44`；测试 **23 = 18/3/2**，本轮实跑全绿 |
| **P0-a 多散射 GGX + furnace CI** | ✅（计数需修订） | 五张 const 烘表（`eon_albedo_table.rs` / `ggx_albedo_table.rs` / `dielectric_albedo_table.rs` / `schlick_bias_table.rs` / `charlie_albedo_table.rs`）+ WESL 镜像（`brdf.wesl:464/477/483/546/646/659`、`cloth.wesl:34`）；白炉门禁 `openpbr_reference.rs:940` 断言 `worst ≤ 0.02` |
| **P1-a 第一片（解析式纹理 LOD）** | ✅（余量见 §4） | Rust `uv_gradients`（`texture_lod.rs:78`）/`texture_lod`（`:124`）/`triangle_lod`（`:157`）；WESL `prism_uv_gradients`（`material_sample.wesl:329`）/`prism_texture_lod`（`:368`）/`prism_lod_in_range`（`:313`）/`sample_texture_slot`（`:228`）；LOD 真传入 `shading_resolve.wesl:2329-2330`，且 lod 进入 VT 页请求（`material_sample.wesl:240`）；SDF 显式 mip0（`:1544`）、SSR 显式 0.0（`ssr_repack.wesl:388`，理由见 `:379-388`）；退化 `MIN_SCREEN_AREA = 1e-6`（`texture_lod.rs:51/94`）；眼后顶点 `projected_*.z > 0.0`（`shading_resolve.wesl:2289`）；非有限用 `FINITE_RANGE = 1e37` 范围比较（`texture_lod.rs:63/67`） |
| **P1-c part 2 分层混合 bake** | ✅ | `bake.rs`：`sanitize_weight:71`、`bake_layer_stack:254`、`LayerParams::from_record:377`、`LayerStack:501`、`PageBake:478`、`bake_texel:598`、`bake_page:614`；权重 `coverage * mask`（`:580`）；单层逐位恒等门禁 `:666` 与 `:1002`；f64 累加 `:593`、零分配 `:600`；release fail-fast（`assert_eq!` 非 debug_assert）`:258`，门禁 `:791` |

### 3.2 部分成立

| 项 | 文档声明 | 代码裁决 | 缺口 |
|---|---|---|---|
| **P2-e 四线正交** | 部分（缺 ①逐区域 illumination ②作者侧 `respond_custom` WESL） | **① 已落地（文档错）**；② 确实缺失 | ①：`material_classification.wesl:87` `material_class(header, region_illumination)`、`:104-106` 覆盖规则、`:152` 消费 `metadata.z`；CPU `classification.rs:686/716-722`；来源 `visibility_raster.wesl:127` `region_illumination_selector`（扫描 `SEMANTIC_ILLUMINATION_MASK` 纹理行，`:142-171`）+ `:47` `FEATURE_REGION_ILLUMINATION`；作者 API `record.rs:106`、`authoring.rs:82`；`raster.rs:179` 发 shader def；门禁 `classification.rs:788`、`:927`、`authoring.rs:154`。语义限制：选择子仅在 `header.illumination ∈ {Lit,Stylized}` 时生效，`Unlit`/`Custom` 忽略区域遮罩（`:716-722`）<br>②：全仓 `respond_custom` 仅 1 处注释（`prism_render_shading/src/classification.rs:145`），无 WESL；`custom_program` 只是 header u32 标记（`record.rs:158`、`material_classification.wesl:114`） |
| **P2-h 原理化分层 BSDF** | 未做（路线图表无 ✅） | **实时与 reference 局部验收成立；已补受限 adapter + CPU image parity fixture** | 实时侧：闭式 `t_slab=(1−f_v)(1−f_l)/(1−f_v·f_l)`、`r_slab`、基底乘 `coat_attenuation = 1 + coat·(coat_t_slab−1)`、零反照率再分配项 `coat·(r_slab−A)·n·l/π`，以及 `coat_darkening = lerp(1,Δ,C·δ)` 对 reflective base 与 transmission substrate 的统一消费（CPU `lighting.rs:765-926`，WESL `brdf.wesl:1269-1386`，CPU/WESL 结构门禁 `lighting.rs:2070` / `shading/resolve/shader_tests.rs:223`）。reference 侧 `reference_pt/coat.rs` darkening 核、conductor/diffuse substrate 消费及 `Bsdf` 参数贯通由定向测试覆盖；`reference_pt::clearcoat` 结果为 **13 passed, 0 failed**。新增 `prism_render_shading::reference_pt_adapter`：只接受已证明的 Lambert diffuse substrate，并直接映射到 `reference_pt::Bsdf::ClearcoatDiffuse`；对不可表示或尚未严格对拍的 diffuse roughness、metallic/specular/transmission/SSS/sheen/thin-film/glint/coat-frame/非白 coat-color 显式拒绝。4×4 固定方向像素 fixture 覆盖 coat off、coat on + darkening 0/1，`cargo test -p prism_render_shading reference_pt_adapter --lib` **2 passed, 0 failed**，fixture `max_abs < 3e-5`。这仍是 CPU 生产 evaluator ↔ reference BSDF 的小图对拍，不是实时 GPU readback，也不是完整 path-traced 场景图像 parity；粗糙 diffuse 的严格 parity 尚未闭合。 |
| **P1-g 自描述反射瓣 schema** | ✅（实现与门禁源码存在；本轮未复跑） | **S1–S4 已实现；当前工作树 Cargo 验证状态未复核** | `material_classification.wesl` 的 schema metadata block 由 `emit_material_classification` 生成；`generated_contract.rs` 提供 schema 派生的 `SpecializationId` 布局/位打包和 CPU sample/golden field directory；`codegen_freshness.rs` 覆盖生成物新鲜度、ID 边界/往返、schema mutation 与 sample/golden 索引。不得由其他 crate 的 check 推导这些 target 本轮通过 |
| **P0-b 高光抗锯齿** | P0 | **主路径接线 + 生产 WESL 小图 GPU/readback 已成立；完整验收仍有边界** | 消费端：`gi/specular_aa/normal_variance.rs:124` `filter_alpha_sq`、`:104` `kernel_roughness_sq = (2v).min(cap)`、`toksvig.rs`/`lean.rs`；主高光 `lighting.rs:408`；GPU 孪生 `specular_aa.wesl:81/106` + `brdf.wesl:836`。主 resolve 已从同一 normal-map 记录、同一 mip 和同一 UV footprint 生成方差，并由顺序门禁锁定 `reset < center_sample < normal_record < forward samples < variance < response`，且 `variance..response` 不允许再次 reset。新增 29×17 fixture 真实编译并 dispatch 生产 `specular_aa.wesl + brdf.wesl`，readback 493 像素 / 1972 通道与 CPU golden 对拍：Apple M2 / Metal 最差 34 ULP（门限 64）、最大绝对误差 `4.768371582e-6`；493/493 像素均有非零方差且 filtered alpha 增大，尾部 guard 未被覆盖。离线方差语义、texel 编码/解码及 CPU resolve fallback 已有实现与单测；仍缺完整 `shading_resolve` 全帧 dispatch/readback，以及 Metal 外后端验证。**例外**：SSR 分支已有独立方差来源 —— `ssr_repack.wesl:154` / `:445-447` |
| **P1-e 贵瓣采样缩放** | 路线图表无 ✅（完整采样验收仍未完成） | **GPU uniform、贵瓣选择器、天空 LUT 与 direct SSS 三项动态积分循环均有真实消费；定向 GPU 门禁已通过** | `PrepareResources` 上传 32-byte `LobeSampleBudgetGpu`，resolve 消费 `max_full_lobes`；天空 LUT 方向/步进预算见 §0.3。生产 `subsurface_direct` 与 principled SSS 读取同一预算，taps/steps/profile samples 分别改变真实 GPU 循环。§0.5 严格 M2/Metal 定向模块 **10 passed**（含两项 GPU dispatch/readback），实际 layout 门禁 **2 passed**、完整 resolve WESL 链接 **1 passed**。剩余：完整 resolve 全帧 readback、各质量档图像误差与 GPU timing/成本、跨设备验收 |
| **P2-a 作者前端（完整）** | 部分 | **首片成立，剩余未做** | 缺完整节点图求值/bake、作者侧 VT 产线、编辑器集成；noise/curve/blend 被白名单排除（`parse.rs:215-232`），evaluator 返回 `UnsafeConnection` |
| **P3-a 解耦着色地基** | 部分 | **地基与独立 slice 成立；生产闭环缺失** | 见 §4 与 §5 |

### 3.3 不成立 / 与代码相反

| 文档说法 | 裁决 | 事实 |
|---|---|---|
| doc:1041 "coat_darkening 仍因公式不确定、仓库 path-traced 参考亦未实现而搁置" | **不成立** | CPU + WESL 双线实现（`clearcoat.rs:117/131/450/459-460/475`、`clearcoat.wesl:122/133/378-379/387`）与门禁已存在；reference path 现已实现 substrate darkening，并由 `reference_pt::clearcoat` 定向测试 **13/13** 覆盖。仍未证明完整离线/实时图像级 parity。 |
| doc:1183/doc:1189 "逐区域 illumination 未完成" | **不成立** | 见 §3.2 P2-e ①，且 doc:1353 自证已实现 |
| doc:1187 P3-b "未开始" | **不成立（低估）** | `shade.wgsl` 写 `incoming = light_radiance * max(dot(n, l), 0)`，`gpu/shade.rs:1-4` 明写 "stores view-independent incoming irradiance, not a material response" → §14.3 的架构约束已在独立 slice 落地；缺的是真实 producer、多光源/GI/SSS/emission |
| doc:1166-1168 "`cargo test --lib -- --list` 因 `texel_cache::gpu::dispatch` 重复声明/重复导出失败" | **不成立（不可复现）** | `texel_cache/gpu/mod.rs:9` 只有一次 `pub mod dispatch;`；`texel_cache/mod.rs:22-24` 只用一次 `#[path = "gpu/tests.rs"] mod gpu_tests`；`gpu/mod.rs:8-17` 无 `tests` 声明；无重复 `pub use`。本轮 `-- --list` 正常输出 **1718 tests, 0 benchmarks** |
| doc:1091 "normal_variance 恒为 0.0" | **部分不成立** | 主路径成立，但 SSR 分支已有真实方差来源（`ssr_repack.wesl:154/445-447`） |
| doc:795 "bake::tests 12 项" | **计数不成立** | 实为 **13** 个 `#[test]` |
| doc:1144/1039 等"核心 12 words"、`SURFACE_LOBE_WORDS_BY_SLOT=[4,12,4,4,4,4,4]` | **过期** | 实际 `SURFACE_CORE_WORDS = 20`（`surface_layout.rs:11` ↔ `material_unpack.wesl:31`），`SURFACE_LOBE_WORDS_BY_SLOT = [8,16,4,8,16,16,4,0×9,4,4,4]`（19 项） |
| doc:787/801 header 字节数 | **自相矛盾且都不对** | 真实链：v4 20 word = 80 B → 删 `closure_graph_offset` 后 19 word = 76 B → +4 PCM = 92 B → +4 动态 = **27 word = 108 B**（`record.rs:135-218`，门禁按 `field_count × 4` 锁定） |
| doc:689 P1-h 计数 "lobe(8) / 10 槽 / 22 行 / response_mode 22-24" | **过期** | 实际 `SURFACE_LOBES = 10`（KHR 7 + PRISM 3）、PRISM 常驻槽 **16..26 = 11 槽**、目录 **23 行**、response_mode 槽 **23/24/25** |
| doc:765 "约 765 个 const"、"561 采样点"、"4 张对拍表" | **口径偏差** | 主表 5×153 = 765 加两行均值各 9 → **783**；对拍 `sample_grid()` = 33×17 + 17×9 = **714** 点；`CASES` 只有 4 表（**不含 EON**） |
| doc: "白炉覆盖规范 §Testing 全部配置" / §13.a 第七片 "7 配置" | **不成立 / 口径不同** | 实跑门禁 `openpbr_reference.rs:940` 为 `configs: [8]` + Subsurface + Cloth = **10**，且 `metal(0.1)` 被 **omit**（同文件注释仍称 coat/fuzz/subsurface "not covered yet"，属陈旧注释） |
| doc:1176 "今天仍是 bindless 2D + 显式 LOD" | **措辞过时** | VT 采样路径已接线（`material/bindings.rs:174/195` 绑定 group(1) binding 5..8；config 由 `material/vt_producer.rs:256` `enabled = !snapshot.is_empty()` 生产；`material_sample.wesl:264` `material_vt_resolve` 消费）。真正缺的只有"喂真实 VT 行 + header 改 `vt_page_offset/count`"：`MaterialVtTexture::texture_row`（`pkg/prism_render_material/src/texture_lod/material_vt.rs:481`）**无生产调用者**（仅 `prism_render_app/src/material/tests.rs:904`）→ 请求 invalid → 回落 bindless。doc:513 的表述比 doc:1176 准确 |
| doc: "P2-i nodedef 往返未做" | **部分不成立** | 结构级已存在：`parse.rs:33/69/97` 解析 nodedef、`export.rs:14/24` `export_nodedef`、往返测试 `tests/materialx.rs:41`。缺的是 **nodedef ↔ `author_node` 映射与实例** |
| doc: "致密性由 `validate()` 机检" | **归因不准** | `validate()` 只查碰撞（`DuplicateHeaderSlot`）；致密性由门禁承担：`pkg/prism_render_material/tests/pcm_parity.rs:437-456`（非瓣常驻槽致密）与 `:197-207`（瓣区致密） |
| doc: "今天无任何注册瓣声明能力需求"（若推广为全局） | **仅对 lobe 成立** | 10 个 lobe 的 `requires_caps` 全空 ✓；但非 lobe 行非空：wetness `["object_space_cache"]`（`surface.toml:1104`）、snow `["virtual_texture"]`（`:1117`）、surfel `["object_space_cache"]`（`:1198`）、measured_brdf `["heavy_sampling"]`（`:1224`） |
| 旧快照称 `visibility_raster.wesl` 未受 ABI 门禁保护 | **已修正** | 当前 `material_header_abi.rs` 的 `MIRRORS` 覆盖六个镜像并逐字段比较；§3.1 P1-b 记录当前门禁。此处只保留历史错误裁决，不作为当前缺口。 |

---

## 4. 确认仍未完成的项（未完成清单）

1. **P1-a 余量**：VT source 按实际材质引用裁剪；材质 row 与 VT publication 在 completion 后同帧重新发布；page-table/atlas publication 与 `shading_resolve` VT sampling 已接线。仍缺 non-bindless miss 回 authored texture，以及完整默认主 render pass 的图像级、跨设备门禁。现行 row ABI 以 `texture_id + 尺寸 + page_size` 寻址 VT，不应再把旧提案中的 `vt_page_offset/count` 当作当前字段。
2. **P0-b 剩余边界**：同一 normal-map 记录的 mip/UV 前向样本已接入 CPU resolve 与 WESL 主 resolve，并由 `spec_aa_screen_space_variance` 生产 `normal_variance`；CPU 行为、shader-source 顺序门禁以及生产 `specular_aa.wesl + brdf.wesl` 的 29×17 真机 GPU dispatch/readback 与 CPU image parity 均有记录。无有效前向足迹时使用的离线方差语义、texel 编码/解码、CPU fallback 与 bake helper 也已有代码和单测；这不等于资产烘焙/生产数据管线已验收。仍缺完整 `shading_resolve` 全帧 dispatch/readback，以及 Metal 外后端验证。
3. **P1-e 完整验收**：pass uniform、贵瓣选择上限、天空 LUT 与 direct SSS 的 taps/preintegration/profile 三项动态积分循环均已接入；§0.5 已补定向 SSS 与 selector 真机 GPU readback、CPU 对拍及 layout/WESL 门禁。仍缺完整 resolve 全帧 readback、所有质量档的图像误差与 GPU timing/成本回归、跨设备/后端验收；不能把局部生产函数 wrapper 对拍等同于完整帧闭环。
4. **P1-g 当前验证状态**：schema 生成 `material_classification.wesl`、`SpecializationId` 布局/位打包及 CPU sample/golden 的 codegen 与 freshness/parity 门禁源码均存在（见 §3、§7）；本报告没有在 2026-10-22 后复跑对应 Cargo targets，不能由 `prism_render_app` check 结果推定它们通过。当前没有证据表明其仍是待实现功能。
5. **P2-b 程序化节点 → VT 烘焙**：受限首片已实现：支持的 MaterialX procedural graph 可 lowering 到 runtime `MaterialGraph` 并烘焙为 `MaterialVtSourceSnapshot`，noise/curve/ramp/add/multiply/mix 等受限类别有对应实现与门禁。未完成的是完整 MaterialX 节点覆盖、nested graph 外部参数、方差图烘焙与编辑器集成；不得再把该项记为“完全未开始”。
6. **P2-a 剩余 / 编辑器前端**：完整节点图求值、作者侧 VT 产线、编辑器集成。
7. **P2-e ②**：作者侧 `respond_custom` WESL 逃生口不存在（`custom_program` 仅为 header 标记，运行期安全退化到 über）。
8. **P2-i 第 9 类**：`author_node` 无任何实例（`pkg/prism_material_pcm/src/builtin.rs:274` `PRISM_IDENTITY_ONLY` 仅 5 行，`:260` 注释说明故意缺席，测试 `src/tests.rs:607` 断言其行为空）；schema 无 `[[author_node]]`。缺 nodedef ↔ author_node 映射与 PCM 实例。
9. **P3-a 生产闭环**：独立 shade-cache 已有自动的可见性 demand→确定性 page allocation/chart provenance/writer-job 构建、GPU writer 与 `shading_resolve` group 8 consumer；不能再称“无 producer”或“无 resolve sampling”。仍缺从这些 job 到 active page publication 的完整生产生命周期闭环、默认主 render pass 端到端硬件 readback/parity，以及反馈驱动粗 mip、失效和时序复用门禁。Material-VT 与独立 shade-cache 是不同页表/资源，不可混为同一闭环。
10. **P3-b 生产化**：独立 slice 已存入射辐照度，但仅单方向光、无 GI/SSS/emission 接入生产 resolve。
11. **P3-c 失效摊销 + 时序复用 + mip LOD**：`coarsen`/`Stale` 原语在，全库无 temporal/reprojection 实现。
12. **P3-d 光谱（hero-wavelength）**：材质线完全没有；仅 hair 子系统 `prism_render_architecture/src/hair/spectral_absorption.rs:187` 有 `hero_wavelengths`（与 P3-d 无关）。
13. **P3-e 四线 × 自定义契约的 pass/TAA 接线**：数学核 + 孪生已落地，接线未做。
14. **P3-f 内存与压缩**：未开始。
15. **P2-f "全部配置"白炉**：`metal(0.1)` 仍被 omit，且门禁文件内注释陈旧。
16. **P1-b 验收重跑**：原核验快照曾发现 `visibility_raster.wesl` 未纳入 `MIRRORS`；当前六个 mirror 已纳入并逐字段检查，也覆盖 `material_classification.wesl`。剩余是 M0 恢复后重跑 header ABI target，不再列为实现缺口。
17. **历史生成物门禁**：曾因生成器排版差异失败；本轮已重新生成并由 `codegen_freshness` 的 **8 passed / 0 failed** 关闭该问题。

---

## 5. 已关闭的文档漂移（历史审计记录）

以下项目已由后续章节/状态表修正，不再是待办。本节保留历史裁决以说明审计口径变化；当前状态以 §0 增量记录、§3 裁决表、§4 剩余清单及 §7 优先级表为准。

### 5.1 已纠正的内容偏差

1. `doc:1041` 曾把 coat_darkening 写成"搁置"；该说法已过期。CPU/WESL 与 reference_pt 局部实现均有定向门禁；reference_pt clearcoat 定向测试为 13/13，且 `reference_pt_adapter` 有 4×4 CPU image parity fixture。完整实时 GPU readback 与离线场景图像 parity 仍需独立验收。
2. `doc:1183/1189` 曾把逐区域 illumination 写成"未完成"；该说法已过期。当前实现为 CPU + WESL + 作者 API + 门禁，和 `doc:1353` 一致。
3. `doc:1153/1187` 曾把 P3-b 写成"未开始"；该说法已过期。入射辐照度语义已落地于独立 slice，生产 producer 与主 pass 门禁仍缺。
4. `doc:795` `bake::tests` 12 → **13**。
5. `doc:787/801` header 字节链 → 统一为 `20w(80B) → 19w(76B) → +4 PCM(92B) → +4 动态(27w/108B)`。
6. "核心 12 words" / `SURFACE_LOBE_WORDS_BY_SLOT=[4,12,4,4,4,4,4]` → `20` / `[8,16,4,8,16,16,4,0×9,4,4,4]`。
7. `doc:689` 的 `lobe(8)/10 槽/22 行/response_mode 22-24` → `10 lobes/11 槽/23 行/23-25`。
8. `doc:765` "765 const / 561 点 / 4 表" → `783`（含两行均值）/`714`/4 表不含 EON。
9. "白炉覆盖全部配置"的概括已被纠正：当前门禁配置中 `metal(0.1)` 被 omit；`openpbr_reference.rs` 中陈旧注释属于源码维护事项，不是本次文档审计的待办，也不据此声称该项已修复。
10. `doc:1176` 的 VT 措辞按 `doc:513` 口径改写（采样路径已接线，仅缺数据）。
11. "nodedef 往返未做" → 限定为"nodedef ↔ author_node 映射与实例未落地"。
12. "致密性由 `validate()` 机检" → 归因到 `tests/pcm_parity.rs`。
13. "无任何注册瓣声明能力需求" → 限定为 lobe；非 lobe 有 4 处非空。
14. `texture_uv_gradients` 若被写成 Rust 符号 → 实为 `uv_gradients`（`prism_uv_gradients` 是 WESL 名）。
15. "normal_variance 恒为 0.0" → 补 SSR 分支例外（`ssr_repack.wesl:154/445-447`）。
16. P1-e“只有类型、无上传/消费” → 改为“类型、CPU 全链、GPU pass uniform、贵瓣 selector、天空 LUT 与 direct SSS 预算消费均已有；定向 CPU/WESL/真机 GPU 门禁见 §0.3/§0.5，完整帧画质与成本仍未验收”。
17. 旧快照记录 `pkg/prism_material_schema/src/lib.rs:18-20` 头注释曾写"当前 S1、nothing is generated yet"；这是源码注释的历史观察，不是当前实现状态的证据，也不构成本次文档限定范围内的修改项。
18. `doc:1166-1168` 的 `--list` 失败记录已不可复现；该历史裁决保留为审计轨迹，不作为当前构建阻塞或待办。

### 5.2 已消除或需按当前表理解的历史冲突

1. 路线图中 P3 子项编号曾在光谱/缓存工作间复用；§13.a 按各工作包定义解释，避免仅凭 P3-d/e 编号跨章节映射。
2. P1-g 的 S1–S4 实现与门禁源码已存在；当前未复跑与实现未完成是不同状态，见 §4.4 与优先级表。
3. P2-e 区域 illumination 已完成；余项是作者侧 `respond_custom` WESL，见 §3.2。
4. header ABI、furnace 配置计数和历史 `--list`/test-count 数字均按代码事实在 §3.3 标为过期或不成立；不应作为当前缺口。

### 5.3 P0-b 验收证据（历史记录；完整帧门禁仍开）

1. `production_resolve_derives_specular_aa_before_response_evaluation` 单独定位 center sample，并锁定 `reset < center_sample < normal_record`；reset 检查改为 `variance..response` 区间，避免把切片起点自身误判为 response 前二次 reset。
2. `specular_aa_image_gpu` 不是 uniform readback 或只读 link-test：它通过生产 WESL 前端编译 `specular_aa.wesl + brdf.wesl`，在真实 compute pass 中 dispatch，拷贝 storage output 到 MAP_READ buffer，并逐像素与 CPU production evaluator 对拍。
3. Apple M2 / Metal 结果：29×17 = 493 像素，1972 个输出通道；最差 34 ULP（门限 64），最大绝对误差 `4.768371582e-6`；493/493 像素产生非零 normal variance，493/493 像素的 filtered alpha 均大于原始高光 alpha，尾部 guard 保持不变。
4. 本证据只关闭生产 specular-AA/BRDF kernel 的小图 GPU parity，不等同于完整 `shading_resolve` 全帧 readback；后者与无 footprint 的离线方差 fallback 继续列为未闭合项。

---

## 6. 当前仍未执行 / 无法判定的项

1. **其余真机 GPU 对拍**（7 个 target 文件确实存在）：
   `pkg/prism_render_material_gpu/tests/{bc_parity, albedo_table_gpu_parity, dynamic_channels_gpu_parity, glint_gpu_parity, lobe_sampler_gpu_parity, measured_brdf_gpu_parity, texture_lod_gpu_parity}.rs`。
   response composition 已有跨沙盒 Metal readback 通过；上述 7 个材质 GPU target 本轮没有逐项执行，因此不把它们写成通过。门禁文件与 `require_gpu()`/skip 逻辑存在（`tests.rs:53-55`、`shade_tests.rs:36-38`）。
2. **性能数字**：`bench_const_table_vs_buffer_table` 的 5.561 / 8.255 ms、逐光源去重 7.397→5.767 ms（−22.0%）——函数存在（`albedo_table_gpu_parity.rs:374` 等），但数值需真机 GPU 复跑。
3. **GPU 侧语义**（shade-cache 写入是否为辐照度、bindless 采样、page-table 消费）依据源码与门禁代码判定，未在真机二次确认。

## 7. 材质重构执行优先级（2026-10-20 裁决）

本节把路线图中的 P0/P1/P2/P3 与当前代码审计得到的真实阻塞合并排序。它不是“已完成项”清单；只有对应的生产调用、运行结果和图像级门禁同时成立，才可以把一项从未闭合移出本节。表内“本轮/当前”限定为最后一条注明日期的证据，不把早期快照直接当作现状。

| 顺序 | 工作包 | 必须达到的闭环 | 排序理由 |
|---|---|---|---|
| **M0** | workspace 编译与验收基线 | `prism_render_app --lib --tests` 已于 2026-10-22 通过；仍需按材质范围复核 `prism_render_scene`、`prism_render_shading`、schema/codegen 与 GPU 定向 targets，并保留严格 GPU/skip 语义 | `prism_render_app` 的 check 关闭了此前测试模块缺失这一阻塞，但不能替代其他 crate/门禁的当前验证 |
| **P0-material** | 默认 Native mesh 路径贯通 canonical 材质求值 | 已有：默认 opaque queue 绑定 canonical header/parameter heap；surface encode 校验 index/generation/revision/execution-plan 并调用生成的 `prism_unpack_surface`；surface encode/GBuffer/composite 有真实 GPU 边界门禁。仍缺：Native 默认路径的 texture rows/texture heap 采样、registered response 与统一 `SurfaceData → ShadingPlan → response → composition` 消费；真实默认材质 draw 的完整 resolve/readback 与 CPU golden parity。Bevy GPU Scene 的 `shading_resolve` 已有 VT/registered-response 实现，但独立 opaque plugin 非默认自动入口，不能替代 Native 闭环。 |
| **P0-b** | 高光抗锯齿主路径 | normal-map 同记录、同 mip、同 UV footprint 的方差生产，CPU/WESL/GPU readback 一致；无有效 footprint 时有明确离线方差图输入；至少有图像级回归 | 直接影响高光稳定性，且现有消费端已存在，剩余是主路径生产和验收闭环 |
| **P1-a** | Material-VT 真实消费 | 只为实际引用材质供给 source；material row 与 VT publication 同一代际原子可见；non-bindless miss 回 authored texture；主 render pass 与跨设备门禁通过 | 它决定开放纹理数量、驻留容量和材质画质是否在真实场景中成立 |
| **P1-e** | GPU 贵瓣预算完整验收 | 在已通过的 SSS/透射动态循环与 selector/LUT 定向 GPU 门禁上，补完整 resolve 全帧 readback、各质量档图像误差、GPU timing/成本及跨设备回归 | 三项 SSS/透射 sample-count 已真实改变生产循环与输出，group 7 binding 9 对全部当前预算消费者的生产 resolve layout 相容；尚缺完整帧与质量/成本验收，见 §0.5 |
| **P1-g / ABI 护栏** | 当前源码具备 S1–S4 与六 mirror 门禁；在材质相关验证环境可用时重跑 codegen freshness、schema parity 和 header ABI targets | 生成产品与门禁源码存在；本轮未验证这些 Cargo targets 状态，不再沿用已关闭的粒子模块阻塞说法 | 防止 schema 生成链发生漂移；区分实现缺口与尚未复验 |
| **P2-h** | 原理化分层 BSDF | 实时 CPU/WESL、GPU resolve 和 `reference_pt` 使用同一组 OpenPBR 参数语义；coat/base/transmission darkening 做图像级 parity，而不只做局部公式测试 | 实时与 reference 局部实现均已存在；clearcoat reference 定向测试 13/13。剩余是 GPU resolve 到 reference 的图像级 parity，不是补写 reference BSDF |
| **P2-a/b/i** | 完整作者链 | MaterialX 图求值、procedural → VT、`author_node` 实例和 nodedef 往返接入同一 MaterialArtifact；不再把不支持节点静默降成默认材质 | 依赖 ABI、VT 和 über 消费稳定，提前做会反复改作者产物格式 |
| **P3-a~e** | 解耦着色与材质-GI | shade producer/publication、主 render pass、GI/SSS/emission 材质辐射求值、失效/时序复用及单瓣 pass/TAA 接线全部有生产消费者和图像门禁 | 这是性能上限和 GI 扩展，不应阻塞屏幕空间材质主链 |

两个明确裁决：

1. **über 分层消费有必要**。它是当前材质主链的运行时执行模型，必须作为 `SurfaceData → ShadingPlan → response → composition` 的唯一消费路径收口；不能用“有 path-traced reference”替代实时消费。
2. **path-traced reference 也有必要，但角色不同**。production material ABI 到 `reference_pt::Bsdf` 的 coat/darkening 对应实现已存在，并有 clearcoat 定向测试 **13/13**；它应继续作为独立 oracle / 参考图生成器，用于 P2-h 图像级 parity，不应成为实时渲染运行时依赖，也不应阻塞 M0、P0-material、P0-b、P1-a 的先行闭环。

因此，当前最合理的并行关系是：M0 走主线；先闭合 P0-material 的默认材质消费路径，再让 P0-b、P1-a、P1-e 并行收口实时质量、纹理驻留和预算；P2-h 可同步补 reference adapter 与图像门禁；P2-a/b/i 和 P3 暂不抢占主线资源。当前默认 mesh draw 已消费 canonical header/参数 ABI 并解包基础 surface，但 `ForwardLitProvider` / `ForwardLitMaterial` 仍是独立简化契约，不能代表注册 response 与 texture heap 已进入默认 resolve。P0-material 完成前，独立 shader、VT 或材质预算门禁仍按局部证据登记。

### 7.1 2026-10-22 当前工作树复核与 Material-VT 设备丢失回收增量（验证未闭合）

本轮只读复核确认 P1-b 的六个 header mirror 已全部列入 `MIRRORS`；P1-g 的 S3/S4 生成器、`generated_contract.rs` 和 freshness/sample/golden 门禁也已存在。此前 §3/§7 将两者继续列为缺失属于审计快照过期，已在表述中纠正。该更正尚未通过当前工作树 Cargo 重跑。另核实默认 mesh 已有 canonical header/参数堆绑定、generation/revision/execution-plan 校验和 surface 解包；P0-material 的剩余是注册 response/texture heap 到完整着色生产路径及端到端 readback，不是从零建立 ABI 绑定。

**当前复核更正（2026-10-22）**：前述 `pipeline_build/strip_tessellation.rs` 阻塞描述不成立——当前 `pipeline_build.rs` 没有该模块声明或引用，条带细分实现位于 `prism_render_gpudriven`。实际缺失且被引用的是 `gameplay_feedback/tests.rs` 声明的 `gameplay_feedback/gpu.rs`；它是物理 GPU 验收子模块，生产 readback 实现在父模块 `gameplay_feedback.rs`。因此当前尚未证明它会阻断生产 lib target，但会阻断包含该测试模块的 lib-test 编译；不得用空文件或删除声明来伪造通过。复核时可用磁盘约 2.4 GiB，系统拒绝进程清单查询，故没有启动会写入共享 target 的 Cargo 构建。待真实 GPU 门禁恢复后，再以定向编译确认 M0 的实际边界。

**2026-10-22 后续验证**：`CARGO_INCREMENTAL=0 cargo check -p prism_render_app --lib --tests --jobs 1` 已在当前工作树通过，证明前述 `gameplay_feedback/gpu.rs` 已被测试模块编译覆盖，M0 的该项缺失已关闭。严格物理 GPU ABI readback 首次发现测试只运行一帧，早于 completion-gated native-VT 首次 publication，因而读到新 GPU buffer 的零值；这不是 fallback 编码或 GPU upload 内容错误。测试现改为最多运行四个真实帧并等待生产 `native_vt_ready`，同时断言 fallback 与两条材质 header 均发生上传。跨沙盒 Apple M2/Metal 严格复验 **1 passed / 0 failed**，fallback header、参数 heap、纹理行和材质 ABI GPU bytes 对拍通过。该结果关闭默认材质 ABI buffer 的真机 readback 缺口，但不等于默认 `surface_encode → 完整 shading resolve → response/composition` 主帧 draw 的材质图像 golden parity；P0-material 仍未完成。沙盒首次链接因磁盘空间不足、沙盒 GPU 无 adapter 的失败均未作为代码失败或通过证据。

Material-VT GPU slice 新增单调设备放弃标记：已知 device loss、恢复中的资源移除，或 shutdown 无法证明 queue idle 时，两份 atlas/table/config/sampler 资源均不再向旧 device 发显式 destroy；晚到的 completion callback 只释放未确认工作计数，不会恢复资源可销毁状态。owner API、scheduler、resident-upload 和 binding submit 均拒绝已放弃 slice；健康设备且 `wait_idle` 成功时仍按 creator device 正常回收。新增 `native_vt_known_device_loss_abandons_both_idle_slices` 回归测试。

前述未编译结论仍有效，但其阻塞归因已由本节当前复核更正：`strip_tessellation.rs` 缺失不构成 `prism_render_app` 的阻塞；真正缺失的 `gameplay_feedback/gpu.rs` 位于 `#[cfg(test)]` 测试模块中，阻断 lib-test 而非生产 lib check。此前“lib check 因缺少 strip_tessellation 失败”的命令结论不得继续引用。Material-VT 设备丢失回收改动的最终状态仍未由 Cargo 编译；格式检查与 `git diff --check` 不替代 Cargo 成功验证。待真实 GPU 测试模块补齐且可用构建资源确认后，运行材质定向 targets，并重跑 `material::virtual_texture`、recovery 与 plugin 定向测试，再更新验证状态。

### 7.2 2026-10-22 材质主路径分叉与堆范围安全复核

当前 checkout 有两套不同的默认 opaque 消费入口，不得混称为一条已闭合的“默认材质路径”：

1. Native `prism_render_app` 默认 mesh 路径为 `PrismRenderPlugin → SurfaceEncodeProvider → surface_encode.wesl → 六附件 GBuffer → lighting_gpu composite`。`surface_encode` 顶点 ABI 含 position/normal/UV0，fragment 解包 canonical header/parameter heap，并扫描已绑定的 texture rows；portable Material-VT base-color 行可采样 atlas 后写入 GBuffer。Native composite 仍直接以 GBuffer albedo 做简化照明，没有从 canonical surface 恢复完整材质参数，也未执行 registered response 或 `ShadingPlan`。
2. Bevy GPU Scene 的 `visibility_raster → material_classification → shading_resolve → response_composition` 已有材质纹理采样、Material-VT 与 registered response shader/host 接线；但 `PrismGpuScenePlugin` 不自动安装独立的 `PrismGpuSceneOpaquePlugin`，且它不是 native `prism_render_app` 默认入口。因此该实现不能抵销 native 主路径的 P0-material。
3. P0-material 的系统级完成条件是选定并贯通真实生产入口，使 canonical `SurfaceData → ShadingPlan → texture/VT → response → composition` 在默认 draw 上运行，并建立该入口的 GPU readback/CPU golden。不能仅将 Bevy resolve 代码存在、surface ABI 绑定存在或局部 shader 测试记作闭环。

本轮另将 opaque、transparent、shading resolve、SSR repack 的材质参数堆范围检查统一为先验 `offset <= heap_len`，再检查 `count <= heap_len - offset`，避免 `u32 offset + count` 回绕后通过边界检查；材质纹理行扫描也改用有界迭代，避免 `offset + count` 回绕。源码契约回归覆盖 `material_parameter_heap_guards_cannot_wrap_u32_ranges` 与 `material_texture_row_scans_cannot_wrap_u32_ranges`。独立 `rustc --edition=2024 --test pkg/prism_render_scene/src/shading/resolve/opacity_shader_tests.rs` 复验结果为 **8 passed / 0 failed**，`git diff --check` 通过。此源码契约测试不执行 GPU，也不替代完整 WESL/Cargo 验证；后者因此前首次构建使共享磁盘空间降至约 1.3 GiB 而主动中止，随后 `cargo clean` 回收 3.8 GiB，仍需在可控构建空间下重跑。

### 7.3 2026-10-23 默认 surface UV ABI 切片

`ForwardLitProvider` 的默认 opaque surface ABI 现要求 location 2 为 `Float32x2` UV0；普通与变换顶点入口都将 UV0 传至 fragment stage。缺失、重复或格式不符会被明确拒绝，没有从 world position 合成 UV。定向命令 `cargo test -p prism_render_app --lib forward_lit::tests` 报告 **12 passed / 0 failed**，在隔离 target 目录完成；其中物理 GPU 用例支持无 adapter 时跳过，因此该总数本身不证明实际执行过 GPU dispatch。

此切片建立了稳定 UV 输入 ABI。其后当前 shader 已绑定 texture-row buffer，并能采样 portable Material-VT base-color；但尚无通用 bindless texture heap 消费，registered response 与 `ShadingPlan` 也未进入 native composite。故这些局部接线没有关闭 P0-material；默认主路径的完整材质消费和端到端 readback/CPU golden 仍是阻塞项。复合阶段曾新增的 `CompositeMaterialBufferBindings` 及未接入 active pipeline 的 layout 草案已移除，不计为实现或验证证据。

### 7.4 2026-10-23 Bindless generation-table ABI 修正与验证边界

`MaterialTextureSlots::build` 的 generation table 现按 heap 的 `high_water` 物化：reserved slots 使用固定 generation `1`，从未分配过的动态 slots 写入无效哨兵 `0`，已分配或回收过的 slots 使用 heap 保存的当前 generation。此前直接对 capacity 内每个 slot 调用 `generation_of` 会在首次未使用的动态 slot 上 panic；该问题由 `material_texture_generation_table_tracks_empty_and_recycled_slots` 首次运行复现，按上述语义修正后，重新编译的定向测试 **1 passed / 0 failed**。`git diff --check` 通过。

`install_runtime_tail` fixture 另有一处 family 选择错误：Unlit 声明只支持 split，Lit 才支持 fused；修正为按 response manifest 能力注册后，低调试信息/关闭增量缓存的 `cargo test -p prism_render_app --lib material::tests:: --jobs 1` **18 passed / 0 failed**。`the_material_binding_creates_on_a_gpu` 也加上严格能力断言，避免 adapter 存在但不支持 bindless 时静默返回。当前物理设备严格执行 `PRISM_REQUIRE_GPU=1 ... the_material_binding_creates_on_a_gpu` 时明确失败：adapter 不具备 `Features::BINDLESS`，因此 **没有完成 bindless layout/bind-group 的真实 GPU 验证**；这属于验证设备能力缺失，不是 GPU 通过，也不是 shader/layout 断言失败。

### 7.5 2026-10-23 Native 默认纹理消费复核

- 当前工作树复跑 `CARGO_INCREMENTAL=0 cargo --config profile.test.debug=0 test -p prism_render_app --lib material::tests:: --jobs 1 -- --nocapture`：**18 passed / 0 failed**。其中 GPU ABI 两项依赖设备能力；bindless 专项在不支持 `Features::BINDLESS` 的物理 adapter 上按严格门禁失败，不能计作 bindless GPU 通过。
- Native 默认 `surface_encode` 已有 UV0、canonical header/parameter 解包、texture-row buffer 绑定与 portable Material-VT base-color atlas 采样，并有 surface/GBuffer GPU 边界门禁；但这不是通用 bindless texture heap 消费，registered response/统一 response-composition 也未进入默认路径。故这些 18 项只验证材质存储、行分配及现有 bindless helper，不关闭 P0-material。
- 具备 non-bindless 采样能力的现成基础是 Native Material-VT portable atlas：它复用 texture row 的 portable 标记以及 page-table/config/atlas/sampler ABI；普通 non-VT 行在 non-bindless variant 中仍按语义返回 fallback。因此它可作为默认路径的纹理消费者基础，但**尚未**接入默认 mesh pipeline/bind groups/frame graph。接线还必须把 active VT slice lease 持有到真实帧提交完成，并声明 atlas sampled-texture 依赖；仅保存裸 GPU ID 不满足双 slice 覆写与设备恢复生命周期契约。
- 当前纵向闭环仍要求真实默认材质 draw 的完整 resolve/readback 和 CPU golden parity。Portable VT base-color 采样仅是必要子项，不等于 canonical `SurfaceData → ShadingPlan → response → composition` 全链完成。

最重要的集成边界没有变化：全仓 app 源码搜索显示 `MaterialBindingLayout`、`MaterialTextureSlots::build` 与 `create_group` 仍无生产调用点；Native 默认 `SurfaceBindingContract` 和 `CanonicalMaterialBindings` 已绑定 frame/material/header/parameter/texture rows，另有 VT group 1 供 portable atlas 采样。尚未接入的是通用 texture heap/generation-table 采样、registered response 及 `ShadingPlan → response → composition` 的实际默认 draw/resolve。故本切片仅关闭 generation table 容量内未使用槽导致 panic 的局部缺陷，**不关闭 P0-material**。测试过程中一次并行构建令可用空间降至 116 MiB；确认没有活动 Cargo/rustc 后执行 `cargo clean` 回收 **5.5 GiB**。随后单 job 低调试信息复验通过；当前 `target` 已由本轮重建及其他工作占用，清理前仍须重新确认无活动构建。

### 7.6 2026-10-23 Native execution-table mirror 与 composite 接线审计

- 当前工作树新增 `NativeExecutionTableBuffers`，能镜像 registry 的 plan/program rows，并以 queue completion callback 发布版本；plugin、PrepareAssets 与 device recovery 已管理该资源。它目前仍**没有被 `LightingGpuState`/`LightingGpuPass`、composite bind group 或 frame graph 消费**，所以只是 GPU mirror，不是 `ShadingPlan` 生产执行接线。
- Native composite 当前仍只读 GBuffer 的 position/normal/albedo/surface id 与 GI，并以 albedo 做简化 direct lighting。继续闭合 P0-material 还需将已完成上传的 plan/program、canonical headers 与 parameter heap 作为只读 storage bindings传入 composite，在 frame graph 声明读依赖，并按真实 GPU material header 校验 plan/program tuple；缺少/过期/越界状态须 fail-closed。
- `sync_registry` 在表版本变化时使用 `queue.wait_idle()` 后重写/扩容 buffer；这能避免覆盖仍被旧帧读取的 storage，但会在材质表更新时引入队列级同步。接入前需替换为 completion-token 保护的版本化 buffer/退休策略，或以实测证明更新频率和停顿成本可接受；不能把当前等待方案当作最终性能闭环。
- registered response 不能直接复用 scene resolve 的 evaluator/linker：它依赖 scene resolve 提供的完整 surface、视图和光照上下文。可共享 `ResponseOutput` 通道/组合语义；Native 必须先定义其可提供的响应上下文，并对暂不支持的能力明确拒绝，不能复制 context-dependent evaluator 或静默降为 albedo-only。
- **不得把 direct-only Lit 或“只绑定执行表”当作完整主链切片。** 当前 `prism_render_scene` 的 resolve/opaque host 依赖 Bevy RenderApp（legacy-bevy），而 `prism_render_app` 是 Bevy-free 的 Native RHI 主机；直接调用 scene evaluator 不会把它变成 Native 默认入口。完整实现要么将可复用的材质求值/响应契约下沉到后端无关层并为 Native 实现完整 pass host，要么在 Native renderer 中完整实现等价的 resolve；两者都必须覆盖已声明的 builtin、registered、mask blend、纹理/VT 与响应组合，不能缩窄为简化 albedo 着色。
- shader 侧也存在明确的迁移边界：现有 `shading_resolve.wesl` 导入 `prism_render_scene::shaders::*` 的 surface/BRDF/lighting/geometry 等模块，并直接导入 `bevy_render::maths::affine3_to_square`；Native `prism_render_app/build.rs` 当前只链接 `prism_render_material` 的 material/material_unpack/material_virtual_texture 与 `prism_surface_encode`。因此要复用完整 resolver，必须把完整的 shader dependency closure 与其 host ABI 一起移出 Bevy 专属包/路径，不能仅复制入口 shader 或只改一条 import。
- **生命周期必须按整份材质快照闭合，而不是只给 execution tables 做 COW。** `on_submitted_work_done` 不能保护已经录制但尚未提交的 graph；graph 必须 pin 实际绑定的不可变快照，覆盖 header、parameter heap、texture rows、execution tables 及对应 VT publication/lease。快照退休后，仅在最后 graph/提交 lease 释放且 GPU completion 可证时，才由原 device 所有者回收；普通更新不得 `wait_idle`，也不得覆写仍被读的版本。否则 draw 与 resolve 仍可能跨代读取。
- 本轮定向镜像测试因共享构建过程中 `target/debug/deps` 创建临时目录时报 `No space left on device` 而未能运行；这不是断言失败。随后 `target` 已不存在，磁盘剩余约 3.7 GiB；因无法证明完整依赖重建在当前空间可完成，本轮不重启 Cargo build。`git diff --check` 对上述材质/render 文件通过。Native composite 资源接线、无停顿版本化生命周期和完整默认 draw parity 均仍未完成。
- **续接状态**：之后工作区 `target` 被其他任务重新建立；本次观察到约 916 MiB 构建产物、磁盘可用约 1.9 GiB，且有多个活动 Cargo/rustc 进程使用共享与隔离 target。故本次不清理 `target`、不启动竞争构建；该环境状态不改变上条测试失败原因，也不构成测试通过证据。
