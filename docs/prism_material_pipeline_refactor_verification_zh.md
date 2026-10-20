# Prism 材质重构文档 — 完成项 / 未完成项 核验报告

- 核验对象：`docs/prism_material_pipeline_refactor_zh.md`（v10，1772 行）
- 核验时间：2026-10-20；仓库 `/Users/wangkailong/Documents/GitHub/Prism`，本轮基线为当前 `HEAD`（当前工作树改动见 §2.2）
- 核验范围（用户口径）：**只核验文档中的完成项与未完成项**，不实现任何新功能
- 方法：只读代码核对（含 6 路并行只读子代理逐项对抗核验）+ 本轮真跑部分门禁
- 裁决档位：**成立** / **部分成立** / **不成立** / **未执行（无法判定）**

> 文档 §13.a 自称口径：*"只记已落地到代码并带测试门禁的项；实现以仓库代码为准，与本文冲突时以代码为准。"* 本报告按此口径执行，并把"代码存在"与"有门禁守护"分开判定。

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
2. **有 4 处"文档说未完成 / 搁置，代码其实已实现"**，属文档低估：
   - **P2-e 逐区域 illumination**：文档 doc:1183/doc:1189 列为未完成，实际 CPU + WESL 全链路已落地（文档自己在 doc:1353 的"状态修订"里又描述了它已实现 → 自相矛盾）。
   - **P3-b 的"只存入射辐照度"语义**：文档 doc:1153/doc:1187 列为"未开始"，但独立 shade-cache slice 已按 §14.3 架构约束实现该语义。
   - **P1-g**：路线图表无 ✅，但 S1+S2（schema 数据化 + codegen 生成物）已落地。
   - **coat_darkening**：doc:1041 记"仍因公式不确定而搁置"，实际 CPU + WESL 双线实现、双消费点、两个门禁，源码无占位标记。
3. **有 1 处"文档说未做、实际已有结构级实现"**：nodedef 往返（`prism_render_material_authoring` 已有解析 + 导出 + 往返测试）；真正缺的是 nodedef ↔ `author_node` 映射与实例。
4. **P2-h 原理化分层 BSDF 已通过局部验收**：coat/base 能量耦合闭式、零反照率再分配项，以及 `coat_darkening` 对 reflective base 与 transmission substrate 的统一消费均已落地。path-traced reference 已接入 `CoatLayer`、clearcoat conductor/diffuse 与 `Bsdf` 分发；2026-10-20 定向 `cargo test -p prism_render_architecture --lib reference_pt::clearcoat -- --nocapture` **13 passed, 0 failed**，覆盖两种 substrate darkening 和 coat lobe 保持不变。该结果证明局部参考 BSDF，不等于完整离线渲染/实时 GPU 跨路径图像 parity。
5. **计数类偏差普遍存在**（都是历史快照，不影响结论、但影响可信度）：门禁数字、header 字数、烘表常量数、PRISM 占槽数与目录行数、`bake::tests` 项数。
6. **文档内部自相矛盾 6 处**（编号冲突、状态冲突、已失效的失败记录），详见 §6.2。
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
| `prism_render_material` `material_header_abi` | 1 | **2 passed** | ⚠️ 快照过时 |
| `prism_render_material` `pcm_parity` | 7 | **7 passed** | ✅ |
| `prism_render_material` `schema_parity` | 6 | **7 passed** | ⚠️ 快照过时 |
| `prism_render_material_authoring` | 23（18/3/2） | **23 passed（18/3/2）** | ✅ 一致 |
| `prism_render_app` lib | 114 | **124 passed** | ⚠️ 快照过时 |
| `prism_render_shading` lib | 3377 | **3379 passed / 6 ignored** | ✅ 一致 |
| `prism_render_scene` lib | 1602 | **1729 passed / 0 failed** | ⚠️ 快照过时 |
| response composition GPU | 旧快照未执行 | **2 passed / 0 failed**；1320 cases、14520 auxiliary jobs readback | ✅ Apple M2 Metal |
| water Hybrid GPU 能力门禁 | 旧快照未执行 | **15 passed / 0 failed**；Apple M2 Metal，7 groups ≤ `max_bind_groups=8` | ✅ 严格 GPU 通过 |

`prism_render_app` 另有 3 条无关 warning：unused import `Instance`、`PrepareContextError`、`prepare_extracted_assets_with_context`。

2026-10-20 增量复验：`cargo test -p prism_render_architecture --lib reference_pt::clearcoat -- --nocapture` **13 passed / 0 failed**，覆盖 path-traced clearcoat conductor/diffuse substrate darkening 与 coat lobe 隔离。`cargo test -p prism_render_scene --lib material::systems::tests -- --nocapture` 未能编译到测试执行阶段：该次复验所见工作树中 `prism_render_scene` 有 12 个编译错误，涉及 composition 的 `CommandBuffer` 导入、water/spray 的 Fabric API 可见性与类型不匹配、area-light 对私有 `resolve` 模块的访问等；因此材料提取新增测试仍未验收。为避免并发 Cargo 使用期间删共享产物，本轮清理后又有并行 Cargo 持有 build lock，暂不继续清理或重跑。

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
| **P1-b header/分类对齐** | ✅（门禁有 1 处盲区，见 §6.1） | 5 处 WESL 镜像逐 (名,类型) 比对 + `rust.len()*4 == size_of::<GpuMaterialHeader>()`（`pkg/prism_render_material/tests/material_header_abi.rs:94-125`）；`is_active` 改名已统一（WGSL 保留字 `active`） |
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
| **P2-h 原理化分层 BSDF** | 未做（路线图表无 ✅） | **实时与 reference 局部验收已成立** | 实时侧：闭式 `t_slab=(1−f_v)(1−f_l)/(1−f_v·f_l)`、`r_slab`、基底乘 `coat_attenuation = 1 + coat·(coat_t_slab−1)`、零反照率再分配项 `coat·(r_slab−A)·n·l/π`，以及 `coat_darkening = lerp(1,Δ,C·δ)` 对 reflective base 与 transmission substrate 的统一消费（CPU `lighting.rs:765-926`，WESL `brdf.wesl:1269-1386`，CPU/WESL 结构门禁 `lighting.rs:2070` / `shading/resolve/shader_tests.rs:223`）。reference 侧 `reference_pt/coat.rs` darkening 核、conductor/diffuse substrate 消费及 `Bsdf` 参数贯通由定向测试覆盖；`reference_pt::clearcoat` 结果为 **13 passed, 0 failed**。coat 自身高光不消费该衰减；尚无完整实时 GPU 与 path-traced 图像级 parity 证据。 |
| **P1-g 自描述反射瓣 schema** | 路线图表无 ✅；§17.5.8 S2 标 ✅ | **S1+S2 已落地；S3/S4 未做** | 已做：`Field.name`（`prism_material_schema/src/lib.rs:286`）、字足迹由 `Field.ty` 派生（`:288-289`）、`registry_slot`、per-field `spec_subfield="closure_mask"`（`surface.toml:304`）、`closure` → `SURFACE_LOBE_CLOSURES`（`surface_layout.rs:190`，消费于 `classification.rs:486`）；`material_unpack.wesl` 与 `surface_layout.rs` 由 codegen 生成<br>未做：`material_classification.wesl` 仍手写；`SpecializationId` 位打包仍手写（`axis.rs:103`）；CPU golden/sample 未生成 |
| **P0-b 高光抗锯齿** | P0 | **消费端成立，生产者缺失** | 消费端：`gi/specular_aa/normal_variance.rs:124` `filter_alpha_sq`、`:104` `kernel_roughness_sq = (2v).min(cap)`、`toksvig.rs`/`lean.rs`；主高光 `lighting.rs:408`；GPU 孪生 `specular_aa.wesl:81/106` + `brdf.wesl:836`<br>缺口：`normal_variance` 主路径生产者缺失（字段 `lighting.rs:171`，赋 0 于 `lighting.rs:274`、`resolve.rs:319`、`cloth_advanced.rs:374`；`screen_space_variance` 无生产消费者）。**例外**：SSR 分支已真实喂方差 —— `ssr_repack.wesl:154` `spec_aa_filter_roughness` 由 view-normal prepass 屏幕差分算方差并写 `normal_roughness`（`:445-447`） |
| **P1-e 贵瓣采样缩放** | 路线图表无 ✅（§13.a 自述 pass uniform 未接线） | **类型 + CPU 全链已落地；GPU 侧未落地** | 已做：`prism_render_shading/src/quality.rs:109` `LobeSampleBudget` + `ShadingQuality`；CPU 真消费 `gi/subsurface/preintegrated.rs:211/213`、`separable.rs:197/208`、`sky_lut/multiscatter_lut.rs:264`、`translucency/diffusion.rs:278`、`lobe_sampling.rs:145/189`；生产设置画质档 `scene/src/shading/runtime.rs:9`（默认 High `:244`）、`shading/resolve/dispatch.rs:93`<br>缺口：`shaders/lobe_budget.wesl` 只被测试引用（`shading/lobe_budget/mod.rs` 只有 `#[cfg(test)] mod shader_tests;`）；`quality.rs:98` 写"Uploaded to the GPU as a plain pass uniform"但无生产写入者；PCM 只有 UID 登记（`builtin.rs:88`） |
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
| doc: "第 6 个镜像受门禁保护" | **不成立** | `visibility_raster.wesl:24-35` 有完整 27 字段 `MaterialHeader`（顺序与 Rust 逐字一致，27/27），绑定于 `:57` `@group(3) @binding(0)`，但**不在** `material_header_abi.rs:94-100` 的 `MIRRORS` 5 项内 → 无守护 |

---

## 4. 确认仍未完成的项（未完成清单）

1. **P1-a 余量**：纹理段全面切到 VT 页引用 —— 缺"喂真实 VT 行 + header 改 `vt_page_offset/count`"（`material_vt.rs:481` 无生产调用者）。
2. **P0-b 主路径方差生产者**：`normal_variance` 的生产侧（纹理足迹/烘焙方差图）未接线；消费端（Toksvig/LEAN）已实现。
3. **P1-e GPU 侧**：`lobe_budget.wesl` 无生产写入者，贵瓣 shader 未消费采样预算（CPU 全链已落地）。
4. **P1-g S3/S4**：`material_classification.wesl` 与 `SpecializationId` 位打包仍未由 schema 生成；CPU golden/sample 未 codegen。
5. **P2-b 程序化节点 → VT 烘焙**：完全未开始。`bake.rs:254/614` 只是任意深度 layer stack 烘焙；无 noise/curve/blend 图求值。
6. **P2-a 剩余 / 编辑器前端**：完整节点图求值、作者侧 VT 产线、编辑器集成。
7. **P2-e ②**：作者侧 `respond_custom` WESL 逃生口不存在（`custom_program` 仅为 header 标记，运行期安全退化到 über）。
8. **P2-i 第 9 类**：`author_node` 无任何实例（`pkg/prism_material_pcm/src/builtin.rs:274` `PRISM_IDENTITY_ONLY` 仅 5 行，`:260` 注释说明故意缺席，测试 `src/tests.rs:607` 断言其行为空）；schema 无 `[[author_node]]`。缺 nodedef ↔ author_node 映射与 PCM 实例。
9. **P3-a 生产闭环**：① `ShadeDispatchInputs` 只有 init/extract（`plugin.rs:263/376`）**无 producer**；② `begin_shade_writes_from_registry` 仅测试调用，`ShadeDemandRegistry` 不自动提交 GPU 工作；③ **无主 render pass 端到端硬件门禁**（现有为 group-1 生产绑定 + test-owned wrapper）。
10. **P3-b 生产化**：独立 slice 已存入射辐照度，但仅单方向光、无 GI/SSS/emission 接入生产 resolve。
11. **P3-c 失效摊销 + 时序复用 + mip LOD**：`coarsen`/`Stale` 原语在，全库无 temporal/reprojection 实现。
12. **P3-d 光谱（hero-wavelength）**：材质线完全没有；仅 hair 子系统 `prism_render_architecture/src/hair/spectral_absorption.rs:187` 有 `hero_wavelengths`（与 P3-d 无关）。
13. **P3-e 四线 × 自定义契约的 pass/TAA 接线**：数学核 + 孪生已落地，接线未做。
14. **P3-f 内存与压缩**：未开始。
15. **P2-f "全部配置"白炉**：`metal(0.1)` 仍被 omit，且门禁文件内注释陈旧。
16. **P1-b 门禁盲区**：第 6 个 `MaterialHeader` 镜像（`visibility_raster.wesl:24-35`）未纳入 `MIRRORS`。
17. **历史生成物门禁**：曾因生成器排版差异失败；本轮已重新生成并由 `codegen_freshness` 的 **8 passed / 0 failed** 关闭该问题。

---

## 5. 文档与代码不符清单（建议修订）

### 5.1 内容偏差

1. `doc:1041` coat_darkening"搁置"→ 改为"CPU/WESL 与 reference_pt 局部实现均已有定向门禁；当前 reference_pt clearcoat 定向测试 13/13；完整离线/实时图像级 parity 仍需独立验收"。
2. `doc:1183/1189` 逐区域 illumination"未完成"→ 改为"已落地（CPU + WESL + 作者 API + 门禁）"，并删除与 `doc:1353` 的矛盾。
3. `doc:1153/1187` P3-b"未开始"→ 改为"语义已落地于独立 slice（只存入射辐照度），缺生产 producer 与主 pass 门禁"。
4. `doc:795` `bake::tests` 12 → **13**。
5. `doc:787/801` header 字节链 → 统一为 `20w(80B) → 19w(76B) → +4 PCM(92B) → +4 动态(27w/108B)`。
6. "核心 12 words" / `SURFACE_LOBE_WORDS_BY_SLOT=[4,12,4,4,4,4,4]` → `20` / `[8,16,4,8,16,16,4,0×9,4,4,4]`。
7. `doc:689` 的 `lobe(8)/10 槽/22 行/response_mode 22-24` → `10 lobes/11 槽/23 行/23-25`。
8. `doc:765` "765 const / 561 点 / 4 表" → `783`（含两行均值）/`714`/4 表不含 EON。
9. "白炉覆盖全部配置" → 明确 `metal(0.1)` 被 omit，并清理 `openpbr_reference.rs` 内陈旧注释。
10. `doc:1176` 的 VT 措辞按 `doc:513` 口径改写（采样路径已接线，仅缺数据）。
11. "nodedef 往返未做" → 限定为"nodedef ↔ author_node 映射与实例未落地"。
12. "致密性由 `validate()` 机检" → 归因到 `tests/pcm_parity.rs`。
13. "无任何注册瓣声明能力需求" → 限定为 lobe；非 lobe 有 4 处非空。
14. `texture_uv_gradients` 若被写成 Rust 符号 → 实为 `uv_gradients`（`prism_uv_gradients` 是 WESL 名）。
15. "normal_variance 恒为 0.0" → 补 SSR 分支例外（`ssr_repack.wesl:154/445-447`）。
16. P1-e"只有类型、无上传/消费" → 改为"类型 + CPU 全链已落地；GPU pass uniform 与贵瓣 shader 未落地"。
17. `pkg/prism_material_schema/src/lib.rs:18-20` 头注释仍写"当前 S1、nothing is generated yet" → 陈旧，应改为 S1+S2 已落地。
18. `doc:1166-1168` 的 `--list` 失败记录 → 已不可复现，应删除或标注已修复。

### 5.2 文档内部自相矛盾

1. **P3 编号冲突**：§13 的 `P3-d` = 光谱提升、`P3-e` = 随机单瓣采样；§14.10 的 `P3-d` = seam/一致性、`P3-e` = 四线/自定义契约、`P3-f` = 内存/压缩。"P3-d" 指两件事。
2. **P1-g 状态冲突**：路线图表无 ✅ vs §17.5.8 S2 标 ✅ 已落地。
3. **P2-e 状态冲突**：`doc:1183/1189` 未完成 vs `doc:1353` 已实现。
4. **header 字节冲突**：`doc:787`（24→23 word = 92 B）vs `doc:801`（19+4 = 92 B），两者都不等于实际 27 word / 108 B。
5. **白炉配置数冲突**：§13.a 第七片"7 配置" vs 门禁 `configs[8] + 2`。
6. **§13.a.1 快照与现状冲突**：`prism_render_scene --lib` 1602 vs 现 1718；`--list` 失败记录已不可复现。

---

## 6. 当前仍未执行 / 无法判定的项

1. **其余真机 GPU 对拍**（7 个 target 文件确实存在）：
   `pkg/prism_render_material_gpu/tests/{bc_parity, albedo_table_gpu_parity, dynamic_channels_gpu_parity, glint_gpu_parity, lobe_sampler_gpu_parity, measured_brdf_gpu_parity, texture_lod_gpu_parity}.rs`。
   response composition 已有跨沙盒 Metal readback 通过；上述 7 个材质 GPU target 本轮没有逐项执行，因此不把它们写成通过。门禁文件与 `require_gpu()`/skip 逻辑存在（`tests.rs:53-55`、`shade_tests.rs:36-38`）。
2. **性能数字**：`bench_const_table_vs_buffer_table` 的 5.561 / 8.255 ms、逐光源去重 7.397→5.767 ms（−22.0%）——函数存在（`albedo_table_gpu_parity.rs:374` 等），但数值需真机 GPU 复跑。
3. **GPU 侧语义**（shade-cache 写入是否为辐照度、bindless 采样、page-table 消费）依据源码与门禁代码判定，未在真机二次确认。
