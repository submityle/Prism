# 01 · Surface ABI 与数据布局

> 本文记录当前 Surface ABI 的真源、字段边界、身份语义和可复现门禁。响应程序与组合规则见 [`02-response-composition_refactor_zh.md`](02-response-composition_refactor_zh.md)。

## 1. ABI 真源与职责

Surface ABI 的真源是 `pkg/prism_material_schema/schema/surface.toml`。`prism_material_schema` 的 codegen 从 schema 派生：

- Rust surface layout 与字段目录；
- WESL `material_unpack` 与 mirror struct；
- lobe、channel、response program、cache auxiliary 和 identity extension 表；
- 槽位、字足迹、默认值、闭包关系和 ABI 元数据。

运行时只消费生成结果。`SurfaceData` 是 OpenPBR 核心参数和活动扩展瓣的 packed 表示；`GpuMaterialHeader` 负责运行时记录、渲染状态、参数/纹理 heap 地址、PCM digest、动态通道和 dense execution binding。

## 2. 当前字段分层

### 2.1 SurfaceData

Surface layout 分为：

1. OpenPBR 核心字段；
2. KHR 区域瓣；
3. PRISM 扩展瓣；
4. schema 声明的 channel 与 cache auxiliary。

每个字段的名称、类型、默认值、所在 lobe、slot 和 word footprint 都来自 schema/codegen。packed block 的长度由核心 word 数与实际活动瓣足迹相加得到，`lobe_mask` 用于 GPU/CPU 对称解包。

### 2.2 GpuMaterialHeader

当前 header 明确包含：

- generation / revision；
- `render_class` 与 `render_domain`；
- feature flags 与 `closure_mask`；
- packed surface block 的 `parameter_offset` / `parameter_size`；
- texture / sampler heap 范围；
- active 状态与 material epoch；
- `lobe_mask`；
- PCM `used` / `required` digest；
- `dynamic_wetness`、`dynamic_snow`、`dynamic_damage`、`dynamic_dissolve`；
- 五个 dense execution binding word。

旧的 illumination、custom-program、specialization-low/high 语义字段不在当前 Rust header，也不在 WESL mirror 中。响应身份由 `ShadingPlan` 和执行绑定表表达；自定义程序 id 由 `ProgramRef::Registered(ProgramId)` 进入 plan / program 表。

## 3. 身份与索引

当前有三种不同身份，不互相替代：

| 身份 | 作用 | 权威来源 |
|---|---|---|
| `closure_mask` | 记录作者图授权的 closure 集合 | IR / authoring |
| `lobe_mask` / `active_lobes` | packed surface 实际携带的活动瓣 | lowering + schema lobe table |
| execution binding | 当前 plan、program、package、permutation 的可执行身份 | execution tables + verified package |

`active_lobes` 是 Surface packed decode 和执行身份的输入；`closure_mask` 不是 pipeline key。这样可以保留授权记录，同时避免原始 closure 位与派生 surface 形状不一致。

`MaterialRenderClass` 不属于 Surface lobe identity。当前七个值只表达 blend/raster family；响应 program、execution family、render phase 和 package identity 由各自的表表达。

## 4. ClosureKind 的分层边界

`pkg/prism_render_material/src/ir.rs` 中的 `ClosureKind` 仍保留作者图语义，包括 `Npr`、`Custom`、`Mix`、`Layer` 等变体。它们不直接充当 runtime response program：

```text
Authoring IR / ClosureKind
        ↓ lowering
SurfaceData + ShadingPlan
        ↓ resolve
GPU header + dense execution binding
```

因此 closure 图可以表达作者语义，Surface ABI 可以表达实际 packed 数据，response registry 可以表达可执行程序；三者各自只有一个 owner。

## 5. OpenPBR 导入与区域隔离

- `adapt_openpbr_surface` / `import_openpbr_surface` 是当前 OpenPBR 导入入口。
- Standard Surface 别名只在语义一致时映射；未知字段直接返回错误。
- `project_to_openpbr` 是有损向下投影。
- 纯 OpenPBR 子集通过 `pure_openpbr_projects_to_a_byte_identical_fixed_point` 保持字节不动点。
- KHR 区和 PRISM 区的字段、slot、默认值与投影规则由 schema namespace 隔离。

这保证扩展瓣可以增加 Surface 能力，而不会把扩展字段偷偷改写成 OpenPBR 核心字段。

## 6. 当前门禁

以下是当前仓库中可直接复现的 ABI 检查：

| 门禁 | 证明内容 |
|---|---|
| `committed_wesl_matches_generator` | 提交的 WESL 生成物与 codegen 输出一致 |
| `committed_rust_layout_matches_generator` | 提交的 Rust layout 与 codegen 输出一致 |
| `generated_field_directory_covers_exactly_non_padding_schema_fields` | schema 非 padding 字段都有且只有一条生成目录记录 |
| `generated_unpack_covers_every_schema_default_once` | 默认值与 unpack 覆盖关系完整且无重复 |
| `handwritten_surface_has_no_schema_derived_layout_definitions` | 消费端没有另写派生布局定义 |
| `rust_and_wesl_surface_mirrors_match_in_order_and_type` | Rust/WESL mirror 字段顺序与类型一致 |
| `surface_mirrors_cover_exactly_the_schema_fields_and_bytes` | mirror 覆盖字段与字节数一致 |
| `core_and_lobe_word_counts_match_surface_rs` / `typed_surface_struct_offsets_match_schema_pack_slots` | word count、offset、slot 一致 |
| `pure_openpbr_projects_to_a_byte_identical_fixed_point` | 纯 OpenPBR 子集保持字节级 fixed point |
| `every_surface_lobe_has_a_stable_activation_source` | 每个 lobe 都有稳定的活动来源 |
| `header_and_packed_parameters_share_the_derived_lobe_mask` | header 与 packed block 使用同一派生活动瓣集合 |
| `abi7_execution_binding_is_five_dense_words_with_full_permutation_width` | 五个 dense binding word 覆盖完整 permutation 宽度 |

## 7. 当前源码与测试证据

直接源码位置：

- schema：`pkg/prism_material_schema/schema/surface.toml`；
- codegen：`pkg/prism_material_schema/src/`；
- header / class / domain：`pkg/prism_render_material/src/record.rs`；
- packed surface 与 lobe mask：`pkg/prism_render_material/src/surface.rs`；
- execution binding：`pkg/prism_render_material/src/execution.rs`；
- authoring IR 与 closure lowering：`pkg/prism_render_material/src/ir.rs`。

本轮 `cargo test -p prism_render_material --tests` 通过：单元测试 `878 passed; 0 failed`；`codegen_freshness` 11 项、`material_header_abi` 3 项、`pcm_parity` 7 项、`schema_parity` 9 项、`shading_plan` 33 项、`surface_abi_mirror` 2 项均通过。

## 8. Git 历史锚点

- `873d8c407 refactor(material): derive surface ABI layout metadata`：建立 schema 派生布局。
- `0de14a400 refactor surface ABI lobe and mirror gates`：加入生成物 freshness、Rust/WESL mirror 和 lobe ABI 门禁。
- `8400fe972 feat: complete runtime and render refactoring implementation`：将 header、执行绑定和材质运行路径接入整体实现。

当前文档不再把手写 `SpecializationId` 位布局或 illumination 轴当作 ABI 现状；它们只作为历史背景，不是当前数据来源。
