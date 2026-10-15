# Response Composition S1/S9 契约实现记录

截至 2026-10-15，本切片建立 CPU 构建期契约、真实 SPIR-V 生产注册入口和注册绑定，并已接入动态分桶、resolve response producer 与稳定 Custom evaluator。实现位于 `pkg/prism_render_material/src/shading_plan.rs`、`pkg/prism_render_shading/src/{classification.rs,resolve.rs,custom_response.rs}`，集成测试位于 `pkg/prism_render_material/tests/shading_plan.rs`。

## 数据模型和入口

- `ShadingPlan::Single { program }` / `MaskBlend { left, right, mask, domain }`；`ProgramRef::Builtin(Lit | Npr | Unlit)` / `Registered(ProgramId)`。`MaskSource` 含纹理语义、顶点属性、UV 通道、区域遮罩、schema 动态通道。`BlendDomain` 含全辐射、仅 direct、固定辐射通道位掩码。
- `ResponseManifest { surface, lighting, output, execution, cache }`。`SurfaceContract` 包含 required/optional/ignored-with-reason 字段、supported/ignored-with-reason 瓣，映射字段还须有非空 `FieldMapping { field, semantic, test }`。这只是可追溯的声明与测试标识，不是数学正确性证明；实际语义需独立 shader/CPU golden 证据。
- `SurfaceFieldCatalog::generated()` 直接投影 codegen 的 `SURFACE_FIELDS`（core / lobe owner + name，所有非 padding 字段）；`new(fields)` 用于其他生成目录或测试。它不复制 offset、word 数、字段名或瓣 bit。构造时拒绝空表、重复字段、未知瓣，以及没有 core 或任一生成瓣字段的目录；构造器无法单独证明一个恶意的非空子集包含某一 owner 的**全部**字段，生产入口必须用 `generated()`。
- `ResponseManifest::validate(catalog)` 检查全字段声明（包括未活动瓣）、映射测试证据、重复/冲突、理由、所有五类契约结构与已知输出/缓存通道。`validate_for_surface(catalog, active_lobes)` 另外拒绝未知 bit、每个未声明的活动瓣，以及非活动瓣的 required 字段；调用者须传派生的 active mask，不能用 authored closure mask 冒充。
- `ResponseRegistry::new(catalog, abi_expectation)`；`register(program, manifest, compiled, packages)` 原子注册；`resolve(program, packages)` 在每次消费前重验 package；`validate_plan(plan, context, packages)` 验证两臂、输入光照/缓存、遮罩能力与输出交集，返回借用注册项的 `ValidatedPlan { execution_family, left, right, blended_channels }` 给 bake。`unregister` 后数值 ID 永不复用；ID `0` 有效，`u32::MAX` 保留，缺失/注销 ID 均为硬错误，不做隐式 Lit fallback。
- 生产侧的唯一入口是 `ResponseRegistry::register_custom_spirv`：它先校验 program ID、Custom 必须走 split、manifest 全覆盖契约和真实 SPIR-V 结构/Naga 验证，再把 `VerifiedShaderArtifact` 放入 `ShaderPackageRegistry`，驱动 package 到 `Ready`，复核 ABI、manifest 全字段和 artifact digest，最后才发布 `ProgramRef::Registered`。成功返回 `CustomResponseReceipt { program, package, digest, function_id }`，供资产/管线层持有稳定绑定；任何 stale/reserved/重复 ID、无效 SPIR-V、包 ABI 不匹配或代码 digest 变化都会明确失败，不会把响应回退到 Lit 或默认程序。无效字节在编译前失败，因此不会污染 package registry；ABI 不匹配会留下可观测的 `Rejected` package，但不会发布 response entry。

## 编译入口的真实绑定边界

`CompiledEntry::from_spirv(package_manifest, words)` 读取真实 SPIR-V 字节码字数组，校验魔数、头、指令边界、匹配 `OpEntryPoint` 名称与对应已定义的 `OpFunction`（含 label 和 function end），拒绝只有元数据/函数声明的伪入口。注册还要求 architecture 的 `ShaderPackageRegistry` 对应包状态 `Ready`，manifest 的 package ID、source revision、entry point、ABI version/hash、permutation count 与编译入口完全一致，且符合期望 ABI。仅 package manifest 的 `Ready` **不够**。

此轻量反射不是完整 SPIR-V validator：它不证明响应数学，也不证明某个 `respond_custom` 签名；`CompiledEntry::from_spirv` 会把入口定义检查与 `VerifiedShaderArtifact` 的 Naga 全模块验证绑定起来，注册/解析阶段还会比较 package artifact digest，因此同一 package identity 下的代码替换会硬失败。设备特性与限额仍由实际 pipeline 创建验证，响应数学由对应对拍套件验证；这三类证据不互相替代。当前禁止裸字符串、裸 bool 作为编译入口证明。

Custom 的 CPU ABI 已落地为 `respond_custom(&SurfaceSample, &LightingContext, &CustomViewContext, &CustomData) -> ResponseOutput`；输出包含五个 radiance 通道与 alpha，auxiliary 仍由组合消费者独立管理。`CustomData::from_execution(program_id, permutation_index)` 与 `shading_resolve.wesl` 使用相同的 wrapping seed、gain、tint、alpha-scale 公式；执行身份是数据，不进入材质 permutation key。当前 GPU resolve 执行的是稳定内建 ABI evaluator，注册 SPIR-V 已完成结构/Naga/digest/身份验证，但任意注册函数体尚未在 resolve kernel 中动态调用，不能把注册成功表述为任意 Custom 字节码已执行。

## 上游 S2 接缝与责任边界

schema 的 `SURFACE_FIELDS` 是字段全集；`SURFACE_RESPONSE_FIELD_CONSUMPTION` 应按 `(owner, name, builtin program)` 为每个非 padding 字段提供映射状态（mapped + semantic + test，或 ignored + reason）。父生成后将每个 builtin 的 mapped 行投影成 required/optional + `FieldMapping`，ignored 行投影为 `ignored_fields_with_reason`；lobe supported/ignored 必须有独立显式策略，不能凭一行布尔宣称支持。生成 manifest 后仍调用 `ResponseManifest::validate` 和 `validate_plan`，不能以矩阵替代运行时 active-lobe 断言。

schema 只维护 `lit` / `npr` / `unlit` 内建消费者身份；custom `ProgramId` 的数值身份只由编译产物注册表持有，**不在 schema 再编号**。runtime/bake 不能从 `ClosureKind` 推导响应程序。当前 schema 消费矩阵未落盘时，本模块不会伪造默认 Builtin manifest 或数学函数：注册必须由调用者显式给出真实 manifest 与编译入口。

`ExecutionFamily::{FusedBuiltin, SplitResponse, Composition}` 和 `BlendDomain` 当前只声明契约 enum，不维护派生值表；architecture 的统一身份一旦落盘，应将本地 `ExecutionFamily` 替换成 reexport 同一类型。`shading_plan::MaskSource` 是响应数据枚举；crate 根部旧的 `bake::MaskSource` 是 trait，新枚举以 `ResponseMaskSource` 根 reexport 避免碰撞。

## 验证记录

- `cargo test -p prism_render_material --test shading_plan --offline -- --nocapture`：30/30 通过；除原有 schema 字段投影、每一字段漏报、十个活动瓣逐一漏报、映射/理由、五契约、两臂、三内建身份、四个动态 channel 的能力继承、ID 0 / MAX / stale、包状态/ABI/修订以及真实 SPIR-V 入口外，还覆盖 custom SPIR-V 成功 receipt、Ready package、digest/function ID、stale ID、无效字节和 ABI mismatch 的失败原子性。
- 2026-10-15 当前定向复验：材质 lib 870 项、`shading_plan` 30 项全部通过；`surface_layout.rs` 与 `material_unpack.wesl` 已由 `prism-material-codegen write` 重新生成。现存上游 warnings 未在本切片处理。
- 2026-10-15 增量复验：`cargo test -p prism_render_shading --offline --lib custom_response` 为 4/4；Custom CPU evaluator 的执行身份公式对拍覆盖 `0/0`、普通值和 `u32::MAX/u64::MAX`；`cargo test -p prism_render_scene --offline --lib shading_resolve_wesl_compiles_and_resolves_all_imports` 为 1/1，真实 WESL import/type-check 通过。
- `rustfmt --edition 2024 --check` 对本模块和新增测试通过；`git diff --check` 对 `lib.rs` 通过。

## Resolve producer 接入

`resolve_pixel` 现在从 `response_program` 和执行 permutation 派生 Custom 身份：分类/动态 bucket 选择 program id，resolve 读取 `execution_permutation_low/high`，Custom 分支经过独立 `apply_custom_variant`，而不是把 Custom 直接当作 Principled 的同义分支。WESL resolve 同时写入 `response_left`，图中由 `shading_resolve -> response_composition` 形成明确 RAW barrier；结构化五通道仍以 legacy HDR 聚合结果为守恒分解，transmission 在没有独立 transport term 时为零。这是稳定 producer ABI 和数据流接线，不是任意 SPIR-V 函数体动态链接证明。
