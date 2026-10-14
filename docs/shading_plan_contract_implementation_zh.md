# Response Composition S1/S9 契约实现记录

截至 2026-10-14，本切片只建立 CPU 构建期契约和注册绑定，不宣称已有 GPU 响应数学、逐通道合成、动态分桶、CompositionPass 或 `respond_custom` 函数。实现位于 `pkg/prism_render_material/src/shading_plan.rs`，集成测试位于 `pkg/prism_render_material/tests/shading_plan.rs`。

## 数据模型和入口

- `ShadingPlan::Single { program }` / `MaskBlend { left, right, mask, domain }`；`ProgramRef::Builtin(Lit | Npr | Unlit)` / `Registered(ProgramId)`。`MaskSource` 含纹理语义、顶点属性、UV 通道、区域遮罩、schema 动态通道。`BlendDomain` 含全辐射、仅 direct、固定辐射通道位掩码。
- `ResponseManifest { surface, lighting, output, execution, cache }`。`SurfaceContract` 包含 required/optional/ignored-with-reason 字段、supported/ignored-with-reason 瓣，映射字段还须有非空 `FieldMapping { field, semantic, test }`。这只是可追溯的声明与测试标识，不是数学正确性证明；实际语义需独立 shader/CPU golden 证据。
- `SurfaceFieldCatalog::generated()` 直接投影 codegen 的 `SURFACE_FIELDS`（core / lobe owner + name，所有非 padding 字段）；`new(fields)` 用于其他生成目录或测试。它不复制 offset、word 数、字段名或瓣 bit。构造时拒绝空表、重复字段、未知瓣，以及没有 core 或任一生成瓣字段的目录；构造器无法单独证明一个恶意的非空子集包含某一 owner 的**全部**字段，生产入口必须用 `generated()`。
- `ResponseManifest::validate(catalog)` 检查全字段声明（包括未活动瓣）、映射测试证据、重复/冲突、理由、所有五类契约结构与已知输出/缓存通道。`validate_for_surface(catalog, active_lobes)` 另外拒绝未知 bit、每个未声明的活动瓣，以及非活动瓣的 required 字段；调用者须传派生的 active mask，不能用 authored closure mask 冒充。
- `ResponseRegistry::new(catalog, abi_expectation)`；`register(program, manifest, compiled, packages)` 原子注册；`resolve(program, packages)` 在每次消费前重验 package；`validate_plan(plan, context, packages)` 验证两臂、输入光照/缓存、遮罩能力与输出交集，返回借用注册项的 `ValidatedPlan { execution_family, left, right, blended_channels }` 给 bake。`unregister` 后数值 ID 永不复用；ID `0` 有效，`u32::MAX` 保留，缺失/注销 ID 均为硬错误，不做隐式 Lit fallback。

## 编译入口的真实绑定边界

`CompiledEntry::from_spirv(package_manifest, words)` 读取真实 SPIR-V 字节码字数组，校验魔数、头、指令边界、匹配 `OpEntryPoint` 名称与对应已定义的 `OpFunction`（含 label 和 function end），拒绝只有元数据/函数声明的伪入口。注册还要求 architecture 的 `ShaderPackageRegistry` 对应包状态 `Ready`，manifest 的 package ID、source revision、entry point、ABI version/hash、permutation count 与编译入口完全一致，且符合期望 ABI。仅 package manifest 的 `Ready` **不够**。

此轻量反射不是完整 SPIR-V validator：它不证明函数类型、执行模型、shader 语义，也不证明某个 `respond_custom` 签名；编译流水线仍须对产物运行正式 validator 并提供与 module 字节绑定的包证明。现有 `ShaderPackageRegistry` 不记录模块 hash，因此**同 revision 的二进制替换不可由此 API 独立检测**。父/后续 compiler registry 应扩展 package artifact digest / validated-module handle 才能关闭这一信任边界。当前禁止裸字符串、裸 bool 作为编译入口证明。

目标 shader ABI（未实现的规范签名）：`respond_custom(&SurfaceData, &LightingContext, &ViewContext, &CustomData) -> ResponseOutput`。输出计划包含 direct、indirect_diffuse、reflection、transmission、emission、alpha、auxiliary；S7 必须先定义 alpha / auxiliary 合并规则。当前 `ChannelMask` 只允许五个辐射通道参与组合，禁止把 alpha 或 auxiliary 当作可 lerp 的辐射量。

## 上游 S2 接缝与责任边界

schema 的 `SURFACE_FIELDS` 是字段全集；`SURFACE_RESPONSE_FIELD_CONSUMPTION` 应按 `(owner, name, builtin program)` 为每个非 padding 字段提供映射状态（mapped + semantic + test，或 ignored + reason）。父生成后将每个 builtin 的 mapped 行投影成 required/optional + `FieldMapping`，ignored 行投影为 `ignored_fields_with_reason`；lobe supported/ignored 必须有独立显式策略，不能凭一行布尔宣称支持。生成 manifest 后仍调用 `ResponseManifest::validate` 和 `validate_plan`，不能以矩阵替代运行时 active-lobe 断言。

schema 只维护 `lit` / `npr` / `unlit` 内建消费者身份；custom `ProgramId` 的数值身份只由编译产物注册表持有，**不在 schema 再编号**。runtime/bake 不能从 `ClosureKind` 推导响应程序。当前 schema 消费矩阵未落盘时，本模块不会伪造默认 Builtin manifest 或数学函数：注册必须由调用者显式给出真实 manifest 与编译入口。

`ExecutionFamily::{FusedBuiltin, SplitResponse, Composition}` 和 `BlendDomain` 当前只声明契约 enum，不维护派生值表；architecture 的统一身份一旦落盘，应将本地 `ExecutionFamily` 替换成 reexport 同一类型。`shading_plan::MaskSource` 是响应数据枚举；crate 根部旧的 `bake::MaskSource` 是 trait，新枚举以 `ResponseMaskSource` 根 reexport 避免碰撞。

## 验证记录

- `cargo test -p prism_render_material --test shading_plan --offline -- --nocapture`：28/28 通过；除原有 schema 字段投影、每一字段漏报、十个活动瓣逐一漏报、映射/理由、五契约、两臂、三内建身份、四个动态 channel 的能力继承、ID 0 / MAX / stale、包状态/ABI/修订以及真实 SPIR-V 入口外，新增验证所有 builtin 的字段/lobe 分区、Unlit decode-only contract 与 execution-table kind=2，以及 stale `Registered(ProgramId)` 在 `validate_plan` 构建期硬失败。
- 2026-10-15 当前定向复验：材质 lib 870 项、`schema_parity` 7 项、`shading_plan` 28 项、`material_header_abi` 2 项、`pcm_parity` 7 项和 `codegen_freshness` 8 项全部通过；`surface_layout.rs` 与 `material_unpack.wesl` 已由 `prism-material-codegen write` 重新生成。现存上游 warnings 未在本切片处理。
- `rustfmt --edition 2024 --check` 对本模块和新增测试通过；`git diff --check` 对 `lib.rs` 通过。
