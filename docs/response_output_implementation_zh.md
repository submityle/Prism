# 02-S7：结构化响应输出的 CPU 契约实现

日期：2026-10-15。依据完整读取的 `docs/refactoring-dev/02-response-composition_refactor_zh.md`，实现 §4.3–4.5 的 CPU 输出、组合语义、共享光照输入接缝与守恒测试准入门禁，并补齐 resolve producer 的结构化左响应写入和 resident CompositionPass 的图依赖。

## 交付范围

本切片的历史 CPU 组合交付写入四个文件；后续 producer/graph 接线另有独立提交：

- `pkg/prism_render_shading/src/response_output.rs`：新独立模块。
- `pkg/prism_render_shading/src/lib.rs`：仅新增公开模块与 crate-root reexport。
- `pkg/prism_render_shading/tests/response_output.rs`：新集成测试。
- `docs/response_output_implementation_zh.md`：本文。

当前实现已修改 `pkg/prism_render_shading/src/classification.rs`、`resolve.rs`、`custom_response.rs`、`pkg/prism_render_scene/src/shaders/shading_resolve.wesl`、`shading/graph.rs` 与 `shading/post_graph.rs`，并通过对应定向验证。注册 SPIR-V 的身份和完整性已验证，但任意外部函数体的 GPU 动态调用不属于当前 ABI，因此不作该完成声明。

## 已完成的接口

所有接口可从 `prism_render_shading::response_output` 使用；主要接口亦在 shading crate 根部 reexport。

```rust
pub struct ResponseOutput {
    pub direct: [f32; 3],
    pub indirect_diffuse: [f32; 3],
    pub reflection: [f32; 3],
    pub transmission: [f32; 3],
    pub emission: [f32; 3],
    pub alpha: f32,
    pub auxiliary: Auxiliary,
}

pub struct LightingContext<'terms, T: ?Sized = ()> {
    pub direct: &'terms [DirectLightSample],
    pub indirect_diffuse_irradiance: Option<[f32; 3]>,
    pub reflection_radiance: Option<[f32; 3]>,
    pub transmission_radiance: Option<[f32; 3]>,
    pub shared_terms: &'terms T,
}

pub struct CompositionOptions<'a> {
    pub mask: f32,
    pub domain: BlendDomain,
    pub shared_alpha: SharedSurfaceAlpha,
    pub alpha_rule: AlphaMergeRule,
    pub auxiliary_rules: &'a AuxiliaryMergeRules,
}
```

- `ResponseOutput::validate()`：每个辐射通道必须 finite，alpha 必须 finite 且位于 `[0,1]`，aux 字段名非空、浮点值 finite。
- `ResponseOutput::radiance()`：线性、非预乘的五个 radiance 通道各相加一次，不重新添加 emission，不乘 alpha。
- `LightingContext::bind(output)`：校验并返回私有字段的 `EvaluatedResponse<'_, 'terms, T>`；只暴露 `output()` 与 `lighting()` 只读引用。
- `evaluate_response_pair(lighting, left, right)`：两条 `FnOnce(&LightingContext) -> ResponseOutput` 臂在成功路径各执行一次。输入或左臂输出校验失败时提前退出，不宣称错误路径仍执行两臂。
- `blend_responses(&left, &right, options) -> Result<ResponseOutput, CompositionError>`：要求两臂绑定同一个 context 引用。
- `validate_response_composition_inputs(left: &ResponseOutput, right: &ResponseOutput, options: CompositionOptions<'_>) -> Result<Auxiliary, CompositionError>`：GPU 上传/构建阶段可直接借用原始输出，校验 mask、通道域、两臂全部 radiance/alpha/aux 与逐字段规则，返回校验后的合并 aux。无需 Lighting 绑定，不 clone 完整输出，不混合或求和 radiance；aux 合并仍会分配字段集合和结果。`blend_responses` 复用该权威入口，另保留 CPU context 引用相同校验及最终运算结果校验。该入口不构成 GPU 两臂 shared lighting 的证明。
- `SharedSurfaceAlpha::new(alpha, render_class)`：保留共享表面阶段已经求出的 alpha 和原 `MaterialRenderClass`，以 getter 读取；不推导新的渲染类别策略。

`BlendDomain`、`ChannelMask`、`ResponseChannel` 直接复用 `prism_render_material` 身份，在模块内 reexport；没有第二套程序或通道编号。

## 三档 radiance 语义

`FullRadiance` 选择 direct、indirect_diffuse、reflection、transmission、emission 全部五个通道。`DirectOnly` 仅选择 direct。`CustomChannelMask` 是这五个固定通道上的非空位掩码；空掩码、alpha/auxiliary 位或未知位均报 `InvalidChannelMask`。

每个已选通道的每个分量计算 `(1-mask)*left + mask*right`。所有未选通道明确保留 **left 基线**，包括 mask=1 时；DirectOnly 并不是把其他通道清零。mask 必须 finite 且在 `[0,1]`，否则报错，不静默 clamp。

mask=0/1 分别直接返回已选通道的左/右值，保持端点位模式，包括 signed zero。中间值不用 `left + mask*(right-left)`，避免差值在大有限输入下溢出为 infinity。浮点运算仍受 f32 精度约束；“精确通道语义”不等于实数无限精度或 GPU bitwise 等价保证。

最终结果再校验。绑定时已校验两臂的所有通道，包括本次不选择的通道；非有限数据不能因为 mask=0 或 DirectOnly 被隐藏。

## Alpha 与逐字段 auxiliary

`AlphaMergeRule::Shared` 是默认值，从 `SharedSurfaceAlpha` 直接取值，与 mask 和两臂 alpha 无关。调用者必须先从 geometry opacity 和 render class 的共享表面策略求值，再传入本模块；本模块没有伪造 hair/water/volume/decal 的光栅策略。

显式 override 为 `Left`、`Right`、`Min`、`Max`，只替换输出 alpha，不改变 render class；不存在隐式 alpha lerp。

`Auxiliary` 用 `BTreeMap<String, AuxiliaryValue>` 表示调用者命名的 CPU 载荷。类型仅有 `Scalar(f32)`、`Vector3([f32;3])`、`Unsigned(u32)`、`Flags(u32)`。`AuxiliaryMergeRules` 保存逐字段规则和共享阶段的 `shared` 值，规则如下：

| 规则 | 行为 / 可用类型 |
|---|---|
| Shared | 取共享阶段同名值，必须存在 |
| Left / Right | 取指定臂同名值，必须存在 |
| RequireEqual | 两臂值必须存在且相等，否则报错 |
| Linear | 两臂 Scalar/Vector3 按同一 mask 逐分量混合 |
| Min / Max | 两臂 Scalar/Vector3 逐分量取极值；Unsigned 取整数极值 |
| BitOr | 仅两臂 Flags 位或 |

对 left/right/shared 字段并集逐条处理：任何 present 字段都必须有规则；未对应任何 present 字段的规则也报错。字段在各来源的类型必须一致，即使某来源此次不被选中。缺来源不自动替换为另一个来源；ID 不插值、flags 不取数值极值、normal 不隐式当颜色混合或自动归一化。字段排序由 BTreeMap 保证确定性。

这是 CPU 组合载荷，不是 schema 字段目录，也不是新的 GPU cache/aux ABI。某个真实缓存字段应选择何种规则、如何编码和分配仍由该缓存消费者声明；本模块不宣称缓存页布局已经扩容。

## 共享 LightingContext 的实际保证与边界

仓库原有 `DirectLightSample { direction, illuminance, visibility }` 被直接借用。现有 `LightingEnvironment` 是源列表与配置，而不是已采样项，因而没有错误地把它重命名为 shared context。

context 的可选环境值是采样后的光照项，而非响应已加权的最终颜色。调用者可用 `shared_terms: &T` 借用已有 LTC/history 等表示；本模块不复制它们的格式，不假设依赖材质的 LTC 可以提前完整求值，也不能验证不透明 `T` 的内容。

两个 `EvaluatedResponse` 必须借用同一个 context 对象；即便两个 context 内容相等，组合仍拒绝不同引用。这支持同一共享输入下的两臂独立 CPU 求值，不要求臂联合编译。

`bind` 只是 CPU 引用身份与值校验，不证明作者回调实际只读取该 context，更不是 shader trust 证明。不会在此执行阴影/环境采样；回调可执行其他代码，不能声称机制已经禁止其自行重采样。GPU split 两 pass 的共享 shadow/LTC/环境项资源绑定、生产 shader 来源以及真实采样次数，应由运行时/帧图/shader trust 测试另行验证。

## 守恒测试排除接缝

`ConservationGate::run_case(suite, &plan, closure)` 支持 `WhiteFurnace` 与 `SpecificationComparison` 两套独立统计：

- 任意 `ShadingPlan::MaskBlend`，无论 BlendDomain，均返回 `ConservationCaseResult::SkippedMaskBlend`；不执行 closure，递增该 suite 的 `skipped_mask_blend`。
- `Single` 递增 `admitted`，执行 closure 一次，把结果原样包为 `Tested(result)`。closure 返回的测试错误不被转换为成功；调用方仍须检查内层结果。
- 计数采用 checked addition，溢出返回 `CounterOverflow`。

`admitted` 是准入数量，不是守恒通过数量。本模块既未削弱旧白炉阈值，也未把 MaskBlend 的视觉组合称为物理守恒。原有纯 BSDF 数学套件未改；接受 ShadingPlan 的材质/场景守恒套件需要采用此接缝后，才能证明其生产测试路径已经机械排除并上报跳过数。

## 实际验证记录

最终验证命令：

```text
cargo test -p prism_render_shading --test response_output --offline -q
cargo test -p prism_render_shading --offline -q
rustfmt --edition 2024 --check pkg/prism_render_shading/src/lib.rs pkg/prism_render_shading/src/response_output.rs pkg/prism_render_shading/tests/response_output.rs
git diff --check -- pkg/prism_render_shading/src/lib.rs pkg/prism_render_shading/src/response_output.rs pkg/prism_render_shading/tests/response_output.rs docs/response_output_implementation_zh.md
```

新增集成测试 **27 passed，0 failed，0 ignored**（包括独立验证入口的五个新增用例）。最新完整 shading crate 复验为 **3379 个单元测试 passed，0 failed，6 个既有测试 ignored**，另有上述 **27 个集成测试 passed**；doc-test 为 0。格式与 whitespace 检查通过。

主要证据包括：

- 五通道 FullRadiance 的逐分量值、emission 仅加一次，DirectOnly 的 left 基线，全部 31 个非空 radiance 掩码。
- 非法掩码/NaN/无穷值拒绝，mask 端点 bit-exact 与大有限正负值中点。
- 全 11 种既有 render class 使用共享 alpha，各显式 override 不插值。
- aux 各规则、类型/缺值/缺规则/多余规则硬错误以及确定性排序。
- 原始输出无需 Lighting 即可验证：aux 结果与 CPU blend 相同、借用输入不变，全部 31 个合法 mask 及非法 mask/domain；两臂未选通道、alpha、未选 aux 的非法值均拒绝。独立入口与 blend 的 aux 错误一致；各通道 finite 但五通道求和会溢出的输入仍通过输入校验，不把 radiance 总和作为额外输入限制。这不是性能基准或 GPU 对拍证据。
- 不同 context 引用拒绝；两臂读取同一 visibility 与 caller-owned 扩展项。
- 调用真实 `evaluate_principled_direct` / `evaluate_stylized_direct` 的 **zero-emission、direct-only** 数学接入测试。旧函数返回聚合颜色，该测试不是五通道完整响应迁移或全部 OpenPBR 字段覆盖证据。
- 共享采样计数测试使用显式 synthetic 样本构造计数器，验证 CPU 调用接缝的“一份输入、两次响应”；不是生产 GPU 阴影采样性能实测。
- 两套守恒门禁对全部三档 MaskBlend 跳过、计数，并保持 Single 测试错误。

过程中首次 check 被并发上游缺失的 `architecture/render_fabric/compile.rs` 阻断，未修改该文件；上游恢复后继续验证。首次集成测试还发现并修复了本模块 aux 取值闭包的生命周期错误。完整套件仍输出旧 `lobe_sampling` / `resolve` / `openpbr_reference` 的 unused qualification/import/variable warnings，未扩大写域去修它们。

## 当前 producer 与 GPU 边界

resolve producer 已写入 `response_left` 的 64 字节结构化输出，并由常驻 `response_composition` 消费；GPU 组合内核保留这里定义的已选/未选通道、shared alpha 与 aux 语义。WESL 侧的五通道分解保证与 legacy HDR 聚合值相加一致，但 transmission 在没有独立 transport term 时为零；这不是“所有 OpenPBR transport 已完整覆盖”的声明。

Custom GPU 分支采用与 CPU 相同的稳定执行身份公式；它是可测试、可缓存、无动态函数指针的 ABI evaluator。`register_custom_spirv` 仍负责真实 SPIR-V 的结构、Naga、manifest、digest 和 package identity 验证，但当前 resolve 不解释任意 SPIR-V 函数体。字段的 mapped 数学语义及 shader trust 仍必须由真实消费/测试支撑，不能由本模块的组合测试代替。

## 2026-10-15 增量验证

- `cargo test -p prism_render_shading --offline --lib custom_response`：4/4 通过，包含 CPU/WESL 执行身份公式对拍和极值 wrapping 回归。
- `cargo test -p prism_render_scene --offline --lib shading_resolve_wesl_compiles_and_resolves_all_imports`：1/1 通过，真实 WESL 依赖导入和类型检查通过。
- 当前工作区未获得可用 Metal/Vulkan/DX12 adapter；因此本次没有新增真实 GPU render/readback parity 证据。既有跨沙盒 response composition 记录仍以 `docs/response_gpu_implementation_zh.md` 为准，不能外推到本次 Custom resolve。
