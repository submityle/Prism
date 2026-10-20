# prism_asset_macros 设计

版本：0.1.0。适用引擎：Prism。状态：`PLANNED`（代码与测试已落地，等待主代理统一执行 Cargo 验证；尚不宣称 SHIPPED）。

## 1. 职责与运行时契约

对应运行时资产设计 §26 的派生宏切片。读取时的设计文档末章为 §26，未发现 §27。
本 crate 只生成身份实现与借用式直接依赖遍历，不涉及 scheduler、container、渲染或 ECS。

`Asset` derive 同时实现 `Asset` 和 `VisitAssetDependencies`。资产身份通过
`AssetTypeId::of::<Self>()` 取得，继续使用运行时冻结的 FNV-1a 64-bit 名称契约。
不把原设计的反射身份描述直接套到现有容器身份上，避免无迁移更改持久化数据。
默认 `TYPE_NAME = concat!(module_path!(), "::", stringify!(Type))`；
可用 `#[asset(type_name = "project::stable::Name")]` 固定身份。

依赖值明确区分：

- `Strong(UntypedAssetId)`：运行时已分配/预留 slot，包含 generation 与类型标签。
- `Soft { guid: StableGuid, type_id: AssetTypeId }`：持久化非保活引用，不要求存在 slot。

`Asset::visit_references` 枚举完整引用；`visit_dependencies` 是强引用 slot 投影，
供运行时图使用。两个函数由同一生成遍历实现，避免两份字段列表。
手写 Asset 的默认完整视图桥接其强引用 visitor，不丢失既有声明。
`direct_asset_dependencies` 与 `direct_dependencies` 是分配 Vec 的便利层；visitor 本身不分配。
本切片不负责把软引用装入运行时图、解析 GUID、发起加载或修改引用计数。

## 2. 字段发现与声明便利性

自动识别强/软/擦除强句柄，通过 `Option`、`Vec`、数组、切片、`Box`、`Arc`、
引用、元组逐层生成有序遍历；跳过无关标量字段。顺序为声明顺序、容器迭代顺序；
保留重复引用。null 软句柄不报告；弱句柄及 `PhantomData` 不产生依赖。

`#[dependency]` 将整个字段交给 `VisitAssetDependencies`，支持别名、自定义容器和
派生分组，无需手写字段登记函数。运行时提供常用容器与 1–8 元组的 trait 实现；
自动元组扫描不受该 trait 元组 arity 限制。`#[asset(skip)]` 是明确排除入口。
未知容器含可见句柄、属性拼写错误、重复属性、冲突属性均返回带 span 的编译错误。

宏不做名称解析：同名非资产 Handle 类型仍需兼容 visitor 或明确 skip；
隐藏在不透明别名/普通字段后的句柄须显式 dependency。此限制是 Rust 过程宏边界，
不能宣称能够自动发现任意用户类型中的引用。

## 3. 泛型、类型形态与路径

具名/元组/unit struct、enum 以及空 enum 均支持。union 无法安全读取活跃字段，
两种 derive 均拒绝。`Asset` 的类型、生命周期和 const 参数全部给出明确编译期诊断：
一个默认常量名称不能区分任意单态化类型，借用资产也不符合 `'static`。
选择具体包装类型或手写有独立身份的 Asset，而不是静默生成有碰撞风险的名称。

`VisitAssetDependencies` 分组支持泛型/生命周期/const，保留 where 条件，
只为被访问的类型添加 visitor 约束。具体递归分组不添加 `Self: Visitor` 循环条件。
Cargo crate 依赖重命名通过 `proc-macro-crate` 自动识别；facade 可显式覆盖 crate 路径。
生成代码只使用 `core` 和运行时公共契约；宿主路径解析不会给运行时引入 std。

## 4. 验证与集成

新 workspace member 为 `pkg/prism_asset_macros`，根 Cargo 清单及 lock 由主代理维护。
遵循用户明确要求，元数据与现有 `prism_reflect_macros` 一致使用显式值，
不因规则 §9 对全仓库引入 `workspace.package` 迁移。

测试注册：宏 crate 的 `expansion`、`renamed_dependency`；运行时的
`asset_derive`、`dependency_visitor`。覆盖顺序、重复、引用计数、预留 slot、null soft、
强/软分离、结构体/枚举形态、递归、泛型分组、Cargo 重命名和 facade 路径。
负例包含 AST 诊断断言及编译失败 doctest。

以下命令由主代理统一执行，本代理不占用共享 target：

```sh
CARGO_INCREMENTAL=0 cargo test --locked --target-dir target/asset-system -p prism_asset_macros --all-targets
CARGO_INCREMENTAL=0 cargo test --locked --target-dir target/asset-system -p prism_asset_macros --doc
CARGO_INCREMENTAL=0 cargo test --locked --target-dir target/asset-system -p prism_asset --test asset_derive --test dependency_visitor
CARGO_INCREMENTAL=0 cargo test --locked --target-dir target/asset-system -p prism_asset --no-default-features --test asset_derive --test dependency_visitor
CARGO_INCREMENTAL=0 cargo test --locked --target-dir target/asset-system -p prism_asset --doc
CARGO_INCREMENTAL=0 cargo clippy --locked --target-dir target/asset-system -p prism_asset_macros --all-targets -- -D warnings
```

## 5. 多维质量评估

| 维度 | 现状与取舍 |
| --- | --- |
| 性能 | 借用遍历无分配、无锁、不克隆句柄；只 convenience collect 分配；无基准数值声明。 |
| 效果 | 类型标签与 GUID/slot 语义分离，重复保留；不宣称已验证尚未运行的测试。 |
| 易用性 | 默认容器无字段登记；自定义分组以 dependency 标注，无需维护 visitor 列表。 |
| 易维护性 | 宏逻辑与 runtime 分离，生成代码依赖公开 trait；不改调度/容器。 |
| 易拓展性 | 用户 visitor/custom container 可组合；类型别名必须显式标注。 |
| 易迭代性 | 独立测试与版本文档；稳定名称变更需迁移；暂不支持泛型 Asset。 |
| 健壮性 | 非法属性、union、未知可见句柄容器及身份歧义编译时报错；不解析不透明类型。 |
| 安全 | 无 unsafe；软引用不解析不保活；不读 union，不做指针解引用；依赖版本复用现有生态。 |

未覆盖：全 workspace 回归、性能 benchmark、任意不透明类型的自动发现。
前两者需独立验证，最后一项是明确的能力边界。
