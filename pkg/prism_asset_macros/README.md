# prism_asset_macros

资产身份与依赖声明派生宏，通过 `prism_asset::prelude::*` 使用。

```rust
use prism_asset::prelude::*;

#[derive(Asset)]
#[asset(type_name = "example::Image")]
struct Image;

#[derive(Asset)]
#[asset(type_name = "example::Material")]
struct Material {
    albedo: Handle<Image>,
    layers: Option<Vec<Option<SoftHandle<Image>>>>,
    tint: [f32; 4],
}
```

派生覆盖具名/元组/unit struct 及 enum。自动识别 `Handle`、`UntypedHandle`、
`SoftHandle`，递归处理 `Option`、`Vec`、数组、切片、`Box`、`Arc`、引用和元组。
弱句柄不升级；null 软句柄不报告；遍历不分配、不克隆句柄、不加载资产。

`Asset::visit_references` / `direct_asset_dependencies` 返回
`AssetDependency::Strong(UntypedAssetId)` 或 `Soft { guid, type_id }`。
`Asset::visit_dependencies` / `direct_dependencies` 仅投影强引用 slot；GUID 不冒充 slot。

## 自定义分组与边界

```rust
use prism_asset::prelude::*;

#[derive(VisitAssetDependencies)]
struct Group<T> {
    #[dependency]
    values: Vec<T>,
}

#[derive(Asset)]
struct Scene {
    #[dependency]
    group: Group<SoftHandle<Scene>>,
}
```

Rust 过程宏不解析类型别名或跨模块类型定义：别名、自定义容器和嵌套分组须
`#[dependency]`，其类型须实现 `VisitAssetDependencies`。不希望访问的字段用
`#[asset(skip)]` 明确排除。未知容器中可见的句柄会触发编译错误，不会静默漏掉。
无法在宏展开时发现隐藏在未标记不透明类型内的引用，这一边界不能代替 review。

`Asset` 不接受类型/const/生命周期参数，编译期提示使用独立稳定身份的具体包装类型。
分组 visitor 支持泛型、生命周期、const 参数及已有 where 约束。
默认身份为 `module_path!()::Type`，建议持久化资产显式指定稳定 `type_name`；重命名需迁移。
身份仍走运行时冻结的 `AssetTypeId` 名称哈希契约，不替换为反射 ID。

Cargo 依赖重命名自动识别；facade 用 `#[asset(crate = "my_facade::assets")]`。
运行时消费者保持 `no_std + alloc`；过程宏只在编译宿主使用 `std`。

设计、验证命令与质量边界见 [设计文档](docs/prism_asset_macros_design_zh.md)。
