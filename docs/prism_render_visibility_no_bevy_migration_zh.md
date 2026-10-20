# `prism_render_visibility` 无 Bevy 迁移验收

## 当前边界

`prism_render_visibility` 只提供 Prism 自身的可见性 ABI、CPU 参考路径和 GPU 工作生成契约。它不再暴露或适配 Bevy 类型，也不再通过可选 feature 依赖 `bevy_*` crate。

本次迁移移除了旧的 `src/bevy_bridge.rs` 及其测试依赖，数学运算统一使用 `prism_math::ops`。上层宿主如果需要接入外部场景数据，必须在宿主边界将数据转换为 Prism 的 `GpuViewRecord`、`SceneBounds` 和 `InstanceRecord`，不得把引擎私有类型重新引入本 crate。

## 验收门

- `Cargo.toml` 不包含 `bevy`、`bevy_math`、`bevy_camera`、`bevy_transform` 或 `bevy_shape` 依赖。
- 源码不包含 `bevy_bridge`、`bevy_math` 或 `bevy_*` 引用。
- 此前迁移切片实际新跑的是默认 feature 下的 124 项 lib 单元测试，全部通过。本次文档纠正未重跑该测试；没有新增默认/`--no-default-features`/`--all-features` 三档独立检查的证据。
- lib 单元测试覆盖属性插值、纹理 LOD、可见性裁剪、HZB 历史失效和两阶段解析。真实 GPU acceptance 未在该切片验证，仍须由渲染宿主执行。

## 性能边界

移除桥接层减少了跨引擎类型转换入口，但不能据此宣称核心路径无分配。工作项集合、排序及 CPU 参考路径是否分配，须逐条核对容器和调用方式，并通过分配计数或性能测量验证；该迁移切片未提供此类测量。
