# `prism_render_visibility` 无 Bevy 迁移验收

## 当前边界

`prism_render_visibility` 只提供 Prism 自身的可见性 ABI、CPU 参考路径和 GPU 工作生成契约。它不再暴露或适配 Bevy 类型，也不再通过可选 feature 依赖 `bevy_*` crate。

本次迁移移除了旧的 `src/bevy_bridge.rs` 及其测试依赖，数学运算统一使用 `prism_math::ops`。上层宿主如果需要接入外部场景数据，必须在宿主边界将数据转换为 Prism 的 `GpuViewRecord`、`SceneBounds` 和 `InstanceRecord`，不得把引擎私有类型重新引入本 crate。

## 验收门

- `Cargo.toml` 不包含 `bevy`、`bevy_math`、`bevy_camera`、`bevy_transform` 或 `bevy_shape` 依赖。
- 源码不包含 `bevy_bridge`、`bevy_math` 或 `bevy_*` 引用。
- 默认 feature、`--no-default-features` 和 `--all-features` 的检查均通过。
- 单元测试覆盖属性插值、纹理 LOD、可见性裁剪、HZB 历史失效和两阶段解析；真实 GPU 验证由渲染宿主的 GPU acceptance suite 负责。

## 性能边界

核心路径保持无分配：视图裁剪、LOD 选择、属性插值和工作项排序只操作定长记录或调用方提供的容器。Bevy 到 Prism 的转换不再隐藏在可选桥接层中，避免在渲染热路径引入跨引擎类型转换和 feature 组合膨胀。
