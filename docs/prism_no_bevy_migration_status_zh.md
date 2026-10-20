# Prism 无 Bevy 迁移状态

截至 2026-10-20，Prism 自有 `pkg/prism_*` 依赖闭包已完成以下迁移：

- `prism_math` 删除 `compat_bevy` 与旧兼容测试，提供 `prism_math::ops`。
- 13 个音频 crate 的数学依赖改为 `prism_math`，并保留 `std`、`no_std`、`serialize` 语义。
- `prism_render_material` 删除 Bevy bridge，改用 Prism 数学与材质契约。
- `prism_render_visibility` 删除 Bevy bridge，视图/场景数据只接受 Prism ABI。
- `prism_physics_gpu` 的 CPU 参考关节约束改用 `prism_math::ops`。

当前唯一仍直接声明 `bevy_*` 依赖的 Prism crate 是 `prism_render_scene`。它正在进行渲染场景运行时迁移，完成前不能宣称 Prism workspace 已经无 Bevy。仓库内 `crates/bevy_*` 是尚未删除的旧上游源码树，不能作为 Prism 新 API 的依赖入口。

## 验证

使用 `cargo metadata --locked --no-deps --format-version 1`，过滤包名以 `prism_` 开头且依赖名以 `bevy` 开头的包，当前结果只有 `prism_render_scene`。已完成切片分别通过其独立的 crate 检查/测试；完整 workspace 验收须等待 `prism_render_scene` 迁移完成后执行。
