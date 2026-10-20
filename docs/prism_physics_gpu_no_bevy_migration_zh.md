# `prism_physics_gpu` 无 Bevy 数学迁移

`prism_physics_gpu` 的 CPU 参考关节约束路径现在统一使用 `prism_math::ops`，与 GPU/WGSL 数学镜像共享 Prism 的确定性标量实现。旧的 `bevy_math` 依赖和源码引用已移除；该迁移不改变 GPU buffer ABI、约束求解顺序或 CPU/GPU 对齐契约。

验收边界：

- manifest 不再声明 `bevy_math`。
- rigid joint CPU reference 不再导入 `bevy_math::ops`。
- `cargo check --offline -p prism_physics_gpu --lib` 已在独立 target 通过。
- cloth/field 相关并发切片不属于本文档变更，仍按各自验收矩阵验证。
