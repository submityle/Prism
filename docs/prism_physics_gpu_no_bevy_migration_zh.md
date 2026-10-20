# `prism_physics_gpu` 无 Bevy 数学迁移

`prism_physics_gpu` 的 CPU 参考关节约束路径现在统一使用 Rust 侧的 `prism_math::ops`。GPU/WGSL 使用独立的 shader 实现，不能由 Rust 数学依赖的替换推导两者共享确定性标量实现，也不能据此宣称 CPU/GPU 数值或位级一致。旧的 `bevy_math` 依赖和源码引用已移除；该数学依赖替换的范围不包含 GPU buffer ABI 或约束求解顺序变更。

验收边界：

- manifest 不再声明 `bevy_math`。
- rigid joint CPU reference 不再导入 `bevy_math::ops`。
- 此前迁移切片仅确认 `cargo check --offline -p prism_physics_gpu --lib` 在独立 target 通过；本次文档纠正未重跑该检查。
- 未验证真实 GPU 执行、CPU/WGSL parity 或跨后端确定性；这些仍需独立的 GPU 对拍和误差边界验收。
- cloth/field 相关并发切片不属于本文档变更，仍按各自验收矩阵验证。
