# Material Artifact 实现与验证记录

本记录对应 Material Artifact 重构的实际代码与实跑验证，每次提交仅记录已完成并验证的实现，不将局部切片的通过表述为整个重构完成。需求源目录 `docs/refactoring-dev/` 不进入提交。

## 2026-10-14：schema 派生瓣布局

`prism_material_schema::codegen::emit_rust_layout` 从 schema 声明生成 `LobeMask` 的全部瓣常量、KHR/扩展区域、瓣数、完整集合与规范打包次序。`surface.rs` 仅保留掩码类型和行为方法，不维护第二份位号表。区域中无瓣时生成零掩码，两个区域同时为空也有回归覆盖。

打包长度与 schema parity 的全子集测试使用真实声明槽位构造组合，覆盖 KHR 0–6 与扩展 16–18 的全部 1024 种组合，不再误用连续低十位。生成物 freshness 测试同时约束布局事实的来源。生成器的单项缓存辅助表输出与 rustfmt 一致，避免格式化产生 freshness 假失败。

性能：掩码操作仍为编译期常量与整数运算，运行时不加载 schema、不新增分配或查表。维护与扩展：新增瓣只修改 schema 声明，布局和规范次序均由同一生成器派生，freshness/parity 自动覆盖实际槽位。

验证命令：

```sh
cargo run -p prism_material_schema --bin prism-material-codegen -- write
cargo test -p prism_material_schema -p prism_render_material
```

实跑结果：schema 20 项、材质单元 872 项、集成门禁 19 项全部通过。CPU/WESL header 镜像、纯 OpenPBR 不动点与扩展区域隔离保持通过。此切片不改变 GPU ABI 或着色数学，无新增 GPU 性能结论。
