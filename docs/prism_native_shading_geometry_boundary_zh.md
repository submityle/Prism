# Prism native shading geometry 边界

## 目的

shading surface table 的 CPU 构建不应依赖 ECS、旧渲染世界或具体图形后端。
`prism_render_shading` 现在以 `prism_mesh::Mesh` 为输入，生成由 `GpuShadingVertex` 和
`GpuShadingPrimitive` 组成的稳定 payload；`prism_render_scene` 只负责旧资产表示到
`prism_mesh` 的适配、生命周期编排和 GPU 上传。

## 责任边界

| 层 | 负责内容 | 不负责内容 |
| --- | --- | --- |
| `prism_mesh` | 属性、索引、拓扑的 native 数据契约 | ECS、GPU 资源和 Bevy 类型 |
| `prism_render_shading` | 属性读取、缺失属性 flags、三角形展开、索引边界检查、surface ABI payload | 资产事件、RenderWorld、设备队列 |
| `prism_render_scene` | Bevy 资产转换、变更提取、stable geometry handle、buffer 重建与上传 | 重复实现 mesh→surface 的算法 |

构建器只接受 triangle list/triangle strip；缺失位置、属性类型或长度不一致、索引越界、
triangle-list 索引数量不是 3 的倍数和不支持拓扑均返回结构化错误。非索引 mesh 使用顺序
顶点；缺失 normal/UV/tangent 会记录明确 flags，交给后续 surface reconstruction 处理，
不静默伪装成 authored data，也不静默丢弃不完整的末尾 primitive。

## 验证

- `cargo test -p prism_render_shading --lib surface_geometry`：覆盖索引/非索引 mesh、
  triangle-strip 展开、缺失位置和属性、越界索引、不完整 triangle-list 以及 unsupported topology。
- scene 侧保留 header/row ABI 尺寸与 packed-offset 单元测试。
- `legacy-bevy` 只存在于旧 scene adapter 的显式 feature；native shading builder 不引入
  Bevy 依赖。
