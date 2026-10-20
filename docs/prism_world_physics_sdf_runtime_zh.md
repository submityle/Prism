# Prism World SDF Physics Runtime

本文记录 World sparse SDF 到 physics 查询消费者的运行时边界。它是派生数据接缝，不改变 `prism_scene` 作者真相，也不宣称已经把 SDF 接入刚体 solver、渲染或 GI。

## 已交付

- `prism_world_physics_bridge` 为 sparse brick 增加显式 `origin_mm`、`voxel_size_mm`、`distance_scale_mm` 与 `revision`，拒绝零尺度。
- `WorldSdfPhysicsBridge::stage` 在发布前检查 brick、sample 和逻辑字节预算；`publish` 对版本保持单调，失败保留旧快照。
- `prism_world_physics_cell_runtime::SdfPhysicsCellHost` 将解码、每 cell staging、聚合驻留预算和 `SceneHost` 生命周期接成一个提交点；reload、decode、revision、预算或 host 失败都不替换旧快照。
- 查询入口按 cell 返回 `Value`、`Miss` 或 `Unknown`。未驻留 cell 和缺失 brick 都是 `Unknown`，不会被解释为空场。
- 物理查询快照保留量化 grid 与世界尺度，逻辑 footprint 计为 48-byte header、16-byte brick metadata 和 2-byte sample；这不是 allocator RSS 或 GPU 显存峰值。

## 验证边界

应至少运行：

```text
cargo test -p prism_world_physics_bridge --lib --offline
cargo test -p prism_world_physics_cell_runtime --lib --offline
```

这些测试证明 cell 生命周期到 physics 查询的原子接入，不证明 `prism_physics_core` 已使用该场生成 SDF 接触，也不证明 `prism_render_scene`、GI 或 GPU renderer graph 已消费同一份 World SDF。后续消费者必须复用 `WorldSdfSnapshot` 的尺度、revision 和 `Unknown` 语义，不能另建未版本化的平行场。
