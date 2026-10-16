# 行星 Heightfield 增量投影切片

日期：2026-10-16

## 交付内容

`prism_world_terrain::PlanetHeightfieldMesh` 现在可消费同一
`HeightfieldTile` 产生的有序 `GridChange`，通过 `update` 原地更新已发布网格：

- 高度变更矩形更新位置；其一圈网格邻域更新有限差分法线。
- mesh 保存 `cache_id / tile / spec / revision`，拒绝跨缓存、跨 tile、跨规格、跳 revision 或重复消费的 change。
- 计算分为校验、完整 preflight、提交三阶段；半径无效、法线退化、跨 tile 共享边不一致时，旧 positions、normals、revision 和 error bound 保持不变。
- 已提供 `PlanetHeightfieldNormalNeighborhood`，边界法线可读取相邻 tile 的共享边与内侧一行；跨面方向由 `TileKey` 的整数邻接元数据决定，不通过浮点方向猜测。
- `update` 不重新分配 positions、normals、uvs 或 indices；上传方可使用返回的保守一圈矩形作为局部更新范围。

## 性能与资源边界

设变更矩形覆盖 `w × h` 个样本，增量工作量为
`O((w + 2) × (h + 2))` 个投影/法线计算；全量索引与 UV 不参与更新。
实现不创建完整 mesh 的临时副本，也不执行 GPU 上传、提交、等待或读回。
返回矩形包含法线一圈，因此调用方可以一次覆盖所有受影响顶点；跨 tile 共享边影响的邻 tile mesh 必须由其自身 owner 更新。

## 验证

```text
CARGO_TARGET_DIR=/private/tmp/prism-world-planet-incremental-20261016 \
  cargo test --offline -p prism_world_terrain --no-default-features

CARGO_TARGET_DIR=/private/tmp/prism-world-planet-incremental-20261016 \
  cargo test --offline -p prism_world_terrain --features mesh

CARGO_TARGET_DIR=/private/tmp/prism-world-planet-incremental-20261016 \
  cargo clippy --offline -p prism_world_terrain --no-default-features --all-targets -- -D warnings
```

本轮结果分别为 103、106、通过；新增测试将增量结果与全量重建逐顶点对拍，并覆盖 stale change 与跨 tile 边界失败原子性。

## 未扩大的边界

本切片不声称完成 Dual Contouring、Transvoxel、混合 LOD seam、跨 tile 多 mesh 原子发布、GPU terrain upload 或 renderer graph 接入；这些仍需独立的运行时接缝和硬件验收。
