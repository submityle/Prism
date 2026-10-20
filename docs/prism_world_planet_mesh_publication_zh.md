# 行星 Heightfield Mesh 原子发布

## 交付范围

`prism_world_terrain::PlanetMeshPublication` 为跨 tile 行星网格提供 engine-neutral 的事务发布边界：

- build、增量 update 和 remove 先进入 transaction staging，不直接修改当前快照；
- commit 前统一检查 tile 数量、逻辑 mesh 字节、source/mesh tile 与 revision 身份；
- 任何被本批 replacement 影响的已发布邻边都按精确解码毫米高度做跨面校验；
- 使用 generation 做乐观并发检测，旧 transaction 不能覆盖新快照；
- 所有校验失败均保持旧快照、generation 和 tile 顺序不变；
- 成功快照按 `TileKey` 稳定排序，查找为二分搜索，提交只为新快照预留一次 Vec 容量。

该模块只负责 CPU 派生 mesh 的批量发布，不拥有 scene 真相，不创建 GPU 资源，也不宣称已经接入 renderer graph。

## 入口

```rust
use alloc::sync::Arc;
use prism_world_terrain::{
    PlanetMeshPublication, PlanetMeshPublicationLimits,
};

let mut publication = PlanetMeshPublication::new(PlanetMeshPublicationLimits {
    max_tiles: 256,
    max_mesh_bytes: 64 * 1024 * 1024,
    max_neighbor_comparisons: 256 * 65,
});
let mut transaction = publication.begin();
// transaction.stage_build(...);
// transaction.stage_remove(...);
// publication.stage_update(&mut transaction, ...);
let receipt = publication.commit(transaction)?;
```

## 验证

```bash
rustfmt --edition 2024 --check \
  pkg/prism_world_terrain/src/cube_sphere.rs \
  pkg/prism_world_terrain/src/tile.rs \
  pkg/prism_world_terrain/src/planet_mesh.rs \
  pkg/prism_world_terrain/src/planet_publication.rs \
  pkg/prism_world_terrain/src/lib.rs

cargo test --offline -p prism_world_terrain --no-default-features --lib
```

本切片覆盖：匹配邻 tile 的一次性提交、共享边高度不一致时的全批回滚、过期 transaction 拒绝，以及容量失败不改变旧快照。

## 未包含边界

- 没有 GPU buffer swap、render graph resource handoff 或 renderer/ECS decoder 接线；
- 没有跨不同 LOD 的 mesh 资源合并，LOD seam topology 仍由 `HeightfieldLodSeamPlan` 独立提供；
- 没有多线程锁或无等待 GPU fence，generation 只提供调用方可验证的乐观并发门禁；
- 没有物理、GI、网络或持久化消费者接入。
