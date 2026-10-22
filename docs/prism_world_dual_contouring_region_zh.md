# World Dual Contouring 区域重提取

## 交付范围

`prism_world_terrain::DualContouringMesh::build_region` 接收明确的
`DualContouringCellRegion`（半开 cell 坐标区间），直接读取不可变 `SdfGrid`，不复制整张
体数据。每个 cell 使用自己的八个 sample；区域末端的 sample 作为一圈 halo 读取。

输出只包含四个 incident cell 全部位于请求区域内的 primal-edge 面，因此区域边界保持开放，
不会凭空封盖，也不会把相邻 dirty brick 的面重复写入本区域结果。位置仍使用全局 sample
坐标和原有 `DualContouringGeometry` 编码，`revision` 继续指向源 grid revision。

## 预算与失败语义

- cell 数、顶点数、三角形数、遍历 work 和逻辑峰值字节沿用完整提取的 checked admission。
- `max_bytes` 包含区域 cell-vertex map、position/normal 和 index payload；不包含分配器元数据。
- `InvalidRegion` 表示空区间；`RegionOutOfBounds` 表示超出 `dimensions - 1` 的 cell 域。
- 所有预算和拓扑检查在结果发布前完成；失败不返回部分 mesh。
- 区域内部仍拒绝 checkerboard face、缺失 cell vertex、退化面和 f32 精度损失。

## 复杂度与边界

区域提取只扫描请求 cell 与一圈 sample halo，时间和临时空间随区域大小增长，不随完整
world grid 大小增长。区域边界不提供 cap；调用方若要组成完整表面，必须在相邻区域之间使用
同一 source revision、geometry 与拓扑策略，并在更高层负责拼接/发布。

这不是 Transvoxel、mixed-LOD stitching、GPU/RHI consumer 或 renderer graph 接线；也不
自动维护 dirty queue。它提供的是可用于 dirty-brick/局部重提取的确定性 CPU primitive。

## 验证

```text
CARGO_TARGET_DIR=/private/tmp/prism-world-dc-region-20261023 \
CARGO_INCREMENTAL=0 \
cargo test -p prism_world_terrain --lib --offline
# 145 passed

CARGO_TARGET_DIR=/private/tmp/prism-world-dc-region-20261023 \
CARGO_INCREMENTAL=0 \
cargo test -p prism_world_terrain --features mesh --lib --offline
# 149 passed

CARGO_TARGET_DIR=/private/tmp/prism-world-dc-region-20261023 \
CARGO_INCREMENTAL=0 \
cargo check -p prism_world_terrain --no-default-features --offline
```

测试覆盖全局坐标保持、开放边界、空/越界区域、原有 Dual Contouring 拓扑、预算、确定性
和 `mesh` adapter 回归。
