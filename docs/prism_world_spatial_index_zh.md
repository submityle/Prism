# World Manifest Hierarchical Spatial Index

本切片补齐 World M0/M1 的候选集查询基础：`prism_world_stream` 新增
`ManifestSpatialIndex`，按 `SectorId.level` 建立独立的确定性八叉层级。它是
manifest snapshot 的加速结构，不取代 `CellManifest`、scheduler resident state
或 IO/scene 事实源。

## 合同

- 构建前调用现有 `validate_manifests`，重复完整 `SectorId` 立即拒绝；构建后按
  `(level, x, y, z)` 排序，查询结果始终按完整 `SectorId` 排序。
- 每个 level 独立建树，节点保存有界整数坐标 AABB；叶节点最多 16 个条目，
  坐标继续细分到条目收敛或 64 层上限，覆盖完整 `i64` 坐标范围，不使用有损
  Morton hint 作为身份。
- `query_sector_range` 和 `query_sector_cube` 只返回目标 level 且落入闭区间的
  manifest；`query_bounds` 先按 sector 层级裁剪，再对 manifest 的权威世界 AABB
  做闭区间精确相交判断。
- `StreamScheduler::try_plan_indexed` 复用既有 priority、Data Layer、stale
  revision、hysteresis、byte/action budget、eviction 和 VM trace 规则。默认的
  `try_plan` 保持线性兼容路径，索引由上层在 immutable manifest snapshot 更新时
  重建。

## 性能边界

索引把 load/prefetch 候选生成从每个 demand 扫描全部 manifest 改为层级 AABB
裁剪；查询结果仍需按完整键排序，构建成本和节点内存随 manifest snapshot 变化
支付。当前 eviction 仍只检查 scheduler 已记录的 resident sector，未引入跨帧
持久化索引、磁盘排序、Hilbert on-disk layout 或 GPU query。它也不证明完整
World 的“无 pop、无尖峰”视觉/帧时验收。

## 验证

```text
CARGO_TARGET_DIR=/private/tmp/prism-world-spatial-index-20261021 \
cargo test --offline -p prism_world_stream --lib
```

当前切片定向结果：**13 passed**。覆盖负数与 `i64::MIN`/`i64::MAX` 邻域、level
隔离、确定性顺序、AABB 边界相交、重复 sector 拒绝，以及 indexed/linear planner
的 action、admitted bytes、trace 和 hysteresis 对拍。

## 未覆盖

本切片未修改默认 scheduler 的调用方、runtime 的 Prefetch 物化语义、GPU/renderer
接线、Hilbert 排序接入、持久化索引格式或产品级 P50/P95/P99 基准；这些需要独立
的 acceptance 和目标硬件/内容验证。
