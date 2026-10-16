# World Runtime Asset-Container Range Assembly

## 目标

本切片把 `RuntimeManifest`/`OverlaySet` 的逻辑 asset range 读取接入现有
`AssetContainerCellProvider`。一个 `CellManifest` 被规划成确定性的、连续的
逻辑 payload ranges，范围通过 `ManifestReader::read_ranges` 路由到当前已验证
的 container，并按声明顺序组装回完整 PRWC cell bytes。

## 合同

- 旧的 `read_cell` API 保留；单 range 仍使用原有完整读取路径，多 range 使用
  range assembly。
- `AssetContainerCellRangePlan` 的 range 从 offset 0 开始，连续覆盖
  `CellManifest.io_bytes`，不重排、不重叠、不遗漏。
- `max_io_bytes`、`max_range_bytes`、`max_ranges` 和物理 batch/gap 预算均在
  读取前检查；预算不足时 fail-closed，不分配完整 payload。
- `ManifestReader` 负责 route、overlay winner、container identity、非零逻辑
  offset、raw/Deflate entry 和底层 source 越界检查。
- 组装完成后复用现有 PRWC header 校验和请求 checksum；失败或取消只返回错误，
  不向 `SceneHost` 提交部分 bytes。`CellFrameCoordinator` 已有的 previous
  manifest 恢复语义保持不变。

## 性能边界

同一 container 的 ranges 由 `ManifestReader`/`ContainerReader` 按物理位置做
确定性 coalescing；Deflate entry 按现有 container reader 的解码合同读取，未
引入 GPU 解压或 GPU 直传。当前 assembly 需要一个完整的 host-side PRWC
`Vec<u8>`，因此不是部分场景实例化或零拷贝提交。

## 验证

```text
CARGO_TARGET_DIR=/private/tmp/prism-world-range-stream-validation \
cargo test --offline -p prism_world_runtime --features asset_container --lib

CARGO_TARGET_DIR=/private/tmp/prism-world-range-stream-validation \
cargo clippy --offline -p prism_world_runtime --features asset_container \
  --all-targets -- -D warnings
```

专用测试覆盖 raw/Deflate、非零逻辑 offset、声明顺序、短读/底层 range 失败、
越界由现有 reader 拒绝、取消、checksum、range-count 和物理 batch 预算。

## 明确未覆盖

本切片不改变 asset-container 格式、`RangeSource` ABI、GPU/renderer、terrain、
场景 payload 解析、远程 transport、加密和 GPU 直传；也不声称已实现部分 cell
提交或 GPU-side 解压。
