# World Runtime Asset-Container Range Assembly

## 目标

本切片把 `RuntimeManifest`/`OverlaySet` 的逻辑 asset range 读取接入现有
`AssetContainerCellProvider`。一个 `CellManifest` 被规划成确定性的、连续的
逻辑 payload ranges，范围通过 `ManifestReader::read_ranges_into` 路由到当前已验证
的 container，并按声明顺序直接写入同一个完整 PRWC cell buffer。

## 合同

- 旧的 `read_cell` API 保留；单 range 仍使用原有完整读取路径，多 range 使用
  range assembly。
- `AssetContainerCellRangePlan` 的 range 从 offset 0 开始，连续覆盖
  `CellManifest.io_bytes`，不重排、不重叠、不遗漏。
- `max_io_bytes`、`max_range_bytes`、`max_ranges` 和物理 batch/gap 预算均在
  读取前检查；预算不足时 fail-closed，不分配完整 payload。
- `ManifestReader` 负责 route、overlay winner、container identity、非零逻辑
  offset、Raw/Deflate/BlockDeflate 和底层 source 越界检查。
- 组装完成后复用现有 PRWC header 校验和请求 checksum；失败或取消只返回错误，
  不向 `SceneHost` 提交部分 bytes。`CellFrameCoordinator` 已有的 previous
  manifest 恢复语义保持不变。

## 性能边界

同一 container 的 ranges 由 `ManifestReader` 分组后直接写入调用方的 cell
buffer，避免每段各自分配 `Vec<u8>` 再复制成整块 payload。Raw range 按物理位置
做确定性 coalescing；Deflate chunk 在同一批次按 chunk 解码一次；BlockDeflate
索引按 chunk 解析一次、每个被请求的压缩 block 在该批次解码一次，再散写到相应
输出区间。底层仍保留一个完整的 host-side PRWC `Vec<u8>`，并非 mmap 零拷贝、
部分场景实例化或 GPU 直传；异步 cell I/O 与 mmap provider 的连接也未由本切片
完成。

## 验证

```text
cargo test --offline -p prism_asset_container --features mmap --lib

cargo test --offline -p prism_world_runtime --features asset_container_mmap --lib

cargo check --offline -p prism_asset_container --no-default-features

cargo test --offline -p prism_world_gpu --features gpu --lib
```

本次本机离线验证：asset-container mmap **92 passed**；world runtime
`asset_container_mmap` **121 passed**；asset-container no-default-features check
通过；world GPU `gpu` feature **123 passed**。新增 BlockDeflate 测试以计数 backend
验证多个范围共享同一 block 时只执行一次 block decode。原有专用测试覆盖 raw、
Deflate、非零逻辑 offset、声明顺序、短读/底层 range 失败、越界、取消、checksum、
range-count 和物理 batch 预算。

## 明确未覆盖

本切片不改变 asset-container 格式、`RangeSource` ABI、GPU/renderer、terrain、
场景 payload 解析、远程 transport、加密和 GPU 直传；也不声称已实现部分 cell
提交或 GPU-side 解压。产品级吞吐、不同存储设备的 P95/P99 延迟与多平台 GPU
驱动验证尚未提供，因此不能据此宣称达到最终性能目标。
