# Prism Asset Container 设计

- 版本：v0.1.0
- 适用引擎：Prism
- 状态：SHIPPED（下述本地容器读取切片）；完整发行与流送目标单独验收。
- 对应总设计：`docs/prism_asset_design_zh.md` §10.5、§23.7；
  `docs/prism_asset_build_design_zh.md` 实施状态及可执行验收矩阵。

## 1. 容器与预算契约

v1 记录独立 payload；v2 以独立 Chunk Table 保存 Raw/Deflate Chunk，目录
保存 Chunk 内原始偏移。挂载校验规范目录、长度、偏移、零填充与原始内容哈希。
Deflate 仍是整单元解码，不是 block 随机访问。Raw-only 批次按物理范围合并；
压缩批次按单元去重解码，混合批次的 Raw 请求分别读取，结果保持调用方顺序。

`ContainerReadLimits` 在分配/解码前限制：文件长度、编码元数据总字节、
entry/chunk 数量、单元 stored/raw 长度、展开比例、输出字节，以及一次挂载
累计解码校验字节。`ContainerReadBudget` 用原子预留约束共享 mount/clone
的活跃字节缓冲，所有错误路径由 RAII 归还；没有空间时返回 `ResourceLimit`，
不阻塞等待其他读取完成。底层字节分配使用可失败的容量预留。

挂载 v2 压缩内容按 Chunk 顺序验证，最多保留一个解码 Chunk，避免按 GUID
交错访问导致同时缓存所有 Chunk。`read_ranges` 在首次 I/O 前验证整个批次，
重复请求的输出分别计费，但同单元只解码一次。Manifest/VFS 的 owned-range
入口使用同一预算，不允许先分配输出再检查限额。

## 2. 边界与未覆盖项

返回 `Vec` 之后的驻留、调用方提供的内存、已存在的 memory-backed 文件、
元数据树节点及解压库内部状态不计入活跃字节缓冲；元数据仍受编码长度和
记录数的独立上限约束。本预算不是整个进程的精确 heap 限额。

挂载需要读取和验证完整内容，并非懒校验远程挂载。range source 在 reader
生命周期内必须不可变；`read_all` 再校验 payload hash，局部读取不认证发布者。
block-level 压缩索引、加密、远程发行、DirectStorage/GPU 解码是 PLANNED，
不得以本切片的通过替代这些目标的验收。

## 3. 多维质量评估

| 维度 | 现状与取舍 |
|---|---|
| 性能 | 单 Chunk 活跃解码、批次去重、Raw coalescing；尚无规模吞吐基准，挂载完整校验有成本 |
| 效果 | 原始字节、hash、输出顺序与短读错误保持严格一致 |
| 易用性 | 默认限额与显式 shared budget；owned-range 防止桥接错误分配 |
| 易维护性 | 预算集中在独立模块，不把驻留/渲染逻辑混入容器 |
| 易拓展性 | RangeSource 隔离后端；可共享预算跨 mount |
| 易迭代性 | wire v1/v2 不变；资源拒绝为结构化新错误 |
| 健壮性 | 恶意长度、批次预验证、并发拒绝、源故障及 RAII 回收均有回归 |
| 安全 | 新代码无 unsafe；哈希不等于认证，明确不可变源与非精确 heap 边界 |

## 4. 验证

2026-10-20：`cargo test -p prism_asset_container --all-features` 通过
75 项原有测试和 `text/integration/read_budget.rs` 的 10 项预算/批次回归。
并发测试通过通道握手验证预留重叠，不使用耗时推断并发。
