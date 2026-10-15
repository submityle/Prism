# World streaming placement runtime slice

本切片把 world streaming 的剩余可落地部分限制在两个 engine-neutral 契约内：

- `prism_world_runtime::WorldCellStore` 维护 resident cell 与 `PersistentId` 的稳定索引。索引支持多 cell、跨 cell `relocate`、cell replacement/eviction，以及有明确 byte/cell/object 上限的 canonical restart snapshot。
- `StableRefSceneHost` 是 `SceneHost` 的生命周期适配器。stable-ref decoder 必须由产品 scene 提供，不假设 cell payload 格式；decode/store validation 失败不会调用 scene host，scene host load/reload 失败保留旧索引，eviction 只有在 host 成功后才删除索引。
- `prism_world_gpu::PlacementRenderGraph` 生成确定性的多 cell `Upload → Project → Compact → DrawIndirect` pass plan，并显式声明 source、placement、compact、indirect、scene 的资源读写关系。相同 cell/revision 或同一 cell 的冲突 revision 会在发布 graph 前拒绝。

验证覆盖：runtime 多 cell load、跨 cell resolve/relocation、reload 失败保持旧状态、eviction 后不可解析、磁盘 restart round-trip；GPU graph 的排序、hazard chain、重复/冲突 revision rejection。当前验证是 CPU contract/plan 层；真实 renderer graph/device pass registration、产品 scene decoder、GPU placement buffer 与 graph 的接线仍未声称完成。
