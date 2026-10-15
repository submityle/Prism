# Asset Build Core 文件协调切片

## 状态

截至 2026-10-16，本切片已完成并提交。范围仅覆盖：

- `pkg/prism_asset_build_core/src/transaction.rs`
- `pkg/prism_asset_build_core/src/ddc.rs`
- 对应 cfg 单测与本状态说明

Unix 使用稳定锁文件和 `flock` 提供跨进程排他协调。非 Unix 当前没有经过验证的等价后端，因此采取 fail-closed 语义：

- `RevisionStore::open` 返回 `PublishError::CoordinationUnavailable`；
- `DerivedDataCache::open`、写入、驱逐和失效路径返回 `DdcError::CoordinationUnavailable`；
- 不再把单进程互斥锁或空实现宣称为跨进程协调；
- 不修改 `release.rs` 或 CLI，也不改变 Unix 的发布协议。

`#[cfg(not(unix))]` 单测分别验证 RevisionStore 和 DDC 的明确错误。Windows 原生等价锁后端仍是后续独立切片，不属于本次实现范围。
