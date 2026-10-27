//! `debug` —— 调试工具层。
//!
//! - [`timetravel`]：时间旅行调试，记录调用快照序列并支持倒带/差分/重放。

pub mod timetravel;

pub use timetravel::{ReplayResult, Snapshot, SnapshotDiff, TimeTravelSession};
