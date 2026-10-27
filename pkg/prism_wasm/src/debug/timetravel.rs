//! 时间旅行调试会话（Time-Travel Debugging）。
//!
//! # 概念
//! 每次 [`Sandbox::call_typed`](crate::runtime::sandbox::Sandbox::call_typed) 调用
//! 都会产生一个 [`CallReceipt`](crate::runtime::sandbox::CallReceipt)，其中包含
//! `module_digest / input_digest / output_digest / trace_digest` 四要素。
//! [`TimeTravelSession`] 把这些回执**按时间序列**记录为 [`Snapshot`]，提供：
//!
//! - **倒带（Rewind）**：`at(index)` 回到任意历史调用点查看当时的 Receipt；
//! - **差分（Diff）**：`diff(a, b)` 比较两个快照的摘要变化，快速定位输入/输出/trace 的改变；
//! - **首处输出变化定位**：`find_first_output_change()` 在热插拔后找出第一次行为发生变化的调用；
//! - **重放（Replay）**：`replay(start, end, executor)` 把历史调用序列重新执行，
//!   自动对比 Receipt 是否与记录一致，用于回归测试。
//!
//! # 使用示例
//! ```rust,ignore
//! let mut session = TimeTravelSession::with_capacity(1000);
//!
//! // 记录调用
//! let receipt = sandbox.call_typed::<(i32, i32), (i32,)>("adder", "add", (3, 4))?;
//! session.record_receipt("adder", "add", "(3, 4)", &receipt);
//!
//! // 热插拔后继续记录...
//!
//! // 倒带：查看第 0 个快照
//! if let Some(snap) = session.at(0) {
//!     println!("第 0 次调用输出: {}", snap.returns_repr);
//! }
//!
//! // 差分：比较第 0 次和第 5 次调用
//! if let Some(diff) = session.diff(0, 5) {
//!     if diff.output_changed {
//!         println!("输出在第 5 次调用时发生了变化！");
//!     }
//! }
//!
//! // 找出第一处输出变化
//! if let Some((prev, curr)) = session.find_first_output_change() {
//!     println!("第 {} 次→第 {} 次调用之间输出首次改变", prev, curr);
//! }
//!
//! // 重放范围 [0, 9]
//! let results = session.replay(0, 9, |snap| {
//!     // 调用方负责从 args_repr 反序列化参数
//!     let r = sandbox.call_typed::<(i32,i32),(i32,)>("adder", "add", (3, 4))?;
//!     Ok((format!("{:?}", r.returns), r.receipt.commitment))
//! });
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use crate::security::attestation::Receipt;

/// 单次调用的时间旅行快照。
#[derive(Clone, Debug)]
pub struct Snapshot {
    /// 全局序号（从 0 开始单调递增）。
    pub index: usize,
    /// 被调用的模块逻辑名。
    pub module_name: String,
    /// 被调用的函数名。
    pub func_name: String,
    /// 输入参数的 `Debug` 字符串（仅用于展示，不参与摘要计算）。
    pub args_repr: String,
    /// 返回值的 `Debug` 字符串。
    pub returns_repr: String,
    /// 本次调用的可验证回执。
    pub receipt: Receipt,
    /// 本次调用耗时（毫秒）。
    pub elapsed_ms: u128,
    /// 消耗的燃料（若策略未启用燃料限制则为 `None`）。
    pub fuel_consumed: Option<u64>,
    /// 绝对时间戳（Unix 毫秒）。
    pub timestamp_ms: u64,
}

/// 两个快照之间的差分摘要。
#[derive(Clone, Debug)]
pub struct SnapshotDiff {
    pub from_index: usize,
    pub to_index: usize,
    /// 两个快照调用的是不同模块版本（`module_digest` 不同）。
    pub module_changed: bool,
    /// 输入摘要发生变化（相同函数接收了不同输入）。
    pub input_changed: bool,
    /// 输出摘要发生变化。
    pub output_changed: bool,
    /// 宿主 trace 摘要发生变化（宿主交互模式改变）。
    pub trace_changed: bool,
    /// 聚合承诺发生变化（等价于上述任一变化）。
    pub commitment_changed: bool,
}

/// 单次重放结果。
#[derive(Debug)]
pub struct ReplayResult {
    /// 对应的历史快照序号。
    pub original_index: usize,
    /// 重放后的返回值 `Debug` 表示，`Err` 表示重放执行失败。
    pub replayed_returns: Result<String, String>,
    /// 返回值字符串是否与历史快照一致。
    pub output_match: bool,
    /// Receipt commitment 是否与历史快照一致。
    pub commitment_match: bool,
    /// 历史快照的 commitment（用于对比）。
    pub original_commitment: [u8; 32],
    /// 重放后计算出的 commitment（执行失败时为 `None`）。
    pub replayed_commitment: Option<[u8; 32]>,
}

impl ReplayResult {
    /// 是否完全匹配（输出和承诺均一致）。
    pub fn is_match(&self) -> bool {
        self.output_match && self.commitment_match
    }
}

/// 时间旅行调试会话。
///
/// 非 `Send`/`Sync`（未加锁），建议每个线程/任务持有独立实例。
/// 若需多线程共享，可包裹在 `Arc<Mutex<TimeTravelSession>>` 中。
pub struct TimeTravelSession {
    snapshots: Vec<Snapshot>,
    max_snapshots: usize,
}

impl TimeTravelSession {
    /// 创建无容量上限的会话。
    pub fn new() -> Self {
        Self::with_capacity(usize::MAX)
    }

    /// 创建有容量上限的会话；超出后以 FIFO 方式覆盖最旧的快照。
    pub fn with_capacity(max_snapshots: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            max_snapshots,
        }
    }

    /// 记录一次调用的原始字段。
    ///
    /// 通常直接使用 [`record_receipt`](TimeTravelSession::record_receipt) 更方便。
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        module_name: impl Into<String>,
        func_name: impl Into<String>,
        args_repr: impl Into<String>,
        returns_repr: impl Into<String>,
        receipt: Receipt,
        elapsed_ms: u128,
        fuel_consumed: Option<u64>,
    ) {
        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.remove(0); // FIFO 环形覆盖
        }
        let index = self.snapshots.len();
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.snapshots.push(Snapshot {
            index,
            module_name: module_name.into(),
            func_name: func_name.into(),
            args_repr: args_repr.into(),
            returns_repr: returns_repr.into(),
            receipt,
            elapsed_ms,
            fuel_consumed,
            timestamp_ms,
        });
    }

    /// 便捷方法：从 [`CallReceipt`](crate::runtime::sandbox::CallReceipt) 中
    /// 提取信息并记录。
    ///
    /// ```rust,ignore
    /// let cr = sandbox.call_typed::<(i32,i32),(i32,)>("adder", "add", (1, 2))?;
    /// session.record_receipt("adder", "add", "(1, 2)", &cr);
    /// ```
    pub fn record_receipt<R: std::fmt::Debug>(
        &mut self,
        module_name: impl Into<String>,
        func_name: impl Into<String>,
        args_repr: impl Into<String>,
        receipt: &crate::runtime::sandbox::CallReceipt<R>,
    ) {
        let returns_repr = format!("{:?}", receipt.returns);
        self.record(
            module_name,
            func_name,
            args_repr,
            returns_repr,
            receipt.receipt.clone(),
            receipt.elapsed_ms,
            receipt.fuel_consumed,
        );
    }

    /// 快照总数。
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// **倒带**：获取第 `index` 个快照（从 0 开始）。
    pub fn at(&self, index: usize) -> Option<&Snapshot> {
        self.snapshots.get(index)
    }

    /// 获取最新一个快照。
    pub fn latest(&self) -> Option<&Snapshot> {
        self.snapshots.last()
    }

    /// 获取从第 `from` 个快照开始的所有快照切片。
    pub fn from_index(&self, from: usize) -> &[Snapshot] {
        let start = from.min(self.snapshots.len());
        &self.snapshots[start..]
    }

    /// 获取 `[start, end]`（闭区间）范围内的快照切片。
    pub fn range(&self, start: usize, end: usize) -> &[Snapshot] {
        let n = self.snapshots.len();
        if n == 0 {
            return &[];
        }
        let s = start.min(n - 1);
        let e = end.min(n - 1);
        &self.snapshots[s..=e]
    }

    /// **差分**：比较两个快照的 Receipt，返回差异摘要。
    ///
    /// 若任一索引越界则返回 `None`。
    pub fn diff(&self, from: usize, to: usize) -> Option<SnapshotDiff> {
        let a = self.snapshots.get(from)?;
        let b = self.snapshots.get(to)?;
        Some(SnapshotDiff {
            from_index: from,
            to_index: to,
            module_changed: a.receipt.module_digest != b.receipt.module_digest,
            input_changed: a.receipt.input_digest != b.receipt.input_digest,
            output_changed: a.receipt.output_digest != b.receipt.output_digest,
            trace_changed: a.receipt.trace_digest != b.receipt.trace_digest,
            commitment_changed: a.receipt.commitment != b.receipt.commitment,
        })
    }

    /// **首处输出变化定位**：在快照序列中找出**同函数同输入但输出不同**的第一对。
    ///
    /// 适合在热插拔后调用，快速确认「从哪次调用起行为发生了变化」。
    /// 返回 `Some((prev_index, curr_index))`。
    pub fn find_first_output_change(&self) -> Option<(usize, usize)> {
        self.snapshots
            .windows(2)
            .enumerate()
            .find(|(_, w)| {
                w[0].func_name == w[1].func_name
                    && w[0].args_repr == w[1].args_repr
                    && w[0].receipt.output_digest != w[1].receipt.output_digest
            })
            .map(|(i, _)| (i, i + 1))
    }

    /// **重放**：对 `[start, end]` 范围内的历史快照，
    /// 用 `executor` 函数重新执行并与历史记录比对。
    ///
    /// `executor` 接收一个 `&Snapshot`（含 `args_repr` 用于重建参数），
    /// 返回 `(returns_repr, commitment)` 或错误字符串。
    ///
    /// # 典型用法
    /// ```rust,ignore
    /// let results = session.replay(0, session.len().saturating_sub(1), |snap| {
    ///     // 调用方负责从 snap.args_repr 反序列化参数（业务层实现）
    ///     let cr = sandbox.call_typed::<(i32,i32),(i32,)>(
    ///         &snap.module_name, &snap.func_name, (1, 2),
    ///     ).map_err(|e| e.to_string())?;
    ///     Ok((format!("{:?}", cr.returns), cr.receipt.commitment))
    /// });
    /// for r in &results {
    ///     println!("#{}: output_match={}, commitment_match={}",
    ///         r.original_index, r.output_match, r.commitment_match);
    /// }
    /// ```
    pub fn replay<F>(&self, start: usize, end: usize, executor: F) -> Vec<ReplayResult>
    where
        F: Fn(&Snapshot) -> Result<(String, [u8; 32]), String>,
    {
        self.range(start, end)
            .iter()
            .map(|snap| {
                let original_commitment = snap.receipt.commitment;
                match executor(snap) {
                    Ok((returns_repr, commitment)) => ReplayResult {
                        original_index: snap.index,
                        output_match: returns_repr == snap.returns_repr,
                        commitment_match: commitment == original_commitment,
                        replayed_returns: Ok(returns_repr),
                        original_commitment,
                        replayed_commitment: Some(commitment),
                    },
                    Err(e) => ReplayResult {
                        original_index: snap.index,
                        replayed_returns: Err(e),
                        output_match: false,
                        commitment_match: false,
                        original_commitment,
                        replayed_commitment: None,
                    },
                }
            })
            .collect()
    }

    /// 打印快照序列的摘要到标准输出（调试辅助）。
    pub fn dump_summary(&self) {
        println!(
            "=== TimeTravelSession ({} snapshots) ===",
            self.snapshots.len()
        );
        for s in &self.snapshots {
            println!(
                "[{:>4}] {:>6}ms | {}::{} | args={} | out={} | commitment={}",
                s.index,
                s.elapsed_ms,
                s.module_name,
                s.func_name,
                &s.args_repr,
                &s.returns_repr,
                &hex::encode(&s.receipt.commitment)[..8],
            );
        }
    }
}

impl Default for TimeTravelSession {
    fn default() -> Self {
        Self::new()
    }
}
