//! Per-instance metric reports and summaries (§24.6).
//!
//! A dedicated server or large-world deployment runs N process instances, each
//! producing high-frequency frame-time metrics. Before they can be aggregated
//! into a cluster view, each instance's raw samples are summarized into a
//! compact [`InstanceSummary`] (count + min/max/mean + p50/p90/p99/p999). An
//! instance may submit raw samples (summarized here) or a precomputed summary
//! (already reduced on the edge to save bandwidth, §24.6 sampled reporting).
//!
//! Instances are identified by a stable string label (hostname / shard id).
//! Pure `core`/`alloc` integer arithmetic — deterministic, `no_std` + `alloc`,
//! no `unsafe`.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// Nearest-rank percentile of a slice, integer-friendly `core` arithmetic.
///
/// Returns `0` for an empty input; `p` is clamped to `[0, 1]`. Matches the
/// method used by [`crate::hitch`] and [`crate::budget`] so percentiles agree
/// across the crate.
#[must_use]
pub(crate) fn percentile_nearest_rank(sorted: &[u64], p: f64) -> u64 {
    let n = sorted.len();
    if n == 0 {
        return 0;
    }
    let p = p.clamp(0.0, 1.0);
    let product = p * (n as f64);
    let mut rank = product as usize;
    if (rank as f64) < product {
        rank += 1;
    }
    let rank = rank.clamp(1, n);
    sorted[rank - 1]
}

/// A compact statistical summary of one instance's frame times (nanoseconds).
///
/// This is the unit that flows from each instance to the aggregator: either
/// built here from raw samples via [`InstanceSummary::from_samples`], or
/// supplied directly by an instance that already reduced on the edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceSummary {
    /// Stable instance label (hostname / shard id).
    pub label: String,
    /// Number of frame-time samples behind this summary.
    pub count: u64,
    /// Minimum frame time, in nanoseconds (`0` when `count == 0`).
    pub min_nanos: u64,
    /// Maximum frame time, in nanoseconds (`0` when `count == 0`).
    pub max_nanos: u64,
    /// Sum of all frame times, in nanoseconds (for exact cross-instance mean).
    pub sum_nanos: u64,
    /// Median (p50) frame time, in nanoseconds.
    pub p50_nanos: u64,
    /// 90th-percentile frame time, in nanoseconds.
    pub p90_nanos: u64,
    /// 99th-percentile frame time, in nanoseconds.
    pub p99_nanos: u64,
    /// 99.9th-percentile frame time, in nanoseconds.
    pub p999_nanos: u64,
}

impl InstanceSummary {
    /// Summarize an instance's raw frame-time samples (nanoseconds).
    ///
    /// An empty sample set yields an all-zero summary with `count == 0`, which
    /// the aggregator skips when computing cluster statistics.
    #[must_use]
    pub fn from_samples(label: impl Into<String>, samples: &[u64]) -> Self {
        let label = label.into();
        if samples.is_empty() {
            return Self {
                label,
                count: 0,
                min_nanos: 0,
                max_nanos: 0,
                sum_nanos: 0,
                p50_nanos: 0,
                p90_nanos: 0,
                p99_nanos: 0,
                p999_nanos: 0,
            };
        }
        let mut sorted: Vec<u64> = samples.to_vec();
        sorted.sort_unstable();
        let sum_nanos = sorted.iter().copied().fold(0u64, u64::saturating_add);
        Self {
            label,
            count: sorted.len() as u64,
            min_nanos: sorted[0],
            max_nanos: sorted[sorted.len() - 1],
            sum_nanos,
            p50_nanos: percentile_nearest_rank(&sorted, 0.50),
            p90_nanos: percentile_nearest_rank(&sorted, 0.90),
            p99_nanos: percentile_nearest_rank(&sorted, 0.99),
            p999_nanos: percentile_nearest_rank(&sorted, 0.999),
        }
    }

    /// The arithmetic mean frame time, in nanoseconds (`0` when `count == 0`).
    #[must_use]
    pub fn mean_nanos(&self) -> u64 {
        self.sum_nanos.checked_div(self.count).unwrap_or(0)
    }

    /// Whether this summary carries any samples.
    #[inline]
    #[must_use]
    pub fn is_populated(&self) -> bool {
        self.count > 0
    }
}

/// A raw per-instance frame-time report an instance can submit to the
/// aggregator before summarization.
///
/// Carrying raw samples lets the aggregator compute an *exact* pooled cluster
/// distribution (percentiles over the merged sample set) rather than only
/// averaging per-instance summaries. Instances that cannot afford the bandwidth
/// submit an [`InstanceSummary`] directly instead.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InstanceFrameReport {
    /// Stable instance label (hostname / shard id).
    pub label: String,
    /// Raw per-frame times, in nanoseconds.
    pub frametimes_nanos: Vec<u64>,
}

impl InstanceFrameReport {
    /// A report for `label` with no samples yet.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            frametimes_nanos: Vec::new(),
        }
    }

    /// Record one frame time (nanoseconds).
    #[inline]
    pub fn record(&mut self, nanos: u64) {
        self.frametimes_nanos.push(nanos);
    }

    /// Summarize this report into an [`InstanceSummary`].
    #[must_use]
    pub fn summarize(&self) -> InstanceSummary {
        InstanceSummary::from_samples(self.label.clone(), &self.frametimes_nanos)
    }
}
