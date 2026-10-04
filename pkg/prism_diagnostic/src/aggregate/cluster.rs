//! Cluster-level frame-time aggregation + anomaly localization (§24.6).
//!
//! Given N instances' reports, the operator needs one cluster view: the pooled
//! frame-time distribution (p50/p99/p999 across the whole fleet) and a
//! shortlist of *which instances are misbehaving*. [`ClusterAggregator`]
//! collects raw [`InstanceFrameReport`]s and/or precomputed
//! [`InstanceSummary`]s and produces a [`ClusterFrametimeReport`].
//!
//! Anomaly localization is robust by construction: an instance is flagged when
//! its p99 exceeds the fleet's **median p99** by a configurable factor, with a
//! secondary median-absolute-deviation (MAD) test so a fleet with naturally
//! spread latencies does not flag everything. Using the median (not the mean)
//! keeps one already-broken instance from dragging the baseline up and masking
//! a second. Pure `core`/`alloc` integer arithmetic — deterministic,
//! `no_std` + `alloc`, no `unsafe`.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::instance::{percentile_nearest_rank, InstanceFrameReport, InstanceSummary};

/// Thresholds controlling cluster anomaly detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnomalyConfig {
    /// An instance is a candidate anomaly when its p99 exceeds
    /// `median_p99 * p99_factor` (e.g. `1.5` for +50%).
    pub p99_factor: f64,
    /// Secondary gate: the excess over the median must also exceed
    /// `mad_factor * MAD` of the per-instance p99s, so a naturally wide fleet
    /// is not over-flagged. Set to `0.0` to disable the MAD gate.
    pub mad_factor: f64,
    /// Minimum median p99 (nanoseconds) below which no instance is flagged, so
    /// a fast, healthy fleet with tiny jitter never raises noise.
    pub min_median_nanos: u64,
}

impl Default for AnomalyConfig {
    /// +50% over the median p99, a 3x MAD secondary gate, and a 1 ms floor.
    #[inline]
    fn default() -> Self {
        Self {
            p99_factor: 1.5,
            mad_factor: 3.0,
            min_median_nanos: 1_000_000,
        }
    }
}

/// One instance flagged as anomalous, with the evidence behind the flag.
#[derive(Clone, Debug, PartialEq)]
pub struct InstanceAnomaly {
    /// The flagged instance's label.
    pub label: String,
    /// That instance's p99 frame time, in nanoseconds.
    pub p99_nanos: u64,
    /// The fleet's median p99, in nanoseconds.
    pub median_p99_nanos: u64,
    /// `instance_p99 / median_p99` (how many times the fleet median).
    pub ratio: f64,
}

/// The cluster-wide frame-time report produced by [`ClusterAggregator::report`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClusterFrametimeReport {
    /// Per-instance summaries, sorted by label for determinism.
    pub instances: Vec<InstanceSummary>,
    /// Number of instances contributing at least one sample.
    pub populated_instances: usize,
    /// Total frame-time samples pooled across the fleet.
    pub total_samples: u64,
    /// Pooled minimum frame time, in nanoseconds.
    pub min_nanos: u64,
    /// Pooled maximum frame time, in nanoseconds.
    pub max_nanos: u64,
    /// Pooled mean frame time, in nanoseconds.
    pub mean_nanos: u64,
    /// Pooled p50 across all instances' samples, in nanoseconds.
    pub p50_nanos: u64,
    /// Pooled p99 across all instances' samples, in nanoseconds.
    pub p99_nanos: u64,
    /// Pooled p999 across all instances' samples, in nanoseconds.
    pub p999_nanos: u64,
    /// Median of the per-instance p99s, in nanoseconds (the anomaly baseline).
    pub median_instance_p99_nanos: u64,
    /// Instances flagged as anomalous, worst ratio first.
    pub anomalies: Vec<InstanceAnomaly>,
}

impl ClusterFrametimeReport {
    /// Whether any instance was flagged anomalous.
    #[inline]
    #[must_use]
    pub fn has_anomalies(&self) -> bool {
        !self.anomalies.is_empty()
    }

    /// Look up a per-instance summary by label.
    #[must_use]
    pub fn instance(&self, label: &str) -> Option<&InstanceSummary> {
        self.instances.iter().find(|s| s.label == label)
    }

    /// Whether a given instance was flagged anomalous.
    #[must_use]
    pub fn is_anomalous(&self, label: &str) -> bool {
        self.anomalies.iter().any(|a| a.label == label)
    }
}

/// Aggregates per-instance reports into a [`ClusterFrametimeReport`].
///
/// Instances submitting raw samples ([`add_report`](Self::add_report)) are
/// pooled exactly for cluster percentiles; instances submitting a precomputed
/// [`InstanceSummary`] ([`add_summary`](Self::add_summary)) still contribute to
/// per-instance anomaly detection and the pooled min/max/mean/count, but their
/// samples are not re-pooled for the fleet percentiles (the summary carries no
/// raw samples). Mixing both is supported.
#[derive(Clone, Debug, Default)]
pub struct ClusterAggregator {
    /// Raw sample sets by instance (for exact pooled percentiles).
    raw: Vec<InstanceFrameReport>,
    /// Precomputed summaries by instance (edge-reduced inputs).
    summaries: Vec<InstanceSummary>,
    /// Anomaly detection thresholds.
    config: AnomalyConfig,
}

impl ClusterAggregator {
    /// A new aggregator with default [`AnomalyConfig`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(AnomalyConfig::default())
    }

    /// A new aggregator with explicit anomaly thresholds.
    #[must_use]
    pub fn with_config(config: AnomalyConfig) -> Self {
        Self {
            raw: Vec::new(),
            summaries: Vec::new(),
            config,
        }
    }

    /// Add an instance's raw frame-time report (pooled exactly).
    pub fn add_report(&mut self, report: InstanceFrameReport) {
        self.raw.push(report);
    }

    /// Add an instance's precomputed summary (edge-reduced input).
    pub fn add_summary(&mut self, summary: InstanceSummary) {
        self.summaries.push(summary);
    }

    /// Produce the cluster report: pooled distribution + flagged anomalies.
    #[must_use]
    pub fn report(&self) -> ClusterFrametimeReport {
        // Build the per-instance summary list from both input kinds.
        let mut instances: Vec<InstanceSummary> = Vec::new();
        for report in &self.raw {
            instances.push(report.summarize());
        }
        instances.extend(self.summaries.iter().cloned());
        instances.sort_by(|a, b| a.label.cmp(&b.label));

        // Pooled exact percentiles come only from instances with raw samples.
        let mut pooled: Vec<u64> = Vec::new();
        for report in &self.raw {
            pooled.extend_from_slice(&report.frametimes_nanos);
        }
        pooled.sort_unstable();

        // Pooled count / sum / min / max span both raw and summarized inputs.
        let populated: Vec<&InstanceSummary> =
            instances.iter().filter(|s| s.is_populated()).collect();
        let populated_count = populated.len();
        let total_samples: u64 = populated.iter().map(|s| s.count).sum();
        let sum_nanos: u64 = populated
            .iter()
            .map(|s| s.sum_nanos)
            .fold(0u64, u64::saturating_add);
        let min_nanos = populated.iter().map(|s| s.min_nanos).min().unwrap_or(0);
        let max_nanos = populated.iter().map(|s| s.max_nanos).max().unwrap_or(0);
        let mean_nanos = sum_nanos.checked_div(total_samples).unwrap_or(0);

        let (p50_nanos, p99_nanos, p999_nanos) = if pooled.is_empty() {
            (0, 0, 0)
        } else {
            (
                percentile_nearest_rank(&pooled, 0.50),
                percentile_nearest_rank(&pooled, 0.99),
                percentile_nearest_rank(&pooled, 0.999),
            )
        };

        // Per-instance p99s drive anomaly localization.
        let mut instance_p99s: Vec<u64> = populated.iter().map(|s| s.p99_nanos).collect();
        instance_p99s.sort_unstable();
        let median_p99 = percentile_nearest_rank(&instance_p99s, 0.50);
        let anomalies = self.flag_anomalies(&populated, &instance_p99s, median_p99);

        ClusterFrametimeReport {
            instances,
            populated_instances: populated_count,
            total_samples,
            min_nanos,
            max_nanos,
            mean_nanos,
            p50_nanos,
            p99_nanos,
            p999_nanos,
            median_instance_p99_nanos: median_p99,
            anomalies,
        }
    }

    /// Flag instances whose p99 exceeds the median p99 by the configured factor
    /// (and passes the MAD secondary gate), worst ratio first.
    fn flag_anomalies(
        &self,
        populated: &[&InstanceSummary],
        sorted_p99s: &[u64],
        median_p99: u64,
    ) -> Vec<InstanceAnomaly> {
        if populated.len() < 2 || median_p99 < self.config.min_median_nanos {
            return Vec::new();
        }

        // Median absolute deviation of the per-instance p99s.
        let mut abs_dev: Vec<u64> = sorted_p99s
            .iter()
            .map(|&v| v.abs_diff(median_p99))
            .collect();
        abs_dev.sort_unstable();
        let mad = percentile_nearest_rank(&abs_dev, 0.50);

        let threshold_ratio = self.config.p99_factor;
        let mut anomalies: Vec<InstanceAnomaly> = Vec::new();
        for summary in populated {
            let p99 = summary.p99_nanos;
            let over_factor = p99 as f64 > threshold_ratio * median_p99 as f64;
            // MAD gate: excess over median must exceed mad_factor * MAD. When
            // MAD is zero (a tight fleet) the factor test alone decides.
            let excess = p99.saturating_sub(median_p99);
            let over_mad = self.config.mad_factor <= 0.0
                || mad == 0
                || excess as f64 > self.config.mad_factor * mad as f64;
            if over_factor && over_mad {
                anomalies.push(InstanceAnomaly {
                    label: summary.label.clone(),
                    p99_nanos: p99,
                    median_p99_nanos: median_p99,
                    ratio: if median_p99 == 0 {
                        0.0
                    } else {
                        p99 as f64 / median_p99 as f64
                    },
                });
            }
        }
        // Worst offender first; ties by label for determinism.
        anomalies.sort_by(|a, b| {
            b.p99_nanos
                .cmp(&a.p99_nanos)
                .then_with(|| a.label.cmp(&b.label))
        });
        anomalies
    }
}
