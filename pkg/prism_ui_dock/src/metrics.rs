//! Bounded, allocation-aware dock frame metrics.

use alloc::vec::Vec;

const MAX_SAMPLES: usize = 256;

/// One frame observation supplied by the host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DockFrameSample {
    /// Layout solve duration in nanoseconds.
    pub solve_ns: u64,
    /// Retained Element update duration in nanoseconds.
    pub element_update_ns: u64,
    /// Draw-list command count.
    pub draw_commands: u32,
    /// Number of visible content views.
    pub visible_panels: u32,
    /// Number of factory calls.
    pub factory_builds: u32,
    /// GPU submission duration in nanoseconds.
    pub gpu_submission_ns: u64,
    /// Present-to-present latency in nanoseconds.
    pub present_latency_ns: u64,
    /// Whether this frame missed its host frame budget.
    pub dropped_frame: bool,
}

/// Percentile summary over the retained frame samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DockMetricSummary {
    /// Number of samples included in the summary.
    pub samples: usize,
    /// Layout solve P50 in nanoseconds.
    pub solve_p50_ns: u64,
    /// Layout solve P95 in nanoseconds.
    pub solve_p95_ns: u64,
    /// Layout solve P99 in nanoseconds.
    pub solve_p99_ns: u64,
    /// Element update P50 in nanoseconds.
    pub element_p50_ns: u64,
    /// Element update P95 in nanoseconds.
    pub element_p95_ns: u64,
    /// Element update P99 in nanoseconds.
    pub element_p99_ns: u64,
    /// GPU submission P50 in nanoseconds.
    pub gpu_p50_ns: u64,
    /// GPU submission P95 in nanoseconds.
    pub gpu_p95_ns: u64,
    /// GPU submission P99 in nanoseconds.
    pub gpu_p99_ns: u64,
    /// Total dropped-frame count observed by this collector.
    pub dropped_frames: u64,
}

/// Fixed-capacity metrics collector.
#[derive(Clone, Debug, Default)]
pub struct DockRuntimeMetrics {
    samples: Vec<DockFrameSample>,
    dropped_frames: u64,
}

impl DockRuntimeMetrics {
    /// Creates an empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one frame and retains only the newest 256 observations.
    pub fn record(&mut self, sample: DockFrameSample) {
        if sample.dropped_frame {
            self.dropped_frames = self.dropped_frames.saturating_add(1);
        }
        self.samples.push(sample);
        if self.samples.len() > MAX_SAMPLES {
            self.samples.remove(0);
        }
    }

    /// Returns the number of retained observations.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns whether no observations are retained.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Builds deterministic P50/P95/P99 summaries.
    pub fn summary(&self) -> DockMetricSummary {
        DockMetricSummary {
            samples: self.samples.len(),
            solve_p50_ns: percentile(&self.samples, |sample| sample.solve_ns, 50),
            solve_p95_ns: percentile(&self.samples, |sample| sample.solve_ns, 95),
            solve_p99_ns: percentile(&self.samples, |sample| sample.solve_ns, 99),
            element_p50_ns: percentile(&self.samples, |sample| sample.element_update_ns, 50),
            element_p95_ns: percentile(&self.samples, |sample| sample.element_update_ns, 95),
            element_p99_ns: percentile(&self.samples, |sample| sample.element_update_ns, 99),
            gpu_p50_ns: percentile(&self.samples, |sample| sample.gpu_submission_ns, 50),
            gpu_p95_ns: percentile(&self.samples, |sample| sample.gpu_submission_ns, 95),
            gpu_p99_ns: percentile(&self.samples, |sample| sample.gpu_submission_ns, 99),
            dropped_frames: self.dropped_frames,
        }
    }

    /// Clears samples and dropped-frame counters.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.dropped_frames = 0;
    }
}

fn percentile(
    samples: &[DockFrameSample],
    value: impl Fn(DockFrameSample) -> u64,
    percentile: usize,
) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut values: Vec<u64> = samples.iter().copied().map(value).collect();
    values.sort_unstable();
    let rank = (values.len() - 1)
        .saturating_mul(percentile)
        .saturating_add(99)
        / 100;
    values[rank.min(values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_are_deterministic_and_bounded() {
        let mut metrics = DockRuntimeMetrics::new();
        for value in 1..=300 {
            metrics.record(DockFrameSample {
                solve_ns: value,
                element_update_ns: value * 2,
                gpu_submission_ns: value * 3,
                dropped_frame: value % 10 == 0,
                ..DockFrameSample::default()
            });
        }
        assert_eq!(metrics.len(), MAX_SAMPLES);
        let summary = metrics.summary();
        assert_eq!(summary.samples, MAX_SAMPLES);
        assert!(summary.solve_p50_ns <= summary.solve_p95_ns);
        assert!(summary.solve_p95_ns <= summary.solve_p99_ns);
        assert_eq!(summary.dropped_frames, 30);
    }
}
