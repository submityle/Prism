//! Deterministic, `no_std`-friendly core for pool health metrics
//! (design §24.8 健康监测).
//!
//! This module owns the *decision* half of §24.8's health monitoring: given
//! raw observations (job latencies, per-worker idle streaks, steal
//! attempts/failures, queue depths), it derives the deterministic indicators a
//! scheduler or `prism_diagnostic` exports — worker starvation, steal-failure
//! rate, latency percentiles, longest job, and peak queue depth. Every
//! derivation is integer-only and clock-free, so it is directly unit-testable
//! against a serial oracle. The timing façade that samples *real* wall-clock
//! latencies (the honest boundary) lives in the parent
//! [`health`](crate::health) module.

use alloc::vec::Vec;

/// A fixed-bucket latency histogram over nanosecond samples.
///
/// Construct with [`LatencyHistogram::new`] from a list of ascending upper
/// bounds; a sample lands in the first bucket whose bound is `>=` the sample,
/// or in a final overflow bucket if it exceeds every bound. Percentiles are
/// reported as the upper bound of the bucket the percentile falls in (integer
/// math), so the result is a deterministic, monotone step function of the
/// recorded samples.
#[derive(Clone, Debug)]
pub struct LatencyHistogram {
    /// Ascending bucket upper bounds, in nanoseconds.
    bounds: Vec<u64>,
    /// Per-bucket counts; `counts.len() == bounds.len() + 1` (the last entry is
    /// the overflow bucket for samples above every bound).
    counts: Vec<u64>,
    /// Total number of recorded samples.
    total: u64,
    /// Largest sample recorded so far, in nanoseconds.
    max_sample: u64,
}

impl LatencyHistogram {
    /// Create a histogram with the given ascending bucket upper `bounds` (in
    /// nanoseconds) plus an implicit overflow bucket. Out-of-order or duplicate
    /// bounds are sorted and de-duplicated. An empty `bounds` yields a single
    /// overflow bucket (every sample overflows), which still tracks the count
    /// and maximum.
    #[must_use]
    pub fn new(bounds: &[u64]) -> Self {
        let mut sorted: Vec<u64> = bounds.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let buckets = sorted.len() + 1;
        Self {
            bounds: sorted,
            counts: alloc::vec![0; buckets],
            total: 0,
            max_sample: 0,
        }
    }

    /// Record one latency sample in nanoseconds.
    pub fn record(&mut self, nanos: u64) {
        // `Ok(ix)`: exact hit on a bound (bounds are inclusive, that bucket owns
        // it). `Err(ix)`: `ix` is the first bound strictly greater than `nanos`,
        // which is the bucket that owns it; `bounds.len()` is the overflow bucket.
        let (Ok(bucket) | Err(bucket)) = self.bounds.binary_search(&nanos);
        self.counts[bucket] += 1;
        self.total += 1;
        self.max_sample = self.max_sample.max(nanos);
    }

    /// Total number of samples recorded.
    #[must_use]
    #[inline]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Whether no samples have been recorded.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// The largest sample recorded, in nanoseconds (`0` if empty).
    #[must_use]
    #[inline]
    pub fn max_nanos(&self) -> u64 {
        self.max_sample
    }

    /// The bucket upper bounds (ascending), excluding the overflow bucket.
    #[must_use]
    #[inline]
    pub fn bounds(&self) -> &[u64] {
        &self.bounds
    }

    /// The estimated `p`th-percentile latency in nanoseconds, where `p` is in
    /// `1..=100`. Returns the upper bound of the bucket containing the
    /// percentile rank; a percentile that falls in the overflow bucket reports
    /// [`LatencyHistogram::max_nanos`]. Returns `0` when empty. `p` is clamped
    /// into `1..=100`.
    ///
    /// The rank is `ceil(p * total / 100)` (nearest-rank, integer math), so the
    /// result depends only on the recorded counts, never on sampling order.
    #[must_use]
    pub fn percentile_nanos(&self, p: u8) -> u64 {
        if self.total == 0 {
            return 0;
        }
        let p = u64::from(p.clamp(1, 100));
        // Nearest-rank with ceiling division: the smallest 1-based rank whose
        // cumulative count reaches `p%` of the samples.
        let rank = (p * self.total).div_ceil(100);
        let mut cumulative = 0;
        for (ix, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= rank {
                return match self.bounds.get(ix) {
                    Some(&bound) => bound,
                    None => self.max_sample,
                };
            }
        }
        self.max_sample
    }
}

/// Per-worker starvation detector based on consecutive idle streaks.
///
/// A worker whose idle streak reaches a configured threshold is considered
/// *starving* (chronically unable to find work). Feed observations with
/// [`StarvationDetector::record_idle`] / [`StarvationDetector::record_busy`];
/// the counts are a deterministic function of the observation sequence.
#[derive(Clone, Debug)]
pub struct StarvationDetector {
    /// Consecutive idle observations per worker.
    idle_streak: Vec<u32>,
    /// Idle-streak length at which a worker is deemed starving.
    threshold: u32,
}

impl StarvationDetector {
    /// Create a detector for `worker_count` workers that flags a worker once
    /// its consecutive idle streak reaches `threshold`. `threshold` is raised
    /// to at least `1`.
    #[must_use]
    pub fn new(worker_count: usize, threshold: u32) -> Self {
        Self {
            idle_streak: alloc::vec![0; worker_count],
            threshold: threshold.max(1),
        }
    }

    /// Number of workers tracked.
    #[must_use]
    #[inline]
    pub fn worker_count(&self) -> usize {
        self.idle_streak.len()
    }

    /// The configured starvation threshold.
    #[must_use]
    #[inline]
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Record that `worker` found no work this observation, extending its idle
    /// streak.
    ///
    /// # Panics
    /// Panics if `worker` is out of range.
    pub fn record_idle(&mut self, worker: usize) {
        self.idle_streak[worker] = self.idle_streak[worker].saturating_add(1);
    }

    /// Record that `worker` ran a job this observation, resetting its idle
    /// streak.
    ///
    /// # Panics
    /// Panics if `worker` is out of range.
    pub fn record_busy(&mut self, worker: usize) {
        self.idle_streak[worker] = 0;
    }

    /// The current idle streak of `worker`.
    ///
    /// # Panics
    /// Panics if `worker` is out of range.
    #[must_use]
    #[inline]
    pub fn idle_streak(&self, worker: usize) -> u32 {
        self.idle_streak[worker]
    }

    /// Whether `worker`'s idle streak has reached the starvation threshold.
    ///
    /// # Panics
    /// Panics if `worker` is out of range.
    #[must_use]
    #[inline]
    pub fn is_starving(&self, worker: usize) -> bool {
        self.idle_streak[worker] >= self.threshold
    }

    /// Number of workers currently starving.
    #[must_use]
    pub fn starving_workers(&self) -> usize {
        self.idle_streak
            .iter()
            .filter(|&&streak| streak >= self.threshold)
            .count()
    }
}

/// Work-stealing attempt/failure tallies.
///
/// A *failure* is a steal attempt that found an empty victim deque. The
/// per-mille failure rate is a coarse, scale-free indicator of pool
/// contention / under-subscription.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StealStats {
    /// Total steal attempts.
    attempts: u64,
    /// Steal attempts that found nothing to steal.
    failures: u64,
}

impl StealStats {
    /// Create empty steal statistics.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one steal attempt; `found_work` is `true` if it stole a job.
    pub fn record_attempt(&mut self, found_work: bool) {
        self.attempts += 1;
        if !found_work {
            self.failures += 1;
        }
    }

    /// Total steal attempts.
    #[must_use]
    #[inline]
    pub fn attempts(self) -> u64 {
        self.attempts
    }

    /// Steal attempts that found nothing.
    #[must_use]
    #[inline]
    pub fn failures(self) -> u64 {
        self.failures
    }

    /// Steal-failure rate in parts-per-thousand (`0` when there were no
    /// attempts). Integer `failures * 1000 / attempts`, so it is deterministic.
    #[must_use]
    pub fn failure_per_mille(self) -> u64 {
        (self.failures * 1000).checked_div(self.attempts).unwrap_or(0)
    }
}

/// A deterministic snapshot of pool health (design §24.8).
///
/// Every field is derived by [`PoolHealthMonitor::report`] from recorded
/// observations with integer-only math, so a given observation sequence always
/// yields the same report regardless of thread timing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HealthReport {
    /// Peak observed queue depth across all lanes.
    pub max_queue_depth: usize,
    /// Number of workers whose idle streak reached the starvation threshold.
    pub starving_workers: usize,
    /// Steal-failure rate in parts-per-thousand.
    pub steal_failure_per_mille: u64,
    /// Median (50th-percentile) job latency, in nanoseconds.
    pub p50_latency_nanos: u64,
    /// Tail (99th-percentile) job latency, in nanoseconds.
    pub p99_latency_nanos: u64,
    /// Longest single job latency observed, in nanoseconds.
    pub longest_job_nanos: u64,
    /// Total number of latency samples the report is derived from.
    pub latency_sample_count: u64,
}

/// Aggregates the health sub-metrics and renders a [`HealthReport`].
///
/// Combines a [`LatencyHistogram`], a [`StarvationDetector`], [`StealStats`],
/// and a running peak queue depth. The latency samples it ingests come from the
/// real-clock façade in [`health`](crate::health); everything this type does
/// with them afterwards is deterministic.
#[derive(Clone, Debug)]
pub struct PoolHealthMonitor {
    /// Job-latency histogram.
    latency: LatencyHistogram,
    /// Per-worker starvation detector.
    starvation: StarvationDetector,
    /// Steal attempt/failure tallies.
    steals: StealStats,
    /// Peak queue depth observed.
    max_queue_depth: usize,
}

impl PoolHealthMonitor {
    /// Create a monitor for `worker_count` workers, flagging starvation at
    /// `starvation_threshold` consecutive idle observations and bucketing
    /// latencies with the ascending `latency_bounds_nanos`.
    #[must_use]
    pub fn new(
        worker_count: usize,
        starvation_threshold: u32,
        latency_bounds_nanos: &[u64],
    ) -> Self {
        Self {
            latency: LatencyHistogram::new(latency_bounds_nanos),
            starvation: StarvationDetector::new(worker_count, starvation_threshold),
            steals: StealStats::new(),
            max_queue_depth: 0,
        }
    }

    /// Record one job-latency sample in nanoseconds.
    pub fn record_latency(&mut self, nanos: u64) {
        self.latency.record(nanos);
    }

    /// Record that `worker` found no work this observation.
    ///
    /// # Panics
    /// Panics if `worker` is out of range.
    pub fn record_idle(&mut self, worker: usize) {
        self.starvation.record_idle(worker);
    }

    /// Record that `worker` ran a job this observation.
    ///
    /// # Panics
    /// Panics if `worker` is out of range.
    pub fn record_busy(&mut self, worker: usize) {
        self.starvation.record_busy(worker);
    }

    /// Record one steal attempt; `found_work` is `true` if it stole a job.
    pub fn record_steal(&mut self, found_work: bool) {
        self.steals.record_attempt(found_work);
    }

    /// Observe the current total queue depth, updating the running peak.
    pub fn observe_queue_depth(&mut self, depth: usize) {
        self.max_queue_depth = self.max_queue_depth.max(depth);
    }

    /// Borrow the latency histogram.
    #[must_use]
    #[inline]
    pub fn latency(&self) -> &LatencyHistogram {
        &self.latency
    }

    /// Borrow the starvation detector.
    #[must_use]
    #[inline]
    pub fn starvation(&self) -> &StarvationDetector {
        &self.starvation
    }

    /// The steal statistics.
    #[must_use]
    #[inline]
    pub fn steals(&self) -> StealStats {
        self.steals
    }

    /// Render the current deterministic [`HealthReport`].
    #[must_use]
    pub fn report(&self) -> HealthReport {
        HealthReport {
            max_queue_depth: self.max_queue_depth,
            starving_workers: self.starvation.starving_workers(),
            steal_failure_per_mille: self.steals.failure_per_mille(),
            p50_latency_nanos: self.latency.percentile_nanos(50),
            p99_latency_nanos: self.latency.percentile_nanos(99),
            longest_job_nanos: self.latency.max_nanos(),
            latency_sample_count: self.latency.total(),
        }
    }
}
