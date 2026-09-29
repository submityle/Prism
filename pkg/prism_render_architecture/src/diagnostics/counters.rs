//! Sliding-window frame-pacing statistics.
//!
//! Frame pacing is judged over a *window*, not a single frame: one long frame
//! is noise, a rising `p95` over the last few hundred frames is a real
//! regression. This module keeps the most recent [`FrameCounters`] samples in a
//! fixed-capacity ring and folds them into average / peak / `p95` aggregates
//! for `CPU` frame time, `GPU` frame time, and `GPU` memory residency.
//!
//! The `p95` is computed by sorting the window (no external dependency) and
//! taking the nearest-rank sample, so it is exact for the window and fully
//! deterministic. `f32` inputs are handled `NaN`-safely: non-finite timing
//! samples are dropped from the timing aggregates rather than poisoning them,
//! and an all-non-finite (or empty) window yields a zeroed aggregate.

use alloc::vec::Vec;

use super::FrameCounters;

/// Average / peak / `p95` aggregate of an `f32` quantity over the window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aggregate {
    /// Arithmetic mean of the finite samples.
    pub average: f32,
    /// Largest finite sample.
    pub peak: f32,
    /// Nearest-rank 95th-percentile finite sample.
    pub p95: f32,
}

impl Aggregate {
    /// The zeroed aggregate returned for an empty / all-non-finite input.
    pub const ZERO: Self = Self {
        average: 0.0,
        peak: 0.0,
        p95: 0.0,
    };
}

impl Default for Aggregate {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Average / peak / `p95` aggregate of a `u64` quantity over the window.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MemoryAggregate {
    /// Arithmetic mean (integer, truncated toward zero).
    pub average: u64,
    /// Largest sample.
    pub peak: u64,
    /// Nearest-rank 95th-percentile sample.
    pub p95: u64,
}

impl MemoryAggregate {
    /// The zeroed aggregate returned for an empty window.
    pub const ZERO: Self = Self {
        average: 0,
        peak: 0,
        p95: 0,
    };
}

/// The folded statistics of a [`FrameWindow`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStats {
    /// Number of samples the window held when folded.
    pub sample_count: usize,
    /// `CPU` frame-time aggregate in milliseconds.
    pub cpu_frame_ms: Aggregate,
    /// `GPU` frame-time aggregate in milliseconds.
    pub gpu_frame_ms: Aggregate,
    /// `GPU` resident-memory aggregate in bytes.
    pub gpu_memory_bytes: MemoryAggregate,
}

impl FrameStats {
    /// The stats of an empty window: zero samples, zeroed aggregates.
    pub const EMPTY: Self = Self {
        sample_count: 0,
        cpu_frame_ms: Aggregate::ZERO,
        gpu_frame_ms: Aggregate::ZERO,
        gpu_memory_bytes: MemoryAggregate::ZERO,
    };
}

impl Default for FrameStats {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Nearest-rank index for the 95th percentile of `n` sorted samples.
///
/// Uses integer rounding of `(n - 1) * 0.95` to stay exact and dependency-free:
/// `((n - 1) * 95 + 50) / 100`. For `n == 1` this is `0`.
#[must_use]
const fn p95_index(n: usize) -> usize {
    ((n - 1) * 95 + 50) / 100
}

/// Folds a set of finite `f32` samples into an [`Aggregate`].
///
/// The caller passes only finite values (non-finite samples are dropped
/// upstream); an empty slice yields [`Aggregate::ZERO`].
fn aggregate_f32(mut values: Vec<f32>) -> Aggregate {
    let n = values.len();
    if n == 0 {
        return Aggregate::ZERO;
    }
    let sum: f32 = values.iter().copied().sum();
    let average = sum / n as f32;
    // `total_cmp` gives a total order without an `unwrap`; values are finite.
    values.sort_by(f32::total_cmp);
    let peak = values[n - 1];
    let p95 = values[p95_index(n)];
    Aggregate { average, peak, p95 }
}

/// Folds `u64` samples into a [`MemoryAggregate`]; empty yields
/// [`MemoryAggregate::ZERO`].
fn aggregate_u64(mut values: Vec<u64>) -> MemoryAggregate {
    let n = values.len();
    if n == 0 {
        return MemoryAggregate::ZERO;
    }
    // Accumulate in `u128` so a long window of large residencies cannot overflow.
    let sum: u128 = values.iter().map(|&v| v as u128).sum();
    let average = (sum / n as u128) as u64;
    values.sort_unstable();
    let peak = values[n - 1];
    let p95 = values[p95_index(n)];
    MemoryAggregate { average, peak, p95 }
}

/// A fixed-capacity sliding window of [`FrameCounters`] samples.
///
/// Holds at most `capacity` of the most recent samples; pushing into a full
/// window drops the oldest. [`FrameWindow::stats`] folds the current contents
/// into a [`FrameStats`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameWindow {
    capacity: usize,
    samples: Vec<FrameCounters>,
    head: usize,
    recorded: u64,
}

impl FrameWindow {
    /// Creates an empty window retaining up to `capacity` samples.
    ///
    /// A `capacity` of `0` is clamped up to `1`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            samples: Vec::with_capacity(capacity),
            head: 0,
            recorded: 0,
        }
    }

    /// Maximum number of samples retained.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of samples currently stored (`<= capacity`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` when no samples are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Returns `true` when the window is at capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.samples.len() == self.capacity
    }

    /// Lifetime count of pushes, including those that dropped older samples.
    #[must_use]
    pub const fn total_recorded(&self) -> u64 {
        self.recorded
    }

    /// Records `sample`, dropping the oldest when the window is full.
    pub fn push(&mut self, sample: FrameCounters) {
        if self.samples.len() < self.capacity {
            self.samples.push(sample);
        } else {
            self.samples[self.head] = sample;
            self.head = (self.head + 1) % self.capacity;
        }
        self.recorded = self.recorded.saturating_add(1);
    }

    /// Iterates stored samples oldest-first.
    pub fn iter(&self) -> impl Iterator<Item = &FrameCounters> + '_ {
        let len = self.samples.len();
        let head = self.head;
        let capacity = self.capacity;
        (0..len).map(move |i| &self.samples[(head + i) % capacity])
    }

    /// Folds the current window into average / peak / `p95` statistics.
    ///
    /// Non-finite `cpu_frame_ms` / `gpu_frame_ms` samples are excluded from the
    /// timing aggregates. `sample_count` still reflects the raw number of
    /// stored samples. An empty window yields [`FrameStats::EMPTY`].
    #[must_use]
    pub fn stats(&self) -> FrameStats {
        let mut cpu = Vec::new();
        let mut gpu = Vec::new();
        let mut mem = Vec::new();
        for c in self.iter() {
            if c.cpu_frame_ms.is_finite() {
                cpu.push(c.cpu_frame_ms);
            }
            if c.gpu_frame_ms.is_finite() {
                gpu.push(c.gpu_frame_ms);
            }
            mem.push(c.gpu_memory_bytes);
        }
        FrameStats {
            sample_count: self.samples.len(),
            cpu_frame_ms: aggregate_f32(cpu),
            gpu_frame_ms: aggregate_f32(gpu),
            gpu_memory_bytes: aggregate_u64(mem),
        }
    }

    /// Removes all stored samples, preserving the lifetime counter.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.head = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::EPS;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    fn sample(idx: u64, cpu: f32, gpu: f32, mem: u64) -> FrameCounters {
        FrameCounters::new(idx, cpu, gpu, mem)
    }

    #[test]
    fn empty_window_yields_empty_stats() {
        let w = FrameWindow::new(8);
        assert!(w.is_empty());
        let s = w.stats();
        assert_eq!(s, FrameStats::EMPTY);
        assert_eq!(s.sample_count, 0);
        assert!(approx(s.cpu_frame_ms.average, 0.0));
        assert_eq!(s.gpu_memory_bytes, MemoryAggregate::ZERO);
    }

    #[test]
    fn p95_index_boundaries() {
        assert_eq!(p95_index(1), 0);
        assert_eq!(p95_index(2), 1);
        assert_eq!(p95_index(20), 18);
        assert_eq!(p95_index(100), 94);
    }

    #[test]
    fn single_sample_aggregates_to_itself() {
        let mut w = FrameWindow::new(4);
        w.push(sample(0, 12.5, 9.0, 2048));
        let s = w.stats();
        assert_eq!(s.sample_count, 1);
        assert!(approx(s.cpu_frame_ms.average, 12.5));
        assert!(approx(s.cpu_frame_ms.peak, 12.5));
        assert!(approx(s.cpu_frame_ms.p95, 12.5));
        assert_eq!(s.gpu_memory_bytes.average, 2048);
        assert_eq!(s.gpu_memory_bytes.peak, 2048);
        assert_eq!(s.gpu_memory_bytes.p95, 2048);
    }

    #[test]
    fn averages_peaks_and_p95_over_window() {
        let mut w = FrameWindow::new(100);
        // 1..=100 ms on CPU, doubled on GPU, memory = idx * 1000.
        for i in 1..=100u64 {
            w.push(sample(i, i as f32, (i as f32) * 2.0, i * 1000));
        }
        let s = w.stats();
        assert_eq!(s.sample_count, 100);
        // mean of 1..=100 is 50.5.
        assert!(approx(s.cpu_frame_ms.average, 50.5));
        assert!(approx(s.cpu_frame_ms.peak, 100.0));
        // p95 nearest-rank index for n=100 is 94 -> the 95th value (95.0).
        assert!(approx(s.cpu_frame_ms.p95, 95.0));
        assert!(approx(s.gpu_frame_ms.peak, 200.0));
        assert!(approx(s.gpu_frame_ms.p95, 190.0));
        assert_eq!(s.gpu_memory_bytes.peak, 100_000);
        assert_eq!(s.gpu_memory_bytes.p95, 95_000);
        // mean of idx*1000 for 1..=100 is 50_500.
        assert_eq!(s.gpu_memory_bytes.average, 50_500);
    }

    #[test]
    fn window_overflow_keeps_recent_samples() {
        let mut w = FrameWindow::new(3);
        for i in 0..6u64 {
            w.push(sample(i, i as f32, 0.0, 0));
        }
        assert_eq!(w.len(), 3);
        assert_eq!(w.total_recorded(), 6);
        // Only frames 3,4,5 remain.
        assert_eq!(
            w.iter().map(|c| c.frame_index).collect::<Vec<_>>(),
            [3, 4, 5]
        );
        let s = w.stats();
        assert!(approx(s.cpu_frame_ms.average, 4.0)); // (3+4+5)/3
        assert!(approx(s.cpu_frame_ms.peak, 5.0));
    }

    #[test]
    fn peak_is_order_independent() {
        let mut ascending = FrameWindow::new(8);
        let mut descending = FrameWindow::new(8);
        for i in 0..8u64 {
            ascending.push(sample(i, i as f32, 0.0, 0));
        }
        for i in (0..8u64).rev() {
            descending.push(sample(i, i as f32, 0.0, 0));
        }
        assert!(approx(
            ascending.stats().cpu_frame_ms.peak,
            descending.stats().cpu_frame_ms.peak
        ));
        assert!(approx(
            ascending.stats().cpu_frame_ms.p95,
            descending.stats().cpu_frame_ms.p95
        ));
    }

    #[test]
    fn nan_and_infinite_timings_are_dropped() {
        let mut w = FrameWindow::new(8);
        w.push(sample(0, 10.0, 10.0, 100));
        w.push(sample(1, f32::NAN, f32::INFINITY, 200));
        w.push(sample(2, 20.0, 30.0, 300));
        let s = w.stats();
        // Raw sample count still 3.
        assert_eq!(s.sample_count, 3);
        // CPU aggregate over the two finite values 10 and 20.
        assert!(approx(s.cpu_frame_ms.average, 15.0));
        assert!(approx(s.cpu_frame_ms.peak, 20.0));
        // Memory has no NaN concept; all three counted.
        assert_eq!(s.gpu_memory_bytes.peak, 300);
        assert_eq!(s.gpu_memory_bytes.average, 200); // (100+200+300)/3
    }

    #[test]
    fn all_nonfinite_timings_yield_zero_aggregate() {
        let mut w = FrameWindow::new(4);
        w.push(sample(0, f32::NAN, f32::NEG_INFINITY, 10));
        w.push(sample(1, f32::INFINITY, f32::NAN, 20));
        let s = w.stats();
        assert_eq!(s.cpu_frame_ms, Aggregate::ZERO);
        assert_eq!(s.gpu_frame_ms, Aggregate::ZERO);
        // Memory still aggregates.
        assert_eq!(s.gpu_memory_bytes.peak, 20);
    }

    #[test]
    fn stats_are_deterministic() {
        let build = || {
            let mut w = FrameWindow::new(16);
            for i in 0..40u64 {
                w.push(sample(i, (i % 7) as f32, (i % 5) as f32, i * 8));
            }
            w.stats()
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn memory_average_uses_wide_accumulator() {
        let mut w = FrameWindow::new(4);
        // Values near u64::MAX would overflow a u64 sum; u128 accumulation is safe.
        let big = u64::MAX / 2;
        for i in 0..4u64 {
            w.push(sample(i, 0.0, 0.0, big));
        }
        let s = w.stats();
        assert_eq!(s.gpu_memory_bytes.average, big);
        assert_eq!(s.gpu_memory_bytes.peak, big);
    }

    #[test]
    fn clear_resets_contents_only() {
        let mut w = FrameWindow::new(4);
        w.push(sample(0, 1.0, 1.0, 1));
        w.push(sample(1, 2.0, 2.0, 2));
        w.clear();
        assert!(w.is_empty());
        assert_eq!(w.total_recorded(), 2);
        assert_eq!(w.stats(), FrameStats::EMPTY);
    }

    #[test]
    fn zero_capacity_clamped() {
        let mut w = FrameWindow::new(0);
        assert_eq!(w.capacity(), 1);
        w.push(sample(0, 5.0, 5.0, 5));
        w.push(sample(1, 7.0, 7.0, 7));
        assert_eq!(w.len(), 1);
        assert!(approx(w.stats().cpu_frame_ms.average, 7.0));
    }
}
