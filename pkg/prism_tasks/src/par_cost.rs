//! Cost-calibrated adaptive granularity for the data-parallel facade (§24.4).
//!
//! The chunked primitives in [`parallel`](crate::parallel) and
//! [`par_iter`](crate::par_iter) size their chunks from the input length and
//! worker count alone. That is robust but blind to how expensive each item
//! actually is: a range of a million trivial increments and a range of a
//! thousand heavy solves get split by the same rule, so cheap work is
//! over-scheduled (task overhead dominates) and expensive work can be
//! under-split (one slow chunk stalls a worker).
//!
//! This module closes that gap. [`CostModel`] turns a *measured* per-item cost
//! into a chunk grain that targets a desired per-chunk run time, and the
//! [`TaskPool::par_for_calibrated`] / [`TaskPool::par_reduce_calibrated`]
//! facades calibrate that cost at runtime by timing a small probe prefix of the
//! real work (the probe is useful work, not a throwaway sample) before fanning
//! the remainder out across the pool.
//!
//! # Determinism
//! Calibration reads a wall clock, so the chosen grain — and therefore the
//! number of chunks — varies run to run. Results are still correct (`body`
//! runs on every index exactly once; the reduction visits every mapped value),
//! but [`TaskPool::par_reduce_calibrated`] is **not** bit-reproducible for a
//! non-associative `combine`. When you need a result that is identical across
//! machines and worker counts, use the deterministic path
//! ([`TaskPool::deterministic_reduce`]) instead; this facade is the throughput
//! path for the common non-lockstep case.

use core::ops::Range;
use std::time::Instant;

use crate::TaskPool;

/// Maps a measured per-item cost to a chunk grain for calibrated parallelism.
///
/// The model balances two competing pressures:
/// - **Amortize scheduling.** A chunk should run for about
///   [`target_chunk_nanos`](CostModel::target_chunk_nanos) so that per-task
///   enqueue/steal overhead stays a small fraction of useful work. This sets a
///   *lower* bound on the grain from the measured item cost.
/// - **Balance load.** There should be at least
///   `workers * oversubscription` chunks so the work-stealing scheduler can
///   even out uneven items. This sets an *upper* bound on the grain.
///
/// A hard [`min_grain`](CostModel::min_grain) floor keeps chunks from
/// degenerating into near-empty tasks even when items are almost free, and the
/// grain is finally capped at the input length.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CostModel {
    /// Target wall-clock run time of a single chunk, in nanoseconds.
    pub target_chunk_nanos: u64,
    /// Hard lower bound on chunk size, regardless of how cheap an item is.
    pub min_grain: usize,
    /// Number of items timed inline to estimate the per-item cost.
    pub probe_items: usize,
    /// Desired number of chunks per worker for steal-based load balancing.
    pub oversubscription: usize,
}

impl CostModel {
    /// A balanced default: 100 µs chunks, a 64-item floor, a 256-item probe,
    /// and 4× oversubscription.
    pub const DEFAULT: Self = Self {
        target_chunk_nanos: 100_000,
        min_grain: 64,
        probe_items: 256,
        oversubscription: 4,
    };

    /// Create a model from the balanced [`CostModel::DEFAULT`].
    #[must_use]
    pub const fn new() -> Self {
        Self::DEFAULT
    }

    /// Override the target per-chunk run time (nanoseconds); clamped to ≥ 1.
    #[must_use]
    pub const fn with_target_chunk_nanos(mut self, nanos: u64) -> Self {
        self.target_chunk_nanos = if nanos == 0 { 1 } else { nanos };
        self
    }

    /// Override the hard minimum grain; clamped to ≥ 1.
    #[must_use]
    pub const fn with_min_grain(mut self, grain: usize) -> Self {
        self.min_grain = if grain == 0 { 1 } else { grain };
        self
    }

    /// Override the inline probe length; clamped to ≥ 1.
    #[must_use]
    pub const fn with_probe_items(mut self, items: usize) -> Self {
        self.probe_items = if items == 0 { 1 } else { items };
        self
    }

    /// Override the per-worker oversubscription factor; clamped to ≥ 1.
    #[must_use]
    pub const fn with_oversubscription(mut self, factor: usize) -> Self {
        self.oversubscription = if factor == 0 { 1 } else { factor };
        self
    }

    /// Compute the chunk grain for `len` items across `workers` workers given a
    /// measured `cost_per_item_nanos`.
    ///
    /// Returns a value in `[1, len]` (or [`min_grain`](CostModel::min_grain)
    /// clamped to `len` when `len == 0` is not the case). Pure and
    /// deterministic in its inputs, so it is unit-testable without a clock.
    #[must_use]
    pub fn grain(&self, len: usize, workers: usize, cost_per_item_nanos: u64) -> usize {
        if len == 0 {
            return 1;
        }
        let workers = workers.max(1);
        let cost = cost_per_item_nanos.max(1);
        // Items whose combined cost ≈ one target chunk duration.
        let cost_grain = (self.target_chunk_nanos / cost).max(1);
        let cost_grain = usize::try_from(cost_grain).unwrap_or(usize::MAX);
        // Coarsest grain that still yields `workers * oversubscription` chunks.
        let target_chunks = workers.saturating_mul(self.oversubscription).max(1);
        let balance_cap = len.div_ceil(target_chunks).max(1);
        // Amortize first, then never exceed the balance cap, then honour the
        // hard floor, and finally never exceed the input length.
        cost_grain.min(balance_cap).max(self.min_grain).min(len)
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TaskPool {
    /// Run `body` over every index in `range`, sizing chunks from a measured
    /// per-item cost (§24.4 calibrated data-parallel facade).
    ///
    /// A prefix of up to [`CostModel::probe_items`] indices is run inline and
    /// timed to estimate the per-item cost; the remaining indices are fanned
    /// out through a structured [`scope`](TaskPool::scope) in calibrated
    /// chunks. Every index is visited exactly once, so the observable effect is
    /// identical to a serial `for i in range { body(i) }` for a pure `body`.
    pub fn par_for_calibrated<F>(&self, range: Range<usize>, model: &CostModel, body: F)
    where
        F: Fn(usize) + Sync,
    {
        let start = range.start;
        let end = range.end;
        if start >= end {
            return;
        }
        let len = end - start;
        let workers = self.worker_count().max(1);

        // One worker (or the single-threaded fallback): no point calibrating.
        if workers <= 1 {
            for i in start..end {
                body(i);
            }
            return;
        }

        // Probe: run and time the leading items as genuine work.
        let probe = model.probe_items.max(1).min(len);
        let probe_end = start + probe;
        let probe_start = Instant::now();
        for i in start..probe_end {
            body(i);
        }
        let elapsed = probe_start.elapsed().as_nanos();
        if probe_end >= end {
            return;
        }

        let per_item = (elapsed / probe as u128).max(1);
        let per_item = u64::try_from(per_item).unwrap_or(u64::MAX);
        let remaining = end - probe_end;
        let grain = model.grain(remaining, workers, per_item);

        let body = &body;
        self.scope(|s| {
            let mut chunk_start = probe_end;
            while chunk_start < end {
                let chunk_end = chunk_start.saturating_add(grain).min(end);
                s.spawn(move || {
                    for i in chunk_start..chunk_end {
                        body(i);
                    }
                });
                chunk_start = chunk_end;
            }
        });
    }

    /// Map every index in `range` and fold the results, sizing chunks from a
    /// measured per-item cost (§24.4 calibrated reduce facade).
    ///
    /// `identity` seeds each chunk's accumulator, `map` turns an index into a
    /// value, and `combine` merges values. Partial chunk results are merged in
    /// a fixed tree order *for a given chunk count*, but because calibration
    /// chooses the chunk count from a timing measurement, the merge shape is
    /// not reproducible across runs — see the [module docs](self) and prefer
    /// [`TaskPool::deterministic_reduce`] when bit-reproducibility is required.
    #[must_use]
    pub fn par_reduce_calibrated<R, ID, M, C>(
        &self,
        range: Range<usize>,
        model: &CostModel,
        identity: ID,
        map: M,
        combine: C,
    ) -> R
    where
        R: Send,
        ID: Fn() -> R + Sync,
        M: Fn(usize) -> R + Sync,
        C: Fn(R, R) -> R + Sync,
    {
        let start = range.start;
        let end = range.end;
        if start >= end {
            return identity();
        }
        let len = end - start;
        let workers = self.worker_count().max(1);

        let fold_serial = |lo: usize, hi: usize| {
            let mut acc = identity();
            for i in lo..hi {
                acc = combine(acc, map(i));
            }
            acc
        };

        if workers <= 1 {
            return fold_serial(start, end);
        }

        let probe = model.probe_items.max(1).min(len);
        let probe_end = start + probe;
        let probe_start = Instant::now();
        let probe_acc = fold_serial(start, probe_end);
        let elapsed = probe_start.elapsed().as_nanos();
        if probe_end >= end {
            return probe_acc;
        }

        let per_item = (elapsed / probe as u128).max(1);
        let per_item = u64::try_from(per_item).unwrap_or(u64::MAX);
        let remaining = end - probe_end;
        let grain = model.grain(remaining, workers, per_item);
        let num_chunks = remaining.div_ceil(grain);

        let map = &map;
        let identity = &identity;
        let combine = &combine;
        let mut partials: Vec<Option<R>> = (0..num_chunks).map(|_| None).collect();
        self.scope(|s| {
            for (chunk_index, slot) in partials.iter_mut().enumerate() {
                let chunk_start = probe_end + chunk_index * grain;
                let chunk_end = chunk_start.saturating_add(grain).min(end);
                s.spawn(move || {
                    let mut acc = identity();
                    for i in chunk_start..chunk_end {
                        acc = combine(acc, map(i));
                    }
                    *slot = Some(acc);
                });
            }
        });

        // Fold the probe result and the chunk partials with a tree-shaped order
        // (deterministic for a fixed chunk count).
        let mut values: Vec<R> = Vec::with_capacity(num_chunks + 1);
        values.push(probe_acc);
        for partial in partials {
            values.push(partial.expect("every chunk ran under the scope"));
        }
        while values.len() > 1 {
            let mut next = Vec::with_capacity(values.len().div_ceil(2));
            let mut iter = values.into_iter();
            while let Some(a) = iter.next() {
                match iter.next() {
                    Some(b) => next.push(combine(a, b)),
                    None => next.push(a),
                }
            }
            values = next;
        }
        values.into_iter().next().expect("non-empty values")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grain_amortizes_cheap_items_at_floor() {
        let m = CostModel::DEFAULT;
        // 1 ns per item, target 100_000 ns -> cost_grain = 100_000, but a large
        // range with 4 workers caps it by load balance well below that; the
        // floor still guarantees at least `min_grain`.
        let g = m.grain(10_000, 4, 1);
        assert!(g >= m.min_grain, "grain {g} below floor");
        assert!(g <= 10_000);
    }

    #[test]
    fn grain_splits_expensive_items_finely() {
        let m = CostModel::DEFAULT;
        // 50_000 ns per item, target 100_000 ns -> ~2 items per chunk, but the
        // floor forces min_grain; the point is it never exceeds balance_cap.
        let workers = 8;
        let g = m.grain(1_000_000, workers, 50_000);
        let balance_cap = (1_000_000usize).div_ceil(workers * m.oversubscription);
        assert!(g <= balance_cap.max(m.min_grain));
    }

    #[test]
    fn grain_respects_balance_cap() {
        let m = CostModel::DEFAULT.with_min_grain(1);
        // Extremely cheap items would want a huge grain, but we must keep
        // `workers * oversubscription` chunks.
        let workers = 4;
        let g = m.grain(100_000, workers, 1);
        let balance_cap = (100_000usize).div_ceil(workers * m.oversubscription);
        assert_eq!(g, balance_cap);
    }

    #[test]
    fn grain_never_exceeds_len() {
        let m = CostModel::DEFAULT;
        assert!(m.grain(10, 8, 1) <= 10);
    }

    #[test]
    fn builders_clamp_to_one() {
        let m = CostModel::new()
            .with_target_chunk_nanos(0)
            .with_min_grain(0)
            .with_probe_items(0)
            .with_oversubscription(0);
        assert_eq!(m.target_chunk_nanos, 1);
        assert_eq!(m.min_grain, 1);
        assert_eq!(m.probe_items, 1);
        assert_eq!(m.oversubscription, 1);
    }

    #[test]
    fn par_for_calibrated_visits_every_index() {
        let pool = TaskPool::with_threads(4);
        let n = 50_000usize;
        let hits: Vec<std::sync::atomic::AtomicU32> = (0..n)
            .map(|_| std::sync::atomic::AtomicU32::new(0))
            .collect();
        pool.par_for_calibrated(0..n, &CostModel::DEFAULT, |i| {
            hits[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        assert!(hits
            .iter()
            .all(|h| h.load(std::sync::atomic::Ordering::Relaxed) == 1));
    }

    #[test]
    fn par_for_calibrated_handles_tiny_and_empty_ranges() {
        let pool = TaskPool::with_threads(4);
        let counter = std::sync::atomic::AtomicUsize::new(0);
        pool.par_for_calibrated(0..0, &CostModel::DEFAULT, |_| {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 0);
        pool.par_for_calibrated(5..8, &CostModel::DEFAULT, |_| {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 3);
    }

    #[test]
    fn par_reduce_calibrated_matches_serial_sum() {
        let pool = TaskPool::with_threads(4);
        let n = 100_000usize;
        let got: u64 = pool.par_reduce_calibrated(
            0..n,
            &CostModel::DEFAULT,
            || 0u64,
            |i| i as u64,
            |a, b| a + b,
        );
        let expected: u64 = (0..n as u64).sum();
        assert_eq!(got, expected);
    }

    #[test]
    fn par_reduce_calibrated_small_range_uses_probe_result() {
        let pool = TaskPool::with_threads(4);
        // Range shorter than the probe window returns the inline fold directly.
        let got: u64 = pool.par_reduce_calibrated(
            0..10,
            &CostModel::DEFAULT,
            || 0u64,
            |i| i as u64,
            |a, b| a + b,
        );
        assert_eq!(got, (0..10u64).sum());
    }
}
