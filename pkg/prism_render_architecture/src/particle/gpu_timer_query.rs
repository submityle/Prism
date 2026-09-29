//! Device-free `GPU` timestamp-query-pool contract for particle performance
//! markers (design §28).
//!
//! Profiling a particle frame on real hardware means writing a `GPU` timestamp
//! at the start and end of each measured span, resolving those timestamps into
//! a readback buffer, and converting the raw `tick` counts into wall-clock
//! nanoseconds using the device's timestamp period. This module owns only the
//! *contract* for that flow: how many queries a pool holds, how the two queries
//! of one measured span (a begin/end pair) map onto query slots, how many bytes
//! the resolve buffer needs, and the `tick`-to-nanosecond conversion — so the
//! render graph binds against a stable layout instead of re-deriving the query
//! index arithmetic next to the `wgpu` query set.
//!
//! It is deliberately orthogonal to [`super::perf_budget`], which plans budgets
//! and thresholds: this file never reasons about limits or overspend, only
//! about the physical timestamp-query pool layout and the measured elapsed
//! time it decodes. The scene/render crate owns the `wgpu` `QuerySet`, the
//! resolve buffer, and the readback; this zero-dependency module supplies the
//! authoritative slot layout and conversion so both sides agree on the `ABI`.
//!
//! Everything is pure integer and hand-written floating-point arithmetic and is
//! panic-free: an empty pool holds no pairs, out-of-range pairs are rejected by
//! `Option`, and a wrapped `tick` interval decodes to zero elapsed time rather
//! than a garbage or negative duration.

use alloc::vec::Vec;

use crate::particle::gpu_layout::U32_STRIDE;

/// Byte size of a single resolved `GPU` timestamp: a `u64` `tick` counter, so
/// `2 * 4 = 8` bytes. Expressed via [`U32_STRIDE`] so the `std430` scalar
/// stride is defined in exactly one place.
pub const TIMESTAMP_BYTES: usize = 2 * U32_STRIDE;

/// The number of query slots a measured span consumes: one for its begin
/// timestamp and one for its end timestamp.
pub const QUERIES_PER_PAIR: u32 = 2;

/// A pool of `GPU` timestamp query slots.
///
/// The pool holds `capacity` individual query slots. Measured spans claim slots
/// two at a time (a begin/end pair), so a pool with an odd capacity leaves one
/// trailing slot unusable for pairing; [`Self::pair_capacity`] reports how many
/// complete spans fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimestampQueryPool {
    capacity: u32,
}

impl TimestampQueryPool {
    /// Creates a pool holding `capacity` individual timestamp query slots.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self { capacity }
    }

    /// The total number of individual query slots in the pool.
    #[must_use]
    pub fn query_count(self) -> u32 {
        self.capacity
    }

    /// The number of complete begin/end pairs (measured spans) that fit in the
    /// pool. An odd trailing slot cannot form a pair and is excluded.
    #[must_use]
    pub fn pair_capacity(self) -> u32 {
        self.capacity / QUERIES_PER_PAIR
    }

    /// The query slot index that holds the *begin* timestamp of `pair`, or
    /// `None` when `pair` is outside [`Self::pair_capacity`].
    #[must_use]
    pub fn begin_query_index(self, pair: u32) -> Option<u32> {
        if pair < self.pair_capacity() {
            Some(pair.saturating_mul(QUERIES_PER_PAIR))
        } else {
            None
        }
    }

    /// The query slot index that holds the *end* timestamp of `pair`, or `None`
    /// when `pair` is outside [`Self::pair_capacity`].
    #[must_use]
    pub fn end_query_index(self, pair: u32) -> Option<u32> {
        self.begin_query_index(pair)
            .map(|begin| begin.saturating_add(1))
    }

    /// Total byte size of the resolve buffer that receives every timestamp in
    /// the pool: [`TIMESTAMP_BYTES`] per query slot. An empty pool needs no
    /// resolve storage and reports zero bytes.
    #[must_use]
    pub fn resolve_buffer_bytes(self) -> usize {
        let count = usize::try_from(self.capacity).unwrap_or(usize::MAX);
        TIMESTAMP_BYTES.saturating_mul(count)
    }

    /// Materializes every complete `(begin, end)` query-slot index pair, in
    /// ascending pair order.
    #[must_use]
    pub fn all_pairs(self) -> Vec<(u32, u32)> {
        let pairs = self.pair_capacity();
        let hint = usize::try_from(pairs).unwrap_or(usize::MAX);
        let mut out = Vec::with_capacity(hint);
        for pair in 0..pairs {
            let begin = pair.saturating_mul(QUERIES_PER_PAIR);
            out.push((begin, begin.saturating_add(1)));
        }
        out
    }
}

/// The device timestamp period: how many nanoseconds one raw `GPU` timestamp
/// `tick` represents. Reported by the driver as `f32` nanoseconds per `tick`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestampPeriod {
    nanos_per_tick: f32,
}

impl TimestampPeriod {
    /// Creates a period from the device's nanoseconds-per-`tick` value.
    #[must_use]
    pub fn new(nanos_per_tick: f32) -> Self {
        Self { nanos_per_tick }
    }

    /// The nanoseconds represented by one raw timestamp `tick`.
    #[must_use]
    pub fn nanos_per_tick(self) -> f32 {
        self.nanos_per_tick
    }

    /// Converts a begin/end `tick` interval into elapsed nanoseconds.
    ///
    /// A timestamp counter can wrap between the begin and end reads; when the
    /// end `tick` precedes the begin `tick` the interval is treated as zero
    /// rather than decoding a wrapped, meaningless duration.
    #[must_use]
    pub fn elapsed_nanos(self, begin_tick: u64, end_tick: u64) -> f32 {
        if end_tick < begin_tick {
            return 0.0;
        }
        let ticks = end_tick - begin_tick;
        ticks as f32 * self.nanos_per_tick
    }

    /// Converts a begin/end `tick` interval into elapsed milliseconds.
    ///
    /// This divides the nanosecond result by `1e6`; the same wrap guard as
    /// [`Self::elapsed_nanos`] applies.
    #[must_use]
    pub fn elapsed_millis(self, begin_tick: u64, end_tick: u64) -> f32 {
        self.elapsed_nanos(begin_tick, end_tick) / 1_000_000.0
    }
}

/// Aggregate statistics over a set of measured span durations (in nanoseconds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimerStats {
    /// The smallest sampled duration, in nanoseconds.
    pub min_nanos: f32,
    /// The largest sampled duration, in nanoseconds.
    pub max_nanos: f32,
    /// The arithmetic mean of the sampled durations, in nanoseconds.
    pub avg_nanos: f32,
    /// The number of samples the statistics were computed over.
    pub sample_count: u32,
}

impl TimerStats {
    /// Computes min/max/average over `samples` (nanosecond durations), or
    /// `None` for an empty sample set — there is no meaningful minimum,
    /// maximum, or mean of zero samples.
    #[must_use]
    pub fn from_samples(samples: &[f32]) -> Option<Self> {
        let (first, rest) = samples.split_first()?;
        let mut min_nanos = *first;
        let mut max_nanos = *first;
        let mut sum = *first;
        for &sample in rest {
            if sample < min_nanos {
                min_nanos = sample;
            }
            if sample > max_nanos {
                max_nanos = sample;
            }
            sum += sample;
        }
        let count = samples.len();
        let avg_nanos = sum / count as f32;
        let sample_count = u32::try_from(count).unwrap_or(u32::MAX);
        Some(Self {
            min_nanos,
            max_nanos,
            avg_nanos,
            sample_count,
        })
    }
}

/// A stable identifier for a named `GPU` performance marker (a measured span).
///
/// `name_hash` is a precomputed hash of the marker's human-readable name so the
/// `CPU`-side contract can identify a span without carrying its string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuMarker {
    /// The marker's dense slot identity (typically its pair index).
    pub id: u32,
    /// A precomputed hash of the marker's display name.
    pub name_hash: u32,
}

impl GpuMarker {
    /// Creates a marker from its slot `id` and precomputed `name_hash`.
    #[must_use]
    pub fn new(id: u32, name_hash: u32) -> Self {
        Self { id, name_hash }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn timestamp_bytes_is_two_scalars() {
        assert_eq!(TIMESTAMP_BYTES, 8);
        assert_eq!(TIMESTAMP_BYTES, 2 * U32_STRIDE);
    }

    #[test]
    fn query_count_and_pair_capacity() {
        let pool = TimestampQueryPool::new(8);
        assert_eq!(pool.query_count(), 8);
        assert_eq!(pool.pair_capacity(), 4);
    }

    #[test]
    fn odd_capacity_drops_the_trailing_slot() {
        let pool = TimestampQueryPool::new(7);
        assert_eq!(pool.query_count(), 7);
        assert_eq!(pool.pair_capacity(), 3);
    }

    #[test]
    fn pair_indices_are_interleaved() {
        let pool = TimestampQueryPool::new(6);
        assert_eq!(pool.begin_query_index(0), Some(0));
        assert_eq!(pool.end_query_index(0), Some(1));
        assert_eq!(pool.begin_query_index(1), Some(2));
        assert_eq!(pool.end_query_index(1), Some(3));
        assert_eq!(pool.begin_query_index(2), Some(4));
        assert_eq!(pool.end_query_index(2), Some(5));
    }

    #[test]
    fn out_of_range_pair_is_rejected() {
        let pool = TimestampQueryPool::new(6);
        assert_eq!(pool.pair_capacity(), 3);
        assert_eq!(pool.begin_query_index(3), None);
        assert_eq!(pool.end_query_index(3), None);
        assert_eq!(pool.begin_query_index(u32::MAX), None);
    }

    #[test]
    fn empty_pool_holds_no_pairs() {
        let pool = TimestampQueryPool::new(0);
        assert_eq!(pool.pair_capacity(), 0);
        assert_eq!(pool.begin_query_index(0), None);
        assert!(pool.all_pairs().is_empty());
        assert_eq!(pool.resolve_buffer_bytes(), 0);
    }

    #[test]
    fn resolve_buffer_bytes_is_eight_per_query() {
        assert_eq!(TimestampQueryPool::new(8).resolve_buffer_bytes(), 64);
        assert_eq!(TimestampQueryPool::new(1).resolve_buffer_bytes(), 8);
        assert_eq!(TimestampQueryPool::new(256).resolve_buffer_bytes(), 2048);
    }

    #[test]
    fn all_pairs_enumerates_every_span() {
        let pool = TimestampQueryPool::new(6);
        assert_eq!(pool.all_pairs(), alloc::vec![(0, 1), (2, 3), (4, 5)]);
    }

    #[test]
    fn elapsed_nanos_scales_by_period() {
        let period = TimestampPeriod::new(2.5);
        assert!(approx(period.nanos_per_tick(), 2.5));
        assert!(approx(period.elapsed_nanos(100, 140), 100.0));
        assert!(approx(period.elapsed_nanos(0, 0), 0.0));
    }

    #[test]
    fn elapsed_nanos_guards_wraparound() {
        let period = TimestampPeriod::new(1.0);
        assert!(approx(period.elapsed_nanos(500, 100), 0.0));
    }

    #[test]
    fn elapsed_millis_divides_by_one_million() {
        let period = TimestampPeriod::new(1.0);
        // 3_000_000 ticks * 1 ns = 3 ms.
        assert!(approx(period.elapsed_millis(0, 3_000_000), 3.0));
        assert!(approx(period.elapsed_millis(9, 4), 0.0));
    }

    #[test]
    fn period_conversion_precision_within_tolerance() {
        // A common desktop period: ~0.833... ns/tick over 1200 ticks ~= 1000 ns.
        let period = TimestampPeriod::new(0.833_333_3);
        let nanos = period.elapsed_nanos(0, 1200);
        assert!((nanos - 1000.0).abs() < 1.0);
    }

    #[test]
    fn stats_over_empty_set_is_none() {
        assert_eq!(TimerStats::from_samples(&[]), None);
    }

    #[test]
    fn stats_compute_min_max_avg() {
        let stats = TimerStats::from_samples(&[4.0, 1.0, 7.0, 2.0]).unwrap();
        assert!(approx(stats.min_nanos, 1.0));
        assert!(approx(stats.max_nanos, 7.0));
        assert!(approx(stats.avg_nanos, 3.5));
        assert_eq!(stats.sample_count, 4);
    }

    #[test]
    fn stats_single_sample() {
        let stats = TimerStats::from_samples(&[5.0]).unwrap();
        assert!(approx(stats.min_nanos, 5.0));
        assert!(approx(stats.max_nanos, 5.0));
        assert!(approx(stats.avg_nanos, 5.0));
        assert_eq!(stats.sample_count, 1);
    }

    #[test]
    fn gpu_marker_carries_id_and_hash() {
        let marker = GpuMarker::new(3, 0xDEAD_BEEF);
        assert_eq!(marker.id, 3);
        assert_eq!(marker.name_hash, 0xDEAD_BEEF);
    }
}
