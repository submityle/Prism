//! Per-lane and per-thread faceting of samples (§24.5).
//!
//! A single flat profile blends every thread together, so a main-thread stall
//! and a saturated I-O worker look alike. Faceting splits the sample buffer by
//! lane (thread class) or by thread id and folds each facet independently, so
//! "the main thread spends 40% in `wait_for_gpu`" becomes visible separately
//! from compute/I-O work (tasks §24.2 thread-class separation).
//!
//! Facets are returned in a deterministic order (lane facets in [`LaneKind`]
//! canonical order, thread facets by ascending thread id) and each embeds a
//! full [`FlatProfile`] over just that facet's samples. Pure `core`/`alloc`,
//! no `unsafe`.

extern crate alloc;

use alloc::vec::Vec;

use super::fold::{flat_profile, FlatProfile};
use super::sample::{LaneKind, StackSample};

/// A per-facet profile: the facet's total weight, estimated wall time, and a
/// [`FlatProfile`] folded over only that facet's samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneProfile {
    /// The lane this facet describes.
    pub lane: LaneKind,
    /// Number of samples (entries) in this facet.
    pub sample_count: usize,
    /// Summed weight (ticks) in this facet.
    pub total_weight: u64,
    /// Estimated wall time for this facet, `total_weight * interval_nanos`.
    pub total_nanos: u64,
    /// The flat hotspot table for this lane's samples.
    pub flat: FlatProfile,
}

/// A per-thread facet: like [`LaneProfile`] but keyed by thread id (and the
/// lane that thread was classified as).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadProfile {
    /// The thread id this facet describes.
    pub thread_id: u64,
    /// The lane this thread was classified as (from its samples).
    pub lane: LaneKind,
    /// Number of samples (entries) in this facet.
    pub sample_count: usize,
    /// Summed weight (ticks) in this facet.
    pub total_weight: u64,
    /// Estimated wall time for this facet, `total_weight * interval_nanos`.
    pub total_nanos: u64,
    /// The flat hotspot table for this thread's samples.
    pub flat: FlatProfile,
}

/// Facet a sample buffer by lane, returning one [`LaneProfile`] per lane that
/// has at least one sample, in [`LaneKind`] canonical order.
#[must_use]
pub fn facet_by_lane(samples: &[StackSample], interval_nanos: u64) -> Vec<LaneProfile> {
    let interval_nanos = interval_nanos.max(1);
    let mut out: Vec<LaneProfile> = Vec::new();
    for lane in LaneKind::all() {
        let facet: Vec<StackSample> = samples.iter().filter(|s| s.lane == lane).cloned().collect();
        if facet.is_empty() {
            continue;
        }
        let total_weight: u64 = facet.iter().map(|s| s.weight).sum();
        out.push(LaneProfile {
            lane,
            sample_count: facet.len(),
            total_weight,
            total_nanos: total_weight.saturating_mul(interval_nanos),
            flat: flat_profile(&facet, interval_nanos),
        });
    }
    out
}

/// Facet a sample buffer by thread id, returning one [`ThreadProfile`] per
/// distinct thread in ascending thread-id order.
///
/// A thread's reported [`LaneKind`] is taken from its first sample in capture
/// order; a well-behaved capture keeps a thread's lane stable.
#[must_use]
pub fn facet_by_thread(samples: &[StackSample], interval_nanos: u64) -> Vec<ThreadProfile> {
    let interval_nanos = interval_nanos.max(1);
    // Distinct thread ids in ascending order.
    let mut thread_ids: Vec<u64> = Vec::new();
    for sample in samples {
        if !thread_ids.contains(&sample.thread_id) {
            thread_ids.push(sample.thread_id);
        }
    }
    thread_ids.sort_unstable();

    let mut out: Vec<ThreadProfile> = Vec::with_capacity(thread_ids.len());
    for thread_id in thread_ids {
        let facet: Vec<StackSample> = samples
            .iter()
            .filter(|s| s.thread_id == thread_id)
            .cloned()
            .collect();
        // First sample in capture order fixes the reported lane.
        let lane = samples
            .iter()
            .find(|s| s.thread_id == thread_id)
            .map_or(LaneKind::Other, |s| s.lane);
        let total_weight: u64 = facet.iter().map(|s| s.weight).sum();
        out.push(ThreadProfile {
            thread_id,
            lane,
            sample_count: facet.len(),
            total_weight,
            total_nanos: total_weight.saturating_mul(interval_nanos),
            flat: flat_profile(&facet, interval_nanos),
        });
    }
    out
}
