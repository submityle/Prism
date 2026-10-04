//! Dormant-entity id-space census (design §13.2 / §16.6).
//!
//! Entity dormancy (design §13.2, [`DormancySet`]) pulls settled entities out
//! of the active schedule so per-frame cost tracks the *awake* population, not
//! the total one. How healthy that pool is depends not just on its size but on
//! *where its entities live in the id-space*: a dormant set that occupies a
//! single contiguous block of slot indices serialises, migrates (design §13.1)
//! and iterates far more cheaply than the same count scattered as isolated
//! holes across the whole 32-bit index range, and a dormant set dominated by
//! high-generation handles signals a heavily *recycled* slot pool (streaming
//! churn, design §13.1) rather than a stable settled population.
//!
//! The [`DormancySet`] itself only answers "how many" and "which entities".
//! This report reads that set once and makes the *shape* of the dormant
//! population legible:
//!
//! * **span & density** — the min/max slot index spanned by the dormant set
//!   and how densely those slots are filled (`count / span`), distinguishing a
//!   tightly-pooled block from a sparse scatter;
//! * **contiguity** — the number of maximal runs of consecutive slot indices
//!   and the longest such run, so a pool-allocated dormant block (one run)
//!   reads differently from fragmentation (many runs);
//! * **index histogram** — a bit-width bucketing of slot indices
//!   (`[2^(w-1), 2^w)`), separating long-settled low-index entities from
//!   recently-spawned high-index ones;
//! * **generation histogram & recycle depth** — a bit-width bucketing of
//!   generations plus the count of recycled slots (`generation > 1`), exposing
//!   slot churn;
//! * **wake churn** — the pending wake events relative to the dormant
//!   population, i.e. how much of the recently-dormant set is about to be
//!   re-admitted this frame.
//!
//! All ordered output is deterministic (buckets ascending by bit width,
//! contiguity computed from sorted indices), so the census is independent of
//! the set's internal hash order (design §14). It owns no
//! [`World`](crate::world::World) state and reads the set in `O(n log n)` for
//! the contiguity sort.

use alloc::vec::Vec;

use crate::partition::dormant::DormancySet;

/// Number of distinct bit-width buckets for a 32-bit value: widths `0..=32`.
const BIT_WIDTH_BUCKETS: usize = 33;

/// Number of significant bits in `x` (`floor(log2(x)) + 1` for `x >= 1`, and
/// `0` for `x == 0`) — the `no_std`-clean integer magnitude bucket key.
#[inline]
fn bit_width(x: u32) -> u32 {
    u32::BITS - x.leading_zeros()
}

/// Inclusive low / exclusive high bound of the index range a bit width covers:
/// width `0` is the singleton `{0}` (`[0, 1)`), width `w >= 1` is
/// `[2^(w-1), 2^w)`.
#[inline]
fn width_range(w: u32) -> (u32, u64) {
    if w == 0 {
        (0, 1)
    } else {
        (1u32 << (w - 1), 1u64 << w)
    }
}

/// Integer permille (`parts per thousand`) of `num / den`, saturating and
/// returning `0` when `den` is zero.
#[inline]
fn permille(num: u64, den: u64) -> u64 {
    (num * 1000).checked_div(den).unwrap_or(0)
}

/// One bit-width bucket of entity slot indices: how many dormant entities have
/// a slot index in `[index_lo, index_hi)` (design §13.2).
#[derive(Clone, Copy, Debug)]
pub struct IndexBucketEntry {
    /// Bit width shared by every index in this bucket (`0` for index `0`,
    /// else `floor(log2(index)) + 1`).
    pub bit_width: u32,
    /// Inclusive low slot index of the bucket.
    pub index_lo: u32,
    /// Exclusive high slot index of the bucket (a `u64` so width `32` can name
    /// `2^32`).
    pub index_hi: u64,
    /// Number of dormant entities whose slot index falls in this bucket.
    pub count: usize,
}

/// One bit-width bucket of entity generations: how many dormant entities carry
/// a generation in `[generation_lo, generation_hi)` — a recycle-depth
/// histogram (design §13.2).
#[derive(Clone, Copy, Debug)]
pub struct GenerationBucketEntry {
    /// Bit width shared by every generation in this bucket (generations are
    /// non-zero, so this is always `>= 1`).
    pub bit_width: u32,
    /// Inclusive low generation of the bucket.
    pub generation_lo: u32,
    /// Exclusive high generation of the bucket.
    pub generation_hi: u64,
    /// Number of dormant entities whose generation falls in this bucket.
    pub count: usize,
}

/// Read-only census of the dormant population's shape in the entity id-space:
/// index span / density, contiguity, and bit-width histograms of slot index
/// and generation (design §13.2 / §16.6).
#[derive(Clone, Debug)]
pub struct DormancyCensus {
    dormant_count: usize,
    pending_wake_count: usize,
    min_index: Option<u32>,
    max_index: Option<u32>,
    max_generation: u32,
    recycled_count: usize,
    contiguous_run_count: usize,
    longest_run_len: u64,
    index_buckets: Vec<IndexBucketEntry>,
    generation_buckets: Vec<GenerationBucketEntry>,
}

/// Folds a bit-width tally into the sorted, non-empty bucket entries, invoking
/// `make` to build each entry from its width, range and count.
fn emit_buckets<T>(tally: &[usize; BIT_WIDTH_BUCKETS], mut make: impl FnMut(u32, u32, u64, usize) -> T) -> Vec<T> {
    let mut out = Vec::new();
    for (w, &count) in tally.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let w = w as u32;
        let (lo, hi) = width_range(w);
        out.push(make(w, lo, hi, count));
    }
    out
}

impl DormancyCensus {
    /// Censuses a [`DormancySet`]: reads its dormant entities and pending wake
    /// count once and summarises the population's id-space shape. Read-only;
    /// `O(n log n)` for the contiguity sort.
    pub fn from_dormancy(set: &DormancySet) -> Self {
        let pending_wake_count = set.pending_wake_count();

        let mut indices: Vec<u32> = Vec::new();
        let mut index_tally = [0usize; BIT_WIDTH_BUCKETS];
        let mut generation_tally = [0usize; BIT_WIDTH_BUCKETS];
        let mut min_index: Option<u32> = None;
        let mut max_index: Option<u32> = None;
        let mut max_generation = 0u32;
        let mut recycled_count = 0usize;

        for entity in set.iter() {
            let idx = entity.index();
            let generation = entity.generation();
            indices.push(idx);
            index_tally[bit_width(idx) as usize] += 1;
            generation_tally[bit_width(generation) as usize] += 1;
            min_index = Some(min_index.map_or(idx, |m| m.min(idx)));
            max_index = Some(max_index.map_or(idx, |m| m.max(idx)));
            if generation > max_generation {
                max_generation = generation;
            }
            if generation > 1 {
                recycled_count += 1;
            }
        }

        let dormant_count = indices.len();

        // Contiguity: on the sorted slot indices, count maximal runs of
        // consecutive indices and track the longest. Indices are unique (one
        // live entity per slot), so no de-duplication is required.
        indices.sort_unstable();
        let mut contiguous_run_count = 0usize;
        let mut longest_run_len = 0u64;
        let mut run_len = 0u64;
        let mut prev: Option<u32> = None;
        for &idx in &indices {
            match prev {
                Some(p) if idx == p + 1 => run_len += 1,
                _ => {
                    contiguous_run_count += 1;
                    run_len = 1;
                }
            }
            if run_len > longest_run_len {
                longest_run_len = run_len;
            }
            prev = Some(idx);
        }

        let index_buckets = emit_buckets(&index_tally, |bit_width, lo, hi, count| IndexBucketEntry {
            bit_width,
            index_lo: lo,
            index_hi: hi,
            count,
        });
        let generation_buckets =
            emit_buckets(&generation_tally, |bit_width, lo, hi, count| GenerationBucketEntry {
                bit_width,
                generation_lo: lo,
                generation_hi: hi,
                count,
            });

        Self {
            dormant_count,
            pending_wake_count,
            min_index,
            max_index,
            max_generation,
            recycled_count,
            contiguous_run_count,
            longest_run_len,
            index_buckets,
            generation_buckets,
        }
    }

    /// Number of currently-dormant entities.
    #[inline]
    pub fn dormant_count(&self) -> usize {
        self.dormant_count
    }

    /// Whether the dormant set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dormant_count == 0
    }

    /// Wake events pending drain on the set at census time.
    #[inline]
    pub fn pending_wake_count(&self) -> usize {
        self.pending_wake_count
    }

    /// Fraction, in permille, of the recently-dormant population
    /// (`dormant + pending`) that is about to be re-admitted this frame:
    /// `pending × 1000 / (dormant + pending)`. `0` when both are zero.
    pub fn wake_churn_permille(&self) -> u64 {
        permille(
            self.pending_wake_count as u64,
            self.dormant_count as u64 + self.pending_wake_count as u64,
        )
    }

    /// Lowest slot index among the dormant entities, or `None` if empty.
    #[inline]
    pub fn min_index(&self) -> Option<u32> {
        self.min_index
    }

    /// Highest slot index among the dormant entities, or `None` if empty.
    #[inline]
    pub fn max_index(&self) -> Option<u32> {
        self.max_index
    }

    /// Width of the slot-index range the dormant set spans, inclusive
    /// (`max - min + 1`). `0` when empty.
    pub fn index_span(&self) -> u64 {
        match (self.min_index, self.max_index) {
            (Some(lo), Some(hi)) => (hi as u64 - lo as u64) + 1,
            _ => 0,
        }
    }

    /// How densely the spanned index range is filled, in permille:
    /// `dormant_count × 1000 / index_span`. `1000` means a perfectly
    /// contiguous block; a low value means a sparse scatter. `0` when empty.
    pub fn index_density_permille(&self) -> u64 {
        permille(self.dormant_count as u64, self.index_span())
    }

    /// Number of maximal runs of consecutive slot indices in the dormant set.
    /// `1` means a single contiguous block; higher means fragmentation.
    #[inline]
    pub fn contiguous_run_count(&self) -> usize {
        self.contiguous_run_count
    }

    /// Length of the longest run of consecutive slot indices. `0` when empty.
    #[inline]
    pub fn longest_run_len(&self) -> u64 {
        self.longest_run_len
    }

    /// Whether the dormant set occupies a single contiguous block of slot
    /// indices (at most one run) — the cheapest shape to serialise and migrate.
    pub fn is_contiguous(&self) -> bool {
        self.contiguous_run_count <= 1
    }

    /// Highest generation among the dormant entities (`0` when empty). A high
    /// value means slots have been recycled many times.
    #[inline]
    pub fn max_generation(&self) -> u32 {
        self.max_generation
    }

    /// Number of dormant entities occupying a recycled slot (`generation > 1`).
    #[inline]
    pub fn recycled_count(&self) -> usize {
        self.recycled_count
    }

    /// Fraction, in permille, of the dormant set sitting on recycled slots:
    /// `recycled_count × 1000 / dormant_count`. `0` when empty.
    pub fn recycled_permille(&self) -> u64 {
        permille(self.recycled_count as u64, self.dormant_count as u64)
    }

    /// The slot-index histogram: one entry per non-empty bit-width bucket,
    /// ascending by bit width (nearest the origin of the id-space first).
    #[inline]
    pub fn index_buckets(&self) -> &[IndexBucketEntry] {
        &self.index_buckets
    }

    /// Number of non-empty slot-index buckets.
    #[inline]
    pub fn index_bucket_count(&self) -> usize {
        self.index_buckets.len()
    }

    /// The generation (recycle-depth) histogram: one entry per non-empty
    /// bit-width bucket, ascending by bit width (freshest slots first).
    #[inline]
    pub fn generation_buckets(&self) -> &[GenerationBucketEntry] {
        &self.generation_buckets
    }

    /// Number of non-empty generation buckets.
    #[inline]
    pub fn generation_bucket_count(&self) -> usize {
        self.generation_buckets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entity;

    /// Builds an entity handle from an explicit `(index, generation)` pair.
    fn entity(index: u32, generation: u32) -> Entity {
        Entity::from_bits(((generation as u64) << 32) | index as u64).expect("non-zero generation")
    }

    #[test]
    fn empty_set_is_empty_and_zeroed() {
        let set = DormancySet::new();
        let census = DormancyCensus::from_dormancy(&set);
        assert!(census.is_empty());
        assert_eq!(census.dormant_count(), 0);
        assert_eq!(census.index_span(), 0);
        assert_eq!(census.index_density_permille(), 0);
        assert_eq!(census.contiguous_run_count(), 0);
        assert_eq!(census.longest_run_len(), 0);
        assert_eq!(census.max_generation(), 0);
        assert_eq!(census.recycled_count(), 0);
        assert_eq!(census.recycled_permille(), 0);
        assert_eq!(census.wake_churn_permille(), 0);
        assert!(census.index_buckets().is_empty());
        assert!(census.generation_buckets().is_empty());
    }

    #[test]
    fn contiguous_block_is_one_run_full_density() {
        let mut set = DormancySet::new();
        for i in 100..110 {
            set.sleep(entity(i, 1));
        }
        let census = DormancyCensus::from_dormancy(&set);
        assert_eq!(census.dormant_count(), 10);
        assert_eq!(census.min_index(), Some(100));
        assert_eq!(census.max_index(), Some(109));
        assert_eq!(census.index_span(), 10);
        assert_eq!(census.index_density_permille(), 1000);
        assert_eq!(census.contiguous_run_count(), 1);
        assert_eq!(census.longest_run_len(), 10);
        assert!(census.is_contiguous());
    }

    #[test]
    fn scattered_entities_many_runs_low_density() {
        let mut set = DormancySet::new();
        // Three isolated indices far apart: three runs, sparse span.
        for &i in &[0u32, 500, 1000] {
            set.sleep(entity(i, 1));
        }
        let census = DormancyCensus::from_dormancy(&set);
        assert_eq!(census.dormant_count(), 3);
        assert_eq!(census.index_span(), 1001);
        assert_eq!(census.contiguous_run_count(), 3);
        assert_eq!(census.longest_run_len(), 1);
        assert!(!census.is_contiguous());
        // 3 entities over a 1001-wide span is a very low density.
        assert!(census.index_density_permille() < 10);
    }

    #[test]
    fn two_blocks_are_two_runs_longest_tracked() {
        let mut set = DormancySet::new();
        for i in 10..13 {
            set.sleep(entity(i, 1)); // run of 3
        }
        for i in 50..55 {
            set.sleep(entity(i, 1)); // run of 5
        }
        let census = DormancyCensus::from_dormancy(&set);
        assert_eq!(census.contiguous_run_count(), 2);
        assert_eq!(census.longest_run_len(), 5);
        assert!(!census.is_contiguous());
    }

    #[test]
    fn index_buckets_partition_by_bit_width_and_are_sorted() {
        let mut set = DormancySet::new();
        // index 0 -> width 0; index 1 -> width 1; index 300 -> width 9.
        set.sleep(entity(0, 1));
        set.sleep(entity(1, 1));
        set.sleep(entity(300, 1));
        let census = DormancyCensus::from_dormancy(&set);
        let buckets = census.index_buckets();
        assert_eq!(buckets.len(), 3);
        // Ascending by bit width, every entity accounted for exactly once.
        let mut prev = 0u32;
        let mut total = 0usize;
        for (i, b) in buckets.iter().enumerate() {
            if i > 0 {
                assert!(b.bit_width > prev);
            }
            assert!((b.index_lo as u64) < b.index_hi);
            total += b.count;
            prev = b.bit_width;
        }
        assert_eq!(total, census.dormant_count());
        assert_eq!(buckets[0].bit_width, 0);
        assert_eq!(buckets[0].index_lo, 0);
        assert_eq!(buckets[0].index_hi, 1);
    }

    #[test]
    fn generation_histogram_counts_recycled_slots() {
        let mut set = DormancySet::new();
        set.sleep(entity(0, 1)); // fresh
        set.sleep(entity(1, 1)); // fresh
        set.sleep(entity(2, 4)); // recycled, width 3
        set.sleep(entity(3, 9)); // recycled, width 4
        let census = DormancyCensus::from_dormancy(&set);
        assert_eq!(census.recycled_count(), 2);
        assert_eq!(census.recycled_permille(), 500);
        assert_eq!(census.max_generation(), 9);
        // Fresh (gen 1, width 1) and the two recycled widths -> 3 buckets.
        let gens = census.generation_buckets();
        assert_eq!(gens.len(), 3);
        assert_eq!(gens[0].bit_width, 1);
        assert_eq!(gens[0].count, 2); // the two gen-1 entities
        let total: usize = gens.iter().map(|g| g.count).sum();
        assert_eq!(total, census.dormant_count());
    }

    #[test]
    fn pending_wake_drives_churn() {
        let mut set = DormancySet::new();
        for i in 0..3 {
            set.sleep(entity(i, 1));
        }
        // Wake one: it leaves the dormant set and becomes a pending wake event.
        assert!(set.wake(entity(0, 1)));
        let census = DormancyCensus::from_dormancy(&set);
        assert_eq!(census.dormant_count(), 2);
        assert_eq!(census.pending_wake_count(), 1);
        // 1 pending out of (2 dormant + 1 pending) = 333 permille.
        assert_eq!(census.wake_churn_permille(), 333);
    }

    #[test]
    fn span_and_density_are_consistent() {
        let mut set = DormancySet::new();
        // Indices 10, 11, 12, 20: span 11, 4 entities -> 363 permille.
        for &i in &[10u32, 11, 12, 20] {
            set.sleep(entity(i, 1));
        }
        let census = DormancyCensus::from_dormancy(&set);
        assert_eq!(census.index_span(), 11);
        assert_eq!(census.index_density_permille(), 363);
        assert_eq!(census.contiguous_run_count(), 2);
        assert_eq!(census.longest_run_len(), 3);
    }
}
