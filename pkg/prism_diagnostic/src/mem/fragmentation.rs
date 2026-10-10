//! Fragmentation visualization for a pool / virtual address range (§24.3).
//!
//! A pool can have plenty of free bytes yet still fail a large allocation
//! because the free space is chopped into small holes. This module turns an
//! occupancy list — the set of allocated [`Span`]s in a fixed-capacity range —
//! into actionable fragmentation metrics ([`FragmentationReport`]) and a coarse
//! [`occupancy_map`] suitable for a HUD/console heat strip.
//!
//! Everything here is pure `core`/`alloc` arithmetic: no allocator hot-path
//! involvement, no `unsafe`, deterministic, always compiled regardless of the
//! `alloc-track` feature.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// A half-open occupied byte span `[offset, offset + size)` within a pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Span {
    /// Start offset of the occupied region, in bytes from the pool base.
    pub offset: u64,
    /// Length of the occupied region, in bytes.
    pub size: u64,
}

impl Span {
    /// Construct an occupied span.
    #[must_use]
    pub fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    /// Exclusive end offset `offset + size`, saturating at [`u64::MAX`].
    #[must_use]
    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.size)
    }
}

/// Fragmentation metrics for one pool snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FragmentationReport {
    /// Total pool capacity in bytes.
    pub capacity: u64,
    /// Bytes occupied by allocations (overlaps/clamping already resolved).
    pub used_bytes: u64,
    /// Bytes that are free (`capacity - used_bytes`).
    pub free_bytes: u64,
    /// Number of maximal contiguous free runs between/around allocations.
    pub free_run_count: u64,
    /// Size in bytes of the single largest contiguous free run.
    pub largest_free_run: u64,
}

impl FragmentationReport {
    /// Occupancy ratio `used / capacity` in `[0, 1]`; `0.0` for a zero-capacity
    /// pool.
    #[must_use]
    pub fn occupancy_ratio(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            self.used_bytes as f64 / self.capacity as f64
        }
    }

    /// Fragmentation ratio in `[0, 1]`: `1 - largest_free_run / free_bytes`.
    ///
    /// `0.0` means all free space is in one run (not fragmented); values near
    /// `1.0` mean the free space is shattered into many small holes relative to
    /// the largest one. A pool with no free bytes reports `0.0` (nothing left
    /// to fragment).
    #[must_use]
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.free_bytes == 0 {
            0.0
        } else {
            1.0 - (self.largest_free_run as f64 / self.free_bytes as f64)
        }
    }

    /// Whether an allocation request of `size` bytes can be satisfied by the
    /// largest contiguous free run.
    #[must_use]
    pub fn can_fit(&self, size: u64) -> bool {
        self.largest_free_run >= size
    }
}

/// Analyze a pool's fragmentation from its capacity and occupied spans.
///
/// Input spans may be unsorted, may overlap, and may extend past `capacity`;
/// they are normalized (sorted + merged + clamped to `[0, capacity)`) so the
/// result is correct regardless of input hygiene. Zero-length spans are
/// ignored.
#[must_use]
pub fn analyze_fragmentation(capacity: u64, occupied: &[Span]) -> FragmentationReport {
    let merged = normalize(capacity, occupied);

    let mut used_bytes = 0u64;
    for span in &merged {
        used_bytes = used_bytes.saturating_add(span.size);
    }
    let free_bytes = capacity.saturating_sub(used_bytes);

    // Walk the gaps between consecutive occupied runs (plus the leading gap
    // before the first run and the trailing gap after the last) to find free
    // runs.
    let mut free_run_count = 0u64;
    let mut largest_free_run = 0u64;
    let mut cursor = 0u64;
    let note_gap = |from: u64, to: u64, count: &mut u64, largest: &mut u64| {
        if to > from {
            let run = to - from;
            *count += 1;
            if run > *largest {
                *largest = run;
            }
        }
    };
    for span in &merged {
        note_gap(
            cursor,
            span.offset,
            &mut free_run_count,
            &mut largest_free_run,
        );
        cursor = span.end();
    }
    note_gap(cursor, capacity, &mut free_run_count, &mut largest_free_run);

    FragmentationReport {
        capacity,
        used_bytes,
        free_bytes,
        free_run_count,
        largest_free_run,
    }
}

/// Render a coarse per-bucket occupancy map over `[0, capacity)`.
///
/// The range is split into `buckets` equal slices; each output byte is the
/// percentage (`0..=100`) of that slice covered by occupied spans, suitable for
/// a HUD heat strip. Returns an empty vector when `buckets == 0` or
/// `capacity == 0`.
#[must_use]
pub fn occupancy_map(capacity: u64, occupied: &[Span], buckets: usize) -> Vec<u8> {
    if buckets == 0 || capacity == 0 {
        return Vec::new();
    }
    let merged = normalize(capacity, occupied);
    let mut out = vec![0u8; buckets];
    let buckets_u = buckets as u64;
    for (index, slot) in out.iter_mut().enumerate() {
        // Bucket `i` covers `[lo, hi)`; use u128 to avoid overflow in the
        // offset multiply for large capacities.
        let lo = (index as u128 * capacity as u128 / buckets_u as u128) as u64;
        let hi = ((index as u128 + 1) * capacity as u128 / buckets_u as u128) as u64;
        let width = hi.saturating_sub(lo);
        if width == 0 {
            continue;
        }
        let mut covered = 0u64;
        for span in &merged {
            let s = span.offset.max(lo);
            let e = span.end().min(hi);
            if e > s {
                covered = covered.saturating_add(e - s);
            }
        }
        // Percentage, rounded to nearest, clamped to 100.
        let pct = (covered as u128 * 100 + width as u128 / 2) / width as u128;
        *slot = pct.min(100) as u8;
    }
    out
}

/// Sort, clamp to `[0, capacity)`, drop empties, and merge overlapping/adjacent
/// occupied spans into a canonical, disjoint, ascending list.
fn normalize(capacity: u64, occupied: &[Span]) -> Vec<Span> {
    let mut spans: Vec<Span> = occupied
        .iter()
        .filter_map(|s| {
            let start = s.offset.min(capacity);
            let end = s.end().min(capacity);
            if end > start {
                Some(Span::new(start, end - start))
            } else {
                None
            }
        })
        .collect();
    spans.sort_by_key(|s| s.offset);

    let mut merged: Vec<Span> = Vec::with_capacity(spans.len());
    for span in spans {
        if let Some(last) = merged.last_mut()
            && span.offset <= last.end()
        {
            let new_end = last.end().max(span.end());
            last.size = new_end - last.offset;
            continue;
        }
        merged.push(span);
    }
    merged
}
