//! Append / consume atomic-counter buffer semantics: the `CPU`-verifiable
//! gold standard for the particle subsystem's `GPU` append/consume storage
//! primitive (design §5.2 pooling counters, §9 `Spawn` / `Event Scatter` writes,
//! §11 atomic counters).
//!
//! An *append buffer* is the write side of the produce/consume pattern every
//! production `GPU` VFX stack leans on: `Direct3D`'s `AppendStructuredBuffer`
//! (`.Append()` atomically increments a hidden counter and returns the slot the
//! caller writes) and `Vulkan`'s `atomicAdd` on a dedicated *atomic-counter*
//! binding are the same idea: many concurrent invocations race to reserve a
//! monotonically increasing slot index, the reservations never collide, and a
//! run that overflows the fixed capacity is dropped rather than corrupting a
//! neighbour. Its mirror is the *consume buffer* (`ConsumeStructuredBuffer` /
//! an atomic decrement) which hands slots back from the tail. Unreal `Niagara`
//! uses exactly this to append freshly spawned particles into a compact write
//! list and to consume work items from a stack.
//!
//! On the device a single `.Append()` is one atomic increment, but a whole
//! `workgroup` appending at once is not left to raw hardware ordering: the
//! deterministic contract is the classic *two-pass base allocation*. Pass one
//! computes each `workgroup`'s *local count* (how many slots it wants, clamped
//! against the remaining capacity); pass two exclusive-scans those local counts
//! into a per-`workgroup` *global base offset*, so `workgroup` `g` owns the
//! contiguous slot range `[base_g, base_g + local_g)` and no two ranges overlap.
//! Within a `workgroup` the elements keep their given order, which is what makes
//! a concurrent batch *serialize deterministically*: the result depends only on
//! the group index and the in-group position, never on which lane happened to
//! win an atomic race.
//!
//! This module owns only that deterministic `CPU` reference so the eventual
//! `GPU` build can be validated bit for bit. Everything is pure integer
//! arithmetic: it never panics on an empty buffer or an empty batch, it clamps
//! rather than overflowing its slot store, and it never divides by zero.
//!
//! **Deliberately out of scope (no overlap with siblings):** this file owns only
//! the `append / consume atomic counter + two-pass workgroup base allocation +`
//! `overflow` semantics. It deliberately does **not** perform predicate stream
//! compaction `scatter` — the just-added [`super::gpu_compact`] owns the
//! two-pass prefix-sum keep/drop packing, and nothing here touches its scan; it
//! does not run a *segmented* head-flag scan (that is
//! [`super::gpu_scan_segmented`]); it does not recycle object-pool slots (that
//! is [`super::pool`]'s free list); and it does not build dispatch arguments
//! (that is [`super::gpu_dispatch`] / [`super::indirect_dispatch`]). Its only
//! dependency is the shared [`super::gpu_layout`] `std430` stride rule.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};

/// Number of `u32` words in the packed `std430` append-counter record.
///
/// Layout is one `vec4`-aligned word block `[count, overflow, capacity, reserved]`,
/// mirroring the hidden atomic counter a `GPU` `AppendStructuredBuffer` keeps
/// beside its data buffer (padded up to a natural four-word slot).
pub const COUNTER_WORDS: usize = 4;

/// Widens a `usize` slot count into the `u32` a `GPU` counter stores,
/// saturating at [`u32::MAX`] rather than panicking on an astronomically large
/// value.
fn count_to_u32(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}

/// Outcome of a [`batch_append`](AppendCounter::batch_append): the two-pass
/// base allocation of a concurrently-grouped batch, serialized deterministically.
///
/// Mirrors the intermediates a `GPU` batch append publishes: each simulated
/// `workgroup`'s granted slot count and its global base offset (the exclusive
/// prefix sum of the granted counts), plus the batch's overall appended and
/// `overflow` totals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchAppendResult {
    /// Per-`workgroup` global base offset: the first slot index each group
    /// owns, one entry per input group. Equals the group's starting position
    /// plus the exclusive prefix sum of the *granted* counts, so consecutive
    /// groups never overlap.
    pub group_bases: Vec<usize>,
    /// Per-`workgroup` granted count: how many of the group's elements were
    /// actually written after the capacity clamp, one entry per input group.
    pub group_counts: Vec<usize>,
    /// Total slots appended by this batch (the sum of
    /// [`group_counts`](Self::group_counts)).
    pub appended: usize,
    /// Number of elements this batch dropped because capacity was exhausted.
    pub overflow: usize,
}

/// Configuration for an append/consume buffer: the fixed slot `capacity` and
/// the byte `stride` of one stored element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AppendConfig {
    /// Maximum number of live slots the append buffer can hold. May be zero (a
    /// degenerate buffer that overflows every append); the `std430` byte size
    /// still clamps up to one element for a valid `GPU` binding.
    pub capacity: usize,
    /// Byte stride of one stored element, clamped to at least one so a
    /// degenerate zero can never collapse the data buffer size.
    pub stride: usize,
}

impl AppendConfig {
    /// Builds a config with the given `capacity` and element `stride`,
    /// clamping the stride to at least one byte.
    #[must_use]
    pub fn new(capacity: usize, stride: usize) -> Self {
        Self {
            capacity,
            stride: stride.max(1),
        }
    }

    /// Byte size of the `std430` data buffer holding all `capacity` slots at
    /// this element `stride`, via the shared [`storage_bytes`] clamp-to-one
    /// rule.
    #[must_use]
    pub fn buffer_bytes(self) -> usize {
        storage_bytes(self.stride, self.capacity)
    }

    /// Byte size of the `std430` hidden atomic-counter buffer: the packed
    /// [`COUNTER_WORDS`] `u32` record beside the data buffer.
    #[must_use]
    pub fn counter_bytes(self) -> usize {
        storage_bytes(U32_STRIDE, COUNTER_WORDS)
    }

    /// Constructs an empty [`AppendCounter`] with this config's `capacity`.
    #[must_use]
    pub fn make_counter(self) -> AppendCounter {
        AppendCounter::new(self.capacity)
    }
}

/// A `CPU` model of a `GPU` append/consume buffer: the monotonic atomic slot
/// index (the length of the compact backing store), a fixed `capacity`, and the
/// running count of appends dropped after the buffer filled.
///
/// The compact backing store holds the appended payloads (modelled as `u32`
/// slot values, e.g. particle indices) in allocation order, exactly the dense
/// prefix a `GPU` `AppendStructuredBuffer` produces. The atomic "current
/// counter" is simply the store's length; [`consume`](Self::consume) pops from
/// the tail, matching an atomic decrement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppendCounter {
    /// Compact backing store of appended payloads, in allocation order. Its
    /// length is the atomic append index and never exceeds `capacity`.
    slots: Vec<u32>,
    /// Fixed upper bound on the number of live slots.
    capacity: usize,
    /// Count of append attempts rejected because the buffer was full.
    overflow: usize,
}

impl AppendCounter {
    /// Creates an empty append buffer with the given slot `capacity`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::new(),
            capacity,
            overflow: 0,
        }
    }

    /// The fixed slot capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The current atomic append index: how many slots are live.
    #[must_use]
    pub fn count(&self) -> usize {
        self.slots.len()
    }

    /// Number of append attempts dropped so far because the buffer was full.
    #[must_use]
    pub fn overflow_count(&self) -> usize {
        self.overflow
    }

    /// Slots still free before the buffer overflows.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.slots.len())
    }

    /// Whether no slots are live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Whether every slot is occupied (the next append will overflow).
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.slots.len() >= self.capacity
    }

    /// The compact backing store of appended payloads, in allocation order.
    #[must_use]
    pub fn slots(&self) -> &[u32] {
        &self.slots
    }

    /// Atomically reserves the next slot and writes `value` into it, returning
    /// the allocated slot index.
    ///
    /// This is the core of `Direct3D`'s `AppendStructuredBuffer.Append()` /
    /// `Vulkan`'s `atomicAdd(counter, 1)`: while capacity remains it returns
    /// `Some(index)` for the freshly reserved slot, and once the buffer is full
    /// it drops the value, bumps the [`overflow_count`](Self::overflow_count),
    /// and returns `None`. Because the reserved index is always the current
    /// length, sequential appends yield strictly increasing indices with no
    /// collisions.
    pub fn atomic_append(&mut self, value: u32) -> Option<usize> {
        if self.slots.len() < self.capacity {
            let index = self.slots.len();
            self.slots.push(value);
            Some(index)
        } else {
            self.overflow += 1;
            None
        }
    }

    /// Appends `value`, returning whether it fit.
    ///
    /// Thin wrapper over [`atomic_append`](Self::atomic_append) that reports
    /// success as a bool rather than the allocated index, for callers that only
    /// need the fit/overflow outcome.
    pub fn append(&mut self, value: u32) -> bool {
        self.atomic_append(value).is_some()
    }

    /// Consumes one slot from the tail, returning its payload.
    ///
    /// Mirrors `ConsumeStructuredBuffer.Consume()` / an atomic decrement: while
    /// the buffer is non-empty it pops the most recently appended slot (last in,
    /// first out), decrementing the atomic count, and returns `Some(value)`; on
    /// an empty buffer it returns `None` and leaves the count at zero.
    /// Consuming never touches the `overflow` tally.
    pub fn consume(&mut self) -> Option<u32> {
        self.slots.pop()
    }

    /// Clears every live slot and the `overflow` tally, keeping the capacity.
    pub fn reset(&mut self) {
        self.slots.clear();
        self.overflow = 0;
    }

    /// Appends a concurrently-grouped batch under the deterministic two-pass
    /// base-allocation contract, returning the per-`workgroup` bases and counts.
    ///
    /// Each entry of `groups` is one simulated `workgroup`'s ordered list of
    /// payloads. The contract has two passes over the groups, always in group
    /// index order:
    ///
    /// 1. **Local count** — each `workgroup` wants `group.len()` slots,
    ///    clamped against the buffer's remaining capacity as earlier groups
    ///    consume it, yielding the *granted* count; the shortfall is added to
    ///    `overflow`.
    /// 2. **Base offset** — the exclusive prefix sum of the granted counts,
    ///    offset by the buffer's starting length, gives each `workgroup`'s
    ///    global base. `workgroup` `g` then writes its granted elements into
    ///    the contiguous range `[base_g, base_g + granted_g)`, so no two groups
    ///    overlap and the whole batch lands as one compact run.
    ///
    /// Within a group the elements keep their given order, which is what makes
    /// the concurrent batch serialize deterministically. An empty `groups`
    /// slice, or groups that are individually empty, append nothing and never
    /// panic.
    pub fn batch_append(&mut self, groups: &[&[u32]]) -> BatchAppendResult {
        let start = self.slots.len();
        let mut group_bases = Vec::with_capacity(groups.len());
        let mut group_counts = Vec::with_capacity(groups.len());
        let mut appended = 0usize;
        let mut overflow = 0usize;
        // `running` is the exclusive prefix sum of granted counts within this
        // batch; it never exceeds the remaining capacity by construction.
        let mut running = 0usize;
        let remaining = self.capacity.saturating_sub(start);
        for group in groups {
            let base = start + running;
            let free = remaining.saturating_sub(running);
            let granted = group.len().min(free);
            group_bases.push(base);
            group_counts.push(granted);
            self.slots.extend_from_slice(&group[..granted]);
            overflow += group.len() - granted;
            appended += granted;
            running += granted;
        }
        self.overflow += overflow;
        BatchAppendResult {
            group_bases,
            group_counts,
            appended,
            overflow,
        }
    }

    /// Packs the atomic counter into its `std430` `vec4`-aligned word layout.
    ///
    /// Layout: `[count, overflow, capacity, reserved]` as raw `u32` words, one
    /// `vec4` slot matching [`COUNTER_WORDS`]. Each `usize` saturates into
    /// `u32` via [`count_to_u32`]; the trailing word is reserved padding.
    #[must_use]
    pub fn to_std430(&self) -> [u32; COUNTER_WORDS] {
        [
            count_to_u32(self.slots.len()),
            count_to_u32(self.overflow),
            count_to_u32(self.capacity),
            0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // A local stride constant so the config test does not import a second
    // gpu_layout symbol beyond the module's declared reuse surface.
    const VEC4_STRIDE_LOCAL: usize = 16;

    #[test]
    fn sequential_atomic_append_indices_increment() {
        let mut counter = AppendCounter::new(8);
        assert_eq!(counter.atomic_append(10), Some(0));
        assert_eq!(counter.atomic_append(20), Some(1));
        assert_eq!(counter.atomic_append(30), Some(2));
        assert_eq!(counter.count(), 3);
        assert_eq!(counter.slots(), &[10, 20, 30]);
        assert_eq!(counter.overflow_count(), 0);
    }

    #[test]
    fn append_reports_fit_then_overflow() {
        let mut counter = AppendCounter::new(2);
        assert!(counter.append(1));
        assert!(counter.append(2));
        assert!(!counter.append(3));
        assert!(!counter.append(4));
        assert_eq!(counter.count(), 2);
        assert_eq!(counter.overflow_count(), 2);
    }

    #[test]
    fn full_buffer_overflow_never_exceeds_capacity() {
        let mut counter = AppendCounter::new(3);
        for value in 0u32..10 {
            let _ = counter.atomic_append(value);
        }
        // The store is clamped to capacity and holds the first-arrived values.
        assert_eq!(counter.count(), 3);
        assert!(counter.count() <= counter.capacity());
        assert_eq!(counter.slots(), &[0, 1, 2]);
        assert_eq!(counter.overflow_count(), 7);
        assert!(counter.is_full());
    }

    #[test]
    fn atomic_append_returns_none_when_full() {
        let mut counter = AppendCounter::new(1);
        assert_eq!(counter.atomic_append(99), Some(0));
        assert_eq!(counter.atomic_append(100), None);
        assert_eq!(counter.overflow_count(), 1);
    }

    #[test]
    fn consume_is_last_in_first_out() {
        let mut counter = AppendCounter::new(4);
        counter.append(7);
        counter.append(8);
        counter.append(9);
        assert_eq!(counter.consume(), Some(9));
        assert_eq!(counter.consume(), Some(8));
        assert_eq!(counter.count(), 1);
        assert_eq!(counter.slots(), &[7]);
    }

    #[test]
    fn consume_on_empty_returns_none() {
        let mut counter = AppendCounter::new(4);
        assert_eq!(counter.consume(), None);
        assert_eq!(counter.count(), 0);
        assert!(counter.is_empty());
    }

    #[test]
    fn append_then_consume_roundtrip_frees_slots() {
        let mut counter = AppendCounter::new(2);
        counter.append(5);
        counter.append(6);
        assert!(counter.is_full());
        assert_eq!(counter.consume(), Some(6));
        assert_eq!(counter.remaining(), 1);
        // A freed slot can be reused by a later append.
        assert_eq!(counter.atomic_append(60), Some(1));
        assert_eq!(counter.slots(), &[5, 60]);
    }

    #[test]
    fn batch_append_bases_are_prefix_sum_and_stable_within_group() {
        let mut counter = AppendCounter::new(16);
        let g0: &[u32] = &[10, 11];
        let g1: &[u32] = &[20, 21, 22];
        let g2: &[u32] = &[30];
        let result = counter.batch_append(&[g0, g1, g2]);
        // Bases are the exclusive prefix sum of the granted counts: 0, 2, 5.
        assert_eq!(result.group_bases, vec![0, 2, 5]);
        assert_eq!(result.group_counts, vec![2, 3, 1]);
        assert_eq!(result.appended, 6);
        assert_eq!(result.overflow, 0);
        // In-group order is preserved and groups land contiguously.
        assert_eq!(counter.slots(), &[10, 11, 20, 21, 22, 30]);
    }

    #[test]
    fn batch_append_group_ranges_do_not_overlap() {
        let mut counter = AppendCounter::new(32);
        let g0: &[u32] = &[1, 2, 3];
        let g1: &[u32] = &[4, 5];
        let g2: &[u32] = &[6, 7, 8, 9];
        let result = counter.batch_append(&[g0, g1, g2]);
        // Each group's [base, base + count) range is disjoint and ordered.
        for window in 0..result.group_bases.len() - 1 {
            let end = result.group_bases[window] + result.group_counts[window];
            assert!(end <= result.group_bases[window + 1]);
        }
        // The bases also match the exclusive prefix sum of the granted counts.
        for (idx, &base) in result.group_bases.iter().enumerate() {
            let expected_base: usize = result.group_counts[..idx].iter().sum();
            assert_eq!(base, expected_base);
        }
    }

    #[test]
    fn batch_append_partial_overflow_across_groups() {
        // Capacity 4: first group fully fits, second partially, third drops.
        let mut counter = AppendCounter::new(4);
        let g0: &[u32] = &[1, 2, 3];
        let g1: &[u32] = &[4, 5, 6];
        let g2: &[u32] = &[7, 8];
        let result = counter.batch_append(&[g0, g1, g2]);
        assert_eq!(result.group_counts, vec![3, 1, 0]);
        assert_eq!(result.group_bases, vec![0, 3, 4]);
        assert_eq!(result.appended, 4);
        assert_eq!(result.overflow, 4);
        assert_eq!(counter.slots(), &[1, 2, 3, 4]);
        assert_eq!(counter.overflow_count(), 4);
        assert!(counter.is_full());
    }

    #[test]
    fn batch_append_onto_non_empty_buffer_offsets_bases() {
        let mut counter = AppendCounter::new(16);
        counter.append(100);
        counter.append(101);
        let g0: &[u32] = &[1, 2];
        let g1: &[u32] = &[3];
        let result = counter.batch_append(&[g0, g1]);
        // Bases start at the buffer's current length (2), not at zero.
        assert_eq!(result.group_bases, vec![2, 4]);
        assert_eq!(counter.slots(), &[100, 101, 1, 2, 3]);
    }

    #[test]
    fn batch_append_empty_inputs_do_nothing() {
        let mut counter = AppendCounter::new(8);
        let empty = counter.batch_append(&[]);
        assert!(empty.group_bases.is_empty());
        assert_eq!(empty.appended, 0);
        assert_eq!(empty.overflow, 0);

        // Individually-empty groups still yield a base but append nothing.
        let g0: &[u32] = &[];
        let g1: &[u32] = &[42];
        let result = counter.batch_append(&[g0, g1]);
        assert_eq!(result.group_bases, vec![0, 0]);
        assert_eq!(result.group_counts, vec![0, 1]);
        assert_eq!(counter.slots(), &[42]);
    }

    #[test]
    fn batch_append_into_full_buffer_all_overflow() {
        let mut counter = AppendCounter::new(1);
        counter.append(1);
        let g0: &[u32] = &[2, 3];
        let g1: &[u32] = &[4];
        let result = counter.batch_append(&[g0, g1]);
        assert_eq!(result.appended, 0);
        assert_eq!(result.overflow, 3);
        assert_eq!(result.group_counts, vec![0, 0]);
        // Every base points at the (full) tail; the ranges are empty.
        assert_eq!(result.group_bases, vec![1, 1]);
        assert_eq!(counter.slots(), &[1]);
    }

    #[test]
    fn consume_after_batch_returns_last_written_value() {
        let mut counter = AppendCounter::new(8);
        let g0: &[u32] = &[10, 11];
        let g1: &[u32] = &[12, 13];
        counter.batch_append(&[g0, g1]);
        assert_eq!(counter.consume(), Some(13));
        assert_eq!(counter.consume(), Some(12));
        assert_eq!(counter.consume(), Some(11));
        assert_eq!(counter.count(), 1);
    }

    #[test]
    fn reset_clears_slots_and_overflow() {
        let mut counter = AppendCounter::new(2);
        counter.append(1);
        counter.append(2);
        counter.append(3); // overflow
        assert_eq!(counter.overflow_count(), 1);
        counter.reset();
        assert!(counter.is_empty());
        assert_eq!(counter.overflow_count(), 0);
        assert_eq!(counter.capacity(), 2);
        assert_eq!(counter.remaining(), 2);
    }

    #[test]
    fn remaining_and_full_track_capacity() {
        let mut counter = AppendCounter::new(3);
        assert_eq!(counter.remaining(), 3);
        assert!(!counter.is_full());
        counter.append(1);
        assert_eq!(counter.remaining(), 2);
        counter.append(2);
        counter.append(3);
        assert_eq!(counter.remaining(), 0);
        assert!(counter.is_full());
    }

    #[test]
    fn std430_counter_layout_packs_count_overflow_capacity() {
        let mut counter = AppendCounter::new(2);
        counter.append(7);
        counter.append(8);
        counter.append(9); // overflow
        let packed = counter.to_std430();
        assert_eq!(packed, [2, 1, 2, 0]);
    }

    #[test]
    fn config_bytes_and_empty_clamp() {
        let config = AppendConfig::new(10, VEC4_STRIDE_LOCAL);
        assert_eq!(config.buffer_bytes(), 10 * VEC4_STRIDE_LOCAL);
        // The counter is one packed vec4 word block.
        assert_eq!(config.counter_bytes(), COUNTER_WORDS * U32_STRIDE);

        // An empty-capacity buffer still reserves one element for a valid GPU
        // binding via the shared clamp-to-one rule.
        let empty = AppendConfig::new(0, U32_STRIDE);
        assert_eq!(empty.buffer_bytes(), U32_STRIDE);
        assert_eq!(empty.counter_bytes(), COUNTER_WORDS * U32_STRIDE);
    }

    #[test]
    fn config_clamps_zero_stride_and_builds_counter() {
        let config = AppendConfig::new(4, 0);
        assert_eq!(config.stride, 1);
        assert_eq!(config.buffer_bytes(), 4);
        let counter = config.make_counter();
        assert_eq!(counter.capacity(), 4);
        assert!(counter.is_empty());
    }

    #[test]
    fn zero_capacity_counter_overflows_every_append() {
        let mut counter = AppendCounter::new(0);
        assert_eq!(counter.atomic_append(1), None);
        assert!(!counter.append(2));
        assert_eq!(counter.overflow_count(), 2);
        assert!(counter.is_full());
        assert_eq!(counter.remaining(), 0);
    }
}
