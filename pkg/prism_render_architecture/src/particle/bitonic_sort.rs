//! Bitonic sorting-network dispatch contract for the particle sort pass.
//!
//! The §12 strategy matrix (see [`super::sort_cull`]) routes small per-emitter
//! sorts to a bitonic network and large ones to a radix count-sort. Those two
//! kernels are complementary: [`super::sort_cull`] owns the *radix* key
//! quantization and strategy choice, while this module owns the *bitonic
//! compare-exchange network* — the fixed schedule of stages and passes, the
//! per-invocation partner index, and the per-element sort direction that a
//! `GPU` compare-exchange kernel replays.
//!
//! A bitonic network sorts a power-of-two element count in
//! `stage * (stage + 1) / 2` compare-exchange passes, where `stage` is the
//! base-two logarithm of the padded count. Pass `p` of stage `s` compares each
//! element with the partner reached by flipping one bit (an `XOR` by the
//! *compare distance* `1 << (s - p)`), and the *box* the element sits in
//! (`1 << s`) decides whether that pair sorts ascending or descending. Padding
//! the input up to a power of two with a sentinel maximum key keeps the padded
//! tail sorted to the end so an ascending network leaves the real keys in
//! order.
//!
//! The compare/exchange kernel itself runs on the `GPU` and is pending the
//! backend; this zero-dependency module owns only the `CPU`-verifiable
//! *contract*: the network size arithmetic, the `std430` dispatch byte layout
//! (reusing [`U32_STRIDE`]), and a full `CPU` reference network that sorts a
//! `Vec<u32>` so the schedule can be checked against an ordinary sort. Only
//! integer arithmetic and shifts are used — no transcendental functions — and
//! nothing panics or divides by zero, so an empty or single-element input is a
//! valid no-op.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};

/// `std430` byte stride of a `WebGPU` `DispatchIndirectArgs` record consumed by
/// an indirect bitonic dispatch: the three `u32` workgroup counts `x`, `y`, `z`
/// (`3 * 4 = 12` bytes).
pub const DISPATCH_ARGS_STRIDE: usize = 3 * U32_STRIDE;

/// The size and derived schedule of one bitonic sorting network.
///
/// Constructed from an unpadded `element_count`; every derived quantity pads up
/// to the next power of two, because a bitonic network only sorts power-of-two
/// lengths. The type is a pure description — it holds no keys and allocates
/// nothing until [`BitonicSort::sort_keys`] runs the `CPU` reference network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BitonicSort {
    /// Unpadded number of keys the caller wants sorted.
    pub element_count: usize,
}

impl BitonicSort {
    /// Creates a network descriptor for `element_count` unpadded keys.
    #[must_use]
    pub fn new(element_count: usize) -> Self {
        Self { element_count }
    }

    /// The `element_count` rounded up to the next power of two.
    ///
    /// Computed with a doubling loop (no `log2`/`powf`): `0` and `1` are already
    /// valid network sizes and pass through unchanged, while any larger count
    /// grows to the smallest power of two that covers it. A count so large that
    /// the next doubling would overflow `usize` returns the count unchanged
    /// rather than looping forever.
    #[must_use]
    pub fn padded_count(self) -> usize {
        let n = self.element_count;
        if n <= 1 {
            return n;
        }
        let mut p: usize = 1;
        while p < n {
            if p > usize::MAX >> 1 {
                return n;
            }
            p <<= 1;
        }
        p
    }

    /// Number of bitonic *stages* — the base-two logarithm of the padded count.
    ///
    /// Counted by right-shifting the padded count to zero (no `log2`): a padded
    /// count of `1` (or the empty `0`) needs no stages, `2` needs one, `4` two,
    /// and so on.
    #[must_use]
    pub fn stage_count(self) -> u32 {
        let padded = self.padded_count();
        if padded <= 1 {
            return 0;
        }
        let mut count: u32 = 0;
        let mut p = padded;
        while p > 1 {
            p >>= 1;
            count += 1;
        }
        count
    }

    /// Total number of compare-exchange passes in the whole network.
    ///
    /// Stage `s` (one-based) runs `s` passes, so the network runs
    /// `1 + 2 + ... + stage = stage * (stage + 1) / 2` passes in all.
    #[must_use]
    pub fn total_pass_count(self) -> u32 {
        let s = self.stage_count();
        s * (s + 1) / 2
    }

    /// Number of 1-D workgroups needed to cover the padded element count at a
    /// given `@workgroup_size` (one invocation per padded element).
    ///
    /// A zero `workgroup_size` yields zero rather than dividing by zero.
    #[must_use]
    pub fn workgroup_count(self, workgroup_size: u32) -> u32 {
        if workgroup_size == 0 {
            return 0;
        }
        let padded = u32::try_from(self.padded_count()).unwrap_or(u32::MAX);
        padded.div_ceil(workgroup_size)
    }

    /// `std430` byte size of the padded `u32` key buffer the network sorts.
    ///
    /// Reuses the shared clamp-to-one-element rule from [`storage_bytes`], so an
    /// empty network still reserves one non-empty `WebGPU` storage element.
    #[must_use]
    pub fn key_buffer_bytes(self) -> usize {
        storage_bytes(U32_STRIDE, self.padded_count())
    }

    /// The ordered list of every compare-exchange step in the network.
    ///
    /// Stages run one-based from `1..=stage_count`, and within stage `s` the
    /// passes run `1..=s` (compare distance halving from `1 << (s - 1)` down to
    /// `1`). The length always equals [`BitonicSort::total_pass_count`].
    #[must_use]
    pub fn steps(self) -> Vec<CompareStep> {
        let stages = self.stage_count();
        let mut out = Vec::new();
        for stage in 1..=stages {
            for pass in 1..=stage {
                out.push(CompareStep::new(stage, pass));
            }
        }
        out
    }

    /// Runs the full bitonic network on `keys` on the `CPU` and returns them in
    /// ascending order.
    ///
    /// The input is padded up to a power of two with the sentinel `u32::MAX`;
    /// every step performs the compare-exchange described by its
    /// [`CompareStep`], and the sentinel tail is dropped before returning. The
    /// result is identical to an ordinary ascending sort of `keys`, which is
    /// exactly what the `GPU` kernel this contract describes must reproduce.
    #[must_use]
    pub fn sort_keys(keys: &[u32]) -> Vec<u32> {
        let sorter = Self::new(keys.len());
        let padded = sorter.padded_count();
        let mut buf: Vec<u32> = Vec::with_capacity(padded.max(keys.len()));
        buf.extend_from_slice(keys);
        buf.resize(padded, u32::MAX);
        for stage in 1..=sorter.stage_count() {
            for pass in 1..=stage {
                let step = CompareStep::new(stage, pass);
                let distance = step.compare_distance();
                for i in 0..buf.len() {
                    let partner = i ^ distance;
                    if partner <= i {
                        continue;
                    }
                    let ascending = step.sort_ascending(i);
                    let out_of_order = if ascending {
                        buf[i] > buf[partner]
                    } else {
                        buf[i] < buf[partner]
                    };
                    if out_of_order {
                        buf.swap(i, partner);
                    }
                }
            }
        }
        buf.truncate(keys.len());
        buf
    }
}

/// One compare-exchange pass of a bitonic network, identified by its one-based
/// `stage` and `pass` within that stage.
///
/// A `GPU` kernel invocation for element `index` uses [`CompareStep`] to find
/// its [`compare_partner`](CompareStep::compare_partner) and whether that pair
/// sorts [`ascending`](CompareStep::sort_ascending); the pass is the same for
/// every invocation, so it is a tiny push-constant, not per-element data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompareStep {
    /// One-based stage index (`1..=stage_count`); the box size is `1 << stage`.
    pub stage: u32,
    /// One-based pass index within the stage (`1..=stage`).
    pub pass: u32,
}

impl CompareStep {
    /// Creates a step for the given one-based `stage` and `pass`.
    #[must_use]
    pub fn new(stage: u32, pass: u32) -> Self {
        Self { stage, pass }
    }

    /// The `XOR` distance between an element and its compare partner in this
    /// pass: `1 << (stage - pass)`.
    ///
    /// A degenerate step (`stage == 0`, `pass == 0`, or `pass > stage`) has no
    /// well-defined distance and returns `0`, which
    /// [`compare_partner`](CompareStep::compare_partner) turns into a self-pair
    /// the reference network skips.
    #[must_use]
    pub fn compare_distance(self) -> usize {
        if self.stage == 0 || self.pass == 0 || self.pass > self.stage {
            return 0;
        }
        1usize << (self.stage - self.pass)
    }

    /// The bitonic *box* size for this pass: `1 << stage`.
    ///
    /// Every contiguous run of `box_size` elements is sorted into one monotone
    /// sequence; whether that run is ascending or descending is what
    /// [`sort_ascending`](CompareStep::sort_ascending) reports.
    #[must_use]
    pub fn box_size(self) -> usize {
        1usize << self.stage
    }

    /// The partner element `index` compares against in this pass: `index`
    /// `XOR` [`compare_distance`](CompareStep::compare_distance).
    #[must_use]
    pub fn compare_partner(self, index: usize) -> usize {
        index ^ self.compare_distance()
    }

    /// Whether the pair containing `index` sorts ascending in this pass.
    ///
    /// Elements whose `box_size` bit is clear head an ascending run; those with
    /// it set head a descending run, which is what makes the merged sequence
    /// bitonic before the next stage folds it back to fully ascending.
    #[must_use]
    pub fn sort_ascending(self, index: usize) -> bool {
        (index & self.box_size()) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic linear-congruential generator (`LCG`) for randomized
    /// inputs, so tests never pull in an external rng crate. Constants are the
    /// Numerical Recipes 32-bit set.
    struct Lcg {
        state: u32,
    }

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self
                .state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            self.state
        }
    }

    #[test]
    fn padded_count_rounds_up_to_power_of_two() {
        assert_eq!(BitonicSort::new(0).padded_count(), 0);
        assert_eq!(BitonicSort::new(1).padded_count(), 1);
        assert_eq!(BitonicSort::new(2).padded_count(), 2);
        assert_eq!(BitonicSort::new(3).padded_count(), 4);
        assert_eq!(BitonicSort::new(5).padded_count(), 8);
        assert_eq!(BitonicSort::new(1000).padded_count(), 1024);
    }

    #[test]
    fn padded_count_saturates_instead_of_looping() {
        // A count above the largest representable power of two returns unchanged
        // rather than overflowing the doubling loop.
        let huge = (usize::MAX >> 1) + 2;
        assert_eq!(BitonicSort::new(huge).padded_count(), huge);
    }

    #[test]
    fn stage_count_is_log2_of_padded() {
        assert_eq!(BitonicSort::new(0).stage_count(), 0);
        assert_eq!(BitonicSort::new(1).stage_count(), 0);
        assert_eq!(BitonicSort::new(2).stage_count(), 1);
        assert_eq!(BitonicSort::new(4).stage_count(), 2);
        assert_eq!(BitonicSort::new(5).stage_count(), 3);
        assert_eq!(BitonicSort::new(1024).stage_count(), 10);
    }

    #[test]
    fn total_pass_count_is_triangular() {
        // stage * (stage + 1) / 2 for a few sizes.
        assert_eq!(BitonicSort::new(1).total_pass_count(), 0);
        assert_eq!(BitonicSort::new(2).total_pass_count(), 1);
        assert_eq!(BitonicSort::new(4).total_pass_count(), 3);
        assert_eq!(BitonicSort::new(8).total_pass_count(), 6);
        assert_eq!(BitonicSort::new(1024).total_pass_count(), 55);
    }

    #[test]
    fn workgroup_count_uses_div_ceil_and_guards_zero() {
        assert_eq!(BitonicSort::new(1000).workgroup_count(256), 4);
        assert_eq!(BitonicSort::new(256).workgroup_count(256), 1);
        assert_eq!(BitonicSort::new(0).workgroup_count(64), 0);
        assert_eq!(BitonicSort::new(10).workgroup_count(0), 0);
    }

    #[test]
    fn key_buffer_bytes_pads_and_clamps() {
        assert_eq!(BitonicSort::new(3).key_buffer_bytes(), 4 * U32_STRIDE);
        assert_eq!(BitonicSort::new(0).key_buffer_bytes(), U32_STRIDE);
        assert_eq!(DISPATCH_ARGS_STRIDE, 12);
    }

    #[test]
    fn compare_distance_and_box_size_follow_shifts() {
        // Stage 3: passes 1,2,3 give distances 4,2,1; box size is 8.
        assert_eq!(CompareStep::new(3, 1).compare_distance(), 4);
        assert_eq!(CompareStep::new(3, 2).compare_distance(), 2);
        assert_eq!(CompareStep::new(3, 3).compare_distance(), 1);
        assert_eq!(CompareStep::new(3, 1).box_size(), 8);
    }

    #[test]
    fn degenerate_step_has_zero_distance() {
        assert_eq!(CompareStep::new(0, 0).compare_distance(), 0);
        assert_eq!(CompareStep::new(2, 3).compare_distance(), 0);
        // A zero distance makes the partner the element itself.
        assert_eq!(CompareStep::new(0, 0).compare_partner(7), 7);
    }

    #[test]
    fn compare_partner_is_xor_of_distance() {
        let step = CompareStep::new(3, 1); // distance 4
        assert_eq!(step.compare_partner(0), 4);
        assert_eq!(step.compare_partner(4), 0);
        assert_eq!(step.compare_partner(1), 5);
    }

    #[test]
    fn sort_ascending_follows_box_bit() {
        let step = CompareStep::new(2, 1); // box size 4
        assert!(step.sort_ascending(0));
        assert!(step.sort_ascending(3));
        assert!(!step.sort_ascending(4));
        assert!(!step.sort_ascending(7));
    }

    #[test]
    fn steps_length_matches_total_pass_count() {
        for count in [0usize, 1, 2, 3, 8, 100] {
            let sorter = BitonicSort::new(count);
            let steps = sorter.steps();
            assert_eq!(
                u32::try_from(steps.len()).expect("pass count fits u32"),
                sorter.total_pass_count()
            );
        }
        // Stage-2 network is exactly (1,1),(2,1),(2,2).
        assert_eq!(
            BitonicSort::new(4).steps(),
            [
                CompareStep::new(1, 1),
                CompareStep::new(2, 1),
                CompareStep::new(2, 2),
            ]
        );
    }

    #[test]
    fn sort_empty_and_single_are_noops() {
        assert!(BitonicSort::sort_keys(&[]).is_empty());
        assert_eq!(BitonicSort::sort_keys(&[42]), [42]);
    }

    #[test]
    fn sort_power_of_two_matches_reference() {
        let keys = [7u32, 3, 9, 1, 5, 2, 8, 4];
        let sorted = BitonicSort::sort_keys(&keys);
        assert_eq!(sorted, [1, 2, 3, 4, 5, 7, 8, 9]);
    }

    #[test]
    fn sort_non_power_of_two_matches_reference() {
        let keys = [5u32, 1, 4, 2, 3]; // length 5 pads to 8
        let sorted = BitonicSort::sort_keys(&keys);
        assert_eq!(sorted, [1, 2, 3, 4, 5]);
        // The returned slice keeps only the real keys, dropping the sentinels.
        assert_eq!(sorted.len(), keys.len());
    }

    #[test]
    fn sort_handles_already_sorted_and_reversed() {
        let ascending: Vec<u32> = (0..16).collect();
        assert_eq!(BitonicSort::sort_keys(&ascending), ascending);
        let descending: Vec<u32> = (0..16).rev().collect();
        assert_eq!(BitonicSort::sort_keys(&descending), ascending);
    }

    #[test]
    fn sort_matches_reference_over_randomized_inputs() {
        let mut lcg = Lcg::new(0x1234_5678);
        for len in [0usize, 1, 2, 3, 6, 7, 15, 33, 64, 100] {
            for _trial in 0..8 {
                let mut input: Vec<u32> = Vec::with_capacity(len);
                for _ in 0..len {
                    // Bound the values so duplicates appear and stress equality.
                    input.push(lcg.next_u32() % 50);
                }
                let mut expected = input.clone();
                expected.sort_unstable();
                assert_eq!(BitonicSort::sort_keys(&input), expected);
            }
        }
    }

    #[test]
    fn padding_sentinel_keeps_real_keys_when_max_present() {
        // A real key equal to the sentinel must still land in order because the
        // network is a total order over u32.
        let keys = [u32::MAX, 0, u32::MAX, 5];
        assert_eq!(BitonicSort::sort_keys(&keys), [0, 5, u32::MAX, u32::MAX]);
    }
}
