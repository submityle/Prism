//! `GPU`-faithful stable LSD radix sort over 64-bit coherence keys.
//!
//! [`plan::plan_reorder`](super::plan::plan_reorder) builds the reorder plan
//! with a comparison sort — the clearest statement of *what* the ordering is.
//! A `GPU` cannot run a comparison sort efficiently; the production device path
//! is a **least-significant-digit (LSD) radix sort**, exactly the scheme the
//! engine's compute kernels already use for 32-bit keys
//! (`prism_physics_gpu::radix`). This module is the device-faithful `CPU`
//! golden for the 64-bit case: it decomposes the key into
//! [`PASSES`] passes of [`RADIX_BITS`]-bit digits and runs one stable
//! counting-sort pass per digit, carrying the ray index as the payload.
//!
//! Because every pass is a *stable* counting sort, equal keys keep their
//! ascending source order, so [`radix_order`] returns the **same permutation**
//! as the comparison-sort plan — this equivalence is asserted over randomised
//! inputs in the tests, and is the device-free half of eventual `GPU` parity:
//! a device radix kernel reproduces this permutation bit-for-bit.
//!
//! A 64-bit key needs `64 / 8 = 8` passes (versus 4 for a 32-bit key); the
//! digit geometry is otherwise identical to the proven 32-bit kernels, so the
//! same histogram → exclusive-scan → stable-scatter decomposition ports
//! directly to two cascaded 32-bit (low word, then high word) dispatches or one
//! 8-pass 64-bit dispatch on the device.
//!
//! # References
//! - Blelloch, *Prefix Sums and Their Applications*, 1990 (scan-based bucketing).
//! - Satish, Harris, Garland, *Designing Efficient Sorting Algorithms for
//!   Manycore GPUs*, IPDPS 2009 (per-block histogram LSD radix sort).

use super::plan::{finalize_plan, ReorderPlan};
use super::sort_key::CoherenceKey;

/// Bits consumed per radix pass (8-bit digit → 256 buckets).
pub const RADIX_BITS: u32 = 8;
/// Buckets per pass, `2^RADIX_BITS`.
pub const RADIX: u32 = 1 << RADIX_BITS;
/// Digit mask, `RADIX - 1`.
pub const RADIX_MASK: u64 = (RADIX as u64) - 1;
/// Passes needed to consume a 64-bit key, `64 / RADIX_BITS`.
pub const PASSES: u32 = u64::BITS / RADIX_BITS;

/// Extracts the `pass`-th [`RADIX_BITS`]-bit digit of `key`.
#[must_use]
fn digit(key: u64, pass: u32) -> usize {
    ((key >> (pass * RADIX_BITS)) & RADIX_MASK) as usize
}

/// Returns the stable permutation ordering `keys` ascending by their raw 64-bit
/// value, computed with an LSD radix sort.
///
/// `order[i]` is the source index placed at reordered slot `i`. Equal keys keep
/// their ascending source order (stable), so the result is identical to the
/// comparison-sort permutation in
/// [`plan::plan_reorder`](super::plan::plan_reorder). Empty input yields an
/// empty permutation.
#[must_use]
pub fn radix_order(keys: &[CoherenceKey]) -> Vec<u32> {
    let n = keys.len();
    if n == 0 {
        return Vec::new();
    }

    // Payload = source index; the key array is immutable, so we permute indices.
    let mut src: Vec<u32> = (0..n as u32).collect();
    let mut dst: Vec<u32> = vec![0u32; n];

    for pass in 0..PASSES {
        let mut counts = [0u32; RADIX as usize];
        for &i in &src {
            counts[digit(keys[i as usize].raw(), pass)] += 1;
        }
        // Exclusive scan → per-digit output base (stable bucket layout).
        let mut offsets = [0u32; RADIX as usize];
        let mut running = 0u32;
        for (offset, &count) in offsets.iter_mut().zip(counts.iter()) {
            *offset = running;
            running += count;
        }
        for &i in &src {
            let d = digit(keys[i as usize].raw(), pass);
            dst[offsets[d] as usize] = i;
            offsets[d] += 1;
        }
        std::mem::swap(&mut src, &mut dst);
    }
    src
}

/// Builds the reorder plan via the `GPU`-faithful radix sort.
///
/// Produces a [`ReorderPlan`] identical to
/// [`plan::plan_reorder`](super::plan::plan_reorder) (same `order`, batches, and
/// stats) but derives the permutation with [`radix_order`] rather than a
/// comparison sort, so it exercises the exact algorithm a device kernel runs.
#[must_use]
pub fn plan_reorder_radix(keys: &[CoherenceKey], batch_shift: u32) -> ReorderPlan {
    if keys.is_empty() {
        return ReorderPlan::default();
    }
    let order = radix_order(keys);
    finalize_plan(keys, order, batch_shift)
}

#[cfg(test)]
mod tests {
    use super::super::plan::plan_reorder;
    use super::super::sort_key::{CoherenceKey, CoherenceKeyLayout};
    use super::*;

    fn keys(raws: &[u64]) -> Vec<CoherenceKey> {
        raws.iter().copied().map(CoherenceKey).collect()
    }

    // Small xorshift so the property tests stay dependency-free and reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(radix_order(&[]).is_empty());
        assert!(plan_reorder_radix(&[], 0).is_empty());
    }

    #[test]
    fn digit_geometry_consumes_the_whole_key() {
        assert_eq!(PASSES * RADIX_BITS, u64::BITS);
        assert_eq!(RADIX, 256);
    }

    #[test]
    fn radix_matches_comparison_order_on_random_keys() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for case in 0..64 {
            let n = (rng.next() % 300) as usize;
            // Mask to a few high bits sometimes to force many ties.
            let mask = if case % 2 == 0 { u64::MAX } else { 0xFF };
            let raws: Vec<u64> = (0..n).map(|_| rng.next() & mask).collect();
            let k = keys(&raws);
            let radix = radix_order(&k);
            let golden = plan_reorder(&k, 0).order;
            assert_eq!(radix, golden, "case {case}, n={n}, mask={mask:#x}");
        }
    }

    #[test]
    fn radix_is_stable_on_equal_keys() {
        // All keys equal → identity permutation (ascending source order).
        let k = keys(&[42; 16]);
        let order = radix_order(&k);
        let identity: Vec<u32> = (0..16).collect();
        assert_eq!(order, identity);
    }

    #[test]
    fn radix_order_is_non_decreasing() {
        let k = keys(&[9, 1, 1, 8, 3, 3, 3, 0, u64::MAX, 1 << 40]);
        let order = radix_order(&k);
        let sorted: Vec<u64> = order.iter().map(|&i| k[i as usize].raw()).collect();
        assert!(sorted.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn plan_reorder_radix_equals_comparison_plan() {
        let layout = CoherenceKeyLayout::balanced([-1.0; 3], [1.0; 3]).unwrap();
        let k: Vec<CoherenceKey> = [
            (3u32, [0.0f32, 1.0, 0.0], [0.1f32, 0.1, 0.1]),
            (1, [1.0, 0.0, 0.0], [0.5, 0.2, 0.9]),
            (3, [0.0, 0.0, 1.0], [0.3, 0.7, 0.2]),
            (1, [-1.0, 0.0, 0.0], [0.8, 0.1, 0.4]),
            (3, [0.0, -1.0, 0.0], [0.6, 0.6, 0.6]),
            (2, [0.0, 0.0, -1.0], [0.2, 0.2, 0.2]),
        ]
        .iter()
        .map(|&(m, d, o)| layout.encode(m, d, o))
        .collect();

        for &shift in &[0u32, layout.material_shift(), 64] {
            assert_eq!(
                plan_reorder_radix(&k, shift),
                plan_reorder(&k, shift),
                "shift={shift}"
            );
        }
    }
}
