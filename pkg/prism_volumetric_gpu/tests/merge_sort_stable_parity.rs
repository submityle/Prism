//! Real-device parity for the stable `u32` merge-sort twin:
//! [`GpuMergeSortStable`](prism_volumetric_gpu::merge_sort_stable::GpuMergeSortStable)
//! must reproduce the `CPU` golden
//! [`merge_sort_stable`](prism_render_architecture::particle::merge_sort_stable)
//! both in the ascending key order and in the stable argsort permutation of
//! original indices.
//!
//! The fixtures cover the shapes the golden unit tests call out: the empty and
//! single-element no-ops, an already-sorted and a strictly reversed input, an
//! all-equal-keys array (whose argsort must be the identity, the sharpest
//! stability probe), duplicate-heavy arrays drawn from a small key domain to
//! force many ties, odd / power-of-two / non-power-of-two lengths that exercise
//! the ragged final run, a full-`u32`-range input proving the compare is purely
//! unsigned, and a maximum-capacity `64`-element array. The random fixtures use
//! a host-side `u64` linear-congruential generator so the suite needs no
//! external crate, and every key is an exact integer so there is no branch
//! threshold to steer away from.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The sort is pure unsigned integer work — `<=` comparisons, index additions
//! and a left shift — so the `CPU` and `GPU` results are bit-exact: both the
//! sorted keys and the argsort indices are compared with an exact `==`, lane for
//! lane over the valid `count` prefix.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::merge_sort_stable`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::merge_sort_stable::{merge_runs, merge_sort_u32};
use prism_volumetric_gpu::merge_sort_stable::GpuMergeSortStable;
use prism_volumetric_gpu::GpuContext;

/// A small deterministic linear-congruential generator so the fixtures need no
/// external crate. Returns a fresh integer each call.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes constants; full-period over `u64`.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

/// Computes the `CPU` golden stable argsort (the permutation of original
/// indices) by driving the reference `merge_runs` with the indices as payload.
fn golden_argsort(keys: &[u32]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..keys.len() as u32).collect();
    let mut carried = keys.to_vec();
    merge_runs(&mut idx, &mut carried);
    idx
}

/// Asserts the `GPU` sort of `keys` matches the `CPU` golden exactly: the sorted
/// keys reproduce `merge_sort_u32`, the argsort reproduces `merge_runs` carrying
/// the original indices, and every equal-key run keeps its original order.
fn assert_parity(gpu: &GpuMergeSortStable, ctx: &GpuContext, keys: &[u32]) {
    let got = gpu.sort(ctx, keys);
    let count = keys.len();
    assert_eq!(got.count as usize, count, "count mismatch for {keys:?}");

    let mut expected_sorted = keys.to_vec();
    merge_sort_u32(&mut expected_sorted);
    assert_eq!(
        &got.sorted[..count],
        expected_sorted.as_slice(),
        "sorted keys mismatch for {keys:?}"
    );

    let expected_argsort = golden_argsort(keys);
    assert_eq!(
        &got.argsort[..count],
        expected_argsort.as_slice(),
        "argsort mismatch for {keys:?}"
    );

    // Stability, read straight off the GPU output: within any equal-key run the
    // original indices must strictly increase.
    for pos in 1..count {
        if got.sorted[pos] == got.sorted[pos - 1] {
            assert!(
                got.argsort[pos] > got.argsort[pos - 1],
                "unstable tie at {pos} for {keys:?}: argsort {:?}",
                &got.argsort[..count]
            );
        }
    }

    // The argsort must be a genuine permutation of 0..count that carries each
    // original key to its sorted slot: keys[argsort[pos]] == sorted[pos].
    let mut seen = vec![false; count];
    for pos in 0..count {
        let src = got.argsort[pos] as usize;
        assert!(src < count, "argsort index {src} out of range for {keys:?}");
        assert!(!seen[src], "argsort repeats index {src} for {keys:?}");
        seen[src] = true;
        assert_eq!(
            keys[src], got.sorted[pos],
            "argsort[{pos}] does not carry its key for {keys:?}"
        );
    }
}

#[test]
fn empty_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMergeSortStable::new(&ctx);
    // No dispatch is issued; the result is a zeroed count-0 sort.
    let got = gpu.sort(&ctx, &[]);
    assert_eq!(got.count, 0);
    assert!(got.sorted.iter().all(|&k| k == 0));
    assert!(got.argsort.iter().all(|&i| i == 0));
}

#[test]
fn single_element_is_noop() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMergeSortStable::new(&ctx);
    assert_parity(&gpu, &ctx, &[42u32]);
}

#[test]
fn two_elements_sorted_and_swapped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMergeSortStable::new(&ctx);
    assert_parity(&gpu, &ctx, &[1u32, 2u32]);
    assert_parity(&gpu, &ctx, &[2u32, 1u32]);
}

#[test]
fn already_sorted_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMergeSortStable::new(&ctx);
    let keys: Vec<u32> = (0..40u32).collect();
    assert_parity(&gpu, &ctx, &keys);
}

#[test]
fn reversed_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMergeSortStable::new(&ctx);
    let keys: Vec<u32> = (0..40u32).rev().collect();
    assert_parity(&gpu, &ctx, &keys);
}

#[test]
fn all_equal_keys_argsort_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMergeSortStable::new(&ctx);
    // The sharpest stability probe: every key is equal, so the stable argsort
    // must be exactly 0, 1, 2, ... with no reordering.
    let keys = vec![9u32; 64];
    let got = gpu.sort(&ctx, &keys);
    assert_eq!(got.count, 64);
    let identity: Vec<u32> = (0..64u32).collect();
    assert_eq!(&got.argsort[..], identity.as_slice());
    assert_parity(&gpu, &ctx, &keys);
}

#[test]
fn duplicate_heavy_small_domain_is_stable() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMergeSortStable::new(&ctx);
    let mut rng = Lcg::new(0x0BAD_F00D);
    // A tiny key domain forces many ties so the stability guarantee is exercised
    // across every run width.
    for len in [7usize, 16, 31, 50, 64] {
        let keys: Vec<u32> = (0..len).map(|_| rng.next_u32() % 5).collect();
        assert_parity(&gpu, &ctx, &keys);
    }
}

#[test]
fn odd_and_power_of_two_lengths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMergeSortStable::new(&ctx);
    assert_parity(&gpu, &ctx, &[5u32, 3, 8, 1, 9, 2, 7]);
    assert_parity(&gpu, &ctx, &[8u32, 4, 6, 2, 7, 3, 5, 1]);
    assert_parity(&gpu, &ctx, &[5u32, 3, 8, 1, 9, 2, 7, 4, 6]);
}

#[test]
fn full_range_is_unsigned() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMergeSortStable::new(&ctx);
    // Values near both ends of the u32 range must not be reordered by any signed
    // interpretation; the compare is purely unsigned.
    let keys = vec![4_000_000_000u32, 5, 3_000_000_000, 1, u32::MAX, 0];
    assert_parity(&gpu, &ctx, &keys);
}

#[test]
fn random_full_capacity_batches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMergeSortStable::new(&ctx);
    let mut rng = Lcg::new(0x9E37_79B9);
    for len in [1usize, 2, 13, 33, 64] {
        // Mix a moderate domain so there are some ties but a varied order too.
        let keys: Vec<u32> = (0..len).map(|_| rng.next_u32() % 1000).collect();
        assert_parity(&gpu, &ctx, &keys);
    }
}
