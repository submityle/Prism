//! Real-device parity for the bitonic `u32`-sort twin:
//! [`GpuBitonicSortU32`](prism_volumetric_gpu::bitonic_sort::GpuBitonicSortU32)
//! must reproduce the `CPU` golden
//! [`BitonicSort::sort_keys`](prism_render_architecture::particle::bitonic_sort::BitonicSort::sort_keys)
//! element for element.
//!
//! The fixtures cover the shapes the golden calls out plus the stressors this
//! twin must honour: an empty array (host short-circuit), a single element,
//! two-element ascending/equal pairs, already-sorted and reverse-sorted runs,
//! heavy-duplicate arrays over a tiny key range, random keys from a host-side
//! `u64` `LCG`, several non-power-of-two lengths (which exercise the sentinel
//! padding), the `u32::MAX`-as-real-key case, and a full [`MAX_N`]-capacity
//! array. Every key is an integer produced directly or by the `LCG`, so the
//! fixtures stay pure and need no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every value on the path is a `u32`, so `CPU` and `GPU` must agree
//! **exactly**: the comparison is an exact `==` on the written count and on the
//! live sorted-key prefix, and every slot from `count` to [`MAX_N`] must be the
//! `u32::MAX` sentinel. Any mismatch is a genuine port bug, not a rounding
//! artifact.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::bitonic_sort::BitonicSort;
use prism_volumetric_gpu::bitonic_sort::{GpuBitonicSort, GpuBitonicSortU32, MAX_N};
use prism_volumetric_gpu::GpuContext;

/// A small deterministic linear-congruential generator so the fixtures need no
/// external crate.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes constants; full-period over `u64`.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }
}

/// Asserts the twin reproduces the golden ascending sort of `keys` exactly.
///
/// Checks the written count, the live sorted prefix against
/// [`BitonicSort::sort_keys`](prism_render_architecture::particle::bitonic_sort::BitonicSort::sort_keys),
/// and that every slot past the live count holds the `u32::MAX` sentinel.
fn assert_parity(gpu: &GpuBitonicSortU32, ctx: &GpuContext, keys: &[u32]) {
    let got: GpuBitonicSort = gpu.evaluate(ctx, keys);
    assert_eq!(
        got.count as usize,
        keys.len(),
        "written count must equal the input length"
    );

    let cpu_sorted = BitonicSort::sort_keys(keys);
    assert_eq!(
        got.sorted_keys(),
        cpu_sorted.as_slice(),
        "sorted keys must match the CPU golden bitonic sort"
    );

    // Every slot past the live count must be the sentinel padding.
    for (lane, &slot) in got.sorted.iter().enumerate().skip(keys.len()) {
        assert_eq!(
            slot,
            u32::MAX,
            "sorted lane {lane} past count must be the u32::MAX sentinel"
        );
    }
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitonicSortU32::new(&ctx);
    // No dispatch is issued; the result is the all-sentinel sort with count 0.
    let got = gpu.evaluate(&ctx, &[]);
    assert_eq!(got.count, 0);
    assert!(got.sorted_keys().is_empty());
    assert_eq!(got.sorted, [u32::MAX; MAX_N]);
}

#[test]
fn single_element_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitonicSortU32::new(&ctx);
    assert_parity(&gpu, &ctx, &[0x1234_5678]);
}

#[test]
fn two_elements_ascending_and_equal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitonicSortU32::new(&ctx);
    // Descending pair sorts to [3, 7]; equal pair stays [5, 5].
    assert_parity(&gpu, &ctx, &[7, 3]);
    assert_parity(&gpu, &ctx, &[5, 5]);
}

#[test]
fn power_of_two_mixed_keys() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitonicSortU32::new(&ctx);
    assert_parity(&gpu, &ctx, &[7, 3, 9, 1, 5, 2, 8, 4]);
}

#[test]
fn already_sorted_and_reversed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitonicSortU32::new(&ctx);
    let ascending: Vec<u32> = (0..64).collect();
    assert_parity(&gpu, &ctx, &ascending);
    let descending: Vec<u32> = (0..64).rev().collect();
    assert_parity(&gpu, &ctx, &descending);
}

#[test]
fn heavy_duplicates_tiny_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitonicSortU32::new(&ctx);
    // A tiny key range over many keys guarantees heavy duplication; equal keys
    // are indistinguishable so the ascending order is still exact.
    let mut rng = Lcg::new(0x5EED_5EED);
    let keys: Vec<u32> = (0..100).map(|_| rng.next_u32() % 5).collect();
    assert_parity(&gpu, &ctx, &keys);
    let patterned = [2u32, 1, 2, 1, 2, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1];
    assert_parity(&gpu, &ctx, &patterned);
}

#[test]
fn non_power_of_two_lengths_pad() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitonicSortU32::new(&ctx);
    // Lengths that are not powers of two force the sentinel padding into the
    // network; the live prefix must still come back fully ordered.
    assert_parity(&gpu, &ctx, &[5, 1, 4, 2, 3]);
    let mut rng = Lcg::new(0x0BAD_F00D);
    for len in [3usize, 6, 7, 15, 33, 100, 500, 777] {
        let keys: Vec<u32> = (0..len).map(|_| rng.next_u32() % 1000).collect();
        assert_parity(&gpu, &ctx, &keys);
    }
}

#[test]
fn max_sentinel_as_real_key() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitonicSortU32::new(&ctx);
    // Real keys equal to the padding sentinel must still land in order because
    // the network is a total order over u32.
    assert_parity(&gpu, &ctx, &[u32::MAX, 0, u32::MAX, 5]);
    assert_parity(&gpu, &ctx, &[u32::MAX, u32::MAX, 0, 1, u32::MAX, 2, 0]);
}

#[test]
fn random_full_width_keys() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitonicSortU32::new(&ctx);
    // Full 32-bit keys exercise the whole key range with few ties.
    let mut rng = Lcg::new(0x2468_ACE0);
    let keys: Vec<u32> = (0..128).map(|_| rng.next_u32()).collect();
    assert_parity(&gpu, &ctx, &keys);
}

#[test]
fn full_capacity_mixed_keys() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitonicSortU32::new(&ctx);
    // A full MAX_N-length array (already a power of two, so no padding) mixing a
    // moderate range so duplicates appear across the full capacity.
    let mut rng = Lcg::new(0x1357_9BDF);
    let keys: Vec<u32> = (0..MAX_N).map(|_| rng.next_u32() % 500).collect();
    assert_eq!(keys.len(), MAX_N);
    assert_parity(&gpu, &ctx, &keys);
}
