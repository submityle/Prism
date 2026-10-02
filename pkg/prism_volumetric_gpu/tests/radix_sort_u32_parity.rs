//! Real-device parity for the `u32` `radix`-sort twin:
//! [`GpuRadixSortU32`](prism_volumetric_gpu::radix_sort_u32::GpuRadixSortU32)
//! must reproduce the `CPU` golden
//! [`radix_sort_u32`](prism_render_architecture::particle::radix_sort_u32)
//! across both entry points — the sorted keys
//! ([`sort`](prism_render_architecture::particle::radix_sort_u32::sort)) and the
//! stable permutation
//! ([`argsort`](prism_render_architecture::particle::radix_sort_u32::argsort)).
//!
//! The fixtures cover the shapes the golden unit tests call out plus the
//! stability stressors this twin must honour: an empty array (host
//! short-circuit), a single element, two-element ascending and equal pairs, an
//! all-equal array (whose `argsort` must be the identity), heavy-duplicate
//! arrays over a tiny key range (the core stability check), the zero/`u32::MAX`
//! extremes, a reverse-sorted array, random full-width keys, random
//! high-byte-only keys that exercise the final pass, a payload-tracking
//! stability case, and a full `MAX_N`-capacity array. Every key is an integer
//! produced directly or by a host-side `u64` `LCG`, so the fixtures stay pure
//! and need no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every value on the path is a `u32`, so `CPU` and `GPU` must agree **exactly**:
//! the comparison is an exact `==` on the written count, on the sorted-key
//! prefix and on the `argsort` permutation prefix. Stability is verified by
//! requiring the `argsort` permutation to equal the golden's element for
//! element, so equal keys must keep their original order.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::radix_sort_u32::{argsort, sort};
use prism_volumetric_gpu::radix_sort_u32::{GpuRadixSort, GpuRadixSortU32, MAX_N};
use prism_volumetric_gpu::GpuContext;

/// A small deterministic linear-congruential generator so the fixtures need no
/// external crate. Mirrors the golden's own test `RNG`.
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

/// Asserts the twin reproduces both golden entry points for `keys` exactly.
///
/// Checks the written count, the sorted-key prefix against
/// [`sort`](prism_render_architecture::particle::radix_sort_u32::sort), the
/// `argsort` prefix against
/// [`argsort`](prism_render_architecture::particle::radix_sort_u32::argsort),
/// that the permutation reproduces the sorted keys, and that lanes past the
/// written count are left zero.
fn assert_parity(gpu: &GpuRadixSortU32, ctx: &GpuContext, keys: &[u32]) {
    let got: GpuRadixSort = gpu.evaluate(ctx, keys);
    assert_eq!(
        got.count as usize,
        keys.len(),
        "written count must equal the input length"
    );

    let cpu_sorted = sort(keys);
    assert_eq!(
        got.sorted_keys(),
        cpu_sorted.as_slice(),
        "sorted keys must match the CPU golden sort"
    );

    let cpu_arg = argsort(keys);
    let got_arg: Vec<usize> = got.argsort_indices().iter().map(|&i| i as usize).collect();
    assert_eq!(
        got_arg, cpu_arg,
        "argsort permutation must match the CPU golden argsort (stable)"
    );

    // The permutation must index the original keys back into sorted order.
    let via_perm: Vec<u32> = got_arg.iter().map(|&i| keys[i]).collect();
    assert_eq!(
        via_perm, cpu_sorted,
        "argsort permutation must reproduce the sorted keys"
    );

    // Lanes past the written count must stay zero.
    for lane in got.count as usize..MAX_N {
        assert_eq!(
            got.sorted[lane], 0,
            "sorted lane {lane} past count must be 0"
        );
        assert_eq!(
            got.argsort[lane], 0,
            "argsort lane {lane} past count must be 0"
        );
    }
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixSortU32::new(&ctx);
    // No dispatch is issued; the result is the all-zero sort with count 0.
    let got = gpu.evaluate(&ctx, &[]);
    assert_eq!(got.count, 0);
    assert!(got.sorted_keys().is_empty());
    assert!(got.argsort_indices().is_empty());
    assert_eq!(got.sorted, [0u32; MAX_N]);
    assert_eq!(got.argsort, [0u32; MAX_N]);
}

#[test]
fn single_element_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixSortU32::new(&ctx);
    assert_parity(&gpu, &ctx, &[0x1234_5678]);
}

#[test]
fn two_elements_ascending_and_equal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixSortU32::new(&ctx);
    // Descending pair sorts to [3, 7] with permutation [1, 0].
    assert_parity(&gpu, &ctx, &[7, 3]);
    // Equal pair keeps original order: permutation [0, 1].
    assert_parity(&gpu, &ctx, &[5, 5]);
}

#[test]
fn all_equal_keys_argsort_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixSortU32::new(&ctx);
    // Every key equal: the stable permutation must be the identity order.
    let keys = [4u32; 64];
    assert_parity(&gpu, &ctx, &keys);
}

#[test]
fn heavy_duplicates_tiny_range_are_stable() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixSortU32::new(&ctx);
    // A tiny key range over many keys guarantees heavy duplication, the core
    // stability stressor: every equal-key run must preserve original order.
    let mut rng = Lcg::new(0x5EED_5EED);
    let keys: Vec<u32> = (0..64).map(|_| rng.next_u32() % 5).collect();
    assert_parity(&gpu, &ctx, &keys);
    // A hand-built duplicate pattern with known ties.
    let patterned = [2u32, 1, 2, 1, 2, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1];
    assert_parity(&gpu, &ctx, &patterned);
}

#[test]
fn payload_tracking_stability() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixSortU32::new(&ctx);
    // Duplicate keys interleaved: the argsort must keep source indices ascending
    // within each equal-key group, which assert_parity verifies against the
    // golden permutation directly.
    let keys = [5u32, 5, 1, 5, 1, 5, 1, 9, 1, 5, 1, 9, 5, 1, 9, 5];
    assert_parity(&gpu, &ctx, &keys);
}

#[test]
fn zero_and_max_extremes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixSortU32::new(&ctx);
    let keys = [u32::MAX, 0, 3, u32::MAX, 0, 1, u32::MAX, 0];
    assert_parity(&gpu, &ctx, &keys);
}

#[test]
fn reverse_sorted_full() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixSortU32::new(&ctx);
    let keys: Vec<u32> = (0..64u32).rev().collect();
    assert_parity(&gpu, &ctx, &keys);
}

#[test]
fn random_full_width_keys() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixSortU32::new(&ctx);
    // Full 32-bit keys exercise all four passes with few ties.
    let mut rng = Lcg::new(0x0BAD_F00D);
    let keys: Vec<u32> = (0..64).map(|_| rng.next_u32()).collect();
    assert_parity(&gpu, &ctx, &keys);
}

#[test]
fn random_high_bytes_only() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixSortU32::new(&ctx);
    // Only the most-significant byte varies, so the first three passes are
    // no-ops and the last pass does the ordering.
    let mut rng = Lcg::new(0xABCD_0001);
    let keys: Vec<u32> = (0..48).map(|_| rng.next_u32() & 0xFF00_0000).collect();
    assert_parity(&gpu, &ctx, &keys);
}

#[test]
fn full_capacity_mixed_keys() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixSortU32::new(&ctx);
    // A full MAX_N-length array mixing a moderate range (so duplicates appear)
    // with the full capacity the fixed-length buffers allow.
    let mut rng = Lcg::new(0x2468_ACE0);
    let keys: Vec<u32> = (0..MAX_N).map(|_| rng.next_u32() % 17).collect();
    assert_eq!(keys.len(), MAX_N);
    assert_parity(&gpu, &ctx, &keys);
}
