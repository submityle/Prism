//! Real-device parity for the in-place `u32` heapsort twin:
//! [`GpuHeapSortU32`](prism_volumetric_gpu::heap_sort_u32::GpuHeapSortU32) must
//! reproduce the `CPU` golden
//! [`heap_sort`](prism_render_architecture::particle::heap_sort_u32::heap_sort)
//! and
//! [`is_max_heap`](prism_render_architecture::particle::heap_sort_u32::is_max_heap)
//! across empty and single-element arrays, arrays dense with duplicates,
//! already-ascending and strictly-reverse arrays, hand-built valid max-heaps,
//! and large pseudo-random arrays (broad-spread and heavily-clustered), all
//! batched one thread per array in a single dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Sorted keys and the max-heap flag are integers, so parity is asserted with
//! **exact equality**, not a float tolerance: there is no floating point on the
//! path and therefore no `ULP` boundary or degenerate region, and any mismatch
//! is a genuine port bug (a wrong child pick, a dropped swap, a miscounted heap
//! bound). The padding slots beyond `count` are asserted zero. `WGSL` has no
//! `u64`, and none appears in the twin; the `u64` here lives only in this
//! harness's `LCG`, mirroring the golden's own internal test generator so the
//! fixtures are reproducible across runs and platforms.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::heap_sort_u32`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::heap_sort_u32::{heap_sort, is_max_heap};
use prism_volumetric_gpu::heap_sort_u32::{GpuHeapSortU32, HeapSortU32Query, MAX_N};
use prism_volumetric_gpu::GpuContext;

/// A deterministic linear-congruential generator so the randomized fixtures are
/// reproducible bit for bit across runs and platforms. Only integer arithmetic
/// appears; the `u64` state never leaves this harness and is not part of the
/// twinned kernel.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Draws the next pseudo-random `u32` from `state`.
fn next_u32(state: &mut u64) -> u32 {
    (lcg_next(state) >> 32) as u32
}

/// Runs the `GPU` heapsort over a batch of arrays and asserts exact parity
/// against the `CPU` golden for every array: the sorted live prefix equals the
/// golden [`heap_sort`] output, the padding slots are zero, and the
/// `is_max_heap_input` flag matches the golden [`is_max_heap`] over the pristine
/// input.
fn check(ctx: &GpuContext, gpu: &GpuHeapSortU32, arrays: &[Vec<u32>]) {
    let queries: Vec<HeapSortU32Query> = arrays
        .iter()
        .map(|data| HeapSortU32Query { data: data.clone() })
        .collect();
    let got = gpu.eval(ctx, &queries);
    assert_eq!(got.len(), arrays.len(), "one result per array");

    for (index, (result, input)) in got.iter().zip(arrays.iter()).enumerate() {
        assert_eq!(
            result.count as usize,
            input.len(),
            "array {index}: live key count is preserved"
        );

        let mut want = input.clone();
        heap_sort(&mut want);
        assert_eq!(
            &result.sorted[..input.len()],
            want.as_slice(),
            "array {index}: sorted prefix matches the golden heap_sort"
        );
        assert!(
            result.sorted[input.len()..MAX_N].iter().all(|&v| v == 0),
            "array {index}: padding slots stay zero"
        );

        let want_heap = u32::from(is_max_heap(input));
        assert_eq!(
            result.is_max_heap_input, want_heap,
            "array {index}: is_max_heap_input matches the golden predicate"
        );
    }
}

#[test]
fn empty_batch_issues_no_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeapSortU32::new(&ctx);
    // No array to dispatch; the twin short-circuits before touching the device
    // and returns an empty result vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields no results");
}

#[test]
fn empty_and_single_element_arrays_are_noops() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeapSortU32::new(&ctx);
    // An empty live array (count 0) and single-element arrays are already
    // sorted and are trivially max-heaps.
    let arrays = vec![vec![], vec![42u32], vec![0u32], vec![u32::MAX]];
    check(&ctx, &gpu, &arrays);
}

#[test]
fn small_hand_cases_cover_order_and_duplicates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeapSortU32::new(&ctx);
    let arrays = vec![
        // Two-element orderings and an equal pair.
        vec![9u32, 4],
        vec![4u32, 9],
        vec![7u32, 7],
        // Already ascending, strictly reverse, and duplicate-dense.
        vec![1u32, 2, 3, 4, 5, 6, 7, 8],
        vec![8u32, 7, 6, 5, 4, 3, 2, 1],
        vec![5u32, 1, 5, 3, 1, 3, 5, 2],
        vec![3u32; 16],
        // Hand-built valid max-heap and a near-heap with one violation.
        vec![9u32, 7, 8, 4, 6, 1, 3],
        vec![9u32, 2, 8, 5, 6, 1, 3],
        // A strictly ascending multi-element array is not a max-heap.
        vec![1u32, 2, 3, 4, 5],
    ];
    check(&ctx, &gpu, &arrays);
}

#[test]
fn boundary_full_width_arrays() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeapSortU32::new(&ctx);
    // Full MAX_N width in several shapes: reverse, all-identical, and a
    // valid max-heap built from an ascending fill.
    let reverse: Vec<u32> = (0..MAX_N as u32).rev().collect();
    let identical = vec![123u32; MAX_N];
    let mut heapified: Vec<u32> = (1..=MAX_N as u32).collect();
    heap_sort(&mut heapified);
    // Reverse the ascending sort to obtain a descending array, which is a valid
    // max-heap (every parent dominates its children).
    heapified.reverse();
    check(&ctx, &gpu, &[reverse, identical, heapified]);
}

#[test]
fn random_broad_spread_full_u32_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeapSortU32::new(&ctx);
    // A batch of random arrays of assorted lengths across the full u32 range.
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    let mut arrays: Vec<Vec<u32>> = Vec::new();
    for len in [1usize, 2, 3, 7, 13, 31, 48, 63, MAX_N] {
        let array: Vec<u32> = (0..len).map(|_| next_u32(&mut state)).collect();
        arrays.push(array);
    }
    check(&ctx, &gpu, &arrays);
}

#[test]
fn random_narrow_domain_many_duplicates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeapSortU32::new(&ctx);
    // Keys crushed into a tiny domain so duplicates pile up and ties exercise
    // the `>=` dominance guard in `sift_down`.
    let mut state = 0x0bad_c0de_0f0f_0f0f_u64;
    let mut arrays: Vec<Vec<u32>> = Vec::new();
    for len in [5usize, 16, 33, MAX_N] {
        let array: Vec<u32> = (0..len).map(|_| next_u32(&mut state) % 7).collect();
        arrays.push(array);
    }
    check(&ctx, &gpu, &arrays);
}

#[test]
fn every_length_up_to_max_n() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeapSortU32::new(&ctx);
    // Exercise every live length from 0 to MAX_N in one batch, hitting the
    // parity boundaries and the `len / 2` last-parent edge cases.
    let mut state = 0xabcd_ef01_2345_6789_u64;
    let arrays: Vec<Vec<u32>> = (0..=MAX_N)
        .map(|len| (0..len).map(|_| next_u32(&mut state) % 1_000).collect())
        .collect();
    check(&ctx, &gpu, &arrays);
}
