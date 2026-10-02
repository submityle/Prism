//! Real-device parity for the trail ring-ordering twin:
//! [`GpuRibbonTrail`](prism_volumetric_gpu::ribbon_trail::GpuRibbonTrail) must
//! reproduce the `CPU` golden
//! [`iter_ordered_indices`](prism_render_architecture::particle::ribbon_trail::iter_ordered_indices)
//! across the resolved ring `slot`, the ordered-run length `ordered_len` and
//! the `valid` flag, over a structured spread of rings — a filling ring, a
//! saturated no-wrap ring, saturated wrapping rings with the head at and near
//! the capacity, the `capacity == 1` boundary, the empty (`count == 0`) ring
//! and the degenerate `capacity == 0` ring — plus a large random batch spanning
//! many workgroups.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every value is a `u32` ring slot / length or a discrete `1`/`0` flag with no
//! rounding anywhere on the path, so the outputs are bit-identical and asserted
//! with exact `==` and no tolerance. Fixtures use rejection sampling to keep
//! `head < capacity` and `count <= capacity` (so the single conditional
//! subtract reproduces the golden `u64` modulo exactly) and keep `head + count`
//! far below [`u32::MAX`], so the device `u32` add never wraps. Several
//! scenarios additionally assert a genuine wrap (a non-monotone slot sequence)
//! and a mix of valid and out-of-range elements, so a degenerate constant
//! kernel could not pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ribbon_trail`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::ribbon_trail::iter_ordered_indices;
use prism_volumetric_gpu::ribbon_trail::{
    GpuRibbonTrail, GpuRibbonTrailQuery, GpuRibbonTrailResult,
};
use prism_volumetric_gpu::GpuContext;

/// Computes the golden answer for one query by calling the `CPU` reference
/// directly, matching exactly what the twin must reproduce.
fn golden_result(q: GpuRibbonTrailQuery) -> GpuRibbonTrailResult {
    let ordered = iter_ordered_indices(q.head, q.count, q.capacity);
    let ordered_len = ordered.len() as u32;
    if q.element < ordered_len {
        GpuRibbonTrailResult {
            slot: ordered[q.element as usize] as u32,
            valid: 1,
            ordered_len,
            pad0: 0,
        }
    } else {
        GpuRibbonTrailResult {
            slot: 0,
            valid: 0,
            ordered_len,
            pad0: 0,
        }
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuRibbonTrail,
    queries: &[GpuRibbonTrailQuery],
) -> Vec<GpuRibbonTrailResult> {
    let results = gpu.evaluate(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (&q, &r) in queries.iter().zip(results.iter()) {
        assert_eq!(r, golden_result(q), "mismatch for query {q:?}");
    }
    results
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns a raw `u64` state word.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Pushes one query per ordered element for the ring `(head, count, capacity)`,
/// plus two indices past the ordered run so the out-of-range (`valid == 0`)
/// path is exercised too.
fn push_ring(queries: &mut Vec<GpuRibbonTrailQuery>, head: u32, count: u32, capacity: u32) {
    let ordered_len = iter_ordered_indices(head, count, capacity).len() as u32;
    for element in 0..ordered_len.saturating_add(2) {
        queries.push(GpuRibbonTrailQuery {
            head,
            count,
            capacity,
            element,
        });
    }
}

/// Builds the structured ring fixtures: a filling ring, a saturated no-wrap
/// ring, saturated wrapping rings (head at and near capacity), the
/// `capacity == 1` boundary, the empty ring and the degenerate zero-capacity
/// ring.
fn structured_queries() -> Vec<GpuRibbonTrailQuery> {
    let mut queries = Vec::new();
    // Filling ring: head 0, three of eight slots used -> slots 0,1,2.
    push_ring(&mut queries, 0, 3, 8);
    // Saturated, no wrap: head 0, full -> slots 0..4.
    push_ring(&mut queries, 0, 4, 4);
    // Saturated, wrap crossing: head 2, full -> slots 2,3,0,1.
    push_ring(&mut queries, 2, 4, 4);
    // Saturated, head adjacent to capacity: head 3, full -> slots 3,0,1,2.
    push_ring(&mut queries, 3, 4, 4);
    // Saturated single-slot ring: capacity 1 -> slot 0.
    push_ring(&mut queries, 0, 1, 1);
    // Empty ring: zero captures -> no ordered element.
    push_ring(&mut queries, 0, 0, 8);
    // Degenerate zero-capacity ring -> empty ordering.
    push_ring(&mut queries, 0, 0, 0);
    // A larger saturated wrap, head near capacity, to span many slots.
    push_ring(&mut queries, 1000, 1024, 1024);
    queries
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_structured_rings() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ribbon-trail parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuRibbonTrail::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: both validity classes appear (ordered elements and the two
    // out-of-range probes per ring), so a degenerate always-valid kernel fails.
    let valid = results.iter().filter(|r| r.valid == 1).count();
    let invalid = results.iter().filter(|r| r.valid == 0).count();
    assert!(valid > 0, "fixture must include in-range ordered elements");
    assert!(invalid > 0, "fixture must include out-of-range probes");

    // A genuine wrap: the head-2, full capacity-4 ring produces the non-monotone
    // slot sequence 2,3,0,1, which a non-wrapping identity kernel could not emit.
    let wrap: Vec<u32> = (0..4)
        .map(|element| {
            golden_result(GpuRibbonTrailQuery {
                head: 2,
                count: 4,
                capacity: 4,
                element,
            })
            .slot
        })
        .collect();
    assert_eq!(wrap, [2, 3, 0, 1], "head-2 full ring must wrap");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_full_ring_slot_permutation() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ribbon-trail parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuRibbonTrail::new(&ctx);

    // For every head of a saturated capacity-16 ring, the ordered slots must be
    // a permutation of 0..16 (a rotation starting at head), proving the wrap is
    // exact for all heads, not just the sampled ones.
    let capacity = 16u32;
    let mut queries = Vec::new();
    for head in 0..capacity {
        push_ring(&mut queries, head, capacity, capacity);
    }
    let results = check(&ctx, &gpu, &queries);

    let mut cursor = 0usize;
    for head in 0..capacity {
        let mut slots = Vec::new();
        for _ in 0..capacity {
            let r = results[cursor];
            cursor += 1;
            assert_eq!(r.valid, 1, "ordered element must be valid");
            slots.push(r.slot);
        }
        // Skip the two out-of-range probes push_ring appended.
        cursor += 2;
        let mut sorted = slots.clone();
        sorted.sort_unstable();
        let expected: Vec<u32> = (0..capacity).collect();
        assert_eq!(
            sorted, expected,
            "head {head} slots must permute 0..capacity"
        );
        assert_eq!(slots[0], head, "ordered run must start at the head slot");
    }
}

/// Draws a random in-contract ring query via rejection sampling: `capacity` in
/// `1..=4096`, `head` in `0..capacity`, `count` in `0..=capacity`, and
/// `element` spanning the ordered run plus a little beyond. Keeping
/// `head < capacity` and `count <= capacity` makes the single conditional
/// subtract bit-exact against the golden `u64` modulo.
fn ring_query(state: &mut u64) -> GpuRibbonTrailQuery {
    let capacity = 1 + (lcg(state) >> 40) as u32 % 4096;
    let head = (lcg(state) >> 40) as u32 % capacity;
    let count = (lcg(state) >> 40) as u32 % (capacity + 1);
    let span = capacity + 2;
    let element = (lcg(state) >> 40) as u32 % span;
    GpuRibbonTrailQuery {
        head,
        count,
        capacity,
        element,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ribbon-trail parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuRibbonTrail::new(&ctx);

    // Several thousand in-contract queries spanning many workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(ring_query(&mut state));
    }
    let results = check(&ctx, &gpu, &queries);

    // The batch mixes valid and out-of-range elements, so a degenerate
    // single-class kernel could not pass.
    let any_valid = results.iter().any(|r| r.valid == 1);
    let any_invalid = results.iter().any(|r| r.valid == 0);
    assert!(any_valid, "random batch should include valid elements");
    assert!(
        any_invalid,
        "random batch should include out-of-range probes"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ribbon-trail parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuRibbonTrail::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
