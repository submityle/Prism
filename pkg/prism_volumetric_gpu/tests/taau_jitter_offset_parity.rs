//! Real-device parity for the deterministic jitter twin:
//! [`GpuTaauJitterOffset`](prism_volumetric_gpu::taau_jitter_offset::GpuTaauJitterOffset)
//! must reproduce the `CPU` golden
//! [`radical_inverse`](prism_render_architecture::temporal_upscale::jitter::radical_inverse)
//! and
//! [`offset_for_phase`](prism_render_architecture::temporal_upscale::jitter::JitterSequence::offset_for_phase)
//! across the known base-`2`/base-`3` radical inverses, a degenerate base, a
//! whole-period phase scan, and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values are produced by calling the golden
//! [`radical_inverse`](prism_render_architecture::temporal_upscale::jitter::radical_inverse)
//! directly for the raw radical inverses, and the golden
//! [`offset_for_phase`](prism_render_architecture::temporal_upscale::jitter::JitterSequence::offset_for_phase)
//! for the Halton `(2, 3)` offset. For non-`(2, 3)` bases the offset is pinned
//! against the same golden `radical_inverse` closed form (the recenter is a
//! plain `- 0.5`), and `grounding_offset_matches_sequence` proves that closed
//! form reproduces `offset_for_phase` for bases `(2, 3)`.
//!
//! # Parity criterion
//!
//! The radical inverse threads a `u32` digit through an integer modulo/divide
//! loop and accumulates `f32` products, so a `GPU` reciprocal and fused
//! multiply-add may land a few units in the last place from the scalar
//! reference. Every continuous output is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::jitter`；无第三方引擎源码或衍生代码。

use prism_render_architecture::temporal_upscale::jitter::{
    radical_inverse, JitterSequence, HALTON_BASE_X, HALTON_BASE_Y,
};
use prism_volumetric_gpu::taau_jitter_offset::{
    GpuTaauJitterOffset, TaauJitterOffsetQuery, TaauJitterOffsetResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` reciprocal and fused multiply-add may land a
/// few units in the last place from the scalar reference; `1e-4` admits that
/// legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Reconstructs the golden result in-host: raw radical inverses from
/// `radical_inverse`, and the Halton offset from `offset_for_phase` for bases
/// `(2, 3)` or from the same `radical_inverse` closed form otherwise.
fn oracle(q: &TaauJitterOffsetQuery) -> TaauJitterOffsetResult {
    let rinv_x = radical_inverse(q.base_x, q.index);
    let rinv_y = radical_inverse(q.base_y, q.index);
    let (offset_x, offset_y) = if q.base_x == HALTON_BASE_X && q.base_y == HALTON_BASE_Y {
        // The golden `offset_for_phase` fixes bases (2, 3); its period does not
        // affect the per-phase offset, so any sequence works as the oracle.
        let [ox, oy] = JitterSequence::default().offset_for_phase(q.phase);
        (ox, oy)
    } else {
        let pi = q.phase.wrapping_add(1);
        (
            radical_inverse(q.base_x, pi) - 0.5,
            radical_inverse(q.base_y, pi) - 0.5,
        )
    };
    TaauJitterOffsetResult {
        rinv_x,
        rinv_y,
        offset_x,
        offset_y,
    }
}

/// Pins one `GPU` result against the in-host oracle: every continuous output
/// within tolerance.
fn check_query(idx: usize, got: &TaauJitterOffsetResult, want: &TaauJitterOffsetResult) {
    assert!(
        close(got.rinv_x, want.rinv_x),
        "query {idx} rinv_x: gpu {} vs cpu {}",
        got.rinv_x,
        want.rinv_x
    );
    assert!(
        close(got.rinv_y, want.rinv_y),
        "query {idx} rinv_y: gpu {} vs cpu {}",
        got.rinv_y,
        want.rinv_y
    );
    assert!(
        close(got.offset_x, want.offset_x),
        "query {idx} offset_x: gpu {} vs cpu {}",
        got.offset_x,
        want.offset_x
    );
    assert!(
        close(got.offset_y, want.offset_y),
        "query {idx} offset_y: gpu {} vs cpu {}",
        got.offset_y,
        want.offset_y
    );
}

/// Dispatches `queries` and checks every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuTaauJitterOffset, queries: &[TaauJitterOffsetQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_query(idx, g, &want);
    }
}

/// A small `LCG` for the randomized sweep (host-only; the kernel is portable).
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

#[test]
fn empty_batch_produces_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauJitterOffset::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn known_radical_inverses_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauJitterOffset::new(&ctx);
    // The textbook base-2/base-3 radical inverses: index 0 is the origin,
    // index 1/2/3 base-2 are 0.5/0.25/0.75, index 1 base-3 is 1/3.
    let queries = [
        TaauJitterOffsetQuery::new(HALTON_BASE_X, HALTON_BASE_Y, 0, 0),
        TaauJitterOffsetQuery::new(HALTON_BASE_X, HALTON_BASE_Y, 1, 0),
        TaauJitterOffsetQuery::new(HALTON_BASE_X, HALTON_BASE_Y, 2, 1),
        TaauJitterOffsetQuery::new(HALTON_BASE_X, HALTON_BASE_Y, 3, 2),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn large_indices_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauJitterOffset::new(&ctx);
    // Deep digit expansions: many loop iterations stress the running reciprocal
    // weight accumulation.
    let queries = [
        TaauJitterOffsetQuery::new(HALTON_BASE_X, HALTON_BASE_Y, 1023, 1022),
        TaauJitterOffsetQuery::new(HALTON_BASE_X, HALTON_BASE_Y, 65_535, 4096),
        TaauJitterOffsetQuery::new(5, 7, 999_983, 12_345),
        TaauJitterOffsetQuery::new(11, 13, 1_000_000, 7),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_bases_return_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauJitterOffset::new(&ctx);
    // A base below 2 has no positional expansion: the raw radical inverse is 0
    // and the recentered offset collapses to -0.5 on that axis.
    let queries = [
        TaauJitterOffsetQuery::new(0, 1, 7, 3),
        TaauJitterOffsetQuery::new(1, 0, 42, 10),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn full_period_phase_scan_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauJitterOffset::new(&ctx);
    // Scan a whole reconstruction window's worth of phases, pinning the Halton
    // (2, 3) offset against the golden `offset_for_phase` for each.
    let queries: Vec<TaauJitterOffsetQuery> = (0u32..64)
        .map(|phase| TaauJitterOffsetQuery::new(HALTON_BASE_X, HALTON_BASE_Y, phase, phase))
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauJitterOffset::new(&ctx);
    let bases = [2u32, 3, 5, 7, 11];
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries: Vec<TaauJitterOffsetQuery> = Vec::new();
    // Several workgroups' worth of random bases, indices and phases pin the
    // digit loop across a wide span.
    for _ in 0..512 {
        let base_x = bases[(lcg(&mut state) % bases.len() as u32) as usize];
        let base_y = bases[(lcg(&mut state) % bases.len() as u32) as usize];
        let index = lcg(&mut state) % 500_000;
        let phase = lcg(&mut state) % 500_000;
        queries.push(TaauJitterOffsetQuery::new(base_x, base_y, index, phase));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn grounding_offset_matches_sequence() {
    // Prove the in-host offset oracle is faithful to the public golden: for
    // bases (2, 3) the `radical_inverse(base, phase + 1) - 0.5` closed form must
    // equal `offset_for_phase(phase)` exactly, grounding the transitive
    // GPU == oracle == golden argument.
    let seq = JitterSequence::default();
    for phase in 0u32..256 {
        let [ox, oy] = seq.offset_for_phase(phase);
        let pi = phase.wrapping_add(1);
        let cx = radical_inverse(HALTON_BASE_X, pi) - 0.5;
        let cy = radical_inverse(HALTON_BASE_Y, pi) - 0.5;
        assert!(
            close(cx, ox),
            "offset x mismatch at phase {phase}: {cx} vs {ox}"
        );
        assert!(
            close(cy, oy),
            "offset y mismatch at phase {phase}: {cy} vs {oy}"
        );
    }
}
