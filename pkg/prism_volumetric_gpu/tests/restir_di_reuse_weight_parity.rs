//! Real-device parity for the `ReSTIR` DI reuse-weight per-source twin:
//! [`GpuRestirDiReuseWeight`](prism_volumetric_gpu::restir_di_reuse_weight::GpuRestirDiReuseWeight)
//! must reproduce the `CPU` golden `reuse_weight` in
//! [`prism_render_architecture::lighting::restir_di`] — the combined weight
//! contribution `w_i = p̂_dst(y) * W * M` of one source reservoir re-weighted for
//! a destination pixel — across hand-checked fixtures, a mixed batch, and a
//! randomized sweep compared source-for-source.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden `reuse_weight` is private, so the expected value is reconstructed
//! in-host from its exact closed form `target_pdf_at_dst * w * (m as f32)` (one
//! integer-to-`f32` widening and two multiplies, never an `f32` `==`). A
//! separate grounding test
//! (`grounding_oracle_matches_combine_biased_weight_sum`) proves that closed
//! form is faithful by driving the public
//! [`combine_biased`](prism_render_architecture::lighting::restir_di::combine_biased)
//! with a single source and a constant target closure: the accumulated
//! [`w_sum`](prism_render_architecture::particle::reservoir_sample::Reservoir)
//! after the fold equals exactly one `reuse_weight` evaluation, so a
//! `GPU == oracle` pass transitively establishes `GPU == golden`.
//!
//! # Parity criterion
//!
//! The weight threads through two multiplies, so a `GPU` multiply may land a few
//! units in the last place from the scalar reference; `weight` is asserted
//! within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The count factor is an exact
//! integer-to-`f32` widening, so it adds no error.
//!
//! # Conditioning
//!
//! The twin is a plain product with no branch, so there is no discrete tie to
//! straddle; fixtures simply span a wide range of magnitudes and signs of `W`
//! (finalized weights are non-negative in practice, but the product is tested
//! for small and large densities and counts) to exercise the multiply path.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_di`；无第三方引擎源码或衍生代码。

use prism_render_architecture::lighting::restir_di::{combine_biased, DiReservoir};
use prism_render_architecture::particle::reservoir_sample::{Reservoir, Rng};
use prism_volumetric_gpu::restir_di_reuse_weight::{
    GpuRestirDiReuseWeight, RestirDiReuseWeightQuery, RestirDiReuseWeightResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a weight. A `GPU` multiply may land a few units in
/// the last place from the scalar reference; `1e-4` admits that legal slack
/// while still failing a wrong port.
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

/// Reconstructs the golden `reuse_weight` closed form in-host: the faithful
/// oracle the `GPU` is pinned against. Uses only a widening and two multiplies,
/// never an `f32` `==`, and no transcendental method.
fn oracle(q: &RestirDiReuseWeightQuery) -> RestirDiReuseWeightResult {
    RestirDiReuseWeightResult {
        weight: q.target_pdf_at_dst * q.w * (q.m as f32),
    }
}

/// Pins one `GPU` reuse weight against the in-host oracle, within tolerance.
fn check_one(idx: usize, got: &RestirDiReuseWeightResult, want: &RestirDiReuseWeightResult) {
    assert!(
        close(got.weight, want.weight),
        "source {idx} weight: gpu {} vs cpu {}",
        got.weight,
        want.weight
    );
}

/// Dispatches `queries`, then pins every `GPU` result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuRestirDiReuseWeight, queries: &[RestirDiReuseWeightQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, g, &want);
    }
}

/// A 64-bit linear congruential generator producing a `u32` word per step. Uses
/// only integer arithmetic — no external math library, no transcendental method.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a `u32` word and maps it into `[0, 1)` as a 24-bit fixed-point value.
fn unit(state: &mut u64) -> f32 {
    let bits = lcg(state) >> 8;
    (bits as f32) / 16_777_216.0
}

/// A hand-written batch spanning small and large densities, weights and counts.
fn fixture_queries() -> Vec<RestirDiReuseWeightQuery> {
    vec![
        // Plain product: 2 * 3 * 4 = 24.
        RestirDiReuseWeightQuery::new(2.0, 3.0, 4),
        // Unit factors: 1 * 1 * 1 = 1.
        RestirDiReuseWeightQuery::new(1.0, 1.0, 1),
        // Zero count collapses the product to zero.
        RestirDiReuseWeightQuery::new(5.0, 2.0, 0),
        // Zero weight collapses the product to zero.
        RestirDiReuseWeightQuery::new(5.0, 0.0, 9),
        // Large magnitude.
        RestirDiReuseWeightQuery::new(1.0e3, 7.5, 32),
        // Small magnitude.
        RestirDiReuseWeightQuery::new(1.0e-3, 2.0e-2, 3),
        // Mixed scales.
        RestirDiReuseWeightQuery::new(0.125, 48.0, 7),
    ]
}

#[test]
fn empty_input_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirDiReuseWeight::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn plain_product_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirDiReuseWeight::new(&ctx);
    let q = RestirDiReuseWeightQuery::new(2.0, 3.0, 4);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(close(want.weight, 24.0), "fixture should be 2 * 3 * 4 = 24");
    check_one(0, &got[0], &want);
}

#[test]
fn zero_count_collapses_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirDiReuseWeight::new(&ctx);
    // m == 0 drives the product to zero regardless of the density and weight.
    let q = RestirDiReuseWeightQuery::new(5.0, 2.0, 0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(close(want.weight, 0.0), "zero count must yield weight 0");
    check_one(0, &got[0], &want);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirDiReuseWeight::new(&ctx);
    // The whole fixture batch dispatched together so the per-thread indexing and
    // the contiguous output slots are both exercised.
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirDiReuseWeight::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random sources: non-negative densities and
    // weights spanning a wide range, counts in [0, 64].
    for _ in 0..512 {
        let target_pdf_at_dst = unit(&mut state) * 10.0;
        let w = unit(&mut state) * 100.0;
        let m = lcg(&mut state) % 65;
        queries.push(RestirDiReuseWeightQuery::new(target_pdf_at_dst, w, m));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn grounding_oracle_matches_combine_biased_weight_sum() {
    // Prove the in-host oracle is faithful to the public golden: driving
    // `combine_biased` with a single non-empty source and a constant target
    // closure accumulates exactly one `reuse_weight` evaluation into the output
    // reservoir's `w_sum`. The constant closure makes the destination target
    // density `target_pdf_at_dst` independent of the held light, so the folded
    // weight is precisely `target_pdf_at_dst * w * (m as f32)` — the oracle.
    // This grounds the transitive GPU == oracle == golden argument.
    let mut state = 0x1234_5678_9abc_def0_u64;
    for _ in 0..64 {
        let target_pdf_at_dst = 0.25 + unit(&mut state) * 8.0;
        let w = 0.25 + unit(&mut state) * 8.0;
        let m = 1 + (lcg(&mut state) % 48);

        let source = DiReservoir {
            reservoir: Reservoir {
                sample: 7,
                w_sum: 123.0,
                m,
                w,
            },
            target_pdf: 0.0,
        };
        let mut rng = Rng::new(0x5151_2727);
        let out = combine_biased(&[source], |_light| target_pdf_at_dst, &mut rng);

        let q = RestirDiReuseWeightQuery::new(target_pdf_at_dst, w, m);
        let want = oracle(&q);
        assert!(
            close(out.reservoir.w_sum, want.weight),
            "grounding: combine_biased w_sum {} vs oracle {} (p={target_pdf_at_dst}, w={w}, m={m})",
            out.reservoir.w_sum,
            want.weight
        );
    }
}
