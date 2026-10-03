//! Real-device parity for the `ReSTIR` DI finalize per-reservoir twin:
//! [`GpuRestirDiFinalize`](prism_volumetric_gpu::restir_di_finalize::GpuRestirDiFinalize)
//! must reproduce the `CPU` golden
//! [`DiReservoir::finalize`](prism_render_architecture::lighting::restir_di::DiReservoir::finalize)
//! — the unbiased contribution weight `W = w_sum / (M * target_pdf)` of one
//! finished reservoir, guarded to zero when the target density vanishes or the
//! reservoir is empty — across hand-checked fixtures, the two guard branches, a
//! mixed batch, and a randomized sweep compared reservoir-for-reservoir.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`DiReservoir`](prism_render_architecture::lighting::restir_di::DiReservoir)
//! and its underlying
//! [`Reservoir`](prism_render_architecture::particle::reservoir_sample::Reservoir)
//! expose public fields, so the expected weight is produced by constructing a
//! reservoir from each query's `w_sum`, `target_pdf` and `m`, calling the public
//! [`finalize`](prism_render_architecture::lighting::restir_di::DiReservoir::finalize),
//! and reading back `W`. A `GPU == golden` pass is therefore direct evidence the
//! ported kernel computes the same unbiased weight the reference does.
//!
//! # Parity criterion
//!
//! The guard decision (clamp to zero or not) is a magnitude compare and an
//! integer compare, so it agrees exactly between host and device for fixtures
//! clear of the `1e-6` density knot. The surviving weight threads through one
//! divide, so a `GPU` divide may land a few units in the last place from the
//! scalar reference; `W` is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Every fixture keeps `target_pdf` either clearly above or clearly below the
//! `1e-6` guard knot (a rejection band around `1e-6` is skipped in the random
//! sweep), and `m` is discrete so both sides of the count guard are exact. The
//! surviving weights stay well within the `f32` slack, so `CPU` and `GPU` stay
//! on the same side of the guard decision.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_di`；无第三方引擎源码或衍生代码。

use prism_render_architecture::lighting::restir_di::DiReservoir;
use prism_render_architecture::particle::reservoir_sample::Reservoir;
use prism_volumetric_gpu::restir_di_finalize::{
    GpuRestirDiFinalize, RestirDiFinalizeQuery, RestirDiFinalizeResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a weight. A `GPU` divide may land a few units in the
/// last place from the scalar reference; `1e-4` admits that legal slack while
/// still failing a wrong port.
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

/// Reconstructs the golden weight in-host: constructs a reservoir from the query
/// and finalizes it through the public reference entry point, the oracle the
/// `GPU` is pinned against.
fn oracle(q: &RestirDiFinalizeQuery) -> RestirDiFinalizeResult {
    let mut r = DiReservoir {
        reservoir: Reservoir {
            sample: 0,
            w_sum: q.w_sum,
            m: q.m,
            w: 0.0,
        },
        target_pdf: q.target_pdf,
    };
    r.finalize();
    RestirDiFinalizeResult { w: r.reservoir.w }
}

/// Pins one `GPU` finalized weight against the in-host oracle, within tolerance.
fn check_one(idx: usize, got: &RestirDiFinalizeResult, want: &RestirDiFinalizeResult) {
    assert!(
        close(got.w, want.w),
        "reservoir {idx} W: gpu {} vs cpu {}",
        got.w,
        want.w
    );
}

/// Dispatches `queries`, then pins every `GPU` result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuRestirDiFinalize, queries: &[RestirDiFinalizeQuery]) {
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

/// Draws a positive target density well above the `1e-6` guard knot: in
/// `[0.5, 8.5)`, far from the degenerate band so the guard decision is stable.
fn density(state: &mut u64) -> f32 {
    0.5 + unit(state) * 8.0
}

/// A hand-written batch exercising both the finite-weight path and both guard
/// branches, each conditioned clear of the `1e-6` density knot.
fn fixture_queries() -> Vec<RestirDiFinalizeQuery> {
    vec![
        // Finite weight: W = 12 / (4 * 3) = 1.
        RestirDiFinalizeQuery::new(12.0, 3.0, 4),
        // Finite weight: W = 7.5 / (3 * 2.5) = 1.
        RestirDiFinalizeQuery::new(7.5, 2.5, 3),
        // Empty reservoir (m == 0): W = 0 regardless of the other fields.
        RestirDiFinalizeQuery::new(5.0, 2.0, 0),
        // Vanishing target density (<= 1e-6): W = 0.
        RestirDiFinalizeQuery::new(5.0, 1.0e-9, 7),
        // Large magnitude weight.
        RestirDiFinalizeQuery::new(1.0e6, 4.0, 2),
        // Small magnitude weight.
        RestirDiFinalizeQuery::new(1.0e-3, 2.0, 5),
        // Single candidate: W = w_sum / target_pdf.
        RestirDiFinalizeQuery::new(3.25, 1.3, 1),
    ]
}

#[test]
fn empty_input_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirDiFinalize::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn finite_weight_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirDiFinalize::new(&ctx);
    let q = RestirDiFinalizeQuery::new(12.0, 3.0, 4);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(close(want.w, 1.0), "fixture should finalize to W = 1");
    check_one(0, &got[0], &want);
}

#[test]
fn empty_reservoir_forces_zero_weight() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirDiFinalize::new(&ctx);
    // m == 0 clamps W to zero even with a positive weight sum and density.
    let q = RestirDiFinalizeQuery::new(5.0, 2.0, 0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(close(want.w, 0.0), "empty reservoir must finalize to W = 0");
    check_one(0, &got[0], &want);
}

#[test]
fn vanishing_density_forces_zero_weight() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirDiFinalize::new(&ctx);
    // target_pdf well below the 1e-6 guard knot clamps W to zero.
    let q = RestirDiFinalizeQuery::new(5.0, 1.0e-9, 7);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        close(want.w, 0.0),
        "vanishing density must finalize to W = 0"
    );
    check_one(0, &got[0], &want);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirDiFinalize::new(&ctx);
    // Both guard branches and the finite path dispatched together so the
    // per-thread indexing and the contiguous output slots are both exercised.
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirDiFinalize::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random reservoirs: positive weight sums,
    // densities far from the guard knot, counts in [1, 64]. A random subset is
    // forced empty (m == 0) so the guard branch is exercised under randomization
    // without ever landing inside the degenerate density band.
    for _ in 0..512 {
        let w_sum = unit(&mut state) * 1000.0;
        let target_pdf = density(&mut state);
        let m = if lcg(&mut state).is_multiple_of(8) {
            0
        } else {
            1 + (lcg(&mut state) % 64)
        };
        queries.push(RestirDiFinalizeQuery::new(w_sum, target_pdf, m));
    }
    check(&ctx, &gpu, &queries);
}
