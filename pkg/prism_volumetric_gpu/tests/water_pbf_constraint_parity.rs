//! Real-device parity for the `PBF` density-solve twin:
//! [`GpuWaterPbfConstraint`](prism_volumetric_gpu::water_pbf_constraint::GpuWaterPbfConstraint)
//! must reproduce the stateless outputs of the `CPU` goldens
//! [`density_constraint`](prism_render_architecture::water::pbf::density_constraint)
//! and
//! [`constraint_lambda`](prism_render_architecture::water::pbf::constraint_lambda)
//! — the density constraint and the `XPBD` scaling factor — across the
//! compressed, rarefied, rest-density-guard, denominator-guard and
//! negative-input cases plus a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference functions are public, so they drive the oracle directly: each
//! query's density and rest density feed
//! [`density_constraint`](prism_render_architecture::water::pbf::density_constraint)
//! for the expected constraint, and the gradient sum, squared-gradient sum and
//! epsilon additionally feed
//! [`constraint_lambda`](prism_render_architecture::water::pbf::constraint_lambda)
//! for the expected scaling factor. A passing `GPU == oracle` run is direct
//! evidence the kernel computes the same scheduling.
//!
//! # Parity criterion
//!
//! Both sides evaluate the identical guard, density ratio and gradient
//! denominator, so both outputs agree to within floating-point tolerance (`abs
//! <= 1e-4` or `rel <= 1e-3`, with a `REL_FLOOR` of `1e-6`). The two discrete
//! branch decisions — the rest-density guard and the denominator guard — are
//! kept clear of their `EPS` crossings by the fixtures and by reject-sampling
//! the random sweep, so a last-place rounding difference cannot flip a branch.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pbf`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::pbf::{constraint_lambda, density_constraint};
use prism_render_architecture::water::Vec3;
use prism_volumetric_gpu::water_pbf_constraint::{
    GpuWaterPbfConstraint, WaterPbfConstraintQuery, WaterPbfConstraintResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute closeness floor for the parity comparison.
const ABS: f32 = 1e-4;
/// Relative closeness bound for the parity comparison.
const REL: f32 = 1e-3;
/// Smallest denominator used in the relative comparison, guarding `0 == 0`.
const REL_FLOOR: f32 = 1e-6;

/// Absolute-or-relative closeness: `true` when `a` and `b` agree to within the
/// shared tolerance. Values that both land near zero pass through the absolute
/// bound; larger values use the relative bound against the larger magnitude.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS || rel <= REL
}

/// Computes the reference response for one query by calling the goldens
/// directly.
fn oracle(q: &WaterPbfConstraintQuery) -> WaterPbfConstraintResult {
    let constraint = density_constraint(q.density, q.rest_density);
    let grad_sum = Vec3::new(q.grad_sum_x, q.grad_sum_y, q.grad_sum_z);
    let lambda = constraint_lambda(
        q.density,
        q.rest_density,
        grad_sum,
        q.grad_sq_sum,
        q.epsilon,
    );
    WaterPbfConstraintResult { constraint, lambda }
}

/// Pins one `GPU` result against the oracle: both the constraint and the
/// scaling factor within tolerance.
fn check_result(idx: usize, got: &WaterPbfConstraintResult, want: &WaterPbfConstraintResult) {
    assert!(
        close(got.constraint, want.constraint),
        "query {idx} constraint: gpu {} vs cpu {}",
        got.constraint,
        want.constraint
    );
    assert!(
        close(got.lambda, want.lambda),
        "query {idx} lambda: gpu {} vs cpu {}",
        got.lambda,
        want.lambda
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterPbfConstraint, queries: &[WaterPbfConstraintQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_result(idx, result, &want);
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Maps a raw `u32` into `[lo, hi)` as an `f32`, using only integer and
/// floating-point arithmetic (no transcendental), for the random sweep.
fn uniform(raw: u32, lo: f32, hi: f32) -> f32 {
    let unit = (raw as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_pbf_constraint parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterPbfConstraint::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn compressed_particle_pushes_apart() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfConstraint::new(&ctx);
    // Denser than rest: a positive constraint and (with a healthy gradient
    // denominator) a negative lambda that spreads neighbours apart.
    let queries = [WaterPbfConstraintQuery::new(
        1200.0, 1000.0, 2.0, -1.0, 0.5, 3.0, 0.1,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        got[0].constraint > 0.0,
        "a compressed particle has a positive constraint"
    );
    assert!(
        got[0].lambda < 0.0,
        "a compressed particle's lambda pushes neighbours apart"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn rarefied_particle_pulls_together() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfConstraint::new(&ctx);
    // Sparser than rest: a negative constraint and a positive lambda.
    let queries = [WaterPbfConstraintQuery::new(
        800.0, 1000.0, 1.5, 0.5, -2.0, 4.0, 0.2,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        got[0].constraint < 0.0,
        "a rarefied particle has a negative constraint"
    );
    assert!(
        got[0].lambda > 0.0,
        "a rarefied particle's lambda pulls neighbours together"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn rest_density_zero_returns_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfConstraint::new(&ctx);
    // A non-positive rest density is the degenerate no-constraint case: both the
    // constraint and the scaling factor are zero regardless of the gradients. A
    // tiny positive value below EPS exercises the same guard.
    let queries = [
        WaterPbfConstraintQuery::new(1200.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.1),
        WaterPbfConstraintQuery::new(1200.0, 1e-7, 2.0, 1.0, 1.0, 3.0, 0.1),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        close(got[0].constraint, 0.0),
        "zero rest density: constraint 0"
    );
    assert!(close(got[0].lambda, 0.0), "zero rest density: lambda 0");
    assert!(
        close(got[1].constraint, 0.0),
        "sub-EPS rest density: constraint 0"
    );
    assert!(close(got[1].lambda, 0.0), "sub-EPS rest density: lambda 0");
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_denominator_zeroes_lambda() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfConstraint::new(&ctx);
    // Zero gradients and zero epsilon collapse the denominator to zero, so lambda
    // is zero even though the constraint is non-zero.
    let queries = [WaterPbfConstraintQuery::new(
        1300.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        got[0].constraint > 0.0,
        "the constraint is still formed when the denominator collapses"
    );
    assert!(
        close(got[0].lambda, 0.0),
        "a zero denominator relaxes lambda to zero"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn at_rest_density_has_zero_constraint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfConstraint::new(&ctx);
    // Exactly at rest density: the constraint is zero, so lambda is zero too even
    // with a healthy denominator.
    let queries = [WaterPbfConstraintQuery::new(
        1000.0, 1000.0, 1.0, 1.0, 1.0, 2.0, 0.3,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        close(got[0].constraint, 0.0),
        "at rest density the constraint is zero"
    );
    assert!(
        close(got[0].lambda, 0.0),
        "a zero constraint gives a zero lambda"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfConstraint::new(&ctx);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of queries spanning compressed and rarefied
    // particles with varied gradients. The rest density stays well above EPS and
    // the denominator is reject-sampled to stay comfortably above EPS, so both
    // guarded branches are unambiguous and cannot flip on a last-place rounding
    // difference.
    while queries.len() < 300 {
        let density = uniform(lcg(&mut state), 0.0, 2000.0);
        let rest_density = uniform(lcg(&mut state), 500.0, 1500.0);
        let grad_sum_x = uniform(lcg(&mut state), -5.0, 5.0);
        let grad_sum_y = uniform(lcg(&mut state), -5.0, 5.0);
        let grad_sum_z = uniform(lcg(&mut state), -5.0, 5.0);
        let grad_sq_sum = uniform(lcg(&mut state), 0.0, 10.0);
        let epsilon = uniform(lcg(&mut state), 0.0, 1.0);
        // Reject when the denominator sits within the no-flip margin of EPS.
        let len_sq = grad_sum_x * grad_sum_x + grad_sum_y * grad_sum_y + grad_sum_z * grad_sum_z;
        let inv_rho2 = 1.0 / (rest_density * rest_density);
        let denom = (len_sq + grad_sq_sum.max(0.0)) * inv_rho2 + epsilon.max(0.0);
        if denom < 1e-3 {
            continue;
        }
        queries.push(WaterPbfConstraintQuery::new(
            density,
            rest_density,
            grad_sum_x,
            grad_sum_y,
            grad_sum_z,
            grad_sq_sum,
            epsilon,
        ));
    }
    check(&ctx, &gpu, &queries);
}
