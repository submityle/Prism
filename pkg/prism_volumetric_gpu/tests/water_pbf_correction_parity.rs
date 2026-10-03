//! Real-device parity for the Position-Based Fluids correction twin:
//! [`GpuWaterPbfCorrection`](prism_volumetric_gpu::water_pbf_correction::GpuWaterPbfCorrection)
//! must reproduce the `CPU` golden
//! [`artificial_pressure`](prism_render_architecture::water::pbf::artificial_pressure)
//! and
//! [`position_correction`](prism_render_architecture::water::pbf::position_correction)
//! across the normal, out-of-support, and all three degenerate short-circuit
//! regimes plus a randomized batch compared value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The two golden functions are `pub`, so each `GPU` result is pinned directly
//! against the golden run on the same input. The `s_corr` term comes from
//! [`artificial_pressure`](prism_render_architecture::water::pbf::artificial_pressure)
//! and the three correction components from
//! [`position_correction`](prism_render_architecture::water::pbf::position_correction),
//! whose [`NeighborContribution`](prism_render_architecture::water::pbf::NeighborContribution)
//! inputs are rebuilt from the query's bounded neighbour list.
//!
//! # Parity criterion
//!
//! Every result is a continuous `f32` (a guarded `Poly6` ratio raised to a
//! small integer power, or a gradient-weighted sum scaled by a reciprocal
//! density), asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Fixtures stay away from the `Poly6` support cliff: `r_squared` is kept
//! clearly below `h^2` (except the deliberate out-of-support case, which both
//! sides agree is exactly `0`), so a last-bit difference cannot flip the kernel
//! across its discontinuity. The three short-circuit guards (`k <= EPS`,
//! `reference <= EPS`, `rest_density <= EPS`) are exercised with exact-zero
//! inputs on which both sides agree deterministically.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pbf`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::pbf::{
    artificial_pressure, position_correction, NeighborContribution, PbfParams,
};
use prism_render_architecture::water::Vec3;
use prism_volumetric_gpu::water_pbf_correction::{
    GpuWaterPbfCorrection, WaterPbfCorrectionQuery, WaterPbfCorrectionResult, WaterPbfNeighbor,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on each continuous component.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
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

/// Builds a [`PbfParams`] from a query's artificial-pressure operands, filling
/// the fields the golden `artificial_pressure` never reads with valid dummies.
fn params_for(q: &WaterPbfCorrectionQuery) -> PbfParams {
    PbfParams {
        rest_density: q.rest_density.max(1.0),
        particle_mass: 1.0,
        smoothing_radius: q.ap_h,
        relaxation_epsilon: 0.0,
        artificial_pressure_k: q.ap_k,
        artificial_pressure_n: q.ap_n,
        artificial_pressure_delta_q: q.ap_delta_q,
        solver_iterations: 1,
    }
}

/// Computes the golden result for one query, so the oracle lives beside the
/// device call and both read the same input.
fn expected(q: &WaterPbfCorrectionQuery) -> WaterPbfCorrectionResult {
    let s_corr = artificial_pressure(q.r_squared, params_for(q));
    let neighbors: Vec<NeighborContribution> = q
        .neighbors
        .iter()
        .map(|n| NeighborContribution {
            lambda_j: n.lambda_j,
            scorr: n.scorr,
            gradient: Vec3::new(n.grad_x, n.grad_y, n.grad_z),
        })
        .collect();
    let corr = position_correction(q.lambda_i, q.rest_density, &neighbors);
    WaterPbfCorrectionResult {
        s_corr,
        corr_x: corr.x,
        corr_y: corr.y,
        corr_z: corr.z,
    }
}

/// Pins one `GPU` result against the golden oracle within tolerance.
fn assert_result(idx: usize, got: &WaterPbfCorrectionResult, want: &WaterPbfCorrectionResult) {
    assert!(
        close(got.s_corr, want.s_corr),
        "result {idx} s_corr: gpu {} vs cpu {}",
        got.s_corr,
        want.s_corr
    );
    assert!(
        close(got.corr_x, want.corr_x),
        "result {idx} corr_x: gpu {} vs cpu {}",
        got.corr_x,
        want.corr_x
    );
    assert!(
        close(got.corr_y, want.corr_y),
        "result {idx} corr_y: gpu {} vs cpu {}",
        got.corr_y,
        want.corr_y
    );
    assert!(
        close(got.corr_z, want.corr_z),
        "result {idx} corr_z: gpu {} vs cpu {}",
        got.corr_z,
        want.corr_z
    );
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[WaterPbfCorrectionQuery]) {
    let gpu = GpuWaterPbfCorrection::new(ctx);
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        assert_result(idx, g, &expected(q));
    }
}

/// Builds one neighbour contribution.
fn nb(lambda_j: f32, scorr: f32, grad_x: f32, grad_y: f32, grad_z: f32) -> WaterPbfNeighbor {
    WaterPbfNeighbor {
        lambda_j,
        scorr,
        grad_x,
        grad_y,
        grad_z,
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

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_pbf_correction parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterPbfCorrection::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn artificial_pressure_regimes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // h = 1.0 so h^2 = 1.0; keep r_squared clear of that cliff.
    let queries = vec![
        // Normal in-support pair: r^2 = 0.09 well below h^2 = 1.0.
        WaterPbfCorrectionQuery {
            r_squared: 0.09,
            ap_k: 1.0,
            ap_h: 1.0,
            ap_delta_q: 0.2,
            ap_n: 4,
            lambda_i: 0.0,
            rest_density: 1000.0,
            neighbors: Vec::new(),
        },
        // Out of support: r^2 = 2.25 > h^2 = 1.0 so poly6 -> 0 -> s_corr = 0.
        WaterPbfCorrectionQuery {
            r_squared: 2.25,
            ap_k: 1.0,
            ap_h: 1.0,
            ap_delta_q: 0.2,
            ap_n: 4,
            lambda_i: 0.0,
            rest_density: 1000.0,
            neighbors: Vec::new(),
        },
        // k <= EPS branch: zero strength short-circuits to 0.
        WaterPbfCorrectionQuery {
            r_squared: 0.09,
            ap_k: 0.0,
            ap_h: 1.0,
            ap_delta_q: 0.2,
            ap_n: 4,
            lambda_i: 0.0,
            rest_density: 1000.0,
            neighbors: Vec::new(),
        },
        // reference <= EPS branch: delta_q = 1 so dq = h, poly6(h^2, h) = 0.
        WaterPbfCorrectionQuery {
            r_squared: 0.09,
            ap_k: 1.0,
            ap_h: 1.0,
            ap_delta_q: 1.0,
            ap_n: 4,
            lambda_i: 0.0,
            rest_density: 1000.0,
            neighbors: Vec::new(),
        },
        // Smaller exponent and a different radius, still in-support.
        WaterPbfCorrectionQuery {
            r_squared: 0.4,
            ap_k: 2.5,
            ap_h: 1.5,
            ap_delta_q: 0.15,
            ap_n: 1,
            lambda_i: 0.0,
            rest_density: 1000.0,
            neighbors: Vec::new(),
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn position_correction_regimes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // Several neighbours fold into a weighted gradient sum.
        WaterPbfCorrectionQuery {
            r_squared: 0.09,
            ap_k: 1.0,
            ap_h: 1.0,
            ap_delta_q: 0.2,
            ap_n: 4,
            lambda_i: -0.5,
            rest_density: 1000.0,
            neighbors: vec![
                nb(-0.3, 0.01, 0.2, -0.1, 0.4),
                nb(0.2, -0.02, -0.3, 0.5, 0.1),
                nb(-0.1, 0.0, 0.05, 0.05, -0.2),
            ],
        },
        // Zero neighbours yield a zero correction (empty sum).
        WaterPbfCorrectionQuery {
            r_squared: 0.09,
            ap_k: 1.0,
            ap_h: 1.0,
            ap_delta_q: 0.2,
            ap_n: 4,
            lambda_i: 0.7,
            rest_density: 1000.0,
            neighbors: Vec::new(),
        },
        // rest_density <= EPS branch: correction short-circuits to zero.
        WaterPbfCorrectionQuery {
            r_squared: 0.09,
            ap_k: 1.0,
            ap_h: 1.0,
            ap_delta_q: 0.2,
            ap_n: 4,
            lambda_i: -0.5,
            rest_density: 0.0,
            neighbors: vec![nb(-0.3, 0.01, 0.2, -0.1, 0.4)],
        },
        // Full eight-neighbour list exercises the fixed-capacity bound.
        WaterPbfCorrectionQuery {
            r_squared: 0.25,
            ap_k: 1.5,
            ap_h: 1.2,
            ap_delta_q: 0.2,
            ap_n: 2,
            lambda_i: 0.1,
            rest_density: 800.0,
            neighbors: vec![
                nb(0.1, 0.001, 0.1, 0.0, 0.0),
                nb(-0.1, -0.001, 0.0, 0.1, 0.0),
                nb(0.2, 0.002, 0.0, 0.0, 0.1),
                nb(-0.2, -0.002, -0.1, 0.0, 0.0),
                nb(0.05, 0.0, 0.0, -0.1, 0.0),
                nb(-0.05, 0.0, 0.0, 0.0, -0.1),
                nb(0.15, 0.003, 0.05, 0.05, 0.05),
                nb(-0.15, -0.003, -0.05, -0.05, -0.05),
            ],
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x51a7_3f2e_9d04_6c18_u64;

    let mut queries: Vec<WaterPbfCorrectionQuery> = Vec::new();
    while queries.len() < 256 {
        // Smoothing radius in [0.5, 2]; h^2 in [0.25, 4].
        let ap_h = ranged(&mut state, 0.5, 2.0);
        let h2 = ap_h * ap_h;
        // r_squared kept in [0, 0.6 * h^2] with a clear margin from the cliff.
        let r_squared = ranged(&mut state, 0.0, 0.6 * h2);
        let ap_k = ranged(&mut state, 0.5, 5.0);
        let ap_delta_q = ranged(&mut state, 0.1, 0.3);
        let ap_n = (lcg(&mut state) % 4) + 1;
        let lambda_i = ranged(&mut state, -1.0, 1.0);
        let rest_density = ranged(&mut state, 500.0, 1200.0);

        let neighbor_count = lcg(&mut state) % 9;
        let mut neighbors = Vec::new();
        for _ in 0..neighbor_count {
            neighbors.push(nb(
                ranged(&mut state, -1.0, 1.0),
                ranged(&mut state, -0.05, 0.05),
                ranged(&mut state, -0.5, 0.5),
                ranged(&mut state, -0.5, 0.5),
                ranged(&mut state, -0.5, 0.5),
            ));
        }

        queries.push(WaterPbfCorrectionQuery {
            r_squared,
            ap_k,
            ap_h,
            ap_delta_q,
            ap_n,
            lambda_i,
            rest_density,
            neighbors,
        });
    }
    run_and_check(&ctx, &queries);
}
