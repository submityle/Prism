//! Real-device parity for the `ReSTIR` DI temporal-reprojection twin:
//! [`GpuRestirTemporalReproject`](prism_volumetric_gpu::restir_temporal_reproject::GpuRestirTemporalReproject)
//! must reproduce the exact admissibility decision and `M`-capped reservoir of
//! the `CPU` golden
//! [`reproject_history`](prism_render_architecture::lighting::restir_temporal::reproject_history)
//! across empty history, invalid (non-positive, infinite, `NaN`) surfaces,
//! depth and normal rejections, accepted reprojections with and without an
//! active `M`-cap, boundary-adjacent cases kept clear of a tie, and a
//! randomized sweep compared sample-for-sample.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Each `GPU` result is pinned directly against the public golden
//! [`reproject_history`](prism_render_architecture::lighting::restir_temporal::reproject_history),
//! fed the same history reservoir, surfaces and tolerances, so a `GPU == golden`
//! pass is direct evidence the ported kernel gates and caps history the same way
//! the reference does.
//!
//! # Parity criterion
//!
//! The `hit` flag and the integer count `m` are asserted with an exact `==`.
//! The passthrough weights (`w_sum`, `w`, `target_pdf`) are copies of the input
//! (`cap_history` touches only the count) and are asserted with the shared
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` tolerance.
//!
//! # Conditioning
//!
//! A `GPU` multiply-add can land a few units in the last place from the scalar
//! reference, so a fixture whose `depth_diff` sits right on
//! `depth_rel_tolerance * current.view_depth`, or whose normal `dot` sits right
//! on `normal_cos_tolerance`, could flip the gate. Every random fixture is held
//! a clear margin away from both thresholds (rejection sampling), far beyond the
//! `f32` slack, so `CPU` and `GPU` stay on the same side of every comparison.
//! Random normals are built from integer-derived components and normalized with
//! `sqrt`; no `f32` transcendental method and no `f32` `==` appears.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_temporal`；无第三方引擎源码或衍生代码。

use prism_render_architecture::lighting::restir_di::DiReservoir;
use prism_render_architecture::lighting::restir_temporal::{
    reproject_history, GeomReservoir, SurfaceGeometry, TemporalParams,
};
use prism_render_architecture::particle::reservoir_sample::Reservoir;
use prism_volumetric_gpu::restir_temporal_reproject::{
    GpuRestirTemporalReproject, RestirTemporalReprojectQuery, RestirTemporalReprojectResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous passthrough weights.
const EPS: f32 = 1e-4;
/// Relative tolerance for the continuous passthrough weights.
const REL: f32 = 1e-3;
/// Relative-tolerance denominator floor so near-zero values use the absolute arm.
const REL_FLOOR: f32 = 1e-6;

/// Default relative view-depth tolerance, mirroring the reference default.
const DEPTH_REL_TOL: f32 = 0.1;
/// Default minimum normal-agreement cosine, mirroring the reference default.
const NORMAL_COS_TOL: f32 = 0.906;

/// Shared closeness test: within `EPS` absolutely, or within `REL` relatively.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL
}

/// The authoritative result for one query: feeds the exact same history
/// reservoir, surfaces and tolerances to the public golden
/// [`reproject_history`](prism_render_architecture::lighting::restir_temporal::reproject_history),
/// then flattens its `Option<DiReservoir>` into the twin's result shape.
fn golden(q: &RestirTemporalReprojectQuery) -> RestirTemporalReprojectResult {
    let reservoir = DiReservoir {
        reservoir: Reservoir {
            sample: q.h_sample,
            w_sum: q.h_w_sum,
            m: q.h_m,
            w: q.h_w,
        },
        target_pdf: q.h_target_pdf,
    };
    let history = GeomReservoir::new(reservoir, SurfaceGeometry::new(q.h_depth, q.h_normal));
    let current = SurfaceGeometry::new(q.c_depth, q.c_normal);
    let params = TemporalParams {
        max_history_m: q.max_history_m,
        depth_rel_tolerance: q.depth_rel_tolerance,
        normal_cos_tolerance: q.normal_cos_tolerance,
    };
    match reproject_history(history, current, params) {
        Some(out) => RestirTemporalReprojectResult {
            hit: true,
            sample: out.reservoir.sample,
            m: out.reservoir.m,
            w_sum: out.reservoir.w_sum,
            w: out.reservoir.w,
            target_pdf: out.target_pdf,
        },
        None => RestirTemporalReprojectResult {
            hit: false,
            sample: 0,
            m: 0,
            w_sum: 0.0,
            w: 0.0,
            target_pdf: 0.0,
        },
    }
}

/// Dispatches every query and pins each `GPU` result against the golden: exact
/// `hit` and `m`, tolerant passthrough weights.
fn check(
    ctx: &GpuContext,
    gpu: &GpuRestirTemporalReproject,
    queries: &[RestirTemporalReprojectQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = golden(q);
        assert_eq!(result.hit, want.hit, "sample {idx}: hit flag must match");
        assert_eq!(
            result.sample, want.sample,
            "sample {idx}: held light index must match"
        );
        assert_eq!(result.m, want.m, "sample {idx}: capped count must match");
        assert!(
            close(result.w_sum, want.w_sum),
            "sample {idx}: w_sum gpu {} vs cpu {}",
            result.w_sum,
            want.w_sum
        );
        assert!(
            close(result.w, want.w),
            "sample {idx}: w gpu {} vs cpu {}",
            result.w,
            want.w
        );
        assert!(
            close(result.target_pdf, want.target_pdf),
            "sample {idx}: target_pdf gpu {} vs cpu {}",
            result.target_pdf,
            want.target_pdf
        );
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

/// Draws a unit-interval `f32` in `[0.0, 1.0)` at 24-bit resolution.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / 16_777_216.0
}

/// Draws a signed component in `[-1.0, 1.0]` at milli resolution from `state`.
fn signed_component(state: &mut u64) -> f32 {
    (lcg(state) % 2001) as f32 / 1000.0 - 1.0
}

/// Euclidean length (uses only `sqrt`, no transcendental method).
fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Normalizes a 3-vector (caller guarantees a non-degenerate length).
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = length(v);
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Draws a well-conditioned random unit normal, rejecting near-zero vectors so
/// the normalization stays stable.
fn rand_unit(state: &mut u64) -> [f32; 3] {
    loop {
        let v = [
            signed_component(state),
            signed_component(state),
            signed_component(state),
        ];
        if length(v) > 0.25 {
            return normalize(v);
        }
    }
}

/// Dot product of two 3-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Builds a conditioned random sweep: non-empty histories, valid surfaces, and
/// every fixture held a clear margin from the depth and cosine thresholds so the
/// gate cannot flip between `CPU` and `GPU`. Covers both the capped and uncapped
/// `M` paths and both the accepted and rejected branches.
fn conditioned_sweep() -> Vec<RestirTemporalReprojectQuery> {
    let mut state: u64 = 0x5eed_1234_abcd_0001;
    let mut out = Vec::new();
    let mut attempts = 0;
    while out.len() < 192 && attempts < 20_000 {
        attempts += 1;
        // Current surface: valid positive depth.
        let c_depth = 4.0 + unit(&mut state) * 16.0;
        let bound = DEPTH_REL_TOL * c_depth;

        // History depth: either comfortably inside or comfortably outside the
        // relative-depth band (clear 0.1*bound margin either way).
        let want_depth_in = (lcg(&mut state) & 1) == 0;
        let frac = 0.1 + unit(&mut state) * 0.6;
        let delta = if want_depth_in {
            frac * bound
        } else {
            bound * (1.4 + unit(&mut state))
        };
        let signed_delta = if (lcg(&mut state) & 1) == 0 {
            delta
        } else {
            -delta
        };
        let h_depth = c_depth + signed_delta;
        if h_depth <= 0.5 {
            continue;
        }

        let c_normal = rand_unit(&mut state);
        let h_normal = rand_unit(&mut state);
        let dotn = dot3(h_normal, c_normal);
        // Reject fixtures whose normal dot sits within 0.03 of the cosine
        // threshold so a last-place difference cannot flip the normal gate.
        if (dotn - NORMAL_COS_TOL).abs() < 0.03 {
            continue;
        }
        // Reject depth fixtures within 0.05*bound of the band edge.
        let depth_diff = (h_depth - c_depth).abs();
        if (depth_diff - bound).abs() < 0.05 * bound {
            continue;
        }

        let h_sample = lcg(&mut state) % 4096;
        let h_m = 1 + lcg(&mut state) % 2000;
        let max_history_m = 1 + lcg(&mut state) % 1000;
        let h_w_sum = unit(&mut state) * 32.0;
        let h_w = unit(&mut state) * 8.0;
        let h_target_pdf = unit(&mut state) * 4.0;

        out.push(RestirTemporalReprojectQuery::new(
            h_sample,
            h_w_sum,
            h_m,
            h_w,
            h_target_pdf,
            h_depth,
            h_normal,
            c_depth,
            c_normal,
            max_history_m,
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ));
    }
    out
}

#[test]
fn empty_input_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirTemporalReproject::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty input must return an empty result");
}

#[test]
fn empty_history_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirTemporalReproject::new(&ctx);
    // A zero-M history is empty: rejected regardless of a perfectly matching
    // surface.
    let q = RestirTemporalReprojectQuery::new(
        7,
        3.0,
        0,
        1.5,
        2.0,
        10.0,
        [0.0, 0.0, 1.0],
        10.0,
        [0.0, 0.0, 1.0],
        500,
        DEPTH_REL_TOL,
        NORMAL_COS_TOL,
    );
    assert!(!golden(&q).hit, "empty history must be rejected");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn accepted_without_cap_passes_through() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirTemporalReproject::new(&ctx);
    // Matching surface, M below the cap: the reservoir passes through unchanged.
    let q = RestirTemporalReprojectQuery::new(
        12,
        9.5,
        30,
        2.25,
        4.0,
        10.0,
        [0.0, 0.0, 1.0],
        10.3,
        [0.0, 0.0, 1.0],
        500,
        DEPTH_REL_TOL,
        NORMAL_COS_TOL,
    );
    let want = golden(&q);
    assert!(want.hit, "fixture must be accepted");
    assert_eq!(want.m, 30, "M below the cap stays unchanged");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn accepted_with_cap_clamps_count() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirTemporalReproject::new(&ctx);
    // Matching surface, M above the cap: the count is clamped, weights intact.
    let q = RestirTemporalReprojectQuery::new(
        3,
        18.0,
        900,
        5.5,
        7.0,
        20.0,
        [0.0, 0.0, 1.0],
        19.0,
        [0.0, 0.0, 1.0],
        500,
        DEPTH_REL_TOL,
        NORMAL_COS_TOL,
    );
    let want = golden(&q);
    assert!(want.hit, "fixture must be accepted");
    assert_eq!(want.m, 500, "M above the cap must be clamped");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn depth_disocclusion_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirTemporalReproject::new(&ctx);
    // depth_diff 2.5 vs bound 1.0 (0.1 * 10): a disocclusion, rejected.
    let q = RestirTemporalReprojectQuery::new(
        5,
        4.0,
        40,
        1.0,
        2.0,
        12.5,
        [0.0, 0.0, 1.0],
        10.0,
        [0.0, 0.0, 1.0],
        500,
        DEPTH_REL_TOL,
        NORMAL_COS_TOL,
    );
    assert!(!golden(&q).hit, "depth mismatch must be rejected");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn normal_crease_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirTemporalReproject::new(&ctx);
    // Identical depths, but normals meet at a dot of 0.6 < 0.906: rejected.
    let q = RestirTemporalReprojectQuery::new(
        5,
        4.0,
        40,
        1.0,
        2.0,
        10.0,
        [0.0, 0.8, 0.6],
        10.0,
        [0.0, 0.0, 1.0],
        500,
        DEPTH_REL_TOL,
        NORMAL_COS_TOL,
    );
    assert!(!golden(&q).hit, "normal crease must be rejected");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn invalid_surfaces_are_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirTemporalReproject::new(&ctx);
    // Non-positive / non-finite depths on either surface are never admissible.
    // The infinity and NaN cases exercise the ordered-compare `is_valid` replica.
    let bad = [0.0_f32, -3.0, f32::INFINITY, f32::NAN];
    let mut queries = Vec::new();
    for &d in &bad {
        // Bad current depth.
        queries.push(RestirTemporalReprojectQuery::new(
            9,
            2.0,
            25,
            1.0,
            1.5,
            10.0,
            [0.0, 0.0, 1.0],
            d,
            [0.0, 0.0, 1.0],
            500,
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ));
        // Bad history depth.
        queries.push(RestirTemporalReprojectQuery::new(
            9,
            2.0,
            25,
            1.0,
            1.5,
            d,
            [0.0, 0.0, 1.0],
            10.0,
            [0.0, 0.0, 1.0],
            500,
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ));
    }
    for q in &queries {
        assert!(!golden(q).hit, "invalid surface must be rejected");
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn boundary_adjacent_cases_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirTemporalReproject::new(&ctx);
    // Four cases, each a clear margin from its threshold: depth just inside,
    // depth just outside, normal just above the cosine, normal just below it.
    let just_above_cos = normalize([0.0, 0.3, 1.0]);
    let just_below_cos = normalize([0.0, 0.55, 1.0]);
    let queries = vec![
        // depth_diff 0.5 vs bound 1.0 (accepted), normals equal.
        RestirTemporalReprojectQuery::new(
            6,
            5.0,
            50,
            1.0,
            2.0,
            10.5,
            [0.0, 0.0, 1.0],
            10.0,
            [0.0, 0.0, 1.0],
            500,
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ),
        // depth_diff 1.5 vs bound 1.0 (rejected), normals equal.
        RestirTemporalReprojectQuery::new(
            6,
            5.0,
            50,
            1.0,
            2.0,
            11.5,
            [0.0, 0.0, 1.0],
            10.0,
            [0.0, 0.0, 1.0],
            500,
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ),
        // dot ~0.958 >= 0.906 (accepted), depths equal.
        RestirTemporalReprojectQuery::new(
            6,
            5.0,
            50,
            1.0,
            2.0,
            10.0,
            just_above_cos,
            10.0,
            [0.0, 0.0, 1.0],
            500,
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ),
        // dot ~0.876 < 0.906 (rejected), depths equal.
        RestirTemporalReprojectQuery::new(
            6,
            5.0,
            50,
            1.0,
            2.0,
            10.0,
            just_below_cos,
            10.0,
            [0.0, 0.0, 1.0],
            500,
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ),
    ];
    let expected = [true, false, true, false];
    for (q, &want) in queries.iter().zip(expected.iter()) {
        assert_eq!(
            golden(q).hit,
            want,
            "boundary fixture must match the golden"
        );
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirTemporalReproject::new(&ctx);
    let queries = conditioned_sweep();
    assert!(
        queries.len() >= 128,
        "the conditioned sweep must yield a sizable batch, got {}",
        queries.len()
    );
    // Both branches must be exercised so the sweep is meaningful.
    let hits = queries.iter().filter(|q| golden(q).hit).count();
    assert!(hits > 0, "sweep must contain accepted samples");
    assert!(hits < queries.len(), "sweep must contain rejected samples");
    // And both cap paths must appear so the clamp is exercised.
    let capped = queries
        .iter()
        .filter(|q| golden(q).hit && q.h_m > q.max_history_m)
        .count();
    assert!(capped > 0, "sweep must contain a capped acceptance");
    check(&ctx, &gpu, &queries);
}
