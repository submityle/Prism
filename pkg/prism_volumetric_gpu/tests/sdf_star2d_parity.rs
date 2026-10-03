//! Real-device parity for the star / parallelogram signed-distance twin:
//! [`GpuSdfStar2d`](prism_volumetric_gpu::sdf_star2d::GpuSdfStar2d) must
//! reproduce the three closed-form fields of the host-side independent
//! reimplementations
//! [`hexagram_2d_sdf`](prism_volumetric_gpu::sdf_star2d::hexagram_2d_sdf),
//! [`pentagram_2d_sdf`](prism_volumetric_gpu::sdf_star2d::pentagram_2d_sdf) and
//! [`parallelogram_sdf`](prism_volumetric_gpu::sdf_star2d::parallelogram_sdf)
//! across interior, surface and exterior points, the sharp star-tip and
//! parallelogram-fold conditioning hot-spots (kept clear of the exact feature),
//! and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected distances come from the module's own host-side independent
//! reimplementations; the twin never imports the golden crate, so both the
//! kernel and the oracle are faithful, independent ports of the same closed
//! form. A `GPU` parity pass is therefore direct evidence the ported kernel
//! evaluates the same distance.
//!
//! # Parity criterion
//!
//! Each distance is a *continuous* quantity threaded through `sqrt` and
//! products, so every assertion compares with an absolute-or-relative tolerance
//! (`abs <= 1e-4 || rel <= 1e-3`, relative floor `1e-6`). Each field closes with
//! a positive-at-zero sign branch, and the parallelogram also pivots on the
//! lower-half fold and a signed area; the randomized sweep rejects points
//! within a small margin of any sign-flip boundary so the `CPU` and `GPU` agree
//! on the interior sign away from the exact edge.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_star2d::{
    hexagram_2d_sdf, parallelogram_sdf, pentagram_2d_sdf, GpuSdfStar2d, SdfStar2dQuery,
    SdfStar2dResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous distance comparison.
const EPS: f32 = 1e-4;
/// Relative tolerance for the continuous distance comparison.
const REL: f32 = 1e-3;
/// Relative-tolerance floor, so near-zero magnitudes do not inflate the ratio.
const REL_FLOOR: f32 = 1e-6;

/// Hexagram fold constant `x`: baked `-cos 60deg`.
const HG_KX: f32 = -0.5;
/// Hexagram fold constant `y`: baked `cos 30deg`.
const HG_KY: f32 = 0.866_025_4;
/// Hexagram inner edge-clamp constant: baked `tan 30deg`.
const HG_KZ: f32 = 0.577_350_26;
/// Hexagram outer edge-clamp constant: baked `sqrt 3`.
const HG_KW: f32 = 1.732_050_8;
/// Pentagram fold constant `x`: baked `cos 36deg`.
const PENTA_K1X: f32 = 0.809_017;
/// Pentagram fold constant `y`: baked `-sin 36deg`.
const PENTA_K1Y: f32 = -0.587_785_25;
/// Pentagram inner/outer radius ratio: baked `(3 - sqrt 5) / 2`.
const PENTA_INNER: f32 = 0.381_966_02;
/// Rejection margin around each sign-flip boundary for the randomized sweep.
const SIGN_MARGIN: f32 = 0.02;

/// Returns whether `a` and `b` agree within the absolute-or-relative tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL
}

/// Computes the three expected distances from the host-side reimplementations.
fn oracle(q: &SdfStar2dQuery) -> [f32; 3] {
    [
        hexagram_2d_sdf([q.px, q.py], q.hex_r),
        pentagram_2d_sdf([q.px, q.py], q.penta_radius),
        parallelogram_sdf(
            [q.px, q.py],
            q.para_half_width,
            q.para_half_height,
            q.para_skew,
        ),
    ]
}

/// Dispatches every query and pins each `GPU` distance against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfStar2d, queries: &[SdfStar2dQuery]) {
    let got: Vec<SdfStar2dResult> = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert!(
            close(result.dist_hexagram, want[0])
                && close(result.dist_pentagram, want[1])
                && close(result.dist_parallelogram, want[2]),
            "query {idx}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
            result.dist_hexagram,
            result.dist_pentagram,
            result.dist_parallelogram,
            want[0],
            want[1],
            want[2],
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

/// Draws an `f32` in `[lo, hi]` at milli resolution from `state`, using only
/// integer arithmetic so no transcendental method appears.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = ((hi - lo) * 1000.0) as u32;
    let step = lcg(state) % (span + 1);
    lo + step as f32 / 1000.0
}

/// Replays the hexagram fold and returns the `sign(py)` argument the twin feeds
/// to its positive-at-zero branch.
fn hexagram_sign_arg(px0: f32, py0: f32, r: f32) -> f32 {
    let mut px = px0.abs();
    let mut py = py0.abs();
    let f1 = 2.0 * (HG_KX * px + HG_KY * py).min(0.0);
    px -= f1 * HG_KX;
    py -= f1 * HG_KY;
    let f2 = 2.0 * (HG_KY * px + HG_KX * py).min(0.0);
    px -= f2 * HG_KY;
    py -= f2 * HG_KX;
    px -= px.clamp(HG_KZ * r, HG_KW * r);
    py -= r;
    let _ = px;
    py
}

/// Replays the pentagram fold and returns the `sign(py*bax - px*bay)` argument.
fn pentagram_sign_arg(px0: f32, py0: f32, radius: f32) -> f32 {
    let mut px = px0.abs();
    let mut py = py0;
    let d1 = (PENTA_K1X * px + PENTA_K1Y * py).max(0.0);
    px -= 2.0 * d1 * PENTA_K1X;
    py -= 2.0 * d1 * PENTA_K1Y;
    let d2 = ((-PENTA_K1X) * px + PENTA_K1Y * py).max(0.0);
    px -= 2.0 * d2 * (-PENTA_K1X);
    py -= 2.0 * d2 * PENTA_K1Y;
    px = px.abs();
    py -= radius;
    let bax = PENTA_INNER * (-PENTA_K1Y);
    let bay = PENTA_INNER * PENTA_K1X - 1.0;
    py * bax - px * bay
}

/// Replays the parallelogram folds and returns the three discrete pivots: the
/// lower-half fold argument `point.y`, the signed area `s`, and the final
/// `sign(-d1)` argument.
fn parallelogram_sign_args(
    px0: f32,
    py0: f32,
    half_width: f32,
    half_height: f32,
    skew: f32,
) -> (f32, f32, f32) {
    let ex = skew;
    let ey = half_height;
    let fold = py0;
    let (px, py) = if py0 < 0.0 { (-px0, -py0) } else { (px0, py0) };
    // Only the sign-carrying `d1` channel matters here; the slant distance that
    // also feeds `d0` does not affect the final `sign(-d1)` branch.
    let wy = py - ey;
    let s = px * ey - py * ex;
    let d1 = (-wy).min(half_width * half_height - s.abs());
    (fold, s, -d1)
}

/// Rejects queries whose sign arguments sit within `SIGN_MARGIN` of a boundary,
/// where a `CPU` and `GPU` last-place difference could flip the discrete sign.
fn well_conditioned(q: &SdfStar2dQuery) -> bool {
    if hexagram_sign_arg(q.px, q.py, q.hex_r).abs() < SIGN_MARGIN {
        return false;
    }
    if pentagram_sign_arg(q.px, q.py, q.penta_radius).abs() < SIGN_MARGIN {
        return false;
    }
    let (fold, s, sign) = parallelogram_sign_args(
        q.px,
        q.py,
        q.para_half_width,
        q.para_half_height,
        q.para_skew,
    );
    fold.abs() >= SIGN_MARGIN && s.abs() >= SIGN_MARGIN && sign.abs() >= SIGN_MARGIN
}

/// The deterministic fixtures: interior, surface-adjacent and exterior points
/// for each shape, all kept clear of the exact tips and folds. Every query
/// carries well-conditioned, strictly positive shape parameters.
fn edge_fixtures() -> Vec<SdfStar2dQuery> {
    vec![
        // Interior near the centre for all three shapes.
        SdfStar2dQuery::new(0.05, 0.08, 1.0, 1.0, 1.0, 0.8, 0.3),
        // Far exterior for all three shapes.
        SdfStar2dQuery::new(3.0, 2.4, 1.0, 1.0, 1.0, 0.8, 0.3),
        // Off-axis interior, below the x-axis, clear of folds.
        SdfStar2dQuery::new(-0.30, -0.45, 1.1, 1.2, 1.1, 0.9, -0.25),
        // Above the top edge, exterior for every shape.
        SdfStar2dQuery::new(0.12, 1.9, 1.0, 1.0, 0.9, 0.7, 0.4),
        // Right exterior, clear of the parallelogram slant.
        SdfStar2dQuery::new(1.7, 0.20, 1.0, 1.0, 1.0, 0.8, 0.3),
        // Lower-left exterior for every shape.
        SdfStar2dQuery::new(-1.7, -1.4, 0.8, 0.9, 0.8, 0.7, -0.3),
        // Smaller shapes, mixed interior / exterior.
        SdfStar2dQuery::new(-0.20, 0.30, 0.6, 0.7, 0.6, 0.5, 0.2),
        // Off-axis interior for the larger scale.
        SdfStar2dQuery::new(0.45, -0.35, 1.3, 1.1, 1.2, 0.9, 0.35),
        // Near a hexagram wedge seam but offset off the exact fold line.
        SdfStar2dQuery::new(0.70, 0.45, 1.0, 1.0, 1.0, 0.8, 0.3),
        // Larger parallelogram with a strong skew, exterior point.
        SdfStar2dQuery::new(-1.1, 0.95, 1.0, 1.0, 1.3, 1.0, 0.6),
    ]
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        #[expect(
            clippy::print_stderr,
            reason = "a skipped device test should report why on hosts without a GPU"
        )]
        {
            eprintln!("skipping sdf_star2d parity: no wgpu adapter on this host");
        }
        return;
    };
    let gpu = GpuSdfStar2d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn hexagram_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfStar2d::new(&ctx);
    let q = SdfStar2dQuery::new(0.03, 0.05, 1.0, 1.0, 1.0, 0.8, 0.3);
    let want = oracle(&q);
    assert!(
        want[0] < 0.0,
        "a point inside the hexagram has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn hexagram_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfStar2d::new(&ctx);
    let q = SdfStar2dQuery::new(2.5, 0.10, 1.0, 1.0, 1.0, 0.8, 0.3);
    let want = oracle(&q);
    assert!(
        want[0] > 0.0,
        "a far point outside the hexagram is positive"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn pentagram_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfStar2d::new(&ctx);
    let q = SdfStar2dQuery::new(-0.08, -0.12, 1.0, 1.0, 1.0, 0.8, 0.3);
    let want = oracle(&q);
    assert!(
        want[1] < 0.0,
        "a point inside the pentagram has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn pentagram_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfStar2d::new(&ctx);
    let q = SdfStar2dQuery::new(0.10, 1.9, 1.0, 1.0, 1.0, 0.8, 0.3);
    let want = oracle(&q);
    assert!(
        want[1] > 0.0,
        "a point above the pentagram top tip is positive"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn parallelogram_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfStar2d::new(&ctx);
    let q = SdfStar2dQuery::new(0.10, 0.12, 1.0, 1.0, 1.0, 0.8, 0.3);
    let want = oracle(&q);
    assert!(
        want[2] < 0.0,
        "a point inside the parallelogram has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn parallelogram_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfStar2d::new(&ctx);
    let q = SdfStar2dQuery::new(2.4, 0.10, 1.0, 1.0, 1.0, 0.8, 0.3);
    let want = oracle(&q);
    assert!(
        want[2] > 0.0,
        "a far point outside the parallelogram is positive"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn edge_fixtures_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfStar2d::new(&ctx);
    check(&ctx, &gpu, &edge_fixtures());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfStar2d::new(&ctx);
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let q = SdfStar2dQuery::new(
            uniform(&mut state, -2.5, 2.5),
            uniform(&mut state, -2.5, 2.5),
            uniform(&mut state, 0.4, 1.5),
            uniform(&mut state, 0.4, 1.5),
            uniform(&mut state, 0.4, 1.5),
            uniform(&mut state, 0.4, 1.2),
            uniform(&mut state, -0.6, 0.6),
        );
        // Reject points within a small margin of any sign-flip boundary so the
        // discrete sign cannot differ between the CPU and GPU.
        if !well_conditioned(&q) {
            continue;
        }
        queries.push(q);
    }
    check(&ctx, &gpu, &queries);
}
