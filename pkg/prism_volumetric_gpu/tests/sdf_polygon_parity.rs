//! Real-device parity for the polygon / prism signed-distance twin:
//! [`GpuSdfPolygon`](prism_volumetric_gpu::sdf_polygon::GpuSdfPolygon) must
//! reproduce the three closed-form fields of the host-side independent
//! reimplementations
//! [`triangular_prism_sdf`](prism_volumetric_gpu::sdf_polygon::triangular_prism_sdf),
//! [`trapezoid_isosceles_sdf`](prism_volumetric_gpu::sdf_polygon::trapezoid_isosceles_sdf)
//! and [`star5_2d_sdf`](prism_volumetric_gpu::sdf_polygon::star5_2d_sdf) across
//! interior, surface and exterior points, the sharp apex / corner conditioning
//! hot-spots (kept clear of the exact feature), and a randomized sweep.
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
//! Each distance is a *continuous* quantity threaded through `sqrt`, products
//! and quotients, so every assertion compares with an absolute-or-relative
//! tolerance (`abs <= 1e-4 || rel <= 1e-3`, relative floor `1e-6`). Sharp
//! corners are a conditioning hot-spot where the nearest-feature and interior
//! sign branches flip; the randomized sweep rejects points within a small
//! margin of any sign-flip boundary so the `CPU` and `GPU` agree on the sign.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_polygon::{
    star5_2d_sdf, trapezoid_isosceles_sdf, triangular_prism_sdf, GpuSdfPolygon, SdfPolygonQuery,
    SdfPolygonResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous distance comparison.
const EPS: f32 = 1e-4;
/// Relative tolerance for the continuous distance comparison.
const REL: f32 = 1e-3;
/// Relative-tolerance floor, so near-zero magnitudes do not inflate the ratio.
const REL_FLOOR: f32 = 1e-6;
/// Baked `sqrt 3`, matching the twin's equilateral-triangle fold constant.
const SQRT3: f32 = 1.732_050_8;
/// Baked `cos 36 deg`, the first star-fold reflection constant.
const K1X: f32 = 0.809_017;
/// Baked `-sin 36 deg`, the second star-fold reflection constant.
const K1Y: f32 = -0.587_785_25;
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
fn oracle(q: &SdfPolygonQuery) -> [f32; 3] {
    [
        triangular_prism_sdf([q.px, q.py, q.pz], q.size, q.half_depth),
        trapezoid_isosceles_sdf([q.px, q.py], q.bottom_half, q.top_half, q.half_height),
        star5_2d_sdf([q.px, q.py], q.radius, q.inner_ratio),
    ]
}

/// Dispatches every query and pins each `GPU` distance against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfPolygon, queries: &[SdfPolygonQuery]) {
    let got: Vec<SdfPolygonResult> = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert!(
            close(result.dist_prism, want[0])
                && close(result.dist_trapezoid, want[1])
                && close(result.dist_star5, want[2]),
            "query {idx}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
            result.dist_prism,
            result.dist_trapezoid,
            result.dist_star5,
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

/// Rejects queries that sit within [`SIGN_MARGIN`] of any interior sign-flip or
/// branch boundary, where a `CPU`/`GPU` last-place difference could flip a
/// discrete sign and defeat the continuous tolerance. Replays the three fields'
/// sign arguments exactly as the twin computes them.
fn well_conditioned(q: &SdfPolygonQuery) -> bool {
    // Trapezoid `edge` branch switches on the raw y sign; guard it directly.
    if q.py.abs() < SIGN_MARGIN {
        return false;
    }

    // Prism planar sign uses the folded y.
    let mut x = q.px.abs() - q.size;
    let mut y = q.py + q.size / SQRT3;
    if x + SQRT3 * y > 0.0 {
        let folded_x = (x - SQRT3 * y) * 0.5;
        let folded_y = (-SQRT3 * x - y) * 0.5;
        x = folded_x;
        y = folded_y;
    }
    let _ = x;
    if y.abs() < SIGN_MARGIN {
        return false;
    }

    // Trapezoid interior sign uses cb.x and ca.y.
    let (r1, r2, he) = (q.bottom_half, q.top_half, q.half_height);
    let px = q.px.abs();
    let py = q.py;
    let cay = py.abs() - he;
    let k1x = r2;
    let k1y = he;
    let k2x = r2 - r1;
    let k2y = 2.0 * he;
    let denom = k2x * k2x + k2y * k2y;
    let t = (((k1x - px) * k2x + (k1y - py) * k2y) / denom).clamp(0.0, 1.0);
    let cbx = px - k1x + k2x * t;
    if cbx.abs() < SIGN_MARGIN || cay.abs() < SIGN_MARGIN {
        return false;
    }

    // Star interior sign uses the folded cross-product argument.
    let mut sx = q.px.abs();
    let mut sy = q.py;
    let d1 = (K1X * sx + K1Y * sy).max(0.0);
    sx -= 2.0 * d1 * K1X;
    sy -= 2.0 * d1 * K1Y;
    let d2 = (-K1X * sx + K1Y * sy).max(0.0);
    sx -= 2.0 * d2 * (-K1X);
    sy -= 2.0 * d2 * K1Y;
    sx = sx.abs();
    sy -= q.radius;
    let bax = q.inner_ratio * (-K1Y);
    let bay = q.inner_ratio * K1X - 1.0;
    let sign_arg = sy * bax - sx * bay;
    sign_arg.abs() >= SIGN_MARGIN
}

/// The deterministic fixtures: interior, surface-adjacent and exterior points
/// for each shape, plus the apex / corner conditioning hot-spots kept clear of
/// the exact feature. Every query carries well-conditioned, strictly positive
/// shape parameters shared across the three fields.
fn edge_fixtures() -> Vec<SdfPolygonQuery> {
    vec![
        // Interior-ish origin-adjacent point (below the x-axis to keep the
        // prism and trapezoid sign arguments clear of zero).
        SdfPolygonQuery::new(0.0, -0.30, 0.0, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4),
        // Far exterior point for all three shapes.
        SdfPolygonQuery::new(3.0, 2.0, 1.5, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4),
        // Just outside the prism side wall in x, interior in depth.
        SdfPolygonQuery::new(1.4, -0.25, 0.0, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4),
        // Past the prism depth cap but inside the triangle section.
        SdfPolygonQuery::new(0.0, -0.30, 1.6, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4),
        // Trapezoid: outside the slanted side.
        SdfPolygonQuery::new(1.6, 0.30, 0.0, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4),
        // Trapezoid: above the top edge.
        SdfPolygonQuery::new(0.20, 1.6, 0.0, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4),
        // Star: outside, above the top tip.
        SdfPolygonQuery::new(0.10, 1.7, 0.0, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4),
        // Star: near an inner concave vertex but offset off the exact edge.
        SdfPolygonQuery::new(0.55, 0.42, 0.0, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4),
        // A second shape set: smaller prism, wider trapezoid, slimmer star.
        SdfPolygonQuery::new(-0.25, -0.40, 0.20, 0.6, 0.5, 1.4, 0.3, 1.0, 0.8, 0.55),
        SdfPolygonQuery::new(2.2, -1.3, 0.9, 0.6, 0.5, 1.4, 0.3, 1.0, 0.8, 0.55),
        // Mixed off-axis exterior / interior.
        SdfPolygonQuery::new(0.70, -0.90, 0.30, 1.2, 1.0, 0.9, 0.5, 0.7, 1.3, 0.35),
        SdfPolygonQuery::new(-1.8, 1.5, -1.1, 1.2, 1.0, 0.9, 0.5, 0.7, 1.3, 0.35),
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
            eprintln!("skipping sdf_polygon parity: no wgpu adapter on this host");
        }
        return;
    };
    let gpu = GpuSdfPolygon::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn prism_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPolygon::new(&ctx);
    let q = SdfPolygonQuery::new(0.0, -0.30, 0.0, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4);
    let want = oracle(&q);
    assert!(
        want[0] < 0.0,
        "a point inside the prism has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn prism_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPolygon::new(&ctx);
    let q = SdfPolygonQuery::new(3.0, 2.0, 1.5, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4);
    let want = oracle(&q);
    assert!(want[0] > 0.0, "a far point outside the prism is positive");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn prism_depth_cap_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPolygon::new(&ctx);
    // Inside the triangular section but past the depth cap: the depth excess
    // dominates the distance.
    let q = SdfPolygonQuery::new(0.0, -0.30, 1.6, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4);
    let want = oracle(&q);
    assert!(
        want[0] > 0.0,
        "past the depth cap the prism distance is positive"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn trapezoid_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPolygon::new(&ctx);
    let q = SdfPolygonQuery::new(0.0, -0.30, 0.0, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4);
    let want = oracle(&q);
    assert!(
        want[1] < 0.0,
        "a point inside the trapezoid has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn trapezoid_slant_exterior_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPolygon::new(&ctx);
    let q = SdfPolygonQuery::new(1.6, 0.30, 0.0, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4);
    let want = oracle(&q);
    assert!(want[1] > 0.0, "a point beyond the slanted side is positive");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn star_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPolygon::new(&ctx);
    let q = SdfPolygonQuery::new(0.0, -0.30, 0.0, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4);
    let want = oracle(&q);
    assert!(
        want[2] < 0.0,
        "a point inside the star has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn star_tip_exterior_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPolygon::new(&ctx);
    let q = SdfPolygonQuery::new(0.10, 1.7, 0.0, 1.0, 0.8, 1.2, 0.6, 0.8, 1.0, 0.4);
    let want = oracle(&q);
    assert!(want[2] > 0.0, "a point beyond the upper tip is positive");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn edge_fixtures_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPolygon::new(&ctx);
    check(&ctx, &gpu, &edge_fixtures());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPolygon::new(&ctx);
    let mut state: u64 = 0x5DEE_CE66_D7A1_2B4F;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let q = SdfPolygonQuery::new(
            uniform(&mut state, -2.5, 2.5),
            uniform(&mut state, -2.5, 2.5),
            uniform(&mut state, -2.5, 2.5),
            uniform(&mut state, 0.4, 1.5),
            uniform(&mut state, 0.4, 1.2),
            uniform(&mut state, 0.6, 1.6),
            uniform(&mut state, 0.2, 0.9),
            uniform(&mut state, 0.4, 1.2),
            uniform(&mut state, 0.5, 1.5),
            uniform(&mut state, 0.3, 0.7),
        );
        // Reject points within a small margin of any interior sign-flip
        // boundary so the discrete sign cannot differ between the CPU and GPU.
        if !well_conditioned(&q) {
            continue;
        }
        queries.push(q);
    }
    check(&ctx, &gpu, &queries);
}
