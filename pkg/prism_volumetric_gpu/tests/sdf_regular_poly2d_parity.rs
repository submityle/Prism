//! Real-device parity for the regular-polygon signed-distance twin:
//! [`GpuSdfRegularPoly2d`](prism_volumetric_gpu::sdf_regular_poly2d::GpuSdfRegularPoly2d)
//! must reproduce the four closed-form fields of the host-side independent
//! reimplementations
//! [`regular_hexagon_2d_sdf`](prism_volumetric_gpu::sdf_regular_poly2d::regular_hexagon_2d_sdf),
//! [`regular_pentagon_2d_sdf`](prism_volumetric_gpu::sdf_regular_poly2d::regular_pentagon_2d_sdf),
//! [`regular_octagon_2d_sdf`](prism_volumetric_gpu::sdf_regular_poly2d::regular_octagon_2d_sdf)
//! and
//! [`equilateral_triangle_2d_sdf`](prism_volumetric_gpu::sdf_regular_poly2d::equilateral_triangle_2d_sdf)
//! across interior, surface and exterior points, the sharp apex / corner
//! conditioning hot-spots (kept clear of the exact feature), and a randomized
//! sweep.
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
//! a positive-at-zero sign branch; the randomized sweep rejects points within a
//! small margin of any sign-flip boundary so the `CPU` and `GPU` agree on the
//! interior sign away from the exact edge.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_regular_poly2d::{
    equilateral_triangle_2d_sdf, regular_hexagon_2d_sdf, regular_octagon_2d_sdf,
    regular_pentagon_2d_sdf, GpuSdfRegularPoly2d, SdfRegularPoly2dQuery, SdfRegularPoly2dResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous distance comparison.
const EPS: f32 = 1e-4;
/// Relative tolerance for the continuous distance comparison.
const REL: f32 = 1e-3;
/// Relative-tolerance floor, so near-zero magnitudes do not inflate the ratio.
const REL_FLOOR: f32 = 1e-6;

/// Hexagon fold constant `x`: baked `-cos 30deg`.
const HEX_KX: f32 = -0.866_025_4;
/// Hexagon fold constant `y`: baked `sin 30deg`.
const HEX_KY: f32 = 0.5;
/// Equilateral-triangle fold constant: baked `sqrt 3`.
const TRI_K: f32 = 1.732_050_8;
/// Pentagon fold constant `x`: baked `cos 36deg`.
const PENT_KX: f32 = 0.809_017;
/// Pentagon fold constant `y`: baked `sin 36deg`.
const PENT_KY: f32 = 0.587_785_25;
/// Octagon fold constant `x`: baked `-cos 22.5deg`.
const OCT_KX: f32 = -0.923_879_5;
/// Octagon fold constant `y`: baked `sin 22.5deg`.
const OCT_KY: f32 = 0.382_683_43;
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

/// Computes the four expected distances from the host-side reimplementations.
fn oracle(q: &SdfRegularPoly2dQuery) -> [f32; 4] {
    [
        regular_hexagon_2d_sdf([q.px, q.py], q.apothem),
        regular_pentagon_2d_sdf([q.px, q.py], q.apothem),
        regular_octagon_2d_sdf([q.px, q.py], q.apothem),
        equilateral_triangle_2d_sdf([q.px, q.py], q.half_width),
    ]
}

/// Dispatches every query and pins each `GPU` distance against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfRegularPoly2d, queries: &[SdfRegularPoly2dQuery]) {
    let got: Vec<SdfRegularPoly2dResult> = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert!(
            close(result.dist_hexagon, want[0])
                && close(result.dist_pentagon, want[1])
                && close(result.dist_octagon, want[2])
                && close(result.dist_triangle, want[3]),
            "query {idx}: gpu ({}, {}, {}, {}) vs cpu ({}, {}, {}, {})",
            result.dist_hexagon,
            result.dist_pentagon,
            result.dist_octagon,
            result.dist_triangle,
            want[0],
            want[1],
            want[2],
            want[3],
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

/// Replays the hexagon fold and returns the `sign(ey)` argument the twin feeds
/// to its positive-at-zero branch.
fn hexagon_sign_arg(px0: f32, py0: f32, apothem: f32) -> f32 {
    let mut px = px0.abs();
    let mut py = py0.abs();
    let fold = 2.0 * (HEX_KX * px + HEX_KY * py).min(0.0);
    px -= fold * HEX_KX;
    py -= fold * HEX_KY;
    let _ = px;
    py - apothem
}

/// Replays the pentagon fold and returns the `sign(ey)` argument.
fn pentagon_sign_arg(px0: f32, py0: f32, apothem: f32) -> f32 {
    let r = apothem;
    let mut px = px0.abs();
    let mut py = py0;
    let f1 = 2.0 * ((-PENT_KX) * px + PENT_KY * py).min(0.0);
    px -= f1 * (-PENT_KX);
    py -= f1 * PENT_KY;
    let f2 = 2.0 * (PENT_KX * px + PENT_KY * py).min(0.0);
    px -= f2 * PENT_KX;
    py -= f2 * PENT_KY;
    let _ = px;
    py - r
}

/// Replays the octagon fold and returns the `sign(ey)` argument.
fn octagon_sign_arg(px0: f32, py0: f32, apothem: f32) -> f32 {
    let r = apothem;
    let mut px = px0.abs();
    let mut py = py0.abs();
    let f1 = 2.0 * (OCT_KX * px + OCT_KY * py).min(0.0);
    px -= f1 * OCT_KX;
    py -= f1 * OCT_KY;
    let f2 = 2.0 * ((-OCT_KX) * px + OCT_KY * py).min(0.0);
    px -= f2 * (-OCT_KX);
    py -= f2 * OCT_KY;
    let _ = px;
    py - r
}

/// Replays the triangle fold and returns the `sign(py)` argument.
fn triangle_sign_arg(px0: f32, py0: f32, half_width: f32) -> f32 {
    let r = half_width;
    let mut px = px0.abs() - r;
    let mut py = py0 + r / TRI_K;
    if px + TRI_K * py > 0.0 {
        let folded_x = (px - TRI_K * py) * 0.5;
        let folded_y = (-TRI_K * px - py) * 0.5;
        px = folded_x;
        py = folded_y;
    }
    let _ = px;
    py
}

/// Rejects queries whose sign arguments sit within `SIGN_MARGIN` of a boundary,
/// where a `CPU` and `GPU` last-place difference could flip the discrete sign.
fn well_conditioned(q: &SdfRegularPoly2dQuery) -> bool {
    hexagon_sign_arg(q.px, q.py, q.apothem).abs() >= SIGN_MARGIN
        && pentagon_sign_arg(q.px, q.py, q.apothem).abs() >= SIGN_MARGIN
        && octagon_sign_arg(q.px, q.py, q.apothem).abs() >= SIGN_MARGIN
        && triangle_sign_arg(q.px, q.py, q.half_width).abs() >= SIGN_MARGIN
}

/// The deterministic fixtures: interior, surface-adjacent and exterior points
/// for each shape, all kept clear of the exact apex and edges. Every query
/// carries well-conditioned, strictly positive shape parameters shared across
/// the four fields.
fn edge_fixtures() -> Vec<SdfRegularPoly2dQuery> {
    vec![
        // Interior point near the centre for all four shapes.
        SdfRegularPoly2dQuery::new(0.05, 0.08, 1.0, 1.0),
        // Far exterior point for all four shapes.
        SdfRegularPoly2dQuery::new(3.0, 2.4, 1.0, 1.0),
        // Just outside the right flat of hexagon / octagon.
        SdfRegularPoly2dQuery::new(1.25, 0.10, 1.0, 1.0),
        // Interior, below the x-axis, well clear of the top sign boundary.
        SdfRegularPoly2dQuery::new(-0.30, -0.45, 1.1, 1.2),
        // Above the top edge, exterior for every shape.
        SdfRegularPoly2dQuery::new(0.12, 1.6, 1.0, 1.0),
        // Off-axis interior for the larger apothem.
        SdfRegularPoly2dQuery::new(0.55, -0.35, 1.3, 1.1),
        // Lower-left exterior for every shape.
        SdfRegularPoly2dQuery::new(-1.7, -1.4, 0.8, 0.9),
        // Near a hexagon sextant seam but offset off the exact fold line.
        SdfRegularPoly2dQuery::new(0.70, 0.55, 1.0, 1.0),
        // Smaller shapes, mixed interior / exterior.
        SdfRegularPoly2dQuery::new(-0.20, 0.30, 0.6, 0.7),
        SdfRegularPoly2dQuery::new(2.1, -1.1, 0.6, 0.7),
        // Triangle base-adjacent exterior (below the base, outside).
        SdfRegularPoly2dQuery::new(0.40, -1.3, 1.0, 0.9),
        // Larger apothem, off-axis exterior.
        SdfRegularPoly2dQuery::new(-1.3, 0.95, 1.4, 1.3),
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
            eprintln!("skipping sdf_regular_poly2d parity: no wgpu adapter on this host");
        }
        return;
    };
    let gpu = GpuSdfRegularPoly2d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn hexagon_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfRegularPoly2d::new(&ctx);
    let q = SdfRegularPoly2dQuery::new(0.05, 0.08, 1.0, 1.0);
    let want = oracle(&q);
    assert!(
        want[0] < 0.0,
        "a point inside the hexagon has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn hexagon_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfRegularPoly2d::new(&ctx);
    let q = SdfRegularPoly2dQuery::new(2.5, 0.10, 1.0, 1.0);
    let want = oracle(&q);
    assert!(want[0] > 0.0, "a far point outside the hexagon is positive");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn pentagon_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfRegularPoly2d::new(&ctx);
    let q = SdfRegularPoly2dQuery::new(-0.10, -0.20, 1.0, 1.0);
    let want = oracle(&q);
    assert!(
        want[1] < 0.0,
        "a point inside the pentagon has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn pentagon_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfRegularPoly2d::new(&ctx);
    let q = SdfRegularPoly2dQuery::new(0.12, 1.7, 1.0, 1.0);
    let want = oracle(&q);
    assert!(
        want[1] > 0.0,
        "a point above the pentagon top edge is positive"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn octagon_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfRegularPoly2d::new(&ctx);
    let q = SdfRegularPoly2dQuery::new(0.30, -0.25, 1.0, 1.0);
    let want = oracle(&q);
    assert!(
        want[2] < 0.0,
        "a point inside the octagon has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn octagon_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfRegularPoly2d::new(&ctx);
    let q = SdfRegularPoly2dQuery::new(1.6, 1.4, 1.0, 1.0);
    let want = oracle(&q);
    assert!(
        want[2] > 0.0,
        "a far diagonal point outside the octagon is positive"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn triangle_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfRegularPoly2d::new(&ctx);
    let q = SdfRegularPoly2dQuery::new(0.0, 0.10, 1.0, 1.0);
    let want = oracle(&q);
    assert!(
        want[3] < 0.0,
        "a point inside the triangle has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn triangle_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfRegularPoly2d::new(&ctx);
    let q = SdfRegularPoly2dQuery::new(0.40, -1.5, 1.0, 1.0);
    let want = oracle(&q);
    assert!(want[3] > 0.0, "a point below the triangle base is positive");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn edge_fixtures_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfRegularPoly2d::new(&ctx);
    check(&ctx, &gpu, &edge_fixtures());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfRegularPoly2d::new(&ctx);
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let q = SdfRegularPoly2dQuery::new(
            uniform(&mut state, -2.5, 2.5),
            uniform(&mut state, -2.5, 2.5),
            uniform(&mut state, 0.4, 1.5),
            uniform(&mut state, 0.4, 1.5),
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
