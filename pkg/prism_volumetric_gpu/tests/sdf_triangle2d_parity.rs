//! Real-device parity for the triangle-family signed-distance twin:
//! [`GpuSdfTriangle2d`](prism_volumetric_gpu::sdf_triangle2d::GpuSdfTriangle2d)
//! must reproduce the three closed-form fields of the host-side independent
//! reimplementations
//! [`triangle_2d_sdf`](prism_volumetric_gpu::sdf_triangle2d::triangle_2d_sdf),
//! [`isosceles_triangle_2d_sdf`](prism_volumetric_gpu::sdf_triangle2d::isosceles_triangle_2d_sdf)
//! and
//! [`oriented_vesica_2d_sdf`](prism_volumetric_gpu::sdf_triangle2d::oriented_vesica_2d_sdf)
//! across interior, surface and exterior points, the sharp triangle-tip and
//! vesica-cusp conditioning hot-spots (kept clear of the exact feature), and a
//! randomized sweep.
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
//! (`abs <= 1e-4 || rel <= 1e-3`, relative floor `1e-6`). Each triangle field
//! closes with a positive-at-zero sign branch, so a discrete sign flip can only
//! happen on the exact boundary; the fixtures and the randomized sweep reject
//! points within a small margin of any zero-crossing so the `CPU` and `GPU`
//! agree on the interior sign away from the exact edge.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_triangle2d::{
    isosceles_triangle_2d_sdf, oriented_vesica_2d_sdf, triangle_2d_sdf, GpuSdfTriangle2d,
    SdfTriangle2dQuery, SdfTriangle2dResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous distance comparison.
const EPS: f32 = 1e-4;
/// Relative tolerance for the continuous distance comparison.
const REL: f32 = 1e-3;
/// Relative-tolerance floor, so near-zero magnitudes do not inflate the ratio.
const REL_FLOOR: f32 = 1e-6;
/// Rejection margin around each zero-crossing for the randomized sweep and the
/// fixtures, where a `CPU` and `GPU` last-place difference could flip the
/// discrete interior sign.
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
fn oracle(q: &SdfTriangle2dQuery) -> [f32; 3] {
    [
        triangle_2d_sdf(
            [q.px, q.py],
            [q.tri_ax, q.tri_ay],
            [q.tri_bx, q.tri_by],
            [q.tri_cx, q.tri_cy],
        ),
        isosceles_triangle_2d_sdf([q.px, q.py], q.iso_half_base, q.iso_height),
        oriented_vesica_2d_sdf(
            [q.px, q.py],
            [q.ves_ax, q.ves_ay],
            [q.ves_bx, q.ves_by],
            q.ves_w,
        ),
    ]
}

/// Dispatches every query and pins each `GPU` distance against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfTriangle2d, queries: &[SdfTriangle2dQuery]) {
    let got: Vec<SdfTriangle2dResult> = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert!(
            close(result.dist_triangle, want[0])
                && close(result.dist_isosceles, want[1])
                && close(result.dist_vesica, want[2]),
            "query {idx}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
            result.dist_triangle,
            result.dist_isosceles,
            result.dist_vesica,
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

/// Rejects queries whose nearest field magnitude sits within `SIGN_MARGIN` of
/// zero. Every discrete sign flip in the three fields occurs on the exact
/// boundary (distance zero), so keeping clear of every zero-crossing keeps the
/// `CPU` and `GPU` on the same interior side.
fn well_conditioned(q: &SdfTriangle2dQuery) -> bool {
    let want = oracle(q);
    want[0].abs() >= SIGN_MARGIN && want[1].abs() >= SIGN_MARGIN && want[2].abs() >= SIGN_MARGIN
}

/// The standard non-degenerate triangle vertices used by the named sign tests:
/// a wide base below the `x`-axis and an apex above it, well clear of any
/// collinear or zero-area configuration.
const TRI: [f32; 6] = [-1.0, -0.6, 1.0, -0.6, 0.0, 1.1];
/// The standard isosceles half-base and height for the named sign tests.
const ISO: [f32; 2] = [1.0, 1.4];
/// The standard oriented-vesica tips and apex half-width for the named sign
/// tests; the tip separation gives a half-length near `1.05`, so `0.35`
/// satisfies the proper-lens requirement `0 < w < half`.
const VES: [f32; 5] = [-0.9, -0.5, 0.9, 0.6, 0.35];

/// Builds a query at `point` with the standard shape parameters.
fn standard(px: f32, py: f32) -> SdfTriangle2dQuery {
    SdfTriangle2dQuery::new(
        px, py, TRI[0], TRI[1], TRI[2], TRI[3], TRI[4], TRI[5], ISO[0], ISO[1], VES[0], VES[1],
        VES[2], VES[3], VES[4],
    )
}

/// The deterministic fixtures: interior, surface-adjacent and exterior points
/// across varied-but-well-formed shapes, all kept clear of the exact tips,
/// edges and cusps. Every vesica keeps `0 < w < half`.
fn edge_fixtures() -> Vec<SdfTriangle2dQuery> {
    vec![
        // Interior near the triangle centroid for the standard shapes.
        standard(0.0, -0.1),
        // Far exterior for every shape.
        standard(2.6, 0.1),
        // Off-axis interior of a taller triangle with a shifted apex.
        SdfTriangle2dQuery::new(
            -0.2, 0.45, -1.2, -0.5, 1.1, -0.7, 0.1, 1.2, 0.9, 1.5, -0.8, -0.4, 1.0, 0.5, 0.3,
        ),
        // Above the base, exterior for every shape, with a flatter isosceles.
        SdfTriangle2dQuery::new(
            0.0, 1.9, -1.0, -0.6, 1.0, -0.6, 0.0, 1.1, 0.8, 1.1, -0.9, -0.5, 0.9, 0.6, 0.4,
        ),
        // Right exterior clear of the triangle slant.
        SdfTriangle2dQuery::new(
            1.6, -0.2, -1.0, -0.6, 1.0, -0.6, 0.0, 1.1, 1.0, 1.4, -1.0, -0.3, 1.0, 0.4, 0.3,
        ),
        // Lower-left exterior with a wider triangle and a thicker lens.
        SdfTriangle2dQuery::new(
            -1.6, -1.3, -0.9, -0.7, 1.2, -0.5, 0.0, 1.3, 1.3, 1.7, -0.9, -0.5, 0.9, 0.6, 0.5,
        ),
        // Interior of a small isosceles, exterior of the slim lens.
        SdfTriangle2dQuery::new(
            0.4, -0.3, -1.0, -0.6, 1.0, -0.6, 0.0, 1.1, 0.6, 0.9, -0.9, -0.5, 0.9, 0.6, 0.2,
        ),
        // Mixed interior / exterior with a narrower triangle and tilted lens.
        SdfTriangle2dQuery::new(
            0.3, 0.8, -1.1, -0.6, 0.9, -0.6, 0.1, 1.0, 1.0, 1.4, -0.7, -0.4, 0.8, 0.5, 0.25,
        ),
        // Lower-left interior for the standard triangle and lens.
        SdfTriangle2dQuery::new(
            -0.5, -0.2, -1.0, -0.6, 1.0, -0.6, 0.0, 1.1, 1.1, 1.6, -0.9, -0.5, 0.9, 0.6, 0.35,
        ),
        // Upper-right exterior with a strong lens half-width.
        SdfTriangle2dQuery::new(
            1.2, 1.0, -1.0, -0.6, 1.0, -0.6, 0.0, 1.1, 1.0, 1.3, -1.0, -0.6, 1.0, 0.6, 0.45,
        ),
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
            eprintln!("skipping sdf_triangle2d parity: no wgpu adapter on this host");
        }
        return;
    };
    let gpu = GpuSdfTriangle2d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn triangle_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTriangle2d::new(&ctx);
    let q = standard(0.0, -0.1);
    let want = oracle(&q);
    assert!(
        want[0] < 0.0,
        "a point inside the triangle has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn triangle_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTriangle2d::new(&ctx);
    let q = standard(2.5, 0.0);
    let want = oracle(&q);
    assert!(
        want[0] > 0.0,
        "a far point outside the triangle is positive"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn isosceles_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTriangle2d::new(&ctx);
    let q = standard(0.0, 0.7);
    let want = oracle(&q);
    assert!(
        want[1] < 0.0,
        "a point inside the isosceles triangle has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn isosceles_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTriangle2d::new(&ctx);
    let q = standard(0.0, 2.0);
    let want = oracle(&q);
    assert!(
        want[1] > 0.0,
        "a point above the isosceles base is positive"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn vesica_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTriangle2d::new(&ctx);
    let q = standard(0.15, 0.0);
    let want = oracle(&q);
    assert!(
        want[2] < 0.0,
        "a point inside the oriented vesica has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn vesica_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTriangle2d::new(&ctx);
    let q = standard(2.0, 2.0);
    let want = oracle(&q);
    assert!(
        want[2] > 0.0,
        "a far point outside the oriented vesica is positive"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn edge_fixtures_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTriangle2d::new(&ctx);
    let fixtures = edge_fixtures();
    for q in &fixtures {
        assert!(
            well_conditioned(q),
            "edge fixtures must stay clear of every zero-crossing"
        );
    }
    check(&ctx, &gpu, &fixtures);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTriangle2d::new(&ctx);
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let px = uniform(&mut state, -2.5, 2.5);
        let py = uniform(&mut state, -2.5, 2.5);
        let iso_half_base = uniform(&mut state, 0.5, 1.3);
        let iso_height = uniform(&mut state, 0.8, 1.7);
        let ves_w = uniform(&mut state, 0.15, 0.5);
        // The triangle and vesica tips stay fixed and non-degenerate; only the
        // point and the isosceles / lens scalars sweep so every shape is
        // well-formed (the fixed tip separation keeps `0 < w < half`).
        let q = SdfTriangle2dQuery::new(
            px,
            py,
            TRI[0],
            TRI[1],
            TRI[2],
            TRI[3],
            TRI[4],
            TRI[5],
            iso_half_base,
            iso_height,
            VES[0],
            VES[1],
            VES[2],
            VES[3],
            ves_w,
        );
        // Reject points within a small margin of any zero-crossing so the
        // discrete interior sign cannot differ between the CPU and GPU.
        if !well_conditioned(&q) {
            continue;
        }
        queries.push(q);
    }
    check(&ctx, &gpu, &queries);
}
