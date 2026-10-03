//! Real-device parity for the analytic sphere-ops signed-distance twin:
//! [`GpuSdfSphereOps`](prism_volumetric_gpu::sdf_sphere_ops::GpuSdfSphereOps)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the filleted
//! [`round_box`], the Inigo-Quilez [`cut_sphere`] and its hollow shell
//! [`cut_hollow_sphere`] — across interior, surface and exterior points, every
//! branch of each selector, and a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same three closed forms (the box
//! interior/exterior split for [`round_box`], the meridian reduction plus the
//! `cap` / `disc` / `rim` selector for [`cut_sphere`], and the cone-test shell
//! offset for [`cut_hollow_sphere`]). Because the reference and this oracle are
//! both scalar `f32`, a `GPU == oracle` pass is direct evidence the ported
//! kernel computes the same distances the reference does.
//!
//! # Parity criterion
//!
//! Each distance threads through `sqrt`, products and quotients, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a
//! `rel_diff` floor of `1e-6` so a near-zero expected value (a point on the
//! surface) does not inflate the relative error.
//!
//! # Conditioning
//!
//! The cut-sphere kernel selects by the ordered comparisons `s < 0` (cap vs
//! not) and `radial < w` (disc vs rim), and the hollow-sphere kernel by
//! `cut_height * radial < rim * y`. On each knife-edge a last-place difference
//! between the `CPU` and `GPU` could pick different sides, so both the named
//! fixtures and the randomized sweep stay a safe margin away from every
//! boundary (and from a near-zero cap radius `w`), keeping the branch choice
//! identical on both sides.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_sphere_ops::{
    GpuSdfSphereOps, SdfSphereOpsQuery, SdfSphereOpsResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each signed distance. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const DIST_ABS: f32 = 1.0e-4;

/// Relative bound on each signed distance, applied for larger magnitudes where
/// a few units in the last place exceed the absolute floor.
const DIST_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value (a
/// surface point) does not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Euclidean length of a 2-vector; the shared radial/meridian reduction, with
/// only `sqrt` and products so it stays free of transcendental calls.
fn length2(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

/// Independent reimplementation of the reference rounded-box signed distance:
/// the exact box field (exterior overshoot length plus interior least-negative
/// face distance) inflated outward by the corner radius.
fn round_box_oracle(point: [f32; 3], half_extent: [f32; 3], radius: f32) -> f32 {
    let dx = point[0].abs() - half_extent[0];
    let dy = point[1].abs() - half_extent[1];
    let dz = point[2].abs() - half_extent[2];
    let outside = ((dx.max(0.0)) * (dx.max(0.0))
        + (dy.max(0.0)) * (dy.max(0.0))
        + (dz.max(0.0)) * (dz.max(0.0)))
    .sqrt();
    let inside = dx.max(dy.max(dz)).min(0.0);
    outside + inside - radius
}

/// Independent reimplementation of the reference cut-sphere signed distance:
/// the meridian `(radial, y)` reduction with the single selector `s` choosing
/// the spherical cap, the flat disc face, or the circular rim.
fn cut_sphere_oracle(point: [f32; 3], radius: f32, cut_height: f32) -> f32 {
    let r = radius;
    let h = cut_height;
    let w = (r * r - h * h).max(0.0).sqrt();
    let qx = length2(point[0], point[2]);
    let qy = point[1];
    let s = ((h - r) * qx * qx + w * w * (h + r - 2.0 * qy)).max(h * qx - w * qy);
    if s < 0.0 {
        length2(qx, qy) - r
    } else if qx < w {
        h - qy
    } else {
        length2(qx - w, qy - h)
    }
}

/// Independent reimplementation of the reference cut-hollow-sphere signed
/// distance: the same meridian reduction, the rim circle past the rim's radial
/// cone and the sphere surface otherwise, offset inward by the shell thickness.
fn cut_hollow_sphere_oracle(point: [f32; 3], radius: f32, cut_height: f32, thickness: f32) -> f32 {
    let rim = (radius * radius - cut_height * cut_height).max(0.0).sqrt();
    let q0 = length2(point[0], point[2]);
    let q1 = point[1];
    let surface = if cut_height * q0 < rim * q1 {
        length2(q0 - rim, q1 - cut_height)
    } else {
        (length2(q0, q1) - radius).abs()
    };
    surface - thickness
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfSphereOpsQuery) -> SdfSphereOpsResult {
    SdfSphereOpsResult {
        round_box_sd: round_box_oracle(q.point, q.half_extent, q.round_radius),
        cut_sphere_sd: cut_sphere_oracle(q.point, q.cut_radius, q.cut_height),
        cut_hollow_sphere_sd: cut_hollow_sphere_oracle(
            q.point,
            q.hollow_radius,
            q.hollow_cut_height,
            q.hollow_thickness,
        ),
    }
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfSphereOpsResult, want: &SdfSphereOpsResult) {
    assert!(
        close(got.round_box_sd, want.round_box_sd, DIST_ABS, DIST_REL),
        "query {idx} round_box_sd: gpu {} vs cpu {}",
        got.round_box_sd,
        want.round_box_sd
    );
    assert!(
        close(got.cut_sphere_sd, want.cut_sphere_sd, DIST_ABS, DIST_REL),
        "query {idx} cut_sphere_sd: gpu {} vs cpu {}",
        got.cut_sphere_sd,
        want.cut_sphere_sd
    );
    assert!(
        close(
            got.cut_hollow_sphere_sd,
            want.cut_hollow_sphere_sd,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} cut_hollow_sphere_sd: gpu {} vs cpu {}",
        got.cut_hollow_sphere_sd,
        want.cut_hollow_sphere_sd
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfSphereOps, queries: &[SdfSphereOpsQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
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

/// Draws a `f32` in `[lo, hi)` from the generator, using only integer-to-float
/// division (no transcendental).
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = lcg(state) as f32 * (1.0 / 4_294_967_296.0);
    lo + (hi - lo) * u
}

/// Builds one query from the point and all three shapes' parameters.
fn make_query(
    point: [f32; 3],
    half_extent: [f32; 3],
    round_radius: f32,
    cut_radius: f32,
    cut_height: f32,
    hollow_radius: f32,
    hollow_cut_height: f32,
    hollow_thickness: f32,
) -> SdfSphereOpsQuery {
    SdfSphereOpsQuery::new(
        point,
        half_extent,
        round_radius,
        cut_radius,
        cut_height,
        hollow_radius,
        hollow_cut_height,
        hollow_thickness,
    )
}

/// Returns whether a query is a safe margin away from every ordered-comparison
/// boundary, so the `CPU` and `GPU` are guaranteed to pick the same branch.
///
/// Rejects configurations whose cut-sphere selector `s` or `radial - w` sits
/// within `0.05` of its switch, or whose hollow-sphere cone test
/// `cut_height * radial - rim * y` sits within `0.05` of its switch.
fn well_conditioned(q: &SdfSphereOpsQuery) -> bool {
    let r = q.cut_radius;
    let h = q.cut_height;
    let w = (r * r - h * h).max(0.0).sqrt();
    let qx = length2(q.point[0], q.point[2]);
    let qy = q.point[1];
    let s = ((h - r) * qx * qx + w * w * (h + r - 2.0 * qy)).max(h * qx - w * qy);
    if s.abs() < 0.05 {
        return false;
    }
    if (qx - w).abs() < 0.05 {
        return false;
    }
    let hr = q.hollow_radius;
    let hh = q.hollow_cut_height;
    let rim = (hr * hr - hh * hh).max(0.0).sqrt();
    let hq0 = length2(q.point[0], q.point[2]);
    let hq1 = q.point[1];
    if (hh * hq0 - rim * hq1).abs() < 0.05 {
        return false;
    }
    true
}

/// A fixed battery of named cases spanning interior/surface/exterior points for
/// the box and every branch of both sphere selectors.
fn fixture_queries() -> Vec<SdfSphereOpsQuery> {
    vec![
        // Rounded box: deep interior, distance strongly negative.
        make_query(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            0.2,
            1.0,
            0.3,
            1.0,
            0.3,
            0.1,
        ),
        // Rounded box: on the +x face plus the corner radius, distance ~= 0.
        make_query(
            [1.2, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            0.2,
            1.0,
            0.3,
            1.0,
            0.3,
            0.1,
        ),
        // Rounded box: far exterior corner, distance strongly positive.
        make_query(
            [3.0, 3.0, 3.0],
            [1.0, 1.0, 1.0],
            0.2,
            1.0,
            0.3,
            1.0,
            0.3,
            0.1,
        ),
        // Cut sphere: cap branch (s < 0) at a near-axis interior point.
        make_query(
            [0.1, 0.8, 0.0],
            [1.0, 1.0, 1.0],
            0.2,
            1.0,
            0.3,
            1.0,
            0.3,
            0.1,
        ),
        // Cut sphere: disc branch (s >= 0, radial < w) directly under the cap.
        make_query(
            [0.3, 0.1, 0.0],
            [1.0, 1.0, 1.0],
            0.2,
            1.0,
            0.3,
            1.0,
            0.3,
            0.1,
        ),
        // Cut sphere: rim branch (s >= 0, radial >= w) outside the cap radius.
        make_query(
            [1.3, -0.2, 0.0],
            [1.0, 1.0, 1.0],
            0.2,
            1.0,
            0.3,
            1.0,
            0.3,
            0.1,
        ),
        // Cut hollow sphere: rim branch (cut_height * radial < rim * y).
        make_query(
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            0.2,
            1.0,
            0.3,
            1.0,
            0.3,
            0.1,
        ),
        // Cut hollow sphere: sphere branch (cone test false).
        make_query(
            [1.5, 0.1, 0.0],
            [1.0, 1.0, 1.0],
            0.2,
            1.0,
            0.3,
            1.0,
            0.3,
            0.1,
        ),
        // Mixed: off-plane point exercising all three radial reductions at once.
        make_query(
            [0.6, -0.4, 0.7],
            [0.8, 1.2, 0.5],
            0.15,
            1.5,
            0.4,
            1.5,
            -0.3,
            0.2,
        ),
        // Larger shapes with a thin shell and a wide box.
        make_query(
            [1.1, 0.9, -0.3],
            [1.4, 0.6, 1.1],
            0.3,
            2.0,
            -0.5,
            2.0,
            0.5,
            0.25,
        ),
    ]
}

/// Builds one well-conditioned random query: a point in `[-2, 2]^3`, positive
/// bounded box half extents and corner radius, cut/hollow radii in `[1, 2]`,
/// slice heights within `±0.6` of their radius (so the cap radius stays well
/// above zero) and a bounded shell thickness, rejection-sampled to stay away
/// from every selector boundary.
fn random_query(state: &mut u64) -> SdfSphereOpsQuery {
    loop {
        let point = [
            uniform(state, -2.0, 2.0),
            uniform(state, -2.0, 2.0),
            uniform(state, -2.0, 2.0),
        ];
        let half_extent = [
            uniform(state, 0.5, 1.5),
            uniform(state, 0.5, 1.5),
            uniform(state, 0.5, 1.5),
        ];
        let round_radius = uniform(state, 0.1, 0.4);
        let cut_radius = uniform(state, 1.0, 2.0);
        let cut_height = uniform(state, -0.6, 0.6) * cut_radius;
        let hollow_radius = uniform(state, 1.0, 2.0);
        let hollow_cut_height = uniform(state, -0.6, 0.6) * hollow_radius;
        let hollow_thickness = uniform(state, 0.05, 0.3);
        let q = make_query(
            point,
            half_extent,
            round_radius,
            cut_radius,
            cut_height,
            hollow_radius,
            hollow_cut_height,
            hollow_thickness,
        );
        if well_conditioned(&q) {
            return q;
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_sphere_ops parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfSphereOps::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn round_box_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSphereOps::new(&ctx);
    // The box centre is deeply inside the solid, so its distance is negative.
    let q = make_query(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        0.2,
        1.0,
        0.3,
        1.0,
        0.3,
        0.1,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].round_box_sd < 0.0,
        "box-centre distance should be negative: {}",
        got[0].round_box_sd
    );
}

#[test]
fn round_box_surface_is_near_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSphereOps::new(&ctx);
    // The +x face plus the corner radius sits on the surface, so distance ~= 0.
    let q = make_query(
        [1.2, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        0.2,
        1.0,
        0.3,
        1.0,
        0.3,
        0.1,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].round_box_sd.abs() < 1.0e-4,
        "rounded-box surface distance should be ~0: {}",
        got[0].round_box_sd
    );
}

#[test]
fn round_box_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSphereOps::new(&ctx);
    // A far corner point is strictly outside, so its distance is positive.
    let q = make_query(
        [3.0, 3.0, 3.0],
        [1.0, 1.0, 1.0],
        0.2,
        1.0,
        0.3,
        1.0,
        0.3,
        0.1,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].round_box_sd > 0.0,
        "far-corner distance should be positive: {}",
        got[0].round_box_sd
    );
}

#[test]
fn cut_sphere_cap_branch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSphereOps::new(&ctx);
    // A near-axis interior point lands on the spherical-cap branch (s < 0).
    let q = make_query(
        [0.1, 0.8, 0.0],
        [1.0, 1.0, 1.0],
        0.2,
        1.0,
        0.3,
        1.0,
        0.3,
        0.1,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
}

#[test]
fn cut_sphere_disc_branch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSphereOps::new(&ctx);
    // Directly under the cap (s >= 0, radial < w) lands on the flat disc branch.
    let q = make_query(
        [0.3, 0.1, 0.0],
        [1.0, 1.0, 1.0],
        0.2,
        1.0,
        0.3,
        1.0,
        0.3,
        0.1,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
}

#[test]
fn cut_sphere_rim_branch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSphereOps::new(&ctx);
    // Outside the cap radius (s >= 0, radial >= w) lands on the circular rim.
    let q = make_query(
        [1.3, -0.2, 0.0],
        [1.0, 1.0, 1.0],
        0.2,
        1.0,
        0.3,
        1.0,
        0.3,
        0.1,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
}

#[test]
fn cut_hollow_sphere_rim_branch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSphereOps::new(&ctx);
    // The cone test `cut_height * radial < rim * y` holds: the rim circle rules.
    let q = make_query(
        [0.5, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        0.2,
        1.0,
        0.3,
        1.0,
        0.3,
        0.1,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
}

#[test]
fn cut_hollow_sphere_sphere_branch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSphereOps::new(&ctx);
    // The cone test fails: the unsigned sphere-surface distance rules.
    let q = make_query(
        [1.5, 0.1, 0.0],
        [1.0, 1.0, 1.0],
        0.2,
        1.0,
        0.3,
        1.0,
        0.3,
        0.1,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSphereOps::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSphereOps::new(&ctx);
    let mut state = 0x0f0e_0d0c_0b0a_0908_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin all
    // three reported distances across a wide span of points and parameters.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
