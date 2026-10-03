//! Real-device parity for the analytic ellipsoid/carved signed-distance twin:
//! [`GpuSdfEllipsoid`](prism_volumetric_gpu::sdf_ellipsoid::GpuSdfEllipsoid)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the gradient-bound
//! `ellipsoid_sdf`, the carved `death_star`, and the planar convex `quad_sdf` —
//! across interior, surface and exterior points for all three shapes plus a
//! randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same three closed forms (the
//! gradient-bound `k0 * (k0 - 1) / k1` for the ellipsoid, the meridian
//! half-plane rim/body select for the death star, and the edge-sign
//! classification plus clamped-edge-foot minimum for the quad). Because the
//! reference and this oracle are both scalar `f32`, a `GPU == oracle` pass is
//! direct evidence the ported kernel computes the same distances the reference
//! does.
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
//! The death-star half-plane test and the quad edge-region classification flip
//! on a seam where the two branches meet; because both signed-distance forms
//! are continuous across that seam, a last-place disagreement there changes the
//! reported distance only negligibly and is absorbed by the absolute bound.
//! Named fixtures still stay a safe margin from the seams and from the
//! ellipsoid's near-centre guard; the randomized sweep keeps radii, bite
//! distance and the fixed convex quad well away from their degenerate edges.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_ellipsoid::{GpuSdfEllipsoid, SdfEllipsoidQuery, SdfEllipsoidResult};
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

/// A fixed convex, coplanar quad (an axis-aligned square of side `2` in the
/// `z = 0` plane, wound counter-clockwise) shared by every fixture so the
/// edge-region classification stays well-conditioned.
const QUAD_A: [f32; 3] = [-1.0, -1.0, 0.0];
/// Second quad vertex.
const QUAD_B: [f32; 3] = [1.0, -1.0, 0.0];
/// Third quad vertex.
const QUAD_C: [f32; 3] = [1.0, 1.0, 0.0];
/// Fourth quad vertex.
const QUAD_D: [f32; 3] = [-1.0, 1.0, 0.0];

/// Death-star large radius shared by the fixtures and the sweep. The triple
/// `(1.0, 0.6, 0.8)` satisfies `|ra - rb| < d < ra + rb`, so the rim radius
/// `b` stays comfortably positive and far from its degenerate collapse.
const DS_LARGE: f32 = 1.0;
/// Death-star small (biting) radius.
const DS_SMALL: f32 = 0.6;
/// Death-star bite distance.
const DS_BITE: f32 = 0.8;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Euclidean length of a 3-vector, matching the reference `length` helper.
fn length3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Euclidean length of a 2-vector, matching the reference `length2` helper.
fn length2(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

/// Dot product of two 3-vectors, matching the reference `dot` helper.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Squared length of a 3-vector, matching the reference `dot2_3` helper.
fn dot2_3(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// Component-wise difference `a - b`, matching the reference `sub3` helper.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product `a x b`, matching the reference `cross` helper.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Independent reimplementation of the reference `ellipsoid_sdf`: Inigo
/// Quilez's gradient bound `k0 * (k0 - 1) / k1`, with the near-centre guard
/// returning the negated smallest radius where `k1` collapses to zero.
fn ellipsoid_oracle(point: [f32; 3], radii: [f32; 3]) -> f32 {
    let scaled = [
        point[0] / radii[0],
        point[1] / radii[1],
        point[2] / radii[2],
    ];
    let k0 = length3(scaled);
    let k1 = length3([
        scaled[0] / radii[0],
        scaled[1] / radii[1],
        scaled[2] / radii[2],
    ]);
    if k1 <= f32::MIN_POSITIVE {
        return -radii[0].min(radii[1]).min(radii[2]);
    }
    k0 * (k0 - 1.0) / k1
}

/// Independent reimplementation of the reference `death_star`: the carved
/// crescent reduced to the meridian half-plane, selecting the crater-rim circle
/// or the sphere-difference body by a single half-plane test.
fn death_star_oracle(point: [f32; 3], large_radius: f32, small_radius: f32, bite: f32) -> f32 {
    let ra = large_radius;
    let rb = small_radius;
    let d = bite;
    let a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    let b = (ra * ra - a * a).max(0.0).sqrt();
    let p = [point[0], length2(point[1], point[2])];
    if p[0] * b - p[1] * a > d * (b - p[1]).max(0.0) {
        length2(p[0] - a, p[1] - b)
    } else {
        (length2(p[0], p[1]) - ra).max(-(length2(p[0] - d, p[1]) - rb))
    }
}

/// Independent reimplementation of the reference `quad_sdf`: classifies the
/// point into the face-interior or an edge/vertex region by the sum of four
/// edge-sign tests, then takes the perpendicular projection or the minimum over
/// the four clamped edge feet.
fn quad_oracle(point: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> f32 {
    let ba = sub3(b, a);
    let pa = sub3(point, a);
    let cb = sub3(c, b);
    let pb = sub3(point, b);
    let dc = sub3(d, c);
    let pc = sub3(point, c);
    let ad = sub3(a, d);
    let pd = sub3(point, d);
    let nor = cross(ba, ad);
    let edge_region = dot3(cross(ba, nor), pa).signum()
        + dot3(cross(cb, nor), pb).signum()
        + dot3(cross(dc, nor), pc).signum()
        + dot3(cross(ad, nor), pd).signum()
        < 3.0;
    let squared = if edge_region {
        let e0 = {
            let t = (dot3(ba, pa) / dot2_3(ba)).clamp(0.0, 1.0);
            dot2_3(sub3([ba[0] * t, ba[1] * t, ba[2] * t], pa))
        };
        let e1 = {
            let t = (dot3(cb, pb) / dot2_3(cb)).clamp(0.0, 1.0);
            dot2_3(sub3([cb[0] * t, cb[1] * t, cb[2] * t], pb))
        };
        let e2 = {
            let t = (dot3(dc, pc) / dot2_3(dc)).clamp(0.0, 1.0);
            dot2_3(sub3([dc[0] * t, dc[1] * t, dc[2] * t], pc))
        };
        let e3 = {
            let t = (dot3(ad, pd) / dot2_3(ad)).clamp(0.0, 1.0);
            dot2_3(sub3([ad[0] * t, ad[1] * t, ad[2] * t], pd))
        };
        e0.min(e1).min(e2).min(e3)
    } else {
        let np = dot3(nor, pa);
        np * np / dot2_3(nor)
    };
    squared.sqrt()
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfEllipsoidQuery) -> SdfEllipsoidResult {
    SdfEllipsoidResult {
        ellipsoid_sd: ellipsoid_oracle(q.point, q.radii),
        death_star_sd: death_star_oracle(q.point, q.large_radius, q.small_radius, q.bite_distance),
        quad_sd: quad_oracle(q.point, q.quad_a, q.quad_b, q.quad_c, q.quad_d),
    }
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfEllipsoidResult, want: &SdfEllipsoidResult) {
    assert!(
        close(got.ellipsoid_sd, want.ellipsoid_sd, DIST_ABS, DIST_REL),
        "query {idx} ellipsoid_sd: gpu {} vs cpu {}",
        got.ellipsoid_sd,
        want.ellipsoid_sd
    );
    assert!(
        close(got.death_star_sd, want.death_star_sd, DIST_ABS, DIST_REL),
        "query {idx} death_star_sd: gpu {} vs cpu {}",
        got.death_star_sd,
        want.death_star_sd
    );
    assert!(
        close(got.quad_sd, want.quad_sd, DIST_ABS, DIST_REL),
        "query {idx} quad_sd: gpu {} vs cpu {}",
        got.quad_sd,
        want.quad_sd
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfEllipsoid, queries: &[SdfEllipsoidQuery]) {
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

/// Builds a query carrying the point, ellipsoid radii and death-star
/// parameters, with the shared fixed convex quad.
fn make_query(
    point: [f32; 3],
    radii: [f32; 3],
    large: f32,
    small: f32,
    bite: f32,
) -> SdfEllipsoidQuery {
    SdfEllipsoidQuery::new(
        point, radii, large, small, bite, QUAD_A, QUAD_B, QUAD_C, QUAD_D,
    )
}

/// A fixed battery of named cases spanning interior/surface/exterior points for
/// all three shapes, both death-star branches and both quad regions.
fn fixture_queries() -> Vec<SdfEllipsoidQuery> {
    let radii = [1.0, 2.0, 1.5];
    vec![
        // Ellipsoid exact centre: hits the near-centre guard (negated smallest radius).
        make_query([0.0, 0.0, 0.0], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Ellipsoid interior: a small offset from the centre, clearly negative.
        make_query([0.1, 0.1, 0.1], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Ellipsoid just inside the +x surface.
        make_query([0.95, 0.0, 0.0], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Ellipsoid exterior: well past the +x surface.
        make_query([2.0, 0.0, 0.0], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Death star, crater-rim branch (facing the biting sphere along +x).
        make_query([1.5, 0.3, 0.0], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Death star, sphere-difference body branch (far side along -x).
        make_query([-1.0, 0.0, 0.0], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Death star exterior along +x.
        make_query([3.0, 0.0, 0.0], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Death star off-meridian point (nonzero y and z fold into the radial coordinate).
        make_query([0.5, 0.4, 0.3], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Quad face-interior: directly above the square centre.
        make_query([0.0, 0.0, 2.0], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Quad face-interior: off-centre but still projecting inside the patch.
        make_query([0.3, -0.2, 1.5], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Quad edge region: off the +x side, projecting onto the x = 1 edge.
        make_query([3.0, 0.0, 0.5], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Quad vertex region: beyond the (1, 1) corner.
        make_query([2.0, 2.0, 1.0], radii, DS_LARGE, DS_SMALL, DS_BITE),
        // Generic off-axis point with distinct ellipsoid radii mixing all three forms.
        make_query(
            [0.6, -0.5, 0.7],
            [1.2, 2.5, 0.9],
            DS_LARGE,
            DS_SMALL,
            DS_BITE,
        ),
    ]
}

/// Builds one well-conditioned random query: a point in `[-3, 3]^3` and
/// positive bounded ellipsoid radii, with the fixed death-star parameters and
/// convex quad so the branch seams stay a safe margin from the sampled points.
fn random_query(state: &mut u64) -> SdfEllipsoidQuery {
    let point = [
        uniform(state, -3.0, 3.0),
        uniform(state, -3.0, 3.0),
        uniform(state, -3.0, 3.0),
    ];
    let radii = [
        uniform(state, 0.5, 2.5),
        uniform(state, 0.5, 2.5),
        uniform(state, 0.5, 2.5),
    ];
    make_query(point, radii, DS_LARGE, DS_SMALL, DS_BITE)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_ellipsoid parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfEllipsoid::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn ellipsoid_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfEllipsoid::new(&ctx);
    // The exact centre (near-centre guard) and a small interior offset must both
    // report a negative ellipsoid distance.
    let centre = make_query(
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.5],
        DS_LARGE,
        DS_SMALL,
        DS_BITE,
    );
    let inside = make_query(
        [0.1, 0.1, 0.1],
        [1.0, 2.0, 1.5],
        DS_LARGE,
        DS_SMALL,
        DS_BITE,
    );
    let got = gpu.evaluate(&ctx, &[centre, inside]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&centre));
    check_one(1, &got[1], &oracle(&inside));
    assert!(
        got[0].ellipsoid_sd < 0.0,
        "centre ellipsoid distance should be negative: {}",
        got[0].ellipsoid_sd
    );
    assert!(
        got[1].ellipsoid_sd < 0.0,
        "interior ellipsoid distance should be negative: {}",
        got[1].ellipsoid_sd
    );
}

#[test]
fn death_star_both_branches_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfEllipsoid::new(&ctx);
    // One query per governing branch of the half-plane test; both must agree.
    let rim = make_query(
        [1.5, 0.3, 0.0],
        [1.0, 2.0, 1.5],
        DS_LARGE,
        DS_SMALL,
        DS_BITE,
    );
    let body = make_query(
        [-1.0, 0.0, 0.0],
        [1.0, 2.0, 1.5],
        DS_LARGE,
        DS_SMALL,
        DS_BITE,
    );
    let got = gpu.evaluate(&ctx, &[rim, body]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&rim));
    check_one(1, &got[1], &oracle(&body));
}

#[test]
fn quad_regions_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfEllipsoid::new(&ctx);
    // One face-interior projection and one edge-region projection; both must agree.
    let face = make_query(
        [0.0, 0.0, 2.0],
        [1.0, 2.0, 1.5],
        DS_LARGE,
        DS_SMALL,
        DS_BITE,
    );
    let edge = make_query(
        [3.0, 0.0, 0.5],
        [1.0, 2.0, 1.5],
        DS_LARGE,
        DS_SMALL,
        DS_BITE,
    );
    let got = gpu.evaluate(&ctx, &[face, edge]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&face));
    check_one(1, &got[1], &oracle(&edge));
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfEllipsoid::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfEllipsoid::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin all
    // three reported distances across a wide span of points and ellipsoid radii.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
