//! Real-device parity for the planar box-family signed-distance twin:
//! [`GpuSdfBox2d`](prism_volumetric_gpu::sdf_box2d::GpuSdfBox2d) must reproduce
//! the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the axis-aligned
//! rectangle [`box_2d`], its hollow border [`box_frame_2d`], the endpoint-based
//! [`oriented_box_2d`] and the Inigo-Quilez [`rhombus_2d`] — across interior,
//! surface and exterior points for every shape plus a randomized sweep compared
//! query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same four closed forms (the rectangle
//! overshoot/least-negative-face split for [`box_2d`], its magnitude thinned by
//! a wall thickness for [`box_frame_2d`], the local-frame rotation plus the same
//! rectangle split for [`oriented_box_2d`], and the first-quadrant fold with the
//! `ndot` edge parameter and half-plane sign for [`rhombus_2d`]). Because the
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
//! The three rectangle-style fields are continuous compositions of `max(., 0)`,
//! `min(., 0)` and `sqrt`, so a last-place difference near one of their creases
//! perturbs the output only in the last place — never a jump. The one genuine
//! discontinuity is the [`rhombus_2d`] half-plane sign
//! `|px|*by + |py|*bx - bx*by`, whose flip can move the result by twice the
//! edge-line distance. Both the named fixtures and the randomized sweep keep
//! that sign argument a safe margin away from zero (and the oriented-box
//! centre-line length healthily positive), so the `CPU` and `GPU` always pick
//! the same sign.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_box2d::{GpuSdfBox2d, SdfBox2dQuery, SdfBox2dResult};
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

/// Safe margin keeping the randomized sweep's rhombus half-plane sign argument
/// and the oriented-box centre-line length away from their switch, so the `CPU`
/// and `GPU` pick the same sign.
const MARGIN: f32 = 0.05;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Euclidean length of a 2-vector; the shared planar reduction, with only
/// `sqrt` and products so it stays free of transcendental calls.
fn length2(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

/// The reference rhombus half-plane sign convention, matching the kernel's
/// `select(-1.0, 1.0, x >= 0.0)` exactly (so near the measure-zero switch the
/// two never disagree). Fixtures stay away from zero regardless.
fn signum(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Independent reimplementation of the reference axis-aligned rectangle signed
/// distance: exterior overshoot length plus the interior least-negative face
/// distance.
fn box_2d_oracle(px: f32, py: f32, hx: f32, hy: f32) -> f32 {
    let qx = px.abs() - hx;
    let qy = py.abs() - hy;
    length2(qx.max(0.0), qy.max(0.0)) + qx.max(qy).min(0.0)
}

/// Independent reimplementation of the reference hollow-frame signed distance:
/// the magnitude of the rectangle field thinned by the wall thickness.
fn box_frame_2d_oracle(px: f32, py: f32, hx: f32, hy: f32, thickness: f32) -> f32 {
    box_2d_oracle(px, py, hx, hy).abs() - thickness
}

/// Independent reimplementation of the reference oriented-box signed distance:
/// the centred point rotated into the box's local frame (local `x` along the
/// unit endpoint direction, local `y` along its normal) then the rectangle
/// split against the half-length and half-thickness.
fn oriented_box_2d_oracle(
    px: f32,
    py: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    thickness: f32,
) -> f32 {
    let abx = bx - ax;
    let aby = by - ay;
    let seg_len = length2(abx, aby);
    let dx = abx / seg_len;
    let dy = aby / seg_len;
    let cx = px - 0.5 * (ax + bx);
    let cy = py - 0.5 * (ay + by);
    let o0 = (dx * cx + dy * cy).abs() - 0.5 * seg_len;
    let o1 = (-dy * cx + dx * cy).abs() - 0.5 * thickness;
    length2(o0.max(0.0), o1.max(0.0)) + o0.max(o1).min(0.0)
}

/// Independent reimplementation of the reference rhombus signed distance: the
/// first-quadrant fold, the nearest point on the slanted edge via the `ndot`
/// parameter `h`, and the interior sign from the edge half-plane test.
fn rhombus_2d_oracle(px: f32, py: f32, bx: f32, by: f32) -> f32 {
    let rpx = px.abs();
    let rpy = py.abs();
    let nd = (bx - 2.0 * rpx) * bx - (by - 2.0 * rpy) * by;
    let h = (nd / (bx * bx + by * by)).clamp(-1.0, 1.0);
    let qx = rpx - 0.5 * bx * (1.0 - h);
    let qy = rpy - 0.5 * by * (1.0 + h);
    let d = length2(qx, qy);
    d * signum(rpx * by + rpy * bx - bx * by)
}

/// Computes all four reference signed distances for one query.
fn oracle(q: &SdfBox2dQuery) -> SdfBox2dResult {
    SdfBox2dResult {
        box_2d_sd: box_2d_oracle(
            q.point[0],
            q.point[1],
            q.box_half_extent[0],
            q.box_half_extent[1],
        ),
        box_frame_2d_sd: box_frame_2d_oracle(
            q.point[0],
            q.point[1],
            q.frame_half_extent[0],
            q.frame_half_extent[1],
            q.frame_thickness,
        ),
        oriented_box_2d_sd: oriented_box_2d_oracle(
            q.point[0],
            q.point[1],
            q.oriented_a[0],
            q.oriented_a[1],
            q.oriented_b[0],
            q.oriented_b[1],
            q.oriented_thickness,
        ),
        rhombus_2d_sd: rhombus_2d_oracle(
            q.point[0],
            q.point[1],
            q.rhombus_half_diag[0],
            q.rhombus_half_diag[1],
        ),
    }
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfBox2dResult, want: &SdfBox2dResult) {
    assert!(
        close(got.box_2d_sd, want.box_2d_sd, DIST_ABS, DIST_REL),
        "query {idx} box_2d_sd: gpu {} vs cpu {}",
        got.box_2d_sd,
        want.box_2d_sd
    );
    assert!(
        close(
            got.box_frame_2d_sd,
            want.box_frame_2d_sd,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} box_frame_2d_sd: gpu {} vs cpu {}",
        got.box_frame_2d_sd,
        want.box_frame_2d_sd
    );
    assert!(
        close(
            got.oriented_box_2d_sd,
            want.oriented_box_2d_sd,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} oriented_box_2d_sd: gpu {} vs cpu {}",
        got.oriented_box_2d_sd,
        want.oriented_box_2d_sd
    );
    assert!(
        close(got.rhombus_2d_sd, want.rhombus_2d_sd, DIST_ABS, DIST_REL),
        "query {idx} rhombus_2d_sd: gpu {} vs cpu {}",
        got.rhombus_2d_sd,
        want.rhombus_2d_sd
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfBox2d, queries: &[SdfBox2dQuery]) {
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

/// Builds one query from the point and all four shapes' parameters, in the
/// public constructor's argument order.
fn make_query(
    point: [f32; 2],
    box_half_extent: [f32; 2],
    frame_half_extent: [f32; 2],
    frame_thickness: f32,
    oriented_a: [f32; 2],
    oriented_b: [f32; 2],
    oriented_thickness: f32,
    rhombus_half_diag: [f32; 2],
) -> SdfBox2dQuery {
    SdfBox2dQuery::new(
        point,
        box_half_extent,
        frame_half_extent,
        frame_thickness,
        oriented_a,
        oriented_b,
        oriented_thickness,
        rhombus_half_diag,
    )
}

/// Returns whether a query is a safe margin away from the one genuine
/// discontinuity (the rhombus half-plane sign) and keeps the oriented-box
/// centre-line length healthily positive, so the `CPU` and `GPU` agree.
fn well_conditioned(q: &SdfBox2dQuery) -> bool {
    let rpx = q.point[0].abs();
    let rpy = q.point[1].abs();
    let bx = q.rhombus_half_diag[0];
    let by = q.rhombus_half_diag[1];
    if (rpx * by + rpy * bx - bx * by).abs() < MARGIN {
        return false;
    }
    let abx = q.oriented_b[0] - q.oriented_a[0];
    let aby = q.oriented_b[1] - q.oriented_a[1];
    if length2(abx, aby) < 0.5 {
        return false;
    }
    true
}

/// A fixed battery of named cases spanning interior/surface/exterior points for
/// each of the four shapes, with the other shapes' parameters held healthy.
fn fixture_queries() -> Vec<SdfBox2dQuery> {
    vec![
        // Rectangle: deep interior, distance strongly negative.
        make_query(
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.8],
            0.1,
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [1.0, 0.6],
        ),
        // Rectangle: on the +x face, distance ~= 0.
        make_query(
            [1.0, 0.3],
            [1.0, 1.0],
            [1.0, 0.8],
            0.1,
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [1.0, 0.6],
        ),
        // Rectangle: far exterior corner, distance strongly positive.
        make_query(
            [2.0, 2.0],
            [1.0, 1.0],
            [1.0, 0.8],
            0.1,
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [1.0, 0.6],
        ),
        // Frame: hollow centre is outside the wall, distance positive.
        make_query(
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.8],
            0.1,
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [1.0, 0.6],
        ),
        // Frame: inside the top wall, distance negative.
        make_query(
            [0.0, 0.75],
            [1.0, 1.0],
            [1.0, 0.8],
            0.1,
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [1.0, 0.6],
        ),
        // Oriented box: centre interior, distance negative.
        make_query(
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.8],
            0.1,
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [1.0, 0.6],
        ),
        // Oriented box: on the long edge (local y = half-thickness), ~= 0.
        make_query(
            [0.0, 0.25],
            [1.0, 1.0],
            [1.0, 0.8],
            0.1,
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [1.0, 0.6],
        ),
        // Oriented box: above the slab, distance positive.
        make_query(
            [0.0, 1.5],
            [1.0, 1.0],
            [1.0, 0.8],
            0.1,
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [1.0, 0.6],
        ),
        // Rhombus: centre interior, distance negative.
        make_query(
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.8],
            0.1,
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [1.0, 0.6],
        ),
        // Rhombus: far exterior, distance positive.
        make_query(
            [2.0, 2.0],
            [1.0, 1.0],
            [1.0, 0.8],
            0.1,
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [1.0, 0.6],
        ),
        // Mixed: an off-centre point with asymmetric shapes, every field live.
        make_query(
            [0.6, -0.4],
            [0.8, 1.2],
            [0.9, 0.7],
            0.15,
            [-1.2, -0.3],
            [1.1, 0.4],
            0.6,
            [1.3, 0.9],
        ),
        // Oriented box tilted off-axis, point near but outside its slab.
        make_query(
            [-0.3, 0.9],
            [1.4, 0.6],
            [1.1, 1.0],
            0.2,
            [-1.0, -0.8],
            [1.2, 0.9],
            0.4,
            [0.7, 1.4],
        ),
    ]
}

/// Builds one well-conditioned random query: a point in `[-2, 2]^2`, bounded
/// positive half extents and thicknesses, oriented-box endpoints spread so the
/// centre-line length stays well above zero, and a rhombus half-diagonal in
/// `[0.5, 1.5]`, rejection-sampled to stay away from the sign switch.
fn random_query(state: &mut u64) -> SdfBox2dQuery {
    loop {
        let point = [uniform(state, -2.0, 2.0), uniform(state, -2.0, 2.0)];
        let box_half_extent = [uniform(state, 0.5, 1.5), uniform(state, 0.5, 1.5)];
        let frame_half_extent = [uniform(state, 0.5, 1.5), uniform(state, 0.5, 1.5)];
        let frame_thickness = uniform(state, 0.05, 0.3);
        let oriented_a = [uniform(state, -1.5, -0.8), uniform(state, -0.5, 0.5)];
        let oriented_b = [uniform(state, 0.8, 1.5), uniform(state, -0.5, 0.5)];
        let oriented_thickness = uniform(state, 0.2, 0.8);
        let rhombus_half_diag = [uniform(state, 0.5, 1.5), uniform(state, 0.5, 1.5)];
        let q = make_query(
            point,
            box_half_extent,
            frame_half_extent,
            frame_thickness,
            oriented_a,
            oriented_b,
            oriented_thickness,
            rhombus_half_diag,
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
        eprintln!("skipping sdf_box2d parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn box_2d_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    let q = fixture_queries()[0];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].box_2d_sd < 0.0,
        "rectangle-centre distance should be negative: {}",
        got[0].box_2d_sd
    );
}

#[test]
fn box_2d_surface_is_near_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    let q = fixture_queries()[1];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].box_2d_sd.abs() < 1.0e-4,
        "rectangle surface distance should be ~0: {}",
        got[0].box_2d_sd
    );
}

#[test]
fn box_2d_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    let q = fixture_queries()[2];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].box_2d_sd > 0.0,
        "far-corner distance should be positive: {}",
        got[0].box_2d_sd
    );
}

#[test]
fn box_frame_hollow_center_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    let q = fixture_queries()[3];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].box_frame_2d_sd > 0.0,
        "the hollow centre lies outside the wall, distance positive: {}",
        got[0].box_frame_2d_sd
    );
}

#[test]
fn box_frame_inside_wall_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    let q = fixture_queries()[4];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].box_frame_2d_sd < 0.0,
        "a point inside the top wall should be negative: {}",
        got[0].box_frame_2d_sd
    );
}

#[test]
fn oriented_box_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    let q = fixture_queries()[5];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].oriented_box_2d_sd < 0.0,
        "oriented-box centre distance should be negative: {}",
        got[0].oriented_box_2d_sd
    );
}

#[test]
fn oriented_box_surface_is_near_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    let q = fixture_queries()[6];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].oriented_box_2d_sd.abs() < 1.0e-4,
        "oriented-box edge distance should be ~0: {}",
        got[0].oriented_box_2d_sd
    );
}

#[test]
fn oriented_box_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    let q = fixture_queries()[7];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].oriented_box_2d_sd > 0.0,
        "a point above the slab should be positive: {}",
        got[0].oriented_box_2d_sd
    );
}

#[test]
fn rhombus_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    let q = fixture_queries()[8];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].rhombus_2d_sd < 0.0,
        "rhombus-centre distance should be negative: {}",
        got[0].rhombus_2d_sd
    );
}

#[test]
fn rhombus_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    let q = fixture_queries()[9];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].rhombus_2d_sd > 0.0,
        "a far exterior point should be positive: {}",
        got[0].rhombus_2d_sd
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBox2d::new(&ctx);
    let mut state = 0x0f0e_0d0c_0b0a_0908_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin all
    // four reported distances across a wide span of points and parameters.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
