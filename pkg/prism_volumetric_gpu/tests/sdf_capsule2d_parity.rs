//! Real-device parity for the planar line/stadium/ring signed-distance twin:
//! [`GpuSdfCapsule2d`](prism_volumetric_gpu::sdf_capsule2d::GpuSdfCapsule2d)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the unsigned
//! distance to a finite segment [`segment_2d`], the stadium that inflates that
//! segment by a radius [`capsule_2d`], and the ring between two radii
//! [`annulus_2d`] — across interior, surface and exterior points for every
//! shape plus a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same three closed forms (the clamped
//! segment projection for [`segment_2d`], that distance thinned by the stadium
//! radius for [`capsule_2d`], and the circle fold thinned by the band
//! half-width for [`annulus_2d`]). Because the reference and this oracle are
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
//! The segment distance is a continuous composition of a `clamp` to `[0, 1]`
//! and a `sqrt`, and the annulus field a continuous composition of an absolute
//! value and a `sqrt`, so a last-place difference near a crease perturbs the
//! output only in the last place — never a jump. The one genuine hazard is a
//! degenerate segment whose endpoints coincide, dividing by a zero squared
//! length; both the named fixtures and the randomized sweep keep every segment
//! a safe margin longer than zero, so that division is always well formed and
//! the `CPU` and `GPU` agree.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_capsule2d::{GpuSdfCapsule2d, SdfCapsule2dQuery, SdfCapsule2dResult};
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

/// Safe margin keeping every segment's squared length away from zero so the
/// clamped projection divisor is always well formed on both the `CPU` and
/// `GPU`.
const MARGIN: f32 = 0.5;

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

/// Independent reimplementation of the reference unsigned distance to the
/// finite segment `a`-`b`: project the query onto the line, clamp the parameter
/// to `[0, 1]`, then measure the residual.
fn segment_2d_oracle(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let pax = px - ax;
    let pay = py - ay;
    let bax = bx - ax;
    let bay = by - ay;
    let h = ((pax * bax + pay * bay) / (bax * bax + bay * bay)).clamp(0.0, 1.0);
    length2(pax - bax * h, pay - bay * h)
}

/// Independent reimplementation of the reference capsule (stadium) signed
/// distance: the segment distance to the endpoints thinned by the inflation
/// radius.
fn capsule_2d_oracle(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32, radius: f32) -> f32 {
    segment_2d_oracle(px, py, ax, ay, bx, by) - radius
}

/// Independent reimplementation of the reference annulus (ring) signed
/// distance: the circle field folded by an absolute value and thinned by the
/// band half-width.
fn annulus_2d_oracle(px: f32, py: f32, radius: f32, half_width: f32) -> f32 {
    (length2(px, py) - radius).abs() - half_width
}

/// Computes all three reference signed distances for one query.
fn oracle(q: &SdfCapsule2dQuery) -> SdfCapsule2dResult {
    SdfCapsule2dResult {
        capsule_2d_sd: capsule_2d_oracle(
            q.point[0],
            q.point[1],
            q.capsule_a[0],
            q.capsule_a[1],
            q.capsule_b[0],
            q.capsule_b[1],
            q.capsule_radius,
        ),
        segment_2d_sd: segment_2d_oracle(
            q.point[0],
            q.point[1],
            q.segment_a[0],
            q.segment_a[1],
            q.segment_b[0],
            q.segment_b[1],
        ),
        annulus_2d_sd: annulus_2d_oracle(
            q.point[0],
            q.point[1],
            q.annulus_radius,
            q.annulus_half_width,
        ),
    }
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfCapsule2dResult, want: &SdfCapsule2dResult) {
    assert!(
        close(got.capsule_2d_sd, want.capsule_2d_sd, DIST_ABS, DIST_REL),
        "query {idx} capsule_2d_sd: gpu {} vs cpu {}",
        got.capsule_2d_sd,
        want.capsule_2d_sd
    );
    assert!(
        close(got.segment_2d_sd, want.segment_2d_sd, DIST_ABS, DIST_REL),
        "query {idx} segment_2d_sd: gpu {} vs cpu {}",
        got.segment_2d_sd,
        want.segment_2d_sd
    );
    assert!(
        close(got.annulus_2d_sd, want.annulus_2d_sd, DIST_ABS, DIST_REL),
        "query {idx} annulus_2d_sd: gpu {} vs cpu {}",
        got.annulus_2d_sd,
        want.annulus_2d_sd
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfCapsule2d, queries: &[SdfCapsule2dQuery]) {
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

/// Builds one query from the point and all three shapes' parameters, in the
/// public constructor's argument order.
fn make_query(
    point: [f32; 2],
    capsule_a: [f32; 2],
    capsule_b: [f32; 2],
    capsule_radius: f32,
    segment_a: [f32; 2],
    segment_b: [f32; 2],
    annulus_radius: f32,
    annulus_half_width: f32,
) -> SdfCapsule2dQuery {
    SdfCapsule2dQuery::new(
        point,
        capsule_a,
        capsule_b,
        capsule_radius,
        segment_a,
        segment_b,
        annulus_radius,
        annulus_half_width,
    )
}

/// Returns whether a query keeps both the capsule and standalone segment a safe
/// margin longer than zero, so the clamped projection divisor is well formed
/// and the `CPU` and `GPU` agree.
fn well_conditioned(q: &SdfCapsule2dQuery) -> bool {
    let cap = length2(
        q.capsule_b[0] - q.capsule_a[0],
        q.capsule_b[1] - q.capsule_a[1],
    );
    if cap < MARGIN {
        return false;
    }
    let seg = length2(
        q.segment_b[0] - q.segment_a[0],
        q.segment_b[1] - q.segment_a[1],
    );
    if seg < MARGIN {
        return false;
    }
    true
}

/// A fixed battery of named cases spanning interior/surface/exterior points for
/// each of the three shapes, with the other shapes' parameters held healthy.
fn fixture_queries() -> Vec<SdfCapsule2dQuery> {
    vec![
        // Capsule: query on the segment axis, well inside the radius -> negative.
        make_query(
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [-1.0, -0.5],
            [1.0, 0.5],
            1.0,
            0.2,
        ),
        // Capsule: query exactly radius away from the slab -> ~0.
        make_query(
            [0.0, 0.5],
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [-1.0, -0.5],
            [1.0, 0.5],
            1.0,
            0.2,
        ),
        // Capsule: query far past an end cap -> strongly positive.
        make_query(
            [3.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [-1.0, -0.5],
            [1.0, 0.5],
            1.0,
            0.2,
        ),
        // Segment: foot of the perpendicular drop in the interior span.
        make_query(
            [0.0, 0.4],
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [-1.0, 0.0],
            [1.0, 0.0],
            1.0,
            0.2,
        ),
        // Segment: query beyond the +x endpoint so the clamp pins to the cap.
        make_query(
            [2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [-1.0, 0.0],
            [1.0, 0.0],
            1.0,
            0.2,
        ),
        // Segment: query right on the segment interior -> ~0.
        make_query(
            [0.25, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [-1.0, 0.0],
            [1.0, 0.0],
            1.0,
            0.2,
        ),
        // Annulus: query inside the band (on the circle radius) -> negative.
        make_query(
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [-1.0, -0.5],
            [1.0, 0.5],
            1.0,
            0.2,
        ),
        // Annulus: query on the outer band edge -> ~0.
        make_query(
            [1.2, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [-1.0, -0.5],
            [1.0, 0.5],
            1.0,
            0.2,
        ),
        // Annulus: query at the ring centre, far from the band -> positive.
        make_query(
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            0.5,
            [-1.0, -0.5],
            [1.0, 0.5],
            1.0,
            0.2,
        ),
        // Mixed: off-centre point with asymmetric shapes, every field live.
        make_query(
            [0.6, -0.4],
            [-1.2, -0.3],
            [1.1, 0.4],
            0.35,
            [-0.8, 0.9],
            [1.3, -0.2],
            1.4,
            0.25,
        ),
        // Mixed: a tilted capsule and segment with a point near but outside.
        make_query(
            [-0.3, 1.1],
            [-1.0, -0.8],
            [1.2, 0.9],
            0.3,
            [-1.1, 0.6],
            [0.9, -0.7],
            0.8,
            0.15,
        ),
    ]
}

/// Builds one well-conditioned random query: a point in `[-2, 2]^2`, capsule
/// and segment endpoints spread so each centre-line length stays well above
/// zero, bounded positive radii and band half-width, rejection-sampled to keep
/// both segments a safe margin longer than zero.
fn random_query(state: &mut u64) -> SdfCapsule2dQuery {
    loop {
        let point = [uniform(state, -2.0, 2.0), uniform(state, -2.0, 2.0)];
        let capsule_a = [uniform(state, -1.5, -0.8), uniform(state, -0.8, 0.8)];
        let capsule_b = [uniform(state, 0.8, 1.5), uniform(state, -0.8, 0.8)];
        let capsule_radius = uniform(state, 0.2, 0.8);
        let segment_a = [uniform(state, -1.5, -0.8), uniform(state, -0.8, 0.8)];
        let segment_b = [uniform(state, 0.8, 1.5), uniform(state, -0.8, 0.8)];
        let annulus_radius = uniform(state, 0.5, 1.5);
        let annulus_half_width = uniform(state, 0.1, 0.4);
        let q = make_query(
            point,
            capsule_a,
            capsule_b,
            capsule_radius,
            segment_a,
            segment_b,
            annulus_radius,
            annulus_half_width,
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
        eprintln!("skipping sdf_capsule2d parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn capsule_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    let q = fixture_queries()[0];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].capsule_2d_sd < 0.0,
        "capsule-axis distance should be negative: {}",
        got[0].capsule_2d_sd
    );
}

#[test]
fn capsule_surface_is_near_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    let q = fixture_queries()[1];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].capsule_2d_sd.abs() < 1.0e-4,
        "capsule surface distance should be ~0: {}",
        got[0].capsule_2d_sd
    );
}

#[test]
fn capsule_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    let q = fixture_queries()[2];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].capsule_2d_sd > 0.0,
        "far end-cap distance should be positive: {}",
        got[0].capsule_2d_sd
    );
}

#[test]
fn segment_perpendicular_drop_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    let q = fixture_queries()[3];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].segment_2d_sd > 0.0,
        "perpendicular drop distance should be positive: {}",
        got[0].segment_2d_sd
    );
}

#[test]
fn segment_past_endpoint_measures_to_cap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    let q = fixture_queries()[4];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].segment_2d_sd > 0.0,
        "distance past the endpoint should be positive: {}",
        got[0].segment_2d_sd
    );
}

#[test]
fn segment_on_axis_is_near_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    let q = fixture_queries()[5];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].segment_2d_sd.abs() < 1.0e-4,
        "a point on the segment should be ~0: {}",
        got[0].segment_2d_sd
    );
}

#[test]
fn annulus_inside_band_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    let q = fixture_queries()[6];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].annulus_2d_sd < 0.0,
        "a point on the circle radius lies inside the band: {}",
        got[0].annulus_2d_sd
    );
}

#[test]
fn annulus_outer_edge_is_near_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    let q = fixture_queries()[7];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].annulus_2d_sd.abs() < 1.0e-4,
        "the outer band edge distance should be ~0: {}",
        got[0].annulus_2d_sd
    );
}

#[test]
fn annulus_centre_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    let q = fixture_queries()[8];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].annulus_2d_sd > 0.0,
        "the ring centre lies outside the band: {}",
        got[0].annulus_2d_sd
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCapsule2d::new(&ctx);
    let mut state = 0x0f0e_0d0c_0b0a_0908_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin all
    // three reported distances across a wide span of points and parameters.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
