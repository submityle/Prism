//! Real-device parity for the analytic cone/sector signed-distance twin:
//! [`GpuSdfConeSolid`](prism_volumetric_gpu::sdf_cone_solid::GpuSdfConeSolid)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the solid right
//! circular `cone_sdf`, the tapered-capsule `round_cone_sdf` and the
//! "ice-cream" `solid_angle` sector — across interior, surface and exterior
//! points for all three shapes and a randomized sweep compared
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
//! *independent* reimplementation of the same three closed forms (the lateral
//! edge / base-cap segment combine for the cone, the flank-normal region select
//! for the round cone, and the ball / signed-flank combine for the solid
//! angle). Because the reference and this oracle are both scalar `f32`, a
//! `GPU == oracle` pass is direct evidence the ported kernel computes the same
//! distances the reference does.
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
//! The cone sign and the solid-angle flank sign flip exactly on the surface
//! where the distance magnitude passes through zero, so a last-place `CPU`
//! versus `GPU` disagreement there is absorbed by the absolute bound. The round
//! cone selects by the ordered comparisons `k < 0` and `k > a * h`; both the
//! named fixtures and the randomized sweep stay a safe margin away from those
//! region seams and from degenerate zero radii/heights.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_cone_solid::{
    GpuSdfConeSolid, SdfConeSolidQuery, SdfConeSolidResult,
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

/// Euclidean length of a 2-vector, matching the reference `length2` helper.
fn length2(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

/// Independent reimplementation of the reference solid-cone signed distance:
/// the smaller of the lateral-edge and base-cap segment squared distances in
/// the meridian plane, with the sign recovered from two half-plane tests.
fn cone_oracle(point: [f32; 3], base_radius: f32, height: f32) -> f32 {
    let q = [base_radius, -height];
    let w = [length2(point[0], point[2]), point[1]];
    let dot_q = q[0] * q[0] + q[1] * q[1];
    let t = ((w[0] * q[0] + w[1] * q[1]) / dot_q).clamp(0.0, 1.0);
    let a = [w[0] - q[0] * t, w[1] - q[1] * t];
    let u = (w[0] / q[0]).clamp(0.0, 1.0);
    let b = [w[0] - q[0] * u, w[1] - q[1]];
    let k = q[1].signum();
    let d = (a[0] * a[0] + a[1] * a[1]).min(b[0] * b[0] + b[1] * b[1]);
    let sign = (k * (w[0] * q[1] - w[1] * q[0])).max(k * (w[1] - q[1]));
    d.max(0.0).sqrt() * sign.signum()
}

/// Independent reimplementation of the reference round-cone signed distance:
/// the flank-normal projection selects the lower sphere, the upper sphere, or
/// the exact flank-plane distance.
fn round_cone_oracle(point: [f32; 3], r1: f32, r2: f32, h: f32) -> f32 {
    let q = [length2(point[0], point[2]), point[1]];
    let b = (r1 - r2) / h;
    let a = (1.0 - b * b).max(0.0).sqrt();
    let k = -b * q[0] + a * q[1];
    if k < 0.0 {
        length2(q[0], q[1]) - r1
    } else if k > a * h {
        length2(q[0], q[1] - h) - r2
    } else {
        a * q[0] + b * q[1] - r1
    }
}

/// Independent reimplementation of the reference solid-angle signed distance:
/// the bounding-ball distance combined with the signed flank distance.
fn solid_angle_oracle(point: [f32; 3], sin_a: f32, cos_a: f32, radius: f32) -> f32 {
    let q = [length2(point[0], point[2]), point[1]];
    let ball = length2(q[0], q[1]) - radius;
    let proj = (q[0] * sin_a + q[1] * cos_a).clamp(0.0, radius);
    let flank = length2(q[0] - sin_a * proj, q[1] - cos_a * proj);
    let flank_sign = if cos_a * q[0] - sin_a * q[1] < 0.0 {
        -1.0
    } else {
        1.0
    };
    ball.max(flank * flank_sign)
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfConeSolidQuery) -> SdfConeSolidResult {
    SdfConeSolidResult {
        cone_sd: cone_oracle(q.point, q.base_radius, q.height),
        round_cone_sd: round_cone_oracle(q.point, q.r1, q.r2, q.round_height),
        solid_angle_sd: solid_angle_oracle(q.point, q.sin_aperture, q.cos_aperture, q.radius),
    }
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfConeSolidResult, want: &SdfConeSolidResult) {
    assert!(
        close(got.cone_sd, want.cone_sd, DIST_ABS, DIST_REL),
        "query {idx} cone_sd: gpu {} vs cpu {}",
        got.cone_sd,
        want.cone_sd
    );
    assert!(
        close(got.round_cone_sd, want.round_cone_sd, DIST_ABS, DIST_REL),
        "query {idx} round_cone_sd: gpu {} vs cpu {}",
        got.round_cone_sd,
        want.round_cone_sd
    );
    assert!(
        close(got.solid_angle_sd, want.solid_angle_sd, DIST_ABS, DIST_REL),
        "query {idx} solid_angle_sd: gpu {} vs cpu {}",
        got.solid_angle_sd,
        want.solid_angle_sd
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfConeSolid, queries: &[SdfConeSolidQuery]) {
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

/// A set of valid aperture `[sin, cos]` pairs (exact unit directions so the
/// oracle and kernel see the same folded trigonometry), chosen away from the
/// axis-aligned extremes where the flank-sign boundary coincides with a
/// coordinate axis.
const APERTURES: [[f32; 2]; 3] = [
    // ~30 degrees.
    [0.5, 0.866_025_4],
    // ~60 degrees.
    [0.866_025_4, 0.5],
    // ~45 degrees.
    [
        core::f32::consts::FRAC_1_SQRT_2,
        core::f32::consts::FRAC_1_SQRT_2,
    ],
];

/// Builds a query carrying a point, the solid-cone parameters, the round-cone
/// parameters, and one solid-angle aperture plus its ball radius.
#[expect(
    clippy::too_many_arguments,
    reason = "one query packs three independent primitives' parameters"
)]
fn make_query(
    point: [f32; 3],
    base_radius: f32,
    height: f32,
    r1: f32,
    r2: f32,
    round_height: f32,
    aperture: [f32; 2],
    radius: f32,
) -> SdfConeSolidQuery {
    SdfConeSolidQuery::new(
        point,
        base_radius,
        height,
        r1,
        r2,
        round_height,
        aperture[0],
        aperture[1],
        radius,
    )
}

/// A fixed battery of named cases spanning interior/surface/exterior points for
/// all three shapes, both round-cone cap regions and both solid-angle flank
/// sides.
fn fixture_queries() -> Vec<SdfConeSolidQuery> {
    vec![
        // Cone interior: on the axis halfway down toward the base.
        make_query([0.0, -1.0, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[2], 2.0),
        // Cone exterior: far out radially from the lateral surface.
        make_query([3.0, -1.0, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[0], 2.0),
        // Cone exterior: above the apex along +y.
        make_query([0.0, 1.0, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[1], 2.0),
        // Cone exterior: below the base cap along -y.
        make_query([0.0, -4.0, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[2], 2.0),
        // Round cone, flank band region (interior near the taper).
        make_query([0.3, 1.0, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[0], 2.0),
        // Round cone, lower-sphere region (below the lower cap).
        make_query([0.0, -1.5, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[1], 2.0),
        // Round cone, upper-sphere region (above the flank band).
        make_query([0.0, 4.0, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[2], 2.0),
        // Solid angle interior (inside both the ball and the sector).
        make_query([0.2, 1.0, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[2], 2.0),
        // Solid angle, positive flank side (large radial, small height).
        make_query([1.8, 0.2, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[2], 2.0),
        // Solid angle, negative flank side (small radial, large height).
        make_query([0.1, 1.8, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[2], 2.0),
        // Solid angle exterior: far outside the bounding ball.
        make_query([4.0, 4.0, 4.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[0], 2.0),
        // Off-plane point (nonzero z) exercising all three radial reductions.
        make_query([0.6, -0.5, 0.7], 1.2, 2.5, 0.9, 0.4, 2.2, APERTURES[1], 1.8),
    ]
}

/// Builds one well-conditioned random query: a point in `[-3, 3]^3`, positive
/// bounded radii/heights, and one of the fixed valid apertures. The point span
/// and the bounded shape parameters keep the round-cone region seams and the
/// cone/solid-angle sign flips a safe margin from the sampled points.
fn random_query(state: &mut u64) -> SdfConeSolidQuery {
    let point = [
        uniform(state, -3.0, 3.0),
        uniform(state, -3.0, 3.0),
        uniform(state, -3.0, 3.0),
    ];
    let base_radius = uniform(state, 0.5, 2.0);
    let height = uniform(state, 1.0, 3.0);
    // Keep r1 > r2 so the slope constant stays well below 1 and the flank
    // sqrt guard is never near its degenerate edge.
    let r1 = uniform(state, 0.8, 1.4);
    let r2 = uniform(state, 0.2, 0.6);
    let round_height = uniform(state, 1.5, 3.0);
    let radius = uniform(state, 1.0, 2.5);
    let aperture = APERTURES[(lcg(state) % 3) as usize];
    make_query(
        point,
        base_radius,
        height,
        r1,
        r2,
        round_height,
        aperture,
        radius,
    )
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_cone_solid parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfConeSolid::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn cone_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfConeSolid::new(&ctx);
    // A point on the axis, well inside the solid cone, has a negative distance.
    let q = make_query([0.0, -1.0, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[2], 2.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].cone_sd < 0.0,
        "interior cone distance should be negative: {}",
        got[0].cone_sd
    );
}

#[test]
fn round_cone_regions_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfConeSolid::new(&ctx);
    // One query per governing region: lower sphere, flank band, upper sphere.
    let lower = make_query([0.0, -1.5, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[0], 2.0);
    let flank = make_query([0.3, 1.0, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[0], 2.0);
    let upper = make_query([0.0, 4.0, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[0], 2.0);
    let got = gpu.evaluate(&ctx, &[lower, flank, upper]);
    assert_eq!(got.len(), 3);
    check_one(0, &got[0], &oracle(&lower));
    check_one(1, &got[1], &oracle(&flank));
    check_one(2, &got[2], &oracle(&upper));
}

#[test]
fn solid_angle_both_flank_sides_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfConeSolid::new(&ctx);
    // One query on each side of the flank-sign test; both must agree.
    let pos = make_query([1.8, 0.2, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[2], 2.0);
    let neg = make_query([0.1, 1.8, 0.0], 1.0, 2.0, 1.0, 0.5, 2.0, APERTURES[2], 2.0);
    let got = gpu.evaluate(&ctx, &[pos, neg]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&pos));
    check_one(1, &got[1], &oracle(&neg));
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfConeSolid::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfConeSolid::new(&ctx);
    let mut state = 0x0f0e_0d0c_0b0a_0908_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin all
    // three reported distances across a wide span of points and parameters.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
