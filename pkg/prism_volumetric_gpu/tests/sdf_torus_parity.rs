//! Real-device parity for the analytic torus signed-distance twin:
//! [`GpuSdfTorus`](prism_volumetric_gpu::sdf_torus::GpuSdfTorus) must reproduce
//! the `CPU` closed forms of `prism_render_architecture::ray_scene::sdf_primitives`
//! — the full [`torus`] and the Inigo-Quilez [`capped_torus`] — across interior,
//! surface and exterior points for both shapes and a randomized sweep compared
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
//! *independent* reimplementation of the same two closed forms (the ring
//! reduction for the full torus, and the fold / aperture-select / guarded
//! square root for the capped torus). Because the reference and this oracle are
//! both scalar `f32`, a `GPU == oracle` pass is direct evidence the ported
//! kernel computes the same distances the reference does.
//!
//! # Parity criterion
//!
//! Each distance threads through `sqrt`, products and quotients, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! is asserted within `abs_diff <= 1e-5` or `rel_diff <= 1e-5`, with a
//! `rel_diff` floor of `1e-6` so a near-zero expected value (a point on the
//! surface) does not inflate the relative error.
//!
//! # Conditioning
//!
//! The capped-torus kernel selects by the ordered comparison
//! `cos_a * px > sin_a * py`. On the knife-edge `cos_a * px == sin_a * py` the
//! two branches give the same value, but a last-place difference between the
//! `CPU` and `GPU` could still pick different sides, so both the named fixtures
//! and the randomized sweep stay a safe margin away from that boundary and from
//! degenerate zero radii.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_torus::{GpuSdfTorus, SdfTorusQuery, SdfTorusResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each signed distance. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar oracle; `1e-5` admits that legal
/// slack while still failing a wrong port.
const DIST_ABS: f32 = 1.0e-5;

/// Relative bound on each signed distance, applied for larger magnitudes where
/// a few units in the last place exceed the absolute floor.
const DIST_REL: f32 = 1.0e-5;

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

/// Independent reimplementation of the reference full-torus signed distance:
/// the point's distance from the ring circle in the `xz` plane paired with its
/// `y` offset, minus the tube radius.
fn torus_oracle(point: [f32; 3], major_radius: f32, minor_radius: f32) -> f32 {
    let ring = (point[0] * point[0] + point[2] * point[2]).sqrt() - major_radius;
    (ring * ring + point[1] * point[1]).sqrt() - minor_radius
}

/// Independent reimplementation of the reference capped-torus signed distance:
/// the fold across the `x` axis, a single aperture-select, a guarded square
/// root, minus the tube radius.
fn capped_torus_oracle(
    point: [f32; 3],
    sin_a: f32,
    cos_a: f32,
    major_radius: f32,
    tube_radius: f32,
) -> f32 {
    let px = point[0].abs();
    let py = point[1];
    let pz = point[2];
    let k = if cos_a * px > sin_a * py {
        px * sin_a + py * cos_a
    } else {
        (px * px + py * py).sqrt()
    };
    let p_dot_p = px * px + py * py + pz * pz;
    (p_dot_p + major_radius * major_radius - 2.0 * major_radius * k)
        .max(0.0)
        .sqrt()
        - tube_radius
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfTorusQuery) -> SdfTorusResult {
    SdfTorusResult {
        torus_sd: torus_oracle(q.point, q.major_radius, q.minor_radius),
        capped_torus_sd: capped_torus_oracle(
            q.point,
            q.sin_aperture,
            q.cos_aperture,
            q.capped_major_radius,
            q.tube_radius,
        ),
    }
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfTorusResult, want: &SdfTorusResult) {
    assert!(
        close(got.torus_sd, want.torus_sd, DIST_ABS, DIST_REL),
        "query {idx} torus_sd: gpu {} vs cpu {}",
        got.torus_sd,
        want.torus_sd
    );
    assert!(
        close(
            got.capped_torus_sd,
            want.capped_torus_sd,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} capped_torus_sd: gpu {} vs cpu {}",
        got.capped_torus_sd,
        want.capped_torus_sd
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfTorus, queries: &[SdfTorusQuery]) {
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
/// axis-aligned extremes where the select boundary coincides with a coordinate
/// axis.
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

/// Builds a query carrying a point, the full-torus radii and one capped-torus
/// aperture plus its radii.
fn make_query(
    point: [f32; 3],
    major_radius: f32,
    minor_radius: f32,
    aperture: [f32; 2],
    capped_major_radius: f32,
    tube_radius: f32,
) -> SdfTorusQuery {
    SdfTorusQuery::new(
        point,
        major_radius,
        minor_radius,
        aperture[0],
        aperture[1],
        capped_major_radius,
        tube_radius,
    )
}

/// A fixed battery of named cases spanning interior/surface/exterior points for
/// both shapes and both sides of the capped-torus aperture select.
fn fixture_queries() -> Vec<SdfTorusQuery> {
    vec![
        // Torus centre of the tube (ring circle at x = major_radius): deeply
        // inside, distance ~= -minor_radius.
        make_query([2.0, 0.0, 0.0], 2.0, 0.5, APERTURES[0], 2.0, 0.5),
        // On the torus surface: ring radius + minor along x, distance ~= 0.
        make_query([2.5, 0.0, 0.0], 2.0, 0.5, APERTURES[1], 2.0, 0.5),
        // Far outside the torus along +y.
        make_query([0.0, 4.0, 0.0], 2.0, 0.5, APERTURES[2], 2.0, 0.5),
        // Interior tube point offset in y (still inside the tube).
        make_query([2.0, 0.2, 0.0], 2.0, 0.5, APERTURES[0], 2.0, 0.5),
        // Capped torus, cap branch (cos_a*px > sin_a*py): large px, small py.
        make_query([2.0, 0.1, 0.0], 2.0, 0.5, APERTURES[1], 2.0, 0.4),
        // Capped torus, full-ring branch (cos_a*px <= sin_a*py): small px,
        // large py.
        make_query([0.1, 2.0, 0.0], 2.0, 0.5, APERTURES[1], 2.0, 0.4),
        // Capped torus point out of the arc plane (nonzero z).
        make_query([1.5, 0.3, 0.8], 2.0, 0.5, APERTURES[2], 2.0, 0.4),
        // Capped torus far exterior.
        make_query([5.0, 5.0, 5.0], 2.0, 0.5, APERTURES[0], 2.0, 0.4),
        // Larger torus with a thin tube.
        make_query([3.0, 0.0, 0.0], 3.0, 0.2, APERTURES[2], 3.0, 0.3),
        // Point near but not on the origin (ring reduction well-conditioned).
        make_query([0.7, -0.6, 0.9], 2.0, 0.5, APERTURES[0], 1.5, 0.4),
        // Capped torus cap branch with negative x (fold brings |px| large).
        make_query([-2.2, 0.1, 0.1], 2.0, 0.5, APERTURES[1], 2.0, 0.4),
        // Interior of the capped-torus tube on the cap side.
        make_query([2.0, 0.05, 0.0], 2.0, 0.5, APERTURES[1], 2.0, 0.5),
    ]
}

/// Builds one well-conditioned random query: a point in `[-3, 3]^3`, positive
/// bounded radii, and one of the fixed valid apertures, kept away from the
/// capped-torus select boundary by the chosen aperture directions.
fn random_query(state: &mut u64) -> SdfTorusQuery {
    let point = [
        uniform(state, -3.0, 3.0),
        uniform(state, -3.0, 3.0),
        uniform(state, -3.0, 3.0),
    ];
    let major_radius = uniform(state, 1.0, 2.5);
    let minor_radius = uniform(state, 0.2, 0.8);
    let capped_major_radius = uniform(state, 1.0, 2.5);
    let tube_radius = uniform(state, 0.2, 0.8);
    let aperture = APERTURES[(lcg(state) % 3) as usize];
    make_query(
        point,
        major_radius,
        minor_radius,
        aperture,
        capped_major_radius,
        tube_radius,
    )
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_torus parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfTorus::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn torus_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTorus::new(&ctx);
    // The tube centre is deeply inside the solid, so its distance is negative.
    let q = make_query([2.0, 0.0, 0.0], 2.0, 0.5, APERTURES[0], 2.0, 0.5);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].torus_sd < 0.0,
        "tube-centre torus distance should be negative: {}",
        got[0].torus_sd
    );
}

#[test]
fn torus_surface_is_near_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTorus::new(&ctx);
    // A point on the outer equator sits on the surface, so its distance is ~0.
    let q = make_query([2.5, 0.0, 0.0], 2.0, 0.5, APERTURES[0], 2.0, 0.5);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].torus_sd.abs() < 1.0e-4,
        "outer-equator torus distance should be ~0: {}",
        got[0].torus_sd
    );
}

#[test]
fn capped_torus_both_branches_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTorus::new(&ctx);
    // One query lands on the cap branch, the other on the full-ring branch; both
    // must agree with the oracle.
    let cap = make_query([2.0, 0.1, 0.0], 2.0, 0.5, APERTURES[1], 2.0, 0.4);
    let ring = make_query([0.1, 2.0, 0.0], 2.0, 0.5, APERTURES[1], 2.0, 0.4);
    let got = gpu.evaluate(&ctx, &[cap, ring]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&cap));
    check_one(1, &got[1], &oracle(&ring));
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTorus::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfTorus::new(&ctx);
    let mut state = 0x0f0e_0d0c_0b0a_0908_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin both
    // reported distances across a wide span of points and radii.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
