//! Real-device parity for the signed-distance domain/scalar-operator twin:
//! [`GpuSdfDomainShell`](prism_volumetric_gpu::sdf_domain_shell::GpuSdfDomainShell)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_domain` — the `round_distance`
//! chamfer (`d - radius`), the `onion` shell (`|d| - thickness`), the uniform
//! `scale_point` (`point / factor`) paired with `scale_distance`
//! (`distance * factor`), and the rounded-rim `extrude_round` combine — across
//! interior and exterior inputs for every operator plus a randomized sweep
//! compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same five closed forms (the affine
//! chamfer, the fold-and-subtract shell, the per-axis divide/multiply scale
//! pair, and the inset-then-round extrusion combine). Because the reference and
//! this oracle are both scalar `f32`, a `GPU == oracle` pass is direct evidence
//! the ported kernel computes the same values the reference does.
//!
//! # Parity criterion
//!
//! `round_distance`, `onion`, `scale_distance` and `scale_point` are affine or
//! fold-and-subtract maps; `extrude_round` threads through one `sqrt`, so a
//! `GPU` result may land a few units in the last place from the scalar oracle.
//! Each value is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with
//! a `rel_diff` floor of `1e-6` so a near-zero expected value does not inflate
//! the relative error.
//!
//! # Conditioning
//!
//! `scale_point` divides by `factor`, so fixtures keep `|factor|` a safe margin
//! off zero via rejection sampling; `extrude_round` picks the governing feature
//! by `max(wx, wy)`, so fixtures keep the two components a safe margin apart.
//! `round_distance` and `onion` are continuous through `d == 0`, so no branch
//! flips there and they need no guard.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_domain`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_domain_shell::{
    GpuSdfDomainShell, SdfDomainShellQuery, SdfDomainShellResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each reported value. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const DIST_ABS: f32 = 1.0e-4;

/// Relative bound on each reported value, applied for larger magnitudes where a
/// few units in the last place exceed the absolute floor.
const DIST_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Minimum magnitude kept on `|factor|`, so the `scale_point` divide stays well
/// away from the degenerate zero factor.
const FACTOR_GUARD: f32 = 0.2;

/// Minimum separation kept between the `extrude_round` components `wx` and
/// `wy`, so the `max(wx, wy)` feature select never straddles the `CPU`/`GPU`
/// boundary.
const CORNER_GUARD: f32 = 0.08;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Independent reimplementation of the reference `round_distance` chamfer:
/// inflate the surface outward by subtracting the radius.
fn round_distance_oracle(d: f32, radius: f32) -> f32 {
    d - radius
}

/// Independent reimplementation of the reference `onion` shell: fold the
/// distance about zero and subtract the half-width.
fn onion_oracle(d: f32, thickness: f32) -> f32 {
    d.abs() - thickness
}

/// Independent reimplementation of the reference `scale_point`: divide each
/// axis by the uniform factor.
fn scale_point_oracle(point: [f32; 3], factor: f32) -> [f32; 3] {
    [point[0] / factor, point[1] / factor, point[2] / factor]
}

/// Independent reimplementation of the reference `scale_distance`: multiply the
/// base distance by the uniform factor.
fn scale_distance_oracle(distance: f32, factor: f32) -> f32 {
    distance * factor
}

/// Independent reimplementation of the reference `extrude_round`: inset the
/// profile by `rounding` on both axes, do the standard interior/exterior
/// combine, and offset back out by `rounding`. Uses only `abs`, `min`/`max` and
/// a single `sqrt`, matching the golden operation order exactly.
fn extrude_round_oracle(d2d: f32, z: f32, half_height: f32, rounding: f32) -> f32 {
    let wx = d2d + rounding;
    let wy = z.abs() - (half_height - rounding);
    let inside = wx.max(wy).min(0.0);
    let ox = wx.max(0.0);
    let oy = wy.max(0.0);
    let outside = (ox * ox + oy * oy).sqrt();
    inside + outside - rounding
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfDomainShellQuery) -> SdfDomainShellResult {
    SdfDomainShellResult {
        round_distance_value: round_distance_oracle(q.d, q.radius),
        onion_value: onion_oracle(q.d, q.thickness),
        scale_distance_value: scale_distance_oracle(q.distance, q.factor),
        extrude_round_value: extrude_round_oracle(q.d2d, q.z, q.half_height, q.rounding),
        scale_point_value: scale_point_oracle(q.point, q.factor),
    }
}

/// Returns whether a query keeps `|factor|` a safe margin off zero and the two
/// `extrude_round` components a safe margin apart, so neither the `scale_point`
/// divide nor the extrusion feature select can diverge between `CPU` and `GPU`.
fn well_conditioned(q: &SdfDomainShellQuery) -> bool {
    if q.factor.abs() < FACTOR_GUARD {
        return false;
    }
    let wx = q.d2d + q.rounding;
    let wy = q.z.abs() - (q.half_height - q.rounding);
    (wx - wy).abs() >= CORNER_GUARD
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfDomainShellResult, want: &SdfDomainShellResult) {
    assert!(
        close(
            got.round_distance_value,
            want.round_distance_value,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} round_distance: gpu {} vs oracle {}",
        got.round_distance_value,
        want.round_distance_value
    );
    assert!(
        close(got.onion_value, want.onion_value, DIST_ABS, DIST_REL),
        "query {idx} onion: gpu {} vs oracle {}",
        got.onion_value,
        want.onion_value
    );
    assert!(
        close(
            got.scale_distance_value,
            want.scale_distance_value,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} scale_distance: gpu {} vs oracle {}",
        got.scale_distance_value,
        want.scale_distance_value
    );
    assert!(
        close(
            got.extrude_round_value,
            want.extrude_round_value,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} extrude_round: gpu {} vs oracle {}",
        got.extrude_round_value,
        want.extrude_round_value
    );
    for (axis, (g, w)) in got
        .scale_point_value
        .iter()
        .zip(want.scale_point_value.iter())
        .enumerate()
    {
        assert!(
            close(*g, *w, DIST_ABS, DIST_REL),
            "query {idx} scale_point[{axis}]: gpu {g} vs oracle {w}"
        );
    }
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfDomainShell, queries: &[SdfDomainShellQuery]) {
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

/// Builds a query from every operator's inputs.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the five golden signatures packed into one query slot"
)]
fn make_query(
    d: f32,
    radius: f32,
    thickness: f32,
    point: [f32; 3],
    factor: f32,
    distance: f32,
    d2d: f32,
    z: f32,
    half_height: f32,
    rounding: f32,
) -> SdfDomainShellQuery {
    SdfDomainShellQuery::new(
        d,
        radius,
        thickness,
        point,
        factor,
        distance,
        d2d,
        z,
        half_height,
        rounding,
    )
}

/// A fixed battery of named cases spanning interior/exterior inputs for every
/// operator plus a zero-rounding extrusion and a cap-edge case. Each case
/// clears the factor and extrusion-corner guards.
fn fixture_queries() -> Vec<SdfDomainShellQuery> {
    vec![
        // Interior base distance: chamfer and onion both pull the surface, the
        // extrusion sits inside its slab, a modest positive scale factor.
        make_query(
            -0.6,
            0.2,
            0.15,
            [0.3, -0.4, 0.5],
            1.5,
            0.8,
            -0.5,
            0.1,
            1.0,
            0.2,
        ),
        // Exterior base distance: positive `d` well outside, extrusion outside
        // its slab in the planar direction.
        make_query(
            0.9,
            0.25,
            0.2,
            [1.0, 2.0, -1.5],
            2.0,
            -0.7,
            0.6,
            0.0,
            0.8,
            0.15,
        ),
        // Negative scale factor flips the sampled point sign; distance rescale
        // negates accordingly.
        make_query(
            0.4,
            0.3,
            0.1,
            [-0.8, 0.6, 1.2],
            -1.25,
            1.3,
            -0.9,
            0.4,
            1.2,
            0.25,
        ),
        // Zero rounding: a sharp (non-filleted) extrusion rim, interior slab.
        make_query(
            -0.3,
            0.1,
            0.05,
            [0.2, 0.2, 0.2],
            0.8,
            -0.5,
            -0.4,
            0.2,
            0.9,
            0.0,
        ),
        // Beyond the far cap along the axial direction: `|z|` clears the slab.
        make_query(
            0.2,
            0.4,
            0.3,
            [0.5, -0.5, 0.5],
            1.1,
            0.4,
            -0.6,
            1.6,
            0.7,
            0.2,
        ),
        // Planar-exterior extrusion: the profile distance is positive, so the
        // side wall governs well off the corner.
        make_query(
            -1.1,
            0.5,
            0.4,
            [2.0, -1.0, 0.5],
            2.5,
            2.2,
            0.9,
            0.1,
            1.0,
            0.3,
        ),
        // Deep interior of the slab with a small factor near the guard edge.
        make_query(
            -0.2,
            0.15,
            0.25,
            [0.1, -0.1, 0.05],
            0.25,
            -0.3,
            -0.8,
            0.0,
            1.1,
            0.2,
        ),
        // Large point magnitude stresses the scale divide; corner well apart.
        make_query(
            0.7,
            0.2,
            0.6,
            [3.0, -2.5, 2.0],
            3.0,
            -1.5,
            -1.2,
            0.5,
            1.3,
            0.1,
        ),
        // `d` exactly inside the onion wall band yet off its surfaces.
        make_query(
            0.05,
            0.3,
            0.5,
            [-0.3, 0.4, -0.2],
            1.75,
            0.9,
            -0.7,
            0.3,
            0.9,
            0.25,
        ),
    ]
}

/// Builds one well-conditioned random query via rejection sampling, retried
/// until it clears the factor and extrusion-corner guards.
fn random_query(state: &mut u64) -> SdfDomainShellQuery {
    loop {
        // Draw a factor magnitude clear of zero, with a random sign.
        let factor_mag = uniform(state, FACTOR_GUARD + 0.05, 3.0);
        let factor = if lcg(state) & 1 == 0 {
            factor_mag
        } else {
            -factor_mag
        };
        let candidate = make_query(
            uniform(state, -3.0, 3.0),
            uniform(state, 0.05, 1.0),
            uniform(state, 0.05, 1.0),
            [
                uniform(state, -3.0, 3.0),
                uniform(state, -3.0, 3.0),
                uniform(state, -3.0, 3.0),
            ],
            factor,
            uniform(state, -3.0, 3.0),
            uniform(state, -2.0, 2.0),
            uniform(state, -2.0, 2.0),
            uniform(state, 0.3, 2.0),
            uniform(state, 0.02, 0.3),
        );
        if well_conditioned(&candidate) {
            return candidate;
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
        eprintln!("skipping sdf_domain_shell parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfDomainShell::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn onion_centre_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainShell::new(&ctx);
    // A base distance inside the onion wall band (|d| < thickness) makes the
    // shell interior negative, since onion(d) = |d| - thickness.
    let q = make_query(
        -0.1,
        0.2,
        0.15,
        [0.3, -0.4, 0.5],
        1.5,
        0.8,
        -0.5,
        0.1,
        1.0,
        0.2,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].onion_value < 0.0,
        "interior onion distance should be negative: {}",
        got[0].onion_value
    );
}

#[test]
fn scale_point_divides_by_factor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainShell::new(&ctx);
    // A positive factor shrinks the sampled point toward the origin.
    let q = make_query(
        0.9,
        0.25,
        0.2,
        [1.0, 2.0, -1.5],
        2.0,
        -0.7,
        0.6,
        0.0,
        0.8,
        0.15,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        close(got[0].scale_point_value[0], 0.5, DIST_ABS, DIST_REL),
        "scale_point x should halve: {}",
        got[0].scale_point_value[0]
    );
}

#[test]
fn extrude_round_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainShell::new(&ctx);
    // A positive profile distance well outside the slab is positive everywhere.
    let q = make_query(
        -1.1,
        0.5,
        0.4,
        [2.0, -1.0, 0.5],
        2.5,
        2.2,
        0.9,
        0.1,
        1.0,
        0.3,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].extrude_round_value > 0.0,
        "exterior extrusion distance should be positive: {}",
        got[0].extrude_round_value
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainShell::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainShell::new(&ctx);
    let mut state = 0x0f0e_0d0c_0b0a_0908_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin all
    // five reported values across a wide span of inputs and parameters.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
