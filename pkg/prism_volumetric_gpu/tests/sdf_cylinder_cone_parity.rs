//! Real-device parity for the exact cylinder/cone signed-distance twin:
//! `GpuSdfCylinderCone` must reproduce the `CPU` closed forms of the analytic
//! oracle `prism_render_architecture::ray_scene::sdf_primitives` — the
//! `capped_cylinder` box profile, the Inigo-Quilez `capped_cone` and the
//! filleted `rounded_cylinder` — across interior, exterior, on-surface, on-axis
//! and reduction cases plus a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same three closed forms in scalar
//! `f32`, operation-for-operation. Because the reference and this oracle are
//! both scalar `f32`, a `GPU == oracle` pass is direct evidence the ported
//! kernel computes the same signed distance the reference does.
//!
//! # Parity criterion
//!
//! Each distance threads through `sqrt`, products and quotients, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a
//! `rel_diff` floor of `1e-6` so a near-zero expected value does not inflate the
//! relative error.
//!
//! # Conditioning
//!
//! The randomized sweep draws the point in `[-4, 4]^3` and positive shape
//! extents, with `round_rounding` kept below both `round_outer_radius` and
//! `round_half_height` so the fillet describes a real solid. It rejects samples
//! where the `capped_cone` sign predicate terms `ca_y` or `cb_x` fall within a
//! small margin of zero, since there a last-place wobble could flip the sign of
//! a non-tiny distance between the `CPU` and `GPU`. Everywhere else every
//! intermediate is a well-conditioned non-negative square root, a `min`/`max`
//! split (continuous across its tie) or a guarded quotient (`dot_k2` is bounded
//! below by `(2 * half_height)^2 > 0`), so a `GPU` evaluation lands a few units
//! in the last place from the scalar oracle and never straddles a branch cliff.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_cylinder_cone::{
    GpuSdfCylinderCone, SdfCylinderConeQuery, SdfCylinderConeResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each signed distance. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const SD_ABS: f32 = 1.0e-4;

/// Relative bound on each signed distance, applied for larger magnitudes where
/// a few units in the last place exceed the absolute floor.
const SD_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Margin keeping the randomized sweep clear of the `capped_cone` sign-flip
/// predicate, where a last-place wobble could flip a non-tiny distance's sign.
const CONE_SIGN_MARGIN: f32 = 2.0e-3;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Independent reimplementation of the reference `capped_cylinder`: the exact
/// signed distance to a `y`-axis cylinder of `half_height` and `radius`.
fn capped_cylinder(point: [f32; 3], half_height: f32, radius: f32) -> f32 {
    let radial = (point[0] * point[0] + point[2] * point[2]).sqrt();
    let d0 = radial - radius;
    let d1 = point[1].abs() - half_height;
    let inside = d0.max(d1).min(0.0);
    let mx = d0.max(0.0);
    let my = d1.max(0.0);
    inside + (mx * mx + my * my).sqrt()
}

/// Independent reimplementation of the reference `capped_cone`: the exact
/// signed distance to a `y`-axis cone with `bottom_radius`/`top_radius` caps.
fn capped_cone(point: [f32; 3], half_height: f32, bottom_radius: f32, top_radius: f32) -> f32 {
    let qx = (point[0] * point[0] + point[2] * point[2]).sqrt();
    let qy = point[1];
    let k1x = top_radius;
    let k1y = half_height;
    let k2x = top_radius - bottom_radius;
    let k2y = 2.0 * half_height;
    let cap_radius = if qy < 0.0 { bottom_radius } else { top_radius };
    let ca_x = qx - qx.min(cap_radius);
    let ca_y = qy.abs() - half_height;
    let km_x = k1x - qx;
    let km_y = k1y - qy;
    let dot_k2 = k2x * k2x + k2y * k2y;
    let proj = ((km_x * k2x + km_y * k2y) / dot_k2).clamp(0.0, 1.0);
    let cb_x = qx - k1x + k2x * proj;
    let cb_y = qy - k1y + k2y * proj;
    let sign = if cb_x < 0.0 && ca_y < 0.0 { -1.0 } else { 1.0 };
    let dca = ca_x * ca_x + ca_y * ca_y;
    let dcb = cb_x * cb_x + cb_y * cb_y;
    sign * dca.min(dcb).sqrt()
}

/// Independent reimplementation of the reference `rounded_cylinder`: the exact
/// signed distance to a `y`-axis cylinder with filleted vertical edges.
fn rounded_cylinder(point: [f32; 3], outer_radius: f32, rounding: f32, half_height: f32) -> f32 {
    let dx = (point[0] * point[0] + point[2] * point[2]).sqrt() - (outer_radius - rounding);
    let dy = point[1].abs() - (half_height - rounding);
    let mx = dx.max(0.0);
    let my = dy.max(0.0);
    dx.max(dy).min(0.0) + (mx * mx + my * my).sqrt() - rounding
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfCylinderConeQuery) -> SdfCylinderConeResult {
    SdfCylinderConeResult {
        capped_cylinder_sd: capped_cylinder(q.point, q.cyl_half_height, q.cyl_radius),
        capped_cone_sd: capped_cone(
            q.point,
            q.cone_half_height,
            q.cone_bottom_radius,
            q.cone_top_radius,
        ),
        rounded_cylinder_sd: rounded_cylinder(
            q.point,
            q.round_outer_radius,
            q.round_rounding,
            q.round_half_height,
        ),
    }
}

/// Pins one `GPU` result against the host oracle: each of the three signed
/// distances under the shared tolerance.
fn check_one(idx: usize, got: &SdfCylinderConeResult, want: &SdfCylinderConeResult) {
    assert!(
        close(
            got.capped_cylinder_sd,
            want.capped_cylinder_sd,
            SD_ABS,
            SD_REL
        ),
        "query {idx} capped_cylinder_sd: gpu {} vs cpu {}",
        got.capped_cylinder_sd,
        want.capped_cylinder_sd
    );
    assert!(
        close(got.capped_cone_sd, want.capped_cone_sd, SD_ABS, SD_REL),
        "query {idx} capped_cone_sd: gpu {} vs cpu {}",
        got.capped_cone_sd,
        want.capped_cone_sd
    );
    assert!(
        close(
            got.rounded_cylinder_sd,
            want.rounded_cylinder_sd,
            SD_ABS,
            SD_REL
        ),
        "query {idx} rounded_cylinder_sd: gpu {} vs cpu {}",
        got.rounded_cylinder_sd,
        want.rounded_cylinder_sd
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfCylinderCone, queries: &[SdfCylinderConeQuery]) {
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

/// Builds a query at `point` with a fixed, well-conditioned set of shape
/// parameters shared by the named fixtures.
fn query_at(point: [f32; 3]) -> SdfCylinderConeQuery {
    SdfCylinderConeQuery::new(
        point, // point
        1.0,   // cyl_half_height
        0.8,   // cyl_radius
        1.0,   // cone_half_height
        1.0,   // cone_bottom_radius
        0.4,   // cone_top_radius
        1.0,   // round_outer_radius
        0.2,   // round_rounding
        1.0,   // round_half_height
    )
}

/// A fixed battery of named points spanning interior, exterior, on-surface,
/// on-axis and off-axis cases, dispatched together.
fn fixture_queries() -> Vec<SdfCylinderConeQuery> {
    vec![
        // Deep interior near the origin.
        query_at([0.0, 0.0, 0.0]),
        // On the central axis, below and above the caps.
        query_at([0.0, -0.5, 0.0]),
        query_at([0.0, 0.5, 0.0]),
        query_at([0.0, 2.0, 0.0]),
        query_at([0.0, -2.0, 0.0]),
        // Radially outside the lateral wall.
        query_at([2.0, 0.0, 0.0]),
        query_at([0.0, 0.0, 2.5]),
        // Diagonally outside a top corner.
        query_at([1.5, 1.5, 0.0]),
        // Diagonally outside a bottom corner.
        query_at([1.3, -1.4, 0.6]),
        // Just inside the lateral wall of the cylinder.
        query_at([0.6, 0.2, 0.1]),
        // Mid-height off the axis, inside the cone frustum.
        query_at([0.3, 0.0, 0.2]),
        // Far away in every direction.
        query_at([3.0, 3.0, 3.0]),
        query_at([-3.0, -2.0, 1.0]),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_cylinder_cone parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfCylinderCone::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn cylinder_interior_is_negative_exterior_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCylinderCone::new(&ctx);
    let inside = query_at([0.0, 0.0, 0.0]);
    let outside = query_at([2.0, 0.0, 0.0]);
    let got = gpu.evaluate(&ctx, &[inside, outside]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&inside));
    check_one(1, &got[1], &oracle(&outside));
    assert!(
        got[0].capped_cylinder_sd < 0.0,
        "interior cylinder distance should be negative: {:?}",
        got[0]
    );
    assert!(
        got[1].capped_cylinder_sd > 0.0,
        "exterior cylinder distance should be positive: {:?}",
        got[1]
    );
}

#[test]
fn cone_interior_is_negative_exterior_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCylinderCone::new(&ctx);
    // Near the wide bottom cap, inside the frustum.
    let inside = query_at([0.2, -0.6, 0.1]);
    // Well outside the slanted side.
    let outside = query_at([2.5, 0.0, 0.0]);
    let got = gpu.evaluate(&ctx, &[inside, outside]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&inside));
    check_one(1, &got[1], &oracle(&outside));
    assert!(
        got[0].capped_cone_sd < 0.0,
        "interior cone distance should be negative: {:?}",
        got[0]
    );
    assert!(
        got[1].capped_cone_sd > 0.0,
        "exterior cone distance should be positive: {:?}",
        got[1]
    );
}

#[test]
fn rounded_cylinder_interior_is_negative_exterior_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCylinderCone::new(&ctx);
    let inside = query_at([0.0, 0.0, 0.0]);
    let outside = query_at([0.0, 0.0, 2.5]);
    let got = gpu.evaluate(&ctx, &[inside, outside]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&inside));
    check_one(1, &got[1], &oracle(&outside));
    assert!(
        got[0].rounded_cylinder_sd < 0.0,
        "interior rounded-cylinder distance should be negative: {:?}",
        got[0]
    );
    assert!(
        got[1].rounded_cylinder_sd > 0.0,
        "exterior rounded-cylinder distance should be positive: {:?}",
        got[1]
    );
}

#[test]
fn cone_with_equal_radii_matches_cylinder() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCylinderCone::new(&ctx);
    // Equal cap radii collapse the cone onto a cylinder; both are exact signed
    // distances of the same solid, so they must agree. The cylinder params are
    // matched to the cone's radius and half-height for the comparison.
    let point = [2.0, 0.0, 0.0];
    let q = SdfCylinderConeQuery::new(
        point, 1.0, // cyl_half_height
        1.0, // cyl_radius (matches cone radius below)
        1.0, // cone_half_height
        1.0, // cone_bottom_radius
        1.0, // cone_top_radius (equal to bottom -> cylinder)
        1.0, // round_outer_radius (unused by this assertion)
        0.2, // round_rounding
        1.0, // round_half_height
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        close(
            got[0].capped_cone_sd,
            got[0].capped_cylinder_sd,
            SD_ABS,
            SD_REL
        ),
        "equal-radius cone must match the cylinder: cone {} vs cylinder {}",
        got[0].capped_cone_sd,
        got[0].capped_cylinder_sd
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCylinderCone::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

/// Builds one well-conditioned random query: the point drawn in `[-4, 4]^3` and
/// positive shape extents, with `round_rounding` below both the outer radius
/// and the half-height. Samples near the `capped_cone` sign-flip predicate are
/// rejected (see `# Conditioning`).
fn random_query(state: &mut u64) -> SdfCylinderConeQuery {
    loop {
        let point = [
            uniform(state, -4.0, 4.0),
            uniform(state, -4.0, 4.0),
            uniform(state, -4.0, 4.0),
        ];
        let cyl_half_height = uniform(state, 0.5, 3.0);
        let cyl_radius = uniform(state, 0.2, 3.0);
        let cone_half_height = uniform(state, 0.5, 3.0);
        let cone_bottom_radius = uniform(state, 0.2, 3.0);
        let cone_top_radius = uniform(state, 0.2, 3.0);
        let round_outer_radius = uniform(state, 0.5, 3.0);
        let round_rounding = uniform(state, 0.05, 0.4) * round_outer_radius.min(1.0);
        let round_half_height = uniform(state, 0.5, 3.0);

        // Reject samples near the cone sign predicate terms, where a last-place
        // wobble could flip a non-tiny distance's sign between CPU and GPU.
        let qx = (point[0] * point[0] + point[2] * point[2]).sqrt();
        let qy = point[1];
        let k1x = cone_top_radius;
        let k1y = cone_half_height;
        let k2x = cone_top_radius - cone_bottom_radius;
        let k2y = 2.0 * cone_half_height;
        let ca_y = qy.abs() - cone_half_height;
        let km_x = k1x - qx;
        let km_y = k1y - qy;
        let dot_k2 = k2x * k2x + k2y * k2y;
        let proj = ((km_x * k2x + km_y * k2y) / dot_k2).clamp(0.0, 1.0);
        let cb_x = qx - k1x + k2x * proj;
        if ca_y.abs() < CONE_SIGN_MARGIN || cb_x.abs() < CONE_SIGN_MARGIN {
            continue;
        }
        // Keep the fillet below both extents so the rounded cylinder is a real
        // solid (a precondition of the reference parameterisation).
        if round_rounding >= round_outer_radius || round_rounding >= round_half_height {
            continue;
        }

        return SdfCylinderConeQuery::new(
            point,
            cyl_half_height,
            cyl_radius,
            cone_half_height,
            cone_bottom_radius,
            cone_top_radius,
            round_outer_radius,
            round_rounding,
            round_half_height,
        );
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCylinderCone::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin every
    // reported distance across a wide span of points and shape extents.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
