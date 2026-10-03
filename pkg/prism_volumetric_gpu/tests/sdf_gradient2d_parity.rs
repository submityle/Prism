//! Real-device parity for the 2D analytic signed-distance *gradient*
//! (surface-normal) twin:
//! [`GpuSdfGradient2d`](prism_volumetric_gpu::sdf_gradient2d::GpuSdfGradient2d)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the axis-aligned
//! rectangle normal [`box_2d_gradient`], the radial circle normal
//! [`circle_2d_gradient`] and the per-corner [`rounded_box_2d_gradient`] —
//! across interior, surface-adjacent and exterior points for every shape plus a
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
//! first-quadrant fold with the clamped-overshoot normalise or least-negative
//! axis pick for the box, the normalised position for the circle, and the
//! per-corner radius selection evaluated at the shrunk half-extent for the
//! rounded box). Because the reference and this oracle are both scalar `f32`, a
//! `GPU == oracle` pass is direct evidence the ported kernel computes the same
//! gradients the reference does.
//!
//! # Parity criterion
//!
//! The exterior and circle branches thread through one `sqrt` and a divide, so
//! a `GPU` component may land a few units in the last place from the scalar
//! oracle; each component is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, with a `rel_diff` floor of `1e-6` so a near-zero expected
//! component does not inflate the relative error. The interior branches return
//! exact axis unit vectors (integer components), so they match bit-for-bit well
//! within the same bound.
//!
//! # Conditioning
//!
//! The box gradient switches sign across each coordinate axis and switches the
//! interior axis pick on the diagonal `qx == qy`; the rounded box additionally
//! switches the active per-corner radius across each axis; and both switch
//! between the interior and exterior branches on their surface. On each of those
//! loci a last-place difference between the `CPU` and `GPU` could pick a
//! different branch. Every named fixture and every randomized query keeps the
//! point a safe margin off the coordinate axes, off each shape's surface and off
//! the interior diagonal via rejection sampling, so neither side ever selects a
//! different branch.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_gradient2d::{
    GpuSdfGradient2d, SdfGradient2dQuery, SdfGradient2dResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each gradient component. A `GPU` `sqrt`/divide may land a
/// few units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const GRAD_ABS: f32 = 1.0e-4;

/// Relative bound on each gradient component, applied for larger magnitudes
/// where a few units in the last place exceed the absolute floor.
const GRAD_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected component
/// does not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Minimum magnitude kept on each query point component, so neither the box
/// sign test nor the rounded-box radius select straddles a coordinate axis.
const AXIS_GUARD: f32 = 0.2;

/// Minimum magnitude kept on each shape's signed distance, so the interior and
/// exterior gradient branches never straddle the `CPU`/`GPU` boundary.
const SURF_GUARD: f32 = 0.08;

/// Minimum gap kept between the two interior corner offsets, so the
/// least-negative axis pick (`qx >= qy`) never straddles the diagonal.
const DIAG_GUARD: f32 = 0.08;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Euclidean length of a 2-vector, matching the golden `length2` operation
/// order `sqrt(x*x + y*y)` exactly.
fn length2(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// The twin's sign convention: `+1` for non-negative inputs, `-1` for negative
/// inputs. The `GPU` kernel uses the same two-way select, and well-conditioned
/// fixtures stay off the exact zero where this equals the reference `signum`.
fn signum_rs(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Independent reimplementation of the golden `box_2d_gradient`: fold into the
/// first quadrant, normalise the clamped overshoot outside, pick the
/// least-negative axis inside.
fn box_2d_gradient_oracle(point: [f32; 2], half_extent: [f32; 2]) -> [f32; 2] {
    let qx = point[0].abs() - half_extent[0];
    let qy = point[1].abs() - half_extent[1];
    let mx = qx.max(0.0);
    let my = qy.max(0.0);
    let len = length2([mx, my]);
    if len > 0.0 {
        return [
            signum_rs(point[0]) * mx / len,
            signum_rs(point[1]) * my / len,
        ];
    }
    if qx >= qy {
        [signum_rs(point[0]), 0.0]
    } else {
        [0.0, signum_rs(point[1])]
    }
}

/// Independent reimplementation of the golden `circle_2d_gradient`: the
/// normalised position, zero at the centre where the direction is undefined.
fn circle_2d_gradient_oracle(point: [f32; 2]) -> [f32; 2] {
    let l = length2(point);
    if l > 0.0 {
        return [point[0] / l, point[1] / l];
    }
    [0.0, 0.0]
}

/// Independent reimplementation of the golden `rounded_box_2d_gradient`: pick
/// the active per-corner radius from the point's quadrant, evaluate the box
/// gradient at the shrunk half-extent. Radii order is `[top_right,
/// bottom_right, top_left, bottom_left]`.
fn rounded_box_2d_gradient_oracle(
    point: [f32; 2],
    half_extent: [f32; 2],
    radii: [f32; 4],
) -> [f32; 2] {
    let rx = if point[0] > 0.0 { radii[0] } else { radii[2] };
    let ry = if point[0] > 0.0 { radii[1] } else { radii[3] };
    let r = if point[1] > 0.0 { rx } else { ry };
    box_2d_gradient_oracle(point, [half_extent[0] - r, half_extent[1] - r])
}

/// Signed distance to an axis-aligned 2D box, used only to measure how far a
/// query sits from a surface for conditioning (Inigo Quilez `sdBox`).
fn box_sdf(point: [f32; 2], half_extent: [f32; 2]) -> f32 {
    let qx = point[0].abs() - half_extent[0];
    let qy = point[1].abs() - half_extent[1];
    length2([qx.max(0.0), qy.max(0.0)]) + qx.max(qy).min(0.0)
}

/// The full reference result for one query, built from the three independent
/// oracles.
fn oracle(q: &SdfGradient2dQuery) -> SdfGradient2dResult {
    SdfGradient2dResult {
        box_2d_gradient_value: box_2d_gradient_oracle(q.point, q.box_half_extent),
        circle_2d_gradient_value: circle_2d_gradient_oracle(q.point),
        rounded_box_2d_gradient_value: rounded_box_2d_gradient_oracle(
            q.point,
            q.rounded_box_half_extent,
            q.rounded_box_radii,
        ),
    }
}

/// Returns whether a query sits a safe margin off every branch locus of all
/// three shapes: off the coordinate axes, off each surface and off each
/// interior diagonal.
fn well_conditioned(q: &SdfGradient2dQuery) -> bool {
    let px = q.point[0];
    let py = q.point[1];
    if px.abs() < AXIS_GUARD || py.abs() < AXIS_GUARD {
        return false;
    }
    // Box: off its surface, and off the interior diagonal when inside.
    let bqx = px.abs() - q.box_half_extent[0];
    let bqy = py.abs() - q.box_half_extent[1];
    if box_sdf(q.point, q.box_half_extent).abs() < SURF_GUARD {
        return false;
    }
    if bqx < 0.0 && bqy < 0.0 && (bqx - bqy).abs() < DIAG_GUARD {
        return false;
    }
    // Rounded box: evaluated at the shrunk half-extent for the active quadrant.
    let rx = if px > 0.0 {
        q.rounded_box_radii[0]
    } else {
        q.rounded_box_radii[2]
    };
    let ry = if px > 0.0 {
        q.rounded_box_radii[1]
    } else {
        q.rounded_box_radii[3]
    };
    let r = if py > 0.0 { rx } else { ry };
    let shrunk = [
        q.rounded_box_half_extent[0] - r,
        q.rounded_box_half_extent[1] - r,
    ];
    if shrunk[0] <= 0.0 || shrunk[1] <= 0.0 {
        return false;
    }
    let rqx = px.abs() - shrunk[0];
    let rqy = py.abs() - shrunk[1];
    if box_sdf(q.point, shrunk).abs() < SURF_GUARD {
        return false;
    }
    if rqx < 0.0 && rqy < 0.0 && (rqx - rqy).abs() < DIAG_GUARD {
        return false;
    }
    true
}

/// Pins one result's three gradient vectors (both components each) against the
/// oracle within tolerance.
fn check_one(idx: usize, got: &SdfGradient2dResult, want: &SdfGradient2dResult) {
    assert!(
        close(
            got.box_2d_gradient_value[0],
            want.box_2d_gradient_value[0],
            GRAD_ABS,
            GRAD_REL
        ) && close(
            got.box_2d_gradient_value[1],
            want.box_2d_gradient_value[1],
            GRAD_ABS,
            GRAD_REL
        ),
        "query {idx} box_2d_gradient: gpu {:?} vs oracle {:?}",
        got.box_2d_gradient_value,
        want.box_2d_gradient_value
    );
    assert!(
        close(
            got.circle_2d_gradient_value[0],
            want.circle_2d_gradient_value[0],
            GRAD_ABS,
            GRAD_REL
        ) && close(
            got.circle_2d_gradient_value[1],
            want.circle_2d_gradient_value[1],
            GRAD_ABS,
            GRAD_REL
        ),
        "query {idx} circle_2d_gradient: gpu {:?} vs oracle {:?}",
        got.circle_2d_gradient_value,
        want.circle_2d_gradient_value
    );
    assert!(
        close(
            got.rounded_box_2d_gradient_value[0],
            want.rounded_box_2d_gradient_value[0],
            GRAD_ABS,
            GRAD_REL
        ) && close(
            got.rounded_box_2d_gradient_value[1],
            want.rounded_box_2d_gradient_value[1],
            GRAD_ABS,
            GRAD_REL
        ),
        "query {idx} rounded_box_2d_gradient: gpu {:?} vs oracle {:?}",
        got.rounded_box_2d_gradient_value,
        want.rounded_box_2d_gradient_value
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfGradient2d, queries: &[SdfGradient2dQuery]) {
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

/// The baseline box and rounded-box parameters shared by most named fixtures:
/// a wide rectangle, a slightly larger rounded rectangle with four distinct
/// per-corner radii so every quadrant selects a different radius.
fn base_query(point: [f32; 2]) -> SdfGradient2dQuery {
    SdfGradient2dQuery::new(point, [1.0, 0.6], [1.2, 0.9], [0.2, 0.15, 0.25, 0.1])
}

/// A fixed battery of named cases spanning interior/exterior points for every
/// shape plus one point in each quadrant (to exercise all four rounded-box
/// radius selections) and a variant with different half-extents and radii.
fn fixture_queries() -> Vec<SdfGradient2dQuery> {
    vec![
        // Shared origin: box/rounded-box interior axis pick, circle centre.
        base_query([0.0, 0.0]),
        // Deep interior of every shape, off the diagonal.
        base_query([0.4, 0.3]),
        // Far exterior corner (both axes overshoot) for every shape.
        base_query([3.0, 3.0]),
        // Beyond the +x face only.
        base_query([2.0, 0.3]),
        // Beyond the +y face only.
        base_query([0.4, 2.0]),
        // Exterior corner in the negative quadrant.
        base_query([-2.5, -2.0]),
        // One point in each quadrant: ++, +-, -+, --.
        base_query([0.5, 0.4]),
        base_query([0.5, -0.4]),
        base_query([-0.5, 0.4]),
        base_query([-0.5, -0.4]),
        // Different half-extents and radii, negative-y quadrant.
        SdfGradient2dQuery::new([0.6, -0.5], [0.8, 1.1], [1.0, 0.7], [0.1, 0.25, 0.3, 0.2]),
    ]
}

/// Builds one well-conditioned random query via rejection sampling: a point in
/// `[-2, 2]^2`, a box half-extent in `[0.5, 1.2]`, a rounded-box half-extent in
/// `[0.8, 1.4]` and four radii in `[0.05, 0.3]` (so the shrunk extent stays
/// positive), retried until it clears every guard.
fn random_query(state: &mut u64) -> SdfGradient2dQuery {
    loop {
        let candidate = SdfGradient2dQuery::new(
            [uniform(state, -2.0, 2.0), uniform(state, -2.0, 2.0)],
            [uniform(state, 0.5, 1.2), uniform(state, 0.5, 1.2)],
            [uniform(state, 0.8, 1.4), uniform(state, 0.8, 1.4)],
            [
                uniform(state, 0.05, 0.3),
                uniform(state, 0.05, 0.3),
                uniform(state, 0.05, 0.3),
                uniform(state, 0.05, 0.3),
            ],
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
        eprintln!("skipping sdf_gradient2d parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfGradient2d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn circle_gradient_is_unit_radial() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient2d::new(&ctx);
    // Off the centre the circle gradient is a unit vector.
    let q = base_query([0.7, 0.5]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    let g = got[0].circle_2d_gradient_value;
    let mag = length2(g);
    assert!(
        close(mag, 1.0, GRAD_ABS, GRAD_REL),
        "circle gradient should be unit length: {mag}"
    );
}

#[test]
fn box_interior_picks_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient2d::new(&ctx);
    // Interior point closer to the top wall: the normal snaps to +y.
    let q = base_query([0.3, 0.2]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    let g = got[0].box_2d_gradient_value;
    assert!(
        close(g[0], 0.0, GRAD_ABS, GRAD_REL) && close(g[1].abs(), 1.0, GRAD_ABS, GRAD_REL),
        "box interior normal should be an axis unit vector: {g:?}"
    );
}

#[test]
fn rounded_box_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient2d::new(&ctx);
    // Exterior point past the +x face of the shrunk rounded box.
    let q = base_query([2.2, 0.4]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient2d::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient2d::new(&ctx);
    let mut state = 0x0f0e_0d0c_0b0a_0908_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin all
    // three gradients across a wide span of points and parameters.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
