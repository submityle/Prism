//! Real-device parity for the bicubic uniform *B-spline* surface patch twin:
//! `GpuBsplinePatch` must reproduce the `CPU` closed form of the oracle
//! `prism_render_architecture::ray_scene::bspline_patch` —
//! `BsplinePatch::{point, normal}`, which first converts the `4x4`
//! approximating control net to Bézier form by the uniform-knot basis change
//! and then evaluates through `prism_render_architecture::ray_scene::bezier_patch`
//! — across a flat net, a curved net, a fully collapsed net, a colinear net,
//! a named fixture batch and a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed form in scalar `f32`,
//! operation-for-operation. First the `4x4` uniform cubic B-spline net is
//! converted to a `4x4` Bézier net by applying the uniform-knot span conversion
//! (`b0 = (c0 + 4·c1 + c2)/6`, `b1 = (2·c1 + c2)/3`, `b2 = (c1 + 2·c2)/3`,
//! `b3 = (c1 + 4·c2 + c3)/6`) along each `u` row then along each `v` column.
//! The surface point then comes from a cubic De Casteljau per `v`-row in `u`
//! collapsed by a cubic De Casteljau in `v`; the analytic normal is the
//! normalized cross product of the two partial derivatives, using the
//! reference's four-step interior nudge (`eps = 1e-3 · step`) and the
//! `[0, 0, 1]` fallback when the whole loop is exhausted. Because the reference
//! and this oracle are both scalar `f32`, a `GPU == oracle` pass is direct
//! evidence the ported kernel evaluates the same surface and classifies the
//! same degenerate frames.
//!
//! # Parity criterion
//!
//! The discrete `degenerate` flag is asserted exactly. The continuous `point`
//! and `normal` thread through nested interpolations, a cross product and one
//! normalize `sqrt`, so a `GPU` result may land a few units in the last place
//! from the scalar oracle; each component is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a `rel_diff` floor of `1e-6`
//! so a near-zero expected component does not inflate the relative error.
//!
//! # Conditioning
//!
//! Fixtures and the randomized sweep reject-sample away from the one branch
//! cliff — the step-0 cross product near zero — so a last-place wobble never
//! flips which side of the degeneracy guard the `CPU` and `GPU` land on. Random
//! samples also keep `u` and `v` well inside `[0, 1]`. The collapsed and
//! colinear fixtures deliberately drive the whole nudge loop to its fallback,
//! which both sides reach deterministically as `[0, 0, 1]`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::bspline_patch`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::bspline_patch::{BsplinePatchQuery, BsplinePatchResult, GpuBsplinePatch};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each continuous component. A `GPU` multiply-add may land a
/// few units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const SD_ABS: f32 = 1.0e-4;

/// Relative bound on each continuous component, applied for larger magnitudes
/// where a few units in the last place exceed the absolute floor.
const SD_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= SD_ABS || rel <= SD_REL
}

/// Component-wise sum `a + b`.
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales `a` by the scalar `s`.
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Component-wise difference `a - b`.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Dot product of two 3-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a × b`.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Linear interpolation `a + (b - a) * t`, component-wise.
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Converts one uniform cubic B-spline span into the four cubic Bézier control
/// points of its central segment, matching the reference `span_to_bezier` term
/// and operation order.
fn span_to_bezier(c0: [f32; 3], c1: [f32; 3], c2: [f32; 3], c3: [f32; 3]) -> [[f32; 3]; 4] {
    let sixth = 1.0 / 6.0;
    let third = 1.0 / 3.0;
    let b0 = scale3(add3(add3(c0, scale3(c1, 4.0)), c2), sixth);
    let b1 = scale3(add3(scale3(c1, 2.0), c2), third);
    let b2 = scale3(add3(c1, scale3(c2, 2.0)), third);
    let b3 = scale3(add3(add3(c1, scale3(c2, 4.0)), c3), sixth);
    [b0, b1, b2, b3]
}

/// Tensor-product B-spline-to-Bézier conversion: run `span_to_bezier` along each
/// of the four `u` rows, then along each of the four `v` columns of the
/// intermediate net, matching the reference `to_bezier`.
fn to_bezier(control: &[[f32; 3]; 16]) -> [[f32; 3]; 16] {
    let mut tmp = [[0.0f32; 3]; 16];
    for row in 0..4 {
        let base = row * 4;
        let span = span_to_bezier(
            control[base],
            control[base + 1],
            control[base + 2],
            control[base + 3],
        );
        tmp[base] = span[0];
        tmp[base + 1] = span[1];
        tmp[base + 2] = span[2];
        tmp[base + 3] = span[3];
    }
    let mut out = [[0.0f32; 3]; 16];
    for col in 0..4 {
        let span = span_to_bezier(tmp[col], tmp[4 + col], tmp[8 + col], tmp[12 + col]);
        out[col] = span[0];
        out[4 + col] = span[1];
        out[8 + col] = span[2];
        out[12 + col] = span[3];
    }
    out
}

/// Cubic Bézier point over four control points: three nested `lerp`s.
fn cubic_point(p: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let a = lerp3(p[0], p[1], t);
    let b = lerp3(p[1], p[2], t);
    let c = lerp3(p[2], p[3], t);
    let d = lerp3(a, b, t);
    let e = lerp3(b, c, t);
    lerp3(d, e, t)
}

/// Cubic Bézier derivative: `3·` the quadratic De Casteljau over the three
/// adjacent control-point differences.
fn cubic_deriv(p: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let d0 = sub3(p[1], p[0]);
    let d1 = sub3(p[2], p[1]);
    let d2 = sub3(p[3], p[2]);
    let a = lerp3(d0, d1, t);
    let b = lerp3(d1, d2, t);
    let qd = lerp3(a, b, t);
    [qd[0] * 3.0, qd[1] * 3.0, qd[2] * 3.0]
}

/// Extracts the four control points of `v`-row `row` (`col` running along `u`).
fn bez_row(bez: &[[f32; 3]; 16], row: usize) -> [[f32; 3]; 4] {
    [
        bez[row * 4],
        bez[row * 4 + 1],
        bez[row * 4 + 2],
        bez[row * 4 + 3],
    ]
}

/// Collapses every `v`-row to a point by a cubic De Casteljau in `u`.
fn rows_in_u(bez: &[[f32; 3]; 16], u: f32) -> [[f32; 3]; 4] {
    [
        cubic_point(bez_row(bez, 0), u),
        cubic_point(bez_row(bez, 1), u),
        cubic_point(bez_row(bez, 2), u),
        cubic_point(bez_row(bez, 3), u),
    ]
}

/// Surface point `P(u, v)` on the Bézier net.
fn bez_point(bez: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    cubic_point(rows_in_u(bez, u), v)
}

/// Partial derivative `dP/du` on the Bézier net.
fn bez_partial_u(bez: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    let du = [
        cubic_deriv(bez_row(bez, 0), u),
        cubic_deriv(bez_row(bez, 1), u),
        cubic_deriv(bez_row(bez, 2), u),
        cubic_deriv(bez_row(bez, 3), u),
    ];
    cubic_point(du, v)
}

/// Partial derivative `dP/dv` on the Bézier net.
fn bez_partial_v(bez: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    cubic_deriv(rows_in_u(bez, u), v)
}

/// Unit surface normal with the reference four-step interior nudge: at each step
/// evaluate the two partials, cross them and accept the first non-degenerate
/// frame (`len2 > 0.0`); failing all four steps fall back to `[0, 0, 1]`.
/// Returns the normal plus the degeneracy flag (`1` on fallback).
fn bez_normal(bez: &[[f32; 3]; 16], u: f32, v: f32) -> ([f32; 3], u32) {
    for step in 0..4 {
        let eps = 1.0e-3 * step as f32;
        let uu = (u + eps).clamp(0.0, 1.0);
        let vv = (v + eps).clamp(0.0, 1.0);
        let du = bez_partial_u(bez, uu, vv);
        let dv = bez_partial_v(bez, uu, vv);
        let n = cross3(du, dv);
        let len2 = dot3(n, n);
        if len2 > 0.0 {
            let inv = 1.0 / len2.sqrt();
            return ([n[0] * inv, n[1] * inv, n[2] * inv], 0);
        }
    }
    ([0.0, 0.0, 1.0], 1)
}

/// Full pipeline oracle: convert the B-spline net to Bézier form, then read the
/// surface point and analytic normal, returning the point, normal and
/// degeneracy flag.
fn oracle(control: &[[f32; 3]; 16], u: f32, v: f32) -> ([f32; 3], [f32; 3], u32) {
    let bez = to_bezier(control);
    let point = bez_point(&bez, u, v);
    let (normal, degenerate) = bez_normal(&bez, u, v);
    (point, normal, degenerate)
}

/// Pins one `GPU` result against the host oracle: the discrete `degenerate`
/// flag exactly, and `point`/`normal` under the shared tolerance. The fallback
/// normal `[0, 0, 1]` is deterministic on both sides, so `normal` is always
/// compared componentwise.
fn check_one(idx: usize, got: &BsplinePatchResult, control: &[[f32; 3]; 16], u: f32, v: f32) {
    let (want_point, want_normal, want_degen) = oracle(control, u, v);

    assert_eq!(
        got.degenerate, want_degen,
        "query {idx}: degenerate flag disagrees (gpu {} vs cpu {})",
        got.degenerate, want_degen
    );
    for (axis, (g, w)) in got.point.iter().zip(want_point.iter()).enumerate() {
        assert!(
            close(*g, *w),
            "query {idx} point[{axis}]: gpu {g} vs cpu {w}"
        );
    }
    for (axis, (g, w)) in got.normal.iter().zip(want_normal.iter()).enumerate() {
        assert!(
            close(*g, *w),
            "query {idx} normal[{axis}]: gpu {g} vs cpu {w}"
        );
    }
    // A non-degenerate frame must yield a unit normal.
    if want_degen == 0 {
        let len = dot3(got.normal, got.normal).sqrt();
        assert!(
            close(len, 1.0),
            "query {idx}: gpu normal must be unit length, got {len}"
        );
    }
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuBsplinePatch, queries: &[BsplinePatchQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (result, query)) in got.iter().zip(queries.iter()).enumerate() {
        check_one(idx, result, &query.control, query.u, query.v);
    }
}

/// A planar net: `col` along `x`, `row` along `y`, flat in `z`. A uniformly
/// spaced planar net stays planar after the (affine-exact) conversion, so the
/// tangent frame is well defined and the normal is `±z`.
fn flat_net() -> [[f32; 3]; 16] {
    let mut control = [[0.0f32; 3]; 16];
    for (r, chunk) in control.chunks_mut(4).enumerate() {
        for (c, slot) in chunk.iter_mut().enumerate() {
            *slot = [c as f32 - 1.0, r as f32 - 1.0, 0.0];
        }
    }
    control
}

/// A curved net: the planar base with the interior control points pushed up in
/// `z`, so the surface bulges and the tangent frame stays non-degenerate.
fn curved_net() -> [[f32; 3]; 16] {
    let mut control = flat_net();
    // Interior points of the 4x4 net are rows/cols 1..=2.
    control[5][2] = 1.0;
    control[6][2] = 1.3;
    control[9][2] = 1.1;
    control[10][2] = 0.8;
    // A couple of edge points nudged to break symmetry.
    control[2][2] = 0.4;
    control[13][2] = -0.5;
    control
}

/// A fully collapsed net: all 16 control points are identical, so every Bézier
/// control point collapses, both partials vanish and the nudge loop falls back
/// to `[0, 0, 1]` with `degenerate = 1`.
fn collapsed_net() -> [[f32; 3]; 16] {
    [[0.7, -0.3, 0.2]; 16]
}

/// A colinear net: every control point lies on one line, so the two partials
/// are parallel, their cross product vanishes at every nudge step and the loop
/// falls back to `[0, 0, 1]` with `degenerate = 1`.
///
/// The line is deliberately parallel to the `x` axis: `y` and `z` are the same
/// constant for all 16 points, so both partials carry only an `x` component and
/// their cross product is **exactly** `[0, 0, 0]` in `f32` on every device
/// (each term is `0 * finite` or `0 - 0`). The `len2 > 0.0` guard therefore
/// fails identically on the `CPU` oracle and the `GPU`, and both sides reach the
/// `[0, 0, 1]` fallback deterministically. A slanted (non-axis) colinear line
/// leaves a rounding-noise cross product that the two devices round
/// differently, flipping that ill-conditioned guard and desynchronizing the
/// `degenerate` flag and the reported normal.
fn colinear_net() -> [[f32; 3]; 16] {
    let origin = [-2.0f32, 1.0, 0.5];
    let step = [0.3f32, 0.0, 0.0];
    let mut control = [[0.0f32; 3]; 16];
    for (i, slot) in control.iter_mut().enumerate() {
        *slot = add3(origin, scale3(step, i as f32));
    }
    control
}

/// A small bank of named, well-conditioned fixtures evaluated at interior
/// parameters.
fn fixture_queries() -> Vec<BsplinePatchQuery> {
    vec![
        BsplinePatchQuery::new(flat_net(), 0.5, 0.5),
        BsplinePatchQuery::new(flat_net(), 0.25, 0.75),
        BsplinePatchQuery::new(curved_net(), 0.4, 0.6),
        BsplinePatchQuery::new(curved_net(), 0.2, 0.3),
        BsplinePatchQuery::new(curved_net(), 0.8, 0.15),
        BsplinePatchQuery::new(collapsed_net(), 0.5, 0.5),
        BsplinePatchQuery::new(colinear_net(), 0.45, 0.55),
    ]
}

/// Advances a 64-bit LCG and returns a uniform `f32` in `[0, 1)` built from the
/// high mantissa bits. The host may use `u64`; the kernel never does.
fn next_unit(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Uniform `f32` in `[lo, hi)`.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * next_unit(state)
}

/// Screens a query away from the degeneracy cliff: the step-0 cross product's
/// squared length must sit well above the kernel's `len2 > 0.0` guard so a
/// last-place wobble never flips the branch.
fn well_conditioned(q: &BsplinePatchQuery) -> bool {
    let bez = to_bezier(&q.control);
    let du = bez_partial_u(&bez, q.u, q.v);
    let dv = bez_partial_v(&bez, q.u, q.v);
    let n = cross3(du, dv);
    dot3(n, n) > 1.0e-2
}

/// Builds one random, well-conditioned query: a planar base grid perturbed in
/// all three axes so the tangent frame stays non-degenerate, sampled well
/// inside `[0, 1]`.
fn random_query(state: &mut u64) -> BsplinePatchQuery {
    loop {
        let mut control = [[0.0f32; 3]; 16];
        for (r, chunk) in control.chunks_mut(4).enumerate() {
            for (c, slot) in chunk.iter_mut().enumerate() {
                *slot = [
                    c as f32 - 1.5 + uniform(state, -0.3, 0.3),
                    r as f32 - 1.5 + uniform(state, -0.3, 0.3),
                    uniform(state, -1.0, 1.0),
                ];
            }
        }
        let u = uniform(state, 0.1, 0.9);
        let v = uniform(state, 0.1, 0.9);
        let q = BsplinePatchQuery::new(control, u, v);
        if well_conditioned(&q) {
            return q;
        }
    }
}

#[test]
fn flat_planar_patch_normal_is_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplinePatch::new(&ctx);
    let q = BsplinePatchQuery::new(flat_net(), 0.5, 0.5);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &q.control, q.u, q.v);
    // A planar net has a usable (non-degenerate) tangent frame.
    assert_eq!(got[0].degenerate, 0, "a planar patch is non-degenerate");
    // The surface normal of a flat z = 0 net points along ±z.
    assert!(
        close(got[0].normal[0].abs(), 0.0) && close(got[0].normal[1].abs(), 0.0),
        "flat patch normal must be axis-aligned, got {:?}",
        got[0].normal
    );
    assert!(
        close(got[0].normal[2].abs(), 1.0),
        "flat patch normal z must be unit, got {}",
        got[0].normal[2]
    );
}

#[test]
fn curved_patch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplinePatch::new(&ctx);
    let q = BsplinePatchQuery::new(curved_net(), 0.4, 0.6);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &q.control, q.u, q.v);
    assert_eq!(got[0].degenerate, 0, "a curved patch is non-degenerate");
}

#[test]
fn collapsed_net_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplinePatch::new(&ctx);
    let q = BsplinePatchQuery::new(collapsed_net(), 0.5, 0.5);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    assert_eq!(
        got[0].degenerate, 1,
        "a collapsed control net must report degenerate"
    );
    // The reference fallback normal is [0, 0, 1] on both sides.
    assert!(
        close(got[0].normal[0], 0.0)
            && close(got[0].normal[1], 0.0)
            && close(got[0].normal[2], 1.0),
        "collapsed net must fall back to [0, 0, 1], got {:?}",
        got[0].normal
    );
    check_one(0, &got[0], &q.control, q.u, q.v);
}

#[test]
fn colinear_net_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplinePatch::new(&ctx);
    let q = BsplinePatchQuery::new(colinear_net(), 0.45, 0.55);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    assert_eq!(
        got[0].degenerate, 1,
        "a colinear control net has parallel partials and must report degenerate"
    );
    check_one(0, &got[0], &q.control, q.u, q.v);
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplinePatch::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplinePatch::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned nets pin the
    // reported point and normal across a wide span of control cages.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping bspline_patch parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBsplinePatch::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}
