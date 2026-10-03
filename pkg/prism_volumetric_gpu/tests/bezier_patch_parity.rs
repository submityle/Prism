//! Real-device parity for the analytic *bicubic Bézier patch* evaluation twin:
//! `GpuBezierPatch` must reproduce the `CPU` closed form of the oracle
//! `prism_render_architecture::ray_scene::bezier_patch` —
//! `BezierPatch::{point, partial_u, partial_v, normal}` — across a flat patch,
//! curved patches, a degenerate (collapsed) control net, corner interpolation
//! and a randomized sweep compared query-for-query.
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
//! operation-for-operation: a cubic De Casteljau (three nested `lerp`s) per
//! `v`-row in `u` collapsed by a cubic De Casteljau in `v` for the surface
//! point; the row `u`-derivatives (`3·` a quadratic De Casteljau over adjacent
//! control-point differences) blended in `v` for `partial_u`; the row points
//! differentiated in `v` for `partial_v`; and the normalized cross product
//! `partial_u × partial_v` — with a squared-length degeneracy guard — for the
//! normal. Because the reference and this oracle are both scalar `f32`, a
//! `GPU == oracle` pass is direct evidence the ported kernel computes the same
//! surface the reference does.
//!
//! # Parity criterion
//!
//! The discrete `degenerate` flag is asserted exactly. The continuous `point`,
//! `partial_u`, `partial_v` and (when `degenerate` is `0`) `normal` thread
//! through nested interpolations, a cross product and one normalize `sqrt`, so
//! a `GPU` result may land a few units in the last place from the scalar
//! oracle; each component is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, with a `rel_diff` floor of `1e-6` so a near-zero
//! expected component does not inflate the relative error.
//!
//! # Conditioning
//!
//! Fixtures and the randomized sweep reject-sample away from the one branch
//! cliff — the cross product near zero (a collapsed tangent frame) — so a
//! last-place wobble never flips which side of the degeneracy guard the `CPU`
//! and `GPU` land on. Random samples also keep `u` and `v` well inside `[0, 1]`
//! so an edge-degenerate tangent never appears.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::bezier_patch`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::bezier_patch::{BezierPatchQuery, BezierPatchResult, GpuBezierPatch};
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

/// Squared-length guard separating a usable tangent frame from the degenerate
/// cross-product branch, matching the kernel's `N_EPS`. The oracle uses the same
/// threshold so the two never disagree on which branch is taken; the sweep then
/// keeps real queries far above it.
const N_EPS: f32 = 1.0e-12;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
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

/// Linear interpolation `a + t * (b - a)`, component-wise.
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
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
fn row(control: &[[f32; 3]; 16], row: usize) -> [[f32; 3]; 4] {
    [
        control[row * 4],
        control[row * 4 + 1],
        control[row * 4 + 2],
        control[row * 4 + 3],
    ]
}

/// Collapses every `v`-row to a point by a cubic De Casteljau in `u`.
fn rows_in_u(control: &[[f32; 3]; 16], u: f32) -> [[f32; 3]; 4] {
    [
        cubic_point(row(control, 0), u),
        cubic_point(row(control, 1), u),
        cubic_point(row(control, 2), u),
        cubic_point(row(control, 3), u),
    ]
}

/// Surface point `P(u, v)`.
fn oracle_point(control: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    cubic_point(rows_in_u(control, u), v)
}

/// Partial derivative `dP/du`.
fn oracle_partial_u(control: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    let du = [
        cubic_deriv(row(control, 0), u),
        cubic_deriv(row(control, 1), u),
        cubic_deriv(row(control, 2), u),
        cubic_deriv(row(control, 3), u),
    ];
    cubic_point(du, v)
}

/// Partial derivative `dP/dv`.
fn oracle_partial_v(control: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    cubic_deriv(rows_in_u(control, u), v)
}

/// Unit normal `cross(partial_u, partial_v)` and the degeneracy flag, using the
/// same squared-length guard as the kernel.
fn oracle_normal(control: &[[f32; 3]; 16], u: f32, v: f32) -> ([f32; 3], u32) {
    let pu = oracle_partial_u(control, u, v);
    let pv = oracle_partial_v(control, u, v);
    let n = cross3(pu, pv);
    let len2 = dot3(n, n);
    if len2 <= N_EPS {
        ([0.0, 0.0, 0.0], 1)
    } else {
        let inv = 1.0 / len2.sqrt();
        ([n[0] * inv, n[1] * inv, n[2] * inv], 0)
    }
}

/// Pins one `GPU` result against the host oracle: the discrete `degenerate`
/// flag exactly, `point`/`partial_u`/`partial_v` under the shared tolerance,
/// and `normal` only when the frame is non-degenerate (zeroed otherwise).
fn check_one(idx: usize, got: &BezierPatchResult, control: &[[f32; 3]; 16], u: f32, v: f32) {
    let want_point = oracle_point(control, u, v);
    let want_pu = oracle_partial_u(control, u, v);
    let want_pv = oracle_partial_v(control, u, v);
    let (want_normal, want_degen) = oracle_normal(control, u, v);

    assert_eq!(
        got.degenerate, want_degen,
        "query {idx}: degenerate flag disagrees (gpu {} vs cpu {})",
        got.degenerate, want_degen
    );
    for (axis, (g, w)) in got.point.iter().zip(want_point.iter()).enumerate() {
        assert!(
            close(*g, *w, SD_ABS, SD_REL),
            "query {idx} point[{axis}]: gpu {g} vs cpu {w}"
        );
    }
    for (axis, (g, w)) in got.partial_u.iter().zip(want_pu.iter()).enumerate() {
        assert!(
            close(*g, *w, SD_ABS, SD_REL),
            "query {idx} partial_u[{axis}]: gpu {g} vs cpu {w}"
        );
    }
    for (axis, (g, w)) in got.partial_v.iter().zip(want_pv.iter()).enumerate() {
        assert!(
            close(*g, *w, SD_ABS, SD_REL),
            "query {idx} partial_v[{axis}]: gpu {g} vs cpu {w}"
        );
    }
    if want_degen == 0 {
        for (axis, (g, w)) in got.normal.iter().zip(want_normal.iter()).enumerate() {
            assert!(
                close(*g, *w, SD_ABS, SD_REL),
                "query {idx} normal[{axis}]: gpu {g} vs cpu {w}"
            );
        }
        let len = dot3(got.normal, got.normal).sqrt();
        assert!(
            close(len, 1.0, SD_ABS, SD_REL),
            "query {idx}: gpu normal must be unit length, got {len}"
        );
    } else {
        for (axis, g) in got.normal.iter().enumerate() {
            assert!(
                g.abs() <= SD_ABS,
                "query {idx} normal[{axis}]: degenerate frame must zero the normal, got {g}"
            );
        }
    }
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuBezierPatch, queries: &[BezierPatchQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        check_one(idx, result, &q.control, q.u, q.v);
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

/// A flat planar control net on the `z = 0` plane: a regular `4x4` grid in
/// `x`/`y`. Its surface is planar, so the normal is `±z` and well conditioned.
fn flat_patch() -> [[f32; 3]; 16] {
    let mut control = [[0.0_f32; 3]; 16];
    for r in 0..4 {
        for c in 0..4 {
            control[r * 4 + c] = [c as f32, r as f32, 0.0];
        }
    }
    control
}

/// A curved control net: the flat grid with the interior points lifted in `z`
/// so the tangent frame is well clear of the degeneracy guard.
fn curved_patch() -> [[f32; 3]; 16] {
    let mut control = flat_patch();
    control[5][2] = 1.5;
    control[6][2] = 1.25;
    control[9][2] = 1.75;
    control[10][2] = 0.75;
    control[0][2] = 0.25;
    control[15][2] = -0.5;
    control
}

/// A fully degenerate control net: every control point coincides, so the
/// surface is a single point and both partials vanish — the cross product
/// collapses and the twin reports `degenerate = 1`.
fn degenerate_patch() -> [[f32; 3]; 16] {
    [[1.0, -2.0, 0.5]; 16]
}

/// The named parity fixtures, each paired with a sample `(u, v)`.
fn fixture_queries() -> Vec<BezierPatchQuery> {
    vec![
        BezierPatchQuery::new(flat_patch(), 0.5, 0.5),
        BezierPatchQuery::new(curved_patch(), 0.5, 0.5),
        BezierPatchQuery::new(curved_patch(), 0.25, 0.75),
        BezierPatchQuery::new(curved_patch(), 0.8, 0.3),
        BezierPatchQuery::new(flat_patch(), 0.2, 0.9),
    ]
}

/// Whether a random query sits clear of the one branch cliff — the cross
/// product near zero — so the `CPU` and `GPU` cannot pick different sides of the
/// degeneracy guard.
fn well_conditioned(q: &BezierPatchQuery) -> bool {
    let pu = oracle_partial_u(&q.control, q.u, q.v);
    let pv = oracle_partial_v(&q.control, q.u, q.v);
    let n = cross3(pu, pv);
    // Keep the squared cross length comfortably above the 1e-12 guard.
    dot3(n, n) > 1.0e-3
}

/// Builds one random, well-conditioned query: a flat base grid perturbed in all
/// three axes so the tangent frame stays non-degenerate, sampled well inside
/// `[0, 1]`.
fn random_query(state: &mut u64) -> BezierPatchQuery {
    loop {
        let mut control = [[0.0_f32; 3]; 16];
        for r in 0..4 {
            for c in 0..4 {
                control[r * 4 + c] = [
                    c as f32 + uniform(state, -0.3, 0.3),
                    r as f32 + uniform(state, -0.3, 0.3),
                    uniform(state, -1.0, 1.0),
                ];
            }
        }
        let u = uniform(state, 0.05, 0.95);
        let v = uniform(state, 0.05, 0.95);
        let q = BezierPatchQuery::new(control, u, v);
        if well_conditioned(&q) {
            return q;
        }
    }
}

#[test]
fn flat_patch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBezierPatch::new(&ctx);
    let q = BezierPatchQuery::new(flat_patch(), 0.5, 0.5);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &q.control, q.u, q.v);
    // A planar net has a usable (non-degenerate) tangent frame.
    assert_eq!(got[0].degenerate, 0, "a planar patch is non-degenerate");
}

#[test]
fn curved_patch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBezierPatch::new(&ctx);
    let q = BezierPatchQuery::new(curved_patch(), 0.4, 0.6);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &q.control, q.u, q.v);
    assert_eq!(got[0].degenerate, 0, "a curved patch is non-degenerate");
}

#[test]
fn degenerate_patch_reports_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBezierPatch::new(&ctx);
    let q = BezierPatchQuery::new(degenerate_patch(), 0.5, 0.5);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    assert_eq!(
        got[0].degenerate, 1,
        "a collapsed control net must report degenerate"
    );
    for (axis, g) in got[0].normal.iter().enumerate() {
        assert!(
            g.abs() <= SD_ABS,
            "degenerate normal[{axis}] must be zero, got {g}"
        );
    }
    check_one(0, &got[0], &q.control, q.u, q.v);
}

#[test]
fn corner_interpolation_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBezierPatch::new(&ctx);
    // A Bézier patch interpolates its four corner control points. Sample the
    // corners and confirm the surface point matches the net corners exactly
    // within tolerance (the normal may be degenerate at a corner, so compare it
    // only through the shared oracle check).
    let control = curved_patch();
    let corners = [(0.0_f32, 0.0_f32), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
    let queries: Vec<BezierPatchQuery> = corners
        .iter()
        .map(|&(u, v)| BezierPatchQuery::new(control, u, v))
        .collect();
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 4);
    let net_corners = [control[0], control[3], control[12], control[15]];
    for (idx, (result, net)) in got.iter().zip(net_corners.iter()).enumerate() {
        check_one(idx, result, &control, queries[idx].u, queries[idx].v);
        for (axis, (g, w)) in result.point.iter().zip(net.iter()).enumerate() {
            assert!(
                close(*g, *w, SD_ABS, SD_REL),
                "corner {idx} point[{axis}]: gpu {g} vs net {w}"
            );
        }
    }
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBezierPatch::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBezierPatch::new(&ctx);
    let mut state = 0x5eed_face_cafe_1234_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned patches pin the
    // reported point, partials and normal across a wide span of control nets.
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
        eprintln!("skipping bezier_patch parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBezierPatch::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}
