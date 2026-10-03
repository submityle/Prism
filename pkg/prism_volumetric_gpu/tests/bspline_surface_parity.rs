//! Real-device parity for the uniform bicubic B-spline *surface* evaluation
//! twin: `GpuBsplineSurface` must reproduce the `CPU` closed form of the oracle
//! `prism_render_architecture::ray_scene::bspline_surface` —
//! `BsplineSurface::{point, normal}` — across a planar net, a domed net, an
//! axis-aligned collinear (degenerate) net, interior spans of a larger grid and
//! a randomized sweep compared query-for-query against a shared control net.
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
//! operation-for-operation: `locate` scales the global parameter by its span
//! count, takes the `floor` for the integer span (clamped to the last span) and
//! the remainder for the local parameter; `patch_at` extracts the overlapping
//! `4x4` window `control[(span_v + wr) * cols + (span_u + wc)]`; the window is
//! converted from the uniform B-spline basis into the equivalent cubic Bézier
//! net — `span_to_bezier` along the four rows then the four columns, a purely
//! linear conversion; the surface point and partial derivatives are evaluated
//! by De Casteljau (nested `lerp`s and the `3x` quadratic derivative); and the
//! normal is the normalized cross product `partial_u × partial_v`, nudged a few
//! interior steps when the tangent frame collapses. Because the reference and
//! this oracle are both scalar `f32`, a `GPU == oracle` pass is direct evidence
//! the ported kernel computes the same surface the reference does.
//!
//! # Parity criterion
//!
//! The discrete `degenerate` flag is asserted exactly. The continuous `point`
//! and `normal` thread through nested interpolations, a basis conversion, a
//! cross product and one normalize `sqrt`, so a `GPU` result may land a few
//! units in the last place from the scalar oracle; each component is asserted
//! within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a `rel_diff` floor of
//! `1e-6` so a near-zero expected component does not inflate the relative error.
//!
//! # Conditioning
//!
//! Fixtures and the randomized sweep reject-sample away from the one branch
//! cliff — the cross product near zero (a collapsed tangent frame) — so a
//! last-place wobble never flips which side of the degeneracy guard the `CPU`
//! and `GPU` land on. The one deliberately degenerate fixture is axis-aligned
//! collinear, so both partials are pure `x` vectors and the cross product is
//! exactly zero on both sides. Random samples also keep `u` and `v` well inside
//! `[0, 1]`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::bspline_surface`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::bspline_surface::{
    BsplineSurfaceQuery, BsplineSurfaceResult, GpuBsplineSurface,
};
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

/// Squared-length conditioning threshold used by the fixtures and sweep to stay
/// well clear of the kernel's strict `len2 > 0.0` degeneracy guard, so a
/// last-place wobble never flips the branch between `CPU` and `GPU`.
const COND_FLOOR: f32 = 1.0e-3;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Component-wise sum `a + b`.
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Component-wise difference `a - b`.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scales `a` by the scalar `s`.
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
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

/// Locates the cubic span and local parameter for a global parameter `t` over
/// `spans` spans: clamp, scale by the span count, `floor` for the span (clamped
/// to the last span) and the remainder for the local parameter.
fn locate(t: f32, spans: u32) -> (u32, f32) {
    let clamped = t.clamp(0.0, 1.0);
    let scaled = clamped * spans as f32;
    let last = spans - 1;
    let index = (scaled.floor() as u32).min(last);
    let local = scaled - index as f32;
    (index, local)
}

/// Converts one uniform cubic B-spline span into the four cubic Bézier control
/// points of its central segment, operation-for-operation with the reference.
fn span_to_bezier(c0: [f32; 3], c1: [f32; 3], c2: [f32; 3], c3: [f32; 3]) -> [[f32; 3]; 4] {
    let sixth = 1.0 / 6.0;
    let third = 1.0 / 3.0;
    let b0 = scale3(add3(add3(c0, scale3(c1, 4.0)), c2), sixth);
    let b1 = scale3(add3(scale3(c1, 2.0), c2), third);
    let b2 = scale3(add3(c1, scale3(c2, 2.0)), third);
    let b3 = scale3(add3(add3(c1, scale3(c2, 4.0)), c3), sixth);
    [b0, b1, b2, b3]
}

/// Converts a `4x4` row-major B-spline window into the equivalent Bézier net,
/// `span_to_bezier` along the four `u` rows then the four `v` columns.
fn to_bezier(win: &[[f32; 3]; 16]) -> [[f32; 3]; 16] {
    let mut tmp = [[0.0f32; 3]; 16];
    for row in 0..4 {
        let base = row * 4;
        let span = span_to_bezier(win[base], win[base + 1], win[base + 2], win[base + 3]);
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
    let q = lerp3(a, b, t);
    [q[0] * 3.0, q[1] * 3.0, q[2] * 3.0]
}

/// The four control points of `v`-row `row` of a `4x4` row-major Bézier net.
fn bez_row(bez: &[[f32; 3]; 16], row: usize) -> [[f32; 3]; 4] {
    let b = row * 4;
    [bez[b], bez[b + 1], bez[b + 2], bez[b + 3]]
}

/// Bézier surface point `P(u, v)` from a `4x4` row-major Bézier net.
fn bez_point(bez: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    let column = [
        cubic_point(bez_row(bez, 0), u),
        cubic_point(bez_row(bez, 1), u),
        cubic_point(bez_row(bez, 2), u),
        cubic_point(bez_row(bez, 3), u),
    ];
    cubic_point(column, v)
}

/// Partial derivative `dP/du` from a `4x4` row-major Bézier net.
fn bez_partial_u(bez: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    let column = [
        cubic_deriv(bez_row(bez, 0), u),
        cubic_deriv(bez_row(bez, 1), u),
        cubic_deriv(bez_row(bez, 2), u),
        cubic_deriv(bez_row(bez, 3), u),
    ];
    cubic_point(column, v)
}

/// Partial derivative `dP/dv` from a `4x4` row-major Bézier net.
fn bez_partial_v(bez: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    let column = [
        cubic_point(bez_row(bez, 0), u),
        cubic_point(bez_row(bez, 1), u),
        cubic_point(bez_row(bez, 2), u),
        cubic_point(bez_row(bez, 3), u),
    ];
    cubic_deriv(column, v)
}

/// Extracts the overlapping `4x4` B-spline window for the span at
/// `(span_u, span_v)` from the row-major control net.
fn window(control: &[[f32; 3]], cols: u32, span_u: u32, span_v: u32) -> [[f32; 3]; 16] {
    let mut win = [[0.0f32; 3]; 16];
    for wr in 0..4u32 {
        for wc in 0..4u32 {
            let slot = ((span_v + wr) * cols + (span_u + wc)) as usize;
            win[(wr * 4 + wc) as usize] = control[slot];
        }
    }
    win
}

/// Unit normal and degeneracy flag from a Bézier net at `(u, v)`, replicating
/// the reference nudge loop: step `0..4`, `eps = 1e-3 * step`, clamp the sample,
/// take the normalized cross `partial_u × partial_v` on the first non-degenerate
/// frame, else the deterministic fallback `[0, 0, 1]` with `degenerate = 1`.
fn oracle_normal(bez: &[[f32; 3]; 16], u: f32, v: f32) -> ([f32; 3], u32) {
    const FALLBACK: [f32; 3] = [0.0, 0.0, 1.0];
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
    (FALLBACK, 1)
}

/// Full host oracle for one `(u, v)` sample: locate spans, extract and convert
/// the window, evaluate point and normal.
fn oracle(control: &[[f32; 3]], rows: u32, cols: u32, u: f32, v: f32) -> ([f32; 3], [f32; 3], u32) {
    let u_spans = cols - 3;
    let v_spans = rows - 3;
    let (span_u, local_u) = locate(u, u_spans);
    let (span_v, local_v) = locate(v, v_spans);
    let win = window(control, cols, span_u, span_v);
    let bez = to_bezier(&win);
    let point = bez_point(&bez, local_u, local_v);
    let (normal, degen) = oracle_normal(&bez, local_u, local_v);
    (point, normal, degen)
}

/// Squared cross length of the step-`0` tangent frame; a query is well
/// conditioned when this sits comfortably above the degeneracy guard.
fn conditioning(control: &[[f32; 3]], rows: u32, cols: u32, u: f32, v: f32) -> f32 {
    let (span_u, local_u) = locate(u, cols - 3);
    let (span_v, local_v) = locate(v, rows - 3);
    let win = window(control, cols, span_u, span_v);
    let bez = to_bezier(&win);
    let du = bez_partial_u(&bez, local_u, local_v);
    let dv = bez_partial_v(&bez, local_u, local_v);
    let n = cross3(du, dv);
    dot3(n, n)
}

/// Whether a sample sits clear of the one branch cliff — the cross product near
/// zero — so the `CPU` and `GPU` cannot pick different sides of the guard.
fn well_conditioned(control: &[[f32; 3]], rows: u32, cols: u32, u: f32, v: f32) -> bool {
    conditioning(control, rows, cols, u, v) > COND_FLOOR
}

/// Pins one `GPU` result against the host oracle: the discrete `degenerate`
/// flag exactly, `point` and `normal` under the shared tolerance (both the
/// reference and twin emit the `[0, 0, 1]` fallback when degenerate, so the
/// normal is always compared), and unit length when non-degenerate.
fn check_one(
    idx: usize,
    got: &BsplineSurfaceResult,
    control: &[[f32; 3]],
    rows: u32,
    cols: u32,
    u: f32,
    v: f32,
) {
    let (want_point, want_normal, want_degen) = oracle(control, rows, cols, u, v);
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
    for (axis, (g, w)) in got.normal.iter().zip(want_normal.iter()).enumerate() {
        assert!(
            close(*g, *w, SD_ABS, SD_REL),
            "query {idx} normal[{axis}]: gpu {g} vs cpu {w}"
        );
    }
    if want_degen == 0 {
        let len = dot3(got.normal, got.normal).sqrt();
        assert!(
            close(len, 1.0, SD_ABS, SD_REL),
            "query {idx}: gpu normal must be unit length, got {len}"
        );
    }
}

/// Dispatches every query against the shared control net and pins each result.
fn check(
    ctx: &GpuContext,
    gpu: &GpuBsplineSurface,
    control: &[[f32; 3]],
    rows: u32,
    cols: u32,
    queries: &[BsplineSurfaceQuery],
) {
    let got = gpu.evaluate(ctx, control, rows, cols, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        check_one(idx, result, control, rows, cols, q.u, q.v);
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

/// A planar `z = 0` control net: a regular `rows x cols` grid in `x`/`y`. Its
/// surface is planar, so the normal is `±z` and the tangent frame is well clear
/// of the degeneracy guard everywhere.
fn planar_grid(rows: u32, cols: u32) -> Vec<[f32; 3]> {
    let mut net = Vec::with_capacity((rows * cols) as usize);
    for r in 0..rows {
        for c in 0..cols {
            net.push([c as f32, r as f32, 0.0]);
        }
    }
    net
}

/// A domed control net: the planar grid with the interior handles lifted in `z`
/// so the surface is genuinely curved while staying well conditioned.
fn domed_grid(rows: u32, cols: u32) -> Vec<[f32; 3]> {
    let mut net = planar_grid(rows, cols);
    for r in 1..rows - 1 {
        for c in 1..cols - 1 {
            let idx = (r * cols + c) as usize;
            net[idx][2] = 1.0 + 0.25 * ((r + c) as f32);
        }
    }
    net
}

/// An axis-aligned collinear control net: every handle lies on the `x` axis, so
/// both partials are pure `x` vectors and their cross product is exactly zero on
/// both `CPU` and `GPU`, forcing `degenerate = 1`.
fn collinear_grid(rows: u32, cols: u32) -> Vec<[f32; 3]> {
    let mut net = Vec::with_capacity((rows * cols) as usize);
    for r in 0..rows {
        for c in 0..cols {
            net.push([(r * cols + c) as f32, 0.0, 0.0]);
        }
    }
    net
}

/// A random, non-degenerate control net: the planar base perturbed in all three
/// axes by small offsets, keeping the `x`/`y` lattice dominant so the tangent
/// frame stays well conditioned.
fn random_net(state: &mut u64, rows: u32, cols: u32) -> Vec<[f32; 3]> {
    let mut net = Vec::with_capacity((rows * cols) as usize);
    for r in 0..rows {
        for c in 0..cols {
            net.push([
                c as f32 + uniform(state, -0.3, 0.3),
                r as f32 + uniform(state, -0.3, 0.3),
                uniform(state, -1.0, 1.0),
            ]);
        }
    }
    net
}

/// Interior sample parameters shared by the well-conditioned fixtures; kept away
/// from the exact `0`/`1` edges so a near-degenerate corner frame never appears.
fn interior_samples() -> Vec<BsplineSurfaceQuery> {
    vec![
        BsplineSurfaceQuery::new(0.5, 0.5),
        BsplineSurfaceQuery::new(0.25, 0.75),
        BsplineSurfaceQuery::new(0.1, 0.9),
        BsplineSurfaceQuery::new(0.8, 0.3),
        BsplineSurfaceQuery::new(0.33, 0.66),
        BsplineSurfaceQuery::new(0.9, 0.1),
    ]
}

#[test]
fn planar_surface_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplineSurface::new(&ctx);
    let net = planar_grid(4, 4);
    let q = BsplineSurfaceQuery::new(0.5, 0.5);
    let got = gpu.evaluate(&ctx, &net, 4, 4, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &net, 4, 4, q.u, q.v);
    assert_eq!(got[0].degenerate, 0, "a planar surface is non-degenerate");
}

#[test]
fn domed_surface_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplineSurface::new(&ctx);
    let net = domed_grid(5, 5);
    check(&ctx, &gpu, &net, 5, 5, &interior_samples());
}

#[test]
fn collinear_surface_reports_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplineSurface::new(&ctx);
    let net = collinear_grid(4, 4);
    let q = BsplineSurfaceQuery::new(0.5, 0.5);
    let got = gpu.evaluate(&ctx, &net, 4, 4, &[q]);
    assert_eq!(got.len(), 1);
    assert_eq!(
        got[0].degenerate, 1,
        "a collinear control net must report degenerate"
    );
    // The deterministic fallback normal is [0, 0, 1].
    assert!(
        got[0].normal[0].abs() <= SD_ABS,
        "fallback normal x must be 0"
    );
    assert!(
        got[0].normal[1].abs() <= SD_ABS,
        "fallback normal y must be 0"
    );
    assert!(
        close(got[0].normal[2], 1.0, SD_ABS, SD_REL),
        "fallback normal z must be 1"
    );
    check_one(0, &got[0], &net, 4, 4, q.u, q.v);
}

#[test]
fn interior_span_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplineSurface::new(&ctx);
    // 6x5 grid: u_spans = 2, v_spans = 3, so these samples exercise span > 0 on
    // both axes.
    let net = domed_grid(6, 5);
    let queries = vec![
        BsplineSurfaceQuery::new(0.05, 0.95),
        BsplineSurfaceQuery::new(0.5, 0.5),
        BsplineSurfaceQuery::new(0.95, 0.05),
        BsplineSurfaceQuery::new(0.7, 0.2),
        BsplineSurfaceQuery::new(0.2, 0.7),
    ];
    check(&ctx, &gpu, &net, 6, 5, &queries);
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplineSurface::new(&ctx);
    check(&ctx, &gpu, &planar_grid(4, 4), 4, 4, &interior_samples());
    check(&ctx, &gpu, &domed_grid(5, 5), 5, 5, &interior_samples());
    check(&ctx, &gpu, &domed_grid(6, 5), 6, 5, &interior_samples());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBsplineSurface::new(&ctx);
    let mut state = 0x5eed_face_cafe_1234_u64;
    // Several control-net shapes, each with a batch of random, well-conditioned
    // samples: ~516 queries total across varied span counts.
    let shapes = [(4u32, 4u32), (5, 4), (4, 6), (6, 5), (5, 7), (7, 6)];
    for &(rows, cols) in &shapes {
        let net = random_net(&mut state, rows, cols);
        let mut queries = Vec::new();
        while queries.len() < 86 {
            let u = uniform(&mut state, 0.05, 0.95);
            let v = uniform(&mut state, 0.05, 0.95);
            if well_conditioned(&net, rows, cols, u, v) {
                queries.push(BsplineSurfaceQuery::new(u, v));
            }
        }
        check(&ctx, &gpu, &net, rows, cols, &queries);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping bspline_surface parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBsplineSurface::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &planar_grid(4, 4), 4, 4, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}
