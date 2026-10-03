//! Real-device parity for the single-patch bicubic Bézier-surface twin:
//! [`GpuBezierSurface`](prism_volumetric_gpu::bezier_surface::GpuBezierSurface)
//! must reproduce the surface point, unit normal and degenerate classification
//! of the host-side independent reimplementation
//! [`eval_bezier_surface`](prism_volumetric_gpu::bezier_surface::eval_bezier_surface)
//! across a planar patch, a domed patch, the interpolated corner control
//! points, a fully collapsed degenerate net and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected result comes from the module's own host-side independent
//! reimplementation; the twin never imports the golden crate, so both the
//! kernel and the oracle are faithful, independent ports of the same closed
//! form. A `GPU` parity pass is therefore direct evidence the ported kernel
//! runs the same bicubic De Casteljau recursion and classifies the same
//! degenerate normal.
//!
//! # Parity criterion
//!
//! The surface point and normal are *continuous* quantities threaded through
//! nested `lerp`s, a `cross`, a `sqrt` and divisions, so every continuous
//! assertion compares with an absolute-or-relative tolerance (`abs <= 1e-4 ||
//! rel <= 1e-3`, relative floor `1e-6`). The discrete `degenerate` flag is
//! pinned exactly. The genuine degeneracy — a collapsed tangent frame whose
//! step-0 cross product crosses zero — is kept off its threshold by the
//! fixtures and the rejection-sampled sweep so the two sides agree on the
//! discrete classification.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::bezier_surface`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::bezier_surface::{
    eval_bezier_surface, BezierSurfaceQuery, BezierSurfaceResult, GpuBezierSurface,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous point/normal comparison.
const EPS: f32 = 1e-4;
/// Relative tolerance for the continuous point/normal comparison.
const REL: f32 = 1e-3;
/// Relative-tolerance floor, so near-zero magnitudes do not inflate the ratio.
const REL_FLOOR: f32 = 1e-6;

/// Smallest accepted step-0 tangent cross-product length, keeping the
/// degenerate-normal branch off its threshold in the randomized sweep so the
/// `degenerate` flag cannot flip between host and device.
const CROSS_MARGIN: f32 = 0.05;
/// Interior margin on the global `(u, v)` parameters, keeping the sweep off the
/// `clamp` boundaries where the `eps` step-search saturates.
const UV_MARGIN: f32 = 0.05;

/// Returns whether `a` and `b` agree within the absolute-or-relative tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL
}

/// Dot product of two 3-vectors, used only by the conditioning check.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3-vectors, used only by the conditioning check.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Component-wise linear interpolation `a + (b - a) * t`.
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Cubic Bézier point at `t` over four control points.
fn cubic_point(p: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let a = lerp3(p[0], p[1], t);
    let b = lerp3(p[1], p[2], t);
    let c = lerp3(p[2], p[3], t);
    let d = lerp3(a, b, t);
    let e = lerp3(b, c, t);
    lerp3(d, e, t)
}

/// Cubic Bézier derivative at `t` over four control points.
fn cubic_deriv(p: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let d0 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let d1 = [p[2][0] - p[1][0], p[2][1] - p[1][1], p[2][2] - p[1][2]];
    let d2 = [p[3][0] - p[2][0], p[3][1] - p[2][1], p[3][2] - p[2][2]];
    let a = lerp3(d0, d1, t);
    let b = lerp3(d1, d2, t);
    let q = lerp3(a, b, t);
    [q[0] * 3.0, q[1] * 3.0, q[2] * 3.0]
}

/// Returns the four control points of row `r` of the single `4 × 4` net.
fn row(control: &[[f32; 3]; 16], r: usize) -> [[f32; 3]; 4] {
    [
        control[r * 4],
        control[r * 4 + 1],
        control[r * 4 + 2],
        control[r * 4 + 3],
    ]
}

/// Evaluates `∂P/∂u` at local `(u, v)`.
fn partial_u(control: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    let column = [
        cubic_deriv(row(control, 0), u),
        cubic_deriv(row(control, 1), u),
        cubic_deriv(row(control, 2), u),
        cubic_deriv(row(control, 3), u),
    ];
    cubic_point(column, v)
}

/// Evaluates `∂P/∂v` at local `(u, v)`.
fn partial_v(control: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    let column = [
        cubic_point(row(control, 0), u),
        cubic_point(row(control, 1), u),
        cubic_point(row(control, 2), u),
        cubic_point(row(control, 3), u),
    ];
    cubic_deriv(column, v)
}

/// Computes the expected result from the host-side reimplementation.
fn oracle(q: &BezierSurfaceQuery) -> ([f32; 3], [f32; 3], bool) {
    eval_bezier_surface(&q.control, q.u, q.v)
}

/// Dispatches every query and pins each `GPU` result against the oracle: the
/// `degenerate` flag matches exactly, and the point and normal match within
/// tolerance.
fn check(ctx: &GpuContext, gpu: &GpuBezierSurface, queries: &[BezierSurfaceQuery]) {
    let got: Vec<BezierSurfaceResult> = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let (point, normal, degenerate) = oracle(q);
        assert_eq!(
            result.degenerate, degenerate,
            "query {idx}: degenerate flag mismatch"
        );
        assert!(
            close(result.point[0], point[0])
                && close(result.point[1], point[1])
                && close(result.point[2], point[2]),
            "query {idx}: point gpu {:?} vs cpu {:?}",
            result.point,
            point
        );
        assert!(
            close(result.normal[0], normal[0])
                && close(result.normal[1], normal[1])
                && close(result.normal[2], normal[2]),
            "query {idx}: normal gpu {:?} vs cpu {:?}",
            result.normal,
            normal
        );
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

/// Draws an `f32` in `[lo, hi]` at milli resolution from `state`, using only
/// integer arithmetic so no transcendental method appears.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = ((hi - lo) * 1000.0) as u32;
    let step = lcg(state) % (span + 1);
    lo + step as f32 / 1000.0
}

/// Rejects queries that sit on the genuine degeneracy of the normal solve, so
/// the discrete `degenerate` flag cannot flip between host and device: the
/// step-0 tangent cross product must be decisively non-zero and the global
/// `(u, v)` must stay off the `clamp` boundaries. It mirrors the kernel's
/// step-0 tangent frame so the margin pins the exact threshold it branches on.
fn well_conditioned(q: &BezierSurfaceQuery) -> bool {
    if q.u < UV_MARGIN || q.u > 1.0 - UV_MARGIN || q.v < UV_MARGIN || q.v > 1.0 - UV_MARGIN {
        return false;
    }
    let du = partial_u(&q.control, q.u, q.v);
    let dv = partial_v(&q.control, q.u, q.v);
    let n = cross3(du, dv);
    dot3(n, n).sqrt() >= CROSS_MARGIN
}

/// Builds a planar `4 × 4` net in the `xy` plane: `control[r*4 + c] =
/// [c / 3, r / 3, 0]`, whose tangents are the two axes and whose normal is
/// `+z` everywhere.
fn flat_net() -> [[f32; 3]; 16] {
    let mut control = [[0.0f32; 3]; 16];
    for r in 0..4 {
        for c in 0..4 {
            control[r * 4 + c] = [c as f32 / 3.0, r as f32 / 3.0, 0.0];
        }
    }
    control
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        #[expect(
            clippy::print_stderr,
            reason = "a skipped device test should report why on hosts without a GPU"
        )]
        {
            eprintln!("skipping bezier_surface parity: no wgpu adapter on this host");
        }
        return;
    };
    let gpu = GpuBezierSurface::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn flat_patch_has_axis_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBezierSurface::new(&ctx);
    // A planar net lies in the z = 0 plane; its analytic normal is +z and the
    // tangent frame never collapses, so the patch is never degenerate.
    let q = BezierSurfaceQuery::new(flat_net(), 0.5, 0.5);
    assert!(well_conditioned(&q), "fixture must stay off the threshold");
    let (point, normal, degenerate) = oracle(&q);
    assert!(!degenerate, "a planar net has a well-defined normal");
    assert!(close(point[2], 0.0), "the planar patch stays in z = 0");
    assert!(
        close(normal[2].abs(), 1.0),
        "the planar patch normal is the z axis"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn domed_patch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBezierSurface::new(&ctx);
    // Lift the four interior control points off the plane into a smooth dome, so
    // the surface curves and the normal tilts away from the z axis at the edges.
    let mut control = flat_net();
    for r in 1..=2 {
        for c in 1..=2 {
            control[r * 4 + c][2] = 0.6;
        }
    }
    let center = BezierSurfaceQuery::new(control, 0.5, 0.5);
    let off_center = BezierSurfaceQuery::new(control, 0.3, 0.7);
    assert!(
        well_conditioned(&center) && well_conditioned(&off_center),
        "dome fixtures must stay off the threshold"
    );
    let (_, _, degenerate) = oracle(&center);
    assert!(!degenerate, "the dome has a well-defined normal");
    check(&ctx, &gpu, &[center, off_center]);
}

#[test]
fn corner_interpolates_control_points() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBezierSurface::new(&ctx);
    // A Bézier patch interpolates its four corner control points, so (0, 0)
    // yields control[0] and (1, 1) yields control[15]. Use a non-planar net so
    // the corners are distinct in all three axes.
    let mut control = flat_net();
    control[0] = [-1.0, -1.0, 0.3];
    control[15] = [2.0, 2.0, -0.4];
    let origin = BezierSurfaceQuery::new(control, 0.0, 0.0);
    let far = BezierSurfaceQuery::new(control, 1.0, 1.0);
    let (p0, _, _) = oracle(&origin);
    let (p1, _, _) = oracle(&far);
    assert!(
        close(p0[0], -1.0) && close(p0[1], -1.0) && close(p0[2], 0.3),
        "(0, 0) interpolates the first control point"
    );
    assert!(
        close(p1[0], 2.0) && close(p1[1], 2.0) && close(p1[2], -0.4),
        "(1, 1) interpolates the last control point"
    );
    check(&ctx, &gpu, &[origin, far]);
}

#[test]
fn collapsed_net_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBezierSurface::new(&ctx);
    // Every control point coincides, so both partials vanish at every eps step:
    // the tangent frame never recovers, the normal falls back to [0, 0, 1] and
    // the degenerate flag is set. The point collapses to the shared location.
    let control = [[0.7f32, -0.2, 0.5]; 16];
    let q = BezierSurfaceQuery::new(control, 0.42, 0.63);
    let (point, normal, degenerate) = oracle(&q);
    assert!(degenerate, "a collapsed net has no tangent frame");
    assert!(
        close(point[0], 0.7) && close(point[1], -0.2) && close(point[2], 0.5),
        "a collapsed net evaluates to its shared point"
    );
    assert!(
        close(normal[0], 0.0) && close(normal[1], 0.0) && close(normal[2], 1.0),
        "a degenerate normal falls back to the z axis"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBezierSurface::new(&ctx);
    let mut state: u64 = 0x1357_9BDF_2468_ACE0;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let mut control = [[0.0f32; 3]; 16];
        for point in &mut control {
            point[0] = uniform(&mut state, -2.0, 2.0);
            point[1] = uniform(&mut state, -2.0, 2.0);
            point[2] = uniform(&mut state, -2.0, 2.0);
        }
        let u = uniform(&mut state, 0.0, 1.0);
        let v = uniform(&mut state, 0.0, 1.0);
        let q = BezierSurfaceQuery::new(control, u, v);
        // Reject anything whose step-0 tangent frame is near collapse or whose
        // parameters sit on a clamp boundary, so the discrete degenerate flag
        // agrees between host and device.
        if !well_conditioned(&q) {
            continue;
        }
        queries.push(q);
    }
    check(&ctx, &gpu, &queries);
}
