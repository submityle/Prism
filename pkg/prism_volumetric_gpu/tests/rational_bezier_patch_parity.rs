//! Real-device parity for the rational bicubic Bézier (single-span `NURBS`)
//! surface twin:
//! [`GpuRationalBezierPatch`](prism_volumetric_gpu::rational_bezier_patch::GpuRationalBezierPatch)
//! must reproduce the closed-form reference evaluation of the projected point,
//! both projected partial derivatives and the oriented unit normal across a
//! flat planar patch, a unit-weight (polynomial) dome, a non-uniform-weight
//! conic-like patch, several interior parameters, plus rejection-sampled random
//! batches compared element for element.
//!
//! # Independent oracle
//!
//! This suite does not depend on the reference crate. The host [`oracle`] is an
//! independent reimplementation of the same closed form documented on the twin:
//! lift each control point to the homogeneous coordinate `[w*x, w*y, w*z, w]`,
//! run De Casteljau along `u` then `v` (pure linear interpolation), project the
//! point by one division by `H.w`, form the quotient-rule partial numerators
//! `Hu.xyz * H.w - H.xyz * Hu.w` divided by `H.w^2`, and normalize the cross
//! product of the numerators with the reference's interior-nudging fallback
//! loop. A passing parity run is therefore evidence the `WGSL` kernel and an
//! independent `CPU` evaluation of the same rational surface agree, not merely
//! that the shader compiles.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! `sqrt`, so the two evaluations compute the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! host leaves separate, perturbing the low mantissa bits by a few units in the
//! last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on the `f32` fields while pinning the
//! `degenerate` flag exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from the degenerate region and the
//! branch boundary: all weights are strictly positive (so the homogeneous
//! weight `H.w` is a positive convex combination far from the `1e-20` guard),
//! the control net is well spread so the two partials are far from collinear
//! and the normal cross product is comfortably non-zero, and the parameters
//! stay inside `[0.1, 0.9]` away from the clamp edges. Random draws are
//! rejection-sampled against the same partials the kernel evaluates, so `CPU`
//! and `GPU` land on the same side of every branch and the `degenerate` flag is
//! `0` on both regardless of a few units in the last place.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::rational_bezier_patch`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::rational_bezier_patch::{
    GpuRationalBezierPatch, RationalBezierPatchQuery, RationalBezierPatchResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity floor on the `f32` fields.
const DIST_ABS: f32 = 1.0e-4;
/// Relative parity slope on the `f32` fields.
const DIST_REL: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero magnitudes stay meaningful.
const REL_FLOOR: f32 = 1.0e-6;
/// Division/normalization guard matching the reference `1e-20` thresholds.
const GUARD: f32 = 1.0e-20;

/// Returns whether `a` and `b` agree within the documented parity bound: an
/// absolute floor or a relative term keeping large-magnitude values meaningful.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= DIST_ABS || diff <= DIST_REL * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Component-wise linear interpolation `a + (b - a) * t` over a 4-vector.
fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Component-wise difference `a - b` over a 4-vector.
fn sub4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// Cross product of two 3-vectors.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Returns the unit vector along `v`, or `fallback` when `v` is near zero (only
/// `sqrt` is used, no transcendental).
fn normalize_or3(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > GUARD {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        fallback
    }
}

/// Cubic De Casteljau point at `t` over four homogeneous control points.
fn cubic_point4(p: [[f32; 4]; 4], t: f32) -> [f32; 4] {
    let a = lerp4(p[0], p[1], t);
    let b = lerp4(p[1], p[2], t);
    let c = lerp4(p[2], p[3], t);
    let d = lerp4(a, b, t);
    let e = lerp4(b, c, t);
    lerp4(d, e, t)
}

/// Cubic De Casteljau derivative at `t`: `3 *` the quadratic De Casteljau over
/// the adjacent control-point differences, in homogeneous coordinates.
fn cubic_deriv4(p: [[f32; 4]; 4], t: f32) -> [f32; 4] {
    let d0 = sub4(p[1], p[0]);
    let d1 = sub4(p[2], p[1]);
    let d2 = sub4(p[3], p[2]);
    let a = lerp4(d0, d1, t);
    let b = lerp4(d1, d2, t);
    let q = lerp4(a, b, t);
    [q[0] * 3.0, q[1] * 3.0, q[2] * 3.0, q[3] * 3.0]
}

/// Builds the homogeneous control row `[w*x, w*y, w*z, w]` for net row `i`.
fn homogeneous_row(query: &RationalBezierPatchQuery, i: usize) -> [[f32; 4]; 4] {
    let mut row = [[0.0f32; 4]; 4];
    for (col, slot) in row.iter_mut().enumerate() {
        let p = query.control[i * 4 + col];
        let w = query.weights[i * 4 + col];
        *slot = [p[0] * w, p[1] * w, p[2] * w, w];
    }
    row
}

/// Evaluates the homogeneous surface point `H(u, v) = [w*x, w*y, w*z, w]`.
fn homogeneous_point(query: &RationalBezierPatchQuery, u: f32, v: f32) -> [f32; 4] {
    let column = [
        cubic_point4(homogeneous_row(query, 0), u),
        cubic_point4(homogeneous_row(query, 1), u),
        cubic_point4(homogeneous_row(query, 2), u),
        cubic_point4(homogeneous_row(query, 3), u),
    ];
    cubic_point4(column, v)
}

/// Evaluates the homogeneous partial `dH/du`.
fn homogeneous_partial_u(query: &RationalBezierPatchQuery, u: f32, v: f32) -> [f32; 4] {
    let column = [
        cubic_deriv4(homogeneous_row(query, 0), u),
        cubic_deriv4(homogeneous_row(query, 1), u),
        cubic_deriv4(homogeneous_row(query, 2), u),
        cubic_deriv4(homogeneous_row(query, 3), u),
    ];
    cubic_point4(column, v)
}

/// Evaluates the homogeneous partial `dH/dv`.
fn homogeneous_partial_v(query: &RationalBezierPatchQuery, u: f32, v: f32) -> [f32; 4] {
    let column = [
        cubic_point4(homogeneous_row(query, 0), u),
        cubic_point4(homogeneous_row(query, 1), u),
        cubic_point4(homogeneous_row(query, 2), u),
        cubic_point4(homogeneous_row(query, 3), u),
    ];
    cubic_deriv4(column, v)
}

/// Quotient-rule numerator of `dP/du`: `Hu.xyz * H.w - H.xyz * Hu.w`.
fn partial_u_numerator(query: &RationalBezierPatchQuery, u: f32, v: f32) -> [f32; 3] {
    let h = homogeneous_point(query, u, v);
    let hu = homogeneous_partial_u(query, u, v);
    [
        hu[0] * h[3] - h[0] * hu[3],
        hu[1] * h[3] - h[1] * hu[3],
        hu[2] * h[3] - h[2] * hu[3],
    ]
}

/// Quotient-rule numerator of `dP/dv`: `Hv.xyz * H.w - H.xyz * Hv.w`.
fn partial_v_numerator(query: &RationalBezierPatchQuery, u: f32, v: f32) -> [f32; 3] {
    let h = homogeneous_point(query, u, v);
    let hv = homogeneous_partial_v(query, u, v);
    [
        hv[0] * h[3] - h[0] * hv[3],
        hv[1] * h[3] - h[1] * hv[3],
        hv[2] * h[3] - h[2] * hv[3],
    ]
}

/// Projected surface point `H.xyz / H.w`, falling back to the raw numerator
/// when the homogeneous weight collapses.
fn point(query: &RationalBezierPatchQuery, u: f32, v: f32) -> [f32; 3] {
    let h = homogeneous_point(query, u, v);
    if h[3].abs() > GUARD {
        [h[0] / h[3], h[1] / h[3], h[2] / h[3]]
    } else {
        [h[0], h[1], h[2]]
    }
}

/// Projected partial `dP/du`, dividing the quotient-rule numerator by `H.w^2`
/// when safe.
fn partial_u(query: &RationalBezierPatchQuery, u: f32, v: f32) -> [f32; 3] {
    let h = homogeneous_point(query, u, v);
    let n = partial_u_numerator(query, u, v);
    let w2 = h[3] * h[3];
    if w2 > GUARD {
        [n[0] / w2, n[1] / w2, n[2] / w2]
    } else {
        n
    }
}

/// Projected partial `dP/dv`; see [`partial_u`].
fn partial_v(query: &RationalBezierPatchQuery, u: f32, v: f32) -> [f32; 3] {
    let h = homogeneous_point(query, u, v);
    let n = partial_v_numerator(query, u, v);
    let w2 = h[3] * h[3];
    if w2 > GUARD {
        [n[0] / w2, n[1] / w2, n[2] / w2]
    } else {
        n
    }
}

/// The host result shape: projected point, both partials, oriented unit normal
/// and the degenerate flag, mirroring [`RationalBezierPatchResult`].
struct Oracle {
    point: [f32; 3],
    partial_u: [f32; 3],
    partial_v: [f32; 3],
    normal: [f32; 3],
    degenerate: u32,
}

/// Independent reimplementation of the reference `RationalBezierPatch`
/// evaluation, used as the parity oracle. The normal loop mirrors the kernel:
/// the `degenerate` flag starts at `1` and is cleared on the first interior
/// step whose numerator cross product is non-zero, matching the fixed `[0, 0,
/// 1]` fallback semantics branch for branch.
fn oracle(query: &RationalBezierPatchQuery) -> Oracle {
    let u = query.u;
    let v = query.v;
    let fallback = [0.0f32, 0.0, 1.0];
    let mut normal = fallback;
    let mut degenerate = 1u32;
    for step in 0..4 {
        let eps = 1.0e-3 * step as f32;
        let uu = (u + eps).clamp(0.0, 1.0);
        let vv = (v + eps).clamp(0.0, 1.0);
        let du = partial_u_numerator(query, uu, vv);
        let dv = partial_v_numerator(query, uu, vv);
        let n = cross3(du, dv);
        let len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        if len2 > 0.0 {
            normal = normalize_or3(n, fallback);
            degenerate = 0;
            break;
        }
    }
    Oracle {
        point: point(query, u, v),
        partial_u: partial_u(query, u, v),
        partial_v: partial_v(query, u, v),
        normal,
        degenerate,
    }
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// Builds a clearly-conditioned random patch query: a well-spread control net
/// (a planar grid along `u`/`v` with a bounded perturbation in all three axes),
/// strictly positive weights in `[0.5, 2.0)` and an interior parameter in
/// `[0.1, 0.9]`. The draw is rejection-sampled against the same partials the
/// kernel evaluates so the normal is comfortably non-degenerate (the partials
/// are far from collinear and neither is tiny), keeping `CPU` and `GPU` on the
/// same branch and the `degenerate` flag `0` on both.
fn rand_patch(state: &mut u64) -> RationalBezierPatchQuery {
    loop {
        let mut control = [[0.0f32; 3]; 16];
        let mut weights = [0.0f32; 16];
        for row in 0..4 {
            for col in 0..4 {
                let i = row * 4 + col;
                control[i] = [
                    col as f32 * 1.5 + signed(state, 0.3),
                    row as f32 * 1.5 + signed(state, 0.3),
                    signed(state, 0.8),
                ];
                weights[i] = 0.5 + lcg(state) * 1.5;
            }
        }
        let u = 0.1 + lcg(state) * 0.8;
        let v = 0.1 + lcg(state) * 0.8;
        let query = RationalBezierPatchQuery::new(control, weights, u, v);

        // Reject ill-conditioned draws: the projected partials must be well
        // scaled and far from collinear so the cross-product normal is
        // comfortably non-degenerate on both evaluators.
        let got = oracle(&query);
        if got.degenerate != 0 {
            continue;
        }
        let pu = partial_u(&query, u, v);
        let pv = partial_v(&query, u, v);
        let lpu = dot3(pu, pu);
        let lpv = dot3(pv, pv);
        if lpu < 0.5 || lpv < 0.5 {
            continue;
        }
        let c = cross3(pu, pv);
        // sin^2(angle) = |cross|^2 / (|pu|^2 |pv|^2); require it well above zero.
        if dot3(c, c) < 0.1 * lpu * lpv {
            continue;
        }
        return query;
    }
}

/// Pins one `GPU` result against the host oracle for `query`: the `degenerate`
/// flag must match exactly, and the point, both partials and the unit normal
/// must agree within the parity bound.
fn pin(idx: usize, query: &RationalBezierPatchQuery, got: &RationalBezierPatchResult) {
    let want = oracle(query);
    assert_eq!(
        got.degenerate, want.degenerate,
        "query {idx}: degenerate flag must match the oracle (gpu {}, cpu {})",
        got.degenerate, want.degenerate
    );
    assert!(
        close(got.point[0], want.point[0])
            && close(got.point[1], want.point[1])
            && close(got.point[2], want.point[2]),
        "query {idx} point: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.point[0],
        got.point[1],
        got.point[2],
        want.point[0],
        want.point[1],
        want.point[2]
    );
    assert!(
        close(got.partial_u[0], want.partial_u[0])
            && close(got.partial_u[1], want.partial_u[1])
            && close(got.partial_u[2], want.partial_u[2]),
        "query {idx} partial_u: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.partial_u[0],
        got.partial_u[1],
        got.partial_u[2],
        want.partial_u[0],
        want.partial_u[1],
        want.partial_u[2]
    );
    assert!(
        close(got.partial_v[0], want.partial_v[0])
            && close(got.partial_v[1], want.partial_v[1])
            && close(got.partial_v[2], want.partial_v[2]),
        "query {idx} partial_v: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.partial_v[0],
        got.partial_v[1],
        got.partial_v[2],
        want.partial_v[0],
        want.partial_v[1],
        want.partial_v[2]
    );
    assert!(
        close(got.normal[0], want.normal[0])
            && close(got.normal[1], want.normal[1])
            && close(got.normal[2], want.normal[2]),
        "query {idx} normal: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.normal[0],
        got.normal[1],
        got.normal[2],
        want.normal[0],
        want.normal[1],
        want.normal[2]
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuRationalBezierPatch, queries: &[RationalBezierPatchQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

/// A flat planar control net on the `z = 0` plane: `control[row * 4 + col] =
/// [col, row, 0]`, so with unit weights the surface is `P(u, v) = (3u, 3v, 0)`,
/// its partials are the constants `(3, 0, 0)` and `(0, 3, 0)` and its normal is
/// `(0, 0, 1)`.
fn planar_patch(u: f32, v: f32) -> RationalBezierPatchQuery {
    let mut control = [[0.0f32; 3]; 16];
    for row in 0..4 {
        for col in 0..4 {
            control[row * 4 + col] = [col as f32, row as f32, 0.0];
        }
    }
    RationalBezierPatchQuery::new(control, [1.0; 16], u, v)
}

/// A unit-weight (polynomial) dome: the planar grid with the four interior
/// control points lifted in `z`, so the patch bulges smoothly with a
/// well-conditioned normal.
fn dome_patch(u: f32, v: f32) -> RationalBezierPatchQuery {
    let mut control = [[0.0f32; 3]; 16];
    for row in 0..4 {
        for col in 0..4 {
            control[row * 4 + col] = [col as f32, row as f32, 0.0];
        }
    }
    control[5][2] = 1.2;
    control[6][2] = 1.2;
    control[9][2] = 1.2;
    control[10][2] = 1.2;
    RationalBezierPatchQuery::new(control, [1.0; 16], u, v)
}

/// A non-uniform positive-weight patch over the dome net: corner weights `1`,
/// edge weights `0.8`, interior weights `1.4`, bending the rational surface
/// toward the heavier interior points as a conic-like span would.
fn weighted_patch(u: f32, v: f32) -> RationalBezierPatchQuery {
    let mut control = [[0.0f32; 3]; 16];
    for row in 0..4 {
        for col in 0..4 {
            control[row * 4 + col] = [col as f32, row as f32, 0.0];
        }
    }
    control[5][2] = 1.2;
    control[6][2] = 1.2;
    control[9][2] = 1.2;
    control[10][2] = 1.2;
    let weights = [
        1.0, 0.8, 0.8, 1.0, 0.8, 1.4, 1.4, 0.8, 0.8, 1.4, 1.4, 0.8, 1.0, 0.8, 0.8, 1.0,
    ];
    RationalBezierPatchQuery::new(control, weights, u, v)
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRationalBezierPatch::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn planar_patch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRationalBezierPatch::new(&ctx);
    let queries = [
        planar_patch(0.5, 0.5),
        planar_patch(0.25, 0.75),
        planar_patch(0.1, 0.9),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn unit_weight_dome_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRationalBezierPatch::new(&ctx);
    let queries = [
        dome_patch(0.5, 0.5),
        dome_patch(0.3, 0.6),
        dome_patch(0.8, 0.2),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn nonuniform_weight_patch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRationalBezierPatch::new(&ctx);
    let queries = [
        weighted_patch(0.5, 0.5),
        weighted_patch(0.2, 0.7),
        weighted_patch(0.65, 0.35),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRationalBezierPatch::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random patches,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element for element.
    let mut queries = vec![
        planar_patch(0.5, 0.5),
        dome_patch(0.4, 0.6),
        weighted_patch(0.55, 0.45),
    ];
    for _ in 0..48 {
        queries.push(rand_patch(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRationalBezierPatch::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned random patches (several workgroups'
    // worth) pins the point, both partials and the oriented normal across many
    // random rational surfaces.
    let queries: Vec<RationalBezierPatchQuery> = (0..512).map(|_| rand_patch(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
