//! Real-device parity for the bicubic Catmull-Rom surface patch twin:
//! [`GpuCatmullRomPatch`](prism_volumetric_gpu::catmull_rom_patch::GpuCatmullRomPatch)
//! must reproduce the analytic golden
//! `prism_render_architecture::ray_scene::catmull_rom_patch::CatmullRomPatch`
//! (`point` and `normal`) across an empty batch, a planar net, a domed net, the
//! four interpolated corners and a large pseudo-random sweep compared lane for
//! lane.
//!
//! The host oracle in this file re-derives the closed form independently (it
//! does **not** import the golden crate): the two-pass span-to-Bézier change of
//! basis, the bicubic De Casteljau `point`, the `∂P/∂u` and `∂P/∂v` partials
//! and the degenerate-tangent normal search. A passing run is therefore direct
//! evidence the ported kernel folds the same conversion and evaluation, not
//! merely that its shader compiles.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of linear interpolations and
//! one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The comparison therefore allows `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` on the continuous `point` and `normal` values yet
//! asserts an *exact* match on the discrete `degenerate` flag. Every fixture
//! and every random net is placed clear of the tangent-collapse boundary, the
//! only place a legal `ULP` perturbation can flip that flag, so the exact-flag
//! assertion is unconditional.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::catmull_rom_patch`；
//! 无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::catmull_rom_patch::{
    CatmullRomPatchQuery, CatmullRomPatchResult, GpuCatmullRomPatch,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the surface point and normal components. A `GPU`
/// may fuse a multiply-add the scalar reference leaves separate, perturbing the
/// low mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Componentwise `a - b`.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Componentwise `a + b`.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales `a` by the scalar `s`.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean dot product.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Right-handed cross product `a × b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Linear interpolation `a + t·(b − a)`.
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    add(a, scale(sub(b, a), t))
}

/// Converts one uniform Catmull-Rom span `[c0, c1, c2, c3]` into the four
/// cubic Bézier control points over `[c1, c2]`: `b0 = c1`,
/// `b1 = c1 + (c2 − c0)/6`, `b2 = c2 − (c3 − c1)/6`, `b3 = c2`.
fn span_to_bezier(c0: [f32; 3], c1: [f32; 3], c2: [f32; 3], c3: [f32; 3]) -> [[f32; 3]; 4] {
    let sixth = 1.0 / 6.0;
    let b1 = add(c1, scale(sub(c2, c0), sixth));
    let b2 = sub(c2, scale(sub(c3, c1), sixth));
    [c1, b1, b2, c2]
}

/// Two-pass tensor-product conversion of the row-major `4×4` Catmull-Rom net
/// into its equivalent Bézier net (spans along each `u` row, then along each
/// `v` column).
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

/// The four control points of `v`-row `r` (its `u`-direction curve).
fn row(bez: &[[f32; 3]; 16], r: usize) -> [[f32; 3]; 4] {
    let b = r * 4;
    [bez[b], bez[b + 1], bez[b + 2], bez[b + 3]]
}

/// Cubic Bézier point at `t` over four control points (three nested `lerp`s).
fn cubic_point(p: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let a = lerp3(p[0], p[1], t);
    let b = lerp3(p[1], p[2], t);
    let c = lerp3(p[2], p[3], t);
    let d = lerp3(a, b, t);
    let e = lerp3(b, c, t);
    lerp3(d, e, t)
}

/// Cubic Bézier derivative at `t`: `3·` the quadratic De Casteljau over the
/// three adjacent control-point differences.
fn cubic_deriv(p: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let d0 = sub(p[1], p[0]);
    let d1 = sub(p[2], p[1]);
    let d2 = sub(p[3], p[2]);
    let a = lerp3(d0, d1, t);
    let b = lerp3(d1, d2, t);
    let q = lerp3(a, b, t);
    scale(q, 3.0)
}

/// Surface point `P(u, v)`: collapse each `v`-row to a point by a cubic De
/// Casteljau in `u`, then collapse the four results by a cubic De Casteljau in
/// `v`.
fn surface_point(bez: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    let column = [
        cubic_point(row(bez, 0), u),
        cubic_point(row(bez, 1), u),
        cubic_point(row(bez, 2), u),
        cubic_point(row(bez, 3), u),
    ];
    cubic_point(column, v)
}

/// Partial derivative `∂P/∂u`.
fn surface_partial_u(bez: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    let column = [
        cubic_deriv(row(bez, 0), u),
        cubic_deriv(row(bez, 1), u),
        cubic_deriv(row(bez, 2), u),
        cubic_deriv(row(bez, 3), u),
    ];
    cubic_point(column, v)
}

/// Partial derivative `∂P/∂v`.
fn surface_partial_v(bez: &[[f32; 3]; 16], u: f32, v: f32) -> [f32; 3] {
    let column = [
        cubic_point(row(bez, 0), u),
        cubic_point(row(bez, 1), u),
        cubic_point(row(bez, 2), u),
        cubic_point(row(bez, 3), u),
    ];
    cubic_deriv(column, v)
}

/// Unit surface normal with the reference's degenerate-tangent search: nudge
/// the sample a few steps toward the interior until a non-degenerate frame is
/// found, else fall back to the up axis and flag the sample degenerate.
fn surface_normal(bez: &[[f32; 3]; 16], u: f32, v: f32) -> ([f32; 3], bool) {
    let fallback = [0.0, 0.0, 1.0];
    for step in 0..4 {
        let eps = 1.0e-3 * step as f32;
        let uu = (u + eps).clamp(0.0, 1.0);
        let vv = (v + eps).clamp(0.0, 1.0);
        let du = surface_partial_u(bez, uu, vv);
        let dv = surface_partial_v(bez, uu, vv);
        let n = cross(du, dv);
        let len2 = dot(n, n);
        if len2 > 0.0 {
            let inv = 1.0 / len2.sqrt();
            return (scale(n, inv), false);
        }
    }
    (fallback, true)
}

/// Independent host re-derivation of the golden Catmull-Rom patch evaluation.
/// It does not call the reference crate; it reimplements the closed form so the
/// parity assertion compares two independent solutions.
fn oracle(q: &CatmullRomPatchQuery) -> CatmullRomPatchResult {
    let bez = to_bezier(&q.control);
    let point = surface_point(&bez, q.u, q.v);
    let (normal, degenerate) = surface_normal(&bez, q.u, q.v);
    CatmullRomPatchResult {
        point,
        normal,
        degenerate,
    }
}

/// A planar `4×4` net on the integer grid: `control[row*4+col] = [col-1, row-1, 0]`.
/// Over the central cell the surface is the plane `P(u, v) = (u, v, 0)`.
fn flat_net() -> [[f32; 3]; 16] {
    let mut c = [[0.0f32; 3]; 16];
    for (slot, p) in c.iter_mut().enumerate() {
        let col = (slot % 4) as f32 - 1.0;
        let r = (slot / 4) as f32 - 1.0;
        *p = [col, r, 0.0];
    }
    c
}

/// The planar net with its inner `2×2` points lifted in `z`, giving a smoothly
/// domed, non-planar patch.
fn domed_net() -> [[f32; 3]; 16] {
    let mut c = flat_net();
    for row in 1..3 {
        for col in 1..3 {
            c[row * 4 + col][2] = 0.5;
        }
    }
    c
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

/// Maps a `[0, 1)` sample into `[lo, hi)`.
fn uniform(x: f32, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * x
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// independent host oracle: the `degenerate` flag matches exactly, and the
/// point and each normal component match within tolerance. Returns the `GPU`
/// verdicts for extra per-test assertions. Use only for queries placed clear of
/// the tangent-collapse boundary.
fn check(
    ctx: &GpuContext,
    gpu: &GpuCatmullRomPatch,
    queries: &[CatmullRomPatchQuery],
) -> Vec<CatmullRomPatchResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let o = oracle(q);
        assert_eq!(
            g.degenerate, o.degenerate,
            "lane {lane}: degenerate gpu {} vs cpu {}",
            g.degenerate, o.degenerate
        );
        for axis in 0..3 {
            assert!(
                close(g.point[axis], o.point[axis]),
                "lane {lane}: point[{axis}] gpu {} vs cpu {}",
                g.point[axis],
                o.point[axis]
            );
            assert!(
                close(g.normal[axis], o.normal[axis]),
                "lane {lane}: normal[{axis}] gpu {} vs cpu {}",
                g.normal[axis],
                o.normal[axis]
            );
        }
    }
    got
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCatmullRomPatch::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn flat_patch_is_planar() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCatmullRomPatch::new(&ctx);
    let net = flat_net();

    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut queries: Vec<CatmullRomPatchQuery> = Vec::new();
    for _ in 0..16 {
        let u = uniform(lcg(&mut state), 0.1, 0.9);
        let v = uniform(lcg(&mut state), 0.1, 0.9);
        queries.push(CatmullRomPatchQuery::new(net, u, v));
    }

    let got = check(&ctx, &gpu, &queries);
    for (g, q) in got.iter().zip(queries.iter()) {
        // On the central cell the planar net maps (u, v) -> (u, v, 0).
        assert!(close(g.point[0], q.u), "planar point x = u");
        assert!(close(g.point[1], q.v), "planar point y = v");
        assert!(close(g.point[2], 0.0), "planar point z = 0");
        assert!(!g.degenerate, "planar net has a well-conditioned frame");
        // Normal is the up axis (+z) for the planar grid.
        assert!(close(g.normal[0], 0.0), "planar normal x = 0");
        assert!(close(g.normal[1], 0.0), "planar normal y = 0");
        assert!(close(g.normal[2], 1.0), "planar normal z = 1");
    }
}

#[test]
fn domed_patch_curves() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCatmullRomPatch::new(&ctx);
    let net = domed_net();

    let mut state: u64 = 0x0bad_c0de_1234_5678;
    let mut queries: Vec<CatmullRomPatchQuery> = Vec::new();
    for _ in 0..24 {
        let u = uniform(lcg(&mut state), 0.1, 0.9);
        let v = uniform(lcg(&mut state), 0.1, 0.9);
        queries.push(CatmullRomPatchQuery::new(net, u, v));
    }

    let got = check(&ctx, &gpu, &queries);
    // A domed patch bulges in z near its center, so at least one sample has a
    // clearly positive surface height; this guards against a degenerate fixture
    // collapsing to the planar case.
    let saw_bulge = got.iter().any(|g| g.point[2] > 0.05);
    assert!(saw_bulge, "domed patch should rise above the base plane");
}

#[test]
fn corner_interpolation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCatmullRomPatch::new(&ctx);
    let net = domed_net();

    // The Catmull-Rom surface interpolates its inner 2x2 control points at the
    // domain corners: P(0,0)=control[5], P(1,0)=control[6], P(0,1)=control[9],
    // P(1,1)=control[10].
    let corners = [
        (0.0_f32, 0.0_f32, 5usize),
        (1.0, 0.0, 6),
        (0.0, 1.0, 9),
        (1.0, 1.0, 10),
    ];
    let queries: Vec<CatmullRomPatchQuery> = corners
        .iter()
        .map(|&(u, v, _)| CatmullRomPatchQuery::new(net, u, v))
        .collect();

    let got = check(&ctx, &gpu, &queries);
    for (g, &(_, _, idx)) in got.iter().zip(corners.iter()) {
        for (axis, (&gp, &np)) in g.point.iter().zip(net[idx].iter()).enumerate() {
            assert!(close(gp, np), "corner point[{axis}] gpu {gp} vs net {np}");
        }
    }
}

#[test]
fn random_sweep_matches_lane_for_lane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCatmullRomPatch::new(&ctx);

    let mut state: u64 = 0x0f0e_0d0c_0b0a_0908;
    let mut queries: Vec<CatmullRomPatchQuery> = Vec::new();

    for _ in 0..512 {
        // Start from the planar integer grid and perturb every control point:
        // small in-plane jitter plus a moderate z lift keeps the tangent frame
        // comfortably x/y-dominant, so the surface stays a well-conditioned
        // non-degenerate patch.
        let mut net = flat_net();
        for p in net.iter_mut() {
            p[0] += uniform(lcg(&mut state), -0.15, 0.15);
            p[1] += uniform(lcg(&mut state), -0.15, 0.15);
            p[2] += uniform(lcg(&mut state), -0.5, 0.5);
        }
        let u = uniform(lcg(&mut state), 0.05, 0.95);
        let v = uniform(lcg(&mut state), 0.05, 0.95);

        // Reject any net whose tangent frame is near collapse, where a legal
        // ULP perturbation could flip the discrete degenerate flag.
        let bez = to_bezier(&net);
        let du = surface_partial_u(&bez, u, v);
        let dv = surface_partial_v(&bez, u, v);
        let n = cross(du, dv);
        if dot(n, n) < 0.04 {
            continue;
        }

        queries.push(CatmullRomPatchQuery::new(net, u, v));
    }

    assert!(
        !queries.is_empty(),
        "sweep should retain conditioned queries"
    );
    let got = check(&ctx, &gpu, &queries);
    let saw_nondegenerate = got.iter().any(|g| !g.degenerate);
    assert!(
        saw_nondegenerate,
        "sweep should produce non-degenerate frames"
    );
}
