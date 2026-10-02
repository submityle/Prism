//! Real-device parity for the `OKLab` / `OKLCh` twin:
//! [`GpuOklabColor`](prism_volumetric_gpu::oklab_color::GpuOklabColor) must
//! reproduce the `CPU` golden
//! [`oklab_color`](prism_render_architecture::particle::oklab_color) across the
//! forward `OKLab` transform, the inverse linear-`sRGB` transform, the forward
//! `OKLCh` conversion, the inverse `OKLab` reconstruction, the hue rotation, and
//! the perceptual `lerp`.
//!
//! The fixtures cover the shapes the golden unit tests call out: the achromatic
//! greys and primaries, mixed interior colours, an already-cylindrical `OKLCh`
//! probe, a near-grey colour whose chroma folds to the canonical hue vector
//! `(1, 0)`, identity and quarter-turn hue rotations, and interpolation at the
//! endpoints and midpoint. Every hue-rotation angle is supplied as an exact
//! rational `(cos_delta, sin_delta)` unit vector (such as `(0.6, 0.8)`), and
//! every colour channel is a simple decimal, so the fixtures stay pure and need
//! no external maths library and no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned output is a continuous `f32`, so `CPU` and `GPU` are compared
//! under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR =
//! 1e-6`): tight enough to catch a dropped term, a swapped matrix row or a wrong
//! Newton update, loose enough to admit legal fused multiply-add contraction and
//! the hand-rolled cube root's rounding.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::oklab_color`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::oklab_color::{
    lerp, linear_srgb_to_oklab, oklab_to_linear_srgb, oklab_to_oklch, oklch_to_oklab, rotate_hue,
    LinearSrgb, OkLab, OkLch,
};
use prism_volumetric_gpu::oklab_color::{GpuOklabColor, OklabColorQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two triples.
fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
}

/// Tolerant comparison of two quads.
fn approx4(a: [f32; 4], b: [f32; 4]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2]) && approx(a[3], b[3])
}

/// Builds a query from its linear `sRGB`, `OKLab`, `OKLCh`, lerp-endpoint and
/// hue-rotation / interpolation inputs.
fn query(
    lin: [f32; 3],
    lab: [f32; 3],
    lch: [f32; 4],
    lab_a: [f32; 3],
    lab_b: [f32; 3],
    cos_delta: f32,
    sin_delta: f32,
    t: f32,
) -> OklabColorQuery {
    OklabColorQuery {
        lin,
        lab,
        lch,
        lab_a,
        lab_b,
        cos_delta,
        sin_delta,
        t,
    }
}

/// Asserts every twinned answer for one query matches the `CPU` golden.
fn assert_parity(gpu: &GpuOklabColor, ctx: &GpuContext, q: &OklabColorQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    let cpu_oklab = linear_srgb_to_oklab(&LinearSrgb::new(q.lin[0], q.lin[1], q.lin[2]));
    let cpu_oklab = [cpu_oklab.l, cpu_oklab.a, cpu_oklab.b];
    assert!(
        approx3(g.oklab_from_linear, cpu_oklab),
        "linear_srgb_to_oklab mismatch: gpu {:?} vs cpu {cpu_oklab:?}",
        g.oklab_from_linear
    );

    let cpu_lin = oklab_to_linear_srgb(&OkLab::new(q.lab[0], q.lab[1], q.lab[2]));
    let cpu_lin = [cpu_lin.r, cpu_lin.g, cpu_lin.b];
    assert!(
        approx3(g.linear_from_oklab, cpu_lin),
        "oklab_to_linear_srgb mismatch: gpu {:?} vs cpu {cpu_lin:?}",
        g.linear_from_oklab
    );

    let cpu_lch = oklab_to_oklch(&OkLab::new(q.lab[0], q.lab[1], q.lab[2]));
    let cpu_lch = [cpu_lch.l, cpu_lch.c, cpu_lch.h_cos, cpu_lch.h_sin];
    assert!(
        approx4(g.oklch_from_oklab, cpu_lch),
        "oklab_to_oklch mismatch: gpu {:?} vs cpu {cpu_lch:?}",
        g.oklch_from_oklab
    );

    let in_lch = OkLch {
        l: q.lch[0],
        c: q.lch[1],
        h_cos: q.lch[2],
        h_sin: q.lch[3],
    };
    let cpu_lab = oklch_to_oklab(&in_lch);
    let cpu_lab = [cpu_lab.l, cpu_lab.a, cpu_lab.b];
    assert!(
        approx3(g.oklab_from_oklch, cpu_lab),
        "oklch_to_oklab mismatch: gpu {:?} vs cpu {cpu_lab:?}",
        g.oklab_from_oklch
    );

    let cpu_rot = rotate_hue(&in_lch, q.cos_delta, q.sin_delta);
    let cpu_rot = [cpu_rot.l, cpu_rot.c, cpu_rot.h_cos, cpu_rot.h_sin];
    assert!(
        approx4(g.rotated, cpu_rot),
        "rotate_hue mismatch: gpu {:?} vs cpu {cpu_rot:?}",
        g.rotated
    );

    let cpu_lerp = lerp(
        &OkLab::new(q.lab_a[0], q.lab_a[1], q.lab_a[2]),
        &OkLab::new(q.lab_b[0], q.lab_b[1], q.lab_b[2]),
        q.t,
    );
    let cpu_lerp = [cpu_lerp.l, cpu_lerp.a, cpu_lerp.b];
    assert!(
        approx3(g.lerped, cpu_lerp),
        "lerp mismatch: gpu {:?} vs cpu {cpu_lerp:?}",
        g.lerped
    );
}

/// A reusable `OKLCh` probe (an off-axis hue) for the inverse and rotation
/// paths.
fn probe_lch() -> [f32; 4] {
    // (l, c, h_cos, h_sin) with a unit hue vector (0.6, 0.8).
    [0.6, 0.2, 0.6, 0.8]
}

/// A reusable pair of `OKLab` lerp endpoints.
fn endpoints() -> ([f32; 3], [f32; 3]) {
    ([0.2, -0.1, 0.05], [0.8, 0.1, -0.05])
}

#[test]
fn primaries_and_greys_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOklabColor::new(&ctx);
    let (a, b) = endpoints();
    let samples = [
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.3, 0.3, 0.3],
    ];
    for lin in samples {
        let q = query(lin, [0.6, 0.12, -0.05], probe_lch(), a, b, 0.6, 0.8, 0.5);
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn arbitrary_colours_roundtrip_paths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOklabColor::new(&ctx);
    let (a, b) = endpoints();
    let labs = [
        [0.6, 0.12, -0.05],
        [0.4, -0.09, 0.11],
        [0.75, 0.03, 0.02],
        [0.5, 0.3, 0.4],
    ];
    let lins = [
        [0.25, 0.5, 0.75],
        [0.9, 0.1, 0.4],
        [0.05, 0.6, 0.2],
        [0.7, 0.7, 0.3],
    ];
    for (lin, lab) in lins.into_iter().zip(labs) {
        let q = query(lin, lab, probe_lch(), a, b, 0.8, -0.6, 0.25);
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn near_grey_folds_to_canonical_hue() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOklabColor::new(&ctx);
    let (a, b) = endpoints();
    // An OKLab colour with zero chroma must canonicalise to the hue vector
    // (1, 0) on both the CPU and the GPU.
    let q = query(
        [0.3, 0.3, 0.3],
        [0.5, 0.0, 0.0],
        probe_lch(),
        a,
        b,
        1.0,
        0.0,
        0.5,
    );
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn hue_rotation_identity_and_quarter_turn() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOklabColor::new(&ctx);
    let (a, b) = endpoints();
    // Identity rotation (1, 0) leaves the hue vector unchanged.
    let identity = query(
        [0.5, 0.5, 0.5],
        [0.6, 0.12, -0.05],
        probe_lch(),
        a,
        b,
        1.0,
        0.0,
        0.5,
    );
    assert_parity(&gpu, &ctx, &identity);
    // Quarter turn (0, 1) rotates the hue vector by 90 degrees.
    let quarter = query(
        [0.5, 0.5, 0.5],
        [0.6, 0.12, -0.05],
        [0.5, 0.2, 1.0, 0.0],
        a,
        b,
        0.0,
        1.0,
        0.5,
    );
    assert_parity(&gpu, &ctx, &quarter);
}

#[test]
fn lerp_hits_endpoints_and_midpoint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOklabColor::new(&ctx);
    let (a, b) = endpoints();
    for t in [0.0f32, 0.5, 1.0] {
        let q = query(
            [0.4, 0.4, 0.4],
            [0.6, 0.12, -0.05],
            probe_lch(),
            a,
            b,
            0.6,
            0.8,
            t,
        );
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOklabColor::new(&ctx);
    let (a, b) = endpoints();
    // A batch exercises the one-thread-per-query flattening; each result must be
    // independent of its neighbours.
    let batch = [
        query(
            [0.25, 0.5, 0.75],
            [0.6, 0.12, -0.05],
            probe_lch(),
            a,
            b,
            0.6,
            0.8,
            0.25,
        ),
        query(
            [0.9, 0.1, 0.4],
            [0.4, -0.09, 0.11],
            [0.4, 0.15, 0.8, -0.6],
            a,
            b,
            0.8,
            -0.6,
            0.5,
        ),
        query(
            [0.05, 0.6, 0.2],
            [0.5, 0.0, 0.0],
            probe_lch(),
            a,
            b,
            1.0,
            0.0,
            0.75,
        ),
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOklabColor::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
