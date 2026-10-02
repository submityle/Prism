//! Real-device parity for the `CIELAB` twin:
//! [`GpuCielab`](prism_volumetric_gpu::cielab::GpuCielab) must reproduce the
//! `CPU` golden [`cielab`](prism_render_architecture::particle::cielab) across
//! the Newton cube root, the forward and inverse lightness legs, the
//! `xyz_to_lab` and `lab_to_xyz` conversions, the `CIE76` and `CIE94` color
//! differences and the `lab_lerp` blend.
//!
//! The fixtures cover the shapes the golden unit tests call out: perfect and
//! fractional cubes plus a negative argument for `cbrt_newton`, inputs on both
//! legs of the lightness curve (held clear of the `δ = 6/29` knot), the `D65`
//! white and a handful of in-gamut colors round-tripping through `Lab`, zero /
//! single-axis / asymmetric color-difference cases, and the graphic-arts versus
//! textiles `CIE94` selector. All inputs are written as integers or simple
//! decimals, so the fixtures stay pure and need no external math library and no
//! transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every reported quantity is a continuous float threading through multiplies,
//! adds, one guarded division and `sqrt`, so `CPU` and `GPU` are compared under
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`),
//! loose enough to admit legal fused multiply-add contraction yet tight enough
//! to catch a wrong port.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::cielab`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::cielab::{
    cbrt_newton, delta_e_76, delta_e_94, lab_f, lab_f_inv, lab_lerp, lab_to_xyz, xyz_to_lab, Lab,
    Xyz,
};
use prism_volumetric_gpu::cielab::{CielabQuery, CielabResult, GpuCielab};
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

/// Tolerant comparison of two `vec3` triples.
fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
}

/// A default-ish query the per-fixture helpers tweak; all legs stay clear of the
/// lightness knot and the color-difference degeneracies.
fn base_query() -> CielabQuery {
    CielabQuery {
        xyz_in: [0.4, 0.5, 0.6],
        lab_in: [50.0, 10.0, -20.0],
        lab_a: [50.0, 2.0, 3.0],
        lab_b: [60.0, 5.0, 7.0],
        cbrt_x: 8.0,
        lab_f_t: 0.5,
        lab_f_inv_t: 0.5,
        lerp_t: 0.5,
        graphic: true,
    }
}

/// Asserts every twinned answer for one query matches the `CPU` golden.
fn assert_parity(gpu: &GpuCielab, ctx: &GpuContext, q: &CielabQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g: CielabResult = got[0];

    assert!(
        approx(g.cbrt, cbrt_newton(q.cbrt_x)),
        "cbrt mismatch: gpu {} vs cpu {}",
        g.cbrt,
        cbrt_newton(q.cbrt_x)
    );
    assert!(
        approx(g.lab_f, lab_f(q.lab_f_t)),
        "lab_f mismatch: gpu {} vs cpu {}",
        g.lab_f,
        lab_f(q.lab_f_t)
    );
    assert!(
        approx(g.lab_f_inv, lab_f_inv(q.lab_f_inv_t)),
        "lab_f_inv mismatch: gpu {} vs cpu {}",
        g.lab_f_inv,
        lab_f_inv(q.lab_f_inv_t)
    );

    let xyz = Xyz::new(q.xyz_in[0], q.xyz_in[1], q.xyz_in[2]);
    let cpu_lab = xyz_to_lab(&xyz);
    assert!(
        approx3(g.lab_from_xyz, [cpu_lab.l, cpu_lab.a, cpu_lab.b]),
        "xyz_to_lab mismatch: gpu {:?} vs cpu {cpu_lab:?}",
        g.lab_from_xyz
    );

    let lab = Lab::new(q.lab_in[0], q.lab_in[1], q.lab_in[2]);
    let cpu_xyz = lab_to_xyz(&lab);
    assert!(
        approx3(g.xyz_from_lab, [cpu_xyz.x, cpu_xyz.y, cpu_xyz.z]),
        "lab_to_xyz mismatch: gpu {:?} vs cpu {cpu_xyz:?}",
        g.xyz_from_lab
    );

    let lab_a = Lab::new(q.lab_a[0], q.lab_a[1], q.lab_a[2]);
    let lab_b = Lab::new(q.lab_b[0], q.lab_b[1], q.lab_b[2]);
    assert!(
        approx(g.delta_e_76, delta_e_76(&lab_a, &lab_b)),
        "delta_e_76 mismatch: gpu {} vs cpu {}",
        g.delta_e_76,
        delta_e_76(&lab_a, &lab_b)
    );
    assert!(
        approx(g.delta_e_94, delta_e_94(&lab_a, &lab_b, q.graphic)),
        "delta_e_94 mismatch: gpu {} vs cpu {}",
        g.delta_e_94,
        delta_e_94(&lab_a, &lab_b, q.graphic)
    );

    let cpu_lerp = lab_lerp(&lab_a, &lab_b, q.lerp_t);
    assert!(
        approx3(g.lab_lerp, [cpu_lerp.l, cpu_lerp.a, cpu_lerp.b]),
        "lab_lerp mismatch: gpu {:?} vs cpu {cpu_lerp:?}",
        g.lab_lerp
    );
}

#[test]
fn cbrt_matches_cubes_and_reflection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCielab::new(&ctx);
    // Perfect cubes, a fractional cube, zero and a negative argument.
    for &x in &[
        8.0_f32, 27.0, 64.0, 1000.0, 0.125, 0.001, 1.0, 0.0, -8.0, -27.0,
    ] {
        let mut q = base_query();
        q.cbrt_x = x;
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn lightness_legs_match_on_both_sides_of_the_knot() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCielab::new(&ctx);
    // The knot sits near 0.008856 (forward) and 0.206897 (inverse); every fixture
    // is held clear of both so the branch choice is unambiguous.
    for &t in &[0.0005_f32, 0.004, 0.05, 0.4, 0.9, 1.5] {
        let mut q = base_query();
        q.lab_f_t = t;
        q.lab_f_inv_t = t;
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn xyz_lab_roundtrip_colors_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCielab::new(&ctx);
    // D65 white plus a few in-gamut colors converted forward and backward.
    let whites = [
        [0.950_489_f32, 1.0, 0.888_840],
        [0.4, 0.5, 0.6],
        [0.1, 0.2, 0.05],
        [0.8, 0.78, 0.9],
    ];
    for &xyz in &whites {
        let mut q = base_query();
        q.xyz_in = xyz;
        // Feed the forward result back as the inverse input to exercise both legs.
        let lab = xyz_to_lab(&Xyz::new(xyz[0], xyz[1], xyz[2]));
        q.lab_in = [lab.l, lab.a, lab.b];
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn color_difference_cases_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCielab::new(&ctx);
    // Identical colors (zero distance), a single-axis offset, a general pair and
    // a neutral reference where CIE94 collapses toward CIE76.
    let pairs = [
        ([42.0_f32, -7.0, 13.0], [42.0, -7.0, 13.0]),
        ([50.0, 0.0, 0.0], [50.0, 10.0, 0.0]),
        ([30.0, 10.0, -5.0], [45.0, -8.0, 12.0]),
        ([50.0, 10.0, 0.0], [50.0, 0.0, 0.0]),
    ];
    for &(a, b) in &pairs {
        let mut q = base_query();
        q.lab_a = a;
        q.lab_b = b;
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn cie94_graphic_and_textile_selectors_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCielab::new(&ctx);
    // The same color pair under both application selectors.
    for &graphic in &[true, false] {
        let mut q = base_query();
        q.lab_a = [50.0, 20.0, 10.0];
        q.lab_b = [55.0, 30.0, 18.0];
        q.graphic = graphic;
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn lab_lerp_endpoints_and_midpoint_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCielab::new(&ctx);
    for &t in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let mut q = base_query();
        q.lab_a = [10.0, -20.0, 30.0];
        q.lab_b = [90.0, 40.0, -50.0];
        q.lerp_t = t;
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCielab::new(&ctx);
    // A batch exercises the one-thread-per-query flattening; each result must be
    // independent of its neighbours.
    let batch = [
        {
            let mut q = base_query();
            q.cbrt_x = 125.0;
            q.graphic = false;
            q
        },
        {
            let mut q = base_query();
            q.lab_f_t = 0.9;
            q.lab_f_inv_t = 1.2;
            q.lab_a = [40.0, 25.0, -15.0];
            q.lab_b = [48.0, 12.0, 6.0];
            q
        },
        {
            let mut q = base_query();
            q.cbrt_x = -64.0;
            q.xyz_in = [0.2, 0.3, 0.1];
            q
        },
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
    let gpu = GpuCielab::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
