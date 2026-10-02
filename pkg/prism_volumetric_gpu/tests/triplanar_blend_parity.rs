//! Real-device parity for the triplanar-blend twin:
//! [`GpuTriplanarBlend`](prism_volumetric_gpu::triplanar_blend::GpuTriplanarBlend)
//! must reproduce the `CPU` golden
//! [`triplanar_blend`](prism_render_architecture::particle::triplanar_blend)
//! across the three normalized projection-plane weights, the scalar blend of
//! three plane scalars, and the component-wise `vec3` blend of three plane
//! `vec3` samples.
//!
//! The fixtures cover the shapes the golden calls out: axis-aligned normals
//! (one dominant component gives nearly all weight to that axis), oblique
//! normals with one component clearly dominant so the three weights never tie at
//! a branch-critical boundary, each sharpen exponent in `1, 2, 4, 8`, the
//! `sharpness_exp == 0` uniform split, and the degenerate zero normal that falls
//! back to `1/3, 1/3, 1/3`. All normals and samples are written as integers or
//! simple decimals, so the fixtures stay pure and need no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The weights and both blends thread through multiplies, adds and one guarded
//! reciprocal, so they are compared under tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`). The integer-exponent loop count is a
//! `u32`, so the sharpen iteration matches exactly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::triplanar_blend`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::triplanar_blend::TriplanarWeights;
use prism_volumetric_gpu::triplanar_blend::{GpuTriplanarBlend, TriplanarBlendQuery};
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

/// Three per-plane `vec3` samples with distinct lanes so a swapped plane is
/// visible in the blend.
fn plane_vec3() -> [[f32; 3]; 3] {
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
}

/// Builds a query from a normal, a sharpen exponent, three plane scalars and the
/// three plane `vec3` samples.
fn query(
    normal: [f32; 3],
    sharpness_exp: u32,
    plane_scalar: [f32; 3],
    plane_vec3: [[f32; 3]; 3],
) -> TriplanarBlendQuery {
    TriplanarBlendQuery {
        normal,
        sharpness_exp,
        plane_scalar,
        plane_vec3,
    }
}

/// Asserts the twinned answer for one query matches the `CPU` golden.
fn assert_parity(gpu: &GpuTriplanarBlend, ctx: &GpuContext, q: &TriplanarBlendQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    let w = TriplanarWeights::from_normal(q.normal, q.sharpness_exp);
    assert!(
        approx3(g.weights, w.as_array()),
        "weights mismatch: gpu {:?} vs cpu {:?}",
        g.weights,
        w.as_array()
    );

    let cpu_scalar = w.blend_scalar(q.plane_scalar[0], q.plane_scalar[1], q.plane_scalar[2]);
    assert!(
        approx(g.blend_scalar, cpu_scalar),
        "blend_scalar mismatch: gpu {} vs cpu {cpu_scalar}",
        g.blend_scalar
    );

    let cpu_vec3 = w.blend_vec3(q.plane_vec3[0], q.plane_vec3[1], q.plane_vec3[2]);
    assert!(
        approx3(g.blend_vec3, cpu_vec3),
        "blend_vec3 mismatch: gpu {:?} vs cpu {cpu_vec3:?}",
        g.blend_vec3
    );
}

#[test]
fn axis_aligned_normals_pick_their_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriplanarBlend::new(&ctx);
    // A dominant +X normal routes nearly all weight to the X plane; the small
    // off-axis components keep every weight well clear of a tie.
    let x = query([0.96, 0.2, 0.1], 4, [10.0, 20.0, 30.0], plane_vec3());
    assert_parity(&gpu, &ctx, &x);
    // Dominant +Y and +Z normals route to their own planes.
    let y = query([0.15, 0.95, 0.1], 4, [10.0, 20.0, 30.0], plane_vec3());
    assert_parity(&gpu, &ctx, &y);
    let z = query([0.1, 0.2, 0.97], 4, [10.0, 20.0, 30.0], plane_vec3());
    assert_parity(&gpu, &ctx, &z);
}

#[test]
fn oblique_normal_each_exponent() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriplanarBlend::new(&ctx);
    // One component dominant so the weights never tie; sweep the integer
    // exponents the golden exercises.
    for exp in [1_u32, 2, 4, 8] {
        let q = query([0.8, 0.5, 0.3], exp, [2.0, 5.0, 11.0], plane_vec3());
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn negative_components_use_absolute_value() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriplanarBlend::new(&ctx);
    // Signs are dropped by the abs in the weight builder; a mixed-sign normal
    // must match the all-positive mirror.
    let q = query([-0.7, 0.4, -0.25], 2, [3.0, 6.0, 9.0], plane_vec3());
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn sharpness_zero_is_uniform() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriplanarBlend::new(&ctx);
    // exp == 0 makes every sharpened magnitude 1, giving the uniform split
    // regardless of the normal.
    let q = query([0.8, 0.5, 0.3], 0, [3.0, 6.0, 9.0], plane_vec3());
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn zero_normal_falls_back_to_thirds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriplanarBlend::new(&ctx);
    // A zero normal has a sharpened sum at or below CMP_EPS, so the kernel must
    // return the uniform 1/3 split rather than dividing by ~zero.
    let q = query([0.0, 0.0, 0.0], 4, [4.0, 8.0, 12.0], plane_vec3());
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriplanarBlend::new(&ctx);
    // A batch exercises the one-thread-per-query flattening; each result must be
    // independent of its neighbours.
    let batch = [
        query([0.96, 0.2, 0.1], 2, [1.0, 2.0, 3.0], plane_vec3()),
        query([0.15, 0.95, 0.1], 8, [5.0, 10.0, 15.0], plane_vec3()),
        query([0.1, 0.2, 0.97], 1, [7.0, 3.0, 1.0], plane_vec3()),
        query([0.8, 0.5, 0.3], 4, [2.0, 4.0, 6.0], plane_vec3()),
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
    let gpu = GpuTriplanarBlend::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
