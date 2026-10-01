//! Real-device parity for the view-space normal-reconstruction twin:
//! [`GpuNormalReconstruct`](prism_volumetric_gpu::normal_reconstruct::GpuNormalReconstruct)
//! must reproduce the `CPU` golden
//! [`particle::normal_reconstruct`](prism_render_architecture::particle::normal_reconstruct)
//! across several surface configurations, projection parameters and depth
//! neighborhoods.
//!
//! The tests skip (with a printed notice on the first) when the host has no
//! `wgpu` adapter, so the suite stays green everywhere while still exercising
//! the full dispatch-and-readback on any real device such as an Apple
//! `M`-series `GPU`. The kernel is portable core-`WGSL`, so it needs no optional
//! device feature.
//!
//! # Parity criterion
//!
//! Each normal is an affine reprojection plus one cross product and a guarded
//! normalization, with no transcendental call and no reorderable reduction, so
//! `CPU` and `GPU` evaluate the same closed form in the same order. The
//! comparison allows `abs_diff <= 1e-5` or `rel_diff <= 1e-5` — loose enough to
//! admit a legal fused multiply-add contraction yet tight enough to fail a
//! wrong port (a swapped tap, a flipped cross product, a dropped four-tap
//! selection, a missing normalization guard). Every scenario compares both the
//! naive and the improved normal against the reference, and the discontinuity
//! and random scenes additionally assert the two differ or that a non-trivial
//! normal appears, so a degenerate all-zero kernel could not pass.
//!
//! Provenance: standard Valve/Unreal improved depth-to-normal reconstruction;
//! no Unreal Engine source or derived code.

use prism_render_architecture::particle::normal_reconstruct::{
    reconstruct_normal_improved, reconstruct_normal_naive, DepthTaps, NormalReconstructParams,
};
use prism_volumetric_gpu::normal_reconstruct::{GpuNormalReconstruct, NormalQuery, NormalResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute/relative parity bound. A `GPU` may fuse a multiply-add the scalar
/// reference leaves separate, perturbing the low mantissa bits by a few units
/// in the last place; `1e-5` admits that legal slack while still failing a
/// genuinely wrong port.
const EPS: f32 = 1.0e-5;

/// Floor keeping the relative-error denominator away from zero.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= EPS
}

/// Asserts two 3-vectors agree component-wise within [`close`].
fn close_vec(a: [f32; 3], b: [f32; 3], what: &str, idx: usize) {
    assert!(
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2]),
        "{what} mismatch at query {idx}: gpu ({}, {}, {}), cpu ({}, {}, {})",
        a[0],
        a[1],
        a[2],
        b[0],
        b[1],
        b[2]
    );
}

/// Runs `queries` through the twin and asserts every naive and improved normal
/// matches the `CPU` golden, returning the device results for extra assertions.
fn check_queries(
    ctx: &GpuContext,
    gpu: &GpuNormalReconstruct,
    params: NormalReconstructParams,
    queries: &[NormalQuery],
) -> Vec<NormalResult> {
    let results = gpu.eval(ctx, params, queries);
    assert_eq!(results.len(), queries.len());
    for (idx, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        let cpu_naive = reconstruct_normal_naive(
            q.uv,
            q.texel,
            q.taps.center,
            q.taps.right,
            q.taps.up,
            &params,
        );
        let cpu_improved = reconstruct_normal_improved(q.uv, q.texel, &q.taps, &params);
        close_vec(r.naive, cpu_naive, "naive", idx);
        close_vec(r.improved, cpu_improved, "improved", idx);
    }
    results
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg_unit(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1).
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Dot product, used only to compare how well a normal aligns with a truth.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[test]
fn gpu_matches_cpu_on_front_facing_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNormalReconstruct::new(&ctx);
    // A plane squarely facing the camera: constant depth everywhere, so both
    // reconstructions must return the +Z view-space normal.
    let params = NormalReconstructParams::new(1.0, 1.0);
    let queries = [NormalQuery {
        uv: [0.5, 0.5],
        texel: [0.02, 0.02],
        // Equal depths on every tap.
        taps: DepthTaps::new(4.0, 4.0, 4.0, 4.0, 4.0),
    }];
    let results = check_queries(&ctx, &gpu, params, &queries);
    close_vec(results[0].naive, [0.0, 0.0, 1.0], "front naive", 0);
    close_vec(results[0].improved, [0.0, 0.0, 1.0], "front improved", 0);
}

#[test]
fn gpu_matches_cpu_on_tilted_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNormalReconstruct::new(&ctx);
    // A plane whose depth increases to the right and up; the reconstructed
    // normal must still face the camera (+z) and match the reference exactly.
    let params = NormalReconstructParams::new(1.0, 1.0);
    let queries = [NormalQuery {
        uv: [0.5, 0.5],
        texel: [0.05, 0.05],
        // Smoothly varying depths (no discontinuity), so naive == improved.
        taps: DepthTaps::new(4.0, 3.7, 4.3, 3.8, 4.2),
    }];
    let results = check_queries(&ctx, &gpu, params, &queries);
    assert!(
        results[0].improved[2] > 0.0,
        "tilted-plane improved normal should face the camera, got {}",
        results[0].improved[2]
    );
}

#[test]
fn gpu_matches_cpu_at_depth_discontinuity_tap_selection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNormalReconstruct::new(&ctx);
    // A silhouette edge: the left/down neighbors lie on the facing plane while
    // the right/up neighbors jump far into the background. The improved four-tap
    // selection must keep the near left/down neighbors (truth is +Z); the naive
    // forward difference trusts the far jump and skews away. Depths are widely
    // separated so the tap-selection comparison is unambiguous on both backends.
    let params = NormalReconstructParams::new(1.0, 1.0);
    let truth = [0.0, 0.0, 1.0];
    let queries = [NormalQuery {
        uv: [0.5, 0.5],
        texel: [0.05, 0.05],
        // center/left/right/down/up: near facing plane on left+down, far on
        // right+up.
        taps: DepthTaps::new(4.0, 4.0, 24.0, 4.0, 24.0),
    }];
    let results = check_queries(&ctx, &gpu, params, &queries);
    // The improved normal should keep the facing plane and beat naive alignment.
    let naive_align = dot3(results[0].naive, truth);
    let improved_align = dot3(results[0].improved, truth);
    assert!(
        improved_align > naive_align,
        "improved ({improved_align}) should align with +Z better than naive ({naive_align})"
    );
    close_vec(results[0].improved, truth, "discontinuity improved", 0);
}

#[test]
fn gpu_matches_cpu_across_projection_parameters() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNormalReconstruct::new(&ctx);
    // The same depth neighborhood reconstructed under several frustums,
    // including a non-square (anisotropic) field of view and a very wide one.
    let param_sets = [
        NormalReconstructParams::new(1.0, 1.0),
        NormalReconstructParams::new(1.6, 0.9),
        NormalReconstructParams::new(0.4, 0.4),
        NormalReconstructParams::new(2.5, 1.3),
    ];
    let queries = [
        NormalQuery {
            uv: [0.3, 0.7],
            texel: [0.03, 0.04],
            taps: DepthTaps::new(5.0, 4.8, 5.3, 4.7, 5.2),
        },
        NormalQuery {
            uv: [0.8, 0.2],
            texel: [0.02, 0.02],
            taps: DepthTaps::new(2.0, 2.1, 1.95, 2.2, 1.9),
        },
    ];
    for params in param_sets {
        check_queries(&ctx, &gpu, params, &queries);
    }
}

#[test]
fn gpu_matches_cpu_on_random_depth_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNormalReconstruct::new(&ctx);
    // A batch of random pixel queries with random depths and texel steps. To
    // keep the four-tap selection deterministic across backends, each axis gets
    // a clear near/far margin (one side close to center, the other far), so the
    // tie-break comparison can never straddle a floating-point boundary.
    let params = NormalReconstructParams::new(1.3, 0.95);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = Vec::new();
    for _ in 0..64 {
        let uv = [
            0.1 + lcg_unit(&mut state) * 0.8,
            0.1 + lcg_unit(&mut state) * 0.8,
        ];
        let texel = [
            0.01 + lcg_unit(&mut state) * 0.04,
            0.01 + lcg_unit(&mut state) * 0.04,
        ];
        let center = 2.0 + lcg_unit(&mut state) * 6.0;
        // Near neighbors stay within a small band of center; far neighbors are
        // pushed well beyond it. A per-query bit decides which side is near.
        let near_x = center + (lcg_unit(&mut state) - 0.5) * 0.2;
        let near_y = center + (lcg_unit(&mut state) - 0.5) * 0.2;
        let far = center + 15.0 + lcg_unit(&mut state) * 5.0;
        let flip_x = lcg_unit(&mut state) < 0.5;
        let flip_y = lcg_unit(&mut state) < 0.5;
        let (left, right) = if flip_x {
            (near_x, far)
        } else {
            (far, near_x)
        };
        let (down, up) = if flip_y { (near_y, far) } else { (far, near_y) };
        queries.push(NormalQuery {
            uv,
            texel,
            taps: DepthTaps::new(center, left, right, down, up),
        });
    }
    let results = check_queries(&ctx, &gpu, params, &queries);
    // At least one reconstructed normal must be non-trivial (non-zero length),
    // so a degenerate all-zero kernel could not pass this scene.
    let any_nontrivial = results.iter().any(|r| {
        let n = r.improved;
        (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) > 0.5
    });
    assert!(any_nontrivial, "random field should yield non-trivial normals");
}

#[test]
fn gpu_matches_cpu_on_degenerate_depth_collapses_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNormalReconstruct::new(&ctx);
    // All depths identical and a zero texel step: both edge vectors vanish, the
    // cross product collapses and the normalization guard must return the zero
    // vector exactly, matching the reference guard. Also a finiteness check so a
    // NaN from an unguarded divide could not sneak through.
    let params = NormalReconstructParams::new(1.0, 1.0);
    let queries = [NormalQuery {
        uv: [0.5, 0.5],
        texel: [0.0, 0.0],
        taps: DepthTaps::new(4.0, 4.0, 4.0, 4.0, 4.0),
    }];
    let results = check_queries(&ctx, &gpu, params, &queries);
    close_vec(results[0].improved, [0.0, 0.0, 0.0], "degenerate improved", 0);
    close_vec(results[0].naive, [0.0, 0.0, 0.0], "degenerate naive", 0);
    for component in results[0].improved {
        assert!(component.is_finite(), "degenerate output must stay finite");
    }
}

#[test]
fn empty_queries_yield_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNormalReconstruct::new(&ctx);
    let out = gpu.eval(&ctx, NormalReconstructParams::new(1.0, 1.0), &[]);
    assert!(out.is_empty(), "an empty query slice yields no results");
}
