//! Real-device parity for the frustum-plane-extraction twin:
//! [`GpuFrustumPlaneExtract`](prism_volumetric_gpu::frustum_plane_extract::GpuFrustumPlaneExtract)
//! must reproduce the `CPU` golden
//! [`frustum_plane_extract`](prism_render_architecture::particle::frustum_plane_extract)
//! across the identity matrix (the canonical axis-aligned clip cube), a
//! symmetric perspective frustum, an asymmetric orthographic box, and a
//! randomized batch of well-conditioned perspective and orthographic matrices
//! compared plane-for-plane and coefficient-for-coefficient.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each matrix is a fixed, non-reorderable sequence of adds, multiplies, a
//! divide and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in
//! the same order. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The comparison therefore allows `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` on every `f32` plane coefficient.
//!
//! # Conditioning
//!
//! Every fixture is kept well away from the single degeneracy crack: the random
//! batch rejects any projection whose view extents are near zero, so each of
//! the six extracted normals has a squared length vastly above the normalize
//! epsilon and `CPU` and `GPU` stay on the same side of the degenerate branch
//! regardless of a few units in the last place of slack. The fixtures use only
//! division inside the projection constructors, so no transcendental appears.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::frustum_plane_extract`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::frustum_plane_extract::{FrustumPlanes, Mat4};
use prism_volumetric_gpu::frustum_plane_extract::{golden, GpuFrustumPlaneExtract};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
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

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Builds a well-conditioned perspective matrix by rejection sampling: the view
/// extents are drawn wide and accepted only when every span is comfortably
/// positive, so none of the six extracted normals approaches the degenerate
/// floor. Uses only the division inside [`Mat4::perspective`].
fn rand_perspective(state: &mut u64) -> Mat4 {
    loop {
        let l = -(lcg(state) * 1.5 + 0.25);
        let r = lcg(state) * 1.5 + 0.25;
        let b = -(lcg(state) * 1.5 + 0.25);
        let t = lcg(state) * 1.5 + 0.25;
        let n = lcg(state) * 2.0 + 0.5;
        let f = n + (lcg(state) * 40.0 + 5.0);
        if (r - l) < 0.5 || (t - b) < 0.5 || (f - n) < 0.5 {
            continue;
        }
        return Mat4::perspective(l, r, b, t, n, f);
    }
}

/// Builds a well-conditioned orthographic matrix by rejection sampling, mirror
/// of [`rand_perspective`] for the parallel-projection path. Uses only the
/// division inside [`Mat4::orthographic`].
fn rand_orthographic(state: &mut u64) -> Mat4 {
    loop {
        let l = -(lcg(state) * 3.0 + 0.5);
        let r = lcg(state) * 3.0 + 0.5;
        let b = -(lcg(state) * 3.0 + 0.5);
        let t = lcg(state) * 3.0 + 0.5;
        let n = lcg(state) * 2.0 + 0.5;
        let f = n + (lcg(state) * 40.0 + 5.0);
        if (r - l) < 0.5 || (t - b) < 0.5 || (f - n) < 0.5 {
            continue;
        }
        return Mat4::orthographic(l, r, b, t, n, f);
    }
}

/// Pins one `GPU` [`FrustumPlanes`] block against the `CPU` golden for `mat`:
/// all six planes and all four coefficients of each must agree within bound.
fn pin(idx: usize, mat: &Mat4, got: &FrustumPlanes) {
    let want = golden(mat);
    for (plane_idx, (g, w)) in got.planes.iter().zip(want.planes.iter()).enumerate() {
        assert!(
            close(g.nx, w.nx),
            "matrix {idx} plane {plane_idx} nx: gpu {} vs cpu {}",
            g.nx,
            w.nx
        );
        assert!(
            close(g.ny, w.ny),
            "matrix {idx} plane {plane_idx} ny: gpu {} vs cpu {}",
            g.ny,
            w.ny
        );
        assert!(
            close(g.nz, w.nz),
            "matrix {idx} plane {plane_idx} nz: gpu {} vs cpu {}",
            g.nz,
            w.nz
        );
        assert!(
            close(g.d, w.d),
            "matrix {idx} plane {plane_idx} d: gpu {} vs cpu {}",
            g.d,
            w.d
        );
    }
}

/// Dispatches `matrices` and pins every result against the `CPU` golden.
fn check(ctx: &GpuContext, gpu: &GpuFrustumPlaneExtract, matrices: &[Mat4]) {
    let got = gpu.eval(ctx, matrices);
    assert_eq!(
        got.len(),
        matrices.len(),
        "result count must match the input count"
    );
    for (idx, (mat, planes)) in matrices.iter().zip(got.iter()).enumerate() {
        pin(idx, mat, planes);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumPlaneExtract::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn identity_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumPlaneExtract::new(&ctx);
    // The identity matrix yields the canonical [-1, 1]^3 clip cube with
    // axis-aligned, unit-length planes; integer geometry is exact on both
    // devices.
    check(&ctx, &gpu, &[Mat4::identity()]);
}

#[test]
fn symmetric_perspective_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumPlaneExtract::new(&ctx);
    // A symmetric perspective frustum exercises the oblique side planes and the
    // near/far split.
    let mat = Mat4::perspective(-1.0, 1.0, -1.0, 1.0, 1.0, 100.0);
    check(&ctx, &gpu, &[mat]);
}

#[test]
fn asymmetric_orthographic_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumPlaneExtract::new(&ctx);
    // An asymmetric orthographic box exercises the non-zero plane constants on
    // every axis.
    let mat = Mat4::orthographic(-2.0, 3.0, -1.0, 4.0, 0.5, 50.0);
    check(&ctx, &gpu, &[mat]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumPlaneExtract::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random projections,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned matrix-for-matrix.
    let mut matrices = vec![
        Mat4::identity(),
        Mat4::perspective(-1.0, 1.0, -1.0, 1.0, 1.0, 100.0),
        Mat4::orthographic(-2.0, 3.0, -1.0, 4.0, 0.5, 50.0),
    ];
    for _ in 0..24 {
        matrices.push(rand_perspective(&mut state));
        matrices.push(rand_orthographic(&mut state));
    }
    check(&ctx, &gpu, &matrices);
}

#[test]
fn many_matrices_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumPlaneExtract::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins all six planes across many
    // random perspective projections.
    let matrices: Vec<Mat4> = (0..200).map(|_| rand_perspective(&mut state)).collect();
    check(&ctx, &gpu, &matrices);
}
