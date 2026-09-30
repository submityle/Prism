//! Real-device parity: the `GPU` free-surface velocity extrapolation must
//! reproduce the `CPU` golden twin's grown field within a tight tolerance.
//!
//! Each Jacobi sweep only ever copies a known face forward or writes the mean
//! of a face's known neighbours, so the sole floating-point operation is the
//! single division per filled face. There is no iterative reassociation to
//! bound as in the pressure solve, so a small relative tolerance suffices; the
//! integer-exact known-mask growth is otherwise identical between the engines.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / ping-pong dispatch / readback path on any
//! machine with a real device.
//!
//! Provenance: iterative velocity extrapolation from the known band into the
//! air region is a standard free-surface technique (Bridson; Zhu and Bridson
//! 2005). No Unreal Engine source or derived code.

use prism_physics_gpu::{
    fluid::cpu::extrapolate::{extrapolate_axis, AxisDims},
    GpuContext, GpuExtrapolate,
};

/// Builds a partially known face field: an off-centre solid block of known
/// faces carrying a spatially varying velocity, surrounded by unknown air
/// faces. This exercises both the copy-forward path (known faces) and the
/// neighbour-averaging path (the air band around the block).
fn seeded_surface(dims: AxisDims) -> (Vec<f32>, Vec<f32>) {
    let mut field = vec![0.0f32; dims.count()];
    let mut weights = vec![0.0f32; dims.count()];
    for k in 2..5 {
        for j in 3..6 {
            for i in 1..4 {
                let id = dims.flat(i, j, k);
                field[id] = 0.25 + (i as f32) * 0.5 - (j as f32) * 0.3 + (k as f32) * 0.15;
                weights[id] = 1.0;
            }
        }
    }
    (field, weights)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_extrapolate_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU extrapolation parity: no wgpu adapter on this host");
        return;
    };
    let extrap = GpuExtrapolate::new(&ctx);
    // A non-cubic axis field so the flat indexing is exercised on every axis.
    let dims = AxisDims::new(8, 9, 7);
    let iterations = 6;
    let (field, weights) = seeded_surface(dims);

    // CPU golden.
    let mut cpu_field = field.clone();
    extrapolate_axis(&mut cpu_field, &weights, dims, iterations);

    // GPU.
    let mut gpu_field = field.clone();
    extrap
        .extrapolate_axis(&ctx, dims, &mut gpu_field, &weights, iterations)
        .expect("gpu extrapolate");

    // Only the single per-face division is floating point, so a tight relative
    // tolerance holds everywhere.
    let mut max_rel = 0.0f32;
    for (idx, (c, g)) in cpu_field.iter().zip(gpu_field.iter()).enumerate() {
        let denom = c.abs().max(1.0e-4);
        let rel = (c - g).abs() / denom;
        assert!(
            rel < 5.0e-3,
            "face {idx} differs: cpu {c} vs gpu {g} (rel {rel})"
        );
        max_rel = max_rel.max(rel);
    }

    // Sanity: the band actually grew past the seeded block into former air.
    // (4, 4, 3) is one face outside the seeded block (known `i` spans 1..4),
    // so six sweeps must have filled it from the band.
    let air = dims.flat(4, 4, 3);
    assert!(
        gpu_field[air] != 0.0,
        "expected extrapolation to reach the air face next to the block"
    );
    eprintln!("gpu extrapolation parity: max relative error {max_rel}");
}
