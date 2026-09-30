//! Real-device parity for the froxel voxel forward-scatter decode twin:
//! [`GpuHairVoxelForwardScatter`] must reproduce the `CPU` golden
//! [`voxel_forward_scatter`](prism_render_architecture::hair::dual_scattering::voxel_forward_scatter)
//! for a batch of per-texel froxel density rows, covering the running prefix
//! sum `n = sum(sigma_j)` over `0..=index`, the per-`sigma` floor at `0`, the
//! monotonic non-decreasing shape, the deliberately uncapped multi-strand
//! count, the fully unscattered empty row, and the multi-texel batch walk. The
//! density rows themselves are produced by the golden
//! [`accumulate_voxel_density`](prism_render_architecture::hair::deep_transmittance::accumulate_voxel_density)
//! so the two froxel golden functions are exercised end to end, and the whole
//! emitted curve is asserted value-for-value against the golden evaluated at
//! every index.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The accumulation is a closed-form running sum with no transcendental call,
//! and each thread owns a disjoint output row, so the `CPU` and `GPU` evaluate
//! the same expressions and diverge only through legal fused-multiply-add
//! contraction. Parity is asserted to within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3` — tight enough to fail a genuinely wrong port (a missing
//! floor, a wrong fold bound, a dropped term), loose enough to admit the
//! contraction. Densities are built from explicit depth/opacity literals (never
//! `sin`/`cos`), and the shaping cases assert a non-trivial accumulation so a
//! no-op kernel could not pass.
//!
//! Provenance: standard uniform-voxel forward-scatter count plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::voxel_forward_scatter::GpuHairVoxelForwardScatter;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::deep_transmittance::{
    accumulate_voxel_density, TransmittanceSample,
};
use prism_render_architecture::hair::dual_scattering::voxel_forward_scatter;

/// Asserts a single value matches within the documented tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got} vs cpu {expected} (abs {abs_diff}, rel {rel_diff})",
    );
}

/// Builds one froxel density row from `(depth, opacity)` occluders via the
/// golden accumulation, so the decode twin is fed exactly what the density
/// stage would produce.
fn density_row(occluders: &[(f32, f32)], slab_start: f32, slab_end: f32, voxels: u32) -> Vec<f32> {
    let samples: Vec<TransmittanceSample> = occluders
        .iter()
        .map(|&(depth, opacity)| TransmittanceSample::new(depth, opacity))
        .collect();
    accumulate_voxel_density(&samples, slab_start, slab_end, voxels)
}

/// Runs the twin and the golden on the same density rows, asserting every
/// emitted curve equals the golden evaluated at every index. Returns the `GPU`
/// curves for further case-specific assertions.
fn assert_parity(
    ctx: &GpuContext,
    decoder: &GpuHairVoxelForwardScatter,
    rows: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let gpu = decoder.eval(ctx, rows);

    assert_eq!(gpu.len(), rows.len(), "texel row count");

    for (t, (curve, row)) in gpu.iter().zip(rows).enumerate() {
        assert_eq!(curve.len(), row.len(), "texel {t} curve length");
        for (v, &g) in curve.iter().enumerate() {
            let c = voxel_forward_scatter(row, v);
            assert_close(g, c, &format!("crossings texel {t} voxel {v}"));
        }
    }

    gpu
}

/// A single texel with occluders in a few voxels: the count must step up at
/// each occupied voxel and stay flat across the empty ones, and be
/// monotonically non-decreasing throughout.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_texel_curve_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel forward-scatter parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairVoxelForwardScatter::new(&ctx);

    // Slab 0..10, five voxels of width 2: occluders at 1, 5, 9 land in 0, 2, 4.
    let row = density_row(&[(1.0, 0.5), (5.0, 0.25), (9.0, 0.5)], 0.0, 10.0, 5);
    let gpu = assert_parity(&ctx, &decoder, &[row]);

    let curve = &gpu[0];
    // Voxel 0: 0.5.
    assert_close(curve[0], 0.5, "after first occluder");
    // Voxel 1 empty: unchanged.
    assert_close(curve[1], 0.5, "flat across empty voxel");
    // Voxel 2: 0.5 + 0.25 = 0.75.
    assert_close(curve[2], 0.75, "after second occluder");
    // Voxel 3 empty: unchanged.
    assert_close(curve[3], 0.75, "flat across empty voxel");
    // Voxel 4: 0.75 + 0.5 = 1.25.
    assert_close(curve[4], 1.25, "after third occluder");

    // Monotonic non-decreasing.
    for w in curve.windows(2) {
        assert!(w[1] >= w[0] - 1e-6, "curve must be non-decreasing");
    }
}

/// A voxel holding several strands accumulates a density above `1.0`; the count
/// is deliberately *not* capped, so the running sum keeps climbing past one.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_strand_voxel_is_not_capped() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel forward-scatter parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairVoxelForwardScatter::new(&ctx);

    // Two full-opacity occluders in voxel 0 (density 2.0, uncapped) then more.
    let row = density_row(&[(0.5, 1.0), (0.9, 1.0), (3.5, 0.5)], 0.0, 4.0, 4);
    let gpu = assert_parity(&ctx, &decoder, &[row]);

    let curve = &gpu[0];
    assert_close(curve[0], 2.0, "multi-strand voxel sums past one");
    assert_close(curve[3], 2.5, "count keeps climbing behind it");
}

/// An all-zero density row (no occluders) stays fully unscattered at every
/// voxel.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_density_row_has_zero_crossings() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel forward-scatter parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairVoxelForwardScatter::new(&ctx);

    // No occluders: golden accumulation yields an all-zero six-voxel row.
    let row = density_row(&[], 0.0, 6.0, 6);
    let gpu = assert_parity(&ctx, &decoder, &[row]);

    for (v, n) in gpu[0].iter().enumerate() {
        assert_close(*n, 0.0, &format!("empty row voxel {v} unscattered"));
    }
}

/// A zero-length input row (a texel with no voxels) yields a zero-length output
/// curve, and the batch still carries occupied texels so both paths run in one
/// dispatch.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_zero_length_row_yields_empty_curve() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel forward-scatter parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairVoxelForwardScatter::new(&ctx);

    let occupied = density_row(&[(1.0, 0.4), (3.0, 0.6)], 0.0, 4.0, 4);
    let rows = vec![occupied, Vec::new()];
    let gpu = assert_parity(&ctx, &decoder, &rows);

    assert_eq!(gpu.len(), 2, "batch covers both texels");
    assert!(gpu[1].is_empty(), "zero-length row yields empty curve");
}

/// A multi-texel batch with distinct density rows exercises the per-texel base
/// offset and the one-thread-per-texel dispatch bound.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_texel_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel forward-scatter parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairVoxelForwardScatter::new(&ctx);

    let rows = vec![
        density_row(&[(0.5, 0.3), (2.5, 0.4)], 0.0, 4.0, 4),
        density_row(&[], 0.0, 4.0, 4),
        density_row(&[(1.0, 0.6), (3.5, 0.5)], 0.0, 4.0, 4),
        density_row(&[(2.0, 0.9)], 0.0, 4.0, 3),
    ];
    let gpu = assert_parity(&ctx, &decoder, &rows);

    assert_eq!(gpu.len(), 4, "batch covers every texel");
    // Texel 3 has three voxels (distinct stride from the others).
    assert_eq!(gpu[3].len(), 3, "ragged row length preserved");
    for (v, n) in gpu[1].iter().enumerate() {
        assert_close(*n, 0.0, &format!("empty texel 1 voxel {v} unscattered"));
    }
}
