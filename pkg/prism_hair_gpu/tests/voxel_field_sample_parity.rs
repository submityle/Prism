//! Real-device parity for the per-query voxel-field-sample twin:
//! [`GpuHairVoxelFieldSample`] must reproduce the `CPU` goldens
//! [`DensityField::sample_density`](prism_render_architecture::hair::self_collision_voxel::DensityField::sample_density)
//! and
//! [`sample_gradient`](prism_render_architecture::hair::self_collision_voxel::sample_gradient)
//! for a batch of world-space query positions against one shared splatted
//! density field, emitting one `(density, gradient)` pair per query in input
//! order.
//!
//! # Parity criterion
//!
//! Both goldens are closed-form gather/stencil evaluations (a handful of
//! add/sub/mul/divide/floor/clamp) with no chained recurrence, so the only
//! `CPU` vs `GPU` divergence is legal fused-multiply-add contraction and
//! correctly rounded division. The density and each gradient component are
//! asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3`. Every query here
//! stays clear of integer cell faces (so `floor(g)` picks the same base cell on
//! both sides) and of the grid-box faces for the interior checks (so the
//! gradient stencil clamp picks the identical pair), so a stray `fma` cannot
//! flip which cells or stencil points are read.
//!
//! The suite drives interior samples straddling a density ridge (positive
//! density and a strictly non-zero gradient along the ridge axis), a far-field
//! sample in an all-zero region (bit-exact zero density and gradient on both
//! sides, asserted on raw `f32` bit patterns), an empty batch (handled with no
//! dispatch), and a 100-query batch that crosses the 64-wide dispatch boundary.
//! The ridge sample carries strictly positive density and a non-zero gradient,
//! so neither a zero-writing nor a constant-writing no-op kernel could pass.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature. All positions are
//! explicit decimal literals — never `f32::sin`/`cos` — so test data stays
//! deterministic without introducing transcendental divergence.
//!
//! Provenance: standard trilinear voxel sampling plus a central-difference
//! stencil and a `wgpu` compute dispatch; no Unreal Engine source or derived
//! code.

use prism_hair_gpu::voxel_field_sample::{reference_voxel_field_sample, GpuHairVoxelFieldSample};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::self_collision_voxel::{
    splat_density, DensityField, HairPoint, Vec3, VoxelGrid,
};

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// True when `got` matches `want` within the fused-multiply-add tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// A `4x4x4` unit-cell grid anchored at the origin (64 cells, box `[0, 4]^3`).
fn grid() -> VoxelGrid {
    VoxelGrid::new(Vec3::new(0.0, 0.0, 0.0), 1.0, [4, 4, 4])
}

/// A density field with a mass ridge along `x`: a heavier splat into cell
/// `(1,1,1)` and a lighter one into `(2,1,1)`, built by splatting unit-centre
/// particles (frac `0`, so each deposits wholly into its own cell).
fn ridge_field() -> DensityField {
    let points = [
        HairPoint {
            position: Vec3::new(1.5, 1.5, 1.5),
            velocity: Vec3::ZERO,
            mass: 3.0,
        },
        HairPoint {
            position: Vec3::new(2.5, 1.5, 1.5),
            velocity: Vec3::ZERO,
            mass: 1.0,
        },
    ];
    splat_density(&points, grid())
}

/// Asserts every query's `(density, gradient)` matches the `CPU` goldens.
fn assert_batch_matches(gpu: &[(f32, Vec3)], field: &DensityField, g: VoxelGrid, queries: &[Vec3]) {
    assert_eq!(gpu.len(), queries.len(), "one output pair per query");
    for (i, &p) in queries.iter().enumerate() {
        let (wd, wg) = reference_voxel_field_sample(field, g, p);
        let (gd, gg) = gpu[i];
        assert!(
            close(gd, wd),
            "query {i} ({p:?}) density: gpu {gd} vs cpu {wd}",
        );
        assert!(
            close(gg.x, wg.x) && close(gg.y, wg.y) && close(gg.z, wg.z),
            "query {i} ({p:?}) gradient: gpu {gg:?} vs cpu {wg:?}",
        );
    }
}

#[test]
fn gpu_interior_ridge_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_interior_ridge_matches_cpu") else {
        return;
    };
    let twin = GpuHairVoxelFieldSample::new(&ctx);
    let g = grid();
    let field = ridge_field();
    // All off integer cell faces (pos != k + 0.5) and inside the box; they
    // straddle the x-ridge between cells (1,1,1) and (2,1,1).
    let queries = [
        Vec3::new(1.6, 1.7, 1.7),
        Vec3::new(1.8, 1.7, 1.7),
        Vec3::new(2.2, 1.7, 1.7),
        Vec3::new(1.7, 1.3, 1.8),
    ];
    let gpu = twin.eval(&ctx, g, &field.cells, &queries);
    assert_batch_matches(&gpu, &field, g, &queries);
    // The ridge interior must carry strictly positive density and a non-zero
    // gradient along x, so a zero/constant no-op kernel could not pass.
    assert!(
        gpu[0].0 > 0.0,
        "interior sample must have positive density, got {}",
        gpu[0].0,
    );
    assert!(
        gpu[0].1.x.abs() > 0.0,
        "x-ridge must give a non-zero gradient, got {:?}",
        gpu[0].1,
    );
}

#[test]
fn gpu_far_field_is_bit_exact_zero() {
    let Some(ctx) = context_or_skip("gpu_far_field_is_bit_exact_zero") else {
        return;
    };
    let twin = GpuHairVoxelFieldSample::new(&ctx);
    let g = grid();
    let field = ridge_field();
    // Deep in the empty (3,3,3) corner: every trilinear corner and every
    // gradient stencil sample reads only zero cells, so both outputs are a
    // bit-exact zero on each side.
    let queries = [Vec3::new(3.3, 3.3, 3.3)];
    let gpu = twin.eval(&ctx, g, &field.cells, &queries);
    assert_batch_matches(&gpu, &field, g, &queries);
    let (d, grad) = gpu[0];
    assert_eq!(
        d.to_bits(),
        0.0_f32.to_bits(),
        "far-field density must be bit-exact zero, got {d}",
    );
    for (axis, c) in [grad.x, grad.y, grad.z].into_iter().enumerate() {
        assert_eq!(
            c.to_bits(),
            0.0_f32.to_bits(),
            "far-field gradient axis {axis} must be bit-exact zero, got {c}",
        );
    }
}

#[test]
fn gpu_empty_batch_is_empty() {
    let Some(ctx) = context_or_skip("gpu_empty_batch_is_empty") else {
        return;
    };
    let twin = GpuHairVoxelFieldSample::new(&ctx);
    let field = ridge_field();
    let gpu = twin.eval(&ctx, grid(), &field.cells, &[]);
    assert!(gpu.is_empty(), "an empty batch must return no pairs");
}

#[test]
fn gpu_many_queries_cross_workgroup_boundary() {
    let Some(ctx) = context_or_skip("gpu_many_queries_cross_workgroup_boundary") else {
        return;
    };
    let twin = GpuHairVoxelFieldSample::new(&ctx);
    let g = grid();
    let field = ridge_field();
    // 100 queries sweep x across the box, crossing the 64-wide dispatch
    // boundary, so threads in distinct workgroups each recover their own value.
    // The 0.02 step offset by 0.61 never lands on a k + 0.5 cell face.
    let queries: Vec<Vec3> = (0..100)
        .map(|i| Vec3::new(0.61 + (i as f32) * 0.02, 1.7, 1.7))
        .collect();
    let gpu = twin.eval(&ctx, g, &field.cells, &queries);
    assert_batch_matches(&gpu, &field, g, &queries);
}
