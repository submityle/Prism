//! Real-device parity for the per-query voxel average-velocity twin:
//! [`GpuHairVoxelAvgVelocity`] must reproduce the `CPU` golden
//! [`DensityField::sample_avg_velocity`](prism_render_architecture::hair::self_collision_voxel::DensityField::sample_avg_velocity)
//! for a batch of world-space query positions against one shared splatted
//! density + velocity field, emitting one average-velocity vector per query in
//! input order.
//!
//! # Parity criterion
//!
//! The golden is a closed-form gather-and-divide (a handful of
//! add/sub/mul/divide/floor) with no chained recurrence, so the only `CPU` vs
//! `GPU` divergence is legal fused-multiply-add contraction and correctly
//! rounded division. Each velocity component is asserted to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3`. Every query here stays clear of
//! integer cell faces (so `floor(g)` picks the same base cell on both sides), so
//! a stray `fma` cannot flip which cells are read.
//!
//! The suite drives interior samples straddling a velocity ridge (positive
//! density with a strictly non-zero average velocity, so neither a zero-writing
//! nor a constant-writing no-op kernel could pass), a far-field sample in an
//! all-zero region (density `<= EPS`, so a bit-exact zero vector on both sides),
//! an empty batch (handled with no dispatch), and a 100-query batch that crosses
//! the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature. All positions and
//! velocities are explicit decimal literals — never `f32::sin`/`cos` — so test
//! data stays deterministic without introducing transcendental divergence.
//!
//! Provenance: standard trilinear voxel sampling plus a `wgpu` compute dispatch;
//! no Unreal Engine source or derived code.

use prism_hair_gpu::voxel_avg_velocity::{reference_voxel_avg_velocity, GpuHairVoxelAvgVelocity};
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

/// A field with a velocity ridge along `x`: a heavier splat moving `+x` into
/// cell `(1,1,1)` and a lighter one moving `+y` into cell `(2,1,1)`, built from
/// unit-centre particles (frac `0`, so each deposits wholly into its own cell).
/// Blending between the two cells yields a smoothly varying average velocity.
fn ridge_field() -> DensityField {
    let points = [
        HairPoint {
            position: Vec3::new(1.5, 1.5, 1.5),
            velocity: Vec3::new(2.0, 0.0, 0.0),
            mass: 3.0,
        },
        HairPoint {
            position: Vec3::new(2.5, 1.5, 1.5),
            velocity: Vec3::new(0.0, 1.0, 0.0),
            mass: 1.0,
        },
    ];
    splat_density(&points, grid())
}

/// Asserts every query's average velocity matches the `CPU` golden.
fn assert_batch_matches(gpu: &[Vec3], field: &DensityField, g: VoxelGrid, queries: &[Vec3]) {
    assert_eq!(gpu.len(), queries.len(), "one output vector per query");
    for (i, &p) in queries.iter().enumerate() {
        let want = reference_voxel_avg_velocity(field, g, p);
        let got = gpu[i];
        assert!(
            close(got.x, want.x) && close(got.y, want.y) && close(got.z, want.z),
            "query {i} ({p:?}) avg velocity: gpu {got:?} vs cpu {want:?}",
        );
    }
}

#[test]
fn gpu_interior_ridge_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_interior_ridge_matches_cpu") else {
        return;
    };
    let twin = GpuHairVoxelAvgVelocity::new(&ctx);
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
    let gpu = twin.eval(&ctx, g, &field.cells, &field.velocity, &queries);
    assert_batch_matches(&gpu, &field, g, &queries);
    // The ridge interior must carry a strictly non-zero average velocity, so a
    // zero-writing no-op kernel could not pass.
    let v = gpu[0];
    assert!(
        v.x.abs() + v.y.abs() + v.z.abs() > 0.0,
        "interior sample must have non-zero average velocity, got {v:?}",
    );
}

#[test]
fn gpu_far_field_is_bit_exact_zero() {
    let Some(ctx) = context_or_skip("gpu_far_field_is_bit_exact_zero") else {
        return;
    };
    let twin = GpuHairVoxelAvgVelocity::new(&ctx);
    let g = grid();
    let field = ridge_field();
    // Deep in the empty (3,3,3) corner: every trilinear corner reads only zero
    // cells, so density is zero (<= EPS) and the average velocity is a bit-exact
    // zero vector on each side.
    let queries = [Vec3::new(3.3, 3.3, 3.3)];
    let gpu = twin.eval(&ctx, g, &field.cells, &field.velocity, &queries);
    assert_batch_matches(&gpu, &field, g, &queries);
    let v = gpu[0];
    for (axis, c) in [v.x, v.y, v.z].into_iter().enumerate() {
        assert_eq!(
            c.to_bits(),
            0.0_f32.to_bits(),
            "far-field average velocity axis {axis} must be bit-exact zero, got {c}",
        );
    }
}

#[test]
fn gpu_empty_batch_is_empty() {
    let Some(ctx) = context_or_skip("gpu_empty_batch_is_empty") else {
        return;
    };
    let twin = GpuHairVoxelAvgVelocity::new(&ctx);
    let field = ridge_field();
    let gpu = twin.eval(&ctx, grid(), &field.cells, &field.velocity, &[]);
    assert!(gpu.is_empty(), "an empty batch must return no vectors");
}

#[test]
fn gpu_many_queries_cross_workgroup_boundary() {
    let Some(ctx) = context_or_skip("gpu_many_queries_cross_workgroup_boundary") else {
        return;
    };
    let twin = GpuHairVoxelAvgVelocity::new(&ctx);
    let g = grid();
    let field = ridge_field();
    // 100 queries sweep x across the box, crossing the 64-wide dispatch
    // boundary, so threads in distinct workgroups each recover their own value.
    // The 0.02 step offset by 0.61 never lands on a k + 0.5 cell face.
    let queries: Vec<Vec3> = (0..100)
        .map(|i| Vec3::new(0.61 + (i as f32) * 0.02, 1.7, 1.7))
        .collect();
    let gpu = twin.eval(&ctx, g, &field.cells, &field.velocity, &queries);
    assert_batch_matches(&gpu, &field, g, &queries);
}
