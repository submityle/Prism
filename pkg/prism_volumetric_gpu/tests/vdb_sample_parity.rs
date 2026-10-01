//! Real-device parity for the sparse-`VDB` density twin: [`GpuVdbSample`] must
//! reproduce the `CPU` golden
//! [`sample_vdb_density`](prism_render_architecture::particle::vdb_volume_sample::sample_vdb_density)
//! (design `docs/prism_particle_engine_design_zh.md` §8.3) for every query
//! point across empty trees, direct voxel hits, inactive-neighbour misses,
//! trilinear midpoint averaging, cross-leaf-boundary continuity, out-of-domain
//! reads and multi-root-slot addressing.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! upload-dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL` (integer shifts, masks, compares plus `floor` and
//! multiply/add), so it needs no optional device feature and runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! # Parity criterion
//!
//! Each corner lookup is an exact integer tree descent (no floating point) and
//! the trilinear blend applies the identical multiply/add sequence as the
//! reference, so `CPU` and `GPU` evaluate the same closed-form algebra. Values
//! are asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-4` — far tighter
//! than any physically meaningful density difference, yet enough to fail a
//! wrong port (a swapped shift, a dropped active-mask test, a misordered corner
//! weight). Several scenes also assert that an active voxel reads back a value
//! distinct from the `background`, so a degenerate constant kernel could not
//! pass.
//!
//! Provenance: standard `OpenVDB` / `NanoVDB` sparse-tree descent plus a
//! trilinear reconstruction and `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

use prism_render_architecture::particle::vdb_volume_sample::{sample_vdb_density, VdbTree};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::{GpuContext, GpuVdbSample};

/// Parity tolerance: a density matches when it is within this absolute error or
/// this relative error of the reference. Chosen to absorb a fused multiply-add
/// the scalar reference leaves separate while still rejecting a wrong port.
const TOL: f32 = 1e-4;

/// Floor used in the relative-error denominator so a near-zero reference does
/// not inflate the ratio.
const REL_FLOOR: f32 = 1e-6;

/// A per-component threshold proving an active voxel reads back a value clearly
/// distinct from the `background` (non-degeneracy guard).
const DISTINCT: f32 = 1e-2;

/// Asserts every `gpu` density matches the `CPU` golden
/// [`sample_vdb_density`] at the matching point to within [`TOL`].
fn assert_parity(tree: &VdbTree, points: &[Vec3], gpu: &[f32]) {
    assert_eq!(gpu.len(), points.len(), "one density per query point");
    for (i, &p) in points.iter().enumerate() {
        let exp = sample_vdb_density(tree, p);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(REL_FLOOR);
        assert!(
            abs_diff < TOL || rel_diff < TOL,
            "density mismatch at point {i} {p:?}: gpu {got}, cpu {exp} (abs {abs_diff}, rel {rel_diff})"
        );
    }
}

/// Builds a tree, failing loudly if the root dimensions are degenerate (every
/// test picks a valid non-zero extent).
fn tree_with(root_dims: [u32; 3], background: f32) -> VdbTree {
    VdbTree::new(root_dims, background).expect("non-zero root dimensions build a tree")
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_tree_reads_background_everywhere() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping vdb-sample parity: no wgpu adapter on this host");
        return;
    };
    let sampler = GpuVdbSample::new(&ctx);

    // A fully sparse tree: every query, on-grid or fractional, reads the uniform
    // background with no allocated node to descend into.
    let background = 0.25_f32;
    let tree = tree_with([1, 1, 1], background);
    let points = vec![
        Vec3::ZERO,
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(3.0, 4.0, 5.0),
        Vec3::new(7.5, 7.5, 7.5),
        Vec3::new(31.0, 31.0, 31.0),
    ];

    let gpu = sampler.eval(&ctx, &tree, &points);
    assert_parity(&tree, &points, &gpu);
    for &density in &gpu {
        let diff = (density - background).abs();
        assert!(
            diff < TOL,
            "empty tree must read background {background}, got {density}"
        );
    }
}

#[test]
fn empty_points_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sampler = GpuVdbSample::new(&ctx);
    let tree = tree_with([1, 1, 1], 0.0);
    let out = sampler.eval(&ctx, &tree, &[]);
    assert!(out.is_empty(), "an empty point slice yields no densities");
}

#[test]
fn direct_hit_and_inactive_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sampler = GpuVdbSample::new(&ctx);

    // A single active voxel in an otherwise sparse tree. Sampling exactly on its
    // integer coordinate returns its value (every fractional weight is zero, so
    // only that corner contributes); an on-grid neighbour that was never written
    // is inactive and reads the background.
    let background = 0.1_f32;
    let hit_value = 0.875_f32;
    let mut tree = tree_with([1, 1, 1], background);
    assert!(tree.set_voxel([2, 3, 4], hit_value), "voxel is in domain");

    let hit = Vec3::new(2.0, 3.0, 4.0);
    let miss = Vec3::new(10.0, 11.0, 12.0);
    let points = vec![hit, miss];

    let gpu = sampler.eval(&ctx, &tree, &points);
    assert_parity(&tree, &points, &gpu);

    // Non-degeneracy: the hit reads the voxel value, clearly distinct from the
    // background, and the miss reads the background.
    assert!(
        (gpu[0] - hit_value).abs() < TOL,
        "on-grid hit must read the voxel value {hit_value}, got {}",
        gpu[0]
    );
    assert!(
        (gpu[0] - background).abs() > DISTINCT,
        "the hit must differ from the background (non-degenerate kernel)"
    );
    assert!(
        (gpu[1] - background).abs() < TOL,
        "an inactive on-grid neighbour must read the background, got {}",
        gpu[1]
    );
}

#[test]
fn trilinear_midpoint_averages_two_voxels() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sampler = GpuVdbSample::new(&ctx);

    // Two active voxels adjacent along X; the midpoint between them averages the
    // pair while the Y/Z fractional weights stay zero (so the other, inactive
    // corners never contribute).
    let background = 0.0_f32;
    let left = 1.0_f32;
    let right = 3.0_f32;
    let mut tree = tree_with([1, 1, 1], background);
    assert!(tree.set_voxel([0, 0, 0], left));
    assert!(tree.set_voxel([1, 0, 0], right));

    let midpoint = Vec3::new(0.5, 0.0, 0.0);
    let points = vec![
        Vec3::new(0.0, 0.0, 0.0),
        midpoint,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.25, 0.0, 0.0),
    ];

    let gpu = sampler.eval(&ctx, &tree, &points);
    assert_parity(&tree, &points, &gpu);

    // Non-degeneracy: the midpoint is the average of the two voxel values.
    let expected_mid = 0.5 * (left + right);
    assert!(
        (gpu[1] - expected_mid).abs() < TOL,
        "midpoint must average the two voxels to {expected_mid}, got {}",
        gpu[1]
    );
}

#[test]
fn cross_leaf_boundary_is_continuous() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sampler = GpuVdbSample::new(&ctx);

    // Voxels 7 and 8 straddle the leaf boundary at 8 voxels per axis: 7 is the
    // last voxel of leaf 0, 8 the first of leaf 1. A sample at 7.5 must average
    // them across that boundary, proving the descent re-enters a different leaf
    // for the far corner.
    let background = 0.0_f32;
    let inside = 2.0_f32;
    let outside = 6.0_f32;
    let mut tree = tree_with([1, 1, 1], background);
    assert!(tree.set_voxel([7, 0, 0], inside));
    assert!(tree.set_voxel([8, 0, 0], outside));

    let points = vec![
        Vec3::new(7.0, 0.0, 0.0),
        Vec3::new(7.5, 0.0, 0.0),
        Vec3::new(8.0, 0.0, 0.0),
    ];

    let gpu = sampler.eval(&ctx, &tree, &points);
    assert_parity(&tree, &points, &gpu);

    let expected_mid = 0.5 * (inside + outside);
    assert!(
        (gpu[1] - expected_mid).abs() < TOL,
        "cross-leaf midpoint must average to {expected_mid}, got {}",
        gpu[1]
    );
}

#[test]
fn out_of_domain_reads_background() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sampler = GpuVdbSample::new(&ctx);

    // Negative coordinates and coordinates at or beyond the domain extent both
    // read the background. A `[1,1,1]` tree spans 32 voxels per axis, so 32 and
    // beyond are out of bounds.
    let background = 0.4_f32;
    let mut tree = tree_with([1, 1, 1], background);
    assert!(tree.set_voxel([0, 0, 0], 0.9));

    let points = vec![
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, -5.0, 0.0),
        Vec3::new(32.0, 0.0, 0.0),
        Vec3::new(0.0, 100.0, 0.0),
        Vec3::new(0.0, 0.0, 40.0),
    ];

    let gpu = sampler.eval(&ctx, &tree, &points);
    assert_parity(&tree, &points, &gpu);
    for (i, &density) in gpu.iter().enumerate() {
        let diff = (density - background).abs();
        assert!(
            diff < TOL,
            "out-of-domain point {i} must read background {background}, got {density}"
        );
    }
}

#[test]
fn multi_root_slot_addressing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sampler = GpuVdbSample::new(&ctx);

    // A `[2,1,1]` tree spans 64 voxels along X across two root slots (each slot
    // covers 32 voxels per axis). A voxel at X=40 lives in the second slot, so a
    // correct hit there proves the root index `((sz*ny)+sy)*nx+sx` is wired up.
    let background = 0.05_f32;
    let slot0_value = 0.6_f32;
    let slot1_value = 0.95_f32;
    let mut tree = tree_with([2, 1, 1], background);
    assert!(tree.set_voxel([5, 0, 0], slot0_value), "first slot voxel");
    assert!(tree.set_voxel([40, 0, 0], slot1_value), "second slot voxel");

    let points = vec![
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(40.0, 0.0, 0.0),
        // A fractional sample inside the second slot, between an active and an
        // inactive voxel, so it averages the hit with the background.
        Vec3::new(40.5, 0.0, 0.0),
        // On the far edge of the domain.
        Vec3::new(20.0, 0.0, 0.0),
    ];

    let gpu = sampler.eval(&ctx, &tree, &points);
    assert_parity(&tree, &points, &gpu);

    assert!(
        (gpu[1] - slot1_value).abs() < TOL,
        "second-slot hit must read {slot1_value}, got {}",
        gpu[1]
    );
    assert!(
        (gpu[1] - background).abs() > DISTINCT,
        "the second-slot hit must differ from the background"
    );
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sampler = GpuVdbSample::new(&ctx);

    // A larger mixed scene: several active voxels scattered across leaves and
    // internal nodes of a `[2,2,2]` tree (64 voxels per axis), queried with a
    // batch of on-grid, fractional, boundary and out-of-domain points in one
    // dispatch, all compared point-for-point against the reference.
    let background = 0.15_f32;
    let mut tree = tree_with([2, 2, 2], background);
    let active = [
        ([0, 0, 0], 0.3_f32),
        ([1, 0, 0], 0.7),
        ([0, 1, 0], 0.5),
        ([1, 1, 1], 0.9),
        ([8, 8, 8], 0.2),
        ([33, 2, 60], 0.8),
        ([63, 63, 63], 0.45),
        ([60, 10, 5], 0.65),
    ];
    for (coord, value) in active {
        assert!(tree.set_voxel(coord, value), "voxel {coord:?} is in domain");
    }

    let points = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(7.5, 8.0, 8.0),
        Vec3::new(8.25, 8.0, 8.0),
        Vec3::new(33.0, 2.0, 60.0),
        Vec3::new(63.0, 63.0, 63.0),
        Vec3::new(60.0, 10.0, 5.0),
        Vec3::new(50.5, 50.5, 50.5),
        Vec3::new(-2.0, -2.0, -2.0),
        Vec3::new(128.0, 0.0, 0.0),
        Vec3::new(2.0, 3.0, 4.0),
    ];

    let gpu = sampler.eval(&ctx, &tree, &points);
    assert_parity(&tree, &points, &gpu);

    // Non-degeneracy: at least one on-grid hit differs from the background.
    assert!(
        (gpu[4] - background).abs() > DISTINCT,
        "the active voxel at (1,1,1) must differ from the background"
    );
}
