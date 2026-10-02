//! Real-device parity for the `Amanatides-Woo` integer voxel-grid ray
//! traversal twin:
//! [`GpuVoxelTraversal`](prism_volumetric_gpu::voxel_traversal::GpuVoxelTraversal)
//! must reproduce the `CPU` golden
//! [`traverse`](prism_render_architecture::particle::voxel_traversal::traverse)
//! across an empty batch, axis-aligned `x`/`y`/`z` rays, a space diagonal, a
//! negative-direction ray, all eight octant directions, a distance-only bound,
//! a step-only bound, a both-bounds query, the degenerate guards (unbounded
//! limit, zero step cap, negative distance, zero-length direction) and a large
//! pseudo-random batch of varied step caps, compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each ray is a fixed, non-reorderable sequence of guarded divisions and
//! incremental additions, so `CPU` and `GPU` evaluate the same closed form in
//! the same associativity. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! asserts an *exact* match on the emitted voxel count and on every integer
//! voxel coordinate (in order), yet a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on the world-distance `t_enter` values, and it checks the
//! entry parameters are non-decreasing. Every fixture is placed clear of
//! voxel-face boundaries (non-integer origin offsets) and axis ties; the random
//! batch reject-samples so consecutive boundary crossings keep a comfortable
//! gap, so the discrete voxel path never flips under a legal perturbation.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::voxel_traversal`；
//! classic `Amanatides-Woo` fast voxel-traversal grid `DDA`; no third-party
//! engine source or derived code.

use prism_render_architecture::particle::voxel_traversal::{Limit, Ray, Vec3, VoxelGrid};
use prism_volumetric_gpu::voxel_traversal::{
    cpu_reference, GpuVoxelTraversal, VoxelTraversalQuery, VoxelTraversalResult, MAX_HITS,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the world-distance `t_enter` values. A `GPU` may
/// fuse a multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Minimum gap (in world distance) required between consecutive boundary
/// crossings of a reject-sampled random ray, so no two voxel entries sit within
/// a `ULP`-scale tie that could flip the step order on one device.
const MIN_GAP: f32 = 2.0e-3;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// The unit cubic grid at the world origin, the fixture most named cases probe.
fn unit_grid() -> VoxelGrid {
    VoxelGrid::cubic(Vec3::ZERO, 1.0)
}

/// Builds one traversal query from a ray origin, a direction, a grid and a
/// limit.
fn query(origin: Vec3, dir: Vec3, grid: VoxelGrid, limit: Limit) -> VoxelTraversalQuery {
    VoxelTraversalQuery {
        ray: Ray::new(origin, dir),
        grid,
        limit,
    }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: the emitted voxel count matches exactly, every integer voxel
/// coordinate matches in order, each `t_enter` matches within tolerance, and the
/// entry parameters are non-decreasing. Returns the `GPU` results for extra
/// per-test assertions. Use only for fixtures placed clear of every voxel-face
/// boundary and axis tie.
fn check(
    ctx: &GpuContext,
    gpu: &GpuVoxelTraversal,
    queries: &[VoxelTraversalQuery],
) -> Vec<VoxelTraversalResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let cpu = cpu_reference(q);
        assert_eq!(
            g.hit_count as usize,
            cpu.len(),
            "lane {lane}: hit_count gpu {} vs cpu {}",
            g.hit_count,
            cpu.len()
        );
        let mut prev = f32::NEG_INFINITY;
        for (i, hit) in cpu.iter().enumerate() {
            assert_eq!(
                g.voxels[i],
                [hit.voxel.x, hit.voxel.y, hit.voxel.z],
                "lane {lane} voxel {i}: gpu {:?} vs cpu [{}, {}, {}]",
                g.voxels[i],
                hit.voxel.x,
                hit.voxel.y,
                hit.voxel.z
            );
            assert!(
                close(g.t_enters[i], hit.t_enter),
                "lane {lane} t_enter {i}: gpu {} vs cpu {}",
                g.t_enters[i],
                hit.t_enter
            );
            assert!(
                g.t_enters[i] + EPS >= prev,
                "lane {lane} t_enter {i}: {} is below the previous {prev}",
                g.t_enters[i]
            );
            prev = g.t_enters[i];
        }
    }
    got
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

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn axis_aligned_x_walk() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    // Straight +x from a non-integer origin: y and z are parallel axes (fixed
    // coordinate), x advances one cell per unit distance.
    let q = query(
        Vec3::new(0.3, 0.4, 0.6),
        Vec3::new(1.0, 0.0, 0.0),
        unit_grid(),
        Limit::steps(12),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].hit_count, 12, "the step cap is the binding limit");
    for i in 0..got[0].hit_count as usize {
        assert_eq!(
            got[0].voxels[i][0], i as i32,
            "x advances one cell per step"
        );
        assert_eq!(got[0].voxels[i][1], 0, "y stays fixed on a parallel axis");
        assert_eq!(got[0].voxels[i][2], 0, "z stays fixed on a parallel axis");
    }
}

#[test]
fn axis_aligned_y_walk() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    let q = query(
        Vec3::new(0.6, 0.3, 0.4),
        Vec3::new(0.0, 1.0, 0.0),
        unit_grid(),
        Limit::steps(10),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].hit_count, 10, "the step cap is the binding limit");
    for i in 0..got[0].hit_count as usize {
        assert_eq!(got[0].voxels[i][0], 0, "x stays fixed on a parallel axis");
        assert_eq!(
            got[0].voxels[i][1], i as i32,
            "y advances one cell per step"
        );
        assert_eq!(got[0].voxels[i][2], 0, "z stays fixed on a parallel axis");
    }
}

#[test]
fn axis_aligned_z_walk() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    let q = query(
        Vec3::new(0.4, 0.6, 0.3),
        Vec3::new(0.0, 0.0, 1.0),
        unit_grid(),
        Limit::steps(9),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].hit_count, 9, "the step cap is the binding limit");
    for i in 0..got[0].hit_count as usize {
        assert_eq!(got[0].voxels[i][0], 0, "x stays fixed on a parallel axis");
        assert_eq!(got[0].voxels[i][1], 0, "y stays fixed on a parallel axis");
        assert_eq!(
            got[0].voxels[i][2], i as i32,
            "z advances one cell per step"
        );
    }
}

#[test]
fn space_diagonal_walk() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    // A (1,1,1) ray from distinct non-integer offsets: the three per-axis
    // boundary parameters stay distinct, so no axis tie and no face boundary is
    // grazed. Over the walk the Manhattan progress equals the number of steps
    // taken after the origin.
    let q = query(
        Vec3::new(0.1, 0.2, 0.3),
        Vec3::new(1.0, 1.0, 1.0),
        unit_grid(),
        Limit::steps(15),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].hit_count, 15, "the step cap is the binding limit");
    let n = got[0].hit_count as usize;
    let first = got[0].voxels[0];
    let last = got[0].voxels[n - 1];
    let manhattan =
        (last[0] - first[0]).abs() + (last[1] - first[1]).abs() + (last[2] - first[2]).abs();
    assert_eq!(
        manhattan as usize,
        n - 1,
        "each diagonal step changes exactly one coordinate by one"
    );
}

#[test]
fn negative_direction_walk() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    // A fully negative direction from a positive non-integer origin: all three
    // axes step by -1 and the coordinates must march downward.
    let q = query(
        Vec3::new(3.3, 4.4, 5.6),
        Vec3::new(-1.0, -1.0, -1.0),
        unit_grid(),
        Limit::steps(14),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].hit_count, 14, "the step cap is the binding limit");
    let n = got[0].hit_count as usize;
    for i in 1..n {
        let d = (got[0].voxels[i][0] - got[0].voxels[i - 1][0])
            + (got[0].voxels[i][1] - got[0].voxels[i - 1][1])
            + (got[0].voxels[i][2] - got[0].voxels[i - 1][2]);
        assert_eq!(
            d, -1,
            "a negative-direction step lowers one coordinate by one"
        );
    }
}

#[test]
fn all_octants_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    // Eight direction octants, each with distinct per-axis magnitudes chosen so
    // no t_max tie ever arises (margins verified > 1e-2 across all 16 steps),
    // launched from a non-integer origin clear of every face.
    let origin = Vec3::new(0.3, 0.45, 0.6);
    let mut queries = Vec::new();
    for sx in [-1.0_f32, 1.0] {
        for sy in [-1.0_f32, 1.0] {
            for sz in [-1.0_f32, 1.0] {
                queries.push(query(
                    origin,
                    Vec3::new(sx * 0.57, sy * 0.71, sz * 0.83),
                    unit_grid(),
                    Limit::steps(16),
                ));
            }
        }
    }
    let got = check(&ctx, &gpu, &queries);
    for g in &got {
        assert_eq!(g.hit_count, 16, "every octant walk fills the step cap");
    }
}

#[test]
fn distance_only_limit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    // Cells of edge 2 along x: world boundaries at x = 2, 4, 6, ...; the ray
    // starts at x = 1 so entries land at t = 1, 3, 5. A distance bound of 4.5
    // admits the origin voxel plus the crossings at t = 1 and t = 3, but not the
    // next crossing at t = 5. The bound sits clear of any boundary.
    let grid = VoxelGrid::cubic(Vec3::ZERO, 2.0);
    let q = query(
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        grid,
        Limit::distance(4.5),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(
        got[0].hit_count, 3,
        "origin plus two crossings fit under 4.5"
    );
    assert!(close(got[0].t_enters[0], 0.0), "origin at t = 0");
    assert!(close(got[0].t_enters[1], 1.0), "first crossing at t = 1");
    assert!(close(got[0].t_enters[2], 3.0), "second crossing at t = 3");
}

#[test]
fn steps_only_limit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    // No distance bound: the step cap alone decides the length on the infinite
    // lattice.
    let q = query(
        Vec3::new(0.2, 0.3, 0.7),
        Vec3::new(0.9, 0.4, 0.2),
        unit_grid(),
        Limit::steps(7),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].hit_count, 7, "the step cap is the binding limit");
}

#[test]
fn both_limits_tighter_wins() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    // Edge-2 cells again (entries at t = 0, 1, 3, 5, 7, ...). With a generous
    // distance of 20 but a step cap of 4, the step cap wins and only four voxels
    // are emitted.
    let grid = VoxelGrid::cubic(Vec3::ZERO, 2.0);
    let q = query(
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        grid,
        Limit::both(20.0, 4),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].hit_count, 4, "the step cap is the tighter bound");

    // Flip the roles: a tight distance of 2.5 wins over a generous step cap of
    // 32, admitting the origin plus the crossings at t = 1 and t = ... none past
    // 2.5, so origin plus the t = 1 entry.
    let q2 = query(
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        grid,
        Limit::both(2.5, 32),
    );
    let got2 = check(&ctx, &gpu, &[q2]);
    assert_eq!(
        got2[0].hit_count, 2,
        "the distance bound is the tighter bound"
    );
}

#[test]
fn degenerate_cases_are_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    let grid = unit_grid();
    let origin = Vec3::new(0.5, 0.5, 0.5);
    let dir = Vec3::new(1.0, 0.0, 0.0);
    let queries = [
        // A fully unbounded limit refuses by contract.
        query(origin, dir, grid, Limit::NONE),
        // A zero step cap can never emit a voxel.
        query(origin, dir, grid, Limit::steps(0)),
        // A negative distance bound admits no entry.
        query(origin, dir, grid, Limit::distance(-1.0)),
        // A zero-length direction is the degenerate ray.
        query(origin, Vec3::ZERO, grid, Limit::steps(8)),
    ];
    let got = check(&ctx, &gpu, &queries);
    for g in &got {
        assert_eq!(
            g.hit_count, 0,
            "every degenerate guard yields an empty walk"
        );
    }
}

#[test]
fn zero_distance_emits_only_origin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);
    // A distance bound of exactly zero still admits the origin voxel at t = 0,
    // but no crossing (the first crossing has t > 0 > the bound).
    let q = query(
        Vec3::new(0.3, 0.4, 0.6),
        Vec3::new(1.0, 0.0, 0.0),
        unit_grid(),
        Limit::distance(0.0),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(
        got[0].hit_count, 1,
        "only the origin voxel fits at distance 0"
    );
    assert_eq!(got[0].voxels[0], [0, 0, 0], "the origin voxel is (0, 0, 0)");
}

#[test]
fn random_batch_matches_lane_for_lane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVoxelTraversal::new(&ctx);

    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries: Vec<VoxelTraversalQuery> = Vec::new();
    let mut shortest = usize::MAX;
    let mut longest = 0usize;

    // Reject-sample 256 non-degenerate rays whose consecutive boundary crossings
    // keep a comfortable gap, so no voxel entry sits near an axis tie that a
    // legal ULP perturbation could flip. Each ray carries its own step cap in
    // [2, MAX_HITS / 2], so the walk length — exactly the cap on the infinite
    // lattice — varies lane to lane and the batch exercises many walk lengths.
    while queries.len() < 256 {
        let origin = Vec3::new(
            (lcg(&mut state) * 6.0 - 3.0) + 0.2,
            (lcg(&mut state) * 6.0 - 3.0) + 0.35,
            (lcg(&mut state) * 6.0 - 3.0) + 0.55,
        );
        // Direction components kept at magnitude in [0.2, 1.0] (sign random), so
        // no axis is near-parallel and the three magnitudes are generically
        // distinct.
        let dx = (lcg(&mut state) * 0.8 + 0.2) * if lcg(&mut state) < 0.5 { -1.0 } else { 1.0 };
        let dy = (lcg(&mut state) * 0.8 + 0.2) * if lcg(&mut state) < 0.5 { -1.0 } else { 1.0 };
        let dz = (lcg(&mut state) * 0.8 + 0.2) * if lcg(&mut state) < 0.5 { -1.0 } else { 1.0 };
        let dir = Vec3::new(dx, dy, dz);
        let cap = 2 + (lcg(&mut state) * ((MAX_HITS / 2) as f32 - 2.0)) as usize;
        let q = query(origin, dir, unit_grid(), Limit::steps(cap));

        let cpu = cpu_reference(&q);
        if cpu.len() < 2 {
            continue;
        }
        // Reject any ray with two boundary crossings closer than MIN_GAP: such a
        // near-tie is the only place the integer path could legally differ.
        let mut ok = true;
        for i in 1..cpu.len() {
            if cpu[i].t_enter - cpu[i - 1].t_enter < MIN_GAP {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        shortest = shortest.min(cpu.len());
        longest = longest.max(cpu.len());
        queries.push(q);
    }

    check(&ctx, &gpu, &queries);

    // A large spread must exercise many walk lengths, so the test is not
    // trivially passing on a single fixed length.
    assert!(
        longest > shortest + 4,
        "random batch should span a wide range of walk lengths: {shortest}..={longest}"
    );
}
