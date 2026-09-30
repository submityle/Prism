//! Real-device parity for the uniform-grid self-collision neighbor-gather twin:
//! [`GpuSelfCollisionGrid`] must reproduce the `CPU` golden
//! [`UniformGrid::neighbors`](prism_render_architecture::hair::self_collision_grid::UniformGrid::neighbors)
//! for every particle's query cell, covering the ascending 27-cell union: same
//! cell sharing, cross-cell gather, isolated particles, ragged multi-cell
//! batches, negative-coordinate cells, and non-finite particles skipped by the
//! build. The grid, its `CSR`, and each query cell are produced by the golden
//! [`UniformGrid::build`](prism_render_architecture::hair::self_collision_grid::UniformGrid::build),
//! [`build_csr`](prism_render_architecture::hair::self_collision_grid::build_csr)
//! and [`grid_cell_of`](prism_render_architecture::hair::self_collision_grid::grid_cell_of),
//! so the whole acceleration structure is exercised end to end and each
//! emitted neighbor list is asserted value-for-value against the golden.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel routes `u32` particle indices with no floating-point arithmetic
//! on the payload — a device-side binary search over the sorted cell keys plus
//! an insertion sort of the gathered slice — so parity is asserted bit-identical
//! with `assert_eq!`, not to a tolerance. Because every particle sits in exactly
//! one cell, the 27-cell union carries no duplicates, so the sorted slice must
//! match the golden's `sort_unstable` output exactly. Positions are explicit
//! literals (never `sin`/`cos`), and the scenes place several particles in
//! shared and adjacent cells so a no-op kernel (empty lists) could not pass.
//!
//! Provenance: standard uniform-grid spatial hash neighbor gather plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::self_collision_grid::GpuSelfCollisionGrid;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::dynamics::{StrandParticle, Vec3};
use prism_render_architecture::hair::self_collision_grid::{grid_cell_of, UniformGrid};

/// A free particle at `(x, y, z)`.
fn particle(x: f32, y: f32, z: f32) -> StrandParticle {
    StrandParticle::free(Vec3::new(x, y, z))
}

/// Runs the twin and the golden on the same particles, asserting every emitted
/// neighbor list equals the golden `neighbors` of that particle's query cell.
/// Returns the `GPU` lists for further case-specific assertions.
fn assert_parity(
    ctx: &GpuContext,
    gatherer: &GpuSelfCollisionGrid,
    particles: &[StrandParticle],
    cell_size: f32,
) -> Vec<Vec<u32>> {
    let gpu = gatherer.eval(ctx, particles, cell_size);
    assert_eq!(gpu.len(), particles.len(), "neighbor list count");

    let grid = UniformGrid::build(particles, cell_size);
    let mut scratch: Vec<u32> = Vec::new();
    for (p, particle) in particles.iter().enumerate() {
        let cell = grid_cell_of(particle.position, cell_size);
        grid.neighbors(cell, &mut scratch);
        assert_eq!(gpu[p], scratch, "neighbor list mismatch at particle {p}");
    }

    gpu
}

/// Several particles in the same and adjacent cells: each particle's 27-cell
/// union must gather every nearby index in ascending order, including itself.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_shared_and_adjacent_cells_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping self-collision grid parity: no wgpu adapter on this host");
        return;
    };
    let gatherer = GpuSelfCollisionGrid::new(&ctx);

    let cell_size = 0.5;
    // Indices 0,1 share cell (0,0,0); 2 is in the +x adjacent cell (0..); 3 in
    // a -x adjacent cell; 4 is diagonally adjacent; 5 is far away.
    let particles = [
        particle(0.05, 0.05, 0.05),
        particle(0.20, 0.10, 0.40),
        particle(0.60, 0.10, 0.10),
        particle(-0.30, 0.10, 0.10),
        particle(0.55, 0.55, 0.05),
        particle(5.00, 5.00, 5.00),
    ];
    let gpu = assert_parity(&ctx, &gatherer, &particles, cell_size);

    // Particle 0 and 1 share a cell, and 2/3/4 are all in adjacent cells, so
    // particle 0's neighborhood is the full 0..=4 union (5 is far).
    assert_eq!(gpu[0], vec![0, 1, 2, 3, 4], "particle 0 gathers all nearby");
    // The far particle sees only itself.
    assert_eq!(gpu[5], vec![5], "isolated particle sees only itself");
}

/// Every particle in its own well-separated cell: each neighbor list is exactly
/// the singleton of that particle.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_isolated_particles_see_only_themselves() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping self-collision grid parity: no wgpu adapter on this host");
        return;
    };
    let gatherer = GpuSelfCollisionGrid::new(&ctx);

    let cell_size = 1.0;
    // Each particle is >= 2 cells from any other on at least one axis.
    let particles = [
        particle(0.5, 0.5, 0.5),
        particle(5.5, 0.5, 0.5),
        particle(0.5, 5.5, 0.5),
        particle(0.5, 0.5, 5.5),
    ];
    let gpu = assert_parity(&ctx, &gatherer, &particles, cell_size);

    for (i, list) in gpu.iter().enumerate() {
        assert_eq!(list, &vec![i as u32], "particle {i} is isolated");
    }
}

/// A ragged multi-cell scene with uneven bucket sizes: the disjoint per-particle
/// output layout must reproduce every golden list, whatever its length.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_ragged_multi_cell_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping self-collision grid parity: no wgpu adapter on this host");
        return;
    };
    let gatherer = GpuSelfCollisionGrid::new(&ctx);

    let cell_size = 1.0;
    // A dense cluster around the origin (indices 0..4 all within adjacent cells)
    // and a lone pair elsewhere (5,6 sharing a cell far off).
    let particles = [
        particle(0.10, 0.10, 0.10),
        particle(0.90, 0.10, 0.10),
        particle(1.20, 0.10, 0.10),
        particle(0.10, 0.90, 0.10),
        particle(0.10, 0.10, 1.20),
        particle(10.10, 10.10, 10.10),
        particle(10.40, 10.20, 10.30),
    ];
    let gpu = assert_parity(&ctx, &gatherer, &particles, cell_size);

    // The far pair only ever see each other.
    assert_eq!(gpu[5], vec![5, 6], "far pair share a neighborhood");
    assert_eq!(gpu[6], vec![5, 6], "far pair share a neighborhood");
}

/// Negative-coordinate cells: `floor` on negative positions yields negative cell
/// keys, and the lexicographic binary search must still find the right buckets.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_negative_coordinate_cells_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping self-collision grid parity: no wgpu adapter on this host");
        return;
    };
    let gatherer = GpuSelfCollisionGrid::new(&ctx);

    let cell_size = 0.5;
    let particles = [
        particle(-0.10, -0.10, -0.10),
        particle(-0.40, -0.10, -0.10),
        particle(-0.60, -0.10, -0.10),
        particle(0.10, 0.10, 0.10),
        particle(-0.10, 0.40, -0.30),
    ];
    let gpu = assert_parity(&ctx, &gatherer, &particles, cell_size);

    // Sanity: particle 0 gathers at least itself and its cell-mates/neighbors,
    // never empty, so a no-op kernel would fail here too.
    assert!(gpu[0].contains(&0), "particle 0 gathers itself");
    assert!(gpu[0].len() >= 2, "particle 0 has real neighbors");
}

/// Non-finite particles are skipped by the golden build, so they never appear in
/// any neighbor list; finite particles still gather correctly around them.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_non_finite_particles_are_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping self-collision grid parity: no wgpu adapter on this host");
        return;
    };
    let gatherer = GpuSelfCollisionGrid::new(&ctx);

    let cell_size = 1.0;
    let particles = [
        particle(0.10, 0.10, 0.10),
        particle(f32::NAN, 0.10, 0.10),
        particle(0.30, 0.20, 0.40),
        particle(f32::INFINITY, 0.0, 0.0),
        particle(0.50, 0.50, 0.50),
    ];
    let gpu = assert_parity(&ctx, &gatherer, &particles, cell_size);

    // The finite indices 0,2,4 share the origin cell; the non-finite 1,3 are not
    // bucketed, so no list may contain them.
    for (p, list) in gpu.iter().enumerate() {
        assert!(
            !list.contains(&1),
            "particle {p} must not gather non-finite 1"
        );
        assert!(
            !list.contains(&3),
            "particle {p} must not gather non-finite 3"
        );
    }
    assert_eq!(gpu[0], vec![0, 2, 4], "finite particles gather each other");
}

/// A degenerate `cell_size` yields an empty grid, so every particle's neighbor
/// list is empty; an empty particle batch returns no lists at all.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_and_empty_inputs() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping self-collision grid parity: no wgpu adapter on this host");
        return;
    };
    let gatherer = GpuSelfCollisionGrid::new(&ctx);

    // Empty batch: no dispatch, no lists.
    let empty = gatherer.eval(&ctx, &[], 1.0);
    assert!(empty.is_empty(), "empty batch returns no lists");

    // Degenerate cell size: empty grid, so every list is empty.
    let particles = [particle(0.0, 0.0, 0.0), particle(1.0, 1.0, 1.0)];
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        let gpu = assert_parity(&ctx, &gatherer, &particles, bad);
        for (p, list) in gpu.iter().enumerate() {
            assert!(
                list.is_empty(),
                "particle {p} has no neighbors in empty grid"
            );
        }
    }
}
