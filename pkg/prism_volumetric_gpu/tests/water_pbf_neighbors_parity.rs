//! Real-device parity for the `PBF` neighbour-gather twin:
//! [`GpuWaterPbfNeighbors`](prism_volumetric_gpu::water_pbf_neighbors::GpuWaterPbfNeighbors)
//! must reproduce the spatial-hash scan of the `CPU` golden
//! [`pbf`](prism_render_architecture::water::pbf) —
//! [`gather_neighbors`](prism_render_architecture::water::pbf::gather_neighbors)
//! — across a corner cell, an isolated particle, a full bucket, an out-of-grid
//! particle, the keep boundary and a randomized point-cloud sweep, compared
//! list-for-list and in order.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected lists come straight from the public golden
//! [`gather_neighbors`](prism_render_architecture::water::pbf::gather_neighbors)
//! fed by the golden
//! [`bin_particles`](prism_render_architecture::water::pbf::bin_particles), so a
//! `GPU == golden` pass is direct evidence the ported kernel gathers the same
//! neighbours in the same order the reference does.
//!
//! # Parity criterion
//!
//! Every gathered output is an integer (a neighbour index or a count), so the
//! whole list — contents, order and length — is asserted with an exact
//! `assert_eq!`. No tolerance is involved.
//!
//! # Conditioning
//!
//! Two boundaries would otherwise let `CPU` and `GPU` disagree: the query
//! particle's cell assignment, where a position near a cell face could truncate
//! to a different cell, and the keep test `d^2 <= cell_size^2`, where a
//! candidate pair near that distance could straddle the comparison. Fixtures
//! and the randomized sweep keep the query particle's per-axis cell fraction
//! clear of a face and keep every candidate pair's squared distance clear of
//! `cell_size^2` by rejection sampling.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pbf`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::pbf::{bin_particles, gather_neighbors, PbfGrid};
use prism_render_architecture::water::Vec3;
use prism_volumetric_gpu::water_pbf_neighbors::{
    GpuWaterPbfNeighbors, WaterPbfNeighborsQuery, WaterPbfNeighborsResult,
};
use prism_volumetric_gpu::GpuContext;

/// A 64-bit linear-congruential generator (`PCG`-style multiplier) for
/// host-side fixtures; no transcendental and no float equality.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a value in `[0, span)` with milli-resolution from the generator.
fn draw(state: &mut u64, span: f32) -> f32 {
    (lcg(state) % 1000) as f32 / 1000.0 * span
}

/// Flattens the golden bins for `positions` on `grid` into the compressed-
/// sparse-row pair the twin consumes, and returns the twin query for
/// `particle`.
fn make_query(grid: PbfGrid, positions: &[Vec3], particle: u32) -> WaterPbfNeighborsQuery {
    let bins = bin_particles(grid, positions);
    let mut cell_offsets: Vec<u32> = Vec::with_capacity(bins.cells.len() + 1);
    let mut cell_items: Vec<u32> = Vec::new();
    cell_offsets.push(0);
    for cell in &bins.cells {
        cell_items.extend_from_slice(cell);
        cell_offsets.push(cell_items.len() as u32);
    }
    let flat_positions: Vec<[f32; 3]> = positions.iter().map(|p| [p.x, p.y, p.z]).collect();
    WaterPbfNeighborsQuery::new(
        [grid.origin.x, grid.origin.y, grid.origin.z],
        grid.cell_size,
        grid.nx,
        grid.ny,
        grid.nz,
        flat_positions,
        cell_offsets,
        cell_items,
        particle,
    )
}

/// Computes the golden gathered list for one case.
fn oracle(grid: PbfGrid, positions: &[Vec3], particle: u32) -> WaterPbfNeighborsResult {
    let bins = bin_particles(grid, positions);
    WaterPbfNeighborsResult {
        neighbors: gather_neighbors(grid, &bins, positions, particle),
    }
}

/// Dispatches one case and pins the `GPU` neighbour list against the golden
/// list exactly, order and length included.
fn check_case(
    ctx: &GpuContext,
    gpu: &GpuWaterPbfNeighbors,
    grid: PbfGrid,
    positions: &[Vec3],
    particle: u32,
) {
    let query = make_query(grid, positions, particle);
    let got = gpu.evaluate(ctx, std::slice::from_ref(&query));
    assert_eq!(got.len(), 1, "one result per query is expected");
    let want = oracle(grid, positions, particle);
    assert_eq!(
        got[0].neighbors, want.neighbors,
        "particle {particle}: gpu list {:?} vs cpu list {:?}",
        got[0].neighbors, want.neighbors
    );
}

/// Returns whether the per-axis cell fraction of `p` is clear of a cell face by
/// `margin`, so the truncating cell assignment cannot disagree between engines.
fn cell_fraction_ok(grid: PbfGrid, p: Vec3, margin: f32) -> bool {
    let axes = [
        (p.x - grid.origin.x),
        (p.y - grid.origin.y),
        (p.z - grid.origin.z),
    ];
    for a in axes {
        if a < 0.0 {
            return false;
        }
        let c = a / grid.cell_size;
        let frac = c - (c as u32) as f32;
        if frac < margin || frac > 1.0 - margin {
            return false;
        }
    }
    true
}

/// Returns whether every candidate pair in the query particle's `3x3x3`
/// neighbourhood has a squared distance clear of `cell_size^2` by `rel_margin`
/// of it, so the keep test cannot straddle the comparison between engines.
fn keep_boundary_ok(grid: PbfGrid, positions: &[Vec3], particle: u32, rel_margin: f32) -> bool {
    let radius_sq = grid.cell_size * grid.cell_size;
    let band = radius_sq * rel_margin;
    let p = positions[particle as usize];
    for (j, &q) in positions.iter().enumerate() {
        if j as u32 == particle {
            continue;
        }
        let d2 = p.sub(q).length_squared();
        if (d2 - radius_sq).abs() < band {
            return false;
        }
    }
    true
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfNeighbors::new(&ctx);
    // An empty batch never dispatches (a storage buffer cannot be zero-sized)
    // and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn corner_cell_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfNeighbors::new(&ctx);
    let grid = PbfGrid {
        origin: Vec3::ZERO,
        cell_size: 1.0,
        nx: 4,
        ny: 4,
        nz: 4,
    };
    // Query particle sits in corner cell (0,0,0); its neighbourhood clamps to
    // cells [0,1] on every axis. A close partner stays within the keep radius,
    // a far one in-cell exceeds it, and one lives outside the neighbourhood.
    let positions = vec![
        Vec3::new(0.3, 0.3, 0.3),
        Vec3::new(0.6, 0.4, 0.35),
        Vec3::new(1.4, 1.4, 1.4),
        Vec3::new(3.3, 3.3, 3.3),
    ];
    check_case(&ctx, &gpu, grid, &positions, 0);
}

#[test]
fn isolated_particle_has_no_neighbours() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfNeighbors::new(&ctx);
    let grid = PbfGrid {
        origin: Vec3::ZERO,
        cell_size: 1.0,
        nx: 6,
        ny: 6,
        nz: 6,
    };
    // The query particle's whole neighbourhood is empty; the only other
    // particle is many cells away.
    let positions = vec![Vec3::new(2.5, 2.5, 2.5), Vec3::new(5.4, 5.4, 5.4)];
    let want = oracle(grid, &positions, 0);
    assert!(
        want.neighbors.is_empty(),
        "fixture must leave the query particle isolated"
    );
    check_case(&ctx, &gpu, grid, &positions, 0);
}

#[test]
fn full_bucket_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfNeighbors::new(&ctx);
    let grid = PbfGrid {
        origin: Vec3::ZERO,
        cell_size: 2.0,
        nx: 3,
        ny: 3,
        nz: 3,
    };
    // A tight cluster inside one cell, all within a small ball well under the
    // keep radius, so every partner is a neighbour and the bucket is full.
    let center = Vec3::new(3.0, 3.0, 3.0);
    let mut positions = vec![center];
    let mut state = 0x1234_5678_9abc_def0_u64;
    while positions.len() < 24 {
        let off = Vec3::new(
            draw(&mut state, 0.6) - 0.3,
            draw(&mut state, 0.6) - 0.3,
            draw(&mut state, 0.6) - 0.3,
        );
        positions.push(Vec3::new(
            center.x + off.x,
            center.y + off.y,
            center.z + off.z,
        ));
    }
    let want = oracle(grid, &positions, 0);
    assert_eq!(
        want.neighbors.len(),
        positions.len() - 1,
        "every clustered partner must be a neighbour"
    );
    check_case(&ctx, &gpu, grid, &positions, 0);
}

#[test]
fn out_of_grid_particle_has_no_neighbours() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfNeighbors::new(&ctx);
    let grid = PbfGrid {
        origin: Vec3::ZERO,
        cell_size: 1.0,
        nx: 4,
        ny: 4,
        nz: 4,
    };
    // The query particle sits outside the grid (negative), so the golden
    // returns an empty list; a couple of in-grid particles are present.
    let positions = vec![
        Vec3::new(-1.5, 0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(1.5, 1.5, 1.5),
    ];
    let want = oracle(grid, &positions, 0);
    assert!(
        want.neighbors.is_empty(),
        "an out-of-grid particle must gather nothing"
    );
    check_case(&ctx, &gpu, grid, &positions, 0);
}

#[test]
fn out_of_range_index_has_no_neighbours() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfNeighbors::new(&ctx);
    let grid = PbfGrid {
        origin: Vec3::ZERO,
        cell_size: 1.0,
        nx: 3,
        ny: 3,
        nz: 3,
    };
    let positions = vec![Vec3::new(1.5, 1.5, 1.5), Vec3::new(1.6, 1.4, 1.5)];
    // Particle index past the end: the golden `positions.get(particle)?` bails
    // out with an empty list.
    let want = oracle(grid, &positions, 7);
    assert!(
        want.neighbors.is_empty(),
        "an out-of-range index must gather nothing"
    );
    check_case(&ctx, &gpu, grid, &positions, 7);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfNeighbors::new(&ctx);
    let grid = PbfGrid {
        origin: Vec3::new(-1.0, -2.0, 0.5),
        cell_size: 1.3,
        nx: 5,
        ny: 5,
        nz: 5,
    };
    let span_x = grid.cell_size * grid.nx as f32;
    let span_y = grid.cell_size * grid.ny as f32;
    let span_z = grid.cell_size * grid.nz as f32;
    let mut state = 0x0bad_c0de_dead_beef_u64;
    let mut built = 0u32;
    // Many well-conditioned random clouds pin the gather across a wide range of
    // occupancy. Each cloud's query particle is kept clear of a cell face and
    // every candidate pair clear of the keep radius by rejection sampling, so
    // neither the cell assignment nor the keep test can disagree.
    while built < 160 {
        let n = 4 + (lcg(&mut state) as usize) % 29;
        let mut positions: Vec<Vec3> = Vec::with_capacity(n);
        let mut j = 0;
        while j < n {
            positions.push(Vec3::new(
                grid.origin.x + draw(&mut state, span_x),
                grid.origin.y + draw(&mut state, span_y),
                grid.origin.z + draw(&mut state, span_z),
            ));
            j += 1;
        }
        let particle = (lcg(&mut state) as usize % positions.len()) as u32;
        // Require the query particle to be in-grid, clear of a cell face, and
        // every candidate pair clear of the keep boundary; else skip this draw.
        let p = positions[particle as usize];
        let in_grid = (0..3).all(|axis| {
            let (coord, origin, n) = match axis {
                0 => (p.x, grid.origin.x, grid.nx),
                1 => (p.y, grid.origin.y, grid.ny),
                _ => (p.z, grid.origin.z, grid.nz),
            };
            let local = coord - origin;
            local >= 0.0 && (local / grid.cell_size) < n as f32
        });
        if !in_grid
            || !cell_fraction_ok(grid, p, 0.02)
            || !keep_boundary_ok(grid, &positions, particle, 0.05)
        {
            continue;
        }
        built += 1;
        check_case(&ctx, &gpu, grid, &positions, particle);
    }
}
