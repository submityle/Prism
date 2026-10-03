//! Water `PBF` density-solve compute kernels: the two-pass `WESL` shader plus
//! their bit-exact `CPU` twins.
//!
//! Position-Based Fluids (`Macklin` & Müller 2013) projects one density
//! constraint per particle. The classical two-pass split runs
//! [`WaterKernel::PbfComputeLambda`](super::super::kernels::WaterKernel), which
//! writes every particle's `XPBD` scaling factor `lambda_i` once, before
//! [`WaterKernel::PbfDensitySolve`](super::super::kernels::WaterKernel) reads
//! each neighbour's `lambda_j` from that buffer rather than re-gathering its
//! 27-cell neighbourhood — turning the solve from `O(n * k^2)` into `O(n * k)`
//! while staying numerically identical.
//!
//! [`WATER_PBF_DENSITY_WESL`] is the shader (entry points `pbf_compute_lambda`
//! and `water_pbf_density_solve`); [`dispatch_pbf_compute_lambda`] and
//! [`dispatch_pbf_density_solve`] are its bit-exact `CPU` twins. Because the
//! sandbox has no `GPU`, the twins are the correctness proof: they consume the
//! identical buffer `ABI` (four storage buffers — `positions_in`,
//! `positions_out`, the packed neighbour `hash`, and the `lambdas` scratch —
//! plus one uniform param block, a 64-lane group over the `Particle` domain)
//! and reconstruct the shader arithmetic. Both twins delegate the actual
//! numerics to the `CPU` golden [`super::super::pbf`] primitives
//! ([`poly6`](super::super::pbf::poly6) via
//! [`estimate_density`](super::super::pbf::estimate_density),
//! [`spiky_gradient`](super::super::pbf::spiky_gradient),
//! [`constraint_lambda`](super::super::pbf::constraint_lambda),
//! [`artificial_pressure`](super::super::pbf::artificial_pressure) and
//! [`position_correction`](super::super::pbf::position_correction)), so the
//! twins are bit-exact with the golden by construction. The parity tests drive
//! neighbour finding through the packed `hash` buffer and diff lane-for-lane
//! against an independent reference built from the golden
//! [`PbfGrid`](super::super::pbf::PbfGrid) /
//! [`gather_neighbors`](super::super::pbf::gather_neighbors) path, so they prove
//! the packed spatial-hash layout reproduces the golden gather rather than
//! asserting a tautology.

use alloc::vec;
use alloc::vec::Vec;

use super::super::pbf::{
    artificial_pressure, constraint_lambda, estimate_density, position_correction, spiky_gradient,
    NeighborContribution, PbfParams,
};
use super::super::{Vec3, EPS};

/// `WESL` source of the water `PBF` density-solve compute kernels.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_PBF_DENSITY_WESL: &str = include_str!("water_pbf_density.wesl");

/// `f32` lanes per particle in the `positions_in` / `positions_out` buffers:
/// the `vec4` `(x, y, z, w)` whose `w` lane is carried through untouched.
pub const PBF_POSITION_FLOATS: usize = 4;

/// Uniform parameter block for the `PBF` density solve, mirroring the shader's
/// `PbfParams`.
///
/// It folds the `CPU` [`PbfParams`](super::super::pbf::PbfParams) tuning
/// together with the [`PbfGrid`](super::super::pbf::PbfGrid) spatial-hash
/// description the shader needs inline, plus the per-dispatch `particle_count`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PbfDensityParams {
    /// Minimum corner of the spatial-hash grid, world space.
    pub grid_origin: Vec3,
    /// Grid cell edge length (`> 0`), equal to the smoothing radius.
    pub cell_size: f32,
    /// Target rest density `rho_0` (`> 0`).
    pub rest_density: f32,
    /// Per-particle mass (`> 0`) for the `SPH` density sum.
    pub particle_mass: f32,
    /// Smoothing radius `h` (`> 0`); support radius of both kernels.
    pub smoothing_radius: f32,
    /// `XPBD` relaxation/compliance term added to the `lambda` denominator.
    pub relaxation_epsilon: f32,
    /// Artificial-pressure strength `k` (`>= 0`).
    pub artificial_pressure_k: f32,
    /// Fraction of `h` at which the artificial-pressure reference is sampled.
    pub artificial_pressure_delta_q: f32,
    /// Artificial-pressure exponent `n` (`>= 1`).
    pub artificial_pressure_n: u32,
    /// Number of particles bounding the per-particle dispatch.
    pub particle_count: u32,
    /// Grid cell count along x (`>= 1`).
    pub grid_nx: u32,
    /// Grid cell count along y (`>= 1`).
    pub grid_ny: u32,
    /// Grid cell count along z (`>= 1`).
    pub grid_nz: u32,
}

impl PbfDensityParams {
    /// Total grid cell count (`nx * ny * nz`), saturating so a pathological
    /// descriptor can never overflow the index math.
    #[must_use]
    fn cell_count(self) -> usize {
        (self.grid_nx as usize)
            .saturating_mul(self.grid_ny as usize)
            .saturating_mul(self.grid_nz as usize)
    }

    /// Reconstructs the `CPU` [`PbfParams`](super::super::pbf::PbfParams) tuning
    /// (sans the grid fields) so the twin can reuse the golden
    /// [`artificial_pressure`](super::super::pbf::artificial_pressure). The
    /// solver-iteration count is irrelevant to the per-pass arithmetic and is
    /// set to one.
    #[must_use]
    fn tuning(self) -> PbfParams {
        PbfParams {
            rest_density: self.rest_density,
            particle_mass: self.particle_mass,
            smoothing_radius: self.smoothing_radius,
            relaxation_epsilon: self.relaxation_epsilon,
            artificial_pressure_k: self.artificial_pressure_k,
            artificial_pressure_n: self.artificial_pressure_n,
            artificial_pressure_delta_q: self.artificial_pressure_delta_q,
            solver_iterations: 1,
        }
    }
}

/// Reads particle `idx`'s world position (the `xyz` lanes) from the packed
/// `(x, y, z, w)` lane buffer.
#[inline]
fn read_position(positions: &[f32], idx: usize) -> Vec3 {
    let base = idx * PBF_POSITION_FLOATS;
    Vec3::new(positions[base], positions[base + 1], positions[base + 2])
}

/// Integer cell coordinate of a world position, or `None` when it falls outside
/// the grid. Mirrors the shader's `pbf_cell_coord` and the `CPU`
/// [`PbfGrid::cell_coord`](super::super::pbf::PbfGrid::cell_coord).
#[inline]
fn cell_coord(params: PbfDensityParams, p: Vec3) -> Option<(u32, u32, u32)> {
    if params.cell_size <= EPS {
        return None;
    }
    let local = p.sub(params.grid_origin);
    if local.x < 0.0 || local.y < 0.0 || local.z < 0.0 {
        return None;
    }
    let cx = (local.x / params.cell_size) as u32;
    let cy = (local.y / params.cell_size) as u32;
    let cz = (local.z / params.cell_size) as u32;
    if cx >= params.grid_nx || cy >= params.grid_ny || cz >= params.grid_nz {
        return None;
    }
    Some((cx, cy, cz))
}

/// Row-major flat cell index. Mirrors the shader's `pbf_flat_index`.
#[inline]
fn flat_index(params: PbfDensityParams, cx: u32, cy: u32, cz: u32) -> usize {
    let nx = params.grid_nx as usize;
    let ny = params.grid_ny as usize;
    (cz as usize) * nx * ny + (cy as usize) * nx + (cx as usize)
}

/// Clamped `(z, y, x)` neighbourhood bounds (the particle's cell plus the 26
/// adjacent cells), identical to the shader's `select`/`min` fences.
#[inline]
fn neighbourhood_bounds(
    params: PbfDensityParams,
    cell: (u32, u32, u32),
) -> ((u32, u32), (u32, u32), (u32, u32)) {
    let (cx, cy, cz) = cell;
    let z_lo = if cz == 0 { 0 } else { cz - 1 };
    let z_hi = (cz + 1).min(params.grid_nz - 1);
    let y_lo = if cy == 0 { 0 } else { cy - 1 };
    let y_hi = (cy + 1).min(params.grid_ny - 1);
    let x_lo = if cx == 0 { 0 } else { cx - 1 };
    let x_hi = (cx + 1).min(params.grid_nx - 1);
    ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))
}

/// A neighbour discovered by walking the packed spatial hash, carrying the
/// squared distance and relative vector the kernel math needs.
struct Neighbour {
    /// Neighbour particle index.
    j: usize,
    /// Squared distance `|p_i - p_j|^2`.
    r_squared: f32,
    /// Relative vector `p_i - p_j`.
    r_vec: Vec3,
}

/// Walks the packed `hash` buffer over particle `i`'s 27-cell neighbourhood in
/// the shader's fixed `(z, y, x)` cell order and stored index order, keeping the
/// in-radius neighbours (excluding `i` itself). Returns an empty list for an
/// out-of-grid particle, matching the shader's "skip, do not crash" contract.
fn gather_from_hash(
    positions: &[f32],
    hash: &[u32],
    params: PbfDensityParams,
    cell_count: usize,
    i: usize,
) -> Vec<Neighbour> {
    let mut out = Vec::new();
    let p_i = read_position(positions, i);
    let Some(cell) = cell_coord(params, p_i) else {
        return out;
    };
    let radius_sq = params.cell_size * params.cell_size;
    let index_base = 2 * cell_count;
    let ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)) = neighbourhood_bounds(params, cell);

    let mut z = z_lo;
    while z <= z_hi {
        let mut y = y_lo;
        while y <= y_hi {
            let mut x = x_lo;
            while x <= x_hi {
                let flat = flat_index(params, x, y, z);
                let start = hash[2 * flat] as usize;
                let count = hash[2 * flat + 1] as usize;
                let mut s = 0usize;
                while s < count {
                    let j = hash[index_base + start + s] as usize;
                    s += 1;
                    if j == i || j >= params.particle_count as usize {
                        continue;
                    }
                    let r_vec = p_i.sub(read_position(positions, j));
                    let r2 = r_vec.length_squared();
                    if r2 > radius_sq {
                        continue;
                    }
                    out.push(Neighbour {
                        j,
                        r_squared: r2,
                        r_vec,
                    });
                }
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }
    out
}

/// `XPBD` scaling factor `lambda_i` for particle `i`, bit-exact with the
/// shader's `pbf_lambda_at`. Reuses the golden
/// [`estimate_density`](super::super::pbf::estimate_density) (self term `r = 0`
/// first, then the neighbours in walk order) and
/// [`constraint_lambda`](super::super::pbf::constraint_lambda), so the result
/// matches the `CPU` reference lane-for-lane.
fn lambda_at(
    positions: &[f32],
    hash: &[u32],
    params: PbfDensityParams,
    cell_count: usize,
    i: usize,
) -> f32 {
    if params.rest_density <= EPS || i >= params.particle_count as usize || cell_count == 0 {
        return 0.0;
    }
    let p_i = read_position(positions, i);
    if cell_coord(params, p_i).is_none() {
        return 0.0;
    }
    let h = params.smoothing_radius;
    let neighbours = gather_from_hash(positions, hash, params, cell_count, i);

    // Density: self term first, then the neighbours in walk order (identical
    // summation order to the shader and the golden `estimate_density`).
    let mut r_squared = Vec::with_capacity(neighbours.len() + 1);
    r_squared.push(0.0_f32);
    let mut grad_sum = Vec3::ZERO;
    let mut grad_sq_sum = 0.0_f32;
    for n in &neighbours {
        r_squared.push(n.r_squared);
        let grad = spiky_gradient(n.r_vec, h);
        grad_sum = grad_sum.add(grad);
        grad_sq_sum += grad.dot(grad);
    }
    let density = estimate_density(params.particle_mass, &r_squared, h);
    constraint_lambda(
        density,
        params.rest_density,
        grad_sum,
        grad_sq_sum,
        params.relaxation_epsilon,
    )
}

/// Bit-exact `CPU` twin of the `pbf_compute_lambda` kernel.
///
/// `positions` is the packed `(x, y, z, w)` lane buffer of the frozen predicted
/// positions; `hash` is the packed uniform spatial hash (`[start, count]` pairs
/// per cell followed by the index-ordered binned particle indices). Returns the
/// `lambdas` buffer (`params.particle_count` lanes), each lane the particle's
/// `XPBD` scaling factor. A degenerate request — a buffer that cannot hold the
/// declared particle count or a hash too short for the declared grid — yields
/// an all-zero `lambdas` buffer, matching the shader's "skip, do not crash"
/// contract.
#[must_use]
pub fn dispatch_pbf_compute_lambda(
    positions: &[f32],
    hash: &[u32],
    params: PbfDensityParams,
) -> Vec<f32> {
    let count = params.particle_count as usize;
    let cell_count = params.cell_count();
    if positions.len() < count * PBF_POSITION_FLOATS || hash.len() < 2 * cell_count {
        return vec![0.0_f32; count];
    }
    let mut lambdas = vec![0.0_f32; count];
    let mut i = 0usize;
    while i < count {
        lambdas[i] = lambda_at(positions, hash, params, cell_count, i);
        i += 1;
    }
    lambdas
}

/// Bit-exact `CPU` twin of the `water_pbf_density_solve` kernel.
///
/// Reads the frozen `positions` and the `lambdas` buffer the compute-lambda
/// pass filled, accumulates each particle's position correction
/// `delta_p_i = (1 / rho_0) * sum_j (lambda_i + lambda_j + s_corr) * grad W_ij`
/// over its neighbourhood (reusing the golden
/// [`position_correction`](super::super::pbf::position_correction) and
/// [`artificial_pressure`](super::super::pbf::artificial_pressure)), and returns
/// the projected `positions_out` buffer (`xyz` corrected, `w` carried through).
/// Out-of-range invocations, out-of-grid particles and degenerate requests pass
/// the position through unchanged.
#[must_use]
pub fn dispatch_pbf_density_solve(
    positions: &[f32],
    hash: &[u32],
    lambdas: &[f32],
    params: PbfDensityParams,
) -> Vec<f32> {
    let count = params.particle_count as usize;
    let cell_count = params.cell_count();
    // The output buffer is sized to the input positions; a short lambda buffer
    // or hash leaves everything as a pass-through copy.
    let mut out = positions.to_vec();
    if positions.len() < count * PBF_POSITION_FLOATS
        || lambdas.len() < count
        || hash.len() < 2 * cell_count
        || params.rest_density <= EPS
        || cell_count == 0
    {
        return out;
    }
    let tuning = params.tuning();
    let mut i = 0usize;
    while i < count {
        let p_i = read_position(positions, i);
        if cell_coord(params, p_i).is_none() {
            i += 1;
            continue;
        }
        let neighbours = gather_from_hash(positions, hash, params, cell_count, i);
        let mut contribs = Vec::with_capacity(neighbours.len());
        for n in &neighbours {
            contribs.push(NeighborContribution {
                lambda_j: lambdas[n.j],
                scorr: artificial_pressure(n.r_squared, tuning),
                gradient: spiky_gradient(n.r_vec, params.smoothing_radius),
            });
        }
        let delta = position_correction(lambdas[i], params.rest_density, &contribs);
        let corrected = p_i.add(delta);
        let base = i * PBF_POSITION_FLOATS;
        out[base] = corrected.x;
        out[base + 1] = corrected.y;
        out[base + 2] = corrected.z;
        // `w` lane already carried through by the `to_vec` copy.
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::kernels::{DispatchDomain, WaterKernel};
    use super::super::super::pbf::{bin_particles, gather_neighbors, PbfBins, PbfGrid};
    use super::*;

    const H: f32 = 1.0;

    fn test_params(count: u32, grid: PbfGrid) -> PbfDensityParams {
        PbfDensityParams {
            grid_origin: grid.origin,
            cell_size: grid.cell_size,
            rest_density: 1000.0,
            particle_mass: 12.0,
            smoothing_radius: H,
            relaxation_epsilon: 1.0e-3,
            artificial_pressure_k: 0.1,
            artificial_pressure_delta_q: 0.2,
            artificial_pressure_n: 4,
            particle_count: count,
            grid_nx: grid.nx,
            grid_ny: grid.ny,
            grid_nz: grid.nz,
        }
    }

    /// Flattens world positions into the `(x, y, z, w)` lane buffer, coding the
    /// index into the `w` lane so the pass-through test can prove `w` survives.
    fn flatten(positions: &[Vec3]) -> Vec<f32> {
        let mut out = vec![0.0_f32; positions.len() * PBF_POSITION_FLOATS];
        for (i, p) in positions.iter().enumerate() {
            let base = i * PBF_POSITION_FLOATS;
            out[base] = p.x;
            out[base + 1] = p.y;
            out[base + 2] = p.z;
            out[base + 3] = i as f32;
        }
        out
    }

    /// Packs the golden [`PbfBins`] into the shader's `[start, count]` header +
    /// index-region layout. This is the independent neighbour-finding path the
    /// parity tests bind the packed-hash twin against.
    fn pack_hash(grid: PbfGrid, bins: &PbfBins) -> Vec<u32> {
        let cell_count = grid.cell_count();
        let mut header = vec![0u32; 2 * cell_count];
        let mut indices = Vec::new();
        let mut offset = 0u32;
        for c in 0..cell_count {
            let bucket = &bins.cells[c];
            header[2 * c] = offset;
            header[2 * c + 1] = bucket.len() as u32;
            for &idx in bucket {
                indices.push(idx);
            }
            offset += bucket.len() as u32;
        }
        header.extend(indices);
        header
    }

    /// Independent reference `lambda_i`, driven by the golden `PbfGrid` /
    /// `gather_neighbors` path (not the packed hash). Mirrors the kernel math
    /// via the same golden primitives but a different neighbour source.
    fn reference_lambda(
        positions: &[Vec3],
        grid: PbfGrid,
        bins: &PbfBins,
        params: PbfDensityParams,
        i: usize,
    ) -> f32 {
        let p_i = positions[i];
        let neighbours = gather_neighbors(grid, bins, positions, i as u32);
        let mut r_squared = vec![0.0_f32];
        let mut grad_sum = Vec3::ZERO;
        let mut grad_sq_sum = 0.0_f32;
        for &j in &neighbours {
            let r_vec = p_i.sub(positions[j as usize]);
            r_squared.push(r_vec.length_squared());
            let grad = spiky_gradient(r_vec, params.smoothing_radius);
            grad_sum = grad_sum.add(grad);
            grad_sq_sum += grad.dot(grad);
        }
        let density = estimate_density(params.particle_mass, &r_squared, params.smoothing_radius);
        constraint_lambda(
            density,
            params.rest_density,
            grad_sum,
            grad_sq_sum,
            params.relaxation_epsilon,
        )
    }

    /// A small deterministic clump of particles that share a few cells so the
    /// neighbourhood is non-trivial.
    fn clump() -> (Vec<Vec3>, PbfGrid) {
        let grid = PbfGrid {
            origin: Vec3::new(0.0, 0.0, 0.0),
            cell_size: H,
            nx: 4,
            ny: 4,
            nz: 4,
        };
        let positions = vec![
            Vec3::new(0.3, 0.3, 0.3),
            Vec3::new(0.7, 0.4, 0.5),
            Vec3::new(1.2, 0.6, 0.4),
            Vec3::new(0.5, 1.1, 0.8),
            Vec3::new(1.6, 1.4, 1.2),
            Vec3::new(0.9, 0.9, 0.9),
            Vec3::new(2.4, 2.2, 2.1),
            Vec3::new(0.2, 0.8, 0.6),
        ];
        (positions, grid)
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        let src = WATER_PBF_DENSITY_WESL;
        assert!(src.contains("fn pbf_compute_lambda"));
        assert!(src.contains("fn water_pbf_density_solve"));
        assert!(src.contains(WaterKernel::PbfComputeLambda.wesl_entry_point()));
        assert!(src.contains(WaterKernel::PbfDensitySolve.wesl_entry_point()));
        assert!(src.contains("@workgroup_size(64)"));
        assert!(src.contains("struct PbfParams"));
        assert!(src.contains("var<storage, read>"));
        assert!(src.contains("var<storage, read_write>"));
        assert!(src.contains("var<uniform>"));

        // Descriptor parity: four storage buffers, one uniform, no textures,
        // a 64-lane group over the particle domain.
        let desc = WaterKernel::PbfComputeLambda.descriptor();
        assert_eq!(desc.layout.storage_buffers, 4);
        assert_eq!(desc.layout.uniform_buffers, 1);
        assert_eq!(desc.layout.storage_textures, 0);
        assert_eq!(desc.layout.sampled_textures, 0);
        assert_eq!(desc.workgroup.x, 64);
        assert_eq!(desc.workgroup.y, 1);
        assert_eq!(desc.workgroup.z, 1);
        assert_eq!(desc.domain, DispatchDomain::Particle);
        // The solve pass shares the identical layout.
        assert_eq!(
            WaterKernel::PbfDensitySolve.descriptor().layout,
            desc.layout
        );
    }

    #[test]
    fn compute_lambda_matches_golden_gather() {
        let (positions, grid) = clump();
        let params = test_params(positions.len() as u32, grid);
        let bins = bin_particles(grid, &positions);
        let hash = pack_hash(grid, &bins);
        let flat = flatten(&positions);

        let lambdas = dispatch_pbf_compute_lambda(&flat, &hash, params);
        assert_eq!(lambdas.len(), positions.len());

        // Non-trivial: at least one particle must have a non-zero lambda, else
        // the parity below would be vacuous.
        assert!(lambdas.iter().any(|l| l.to_bits() != 0.0_f32.to_bits()));

        for (i, lambda) in lambdas.iter().enumerate() {
            let reference = reference_lambda(&positions, grid, &bins, params, i);
            assert_eq!(
                lambda.to_bits(),
                reference.to_bits(),
                "lambda mismatch at particle {i}"
            );
        }
    }

    #[test]
    fn density_solve_matches_golden_pipeline() {
        let (positions, grid) = clump();
        let params = test_params(positions.len() as u32, grid);
        let bins = bin_particles(grid, &positions);
        let hash = pack_hash(grid, &bins);
        let flat = flatten(&positions);

        // Compose the two passes exactly as the host would.
        let lambdas = dispatch_pbf_compute_lambda(&flat, &hash, params);
        let out = dispatch_pbf_density_solve(&flat, &hash, &lambdas, params);
        assert_eq!(out.len(), flat.len());

        // Independent reference: reference lambdas, then reference corrections
        // through the golden `gather_neighbors` path.
        let ref_lambdas: Vec<f32> = (0..positions.len())
            .map(|i| reference_lambda(&positions, grid, &bins, params, i))
            .collect();
        let tuning = params.tuning();

        let mut moved = false;
        for i in 0..positions.len() {
            let p_i = positions[i];
            let neighbours = gather_neighbors(grid, &bins, &positions, i as u32);
            let mut contribs = Vec::new();
            for &j in &neighbours {
                let r_vec = p_i.sub(positions[j as usize]);
                contribs.push(NeighborContribution {
                    lambda_j: ref_lambdas[j as usize],
                    scorr: artificial_pressure(r_vec.length_squared(), tuning),
                    gradient: spiky_gradient(r_vec, params.smoothing_radius),
                });
            }
            let delta = position_correction(ref_lambdas[i], params.rest_density, &contribs);
            let expected = p_i.add(delta);
            let base = i * PBF_POSITION_FLOATS;
            assert_eq!(out[base].to_bits(), expected.x.to_bits(), "x at {i}");
            assert_eq!(out[base + 1].to_bits(), expected.y.to_bits(), "y at {i}");
            assert_eq!(out[base + 2].to_bits(), expected.z.to_bits(), "z at {i}");
            // `w` lane carried through untouched (coded to the index).
            assert_eq!(out[base + 3].to_bits(), (i as f32).to_bits(), "w at {i}");
            if delta.length_squared() > 0.0 {
                moved = true;
            }
        }
        // The projection must actually move at least one particle, else the
        // parity would be vacuous.
        assert!(moved, "no particle was corrected; parity would be vacuous");
    }

    #[test]
    fn degenerate_requests_pass_through() {
        let (positions, grid) = clump();
        let flat = flatten(&positions);
        let bins = bin_particles(grid, &positions);
        let hash = pack_hash(grid, &bins);

        // Zero rest density: no constraint, positions untouched, zero lambdas.
        let mut params = test_params(positions.len() as u32, grid);
        params.rest_density = 0.0;
        let lambdas = dispatch_pbf_compute_lambda(&flat, &hash, params);
        assert!(lambdas.iter().all(|l| l.to_bits() == 0.0_f32.to_bits()));
        let out = dispatch_pbf_density_solve(&flat, &hash, &lambdas, params);
        assert_eq!(out, flat);

        // Mis-sized positions buffer: all zero lambdas, pass-through solve.
        let params = test_params(positions.len() as u32, grid);
        let short = &flat[..flat.len() - 1];
        let lambdas = dispatch_pbf_compute_lambda(short, &hash, params);
        assert!(lambdas.iter().all(|l| l.to_bits() == 0.0_f32.to_bits()));
        let out = dispatch_pbf_density_solve(short, &hash, &lambdas, params);
        assert_eq!(out, short.to_vec());
    }

    #[test]
    fn out_of_grid_particle_is_untouched() {
        let (mut positions, grid) = clump();
        // Push the last particle well outside the grid bounds.
        let last = positions.len() - 1;
        positions[last] = Vec3::new(-5.0, -5.0, -5.0);
        let params = test_params(positions.len() as u32, grid);
        let bins = bin_particles(grid, &positions);
        let hash = pack_hash(grid, &bins);
        let flat = flatten(&positions);

        let lambdas = dispatch_pbf_compute_lambda(&flat, &hash, params);
        assert_eq!(
            lambdas[last].to_bits(),
            0.0_f32.to_bits(),
            "out-of-grid particle must have zero lambda"
        );
        let out = dispatch_pbf_density_solve(&flat, &hash, &lambdas, params);
        let base = last * PBF_POSITION_FLOATS;
        assert_eq!(out[base].to_bits(), (-5.0_f32).to_bits());
        assert_eq!(out[base + 1].to_bits(), (-5.0_f32).to_bits());
        assert_eq!(out[base + 2].to_bits(), (-5.0_f32).to_bits());
    }
}
