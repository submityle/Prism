//! `Position-Based Fluids` (`PBF`) particle-pool authoring preset.
//!
//! `PBF` (Macklin & Müller) enforces incompressibility as a density constraint
//! solved with `XPBD` iterations over neighbours gathered from a uniform
//! spatial hash. Two things have to be internally consistent or the pool is
//! wrong from frame zero: the per-particle mass has to match the rest density
//! at the seeding lattice (`m = rho_0 * spacing^3`), and the spatial hash has to
//! actually cover the particle box at the smoothing radius. This preset seeds a
//! deterministic lattice of particles in a box, derives the mass from the rest
//! density, and sizes the hash grid from the real bounding box so every
//! neighbour lookup lands in range. An empty box yields an honest no-op body.

use prism_render_architecture::water::gpu::buffers::WaterBufferCounts;
use prism_render_architecture::water::gpu::pipeline::WaterPasses;
use prism_render_architecture::water::pbf::{plan_solve, PbfGrid, PbfParams};
use prism_render_architecture::water::Vec3;

use crate::water::abi::GpuPbfParams;
use crate::water::body::WaterBody;

/// An art-directable description of a `PBF` liquid pool.
///
/// Every field is a plain scalar so the preset is safe to expose across the
/// crate boundary. Construct one with [`PbfPoolPreset::default`] and override
/// the few fields that matter; the per-particle mass and the spatial-hash
/// extent are derived, not authored, so they can never drift out of sync.
#[derive(Clone, Copy, Debug)]
pub struct PbfPoolPreset {
    /// Particle lattice count along x / y / z (`>= 1` each).
    pub particles: [u32; 3],
    /// Seeding lattice spacing (m, `> 0`); also fixes the per-particle mass.
    pub spacing: f32,
    /// World-space minimum corner the lattice is seeded from.
    pub origin: [f32; 3],
    /// Target rest density `rho_0` (kg/m^3, `> 0`); water is `~1000`.
    pub rest_density: f32,
    /// `SPH` smoothing radius `h` (m, `> 0`); also the spatial-hash cell size
    /// and the support radius of both kernels. Should exceed `spacing` so each
    /// particle sees a full neighbourhood.
    pub smoothing_radius: f32,
    /// `XPBD` constraint-projection iterations per frame (`>= 1`).
    pub solver_iterations: u32,
    /// `XPBD` relaxation / compliance term added to the `lambda` denominator so
    /// the scaling factor never divides by (near) zero (`>= 0`).
    pub relaxation_epsilon: f32,
    /// Artificial-pressure strength `k` (`>= 0`) fighting particle clustering.
    pub artificial_pressure_k: f32,
    /// Artificial-pressure exponent `n` (`>= 1`).
    pub artificial_pressure_n: u32,
    /// Fraction of `h` at which the artificial-pressure reference kernel is
    /// sampled, in `0..=1` (Macklin & Müller use about `0.1..0.3`).
    pub artificial_pressure_delta_q: f32,
}

impl Default for PbfPoolPreset {
    fn default() -> Self {
        Self {
            particles: [16, 16, 16],
            spacing: 0.1,
            origin: [0.0, 0.0, 0.0],
            rest_density: 1000.0,
            smoothing_radius: 0.2,
            solver_iterations: 4,
            relaxation_epsilon: 100.0,
            artificial_pressure_k: 0.1,
            artificial_pressure_n: 4,
            artificial_pressure_delta_q: 0.2,
        }
    }
}

impl WaterBody {
    /// Builds a fully live `PBF` pool body from a high-level [`PbfPoolPreset`].
    ///
    /// The lattice is seeded deterministically, the per-particle mass is derived
    /// from the rest density (`m = rho_0 * spacing^3`) so the seeded pool sits at
    /// rest density, and the spatial hash is sized from the real particle box at
    /// the smoothing radius (with a one-cell halo) so every neighbour gather
    /// lands in range. An empty box carries no fluid, so this returns the
    /// default no-op body.
    #[must_use]
    pub fn pbf_pool(preset: PbfPoolPreset) -> Self {
        let [px, py, pz] = preset.particles;
        if px == 0
            || py == 0
            || pz == 0
            || preset.spacing <= 0.0
            || preset.smoothing_radius <= 0.0
            || preset.rest_density <= 0.0
        {
            return Self::default();
        }

        let spacing = preset.spacing;
        let [ox, oy, oz] = preset.origin;

        // Seed the lattice in deterministic row-major (z, y, x) order; `w = 1`
        // marks the particle active.
        let mut positions = Vec::with_capacity((px * py * pz) as usize);
        for k in 0..pz {
            for j in 0..py {
                for i in 0..px {
                    positions.push([
                        ox + i as f32 * spacing,
                        oy + j as f32 * spacing,
                        oz + k as f32 * spacing,
                        1.0,
                    ]);
                }
            }
        }
        let particle_count = px * py * pz;

        // Per-particle mass so the seeding lattice sits exactly at rest density.
        let particle_mass = preset.rest_density * spacing * spacing * spacing;

        // Size the spatial hash from the real particle box at cell size `h`,
        // with a one-cell halo on the low corner so the 3x3x3 neighbour stencil
        // never reads out of range.
        let h = preset.smoothing_radius;
        let span = |n: u32| -> u32 { (((n - 1) as f32 * spacing) / h) as u32 + 3 };
        let grid = PbfGrid {
            origin: Vec3::new(ox - h, oy - h, oz - h),
            cell_size: h,
            nx: span(px),
            ny: span(py),
            nz: span(pz),
        };
        let hash_entries = grid.cell_count() as u32;

        let params = PbfParams {
            rest_density: preset.rest_density,
            particle_mass,
            smoothing_radius: h,
            relaxation_epsilon: preset.relaxation_epsilon.max(0.0),
            artificial_pressure_k: preset.artificial_pressure_k.max(0.0),
            artificial_pressure_n: preset.artificial_pressure_n.max(1),
            artificial_pressure_delta_q: preset.artificial_pressure_delta_q.clamp(0.0, 1.0),
            solver_iterations: preset.solver_iterations.max(1),
        };
        // Validate the solve plan (iterations clamp, artificial-pressure gate).
        let solve = plan_solve(params);

        Self {
            pbf_positions: positions,
            pbf_hash_entries: hash_entries,
            pbf_params: GpuPbfParams {
                grid_origin: [grid.origin.x, grid.origin.y, grid.origin.z],
                cell_size: h,
                rest_density: params.rest_density,
                particle_mass: params.particle_mass,
                smoothing_radius: h,
                relaxation_epsilon: params.relaxation_epsilon,
                artificial_pressure_k: params.artificial_pressure_k,
                artificial_pressure_delta_q: params.artificial_pressure_delta_q,
                artificial_pressure_n: params.artificial_pressure_n,
                particle_count,
                grid_nx: grid.nx,
                grid_ny: grid.ny,
                grid_nz: grid.nz,
                _pad: 0,
            },
            passes: WaterPasses {
                pbf: true,
                ..WaterPasses::default()
            },
            substeps: 1,
            solver_iterations: solve.iterations,
            particle_count,
            counts: WaterBufferCounts {
                pbf_particles: particle_count,
                pbf_hash_entries: hash_entries,
                ..WaterBufferCounts::default()
            },
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::water::gpu::pipeline::prepare;
    use prism_render_architecture::water::kernels::WaterKernel;

    /// The preset expands into a live `PBF` body whose schedule solves density.
    #[test]
    fn pbf_preset_builds_a_live_pool() {
        let preset = PbfPoolPreset {
            particles: [4, 4, 4],
            ..PbfPoolPreset::default()
        };
        let body = WaterBody::pbf_pool(preset);

        assert_eq!(body.pbf_positions.len(), 64);
        assert_eq!(body.particle_count, 64);
        assert_eq!(body.counts.pbf_particles, 64);
        assert!(body.pbf_hash_entries > 0);
        assert_eq!(body.counts.pbf_hash_entries, body.pbf_hash_entries);

        let ex = body.as_extract();
        assert!(ex.pbf);
        let plan = prepare(&ex);
        assert!(plan
            .dispatches
            .iter()
            .any(|d| d.kernel == WaterKernel::PbfDensitySolve));
    }

    /// The per-particle mass is derived from the rest density at the seeding
    /// lattice, so the seeded pool sits exactly at rest density.
    #[test]
    fn particle_mass_matches_rest_density_at_the_lattice() {
        let preset = PbfPoolPreset {
            particles: [3, 3, 3],
            spacing: 0.1,
            rest_density: 1000.0,
            ..PbfPoolPreset::default()
        };
        let body = WaterBody::pbf_pool(preset);
        // m = rho_0 * spacing^3 = 1000 * 0.001 = 1.0
        assert!((body.pbf_params.particle_mass - 1.0).abs() < 1.0e-4);
    }

    /// The spatial hash covers every seeded particle: each lattice position
    /// hashes to an in-range cell.
    #[test]
    fn spatial_hash_covers_the_whole_lattice() {
        let preset = PbfPoolPreset {
            particles: [5, 4, 6],
            ..PbfPoolPreset::default()
        };
        let body = WaterBody::pbf_pool(preset);
        let p = &body.pbf_params;
        let grid = PbfGrid {
            origin: Vec3::new(p.grid_origin[0], p.grid_origin[1], p.grid_origin[2]),
            cell_size: p.cell_size,
            nx: p.grid_nx,
            ny: p.grid_ny,
            nz: p.grid_nz,
        };
        for pos in &body.pbf_positions {
            let world = Vec3::new(pos[0], pos[1], pos[2]);
            assert!(
                grid.cell_coord(world).is_some(),
                "every seeded particle must hash into the grid"
            );
        }
    }

    /// An empty box carries no fluid, so the preset is an honest no-op.
    #[test]
    fn empty_box_is_an_honest_noop() {
        let body = WaterBody::pbf_pool(PbfPoolPreset {
            particles: [0, 8, 8],
            ..PbfPoolPreset::default()
        });
        assert!(body.pbf_positions.is_empty());
        assert_eq!(body.particle_count, 0);
        assert!(prepare(&body.as_extract()).dispatches.is_empty());
    }
}
