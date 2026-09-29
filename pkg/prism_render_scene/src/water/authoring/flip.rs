//! `FLIP`/`APIC` liquid-volume authoring preset.
//!
//! A `FLIP`/`APIC` volume carries its state on particles and transfers it
//! through a background `MAC` grid (`P2G` scatter, a pressure projection that
//! makes the grid velocity divergence-free, then `G2P` gather). Two things must
//! be internally consistent: the particle mass has to match the target liquid
//! density at the seeding sub-lattice, and the grid cell count the pressure
//! solve loops over has to match the authored `MAC` grid. This preset seeds a
//! deterministic block of liquid inside the grid, derives the particle mass
//! from the density, runs the seeded state through the architecture core's
//! [`plan_flip`] to clamp the `FLIP`/`PIC` blend and pick the pressure solver,
//! and lights the `FLIP` solve plus its screen-space surface reconstruction. An
//! empty fill region yields an honest no-op body.

use prism_render_architecture::water::flip::{plan_flip, FlipParams, PressureSolverThresholds};
use prism_render_architecture::water::gpu::buffers::WaterBufferCounts;
use prism_render_architecture::water::gpu::pipeline::WaterPasses;

use crate::water::abi::{GpuFlipParticle, GpuFlipSimParams, GpuFlipSurfaceParams};
use crate::water::bind_groups::WaterSurfaceExtent;
use crate::water::body::WaterBody;

/// An art-directable description of a `FLIP`/`APIC` liquid volume.
///
/// Every field is a plain scalar so the preset is safe to expose across the
/// crate boundary. Construct one with [`FlipPoolPreset::default`] and override
/// the few fields that matter; the particle mass, cell count and blend are
/// derived so they can never drift out of sync with the seeded volume.
#[derive(Clone, Copy, Debug)]
pub struct FlipPoolPreset {
    /// `MAC` grid resolution in cells along x / y / z (`>= 1` each).
    pub grid: [u32; 3],
    /// `MAC` cell edge length (m, `> 0`).
    pub cell_size: f32,
    /// World-space corner of cell `(0, 0, 0)`.
    pub origin: [f32; 3],
    /// Fill region in cells (from the origin corner) seeded with liquid; must
    /// fit inside `grid`.
    pub fill: [u32; 3],
    /// Particles seeded per cell axis, so each filled cell carries
    /// `subdivisions^3` particles on a regular sub-lattice (`>= 1`).
    pub subdivisions: u32,
    /// Target liquid density (kg/m^3, `> 0`) fixing the per-particle mass; water
    /// is `~1000`.
    pub density: f32,
    /// `FLIP`/`PIC` blend in `0..=1` (`0` = `PIC`, stable/dissipative; `1` =
    /// `FLIP`, energetic/noisier). Production values sit high, around `0.95`.
    pub flip_blend: f32,
    /// Whether the affine (`APIC`) velocity field is carried through transfer.
    pub use_affine: bool,
    /// Damped-`Jacobi` relaxation factor for the pressure solve, in `(0, 1]`.
    pub jacobi_omega: f32,
    /// `CFL` number bounding a sub-step (`> 0`); consumed by the solver's
    /// runtime timestep, validated here through [`plan_flip`].
    pub cfl: f32,
    /// Simulation sub-steps per frame (`>= 1`).
    pub substeps: u32,
    /// Pressure-projection iterations per sub-step (`>= 1`).
    pub solver_iterations: u32,
    /// Surface-reconstruction viewport width in pixels (`>= 1`).
    pub screen_width: u32,
    /// Surface-reconstruction viewport height in pixels (`>= 1`).
    pub screen_height: u32,
    /// Bilateral depth-filter half-width in pixels (`>= 0`).
    pub filter_radius: i32,
    /// Spatial Gaussian standard deviation of the bilateral filter, in pixels
    /// (`> 0`).
    pub spatial_sigma: f32,
    /// Range (depth) Gaussian standard deviation of the bilateral filter, in
    /// view-space depth units (`> 0`).
    pub range_sigma: f32,
    /// View-space width of a one-pixel step at unit depth (m/px); derived from
    /// the camera's vertical `FOV` and viewport height. The default is a
    /// plausible `~1080p`, `~1 rad FOV` value and should be overridden from the
    /// live camera.
    pub pixel_world_scale: f32,
}

impl Default for FlipPoolPreset {
    fn default() -> Self {
        Self {
            grid: [48, 24, 48],
            cell_size: 0.1,
            origin: [0.0, 0.0, 0.0],
            fill: [24, 12, 24],
            subdivisions: 2,
            density: 1000.0,
            flip_blend: 0.95,
            use_affine: true,
            jacobi_omega: 0.8,
            cfl: 1.0,
            substeps: 2,
            solver_iterations: 40,
            screen_width: 1920,
            screen_height: 1080,
            filter_radius: 8,
            spatial_sigma: 4.0,
            range_sigma: 0.1,
            pixel_world_scale: 0.001,
        }
    }
}

impl WaterBody {
    /// Builds a fully live `FLIP`/`APIC` liquid body from a high-level
    /// [`FlipPoolPreset`].
    ///
    /// The fill region is seeded deterministically on a `subdivisions^3`
    /// sub-lattice per cell, the per-particle mass is derived from the density
    /// and sub-cell volume, and the seeded state is run through [`plan_flip`] to
    /// clamp the blend and select the pressure solver. The `FLIP` solve and its
    /// screen-space surface reconstruction are lit. An empty fill region carries
    /// no liquid, so this returns the default no-op body.
    #[must_use]
    pub fn flip_pool(preset: FlipPoolPreset) -> Self {
        let [gx, gy, gz] = preset.grid;
        let subdivisions = preset.subdivisions.max(1);
        // Clamp the fill region to the grid so seeded particles stay in bounds.
        let fx = preset.fill[0].min(gx);
        let fy = preset.fill[1].min(gy);
        let fz = preset.fill[2].min(gz);

        if gx == 0
            || gy == 0
            || gz == 0
            || preset.cell_size <= 0.0
            || preset.density <= 0.0
            || fx == 0
            || fy == 0
            || fz == 0
        {
            return Self::default();
        }

        let dx = preset.cell_size;
        let [ox, oy, oz] = preset.origin;
        let sub = subdivisions;
        let sub_step = dx / sub as f32;

        // Seed the liquid block: each filled cell carries `sub^3` particles on a
        // centred regular sub-lattice, walked in deterministic (z, y, x) order.
        let per_cell = (sub * sub * sub) as usize;
        let mut particles =
            Vec::with_capacity((fx as usize) * (fy as usize) * (fz as usize) * per_cell);
        for cz in 0..fz {
            for cy in 0..fy {
                for cx in 0..fx {
                    let base_x = ox + cx as f32 * dx;
                    let base_y = oy + cy as f32 * dx;
                    let base_z = oz + cz as f32 * dx;
                    for sz in 0..sub {
                        for sy in 0..sub {
                            for sx in 0..sub {
                                let px = base_x + (sx as f32 + 0.5) * sub_step;
                                let py = base_y + (sy as f32 + 0.5) * sub_step;
                                let pz = base_z + (sz as f32 + 0.5) * sub_step;
                                particles.push(GpuFlipParticle {
                                    pos: [px, py, pz, 1.0],
                                    vel: [0.0, 0.0, 0.0, 0.0],
                                    c0: [0.0, 0.0, 0.0, 0.0],
                                    c1: [0.0, 0.0, 0.0, 0.0],
                                    c2: [0.0, 0.0, 0.0, 0.0],
                                });
                            }
                        }
                    }
                }
            }
        }
        let particle_count = particles.len() as u32;
        let cell_count = gx * gy * gz;

        // Per-particle mass so the seeded sub-lattice sits at the target density.
        let particle_mass = preset.density * sub_step * sub_step * sub_step;

        // Validate the transfer/solver plan against the seeded grid.
        let plan = plan_flip(
            FlipParams {
                flip_blend: preset.flip_blend,
                use_affine: preset.use_affine,
                cfl: preset.cfl.max(f32::EPSILON),
                dx,
            },
            cell_count,
            0.0,
            PressureSolverThresholds {
                jacobi_max_cells: 32 * 32 * 32,
                cg_max_cells: 64 * 64 * 64,
            },
        );

        let screen_w = preset.screen_width.max(1);
        let screen_h = preset.screen_height.max(1);
        let spatial_sigma = preset.spatial_sigma.max(f32::EPSILON);
        let range_sigma = preset.range_sigma.max(f32::EPSILON);

        Self {
            flip_particles: particles,
            flip_grid_cells: cell_count,
            flip_params: GpuFlipSimParams {
                origin: [ox, oy, oz, 0.0],
                dim: [gx, gy, gz, 0],
                dx,
                inv_dx: 1.0 / dx,
                flip_blend: plan.blend,
                particle_mass,
                jacobi_omega: preset.jacobi_omega.clamp(f32::EPSILON, 1.0),
                use_affine: u32::from(plan.affine),
                particle_count,
                cell_count,
            },
            flip_surface_params: GpuFlipSurfaceParams {
                resolution: [screen_w, screen_h],
                filter_radius: preset.filter_radius.max(0),
                spatial_sigma2: 2.0 * spatial_sigma * spatial_sigma,
                range_sigma2: 2.0 * range_sigma * range_sigma,
                pixel_world_scale: preset.pixel_world_scale,
                _pad: [0; 2],
            },
            flip_surface_extent: WaterSurfaceExtent {
                width: screen_w,
                height: screen_h,
            },
            passes: WaterPasses {
                flip: true,
                reconstruct: true,
                ..WaterPasses::default()
            },
            substeps: preset.substeps.max(1),
            solver_iterations: preset.solver_iterations.max(1),
            grid3d_voxels: cell_count,
            particle_count,
            screen_pixels: screen_w * screen_h,
            counts: WaterBufferCounts {
                flip_particles: particle_count,
                flip_grid_cells: cell_count,
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

    /// The preset expands into a live `FLIP` body whose schedule scatters,
    /// projects, gathers and reconstructs a surface.
    #[test]
    fn flip_preset_builds_a_live_volume() {
        let preset = FlipPoolPreset {
            grid: [8, 8, 8],
            fill: [4, 4, 4],
            subdivisions: 2,
            ..FlipPoolPreset::default()
        };
        let body = WaterBody::flip_pool(preset);

        // 4*4*4 filled cells * 2^3 particles = 512.
        assert_eq!(body.flip_particles.len(), 512);
        assert_eq!(body.particle_count, 512);
        assert_eq!(body.flip_grid_cells, 8 * 8 * 8);
        assert_eq!(body.counts.flip_particles, 512);

        let ex = body.as_extract();
        assert!(ex.flip);
        assert!(ex.reconstruct);
        let plan = prepare(&ex);
        let kinds: Vec<WaterKernel> = plan.dispatches.iter().map(|d| d.kernel).collect();
        assert!(kinds.contains(&WaterKernel::FlipP2G));
        assert!(kinds.contains(&WaterKernel::FlipPressureSolve));
        assert!(kinds.contains(&WaterKernel::FlipG2P));
        assert!(kinds.contains(&WaterKernel::SurfaceReconstruct));
    }

    /// The per-particle mass is derived from the density and the sub-cell
    /// volume, so the seeded lattice sits at the target density.
    #[test]
    fn particle_mass_matches_density_at_the_sublattice() {
        let preset = FlipPoolPreset {
            grid: [4, 4, 4],
            fill: [2, 2, 2],
            cell_size: 0.1,
            subdivisions: 2,
            density: 1000.0,
            ..FlipPoolPreset::default()
        };
        let body = WaterBody::flip_pool(preset);
        // sub_step = 0.1 / 2 = 0.05; m = 1000 * 0.05^3 = 0.125.
        assert!((body.flip_params.particle_mass - 0.125).abs() < 1.0e-5);
    }

    /// Every seeded particle lands inside the `MAC` grid bounds.
    #[test]
    fn seeded_particles_stay_inside_the_grid() {
        let preset = FlipPoolPreset {
            grid: [6, 5, 4],
            fill: [6, 5, 4],
            origin: [1.0, -2.0, 0.5],
            cell_size: 0.2,
            subdivisions: 2,
            ..FlipPoolPreset::default()
        };
        let body = WaterBody::flip_pool(preset);
        let p = &body.flip_params;
        let lo = p.origin;
        let hi = [
            p.origin[0] + p.dim[0] as f32 * p.dx,
            p.origin[1] + p.dim[1] as f32 * p.dx,
            p.origin[2] + p.dim[2] as f32 * p.dx,
        ];
        for part in &body.flip_particles {
            for axis in 0..3 {
                assert!(part.pos[axis] > lo[axis]);
                assert!(part.pos[axis] < hi[axis]);
            }
        }
    }

    /// An empty fill region carries no liquid, so the preset is an honest no-op.
    #[test]
    fn empty_fill_is_an_honest_noop() {
        let body = WaterBody::flip_pool(FlipPoolPreset {
            fill: [0, 12, 24],
            ..FlipPoolPreset::default()
        });
        assert!(body.flip_particles.is_empty());
        assert_eq!(body.particle_count, 0);
        assert!(prepare(&body.as_extract()).dispatches.is_empty());
    }
}
