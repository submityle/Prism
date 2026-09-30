//! Shallow-Water pond authoring preset.
//!
//! The Shallow-Water solver steps a height field `h(x, z)` and its depth-
//! averaged momentum under an explicit scheme, so its one hard correctness
//! constraint is the `CFL` bound: the timestep must be small enough that no
//! surface wave crosses more than a fraction of a cell per step, or the
//! explicit update diverges. This preset derives that bound from the *actual*
//! still-water wave celerity (`sqrt(g * depth)`) reported by the architecture
//! core rather than guessing a timestep, then feeds the pond from a steady
//! central inflow so the surface carries visible, radiating ripples. A pond
//! with no inflow is genuinely still, so the preset returns an honest no-op
//! body rather than lighting a solver that steps over dead water.

use prism_render_architecture::water::gpu::buffers::WaterBufferCounts;
use prism_render_architecture::water::gpu::pipeline::WaterPasses;
use prism_render_architecture::water::swe::{cfl_timestep, max_wave_speed, SweConfig, SweState};
use prism_render_architecture::water::GRAVITY;

use crate::water::abi::GpuWaterSweParams;
use crate::water::body::WaterBody;

/// An art-directable description of a Shallow-Water pond / lake surface.
///
/// Every field is a plain scalar so the preset is safe to expose across the
/// crate boundary. Construct one with [`ShallowWaterPreset::default`] and
/// override the few fields that matter.
#[derive(Clone, Copy, Debug)]
pub struct ShallowWaterPreset {
    /// Height-field cell count along x (`>= 1`).
    pub cells_x: u32,
    /// Height-field cell count along z (`>= 1`).
    pub cells_z: u32,
    /// Square cell edge length (m, `> 0`); the grid tiles `cells_x * cells_z`
    /// cells of this size across the world.
    pub cell_size: f32,
    /// Still rest depth (m, `> 0`); sets the wave celerity `sqrt(g * depth)`
    /// and therefore the `CFL`-bounded timestep.
    pub rest_depth: f32,
    /// Linear velocity damping per second in `0..=1` (drag toward rest).
    pub damping: f32,
    /// Requested frame timestep (s); the preset clamps it down to the `CFL`
    /// bound so the explicit step can never diverge.
    pub timestep: f32,
    /// Courant number in `(0, 1]` the explicit step must satisfy; `0.5` leaves
    /// generous headroom, `0.9` runs closer to the stability edge.
    pub cfl_number: f32,
    /// Steady central inflow rate (height per second, `>= 0`) injected at the
    /// pond center to feed radiating ripples; `0` leaves the pond still (and so
    /// yields an honest no-op body).
    pub source_rate: f32,
}

impl Default for ShallowWaterPreset {
    fn default() -> Self {
        Self {
            cells_x: 128,
            cells_z: 128,
            cell_size: 0.25,
            rest_depth: 1.0,
            damping: 0.05,
            timestep: 1.0 / 60.0,
            cfl_number: 0.5,
            source_rate: 0.15,
        }
    }
}

impl WaterBody {
    /// Builds a fully live Shallow-Water pond body from a high-level
    /// [`ShallowWaterPreset`].
    ///
    /// The still state's wave celerity is measured by the architecture core and
    /// fed through [`cfl_timestep`] so the stored timestep is guaranteed stable;
    /// a steady central inflow source drives the ripples. A degenerate grid or a
    /// zero inflow carries no motion, so this returns the default no-op body
    /// rather than a pond that steps over dead water.
    #[must_use]
    pub fn shallow_water(preset: ShallowWaterPreset) -> Self {
        // A degenerate grid, a dry bed or a pond with no inflow carries no
        // motion: keep the body an honest no-op rather than lighting a solver
        // that steps over dead water.
        if preset.cells_x == 0
            || preset.cells_z == 0
            || preset.cell_size <= 0.0
            || preset.rest_depth <= 0.0
            || preset.source_rate <= 0.0
        {
            return Self::default();
        }

        let cfg = SweConfig {
            nx: preset.cells_x,
            nz: preset.cells_z,
            dx: preset.cell_size,
            gravity: GRAVITY,
            damping: preset.damping.clamp(0.0, 1.0),
        };

        // Measure the still-water wave celerity and bound the timestep by it.
        let still = SweState::still(cfg, preset.rest_depth);
        let max_speed = max_wave_speed(&still, cfg);
        let cfl_dt = cfl_timestep(max_speed, cfg.dx, preset.cfl_number);
        let dt = preset.timestep.max(0.0).min(cfl_dt);

        let cells = cfg.nx * cfg.nz;

        // A dense per-cell source field: zero everywhere except a steady inflow
        // at the pond center. The interaction-source layout the `SWE` kernel
        // reads is `[.x depth delta, .y u impulse, .z v impulse, .w unused]`
        // (see `water_surface.wesl`), so the steady inflow is a pure height
        // source in `.x` and the momentum lanes stay zero — the spring adds
        // water without stirring a direction into it.
        let mut sources = vec![[0.0_f32; 4]; cells as usize];
        let center = (cfg.nz / 2) * cfg.nx + (cfg.nx / 2);
        sources[center as usize] = [preset.source_rate, 0.0, 0.0, 0.0];

        Self {
            swe_cells: cells,
            swe_sources: sources,
            swe_params: GpuWaterSweParams {
                nx: cfg.nx,
                nz: cfg.nz,
                dx: cfg.dx,
                gravity: cfg.gravity,
                damping: cfg.damping,
                dt,
                cfl_number: preset.cfl_number,
                max_wave_speed: max_speed,
            },
            passes: WaterPasses {
                swe: true,
                ..WaterPasses::default()
            },
            substeps: 1,
            grid2d_texels: cells,
            counts: WaterBufferCounts {
                swe_cells: cells,
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

    /// The preset expands into a live Shallow-Water body whose schedule steps
    /// the pond.
    #[test]
    fn shallow_water_preset_builds_a_live_pond() {
        let preset = ShallowWaterPreset {
            cells_x: 32,
            cells_z: 24,
            ..ShallowWaterPreset::default()
        };
        let body = WaterBody::shallow_water(preset);

        assert_eq!(body.swe_cells, 32 * 24);
        assert_eq!(body.swe_sources.len(), (32 * 24) as usize);
        assert_eq!(body.counts.swe_cells, 32 * 24);
        assert_eq!(body.grid2d_texels, 32 * 24);

        let ex = body.as_extract();
        assert!(ex.swe);
        let plan = prepare(&ex);
        assert!(plan
            .dispatches
            .iter()
            .any(|d| d.kernel == WaterKernel::SweStep));
    }

    /// The stored timestep never exceeds the `CFL` bound implied by the
    /// still-water wave celerity `sqrt(g * depth)`.
    #[test]
    fn timestep_respects_the_cfl_bound() {
        let preset = ShallowWaterPreset {
            cells_x: 16,
            cells_z: 16,
            cell_size: 0.25,
            rest_depth: 4.0,
            // Ask for a wildly too-large step; the preset must clamp it down.
            timestep: 10.0,
            cfl_number: 0.5,
            ..ShallowWaterPreset::default()
        };
        let body = WaterBody::shallow_water(preset);
        let celerity = (GRAVITY * 4.0).sqrt();
        let bound = preset.cfl_number * preset.cell_size / celerity;
        assert!(body.swe_params.dt <= bound + 1.0e-6);
        assert!(body.swe_params.dt > 0.0);
        // The stored celerity matches the still-water wave speed.
        assert!((body.swe_params.max_wave_speed - celerity).abs() < 1.0e-3);
    }

    /// A pond with no inflow is genuinely still, so the preset is an honest
    /// no-op: an empty body whose schedule dispatches nothing.
    #[test]
    fn still_pond_is_an_honest_noop() {
        let body = WaterBody::shallow_water(ShallowWaterPreset {
            source_rate: 0.0,
            ..ShallowWaterPreset::default()
        });
        assert_eq!(body.swe_cells, 0);
        assert!(body.swe_sources.is_empty());
        assert!(prepare(&body.as_extract()).dispatches.is_empty());
    }

    /// A single steady inflow sits at the pond center in the depth lane (`.x`)
    /// and adds water without stirring a momentum direction.
    #[test]
    fn central_inflow_is_a_pure_height_source() {
        let preset = ShallowWaterPreset {
            cells_x: 9,
            cells_z: 9,
            source_rate: 0.2,
            ..ShallowWaterPreset::default()
        };
        let body = WaterBody::shallow_water(preset);
        let center = (9 / 2) * 9 + (9 / 2);
        // The depth delta lives in `.x` (the only volume-changing lane).
        let injected: f32 = body.swe_sources.iter().map(|s| s[0]).sum();
        assert!(
            (injected - 0.2).abs() < 1.0e-6,
            "only the inflow adds height"
        );
        assert!((body.swe_sources[center as usize][0] - 0.2).abs() < 1.0e-6);
        // No source carries momentum (`.y`/`.z`) and the `.w` lane is unused.
        for s in &body.swe_sources {
            assert!(s[1].abs() < 1.0e-12);
            assert!(s[2].abs() < 1.0e-12);
            assert!(s[3].abs() < 1.0e-12);
        }
    }
}
