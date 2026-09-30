//! Terrain-flooded coastline authoring preset (a `UE5` Water `Landscape` blend).
//!
//! `UE5` Water blends its ocean and lakes against the `Landscape` heightfield:
//! the terrain bed decides where water sits, how deep it is, and where the surf
//! zone meets dry land. This preset does the same against the dependency-free
//! [`prism_render_architecture::water`] core. It floods an authored terrain
//! heightfield under a sea level to derive a per-cell water depth and a chamfer
//! distance-to-shore field, sizes a Shallow-Water (`SWE`) grid that matches the
//! terrain (coarsened by an integer stride when the terrain would exceed the
//! cell cap), and lights both the `SWE` step and the semi-Lagrangian foam pass
//! the golden schedule already consumes.
//!
//! Two fields are baked from the flood:
//!
//! * The `SWE` interaction sources carry a wind-driven, momentum-only surface
//!   stress (`[.x` depth feed, `.y`/`.z` momentum drive, `.w` unused] — the same
//!   layout the lake and river presets and `water_surface.wesl` agree on),
//!   ramped *up* away from the shore over a `shore_falloff` band so the surf
//!   zone at the water's edge stays calm while the open water drifts.
//! * The foam coverage field seeds *coastal foam*: it concentrates at the
//!   shoreline and fades offshore across the same band, mirroring the persistent
//!   surf line `UE5` Water paints where waves meet the coast.
//!
//! A degenerate request (a terrain grid whose length disagrees with its
//! dimensions, a non-positive cell size, a coast with no wind, or a terrain that
//! never dips below the sea level) carries no live motion, so the preset returns
//! the default no-op body rather than lighting a solver over dead water.

use prism_render_architecture::water::coastline::{build_coastline_field, downsample_heightfield};
use prism_render_architecture::water::gpu::buffers::WaterBufferCounts;
use prism_render_architecture::water::gpu::pipeline::WaterPasses;
use prism_render_architecture::water::swe::{cfl_timestep, max_wave_speed, SweConfig, SweState};
use prism_render_architecture::water::{Vec2, GRAVITY};

use crate::water::abi::{GpuWaterFoamParams, GpuWaterSweParams};
use crate::water::body::WaterBody;

/// An art-directable description of a terrain-flooded coastline.
///
/// Construct one with [`CoastlinePreset::default`] (a small beach ramp under a
/// gentle onshore breeze) and override [`heights`](Self::heights) with the
/// terrain bed the water floods.
#[derive(Clone, Debug, PartialEq)]
pub struct CoastlinePreset {
    /// Row-major `terrain_nx * terrain_nz` terrain heights in meters. A sample
    /// below `sea_level` floods to open water; a sample above stays dry land.
    pub heights: Vec<f32>,
    /// Terrain cell count along x (`>= 1`).
    pub terrain_nx: u32,
    /// Terrain cell count along z (`>= 1`).
    pub terrain_nz: u32,
    /// Square terrain cell edge length in meters (`> 0`).
    pub cell_size: f32,
    /// World sea level in meters; terrain below it floods.
    pub sea_level: f32,
    /// Minimum flood depth (m, `>= 0`) a cell must exceed to count as wet, so a
    /// paper-thin puddle at the exact waterline is not solved.
    pub min_depth: f32,
    /// Width (m, `>= 0`) of the near-shore band. The wind drift fades to zero
    /// and the coastal foam peaks within it; `0` drives the whole basin
    /// uniformly and seeds no foam.
    pub shore_falloff: f32,
    /// World-space wind drift velocity (m/s) applied as a momentum-only surface
    /// stress across the open water; zero leaves a mirror-still coast (no-op).
    pub wind: Vec2,
    /// Linear velocity damping per second in `0..=1`; the wind drift settles
    /// against this drag rather than accelerating without bound.
    pub damping: f32,
    /// Requested frame timestep (s); clamped down to the `CFL` bound implied by
    /// the deepest wet cell so the explicit step can never diverge.
    pub timestep: f32,
    /// Courant number in `(0, 1]` the explicit step must satisfy.
    pub cfl_number: f32,
    /// Baseline foam decay rate per second at or above the reference speed
    /// (`>= 0`); feeds [`GpuWaterFoamParams::base_decay`].
    pub foam_base_decay: f32,
    /// Fraction of `foam_base_decay` still applied in still water, `0..=1`;
    /// feeds [`GpuWaterFoamParams::persistence_floor`].
    pub foam_persistence_floor: f32,
    /// Hard cap on the `SWE` grid cell count (`nx * nz`); a terrain that would
    /// exceed it is coarsened by an integer stride until it fits.
    pub max_cells: u32,
}

impl Default for CoastlinePreset {
    fn default() -> Self {
        // A 32 x 32 m beach ramp: dry dunes at the back, flooding to about two
        // meters at the seaward edge, under a gentle onshore breeze.
        let n = 32_usize;
        let mut heights = Vec::with_capacity(n * n);
        for j in 0..n {
            // Bed drops from +1 m (dry dune) to about -2 m (open water) along z.
            let h = 1.0 - (j as f32 / (n as f32 - 1.0)) * 3.0;
            for _ in 0..n {
                heights.push(h);
            }
        }
        Self {
            heights,
            terrain_nx: n as u32,
            terrain_nz: n as u32,
            cell_size: 1.0,
            sea_level: 0.0,
            min_depth: 0.05,
            shore_falloff: 6.0,
            wind: Vec2::new(0.0, 0.4),
            damping: 0.08,
            timestep: 1.0 / 60.0,
            cfl_number: 0.5,
            foam_base_decay: 0.6,
            foam_persistence_floor: 0.2,
            max_cells: 1 << 20,
        }
    }
}

/// Cubic smoothstep `3t^2 - 2t^3` on a clamped `0..=1` argument; pure arithmetic
/// (no transcendental) so it honors the core's determinism policy.
#[must_use]
fn smoothstep01(t: f32) -> f32 {
    let c = t.clamp(0.0, 1.0);
    c * c * (3.0 - 2.0 * c)
}

/// Smallest integer stride whose coarsened grid fits `cap` cells.
///
/// Coarsening `nx * nz` by a stride yields `nx.div_ceil(stride)` by
/// `nz.div_ceil(stride)` cells; this returns the least `stride >= 1` for which
/// that product is within `cap`. A grid already inside the cap needs stride `1`.
#[must_use]
fn coarsening_stride(nx: u32, nz: u32, cap: u32) -> u32 {
    let cap = cap.max(1) as usize;
    let mut stride = 1_u32;
    loop {
        let cx = nx.div_ceil(stride) as usize;
        let cz = nz.div_ceil(stride) as usize;
        if cx.saturating_mul(cz) <= cap {
            return stride;
        }
        stride += 1;
    }
}

impl WaterBody {
    /// Builds a fully live terrain-flooded coastline body from a
    /// [`CoastlinePreset`].
    ///
    /// The terrain heightfield is flooded under the sea level into a per-cell
    /// depth and a chamfer distance-to-shore field. A Shallow-Water grid matches
    /// the terrain (coarsened by an integer stride when it would exceed
    /// [`CoastlinePreset::max_cells`]); every wet cell is filled with a
    /// wind-driven, momentum-only surface stress ramped up away from the shore
    /// and with a coastal-foam coverage that peaks at the shoreline. The
    /// timestep is clamped to the `CFL` bound implied by the deepest wet cell. A
    /// coast that carries no motion (a degenerate terrain, no wind, or a bed
    /// that never floods) yields the default no-op body.
    #[must_use]
    pub fn coastline(preset: CoastlinePreset) -> Self {
        // The terrain grid must be well formed.
        if preset.terrain_nx == 0 || preset.terrain_nz == 0 || preset.cell_size <= 0.0 {
            return Self::default();
        }
        let expected = (preset.terrain_nx as usize).saturating_mul(preset.terrain_nz as usize);
        if preset.heights.len() != expected {
            return Self::default();
        }
        // A calm coast (no wind) carries no live surface motion: keep it an
        // honest no-op rather than stepping a solver over a mirror-still basin.
        if preset.wind.length_squared() <= f32::EPSILON {
            return Self::default();
        }

        // Coarsen the terrain by an integer stride until it fits the cell cap, so
        // a vast coast is covered edge to edge at a coarser step rather than
        // clipped down to a corner of the shoreline.
        let cap = preset.max_cells.max(4);
        let stride = coarsening_stride(preset.terrain_nx, preset.terrain_nz, cap);
        let (heights, nx, nz, cell) = if stride <= 1 {
            (
                preset.heights.clone(),
                preset.terrain_nx,
                preset.terrain_nz,
                preset.cell_size,
            )
        } else {
            match downsample_heightfield(
                &preset.heights,
                preset.terrain_nx,
                preset.terrain_nz,
                stride,
            ) {
                Some((coarse, cx, cz)) => (coarse, cx, cz, preset.cell_size * stride as f32),
                None => return Self::default(),
            }
        };

        // Flood the terrain against the sea level.
        let Some(field) =
            build_coastline_field(&heights, nx, nz, cell, preset.sea_level, preset.min_depth)
        else {
            return Self::default();
        };

        let cfg = SweConfig {
            nx: field.nx,
            nz: field.nz,
            dx: field.dx,
            gravity: GRAVITY,
            damping: preset.damping.clamp(0.0, 1.0),
        };

        // Bound the timestep by the celerity of the deepest wet cell, the fastest
        // wave the basin can carry.
        let still = SweState::still(cfg, field.max_depth);
        let max_speed = max_wave_speed(&still, cfg);
        let cfl_dt = cfl_timestep(max_speed, cfg.dx, preset.cfl_number);
        let dt = preset.timestep.max(0.0).min(cfl_dt);

        let cells = (cfg.nx * cfg.nz) as usize;
        let mut sources = vec![[0.0_f32; 4]; cells];
        let mut foam = vec![0.0_f32; cells];
        let falloff = preset.shore_falloff.max(0.0);
        let threshold = preset.min_depth.max(0.0);
        let wind = preset.wind;

        // Wind drift ramps up away from the shore so the surf zone stays calm;
        // coastal foam does the opposite, peaking at the waterline.
        for (idx, (&depth, &dist)) in field
            .depth
            .iter()
            .zip(field.shore_distance.iter())
            .enumerate()
        {
            if depth <= threshold {
                continue; // dry land carries no surface source
            }
            let ramp = if falloff <= f32::EPSILON {
                1.0
            } else {
                smoothstep01(dist / falloff)
            };
            sources[idx] = [0.0, wind.x * ramp, wind.y * ramp, 0.0];
            foam[idx] = if falloff <= f32::EPSILON {
                0.0
            } else {
                1.0 - ramp
            };
        }

        // A coast whose entire wetted area fell inside the shore band (all drift
        // faded to zero) carries no live momentum: keep it an honest no-op.
        let live = sources
            .iter()
            .any(|s| s[1].abs() > f32::EPSILON || s[2].abs() > f32::EPSILON);
        if !live {
            return Self::default();
        }

        let cell_count = cfg.nx * cfg.nz;
        let reference_speed = wind.length().max(0.1);
        Self {
            swe_cells: cell_count,
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
            foam_cells: cell_count,
            foam_sources: foam,
            foam_params: GpuWaterFoamParams {
                nx: cfg.nx,
                nz: cfg.nz,
                dx: cfg.dx,
                dt,
                base_decay: preset.foam_base_decay.max(0.0),
                persistence_floor: preset.foam_persistence_floor.clamp(0.0, 1.0),
                reference_speed,
                _pad: 0,
            },
            passes: WaterPasses {
                swe: true,
                foam: true,
                ..WaterPasses::default()
            },
            substeps: 1,
            grid2d_texels: cell_count,
            counts: WaterBufferCounts {
                swe_cells: cell_count,
                foam_cells: cell_count,
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

    /// A 9 x 9 terrain bowl: a central diamond dips below the sea level, ringed
    /// by dry land on every side, so the wet basin's shore is the interior ring
    /// (never the grid edge). Returns the preset.
    fn bowl() -> CoastlinePreset {
        let n = 9_usize;
        let mut heights = vec![1.2_f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let r = (i as i32 - 4).abs().max((j as i32 - 4).abs());
                let h = match r {
                    0 => -2.0,
                    1 => -1.2,
                    2 => -0.4,
                    3 => 0.4,
                    _ => 1.2,
                };
                heights[j * n + i] = h;
            }
        }
        CoastlinePreset {
            heights,
            terrain_nx: n as u32,
            terrain_nz: n as u32,
            cell_size: 1.0,
            sea_level: 0.0,
            min_depth: 0.05,
            shore_falloff: 5.0,
            wind: Vec2::new(0.5, 0.0),
            damping: 0.05,
            timestep: 1.0 / 60.0,
            cfl_number: 0.5,
            foam_base_decay: 0.6,
            foam_persistence_floor: 0.2,
            max_cells: 1 << 20,
        }
    }

    /// A flooded coast lights both the `SWE` step and the foam advection pass.
    #[test]
    fn a_flooded_coast_lights_swe_and_foam() {
        let body = WaterBody::coastline(bowl());
        assert!(body.swe_cells > 0);
        assert!(body.foam_cells > 0);
        let extract = body.as_extract();
        assert!(extract.swe && extract.foam);
        let plan = prepare(&extract);
        assert!(plan
            .dispatches
            .iter()
            .any(|d| d.kernel == WaterKernel::SweStep));
        assert!(plan
            .dispatches
            .iter()
            .any(|d| d.kernel == WaterKernel::FoamAdvect));
    }

    /// Terrain that never dips below the sea level floods nothing, so the preset
    /// is an honest no-op.
    #[test]
    fn dry_terrain_is_a_no_op() {
        let body = WaterBody::coastline(CoastlinePreset {
            heights: vec![5.0; 9 * 9],
            ..bowl()
        });
        assert_eq!(body, WaterBody::default());
    }

    /// A coast under no wind carries no live surface motion, so it is a no-op.
    #[test]
    fn a_windless_coast_is_a_no_op() {
        let body = WaterBody::coastline(CoastlinePreset {
            wind: Vec2::ZERO,
            ..bowl()
        });
        assert_eq!(body, WaterBody::default());
    }

    /// Coastal foam concentrates at the shoreline and fades offshore, the mirror
    /// image of the wind drift, which ramps up into the open water.
    #[test]
    fn coastal_foam_concentrates_near_the_shore() {
        let body = WaterBody::coastline(bowl());
        let center = 4 * 9 + 4; // deepest, farthest-from-shore wet cell
        let near_shore = 2 * 9 + 4; // shallow wet cell one ring off the bank

        // The bowl center sits well offshore: strong drift, little foam.
        let cx = body.swe_sources[center][1];
        let cz = body.swe_sources[center][2];
        let center_drive = (cx * cx + cz * cz).sqrt();
        let sx = body.swe_sources[near_shore][1];
        let sz = body.swe_sources[near_shore][2];
        let shore_drive = (sx * sx + sz * sz).sqrt();
        assert!(
            center_drive > shore_drive,
            "open-water drift {center_drive} must exceed the surf-zone drift {shore_drive}"
        );

        // Foam is the mirror: heavier at the shore than offshore.
        assert!(
            body.foam_sources[near_shore] > body.foam_sources[center],
            "surf-line foam {} must exceed offshore foam {}",
            body.foam_sources[near_shore],
            body.foam_sources[center]
        );
        let max_foam = body.foam_sources.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max_foam > 0.0,
            "a flooded coast must seed some coastal foam"
        );
    }

    /// A terrain far larger than the cell cap is coarsened by an integer stride
    /// so the live grid never exceeds the budget.
    #[test]
    fn an_oversized_terrain_is_coarsened_to_the_cap() {
        let (nx, nz) = (200_u32, 200_u32);
        let mut heights = vec![1.0_f32; (nx * nz) as usize];
        for j in 0..nz {
            for i in 0..nx {
                if j >= nz / 2 {
                    heights[(j * nx + i) as usize] = -1.0; // seaward half floods
                }
            }
        }
        let body = WaterBody::coastline(CoastlinePreset {
            heights,
            terrain_nx: nx,
            terrain_nz: nz,
            cell_size: 1.0,
            max_cells: 1024,
            ..bowl()
        });
        assert!(body.swe_cells > 0);
        assert!(
            body.swe_cells <= 1024,
            "cells={} exceeds cap",
            body.swe_cells
        );
    }

    /// The bake is deterministic: the same preset yields byte-identical fields.
    #[test]
    fn coastline_bake_is_deterministic() {
        let a = WaterBody::coastline(bowl());
        let b = WaterBody::coastline(bowl());
        assert_eq!(a.swe_cells, b.swe_cells);
        assert_eq!(a.swe_sources, b.swe_sources);
        assert_eq!(a.foam_sources, b.foam_sources);
        assert!((a.swe_params.dt - b.swe_params.dt).abs() < 1.0e-12);
    }
}
