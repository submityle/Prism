//! Offline bake orchestration: driving the solver and encoder across every
//! probe of a grid to produce a finished parameter field.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Ties together the offline half of design section 43: for a source and a
//! probe [`crate::grid::ProbeGrid`], it runs the ARD solve
//! ([`crate::solver`]) once per probe set, derives a free-field reference by
//! re-solving in an all-air copy of the scene, encodes each response into
//! perceptual parameters ([`crate::encoding`]), and quantises the lot into a
//! [`ParameterField`] ([`crate::field`]). This is strictly an offline/task
//! step: it allocates and must not run on the audio thread.

use alloc::vec::Vec;
use bevy_math::Vec3;

use crate::encoding::{encode_perceptual, energy, DirectionalProbe, EncodeConfig, PerceptualParams};
use crate::field::{BitDepth, ParameterField, ParameterFieldBuilder};
use crate::grid::ProbeGrid;
use crate::solver::{ImpulseResponse, SolveConfig, VoxelScene, WaveSolver};

/// Placement of the baked source and whether to estimate arrival direction.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SourcePlacement {
    /// World-space source position; snapped to the nearest air voxel.
    pub position: Vec3,
    /// When `true`, each probe also solves its six axis neighbours so the
    /// encoder can estimate an arrival direction. This multiplies the probe
    /// workload and is skipped when only scalar parameters are needed.
    pub estimate_direction: bool,
}

impl SourcePlacement {
    /// A source at `position` with arrival-direction estimation enabled.
    #[must_use]
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            estimate_direction: true,
        }
    }
}

/// Configuration bundle for [`bake_field`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BakeConfig {
    /// Wave-solve configuration.
    pub solve: SolveConfig,
    /// Perceptual-encoding configuration.
    pub encode: EncodeConfig,
    /// Quantisation bit depth of the baked field.
    pub bit_depth: BitDepth,
}

impl Default for BakeConfig {
    #[inline]
    fn default() -> Self {
        Self {
            solve: SolveConfig::default(),
            encode: EncodeConfig::default(),
            bit_depth: BitDepth::Twelve,
        }
    }
}

/// The six axis-neighbour offsets in [`DirectionalProbe`] field order:
/// `x_neg, x_pos, y_neg, y_pos, z_neg, z_pos`.
const NEIGHBOR_OFFSETS: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Per-probe bookkeeping mapping into the flat solve-result vector.
struct ProbeCells {
    center: usize,
    neighbors: [Option<usize>; 6],
}

/// Offsets a cell triplet by a signed delta, returning `None` on underflow or
/// when the result leaves the scene.
#[must_use]
fn offset_cell(cell: [u32; 3], delta: (i32, i32, i32), dims: [u32; 3]) -> Option<[u32; 3]> {
    let x = cell[0] as i32 + delta.0;
    let y = cell[1] as i32 + delta.1;
    let z = cell[2] as i32 + delta.2;
    if x < 0 || y < 0 || z < 0 {
        return None;
    }
    let (x, y, z) = (x as u32, y as u32, z as u32);
    if x >= dims[0] || y >= dims[1] || z >= dims[2] {
        return None;
    }
    Some([x, y, z])
}

/// Bakes a parameter field for one source over `grid` inside `scene`.
///
/// The pipeline is: snap the source to air; solve the real scene for every
/// probe cell (plus neighbour cells when direction estimation is on); re-solve
/// an all-air copy to obtain a free-field direct-energy reference per probe;
/// encode each probe's response into [`PerceptualParams`]; and quantise the
/// result into a [`ParameterField`]. When the whole scene is rigid (no air at
/// all), every probe is encoded as fully occluded.
///
/// This allocates and runs the solver, so it belongs on an offline/task
/// thread, never the audio callback.
#[must_use]
pub fn bake_field(
    scene: &VoxelScene,
    grid: &ProbeGrid,
    source: &SourcePlacement,
    config: &BakeConfig,
) -> ParameterField {
    let probe_count = grid.probe_count();
    let mut builder = ParameterFieldBuilder::new(*grid);

    // Snap the source to air; a fully-rigid scene yields an occluded field.
    let Some(source_cell) = scene.nearest_air(source.position) else {
        for i in 0..probe_count {
            builder.set_probe(i, PerceptualParams::OCCLUDED);
        }
        return builder.build(config.bit_depth);
    };

    let dims = scene.dims();

    // Gather the real-scene probe (and neighbour) cells into one solve call.
    let mut probe_cells: Vec<[u32; 3]> = Vec::new();
    let mut meta: Vec<ProbeCells> = Vec::with_capacity(probe_count);
    for i in 0..probe_count {
        let [ix, iy, iz] = grid.triplet(i);
        let pos = grid.probe_position(ix, iy, iz);
        let cell = scene.nearest_air(pos).unwrap_or_else(|| scene.voxel_of(pos));
        let center = probe_cells.len();
        probe_cells.push(cell);
        let mut neighbors = [None; 6];
        if source.estimate_direction {
            for (slot, delta) in NEIGHBOR_OFFSETS.iter().enumerate() {
                if let Some(nc) = offset_cell(cell, *delta, dims)
                    && scene.is_air(nc[0], nc[1], nc[2])
                {
                    neighbors[slot] = Some(probe_cells.len());
                    probe_cells.push(nc);
                }
            }
        }
        meta.push(ProbeCells { center, neighbors });
    }

    let mut solver = WaveSolver::new(scene, config.solve);
    let responses = solver.solve(source_cell, &probe_cells);

    // Free-field reference: an all-air copy solved from the same source gives
    // the distance-only direct energy at each probe.
    let reference = free_field_reference(scene, grid, source, config);

    for (i, pc) in meta.iter().enumerate() {
        let center_ir = &responses[pc.center];
        let reference_energy = reference[i];
        let params = if source.estimate_direction {
            let nb = |slot: usize| -> &ImpulseResponse {
                match pc.neighbors[slot] {
                    Some(idx) => &responses[idx],
                    None => center_ir,
                }
            };
            let probe = DirectionalProbe {
                center: center_ir,
                x_neg: nb(0),
                x_pos: nb(1),
                y_neg: nb(2),
                y_pos: nb(3),
                z_neg: nb(4),
                z_pos: nb(5),
            };
            encode_perceptual(center_ir, Some(&probe), reference_energy, &config.encode)
        } else {
            encode_perceptual(center_ir, None, reference_energy, &config.encode)
        };
        builder.set_probe(i, params);
    }

    builder.build(config.bit_depth)
}

/// Solves an all-air copy of `scene` and returns the free-field direct energy
/// at each probe, used to normalise occlusion against distance.
#[must_use]
fn free_field_reference(
    scene: &VoxelScene,
    grid: &ProbeGrid,
    source: &SourcePlacement,
    config: &BakeConfig,
) -> Vec<f32> {
    let dims = scene.dims();
    let free = VoxelScene::new(scene.origin(), scene.cell_size(), dims[0], dims[1], dims[2]);
    let probe_count = grid.probe_count();
    let cells: Vec<[u32; 3]> = (0..probe_count)
        .map(|i| {
            let [ix, iy, iz] = grid.triplet(i);
            free.voxel_of(grid.probe_position(ix, iy, iz))
        })
        .collect();
    let source_cell = free
        .nearest_air(source.position)
        .unwrap_or_else(|| free.voxel_of(source.position));
    let mut solver = WaveSolver::new(&free, config.solve);
    let responses = solver.solve(source_cell, &cells);
    responses
        .iter()
        .map(|ir| {
            let onset = ir.onset_index(config.encode.onset_threshold);
            let win = ir.sample_at_ms(config.encode.direct_window_ms).max(1);
            energy::direct_energy(ir, onset, win)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Aabb, ProbeGrid};

    #[test]
    fn bake_produces_a_full_field() {
        let scene = VoxelScene::new(Vec3::ZERO, 0.5, 6, 6, 6);
        let grid = ProbeGrid::new(Aabb::new(Vec3::splat(0.25), Vec3::splat(2.75)), 3, 3, 3);
        let source = SourcePlacement::new(Vec3::splat(0.5));
        let config = BakeConfig::default();
        let field = bake_field(&scene, &grid, &source, &config);
        assert_eq!(field.probe_count(), 27);
        // Every probe decodes to a sane, bounded gain.
        for i in 0..field.probe_count() {
            let p = field.decode(i);
            assert!(p.direct_gain >= 0.0 && p.direct_gain <= 1.0001);
            assert!(p.wet_gain >= 0.0 && p.wet_gain <= 1.0001);
        }
    }

    #[test]
    fn wall_occludes_the_far_probe() {
        // A rigid slab across the middle of the scene should drop the direct
        // gain for probes shadowed from the source.
        let mut scene = VoxelScene::new(Vec3::ZERO, 0.5, 8, 4, 4);
        for z in 0..4 {
            for y in 0..4 {
                scene.set_solid(4, y, z, true);
            }
        }
        let grid = ProbeGrid::new(Aabb::new(Vec3::new(0.25, 1.0, 1.0), Vec3::new(3.75, 1.0, 1.0)), 2, 1, 1);
        // Source on the near (low-x) side.
        let source = SourcePlacement {
            position: Vec3::new(0.5, 1.0, 1.0),
            estimate_direction: false,
        };
        let mut config = BakeConfig::default();
        config.solve.duration_s = 0.05;
        let field = bake_field(&scene, &grid, &source, &config);
        let near = field.decode(grid_index(&field.grid().clone(), 0));
        let far = field.decode(grid_index(&field.grid().clone(), 1));
        assert!(
            near.direct_gain >= far.direct_gain,
            "near {} should not be quieter than far {}",
            near.direct_gain,
            far.direct_gain
        );
    }

    fn grid_index(grid: &ProbeGrid, x: u32) -> usize {
        grid.linear_index(x, 0, 0)
    }

    #[test]
    fn fully_rigid_scene_bakes_occluded() {
        let mut scene = VoxelScene::new(Vec3::ZERO, 1.0, 2, 2, 2);
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    scene.set_solid(x, y, z, true);
                }
            }
        }
        let grid = ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(2.0)), 2, 2, 2);
        let source = SourcePlacement::new(Vec3::splat(1.0));
        let field = bake_field(&scene, &grid, &source, &BakeConfig::default());
        let p = field.decode(0);
        assert!(p.direct_gain < 1e-3, "expected occluded, got {}", p.direct_gain);
    }
}
