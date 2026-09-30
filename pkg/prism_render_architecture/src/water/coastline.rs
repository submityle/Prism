//! Coastline flood field: a terrain heightfield blended against a sea level.
//!
//! `UE5` Water authors an ocean-meets-land coast with a *Landscape blend*: the
//! water body floods every part of the terrain that sits below the still sea
//! level, and a shoreline mask drives where the coast breaks into foam, where
//! sand goes wet, and how far the surf reaches inland. This module is the pure,
//! dependency-free core of that blend. Given a sampled terrain heightfield and
//! a sea level it derives, per cell:
//!
//! * the still flood depth `max(0, sea_level - height)` (zero on dry land), and
//! * the chamfer distance, in meters, from each wet cell to the nearest dry
//!   cell or grid boundary (the shoreline band a coastal foam / wetness pass
//!   tapers against).
//!
//! It also offers an integer-stride [`downsample_heightfield`] so an authoring
//! layer can coarsen an over-resolved terrain tile to fit a solver-grid budget
//! before flooding, exactly the way a large `UE5` Landscape is decimated to the
//! water grid it can afford.
//!
//! Everything here is deterministic and allocation-only (no `f32` equality, no
//! sampling, no hidden state): the flood depth is a clamped subtraction and the
//! shoreline distance is a two-pass chamfer transform over the wet mask. A
//! degenerate request (empty field, mismatched length, or a terrain that never
//! dips below the sea level) returns [`None`] rather than a fabricated coast.

use alloc::vec;
use alloc::vec::Vec;

/// Still flood depth of one terrain sample under a sea level, in meters.
///
/// Returns `sea_level - terrain_height` where the terrain sits below the sea
/// level and `0` everywhere else, so dry land never reports negative water.
#[must_use]
pub fn flood_depth(terrain_height: f32, sea_level: f32) -> f32 {
    (sea_level - terrain_height).max(0.0)
}

/// A terrain heightfield flooded against a sea level.
///
/// The `depth` and `shore_distance` fields are row-major `nx * nz` grids in the
/// same layout as the terrain input: `index = row * nx + col`. `shore_distance`
/// is `0` on dry land and on the wet cells that touch a dry cell or the grid
/// edge, and grows toward the deepest interior.
#[derive(Clone, Debug, PartialEq)]
pub struct CoastlineField {
    /// Cell count along x (`> 0`).
    pub nx: u32,
    /// Cell count along z (`> 0`).
    pub nz: u32,
    /// Square cell edge length in meters (`> 0`), carried through so a consumer
    /// can size a solver grid without re-deriving it.
    pub dx: f32,
    /// Per-cell still flood depth in meters; `0` on dry land.
    pub depth: Vec<f32>,
    /// Per-cell chamfer distance in meters from each wet cell to the nearest dry
    /// cell or grid boundary; `0` on dry land and shore-adjacent wet cells.
    pub shore_distance: Vec<f32>,
    /// Number of wet cells (`depth` above the wet threshold).
    pub wet_cells: u32,
    /// Deepest flood depth over the field, in meters.
    pub max_depth: f32,
}

/// Floods a terrain heightfield against a sea level.
///
/// `heights` is a row-major `nx * nz` terrain grid in meters. A cell is *wet*
/// when its [`flood_depth`] strictly exceeds `min_depth` (a non-negative
/// threshold that suppresses a paper-thin puddle at the exact shoreline). The
/// returned [`CoastlineField`] carries the per-cell depth, the chamfer shore
/// distance, and the wet-cell tally.
///
/// Returns [`None`] when the grid is degenerate (`nx`/`nz`/`dx` non-positive or
/// `heights.len() != nx * nz`) or when no cell floods, so the caller can emit an
/// honest no-op body rather than a coast with no water.
#[must_use]
pub fn build_coastline_field(
    heights: &[f32],
    nx: u32,
    nz: u32,
    dx: f32,
    sea_level: f32,
    min_depth: f32,
) -> Option<CoastlineField> {
    if nx == 0 || nz == 0 || dx <= 0.0 {
        return None;
    }
    let cells = (nx as usize).checked_mul(nz as usize)?;
    if heights.len() != cells {
        return None;
    }
    let threshold = min_depth.max(0.0);

    let mut depth = vec![0.0_f32; cells];
    let mut wet_cells = 0_u32;
    let mut max_depth = 0.0_f32;
    for (dst, &h) in depth.iter_mut().zip(heights.iter()) {
        let d = flood_depth(h, sea_level);
        *dst = d;
        if d > threshold {
            wet_cells += 1;
            if d > max_depth {
                max_depth = d;
            }
        }
    }
    if wet_cells == 0 {
        return None;
    }

    let shore_distance = chamfer_shore_distance(&depth, nx, nz, dx, threshold);

    Some(CoastlineField {
        nx,
        nz,
        dx,
        depth,
        shore_distance,
        wet_cells,
        max_depth,
    })
}

/// Two-pass chamfer distance (in meters) from each wet cell to the nearest dry
/// cell or grid boundary.
///
/// Dry cells and out-of-grid neighbors are distance sources at `0`; a wet cell
/// touching one reads one step away. The 3x3 chamfer weights `dx` for the four
/// axial neighbors and `dx * sqrt(2)` for the diagonals give a bounded, isotropy-
/// corrected approximation to the true Euclidean shore distance in a single
/// forward and backward sweep.
#[must_use]
fn chamfer_shore_distance(depth: &[f32], nx: u32, nz: u32, dx: f32, threshold: f32) -> Vec<f32> {
    let w = nx as usize;
    let h = nz as usize;
    let axial = dx;
    let diag = dx * core::f32::consts::SQRT_2;
    // A ceiling larger than any reachable in-grid distance so the min-relaxation
    // never latches onto an unvisited sentinel.
    let far = axial * (w as f32 + h as f32) + diag;

    let mut dist = vec![0.0_f32; depth.len()];
    for (d, &depth_here) in dist.iter_mut().zip(depth.iter()) {
        // Dry cells stay at 0 (they are the shoreline sources); wet cells start
        // at the ceiling and relax down toward the nearest source.
        *d = if depth_here > threshold { far } else { 0.0 };
    }

    // Forward sweep: relax against the already-visited up/left neighbors. An
    // off-grid neighbor is the open shore boundary, a distance source at 0.
    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;
            if dist[idx] <= 0.0 {
                continue;
            }
            let mut best = dist[idx];
            // Left (axial).
            best = best.min(if col > 0 { dist[idx - 1] } else { 0.0 } + axial);
            // Top (axial).
            best = best.min(if row > 0 { dist[idx - w] } else { 0.0 } + axial);
            // Top-left (diagonal).
            best = best.min(
                if row > 0 && col > 0 {
                    dist[idx - w - 1]
                } else {
                    0.0
                } + diag,
            );
            // Top-right (diagonal).
            best = best.min(
                if row > 0 && col + 1 < w {
                    dist[idx - w + 1]
                } else {
                    0.0
                } + diag,
            );
            dist[idx] = best;
        }
    }

    // Backward sweep: relax against the down/right neighbors, with the same
    // off-grid-is-shore boundary sources.
    for row in (0..h).rev() {
        for col in (0..w).rev() {
            let idx = row * w + col;
            if dist[idx] <= 0.0 {
                continue;
            }
            let mut best = dist[idx];
            // Right (axial).
            best = best.min(if col + 1 < w { dist[idx + 1] } else { 0.0 } + axial);
            // Bottom (axial).
            best = best.min(if row + 1 < h { dist[idx + w] } else { 0.0 } + axial);
            // Bottom-right (diagonal).
            best = best.min(
                if row + 1 < h && col + 1 < w {
                    dist[idx + w + 1]
                } else {
                    0.0
                } + diag,
            );
            // Bottom-left (diagonal).
            best = best.min(
                if row + 1 < h && col > 0 {
                    dist[idx + w - 1]
                } else {
                    0.0
                } + diag,
            );
            dist[idx] = best;
        }
    }

    dist
}

/// A terrain heightfield coarsened by an integer stride via block averaging.
///
/// The output is a row-major `ceil(nx / stride) * ceil(nz / stride)` grid whose
/// every cell is the mean of the up-to `stride * stride` source samples it
/// covers (partial edge blocks average only the samples that exist). A stride of
/// `1` reproduces the input. Averaging is unbiased and deterministic, so a
/// coarsened coast floods to the same mean bed the full grid would.
///
/// Returns [`None`] on a degenerate request (`stride` zero, `nx`/`nz` zero, or a
/// `heights` length that does not match `nx * nz`).
#[must_use]
pub fn downsample_heightfield(
    heights: &[f32],
    nx: u32,
    nz: u32,
    stride: u32,
) -> Option<(Vec<f32>, u32, u32)> {
    if stride == 0 || nx == 0 || nz == 0 {
        return None;
    }
    let cells = (nx as usize).checked_mul(nz as usize)?;
    if heights.len() != cells {
        return None;
    }
    if stride == 1 {
        return Some((heights.to_vec(), nx, nz));
    }

    let w = nx as usize;
    let h = nz as usize;
    let s = stride as usize;
    let out_w = w.div_ceil(s);
    let out_h = h.div_ceil(s);
    let mut out = vec![0.0_f32; out_w * out_h];

    for (out_row, out_cell_row) in out.chunks_mut(out_w).enumerate() {
        let row0 = out_row * s;
        let row1 = (row0 + s).min(h);
        for (out_col, dst) in out_cell_row.iter_mut().enumerate() {
            let col0 = out_col * s;
            let col1 = (col0 + s).min(w);
            let mut sum = 0.0_f32;
            let mut count = 0_u32;
            for row in row0..row1 {
                let base = row * w;
                for col in col0..col1 {
                    sum += heights[base + col];
                    count += 1;
                }
            }
            // Every output cell covers at least the (row0, col0) sample, so the
            // count is always positive; guard defensively regardless.
            *dst = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }

    Some((out, out_w as u32, out_h as u32))
}

#[cfg(test)]
mod tests {
    use super::super::EPS;
    use super::*;

    /// The flood depth is a clamped subtraction: monotonic in the sea level and
    /// never negative on dry land.
    #[test]
    fn flood_depth_is_clamped_and_monotonic() {
        assert!((flood_depth(-3.0, 0.0) - 3.0).abs() < EPS);
        assert!(flood_depth(5.0, 0.0).abs() < EPS);
        // Raising the sea level can only raise (never lower) the flood depth.
        let low = flood_depth(-1.0, 0.5);
        let high = flood_depth(-1.0, 2.0);
        assert!(high > low);
    }

    /// A bowl dipping below the sea level floods; a plateau above it stays dry.
    #[test]
    fn a_bowl_floods_and_a_plateau_stays_dry() {
        // 3x3: a single deep center cell, dry rim.
        let heights = [
            2.0, 2.0, 2.0, //
            2.0, -4.0, 2.0, //
            2.0, 2.0, 2.0,
        ];
        let field = build_coastline_field(&heights, 3, 3, 1.0, 0.0, 0.0).expect("center floods");
        assert_eq!(field.wet_cells, 1);
        assert!((field.max_depth - 4.0).abs() < EPS);
        // The dry rim carries no water.
        assert!(field.depth[0].abs() < EPS);
        assert!((field.depth[4] - 4.0).abs() < EPS);

        // A terrain wholly above the sea level never floods.
        let plateau = [1.0_f32; 9];
        assert!(build_coastline_field(&plateau, 3, 3, 1.0, 0.0, 0.0).is_none());
    }

    /// The wet-cell tally matches the count of cells over the threshold.
    #[test]
    fn wet_cell_tally_matches_the_threshold() {
        // A ramp from -2 to +2 across four cells: two dip below the sea level.
        let heights = [-2.0, -1.0, 1.0, 2.0];
        let field = build_coastline_field(&heights, 4, 1, 1.0, 0.0, 0.0).expect("two flood");
        assert_eq!(field.wet_cells, 2);
        let counted = field.depth.iter().filter(|&&d| d > 0.0).count();
        assert_eq!(counted as u32, field.wet_cells);
    }

    /// The shore distance is `0` on shore-adjacent wet cells and grows toward
    /// the interior; deeper cells are never closer to shore than the rim.
    #[test]
    fn shore_distance_grows_inward() {
        // 5x5 basin: everything floods, so the center is the farthest from the
        // grid boundary (the only "dry" source is off-grid).
        let heights = [-1.0_f32; 25];
        let field = build_coastline_field(&heights, 5, 5, 1.0, 0.0, 0.0).expect("all wet");
        let center = 2 * 5 + 2;
        let edge = 0; // corner cell
        assert!(
            field.shore_distance[center] > field.shore_distance[edge],
            "center {} must be deeper into the basin than the corner {}",
            field.shore_distance[center],
            field.shore_distance[edge]
        );
        // Every distance is finite and non-negative.
        for d in &field.shore_distance {
            assert!(d.is_finite() && *d >= 0.0);
        }
    }

    /// Degenerate requests never fabricate a coast.
    #[test]
    fn degenerate_requests_return_none() {
        assert!(build_coastline_field(&[], 0, 0, 1.0, 0.0, 0.0).is_none());
        // Length mismatch.
        assert!(build_coastline_field(&[0.0, 0.0], 3, 3, 1.0, 0.0, 0.0).is_none());
        // Non-positive cell size.
        assert!(build_coastline_field(&[-1.0], 1, 1, 0.0, 0.0, 0.0).is_none());
    }

    /// Flooding the same terrain twice yields byte-identical fields.
    #[test]
    fn flooding_is_deterministic() {
        let heights = [
            -1.0, -2.0, 0.5, //
            -3.0, -1.5, 1.0, //
            0.2, -0.5, -2.5,
        ];
        let a = build_coastline_field(&heights, 3, 3, 0.5, 0.0, 0.0).expect("floods");
        let b = build_coastline_field(&heights, 3, 3, 0.5, 0.0, 0.0).expect("floods");
        assert_eq!(a, b);
    }

    /// A stride of one reproduces the input; a larger stride averages blocks and
    /// halves the dimensions (rounding up).
    #[test]
    fn downsample_averages_blocks() {
        let heights = [
            1.0, 3.0, 5.0, //
            3.0, 5.0, 7.0, //
            5.0, 7.0, 9.0,
        ];
        // Identity.
        let (same, sx, sz) = downsample_heightfield(&heights, 3, 3, 1).expect("identity");
        assert_eq!((sx, sz), (3, 3));
        assert_eq!(same, heights);

        // Stride 2 over 3x3 -> 2x2; the top-left block averages {1,3,3,5}=3.0.
        let (coarse, cx, cz) = downsample_heightfield(&heights, 3, 3, 2).expect("coarse");
        assert_eq!((cx, cz), (2, 2));
        assert!((coarse[0] - 3.0).abs() < EPS, "top-left mean {}", coarse[0]);
        // Bottom-right partial block is the single sample 9.0.
        assert!((coarse[3] - 9.0).abs() < EPS, "corner {}", coarse[3]);
    }

    /// A downsampled bed floods to the same mean level the full grid implies.
    #[test]
    fn downsample_preserves_the_flooded_mean() {
        let heights = [-2.0, -2.0, -2.0, -2.0];
        let (coarse, cx, cz) = downsample_heightfield(&heights, 2, 2, 2).expect("coarse");
        assert_eq!((cx, cz), (1, 1));
        assert!((coarse[0] + 2.0).abs() < EPS);
        let field = build_coastline_field(&coarse, cx, cz, 2.0, 0.0, 0.0).expect("floods");
        assert!((field.max_depth - 2.0).abs() < EPS);
    }

    /// Degenerate downsample requests return `None`.
    #[test]
    fn downsample_rejects_degenerate_input() {
        assert!(downsample_heightfield(&[0.0], 1, 1, 0).is_none());
        assert!(downsample_heightfield(&[0.0, 0.0], 3, 3, 2).is_none());
        assert!(downsample_heightfield(&[], 0, 0, 2).is_none());
    }
}
