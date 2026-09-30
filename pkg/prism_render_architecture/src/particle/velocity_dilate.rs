//! Velocity-field `tile`-max dilation: the `CPU`-verifiable reference for the
//! `McGuire` 2012 motion-blur velocity pre-pass (design §21).
//!
//! Reconstruction-filter motion blur (`McGuire` et al., *A Reconstruction Filter
//! for Plausible Motion Blur*, 2012) does not sample every screen pixel's own
//! velocity. Instead it runs a two-stage dilation over the per-pixel velocity
//! field so a fast-moving silhouette can still smear across the slower (or
//! static) pixels it sweeps past:
//!
//! 1. **`TileMax`** — the per-pixel velocity grid is down-sampled into `tile`s
//!    (typically `20px` square). Each `tile` keeps the single velocity of
//!    *largest magnitude* found inside it. This collapses a `width x height`
//!    field into a coarse `tile_cols x tile_rows` grid of dominant velocities.
//! 2. **`NeighborMax`** — every `tile` then looks at its `3x3` `tile`
//!    neighbourhood (clamped at the grid border) and adopts the largest-
//!    magnitude velocity among those nine `tile`s. This is what lets a blur
//!    "reach" one `tile` beyond the moving object, matching the paper's
//!    scatter-as-gather range.
//!
//! The dilated grid produced here is the *input* a later blur pass reads to
//! decide, per pixel, how far and along which direction to gather taps.
//!
//! **Strict boundaries — what this module deliberately does *not* do.** It does
//! not perform the motion-blur tap sampling / weighting itself: turning a
//! dominant velocity into along-vector taps and a weighted colour accumulation
//! is [`super::motion_blur`]'s job, and this module never touches it. It also
//! never *generates* the per-pixel velocity field (screen-space reprojection of
//! world motion) — that is [`super::motion_vectors`] territory; here the
//! per-pixel field is a *given input*. Finally it is not a general parallel
//! reduction primitive ([`super::gpu_reduce`]); it implements exactly one shape
//! of reduction: a `2D` `tile`d `TileMax` followed by a `3x3` `NeighborMax` over
//! a velocity field. The single dependency on a sibling module is the shared
//! `std430` layout helpers below.
//!
//! Everything is ordinary arithmetic plus `sqrt` only (used for the reported
//! Euclidean length); magnitude *comparisons* use squared length so no `sqrt`
//! is needed to decide which velocity is larger.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC2_STRIDE};

/// Byte size of the [`VelocityDilateConfig`] `std430` parameter block.
///
/// The block is four `u32` scalars (`width`, `height`, `tile_size`,
/// `num_tiles`), which pack into a single `vec4`-sized `std430` slot with no
/// interior padding: `4 * U32_STRIDE == 16` bytes.
pub const VELOCITY_DILATE_STD430_SIZE: usize = 4 * U32_STRIDE;

/// A two-component screen-space velocity, in pixels per frame.
///
/// The velocity field is inherently `2D` (screen plane), so this module carries
/// its own small vector type rather than reaching for a wider shared vector. It
/// uses only add / multiply plus `sqrt` for the reported length, never a
/// transcendental function, and derives only [`PartialEq`] (no [`Eq`] /
/// [`Hash`]) because it holds `f32` fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vel2 {
    /// Horizontal component (pixels per frame).
    pub x: f32,
    /// Vertical component (pixels per frame).
    pub y: f32,
}

impl Vel2 {
    /// The zero velocity.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Constructs a velocity from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Squared Euclidean magnitude (no `sqrt`).
    ///
    /// Magnitude comparisons throughout this module use this squared value so
    /// deciding which of two velocities is larger never needs a square root:
    /// `a` dominates `b` exactly when `a.length_squared() > b.length_squared()`.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Euclidean magnitude in pixels per frame.
    ///
    /// This is the only `sqrt` in the module and exists for callers that want
    /// the reported length; the dilation logic itself compares squared lengths.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }
}

/// Configuration for a velocity-field `tile`-max dilation pass.
///
/// `width` / `height` are the per-pixel velocity field dimensions; `tile_size`
/// is the square `tile` edge in pixels (typically `20`). The derived
/// `tile_cols` / `tile_rows` use `div_ceil` so a field whose size is not an
/// exact multiple of `tile_size` still gets a final partial `tile` covering the
/// remainder — every pixel belongs to exactly one `tile`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VelocityDilateConfig {
    /// Per-pixel velocity field width in pixels.
    pub width: usize,
    /// Per-pixel velocity field height in pixels.
    pub height: usize,
    /// Square `tile` edge length in pixels (clamped to at least `1`).
    pub tile_size: usize,
}

impl VelocityDilateConfig {
    /// Builds a configuration, clamping `tile_size` up to at least `1` so the
    /// `div_ceil` `tile` counts can never divide by zero.
    #[must_use]
    pub fn new(width: usize, height: usize, tile_size: usize) -> Self {
        Self {
            width,
            height,
            tile_size: tile_size.max(1),
        }
    }

    /// Number of `tile` columns: `ceil(width / tile_size)`.
    #[must_use]
    pub fn tile_cols(self) -> usize {
        self.width.div_ceil(self.tile_size)
    }

    /// Number of `tile` rows: `ceil(height / tile_size)`.
    #[must_use]
    pub fn tile_rows(self) -> usize {
        self.height.div_ceil(self.tile_size)
    }

    /// Total number of `tile`s in the dilated grid.
    #[must_use]
    pub fn num_tiles(self) -> usize {
        self.tile_cols() * self.tile_rows()
    }

    /// Byte size of the dilated `tile` grid as a `vec2<f32>` `std430` storage
    /// buffer, clamped up to one element for an empty grid.
    #[must_use]
    pub fn tile_bytes(self) -> usize {
        storage_bytes(VEC2_STRIDE, self.num_tiles())
    }

    /// Packs the configuration into its `std430` parameter block.
    ///
    /// Layout: `width`, `height`, `tile_size`, `num_tiles`, each a little-endian
    /// `u32`, filling exactly one `vec4`-sized slot
    /// ([`VELOCITY_DILATE_STD430_SIZE`] bytes). Each `usize` is narrowed with a
    /// saturating `try_from` so a pathological host value clamps to `u32::MAX`
    /// rather than panicking.
    #[must_use]
    pub fn to_std430(self) -> [u8; VELOCITY_DILATE_STD430_SIZE] {
        let mut bytes = [0u8; VELOCITY_DILATE_STD430_SIZE];
        let fields = [
            u32::try_from(self.width).unwrap_or(u32::MAX),
            u32::try_from(self.height).unwrap_or(u32::MAX),
            u32::try_from(self.tile_size).unwrap_or(u32::MAX),
            u32::try_from(self.num_tiles()).unwrap_or(u32::MAX),
        ];
        for (slot, value) in bytes.chunks_exact_mut(U32_STRIDE).zip(fields) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// Returns the larger-magnitude of two velocities, preferring `challenger` only
/// when it is *strictly* larger so ties keep the incumbent.
///
/// Magnitude is compared via squared length, so no `sqrt` is involved.
fn keep_larger(incumbent: Vel2, challenger: Vel2) -> Vel2 {
    if challenger.length_squared() > incumbent.length_squared() {
        challenger
    } else {
        incumbent
    }
}

/// Stage 1 — `TileMax`: down-samples the per-pixel velocity field into a
/// `tile_cols x tile_rows` grid where each `tile` holds the largest-magnitude
/// velocity found among its pixels.
///
/// `velocity` is the row-major per-pixel field of length `width * height`. The
/// returned grid is row-major (`tile` index `ty * tile_cols + tx`) and has
/// [`VelocityDilateConfig::num_tiles`] entries. Pixels beyond `velocity.len()`
/// (a short input) are treated as zero velocity rather than panicking.
#[must_use]
pub fn tile_max(velocity: &[Vel2], config: VelocityDilateConfig) -> Vec<Vel2> {
    let cols = config.tile_cols();
    let rows = config.tile_rows();
    let tile = config.tile_size;
    let width = config.width;
    let height = config.height;

    let mut grid = Vec::with_capacity(cols.saturating_mul(rows));
    for ty in 0..rows {
        let y0 = ty * tile;
        let y1 = (y0 + tile).min(height);
        for tx in 0..cols {
            let x0 = tx * tile;
            let x1 = (x0 + tile).min(width);
            let mut best = Vel2::ZERO;
            for y in y0..y1 {
                let row = y * width;
                for x in x0..x1 {
                    if let Some(&v) = velocity.get(row + x) {
                        best = keep_larger(best, v);
                    }
                }
            }
            grid.push(best);
        }
    }
    grid
}

/// Stage 2 — `NeighborMax`: over a `tile`-max grid, replaces each `tile` with
/// the largest-magnitude velocity among its `3x3` `tile` neighbourhood, with the
/// neighbourhood clamped at the grid border.
///
/// `grid` is the row-major `cols x rows` grid produced by [`tile_max`]. The
/// border clamp reuses the edge `tile`s (rather than wrapping or treating the
/// outside as zero), so a `tile` on the border still surveys a full `3x3`
/// window of in-bounds `tile`s. The output has the same shape as the input; an
/// input whose length disagrees with `cols * rows` yields an empty grid.
#[must_use]
pub fn neighbor_max(grid: &[Vel2], cols: usize, rows: usize) -> Vec<Vel2> {
    if grid.len() != cols.saturating_mul(rows) {
        return Vec::new();
    }
    let last_col = cols.saturating_sub(1);
    let last_row = rows.saturating_sub(1);

    let mut out = Vec::with_capacity(grid.len());
    for ty in 0..rows {
        // The three clamped neighbour rows; duplicates are harmless because the
        // reduction is idempotent under repeated maxima.
        let ny = [ty.saturating_sub(1), ty, (ty + 1).min(last_row)];
        for tx in 0..cols {
            let nx = [tx.saturating_sub(1), tx, (tx + 1).min(last_col)];
            let mut best = Vel2::ZERO;
            for &row in &ny {
                let base = row * cols;
                for &col in &nx {
                    best = keep_larger(best, grid[base + col]);
                }
            }
            out.push(best);
        }
    }
    out
}

/// Main entry point — the dilated dominant-velocity grid.
///
/// Runs [`tile_max`] then [`neighbor_max`] and returns the
/// [`VelocityDilateConfig::num_tiles`]-length row-major grid of dominant
/// velocities a downstream motion-blur pass reads. This module stops here: it
/// does not perform the blur tap sampling itself (see [`super::motion_blur`]).
#[must_use]
pub fn dominant_velocity(velocity: &[Vel2], config: VelocityDilateConfig) -> Vec<Vel2> {
    let tiles = tile_max(velocity, config);
    neighbor_max(&tiles, config.tile_cols(), config.tile_rows())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparing `f32` results, avoiding `==` on floats.
    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= CMP_EPS
    }

    fn vel_approx(a: Vel2, b: Vel2) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y)
    }

    #[test]
    fn uniform_field_dilates_to_the_same_velocity() {
        let cfg = VelocityDilateConfig::new(40, 40, 20);
        let field = alloc::vec![Vel2::new(1.5, -2.0); cfg.width * cfg.height];
        let out = dominant_velocity(&field, cfg);
        assert_eq!(out.len(), cfg.num_tiles());
        for v in out {
            assert!(vel_approx(v, Vel2::new(1.5, -2.0)));
        }
    }

    #[test]
    fn single_peak_dominates_its_own_tile() {
        // One 20px tile, with a single fast pixel among slow ones.
        let cfg = VelocityDilateConfig::new(20, 20, 20);
        let mut field = alloc::vec![Vel2::new(0.1, 0.0); cfg.width * cfg.height];
        field[7 * cfg.width + 3] = Vel2::new(9.0, 0.0);
        let tiles = tile_max(&field, cfg);
        assert_eq!(tiles.len(), 1);
        assert!(vel_approx(tiles[0], Vel2::new(9.0, 0.0)));
    }

    #[test]
    fn non_divisible_width_adds_a_partial_tile() {
        // 45 wide with 20px tiles -> ceil(45/20) = 3 columns.
        let cfg = VelocityDilateConfig::new(45, 20, 20);
        assert_eq!(cfg.tile_cols(), 3);
        assert_eq!(cfg.tile_rows(), 1);
        assert_eq!(cfg.num_tiles(), 3);
        // A peak in the trailing partial column must still be captured.
        let mut field = alloc::vec![Vel2::ZERO; cfg.width * cfg.height];
        field[5 * cfg.width + 44] = Vel2::new(0.0, 4.0);
        let tiles = tile_max(&field, cfg);
        assert!(vel_approx(tiles[2], Vel2::new(0.0, 4.0)));
    }

    #[test]
    fn non_divisible_height_adds_a_partial_row() {
        let cfg = VelocityDilateConfig::new(20, 45, 20);
        assert_eq!(cfg.tile_cols(), 1);
        assert_eq!(cfg.tile_rows(), 3);
        assert_eq!(cfg.num_tiles(), 3);
    }

    #[test]
    fn neighbor_max_propagates_a_peak_to_all_eight_neighbors() {
        // 3x3 tile grid, single hot tile in the centre.
        let cols = 3;
        let rows = 3;
        let mut grid = alloc::vec![Vel2::ZERO; cols * rows];
        grid[cols + 1] = Vel2::new(5.0, 0.0);
        let out = neighbor_max(&grid, cols, rows);
        // Every tile is within the 3x3 window of the centre, so all adopt it.
        for v in out {
            assert!(vel_approx(v, Vel2::new(5.0, 0.0)));
        }
    }

    #[test]
    fn neighbor_max_reaches_exactly_one_tile() {
        // A peak in a corner should not reach the opposite corner of a 3x3 grid.
        let cols = 3;
        let rows = 3;
        let mut grid = alloc::vec![Vel2::ZERO; cols * rows];
        grid[0] = Vel2::new(7.0, 0.0);
        let out = neighbor_max(&grid, cols, rows);
        // Opposite corner (2,2) is two tiles away -> stays zero.
        assert!(vel_approx(out[2 * cols + 2], Vel2::ZERO));
        // Adjacent (1,1) diagonal neighbour picks it up.
        assert!(vel_approx(out[cols + 1], Vel2::new(7.0, 0.0)));
    }

    #[test]
    fn magnitude_uses_length_not_a_single_component() {
        // (0,3) has magnitude 3; (2,2) has magnitude sqrt(8) < 3, even though its
        // component sum is larger. The full-magnitude winner must be (0,3).
        let cfg = VelocityDilateConfig::new(20, 20, 20);
        let mut field = alloc::vec![Vel2::ZERO; cfg.width * cfg.height];
        field[0] = Vel2::new(0.0, 3.0);
        field[1] = Vel2::new(2.0, 2.0);
        let tiles = tile_max(&field, cfg);
        assert!(vel_approx(tiles[0], Vel2::new(0.0, 3.0)));
    }

    #[test]
    fn zero_field_stays_zero() {
        let cfg = VelocityDilateConfig::new(60, 40, 20);
        let field = alloc::vec![Vel2::ZERO; cfg.width * cfg.height];
        let out = dominant_velocity(&field, cfg);
        assert_eq!(out.len(), cfg.num_tiles());
        for v in out {
            assert!(vel_approx(v, Vel2::ZERO));
        }
    }

    #[test]
    fn tile_max_picks_the_largest_of_mixed_magnitudes() {
        let cfg = VelocityDilateConfig::new(20, 20, 20);
        let mut field = alloc::vec![Vel2::ZERO; cfg.width * cfg.height];
        field[0] = Vel2::new(1.0, 1.0);
        field[1] = Vel2::new(3.0, 4.0); // magnitude 5, the max
        field[2] = Vel2::new(0.0, 2.5);
        let tiles = tile_max(&field, cfg);
        assert!(vel_approx(tiles[0], Vel2::new(3.0, 4.0)));
    }

    #[test]
    fn dominant_velocity_equals_neighbor_max_of_tile_max() {
        let cfg = VelocityDilateConfig::new(50, 30, 20);
        let mut field = alloc::vec![Vel2::new(0.2, 0.2); cfg.width * cfg.height];
        field[10 * cfg.width + 25] = Vel2::new(6.0, -1.0);
        let expected = neighbor_max(&tile_max(&field, cfg), cfg.tile_cols(), cfg.tile_rows());
        let actual = dominant_velocity(&field, cfg);
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.into_iter().zip(expected) {
            assert!(vel_approx(a, e));
        }
    }

    #[test]
    fn std430_size_is_one_vec4_slot() {
        assert_eq!(VELOCITY_DILATE_STD430_SIZE, 16);
        assert_eq!(4 * U32_STRIDE, VELOCITY_DILATE_STD430_SIZE);
    }

    #[test]
    fn std430_pack_roundtrips_fields() {
        let cfg = VelocityDilateConfig::new(45, 20, 20);
        let bytes = cfg.to_std430();
        let w = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let h = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let t = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let n = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(w, 45);
        assert_eq!(h, 20);
        assert_eq!(t, 20);
        assert_eq!(n, u32::try_from(cfg.num_tiles()).unwrap());
        assert_eq!(n, 3);
    }

    #[test]
    fn tile_bytes_matches_storage_bytes_of_grid() {
        let cfg = VelocityDilateConfig::new(80, 60, 20);
        assert_eq!(
            cfg.tile_bytes(),
            storage_bytes(VEC2_STRIDE, cfg.num_tiles())
        );
        assert_eq!(cfg.tile_bytes(), VEC2_STRIDE * cfg.num_tiles());
    }

    #[test]
    fn tile_bytes_reserves_one_element_for_empty_grid() {
        let cfg = VelocityDilateConfig::new(0, 0, 20);
        assert_eq!(cfg.num_tiles(), 0);
        assert_eq!(cfg.tile_bytes(), VEC2_STRIDE);
    }

    #[test]
    fn cmp_eps_is_a_small_positive_tolerance() {
        let eps = core::hint::black_box(CMP_EPS);
        assert!(eps > 0.0);
        assert!(eps < 1.0);
        // Values within CMP_EPS compare approximately equal without using `==`.
        assert!(approx(1.0, 1.0 + CMP_EPS * 0.5));
        assert!(!approx(1.0, 1.0 + CMP_EPS * 10.0));
    }

    #[test]
    fn length_squared_and_length_are_consistent() {
        let v = Vel2::new(3.0, 4.0);
        assert!(approx(v.length_squared(), 25.0));
        assert!(approx(v.length(), 5.0));
    }

    #[test]
    fn tile_size_is_clamped_to_at_least_one() {
        let cfg = VelocityDilateConfig::new(4, 4, 0);
        assert_eq!(cfg.tile_size, 1);
        // Each pixel becomes its own tile.
        assert_eq!(cfg.num_tiles(), 16);
    }

    #[test]
    fn neighbor_max_rejects_mismatched_grid_length() {
        let grid = alloc::vec![Vel2::ZERO; 5];
        // 5 != 2 * 3, so the guard returns an empty grid.
        assert!(neighbor_max(&grid, 2, 3).is_empty());
    }

    #[test]
    fn single_tile_covers_a_field_smaller_than_the_tile() {
        let cfg = VelocityDilateConfig::new(10, 10, 20);
        assert_eq!(cfg.num_tiles(), 1);
        let mut field = alloc::vec![Vel2::ZERO; cfg.width * cfg.height];
        field[cfg.width * 9 + 9] = Vel2::new(-2.0, 0.0);
        let out = dominant_velocity(&field, cfg);
        assert_eq!(out.len(), 1);
        assert!(vel_approx(out[0], Vel2::new(-2.0, 0.0)));
    }

    #[test]
    fn short_input_treats_missing_pixels_as_zero() {
        let cfg = VelocityDilateConfig::new(20, 20, 20);
        // Provide fewer velocities than width*height; must not panic.
        let field = alloc::vec![Vel2::new(1.0, 0.0); 5];
        let tiles = tile_max(&field, cfg);
        assert_eq!(tiles.len(), 1);
        assert!(vel_approx(tiles[0], Vel2::new(1.0, 0.0)));
    }
}
