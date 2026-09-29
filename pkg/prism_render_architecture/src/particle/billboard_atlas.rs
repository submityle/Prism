//! Multi-view `impostor` `atlas` view-selection and cell `UV` location
//! contract for the `billboard` particle renderer (design §16, §22).
//!
//! An `impostor` (a.k.a. octahedral `billboard`) replaces an expensive mesh with
//! a small grid of pre-baked view sprites: the mesh is rendered once per camera
//! direction into an `N`x`N` `atlas`, and at draw time the renderer selects the
//! baked view whose direction is closest to the current view vector and blends
//! the few nearest neighbours to hide the pop between discrete views. This is
//! the same technique shipped by Unreal's `impostor` baker, `Amplify Impostors`,
//! and countless foliage systems, reproduced here at the algorithm level for a
//! device-free, `CPU`-verifiable contract.
//!
//! This module owns exactly two responsibilities:
//!
//! 1. **View selection** — map a view direction to the nearest baked view cell
//!    (and, for blending, the nearest `k` cells with normalized weights) using
//!    an `octahedral` direction encoding so the whole sphere of directions maps
//!    uniformly onto the square grid.
//! 2. **Cell `UV` location** — turn a cell index into its normalized `UV`
//!    rectangle inside the `N`x`N` grid `atlas` with pure grid arithmetic.
//!
//! It deliberately does **not** do rectangle bin-packing (that is
//! [`super::atlas_packing`]'s `ShelfPacker`) nor flipbook sequence-frame
//! stepping (that is [`super::uv_animation`]); the grid here is a regular `N`x`N`
//! lattice of equal cells, not a packed set of arbitrary rectangles, and it is
//! addressed by view direction, not by an animation clock. Those modules are
//! neither imported nor re-derived.
//!
//! Everything is algebraic: the `octahedral` map, the cell selection, and the
//! neighbour weighting use only `abs`, division, comparison, `f32::floor`, and a
//! single `f32::sqrt` for direction normalization. No transcendental function
//! (`sin`/`cos`/`atan`/`exp`/`ln`/`powf`) is used anywhere, so the `CPU` contract
//! matches what a branch-light `GPU` shader would compute. Nothing panics or
//! divides by zero: a degenerate grid dimension is clamped up to one cell.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// The maximum number of baked views this contract blends together for a single
/// sample: the `octahedral` grid is a 2-D lattice, so the natural neighbourhood
/// of any direction is the surrounding 2x2 block of cells.
pub const MAX_BLEND_VIEWS: usize = 4;

/// Comparison epsilon for normalized `UV` and direction values in the tests: two
/// `f32` values closer than this are treated as equal so the contract never
/// relies on exact floating-point equality.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// Returns `1.0` for a non-negative input and `-1.0` otherwise.
///
/// This is the branch-free-friendly sign used by the `octahedral` fold; zero is
/// treated as positive so the seam of the map is handled deterministically.
fn sign_not_zero(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// The dot product of two 3-component direction vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalizes a 3-component vector, returning `+Z` for a zero-length input so
/// the result is always a valid unit direction and never a `NaN`.
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq > 0.0 {
        let inv = 1.0 / len_sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Encodes a direction onto the full-sphere `octahedral` square `[-1, 1]^2`.
///
/// The direction is projected onto the octahedron by dividing by its `L1` norm;
/// the lower hemisphere (`z < 0`) is folded out to the outer ring so the whole
/// sphere maps bijectively onto the square. This uses only `abs`, division, and
/// sign selection — no trigonometry.
#[must_use]
pub fn oct_encode(dir: [f32; 3]) -> [f32; 2] {
    let l1 = dir[0].abs() + dir[1].abs() + dir[2].abs();
    if l1 <= 0.0 {
        return [0.0, 0.0];
    }
    let inv = 1.0 / l1;
    let mut px = dir[0] * inv;
    let mut py = dir[1] * inv;
    let pz = dir[2] * inv;
    if pz < 0.0 {
        let folded_x = (1.0 - py.abs()) * sign_not_zero(px);
        let folded_y = (1.0 - px.abs()) * sign_not_zero(py);
        px = folded_x;
        py = folded_y;
    }
    [px, py]
}

/// Decodes an `octahedral` square coordinate `[-1, 1]^2` back to a unit
/// direction, the inverse of [`oct_encode`].
#[must_use]
pub fn oct_decode(oct: [f32; 2]) -> [f32; 3] {
    let mut x = oct[0];
    let mut y = oct[1];
    let z = 1.0 - x.abs() - y.abs();
    let t = (-z).max(0.0);
    x += if x >= 0.0 { -t } else { t };
    y += if y >= 0.0 { -t } else { t };
    normalize3([x, y, z])
}

/// The integer location of a baked view inside the `N`x`N` `impostor` grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CellCoord {
    /// The flattened cell index, `row * grid_dim + col`, in `[0, grid_dim^2)`.
    pub index: u32,
    /// The column of the cell in `[0, grid_dim)`.
    pub col: u32,
    /// The row of the cell in `[0, grid_dim)`.
    pub row: u32,
}

/// The result of sampling the `impostor` `atlas` for one view direction: the
/// nearest baked view cells, their normalized blend weights, and their `UV`
/// rectangles inside the `atlas`.
///
/// Only the first `count` entries of each array are meaningful; the remainder
/// are zero-filled padding so the type stays `Copy` and fixed-size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpostorSample {
    /// The selected cell indices, most-significant (nearest) first.
    pub cells: [u32; MAX_BLEND_VIEWS],
    /// The normalized blend weights aligned with `cells`; the first `count`
    /// entries sum to approximately `1.0`.
    pub weights: [f32; MAX_BLEND_VIEWS],
    /// The normalized `UV` rectangle `[u0, v0, u1, v1]` of each selected cell.
    pub uv_rects: [[f32; 4]; MAX_BLEND_VIEWS],
    /// How many leading entries of the arrays are valid, in `[1, MAX_BLEND_VIEWS]`.
    pub count: u32,
}

/// A regular `N`x`N` grid of baked `impostor` views packed into one `atlas`.
///
/// The grid maps `octahedral`-encoded directions to cells: column indexes the
/// encoded `x`, row indexes the encoded `y`. Cell geometry is pure grid
/// arithmetic and is independent of the texel dimensions; the `atlas` size is
/// carried only so the `std430` parameter block can hand the shader the texel
/// resolution it samples at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImpostorGrid {
    grid_dim: u32,
    atlas_width: u32,
    atlas_height: u32,
}

/// Clamps a floating grid coordinate to a valid integer cell axis in
/// `[0, dim - 1]`. Negative inputs floor to zero and the `f32`-to-`u32` cast
/// truncates toward zero, which equals `floor` for the non-negative value.
fn clamp_index(value: f32, dim: u32) -> u32 {
    let clamped = value.max(0.0);
    let idx = clamped as u32;
    idx.min(dim.saturating_sub(1))
}

/// Whether candidate `(w, idx)` should sort ahead of `(pw, pidx)`: greater
/// weight wins, ties break to the lower cell index. Uses only `>` comparisons
/// so it never relies on exact `f32` equality.
fn ranks_before(w: f32, idx: u32, pw: f32, pidx: u32) -> bool {
    if w > pw {
        true
    } else if pw > w {
        false
    } else {
        idx < pidx
    }
}

impl ImpostorGrid {
    /// Creates a grid of `grid_dim` x `grid_dim` baked views packed into an
    /// `atlas` of the given texel dimensions.
    ///
    /// `grid_dim` is clamped up to one so the grid always has at least a single
    /// cell and cell arithmetic never divides by zero.
    #[must_use]
    pub fn new(grid_dim: u32, atlas_width: u32, atlas_height: u32) -> Self {
        Self {
            grid_dim: grid_dim.max(1),
            atlas_width,
            atlas_height,
        }
    }

    /// The number of cells per axis.
    #[must_use]
    pub fn grid_dim(&self) -> u32 {
        self.grid_dim
    }

    /// The `atlas` width in texels.
    #[must_use]
    pub fn atlas_width(&self) -> u32 {
        self.atlas_width
    }

    /// The `atlas` height in texels.
    #[must_use]
    pub fn atlas_height(&self) -> u32 {
        self.atlas_height
    }

    /// The total number of baked view cells, `grid_dim^2`, saturating instead
    /// of overflowing.
    #[must_use]
    pub fn cell_count(&self) -> u32 {
        self.grid_dim.saturating_mul(self.grid_dim)
    }

    /// The `(col, row)` of a flattened cell index, clamped into range.
    #[must_use]
    pub fn cell_coord(&self, cell: u32) -> (u32, u32) {
        let clamped = cell.min(self.cell_count().saturating_sub(1));
        (clamped % self.grid_dim, clamped / self.grid_dim)
    }

    /// The flattened index of a `(col, row)` pair, each clamped into range.
    #[must_use]
    pub fn cell_index(&self, col: u32, row: u32) -> u32 {
        let max_axis = self.grid_dim.saturating_sub(1);
        let c = col.min(max_axis);
        let r = row.min(max_axis);
        r * self.grid_dim + c
    }

    /// The unit direction a cell was baked from: the direction whose
    /// `octahedral` encoding lands on the cell's center.
    fn cell_center_dir(&self, col: u32, row: u32) -> [f32; 3] {
        let g = self.grid_dim as f32;
        let s = (col as f32 + 0.5) / g;
        let t = (row as f32 + 0.5) / g;
        oct_decode([s * 2.0 - 1.0, t * 2.0 - 1.0])
    }

    /// Selects the single baked view cell nearest to `dir`.
    #[must_use]
    pub fn view_to_cell(&self, dir: [f32; 3]) -> CellCoord {
        let ndir = normalize3(dir);
        let oct = oct_encode(ndir);
        let s = oct[0] * 0.5 + 0.5;
        let t = oct[1] * 0.5 + 0.5;
        let g = self.grid_dim as f32;
        let col = clamp_index(s * g, self.grid_dim);
        let row = clamp_index(t * g, self.grid_dim);
        CellCoord {
            index: row * self.grid_dim + col,
            col,
            row,
        }
    }

    /// The normalized `UV` rectangle `[u0, v0, u1, v1]` of a cell inside the
    /// `atlas`, in `[0, 1]`. Adjacent cells share an edge but never overlap in
    /// area. This is pure grid arithmetic and does not depend on the texel size.
    #[must_use]
    pub fn cell_uv_rect(&self, cell: u32) -> [f32; 4] {
        let (col, row) = self.cell_coord(cell);
        let inv = 1.0 / self.grid_dim as f32;
        let u0 = col as f32 * inv;
        let v0 = row as f32 * inv;
        [u0, v0, u0 + inv, v0 + inv]
    }

    /// Selects the nearest `k` baked view cells to `dir` and returns them with
    /// normalized blend weights and their `UV` rectangles.
    ///
    /// `k` is clamped to `[1, MAX_BLEND_VIEWS]`. Candidate cells are gathered
    /// from the surrounding 2x2 `octahedral` block; each candidate is weighted by
    /// the clamped dot product between `dir` and the cell's baked direction (a
    /// transcendental-free proximity measure), the best `k` are kept, and their
    /// weights are renormalized to sum to approximately `1.0`. If every
    /// candidate faces away, the weights fall back to an equal split so the sum
    /// is still `1.0`.
    #[must_use]
    pub fn nearest_views(&self, dir: [f32; 3], k: usize) -> ImpostorSample {
        let ndir = normalize3(dir);
        let k = k.clamp(1, MAX_BLEND_VIEWS);
        let g = self.grid_dim as f32;

        let oct = oct_encode(ndir);
        let s = oct[0] * 0.5 + 0.5;
        let t = oct[1] * 0.5 + 0.5;
        let base_col = (s * g - 0.5).floor();
        let base_row = (t * g - 0.5).floor();

        let mut cand_idx = [0u32; MAX_BLEND_VIEWS];
        let mut cand_w = [0.0f32; MAX_BLEND_VIEWS];
        let mut n = 0usize;

        let offsets = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        for &(dc, dr) in &offsets {
            let col = clamp_index(base_col + dc, self.grid_dim);
            let row = clamp_index(base_row + dr, self.grid_dim);
            let idx = row * self.grid_dim + col;
            if cand_idx[..n].contains(&idx) {
                continue;
            }
            let center = self.cell_center_dir(col, row);
            cand_idx[n] = idx;
            cand_w[n] = dot3(ndir, center).max(0.0);
            n += 1;
        }

        // Insertion sort the small candidate set: nearest (largest weight)
        // first, ties broken toward the lower cell index for determinism.
        let mut i = 1usize;
        while i < n {
            let wi = cand_w[i];
            let xi = cand_idx[i];
            let mut j = i;
            while j > 0 && ranks_before(wi, xi, cand_w[j - 1], cand_idx[j - 1]) {
                cand_w[j] = cand_w[j - 1];
                cand_idx[j] = cand_idx[j - 1];
                j -= 1;
            }
            cand_w[j] = wi;
            cand_idx[j] = xi;
            i += 1;
        }

        let count = n.min(k);
        let mut sum = 0.0f32;
        let mut a = 0usize;
        while a < count {
            sum += cand_w[a];
            a += 1;
        }

        let mut cells = [0u32; MAX_BLEND_VIEWS];
        let mut weights = [0.0f32; MAX_BLEND_VIEWS];
        let mut uv_rects = [[0.0f32; 4]; MAX_BLEND_VIEWS];
        let equal = 1.0 / count as f32;
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let mut b = 0usize;
        while b < count {
            cells[b] = cand_idx[b];
            weights[b] = if sum > 0.0 {
                cand_w[b] * inv_sum
            } else {
                equal
            };
            uv_rects[b] = self.cell_uv_rect(cand_idx[b]);
            b += 1;
        }

        ImpostorSample {
            cells,
            weights,
            uv_rects,
            count: count as u32,
        }
    }

    /// Samples the `atlas` for `dir`, blending the full 2x2 neighbourhood
    /// (`MAX_BLEND_VIEWS` cells) — the common draw-time path.
    #[must_use]
    pub fn sample(&self, dir: [f32; 3]) -> ImpostorSample {
        self.nearest_views(dir, MAX_BLEND_VIEWS)
    }

    /// Packs the grid parameters into a `std430` `vec4<u32>`
    /// `[grid_dim, atlas_width, atlas_height, pad]` for a uniform/storage bind.
    ///
    /// The trailing element is explicit padding so the block is exactly one
    /// `vec4` and honours `std430`'s 16-byte alignment.
    #[must_use]
    pub fn to_std430(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(VEC4_STRIDE);
        bytes.extend_from_slice(&self.grid_dim.to_le_bytes());
        bytes.extend_from_slice(&self.atlas_width.to_le_bytes());
        bytes.extend_from_slice(&self.atlas_height.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes
    }

    /// The `std430` byte size of the grid parameter block: a single padded
    /// `vec4`, using the shared [`storage_bytes`] rule.
    #[must_use]
    pub fn std430_size() -> usize {
        storage_bytes(VEC4_STRIDE, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn dir_approx(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        approx(a[0], b[0], eps) && approx(a[1], b[1], eps) && approx(a[2], b[2], eps)
    }

    const AXES: [[f32; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];

    #[test]
    fn oct_roundtrip_main_axes_is_exact() {
        for &axis in &AXES {
            let decoded = oct_decode(oct_encode(axis));
            assert!(
                dir_approx(decoded, axis, CMP_EPS),
                "axis {axis:?} decoded to {decoded:?}",
            );
        }
    }

    #[test]
    fn oct_roundtrip_general_directions() {
        let raw = [
            [0.5, 0.5, 0.5],
            [-0.3, 0.7, -0.2],
            [0.1, -0.9, 0.4],
            [-0.6, -0.6, -0.6],
            [0.8, -0.1, 0.55],
        ];
        for &v in &raw {
            let unit = normalize3(v);
            let decoded = oct_decode(oct_encode(unit));
            assert!(
                dir_approx(decoded, unit, 1e-5),
                "dir {unit:?} decoded to {decoded:?}",
            );
        }
    }

    #[test]
    fn oct_encoded_coords_stay_in_square() {
        for &v in &AXES {
            let e = oct_encode(v);
            assert!((-1.0 - CMP_EPS..=1.0 + CMP_EPS).contains(&e[0]));
            assert!((-1.0 - CMP_EPS..=1.0 + CMP_EPS).contains(&e[1]));
        }
    }

    #[test]
    fn cell_index_stays_in_range() {
        let grid = ImpostorGrid::new(8, 4096, 4096);
        let count = grid.cell_count();
        let dirs = [
            [1.0, 0.2, 0.3],
            [-0.4, 1.0, -0.9],
            [0.7, -0.7, 0.1],
            [0.0, 0.0, -1.0],
            [-1.0, -1.0, -1.0],
            [0.05, 0.02, 0.99],
        ];
        for &d in &dirs {
            let cell = grid.view_to_cell(d);
            assert!(cell.index < count, "index {} >= {count}", cell.index);
            assert!(cell.col < grid.grid_dim());
            assert!(cell.row < grid.grid_dim());
            assert_eq!(cell.index, cell.row * grid.grid_dim() + cell.col);
        }
    }

    #[test]
    fn degenerate_grid_has_one_cell() {
        let grid = ImpostorGrid::new(0, 64, 64);
        assert_eq!(grid.grid_dim(), 1);
        assert_eq!(grid.cell_count(), 1);
        let rect = grid.cell_uv_rect(0);
        assert!(approx(rect[0], 0.0, CMP_EPS) && approx(rect[1], 0.0, CMP_EPS));
        assert!(approx(rect[2], 1.0, CMP_EPS) && approx(rect[3], 1.0, CMP_EPS));
    }

    #[test]
    fn uv_rects_are_normalized() {
        let grid = ImpostorGrid::new(5, 1024, 1024);
        let count = grid.cell_count();
        let mut cell = 0u32;
        while cell < count {
            let r = grid.cell_uv_rect(cell);
            for &c in &r {
                assert!(
                    (-CMP_EPS..=1.0 + CMP_EPS).contains(&c),
                    "uv {c} out of range"
                );
            }
            assert!(r[2] > r[0] && r[3] > r[1], "rect {r:?} not min<max");
            cell += 1;
        }
    }

    fn rects_overlap(a: [f32; 4], b: [f32; 4]) -> bool {
        // Positive-area intersection on both axes means a genuine overlap;
        // touching edges (zero-area) do not count.
        let x = a[0] < b[2] - CMP_EPS && b[0] < a[2] - CMP_EPS;
        let y = a[1] < b[3] - CMP_EPS && b[1] < a[3] - CMP_EPS;
        x && y
    }

    #[test]
    fn adjacent_uv_rects_touch_but_do_not_overlap() {
        let grid = ImpostorGrid::new(4, 512, 512);
        let count = grid.cell_count();
        // No two distinct cells overlap in area.
        let mut i = 0u32;
        while i < count {
            let mut j = i + 1;
            while j < count {
                assert!(
                    !rects_overlap(grid.cell_uv_rect(i), grid.cell_uv_rect(j)),
                    "cells {i} and {j} overlap",
                );
                j += 1;
            }
            i += 1;
        }
        // Horizontally adjacent cells share the seam u1(left) == u0(right).
        let left = grid.cell_uv_rect(grid.cell_index(1, 2));
        let right = grid.cell_uv_rect(grid.cell_index(2, 2));
        assert!(approx(left[2], right[0], CMP_EPS));
    }

    #[test]
    fn nearest_view_weights_sum_to_one() {
        let grid = ImpostorGrid::new(6, 2048, 2048);
        let dirs = [
            [0.3, 0.4, 0.85],
            [-0.9, 0.1, 0.4],
            [0.2, -0.95, 0.2],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ];
        for &d in &dirs {
            let sample = grid.sample(d);
            assert!(sample.count >= 1 && sample.count as usize <= MAX_BLEND_VIEWS);
            let mut sum = 0.0f32;
            let mut a = 0usize;
            while a < sample.count as usize {
                assert!(sample.weights[a] >= -CMP_EPS, "negative weight");
                sum += sample.weights[a];
                a += 1;
            }
            assert!(approx(sum, 1.0, 1e-5), "weights sum {sum} for dir {d:?}");
        }
    }

    #[test]
    fn nearest_views_respects_k() {
        let grid = ImpostorGrid::new(8, 1024, 1024);
        let dir = [0.4, 0.35, 0.8];
        assert_eq!(grid.nearest_views(dir, 1).count, 1);
        let two = grid.nearest_views(dir, 2).count;
        assert!((1..=2).contains(&two));
        let capped = grid.nearest_views(dir, 999);
        assert!(capped.count as usize <= MAX_BLEND_VIEWS);
    }

    #[test]
    fn nearest_views_are_ordered_by_weight() {
        let grid = ImpostorGrid::new(7, 1024, 1024);
        let sample = grid.sample([0.25, 0.4, 0.88]);
        let mut a = 1usize;
        while a < sample.count as usize {
            assert!(
                sample.weights[a - 1] >= sample.weights[a] - CMP_EPS,
                "weights not descending",
            );
            a += 1;
        }
    }

    #[test]
    fn sample_is_deterministic() {
        let grid = ImpostorGrid::new(5, 256, 256);
        let dir = [0.6, -0.2, 0.7];
        assert_eq!(grid.sample(dir), grid.sample(dir));
        assert_eq!(grid.view_to_cell(dir), grid.view_to_cell(dir));
    }

    #[test]
    fn std430_block_is_one_padded_vec4() {
        let grid = ImpostorGrid::new(12, 8192, 4096);
        let bytes = grid.to_std430();
        assert_eq!(bytes.len(), ImpostorGrid::std430_size());
        assert_eq!(bytes.len(), VEC4_STRIDE);
        assert_eq!(bytes.len() % VEC4_STRIDE, 0);

        let dim = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let w = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let h = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let pad = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(dim, 12);
        assert_eq!(w, 8192);
        assert_eq!(h, 4096);
        assert_eq!(pad, 0);
    }
}
