//! Regular-grid height-field collider geometry.
//!
//! A height field is the standard terrain-collision primitive used by AAA
//! engines (UE `UHeightField` / `PhysX` `PxHeightField` / `Jolt` `HeightFieldShape`).
//! It stores a dense `rows x cols` grid of sampled heights and reconstructs the
//! implied surface as two triangles per grid cell, which is dramatically
//! cheaper to store and query than an equivalent triangle soup.
//!
//! This module provides the *geometry* only - the data layout plus the
//! closed-form queries the broad and narrow phases need:
//!
//! - [`HeightField::local_aabb`] for broad-phase bounds,
//! - [`HeightField::sample_height`] for an analytic surface height at a point,
//! - [`HeightField::cell_triangles`] / [`HeightField::overlapping_triangles`]
//!   to hand candidate triangles to the narrow phase, and
//! - [`HeightField::ray_cast`] for scene queries, using an Amanatides and Woo
//!   grid traversal so cost scales with the number of cells the ray crosses
//!   rather than the whole field.
//!
//! # Coordinate convention
//!
//! The field lies in the local `XZ` plane with height measured along `+Y`.
//! Sample `(row, col)` sits at local position
//! `(col * scale.x, height[row][col] * scale.y, row * scale.z)`, so `col`
//! indexes the `X` axis, `row` indexes the `Z` axis, and the field's corner
//! sample `(0, 0)` is at the local origin's `XZ`. Heights are stored row-major
//! (`row * cols + col`).
//!
//! Every quad cell is split along its `(row, col) -> (row + 1, col + 1)`
//! diagonal into two triangles wound so their geometric normal points towards
//! `+Y`. The height reconstructed by [`sample_height`](HeightField::sample_height)
//! is exactly the planar interpolation over whichever of those two triangles
//! contains the query point, so sampling and triangle collision agree.
//!
//! These are standard regular-grid formulas and a standard voxel-traversal ray
//! cast; nothing here is derived from Unreal Engine source.

use glam::Vec3;

/// Minimum number of samples along each axis (one cell needs a 2x2 block).
const MIN_SAMPLES: usize = 2;

/// Immutable regular-grid height-field collider.
///
/// Build one with [`HeightField::new`]. The grid is `rows x cols` samples; the
/// surface spans `(cols - 1) x (rows - 1)` quad cells.
#[derive(Clone, PartialEq, Debug)]
pub struct HeightField {
    /// Number of samples along `Z` (grid rows).
    rows: usize,
    /// Number of samples along `X` (grid columns).
    cols: usize,
    /// Per-axis scale: cell spacing in `X`/`Z` and the height multiplier in `Y`.
    scale: Vec3,
    /// Row-major sampled heights (`rows * cols` entries).
    heights: Vec<f32>,
    /// Cached minimum and maximum raw height samples (pre-`scale.y`).
    min_height: f32,
    /// Cached maximum raw height sample (pre-`scale.y`).
    max_height: f32,
}

/// A ray hit against a [`HeightField`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct HeightFieldRayHit {
    /// Ray parameter `t` at the hit (`point == origin + dir * t`).
    pub time: f32,
    /// Hit position in the field's local frame.
    pub point: Vec3,
    /// Outward (towards `+Y`) unit surface normal of the struck triangle.
    pub normal: Vec3,
    /// Grid cell `(row, col)` whose triangle was hit.
    pub cell: (usize, usize),
}

impl HeightField {
    /// Builds a height field from a row-major height grid.
    ///
    /// `rows` and `cols` are the sample counts along `Z` and `X`; `heights`
    /// must contain exactly `rows * cols` entries. `scale` gives the cell
    /// spacing (`scale.x`, `scale.z`) and the height multiplier (`scale.y`).
    ///
    /// Returns [`None`] when the grid is smaller than `2x2`, when the height
    /// buffer length does not match `rows * cols`, when any sample is
    /// non-finite, or when a lateral scale component is non-positive.
    #[must_use]
    pub fn new(rows: usize, cols: usize, scale: Vec3, heights: Vec<f32>) -> Option<HeightField> {
        if rows < MIN_SAMPLES || cols < MIN_SAMPLES {
            return None;
        }
        if heights.len() != rows * cols {
            return None;
        }
        if scale.x <= 0.0 || scale.z <= 0.0 || !scale.is_finite() {
            return None;
        }
        let mut min_height = f32::INFINITY;
        let mut max_height = f32::NEG_INFINITY;
        for &h in &heights {
            if !h.is_finite() {
                return None;
            }
            min_height = min_height.min(h);
            max_height = max_height.max(h);
        }
        Some(HeightField {
            rows,
            cols,
            scale,
            heights,
            min_height,
            max_height,
        })
    }

    /// Returns the grid dimensions as `(rows, cols)` sample counts.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the per-axis scale (`x`/`z` spacing, `y` height multiplier).
    #[must_use]
    pub fn scale(&self) -> Vec3 {
        self.scale
    }

    /// Returns the raw (pre-`scale.y`) height sample at `(row, col)`, or
    /// [`None`] when the indices are out of range.
    #[must_use]
    pub fn raw_sample(&self, row: usize, col: usize) -> Option<f32> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        Some(self.heights[row * self.cols + col])
    }

    /// Returns the local-space position of grid sample `(row, col)`, or
    /// [`None`] when the indices are out of range.
    #[must_use]
    pub fn vertex(&self, row: usize, col: usize) -> Option<Vec3> {
        let h = self.raw_sample(row, col)?;
        Some(Vec3::new(
            col as f32 * self.scale.x,
            h * self.scale.y,
            row as f32 * self.scale.z,
        ))
    }

    /// Returns the local-space axis-aligned bounding box as `(min, max)`.
    #[must_use]
    pub fn local_aabb(&self) -> (Vec3, Vec3) {
        let max_x = (self.cols - 1) as f32 * self.scale.x;
        let max_z = (self.rows - 1) as f32 * self.scale.z;
        let min_y = self.min_height * self.scale.y;
        let max_y = self.max_height * self.scale.y;
        // `scale.y` may be negative; order the Y bounds defensively.
        let (lo_y, hi_y) = if min_y <= max_y {
            (min_y, max_y)
        } else {
            (max_y, min_y)
        };
        (Vec3::new(0.0, lo_y, 0.0), Vec3::new(max_x, hi_y, max_z))
    }

    /// Returns the two triangles of grid cell `(row, col)`, or [`None`] when
    /// the cell is out of range (valid cells are `row < rows - 1` and
    /// `col < cols - 1`).
    ///
    /// Both triangles are wound counter-clockwise seen from `+Y` so their
    /// `edge1 x edge2` normal points upward.
    #[must_use]
    pub fn cell_triangles(&self, row: usize, col: usize) -> Option<[[Vec3; 3]; 2]> {
        if row + 1 >= self.rows || col + 1 >= self.cols {
            return None;
        }
        let v00 = self.vertex(row, col)?;
        let v01 = self.vertex(row, col + 1)?;
        let v10 = self.vertex(row + 1, col)?;
        let v11 = self.vertex(row + 1, col + 1)?;
        // Diagonal v00 -> v11. The `fx >= fz` triangle carries corner v01, the
        // `fx <= fz` triangle carries v10; both are wound so `edge1 x edge2`
        // points towards +Y.
        Some([[v00, v11, v01], [v00, v10, v11]])
    }

    /// Appends every cell triangle whose `XZ` footprint overlaps the lateral
    /// box `[min.x, max.x] x [min.z, max.z]` to `out`.
    ///
    /// Only the `X` and `Z` components of the bounds are used; the caller is
    /// expected to reject triangles outside the vertical range separately. This
    /// is the mid-phase bridge that hands terrain triangles to the convex /
    /// primitive narrow phase.
    pub fn overlapping_triangles(&self, min: Vec3, max: Vec3, out: &mut Vec<[Vec3; 3]>) {
        let (lo_x, hi_x) = (min.x.min(max.x), min.x.max(max.x));
        let (lo_z, hi_z) = (min.z.min(max.z), min.z.max(max.z));
        if hi_x < 0.0 || hi_z < 0.0 {
            return;
        }
        // Convert the lateral box to inclusive cell index ranges.
        let col_lo = (lo_x / self.scale.x).floor().max(0.0) as usize;
        let row_lo = (lo_z / self.scale.z).floor().max(0.0) as usize;
        let col_hi =
            ((hi_x / self.scale.x).floor() as isize).clamp(0, self.cols as isize - 2) as usize;
        let row_hi =
            ((hi_z / self.scale.z).floor() as isize).clamp(0, self.rows as isize - 2) as usize;
        if col_lo > col_hi || row_lo > row_hi {
            return;
        }
        for row in row_lo..=row_hi {
            for col in col_lo..=col_hi {
                if let Some([t0, t1]) = self.cell_triangles(row, col) {
                    out.push(t0);
                    out.push(t1);
                }
            }
        }
    }

    /// Returns the analytic surface height at local `(x, z)`, matching the
    /// planar interpolation of whichever cell triangle contains the point.
    ///
    /// Returns [`None`] when `(x, z)` lies outside the field footprint.
    #[must_use]
    pub fn sample_height(&self, x: f32, z: f32) -> Option<f32> {
        let max_x = (self.cols - 1) as f32 * self.scale.x;
        let max_z = (self.rows - 1) as f32 * self.scale.z;
        if x < 0.0 || z < 0.0 || x > max_x || z > max_z {
            return None;
        }
        // Locate the cell and the fractional position inside it.
        let gx = x / self.scale.x;
        let gz = z / self.scale.z;
        let col = (gx.floor() as usize).min(self.cols - 2);
        let row = (gz.floor() as usize).min(self.rows - 2);
        let fx = gx - col as f32;
        let fz = gz - row as f32;

        let h00 = self.heights[row * self.cols + col];
        let h01 = self.heights[row * self.cols + col + 1];
        let h10 = self.heights[(row + 1) * self.cols + col];
        let h11 = self.heights[(row + 1) * self.cols + col + 1];

        // Diagonal fz == fx splits the cell; interpolate within the matching
        // triangle so the result equals the collision surface exactly.
        let raw = if fx >= fz {
            // Lower triangle (v00, v01, v11): h00 + (h01-h00)*fx + (h11-h01)*fz.
            h00 + (h01 - h00) * fx + (h11 - h01) * fz
        } else {
            // Upper triangle (v00, v11, v10): h00 + (h11-h10)*fx + (h10-h00)*fz.
            h00 + (h11 - h10) * fx + (h10 - h00) * fz
        };
        Some(raw * self.scale.y)
    }

    /// Casts a ray `origin + dir * t` for `t in [0, max_time]` against the
    /// field, returning the first triangle hit or [`None`] on a miss.
    ///
    /// The lateral `XZ` projection of the ray is marched cell-by-cell with an
    /// Amanatides and Woo traversal; each visited cell's two triangles are
    /// tested with Moller-Trumbore and the earliest valid hit is returned.
    #[must_use]
    pub fn ray_cast(&self, origin: Vec3, dir: Vec3, max_time: f32) -> Option<HeightFieldRayHit> {
        if max_time <= 0.0 || dir.length_squared() <= 0.0 {
            return None;
        }
        let max_x = (self.cols - 1) as f32 * self.scale.x;
        let max_z = (self.rows - 1) as f32 * self.scale.z;

        // Clip the ray's parameter interval to the lateral field slab so the
        // traversal starts at the first in-bounds cell.
        let (mut t_enter, mut t_exit) = (0.0_f32, max_time);
        if !clip_slab(origin.x, dir.x, 0.0, max_x, &mut t_enter, &mut t_exit) {
            return None;
        }
        if !clip_slab(origin.z, dir.z, 0.0, max_z, &mut t_enter, &mut t_exit) {
            return None;
        }
        if t_enter > t_exit {
            return None;
        }

        // Entry point in grid-cell coordinates.
        let entry = origin + dir * t_enter;
        let mut col = ((entry.x / self.scale.x).floor() as isize).clamp(0, self.cols as isize - 2);
        let mut row = ((entry.z / self.scale.z).floor() as isize).clamp(0, self.rows as isize - 2);

        let step_col: isize = if dir.x > 0.0 { 1 } else { -1 };
        let step_row: isize = if dir.z > 0.0 { 1 } else { -1 };

        // Parametric distance to the next cell boundary along each axis, and
        // the parametric cell width.
        let (mut t_max_x, t_delta_x) = axis_traversal(entry.x, dir.x, self.scale.x, col, step_col);
        let (mut t_max_z, t_delta_z) = axis_traversal(entry.z, dir.z, self.scale.z, row, step_row);
        // `t_max_*` are measured from the entry point; shift to global `t`.
        t_max_x += t_enter;
        t_max_z += t_enter;

        loop {
            if let Some(triangles) = self.cell_triangles(row as usize, col as usize) {
                let mut best: Option<HeightFieldRayHit> = None;
                for tri in triangles {
                    if let Some((t, n)) = ray_triangle(origin, dir, tri, max_time)
                        && t >= t_enter - 1e-5
                        && best.is_none_or(|h| t < h.time)
                    {
                        best = Some(HeightFieldRayHit {
                            time: t,
                            point: origin + dir * t,
                            normal: if n.y >= 0.0 { n } else { -n },
                            cell: (row as usize, col as usize),
                        });
                    }
                }
                if let Some(hit) = best {
                    // A hit before the exit of the current cell is final; the
                    // marched order guarantees no earlier cell can beat it.
                    let cell_exit = t_max_x.min(t_max_z);
                    if hit.time <= cell_exit + 1e-5 {
                        return Some(hit);
                    }
                }
            }

            // Advance to the next cell along the smaller boundary distance.
            if t_max_x <= t_max_z {
                if t_max_x > t_exit {
                    return None;
                }
                col += step_col;
                t_max_x += t_delta_x;
            } else {
                if t_max_z > t_exit {
                    return None;
                }
                row += step_row;
                t_max_z += t_delta_z;
            }
            if col < 0 || row < 0 || col >= self.cols as isize - 1 || row >= self.rows as isize - 1
            {
                return None;
            }
        }
    }
}

/// Clips the parameter interval `[*t_enter, *t_exit]` of `o + d * t` to the
/// slab `lo <= o + d*t <= hi`. Returns `false` when the ray misses the slab.
fn clip_slab(o: f32, d: f32, lo: f32, hi: f32, t_enter: &mut f32, t_exit: &mut f32) -> bool {
    if d.abs() <= 1e-12 {
        // Parallel to the slab: in-bounds only if the origin already lies in it.
        return o >= lo && o <= hi;
    }
    let inv = 1.0 / d;
    let mut t0 = (lo - o) * inv;
    let mut t1 = (hi - o) * inv;
    if t0 > t1 {
        core::mem::swap(&mut t0, &mut t1);
    }
    *t_enter = t_enter.max(t0);
    *t_exit = t_exit.min(t1);
    *t_enter <= *t_exit
}

/// Computes the initial `t_max` (distance from `pos` to the first cell boundary
/// along this axis) and the per-cell `t_delta` for a voxel traversal.
///
/// `pos` is the entry coordinate, `d` the ray direction component, `cell_size`
/// the lateral spacing, `cell` the starting integer cell, and `step` the march
/// direction (`+1`/`-1`). A near-zero direction yields an infinite `t_max` so
/// that axis never triggers a step.
fn axis_traversal(pos: f32, d: f32, cell_size: f32, cell: isize, step: isize) -> (f32, f32) {
    if d.abs() <= 1e-12 {
        return (f32::INFINITY, f32::INFINITY);
    }
    let inv = 1.0 / d;
    let next_boundary = if step > 0 {
        (cell + 1) as f32 * cell_size
    } else {
        cell as f32 * cell_size
    };
    let t_max = (next_boundary - pos) * inv;
    let t_delta = (cell_size * inv).abs();
    (t_max.max(0.0), t_delta)
}

/// Moller-Trumbore ray/triangle intersection. Returns `(t, unit_normal)` for a
/// front- or back-face hit with `0 <= t <= max_time`, or [`None`] on a miss.
fn ray_triangle(origin: Vec3, dir: Vec3, tri: [Vec3; 3], max_time: f32) -> Option<(f32, Vec3)> {
    let edge1 = tri[1] - tri[0];
    let edge2 = tri[2] - tri[0];
    let pvec = dir.cross(edge2);
    let det = edge1.dot(pvec);
    if det.abs() <= 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = origin - tri[0];
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(edge1);
    let v = dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(qvec) * inv_det;
    if t < 0.0 || t > max_time {
        return None;
    }
    let normal = edge1.cross(edge2).normalize_or_zero();
    Some((t, normal))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat `3x3` field at height `h`, unit spacing.
    fn flat(h: f32) -> HeightField {
        HeightField::new(3, 3, Vec3::ONE, vec![h; 9]).expect("valid flat field")
    }

    #[test]
    fn rejects_degenerate_input() {
        assert!(HeightField::new(1, 4, Vec3::ONE, vec![0.0; 4]).is_none());
        assert!(HeightField::new(2, 2, Vec3::ONE, vec![0.0; 3]).is_none());
        assert!(HeightField::new(2, 2, Vec3::new(0.0, 1.0, 1.0), vec![0.0; 4]).is_none());
        assert!(HeightField::new(2, 2, Vec3::ONE, vec![f32::NAN, 0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn local_aabb_spans_the_grid() {
        let hf = HeightField::new(
            2,
            3,
            Vec3::new(2.0, 1.0, 4.0),
            vec![0.0, 1.0, -2.0, 3.0, 0.5, 1.0],
        )
        .expect("ok");
        let (min, max) = hf.local_aabb();
        assert_eq!(min, Vec3::new(0.0, -2.0, 0.0));
        assert_eq!(max, Vec3::new(4.0, 3.0, 4.0));
    }

    #[test]
    fn sample_height_is_planar_on_a_ramp() {
        // Height increases linearly with x: h(col) = col.
        let hf = HeightField::new(2, 3, Vec3::ONE, vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]).expect("ok");
        assert!((hf.sample_height(0.0, 0.0).unwrap() - 0.0).abs() < 1e-6);
        assert!((hf.sample_height(1.5, 0.3).unwrap() - 1.5).abs() < 1e-6);
        assert!((hf.sample_height(2.0, 1.0).unwrap() - 2.0).abs() < 1e-6);
        assert!(hf.sample_height(-0.1, 0.0).is_none());
        assert!(hf.sample_height(0.0, 2.0).is_none());
    }

    #[test]
    fn sample_matches_triangle_plane_on_both_halves() {
        // A single tilted cell: distinct corner heights so the two triangles
        // lie on different planes.
        let hf = HeightField::new(2, 2, Vec3::ONE, vec![0.0, 1.0, 2.0, 5.0]).expect("ok");
        // Lower triangle (fx >= fz): corners v00(0), v01(1), v11(5).
        let p = (0.8_f32, 0.2_f32);
        let expect = 0.0 + (1.0 - 0.0) * p.0 + (5.0 - 1.0) * p.1;
        assert!((hf.sample_height(p.0, p.1).unwrap() - expect).abs() < 1e-6);
        // Upper triangle (fx < fz): corners v00(0), v11(5), v10(2).
        let q = (0.2_f32, 0.8_f32);
        let expect_q = 0.0 + (5.0 - 2.0) * q.0 + (2.0 - 0.0) * q.1;
        assert!((hf.sample_height(q.0, q.1).unwrap() - expect_q).abs() < 1e-6);
    }

    #[test]
    fn cell_triangles_wind_upward() {
        let hf = flat(2.0);
        let [t0, t1] = hf.cell_triangles(0, 0).expect("cell exists");
        for tri in [t0, t1] {
            let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
            assert!(n.y > 0.0, "triangle normal should point towards +Y");
        }
        assert!(hf.cell_triangles(2, 0).is_none());
    }

    #[test]
    fn overlapping_triangles_selects_cell_range() {
        let hf = flat(0.0); // 3x3 samples => 2x2 cells.
        let mut out = Vec::new();
        hf.overlapping_triangles(
            Vec3::new(0.1, -1.0, 0.1),
            Vec3::new(0.9, 1.0, 0.9),
            &mut out,
        );
        assert_eq!(
            out.len(),
            2,
            "a sub-cell box hits exactly one cell (2 tris)"
        );
        out.clear();
        hf.overlapping_triangles(
            Vec3::new(-5.0, -1.0, -5.0),
            Vec3::new(5.0, 1.0, 5.0),
            &mut out,
        );
        assert_eq!(out.len(), 8, "the whole box hits all four cells");
    }

    #[test]
    fn ray_cast_hits_flat_field_from_above() {
        let hf = flat(0.0);
        let hit = hf
            .ray_cast(Vec3::new(1.0, 5.0, 1.0), Vec3::new(0.0, -1.0, 0.0), 100.0)
            .expect("downward ray hits the ground");
        assert!((hit.time - 5.0).abs() < 1e-4);
        assert!((hit.point - Vec3::new(1.0, 0.0, 1.0)).length() < 1e-4);
        assert!(hit.normal.y > 0.9);
    }

    #[test]
    fn ray_cast_hits_a_slope_along_the_march() {
        // Ramp rising in +x. A ray travelling in +x just above the surface
        // should strike the rising terrain.
        let hf = HeightField::new(
            2,
            5,
            Vec3::ONE,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        )
        .expect("ok");
        let hit = hf
            .ray_cast(Vec3::new(0.0, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0), 10.0)
            .expect("ray should hit the rising ramp");
        assert!(hit.point.x > 0.0 && hit.point.x <= 4.0);
        assert!((hit.point.y - 0.5).abs() < 1e-3);
    }

    #[test]
    fn ray_cast_misses_when_above_and_parallel() {
        let hf = flat(0.0);
        assert!(hf
            .ray_cast(Vec3::new(1.0, 5.0, 1.0), Vec3::new(1.0, 0.0, 0.0), 100.0)
            .is_none());
        // Ray pointing away from the field laterally.
        assert!(hf
            .ray_cast(Vec3::new(-1.0, 5.0, 1.0), Vec3::new(-1.0, -1.0, 0.0), 100.0)
            .is_none());
    }
}
