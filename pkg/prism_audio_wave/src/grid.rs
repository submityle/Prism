//! Probe grid: the world-space lattice of sampling positions and the
//! trilinear interpolation primitives used both when baking (where each probe
//! is solved) and at runtime (where a listener position is blended from its
//! eight surrounding probes).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the "Probe Grid" of design section 43: an axis-aligned bounding
//! box subdivided into a regular lattice. The same lattice indexes the baked
//! parameter field ([`crate::field`]) and drives the runtime lookup
//! ([`crate::lookup`]). Coordinates use the Bevy right-handed convention
//! shared with [`prism_audio_spatial`].

use bevy_math::{ops, Vec3};

/// An axis-aligned bounding box in world space.
///
/// Construction normalises the corners so `min` is componentwise the lesser
/// corner and `max` the greater, so callers may pass either diagonal.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb {
    /// Componentwise minimum corner.
    pub min: Vec3,
    /// Componentwise maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// Builds a box from two opposite corners, normalising so `min <= max`
    /// componentwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::Vec3;
    /// # use prism_audio_wave::grid::Aabb;
    /// let b = Aabb::new(Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO);
    /// assert_eq!(b.min, Vec3::ZERO);
    /// assert_eq!(b.max, Vec3::new(1.0, 2.0, 3.0));
    /// ```
    #[must_use]
    pub fn new(a: Vec3, b: Vec3) -> Self {
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// The box extent (`max - min`), always componentwise non-negative.
    #[must_use]
    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// The geometric centre of the box.
    #[must_use]
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Returns `true` if `point` lies within the closed box.
    #[must_use]
    #[inline]
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.y >= self.min.y
            && point.z >= self.min.z
            && point.x <= self.max.x
            && point.y <= self.max.y
            && point.z <= self.max.z
    }
}

/// A regular lattice of probe positions filling an [`Aabb`].
///
/// The grid stores a per-axis probe count. Probes are addressed either by
/// `(ix, iy, iz)` triplets or by a single linear index with `x` fastest, then
/// `y`, then `z`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ProbeGrid {
    bounds: Aabb,
    counts: [u32; 3],
}

impl ProbeGrid {
    /// Builds a grid over `bounds` with `nx * ny * nz` probes. Each count is
    /// clamped up to at least `1` so the grid is never empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::Vec3;
    /// # use prism_audio_wave::grid::{Aabb, ProbeGrid};
    /// let grid = ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(4.0)), 3, 3, 3);
    /// assert_eq!(grid.probe_count(), 27);
    /// ```
    #[must_use]
    pub fn new(bounds: Aabb, nx: u32, ny: u32, nz: u32) -> Self {
        Self {
            bounds,
            counts: [nx.max(1), ny.max(1), nz.max(1)],
        }
    }

    /// The bounding box the grid fills.
    #[must_use]
    #[inline]
    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    /// The per-axis probe counts `[nx, ny, nz]`.
    #[must_use]
    #[inline]
    pub fn counts(&self) -> [u32; 3] {
        self.counts
    }

    /// Total number of probes (`nx * ny * nz`).
    #[must_use]
    #[inline]
    pub fn probe_count(&self) -> usize {
        self.counts[0] as usize * self.counts[1] as usize * self.counts[2] as usize
    }

    /// Linearises a probe triplet into its storage index (`x` fastest).
    ///
    /// Out-of-range components are clamped to the last probe on their axis.
    #[must_use]
    #[inline]
    pub fn linear_index(&self, ix: u32, iy: u32, iz: u32) -> usize {
        let x = ix.min(self.counts[0] - 1) as usize;
        let y = iy.min(self.counts[1] - 1) as usize;
        let z = iz.min(self.counts[2] - 1) as usize;
        let nx = self.counts[0] as usize;
        let ny = self.counts[1] as usize;
        (z * ny + y) * nx + x
    }

    /// Inverse of [`linear_index`](Self::linear_index): recovers the triplet.
    ///
    /// Indices beyond the grid are clamped to the final probe.
    #[must_use]
    #[inline]
    pub fn triplet(&self, index: usize) -> [u32; 3] {
        let nx = self.counts[0] as usize;
        let ny = self.counts[1] as usize;
        let count = self.probe_count();
        let i = index.min(count - 1);
        let x = i % nx;
        let y = (i / nx) % ny;
        let z = i / (nx * ny);
        [x as u32, y as u32, z as u32]
    }

    /// World-space position of probe `(ix, iy, iz)`.
    ///
    /// Along an axis with a single probe the position is the box centre on
    /// that axis; otherwise probes are spread evenly from `min` to `max`.
    #[must_use]
    pub fn probe_position(&self, ix: u32, iy: u32, iz: u32) -> Vec3 {
        let t = Vec3::new(
            axis_fraction(ix, self.counts[0]),
            axis_fraction(iy, self.counts[1]),
            axis_fraction(iz, self.counts[2]),
        );
        self.bounds.min + self.bounds.size() * t
    }

    /// Computes the trilinear blend of the eight probes surrounding
    /// `world_pos`.
    ///
    /// Positions outside the grid clamp to the boundary, so the returned
    /// weights always describe real probes and always sum to `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::Vec3;
    /// # use prism_audio_wave::grid::{Aabb, ProbeGrid};
    /// let grid = ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(2.0)), 3, 3, 3);
    /// let s = grid.trilinear(Vec3::splat(1.0));
    /// let w: f32 = s.corner_weights(&grid).iter().map(|(_, w)| *w).sum();
    /// assert!((w - 1.0).abs() < 1e-5);
    /// ```
    #[must_use]
    pub fn trilinear(&self, world_pos: Vec3) -> TrilinearSample {
        let size = self.bounds.size();
        let local = world_pos - self.bounds.min;
        let base_x = axis_cell(local.x, size.x, self.counts[0]);
        let base_y = axis_cell(local.y, size.y, self.counts[1]);
        let base_z = axis_cell(local.z, size.z, self.counts[2]);
        TrilinearSample {
            base: [base_x.0, base_y.0, base_z.0],
            frac: [base_x.1, base_y.1, base_z.1],
        }
    }
}

/// Fractional axis position of probe `i` in `[0, 1]`.
fn axis_fraction(i: u32, count: u32) -> f32 {
    if count <= 1 {
        0.5
    } else {
        (i.min(count - 1) as f32) / ((count - 1) as f32)
    }
}

/// Resolves a local coordinate into `(base_index, fraction)` for trilinear
/// blending, clamping to the valid cell range.
fn axis_cell(local: f32, extent: f32, count: u32) -> (u32, f32) {
    if count <= 1 || extent <= 0.0 {
        return (0, 0.0);
    }
    let last = count - 1;
    // Continuous probe coordinate in [0, last].
    let coord = (local / extent * last as f32).clamp(0.0, last as f32);
    let base = ops::floor(coord).min((last - 1) as f32);
    let frac = (coord - base).clamp(0.0, 1.0);
    (base as u32, frac)
}

/// The result of a [`ProbeGrid::trilinear`] query: the lower corner probe
/// triplet and the per-axis interpolation fractions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrilinearSample {
    /// Lower-corner probe triplet of the containing cell.
    pub base: [u32; 3],
    /// Per-axis interpolation fractions in `[0, 1]`.
    pub frac: [f32; 3],
}

impl TrilinearSample {
    /// Expands the sample into the eight `(linear_index, weight)` corner pairs,
    /// written onto the stack (no allocation).
    ///
    /// The weights are the standard trilinear products and sum to `1`.
    #[must_use]
    pub fn corner_weights(&self, grid: &ProbeGrid) -> [(usize, f32); 8] {
        let [fx, fy, fz] = self.frac;
        let gx = 1.0 - fx;
        let gy = 1.0 - fy;
        let gz = 1.0 - fz;
        let [bx, by, bz] = self.base;
        let x1 = (bx + 1).min(grid.counts()[0] - 1);
        let y1 = (by + 1).min(grid.counts()[1] - 1);
        let z1 = (bz + 1).min(grid.counts()[2] - 1);
        [
            (grid.linear_index(bx, by, bz), gx * gy * gz),
            (grid.linear_index(x1, by, bz), fx * gy * gz),
            (grid.linear_index(bx, y1, bz), gx * fy * gz),
            (grid.linear_index(x1, y1, bz), fx * fy * gz),
            (grid.linear_index(bx, by, z1), gx * gy * fz),
            (grid.linear_index(x1, by, z1), fx * gy * fz),
            (grid.linear_index(bx, y1, z1), gx * fy * fz),
            (grid.linear_index(x1, y1, z1), fx * fy * fz),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn aabb_normalises_corners() {
        let b = Aabb::new(Vec3::new(2.0, 0.0, 5.0), Vec3::new(-1.0, 3.0, 1.0));
        assert_eq!(b.min, Vec3::new(-1.0, 0.0, 1.0));
        assert_eq!(b.max, Vec3::new(2.0, 3.0, 5.0));
    }

    #[test]
    fn probe_positions_span_the_box() {
        let grid = ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(4.0)), 3, 3, 3);
        assert_eq!(grid.probe_position(0, 0, 0), Vec3::ZERO);
        assert_eq!(grid.probe_position(2, 2, 2), Vec3::splat(4.0));
        assert_eq!(grid.probe_position(1, 1, 1), Vec3::splat(2.0));
    }

    #[test]
    fn index_round_trips() {
        let grid = ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(1.0)), 4, 3, 2);
        for i in 0..grid.probe_count() {
            let [x, y, z] = grid.triplet(i);
            assert_eq!(grid.linear_index(x, y, z), i);
        }
    }

    #[test]
    fn trilinear_weights_sum_to_one() {
        let grid = ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(2.0)), 3, 3, 3);
        for p in [
            Vec3::new(0.3, 1.7, 0.9),
            Vec3::new(-5.0, 0.0, 10.0),
            Vec3::splat(1.0),
        ] {
            let s = grid.trilinear(p);
            let total: f32 = s.corner_weights(&grid).iter().map(|(_, w)| *w).sum();
            assert!(approx(total, 1.0, 1e-5), "weights summed to {total}");
        }
    }

    #[test]
    fn trilinear_at_probe_is_exact() {
        let grid = ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(2.0)), 3, 3, 3);
        let pos = grid.probe_position(1, 2, 0);
        let s = grid.trilinear(pos);
        let corners = s.corner_weights(&grid);
        let target = grid.linear_index(1, 2, 0);
        let w: f32 = corners
            .iter()
            .filter(|(idx, _)| *idx == target)
            .map(|(_, w)| *w)
            .sum();
        assert!(approx(w, 1.0, 1e-4), "expected full weight at probe, got {w}");
    }

    #[test]
    fn single_probe_axis_is_centered() {
        let grid = ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(4.0)), 1, 1, 1);
        assert_eq!(grid.probe_position(0, 0, 0), Vec3::splat(2.0));
        let s = grid.trilinear(Vec3::splat(3.0));
        let total: f32 = s.corner_weights(&grid).iter().map(|(_, w)| *w).sum();
        assert!(approx(total, 1.0, 1e-5));
    }
}
