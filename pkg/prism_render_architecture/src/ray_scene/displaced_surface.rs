//! Normal-displaced parametric surfaces for the `CPU` golden path.
//!
//! Displacement mapping is the standard way AAA pipelines add high-frequency
//! geometric detail — pores, bark, brick mortar, terrain micro-relief — on top
//! of a smooth low-order base surface. This module couples any member of the
//! `ray_scene` parametric-surface family (Bézier/B-spline/NURBS patches and
//! their authored control-net surfaces) with a sampled scalar height map and
//! offsets every surface point **along its own analytic normal**:
//!
//! ```text
//! P'(u, v) = P(u, v) + scale · h(u, v) · N(u, v)
//! ```
//!
//! The abstraction is the small [`ParametricSurface`] trait (position + unit
//! normal at `(u, v) ∈ [0, 1]²`), implemented here for every existing surface
//! primitive so a single [`DisplacedSurface`] wrapper works uniformly over all
//! of them. Because displacement destroys the base surface's analytic normal,
//! tessellation **re-derives shading normals from the displaced sample grid**
//! (central differences in the interior, one-sided at the border) and orients
//! each to the same hemisphere as the base normal, so the lit result is
//! correct rather than merely plausible. Shared grid edges land on identical
//! vertices, so the output [`IndexedBilinearPatchMesh`] stays watertight.
//!
//! All evaluation is purely linear (bilinear height sampling + the base
//! surfaces' own transcendental-free De Casteljau), so the module honours the
//! golden-path ban on `f32` transcendental functions.

use super::bezier_patch::BezierPatch;
use super::bezier_surface::BezierSurface;
use super::bspline_patch::BsplinePatch;
use super::bspline_surface::BsplineSurface;
use super::bvh::Aabb;
use super::indexed_bilinear_patch_mesh::{IndexedBilinearPatchMesh, IndexedBilinearPatchMeshBvh};
use super::nurbs_surface::NurbsSurface;
use super::rational_bezier_patch::RationalBezierPatch;

/// A smooth base surface that can be sampled for a position and a unit normal
/// at parameters `(u, v) ∈ [0, 1]²`.
///
/// Implemented for every `ray_scene` parametric-surface primitive so that
/// [`DisplacedSurface`] can displace any of them through one code path.
pub trait ParametricSurface {
    /// Returns the surface position at `(u, v)`.
    fn point(&self, u: f32, v: f32) -> [f32; 3];
    /// Returns the unit surface normal at `(u, v)`.
    fn normal(&self, u: f32, v: f32) -> [f32; 3];
}

impl ParametricSurface for BezierPatch {
    /// Forwards to [`BezierPatch::point`].
    fn point(&self, u: f32, v: f32) -> [f32; 3] {
        BezierPatch::point(self, u, v)
    }
    /// Forwards to [`BezierPatch::normal`].
    fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        BezierPatch::normal(self, u, v)
    }
}

impl ParametricSurface for BsplinePatch {
    /// Forwards to [`BsplinePatch::point`].
    fn point(&self, u: f32, v: f32) -> [f32; 3] {
        BsplinePatch::point(self, u, v)
    }
    /// Forwards to [`BsplinePatch::normal`].
    fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        BsplinePatch::normal(self, u, v)
    }
}

impl ParametricSurface for RationalBezierPatch {
    /// Forwards to [`RationalBezierPatch::point`].
    fn point(&self, u: f32, v: f32) -> [f32; 3] {
        RationalBezierPatch::point(self, u, v)
    }
    /// Forwards to [`RationalBezierPatch::normal`].
    fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        RationalBezierPatch::normal(self, u, v)
    }
}

impl ParametricSurface for BezierSurface {
    /// Forwards to [`BezierSurface::point`].
    fn point(&self, u: f32, v: f32) -> [f32; 3] {
        BezierSurface::point(self, u, v)
    }
    /// Forwards to [`BezierSurface::normal`].
    fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        BezierSurface::normal(self, u, v)
    }
}

impl ParametricSurface for BsplineSurface {
    /// Forwards to [`BsplineSurface::point`].
    fn point(&self, u: f32, v: f32) -> [f32; 3] {
        BsplineSurface::point(self, u, v)
    }
    /// Forwards to [`BsplineSurface::normal`].
    fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        BsplineSurface::normal(self, u, v)
    }
}

impl ParametricSurface for NurbsSurface {
    /// Forwards to [`NurbsSurface::point`].
    fn point(&self, u: f32, v: f32) -> [f32; 3] {
        NurbsSurface::point(self, u, v)
    }
    /// Forwards to [`NurbsSurface::normal`].
    fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        NurbsSurface::normal(self, u, v)
    }
}

/// Why a [`HeightMap`] could not be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeightMapError {
    /// A grid dimension was zero (both must be `>= 1`).
    EmptyDimension {
        /// Supplied width.
        width: usize,
        /// Supplied height.
        height: usize,
    },
    /// The flat sample pool length did not equal `width * height`.
    SampleCountMismatch {
        /// Number of samples actually supplied.
        actual: usize,
        /// Number of samples expected (`width * height`).
        expected: usize,
    },
}

impl core::fmt::Display for HeightMapError {
    /// Formats the error as a single human-readable diagnostic line.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyDimension { width, height } => {
                write!(f, "height map dimensions must be >= 1, got {width}x{height}")
            }
            Self::SampleCountMismatch { actual, expected } => {
                write!(f, "height map has {actual} samples, expected {expected}")
            }
        }
    }
}

impl std::error::Error for HeightMapError {}

/// A row-major scalar height field sampled bilinearly over `(u, v) ∈ [0, 1]²`.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightMap {
    /// Row-major heights, length `width * height` (`row` along `v`, `col`
    /// along `u`).
    data: Vec<f32>,
    /// Number of samples along `u`; always `>= 1`.
    width: usize,
    /// Number of samples along `v`; always `>= 1`.
    height: usize,
}

impl HeightMap {
    /// Builds a height map from a row-major `height × width` sample grid.
    ///
    /// Returns [`HeightMapError`] when either dimension is zero or
    /// `data.len() != width * height`.
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Result<Self, HeightMapError> {
        if width == 0 || height == 0 {
            return Err(HeightMapError::EmptyDimension { width, height });
        }
        let expected = width * height;
        if data.len() != expected {
            return Err(HeightMapError::SampleCountMismatch {
                actual: data.len(),
                expected,
            });
        }
        Ok(Self { data, width, height })
    }

    /// Returns the sample width (count along `u`).
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the sample height (count along `v`).
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the raw row-major samples.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Reads the raw sample at integer grid coordinates, clamped to bounds.
    fn texel(&self, col: usize, row: usize) -> f32 {
        let c = col.min(self.width - 1);
        let r = row.min(self.height - 1);
        self.data[r * self.width + c]
    }

    /// Bilinearly samples the height at `(u, v) ∈ [0, 1]²` (clamped at the
    /// border). A `1`-wide or `1`-tall map degenerates to nearest/row/column
    /// interpolation without any division by zero.
    #[must_use]
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let cu = u.clamp(0.0, 1.0);
        let cv = v.clamp(0.0, 1.0);
        let fx = cu * (self.width - 1) as f32;
        let fy = cv * (self.height - 1) as f32;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let x0 = x0 as usize;
        let y0 = y0 as usize;
        let h00 = self.texel(x0, y0);
        let h10 = self.texel(x0 + 1, y0);
        let h01 = self.texel(x0, y0 + 1);
        let h11 = self.texel(x0 + 1, y0 + 1);
        let top = h00 + (h10 - h00) * tx;
        let bot = h01 + (h11 - h01) * tx;
        top + (bot - top) * ty
    }
}

/// A parametric base surface displaced along its normal by a scaled height
/// field.
#[derive(Clone, Debug, PartialEq)]
pub struct DisplacedSurface<S: ParametricSurface> {
    /// The smooth base surface being displaced.
    base: S,
    /// The scalar height field sampled over `(u, v)`.
    height: HeightMap,
    /// Multiplier applied to every sampled height before offsetting.
    scale: f32,
}

impl<S: ParametricSurface> DisplacedSurface<S> {
    /// Wraps `base` with a `height` field scaled by `scale`.
    #[must_use]
    pub fn new(base: S, height: HeightMap, scale: f32) -> Self {
        Self { base, height, scale }
    }

    /// Borrows the underlying base surface.
    #[must_use]
    pub fn base(&self) -> &S {
        &self.base
    }

    /// Borrows the height field.
    #[must_use]
    pub fn height_map(&self) -> &HeightMap {
        &self.height
    }

    /// Returns the displacement scale.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Evaluates the displaced surface point `P + scale · h · N` at `(u, v)`.
    #[must_use]
    pub fn point(&self, u: f32, v: f32) -> [f32; 3] {
        let base = self.base.point(u, v);
        let n = self.base.normal(u, v);
        let offset = self.scale * self.height.sample(u, v);
        [
            base[0] + n[0] * offset,
            base[1] + n[1] * offset,
            base[2] + n[2] * offset,
        ]
    }

    /// Tessellates the displaced surface into one welded
    /// [`IndexedBilinearPatchMesh`] of `res_u × res_v` quad cells.
    ///
    /// Positions come from [`Self::point`]; shading normals are re-derived from
    /// the displaced sample grid with central differences (one-sided at the
    /// border) and oriented to the base normal's hemisphere, because
    /// displacement invalidates the base analytic normal. `UV`s are the global
    /// `(u, v)` parameters. Each resolution is clamped to at least `1`.
    #[must_use]
    pub fn tessellate(&self, res_u: usize, res_v: usize) -> IndexedBilinearPatchMesh {
        let nu = res_u.max(1);
        let nv = res_v.max(1);
        let cols = nu + 1;
        let rows = nv + 1;
        let mut positions = Vec::with_capacity(cols * rows);
        let mut base_normals = Vec::with_capacity(cols * rows);
        let mut uvs = Vec::with_capacity(cols * rows);
        for j in 0..rows {
            let v = j as f32 / nv as f32;
            for i in 0..cols {
                let u = i as f32 / nu as f32;
                positions.push(self.point(u, v));
                base_normals.push(self.base.normal(u, v));
                uvs.push([u, v]);
            }
        }
        let normals = recompute_grid_normals(&positions, &base_normals, cols, rows);
        let mut indices = Vec::with_capacity(nu * nv);
        for j in 0..nv {
            for i in 0..nu {
                let vid = |ii: usize, jj: usize| (jj * cols + ii) as u32;
                // (u, v) corner order 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1).
                indices.push([
                    vid(i, j),
                    vid(i + 1, j),
                    vid(i + 1, j + 1),
                    vid(i, j + 1),
                ]);
            }
        }
        IndexedBilinearPatchMesh::new(positions, normals, uvs, indices)
            .expect("displaced tessellation indices are always in range")
    }

    /// Tessellates the displaced surface and builds a traversable
    /// [`IndexedBilinearPatchMeshBvh`] in one step.
    #[must_use]
    pub fn tessellate_bvh(&self, res_u: usize, res_v: usize) -> IndexedBilinearPatchMeshBvh {
        IndexedBilinearPatchMeshBvh::build(self.tessellate(res_u, res_v))
    }

    /// Returns a conservative world-space bound of the displaced surface.
    ///
    /// It expands the base control extent — approximated here by sampling the
    /// displaced grid at `res × res` — into an axis-aligned box. Traversal still
    /// uses the tessellated mesh's own `BVH` for exact hits; this box is only a
    /// coarse culling bound.
    #[must_use]
    pub fn sampled_aabb(&self, res: usize) -> Aabb {
        let n = res.max(1);
        let mut lo = self.point(0.0, 0.0);
        let mut hi = lo;
        for j in 0..=n {
            let v = j as f32 / n as f32;
            for i in 0..=n {
                let u = i as f32 / n as f32;
                let p = self.point(u, v);
                for k in 0..3 {
                    if p[k] < lo[k] {
                        lo[k] = p[k];
                    }
                    if p[k] > hi[k] {
                        hi[k] = p[k];
                    }
                }
            }
        }
        Aabb::new(lo, hi)
    }
}

/// Re-derives per-vertex shading normals from a displaced position grid.
///
/// For each `(col, row)` vertex it forms the `u`-tangent and `v`-tangent from
/// neighbouring grid positions (central difference in the interior, one-sided
/// at the border), takes their cross product, and orients the result into the
/// same hemisphere as the corresponding base normal. Degenerate (zero-length)
/// cross products fall back to the base normal so every normal is finite and
/// unit length. `cols` and `rows` are the grid dimensions and both inputs are
/// row-major of length `cols * rows`.
fn recompute_grid_normals(
    positions: &[[f32; 3]],
    base_normals: &[[f32; 3]],
    cols: usize,
    rows: usize,
) -> Vec<[f32; 3]> {
    let mut normals = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let (cl, cr) = if col == 0 {
                (col, col + 1)
            } else if col + 1 == cols {
                (col - 1, col)
            } else {
                (col - 1, col + 1)
            };
            let (rb, rt) = if row == 0 {
                (row, row + 1)
            } else if row + 1 == rows {
                (row - 1, row)
            } else {
                (row - 1, row + 1)
            };
            let pu_lo = positions[row * cols + cl];
            let pu_hi = positions[row * cols + cr];
            let pv_lo = positions[rb * cols + col];
            let pv_hi = positions[rt * cols + col];
            let tu = [pu_hi[0] - pu_lo[0], pu_hi[1] - pu_lo[1], pu_hi[2] - pu_lo[2]];
            let tv = [pv_hi[0] - pv_lo[0], pv_hi[1] - pv_lo[1], pv_hi[2] - pv_lo[2]];
            let n = [
                tu[1] * tv[2] - tu[2] * tv[1],
                tu[2] * tv[0] - tu[0] * tv[2],
                tu[0] * tv[1] - tu[1] * tv[0],
            ];
            let len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
            let base = base_normals[idx];
            if len2 > 0.0 {
                let inv = 1.0 / len2.sqrt();
                let mut unit = [n[0] * inv, n[1] * inv, n[2] * inv];
                let dot = unit[0] * base[0] + unit[1] * base[1] + unit[2] * base[2];
                if dot < 0.0 {
                    unit = [-unit[0], -unit[1], -unit[2]];
                }
                normals.push(unit);
            } else {
                normals.push(base);
            }
        }
    }
    normals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;

    /// Minimal xorshift `RNG` for deterministic test parameters.
    struct Rng(u64);

    impl Rng {
        /// Seeds the generator, forcing a non-zero state.
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        /// Advances the state and returns the top 32 bits.
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }

        /// Returns a float uniformly in `[0, 1]`.
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }
    }

    /// Euclidean length of `a`.
    fn len(a: [f32; 3]) -> f32 {
        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
    }

    /// Component-wise difference `a - b`.
    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    /// A flat `z = 0` bicubic Bézier patch (unit square in `xy`).
    fn flat_patch() -> BezierPatch {
        let mut net = [[0.0f32; 3]; 16];
        for r in 0..4 {
            for c in 0..4 {
                net[r * 4 + c] = [c as f32 / 3.0, r as f32 / 3.0, 0.0];
            }
        }
        BezierPatch::new(net)
    }

    /// A constant height map of value `h`.
    fn constant_map(w: usize, h: usize, value: f32) -> HeightMap {
        HeightMap::new(vec![value; w * h], w, h).unwrap()
    }

    #[test]
    fn heightmap_rejects_degenerate() {
        assert_eq!(
            HeightMap::new(vec![], 0, 4),
            Err(HeightMapError::EmptyDimension { width: 0, height: 4 })
        );
        assert_eq!(
            HeightMap::new(vec![1.0; 3], 2, 2),
            Err(HeightMapError::SampleCountMismatch { actual: 3, expected: 4 })
        );
    }

    #[test]
    fn heightmap_bilinear_interpolates() {
        // 2x2 map: corners 0,1,2,3 row-major -> (0,0)=0 (1,0)=1 (0,1)=2 (1,1)=3.
        let m = HeightMap::new(vec![0.0, 1.0, 2.0, 3.0], 2, 2).unwrap();
        assert!((m.sample(0.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((m.sample(1.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((m.sample(0.0, 1.0) - 2.0).abs() < 1e-6);
        assert!((m.sample(1.0, 1.0) - 3.0).abs() < 1e-6);
        // Centre is the average of all four corners = 1.5.
        assert!((m.sample(0.5, 0.5) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn heightmap_single_row_or_column_is_safe() {
        let col = HeightMap::new(vec![1.0, 3.0], 1, 2).unwrap();
        // Width 1: no u interpolation; v interpolates 1 -> 3.
        assert!((col.sample(0.7, 0.5) - 2.0).abs() < 1e-6);
        let single = HeightMap::new(vec![5.0], 1, 1).unwrap();
        assert!((single.sample(0.3, 0.9) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn zero_scale_matches_base_point() {
        let base = flat_patch();
        let d = DisplacedSurface::new(flat_patch(), constant_map(4, 4, 7.0), 0.0);
        let mut rng = Rng::new(0x11);
        for _ in 0..64 {
            let u = rng.unit();
            let v = rng.unit();
            assert!(len(sub(d.point(u, v), base.point(u, v))) < 1e-6);
        }
    }

    #[test]
    fn flat_base_displaces_along_up_normal() {
        // Flat patch normal is +z; constant height h with scale s -> z = s*h.
        let d = DisplacedSurface::new(flat_patch(), constant_map(2, 2, 2.0), 0.5);
        let mut rng = Rng::new(0x22);
        for _ in 0..64 {
            let u = rng.unit();
            let v = rng.unit();
            let p = d.point(u, v);
            assert!((p[2] - 1.0).abs() < 1e-5, "expected z=1.0, got {}", p[2]);
        }
    }

    #[test]
    fn recomputed_normal_flat_is_up() {
        // A flat base with constant displacement stays planar, so the
        // re-derived shading normal returned by a BVH hit must point straight
        // up (+z) — proving the grid-normal recomputation is correct.
        let d = DisplacedSurface::new(flat_patch(), constant_map(4, 4, 1.0), 0.3);
        let bvh = d.tessellate_bvh(8, 8);
        let target = d.point(0.53, 0.47);
        let origin = [target[0], target[1], target[2] + 3.0];
        let ray = Ray::infinite(origin, [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray should hit the flat surface");
        let n = hit.shading_normal;
        assert!((n[2].abs() - 1.0).abs() < 1e-4, "flat normal must be ±z, got {n:?}");
        assert!(n[0].abs() < 1e-4 && n[1].abs() < 1e-4, "flat normal must be axis-aligned, got {n:?}");
    }

    #[test]
    fn trait_works_over_control_net_surface() {
        // A 4x4 flat B-spline-cage patch -> B-spline surface, displaced.
        let mut grid = Vec::with_capacity(16);
        for r in 0..4 {
            for c in 0..4 {
                grid.push([c as f32, r as f32, 0.0]);
            }
        }
        let surface = BsplineSurface::new(grid, 4, 4).unwrap();
        let d = DisplacedSurface::new(surface, constant_map(2, 2, 1.0), 1.0);
        // Flat cage normal is +z, so displaced z == scale*h == 1 everywhere.
        let p = d.point(0.5, 0.5);
        assert!((p[2] - 1.0).abs() < 1e-4, "expected z≈1, got {}", p[2]);
    }

    #[test]
    fn tessellation_has_expected_shape() {
        let d = DisplacedSurface::new(flat_patch(), constant_map(4, 4, 1.0), 0.2);
        let mesh = d.tessellate(5, 7);
        assert_eq!(mesh.patch_count(), 5 * 7);
        assert_eq!(mesh.vertex_count(), 6 * 8);
    }

    #[test]
    fn bump_raises_surface_above_base() {
        // Height map with a central spike; displaced bump should rise in +z.
        let w = 5;
        let h = 5;
        let mut data = vec![0.0f32; w * h];
        data[2 * w + 2] = 1.0; // centre texel
        let map = HeightMap::new(data, w, h).unwrap();
        let d = DisplacedSurface::new(flat_patch(), map, 1.0);
        let centre = d.point(0.5, 0.5);
        assert!(centre[2] > 0.5, "centre should bulge up, z={}", centre[2]);
        let corner = d.point(0.0, 0.0);
        assert!(corner[2].abs() < 1e-5, "corner should stay flat, z={}", corner[2]);
    }

    #[test]
    fn tessellated_bvh_hits_displaced_bump() {
        let w = 5;
        let h = 5;
        let mut data = vec![0.0f32; w * h];
        data[2 * w + 2] = 1.0;
        let map = HeightMap::new(data, w, h).unwrap();
        let d = DisplacedSurface::new(flat_patch(), map, 1.0);
        let bvh = d.tessellate_bvh(15, 15);
        // Aim at an off-vertex parameter inside a quad.
        let target = d.point(0.53, 0.47);
        let origin = [target[0], target[1], target[2] + 5.0];
        let ray = Ray::infinite(origin, [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray should hit the bump");
        assert!(hit.t > 0.0);
        let p = ray.at(hit.t);
        assert!(
            (p[2] - target[2]).abs() < 0.1,
            "hit z={} should match sampled z={}",
            p[2],
            target[2]
        );
    }

    #[test]
    fn sampled_aabb_contains_bump() {
        let w = 3;
        let h = 3;
        let mut data = vec![0.0f32; w * h];
        data[w + 1] = 1.0;
        let map = HeightMap::new(data, w, h).unwrap();
        let d = DisplacedSurface::new(flat_patch(), map, 1.0);
        let aabb = d.sampled_aabb(16);
        assert!(aabb.max[2] > 0.4, "aabb must enclose the raised centre");
        assert!(aabb.min[2] <= 1e-4, "aabb must enclose the flat border");
    }
}
