//! Composite bicubic **Bézier surface** over a `(3m + 1) × (3n + 1)` control
//! grid, tessellated into a single welded [`IndexedBilinearPatchMesh`] for the
//! `CPU` golden path.
//!
//! Where [`super::bezier_patch::BezierPatch`] is one isolated 4×4 net, an
//! authored Bézier surface tiles many such patches edge to edge. Unlike the
//! B-spline surface in [`super::bspline_surface`] — whose spans overlap by
//! three control rows/columns (stride 1) and are therefore globally `C²` — a
//! Bézier surface's patches meet with a **stride of three**: adjacent patches
//! share exactly one boundary control row/column. A grid of `(3m + 1)` columns
//! by `(3n + 1)` rows therefore partitions into exactly `m × n` bicubic
//! patches. Because neighbours share only the boundary handles, the surface is
//! `C⁰` across patch seams (and `G¹` only when the straddling control points
//! are made collinear), while every patch *interpolates* its four corner
//! handles. This is the historical authored primitive — the Utah teapot and a
//! great many classic assets are expressed as nets of bicubic Bézier patches —
//! and remains the natural import target for exported Bézier cages.
//!
//! The whole surface is parameterised by `(u, v) ∈ [0, 1]²`. The parameter is
//! scaled by the patch count, the integer part selects the patch and the
//! fractional part is the local patch parameter, so `u = 1` lands on the last
//! patch's trailing edge. Each patch is evaluated by the purely linear (no
//! transcendental basis functions) De Casteljau recursion of
//! [`super::bezier_patch`], and because neighbouring patches share identical
//! boundary control points their shared edge curves agree exactly. Sampling a
//! single global `(nu + 1) × (nv + 1)` grid therefore welds into a watertight
//! mesh with no cracks between patches, even though shading normals are only
//! piecewise-continuous across seams.

use super::bezier_patch::BezierPatch;
use super::bvh::Aabb;
use super::indexed_bilinear_patch_mesh::{IndexedBilinearPatchMesh, IndexedBilinearPatchMeshBvh};

/// Why a [`BezierSurface`] could not be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BezierSurfaceError {
    /// The grid had fewer than four control columns (minimum for one cubic
    /// patch along `u`).
    TooFewColumns {
        /// Number of columns supplied.
        cols: usize,
    },
    /// The grid had fewer than four control rows (minimum for one cubic patch
    /// along `v`).
    TooFewRows {
        /// Number of rows supplied.
        rows: usize,
    },
    /// The column count was not of the form `3m + 1`, so the net could not be
    /// tiled into whole stride-three patches along `u`.
    InvalidColumnCount {
        /// Number of columns supplied.
        cols: usize,
    },
    /// The row count was not of the form `3n + 1`, so the net could not be
    /// tiled into whole stride-three patches along `v`.
    InvalidRowCount {
        /// Number of rows supplied.
        rows: usize,
    },
    /// The flat control pool length did not equal `rows * cols`.
    ControlCountMismatch {
        /// Number of control points actually supplied.
        actual: usize,
        /// Number of control points expected (`rows * cols`).
        expected: usize,
    },
}

impl core::fmt::Display for BezierSurfaceError {
    /// Formats the error as a single human-readable diagnostic line.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooFewColumns { cols } => {
                write!(f, "control grid needs at least 4 columns, got {cols}")
            }
            Self::TooFewRows { rows } => {
                write!(f, "control grid needs at least 4 rows, got {rows}")
            }
            Self::InvalidColumnCount { cols } => {
                write!(f, "column count must be 3m+1 to tile bicubic patches, got {cols}")
            }
            Self::InvalidRowCount { rows } => {
                write!(f, "row count must be 3n+1 to tile bicubic patches, got {rows}")
            }
            Self::ControlCountMismatch { actual, expected } => {
                write!(f, "control grid has {actual} points, expected {expected}")
            }
        }
    }
}

impl std::error::Error for BezierSurfaceError {}

/// A composite bicubic Bézier surface defined by a row-major control grid.
#[derive(Clone, Debug, PartialEq)]
pub struct BezierSurface {
    /// Row-major control grid (`col` along `u`, `row` along `v`), length
    /// `rows * cols`.
    control: Vec<[f32; 3]>,
    /// Number of control rows (along `v`); always `3n + 1` with `n >= 1`.
    rows: usize,
    /// Number of control columns (along `u`); always `3m + 1` with `m >= 1`.
    cols: usize,
}

/// Locates the patch index and local parameter for a global parameter `t`.
///
/// `t` is clamped to `[0, 1]`, scaled by `patches`, and split into an integer
/// patch index (clamped to the last patch) and a fractional local parameter in
/// `[0, 1]`. `patches` is always `>= 1`, so the clamp is well defined.
fn locate(t: f32, patches: usize) -> (usize, f32) {
    let clamped = t.clamp(0.0, 1.0);
    let scaled = clamped * patches as f32;
    let last = patches - 1;
    let index = (scaled.floor() as usize).min(last);
    let local = scaled - index as f32;
    (index, local)
}

impl BezierSurface {
    /// Builds a surface from a row-major `rows × cols` control grid.
    ///
    /// Returns [`BezierSurfaceError`] when `cols < 4`, `rows < 4`, the column
    /// or row count is not of the form `3k + 1`, or
    /// `control.len() != rows * cols`.
    pub fn new(
        control: Vec<[f32; 3]>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, BezierSurfaceError> {
        if cols < 4 {
            return Err(BezierSurfaceError::TooFewColumns { cols });
        }
        if rows < 4 {
            return Err(BezierSurfaceError::TooFewRows { rows });
        }
        if !(cols - 1).is_multiple_of(3) {
            return Err(BezierSurfaceError::InvalidColumnCount { cols });
        }
        if !(rows - 1).is_multiple_of(3) {
            return Err(BezierSurfaceError::InvalidRowCount { rows });
        }
        let expected = rows * cols;
        if control.len() != expected {
            return Err(BezierSurfaceError::ControlCountMismatch {
                actual: control.len(),
                expected,
            });
        }
        Ok(Self { control, rows, cols })
    }

    /// Returns the control grid as a flat row-major slice.
    #[must_use]
    pub fn control(&self) -> &[[f32; 3]] {
        &self.control
    }

    /// Returns the number of control rows (along `v`).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of control columns (along `u`).
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of bicubic patches along `u` (`(cols - 1) / 3`).
    #[must_use]
    pub fn u_patch_count(&self) -> usize {
        (self.cols - 1) / 3
    }

    /// Returns the number of bicubic patches along `v` (`(rows - 1) / 3`).
    #[must_use]
    pub fn v_patch_count(&self) -> usize {
        (self.rows - 1) / 3
    }

    /// Extracts the [`super::bezier_patch::BezierPatch`] for the patch at
    /// column index `patch_u` and row index `patch_v`.
    ///
    /// The 4×4 window is `control[3 * patch_v + wr][3 * patch_u + wc]` for
    /// `wr, wc ∈ 0..4`, so neighbouring patches share exactly one control
    /// row/column (stride three) and the surface is `C⁰` across seams. The
    /// caller guarantees `patch_u < u_patch_count()` and
    /// `patch_v < v_patch_count()`.
    #[must_use]
    pub fn patch_at(&self, patch_u: usize, patch_v: usize) -> BezierPatch {
        let mut window = [[0.0f32; 3]; 16];
        for wr in 0..4 {
            for wc in 0..4 {
                window[wr * 4 + wc] =
                    self.control[(3 * patch_v + wr) * self.cols + (3 * patch_u + wc)];
            }
        }
        BezierPatch::new(window)
    }

    /// Evaluates the surface position at global parameters `(u, v) ∈ [0, 1]²`.
    #[must_use]
    pub fn point(&self, u: f32, v: f32) -> [f32; 3] {
        let (patch_u, local_u) = locate(u, self.u_patch_count());
        let (patch_v, local_v) = locate(v, self.v_patch_count());
        self.patch_at(patch_u, patch_v).point(local_u, local_v)
    }

    /// Evaluates the unit analytic surface normal at global `(u, v)`.
    ///
    /// Because the surface is only `C⁰` across patch seams, the normal is
    /// discontinuous there; `locate` resolves exactly on-seam parameters to the
    /// lower-indexed patch so the result stays well defined and finite.
    #[must_use]
    pub fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        let (patch_u, local_u) = locate(u, self.u_patch_count());
        let (patch_v, local_v) = locate(v, self.v_patch_count());
        self.patch_at(patch_u, patch_v).normal(local_u, local_v)
    }

    /// Returns the axis-aligned bounding box of every control point.
    ///
    /// Because each Bézier patch stays within the convex hull of its net, this
    /// control-point box is also a valid (loose) bound on the surface itself;
    /// traversal still uses the tessellated mesh's own `BVH` for exact hits.
    #[must_use]
    pub fn control_aabb(&self) -> Aabb {
        let mut lo = self.control[0];
        let mut hi = self.control[0];
        for p in &self.control[1..] {
            for k in 0..3 {
                if p[k] < lo[k] {
                    lo[k] = p[k];
                }
                if p[k] > hi[k] {
                    hi[k] = p[k];
                }
            }
        }
        Aabb::new(lo, hi)
    }

    /// Tessellates the whole surface into one welded
    /// [`IndexedBilinearPatchMesh`] with `res_u × res_v` quad cells **per
    /// patch**.
    ///
    /// A single global `(nu + 1) × (nv + 1)` sample grid is built with
    /// `nu = u_patch_count() * res_u` and `nv = v_patch_count() * res_v`, where
    /// each resolution is clamped to at least `1`. Positions and analytic
    /// normals are sampled from [`Self::point`]/[`Self::normal`], and `UV`s are
    /// the global `(u, v)` parameters. Shared patch edges land on identical grid
    /// vertices, so the mesh is watertight even though shading normals may crease
    /// across `C⁰` seams.
    #[must_use]
    pub fn tessellate(&self, res_u: usize, res_v: usize) -> IndexedBilinearPatchMesh {
        let ru = res_u.max(1);
        let rv = res_v.max(1);
        let nu = self.u_patch_count() * ru;
        let nv = self.v_patch_count() * rv;
        let cols = nu + 1;
        let rows = nv + 1;
        let mut positions = Vec::with_capacity(cols * rows);
        let mut normals = Vec::with_capacity(cols * rows);
        let mut uvs = Vec::with_capacity(cols * rows);
        for j in 0..rows {
            let v = j as f32 / nv as f32;
            for i in 0..cols {
                let u = i as f32 / nu as f32;
                positions.push(self.point(u, v));
                normals.push(self.normal(u, v));
                uvs.push([u, v]);
            }
        }
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
            .expect("surface tessellation indices are always in range")
    }

    /// Tessellates the surface and builds a traversable
    /// [`IndexedBilinearPatchMeshBvh`] in one step.
    #[must_use]
    pub fn tessellate_bvh(&self, res_u: usize, res_v: usize) -> IndexedBilinearPatchMeshBvh {
        IndexedBilinearPatchMeshBvh::build(self.tessellate(res_u, res_v))
    }
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

        /// Returns a float uniformly in `[lo, hi]`.
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// Component-wise difference `a - b`.
    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    /// Euclidean length of `a`.
    fn len(a: [f32; 3]) -> f32 {
        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
    }

    /// Builds a flat `rows × cols` grid in the `z = 0` plane.
    fn flat_grid(rows: usize, cols: usize) -> Vec<[f32; 3]> {
        let mut g = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let x = c as f32 / (cols - 1) as f32;
                let y = r as f32 / (rows - 1) as f32;
                g.push([x, y, 0.0]);
            }
        }
        g
    }

    /// Builds a `rows × cols` grid raised toward the centre by `height`.
    fn domed_grid(rows: usize, cols: usize, height: f32) -> Vec<[f32; 3]> {
        let mut g = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let u = c as f32 / (cols - 1) as f32;
                let v = r as f32 / (rows - 1) as f32;
                let bump = (u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5);
                g.push([u, v, height * (0.25 - bump)]);
            }
        }
        g
    }

    #[test]
    fn rejects_degenerate_grids() {
        assert_eq!(
            BezierSurface::new(vec![[0.0; 3]; 12], 4, 3),
            Err(BezierSurfaceError::TooFewColumns { cols: 3 })
        );
        assert_eq!(
            BezierSurface::new(vec![[0.0; 3]; 12], 3, 4),
            Err(BezierSurfaceError::TooFewRows { rows: 3 })
        );
        // 4 rows is valid, 6 columns is not 3m+1.
        assert_eq!(
            BezierSurface::new(vec![[0.0; 3]; 24], 4, 6),
            Err(BezierSurfaceError::InvalidColumnCount { cols: 6 })
        );
        // 7 columns valid, 5 rows is not 3n+1.
        assert_eq!(
            BezierSurface::new(vec![[0.0; 3]; 35], 5, 7),
            Err(BezierSurfaceError::InvalidRowCount { rows: 5 })
        );
        assert_eq!(
            BezierSurface::new(vec![[0.0; 3]; 15], 4, 4),
            Err(BezierSurfaceError::ControlCountMismatch {
                actual: 15,
                expected: 16,
            })
        );
    }

    #[test]
    fn patch_counts_follow_grid_size() {
        // 7 cols -> 2 patches along u, 10 rows -> 3 patches along v.
        let s = BezierSurface::new(flat_grid(10, 7), 10, 7).unwrap();
        assert_eq!(s.u_patch_count(), 2);
        assert_eq!(s.v_patch_count(), 3);
        assert_eq!(s.rows(), 10);
        assert_eq!(s.cols(), 7);
    }

    #[test]
    fn flat_surface_stays_planar_with_up_normal() {
        let s = BezierSurface::new(flat_grid(7, 7), 7, 7).unwrap();
        let mut rng = Rng::new(0x5eed);
        for _ in 0..64 {
            let u = rng.unit();
            let v = rng.unit();
            let p = s.point(u, v);
            assert!(p[2].abs() < 1e-5, "flat surface should stay at z=0, got {p:?}");
            let n = s.normal(u, v);
            assert!((n[2].abs() - 1.0).abs() < 1e-4, "flat normal should be ±z, got {n:?}");
        }
    }

    #[test]
    fn minimal_grid_matches_single_patch() {
        let mut rng = Rng::new(0xabc);
        let mut net = [[0.0f32; 3]; 16];
        let mut grid = Vec::with_capacity(16);
        for item in net.iter_mut() {
            let p = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            *item = p;
            grid.push(p);
        }
        let patch = BezierPatch::new(net);
        let surface = BezierSurface::new(grid, 4, 4).unwrap();
        assert_eq!(surface.u_patch_count(), 1);
        assert_eq!(surface.v_patch_count(), 1);
        for _ in 0..64 {
            let u = rng.unit();
            let v = rng.unit();
            let a = surface.point(u, v);
            let b = patch.point(u, v);
            assert!(len(sub(a, b)) < 1e-5, "surface should match single patch");
        }
    }

    #[test]
    fn interior_patch_matches_its_patch() {
        let s = BezierSurface::new(domed_grid(7, 7, 0.8), 7, 7).unwrap();
        // Patch (1, 1) spans u,v in [0.5, 1.0]; its local (0.3, 0.7) maps to
        // global (0.65, 0.85).
        let patch = s.patch_at(1, 1);
        let expected = patch.point(0.3, 0.7);
        let got = s.point(0.65, 0.85);
        assert!(len(sub(expected, got)) < 1e-5, "point must agree with patch_at");
    }

    #[test]
    fn adjacent_patches_agree_on_shared_edge() {
        // C0: patch (0,0) trailing u-edge equals patch (1,0) leading u-edge
        // because they share the same boundary control column.
        let s = BezierSurface::new(domed_grid(7, 10, 0.6), 7, 10).unwrap();
        let left = s.patch_at(0, 0);
        let right = s.patch_at(1, 0);
        let mut rng = Rng::new(0xf00d);
        for _ in 0..32 {
            let v = rng.unit();
            let a = left.point(1.0, v);
            let b = right.point(0.0, v);
            assert!(len(sub(a, b)) < 1e-5, "patch seam must be watertight: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn analytic_normal_matches_finite_difference() {
        let s = BezierSurface::new(domed_grid(7, 7, 0.7), 7, 7).unwrap();
        let mut rng = Rng::new(0x1234);
        for _ in 0..48 {
            // Stay well inside a single patch so the normal is smooth (avoid
            // the C0 seam at the patch midline u=0.5 / v=0.5).
            let u = rng.range(0.05, 0.45);
            let v = rng.range(0.05, 0.45);
            let eps = 1e-3;
            let du = sub(s.point(u + eps, v), s.point(u - eps, v));
            let dv = sub(s.point(u, v + eps), s.point(u, v - eps));
            let fd = [
                du[1] * dv[2] - du[2] * dv[1],
                du[2] * dv[0] - du[0] * dv[2],
                du[0] * dv[1] - du[1] * dv[0],
            ];
            let fl = len(fd);
            if fl < 1e-6 {
                continue;
            }
            let fd = [fd[0] / fl, fd[1] / fl, fd[2] / fl];
            let n = s.normal(u, v);
            // Allow either orientation; compare absolute dot.
            let dot = (n[0] * fd[0] + n[1] * fd[1] + n[2] * fd[2]).abs();
            assert!(dot > 0.99, "analytic normal must match finite difference, dot={dot}");
        }
    }

    #[test]
    fn tessellation_has_expected_shape() {
        let s = BezierSurface::new(domed_grid(7, 10, 0.5), 7, 10).unwrap();
        // 3 patches along u, 2 along v, res 4x5 -> 12x10 quads.
        let mesh = s.tessellate(4, 5);
        let nu = 3 * 4;
        let nv = 2 * 5;
        assert_eq!(mesh.patch_count(), nu * nv);
        assert_eq!(mesh.vertex_count(), (nu + 1) * (nv + 1));
    }

    #[test]
    fn tessellated_bvh_hits_the_surface() {
        // Dome bulging toward +z; shoot straight down from above.
        let s = BezierSurface::new(domed_grid(7, 7, 1.0), 7, 7).unwrap();
        let bvh = s.tessellate_bvh(15, 15);
        // Aim at an off-vertex, off-seam parameter: with 2 patches × 15 cells
        // the grid vertices land on multiples of 1/30 and the seams on 0.5, so
        // (0.53, 0.47) sits strictly inside a quad and never on a shared vertex
        // where a watertight patch mesh would legitimately report a miss.
        let target = s.point(0.53, 0.47);
        let origin = [target[0], target[1], target[2] + 5.0];
        let ray = Ray::infinite(origin, [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray should hit the dome");
        assert!(hit.t > 0.0, "hit must be in front of the ray");
        let hit_point = ray.at(hit.t);
        assert!(
            (hit_point[2] - target[2]).abs() < 0.1,
            "hit should land near the sampled surface point, got z={} want z={}",
            hit_point[2],
            target[2]
        );
    }
}
