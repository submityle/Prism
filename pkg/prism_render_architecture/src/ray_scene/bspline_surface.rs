//! Uniform bicubic B-spline **surface** over an arbitrary `R × C` control grid,
//! tessellated into a single welded [`IndexedBilinearPatchMesh`] for the `CPU`
//! golden path.
//!
//! Where [`super::bspline_patch::BsplinePatch`] is a single 4×4 span, a real
//! authored surface is a whole control net: an `R × C` grid of handles that is
//! partitioned into `(C - 3) × (R - 3)` overlapping cubic spans sharing their
//! boundary control rows and columns. Each span is exactly one
//! [`super::bspline_patch::BsplinePatch`], so the surface inherits the span's
//! properties: it stays inside the convex hull of its net, interpolates no
//! control point, and is globally `C²` continuous across span boundaries
//! because adjacent spans share three control columns/rows. This is the
//! authoring-level primitive for smooth deformable cages — subdivision-surface
//! limit stand-ins, cloth/skin shells, blobby organic forms — where the grid
//! is a handle cage rather than a set of samples.
//!
//! The whole surface is parameterised by `(u, v) ∈ [0, 1]²`. The parameter is
//! scaled by the span count, the integer part selects the span and the
//! fractional part is the local span parameter, so `u = 1` lands on the last
//! span's trailing edge. Because the span conversion is purely linear (no
//! transcendental basis functions, see [`super::bspline_patch`]) and the global
//! `C²` continuity guarantees adjacent spans agree on their shared edge,
//! sampling a single global `(nu + 1) × (nv + 1)` grid yields a watertight,
//! smoothly shaded mesh with no cracks between spans.

use super::bspline_patch::BsplinePatch;
use super::bvh::Aabb;
use super::indexed_bilinear_patch_mesh::{IndexedBilinearPatchMesh, IndexedBilinearPatchMeshBvh};

/// Why a [`BsplineSurface`] could not be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BsplineSurfaceError {
    /// The grid had fewer than four control columns (minimum for one cubic
    /// span along `u`).
    TooFewColumns {
        /// Number of columns supplied.
        cols: usize,
    },
    /// The grid had fewer than four control rows (minimum for one cubic span
    /// along `v`).
    TooFewRows {
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

impl core::fmt::Display for BsplineSurfaceError {
    /// Formats the error as a single human-readable diagnostic line.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooFewColumns { cols } => {
                write!(f, "control grid needs at least 4 columns, got {cols}")
            }
            Self::TooFewRows { rows } => {
                write!(f, "control grid needs at least 4 rows, got {rows}")
            }
            Self::ControlCountMismatch { actual, expected } => write!(
                f,
                "control-point count {actual} does not match rows * cols = {expected}"
            ),
        }
    }
}

impl std::error::Error for BsplineSurfaceError {}

/// A uniform bicubic B-spline surface defined by an `R × C` control grid.
///
/// The net is stored row-major (`control[row * cols + col]`), with `col`
/// advancing along `u` and `row` advancing along `v`, matching
/// [`super::bspline_patch::BsplinePatch`]. The surface stays inside the convex
/// hull of the net and interpolates no control point.
#[derive(Clone, Debug, PartialEq)]
pub struct BsplineSurface {
    /// Row-major control grid (`col` along `u`, `row` along `v`), length
    /// `rows * cols`.
    control: Vec<[f32; 3]>,
    /// Number of control rows (along `v`); always `>= 4`.
    rows: usize,
    /// Number of control columns (along `u`); always `>= 4`.
    cols: usize,
}

/// Locates the span index and local parameter for a global parameter `t`.
///
/// `t` is clamped to `[0, 1]`, scaled by `spans`, and split into an integer
/// span index (clamped to the last span) and a fractional local parameter in
/// `[0, 1]`. `spans` is always `>= 1`, so the clamp is well defined.
fn locate(t: f32, spans: usize) -> (usize, f32) {
    let clamped = t.clamp(0.0, 1.0);
    let scaled = clamped * spans as f32;
    let last = spans - 1;
    let index = (scaled.floor() as usize).min(last);
    let local = scaled - index as f32;
    (index, local)
}

impl BsplineSurface {
    /// Builds a surface from a row-major `rows × cols` control grid.
    ///
    /// Returns [`BsplineSurfaceError`] when `cols < 4`, `rows < 4`, or
    /// `control.len() != rows * cols`.
    pub fn new(
        control: Vec<[f32; 3]>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, BsplineSurfaceError> {
        if cols < 4 {
            return Err(BsplineSurfaceError::TooFewColumns { cols });
        }
        if rows < 4 {
            return Err(BsplineSurfaceError::TooFewRows { rows });
        }
        let expected = rows * cols;
        if control.len() != expected {
            return Err(BsplineSurfaceError::ControlCountMismatch {
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

    /// Returns the number of cubic spans along `u` (`cols - 3`).
    #[must_use]
    pub fn u_span_count(&self) -> usize {
        self.cols - 3
    }

    /// Returns the number of cubic spans along `v` (`rows - 3`).
    #[must_use]
    pub fn v_span_count(&self) -> usize {
        self.rows - 3
    }

    /// Extracts the [`super::bspline_patch::BsplinePatch`] for the span at
    /// column index `span_u` and row index `span_v`.
    ///
    /// The 4×4 window is `control[span_v + wr][span_u + wc]` for
    /// `wr, wc ∈ 0..4`, so neighbouring spans share three control rows/columns
    /// and the global surface is `C²`. The caller guarantees
    /// `span_u < u_span_count()` and `span_v < v_span_count()`.
    #[must_use]
    pub fn patch_at(&self, span_u: usize, span_v: usize) -> BsplinePatch {
        let mut window = [[0.0f32; 3]; 16];
        for wr in 0..4 {
            for wc in 0..4 {
                window[wr * 4 + wc] = self.control[(span_v + wr) * self.cols + (span_u + wc)];
            }
        }
        BsplinePatch::new(window)
    }

    /// Evaluates the surface position at global parameters `(u, v) ∈ [0, 1]²`.
    #[must_use]
    pub fn point(&self, u: f32, v: f32) -> [f32; 3] {
        let (span_u, local_u) = locate(u, self.u_span_count());
        let (span_v, local_v) = locate(v, self.v_span_count());
        self.patch_at(span_u, span_v).point(local_u, local_v)
    }

    /// Evaluates the unit analytic surface normal at global `(u, v)`.
    #[must_use]
    pub fn normal(&self, u: f32, v: f32) -> [f32; 3] {
        let (span_u, local_u) = locate(u, self.u_span_count());
        let (span_v, local_v) = locate(v, self.v_span_count());
        self.patch_at(span_u, span_v).normal(local_u, local_v)
    }

    /// Returns the axis-aligned bounding box of every control point.
    ///
    /// Because a B-spline surface stays within the convex hull of its net, this
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
    /// span**.
    ///
    /// A single global `(nu + 1) × (nv + 1)` sample grid is built with
    /// `nu = u_span_count() * res_u` and `nv = v_span_count() * res_v`, where
    /// each resolution is clamped to at least `1`. Positions and exact analytic
    /// normals are sampled from [`Self::point`]/[`Self::normal`], and `UV`s are
    /// the global `(u, v)` parameters. Shared span edges land on identical grid
    /// vertices, so the mesh is watertight and smoothly shaded across spans.
    #[must_use]
    pub fn tessellate(&self, res_u: usize, res_v: usize) -> IndexedBilinearPatchMesh {
        let ru = res_u.max(1);
        let rv = res_v.max(1);
        let nu = self.u_span_count() * ru;
        let nv = self.v_span_count() * rv;
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
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// Component-wise difference, for test assertions.
    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    /// Euclidean length of a 3-vector.
    fn len(a: [f32; 3]) -> f32 {
        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
    }

    /// A planar `z = 0` grid of `rows × cols` with unit spacing, centred so the
    /// control net spans `[-1, cols - 2] × [-1, rows - 2]` in `xy`.
    fn flat_grid(rows: usize, cols: usize) -> Vec<[f32; 3]> {
        let mut control = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                control.push([col as f32 - 1.0, row as f32 - 1.0, 0.0]);
            }
        }
        control
    }

    /// A flat grid with every strictly-interior control point lifted in `+z`,
    /// so the surface bulges upward while the border stays at `z = 0`.
    fn domed_grid(rows: usize, cols: usize, height: f32) -> Vec<[f32; 3]> {
        let mut control = flat_grid(rows, cols);
        for row in 1..rows - 1 {
            for col in 1..cols - 1 {
                control[row * cols + col][2] = height;
            }
        }
        control
    }

    #[test]
    fn rejects_degenerate_grids() {
        assert_eq!(
            BsplineSurface::new(flat_grid(4, 3), 4, 3),
            Err(BsplineSurfaceError::TooFewColumns { cols: 3 })
        );
        assert_eq!(
            BsplineSurface::new(flat_grid(3, 4), 3, 4),
            Err(BsplineSurfaceError::TooFewRows { rows: 3 })
        );
        // Pool length that disagrees with rows * cols.
        let short = flat_grid(4, 4);
        assert_eq!(
            BsplineSurface::new(short, 4, 5),
            Err(BsplineSurfaceError::ControlCountMismatch {
                actual: 16,
                expected: 20,
            })
        );
    }

    #[test]
    fn span_counts_follow_grid_size() {
        let s = BsplineSurface::new(flat_grid(5, 7), 5, 7).unwrap();
        assert_eq!(s.u_span_count(), 4);
        assert_eq!(s.v_span_count(), 2);
    }

    #[test]
    fn flat_surface_stays_planar_with_up_normal() {
        let s = BsplineSurface::new(flat_grid(6, 5), 6, 5).unwrap();
        let mut rng = Rng::new(0xB59);
        for _ in 0..200 {
            let u = rng.range(0.0, 1.0);
            let v = rng.range(0.0, 1.0);
            let pt = s.point(u, v);
            assert!(pt[2].abs() < 1e-5, "z should be ~0, got {}", pt[2]);
            let n = s.normal(u, v);
            assert!((len(n) - 1.0).abs() < 1e-5);
            assert!(n[2].abs() > 1.0 - 1e-5);
            assert!(n[0].abs() < 1e-4 && n[1].abs() < 1e-4);
        }
    }

    #[test]
    fn minimal_grid_matches_single_patch() {
        // A 4x4 grid has exactly one span, so the surface must reproduce the
        // standalone BsplinePatch bit-for-bit over the whole domain.
        let mut rng = Rng::new(0x1234);
        let mut control = [[0.0f32; 3]; 16];
        for c in &mut control {
            *c = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(-1.0, 1.0)];
        }
        let patch = BsplinePatch::new(control);
        let surface = BsplineSurface::new(control.to_vec(), 4, 4).unwrap();
        for _ in 0..100 {
            let u = rng.range(0.0, 1.0);
            let v = rng.range(0.0, 1.0);
            assert_eq!(surface.point(u, v), patch.point(u, v));
            assert_eq!(surface.normal(u, v), patch.normal(u, v));
        }
    }

    #[test]
    fn interior_span_matches_its_patch() {
        // On a larger grid, a global parameter inside span (span_u, span_v)
        // must agree with that span's own BsplinePatch at the local parameter.
        let s = BsplineSurface::new(domed_grid(7, 6, 0.7), 7, 6).unwrap();
        let u_spans = s.u_span_count(); // 3
        let v_spans = s.v_span_count(); // 4
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..100 {
            let span_u = (rng.next_u32() as usize) % u_spans;
            let span_v = (rng.next_u32() as usize) % v_spans;
            let local_u = rng.range(0.05, 0.95);
            let local_v = rng.range(0.05, 0.95);
            let global_u = (span_u as f32 + local_u) / u_spans as f32;
            let global_v = (span_v as f32 + local_v) / v_spans as f32;
            let patch = s.patch_at(span_u, span_v);
            let want = patch.point(local_u, local_v);
            let got = s.point(global_u, global_v);
            assert!(len(sub(got, want)) < 1e-4, "got {got:?} want {want:?}");
        }
    }

    #[test]
    fn adjacent_spans_agree_on_shared_edge() {
        // C^2 continuity: the trailing edge of one span equals the leading edge
        // of the next, so sampling the global grid is watertight across spans.
        let s = BsplineSurface::new(domed_grid(6, 6, 0.6), 6, 6).unwrap();
        let u_spans = s.u_span_count(); // 3
        let v_spans = s.v_span_count(); // 3
        let seam = 1.0 / u_spans as f32; // boundary between u-span 0 and 1
        let mut rng = Rng::new(0x5EA);
        for _ in 0..64 {
            // Keep v strictly inside the first v-span so both the hand-built
            // patch edges and the global sampler resolve to v-span 0.
            let local_v = rng.range(0.0, 1.0);
            let left = s.patch_at(0, 0).point(1.0, local_v);
            let right = s.patch_at(1, 0).point(0.0, local_v);
            assert!(len(sub(left, right)) < 1e-4, "seam gap at local_v={local_v}");
            // The global sampler at the seam must agree with that shared edge.
            let global_v = local_v / v_spans as f32;
            let mid = s.point(seam, global_v);
            assert!(len(sub(mid, left)) < 1e-3, "sampler off seam at {global_v}");
        }
    }

    #[test]
    fn analytic_normal_matches_finite_difference() {
        let s = BsplineSurface::new(domed_grid(6, 7, 0.8), 6, 7).unwrap();
        let mut rng = Rng::new(0xD00D);
        let h = 1e-3_f32;
        for _ in 0..100 {
            // Stay strictly interior to avoid straddling a span boundary with
            // the finite-difference stencil.
            let u = rng.range(0.1, 0.9);
            let v = rng.range(0.1, 0.9);
            let du = sub(s.point(u + h, v), s.point(u - h, v));
            let dv = sub(s.point(u, v + h), s.point(u, v - h));
            let fd = [
                du[1] * dv[2] - du[2] * dv[1],
                du[2] * dv[0] - du[0] * dv[2],
                du[0] * dv[1] - du[1] * dv[0],
            ];
            let l = len(fd);
            assert!(l > 1e-6, "degenerate finite-difference normal");
            let fd_n = [fd[0] / l, fd[1] / l, fd[2] / l];
            let n = s.normal(u, v);
            let dot = n[0] * fd_n[0] + n[1] * fd_n[1] + n[2] * fd_n[2];
            assert!(dot > 0.999, "analytic vs FD normal dot = {dot}");
        }
    }

    #[test]
    fn tessellation_has_expected_shape() {
        let s = BsplineSurface::new(domed_grid(5, 6, 0.5), 5, 6).unwrap();
        let ru = 4;
        let rv = 3;
        let mesh = s.tessellate(ru, rv);
        let nu = s.u_span_count() * ru; // 3 * 4 = 12
        let nv = s.v_span_count() * rv; // 2 * 3 = 6
        assert_eq!(mesh.vertex_count(), (nu + 1) * (nv + 1));
        assert_eq!(mesh.patch_count(), nu * nv);
    }

    #[test]
    fn tessellated_bvh_hits_the_surface() {
        let s = BsplineSurface::new(domed_grid(6, 6, 0.5), 6, 6).unwrap();
        // Odd per-span resolution keeps the dome center strictly interior to a
        // quad instead of on a shared vertex (the quad mesh is not watertight
        // at vertices).
        let bvh = s.tessellate_bvh(15, 15);
        let center = s.point(0.5, 0.5);
        let origin = [center[0], center[1], center[2] + 5.0];
        let ray = Ray::infinite(origin, [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray must hit the dome");
        let hit_pt = ray.at(hit.t);
        assert!(len(sub(hit_pt, center)) < 0.05, "hit {hit_pt:?} vs {center:?}");
        assert!(bvh.any_hit(&ray));
        assert!(hit.shading_normal[2] > 0.3);
    }
}
