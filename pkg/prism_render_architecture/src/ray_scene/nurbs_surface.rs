//! Uniform **rational** bicubic B-spline (`NURBS`) surface over an arbitrary
//! `R × C` control grid with per-control-point weights, tessellated into a
//! single welded [`IndexedBilinearPatchMesh`] for the `CPU` golden path.
//!
//! This is the rational, weighted generalisation of
//! [`super::bspline_surface::BsplineSurface`] and the authored counterpart to
//! the single-span [`super::rational_bezier_patch::RationalBezierPatch`]. An
//! `R × C` control net plus matching positive weights is partitioned into
//! `(C - 3) × (R - 3)` overlapping cubic spans that share three control
//! rows/columns. Each span is converted to a
//! [`super::rational_bezier_patch::RationalBezierPatch`] and the whole surface
//! is sampled on one global grid, so adjacent spans weld with no cracks.
//!
//! The conversion runs entirely in homogeneous `[w·x, w·y, w·z, w]` space:
//! each control point is promoted to its weighted homogeneous coordinate, the
//! standard uniform B-spline → Bézier span conversion (`b0 = (c0 + 4·c1 + c2)/6`,
//! …) is applied along `u` then `v` as a purely linear combination of those
//! 4-vectors, and the resulting homogeneous Bézier net is projected back by a
//! single division per control point. Because every conversion coefficient is a
//! positive convex combination, the projected weights stay positive and the
//! surface stays inside the convex hull of its net. With all weights equal the
//! surface reproduces the polynomial [`super::bspline_surface::BsplineSurface`]
//! exactly, while unequal weights bend it toward the heavier handles and let it
//! represent conics (spherical caps, cylinders, swept arcs) exactly.

use super::bvh::Aabb;
use super::indexed_bilinear_patch_mesh::{IndexedBilinearPatchMesh, IndexedBilinearPatchMeshBvh};
use super::rational_bezier_patch::RationalBezierPatch;

/// Why a [`NurbsSurface`] could not be constructed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NurbsSurfaceError {
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
    /// The weight pool length did not equal the control pool length.
    WeightCountMismatch {
        /// Number of weights actually supplied.
        weights: usize,
        /// Number of control points (and expected weights).
        control: usize,
    },
    /// A weight was not strictly positive (zero, negative, or `NaN`), which
    /// would break the rational projection and the convex-hull guarantee.
    NonPositiveWeight {
        /// Index of the offending weight.
        index: usize,
        /// The offending value.
        value: f32,
    },
}

impl core::fmt::Display for NurbsSurfaceError {
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
            Self::WeightCountMismatch { weights, control } => write!(
                f,
                "weight count {weights} does not match control-point count {control}"
            ),
            Self::NonPositiveWeight { index, value } => {
                write!(f, "weight {index} must be strictly positive, got {value}")
            }
        }
    }
}

impl std::error::Error for NurbsSurfaceError {}

/// Component-wise sum of two homogeneous 4-vectors.
fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// Scales a homogeneous 4-vector by the scalar `s`.
fn scale4(a: [f32; 4], s: f32) -> [f32; 4] {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

/// Converts one uniform cubic B-spline span `[c0, c1, c2, c3]` of homogeneous
/// 4-vectors into the four cubic Bézier control points of its central segment.
///
/// Uses the standard uniform-knot conversion `b0 = (c0 + 4·c1 + c2)/6`,
/// `b1 = (2·c1 + c2)/3`, `b2 = (c1 + 2·c2)/3`, `b3 = (c1 + 4·c2 + c3)/6` applied
/// coordinate-wise to the homogeneous vectors. Each coefficient is a convex
/// combination, so a strictly-positive `w` component stays strictly positive.
fn span_to_bezier4(c0: [f32; 4], c1: [f32; 4], c2: [f32; 4], c3: [f32; 4]) -> [[f32; 4]; 4] {
    let sixth = 1.0 / 6.0;
    let third = 1.0 / 3.0;
    let b0 = scale4(add4(add4(c0, scale4(c1, 4.0)), c2), sixth);
    let b1 = scale4(add4(scale4(c1, 2.0), c2), third);
    let b2 = scale4(add4(c1, scale4(c2, 2.0)), third);
    let b3 = scale4(add4(add4(c1, scale4(c2, 4.0)), c3), sixth);
    [b0, b1, b2, b3]
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

/// A uniform rational bicubic B-spline (`NURBS`) surface defined by an `R × C`
/// control grid and matching positive weights.
///
/// Both pools are stored row-major (`control[row * cols + col]`), with `col`
/// advancing along `u` and `row` advancing along `v`, matching
/// [`super::rational_bezier_patch::RationalBezierPatch`]. The surface stays
/// inside the convex hull of the net and interpolates no control point.
#[derive(Clone, Debug, PartialEq)]
pub struct NurbsSurface {
    /// Row-major control grid (`col` along `u`, `row` along `v`), length
    /// `rows * cols`.
    control: Vec<[f32; 3]>,
    /// Row-major per-control-point weights, index-aligned with `control` and
    /// all strictly positive.
    weights: Vec<f32>,
    /// Number of control rows (along `v`); always `>= 4`.
    rows: usize,
    /// Number of control columns (along `u`); always `>= 4`.
    cols: usize,
}

impl NurbsSurface {
    /// Builds a surface from a row-major `rows × cols` control grid and matching
    /// weights.
    ///
    /// Returns [`NurbsSurfaceError`] when `cols < 4`, `rows < 4`, the control
    /// pool length is not `rows * cols`, the weight pool length does not match
    /// the control pool, or any weight is not strictly positive.
    pub fn new(
        control: Vec<[f32; 3]>,
        weights: Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, NurbsSurfaceError> {
        if cols < 4 {
            return Err(NurbsSurfaceError::TooFewColumns { cols });
        }
        if rows < 4 {
            return Err(NurbsSurfaceError::TooFewRows { rows });
        }
        let expected = rows * cols;
        if control.len() != expected {
            return Err(NurbsSurfaceError::ControlCountMismatch {
                actual: control.len(),
                expected,
            });
        }
        if weights.len() != control.len() {
            return Err(NurbsSurfaceError::WeightCountMismatch {
                weights: weights.len(),
                control: control.len(),
            });
        }
        for (index, &value) in weights.iter().enumerate() {
            if value <= 0.0 || value.is_nan() {
                return Err(NurbsSurfaceError::NonPositiveWeight { index, value });
            }
        }
        Ok(Self {
            control,
            weights,
            rows,
            cols,
        })
    }

    /// Returns the control grid as a flat row-major slice.
    #[must_use]
    pub fn control(&self) -> &[[f32; 3]] {
        &self.control
    }

    /// Returns the weights as a flat row-major slice, index-aligned with
    /// [`Self::control`].
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
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

    /// Converts the span at column `span_u` and row `span_v` into its
    /// equivalent [`super::rational_bezier_patch::RationalBezierPatch`].
    ///
    /// The 4×4 control/weight window is promoted to homogeneous `[w·x, w·y,
    /// w·z, w]` coordinates, converted B-spline → Bézier along `u` then `v`
    /// with [`span_to_bezier4`], and projected back by one division per
    /// control point. The caller guarantees `span_u < u_span_count()` and
    /// `span_v < v_span_count()`.
    #[must_use]
    pub fn patch_at(&self, span_u: usize, span_v: usize) -> RationalBezierPatch {
        // Promote the 4×4 window to weighted homogeneous coordinates.
        let mut hwin = [[0.0f32; 4]; 16];
        for wr in 0..4 {
            for wc in 0..4 {
                let idx = (span_v + wr) * self.cols + (span_u + wc);
                let p = self.control[idx];
                let w = self.weights[idx];
                hwin[wr * 4 + wc] = [p[0] * w, p[1] * w, p[2] * w, w];
            }
        }
        // Pass 1: convert each row along u.
        let mut tmp = [[0.0f32; 4]; 16];
        for row in 0..4 {
            let base = row * 4;
            let span = span_to_bezier4(hwin[base], hwin[base + 1], hwin[base + 2], hwin[base + 3]);
            tmp[base] = span[0];
            tmp[base + 1] = span[1];
            tmp[base + 2] = span[2];
            tmp[base + 3] = span[3];
        }
        // Pass 2: convert each column along v.
        let mut out = [[0.0f32; 4]; 16];
        for col in 0..4 {
            let span = span_to_bezier4(tmp[col], tmp[4 + col], tmp[8 + col], tmp[12 + col]);
            out[col] = span[0];
            out[4 + col] = span[1];
            out[8 + col] = span[2];
            out[12 + col] = span[3];
        }
        // Project the homogeneous Bézier net back to control + weights.
        let mut control = [[0.0f32; 3]; 16];
        let mut weights = [0.0f32; 16];
        for k in 0..16 {
            let h = out[k];
            let w = h[3];
            control[k] = [h[0] / w, h[1] / w, h[2] / w];
            weights[k] = w;
        }
        RationalBezierPatch::new(control, weights)
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
    /// Because a rational B-spline surface with positive weights stays within
    /// the convex hull of its net, this control-point box is also a valid
    /// (loose) bound on the surface; traversal still uses the tessellated
    /// mesh's own `BVH` for exact hits.
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
    /// normals are sampled from [`Self::point`]/[`Self::normal`] and `UV`s are
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
    use crate::ray_scene::bspline_surface::BsplineSurface;
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

    /// A flat grid with every strictly-interior control point lifted in `+z`.
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
            NurbsSurface::new(flat_grid(4, 3), vec![1.0; 12], 4, 3),
            Err(NurbsSurfaceError::TooFewColumns { cols: 3 })
        );
        assert_eq!(
            NurbsSurface::new(flat_grid(3, 4), vec![1.0; 12], 3, 4),
            Err(NurbsSurfaceError::TooFewRows { rows: 3 })
        );
        assert_eq!(
            NurbsSurface::new(flat_grid(4, 4), vec![1.0; 16], 4, 5),
            Err(NurbsSurfaceError::ControlCountMismatch {
                actual: 16,
                expected: 20,
            })
        );
        assert_eq!(
            NurbsSurface::new(flat_grid(4, 4), vec![1.0; 15], 4, 4),
            Err(NurbsSurfaceError::WeightCountMismatch {
                weights: 15,
                control: 16,
            })
        );
        let mut bad = vec![1.0; 16];
        bad[7] = 0.0;
        assert_eq!(
            NurbsSurface::new(flat_grid(4, 4), bad, 4, 4),
            Err(NurbsSurfaceError::NonPositiveWeight {
                index: 7,
                value: 0.0,
            })
        );
    }

    #[test]
    fn unit_weights_match_polynomial_bspline_surface() {
        // All weights == 1 ⇒ the rational surface collapses to the polynomial
        // B-spline surface over the same net.
        let control = domed_grid(6, 7, 0.8);
        let nurbs = NurbsSurface::new(control.clone(), vec![1.0; 42], 6, 7).unwrap();
        let bspline = BsplineSurface::new(control, 6, 7).unwrap();
        let mut rng = Rng::new(0xABCD);
        for _ in 0..200 {
            let u = rng.range(0.0, 1.0);
            let v = rng.range(0.0, 1.0);
            assert!(len(sub(nurbs.point(u, v), bspline.point(u, v))) < 1e-5);
            let dot = {
                let a = nurbs.normal(u, v);
                let b = bspline.normal(u, v);
                a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
            };
            assert!(dot > 0.9999, "normal mismatch dot={dot}");
        }
    }

    #[test]
    fn equal_nonunit_weights_also_match_polynomial() {
        // A constant non-unit weight cancels in the rational quotient, so the
        // surface is still the polynomial B-spline surface.
        let control = domed_grid(5, 6, 0.6);
        let nurbs = NurbsSurface::new(control.clone(), vec![3.5; 30], 5, 6).unwrap();
        let bspline = BsplineSurface::new(control, 5, 6).unwrap();
        let mut rng = Rng::new(0x7777);
        for _ in 0..100 {
            let u = rng.range(0.0, 1.0);
            let v = rng.range(0.0, 1.0);
            assert!(len(sub(nurbs.point(u, v), bspline.point(u, v))) < 1e-4);
        }
    }

    #[test]
    fn interior_span_matches_its_patch() {
        let control = domed_grid(7, 6, 0.7);
        let mut weights = vec![1.0; 42];
        weights[3 * 6 + 3] = 4.0; // bias one interior handle
        let s = NurbsSurface::new(control, weights, 7, 6).unwrap();
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
            let want = s.patch_at(span_u, span_v).point(local_u, local_v);
            let got = s.point(global_u, global_v);
            assert!(len(sub(got, want)) < 1e-4, "got {got:?} want {want:?}");
        }
    }

    /// Peak sampled `z` over a dense global parameter grid.
    fn peak_z(s: &NurbsSurface) -> f32 {
        let mut peak = f32::MIN;
        for j in 0..=40 {
            let v = j as f32 / 40.0;
            for i in 0..=40 {
                let u = i as f32 / 40.0;
                let z = s.point(u, v)[2];
                if z > peak {
                    peak = z;
                }
            }
        }
        peak
    }

    #[test]
    fn heavier_interior_weight_pulls_surface_up() {
        // Lift exactly one interior handle to z = 2; a larger weight on that
        // single handle must pull the sampled surface closer to it, i.e. raise
        // the surface peak, while staying under the handle height (convex hull).
        let mut control = flat_grid(6, 6);
        let peak_idx = 2 * 6 + 2;
        control[peak_idx][2] = 2.0;
        let light = NurbsSurface::new(control.clone(), vec![1.0; 36], 6, 6).unwrap();
        let mut heavy_w = vec![1.0; 36];
        heavy_w[peak_idx] = 8.0;
        let heavy = NurbsSurface::new(control, heavy_w, 6, 6).unwrap();
        let light_peak = peak_z(&light);
        let heavy_peak = peak_z(&heavy);
        assert!(
            heavy_peak > light_peak + 0.1,
            "heavy weight should raise the peak: light={light_peak} heavy={heavy_peak}"
        );
        // The surface still stays below the handle height (convex hull).
        assert!(heavy_peak < 2.0 + 1e-4, "peak {heavy_peak} broke convex hull");
    }

    #[test]
    fn analytic_normal_matches_finite_difference() {
        let control = domed_grid(6, 7, 0.8);
        let mut weights = vec![1.0; 42];
        let mut seed = Rng::new(0x51A7E);
        for w in &mut weights {
            *w = seed.range(0.5, 3.0);
        }
        let s = NurbsSurface::new(control, weights, 6, 7).unwrap();
        let mut rng = Rng::new(0xD00D);
        let h = 1e-3_f32;
        for _ in 0..100 {
            // Stay strictly interior to a span to avoid straddling a boundary.
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
            assert!(dot > 0.998, "analytic vs FD normal dot = {dot}");
        }
    }

    #[test]
    fn tessellation_has_expected_shape() {
        let s = NurbsSurface::new(domed_grid(5, 6, 0.5), vec![1.0; 30], 5, 6).unwrap();
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
        let mut weights = vec![1.0; 36];
        for row in 1..5 {
            for col in 1..5 {
                weights[row * 6 + col] = 3.0;
            }
        }
        let s = NurbsSurface::new(domed_grid(6, 6, 0.6), weights, 6, 6).unwrap();
        // Odd per-span resolution keeps the dome center strictly interior to a
        // quad instead of on a shared vertex.
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
