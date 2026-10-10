//! Tensor-product spline surfaces: bicubic Bézier and uniform cubic B-spline.
//!
//! A tensor-product surface is the 2D analogue of the cubic segments in
//! [`crate::curve::spline`]: a 4x4 grid of control points is evaluated along
//! one parameter to collapse each row to a single point, then those four
//! points are evaluated along the other parameter. The result `S(u, v)` is a
//! smooth patch suitable for terrain detail, procedural geometry, path
//! corridors, and vehicle motion surfaces.
//!
//! Both patches expose `sample`, the partial derivatives `tangent_u` /
//! `tangent_v`, and `normal` (their normalized cross product). [`BezierPatch`]
//! interpolates its corner control points and gives direct tangent control;
//! [`BSplineSurface`] trades interpolation for `C2` continuity across tiled
//! patches, which is what large procedural surfaces usually want.

use crate::curve::spline::{bezier_cubic, bezier_cubic_tangent};
use crate::curve::Interpolatable;
use crate::vec::Vec3;

/// Uniform cubic B-spline basis evaluation for four control points.
///
/// Unlike Bézier, the curve does not pass through the control points; it stays
/// within their convex hull and is `C2`-continuous when segments are tiled.
#[inline]
fn bspline_cubic<T: Interpolatable>(p0: T, p1: T, p2: T, p3: T, t: f32) -> T {
    let t2 = t * t;
    let t3 = t2 * t;
    let b0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;
    let b1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
    let b2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;
    let b3 = t3 / 6.0;
    p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3
}

/// Derivative of [`bspline_cubic`] with respect to `t`.
#[inline]
fn bspline_cubic_tangent<T: Interpolatable>(p0: T, p1: T, p2: T, p3: T, t: f32) -> T {
    let t2 = t * t;
    let b0 = (-3.0 + 6.0 * t - 3.0 * t2) / 6.0;
    let b1 = (-12.0 * t + 9.0 * t2) / 6.0;
    let b2 = (3.0 + 6.0 * t - 9.0 * t2) / 6.0;
    let b3 = (3.0 * t2) / 6.0;
    p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3
}

/// A bicubic Bézier patch: a 4x4 grid of control points.
///
/// `p[i][j]` is the control point at row `i` (the `u` direction) and column
/// `j` (the `v` direction). The patch interpolates its four corner control
/// points `p[0][0]`, `p[0][3]`, `p[3][0]`, `p[3][3]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BezierPatch {
    /// Control grid indexed as `[u_row][v_col]`.
    pub p: [[Vec3; 4]; 4],
}

impl BezierPatch {
    /// Build a patch from a 4x4 control grid.
    #[inline]
    #[must_use]
    pub const fn new(p: [[Vec3; 4]; 4]) -> Self {
        Self { p }
    }

    /// Evaluate the surface point `S(u, v)` for `u, v` in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn sample(&self, u: f32, v: f32) -> Vec3 {
        let q = self.rows_along_v(v, bezier_cubic);
        bezier_cubic(q[0], q[1], q[2], q[3], u)
    }

    /// Partial derivative `dS/du` at `(u, v)`.
    #[inline]
    #[must_use]
    pub fn tangent_u(&self, u: f32, v: f32) -> Vec3 {
        let q = self.rows_along_v(v, bezier_cubic);
        bezier_cubic_tangent(q[0], q[1], q[2], q[3], u)
    }

    /// Partial derivative `dS/dv` at `(u, v)`.
    #[inline]
    #[must_use]
    pub fn tangent_v(&self, u: f32, v: f32) -> Vec3 {
        let q = self.rows_along_v(v, bezier_cubic_tangent);
        bezier_cubic(q[0], q[1], q[2], q[3], u)
    }

    /// Unit surface normal `normalize(dS/du x dS/dv)` at `(u, v)`.
    ///
    /// Returns the zero vector at degenerate points where the tangents are
    /// parallel or vanish.
    #[inline]
    #[must_use]
    pub fn normal(&self, u: f32, v: f32) -> Vec3 {
        self.tangent_u(u, v)
            .cross(self.tangent_v(u, v))
            .normalize_or_zero()
    }

    /// Collapse each `u`-row to one point by applying `basis` along `v`.
    #[inline]
    fn rows_along_v(&self, v: f32, basis: fn(Vec3, Vec3, Vec3, Vec3, f32) -> Vec3) -> [Vec3; 4] {
        [
            basis(self.p[0][0], self.p[0][1], self.p[0][2], self.p[0][3], v),
            basis(self.p[1][0], self.p[1][1], self.p[1][2], self.p[1][3], v),
            basis(self.p[2][0], self.p[2][1], self.p[2][2], self.p[2][3], v),
            basis(self.p[3][0], self.p[3][1], self.p[3][2], self.p[3][3], v),
        ]
    }
}

/// A uniform bicubic B-spline surface patch over a 4x4 control grid.
///
/// `C2`-continuous and contained in the convex hull of the control points;
/// it does not interpolate them. Tiling patches that share three columns of
/// control points yields a seamless `C2` surface, which is the usual choice
/// for large procedural terrain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BSplineSurface {
    /// Control grid indexed as `[u_row][v_col]`.
    pub p: [[Vec3; 4]; 4],
}

impl BSplineSurface {
    /// Build a patch from a 4x4 control grid.
    #[inline]
    #[must_use]
    pub const fn new(p: [[Vec3; 4]; 4]) -> Self {
        Self { p }
    }

    /// Evaluate the surface point `S(u, v)` for `u, v` in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn sample(&self, u: f32, v: f32) -> Vec3 {
        let q = self.rows_along_v(v, bspline_cubic);
        bspline_cubic(q[0], q[1], q[2], q[3], u)
    }

    /// Partial derivative `dS/du` at `(u, v)`.
    #[inline]
    #[must_use]
    pub fn tangent_u(&self, u: f32, v: f32) -> Vec3 {
        let q = self.rows_along_v(v, bspline_cubic);
        bspline_cubic_tangent(q[0], q[1], q[2], q[3], u)
    }

    /// Partial derivative `dS/dv` at `(u, v)`.
    #[inline]
    #[must_use]
    pub fn tangent_v(&self, u: f32, v: f32) -> Vec3 {
        let q = self.rows_along_v(v, bspline_cubic_tangent);
        bspline_cubic(q[0], q[1], q[2], q[3], u)
    }

    /// Unit surface normal `normalize(dS/du x dS/dv)` at `(u, v)`.
    #[inline]
    #[must_use]
    pub fn normal(&self, u: f32, v: f32) -> Vec3 {
        self.tangent_u(u, v)
            .cross(self.tangent_v(u, v))
            .normalize_or_zero()
    }

    /// Collapse each `u`-row to one point by applying `basis` along `v`.
    #[inline]
    fn rows_along_v(&self, v: f32, basis: fn(Vec3, Vec3, Vec3, Vec3, f32) -> Vec3) -> [Vec3; 4] {
        [
            basis(self.p[0][0], self.p[0][1], self.p[0][2], self.p[0][3], v),
            basis(self.p[1][0], self.p[1][1], self.p[1][2], self.p[1][3], v),
            basis(self.p[2][0], self.p[2][1], self.p[2][2], self.p[2][3], v),
            basis(self.p[3][0], self.p[3][1], self.p[3][2], self.p[3][3], v),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A planar control grid: z is an affine function f(x, y) = 2x - 3y + 1,
    /// so any correct patch must reproduce that plane exactly everywhere.
    fn planar_grid() -> [[Vec3; 4]; 4] {
        let mut g = [[Vec3::ZERO; 4]; 4];
        for (i, row) in g.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let x = i as f32 / 3.0;
                let y = j as f32 / 3.0;
                *cell = Vec3::new(x, y, 2.0 * x - 3.0 * y + 1.0);
            }
        }
        g
    }

    #[test]
    fn bezier_interpolates_corners() {
        let g = planar_grid();
        let patch = BezierPatch::new(g);
        assert!((patch.sample(0.0, 0.0) - g[0][0]).length() < 1e-5);
        assert!((patch.sample(1.0, 0.0) - g[3][0]).length() < 1e-5);
        assert!((patch.sample(0.0, 1.0) - g[0][3]).length() < 1e-5);
        assert!((patch.sample(1.0, 1.0) - g[3][3]).length() < 1e-5);
    }

    #[test]
    fn bezier_reproduces_a_plane() {
        // A patch over a planar grid is that plane; its normal is constant.
        let patch = BezierPatch::new(planar_grid());
        let expected_n = Vec3::new(2.0, -3.0, -1.0).normalize();
        for &(u, v) in &[(0.1, 0.2), (0.5, 0.5), (0.9, 0.3), (0.25, 0.75)] {
            let s = patch.sample(u, v);
            assert!((s.z - (2.0 * s.x - 3.0 * s.y + 1.0)).abs() < 1e-4, "plane");
            let n = patch.normal(u, v);
            // Normal may point either way; compare absolute alignment.
            assert!(n.dot(expected_n).abs() > 0.999, "normal u={u} v={v}");
        }
    }

    #[test]
    fn bezier_tangents_match_finite_difference() {
        let patch = BezierPatch::new(planar_grid());
        let h = 1e-3;
        for &(u, v) in &[(0.3, 0.4), (0.6, 0.2)] {
            let fd_u = (patch.sample(u + h, v) - patch.sample(u - h, v)) * (1.0 / (2.0 * h));
            let fd_v = (patch.sample(u, v + h) - patch.sample(u, v - h)) * (1.0 / (2.0 * h));
            assert!((patch.tangent_u(u, v) - fd_u).length() < 1e-2, "du");
            assert!((patch.tangent_v(u, v) - fd_v).length() < 1e-2, "dv");
        }
    }

    #[test]
    fn bspline_stays_in_convex_hull_and_tangents_are_consistent() {
        let patch = BSplineSurface::new(planar_grid());
        let h = 1e-3;
        for &(u, v) in &[(0.2, 0.5), (0.7, 0.8), (0.5, 0.1)] {
            // Planar control grid -> planar B-spline surface too.
            let s = patch.sample(u, v);
            assert!((s.z - (2.0 * s.x - 3.0 * s.y + 1.0)).abs() < 1e-4, "plane");
            let fd_u = (patch.sample(u + h, v) - patch.sample(u - h, v)) * (1.0 / (2.0 * h));
            let fd_v = (patch.sample(u, v + h) - patch.sample(u, v - h)) * (1.0 / (2.0 * h));
            assert!((patch.tangent_u(u, v) - fd_u).length() < 1e-2, "du");
            assert!((patch.tangent_v(u, v) - fd_v).length() < 1e-2, "dv");
        }
    }

    #[test]
    fn bspline_basis_partitions_unity() {
        // The four basis weights sum to 1 at every t (affine invariance).
        for k in 0..=10 {
            let t = k as f32 / 10.0;
            let sum = bspline_cubic(1.0f32, 1.0, 1.0, 1.0, t);
            assert!((sum - 1.0).abs() < 1e-6, "t={t}");
            let dsum = bspline_cubic_tangent(1.0f32, 1.0, 1.0, 1.0, t);
            assert!(dsum.abs() < 1e-6, "derivative of constant t={t}");
        }
    }

    #[test]
    fn curved_patch_has_nonzero_curvature() {
        // Lift the center control points to create a bump; the surface should
        // bulge above the corner plane at the center.
        let mut g = planar_grid();
        g[1][1].z += 1.0;
        g[1][2].z += 1.0;
        g[2][1].z += 1.0;
        g[2][2].z += 1.0;
        let patch = BezierPatch::new(g);
        let center = patch.sample(0.5, 0.5);
        let plane_z = 2.0 * center.x - 3.0 * center.y + 1.0;
        assert!(center.z > plane_z + 0.1, "center should bulge up");
    }
}
