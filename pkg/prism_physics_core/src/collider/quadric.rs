//! Garland--Heckbert error quadrics for mesh simplification.
//!
//! A *quadric* is the symmetric `4x4` matrix `Q` that accumulates, for a vertex,
//! the sum of squared distances to the planes of the triangles that meet there.
//! Evaluated at a homogeneous point `v = (x, y, z, 1)`, the quadratic form
//! `v^T Q v` is exactly that summed squared plane distance, so it is the natural
//! cost to minimise when an edge collapse must choose where to place the merged
//! vertex. This is the quadric error metric (QEM) behind every production mesh
//! decimator used to cook collision LODs (`PhysX` cooking, `Jolt` mesh
//! reduction, Chaos `FMeshSimplifier`).
//!
//! The matrix is symmetric, so only its ten distinct coefficients are stored:
//!
//! ```text
//! [ a2 ab ac ad ]
//! [ ab b2 bc bd ]
//! [ ac bc c2 cd ]
//! [ ad bd cd d2 ]
//! ```
//!
//! Everything here is the standard QEM construction (Garland & Heckbert,
//! *Surface Simplification Using Quadric Error Metrics*, SIGGRAPH 1997); nothing
//! is derived from Unreal Engine source.

use glam::Vec3;

/// A symmetric `4x4` error quadric stored as its ten distinct coefficients.
///
/// Quadrics are additive: the quadric of a vertex is the sum of the plane
/// quadrics of its incident faces, and merging two vertices sums their
/// quadrics. [`Quadric::add`] / [`core::ops::Add`] provide that accumulation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Quadric {
    a2: f32,
    ab: f32,
    ac: f32,
    ad: f32,
    b2: f32,
    bc: f32,
    bd: f32,
    c2: f32,
    cd: f32,
    d2: f32,
}

impl Quadric {
    /// The zero quadric (adds nothing, reports zero error everywhere).
    pub const ZERO: Self = Self {
        a2: 0.0,
        ab: 0.0,
        ac: 0.0,
        ad: 0.0,
        b2: 0.0,
        bc: 0.0,
        bd: 0.0,
        c2: 0.0,
        cd: 0.0,
        d2: 0.0,
    };

    /// Builds the quadric of the plane `n . x + d = 0`, where `n` must be a unit
    /// normal. The resulting quadric reports the squared distance from any point
    /// to that plane.
    #[must_use]
    pub fn from_plane(n: Vec3, d: f32) -> Self {
        let (a, b, c) = (n.x, n.y, n.z);
        Self {
            a2: a * a,
            ab: a * b,
            ac: a * c,
            ad: a * d,
            b2: b * b,
            bc: b * c,
            bd: b * d,
            c2: c * c,
            cd: c * d,
            d2: d * d,
        }
    }

    /// Builds the plane quadric of the triangle `abc`. Returns [`Quadric::ZERO`]
    /// for a degenerate (zero-area) triangle, which contributes no constraint.
    #[must_use]
    pub fn from_triangle(a: Vec3, b: Vec3, c: Vec3) -> Self {
        let normal = (b - a).cross(c - a);
        let len = normal.length();
        if len <= f32::MIN_POSITIVE {
            return Self::ZERO;
        }
        let n = normal / len;
        Self::from_plane(n, -n.dot(a))
    }

    /// Sum of two quadrics (plane constraints accumulate additively).
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            a2: self.a2 + other.a2,
            ab: self.ab + other.ab,
            ac: self.ac + other.ac,
            ad: self.ad + other.ad,
            b2: self.b2 + other.b2,
            bc: self.bc + other.bc,
            bd: self.bd + other.bd,
            c2: self.c2 + other.c2,
            cd: self.cd + other.cd,
            d2: self.d2 + other.d2,
        }
    }

    /// The quadratic form `v^T Q v`: the summed squared distance from `v` to the
    /// accumulated planes. Clamped at zero so floating-point round-off never
    /// yields a spuriously negative cost.
    #[must_use]
    pub fn error(&self, v: Vec3) -> f32 {
        let (x, y, z) = (v.x, v.y, v.z);
        let e = self.a2 * x * x
            + 2.0 * self.ab * x * y
            + 2.0 * self.ac * x * z
            + 2.0 * self.ad * x
            + self.b2 * y * y
            + 2.0 * self.bc * y * z
            + 2.0 * self.bd * y
            + self.c2 * z * z
            + 2.0 * self.cd * z
            + self.d2;
        e.max(0.0)
    }

    /// Solves for the point that minimises [`Quadric::error`], i.e. the vertex
    /// position an edge collapse should use. Returns `None` when the upper-left
    /// `3x3` block is singular (the planes do not pin a unique point), leaving
    /// the caller to fall back to an endpoint or midpoint.
    #[must_use]
    pub fn optimal_point(&self) -> Option<Vec3> {
        // Minimise v^T Q v over (x, y, z): gradient = 0 gives the 3x3 system
        //   [a2 ab ac] [x]   [-ad]
        //   [ab b2 bc] [y] = [-bd]
        //   [ac bc c2] [z]   [-cd]
        let m00 = self.a2;
        let m01 = self.ab;
        let m02 = self.ac;
        let m11 = self.b2;
        let m12 = self.bc;
        let m22 = self.c2;

        // Cofactors of the symmetric matrix.
        let c00 = m11 * m22 - m12 * m12;
        let c01 = m02 * m12 - m01 * m22;
        let c02 = m01 * m12 - m02 * m11;
        let det = m00 * c00 + m01 * c01 + m02 * c02;
        if det.abs() <= 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;

        let c11 = m00 * m22 - m02 * m02;
        let c12 = m02 * m01 - m00 * m12;
        let c22 = m00 * m11 - m01 * m01;

        // rhs = -(ad, bd, cd).
        let r0 = -self.ad;
        let r1 = -self.bd;
        let r2 = -self.cd;

        let x = (c00 * r0 + c01 * r1 + c02 * r2) * inv_det;
        let y = (c01 * r0 + c11 * r1 + c12 * r2) * inv_det;
        let z = (c02 * r0 + c12 * r1 + c22 * r2) * inv_det;
        let p = Vec3::new(x, y, z);
        if p.is_finite() {
            Some(p)
        } else {
            None
        }
    }
}

impl core::ops::Add for Quadric {
    type Output = Quadric;

    fn add(self, rhs: Quadric) -> Quadric {
        Quadric::add(&self, &rhs)
    }
}

impl core::ops::AddAssign for Quadric {
    fn add_assign(&mut self, rhs: Quadric) {
        *self = Quadric::add(self, &rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_quadric_reports_squared_distance() {
        // Plane z = 0 => n = (0,0,1), d = 0.
        let q = Quadric::from_plane(Vec3::Z, 0.0);
        assert!((q.error(Vec3::new(3.0, -2.0, 0.0))).abs() < 1e-6);
        // A point at height 2 is squared-distance 4 from the plane.
        assert!((q.error(Vec3::new(1.0, 1.0, 2.0)) - 4.0).abs() < 1e-5);
    }

    #[test]
    fn triangle_quadric_matches_its_plane() {
        let a = Vec3::new(0.0, 0.0, 1.0);
        let b = Vec3::new(1.0, 0.0, 1.0);
        let c = Vec3::new(0.0, 1.0, 1.0);
        let q = Quadric::from_triangle(a, b, c);
        // On the plane z = 1: zero error at all three vertices.
        for v in [a, b, c] {
            assert!(q.error(v) < 1e-6);
        }
        // One unit above the plane: squared distance 1.
        assert!((q.error(Vec3::new(0.3, 0.2, 2.0)) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn degenerate_triangle_is_zero() {
        let q = Quadric::from_triangle(Vec3::ZERO, Vec3::ZERO, Vec3::X);
        assert_eq!(q, Quadric::ZERO);
    }

    #[test]
    fn quadrics_sum() {
        let q1 = Quadric::from_plane(Vec3::X, 0.0); // plane x = 0
        let q2 = Quadric::from_plane(Vec3::Y, 0.0); // plane y = 0
        let sum = q1 + q2;
        // At (3, 4, 0): squared distances 9 + 16 = 25.
        assert!((sum.error(Vec3::new(3.0, 4.0, 0.0)) - 25.0).abs() < 1e-4);
    }

    #[test]
    fn optimal_point_sits_at_plane_intersection() {
        // Three orthogonal planes x=1, y=2, z=3 pin the point (1, 2, 3).
        let q = Quadric::from_plane(Vec3::X, -1.0)
            + Quadric::from_plane(Vec3::Y, -2.0)
            + Quadric::from_plane(Vec3::Z, -3.0);
        let p = q.optimal_point().expect("non-singular");
        assert!((p - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-4);
        assert!(q.error(p) < 1e-4);
    }

    #[test]
    fn optimal_point_is_none_when_singular() {
        // A single plane does not pin a unique minimiser.
        let q = Quadric::from_plane(Vec3::Z, 0.0);
        assert!(q.optimal_point().is_none());
    }

    #[test]
    fn error_is_never_negative() {
        let q = Quadric::from_plane(Vec3::new(0.6, 0.0, 0.8), -1.5);
        for p in [Vec3::ZERO, Vec3::splat(10.0), Vec3::new(-4.0, 2.0, -7.0)] {
            assert!(q.error(p) >= 0.0);
        }
    }
}
