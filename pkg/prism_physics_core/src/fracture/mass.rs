//! Rigid-body mass properties (volume, mass, centroid, inertia tensor) of a
//! convex fragment.
//!
//! When a fractured piece becomes a dynamic rigid body it needs a full inertia
//! tensor, not just a mass. [`MassProperties::from_polyhedron`] integrates the
//! solid exactly by decomposing it into origin-based tetrahedra (one per face
//! triangle) and summing the analytic second-moment integral of each, then
//! shifting to the centre of mass.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! tetrahedral covariance integral (with the canonical `[1/60, 1/120]`
//! second-moment matrix) and the `I = tr(C) Id - C` conversion are standard,
//! publicly documented rigid-body results.

use glam::{Mat3, Vec3};

use crate::fracture::polyhedron::ConvexPolyhedron;
use crate::math::scalar::Real;

/// Full rigid-body mass properties of a uniform-density solid.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MassProperties {
    /// Enclosed volume in cubic world units.
    pub volume: Real,
    /// Total mass (`density * volume`).
    pub mass: Real,
    /// Centre of mass in world coordinates.
    pub centroid: Vec3,
    /// Inertia tensor about the centre of mass.
    pub inertia: Mat3,
}

impl MassProperties {
    /// Computes the mass properties of `poly` filled at uniform `density`.
    ///
    /// A degenerate (zero-volume) polyhedron yields zero mass and inertia with
    /// the vertex-average centroid, avoiding division by zero.
    #[must_use]
    pub fn from_polyhedron(poly: &ConvexPolyhedron, density: Real) -> MassProperties {
        let mut v6: Real = 0.0;
        let mut m1 = Vec3::ZERO;
        let mut cov = Mat3::ZERO;

        for [a, b, c] in poly.face_triangles() {
            let det = a.dot(b.cross(c));
            v6 += det;
            m1 += det * (a + b + c);
            cov += det * canonical_second_moment(a, b, c);
        }

        // Enforce positive orientation so covariance signs stay consistent.
        if v6 < 0.0 {
            v6 = -v6;
            m1 = -m1;
            cov *= -1.0;
        }

        if v6 <= Real::EPSILON {
            return MassProperties {
                volume: 0.0,
                mass: 0.0,
                centroid: poly.centroid(),
                inertia: Mat3::ZERO,
            };
        }

        let volume = v6 / 6.0;
        let centroid = m1 / (4.0 * v6);

        // Shift covariance from the origin to the centre of mass.
        let cov_c = cov - volume * outer(centroid, centroid);
        let inertia = (trace(cov_c) * Mat3::IDENTITY - cov_c) * density;

        MassProperties {
            volume,
            mass: density * volume,
            centroid,
            inertia,
        }
    }
}

/// Returns `detJ * J S Jᵀ` for the tetrahedron `(0, a, b, c)`, i.e. the
/// contribution of that tetrahedron to the origin-referenced covariance
/// integral `∫ x xᵀ dV` (with `S` the canonical second-moment matrix that has
/// `1/60` on the diagonal and `1/120` off-diagonal). The caller multiplies by
/// the signed determinant, so this returns only the `J S Jᵀ` part.
fn canonical_second_moment(a: Vec3, b: Vec3, c: Vec3) -> Mat3 {
    let diag = (outer(a, a) + outer(b, b) + outer(c, c)) * (1.0 / 60.0);
    let off = (outer(a, b) + outer(b, a) + outer(a, c) + outer(c, a) + outer(b, c) + outer(c, b))
        * (1.0 / 120.0);
    diag + off
}

/// Outer product `a bᵀ` as a 3x3 matrix (column `j` is `a * b[j]`).
fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Trace of a 3x3 matrix.
fn trace(m: Mat3) -> Real {
    m.x_axis.x + m.y_axis.y + m.z_axis.z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_matches_analytic_inertia() {
        // Side-2 cube, unit density -> mass 8, I = m s^2 / 6 on the diagonal.
        let cube = ConvexPolyhedron::box_aabb(Vec3::splat(-1.0), Vec3::splat(1.0));
        let mp = MassProperties::from_polyhedron(&cube, 1.0);
        assert!((mp.volume - 8.0).abs() < 1e-4);
        assert!((mp.mass - 8.0).abs() < 1e-4);
        assert!(mp.centroid.length() < 1e-4);
        let expected = 8.0 * 4.0 / 6.0;
        assert!((mp.inertia.x_axis.x - expected).abs() < 1e-3);
        assert!((mp.inertia.y_axis.y - expected).abs() < 1e-3);
        assert!((mp.inertia.z_axis.z - expected).abs() < 1e-3);
        // Off-diagonal terms vanish for a centred cube.
        assert!(mp.inertia.x_axis.y.abs() < 1e-3);
    }
}
