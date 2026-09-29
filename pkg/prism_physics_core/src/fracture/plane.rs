//! Oriented planes and half-spaces used to carve convex fracture cells.
//!
//! A [`Plane`] stores a unit normal `n` and a scalar `offset` so that its
//! surface is the set of points satisfying `n · x = offset`. The associated
//! *interior* half-space is `n · x <= offset`; a convex polyhedron is the
//! intersection of the interior half-spaces of its face planes, and a Voronoi
//! cell is the intersection of the bisector half-spaces between one site and
//! all of its neighbours.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Plane
//! algebra, perpendicular bisectors, and three-plane intersection via Cramer's
//! rule are standard, publicly documented analytic-geometry results.

use glam::{Mat3, Vec3};

use crate::math::scalar::Real;

/// An oriented plane `n · x = offset` with a unit normal.
///
/// The interior half-space is `n · x <= offset`, i.e. the side the normal
/// points *away* from. [`Plane::signed_distance`] is therefore negative for
/// interior points and positive for exterior points.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Plane {
    /// Unit-length outward normal of the plane.
    pub normal: Vec3,
    /// Signed offset along the normal: the plane passes through `offset * normal`.
    pub offset: Real,
}

impl Plane {
    /// Builds a plane from a (not necessarily unit) `normal` and `offset`,
    /// normalising both so that [`Plane::signed_distance`] returns true
    /// Euclidean distance.
    ///
    /// A near-zero normal is replaced by `+X` with a zero offset, yielding a
    /// harmless degenerate plane rather than `NaN`.
    #[must_use]
    pub fn new(normal: Vec3, offset: Real) -> Plane {
        let len = normal.length();
        if len <= Real::EPSILON {
            return Plane {
                normal: Vec3::X,
                offset: 0.0,
            };
        }
        Plane {
            normal: normal / len,
            offset: offset / len,
        }
    }

    /// Builds the plane with the given unit-ish `normal` that passes through
    /// `point`.
    #[must_use]
    pub fn from_point_normal(point: Vec3, normal: Vec3) -> Plane {
        let unit = normal.normalize_or_zero();
        let unit = if unit == Vec3::ZERO { Vec3::X } else { unit };
        Plane {
            normal: unit,
            offset: unit.dot(point),
        }
    }

    /// Returns the perpendicular-bisector plane between sites `a` and `b`
    /// oriented so that its interior half-space (`n · x <= offset`) contains
    /// `a`.
    ///
    /// This is the half-space of the Voronoi cell of `a` induced by neighbour
    /// `b`. Coincident sites produce a harmless degenerate plane.
    #[must_use]
    pub fn bisector(a: Vec3, b: Vec3) -> Plane {
        let mid = (a + b) * 0.5;
        Plane::from_point_normal(mid, b - a)
    }

    /// Signed distance from `p` to the plane: negative inside, positive
    /// outside, zero on the surface.
    #[must_use]
    pub fn signed_distance(&self, p: Vec3) -> Real {
        self.normal.dot(p) - self.offset
    }

    /// Returns `true` when `p` lies in the closed interior half-space within
    /// `eps` tolerance.
    #[must_use]
    pub fn contains(&self, p: Vec3, eps: Real) -> bool {
        self.signed_distance(p) <= eps
    }
}

/// Solves for the unique point lying on all three planes, or returns `None`
/// when they do not meet in a single point (near-parallel / near-degenerate).
///
/// The linear system stacks the three plane normals as matrix rows and solves
/// `M x = d` via matrix inversion; the determinant magnitude gates
/// degeneracy so nearly parallel planes are rejected instead of producing a
/// far-away spurious vertex.
#[must_use]
pub fn intersect_three(p1: &Plane, p2: &Plane, p3: &Plane, det_eps: Real) -> Option<Vec3> {
    // Rows of `m` are the plane normals: column j collects the j-th component
    // of each normal, so row i reconstructs normal i.
    let m = Mat3::from_cols(
        Vec3::new(p1.normal.x, p2.normal.x, p3.normal.x),
        Vec3::new(p1.normal.y, p2.normal.y, p3.normal.y),
        Vec3::new(p1.normal.z, p2.normal.z, p3.normal.z),
    );
    let det = m.determinant();
    if det.abs() <= det_eps {
        return None;
    }
    let d = Vec3::new(p1.offset, p2.offset, p3.offset);
    Some(m.inverse() * d)
}
