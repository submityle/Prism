//! A single triangle exposed as a convex [`SupportMap`] for continuous
//! collision against [`ColliderShape::TriangleMesh`](crate::collider::ColliderShape::TriangleMesh)
//! scene geometry.
//!
//! A triangle mesh is concave as a whole, so it cannot be support-mapped
//! directly. Continuous collision instead treats each candidate triangle as its
//! own degenerate convex hull (three coplanar vertices) and advances the fast
//! mover against it with the same
//! [`rotational_conservative_advancement`](prism_physics_geometry::rotational_conservative_advancement)
//! query used for convex-vs-convex sweeps. The broad-phase narrows the triangle
//! set to those overlapping the mover's swept bounds first, so only a handful of
//! triangles are ever support-mapped per sweep.
//!
//! The vertices are stored in **world space**: the caller transforms the three
//! mesh-local triangle vertices by the target body's rigid pose once, so the
//! support map itself is a plain argmax over `dot(dir, vertex)` with no pose
//! bookkeeping.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! argmax-over-vertices support map of a triangle is elementary geometry.

use glam::Vec3;
use prism_physics_geometry::SupportMap;

/// A triangle, in world space, exposed as a convex support map.
///
/// The triangle is the convex hull of its three vertices, so its support point
/// in any direction is simply the vertex with the greatest projection onto that
/// direction.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TriangleSupport {
    a: Vec3,
    b: Vec3,
    c: Vec3,
}

impl TriangleSupport {
    /// Builds a triangle support map from three world-space vertices.
    #[must_use]
    pub(crate) fn new(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self { a, b, c }
    }
}

impl SupportMap for TriangleSupport {
    fn support_point(&self, dir: Vec3) -> Vec3 {
        let da = dir.dot(self.a);
        let db = dir.dot(self.b);
        let dc = dir.dot(self.c);
        if da >= db && da >= dc {
            self.a
        } else if db >= dc {
            self.b
        } else {
            self.c
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_triangle() -> TriangleSupport {
        TriangleSupport::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn support_picks_the_farthest_vertex() {
        let tri = unit_triangle();
        assert_eq!(tri.support_point(Vec3::X), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(tri.support_point(Vec3::Y), Vec3::new(0.0, 1.0, 0.0));
        // Toward -X and -Y the origin vertex wins.
        assert_eq!(
            tri.support_point(Vec3::new(-1.0, -1.0, 0.0)),
            Vec3::new(0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn support_is_stable_for_a_tie() {
        // A direction orthogonal to the triangle plane ties all three vertices;
        // the argmax must still return one of them deterministically.
        let tri = unit_triangle();
        let p = tri.support_point(Vec3::Z);
        assert!(
            p == Vec3::ZERO || p == Vec3::new(1.0, 0.0, 0.0) || p == Vec3::new(0.0, 1.0, 0.0),
            "tie resolved to a non-vertex point {p:?}"
        );
    }
}
