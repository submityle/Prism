//! Core-radius extraction for the continuous-collision sweep.
//!
//! Continuous collision detection approximates a moving body by the largest
//! sphere that fits inside it (its *core radius*) and sweeps that sphere along
//! the body's sub-step motion. Using the inscribed sphere is conservative: the
//! swept sphere never pokes outside the true shape, so a reported time of
//! impact is a lower bound on when the real geometry would touch. This keeps
//! the clamp safe (it never lets the body pass through) at the cost of stopping
//! slightly early for non-spherical shapes.
//!
//! Analytic primitives expose their inscribed radius in closed form through
//! [`sweep_radius`]. A [`ColliderShape::ConvexHull`] stores its vertices in the
//! owning [`ShapeRegistry`]'s arena, so its inscribed radius is only available
//! through the registry-aware [`sweep_radius_in`]; a
//! [`ColliderShape::TriangleMesh`] is immovable scene geometry and is never a
//! swept mover, so it has no core radius.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Taking a
//! shape's inscribed-sphere radius is elementary geometry.

use crate::collider::{ColliderShape, ShapeRegistry};

/// Smallest core radius a swept mover may report; keeps the motion gate and the
/// swept-sphere padding well away from a zero divide for near-degenerate hulls.
const MIN_CORE_RADIUS: f32 = 1.0e-4;

/// Returns the conservative core (inscribed-sphere) radius used to sweep
/// `shape` for continuous collision detection, or `None` for shapes that cannot
/// be swept from the shape alone.
///
/// - [`ColliderShape::Sphere`] uses its own radius exactly.
/// - [`ColliderShape::Cuboid`] uses the smallest half-extent, the largest
///   sphere that fits inside the box.
/// - [`ColliderShape::Capsule`] uses its cap radius, the largest sphere that
///   fits inside the capsule's cross-section.
/// - [`ColliderShape::Plane`] is an unbounded static half-space that never
///   moves, so it is never swept and returns `None`.
/// - [`ColliderShape::ConvexHull`] needs its registry arena to measure its
///   inscribed radius and [`ColliderShape::TriangleMesh`] is immovable scene
///   geometry; both return `None` here. Resolve a convex hull through
///   [`sweep_radius_in`] instead.
#[must_use]
pub fn sweep_radius(shape: &ColliderShape) -> Option<f32> {
    match *shape {
        // A sphere is its own inscribed sphere; a capsule's inscribed sphere is
        // its cap radius, so both map to the same radius expression.
        ColliderShape::Sphere { radius } | ColliderShape::Capsule { radius, .. } => Some(radius),
        ColliderShape::Cuboid { half_extents } => {
            Some(half_extents.x.min(half_extents.y).min(half_extents.z))
        }
        // A plane has no inscribed sphere. A convex hull's inscribed radius
        // lives behind the arena and a triangle mesh is never a mover; both are
        // resolved (or rejected) by `sweep_radius_in`.
        ColliderShape::Plane { .. }
        | ColliderShape::ConvexHull { .. }
        | ColliderShape::TriangleMesh { .. } => None,
    }
}

/// Registry-aware core radius for `shape`, resolving a
/// [`ColliderShape::ConvexHull`] against the convex-mesh arena in `shapes`.
///
/// Analytic primitives defer to [`sweep_radius`]. A convex hull reports the
/// radius of its largest inscribed sphere (clamped to a small positive floor so
/// a degenerate hull still gates and pads safely). A
/// [`ColliderShape::TriangleMesh`] is immovable scene geometry and returns
/// `None`, so it is only ever a swept-into target, never a mover.
#[must_use]
pub fn sweep_radius_in(shape: &ColliderShape, shapes: &ShapeRegistry) -> Option<f32> {
    match *shape {
        ColliderShape::ConvexHull { mesh, .. } => Some(
            shapes
                .convex_mesh(mesh)?
                .inscribed_radius()
                .max(MIN_CORE_RADIUS),
        ),
        ColliderShape::TriangleMesh { .. } => None,
        _ => sweep_radius(shape),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::convex_mesh::ConvexMeshData;
    use crate::collider::tri_mesh::TriMeshData;
    use glam::Vec3;

    #[test]
    fn sphere_uses_own_radius() {
        let r = sweep_radius(&ColliderShape::Sphere { radius: 0.75 });
        assert_eq!(r, Some(0.75));
    }

    #[test]
    fn cuboid_uses_smallest_half_extent() {
        let r = sweep_radius(&ColliderShape::Cuboid {
            half_extents: Vec3::new(2.0, 0.1, 3.0),
        });
        assert_eq!(r, Some(0.1));
    }

    #[test]
    fn capsule_uses_cap_radius() {
        let r = sweep_radius(&ColliderShape::Capsule {
            half_height: 1.0,
            radius: 0.3,
        });
        assert_eq!(r, Some(0.3));
    }

    #[test]
    fn plane_is_not_swept() {
        let r = sweep_radius(&ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        });
        assert_eq!(r, None);
    }

    #[test]
    fn mesh_variants_need_the_registry() {
        let mut shapes = ShapeRegistry::new();
        let hull = shapes.insert_convex_mesh(ConvexMeshData::from_box(Vec3::new(0.5, 1.0, 2.0)));
        let hull_shape = shapes.convex_hull_shape(hull).expect("valid handle");
        // The shape-only path cannot measure a hull's inscribed radius.
        assert_eq!(sweep_radius(&hull_shape), None);
        // The registry-aware path returns the inscribed sphere (smallest
        // half-extent for an axis-aligned box, 0.5 here).
        let r = sweep_radius_in(&hull_shape, &shapes).expect("hull resolves");
        assert!((r - 0.5).abs() < 1e-4, "inscribed radius was {r}");
    }

    #[test]
    fn triangle_mesh_is_never_a_mover() {
        let mut shapes = ShapeRegistry::new();
        let tri = shapes.insert_tri_mesh(TriMeshData::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        ));
        let tri_shape = shapes.tri_mesh_shape(tri).expect("valid handle");
        assert_eq!(sweep_radius(&tri_shape), None);
        assert_eq!(sweep_radius_in(&tri_shape, &shapes), None);
    }
}
