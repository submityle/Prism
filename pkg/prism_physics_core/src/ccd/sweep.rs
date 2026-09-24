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
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Taking a
//! shape's inscribed-sphere radius is elementary geometry.

use crate::collider::ColliderShape;

/// Returns the conservative core (inscribed-sphere) radius used to sweep
/// `shape` for continuous collision detection, or `None` for shapes that cannot
/// be swept.
///
/// - [`ColliderShape::Sphere`] uses its own radius exactly.
/// - [`ColliderShape::Cuboid`] uses the smallest half-extent, the largest
///   sphere that fits inside the box.
/// - [`ColliderShape::Capsule`] uses its cap radius, the largest sphere that
///   fits inside the capsule's cross-section.
/// - [`ColliderShape::Plane`] is an unbounded static half-space that never
///   moves, so it is never swept and returns `None`.
#[must_use]
pub fn sweep_radius(shape: &ColliderShape) -> Option<f32> {
    match *shape {
        // A sphere is its own inscribed sphere; a capsule's inscribed sphere is
        // its cap radius, so both map to the same radius expression.
        ColliderShape::Sphere { radius } | ColliderShape::Capsule { radius, .. } => Some(radius),
        ColliderShape::Cuboid { half_extents } => {
            Some(half_extents.x.min(half_extents.y).min(half_extents.z))
        }
        ColliderShape::Plane { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
