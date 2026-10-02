//! Bridges a core [`ColliderShape`] placed at a world pose into a geometry
//! [`SupportMap`], so continuous-collision detection can sweep a body's actual
//! convex shape rather than only its bounding sphere.
//!
//! The sphere-only [`spherecast`](crate::world::PhysicsWorld::spherecast) sweep
//! used by the baseline [`resolve_ccd`](super::resolve_ccd) over-approximates a
//! box or capsule by its circumscribed sphere, which clamps such movers earlier
//! than their true geometry would and ignores orientation entirely. Exposing the
//! real shape as a [`SupportMap`] lets the shape-aware
//! [`conservative_advancement`](prism_physics_geometry::conservative_advancement)
//! time-of-impact query drive CCD instead, matching the fidelity of
//! shape-cast CCD in UE/Chaos, `PhysX` and Jolt.
//!
//! Only the three bounded convex primitives participate; a
//! [`ColliderShape::Plane`] is an unbounded half-space that acts as a swept-into
//! target, never as a swept mover, so it maps to `None`.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is a
//! thin adapter over this workspace's own geometry support maps.

use crate::collider::ColliderShape;
use glam::{Quat, Vec3};
use prism_physics_geometry::{BoundingSphere, Capsule, Obb, SupportMap};

/// A core [`ColliderShape`] positioned at a world pose, exposed as a convex
/// [`SupportMap`] for GJK-based continuous-collision queries.
///
/// Build one with [`CcdSupport::from_shape`]. The variant mirrors the source
/// collider: a sphere stays rotation-invariant, a cuboid becomes an oriented
/// box, and a capsule becomes its world-space segment plus radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CcdSupport {
    /// A solid sphere; orientation does not affect its support map.
    Sphere(BoundingSphere),
    /// An oriented box carrying the body's rotation.
    Cuboid(Obb),
    /// A capsule whose segment endpoints are placed in world space.
    Capsule(Capsule),
}

impl CcdSupport {
    /// Builds the support map for `shape` placed at `position` with orientation
    /// `rotation`.
    ///
    /// Returns `None` for shapes that are not bounded convex volumes (today only
    /// [`ColliderShape::Plane`]): such shapes are the geometry a fast mover is
    /// swept *into*, handled analytically by the existing world queries, not a
    /// mover that is itself swept.
    #[must_use]
    pub fn from_shape(shape: &ColliderShape, position: Vec3, rotation: Quat) -> Option<CcdSupport> {
        match *shape {
            ColliderShape::Sphere { radius } => {
                Some(CcdSupport::Sphere(BoundingSphere::new(position, radius)))
            }
            ColliderShape::Cuboid { half_extents } => {
                Some(CcdSupport::Cuboid(Obb::new(position, half_extents, rotation)))
            }
            ColliderShape::Capsule {
                half_height,
                radius,
            } => {
                // The capsule's central axis is local +Y; rotate it into world
                // space and offset the two cap centres from the body origin.
                let axis = rotation * (Vec3::Y * half_height);
                Some(CcdSupport::Capsule(Capsule::new(
                    position - axis,
                    position + axis,
                    radius,
                )))
            }
            ColliderShape::Plane { .. } => None,
        }
    }
}

impl SupportMap for CcdSupport {
    fn support_point(&self, dir: Vec3) -> Vec3 {
        match self {
            CcdSupport::Sphere(sphere) => sphere.support_point(dir),
            CcdSupport::Cuboid(obb) => obb.support_point(dir),
            CcdSupport::Capsule(capsule) => capsule.support_point(dir),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_has_no_swept_support() {
        let plane = ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        };
        assert_eq!(
            CcdSupport::from_shape(&plane, Vec3::ZERO, Quat::IDENTITY),
            None
        );
    }

    #[test]
    fn sphere_support_is_centre_plus_radius_along_dir() {
        let shape = ColliderShape::Sphere { radius: 2.0 };
        let support = CcdSupport::from_shape(&shape, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY)
            .expect("sphere is a bounded convex volume");
        let p = support.support_point(Vec3::X);
        assert!((p - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-5, "support was {p:?}");
    }

    #[test]
    fn cuboid_support_rotates_with_the_body() {
        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::new(1.0, 0.5, 0.5),
        };
        // Rotate 90 degrees about Z so the long local-X axis points along world +Y.
        let rot = Quat::from_rotation_z(core::f32::consts::FRAC_PI_2);
        let support = CcdSupport::from_shape(&shape, Vec3::ZERO, rot).expect("cuboid is convex");
        let p = support.support_point(Vec3::Y);
        // The farthest point along +Y is the rotated local +X half-extent (1.0).
        assert!((p.y - 1.0).abs() < 1e-5, "support was {p:?}");
    }

    #[test]
    fn capsule_support_reaches_the_far_cap() {
        let shape = ColliderShape::Capsule {
            half_height: 2.0,
            radius: 0.5,
        };
        let support = CcdSupport::from_shape(&shape, Vec3::ZERO, Quat::IDENTITY).expect("capsule");
        // Along +Y the support is the top cap centre (0, 2, 0) plus the radius.
        let p = support.support_point(Vec3::Y);
        assert!((p.y - 2.5).abs() < 1e-5, "support was {p:?}");
    }

    #[test]
    fn capsule_axis_follows_rotation() {
        let shape = ColliderShape::Capsule {
            half_height: 2.0,
            radius: 0.5,
        };
        // Rotate the local +Y axis onto world +X.
        let rot = Quat::from_rotation_z(-core::f32::consts::FRAC_PI_2);
        let support = CcdSupport::from_shape(&shape, Vec3::ZERO, rot).expect("capsule");
        let p = support.support_point(Vec3::X);
        assert!((p.x - 2.5).abs() < 1e-5, "support was {p:?}");
    }
}
