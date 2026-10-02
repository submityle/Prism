//! Mapping from rigid [`ColliderShape`] bodies to soft-body [`BodyCollider`]
//! coupling proxies, plus the lightweight AABB overlap used to cull proxies.
//!
//! The two-way coupling kernel ([`crate::soft::resolve_two_way_coupling`])
//! works against the soft side's analytic [`BodyCollider`] primitives, not the
//! rigid-body [`ColliderShape`] registry. This module is the adapter: it turns
//! a rigid body's shape and world pose into the matching [`BodyCollider`], and
//! derives the proxy's inverse mass from its [`BodyKind`] and mass properties.
//!
//! # Shape mapping
//!
//! * [`ColliderShape::Sphere`] → [`BodyCollider::Sphere`] at the body's
//!   world position.
//! * [`ColliderShape::Capsule`] (local `Y` axis) → [`BodyCollider::Capsule`]
//!   whose endpoints are the body position offset by `±half_height` along the
//!   oriented local `Y` axis.
//! * [`ColliderShape::Plane`] (feasible region `dot(normal, x) >= offset` in the
//!   rigid convention, with the world plane shifted by the body position) →
//!   [`BodyCollider::HalfSpace`] with the same feasible-region convention.
//! * [`ColliderShape::Cuboid`] has no analytic soft proxy and is skipped
//!   ([`None`]); a box prop simply does not participate in coupling yet.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The shape
//! conversions are elementary rigid-transform algebra.

use glam::{Quat, Vec3};

use crate::collider::ColliderShape;
use crate::math::scalar::Real;
use crate::soft::{BodyCollider, CouplingBody};
use crate::state::body::BodyKind;
use crate::state::handle::BodyHandle;

/// A rigid body paired with the soft-side coupling proxy built from it.
///
/// The [`handle`](RigidProxy::handle) identifies which rigid body the proxy
/// stands in for, so the driver can write the accumulated reaction impulse back
/// onto the correct body after the coupling pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidProxy {
    /// The rigid body this proxy represents.
    pub handle: BodyHandle,
    /// The coupling body (collider + inverse mass + reaction accumulator) the
    /// two-way pass operates on.
    pub body: CouplingBody,
}

/// Builds the soft-side [`BodyCollider`] for a rigid [`ColliderShape`] at the
/// given world pose, or [`None`] when the shape has no analytic proxy.
///
/// `position` is the body's world-space origin and `orientation` its world
/// orientation. See the module docs for the per-shape mapping. A
/// [`ColliderShape::Cuboid`] returns [`None`] because the coupling kernel has no
/// box primitive.
#[must_use]
pub fn body_collider_from_shape(
    shape: &ColliderShape,
    position: Vec3,
    orientation: Quat,
) -> Option<BodyCollider> {
    match *shape {
        ColliderShape::Sphere { radius } => Some(BodyCollider::Sphere {
            center: position,
            radius,
        }),
        ColliderShape::Capsule {
            half_height,
            radius,
        } => {
            let axis = orientation * Vec3::Y;
            Some(BodyCollider::Capsule {
                p0: position - axis * half_height,
                p1: position + axis * half_height,
                radius,
            })
        }
        ColliderShape::Plane { normal, offset } => {
            // Rotate the local plane normal into the world and shift the offset
            // so the plane passes through the body position: a point `x` on the
            // world plane satisfies `world_normal.dot(x) == offset +
            // world_normal.dot(position)`.
            let world_normal = orientation * normal;
            Some(BodyCollider::HalfSpace {
                normal: world_normal,
                offset: offset + world_normal.dot(position),
            })
        }
        ColliderShape::Cuboid { .. } => None,
    }
}

/// Returns the inverse mass to give a coupling proxy for a rigid body.
///
/// Only [`BodyKind::Dynamic`] bodies get a movable proxy (their clamped
/// `inv_mass`); [`BodyKind::Static`] and [`BodyKind::Kinematic`] bodies get a
/// zero inverse mass so the proxy never moves but still records the reaction
/// impulse the cloth applies to it.
#[must_use]
pub fn proxy_inverse_mass(kind: BodyKind, inv_mass: Real) -> Real {
    match kind {
        BodyKind::Dynamic => inv_mass.max(0.0),
        BodyKind::Static | BodyKind::Kinematic => 0.0,
    }
}

/// Returns the representative anchor point of a coupling collider.
///
/// This is the point whose displacement over the coupling pass equals the rigid
/// translation the pass applied to the proxy: a sphere's center, a capsule's
/// segment midpoint, or the origin for a half-space (which the pass shifts by
/// its offset rather than translating a point).
#[must_use]
pub fn collider_anchor(collider: BodyCollider) -> Vec3 {
    match collider {
        BodyCollider::Sphere { center, .. } => center,
        BodyCollider::Capsule { p0, p1, .. } => (p0 + p1) * 0.5,
        BodyCollider::HalfSpace { .. } => Vec3::ZERO,
    }
}

/// An axis-aligned bounding box used to cull coupling proxies against a soft
/// body's extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Minimum corner (component-wise).
    pub min: Vec3,
    /// Maximum corner (component-wise).
    pub max: Vec3,
}

impl Aabb {
    /// Builds the tight AABB enclosing `points`, or [`None`] when the slice is
    /// empty (an empty soft body has no extent to cull against).
    #[must_use]
    pub fn of_points(points: &[Vec3]) -> Option<Aabb> {
        let mut iter = points.iter();
        let first = *iter.next()?;
        let mut min = first;
        let mut max = first;
        for &p in iter {
            min = min.min(p);
            max = max.max(p);
        }
        Some(Aabb { min, max })
    }

    /// Returns `true` when this box overlaps `other` on all three axes
    /// (closed intervals, so touching counts as overlap).
    #[must_use]
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

/// Returns the world-space AABB of a coupling collider, or [`None`] for an
/// unbounded half-space (which overlaps every finite box).
#[must_use]
pub fn collider_world_aabb(collider: BodyCollider) -> Option<Aabb> {
    match collider {
        BodyCollider::Sphere { center, radius } => {
            let r = Vec3::splat(radius.max(0.0));
            Some(Aabb {
                min: center - r,
                max: center + r,
            })
        }
        BodyCollider::Capsule { p0, p1, radius } => {
            let r = Vec3::splat(radius.max(0.0));
            Some(Aabb {
                min: p0.min(p1) - r,
                max: p0.max(p1) + r,
            })
        }
        BodyCollider::HalfSpace { .. } => None,
    }
}

/// Returns `true` when a coupling collider overlaps the soft body's AABB.
///
/// A bounded collider is tested with [`Aabb::intersects`]; an unbounded
/// half-space always overlaps (it extends to infinity), so it is never culled.
#[must_use]
pub fn collider_overlaps(collider: BodyCollider, soft_aabb: &Aabb) -> bool {
    match collider_world_aabb(collider) {
        Some(box_aabb) => box_aabb.intersects(soft_aabb),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_shape_maps_to_sphere_at_position() {
        let shape = ColliderShape::Sphere { radius: 0.5 };
        let got = body_collider_from_shape(&shape, Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        assert_eq!(
            got,
            Some(BodyCollider::Sphere {
                center: Vec3::new(1.0, 2.0, 3.0),
                radius: 0.5,
            })
        );
    }

    #[test]
    fn capsule_shape_maps_endpoints_along_oriented_axis() {
        let shape = ColliderShape::Capsule {
            half_height: 1.0,
            radius: 0.25,
        };
        // Identity orientation: axis is +Y.
        let got =
            body_collider_from_shape(&shape, Vec3::ZERO, Quat::IDENTITY).expect("capsule proxy");
        let BodyCollider::Capsule { p0, p1, radius } = got else {
            panic!("expected capsule");
        };
        assert!((p0 - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-6);
        assert!((p1 - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
        assert!((radius - 0.25).abs() < 1e-9);
    }

    #[test]
    fn plane_shape_maps_to_half_space_through_position() {
        let shape = ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        };
        let got = body_collider_from_shape(&shape, Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY)
            .expect("plane proxy");
        let BodyCollider::HalfSpace { normal, offset } = got else {
            panic!("expected half-space");
        };
        assert!((normal - Vec3::Y).length() < 1e-6);
        // Plane now passes through y = 2.
        assert!((offset - 2.0).abs() < 1e-6);
    }

    #[test]
    fn cuboid_shape_has_no_proxy() {
        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        assert_eq!(
            body_collider_from_shape(&shape, Vec3::ZERO, Quat::IDENTITY),
            None
        );
    }

    #[test]
    fn inverse_mass_zero_for_non_dynamic() {
        assert_eq!(proxy_inverse_mass(BodyKind::Static, 5.0), 0.0);
        assert_eq!(proxy_inverse_mass(BodyKind::Kinematic, 5.0), 0.0);
        assert_eq!(proxy_inverse_mass(BodyKind::Dynamic, 5.0), 5.0);
        // Negative clamps to zero.
        assert_eq!(proxy_inverse_mass(BodyKind::Dynamic, -1.0), 0.0);
    }

    #[test]
    fn anchor_is_center_or_midpoint() {
        assert_eq!(
            collider_anchor(BodyCollider::Sphere {
                center: Vec3::new(1.0, 2.0, 3.0),
                radius: 1.0,
            }),
            Vec3::new(1.0, 2.0, 3.0)
        );
        assert_eq!(
            collider_anchor(BodyCollider::Capsule {
                p0: Vec3::new(0.0, 0.0, 0.0),
                p1: Vec3::new(0.0, 2.0, 0.0),
                radius: 1.0,
            }),
            Vec3::new(0.0, 1.0, 0.0)
        );
    }

    #[test]
    fn aabb_of_points_handles_empty_and_nonempty() {
        assert_eq!(Aabb::of_points(&[]), None);
        let aabb = Aabb::of_points(&[Vec3::new(-1.0, 0.0, 2.0), Vec3::new(1.0, 3.0, -1.0)])
            .expect("non-empty");
        assert_eq!(aabb.min, Vec3::new(-1.0, 0.0, -1.0));
        assert_eq!(aabb.max, Vec3::new(1.0, 3.0, 2.0));
    }

    #[test]
    fn aabb_intersection_is_closed() {
        let a = Aabb {
            min: Vec3::ZERO,
            max: Vec3::splat(1.0),
        };
        let touching = Aabb {
            min: Vec3::new(1.0, 0.0, 0.0),
            max: Vec3::new(2.0, 1.0, 1.0),
        };
        let apart = Aabb {
            min: Vec3::new(1.5, 0.0, 0.0),
            max: Vec3::new(2.0, 1.0, 1.0),
        };
        assert!(a.intersects(&touching));
        assert!(!a.intersects(&apart));
    }

    #[test]
    fn half_space_never_culled() {
        let soft = Aabb {
            min: Vec3::splat(100.0),
            max: Vec3::splat(101.0),
        };
        let hs = BodyCollider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        };
        assert_eq!(collider_world_aabb(hs), None);
        assert!(collider_overlaps(hs, &soft));
    }

    #[test]
    fn sphere_culled_when_far() {
        let soft = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let near = BodyCollider::Sphere {
            center: Vec3::new(1.5, 0.0, 0.0),
            radius: 1.0,
        };
        let far = BodyCollider::Sphere {
            center: Vec3::new(10.0, 0.0, 0.0),
            radius: 1.0,
        };
        assert!(collider_overlaps(near, &soft));
        assert!(!collider_overlaps(far, &soft));
    }
}
