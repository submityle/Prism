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
//! * [`ColliderShape::Cuboid`] → [`BodyCollider::Obb`] at the body's world pose
//!   (center = position, orientation = body orientation, matching half-extents),
//!   so a box prop collides against cloth via its true oriented-box faces.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The shape
//! conversions are elementary rigid-transform algebra.

use glam::{Quat, Vec3};

use crate::collider::ColliderShape;
use crate::math::scalar::Real;
use crate::soft::{BodyCollider, ConvexProxy, CouplingBody};
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
/// [`ColliderShape::Cuboid`] maps to a [`BodyCollider::Obb`] carrying the body's
/// orientation and half-extents. The function still returns [`None`] for any
/// future shape that has no analytic proxy.
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
        ColliderShape::Cuboid { half_extents } => Some(BodyCollider::Obb {
            center: position,
            orientation,
            half_extents,
        }),
    }
}

/// Builds a bounded [`ConvexProxy`] for a cuboid at the given world pose.
///
/// This is the opt-in convex mapping of [`ColliderShape::Cuboid`]: it produces
/// the box as the intersection of its six oriented face half-spaces, which
/// projects interior particles identically to the default
/// [`BodyCollider::Obb`] mapping. It exists so callers (and the cross-validation
/// goldens) can exercise the convex arm against a shape whose oriented-box
/// answer is already known; the default [`body_collider_from_shape`] mapping is
/// unchanged, so existing goldens stay bit-identical.
#[must_use]
pub fn convex_proxy_from_cuboid(
    half_extents: Vec3,
    position: Vec3,
    orientation: Quat,
) -> ConvexProxy {
    ConvexProxy::from_box(position, orientation, half_extents)
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
        BodyCollider::Sphere { center, .. } | BodyCollider::Obb { center, .. } => center,
        BodyCollider::Capsule { p0, p1, .. } => (p0 + p1) * 0.5,
        BodyCollider::HalfSpace { .. } => Vec3::ZERO,
        BodyCollider::ConvexHull(proxy) => proxy.center(),
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
        BodyCollider::Obb {
            center,
            orientation,
            half_extents,
        } => {
            // World AABB of the oriented box: the world half-extent along each
            // world axis is the sum over the local axes of
            // `|rotation_component| * half_extent`, i.e. `|R| * he` with the
            // component-wise absolute rotation matrix. Negative extents are
            // clamped to zero so a collapsed axis contributes nothing.
            let m = glam::Mat3::from_quat(orientation);
            let abs = glam::Mat3::from_cols(m.x_axis.abs(), m.y_axis.abs(), m.z_axis.abs());
            let world_half = abs * half_extents.max(Vec3::ZERO);
            Some(Aabb {
                min: center - world_half,
                max: center + world_half,
            })
        }
        BodyCollider::ConvexHull(proxy) => {
            // Conservative bounding box from the proxy's cached bounding sphere.
            let r = Vec3::splat(proxy.bounding_radius().max(0.0));
            let center = proxy.center();
            Some(Aabb {
                min: center - r,
                max: center + r,
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
    fn cuboid_shape_maps_to_obb_at_pose() {
        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::new(0.5, 0.25, 1.0),
        };
        let orientation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let got = body_collider_from_shape(&shape, Vec3::new(1.0, 2.0, 3.0), orientation)
            .expect("cuboid proxy");
        assert_eq!(
            got,
            BodyCollider::Obb {
                center: Vec3::new(1.0, 2.0, 3.0),
                orientation,
                half_extents: Vec3::new(0.5, 0.25, 1.0),
            }
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
        assert_eq!(
            collider_anchor(BodyCollider::Obb {
                center: Vec3::new(-2.0, 1.0, 4.0),
                orientation: Quat::from_rotation_x(0.3),
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            }),
            Vec3::new(-2.0, 1.0, 4.0)
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
    fn obb_world_aabb_is_axis_aligned_when_unrotated() {
        let obb = BodyCollider::Obb {
            center: Vec3::new(1.0, 2.0, 3.0),
            orientation: Quat::IDENTITY,
            half_extents: Vec3::new(0.5, 0.25, 2.0),
        };
        let aabb = collider_world_aabb(obb).expect("bounded");
        assert!((aabb.min - Vec3::new(0.5, 1.75, 1.0)).length() < 1e-6);
        assert!((aabb.max - Vec3::new(1.5, 2.25, 5.0)).length() < 1e-6);
    }

    #[test]
    fn obb_world_aabb_grows_when_rotated() {
        // A 45-degree yaw grows the XZ footprint of a square base from
        // half-extent 1 to half-extent sqrt(2); the Y extent is unchanged.
        let obb = BodyCollider::Obb {
            center: Vec3::ZERO,
            orientation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            half_extents: Vec3::new(1.0, 0.5, 1.0),
        };
        let aabb = collider_world_aabb(obb).expect("bounded");
        let expected = 2.0_f32.sqrt();
        assert!((aabb.max.x - expected).abs() < 1e-6, "x: {}", aabb.max.x);
        assert!((aabb.max.z - expected).abs() < 1e-6, "z: {}", aabb.max.z);
        assert!((aabb.max.y - 0.5).abs() < 1e-6, "y: {}", aabb.max.y);
        assert!(
            (aabb.min + aabb.max).length() < 1e-6,
            "symmetric about center"
        );
    }

    #[test]
    fn obb_overlaps_soft_box_when_near() {
        let soft = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let near = BodyCollider::Obb {
            center: Vec3::new(1.4, 0.0, 0.0),
            orientation: Quat::IDENTITY,
            half_extents: Vec3::splat(0.5),
        };
        let far = BodyCollider::Obb {
            center: Vec3::new(10.0, 0.0, 0.0),
            orientation: Quat::IDENTITY,
            half_extents: Vec3::splat(0.5),
        };
        assert!(collider_overlaps(near, &soft));
        assert!(!collider_overlaps(far, &soft));
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

    // --- Convex-hull proxy coupling goldens -------------------------------
    //
    // These exercise the `BodyCollider::ConvexHull` arm through the shared
    // per-particle coupling kernel (`couple_particle_against_body`), the same
    // entry point the linear/angular/friction drivers all funnel through, so a
    // passing cross-check here means those three drivers auto-work for convex
    // props without any driver change.
    use crate::soft::couple_particle_against_body;

    /// An oriented box and its convex-from-box proxy at the same pose must
    /// produce bit-identical coupling contributions for interior particles, so
    /// a convex box couples exactly like the existing OBB arm.
    #[test]
    fn convex_hull_couples_identically_to_obb() {
        let center = Vec3::new(0.4, 1.0, -0.7);
        let orientation =
            Quat::from_rotation_y(0.7) * Quat::from_rotation_x(0.3) * Quat::from_rotation_z(-0.2);
        let he = Vec3::new(0.8, 0.5, 1.2);
        let obb = BodyCollider::Obb {
            center,
            orientation,
            half_extents: he,
        };
        let hull = BodyCollider::ConvexHull(ConvexProxy::from_box(center, orientation, he));
        let samples = [
            center + orientation * Vec3::new(0.1, 0.3, -0.4),
            center + orientation * Vec3::new(-0.5, 0.1, 0.6),
            center + orientation * Vec3::new(0.2, -0.35, 0.1),
        ];
        for pos in samples {
            let a = couple_particle_against_body(pos, 1.0, obb, 0.5, 1.0 / 60.0);
            let b = couple_particle_against_body(pos, 1.0, hull, 0.5, 1.0 / 60.0);
            // The two arms share the same geometry but reach it via different
            // arithmetic (box-local clamp vs. plane dot products), so they agree
            // to float tolerance rather than bit-for-bit on a rotated box.
            assert!(
                (a.particle_delta - b.particle_delta).length() < 1e-5,
                "particle delta mismatch: {a:?} vs {b:?}"
            );
            assert!((a.body_delta - b.body_delta).length() < 1e-5);
            assert!((a.impulse - b.impulse).length() < 1e-3);
            // And the interior particle actually moved (a real contact).
            assert!(b.particle_delta.length() > 1e-6);
        }
    }

    /// The opt-in `convex_proxy_from_cuboid` builder must reproduce the default
    /// `Cuboid -> Obb` mapping's projection, the cross-validation entry point.
    #[test]
    fn convex_proxy_from_cuboid_matches_obb_mapping() {
        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::new(0.6, 0.3, 0.9),
        };
        let position = Vec3::new(-1.0, 2.0, 0.5);
        let orientation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_3);
        let obb = body_collider_from_shape(&shape, position, orientation).expect("obb proxy");
        let hull = BodyCollider::ConvexHull(convex_proxy_from_cuboid(
            Vec3::new(0.6, 0.3, 0.9),
            position,
            orientation,
        ));
        let pos = position + orientation * Vec3::new(0.1, 0.2, -0.3);
        // Same surface via different arithmetic => float-tolerance equal.
        assert!((obb.project(pos) - hull.project(pos)).length() < 1e-5);
    }

    /// An interior particle under a tilted face is pushed out along that face's
    /// world normal (so the follow-up solve lets it slide along the face).
    #[test]
    fn convex_hull_push_is_along_tilted_face_normal() {
        let orientation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_6);
        let hull = BodyCollider::ConvexHull(ConvexProxy::from_box(
            Vec3::ZERO,
            orientation,
            Vec3::new(1.0, 0.4, 1.0),
        ));
        // Just inside the +Y face in box-local coordinates.
        let pos = orientation * Vec3::new(0.1, 0.35, -0.2);
        let contrib = couple_particle_against_body(pos, 1.0, hull, 0.0, 1.0 / 60.0);
        // Body is infinite-mass (w_body 0) so the particle takes the full push.
        let correction = contrib.particle_delta;
        assert!(correction.length() > 1e-4, "expected a push");
        let expected_normal = orientation * Vec3::Y;
        let along = correction.normalize_or_zero().dot(expected_normal);
        assert!(
            (along - 1.0).abs() < 1e-5,
            "push not along face normal: {along}"
        );
    }

    /// A particle outside the convex solid produces no coupling (zero
    /// contribution), the back-face / outside miss case.
    #[test]
    fn convex_hull_outside_particle_is_no_op() {
        let hull = BodyCollider::ConvexHull(ConvexProxy::from_box(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(1.0, 0.5, 2.0),
        ));
        let outside = Vec3::new(5.0, 0.0, 0.0);
        let contrib = couple_particle_against_body(outside, 1.0, hull, 1.0, 1.0 / 60.0);
        assert_eq!(contrib, crate::soft::CouplingContribution::ZERO);
    }

    /// A degenerate (empty) convex proxy is inert: no projection, no coupling.
    #[test]
    fn convex_hull_empty_proxy_is_inert() {
        let hull = BodyCollider::ConvexHull(ConvexProxy::EMPTY);
        let pos = Vec3::new(0.1, 0.2, 0.3);
        assert_eq!(hull.project(pos), pos);
        let contrib = couple_particle_against_body(pos, 1.0, hull, 1.0, 1.0 / 60.0);
        assert_eq!(contrib, crate::soft::CouplingContribution::ZERO);
    }

    /// Identical inputs produce identical contributions on repeat (determinism).
    #[test]
    fn convex_hull_coupling_is_deterministic() {
        let hull = BodyCollider::ConvexHull(ConvexProxy::from_box(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::from_rotation_x(0.4),
            Vec3::new(0.7, 0.5, 0.9),
        ));
        let pos = Vec3::new(0.05, 1.1, 0.1);
        let a = couple_particle_against_body(pos, 1.0, hull, 0.5, 1.0 / 60.0);
        let b = couple_particle_against_body(pos, 1.0, hull, 0.5, 1.0 / 60.0);
        assert_eq!(a, b);
    }

    /// The convex anchor is its center and its world AABB encloses that center.
    #[test]
    fn convex_hull_anchor_and_aabb() {
        let center = Vec3::new(1.0, -2.0, 3.0);
        let proxy = ConvexProxy::from_box(center, Quat::IDENTITY, Vec3::new(0.5, 0.5, 0.5));
        let hull = BodyCollider::ConvexHull(proxy);
        assert_eq!(collider_anchor(hull), center);
        let aabb = collider_world_aabb(hull).expect("convex hull is bounded");
        assert!(aabb.min.x <= center.x && aabb.max.x >= center.x);
        assert!(aabb.min.y <= center.y && aabb.max.y >= center.y);
        assert!(aabb.min.z <= center.z && aabb.max.z >= center.z);
    }
}
