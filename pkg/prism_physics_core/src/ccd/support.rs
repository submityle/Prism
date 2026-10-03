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
//! Four shape families participate as swept movers: the three bounded analytic
//! primitives (sphere, cuboid, capsule) and the [`ColliderShape::ConvexHull`]
//! polytope, which borrows its vertex data from the owning [`ShapeRegistry`]'s
//! convex-mesh arena and support-maps it under the body's rigid pose. A
//! [`ColliderShape::Plane`] is an unbounded half-space and a
//! [`ColliderShape::TriangleMesh`] is immovable scene geometry; both act only as
//! swept-into targets, never as movers, so they map to `None`.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is a
//! thin adapter over this workspace's own geometry support maps.

use crate::collider::convex_mesh::ConvexMeshData;
use crate::collider::{ColliderShape, ShapeRegistry};
use glam::{Quat, Vec3};
use prism_physics_geometry::{BoundingSphere, Capsule, Obb, SupportMap};

/// A core [`ColliderShape`] positioned at a world pose, exposed as a convex
/// [`SupportMap`] for GJK-based continuous-collision queries.
///
/// Build an analytic one with [`CcdSupport::from_shape`], or a convex-hull one
/// (which borrows the registry arena) with [`CcdSupport::from_shape_in`]. The
/// variant mirrors the source collider: a sphere stays rotation-invariant, a
/// cuboid becomes an oriented box, a capsule becomes its world-space segment
/// plus radius, and a convex hull keeps its local polytope and the rigid pose
/// used to map support directions in and out of world space.
///
/// The convex-hull variant borrows the [`ConvexMeshData`] from the registry, so
/// the support map carries the arena's lifetime; because the borrow is shared
/// and immutable the enum remains [`Copy`].
#[derive(Clone, Copy, Debug)]
pub enum CcdSupport<'a> {
    /// A solid sphere; orientation does not affect its support map.
    Sphere(BoundingSphere),
    /// An oriented box carrying the body's rotation.
    Cuboid(Obb),
    /// A capsule whose segment endpoints are placed in world space.
    Capsule(Capsule),
    /// A convex hull borrowed from the registry arena, support-mapped under the
    /// body's rigid world pose.
    ConvexMesh {
        /// Borrowed convex polytope expressed in its own local frame.
        mesh: &'a ConvexMeshData,
        /// World-space position of the body origin.
        position: Vec3,
        /// Rotation from the hull's local frame into world space.
        rotation: Quat,
    },
}

impl CcdSupport<'static> {
    /// Builds the support map for an **analytic** `shape` placed at `position`
    /// with orientation `rotation`.
    ///
    /// Returns `None` for shapes that are not analytic bounded convex volumes:
    /// a [`ColliderShape::Plane`] (an unbounded half-space) and a
    /// [`ColliderShape::TriangleMesh`] (immovable concave geometry) are the
    /// geometry a fast mover is swept *into*, and a
    /// [`ColliderShape::ConvexHull`] needs its registry arena, so it is built by
    /// [`CcdSupport::from_shape_in`] instead. All three return `None` here.
    #[must_use]
    pub fn from_shape(
        shape: &ColliderShape,
        position: Vec3,
        rotation: Quat,
    ) -> Option<CcdSupport<'static>> {
        match *shape {
            ColliderShape::Sphere { radius } => {
                Some(CcdSupport::Sphere(BoundingSphere::new(position, radius)))
            }
            ColliderShape::Cuboid { half_extents } => Some(CcdSupport::Cuboid(Obb::new(
                position,
                half_extents,
                rotation,
            ))),
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
            // Unbounded half-space, immovable mesh, or registry-backed hull:
            // none can be produced from the shape alone. See `from_shape_in`
            // for the convex-hull path.
            ColliderShape::Plane { .. }
            | ColliderShape::TriangleMesh { .. }
            | ColliderShape::ConvexHull { .. } => None,
        }
    }
}

impl<'a> CcdSupport<'a> {
    /// Builds the support map for `shape` placed at `position` with orientation
    /// `rotation`, resolving a [`ColliderShape::ConvexHull`] against `shapes`.
    ///
    /// Analytic shapes defer to [`CcdSupport::from_shape`]. A convex hull
    /// borrows its [`ConvexMeshData`] from the registry arena and keeps the
    /// rigid pose for support mapping. A [`ColliderShape::Plane`] or
    /// [`ColliderShape::TriangleMesh`] is never a swept mover and returns
    /// `None`.
    #[must_use]
    pub fn from_shape_in(
        shape: &ColliderShape,
        position: Vec3,
        rotation: Quat,
        shapes: &'a ShapeRegistry,
    ) -> Option<CcdSupport<'a>> {
        match *shape {
            ColliderShape::ConvexHull { mesh, .. } => {
                let data = shapes.convex_mesh(mesh)?;
                Some(CcdSupport::ConvexMesh {
                    mesh: data,
                    position,
                    rotation,
                })
            }
            ColliderShape::TriangleMesh { .. } => None,
            _ => CcdSupport::from_shape(shape, position, rotation),
        }
    }
}

impl SupportMap for CcdSupport<'_> {
    fn support_point(&self, dir: Vec3) -> Vec3 {
        match self {
            CcdSupport::Sphere(sphere) => sphere.support_point(dir),
            CcdSupport::Cuboid(obb) => obb.support_point(dir),
            CcdSupport::Capsule(capsule) => capsule.support_point(dir),
            CcdSupport::ConvexMesh {
                mesh,
                position,
                rotation,
            } => {
                // Support maps commute with rigid motion: rotate the query
                // direction into the hull's local frame, take the local
                // support, then map the witness back out to world space.
                let local_dir = rotation.conjugate() * dir;
                *position + *rotation * mesh.support_point(local_dir)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::convex_mesh::ConvexMeshData;

    #[test]
    fn plane_has_no_swept_support() {
        let plane = ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        };
        assert!(CcdSupport::from_shape(&plane, Vec3::ZERO, Quat::IDENTITY).is_none());
    }

    #[test]
    fn sphere_support_is_centre_plus_radius_along_dir() {
        let shape = ColliderShape::Sphere { radius: 2.0 };
        let support = CcdSupport::from_shape(&shape, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY)
            .expect("sphere is a bounded convex volume");
        let p = support.support_point(Vec3::X);
        assert!(
            (p - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-5,
            "support was {p:?}"
        );
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

    #[test]
    fn convex_hull_support_requires_registry() {
        // A convex-hull shape cannot be support-mapped from the shape alone; it
        // needs the registry arena holding its vertices.
        let mut shapes = ShapeRegistry::new();
        let handle = shapes.insert_convex_mesh(ConvexMeshData::from_box(Vec3::new(1.0, 2.0, 3.0)));
        let shape = shapes
            .convex_hull_shape(handle)
            .expect("handle is valid so the shape builds");

        assert!(CcdSupport::from_shape(&shape, Vec3::ZERO, Quat::IDENTITY).is_none());

        let support = CcdSupport::from_shape_in(&shape, Vec3::ZERO, Quat::IDENTITY, &shapes)
            .expect("registry resolves the hull");
        // The box half-extents are (1, 2, 3); support toward +X/+Y/+Z lands on
        // the matching corner.
        let p = support.support_point(Vec3::new(1.0, 1.0, 1.0));
        assert!(
            (p - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5,
            "support was {p:?}"
        );
    }

    #[test]
    fn triangle_mesh_is_never_a_mover() {
        let mut shapes = ShapeRegistry::new();
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let handle = shapes.insert_tri_mesh(crate::collider::tri_mesh::TriMeshData::new(
            verts,
            vec![[0, 1, 2]],
        ));
        let shape = shapes.tri_mesh_shape(handle).expect("valid handle");
        assert!(CcdSupport::from_shape_in(&shape, Vec3::ZERO, Quat::IDENTITY, &shapes).is_none());
    }
}
