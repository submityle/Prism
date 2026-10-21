//! Shape-aware CCD: real convex leaves and mesh triangles under root motion.
//! Compound bounds only cull candidates; they never enter narrow-phase GJK.
//! See leaf_sweep for two-sided rotational conservative advancement.

#[cfg(test)]
use crate::ccd::support::CcdSupport;
use crate::collider::ColliderShape;
use crate::math::transform::Isometry;
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;
use glam::{Quat, Vec3};

/// First contact fraction for the mover's rigid motion, including compounds.
/// Both moving roots are interpolated; children rotate about their owner COM.
#[must_use]
pub(super) fn sweep_toi(
    world: &PhysicsWorld,
    mover: BodyHandle,
    mover_shape: &ColliderShape,
    prev_pos: Vec3,
    prev_rot: Quat,
    curr_rot: Quat,
    displacement: Vec3,
) -> Option<f32> {
    super::leaf_sweep::sweep_toi(
        world,
        mover,
        mover_shape,
        Isometry::new(prev_pos, prev_rot),
        Isometry::new(prev_pos + displacement, curr_rot),
    )
}

/// Radius of the smallest sphere centred on the body origin that encloses
/// `shape`, used both to pad the broad-phase swept box and as the lever arm
/// that converts the sub-step spin into a worst-case point speed.
#[must_use]
pub(super) fn bounding_radius(shape: &ColliderShape) -> f32 {
    match *shape {
        ColliderShape::Sphere { radius } => radius,
        ColliderShape::Capsule {
            half_height,
            radius,
        } => half_height + radius,
        ColliderShape::Cuboid { half_extents } => half_extents.length(),
        // Mesh and compound colliders cache their local AABB; the farthest
        // corner from the body origin bounds every vertex, conservative for the swept
        // pad and the angular lever arm. This stays registry-free so the sweep
        // broad-phase can pad without resolving the arena.
        ColliderShape::ConvexHull {
            local_aabb_min,
            local_aabb_max,
            ..
        }
        | ColliderShape::TriangleMesh {
            local_aabb_min,
            local_aabb_max,
            ..
        }
        | ColliderShape::Compound {
            local_aabb_min,
            local_aabb_max,
            ..
        } => local_aabb_min.abs().max(local_aabb_max.abs()).length(),
        // Planes are never movers; a zero pad is harmless if one is ever passed.
        ColliderShape::Plane { .. } => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_bounding_radius_covers_offset_children_without_convex_support() {
        use crate::collider::{CompoundChild, ShapeRegistry};
        use crate::math::transform::Isometry;

        let mut shapes = ShapeRegistry::new();
        let sphere = shapes.insert(ColliderShape::Sphere { radius: 0.5 });
        let children = [-4.0, -2.0].map(|x| CompoundChild {
            collider: sphere,
            local_pose: Isometry::new(Vec3::new(x, 0.0, 0.0), Quat::IDENTITY),
        });
        let handle = shapes.insert_compound(&children).expect("valid union");
        let shape = shapes.get(handle).expect("registered compound");
        // Bounds are [-4.5, -0.5, -0.5] .. [-1.5, 0.5, 0.5]. The radius is
        // measured from the body origin, not the bounds' off-centre midpoint.
        let expected = Vec3::new(4.5, 0.5, 0.5).length();
        assert!((bounding_radius(shape) - expected).abs() < 1e-6);
        assert!(CcdSupport::from_shape(shape, Vec3::ZERO, Quat::IDENTITY).is_none());
        assert!(CcdSupport::from_shape_in(shape, Vec3::ZERO, Quat::IDENTITY, &shapes).is_none());
    }

    #[test]
    fn bounding_radius_covers_each_primitive() {
        assert_eq!(bounding_radius(&ColliderShape::Sphere { radius: 2.0 }), 2.0);
        assert_eq!(
            bounding_radius(&ColliderShape::Capsule {
                half_height: 1.5,
                radius: 0.5,
            }),
            2.0
        );
        let r = bounding_radius(&ColliderShape::Cuboid {
            half_extents: Vec3::new(3.0, 4.0, 0.0),
        });
        assert!((r - 5.0).abs() < 1e-6, "cuboid radius was {r}");
    }

    #[test]
    fn sweep_toi_is_none_for_a_body_at_rest() {
        // A body that neither translates nor rotates must not be swept, even if
        // a wall sits right next to it.
        use crate::collider::PhysicsMaterial;
        use crate::state::body::BodyDesc;

        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let wall = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::new(0.05, 1.0, 1.0),
        });
        world.spawn(
            BodyDesc::static_at(Vec3::ZERO)
                .with_collider(wall)
                .with_material(PhysicsMaterial::DEFAULT),
        );

        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.1),
        };
        let collider = world.shapes.insert(shape);
        let mover = world.spawn(
            BodyDesc::dynamic_at(Vec3::new(-0.5, 0.0, 0.0))
                .with_collider(collider)
                .with_material(PhysicsMaterial::DEFAULT),
        );

        assert_eq!(
            sweep_toi(
                &world,
                mover,
                &shape,
                Vec3::new(-0.5, 0.0, 0.0),
                Quat::IDENTITY,
                Quat::IDENTITY,
                Vec3::ZERO,
            ),
            None
        );
    }

    #[test]
    fn sweep_toi_catches_a_spin_a_linear_sweep_would_miss() {
        // A long thin plank centred just left of a thin wall, with its long axis
        // pointing *up* so no corner reaches the wall in the start pose and the
        // centre never translates. A quarter turn about Z swings the top corner
        // horizontally into the wall. A pure-translation sweep (zero
        // displacement) returns None; the rotational sweep must return a finite
        // sub-step fraction well before the end of the step.
        use crate::collider::PhysicsMaterial;
        use crate::state::body::BodyDesc;

        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let wall = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::new(0.05, 2.0, 2.0),
        });
        world.spawn(
            BodyDesc::static_at(Vec3::ZERO)
                .with_collider(wall)
                .with_material(PhysicsMaterial::DEFAULT),
        );

        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::new(0.05, 0.6, 0.05),
        };
        let collider = world.shapes.insert(shape);
        // Centre at x = -0.5: the plank's upright half-extent (0.6) does not
        // reach the wall's near face (x = -0.05), but a 90 degree turn about Z
        // lays it flat so the former top corner swings to about x = -0.5 + 0.6 =
        // +0.1, through the wall.
        let center = Vec3::new(-0.5, 0.0, 0.0);
        let mover = world.spawn(
            BodyDesc::dynamic_at(center)
                .with_collider(collider)
                .with_material(PhysicsMaterial::DEFAULT),
        );

        let prev_rot = Quat::IDENTITY;
        let curr_rot = Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2);

        // Pure-translation sweep sees no motion and misses the tunnelling spin.
        assert_eq!(
            sweep_toi(
                &world,
                mover,
                &shape,
                center,
                prev_rot,
                prev_rot,
                Vec3::ZERO
            ),
            None,
            "a non-spinning, non-translating plank must not be clamped"
        );

        // The rotational sweep catches it before the step completes.
        let toi = sweep_toi(
            &world,
            mover,
            &shape,
            center,
            prev_rot,
            curr_rot,
            Vec3::ZERO,
        )
        .expect("the spinning corner must register a time of impact");
        assert!(
            (0.0..1.0).contains(&toi),
            "rotational toi should clamp before the end of the sub-step, got {toi}"
        );
    }
}
