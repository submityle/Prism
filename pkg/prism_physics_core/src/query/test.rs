//! Boolean "any-hit" scene queries: does the probe touch *anything*?
//!
//! The single-hit queries ([`raycast`](crate::world::PhysicsWorld::raycast),
//! [`spherecast`](crate::world::PhysicsWorld::spherecast),
//! [`shapecast`](crate::world::PhysicsWorld::shapecast)) and the multi-hit
//! collectors ([`raycast_all`](crate::world::PhysicsWorld::raycast_all) and its
//! siblings) both answer *where* a probe hits. Many gameplay checks only need a
//! yes/no: is this spawn point clear, can the AI see the player, is the jump arc
//! blocked? Finding the nearest contact — let alone every contact — is wasted
//! work for those.
//!
//! These `*_test` / `*_any` variants return a `bool` and **stop at the first
//! accepted contact**: no time-of-impact comparison, no sort, no allocation.
//! They mirror UE's `LineTraceTestByChannel` / `SweepTestByChannel` /
//! `OverlapAnyTestByChannel`, `PhysX`'s `eANY_HIT` scene-query flag, and Jolt's
//! `AnyHitCollisionCollector`, and are the physics analogue of a ray tracer's
//! `ANY_HIT` versus `CLOSEST_HIT` shaders.
//!
//! Because the early-out only changes *when* iteration stops, a `*_test` call
//! returns `true` exactly when the matching single-hit query returns `Some`
//! (and `overlap_any` matches a non-empty [`overlap_shape`]), so the fast path
//! never disagrees with the precise path.
//!
//! # Provenance
//!
//! These predicates reuse the crate's own per-shape intersection routines and
//! body iteration; they contain no Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_geometry::Ray;

use crate::collider::ColliderShape;
use crate::math::transform::Isometry;
use crate::query::{convex_sweep, overlap, ray, shape, QueryFilter};
use crate::world::PhysicsWorld;

impl PhysicsWorld {
    /// Returns `true` as soon as `ray` hits any accepted body, without finding
    /// the nearest one.
    ///
    /// Equivalent in result to `self.raycast(ray, filter).is_some()`, but stops
    /// at the first contact instead of scanning for the closest. Use it for
    /// line-of-sight and "is this clear?" checks.
    #[must_use]
    pub fn raycast_test(&self, ray: &Ray, filter: &QueryFilter) -> bool {
        self.query_bodies(filter)
            .iter()
            .any(|body| ray::raycast_shape(body.shape, &body.pose, ray).is_some())
    }

    /// Returns `true` as soon as a sphere of `radius` swept along `ray` touches
    /// any accepted body.
    ///
    /// Equivalent in result to `self.spherecast(ray, radius, filter).is_some()`,
    /// but stops at the first contact.
    #[must_use]
    pub fn spherecast_test(&self, ray: &Ray, radius: f32, filter: &QueryFilter) -> bool {
        self.query_bodies(filter)
            .iter()
            .any(|body| shape::spherecast_shape(body.shape, &body.pose, ray, radius).is_some())
    }

    /// Returns `true` as soon as the bounded convex `shape` posed at `pose` and
    /// swept along `motion` touches any accepted body.
    ///
    /// Equivalent in result to
    /// `self.shapecast(shape, pose, motion, filter).is_some()`, but stops at the
    /// first contact. A [`ColliderShape::Plane`] mover is rejected by the
    /// underlying sweep (an unbounded half-space cannot be swept), so this
    /// returns `false` for it.
    #[must_use]
    pub fn shapecast_test(
        &self,
        shape: &ColliderShape,
        pose: Isometry,
        motion: Vec3,
        filter: &QueryFilter,
    ) -> bool {
        self.query_bodies(filter).iter().any(|body| {
            convex_sweep::shapecast_shape(shape, &pose, body.shape, &body.pose, motion).is_some()
        })
    }

    /// Returns `true` as soon as the posed `shape` overlaps any accepted body.
    ///
    /// Equivalent in result to `!self.overlap_shape(shape, pose, filter).is_empty()`,
    /// but stops at the first overlapping body instead of collecting them all.
    /// Use it for spawn-point and placement validity checks.
    #[must_use]
    pub fn overlap_any(
        &self,
        shape: &ColliderShape,
        pose: &Isometry,
        filter: &QueryFilter,
    ) -> bool {
        self.query_bodies(filter)
            .iter()
            .any(|body| overlap::shapes_overlap(shape, pose, body.shape, &body.pose))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::ColliderShape;
    use crate::state::body::BodyDesc;

    /// Builds a world holding a single static unit sphere centred at `center`.
    fn world_with_sphere(center: Vec3) -> PhysicsWorld {
        let mut world = PhysicsWorld::default();
        let shape = world.shapes.insert(ColliderShape::Sphere { radius: 1.0 });
        world.spawn(BodyDesc::static_at(center).with_collider(shape));
        world
    }

    #[test]
    fn raycast_test_matches_single_hit_presence() {
        let world = world_with_sphere(Vec3::new(5.0, 0.0, 0.0));
        let hit = Ray::new(Vec3::ZERO, Vec3::X);
        let miss = Ray::new(Vec3::ZERO, Vec3::Y);

        assert!(world.raycast_test(&hit, &QueryFilter::ALL));
        assert_eq!(
            world.raycast_test(&hit, &QueryFilter::ALL),
            world.raycast(&hit, &QueryFilter::ALL).is_some()
        );
        assert!(!world.raycast_test(&miss, &QueryFilter::ALL));
        assert_eq!(
            world.raycast_test(&miss, &QueryFilter::ALL),
            world.raycast(&miss, &QueryFilter::ALL).is_some()
        );
    }

    #[test]
    fn raycast_test_true_on_empty_world_is_false() {
        let world = PhysicsWorld::default();
        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        assert!(!world.raycast_test(&ray, &QueryFilter::ALL));
    }

    #[test]
    fn spherecast_test_matches_single_hit_presence() {
        let world = world_with_sphere(Vec3::new(5.0, 0.0, 0.0));
        // A thin ray that would miss the sphere's centre line still hits once a
        // probe radius is swept, so test and single-hit must agree on both.
        let grazing = Ray::new(Vec3::new(0.0, 1.5, 0.0), Vec3::X);

        assert_eq!(
            world.spherecast_test(&grazing, 0.75, &QueryFilter::ALL),
            world
                .spherecast(&grazing, 0.75, &QueryFilter::ALL)
                .is_some()
        );
        let direct = Ray::new(Vec3::ZERO, Vec3::X);
        assert!(world.spherecast_test(&direct, 0.5, &QueryFilter::ALL));
    }

    #[test]
    fn shapecast_test_matches_single_hit_presence() {
        let world = world_with_sphere(Vec3::new(5.0, 0.0, 0.0));
        let box_shape = ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        let start = Isometry::from_translation(Vec3::ZERO);
        let toward = Vec3::new(10.0, 0.0, 0.0);
        let away = Vec3::new(0.0, 10.0, 0.0);

        assert!(world.shapecast_test(&box_shape, start, toward, &QueryFilter::ALL));
        assert_eq!(
            world.shapecast_test(&box_shape, start, toward, &QueryFilter::ALL),
            world
                .shapecast(&box_shape, start, toward, &QueryFilter::ALL)
                .is_some()
        );
        assert_eq!(
            world.shapecast_test(&box_shape, start, away, &QueryFilter::ALL),
            world
                .shapecast(&box_shape, start, away, &QueryFilter::ALL)
                .is_some()
        );
    }

    #[test]
    fn shapecast_test_rejects_plane_mover() {
        let world = world_with_sphere(Vec3::new(5.0, 0.0, 0.0));
        let plane = ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        };
        assert!(!world.shapecast_test(
            &plane,
            Isometry::from_translation(Vec3::ZERO),
            Vec3::X,
            &QueryFilter::ALL
        ));
    }

    #[test]
    fn overlap_any_matches_overlap_shape() {
        let world = world_with_sphere(Vec3::ZERO);
        let probe = ColliderShape::Sphere { radius: 1.0 };
        let touching = Isometry::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let apart = Isometry::from_translation(Vec3::new(100.0, 0.0, 0.0));

        assert_eq!(
            world.overlap_any(&probe, &touching, &QueryFilter::ALL),
            !world
                .overlap_shape(&probe, &touching, &QueryFilter::ALL)
                .is_empty()
        );
        assert!(world.overlap_any(&probe, &touching, &QueryFilter::ALL));
        assert_eq!(
            world.overlap_any(&probe, &apart, &QueryFilter::ALL),
            !world
                .overlap_shape(&probe, &apart, &QueryFilter::ALL)
                .is_empty()
        );
        assert!(!world.overlap_any(&probe, &apart, &QueryFilter::ALL));
    }
}
