//! Multi-hit spatial queries: every body a probe touches, sorted near to far.
//!
//! The single-hit queries ([`raycast`](crate::world::PhysicsWorld::raycast),
//! [`spherecast`](crate::world::PhysicsWorld::spherecast),
//! [`shapecast`](crate::world::PhysicsWorld::shapecast)) return only the closest
//! contact, which is all a line-of-sight or character-controller step needs.
//! Gameplay systems such as penetrating projectiles, cover/visibility scans,
//! laser "paint-through" effects, and trigger volumes along a path instead need
//! *every* body the probe crosses. These `*_all` variants mirror UE's
//! `RaycastMulti`/`SweepMulti`, `PhysX`'s multi-hit scene queries, and Jolt's
//! `CollectAllHits`.
//!
//! Each returned vector is sorted by ascending `time_of_impact` (distance along
//! the probe), so index `0` matches the corresponding single-hit query and
//! callers can early-out once a hit is beyond a cutoff. Bodies are visited in
//! deterministic storage-slot order and the sort is stable, so equal-distance
//! hits keep that order and identical inputs yield an identical vector.
//!
//! # Provenance
//!
//! These collectors reuse the crate's own per-shape intersection routines and
//! body iteration; they contain no Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_geometry::Ray;

use crate::collider::ColliderShape;
use crate::math::transform::Isometry;
use crate::query::{convex_sweep, ray, shape, QueryFilter, RayHit, SweepHit};
use crate::world::PhysicsWorld;

/// Orders hits near-to-far by `time_of_impact`, keeping ties in slot order.
///
/// `f32` is only `PartialOrd`; a `NaN` time of impact (which the intersection
/// routines never emit) sorts last rather than panicking.
fn sort_by_toi<T, F>(hits: &mut [T], toi: F)
where
    F: Fn(&T) -> f32,
{
    hits.sort_by(|a, b| {
        toi(a)
            .partial_cmp(&toi(b))
            .unwrap_or(std::cmp::Ordering::Greater)
    });
}

impl PhysicsWorld {
    /// Casts `ray` against every accepted body and returns **all** hits, sorted
    /// nearest first.
    ///
    /// The nearest element equals [`raycast`](Self::raycast); an empty vector
    /// means the ray missed everything.
    #[must_use]
    pub fn raycast_all(&self, ray: &Ray, filter: &QueryFilter) -> Vec<RayHit> {
        let mut hits = Vec::new();
        for body in self.query_bodies(filter) {
            if let Some(hit) = ray::raycast_shape(body.shape, &body.pose, ray) {
                hits.push(RayHit {
                    body: body.handle,
                    collider: body.collider,
                    time_of_impact: hit.time_of_impact,
                    point: hit.point,
                    normal: hit.normal,
                });
            }
        }
        sort_by_toi(&mut hits, |h| h.time_of_impact);
        hits
    }

    /// Sweeps a sphere of `radius` along `ray` against every accepted body and
    /// returns **all** contacts, sorted nearest first.
    ///
    /// The nearest element equals [`spherecast`](Self::spherecast).
    #[must_use]
    pub fn spherecast_all(&self, ray: &Ray, radius: f32, filter: &QueryFilter) -> Vec<SweepHit> {
        let mut hits = Vec::new();
        for body in self.query_bodies(filter) {
            if let Some(hit) = shape::spherecast_shape(body.shape, &body.pose, ray, radius) {
                hits.push(SweepHit {
                    body: body.handle,
                    collider: body.collider,
                    time_of_impact: hit.time_of_impact,
                    point: hit.point,
                    normal: hit.normal,
                });
            }
        }
        sort_by_toi(&mut hits, |h| h.time_of_impact);
        hits
    }

    /// Sweeps an arbitrary bounded convex `shape` posed at `pose` along `motion`
    /// against every accepted body and returns **all** contacts, sorted nearest
    /// first.
    ///
    /// The nearest element equals [`shapecast`](Self::shapecast).
    #[must_use]
    pub fn shapecast_all(
        &self,
        shape: &ColliderShape,
        pose: Isometry,
        motion: Vec3,
        filter: &QueryFilter,
    ) -> Vec<SweepHit> {
        let mut hits = Vec::new();
        for body in self.query_bodies(filter) {
            if let Some(hit) =
                convex_sweep::shapecast_shape(shape, &pose, body.shape, &body.pose, motion)
            {
                hits.push(SweepHit {
                    body: body.handle,
                    collider: body.collider,
                    time_of_impact: hit.time_of_impact,
                    point: hit.point,
                    normal: hit.normal,
                });
            }
        }
        sort_by_toi(&mut hits, |h| h.time_of_impact);
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::ColliderShape;
    use crate::state::body::BodyDesc;
    use crate::state::handle::BodyHandle;
    use glam::Quat;

    /// Spawns a static body carrying `shape` at `pos` and returns its handle.
    fn spawn_shape(world: &mut PhysicsWorld, shape: ColliderShape, pos: Vec3) -> BodyHandle {
        let collider = world.shapes.insert(shape);
        world.spawn(BodyDesc::static_at(pos).with_collider(collider))
    }

    /// Builds a world with three unit spheres strung along +X at x = 2, 5, 9.
    fn three_spheres() -> PhysicsWorld {
        let mut world = PhysicsWorld::default();
        for x in [2.0f32, 5.0, 9.0] {
            spawn_shape(
                &mut world,
                ColliderShape::Sphere { radius: 1.0 },
                Vec3::new(x, 0.0, 0.0),
            );
        }
        world
    }

    #[test]
    fn raycast_all_returns_every_hit_sorted() {
        let world = three_spheres();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
        let hits = world.raycast_all(&ray, &QueryFilter::ALL);
        assert_eq!(hits.len(), 3, "the ray threads all three spheres");
        // Entry faces at x = 1, 4, 8 -> distances 6, 9, 13 from origin -5.
        assert!(hits[0].time_of_impact < hits[1].time_of_impact);
        assert!(hits[1].time_of_impact < hits[2].time_of_impact);
        assert!((hits[0].time_of_impact - 6.0).abs() < 1.0e-3);
    }

    #[test]
    fn raycast_all_nearest_matches_single_hit() {
        let world = three_spheres();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
        let all = world.raycast_all(&ray, &QueryFilter::ALL);
        let single = world.raycast(&ray, &QueryFilter::ALL).expect("single hit");
        assert_eq!(all[0].body, single.body);
        assert!((all[0].time_of_impact - single.time_of_impact).abs() < 1.0e-6);
    }

    #[test]
    fn raycast_all_empty_on_miss() {
        let world = three_spheres();
        // Parallel ray well above the spheres.
        let ray = Ray::new(Vec3::new(-5.0, 50.0, 0.0), Vec3::X);
        assert!(world.raycast_all(&ray, &QueryFilter::ALL).is_empty());
    }

    #[test]
    fn shapecast_all_collects_sorted_contacts() {
        let world = three_spheres();
        let mover = ColliderShape::Sphere { radius: 0.5 };
        let hits = world.shapecast_all(
            &mover,
            Isometry::new(Vec3::new(-5.0, 0.0, 0.0), Quat::IDENTITY),
            Vec3::new(100.0, 0.0, 0.0),
            &QueryFilter::ALL,
        );
        assert_eq!(hits.len(), 3, "the swept sphere reaches all three");
        assert!(hits[0].time_of_impact <= hits[1].time_of_impact);
        assert!(hits[1].time_of_impact <= hits[2].time_of_impact);
        let single = world
            .shapecast(
                &mover,
                Isometry::new(Vec3::new(-5.0, 0.0, 0.0), Quat::IDENTITY),
                Vec3::new(100.0, 0.0, 0.0),
                &QueryFilter::ALL,
            )
            .expect("single shapecast");
        assert_eq!(hits[0].body, single.body);
    }

    #[test]
    fn spherecast_all_collects_sorted_contacts() {
        let world = three_spheres();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
        let hits = world.spherecast_all(&ray, 0.5, &QueryFilter::ALL);
        assert_eq!(hits.len(), 3, "the swept sphere reaches all three");
        assert!(hits[0].time_of_impact <= hits[1].time_of_impact);
        assert!(hits[1].time_of_impact <= hits[2].time_of_impact);
    }
}
