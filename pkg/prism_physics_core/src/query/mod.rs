//! Spatial queries against a [`PhysicsWorld`]: raycasts, sphere sweeps,
//! general convex shape-casts, nearest-point projection, and overlap tests.
//!
//! These queries are *read-only*: they never mutate world state, so gameplay
//! code can freely probe geometry (line-of-sight checks, character-controller
//! sweeps, proximity triggers, click-to-select) between simulation steps.
//!
//! # Structure
//!
//! - [`ray`] holds ray/shape intersection primitives.
//! - [`shape`] holds swept-sphere (spherecast) primitives.
//! - [`convex_sweep`] holds general convex shape-cast (box/capsule/sphere
//!   sweep) primitives.
//! - [`multi`] holds the multi-hit `*_all` collectors (every body a probe
//!   touches, sorted near to far).
//! - [`project`] holds closest-point projection.
//! - [`overlap`] holds overlap predicates and world-space bounding boxes.
//!
//! Every query accepts a [`QueryFilter`] that selects which bodies participate,
//! and returns the nearest result (raycast, spherecast, shapecast, projection)
//! or the full set of matches (overlap).
//!
//! # Provenance
//!
//! The query dispatch and body iteration reuse the crate's own storage and
//! narrow-phase routines; the underlying geometric tests are standard and
//! contain no Unreal Engine source or derived code.

pub(crate) mod overlap;
pub(crate) mod project;
pub(crate) mod ray;
pub(crate) mod shape;
pub(crate) mod convex_sweep;
pub(crate) mod multi;

use crate::collider::{ColliderHandle, ColliderShape};
use crate::math::transform::Isometry;
use crate::state::body::BodyKind;
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;
use glam::Vec3;
use prism_physics_geometry::{Aabb, Ray};

/// Selects which bodies a spatial query considers.
///
/// The body-kind flags gate participation by simulation category, and
/// [`QueryFilter::exclude`] drops a single specific body (typically the caller's
/// own body, so a self-raycast does not report the source).
#[derive(Clone, Copy, Debug)]
pub struct QueryFilter {
    /// A body to skip entirely, if any.
    pub exclude: Option<BodyHandle>,
    /// Whether dynamic bodies participate.
    pub include_dynamic: bool,
    /// Whether kinematic bodies participate.
    pub include_kinematic: bool,
    /// Whether static bodies (including planes) participate.
    pub include_static: bool,
}

impl QueryFilter {
    /// A filter that accepts every body of every kind.
    pub const ALL: QueryFilter = QueryFilter {
        exclude: None,
        include_dynamic: true,
        include_kinematic: true,
        include_static: true,
    };

    /// Returns a copy of [`QueryFilter::ALL`] that excludes `body`.
    #[must_use]
    pub fn excluding(body: BodyHandle) -> QueryFilter {
        QueryFilter {
            exclude: Some(body),
            ..QueryFilter::ALL
        }
    }

    /// Returns a filter that accepts only dynamic bodies.
    #[must_use]
    pub fn dynamic_only() -> QueryFilter {
        QueryFilter {
            exclude: None,
            include_dynamic: true,
            include_kinematic: false,
            include_static: false,
        }
    }

    /// Returns `true` when a body of `kind` with handle `handle` passes.
    #[must_use]
    fn accepts(&self, handle: BodyHandle, kind: BodyKind) -> bool {
        if self.exclude == Some(handle) {
            return false;
        }
        match kind {
            BodyKind::Dynamic => self.include_dynamic,
            BodyKind::Kinematic => self.include_kinematic,
            BodyKind::Static => self.include_static,
        }
    }
}

impl Default for QueryFilter {
    fn default() -> Self {
        QueryFilter::ALL
    }
}

/// The result of a raycast: the body hit and the first-intersection geometry.
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// The body that was hit.
    pub body: BodyHandle,
    /// The collider shape that was hit.
    pub collider: ColliderHandle,
    /// Parametric distance along the ray direction to the hit.
    pub time_of_impact: f32,
    /// World-space intersection point.
    pub point: Vec3,
    /// Outward unit surface normal at the intersection.
    pub normal: Vec3,
}

/// The result of a spherecast: the body hit and the swept contact geometry.
#[derive(Clone, Copy, Debug)]
pub struct SweepHit {
    /// The body that was hit.
    pub body: BodyHandle,
    /// The collider shape that was hit.
    pub collider: ColliderHandle,
    /// Distance travelled by the sphere center before contact.
    pub time_of_impact: f32,
    /// World-space contact point on the shape surface.
    pub point: Vec3,
    /// Outward unit contact normal at the contact point.
    pub normal: Vec3,
}

/// The result of projecting a point onto the nearest body surface.
#[derive(Clone, Copy, Debug)]
pub struct PointProjection {
    /// The body whose surface is nearest.
    pub body: BodyHandle,
    /// The collider shape that was projected onto.
    pub collider: ColliderHandle,
    /// World-space nearest point on the shape surface.
    pub point: Vec3,
    /// Outward unit surface normal at [`PointProjection::point`].
    pub normal: Vec3,
    /// Unsigned distance from the query point to the surface.
    pub distance: f32,
    /// Whether the query point lies inside (or on) the shape.
    pub is_inside: bool,
}

/// A live, collidable body gathered for a query: its handle, collider, shape,
/// world pose, and simulation kind.
struct QueryBody<'a> {
    handle: BodyHandle,
    collider: ColliderHandle,
    shape: &'a ColliderShape,
    pose: Isometry,
}

impl PhysicsWorld {
    /// Gathers every live, collidable body accepted by `filter`.
    fn query_bodies(&self, filter: &QueryFilter) -> Vec<QueryBody<'_>> {
        let mut out = Vec::new();
        for slot in 0..self.bodies.slot_count() {
            let Some(handle) = self.bodies.handle_at_slot(slot) else {
                continue;
            };
            let kind = self.bodies.kind(handle).unwrap_or(BodyKind::Static);
            if !filter.accepts(handle, kind) {
                continue;
            }
            let Some(collider) = self.bodies.collider(handle) else {
                continue;
            };
            let Some(shape) = self.shapes.get(collider) else {
                continue;
            };
            let position = self.bodies.position(handle).unwrap_or(Vec3::ZERO);
            let orientation = self.bodies.orientation(handle).unwrap_or_default();
            out.push(QueryBody {
                handle,
                collider,
                shape,
                pose: Isometry::new(position, orientation),
            });
        }
        out
    }

    /// Casts `ray` against every accepted body and returns the nearest hit.
    #[must_use]
    pub fn raycast(&self, ray: &Ray, filter: &QueryFilter) -> Option<RayHit> {
        let mut best: Option<RayHit> = None;
        for body in self.query_bodies(filter) {
            if let Some(hit) = ray::raycast_shape(body.shape, &body.pose, ray) {
                let candidate = RayHit {
                    body: body.handle,
                    collider: body.collider,
                    time_of_impact: hit.time_of_impact,
                    point: hit.point,
                    normal: hit.normal,
                };
                if best.is_none_or(|b| candidate.time_of_impact < b.time_of_impact) {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    /// Sweeps a sphere of `radius` along `ray` and returns the nearest contact.
    #[must_use]
    pub fn spherecast(&self, ray: &Ray, radius: f32, filter: &QueryFilter) -> Option<SweepHit> {
        let mut best: Option<SweepHit> = None;
        for body in self.query_bodies(filter) {
            if let Some(hit) = shape::spherecast_shape(body.shape, &body.pose, ray, radius) {
                let candidate = SweepHit {
                    body: body.handle,
                    collider: body.collider,
                    time_of_impact: hit.time_of_impact,
                    point: hit.point,
                    normal: hit.normal,
                };
                if best.is_none_or(|b| candidate.time_of_impact < b.time_of_impact) {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    /// Sweeps an arbitrary bounded convex `shape` (sphere, box, or capsule)
    /// posed at `pose` along `motion` and returns the nearest contact.
    ///
    /// Unlike [`spherecast`](Self::spherecast), which can only sweep a sphere,
    /// this respects the mover's and every target's true shape and orientation,
    /// so swept box/capsule character controllers, cameras, and projectiles do
    /// not tunnel. `motion` is the full world-space displacement to test; the
    /// returned [`SweepHit::time_of_impact`] is the distance the mover's origin
    /// travels before contact.
    ///
    /// A [`ColliderShape::Plane`] mover is rejected (returns [`None`]): a plane
    /// is an unbounded half-space and can only be swept *into*, never swept.
    #[must_use]
    pub fn shapecast(
        &self,
        shape: &ColliderShape,
        pose: Isometry,
        motion: Vec3,
        filter: &QueryFilter,
    ) -> Option<SweepHit> {
        let mut best: Option<SweepHit> = None;
        for body in self.query_bodies(filter) {
            if let Some(hit) =
                convex_sweep::shapecast_shape(shape, &pose, body.shape, &body.pose, motion)
            {
                let candidate = SweepHit {
                    body: body.handle,
                    collider: body.collider,
                    time_of_impact: hit.time_of_impact,
                    point: hit.point,
                    normal: hit.normal,
                };
                if best.is_none_or(|b| candidate.time_of_impact < b.time_of_impact) {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    /// Projects `point` onto the nearest accepted body surface.
    #[must_use]
    pub fn project_point(&self, point: Vec3, filter: &QueryFilter) -> Option<PointProjection> {
        let mut best: Option<PointProjection> = None;
        for body in self.query_bodies(filter) {
            let proj = project::project_point_shape(body.shape, &body.pose, point);
            let candidate = PointProjection {
                body: body.handle,
                collider: body.collider,
                point: proj.point,
                normal: proj.normal,
                distance: proj.distance,
                is_inside: proj.is_inside,
            };
            if best.is_none_or(|b| candidate.distance < b.distance) {
                best = Some(candidate);
            }
        }
        best
    }

    /// Returns every accepted body whose shape overlaps the posed `shape`.
    #[must_use]
    pub fn overlap_shape(
        &self,
        shape: &ColliderShape,
        pose: &Isometry,
        filter: &QueryFilter,
    ) -> Vec<BodyHandle> {
        let mut out = Vec::new();
        for body in self.query_bodies(filter) {
            if overlap::shapes_overlap(shape, pose, body.shape, &body.pose) {
                out.push(body.handle);
            }
        }
        out
    }

    /// Returns every accepted body whose shape overlaps the given sphere.
    #[must_use]
    pub fn overlap_sphere(
        &self,
        center: Vec3,
        radius: f32,
        filter: &QueryFilter,
    ) -> Vec<BodyHandle> {
        let sphere = ColliderShape::Sphere { radius };
        self.overlap_shape(&sphere, &Isometry::from_translation(center), filter)
    }

    /// Returns every accepted body that contains `point`.
    #[must_use]
    pub fn overlap_point(&self, point: Vec3, filter: &QueryFilter) -> Vec<BodyHandle> {
        let mut out = Vec::new();
        for body in self.query_bodies(filter) {
            if project::project_point_shape(body.shape, &body.pose, point).is_inside {
                out.push(body.handle);
            }
        }
        out
    }

    /// Returns every accepted body whose world-space bounding box intersects
    /// `aabb`.
    #[must_use]
    pub fn overlap_aabb(&self, aabb: &Aabb, filter: &QueryFilter) -> Vec<BodyHandle> {
        let mut out = Vec::new();
        for body in self.query_bodies(filter) {
            if overlap::world_shape_aabb(body.shape, &body.pose).intersects(aabb) {
                out.push(body.handle);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::ColliderShape;
    use crate::state::body::BodyDesc;
    use glam::Quat;

    /// Spawns a static body carrying `shape` at `pos` and returns its handle.
    fn spawn_shape(world: &mut PhysicsWorld, shape: ColliderShape, pos: Vec3) -> BodyHandle {
        let collider = world.shapes.insert(shape);
        world.spawn(BodyDesc::static_at(pos).with_collider(collider))
    }

    #[test]
    fn raycast_hits_nearest_sphere() {
        let mut world = PhysicsWorld::default();
        let far = spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 0.0, 8.0),
        );
        let near = spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 0.0, 4.0),
        );
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let hit = world.raycast(&ray, &QueryFilter::ALL).expect("hit");
        assert_eq!(hit.body, near);
        assert_ne!(hit.body, far);
        assert!(
            (hit.time_of_impact - 3.0).abs() < 1e-4,
            "toi {}",
            hit.time_of_impact
        );
        assert!((hit.point - Vec3::new(0.0, 0.0, 3.0)).length() < 1e-4);
        assert!(
            hit.normal.dot(Vec3::NEG_Z) > 0.99,
            "normal {:?}",
            hit.normal
        );
    }

    #[test]
    fn raycast_pointing_away_misses() {
        let mut world = PhysicsWorld::default();
        spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 0.0, 4.0),
        );
        let ray = Ray::new(Vec3::ZERO, Vec3::NEG_Z);
        assert!(world.raycast(&ray, &QueryFilter::ALL).is_none());
    }

    #[test]
    fn raycast_respects_tmax() {
        let mut world = PhysicsWorld::default();
        spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 0.0, 8.0),
        );
        let ray = Ray::with_tmax(Vec3::ZERO, Vec3::Z, 5.0);
        assert!(world.raycast(&ray, &QueryFilter::ALL).is_none());
    }

    #[test]
    fn raycast_hits_plane_from_above() {
        let mut world = PhysicsWorld::default();
        spawn_shape(
            &mut world,
            ColliderShape::Plane {
                normal: Vec3::Y,
                offset: 0.0,
            },
            Vec3::ZERO,
        );
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y);
        let hit = world.raycast(&ray, &QueryFilter::ALL).expect("hit");
        assert!((hit.time_of_impact - 5.0).abs() < 1e-4);
        assert!(hit.normal.dot(Vec3::Y) > 0.99);
    }

    #[test]
    fn raycast_hits_cuboid_face() {
        let mut world = PhysicsWorld::default();
        spawn_shape(
            &mut world,
            ColliderShape::Cuboid {
                half_extents: Vec3::splat(1.0),
            },
            Vec3::new(0.0, 0.0, 5.0),
        );
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let hit = world.raycast(&ray, &QueryFilter::ALL).expect("hit");
        assert!((hit.time_of_impact - 4.0).abs() < 1e-4);
        assert!(hit.normal.dot(Vec3::NEG_Z) > 0.99);
    }

    #[test]
    fn raycast_hits_rotated_cuboid() {
        let mut world = PhysicsWorld::default();
        let body = spawn_shape(
            &mut world,
            ColliderShape::Cuboid {
                half_extents: Vec3::new(1.0, 1.0, 1.0),
            },
            Vec3::new(0.0, 0.0, 5.0),
        );
        world
            .bodies
            .set_orientation(body, Quat::from_rotation_y(core::f32::consts::FRAC_PI_4));
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let hit = world.raycast(&ray, &QueryFilter::ALL).expect("hit");
        // Rotated 45 degrees, the near corner is sqrt(2) closer to the origin.
        let expected = 5.0 - core::f32::consts::SQRT_2;
        assert!(
            (hit.time_of_impact - expected).abs() < 1e-3,
            "toi {}",
            hit.time_of_impact
        );
    }

    #[test]
    fn raycast_hits_capsule_side() {
        let mut world = PhysicsWorld::default();
        spawn_shape(
            &mut world,
            ColliderShape::Capsule {
                half_height: 1.0,
                radius: 0.5,
            },
            Vec3::new(0.0, 0.0, 5.0),
        );
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let hit = world.raycast(&ray, &QueryFilter::ALL).expect("hit");
        assert!(
            (hit.time_of_impact - 4.5).abs() < 1e-4,
            "toi {}",
            hit.time_of_impact
        );
        assert!(hit.normal.dot(Vec3::NEG_Z) > 0.99);
    }

    #[test]
    fn filter_can_exclude_a_body() {
        let mut world = PhysicsWorld::default();
        let front = spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 0.0, 4.0),
        );
        let back = spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 0.0, 8.0),
        );
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let hit = world
            .raycast(&ray, &QueryFilter::excluding(front))
            .expect("hit");
        assert_eq!(hit.body, back);
    }

    #[test]
    fn spherecast_sphere_stops_before_contact() {
        let mut world = PhysicsWorld::default();
        spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 0.0, 5.0),
        );
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let hit = world.spherecast(&ray, 0.5, &QueryFilter::ALL).expect("hit");
        // Contact when the moving center is at radius 1 + 0.5 from the target.
        assert!(
            (hit.time_of_impact - 3.5).abs() < 1e-4,
            "toi {}",
            hit.time_of_impact
        );
        assert!(hit.normal.dot(Vec3::NEG_Z) > 0.99);
        assert!((hit.point - Vec3::new(0.0, 0.0, 4.0)).length() < 1e-3);
    }

    #[test]
    fn spherecast_plane_offsets_by_radius() {
        let mut world = PhysicsWorld::default();
        spawn_shape(
            &mut world,
            ColliderShape::Plane {
                normal: Vec3::Y,
                offset: 0.0,
            },
            Vec3::ZERO,
        );
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y);
        let hit = world.spherecast(&ray, 0.5, &QueryFilter::ALL).expect("hit");
        assert!(
            (hit.time_of_impact - 4.5).abs() < 1e-4,
            "toi {}",
            hit.time_of_impact
        );
        assert!(hit.normal.dot(Vec3::Y) > 0.99);
    }

    #[test]
    fn spherecast_cuboid_face() {
        let mut world = PhysicsWorld::default();
        spawn_shape(
            &mut world,
            ColliderShape::Cuboid {
                half_extents: Vec3::splat(1.0),
            },
            Vec3::new(0.0, 0.0, 5.0),
        );
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let hit = world.spherecast(&ray, 0.5, &QueryFilter::ALL).expect("hit");
        assert!(
            (hit.time_of_impact - 3.5).abs() < 1e-4,
            "toi {}",
            hit.time_of_impact
        );
        assert!(hit.normal.dot(Vec3::NEG_Z) > 0.99);
    }

    #[test]
    fn spherecast_cuboid_corner() {
        let mut world = PhysicsWorld::default();
        // Box corner nearest origin is at (1,1,1) offset by center (3,3,3) -> (2,2,2).
        spawn_shape(
            &mut world,
            ColliderShape::Cuboid {
                half_extents: Vec3::splat(1.0),
            },
            Vec3::splat(3.0),
        );
        let dir = Vec3::splat(1.0).normalize();
        let ray = Ray::new(Vec3::ZERO, dir);
        let hit = world
            .spherecast(&ray, 0.25, &QueryFilter::ALL)
            .expect("hit");
        // The corner sits at distance sqrt(12) ~= 3.4641; contact removes 0.25.
        let corner_dist = Vec3::splat(2.0).length();
        assert!(
            (hit.time_of_impact - (corner_dist - 0.25)).abs() < 1e-3,
            "toi {}",
            hit.time_of_impact
        );
    }

    #[test]
    fn project_point_outside_and_inside_sphere() {
        let mut world = PhysicsWorld::default();
        spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 1.0 },
            Vec3::ZERO,
        );
        let outside = world
            .project_point(Vec3::new(3.0, 0.0, 0.0), &QueryFilter::ALL)
            .expect("proj");
        assert!(!outside.is_inside);
        assert!((outside.distance - 2.0).abs() < 1e-4);
        assert!((outside.point - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-4);
        let inside = world
            .project_point(Vec3::new(0.2, 0.0, 0.0), &QueryFilter::ALL)
            .expect("proj");
        assert!(inside.is_inside);
    }

    #[test]
    fn project_point_onto_cuboid_face() {
        let mut world = PhysicsWorld::default();
        spawn_shape(
            &mut world,
            ColliderShape::Cuboid {
                half_extents: Vec3::splat(1.0),
            },
            Vec3::ZERO,
        );
        let proj = world
            .project_point(Vec3::new(2.0, 0.5, 0.0), &QueryFilter::ALL)
            .expect("proj");
        assert!(!proj.is_inside);
        assert!((proj.point - Vec3::new(1.0, 0.5, 0.0)).length() < 1e-4);
        assert!(proj.normal.dot(Vec3::X) > 0.99);
    }

    #[test]
    fn overlap_sphere_reports_touching_bodies() {
        let mut world = PhysicsWorld::default();
        let hit = spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(1.0, 0.0, 0.0),
        );
        let miss = spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(10.0, 0.0, 0.0),
        );
        let found = world.overlap_sphere(Vec3::ZERO, 1.0, &QueryFilter::ALL);
        assert!(found.contains(&hit));
        assert!(!found.contains(&miss));
    }

    #[test]
    fn overlap_point_reports_containing_bodies() {
        let mut world = PhysicsWorld::default();
        let box_body = spawn_shape(
            &mut world,
            ColliderShape::Cuboid {
                half_extents: Vec3::splat(1.0),
            },
            Vec3::ZERO,
        );
        let found = world.overlap_point(Vec3::new(0.5, 0.5, 0.5), &QueryFilter::ALL);
        assert_eq!(found, vec![box_body]);
        assert!(world
            .overlap_point(Vec3::new(5.0, 0.0, 0.0), &QueryFilter::ALL)
            .is_empty());
    }

    #[test]
    fn overlap_aabb_reports_bodies_in_region() {
        let mut world = PhysicsWorld::default();
        let inside = spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 0.5 },
            Vec3::new(0.0, 0.0, 0.0),
        );
        let outside = spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 0.5 },
            Vec3::new(10.0, 0.0, 0.0),
        );
        let region = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
        let found = world.overlap_aabb(&region, &QueryFilter::ALL);
        assert!(found.contains(&inside));
        assert!(!found.contains(&outside));
    }

    #[test]
    fn dynamic_only_filter_skips_static() {
        let mut world = PhysicsWorld::default();
        spawn_shape(
            &mut world,
            ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 0.0, 4.0),
        );
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        assert!(world.raycast(&ray, &QueryFilter::dynamic_only()).is_none());
    }
}
