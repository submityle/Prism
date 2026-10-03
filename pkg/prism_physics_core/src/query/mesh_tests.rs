//! End-to-end scene-query goldens for mesh colliders.
//!
//! The per-shape unit tests in the sibling modules exercise the registry-aware
//! dispatch in isolation. These integration goldens drive the *public*
//! [`PhysicsWorld`] query surface — [`raycast`](PhysicsWorld::raycast),
//! [`spherecast`](PhysicsWorld::spherecast),
//! [`shapecast`](PhysicsWorld::shapecast),
//! [`project_point`](PhysicsWorld::project_point), and
//! [`overlap_shape`](PhysicsWorld::overlap_shape) — against bodies carrying a
//! [`ColliderShape::ConvexHull`] or [`ColliderShape::TriangleMesh`], so a
//! regression anywhere along `world → query_bodies → dispatch → mesh` is caught
//! by a deterministic, hand-computed expectation rather than only by the
//! lower-level units.
//!
//! Every box used here is an axis-aligned cube of half-extent 1 so the expected
//! time-of-impact, contact point, and inside/outside classification are exact
//! closed-form values.
//!
//! # Provenance
//!
//! These tests compose the crate's own public API and locally built mesh data;
//! they contain no Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_geometry::Ray;

use crate::collider::{ColliderShape, ConvexMeshData, TriMeshData};
use crate::math::transform::Isometry;
use crate::query::QueryFilter;
use crate::state::body::BodyDesc;
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;

/// Loose tolerance for GJK/BVH-derived positions (world units).
const POS_TOL: f32 = 2.0e-2;

/// The eight corners of an axis-aligned box of the given half extents.
fn box_corners(he: Vec3) -> [Vec3; 8] {
    [
        Vec3::new(-he.x, -he.y, -he.z),
        Vec3::new(he.x, -he.y, -he.z),
        Vec3::new(he.x, he.y, -he.z),
        Vec3::new(-he.x, he.y, -he.z),
        Vec3::new(-he.x, -he.y, he.z),
        Vec3::new(he.x, -he.y, he.z),
        Vec3::new(he.x, he.y, he.z),
        Vec3::new(-he.x, he.y, he.z),
    ]
}

/// Builds closed-box triangle-mesh data with outward-facing winding.
fn box_tri_mesh(he: Vec3) -> TriMeshData {
    let verts = box_corners(he).to_vec();
    // Twelve triangles, two per face, wound counter-clockwise seen from outside
    // so the geometry's face normals point out of the solid.
    let indices = vec![
        [1, 2, 6],
        [1, 6, 5], // +X
        [0, 4, 7],
        [0, 7, 3], // -X
        [3, 7, 6],
        [3, 6, 2], // +Y
        [0, 1, 5],
        [0, 5, 4], // -Y
        [4, 5, 6],
        [4, 6, 7], // +Z
        [0, 3, 2],
        [0, 2, 1], // -Z
    ];
    TriMeshData::new(verts, indices)
}

/// Spawns a static body carrying a convex-hull box of half extents `he`.
fn spawn_convex_box(world: &mut PhysicsWorld, he: Vec3, pos: Vec3) -> BodyHandle {
    let handle = world
        .shapes
        .insert_convex_mesh(ConvexMeshData::from_box(he));
    let shape = world
        .shapes
        .convex_hull_shape(handle)
        .expect("convex hull shape");
    let collider = world.shapes.insert(shape);
    world.spawn(BodyDesc::static_at(pos).with_collider(collider))
}

/// Spawns a static body carrying a triangle-mesh box of half extents `he`.
fn spawn_trimesh_box(world: &mut PhysicsWorld, he: Vec3, pos: Vec3) -> BodyHandle {
    let handle = world.shapes.insert_tri_mesh(box_tri_mesh(he));
    let shape = world.shapes.tri_mesh_shape(handle).expect("tri mesh shape");
    let collider = world.shapes.insert(shape);
    world.spawn(BodyDesc::static_at(pos).with_collider(collider))
}

#[test]
fn raycast_hits_convex_hull_front_face() {
    let mut world = PhysicsWorld::default();
    let body = spawn_convex_box(&mut world, Vec3::ONE, Vec3::new(0.0, 0.0, 5.0));

    let ray = Ray::new(Vec3::ZERO, Vec3::Z);
    let hit = world.raycast(&ray, &QueryFilter::ALL).expect("hit");

    assert_eq!(hit.body, body);
    assert!(
        (hit.time_of_impact - 4.0).abs() < POS_TOL,
        "toi {}",
        hit.time_of_impact
    );
    assert!((hit.point - Vec3::new(0.0, 0.0, 4.0)).length() < POS_TOL);
    assert!(hit.normal.dot(Vec3::NEG_Z) > 0.9, "normal {:?}", hit.normal);
}

#[test]
fn raycast_hits_trimesh_front_face() {
    let mut world = PhysicsWorld::default();
    let body = spawn_trimesh_box(&mut world, Vec3::ONE, Vec3::new(0.0, 0.0, 5.0));

    let ray = Ray::new(Vec3::ZERO, Vec3::Z);
    let hit = world.raycast(&ray, &QueryFilter::ALL).expect("hit");

    assert_eq!(hit.body, body);
    assert!(
        (hit.time_of_impact - 4.0).abs() < POS_TOL,
        "toi {}",
        hit.time_of_impact
    );
    assert!((hit.point.z - 4.0).abs() < POS_TOL, "point {:?}", hit.point);
}

#[test]
fn raycast_pointing_away_from_trimesh_misses() {
    let mut world = PhysicsWorld::default();
    spawn_trimesh_box(&mut world, Vec3::ONE, Vec3::new(0.0, 0.0, 5.0));

    let ray = Ray::new(Vec3::ZERO, Vec3::NEG_Z);
    assert!(world.raycast(&ray, &QueryFilter::ALL).is_none());
}

#[test]
fn raycast_picks_nearest_of_convex_and_trimesh() {
    let mut world = PhysicsWorld::default();
    let near = spawn_trimesh_box(&mut world, Vec3::ONE, Vec3::new(0.0, 0.0, 4.0));
    let far = spawn_convex_box(&mut world, Vec3::ONE, Vec3::new(0.0, 0.0, 9.0));

    let ray = Ray::new(Vec3::ZERO, Vec3::Z);
    let hit = world.raycast(&ray, &QueryFilter::ALL).expect("hit");

    assert_eq!(hit.body, near);
    assert_ne!(hit.body, far);
    assert!((hit.time_of_impact - 3.0).abs() < POS_TOL);
}

#[test]
fn project_point_classifies_inside_and_outside_convex_hull() {
    let mut world = PhysicsWorld::default();
    spawn_convex_box(&mut world, Vec3::ONE, Vec3::new(0.0, 0.0, 0.0));

    let inside = world
        .project_point(Vec3::new(0.2, -0.1, 0.3), &QueryFilter::ALL)
        .expect("projection");
    assert!(inside.is_inside, "expected interior classification");

    let outside = world
        .project_point(Vec3::new(0.0, 0.0, 3.0), &QueryFilter::ALL)
        .expect("projection");
    assert!(!outside.is_inside, "expected exterior classification");
    assert!(
        (outside.distance - 2.0).abs() < POS_TOL,
        "distance {}",
        outside.distance
    );
    assert!((outside.point - Vec3::new(0.0, 0.0, 1.0)).length() < POS_TOL);
}

#[test]
fn overlap_shape_sphere_touches_trimesh_box() {
    let mut world = PhysicsWorld::default();
    let body = spawn_trimesh_box(&mut world, Vec3::ONE, Vec3::ZERO);

    let probe = ColliderShape::Sphere { radius: 0.5 };

    // Centre 1.2 from the origin leaves a 0.2 gap to the box face, inside the
    // 0.5 probe radius: overlap.
    let touching = Isometry::from_translation(Vec3::new(1.2, 0.0, 0.0));
    assert_eq!(
        world.overlap_shape(&probe, &touching, &QueryFilter::ALL),
        vec![body]
    );

    // Centre 2.0 away leaves a 1.0 gap, outside the probe radius: clear.
    let clear = Isometry::from_translation(Vec3::new(2.0, 0.0, 0.0));
    assert!(world
        .overlap_shape(&probe, &clear, &QueryFilter::ALL)
        .is_empty());
}

#[test]
fn spherecast_sweeps_into_trimesh_box() {
    let mut world = PhysicsWorld::default();
    let body = spawn_trimesh_box(&mut world, Vec3::ONE, Vec3::new(0.0, 0.0, 5.0));

    let ray = Ray::new(Vec3::ZERO, Vec3::Z);
    let hit = world
        .spherecast(&ray, 0.5, &QueryFilter::ALL)
        .expect("sphere hit");

    assert_eq!(hit.body, body);
    // The sphere surface reaches the box face at z = 4, so its centre travels
    // 4 - 0.5 = 3.5 units.
    assert!(
        (hit.time_of_impact - 3.5).abs() < POS_TOL,
        "toi {}",
        hit.time_of_impact
    );
}

#[test]
fn shapecast_box_into_trimesh_box() {
    let mut world = PhysicsWorld::default();
    let body = spawn_trimesh_box(&mut world, Vec3::ONE, Vec3::new(0.0, 0.0, 6.0));

    let mover = ColliderShape::Cuboid {
        half_extents: Vec3::ONE,
    };
    let hit = world
        .shapecast(
            &mover,
            Isometry::from_translation(Vec3::ZERO),
            Vec3::new(0.0, 0.0, 10.0),
            &QueryFilter::ALL,
        )
        .expect("box sweep hit");

    assert_eq!(hit.body, body);
    // Mover front face starts at z = 1, target front face at z = 5, so the
    // mover origin travels 4 units before contact.
    assert!(
        (hit.time_of_impact - 4.0).abs() < POS_TOL,
        "toi {}",
        hit.time_of_impact
    );
}
