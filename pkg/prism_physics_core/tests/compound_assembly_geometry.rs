//! Integration coverage for true compound geometry and assembly lifecycle.
//!
//! These tests intentionally probe points inside the cached compound AABB but
//! outside every real child. A hull/AABB collision shortcut would fill those
//! gaps; leaf traversal must leave them empty.

use glam::{Quat, Vec3};
use prism_physics_core::ccd::resolve_ccd as resolve_rigid_ccd;
use prism_physics_core::collider::{CompoundChild, TriMeshData};
use prism_physics_core::joint::{DistanceJoint, JointDesc, JointKind};
use prism_physics_core::pipeline::detect_contacts;
use prism_physics_core::query::QueryFilter;
use prism_physics_core::soft::rigid_coupling::ClothRigidCouplingConfig;
use prism_physics_core::state::body::{BodyDesc, MassProperties};
use prism_physics_core::{ColliderShape, Isometry, PhysicsWorld, WorldConfig};
use prism_physics_geometry::Ray;

fn sphere(world: &mut PhysicsWorld, radius: f32) -> prism_physics_core::ColliderHandle {
    world.shapes.insert(ColliderShape::Sphere { radius })
}

fn dynamic_body(
    world: &mut PhysicsWorld,
    position: Vec3,
    collider: prism_physics_core::ColliderHandle,
    velocity: Vec3,
) -> prism_physics_core::BodyHandle {
    world.spawn(
        BodyDesc::dynamic_at(position)
            .with_collider(collider)
            .with_linear_velocity(velocity)
            .with_mass_properties(MassProperties {
                inv_mass: 1.0,
                inv_inertia: Vec3::ONE,
            }),
    )
}

#[test]
fn compound_gap_and_curved_leaf_are_not_replaced_by_aabb_geometry() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let leaf = sphere(&mut world, 0.5);
    let compound = world
        .shapes
        .insert_compound(&[
            CompoundChild {
                collider: leaf,
                local_pose: Isometry::from_translation(Vec3::new(-1.25, 0.0, 0.0)),
            },
            CompoundChild {
                collider: leaf,
                local_pose: Isometry::from_translation(Vec3::new(1.25, 0.0, 0.0)),
            },
        ])
        .expect("valid compound");
    world.spawn(BodyDesc::static_at(Vec3::ZERO).with_collider(compound));

    // The origin lies inside the compound's cached AABB but in the true gap.
    assert!(world
        .overlap_sphere(Vec3::ZERO, 0.1, &QueryFilter::ALL)
        .is_empty());
    assert!(world
        .overlap_point(Vec3::ZERO, &QueryFilter::ALL)
        .is_empty());
    // The point below is inside the left child's AABB corner but outside the
    // sphere itself; the curved leaf must remain curved.
    assert!(world
        .overlap_sphere(Vec3::new(-1.75, 0.45, 0.0), 0.02, &QueryFilter::ALL)
        .is_empty());

    let projection = world
        .project_point(Vec3::ZERO, &QueryFilter::ALL)
        .expect("compound projection");
    assert!(!projection.is_inside);
    assert!((projection.distance - 0.75).abs() < 1.0e-5);
}

#[test]
fn nested_child_transforms_are_composed_before_query_dispatch() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let leaf = sphere(&mut world, 0.25);
    let inner = world
        .shapes
        .insert_compound(&[CompoundChild {
            collider: leaf,
            local_pose: Isometry::from_translation(Vec3::new(2.0, 0.0, 0.0)),
        }])
        .expect("inner compound");
    let outer = world
        .shapes
        .insert_compound(&[CompoundChild {
            collider: inner,
            local_pose: Isometry::from_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        }])
        .expect("outer compound");

    // The root pose rotates the flattened local child back onto +X.
    let mut root = BodyDesc::static_at(Vec3::ZERO).with_collider(outer);
    root.orientation = Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2);
    world.spawn(root);
    let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::X);
    let hit = world
        .raycast(&ray, &QueryFilter::ALL)
        .expect("flattened child should be hit");
    assert!((hit.time_of_impact - 1.75).abs() < 1.0e-4);
    assert!(world.raycast_test(&ray, &QueryFilter::ALL));

    let sphere_shape = ColliderShape::Sphere { radius: 0.1 };
    let sweep = world.shapecast(
        &sphere_shape,
        Isometry::from_translation(Vec3::ZERO),
        Vec3::new(3.0, 0.0, 0.0),
        &QueryFilter::ALL,
    );
    assert!(
        sweep.is_some(),
        "shapecast must traverse the same child pose"
    );
}

#[test]
fn pipeline_keeps_leaf_manifolds_and_child_indices() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let leaf = sphere(&mut world, 0.5);
    let compound = world
        .shapes
        .insert_compound(&[
            CompoundChild {
                collider: leaf,
                local_pose: Isometry::from_translation(Vec3::new(-1.25, 0.0, 0.0)),
            },
            CompoundChild {
                collider: leaf,
                local_pose: Isometry::from_translation(Vec3::new(1.25, 0.0, 0.0)),
            },
        ])
        .expect("valid compound");
    let compound_body = world.spawn(
        BodyDesc::dynamic_at(Vec3::ZERO)
            .with_collider(compound)
            .with_mass_properties(MassProperties {
                inv_mass: 1.0,
                inv_inertia: Vec3::ONE,
            }),
    );
    let other = dynamic_body(&mut world, Vec3::new(-1.25, 0.0, 0.0), leaf, Vec3::ZERO);

    let manifolds = detect_contacts(&world);
    let manifold = manifolds
        .iter()
        .find(|manifold| {
            (manifold.body_a == compound_body && manifold.body_b == other)
                || (manifold.body_a == other && manifold.body_b == compound_body)
        })
        .expect("left leaf contact");
    assert_eq!(manifold.child_indices.iter().flatten().count(), 1);
    assert!(manifold.child_indices.contains(&Some(0)));
    assert!(!manifold.is_empty());
    let point = manifold.points()[0];
    assert!(point.point_a.is_finite() && point.point_b.is_finite());
}

#[test]
fn compound_soft_coupling_culls_gap_and_aggregates_siblings() {
    let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
    world.cloth_coupling = ClothRigidCouplingConfig::active();
    let leaf = sphere(&mut world, 0.5);
    let gap_compound = world
        .shapes
        .insert_compound(&[
            CompoundChild {
                collider: leaf,
                local_pose: Isometry::from_translation(Vec3::new(-1.25, 0.0, 0.0)),
            },
            CompoundChild {
                collider: leaf,
                local_pose: Isometry::from_translation(Vec3::new(1.25, 0.0, 0.0)),
            },
        ])
        .expect("gap compound");
    let gap_body = dynamic_body(&mut world, Vec3::ZERO, gap_compound, Vec3::ZERO);
    let mut gap_particle = [Vec3::ZERO];
    let gap_report = world.resolve_cloth_coupling(&mut gap_particle, &[1.0], 1.0 / 60.0);
    assert_eq!(gap_report.proxy_count, 0);
    assert_eq!(gap_particle[0], Vec3::ZERO);
    assert_eq!(world.bodies.position(gap_body), Some(Vec3::ZERO));

    let sibling_compound = world
        .shapes
        .insert_compound(&[
            CompoundChild {
                collider: leaf,
                local_pose: Isometry::IDENTITY,
            },
            CompoundChild {
                collider: leaf,
                local_pose: Isometry::IDENTITY,
            },
        ])
        .expect("sibling compound");
    let sibling_body = dynamic_body(
        &mut world,
        Vec3::new(4.0, 0.0, 0.0),
        sibling_compound,
        Vec3::ZERO,
    );
    let mut sibling_particle = [Vec3::new(4.0, 0.0, 0.0)];
    let sibling_report = world.resolve_cloth_coupling(&mut sibling_particle, &[1.0], 1.0 / 60.0);
    assert_eq!(sibling_report.proxy_count, 2);
    assert_eq!(sibling_report.applied_count, 1);
    assert!(sibling_report.applied_impulse.length_squared() > 0.0);
    assert_ne!(
        world.bodies.position(sibling_body),
        Some(Vec3::new(4.0, 0.0, 0.0))
    );
}

#[test]
fn compound_ccd_sweeps_real_leaves_without_filling_the_gap() {
    fn run(x: f32) -> f32 {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let leaf = sphere(&mut world, 0.5);
        let compound = world
            .shapes
            .insert_compound(&[
                CompoundChild {
                    collider: leaf,
                    local_pose: Isometry::from_translation(Vec3::new(-2.0, 0.0, 0.0)),
                },
                CompoundChild {
                    collider: leaf,
                    local_pose: Isometry::from_translation(Vec3::new(2.0, 0.0, 0.0)),
                },
            ])
            .expect("valid compound");
        world.spawn(BodyDesc::static_at(Vec3::ZERO).with_collider(compound));

        let mover_shape = sphere(&mut world, 0.1);
        let mover = world.spawn(
            BodyDesc::dynamic_at(Vec3::new(x, 0.0, -4.0))
                .with_collider(mover_shape)
                .with_ccd(true),
        );
        // `resolve_ccd` consumes the stored previous pose and current predicted
        // pose. This models a fast sub-step without running the rest of the
        // solver or changing unrelated world state.
        world.bodies.set_position(mover, Vec3::new(x, 0.0, 4.0));
        resolve_rigid_ccd(&mut world, 1.0);
        world.bodies.position(mover).unwrap().z
    }

    let gap_end = run(0.0);
    assert!(
        (gap_end - 4.0).abs() < 1.0e-6,
        "gap was filled by CCD: {gap_end}"
    );

    let leaf_end = run(2.0);
    assert!(leaf_end < -0.5, "real child was not swept: {leaf_end}");
    assert!(leaf_end > -0.8, "CCD stopped too early: {leaf_end}");
}

#[test]
fn compound_soft_coupling_reports_unsupported_leaf_atomically() {
    let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
    world.cloth_coupling = ClothRigidCouplingConfig::active();
    let sphere = sphere(&mut world, 0.5);
    let mesh = world.shapes.insert_tri_mesh(TriMeshData::new(
        vec![
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
        vec![[0, 1, 2]],
    ));
    let mesh_shape = world.shapes.tri_mesh_shape(mesh).expect("mesh shape");
    let mesh_collider = world.shapes.insert(mesh_shape);
    let compound = world
        .shapes
        .insert_compound(&[
            CompoundChild {
                collider: sphere,
                local_pose: Isometry::IDENTITY,
            },
            CompoundChild {
                collider: mesh_collider,
                local_pose: Isometry::IDENTITY,
            },
        ])
        .expect("valid mixed compound");
    world.spawn(BodyDesc::dynamic_at(Vec3::ZERO).with_collider(compound));

    let mut particle = [Vec3::ZERO];
    let report = world.resolve_cloth_coupling(&mut particle, &[1.0], 1.0 / 60.0);
    assert_eq!(report.proxy_count, 0);
    assert_eq!(report.applied_count, 0);
    assert_eq!(report.unsupported_count, 1);
    assert_eq!(particle[0], Vec3::ZERO);
}

#[test]
fn assembly_merge_and_split_preserve_linear_momentum_with_true_compound() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let leaf = sphere(&mut world, 0.5);
    let a = dynamic_body(
        &mut world,
        Vec3::new(-1.25, 0.0, 0.0),
        leaf,
        Vec3::new(0.0, 2.0, 0.0),
    );
    let b = dynamic_body(
        &mut world,
        Vec3::new(1.25, 0.0, 0.0),
        leaf,
        Vec3::new(0.0, 2.0, 0.0),
    );
    let joint = world.joints.insert(JointDesc::distance(
        a,
        Vec3::ZERO,
        b,
        Vec3::ZERO,
        DistanceJoint::rigid(2.5),
    ));
    world
        .connect_assembly(
            a,
            b,
            Some(joint),
            JointKind::Distance(DistanceJoint::rigid(2.5)),
            None,
        )
        .expect("connected assembly");

    let before =
        world.bodies.linear_velocity(a).unwrap() + world.bodies.linear_velocity(b).unwrap();
    let proxy = world.merge_assembly(&[a, b]).expect("merge assembly");
    let proxy_shape = world
        .bodies
        .collider(proxy)
        .and_then(|handle| world.shapes.get(handle))
        .expect("compound proxy collider");
    assert!(matches!(proxy_shape, ColliderShape::Compound { .. }));
    let proxy_momentum = world.bodies.linear_velocity(proxy).unwrap() * 2.0;
    assert!((proxy_momentum - before).length() < 1.0e-5);

    let split = world.split_assembly(proxy).expect("split assembly");
    assert_eq!(split.len(), 2);
    let after = split
        .iter()
        .map(|handle| world.bodies.linear_velocity(*handle).unwrap())
        .fold(Vec3::ZERO, |sum, velocity| sum + velocity);
    assert!((after - before).length() < 1.0e-5);
}
