//! Moving, heterogeneous COM/principal frames through the ordinary ECS bridge.
//! Complements articulation_handoff.rs without repeating its pose-input cases.
use std::time::Duration;

use glam::{Mat3, Quat, Vec3};
use prism_app::{App, TimeUpdateStrategy};
use prism_ecs::prelude::*;
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::articulation::ArticulationBodyState;
use prism_physics_core::reduced::{
    ArticulationBase, ArticulationColliderDesc, ArticulationHandle, ArticulationJoint,
    ArticulationLink, SpatialInertia, SpatialMotion, SpatialTransform,
};
use prism_physics_core::snapshot::FullStateSnapshot;
use prism_physics_core::solver::Solver;
use prism_physics_core::{BodyHandle, ColliderHandle, ColliderShape};
use prism_physics_ecs::articulation::{
    ArticulationBinding, ArticulationBindings, ArticulationDiagnostics, ArticulationHandoff,
    ArticulationIssue, PrismArticulationPlugin,
};
use prism_physics_ecs::{
    Collider, CollisionShape, PhysicsBindings, PhysicsDiagnostics, PhysicsSolver,
    PhysicsWorldResource, PrismPhysicsPlugin, RigidBody, Velocity,
};
use prism_transform::{GlobalTransform, Transform};

const STEP: Duration = Duration::from_millis(10);
const EPSILON: f32 = 2.0e-4;

struct Fixture {
    app: App,
    owner: Entity,
    source: ArticulationHandle,
    converted: ArticulationHandoff,
    states: Vec<ArticulationBodyState>,
    source_shape: ColliderHandle,
    collider_poses: Vec<Isometry>,
}

fn inertia(index: usize) -> SpatialInertia {
    let i = index as f32;
    let axes = Mat3::from_quat(
        Quat::from_rotation_z(0.35 + 0.17 * i)
            * Quat::from_rotation_x(-0.4 + 0.09 * i)
            * Quat::from_rotation_y(0.2 + 0.11 * i),
    );
    // Distinct, strictly positive moments: this tests frame transport, not the
    // core's separately tested singular-inertia cutoff policy.
    let moments = Vec3::new(0.8 + 0.2 * i, 1.3 + 0.3 * i, 1.9 + 0.4 * i);
    let tensor = axes * Mat3::from_diagonal(moments) * axes.transpose();
    SpatialInertia::new(
        1.5 + 0.4 * i,
        Vec3::new(0.2 + 0.08 * i, -0.3 + 0.04 * i, 0.15 - 0.06 * i),
        tensor.transpose().to_cols_array_2d(),
    )
}

fn fixture(base: ArticulationBase) -> Fixture {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(STEP))
        .set_fixed_timestep(STEP)
        .add_plugins((PrismPhysicsPlugin, PrismArticulationPlugin));
    {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        physics.config.gravity = Vec3::ZERO;
        physics.config.default_substeps = 1;
    }
    let mut binding = ArticulationBinding::new(base, inertia(0));
    binding.base_pose = Isometry::new(
        Vec3::new(1.0, -2.0, 3.0),
        Quat::from_rotation_y(0.3) * Quat::from_rotation_x(-0.2),
    );
    if base == ArticulationBase::Floating {
        binding.base_velocity =
            SpatialMotion::new(Vec3::new(0.2, -0.3, 0.4), Vec3::new(0.7, -0.6, 0.5));
    }
    for (index, joint) in [
        ArticulationJoint::prismatic(Vec3::new(1.0, 2.0, 0.5).normalize()),
        ArticulationJoint::revolute(Vec3::new(-0.5, 1.0, 2.0).normalize()),
        ArticulationJoint::fixed(),
    ]
    .into_iter()
    .enumerate()
    {
        binding.links.push(ArticulationLink {
            parent: index.checked_sub(1),
            x_tree: SpatialTransform::new(
                Quat::from_rotation_z(0.2 * (index + 1) as f32) * Quat::from_rotation_y(-0.15),
                Vec3::new(0.5, 1.2, -0.25),
            ),
            inertia: inertia(index + 1),
            joint,
        });
    }
    binding.positions = vec![0.37, -0.41];
    binding.velocities = vec![0.7, -0.3];
    let owner = app.world_mut().spawn((binding,));
    app.update();
    let source = app
        .world()
        .resource::<ArticulationBindings>()
        .articulation(owner)
        .unwrap();
    let collider_poses: Vec<_> = (0..4)
        .map(|index| {
            let i = index as f32;
            Isometry::new(
                Vec3::new(-0.1 + 0.03 * i, 0.25 - 0.02 * i, 0.2 + 0.04 * i),
                Quat::from_rotation_x(-0.3 + 0.07 * i) * Quat::from_rotation_y(0.2),
            )
        })
        .collect();
    let (source_shape, states) = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        // An anisotropic cuboid makes orientation as well as offset meaningful.
        // Small separated colliders keep the solver comparison contact-free.
        let shape = physics.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::new(0.04, 0.06, 0.08),
        });
        for (index, &local_pose) in collider_poses.iter().enumerate() {
            let mut desc = ArticulationColliderDesc::new(source, index.checked_sub(1), shape);
            desc.local_pose = local_pose;
            physics.attach_articulation_collider(desc).unwrap();
        }
        let tree = physics.articulations.get_mut(source).unwrap();
        assert!(tree.positions().iter().all(|q| q.abs() > 0.1));
        assert!(tree.velocities().iter().all(|qd| qd.abs() > 0.1));
        if base == ArticulationBase::Floating {
            assert!(tree.base_velocity().linear.length() > 0.1);
            assert!(tree.base_velocity().angular.length() > 0.1);
        }
        let states = tree.maximal_body_states();
        // Direct runtime degradation, not the queued world break API. A broken
        // reduced tree must not advance once more before ECS materialization.
        tree.break_joint(1).unwrap();
        (shape, states)
    };
    app.update();
    let converted = app
        .world()
        .resource::<ArticulationBindings>()
        .handoff(owner)
        .unwrap()
        .clone();
    assert_eq!(converted.report.articulation, source);
    assert_eq!(converted.report.broken_link, 1);
    assert_eq!(converted.bodies.len(), 4);
    assert_eq!(converted.report.colliders.len(), 4);
    assert_eq!(converted.report.joints.len(), 2);
    Fixture {
        app,
        owner,
        source,
        converted,
        states,
        source_shape,
        collider_poses,
    }
}

fn close_vec(actual: Vec3, expected: Vec3) {
    assert!(
        actual.distance(expected) < EPSILON,
        "actual={actual:?}, expected={expected:?}"
    );
}

fn close_pose(actual: Isometry, expected: Isometry) {
    for point in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
        close_vec(
            actual.transform_point(point),
            expected.transform_point(point),
        );
    }
}

fn body_pose(world: &World, body: BodyHandle) -> Isometry {
    let physics = world.resource::<PhysicsWorldResource>();
    Isometry::new(
        physics.bodies.position(body).unwrap(),
        physics.bodies.orientation(body).unwrap(),
    )
}

fn local_pose(world: &World, entity: Entity) -> Isometry {
    let transform = world.get::<Transform>(entity).unwrap();
    Isometry::new(
        Vec3::new(
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ),
        Quat::from_xyzw(
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ),
    )
}

fn transform(pose: Isometry) -> Transform {
    Transform {
        translation: prism_math::Vec3::new(
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
        ),
        rotation: prism_math::Quat::from_xyzw(
            pose.rotation.x,
            pose.rotation.y,
            pose.rotation.z,
            pose.rotation.w,
        ),
        scale: prism_math::Vec3::ONE,
    }
}

fn global_pose(world: &World, entity: Entity, expected: Isometry) {
    let global = world.get::<GlobalTransform>(entity).unwrap();
    for point in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
        let actual = global.transform_point(prism_math::Vec3::new(point.x, point.y, point.z));
        close_vec(
            Vec3::new(actual.x, actual.y, actual.z),
            expected.transform_point(point),
        );
    }
}

fn assert_frames_and_geometry(fixture: &Fixture) {
    let world = fixture.app.world();
    let physics = world.resource::<PhysicsWorldResource>();
    let bindings = world.resource::<PhysicsBindings>();
    assert_eq!(
        world
            .resource::<ArticulationBindings>()
            .handoff(fixture.owner),
        Some(&fixture.converted),
        "ordinary sync must preserve the complete generational handoff identity"
    );
    assert!(!physics.articulations.contains(fixture.source));
    assert_eq!(physics.bodies.len(), 4);
    assert_eq!(physics.joints.len(), 2);
    for (member, mapping) in
        fixture.converted.bodies.iter().zip(
            std::iter::once(&fixture.converted.report.base).chain(&fixture.converted.report.links),
        )
    {
        assert_eq!(member.body, mapping.body);
        assert_eq!(member.link, mapping.link);
        assert_eq!(bindings.body(member.entity), Some(member.body));
        assert_eq!(bindings.entity(member.body), Some(member.entity));
        assert_eq!(bindings.body(member.authored_entity), None);
        assert!(world.relation_targets::<ChildOf>(member.entity).is_empty());
        assert_eq!(
            world.relation_targets::<ChildOf>(member.authored_entity),
            [member.entity]
        );
        assert!(world.get::<RigidBody>(member.authored_entity).is_none());
        assert!(world.get::<Collider>(member.authored_entity).is_none());
        let pose = body_pose(world, member.body);
        close_pose(local_pose(world, member.entity), pose);
        global_pose(world, member.entity, pose);
        close_pose(
            local_pose(world, member.authored_entity),
            mapping.link_from_body.inverse(),
        );
        global_pose(
            world,
            member.authored_entity,
            pose.mul(&mapping.link_from_body.inverse()),
        );
        let velocity = world.get::<Velocity>(member.entity).unwrap();
        close_vec(
            velocity.linear,
            physics.bodies.linear_velocity(member.body).unwrap(),
        );
        close_vec(
            velocity.angular,
            physics.bodies.angular_velocity(member.body).unwrap(),
        );
    }
    for mapping in &fixture.converted.report.colliders {
        let index = mapping.link.map_or(0, |link| link + 1);
        let member = &fixture.converted.bodies[index];
        let frame = if index == 0 {
            &fixture.converted.report.base
        } else {
            &fixture.converted.report.links[index - 1]
        };
        assert_eq!(mapping.body, member.body);
        assert_eq!(mapping.source_shape, fixture.source_shape);
        assert_eq!(mapping.local_pose, fixture.collider_poses[index]);
        assert_eq!(
            physics.bodies.collider(member.body),
            Some(mapping.body_shape)
        );
        assert_eq!(
            world.get::<Collider>(member.entity).unwrap().shape,
            CollisionShape::Registered(mapping.body_shape)
        );
        let ColliderShape::Compound { compound, .. } =
            *physics.shapes.get(mapping.body_shape).unwrap()
        else {
            panic!("the authored collider transform must be baked into body-local geometry")
        };
        let children = physics.shapes.compound(compound).unwrap().children();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].collider, fixture.source_shape);
        let expected_local = frame
            .link_from_body
            .inverse()
            .mul(&fixture.collider_poses[index]);
        close_pose(children[0].local_pose, expected_local);
        let pose = body_pose(world, member.body);
        close_pose(
            pose.mul(&children[0].local_pose),
            pose.mul(&frame.link_from_body.inverse())
                .mul(&fixture.collider_poses[index]),
        );
    }
}

fn assert_initial_handoff(fixture: &Fixture) {
    assert_frames_and_geometry(fixture);
    let world = fixture.app.world();
    let physics = world.resource::<PhysicsWorldResource>();
    for ((member, mapping), state) in fixture
        .converted
        .bodies
        .iter()
        .zip(std::iter::once(&fixture.converted.report.base).chain(&fixture.converted.report.links))
        .zip(&fixture.states)
    {
        let pose = body_pose(world, member.body);
        close_pose(pose, state.pose.mul(&mapping.link_from_body));
        global_pose(world, member.authored_entity, state.pose);
        close_vec(
            physics.bodies.linear_velocity(member.body).unwrap(),
            state.linear_velocity,
        );
        close_vec(
            physics.bodies.angular_velocity(member.body).unwrap(),
            state.angular_velocity,
        );
        let mass = physics.bodies.mass_properties(member.body).unwrap();
        if state.fixed {
            assert_eq!(mapping.link_from_body, Isometry::IDENTITY);
            assert_eq!(mass.inv_mass, 0.0);
            assert_eq!(mass.inv_inertia, Vec3::ZERO);
            continue;
        }
        close_vec(pose.translation, state.center_of_mass);
        close_vec(
            mapping.link_from_body.translation,
            state.inertia.center_of_mass,
        );
        assert!(mapping.link_from_body.translation.length() > 0.1);
        assert!((mapping.link_from_body.rotation * Vec3::Z - Vec3::Z).length() > 0.1);
        assert!((mass.inv_mass.recip() - state.inertia.mass).abs() < EPSILON);
        assert!(mass.inv_inertia.min_element() > 0.0);
        let rotation = Mat3::from_quat(pose.rotation);
        let actual =
            rotation * Mat3::from_diagonal(mass.inv_inertia.recip()) * rotation.transpose();
        let authored_rotation = Mat3::from_quat(state.pose.rotation);
        let tensor = Mat3::from_cols_array_2d(&state.inertia.inertia_com).transpose();
        let expected = authored_rotation * tensor * authored_rotation.transpose();
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            close_vec(actual * axis, expected * axis);
        }
    }
    for collider in &fixture.converted.report.colliders {
        let index = collider.link.map_or(0, |link| link + 1);
        let ColliderShape::Compound { compound, .. } =
            *physics.shapes.get(collider.body_shape).unwrap()
        else {
            unreachable!()
        };
        let baked = physics.shapes.compound(compound).unwrap().children()[0].local_pose;
        close_pose(
            body_pose(world, collider.body).mul(&baked),
            fixture.states[index]
                .pose
                .mul(&fixture.collider_poses[index]),
        );
    }
}

// Store the complete authored-local pose for both child and grandchild so the
// same attachment set can check moving frame propagation and group retirement.
fn add_authored_children(fixture: &mut Fixture) -> Vec<(Entity, usize, Isometry)> {
    let mut attachments = Vec::new();
    for (index, member) in fixture.converted.bodies.iter().enumerate() {
        let child_pose = Isometry::new(Vec3::new(0.13, -0.21, 0.17), Quat::from_rotation_y(0.27));
        let grandchild_pose =
            Isometry::new(Vec3::new(-0.09, 0.11, 0.23), Quat::from_rotation_x(-0.19));
        let child = fixture.app.world_mut().spawn((transform(child_pose),));
        fixture
            .app
            .world_mut()
            .add_relation::<ChildOf>(child, member.authored_entity);
        let grandchild = fixture.app.world_mut().spawn((transform(grandchild_pose),));
        fixture
            .app
            .world_mut()
            .add_relation::<ChildOf>(grandchild, child);
        attachments.push((child, index, child_pose));
        attachments.push((grandchild, index, child_pose.mul(&grandchild_pose)));
    }
    attachments
}

fn moving_frames_match_core_future(base: ArticulationBase) {
    let mut fixture = fixture(base);
    assert_initial_handoff(&fixture);
    let attachments = add_authored_children(&mut fixture);
    let mut reference =
        FullStateSnapshot::capture(0, fixture.app.world().resource::<PhysicsWorldResource>())
            .fork_world();
    let mut solver = fixture.app.world().resource::<PhysicsSolver>().0.clone();
    let initial = body_pose(fixture.app.world(), fixture.converted.bodies[2].body);
    for _ in 0..6 {
        solver.step(&mut reference, STEP.as_secs_f32(), 1);
        let tick = fixture.app.world().resource::<PhysicsDiagnostics>().tick;
        fixture.app.update();
        assert_eq!(
            fixture.app.world().resource::<PhysicsDiagnostics>().tick,
            tick + 1
        );
        assert_eq!(
            fixture
                .app
                .world()
                .resource::<PhysicsDiagnostics>()
                .simulated_dt,
            STEP.as_secs_f32()
        );
        assert_frames_and_geometry(&fixture);
        for member in &fixture.converted.bodies {
            close_pose(
                body_pose(fixture.app.world(), member.body),
                Isometry::new(
                    reference.bodies.position(member.body).unwrap(),
                    reference.bodies.orientation(member.body).unwrap(),
                ),
            );
            let velocity = fixture.app.world().get::<Velocity>(member.entity).unwrap();
            close_vec(
                velocity.linear,
                reference.bodies.linear_velocity(member.body).unwrap(),
            );
            close_vec(
                velocity.angular,
                reference.bodies.angular_velocity(member.body).unwrap(),
            );
        }
        for &(entity, index, authored_local) in &attachments {
            let frame = if index == 0 {
                &fixture.converted.report.base
            } else {
                &fixture.converted.report.links[index - 1]
            };
            global_pose(
                fixture.app.world(),
                entity,
                body_pose(fixture.app.world(), frame.body)
                    .mul(&frame.link_from_body.inverse())
                    .mul(&authored_local),
            );
        }
    }
    assert!(
        body_pose(fixture.app.world(), fixture.converted.bodies[2].body)
            .translation
            .distance(initial.translation)
            > EPSILON,
        "the future-frame assertions must exercise moving bodies"
    );
}

#[test]
fn fixed_base_heterogeneous_com_frames_match_core_through_ordinary_fixed_steps() {
    moving_frames_match_core_future(ArticulationBase::Fixed);
}

#[test]
fn floating_base_heterogeneous_com_frames_match_core_through_ordinary_fixed_steps() {
    moving_frames_match_core_future(ArticulationBase::Floating);
}

#[derive(Clone, Copy, Debug)]
enum FrameMutation {
    EditPose,
    Despawn,
    Reparent,
}

#[test]
fn invalid_authored_frame_retires_the_whole_group_and_every_authored_descendant() {
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        for mutation in [
            FrameMutation::EditPose,
            FrameMutation::Despawn,
            FrameMutation::Reparent,
        ] {
            let mut fixture = fixture(base);
            let attachments = add_authored_children(&mut fixture);
            // An unrelated hierarchy must survive even when it temporarily
            // becomes the parent of an owned authored frame.
            let foreign = fixture.app.world_mut().spawn((Transform::default(),));
            let foreign_child = fixture.app.world_mut().spawn((Transform::default(),));
            fixture
                .app
                .world_mut()
                .add_relation::<ChildOf>(foreign_child, foreign);
            let frame = fixture.converted.bodies[2].authored_entity;
            match mutation {
                FrameMutation::EditPose => {
                    let mut pose = *fixture.app.world().get::<Transform>(frame).unwrap();
                    pose.rotation = prism_math::Quat::from_rotation_y(0.7) * pose.rotation;
                    fixture.app.world_mut().insert(frame, (pose,));
                }
                FrameMutation::Despawn => {
                    assert!(fixture.app.world_mut().despawn(frame));
                }
                FrameMutation::Reparent => {
                    fixture
                        .app
                        .world_mut()
                        .add_relation::<ChildOf>(frame, foreign);
                }
            }
            fixture.app.update();
            assert!(
                fixture
                    .app
                    .world()
                    .resource::<ArticulationDiagnostics>()
                    .issues
                    .iter()
                    .any(|diagnostic| diagnostic.entity == Some(fixture.owner)
                        && diagnostic.issue == ArticulationIssue::HandoffInvalidated),
                "missing invalidation for {base:?}/{mutation:?}"
            );
            // Check two boundaries: retirement must not leave sibling visuals,
            // recreate the frozen authoring, or affect the foreign hierarchy.
            for _ in 0..2 {
                let world = fixture.app.world();
                let physics = world.resource::<PhysicsWorldResource>();
                let bindings = world.resource::<PhysicsBindings>();
                assert!(world.contains(fixture.owner));
                assert!(world.get::<ArticulationBinding>(fixture.owner).is_some());
                assert!(world
                    .resource::<ArticulationBindings>()
                    .handoff(fixture.owner)
                    .is_none());
                assert!(world
                    .resource::<ArticulationBindings>()
                    .articulation(fixture.owner)
                    .is_none());
                assert!(!physics.articulations.contains(fixture.source));
                assert!(physics.bodies.is_empty());
                assert_eq!(physics.joints.len(), 0);
                for member in &fixture.converted.bodies {
                    assert!(!world.contains(member.entity));
                    assert!(!world.contains(member.authored_entity));
                    assert!(!physics.bodies.contains(member.body));
                    assert_eq!(bindings.body(member.entity), None);
                    assert_eq!(bindings.body(member.authored_entity), None);
                    assert_eq!(bindings.entity(member.body), None);
                }
                for mapping in &fixture.converted.report.joints {
                    assert!(!physics.joints.contains(mapping.joint));
                }
                for &(entity, _, _) in &attachments {
                    assert!(
                        !world.contains(entity),
                        "orphan authored descendant after {mutation:?}"
                    );
                    assert_eq!(bindings.body(entity), None);
                }
                assert!(world.contains(foreign));
                assert!(world.contains(foreign_child));
                assert_eq!(world.relation_targets::<ChildOf>(foreign_child), [foreign]);
                fixture.app.update();
            }
        }
    }
}
