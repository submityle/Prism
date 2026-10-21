//! Production maximal-coordinate takeover through the ordinary ECS body bridge.
use std::time::Duration;

use glam::{Mat3, Quat, Vec3};
use prism_app::{App, FixedUpdate, TimeUpdateStrategy};
use prism_ecs::prelude::*;
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::articulation::handoff::{
    ArticulationHandoffError, UnsupportedJointPolicy,
};
use prism_physics_core::reduced::{
    ArticulationBase, ArticulationColliderDesc, ArticulationHandle, ArticulationJoint,
    ArticulationLink, JointMotor, SpatialInertia, SpatialTransform,
};
use prism_physics_core::snapshot::FullStateSnapshot;
use prism_physics_core::{BodyDesc, ColliderShape};
use prism_physics_ecs::articulation::{
    ArticulationBinding, ArticulationBindings, ArticulationDiagnostics, ArticulationHandoff,
    ArticulationIssue, ArticulationSnapshot, PrismArticulationPlugin,
};
use prism_physics_ecs::{
    Collider, CollisionShape, ExternalForce, PhysicsBindings, PhysicsSystems, PhysicsWorldResource,
    PrismPhysicsPlugin, RigidBody, Velocity,
};
use prism_transform::{GlobalTransform, Transform};

fn app() -> App {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)))
        .set_fixed_timestep(Duration::from_millis(10))
        .add_plugins((PrismPhysicsPlugin, PrismArticulationPlugin));
    let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
    physics.config.gravity = Vec3::ZERO;
    physics.config.default_substeps = 1;
    app
}

fn authored() -> ArticulationBinding {
    let mut config = ArticulationBinding::default();
    config.base_pose = Isometry::from_translation(Vec3::new(2.0, 3.0, 4.0));
    for link in 0..3_usize {
        config.links.push(ArticulationLink {
            parent: link.checked_sub(1),
            x_tree: SpatialTransform::new(Quat::IDENTITY, Vec3::Y),
            inertia: SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::splat(0.2)),
            joint: ArticulationJoint::revolute(Vec3::Z),
        });
    }
    config.velocities = vec![0.1, 0.2, 0.3];
    config
}

fn source(app: &mut App, config: ArticulationBinding) -> (Entity, ArticulationHandle) {
    let owner = app.world_mut().spawn((config,));
    app.update();
    let handle = app
        .world()
        .resource::<ArticulationBindings>()
        .articulation(owner)
        .unwrap();
    (owner, handle)
}

fn break_tree(app: &mut App, handle: ArticulationHandle) {
    assert!(app
        .world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .request_articulation_break(handle, 1));
    app.update();
}

fn view(app: &App, owner: Entity) -> ArticulationHandoff {
    app.world()
        .resource::<ArticulationBindings>()
        .handoff(owner)
        .unwrap()
        .clone()
}

fn retired(app: &App, converted: &ArticulationHandoff) {
    let physics = app.world().resource::<PhysicsWorldResource>();
    let bindings = app.world().resource::<PhysicsBindings>();
    for member in &converted.bodies {
        assert!(!physics.bodies.contains(member.body));
        assert!(!app.world().contains(member.entity));
        assert!(!app.world().contains(member.authored_entity));
        assert_eq!(bindings.body(member.entity), None);
        assert_eq!(bindings.body(member.authored_entity), None);
        assert_eq!(bindings.entity(member.body), None);
    }
    for mapping in &converted.report.joints {
        assert!(!physics.joints.contains(mapping.joint));
    }
}

fn physical_pose(world: &World, body: prism_physics_core::BodyHandle) -> Isometry {
    let physics = world.resource::<PhysicsWorldResource>();
    Isometry::new(
        physics.bodies.position(body).unwrap(),
        physics.bodies.orientation(body).unwrap(),
    )
}

fn local_pose(world: &World, entity: Entity) -> Isometry {
    let pose = world.get::<Transform>(entity).unwrap();
    Isometry::new(
        Vec3::new(pose.translation.x, pose.translation.y, pose.translation.z),
        Quat::from_xyzw(
            pose.rotation.x,
            pose.rotation.y,
            pose.rotation.z,
            pose.rotation.w,
        ),
    )
}

fn assert_pose_close(actual: Isometry, expected: Isometry) {
    for point in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
        assert!(
            actual
                .transform_point(point)
                .distance(expected.transform_point(point))
                < 2.0e-4,
            "actual={actual:?}, expected={expected:?}"
        );
    }
}

fn assert_global_pose(world: &World, entity: Entity, expected: Isometry) {
    let global = world.get::<GlobalTransform>(entity).unwrap();
    for point in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
        let actual = global.transform_point(prism_math::Vec3::new(point.x, point.y, point.z));
        assert!(
            Vec3::new(actual.x, actual.y, actual.z).distance(expected.transform_point(point))
                < 2.0e-4,
            "entity={entity:?}, actual={actual:?}, expected={expected:?}"
        );
    }
}

fn assert_authored_frames(world: &World, converted: &ArticulationHandoff) {
    for (member, mapping) in converted
        .bodies
        .iter()
        .zip(std::iter::once(&converted.report.base).chain(&converted.report.links))
    {
        assert_eq!(member.body, mapping.body);
        assert_ne!(member.entity, member.authored_entity);
        assert_eq!(
            world.relation_targets::<ChildOf>(member.authored_entity),
            [member.entity]
        );
        assert!(world.get::<RigidBody>(member.authored_entity).is_none());
        assert!(world.get::<Collider>(member.authored_entity).is_none());
        assert_eq!(
            world
                .resource::<PhysicsBindings>()
                .body(member.authored_entity),
            None
        );
        let body_pose = physical_pose(world, member.body);
        assert_pose_close(local_pose(world, member.entity), body_pose);
        assert_pose_close(
            local_pose(world, member.authored_entity),
            mapping.link_from_body.inverse(),
        );
        assert_global_pose(
            world,
            member.authored_entity,
            body_pose.mul(&mapping.link_from_body.inverse()),
        );
    }
}

#[test]
fn handoff_phase_order_is_independent_of_plugin_registration_order() {
    #[derive(Resource)]
    struct Probe {
        owner: Entity,
        broken: bool,
        trace: Vec<&'static str>,
    }
    fn during_sync(world: &mut World) {
        let owner = world.resource::<Probe>().owner;
        let handle = world
            .resource::<ArticulationBindings>()
            .articulation(owner)
            .expect("PreSync must publish the reduced identity before every Sync system");
        assert!(world
            .resource::<PhysicsWorldResource>()
            .articulations
            .contains(handle));
        world.resource_mut::<Probe>().trace.push("sync");
    }
    fn during_writeback(world: &mut World) {
        let owner = world.resource::<Probe>().owner;
        assert!(
            world
                .resource::<ArticulationBindings>()
                .handoff(owner)
                .is_none(),
            "handoff must wait for every ordinary Writeback system"
        );
        world.resource_mut::<Probe>().trace.push("writeback");
    }
    fn after_handoff(world: &mut World) {
        let probe = world.resource::<Probe>();
        let owner = probe.owner;
        if probe.broken {
            let converted = world
                .resource::<ArticulationBindings>()
                .handoff(owner)
                .expect("handoff must finish before PostWriteback consumers");
            for member in &converted.bodies {
                assert_eq!(
                    world.resource::<PhysicsBindings>().body(member.entity),
                    Some(member.body)
                );
                assert!(world.get::<RigidBody>(member.entity).is_some());
            }
            assert!(world.get::<ArticulationSnapshot>(owner).is_none());
        } else {
            assert!(world.get::<ArticulationSnapshot>(owner).is_some());
        }
        world.resource_mut::<Probe>().trace.push("post_writeback");
    }
    for articulation_first in [false, true] {
        let mut app = App::new();
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)))
            .set_fixed_timestep(Duration::from_millis(10));
        // Register probes first so insertion order cannot accidentally satisfy
        // the edges required from either plugin registration order.
        app.add_systems(
            FixedUpdate,
            (
                after_handoff.after(PhysicsSystems::PostWriteback),
                during_writeback
                    .in_set(PhysicsSystems::Writeback)
                    .after(PhysicsSystems::Step),
                during_sync
                    .in_set(PhysicsSystems::Sync)
                    .before(PhysicsSystems::Step),
            ),
        );
        if articulation_first {
            app.add_plugins((PrismArticulationPlugin, PrismPhysicsPlugin));
        } else {
            app.add_plugins((PrismPhysicsPlugin, PrismArticulationPlugin));
        }
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        physics.config.gravity = Vec3::ZERO;
        physics.config.default_substeps = 1;
        let owner = app.world_mut().spawn((authored(),));
        app.insert_resource(Probe {
            owner,
            broken: false,
            trace: Vec::new(),
        });
        app.update();
        let handle = app
            .world()
            .resource::<ArticulationBindings>()
            .articulation(owner)
            .unwrap();
        app.world_mut().resource_mut::<Probe>().broken = true;
        break_tree(&mut app, handle);
        assert_eq!(
            app.world().resource::<Probe>().trace,
            vec![
                "sync",
                "writeback",
                "post_writeback",
                "sync",
                "writeback",
                "post_writeback"
            ],
            "articulation_first={articulation_first}"
        );
    }
}

#[test]
fn adopts_exact_core_bodies_colliders_and_surviving_joints_and_snapshots() {
    let mut app = app();
    let (owner, handle) = source(&mut app, authored());
    let (shape, attachment) = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        let shape = physics.shapes.insert(ColliderShape::Sphere { radius: 0.1 });
        let mut desc = ArticulationColliderDesc::new(handle, Some(1), shape);
        desc.local_pose = Isometry::from_translation(Vec3::X * 0.2);
        (shape, physics.attach_articulation_collider(desc).unwrap())
    };
    break_tree(&mut app, handle);
    let converted = view(&app, owner);
    assert_eq!(
        converted.report.colliders.len(),
        1,
        "the XPBD contact pass must preserve broken-tree attachments until handoff"
    );
    assert_eq!(converted.report.colliders[0].source, attachment);
    assert_eq!(converted.report.colliders[0].source_shape, shape);
    assert_eq!(converted.bodies.len(), 4);
    assert_eq!(
        converted
            .report
            .joints
            .iter()
            .map(|m| m.link)
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
    assert_eq!(converted.report.articulation, handle);
    assert!(app.world().get::<ArticulationSnapshot>(owner).is_none());
    assert_eq!(
        app.world()
            .resource::<ArticulationBindings>()
            .articulation(owner),
        None
    );
    assert_eq!(
        app.world()
            .resource::<ArticulationBindings>()
            .entity(handle),
        None
    );
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert!(!physics.articulations.contains(handle));
    assert!(physics
        .articulation_colliders
        .iter()
        .all(|(id, _)| id != attachment));
    assert!(physics.shapes.get(shape).is_some());
    let bindings = app.world().resource::<PhysicsBindings>();
    for member in &converted.bodies {
        assert_eq!(bindings.body(member.entity), Some(member.body));
        assert_eq!(bindings.entity(member.body), Some(member.entity));
        assert!(app
            .world()
            .relation_targets::<ChildOf>(member.entity)
            .is_empty());
        let pose = app.world().get::<Transform>(member.entity).unwrap();
        let position = physics.bodies.position(member.body).unwrap();
        assert_eq!(
            [pose.translation.x, pose.translation.y, pose.translation.z],
            position.to_array()
        );
        assert_eq!(
            app.world().get::<Velocity>(member.entity).unwrap().linear,
            physics.bodies.linear_velocity(member.body).unwrap()
        );
    }
    let collider_member = &converted.bodies[2];
    let collider = app
        .world()
        .get::<Collider>(collider_member.entity)
        .expect("the adopted entity must carry its original link's collider");
    assert_eq!(
        collider.shape,
        CollisionShape::Registered(converted.report.colliders[0].body_shape)
    );
    assert_ne!(converted.report.colliders[0].body_shape, shape);
    for mapping in &converted.report.joints {
        assert!(physics.joints.contains(mapping.joint));
        assert!(physics
            .assemblies
            .connections()
            .any(|c| c.joint == Some(mapping.joint)));
    }
    let restored = FullStateSnapshot::capture(7, physics).fork_world();
    for member in &converted.bodies {
        assert!(restored.bodies.contains(member.body));
    }
    for mapping in &converted.report.joints {
        assert!(restored.joints.contains(mapping.joint));
    }
    assert_authored_frames(app.world(), &converted);
    app.update();
    assert_eq!(view(&app, owner), converted);
    assert_authored_frames(app.world(), &converted);
    assert_eq!(
        app.world().resource::<PhysicsWorldResource>().bodies.len(),
        4
    );
}

#[test]
fn ordinary_velocity_force_pose_and_collider_inputs_keep_adopted_handles() {
    let mut app = app();
    let (owner, handle) = source(&mut app, authored());
    break_tree(&mut app, handle);
    let converted = view(&app, owner);
    // Entire detached subtree translates together: surviving joints remain valid.
    for member in converted.bodies.iter().skip(2) {
        app.world_mut().insert(
            member.entity,
            (
                Velocity {
                    linear: Vec3::X,
                    angular: Vec3::ZERO,
                },
                ExternalForce {
                    force: Vec3::X * 5.0,
                    torque: Vec3::ZERO,
                },
            ),
        );
    }
    let entity = converted.bodies[2].entity;
    let before = app.world().get::<Transform>(entity).unwrap().translation.x;
    app.world_mut().insert(entity, (Collider::sphere(0.1),));
    app.update();
    assert_eq!(view(&app, owner), converted);
    assert!(app.world().get::<Transform>(entity).unwrap().translation.x > before);
    assert!(app.world().get::<Velocity>(entity).unwrap().linear.x > 1.0);
    let mut pose = *app.world().get::<Transform>(entity).unwrap();
    pose.translation.x += 2.0;
    app.world_mut().insert(entity, (pose,));
    app.world_mut().remove::<Collider>(entity);
    app.world_mut().remove::<Velocity>(entity);
    app.update();
    assert_eq!(view(&app, owner), converted);
    assert!(app.world().get::<Velocity>(entity).is_none());
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .bodies
            .collider(converted.bodies[2].body),
        None
    );
}

#[test]
fn authoring_edits_do_not_rebuild_but_remove_readd_starts_new_lifetime() {
    let mut app = app();
    let (owner, handle) = source(&mut app, authored());
    break_tree(&mut app, handle);
    let converted = view(&app, owner);
    let mut edited = authored();
    edited.base_pose.translation = Vec3::splat(100.0);
    // Even invalid generalized initial state is no longer maximal input.
    edited.positions = vec![f32::NAN];
    app.world_mut().insert(owner, (edited,));
    app.update();
    assert_eq!(view(&app, owner), converted);
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .articulations
        .is_empty());
    app.world_mut().remove::<ArticulationBinding>(owner);
    app.world_mut().insert(owner, (authored(),));
    app.update();
    retired(&app, &converted);
    assert_ne!(
        app.world()
            .resource::<ArticulationBindings>()
            .articulation(owner),
        Some(handle)
    );
    assert!(app
        .world()
        .resource::<ArticulationBindings>()
        .articulation(owner)
        .is_some());
    assert!(app.world().get::<ArticulationSnapshot>(owner).is_some());
}

#[test]
fn unsupported_surviving_policy_fails_without_allocating_and_retries() {
    let mut app = app();
    let mut config = authored();
    config.links[0].joint.damping = 0.5;
    let (owner, handle) = source(&mut app, config);
    let shape = app
        .world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .shapes
        .insert(ColliderShape::Sphere { radius: 0.1 });
    let attachment = app
        .world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .attach_articulation_collider(ArticulationColliderDesc::new(handle, Some(1), shape))
        .unwrap();
    let count = app.world().entity_count();
    break_tree(&mut app, handle);
    assert!(app
        .world()
        .resource::<ArticulationBindings>()
        .handoff(owner)
        .is_none());
    assert_eq!(
        app.world()
            .resource::<ArticulationBindings>()
            .articulation(owner),
        Some(handle)
    );
    assert_eq!(
        app.world()
            .get::<ArticulationSnapshot>(owner)
            .unwrap()
            .broken_joint,
        Some(1)
    );
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert_eq!(physics.bodies.len(), 0);
    assert_eq!(physics.joints.len(), 0);
    let expected_attachment = ArticulationColliderDesc::new(handle, Some(1), shape);
    assert_eq!(
        physics.articulation_colliders.get(attachment),
        Some(&expected_attachment),
        "a stopped broken tree still owns its geometry when handoff rejects a surviving policy"
    );
    assert_eq!(physics.shapes.len(), 1);
    assert_eq!(app.world().entity_count(), count);
    assert!(app
        .world()
        .resource::<ArticulationDiagnostics>()
        .issues
        .iter()
        .any(|d| d.entity == Some(owner)
            && d.issue
                == ArticulationIssue::HandoffFailed(
                    ArticulationHandoffError::UnsupportedJointPolicy(
                        UnsupportedJointPolicy::InertOrExtraParameters { link: 0 }
                    )
                )));
    // Another complete fixed step must neither prune the attachment nor allocate
    // a partial replacement while the same unsupported policy is still present.
    app.update();
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert_eq!(
        physics.articulation_colliders.get(attachment),
        Some(&expected_attachment)
    );
    assert_eq!(physics.bodies.len(), 0);
    assert_eq!(physics.joints.len(), 0);
    assert_eq!(physics.shapes.len(), 1);
    assert_eq!(app.world().entity_count(), count);
    // Repair through core without an authoring rebuild; conversion retries.
    let mut repaired = authored().build().unwrap();
    repaired.break_joint(1).unwrap();
    *app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .articulations
        .get_mut(handle)
        .unwrap() = repaired;
    app.update();
    let converted = view(&app, owner);
    assert_eq!(converted.report.articulation, handle);
    assert_eq!(converted.report.colliders.len(), 1);
    assert_eq!(converted.report.colliders[0].source, attachment);
    assert_eq!(converted.report.colliders[0].source_shape, shape);
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .articulation_colliders
        .get(attachment)
        .is_none());
}

#[test]
fn broken_joint_motor_damping_armature_are_not_surviving_policies() {
    let mut app = app();
    let mut config = authored();
    config.links[1].joint.motor = Some(JointMotor::new(
        Vec3::ZERO,
        Vec3::ZERO,
        1.0,
        0.2,
        Vec3::splat(10.0),
    ));
    config.links[1].joint.damping = 0.5;
    config.links[1].joint.armature = 0.3;
    let (owner, handle) = source(&mut app, config);
    break_tree(&mut app, handle);
    assert_eq!(view(&app, owner).report.joints.len(), 2);
}

#[test]
fn despawn_cleans_owned_disabled_joints_and_preserves_foreign_storage() {
    let mut app = app();
    let (owner, handle) = source(&mut app, authored());
    break_tree(&mut app, handle);
    let converted = view(&app, owner);
    let foreign = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        for mapping in &converted.report.joints {
            physics.joints.get_mut(mapping.joint).unwrap().enabled = false;
        }
        physics.spawn(BodyDesc::dynamic_at(Vec3::splat(50.0)))
    };
    app.world_mut().despawn(owner);
    app.update();
    retired(&app, &converted);
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .contains(foreign));
}

#[test]
fn member_remove_readd_or_mass_edit_retires_group_without_aba_resurrection() {
    for remove_readd in [false, true] {
        let mut app = app();
        let (owner, handle) = source(&mut app, authored());
        break_tree(&mut app, handle);
        let converted = view(&app, owner);
        let member = converted.bodies[2].entity;
        let mut body = *app.world().get::<RigidBody>(member).unwrap();
        if remove_readd {
            app.world_mut().remove::<RigidBody>(member);
        } else {
            body.mass.inv_mass *= 2.0;
        }
        app.world_mut().insert(member, (body,));
        app.update();
        retired(&app, &converted);
        assert!(app
            .world()
            .resource::<ArticulationDiagnostics>()
            .issues
            .iter()
            .any(|d| d.entity == Some(owner) && d.issue == ArticulationIssue::HandoffInvalidated));
        app.update();
        assert!(app
            .world()
            .resource::<ArticulationBindings>()
            .articulation(owner)
            .is_none());
        assert!(app
            .world()
            .resource::<ArticulationBindings>()
            .handoff(owner)
            .is_none());
        assert!(app
            .world()
            .resource::<PhysicsWorldResource>()
            .articulations
            .is_empty());
    }
}

#[test]
fn stale_body_and_entity_generations_never_delete_reused_foreign_slots() {
    #[derive(Component)]
    struct Foreign;
    let mut app = app();
    let (owner, handle) = source(&mut app, authored());
    break_tree(&mut app, handle);
    let converted = view(&app, owner);
    let member = &converted.bodies[2];
    let foreign_body = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        assert!(physics.remove_body(member.body));
        physics.spawn(BodyDesc::dynamic_at(Vec3::splat(40.0)))
    };
    assert_ne!(foreign_body, member.body);
    app.world_mut().despawn(member.entity);
    let foreign_entity = app.world_mut().spawn((Foreign,));
    assert_ne!(foreign_entity, member.entity);
    app.update();
    retired(&app, &converted);
    assert!(app.world().contains(foreign_entity));
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .contains(foreign_body));
}

#[test]
fn observer_removal_during_materialization_cleans_all_allocated_members() {
    let mut app = app();
    let (owner, handle) = source(&mut app, authored());
    let entity_count = app.world().entity_count();
    app.world_mut()
        .observe::<RigidBody, _>(LifecycleEvent::Insert, move |ctx| {
            assert!(ctx.world.contains_resource::<PhysicsWorldResource>());
            assert!(ctx.world.contains_resource::<PhysicsBindings>());
            assert!(ctx.world.contains_resource::<ArticulationBindings>());
            ctx.world.remove::<ArticulationBinding>(owner);
        });
    break_tree(&mut app, handle);
    assert!(app
        .world()
        .resource::<ArticulationBindings>()
        .handoff(owner)
        .is_none());
    assert_eq!(
        app.world().resource::<PhysicsWorldResource>().bodies.len(),
        0
    );
    assert_eq!(
        app.world().resource::<PhysicsWorldResource>().joints.len(),
        0
    );
    assert_eq!(
        app.world_mut()
            .query::<&RigidBody>()
            .iter(app.world())
            .count(),
        0
    );
    assert_eq!(app.world().entity_count(), entity_count);
}

#[test]
fn removing_public_views_readopts_original_handles_without_resetting_core_state() {
    let mut app = app();
    let (owner, handle) = source(&mut app, authored());
    break_tree(&mut app, handle);
    let converted = view(&app, owner);
    app.world_mut().remove_resource::<ArticulationBindings>();
    app.world_mut().remove_resource::<PhysicsBindings>();
    app.update();
    assert_eq!(view(&app, owner), converted);
    for member in &converted.bodies {
        assert_eq!(
            app.world()
                .resource::<PhysicsBindings>()
                .body(member.entity),
            Some(member.body)
        );
    }
    assert_eq!(
        app.world().resource::<PhysicsWorldResource>().bodies.len(),
        4
    );
}

#[test]
fn com_principal_body_frames_preserve_authored_frames_through_sync_and_pose_inputs() {
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        let mut app = app();
        let axes = Mat3::from_quat(Quat::from_rotation_z(0.6) * Quat::from_rotation_x(-0.4));
        let tensor = axes * Mat3::from_diagonal(Vec3::new(1.0, 1.5, 2.0)) * axes.transpose();
        let inertia = SpatialInertia::new(
            2.0,
            Vec3::new(0.3, -0.2, 0.4),
            tensor.transpose().to_cols_array_2d(),
        );
        let mut config = authored();
        config.base = base;
        config.base_inertia = inertia;
        config.base_pose.rotation = Quat::from_rotation_y(0.3);
        config.positions.clear();
        config.velocities.clear();
        for (index, link) in config.links.iter_mut().enumerate() {
            link.inertia = inertia;
            link.joint = ArticulationJoint::fixed();
            link.x_tree =
                SpatialTransform::new(Quat::from_rotation_z(0.2 * (index + 1) as f32), Vec3::Y);
        }
        let (owner, handle) = source(&mut app, config);
        let source_shape = {
            let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
            let shape = physics.shapes.insert(ColliderShape::Sphere { radius: 0.1 });
            let mut desc = ArticulationColliderDesc::new(handle, Some(1), shape);
            desc.local_pose = Isometry::new(Vec3::new(0.2, -0.1, 0.3), Quat::from_rotation_y(-0.2));
            physics.attach_articulation_collider(desc).unwrap();
            shape
        };
        let authored_poses: Vec<_> = app
            .world()
            .resource::<PhysicsWorldResource>()
            .articulations
            .get(handle)
            .unwrap()
            .maximal_body_states()
            .iter()
            .map(|state| state.pose)
            .collect();
        break_tree(&mut app, handle);
        let converted = view(&app, owner);
        assert_eq!(converted.report.colliders.len(), 1);
        let collider_mapping = &converted.report.colliders[0];
        assert_eq!(collider_mapping.source_shape, source_shape);
        assert_ne!(collider_mapping.body_shape, source_shape);
        let adopted_collider = *app
            .world()
            .get::<Collider>(converted.bodies[2].entity)
            .unwrap();
        assert_eq!(
            adopted_collider.shape,
            CollisionShape::Registered(collider_mapping.body_shape)
        );
        assert_authored_frames(app.world(), &converted);
        for (member, expected) in converted.bodies.iter().zip(&authored_poses) {
            assert_global_pose(app.world(), member.authored_entity, *expected);
        }
        for mapping in &converted.report.links {
            assert!(mapping.link_from_body.translation.length() > 0.1);
            assert!(
                (mapping.link_from_body.rotation * Vec3::Z - Vec3::Z).length() > 0.1,
                "the fixture must exercise rotated principal axes as well as COM"
            );
        }
        // An unchanged ordinary sync must not mistake authored link poses for
        // physical body poses and apply the offset a second time.
        app.update();
        assert_eq!(view(&app, owner), converted);
        assert_authored_frames(app.world(), &converted);
        assert_eq!(
            app.world().get::<Collider>(converted.bodies[2].entity),
            Some(&adopted_collider),
            "ordinary sync must not re-bake or offset the core body shape"
        );
        for (member, expected) in converted.bodies.iter().zip(&authored_poses) {
            assert_global_pose(app.world(), member.authored_entity, *expected);
        }

        // A visual authored in link-local coordinates needs no COM/axes edits.
        let member = &converted.bodies[2];
        let attachment_local = Vec3::new(0.2, -0.4, 0.6);
        let attachment =
            app.world_mut()
                .spawn((Transform::from_translation(prism_math::Vec3::new(
                    attachment_local.x,
                    attachment_local.y,
                    attachment_local.z,
                )),));
        app.world_mut()
            .add_relation::<ChildOf>(attachment, member.authored_entity);

        // Move and rotate the detached fixed subtree as a unit through the
        // ordinary body Transform input; surviving joint frames remain valid.
        let delta = Isometry::new(
            Vec3::new(3.0, -0.5, 1.0),
            Quat::from_rotation_x(0.4) * Quat::from_rotation_y(-0.3),
        );
        let expected_bodies: Vec<_> = converted
            .bodies
            .iter()
            .skip(2)
            .map(|member| {
                (
                    member.entity,
                    delta.mul(&physical_pose(app.world(), member.body)),
                )
            })
            .collect();
        for (entity, expected) in &expected_bodies {
            let mut pose = *app.world().get::<Transform>(*entity).unwrap();
            pose.translation = prism_math::Vec3::new(
                expected.translation.x,
                expected.translation.y,
                expected.translation.z,
            );
            pose.rotation = prism_math::Quat::from_xyzw(
                expected.rotation.x,
                expected.rotation.y,
                expected.rotation.z,
                expected.rotation.w,
            );
            app.world_mut().insert(*entity, (pose, Velocity::default()));
        }
        app.update();
        assert_eq!(view(&app, owner), converted);
        assert_authored_frames(app.world(), &converted);
        for (entity, expected) in expected_bodies {
            assert_pose_close(local_pose(app.world(), entity), expected);
        }
        let expected_link = delta.mul(&authored_poses[2]);
        assert_global_pose(app.world(), member.authored_entity, expected_link);
        assert_global_pose(
            app.world(),
            attachment,
            expected_link.mul(&Isometry::from_translation(attachment_local)),
        );

        // Ordinary COM velocity input/solver writeback must keep the frame and
        // its attachments following the moving bodies without another adapter.
        let previous = physical_pose(app.world(), member.body).translation;
        for detached in converted.bodies.iter().skip(2) {
            app.world_mut().insert(
                detached.entity,
                (Velocity {
                    linear: Vec3::X,
                    angular: Vec3::ZERO,
                },),
            );
        }
        app.update();
        assert_eq!(view(&app, owner), converted);
        assert_authored_frames(app.world(), &converted);
        assert!(physical_pose(app.world(), member.body).translation.x > previous.x);
        let current_link = physical_pose(app.world(), member.body)
            .mul(&converted.report.links[1].link_from_body.inverse());
        assert_global_pose(
            app.world(),
            attachment,
            current_link.mul(&Isometry::from_translation(attachment_local)),
        );

        // Rotate the detached pair through physical angular-velocity input as
        // well. Use compatible COM velocities for its surviving fixed joint;
        // assert frame following, not a particular solver integration scheme.
        let previous = physical_pose(app.world(), member.body);
        let angular = Vec3::new(0.3, -0.4, 0.2);
        for detached in converted.bodies.iter().skip(2) {
            let center = physical_pose(app.world(), detached.body).translation;
            app.world_mut().insert(
                detached.entity,
                (Velocity {
                    linear: Vec3::X + angular.cross(center - previous.translation),
                    angular,
                },),
            );
        }
        app.update();
        assert_eq!(view(&app, owner), converted);
        assert_authored_frames(app.world(), &converted);
        let current = physical_pose(app.world(), member.body);
        assert!((current.rotation * Vec3::Y - previous.rotation * Vec3::Y).length() > 1.0e-4);
        let current_link = current.mul(&converted.report.links[1].link_from_body.inverse());
        assert_global_pose(
            app.world(),
            attachment,
            current_link.mul(&Isometry::from_translation(attachment_local)),
        );
        app.world_mut().despawn(owner);
        app.update();
        retired(&app, &converted);
        assert!(!app.world().contains(attachment));
    }
}

#[test]
fn authored_frame_edits_reparenting_or_physics_components_retire_without_respawn() {
    for mutation in 0..6 {
        let mut app = app();
        let (owner, handle) = source(&mut app, authored());
        break_tree(&mut app, handle);
        let converted = view(&app, owner);
        let frame = converted.bodies[2].authored_entity;
        match mutation {
            0 => {
                let mut pose = *app.world().get::<Transform>(frame).unwrap();
                pose.translation.x += 1.0;
                app.world_mut().insert(frame, (pose,));
            }
            1 => {
                app.world_mut().remove::<Transform>(frame);
            }
            2 => {
                app.world_mut()
                    .add_relation::<ChildOf>(frame, converted.bodies[0].entity);
            }
            3 => {
                let body = *app
                    .world()
                    .get::<RigidBody>(converted.bodies[2].entity)
                    .unwrap();
                app.world_mut().insert(frame, (body,));
            }
            4 => {
                app.world_mut().insert(frame, (Collider::sphere(0.1),));
            }
            _ => {
                app.world_mut()
                    .remove_relation::<ChildOf>(frame, converted.bodies[2].entity);
            }
        }
        app.update();
        retired(&app, &converted);
        assert!(app
            .world()
            .resource::<ArticulationDiagnostics>()
            .issues
            .iter()
            .any(|d| d.entity == Some(owner) && d.issue == ArticulationIssue::HandoffInvalidated));
        app.update();
        assert!(app
            .world()
            .resource::<ArticulationBindings>()
            .handoff(owner)
            .is_none());
        assert!(app
            .world()
            .resource::<PhysicsWorldResource>()
            .bodies
            .is_empty());
    }
}

#[test]
fn authored_frame_retirement_preserves_reused_foreign_entity_generation() {
    #[derive(Component)]
    struct Foreign;
    let mut app = app();
    let (owner, handle) = source(&mut app, authored());
    break_tree(&mut app, handle);
    let converted = view(&app, owner);
    let stale = converted.bodies[2].authored_entity;
    assert!(app.world_mut().despawn(stale));
    let replacement = app.world_mut().spawn((Foreign,));
    assert_eq!(stale.index(), replacement.index());
    assert_ne!(stale, replacement);
    app.update();
    retired(&app, &converted);
    assert!(app.world().get::<Foreign>(replacement).is_some());
    assert_eq!(
        app.world().resource::<PhysicsBindings>().body(replacement),
        None
    );
}

#[test]
fn authored_frame_materialization_observer_removal_cleans_both_entity_kinds() {
    let mut app = app();
    let (owner, handle) = source(&mut app, authored());
    let entity_count = app.world().entity_count();
    app.world_mut()
        .observe::<Transform, _>(LifecycleEvent::Insert, move |ctx| {
            if !ctx.world.relation_targets::<ChildOf>(ctx.entity).is_empty() {
                ctx.world.remove::<ArticulationBinding>(owner);
            }
        });
    break_tree(&mut app, handle);
    assert_eq!(app.world().entity_count(), entity_count);
    assert!(app
        .world()
        .resource::<ArticulationBindings>()
        .handoff(owner)
        .is_none());
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .is_empty());
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .joints
        .is_empty());
}
