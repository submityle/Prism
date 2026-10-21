//! ECS authoring, fixed-step, and generational lifecycle coverage for M13 ragdolls.

use std::time::Duration;

use prism_app::{App, TimeUpdateStrategy};
use prism_ecs::prelude::*;
use prism_physics_ecs::core::joint::ragdoll::{
    RagdollBuilder, RagdollLink, RagdollPart, RagdollPartId,
};
use prism_physics_ecs::core::joint::{SwingTwistConfig, SwingTwistJoint};
use prism_physics_ecs::core::math::transform::Isometry;
use prism_physics_ecs::core::solver::XpbdConfig;
use prism_physics_ecs::core::{BodyDesc, BodyHandle};
use prism_physics_ecs::*;
use prism_transform::Transform;

const ROOT: RagdollPartId = RagdollPartId::new(10);
const LIMB: RagdollPartId = RagdollPartId::new(20);

fn app() -> App {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)))
        .set_fixed_timestep(Duration::from_millis(10))
        .add_plugins(PrismPhysicsPlugin);
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .gravity = Vec3::ZERO;
    app
}

fn joint() -> SwingTwistJoint {
    SwingTwistJoint::try_new(SwingTwistConfig::symmetric(
        Vec3::X,
        Vec3::X,
        Vec3::Y,
        Vec3::Y,
        0.45,
        0.35,
    ))
    .expect("test joint is valid")
}

fn definition() -> core::joint::ragdoll::RagdollDefinition {
    let mut builder = RagdollBuilder::new();
    // Deliberately add in reverse ID order: the definition contract is sorted.
    builder
        .add_part(RagdollPart {
            id: LIMB,
            body: BodyDesc::dynamic_at(Vec3::ZERO),
            local_pose: Isometry::IDENTITY,
        })
        .unwrap();
    builder
        .add_part(RagdollPart {
            id: ROOT,
            body: BodyDesc::dynamic_at(Vec3::ZERO),
            local_pose: Isometry::IDENTITY,
        })
        .unwrap();
    builder
        .add_link(RagdollLink {
            parent: ROOT,
            child: LIMB,
            anchor_parent: Vec3::ZERO,
            anchor_child: Vec3::ZERO,
            joint: joint(),
        })
        .unwrap();
    builder.build().unwrap()
}

fn body(app: &mut App, transform: Transform) -> Entity {
    app.world_mut()
        .spawn((RigidBody::dynamic(), transform, Velocity::default()))
}

fn author(app: &mut App, root: Entity, limb: Entity) -> Entity {
    app.world_mut()
        .spawn(EcsRagdoll::new(definition(), [(LIMB, limb), (ROOT, root)]))
}

#[test]
fn public_authoring_is_canonical_and_constraints_run_in_fixed_step() {
    let mut app = app();
    let root = body(&mut app, Transform::IDENTITY);
    let limb = body(
        &mut app,
        Transform::from_rotation(prism_math::Quat::from_rotation_x(1.1)),
    );
    let owner = author(&mut app, root, limb);

    // Direct component construction is public too; canonicalize the mapping
    // against definition order even when callers bypass EcsRagdoll::new.
    let authored = EcsRagdoll {
        definition: definition(),
        parts: vec![(LIMB, limb), (ROOT, root)],
    };
    *app.world_mut().get_mut::<EcsRagdoll>(owner).unwrap() = authored;
    app.world_mut().resource_mut::<PhysicsSolver>().0.config = XpbdConfig {
        position_iterations: 8,
        parallel_islands: false,
        ..XpbdConfig::default()
    };

    for _ in 0..8 {
        app.update();
    }

    let bindings = app.world().resource::<EcsRagdollBindings>();
    let binding = bindings.ragdoll(owner).expect("ragdoll is active");
    assert_eq!(binding.constraints().len(), 1);
    assert_eq!(
        binding.body(ROOT),
        app.world().resource::<PhysicsBindings>().body(root)
    );
    assert_eq!(
        binding.body(LIMB),
        app.world().resource::<PhysicsBindings>().body(limb)
    );
    assert_eq!(
        app.world()
            .resource::<PhysicsSolver>()
            .0
            .config
            .position_iterations,
        8,
        "reconciliation must retain caller solver configuration"
    );
    let root_pose = app.world().get::<Transform>(root).unwrap();
    let limb_pose = app.world().get::<Transform>(limb).unwrap();
    let error = joint().measure(
        Quat::from_xyzw(
            root_pose.rotation.x,
            root_pose.rotation.y,
            root_pose.rotation.z,
            root_pose.rotation.w,
        ),
        Quat::from_xyzw(
            limb_pose.rotation.x,
            limb_pose.rotation.y,
            limb_pose.rotation.z,
            limb_pose.rotation.w,
        ),
    );
    assert!(error.twist_violation.abs() < 2.0e-3, "{error:?}");
    assert!(error.swing_violation < 2.0e-3, "{error:?}");
    assert!(app
        .world()
        .resource::<EcsRagdollDiagnostics>()
        .issues
        .is_empty());
}

#[test]
fn replacement_removal_and_despawn_reconcile_bindings() {
    let mut app = app();
    let root = body(&mut app, Transform::IDENTITY);
    let first_limb = body(&mut app, Transform::IDENTITY);
    let replacement_limb = body(&mut app, Transform::from_xyz(0.0, 1.0, 0.0));
    let owner = author(&mut app, root, first_limb);
    app.update();
    let first_handle = app
        .world()
        .resource::<EcsRagdollBindings>()
        .body(owner, LIMB)
        .unwrap();

    *app.world_mut().get_mut::<EcsRagdoll>(owner).unwrap() =
        EcsRagdoll::new(definition(), [(ROOT, root), (LIMB, replacement_limb)]);
    app.update();
    let replacement_handle = app
        .world()
        .resource::<EcsRagdollBindings>()
        .body(owner, LIMB)
        .unwrap();
    assert_ne!(replacement_handle, first_handle);
    assert_eq!(
        replacement_handle,
        app.world()
            .resource::<PhysicsBindings>()
            .body(replacement_limb)
            .unwrap()
    );

    app.world_mut().remove::<EcsRagdoll>(owner);
    app.update();
    assert!(app
        .world()
        .resource::<EcsRagdollBindings>()
        .ragdoll(owner)
        .is_none());

    let second_owner = author(&mut app, root, replacement_limb);
    app.update();
    assert_eq!(app.world().resource::<EcsRagdollBindings>().len(), 1);
    assert!(app.world_mut().despawn(second_owner));
    app.update();
    assert!(app.world().resource::<EcsRagdollBindings>().is_empty());
}

#[test]
fn stale_core_handle_is_replaced_and_ragdoll_rebound_by_generation() {
    let mut app = app();
    let root = body(&mut app, Transform::IDENTITY);
    let limb = body(&mut app, Transform::IDENTITY);
    let owner = author(&mut app, root, limb);
    app.update();
    let old_handle = app
        .world()
        .resource::<EcsRagdollBindings>()
        .body(owner, LIMB)
        .unwrap();
    assert!(app
        .world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .remove_body(old_handle));

    app.update();

    let current = app
        .world()
        .resource::<EcsRagdollBindings>()
        .body(owner, LIMB)
        .expect("the ECS body is respawned and rebound");
    assert_ne!(current, old_handle, "generation must prevent stale reuse");
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .contains(current));
    assert_eq!(
        current,
        app.world()
            .resource::<PhysicsBindings>()
            .body(limb)
            .unwrap()
    );
    assert!(app
        .world()
        .resource::<EcsRagdollDiagnostics>()
        .issues
        .is_empty());
}

#[test]
fn edge_validation_rejects_duplicate_mapping_and_recovers_after_missing_body() {
    let mut app = app();
    let root = body(&mut app, Transform::IDENTITY);
    let limb = body(&mut app, Transform::IDENTITY);
    let missing = app.world_mut().spawn(Transform::IDENTITY);
    let owner = author(&mut app, root, limb);

    *app.world_mut().get_mut::<EcsRagdoll>(owner).unwrap() = EcsRagdoll {
        definition: definition(),
        parts: vec![(ROOT, root), (ROOT, limb)],
    };
    app.update();
    assert!(app
        .world()
        .resource::<EcsRagdollBindings>()
        .ragdoll(owner)
        .is_none());
    assert_eq!(
        app.world().resource::<EcsRagdollDiagnostics>().issues[0].issue,
        EcsRagdollIssue::InvalidPartMapping
    );

    *app.world_mut().get_mut::<EcsRagdoll>(owner).unwrap() = EcsRagdoll {
        definition: definition(),
        parts: vec![(ROOT, root), (LIMB, root)],
    };
    app.update();
    assert_eq!(
        app.world().resource::<EcsRagdollDiagnostics>().issues[0].issue,
        EcsRagdollIssue::InvalidPartMapping
    );

    *app.world_mut().get_mut::<EcsRagdoll>(owner).unwrap() =
        EcsRagdoll::new(definition(), [(ROOT, root), (LIMB, missing)]);
    app.update();
    assert_eq!(
        app.world().resource::<EcsRagdollDiagnostics>().issues[0].issue,
        EcsRagdollIssue::MissingBody
    );

    app.world_mut()
        .insert(missing, (RigidBody::dynamic(), Velocity::default()));
    app.update();
    let binding = app
        .world()
        .resource::<EcsRagdollBindings>()
        .ragdoll(owner)
        .expect("valid body mapping should recover");
    assert_ne!(binding.body(LIMB), Some(BodyHandle::INVALID));
    assert!(app
        .world()
        .resource::<EcsRagdollDiagnostics>()
        .issues
        .is_empty());
}
