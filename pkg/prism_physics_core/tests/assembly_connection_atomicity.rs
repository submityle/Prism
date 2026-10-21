//! World-level assembly edges and break policies must commit together.

use glam::Vec3;
use prism_physics_core::dynamics::assembly::{
    AssemblyError, BreakLimit, ConnectionId, JointLoad, SolverJointLoad,
};
use prism_physics_core::joint::DistanceJoint;
use prism_physics_core::{BodyDesc, BodyHandle, JointDesc, JointHandle, JointKind, PhysicsWorld};

fn fixture() -> (PhysicsWorld, [BodyHandle; 4], JointHandle, JointKind) {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let bodies = [0.0, 1.0, 2.0, 3.0].map(|x| world.spawn(BodyDesc::dynamic_at(Vec3::X * x)));
    let desc = JointDesc::distance(
        bodies[0],
        Vec3::ZERO,
        bodies[1],
        Vec3::ZERO,
        DistanceJoint::rigid(1.0),
    );
    // Register the graph edge explicitly through the API under test, rather
    // than spawn_joint's automatic non-breakable connection.
    let joint = world.joints.insert(desc);
    (world, bodies, joint, desc.kind)
}

fn load(joint: JointHandle, force: f32, torque: f32) -> SolverJointLoad {
    SolverJointLoad {
        joint,
        load: JointLoad {
            linear_impulse: Vec3::X * force * 0.1,
            angular_impulse: Vec3::Y * torque * 0.1,
            substep_dt: 0.1,
        },
    }
}

#[test]
fn duplicate_policy_leaves_entire_world_unchanged() {
    let (mut world, bodies, joint, kind) = fixture();
    let limit = BreakLimit::new(10.0, 10.0).with_fatigue(1.0);
    world.breakable_joints.register(joint, limit).unwrap();
    assert!(world
        .submit_joint_loads(0.1, &[load(joint, 5.0, 0.0)])
        .unwrap()
        .is_empty());
    let before = format!("{world:?}");
    for _ in 0..3 {
        assert_eq!(
            world.connect_assembly(bodies[0], bodies[1], Some(joint), kind, Some(limit)),
            Err(AssemblyError::DuplicatePolicy)
        );
        assert_eq!(
            format!("{world:?}"),
            before,
            "including accumulated fatigue and IDs"
        );
    }
}

#[test]
fn rejected_policy_can_retry_without_consuming_connection_ids() {
    let (mut world, bodies, joint, kind) = fixture();
    let limit = BreakLimit::new(10.0, 10.0);
    world.breakable_joints.register(joint, limit).unwrap();
    assert!(world
        .connect_assembly(bodies[0], bodies[1], Some(joint), kind, Some(limit))
        .is_err());
    assert!(world.breakable_joints.unregister(joint));
    assert_eq!(
        world.connect_assembly(bodies[0], bodies[1], Some(joint), kind, Some(limit)),
        Ok(ConnectionId(1))
    );
    assert_eq!(
        world.connect_assembly(bodies[2], bodies[3], None, kind, None),
        Ok(ConnectionId(2))
    );
    assert!(world.breakable_joints.contains(joint));
}

#[test]
fn rejected_edge_does_not_register_an_orphan_policy() {
    let (mut world, bodies, joint, kind) = fixture();
    world
        .connect_assembly(bodies[0], bodies[1], None, kind, None)
        .unwrap();
    let before = format!("{world:?}");
    assert_eq!(
        world.connect_assembly(
            bodies[1],
            bodies[0],
            Some(joint),
            kind,
            Some(BreakLimit::new(10.0, 10.0))
        ),
        Err(AssemblyError::DuplicateConnection)
    );
    assert_eq!(format!("{world:?}"), before);
    assert!(!world.breakable_joints.contains(joint));
}

#[test]
fn invalid_limits_and_self_edges_leave_no_partial_registration() {
    let (mut world, bodies, joint, kind) = fixture();
    let before = format!("{world:?}");
    for limit in [
        BreakLimit::new(0.0, 10.0),
        BreakLimit::new(10.0, -1.0),
        BreakLimit::new(f32::NAN, 10.0),
        BreakLimit::new(10.0, f32::INFINITY),
        BreakLimit {
            max_force: 10.0,
            max_torque: 10.0,
            fatigue_rate: -1.0,
        },
    ] {
        assert_eq!(
            world.connect_assembly(bodies[0], bodies[1], Some(joint), kind, Some(limit)),
            Err(AssemblyError::InvalidLimit)
        );
        assert_eq!(format!("{world:?}"), before);
    }
    assert_eq!(
        world.connect_assembly(
            bodies[0],
            bodies[0],
            Some(joint),
            kind,
            Some(BreakLimit::new(10.0, 10.0))
        ),
        Err(AssemblyError::SameBody)
    );
    assert_eq!(format!("{world:?}"), before);
}

#[test]
fn successful_registration_breaks_joint_and_disconnects_edge_at_boundary() {
    let (mut world, bodies, joint, kind) = fixture();
    let limit = BreakLimit::new(10.0, 10.0);
    let id = world
        .connect_assembly(bodies[0], bodies[1], Some(joint), kind, Some(limit))
        .unwrap();
    let connection = *world.assemblies.connections().next().unwrap();
    assert_eq!(connection.id, id);
    assert_eq!(connection.break_limit, Some(limit));
    assert!(world.breakable_joints.contains(joint));
    assert!(world
        .submit_joint_loads(0.1, &[load(joint, 5.0, 0.0)])
        .unwrap()
        .is_empty());
    assert!(world.joints.get(joint).unwrap().enabled);
    assert_eq!(world.assemblies.connections().count(), 1);
    let events = world
        .submit_joint_loads(0.1, &[load(joint, 0.0, 20.0)])
        .unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].joint, joint);
    assert!(!world.joints.get(joint).unwrap().enabled);
    assert_eq!(world.assemblies.connections().count(), 0);
    assert!(world
        .submit_joint_loads(0.1, &[load(joint, 0.0, 20.0)])
        .unwrap()
        .is_empty());
}

#[test]
fn non_breakable_and_graph_only_connections_do_not_create_policies() {
    let (mut world, bodies, joint, kind) = fixture();
    world
        .connect_assembly(bodies[0], bodies[1], Some(joint), kind, None)
        .unwrap();
    // Graph-only break metadata has no solver joint on which to register a policy.
    let limit = BreakLimit::new(10.0, 10.0);
    world
        .connect_assembly(bodies[2], bodies[3], None, kind, Some(limit))
        .unwrap();
    assert!(!world.breakable_joints.contains(joint));
    assert_eq!(world.assemblies.connections().count(), 2);
    assert_eq!(
        world.assemblies.connections().last().unwrap().break_limit,
        Some(limit)
    );
}
