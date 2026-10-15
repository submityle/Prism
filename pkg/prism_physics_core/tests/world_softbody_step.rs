//! End-to-end fixed-substep tests for world-owned soft-body state.

use glam::Vec3;
use prism_physics_core::state::body::{BodyDesc, BodyKind, MassProperties};
use prism_physics_core::{
    ClothRigidCouplingConfig, ColliderShape, PhysicsWorld, SoftBody, SoftSolverConfig, Solver,
    WorldConfig, XpbdSolver,
};

fn coupling_world(enabled: bool) -> (PhysicsWorld, prism_physics_core::SoftBodyHandle) {
    let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
    if enabled {
        world.cloth_coupling = ClothRigidCouplingConfig::active();
    }

    let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.4 });
    world.spawn(BodyDesc {
        kind: BodyKind::Dynamic,
        position: Vec3::new(0.0, 0.1, 0.0),
        collider: Some(shape),
        mass_properties: MassProperties {
            inv_mass: 20.0,
            inv_inertia: Vec3::ZERO,
        },
        ..BodyDesc::default()
    });

    let mut soft = SoftBody::new(SoftSolverConfig {
        gravity: Vec3::ZERO,
        damping: 0.0,
        substeps: 1,
        iterations: 1,
        ..SoftSolverConfig::default()
    });
    soft.spawn(Vec3::ZERO, 1.0);
    let handle = world.spawn_soft_body(soft);
    (world, handle)
}

fn state(world: &PhysicsWorld, soft: prism_physics_core::SoftBodyHandle) -> (Vec3, Vec3, Vec3) {
    let body = world.soft_bodies.get(soft).expect("soft body");
    (
        body.particles.positions()[0],
        world
            .bodies
            .position(world.bodies.handle_at_slot(0).unwrap())
            .unwrap(),
        world
            .bodies
            .linear_velocity(world.bodies.handle_at_slot(0).unwrap())
            .unwrap(),
    )
}

#[test]
fn fixed_substep_runs_proxy_coupling_and_reaction_writeback() {
    let (mut world, soft) = coupling_world(true);
    let mut solver = XpbdSolver::new();

    solver.step(&mut world, 1.0 / 60.0, 1);

    let (particle, rigid_position, rigid_velocity) = state(&world, soft);
    assert!(
        particle.y < -1e-5,
        "particle was not pushed out: {particle:?}"
    );
    assert!(rigid_position.y > 0.1 + 1e-5);
    assert!(rigid_velocity.y > 1e-5);
}

#[test]
fn empty_world_owned_soft_body_is_a_fixed_step_noop() {
    let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
    let soft = world.spawn_soft_body(SoftBody::with_default_config());
    let mut solver = XpbdSolver::new();
    solver.step(&mut world, 1.0 / 60.0, 4);

    assert_eq!(world.soft_bodies.len(), 1);
    assert!(world.soft_bodies.get(soft).unwrap().particles.is_empty());
}

#[test]
fn replay_from_cloned_boundary_is_bit_identical() {
    let (base, soft) = coupling_world(true);
    let mut replay = base.clone();
    let replay_soft = soft;
    let mut first = base;
    let mut first_solver = XpbdSolver::new();
    let mut replay_solver = XpbdSolver::new();

    for _ in 0..8 {
        first_solver.step(&mut first, 1.0 / 60.0, 2);
        replay_solver.step(&mut replay, 1.0 / 60.0, 2);
    }

    assert_eq!(state(&first, soft), state(&replay, replay_soft));
    assert_eq!(first.soft_bodies, replay.soft_bodies);
}

#[test]
fn disabled_coupling_preserves_rigid_state_while_soft_state_is_owned() {
    let (mut world, soft) = coupling_world(false);
    let rigid = world.bodies.handle_at_slot(0).unwrap();
    let rigid_position = world.bodies.position(rigid).unwrap();
    let rigid_velocity = world.bodies.linear_velocity(rigid).unwrap();
    let mut solver = XpbdSolver::new();

    solver.step(&mut world, 1.0 / 60.0, 1);

    assert_eq!(world.bodies.position(rigid).unwrap(), rigid_position);
    assert_eq!(world.bodies.linear_velocity(rigid).unwrap(), rigid_velocity);
    assert_eq!(
        world.soft_bodies.get(soft).unwrap().particles.positions()[0],
        Vec3::ZERO
    );
}
