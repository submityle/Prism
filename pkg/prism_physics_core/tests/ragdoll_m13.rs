//! M13 integration coverage for authored ragdolls entering the XPBD fixed-step.
//!
//! This test crosses the authoring boundary: it builds a sorted ragdoll
//! definition, binds its parts to live PhysicsWorld bodies, and advances the
//! actual XpbdSolver. Unit tests of the swing-twist primitive cannot prove
//! that handle ordering and fixed-step solver registration are correct together.

use glam::{Quat, Vec3};
use prism_physics_core::joint::ragdoll::{
    RagdollBinding, RagdollBuilder, RagdollLink, RagdollPart, RagdollPartId,
};
use prism_physics_core::joint::{SwingTwistConfig, SwingTwistJoint};
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::solver::{Solver, XpbdConfig, XpbdSolver};
use prism_physics_core::state::body::BodyDesc;
use prism_physics_core::world::PhysicsWorld;

fn swing_twist() -> SwingTwistJoint {
    SwingTwistJoint::try_new(SwingTwistConfig::symmetric(
        Vec3::X,
        Vec3::X,
        Vec3::Y,
        Vec3::Y,
        0.55,
        0.4,
    ))
    .expect("the integration fixture must use a valid swing-twist contract")
}

#[test]
fn authored_ragdoll_binding_reaches_fixed_step_and_closes_twist_limit() {
    let mut builder = RagdollBuilder::new();
    builder
        .add_part(RagdollPart {
            id: RagdollPartId::new(20),
            body: BodyDesc::static_at(Vec3::ZERO),
            local_pose: Isometry::IDENTITY,
        })
        .expect("parent part should be accepted");
    builder
        .add_part(RagdollPart {
            id: RagdollPartId::new(10),
            body: BodyDesc::dynamic_at(Vec3::ZERO),
            local_pose: Isometry::IDENTITY,
        })
        .expect("child part should be accepted");

    let joint = swing_twist();
    builder
        .add_link(RagdollLink {
            parent: RagdollPartId::new(20),
            child: RagdollPartId::new(10),
            anchor_parent: Vec3::ZERO,
            anchor_child: Vec3::ZERO,
            joint,
        })
        .expect("parent-child link should be accepted");
    let definition = builder.build().expect("fixture topology should be valid");

    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let mut bodies = Vec::with_capacity(definition.part_count());
    for part in &definition.parts {
        let mut body = part.body;
        if part.id == RagdollPartId::new(10) {
            body.orientation = Quat::from_rotation_x(1.2);
        }
        bodies.push(world.spawn(body));
    }
    let binding = RagdollBinding::try_new(definition, bodies.clone())
        .expect("sorted live handles should bind to the sorted definition");
    assert_eq!(binding.constraints().len(), 1);
    assert_eq!(
        binding.body(RagdollPartId::new(10)),
        Some(bodies[0]),
        "binding must preserve definition order after authoring IDs are sorted"
    );
    assert_eq!(
        binding.body(RagdollPartId::new(20)),
        Some(bodies[1]),
        "binding must map the parent to the second sorted slot"
    );

    let parent = bodies[1];
    let child = bodies[0];
    let parent_before = world.bodies.orientation(parent).expect("live parent");
    let mut solver = XpbdSolver::with_config(XpbdConfig {
        position_iterations: 8,
        parallel_islands: false,
        ..XpbdConfig::default()
    });
    solver = XpbdSolver::with_ragdoll_binding(solver.config, &binding);

    for _ in 0..4 {
        solver.step(&mut world, 1.0 / 60.0, 1);
    }

    let parent_after = world.bodies.orientation(parent).expect("live parent");
    let child_after = world.bodies.orientation(child).expect("live child");
    let measurement = joint.measure(parent_after, child_after);
    assert_eq!(
        parent_after, parent_before,
        "static parent must not be projected"
    );
    assert!(measurement.twist_violation.abs() < 1.0e-4);
    assert!(measurement.swing_violation < 1.0e-4);
    assert!(child_after.is_finite());
    assert!(world
        .bodies
        .position(child)
        .expect("live child")
        .is_finite());
}
