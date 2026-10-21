//! Automatic-contact regressions for reduced-coordinate joint limits.
//!
//! These tests intentionally use only the public world/XPBD path.  They do
//! not queue a contact impulse or advance an articulation separately: a
//! dynamic sphere must reach the real narrow-phase contact and the same fixed
//! step must enforce both the joint limit and non-penetration.

use glam::Vec3;
use prism_physics_core::collider::{ColliderShape, PhysicsMaterial};
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::{
    Articulation, ArticulationBase, ArticulationBuilder, ArticulationColliderDesc,
    ArticulationColliderId, ArticulationHandle, ArticulationJoint, JointLimits, JointMotor,
    SpatialInertia, SpatialTransform,
};
use prism_physics_core::solver::{Solver, XpbdSolver};
use prism_physics_core::state::body::{BodyDesc, MassProperties};
use prism_physics_core::state::handle::BodyHandle;
use prism_physics_core::PhysicsWorld;

const RADIUS: f32 = 0.5;
const DIAMETER: f32 = 2.0 * RADIUS;
const DT: f32 = 1.0 / 120.0;
const CONTACT_SLOP: f32 = 0.005;
const INITIAL_GAP: f32 = 0.001;
const LIMIT_EPSILON: f32 = 1.0e-4;
const LIMIT: f32 = 0.35;
const REVOLUTE_ARM: f32 = 1.0;

fn world() -> PhysicsWorld {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    world.config.sleep.enabled = false;
    world
}

fn sphere_inertia(mass: f32) -> SpatialInertia {
    SpatialInertia::diagonal(mass, Vec3::ZERO, Vec3::splat(0.4 * mass * RADIUS * RADIUS))
}

fn fixed_prismatic(q: f32, motor: Option<JointMotor>) -> Articulation {
    let mut joint =
        ArticulationJoint::prismatic(Vec3::X).with_limits(JointLimits::scalar(-LIMIT, LIMIT));
    if let Some(motor) = motor {
        joint = joint.with_motor(motor);
    }
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, sphere_inertia(1.0))
        .with_base_pose(Isometry::IDENTITY)
        .with_gravity(Vec3::ZERO);
    builder
        .add_link(None, SpatialTransform::IDENTITY, sphere_inertia(1.0), joint)
        .unwrap();
    let mut tree = builder.build().unwrap();
    assert!(tree.set_position(0, q));
    tree
}

fn fixed_revolute(q: f32) -> Articulation {
    let joint =
        ArticulationJoint::revolute(Vec3::Z).with_limits(JointLimits::scalar(-LIMIT, LIMIT));
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, sphere_inertia(1.0))
        .with_base_pose(Isometry::IDENTITY)
        .with_gravity(Vec3::ZERO);
    builder
        .add_link(None, SpatialTransform::IDENTITY, sphere_inertia(1.0), joint)
        .unwrap();
    let mut tree = builder.build().unwrap();
    assert!(tree.set_position(0, q));
    tree
}

fn attach_sphere(
    world: &mut PhysicsWorld,
    owner: ArticulationHandle,
    local_pose: Isometry,
) -> ArticulationColliderId {
    let shape = world
        .shapes
        .insert(ColliderShape::Sphere { radius: RADIUS });
    let mut desc = ArticulationColliderDesc::new(owner, Some(0), shape);
    desc.local_pose = local_pose;
    desc.material = PhysicsMaterial::FRICTIONLESS;
    world.attach_articulation_collider(desc).unwrap()
}

fn dynamic_sphere(world: &mut PhysicsWorld, position: Vec3, velocity: Vec3) -> BodyHandle {
    let shape = world
        .shapes
        .insert(ColliderShape::Sphere { radius: RADIUS });
    world.spawn(
        BodyDesc::dynamic_at(position)
            .with_collider(shape)
            .with_mass_properties(MassProperties {
                inv_mass: 1.0,
                inv_inertia: Vec3::splat((0.4 * RADIUS * RADIUS).recip()),
            })
            .with_linear_velocity(velocity)
            .with_material(PhysicsMaterial::FRICTIONLESS)
            .with_sensor(false),
    )
}

fn attachment_center(world: &PhysicsWorld, attachment: ArticulationColliderId) -> Vec3 {
    let desc = world.articulation_colliders.get(attachment).unwrap();
    let tree = world.articulations.get(desc.articulation).unwrap();
    let pose = match desc.link {
        Some(link) => tree.forward_kinematics()[link],
        None => tree.base_pose(),
    };
    pose.transform_point(desc.local_pose.translation)
}

fn assert_nonpenetrating(
    world: &PhysicsWorld,
    attachment: ArticulationColliderId,
    rigid: BodyHandle,
    frame: usize,
) -> f32 {
    let separation =
        (world.bodies.position(rigid).unwrap() - attachment_center(world, attachment)).length();
    assert!(
        separation >= DIAMETER - CONTACT_SLOP,
        "frame {frame}: automatic contact penetrated sphere pair; separation {separation}"
    );
    separation
}

#[test]
fn fixed_base_prismatic_upper_and_lower_limits_hold_against_dynamic_impacts() {
    for &side in &[-1.0, 1.0] {
        let limit = side * LIMIT;
        let mut world = world();
        let owner = world.spawn_articulation(fixed_prismatic(limit, None));
        let attachment = attach_sphere(&mut world, owner, Isometry::IDENTITY);
        let tree_center = attachment_center(&world, attachment);
        let outward = Vec3::X * side;
        let rigid = dynamic_sphere(
            &mut world,
            tree_center - outward * (DIAMETER + INITIAL_GAP),
            outward * 2.0,
        );

        let mut solver = XpbdSolver::new();
        let mut saw_impact_window = false;
        for frame in 0..24 {
            solver.step(&mut world, DT, 1);
            let tree = world.articulations.get(owner).unwrap();
            let q = tree.positions()[0];
            if side > 0.0 {
                assert!(
                    q <= LIMIT + LIMIT_EPSILON,
                    "frame {frame}: prismatic upper limit crossed: q={q}"
                );
            } else {
                assert!(
                    q >= -LIMIT - LIMIT_EPSILON,
                    "frame {frame}: prismatic lower limit crossed: q={q}"
                );
            }
            assert_eq!(tree.base_pose(), Isometry::IDENTITY);
            let separation = assert_nonpenetrating(&world, attachment, rigid, frame);
            saw_impact_window |= separation <= DIAMETER + 0.02;
        }

        assert!(
            saw_impact_window,
            "the dynamic sphere never entered the contact window for the {side}-side limit"
        );
        let outward_speed = world.bodies.linear_velocity(rigid).unwrap().dot(outward);
        assert!(
            outward_speed < 0.25,
            "the dynamic sphere kept travelling through the limited prismatic link: outward speed {outward_speed}"
        );
    }
}

#[test]
fn fixed_base_revolute_upper_and_lower_limits_hold_against_dynamic_impacts() {
    for &side in &[-1.0, 1.0] {
        let limit = side * LIMIT;
        let mut world = world();
        let owner = world.spawn_articulation(fixed_revolute(limit));
        let attachment = attach_sphere(
            &mut world,
            owner,
            Isometry::from_translation(Vec3::X * REVOLUTE_ARM),
        );
        let radial = attachment_center(&world, attachment).normalize();
        let outward = Vec3::Z.cross(radial).normalize() * side;
        let impact_center =
            attachment_center(&world, attachment) - outward * (DIAMETER + INITIAL_GAP);
        let rigid = dynamic_sphere(&mut world, impact_center, outward * 2.0);

        let mut solver = XpbdSolver::new();
        let mut saw_impact_window = false;
        for frame in 0..24 {
            solver.step(&mut world, DT, 1);
            let tree = world.articulations.get(owner).unwrap();
            let q = tree.positions()[0];
            if side > 0.0 {
                assert!(
                    q <= LIMIT + LIMIT_EPSILON,
                    "frame {frame}: revolute upper limit crossed: q={q}"
                );
            } else {
                assert!(
                    q >= -LIMIT - LIMIT_EPSILON,
                    "frame {frame}: revolute lower limit crossed: q={q}"
                );
            }
            assert_eq!(tree.base_pose(), Isometry::IDENTITY);
            let separation = assert_nonpenetrating(&world, attachment, rigid, frame);
            saw_impact_window |= separation <= DIAMETER + 0.02;
        }

        assert!(
            saw_impact_window,
            "the dynamic sphere never entered the revolute contact window for the {side}-side limit"
        );
        let outward_speed = world.bodies.linear_velocity(rigid).unwrap().dot(outward);
        assert!(
            outward_speed < 0.25,
            "the dynamic sphere kept travelling through the limited revolute link: outward speed {outward_speed}"
        );
    }
}

#[test]
fn active_revolute_limit_allows_contact_to_move_away_from_the_bound() {
    for &side in &[-1.0, 1.0] {
        let limit = side * LIMIT;
        let mut world = world();
        let owner = world.spawn_articulation(fixed_revolute(limit));
        let attachment = attach_sphere(
            &mut world,
            owner,
            Isometry::from_translation(Vec3::X * REVOLUTE_ARM),
        );
        let center = attachment_center(&world, attachment);
        let radial = center.normalize();
        let forbidden = Vec3::Z.cross(radial).normalize() * side;
        let rigid = dynamic_sphere(
            &mut world,
            center + forbidden * (DIAMETER + INITIAL_GAP),
            -forbidden * 2.0,
        );

        let mut solver = XpbdSolver::new();
        let mut moved_away = false;
        for frame in 0..12 {
            solver.step(&mut world, DT, 1);
            let q = world.articulations.get(owner).unwrap().positions()[0];
            assert_nonpenetrating(&world, attachment, rigid, frame);
            moved_away |= if side > 0.0 {
                q < LIMIT - LIMIT_EPSILON
            } else {
                q > -LIMIT + LIMIT_EPSILON
            };
        }
        assert!(
            moved_away,
            "contact incorrectly locked the articulation at the {side}-side unilateral limit"
        );
    }
}

#[test]
fn automatic_contact_does_not_spend_more_than_motor_budget_or_allow_limit_penetration() {
    const MAX_FORCE: f32 = 0.25;

    // First isolate the actuator budget from contact reactions.  A large
    // position error saturates the motor, so one fixed step may add no more
    // generalized speed than F_max * dt / m to this unit-mass prismatic link.
    let motor = JointMotor::new(
        Vec3::new(100.0, 0.0, 0.0),
        Vec3::ZERO,
        1_000.0,
        0.0,
        Vec3::splat(MAX_FORCE),
    );
    let mut motor_only = world();
    let motor_owner = motor_only.spawn_articulation(fixed_prismatic(0.0, Some(motor)));
    let mut solver = XpbdSolver::new();
    let before = motor_only
        .articulations
        .get(motor_owner)
        .unwrap()
        .velocities()[0];
    solver.step(&mut motor_only, DT, 1);
    let after = motor_only
        .articulations
        .get(motor_owner)
        .unwrap()
        .velocities()[0];
    let delta = after - before;
    assert!(
        delta > 0.0,
        "the saturated motor did not produce a drive impulse"
    );
    assert!(
        delta <= MAX_FORCE * DT + 5.0e-5,
        "motor exceeded max_force budget: delta_qd={delta}, budget={} ",
        MAX_FORCE * DT
    );

    // The same bounded motor is active while a dynamic sphere drives the
    // limited joint from the inside toward its upper stop.  Contact must still
    // resolve the rigid body even though the actuator budget is deliberately
    // much smaller than the incoming contact momentum.
    let mut world = world();
    let owner = world.spawn_articulation(fixed_prismatic(LIMIT, Some(motor)));
    let attachment = attach_sphere(&mut world, owner, Isometry::IDENTITY);
    let tree_center = attachment_center(&world, attachment);
    let rigid = dynamic_sphere(
        &mut world,
        tree_center - Vec3::X * (DIAMETER + INITIAL_GAP),
        Vec3::X * 2.0,
    );

    let mut solver = XpbdSolver::new();
    for frame in 0..24 {
        solver.step(&mut world, DT, 1);
        let tree = world.articulations.get(owner).unwrap();
        assert!(
            tree.positions()[0] <= LIMIT + LIMIT_EPSILON,
            "frame {frame}: motor/contact solve crossed prismatic upper limit: q={}",
            tree.positions()[0]
        );
        assert_nonpenetrating(&world, attachment, rigid, frame);
    }
    assert!(
        world.bodies.linear_velocity(rigid).unwrap().x < 0.25,
        "contact response was lost behind the bounded motor budget: velocity={:?}",
        world.bodies.linear_velocity(rigid).unwrap()
    );
}
