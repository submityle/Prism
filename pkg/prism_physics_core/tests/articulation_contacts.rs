//! Public-API acceptance tests for automatic world/articulation contacts.
//!
//! Every contact response enters through the default XPBD solver. No test
//! queues contact impulses or advances an articulation separately. These are
//! integration contracts, independent of the analytical dynamics regressions.

use glam::{Quat, Vec3};
use prism_physics_core::collider::{ColliderShape, CompoundChild, PhysicsMaterial, ShapeRegistry};
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::{
    Articulation, ArticulationBase, ArticulationBuilder, ArticulationColliderDesc,
    ArticulationColliderError, ArticulationColliderId, ArticulationHandle, ArticulationJoint,
    SpatialInertia, SpatialMotion, SpatialTransform,
};
use prism_physics_core::solver::{Solver, XpbdSolver};
use prism_physics_core::state::body::{BodyDesc, MassProperties};
use prism_physics_core::state::handle::BodyHandle;
use prism_physics_core::PhysicsWorld;

const RADIUS: f32 = 0.5;
const DT: f32 = 1.0 / 120.0;
const CONTACT_SLOP: f32 = 0.005;

fn close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

fn close_vec(actual: Vec3, expected: Vec3, tolerance: f32) {
    assert!(
        actual.is_finite() && (actual - expected).length() <= tolerance,
        "actual {actual:?}, expected {expected:?}, tolerance {tolerance}"
    );
}

fn world(gravity: Vec3) -> PhysicsWorld {
    let mut world = PhysicsWorld::with_gravity(gravity);
    // Sleeping must not mask a missing response or discard collision momentum.
    // Leave the solver itself at its default contact iteration policy.
    world.config.sleep.enabled = false;
    world
}

fn sphere_inertia(mass: f32) -> SpatialInertia {
    SpatialInertia::diagonal(mass, Vec3::ZERO, Vec3::splat(0.4 * mass * RADIUS * RADIUS))
}

fn floating_tree(position: Vec3, velocity: Vec3, gravity: Vec3) -> Articulation {
    // Coincident base and fixed link each contribute mass one: the composite
    // has total mass two and the inertia of a solid sphere of that mass.
    let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, sphere_inertia(1.0))
        .with_base_pose(Isometry::from_translation(position))
        .with_gravity(gravity);
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            sphere_inertia(1.0),
            ArticulationJoint::fixed(),
        )
        .unwrap();
    let mut tree = builder.build().unwrap();
    assert!(tree.set_base_velocity(SpatialMotion::new(Vec3::ZERO, velocity)));
    tree
}

fn floating_sphere(
    world: &mut PhysicsWorld,
    position: Vec3,
    velocity: Vec3,
    material: PhysicsMaterial,
    local_offset: Vec3,
) -> (ArticulationHandle, ArticulationColliderId) {
    let owner = world.spawn_articulation(floating_tree(position, velocity, world.config.gravity));
    let shape = world
        .shapes
        .insert(ColliderShape::Sphere { radius: RADIUS });
    let mut desc = ArticulationColliderDesc::new(owner, Some(0), shape);
    desc.material = material;
    desc.local_pose = Isometry::from_translation(local_offset);
    let attachment = world.attach_articulation_collider(desc).unwrap();
    (owner, attachment)
}

fn ground(world: &mut PhysicsWorld, material: PhysicsMaterial, sensor: bool) -> BodyHandle {
    let shape = world.shapes.insert(ColliderShape::Plane {
        normal: Vec3::Y,
        offset: 0.0,
    });
    world.spawn(
        BodyDesc::static_at(Vec3::ZERO)
            .with_collider(shape)
            .with_material(material)
            .with_sensor(sensor),
    )
}

fn rigid_sphere(
    world: &mut PhysicsWorld,
    position: Vec3,
    velocity: Vec3,
    mass: f32,
    material: PhysicsMaterial,
    sensor: bool,
) -> BodyHandle {
    let shape = world
        .shapes
        .insert(ColliderShape::Sphere { radius: RADIUS });
    world.spawn(
        BodyDesc::dynamic_at(position)
            .with_collider(shape)
            .with_mass_properties(MassProperties {
                inv_mass: mass.recip(),
                inv_inertia: Vec3::splat((0.4 * mass * RADIUS * RADIUS).recip()),
            })
            .with_linear_velocity(velocity)
            .with_material(material)
            .with_sensor(sensor),
    )
}

fn center(world: &PhysicsWorld, attachment: ArticulationColliderId) -> Vec3 {
    let desc = world.articulation_colliders.get(attachment).unwrap();
    let tree = world.articulations.get(desc.articulation).unwrap();
    let pose = match desc.link {
        Some(link) => tree.forward_kinematics()[link],
        None => tree.base_pose(),
    };
    pose.transform_point(desc.local_pose.translation)
}

fn link_linear_velocity(world: &PhysicsWorld, owner: ArticulationHandle) -> Vec3 {
    let tree = world.articulations.get(owner).unwrap();
    tree.contact_point_velocity(0, tree.forward_kinematics()[0].translation)
        .unwrap()
}

fn assert_same_tree(left: &PhysicsWorld, right: &PhysicsWorld, owner: ArticulationHandle) {
    let left = left.articulations.get(owner).unwrap();
    let right = right.articulations.get(owner).unwrap();
    assert_eq!(left.base_pose(), right.base_pose());
    assert_eq!(left.base_velocity(), right.base_velocity());
    assert_eq!(left.positions(), right.positions());
    assert_eq!(left.velocities(), right.velocities());
    assert_eq!(left.forward_kinematics(), right.forward_kinematics());
    assert_eq!(left.broken_joint(), right.broken_joint());
}

#[test]
fn floating_fixed_link_sphere_falls_and_rests_at_its_attachment_surface() {
    let mut world = world(Vec3::new(0.0, -9.81, 0.0));
    ground(&mut world, PhysicsMaterial::FRICTIONLESS, false);
    // A vertical collider offset checks link/local-pose composition without
    // introducing a torque from the normal impulse.
    let (owner, attachment) = floating_sphere(
        &mut world,
        Vec3::Y * 2.0,
        Vec3::ZERO,
        PhysicsMaterial::FRICTIONLESS,
        -Vec3::Y * 0.25,
    );
    let initial_height = center(&world, attachment).y;
    let mut solver = XpbdSolver::new();
    let mut fell = false;
    for frame in 0..180 {
        solver.step(&mut world, 1.0 / 60.0, 4);
        let height = center(&world, attachment).y;
        let velocity = link_linear_velocity(&world, owner);
        fell |= velocity.y < -0.5 && height < initial_height - 0.1;
        assert!(
            height >= RADIUS - CONTACT_SLOP,
            "frame {frame}: height {height}"
        );
        if frame >= 150 {
            close(height, RADIUS, CONTACT_SLOP);
            close_vec(velocity, Vec3::ZERO, 0.05);
        }
    }
    assert!(fell, "a frozen articulation is not a successful landing");
    close(
        world
            .articulations
            .get(owner)
            .unwrap()
            .base_pose()
            .translation
            .y,
        0.75,
        CONTACT_SLOP,
    );
}

#[test]
fn prismatic_link_ground_contact_stops_fall_without_moving_fixed_base() {
    let mut world = world(-Vec3::Y * 9.81);
    ground(&mut world, PhysicsMaterial::FRICTIONLESS, false);
    let base_pose = Isometry::from_translation(Vec3::Y * 2.0);
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, sphere_inertia(1.0))
        .with_base_pose(base_pose)
        .with_gravity(world.config.gravity);
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            sphere_inertia(2.0),
            ArticulationJoint::prismatic(Vec3::Y),
        )
        .unwrap();
    let owner = world.spawn_articulation(builder.build().unwrap());
    let shape = world
        .shapes
        .insert(ColliderShape::Sphere { radius: RADIUS });
    let attachment = world
        .attach_articulation_collider(ArticulationColliderDesc::new(owner, Some(0), shape))
        .unwrap();
    let mut solver = XpbdSolver::new();
    let mut fell = false;
    for frame in 0..180 {
        solver.step(&mut world, 1.0 / 60.0, 4);
        let tree = world.articulations.get(owner).unwrap();
        fell |= tree.positions()[0] < -0.1 && tree.velocities()[0] < -0.5;
        assert_eq!(tree.base_pose(), base_pose);
        let position = center(&world, attachment);
        assert!(
            position.y >= RADIUS - CONTACT_SLOP,
            "frame {frame}: {position:?}"
        );
        close(position.x, 0.0, 1.0e-6);
        close(position.z, 0.0, 1.0e-6);
    }
    let tree = world.articulations.get(owner).unwrap();
    assert!(fell);
    close(tree.positions()[0], -1.5, CONTACT_SLOP);
    close(tree.velocities()[0], 0.0, 0.05);
}

#[test]
fn dynamic_rigid_collision_transfers_impulse_and_conserves_total_momentum() {
    let mut world = world(Vec3::ZERO);
    let (owner, attachment) = floating_sphere(
        &mut world,
        -Vec3::X * 0.55,
        Vec3::X,
        PhysicsMaterial::FRICTIONLESS,
        Vec3::ZERO,
    );
    let rigid = rigid_sphere(
        &mut world,
        Vec3::X * 0.55,
        -Vec3::X,
        3.0,
        PhysicsMaterial::FRICTIONLESS,
        false,
    );
    // Articulation mass two, rigid mass three: p = 2 - 3 = -1.
    let expected_momentum = -Vec3::X;
    let mut solver = XpbdSolver::new();
    for frame in 0..24 {
        solver.step(&mut world, DT, 1);
        let va = link_linear_velocity(&world, owner);
        let vb = world.bodies.linear_velocity(rigid).unwrap();
        close_vec(2.0 * va + 3.0 * vb, expected_momentum, 2.0e-3);
        assert!(va.is_finite() && vb.is_finite(), "frame {frame}");
        let separation = world.bodies.position(rigid).unwrap().x - center(&world, attachment).x;
        assert!(
            separation >= 2.0 * RADIUS - CONTACT_SLOP,
            "frame {frame}: separation {separation}"
        );
        let energy = va.length_squared() + 1.5 * vb.length_squared();
        assert!(
            energy <= 2.5 + 2.0e-3,
            "frame {frame}: kinetic energy {energy}"
        );
    }
    // Merely conserving momentum during free flight would not prove contact.
    close_vec(link_linear_velocity(&world, owner), -Vec3::X * 0.2, 0.02);
    close_vec(
        world.bodies.linear_velocity(rigid).unwrap(),
        -Vec3::X * 0.2,
        0.02,
    );
    close_vec(
        world
            .articulations
            .get(owner)
            .unwrap()
            .base_velocity()
            .angular,
        Vec3::ZERO,
        1.0e-4,
    );
    close_vec(
        world.bodies.angular_velocity(rigid).unwrap(),
        Vec3::ZERO,
        1.0e-4,
    );
}

fn plane_impact(
    material: PhysicsMaterial,
    velocity: Vec3,
) -> (PhysicsWorld, ArticulationHandle, ArticulationColliderId) {
    let mut world = world(Vec3::ZERO);
    ground(&mut world, material, false);
    let (owner, attachment) = floating_sphere(
        &mut world,
        Vec3::Y * (RADIUS + 0.001),
        velocity,
        material,
        Vec3::ZERO,
    );
    XpbdSolver::new().step(&mut world, DT, 1);
    assert!(center(&world, attachment).y >= RADIUS - CONTACT_SLOP);
    (world, owner, attachment)
}

#[test]
fn restitution_uses_incoming_normal_speed_not_split_position_correction() {
    for restitution in [0.0, 1.0] {
        let (world, owner, _) =
            plane_impact(PhysicsMaterial::new(0.0, restitution), -Vec3::Y * 2.0);
        // Plane is static; normal a->b must stop/reverse downward motion,
        // without turning depenetration distance into extra kinetic energy.
        close_vec(
            link_linear_velocity(&world, owner),
            Vec3::Y * (2.0 * restitution),
            0.03,
        );
        close_vec(
            world
                .articulations
                .get(owner)
                .unwrap()
                .base_velocity()
                .angular,
            Vec3::ZERO,
            1.0e-4,
        );
    }
}

#[test]
fn friction_reduces_tangential_contact_slip_and_can_generate_spin() {
    let incoming = Vec3::new(2.0, -2.0, 0.0);
    let (smooth, smooth_owner, _) = plane_impact(PhysicsMaterial::FRICTIONLESS, incoming);
    let (rough, rough_owner, attachment) = plane_impact(PhysicsMaterial::new(1.0, 0.0), incoming);
    close(link_linear_velocity(&smooth, smooth_owner).x, 2.0, 1.0e-4);
    close_vec(
        smooth
            .articulations
            .get(smooth_owner)
            .unwrap()
            .base_velocity()
            .angular,
        Vec3::ZERO,
        1.0e-4,
    );
    let velocity = link_linear_velocity(&rough, rough_owner);
    assert!(
        velocity.x > 0.0 && velocity.x < 1.9,
        "rough-surface velocity {velocity:?}"
    );
    close(velocity.y, 0.0, 0.03);
    let tree = rough.articulations.get(rough_owner).unwrap();
    let surface_point = center(&rough, attachment) - Vec3::Y * RADIUS;
    let surface_velocity = tree.contact_point_velocity(0, surface_point).unwrap();
    assert!(
        surface_velocity.x.abs() < 0.25,
        "remaining slip {surface_velocity:?}"
    );
    assert!(
        tree.base_velocity().angular.length() > 0.1,
        "friction must not freeze translation instead of allowing rolling"
    );
}

#[test]
fn sensor_ground_is_identical_to_free_flight() {
    let mut world = world(-Vec3::Y * 9.81);
    let sensor = ground(&mut world, PhysicsMaterial::new(1.0, 1.0), true);
    let (owner, attachment) = floating_sphere(
        &mut world,
        Vec3::Y,
        Vec3::ZERO,
        PhysicsMaterial::new(1.0, 1.0),
        Vec3::ZERO,
    );
    let mut free = world.clone();
    assert!(free.remove_body(sensor));
    let mut solver = XpbdSolver::new();
    let mut free_solver = solver.clone();
    for _ in 0..90 {
        solver.step(&mut world, DT, 2);
        free_solver.step(&mut free, DT, 2);
        assert_same_tree(&world, &free, owner);
    }
    assert!(
        center(&world, attachment).y < -RADIUS,
        "the sensor plane must be crossed"
    );
    assert_eq!(world.bodies.position(sensor), Some(Vec3::ZERO));
}

#[test]
fn moving_dynamic_sensor_exchanges_no_impulses_with_articulation() {
    let mut world = world(Vec3::ZERO);
    let (owner, attachment) = floating_sphere(
        &mut world,
        -Vec3::X * 0.6,
        Vec3::X,
        PhysicsMaterial::new(1.0, 1.0),
        Vec3::ZERO,
    );
    let sensor = rigid_sphere(
        &mut world,
        Vec3::X * 0.6,
        -Vec3::X,
        3.0,
        PhysicsMaterial::new(1.0, 1.0),
        true,
    );
    let mut solver = XpbdSolver::new();
    for _ in 0..90 {
        solver.step(&mut world, DT, 1);
        close_vec(link_linear_velocity(&world, owner), Vec3::X, 1.0e-5);
        close_vec(
            world.bodies.linear_velocity(sensor).unwrap(),
            -Vec3::X,
            1.0e-5,
        );
        close_vec(
            world
                .articulations
                .get(owner)
                .unwrap()
                .base_velocity()
                .angular,
            Vec3::ZERO,
            1.0e-5,
        );
        close_vec(
            world.bodies.angular_velocity(sensor).unwrap(),
            Vec3::ZERO,
            1.0e-5,
        );
    }
    close(center(&world, attachment).x, 0.15, 1.0e-5);
    close(world.bodies.position(sensor).unwrap().x, -0.15, 1.0e-5);
}

#[test]
fn removing_attachment_disables_contact_and_old_id_cannot_remove_replacement() {
    let mut world = world(Vec3::ZERO);
    ground(&mut world, PhysicsMaterial::FRICTIONLESS, false);
    let (owner, old_id) = floating_sphere(
        &mut world,
        Vec3::Y * (RADIUS + 0.001),
        -Vec3::Y * 2.0,
        PhysicsMaterial::FRICTIONLESS,
        Vec3::ZERO,
    );
    let desc = world.articulation_colliders.get(old_id).unwrap().clone();
    assert!(world.articulation_colliders.remove(old_id));
    assert!(!world.articulation_colliders.remove(old_id));
    let mut solver = XpbdSolver::new();
    solver.step(&mut world, DT, 1);
    close_vec(link_linear_velocity(&world, owner), -Vec3::Y * 2.0, 1.0e-5);
    close(
        world
            .articulations
            .get(owner)
            .unwrap()
            .base_pose()
            .translation
            .y,
        RADIUS + 0.001 - 2.0 * DT,
        1.0e-5,
    );

    let replacement = world.attach_articulation_collider(desc).unwrap();
    assert_ne!(old_id, replacement);
    assert!(!world.articulation_colliders.remove(old_id));
    assert!(world.articulation_colliders.get(replacement).is_some());
    solver.step(&mut world, DT, 1);
    close_vec(link_linear_velocity(&world, owner), Vec3::ZERO, 0.03);
    assert!(center(&world, replacement).y >= RADIUS - CONTACT_SLOP);
}

#[test]
fn removing_owner_immediately_removes_all_its_attachments_only() {
    let mut world = world(Vec3::ZERO);
    let (owner, link_id) = floating_sphere(
        &mut world,
        Vec3::ZERO,
        Vec3::ZERO,
        PhysicsMaterial::DEFAULT,
        Vec3::ZERO,
    );
    let shape = world.articulation_colliders.get(link_id).unwrap().shape;
    let base_id = world
        .attach_articulation_collider(ArticulationColliderDesc::new(owner, None, shape))
        .unwrap();
    let (_, other_id) = floating_sphere(
        &mut world,
        Vec3::X * 4.0,
        Vec3::ZERO,
        PhysicsMaterial::DEFAULT,
        Vec3::ZERO,
    );
    assert!(world.remove_articulation(owner));
    assert!(!world.remove_articulation(owner));
    assert!(world.articulation_colliders.get(link_id).is_none());
    assert!(world.articulation_colliders.get(base_id).is_none());
    assert!(world.articulation_colliders.get(other_id).is_some());
    assert_eq!(world.articulation_colliders.len(), 1);
    assert_eq!(
        world.attach_articulation_collider(ArticulationColliderDesc::new(owner, Some(0), shape)),
        Err(ArticulationColliderError::InvalidArticulation)
    );
    assert_eq!(world.articulation_colliders.len(), 1);
}

#[test]
fn stale_owner_is_pruned_at_step_and_does_not_bind_reused_storage_slot() {
    let mut world = world(Vec3::ZERO);
    ground(&mut world, PhysicsMaterial::FRICTIONLESS, false);
    let (old_owner, old_id) = floating_sphere(
        &mut world,
        Vec3::Y * (RADIUS + 0.001),
        -Vec3::Y * 2.0,
        PhysicsMaterial::FRICTIONLESS,
        Vec3::ZERO,
    );
    let stale_desc = world.articulation_colliders.get(old_id).unwrap().clone();
    // Deliberately bypass the world's eager owner-removal cleanup.
    assert!(world.articulations.remove(old_owner));
    let new_owner = world.spawn_articulation(floating_tree(
        Vec3::Y * (RADIUS + 0.001),
        -Vec3::Y * 2.0,
        Vec3::ZERO,
    ));
    assert_ne!(old_owner, new_owner);
    assert!(world.articulation_colliders.get(old_id).is_some());
    assert_eq!(
        world.attach_articulation_collider(stale_desc),
        Err(ArticulationColliderError::InvalidArticulation)
    );
    assert_eq!(world.articulation_colliders.len(), 1);
    XpbdSolver::new().step(&mut world, DT, 1);
    assert!(world.articulation_colliders.is_empty());
    close_vec(
        link_linear_velocity(&world, new_owner),
        -Vec3::Y * 2.0,
        1.0e-5,
    );
    close(
        world
            .articulations
            .get(new_owner)
            .unwrap()
            .base_pose()
            .translation
            .y,
        RADIUS + 0.001 - 2.0 * DT,
        1.0e-5,
    );
}

#[test]
fn stale_shape_is_pruned_without_discarding_valid_attachment_or_free_motion() {
    let mut world = world(Vec3::ZERO);
    let (valid_owner, valid_id) = floating_sphere(
        &mut world,
        Vec3::X * 0.49,
        Vec3::ZERO,
        PhysicsMaterial::FRICTIONLESS,
        Vec3::ZERO,
    );
    let (stale_owner, stale_id) = floating_sphere(
        &mut world,
        -Vec3::X * 0.49,
        Vec3::X,
        PhysicsMaterial::FRICTIONLESS,
        Vec3::ZERO,
    );
    let stale_desc = world.articulation_colliders.get(stale_id).unwrap().clone();
    // ShapeRegistry is append-only, but the world exposes its replacement.
    // Preserve handle zero and invalidate handle one rather than inventing a
    // per-shape removal API or aliasing the stale handle to new geometry.
    world.shapes = ShapeRegistry::new();
    let preserved = world
        .shapes
        .insert(ColliderShape::Sphere { radius: RADIUS });
    assert_eq!(
        world.articulation_colliders.get(valid_id).unwrap().shape,
        preserved
    );
    assert_eq!(
        world.attach_articulation_collider(stale_desc),
        Err(ArticulationColliderError::InvalidShape)
    );
    assert_eq!(world.articulation_colliders.len(), 2);
    let mut free = world.clone();
    assert!(free.articulation_colliders.remove(stale_id));
    let mut solver = XpbdSolver::new();
    let mut free_solver = solver.clone();
    solver.step(&mut world, DT, 1);
    free_solver.step(&mut free, DT, 1);
    assert_eq!(world.articulation_colliders.len(), 1);
    assert!(world.articulation_colliders.get(stale_id).is_none());
    assert!(world.articulation_colliders.get(valid_id).is_some());
    assert_same_tree(&world, &free, stale_owner);
    assert_same_tree(&world, &free, valid_owner);
    close_vec(link_linear_velocity(&world, stale_owner), Vec3::X, 1.0e-5);
    close_vec(
        link_linear_velocity(&world, valid_owner),
        Vec3::ZERO,
        1.0e-5,
    );
}

#[test]
fn degraded_owner_retains_geometry_for_handoff_without_participating_in_contacts() {
    // Exercise both the all-stopped fast path and a mixed active/stopped pass.
    for mixed in [false, true] {
        let mut world = world(Vec3::ZERO);
        let (owner, attachment) = floating_sphere(
            &mut world,
            Vec3::ZERO,
            Vec3::X,
            PhysicsMaterial::DEFAULT,
            Vec3::ZERO,
        );
        let active = mixed.then(|| {
            floating_sphere(
                &mut world,
                Vec3::X * 0.49,
                -Vec3::X,
                PhysicsMaterial::DEFAULT,
                Vec3::ZERO,
            )
            .0
        });
        let desc = world
            .articulation_colliders
            .get(attachment)
            .unwrap()
            .clone();
        assert!(world
            .articulations
            .get_mut(owner)
            .unwrap()
            .break_joint(0)
            .is_some());
        let stopped = world.clone();
        assert_eq!(
            world.attach_articulation_collider(desc.clone()),
            Err(ArticulationColliderError::InvalidArticulation)
        );
        let mut reference = world.clone();
        assert!(reference.articulation_colliders.remove(attachment));
        let mut solver = XpbdSolver::new();
        let mut reference_solver = solver.clone();
        for _ in 0..4 {
            solver.step(&mut world, DT, 2);
            reference_solver.step(&mut reference, DT, 2);
            assert_eq!(world.articulation_colliders.get(attachment), Some(&desc));
            assert_same_tree(&world, &stopped, owner);
            if let Some(active) = active {
                assert_same_tree(&world, &reference, active);
            }
        }
        assert!(world.remove_articulation(owner));
        assert!(world.articulation_colliders.get(attachment).is_none());
        assert_eq!(world.articulation_colliders.len(), usize::from(mixed));
    }
}

#[test]
fn cloning_contacting_world_preserves_exact_replay_and_attachment_identities() {
    let mut world = world(-Vec3::Y * 9.81);
    ground(&mut world, PhysicsMaterial::DEFAULT, false);
    let (owner, attachment) = floating_sphere(
        &mut world,
        Vec3::Y * 1.2,
        Vec3::ZERO,
        PhysicsMaterial::DEFAULT,
        Vec3::ZERO,
    );
    let rigid = rigid_sphere(
        &mut world,
        Vec3::new(0.75, RADIUS, 0.0),
        Vec3::ZERO,
        3.0,
        PhysicsMaterial::DEFAULT,
        false,
    );
    let mut solver = XpbdSolver::new();
    for _ in 0..30 {
        solver.step(&mut world, DT, 2);
    }
    let mut replay = world.clone();
    let mut replay_solver = solver.clone();
    for _ in 0..180 {
        solver.step(&mut world, DT, 2);
        replay_solver.step(&mut replay, DT, 2);
        assert_same_tree(&world, &replay, owner);
        assert_eq!(world.bodies.position(rigid), replay.bodies.position(rigid));
        assert_eq!(
            world.bodies.orientation(rigid),
            replay.bodies.orientation(rigid)
        );
        assert_eq!(
            world.bodies.linear_velocity(rigid),
            replay.bodies.linear_velocity(rigid)
        );
        assert_eq!(
            world.bodies.angular_velocity(rigid),
            replay.bodies.angular_velocity(rigid)
        );
        assert_eq!(
            world.articulation_colliders.iter().collect::<Vec<_>>(),
            replay.articulation_colliders.iter().collect::<Vec<_>>()
        );
        assert!(center(&world, attachment).is_finite());
    }
    assert!(center(&world, attachment).y >= RADIUS - CONTACT_SLOP);
    assert!(
        world.bodies.position(rigid).unwrap().x > 0.7501,
        "replay must include real articulation/rigid contact, not identical free flight"
    );
    // Registry allocation state, not just existing entries, must be cloned.
    let desc = world
        .articulation_colliders
        .get(attachment)
        .unwrap()
        .clone();
    assert_eq!(
        world.attach_articulation_collider(desc.clone()),
        replay.attach_articulation_collider(desc)
    );
}

fn sibling_sliders(
    world: &mut PhysicsWorld,
    centers: &[f32],
    flags: &[bool],
    velocities: &[f32],
) -> ArticulationHandle {
    assert_eq!(centers.len(), flags.len());
    assert_eq!(centers.len(), velocities.len());
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, sphere_inertia(1.0));
    for &x in centers {
        builder
            .add_link(
                None,
                SpatialTransform::new(Quat::IDENTITY, Vec3::X * x),
                sphere_inertia(1.0),
                ArticulationJoint::prismatic(Vec3::X),
            )
            .unwrap();
    }
    let mut tree = builder.build().unwrap();
    for (dof, &velocity) in velocities.iter().enumerate() {
        assert!(tree.set_velocity(dof, velocity));
    }
    let owner = world.spawn_articulation(tree);
    let shape = world
        .shapes
        .insert(ColliderShape::Sphere { radius: RADIUS });
    for (link, &opt_in) in flags.iter().enumerate() {
        let mut desc = ArticulationColliderDesc::new(owner, Some(link), shape);
        assert!(!desc.self_collision);
        desc.self_collision = opt_in;
        desc.material = PhysicsMaterial::FRICTIONLESS;
        world.attach_articulation_collider(desc).unwrap();
    }
    owner
}

#[test]
fn same_tree_distinct_links_collide_only_when_both_attachments_opt_in() {
    for flags in [[false, false], [true, false], [false, true], [true, true]] {
        let mut world = world(Vec3::ZERO);
        let owner = sibling_sliders(&mut world, &[-0.4, 0.4], &flags, &[1.0, -1.0]);
        XpbdSolver::new().step(&mut world, DT, 1);
        let tree = world.articulations.get(owner).unwrap();
        let poses = tree.forward_kinematics();
        let separation = poses[1].translation.x - poses[0].translation.x;
        close(tree.velocities().iter().sum(), 0.0, 1.0e-5);
        if flags == [true, true] {
            assert!(
                separation >= 2.0 * RADIUS - CONTACT_SLOP,
                "opt-in separation {separation}"
            );
            close(tree.velocities()[0], 0.0, 0.02);
            close(tree.velocities()[1], 0.0, 0.02);
        } else {
            close(tree.positions()[0], DT, 1.0e-6);
            close(tree.positions()[1], -DT, 1.0e-6);
            close(tree.velocities()[0], 1.0, 1.0e-6);
            close(tree.velocities()[1], -1.0, 1.0e-6);
            close(separation, 0.8 - 2.0 * DT, 1.0e-6);
        }
    }
}

#[test]
fn default_split_iterations_resolve_coupled_link_overlaps_without_adding_momentum() {
    let mut world = world(Vec3::ZERO);
    let owner = sibling_sliders(&mut world, &[0.0, 0.8, 1.6], &[true; 3], &[0.0; 3]);
    XpbdSolver::new().step(&mut world, DT, 1);
    let tree = world.articulations.get(owner).unwrap();
    let poses = tree.forward_kinematics();
    for pair in poses.windows(2) {
        let separation = pair[1].translation.x - pair[0].translation.x;
        assert!(
            separation >= 2.0 * RADIUS - CONTACT_SLOP,
            "coupled contact separation {separation}"
        );
    }
    close(tree.positions().iter().sum(), 0.0, 1.0e-4);
    for &velocity in tree.velocities() {
        close(velocity, 0.0, 1.0e-5);
    }
}

#[test]
fn same_body_overlapping_attachments_never_self_collide_even_when_opted_in() {
    let mut world = world(Vec3::ZERO);
    let owner = sibling_sliders(&mut world, &[0.0], &[true], &[1.0]);
    let existing = world.articulation_colliders.iter().next().unwrap().0;
    let mut second = world.articulation_colliders.get(existing).unwrap().clone();
    second.local_pose = Isometry::from_translation(Vec3::X * 0.25);
    world.attach_articulation_collider(second).unwrap();
    XpbdSolver::new().step(&mut world, DT, 1);
    let tree = world.articulations.get(owner).unwrap();
    close(tree.positions()[0], DT, 1.0e-6);
    close(tree.velocities()[0], 1.0, 1.0e-6);
    assert_eq!(world.articulation_colliders.len(), 2);
}

fn downward_prismatic_press(
    world: &mut PhysicsWorld,
) -> (ArticulationHandle, ArticulationColliderId) {
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, sphere_inertia(1.0))
        .with_base_pose(Isometry::from_translation(Vec3::Y * 2.0))
        .with_gravity(world.config.gravity);
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            sphere_inertia(2.0),
            ArticulationJoint::prismatic(Vec3::Y),
        )
        .unwrap();
    let mut tree = builder.build().unwrap();
    assert!(tree.set_velocity(0, -1.0));
    let owner = world.spawn_articulation(tree);
    let shape = world
        .shapes
        .insert(ColliderShape::Sphere { radius: RADIUS });
    let mut desc = ArticulationColliderDesc::new(owner, Some(0), shape);
    desc.material = PhysicsMaterial::FRICTIONLESS;
    let attachment = world.attach_articulation_collider(desc).unwrap();
    (owner, attachment)
}

#[test]
fn tree_pressing_dynamic_rigid_preserves_static_ground_support() {
    let mut world = world(-Vec3::Y * 9.81);
    let floor = ground(&mut world, PhysicsMaterial::FRICTIONLESS, false);
    let supported = rigid_sphere(
        &mut world,
        Vec3::Y * RADIUS,
        Vec3::ZERO,
        3.0,
        PhysicsMaterial::FRICTIONLESS,
        false,
    );
    let (owner, attachment) = downward_prismatic_press(&mut world);
    let mut solver = XpbdSolver::new();
    let mut approached_support = false;
    for frame in 0..180 {
        solver.step(&mut world, 1.0 / 60.0, 4);
        let rigid_position = world.bodies.position(supported).unwrap();
        let tree_position = center(&world, attachment);
        approached_support |= tree_position.y < 1.8 && link_linear_velocity(&world, owner).y < -0.5;
        // Resolving tree/rigid contact after the ordinary rigid pass must not
        // push the supported rigid into the floor, even under sustained load.
        assert!(
            rigid_position.y >= RADIUS - CONTACT_SLOP,
            "frame {frame}: supported rigid penetrated ground: {rigid_position:?}"
        );
        assert!(
            tree_position.y - rigid_position.y >= 2.0 * RADIUS - CONTACT_SLOP,
            "frame {frame}: tree/rigid overlap: {tree_position:?}, {rigid_position:?}"
        );
        close_vec(
            rigid_position * Vec3::new(1.0, 0.0, 1.0),
            Vec3::ZERO,
            1.0e-4,
        );
        if frame >= 150 {
            close(rigid_position.y, RADIUS, CONTACT_SLOP);
            close(tree_position.y, 3.0 * RADIUS, 2.0 * CONTACT_SLOP);
            close_vec(
                world.bodies.linear_velocity(supported).unwrap(),
                Vec3::ZERO,
                0.05,
            );
            close_vec(link_linear_velocity(&world, owner), Vec3::ZERO, 0.05);
        }
    }
    assert!(
        approached_support,
        "the press must actually fall onto the supported rigid"
    );
    close(
        world.articulations.get(owner).unwrap().positions()[0],
        -0.5,
        2.0 * CONTACT_SLOP,
    );
    assert_eq!(world.bodies.position(floor), Some(Vec3::ZERO));
}

#[test]
fn mixed_contacts_do_not_wake_unrelated_sleeping_body_on_same_static_ground() {
    let mut world = world(-Vec3::Y * 9.81);
    world.config.sleep.enabled = true;
    let floor = ground(&mut world, PhysicsMaterial::FRICTIONLESS, false);
    let sleeper = rigid_sphere(
        &mut world,
        Vec3::new(20.0, RADIUS, 0.0),
        Vec3::ZERO,
        3.0,
        PhysicsMaterial::FRICTIONLESS,
        false,
    );
    let mut solver = XpbdSolver::new();
    // Establish a real, solver-produced sleeping contact with this very same
    // static floor, rather than manually setting an otherwise isolated flag.
    for _ in 0..120 {
        solver.step(&mut world, 1.0 / 60.0, 4);
    }
    assert_eq!(world.bodies.is_sleeping(sleeper), Some(true));
    let sleeping_position = world.bodies.position(sleeper);
    let sleeping_orientation = world.bodies.orientation(sleeper);
    let sleeping_timer = world.bodies.sleep_timer(sleeper);
    close(sleeping_position.unwrap().y, RADIUS, CONTACT_SLOP);

    let supported = rigid_sphere(
        &mut world,
        Vec3::Y * RADIUS,
        Vec3::ZERO,
        3.0,
        PhysicsMaterial::FRICTIONLESS,
        false,
    );
    let (owner, attachment) = downward_prismatic_press(&mut world);
    for frame in 0..240 {
        // Observe after every physical substep: waking then sleeping again
        // later in a multi-substep frame must not hide an island leak.
        solver.step(&mut world, DT, 1);
        assert_eq!(
            world.bodies.is_sleeping(sleeper),
            Some(true),
            "frame {frame}"
        );
        assert_eq!(
            world.bodies.position(sleeper),
            sleeping_position,
            "frame {frame}"
        );
        assert_eq!(
            world.bodies.orientation(sleeper),
            sleeping_orientation,
            "frame {frame}"
        );
        assert_eq!(
            world.bodies.sleep_timer(sleeper),
            sleeping_timer,
            "frame {frame}"
        );
        assert_eq!(world.bodies.linear_velocity(sleeper), Some(Vec3::ZERO));
        assert_eq!(world.bodies.angular_velocity(sleeper), Some(Vec3::ZERO));
    }
    // The nearby branch must really have reached mixed contact; otherwise a
    // no-op mixed solver could trivially leave the remote body asleep.
    close(
        center(&world, attachment).y,
        3.0 * RADIUS,
        2.0 * CONTACT_SLOP,
    );
    close(
        world.articulations.get(owner).unwrap().positions()[0],
        -0.5,
        2.0 * CONTACT_SLOP,
    );
    close(
        world.bodies.position(supported).unwrap().y,
        RADIUS,
        CONTACT_SLOP,
    );
    close_vec(link_linear_velocity(&world, owner), Vec3::ZERO, 0.05);
    assert_eq!(world.bodies.position(floor), Some(Vec3::ZERO));
}

fn nested_compound_slider(world: &mut PhysicsWorld) -> ArticulationHandle {
    let quarter_turn = Quat::from_rotation_z(core::f32::consts::FRAC_PI_2);
    let leaf = world.shapes.insert(ColliderShape::Cuboid {
        half_extents: Vec3::new(0.4, 0.1, 0.2),
    });
    let inner = world
        .shapes
        .insert_compound(&[
            CompoundChild {
                collider: leaf,
                local_pose: Isometry::from_translation(Vec3::Y * 1.25),
            },
            CompoundChild {
                collider: leaf,
                local_pose: Isometry::from_translation(-Vec3::Y * 1.25),
            },
        ])
        .unwrap();
    let middle = world
        .shapes
        .insert_compound(&[CompoundChild {
            collider: inner,
            local_pose: Isometry::new(Vec3::new(0.3, 0.4, 0.0), quarter_turn),
        }])
        .unwrap();
    let outer = world
        .shapes
        .insert_compound(&[CompoundChild {
            collider: middle,
            local_pose: Isometry::new(Vec3::new(-0.2, 0.1, 0.0), quarter_turn.inverse()),
        }])
        .unwrap();
    // Base, attachment, outer and middle rotations cancel. Their translations
    // compose to a world-space compound centre (-1, 0, 0) at q=0. Thus the
    // real leaves are boxes centred at (-1+q, +/-1.25, 0), with half-extents
    // (0.4, 0.1, 0.2). The wide empty gap is NOT solid collision geometry.
    let mut builder =
        ArticulationBuilder::new(ArticulationBase::Fixed, sphere_inertia(1.0)).with_base_pose(
            Isometry::new(Vec3::new(-0.9, 0.8, 0.0), quarter_turn.inverse()),
        );
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            sphere_inertia(2.0),
            ArticulationJoint::prismatic(Vec3::Y),
        )
        .unwrap();
    let mut tree = builder.build().unwrap();
    assert!(tree.set_velocity(0, 1.0));
    let owner = world.spawn_articulation(tree);
    let mut desc = ArticulationColliderDesc::new(owner, Some(0), outer);
    desc.local_pose = Isometry::new(Vec3::new(0.6, -0.3, 0.0), quarter_turn);
    desc.material = PhysicsMaterial::FRICTIONLESS;
    world.attach_articulation_collider(desc).unwrap();
    owner
}

fn static_probe(world: &mut PhysicsWorld, position: Vec3) -> BodyHandle {
    let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.25 });
    world.spawn(
        BodyDesc::static_at(position)
            .with_collider(shape)
            .with_material(PhysicsMaterial::FRICTIONLESS),
    )
}

#[test]
fn nested_compound_gap_with_rotated_local_offsets_is_not_a_bounding_sphere() {
    let mut world = world(Vec3::ZERO);
    let owner = nested_compound_slider(&mut world);
    let probe = static_probe(&mut world, Vec3::ZERO);
    let mut free = world.clone();
    assert!(free.remove_body(probe));
    let mut solver = XpbdSolver::new();
    let mut free_solver = solver.clone();
    for frame in 0..240 {
        solver.step(&mut world, DT, 1);
        free_solver.step(&mut free, DT, 1);
        assert_same_tree(&world, &free, owner);
        let tree = world.articulations.get(owner).unwrap();
        close(tree.positions()[0], (frame + 1) as f32 * DT, 2.0e-4);
        close(tree.velocities()[0], 1.0, 1.0e-5);
        close_vec(link_linear_velocity(&world, owner), Vec3::X, 1.0e-5);
    }
    // q: 0 -> 2 carries the compound centre completely past the probe. A
    // bounding sphere, union AABB or convexified union would obstruct it.
    close(
        world.articulations.get(owner).unwrap().positions()[0],
        2.0,
        2.0e-4,
    );
    assert_eq!(world.bodies.position(probe), Some(Vec3::ZERO));
}

#[test]
fn nested_compound_real_leaf_contact_respects_all_local_transforms() {
    let mut world = world(Vec3::ZERO);
    let owner = nested_compound_slider(&mut world);
    let probe = static_probe(&mut world, Vec3::Y * 1.25);
    let mut solver = XpbdSolver::new();
    for frame in 0..120 {
        solver.step(&mut world, DT, 1);
        let tree = world.articulations.get(owner).unwrap();
        // Right face of the upper box is x=-1+q+0.4; the probe's left
        // surface is x=-0.25. They first touch at q=0.35, not at the
        // compound/attachment origin or its enclosing sphere surface.
        let leaf_right_face = -1.0 + tree.positions()[0] + 0.4;
        assert!(
            leaf_right_face <= -0.25 + CONTACT_SLOP,
            "frame {frame}: leaf right face {leaf_right_face}"
        );
        close_vec(
            tree.base_pose().translation,
            Vec3::new(-0.9, 0.8, 0.0),
            1.0e-6,
        );
        if frame >= 90 {
            close(tree.positions()[0], 0.35, CONTACT_SLOP);
            close(tree.velocities()[0], 0.0, 0.02);
            close_vec(link_linear_velocity(&world, owner), Vec3::ZERO, 0.02);
        }
    }
    assert_eq!(world.bodies.position(probe), Some(Vec3::Y * 1.25));
}
