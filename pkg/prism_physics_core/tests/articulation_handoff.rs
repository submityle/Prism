use glam::{Mat3, Quat, Vec3};
use prism_physics_core::collider::ColliderShape;
use prism_physics_core::joint::{JointAnchor, JointDesc, JointKind};
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::articulation::handoff::{
    ArticulationHandoffError, UnsupportedJointPolicy,
};
use prism_physics_core::reduced::{
    ArticulationBase, ArticulationBuilder, ArticulationColliderDesc, ArticulationHandle,
    ArticulationJoint, JointLimits, JointMotor, SpatialInertia, SpatialMotion, SpatialTransform,
};
use prism_physics_core::solver::{Solver, XpbdSolver};
use prism_physics_core::state::body::{BodyDesc, BodyKind};
use prism_physics_core::world::PhysicsWorld;

fn point_inertia(mass: f32) -> SpatialInertia {
    SpatialInertia::diagonal(mass, Vec3::ZERO, Vec3::splat(mass))
}

#[test]
fn handoff_keeps_multiturn_hinge_coordinates_and_outside_principal_range_limits() {
    use prism_physics_core::snapshot::{world_hash::hash_world, FullStateSnapshot};

    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        for angle in [4.0, -4.0, 20.0, -20.0] {
            let axes = Mat3::from_quat(Quat::from_rotation_y(0.7));
            let tensor = axes * Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0)) * axes.transpose();
            let inertia = SpatialInertia::new(
                2.0,
                Vec3::new(0.2, -0.1, 0.3),
                tensor.transpose().to_cols_array_2d(),
            );
            let mut builder = ArticulationBuilder::new(base, inertia)
                .with_gravity(Vec3::ZERO)
                .with_base_pose(Isometry::new(Vec3::ONE, Quat::from_rotation_x(0.4)));
            builder
                .add_link(
                    None,
                    SpatialTransform::new(Quat::from_rotation_z(0.3), Vec3::Y),
                    inertia,
                    ArticulationJoint::revolute(Vec3::Y)
                        .with_limits(JointLimits::scalar(angle - 0.2, angle + 0.2)),
                )
                .unwrap();
            builder
                .add_link(
                    Some(0),
                    SpatialTransform::IDENTITY,
                    point_inertia(1.0),
                    ArticulationJoint::fixed(),
                )
                .unwrap();
            let mut tree = builder.build().unwrap();
            assert!(tree.set_position(0, angle));
            tree.break_joint(1);
            let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
            let handle = world.spawn_articulation(tree);
            let report = world.handoff_articulation(handle).unwrap();
            let hinge = report.joints[0].joint;
            assert_eq!(
                world.joints.get(hinge).unwrap().revolute_state().angle(),
                Some(f64::from(angle))
            );
            let mut replay = FullStateSnapshot::capture(0, &world).fork_world();
            let mut solver = XpbdSolver::default();
            let mut replay_solver = XpbdSolver::default();
            for _ in 0..12 {
                solver.step(&mut world, 1.0 / 120.0, 1);
                replay_solver.step(&mut replay, 1.0 / 120.0, 1);
                let observed = world
                    .joints
                    .get(hinge)
                    .unwrap()
                    .revolute_state()
                    .angle()
                    .unwrap();
                assert!(
                    (observed - f64::from(angle)).abs() < 2.0e-4,
                    "{base:?}: {angle} became {observed}"
                );
                assert_eq!(hash_world(&world).unwrap(), hash_world(&replay).unwrap());
                for mapping in core::iter::once(&report.base).chain(&report.links) {
                    assert!(
                        world
                            .bodies
                            .angular_velocity(mapping.body)
                            .unwrap()
                            .length()
                            < 2.0e-3
                    );
                }
            }
        }
    }
}

#[test]
fn handoff_preserves_original_tree_collision_policy_across_disconnected_branches() {
    use prism_physics_core::pipeline::detect_contacts;
    use prism_physics_core::snapshot::{world_hash::hash_world, FullStateSnapshot};

    for flags in [[false; 4], [true; 4], [true, false, true, true]] {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
        let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, point_inertia(1.0))
            .with_gravity(Vec3::ZERO);
        for link in 0usize..3 {
            builder
                .add_link(
                    link.checked_sub(1),
                    SpatialTransform::IDENTITY,
                    point_inertia(1.0),
                    ArticulationJoint::fixed(),
                )
                .unwrap();
        }
        let source = world.spawn_articulation(builder.build().unwrap());
        for (index, self_collision) in flags.into_iter().enumerate() {
            let mut desc = ArticulationColliderDesc::new(source, index.checked_sub(1), shape);
            desc.self_collision = self_collision;
            world.attach_articulation_collider(desc).unwrap();
        }
        world.articulations.get_mut(source).unwrap().break_joint(1);
        let report = world.handoff_articulation(source).unwrap();
        let members: Vec<_> = core::iter::once(report.base.body)
            .chain(report.links.iter().map(|link| link.body))
            .collect();
        let mut external = BodyDesc::dynamic_at(Vec3::ZERO);
        external.collider = Some(shape);
        let outside = world.spawn(external);
        let contacts = detect_contacts(&world);
        let has_pair = |a, b| {
            contacts.iter().any(|contact| {
                (contact.body_a == a && contact.body_b == b)
                    || (contact.body_a == b && contact.body_b == a)
            })
        };
        for (i, &a) in members.iter().enumerate() {
            assert!(
                has_pair(a, outside),
                "external contact suppressed, flags {flags:?}"
            );
            for (j, &b) in members.iter().enumerate().skip(i + 1) {
                assert_eq!(world.collision_pair_allowed(a, b), flags[i] && flags[j]);
                assert_eq!(
                    has_pair(a, b),
                    flags[i] && flags[j],
                    "members {i}/{j}, flags {flags:?}"
                );
            }
        }
        assert_eq!(world.collision_exclusions().len(), members.len());
        let snapshot = FullStateSnapshot::capture(0, &world);
        let mut replay = snapshot.fork_world();
        assert_eq!(hash_world(&world), hash_world(&replay));
        let mut solver = XpbdSolver::default();
        let mut replay_solver = XpbdSolver::default();
        for _ in 0..4 {
            solver.step(&mut world, 1.0 / 120.0, 1);
            replay_solver.step(&mut replay, 1.0 / 120.0, 1);
            assert_eq!(hash_world(&world), hash_world(&replay));
        }
    }
}

#[test]
fn handoff_groups_members_without_colliders_and_does_not_group_separate_trees() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let make_tree = || {
        let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, point_inertia(1.0))
            .with_gravity(Vec3::ZERO);
        builder
            .add_link(
                None,
                SpatialTransform::IDENTITY,
                point_inertia(1.0),
                ArticulationJoint::fixed(),
            )
            .unwrap();
        let mut tree = builder.build().unwrap();
        tree.break_joint(0);
        tree
    };
    let a = world.spawn_articulation(make_tree());
    let b = world.spawn_articulation(make_tree());
    let a = world.handoff_articulation(a).unwrap();
    let b = world.handoff_articulation(b).unwrap();
    assert!(!world.collision_pair_allowed(a.base.body, a.links[0].body));
    assert!(!world.collision_pair_allowed(b.base.body, b.links[0].body));
    assert!(world.collision_pair_allowed(a.base.body, b.links[0].body));
    let removed = a.links[0].body;
    assert!(world.remove_body(removed));
    let replacement = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    assert_ne!(removed, replacement);
    assert!(world.collision_pair_allowed(a.base.body, replacement));
    assert_eq!(world.collision_exclusions().len(), 3);
}

fn unsupported_joint_policies() -> [(&'static str, ArticulationJoint); 6] {
    let motor = JointMotor::new(Vec3::ZERO, Vec3::ZERO, 2.0, 0.5, Vec3::splat(10.0));
    let limits = JointLimits::vector(Vec3::splat(-0.5), Vec3::splat(0.5));
    [
        (
            "motor",
            ArticulationJoint::revolute(Vec3::Y).with_motor(motor),
        ),
        (
            "damping",
            ArticulationJoint::revolute(Vec3::Y).with_damping(0.25),
        ),
        (
            "armature",
            ArticulationJoint::revolute(Vec3::Y).with_armature(0.5),
        ),
        (
            "spherical limits",
            ArticulationJoint::spherical().with_limits(limits),
        ),
        (
            "fixed limits",
            ArticulationJoint::fixed().with_limits(limits),
        ),
        (
            "combined",
            ArticulationJoint::spherical()
                .with_motor(motor)
                .with_damping(0.25)
                .with_armature(0.5)
                .with_limits(limits),
        ),
    ]
}

fn world_with_broken_chain(
    joints: [ArticulationJoint; 3],
    broken_link: usize,
) -> (PhysicsWorld, ArticulationHandle) {
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default())
        .with_gravity(Vec3::new(0.0, -9.81, 0.0));
    for (link, joint) in joints.into_iter().enumerate() {
        builder
            .add_link(
                link.checked_sub(1),
                SpatialTransform::new(Quat::IDENTITY, Vec3::Y),
                point_inertia(1.0),
                joint,
            )
            .unwrap();
    }
    let mut world = PhysicsWorld::default();
    let articulation = world.spawn_articulation(builder.build().unwrap());
    assert!(world
        .articulations
        .get_mut(articulation)
        .unwrap()
        .break_joint(broken_link)
        .is_some());
    (world, articulation)
}

fn close(actual: f32, expected: f32) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= 1.0e-5,
        "actual {actual}, expected {expected}"
    );
}

fn close_vec(actual: Vec3, expected: Vec3) {
    assert!(
        actual.is_finite() && (actual - expected).length() <= 1.0e-5,
        "actual {actual:?}, expected {expected:?}"
    );
}

#[test]
fn handoff_ignores_unsupported_policies_only_on_the_broken_joint() {
    for (name, policy) in unsupported_joint_policies() {
        for broken_link in 0..3 {
            let mut joints = [ArticulationJoint::revolute(Vec3::Y); 3];
            joints[broken_link] = policy;
            let (mut world, articulation) = world_with_broken_chain(joints, broken_link);

            let report = world
                .handoff_articulation(articulation)
                .unwrap_or_else(|error| panic!("{name}, broken link {broken_link}: {error:?}"));

            assert_eq!(report.broken_link, broken_link);
            assert_eq!(report.links.len(), 3);
            assert_eq!(world.bodies.len(), 4);
            assert_eq!(world.joints.len(), 2);
            assert_eq!(
                report
                    .joints
                    .iter()
                    .map(|mapping| mapping.link)
                    .collect::<Vec<_>>(),
                (0..3)
                    .filter(|&link| link != broken_link)
                    .collect::<Vec<_>>()
            );
            for mapping in &report.joints {
                let joint = world.joints.get(mapping.joint).unwrap();
                let parent = mapping
                    .link
                    .checked_sub(1)
                    .map_or(report.base.body, |link| report.links[link].body);
                assert_eq!(joint.anchor_a.body, parent);
                assert_eq!(joint.anchor_b.body, report.links[mapping.link].body);
            }
            assert!(!world.articulations.contains(articulation));
        }
    }
}

#[test]
fn handoff_rejects_unsupported_policies_on_every_surviving_joint() {
    let broken_policy = unsupported_joint_policies()[5].1;
    for (name, policy) in unsupported_joint_policies() {
        for broken_link in 0..3 {
            // Descendant joints remain alive even in the detached subtree.
            for surviving_link in (0..3).filter(|&link| link != broken_link) {
                let mut joints = [ArticulationJoint::revolute(Vec3::Y); 3];
                joints[broken_link] = broken_policy;
                joints[surviving_link] = policy;
                let (mut world, articulation) = world_with_broken_chain(joints, broken_link);
                let states_before = world
                    .articulations
                    .get(articulation)
                    .unwrap()
                    .maximal_body_states();
                let expected = if policy.motor.is_some() {
                    UnsupportedJointPolicy::Motor {
                        link: surviving_link,
                    }
                } else {
                    UnsupportedJointPolicy::InertOrExtraParameters {
                        link: surviving_link,
                    }
                };

                assert_eq!(
                    world.handoff_articulation(articulation),
                    Err(ArticulationHandoffError::UnsupportedJointPolicy(expected)),
                    "{name}, broken link {broken_link}, surviving link {surviving_link}"
                );
                assert!(world.bodies.is_empty());
                assert!(world.joints.is_empty());
                let source = world.articulations.get(articulation).unwrap();
                assert_eq!(source.broken_joint(), Some(broken_link));
                assert_eq!(source.maximal_body_states(), states_before);
                for (link, joint) in joints.iter().enumerate() {
                    assert_eq!(source.links()[link].joint, *joint);
                }
            }
        }
    }
}

#[test]
fn handoff_preserves_bodies_surviving_joints_and_local_pose() {
    let mut world = PhysicsWorld::default();
    let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.25 });
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default())
        .with_base_pose(Isometry::from_translation(Vec3::new(1.0, 2.0, 3.0)))
        .with_gravity(Vec3::new(0.0, -9.81, 0.0));
    builder
        .add_link(
            None,
            SpatialTransform::new(Quat::IDENTITY, Vec3::new(0.0, 1.0, 0.0)),
            point_inertia(2.0),
            ArticulationJoint::revolute(Vec3::Y),
        )
        .unwrap();
    builder
        .add_link(
            Some(0),
            SpatialTransform::new(Quat::IDENTITY, Vec3::new(0.0, 1.0, 0.0)),
            point_inertia(1.0),
            ArticulationJoint::spherical(),
        )
        .unwrap();
    let articulation = world.spawn_articulation(builder.build().unwrap());
    world
        .attach_articulation_collider(ArticulationColliderDesc {
            articulation,
            link: Some(0),
            shape,
            local_pose: Isometry::from_translation(Vec3::new(0.1, 0.0, 0.0)),
            material: prism_physics_core::PhysicsMaterial::new(0.7, 0.2),
            self_collision: false,
        })
        .unwrap();
    world
        .articulations
        .get_mut(articulation)
        .unwrap()
        .break_joint(1);
    let report = world.handoff_articulation(articulation).unwrap();
    assert_eq!(report.broken_link, 1);
    assert_eq!(report.links.len(), 2);
    assert_eq!(report.joints.len(), 1);
    assert_eq!(world.bodies.kind(report.base.body), Some(BodyKind::Static));
    assert_eq!(
        world.bodies.kind(report.links[0].body),
        Some(BodyKind::Dynamic)
    );
    let base_mass = world.bodies.mass_properties(report.base.body).unwrap();
    assert_eq!(base_mass.inv_mass, 0.0);
    let first_mass = world.bodies.mass_properties(report.links[0].body).unwrap();
    close(first_mass.inv_mass, 0.5);
    close_vec(first_mass.inv_inertia, Vec3::splat(0.5));
    let second_mass = world.bodies.mass_properties(report.links[1].body).unwrap();
    close(second_mass.inv_mass, 1.0);
    close_vec(second_mass.inv_inertia, Vec3::ONE);
    assert_eq!(world.bodies.len(), 3);
    assert_eq!(world.articulation_colliders.len(), 0);
    assert_eq!(report.colliders.len(), 1);
    assert_ne!(
        report.colliders[0].source_shape,
        report.colliders[0].body_shape
    );
    assert!(!world.articulations.contains(articulation));
}

#[test]
fn handoff_pins_gravity_mismatch_without_changing_world_gravity() {
    let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, point_inertia(2.0))
        .with_gravity(Vec3::ZERO);
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            point_inertia(1.0),
            ArticulationJoint::fixed(),
        )
        .unwrap();
    let mut world = PhysicsWorld::default();
    let articulation = world.spawn_articulation(builder.build().unwrap());
    world
        .articulations
        .get_mut(articulation)
        .unwrap()
        .break_joint(0);
    let gravity = world.config.gravity;
    let report = world.handoff_articulation(articulation).unwrap();
    for body in std::iter::once(report.base.body).chain(report.links.iter().map(|link| link.body)) {
        assert_eq!(world.bodies.gravity_override(body), Some(Some(Vec3::ZERO)));
    }
    assert_eq!(world.config.gravity, gravity);
    assert!(!world.articulations.contains(articulation));
}

#[test]
fn handoff_preserves_com_full_inertia_momentum_energy_geometry_and_joint_frames() {
    let axes = Mat3::from_quat(Quat::from_rotation_z(0.6) * Quat::from_rotation_x(-0.4));
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        for moments in [
            Vec3::new(1.0, 1.5, 2.0),
            Vec3::new(0.0, 2.0, 2.0),
            Vec3::ZERO,
        ] {
            for center in [Vec3::ZERO, Vec3::new(0.3, -0.2, 0.4)] {
                let tensor = axes * Mat3::from_diagonal(moments) * axes.transpose();
                let rows = tensor.transpose().to_cols_array_2d();
                let inertia = SpatialInertia::new(2.0, center, rows);
                let mut builder = ArticulationBuilder::new(base, inertia)
                    .with_base_pose(Isometry::new(
                        Vec3::new(1.0, -2.0, 3.0),
                        Quat::from_rotation_y(0.3),
                    ))
                    .with_gravity(Vec3::ZERO);
                for (index, joint) in [
                    ArticulationJoint::prismatic(Vec3::new(1.0, 2.0, 0.5).normalize()),
                    ArticulationJoint::revolute(Vec3::new(-0.5, 1.0, 2.0).normalize()),
                    ArticulationJoint::fixed(),
                ]
                .into_iter()
                .enumerate()
                {
                    builder
                        .add_link(
                            index.checked_sub(1),
                            SpatialTransform::new(
                                Quat::from_rotation_z(0.2 * (index + 1) as f32),
                                Vec3::new(0.5, 1.0, -0.25),
                            ),
                            inertia,
                            joint,
                        )
                        .unwrap();
                }
                let mut tree = builder.build().unwrap();
                assert!(tree.set_position(0, 0.37));
                assert!(tree.set_position(1, -0.41));
                assert!(tree.set_velocity(0, 0.7));
                assert!(tree.set_velocity(1, -0.3));
                if base == ArticulationBase::Floating {
                    assert!(tree.set_base_velocity(SpatialMotion::new(
                        Vec3::new(0.2, -0.3, 0.4),
                        Vec3::new(0.7, -0.6, 0.5),
                    )));
                }
                let states = tree.maximal_body_states();
                let models = tree.links().to_vec();
                let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
                let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.1 });
                let articulation = world.spawn_articulation(tree);
                let collider_pose =
                    Isometry::new(Vec3::new(-0.1, 0.25, 0.2), Quat::from_rotation_x(-0.3));
                for link in [None, Some(0), Some(1), Some(2)] {
                    world
                        .attach_articulation_collider(ArticulationColliderDesc {
                            articulation,
                            link,
                            shape,
                            local_pose: collider_pose,
                            material: prism_physics_core::PhysicsMaterial::DEFAULT,
                            self_collision: false,
                        })
                        .unwrap();
                }
                world
                    .articulations
                    .get_mut(articulation)
                    .unwrap()
                    .break_joint(2);
                let report = world.handoff_articulation(articulation).unwrap();
                let mut old_momentum = Vec3::ZERO;
                let mut new_momentum = Vec3::ZERO;
                let mut old_angular = Vec3::ZERO;
                let mut new_angular = Vec3::ZERO;
                let mut old_energy = 0.0;
                let mut new_energy = 0.0;
                for (state, mapping) in states
                    .iter()
                    .zip(std::iter::once(&report.base).chain(&report.links))
                {
                    let pose = Isometry::new(
                        world.bodies.position(mapping.body).unwrap(),
                        world.bodies.orientation(mapping.body).unwrap(),
                    );
                    let restored = pose.mul(&mapping.link_from_body.inverse());
                    close_vec(restored.translation, state.pose.translation);
                    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                        close_vec(restored.rotation * axis, state.pose.rotation * axis);
                    }
                    let mass = world.bodies.mass_properties(mapping.body).unwrap();
                    if state.fixed {
                        assert_eq!(mass.inv_mass, 0.0);
                        assert_eq!(mapping.link_from_body, Isometry::IDENTITY);
                        continue;
                    }
                    close_vec(pose.translation, state.center_of_mass);
                    let velocity = world.bodies.linear_velocity(mapping.body).unwrap();
                    let omega = world.bodies.angular_velocity(mapping.body).unwrap();
                    close_vec(velocity, state.linear_velocity);
                    close_vec(omega, state.angular_velocity);
                    let rotation = Mat3::from_quat(pose.rotation);
                    let inv = mass.inv_inertia;
                    let moment = |value: f32| if value == 0.0 { 0.0 } else { value.recip() };
                    let actual = rotation
                        * Mat3::from_diagonal(Vec3::new(
                            moment(inv.x),
                            moment(inv.y),
                            moment(inv.z),
                        ))
                        * rotation.transpose();
                    let authored_rotation = Mat3::from_quat(state.pose.rotation);
                    let expected = authored_rotation * tensor * authored_rotation.transpose();
                    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                        close_vec(actual * axis, expected * axis);
                    }
                    let old_p = state.inertia.mass * state.linear_velocity;
                    let new_p = velocity / mass.inv_mass;
                    old_momentum += old_p;
                    new_momentum += new_p;
                    old_angular +=
                        state.center_of_mass.cross(old_p) + expected * state.angular_velocity;
                    new_angular += pose.translation.cross(new_p) + actual * omega;
                    old_energy += 0.5
                        * (old_p.dot(state.linear_velocity)
                            + state
                                .angular_velocity
                                .dot(expected * state.angular_velocity));
                    new_energy += 0.5 * (new_p.dot(velocity) + omega.dot(actual * omega));
                }
                close_vec(new_momentum, old_momentum);
                close_vec(new_angular, old_angular);
                close(new_energy, old_energy);
                for collider in &report.colliders {
                    let state = &states[collider.link.map_or(0, |link| link + 1)];
                    let pose = Isometry::new(
                        world.bodies.position(collider.body).unwrap(),
                        world.bodies.orientation(collider.body).unwrap(),
                    );
                    let ColliderShape::Compound { compound, .. } =
                        *world.shapes.get(collider.body_shape).unwrap()
                    else {
                        panic!("transformed collider must retain its local pose")
                    };
                    let children = world.shapes.compound(compound).unwrap().children();
                    assert_eq!(children.len(), 1);
                    assert_eq!(children[0].collider, shape);
                    for point in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
                        close_vec(
                            pose.mul(&children[0].local_pose).transform_point(point),
                            state.pose.mul(&collider_pose).transform_point(point),
                        );
                    }
                }
                for mapping in &report.joints {
                    let joint = world.joints.get(mapping.joint).unwrap();
                    let parent = models[mapping.link].parent.map_or(0, |link| link + 1);
                    let a = Isometry::new(
                        world.bodies.position(joint.anchor_a.body).unwrap(),
                        world.bodies.orientation(joint.anchor_a.body).unwrap(),
                    )
                    .mul(&Isometry::new(
                        joint.anchor_a.local_point,
                        joint.anchor_a.local_frame,
                    ));
                    let b = Isometry::new(
                        world.bodies.position(joint.anchor_b.body).unwrap(),
                        world.bodies.orientation(joint.anchor_b.body).unwrap(),
                    )
                    .mul(&Isometry::new(
                        joint.anchor_b.local_point,
                        joint.anchor_b.local_frame,
                    ));
                    let expected_a = states[parent].pose.mul(&Isometry::new(
                        models[mapping.link].x_tree.translation,
                        models[mapping.link].x_tree.rotation.inverse(),
                    ));
                    let expected_b = states[mapping.link + 1].pose;
                    for point in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
                        close_vec(a.transform_point(point), expected_a.transform_point(point));
                        close_vec(b.transform_point(point), expected_b.transform_point(point));
                    }
                }
                XpbdSolver::default().step(&mut world, 1.0 / 600.0, 1);
                for mapping in std::iter::once(&report.base).chain(&report.links) {
                    assert!(world.bodies.position(mapping.body).unwrap().is_finite());
                    assert!(world.bodies.orientation(mapping.body).unwrap().is_finite());
                    assert!(world
                        .bodies
                        .linear_velocity(mapping.body)
                        .unwrap()
                        .is_finite());
                    assert!(world
                        .bodies
                        .angular_velocity(mapping.body)
                        .unwrap()
                        .is_finite());
                }
            }
        }
    }
}

#[test]
fn handoff_bakes_planes_without_changing_signed_distance() {
    let axes = Mat3::from_quat(Quat::from_rotation_z(0.6));
    let tensor = axes * Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0)) * axes.transpose();
    let inertia = SpatialInertia::new(
        2.0,
        Vec3::new(0.3, -0.2, 0.4),
        tensor.transpose().to_cols_array_2d(),
    );
    let normal = Vec3::new(1.0, -2.0, 3.0).normalize();
    let offset = -0.7;
    let local_pose = Isometry::new(
        Vec3::new(0.5, -0.4, 0.2),
        Quat::from_rotation_x(-0.3) * Quat::from_rotation_z(0.5),
    );
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        let mut builder = ArticulationBuilder::new(base, inertia)
            .with_base_pose(Isometry::new(
                Vec3::new(1.0, 2.0, -3.0),
                Quat::from_rotation_y(0.4),
            ))
            .with_gravity(Vec3::ZERO);
        builder
            .add_link(
                None,
                SpatialTransform::new(Quat::from_rotation_x(0.2), Vec3::Y),
                inertia,
                ArticulationJoint::revolute(Vec3::Z),
            )
            .unwrap();
        let mut tree = builder.build().unwrap();
        tree.set_position(0, 0.7);
        let states = tree.maximal_body_states();
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let shape = world.shapes.insert(ColliderShape::Plane { normal, offset });
        let articulation = world.spawn_articulation(tree);
        for link in [None, Some(0)] {
            let mut attachment = ArticulationColliderDesc::new(articulation, link, shape);
            attachment.local_pose = local_pose;
            world.attach_articulation_collider(attachment).unwrap();
        }
        world
            .articulations
            .get_mut(articulation)
            .unwrap()
            .break_joint(0);
        let report = world.handoff_articulation(articulation).unwrap();
        for mapping in &report.colliders {
            let old_world = states[mapping.link.map_or(0, |link| link + 1)]
                .pose
                .mul(&local_pose);
            let new_world = Isometry::new(
                world.bodies.position(mapping.body).unwrap(),
                world.bodies.orientation(mapping.body).unwrap(),
            );
            let ColliderShape::Plane {
                normal: baked_normal,
                offset: baked_offset,
            } = *world.shapes.get(mapping.body_shape).unwrap()
            else {
                panic!("unbounded planes must remain analytic planes")
            };
            assert_eq!(mapping.source_shape, shape);
            assert_eq!(mapping.local_pose, local_pose);
            close_vec(
                new_world.rotation * baked_normal,
                old_world.rotation * normal,
            );
            let tangent = normal.cross(Vec3::X).normalize();
            for point in [
                Vec3::ZERO,
                Vec3::X,
                Vec3::Y,
                Vec3::Z,
                normal * offset + tangent,
            ] {
                let point = old_world.transform_point(point);
                let old_distance = normal.dot(old_world.inverse().transform_point(point)) - offset;
                let new_distance =
                    baked_normal.dot(new_world.inverse().transform_point(point)) - baked_offset;
                close(new_distance, old_distance);
            }
        }
    }
}

#[test]
fn handoff_rejects_plane_overflow_without_allocating_shapes_or_bodies() {
    let mut world = PhysicsWorld::default();
    let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let plane = world.shapes.insert(ColliderShape::Plane {
        normal: Vec3::X,
        offset: f32::MAX,
    });
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default())
        .with_gravity(world.config.gravity);
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            point_inertia(1.0),
            ArticulationJoint::fixed(),
        )
        .unwrap();
    let articulation = world.spawn_articulation(builder.build().unwrap());
    // Stage a finite compound first, then fail on the following plane.
    for (link, shape, translation) in [
        (Some(0), sphere, Vec3::Y),
        (None, plane, Vec3::new(f32::MAX, 0.0, 0.0)),
    ] {
        let mut attachment = ArticulationColliderDesc::new(articulation, link, shape);
        attachment.local_pose = Isometry::from_translation(translation);
        world.attach_articulation_collider(attachment).unwrap();
    }
    world
        .articulations
        .get_mut(articulation)
        .unwrap()
        .break_joint(0);
    let mut untouched = world.clone();
    let before = prism_physics_core::snapshot::hash::hash_world(&world).unwrap();
    assert_eq!(
        world.handoff_articulation(articulation),
        Err(ArticulationHandoffError::InvalidColliderGeometry { link: None })
    );
    // Includes every allocator, pending event, attachment, shape and joint,
    // not just the externally visible rigid-body state.
    assert_eq!(
        prism_physics_core::snapshot::hash::hash_world(&world).unwrap(),
        before
    );
    assert!(world.articulations.contains(articulation));
    assert!(world.bodies.is_empty());
    assert!(world.joints.is_empty());
    assert_eq!(world.articulation_colliders.len(), 2);
    assert_eq!(world.shapes.len(), untouched.shapes.len());
    assert_eq!(
        world.shapes.insert(ColliderShape::Sphere { radius: 0.2 }),
        untouched
            .shapes
            .insert(ColliderShape::Sphere { radius: 0.2 })
    );
    assert_eq!(
        world.spawn(BodyDesc::dynamic_at(Vec3::ZERO)),
        untouched.spawn(BodyDesc::dynamic_at(Vec3::ZERO))
    );
}

#[test]
fn handoff_rejects_unrepresentable_inverse_inertia_atomically() {
    let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, point_inertia(2.0))
        .with_gravity(Vec3::ZERO);
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            SpatialInertia::diagonal(1.0, Vec3::new(0.1, 0.0, 0.0), Vec3::splat(1.0e-40)),
            ArticulationJoint::fixed(),
        )
        .unwrap();
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let articulation = world.spawn_articulation(builder.build().unwrap());
    world
        .articulations
        .get_mut(articulation)
        .unwrap()
        .break_joint(0);
    let states = world
        .articulations
        .get(articulation)
        .unwrap()
        .maximal_body_states();
    let shapes = world.shapes.len();
    assert_eq!(
        world.handoff_articulation(articulation),
        Err(ArticulationHandoffError::UnsupportedMassProperties { link: Some(0) })
    );
    assert!(world.bodies.is_empty());
    assert!(world.joints.is_empty());
    assert_eq!(world.shapes.len(), shapes);
    assert_eq!(
        world
            .articulations
            .get(articulation)
            .unwrap()
            .maximal_body_states(),
        states
    );
}

#[test]
fn multiple_attachments_keep_separate_source_mappings() {
    let mut world = PhysicsWorld::default();
    let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, point_inertia(3.0))
        .with_gravity(Vec3::new(0.0, -9.81, 0.0));
    builder
        .add_link(
            None,
            SpatialTransform::new(Quat::IDENTITY, Vec3::Y),
            point_inertia(1.0),
            ArticulationJoint::fixed(),
        )
        .unwrap();
    let articulation = world.spawn_articulation(builder.build().unwrap());
    world
        .attach_articulation_collider(ArticulationColliderDesc::new(articulation, Some(0), shape))
        .unwrap();
    world
        .attach_articulation_collider(ArticulationColliderDesc {
            articulation,
            link: Some(0),
            shape,
            local_pose: Isometry::IDENTITY,
            material: prism_physics_core::PhysicsMaterial::new(0.1, 0.0),
            self_collision: false,
        })
        .unwrap();
    world
        .articulations
        .get_mut(articulation)
        .unwrap()
        .break_joint(0);
    let report = world.handoff_articulation(articulation).unwrap();
    assert_eq!(report.colliders.len(), 2);
    assert_eq!(report.colliders[0].body, report.colliders[1].body);
    assert_eq!(report.colliders[0].body_child_range, Some(0..1));
    assert_eq!(report.colliders[1].body_child_range, Some(1..2));
    assert_ne!(report.colliders[0].material, report.colliders[1].material);
    assert!(world.articulation_colliders.is_empty());
    assert!(!world.articulations.contains(articulation));
}

#[test]
fn failed_handoff_preserves_observable_allocator_state() {
    let mut world = PhysicsWorld::default();
    let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, point_inertia(3.0))
        .with_gravity(Vec3::new(0.0, -9.81, 0.0));
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            point_inertia(1.0),
            ArticulationJoint::fixed(),
        )
        .unwrap();
    let articulation = world.spawn_articulation(builder.build().unwrap());
    world
        .attach_articulation_collider(ArticulationColliderDesc::new(articulation, Some(0), shape))
        .unwrap();
    let invalid = world.shapes.insert(ColliderShape::Sphere { radius: -0.5 });
    world
        .attach_articulation_collider(ArticulationColliderDesc::new(
            articulation,
            Some(0),
            invalid,
        ))
        .unwrap();
    world
        .articulations
        .get_mut(articulation)
        .unwrap()
        .break_joint(0);
    let mut untouched = world.clone();

    assert!(matches!(
        world.handoff_articulation(articulation),
        Err(ArticulationHandoffError::InvalidCompoundCollider(_))
    ));

    let next_shape = ColliderShape::Cuboid {
        half_extents: Vec3::splat(0.25),
    };
    let world_shape = world.shapes.insert(next_shape);
    let untouched_shape = untouched.shapes.insert(next_shape);
    assert_eq!(world_shape, untouched_shape);

    let world_a = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    let world_b = world.spawn(BodyDesc::dynamic_at(Vec3::X));
    let untouched_a = untouched.spawn(BodyDesc::static_at(Vec3::ZERO));
    let untouched_b = untouched.spawn(BodyDesc::dynamic_at(Vec3::X));
    assert_eq!((world_a, world_b), (untouched_a, untouched_b));
    let world_joint = world.spawn_joint(JointDesc::fixed(
        JointAnchor::at_center(world_a),
        JointAnchor::at_center(world_b),
    ));
    let untouched_joint = untouched.spawn_joint(JointDesc::fixed(
        JointAnchor::at_center(untouched_a),
        JointAnchor::at_center(untouched_b),
    ));
    assert_eq!(world_joint, untouched_joint);

    let fresh_tree = || {
        ArticulationBuilder::new(ArticulationBase::Floating, point_inertia(1.0))
            .build()
            .unwrap()
    };
    let world_articulation = world.spawn_articulation(fresh_tree());
    let untouched_articulation = untouched.spawn_articulation(fresh_tree());
    assert_eq!(world_articulation, untouched_articulation);
    let world_attachment = world
        .attach_articulation_collider(ArticulationColliderDesc::new(
            world_articulation,
            None,
            world_shape,
        ))
        .unwrap();
    let untouched_attachment = untouched
        .attach_articulation_collider(ArticulationColliderDesc::new(
            untouched_articulation,
            None,
            untouched_shape,
        ))
        .unwrap();
    assert_eq!(world_attachment, untouched_attachment);
}

#[test]
fn handoff_preserves_rest_frames_axes_coordinates_and_limits() {
    let base_pose = Isometry::new(
        Vec3::new(1.0, -2.0, 0.5),
        Quat::from_rotation_y(0.4) * Quat::from_rotation_x(-0.2),
    );
    let prismatic_tree =
        SpatialTransform::new(Quat::from_rotation_z(0.35), Vec3::new(0.4, -0.3, 0.2));
    let revolute_tree =
        SpatialTransform::new(Quat::from_rotation_x(-0.45), Vec3::new(-0.2, 0.6, 0.1));
    let prismatic_axis = Vec3::new(1.0, 2.0, -0.5).normalize();
    let revolute_axis = Vec3::new(-0.25, 0.5, 1.0).normalize();
    let prismatic_q = 0.37;
    let revolute_q = -0.41;
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default())
        .with_base_pose(base_pose)
        .with_gravity(Vec3::ZERO);
    builder
        .add_link(
            None,
            prismatic_tree,
            point_inertia(2.0),
            ArticulationJoint::prismatic(prismatic_axis)
                .with_limits(JointLimits::scalar(-0.2, 0.8)),
        )
        .unwrap();
    builder
        .add_link(
            Some(0),
            revolute_tree,
            point_inertia(1.5),
            ArticulationJoint::revolute(revolute_axis).with_limits(JointLimits::scalar(-0.9, 0.25)),
        )
        .unwrap();
    builder
        .add_link(
            Some(1),
            SpatialTransform::IDENTITY,
            point_inertia(1.0),
            ArticulationJoint::fixed(),
        )
        .unwrap();
    let mut tree = builder.build().unwrap();
    assert!(tree.set_position(0, prismatic_q));
    assert!(tree.set_position(1, revolute_q));
    let states = tree.maximal_body_states();
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let articulation = world.spawn_articulation(tree);
    world
        .articulations
        .get_mut(articulation)
        .unwrap()
        .break_joint(2);

    let report = world.handoff_articulation(articulation).unwrap();
    for (link_index, expected_q) in [(0, prismatic_q), (1, revolute_q)] {
        let mapping = report
            .joints
            .iter()
            .find(|mapping| mapping.link == link_index)
            .unwrap();
        let joint = world.joints.get(mapping.joint).unwrap();
        let position_a = world.bodies.position(joint.anchor_a.body).unwrap();
        let rotation_a = world.bodies.orientation(joint.anchor_a.body).unwrap();
        let position_b = world.bodies.position(joint.anchor_b.body).unwrap();
        let rotation_b = world.bodies.orientation(joint.anchor_b.body).unwrap();
        let point_a = position_a + rotation_a * joint.anchor_a.local_point;
        let point_b = position_b + rotation_b * joint.anchor_b.local_point;
        let frame_a = rotation_a * joint.anchor_a.local_frame;
        let frame_b = rotation_b * joint.anchor_b.local_frame;
        let expected_axis = states[link_index + 1].pose.rotation
            * match link_index {
                0 => prismatic_axis,
                1 => revolute_axis,
                _ => unreachable!(),
            };
        match joint.kind {
            JointKind::Prismatic(model) => {
                let axis_a = frame_a * model.axis;
                let axis_b = frame_b * model.axis;
                close_vec(axis_a, expected_axis);
                close_vec(axis_b, expected_axis);
                close((point_b - point_a).dot(axis_a), expected_q);
                close_vec((point_b - point_a) - axis_a * expected_q, Vec3::ZERO);
                assert_eq!(model.limit.unwrap().min, -0.2);
                assert_eq!(model.limit.unwrap().max, 0.8);
            }
            JointKind::Revolute(model) => {
                let axis_a = frame_a * model.axis;
                let axis_b = frame_b * model.axis;
                close_vec(point_a, point_b);
                close_vec(axis_a, expected_axis);
                close_vec(axis_b, expected_axis);
                let relative = frame_a.conjugate() * frame_b;
                close(relative.to_scaled_axis().dot(model.axis), expected_q);
                assert_eq!(model.limit.unwrap().min, -0.9);
                assert_eq!(model.limit.unwrap().max, 0.25);
            }
            _ => panic!("unexpected handoff joint family"),
        }
    }
}

#[test]
fn degraded_articulation_retains_colliders_for_handoff_without_contact_response() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    world.config.sleep.enabled = false;
    let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, point_inertia(1.0))
        .with_gravity(Vec3::ZERO);
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            point_inertia(1.0),
            ArticulationJoint::fixed(),
        )
        .unwrap();
    let articulation = world.spawn_articulation(builder.build().unwrap());
    let attachment = world
        .attach_articulation_collider(ArticulationColliderDesc::new(articulation, Some(0), shape))
        .unwrap();
    let descriptor = world
        .articulation_colliders
        .get(attachment)
        .unwrap()
        .clone();
    let rigid = world.spawn(
        BodyDesc::dynamic_at(Vec3::new(0.25, 0.0, 0.0))
            .with_collider(shape)
            .with_damping(0.0, 0.0),
    );
    assert!(world
        .articulations
        .get_mut(articulation)
        .unwrap()
        .break_joint(0)
        .is_some());
    let tree_before = world.articulations.get(articulation).unwrap().clone();
    let rigid_position = world.bodies.position(rigid).unwrap();
    let rigid_velocity = world.bodies.linear_velocity(rigid).unwrap();

    XpbdSolver::new().step(&mut world, 1.0 / 60.0, 1);

    let tree_after = world.articulations.get(articulation).unwrap();
    assert_eq!(tree_after.base_pose(), tree_before.base_pose());
    assert_eq!(tree_after.base_velocity(), tree_before.base_velocity());
    assert_eq!(tree_after.positions(), tree_before.positions());
    assert_eq!(tree_after.velocities(), tree_before.velocities());
    assert_eq!(world.bodies.position(rigid), Some(rigid_position));
    assert_eq!(world.bodies.linear_velocity(rigid), Some(rigid_velocity));
    assert_eq!(
        world.articulation_colliders.get(attachment),
        Some(&descriptor)
    );

    let report = world.handoff_articulation(articulation).unwrap();
    assert_eq!(report.colliders.len(), 1);
    assert_eq!(report.colliders[0].source, attachment);
    assert_eq!(report.colliders[0].body, report.links[0].body);
    assert_eq!(world.bodies.collider(report.links[0].body), Some(shape));
    assert!(world.articulation_colliders.is_empty());
    assert!(!world.articulations.contains(articulation));
}
