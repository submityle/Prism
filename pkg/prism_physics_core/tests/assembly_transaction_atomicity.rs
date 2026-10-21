//! Assembly replacement must either commit every owned mutation or leave the
//! complete world (including allocators and pending loads) unchanged.

use glam::{Quat, Vec3};
use prism_physics_core::collider::compound::{CompoundLeafPolicy, MAX_COMPOUND_CHILDREN};
use prism_physics_core::collider::{CompoundChild, PhysicsMaterial};
use prism_physics_core::dynamics::assembly::AssemblyError;
use prism_physics_core::dynamics::force::ForceError;
use prism_physics_core::joint::{DistanceJoint, RevoluteJoint};
use prism_physics_core::snapshot::world_hash::hash_world;
use prism_physics_core::world::AssemblyWorldError;
use prism_physics_core::{
    BodyDesc, BodyHandle, ColliderShape, Isometry, JointAnchor, JointDesc, JointKind,
    MassProperties, PhysicsWorld,
};

fn pair() -> (PhysicsWorld, [BodyHandle; 2]) {
    pair_with_desc(|desc| desc)
}

fn pair_with_desc(
    mut configure: impl FnMut(BodyDesc) -> BodyDesc,
) -> (PhysicsWorld, [BodyHandle; 2]) {
    let mut world = PhysicsWorld::with_gravity(-Vec3::Y * 9.0);
    let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.25 });
    let members = [-4.0, 4.0].map(|x| {
        world.spawn(configure(
            BodyDesc::dynamic_at(Vec3::X * x)
                .with_collider(shape)
                .with_mass_properties(MassProperties {
                    inv_mass: 1.0,
                    inv_inertia: Vec3::ONE,
                }),
        ))
    });
    world
        .connect_assembly(
            members[0],
            members[1],
            None,
            JointKind::Distance(DistanceJoint::rigid(8.0)),
            None,
        )
        .unwrap();
    (world, members)
}

fn rejected_unchanged<T: std::fmt::Debug>(
    world: &mut PhysicsWorld,
    expected: AssemblyWorldError,
    operation: impl FnOnce(&mut PhysicsWorld) -> Result<T, AssemblyWorldError>,
) {
    let before = hash_world(world).unwrap();
    let debug = format!("{world:?}");
    assert_eq!(operation(world).unwrap_err(), expected);
    assert_eq!(hash_world(world).unwrap(), before);
    assert_eq!(
        format!("{world:?}"),
        debug,
        "all state, not only visible poses"
    );
}

fn near(a: Vec3, b: Vec3) {
    assert!((a - b).length() < 2.0e-4, "{a:?} != {b:?}");
}

#[test]
fn merge_validates_actual_proxy_generation_before_mutating_anything() {
    let (mut world, members) = pair();
    let mut allocator = world.bodies.clone();
    for body in members {
        assert!(allocator.remove(body));
    }
    let future_proxy = allocator.insert(BodyDesc::dynamic_at(Vec3::ZERO));
    world.assemblies.add_body(future_proxy);
    assert!(world.apply_force(members[0], Vec3::Y, -4.0 * Vec3::X));
    for _ in 0..2 {
        rejected_unchanged(
            &mut world,
            AssemblyWorldError::Graph(AssemblyError::ProxyConflict),
            |w| w.merge_assembly(&members),
        );
    }
    assert!(world.assemblies.remove_body(future_proxy));
    assert_eq!(world.merge_assembly(&members).unwrap(), future_proxy);
}

#[test]
fn failed_graph_split_can_retry_without_losing_proxy_or_recovery_record() {
    let (mut world, members) = pair();
    let proxy = world.merge_assembly(&members).unwrap();
    assert!(world.bodies.set_position(proxy, Vec3::new(3.0, 5.0, 7.0)));
    assert!(world.apply_torque(proxy, Vec3::Z * 3.0));
    assert!(world.assemblies.remove_body(proxy));
    for _ in 0..3 {
        rejected_unchanged(
            &mut world,
            AssemblyWorldError::Graph(AssemblyError::ProxyConflict),
            |w| w.split_assembly_checked(proxy, &members),
        );
    }
    world.assemblies.add_body(proxy);
    let restored = world.split_assembly_checked(proxy, &members).unwrap();
    assert_eq!(restored.len(), members.len());
    assert!(!world.bodies.contains(proxy));
    assert!(world.external_forces.pending(proxy).is_none());
    rejected_unchanged(&mut world, AssemblyWorldError::NotMerged, |w| {
        w.split_assembly(proxy)
    });
}

#[test]
fn split_validates_predicted_restored_handles_before_body_allocation() {
    let (mut world, members) = pair();
    let proxy = world.merge_assembly(&members).unwrap();
    let mut allocator = world.bodies.clone();
    assert!(allocator.remove(proxy));
    let future_member = allocator.insert(BodyDesc::dynamic_at(Vec3::ZERO));
    world.assemblies.add_body(future_member);
    rejected_unchanged(
        &mut world,
        AssemblyWorldError::Graph(AssemblyError::DuplicateConnection),
        |w| w.split_assembly_checked(proxy, &members),
    );
    assert!(world.assemblies.remove_body(future_member));
    assert_eq!(
        world.split_assembly_checked(proxy, &members).unwrap()[0],
        future_member
    );
}

#[test]
fn checked_split_rejects_wrong_member_identity_count_and_order_atomically() {
    let (mut world, members) = pair();
    let proxy = world.merge_assembly(&members).unwrap();
    for expected in [
        vec![],
        vec![members[0]],
        vec![members[1], members[0]],
        vec![members[0], members[0]],
        vec![members[0], proxy],
        vec![members[0], BodyHandle::INVALID],
    ] {
        rejected_unchanged(&mut world, AssemblyWorldError::InvalidProxy, |w| {
            w.split_assembly_checked(proxy, &expected)
        });
    }
    assert_eq!(
        world.split_assembly_checked(proxy, &members).unwrap().len(),
        2
    );
}

#[test]
fn replacement_matches_legacy_allocator_order_with_reuse_and_append() {
    let (mut world, pair) = pair();
    let c = world.spawn(BodyDesc::dynamic_at(Vec3::Y));
    world
        .connect_assembly(
            pair[1],
            c,
            None,
            JointKind::Distance(DistanceJoint::rigid(1.0)),
            None,
        )
        .unwrap();
    let members = [pair[0], pair[1], c];
    let spare = world.spawn(BodyDesc::dynamic_at(Vec3::Z));
    assert!(world.bodies.remove(spare));
    assert!(world.assemblies.remove_body(spare));
    let mut allocator = world.bodies.clone();
    for body in members {
        assert!(allocator.remove(body));
    }
    let expected_proxy = allocator.insert(BodyDesc::default());
    // Caller order must not change the sorted core removal/insertion order.
    let proxy = world.merge_assembly(&[c, pair[1], pair[0]]).unwrap();
    assert_eq!(proxy, expected_proxy);
    // Consume all pre-existing free slots while the proxy is live. Split then
    // has to reuse the proxy slot first and append subsequent members.
    let mut unrelated = Vec::new();
    while world.bodies.len() < world.bodies.slot_count() {
        unrelated.push(world.spawn(BodyDesc::static_at(Vec3::splat(20.0))));
    }
    let mut allocator = world.bodies.clone();
    assert!(allocator.remove(proxy));
    let expected: Vec<_> = (0..members.len())
        .map(|_| allocator.insert(BodyDesc::default()))
        .collect();
    let restored = world.split_assembly_checked(proxy, &members).unwrap();
    assert_eq!(restored, expected);
    for body in unrelated {
        assert!(world.bodies.contains(body));
    }
    assert_eq!(
        world.bodies.insert(BodyDesc::default()),
        allocator.insert(BodyDesc::default())
    );
    assert!(world.bodies.columns_consistent());
}

#[test]
fn split_reuses_preexisting_free_slots_in_lifo_order() {
    let (mut world, members) = pair();
    let proxy = world.merge_assembly(&members).unwrap();
    let spare = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    assert!(world.bodies.remove(spare));
    assert!(world.assemblies.remove_body(spare));
    let restored = world.split_assembly_checked(proxy, &members).unwrap();
    assert_eq!(restored[0].index(), proxy.index());
    assert_eq!(restored[0].generation(), proxy.generation().wrapping_add(1));
    assert_eq!(restored[1].index(), spare.index());
    assert_eq!(restored[1].generation(), spare.generation().wrapping_add(1));
}

#[test]
fn invalid_geometry_and_leaf_policy_never_append_shapes_or_retire_bodies() {
    let (mut world, members) = pair();
    let original = world.bodies.collider(members[0]).unwrap();
    let plane = world.shapes.insert(ColliderShape::Plane {
        normal: Vec3::Y,
        offset: 0.0,
    });
    assert!(world.bodies.set_collider(members[0], Some(plane)));
    assert!(world.apply_force(members[1], Vec3::Z, Vec3::X * 4.0));
    rejected_unchanged(&mut world, AssemblyWorldError::UnsupportedCollider, |w| {
        w.merge_assembly(&members)
    });
    assert!(world.bodies.set_collider(members[0], Some(original)));
    for policies in [
        vec![CompoundLeafPolicy::default()],
        vec![
            CompoundLeafPolicy {
                material: Some(PhysicsMaterial {
                    friction: -1.0,
                    restitution: 0.0
                }),
                self_collision: None
            };
            2
        ],
    ] {
        rejected_unchanged(&mut world, AssemblyWorldError::UnsupportedCollider, |w| {
            w.merge_assembly_with_leaf_policies(&members, &policies)
        });
    }
    let before = world.shapes.len();
    let proxy = world
        .merge_assembly_with_leaf_policies(&members, &[CompoundLeafPolicy::default(); 2])
        .unwrap();
    assert_eq!(world.shapes.len(), before + 1);
    world.split_assembly_checked(proxy, &members).unwrap();
}

#[test]
fn expanded_geometry_budget_rejection_is_atomic_and_retryable() {
    let (mut world, members) = pair();
    let leaf = world.bodies.collider(members[0]).unwrap();
    let compound = world
        .shapes
        .insert_compound(&vec![
            CompoundChild {
                collider: leaf,
                local_pose: Isometry::IDENTITY,
            };
            MAX_COMPOUND_CHILDREN
        ])
        .unwrap();
    world.bodies.set_collider(members[0], Some(compound));
    rejected_unchanged(&mut world, AssemblyWorldError::UnsupportedCollider, |w| {
        w.merge_assembly(&members)
    });
    world.bodies.set_collider(members[1], None);
    let count = world.shapes.len();
    let proxy = world
        .merge_assembly_with_leaf_policies(
            &members,
            &vec![CompoundLeafPolicy::default(); MAX_COMPOUND_CHILDREN],
        )
        .unwrap();
    assert_eq!(world.shapes.len(), count + 1);
    world.split_assembly_checked(proxy, &members).unwrap();
}

#[test]
fn forged_compound_inline_cache_is_not_hidden_by_policy_flattening() {
    let (mut world, members) = pair();
    let leaf = world.bodies.collider(members[0]).unwrap();
    let valid = world
        .shapes
        .insert_compound(&[CompoundChild {
            collider: leaf,
            local_pose: Isometry::IDENTITY,
        }])
        .unwrap();
    let mut forged = *world.shapes.get(valid).unwrap();
    if let ColliderShape::Compound { local_aabb_min, .. } = &mut forged {
        *local_aabb_min = Vec3::splat(-50.0);
    }
    let forged = world.shapes.insert(forged);
    world.bodies.set_collider(members[0], Some(forged));
    rejected_unchanged(&mut world, AssemblyWorldError::UnsupportedCollider, |w| {
        w.merge_assembly_with_leaf_policies(&members, &[CompoundLeafPolicy::default(); 2])
    });
}

#[test]
fn merge_force_sum_and_transport_overflow_leave_all_state_unchanged() {
    for transport in [false, true] {
        let (mut world, members) = pair();
        let huge = f32::MAX * 0.75;
        let force = if transport {
            Vec3::Y * huge
        } else {
            Vec3::X * huge
        };
        assert!(world.apply_force(members[0], force, -4.0 * Vec3::X));
        if !transport {
            assert!(world.apply_force(members[1], force, 4.0 * Vec3::X));
        }
        rejected_unchanged(
            &mut world,
            AssemblyWorldError::Force(ForceError::NonFiniteInput),
            |w| w.merge_assembly(&members),
        );
        assert!(world.external_forces.pending(members[0]).is_some());
        world.external_forces.clear();
        let proxy = world.merge_assembly(&members).unwrap();
        assert!(world.external_forces.pending(proxy).is_none());
    }
}

#[test]
fn split_wrench_overflow_is_atomic_and_can_retry_without_retired_loads() {
    let (mut world, members) = pair();
    let proxy = world.merge_assembly(&members).unwrap();
    let center = world.bodies.position(proxy).unwrap();
    assert!(world.apply_force(proxy, Vec3::Y * (f32::MAX * 0.75), center));
    for _ in 0..2 {
        rejected_unchanged(
            &mut world,
            AssemblyWorldError::Force(ForceError::NonFiniteInput),
            |w| w.split_assembly_checked(proxy, &members),
        );
    }
    assert!(world.external_forces.pending(proxy).is_some());
    world.external_forces.clear();
    assert!(world.apply_force(proxy, Vec3::Y * 2.0, center));
    let restored = world.split_assembly_checked(proxy, &members).unwrap();
    assert!(world.external_forces.pending(proxy).is_none());
    for body in restored {
        assert!(world.external_forces.pending(body).is_some());
    }
}

#[test]
fn wrench_remapping_conserves_force_and_torque_and_leaves_unrelated_loads() {
    let (mut world, members) = pair();
    let other = world.spawn(BodyDesc::dynamic_at(Vec3::Z * 20.0));
    assert!(world.apply_force(other, Vec3::ONE, Vec3::Z * 20.0));
    assert!(world.apply_force(members[0], Vec3::Y * 2.0, -4.0 * Vec3::X));
    assert!(world.apply_force(members[1], Vec3::Z * 3.0, 4.0 * Vec3::X));
    assert!(world.apply_torque(members[0], Vec3::X));
    let unrelated = world.external_forces.pending(other);
    let proxy = world.merge_assembly(&members).unwrap();
    for body in members {
        assert!(world.external_forces.pending(body).is_none());
    }
    let combined = world.external_forces.pending(proxy).unwrap();
    near(combined.force, Vec3::new(0.0, 2.0, 3.0));
    near(combined.torque, Vec3::new(1.0, -12.0, -8.0));
    let center = world.bodies.position(proxy).unwrap();
    let restored = world.split_assembly_checked(proxy, &members).unwrap();
    let mut force = Vec3::ZERO;
    let mut torque = Vec3::ZERO;
    for body in restored {
        let pending = world.external_forces.pending(body).unwrap();
        force += pending.force;
        torque +=
            pending.torque + (world.bodies.position(body).unwrap() - center).cross(pending.force);
    }
    near(force, combined.force);
    near(torque, combined.torque);
    assert_eq!(world.external_forces.pending(other), unrelated);
    assert!(world.external_forces.pending(proxy).is_none());
}

#[test]
fn merge_damping_average_does_not_overflow_for_finite_inputs() {
    let huge = f32::MAX * 0.75;
    let (mut world, members) = pair_with_desc(|mut desc| {
        desc.linear_damping = huge;
        desc.angular_damping = huge;
        desc
    });
    let proxy = world.merge_assembly(&members).unwrap();
    {
        let view = world.bodies.solver_view_mut();
        assert_eq!(view.linear_damping[proxy.index() as usize], huge);
        assert_eq!(view.angular_damping[proxy.index() as usize], huge);
    }
    let restored = world.split_assembly_checked(proxy, &members).unwrap();
    let view = world.bodies.solver_view_mut();
    for body in restored {
        assert_eq!(view.linear_damping[body.index() as usize], huge);
        assert_eq!(view.angular_damping[body.index() as usize], huge);
    }
}

#[test]
fn invalid_member_damping_is_rejected_before_world_mutation() {
    for invalid in [-1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        for angular in [false, true] {
            let (mut world, members) = pair_with_desc(|mut desc| {
                if angular {
                    desc.angular_damping = invalid;
                } else {
                    desc.linear_damping = invalid;
                }
                desc
            });
            // The diagnostic hash intentionally rejects NaN input worlds.
            let before = format!("{world:?}");
            assert_eq!(
                world.merge_assembly(&members),
                Err(AssemblyWorldError::InvalidDamping)
            );
            assert_eq!(format!("{world:?}"), before);
        }
    }
}

#[test]
fn split_preserves_current_gravity_sleep_twist_and_previous_pose_contract() {
    let (mut world, members) = pair();
    let proxy = world.merge_assembly(&members).unwrap();
    let initial_q = world.bodies.orientation(proxy).unwrap();
    let turn = Quat::from_rotation_z(0.7) * Quat::from_rotation_y(0.3);
    let center = Vec3::new(7.0, 8.0, 9.0);
    let velocity = Vec3::new(1.0, -2.0, 3.0);
    let omega = Vec3::new(-0.4, 0.8, 1.2);
    world.bodies.set_position(proxy, center);
    world.bodies.set_orientation(proxy, turn * initial_q);
    world.bodies.set_linear_velocity(proxy, velocity);
    world.bodies.set_angular_velocity(proxy, omega);
    world
        .bodies
        .set_gravity_override(proxy, Some(Vec3::X * 2.0));
    {
        let view = world.bodies.solver_view_mut();
        let i = proxy.index() as usize;
        view.sleeping[i] = true;
        view.sleep_timers[i] = 3.5;
        view.prev_positions[i] = Vec3::splat(-100.0);
    }
    let restored = world.split_assembly_checked(proxy, &members).unwrap();
    for (body, x) in restored.into_iter().zip([-4.0, 4.0]) {
        let radius = turn * (Vec3::X * x);
        near(world.bodies.position(body).unwrap(), center + radius);
        near(
            world.bodies.linear_velocity(body).unwrap(),
            velocity + omega.cross(radius),
        );
        assert_eq!(world.bodies.angular_velocity(body), Some(omega));
        assert_eq!(
            world.bodies.gravity_override(body),
            Some(Some(Vec3::X * 2.0))
        );
        assert_eq!(world.bodies.is_sleeping(body), Some(true));
        assert_eq!(world.bodies.sleep_timer(body), Some(3.5));
        assert_eq!(
            world.bodies.prev_position(body),
            world.bodies.position(body)
        );
        assert_eq!(
            world.bodies.prev_orientation(body),
            world.bodies.orientation(body)
        );
    }
}

#[test]
fn policy_merge_keeps_rotated_nonzero_com_leaf_frames_and_single_shape_append() {
    let (mut world, members) = pair();
    let leaf = world.bodies.collider(members[0]).unwrap();
    let override_material = PhysicsMaterial::new(0.8, 0.6);
    let member_materials = [
        PhysicsMaterial::new(0.2, 0.1),
        PhysicsMaterial::new(0.7, 0.3),
    ];
    let attachments = [
        Isometry::new(Vec3::new(1.0, 0.5, -0.2), Quat::from_rotation_y(0.7)),
        Isometry::new(-Vec3::Y, Quat::from_rotation_z(-0.4)),
    ];
    let member_compound = world
        .shapes
        .insert_compound_with_policies(
            &attachments.map(|local_pose| CompoundChild {
                collider: leaf,
                local_pose,
            }),
            &[
                CompoundLeafPolicy {
                    material: Some(override_material),
                    self_collision: Some(false),
                },
                CompoundLeafPolicy {
                    material: None,
                    self_collision: Some(true),
                },
            ],
        )
        .unwrap();
    world.bodies.set_collider(members[0], Some(member_compound));
    let centers = [Vec3::new(2.0, 3.0, 1.0), Vec3::new(8.0, -1.0, 2.0)];
    let rotations = [Quat::from_rotation_x(0.4), Quat::from_rotation_z(0.8)];
    for i in 0..2 {
        world.bodies.set_position(members[i], centers[i]);
        world.bodies.set_orientation(members[i], rotations[i]);
        world.bodies.set_material(members[i], member_materials[i]);
    }
    let policies = [
        CompoundLeafPolicy {
            material: Some(override_material),
            self_collision: Some(false),
        },
        CompoundLeafPolicy {
            material: Some(member_materials[0]),
            self_collision: Some(true),
        },
        CompoundLeafPolicy {
            material: Some(member_materials[1]),
            self_collision: None,
        },
    ];
    let shape_count = world.shapes.len();
    let proxy = world
        .merge_assembly_with_leaf_policies(&[members[1], members[0]], &policies)
        .unwrap();
    assert_eq!(world.shapes.len(), shape_count + 1);
    near(
        world.bodies.position(proxy).unwrap(),
        (centers[0] + centers[1]) * 0.5,
    );
    let shape = world.bodies.collider(proxy).unwrap();
    let ColliderShape::Compound { compound, .. } = world.shapes.get(shape).unwrap() else {
        panic!("compound proxy");
    };
    let data = world.shapes.compound(*compound).unwrap();
    assert_eq!(data.policies(), &policies);
    let proxy_pose = Isometry::new(
        world.bodies.position(proxy).unwrap(),
        world.bodies.orientation(proxy).unwrap(),
    );
    let expected = [
        Isometry::new(centers[0], rotations[0]).mul(&attachments[0]),
        Isometry::new(centers[0], rotations[0]).mul(&attachments[1]),
        Isometry::new(centers[1], rotations[1]),
    ];
    for (child, expected) in data.children().iter().zip(expected) {
        let actual = proxy_pose.mul(&child.local_pose);
        near(actual.translation, expected.translation);
        assert!(actual.rotation.dot(expected.rotation).abs() > 1.0 - 1.0e-5);
    }
    let restored = world.split_assembly_checked(proxy, &members).unwrap();
    assert_eq!(world.bodies.collider(restored[0]), Some(member_compound));
    assert_eq!(world.bodies.collider(restored[1]), Some(leaf));
    assert_eq!(
        world.bodies.material(restored[0]),
        Some(member_materials[0])
    );
    assert_eq!(
        world.bodies.material(restored[1]),
        Some(member_materials[1])
    );
}

#[test]
fn checked_split_preserves_joint_coordinate_history_across_failed_retry() {
    let (mut world, members) = pair();
    // Use a fresh graph edge owned by a live joint instead of the helper edge.
    let old_edge = world.assemblies.connections().next().unwrap().id;
    world.assemblies.disconnect(old_edge);
    let joint = world.spawn_joint(JointDesc::new(
        JointAnchor::at_center(members[0]),
        JointAnchor::at_center(members[1]),
        JointKind::Revolute(RevoluteJoint::new(Vec3::Z)),
    ));
    assert!(world
        .joints
        .get_mut(joint)
        .unwrap()
        .set_revolute_coordinate(20.0, [Quat::IDENTITY; 2]));
    let proxy = world.merge_assembly(&members).unwrap();
    assert!(!world.joints.get(joint).unwrap().enabled);
    world.assemblies.remove_body(proxy);
    rejected_unchanged(
        &mut world,
        AssemblyWorldError::Graph(AssemblyError::ProxyConflict),
        |w| w.split_assembly_checked(proxy, &members),
    );
    world.assemblies.add_body(proxy);
    let restored = world.split_assembly_checked(proxy, &members).unwrap();
    let result = world.joints.get(joint).unwrap();
    assert_eq!(result.anchor_a.body, restored[0]);
    assert_eq!(result.anchor_b.body, restored[1]);
    assert!(result.enabled);
    let view = world.bodies.solver_view_mut();
    prism_physics_core::solver::xpbd::joint_constraint::begin_revolute_substep(
        &view,
        &mut world.joints,
        1.0 / 60.0,
    );
    prism_physics_core::solver::xpbd::joint_constraint::finish_revolute_substep(
        &view,
        &mut world.joints,
    );
    assert!(
        (world
            .joints
            .get(joint)
            .unwrap()
            .revolute_state()
            .angle()
            .unwrap()
            - 20.0)
            .abs()
            < 1.0e-5
    );
}
