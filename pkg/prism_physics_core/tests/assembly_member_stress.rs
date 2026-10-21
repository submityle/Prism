//! Real member forces drive disabled merged welds through the production XPBD
//! boundary. Happy paths never inject solver/joint load measurements manually.

use glam::{Quat, Vec3};
use prism_physics_core::dynamics::assembly::{AssemblyError, BreakLimit};
use prism_physics_core::dynamics::force::inertia_world;
use prism_physics_core::joint::{DistanceJoint, JointAnchor, JointKind};
use prism_physics_core::snapshot::world::FullStateSnapshot;
use prism_physics_core::snapshot::world_hash::hash_world;
use prism_physics_core::solver::{Solver, XpbdSolver};
use prism_physics_core::world::{AssemblyMemberWrench, AssemblyStressError, AssemblyWorldError};
use prism_physics_core::{
    BodyDesc, BodyHandle, JointDesc, JointHandle, MassProperties, PhysicsWorld,
};

fn wrench(member: BodyHandle, force: Vec3, torque: Vec3) -> AssemblyMemberWrench {
    AssemblyMemberWrench {
        member,
        force,
        torque,
    }
}

fn weld(
    world: &mut PhysicsWorld,
    a: BodyHandle,
    b: BodyHandle,
    anchor: Vec3,
    limit: BreakLimit,
) -> JointHandle {
    let anchors = [a, b].map(|body| {
        let position = world.bodies.position(body).unwrap();
        let orientation = world.bodies.orientation(body).unwrap();
        JointAnchor::new(
            body,
            orientation.inverse() * (anchor - position),
            orientation.inverse(),
        )
    });
    let desc = JointDesc::fixed(anchors[0], anchors[1]);
    let joint = world.joints.insert(desc);
    // Deliberately reverse graph endpoint order. Reaction side A is defined
    // by the actual joint anchors, not graph insertion order.
    world
        .connect_assembly(b, a, Some(joint), desc.kind, Some(limit))
        .unwrap();
    joint
}

fn pair(limit: BreakLimit) -> (PhysicsWorld, [BodyHandle; 2], BodyHandle, JointHandle) {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let members =
        [-1.0, 1.0].map(|x| world.spawn(BodyDesc::dynamic_at(Vec3::X * x).with_damping(0.0, 0.0)));
    let joint = weld(&mut world, members[0], members[1], Vec3::ZERO, limit);
    let proxy = world.merge_assembly(&members).unwrap();
    assert!(!world.joints.get(joint).unwrap().enabled);
    (world, members, proxy, joint)
}

fn opposed(members: [BodyHandle; 2], magnitude: f32) -> [AssemblyMemberWrench; 2] {
    [
        wrench(members[0], -Vec3::X * magnitude, Vec3::ZERO),
        wrench(members[1], Vec3::X * magnitude, Vec3::ZERO),
    ]
}

fn near(actual: Vec3, expected: Vec3) {
    assert!(
        (actual - expected).length() < 2.0e-4,
        "actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn zero_resultant_member_tension_breaks_a_disabled_internal_weld_in_xpbd() {
    let (mut world, members, proxy, joint) = pair(BreakLimit::new(10.0, 1000.0));
    let reactions = world
        .apply_assembly_member_wrenches(proxy, &opposed(members, 20.0))
        .unwrap();
    assert_eq!(reactions.len(), 1);
    assert_eq!(reactions[0].joint, joint);
    near(reactions[0].force, Vec3::X * 20.0);
    near(reactions[0].torque, Vec3::ZERO);
    assert!(
        world.external_forces.is_empty(),
        "member queue has independent ownership"
    );
    assert_eq!(
        world.bodies.len(),
        1,
        "fracture waits for the fixed boundary"
    );
    XpbdSolver::new().step(&mut world, 0.01, 1);
    assert!(!world.bodies.contains(proxy));
    assert!(world.joint_boundary_failure().is_none());
    let reports = world.drain_assembly_split_reports();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].members.len(), 2);
    assert_eq!(world.assemblies.connections().count(), 0);
    for remap in &reports[0].members {
        near(
            world.bodies.linear_velocity(remap.current).unwrap(),
            Vec3::ZERO,
        );
        near(
            world.bodies.angular_velocity(remap.current).unwrap(),
            Vec3::ZERO,
        );
    }
    assert!(!world.joints.get(joint).unwrap().enabled);
}

#[test]
fn one_sided_force_recovers_half_load_after_rigid_acceleration_not_net_wrench() {
    for (threshold, should_break) in [(15.0, false), (9.0, true)] {
        let (mut world, members, proxy, _) = pair(BreakLimit::new(threshold, 1000.0));
        let reactions = world
            .apply_assembly_member_wrenches(
                proxy,
                &[wrench(members[1], Vec3::X * 20.0, Vec3::ZERO)],
            )
            .unwrap();
        near(reactions[0].force, Vec3::X * 10.0);
        XpbdSolver::new().step(&mut world, 0.01, 1);
        assert_eq!(!world.bodies.contains(proxy), should_break);
        if should_break {
            let reports = world.drain_assembly_split_reports();
            assert_eq!(reports.len(), 1);
            for remap in &reports[0].members {
                near(
                    world.bodies.linear_velocity(remap.current).unwrap(),
                    Vec3::X * 0.1,
                );
            }
        } else {
            near(world.bodies.linear_velocity(proxy).unwrap(), Vec3::X * 0.1);
        }
    }
}

#[test]
fn mass_proportional_member_acceleration_has_no_spurious_internal_load() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let members = [(1.0, -1.0), (3.0, 1.0)].map(|(mass, x)| {
        world.spawn(
            BodyDesc::dynamic_at(Vec3::X * x).with_mass_properties(MassProperties {
                inv_mass: 1.0 / mass,
                inv_inertia: Vec3::ONE,
            }),
        )
    });
    weld(
        &mut world,
        members[0],
        members[1],
        Vec3::ZERO,
        BreakLimit::new(0.001, 0.001),
    );
    let proxy = world.merge_assembly(&members).unwrap();
    let reactions = world
        .apply_assembly_member_wrenches(
            proxy,
            &[
                wrench(members[0], Vec3::Y * 2.0, Vec3::ZERO),
                wrench(members[1], Vec3::Y * 6.0, Vec3::ZERO),
            ],
        )
        .unwrap();
    near(reactions[0].force, Vec3::ZERO);
    near(reactions[0].torque, Vec3::ZERO);
    XpbdSolver::new().step(&mut world, 0.01, 1);
    assert!(world.bodies.contains(proxy));
    near(world.bodies.linear_velocity(proxy).unwrap(), Vec3::Y * 0.02);
}

#[test]
fn balanced_member_torques_break_without_proxy_angular_acceleration() {
    let (mut world, members, proxy, _) = pair(BreakLimit::new(1000.0, 5.0));
    let reactions = world
        .apply_assembly_member_wrenches(
            proxy,
            &[
                wrench(members[0], Vec3::ZERO, -Vec3::Z * 8.0),
                wrench(members[1], Vec3::ZERO, Vec3::Z * 8.0),
            ],
        )
        .unwrap();
    near(reactions[0].force, Vec3::ZERO);
    near(reactions[0].torque, Vec3::Z * 8.0);
    XpbdSolver::new().step(&mut world, 0.01, 1);
    assert!(!world.bodies.contains(proxy));
    for remap in &world.drain_assembly_split_reports()[0].members {
        near(
            world.bodies.angular_velocity(remap.current).unwrap(),
            Vec3::ZERO,
        );
    }
}

#[test]
fn transverse_member_force_uses_full_assembly_inertia_and_joint_anchor_couple() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let members = [-1.0, 1.0].map(|x| world.spawn(BodyDesc::dynamic_at(Vec3::X * x)));
    weld(
        &mut world,
        members[0],
        members[1],
        -Vec3::X,
        BreakLimit::new(4.0, 2.0),
    );
    let proxy = world.merge_assembly(&members).unwrap();
    let reactions = world
        .apply_assembly_member_wrenches(proxy, &[wrench(members[1], Vec3::Y * 12.0, Vec3::ZERO)])
        .unwrap();
    // a_y=6, alpha_z=12/(1+1+1+1)=3. Left member needs F_y=3;
    // its own rotational inertia needs a 3 Nm couple at the left-COM anchor.
    near(reactions[0].force, Vec3::Y * 3.0);
    near(reactions[0].torque, Vec3::Z * 3.0);
    XpbdSolver::new().step(&mut world, 0.01, 1);
    assert!(
        !world.bodies.contains(proxy),
        "the anchor couple exceeds the torque limit"
    );
    assert_eq!(world.drain_assembly_split_reports().len(), 1);
}

#[test]
fn rotated_anisotropic_members_recover_zero_reaction_for_their_exact_rigid_inertial_wrenches() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let common = Quat::from_rotation_y(0.6) * Quat::from_rotation_z(0.4);
    let descriptors = [(-1.0, 1.0), (1.0, 2.0)].map(|(x, mass)| {
        let mut desc = BodyDesc::dynamic_at(common * (Vec3::X * x));
        desc.orientation = common * Quat::from_rotation_x(0.3 * x);
        desc.mass_properties = MassProperties {
            inv_mass: 1.0 / mass,
            inv_inertia: Vec3::new(0.5, 0.25, 0.125) / mass,
        };
        desc
    });
    let members = descriptors.map(|desc| world.spawn(desc));
    weld(
        &mut world,
        members[0],
        members[1],
        Vec3::ZERO,
        BreakLimit::new(0.001, 0.001),
    );
    let proxy = world.merge_assembly(&members).unwrap();
    let center = world.bodies.position(proxy).unwrap();
    let acceleration = Vec3::new(1.0, -0.5, 0.2);
    let alpha = Vec3::new(0.3, 0.5, -0.2);
    let loads = [0, 1].map(|i| {
        let desc = descriptors[i];
        wrench(
            members[i],
            (acceleration + alpha.cross(desc.position - center)) / desc.mass_properties.inv_mass,
            inertia_world(desc.orientation, desc.mass_properties) * alpha,
        )
    });
    let reactions = world.apply_assembly_member_wrenches(proxy, &loads).unwrap();
    near(reactions[0].force, Vec3::ZERO);
    near(reactions[0].torque, Vec3::ZERO);
    XpbdSolver::new().step(&mut world, 0.01, 1);
    assert!(world.bodies.contains(proxy));
    near(
        world.bodies.linear_velocity(proxy).unwrap(),
        acceleration * 0.01,
    );
    near(world.bodies.angular_velocity(proxy).unwrap(), alpha * 0.01);
}

#[test]
fn tree_cut_loads_break_only_overloaded_edge_and_preserve_surviving_connection() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let members = [-1.0, 0.0, 1.0].map(|x| world.spawn(BodyDesc::dynamic_at(Vec3::X * x)));
    let weak = weld(
        &mut world,
        members[0],
        members[1],
        -Vec3::X * 0.5,
        BreakLimit::new(10.0, 1000.0),
    );
    let strong = weld(
        &mut world,
        members[1],
        members[2],
        Vec3::X * 0.5,
        BreakLimit::new(100.0, 1000.0),
    );
    let proxy = world.merge_assembly(&members).unwrap();
    let reactions = world
        .apply_assembly_member_wrenches(
            proxy,
            &[
                wrench(members[0], -Vec3::X * 12.0, Vec3::ZERO),
                wrench(members[2], Vec3::X * 12.0, Vec3::ZERO),
            ],
        )
        .unwrap();
    assert_eq!(reactions.len(), 2);
    for value in reactions {
        near(value.force, Vec3::X * 12.0);
    }
    XpbdSolver::new().step(&mut world, 0.01, 1);
    assert!(!world.bodies.contains(proxy));
    assert!(!world.joints.get(weak).unwrap().enabled);
    assert!(world.joints.get(strong).unwrap().enabled);
    assert_eq!(world.assemblies.connections().count(), 1);
    let reports = world.drain_assembly_split_reports();
    assert_eq!(reports.len(), 1);
    assert_eq!(
        reports[0].members.len(),
        3,
        "uses the existing full-split ownership protocol"
    );
    let mut sizes: Vec<_> = world
        .assemblies
        .connected_components()
        .iter()
        .map(Vec::len)
        .collect();
    sizes.sort();
    assert_eq!(sizes, [1, 2]);
}

#[test]
fn member_forces_are_one_shot_across_substeps_and_do_not_double_count_proxy() {
    let (mut world, members, proxy, _) = pair(BreakLimit::new(1000.0, 1000.0));
    world
        .apply_assembly_member_wrenches(proxy, &[wrench(members[1], Vec3::X * 20.0, Vec3::ZERO)])
        .unwrap();
    let center = world.bodies.position(proxy).unwrap();
    assert!(world.apply_force(proxy, Vec3::X * 4.0, center));
    let mut solver = XpbdSolver::new();
    solver.step(&mut world, 0.03, 3);
    near(world.bodies.linear_velocity(proxy).unwrap(), Vec3::X * 0.12);
    solver.step(&mut world, 0.03, 3);
    near(world.bodies.linear_velocity(proxy).unwrap(), Vec3::X * 0.12);
    assert!(world.joint_boundary_failure().is_none());

    world
        .apply_assembly_member_wrenches(proxy, &opposed(members, 1.0))
        .unwrap();
    let center = world.bodies.position(proxy).unwrap();
    assert!(world.apply_force(proxy, Vec3::X, center));
    assert_eq!(
        world.consume_external_forces(0.01),
        1,
        "count unique accelerated bodies"
    );
    world.submit_joint_loads(0.01, &[]).unwrap();
}

#[test]
fn sustained_real_member_tension_accumulates_fatigue_only_once_per_loaded_substep() {
    let (mut world, members, proxy, _) = pair(BreakLimit::new(10.0, 1000.0).with_fatigue(1.0));
    let mut solver = XpbdSolver::new();
    for _ in 0..6 {
        world
            .apply_assembly_member_wrenches(proxy, &opposed(members, 6.0))
            .unwrap();
        solver.step(&mut world, 0.1, 1);
        assert!(world.bodies.contains(proxy));
    }
    // No load during these boundaries: stored impulses must not be replayed.
    solver.step(&mut world, 0.5, 5);
    assert!(world.bodies.contains(proxy));
    world
        .apply_assembly_member_wrenches(proxy, &opposed(members, 6.0))
        .unwrap();
    solver.step(&mut world, 0.1, 1);
    assert!(!world.bodies.contains(proxy));
    assert_eq!(world.drain_assembly_split_reports().len(), 1);
}

#[test]
fn canonical_batches_and_full_snapshot_restore_preserve_pending_member_loads() {
    let (mut forward, members, proxy, _) = pair(BreakLimit::new(9.0, 1000.0));
    let mut reverse = forward.clone();
    let loads = opposed(members, 12.0);
    forward
        .apply_assembly_member_wrenches(proxy, &loads)
        .unwrap();
    reverse
        .apply_assembly_member_wrenches(proxy, &[loads[1]])
        .unwrap();
    reverse
        .apply_assembly_member_wrenches(proxy, &[loads[0]])
        .unwrap();
    assert_eq!(hash_world(&forward).unwrap(), hash_world(&reverse).unwrap());
    let snapshot = FullStateSnapshot::capture(1, &forward);
    let mut restored = PhysicsWorld::default();
    snapshot.restore_into(&mut restored);
    for world in [&mut forward, &mut reverse, &mut restored] {
        XpbdSolver::new().step(world, 0.01, 1);
        assert!(!world.bodies.contains(proxy));
    }
    assert_eq!(hash_world(&forward).unwrap(), hash_world(&reverse).unwrap());
    assert_eq!(
        hash_world(&forward).unwrap(),
        hash_world(&restored).unwrap()
    );
}

#[test]
fn automatic_member_fracture_failure_retains_exact_loads_and_retry_does_not_apply_force_twice() {
    let (mut world, members, proxy, _) = pair(BreakLimit::new(9.0, 1000.0));
    world
        .apply_assembly_member_wrenches(proxy, &[wrench(members[1], Vec3::X * 20.0, Vec3::ZERO)])
        .unwrap();
    assert!(world.assemblies.remove_body(proxy));
    XpbdSolver::new().step(&mut world, 0.01, 1);
    assert!(matches!(
        world.joint_boundary_failure().unwrap().error,
        prism_physics_core::dynamics::assembly::BoundaryLoadError::AssemblySplit {
            error: AssemblyWorldError::Graph(AssemblyError::ProxyConflict),
            ..
        }
    ));
    near(world.bodies.linear_velocity(proxy).unwrap(), Vec3::X * 0.1);
    let snapshot = FullStateSnapshot::capture(2, &world);
    let hash = hash_world(&world).unwrap();
    for _ in 0..3 {
        assert!(world.retry_joint_boundary().is_err());
        assert_eq!(hash_world(&world).unwrap(), hash);
        assert_eq!(
            world.apply_assembly_member_wrenches(proxy, &opposed(members, 20.0)),
            Err(AssemblyStressError::BoundaryPending)
        );
        assert_eq!(hash_world(&world).unwrap(), hash);
    }
    let mut restored = snapshot.fork_world();
    for world in [&mut world, &mut restored] {
        world.assemblies.add_body(proxy);
        world.retry_joint_boundary().unwrap();
        assert!(world.joint_boundary_failure().is_none());
        let reports = world.drain_assembly_split_reports();
        assert_eq!(reports.len(), 1);
        for remap in &reports[0].members {
            near(
                world.bodies.linear_velocity(remap.current).unwrap(),
                Vec3::X * 0.1,
            );
        }
        world.retry_joint_boundary().unwrap();
        assert!(world.drain_assembly_split_reports().is_empty());
    }
    assert_eq!(hash_world(&world).unwrap(), hash_world(&restored).unwrap());
}

#[test]
fn member_load_rejection_is_atomic_for_unknown_generation_nonfinite_and_overflow() {
    let (mut world, members, proxy, _) = pair(BreakLimit::new(1000.0, 1000.0));
    let unrelated = world.spawn(BodyDesc::dynamic_at(Vec3::Z * 10.0));
    world
        .apply_assembly_member_wrenches(proxy, &opposed(members, 1.0))
        .unwrap();
    for (loads, error) in [
        (
            vec![wrench(unrelated, Vec3::X, Vec3::ZERO)],
            AssemblyStressError::UnknownMember,
        ),
        (
            vec![wrench(members[0], Vec3::splat(f32::NAN), Vec3::ZERO)],
            AssemblyStressError::NonFiniteLoad,
        ),
        (
            vec![wrench(members[0], Vec3::splat(f32::MAX), Vec3::ZERO); 2],
            AssemblyStressError::NonFiniteLoad,
        ),
        (
            vec![wrench(members[0], Vec3::ZERO, Vec3::ZERO); 4096],
            AssemblyStressError::CapacityExceeded,
        ),
    ] {
        let before = hash_world(&world).unwrap();
        assert_eq!(
            world.apply_assembly_member_wrenches(proxy, &loads),
            Err(error)
        );
        assert_eq!(hash_world(&world).unwrap(), before);
    }
    let before = hash_world(&world).unwrap();
    assert_eq!(
        world.apply_assembly_member_wrenches(unrelated, &[]),
        Err(AssemblyStressError::InvalidProxy)
    );
    assert_eq!(hash_world(&world).unwrap(), before);
    assert_eq!(world.consume_external_forces(f32::NAN), 0);
    assert_eq!(hash_world(&world).unwrap(), before);
    XpbdSolver::new().step(&mut world, 0.01, 1);
    assert!(world.bodies.contains(proxy));
}

#[test]
fn manual_split_requires_explicit_cancel_and_recycled_proxy_cannot_accept_old_members() {
    let (mut world, members, proxy, _) = pair(BreakLimit::new(1000.0, 1000.0));
    world
        .apply_assembly_member_wrenches(proxy, &opposed(members, 1.0))
        .unwrap();
    let before = hash_world(&world).unwrap();
    assert_eq!(
        world.split_assembly(proxy),
        Err(AssemblyWorldError::PendingMemberWrenches)
    );
    assert_eq!(hash_world(&world).unwrap(), before);
    assert!(world.cancel_assembly_member_wrenches(proxy));
    assert!(!world.cancel_assembly_member_wrenches(proxy));
    let restored = world.split_assembly(proxy).unwrap();
    let replacement = world.merge_assembly(&restored).unwrap();
    assert_ne!(replacement, proxy);
    let before = hash_world(&world).unwrap();
    assert_eq!(
        world.apply_assembly_member_wrenches(proxy, &opposed(members, 10.0)),
        Err(AssemblyStressError::InvalidProxy)
    );
    assert_eq!(
        world.apply_assembly_member_wrenches(replacement, &opposed(members, 10.0)),
        Err(AssemblyStressError::UnknownMember)
    );
    assert_eq!(hash_world(&world).unwrap(), before);
}

#[test]
fn cycles_are_rejected_without_inventing_a_stiffness_distribution() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let members = [Vec3::ZERO, Vec3::X, Vec3::Y].map(|p| world.spawn(BodyDesc::dynamic_at(p)));
    for (a, b) in [(0, 1), (1, 2), (2, 0)] {
        weld(
            &mut world,
            members[a],
            members[b],
            Vec3::ZERO,
            BreakLimit::new(1.0, 1.0),
        );
    }
    let proxy = world.merge_assembly(&members).unwrap();
    let before = hash_world(&world).unwrap();
    assert_eq!(
        world.apply_assembly_member_wrenches(
            proxy,
            &[wrench(members[0], Vec3::X * 100.0, Vec3::ZERO),]
        ),
        Err(AssemblyStressError::UnsupportedTopology)
    );
    assert_eq!(hash_world(&world).unwrap(), before);
}

#[test]
fn nonfixed_compliant_disabled_or_inconsistent_welds_are_rejected_atomically() {
    for case in 0..4 {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let members = [-1.0, 1.0].map(|x| world.spawn(BodyDesc::dynamic_at(Vec3::X * x)));
        let mut desc = JointDesc::fixed(
            JointAnchor::at_point(members[0], Vec3::X),
            JointAnchor::at_point(members[1], -Vec3::X),
        );
        let expected = match case {
            0 => {
                desc.kind = JointKind::Distance(DistanceJoint::rigid(0.0));
                AssemblyStressError::UnsupportedJoint
            }
            1 => {
                desc.kind =
                    JointKind::Fixed(prism_physics_core::joint::FixedJoint { compliance: 0.1 });
                AssemblyStressError::UnsupportedJoint
            }
            2 => {
                desc.enabled = false;
                AssemblyStressError::UnsupportedJoint
            }
            _ => {
                desc.anchor_a.local_point = Vec3::ZERO;
                AssemblyStressError::InvalidGeometry
            }
        };
        let joint = world.joints.insert(desc);
        world
            .connect_assembly(
                members[0],
                members[1],
                Some(joint),
                desc.kind,
                Some(BreakLimit::new(1.0, 1.0)),
            )
            .unwrap();
        let proxy = world.merge_assembly(&members).unwrap();
        let before = hash_world(&world).unwrap();
        assert_eq!(
            world.apply_assembly_member_wrenches(proxy, &opposed(members, 10.0)),
            Err(expected)
        );
        assert_eq!(hash_world(&world).unwrap(), before);
    }
}

#[test]
fn free_body_and_custom_backend_boundaries_consume_the_same_real_member_reactions() {
    let (mut free_body, members, proxy, _) = pair(BreakLimit::new(10.0, 1000.0));
    free_body
        .apply_assembly_member_wrenches(proxy, &opposed(members, 20.0))
        .unwrap();
    let mut custom = free_body.clone();
    assert_eq!(free_body.step_external_dynamics(0.01), 1);
    assert!(!free_body.bodies.contains(proxy));
    assert_eq!(custom.consume_external_forces(0.01), 1);
    let events = custom.submit_joint_loads(0.01, &[]).unwrap();
    assert_eq!(events.len(), 1);
    near(events[0].force * Vec3::X, Vec3::X * 20.0);
    assert!(!custom.bodies.contains(proxy));
    assert!(custom.submit_joint_loads(0.01, &[]).unwrap().is_empty());
    assert_eq!(custom.drain_assembly_split_reports().len(), 1);
}

#[test]
fn surviving_component_can_be_explicitly_remerged_and_fracture_with_fresh_member_generations() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let members = [-1.0, 0.0, 1.0].map(|x| world.spawn(BodyDesc::dynamic_at(Vec3::X * x)));
    let weak = weld(
        &mut world,
        members[0],
        members[1],
        -Vec3::X * 0.5,
        BreakLimit::new(10.0, 1000.0),
    );
    let strong = weld(
        &mut world,
        members[1],
        members[2],
        Vec3::X * 0.5,
        BreakLimit::new(100.0, 1000.0),
    );
    let proxy = world.merge_assembly(&members).unwrap();
    world
        .apply_assembly_member_wrenches(
            proxy,
            &[
                wrench(members[0], -Vec3::X * 12.0, Vec3::ZERO),
                wrench(members[2], Vec3::X * 12.0, Vec3::ZERO),
            ],
        )
        .unwrap();
    XpbdSolver::new().step(&mut world, 0.01, 1);
    let reports = world.drain_assembly_split_reports();
    assert_eq!(reports.len(), 1);
    let current = members.map(|original| {
        reports[0]
            .members
            .iter()
            .find(|entry| entry.original == original)
            .unwrap()
            .current
    });
    assert!(!world.joints.get(weak).unwrap().enabled);
    assert!(world.joints.get(strong).unwrap().enabled);

    // Re-baking is an explicit existing merge operation, not an automatic
    // fracture policy. Its member identity must be the fresh split handles.
    let rebaked = world.merge_assembly(&current[1..]).unwrap();
    assert_ne!(rebaked, proxy);
    let before = hash_world(&world).unwrap();
    assert_eq!(
        world.apply_assembly_member_wrenches(
            rebaked,
            &[
                wrench(members[1], -Vec3::X * 120.0, Vec3::ZERO),
                wrench(members[2], Vec3::X * 120.0, Vec3::ZERO),
            ]
        ),
        Err(AssemblyStressError::UnknownMember)
    );
    assert_eq!(hash_world(&world).unwrap(), before);
    let reactions = world
        .apply_assembly_member_wrenches(
            rebaked,
            &[
                wrench(current[1], -Vec3::X * 120.0, Vec3::ZERO),
                wrench(current[2], Vec3::X * 120.0, Vec3::ZERO),
            ],
        )
        .unwrap();
    assert_eq!(reactions.len(), 1);
    assert_eq!(reactions[0].joint, strong);
    near(reactions[0].force, Vec3::X * 120.0);
    XpbdSolver::new().step(&mut world, 0.01, 1);
    assert!(world.joint_boundary_failure().is_none());
    assert!(world.bodies.contains(current[0]));
    assert!(!world.bodies.contains(rebaked));
    assert!(!world.joints.get(strong).unwrap().enabled);
    assert_eq!(world.assemblies.connections().count(), 0);
    let second = world.drain_assembly_split_reports();
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].proxy, rebaked);
    assert_eq!(second[0].members.len(), 2);
    for entry in &second[0].members {
        assert!(current[1..].contains(&entry.original));
        assert!(!current.contains(&entry.current));
        near(
            world.bodies.linear_velocity(entry.current).unwrap(),
            Vec3::ZERO,
        );
    }
}

#[test]
fn custom_backend_failed_submission_preserves_measured_reactions_for_exact_retry() {
    let (mut world, members, proxy, _) = pair(BreakLimit::new(9.0, 1000.0));
    world
        .apply_assembly_member_wrenches(proxy, &[wrench(members[1], Vec3::X * 20.0, Vec3::ZERO)])
        .unwrap();
    assert_eq!(world.consume_external_forces(0.01), 1);
    assert!(world.assemblies.remove_body(proxy));
    let before = hash_world(&world).unwrap();
    for _ in 0..2 {
        assert!(world.submit_joint_loads(0.01, &[]).is_err());
        assert_eq!(hash_world(&world).unwrap(), before);
        near(world.bodies.linear_velocity(proxy).unwrap(), Vec3::X * 0.1);
    }
    world.assemblies.add_body(proxy);
    let events = world.submit_joint_loads(0.01, &[]).unwrap();
    assert_eq!(events.len(), 1);
    let reports = world.drain_assembly_split_reports();
    assert_eq!(reports.len(), 1);
    for member in &reports[0].members {
        near(
            world.bodies.linear_velocity(member.current).unwrap(),
            Vec3::X * 0.1,
        );
    }
    assert!(world.submit_joint_loads(0.01, &[]).unwrap().is_empty());
    assert!(world.drain_assembly_split_reports().is_empty());
}
