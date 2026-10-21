//! Compare batched scheduling to the original one-scratch-per-island order.

use super::*;
use crate::collide::contact::{ContactManifold, ContactPoint};
use crate::joint::{
    DistanceJoint, JointAnchor, JointDesc, JointStorage, SwingTwistConfig, SwingTwistJoint,
};
use crate::sleep::SleepConfig;
use crate::state::body::{BodyDesc, MassProperties};
use crate::state::storage::BodyStorage;

fn scene() -> (
    BodyStorage,
    JointStorage,
    Vec<RagdollConstraint>,
    Vec<ContactManifold>,
) {
    const SMALL: usize = 137;
    let mut bodies = BodyStorage::new();
    let ground = bodies.insert(BodyDesc::static_at(Vec3::ZERO));
    let mut kinematic_desc = BodyDesc::static_at(Vec3::Y * 0.2);
    kinematic_desc.kind = BodyKind::Kinematic;
    kinematic_desc.linear_velocity = Vec3::new(0.3, 0.0, 0.1);
    let kinematic = bodies.insert(kinematic_desc);
    let free = bodies.insert(BodyDesc::dynamic_at(Vec3::new(-10.0, 4.0, 0.0)));
    assert!(bodies.set_prev_position(free, Vec3::new(-10.1, 4.1, 0.0)));
    let mut pairs = Vec::new();
    // Interleaving slots prevents a batch from accidentally relying on
    // contiguous global bodies or a particular local static-body index.
    for layer in 0..2 {
        for i in 0..SMALL {
            let mut desc = BodyDesc::dynamic_at(Vec3::new(i as f32 * 4.0, 0.4 + layer as f32, 0.0))
                .with_mass_properties(MassProperties {
                    inv_mass: 1.0,
                    inv_inertia: Vec3::ONE,
                });
            desc.orientation = Quat::from_rotation_z(0.07 * (i % 7) as f32);
            desc.linear_velocity = Vec3::new(0.2, -0.3, 0.1);
            desc.angular_velocity = Vec3::new(-0.1, 0.2, 0.3);
            let handle = bodies.insert(desc);
            bodies.set_gravity_override(handle, Some(Vec3::new(0.0, -(i as f32 % 5.0), 0.0)));
            if layer == 0 {
                pairs.push([handle, BodyHandle::INVALID]);
            } else {
                pairs[i][1] = handle;
            }
        }
    }
    let mut joints = JointStorage::new();
    let mut ragdolls = Vec::new();
    let mut manifolds = Vec::new();
    let swing = SwingTwistJoint::try_new(SwingTwistConfig::symmetric(
        Vec3::X,
        Vec3::X,
        Vec3::Y,
        Vec3::Y,
        0.5,
        0.4,
    ))
    .unwrap();
    for (i, &[a, b]) in pairs.iter().enumerate() {
        joints.insert(JointDesc::distance(
            ground,
            Vec3::X * (i as f32 * 4.0),
            a,
            Vec3::ZERO,
            DistanceJoint::rigid(0.5).with_compliance(1.0e-5),
        ));
        joints.insert(JointDesc::spherical(
            JointAnchor::at_point(a, Vec3::Y * 0.4),
            JointAnchor::at_point(b, -Vec3::Y * 0.4),
        ));
        if i % 3 == 0 {
            ragdolls.push(RagdollConstraint {
                body_a: a,
                body_b: b,
                local_point_a: Vec3::Y * 0.4,
                local_point_b: -Vec3::Y * 0.4,
                joint: swing,
            });
        }
        for (body, separator) in [(a, ground), (b, kinematic)] {
            let p = bodies.position(body).unwrap();
            let mut contact = ContactManifold::new(separator, body, Vec3::Y);
            contact.push(ContactPoint::new(p, p - Vec3::Y * 0.05, 0.05));
            manifolds.push(contact);
        }
    }
    // A large island between batches must remain whole, not be split to meet
    // the small-island budget or force its neighbours into an oversized batch.
    let mut parent = ground;
    for i in 0..130 {
        let child = bodies.insert(BodyDesc::dynamic_at(Vec3::new(-20.0, i as f32 + 0.5, 0.0)));
        joints.insert(JointDesc::distance(
            parent,
            Vec3::ZERO,
            child,
            Vec3::ZERO,
            DistanceJoint::rigid(0.8).with_compliance(1.0e-4),
        ));
        parent = child;
    }
    let view = bodies.solver_view_mut();
    for &handle in pairs.last().unwrap() {
        view.sleeping[handle.index() as usize] = true;
        view.linear_velocities[handle.index() as usize] = Vec3::ZERO;
        view.angular_velocities[handle.index() as usize] = Vec3::ZERO;
    }
    (bodies, joints, ragdolls, manifolds)
}

#[test]
fn batch_boundaries_cover_active_once_and_isolate_large_islands() {
    let (mut bodies, mut joints, ragdolls, manifolds) = scene();
    let entries: Vec<_> = joints.active_entries_mut().collect();
    let refs: Vec<_> = entries.iter().map(|(_, j)| &**j).collect();
    let mut view = bodies.solver_view_mut();
    let constraints = ContactConstraint::build(&view, &manifolds);
    let islands = SolveIslands::build_with_ragdolls(&view, &constraints, &refs, &ragdolls);
    let mut active =
        super::super::sleep_solve::classify_and_wake(&mut view, &islands, &SleepConfig::default());
    assert_eq!(
        active.len(),
        137,
        "136 awake small islands and one large island"
    );
    // Exercise an oversized island in the middle, not just the final flush.
    let large = active.pop().unwrap();
    assert_eq!(islands.members(large).len(), 130);
    active.insert(31, large);
    let batches = scratch_batches(&islands, &active);
    assert!(batches.len() > 3 && batches.len() < active.len() / 4);
    assert_eq!(
        batches.iter().flat_map(|r| r.clone()).collect::<Vec<_>>(),
        (0..active.len()).collect::<Vec<_>>()
    );
    assert!(batches.iter().any(|r| active[r.clone()] == [large]));
    assert!(scratch_batches(&islands, &[]).is_empty());
    let all = IslandScratch::gather(
        &view,
        &islands,
        &active[..8],
        &constraints,
        &entries,
        &ragdolls,
    );
    assert_eq!(all.slots.iter().filter(|&&slot| slot == 0).count(), 1);
    assert_eq!(all.slots.iter().filter(|&&slot| slot == 1).count(), 1);
    for (local, &global) in all.slots.iter().enumerate() {
        assert_eq!(all.gravity_overrides[local], view.gravity_overrides[global]);
    }
}

fn run(
    batched: bool,
    colored: bool,
    threads: bool,
    iterations: u32,
) -> (
    BodyStorage,
    JointStorage,
    Vec<crate::dynamics::assembly::SolverJointLoad>,
) {
    let (mut bodies, mut joints, ragdolls, manifolds) = scene();
    let mut entries: Vec<_> = joints.active_entries_mut().collect();
    let refs: Vec<_> = entries.iter().map(|(_, j)| &**j).collect();
    let mut view = bodies.solver_view_mut();
    let constraints = ContactConstraint::build(&view, &manifolds);
    let islands = SolveIslands::build_with_ragdolls(&view, &constraints, &refs, &ragdolls);
    let active =
        super::super::sleep_solve::classify_and_wake(&mut view, &islands, &SleepConfig::default());
    let config = XpbdConfig {
        parallel_within_island: colored,
        position_iterations: iterations,
        ..XpbdConfig::default()
    };
    let h = 1.0 / 60.0;
    let loads = if batched {
        solve_islands_parallel(
            &mut view,
            &islands,
            &active,
            &constraints,
            &mut entries,
            &ragdolls,
            &config,
            h,
            threads,
        )
    } else {
        // Independent reference schedule: one allocation group per original
        // island, with local solve then scatter in unchanged island order.
        let mut loads = JointLoadAccumulator::new();
        let mut in_island = vec![false; view.slot_count()];
        for &island in &active {
            let mut scratch = IslandScratch::gather(
                &view,
                &islands,
                &[island],
                &constraints,
                &entries,
                &ragdolls,
            );
            loads.merge(&scratch.solve(&config, h)).unwrap();
            scratch.scatter(&mut view, &mut entries);
            for &slot in islands.members(island) {
                in_island[slot] = true;
            }
        }
        integrate::recover_velocities_excluding(&mut view, h, &in_island);
        loads
    }
    .loads()
    .unwrap();
    drop(entries);
    (bodies, joints, loads)
}

#[test]
fn batches_preserve_body_bits_joint_state_and_reactions_with_shared_separators() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for colored in [false, true] {
        for iterations in [1, 4] {
            let (expected_bodies, expected_joints, expected_loads) =
                run(false, colored, false, iterations);
            for threads in [false, true] {
                let (mut actual_bodies, actual_joints, actual_loads) =
                    pool.install(|| run(true, colored, threads, iterations));
                let mut expected = expected_bodies.clone();
                let left = expected.solver_view_mut();
                let right = actual_bodies.solver_view_mut();
                for (a, b) in left
                    .positions
                    .iter()
                    .chain(left.prev_positions.iter())
                    .chain(left.linear_velocities.iter())
                    .chain(left.angular_velocities.iter())
                    .zip(
                        right
                            .positions
                            .iter()
                            .chain(right.prev_positions.iter())
                            .chain(right.linear_velocities.iter())
                            .chain(right.angular_velocities.iter()),
                    )
                {
                    assert_eq!(
                        a.to_array().map(f32::to_bits),
                        b.to_array().map(f32::to_bits)
                    );
                }
                for (a, b) in left
                    .orientations
                    .iter()
                    .chain(left.prev_orientations.iter())
                    .zip(
                        right
                            .orientations
                            .iter()
                            .chain(right.prev_orientations.iter()),
                    )
                {
                    assert_eq!(
                        a.to_array().map(f32::to_bits),
                        b.to_array().map(f32::to_bits)
                    );
                }
                assert_eq!(left.sleeping, right.sleeping);
                assert_eq!(left.sleep_timers, right.sleep_timers);
                assert_eq!(left.gravity_overrides, right.gravity_overrides);
                assert!(expected_joints
                    .active_entries()
                    .eq(actual_joints.active_entries()));
                tests::assert_reaction_bits_equal(&expected_loads, &actual_loads);
                assert!(actual_loads
                    .iter()
                    .any(|load| load.load.linear_impulse.length() > 0.0));
            }
        }
    }
}
