//! Once-per-substep explicit generalized-force actuation.
//!
//! All feedback samples precede every actuator write. Impulses are applied
//! before prediction, so both serial and scratch-island velocity recovery see
//! the resulting displacement. This pass is never called by an iteration.

use super::rigid::{inv_inertia_world, inv_mass_world};
use crate::dynamics::assembly::JointLoadAccumulator;
use crate::joint::revolute::unit_axis;
use crate::joint::{BoundedPdFailure, Joint, JointHandle, JointKind, JointStorage};
use crate::state::storage::BodyStorage;
use crate::state::view::BodySolverView;
use glam::{DVec3, Quat, Vec3};
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;

struct Coordinate {
    q: f64,
    linear: [DVec3; 2],
    angular: [DVec3; 2],
    // Side-A pure torque only. Lever torque belongs to the linear load channel.
    pure_angular_a: DVec3,
}

struct Prepared {
    handle: JointHandle,
    slots: [usize; 2],
    linear: [Vec3; 2],
    angular: [Vec3; 2],
    load_linear: Vec3,
    load_angular: Vec3,
    active: bool,
}

/// Disposable solver-owned work buffers. No body/joint references survive a
/// call, and generation validation is rebuilt even when capacity is reused.
#[derive(Default)]
pub(super) struct Scratch {
    valid: Vec<(JointHandle, bool)>,
    prepared: Vec<Prepared>,
    activity: Vec<bool>,
}

/// Exclusive scratch loans keep the existing shared-reference stepping APIs
/// Send + Sync. The mutex is held only while taking/returning the buffers, never
/// during simulation or caller-owned contact generation. A concurrent or
/// reentrant step receives independent scratch rather than sharing feedback.
pub(super) struct ScratchCache(Mutex<Option<Scratch>>);

impl Default for ScratchCache {
    fn default() -> Self {
        Self(Mutex::new(Some(Scratch::default())))
    }
}

impl Clone for ScratchCache {
    fn clone(&self) -> Self {
        // Cloning solver configuration must not copy stale work or its heap.
        Self::default()
    }
}

impl core::fmt::Debug for ScratchCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("ScratchCache")
    }
}

impl ScratchCache {
    pub(super) fn checkout(&self) -> ScratchLoan<'_> {
        let scratch = self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .unwrap_or_default();
        ScratchLoan {
            cache: self,
            scratch,
        }
    }
}

pub(super) struct ScratchLoan<'a> {
    cache: &'a ScratchCache,
    scratch: Scratch,
}

impl Deref for ScratchLoan<'_> {
    type Target = Scratch;
    fn deref(&self) -> &Scratch {
        &self.scratch
    }
}

impl DerefMut for ScratchLoan<'_> {
    fn deref_mut(&mut self) -> &mut Scratch {
        &mut self.scratch
    }
}

impl Drop for ScratchLoan<'_> {
    fn drop(&mut self) {
        // Clear lengths, not capacity. Unwinding a callback cannot leak old
        // activity into another world or make the scratch cache unavailable.
        self.scratch.valid.clear();
        self.scratch.prepared.clear();
        self.scratch.activity.clear();
        let mut cached = self.cache.0.lock().unwrap_or_else(|e| e.into_inner());
        if cached.is_none() {
            *cached = Some(core::mem::take(&mut self.scratch));
        }
    }
}

/// Returns slots with nonzero accepted actuation, for pre-predict wake
/// propagation and post-solve sleep inhibition. Stale handles are checked
/// against storage generations before borrowing its slot-only solver view.
pub(super) fn apply<'a>(
    bodies: &mut BodyStorage,
    joints: &mut JointStorage,
    h: f32,
    loads: &mut JointLoadAccumulator,
    scratch: &'a mut Scratch,
) -> &'a [bool] {
    let Scratch {
        valid,
        prepared,
        activity,
    } = scratch;
    valid.clear();
    prepared.clear();
    activity.clear();
    // The ordinary (no explicit drive) solve must not allocate a body-sized
    // activity mask, endpoint list, or preparation buffer every substep.
    let mut has_drive = false;
    for (_, joint) in joints.active_entries_mut() {
        if joint.bounded_pd.is_some() {
            has_drive = true;
        } else {
            joint.bounded_pd_failure = None;
        }
    }
    if !has_drive {
        return activity;
    }
    valid.extend(
        joints
            .active_entries()
            .filter(|(_, joint)| joint.bounded_pd.is_some())
            .map(|(handle, joint)| {
                (
                    handle,
                    joint.anchor_a.body != joint.anchor_b.body
                        && bodies.contains(joint.anchor_a.body)
                        && bodies.contains(joint.anchor_b.body),
                )
            }),
    );
    let view = bodies.solver_view_mut();
    activity.resize(view.slot_count(), false);
    for ((handle, joint), (validated_handle, valid)) in joints
        .active_entries_mut()
        .filter(|(_, joint)| joint.bounded_pd.is_some())
        .zip(valid.iter().copied())
    {
        debug_assert_eq!(handle, validated_handle);
        let Some(policy) = joint.bounded_pd else {
            joint.bounded_pd_failure = None;
            continue;
        };
        let sample = if valid {
            prepare(&view, handle, joint, policy, h)
        } else {
            Err(BoundedPdFailure::InvalidEndpoints)
        };
        match sample {
            Ok(sample) => {
                joint.bounded_pd_failure = None;
                prepared.push(sample);
            }
            Err(error) => joint.bounded_pd_failure = Some(error),
        }
    }
    // Feedback above used exactly the same pre-actuation velocities, even
    // when several motors share a body. Validate both endpoints before any
    // write, including overflow caused by the sum of accepted impulses.
    for sample in prepared.drain(..) {
        let mut linear = [Vec3::ZERO; 2];
        let mut angular = [Vec3::ZERO; 2];
        let mut finite = true;
        for side in 0..2 {
            let slot = sample.slots[side];
            linear[side] = view.linear_velocities[slot] + sample.linear[side];
            angular[side] = view.angular_velocities[slot] + sample.angular[side];
            finite &= linear[side].is_finite() && angular[side].is_finite();
        }
        if !finite {
            joints
                .get_mut(sample.handle)
                .expect("live sampled joint")
                .bounded_pd_failure = Some(BoundedPdFailure::NonFiniteImpulse);
            continue;
        }
        if sample.active {
            loads
                .record_impulse(sample.handle, sample.load_linear, sample.load_angular, h)
                .expect("prepared finite actuator loads share the substep dt");
        }
        for side in 0..2 {
            let slot = sample.slots[side];
            if view.is_dynamic(slot) {
                view.linear_velocities[slot] = linear[side];
                view.angular_velocities[slot] = angular[side];
                if sample.active {
                    activity[slot] = true;
                    view.sleeping[slot] = false;
                    view.sleep_timers[slot] = 0.0;
                }
            }
        }
    }
    activity
}

fn prepare(
    view: &BodySolverView<'_>,
    handle: JointHandle,
    joint: &mut Joint,
    policy: crate::joint::BoundedPd,
    h: f32,
) -> Result<Prepared, BoundedPdFailure> {
    let slots = [
        joint.anchor_a.body.index() as usize,
        joint.anchor_b.body.index() as usize,
    ];
    let coordinate = coordinate(view, joint)?;
    let mut speed = 0.0;
    for (side, &slot) in slots.iter().enumerate() {
        speed += coordinate.linear[side].dot(view.linear_velocities[slot].as_dvec3())
            + coordinate.angular[side].dot(view.angular_velocities[slot].as_dvec3());
    }
    let effort = policy
        .effort(coordinate.q, speed)
        .map_err(BoundedPdFailure::Sample)?;
    let impulse = effort * f64::from(h);
    if !h.is_finite() || h <= 0.0 || !impulse.is_finite() {
        return Err(BoundedPdFailure::NonFiniteImpulse);
    }
    let mut linear = [Vec3::ZERO; 2];
    let mut angular = [Vec3::ZERO; 2];
    for (side, &slot) in slots.iter().enumerate() {
        if !view.is_dynamic(slot) {
            continue;
        }
        let mass = view.mass_props[slot];
        let orientation = view.orientations[slot];
        linear[side] = (inv_mass_world(mass.inv_mass, orientation).as_dmat3()
            * (coordinate.linear[side] * impulse))
            .as_vec3();
        angular[side] = (inv_inertia_world(mass.inv_inertia, orientation).as_dmat3()
            * (coordinate.angular[side] * impulse))
            .as_vec3();
    }
    let load_linear = (coordinate.linear[0] * impulse).as_vec3();
    let load_angular = (coordinate.pure_angular_a * impulse).as_vec3();
    if !linear
        .into_iter()
        .chain(angular)
        .chain([load_linear, load_angular])
        .all(Vec3::is_finite)
    {
        return Err(BoundedPdFailure::NonFiniteImpulse);
    }
    Ok(Prepared {
        handle,
        slots,
        linear,
        angular,
        load_linear,
        load_angular,
        active: impulse != 0.0 && slots.into_iter().any(|slot| view.is_dynamic(slot)),
    })
}

fn coordinate(
    view: &BodySolverView<'_>,
    joint: &mut Joint,
) -> Result<Coordinate, BoundedPdFailure> {
    let a = joint.anchor_a;
    let b = joint.anchor_b;
    let slots = [a.body.index() as usize, b.body.index() as usize];
    let orientations = [view.orientations[slots[0]], view.orientations[slots[1]]];
    if !orientations
        .into_iter()
        .chain([a.local_frame, b.local_frame])
        .all(unit_frame)
        || ![
            a.local_point,
            b.local_point,
            view.positions[slots[0]],
            view.positions[slots[1]],
        ]
        .into_iter()
        .all(Vec3::is_finite)
    {
        return Err(BoundedPdFailure::InvalidGeometry);
    }
    match joint.kind {
        JointKind::Revolute(params) => {
            if params.motor.is_some() {
                return Err(BoundedPdFailure::ConflictingMotor);
            }
            let gradient = joint
                .revolute_state
                .observe(params.axis, a, b, orientations)
                .ok_or(BoundedPdFailure::InvalidGeometry)?
                .as_dvec3();
            Ok(Coordinate {
                q: joint
                    .revolute_state
                    .angle()
                    .ok_or(BoundedPdFailure::InvalidGeometry)?,
                linear: [DVec3::ZERO; 2],
                angular: [-gradient, gradient],
                pure_angular_a: -gradient,
            })
        }
        JointKind::Prismatic(params) => {
            if params.motor.is_some() {
                return Err(BoundedPdFailure::ConflictingMotor);
            }
            let axis = unit_axis(params.axis).ok_or(BoundedPdFailure::InvalidGeometry)?;
            let n = ((orientations[0] * a.local_frame).normalize() * axis).as_dvec3();
            let ra = (orientations[0] * a.local_point).as_dvec3();
            let rb = (orientations[1] * b.local_point).as_dvec3();
            let d =
                view.positions[slots[1]].as_dvec3() + rb - view.positions[slots[0]].as_dvec3() - ra;
            // q = (pB - pA) dot nA. Rotating A rotates nA as well as pA.
            let transport = n.cross(d);
            Ok(Coordinate {
                q: d.dot(n),
                linear: [-n, n],
                angular: [-ra.cross(n) + transport, rb.cross(n)],
                pure_angular_a: transport,
            })
        }
        _ => Err(BoundedPdFailure::UnsupportedFamily),
    }
}

fn unit_frame(q: Quat) -> bool {
    q.is_finite() && (q.length_squared() - 1.0).abs() <= 1.0e-4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::{BoundedPd, JointAnchor, JointDesc, PrismaticJoint, RevoluteJoint};
    use crate::state::body::{BodyDesc, SolverMassProperties};

    fn slider(
        a: crate::state::handle::BodyHandle,
        b: crate::state::handle::BodyHandle,
    ) -> JointDesc {
        JointDesc::new(
            JointAnchor::at_center(a),
            JointAnchor::at_center(b),
            JointKind::Prismatic(PrismaticJoint::new(Vec3::X)),
        )
    }

    fn near(a: Vec3, b: Vec3, epsilon: f32) {
        assert!((a - b).length() < epsilon, "{a:?} != {b:?}");
    }

    #[test]
    fn no_pd_returns_an_unallocated_activity_mask_and_preserves_state() {
        let mut bodies = BodyStorage::new();
        let a = bodies.insert(BodyDesc::static_at(Vec3::ZERO));
        let b = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut joints = JointStorage::new();
        let handle = joints.insert(slider(a, b));
        joints.get_mut(handle).unwrap().bounded_pd_failure =
            Some(BoundedPdFailure::InvalidGeometry);
        let mut loads = JointLoadAccumulator::new();
        let mut scratch = Scratch::default();
        let activity = apply(&mut bodies, &mut joints, 0.01, &mut loads, &mut scratch);
        assert!(activity.is_empty());
        assert_eq!(scratch.valid.capacity(), 0);
        assert_eq!(scratch.prepared.capacity(), 0);
        assert_eq!(scratch.activity.capacity(), 0);
        assert!(loads.is_empty());
        assert_eq!(bodies.linear_velocity(b), Some(Vec3::ZERO));
        assert_eq!(joints.get(handle).unwrap().bounded_pd_failure, None);
    }

    #[test]
    fn shared_body_feedback_is_frozen_before_any_impulse() {
        let mut bodies = BodyStorage::new();
        let a = bodies.insert(BodyDesc::static_at(Vec3::ZERO));
        let b = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut joints = JointStorage::new();
        let policy = BoundedPd::new(0.0, 1.0, 0.0, 1.0, 10.0).unwrap();
        joints.insert(slider(a, b).with_bounded_pd(policy));
        joints.insert(slider(a, b).with_bounded_pd(policy));
        let mut loads = JointLoadAccumulator::new();
        let mut scratch = Scratch::default();
        let activity = apply(&mut bodies, &mut joints, 0.25, &mut loads, &mut scratch);
        near(bodies.linear_velocity(b).unwrap(), Vec3::X * 0.5, 1.0e-7);
        assert!(activity[b.index() as usize]);
        for entry in loads.loads().unwrap() {
            near(entry.load.linear_impulse, -Vec3::X * 0.25, 1.0e-7);
        }
    }

    #[test]
    fn off_manifold_slider_uses_rotated_effective_mass_and_pure_break_torque() {
        let mut bodies = BodyStorage::new();
        let a = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let b = bodies.insert(BodyDesc::dynamic_at(Vec3::new(2.0, 3.0, 0.0)));
        bodies.set_orientation(a, Quat::from_rotation_z(core::f32::consts::FRAC_PI_2));
        bodies.set_solver_mass_properties(
            a,
            SolverMassProperties {
                inv_mass: Vec3::new(2.0, 3.0, 4.0),
                inv_inertia: Vec3::new(5.0, 6.0, 7.0),
            },
        );
        let mut desc = slider(a, b);
        desc.anchor_a.local_point = Vec3::X;
        desc.anchor_b.local_point = Vec3::Z;
        let mut joints = JointStorage::new();
        let handle =
            joints.insert(desc.with_bounded_pd(BoundedPd::new(100.0, 0.0, 1.0, 0.0, 2.0).unwrap()));
        let mut loads = JointLoadAccumulator::new();
        apply(
            &mut bodies,
            &mut joints,
            0.25,
            &mut loads,
            &mut Scratch::default(),
        );
        // n = +Y, ra = +Y, rb = +Z, d = (2,2,1).
        // n x d = (1,0,-2); rb x n = -X; impulse = 0.5.
        near(bodies.linear_velocity(a).unwrap(), -Vec3::Y, 2.0e-6);
        near(bodies.linear_velocity(b).unwrap(), Vec3::Y * 0.5, 2.0e-6);
        near(
            bodies.angular_velocity(a).unwrap(),
            Vec3::new(3.0, 0.0, -7.0),
            3.0e-6,
        );
        near(bodies.angular_velocity(b).unwrap(), -Vec3::X * 0.5, 2.0e-6);
        let measured = loads.loads().unwrap();
        assert_eq!(measured[0].joint, handle);
        near(measured[0].load.linear_impulse, -Vec3::Y * 0.5, 2.0e-6);
        near(
            measured[0].load.angular_impulse,
            Vec3::new(0.5, 0.0, -1.0),
            2.0e-6,
        );
    }

    #[test]
    fn scalar_jacobians_match_world_pose_finite_differences() {
        for hinge in [false, true] {
            let mut bodies = BodyStorage::new();
            let a = bodies.insert(BodyDesc::dynamic_at(Vec3::new(-0.3, 0.2, 0.6)));
            let b = bodies.insert(BodyDesc::dynamic_at(Vec3::new(0.7, -0.4, 0.5)));
            let qa = Quat::from_rotation_x(0.4) * Quat::from_rotation_y(-0.3);
            let qb = Quat::from_rotation_z(0.7) * Quat::from_rotation_x(-0.2);
            bodies.set_orientation(a, qa);
            bodies.set_orientation(b, qb);
            let mut desc = slider(a, b);
            desc.anchor_a =
                JointAnchor::new(a, Vec3::new(0.2, 0.4, -0.1), Quat::from_rotation_y(0.2));
            desc.anchor_b =
                JointAnchor::new(b, Vec3::new(-0.1, 0.2, 0.3), Quat::from_rotation_x(-0.3));
            if hinge {
                desc.kind = JointKind::Revolute(RevoluteJoint::new(Vec3::X));
            }
            let mut joints = JointStorage::new();
            let handle = joints.insert(desc);
            let original = joints.get(handle).unwrap().clone();
            let analytic = coordinate(&bodies.solver_view_mut(), &mut original.clone()).unwrap();
            let epsilon = 1.0e-3;
            for (side, handle) in [a, b].into_iter().enumerate() {
                for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                    for angular in [false, true] {
                        let pose = bodies.position(handle).unwrap();
                        let orientation = bodies.orientation(handle).unwrap();
                        let mut samples = [0.0; 2];
                        for (sample, sign) in samples.iter_mut().zip([-1.0, 1.0]) {
                            if angular {
                                bodies.set_orientation(
                                    handle,
                                    Quat::from_scaled_axis(axis * (sign * epsilon)) * orientation,
                                );
                            } else {
                                bodies.set_position(handle, pose + axis * (sign * epsilon));
                            }
                            *sample = coordinate(&bodies.solver_view_mut(), &mut original.clone())
                                .unwrap()
                                .q;
                        }
                        bodies.set_position(handle, pose);
                        bodies.set_orientation(handle, orientation);
                        let numerical = (samples[1] - samples[0]) / (2.0 * f64::from(epsilon));
                        let expected = if angular {
                            analytic.angular[side]
                        } else {
                            analytic.linear[side]
                        }
                        .dot(axis.as_dvec3());
                        assert!((numerical - expected).abs() < 3.0e-4,
                            "hinge={hinge} side={side} angular={angular}: {numerical} != {expected}");
                    }
                }
            }
        }
    }

    #[test]
    fn anchor_lever_torque_is_not_counted_twice_in_break_loads() {
        let mut bodies = BodyStorage::new();
        let a = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let b = bodies.insert(BodyDesc::dynamic_at(Vec3::Y));
        let mut desc = slider(a, b);
        desc.anchor_a.local_point = Vec3::Y;
        desc.anchor_b.local_point = Vec3::Z;
        let mut joints = JointStorage::new();
        joints.insert(desc.with_bounded_pd(BoundedPd::new(10.0, 0.0, 1.0, 0.0, 2.0).unwrap()));
        let mut loads = JointLoadAccumulator::new();
        apply(
            &mut bodies,
            &mut joints,
            0.5,
            &mut loads,
            &mut Scratch::default(),
        );
        // A receives +Z lever torque and -Y transport torque, but the
        // break channel records only -Y (the X force accounts for the lever).
        near(
            bodies.angular_velocity(a).unwrap(),
            Vec3::Z - Vec3::Y,
            1.0e-7,
        );
        near(
            loads.loads().unwrap()[0].load.angular_impulse,
            -Vec3::Y,
            1.0e-7,
        );
    }

    #[test]
    fn overflow_is_atomic_for_both_endpoints_and_records_no_load_or_wake() {
        let mut bodies = BodyStorage::new();
        let a = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let b = bodies
            .insert(BodyDesc::dynamic_at(Vec3::ZERO).with_linear_velocity(Vec3::X * f32::MAX));
        let mut joints = JointStorage::new();
        let handle = joints.insert(
            slider(a, b).with_bounded_pd(
                BoundedPd::new(
                    0.0,
                    f64::from(f32::MAX) * 2.0,
                    0.0,
                    1.0,
                    f64::from(f32::MAX),
                )
                .unwrap(),
            ),
        );
        let mut loads = JointLoadAccumulator::new();
        let mut scratch = Scratch::default();
        let active = apply(&mut bodies, &mut joints, 1.0, &mut loads, &mut scratch);
        assert_eq!(
            joints.get(handle).unwrap().bounded_pd_failure,
            Some(BoundedPdFailure::NonFiniteImpulse)
        );
        assert_eq!(bodies.linear_velocity(a), Some(Vec3::ZERO));
        assert_eq!(bodies.linear_velocity(b), Some(Vec3::X * f32::MAX));
        assert!(loads.is_empty());
        assert!(!active.iter().any(|&value| value));
    }

    #[test]
    fn stale_generation_never_actuates_the_replacement_slot() {
        let mut bodies = BodyStorage::new();
        let a = bodies.insert(BodyDesc::static_at(Vec3::ZERO));
        let b = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut joints = JointStorage::new();
        let handle = joints
            .insert(slider(a, b).with_bounded_pd(BoundedPd::new(1.0, 0.0, 1.0, 0.0, 1.0).unwrap()));
        let mut scratch = Scratch::default();
        let mut loads = JointLoadAccumulator::new();
        // Reuse scratch that has already accepted this exact joint/slot pair.
        apply(&mut bodies, &mut joints, 0.1, &mut loads, &mut scratch);
        assert!(bodies.linear_velocity(b).unwrap().x > 0.0);
        loads.clear();
        bodies.remove(b);
        let replacement = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        assert_eq!(replacement.index(), b.index());
        apply(&mut bodies, &mut joints, 0.1, &mut loads, &mut scratch);
        assert_eq!(
            joints.get(handle).unwrap().bounded_pd_failure,
            Some(BoundedPdFailure::InvalidEndpoints)
        );
        assert_eq!(bodies.linear_velocity(replacement), Some(Vec3::ZERO));
        assert!(loads.is_empty());
    }

    fn capacities(scratch: &Scratch) -> (usize, usize, usize) {
        (
            scratch.valid.capacity(),
            scratch.prepared.capacity(),
            scratch.activity.capacity(),
        )
    }

    #[test]
    fn scratch_reuses_all_buffers_after_no_drive_and_disabled_intervals() {
        let mut bodies = BodyStorage::new();
        let a = bodies.insert(BodyDesc::static_at(Vec3::ZERO));
        let b = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut joints = JointStorage::new();
        let policy = BoundedPd::new(1.0, 0.0, 2.0, 3.0, 0.0).unwrap();
        let handles: Vec<_> = (0..32)
            .map(|_| joints.insert(slider(a, b).with_bounded_pd(policy)))
            .collect();
        let mut loads = JointLoadAccumulator::new();
        let mut scratch = Scratch::default();
        apply(&mut bodies, &mut joints, 0.1, &mut loads, &mut scratch);
        let retained = capacities(&scratch);
        let pointers = (
            scratch.valid.as_ptr(),
            scratch.prepared.as_ptr(),
            scratch.activity.as_ptr(),
        );
        assert!(retained.0 >= handles.len());
        assert!(retained.1 >= handles.len());
        assert!(retained.2 >= 2);
        for _ in 0..16 {
            for &handle in &handles {
                joints.get_mut(handle).unwrap().bounded_pd = None;
            }
            assert!(apply(&mut bodies, &mut joints, 0.1, &mut loads, &mut scratch).is_empty());
            for &handle in &handles {
                let joint = joints.get_mut(handle).unwrap();
                joint.bounded_pd = Some(policy);
                joint.enabled = false;
            }
            assert!(apply(&mut bodies, &mut joints, 0.1, &mut loads, &mut scratch).is_empty());
            for &handle in &handles {
                joints.get_mut(handle).unwrap().enabled = true;
            }
            let activity = apply(&mut bodies, &mut joints, 0.1, &mut loads, &mut scratch);
            assert_eq!(activity.len(), 2);
            assert!(!activity.iter().any(|&active| active));
            assert_eq!(scratch.valid.len(), handles.len());
            assert!(scratch.prepared.is_empty());
            assert_eq!(capacities(&scratch), retained);
            assert_eq!(
                (
                    scratch.valid.as_ptr(),
                    scratch.prepared.as_ptr(),
                    scratch.activity.as_ptr()
                ),
                pointers,
            );
        }
        assert!(loads.is_empty());
    }

    #[test]
    fn reused_scratch_rebuilds_joint_generations_and_clears_activity() {
        let mut bodies = BodyStorage::new();
        let a = bodies.insert(BodyDesc::static_at(Vec3::ZERO));
        let b = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut joints = JointStorage::new();
        let old = joints
            .insert(slider(a, b).with_bounded_pd(BoundedPd::new(1.0, 0.0, 1.0, 0.0, 1.0).unwrap()));
        let mut loads = JointLoadAccumulator::new();
        let mut scratch = Scratch::default();
        assert!(apply(&mut bodies, &mut joints, 0.1, &mut loads, &mut scratch)[b.index() as usize]);
        assert!(joints.remove(old));
        let replacement = joints
            .insert(slider(a, b).with_bounded_pd(BoundedPd::new(1.0, 0.0, 1.0, 0.0, 0.0).unwrap()));
        assert_eq!(replacement.index(), old.index());
        assert_ne!(replacement, old);
        loads.clear();
        let velocity = bodies.linear_velocity(b);
        let activity = apply(&mut bodies, &mut joints, 0.1, &mut loads, &mut scratch);
        assert!(!activity.iter().any(|&active| active));
        assert_eq!(scratch.valid, vec![(replacement, true)]);
        assert_eq!(bodies.linear_velocity(b), velocity);
        assert!(loads.is_empty());
    }

    #[test]
    fn scratch_loans_are_independent_return_on_unwind_and_clone_empty() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<super::super::XpbdSolver>();
        let cache = ScratchCache::default();
        let mut outer = cache.checkout();
        outer.activity.resize(32, true);
        let retained = outer.activity.as_ptr();
        // Reentry must neither lock the simulation nor share the outer mask.
        let mut nested = cache.checkout();
        assert_eq!(capacities(&nested), (0, 0, 0));
        nested.activity.resize(64, false);
        assert_ne!(nested.activity.as_ptr(), retained);
        assert!(outer.activity.iter().all(|&active| active));
        let clone = cache.clone();
        assert_eq!(capacities(&clone.checkout()), (0, 0, 0));
        drop(outer);
        drop(nested);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut loan = cache.checkout();
            assert_eq!(loan.activity.as_ptr(), retained);
            assert!(loan.activity.is_empty());
            loan.activity.resize(32, true);
            panic!("simulate a caller callback unwinding");
        }));
        assert!(result.is_err());
        let restored = cache.checkout();
        assert!(restored.valid.is_empty());
        assert!(restored.prepared.is_empty());
        assert!(restored.activity.is_empty());
        assert_eq!(restored.activity.as_ptr(), retained);
        assert!(restored.activity.capacity() >= 32);
    }
}
