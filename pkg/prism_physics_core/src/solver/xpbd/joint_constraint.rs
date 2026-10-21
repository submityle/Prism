//! Position-level joint constraints for the XPBD solver.
//!
//! This module projects every active [`Joint`](crate::joint::Joint) once per
//! position iteration, mirroring the way
//! [`contact_constraint`](crate::solver::xpbd::contact_constraint) projects
//! contacts. Each joint family reduces to a small set of scalar or angular
//! constraints solved with the same compliant XPBD update the contact solver
//! uses:
//!
//! - a *positional* constraint drives a scalar error `c` (a distance or an
//!   offset along a direction) to zero via [`apply_positional`];
//! - an *angular* constraint drives a rotation-vector error `theta` to zero via
//!   [`apply_angular`].
//!
//! Higher-level joints compose these primitives: a hinge is a coincident-point
//! constraint plus a swing-axis alignment plus an optional twist limit and
//! motor; a slider is a perpendicular-offset lock plus a frame alignment plus an
//! optional slide limit and motor.
//!
//! # Provenance
//!
//! The compliant position update, the generalized-inverse-mass weighting, and
//! the angular-constraint algebra follow Müller et al., *Detailed Rigid Body
//! Simulation with Extended Position Based Dynamics* (2020). This file contains
//! **no Unreal Engine source or derived code**.

use super::rigid::{
    apply_angular_delta, apply_position_impulse, generalized_inverse_mass, inv_inertia_world,
    inv_mass_world,
};
use crate::dynamics::assembly::JointLoadAccumulator;
use crate::joint::kind::{DistanceJoint, JointKind, PrismaticJoint, RevoluteJoint};
use crate::joint::motor::{Motor, MotorTarget};
use crate::joint::ragdoll::RagdollConstraint;
use crate::joint::swing_twist::AngularJointInput;
use crate::joint::JointHandle;
use crate::joint::{Joint, JointStorage};
use crate::state::view::BodySolverView;
use glam::{Mat3, Quat, Vec3};

/// Threshold below which a length, an effective inverse mass, or an angle is
/// treated as numerically zero.
const SOLVE_EPS: f32 = 1.0e-9;

/// Per-joint projection multipliers accumulated during one solver pass.
///
/// XPBD's positional multiplier has impulse-times-step units, so it is
/// converted at the fixed-step boundary by [`JointLoadAccumulator`]. Keeping
/// the two channels separate prevents a lever-arm torque from being counted
/// twice as both a linear and angular joint load.
#[derive(Clone, Copy, Debug, Default)]
struct ProjectionAccum {
    linear_lambda: Vec3,
    angular_lambda: Vec3,
}

impl ProjectionAccum {
    fn record_linear(&mut self, lambda: Vec3) {
        self.linear_lambda += lambda;
    }

    fn record_angular(&mut self, lambda: Vec3) {
        self.angular_lambda += lambda;
    }

    fn is_zero(self) -> bool {
        self.linear_lambda == Vec3::ZERO && self.angular_lambda == Vec3::ZERO
    }
}

/// Begin hinge tracking before prediction. Disabled and unrestricted hinges
/// keep their coordinates. Velocity goals and torque budgets belong to one
/// substep, not to each position iteration.
pub fn begin_revolute_substep(view: &BodySolverView<'_>, joints: &mut JointStorage, h: f32) {
    for joint in joints.live_joints_mut() {
        observe_revolute(view, joint);
        joint.revolute_state.clear_substep();
        if let JointKind::Revolute(params) = joint.kind {
            if let Some(Motor {
                target: MotorTarget::Velocity(rate),
                ..
            }) = params.motor
            {
                if h.is_finite() && h > 0.0 && rate.is_finite() {
                    joint.revolute_state.motor_goal = joint
                        .revolute_state
                        .angle()
                        .map(|angle| angle + f64::from(rate) * f64::from(h));
                }
            }
        }
    }
}

/// Observe a pose-writing boundary without resetting substep motor state.
pub fn observe_revolute_joints(view: &BodySolverView<'_>, joints: &mut JointStorage) {
    for joint in joints.live_joints_mut() {
        observe_revolute(view, joint);
    }
}

/// Finish after every pose writer, including contacts and coupling. Full-world
/// boundary snapshots retain winding but no obsolete iteration multiplier.
pub fn finish_revolute_substep(view: &BodySolverView<'_>, joints: &mut JointStorage) {
    observe_revolute_joints(view, joints);
    for joint in joints.live_joints_mut() {
        joint.revolute_state.clear_substep();
    }
}

fn observe_revolute(view: &BodySolverView<'_>, joint: &mut Joint) -> Option<Vec3> {
    let JointKind::Revolute(params) = joint.kind else {
        joint.reset_revolute_state();
        return None;
    };
    let a = joint.anchor_a.body.index() as usize;
    let b = joint.anchor_b.body.index() as usize;
    if !view.is_active(a) || !view.is_active(b) {
        return None;
    }
    joint.revolute_state.observe(
        params.axis,
        joint.anchor_a,
        joint.anchor_b,
        [view.orientations[a], view.orientations[b]],
    )
}

/// Solves every active joint in `joints` against the current poses in `view`.
///
/// This runs one projection pass; the caller loops it alongside the contact
/// position solve for the configured number of position iterations. Joints that
/// reference an inactive slot, an out-of-range slot, or a pair with no dynamic
/// body are skipped.
/// Bracket a substep's iterations with [`super::gear::begin_substep`] and
/// [`super::gear::finish_substep`] to preserve gear winding and reset lambda.
/// For hinges also use [`begin_revolute_substep`] / [`finish_revolute_substep`].
pub fn solve_joints(view: &mut BodySolverView<'_>, joints: &mut JointStorage, h: f32) {
    for (_, joint) in joints.active_entries_mut() {
        let mut projection = ProjectionAccum::default();
        solve_one(view, joint, h, &mut projection);
    }
}

/// Solves a single joint against the current poses in `view`.
///
/// This is the per-joint entry point the island solver uses; it applies one
/// projection pass for the given joint, mirroring one iteration of
/// [`solve_joints`]. Joints referencing an inactive slot, an out-of-range slot,
/// or a pair with no dynamic body are skipped.
pub fn solve_joint(view: &mut BodySolverView<'_>, joint: &mut Joint, h: f32) {
    let mut projection = ProjectionAccum::default();
    solve_one(view, joint, h, &mut projection);
}

/// Solves one ordinary joint and records the exact position-level multipliers
/// generated by this pass. Callers must submit the accumulator exactly once at
/// the sub-step boundary, after all position iterations have run.
pub fn solve_joint_with_load(
    view: &mut BodySolverView<'_>,
    handle: JointHandle,
    joint: &mut Joint,
    h: f32,
    loads: &mut JointLoadAccumulator,
) {
    let mut projection = ProjectionAccum::default();
    solve_one(view, joint, h, &mut projection);
    if !projection.is_zero() {
        loads
            .record_projection(
                handle,
                projection.linear_lambda,
                projection.angular_lambda,
                h,
            )
            .expect("XPBD joint projection must produce a finite fixed-step load");
    }
}

/// Solves one live ragdoll Swing-Twist link in the XPBD position phase.
pub fn solve_ragdoll(view: &mut BodySolverView<'_>, constraint: &RagdollConstraint, h: f32) {
    let mut projection = ProjectionAccum::default();
    let slot_a = constraint.body_a.index() as usize;
    let slot_b = constraint.body_b.index() as usize;
    if !valid_pair(view, slot_a, slot_b) {
        return;
    }

    let (r_a, world_a) = world_anchor(view, slot_a, constraint.local_point_a);
    let (r_b, world_b) = world_anchor(view, slot_b, constraint.local_point_b);
    let delta = world_a - world_b;
    let distance = delta.length();
    if distance > SOLVE_EPS {
        apply_positional(
            view,
            slot_a,
            slot_b,
            r_a,
            r_b,
            delta / distance,
            distance,
            0.0,
            f32::INFINITY,
            &mut projection,
        );
    }

    let measurement = constraint
        .joint
        .measure(view.orientations[slot_a], view.orientations[slot_b]);
    if measurement.swing_violation > SOLVE_EPS {
        apply_angular(
            view,
            slot_a,
            slot_b,
            measurement.swing_correction_axis_world * measurement.swing_violation,
            alpha_tilde(constraint.joint.config().swing_response.compliance, h),
            f32::INFINITY,
            &mut projection,
        );
    }

    let measurement = constraint
        .joint
        .measure(view.orientations[slot_a], view.orientations[slot_b]);
    if measurement.twist_violation.abs() > SOLVE_EPS {
        apply_angular(
            view,
            slot_a,
            slot_b,
            measurement.twist_axis_world * measurement.twist_violation,
            alpha_tilde(constraint.joint.config().twist_response.compliance, h),
            f32::INFINITY,
            &mut projection,
        );
    }
}

/// Applies the bounded friction, soft-limit, and pose-drive response of one
/// ragdoll link after velocity recovery.
pub fn solve_ragdoll_velocity(
    view: &mut BodySolverView<'_>,
    constraint: &RagdollConstraint,
    h: f32,
) {
    let slot_a = constraint.body_a.index() as usize;
    let slot_b = constraint.body_b.index() as usize;
    if !valid_pair(view, slot_a, slot_b) {
        return;
    }
    let input = AngularJointInput {
        orientation_a: view.orientations[slot_a],
        orientation_b: view.orientations[slot_b],
        angular_velocity_a: view.angular_velocities[slot_a],
        angular_velocity_b: view.angular_velocities[slot_b],
        inv_inertia_a: if view.is_dynamic(slot_a) {
            view.mass_props[slot_a].inv_inertia
        } else {
            Vec3::ZERO
        },
        inv_inertia_b: if view.is_dynamic(slot_b) {
            view.mass_props[slot_b].inv_inertia
        } else {
            Vec3::ZERO
        },
        dt: h,
    };
    let mut angular_a = input.angular_velocity_a;
    let mut angular_b = input.angular_velocity_b;
    constraint
        .joint
        .apply_velocity_response(input, &mut angular_a, &mut angular_b);
    if view.is_dynamic(slot_a) {
        view.angular_velocities[slot_a] = angular_a;
    }
    if view.is_dynamic(slot_b) {
        view.angular_velocities[slot_b] = angular_b;
    }
}

/// Resolves a single joint's slots, guards them, and dispatches on its kind.
fn solve_one(
    view: &mut BodySolverView<'_>,
    joint: &mut Joint,
    h: f32,
    projection: &mut ProjectionAccum,
) {
    // Observation is independent of engagement and dynamic/static status.
    observe_revolute(view, joint);
    if !joint.enabled || !h.is_finite() || h <= 0.0 {
        return;
    }
    let slot_a = joint.anchor_a.body.index() as usize;
    let slot_b = joint.anchor_b.body.index() as usize;
    if slot_a >= view.slot_count() || slot_b >= view.slot_count() {
        return;
    }
    if !view.is_active(slot_a) || !view.is_active(slot_b) {
        return;
    }
    if !view.is_dynamic(slot_a) && !view.is_dynamic(slot_b) {
        return;
    }

    let alpha = alpha_tilde(joint.kind.compliance(), h);
    match joint.kind {
        JointKind::Fixed(_) => {
            solve_point(view, joint, slot_a, slot_b, alpha, projection);
            solve_frame_align(view, joint, slot_a, slot_b, alpha, projection);
        }
        JointKind::Distance(params) => {
            solve_distance(view, joint, slot_a, slot_b, params, alpha, projection)
        }
        JointKind::Spherical(_) => solve_point(view, joint, slot_a, slot_b, alpha, projection),
        JointKind::Revolute(params) => {
            solve_revolute(view, joint, slot_a, slot_b, &params, alpha, h, projection);
        }
        JointKind::Prismatic(params) => {
            solve_prismatic(view, joint, slot_a, slot_b, &params, alpha, h, projection);
        }
        JointKind::Gear(_) => {
            // A gear has independent gradients on its two ends, not the
            // equal-and-opposite gradient assumed by apply_angular.
            let load = super::gear::solve(view, joint, h);
            projection.record_angular(load);
        }
    }
}

fn valid_pair(view: &BodySolverView<'_>, slot_a: usize, slot_b: usize) -> bool {
    slot_a < view.slot_count()
        && slot_b < view.slot_count()
        && view.is_active(slot_a)
        && view.is_active(slot_b)
        && (view.is_dynamic(slot_a) || view.is_dynamic(slot_b))
}

/// Holds the two anchor points coincident (ball-and-socket / weld position).
fn solve_point(
    view: &mut BodySolverView<'_>,
    joint: &Joint,
    slot_a: usize,
    slot_b: usize,
    alpha: f32,
    projection: &mut ProjectionAccum,
) {
    let (r_a, world_a) = world_anchor(view, slot_a, joint.anchor_a.local_point);
    let (r_b, world_b) = world_anchor(view, slot_b, joint.anchor_b.local_point);
    let delta = world_a - world_b;
    let c = delta.length();
    if c <= SOLVE_EPS {
        return;
    }
    let dir = delta / c;
    apply_positional(
        view,
        slot_a,
        slot_b,
        r_a,
        r_b,
        dir,
        c,
        alpha,
        f32::INFINITY,
        projection,
    );
}

/// Keeps the anchor separation inside the distance joint's length range.
fn solve_distance(
    view: &mut BodySolverView<'_>,
    joint: &Joint,
    slot_a: usize,
    slot_b: usize,
    params: DistanceJoint,
    alpha: f32,
    projection: &mut ProjectionAccum,
) {
    let (r_a, world_a) = world_anchor(view, slot_a, joint.anchor_a.local_point);
    let (r_b, world_b) = world_anchor(view, slot_b, joint.anchor_b.local_point);
    let delta = world_a - world_b;
    let d = delta.length();
    if d <= SOLVE_EPS {
        return;
    }
    let target = d.clamp(params.min_length, params.max_length);
    let c = d - target;
    if c.abs() <= SOLVE_EPS {
        return;
    }
    let dir = delta / d;
    apply_positional(
        view,
        slot_a,
        slot_b,
        r_a,
        r_b,
        dir,
        c,
        alpha,
        f32::INFINITY,
        projection,
    );
}

/// Aligns the two anchor reference frames (weld orientation / slider lock).
fn solve_frame_align(
    view: &mut BodySolverView<'_>,
    joint: &Joint,
    slot_a: usize,
    slot_b: usize,
    alpha: f32,
    projection: &mut ProjectionAccum,
) {
    let frame_a = world_frame(view, slot_a, joint.anchor_a.local_frame);
    let frame_b = world_frame(view, slot_b, joint.anchor_b.local_frame);
    let theta = frame_error(frame_a, frame_b);
    apply_angular(
        view,
        slot_a,
        slot_b,
        theta,
        alpha,
        f32::INFINITY,
        projection,
    );
}

/// Solves a hinge: coincident anchors, aligned hinge axis, optional twist limit
/// and motor about that axis.
fn solve_revolute(
    view: &mut BodySolverView<'_>,
    joint: &mut Joint,
    slot_a: usize,
    slot_b: usize,
    params: &RevoluteJoint,
    alpha: f32,
    h: f32,
    projection: &mut ProjectionAccum,
) {
    let Some(axis_local) = crate::joint::revolute::unit_axis(params.axis) else {
        return;
    };
    let Some(gradient) = observe_revolute(view, joint) else {
        return;
    };
    // A very large off-centre point correction must not rotate both endpoints
    // through an unobserved full turn. Bound its angular displacement too.
    solve_revolute_point(view, joint, [slot_a, slot_b], alpha, gradient, projection);
    if observe_revolute(view, joint).is_none() {
        return;
    }

    // Swing: force the world-space hinge axes of the two frames to coincide.
    let frame_a = world_frame(view, slot_a, joint.anchor_a.local_frame);
    let frame_b = world_frame(view, slot_b, joint.anchor_b.local_frame);
    let axis_a = (frame_a * axis_local).normalize_or_zero();
    let axis_b = (frame_b * axis_local).normalize_or_zero();
    let swing = axis_a.cross(axis_b);
    let swing_error = swing.length();
    if swing_error > SOLVE_EPS {
        project_revolute_angle(
            view,
            [slot_a, slot_b],
            swing / swing_error,
            f64::from(swing_error),
            f64::from(alpha),
            0.0,
            f64::INFINITY,
            projection,
        );
    }
    let Some(gradient) = observe_revolute(view, joint) else {
        return;
    };
    let angle = joint.revolute_state.angle().expect("observed hinge");
    // All coordinate arithmetic stays in f64, including subtraction. Casting
    // a many-turn coordinate to f32 first would lose its fractional angle.
    let bounds = params.limit.and_then(|limit| {
        (limit.min.is_finite() && limit.max.is_finite() && limit.min <= limit.max)
            .then_some((f64::from(limit.min), f64::from(limit.max)))
    });
    if let Some((min, max)) = bounds {
        let error = angle - angle.clamp(min, max);
        if error.abs() > f64::from(SOLVE_EPS) {
            project_revolute_angle(
                view,
                [slot_a, slot_b],
                gradient,
                error,
                f64::from(alpha),
                0.0,
                f64::INFINITY,
                projection,
            );
        }
    }
    // The limit has changed the pose: never feed the pre-limit angle to a motor.
    let Some(gradient) = observe_revolute(view, joint) else {
        return;
    };
    let angle = joint.revolute_state.angle().expect("observed hinge");
    if let Some(motor) = params.motor {
        if motor.max_force.is_nan()
            || motor.max_force < 0.0
            || !motor.compliance.is_finite()
            || motor.compliance < 0.0
        {
            return;
        }
        let mut goal = match motor.target {
            MotorTarget::Position(target) => f64::from(target),
            MotorTarget::Velocity(rate) => *joint
                .revolute_state
                .motor_goal
                .get_or_insert(angle + f64::from(rate) * f64::from(h)),
        };
        if !goal.is_finite() {
            return;
        }
        // The drive respects the authored unwrapped range, including ranges
        // outside [-pi, pi]. A motor cannot fight its own hard stop.
        if let Some((min, max)) = bounds {
            goal = goal.clamp(min, max);
        }
        let delta = project_revolute_angle(
            view,
            [slot_a, slot_b],
            gradient,
            angle - goal,
            f64::from(motor.compliance) / f64::from(h).powi(2),
            joint.revolute_state.motor_lambda,
            f64::from(motor.max_force) * f64::from(h).powi(2),
            projection,
        );
        joint.revolute_state.motor_lambda += delta;
        observe_revolute(view, joint);
    }
}

/// The same anchor constraint as solve_point, with a per-observation angular
/// cap. Other joint families retain their existing projection behavior.
fn solve_revolute_point(
    view: &mut BodySolverView<'_>,
    joint: &Joint,
    slots: [usize; 2],
    alpha: f32,
    gradient: Vec3,
    projection: &mut ProjectionAccum,
) {
    let [a, b] = slots;
    let (ra, xa) = world_anchor(view, a, joint.anchor_a.local_point);
    let (rb, xb) = world_anchor(view, b, joint.anchor_b.local_point);
    let offset = xa - xb;
    let error = offset.length();
    if !error.is_finite() || error <= SOLVE_EPS {
        return;
    }
    let direction = offset / error;
    let ia = inertia_world(view, a);
    let ib = inertia_world(view, b);
    let weight = generalized_inverse_mass(inv_mass_world_for(view, a), ia, ra, direction)
        + generalized_inverse_mass(inv_mass_world_for(view, b), ib, rb, direction);
    let angular_weight = (ia.as_dmat3() * ra.cross(direction).as_dvec3())
        .length()
        .max((ib.as_dmat3() * rb.cross(direction).as_dvec3()).length());
    let cap = 0.25 / gradient.as_dvec3().length().max(1.0);
    let bounded_error = if angular_weight > 0.0 {
        f64::from(error).min(cap * f64::from(weight + alpha) / angular_weight) as f32
    } else {
        error
    };
    apply_positional(
        view,
        a,
        b,
        ra,
        rb,
        direction,
        bounded_error,
        alpha,
        f32::INFINITY,
        projection,
    );
}

/// Scalar hinge projection with an accumulated motor multiplier. Bound actual
/// endpoint rotations, not the unwrapped error; immediately observe each call.
#[expect(
    clippy::too_many_arguments,
    reason = "scalar hinge projection carries a gradient, error, compliance, accumulated multiplier and cap"
)]
fn project_revolute_angle(
    view: &mut BodySolverView<'_>,
    slots: [usize; 2],
    gradient: Vec3,
    error: f64,
    alpha: f64,
    lambda: f64,
    lambda_max: f64,
    projection: &mut ProjectionAccum,
) -> f64 {
    let [a, b] = slots;
    let g = gradient.as_dvec3();
    let da = inertia_world(view, a).as_dmat3() * g;
    let db = inertia_world(view, b).as_dmat3() * g;
    let weight = g.dot(da + db);
    if !weight.is_finite() || weight <= 0.0 || !alpha.is_finite() || alpha < 0.0 {
        return 0.0;
    }
    let mut delta = ((error - alpha * lambda) / (weight + alpha) + lambda)
        .clamp(-lambda_max, lambda_max)
        - lambda;
    let max_rotation = da.length().max(db.length());
    if max_rotation > 0.0 {
        let cap = 0.5 / (max_rotation * g.length().max(1.0));
        delta = delta.clamp(-cap, cap);
    }
    let rotation_a = (da * delta).as_vec3();
    let rotation_b = (-db * delta).as_vec3();
    let torque = (g * delta).as_vec3();
    if !delta.is_finite()
        || !rotation_a.is_finite()
        || !rotation_b.is_finite()
        || !torque.is_finite()
    {
        return 0.0;
    }
    if view.is_dynamic(a) {
        view.orientations[a] =
            (Quat::from_scaled_axis(rotation_a) * view.orientations[a]).normalize();
    }
    if view.is_dynamic(b) {
        view.orientations[b] =
            (Quat::from_scaled_axis(rotation_b) * view.orientations[b]).normalize();
    }
    projection.record_angular(torque);
    delta
}

/// Solves a slider: locks the perpendicular offset and the relative rotation,
/// with an optional slide limit and motor along the axis.
fn solve_prismatic(
    view: &mut BodySolverView<'_>,
    joint: &Joint,
    slot_a: usize,
    slot_b: usize,
    params: &PrismaticJoint,
    alpha: f32,
    h: f32,
    projection: &mut ProjectionAccum,
) {
    let axis_local = params.axis.normalize_or_zero();
    if axis_local.length_squared() <= SOLVE_EPS {
        return;
    }

    // Perpendicular lock: cancel the anchor separation orthogonal to the axis.
    let frame_a = world_frame(view, slot_a, joint.anchor_a.local_frame);
    let axis = (frame_a * axis_local).normalize_or_zero();
    let (r_a, world_a) = world_anchor(view, slot_a, joint.anchor_a.local_point);
    let (r_b, world_b) = world_anchor(view, slot_b, joint.anchor_b.local_point);
    let delta = world_a - world_b;
    let perp = delta - axis * delta.dot(axis);
    let perp_len = perp.length();
    if perp_len > SOLVE_EPS {
        let dir = perp / perp_len;
        apply_positional(
            view,
            slot_a,
            slot_b,
            r_a,
            r_b,
            dir,
            perp_len,
            alpha,
            f32::INFINITY,
            projection,
        );
    }

    // Rotation lock: a slider keeps the two frames aligned.
    solve_frame_align(view, joint, slot_a, slot_b, alpha, projection);

    if params.limit.is_none() && params.motor.is_none() {
        return;
    }

    // Slide coordinate along the (re-read) axis, with its position gradient.
    let frame_a = world_frame(view, slot_a, joint.anchor_a.local_frame);
    let axis = (frame_a * axis_local).normalize_or_zero();
    let (r_a, world_a) = world_anchor(view, slot_a, joint.anchor_a.local_point);
    let (r_b, world_b) = world_anchor(view, slot_b, joint.anchor_b.local_point);
    let slide = (world_b - world_a).dot(axis);
    let grad_a = -axis;

    if let Some(limit) = params.limit {
        let c = slide - limit.clamp(slide);
        if c.abs() > SOLVE_EPS {
            apply_positional(
                view,
                slot_a,
                slot_b,
                r_a,
                r_b,
                grad_a,
                c,
                alpha,
                f32::INFINITY,
                projection,
            );
        }
    }
    if let Some(motor) = params.motor {
        let c = motor_error(&motor, slide, h);
        let lambda_max = motor.max_force * h * h;
        let motor_alpha = alpha_tilde(motor.compliance, h);
        apply_positional(
            view,
            slot_a,
            slot_b,
            r_a,
            r_b,
            grad_a,
            c,
            motor_alpha,
            lambda_max,
            projection,
        );
    }
}

/// Applies one compliant positional correction for scalar error `c`.
///
/// `dir` is the unit gradient of `c` with respect to body `a`'s anchor
/// position; the correction moves `a` by `+p` and `b` by `-p` so that `c` is
/// driven toward zero. `lambda_max` bounds the impulse magnitude (use
/// [`f32::INFINITY`] for a hard constraint, or `max_force * h^2` for a motor).
#[expect(
    clippy::too_many_arguments,
    reason = "a positional constraint needs the body pair, both lever arms, the gradient, the error, the compliance, and the impulse cap"
)]
fn apply_positional(
    view: &mut BodySolverView<'_>,
    slot_a: usize,
    slot_b: usize,
    r_a: Vec3,
    r_b: Vec3,
    dir: Vec3,
    c: f32,
    alpha: f32,
    lambda_max: f32,
    projection: &mut ProjectionAccum,
) {
    let inv_mass_a = inv_mass_world_for(view, slot_a);
    let inv_mass_b = inv_mass_world_for(view, slot_b);
    let inertia_a = inertia_world(view, slot_a);
    let inertia_b = inertia_world(view, slot_b);

    let w_a = generalized_inverse_mass(inv_mass_a, inertia_a, r_a, dir);
    let w_b = generalized_inverse_mass(inv_mass_b, inertia_b, r_b, dir);
    let w = w_a + w_b;
    if w <= SOLVE_EPS {
        return;
    }

    let delta_lambda = (-c / (w + alpha)).clamp(-lambda_max, lambda_max);
    let p = dir * delta_lambda;
    projection.record_linear(p);

    if view.is_dynamic(slot_a) {
        let mut x = view.positions[slot_a];
        let mut q = view.orientations[slot_a];
        apply_position_impulse(&mut x, &mut q, inv_mass_a, inertia_a, r_a, p);
        view.positions[slot_a] = x;
        view.orientations[slot_a] = q;
    }
    if view.is_dynamic(slot_b) {
        let mut x = view.positions[slot_b];
        let mut q = view.orientations[slot_b];
        apply_position_impulse(&mut x, &mut q, inv_mass_b, inertia_b, r_b, -p);
        view.positions[slot_b] = x;
        view.orientations[slot_b] = q;
    }
}

/// Applies one compliant angular correction for rotation-vector error `theta`.
///
/// `theta` points along the axis about which body `b` leads body `a`, scaled by
/// the angle error; the correction rotates `a` toward `b` and `b` toward `a` so
/// the error is driven to zero. `lambda_max` bounds the impulse magnitude.
fn apply_angular(
    view: &mut BodySolverView<'_>,
    slot_a: usize,
    slot_b: usize,
    theta: Vec3,
    alpha: f32,
    lambda_max: f32,
    projection: &mut ProjectionAccum,
) {
    let c = theta.length();
    if c <= SOLVE_EPS {
        return;
    }
    let axis = theta / c;
    let inertia_a = inertia_world(view, slot_a);
    let inertia_b = inertia_world(view, slot_b);

    let w_a = if view.is_dynamic(slot_a) {
        axis.dot(inertia_a * axis)
    } else {
        0.0
    };
    let w_b = if view.is_dynamic(slot_b) {
        axis.dot(inertia_b * axis)
    } else {
        0.0
    };
    let w = w_a + w_b;
    if w <= SOLVE_EPS {
        return;
    }

    let delta_lambda = (c / (w + alpha)).clamp(-lambda_max, lambda_max);
    let p = axis * delta_lambda;
    projection.record_angular(p);

    if view.is_dynamic(slot_a) {
        let mut q = view.orientations[slot_a];
        apply_angular_delta(&mut q, inertia_a * p);
        view.orientations[slot_a] = q;
    }
    if view.is_dynamic(slot_b) {
        let mut q = view.orientations[slot_b];
        apply_angular_delta(&mut q, -(inertia_b * p));
        view.orientations[slot_b] = q;
    }
}

/// Returns the per-sub-step scalar goal a motor wants to null out.
///
/// A position motor drives the coordinate toward a fixed target; a velocity
/// motor asks for a `-v * h` increment each sub-step, which the compliant
/// projection converts into a bounded corrective impulse.
fn motor_error(motor: &Motor, current: f32, h: f32) -> f32 {
    match motor.target {
        MotorTarget::Position(target) => current - target,
        MotorTarget::Velocity(rate) => -rate * h,
    }
}

/// Converts a compliance (inverse stiffness) into the XPBD `alpha_tilde` term.
fn alpha_tilde(compliance: f32, h: f32) -> f32 {
    if h > 0.0 {
        compliance / (h * h)
    } else {
        0.0
    }
}

/// Returns the world lever arm `r` and world position of an anchor point.
fn world_anchor(view: &BodySolverView<'_>, slot: usize, local_point: Vec3) -> (Vec3, Vec3) {
    let r = view.orientations[slot] * local_point;
    (r, view.positions[slot] + r)
}

/// Returns the world-space orientation of an anchor reference frame.
fn world_frame(view: &BodySolverView<'_>, slot: usize, local_frame: Quat) -> Quat {
    view.orientations[slot] * local_frame
}

/// Returns the small-angle rotation vector that carries frame `a` onto frame
/// `b` (twice the vector part of the relative quaternion, hemisphere-aligned).
fn frame_error(a: Quat, b: Quat) -> Vec3 {
    let mut q_rel = b * a.inverse();
    if q_rel.w < 0.0 {
        q_rel = -q_rel;
    }
    2.0 * Vec3::new(q_rel.x, q_rel.y, q_rel.z)
}

/// Returns the effective inverse mass of `slot`, or `0` for non-dynamic bodies.
fn inv_mass_world_for(view: &BodySolverView<'_>, slot: usize) -> Mat3 {
    if view.is_dynamic(slot) {
        inv_mass_world(view.mass_props[slot].inv_mass, view.orientations[slot])
    } else {
        Mat3::ZERO
    }
}

/// Returns the world-space inverse inertia tensor of `slot`, or the zero tensor
/// for non-dynamic bodies (infinite inertia).
fn inertia_world(view: &BodySolverView<'_>, slot: usize) -> Mat3 {
    if view.is_dynamic(slot) {
        inv_inertia_world(view.mass_props[slot].inv_inertia, view.orientations[slot])
    } else {
        Mat3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::anchor::JointAnchor;
    use crate::joint::desc::JointDesc;
    use crate::joint::kind::{PrismaticJoint, RevoluteJoint};
    use crate::joint::ragdoll::RagdollConstraint;
    use crate::joint::{
        AngularFriction, JointFriction, PoseDrive, SwingTwistConfig, SwingTwistJoint,
    };
    use crate::state::body::BodyDesc;
    use crate::state::storage::BodyStorage;

    fn dynamic(pos: Vec3) -> BodyDesc {
        BodyDesc::dynamic_at(pos)
    }

    fn ragdoll_constraint(
        a: crate::state::handle::BodyHandle,
        b: crate::state::handle::BodyHandle,
        joint: SwingTwistJoint,
    ) -> RagdollConstraint {
        RagdollConstraint {
            body_a: a,
            body_b: b,
            local_point_a: Vec3::ZERO,
            local_point_b: Vec3::ZERO,
            joint,
        }
    }

    #[test]
    fn distance_joint_converges_to_target_length() {
        let mut s = BodyStorage::new();
        let a = s.insert(dynamic(Vec3::ZERO));
        let b = s.insert(dynamic(Vec3::new(2.0, 0.0, 0.0)));
        let mut joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::distance(
                a,
                Vec3::ZERO,
                b,
                Vec3::ZERO,
                DistanceJoint::rigid(1.0),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        for _ in 0..8 {
            solve_joints(&mut view, &mut joints, 1.0 / 60.0);
        }
        let d = (view.positions[a.index() as usize] - view.positions[b.index() as usize]).length();
        assert!(
            (d - 1.0).abs() < 1.0e-4,
            "distance settled at {d}, expected 1.0"
        );
    }

    #[test]
    fn spherical_joint_brings_anchors_together() {
        let mut s = BodyStorage::new();
        let a = s.insert(dynamic(Vec3::ZERO));
        let b = s.insert(dynamic(Vec3::new(1.0, 0.0, 0.0)));
        let mut joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::spherical(
                JointAnchor::at_center(a),
                JointAnchor::at_center(b),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        for _ in 0..8 {
            solve_joints(&mut view, &mut joints, 1.0 / 60.0);
        }
        let gap =
            (view.positions[a.index() as usize] - view.positions[b.index() as usize]).length();
        assert!(gap < 1.0e-4, "anchors did not coincide, gap = {gap}");
    }

    #[test]
    fn fixed_joint_aligns_orientation() {
        let mut s = BodyStorage::new();
        let a = s.insert(dynamic(Vec3::ZERO));
        let b = s.insert(dynamic(Vec3::ZERO));
        s.set_orientation(b, Quat::from_rotation_y(0.4));
        let mut joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::fixed(
                JointAnchor::at_center(a),
                JointAnchor::at_center(b),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        for _ in 0..40 {
            solve_joints(&mut view, &mut joints, 1.0 / 60.0);
        }
        let err = frame_error(
            view.orientations[a.index() as usize],
            view.orientations[b.index() as usize],
        )
        .length();
        assert!(err < 1.0e-2, "orientation not aligned, error = {err}");
    }

    #[test]
    fn ragdoll_position_projection_converges_and_preserves_finite_pose() {
        let mut storage = BodyStorage::new();
        let parent = storage.insert(dynamic(Vec3::ZERO));
        let child = storage.insert(dynamic(Vec3::ZERO));
        storage.set_orientation(child, Quat::from_rotation_x(1.2));
        let joint = SwingTwistJoint::try_new(SwingTwistConfig::symmetric(
            Vec3::X,
            Vec3::X,
            Vec3::Y,
            Vec3::Y,
            0.5,
            0.4,
        ))
        .expect("valid swing-twist config");
        let constraint = ragdoll_constraint(parent, child, joint);
        let mut view = storage.solver_view_mut();
        for _ in 0..24 {
            solve_ragdoll(&mut view, &constraint, 1.0 / 60.0);
        }
        let measurement = joint.measure(
            view.orientations[parent.index() as usize],
            view.orientations[child.index() as usize],
        );
        assert!(measurement.twist_violation.abs() < 1.0e-3);
        assert!(measurement.swing_violation < 1.0e-3);
        assert!(view.orientations[parent.index() as usize].is_finite());
        assert!(view.orientations[child.index() as usize].is_finite());
    }

    #[test]
    fn ragdoll_velocity_response_damps_friction_and_is_bounded() {
        let mut storage = BodyStorage::new();
        let parent = storage.insert(dynamic(Vec3::ZERO));
        let child =
            storage.insert(dynamic(Vec3::ZERO).with_angular_velocity(Vec3::new(8.0, 0.0, 0.0)));
        let mut config = SwingTwistConfig::symmetric(Vec3::X, Vec3::X, Vec3::Y, Vec3::Y, 0.5, 0.5);
        config.friction = JointFriction {
            swing: AngularFriction::default(),
            twist: AngularFriction::new(0.25, 0.5, 0.1).expect("valid friction"),
        };
        config.drive = PoseDrive::new(Quat::IDENTITY, 0.0, 0.0, 0.25);
        let joint = SwingTwistJoint::try_new(config).expect("valid swing-twist config");
        let constraint = ragdoll_constraint(parent, child, joint);
        let mut view = storage.solver_view_mut();
        let initial = view.angular_velocities[child.index() as usize].length();
        for _ in 0..20 {
            solve_ragdoll_velocity(&mut view, &constraint, 1.0 / 60.0);
        }
        let final_speed = view.angular_velocities[child.index() as usize].length();
        assert!(final_speed < initial);
        assert!(view.angular_velocities[child.index() as usize].is_finite());
    }

    #[test]
    fn revolute_joint_aligns_hinge_axis() {
        let mut s = BodyStorage::new();
        let a = s.insert(dynamic(Vec3::ZERO));
        let b = s.insert(dynamic(Vec3::ZERO));
        s.set_orientation(b, Quat::from_rotation_x(0.3));
        let mut joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::revolute(
                JointAnchor::at_center(a),
                JointAnchor::at_center(b),
                RevoluteJoint::new(Vec3::Z),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        for _ in 0..40 {
            solve_joints(&mut view, &mut joints, 1.0 / 60.0);
        }
        let axis_a = view.orientations[a.index() as usize] * Vec3::Z;
        let axis_b = view.orientations[b.index() as usize] * Vec3::Z;
        let misalignment = axis_a.cross(axis_b).length();
        assert!(
            misalignment < 1.0e-2,
            "hinge axes not aligned, cross = {misalignment}"
        );
    }

    #[test]
    fn prismatic_joint_removes_perpendicular_offset() {
        let mut s = BodyStorage::new();
        let a = s.insert(dynamic(Vec3::ZERO));
        let b = s.insert(dynamic(Vec3::new(0.5, 0.4, 0.3)));
        let mut joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::prismatic(
                JointAnchor::at_center(a),
                JointAnchor::at_center(b),
                PrismaticJoint::new(Vec3::X),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        for _ in 0..40 {
            solve_joints(&mut view, &mut joints, 1.0 / 60.0);
        }
        let axis = Vec3::X;
        let delta = view.positions[a.index() as usize] - view.positions[b.index() as usize];
        let perp = (delta - axis * delta.dot(axis)).length();
        assert!(perp < 1.0e-3, "perpendicular offset remains, perp = {perp}");
        // The free slide coordinate along the axis is preserved (unconstrained).
        assert!(
            delta.dot(axis).abs() > 0.1,
            "slide coordinate was wrongly removed"
        );
    }

    #[test]
    fn skips_joint_between_two_static_bodies() {
        let mut s = BodyStorage::new();
        let a = s.insert(BodyDesc::static_at(Vec3::ZERO));
        let b = s.insert(BodyDesc::static_at(Vec3::new(2.0, 0.0, 0.0)));
        let mut joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::distance(
                a,
                Vec3::ZERO,
                b,
                Vec3::ZERO,
                DistanceJoint::rigid(1.0),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        solve_joints(&mut view, &mut joints, 1.0 / 60.0);
        // Neither static body moved.
        assert_eq!(view.positions[a.index() as usize], Vec3::ZERO);
        assert_eq!(view.positions[b.index() as usize], Vec3::new(2.0, 0.0, 0.0));
    }
}
