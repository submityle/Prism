//! World-owned, unwrapped hinge coordinates. A pose alone has no winding number.
//!
//! Consecutive observations must differ by less than pi in relative twist.
//! Substep fast motion; explicitly seed/reset after ambiguous teleports.

use super::JointAnchor;
use crate::state::handle::BodyHandle;
use glam::{Quat, Vec3};

/// Geometry defining a coordinate, excluding limits, motors and anchor points.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
struct RevoluteBinding {
    a: BodyHandle,
    b: BodyHandle,
    frame_a: Quat,
    frame_b: Quat,
    axis: Vec3,
}

/// Continuous angular history, owned and snapshotted by a single live joint.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RevoluteJointState {
    binding: Option<RevoluteBinding>,
    angle: f64,
    principal: f64,
    pub(crate) motor_lambda: f64,
    pub(crate) motor_goal: Option<f64>,
}

impl RevoluteJointState {
    /// Last observed continuous coordinate, or `None` before binding.
    #[must_use]
    pub fn angle(&self) -> Option<f64> {
        self.binding.map(|_| self.angle)
    }

    /// Last principal twist, useful for diagnostics (not for limit/motor errors).
    #[must_use]
    pub fn principal_angle(&self) -> Option<f64> {
        self.binding.map(|_| self.principal)
    }

    pub(crate) fn seed(
        &mut self,
        angle: f64,
        axis: Vec3,
        a: JointAnchor,
        b: JointAnchor,
        orientations: [Quat; 2],
    ) -> bool {
        if !angle.is_finite() {
            return false;
        }
        let mut candidate = Self::default();
        if candidate.observe(axis, a, b, orientations).is_none() {
            return false;
        }
        candidate.angle = angle;
        *self = candidate;
        true
    }

    /// Returns the exact world angular gradient with respect to endpoint B.
    pub(crate) fn observe(
        &mut self,
        axis: Vec3,
        a: JointAnchor,
        b: JointAnchor,
        orientations: [Quat; 2],
    ) -> Option<Vec3> {
        if a.body == b.body
            || !unit_frame(a.local_frame)
            || !unit_frame(b.local_frame)
            || !orientations.into_iter().all(unit_frame)
        {
            return None;
        }
        let axis = unit_axis(axis)?;
        let frame_a = (orientations[0] * a.local_frame).normalize();
        let frame_b = (orientations[1] * b.local_frame).normalize();
        let relative = (frame_a.inverse() * frame_b).normalize();
        let v = Vec3::new(relative.x, relative.y, relative.z).as_dvec3();
        let n = axis.as_dvec3();
        let w = f64::from(relative.w);
        let s = v.dot(n);
        let denominator = w * w + s * s;
        // Twist is undefined at a 180-degree swing; do not invent a sample.
        if denominator <= 1.0e-12 {
            return None;
        }
        let principal = wrap(2.0 * s.atan2(w));
        let gradient = frame_a * ((w * w * n + w * v.cross(n) + s * v) / denominator).as_vec3();
        if !gradient.is_finite() {
            return None;
        }
        let binding = RevoluteBinding {
            a: a.body,
            b: b.body,
            frame_a: canonical_frame(a.local_frame),
            frame_b: canonical_frame(b.local_frame),
            axis,
        };
        if self.binding != Some(binding) {
            self.binding = Some(binding);
            self.angle = principal;
            self.clear_substep();
        } else {
            let angle = self.angle + wrap(principal - self.principal);
            if !angle.is_finite() {
                return None;
            }
            self.angle = angle;
        }
        self.principal = principal;
        Some(gradient)
    }

    pub(crate) fn clear_substep(&mut self) {
        self.motor_lambda = 0.0;
        self.motor_goal = None;
    }

    /// Moving into island scratch changes addresses, not the physical binding.
    pub(crate) fn remap_bodies(&mut self, a: BodyHandle, b: BodyHandle) {
        if let Some(binding) = &mut self.binding {
            binding.a = a;
            binding.b = b;
        }
    }
}

pub(crate) fn unit_axis(axis: Vec3) -> Option<Vec3> {
    let axis = axis.as_dvec3();
    let norm = axis.length();
    (norm.is_finite() && norm > 0.0).then(|| (axis / norm).as_vec3())
}

fn unit_frame(frame: Quat) -> bool {
    frame.is_finite() && (frame.length_squared() - 1.0).abs() <= 1.0e-4
}

fn canonical_frame(frame: Quat) -> Quat {
    // q and -q describe the same local frame, including at w == 0.
    let first = [frame.w, frame.x, frame.y, frame.z]
        .into_iter()
        .find(|component| *component != 0.0)
        .unwrap_or(1.0);
    if first < 0.0 {
        -frame
    } else {
        frame
    }
}

fn wrap(angle: f64) -> f64 {
    (angle + core::f64::consts::PI).rem_euclid(core::f64::consts::TAU) - core::f64::consts::PI
}
