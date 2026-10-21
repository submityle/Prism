//! Continuous shaft coordinates for an angular gear transmission.
//!
//! A quaternion alone cannot encode a winding number. Each live `Joint` owns
//! this history; full-world snapshots clone it and the parallel solver copies
//! it into, and back out of, its private island scratch. No solver-global cache
//! or handle-index-only lookup participates in replay.

use super::JointAnchor;
use crate::state::handle::BodyHandle;
use glam::{Quat, Vec3};

/// A single angular constraint `theta_b - ratio * theta_a = phase`.
///
/// Positive ratios turn the shafts in the same signed direction; negative
/// ratios reverse direction. Each axis is expressed in its own anchor frame.
/// The reference frames are fixed world orientations of those anchor frames
/// at zero angle. Anchor points are unused: this joint does not mount shafts,
/// align their axes, or impose a centre distance. Combine with revolute mounts
/// when those restrictions are desired.
///
/// Angles are continuous swing-twist coordinates about each reference axis,
/// not principal angles reduced modulo one turn. As with pose-based velocity
/// recovery, consecutive observations must be less than pi apart. Use smaller
/// substeps for fast motion; reset the history explicitly after teleports.
/// A 180-degree swing relative to a reference has undefined twist and is not
/// projected. Mounted shafts do not encounter that singularity.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GearJoint {
    /// Shaft A's nonzero local axis (normalised on use).
    pub axis_a: Vec3,
    /// Shaft B's nonzero local axis (normalised on use).
    pub axis_b: Vec3,
    /// Signed angular speed ratio `omega_b / omega_a`; finite and nonzero.
    pub ratio: f32,
    /// Unwrapped angular offset in radians, in B-coordinate units.
    pub phase: f64,
    /// Fixed world-space zero orientation of A's anchor frame (unit quaternion).
    pub reference_frame_a: Quat,
    /// Fixed world-space zero orientation of B's anchor frame (unit quaternion).
    pub reference_frame_b: Quat,
    /// XPBD angular compliance; finite, nonnegative. Zero is rigid.
    pub compliance: f32,
}

impl GearJoint {
    /// Creates a rigid, zero-phase transmission with identity zero frames.
    #[must_use]
    pub const fn new(axis_a: Vec3, axis_b: Vec3, ratio: f32) -> Self {
        Self {
            axis_a,
            axis_b,
            ratio,
            phase: 0.0,
            reference_frame_a: Quat::IDENTITY,
            reference_frame_b: Quat::IDENTITY,
            compliance: 0.0,
        }
    }

    /// Sets the unwrapped offset; no modulo or automatic rest-pose capture.
    #[must_use]
    pub const fn with_phase(mut self, phase: f64) -> Self {
        self.phase = phase;
        self
    }

    /// Sets the two fixed world-space zero frames independently.
    #[must_use]
    pub const fn with_reference_frames(mut self, a: Quat, b: Quat) -> Self {
        self.reference_frame_a = a;
        self.reference_frame_b = b;
        self
    }

    /// Sets the angular compliance.
    #[must_use]
    pub const fn with_compliance(mut self, compliance: f32) -> Self {
        self.compliance = compliance;
        self
    }

    /// Checks authored parameters without changing them. XPBD also checks on
    /// use, so invalid data admitted through a struct literal is a safe no-op.
    pub fn validate(&self) -> Result<(), GearJointError> {
        if unit_axis(self.axis_a).is_none() || unit_axis(self.axis_b).is_none() {
            return Err(GearJointError::InvalidAxis);
        }
        if !self.ratio.is_finite() || self.ratio == 0.0 {
            return Err(GearJointError::InvalidRatio);
        }
        if !self.phase.is_finite() {
            return Err(GearJointError::InvalidPhase);
        }
        if !unit_frame(self.reference_frame_a) || !unit_frame(self.reference_frame_b) {
            return Err(GearJointError::InvalidReferenceFrame);
        }
        if !self.compliance.is_finite() || self.compliance < 0.0 {
            return Err(GearJointError::InvalidCompliance);
        }
        Ok(())
    }
}

/// Invalid authored gear parameters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GearJointError {
    /// An axis is zero or nonfinite.
    InvalidAxis,
    /// The ratio is zero or nonfinite.
    InvalidRatio,
    /// The phase is nonfinite.
    InvalidPhase,
    /// A zero frame is not a finite unit quaternion.
    InvalidReferenceFrame,
    /// Compliance is negative or nonfinite.
    InvalidCompliance,
}

/// Binding geometry, deliberately excluding ratio, phase and compliance.
/// Retuning a transmission preserves its winding numbers; changing either
/// shaft/reference frame or endpoint starts a new coordinate history.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
struct GearBinding {
    a: JointAnchor,
    b: JointAnchor,
    axis_a: Vec3,
    axis_b: Vec3,
    reference_a: Quat,
    reference_b: Quat,
}

/// World-owned gear history, copied/serialized with its joint.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GearJointState {
    binding: Option<GearBinding>,
    angles: [f64; 2],
    principal: [f64; 2],
    pub(crate) lambda: f64,
}

impl GearJointState {
    /// The last observed unwrapped shaft angles, or `None` before binding.
    #[must_use]
    pub fn angles(&self) -> Option<[f64; 2]> {
        self.binding.map(|_| self.angles)
    }

    /// Observes actual poses, never integrated velocity estimates. Quaternion
    /// signs and the principal-angle seam do not change the winding history.
    pub(crate) fn observe(
        &mut self,
        gear: GearJoint,
        a: JointAnchor,
        b: JointAnchor,
        orientations: [Quat; 2],
    ) -> Option<[Vec3; 2]> {
        if gear.validate().is_err() || !unit_frame(a.local_frame) || !unit_frame(b.local_frame) {
            return None;
        }
        let (theta_a, gradient_a) = coordinate(
            orientations[0] * a.local_frame,
            gear.reference_frame_a,
            gear.axis_a,
        )?;
        let (theta_b, gradient_b) = coordinate(
            orientations[1] * b.local_frame,
            gear.reference_frame_b,
            gear.axis_b,
        )?;
        let binding = GearBinding {
            a,
            b,
            axis_a: gear.axis_a,
            axis_b: gear.axis_b,
            reference_a: gear.reference_frame_a,
            reference_b: gear.reference_frame_b,
        };
        let principal = [theta_a, theta_b];
        if self.binding != Some(binding) {
            self.binding = Some(binding);
            self.angles = principal;
            self.lambda = 0.0;
        } else {
            for (i, theta) in principal.iter().enumerate() {
                let delta = (theta - self.principal[i] + core::f64::consts::PI)
                    .rem_euclid(core::f64::consts::TAU)
                    - core::f64::consts::PI;
                self.angles[i] += delta;
            }
        }
        self.principal = principal;
        Some([gradient_a, gradient_b])
    }

    /// Scratch remapping is a change of storage address, not a new shaft.
    pub(crate) fn remap_bodies(&mut self, a: BodyHandle, b: BodyHandle) {
        if let Some(binding) = &mut self.binding {
            binding.a.body = a;
            binding.b.body = b;
        }
    }
}

fn unit_axis(axis: Vec3) -> Option<Vec3> {
    // f64 avoids overflow/underflow even for finite f32 extreme axes.
    let axis = axis.as_dvec3();
    let norm = axis.length();
    (norm.is_finite() && norm > 0.0).then(|| (axis / norm).as_vec3())
}

pub(crate) fn unit_frame(frame: Quat) -> bool {
    frame.is_finite() && (frame.length_squared() - 1.0).abs() <= 1.0e-4
}

/// Principal twist and its *exact* world angular gradient. With relative
/// quaternion `(v,w)` and `s = dot(v,axis)`, theta = 2 atan2(s,w). Its derivative
/// under a world-space infinitesimal rotation is
/// reference * ((w*w*axis + w*(v cross axis) + s*v)/(w*w+s*s)).
/// Using this derivative also gives correct inertia weighting with shaft swing.
fn coordinate(frame: Quat, reference: Quat, axis: Vec3) -> Option<(f64, Vec3)> {
    if !unit_frame(frame) {
        return None;
    }
    let axis = unit_axis(axis)?;
    let reference = reference.normalize();
    let relative = (reference.inverse() * frame).normalize();
    let v = Vec3::new(relative.x, relative.y, relative.z).as_dvec3();
    let axis = axis.as_dvec3();
    let w = f64::from(relative.w);
    let s = v.dot(axis);
    let denominator = w * w + s * s;
    if denominator <= 1.0e-12 {
        return None;
    }
    let principal = (2.0 * s.atan2(w) + core::f64::consts::PI).rem_euclid(core::f64::consts::TAU)
        - core::f64::consts::PI;
    let gradient = reference * ((w * w * axis + w * v.cross(axis) + s * v) / denominator).as_vec3();
    gradient.is_finite().then_some((principal, gradient))
}
