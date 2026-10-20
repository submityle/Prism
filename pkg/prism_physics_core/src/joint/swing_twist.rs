//! Solver-neutral swing-twist limits and angular response.
//!
//! [`SwingTwistJoint`] is the core M13 contract for a ragdoll rotational
//! connection. It measures a relative pose in two joint frames, evaluates an
//! elliptical swing cone and a signed twist interval, and exposes bounded
//! passive and active responses. The position projection and velocity response
//! are intentionally small, deterministic primitives: a world solver owns
//! iteration, warm-start state, island ordering, and contact coupling.
//!
//! No GPU implementation is assumed here. A backend may mirror these equations
//! only after it has established an explicit parity contract with this module.

use glam::{Mat3, Quat, Vec3};

use super::drive::{ActuationBlend, AngularFriction, PoseDrive, SoftLimit};

const EPS: f32 = 1.0e-6;

/// Configuration error returned by [`SwingTwistJoint::try_new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SwingTwistConfigError {
    /// An axis or reference direction was non-finite or zero length.
    InvalidDirection,
    /// A reference direction was parallel to its twist axis.
    ReferenceParallelToAxis,
    /// A swing half-angle was not finite and strictly positive.
    InvalidSwingLimit,
    /// The twist range was malformed or non-finite.
    InvalidTwistRange,
    /// A response parameter was negative or non-finite.
    InvalidResponse,
    /// A pose-drive target quaternion could not be normalised.
    InvalidDriveTarget,
}

/// Elliptical swing and signed twist limits in the two body-local frames.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SwingTwistConfig {
    /// Twist axis on body `a`, in body-local coordinates.
    pub twist_axis_a: Vec3,
    /// Twist axis on body `b`, in body-local coordinates.
    pub twist_axis_b: Vec3,
    /// Zero-twist reference direction on body `a`.
    pub reference_a: Vec3,
    /// Zero-twist reference direction on body `b`.
    pub reference_b: Vec3,
    /// Ellipse half-angle along the reference direction.
    pub swing_limit_1: f32,
    /// Ellipse half-angle along the orthogonal swing direction.
    pub swing_limit_2: f32,
    /// Inclusive signed twist range in radians.
    pub twist_limit_min: f32,
    /// Inclusive signed twist range in radians.
    pub twist_limit_max: f32,
    /// Soft response for swing violation.
    pub swing_response: SoftLimit,
    /// Soft response for twist violation.
    pub twist_response: SoftLimit,
    /// Friction/damping for swing and twist relative motion.
    pub friction: JointFriction,
    /// Optional active pose drive.
    pub drive: Option<PoseDrive>,
    /// Stable active/passive response blend.
    pub blend: ActuationBlend,
}

impl SwingTwistConfig {
    /// Creates a rigid-limit configuration with no drive or friction.
    #[must_use]
    pub fn symmetric(
        twist_axis_a: Vec3,
        twist_axis_b: Vec3,
        reference_a: Vec3,
        reference_b: Vec3,
        swing_limit: f32,
        twist_limit: f32,
    ) -> Self {
        Self {
            twist_axis_a,
            twist_axis_b,
            reference_a,
            reference_b,
            swing_limit_1: swing_limit,
            swing_limit_2: swing_limit,
            twist_limit_min: -twist_limit,
            twist_limit_max: twist_limit,
            swing_response: SoftLimit::hard(),
            twist_response: SoftLimit::hard(),
            friction: JointFriction::default(),
            drive: None,
            blend: ActuationBlend::PASSIVE,
        }
    }

    /// Replaces the two independent swing half-angles.
    #[must_use]
    pub const fn with_swing_limits(mut self, limit_1: f32, limit_2: f32) -> Self {
        self.swing_limit_1 = limit_1;
        self.swing_limit_2 = limit_2;
        self
    }

    /// Replaces the signed twist interval.
    #[must_use]
    pub const fn with_twist_limits(mut self, min: f32, max: f32) -> Self {
        self.twist_limit_min = min;
        self.twist_limit_max = max;
        self
    }

    /// Replaces the soft response of both limits.
    #[must_use]
    pub const fn with_responses(mut self, swing: SoftLimit, twist: SoftLimit) -> Self {
        self.swing_response = swing;
        self.twist_response = twist;
        self
    }

    /// Adds or removes an active pose drive.
    #[must_use]
    pub const fn with_drive(mut self, drive: Option<PoseDrive>) -> Self {
        self.drive = drive;
        self
    }

    /// Sets the active/passive blend.
    #[must_use]
    pub const fn with_blend(mut self, blend: ActuationBlend) -> Self {
        self.blend = blend;
        self
    }
}

/// Swing and twist angular friction coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct JointFriction {
    /// Friction acting on swing-plane relative speed.
    pub swing: AngularFriction,
    /// Friction acting on twist relative speed.
    pub twist: AngularFriction,
}

impl Default for JointFriction {
    fn default() -> Self {
        Self {
            swing: AngularFriction::default(),
            twist: AngularFriction::default(),
        }
    }
}

/// A validated swing-twist joint response contract.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SwingTwistJoint {
    config: SwingTwistConfig,
}

impl SwingTwistJoint {
    /// Validates and creates a swing-twist constraint.
    #[must_use]
    pub fn try_new(config: SwingTwistConfig) -> Result<Self, SwingTwistConfigError> {
        validate_direction(config.twist_axis_a)?;
        validate_direction(config.twist_axis_b)?;
        if project_perpendicular(config.reference_a, config.twist_axis_a).is_none()
            || project_perpendicular(config.reference_b, config.twist_axis_b).is_none()
        {
            return Err(SwingTwistConfigError::ReferenceParallelToAxis);
        }
        if !valid_positive(config.swing_limit_1) || !valid_positive(config.swing_limit_2) {
            return Err(SwingTwistConfigError::InvalidSwingLimit);
        }
        if !config.twist_limit_min.is_finite()
            || !config.twist_limit_max.is_finite()
            || config.twist_limit_min > config.twist_limit_max
        {
            return Err(SwingTwistConfigError::InvalidTwistRange);
        }
        if !valid_soft_limit(config.swing_response) || !valid_soft_limit(config.twist_response) {
            return Err(SwingTwistConfigError::InvalidResponse);
        }
        if !valid_friction(config.friction) {
            return Err(SwingTwistConfigError::InvalidResponse);
        }
        if let Some(drive) = config.drive {
            if !drive.target.is_finite()
                || drive.target.length_squared() <= EPS
                || !drive.stiffness.is_finite()
                || drive.stiffness < 0.0
                || !drive.damping.is_finite()
                || drive.damping < 0.0
                || !drive.max_torque.is_finite()
                || drive.max_torque < 0.0
            {
                return Err(SwingTwistConfigError::InvalidDriveTarget);
            }
        }
        if !config.blend.active.is_finite()
            || !config.blend.passive.is_finite()
            || config.blend.active < 0.0
            || config.blend.passive < 0.0
            || (config.blend.active + config.blend.passive) <= EPS
        {
            return Err(SwingTwistConfigError::InvalidResponse);
        }
        Ok(Self { config })
    }

    /// Returns the validated configuration.
    #[must_use]
    pub const fn config(&self) -> &SwingTwistConfig {
        &self.config
    }

    /// Measures the relative orientation and both limit violations.
    #[must_use]
    pub fn measure(&self, orientation_a: Quat, orientation_b: Quat) -> SwingTwistMeasurement {
        let axis_a = orientation_a * self.config.twist_axis_a.normalize();
        let axis_b = orientation_b * self.config.twist_axis_b.normalize();
        let axis_a = safe_normalize(axis_a, Vec3::X);
        let axis_b = safe_normalize(axis_b, axis_a);
        let dot = axis_a.dot(axis_b).clamp(-1.0, 1.0);
        let mut correction_axis = axis_a.cross(axis_b);
        if correction_axis.length_squared() <= EPS * EPS {
            let fallback = project_perpendicular(orientation_a * self.config.reference_a, axis_a)
                .unwrap_or_else(|| orthogonal(axis_a));
            correction_axis = if dot < 0.0 {
                fallback
            } else {
                axis_a.cross(fallback)
            };
        }
        let correction_axis = safe_normalize(correction_axis, orthogonal(axis_a));
        let swing_angle = dot.acos();

        let frame_a_world =
            orientation_a * joint_frame(self.config.twist_axis_a, self.config.reference_a);
        let swing_axis_1 = safe_normalize(frame_a_world * Vec3::Y, orthogonal(axis_a));
        let swing_axis_2 = safe_normalize(axis_a.cross(swing_axis_1), orthogonal(swing_axis_1));
        let transverse = axis_b - axis_a * dot;
        let transverse_length = transverse.length();
        let (azimuth, swing_limit) = if transverse_length > EPS {
            let unit = transverse / transverse_length;
            let c = unit.dot(swing_axis_1);
            let s = unit.dot(swing_axis_2);
            let azimuth = s.atan2(c);
            (
                azimuth,
                ellipse_radius(
                    self.config.swing_limit_1,
                    self.config.swing_limit_2,
                    azimuth,
                ),
            )
        } else {
            (
                0.0,
                self.config.swing_limit_1.min(self.config.swing_limit_2),
            )
        };
        let swing_violation = (swing_angle - swing_limit).max(0.0);

        let reference_a = safe_normalize(
            project_perpendicular(orientation_a * self.config.reference_a, axis_a)
                .unwrap_or(swing_axis_1),
            swing_axis_1,
        );
        let reference_b = safe_normalize(
            project_perpendicular(orientation_b * self.config.reference_b, axis_a)
                .unwrap_or(swing_axis_1),
            swing_axis_1,
        );
        let twist_angle = reference_a
            .cross(reference_b)
            .dot(axis_a)
            .atan2(reference_a.dot(reference_b));
        let twist_violation = if twist_angle < self.config.twist_limit_min {
            twist_angle - self.config.twist_limit_min
        } else if twist_angle > self.config.twist_limit_max {
            twist_angle - self.config.twist_limit_max
        } else {
            0.0
        };

        let frame_b_world =
            orientation_b * joint_frame(self.config.twist_axis_b, self.config.reference_b);
        let relative_rotation = safe_quat(frame_a_world.inverse() * frame_b_world);
        SwingTwistMeasurement {
            relative_rotation,
            twist_axis_world: axis_a,
            swing_correction_axis_world: correction_axis,
            swing_angle,
            swing_azimuth: azimuth,
            swing_limit,
            swing_violation,
            twist_angle,
            twist_violation,
        }
    }

    /// Computes the bounded passive and active torque response for one state.
    #[must_use]
    pub fn response(&self, input: AngularJointInput) -> AngularJointResponse {
        let measurement = self.measure(input.orientation_a, input.orientation_b);
        let frame_a_world =
            input.orientation_a * joint_frame(self.config.twist_axis_a, self.config.reference_a);
        let relative_velocity_world = input.angular_velocity_b - input.angular_velocity_a;
        let relative_velocity_local = frame_a_world.inverse() * relative_velocity_world;

        let swing_speed = relative_velocity_world
            - measurement.twist_axis_world
                * relative_velocity_world.dot(measurement.twist_axis_world);
        let swing_speed_magnitude = swing_speed.length();
        let swing_friction = if swing_speed_magnitude > EPS {
            let axis = swing_speed / swing_speed_magnitude;
            axis * self.config.friction.swing.torque(swing_speed_magnitude)
        } else {
            Vec3::ZERO
        };
        let twist_speed = relative_velocity_world.dot(measurement.twist_axis_world);
        let twist_friction =
            measurement.twist_axis_world * self.config.friction.twist.torque(twist_speed);

        let mut passive = swing_friction + twist_friction;
        if measurement.swing_violation > 0.0 {
            passive += measurement.swing_correction_axis_world
                * self.config.swing_response.restoring_torque(
                    measurement.swing_violation,
                    relative_velocity_world.dot(measurement.swing_correction_axis_world),
                );
        }
        if measurement.twist_violation != 0.0 {
            passive += measurement.twist_axis_world
                * self
                    .config
                    .twist_response
                    .restoring_torque(measurement.twist_violation, twist_speed);
        }

        let active = self
            .config
            .drive
            .map(|drive| {
                let error = rotation_vector(drive.target * measurement.relative_rotation.inverse());
                let local = error * drive.stiffness - relative_velocity_local * drive.damping;
                let magnitude = local.length();
                let capped = if magnitude > EPS {
                    local * (magnitude.min(drive.max_torque) / magnitude)
                } else {
                    Vec3::ZERO
                };
                frame_a_world * capped
            })
            .unwrap_or(Vec3::ZERO);
        let total = self.config.blend.mix_vec3(active, passive);
        AngularJointResponse {
            measurement,
            active_torque_world: active,
            passive_torque_world: passive,
            torque_on_a_world: -total,
            torque_on_b_world: total,
        }
    }

    /// Applies the response torque to two angular velocity vectors.
    ///
    /// This is a velocity response primitive, not a world solver: the caller
    /// controls iteration order, sleeping, contacts, and force accumulation.
    pub fn apply_velocity_response(
        &self,
        input: AngularJointInput,
        angular_velocity_a: &mut Vec3,
        angular_velocity_b: &mut Vec3,
    ) -> AngularJointResponse {
        let response = self.response(input);
        if input.dt > 0.0 && input.dt.is_finite() {
            *angular_velocity_a += world_inverse_inertia(
                input.orientation_a,
                input.inv_inertia_a,
                response.torque_on_a_world,
            ) * input.dt;
            *angular_velocity_b += world_inverse_inertia(
                input.orientation_b,
                input.inv_inertia_b,
                response.torque_on_b_world,
            ) * input.dt;
        }
        response
    }

    /// Projects swing and twist limit violations into two orientations.
    ///
    /// The correction is split by each body's effective inverse inertia around
    /// the correction axis. A static side therefore receives no correction and
    /// a dynamic side receives the complete relative correction.
    pub fn project_limits(
        &self,
        orientation_a: &mut Quat,
        orientation_b: &mut Quat,
        inv_inertia_a: Vec3,
        inv_inertia_b: Vec3,
        dt: f32,
    ) -> SwingTwistMeasurement {
        let measurement = self.measure(*orientation_a, *orientation_b);
        let swing_fraction = self.config.swing_response.correction_fraction(dt);
        if measurement.swing_violation > 0.0 {
            let axis = measurement.swing_correction_axis_world;
            let (share_a, share_b) = inertia_shares(
                *orientation_a,
                *orientation_b,
                inv_inertia_a,
                inv_inertia_b,
                axis,
            );
            let amount = measurement.swing_violation * swing_fraction;
            *orientation_a =
                (Quat::from_scaled_axis(-axis * amount * share_a) * *orientation_a).normalize();
            *orientation_b =
                (Quat::from_scaled_axis(axis * amount * share_b) * *orientation_b).normalize();
        }

        let after_swing = self.measure(*orientation_a, *orientation_b);
        if after_swing.twist_violation != 0.0 {
            let axis = after_swing.twist_axis_world;
            let (share_a, share_b) = inertia_shares(
                *orientation_a,
                *orientation_b,
                inv_inertia_a,
                inv_inertia_b,
                axis,
            );
            let amount =
                after_swing.twist_violation * self.config.twist_response.correction_fraction(dt);
            *orientation_a =
                (Quat::from_scaled_axis(axis * amount * share_a) * *orientation_a).normalize();
            *orientation_b =
                (Quat::from_scaled_axis(-axis * amount * share_b) * *orientation_b).normalize();
        }
        self.measure(*orientation_a, *orientation_b)
    }
}

/// Relative pose and angular velocity supplied by a world solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularJointInput {
    /// World orientation of body `a`.
    pub orientation_a: Quat,
    /// World orientation of body `b`.
    pub orientation_b: Quat,
    /// World angular velocity of body `a`.
    pub angular_velocity_a: Vec3,
    /// World angular velocity of body `b`.
    pub angular_velocity_b: Vec3,
    /// Local principal inverse inertia of body `a`.
    pub inv_inertia_a: Vec3,
    /// Local principal inverse inertia of body `b`.
    pub inv_inertia_b: Vec3,
    /// Solver substep duration in seconds.
    pub dt: f32,
}

/// Measured state of a swing-twist joint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwingTwistMeasurement {
    /// Relative rotation from joint frame `a` to joint frame `b`.
    pub relative_rotation: Quat,
    /// World twist axis used by the signed twist measurement.
    pub twist_axis_world: Vec3,
    /// World axis that closes the swing violation.
    pub swing_correction_axis_world: Vec3,
    /// Current swing angle in radians.
    pub swing_angle: f32,
    /// Swing azimuth in the ellipse frame, in radians.
    pub swing_azimuth: f32,
    /// Ellipse radius at the current azimuth, in radians.
    pub swing_limit: f32,
    /// Positive swing violation, in radians.
    pub swing_violation: f32,
    /// Signed twist angle, in radians.
    pub twist_angle: f32,
    /// Signed twist violation; zero is inside the interval.
    pub twist_violation: f32,
}

/// Torque response returned by [`SwingTwistJoint::response`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularJointResponse {
    /// Measured state used to compute this response.
    pub measurement: SwingTwistMeasurement,
    /// Active pose-drive torque before blending.
    pub active_torque_world: Vec3,
    /// Passive limit/friction torque before blending.
    pub passive_torque_world: Vec3,
    /// Equal-and-opposite torque on body `a` after blending.
    pub torque_on_a_world: Vec3,
    /// Torque on body `b` after blending.
    pub torque_on_b_world: Vec3,
}

fn validate_direction(direction: Vec3) -> Result<(), SwingTwistConfigError> {
    if direction.is_finite() && direction.length_squared() > EPS * EPS {
        Ok(())
    } else {
        Err(SwingTwistConfigError::InvalidDirection)
    }
}

fn valid_positive(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

fn valid_soft_limit(value: SoftLimit) -> bool {
    value.compliance.is_finite()
        && value.compliance >= 0.0
        && value.stiffness.is_finite()
        && value.stiffness >= 0.0
        && value.damping.is_finite()
        && value.damping >= 0.0
        && value
            .max_torque
            .is_none_or(|max| max.is_finite() && max >= 0.0)
}

fn valid_friction(value: JointFriction) -> bool {
    [value.swing, value.twist].iter().all(|friction| {
        friction.viscous.is_finite()
            && friction.viscous >= 0.0
            && friction.coulomb.is_finite()
            && friction.coulomb >= 0.0
            && friction.static_speed.is_finite()
            && friction.static_speed > 0.0
    })
}

fn project_perpendicular(value: Vec3, axis: Vec3) -> Option<Vec3> {
    let projected = value - axis.normalize_or_zero() * value.dot(axis.normalize_or_zero());
    (projected.is_finite() && projected.length_squared() > EPS * EPS)
        .then_some(projected.normalize())
}

fn safe_normalize(value: Vec3, fallback: Vec3) -> Vec3 {
    if value.is_finite() && value.length_squared() > EPS * EPS {
        value.normalize()
    } else {
        fallback.normalize_or_zero()
    }
}

fn safe_quat(value: Quat) -> Quat {
    if value.is_finite() && value.length_squared() > EPS * EPS {
        value.normalize()
    } else {
        Quat::IDENTITY
    }
}

fn orthogonal(axis: Vec3) -> Vec3 {
    let candidate = if axis.x.abs() < 0.8 { Vec3::X } else { Vec3::Y };
    safe_normalize(axis.cross(candidate), Vec3::Z)
}

fn joint_frame(axis: Vec3, reference: Vec3) -> Quat {
    let x = safe_normalize(axis, Vec3::X);
    let y = project_perpendicular(reference, x).unwrap_or_else(|| orthogonal(x));
    let z = safe_normalize(x.cross(y), Vec3::Z);
    Quat::from_mat3(&Mat3::from_cols(x, y, z))
}

fn ellipse_radius(limit_1: f32, limit_2: f32, azimuth: f32) -> f32 {
    let c = azimuth.cos();
    let s = azimuth.sin();
    let denominator = (c / limit_1).powi(2) + (s / limit_2).powi(2);
    if denominator > EPS {
        denominator.sqrt().recip()
    } else {
        limit_1.min(limit_2)
    }
}

fn rotation_vector(rotation: Quat) -> Vec3 {
    let mut q = safe_quat(rotation);
    if q.w < 0.0 {
        q = -q;
    }
    let angle = 2.0 * q.w.clamp(-1.0, 1.0).acos();
    let sin_half = (1.0 - q.w * q.w).max(0.0).sqrt();
    if sin_half <= EPS || angle <= EPS {
        q.xyz() * 2.0
    } else {
        q.xyz() * (angle / sin_half)
    }
}

fn world_inverse_inertia(orientation: Quat, inv_inertia: Vec3, torque: Vec3) -> Vec3 {
    let local_torque = orientation.inverse() * torque;
    orientation * (inv_inertia * local_torque)
}

fn inertia_shares(
    orientation_a: Quat,
    orientation_b: Quat,
    inv_inertia_a: Vec3,
    inv_inertia_b: Vec3,
    axis: Vec3,
) -> (f32, f32) {
    let effective_a = axis
        .dot(world_inverse_inertia(orientation_a, inv_inertia_a, axis))
        .max(0.0);
    let effective_b = axis
        .dot(world_inverse_inertia(orientation_b, inv_inertia_b, axis))
        .max(0.0);
    let total = effective_a + effective_b;
    if total > EPS {
        (effective_a / total, effective_b / total)
    } else {
        (0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joint() -> SwingTwistJoint {
        SwingTwistJoint::try_new(SwingTwistConfig::symmetric(
            Vec3::X,
            Vec3::X,
            Vec3::Y,
            Vec3::Y,
            0.4,
            0.5,
        ))
        .unwrap()
    }

    #[test]
    fn ellipse_radius_matches_axes_and_interpolates() {
        let config = SwingTwistConfig::symmetric(Vec3::X, Vec3::X, Vec3::Y, Vec3::Y, 0.4, 0.5)
            .with_swing_limits(0.25, 0.75);
        assert!(
            (ellipse_radius(config.swing_limit_1, config.swing_limit_2, 0.0) - 0.25).abs() < 1.0e-6
        );
        assert!(
            (ellipse_radius(
                config.swing_limit_1,
                config.swing_limit_2,
                std::f32::consts::FRAC_PI_2
            ) - 0.75)
                .abs()
                < 1.0e-6
        );
        let diagonal = ellipse_radius(
            config.swing_limit_1,
            config.swing_limit_2,
            std::f32::consts::FRAC_PI_4,
        );
        assert!(diagonal > 0.25 && diagonal < 0.75);
    }

    #[test]
    fn twist_violation_is_signed_and_position_projection_reduces_it() {
        let joint = joint();
        let mut a = Quat::IDENTITY;
        let mut b = Quat::from_axis_angle(Vec3::X, 1.2);
        let before = joint.measure(a, b);
        assert!(before.twist_violation > 0.0);
        let after = joint.project_limits(&mut a, &mut b, Vec3::ONE, Vec3::ONE, 1.0 / 60.0);
        assert!(after.twist_violation.abs() < before.twist_violation.abs());
        assert!(a.is_normalized() && b.is_normalized());
    }

    #[test]
    fn static_side_receives_no_position_correction() {
        let joint = joint();
        let a_before = Quat::IDENTITY;
        let mut a = a_before;
        let mut b = Quat::from_axis_angle(Vec3::Z, 1.0);
        joint.project_limits(&mut a, &mut b, Vec3::ZERO, Vec3::ONE, 1.0 / 60.0);
        assert_eq!(a, a_before);
    }

    #[test]
    fn active_and_passive_torques_are_equal_and_opposite() {
        let response = joint().response(AngularJointInput {
            orientation_a: Quat::IDENTITY,
            orientation_b: Quat::from_axis_angle(Vec3::X, 0.8),
            angular_velocity_a: Vec3::ZERO,
            angular_velocity_b: Vec3::new(0.0, 2.0, 0.0),
            inv_inertia_a: Vec3::ONE,
            inv_inertia_b: Vec3::ONE,
            dt: 1.0 / 60.0,
        });
        assert!((response.torque_on_a_world + response.torque_on_b_world).length() < 1.0e-6);
        assert!(response.torque_on_a_world.is_finite());
    }

    #[test]
    fn invalid_parallel_reference_is_rejected() {
        let result = SwingTwistJoint::try_new(SwingTwistConfig::symmetric(
            Vec3::X,
            Vec3::X,
            Vec3::X,
            Vec3::Y,
            0.4,
            0.5,
        ));
        assert_eq!(result, Err(SwingTwistConfigError::ReferenceParallelToAxis));
    }
}
