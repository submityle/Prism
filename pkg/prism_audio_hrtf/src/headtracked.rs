//! Head-tracked binaural steering: turn a live head pose into the
//! listener-local direction that drives HRIR selection.
//!
//! A head-mounted display or head tracker reports the listener's head
//! orientation (and often its angular velocity). To keep a virtual source
//! anchored in the world as the head turns, the *world-space* direction to the
//! source must be expressed in the *head-local* frame before choosing and
//! interpolating HRIRs. This module performs that transform and adds the two
//! corrections a low-latency binaural renderer needs:
//!
//! 1. **Pose prediction (motion extrapolation).** Tracker-to-ear latency makes
//!    a source feel "glued to the head" during fast turns. Given an angular
//!    velocity and a short look-ahead time, [`HeadPose::predict`] extrapolates
//!    the orientation with a first-order quaternion step, cancelling part of
//!    that latency.
//! 2. **Orientation smoothing.** Raw pose samples jitter. [`HeadTracker`]
//!    slerps its held orientation toward each predicted target with an
//!    exponential (time-constant) coefficient, so the auditory image tracks
//!    smoothly instead of snapping.
//!
//! The output is a head-local unit direction (or its azimuth/elevation), which
//! feeds the existing [`crate::interpolation::interpolate`] path unchanged.
//! This module adds **no new convolution**; rendering still runs through
//! [`crate::binaural::BinauralRenderer`].
//!
//! # Coordinate convention
//!
//! Matches [`prism_audio_spatial`] and the rest of this crate: right-handed
//! world space (`+X` right, `+Y` up, `-Z` forward). An orientation is a unit
//! quaternion mapping **head-local space into world space**, so a world
//! direction is resolved into the head frame by applying the *inverse*
//! rotation. Azimuth is `atan2(x, -z)` (front `= 0`, right positive) and
//! elevation is `atan2(y, hypot(x, z))`.
//!
//! # Real-time contract
//!
//! [`HeadPose::predict`], [`HeadTracker::update`], and the direction/angle
//! queries are **allocation free, lock free, and panic free**: they run pure
//! quaternion and vector arithmetic on stack values and may be called from the
//! audio or a control-rate thread. Prediction and smoothing parameters are
//! clamped to sane ranges so no input produces a NaN or a runaway step.
//!
//! # Determinism
//!
//! All transcendental math routes through [`bevy_math::ops`] (libm-backed):
//! the prediction delta quaternion is built from [`ops::sin_cos`] and the
//! smoothing coefficient from [`ops::exp`], so head steering is
//! bit-reproducible across targets and golden-comparable.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, or Steam Audio source or derived code**. Quaternion
//! kinematics (axis-angle exponential extrapolation), spherical linear
//! interpolation, exponential time-constant smoothing, and the local
//! azimuth/elevation geometry are implemented from standard, publicly
//! documented rotation and signal-processing knowledge.

use bevy_math::{ops, Quat, Vec3};
use prism_audio_core::math::Sample;

/// Upper bound on the look-ahead used for pose prediction, in seconds.
///
/// Head-tracked audio only ever extrapolates a few tens of milliseconds to
/// hide tracker-to-ear latency; a larger window would overshoot badly on fast
/// turns. Requested prediction times are clamped to `[0, MAX_PREDICTION_SECONDS]`.
pub const MAX_PREDICTION_SECONDS: Sample = 0.1;

/// Angular speeds (rad/s) at or below this are treated as "not rotating": no
/// prediction step is applied, avoiding a divide by a near-zero magnitude when
/// normalising the rotation axis.
const MIN_ANGULAR_SPEED: Sample = 1.0e-6;

/// Direction vectors shorter than this are treated as degenerate and collapse
/// to local forward (`-Z`), matching the coincident-source fallback used
/// elsewhere in the engine.
const MIN_DIRECTION_LEN: Sample = 1.0e-6;

/// Quaternions whose norm falls below this are treated as degenerate and
/// replaced by the identity rotation.
const MIN_QUAT_NORM: Sample = 1.0e-6;

/// A live head pose: orientation plus angular velocity.
///
/// `orientation` is a unit quaternion mapping head-local space into world
/// space (the head faces local `-Z`). `angular_velocity` is a world-frame
/// rotation-rate vector in radians per second: its direction is the rotation
/// axis and its magnitude the angular speed. It drives [`predict`](Self::predict).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HeadPose {
    /// Unit quaternion mapping head-local space into world space.
    pub orientation: Quat,
    /// World-frame angular velocity in radians per second (axis * speed).
    pub angular_velocity: Vec3,
}

impl Default for HeadPose {
    #[inline]
    fn default() -> Self {
        Self {
            orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        }
    }
}

impl HeadPose {
    /// Creates a pose from an `orientation` and world-frame `angular_velocity`.
    ///
    /// The orientation is normalised; a degenerate (near-zero) quaternion
    /// falls back to the identity rotation.
    #[must_use]
    #[inline]
    pub fn new(orientation: Quat, angular_velocity: Vec3) -> Self {
        Self {
            orientation: normalize_quat_or_identity(orientation),
            angular_velocity,
        }
    }

    /// Creates a stationary pose (zero angular velocity) from `orientation`.
    #[must_use]
    #[inline]
    pub fn from_orientation(orientation: Quat) -> Self {
        Self::new(orientation, Vec3::ZERO)
    }

    /// Extrapolates the orientation `seconds` into the future using the
    /// world-frame angular velocity.
    ///
    /// This is a first-order quaternion step: the axis-angle delta
    /// `exp(0.5 * omega * dt)` (built from [`ops::sin_cos`]) is composed on the
    /// world side of the current orientation, `predicted = delta * orientation`.
    /// `seconds` is clamped to `[0, MAX_PREDICTION_SECONDS]`, and an angular
    /// speed at or below [`MIN_ANGULAR_SPEED`] returns the current orientation
    /// unchanged. The result is renormalised.
    #[must_use]
    pub fn predict(&self, seconds: Sample) -> Quat {
        let dt = seconds.clamp(0.0, MAX_PREDICTION_SECONDS);
        let omega = self.angular_velocity;
        let speed = ops::sqrt(omega.dot(omega));
        if dt <= 0.0 || speed <= MIN_ANGULAR_SPEED {
            return self.orientation;
        }
        // Axis-angle exponential map: rotate by (speed * dt) about omega/speed.
        let axis = omega / speed;
        let half_angle = speed * dt * 0.5;
        let (sin_h, cos_h) = ops::sin_cos(half_angle);
        let delta = Quat::from_xyzw(axis.x * sin_h, axis.y * sin_h, axis.z * sin_h, cos_h);
        normalize_quat_or_identity(delta * self.orientation)
    }
}

/// Head-local azimuth/elevation for a resolved direction, in radians.
///
/// Uses the engine convention: `azimuth` is `atan2(x, -z)` (front `= 0`, right
/// positive, `(-pi, pi]`) and `elevation` is `atan2(y, hypot(x, z))`
/// (`[-pi/2, pi/2]`). These feed [`crate::interpolation::interpolate`]
/// directly.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HeadLocalAngles {
    /// Horizontal azimuth in radians (front `= 0`, right positive).
    pub azimuth: Sample,
    /// Elevation in radians above the horizontal plane.
    pub elevation: Sample,
}

/// A control-rate head tracker holding a smoothed, prediction-ready
/// orientation.
///
/// Feed it raw [`HeadPose`] samples with [`update`](Self::update); it
/// extrapolates each sample by [`prediction_seconds`](Self::prediction_seconds)
/// and slerps its held orientation toward that target with an exponential
/// coefficient derived from [`smoothing_time_constant`](Self::smoothing_time_constant).
/// Query the result with [`local_direction`](Self::local_direction) or
/// [`local_angles`](Self::local_angles) and hand it to the HRIR interpolation
/// path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadTracker {
    /// The current smoothed head-local-to-world orientation.
    smoothed: Quat,
    /// Look-ahead used for pose prediction, in seconds (clamped).
    prediction_seconds: Sample,
    /// Exponential smoothing time constant, in seconds (`0` = snap, no smoothing).
    smoothing_time_constant: Sample,
}

impl Default for HeadTracker {
    #[inline]
    fn default() -> Self {
        Self {
            smoothed: Quat::IDENTITY,
            prediction_seconds: 0.0,
            smoothing_time_constant: 0.0,
        }
    }
}

impl HeadTracker {
    /// Creates a tracker seeded with `initial_orientation`.
    ///
    /// `prediction_seconds` is clamped to `[0, MAX_PREDICTION_SECONDS]` and
    /// `smoothing_time_constant` to `>= 0` (a value of `0` disables smoothing,
    /// snapping directly to each predicted pose).
    #[must_use]
    #[inline]
    pub fn new(
        initial_orientation: Quat,
        prediction_seconds: Sample,
        smoothing_time_constant: Sample,
    ) -> Self {
        Self {
            smoothed: normalize_quat_or_identity(initial_orientation),
            prediction_seconds: prediction_seconds.clamp(0.0, MAX_PREDICTION_SECONDS),
            smoothing_time_constant: smoothing_time_constant.max(0.0),
        }
    }

    /// The look-ahead used for pose prediction, in seconds.
    #[must_use]
    #[inline]
    pub fn prediction_seconds(&self) -> Sample {
        self.prediction_seconds
    }

    /// The exponential smoothing time constant, in seconds.
    #[must_use]
    #[inline]
    pub fn smoothing_time_constant(&self) -> Sample {
        self.smoothing_time_constant
    }

    /// Sets the prediction look-ahead, clamped to `[0, MAX_PREDICTION_SECONDS]`.
    #[inline]
    pub fn set_prediction_seconds(&mut self, seconds: Sample) {
        self.prediction_seconds = seconds.clamp(0.0, MAX_PREDICTION_SECONDS);
    }

    /// Sets the smoothing time constant, clamped to `>= 0` (`0` = snap).
    #[inline]
    pub fn set_smoothing_time_constant(&mut self, tau: Sample) {
        self.smoothing_time_constant = tau.max(0.0);
    }

    /// The current smoothed head-local-to-world orientation.
    #[must_use]
    #[inline]
    pub fn orientation(&self) -> Quat {
        self.smoothed
    }

    /// Resets the held orientation to `orientation` (normalised), discarding
    /// any smoothing history.
    #[inline]
    pub fn reset(&mut self, orientation: Quat) {
        self.smoothed = normalize_quat_or_identity(orientation);
    }

    /// Advances the tracker with a raw `pose` sampled `dt` seconds after the
    /// previous update, returning the new smoothed orientation.
    ///
    /// The pose is first extrapolated by [`prediction_seconds`](Self::prediction_seconds),
    /// then the held orientation is slerped toward that target by
    /// `alpha = 1 - exp(-dt / tau)` (a standard exponential/one-pole
    /// smoother). When `tau <= 0` or `dt <= 0` the tracker snaps directly to
    /// the target (`alpha = 1`). The result is renormalised.
    ///
    /// Real-time safe: pure quaternion arithmetic, no allocation, no panic.
    pub fn update(&mut self, pose: HeadPose, dt: Sample) -> Quat {
        let target = pose.predict(self.prediction_seconds);
        let alpha = smoothing_alpha(dt, self.smoothing_time_constant);
        // slerp requires unit inputs; both `smoothed` and `target` are already
        // normalised. Renormalise the blend to keep drift out of the state.
        let blended = self.smoothed.slerp(target, alpha);
        self.smoothed = normalize_quat_or_identity(blended);
        self.smoothed
    }

    /// Resolves a `world_direction` (world-space direction *from* the listener
    /// *to* the source) into the current head-local frame as a unit vector.
    ///
    /// A degenerate (near-zero) input collapses to local forward (`-Z`).
    #[must_use]
    #[inline]
    pub fn local_direction(&self, world_direction: Vec3) -> Vec3 {
        world_to_local_direction(self.smoothed, world_direction)
    }

    /// Resolves a `world_direction` into head-local azimuth/elevation for the
    /// HRIR interpolation path.
    #[must_use]
    #[inline]
    pub fn local_angles(&self, world_direction: Vec3) -> HeadLocalAngles {
        angles_of(self.local_direction(world_direction))
    }
}

/// Resolves a world-space direction into a head-local unit direction for the
/// given `orientation` (head-local-to-world), applying the inverse rotation.
///
/// A degenerate (near-zero) `world_direction` collapses to local forward
/// (`-Z`). This is the stateless core used by [`HeadTracker::local_direction`].
#[must_use]
#[inline]
pub fn world_to_local_direction(orientation: Quat, world_direction: Vec3) -> Vec3 {
    let len_sq = world_direction.dot(world_direction);
    let len = ops::sqrt(len_sq);
    if len <= MIN_DIRECTION_LEN {
        return Vec3::NEG_Z;
    }
    let world_unit = world_direction / len;
    orientation.inverse() * world_unit
}

/// Resolves a world-space source direction directly into head-local
/// azimuth/elevation for `pose`, applying `prediction_seconds` of motion
/// extrapolation first (stateless, no smoothing).
///
/// This is the one-shot equivalent of predicting `pose` and calling
/// [`HeadTracker::local_angles`]; use it when no temporal smoothing is wanted.
///
/// # Examples
///
/// A head yawed 90 degrees to the left (about `+Y`) places a world-front
/// source on the right ear:
///
/// ```
/// use bevy_math::{Quat, Vec3};
/// use core::f32::consts::FRAC_PI_2;
/// use prism_audio_hrtf::headtracked::{HeadPose, predicted_local_angles};
///
/// let pose = HeadPose::from_orientation(Quat::from_rotation_y(FRAC_PI_2));
/// let angles = predicted_local_angles(pose, Vec3::NEG_Z, 0.0);
/// assert!((angles.azimuth - FRAC_PI_2).abs() < 1e-4); // source is on the right
/// assert!(angles.elevation.abs() < 1e-4);             // still on the horizon
/// ```
#[must_use]
#[inline]
pub fn predicted_local_angles(
    pose: HeadPose,
    world_direction: Vec3,
    prediction_seconds: Sample,
) -> HeadLocalAngles {
    let predicted = pose.predict(prediction_seconds);
    angles_of(world_to_local_direction(predicted, world_direction))
}

/// Head-local azimuth in radians for a local direction: `atan2(x, -z)`.
#[must_use]
#[inline]
pub fn local_azimuth(local_direction: Vec3) -> Sample {
    ops::atan2(local_direction.x, -local_direction.z)
}

/// Head-local elevation in radians for a local direction:
/// `atan2(y, hypot(x, z))`.
#[must_use]
#[inline]
pub fn local_elevation(local_direction: Vec3) -> Sample {
    let horizontal =
        ops::sqrt(local_direction.x * local_direction.x + local_direction.z * local_direction.z);
    ops::atan2(local_direction.y, horizontal)
}

/// Packs a local direction into its azimuth/elevation angles.
#[inline]
fn angles_of(local_direction: Vec3) -> HeadLocalAngles {
    HeadLocalAngles {
        azimuth: local_azimuth(local_direction),
        elevation: local_elevation(local_direction),
    }
}

/// Exponential (one-pole) smoothing coefficient `1 - exp(-dt / tau)`.
///
/// Returns `1.0` (snap) when `tau` or `dt` is non-positive, and is clamped to
/// `[0, 1]`. Routes the exponential through [`ops::exp`] for determinism.
#[inline]
fn smoothing_alpha(dt: Sample, tau: Sample) -> Sample {
    if tau <= 0.0 || dt <= 0.0 {
        return 1.0;
    }
    let alpha = 1.0 - ops::exp(-dt / tau);
    alpha.clamp(0.0, 1.0)
}

/// Normalises a quaternion, falling back to identity when its norm is
/// degenerate. Deterministic (routes the norm through [`ops::sqrt`]).
#[must_use]
#[inline]
fn normalize_quat_or_identity(q: Quat) -> Quat {
    let norm_sq = q.dot(q);
    let norm = ops::sqrt(norm_sq);
    if norm <= MIN_QUAT_NORM {
        return Quat::IDENTITY;
    }
    let inv = 1.0 / norm;
    Quat::from_xyzw(q.x * inv, q.y * inv, q.z * inv, q.w * inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, PI};

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        ops::abs(a - b) <= eps
    }

    #[test]
    fn zero_head_motion_is_identity() {
        // Identity orientation, no prediction, no smoothing => local == world.
        let mut tracker = HeadTracker::new(Quat::IDENTITY, 0.0, 0.0);
        tracker.update(HeadPose::default(), 0.01);
        let local = tracker.local_direction(Vec3::NEG_Z);
        assert!(approx(local.x, 0.0, 1e-6));
        assert!(approx(local.y, 0.0, 1e-6));
        assert!(approx(local.z, -1.0, 1e-6));
        let a = tracker.local_angles(Vec3::NEG_Z);
        assert!(approx(a.azimuth, 0.0, 1e-5));
        assert!(approx(a.elevation, 0.0, 1e-5));
    }

    #[test]
    fn yaw_90_moves_front_source_to_right() {
        // Head yawed +90 about +Y (local->world). A world-front source (-Z)
        // should land on the right ear (+X, azimuth +pi/2) in the head frame.
        let mut tracker = HeadTracker::new(Quat::from_rotation_y(FRAC_PI_2), 0.0, 0.0);
        tracker.update(
            HeadPose::from_orientation(Quat::from_rotation_y(FRAC_PI_2)),
            0.01,
        );
        let local = tracker.local_direction(Vec3::NEG_Z);
        assert!(approx(local.x, 1.0, 1e-5));
        assert!(approx(local.z, 0.0, 1e-5));
        let a = tracker.local_angles(Vec3::NEG_Z);
        assert!(approx(a.azimuth, FRAC_PI_2, 1e-4));
    }

    #[test]
    fn overhead_source_has_positive_elevation() {
        let tracker = HeadTracker::new(Quat::IDENTITY, 0.0, 0.0);
        let a = tracker.local_angles(Vec3::Y);
        assert!(approx(a.elevation, FRAC_PI_2, 1e-5));
    }

    #[test]
    fn angular_velocity_extrapolation_moves_azimuth_right() {
        // Positive yaw rate about +Y: predicting forward turns the head left,
        // so a world-front source moves to positive (right) azimuth. The shift
        // must grow with the look-ahead time.
        let pose = HeadPose::new(Quat::IDENTITY, Vec3::new(0.0, 1.0, 0.0));
        let a_none = predicted_local_angles(pose, Vec3::NEG_Z, 0.0);
        let a_small = predicted_local_angles(pose, Vec3::NEG_Z, 0.02);
        let a_large = predicted_local_angles(pose, Vec3::NEG_Z, 0.05);
        assert!(approx(a_none.azimuth, 0.0, 1e-6));
        assert!(a_small.azimuth > 0.0);
        assert!(a_large.azimuth > a_small.azimuth);
        // 1 rad/s for 0.05 s => ~0.05 rad azimuth shift.
        assert!(approx(a_large.azimuth, 0.05, 2e-3));
    }

    #[test]
    fn prediction_look_ahead_is_clamped() {
        // A huge requested look-ahead must clamp to MAX_PREDICTION_SECONDS.
        let pose = HeadPose::new(Quat::IDENTITY, Vec3::new(0.0, 1.0, 0.0));
        let clamped = predicted_local_angles(pose, Vec3::NEG_Z, 100.0);
        let at_max = predicted_local_angles(pose, Vec3::NEG_Z, MAX_PREDICTION_SECONDS);
        assert!(approx(clamped.azimuth, at_max.azimuth, 1e-6));
        // Setter clamps too.
        let mut tracker = HeadTracker::new(Quat::IDENTITY, 100.0, -5.0);
        assert!(approx(
            tracker.prediction_seconds(),
            MAX_PREDICTION_SECONDS,
            1e-9
        ));
        assert!(approx(tracker.smoothing_time_constant(), 0.0, 1e-9));
        tracker.set_prediction_seconds(-1.0);
        assert!(approx(tracker.prediction_seconds(), 0.0, 1e-9));
    }

    #[test]
    fn slerp_smoothing_monotonically_approaches_target() {
        // Start at identity, target a +90 yaw with a non-zero time constant.
        // Repeated fixed-dt updates must move the held orientation strictly
        // closer to the target (increasing quaternion dot) and converge.
        let target = Quat::from_rotation_y(FRAC_PI_2);
        let mut tracker = HeadTracker::new(Quat::IDENTITY, 0.0, 0.1);
        let pose = HeadPose::from_orientation(target);
        let mut prev = ops::abs(tracker.orientation().dot(target));
        for _ in 0..20 {
            let q = tracker.update(pose, 0.02);
            let d = ops::abs(q.dot(target));
            assert!(d >= prev - 1e-6);
            prev = d;
        }
        assert!(prev > 0.99);
    }

    #[test]
    fn smoothing_disabled_snaps_immediately() {
        let target = Quat::from_rotation_y(FRAC_PI_2);
        let mut tracker = HeadTracker::new(Quat::IDENTITY, 0.0, 0.0);
        let q = tracker.update(HeadPose::from_orientation(target), 0.02);
        // Snap: one update reaches the target (dot ~ 1).
        assert!(approx(ops::abs(q.dot(target)), 1.0, 1e-5));
    }

    #[test]
    fn degenerate_direction_falls_back_to_forward() {
        let tracker = HeadTracker::new(Quat::from_rotation_y(0.7), 0.0, 0.0);
        let local = tracker.local_direction(Vec3::ZERO);
        assert!(approx(local.x, 0.0, 1e-6));
        assert!(approx(local.y, 0.0, 1e-6));
        assert!(approx(local.z, -1.0, 1e-6));
    }

    #[test]
    fn non_unit_pose_is_normalized() {
        // A scaled quaternion must be normalised to a valid rotation.
        let pose = HeadPose::new(Quat::from_xyzw(0.0, 0.0, 0.0, 4.0), Vec3::ZERO);
        assert!(approx(
            ops::abs(pose.orientation.dot(Quat::IDENTITY)),
            1.0,
            1e-6
        ));
        // A near-zero quaternion falls back to identity without producing NaN.
        let degenerate = HeadPose::from_orientation(Quat::from_xyzw(0.0, 0.0, 0.0, 0.0));
        let local = world_to_local_direction(degenerate.orientation, Vec3::NEG_Z);
        assert!(local.is_finite());
        assert!(approx(local.z, -1.0, 1e-6));
    }

    #[test]
    fn zero_angular_velocity_prediction_is_noop() {
        let pose = HeadPose::new(Quat::from_rotation_y(0.3), Vec3::ZERO);
        let predicted = pose.predict(MAX_PREDICTION_SECONDS);
        assert!(approx(ops::abs(predicted.dot(pose.orientation)), 1.0, 1e-6));
    }

    #[test]
    fn results_are_deterministic() {
        let pose = HeadPose::new(Quat::from_rotation_y(0.4), Vec3::new(0.1, 0.5, -0.2));
        let a = predicted_local_angles(pose, Vec3::new(1.0, 0.2, -3.0), 0.03);
        let b = predicted_local_angles(pose, Vec3::new(1.0, 0.2, -3.0), 0.03);
        assert!(approx(a.azimuth, b.azimuth, 0.0));
        assert!(approx(a.elevation, b.elevation, 0.0));
    }

    #[test]
    fn azimuth_stays_in_range() {
        // A source behind and to the left should yield azimuth in (-pi, pi].
        let tracker = HeadTracker::new(Quat::IDENTITY, 0.0, 0.0);
        let a = tracker.local_angles(Vec3::new(-1.0, 0.0, 1.0));
        assert!(a.azimuth > -PI && a.azimuth <= PI);
        assert!(a.azimuth < 0.0); // left => negative azimuth
    }

    #[test]
    fn feeds_hrir_interpolation_path() {
        // End-to-end: head-local angles drive the existing interpolation path.
        use crate::dataset::{HrtfDataset, Measurement};
        use crate::interpolation::interpolate;
        use alloc::vec;

        let hrir_len = 4;
        let measurements = vec![
            Measurement::new(-FRAC_PI_2, 0.0, 1.0),
            Measurement::new(0.0, 0.0, 1.0),
            Measurement::new(FRAC_PI_2, 0.0, 1.0),
        ];
        // Distinct impulses per measurement so blending is observable.
        let mut left = vec![0.0 as Sample; measurements.len() * hrir_len];
        let mut right = vec![0.0 as Sample; measurements.len() * hrir_len];
        for m in 0..measurements.len() {
            left[m * hrir_len] = 1.0;
            right[m * hrir_len + 1] = 1.0;
        }
        let dataset = HrtfDataset::from_samples(48_000, hrir_len, measurements, left, right)
            .expect("valid dataset");

        // Head yawed +90; a world-front source lands at azimuth +pi/2.
        let tracker = HeadTracker::new(Quat::from_rotation_y(FRAC_PI_2), 0.0, 0.0);
        let angles = tracker.local_angles(Vec3::NEG_Z);
        let mut out_l = [0.0 as Sample; 4];
        let mut out_r = [0.0 as Sample; 4];
        let info = interpolate(
            &dataset,
            angles.azimuth,
            angles.elevation,
            &mut out_l,
            &mut out_r,
        )
        .expect("interpolation succeeds");
        assert!(info.neighbors_used >= 1);
    }
}
