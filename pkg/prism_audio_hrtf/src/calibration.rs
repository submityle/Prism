//! Runtime HRTF calibration from user elevation-cue feedback (classic, no ML).
//!
//! Even the best-matched measured HRTF (see [`crate::personalization`]) leaves
//! a residual front/back and elevation bias per listener. Consumer systems
//! address this with guided personalization: the listener is shown (or played)
//! an elevation cue and nudges it until it sounds correct. This module
//! captures the **classic** form of that loop - a deterministic feedback
//! integrator - and the resulting locked bias, with **no machine learning**.
//!
//! # Model
//!
//! There is an unknown true elevation offset `O` between where the listener
//! perceives a cue and where it is rendered. Each round the listener reports
//! the currently perceived error (perceived minus intended elevation). The
//! [`CalibrationState`] integrates that error toward a stable bias:
//!
//! ```text
//! bias_{k+1} = bias_k + step * reported_error_k
//! ```
//!
//! For `0 < step <= 1` this is a standard first-order integrator that
//! converges to `O`. The per-step change magnitude drives
//! [`CalibrationState::is_converged`]; once converged the state is frozen into
//! an [`ElevationCalibration`] and applied to query directions via
//! [`ElevationCalibration::apply_to`].
//!
//! A matching gain-feedback channel captures a scalar level trim in decibels
//! using the same integrator.
//!
//! # Real-time contract
//!
//! Feedback accumulation runs off the audio thread (it is driven by UI).
//! [`ElevationCalibration::apply_to`] is allocation-free and panic-free and may
//! run on the audio thread when a source direction changes.
//!
//! # Determinism
//!
//! Only clamping and a square root (for the convergence norm) are used, routed
//! through [`bevy_math::ops`], so calibration is bit-reproducible.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, or Steam Audio
//! source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Produces an [`ElevationCalibration`] whose [`ElevationCalibration::apply_to`]
//! adjusts the azimuth/elevation passed to [`crate::interpolation`] /
//! [`crate::dataset`]. Pairs with [`crate::personalization`]: pick the nearest
//! dataset first, then calibrate the residual elevation bias.

use bevy_math::ops;
use core::f32::consts::FRAC_PI_2;
use prism_audio_core::math::Sample;

/// Default integrator step (learning rate) for the feedback loop.
pub const DEFAULT_STEP: Sample = 0.5;

/// Default convergence threshold on the per-step elevation change, radians.
pub const DEFAULT_ELEVATION_EPSILON: Sample = 0.5_f32 * core::f32::consts::PI / 180.0;

/// A locked calibration result: a fixed elevation and gain bias.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ElevationCalibration {
    /// Elevation bias in radians, added to query elevation before lookup.
    pub elevation_bias: Sample,
    /// Gain trim in decibels (informational; applied by the caller's gain
    /// stage).
    pub gain_bias_db: Sample,
}

impl ElevationCalibration {
    /// An identity calibration (no elevation or gain change).
    pub const IDENTITY: Self = Self {
        elevation_bias: 0.0,
        gain_bias_db: 0.0,
    };

    /// Creates a calibration from an `elevation_bias` (radians) and
    /// `gain_bias_db` (decibels).
    #[must_use]
    #[inline]
    pub const fn new(elevation_bias: Sample, gain_bias_db: Sample) -> Self {
        Self {
            elevation_bias,
            gain_bias_db,
        }
    }

    /// Applies the elevation bias to a query direction.
    ///
    /// Azimuth is returned unchanged; elevation is biased and clamped to the
    /// valid `[-pi/2, pi/2]` range. Allocation-free and panic-free.
    #[must_use]
    #[inline]
    pub fn apply_to(&self, azimuth: Sample, elevation: Sample) -> (Sample, Sample) {
        let el = (elevation + self.elevation_bias).clamp(-FRAC_PI_2, FRAC_PI_2);
        (azimuth, el)
    }
}

impl Default for ElevationCalibration {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A running elevation/gain calibration driven by listener feedback.
///
/// Feed perceived-error reports via [`record_elevation_feedback`] (and
/// optionally [`record_gain_feedback`]); poll [`is_converged`]; then [`lock`]
/// into an [`ElevationCalibration`].
///
/// [`record_elevation_feedback`]: CalibrationState::record_elevation_feedback
/// [`record_gain_feedback`]: CalibrationState::record_gain_feedback
/// [`is_converged`]: CalibrationState::is_converged
/// [`lock`]: CalibrationState::lock
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CalibrationState {
    elevation_bias: Sample,
    gain_bias_db: Sample,
    step: Sample,
    last_elevation_delta: Sample,
    rounds: u32,
}

impl CalibrationState {
    /// Creates a fresh state with [`DEFAULT_STEP`].
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::with_step(DEFAULT_STEP)
    }

    /// Creates a fresh state with a custom integrator `step`.
    ///
    /// `step` is clamped to `(0, 1]`; values outside that range do not form a
    /// convergent first-order integrator.
    #[must_use]
    #[inline]
    pub fn with_step(step: Sample) -> Self {
        Self {
            elevation_bias: 0.0,
            gain_bias_db: 0.0,
            step: step.clamp(Sample::MIN_POSITIVE, 1.0),
            last_elevation_delta: Sample::INFINITY,
            rounds: 0,
        }
    }

    /// The current elevation bias (radians).
    #[must_use]
    #[inline]
    pub fn elevation_bias(&self) -> Sample {
        self.elevation_bias
    }

    /// The current gain bias (decibels).
    #[must_use]
    #[inline]
    pub fn gain_bias_db(&self) -> Sample {
        self.gain_bias_db
    }

    /// The number of feedback rounds integrated so far.
    #[must_use]
    #[inline]
    pub fn rounds(&self) -> u32 {
        self.rounds
    }

    /// Integrates one elevation-error report (radians, perceived minus
    /// intended) and returns the updated bias.
    ///
    /// The reported error is clamped to a sane `[-pi/2, pi/2]` range before
    /// integration.
    #[inline]
    pub fn record_elevation_feedback(&mut self, reported_error: Sample) -> Sample {
        let error = reported_error.clamp(-FRAC_PI_2, FRAC_PI_2);
        let delta = self.step * error;
        self.elevation_bias = (self.elevation_bias + delta).clamp(-FRAC_PI_2, FRAC_PI_2);
        self.last_elevation_delta = delta;
        self.rounds += 1;
        self.elevation_bias
    }

    /// Integrates one gain-error report (decibels, perceived minus intended)
    /// and returns the updated gain bias.
    #[inline]
    pub fn record_gain_feedback(&mut self, reported_error_db: Sample) -> Sample {
        self.gain_bias_db += self.step * reported_error_db;
        self.gain_bias_db
    }

    /// Returns `true` once the last elevation step is within `epsilon` radians.
    ///
    /// Requires at least one round so an untouched state is never reported as
    /// converged.
    #[must_use]
    #[inline]
    pub fn is_converged(&self, epsilon: Sample) -> bool {
        self.rounds > 0 && ops::abs(self.last_elevation_delta) <= epsilon
    }

    /// Freezes the current biases into an [`ElevationCalibration`].
    #[must_use]
    #[inline]
    pub fn lock(&self) -> ElevationCalibration {
        ElevationCalibration::new(self.elevation_bias, self.gain_bias_db)
    }

    /// Applies the current (unlocked) elevation bias to a query direction, the
    /// same way the locked [`ElevationCalibration::apply_to`] would.
    #[must_use]
    #[inline]
    pub fn apply_to(&self, azimuth: Sample, elevation: Sample) -> (Sample, Sample) {
        self.lock().apply_to(azimuth, elevation)
    }
}

impl Default for CalibrationState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_PI_4;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn identity_leaves_direction_unchanged() {
        let cal = ElevationCalibration::IDENTITY;
        let (az, el) = cal.apply_to(0.3, 0.2);
        assert!(approx(az, 0.3, 1e-6));
        assert!(approx(el, 0.2, 1e-6));
    }

    #[test]
    fn apply_biases_elevation_only() {
        let cal = ElevationCalibration::new(0.1, 0.0);
        let (az, el) = cal.apply_to(0.3, 0.2);
        assert!(approx(az, 0.3, 1e-6));
        assert!(approx(el, 0.3, 1e-6));
    }

    #[test]
    fn apply_clamps_elevation_to_pole() {
        let cal = ElevationCalibration::new(1.0, 0.0);
        let (_, el) = cal.apply_to(0.0, FRAC_PI_4);
        assert!(approx(el, FRAC_PI_2, 1e-6));
    }

    #[test]
    fn integrator_converges_to_true_offset() {
        // True offset O = 0.2 rad; each round the user reports the residual.
        let true_offset = 0.2_f32;
        let mut state = CalibrationState::with_step(0.5);
        for _ in 0..40 {
            let residual = true_offset - state.elevation_bias();
            state.record_elevation_feedback(residual);
        }
        assert!(state.is_converged(DEFAULT_ELEVATION_EPSILON));
        assert!(approx(state.elevation_bias(), true_offset, 1e-3));
        let locked = state.lock();
        assert!(approx(locked.elevation_bias, true_offset, 1e-3));
    }

    #[test]
    fn single_step_moves_by_step_times_error() {
        let mut state = CalibrationState::with_step(0.5);
        let bias = state.record_elevation_feedback(0.4);
        // 0.5 * 0.4 = 0.2.
        assert!(approx(bias, 0.2, 1e-6));
        assert_eq!(state.rounds(), 1);
    }

    #[test]
    fn fresh_state_is_not_converged() {
        let state = CalibrationState::new();
        assert!(!state.is_converged(DEFAULT_ELEVATION_EPSILON));
    }

    #[test]
    fn converges_when_step_is_small() {
        let mut state = CalibrationState::with_step(0.5);
        // A tiny residual gives a tiny delta -> converged flag set.
        state.record_elevation_feedback(0.001);
        assert!(state.is_converged(DEFAULT_ELEVATION_EPSILON));
    }

    #[test]
    fn step_is_clamped_into_unit_range() {
        let hi = CalibrationState::with_step(5.0);
        let delta = {
            let mut s = hi;
            s.record_elevation_feedback(0.2)
        };
        // step clamped to 1.0 -> full error applied.
        assert!(approx(delta, 0.2, 1e-6));
    }

    #[test]
    fn gain_feedback_integrates() {
        let mut state = CalibrationState::with_step(0.5);
        state.record_gain_feedback(2.0);
        state.record_gain_feedback(2.0);
        // 0.5*2 + 0.5*2 = 2.0.
        assert!(approx(state.gain_bias_db(), 2.0, 1e-6));
        assert!(approx(state.lock().gain_bias_db, 2.0, 1e-6));
    }

    #[test]
    fn state_apply_matches_locked_apply() {
        let mut state = CalibrationState::with_step(0.5);
        state.record_elevation_feedback(0.3);
        let a = state.apply_to(0.1, 0.0);
        let b = state.lock().apply_to(0.1, 0.0);
        assert!(approx(a.0, b.0, 1e-6));
        assert!(approx(a.1, b.1, 1e-6));
    }

    #[test]
    fn reported_error_is_clamped() {
        let mut state = CalibrationState::with_step(1.0);
        // Absurd report is clamped to pi/2 before integration.
        let bias = state.record_elevation_feedback(100.0);
        assert!(approx(bias, FRAC_PI_2, 1e-6));
    }
}
