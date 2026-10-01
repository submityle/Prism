//! Non-destructive measurement taps for metering and compliance.
//!
//! The processors in this submodule observe a signal and report statistics
//! about it without altering the samples that flow through them. They are the
//! analysis half of a mastering and compliance workflow: a mix is shaped by
//! the [`dynamics`](crate::nodes::dynamics) processors and then verified
//! against a loudness target with the meters here.
//!
//! # Analysis catalogue
//!
//! - [`loudness::LoudnessMeter`] / [`loudness::LoudnessMeterNode`] --
//!   `ITU-R` `BS.1770-4` / `EBU` `R128` momentary, short-term, and gated
//!   integrated loudness in `LUFS`, loudness range (`LRA`) in `LU`, and
//!   true-peak level in `dBTP`.
//! - [`loudness::KWeighting`] -- the two-stage `K-weighting` perceptual
//!   pre-filter used by the loudness measurement.
//! - [`loudness::TruePeakMeter`] -- the 4x oversampling polyphase inter-sample
//!   peak estimator used for the `dBTP` reading.
//! - [`correlation::CorrelationMeter`] / [`correlation::CorrelationMeterNode`]
//!   -- stereo phase correlation, mid/side width, left/right balance, and
//!   mid/side `RMS` levels for stereo-field and mono-compatibility checks.
//! - [`goniometer::Goniometer`] / [`goniometer::GoniometerNode`] -- a
//!   vectorscope coordinate generator that emits rotated mid/side (X/Y) point
//!   clouds for Lissajous stereo-field visualisation, complementing the scalar
//!   readings of the [`correlation`] meter.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is an
//! original organisational module that aggregates measurement nodes
//! implemented from publicly published broadcast standards. It is pure classic
//! DSP with no AI/ML.
//!
//! # Relationship
//!
//! This submodule groups the measurement (read-only) counterparts to the
//! signal-altering [`dynamics`](crate::nodes::dynamics) processors. The nodes
//! here reuse the shared filter primitives in
//! [`biquad`](crate::nodes::biquad) rather than re-implementing them.

pub mod correlation;
pub mod goniometer;
pub mod loudness;

pub use correlation::{CorrelationMeasurement, CorrelationMeter, CorrelationMeterNode};
pub use goniometer::{
    DEFAULT_DECIMATION, DEFAULT_POINT_CAPACITY, Goniometer, GoniometerNode, GoniometerPoint,
    GoniometerStats, MIN_POINT_CAPACITY,
};
pub use loudness::{
    KWeighting, LoudnessMeasurement, LoudnessMeter, LoudnessMeterNode, TruePeakMeter,
};
