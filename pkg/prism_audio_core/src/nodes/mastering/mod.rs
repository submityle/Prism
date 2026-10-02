//! Delivery-stage processors that render a finished mix to its target format.
//!
//! The processors in this submodule run at the very end of a mastering chain,
//! after the signal has been shaped by the [`dynamics`](crate::nodes::dynamics)
//! processors and verified against a loudness target with the
//! [`analysis`](crate::nodes::analysis) meters. Their job is not to shape tone
//! or dynamics but to translate the high-resolution float mix into the finite
//! numeric format a delivery medium requires, as transparently as the format
//! allows.
//!
//! # Mastering catalogue
//!
//! - [`dither::Dither`] / [`dither::DitherNode`] -- a mastering-grade
//!   requantizer that reduces bit depth using rectangular (`RPDF`) or
//!   triangular (`TPDF`) dither plus optional first- or second-order
//!   error-feedback noise shaping, decorrelating the quantization error into a
//!   steady, perceptually weighted noise floor instead of signal-correlated
//!   distortion.
//! - [`mastering_chain::MasteringChainNode`] -- a fixed-order mastering chain
//!   that orchestrates the parametric EQ, compressor, limiter, and dither
//!   nodes in the canonical delivery signal order, with per-stage bypass
//!   switches. It implements no DSP of its own and simply sequences the
//!   existing processors.
//! - [`loudness_normalizer::LoudnessNormalizerNode`] -- a target-loudness
//!   normalizer that applies the static makeup gain implied by a measured
//!   integrated loudness and a delivery target, bounded by a true-peak
//!   ceiling. It applies gain only and performs no metering of its own.
//! - [`true_peak_limiter::TruePeakLimiter`] /
//!   [`true_peak_limiter::TruePeakLimiterNode`] -- an inter-sample-peak
//!   (true-peak) brickwall limiter following the ITU-R BS.1770 / EBU R128
//!   true-peak estimation: it oversamples (4x by default) to detect
//!   reconstructed inter-sample peaks and applies look-ahead,
//!   program-dependent gain reduction so the reconstructed true peak never
//!   exceeds a configurable ceiling (default -1.0 dBTP). It reports its
//!   look-ahead as processing latency.
//! - [`hdr::HdrWindow`] / [`hdr::HdrNode`] -- a classic HDR-audio dynamic
//!   window: a windowed loudness/max estimator tracks the loudest recent
//!   signal and derives a broadband makeup gain that maps a wide input
//!   dynamic range into a bounded output window (quiet passages lifted,
//!   loud passages attenuated), with a configurable window width (dB),
//!   attack/release, and output target.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is an
//! original organisational module that aggregates delivery-stage processors
//! implemented from publicly published quantization theory. It is pure classic
//! DSP with no AI/ML.
//!
//! # Relationship
//!
//! This submodule is the signal-altering output counterpart to the read-only
//! [`analysis`](crate::nodes::analysis) meters: the meters verify loudness and
//! true peak, and the processors here render the verified mix down to its
//! delivery bit depth. The requantizer is deliberately distinct from the lo-fi
//! [`BitcrusherNode`](crate::nodes::effects::BitcrusherNode) effect, which
//! degrades a signal for character rather than minimising audible loss.

pub mod dither;
pub mod hdr;
pub mod loudness_normalizer;
pub mod mastering_chain;
pub mod true_peak_limiter;

pub use dither::{
    Dither, DitherNode, DitherParams, DitherType, NoiseShaping, DEFAULT_DITHER_BITS,
    MAX_DITHER_BITS, MIN_DITHER_BITS,
};
pub use loudness_normalizer::{
    normalization_gain_db, LoudnessNormalizerNode, LoudnessNormalizerParams, DEFAULT_MAX_GAIN_DB,
    DEFAULT_MAX_TRUE_PEAK_DBTP, DEFAULT_RAMP_SECONDS, DEFAULT_TARGET_LUFS, SILENCE_GATE_LUFS,
};
pub use mastering_chain::{MasteringChainNode, MasteringChainParams};
pub use hdr::{
    HdrNode, HdrParams, HdrWindow, DEFAULT_ATTACK_MS as DEFAULT_HDR_ATTACK_MS,
    DEFAULT_RELEASE_MS as DEFAULT_HDR_RELEASE_MS, DEFAULT_TARGET_DB as DEFAULT_HDR_TARGET_DB,
    DEFAULT_WINDOW_DB as DEFAULT_HDR_WINDOW_DB,
};
pub use true_peak_limiter::{
    TruePeakLimiter, TruePeakLimiterNode, TruePeakLimiterParams, DEFAULT_CEILING_DBTP,
    DEFAULT_LOOKAHEAD_MS, DEFAULT_OVERSAMPLE_FACTOR,
    DEFAULT_RELEASE_MS as DEFAULT_TRUE_PEAK_RELEASE_MS,
};
