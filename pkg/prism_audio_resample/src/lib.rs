//! Decoupled resampling and time/pitch scaling primitives for Prism's audio
//! engine.
//!
//! Pitch, duration, play rate, and Doppler are four independently controllable
//! quantities; naive engines couple them all onto "change the sample read
//! step", which makes any speed change also shift pitch and makes Doppler click.
//! This crate provides the decoupled, high-quality classic-DSP primitives from
//! section 34 of the engine design doc:
//!
//! - A graded [`Resampler`] trait for sample-rate conversion and
//!   arbitrary/continuously-varying-ratio playback, with three quality grades
//!   on a performance-governed ladder: [`LinearResampler`] (far-field / LOD),
//!   [`PolyphaseSincResampler`] (the default polyphase windowed-sinc grade), and
//!   [`HighOrderSincResampler`] (mastering / near-field). Coefficients are
//!   precomputed at construction; the ratio may sweep continuously with phase
//!   continuity (no clicks), which is what Doppler and play-rate automation
//!   need.
//! - A [`TimeStretcher`] trait for pitch-preserving time scaling and
//!   time-preserving pitch scaling, with the stretch factor independent of the
//!   pitch ratio: [`WsolaStretcher`] (low-overhead WSOLA/SOLA, voice/SFX grade)
//!   and [`PhaseVocoderStretcher`] (STFT phase propagation, music grade).
//!
//! All three families (sample-rate conversion, time stretching, and the
//! continuously-varying delay line used elsewhere for Doppler) share the one
//! fractional-delay interpolation kernel in [`fractional_delay`].
//!
//! # Provenance
//!
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code, and no AI/ML.
//! Classic DSP only.
//!
//! # Relationship
//!
//! Downstream of [`prism_audio_core`]; reuses its [`Sample`](prism_audio_core::math::Sample)
//! scalar and its [`Fft`](prism_audio_core::fft::Fft) plan. Provides the
//! `Resampler` and `TimeStretcher` extension primitives described in the engine
//! design doc, which upstream nodes (Doppler, the sampler, interactive-music
//! beat matching) compose.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod fractional_delay;
pub mod high_order_sinc;
pub mod linear;
pub mod phase_vocoder;
pub mod polyphase_sinc;
pub mod resampler;
pub mod time_stretcher;
pub mod wsola;

pub use fractional_delay::{
    EPS, SincInterpolator, close, linear_interp, sanitize, windowed_sinc,
};
pub use high_order_sinc::HighOrderSincResampler;
pub use linear::LinearResampler;
pub use phase_vocoder::PhaseVocoderStretcher;
pub use polyphase_sinc::PolyphaseSincResampler;
pub use resampler::{
    MAX_RATIO, MIN_RATIO, ResampleProgress, ResampleQuality, Resampler, clamp_ratio,
};
pub use time_stretcher::{
    MAX_PITCH, MAX_STRETCH, MIN_PITCH, MIN_STRETCH, StretchProgress, TimeStretcher,
    clamp_pitch, clamp_stretch, semitones_to_ratio,
};
pub use wsola::WsolaStretcher;
