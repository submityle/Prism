//! Signal-generating source nodes (0-input, N-output).
//!
//! Unlike the insert-style processors in [`super::effects`] and
//! [`super::dynamics`], every node here *originates* a signal rather than
//! transforming an upstream one, so it declares no input ports and writes its
//! output buffer from internal state. They are the primitive voices the
//! authoring and voice-management layers instantiate and route into the mix
//! graph.
//!
//! # Source catalogue
//!
//! - [`additive_oscillator::AdditiveOscillatorNode`] -- real-time additive
//!   (Fourier-series) oscillator: a bank of phase-locked harmonic sines whose
//!   per-partial gains are audio-rate controls, Nyquist-muted and sum-normalized
//!   so timbres can be morphed live rather than baked into a table.
//! - [`oscillator::OscillatorNode`] -- band-limited (`PolyBLEP`) sine/saw/square/
//!   triangle geometric oscillator selected by [`oscillator::Waveform`].
//! - [`pwm_oscillator::PwmOscillatorNode`] -- band-limited (`PolyBLEP`) pulse
//!   oscillator with a continuously smoothed, audio-rate duty cycle: the
//!   classic pulse-width-modulation animation. It reuses the oscillator's
//!   `PolyBLEP` edge correction but places the second (falling) edge at the
//!   variable width, generalizing the fixed 50%-duty square.
//! - [`karplus_strong::KarplusStrongNode`] -- extended Karplus-Strong
//!   plucked-string physical model: a noise burst recirculating through a tuned,
//!   damped feedback delay line with an allpass fractional-delay tuning filter.
//! - [`noise::NoiseNode`] -- deterministic white/pink/brown generator
//!   ([`noise::NoiseColor`]) built on a reproducible `xorshift64`/`SplitMix64`
//!   stream with Paul-Kellet pink shaping and a leaky-integrator brown filter.
//! - [`sample_player::SamplePlayerNode`] -- pitch/rate-resampling PCM player with
//!   [`sample_player::LoopMode`] loop points and selectable
//!   [`sample_player::Interpolation`] (linear / Catmull-Rom).
//! - [`wavetable_oscillator::WavetableOscillatorNode`] -- band-limited mipmap
//!   wavetable oscillator: an octave mipmap of additively synthesized tables
//!   (saw/square/triangle presets or an arbitrary harmonic spectrum) read with
//!   periodic Catmull-Rom interpolation.
//! - [`fm_operator::FmOperatorNode`] -- phase-modulation (DX7-style)
//!   operator: a sine core deflected by an optional modulation input and
//!   two-sample-averaged self-feedback, the primitive voice of FM
//!   synthesis algorithms assembled in the mix graph.
//! - [`supersaw::SupersawNode`] -- detuned saw-stack ("super saw"): seven
//!   `PolyBLEP` band-limited sawtooths spread around one fundamental by an
//!   equal-temperament detune control and blended center-vs-sides by a `mix`
//!   control, the lush JP-8000-style unison lead/pad voice.
//!
//! Every generator is real-time safe: `process` performs no allocation, no
//! locking, and no panics, and reproducible generators are fully deterministic
//! across platforms via [`bevy_math::ops`].

pub mod additive_oscillator;
pub mod fm_operator;
pub mod karplus_strong;
pub mod noise;
pub mod oscillator;
pub mod pwm_oscillator;
pub mod sample_player;
pub mod supersaw;
pub mod wavetable_oscillator;

pub use additive_oscillator::{AdditiveOscillatorNode, AdditiveOscillatorParams};
pub use fm_operator::{FmOperatorNode, FmOperatorParams};
pub use karplus_strong::{KarplusStrongNode, KarplusStrongParams};
pub use noise::{NoiseColor, NoiseNode};
pub use oscillator::{OscillatorNode, Waveform};
pub use pwm_oscillator::{PwmOscillatorNode, PwmOscillatorParams};
pub use sample_player::{Interpolation, LoopMode, SamplePlayerNode};
pub use supersaw::{SupersawNode, SupersawParams};
pub use wavetable_oscillator::{WavetableOscillatorNode, WavetableOscillatorParams};
