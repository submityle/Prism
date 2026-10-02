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
//! - [`impulse_train::ImpulseTrainNode`] -- band-limited impulse train
//!   (BLIT): an alias-free train of narrow spikes evaluated as the
//!   closed-form normalized Dirichlet kernel (periodic sinc), keeping every
//!   harmonic below Nyquist with equal weight. It is the raw excitation that
//!   integrates into alias-free saw/pulse voices and formant sources.
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
//! - [`granular_source::GranularSourceNode`] -- Gabor grain-cloud synthesizer:
//!   a scheduler sprays overlapping Hann-windowed sine grains whose carrier
//!   pitch, lifetime, and stereo placement are randomized from a deterministic
//!   PRNG, with power-preserving level normalization. Unlike the capture-based
//!   granulator effect it synthesizes its grains from scratch, so it is a true
//!   zero-input source.
//! - [`fof_source::FofSourceNode`] -- formant-wave-function (FOF) voice
//!   synthesizer: a fundamental phase accumulator fires one damped-sine
//!   formant grain per active formant on every period, so the periodic
//!   triggering fixes the pitch while each grain's exponential decay and
//!   raised-cosine skirt shape an independent formant peak. Unlike the
//!   stochastic grain cloud of `granular_source` the schedule is
//!   deterministic and pitch-synchronous, the classic sung-vowel voice.
//! - [`bowed_string::BowedStringNode`] -- bowed-string digital-waveguide
//!   physical model: a pair of velocity-wave delay lines (bridge-side and
//!   nut-side) terminated by inverting reflections, driven every sample by the
//!   `McIntyre`-Schumacher-Woodhouse bow-friction nonlinearity so the string
//!   self-oscillates. Unlike the once-plucked `karplus_strong` it is
//!   continuously bowed, sustaining as long as the bow moves.
//! - [`reed_woodwind::ReedWoodwindNode`] -- single-reed woodwind
//!   (clarinet-family) digital-waveguide physical model: a pair of pressure-
//!   wave delay lines form a cylindrical bore closed at the mouthpiece by a
//!   nonlinear pressure-controlled reed valve and opened at the bell by a
//!   lossy inverting reflection. The single inversion per round trip resonates
//!   only the odd harmonics, the physical origin of the hollow clarinet timbre.
//!   Unlike the bow-driven `bowed_string` it is sustained by steady breath
//!   pressure through the reed rather than bow friction.
//! - [`air_jet_flute::AirJetFluteNode`] -- air-jet flute (concert-flute /
//!   recorder family) digital-waveguide physical model: two cross-coupled
//!   pressure-wave delay lines form an open-open cylindrical bore, plus a jet
//!   convective-delay line whose cubic edge-tone deflection `jet = JET_DRIVE *
//!   (x - x^3)` pumps the bore. The two inverting end reflections cancel per
//!   round trip, so unlike the odd-only `reed_woodwind` it resonates the full
//!   harmonic series an octave higher; deterministic breath turbulence breaks
//!   the symmetry to start the tone. Driven by an air jet rather than the
//!   `bowed_string` bow or `karplus_strong` pluck.
//!
//! Every generator is real-time safe: `process` performs no allocation, no
//! locking, and no panics, and reproducible generators are fully deterministic
//! across platforms via [`bevy_math::ops`].

pub mod air_jet_flute;
pub mod additive_oscillator;
pub mod bowed_string;
pub mod fm_operator;
pub mod fof_source;
pub mod granular_source;
pub mod impulse_train;
pub mod karplus_strong;
pub mod noise;
pub mod oscillator;
pub mod pwm_oscillator;
pub mod reed_woodwind;
pub mod sample_player;
pub mod supersaw;
pub mod wavetable_oscillator;

pub use air_jet_flute::{AirJetFluteNode, AirJetFluteParams};
pub use additive_oscillator::{AdditiveOscillatorNode, AdditiveOscillatorParams};
pub use bowed_string::{BowedStringNode, BowedStringParams};
pub use fm_operator::{FmOperatorNode, FmOperatorParams};
pub use fof_source::{FofSourceNode, FofSourceParams, Formant, MAX_FOF_GRAINS, MAX_FORMANTS};
pub use granular_source::{GranularSourceNode, GranularSourceParams, MAX_GRAINS};
pub use impulse_train::{ImpulseTrainNode, ImpulseTrainParams};
pub use karplus_strong::{KarplusStrongNode, KarplusStrongParams};
pub use noise::{NoiseColor, NoiseNode};
pub use oscillator::{OscillatorNode, Waveform};
pub use pwm_oscillator::{PwmOscillatorNode, PwmOscillatorParams};
pub use reed_woodwind::{ReedWoodwindNode, ReedWoodwindParams};
pub use sample_player::{Interpolation, LoopMode, SamplePlayerNode};
pub use supersaw::{SupersawNode, SupersawParams};
pub use wavetable_oscillator::{WavetableOscillatorNode, WavetableOscillatorParams};
