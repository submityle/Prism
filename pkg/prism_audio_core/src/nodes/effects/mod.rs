//! M1 effect-family nodes: insert-style DSP processors that transform a signal
//! in place along a bus or source chain.
//!
//! These are the "insert effects" of the mix graph (mirroring Godot's
//! `AudioEffect` instances, UE's Source/Submix Effect chains, and Unity's mixer
//! effects). Every processor pre-allocates its state at construction so
//! [`AudioNode::process`](crate::graph::AudioNode::process) stays real-time
//! safe.
//!
//! # Catalogue
//!
//! - [`auto_wah::AutoWahNode`] -- envelope-controlled resonant filter
//!   (auto-wah / envelope filter): a rectified attack / release follower sweeps
//!   the shared [`Svf`](crate::nodes::svf::Svf) cutoff with the input loudness.
//! - [`parametric_eq::ParametricEqNode`] — multi-band parametric EQ built by
//!   cascading reusable [`Biquad`](crate::nodes::biquad::Biquad) sections.
//! - [`delay::DelayNode`] — fractional delay line with feedback and wet/dry
//!   mix (the echo / slap-back / modulated-delay primitive).
//! - [`waveshaper::WaveshaperNode`] — `tanh` soft-clip saturation with optional
//!   2x/4x band-limiting oversampling.
//! - [`chorus::ChorusNode`] — multi-voice LFO-modulated delay ensemble (no
//!   feedback) for shimmering thickening.
//! - [`flanger::FlangerNode`] — single short LFO-swept delay with feedback for
//!   the classic sweeping comb / "jet" effect.
//! - [`comb_resonator::CombResonatorNode`] — pitched feedback comb with a
//!   lowpass in the loop for plucked-string / resonant-body voices.
//! - [`phaser::PhaserNode`] — cascaded first-order all-pass stages swept by an
//!   LFO to drag notches through the spectrum.
//! - [`stereo_width::StereoWidthNode`] — Mid-Side stereo widener with an
//!   optional bass-mono crossover for image control.
//! - [`mid_side_matrix::MidSideMatrixNode`] -- pure Mid-Side (sum and
//!   difference) encoder / decoder with independent mid and side trim
//!   gains; exposes `M` / `S` for independent processing between an encode
//!   and a decode stage. Distinct from
//!   [`stereo_width::StereoWidthNode`], which never exposes `M` / `S` and
//!   only applies a single width scale internally.
//! - [`tremolo::TremoloNode`] — low-frequency amplitude modulation / auto-pan
//!   driven by a control-rate LFO.
//! - [`bitcrusher::BitcrusherNode`] — bit-depth quantization plus sample-rate
//!   reduction (sample-and-hold decimation) for gritty lo-fi degradation.
//! - [`ring_modulator::RingModulatorNode`] — multiplies the signal by a
//!   bipolar audio-rate carrier for inharmonic, bell-like, or robotic timbres.
//! - [`frequency_shifter::FrequencyShifterNode`] -- single-sideband (SSB)
//!   shifter that adds a constant Hz offset to every partial via a Hilbert
//!   analytic signal and complex modulation, breaking harmonic ratios for
//!   metallic, clangorous timbres; distinct from
//!   [`ring_modulator::RingModulatorNode`], which produces a symmetric
//!   sideband pair rather than a one-sided shift.
//! - [`vibrato::VibratoNode`] — single LFO-swept fractional delay for
//!   periodic pitch modulation.
//! - [`exciter::ExciterNode`] — high-frequency harmonic exciter / aural
//!   enhancer: a highpass-isolated band is `tanh`-shaped to synthesize added
//!   odd / even harmonics that are mixed back for presence and air.
//! - [`tape::TapeNode`] — analog tape-machine emulation: `tanh` drive / bias
//!   saturation feeding a wow / flutter modulated fractional delay and a
//!   one-pole high-frequency rolloff for vintage warmth and pitch wobble.
//! - [`saturation::SaturationNode`] -- multi-curve asymmetric saturator
//!   (`tanh` / `arctan` / cubic / reciprocal / sine) with a drive + DC bias
//!   stage, oversampling, and an output DC blocker; distinct from
//!   [`waveshaper::WaveshaperNode`], which is a fixed symmetric `tanh` clip.
//! - [`spectral_gate::SpectralGateNode`] -- frequency-domain spectral gate /
//!   downward spectral expander: a weighted overlap-add short-time Fourier
//!   transform (`STFT`) attenuates bins below a `dBFS` threshold toward a floor,
//!   suppressing steady broadband noise that a time-domain gate cannot isolate.

pub mod auto_wah;
pub mod bitcrusher;
pub mod chorus;
pub mod comb_resonator;
pub mod delay;
pub mod exciter;
pub mod flanger;
pub mod frequency_shifter;
pub mod mid_side_matrix;
pub mod parametric_eq;
pub mod phaser;
pub mod ring_modulator;
pub mod saturation;
pub mod spectral_gate;
pub mod stereo_width;
pub mod tape;
pub mod tremolo;
pub mod vibrato;
pub mod waveshaper;

pub use auto_wah::{AutoWah, AutoWahNode, AutoWahParams, SweepDirection, WahMode};
pub use bitcrusher::{BitcrusherNode, BitcrusherParams, MAX_BIT_DEPTH, MIN_BIT_DEPTH};
pub use chorus::{ChorusNode, ChorusParams};
pub use comb_resonator::{CombResonatorNode, CombResonatorParams, MAX_FEEDBACK, MIN_FREQUENCY_HZ};
pub use delay::DelayNode;
pub use exciter::{Exciter, ExciterNode, ExciterParams, HarmonicMode};
pub use flanger::{FlangerNode, FlangerParams};
pub use frequency_shifter::{FrequencyShifterNode, FrequencyShifterParams};
pub use mid_side_matrix::{MidSideMatrixNode, MidSideMatrixParams, MidSideMode};
pub use parametric_eq::{EqBand, ParametricEqNode};
pub use phaser::{PhaserNode, PhaserParams};
pub use ring_modulator::{RingModulatorNode, RingModulatorParams};
pub use saturation::{DEFAULT_DC_BLOCK_COEFF, SaturationCurve, SaturationNode, SaturationParams};
pub use spectral_gate::{
    DEFAULT_ATTACK_MS, DEFAULT_FFT_SIZE, DEFAULT_RELEASE_MS, DEFAULT_REDUCTION_DB,
    DEFAULT_THRESHOLD_DB, MIN_FFT_SIZE, OVERLAP_FACTOR, SpectralGateNode, SpectralGateParams,
};
pub use stereo_width::{MAX_WIDTH, StereoWidthNode, StereoWidthParams};
pub use tape::{Tape, TapeNode, TapeParams};
pub use tremolo::{TremoloMode, TremoloNode, TremoloParams};
pub use vibrato::{VibratoNode, VibratoParams};
pub use waveshaper::{Oversample, WaveshaperNode};
