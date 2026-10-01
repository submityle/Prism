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
//! - [`graphic_eq::GraphicEqNode`] -- fixed ISO octave / third-octave graphic
//!   equalizer: a bank of constant-Q RBJ peaking biquads on standardized
//!   center frequencies, giving bit-exact bypass when every band is flat;
//!   distinct from [`parametric_eq::ParametricEqNode`], which exposes an
//!   arbitrary frequency / Q / gain / shape per band.
//! - [`delay::DelayNode`] — fractional delay line with feedback and wet/dry
//!   mix (the echo / slap-back / modulated-delay primitive).
//! - [`multi_tap_delay::MultiTapDelayNode`] -- a single shared mono delay
//!   line read by up to [`MAX_TAPS`] independently timed, gained, and
//!   panned taps with a global feedback coefficient, for rhythmic echo
//!   patterns, discrete early-reflection clusters, and stereo spreading;
//!   distinct from [`delay::DelayNode`] (one per-channel fractional tap)
//!   and from the LFO-swept [`chorus::ChorusNode`] / [`flanger::FlangerNode`]
//!   whose taps modulate cyclically rather than stay fixed.
//! - [`waveshaper::WaveshaperNode`] — `tanh` soft-clip saturation with optional
//!   2x/4x band-limiting oversampling.
//! - [`wavefolder::WavefolderNode`] -- west-coast reflective wave folder
//!   (triangle / sine fold) with selectable up-to-8x oversampling; unlike the
//!   compressive [`waveshaper::WaveshaperNode`] it mirrors signal past the fold
//!   threshold for bright, metallic, harmonically dense timbres.
//! - [`chorus::ChorusNode`] — multi-voice LFO-modulated delay ensemble (no
//!   feedback) for shimmering thickening.
//! - [`flanger::FlangerNode`] — single short LFO-swept delay with feedback for
//!   the classic sweeping comb / "jet" effect.
//! - [`formant_filter::FormantFilterNode`] -- parallel band-pass resonator bank
//!   tuned to the five cardinal vowels, with continuous vowel morphing, for
//!   talk-box / vocal-pad timbres; distinct from the series EQ nodes and from
//!   the single swept band of [`auto_wah::AutoWahNode`].
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
//! - [`haas_widener::HaasWidenerNode`] -- precedence / Haas stereo widener that
//!   delays only the Mid-Side side component by a few milliseconds, widening
//!   the image through time decorrelation while staying mono-compatible.
//!   Distinct from [`stereo_width::StereoWidthNode`] (side-gain) and
//!   [`mid_side_matrix::MidSideMatrixNode`] (static M/S trim).
//! - [`leslie::LeslieNode`] -- rotary-speaker (Leslie-style) cabinet that
//!   splits the signal into a slow bass rotor and a fast treble horn and
//!   imposes coupled Doppler pitch wobble, amplitude tremolo, and antiphase
//!   stereo motion with mechanical spin-up / spin-down inertia. Distinct from
//!   [`tremolo::TremoloNode`] (amplitude only) and [`vibrato::VibratoNode`]
//!   (pitch only) because it couples pitch, amplitude, and stereo image.
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
//! - [`pitch_shifter::PitchShifterNode`] -- phase-vocoder pitch shifter that
//!   transposes every partial by one ratio (preserving harmonic ratios and
//!   duration) via STFT analysis, instantaneous-frequency estimation, and
//!   phase-accumulating resynthesis; distinct from the inharmonic
//!   [`frequency_shifter::FrequencyShifterNode`] (adds a Hz offset) and the
//!   time-domain [`vibrato::VibratoNode`] (cyclic delay-based bend).
//! - [`vibrato::VibratoNode`] — single LFO-swept fractional delay for
//!   periodic pitch modulation.
//! - [`vocoder::VocoderNode`] -- channel vocoder cross-synthesis: a band bank
//!   imprints the moving spectral envelope of a modulator (input 0) onto a
//!   carrier (input 1); distinct from the fixed vowel bank of
//!   [`formant_filter::FormantFilterNode`] and from the single-signal band
//!   splitting of [`crate::nodes::dynamics::multiband`].
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
//! - [`spectral_freeze::SpectralFreezeNode`] -- frequency-domain spectral
//!   freeze: a weighted overlap-add short-time Fourier transform (`STFT`)
//!   latches the current magnitude spectrum and advances each bin's phase by
//!   its own centre frequency, sustaining the captured timbre indefinitely
//!   with an optional deterministic phase diffusion for a shimmering pad.
//! - [`spectral_gate::SpectralGateNode`] -- frequency-domain spectral gate /
//!   downward spectral expander: a weighted overlap-add short-time Fourier
//!   transform (`STFT`) attenuates bins below a `dBFS` threshold toward a floor,
//!   suppressing steady broadband noise that a time-domain gate cannot isolate.
//! - [`tilt_eq::TiltEqNode`] -- single-knob spectral tilt that rotates the
//!   whole spectrum about a pivot frequency via a matched low-shelf (at
//!   `-tilt_db`) and high-shelf (at `+tilt_db`) pair; distinct from
//!   [`parametric_eq::ParametricEqNode`] (arbitrary independent bands) and
//!   [`graphic_eq::GraphicEqNode`] (fixed ISO grid) by exposing one tilt knob.
//! - [`granular::GranularNode`] -- real-time granular / grain-cloud
//!   texture processor: records input into a mono capture ring and sprays
//!   short Hann-windowed grains that read back from the recent past with
//!   per-grain randomized position, pitch, and stereo pan, dissolving a
//!   steady input into an evolving cloud; reuses
//!   [`pitch_shifter::semitones_to_ratio`] for grain detune and is distinct
//!   from the single-ratio [`pitch_shifter::PitchShifterNode`] and the
//!   static-asset [`crate::nodes::sources::sample_player`].
//! - [`transformer::TransformerNode`] -- audio-transformer (iron-core)
//!   emulation: a frequency-weighted asymmetric saturator (a low-shelf
//!   lift before a biased `tanh` and the inverse shelf after it, so bass
//!   saturates before treble), a winding / leakage resonance peak, and a
//!   `DC`-blocking output high pass. Reuses
//!   [`crate::nodes::biquad::BiquadCoeffs::design`] for every section and
//!   is distinct from the memoryless [`saturation::SaturationNode`], the
//!   wow / flutter [`tape::TapeNode`], the high-harmonic
//!   [`exciter::ExciterNode`], and the reflective [`wavefolder::WavefolderNode`].

pub mod auto_wah;
pub mod bitcrusher;
pub mod chorus;
pub mod comb_resonator;
pub mod delay;
pub mod exciter;
pub mod flanger;
pub mod formant_filter;
pub mod frequency_shifter;
pub mod granular;
pub mod graphic_eq;
pub mod haas_widener;
pub mod leslie;
pub mod mid_side_matrix;
pub mod multi_tap_delay;
pub mod parametric_eq;
pub mod phaser;
pub mod pitch_shifter;
pub mod ring_modulator;
pub mod saturation;
pub mod spectral_freeze;
pub mod spectral_gate;
pub mod stereo_width;
pub mod tape;
pub mod tilt_eq;
pub mod transformer;
pub mod tremolo;
pub mod vibrato;
pub mod vocoder;
pub mod wavefolder;
pub mod waveshaper;

pub use auto_wah::{AutoWah, AutoWahNode, AutoWahParams, SweepDirection, WahMode};
pub use bitcrusher::{BitcrusherNode, BitcrusherParams, MAX_BIT_DEPTH, MIN_BIT_DEPTH};
pub use chorus::{ChorusNode, ChorusParams};
pub use comb_resonator::{CombResonatorNode, CombResonatorParams, MAX_FEEDBACK, MIN_FREQUENCY_HZ};
pub use delay::DelayNode;
pub use exciter::{Exciter, ExciterNode, ExciterParams, HarmonicMode};
pub use flanger::{FlangerNode, FlangerParams};
pub use formant_filter::{
    FormantFilter, FormantFilterNode, FormantFilterParams, FormantSpec, Vowel,
};
pub use frequency_shifter::{FrequencyShifterNode, FrequencyShifterParams};
pub use granular::{
    DEFAULT_CAPTURE_SECONDS, DEFAULT_SEED, GranularNode, GranularParams, MAX_DENSITY_HZ,
    MAX_GRAIN_MS, MAX_GRAINS, MAX_SPREAD, MIN_DENSITY_HZ, MIN_GRAIN_MS,
};
pub use graphic_eq::{GraphicEqNode, GraphicEqSpacing};
pub use haas_widener::{HaasWidenerNode, HaasWidenerParams};
pub use leslie::{
    DEFAULT_CROSSOVER_HZ, DRUM_ACCEL_SECONDS, DRUM_DECEL_SECONDS, DRUM_FAST_HZ, DRUM_SLOW_HZ,
    HORN_ACCEL_SECONDS, HORN_DECEL_SECONDS, HORN_FAST_HZ, HORN_SLOW_HZ, LeslieNode, LeslieParams,
    LeslieSpeed, MAX_AM_DEPTH, MAX_DOPPLER_DEPTH,
};
pub use mid_side_matrix::{MidSideMatrixNode, MidSideMatrixParams, MidSideMode};
pub use multi_tap_delay::{MAX_TAPS, MultiTapDelayNode, MultiTapDelayParams, TapSpec};
pub use parametric_eq::{EqBand, ParametricEqNode};
pub use phaser::{PhaserNode, PhaserParams};
pub use pitch_shifter::{
    MAX_PITCH_RATIO, MIN_PITCH_RATIO, PitchShifterNode, PitchShifterParams, semitones_to_ratio,
};
pub use ring_modulator::{RingModulatorNode, RingModulatorParams};
pub use saturation::{DEFAULT_DC_BLOCK_COEFF, SaturationCurve, SaturationNode, SaturationParams};
pub use spectral_freeze::{
    FREEZE_RAMP_SECONDS, MAX_DIFFUSION, MIN_FREEZE_FFT_SIZE, SpectralFreezeNode,
    SpectralFreezeParams,
};
pub use spectral_gate::{
    DEFAULT_ATTACK_MS, DEFAULT_FFT_SIZE, DEFAULT_RELEASE_MS, DEFAULT_REDUCTION_DB,
    DEFAULT_THRESHOLD_DB, MIN_FFT_SIZE, OVERLAP_FACTOR, SpectralGateNode, SpectralGateParams,
};
pub use stereo_width::{MAX_WIDTH, StereoWidthNode, StereoWidthParams};
pub use tape::{Tape, TapeNode, TapeParams};
pub use tilt_eq::{TiltEq, TiltEqNode, TiltEqParams};
pub use transformer::{Transformer, TransformerNode, TransformerParams};
pub use tremolo::{TremoloMode, TremoloNode, TremoloParams};
pub use vibrato::{VibratoNode, VibratoParams};
pub use vocoder::{Vocoder, VocoderNode, VocoderParams};
pub use wavefolder::{
    FoldShape, Oversample as FolderOversample, WavefolderNode, WavefolderParams,
};
pub use waveshaper::{Oversample, WaveshaperNode};
