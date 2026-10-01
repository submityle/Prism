//! Concrete, fully-implemented [`AudioNode`](crate::graph::AudioNode)s.
//!
//! Every node in this module is production-usable: none of them contain
//! `todo!()`, `unimplemented!()`, or placeholder logic. They are the primitive
//! vocabulary the higher authoring and routing layers assemble into a mix
//! graph.
//!
//! # Node catalogue
//!
//! - [`gain::GainNode`] — click-free level trim driven by a
//!   [`Smoothed`](crate::param::Smoothed) parameter.
//! - [`biquad::BiquadNode`] — a second-order RBJ-cookbook IIR filter
//!   (low/high-pass, band-pass, notch, peaking, shelving).
//! - [`pan::StereoPanNode`] — equal-power mono-to-stereo panner.
//! - [`mix::SumNode`] — explicit N-input summing node (a bus mixer primitive).
//! - [`crossover::LinkwitzRileyCrossover`] — fourth-order Linkwitz-Riley
//!   multi-band splitter whose bands sum back to a flat response.
//! - [`svf::SvfNode`] - topology-preserving (TPT) state-variable filter with
//!   simultaneous low/high/band-pass, notch, peak, all-pass, bell, and shelf
//!   responses; stable under fast cutoff modulation.
//!
//! Insert-style effect processors live in the [`effects`] submodule:
//! [`ParametricEqNode`], [`DelayNode`], the `tanh` [`WaveshaperNode`], and the
//! LFO-modulated [`ChorusNode`], [`FlangerNode`], and [`PhaserNode`].
//!
//! Level-dependent processors live in the [`dynamics`] submodule:
//! [`CompressorNode`], [`LimiterNode`], [`ExpanderGateNode`], and the
//! side-chain [`DuckingNode`].
//!
//! Signal-generating voices live in the [`sources`] submodule: the
//! band-limited [`OscillatorNode`], the deterministic [`NoiseNode`], and the
//! resampling [`SamplePlayerNode`]. They declare no inputs and originate a
//! signal rather than transforming one.
//!
//! Spatial ambience processors live in the [`reverb`] submodule: the
//! [`FdnReverb`], the impulse-response [`Convolver`], and the Freeverb-style
//! [`AlgorithmicRoom`].
//!
//! Read-only measurement taps live in the [`analysis`] submodule: the
//! `ITU-R` `BS.1770` / `EBU` `R128` [`LoudnessMeterNode`] reports momentary,
//! short-term, and integrated loudness, loudness range, and true-peak level
//! without altering the signal.
//!
//! Delivery-stage processors live in the [`mastering`] submodule: the
//! [`Dither`] requantizer renders a finished float mix down to a target bit
//! depth with dither and optional noise shaping.

pub mod analysis;
pub mod biquad;
pub mod crossover;
pub mod dynamics;
pub mod effects;
pub mod gain;
pub mod mastering;
pub mod mix;
pub mod pan;
pub mod reverb;
pub mod sources;
pub mod svf;

pub use analysis::{
    CorrelationMeasurement, CorrelationMeter, CorrelationMeterNode, DEFAULT_DECIMATION,
    DEFAULT_FFT_SIZE, DEFAULT_HOP, DEFAULT_POINT_CAPACITY, Goniometer, GoniometerNode,
    GoniometerPoint, GoniometerStats, KWeighting, LoudnessMeasurement, LoudnessMeter,
    LoudnessMeterNode, MIN_FFT_SIZE, MIN_POINT_CAPACITY, SpectrumAnalyzer, SpectrumNode,
    TruePeakMeter, Window,
};
pub use biquad::{BiquadKind, BiquadNode};
pub use crossover::{LinkwitzRileyCrossover, MAX_BANDS, MAX_CROSSOVERS};
pub use dynamics::{
    CompressorNode, CompressorParams, DeEsserMode, DeEsserNode, DeEsserParams, DetectionMode,
    DuckingNode, DuckingParams, ExpanderGateNode, GateParams, LimiterNode, LimiterParams,
    MultibandCompressorNode, TransientShaperNode, TransientShaperParams,
};
pub use effects::{
    BitcrusherNode, BitcrusherParams, ChorusNode, ChorusParams, CombResonatorNode,
    CombResonatorParams, DelayNode, EqBand, FlangerNode, FlangerParams, MAX_BIT_DEPTH,
    MIN_BIT_DEPTH, Oversample, ParametricEqNode, PhaserNode, PhaserParams, RingModulatorNode,
    RingModulatorParams, SaturationCurve, SaturationNode, SaturationParams, TremoloMode,
    TremoloNode, TremoloParams, VibratoNode, VibratoParams,
    WaveshaperNode,
};
pub use gain::GainNode;
pub use mastering::{
    DEFAULT_DITHER_BITS, Dither, DitherNode, DitherParams, DitherType, MAX_DITHER_BITS,
    MIN_DITHER_BITS, MasteringChainNode, MasteringChainParams, NoiseShaping,
};
pub use mix::SumNode;
pub use pan::StereoPanNode;
pub use reverb::{
    AlgorithmicRoom, AlgorithmicRoomParams, Convolver, FdnOrder, FdnReverb, FdnReverbParams,
};
pub use sources::{
    Interpolation, LoopMode, NoiseColor, NoiseNode, OscillatorNode, SamplePlayerNode, Waveform,
};
pub use svf::{Svf, SvfCoeffs, SvfKind, SvfNode, SvfParams};
