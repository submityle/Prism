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
//!
//! Insert-style effect processors live in the [`effects`] submodule:
//! [`ParametricEqNode`], [`DelayNode`], the `tanh` [`WaveshaperNode`], and the
//! LFO-modulated [`ChorusNode`], [`FlangerNode`], and [`PhaserNode`].
//!
//! Level-dependent processors live in the [`dynamics`] submodule:
//! [`CompressorNode`], [`LimiterNode`], [`ExpanderGateNode`], and the
//! side-chain [`DuckingNode`].
//!
//! Spatial ambience processors live in the [`reverb`] submodule: the
//! [`FdnReverb`], the impulse-response [`Convolver`], and the Freeverb-style
//! [`AlgorithmicRoom`].

pub mod biquad;
pub mod dynamics;
pub mod effects;
pub mod gain;
pub mod mix;
pub mod pan;
pub mod reverb;

pub use biquad::{BiquadKind, BiquadNode};
pub use dynamics::{
    CompressorNode, CompressorParams, DetectionMode, DuckingNode, DuckingParams, ExpanderGateNode,
    GateParams, LimiterNode, LimiterParams,
};
pub use effects::{
    ChorusNode, ChorusParams, DelayNode, EqBand, FlangerNode, FlangerParams, Oversample,
    ParametricEqNode, PhaserNode, PhaserParams, WaveshaperNode,
};
pub use gain::GainNode;
pub use mix::SumNode;
pub use pan::StereoPanNode;
pub use reverb::{
    AlgorithmicRoom, AlgorithmicRoomParams, Convolver, FdnOrder, FdnReverb, FdnReverbParams,
};
