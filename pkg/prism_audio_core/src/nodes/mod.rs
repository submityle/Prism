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
//! Insert-style effect processors (parametric EQ, and the rest of the M1
//! effect family) live in the [`effects`] submodule.

pub mod biquad;
pub mod effects;
pub mod gain;
pub mod mix;
pub mod pan;

pub use biquad::{BiquadKind, BiquadNode};
pub use effects::{EqBand, ParametricEqNode};
pub use gain::GainNode;
pub use mix::SumNode;
pub use pan::StereoPanNode;
