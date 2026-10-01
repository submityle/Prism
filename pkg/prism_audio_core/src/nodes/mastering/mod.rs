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
pub mod mastering_chain;

pub use dither::{
    DEFAULT_DITHER_BITS, Dither, DitherNode, DitherParams, DitherType, MAX_DITHER_BITS,
    MIN_DITHER_BITS, NoiseShaping,
};
pub use mastering_chain::{MasteringChainNode, MasteringChainParams};
