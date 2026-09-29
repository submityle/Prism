//! M1 dynamics-family nodes: level-dependent gain processors that shape a
//! signal's dynamic range.
//!
//! These mirror the dynamics processors every AAA engine ships (UE's
//! compressor/limiter submix effects, Unity's mixer compressor/duck-volume,
//! Godot's `AudioEffectCompressor`/`Limiter`): a compressor to tame peaks, a
//! brick-wall limiter to guarantee a ceiling, an expander/gate to suppress
//! noise, and a side-chain ducker for dialogue-priority mixing. They all share
//! the [`detector`] primitives so their ballistics and gain-computer curves are
//! consistent.
//!
//! # Catalogue
//!
//! - [`compressor::CompressorNode`] — soft-knee feed-forward compressor with
//!   peak/RMS detection, look-ahead, make-up gain, and parallel wet/dry mix.
//! - [`limiter::LimiterNode`] — look-ahead brick-wall peak limiter with a
//!   guaranteed output ceiling.
//! - [`gate::ExpanderGateNode`] — downward expander / noise gate with hold.
//! - [`ducking::DuckingNode`] — side-chain ducker (key on input port 1).

pub mod compressor;
pub mod detector;
pub mod ducking;
pub mod gate;
pub mod limiter;

pub use compressor::{CompressorNode, CompressorParams};
pub use detector::{DetectionMode, GainBallistics, LevelDetector};
pub use ducking::{DuckingNode, DuckingParams};
pub use gate::{ExpanderGateNode, GateParams};
pub use limiter::{LimiterNode, LimiterParams};
