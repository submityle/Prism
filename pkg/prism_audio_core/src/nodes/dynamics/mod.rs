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
//! - [`de_esser::DeEsserNode`] — split-band de-esser: a crossover keyed off the
//!   high band tames vocal sibilance while the body of the voice passes through.
//! - [`dynamic_eq::DynamicEqNode`] -- single parametric bell whose boost/cut is
//!   cross-faded by a band-level detector (dynamic equalisation).
//! - [`limiter::LimiterNode`] — look-ahead brick-wall peak limiter with a
//!   guaranteed output ceiling.
//! - [`multiband::MultibandCompressorNode`] — Linkwitz-Riley band split
//!   feeding an independent compressor per band, recombined flat.
//! - [`transient_shaper::TransientShaperNode`] - differential-envelope
//!   attack / sustain designer (fast vs slow follower), threshold-free.
//! - [`gate::ExpanderGateNode`] — downward expander / noise gate with hold.
//! - [`ducking::DuckingNode`] — side-chain ducker (key on input port 1).
//! - [`upward_compressor::UpwardCompressorNode`] -- feed-forward upward
//!   compressor that lifts signal below the threshold (boost = (1-1/R)*under,
//!   capped by `max_gain_db`); stereo-linked, reusing the shared detector and
//!   ballistics. It is the sign-flipped sibling of the downward compressor and
//!   is distinct from the gate, which attenuates below the threshold.

pub mod compressor;
pub mod de_esser;
pub mod detector;
pub mod ducking;
pub mod dynamic_eq;
pub mod gate;
pub mod limiter;
pub mod multiband;
pub mod transient_shaper;
pub mod upward_compressor;

pub use compressor::{CompressorNode, CompressorParams};
pub use de_esser::{DeEsserMode, DeEsserNode, DeEsserParams};
pub use detector::{DetectionMode, GainBallistics, LevelDetector};
pub use ducking::{DuckingNode, DuckingParams};
pub use dynamic_eq::{DynamicEqMode, DynamicEqNode, DynamicEqParams};
pub use gate::{ExpanderGateNode, GateParams};
pub use limiter::{LimiterNode, LimiterParams};
pub use multiband::MultibandCompressorNode;
pub use transient_shaper::{TransientShaperNode, TransientShaperParams};
pub use upward_compressor::{
    DEFAULT_UPWARD_MAX_GAIN_DB, DEFAULT_UPWARD_RATIO, DEFAULT_UPWARD_THRESHOLD_DB,
    MAX_UPWARD_GAIN_DB, UpwardCompressorNode, UpwardCompressorParams, upward_boost_db,
};
