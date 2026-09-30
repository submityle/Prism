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
//! - [`phaser::PhaserNode`] — cascaded first-order all-pass stages swept by an
//!   LFO to drag notches through the spectrum.
//! - [`stereo_width::StereoWidthNode`] — Mid-Side stereo widener with an
//!   optional bass-mono crossover for image control.
//! - [`tremolo::TremoloNode`] — low-frequency amplitude modulation / auto-pan
//!   driven by a control-rate LFO.

pub mod chorus;
pub mod delay;
pub mod flanger;
pub mod parametric_eq;
pub mod phaser;
pub mod stereo_width;
pub mod tremolo;
pub mod waveshaper;

pub use chorus::{ChorusNode, ChorusParams};
pub use delay::DelayNode;
pub use flanger::{FlangerNode, FlangerParams};
pub use parametric_eq::{EqBand, ParametricEqNode};
pub use phaser::{PhaserNode, PhaserParams};
pub use stereo_width::{MAX_WIDTH, StereoWidthNode, StereoWidthParams};
pub use tremolo::{TremoloMode, TremoloNode, TremoloParams};
pub use waveshaper::{Oversample, WaveshaperNode};
