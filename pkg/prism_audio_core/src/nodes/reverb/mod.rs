//! M1 reverb-family nodes: spatial ambience processors that convolve or
//! recirculate a signal to synthesise the sound of a physical space.
//!
//! Reverberation is the dense wash of reflections a room adds to a dry source.
//! The engine ships four complementary algorithms, mirroring the choices found
//! in production audio middleware:
//!
//! # Catalogue
//!
//! - [`fdn::FdnReverb`] — a Feedback Delay Network: several prime-length delay
//!   lines recirculated through an orthogonal Hadamard mixing matrix with
//!   per-line damping. This is the modern, efficient workhorse for lush,
//!   colour-controllable tails (the family behind most game-audio reverbs).
//! - [`convolver::Convolver`] — a partitioned time-domain convolution reverb.
//!   It reproduces the *exact* acoustic fingerprint of a measured space by
//!   convolving the input with a recorded impulse response, one response per
//!   channel.
//! - [`algorithmic::AlgorithmicRoom`] — a Schroeder/Freeverb-style room built
//!   from a pre-delay, a tapped early-reflection delay line, and a late tail of
//!   parallel feedback combs feeding series all-pass diffusers.
//! - [`plate::PlateReverb`] -- a Dattorro figure-eight plate: a four-stage
//!   input diffuser feeding a single cross-coupled all-pass feedback tank
//!   with modulated all-passes and multi-tap stereo outputs, synthesising
//!   the dense metallic wash of a vintage mechanical plate.
//!
//! Every processor pre-allocates all delay-line and filter state at
//! construction, so [`AudioNode::process`](crate::graph::AudioNode::process)
//! stays allocation-free, lock-free, and panic-free on the audio thread.

pub mod algorithmic;
pub mod convolver;
pub mod fdn;
pub mod plate;

pub use algorithmic::{AlgorithmicRoom, AlgorithmicRoomParams};
pub use convolver::Convolver;
pub use fdn::{FdnOrder, FdnReverb, FdnReverbParams};
pub use plate::{PlateReverb, PlateReverbParams};
