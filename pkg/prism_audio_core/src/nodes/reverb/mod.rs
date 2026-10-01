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
//! - [`shimmer::ShimmerReverb`] -- a shimmer reverb: an FDN tank whose own
//!   tail is pitch-shifted (classically up one octave) and re-injected
//!   through a sub-unity feedback path, so energy perpetually climbs
//!   toward higher frequencies to produce the ethereal, blooming wash of
//!   ambient and cinematic sound design. It composes [`fdn::FdnReverb`]
//!   and the effects-family pitch shifter rather than duplicating them.
//! - [`spring_reverb::SpringReverbNode`] -- a dispersive spring-tank model:
//!   a short recirculating delay whose feedback loop contains a cascade of
//!   first-order all-pass dispersion stages and an in-loop damping low-pass,
//!   synthesising the chirped, metallic "boing" of a guitar-amp or studio
//!   spring reverb. Distinct from the diffuse-field room models above, its
//!   defining feature is the all-pass dispersion chain rather than a dense
//!   reflection field.
//!
//! Every processor pre-allocates all delay-line and filter state at
//! construction, so [`AudioNode::process`](crate::graph::AudioNode::process)
//! stays allocation-free, lock-free, and panic-free on the audio thread.

pub mod algorithmic;
pub mod convolver;
pub mod fdn;
pub mod plate;
pub mod shimmer;
pub mod spring_reverb;

pub use algorithmic::{AlgorithmicRoom, AlgorithmicRoomParams};
pub use convolver::Convolver;
pub use fdn::{FdnOrder, FdnReverb, FdnReverbParams};
pub use plate::{PlateReverb, PlateReverbParams};
pub use shimmer::{MAX_SHIMMER_FEEDBACK, ShimmerReverb, ShimmerReverbParams};
pub use spring_reverb::{
    DEFAULT_SPRING_DAMPING, DEFAULT_SPRING_DECAY, DEFAULT_SPRING_DISPERSION,
    DEFAULT_SPRING_MIX, DEFAULT_SPRING_SIZE_MS, MAX_SPRING_FEEDBACK, MAX_SPRING_SIZE_MS,
    MIN_SPRING_SIZE_MS, SPRING_ALLPASS_STAGES, SpringReverbNode, SpringReverbParams,
};
