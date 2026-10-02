//! Built-in primitive node library for the Patch graph.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Collects the oscillator, sampler-adjacent, envelope, filter, math, and logic
//! primitives of the design section 11 node library. Each primitive is a small
//! `prism_audio_core::graph::AudioNode`; the Patch compiler instantiates them
//! and wires them into a runtime `AudioGraph`. All primitives are mono and
//! allocation-free in their process path.

pub mod adsr_amp;
pub mod constant;
pub mod gain;
pub mod noise;
pub mod one_pole;
pub mod oscillator;
pub mod product;
pub mod sum;

pub use adsr_amp::AdsrAmpNode;
pub use constant::ConstantNode;
pub use gain::GainNode;
pub use noise::NoiseNode;
pub use one_pole::OnePoleLowpassNode;
pub use oscillator::{OscWaveform, OscillatorNode};
pub use product::ProductNode;
pub use sum::SumNode;
