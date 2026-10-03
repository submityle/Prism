//! Modal synthesis: resonant bodies rung by contact excitation.
//!
//! An audible object is modelled as a parallel bank of decaying-sinusoid modes.
//! This module holds the single-mode resonator ([`resonator`]), the fixed
//! capacity bank and its mode table ([`bank`]), the attack transient and
//! spectral tilt shaping ([`excitation`]), and the block-rate renderer that
//! ties them together with a sample-accurate impact queue ([`synth`]).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 47.2 (modal synthesis); excited by the contact bus
//! ([`crate::contact`]) and the continuous stage ([`crate::continuous`]).

pub mod bank;
pub mod excitation;
pub mod resonator;
pub mod synth;

pub use bank::{Mode, ModalBank, MAX_MODES};
pub use excitation::{SpectralTilt, TransientBurst};
pub use resonator::ModeResonator;
pub use synth::ModalSynth;
