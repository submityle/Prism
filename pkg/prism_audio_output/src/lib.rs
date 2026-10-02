//! Output rendering/delivery layer for Prism's next-generation audio engine.
//!
//! This crate is the final, explicit, deterministic stage of the mastering
//! pipeline: it takes the engine's internal high-resolution mix and renders it
//! for a concrete delivery target. It sits after the `prism_audio_core`
//! mastering chain (loudness normalisation, limiting, dither) and before the
//! device callback. See `docs/prism_audio_engine_design_zh.md` section 48
//! (输出渲染链与母带交付档).
//!
//! The chain has three composable stages, one concept per module:
//!
//! 1. [`downmix`] — [`DownmixMatrix`], ITU-R BS.775 fold-down between channel
//!    layouts as an explicit out×in gain matrix (`7.1 → 5.1 → stereo → mono`,
//!    plus `quad → stereo`).
//! 2. [`bass_management`] — [`BassManager`], a Linkwitz-Riley LFE/bass-
//!    management crossover that redirects the mains' deep bass into the LFE
//!    channel with a calibration gain.
//! 3. [`profiles`] — [`OutputProfile`], the delivery dynamic-range/loudness
//!    presets (Home Theater / TV / Night / Headphone) resolved into concrete
//!    processing parameters.
//!
//! # Provenance
//!
//! This crate contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is pure
//! classic DSP with no AI/ML, built on publicly published standards (ITU-R
//! BS.775 downmix, Linkwitz-Riley crossovers, EBU R128 / console listening-mode
//! conventions) and this workspace's own `prism_audio_core` primitives.
//!
//! # Relationship
//!
//! Every stage reuses `prism_audio_core` rather than reimplementing DSP: the
//! downmix and bass manager operate on
//! [`AudioBuffer`](prism_audio_core::buffer::AudioBuffer), the crossover is the
//! core [`LinkwitzRileyCrossover`](prism_audio_core::nodes::crossover::LinkwitzRileyCrossover),
//! and the delivery profiles resolve into the core
//! [`LoudnessNormalizerParams`](prism_audio_core::nodes::mastering::LoudnessNormalizerParams).
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod bass_management;
pub mod downmix;
pub mod profiles;

pub use bass_management::{
    BassManager, BassManagerParams, DEFAULT_CROSSOVER_HZ, DEFAULT_LFE_GAIN_DB,
};
pub use downmix::{DownmixMatrix, MINUS_3DB};
pub use profiles::{OutputProfile, OutputProfileParams};
