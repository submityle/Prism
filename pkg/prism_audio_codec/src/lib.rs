//! Symphonia-backed compressed-codec decoders for Prism's next-generation
//! audio engine.
//!
//! # Provenance
//! Original integration work; contains no Unreal Engine, Unity, Godot, Wwise,
//! FMOD, Steam Audio, or Google Resonance Audio source or derived code; no
//! AI/ML. The actual bit-stream decoding is delegated to the third-party
//! [Symphonia](https://github.com/pdeljanov/Symphonia) crate (MPL-2.0), which
//! this crate adapts to the engine's pluggable decoder abstraction.
//!
//! # Relationship
//! Design sections 20 and 44.1 define a [`SourceDecoder`] trait and a
//! [`DecoderRegistry`] so that container formats and codecs are plug-able and
//! third parties can contribute decoders without the core asset crate taking a
//! dependency on them. `prism_audio_assets` ships the self-contained
//! PCM/ADPCM/WAV path; this crate contributes the compressed families
//! (FLAC, Ogg Vorbis, MP3, AAC/MP4) by wrapping Symphonia behind the same
//! trait and registering the factories through [`register_symphonia_decoders`].
//!
//! Decoding always runs off the real-time audio thread (section 44.1): the
//! [`SymphoniaDecoder`] fills caller-owned buffers that the streaming and bank
//! layers later hand to the read-only audio callback. Symphonia requires the
//! standard library and performs its own bounded heap allocation, which is the
//! reason this crate is `std`-only and lives at the asset layer rather than in
//! the no-std real-time core.
//!
//! # Supported codecs
//! The Symphonia feature set locked for this workspace enables FLAC, Ogg
//! Vorbis, MP3, AAC, ISO-MP4 and RIFF/WAVE demuxing. [`CodecTag::Opus`] is
//! **not** registered because the pinned Symphonia release ships no Opus
//! decoder; attempting to decode an Opus asset therefore honestly surfaces
//! [`DecodeError::UnsupportedFormat`] rather than a stubbed result.
//!
//! [`SourceDecoder`]: prism_audio_assets::codec::decoder::SourceDecoder
//! [`DecoderRegistry`]: prism_audio_assets::codec::registry::DecoderRegistry
//! [`CodecTag::Opus`]: prism_audio_assets::codec::metadata::CodecTag::Opus
//! [`DecodeError::UnsupportedFormat`]: prism_audio_assets::codec::decoder::DecodeError::UnsupportedFormat

mod decoder;
mod registration;

pub use decoder::SymphoniaDecoder;
pub use registration::{register_symphonia_decoders, symphonia_decoder_factory};
