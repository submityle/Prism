//! Registration of the Symphonia-backed decoders into a [`DecoderRegistry`].
//!
//! # Provenance
//! Original integration glue; contains no Unreal Engine, Unity, Godot, Wwise,
//! FMOD, Steam Audio, or Google Resonance Audio source or derived code; no
//! AI/ML.
//!
//! # Relationship
//! Design sections 20 and 44.1 keep container formats and codecs plug-able so
//! that the asset crate need not depend on heavy third-party codecs. This
//! module wires the compressed families decoded by [`SymphoniaDecoder`] into
//! the engine's [`DecoderRegistry`] by container magic-number probing and by
//! codec tag, without displacing the self-contained RIFF/WAVE path that
//! [`DecoderRegistry::with_native`] installs.
//!
//! [`SymphoniaDecoder`]: crate::SymphoniaDecoder
//! [`DecoderRegistry`]: prism_audio_assets::codec::registry::DecoderRegistry
//! [`DecoderRegistry::with_native`]: prism_audio_assets::codec::registry::DecoderRegistry::with_native

use prism_audio_assets::codec::decoder::DecodeError;
use prism_audio_assets::codec::metadata::CodecTag;
use prism_audio_assets::codec::registry::{BoxedDecoder, DecoderRegistry};

use crate::decoder::SymphoniaDecoder;

/// Builds a boxed [`SymphoniaDecoder`] from an in-memory encoded byte slice.
///
/// This is the single factory shared by every container- and tag-based
/// registration below; it constructs the decoder (which probes the container,
/// selects the default track and primes codec parameters) and erases it to the
/// registry's [`BoxedDecoder`] type.
///
/// # Errors
/// Returns a [`DecodeError`] when the bytes cannot be probed, carry no decodable
/// default track, or use a codec outside the compiled Symphonia feature set.
///
/// [`SymphoniaDecoder`]: crate::SymphoniaDecoder
/// [`BoxedDecoder`]: prism_audio_assets::codec::registry::BoxedDecoder
pub fn symphonia_decoder_factory(bytes: &[u8]) -> Result<BoxedDecoder, DecodeError> {
    let decoder = SymphoniaDecoder::new(bytes)?;
    Ok(Box::new(decoder) as BoxedDecoder)
}

/// Registers the Symphonia-backed compressed decoders into `registry`.
///
/// Container magic numbers cover the self-describing streams Symphonia can
/// probe without external context: `OggS` (Ogg, carrying Vorbis), `fLaC`
/// (native FLAC) and `ID3` (ID3v2-tagged MP3). Bare MP3 frames and raw
/// AAC/MP4 are reached through the codec-tag path when a bank already knows the
/// codec. The RIFF/WAVE container installed by [`DecoderRegistry::with_native`]
/// is deliberately left untouched so the self-contained PCM/ADPCM path keeps
/// serving `.wav` assets. [`CodecTag::Opus`] is intentionally not registered:
/// the pinned Symphonia release ships no Opus decoder, so an Opus asset honestly
/// surfaces [`DecodeError::UnsupportedFormat`] instead of a stubbed result.
///
/// [`DecoderRegistry::with_native`]: prism_audio_assets::codec::registry::DecoderRegistry::with_native
/// [`CodecTag::Opus`]: prism_audio_assets::codec::metadata::CodecTag::Opus
/// [`DecodeError::UnsupportedFormat`]: prism_audio_assets::codec::decoder::DecodeError::UnsupportedFormat
pub fn register_symphonia_decoders(registry: &mut DecoderRegistry) {
    registry.register_container(b"OggS", Box::new(symphonia_decoder_factory));
    registry.register_container(b"fLaC", Box::new(symphonia_decoder_factory));
    registry.register_container(b"ID3", Box::new(symphonia_decoder_factory));
    registry.register_tag(CodecTag::Vorbis, Box::new(symphonia_decoder_factory));
    registry.register_tag(CodecTag::Flac, Box::new(symphonia_decoder_factory));
}
