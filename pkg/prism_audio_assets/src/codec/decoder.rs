//! The pluggable [`SourceDecoder`] trait and its error type.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Realises the `SourceDecoder` abstraction referenced by design sections 10,
//! 20, and 44.1. Native PCM/ADPCM decoders in this crate implement the trait;
//! external crates can implement it for Vorbis/Opus/FLAC and register their
//! factories with the [`DecoderRegistry`](crate::codec::registry::DecoderRegistry).
//! Decoding always runs off the real-time thread (section 44.1): the trait
//! fills caller-owned buffers that the streaming layer later hands to the
//! read-only audio callback.

use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::codec::metadata::AudioStreamInfo;

/// Errors that a decoder can surface while parsing or decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DecodeError {
    /// The encoded byte stream ended before a complete unit could be decoded.
    UnexpectedEof,
    /// The container or codec header was malformed.
    MalformedHeader,
    /// The declared format is syntactically valid but not supported natively.
    UnsupportedFormat,
    /// A seek target lies outside the decodable range.
    SeekOutOfRange,
    /// The caller-provided output buffer was too small to hold a whole frame.
    OutputTooSmall,
}

/// A pluggable audio decoder that turns encoded bytes into interleaved
/// `f32` frames.
///
/// Implementations decode incrementally: each [`SourceDecoder::decode`] call
/// produces up to the number of whole frames that fit in the caller's buffer
/// and advances an internal cursor. This lets the streaming layer pump a
/// bounded amount of work per task tick while the real-time thread only ever
/// reads already-decoded samples.
///
/// # Real-time contract
///
/// Decoders are never invoked from the audio callback. They may be stateful
/// but must not block; all work is driven from background tasks (section 44.1).
pub trait SourceDecoder {
    /// Returns the immutable stream description (channels, rate, frames, tag).
    fn info(&self) -> AudioStreamInfo;

    /// Decodes interleaved frames into `out`, returning the number of **frames**
    /// written.
    ///
    /// At most `out.len() / channels` frames are produced. A return value of
    /// `0` together with [`SourceDecoder::is_exhausted`] returning `true`
    /// signals end of stream. `out` must be able to hold at least one whole
    /// frame or [`DecodeError::OutputTooSmall`] is returned.
    fn decode(&mut self, out: &mut [Sample]) -> Result<usize, DecodeError>;

    /// Seeks so that the next [`SourceDecoder::decode`] starts at `frame`.
    ///
    /// Seeking to the current frame count is allowed and leaves the decoder
    /// exhausted. Targets beyond the known frame count yield
    /// [`DecodeError::SeekOutOfRange`].
    fn seek(&mut self, frame: u64) -> Result<(), DecodeError>;

    /// Returns the next frame index that [`SourceDecoder::decode`] will emit.
    fn position(&self) -> u64;

    /// Returns `true` once every frame has been decoded.
    fn is_exhausted(&self) -> bool;

    /// Decodes the entire remaining stream into a freshly allocated,
    /// interleaved buffer.
    ///
    /// This is the convenience path used by the bank layer to turn a
    /// memory-resident asset into a fully decoded PCM block. It is a provided
    /// method implemented on top of [`SourceDecoder::decode`]; streaming
    /// callers use the incremental method directly.
    fn decode_to_end(&mut self) -> Result<Vec<Sample>, DecodeError> {
        let info = self.info();
        let channels = info.channels.max(1) as usize;
        let mut out: Vec<Sample> = Vec::new();
        if let Some(frames) = info.frame_count {
            out.reserve(frames as usize * channels);
        }
        // Decode in bounded chunks so a single call never allocates an
        // unbounded scratch buffer.
        let mut scratch = [0.0 as Sample; 4096];
        let usable = scratch.len() - (scratch.len() % channels);
        loop {
            let produced = self.decode(&mut scratch[..usable])?;
            if produced == 0 {
                break;
            }
            out.extend_from_slice(&scratch[..produced * channels]);
            if self.is_exhausted() {
                break;
            }
        }
        Ok(out)
    }
}
