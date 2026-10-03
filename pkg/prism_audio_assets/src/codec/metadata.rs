//! Codec-neutral stream metadata: sample formats, container and codec tags.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Supports the codec matrix of design section 44.1: a decoder reports its
//! channel count, sample rate, frame count, and codec tag through
//! [`AudioStreamInfo`] so the bank and streaming layers (section 20) can budget
//! memory and schedule decode work without knowing the concrete format.

#[cfg(not(feature = "std"))]
use alloc::string::String;

use prism_audio_core::math::Sample;

/// Linear PCM sample encodings understood by the self-contained PCM decoder.
///
/// These correspond to the sample layouts a WAV `fmt ` chunk can describe for
/// uncompressed audio. All of them are decoded losslessly into the engine's
/// [`Sample`] (`f32`) mixing format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum PcmSampleFormat {
    /// Unsigned 8-bit integer, bias `128` (WAV stores 8-bit PCM unsigned).
    U8,
    /// Signed 16-bit little-endian integer.
    S16Le,
    /// Signed 24-bit little-endian integer packed in three bytes.
    S24Le,
    /// Signed 32-bit little-endian integer.
    S32Le,
    /// 32-bit little-endian IEEE-754 float.
    F32Le,
    /// 64-bit little-endian IEEE-754 float.
    F64Le,
}

impl PcmSampleFormat {
    /// Returns the number of bytes a single sample of this format occupies.
    #[inline]
    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            PcmSampleFormat::U8 => 1,
            PcmSampleFormat::S16Le => 2,
            PcmSampleFormat::S24Le => 3,
            PcmSampleFormat::S32Le | PcmSampleFormat::F32Le => 4,
            PcmSampleFormat::F64Le => 8,
        }
    }

    /// Decodes one little-endian sample from `bytes` into the `f32` mixing
    /// format.
    ///
    /// `bytes` must contain at least [`PcmSampleFormat::bytes_per_sample`]
    /// bytes; only the leading bytes are read. Integer formats are normalised
    /// to the closed range `[-1.0, 1.0)` using the full-scale magnitude of the
    /// signed type (`U8` is first de-biased).
    #[must_use]
    pub fn decode_sample(self, bytes: &[u8]) -> Sample {
        match self {
            PcmSampleFormat::U8 => {
                let raw = i32::from(bytes[0]) - 128;
                raw as Sample / 128.0
            }
            PcmSampleFormat::S16Le => {
                let raw = i16::from_le_bytes([bytes[0], bytes[1]]);
                raw as Sample / 32768.0
            }
            PcmSampleFormat::S24Le => {
                let unsigned =
                    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16);
                // Sign-extend the 24-bit value into an i32.
                let raw = ((unsigned << 8) as i32) >> 8;
                raw as Sample / 8_388_608.0
            }
            PcmSampleFormat::S32Le => {
                let raw = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                raw as Sample / 2_147_483_648.0
            }
            PcmSampleFormat::F32Le => {
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            }
            PcmSampleFormat::F64Le => f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]) as Sample,
        }
    }
}

/// A codec family tag used to dispatch bytes to a registered decoder factory.
///
/// The three PCM/ADPCM families are implemented natively inside this crate.
/// The remaining tags are declared so banks can describe assets that are
/// decoded by externally registered [`SourceDecoder`] implementations (see the
/// crate-level documentation): this crate deliberately ships no stub decoders
/// for them.
///
/// [`SourceDecoder`]: crate::codec::decoder::SourceDecoder
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum CodecTag {
    /// Uncompressed linear PCM (native, lossless).
    Pcm,
    /// IMA / DVI ADPCM, 4 bits per sample (native).
    ImaAdpcm,
    /// Microsoft ADPCM, 4 bits per sample (native).
    MsAdpcm,
    /// Xiph Vorbis in an Ogg container (external decoder plug-in point).
    Vorbis,
    /// Xiph Opus (external decoder plug-in point).
    Opus,
    /// Free Lossless Audio Codec (external decoder plug-in point).
    Flac,
    /// A custom third-party codec identified by a stable string label.
    Custom(CustomCodecId),
}

impl CodecTag {
    /// Returns `true` when this crate ships a native decoder for the tag.
    #[inline]
    #[must_use]
    pub fn is_native(&self) -> bool {
        matches!(self, CodecTag::Pcm | CodecTag::ImaAdpcm | CodecTag::MsAdpcm)
    }
}

/// A stable identifier for a third-party codec contributed through the
/// decoder registry.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CustomCodecId(pub String);

/// Immutable description of a decoded audio stream.
///
/// Produced by every [`SourceDecoder`] so downstream layers can allocate
/// buffers, compute durations, and budget memory before any audio is decoded.
///
/// [`SourceDecoder`]: crate::codec::decoder::SourceDecoder
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioStreamInfo {
    /// Interleaved channel count (`1` = mono, `2` = stereo, ...).
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Total number of frames (one frame = one sample per channel), when known.
    ///
    /// Open-ended live streams report [`None`].
    pub frame_count: Option<u64>,
    /// The codec family that produced the stream.
    pub codec: CodecTag,
}

impl AudioStreamInfo {
    /// Creates a new descriptor.
    #[inline]
    #[must_use]
    pub fn new(
        channels: u16,
        sample_rate: u32,
        frame_count: Option<u64>,
        codec: CodecTag,
    ) -> Self {
        Self {
            channels,
            sample_rate,
            frame_count,
            codec,
        }
    }

    /// Returns the number of interleaved samples in `frames` frames.
    #[inline]
    #[must_use]
    pub const fn samples_for_frames(&self, frames: u64) -> u64 {
        frames * self.channels as u64
    }

    /// Returns the stream duration in seconds when both the frame count and a
    /// non-zero sample rate are known.
    #[must_use]
    pub fn duration_seconds(&self) -> Option<f64> {
        match self.frame_count {
            Some(frames) if self.sample_rate > 0 => {
                Some(frames as f64 / f64::from(self.sample_rate))
            }
            _ => None,
        }
    }
}
