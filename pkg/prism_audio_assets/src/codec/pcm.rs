//! Native linear-PCM decoder over an in-memory data block.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! The lossless, memory-resident leg of the section 44.1 codec matrix. The WAV
//! container parser ([`crate::codec::wav`]) builds a [`PcmDecoder`] once it has
//! located the `data` chunk and resolved the sample layout; the decoder itself
//! is container-agnostic and simply walks interleaved samples.

use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::codec::decoder::{DecodeError, SourceDecoder};
use crate::codec::metadata::{AudioStreamInfo, CodecTag, PcmSampleFormat};

/// Decodes interleaved linear PCM from an owned byte block.
///
/// The decoder owns the raw sample bytes (the WAV `data` payload or any other
/// tightly packed interleaved PCM) and converts them to `f32` on demand. It is
/// fully seekable because every frame has a fixed byte stride.
#[derive(Debug, Clone)]
pub struct PcmDecoder {
    data: Vec<u8>,
    format: PcmSampleFormat,
    channels: u16,
    sample_rate: u32,
    frame_count: u64,
    frame_stride: usize,
    cursor_frame: u64,
}

impl PcmDecoder {
    /// Builds a decoder from a tightly packed interleaved PCM block.
    ///
    /// `data` must contain whole frames; trailing bytes that do not complete a
    /// frame are ignored. Returns [`DecodeError::MalformedHeader`] when
    /// `channels` is zero.
    pub fn new(
        data: Vec<u8>,
        format: PcmSampleFormat,
        channels: u16,
        sample_rate: u32,
    ) -> Result<Self, DecodeError> {
        if channels == 0 {
            return Err(DecodeError::MalformedHeader);
        }
        let frame_stride = format.bytes_per_sample() * channels as usize;
        let frame_count = (data.len() / frame_stride) as u64;
        Ok(Self {
            data,
            format,
            channels,
            sample_rate,
            frame_count,
            frame_stride,
            cursor_frame: 0,
        })
    }

    /// Returns the PCM sample layout this decoder reads.
    #[inline]
    #[must_use]
    pub fn format(&self) -> PcmSampleFormat {
        self.format
    }
}

impl SourceDecoder for PcmDecoder {
    fn info(&self) -> AudioStreamInfo {
        AudioStreamInfo::new(
            self.channels,
            self.sample_rate,
            Some(self.frame_count),
            CodecTag::Pcm,
        )
    }

    fn decode(&mut self, out: &mut [Sample]) -> Result<usize, DecodeError> {
        let channels = self.channels as usize;
        if out.len() < channels {
            return Err(DecodeError::OutputTooSmall);
        }
        let max_frames = out.len() / channels;
        let remaining = (self.frame_count - self.cursor_frame) as usize;
        let frames = max_frames.min(remaining);
        let bytes_per_sample = self.format.bytes_per_sample();
        for frame in 0..frames {
            let frame_base = (self.cursor_frame as usize + frame) * self.frame_stride;
            for channel in 0..channels {
                let sample_base = frame_base + channel * bytes_per_sample;
                let bytes = &self.data[sample_base..sample_base + bytes_per_sample];
                out[frame * channels + channel] = self.format.decode_sample(bytes);
            }
        }
        self.cursor_frame += frames as u64;
        Ok(frames)
    }

    fn seek(&mut self, frame: u64) -> Result<(), DecodeError> {
        if frame > self.frame_count {
            return Err(DecodeError::SeekOutOfRange);
        }
        self.cursor_frame = frame;
        Ok(())
    }

    fn position(&self) -> u64 {
        self.cursor_frame
    }

    fn is_exhausted(&self) -> bool {
        self.cursor_frame >= self.frame_count
    }
}
