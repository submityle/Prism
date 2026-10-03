//! Voice codec abstraction with a self-contained linear PCM default.
//!
//! Encoding and decoding are external insertion points: production builds route
//! voice through a low-latency codec such as Opus with in-band forward error
//! correction (design section 45.3). This module defines the [`VoiceCodec`]
//! trait for that integration and ships one real, dependency-free codec,
//! [`LinearPcmCodec`], which quantises samples to 16-bit signed little-endian
//! PCM so the crate is fully functional on its own. Nothing here is a stub: the
//! default codec losslessly round-trips within 16-bit quantisation error.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the codec insertion point of design section 45.3. Encoded bytes
//! travel as [`crate::transport::packet::VoicePacket`] payloads; the decoder
//! feeds [`crate::transport::plc`] and the downlink in [`crate::pipeline`].

use prism_audio_core::math::Sample;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Error returned when a payload cannot be decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum CodecError {
    /// The payload length is not consistent with the codec's frame format.
    MalformedPayload,
}

/// A pluggable voice codec.
///
/// A codec operates on fixed-size frames: [`VoiceCodec::frame_size`] samples in,
/// one payload out, and the reverse on decode. Both methods append to the
/// caller-provided buffers so the caller controls allocation.
pub trait VoiceCodec {
    /// The number of samples per encoded frame.
    fn frame_size(&self) -> usize;

    /// Encodes `samples` (exactly [`VoiceCodec::frame_size`] of them) into
    /// bytes appended to `out`.
    ///
    /// If the input length differs from the frame size, the implementation
    /// encodes what it can and pads or truncates to the frame size so the
    /// stream stays frame-aligned.
    fn encode(&mut self, samples: &[Sample], out: &mut Vec<u8>);

    /// Decodes `payload` into samples appended to `out`.
    ///
    /// Returns the number of decoded samples, or a [`CodecError`] if the
    /// payload is malformed.
    fn decode(&mut self, payload: &[u8], out: &mut Vec<Sample>) -> Result<usize, CodecError>;

    /// Resets any decoder/encoder state. Stateless codecs may ignore this.
    fn reset(&mut self) {}
}

/// A linear 16-bit signed little-endian PCM codec.
///
/// Samples are clamped to `[-1, 1]`, scaled by `32767`, rounded to the nearest
/// integer, and stored as two little-endian bytes each. Decoding reverses the
/// scaling. The codec is stateless and deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearPcmCodec {
    frame_size: usize,
}

impl LinearPcmCodec {
    /// Creates a codec for frames of `frame_size` samples (at least one).
    #[must_use]
    pub fn new(frame_size: usize) -> Self {
        Self {
            frame_size: frame_size.max(1),
        }
    }

    /// Quantises one sample to a 16-bit integer.
    #[inline]
    fn quantize(sample: Sample) -> i16 {
        let clamped = sample.clamp(-1.0, 1.0);
        let scaled = clamped * 32767.0;
        // Round half away from zero without relying on f32 inherent methods.
        let rounded = if scaled >= 0.0 {
            (scaled + 0.5) as i32
        } else {
            (scaled - 0.5) as i32
        };
        rounded.clamp(-32768, 32767) as i16
    }

    /// Reconstructs one sample from a 16-bit integer.
    #[inline]
    fn dequantize(value: i16) -> Sample {
        value as Sample / 32768.0
    }
}

impl VoiceCodec for LinearPcmCodec {
    fn frame_size(&self) -> usize {
        self.frame_size
    }

    fn encode(&mut self, samples: &[Sample], out: &mut Vec<u8>) {
        for i in 0..self.frame_size {
            let sample = samples.get(i).copied().unwrap_or(0.0);
            let q = Self::quantize(sample);
            let bytes = q.to_le_bytes();
            out.push(bytes[0]);
            out.push(bytes[1]);
        }
    }

    fn decode(&mut self, payload: &[u8], out: &mut Vec<Sample>) -> Result<usize, CodecError> {
        if !payload.len().is_multiple_of(2) {
            return Err(CodecError::MalformedPayload);
        }
        let count = payload.len() / 2;
        for pair in payload.chunks_exact(2) {
            let value = i16::from_le_bytes([pair[0], pair[1]]);
            out.push(Self::dequantize(value));
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;
    use core::f32::consts::PI;

    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    #[test]
    fn round_trip_within_quantization_error() {
        let mut codec = LinearPcmCodec::new(256);
        let input: Vec<Sample> = (0..256)
            .map(|i| 0.8 * ops::sin(2.0 * PI * i as Sample / 64.0))
            .collect();
        let mut payload = Vec::new();
        codec.encode(&input, &mut payload);
        assert_eq!(payload.len(), 512);
        let mut output = Vec::new();
        let n = codec.decode(&payload, &mut output).expect("decodes");
        assert_eq!(n, 256);
        for (a, b) in input.iter().zip(output.iter()) {
            // 16-bit quantisation step is about 3.05e-5.
            assert!(ops::abs(a - b) < 1.0e-4, "a={a} b={b}");
        }
    }

    #[test]
    fn malformed_payload_is_rejected() {
        let mut codec = LinearPcmCodec::new(4);
        let mut out = Vec::new();
        assert_eq!(
            codec.decode(&[1, 2, 3], &mut out),
            Err(CodecError::MalformedPayload)
        );
    }

    #[test]
    fn clamps_out_of_range_input() {
        let mut codec = LinearPcmCodec::new(2);
        let mut payload = Vec::new();
        codec.encode(&[2.0, -2.0], &mut payload);
        let mut out = Vec::new();
        codec.decode(&payload, &mut out).expect("decodes");
        assert!(out[0] <= 1.0 && out[1] >= -1.0);
        assert!(out[0] > 0.99 && out[1] < -0.99);
    }
}
