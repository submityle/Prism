//! The section 44.1 codec matrix: a pluggable decoder trait with native
//! self-contained PCM and ADPCM implementations.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Realises the `SourceDecoder` matrix referenced by design sections 10, 20,
//! and 44.1.
//!
//! # Native vs external decoders
//!
//! This crate implements, from scratch, the lossless and low-cost legs of the
//! matrix:
//!
//! - [`pcm::PcmDecoder`] - linear PCM (8/16/24-bit integer, 32-bit integer,
//!   32/64-bit float), the lossless memory-resident format.
//! - [`ima_adpcm::ImaAdpcmDecoder`] - IMA/DVI ADPCM (with a matching encoder).
//! - [`ms_adpcm::MsAdpcmDecoder`] - Microsoft ADPCM (with a matching encoder).
//! - [`wav`] - the RIFF/WAVE container that feeds all three.
//!
//! The compressed families in the matrix (Vorbis, Opus, FLAC) are **not**
//! implemented here and ship no stub: they are external
//! [`decoder::SourceDecoder`] implementations that a dedicated decoder crate
//! registers through [`registry::DecoderRegistry::register_tag`]. This keeps
//! the asset crate free of heavy third-party codec dependencies while leaving
//! a real, exercised integration seam.

pub mod decoder;
pub mod ima_adpcm;
pub mod metadata;
pub mod ms_adpcm;
pub mod pcm;
pub mod registry;
pub mod wav;

pub use decoder::{DecodeError, SourceDecoder};
pub use ima_adpcm::{ImaAdpcmDecoder, ImaAdpcmEncoder};
pub use metadata::{AudioStreamInfo, CodecTag, CustomCodecId, PcmSampleFormat};
pub use ms_adpcm::{MsAdpcmDecoder, MsAdpcmEncoder};
pub use pcm::PcmDecoder;
pub use registry::{BoxedDecoder, DecoderRegistry};
pub use wav::decode_wav;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const EPSILON: f32 = 1.0e-4;

    /// Builds a minimal canonical WAV file in memory.
    fn build_wav(
        format_tag: u16,
        channels: u16,
        sample_rate: u32,
        bits_per_sample: u16,
        data: &[u8],
    ) -> Vec<u8> {
        let block_align = channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * u32::from(block_align);
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&format_tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&bits_per_sample.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn pcm_s16_roundtrip_is_exact() {
        let samples: [i16; 6] = [0, 16384, -16384, 32767, -32768, 8192];
        let mut data = Vec::new();
        for s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let wav = build_wav(0x0001, 1, 48_000, 16, &data);
        let mut decoder = decode_wav(&wav).unwrap();
        let info = decoder.info();
        assert_eq!(info.channels, 1);
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.frame_count, Some(6));
        let decoded = decoder.decode_to_end().unwrap();
        assert_eq!(decoded.len(), 6);
        for (raw, got) in samples.iter().zip(decoded.iter()) {
            let expected = f32::from(*raw) / 32768.0;
            assert!((expected - got).abs() < EPSILON, "{expected} vs {got}");
        }
    }

    #[test]
    fn pcm_u8_is_debiased() {
        let data: [u8; 4] = [128, 255, 0, 192];
        let wav = build_wav(0x0001, 1, 44_100, 8, &data);
        let mut decoder = decode_wav(&wav).unwrap();
        let decoded = decoder.decode_to_end().unwrap();
        assert!((decoded[0] - 0.0).abs() < EPSILON);
        assert!(decoded[1] > 0.9);
        assert!(decoded[2] < -0.9);
    }

    #[test]
    fn pcm_f32_passthrough() {
        let samples: [f32; 3] = [0.25, -0.5, 0.75];
        let mut data = Vec::new();
        for s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let wav = build_wav(0x0003, 1, 48_000, 32, &data);
        let mut decoder = decode_wav(&wav).unwrap();
        let decoded = decoder.decode_to_end().unwrap();
        for (raw, got) in samples.iter().zip(decoded.iter()) {
            assert!((raw - got).abs() < EPSILON);
        }
    }

    #[test]
    fn pcm_stereo_interleaves_and_seeks() {
        // Two frames, stereo: L0,R0,L1,R1.
        let samples: [i16; 4] = [100, -100, 200, -200];
        let mut data = Vec::new();
        for s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let wav = build_wav(0x0001, 2, 48_000, 16, &data);
        let mut decoder = decode_wav(&wav).unwrap();
        decoder.seek(1).unwrap();
        let mut out = [0.0f32; 2];
        let produced = decoder.decode(&mut out).unwrap();
        assert_eq!(produced, 1);
        assert!((out[0] - 200.0 / 32768.0).abs() < EPSILON);
        assert!((out[1] + 200.0 / 32768.0).abs() < EPSILON);
    }

    #[test]
    fn ima_adpcm_roundtrip_bounded_error() {
        // A smooth ramp/sine-like signal the predictor can track well.
        let samples_per_block = 505; // 4-byte header + 250 data bytes -> 505.
        let encoder = ImaAdpcmEncoder::new(samples_per_block);
        let mut pcm: Vec<i16> = Vec::with_capacity(samples_per_block);
        for n in 0..samples_per_block {
            let phase = n as f32 * 0.05;
            pcm.push((bevy_math::ops::sin(phase) * 8000.0) as i16);
        }
        let block = encoder.encode_block(&pcm);
        assert_eq!(block.len(), encoder.block_align());

        let mut decoder = ImaAdpcmDecoder::new(
            block,
            1,
            48_000,
            encoder.block_align(),
            samples_per_block,
        )
        .unwrap();
        let decoded = decoder.decode_to_end().unwrap();
        assert_eq!(decoded.len(), samples_per_block);
        // The preamble sample is exact.
        assert!((decoded[0] - f32::from(pcm[0]) / 32768.0).abs() < EPSILON);
        // The remaining samples track within an ADPCM error bound.
        let mut max_err = 0.0f32;
        for (raw, got) in pcm.iter().zip(decoded.iter()) {
            let err = (f32::from(*raw) / 32768.0 - got).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 0.05, "IMA ADPCM error too large: {max_err}");
    }

    #[test]
    fn ms_adpcm_roundtrip_bounded_error() {
        let samples_per_block = 500;
        let encoder = MsAdpcmEncoder::new(samples_per_block);
        let mut pcm: Vec<i16> = Vec::with_capacity(samples_per_block);
        for n in 0..samples_per_block {
            let phase = n as f32 * 0.03;
            pcm.push((bevy_math::ops::sin(phase) * 9000.0) as i16);
        }
        let block = encoder.encode_block(&pcm);
        assert_eq!(block.len(), encoder.block_align());

        let mut decoder = MsAdpcmDecoder::new(
            block,
            1,
            48_000,
            encoder.block_align(),
            samples_per_block,
            ms_adpcm::DEFAULT_COEFFICIENTS.to_vec(),
        )
        .unwrap();
        let decoded = decoder.decode_to_end().unwrap();
        assert_eq!(decoded.len(), samples_per_block);
        // The two preamble samples are exact.
        assert!((decoded[0] - f32::from(pcm[0]) / 32768.0).abs() < EPSILON);
        assert!((decoded[1] - f32::from(pcm[1]) / 32768.0).abs() < EPSILON);
        let mut max_err = 0.0f32;
        for (raw, got) in pcm.iter().zip(decoded.iter()) {
            let err = (f32::from(*raw) / 32768.0 - got).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 0.05, "MS ADPCM error too large: {max_err}");
    }

    #[test]
    fn registry_dispatches_wav_by_magic() {
        let data: [u8; 4] = [0, 0, 0, 64];
        let wav = build_wav(0x0003, 1, 48_000, 32, &data);
        let registry = DecoderRegistry::with_native();
        let mut decoder = registry.decode_bytes(&wav).unwrap();
        assert_eq!(decoder.info().codec, CodecTag::Pcm);
        let decoded = decoder.decode_to_end().unwrap();
        assert!((decoded[0] - 2.0).abs() < EPSILON);
    }

    #[test]
    fn registry_reports_missing_external_codec() {
        let registry = DecoderRegistry::with_native();
        assert!(!registry.supports_tag(&CodecTag::Opus));
        let err = registry.decode_tagged(&CodecTag::Opus, &[0, 1, 2, 3]);
        assert_eq!(err.err(), Some(DecodeError::UnsupportedFormat));
    }

    #[test]
    fn unsupported_container_is_rejected() {
        let registry = DecoderRegistry::with_native();
        let err = registry.decode_bytes(b"OggS....");
        assert_eq!(err.err(), Some(DecodeError::UnsupportedFormat));
    }
}
