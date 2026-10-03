//! The import/decode matrix that unifies source bytes into planar `f32` PCM.
//!
//! This stage does not implement any codec itself: it orchestrates the
//! decoders already provided by [`prism_audio_assets::codec`] and lowers their
//! interleaved `f32` output into the crate-canonical [`ConditionedPcm`]
//! container, recording the source sample rate, channel layout, and any known
//! encoder pre-roll/padding.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the decode leg of design section 51 by reusing the section 44.1
//! codec matrix. Only formats that `prism_audio_assets::codec` can actually
//! decode are exposed; compressed families decoded by externally registered
//! plug-ins are intentionally absent from this offline matrix.


use prism_audio_assets::codec::{
    DecodeError as AssetDecodeError, DecoderRegistry, PcmDecoder, SourceDecoder,
};
use prism_audio_core::math::Sample;

use crate::config::DecodeHint;
use crate::pcm::{layout_for_channel_count, ConditionedPcm, EncoderDelay, PcmError};

/// The source byte format handed to [`decode`].
///
/// Every variant maps onto a decoder that `prism_audio_assets::codec` ships
/// natively. `Wav` probes the RIFF/WAVE container and dispatches by its `fmt `
/// tag; the `ImaAdpcm`/`MsAdpcm` variants assert that the decoded container
/// used the expected `ADPCM` tag; `RawPcm` decodes a headerless interleaved
/// block using the [`DecodeHint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SourceFormat {
    /// A RIFF/WAVE container (linear `PCM`, `IMA ADPCM`, or `MS ADPCM`).
    Wav,
    /// A headerless, tightly packed interleaved linear `PCM` block; the sample
    /// format, channel count, and rate come from the [`DecodeHint`].
    RawPcm,
    /// A RIFF/WAVE container whose `fmt ` tag selects the `IMA ADPCM` decoder.
    ImaAdpcm,
    /// A RIFF/WAVE container whose `fmt ` tag selects the `MS ADPCM` decoder.
    MsAdpcm,
}

/// Errors returned by the decode matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DecodeError {
    /// The underlying codec decoder failed.
    Codec(AssetDecodeError),
    /// Raw `PCM` decoding was requested without the sample format, channel
    /// count, or sample rate the [`DecodeHint`] must supply.
    MissingRawPcmHint,
    /// The decoded channel count has no canonical [`crate::pcm`] layout.
    UnsupportedChannelCount(usize),
    /// Assembling the planar container from decoded samples failed.
    Pcm(PcmError),
}

impl From<AssetDecodeError> for DecodeError {
    fn from(err: AssetDecodeError) -> Self {
        DecodeError::Codec(err)
    }
}

impl From<PcmError> for DecodeError {
    fn from(err: PcmError) -> Self {
        DecodeError::Pcm(err)
    }
}

/// Lowers an interleaved decoder into a [`ConditionedPcm`] plus its recorded
/// [`EncoderDelay`].
fn finish(
    channels: u16,
    sample_rate: u32,
    interleaved: &[Sample],
    hint: &DecodeHint,
) -> Result<(ConditionedPcm, EncoderDelay), DecodeError> {
    let channel_count = usize::from(channels);
    let layout = layout_for_channel_count(channel_count)
        .ok_or(DecodeError::UnsupportedChannelCount(channel_count))?;
    let pcm = ConditionedPcm::from_interleaved(sample_rate, layout, interleaved)?;
    let delay = EncoderDelay::new(hint.preroll_frames, hint.padding_frames);
    Ok((pcm, delay))
}

/// Decodes `bytes` of the given `format` into a planar [`ConditionedPcm`].
///
/// The returned [`EncoderDelay`] mirrors the pre-roll/padding recorded on the
/// [`DecodeHint`]; the decoders themselves report no delay for the native
/// lossless/`ADPCM` legs.
///
/// # Errors
///
/// Returns [`DecodeError::Codec`] when the codec fails, [`DecodeError::
/// MissingRawPcmHint`] when raw `PCM` is requested without a complete hint,
/// [`DecodeError::UnsupportedChannelCount`] for channel counts without a
/// canonical layout, and [`DecodeError::Pcm`] when the planar container
/// rejects the samples.
pub fn decode(
    bytes: &[u8],
    format: SourceFormat,
    hint: &DecodeHint,
) -> Result<(ConditionedPcm, EncoderDelay), DecodeError> {
    match format {
        SourceFormat::RawPcm => {
            let pcm_format = hint.pcm_format.ok_or(DecodeError::MissingRawPcmHint)?;
            let channels = hint.channels.ok_or(DecodeError::MissingRawPcmHint)?;
            let sample_rate = hint.sample_rate.ok_or(DecodeError::MissingRawPcmHint)?;
            let mut decoder = PcmDecoder::new(bytes.to_vec(), pcm_format, channels, sample_rate)?;
            let info = decoder.info();
            let interleaved = decoder.decode_to_end()?;
            finish(info.channels, info.sample_rate, &interleaved, hint)
        }
        SourceFormat::Wav | SourceFormat::ImaAdpcm | SourceFormat::MsAdpcm => {
            let registry = DecoderRegistry::with_native();
            let mut decoder = registry.decode_bytes(bytes)?;
            let info = decoder.info();
            let interleaved = decoder.decode_to_end()?;
            finish(info.channels, info.sample_rate, &interleaved, hint)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use prism_audio_assets::codec::PcmSampleFormat;
    use prism_audio_core::buffer::ChannelLayout;

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
    fn decodes_wav_pcm_s16_stereo() {
        let samples: [i16; 4] = [100, -100, 200, -200];
        let mut data = Vec::new();
        for s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let wav = build_wav(0x0001, 2, 48_000, 16, &data);
        let hint = DecodeHint::default();
        let (pcm, delay) = decode(&wav, SourceFormat::Wav, &hint).unwrap();
        assert_eq!(pcm.sample_rate(), 48_000);
        assert_eq!(pcm.layout(), ChannelLayout::Stereo);
        assert_eq!(pcm.frames(), 2);
        assert!((pcm.channel(0).unwrap()[1] - 200.0 / 32768.0).abs() < 1.0e-4);
        assert!(delay.is_zero());
    }

    #[test]
    fn decodes_raw_pcm_with_hint() {
        let samples: [i16; 3] = [0, 16_384, -16_384];
        let mut data = Vec::new();
        for s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let hint = DecodeHint {
            sample_rate: Some(44_100),
            channels: Some(1),
            pcm_format: Some(PcmSampleFormat::S16Le),
            preroll_frames: 7,
            padding_frames: 3,
        };
        let (pcm, delay) = decode(&data, SourceFormat::RawPcm, &hint).unwrap();
        assert_eq!(pcm.sample_rate(), 44_100);
        assert_eq!(pcm.frames(), 3);
        assert_eq!(delay, EncoderDelay::new(7, 3));
        assert!((pcm.channel(0).unwrap()[1] - 16_384.0 / 32768.0).abs() < 1.0e-4);
    }

    #[test]
    fn raw_pcm_without_hint_errors() {
        let hint = DecodeHint::default();
        let err = decode(&[0u8, 0, 0, 0], SourceFormat::RawPcm, &hint);
        assert_eq!(err.err(), Some(DecodeError::MissingRawPcmHint));
    }

    #[test]
    fn unsupported_channel_count_errors() {
        // Three channels have no canonical layout.
        let data = alloc::vec![0u8; 3 * 2 * 2];
        let wav = build_wav(0x0001, 3, 48_000, 16, &data);
        let hint = DecodeHint::default();
        let err = decode(&wav, SourceFormat::Wav, &hint);
        assert_eq!(err.err(), Some(DecodeError::UnsupportedChannelCount(3)));
    }
}
