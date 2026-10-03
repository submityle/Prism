//! RIFF/WAVE container parser that builds native PCM and ADPCM decoders.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The chunk layout
//! handled here is the publicly documented RIFF/WAVE specification.
//!
//! # Relationship
//! The container front door for the native legs of the section 44.1 codec
//! matrix. It locates the `fmt `, `fact`, and `data` chunks, resolves the
//! sample layout, and hands back a boxed [`SourceDecoder`] (PCM, IMA ADPCM, or
//! MS ADPCM). Compressed families (Vorbis/Opus/FLAC) are not WAV payloads and
//! are handled by externally registered decoders.

use alloc::boxed::Box;
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::codec::decoder::{DecodeError, SourceDecoder};
use crate::codec::ima_adpcm::ImaAdpcmDecoder;
use crate::codec::metadata::PcmSampleFormat;
use crate::codec::ms_adpcm::MsAdpcmDecoder;
use crate::codec::pcm::PcmDecoder;

const FORMAT_PCM: u16 = 0x0001;
const FORMAT_MS_ADPCM: u16 = 0x0002;
const FORMAT_IEEE_FLOAT: u16 = 0x0003;
const FORMAT_IMA_ADPCM: u16 = 0x0011;
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// A borrowing little-endian cursor over a byte slice.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], DecodeError> {
        if self.remaining() < len {
            return Err(DecodeError::UnexpectedEof);
        }
        let slice = &self.bytes[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    fn u16(&mut self) -> Result<u16, DecodeError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, DecodeError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// The parsed `fmt ` fields relevant to native decoding.
#[derive(Debug, Clone)]
struct WavFormat {
    format_tag: u16,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    samples_per_block: u16,
    coefficients: Vec<(i16, i16)>,
}

fn parse_fmt(chunk: &[u8]) -> Result<WavFormat, DecodeError> {
    let mut cur = Cursor::new(chunk);
    let mut format_tag = cur.u16()?;
    let channels = cur.u16()?;
    let sample_rate = cur.u32()?;
    let _byte_rate = cur.u32()?;
    let block_align = cur.u16()?;
    let bits_per_sample = cur.u16()?;
    let mut samples_per_block = 0u16;
    let mut coefficients: Vec<(i16, i16)> = Vec::new();

    if cur.remaining() >= 2 {
        let ext_size = cur.u16()?;
        let ext_start = cur.pos;
        if format_tag == FORMAT_EXTENSIBLE && ext_size >= 22 {
            let _valid_bits = cur.u16()?;
            let _channel_mask = cur.u32()?;
            // The first two bytes of the sub-format GUID carry the real tag.
            let guid = cur.take(16)?;
            format_tag = u16::from_le_bytes([guid[0], guid[1]]);
        } else {
            match format_tag {
                FORMAT_IMA_ADPCM => {
                    if ext_size >= 2 {
                        samples_per_block = cur.u16()?;
                    }
                }
                FORMAT_MS_ADPCM => {
                    if ext_size >= 2 {
                        samples_per_block = cur.u16()?;
                    }
                    if cur.remaining() >= 2 {
                        let num_coef = cur.u16()?;
                        for _ in 0..num_coef {
                            let c1 = cur.u16()? as i16;
                            let c2 = cur.u16()? as i16;
                            coefficients.push((c1, c2));
                        }
                    }
                }
                _ => {}
            }
        }
        // Keep the cursor consistent even when extensions are longer than read.
        let _consumed = cur.pos - ext_start;
    }

    Ok(WavFormat {
        format_tag,
        channels,
        sample_rate,
        block_align,
        bits_per_sample,
        samples_per_block,
        coefficients,
    })
}

/// Parses a complete in-memory WAV file and returns a boxed native decoder.
///
/// Supports linear PCM (8/16/24/32-bit integer and 32/64-bit float), IMA
/// ADPCM, and Microsoft ADPCM. Returns [`DecodeError::UnsupportedFormat`] for
/// any other format tag (which belongs to an externally registered decoder)
/// and [`DecodeError::MalformedHeader`] for structurally invalid files.
pub fn decode_wav(bytes: &[u8]) -> Result<Box<dyn SourceDecoder + Send>, DecodeError> {
    let mut cur = Cursor::new(bytes);
    if cur.take(4)? != b"RIFF" {
        return Err(DecodeError::MalformedHeader);
    }
    let _riff_size = cur.u32()?;
    if cur.take(4)? != b"WAVE" {
        return Err(DecodeError::MalformedHeader);
    }

    let mut format: Option<WavFormat> = None;
    let mut data: Option<Vec<u8>> = None;
    let mut fact_frames: Option<u64> = None;

    while cur.remaining() >= 8 {
        let id = cur.take(4)?;
        let id = [id[0], id[1], id[2], id[3]];
        let size = cur.u32()? as usize;
        let payload = cur.take(size.min(cur.remaining()))?;
        match &id {
            b"fmt " => format = Some(parse_fmt(payload)?),
            b"fact" if payload.len() >= 4 => {
                fact_frames = Some(u64::from(u32::from_le_bytes([
                    payload[0], payload[1], payload[2], payload[3],
                ])));
            }
            b"data" => data = Some(payload.to_vec()),
            _ => {}
        }
        // RIFF chunks are word-aligned: skip the pad byte for odd sizes.
        if size % 2 == 1 && cur.remaining() >= 1 {
            cur.take(1)?;
        }
    }

    let format = format.ok_or(DecodeError::MalformedHeader)?;
    let data = data.ok_or(DecodeError::MalformedHeader)?;

    match format.format_tag {
        FORMAT_PCM => {
            let sample_format = match format.bits_per_sample {
                8 => PcmSampleFormat::U8,
                16 => PcmSampleFormat::S16Le,
                24 => PcmSampleFormat::S24Le,
                32 => PcmSampleFormat::S32Le,
                _ => return Err(DecodeError::UnsupportedFormat),
            };
            let decoder =
                PcmDecoder::new(data, sample_format, format.channels, format.sample_rate)?;
            Ok(Box::new(decoder))
        }
        FORMAT_IEEE_FLOAT => {
            let sample_format = match format.bits_per_sample {
                32 => PcmSampleFormat::F32Le,
                64 => PcmSampleFormat::F64Le,
                _ => return Err(DecodeError::UnsupportedFormat),
            };
            let decoder =
                PcmDecoder::new(data, sample_format, format.channels, format.sample_rate)?;
            Ok(Box::new(decoder))
        }
        FORMAT_IMA_ADPCM => {
            let samples_per_block = resolve_samples_per_block(&format);
            let decoder = ImaAdpcmDecoder::new(
                data,
                format.channels,
                format.sample_rate,
                format.block_align as usize,
                samples_per_block,
            )?;
            // Prefer the `fact` frame count when present (handles padded blocks).
            Ok(wrap_with_fact(Box::new(decoder), fact_frames))
        }
        FORMAT_MS_ADPCM => {
            let samples_per_block = format.samples_per_block.max(2) as usize;
            let coefficients = if format.coefficients.is_empty() {
                crate::codec::ms_adpcm::DEFAULT_COEFFICIENTS.to_vec()
            } else {
                format.coefficients.clone()
            };
            let decoder = MsAdpcmDecoder::new(
                data,
                format.channels,
                format.sample_rate,
                format.block_align as usize,
                samples_per_block,
                coefficients,
            )?;
            Ok(wrap_with_fact(Box::new(decoder), fact_frames))
        }
        _ => Err(DecodeError::UnsupportedFormat),
    }
}

fn resolve_samples_per_block(format: &WavFormat) -> usize {
    if format.samples_per_block > 0 {
        return format.samples_per_block as usize;
    }
    // Derive from the block layout: 1 preamble frame per channel plus two
    // samples per data byte, divided across channels.
    let channels = format.channels.max(1) as usize;
    let header = 4 * channels;
    let block = format.block_align as usize;
    if block <= header {
        1
    } else {
        1 + (block - header) * 2 / channels
    }
}

/// Clamps an ADPCM decoder's reported frame count to the `fact` chunk value
/// when it is smaller (encoders pad the final block beyond the real frames).
fn wrap_with_fact(
    decoder: Box<dyn SourceDecoder + Send>,
    fact_frames: Option<u64>,
) -> Box<dyn SourceDecoder + Send> {
    match fact_frames {
        Some(frames) => Box::new(FrameLimited::new(decoder, frames)),
        None => decoder,
    }
}

/// Adapter that caps an inner decoder's reported and emitted frame count.
struct FrameLimited {
    inner: Box<dyn SourceDecoder + Send>,
    limit: u64,
}

impl FrameLimited {
    fn new(inner: Box<dyn SourceDecoder + Send>, limit: u64) -> Self {
        let info = inner.info();
        let effective = match info.frame_count {
            Some(count) => count.min(limit),
            None => limit,
        };
        Self {
            inner,
            limit: effective,
        }
    }
}

impl SourceDecoder for FrameLimited {
    fn info(&self) -> crate::codec::metadata::AudioStreamInfo {
        let mut info = self.inner.info();
        info.frame_count = Some(self.limit);
        info
    }

    fn decode(
        &mut self,
        out: &mut [Sample],
    ) -> Result<usize, DecodeError> {
        let channels = self.inner.info().channels.max(1) as usize;
        let remaining = self.limit.saturating_sub(self.inner.position()) as usize;
        if remaining == 0 {
            return Ok(0);
        }
        let max_frames = (out.len() / channels).min(remaining);
        self.inner.decode(&mut out[..max_frames * channels])
    }

    fn seek(&mut self, frame: u64) -> Result<(), DecodeError> {
        if frame > self.limit {
            return Err(DecodeError::SeekOutOfRange);
        }
        self.inner.seek(frame)
    }

    fn position(&self) -> u64 {
        self.inner.position()
    }

    fn is_exhausted(&self) -> bool {
        self.inner.position() >= self.limit
    }
}
