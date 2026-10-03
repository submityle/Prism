//! Native Microsoft ADPCM decoder and encoder (WAV block layout).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The adaptation
//! table and default coefficient set are the publicly documented Microsoft
//! ADPCM constants; the predictor recurrence follows the published formula.
//!
//! # Relationship
//! The second low-cost leg of the section 44.1 codec matrix. Decodes the WAV
//! `0x0002` format tag with per-channel predictor coefficients. A mono
//! [`MsAdpcmEncoder`] (predictor 0) provides a deterministic round-trip for
//! golden tests.

use alloc::vec;
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::codec::decoder::{DecodeError, SourceDecoder};
use crate::codec::metadata::{AudioStreamInfo, CodecTag};

/// Microsoft ADPCM step adaptation table (16 entries, public constants).
const ADAPTATION_TABLE: [i32; 16] = [
    230, 230, 230, 230, 307, 409, 512, 614, 768, 614, 512, 409, 307, 230, 230, 230,
];

/// The seven default predictor coefficient pairs (`coef1`, `coef2`), scaled by
/// 256. These are the standard values an encoder stores in the `fmt ` chunk;
/// they are also used by the shipped encoder.
pub const DEFAULT_COEFFICIENTS: [(i16, i16); 7] = [
    (256, 0),
    (512, -256),
    (0, 0),
    (192, 64),
    (240, 0),
    (460, -208),
    (392, -232),
];

#[inline]
fn clamp_i16(value: i32) -> i32 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX))
}

#[inline]
fn sign_extend_nibble(nibble: u8) -> i32 {
    let value = i32::from(nibble & 0x0F);
    if value >= 8 { value - 16 } else { value }
}

#[inline]
fn i16_to_sample(value: i32) -> Sample {
    value as Sample / 32768.0
}

#[derive(Debug, Clone, Copy)]
struct ChannelState {
    coef1: i32,
    coef2: i32,
    delta: i32,
    sample1: i32,
    sample2: i32,
}

impl ChannelState {
    #[inline]
    fn step(&mut self, nibble: u8) -> i32 {
        let predict = (self.sample1 * self.coef1 + self.sample2 * self.coef2) >> 8;
        let new = clamp_i16(predict + sign_extend_nibble(nibble) * self.delta);
        self.sample2 = self.sample1;
        self.sample1 = new;
        self.delta = (ADAPTATION_TABLE[(nibble & 0x0F) as usize] * self.delta) >> 8;
        if self.delta < 16 {
            self.delta = 16;
        }
        new
    }
}

/// Decoder for WAV-layout Microsoft ADPCM.
#[derive(Debug, Clone)]
pub struct MsAdpcmDecoder {
    data: Vec<u8>,
    channels: u16,
    sample_rate: u32,
    block_align: usize,
    samples_per_block: usize,
    coefficients: Vec<(i16, i16)>,
    frame_count: u64,
    cursor_frame: u64,
    block_frames: Vec<Sample>,
    block_base_frame: u64,
}

impl MsAdpcmDecoder {
    /// Builds a decoder from the raw ADPCM `data` payload.
    ///
    /// `coefficients` are the predictor pairs from the `fmt ` chunk (use
    /// [`DEFAULT_COEFFICIENTS`] when the stream carries the standard set).
    /// Returns [`DecodeError::MalformedHeader`] for degenerate parameters.
    pub fn new(
        data: Vec<u8>,
        channels: u16,
        sample_rate: u32,
        block_align: usize,
        samples_per_block: usize,
        coefficients: Vec<(i16, i16)>,
    ) -> Result<Self, DecodeError> {
        let header_bytes = 7 * channels as usize;
        if channels == 0
            || channels > 2
            || block_align < header_bytes
            || samples_per_block < 2
            || coefficients.is_empty()
        {
            return Err(DecodeError::MalformedHeader);
        }
        let block_count = data.len() / block_align;
        let frame_count = (block_count * samples_per_block) as u64;
        Ok(Self {
            data,
            channels,
            sample_rate,
            block_align,
            samples_per_block,
            coefficients,
            frame_count,
            cursor_frame: 0,
            block_frames: Vec::new(),
            block_base_frame: u64::MAX,
        })
    }

    fn read_i16(block: &[u8], offset: usize) -> i32 {
        i32::from(i16::from_le_bytes([block[offset], block[offset + 1]]))
    }

    fn decode_block(&self, block_index: usize) -> Result<Vec<Sample>, DecodeError> {
        let channels = self.channels as usize;
        let base = block_index * self.block_align;
        let end = (base + self.block_align).min(self.data.len());
        let block = &self.data[base..end];
        if block.len() < 7 * channels {
            return Err(DecodeError::UnexpectedEof);
        }
        let mut states = vec![
            ChannelState {
                coef1: 0,
                coef2: 0,
                delta: 16,
                sample1: 0,
                sample2: 0,
            };
            channels
        ];
        // Header: predictor indices, then deltas, then sample1s, then sample2s.
        for (channel, state) in states.iter_mut().enumerate() {
            let predictor = block[channel] as usize;
            if predictor >= self.coefficients.len() {
                return Err(DecodeError::MalformedHeader);
            }
            let (c1, c2) = self.coefficients[predictor];
            state.coef1 = i32::from(c1);
            state.coef2 = i32::from(c2);
        }
        let mut offset = channels;
        for state in states.iter_mut() {
            state.delta = Self::read_i16(block, offset);
            offset += 2;
        }
        for state in states.iter_mut() {
            state.sample1 = Self::read_i16(block, offset);
            offset += 2;
        }
        for state in states.iter_mut() {
            state.sample2 = Self::read_i16(block, offset);
            offset += 2;
        }
        let mut frames: Vec<Sample> = Vec::with_capacity(self.samples_per_block * channels);
        // Preamble frames: sample2 then sample1 (oldest first).
        for state in states.iter() {
            frames.push(i16_to_sample(state.sample2));
        }
        for state in states.iter() {
            frames.push(i16_to_sample(state.sample1));
        }
        // Decode nibbles: high nibble first, round-robin across channels.
        let mut per_channel: Vec<Vec<Sample>> = vec![Vec::new(); channels];
        let mut channel = 0usize;
        while offset < block.len() {
            let byte = block[offset];
            let high = (byte >> 4) & 0x0F;
            let low = byte & 0x0F;
            let s_high = states[channel].step(high);
            per_channel[channel].push(i16_to_sample(s_high));
            channel = (channel + 1) % channels;
            let s_low = states[channel].step(low);
            per_channel[channel].push(i16_to_sample(s_low));
            channel = (channel + 1) % channels;
            offset += 1;
        }
        let tail_frames = per_channel
            .iter()
            .map(Vec::len)
            .min()
            .unwrap_or(0)
            .min(self.samples_per_block - 2);
        for frame in 0..tail_frames {
            for ch in per_channel.iter() {
                frames.push(ch[frame]);
            }
        }
        Ok(frames)
    }

    fn ensure_block(&mut self, block_index: usize) -> Result<(), DecodeError> {
        let base_frame = (block_index * self.samples_per_block) as u64;
        if self.block_base_frame == base_frame && !self.block_frames.is_empty() {
            return Ok(());
        }
        self.block_frames = self.decode_block(block_index)?;
        self.block_base_frame = base_frame;
        Ok(())
    }
}

impl SourceDecoder for MsAdpcmDecoder {
    fn info(&self) -> AudioStreamInfo {
        AudioStreamInfo::new(
            self.channels,
            self.sample_rate,
            Some(self.frame_count),
            CodecTag::MsAdpcm,
        )
    }

    fn decode(&mut self, out: &mut [Sample]) -> Result<usize, DecodeError> {
        let channels = self.channels as usize;
        if out.len() < channels {
            return Err(DecodeError::OutputTooSmall);
        }
        let max_frames = out.len() / channels;
        let mut produced = 0usize;
        while produced < max_frames && self.cursor_frame < self.frame_count {
            let block_index = (self.cursor_frame / self.samples_per_block as u64) as usize;
            self.ensure_block(block_index)?;
            let within = (self.cursor_frame - self.block_base_frame) as usize;
            let available = self.block_frames.len() / channels;
            if within >= available {
                break;
            }
            let src = &self.block_frames[within * channels..(within + 1) * channels];
            out[produced * channels..(produced + 1) * channels].copy_from_slice(src);
            produced += 1;
            self.cursor_frame += 1;
        }
        Ok(produced)
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

/// Deterministic mono Microsoft ADPCM encoder (predictor 0) for round-trip
/// tests and lightweight authoring.
///
/// Predictor index 0 selects the `(256, 0)` coefficient pair, so the predicted
/// sample equals the previous reconstructed sample. The encoder then performs
/// an exhaustive 4-bit nibble search per sample to minimise reconstruction
/// error.
#[derive(Debug, Clone)]
pub struct MsAdpcmEncoder {
    samples_per_block: usize,
}

impl MsAdpcmEncoder {
    /// Creates an encoder whose blocks hold `samples_per_block` frames
    /// (minimum 2 for the two preamble samples).
    #[must_use]
    pub fn new(samples_per_block: usize) -> Self {
        Self {
            samples_per_block: samples_per_block.max(2),
        }
    }

    /// Returns the mono block size in bytes this encoder produces.
    #[must_use]
    pub fn block_align(&self) -> usize {
        // 7-byte header + one nibble per sample after the two preamble frames.
        7 + (self.samples_per_block - 2).div_ceil(2)
    }

    /// Encodes one block of mono `i16` samples into WAV-layout MS ADPCM.
    ///
    /// `samples.len()` must equal the configured frames-per-block.
    #[must_use]
    pub fn encode_block(&self, samples: &[i16]) -> Vec<u8> {
        let (c1, c2) = DEFAULT_COEFFICIENTS[0];
        let mut state = ChannelState {
            coef1: i32::from(c1),
            coef2: i32::from(c2),
            delta: 16,
            sample1: i32::from(samples[1]),
            sample2: i32::from(samples[0]),
        };
        let mut out = Vec::with_capacity(self.block_align());
        out.push(0); // predictor index 0
        out.extend_from_slice(&(state.delta as i16).to_le_bytes());
        out.extend_from_slice(&samples[1].to_le_bytes());
        out.extend_from_slice(&samples[0].to_le_bytes());
        let mut nibbles: Vec<u8> = Vec::with_capacity(self.samples_per_block - 2);
        for &target in &samples[2..] {
            nibbles.push(encode_nibble(&mut state, i32::from(target)));
        }
        if nibbles.len() % 2 == 1 {
            nibbles.push(0);
        }
        for pair in nibbles.chunks_exact(2) {
            out.push((pair[0] << 4) | pair[1]);
        }
        out
    }
}

fn encode_nibble(state: &mut ChannelState, target: i32) -> u8 {
    let predict = (state.sample1 * state.coef1 + state.sample2 * state.coef2) >> 8;
    // Exhaustive search over the 16 signed nibbles for the smallest error.
    let mut best_nibble = 0u8;
    let mut best_error = i64::MAX;
    for candidate in 0u8..16 {
        let reconstructed = clamp_i16(predict + sign_extend_nibble(candidate) * state.delta);
        let error = i64::from(reconstructed - target).abs();
        if error < best_error {
            best_error = error;
            best_nibble = candidate;
        }
    }
    state.step(best_nibble);
    best_nibble
}
