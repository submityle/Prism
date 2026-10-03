//! Native IMA / DVI ADPCM decoder and encoder (WAV block layout).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The ADPCM step and
//! index tables are the publicly documented IMA/DVI constants; the predictor
//! recurrence is implemented from the published algorithm description.
//!
//! # Relationship
//! The low-cost, high-concurrency leg of the section 44.1 codec matrix for
//! short effects. Decodes the interleaved per-channel block layout used by the
//! WAV `0x0011` format tag. A matching [`ImaAdpcmEncoder`] is provided so the
//! codec has a deterministic round-trip for golden tests and so authoring
//! tools can produce assets without an external encoder.

use alloc::vec;
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::codec::decoder::{DecodeError, SourceDecoder};
use crate::codec::metadata::{AudioStreamInfo, CodecTag};

/// IMA ADPCM quantiser step-size table (89 entries, public constants).
const STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408,
    449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

/// IMA ADPCM index adaptation table (16 entries, public constants).
const INDEX_TABLE: [i32; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

#[inline]
fn clamp_index(index: i32) -> i32 {
    index.clamp(0, 88)
}

#[inline]
fn clamp_i16(value: i32) -> i32 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX))
}

/// A single ADPCM channel predictor state.
#[derive(Debug, Clone, Copy)]
struct ChannelState {
    predictor: i32,
    index: i32,
}

impl ChannelState {
    #[inline]
    fn step(&mut self, nibble: u8) -> i32 {
        let step = STEP_TABLE[self.index as usize];
        let mut diff = step >> 3;
        if nibble & 4 != 0 {
            diff += step;
        }
        if nibble & 2 != 0 {
            diff += step >> 1;
        }
        if nibble & 1 != 0 {
            diff += step >> 2;
        }
        if nibble & 8 != 0 {
            self.predictor -= diff;
        } else {
            self.predictor += diff;
        }
        self.predictor = clamp_i16(self.predictor);
        self.index = clamp_index(self.index + INDEX_TABLE[nibble as usize]);
        self.predictor
    }
}

/// Decoder for WAV-layout IMA ADPCM.
#[derive(Debug, Clone)]
pub struct ImaAdpcmDecoder {
    data: Vec<u8>,
    channels: u16,
    sample_rate: u32,
    block_align: usize,
    samples_per_block: usize,
    frame_count: u64,
    cursor_frame: u64,
    block_frames: Vec<Sample>,
    block_base_frame: u64,
}

impl ImaAdpcmDecoder {
    /// Builds a decoder from the raw ADPCM `data` payload.
    ///
    /// `block_align` is the WAV block size in bytes; `samples_per_block` is the
    /// number of frames each block decodes to (including the one preamble
    /// frame per channel). Returns [`DecodeError::MalformedHeader`] for
    /// degenerate parameters.
    pub fn new(
        data: Vec<u8>,
        channels: u16,
        sample_rate: u32,
        block_align: usize,
        samples_per_block: usize,
    ) -> Result<Self, DecodeError> {
        if channels == 0 || block_align < 4 * channels as usize || samples_per_block == 0 {
            return Err(DecodeError::MalformedHeader);
        }
        let block_count = data.len() / block_align;
        let tail = data.len() % block_align;
        // A trailing partial block (common when encoders pad the last block)
        // still contributes whole frames up to its data budget.
        let mut frame_count = (block_count * samples_per_block) as u64;
        if tail >= 4 * channels as usize {
            let tail_nibbles = (tail - 4 * channels as usize) * 2;
            let tail_frames = 1 + tail_nibbles / channels as usize;
            frame_count += tail_frames.min(samples_per_block) as u64;
        }
        Ok(Self {
            data,
            channels,
            sample_rate,
            block_align,
            samples_per_block,
            frame_count,
            cursor_frame: 0,
            block_frames: Vec::new(),
            block_base_frame: u64::MAX,
        })
    }

    fn decode_block(&self, block_index: usize) -> Vec<Sample> {
        let channels = self.channels as usize;
        let base = block_index * self.block_align;
        let end = (base + self.block_align).min(self.data.len());
        let block = &self.data[base..end];
        let mut states = vec![
            ChannelState {
                predictor: 0,
                index: 0,
            };
            channels
        ];
        // Per-channel 4-byte preamble header.
        for (channel, state) in states.iter_mut().enumerate() {
            let header = channel * 4;
            let predictor = i16::from_le_bytes([block[header], block[header + 1]]);
            state.predictor = i32::from(predictor);
            state.index = clamp_index(i32::from(block[header + 2]));
        }
        let mut frames: Vec<Sample> = Vec::with_capacity(self.samples_per_block * channels);
        // Frame 0 is the preamble sample of each channel.
        frames.resize(channels, 0.0);
        for (channel, state) in states.iter().enumerate() {
            frames[channel] = i16_to_sample(state.predictor);
        }
        // Remaining nibbles arrive as 4-byte words that round-robin channels.
        let data_start = channels * 4;
        let mut per_channel: Vec<Vec<Sample>> = vec![Vec::new(); channels];
        let mut offset = data_start;
        let mut channel = 0usize;
        while offset + 4 <= block.len() {
            let word = &block[offset..offset + 4];
            for &byte in word {
                let low = byte & 0x0F;
                let high = (byte >> 4) & 0x0F;
                let s0 = states[channel].step(low);
                per_channel[channel].push(i16_to_sample(s0));
                let s1 = states[channel].step(high);
                per_channel[channel].push(i16_to_sample(s1));
            }
            offset += 4;
            channel = (channel + 1) % channels;
        }
        // Interleave the decoded tails, bounded by samples_per_block - 1.
        let tail_frames = per_channel
            .iter()
            .map(Vec::len)
            .min()
            .unwrap_or(0)
            .min(self.samples_per_block - 1);
        for frame in 0..tail_frames {
            for ch in per_channel.iter() {
                frames.push(ch[frame]);
            }
        }
        frames
    }

    fn ensure_block(&mut self, block_index: usize) {
        let base_frame = (block_index * self.samples_per_block) as u64;
        if self.block_base_frame == base_frame && !self.block_frames.is_empty() {
            return;
        }
        self.block_frames = self.decode_block(block_index);
        self.block_base_frame = base_frame;
    }
}

#[inline]
fn i16_to_sample(value: i32) -> Sample {
    value as Sample / 32768.0
}

impl SourceDecoder for ImaAdpcmDecoder {
    fn info(&self) -> AudioStreamInfo {
        AudioStreamInfo::new(
            self.channels,
            self.sample_rate,
            Some(self.frame_count),
            CodecTag::ImaAdpcm,
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
            self.ensure_block(block_index);
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

/// Deterministic IMA ADPCM encoder (single channel) used for round-trip tests
/// and lightweight authoring.
///
/// The encoder emits one WAV-layout block per [`ImaAdpcmEncoder::encode_block`]
/// call using the standard greedy nibble search. It is intentionally mono: the
/// decoder handles any channel count, but the shipped encoder targets the
/// common mono effect case so its output is trivially verifiable.
#[derive(Debug, Clone)]
pub struct ImaAdpcmEncoder {
    samples_per_block: usize,
}

impl ImaAdpcmEncoder {
    /// Creates an encoder whose blocks hold `samples_per_block` frames.
    #[must_use]
    pub fn new(samples_per_block: usize) -> Self {
        Self {
            samples_per_block: samples_per_block.max(1),
        }
    }

    /// Returns the block size in bytes this encoder produces (mono).
    #[must_use]
    pub fn block_align(&self) -> usize {
        // 4-byte header + one nibble per remaining frame (two frames per byte).
        4 + (self.samples_per_block - 1).div_ceil(2)
    }

    /// Encodes one block of mono `i16` samples into WAV-layout IMA ADPCM.
    ///
    /// `samples.len()` must equal the configured frames-per-block. The returned
    /// buffer is exactly [`ImaAdpcmEncoder::block_align`] bytes.
    #[must_use]
    pub fn encode_block(&self, samples: &[i16]) -> Vec<u8> {
        let mut state = ChannelState {
            predictor: i32::from(samples[0]),
            index: 0,
        };
        let mut out = Vec::with_capacity(self.block_align());
        out.extend_from_slice(&samples[0].to_le_bytes());
        out.push(state.index as u8);
        out.push(0);
        let mut nibbles: Vec<u8> = Vec::with_capacity(self.samples_per_block - 1);
        for &target in &samples[1..] {
            nibbles.push(encode_nibble(&mut state, i32::from(target)));
        }
        // Pad to an even nibble count so the last byte is complete.
        if nibbles.len() % 2 == 1 {
            nibbles.push(0);
        }
        for pair in nibbles.chunks_exact(2) {
            out.push(pair[0] | (pair[1] << 4));
        }
        out
    }
}

fn encode_nibble(state: &mut ChannelState, target: i32) -> u8 {
    let step = STEP_TABLE[state.index as usize];
    let mut diff = target - state.predictor;
    let mut nibble = 0u8;
    if diff < 0 {
        nibble = 8;
        diff = -diff;
    }
    let mut vpdiff = step >> 3;
    if diff >= step {
        nibble |= 4;
        diff -= step;
        vpdiff += step;
    }
    if diff >= step >> 1 {
        nibble |= 2;
        diff -= step >> 1;
        vpdiff += step >> 1;
    }
    if diff >= step >> 2 {
        nibble |= 1;
        vpdiff += step >> 2;
    }
    if nibble & 8 != 0 {
        state.predictor -= vpdiff;
    } else {
        state.predictor += vpdiff;
    }
    state.predictor = clamp_i16(state.predictor);
    state.index = clamp_index(state.index + INDEX_TABLE[nibble as usize]);
    nibble
}
