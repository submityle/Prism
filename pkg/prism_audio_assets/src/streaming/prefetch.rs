//! Prefetching stream voice: resident first segment plus a decode-fed ring.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! The streaming playback unit of design section 20. The first segment is kept
//! resident so playback starts with zero latency; the remainder is decoded on
//! demand by a background pump into the [`FrameRing`] and read by the real-time
//! thread. Underruns fall through to silence (never block). The producer
//! ([`PrefetchStream::pump`]) and consumer ([`PrefetchStream::read`]) are the
//! two ends of the lock-free hand-off.

use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::codec::decoder::DecodeError;
use crate::codec::registry::BoxedDecoder;
use crate::streaming::ring::{FrameRing, ReadResult};

/// A streaming voice combining a resident prefetch segment and a ring.
pub struct PrefetchStream {
    decoder: BoxedDecoder,
    ring: FrameRing,
    channels: usize,
    resident: Vec<Sample>,
    prefetch_frames: usize,
    scratch: Vec<Sample>,
    decoder_exhausted: bool,
}

impl PrefetchStream {
    /// Builds a streaming voice around `decoder`.
    ///
    /// `prefetch_frames` whole frames are decoded eagerly and kept resident so
    /// the first [`PrefetchStream::read`] returns real audio immediately.
    /// `ring_capacity_frames` sizes the streamed look-ahead buffer and is
    /// raised to at least `prefetch_frames` so the resident segment always
    /// fits.
    pub fn new(
        decoder: BoxedDecoder,
        prefetch_frames: usize,
        ring_capacity_frames: usize,
    ) -> Result<Self, DecodeError> {
        let channels = decoder.info().channels.max(1) as usize;
        let capacity = ring_capacity_frames.max(prefetch_frames.max(1));
        let mut stream = Self {
            decoder,
            ring: FrameRing::new(channels, capacity),
            channels,
            resident: Vec::new(),
            prefetch_frames,
            scratch: Vec::new(),
            decoder_exhausted: false,
        };
        stream.scratch.resize(1024 * channels, 0.0);
        stream.fill_resident()?;
        Ok(stream)
    }

    fn fill_resident(&mut self) -> Result<(), DecodeError> {
        self.resident.clear();
        self.resident
            .reserve(self.prefetch_frames * self.channels);
        let mut remaining = self.prefetch_frames;
        while remaining > 0 {
            let want = remaining.min(self.scratch.len() / self.channels);
            let produced = self.decoder.decode(&mut self.scratch[..want * self.channels])?;
            if produced == 0 {
                self.decoder_exhausted = true;
                break;
            }
            self.resident
                .extend_from_slice(&self.scratch[..produced * self.channels]);
            remaining -= produced;
        }
        if self.decoder.is_exhausted() {
            self.decoder_exhausted = true;
        }
        // Seed the ring with the resident segment for a single read path.
        self.ring.write_interleaved(&self.resident);
        Ok(())
    }

    /// Returns the channel count of the stream.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the resident prefetch segment (interleaved frames).
    #[inline]
    #[must_use]
    pub fn resident_segment(&self) -> &[Sample] {
        &self.resident
    }

    /// Producer step: decode and enqueue up to `max_frames` frames into the
    /// ring, returning the number of frames enqueued.
    ///
    /// Call this from a background task. It stops early when the ring is full
    /// or the decoder is exhausted and never blocks.
    pub fn pump(&mut self, max_frames: usize) -> Result<usize, DecodeError> {
        let mut pushed = 0usize;
        while pushed < max_frames && !self.decoder_exhausted {
            let space = self.ring.available_to_write();
            if space == 0 {
                break;
            }
            let want = space
                .min(max_frames - pushed)
                .min(self.scratch.len() / self.channels);
            let produced = self.decoder.decode(&mut self.scratch[..want * self.channels])?;
            if produced == 0 {
                self.decoder_exhausted = true;
                break;
            }
            let written = self
                .ring
                .write_interleaved(&self.scratch[..produced * self.channels]);
            pushed += written;
            if self.decoder.is_exhausted() {
                self.decoder_exhausted = true;
            }
        }
        Ok(pushed)
    }

    /// Consumer step (real-time): fill `out` with interleaved frames.
    ///
    /// Always writes the whole buffer; missing frames are silence and counted
    /// as underrun.
    pub fn read(&mut self, out: &mut [Sample]) -> ReadResult {
        self.ring.read_interleaved(out)
    }

    /// Returns the number of frames immediately readable without underrun.
    #[inline]
    #[must_use]
    pub fn buffered_frames(&self) -> usize {
        self.ring.available_to_read()
    }

    /// Returns the cumulative underrun frame count.
    #[inline]
    #[must_use]
    pub fn underrun_frames(&self) -> u64 {
        self.ring.underrun_frames()
    }

    /// Returns `true` once the decoder is drained and the ring is empty.
    #[inline]
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.decoder_exhausted && self.ring.is_empty()
    }

    /// Restarts playback from the beginning using the resident segment.
    ///
    /// The resident first segment is re-seeded into the ring without decoding,
    /// preserving zero-latency restart; the decoder is seeked past it so the
    /// pump continues from the correct position.
    pub fn restart(&mut self) -> Result<(), DecodeError> {
        self.ring.clear();
        self.decoder.seek(self.resident.len() as u64 / self.channels as u64)?;
        self.decoder_exhausted = self.decoder.is_exhausted() && self.resident.is_empty();
        self.ring.write_interleaved(&self.resident);
        Ok(())
    }
}
