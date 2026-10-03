//! A wait-free single-producer/single-consumer interleaved frame ring.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! The lock-free prefetch ring of design section 20: a decode task (producer)
//! writes whole decoded frames while the real-time audio thread (consumer)
//! reads them. Reads never block; when the ring underflows the consumer is
//! handed silence and an underrun counter is incremented (section 20's "output
//! silence on underrun, never block"). The structure contains no `unsafe`:
//! index arithmetic is wait-free and the shared occupancy is an atomic, so the
//! design maps directly onto an SPSC hand-off while remaining `no_std`.

use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use prism_audio_core::math::Sample;

/// The outcome of a consumer read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadResult {
    /// Number of frames filled from real decoded data.
    pub frames_from_data: usize,
    /// Number of trailing frames filled with silence due to underrun.
    pub frames_silenced: usize,
}

/// A fixed-capacity interleaved frame ring buffer.
///
/// Capacity is measured in frames; the backing store holds
/// `capacity_frames * channels` samples. The producer calls
/// [`FrameRing::write_interleaved`]; the real-time consumer calls
/// [`FrameRing::read_interleaved`], which always fills its output completely.
#[derive(Debug)]
pub struct FrameRing {
    buffer: Vec<Sample>,
    channels: usize,
    capacity_frames: usize,
    write_frame: usize,
    read_frame: usize,
    occupancy: AtomicUsize,
    underrun_frames: AtomicU64,
    total_written: u64,
    total_read: u64,
}

impl FrameRing {
    /// Creates a ring that stores up to `capacity_frames` frames of `channels`
    /// interleaved samples.
    ///
    /// # Panics
    ///
    /// Panics if `channels` or `capacity_frames` is zero: a zero-sized ring can
    /// never make progress and always indicates a construction bug.
    #[must_use]
    pub fn new(channels: usize, capacity_frames: usize) -> Self {
        assert!(channels > 0, "ring requires at least one channel");
        assert!(capacity_frames > 0, "ring requires non-zero capacity");
        Self {
            buffer: vec![0.0; capacity_frames * channels],
            channels,
            capacity_frames,
            write_frame: 0,
            read_frame: 0,
            occupancy: AtomicUsize::new(0),
            underrun_frames: AtomicU64::new(0),
            total_written: 0,
            total_read: 0,
        }
    }

    /// Returns the channel count.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the capacity in frames.
    #[inline]
    #[must_use]
    pub fn capacity_frames(&self) -> usize {
        self.capacity_frames
    }

    /// Returns the number of frames currently available to read.
    #[inline]
    #[must_use]
    pub fn available_to_read(&self) -> usize {
        self.occupancy.load(Ordering::Acquire)
    }

    /// Returns the number of frames of free space available to write.
    #[inline]
    #[must_use]
    pub fn available_to_write(&self) -> usize {
        self.capacity_frames - self.occupancy.load(Ordering::Acquire)
    }

    /// Returns `true` when the ring holds no readable frames.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.available_to_read() == 0
    }

    /// Returns `true` when the ring cannot accept another frame.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.available_to_write() == 0
    }

    /// Writes whole interleaved frames from `src`, returning the number of
    /// frames written.
    ///
    /// `src.len()` must be a multiple of the channel count. Writing stops when
    /// the ring is full; the caller retains the un-written tail.
    pub fn write_interleaved(&mut self, src: &[Sample]) -> usize {
        debug_assert_eq!(src.len() % self.channels, 0, "src must hold whole frames");
        let offered = src.len() / self.channels;
        let writable = offered.min(self.available_to_write());
        for frame in 0..writable {
            let dst_base = self.write_frame * self.channels;
            let src_base = frame * self.channels;
            self.buffer[dst_base..dst_base + self.channels]
                .copy_from_slice(&src[src_base..src_base + self.channels]);
            self.write_frame = (self.write_frame + 1) % self.capacity_frames;
        }
        if writable > 0 {
            self.occupancy.fetch_add(writable, Ordering::AcqRel);
            self.total_written += writable as u64;
        }
        writable
    }

    /// Fills `out` with interleaved frames, substituting silence on underrun.
    ///
    /// `out.len()` must be a multiple of the channel count. The output is
    /// always fully written: any frames beyond the available data are zeroed
    /// and counted as underrun. This is the real-time read path and never
    /// blocks.
    pub fn read_interleaved(&mut self, out: &mut [Sample]) -> ReadResult {
        debug_assert_eq!(out.len() % self.channels, 0, "out must hold whole frames");
        let requested = out.len() / self.channels;
        let available = self.available_to_read();
        let from_data = requested.min(available);
        for frame in 0..from_data {
            let src_base = self.read_frame * self.channels;
            let dst_base = frame * self.channels;
            out[dst_base..dst_base + self.channels]
                .copy_from_slice(&self.buffer[src_base..src_base + self.channels]);
            self.read_frame = (self.read_frame + 1) % self.capacity_frames;
        }
        if from_data > 0 {
            self.occupancy.fetch_sub(from_data, Ordering::AcqRel);
            self.total_read += from_data as u64;
        }
        let silenced = requested - from_data;
        if silenced > 0 {
            let tail = from_data * self.channels;
            for sample in &mut out[tail..] {
                *sample = 0.0;
            }
            self.underrun_frames
                .fetch_add(silenced as u64, Ordering::AcqRel);
        }
        ReadResult {
            frames_from_data: from_data,
            frames_silenced: silenced,
        }
    }

    /// Returns the cumulative number of frames silenced due to underrun.
    #[inline]
    #[must_use]
    pub fn underrun_frames(&self) -> u64 {
        self.underrun_frames.load(Ordering::Acquire)
    }

    /// Returns the cumulative number of frames written by the producer.
    #[inline]
    #[must_use]
    pub fn total_written(&self) -> u64 {
        self.total_written
    }

    /// Returns the cumulative number of frames read by the consumer.
    #[inline]
    #[must_use]
    pub fn total_read(&self) -> u64 {
        self.total_read
    }

    /// Discards all buffered frames and resets the underrun counter.
    pub fn clear(&mut self) {
        self.write_frame = 0;
        self.read_frame = 0;
        self.occupancy.store(0, Ordering::Release);
        self.underrun_frames.store(0, Ordering::Release);
    }
}
