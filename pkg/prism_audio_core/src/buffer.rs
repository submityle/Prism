//! Planar block storage for audio and the channel-layout descriptor.
//!
//! Buffers are **planar** (one contiguous slice per channel) rather than
//! interleaved because that layout is friendlier to per-channel DSP and to SIMD
//! auto-vectorization. All processing in the engine happens on fixed-size
//! blocks; a buffer stores `channels * frames` samples.

use alloc::vec;
use alloc::vec::Vec;

use crate::math::Sample;

/// Describes how the channels of a buffer map to physical speaker positions.
///
/// The engine mixes internally in whatever layout a bus declares and only
/// down-/up-mixes at explicit conversion nodes, mirroring the approach taken by
/// production engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ChannelLayout {
    /// Single channel.
    Mono,
    /// Left / Right.
    Stereo,
    /// 4.0 surround: FL, FR, SL, SR.
    Quad,
    /// 5.1 surround: FL, FR, C, LFE, SL, SR.
    Surround5_1,
    /// 7.1 surround: FL, FR, C, LFE, SL, SR, RL, RR.
    Surround7_1,
    /// First-order ambisonics (4 channels: W, X, Y, Z).
    AmbisonicFoa,
}

impl ChannelLayout {
    /// Returns the number of channels this layout occupies.
    #[inline]
    #[must_use]
    #[expect(
        clippy::match_same_arms,
        reason = "distinct layouts may share a channel count (e.g. Quad and FOA are both four channels); separate arms document each layout"
    )]
    pub const fn channel_count(self) -> usize {
        match self {
            ChannelLayout::Mono => 1,
            ChannelLayout::Stereo => 2,
            ChannelLayout::Quad => 4,
            ChannelLayout::Surround5_1 => 6,
            ChannelLayout::Surround7_1 => 8,
            ChannelLayout::AmbisonicFoa => 4,
        }
    }
}

/// A fixed-capacity, planar block of audio samples.
///
/// The capacity (channel count and maximum frame count) is fixed at
/// construction so that the real-time thread never reallocates. The *active*
/// frame count may shrink below capacity for the final (partial) block of a
/// stream via [`AudioBuffer::set_active_frames`].
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    layout: ChannelLayout,
    channels: usize,
    capacity_frames: usize,
    active_frames: usize,
    /// Channel-major storage: `data[ch * capacity_frames + frame]`.
    data: Vec<Sample>,
}

impl AudioBuffer {
    /// Allocates a silent buffer for `layout` holding up to `capacity_frames`.
    ///
    /// # Panics
    ///
    /// Panics if `capacity_frames` is zero.
    #[must_use]
    pub fn new(layout: ChannelLayout, capacity_frames: usize) -> Self {
        assert!(capacity_frames > 0, "capacity_frames must be non-zero");
        let channels = layout.channel_count();
        Self {
            layout,
            channels,
            capacity_frames,
            active_frames: capacity_frames,
            data: vec![0.0; channels * capacity_frames],
        }
    }

    /// Returns the channel layout.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns the number of channels.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the maximum number of frames the buffer can hold.
    #[inline]
    #[must_use]
    pub fn capacity_frames(&self) -> usize {
        self.capacity_frames
    }

    /// Returns the number of currently active frames.
    #[inline]
    #[must_use]
    pub fn active_frames(&self) -> usize {
        self.active_frames
    }

    /// Sets the active frame count (saturated at the capacity).
    #[inline]
    pub fn set_active_frames(&mut self, frames: usize) {
        self.active_frames = frames.min(self.capacity_frames);
    }

    /// Returns an immutable slice of the active samples for `channel`.
    ///
    /// # Panics
    ///
    /// Panics if `channel >= channels()`.
    #[inline]
    #[must_use]
    pub fn channel(&self, channel: usize) -> &[Sample] {
        assert!(channel < self.channels, "channel index out of range");
        let start = channel * self.capacity_frames;
        &self.data[start..start + self.active_frames]
    }

    /// Returns a mutable slice of the active samples for `channel`.
    ///
    /// # Panics
    ///
    /// Panics if `channel >= channels()`.
    #[inline]
    pub fn channel_mut(&mut self, channel: usize) -> &mut [Sample] {
        assert!(channel < self.channels, "channel index out of range");
        let start = channel * self.capacity_frames;
        &mut self.data[start..start + self.active_frames]
    }

    /// Returns mutable slices for two distinct channels simultaneously.
    ///
    /// Useful for stereo panning without borrow-checker gymnastics.
    ///
    /// # Panics
    ///
    /// Panics if the indices are equal or out of range.
    pub fn channel_pair_mut(&mut self, a: usize, b: usize) -> (&mut [Sample], &mut [Sample]) {
        assert!(a != b, "channel indices must differ");
        assert!(a < self.channels && b < self.channels, "channel out of range");
        let cap = self.capacity_frames;
        let active = self.active_frames;
        let (lo, hi, swapped) = if a < b { (a, b, false) } else { (b, a, true) };
        let (left, right) = self.data.split_at_mut(hi * cap);
        let first = &mut left[lo * cap..lo * cap + active];
        let second = &mut right[..active];
        if swapped { (second, first) } else { (first, second) }
    }

    /// Fills every active sample with silence.
    #[inline]
    pub fn clear(&mut self) {
        for v in &mut self.data {
            *v = 0.0;
        }
    }

    /// Copies the active samples from `other` into `self`.
    ///
    /// # Panics
    ///
    /// Panics if the layouts differ.
    pub fn copy_from(&mut self, other: &AudioBuffer) {
        assert_eq!(self.layout, other.layout, "layout mismatch on copy");
        let frames = self.active_frames.min(other.active_frames);
        for ch in 0..self.channels {
            let dst_start = ch * self.capacity_frames;
            let src_start = ch * other.capacity_frames;
            self.data[dst_start..dst_start + frames]
                .copy_from_slice(&other.data[src_start..src_start + frames]);
        }
    }

    /// Adds the active samples of `other` into `self`, scaled by `gain`.
    ///
    /// This is the fundamental mixing primitive used when several graph edges
    /// feed the same input port.
    ///
    /// # Panics
    ///
    /// Panics if the layouts differ.
    pub fn add_scaled(&mut self, other: &AudioBuffer, gain: Sample) {
        assert_eq!(self.layout, other.layout, "layout mismatch on mix");
        let frames = self.active_frames.min(other.active_frames);
        for ch in 0..self.channels {
            let dst_start = ch * self.capacity_frames;
            let src_start = ch * other.capacity_frames;
            let dst = &mut self.data[dst_start..dst_start + frames];
            let src = &other.data[src_start..src_start + frames];
            for (d, s) in dst.iter_mut().zip(src) {
                *d += *s * gain;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_silent() {
        let buf = AudioBuffer::new(ChannelLayout::Stereo, 64);
        assert_eq!(buf.channels(), 2);
        assert!(buf.channel(0).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn add_scaled_mixes() {
        let mut a = AudioBuffer::new(ChannelLayout::Mono, 4);
        let mut b = AudioBuffer::new(ChannelLayout::Mono, 4);
        b.channel_mut(0).copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        a.add_scaled(&b, 0.5);
        assert_eq!(a.channel(0), &[0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn channel_pair_mut_yields_both() {
        let mut buf = AudioBuffer::new(ChannelLayout::Stereo, 4);
        let (l, r) = buf.channel_pair_mut(0, 1);
        l[0] = 1.0;
        r[0] = -1.0;
        assert_eq!(buf.channel(0)[0], 1.0);
        assert_eq!(buf.channel(1)[0], -1.0);
    }

    #[test]
    fn channel_pair_mut_handles_reversed_order() {
        let mut buf = AudioBuffer::new(ChannelLayout::Stereo, 2);
        let (r, l) = buf.channel_pair_mut(1, 0);
        r[0] = 0.7;
        l[0] = 0.3;
        assert_eq!(buf.channel(1)[0], 0.7);
        assert_eq!(buf.channel(0)[0], 0.3);
    }
}
