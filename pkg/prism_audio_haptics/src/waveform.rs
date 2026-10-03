//! Low-sample-rate haptic waveform storage and its actuator channel model.
//!
//! A haptic signal is felt rather than heard, so it is carried at a much lower
//! sample rate than audio (on the order of a kilohertz). [`HapticWaveform`]
//! stores one planar sample track per actuator channel, supports streaming
//! appends from the transcoder, and resamples between rates with linear
//! interpolation so a backend can retime the signal to its own update rate.
//!
//! The channel model is deliberately tiny: a device is either a single
//! wide-band actuator ([`ActuatorLayout::Mono`]) or a left/right pair
//! ([`ActuatorLayout::Dual`]) such as a two-motor rumble pad.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the haptic buffer of design section 36. It is the haptic-rate
//! analogue of the audio-rate planar block in `prism_audio_core::buffer` and is
//! the hand-off container between [`crate::transcode`] and
//! [`crate::backend`].

use alloc::vec::Vec;

use bevy_math::ops;
use prism_audio_core::math::Sample;

/// How many physical actuators a haptic device exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ActuatorLayout {
    /// A single wide-band actuator.
    Mono,
    /// A left/right actuator pair (for example a dual-motor rumble pad).
    Dual,
}

impl ActuatorLayout {
    /// Returns the number of actuator channels this layout occupies.
    #[inline]
    #[must_use]
    pub const fn channel_count(self) -> usize {
        match self {
            ActuatorLayout::Mono => 1,
            ActuatorLayout::Dual => 2,
        }
    }
}

/// A planar, low-rate haptic signal: one sample track per actuator channel.
///
/// The waveform grows as the transcoder pushes frames; it never reallocates
/// mid-frame because [`push_frame`](Self::push_frame) appends to every channel
/// in lock-step. All channels always hold the same number of frames.
#[derive(Debug, Clone)]
pub struct HapticWaveform {
    rate_hz: u32,
    layout: ActuatorLayout,
    channels: Vec<Vec<Sample>>,
}

impl HapticWaveform {
    /// Builds an empty waveform at `rate_hz` for the given actuator `layout`.
    ///
    /// The rate is clamped to at least one hertz so later resampling never
    /// divides by zero.
    #[must_use]
    pub fn new(rate_hz: u32, layout: ActuatorLayout) -> Self {
        let mut channels = Vec::with_capacity(layout.channel_count());
        for _ in 0..layout.channel_count() {
            channels.push(Vec::new());
        }
        Self {
            rate_hz: rate_hz.max(1),
            layout,
            channels,
        }
    }

    /// Returns the haptic sample rate in hertz.
    #[inline]
    #[must_use]
    pub fn rate_hz(&self) -> u32 {
        self.rate_hz
    }

    /// Returns the actuator layout.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> ActuatorLayout {
        self.layout
    }

    /// Returns the number of actuator channels.
    #[inline]
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Returns the number of frames stored per channel.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    /// Returns `true` when no frames have been pushed yet.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the stored samples for `channel`.
    ///
    /// # Panics
    ///
    /// Panics if `channel >= channel_count()`.
    #[inline]
    #[must_use]
    pub fn channel(&self, channel: usize) -> &[Sample] {
        &self.channels[channel]
    }

    /// Returns the stored samples for `channel` mutably.
    ///
    /// # Panics
    ///
    /// Panics if `channel >= channel_count()`.
    #[inline]
    pub fn channel_mut(&mut self, channel: usize) -> &mut [Sample] {
        &mut self.channels[channel]
    }

    /// Appends one frame, taking one value per channel from `frame`.
    ///
    /// Missing entries (when `frame` is shorter than the channel count) are
    /// filled with silence, so every channel keeps the same length.
    pub fn push_frame(&mut self, frame: &[Sample]) {
        for (index, channel) in self.channels.iter_mut().enumerate() {
            channel.push(frame.get(index).copied().unwrap_or(0.0));
        }
    }

    /// Removes every frame, keeping the rate and layout.
    pub fn clear(&mut self) {
        for channel in &mut self.channels {
            channel.clear();
        }
    }

    /// Returns a copy resampled to `target_rate_hz` by linear interpolation.
    ///
    /// Equal source and target rates copy the samples verbatim. The output
    /// frame count scales with the rate ratio; an empty input yields an empty
    /// output.
    #[must_use]
    pub fn resample(&self, target_rate_hz: u32) -> HapticWaveform {
        let target = target_rate_hz.max(1);
        let source = self.rate_hz.max(1);
        let src_len = self.len();
        let mut out = HapticWaveform::new(target, self.layout);
        if src_len == 0 {
            return out;
        }
        if target == source {
            for (dst, src) in out.channels.iter_mut().zip(&self.channels) {
                dst.extend_from_slice(src);
            }
            return out;
        }
        let new_len = ((src_len as u64 * u64::from(target)) / u64::from(source)) as usize;
        let ratio = source as Sample / target as Sample;
        let last = src_len - 1;
        for (dst, src) in out.channels.iter_mut().zip(&self.channels) {
            dst.reserve(new_len);
            for i in 0..new_len {
                let pos = i as Sample * ratio;
                let base_f = ops::floor(pos);
                let base = base_f as usize;
                let frac = pos - base_f;
                let a = src[base.min(last)];
                let b = src[(base + 1).min(last)];
                dst.push(a + (b - a) * frac);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_channel_counts() {
        assert_eq!(ActuatorLayout::Mono.channel_count(), 1);
        assert_eq!(ActuatorLayout::Dual.channel_count(), 2);
    }

    #[test]
    fn new_is_empty_with_right_shape() {
        let wf = HapticWaveform::new(1_000, ActuatorLayout::Dual);
        assert_eq!(wf.rate_hz(), 1_000);
        assert_eq!(wf.channel_count(), 2);
        assert!(wf.is_empty());
        assert_eq!(wf.len(), 0);
    }

    #[test]
    fn rate_is_clamped_to_one() {
        let wf = HapticWaveform::new(0, ActuatorLayout::Mono);
        assert_eq!(wf.rate_hz(), 1);
    }

    #[test]
    fn push_frame_keeps_channels_aligned() {
        let mut wf = HapticWaveform::new(1_000, ActuatorLayout::Dual);
        wf.push_frame(&[0.25, 0.5]);
        wf.push_frame(&[0.75]);
        assert_eq!(wf.len(), 2);
        assert!((wf.channel(0)[0] - 0.25).abs() < 1e-6);
        assert!((wf.channel(1)[0] - 0.5).abs() < 1e-6);
        // Short frame fills the missing channel with silence.
        assert!((wf.channel(1)[1]).abs() < 1e-6);
    }

    #[test]
    fn clear_drops_frames() {
        let mut wf = HapticWaveform::new(1_000, ActuatorLayout::Mono);
        wf.push_frame(&[1.0]);
        assert_eq!(wf.len(), 1);
        wf.clear();
        assert!(wf.is_empty());
    }

    #[test]
    fn resample_same_rate_copies() {
        let mut wf = HapticWaveform::new(1_000, ActuatorLayout::Mono);
        for v in [0.1, 0.2, 0.3] {
            wf.push_frame(&[v]);
        }
        let out = wf.resample(1_000);
        assert_eq!(out.len(), 3);
        for (a, b) in out.channel(0).iter().zip(wf.channel(0)) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn resample_downsamples_length() {
        let mut wf = HapticWaveform::new(2_000, ActuatorLayout::Mono);
        for i in 0..100 {
            wf.push_frame(&[i as Sample]);
        }
        let out = wf.resample(1_000);
        assert_eq!(out.len(), 50);
        // Linear ramp stays monotincreasing after decimation.
        let ch = out.channel(0);
        assert!(ch[0] < ch[ch.len() - 1]);
    }

    #[test]
    fn resample_empty_stays_empty() {
        let wf = HapticWaveform::new(1_000, ActuatorLayout::Dual);
        let out = wf.resample(500);
        assert!(out.is_empty());
        assert_eq!(out.rate_hz(), 500);
    }

    #[test]
    fn resample_interpolates_midpoint() {
        let mut wf = HapticWaveform::new(2, ActuatorLayout::Mono);
        wf.push_frame(&[0.0]);
        wf.push_frame(&[1.0]);
        // Upsample 2 Hz -> 4 Hz: new frame 1 sits at source position 0.5.
        let out = wf.resample(4);
        assert!((out.channel(0)[1] - 0.5).abs() < 1e-6);
    }
}
