//! One-touch mono downmix for players with single-sided hearing.
//!
//! [`MonoDownmix`] produces an equal-power fold-down matrix for any
//! [`ChannelLayout`]: a per-source-channel coefficient vector such that the
//! mono sum preserves the acoustic power of uncorrelated channels. The output
//! downmix stage consumes the resulting [`DownmixMatrix`]; this module only
//! computes coefficients and offers a reference fold helper for testing.
//!
//! The equal-power rule gives every contributing channel the gain
//! `1 / sqrt(n)`, where `n` is the number of contributing channels, so the sum
//! of squared gains is unity. The low-frequency-effects (`LFE`) channel is
//! excluded from the fold because it is not a full-range program channel, and a
//! first-order-ambisonics layout folds to its omnidirectional `W` component.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the mono-downmix item of design section 23. The matrix is applied
//! by the output downmix stage of design section 48 along its normal smoothed
//! parameter path; this crate reuses [`AudioBuffer`] from `prism_audio_core`
//! rather than defining its own sample storage.

use alloc::vec::Vec;

use bevy_math::ops;
use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::math::Sample;

/// Returns `true` when the channel at `index` of `layout` is the `LFE` channel.
///
/// Only the `5.1` and `7.1` layouts carry a dedicated `LFE` channel, and in
/// both it occupies index `3` (after `FL`, `FR`, `C`).
#[must_use]
fn is_lfe_channel(layout: ChannelLayout, index: usize) -> bool {
    matches!(
        layout,
        ChannelLayout::Surround5_1 | ChannelLayout::Surround7_1
    ) && index == 3
}

/// Returns `true` when the channel at `index` contributes to the mono fold.
///
/// Every full-range channel contributes. The `LFE` channel is excluded. For a
/// first-order-ambisonics layout only the omnidirectional `W` component
/// (index `0`) contributes, since the directional components integrate to zero
/// pressure at the listening point.
#[must_use]
fn contributes(layout: ChannelLayout, index: usize) -> bool {
    match layout {
        ChannelLayout::AmbisonicFoa => index == 0,
        _ => !is_lfe_channel(layout, index),
    }
}

/// A per-source-channel fold-down matrix that collapses a layout to mono.
///
/// The matrix stores one linear coefficient per source channel; the mono
/// output of a frame is the dot product of these coefficients with the source
/// samples of that frame.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DownmixMatrix {
    /// Source channel layout the matrix was built for.
    source_layout: ChannelLayout,
    /// One linear coefficient per source channel.
    coefficients: Vec<Sample>,
}

impl DownmixMatrix {
    /// Builds the equal-power mono fold for `layout`.
    #[must_use]
    pub fn equal_power(layout: ChannelLayout) -> Self {
        let channels = layout.channel_count();
        let mut count = 0usize;
        for ch in 0..channels {
            if contributes(layout, ch) {
                count += 1;
            }
        }
        // `count` is always at least one: every layout has one contributing
        // channel (mono has its single channel, ambisonics has `W`).
        let gain = 1.0 / ops::sqrt(count as Sample);
        let mut coefficients = Vec::with_capacity(channels);
        for ch in 0..channels {
            if contributes(layout, ch) {
                coefficients.push(gain);
            } else {
                coefficients.push(0.0);
            }
        }
        Self {
            source_layout: layout,
            coefficients,
        }
    }

    /// Returns the source channel layout.
    #[inline]
    #[must_use]
    pub fn source_layout(&self) -> ChannelLayout {
        self.source_layout
    }

    /// Returns the per-source-channel coefficients.
    #[inline]
    #[must_use]
    pub fn coefficients(&self) -> &[Sample] {
        &self.coefficients
    }

    /// Folds a single frame given one sample per source channel.
    ///
    /// `samples` is read up to the matrix length; extra samples are ignored and
    /// missing channels are treated as silence.
    #[must_use]
    pub fn fold_frame(&self, samples: &[Sample]) -> Sample {
        let mut acc = 0.0;
        for (coeff, sample) in self.coefficients.iter().zip(samples.iter()) {
            acc += coeff * sample;
        }
        acc
    }

    /// Folds `source` into a freshly allocated mono [`AudioBuffer`].
    ///
    /// The result holds the same number of active frames as `source` and uses
    /// the same frame capacity.
    ///
    /// # Panics
    ///
    /// Panics if the layout of `source` differs from [`Self::source_layout`].
    #[must_use]
    pub fn fold_buffer(&self, source: &AudioBuffer) -> AudioBuffer {
        assert_eq!(
            source.layout(),
            self.source_layout,
            "layout mismatch on mono downmix"
        );
        let frames = source.capacity_frames();
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames);
        out.set_active_frames(source.active_frames());
        let active = source.active_frames();
        let channels = source.channels().min(self.coefficients.len());
        let dst = out.channel_mut(0);
        for (f, out_sample) in dst.iter_mut().enumerate().take(active) {
            let mut acc = 0.0;
            for (ch, coeff) in self.coefficients.iter().enumerate().take(channels) {
                acc += coeff * source.channel(ch)[f];
            }
            *out_sample = acc;
        }
        out
    }
}

/// One-touch mono downmix accommodation.
///
/// This is a control-rate toggle: when enabled, the output stage requests the
/// [`DownmixMatrix`] for the current layout and folds the mix to mono so a
/// listener using a single ear or a single speaker hears the full program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MonoDownmix {
    /// Whether the accommodation is active.
    enabled: bool,
}

impl MonoDownmix {
    /// Creates a downmix toggle in the given `enabled` state.
    #[inline]
    #[must_use]
    pub const fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Returns whether the accommodation is active.
    #[inline]
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Sets whether the accommodation is active.
    #[inline]
    pub const fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns the fold matrix for `layout` when enabled, otherwise `None`.
    #[must_use]
    pub fn matrix(&self, layout: ChannelLayout) -> Option<DownmixMatrix> {
        if self.enabled {
            Some(DownmixMatrix::equal_power(layout))
        } else {
            None
        }
    }
}

impl Default for MonoDownmix {
    fn default() -> Self {
        Self::new(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn stereo_is_equal_power() {
        let m = DownmixMatrix::equal_power(ChannelLayout::Stereo);
        let expected = 1.0 / ops::sqrt(2.0);
        for &c in m.coefficients() {
            assert!((c - expected).abs() < EPS);
        }
        let power: Sample = m.coefficients().iter().map(|c| c * c).sum();
        assert!((power - 1.0).abs() < EPS);
    }

    #[test]
    fn lfe_is_excluded_and_power_preserved() {
        let m = DownmixMatrix::equal_power(ChannelLayout::Surround5_1);
        // Index 3 is LFE and must be silent.
        assert!(m.coefficients()[3].abs() < EPS);
        let power: Sample = m.coefficients().iter().map(|c| c * c).sum();
        assert!((power - 1.0).abs() < EPS);
    }

    #[test]
    fn ambisonics_folds_to_w_only() {
        let m = DownmixMatrix::equal_power(ChannelLayout::AmbisonicFoa);
        assert!((m.coefficients()[0] - 1.0).abs() < EPS);
        for &c in &m.coefficients()[1..] {
            assert!(c.abs() < EPS);
        }
    }

    #[test]
    fn fold_frame_matches_dot_product() {
        let m = DownmixMatrix::equal_power(ChannelLayout::Stereo);
        let out = m.fold_frame(&[1.0, 1.0]);
        let expected = 2.0 / ops::sqrt(2.0);
        assert!((out - expected).abs() < EPS);
    }

    #[test]
    fn fold_buffer_sums_channels() {
        let mut src = AudioBuffer::new(ChannelLayout::Stereo, 4);
        for f in 0..4 {
            src.channel_mut(0)[f] = 0.5;
            src.channel_mut(1)[f] = 0.5;
        }
        let m = DownmixMatrix::equal_power(ChannelLayout::Stereo);
        let mono = m.fold_buffer(&src);
        assert_eq!(mono.layout(), ChannelLayout::Mono);
        let expected = 1.0 / ops::sqrt(2.0);
        for f in 0..4 {
            assert!((mono.channel(0)[f] - expected).abs() < EPS);
        }
    }

    #[test]
    fn toggle_gates_matrix() {
        let off = MonoDownmix::default();
        assert!(!off.is_enabled());
        assert!(off.matrix(ChannelLayout::Stereo).is_none());
        let mut on = MonoDownmix::new(true);
        assert!(on.matrix(ChannelLayout::Stereo).is_some());
        on.set_enabled(false);
        assert!(on.matrix(ChannelLayout::Stereo).is_none());
    }
}
