//! Bass management: redirect the low frequencies of the main channels into the
//! LFE/subwoofer channel through a phase-matched crossover.
//!
//! A small-speaker playback setup cannot reproduce the deep bass carried by the
//! full-range main channels, so the delivery chain **splits** each main channel
//! at a configurable crossover (default `80 Hz`), keeps the high band on the
//! main channel, and **sums the extracted low band into the LFE channel** with
//! a cinema calibration gain (the `+10 dB` LFE convention). A bypass switch
//! turns the whole stage into a transparent pass-through.
//!
//! # Provenance
//!
//! The crossover is a fourth-order **Linkwitz-Riley** split (two cascaded
//! Butterworth sections per branch), standard public loudspeaker-crossover
//! theory, reused from this crate's dependency
//! [`LinkwitzRileyCrossover`](prism_audio_core::nodes::crossover::LinkwitzRileyCrossover)
//! rather than reimplemented. The `+10 dB` LFE calibration is the published
//! cinema (SMPTE/ITU) in-band gain convention. This file contains **no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio
//! source or derived code**; it is pure classic DSP with no AI/ML.
//!
//! # Relationship
//!
//! This stage sits between the format-conversion
//! [`DownmixMatrix`](crate::downmix::DownmixMatrix) and the delivery
//! [`OutputProfile`](crate::profiles::OutputProfile) in the output render
//! chain. It does not invent new DSP: the band split is delegated to the core
//! Linkwitz-Riley crossover and the sum/calibration reuses the core
//! [`db_to_linear`](prism_audio_core::math::db_to_linear) helper.
//!
//! # Determinism
//!
//! All filter state and the two scratch band buffers are allocated in
//! [`BassManager::new`]. [`BassManager::process`] performs no allocation, takes
//! no locks, and cannot panic; layouts without a dedicated LFE channel are a
//! graceful no-op.

use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::math::{db_to_linear, Sample};
use prism_audio_core::nodes::crossover::LinkwitzRileyCrossover;

/// Default bass-management crossover frequency, in Hz.
pub const DEFAULT_CROSSOVER_HZ: Sample = 80.0;

/// Default LFE calibration gain applied to the redirected bass, in decibels
/// (the cinema `+10 dB` in-band convention).
pub const DEFAULT_LFE_GAIN_DB: Sample = 10.0;

/// Configuration for a [`BassManager`].
///
/// Plain [`Copy`] description data; changing the crossover frequency requires
/// rebuilding the manager (the filter coefficients are budgeted at
/// construction), while the gain and bypass flag are live-tweakable.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BassManagerParams {
    /// Crossover frequency separating redirected bass from the mains, in Hz.
    pub crossover_hz: Sample,
    /// Linear-domain calibration gain for the summed bass, expressed in dB.
    pub lfe_gain_db: Sample,
    /// When `true` the stage is a transparent pass-through.
    pub bypassed: bool,
}

impl Default for BassManagerParams {
    #[inline]
    fn default() -> Self {
        Self {
            crossover_hz: DEFAULT_CROSSOVER_HZ,
            lfe_gain_db: DEFAULT_LFE_GAIN_DB,
            bypassed: false,
        }
    }
}

/// Returns the LFE channel index for layouts that carry one.
///
/// `5.1` and `7.1` place the LFE at index `3` (`FL, FR, C, LFE, ...`); other
/// layouts have no dedicated LFE channel.
#[inline]
#[must_use]
fn lfe_index(layout: ChannelLayout) -> Option<usize> {
    match layout {
        ChannelLayout::Surround5_1 | ChannelLayout::Surround7_1 => Some(3),
        _ => None,
    }
}

/// A Linkwitz-Riley bass-management / LFE crossover.
///
/// Build one with [`BassManager::new`] for a layout and block size, then call
/// [`BassManager::process`] on each block. The main channels are high-passed in
/// place and their extracted bass is summed, with the calibration gain, into
/// the LFE channel.
#[derive(Debug, Clone)]
pub struct BassManager {
    layout: ChannelLayout,
    lfe: Option<usize>,
    crossover: LinkwitzRileyCrossover,
    /// Scratch band buffers: `bands[0]` is the low band, `bands[1]` the high.
    bands: [AudioBuffer; 2],
    lfe_gain_db: Sample,
    bypassed: bool,
}

impl BassManager {
    /// Builds a bass manager for `layout`, handling blocks up to `max_frames`.
    ///
    /// The crossover filters are designed once here from
    /// [`BassManagerParams::crossover_hz`]; the two scratch band buffers are
    /// allocated to the full layout width so the subsequent per-block
    /// processing never allocates.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        layout: ChannelLayout,
        max_frames: usize,
        params: &BassManagerParams,
    ) -> Self {
        let crossover =
            LinkwitzRileyCrossover::new(sample_rate, layout, &[params.crossover_hz], max_frames);
        let frames = max_frames.max(1);
        Self {
            layout,
            lfe: lfe_index(layout),
            crossover,
            bands: [
                AudioBuffer::new(layout, frames),
                AudioBuffer::new(layout, frames),
            ],
            lfe_gain_db: params.lfe_gain_db,
            bypassed: params.bypassed,
        }
    }

    /// The channel layout this manager was built for.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Whether the stage is currently bypassed (a transparent pass-through).
    #[inline]
    #[must_use]
    pub fn is_bypassed(&self) -> bool {
        self.bypassed
    }

    /// Enables or disables the transparent bypass.
    #[inline]
    pub fn set_bypassed(&mut self, bypassed: bool) {
        self.bypassed = bypassed;
    }

    /// Sets the LFE calibration gain (in dB) applied to the redirected bass.
    #[inline]
    pub fn set_lfe_gain_db(&mut self, gain_db: Sample) {
        self.lfe_gain_db = gain_db;
    }

    /// Returns the current LFE calibration gain, in dB.
    #[inline]
    #[must_use]
    pub fn lfe_gain_db(&self) -> Sample {
        self.lfe_gain_db
    }

    /// Returns `true` if the configured layout carries a dedicated LFE channel.
    #[inline]
    #[must_use]
    pub fn has_lfe(&self) -> bool {
        self.lfe.is_some()
    }

    /// Clears the internal crossover filter state and scratch bands.
    pub fn reset(&mut self) {
        self.crossover.reset();
        self.bands[0].clear();
        self.bands[1].clear();
    }

    /// Applies bass management to `buf` in place.
    ///
    /// With an LFE-bearing layout and the stage active, every channel is split
    /// into a low and high band; the main (non-LFE) channels are replaced by
    /// their high band, and the sum of their low bands (scaled by the
    /// calibration gain) is added to the original LFE channel. When the stage
    /// is bypassed, or the layout has no LFE channel, `buf` is left untouched.
    ///
    /// Allocation-free, lock-free, and panic-free.
    pub fn process(&mut self, buf: &mut AudioBuffer) {
        if self.bypassed {
            return;
        }
        let Some(lfe) = self.lfe else {
            return;
        };
        if buf.layout() != self.layout {
            return;
        }

        // Split every channel into [low, high] bands. `process_block` reads
        // `buf` and writes the scratch band buffers, leaving `buf` unchanged.
        self.crossover.process_block(buf, &mut self.bands);

        let channels = buf.channels();
        let gain = db_to_linear(self.lfe_gain_db);

        // Sum the redirected bass of every main channel into the LFE channel.
        for ch in 0..channels {
            if ch == lfe {
                continue;
            }
            let low = self.bands[0].channel(ch);
            let dst = buf.channel_mut(lfe);
            let n = dst.len().min(low.len());
            for (d, &b) in dst[..n].iter_mut().zip(&low[..n]) {
                *d += b * gain;
            }
        }

        // Replace each main channel with its high-passed band.
        for ch in 0..channels {
            if ch == lfe {
                continue;
            }
            let high = self.bands[1].channel(ch);
            let dst = buf.channel_mut(ch);
            let n = dst.len().min(high.len());
            dst[..n].copy_from_slice(&high[..n]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use bevy_math::ops;

    const SR: u32 = 48_000;

    fn fill_sine(buf: &mut AudioBuffer, channel: usize, freq: Sample) {
        let step = 2.0 * core::f32::consts::PI * freq / SR as Sample;
        let data = buf.channel_mut(channel);
        let mut phase = 0.0;
        for d in data.iter_mut() {
            *d = ops::sin(phase);
            phase += step;
        }
    }

    fn rms(data: &[Sample]) -> Sample {
        if data.is_empty() {
            return 0.0;
        }
        let acc: Sample = data.iter().map(|s| s * s).sum();
        ops::sqrt(acc / data.len() as Sample)
    }

    #[test]
    fn bypass_is_identity() {
        let frames = 512;
        let params = BassManagerParams {
            bypassed: true,
            ..Default::default()
        };
        let mut bm = BassManager::new(SR, ChannelLayout::Surround5_1, frames, &params);
        let mut buf = AudioBuffer::new(ChannelLayout::Surround5_1, frames);
        fill_sine(&mut buf, 0, 40.0);
        let before: Vec<Sample> = buf.channel(0).to_vec();
        bm.process(&mut buf);
        assert_eq!(buf.channel(0), before.as_slice());
        // LFE untouched (silent).
        assert!(buf.channel(3).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn low_frequency_bass_is_redirected_to_lfe() {
        let frames = 4096;
        let params = BassManagerParams::default();
        let mut bm = BassManager::new(SR, ChannelLayout::Surround5_1, frames, &params);
        let mut buf = AudioBuffer::new(ChannelLayout::Surround5_1, frames);
        // 30 Hz tone, well below the 80 Hz crossover, into the front-left main.
        fill_sine(&mut buf, 0, 30.0);
        let main_in = rms(buf.channel(0));
        assert!(buf.channel(3).iter().all(|&s| s == 0.0), "LFE starts silent");

        bm.process(&mut buf);

        // Skip the filter settling transient when measuring the tail.
        let skip = 1024;
        let main_out = rms(&buf.channel(0)[skip..]);
        let lfe_out = rms(&buf.channel(3)[skip..]);

        // The main channel's deep bass is strongly attenuated by the high-pass.
        assert!(
            main_out < main_in * 0.25,
            "main not high-passed: in={main_in} out={main_out}"
        );
        // The LFE now carries substantial redirected (and +10 dB calibrated)
        // bass energy.
        assert!(lfe_out > main_in, "bass not redirected: lfe={lfe_out}");
    }

    #[test]
    fn high_frequency_stays_on_mains() {
        let frames = 4096;
        let mut bm =
            BassManager::new(SR, ChannelLayout::Surround5_1, frames, &BassManagerParams::default());
        let mut buf = AudioBuffer::new(ChannelLayout::Surround5_1, frames);
        // 1 kHz tone, far above the crossover, into the front-right main.
        fill_sine(&mut buf, 1, 1_000.0);
        let main_in = rms(buf.channel(1));

        bm.process(&mut buf);

        let skip = 1024;
        let main_out = rms(&buf.channel(1)[skip..]);
        let lfe_out = rms(&buf.channel(3)[skip..]);
        // The high-passed main is essentially unchanged.
        assert!(
            (main_out - main_in).abs() < main_in * 0.1,
            "high content altered: in={main_in} out={main_out}"
        );
        // Very little leaks into the LFE.
        assert!(lfe_out < main_in * 0.1, "high leaked to lfe: {lfe_out}");
    }

    #[test]
    fn layout_without_lfe_is_noop() {
        let frames = 256;
        let mut bm =
            BassManager::new(SR, ChannelLayout::Stereo, frames, &BassManagerParams::default());
        assert!(!bm.has_lfe());
        let mut buf = AudioBuffer::new(ChannelLayout::Stereo, frames);
        fill_sine(&mut buf, 0, 50.0);
        let before: Vec<Sample> = buf.channel(0).to_vec();
        bm.process(&mut buf);
        assert_eq!(buf.channel(0), before.as_slice());
    }

    #[test]
    fn setters_update_state() {
        let mut bm = BassManager::new(
            SR,
            ChannelLayout::Surround5_1,
            128,
            &BassManagerParams::default(),
        );
        assert!(!bm.is_bypassed());
        bm.set_bypassed(true);
        assert!(bm.is_bypassed());
        bm.set_lfe_gain_db(6.0);
        assert!((bm.lfe_gain_db() - 6.0).abs() < 1.0e-6);
    }
}
