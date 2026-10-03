//! Bus metering snapshots: peak, RMS, LUFS, true-peak, and correlation.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the metering part of design section 26 (`MeterNode`-style
//! read-back). [`MeterProbe`] reuses the `prism_audio_core` loudness and
//! correlation primitives rather than re-implementing them: it drives a
//! [`LoudnessMeter`](prism_audio_core::nodes::analysis::loudness::LoudnessMeter)
//! and, for stereo buffers, a
//! [`CorrelationMeter`](prism_audio_core::nodes::analysis::correlation::CorrelationMeter),
//! then reduces per-block peak/RMS alongside their ballistic readings into a
//! plain [`MeterSnapshot`].

use alloc::vec;
use alloc::vec::Vec;

use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::math::{linear_to_db, Sample};
use prism_audio_core::nodes::analysis::correlation::CorrelationMeter;
use prism_audio_core::nodes::analysis::loudness::LoudnessMeter;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Default correlation integration window, in milliseconds.
pub const DEFAULT_CORRELATION_MS: Sample = 300.0;

/// A compact, read-only reduction of a bus's metering state.
///
/// Per-channel vectors are indexed by channel. Loudness fields follow the
/// `ITU-R BS.1770` convention where under-determined values report
/// [`Sample::NEG_INFINITY`]. The correlation fields are populated only for
/// stereo buffers.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MeterSnapshot {
    /// Number of channels measured.
    pub channels: usize,
    /// Per-channel peak absolute magnitude for the most recent block.
    pub peak: Vec<Sample>,
    /// Per-channel RMS for the most recent block.
    pub rms: Vec<Sample>,
    /// Peak absolute magnitude across all channels for the most recent block.
    pub master_peak: Sample,
    /// RMS across all channels for the most recent block.
    pub master_rms: Sample,
    /// Sliding 400 ms loudness in `LUFS`.
    pub momentary_lufs: Sample,
    /// Sliding 3 s loudness in `LUFS`.
    pub short_term_lufs: Sample,
    /// Gated programme loudness in `LUFS`.
    pub integrated_lufs: Sample,
    /// Loudness range in `LU`.
    pub loudness_range_lu: Sample,
    /// Maximum true-peak level in `dBTP`.
    pub true_peak_dbtp: Sample,
    /// Stereo correlation coefficient in `[-1, 1]`, or `None` when not stereo.
    pub correlation: Option<Sample>,
}

impl MeterSnapshot {
    /// Master peak level expressed in dBFS.
    #[must_use]
    #[inline]
    pub fn master_peak_db(&self) -> Sample {
        linear_to_db(self.master_peak)
    }

    /// Master RMS level expressed in dBFS.
    #[must_use]
    #[inline]
    pub fn master_rms_db(&self) -> Sample {
        linear_to_db(self.master_rms)
    }
}

/// A stateful probe that accumulates metering over successive blocks.
///
/// Call [`MeterProbe::process`] once per rendered block with that block's bus
/// buffer, then read [`MeterProbe::snapshot`] for a reduction. The loudness and
/// correlation ballistics integrate across blocks (matching broadcast meters),
/// while peak/RMS reflect the most recently processed block.
#[derive(Debug, Clone)]
pub struct MeterProbe {
    channels: usize,
    loudness: LoudnessMeter,
    correlation: Option<CorrelationMeter>,
    last_peak: Vec<Sample>,
    last_rms: Vec<Sample>,
    last_master_peak: Sample,
    last_master_rms: Sample,
}

impl MeterProbe {
    /// Build a probe for `layout` at `sample_rate` using the default
    /// correlation integration window.
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout) -> Self {
        Self::with_correlation_ms(sample_rate, layout, DEFAULT_CORRELATION_MS)
    }

    /// Build a probe with an explicit correlation integration window (used only
    /// when `layout` is stereo).
    #[must_use]
    pub fn with_correlation_ms(
        sample_rate: u32,
        layout: ChannelLayout,
        correlation_ms: Sample,
    ) -> Self {
        let channels = layout.channel_count();
        let correlation = if layout == ChannelLayout::Stereo {
            Some(CorrelationMeter::new(sample_rate, correlation_ms))
        } else {
            None
        };
        Self {
            channels,
            loudness: LoudnessMeter::new(sample_rate, layout),
            correlation,
            last_peak: vec![0.0; channels],
            last_rms: vec![0.0; channels],
            last_master_peak: 0.0,
            last_master_rms: 0.0,
        }
    }

    /// Number of channels this probe measures.
    #[must_use]
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Process one block of audio, updating ballistics and per-block peak/RMS.
    ///
    /// Channels in `buffer` beyond this probe's channel count are ignored, and
    /// missing channels are treated as silent. Only the buffer's active frame
    /// range is read.
    pub fn process(&mut self, buffer: &AudioBuffer) {
        let frames = buffer.active_frames();
        let buffer_channels = buffer.channels();

        let mut peak = vec![0.0 as Sample; self.channels];
        let mut sum_sq = vec![0.0 as Sample; self.channels];
        let mut master_peak = 0.0 as Sample;
        let mut master_sum_sq = 0.0 as Sample;

        for frame in 0..frames {
            for (ch, (peak_ch, sum_sq_ch)) in peak.iter_mut().zip(sum_sq.iter_mut()).enumerate() {
                let x = if ch < buffer_channels {
                    buffer.channel(ch)[frame]
                } else {
                    0.0
                };
                let mag = x.abs();
                if mag > *peak_ch {
                    *peak_ch = mag;
                }
                if mag > master_peak {
                    master_peak = mag;
                }
                *sum_sq_ch += x * x;
                master_sum_sq += x * x;
                self.loudness.feed_sample(ch, x);
            }
            self.loudness.advance_frame();

            if let Some(correlation) = self.correlation.as_mut() {
                let left = if buffer_channels > 0 {
                    buffer.channel(0)[frame]
                } else {
                    0.0
                };
                let right = if buffer_channels > 1 {
                    buffer.channel(1)[frame]
                } else {
                    0.0
                };
                correlation.process_sample(left, right);
            }
        }

        if frames == 0 {
            for value in self.last_peak.iter_mut() {
                *value = 0.0;
            }
            for value in self.last_rms.iter_mut() {
                *value = 0.0;
            }
            self.last_master_peak = 0.0;
            self.last_master_rms = 0.0;
            return;
        }

        let inv_frames = 1.0 / frames as Sample;
        for (rms_ch, sum_sq_ch) in self.last_rms.iter_mut().zip(sum_sq.iter()) {
            *rms_ch = sqrt(*sum_sq_ch * inv_frames);
        }
        self.last_peak = peak;
        self.last_master_peak = master_peak;
        let channel_divisor = (self.channels.max(1)) as Sample;
        self.last_master_rms = sqrt(master_sum_sq * inv_frames / channel_divisor);
    }

    /// Produce a read-only snapshot of the current metering state.
    #[must_use]
    pub fn snapshot(&self) -> MeterSnapshot {
        let measurement = self.loudness.measurement();
        MeterSnapshot {
            channels: self.channels,
            peak: self.last_peak.clone(),
            rms: self.last_rms.clone(),
            master_peak: self.last_master_peak,
            master_rms: self.last_master_rms,
            momentary_lufs: measurement.momentary_lufs,
            short_term_lufs: measurement.short_term_lufs,
            integrated_lufs: measurement.integrated_lufs,
            loudness_range_lu: measurement.loudness_range_lu,
            true_peak_dbtp: measurement.true_peak_dbtp,
            correlation: self.correlation.as_ref().map(CorrelationMeter::correlation),
        }
    }

    /// Reset every meter and clear the stored per-block peak/RMS.
    pub fn reset(&mut self) {
        self.loudness.reset();
        if let Some(correlation) = self.correlation.as_mut() {
            correlation.reset();
        }
        for value in self.last_peak.iter_mut() {
            *value = 0.0;
        }
        for value in self.last_rms.iter_mut() {
            *value = 0.0;
        }
        self.last_master_peak = 0.0;
        self.last_master_rms = 0.0;
    }
}

/// Deterministic square root over the audio sample type.
///
/// Routed through `bevy_math::ops` so the profiler shares the engine's
/// determinism guarantees and stays `no_std`-clean.
#[inline]
fn sqrt(value: Sample) -> Sample {
    bevy_math::ops::sqrt(value)
}
