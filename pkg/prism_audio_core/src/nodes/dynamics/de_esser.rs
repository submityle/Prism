//! De-esser: a frequency-selective compressor that tames vocal sibilance
//! ("ess", "sh", "ch" sounds) without dulling the rest of the signal.
//!
//! Sibilance is a burst of high-frequency energy (roughly 5-9 kHz on most
//! voices). A broadband compressor triggered by it would duck the whole vocal
//! and pump; a de-esser instead splits the signal at a crossover, watches only
//! the high band for excess energy, and attenuates the sibilance while the body
//! of the voice passes untouched.
//!
//! # The model (split-band dynamics)
//!
//! A two-way Linkwitz-Riley crossover splits the input into a low band and a
//! high (sibilance) band whose magnitudes sum flat. The stereo-linked loudest
//! high-band sample drives a level detector; its level feeds the standard
//! soft-knee compression curve
//! [`compressor_reduction_db`](super::detector::compressor_reduction_db),
//! producing a reduction in decibels that is capped at `max_reduction_db` and
//! smoothed by an attack/release [`GainBallistics`](super::detector::GainBallistics).
//! The resulting linear gain is applied as follows:
//!
//! - [`DeEsserMode::SplitBand`] scales only the high band and recombines it with
//!   the untouched low band, so the vocal body is preserved.
//! - [`DeEsserMode::Wideband`] scales the whole signal, the classic broadband
//!   de-ess used when a single gain move is preferred.
//!
//! # Real-time contract
//!
//! The crossover, band scratch, detector, and ballistics are all allocated in
//! [`DeEsserNode::new`]. [`process`](crate::graph::AudioNode::process) performs
//! no allocation, takes no locks, and cannot panic: mismatched channel counts
//! and zero-length blocks degrade gracefully.
//!
//! # Provenance
//!
//! The split-band de-esser (crossover plus a compressor keyed off the high
//! band) is a classic studio topology documented across the audio-effects
//! literature (e.g. Reiss and `McPherson`, "Audio Effects", 2014; Zoelzer,
//! "DAFX"). It reuses this crate's own Linkwitz-Riley crossover and dynamics
//! detector primitives. This module contains **no Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**; it is implemented purely from that publicly documented theory.

use alloc::vec::Vec;

use crate::buffer::{AudioBuffer, ChannelLayout};
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear};
use crate::nodes::crossover::LinkwitzRileyCrossover;
use crate::nodes::dynamics::detector::{
    DetectionMode, GainBallistics, LevelDetector, compressor_reduction_db,
};

/// How the computed sibilance gain reduction is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DeEsserMode {
    /// Attenuate only the high band, recombining it with the untouched low
    /// band. Preserves the body of the voice (the default, most transparent
    /// mode).
    #[default]
    SplitBand,
    /// Attenuate the whole signal with a single broadband gain move.
    Wideband,
}

/// Configuration for a [`DeEsserNode`].
///
/// The defaults target a typical vocal: a 6 kHz split, a moderate threshold and
/// ratio, fast attack, and RMS detection over a short window.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DeEsserParams {
    /// Crossover frequency (Hz) separating the low band from the sibilance
    /// band.
    pub crossover_hz: Sample,
    /// Detection threshold (dBFS) above which sibilance is attenuated.
    pub threshold_db: Sample,
    /// Compression ratio (`>= 1`).
    pub ratio: Sample,
    /// Soft-knee width (dB).
    pub knee_db: Sample,
    /// Gain-reduction attack time (ms).
    pub attack_ms: Sample,
    /// Gain-reduction release time (ms).
    pub release_ms: Sample,
    /// Maximum attenuation (dB) the de-esser is allowed to apply.
    pub max_reduction_db: Sample,
    /// Side-chain detection mode (peak or RMS).
    pub detection: DetectionMode,
    /// RMS window length (ms), used when `detection` is
    /// [`DetectionMode::Rms`].
    pub rms_window_ms: Sample,
    /// Whether the gain is applied to the high band only or the whole signal.
    pub mode: DeEsserMode,
}

impl Default for DeEsserParams {
    fn default() -> Self {
        Self {
            crossover_hz: 6000.0,
            threshold_db: -30.0,
            ratio: 4.0,
            knee_db: 6.0,
            attack_ms: 1.0,
            release_ms: 60.0,
            max_reduction_db: 12.0,
            detection: DetectionMode::Rms,
            rms_window_ms: 5.0,
            mode: DeEsserMode::SplitBand,
        }
    }
}

/// A split-band de-esser.
///
/// Stereo-linked: one gain is computed per frame from the loudest high-band
/// channel and applied to every channel, preserving the stereo image.
#[derive(Debug)]
pub struct DeEsserNode {
    /// Two-way Linkwitz-Riley crossover (low band + sibilance band).
    crossover: LinkwitzRileyCrossover,
    /// Band scratch buffers filled by the crossover: `[low, high]`.
    bands: Vec<AudioBuffer>,
    /// Side-chain level detector on the high band.
    detector: LevelDetector,
    /// Attack/release smoothing of the gain reduction.
    ballistics: GainBallistics,
    /// Detection threshold (dBFS).
    threshold_db: Sample,
    /// Compression ratio (`>= 1`).
    ratio: Sample,
    /// Soft-knee width (dB).
    knee_db: Sample,
    /// Maximum attenuation (dB).
    max_reduction_db: Sample,
    /// How the gain is applied.
    mode: DeEsserMode,
    /// Most recent smoothed reduction (dB), for metering.
    last_reduction_db: Sample,
}

impl DeEsserNode {
    /// Builds a de-esser for `layout` at `sample_rate`, sized for blocks of up
    /// to `max_frames`.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        layout: ChannelLayout,
        params: DeEsserParams,
        max_frames: usize,
    ) -> Self {
        let crossover =
            LinkwitzRileyCrossover::new(sample_rate, layout, &[params.crossover_hz], max_frames);
        let capacity = max_frames.max(1);
        let bands: Vec<AudioBuffer> = alloc::vec![AudioBuffer::new(layout, capacity); 2];
        Self {
            crossover,
            bands,
            detector: LevelDetector::new(params.detection, params.rms_window_ms, sample_rate),
            ballistics: GainBallistics::new(params.attack_ms, params.release_ms, sample_rate),
            threshold_db: params.threshold_db,
            ratio: params.ratio.max(1.0),
            knee_db: params.knee_db.max(0.0),
            max_reduction_db: params.max_reduction_db.max(0.0),
            mode: params.mode,
            last_reduction_db: 0.0,
        }
    }

    /// Sets the detection threshold (dBFS).
    pub fn set_threshold_db(&mut self, threshold_db: Sample) {
        self.threshold_db = threshold_db;
    }

    /// Sets the compression ratio (clamped to `>= 1`).
    pub fn set_ratio(&mut self, ratio: Sample) {
        self.ratio = ratio.max(1.0);
    }

    /// Sets the maximum attenuation (dB, clamped to `>= 0`).
    pub fn set_max_reduction_db(&mut self, max_reduction_db: Sample) {
        self.max_reduction_db = max_reduction_db.max(0.0);
    }

    /// Selects how the gain reduction is applied.
    pub fn set_mode(&mut self, mode: DeEsserMode) {
        self.mode = mode;
    }

    /// Returns the most recent smoothed gain reduction in decibels
    /// (non-negative; larger means more sibilance was attenuated).
    #[must_use]
    pub fn reduction_db(&self) -> Sample {
        self.last_reduction_db
    }
}

impl AudioNode for DeEsserNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels());
        let frames = output.active_frames();
        if frames == 0 || channels == 0 {
            return;
        }

        // Split the input into [low, high]; magnitudes sum flat.
        self.crossover.process_block(input, &mut self.bands);

        for f in 0..frames {
            // Stereo-linked side-chain: loudest high-band sample this frame.
            let mut side = 0.0;
            for ch in 0..channels {
                let a = self.bands[1].channel(ch)[f].abs();
                if a > side {
                    side = a;
                }
            }

            let level_db = self.detector.level_db(side);
            let raw_reduction =
                compressor_reduction_db(level_db, self.threshold_db, self.ratio, self.knee_db)
                    .min(self.max_reduction_db);
            let reduction = self.ballistics.process(raw_reduction);
            self.last_reduction_db = reduction;
            let gain = db_to_linear(-reduction);

            match self.mode {
                DeEsserMode::SplitBand => {
                    for ch in 0..channels {
                        let lo = self.bands[0].channel(ch)[f];
                        let hi = self.bands[1].channel(ch)[f];
                        output.channel_mut(ch)[f] = lo + gain * hi;
                    }
                }
                DeEsserMode::Wideband => {
                    for ch in 0..channels {
                        let x = input.channel(ch)[f];
                        output.channel_mut(ch)[f] = gain * x;
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
        self.crossover.reset();
        self.detector.reset();
        self.ballistics.reset();
        self.last_reduction_db = 0.0;
        for band in &mut self.bands {
            band.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    // A mono tone at `freq_hz` with unit amplitude.
    fn tone(frames: usize, freq_hz: Sample) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, frames);
        buf.set_active_frames(frames);
        let w = core::f32::consts::TAU * freq_hz / SR as Sample;
        for (i, s) in buf.channel_mut(0).iter_mut().enumerate() {
            *s = ops::sin(w * i as Sample);
        }
        buf
    }

    fn run(node: &mut DeEsserNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let mut out = AudioBuffer::new(input.layout(), input.capacity_frames());
        out.set_active_frames(frames);
        let inputs = [input.clone()];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [o] = outputs;
        o
    }

    fn rms(buf: &AudioBuffer) -> Sample {
        let mut e = 0.0;
        let ch = buf.channel(0);
        for &s in ch {
            e += s * s;
        }
        ops::sqrt(e / ch.len().max(1) as Sample)
    }

    #[test]
    fn low_frequency_tone_passes_through_unaffected() {
        // A 200 Hz tone lives in the low band and must not be attenuated.
        let input = tone(4096, 200.0);
        let params = DeEsserParams::default();
        let mut node = DeEsserNode::new(SR, ChannelLayout::Mono, params, 4096);
        let out = run(&mut node, &input);
        // Skip the crossover settling transient; compare the steady tail.
        let tail_in: Sample = input.channel(0)[2048..].iter().map(|s| s * s).sum();
        let tail_out: Sample = out.channel(0)[2048..].iter().map(|s| s * s).sum();
        assert!(tail_out > tail_in * 0.9, "low band lost energy: {tail_out} vs {tail_in}");
        assert!(node.reduction_db() < 1.0, "unexpected reduction {}", node.reduction_db());
    }

    #[test]
    fn loud_sibilance_is_attenuated() {
        // A loud 8 kHz tone sits in the sibilance band and must be ducked.
        let input = tone(8192, 8000.0);
        let params = DeEsserParams {
            threshold_db: -30.0,
            ratio: 6.0,
            ..DeEsserParams::default()
        };
        let mut node = DeEsserNode::new(SR, ChannelLayout::Mono, params, 8192);
        let out = run(&mut node, &input);
        assert!(rms(&out) < rms(&input), "sibilance not reduced");
        assert!(node.reduction_db() > 1.0, "expected reduction, got {}", node.reduction_db());
    }

    #[test]
    fn quiet_sibilance_below_threshold_is_left_alone() {
        // The same 8 kHz tone at a low level stays below the threshold.
        let mut input = tone(8192, 8000.0);
        for s in input.channel_mut(0) {
            *s *= 0.02;
        }
        let params = DeEsserParams {
            threshold_db: -20.0,
            ..DeEsserParams::default()
        };
        let mut node = DeEsserNode::new(SR, ChannelLayout::Mono, params, 8192);
        let _ = run(&mut node, &input);
        assert!(node.reduction_db() < 1.0, "quiet ess should not duck: {}", node.reduction_db());
    }

    #[test]
    fn wideband_mode_reduces_full_signal() {
        let input = tone(8192, 8000.0);
        let params = DeEsserParams {
            mode: DeEsserMode::Wideband,
            ratio: 6.0,
            ..DeEsserParams::default()
        };
        let mut node = DeEsserNode::new(SR, ChannelLayout::Mono, params, 8192);
        let out = run(&mut node, &input);
        assert!(rms(&out) < rms(&input), "wideband did not reduce");
        assert!(node.reduction_db() > 1.0);
    }

    #[test]
    fn reduction_is_capped_at_max_reduction_db() {
        // Very loud sibilance, tiny ceiling: reduction must not exceed the cap.
        let input = tone(8192, 8000.0);
        let params = DeEsserParams {
            threshold_db: -60.0,
            ratio: 20.0,
            max_reduction_db: 3.0,
            ..DeEsserParams::default()
        };
        let mut node = DeEsserNode::new(SR, ChannelLayout::Mono, params, 8192);
        let _ = run(&mut node, &input);
        assert!(node.reduction_db() <= 3.0 + 1e-3, "cap exceeded: {}", node.reduction_db());
    }

    #[test]
    fn output_is_finite_and_reset_clears_state() {
        let input = tone(4096, 8000.0);
        let mut node = DeEsserNode::new(SR, ChannelLayout::Mono, DeEsserParams::default(), 4096);
        let out = run(&mut node, &input);
        for s in out.channel(0) {
            assert!(s.is_finite());
        }
        node.reset();
        assert!(node.reduction_db().abs() < 1e-9);
    }

    #[test]
    fn silent_input_is_safe_and_silent() {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, 512);
        buf.set_active_frames(512);
        let mut node = DeEsserNode::new(SR, ChannelLayout::Mono, DeEsserParams::default(), 512);
        let out = run(&mut node, &buf);
        for s in out.channel(0) {
            assert!(s.abs() < 1e-6);
        }
        assert!(node.reduction_db() < 1e-3);
    }

    #[test]
    fn zero_frame_block_does_not_panic() {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, 64);
        buf.set_active_frames(0);
        let mut node = DeEsserNode::new(SR, ChannelLayout::Mono, DeEsserParams::default(), 64);
        let _ = run(&mut node, &buf);
    }

    #[test]
    fn stereo_gain_is_linked_across_channels() {
        let frames = 8192;
        let mut buf = AudioBuffer::new(ChannelLayout::Stereo, frames);
        buf.set_active_frames(frames);
        let sib = tone(frames, 8000.0);
        for i in 0..frames {
            let v = sib.channel(0)[i];
            buf.channel_mut(0)[i] = v;
            buf.channel_mut(1)[i] = 0.5 * v;
        }
        let params = DeEsserParams {
            ratio: 6.0,
            ..DeEsserParams::default()
        };
        let mut node = DeEsserNode::new(SR, ChannelLayout::Stereo, params, frames);
        let out = run(&mut node, &buf);
        // A linked de-esser applies the same gain to both channels, so the
        // right/left ratio is preserved wherever the left channel is non-zero.
        for i in (frames / 2)..frames {
            let l = out.channel(0)[i];
            let r = out.channel(1)[i];
            if l.abs() > 1e-3 {
                assert!((r - 0.5 * l).abs() < 1e-3, "l={l} r={r}");
            }
        }
    }
}
