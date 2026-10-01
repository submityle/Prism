//! Dynamic EQ: a single parametric band whose boost or cut is driven by the
//! signal's own level inside that band, rather than being static.
//!
//! A static parametric bell (see [`parametric_eq`](crate::nodes::effects::parametric_eq))
//! applies the same gain forever. A dynamic EQ instead *measures* how much
//! energy sits inside the band and fades the bell in only when the band crosses
//! a threshold. It is the surgical cousin of a compressor: a compressor rides
//! one broadband gain, a multiband compressor splits the spectrum into fixed
//! crossover bands, while a dynamic EQ reshapes a single resonant region with a
//! smooth minimum-phase bell that is only as active as the material demands.
//!
//! # The model (fixed filter, cross-faded by depth)
//!
//! Running a biquad whose coefficients are recomputed every sample from the
//! control signal is both expensive and prone to zipper noise and transient
//! instability. Instead this node keeps **one fixed peaking bell designed at
//! the full `range_db`** and cross-fades its contribution:
//!
//! ```text
//! shaped[n] = peaking(x[n])           // bell at full range_db
//! delta[n]  = shaped[n] - x[n]        // the EQ's contribution
//! out[n]    = x[n] + depth[n] * delta[n]
//! ```
//!
//! with `depth` in `[0, 1]`: `0` is a flat bypass (output equals input), `1` is
//! the full `range_db` bell. Because the filter never changes, the processor is
//! unconditionally stable and allocation-free; only the scalar `depth`
//! animates. The mapping from `depth` to perceived dB is non-linear but
//! monotonic, which is musically well-behaved.
//!
//! `depth` is driven by a detection path: the per-sample mono average of the
//! active channels is passed through a band-pass tuned to the same
//! `frequency`/`q`, and that band signal feeds a [`LevelDetector`]. The band
//! level in dB produces a target depth that is smoothed with an attack/release
//! one-pole. A single shared `depth` keeps the stereo image coherent (both
//! channels receive the identical bell animation).
//!
//! The control law depends on [`DynamicEqMode`]:
//!
//! - [`DynamicEqMode::Above`] engages the bell as the band rises **above**
//!   `threshold_db` (classic dynamic cut of a resonance, or dynamic boost).
//! - [`DynamicEqMode::Below`] engages the bell as the band falls **below**
//!   `threshold_db` (e.g. lifting a frequency region only in quiet passages).
//!
//! In both modes the overshoot is divided by `width_db` to span the knee from
//! onset (`depth = 0`) to full engagement (`depth = 1`).
//!
//! # Real-time contract
//!
//! The detection filter, per-channel shaping filters, level detector, and
//! smoothing coefficients are all allocated in [`DynamicEqNode::new`].
//! [`process`](crate::graph::AudioNode::process) performs no allocation, takes
//! no locks, and cannot panic: mismatched channel counts and zero-length
//! blocks degrade gracefully, and every recursive filter state is flushed of
//! denormals.
//!
//! # Relationship
//!
//! - [`compressor`](super::compressor): one broadband gain over the whole
//!   signal; a dynamic EQ instead reshapes a single frequency region.
//! - [`multiband`](super::multiband): a fixed Linkwitz-Riley crossover with an
//!   independent compressor per band, recombined flat; a dynamic EQ is a single
//!   overlapping bell, not a split-and-sum.
//! - [`de_esser`](super::de_esser): a split-band compressor dedicated to a
//!   high sibilance band; a dynamic EQ is a general, arbitrarily-tuned bell in
//!   either direction.
//! - [`parametric_eq`](crate::nodes::effects::parametric_eq): the same RBJ
//!   peaking bell but with a *static* gain; a dynamic EQ cross-fades that same
//!   bell by a level-dependent `depth`.
//!
//! It reuses this crate's own [`BiquadCoeffs`] (RBJ cookbook) and dynamics
//! [`detector`](super::detector) primitives so its filter and ballistics match
//! the rest of the engine.
//!
//! # Provenance
//!
//! Dynamic equalisation (a parametric band whose gain tracks a level detector)
//! is a classic studio technique documented across the audio-effects
//! literature (e.g. Reiss and `McPherson`, "Audio Effects", 2014; Zoelzer,
//! "DAFX"). The fixed-filter / depth-crossfade formulation here is a standard
//! real-time-safe realisation. This module contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**; it is implemented purely from that publicly documented theory.

use alloc::vec::Vec;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::nodes::biquad::{BiquadCoeffs, BiquadKind};
use crate::nodes::dynamics::detector::{DetectionMode, LevelDetector, time_to_coef};

/// Default band centre frequency (Hz).
pub const DEFAULT_FREQUENCY_HZ: Sample = 1_000.0;
/// Default band quality factor.
pub const DEFAULT_Q: Sample = 2.0;
/// Default maximum bell gain at full depth (dB). Negative is a dynamic cut.
pub const DEFAULT_RANGE_DB: Sample = -6.0;
/// Default detection threshold (dBFS).
pub const DEFAULT_THRESHOLD_DB: Sample = -24.0;
/// Default knee width (dB) spanning from onset to full depth.
pub const DEFAULT_WIDTH_DB: Sample = 12.0;
/// Default attack time (ms) for the depth envelope.
pub const DEFAULT_ATTACK_MS: Sample = 10.0;
/// Default release time (ms) for the depth envelope.
pub const DEFAULT_RELEASE_MS: Sample = 120.0;
/// RMS averaging window (ms) used by the band level detector.
pub const DETECTION_RMS_WINDOW_MS: Sample = 10.0;
/// Lower bound on `width_db` so the knee division is always numerically valid.
pub const MIN_WIDTH_DB: Sample = 0.1;

/// Direction in which the band crossing engages the bell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DynamicEqMode {
    /// Engage the bell as the band level rises above `threshold_db`.
    #[default]
    Above,
    /// Engage the bell as the band level falls below `threshold_db`.
    Below,
}

/// Configuration for a [`DynamicEqNode`].
///
/// The defaults target a mild 1 kHz dynamic cut: a moderate `Q`, a 6 dB range,
/// a quiet threshold, and musical attack/release times.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DynamicEqParams {
    /// Band centre frequency (Hz).
    pub frequency_hz: Sample,
    /// Band quality factor (bandwidth).
    pub q: Sample,
    /// Maximum bell gain at full depth (dB). Negative cuts, positive boosts.
    pub range_db: Sample,
    /// Detection threshold (dBFS) at which the bell begins to engage.
    pub threshold_db: Sample,
    /// Knee width (dB) spanning the band overshoot from `depth = 0` to `1`.
    pub width_db: Sample,
    /// Attack time (ms) applied when the depth increases.
    pub attack_ms: Sample,
    /// Release time (ms) applied when the depth decreases.
    pub release_ms: Sample,
    /// Direction of engagement (above or below threshold).
    pub mode: DynamicEqMode,
}

impl Default for DynamicEqParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            q: DEFAULT_Q,
            range_db: DEFAULT_RANGE_DB,
            threshold_db: DEFAULT_THRESHOLD_DB,
            width_db: DEFAULT_WIDTH_DB,
            attack_ms: DEFAULT_ATTACK_MS,
            release_ms: DEFAULT_RELEASE_MS,
            mode: DynamicEqMode::Above,
        }
    }
}

/// A single Direct-Form-I biquad with its own delay state.
///
/// The shared [`Biquad`](crate::nodes::biquad::Biquad) exposes only
/// buffer-level processing, but a dynamic EQ needs a per-sample tick so the
/// shaping path can be cross-faded sample by sample; this tiny state carries
/// the two input and two output memories.
#[derive(Debug, Clone, Copy)]
struct DirectFormBiquad {
    b0: Sample,
    b1: Sample,
    b2: Sample,
    a1: Sample,
    a2: Sample,
    x1: Sample,
    x2: Sample,
    y1: Sample,
    y2: Sample,
}

impl DirectFormBiquad {
    fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            b0: coeffs.b0,
            b1: coeffs.b1,
            b2: coeffs.b2,
            a1: coeffs.a1,
            a2: coeffs.a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn set_coeffs(&mut self, coeffs: BiquadCoeffs) {
        self.b0 = coeffs.b0;
        self.b1 = coeffs.b1;
        self.b2 = coeffs.b2;
        self.a1 = coeffs.a1;
        self.a2 = coeffs.a2;
    }

    #[inline]
    fn process(&mut self, x: Sample) -> Sample {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        let y = flush_denormal(y);
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// A level-driven single-band parametric equaliser node.
#[derive(Debug, Clone)]
pub struct DynamicEqNode {
    params: DynamicEqParams,
    sample_rate: u32,
    /// Band-pass used only to measure the band level (mono detection path).
    detect: DirectFormBiquad,
    /// Peaking bell at full `range_db`, one filter state per channel.
    shaping: Vec<DirectFormBiquad>,
    /// RMS level detector following the detection band-pass.
    detector: LevelDetector,
    /// One-pole coefficient applied while depth is increasing.
    attack_coef: Sample,
    /// One-pole coefficient applied while depth is decreasing.
    release_coef: Sample,
    /// Current smoothed cross-fade depth in `[0, 1]`.
    depth: Sample,
}

impl DynamicEqNode {
    /// Builds a dynamic EQ for `channels` channels at `sample_rate`.
    ///
    /// `channels` is clamped to at least one so a filter state always exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    /// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
    /// use prism_audio_core::nodes::dynamics::dynamic_eq::{DynamicEqNode, DynamicEqParams};
    ///
    /// let mut node = DynamicEqNode::new(48_000, 2, DynamicEqParams::default());
    /// let input = AudioBuffer::new(ChannelLayout::Stereo, 64);
    /// let mut output = AudioBuffer::new(ChannelLayout::Stereo, 64);
    /// let inputs = [input];
    /// let mut outputs = [output];
    /// let mut io = ProcessIo::new(&inputs, &mut outputs);
    /// let ctx = RenderContext { sample_rate: 48_000, frames: 64, playhead: 0 };
    /// node.process(&ctx, &mut io);
    /// // Silence in, silence out: the band never crosses the threshold.
    /// assert_eq!(node.depth(), 0.0);
    /// ```
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: DynamicEqParams) -> Self {
        let sample_rate = sample_rate.max(1);
        let channels = channels.max(1);
        let detect_coeffs =
            BiquadCoeffs::design(BiquadKind::BandPass, sample_rate, params.frequency_hz, params.q, 0.0);
        let shaping_coeffs = BiquadCoeffs::design(
            BiquadKind::Peaking,
            sample_rate,
            params.frequency_hz,
            params.q,
            params.range_db,
        );
        let mut shaping = Vec::with_capacity(channels);
        for _ in 0..channels {
            shaping.push(DirectFormBiquad::new(shaping_coeffs));
        }
        Self {
            params,
            sample_rate,
            detect: DirectFormBiquad::new(detect_coeffs),
            shaping,
            detector: LevelDetector::new(DetectionMode::Rms, DETECTION_RMS_WINDOW_MS, sample_rate),
            attack_coef: time_to_coef(params.attack_ms, sample_rate),
            release_coef: time_to_coef(params.release_ms, sample_rate),
            depth: 0.0,
        }
    }

    /// Replaces the parameters, redesigning the filters and recomputing the
    /// smoothing coefficients while preserving the running filter and envelope
    /// state so the change is click-free.
    pub fn set_params(&mut self, params: DynamicEqParams) {
        let detect_coeffs = BiquadCoeffs::design(
            BiquadKind::BandPass,
            self.sample_rate,
            params.frequency_hz,
            params.q,
            0.0,
        );
        let shaping_coeffs = BiquadCoeffs::design(
            BiquadKind::Peaking,
            self.sample_rate,
            params.frequency_hz,
            params.q,
            params.range_db,
        );
        self.detect.set_coeffs(detect_coeffs);
        for filter in &mut self.shaping {
            filter.set_coeffs(shaping_coeffs);
        }
        self.attack_coef = time_to_coef(params.attack_ms, self.sample_rate);
        self.release_coef = time_to_coef(params.release_ms, self.sample_rate);
        self.params = params;
    }

    /// Returns the configured engagement direction.
    #[must_use]
    pub fn mode(&self) -> DynamicEqMode {
        self.params.mode
    }

    /// Returns the current smoothed cross-fade depth in `[0, 1]`.
    #[must_use]
    pub fn depth(&self) -> Sample {
        self.depth
    }
}

impl AudioNode for DynamicEqNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let proc = input
            .channels()
            .min(output.channels())
            .min(self.shaping.len());
        let frames = input.active_frames().min(output.active_frames());
        if proc == 0 || frames == 0 {
            return;
        }
        let inv_channels = 1.0 / proc as Sample;
        let width = self.params.width_db.max(MIN_WIDTH_DB);
        let threshold = self.params.threshold_db;
        for f in 0..frames {
            // Detection: mono average of the active channels through the band-pass.
            let mut mono = 0.0;
            for ch in 0..proc {
                mono += input.channel(ch)[f];
            }
            mono *= inv_channels;
            let band = self.detect.process(mono);
            let band_level_db = self.detector.level_db(band);

            let drive = match self.params.mode {
                DynamicEqMode::Above => (band_level_db - threshold).max(0.0),
                DynamicEqMode::Below => (threshold - band_level_db).max(0.0),
            };
            let depth_target = (drive / width).clamp(0.0, 1.0);
            let coef = if depth_target > self.depth {
                self.attack_coef
            } else {
                self.release_coef
            };
            self.depth = coef * self.depth + (1.0 - coef) * depth_target;
            let depth = self.depth;

            for ch in 0..proc {
                let x = input.channel(ch)[f];
                let shaped = self.shaping[ch].process(x);
                output.channel_mut(ch)[f] = x + depth * (shaped - x);
            }
        }
    }

    fn reset(&mut self) {
        self.detect.reset();
        for filter in &mut self.shaping {
            filter.reset();
        }
        self.detector.reset();
        self.depth = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use bevy_math::ops;

    const SR: u32 = 48_000;

    fn render(node: &mut DynamicEqNode, input: &AudioBuffer) -> AudioBuffer {
        let mut output = AudioBuffer::new(input.layout(), input.capacity_frames());
        output.set_active_frames(input.active_frames());
        let inputs = [input.clone()];
        let mut outputs = [output];
        let ctx = RenderContext {
            sample_rate: SR,
            frames: input.active_frames(),
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        let [out] = outputs;
        out
    }

    fn tone(layout: ChannelLayout, freq_hz: Sample, amp: Sample, frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        buf.set_active_frames(frames);
        let w = 2.0 * core::f32::consts::PI * freq_hz / SR as Sample;
        for ch in 0..buf.channels() {
            let data = buf.channel_mut(ch);
            for (n, s) in data.iter_mut().enumerate() {
                *s = amp * ops::sin(w * n as Sample);
            }
        }
        buf
    }

    fn tail_rms(data: &[Sample]) -> Sample {
        let start = data.len() / 2;
        let slice = &data[start..];
        let mut acc = 0.0f64;
        for &s in slice {
            acc += f64::from(s) * f64::from(s);
        }
        ops::sqrt((acc / slice.len() as f64) as Sample)
    }

    #[test]
    fn silence_is_flat_bypass() {
        let mut node = DynamicEqNode::new(SR, 2, DynamicEqParams::default());
        let input = AudioBuffer::new(ChannelLayout::Stereo, 256);
        let out = render(&mut node, &input);
        for ch in 0..out.channels() {
            for &s in out.channel(ch) {
                assert!(s.abs() < 1.0e-6, "silence must pass through: {s}");
            }
        }
        assert_eq!(node.depth(), 0.0);
    }

    #[test]
    fn below_threshold_bypasses() {
        // Very loud threshold: a quiet in-band tone never engages the bell.
        let params = DynamicEqParams {
            frequency_hz: 1_000.0,
            q: 2.0,
            range_db: -12.0,
            threshold_db: 0.0,
            width_db: 12.0,
            mode: DynamicEqMode::Above,
            ..Default::default()
        };
        let mut node = DynamicEqNode::new(SR, 1, params);
        let input = tone(ChannelLayout::Mono, 1_000.0, 0.01, 4_800);
        let out = render(&mut node, &input);
        let in_rms = tail_rms(input.channel(0));
        let out_rms = tail_rms(out.channel(0));
        assert!(node.depth() < 0.05, "depth must stay low: {}", node.depth());
        assert!((out_rms - in_rms).abs() / in_rms < 0.05, "near flat passthrough");
    }

    #[test]
    fn above_threshold_cuts_in_band_energy() {
        let params = DynamicEqParams {
            frequency_hz: 1_000.0,
            q: 2.0,
            range_db: -12.0,
            threshold_db: -60.0,
            width_db: 6.0,
            attack_ms: 1.0,
            release_ms: 20.0,
            mode: DynamicEqMode::Above,
        };
        let mut node = DynamicEqNode::new(SR, 1, params);
        let input = tone(ChannelLayout::Mono, 1_000.0, 0.5, 9_600);
        let out = render(&mut node, &input);
        let in_rms = tail_rms(input.channel(0));
        let out_rms = tail_rms(out.channel(0));
        assert!(node.depth() > 0.5, "depth must engage: {}", node.depth());
        assert!(out_rms < in_rms * 0.6, "band energy must drop: {out_rms} vs {in_rms}");
    }

    #[test]
    fn above_threshold_boosts_when_range_positive() {
        let params = DynamicEqParams {
            frequency_hz: 1_000.0,
            q: 2.0,
            range_db: 12.0,
            threshold_db: -60.0,
            width_db: 6.0,
            attack_ms: 1.0,
            release_ms: 20.0,
            mode: DynamicEqMode::Above,
        };
        let mut node = DynamicEqNode::new(SR, 1, params);
        let input = tone(ChannelLayout::Mono, 1_000.0, 0.2, 9_600);
        let out = render(&mut node, &input);
        let in_rms = tail_rms(input.channel(0));
        let out_rms = tail_rms(out.channel(0));
        assert!(out_rms > in_rms * 1.5, "positive range must boost: {out_rms} vs {in_rms}");
    }

    #[test]
    fn out_of_band_tone_is_untouched() {
        let params = DynamicEqParams {
            frequency_hz: 1_000.0,
            q: 4.0,
            range_db: -18.0,
            threshold_db: -60.0,
            width_db: 6.0,
            mode: DynamicEqMode::Above,
            ..Default::default()
        };
        let mut node = DynamicEqNode::new(SR, 1, params);
        let input = tone(ChannelLayout::Mono, 8_000.0, 0.5, 9_600);
        let out = render(&mut node, &input);
        // Even if the detector engages, a 1 kHz bell leaves an 8 kHz tone
        // essentially untouched: the shaping is frequency-selective.
        let in_rms = tail_rms(input.channel(0));
        let out_rms = tail_rms(out.channel(0));
        assert!((out_rms - in_rms).abs() / in_rms < 0.1, "out-of-band stays flat");
    }

    #[test]
    fn below_mode_engages_when_quiet() {
        let params = DynamicEqParams {
            frequency_hz: 1_000.0,
            q: 2.0,
            range_db: 12.0,
            threshold_db: -6.0,
            width_db: 12.0,
            attack_ms: 1.0,
            release_ms: 20.0,
            mode: DynamicEqMode::Below,
        };
        let mut node = DynamicEqNode::new(SR, 1, params);
        // Quiet in-band tone: level sits below threshold, so Below mode engages.
        let input = tone(ChannelLayout::Mono, 1_000.0, 0.02, 9_600);
        let out = render(&mut node, &input);
        let in_rms = tail_rms(input.channel(0));
        let out_rms = tail_rms(out.channel(0));
        assert!(node.depth() > 0.5, "below mode must engage when quiet: {}", node.depth());
        assert!(out_rms > in_rms * 1.5, "quiet band is lifted: {out_rms} vs {in_rms}");
    }

    #[test]
    fn stereo_channels_share_depth() {
        let params = DynamicEqParams {
            threshold_db: -60.0,
            range_db: -12.0,
            ..Default::default()
        };
        let mut node = DynamicEqNode::new(SR, 2, params);
        let input = tone(ChannelLayout::Stereo, 1_000.0, 0.5, 4_800);
        let out = render(&mut node, &input);
        // Identical channels in -> identical channels out (one shared depth).
        for (a, b) in out.channel(0).iter().zip(out.channel(1).iter()) {
            assert!((a - b).abs() < 1.0e-6, "channels must match: {a} vs {b}");
        }
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = DynamicEqNode::new(SR, 2, DynamicEqParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 64);
        input.set_active_frames(0);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 64);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 0,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        assert_eq!(node.depth(), 0.0);
    }

    #[test]
    fn reset_clears_state() {
        let params = DynamicEqParams {
            threshold_db: -60.0,
            range_db: -12.0,
            ..Default::default()
        };
        let mut node = DynamicEqNode::new(SR, 1, params);
        let input = tone(ChannelLayout::Mono, 1_000.0, 0.5, 4_800);
        let _ = render(&mut node, &input);
        assert!(node.depth() > 0.0);
        node.reset();
        assert_eq!(node.depth(), 0.0);
    }

    #[test]
    fn mode_accessor_reports_configuration() {
        let node = DynamicEqNode::new(SR, 1, DynamicEqParams::default());
        assert_eq!(node.mode(), DynamicEqMode::Above);
        let node2 = DynamicEqNode::new(
            SR,
            1,
            DynamicEqParams {
                mode: DynamicEqMode::Below,
                ..Default::default()
            },
        );
        assert_eq!(node2.mode(), DynamicEqMode::Below);
    }

    #[test]
    fn set_params_preserves_running_state() {
        let mut node = DynamicEqNode::new(SR, 1, DynamicEqParams::default());
        let input = tone(ChannelLayout::Mono, 1_000.0, 0.3, 2_400);
        let _ = render(&mut node, &input);
        let depth_before = node.depth();
        node.set_params(DynamicEqParams {
            frequency_hz: 2_000.0,
            ..Default::default()
        });
        // set_params does not reset the envelope, so depth is unchanged.
        assert_eq!(node.depth(), depth_before);
    }

    #[test]
    fn default_params_are_sane() {
        let p = DynamicEqParams::default();
        assert_eq!(p.frequency_hz, DEFAULT_FREQUENCY_HZ);
        assert_eq!(p.q, DEFAULT_Q);
        assert_eq!(p.range_db, DEFAULT_RANGE_DB);
        assert_eq!(p.mode, DynamicEqMode::Above);
    }
}
