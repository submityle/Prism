//! Upward compressor: a feed-forward dynamics processor that *raises* the level
//! of signal sitting below a threshold, pulling quiet passages up toward the
//! threshold instead of pushing loud peaks down.
//!
//! A conventional (downward) compressor attenuates everything *above* a
//! threshold. Upward compression is its complement: it applies a positive gain
//! *below* the threshold, so the softest material is brought closer to the
//! loudest, reducing dynamic range from the bottom. It is a staple of modern
//! mastering and dialogue levelling, where the goal is to make low-level detail
//! audible without clamping transients.
//!
//! # The curve
//!
//! Given the side-chain level `L` (dB), threshold `T`, ratio `R >= 1`, and the
//! distance below threshold `under = T - L`, the static boost is
//!
//! ```text
//! boost = (1 - 1/R) * under        (clamped to [0, max_gain_db])
//! ```
//!
//! so the output level becomes `L + boost = T - under / R`: at `R = 1` no boost
//! is applied (transparent), and as `R` grows the quiet level is lifted ever
//! closer to the threshold. A soft knee of `knee_db` smooths the transition
//! across the threshold, and `max_gain_db` caps how far the noise floor can be
//! lifted so the processor cannot run away on silence.
//!
//! # Model
//!
//! Detection is **stereo-linked**: the loudest channel drives one control
//! signal applied to every channel, so upward compression never shifts the
//! stereo image. The boost is smoothed by the shared decoupled attack / release
//! ballistics (`attack` = how fast the boost engages as the signal drops,
//! `release` = how slowly it backs off as the signal returns), then converted
//! to a linear gain and mixed with an optional make-up gain and a dry path for
//! parallel operation.
//!
//! # Real-time contract
//!
//! All detector and ballistics state is allocated in
//! [`UpwardCompressorNode::new`].
//! [`UpwardCompressorNode::process`] performs no allocation, takes no locks, and
//! cannot panic: mismatched channel counts and zero-length blocks degrade
//! gracefully, non-finite inputs are treated as silence, and the smoothed gain
//! is flushed of denormals. Latency is zero (there is no look-ahead).
//!
//! # Relationship
//!
//! This node reuses the dynamics family's shared building blocks -- the
//! [`LevelDetector`](crate::nodes::dynamics::detector::LevelDetector) /
//! [`DetectionMode`](crate::nodes::dynamics::detector::DetectionMode) side-chain
//! and the decoupled
//! [`GainBallistics`](crate::nodes::dynamics::detector::GainBallistics). Its
//! gain computer is the sign-flipped sibling of
//! [`compressor_reduction_db`](crate::nodes::dynamics::detector::compressor_reduction_db):
//! where the downward compressor returns a non-negative *reduction* applied
//! above the threshold, this returns a non-negative *boost* applied below it.
//! It is distinct from the downward expander / gate
//! ([`gate`](crate::nodes::dynamics::gate)), which attenuates below the
//! threshold.
//!
//! # Provenance
//!
//! Pure classic DSP. Upward compression is a standard dynamics technique
//! described in the audio-engineering literature (e.g. Reiss and `McPherson`,
//! "Audio Effects", 2014; Giannoulis et al., "Digital Dynamic Range Compressor
//! Design", JAES 2012). There is no AI/ML of any kind, and no Unreal Engine,
//! Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, or Web Audio
//! source or derived code; only the publicly documented static curve and
//! ballistics are used.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};
use crate::nodes::dynamics::detector::{DetectionMode, GainBallistics, LevelDetector};
use crate::param::{Ramp, Smoothed};

/// Largest upward gain (dB) the node will apply, regardless of `max_gain_db`.
///
/// A hard ceiling that bounds how far the noise floor can ever be lifted.
pub const MAX_UPWARD_GAIN_DB: Sample = 48.0;

/// Default threshold in dBFS below which upward compression engages.
pub const DEFAULT_UPWARD_THRESHOLD_DB: Sample = -24.0;

/// Default upward-compression ratio.
pub const DEFAULT_UPWARD_RATIO: Sample = 2.0;

/// Default ceiling (dB) on the applied upward gain.
pub const DEFAULT_UPWARD_MAX_GAIN_DB: Sample = 12.0;

/// Static upward-compression curve.
///
/// Given the side-chain `level_db`, the `threshold_db`, a ratio `ratio`
/// (`>= 1`), a soft `knee_db` width, and a ceiling `max_gain_db`, returns the
/// gain **boost** to apply as a non-negative number of decibels (0 = above
/// threshold, larger = quieter input lifted further). Mirrors
/// [`compressor_reduction_db`](crate::nodes::dynamics::detector::compressor_reduction_db)
/// with the direction inverted.
#[inline]
#[must_use]
pub fn upward_boost_db(
    level_db: Sample,
    threshold_db: Sample,
    ratio: Sample,
    knee_db: Sample,
    max_gain_db: Sample,
) -> Sample {
    let ratio = ratio.max(1.0);
    let slope = 1.0 - 1.0 / ratio;
    let knee = knee_db.max(0.0);
    let max_gain = max_gain_db.clamp(0.0, MAX_UPWARD_GAIN_DB);
    // Distance below threshold (positive when quieter than the threshold).
    let under = threshold_db - level_db;

    let boost = if under <= -knee * 0.5 {
        // Above threshold (plus half-knee): transparent.
        0.0
    } else if knee > 0.0 && under < knee * 0.5 {
        // Soft-knee quadratic region straddling the threshold.
        let t = under + knee * 0.5;
        slope * (t * t) / (2.0 * knee)
    } else {
        // Fully below threshold: linear upward compression.
        slope * under
    };

    boost.clamp(0.0, max_gain)
}

/// Construction parameters for an [`UpwardCompressorNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct UpwardCompressorParams {
    /// Threshold in dBFS below which upward compression begins.
    pub threshold_db: Sample,
    /// Upward-compression ratio (`>= 1`); `1.0` is transparent.
    pub ratio: Sample,
    /// Soft-knee width in dB centered on the threshold (`0` = hard knee).
    pub knee_db: Sample,
    /// Ceiling in dB on the applied upward gain (bounds noise-floor lift).
    pub max_gain_db: Sample,
    /// Attack time in milliseconds (how fast the boost engages).
    pub attack_ms: Sample,
    /// Release time in milliseconds (how slowly the boost backs off).
    pub release_ms: Sample,
    /// Side-chain detection mode (peak or RMS).
    pub detection: DetectionMode,
    /// RMS averaging window in milliseconds (ignored for peak detection).
    pub rms_window_ms: Sample,
    /// Make-up gain in dB applied after upward compression.
    pub makeup_db: Sample,
    /// Wet (processed) mix gain for parallel operation.
    pub wet: Sample,
    /// Dry (unprocessed) mix gain for parallel operation.
    pub dry: Sample,
}

impl Default for UpwardCompressorParams {
    fn default() -> Self {
        Self {
            threshold_db: DEFAULT_UPWARD_THRESHOLD_DB,
            ratio: DEFAULT_UPWARD_RATIO,
            knee_db: 6.0,
            max_gain_db: DEFAULT_UPWARD_MAX_GAIN_DB,
            attack_ms: 10.0,
            release_ms: 200.0,
            detection: DetectionMode::Rms,
            rms_window_ms: 10.0,
            makeup_db: 0.0,
            wet: 1.0,
            dry: 0.0,
        }
    }
}

/// A stereo-linked feed-forward upward compressor (input port 0 -> output
/// port 0).
#[derive(Debug, Clone)]
pub struct UpwardCompressorNode {
    /// Threshold in dBFS.
    threshold_db: Sample,
    /// Upward-compression ratio.
    ratio: Sample,
    /// Soft-knee width in dB.
    knee_db: Sample,
    /// Ceiling on the applied upward gain in dB.
    max_gain_db: Sample,
    /// Level detector shared across channels (stereo-linked).
    detector: LevelDetector,
    /// Attack / release ballistics on the upward-gain control.
    ballistics: GainBallistics,
    /// Smoothed make-up gain (linear).
    makeup: Smoothed,
    /// Smoothed wet mix gain.
    wet: Smoothed,
    /// Smoothed dry mix gain.
    dry: Smoothed,
}

impl UpwardCompressorNode {
    /// Builds an upward compressor for `channels` channels at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, _channels: usize, params: UpwardCompressorParams) -> Self {
        Self {
            threshold_db: params.threshold_db,
            ratio: params.ratio.max(1.0),
            knee_db: params.knee_db.max(0.0),
            max_gain_db: params.max_gain_db.clamp(0.0, MAX_UPWARD_GAIN_DB),
            detector: LevelDetector::new(params.detection, params.rms_window_ms, sample_rate),
            ballistics: GainBallistics::new(params.attack_ms, params.release_ms, sample_rate),
            makeup: Smoothed::new(db_to_linear(params.makeup_db)),
            wet: Smoothed::new(params.wet),
            dry: Smoothed::new(params.dry),
        }
    }

    /// Returns the current smoothed upward gain in decibels (`>= 0`).
    #[inline]
    #[must_use]
    pub fn gain_boost_db(&self) -> Sample {
        self.ballistics.current_db()
    }

    /// Returns the current threshold in dBFS.
    #[inline]
    #[must_use]
    pub fn threshold_db(&self) -> Sample {
        self.threshold_db
    }

    /// Returns the current upward-compression ratio.
    #[inline]
    #[must_use]
    pub fn ratio(&self) -> Sample {
        self.ratio
    }

    /// Returns the current upward-gain ceiling in dB.
    #[inline]
    #[must_use]
    pub fn max_gain_db(&self) -> Sample {
        self.max_gain_db
    }

    /// Sets the threshold in dBFS.
    #[inline]
    pub fn set_threshold_db(&mut self, threshold_db: Sample) {
        self.threshold_db = threshold_db;
    }

    /// Sets the upward-compression ratio (clamped to `>= 1`).
    #[inline]
    pub fn set_ratio(&mut self, ratio: Sample) {
        self.ratio = ratio.max(1.0);
    }

    /// Sets the upward-gain ceiling in dB (clamped to `[0, MAX_UPWARD_GAIN_DB]`).
    #[inline]
    pub fn set_max_gain_db(&mut self, max_gain_db: Sample) {
        self.max_gain_db = max_gain_db.clamp(0.0, MAX_UPWARD_GAIN_DB);
    }

    /// Updates the attack / release times in milliseconds (state preserved).
    #[inline]
    pub fn set_times(&mut self, attack_ms: Sample, release_ms: Sample, sample_rate: u32) {
        self.ballistics.set_times(attack_ms, release_ms, sample_rate);
    }

    /// Sets the make-up gain in dB.
    #[inline]
    pub fn set_makeup_db(&mut self, makeup_db: Sample, ramp: Ramp) {
        self.makeup.set_target(db_to_linear(makeup_db), ramp);
    }

    /// Sets the wet mix gain.
    #[inline]
    pub fn set_wet(&mut self, wet: Sample, ramp: Ramp) {
        self.wet.set_target(wet, ramp);
    }

    /// Sets the dry mix gain.
    #[inline]
    pub fn set_dry(&mut self, dry: Sample, ramp: Ramp) {
        self.dry.set_target(dry, ramp);
    }
}

impl AudioNode for UpwardCompressorNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels());
        let frames = output.active_frames().min(input.active_frames());
        if frames == 0 || channels == 0 {
            return;
        }

        for f in 0..frames {
            // Stereo-linked detection uses the loudest channel.
            let mut peak = 0.0;
            for ch in 0..channels {
                let v = input.channel(ch)[f];
                let a = if v.is_finite() { v.abs() } else { 0.0 };
                if a > peak {
                    peak = a;
                }
            }

            let level_db = self.detector.level_db(peak);
            let target = upward_boost_db(
                level_db,
                self.threshold_db,
                self.ratio,
                self.knee_db,
                self.max_gain_db,
            );
            let boost_db = self.ballistics.process(target);
            let gain = flush_denormal(db_to_linear(boost_db) * self.makeup.next_sample());
            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();

            for ch in 0..channels {
                let v = input.channel(ch)[f];
                let x = if v.is_finite() { v } else { 0.0 };
                output.channel_mut(ch)[f] = dry * x + wet * (x * gain);
            }
        }
    }

    fn reset(&mut self) {
        self.detector.reset();
        self.ballistics.reset();
        self.makeup = Smoothed::new(self.makeup.target());
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    #[inline]
    fn ops_sin(x: Sample) -> Sample {
        bevy_math::ops::sin(x)
    }

    /// Fills a mono buffer with a steady sine of the given linear amplitude and
    /// runs it through the node, returning the output samples.
    fn run_mono(node: &mut UpwardCompressorNode, amp: Sample, len: usize) -> Vec<Sample> {
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = amp * ops_sin(TAU_1K * i as Sample);
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        outputs[0].channel(0).to_vec()
    }

    /// One kHz phase increment at the test sample rate.
    const TAU_1K: Sample = core::f32::consts::TAU * 1_000.0 / 48_000.0;

    fn tail_peak(v: &[Sample], skip: usize) -> Sample {
        v.iter().skip(skip).fold(0.0_f32, |m, &x| m.max(x.abs()))
    }

    #[test]
    fn latency_is_zero() {
        let node = UpwardCompressorNode::new(SR, 2, UpwardCompressorParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn default_params_are_sane() {
        let p = UpwardCompressorParams::default();
        assert!(p.ratio >= 1.0);
        assert!(p.max_gain_db >= 0.0);
        assert!(p.knee_db >= 0.0);
    }

    #[test]
    fn curve_transparent_above_threshold() {
        // Level well above threshold -> no boost.
        let b = upward_boost_db(-6.0, -24.0, 2.0, 0.0, 12.0);
        assert!(b.abs() < 1e-6, "{b}");
    }

    #[test]
    fn curve_boosts_below_threshold_by_slope() {
        // 24 dB below threshold at 2:1 hard knee -> slope 0.5 -> boost 12 dB,
        // but capped at max_gain 12 -> 12.
        let b = upward_boost_db(-48.0, -24.0, 2.0, 0.0, 24.0);
        assert!((b - 12.0).abs() < 1e-4, "{b}");
    }

    #[test]
    fn curve_capped_by_max_gain() {
        let b = upward_boost_db(-80.0, -24.0, 4.0, 0.0, 6.0);
        assert!((b - 6.0).abs() < 1e-6, "{b}");
    }

    #[test]
    fn curve_ratio_one_is_transparent() {
        for &l in &[-60.0, -40.0, -24.0, -6.0] {
            let b = upward_boost_db(l, -24.0, 1.0, 6.0, 12.0);
            assert!(b.abs() < 1e-6, "level {l} -> {b}");
        }
    }

    #[test]
    fn curve_soft_knee_is_continuous() {
        // At the threshold with a soft knee the boost is small and positive.
        let b = upward_boost_db(-24.0, -24.0, 2.0, 6.0, 12.0);
        assert!(b > 0.0 && b < 1.0, "{b}");
        // The knee value matches the linear region at the lower knee edge.
        let edge_quad = upward_boost_db(-27.0, -24.0, 2.0, 6.0, 48.0);
        let edge_lin = upward_boost_db(-27.001, -24.0, 2.0, 0.0, 48.0);
        assert!((edge_quad - edge_lin).abs() < 1e-2, "{edge_quad} vs {edge_lin}");
    }

    #[test]
    fn quiet_signal_is_boosted() {
        let mut node = UpwardCompressorNode::new(SR, 1, UpwardCompressorParams::default());
        // -40 dBFS sine (amp ~0.01), well below the -24 dB threshold.
        let amp = 0.01;
        let out = run_mono(&mut node, amp, 48_000);
        let peak = tail_peak(&out, 24_000);
        assert!(peak > amp * 1.5, "quiet signal should be lifted: {peak} vs {amp}");
        assert!(node.gain_boost_db() > 0.0);
    }

    #[test]
    fn loud_signal_passes_unchanged() {
        let mut node = UpwardCompressorNode::new(SR, 1, UpwardCompressorParams::default());
        // -6 dBFS sine (amp ~0.5), above the -24 dB threshold -> no boost.
        // Run long enough (1 s) for the start-up detector charge transient to
        // fully release through the 200 ms ballistics, then measure the settled
        // tail (last ~83 ms).
        let amp = 0.5;
        let out = run_mono(&mut node, amp, 48_000);
        let peak = tail_peak(&out, 44_000);
        assert!((peak - amp).abs() < amp * 0.05, "loud signal altered: {peak} vs {amp}");
    }

    #[test]
    fn boost_bounded_by_max_gain() {
        let params = UpwardCompressorParams {
            max_gain_db: 6.0,
            attack_ms: 1.0,
            ..UpwardCompressorParams::default()
        };
        let mut node = UpwardCompressorNode::new(SR, 1, params);
        // Extremely quiet input drives the boost to its ceiling.
        let _ = run_mono(&mut node, 1e-4, 48_000);
        assert!(node.gain_boost_db() <= 6.0 + 1e-3, "{}", node.gain_boost_db());
    }

    #[test]
    fn ratio_one_is_transparent_through_node() {
        let params = UpwardCompressorParams {
            ratio: 1.0,
            ..UpwardCompressorParams::default()
        };
        let mut node = UpwardCompressorNode::new(SR, 1, params);
        let amp = 0.02;
        let out = run_mono(&mut node, amp, 8_000);
        let peak = tail_peak(&out, 4_000);
        assert!((peak - amp).abs() < amp * 0.05, "ratio 1 altered signal: {peak} vs {amp}");
    }

    #[test]
    fn stereo_linked_applies_same_gain() {
        let mut node = UpwardCompressorNode::new(SR, 2, UpwardCompressorParams::default());
        let len = 48_000;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        // Left loud (above threshold), right quiet (below) -> the linked
        // detector follows the loud channel, so the quiet channel is NOT
        // boosted independently.
        for i in 0..len {
            input.channel_mut(0)[i] = 0.5 * ops_sin(TAU_1K * i as Sample);
            input.channel_mut(1)[i] = 0.01 * ops_sin(TAU_1K * i as Sample);
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        // Loud channel is above threshold so almost no gain is applied; the
        // quiet right channel inherits that same (near-unity) gain.
        let right_peak = tail_peak(outputs[0].channel(1), 24_000);
        assert!(right_peak < 0.01 * 2.0, "right over-boosted despite link: {right_peak}");
    }

    #[test]
    fn dry_path_is_bit_identical() {
        let params = UpwardCompressorParams {
            wet: 0.0,
            dry: 1.0,
            ..UpwardCompressorParams::default()
        };
        let mut node = UpwardCompressorNode::new(SR, 1, params);
        let amp = 0.01;
        let out = run_mono(&mut node, amp, 2_048);
        for (i, &o) in out.iter().enumerate() {
            let expected = amp * ops_sin(TAU_1K * i as Sample);
            assert!((o - expected).abs() < 1e-6, "{o} vs {expected}");
        }
    }

    #[test]
    fn makeup_gain_is_applied() {
        let params = UpwardCompressorParams {
            ratio: 1.0, // no upward boost; isolate make-up
            makeup_db: 6.0,
            ..UpwardCompressorParams::default()
        };
        let mut node = UpwardCompressorNode::new(SR, 1, params);
        let amp = 0.1;
        let out = run_mono(&mut node, amp, 8_000);
        let peak = tail_peak(&out, 4_000);
        // +6 dB ~= x1.995.
        assert!(peak > amp * 1.8 && peak < amp * 2.1, "makeup not applied: {peak}");
    }

    #[test]
    fn output_finite_for_non_finite_input() {
        let mut node = UpwardCompressorNode::new(SR, 1, UpwardCompressorParams::default());
        let len = 64;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        input.channel_mut(0)[0] = Sample::NAN;
        input.channel_mut(0)[1] = Sample::INFINITY;
        input.channel_mut(0)[2] = Sample::NEG_INFINITY;
        input.channel_mut(0)[3] = 0.2;
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        assert!(outputs[0].channel(0).iter().all(|s| s.is_finite()));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = UpwardCompressorNode::new(SR, 1, UpwardCompressorParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 1);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 1);
        input.set_active_frames(0);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
    }

    #[test]
    fn reset_reproduces_fresh_state() {
        let mut node = UpwardCompressorNode::new(SR, 1, UpwardCompressorParams::default());
        let first = run_mono(&mut node, 0.01, 4_096);
        node.reset();
        let second = run_mono(&mut node, 0.01, 4_096);
        for (a, b) in first.iter().zip(second.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn gain_boost_is_non_negative() {
        let mut node = UpwardCompressorNode::new(SR, 1, UpwardCompressorParams::default());
        let _ = run_mono(&mut node, 0.005, 8_000);
        assert!(node.gain_boost_db() >= 0.0);
    }

    #[test]
    fn setters_update_state() {
        let mut node = UpwardCompressorNode::new(SR, 1, UpwardCompressorParams::default());
        node.set_threshold_db(-30.0);
        node.set_ratio(0.5); // clamped up to 1
        node.set_max_gain_db(100.0); // clamped to MAX
        assert!((node.threshold_db() + 30.0).abs() < 1e-6);
        assert!((node.ratio() - 1.0).abs() < 1e-6);
        assert!((node.max_gain_db() - MAX_UPWARD_GAIN_DB).abs() < 1e-6);
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let params = UpwardCompressorParams {
            threshold_db: Sample::NAN,
            ratio: -5.0,
            knee_db: -1.0,
            max_gain_db: 1e9,
            attack_ms: -1.0,
            release_ms: -1.0,
            rms_window_ms: -1.0,
            ..UpwardCompressorParams::default()
        };
        let mut node = UpwardCompressorNode::new(SR, 2, params);
        let out = run_mono(&mut node, 0.01, 512);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn mono_into_stereo_is_finite() {
        let mut node = UpwardCompressorNode::new(SR, 2, UpwardCompressorParams::default());
        let len = 256;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = 0.02 * ops_sin(TAU_1K * i as Sample);
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        // Only the shared min-channel count is processed; the output stays
        // finite and the untouched channel stays silent.
        assert!(outputs[0].channel(0).iter().all(|s| s.is_finite()));
        assert!(outputs[0].channel(1).iter().all(|s| *s == 0.0));
    }
}
