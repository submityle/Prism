//! Target-loudness normalizer with a true-peak ceiling guard.
//!
//! Modern delivery pipelines (streaming, broadcast, game cinematics) ship a
//! finished mix at a fixed program loudness rather than a fixed peak level, so
//! that content from different sources plays back at a perceptually consistent
//! level. The standard workflow is two pass: first *measure* the integrated
//! program loudness and true-peak of the finished mix, then apply a single
//! broadband makeup gain that moves the measured loudness onto the delivery
//! target while keeping the true-peak under the delivery ceiling.
//!
//! This node implements the *apply* half of that workflow. It does not measure
//! loudness -- a single streaming pass cannot know the integrated loudness of
//! audio it has not yet seen -- so the measured integrated loudness and
//! measured true-peak are supplied as parameters, exactly the values produced
//! by [`LoudnessMeter`](crate::nodes::analysis::LoudnessMeter) in an analysis
//! pass. Given those measurements and a delivery target it computes the exact
//! static gain and applies it click-free, so there is no hidden estimation or
//! fake real-time measurement.
//!
//! # Model
//!
//! From a measured integrated loudness `M` (in `LUFS`) and a target `T` (in
//! `LUFS`) the ideal makeup gain is `T - M` decibels (`LUFS` differences are
//! decibel differences). Two guards bound that gain:
//!
//! - **True-peak ceiling.** Applying `g` dB of gain raises the measured
//!   true-peak `P` (in `dBTP`) to `P + g`. To keep the output under a ceiling
//!   `C` (in `dBTP`, `-1` `dBTP` by default per `EBU` `R128`) the gain is
//!   capped at `C - P`. This only ever *reduces* an upward gain, and if the
//!   source already exceeds the ceiling it forces an attenuation.
//! - **Gain limit.** The result is clamped to `+/- max_gain_db` so that
//!   near-silent or very quiet sources are not boosted into their noise floor.
//!
//! A source whose measured loudness sits at or below the `BS.1770` absolute
//! gate (`-70` `LUFS`), or whose measurements are not finite, is treated as
//! unmeasurable and left at unity gain.
//!
//! # Real-time contract
//!
//! The gain computation ([`normalization_gain_db`]) is a handful of scalar
//! operations and is allocation-free, lock-free, and panic-free. The node hot
//! path ([`LoudnessNormalizerNode::process`]) applies the gain through a
//! [`Smoothed`] value so that re-targeting during playback glides without
//! zipper noise; it allocates nothing and never panics. Re-targeting setters
//! are control-rate and merely update a stored target.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The loudness and
//! true-peak conventions are implemented from the publicly published standards
//! `ITU-R` `BS.1770-4` and `EBU` `R128`. It is pure classic DSP with no AI/ML.
//!
//! # Relationship
//!
//! This node is the signal-altering counterpart to the read-only
//! [`LoudnessMeter`](crate::nodes::analysis::LoudnessMeter): the meter reports
//! the integrated loudness and true-peak that are fed here as parameters. It
//! applies a broadband level trim like [`GainNode`](crate::nodes::GainNode) and
//! reuses the same [`Smoothed`] click-free glide, but adds the loudness-target
//! and true-peak-ceiling policy on top; it does not re-implement metering,
//! limiting, or filtering. It is a natural pre-stage for the
//! [`MasteringChainNode`](crate::nodes::mastering::MasteringChainNode).

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, linear_to_db};
use crate::param::{Ramp, Smoothed};

/// Default delivery target loudness (`-14` `LUFS`, the common streaming
/// target).
pub const DEFAULT_TARGET_LUFS: Sample = -14.0;

/// Default true-peak delivery ceiling (`-1` `dBTP`, per `EBU` `R128`).
pub const DEFAULT_MAX_TRUE_PEAK_DBTP: Sample = -1.0;

/// Default limit on the absolute makeup gain, in decibels.
pub const DEFAULT_MAX_GAIN_DB: Sample = 24.0;

/// Absolute silence gate from `ITU-R` `BS.1770` (`-70` `LUFS`); a measured
/// loudness at or below this is treated as unmeasurable.
pub const SILENCE_GATE_LUFS: Sample = -70.0;

/// Default glide time for a re-target, in seconds.
pub const DEFAULT_RAMP_SECONDS: Sample = 0.05;

/// Measurement- and target-driven inputs to the loudness normalizer.
///
/// All fields are plain values and the struct is [`Copy`]; it is a pure
/// description of the normalization policy, not a processing state.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LoudnessNormalizerParams {
    /// Measured integrated program loudness of the source, in `LUFS`.
    pub measured_lufs: Sample,
    /// Delivery target loudness, in `LUFS`.
    pub target_lufs: Sample,
    /// Measured true-peak level of the source, in `dBTP`.
    pub measured_true_peak_dbtp: Sample,
    /// Delivery true-peak ceiling, in `dBTP`.
    pub max_true_peak_dbtp: Sample,
    /// Maximum absolute makeup gain, in decibels.
    pub max_gain_db: Sample,
}

impl Default for LoudnessNormalizerParams {
    fn default() -> Self {
        // A default-constructed policy is a safe unity pass-through: the
        // measured loudness sits at the silence gate, which disables gain.
        Self {
            measured_lufs: SILENCE_GATE_LUFS,
            target_lufs: DEFAULT_TARGET_LUFS,
            measured_true_peak_dbtp: DEFAULT_MAX_TRUE_PEAK_DBTP,
            max_true_peak_dbtp: DEFAULT_MAX_TRUE_PEAK_DBTP,
            max_gain_db: DEFAULT_MAX_GAIN_DB,
        }
    }
}

/// Computes the static makeup gain, in decibels, implied by `params`.
///
/// Returns `0` (unity) for unmeasurable sources: a non-finite measurement or
/// target, or a measured loudness at or below [`SILENCE_GATE_LUFS`].
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::mastering::{normalization_gain_db, LoudnessNormalizerParams};
///
/// // A mix measured at -20 LUFS, normalized to a -14 LUFS target, needs
/// // +6 dB of makeup gain (its true peak is well under the ceiling).
/// let params = LoudnessNormalizerParams {
///     measured_lufs: -20.0,
///     target_lufs: -14.0,
///     measured_true_peak_dbtp: -12.0,
///     max_true_peak_dbtp: -1.0,
///     max_gain_db: 24.0,
/// };
/// assert!((normalization_gain_db(params) - 6.0).abs() < 1e-5);
/// ```
#[must_use]
pub fn normalization_gain_db(params: LoudnessNormalizerParams) -> Sample {
    let LoudnessNormalizerParams {
        measured_lufs,
        target_lufs,
        measured_true_peak_dbtp,
        max_true_peak_dbtp,
        max_gain_db,
    } = params;

    if !measured_lufs.is_finite() || !target_lufs.is_finite() || measured_lufs <= SILENCE_GATE_LUFS
    {
        return 0.0;
    }

    // Ideal makeup to hit the target loudness.
    let mut gain_db = target_lufs - measured_lufs;

    // True-peak ceiling guard: applying `gain_db` raises the peak to
    // `measured_peak + gain_db`, which must stay at or below the ceiling. This
    // only reduces an upward gain, and forces attenuation when the source
    // already exceeds the ceiling.
    if measured_true_peak_dbtp.is_finite() && max_true_peak_dbtp.is_finite() {
        let true_peak_headroom = max_true_peak_dbtp - measured_true_peak_dbtp;
        gain_db = gain_db.min(true_peak_headroom);
    }

    // Clamp the absolute gain so quiet sources are not pumped into their noise
    // floor and loud ones are not attenuated without bound.
    let limit = if max_gain_db.is_finite() {
        max_gain_db.abs()
    } else {
        Sample::INFINITY
    };
    gain_db.clamp(-limit, limit)
}

/// Applies the loudness-normalization makeup gain to every channel of its
/// single input, writing the result to its single output.
///
/// The gain is derived from [`LoudnessNormalizerParams`] via
/// [`normalization_gain_db`] and glided through a [`Smoothed`] value so that a
/// change of target during playback is click-free. The same per-frame gain is
/// applied to every channel so the stereo or surround image stays coherent.
#[derive(Debug, Clone)]
pub struct LoudnessNormalizerNode {
    params: LoudnessNormalizerParams,
    gain: Smoothed,
    ramp: Ramp,
}

impl LoudnessNormalizerNode {
    /// Creates a normalizer settled at the gain implied by `params`.
    #[must_use]
    pub fn new(sample_rate: u32, params: &LoudnessNormalizerParams) -> Self {
        let params = *params;
        let gain_linear = db_to_linear(normalization_gain_db(params));
        Self {
            params,
            gain: Smoothed::new(gain_linear),
            ramp: Ramp::linear_seconds(DEFAULT_RAMP_SECONDS, sample_rate.max(1)),
        }
    }

    /// Returns the current normalization policy.
    #[must_use]
    pub fn params(&self) -> LoudnessNormalizerParams {
        self.params
    }

    /// Replaces the whole policy and glides toward the new gain.
    pub fn set_params(&mut self, params: &LoudnessNormalizerParams) {
        self.params = *params;
        self.retarget();
    }

    /// Updates the measured integrated loudness and glides toward the new gain.
    pub fn set_measured_lufs(&mut self, measured_lufs: Sample) {
        self.params.measured_lufs = measured_lufs;
        self.retarget();
    }

    /// Updates the delivery target loudness and glides toward the new gain.
    pub fn set_target_lufs(&mut self, target_lufs: Sample) {
        self.params.target_lufs = target_lufs;
        self.retarget();
    }

    /// Updates the measured true-peak and glides toward the new gain.
    pub fn set_measured_true_peak_dbtp(&mut self, measured_true_peak_dbtp: Sample) {
        self.params.measured_true_peak_dbtp = measured_true_peak_dbtp;
        self.retarget();
    }

    /// Updates the delivery true-peak ceiling and glides toward the new gain.
    pub fn set_max_true_peak_dbtp(&mut self, max_true_peak_dbtp: Sample) {
        self.params.max_true_peak_dbtp = max_true_peak_dbtp;
        self.retarget();
    }

    /// Updates the absolute gain limit and glides toward the new gain.
    pub fn set_max_gain_db(&mut self, max_gain_db: Sample) {
        self.params.max_gain_db = max_gain_db;
        self.retarget();
    }

    /// Sets the glide used by subsequent re-targets.
    pub fn set_ramp(&mut self, ramp: Ramp) {
        self.ramp = ramp;
    }

    /// Returns the instantaneous linear gain being applied.
    #[must_use]
    pub fn current_gain_linear(&self) -> Sample {
        self.gain.current()
    }

    /// Returns the instantaneous applied gain in decibels.
    #[must_use]
    pub fn applied_gain_db(&self) -> Sample {
        linear_to_db(self.gain.current())
    }

    fn retarget(&mut self) {
        let gain_linear = db_to_linear(normalization_gain_db(self.params));
        self.gain.set_target(gain_linear, self.ramp);
    }
}

impl AudioNode for LoudnessNormalizerNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels();

        if self.gain.is_settled() {
            let g = self.gain.current();
            for ch in 0..channels {
                let src = input.channel(ch);
                let dst = output.channel_mut(ch);
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = *s * g;
                }
            }
        } else {
            // Replay the identical smoother sequence per channel so the image
            // stays coherent, then commit the advanced state once.
            let start = self.gain;
            for ch in 0..channels {
                let mut g = start;
                let src = input.channel(ch);
                let dst = output.channel_mut(ch);
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = *s * g.next_sample();
                }
                if ch + 1 == channels {
                    self.gain = g;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.gain = Smoothed::new(self.gain.target());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    fn ctx() -> RenderContext {
        RenderContext {
            sample_rate: 48_000,
            frames: 4,
            playhead: 0,
        }
    }

    fn params(measured_lufs: Sample, target_lufs: Sample) -> LoudnessNormalizerParams {
        LoudnessNormalizerParams {
            measured_lufs,
            target_lufs,
            // Keep the true-peak guard out of the way unless a test asks for it.
            measured_true_peak_dbtp: -60.0,
            max_true_peak_dbtp: 0.0,
            max_gain_db: 60.0,
        }
    }

    #[test]
    fn unity_when_measured_equals_target() {
        let g = normalization_gain_db(params(-14.0, -14.0));
        assert!(g.abs() < 1e-6);
    }

    #[test]
    fn attenuates_when_louder_than_target() {
        let g = normalization_gain_db(params(-10.0, -14.0));
        assert!((g - (-4.0)).abs() < 1e-5);
    }

    #[test]
    fn boosts_when_quieter_than_target() {
        let g = normalization_gain_db(params(-20.0, -14.0));
        assert!((g - 6.0).abs() < 1e-5);
    }

    #[test]
    fn true_peak_ceiling_caps_boost() {
        // +16 dB loudness makeup, but the source peaks at -2 dBTP and the
        // ceiling is -1 dBTP, so only +1 dB of headroom is allowed.
        let p = LoudnessNormalizerParams {
            measured_lufs: -30.0,
            target_lufs: -14.0,
            measured_true_peak_dbtp: -2.0,
            max_true_peak_dbtp: -1.0,
            max_gain_db: 60.0,
        };
        assert!((normalization_gain_db(p) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn true_peak_ceiling_can_force_attenuation() {
        // Loudness wants a small boost, but the source already peaks above the
        // ceiling, so the gain must go negative.
        let p = LoudnessNormalizerParams {
            measured_lufs: -16.0,
            target_lufs: -14.0,
            measured_true_peak_dbtp: 0.0,
            max_true_peak_dbtp: -1.0,
            max_gain_db: 60.0,
        };
        assert!((normalization_gain_db(p) - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn max_gain_db_clamps_boost() {
        let p = LoudnessNormalizerParams {
            measured_lufs: -40.0,
            target_lufs: -14.0,
            measured_true_peak_dbtp: Sample::NEG_INFINITY,
            max_true_peak_dbtp: 0.0,
            max_gain_db: 24.0,
        };
        assert!((normalization_gain_db(p) - 24.0).abs() < 1e-5);
    }

    #[test]
    fn silence_gate_disables_normalization() {
        assert_eq!(normalization_gain_db(params(-70.0, -14.0)), 0.0);
        assert_eq!(normalization_gain_db(params(-80.0, -14.0)), 0.0);
    }

    #[test]
    fn non_finite_measured_is_unity() {
        assert_eq!(normalization_gain_db(params(Sample::NAN, -14.0)), 0.0);
        assert_eq!(
            normalization_gain_db(params(Sample::NEG_INFINITY, -14.0)),
            0.0
        );
    }

    #[test]
    fn non_finite_target_is_unity() {
        assert_eq!(normalization_gain_db(params(-20.0, Sample::NAN)), 0.0);
    }

    #[test]
    fn default_params_are_unity() {
        let g = normalization_gain_db(LoudnessNormalizerParams::default());
        assert_eq!(g, 0.0);
    }

    #[test]
    fn node_applies_settled_gain() {
        // -20 LUFS to -14 LUFS target is +6 dB, true peak out of the way.
        let node_params = params(-20.0, -14.0);
        let mut node = LoudnessNormalizerNode::new(48_000, &node_params);
        let expected = db_to_linear(6.0);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        input.channel_mut(0).copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let output = AudioBuffer::new(ChannelLayout::Mono, 4);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(), &mut io);
        for &y in outputs[0].channel(0) {
            assert!((y - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn node_retarget_glides_click_free() {
        let mut node = LoudnessNormalizerNode::new(48_000, &params(-14.0, -14.0));
        node.set_ramp(Ramp::Linear { samples: 4 });
        // Request a boost; the applied gain should climb monotonically.
        node.set_measured_lufs(-20.0);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        input.channel_mut(0).copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let output = AudioBuffer::new(ChannelLayout::Mono, 4);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(), &mut io);
        let out = outputs[0].channel(0);
        assert!(out[0] < out[1] && out[1] < out[2] && out[2] <= out[3]);
    }

    #[test]
    fn reset_snaps_to_target() {
        let mut node = LoudnessNormalizerNode::new(48_000, &params(-14.0, -14.0));
        node.set_ramp(Ramp::Linear { samples: 1000 });
        node.set_measured_lufs(-20.0);
        node.reset();
        assert!(node.gain.is_settled());
        let expected = db_to_linear(6.0);
        assert!((node.current_gain_linear() - expected).abs() < 1e-4);
    }

    #[test]
    fn gain_is_coherent_across_channels() {
        let node_params = params(-20.0, -14.0);
        let mut node = LoudnessNormalizerNode::new(48_000, &node_params);
        let expected = db_to_linear(6.0);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 4);
        input.channel_mut(0).copy_from_slice(&[1.0, 0.5, -0.5, 0.25]);
        input.channel_mut(1).copy_from_slice(&[0.5, 1.0, 0.25, -0.5]);
        let output = AudioBuffer::new(ChannelLayout::Stereo, 4);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(), &mut io);
        let out = &outputs[0];
        assert!((out.channel(0)[0] - 1.0 * expected).abs() < 1e-5);
        assert!((out.channel(1)[1] - 1.0 * expected).abs() < 1e-5);
    }

    #[test]
    fn applied_gain_db_reports_current() {
        let node = LoudnessNormalizerNode::new(48_000, &params(-20.0, -14.0));
        assert!((node.applied_gain_db() - 6.0).abs() < 1e-4);
    }

    #[test]
    fn zero_frames_is_no_op() {
        let mut node = LoudnessNormalizerNode::new(48_000, &params(-20.0, -14.0));
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        input.set_active_frames(0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 4);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(), &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }
}
