//! Stereo correlation and goniometer-style field metering.
//!
//! Mastering and broadcast workflows watch the *stereo image* of a mix as
//! closely as its loudness: a bus that collapses or inverts when summed to
//! mono, drifts off-centre, or widens past the point of mono compatibility is
//! a defect even when its level is correct. This module implements the
//! read-only measurement side of that workflow as a real-time-safe analysis
//! tap over a stereo pair `(L, R)`:
//!
//! - **Correlation** -- the running Pearson correlation coefficient between the
//!   two channels, in `[-1, 1]`. `+1` is a perfectly in-phase (mono) signal,
//!   `0` is fully decorrelated, and `-1` is anti-phase: a signal that cancels
//!   when summed to mono. This is the classic phase-correlation meter.
//! - **Width** -- the fraction of total energy carried by the mid/side `S`
//!   (difference) component, in `[0, 1]`: `0` is a centred mono signal and
//!   values near `1` are a very wide, side-dominated image.
//! - **Balance** -- the left/right energy balance in `[-1, 1]`: `0` is centred,
//!   negative leans left, positive leans right.
//! - **Mid / side levels** -- the `RMS` level of the mid `M` and side `S`
//!   components in `dBFS`, from the energy-preserving transform
//!   `M = (L + R) / sqrt(2)`, `S = (L - R) / sqrt(2)`.
//!
//! # Integration
//!
//! Every statistic is derived from one-pole exponential moving averages of the
//! instantaneous products `L*L`, `R*R`, `L*R`, `M*M`, and `S*S`. The averages
//! share a single smoothing coefficient `a = exp(-1 / (t * sr))` for an
//! integration time `t` (300 ms by default, the usual correlation-meter
//! ballistic). A longer time steadies the reading; a shorter time tracks fast
//! transients.
//!
//! # Real-time contract
//!
//! All state is five scalar accumulators plus the coefficient, fixed at
//! construction. The per-sample hot path
//! ([`CorrelationMeter::process_sample`] and
//! [`CorrelationMeterNode::process`]) does no allocation, takes no locks, and
//! cannot panic. Non-finite inputs are treated as silence so the averages can
//! never be poisoned by `NaN`.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It implements the
//! textbook stereo-field measures (Pearson correlation, mid/side energy split,
//! inter-channel balance) directly. It is pure classic DSP with no AI/ML.
//!
//! # Relationship
//!
//! This is a measurement complement to the loudness meter in
//! [`loudness`](crate::nodes::analysis::loudness): loudness answers "how loud",
//! correlation answers "how wide and mono-safe". The integration coefficient
//! reuses [`time_to_coef`](crate::nodes::dynamics::detector::time_to_coef) from
//! the dynamics detector (reused, not re-implemented); the mid/side transform
//! is the energy-preserving dual of the [`StereoWidthNode`] processing effect.
//!
//! [`StereoWidthNode`]: crate::nodes::effects::stereo_width

use core::f32::consts::FRAC_1_SQRT_2;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, linear_to_db};
use crate::nodes::dynamics::detector::time_to_coef;

/// Default integration time in milliseconds for the running averages.
///
/// 300 ms is the conventional ballistic for a stereo correlation meter: slow
/// enough to be readable, fast enough to follow a mix.
pub const DEFAULT_INTEGRATION_MS: Sample = 300.0;

/// A snapshot of every stereo-field statistic.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CorrelationMeasurement {
    /// Inter-channel correlation coefficient in `[-1, 1]`.
    pub correlation: Sample,
    /// Side-energy fraction (stereo width) in `[0, 1]`.
    pub width: Sample,
    /// Left/right balance in `[-1, 1]` (negative leans left).
    pub balance: Sample,
    /// `RMS` level of the mid component in `dBFS`.
    pub mid_level_db: Sample,
    /// `RMS` level of the side component in `dBFS`.
    pub side_level_db: Sample,
}

/// A real-time-safe stereo correlation and field meter.
///
/// Drive it sample by sample with [`process_sample`](Self::process_sample),
/// then read the statistics with [`measurement`](Self::measurement) (or the
/// individual accessors). [`CorrelationMeterNode`] wraps this as a pass-through
/// graph node.
///
/// ```
/// use prism_audio_core::nodes::analysis::correlation::CorrelationMeter;
/// let mut meter = CorrelationMeter::new(48_000, 50.0);
/// // An identical signal in both channels is perfectly correlated mono.
/// for n in 0..48_000u32 {
///     let phase = core::f32::consts::TAU * 440.0 * (n as f32) / 48_000.0;
///     let x = 0.5 * phase.sin();
///     meter.process_sample(x, x);
/// }
/// let m = meter.measurement();
/// assert!((m.correlation - 1.0).abs() < 1.0e-3);
/// assert!(m.width < 1.0e-3);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CorrelationMeter {
    /// One-pole smoothing coefficient shared by every average.
    coef: Sample,
    /// Exponential moving average of `L * L`.
    ll: Sample,
    /// Exponential moving average of `R * R`.
    rr: Sample,
    /// Exponential moving average of `L * R`.
    lr: Sample,
    /// Exponential moving average of `M * M`.
    mm: Sample,
    /// Exponential moving average of `S * S`.
    ss: Sample,
}

impl CorrelationMeter {
    /// Builds a meter at `sample_rate` with the given integration time.
    ///
    /// A non-positive `integration_ms` yields an instantaneous (unsmoothed)
    /// meter.
    #[must_use]
    pub fn new(sample_rate: u32, integration_ms: Sample) -> Self {
        Self {
            coef: time_to_coef(integration_ms, sample_rate),
            ll: 0.0,
            rr: 0.0,
            lr: 0.0,
            mm: 0.0,
            ss: 0.0,
        }
    }

    /// Builds a meter with the [`DEFAULT_INTEGRATION_MS`] ballistic.
    #[must_use]
    pub fn with_default_ballistic(sample_rate: u32) -> Self {
        Self::new(sample_rate, DEFAULT_INTEGRATION_MS)
    }

    /// The one-pole smoothing coefficient in use.
    #[inline]
    #[must_use]
    pub fn coefficient(&self) -> Sample {
        self.coef
    }

    /// Retunes the integration time, preserving the current averages.
    #[inline]
    pub fn set_integration_ms(&mut self, integration_ms: Sample, sample_rate: u32) {
        self.coef = time_to_coef(integration_ms, sample_rate);
    }

    /// Clears every running average back to silence.
    #[inline]
    pub fn reset(&mut self) {
        self.ll = 0.0;
        self.rr = 0.0;
        self.lr = 0.0;
        self.mm = 0.0;
        self.ss = 0.0;
    }

    /// Updates the running averages with one stereo sample pair.
    ///
    /// Non-finite inputs are coerced to zero so the averages cannot be
    /// poisoned.
    #[inline]
    pub fn process_sample(&mut self, left: Sample, right: Sample) {
        let l = if left.is_finite() { left } else { 0.0 };
        let r = if right.is_finite() { right } else { 0.0 };
        let mid = (l + r) * FRAC_1_SQRT_2;
        let side = (l - r) * FRAC_1_SQRT_2;

        let keep = self.coef;
        let take = 1.0 - keep;
        self.ll = flush_denormal(keep * self.ll + take * (l * l));
        self.rr = flush_denormal(keep * self.rr + take * (r * r));
        self.lr = flush_denormal(keep * self.lr + take * (l * r));
        self.mm = flush_denormal(keep * self.mm + take * (mid * mid));
        self.ss = flush_denormal(keep * self.ss + take * (side * side));
    }

    /// The running inter-channel correlation coefficient in `[-1, 1]`.
    ///
    /// Returns `0` when both channels are effectively silent (correlation is
    /// undefined for a zero signal).
    #[inline]
    #[must_use]
    pub fn correlation(&self) -> Sample {
        let denom = ops::sqrt(self.ll * self.rr);
        if denom > 0.0 {
            (self.lr / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// The side-energy fraction (stereo width) in `[0, 1]`.
    ///
    /// `0` is a centred mono signal; values near `1` are side-dominated.
    #[inline]
    #[must_use]
    pub fn width(&self) -> Sample {
        let total = self.mm + self.ss;
        if total > 0.0 {
            (self.ss / total).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// The left/right energy balance in `[-1, 1]` (negative leans left).
    #[inline]
    #[must_use]
    pub fn balance(&self) -> Sample {
        let total = self.ll + self.rr;
        if total > 0.0 {
            ((self.rr - self.ll) / total).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// The `RMS` level of the mid component in `dBFS`.
    #[inline]
    #[must_use]
    pub fn mid_level_db(&self) -> Sample {
        linear_to_db(ops::sqrt(self.mm))
    }

    /// The `RMS` level of the side component in `dBFS`.
    #[inline]
    #[must_use]
    pub fn side_level_db(&self) -> Sample {
        linear_to_db(ops::sqrt(self.ss))
    }

    /// A full snapshot of every statistic.
    #[inline]
    #[must_use]
    pub fn measurement(&self) -> CorrelationMeasurement {
        CorrelationMeasurement {
            correlation: self.correlation(),
            width: self.width(),
            balance: self.balance(),
            mid_level_db: self.mid_level_db(),
            side_level_db: self.side_level_db(),
        }
    }
}

/// A pass-through [`AudioNode`] that taps a stereo bus for correlation metering.
///
/// The node copies its input to its output unchanged and measures channels 0
/// (left) and 1 (right). Read the statistics from the embedded
/// [`CorrelationMeter`] via [`meter`](Self::meter).
#[derive(Debug, Clone)]
pub struct CorrelationMeterNode {
    /// The embedded meter.
    meter: CorrelationMeter,
}

impl CorrelationMeterNode {
    /// Builds a metering node at `sample_rate` with the given integration time.
    #[must_use]
    pub fn new(sample_rate: u32, integration_ms: Sample) -> Self {
        Self {
            meter: CorrelationMeter::new(sample_rate, integration_ms),
        }
    }

    /// Builds a metering node with the [`DEFAULT_INTEGRATION_MS`] ballistic.
    #[must_use]
    pub fn with_default_ballistic(sample_rate: u32) -> Self {
        Self {
            meter: CorrelationMeter::with_default_ballistic(sample_rate),
        }
    }

    /// Immutable access to the underlying meter (to read statistics).
    #[inline]
    #[must_use]
    pub fn meter(&self) -> &CorrelationMeter {
        &self.meter
    }

    /// A full snapshot of every statistic.
    #[inline]
    #[must_use]
    pub fn measurement(&self) -> CorrelationMeasurement {
        self.meter.measurement()
    }
}

impl AudioNode for CorrelationMeterNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let frames = output.active_frames();

        // Pass the signal through unchanged.
        let copy_channels = out_channels.min(input.channels());
        for ch in 0..copy_channels {
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            dst[..frames].copy_from_slice(&src[..frames]);
        }

        // Measure the stereo pair. A mono input is read as a centred signal.
        if input.channels() >= 2 {
            let left = input.channel(0);
            let right = input.channel(1);
            for n in 0..frames {
                self.meter.process_sample(left[n], right[n]);
            }
        } else if input.channels() == 1 {
            let mono = input.channel(0);
            for &x in &mono[..frames] {
                self.meter.process_sample(x, x);
            }
        }
    }

    fn reset(&mut self) {
        self.meter.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use alloc::vec::Vec;

    const SR: u32 = 48_000;

    fn settle_mono(meter: &mut CorrelationMeter, amp: Sample, freq: Sample, seconds: Sample) {
        let total = (seconds * SR as Sample) as usize;
        for n in 0..total {
            let phase = core::f32::consts::TAU * freq * (n as Sample) / (SR as Sample);
            let x = amp * ops::sin(phase);
            meter.process_sample(x, x);
        }
    }

    fn settle_pair(
        meter: &mut CorrelationMeter,
        left: impl Fn(usize) -> Sample,
        right: impl Fn(usize) -> Sample,
        seconds: Sample,
    ) {
        let total = (seconds * SR as Sample) as usize;
        for n in 0..total {
            meter.process_sample(left(n), right(n));
        }
    }

    #[test]
    fn identical_channels_are_fully_correlated() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        settle_mono(&mut meter, 0.5, 440.0, 1.0);
        assert!((meter.correlation() - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn inverted_channels_are_anti_correlated() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        settle_pair(
            &mut meter,
            |n| 0.5 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample)),
            |n| -0.5 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample)),
            1.0,
        );
        assert!((meter.correlation() + 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn independent_tones_are_near_decorrelated() {
        let mut meter = CorrelationMeter::new(SR, 200.0);
        // Two incommensurate frequencies integrate to near-zero cross power.
        settle_pair(
            &mut meter,
            |n| 0.5 * ops::sin(core::f32::consts::TAU * 300.0 * (n as Sample) / (SR as Sample)),
            |n| 0.5 * ops::sin(core::f32::consts::TAU * 517.0 * (n as Sample) / (SR as Sample)),
            2.0,
        );
        assert!(meter.correlation().abs() < 0.1);
    }

    #[test]
    fn silence_reports_zero_correlation() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        for _ in 0..SR {
            meter.process_sample(0.0, 0.0);
        }
        assert_eq!(meter.correlation(), 0.0);
        assert_eq!(meter.width(), 0.0);
        assert_eq!(meter.balance(), 0.0);
    }

    #[test]
    fn mono_signal_has_zero_width() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        settle_mono(&mut meter, 0.5, 440.0, 1.0);
        assert!(meter.width() < 1.0e-3);
    }

    #[test]
    fn anti_phase_signal_is_all_side_energy() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        settle_pair(
            &mut meter,
            |n| 0.5 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample)),
            |n| -0.5 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample)),
            1.0,
        );
        assert!(meter.width() > 0.99);
    }

    #[test]
    fn width_is_between_zero_and_one() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        settle_pair(
            &mut meter,
            |n| 0.5 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample)),
            |n| 0.2 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample)),
            1.0,
        );
        let w = meter.width();
        assert!((0.0..=1.0).contains(&w));
    }

    #[test]
    fn balance_leans_toward_louder_channel() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        // Right channel louder -> positive balance.
        settle_pair(
            &mut meter,
            |n| 0.2 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample)),
            |n| 0.8 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample)),
            1.0,
        );
        assert!(meter.balance() > 0.3);
    }

    #[test]
    fn balance_is_negative_for_left_heavy_signal() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        settle_pair(
            &mut meter,
            |n| 0.8 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample)),
            |n| 0.2 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample)),
            1.0,
        );
        assert!(meter.balance() < -0.3);
    }

    #[test]
    fn centred_signal_has_zero_balance() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        settle_mono(&mut meter, 0.5, 440.0, 1.0);
        assert!(meter.balance().abs() < 1.0e-3);
    }

    #[test]
    fn mid_level_tracks_mono_amplitude() {
        let mut meter = CorrelationMeter::new(SR, 100.0);
        // Mono sine of amplitude 0.5 -> mid RMS ~ 0.5/sqrt(2) * sqrt(2) = 0.5/sqrt(2)*...
        // mid = (L+R)/sqrt(2) = 2x/sqrt(2) = x*sqrt(2); RMS of x*sqrt(2) sine
        // with x amp 0.5 is 0.5*sqrt(2)/sqrt(2) = 0.5. So mid level ~ -6 dBFS.
        settle_mono(&mut meter, 0.5, 440.0, 1.0);
        let db = meter.mid_level_db();
        assert!((db - linear_to_db(0.5)).abs() < 0.5);
    }

    #[test]
    fn side_level_is_very_low_for_mono() {
        let mut meter = CorrelationMeter::new(SR, 100.0);
        settle_mono(&mut meter, 0.5, 440.0, 1.0);
        assert!(meter.side_level_db() < -60.0);
    }

    #[test]
    fn non_finite_input_does_not_poison_averages() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        settle_mono(&mut meter, 0.5, 440.0, 0.5);
        meter.process_sample(Sample::NAN, Sample::INFINITY);
        let m = meter.measurement();
        assert!(m.correlation.is_finite());
        assert!(m.width.is_finite());
        assert!(m.balance.is_finite());
    }

    #[test]
    fn reset_clears_state() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        settle_mono(&mut meter, 0.5, 440.0, 0.5);
        meter.reset();
        assert_eq!(meter.correlation(), 0.0);
        assert_eq!(meter.width(), 0.0);
        assert_eq!(meter.mid_level_db(), f32::NEG_INFINITY);
    }

    #[test]
    fn zero_integration_is_instantaneous() {
        let meter = CorrelationMeter::new(SR, 0.0);
        assert_eq!(meter.coefficient(), 0.0);
    }

    #[test]
    fn set_integration_changes_coefficient() {
        let mut meter = CorrelationMeter::new(SR, 50.0);
        let fast = meter.coefficient();
        meter.set_integration_ms(500.0, SR);
        assert!(meter.coefficient() > fast);
    }

    #[test]
    fn node_passes_signal_through_unchanged() {
        let mut node = CorrelationMeterNode::new(SR, 50.0);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 64);
        for n in 0..64 {
            let x = 0.3 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample));
            input.channel_mut(0)[n] = x;
            input.channel_mut(1)[n] = -x;
        }
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 64);
        let inputs = [input.clone()];
        let mut outputs = [output];
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 64,
            playhead: 0,
        };
        let mut pio = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut pio);
        output = outputs.into_iter().next().unwrap();
        for n in 0..64 {
            assert_eq!(output.channel(0)[n], input.channel(0)[n]);
            assert_eq!(output.channel(1)[n], input.channel(1)[n]);
        }
    }

    #[test]
    fn node_measures_anti_phase_width() {
        let mut node = CorrelationMeterNode::new(SR, 20.0);
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 128,
            playhead: 0,
        };
        for _block in 0..400 {
            let mut input = AudioBuffer::new(ChannelLayout::Stereo, 128);
            for n in 0..128 {
                let x =
                    0.4 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample));
                input.channel_mut(0)[n] = x;
                input.channel_mut(1)[n] = -x;
            }
            let inputs = [input];
            let mut outputs = [AudioBuffer::new(ChannelLayout::Stereo, 128)];
            let mut pio = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut pio);
        }
        let m = node.measurement();
        assert!(m.correlation < -0.9);
        assert!(m.width > 0.9);
    }

    #[test]
    fn mono_input_to_node_reads_centred() {
        let mut node = CorrelationMeterNode::new(SR, 20.0);
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 128,
            playhead: 0,
        };
        for _block in 0..400 {
            let mut input = AudioBuffer::new(ChannelLayout::Mono, 128);
            for n in 0..128 {
                input.channel_mut(0)[n] =
                    0.4 * ops::sin(core::f32::consts::TAU * 440.0 * (n as Sample) / (SR as Sample));
            }
            let inputs = [input];
            let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 128)];
            let mut pio = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut pio);
        }
        let m = node.measurement();
        assert!(m.width < 1.0e-3);
        assert!(m.balance.abs() < 1.0e-3);
    }

    #[test]
    fn measurement_collects_into_vec() {
        let mut meters: Vec<CorrelationMeter> = Vec::new();
        meters.push(CorrelationMeter::new(SR, 50.0));
        meters.push(CorrelationMeter::with_default_ballistic(SR));
        for meter in meters.iter_mut() {
            settle_mono(meter, 0.5, 440.0, 0.2);
        }
        let readings: Vec<Sample> = meters.iter().map(CorrelationMeter::correlation).collect();
        assert_eq!(readings.len(), 2);
    }
}
