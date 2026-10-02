//! Moog-style ladder filter: a four-pole (24 dB/oct) resonant low-pass built
//! from a cascade of one-pole topology-preserving (TPT) integrators closed by a
//! single global feedback loop, with an optional two-pole (12 dB/oct) tap.
//!
//! The transistor-ladder filter is the defining colour of the classic Moog
//! synthesizers: four identical one-pole low-pass stages in series, wrapped by
//! a feedback path whose gain `k` controls resonance. As `k` approaches the
//! stability limit the filter rings sharply at the cutoff and finally
//! self-oscillates. A naive explicit discretisation of that loop introduces a
//! unit-delay in the feedback and detunes / destabilises the resonance; this
//! node instead solves the zero-delay feedback (ZDF) loop in closed form every
//! sample so the resonant peak lands on the requested cutoff and stays bounded
//! under fast modulation.
//!
//! # Signal model
//!
//! Each of the four stages is a one-pole TPT low-pass with bilinear pre-warp
//!
//! ```text
//! g  = tan(pi * fc / fs)      // fc clamped below Nyquist
//! G1 = g / (1 + g)            // per-stage instantaneous gain
//! c  = 1 - G1 = 1 / (1 + g)   // state feed-through
//! G  = G1^4                   // four-stage forward gain
//! ```
//!
//! With per-channel integrator states `z0..z3` and the drive-saturated input
//! `x = tanh(input * drive)`, the state contribution to the fourth-stage output
//! is
//!
//! ```text
//! S  = c * (G1^3 * z0 + G1^2 * z1 + G1 * z2 + z3)
//! ```
//!
//! and the zero-delay feedback loop `u = x - k * y3` is resolved in closed form:
//!
//! ```text
//! y3 = (G * x + S) / (1 + k * G)
//! u  = x - k * y3
//! ```
//!
//! The four one-pole integrators are then advanced from `u`, and the output tap
//! is the fourth-stage output `y3` for a 24 dB/oct `FourPole` response or the
//! second-stage output `y1` for a 12 dB/oct `TwoPole` response. The DC gain
//! falls as `1 / (1 + k)` with increasing resonance, which is the real ladder's
//! characteristic low-end thinning and is kept deliberately rather than
//! compensated.
//!
//! # Relationship
//!
//! This is the four-pole, self-oscillation-capable sibling of the crate's
//! two-pole state-variable filter [`Svf`](crate::nodes::svf::Svf) /
//! [`SvfNode`](crate::nodes::svf::SvfNode). The SVF is a single 12 dB/oct
//! biquad-equivalent section with an independent `Q`; this ladder is a cascade
//! of four one-pole sections with a *global* feedback resonance, giving the
//! steeper 24 dB/oct slope and the ringing / screaming resonance that defines
//! the Moog sound. It shares the TPT one-pole update and the `tan` pre-warp idea
//! with [`Svf`](crate::nodes::svf::Svf) but is a distinct topology, and it is
//! unrelated to the cookbook biquad EQ nodes
//! ([`parametric_eq`](crate::nodes::effects::parametric_eq),
//! [`graphic_eq`](crate::nodes::effects::graphic_eq)).
//!
//! # Real-time contract
//!
//! Per-channel integrator state is allocated once in [`LadderFilterNode::new`].
//! [`LadderFilterNode::process`] performs no allocation, takes no locks, and
//! cannot panic: non-finite inputs are treated as silence, the cutoff is clamped
//! below Nyquist every sample, the linear ZDF loop is stable for the clamped
//! feedback (`k < 4`), and all integrator states are flushed of denormals.
//! Latency is zero.
//!
//! # Provenance
//!
//! Pure classic DSP. The transistor-ladder low-pass and its zero-delay feedback
//! solution are standard, publicly documented material: Vadim Zavalishin, "The
//! Art of VA Filter Design" (the trapezoidal ladder chapter); Stilson & Smith,
//! "Analyzing the Moog VCF with Considerations for Digital Implementation"; and
//! Huovilainen's non-linear ladder studies, from which only the publicly
//! described structural idea is taken. There is no AI/ML of any kind, and no
//! Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance
//! Audio, Web Audio, or STK source or derived code; only the publicly documented
//! difference equations above are implemented.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::buffer::ChannelLayout;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Smallest cutoff frequency (Hz) the node accepts.
pub const MIN_CUTOFF_HZ: Sample = 20.0;

/// Largest cutoff frequency (Hz) the node accepts; the effective ceiling is the
/// smaller of this and the Nyquist guard for the current sample rate.
pub const MAX_CUTOFF_HZ: Sample = 20_000.0;

/// Default cutoff frequency (Hz).
pub const DEFAULT_CUTOFF_HZ: Sample = 1_000.0;

/// Default resonance amount (normalised `0..=1`): a gentle lift just below
/// self-oscillation territory.
pub const DEFAULT_RESONANCE: Sample = 0.1;

/// Feedback gain mapped from a full resonance amount of `1.0`. The ladder
/// self-oscillates at `k = 4`; this sits just below that so the filter can
/// scream without the linear loop diverging in `f32`.
pub const MAX_RESONANCE_K: Sample = 3.98;

/// Smallest input drive the node accepts.
pub const MIN_DRIVE: Sample = 0.1;

/// Largest input drive the node accepts, keeping the `tanh` saturation finite
/// for loud transients.
pub const MAX_DRIVE: Sample = 8.0;

/// Default input drive: small-signal behaviour is essentially transparent.
pub const DEFAULT_DRIVE: Sample = 1.0;

/// Fraction of the sample rate used as the hard cutoff ceiling so the bilinear
/// `tan` pre-warp stays finite and the loop stays stable near Nyquist.
pub const NYQUIST_GUARD: Sample = 0.49;

/// Output slope of the ladder: which tap of the four-stage cascade is returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum LadderSlope {
    /// 12 dB/oct: the second-stage tap (two poles).
    TwoPole,
    /// 24 dB/oct: the fourth-stage tap (four poles), the classic Moog slope.
    #[default]
    FourPole,
}

/// Construction parameters for a [`LadderFilterNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LadderFilterParams {
    /// Cutoff frequency in hertz.
    pub cutoff_hz: Sample,
    /// Resonance amount, normalised `0..=1`; `1.0` maps to [`MAX_RESONANCE_K`].
    pub resonance: Sample,
    /// Input drive fed into the `tanh` saturator before the ladder.
    pub drive: Sample,
    /// Which cascade tap to output (12 or 24 dB/oct).
    pub slope: LadderSlope,
}

impl Default for LadderFilterParams {
    fn default() -> Self {
        Self {
            cutoff_hz: DEFAULT_CUTOFF_HZ,
            resonance: DEFAULT_RESONANCE,
            drive: DEFAULT_DRIVE,
            slope: LadderSlope::FourPole,
        }
    }
}

impl LadderFilterParams {
    /// Returns a copy with every field clamped to its supported range (cutoff
    /// also clamped below Nyquist for `sample_rate`) and any non-finite value
    /// replaced by its default.
    #[must_use]
    pub fn sanitised(self, sample_rate: u32) -> Self {
        let cutoff_hz = if self.cutoff_hz.is_finite() {
            clamp_cutoff(self.cutoff_hz, sample_rate)
        } else {
            clamp_cutoff(DEFAULT_CUTOFF_HZ, sample_rate)
        };
        let resonance = if self.resonance.is_finite() {
            self.resonance.clamp(0.0, 1.0)
        } else {
            DEFAULT_RESONANCE
        };
        let drive = if self.drive.is_finite() {
            self.drive.clamp(MIN_DRIVE, MAX_DRIVE)
        } else {
            DEFAULT_DRIVE
        };
        Self {
            cutoff_hz,
            resonance,
            drive,
            slope: self.slope,
        }
    }
}

/// Clamps a cutoff to `[MIN_CUTOFF_HZ, min(MAX_CUTOFF_HZ, Nyquist guard)]`.
#[inline]
fn clamp_cutoff(hz: Sample, sample_rate: u32) -> Sample {
    let fs = sample_rate.max(1) as Sample;
    let ceil = (NYQUIST_GUARD * fs).clamp(MIN_CUTOFF_HZ, MAX_CUTOFF_HZ);
    hz.clamp(MIN_CUTOFF_HZ, ceil)
}

/// A Moog-style zero-delay-feedback ladder low-pass (input port 0 -> output
/// port 0).
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{LadderFilterNode, LadderFilterParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node =
///     LadderFilterNode::new(48_000, ChannelLayout::Mono, LadderFilterParams::default());
/// let mut inputs = [AudioBuffer::new(ChannelLayout::Mono, 256)];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 256)];
/// inputs[0].set_active_frames(256);
/// outputs[0].set_active_frames(256);
/// for (i, s) in inputs[0].channel_mut(0).iter_mut().enumerate() {
///     *s = if i % 2 == 0 { 0.5 } else { -0.5 };
/// }
/// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
/// assert!(outputs[0].channel(0).iter().all(|s| s.is_finite()));
/// ```
#[derive(Debug)]
pub struct LadderFilterNode {
    /// Sample rate, retained so setters can re-clamp the cutoff below Nyquist.
    sample_rate: u32,
    /// Channel layout reported to the host.
    layout: ChannelLayout,
    /// Number of independently filtered channels.
    channels: usize,
    /// Smoothed cutoff frequency, in hertz.
    cutoff: Smoothed,
    /// Smoothed feedback gain `k` (already in `k` units, not the `0..=1` amount).
    resonance: Smoothed,
    /// Smoothed input drive.
    drive: Smoothed,
    /// Which cascade tap to output.
    slope: LadderSlope,
    /// Per-channel four one-pole integrator states `z0..z3`.
    z: Vec<[Sample; 4]>,
}

impl LadderFilterNode {
    /// Builds a ladder filter for `layout` at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout, params: LadderFilterParams) -> Self {
        let channels = layout.channel_count();
        let p = params.sanitised(sample_rate);
        Self {
            sample_rate,
            layout,
            channels,
            cutoff: Smoothed::new(p.cutoff_hz),
            resonance: Smoothed::new(p.resonance * MAX_RESONANCE_K),
            drive: Smoothed::new(p.drive),
            slope: p.slope,
            z: vec![[0.0; 4]; channels],
        }
    }

    /// Returns the number of channels this node processes.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the channel layout reported to the host.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns the target cutoff frequency, in hertz.
    #[inline]
    #[must_use]
    pub fn cutoff_hz(&self) -> Sample {
        self.cutoff.target()
    }

    /// Returns the target resonance as a normalised `0..=1` amount.
    #[inline]
    #[must_use]
    pub fn resonance(&self) -> Sample {
        self.resonance.target() / MAX_RESONANCE_K
    }

    /// Returns the target input drive (linear).
    #[inline]
    #[must_use]
    pub fn drive(&self) -> Sample {
        self.drive.target()
    }

    /// Returns the current output slope.
    #[inline]
    #[must_use]
    pub fn slope(&self) -> LadderSlope {
        self.slope
    }

    /// Sets the cutoff frequency in hertz using the given ramp. Non-finite
    /// requests are ignored so the previous value is preserved. Control-thread
    /// operation, not called from [`AudioNode::process`].
    pub fn set_cutoff(&mut self, cutoff_hz: Sample, ramp: Ramp) {
        if !cutoff_hz.is_finite() {
            return;
        }
        let hz = clamp_cutoff(cutoff_hz, self.sample_rate);
        self.cutoff.set_target(hz, ramp);
    }

    /// Sets the resonance from a normalised `0..=1` amount using the given ramp.
    /// Non-finite requests are ignored.
    pub fn set_resonance(&mut self, amount: Sample, ramp: Ramp) {
        if !amount.is_finite() {
            return;
        }
        let k = amount.clamp(0.0, 1.0) * MAX_RESONANCE_K;
        self.resonance.set_target(k, ramp);
    }

    /// Sets the input drive (linear) using the given ramp. Non-finite requests
    /// are ignored.
    pub fn set_drive(&mut self, drive: Sample, ramp: Ramp) {
        if !drive.is_finite() {
            return;
        }
        let d = drive.clamp(MIN_DRIVE, MAX_DRIVE);
        self.drive.set_target(d, ramp);
    }

    /// Switches the output slope. Takes effect from the next processed sample.
    pub fn set_slope(&mut self, slope: LadderSlope) {
        self.slope = slope;
    }
}

impl AudioNode for LadderFilterNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let in_channels = input.channels();
        let frames = output.active_frames().min(input.active_frames());
        if frames == 0 || out_channels == 0 {
            return;
        }
        let fs = self.sample_rate.max(1) as Sample;
        let nyquist_ceil = (NYQUIST_GUARD * fs).clamp(MIN_CUTOFF_HZ, MAX_CUTOFF_HZ);
        let state_n = self.channels;
        let slope = self.slope;
        for n in 0..frames {
            let fc = self.cutoff.next_sample().clamp(MIN_CUTOFF_HZ, nyquist_ceil);
            let k = self.resonance.next_sample();
            let drive = self.drive.next_sample();
            let g = ops::tan(core::f32::consts::PI * fc / fs);
            let g1 = g / (1.0 + g);
            let c = 1.0 - g1;
            let g1_2 = g1 * g1;
            let g1_3 = g1_2 * g1;
            let gg = g1_3 * g1;
            let denom = 1.0 + k * gg;
            for ch in 0..out_channels {
                if ch >= state_n || ch >= in_channels {
                    output.channel_mut(ch)[n] = 0.0;
                    continue;
                }
                let raw = input.channel(ch)[n];
                let x_in = if raw.is_finite() { raw } else { 0.0 };
                let x = ops::tanh(x_in * drive);
                let z = &mut self.z[ch];
                let s = c * (g1_3 * z[0] + g1_2 * z[1] + g1 * z[2] + z[3]);
                let y3 = (gg * x + s) / denom;
                let u = x - k * y3;
                let v0 = (u - z[0]) * g1;
                let y0 = v0 + z[0];
                z[0] = flush_denormal(y0 + v0);
                let v1 = (y0 - z[1]) * g1;
                let y1 = v1 + z[1];
                z[1] = flush_denormal(y1 + v1);
                let v2 = (y1 - z[2]) * g1;
                let y2 = v2 + z[2];
                z[2] = flush_denormal(y2 + v2);
                let v3 = (y2 - z[3]) * g1;
                let y3b = v3 + z[3];
                z[3] = flush_denormal(y3b + v3);
                let out = match slope {
                    LadderSlope::FourPole => y3b,
                    LadderSlope::TwoPole => y1,
                };
                output.channel_mut(ch)[n] = flush_denormal(out);
            }
        }
    }

    fn reset(&mut self) {
        for z in &mut self.z {
            *z = [0.0; 4];
        }
        let cutoff = self.cutoff.target();
        self.cutoff.set_target(cutoff, Ramp::Immediate);
        let resonance = self.resonance.target();
        self.resonance.set_target(resonance, Ramp::Immediate);
        let drive = self.drive.target();
        self.drive.set_target(drive, Ramp::Immediate);
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::AudioBuffer;
    use alloc::vec::Vec;
    use core::f32::consts::TAU;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn sine(freq: Sample, len: usize) -> Vec<Sample> {
        (0..len)
            .map(|n| ops::sin(TAU * freq * n as Sample / SR as Sample))
            .collect()
    }

    /// Runs a mono signal through the node and returns the output.
    fn run_mono(node: &mut LadderFilterNode, signal: &[Sample]) -> Vec<Sample> {
        let len = signal.len();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len.max(1));
        let mut output = AudioBuffer::new(ChannelLayout::Mono, len.max(1));
        input.set_active_frames(len);
        output.set_active_frames(len);
        input.channel_mut(0)[..len].copy_from_slice(signal);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        outputs[0].channel(0)[..len].to_vec()
    }

    /// Root-mean-square over the tail of a buffer (skipping the transient).
    fn rms(v: &[Sample], skip: usize) -> Sample {
        let tail = &v[skip.min(v.len())..];
        if tail.is_empty() {
            return 0.0;
        }
        let sum: f64 = tail.iter().map(|&x| (x as f64) * (x as f64)).sum();
        (sum / tail.len() as f64).sqrt() as Sample
    }

    /// Peak absolute amplitude over the tail of a buffer.
    fn tail_peak(v: &[Sample], skip: usize) -> Sample {
        v.iter()
            .skip(skip)
            .fold(0.0_f32, |m, &x| m.max(x.abs()))
    }

    /// Goertzel magnitude at `freq` over the tail of a buffer.
    fn goertzel(v: &[Sample], freq: Sample, skip: usize) -> Sample {
        let tail = &v[skip.min(v.len())..];
        let n = tail.len();
        if n == 0 {
            return 0.0;
        }
        let w = TAU as f64 * freq as f64 / SR as f64;
        let (cw, sw) = (w.cos(), w.sin());
        let coeff = 2.0 * cw;
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &x in tail {
            let s0 = coeff * s1 - s2 + x as f64;
            s2 = s1;
            s1 = s0;
        }
        let re = s1 - s2 * cw;
        let im = s2 * sw;
        ((re * re + im * im).sqrt() / n as f64) as Sample
    }

    #[test]
    fn latency_is_zero() {
        let node = LadderFilterNode::new(SR, ChannelLayout::Stereo, LadderFilterParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn geometry_getters() {
        let node = LadderFilterNode::new(SR, ChannelLayout::Quad, LadderFilterParams::default());
        assert_eq!(node.channels(), 4);
        assert_eq!(node.layout(), ChannelLayout::Quad);
    }

    #[test]
    fn default_params_in_domain() {
        let p = LadderFilterParams::default();
        assert!(p.cutoff_hz >= MIN_CUTOFF_HZ && p.cutoff_hz <= MAX_CUTOFF_HZ);
        assert!((0.0..=1.0).contains(&p.resonance));
        assert!(p.drive >= MIN_DRIVE && p.drive <= MAX_DRIVE);
        assert_eq!(p.slope, LadderSlope::FourPole);
    }

    #[test]
    fn getters_report_configured_values() {
        let params = LadderFilterParams {
            cutoff_hz: 800.0,
            resonance: 0.5,
            drive: 2.0,
            slope: LadderSlope::TwoPole,
        };
        let node = LadderFilterNode::new(SR, ChannelLayout::Mono, params);
        assert!((node.cutoff_hz() - 800.0).abs() < 1e-3);
        assert!((node.resonance() - 0.5).abs() < 1e-4);
        assert!((node.drive() - 2.0).abs() < 1e-4);
        assert_eq!(node.slope(), LadderSlope::TwoPole);
    }

    #[test]
    fn sanitise_clamps_fields() {
        let p = LadderFilterParams {
            cutoff_hz: 5.0,
            resonance: 4.0,
            drive: 100.0,
            slope: LadderSlope::FourPole,
        }
        .sanitised(SR);
        assert!((p.cutoff_hz - MIN_CUTOFF_HZ).abs() < 1e-3);
        assert!((p.resonance - 1.0).abs() < 1e-6);
        assert!((p.drive - MAX_DRIVE).abs() < 1e-6);
    }

    #[test]
    fn sanitise_replaces_non_finite() {
        let p = LadderFilterParams {
            cutoff_hz: Sample::NAN,
            resonance: Sample::INFINITY,
            drive: Sample::NAN,
            slope: LadderSlope::FourPole,
        }
        .sanitised(SR);
        assert!((p.cutoff_hz - DEFAULT_CUTOFF_HZ).abs() < 1e-3);
        assert!((p.resonance - DEFAULT_RESONANCE).abs() < 1e-6);
        assert!((p.drive - DEFAULT_DRIVE).abs() < 1e-6);
    }

    #[test]
    fn cutoff_clamped_below_nyquist_at_low_rate() {
        // At 8 kHz, the Nyquist guard caps the cutoff well below MAX_CUTOFF_HZ.
        let node = LadderFilterNode::new(
            8_000,
            ChannelLayout::Mono,
            LadderFilterParams {
                cutoff_hz: 19_000.0,
                ..LadderFilterParams::default()
            },
        );
        assert!(node.cutoff_hz() <= NYQUIST_GUARD * 8_000.0 + 1.0);
    }

    #[test]
    fn setters_ignore_non_finite() {
        let mut node =
            LadderFilterNode::new(SR, ChannelLayout::Mono, LadderFilterParams::default());
        node.set_cutoff(1_500.0, Ramp::Immediate);
        node.set_resonance(0.3, Ramp::Immediate);
        node.set_drive(2.5, Ramp::Immediate);
        node.set_cutoff(Sample::NAN, Ramp::Immediate);
        node.set_resonance(Sample::INFINITY, Ramp::Immediate);
        node.set_drive(Sample::NAN, Ramp::Immediate);
        assert!((node.cutoff_hz() - 1_500.0).abs() < 1e-3);
        assert!((node.resonance() - 0.3).abs() < 1e-4);
        assert!((node.drive() - 2.5).abs() < 1e-4);
    }

    #[test]
    fn resonance_setter_roundtrips() {
        let mut node =
            LadderFilterNode::new(SR, ChannelLayout::Mono, LadderFilterParams::default());
        node.set_resonance(0.75, Ramp::Immediate);
        assert!((node.resonance() - 0.75).abs() < 1e-4);
    }

    #[test]
    fn lowpass_attenuates_above_cutoff() {
        let params = LadderFilterParams {
            cutoff_hz: 1_000.0,
            resonance: 0.0,
            ..LadderFilterParams::default()
        };
        let mut low = LadderFilterNode::new(SR, ChannelLayout::Mono, params);
        let mut high = LadderFilterNode::new(SR, ChannelLayout::Mono, params);
        let low_rms = rms(&run_mono(&mut low, &sine(100.0, 16_384)), 8_000);
        let high_rms = rms(&run_mono(&mut high, &sine(8_000.0, 16_384)), 8_000);
        assert!(
            high_rms < low_rms * 0.1,
            "8 kHz ({high_rms}) should be far below 100 Hz ({low_rms})"
        );
    }

    #[test]
    fn resonance_boosts_near_cutoff() {
        let tone = sine(1_000.0, 16_384);
        let mut flat = LadderFilterNode::new(
            SR,
            ChannelLayout::Mono,
            LadderFilterParams {
                cutoff_hz: 1_000.0,
                resonance: 0.0,
                ..LadderFilterParams::default()
            },
        );
        let mut peaky = LadderFilterNode::new(
            SR,
            ChannelLayout::Mono,
            LadderFilterParams {
                cutoff_hz: 1_000.0,
                resonance: 0.9,
                ..LadderFilterParams::default()
            },
        );
        let flat_rms = rms(&run_mono(&mut flat, &tone), 8_000);
        let peaky_rms = rms(&run_mono(&mut peaky, &tone), 8_000);
        assert!(
            peaky_rms > flat_rms * 1.5,
            "resonance should lift the cutoff band: flat {flat_rms}, peaky {peaky_rms}"
        );
    }

    #[test]
    fn four_pole_steeper_than_two_pole() {
        let probe = sine(2_000.0, 16_384);
        let mut four = LadderFilterNode::new(
            SR,
            ChannelLayout::Mono,
            LadderFilterParams {
                cutoff_hz: 1_000.0,
                resonance: 0.0,
                slope: LadderSlope::FourPole,
                ..LadderFilterParams::default()
            },
        );
        let mut two = LadderFilterNode::new(
            SR,
            ChannelLayout::Mono,
            LadderFilterParams {
                cutoff_hz: 1_000.0,
                resonance: 0.0,
                slope: LadderSlope::TwoPole,
                ..LadderFilterParams::default()
            },
        );
        let four_rms = rms(&run_mono(&mut four, &probe), 8_000);
        let two_rms = rms(&run_mono(&mut two, &probe), 8_000);
        assert!(
            four_rms < two_rms * 0.7,
            "24 dB/oct ({four_rms}) should roll off faster than 12 dB/oct ({two_rms})"
        );
    }

    #[test]
    fn set_slope_changes_output() {
        let probe = sine(2_000.0, 8_192);
        let base = LadderFilterParams {
            cutoff_hz: 1_000.0,
            resonance: 0.0,
            slope: LadderSlope::FourPole,
            ..LadderFilterParams::default()
        };
        let mut node = LadderFilterNode::new(SR, ChannelLayout::Mono, base);
        let four = rms(&run_mono(&mut node, &probe), 4_000);
        node.reset();
        node.set_slope(LadderSlope::TwoPole);
        let two = rms(&run_mono(&mut node, &probe), 4_000);
        assert!((four - two).abs() > 1e-4, "slope switch should change output");
    }

    #[test]
    fn drive_adds_harmonics() {
        let tone: Vec<Sample> = sine(200.0, 16_384).iter().map(|&x| x * 0.9).collect();
        let base = LadderFilterParams {
            cutoff_hz: 6_000.0,
            resonance: 0.0,
            ..LadderFilterParams::default()
        };
        let mut clean = LadderFilterNode::new(
            SR,
            ChannelLayout::Mono,
            LadderFilterParams {
                drive: 1.0,
                ..base
            },
        );
        let mut dirty = LadderFilterNode::new(
            SR,
            ChannelLayout::Mono,
            LadderFilterParams {
                drive: 8.0,
                ..base
            },
        );
        let clean_h3 = goertzel(&run_mono(&mut clean, &tone), 600.0, 8_000);
        let dirty_h3 = goertzel(&run_mono(&mut dirty, &tone), 600.0, 8_000);
        assert!(
            dirty_h3 > clean_h3 * 5.0,
            "drive should add odd harmonics: clean h3 {clean_h3}, dirty h3 {dirty_h3}"
        );
    }

    #[test]
    fn stable_at_max_resonance() {
        let mut node = LadderFilterNode::new(
            SR,
            ChannelLayout::Mono,
            LadderFilterParams {
                cutoff_hz: 1_000.0,
                resonance: 1.0,
                ..LadderFilterParams::default()
            },
        );
        let mut input = vec![0.0_f32; SR as usize];
        input[0] = 1.0;
        let out = run_mono(&mut node, &input);
        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            tail_peak(&out, 1_000) < 100.0,
            "self-oscillation must stay bounded"
        );
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = LadderFilterNode::new(
            SR,
            ChannelLayout::Mono,
            LadderFilterParams {
                resonance: 0.9,
                ..LadderFilterParams::default()
            },
        );
        let out = run_mono(&mut node, &vec![0.0_f32; 4_096]);
        assert!(tail_peak(&out, 0) < 1e-6, "zero input must give zero output");
    }

    #[test]
    fn deterministic() {
        let probe = sine(440.0, 4_096);
        let mut a = LadderFilterNode::new(SR, ChannelLayout::Mono, LadderFilterParams::default());
        let mut b = LadderFilterNode::new(SR, ChannelLayout::Mono, LadderFilterParams::default());
        assert_eq!(run_mono(&mut a, &probe), run_mono(&mut b, &probe));
    }

    #[test]
    fn reset_clears_state() {
        let probe = sine(440.0, 4_096);
        let mut node = LadderFilterNode::new(SR, ChannelLayout::Mono, LadderFilterParams::default());
        let first = run_mono(&mut node, &probe);
        node.reset();
        let second = run_mono(&mut node, &probe);
        assert_eq!(first, second);
    }

    #[test]
    fn channels_filter_independently() {
        let mut node =
            LadderFilterNode::new(SR, ChannelLayout::Stereo, LadderFilterParams::default());
        let len = 2_048;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        let left = sine(500.0, len);
        input.channel_mut(0).copy_from_slice(&left);
        for s in input.channel_mut(1).iter_mut() {
            *s = 0.0;
        }
        let inputs = [input];
        let mut outputs = [output];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(len), &mut io);
        }
        let r_peak = tail_peak(outputs[0].channel(1), 0);
        let l_peak = tail_peak(outputs[0].channel(0), 1_000);
        assert!(r_peak < 1e-6, "silent channel must stay silent: {r_peak}");
        assert!(l_peak > 0.1, "driven channel must pass signal: {l_peak}");
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = LadderFilterNode::new(SR, ChannelLayout::Mono, LadderFilterParams::default());
        let input = AudioBuffer::new(ChannelLayout::Mono, 8);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 8);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }
}
