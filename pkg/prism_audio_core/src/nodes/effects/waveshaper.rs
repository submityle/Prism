//! Oversampled non-linear waveshaper (soft-clipping distortion / saturation).
//!
//! A waveshaper pushes its input through a static non-linear transfer function.
//! Here the shape is the classic hyperbolic-tangent soft clipper
//! `y = tanh(drive * x)`: small signals pass almost linearly while large ones
//! are smoothly compressed toward `±1`, giving the warm, harmonically rich
//! saturation used on drums, bass, and mix buses.
//!
//! # Why oversample
//!
//! Any non-linearity generates harmonics above the input frequency. When those
//! harmonics exceed the Nyquist limit they *fold back* into the audible band as
//! inharmonic aliasing. [`WaveshaperNode`] therefore runs the non-linearity at
//! an integer multiple of the host sample rate (see [`Oversample`]):
//!
//! 1. **Upsample** by inserting `factor - 1` zeros between input samples and
//!    reconstructing with a polyphase windowed-sinc interpolation filter.
//! 2. **Shape** every oversampled sample with `bevy_math::ops::tanh`.
//! 3. **Decimate** back to the host rate through the same half-band-style
//!    low-pass so the freshly created out-of-band harmonics are removed *before*
//!    they can alias.
//!
//! The linear-phase FIR pair introduces a fixed group delay reported by
//! [`AudioNode::latency_frames`]; the internal dry path is delayed by the same
//! amount so the wet/dry blend stays phase-coherent at the node output.
//!
//! All filter state (per-channel interpolation and decimation histories, plus
//! the dry delay line) is pre-allocated at construction, so
//! [`WaveshaperNode::process`] performs no allocation and cannot panic — it is
//! real-time safe.

use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::PI;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Number of filter taps allocated to each polyphase branch of the
/// oversampler. The full prototype filter length is `TAPS_PER_PHASE * factor`,
/// trading a little latency and CPU for strong stop-band rejection.
const TAPS_PER_PHASE: usize = 16;

/// The internal processing rate relative to the host sample rate.
///
/// A higher factor pushes the aliasing products further above Nyquist before
/// they are filtered out, at the cost of proportionally more CPU and a slightly
/// longer filter latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Oversample {
    /// No oversampling: shape directly at the host rate (zero latency, cheapest,
    /// but aliases the most).
    X1,
    /// 2x oversampling: shape at twice the host rate.
    X2,
    /// 4x oversampling: shape at four times the host rate (cleanest, dearest).
    X4,
}

impl Oversample {
    /// Returns the integer oversampling factor (`1`, `2`, or `4`).
    #[inline]
    #[must_use]
    pub const fn factor(self) -> usize {
        match self {
            Oversample::X1 => 1,
            Oversample::X2 => 2,
            Oversample::X4 => 4,
        }
    }
}

/// Per-channel filter memory for the oversampler and the dry-path delay.
///
/// Each channel owns an independent set of histories so a stereo (or surround)
/// image is processed without cross-channel bleed.
#[derive(Debug, Clone)]
struct ChannelState {
    /// Ring buffer of the most recent input samples feeding the polyphase
    /// interpolation filter (length [`TAPS_PER_PHASE`], empty when not
    /// oversampling).
    up_hist: Vec<Sample>,
    /// Write cursor into [`ChannelState::up_hist`].
    up_pos: usize,
    /// Ring buffer of the most recent oversampled, shaped samples feeding the
    /// decimation filter (length `TAPS_PER_PHASE * factor`, empty when not
    /// oversampling).
    down_hist: Vec<Sample>,
    /// Write cursor into [`ChannelState::down_hist`].
    down_pos: usize,
    /// Delay line that aligns the dry signal with the latency of the wet
    /// (oversampled) path. Length equals the reported latency in frames (empty
    /// at zero latency).
    dry_delay: Vec<Sample>,
    /// Write cursor into [`ChannelState::dry_delay`].
    dry_pos: usize,
}

impl ChannelState {
    /// Allocates zeroed histories of the given lengths.
    fn new(up_len: usize, down_len: usize, dry_len: usize) -> Self {
        Self {
            up_hist: zeroed(up_len),
            up_pos: 0,
            down_hist: zeroed(down_len),
            down_pos: 0,
            dry_delay: zeroed(dry_len),
            dry_pos: 0,
        }
    }

    /// Clears every history back to silence.
    fn reset(&mut self) {
        for s in &mut self.up_hist {
            *s = 0.0;
        }
        for s in &mut self.down_hist {
            *s = 0.0;
        }
        for s in &mut self.dry_delay {
            *s = 0.0;
        }
        self.up_pos = 0;
        self.down_pos = 0;
        self.dry_pos = 0;
    }

    /// Shapes one input sample through the oversampled, anti-aliased path and
    /// returns the resulting host-rate sample.
    ///
    /// `phases` are the polyphase interpolation branches and `proto` the shared
    /// decimation prototype; both are borrowed so the whole node keeps a single
    /// copy of its coefficients.
    fn shape_oversampled(
        &mut self,
        x: Sample,
        drive: Sample,
        phases: &[Vec<Sample>],
        proto: &[Sample],
    ) -> Sample {
        // Advance the interpolation input history.
        let up_len = self.up_hist.len();
        self.up_pos = (self.up_pos + 1) % up_len;
        self.up_hist[self.up_pos] = x;

        let down_len = self.down_hist.len();
        // Synthesise `factor` oversampled samples, shape each, and push them
        // into the decimation history in temporal order.
        for phase in phases {
            let mut acc = 0.0;
            for (j, &coeff) in phase.iter().enumerate() {
                let idx = (self.up_pos + up_len - j) % up_len;
                acc += coeff * self.up_hist[idx];
            }
            let shaped = ops::tanh(drive * acc);
            self.down_pos = (self.down_pos + 1) % down_len;
            self.down_hist[self.down_pos] = shaped;
        }

        // Decimate: one host-rate output per `factor` oversampled samples.
        let mut out = 0.0;
        for (j, &coeff) in proto.iter().enumerate() {
            let idx = (self.down_pos + down_len - j) % down_len;
            out += coeff * self.down_hist[idx];
        }
        flush_denormal(out)
    }

    /// Returns `x` delayed by the dry-path delay length, keeping the dry signal
    /// aligned with the latency of the wet path.
    fn dry_delayed(&mut self, x: Sample) -> Sample {
        let len = self.dry_delay.len();
        if len == 0 {
            return x;
        }
        let out = self.dry_delay[self.dry_pos];
        self.dry_delay[self.dry_pos] = x;
        self.dry_pos = (self.dry_pos + 1) % len;
        out
    }
}

/// A soft-clipping waveshaper with selectable anti-aliasing oversampling
/// (input port 0 -> output port 0).
///
/// The transfer function is `tanh(drive * x)`, followed by a make-up
/// `output_gain` and a `wet`/`dry` blend. All four controls are
/// [`Smoothed`](crate::param::Smoothed) so automation never clicks.
#[derive(Debug, Clone)]
pub struct WaveshaperNode {
    /// Selected internal processing rate.
    oversample: Oversample,
    /// Pre-shaping drive gain (the `tanh` input scale). Higher drives clip
    /// harder and generate more harmonics.
    drive: Smoothed,
    /// Post-shaping make-up gain applied to the wet signal.
    output_gain: Smoothed,
    /// Wet (processed) mix coefficient.
    wet: Smoothed,
    /// Dry (unprocessed, latency-aligned) mix coefficient.
    dry: Smoothed,
    /// Decimation prototype filter (unity DC gain). Empty when not oversampling.
    proto: Vec<Sample>,
    /// Polyphase interpolation branches derived from [`WaveshaperNode::proto`].
    /// Empty when not oversampling.
    phases: Vec<Vec<Sample>>,
    /// Per-channel filter state.
    channels: Vec<ChannelState>,
    /// Reported processing latency in host-rate frames.
    latency: u32,
    /// Length of the prototype FIR filter (`0` when not oversampling).
    filter_taps: usize,
}

impl WaveshaperNode {
    /// Builds a waveshaper for a `channels`-wide signal.
    ///
    /// `drive`, `output_gain`, `wet`, and `dry` are the initial (settled)
    /// control values. `sample_rate` is accepted for API symmetry with the
    /// other nodes; the oversampling filters are designed in normalised
    /// frequency and therefore do not depend on it.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        oversample: Oversample,
        drive: Sample,
        output_gain: Sample,
        wet: Sample,
        dry: Sample,
    ) -> Self {
        let _ = sample_rate;
        let factor = oversample.factor();

        let (proto, phases, filter_taps) = if factor > 1 {
            let taps = TAPS_PER_PHASE * factor;
            let proto = design_lowpass(taps, factor);
            let mut phases = Vec::with_capacity(factor);
            for p in 0..factor {
                let mut phase = Vec::with_capacity(TAPS_PER_PHASE);
                let mut k = 0usize;
                while p + k * factor < taps {
                    // Scale by `factor` to compensate for the energy lost to
                    // zero-stuffing during interpolation.
                    phase.push(factor as Sample * proto[p + k * factor]);
                    k += 1;
                }
                phases.push(phase);
            }
            (proto, phases, taps)
        } else {
            (Vec::new(), Vec::new(), 0)
        };

        // Linear-phase group delay of the up/down filter pair is `taps - 1`
        // samples at the oversampled rate, i.e. `(taps - 1) / factor` frames.
        let latency = if factor > 1 {
            ops::round((filter_taps as Sample - 1.0) / factor as Sample) as u32
        } else {
            0
        };

        let up_len = if factor > 1 { TAPS_PER_PHASE } else { 0 };
        let mut states = Vec::with_capacity(channels);
        for _ in 0..channels {
            states.push(ChannelState::new(up_len, filter_taps, latency as usize));
        }

        Self {
            oversample,
            drive: Smoothed::new(drive),
            output_gain: Smoothed::new(output_gain),
            wet: Smoothed::new(wet),
            dry: Smoothed::new(dry),
            proto,
            phases,
            channels: states,
            latency,
            filter_taps,
        }
    }

    /// Sets a new drive gain, gliding toward it with `ramp`.
    #[inline]
    pub fn set_drive(&mut self, target: Sample, ramp: Ramp) {
        self.drive.set_target(target, ramp);
    }

    /// Sets a new post-shaping make-up gain, gliding toward it with `ramp`.
    #[inline]
    pub fn set_output_gain(&mut self, target: Sample, ramp: Ramp) {
        self.output_gain.set_target(target, ramp);
    }

    /// Sets a new wet-mix coefficient, gliding toward it with `ramp`.
    #[inline]
    pub fn set_wet(&mut self, target: Sample, ramp: Ramp) {
        self.wet.set_target(target, ramp);
    }

    /// Sets a new dry-mix coefficient, gliding toward it with `ramp`.
    #[inline]
    pub fn set_dry(&mut self, target: Sample, ramp: Ramp) {
        self.dry.set_target(target, ramp);
    }

    /// Returns the configured oversampling mode.
    #[inline]
    #[must_use]
    pub fn oversample(&self) -> Oversample {
        self.oversample
    }

    /// Returns the length of the oversampling prototype FIR filter, or `0` when
    /// not oversampling. Useful for verifying the reported latency.
    #[inline]
    #[must_use]
    pub fn filter_taps(&self) -> usize {
        self.filter_taps
    }
}

impl AudioNode for WaveshaperNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels()).min(self.channels.len());
        let frames = output.active_frames().min(input.active_frames());
        let factor = self.oversample.factor();

        // Snapshot the smoothers so every channel replays the identical
        // per-sample control trajectory (see `GainNode` for the pattern).
        let drive0 = self.drive;
        let gain0 = self.output_gain;
        let wet0 = self.wet;
        let dry0 = self.dry;

        // Disjoint field borrows: coefficients (shared) and per-channel state
        // (mutable) live in different fields of `self`.
        let proto = &self.proto;
        let phases = &self.phases;
        let states = &mut self.channels;

        let mut committed: Option<(Smoothed, Smoothed, Smoothed, Smoothed)> = None;

        for ch in 0..channels {
            let mut drive = drive0;
            let mut gain = gain0;
            let mut wet = wet0;
            let mut dry = dry0;

            let state = &mut states[ch];
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);

            for i in 0..frames {
                let x = src[i];
                let dv = drive.next_sample();
                let gv = gain.next_sample();
                let wv = wet.next_sample();
                let drv = dry.next_sample();

                let shaped = if factor > 1 {
                    state.shape_oversampled(x, dv, phases, proto)
                } else {
                    flush_denormal(ops::tanh(dv * x))
                };
                let dry_sig = state.dry_delayed(x);
                dst[i] = flush_denormal(wv * (shaped * gv) + drv * dry_sig);
            }

            if ch + 1 == channels {
                committed = Some((drive, gain, wet, dry));
            }
        }

        if let Some((drive, gain, wet, dry)) = committed {
            self.drive = drive;
            self.output_gain = gain;
            self.wet = wet;
            self.dry = dry;
        }
    }

    fn reset(&mut self) {
        for state in &mut self.channels {
            state.reset();
        }
        self.drive = Smoothed::new(self.drive.target());
        self.output_gain = Smoothed::new(self.output_gain.target());
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
    }

    fn latency_frames(&self) -> u32 {
        self.latency
    }
}

/// Allocates a zero-filled sample vector of length `n`.
fn zeroed(n: usize) -> Vec<Sample> {
    let mut v = Vec::with_capacity(n);
    v.resize(n, 0.0);
    v
}

/// Designs a `taps`-long, unity-DC-gain, Hann-windowed sinc low-pass prototype.
///
/// The cutoff is placed at the host-rate Nyquist, i.e. a normalised cutoff of
/// `1/(2*factor)` cycles/sample at the oversampled rate, so the filter both
/// reconstructs interpolated samples and rejects the images/aliases created by
/// the non-linearity.
fn design_lowpass(taps: usize, factor: usize) -> Vec<Sample> {
    let center = (taps - 1) as Sample / 2.0;
    // `cutoff` is twice the cutoff frequency (the sinc argument scale):
    // 2 * (1 / (2 * factor)) = 1 / factor.
    let cutoff = 1.0 / factor as Sample;
    let denom = (taps - 1) as Sample;

    let mut h = Vec::with_capacity(taps);
    let mut sum = 0.0;
    for n in 0..taps {
        let t = n as Sample - center;
        let ideal = if t == 0.0 {
            cutoff
        } else {
            let arg = PI * cutoff * t;
            cutoff * (ops::sin(arg) / arg)
        };
        // Hann window keeps the stop-band clean with a gentle main-lobe trade.
        let window = 0.5 - 0.5 * ops::cos(2.0 * PI * n as Sample / denom);
        let coeff = ideal * window;
        h.push(coeff);
        sum += coeff;
    }

    // Normalise to unity DC gain so the decimation stage preserves level.
    let inv = if sum != 0.0 { 1.0 / sum } else { 1.0 };
    for c in &mut h {
        *c *= inv;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    /// Single-bin DFT magnitude (normalised) at frequency `f`, using the
    /// deterministic `ops` transcendental functions.
    fn bin_magnitude(signal: &[Sample], sample_rate: u32, f: Sample) -> Sample {
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, &s) in signal.iter().enumerate() {
            let phase = 2.0 * PI * f * (i as Sample) / (sample_rate as Sample);
            re += s * ops::cos(phase);
            im += s * ops::sin(phase);
        }
        ops::sqrt(re * re + im * im) / (signal.len() as Sample)
    }

    #[test]
    fn x1_small_signal_is_near_unity() {
        let sr = 48_000u32;
        let n = 64;
        let mut node = WaveshaperNode::new(sr, 1, Oversample::X1, 1.0, 1.0, 1.0, 0.0);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            let t = i as Sample / sr as Sample;
            *s = 0.01 * ops::sin(2.0 * PI * 1_000.0 * t);
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sr, n), &mut io);
        for (o, x) in outputs[0].channel(0).iter().zip(inputs[0].channel(0)) {
            assert!((o - x).abs() < 1.0e-3, "not near-unity: {o} vs {x}");
        }
    }

    #[test]
    fn large_input_is_bounded_by_tanh() {
        let mut node = WaveshaperNode::new(48_000, 1, Oversample::X1, 4.0, 1.0, 1.0, 0.0);
        let n = 32;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = if i % 2 == 0 { 10.0 } else { -10.0 };
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(48_000, n), &mut io);
        for &o in outputs[0].channel(0) {
            assert!(o.abs() < 1.0 + 1.0e-6, "output not bounded by tanh: {o}");
        }
    }

    #[test]
    fn dry_signal_passthrough() {
        // wet = 0, dry = 1, no oversampling => output is bit-exact input.
        let mut node = WaveshaperNode::new(48_000, 1, Oversample::X1, 3.0, 2.0, 0.0, 1.0);
        let n = 48;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = (i as Sample) * 0.1 - 2.0;
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(48_000, n), &mut io);
        assert_eq!(outputs[0].channel(0), inputs[0].channel(0));
    }

    fn alias_magnitude(os: Oversample, sample_rate: u32, n: usize, f0: Sample) -> Sample {
        let mut node = WaveshaperNode::new(sample_rate, 1, os, 8.0, 1.0, 1.0, 0.0);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            let t = i as Sample / sample_rate as Sample;
            *s = 0.8 * ops::sin(2.0 * PI * f0 * t);
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, n), &mut io);
        // The 7th harmonic of 7 kHz (49 kHz) folds back to |49k - 48k| = 1 kHz,
        // an inharmonic alias that oversampling should suppress.
        bin_magnitude(outputs[0].channel(0), sample_rate, 1_000.0)
    }

    #[test]
    fn oversampling_reduces_aliasing() {
        let sr = 48_000u32;
        let n = 4_800usize;
        let f0 = 7_000.0;
        let alias_x1 = alias_magnitude(Oversample::X1, sr, n, f0);
        let alias_x4 = alias_magnitude(Oversample::X4, sr, n, f0);
        assert!(
            alias_x4 < alias_x1 * 0.5,
            "oversampling did not reduce aliasing: x1={alias_x1} x4={alias_x4}"
        );
    }

    #[test]
    fn latency_matches_filter_design() {
        let sr = 48_000u32;
        assert_eq!(
            WaveshaperNode::new(sr, 1, Oversample::X1, 1.0, 1.0, 1.0, 0.0).latency_frames(),
            0
        );

        let n2 = WaveshaperNode::new(sr, 1, Oversample::X2, 1.0, 1.0, 1.0, 0.0);
        let m2 = n2.filter_taps();
        let expected2 = ops::round((m2 as Sample - 1.0) / 2.0) as u32;
        assert_eq!(n2.latency_frames(), expected2);

        let n4 = WaveshaperNode::new(sr, 1, Oversample::X4, 1.0, 1.0, 1.0, 0.0);
        let m4 = n4.filter_taps();
        let expected4 = ops::round((m4 as Sample - 1.0) / 4.0) as u32;
        assert_eq!(n4.latency_frames(), expected4);

        assert!(m2 > 0 && m4 > 0, "oversampled modes must design a filter");
    }

    #[test]
    fn reset_clears_history() {
        let sr = 48_000u32;
        let n = 64;
        let mut node = WaveshaperNode::new(sr, 1, Oversample::X4, 5.0, 1.0, 1.0, 0.0);

        // Push a lively signal so every history buffer holds non-zero state.
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = 0.7 * ops::sin(0.3 * i as Sample);
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(sr, n), &mut io);
        }

        node.reset();

        // After reset, silence in must yield exact silence out.
        let silence = [AudioBuffer::new(ChannelLayout::Mono, n)];
        let mut out2 = [AudioBuffer::new(ChannelLayout::Mono, n)];
        {
            let mut io = ProcessIo::new(&silence, &mut out2);
            node.process(&ctx(sr, n), &mut io);
        }
        for &o in out2[0].channel(0) {
            assert_eq!(o, 0.0, "residual energy after reset: {o}");
        }
    }

    #[test]
    fn stereo_channels_are_independent() {
        // A hard-panned signal (left loud, right silent) must stay that way.
        let sr = 48_000u32;
        let n = 128;
        let mut node = WaveshaperNode::new(sr, 2, Oversample::X2, 2.0, 1.0, 1.0, 0.0);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = 0.5 * ops::sin(0.2 * i as Sample);
        }
        // Right channel left silent.
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Stereo, n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sr, n), &mut io);
        for &o in outputs[0].channel(1) {
            assert_eq!(o, 0.0, "silent channel picked up energy: {o}");
        }
        let left_energy: Sample = outputs[0].channel(0).iter().map(|s| s * s).sum();
        assert!(left_energy > 0.0, "active channel produced no output");
    }
}
