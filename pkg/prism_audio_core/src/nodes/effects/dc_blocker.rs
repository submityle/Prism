//! DC blocker: a first-order high-pass that removes the constant (0 Hz) offset
//! from a signal while leaving the audible band essentially untouched.
//!
//! A DC offset -- a non-zero average level -- wastes headroom, biases later
//! nonlinear stages, and can thump a loudspeaker on playback. The classic fix
//! is the one-pole / one-zero differencer
//!
//! ```text
//! y[n] = x[n] - x[n-1] + R * y[n-1]
//! ```
//!
//! whose transfer function
//!
//! ```text
//! H(z) = (1 - z^-1) / (1 - R * z^-1)
//! ```
//!
//! places a zero exactly at DC (`z = 1`, so `H(1) = 0`) and a real pole at `R`
//! just inside the unit circle. The pole is positioned from the requested
//! corner frequency as `R = exp(-2 * pi * fc / fs)`, which for the low corners
//! used in practice (a few hertz to a few tens of hertz) puts the -3 dB point
//! near `fc` while passing the rest of the spectrum at essentially unity gain.
//!
//! Unlike the DC blockers embedded inside this crate's saturators (which run on
//! a fixed internal coefficient to clean up their own asymmetric distortion),
//! this node exposes the filter as a routable graph element with a tunable
//! corner, so a patch can strip DC or subsonic rumble anywhere in the signal
//! flow.
//!
//! # Relationship
//!
//! This is a dedicated, user-tunable sibling of the fixed one-pole DC blocker
//! that [`saturation::SaturationNode`](crate::nodes::effects::saturation) and
//! [`tube::TubeNode`](crate::nodes::effects::tube) apply internally. It is also
//! distinct from a biquad high-pass
//! ([`biquad`](crate::nodes::biquad)): the biquad is a second-order section with
//! a resonant `Q`, whereas this is the minimal first-order differencer whose
//! only job is to pin the DC gain to zero.
//!
//! # Real-time contract
//!
//! Per-channel filter memory is allocated once in [`DcBlockerNode::new`].
//! [`DcBlockerNode::process`] performs no allocation, takes no locks, and
//! cannot panic: non-finite inputs are treated as silence and the running
//! output is flushed of denormals so the state stays finite. Latency is zero.
//!
//! # Provenance
//!
//! Pure classic DSP. The one-pole / one-zero DC blocker is a textbook building
//! block (e.g. Zoelzer, "DAFX"; the Julius O. Smith online DSP texts). There is
//! no AI/ML of any kind, and no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Google Resonance Audio, or Web Audio source or derived code; only the
//! publicly documented difference equation is used.

use alloc::vec;
use alloc::vec::Vec;

use crate::buffer::ChannelLayout;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};

/// Smallest corner frequency (Hz) the node accepts.
pub const MIN_DC_BLOCKER_CUTOFF_HZ: Sample = 0.1;

/// Largest corner frequency (Hz) the node accepts. Beyond a few hundred hertz
/// a first-order differencer starts to audibly thin the low end, so this is a
/// generous ceiling for a DC / rumble filter.
pub const MAX_DC_BLOCKER_CUTOFF_HZ: Sample = 500.0;

/// Default corner frequency (Hz): low enough to leave the audible band intact
/// while still rejecting DC and subsonic rumble.
pub const DEFAULT_DC_BLOCKER_CUTOFF_HZ: Sample = 20.0;

/// Highest pole magnitude allowed, keeping the filter strictly stable even if
/// the computed coefficient rounds toward unity.
const MAX_POLE: Sample = 0.999_999;

/// Configuration for a [`DcBlockerNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DcBlockerParams {
    /// The -3 dB corner frequency, in hertz. Content below this folds toward
    /// zero; content above passes at essentially unity gain.
    pub cutoff_hz: Sample,
}

impl Default for DcBlockerParams {
    fn default() -> Self {
        Self {
            cutoff_hz: DEFAULT_DC_BLOCKER_CUTOFF_HZ,
        }
    }
}

impl DcBlockerParams {
    /// Returns a copy with the corner frequency clamped to the supported range
    /// and any non-finite value replaced by the default.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let cutoff_hz = if self.cutoff_hz.is_finite() {
            self.cutoff_hz
                .clamp(MIN_DC_BLOCKER_CUTOFF_HZ, MAX_DC_BLOCKER_CUTOFF_HZ)
        } else {
            DEFAULT_DC_BLOCKER_CUTOFF_HZ
        };
        Self { cutoff_hz }
    }
}

/// Maps a corner frequency to the real pole `R = exp(-2 * pi * fc / fs)`,
/// clamped strictly inside the unit circle.
#[inline]
fn pole_for(cutoff_hz: Sample, sample_rate: u32) -> Sample {
    let fs = (sample_rate.max(1)) as Sample;
    let fc = cutoff_hz.clamp(MIN_DC_BLOCKER_CUTOFF_HZ, MAX_DC_BLOCKER_CUTOFF_HZ);
    let r = bevy_math::ops::exp(-core::f32::consts::TAU * fc / fs);
    r.clamp(0.0, MAX_POLE)
}

/// A first-order DC-blocking high-pass (input port 0 -> output port 0).
#[derive(Debug)]
pub struct DcBlockerNode {
    /// Sample rate, retained so [`DcBlockerNode::set_params`] can recompute the
    /// pole.
    sample_rate: u32,
    /// Channel layout reported to the host.
    layout: ChannelLayout,
    /// Number of independently filtered channels.
    channels: usize,
    /// Current corner frequency (Hz), already sanitised.
    cutoff_hz: Sample,
    /// Real pole `R` of the recursion.
    coeff: Sample,
    /// Previous input sample `x[n-1]` per channel.
    x1: Vec<Sample>,
    /// Previous output sample `y[n-1]` per channel.
    y1: Vec<Sample>,
}

impl DcBlockerNode {
    /// Builds a DC blocker for `layout` at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout, params: DcBlockerParams) -> Self {
        let channels = layout.channel_count();
        let p = params.sanitised();
        Self {
            sample_rate,
            layout,
            channels,
            cutoff_hz: p.cutoff_hz,
            coeff: pole_for(p.cutoff_hz, sample_rate),
            x1: vec![0.0; channels],
            y1: vec![0.0; channels],
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

    /// Returns the current -3 dB corner frequency, in hertz.
    #[inline]
    #[must_use]
    pub fn cutoff_hz(&self) -> Sample {
        self.cutoff_hz
    }

    /// Returns the real pole `R` of the difference equation.
    #[inline]
    #[must_use]
    pub fn coeff(&self) -> Sample {
        self.coeff
    }

    /// Updates the corner frequency in place (allocation-free). The filter
    /// memory is preserved so the change is click-free in the DC-removed sense.
    /// This is a control-thread operation, not called from
    /// [`AudioNode::process`].
    pub fn set_params(&mut self, params: DcBlockerParams) {
        let p = params.sanitised();
        self.cutoff_hz = p.cutoff_hz;
        self.coeff = pole_for(p.cutoff_hz, self.sample_rate);
    }
}

impl AudioNode for DcBlockerNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let in_channels = input.channels();
        let frames = output.active_frames().min(input.active_frames());
        if frames == 0 || out_channels == 0 {
            return;
        }
        let state_n = self.channels;
        let r = self.coeff;
        for ch in 0..out_channels {
            if ch >= state_n || ch >= in_channels {
                // No state (or no matching input) for this output channel.
                for s in output.channel_mut(ch)[..frames].iter_mut() {
                    *s = 0.0;
                }
                continue;
            }
            let mut x1 = self.x1[ch];
            let mut y1 = self.y1[ch];
            for n in 0..frames {
                let x = {
                    let v = input.channel(ch)[n];
                    if v.is_finite() { v } else { 0.0 }
                };
                let y = flush_denormal(x - x1 + r * y1);
                output.channel_mut(ch)[n] = y;
                x1 = x;
                y1 = y;
            }
            self.x1[ch] = x1;
            self.y1[ch] = y1;
        }
    }

    fn reset(&mut self) {
        for x in &mut self.x1 {
            *x = 0.0;
        }
        for y in &mut self.y1 {
            *y = 0.0;
        }
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::AudioBuffer;
    use core::f32::consts::TAU;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    /// `bevy_math::ops::sin` wrapper so tests avoid an otherwise-unused import.
    #[inline]
    fn ops_sin(x: Sample) -> Sample {
        bevy_math::ops::sin(x)
    }

    fn sine(freq: Sample, len: usize) -> Vec<Sample> {
        (0..len)
            .map(|n| ops_sin(TAU * freq * n as Sample / SR as Sample))
            .collect()
    }

    /// Runs a mono signal through the node and returns the output.
    fn run_mono(node: &mut DcBlockerNode, signal: &[Sample]) -> Vec<Sample> {
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

    /// Peak absolute amplitude over the tail (skipping the startup transient).
    fn tail_peak(v: &[Sample], skip: usize) -> Sample {
        v.iter()
            .skip(skip)
            .fold(0.0_f32, |m, &x| m.max(x.abs()))
    }

    fn mean(v: &[Sample]) -> Sample {
        if v.is_empty() {
            return 0.0;
        }
        v.iter().sum::<Sample>() / v.len() as Sample
    }

    #[test]
    fn latency_is_zero() {
        let node = DcBlockerNode::new(SR, ChannelLayout::Stereo, DcBlockerParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn geometry_getters() {
        let node = DcBlockerNode::new(SR, ChannelLayout::Quad, DcBlockerParams::default());
        assert_eq!(node.channels(), 4);
        assert_eq!(node.layout(), ChannelLayout::Quad);
    }

    #[test]
    fn default_params_in_domain() {
        let p = DcBlockerParams::default();
        assert!(p.cutoff_hz >= MIN_DC_BLOCKER_CUTOFF_HZ);
        assert!(p.cutoff_hz <= MAX_DC_BLOCKER_CUTOFF_HZ);
    }

    #[test]
    fn sanitise_clamps_cutoff() {
        let low = DcBlockerParams { cutoff_hz: -5.0 }.sanitised();
        assert!((low.cutoff_hz - MIN_DC_BLOCKER_CUTOFF_HZ).abs() < 1e-6);
        let high = DcBlockerParams { cutoff_hz: 10_000.0 }.sanitised();
        assert!((high.cutoff_hz - MAX_DC_BLOCKER_CUTOFF_HZ).abs() < 1e-6);
    }

    #[test]
    fn sanitise_replaces_non_finite() {
        let nan = DcBlockerParams {
            cutoff_hz: Sample::NAN,
        }
        .sanitised();
        assert!((nan.cutoff_hz - DEFAULT_DC_BLOCKER_CUTOFF_HZ).abs() < 1e-6);
        let inf = DcBlockerParams {
            cutoff_hz: Sample::INFINITY,
        }
        .sanitised();
        assert!((inf.cutoff_hz - DEFAULT_DC_BLOCKER_CUTOFF_HZ).abs() < 1e-6);
    }

    #[test]
    fn coefficient_strictly_inside_unit_circle() {
        let node = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        assert!(node.coeff() > 0.0);
        assert!(node.coeff() < 1.0);
    }

    #[test]
    fn removes_constant_offset() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        let input = vec![1.0_f32; 8_192];
        let out = run_mono(&mut node, &input);
        // After the transient the output of a pure DC input decays to zero.
        assert!(tail_peak(&out, 6_000) < 1e-2, "DC should decay away");
    }

    #[test]
    fn passes_audible_tone_near_unity() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        let input = sine(1_000.0, 8_192);
        let out = run_mono(&mut node, &input);
        let peak = tail_peak(&out, 4_000);
        // A 1 kHz tone sits far above the 20 Hz corner, so it passes nearly
        // intact.
        assert!(peak > 0.97, "1 kHz tone attenuated too much: {peak}");
        assert!(peak < 1.03, "1 kHz tone gained unexpectedly: {peak}");
    }

    #[test]
    fn attenuates_low_more_than_high() {
        let mut low = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        let mut high = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        let low_tone = sine(5.0, 16_384);
        let high_tone = sine(1_000.0, 16_384);
        let low_out = tail_peak(&run_mono(&mut low, &low_tone), 8_000);
        let high_out = tail_peak(&run_mono(&mut high, &high_tone), 8_000);
        assert!(
            low_out < high_out,
            "5 Hz ({low_out}) should be attenuated more than 1 kHz ({high_out})"
        );
    }

    #[test]
    fn impulse_response_sums_to_zero() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        let mut input = vec![0.0_f32; 32_768];
        input[0] = 1.0;
        let out = run_mono(&mut node, &input);
        // The DC gain H(1) is exactly zero, so the impulse response sums to ~0.
        let s: Sample = out.iter().sum();
        assert!(s.abs() < 1e-2, "impulse response should sum to zero: {s}");
        // The very first output sample equals the impulse (y[0] = x[0]).
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn output_finite_for_non_finite_input() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        let input = vec![Sample::NAN, Sample::INFINITY, 0.5, -0.5, Sample::NEG_INFINITY];
        let out = run_mono(&mut node, &input);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
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
    fn stereo_channels_independent() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Stereo, DcBlockerParams::default());
        let len = 8_192;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for s in input.channel_mut(0).iter_mut() {
            *s = 1.0;
        }
        // Right channel stays silent.
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        // Left DC decays away; right stays exactly silent.
        assert!(outputs[0].channel(0)[..len].iter().skip(6_000).all(|s| s.abs() < 1e-2));
        assert!(outputs[0].channel(1)[..len].iter().all(|s| s.abs() < 1e-9));
    }

    #[test]
    fn surplus_output_channel_is_silent() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        let len = 256;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for s in input.channel_mut(0).iter_mut() {
            *s = 0.5;
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        assert!(outputs[0].channel(1).iter().all(|s| s.abs() < 1e-12));
        assert!(outputs[0].channel(0).iter().all(|s| s.is_finite()));
    }

    #[test]
    fn reset_reproduces_fresh_state() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        let input = sine(100.0, 2_048);
        let first = run_mono(&mut node, &input);
        node.reset();
        let second = run_mono(&mut node, &input);
        for (a, b) in first.iter().zip(second.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn set_params_changes_coefficient() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        let before = node.coeff();
        node.set_params(DcBlockerParams { cutoff_hz: 200.0 });
        let after = node.coeff();
        // A higher corner pushes the pole further from unity (smaller R).
        assert!(after < before, "higher cutoff should lower the pole: {before} -> {after}");
        assert!((node.cutoff_hz() - 200.0).abs() < 1e-3);
    }

    #[test]
    fn higher_cutoff_removes_more_low_end() {
        let gentle = DcBlockerParams { cutoff_hz: 10.0 };
        let aggressive = DcBlockerParams { cutoff_hz: 200.0 };
        let mut a = DcBlockerNode::new(SR, ChannelLayout::Mono, gentle);
        let mut b = DcBlockerNode::new(SR, ChannelLayout::Mono, aggressive);
        let tone = sine(40.0, 16_384);
        let gentle_peak = tail_peak(&run_mono(&mut a, &tone), 8_000);
        let aggressive_peak = tail_peak(&run_mono(&mut b, &tone), 8_000);
        assert!(
            aggressive_peak < gentle_peak,
            "200 Hz corner ({aggressive_peak}) should cut 40 Hz more than a 10 Hz corner ({gentle_peak})"
        );
    }

    #[test]
    fn output_has_near_zero_mean_for_biased_tone() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        // A tone riding on a large DC offset.
        let input: Vec<Sample> = sine(500.0, 16_384).iter().map(|&s| s + 0.8).collect();
        let out = run_mono(&mut node, &input);
        let tail = &out[8_000..];
        assert!(mean(tail).abs() < 1e-2, "mean should be driven to zero: {}", mean(tail));
    }

    #[test]
    fn silence_stays_silent() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Mono, DcBlockerParams::default());
        let out = run_mono(&mut node, &vec![0.0_f32; 512]);
        assert!(out.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn extreme_params_do_not_panic() {
        for &c in &[Sample::NAN, -1e9, 1e9, 0.0, Sample::INFINITY] {
            let mut node = DcBlockerNode::new(SR, ChannelLayout::Stereo, DcBlockerParams { cutoff_hz: c });
            let out = run_mono(&mut node, &sine(220.0, 512));
            assert!(out.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn mono_input_into_stereo_is_finite() {
        let mut node = DcBlockerNode::new(SR, ChannelLayout::Stereo, DcBlockerParams::default());
        let len = 256;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for s in input.channel_mut(0).iter_mut() {
            *s = 0.3;
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        assert!(outputs[0].channel(0).iter().all(|s| s.is_finite()));
        assert!(outputs[0].channel(1).iter().all(|s| *s == 0.0));
    }
}
