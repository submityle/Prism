//! Slew-rate limiter: caps how fast the output is allowed to move toward the
//! input, independently for rising and falling slopes.
//!
//! Every sample the output is pulled toward the input, but the per-sample step
//! is clamped to a maximum magnitude:
//!
//! ```text
//! up   = 1 / (rise_seconds * fs)      // max increase per sample
//! down = 1 / (fall_seconds * fs)      // max decrease per sample
//! y[n] = clamp(x[n], y[n-1] - down, y[n-1] + up)
//! ```
//!
//! `rise_seconds` is the time the output needs to traverse one full unit of
//! amplitude while climbing, and `fall_seconds` the same while descending; a
//! time of zero removes the limit on that edge so the output follows the input
//! instantly. Because the output can never jump past the input in a single
//! step, it tracks the input along straight-line (constant-slope) segments
//! instead of the exponential curve a one-pole filter would produce.
//!
//! The behaviour spans two musically useful regimes. With long slew times the
//! node is a lag / portamento generator: control-rate steps become smooth
//! glides and transients are rounded off, which is the classic analog
//! "lag processor" used for glide and gentle envelope smoothing. With short
//! slew times applied to a full-range audio signal the slope ceiling clips the
//! steepest parts of the waveform, softening transients and, when pushed,
//! turning sines toward triangles -- the audible signature of an op-amp running
//! out of slew rate. Asymmetric rise and fall times additionally shape the
//! attack and decay of an envelope differently.
//!
//! # Relationship
//!
//! This is deliberately distinct from the
//! [`envelope_follower::EnvelopeFollowerNode`](crate::nodes::effects::envelope_follower).
//! The follower rectifies the signal and tracks its *amplitude envelope* with
//! an exponential one-pole attack / release; this node operates on the raw
//! bipolar waveform and limits its *slope* with a linear, constant-rate step.
//! It is also not a filter in the biquad / one-pole sense
//! ([`biquad`](crate::nodes::biquad), [`dc_blocker`](crate::nodes::effects::dc_blocker)):
//! those are linear time-invariant systems, whereas slope limiting is a
//! memoryless-per-step nonlinearity whose effect depends on the input's
//! instantaneous rate of change.
//!
//! # Real-time contract
//!
//! Per-channel memory (one previous-output sample) is allocated once in
//! [`SlewLimiterNode::new`]. [`SlewLimiterNode::process`] performs no
//! allocation, takes no locks, and cannot panic: non-finite inputs are treated
//! as silence, the clamp bounds are ordered by construction (the down bound is
//! never above the up bound), and the running output is flushed of denormals so
//! the state stays finite. Latency is zero.
//!
//! # Provenance
//!
//! Pure classic DSP. Slew-rate limiting is a textbook analog-electronics
//! behaviour (op-amp slew rate) and a staple modular-synth building block
//! (the "slew limiter" / "lag" / "glide" processor). Only the elementary
//! clamp-the-step recurrence shown above is used. There is no AI/ML of any
//! kind, and no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google
//! Resonance Audio, or Web Audio source or derived code.

use alloc::vec;
use alloc::vec::Vec;

use crate::buffer::ChannelLayout;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};

/// Smallest slew time (milliseconds) the node accepts. Zero means the edge is
/// unlimited and the output follows the input instantly.
pub const MIN_SLEW_MS: Sample = 0.0;

/// Largest slew time (milliseconds) the node accepts: ten seconds to traverse
/// a full unit of amplitude, slow enough for extreme glides while keeping the
/// per-sample step well away from denormal territory.
pub const MAX_SLEW_MS: Sample = 10_000.0;

/// Default rising slew time (milliseconds): a short lag that rounds transients
/// without obviously smearing them.
pub const DEFAULT_RISE_MS: Sample = 5.0;

/// Default falling slew time (milliseconds).
pub const DEFAULT_FALL_MS: Sample = 5.0;

/// Configuration for a [`SlewLimiterNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SlewLimiterParams {
    /// Time, in milliseconds, for the output to climb through one unit of
    /// amplitude. Zero removes the limit on rising slopes.
    pub rise_ms: Sample,
    /// Time, in milliseconds, for the output to descend through one unit of
    /// amplitude. Zero removes the limit on falling slopes.
    pub fall_ms: Sample,
}

impl Default for SlewLimiterParams {
    fn default() -> Self {
        Self {
            rise_ms: DEFAULT_RISE_MS,
            fall_ms: DEFAULT_FALL_MS,
        }
    }
}

impl SlewLimiterParams {
    /// Returns a copy with both slew times clamped to the supported range and
    /// any non-finite value replaced by the default.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let rise_ms = if self.rise_ms.is_finite() {
            self.rise_ms.clamp(MIN_SLEW_MS, MAX_SLEW_MS)
        } else {
            DEFAULT_RISE_MS
        };
        let fall_ms = if self.fall_ms.is_finite() {
            self.fall_ms.clamp(MIN_SLEW_MS, MAX_SLEW_MS)
        } else {
            DEFAULT_FALL_MS
        };
        Self { rise_ms, fall_ms }
    }
}

/// Converts a slew time (milliseconds, time to cross one unit of amplitude) to
/// the maximum per-sample step. A time of zero yields an unlimited step
/// (`f32::INFINITY`), which makes the clamp a no-op on that edge.
#[inline]
fn step_per_sample(slew_ms: Sample, sample_rate: u32) -> Sample {
    let ms = slew_ms.clamp(MIN_SLEW_MS, MAX_SLEW_MS);
    if ms <= 0.0 {
        return Sample::INFINITY;
    }
    let fs = sample_rate.max(1) as Sample;
    let seconds = ms / 1_000.0;
    1.0 / (seconds * fs)
}

/// A per-channel slew-rate limiter (input port 0 -> output port 0).
///
/// Each channel owns one previous-output sample, but all channels share the
/// same rising and falling rates so a stereo or surround signal is limited
/// coherently.
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{SlewLimiterNode, SlewLimiterParams};
///
/// // 1 ms to climb one unit at 48 kHz -> 48 samples to reach 1.0.
/// let mut node = SlewLimiterNode::new(
///     48_000,
///     ChannelLayout::Mono,
///     SlewLimiterParams { rise_ms: 1.0, fall_ms: 1.0 },
/// );
///
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 128);
/// input.set_active_frames(128);
/// for s in input.channel_mut(0).iter_mut() {
///     *s = 1.0;
/// }
///
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, 128);
/// output.set_active_frames(128);
///
/// let ctx = RenderContext { sample_rate: 48_000, frames: 128, playhead: 0 };
/// let inputs = [input];
/// let mut outputs = [output];
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The step is reached in a straight line after ~48 samples, not instantly.
/// let out = &outputs[0];
/// assert!(out.channel(0)[0] < 0.1);
/// assert!((out.channel(0)[47] - 1.0).abs() < 1e-4);
/// ```
#[derive(Debug)]
pub struct SlewLimiterNode {
    /// Sample rate, retained so [`SlewLimiterNode::set_params`] can recompute
    /// the per-sample steps.
    sample_rate: u32,
    /// Channel layout reported to the host.
    layout: ChannelLayout,
    /// Number of independently limited channels.
    channels: usize,
    /// Current rising slew time (milliseconds), already sanitised.
    rise_ms: Sample,
    /// Current falling slew time (milliseconds), already sanitised.
    fall_ms: Sample,
    /// Maximum upward step per sample (may be `f32::INFINITY`).
    rise_step: Sample,
    /// Maximum downward step per sample (may be `f32::INFINITY`).
    fall_step: Sample,
    /// Previous output sample `y[n-1]` per channel.
    y1: Vec<Sample>,
}

impl SlewLimiterNode {
    /// Builds a slew limiter for `layout` at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout, params: SlewLimiterParams) -> Self {
        let channels = layout.channel_count();
        let p = params.sanitised();
        Self {
            sample_rate,
            layout,
            channels,
            rise_ms: p.rise_ms,
            fall_ms: p.fall_ms,
            rise_step: step_per_sample(p.rise_ms, sample_rate),
            fall_step: step_per_sample(p.fall_ms, sample_rate),
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

    /// Returns the current rising slew time, in milliseconds.
    #[inline]
    #[must_use]
    pub fn rise_ms(&self) -> Sample {
        self.rise_ms
    }

    /// Returns the current falling slew time, in milliseconds.
    #[inline]
    #[must_use]
    pub fn fall_ms(&self) -> Sample {
        self.fall_ms
    }

    /// Returns the maximum upward step per sample (possibly `f32::INFINITY`).
    #[inline]
    #[must_use]
    pub fn rise_step(&self) -> Sample {
        self.rise_step
    }

    /// Returns the maximum downward step per sample (possibly `f32::INFINITY`).
    #[inline]
    #[must_use]
    pub fn fall_step(&self) -> Sample {
        self.fall_step
    }

    /// Updates both slew times in place (allocation-free). The stored output
    /// memory is preserved, so changing the rate only bends the slope going
    /// forward and never introduces a discontinuity. This is a control-thread
    /// operation, not called from [`AudioNode::process`].
    pub fn set_params(&mut self, params: SlewLimiterParams) {
        let p = params.sanitised();
        self.rise_ms = p.rise_ms;
        self.fall_ms = p.fall_ms;
        self.rise_step = step_per_sample(p.rise_ms, self.sample_rate);
        self.fall_step = step_per_sample(p.fall_ms, self.sample_rate);
    }
}

impl AudioNode for SlewLimiterNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let in_channels = input.channels();
        let frames = output.active_frames().min(input.active_frames());
        if frames == 0 || out_channels == 0 {
            return;
        }
        let state_n = self.channels;
        let up = self.rise_step;
        let down = self.fall_step;
        for ch in 0..out_channels {
            if ch >= state_n || ch >= in_channels {
                // No state (or no matching input) for this output channel.
                for s in output.channel_mut(ch)[..frames].iter_mut() {
                    *s = 0.0;
                }
                continue;
            }
            let mut y1 = self.y1[ch];
            for n in 0..frames {
                let x = {
                    let v = input.channel(ch)[n];
                    if v.is_finite() { v } else { 0.0 }
                };
                // `down` and `up` are both non-negative (or +inf), so the low
                // bound is never above the high bound and the clamp is valid.
                let lo = y1 - down;
                let hi = y1 + up;
                let y = flush_denormal(x.clamp(lo, hi));
                output.channel_mut(ch)[n] = y;
                y1 = y;
            }
            self.y1[ch] = y1;
        }
    }

    fn reset(&mut self) {
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
    fn run_mono(node: &mut SlewLimiterNode, signal: &[Sample]) -> Vec<Sample> {
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

    fn peak(v: &[Sample]) -> Sample {
        v.iter().fold(0.0_f32, |m, &x| m.max(x.abs()))
    }

    #[test]
    fn latency_is_zero() {
        let node = SlewLimiterNode::new(SR, ChannelLayout::Stereo, SlewLimiterParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn geometry_getters() {
        let node = SlewLimiterNode::new(SR, ChannelLayout::Quad, SlewLimiterParams::default());
        assert_eq!(node.channels(), 4);
        assert_eq!(node.layout(), ChannelLayout::Quad);
    }

    #[test]
    fn default_params_in_domain() {
        let p = SlewLimiterParams::default();
        assert!(p.rise_ms >= MIN_SLEW_MS && p.rise_ms <= MAX_SLEW_MS);
        assert!(p.fall_ms >= MIN_SLEW_MS && p.fall_ms <= MAX_SLEW_MS);
    }

    #[test]
    fn sanitise_clamps_times() {
        let low = SlewLimiterParams { rise_ms: -5.0, fall_ms: -1.0 }.sanitised();
        assert!((low.rise_ms - MIN_SLEW_MS).abs() < 1e-6);
        assert!((low.fall_ms - MIN_SLEW_MS).abs() < 1e-6);
        let high = SlewLimiterParams { rise_ms: 1e9, fall_ms: 1e9 }.sanitised();
        assert!((high.rise_ms - MAX_SLEW_MS).abs() < 1e-3);
        assert!((high.fall_ms - MAX_SLEW_MS).abs() < 1e-3);
    }

    #[test]
    fn zero_times_pass_through() {
        let mut node = SlewLimiterNode::new(
            SR,
            ChannelLayout::Mono,
            SlewLimiterParams { rise_ms: 0.0, fall_ms: 0.0 },
        );
        assert!(node.rise_step().is_infinite());
        assert!(node.fall_step().is_infinite());
        let input = sine(1_000.0, 1_024);
        let out = run_mono(&mut node, &input);
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn rising_edge_is_rate_limited() {
        // 1 ms to traverse one unit at 48 kHz -> 48 samples to reach 1.0.
        let mut node = SlewLimiterNode::new(
            SR,
            ChannelLayout::Mono,
            SlewLimiterParams { rise_ms: 1.0, fall_ms: 1.0 },
        );
        let input = vec![1.0_f32; 128];
        let out = run_mono(&mut node, &input);
        let step = 1.0_f32 / 48.0;
        // Climbs in equal steps.
        assert!((out[0] - step).abs() < 1e-5, "{}", out[0]);
        assert!((out[23] - 24.0 * step).abs() < 1e-4, "{}", out[23]);
        // Reaches and holds the target after ~48 samples.
        assert!((out[47] - 1.0).abs() < 1e-4, "{}", out[47]);
        assert!((out[80] - 1.0).abs() < 1e-6);
        // Never overshoots.
        assert!(out.iter().all(|&y| y <= 1.0 + 1e-6));
    }

    #[test]
    fn falling_edge_is_rate_limited() {
        let mut node = SlewLimiterNode::new(
            SR,
            ChannelLayout::Mono,
            SlewLimiterParams { rise_ms: 0.0, fall_ms: 1.0 },
        );
        // Jump instantly to 1.0 (rise unlimited), then command 0.0.
        let mut input = vec![1.0_f32; 16];
        input.extend(core::iter::repeat_n(0.0_f32, 128));
        let out = run_mono(&mut node, &input);
        assert!((out[0] - 1.0).abs() < 1e-6, "instant rise: {}", out[0]);
        let step = 1.0_f32 / 48.0;
        // After the command drops, the output ramps down one step per sample.
        assert!((out[16] - (1.0 - step)).abs() < 1e-4, "{}", out[16]);
        assert!((out[16 + 47] - 0.0).abs() < 1e-4, "{}", out[16 + 47]);
        assert!(out.iter().all(|&y| y >= -1e-6));
    }

    #[test]
    fn asymmetric_times_shape_edges_differently() {
        // Fast rise, slow fall.
        let mut node = SlewLimiterNode::new(
            SR,
            ChannelLayout::Mono,
            SlewLimiterParams { rise_ms: 1.0, fall_ms: 10.0 },
        );
        let mut input = vec![1.0_f32; 64];
        input.extend(core::iter::repeat_n(0.0_f32, 1_024));
        let out = run_mono(&mut node, &input);
        // Reached 1.0 within the fast-rise window.
        assert!((out[63] - 1.0).abs() < 1e-3, "{}", out[63]);
        // Ten times slower fall: 10 ms -> 480 samples to return to 0.
        assert!(out[64 + 240] > 0.4 && out[64 + 240] < 0.6, "mid-fall: {}", out[64 + 240]);
    }

    #[test]
    fn hard_slew_reduces_sine_peak() {
        // A fast sine through a slow limiter cannot keep up: its peak shrinks.
        let input = sine(2_000.0, 8_192);
        let mut node = SlewLimiterNode::new(
            SR,
            ChannelLayout::Mono,
            SlewLimiterParams { rise_ms: 20.0, fall_ms: 20.0 },
        );
        let out = run_mono(&mut node, &input);
        let tail = &out[4_096..];
        assert!(peak(tail) < 0.9, "slew-limited peak should drop: {}", peak(tail));
    }

    #[test]
    fn output_bounded_by_input() {
        let input = sine(500.0, 4_096);
        let in_peak = peak(&input);
        let mut node = SlewLimiterNode::new(
            SR,
            ChannelLayout::Stereo,
            SlewLimiterParams { rise_ms: 3.0, fall_ms: 7.0 },
        );
        let out = run_mono(&mut node, &input);
        assert!(peak(&out) <= in_peak + 1e-6, "{} vs {}", peak(&out), in_peak);
    }

    #[test]
    fn slow_slew_attenuates_high_frequencies_more() {
        let low = sine(100.0, 16_384);
        let high = sine(2_000.0, 16_384);
        let params = SlewLimiterParams { rise_ms: 10.0, fall_ms: 10.0 };
        let mut a = SlewLimiterNode::new(SR, ChannelLayout::Mono, params);
        let mut b = SlewLimiterNode::new(SR, ChannelLayout::Mono, params);
        let low_peak = peak(&run_mono(&mut a, &low)[8_000..]);
        let high_peak = peak(&run_mono(&mut b, &high)[8_000..]);
        assert!(
            high_peak < low_peak,
            "2 kHz ({high_peak}) should be slewed more than 100 Hz ({low_peak})"
        );
    }

    #[test]
    fn silence_stays_silent() {
        let mut node = SlewLimiterNode::new(SR, ChannelLayout::Mono, SlewLimiterParams::default());
        let out = run_mono(&mut node, &vec![0.0_f32; 512]);
        assert!(out.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn constant_input_converges_and_holds() {
        let mut node = SlewLimiterNode::new(
            SR,
            ChannelLayout::Mono,
            SlewLimiterParams { rise_ms: 2.0, fall_ms: 2.0 },
        );
        let out = run_mono(&mut node, &vec![0.7_f32; 4_096]);
        assert!((out[4_095] - 0.7).abs() < 1e-5, "{}", out[4_095]);
    }

    #[test]
    fn reset_reproduces_fresh_state() {
        let mut node = SlewLimiterNode::new(
            SR,
            ChannelLayout::Mono,
            SlewLimiterParams { rise_ms: 4.0, fall_ms: 4.0 },
        );
        let input = sine(300.0, 2_048);
        let first = run_mono(&mut node, &input);
        node.reset();
        let second = run_mono(&mut node, &input);
        for (a, b) in first.iter().zip(second.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn set_params_changes_step() {
        let mut node = SlewLimiterNode::new(
            SR,
            ChannelLayout::Mono,
            SlewLimiterParams { rise_ms: 1.0, fall_ms: 1.0 },
        );
        let before = node.rise_step();
        node.set_params(SlewLimiterParams { rise_ms: 2.0, fall_ms: 2.0 });
        let after = node.rise_step();
        // A longer slew time means a smaller per-sample step.
        assert!(after < before, "{before} -> {after}");
        assert!((node.rise_ms() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn extreme_params_do_not_panic() {
        for &r in &[Sample::NAN, -1e9, 1e9, 0.0, Sample::INFINITY] {
            let mut node = SlewLimiterNode::new(
                SR,
                ChannelLayout::Stereo,
                SlewLimiterParams { rise_ms: r, fall_ms: r },
            );
            let out = run_mono(&mut node, &sine(220.0, 512));
            assert!(out.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn non_finite_input_falls_back() {
        let mut node = SlewLimiterNode::new(SR, ChannelLayout::Mono, SlewLimiterParams::default());
        let input = vec![Sample::NAN, Sample::INFINITY, 0.5, Sample::NEG_INFINITY, 0.2];
        let out = run_mono(&mut node, &input);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn mono_input_into_stereo_is_finite() {
        let mut node = SlewLimiterNode::new(SR, ChannelLayout::Stereo, SlewLimiterParams::default());
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

    #[test]
    fn stereo_channels_are_coherent() {
        let mut node = SlewLimiterNode::new(
            SR,
            ChannelLayout::Stereo,
            SlewLimiterParams { rise_ms: 2.0, fall_ms: 2.0 },
        );
        let len = 1_024;
        let tone = sine(400.0, len);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        input.channel_mut(0).copy_from_slice(&tone);
        input.channel_mut(1).copy_from_slice(&tone);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        for n in 0..len {
            assert!((outputs[0].channel(0)[n] - outputs[0].channel(1)[n]).abs() < 1e-9);
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = SlewLimiterNode::new(SR, ChannelLayout::Mono, SlewLimiterParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 16);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 16);
        input.set_active_frames(0);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
    }

    #[test]
    fn from_params_matches_direct_fields() {
        let p = SlewLimiterParams { rise_ms: 3.0, fall_ms: 8.0 };
        let node = SlewLimiterNode::new(SR, ChannelLayout::Mono, p);
        assert!((node.rise_ms() - 3.0).abs() < 1e-6);
        assert!((node.fall_ms() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn getters_report_targets() {
        let node = SlewLimiterNode::new(
            SR,
            ChannelLayout::Mono,
            SlewLimiterParams { rise_ms: 1.0, fall_ms: 1.0 },
        );
        assert!(node.rise_step() > 0.0 && node.rise_step().is_finite());
        assert!(node.fall_step() > 0.0 && node.fall_step().is_finite());
    }
}
