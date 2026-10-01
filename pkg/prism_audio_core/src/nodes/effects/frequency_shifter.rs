//! Frequency shifter: a single-sideband (SSB) shifter that translates every
//! partial of the input by the *same* number of Hz, breaking harmonic ratios.
//!
//! A frequency shifter is the inharmonic cousin of a pitch shifter. A pitch
//! shifter multiplies every partial's frequency by a constant ratio (harmonics
//! stay harmonic); a frequency shifter *adds* a constant offset `shift_hz` to
//! every partial, so a harmonic series `f, 2f, 3f, ...` becomes
//! `f + s, 2f + s, 3f + s, ...` -- no longer integer-related, which yields the
//! metallic, clangorous, "Bode/Moog shifter" timbre.
//!
//! # The model (analytic signal times a complex exponential)
//!
//! Shifting a real signal's spectrum rigidly by `+s` Hz means multiplying its
//! *analytic* signal by `exp(j * 2 * pi * s * t)` and taking the real part. The
//! analytic signal `x_r + j * x_i` is formed with a Hilbert transform, which is
//! a broadband 90-degree phase shift realised here as an odd-length,
//! antisymmetric Type-III FIR (`HILBERT_TAPS` taps, windowed with a Blackman
//! window). The real path is delayed by the filter's group delay
//! `(HILBERT_TAPS - 1) / 2` so the two quadrature paths line up, then:
//!
//! ```text
//! y[n] = x_r[n] * cos(theta[n]) - x_i[n] * sin(theta[n])
//! theta[n+1] = theta[n] + 2 * pi * shift_hz / sample_rate
//! ```
//!
//! A positive `shift_hz` moves the spectrum up, a negative one moves it down
//! (the lower sideband). The complex oscillator is shared across all channels
//! so the shift stays phase-coherent across the stereo or surround image.
//!
//! # Real-time contract
//!
//! The Hilbert kernel and the per-channel delay lines are allocated once in
//! [`FrequencyShifterNode::new`].
//! [`process`](crate::graph::AudioNode::process) performs no allocation, takes
//! no locks, and cannot panic: mismatched channel counts and zero-length blocks
//! degrade gracefully. The node reports its group delay through
//! [`latency_frames`](crate::graph::AudioNode::latency_frames), and the dry
//! path of the wet/dry [`mix`](FrequencyShifterParams::mix) is the same
//! delayed signal so dry and wet never comb-filter each other.
//!
//! # Relationship
//!
//! - [`ring_modulator`](super::ring_modulator): multiplying by a real carrier
//!   produces a *symmetric* pair of sidebands at `f_in +/- f_c`; this node uses
//!   the analytic signal to keep only one sideband, so it shifts rather than
//!   mirrors the spectrum.
//! - [`parametric_eq`](super::parametric_eq) / dynamics: those reshape
//!   amplitude per frequency; a frequency shifter relocates frequencies.
//!
//! # Provenance
//!
//! Single-sideband frequency shifting via the analytic signal (Hilbert
//! transform plus complex modulation) is classic communications and
//! audio-effects theory (the Bode/Moog frequency shifter; e.g. Zoelzer,
//! "DAFX"). The windowed-FIR Hilbert approximation is a standard textbook
//! design. This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented theory.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};

/// Number of taps in the Hilbert-transform FIR. Must be odd so the filter has
/// an integer group delay and a true quadrature centre tap.
pub const HILBERT_TAPS: usize = 127;
/// Default frequency shift (Hz).
pub const DEFAULT_SHIFT_HZ: Sample = 100.0;
/// Default wet/dry mix (fully wet).
pub const DEFAULT_MIX: Sample = 1.0;

/// Group delay of the Hilbert FIR, in samples.
const GROUP_DELAY: usize = (HILBERT_TAPS - 1) / 2;

/// Configuration for a [`FrequencyShifterNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FrequencyShifterParams {
    /// Frequency offset in Hz. Positive shifts up, negative shifts down.
    pub shift_hz: Sample,
    /// Wet/dry blend in `[0, 1]`: `0` is the delayed dry signal, `1` is fully
    /// shifted.
    pub mix: Sample,
}

impl Default for FrequencyShifterParams {
    fn default() -> Self {
        Self {
            shift_hz: DEFAULT_SHIFT_HZ,
            mix: DEFAULT_MIX,
        }
    }
}

/// A single-sideband frequency shifter node.
#[derive(Debug, Clone)]
pub struct FrequencyShifterNode {
    sample_rate: u32,
    mix: Sample,
    /// Phase increment per sample for the complex oscillator (radians).
    phase_inc: Sample,
    /// Current oscillator phase (radians, wrapped to `[-pi, pi)`).
    phase: Sample,
    /// Windowed Hilbert FIR kernel (`HILBERT_TAPS` taps).
    kernel: Vec<Sample>,
    /// Per-channel circular delay lines, each `HILBERT_TAPS` samples long.
    delay_lines: Vec<Vec<Sample>>,
    /// Shared write cursor into every delay line.
    write_pos: usize,
}

/// Builds the windowed Hilbert-transform FIR kernel.
///
/// The ideal Hilbert kernel is `h[m] = 2 / (pi * m)` for odd `m` and `0` for
/// even `m` (including the centre), making it antisymmetric about the centre
/// tap. A Blackman window tapers the truncated response to suppress ripple.
fn build_hilbert_kernel() -> Vec<Sample> {
    let mut kernel = vec![0.0; HILBERT_TAPS];
    let denom = (HILBERT_TAPS - 1) as Sample;
    let two_pi = core::f32::consts::TAU;
    for (n, tap) in kernel.iter_mut().enumerate() {
        let m = n as isize - GROUP_DELAY as isize;
        if m % 2 == 0 {
            continue;
        }
        // Blackman window, computed with deterministic `ops` trigonometry.
        let ratio = n as Sample / denom;
        let window = 0.42 - 0.5 * ops::cos(two_pi * ratio) + 0.08 * ops::cos(2.0 * two_pi * ratio);
        let ideal = 2.0 / (core::f64::consts::PI * m as f64);
        *tap = (ideal * f64::from(window)) as Sample;
    }
    kernel
}

impl FrequencyShifterNode {
    /// Builds a frequency shifter for `channels` channels at `sample_rate`.
    ///
    /// `channels` is clamped to at least one so a delay line always exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    /// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
    /// use prism_audio_core::nodes::effects::frequency_shifter::{
    ///     FrequencyShifterNode, FrequencyShifterParams,
    /// };
    ///
    /// let mut node = FrequencyShifterNode::new(48_000, 1, FrequencyShifterParams::default());
    /// let input = AudioBuffer::new(ChannelLayout::Mono, 64);
    /// let mut output = AudioBuffer::new(ChannelLayout::Mono, 64);
    /// let inputs = [input];
    /// let mut outputs = [output];
    /// let ctx = RenderContext { sample_rate: 48_000, frames: 64, playhead: 0 };
    /// let mut io = ProcessIo::new(&inputs, &mut outputs);
    /// node.process(&ctx, &mut io);
    /// // Silence in, silence out.
    /// assert!(outputs[0].channel(0).iter().all(|&s| s == 0.0));
    /// ```
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: FrequencyShifterParams) -> Self {
        let sample_rate = sample_rate.max(1);
        let channels = channels.max(1);
        let delay_lines = vec![vec![0.0; HILBERT_TAPS]; channels];
        Self {
            sample_rate,
            mix: params.mix.clamp(0.0, 1.0),
            phase_inc: Self::shift_to_phase_inc(params.shift_hz, sample_rate),
            phase: 0.0,
            kernel: build_hilbert_kernel(),
            delay_lines,
            write_pos: 0,
        }
    }

    #[inline]
    fn shift_to_phase_inc(shift_hz: Sample, sample_rate: u32) -> Sample {
        core::f32::consts::TAU * shift_hz / sample_rate as Sample
    }

    /// Replaces the parameters. The shift is retuned and the mix updated while
    /// the oscillator phase and delay-line state are preserved, so the change
    /// is click-free.
    pub fn set_params(&mut self, params: FrequencyShifterParams) {
        self.mix = params.mix.clamp(0.0, 1.0);
        self.phase_inc = Self::shift_to_phase_inc(params.shift_hz, self.sample_rate);
    }
}

impl AudioNode for FrequencyShifterNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let proc = input
            .channels()
            .min(output.channels())
            .min(self.delay_lines.len());
        let frames = input.active_frames().min(output.active_frames());
        if proc == 0 || frames == 0 {
            return;
        }
        let mix = self.mix;
        for f in 0..frames {
            let (sin_theta, cos_theta) = ops::sin_cos(self.phase);
            let pos = self.write_pos;
            for ch in 0..proc {
                let line = &mut self.delay_lines[ch];
                line[pos] = input.channel(ch)[f];
                // Hilbert (quadrature) path.
                let mut imag = 0.0;
                for (k, &coeff) in self.kernel.iter().enumerate() {
                    imag += coeff * line[(pos + HILBERT_TAPS - k) % HILBERT_TAPS];
                }
                // Real (in-phase) path: input delayed by the group delay so it
                // lines up with the quadrature path.
                let real = line[(pos + HILBERT_TAPS - GROUP_DELAY) % HILBERT_TAPS];
                let shifted = flush_denormal(real * cos_theta - imag * sin_theta);
                output.channel_mut(ch)[f] = real + mix * (shifted - real);
            }
            self.write_pos = (pos + 1) % HILBERT_TAPS;
            self.phase += self.phase_inc;
            if self.phase >= core::f32::consts::PI {
                self.phase -= core::f32::consts::TAU;
            } else if self.phase < -core::f32::consts::PI {
                self.phase += core::f32::consts::TAU;
            }
        }
    }

    fn reset(&mut self) {
        for line in &mut self.delay_lines {
            line.iter_mut().for_each(|s| *s = 0.0);
        }
        self.write_pos = 0;
        self.phase = 0.0;
    }

    fn latency_frames(&self) -> u32 {
        GROUP_DELAY as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    fn render(node: &mut FrequencyShifterNode, input: &AudioBuffer) -> AudioBuffer {
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

    fn tone(layout: ChannelLayout, freq_hz: Sample, frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        buf.set_active_frames(frames);
        let w = core::f32::consts::TAU * freq_hz / SR as Sample;
        for ch in 0..buf.channels() {
            let data = buf.channel_mut(ch);
            for (n, s) in data.iter_mut().enumerate() {
                *s = ops::sin(w * n as Sample);
            }
        }
        buf
    }

    /// Goertzel single-bin magnitude estimate at `freq_hz`.
    fn bin_magnitude(data: &[Sample], freq_hz: Sample) -> Sample {
        let w = core::f32::consts::TAU * freq_hz / SR as Sample;
        let coeff = 2.0 * ops::cos(w);
        let mut s_prev = 0.0f32;
        let mut s_prev2 = 0.0f32;
        for &x in data {
            let s = x + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
        }
        let power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2;
        ops::sqrt(power.max(0.0))
    }

    #[test]
    fn silence_stays_silent() {
        let mut node = FrequencyShifterNode::new(SR, 2, FrequencyShifterParams::default());
        let input = AudioBuffer::new(ChannelLayout::Stereo, 256);
        let out = render(&mut node, &input);
        for ch in 0..out.channels() {
            assert!(out.channel(ch).iter().all(|&s| s.abs() < 1.0e-6));
        }
    }

    #[test]
    fn positive_shift_moves_energy_up() {
        let params = FrequencyShifterParams {
            shift_hz: 500.0,
            mix: 1.0,
        };
        let mut node = FrequencyShifterNode::new(SR, 1, params);
        let input = tone(ChannelLayout::Mono, 2_000.0, 8_192);
        let out = render(&mut node, &input);
        // Skip the FIR warm-up when measuring.
        let tail = &out.channel(0)[HILBERT_TAPS..];
        let at_shifted = bin_magnitude(tail, 2_500.0);
        let at_original = bin_magnitude(tail, 2_000.0);
        let at_mirror = bin_magnitude(tail, 1_500.0);
        assert!(at_shifted > at_original * 4.0, "energy should sit at f+s");
        // SSB: the mirror (f-s) sideband is strongly suppressed, unlike a ring
        // modulator which would produce it symmetrically.
        assert!(at_shifted > at_mirror * 4.0, "lower sideband must be rejected");
    }

    #[test]
    fn negative_shift_moves_energy_down() {
        let params = FrequencyShifterParams {
            shift_hz: -500.0,
            mix: 1.0,
        };
        let mut node = FrequencyShifterNode::new(SR, 1, params);
        let input = tone(ChannelLayout::Mono, 2_000.0, 8_192);
        let out = render(&mut node, &input);
        let tail = &out.channel(0)[HILBERT_TAPS..];
        let at_shifted = bin_magnitude(tail, 1_500.0);
        let at_mirror = bin_magnitude(tail, 2_500.0);
        assert!(at_shifted > at_mirror * 4.0, "energy should sit at f-s");
    }

    #[test]
    fn zero_shift_is_pure_delay() {
        let params = FrequencyShifterParams {
            shift_hz: 0.0,
            mix: 1.0,
        };
        let mut node = FrequencyShifterNode::new(SR, 1, params);
        let input = tone(ChannelLayout::Mono, 1_000.0, 4_096);
        let out = render(&mut node, &input);
        // With no shift the output is the group-delayed input (real path only).
        for n in HILBERT_TAPS..input.active_frames() {
            let expected = input.channel(0)[n - GROUP_DELAY];
            assert!(
                (out.channel(0)[n] - expected).abs() < 1.0e-5,
                "zero shift must be a pure delay at {n}"
            );
        }
    }

    #[test]
    fn dry_mix_is_delayed_input() {
        let params = FrequencyShifterParams {
            shift_hz: 300.0,
            mix: 0.0,
        };
        let mut node = FrequencyShifterNode::new(SR, 1, params);
        let input = tone(ChannelLayout::Mono, 1_000.0, 4_096);
        let out = render(&mut node, &input);
        for n in HILBERT_TAPS..input.active_frames() {
            let expected = input.channel(0)[n - GROUP_DELAY];
            assert!(
                (out.channel(0)[n] - expected).abs() < 1.0e-5,
                "mix=0 must be the delayed dry signal at {n}"
            );
        }
    }

    #[test]
    fn channels_share_oscillator() {
        let params = FrequencyShifterParams {
            shift_hz: 400.0,
            mix: 1.0,
        };
        let mut node = FrequencyShifterNode::new(SR, 2, params);
        let input = tone(ChannelLayout::Stereo, 1_000.0, 2_048);
        let out = render(&mut node, &input);
        for (a, b) in out.channel(0).iter().zip(out.channel(1).iter()) {
            assert!((a - b).abs() < 1.0e-6, "identical channels must match");
        }
    }

    #[test]
    fn reports_group_delay_latency() {
        let node = FrequencyShifterNode::new(SR, 1, FrequencyShifterParams::default());
        assert_eq!(node.latency_frames(), GROUP_DELAY as u32);
    }

    #[test]
    fn kernel_is_antisymmetric_with_zero_centre() {
        let kernel = build_hilbert_kernel();
        assert_eq!(kernel.len(), HILBERT_TAPS);
        assert_eq!(kernel[GROUP_DELAY], 0.0, "centre tap is zero");
        for k in 1..=GROUP_DELAY {
            let left = kernel[GROUP_DELAY - k];
            let right = kernel[GROUP_DELAY + k];
            assert!((left + right).abs() < 1.0e-6, "antisymmetric about centre");
            if k % 2 == 0 {
                assert_eq!(right, 0.0, "even taps vanish");
            }
        }
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = FrequencyShifterNode::new(SR, 2, FrequencyShifterParams::default());
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
        assert_eq!(node.phase, 0.0);
    }

    #[test]
    fn reset_clears_state() {
        let mut node = FrequencyShifterNode::new(SR, 1, FrequencyShifterParams::default());
        let input = tone(ChannelLayout::Mono, 1_000.0, 1_024);
        let _ = render(&mut node, &input);
        node.reset();
        assert_eq!(node.phase, 0.0);
        assert_eq!(node.write_pos, 0);
        assert!(node.delay_lines[0].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn mix_is_clamped() {
        let node = FrequencyShifterNode::new(
            SR,
            1,
            FrequencyShifterParams {
                shift_hz: 100.0,
                mix: 4.0,
            },
        );
        assert_eq!(node.mix, 1.0);
    }

    #[test]
    fn default_params_are_sane() {
        let p = FrequencyShifterParams::default();
        assert_eq!(p.shift_hz, DEFAULT_SHIFT_HZ);
        assert_eq!(p.mix, DEFAULT_MIX);
    }
}
