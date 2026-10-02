//! Chebyshev waveshaper: a precise harmonic generator built from Chebyshev
//! polynomials of the first kind.
//!
//! The defining identity of the first-kind Chebyshev polynomials is
//!
//! ```text
//! T_n(cos(theta)) = cos(n * theta)
//! ```
//!
//! so feeding a unit-amplitude cosine through `T_n` produces, exactly, the
//! n-th harmonic of that cosine. By driving the input through a weighted sum of
//! Chebyshev polynomials the node synthesises a chosen harmonic series with
//! per-harmonic control:
//!
//! ```text
//! xc   = clamp(drive * x, -1, 1)
//! odd  = sum over odd  k of h[k] * T_k(xc)              (DC-free)
//! even = sum over even k of h[k] * (T_k(xc) - T_k(0))   (DC-corrected)
//! wet  = odd + dcblock(even)
//! y    = output * ((1 - mix) * x + mix * wet)
//! ```
//!
//! The polynomials are evaluated with the stable three-term recurrence
//! `T_0 = 1`, `T_1 = x`, `T_{k+1} = 2*x*T_k - T_{k-1}`, so the whole harmonic
//! stack costs a handful of multiply-adds per sample regardless of order and
//! never evaluates a transcendental function.
//!
//! Two details make this musically usable rather than a textbook curiosity.
//! First, the identity only holds (and the polynomials only stay bounded) for
//! `|x| <= 1`, so the driven input is hard-clamped to `[-1, 1]` before shaping;
//! this both guarantees `|T_k| <= 1` (hence a bounded, stable output) and lets
//! `drive` push the signal into the clamp for extra grit. Second, even-order
//! polynomials carry both a static constant (`T_2 = 2x^2 - 1`, `T_4 = 8x^4 - ...`)
//! and, for an AC signal, an amplitude-dependent DC term, whereas odd-order
//! polynomials are odd functions and inject no DC at all. The static term
//! `T_k(0)` is subtracted analytically (so a zero input maps to zero and no step
//! pedestal appears), and only the even-order sum is routed through a one-pole /
//! one-zero DC blocker to strip the residual amplitude-dependent offset. The
//! odd-order sum -- which includes the fundamental -- bypasses the blocker
//! entirely, so a fundamental-only setting is a phase-faithful passthrough.
//!
//! Because each channel is shaped independently with the same transfer curve, a
//! correlated multi-channel signal is coloured coherently. The summed output
//! can exceed unity magnitude (its bound is `output * sum|h[k]|`); like the
//! other `effects` nodes this stage deliberately does not clamp its output and
//! leaves headroom management to a downstream gain or limiter.
//!
//! # Relationship
//!
//! This node is a *precise, harmonic-by-harmonic* shaper, which sets it apart
//! from the other nonlinearities in `effects`:
//!
//! - [`waveshaper::WaveshaperNode`](crate::nodes::effects::waveshaper) and the
//!   saturators ([`saturation::SaturationNode`](crate::nodes::effects::saturation),
//!   [`tube::TubeNode`](crate::nodes::effects::tube),
//!   [`diode_clipper::DiodeClipperNode`](crate::nodes::effects::diode_clipper))
//!   apply a *fixed* `tanh`-family or diode curve whose harmonic recipe is a
//!   by-product of the chosen shape. Here the transfer curve is *built from* the
//!   requested harmonic amplitudes, so a single harmonic can be isolated or an
//!   arbitrary spectrum sculpted directly.
//! - [`wavefolder::WavefolderNode`](crate::nodes::effects::wavefolder) reflects
//!   the signal past a threshold for dense, inharmonic-sounding folds; the
//!   Chebyshev stack produces strictly integer harmonics instead.
//! - [`exciter::ExciterNode`](crate::nodes::effects::exciter) band-splits the
//!   signal and only generates *high* harmonics psychoacoustically; this node
//!   shapes the full-band signal with explicit harmonic weights.
//!
//! # Real-time contract
//!
//! All per-channel state (the two DC-blocker memories) is allocated once in
//! [`ChebyshevShaperNode::new`]. [`ChebyshevShaperNode::process`] performs no
//! allocation, takes no locks, and cannot panic: non-finite inputs are treated
//! as silence, the driven input is clamped to `[-1, 1]` so the polynomials stay
//! bounded, and every stored and emitted sample is flushed of denormals. The
//! `drive`, `mix`, and `output` macros are [`Smoothed`] and advanced exactly
//! once per frame; the harmonic weights are plain scalars set from the control
//! thread via [`ChebyshevShaperNode::set_harmonics`]. Latency is zero.
//!
//! # Provenance
//!
//! Pure classic DSP. Using Chebyshev polynomials of the first kind as a
//! harmonic-generating waveshaper is long-standing, publicly documented
//! signal-processing knowledge (the `T_n(cos t) = cos n t` identity and the
//! standard three-term recurrence), paired with a textbook one-pole DC blocker.
//! There is no AI/ML of any kind, and no Unreal Engine, Unity, Godot, Wwise,
//! FMOD, Steam Audio, Google Resonance Audio, or Web Audio source or derived
//! code.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::buffer::ChannelLayout;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Number of Chebyshev harmonics the node synthesises (`T_1` through
/// `T_MAX_HARMONICS`). Eight covers the musically useful range while keeping
/// the per-sample recurrence cheap.
pub const MAX_HARMONICS: usize = 8;

/// Default pre-shaper drive. Unity means the input reaches the clamp only when
/// its own magnitude does.
pub const DEFAULT_DRIVE: Sample = 1.0;

/// Smallest accepted drive.
pub const MIN_DRIVE: Sample = 0.0;

/// Largest accepted drive. Beyond a few units the clamp dominates and the curve
/// stops changing, so the ceiling is kept modest.
pub const MAX_DRIVE: Sample = 8.0;

/// Default wet/dry blend: fully wet, since the node is a dedicated shaper.
pub const DEFAULT_MIX: Sample = 1.0;

/// Default output trim (unity).
pub const DEFAULT_OUTPUT: Sample = 1.0;

/// Largest accepted output trim.
pub const MAX_OUTPUT: Sample = 4.0;

/// Largest accepted magnitude for a single harmonic weight.
pub const MAX_HARMONIC_GAIN: Sample = 4.0;

/// Corner frequency, in Hz, of the wet-path DC blocker that removes the
/// constant term contributed by even-order polynomials.
pub const DC_BLOCKER_CUTOFF_HZ: Sample = 20.0;

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Maps the fixed [`DC_BLOCKER_CUTOFF_HZ`] corner to the real pole
/// `R = exp(-2 * pi * fc / fs)` of the one-pole DC blocker.
#[inline]
fn dc_pole(sample_rate: u32) -> Sample {
    let fs = sample_rate.max(1) as Sample;
    ops::exp(-core::f32::consts::TAU * DC_BLOCKER_CUTOFF_HZ / fs)
}

/// Configuration for a [`ChebyshevShaperNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ChebyshevShaperParams {
    /// Pre-shaper gain applied before the clamp. Higher values push the signal
    /// into the `[-1, 1]` clamp for a harder, clipped character.
    pub drive: Sample,
    /// Per-harmonic weights: `harmonics[k]` scales `T_{k+1}`, so index 0 is the
    /// fundamental (`T_1`), index 1 the second harmonic (`T_2`), and so on.
    pub harmonics: [Sample; MAX_HARMONICS],
    /// Wet/dry blend in `[0, 1]`: `0` is fully dry, `1` is fully shaped.
    pub mix: Sample,
    /// Post-mix output trim.
    pub output: Sample,
}

impl Default for ChebyshevShaperParams {
    fn default() -> Self {
        let mut harmonics = [0.0; MAX_HARMONICS];
        // Default to a clean fundamental so an untouched node is close to a
        // (DC-blocked) passthrough.
        harmonics[0] = 1.0;
        Self {
            drive: DEFAULT_DRIVE,
            harmonics,
            mix: DEFAULT_MIX,
            output: DEFAULT_OUTPUT,
        }
    }
}

impl ChebyshevShaperParams {
    /// Returns a copy with every field clamped to its supported range and any
    /// non-finite value replaced by a safe default.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let mut harmonics = self.harmonics;
        for h in &mut harmonics {
            *h = finite_or(*h, 0.0).clamp(-MAX_HARMONIC_GAIN, MAX_HARMONIC_GAIN);
        }
        Self {
            drive: finite_or(self.drive, DEFAULT_DRIVE).clamp(MIN_DRIVE, MAX_DRIVE),
            harmonics,
            mix: finite_or(self.mix, DEFAULT_MIX).clamp(0.0, 1.0),
            output: finite_or(self.output, DEFAULT_OUTPUT).clamp(0.0, MAX_OUTPUT),
        }
    }
}

/// A per-channel Chebyshev harmonic shaper (input port 0 -> output port 0).
///
/// Every channel shares the same drive, harmonic weights, mix, and output trim,
/// so a multi-channel signal is coloured coherently; only the DC-blocker memory
/// is per channel.
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{ChebyshevShaperNode, ChebyshevShaperParams};
///
/// // Pure second harmonic: T_2 turns a 100 Hz sine into a 200 Hz tone.
/// let mut harmonics = [0.0_f32; 8];
/// harmonics[1] = 1.0; // weight on T_2
/// let mut node = ChebyshevShaperNode::new(
///     48_000,
///     ChannelLayout::Mono,
///     ChebyshevShaperParams { drive: 1.0, harmonics, mix: 1.0, output: 1.0 },
/// );
///
/// let frames = 4_800;
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
/// input.set_active_frames(frames);
/// for (n, s) in input.channel_mut(0).iter_mut().enumerate() {
///     let t = n as f32 / 48_000.0;
///     *s = (2.0 * std::f32::consts::PI * 100.0 * t).sin();
/// }
///
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, frames);
/// output.set_active_frames(frames);
///
/// let ctx = RenderContext { sample_rate: 48_000, frames, playhead: 0 };
/// let inputs = [input];
/// let mut outputs = [output];
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The shaped output is real and non-silent once the DC blocker settles.
/// let peak = outputs[0]
///     .channel(0)
///     .iter()
///     .skip(500)
///     .fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.3);
/// ```
#[derive(Debug)]
pub struct ChebyshevShaperNode {
    /// Sample rate in Hz, retained for reporting.
    sample_rate: u32,
    /// Channel layout reported to the host.
    layout: ChannelLayout,
    /// Number of independently shaped channels.
    channels: usize,
    /// Harmonic weights (`harmonics[k]` scales `T_{k+1}`).
    harmonics: [Sample; MAX_HARMONICS],
    /// Smoothed pre-shaper drive.
    drive: Smoothed,
    /// Smoothed wet/dry blend.
    mix: Smoothed,
    /// Smoothed post-mix output trim.
    output: Smoothed,
    /// Real pole `R` of the wet-path DC blocker.
    dc_pole: Sample,
    /// DC-blocker previous input (`x[n-1]`, the pre-blocker wet signal).
    dc_x1: Vec<Sample>,
    /// DC-blocker previous output (`y[n-1]`).
    dc_y1: Vec<Sample>,
}

impl ChebyshevShaperNode {
    /// Builds a Chebyshev shaper for `layout`'s channels running at
    /// `sample_rate` Hz. All parameters are sanitised.
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout, params: ChebyshevShaperParams) -> Self {
        let channels = layout.channel_count();
        let p = params.sanitised();
        Self {
            sample_rate: sample_rate.max(1),
            layout,
            channels,
            harmonics: p.harmonics,
            drive: Smoothed::new(p.drive),
            mix: Smoothed::new(p.mix),
            output: Smoothed::new(p.output),
            dc_pole: dc_pole(sample_rate),
            dc_x1: vec![0.0; channels],
            dc_y1: vec![0.0; channels],
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

    /// Returns the sample rate in Hz.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the target drive the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn drive(&self) -> Sample {
        self.drive.target()
    }

    /// Returns the target wet/dry blend the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn mix(&self) -> Sample {
        self.mix.target()
    }

    /// Returns the target output trim the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn output(&self) -> Sample {
        self.output.target()
    }

    /// Returns the current harmonic weights (`harmonics[k]` scales `T_{k+1}`).
    #[inline]
    #[must_use]
    pub fn harmonics(&self) -> [Sample; MAX_HARMONICS] {
        self.harmonics
    }

    /// Sets a new target drive, gliding with `ramp`. Clamped to
    /// `[MIN_DRIVE, MAX_DRIVE]`.
    #[inline]
    pub fn set_drive(&mut self, drive: Sample, ramp: Ramp) {
        let v = finite_or(drive, self.drive.target()).clamp(MIN_DRIVE, MAX_DRIVE);
        self.drive.set_target(v, ramp);
    }

    /// Sets a new target wet/dry blend, gliding with `ramp`. Clamped to
    /// `[0, 1]`.
    #[inline]
    pub fn set_mix(&mut self, mix: Sample, ramp: Ramp) {
        let v = finite_or(mix, self.mix.target()).clamp(0.0, 1.0);
        self.mix.set_target(v, ramp);
    }

    /// Sets a new target output trim, gliding with `ramp`. Clamped to
    /// `[0, MAX_OUTPUT]`.
    #[inline]
    pub fn set_output(&mut self, output: Sample, ramp: Ramp) {
        let v = finite_or(output, self.output.target()).clamp(0.0, MAX_OUTPUT);
        self.output.set_target(v, ramp);
    }

    /// Replaces the harmonic weights in place (allocation-free). Up to
    /// [`MAX_HARMONICS`] values are copied (extras ignored, missing ones left
    /// unchanged), each clamped to `[-MAX_HARMONIC_GAIN, MAX_HARMONIC_GAIN]`
    /// with non-finite values treated as zero. This is a control-thread
    /// operation, not called from [`AudioNode::process`].
    pub fn set_harmonics(&mut self, harmonics: &[Sample]) {
        for (slot, &value) in self.harmonics.iter_mut().zip(harmonics.iter()) {
            *slot = finite_or(value, 0.0).clamp(-MAX_HARMONIC_GAIN, MAX_HARMONIC_GAIN);
        }
    }

    /// Evaluates the weighted Chebyshev sum for a single already-clamped input,
    /// split into a DC-free part and a part that still needs DC removal.
    ///
    /// Returns `(direct, even_sum)`:
    /// - `direct` is the sum of the odd-order polynomials (`T_1, T_3, ...`),
    ///   which are odd functions and therefore inject no DC; it is routed
    ///   straight to the output with its phase intact.
    /// - `even_sum` is the sum of the even-order polynomials (`T_2, T_4, ...`)
    ///   with their exact static term `T_n(0)` subtracted, so a zero input maps
    ///   to zero; the caller routes it through the one-pole DC blocker to strip
    ///   the remaining amplitude-dependent DC.
    #[inline]
    fn shape(&self, xc: Sample) -> (Sample, Sample) {
        // Three-term recurrence for T_n(xc) evaluated in parallel with T_n(0)
        // so the static DC term of each even-order polynomial is known exactly:
        // T_0 = 1, T_1 = xc, T_{n+1} = 2*xc*T_n - T_{n-1}; at x = 0 this reduces
        // to T_{n+1}(0) = -T_{n-1}(0).
        let mut t_prev = 1.0; // T_0(xc)
        let mut t_cur = xc; // T_1(xc)
        let mut z_prev = 1.0; // T_0(0)
        let mut z_cur = 0.0; // T_1(0)
        // harmonics[0] weights T_1 (odd order): DC-free, routed directly.
        let mut direct = self.harmonics[0] * t_cur;
        let mut even_sum = 0.0;
        let mut order = 2usize; // order of the next polynomial generated (T_2 ...)
        let mut k = 1;
        while k < MAX_HARMONICS {
            let t_next = 2.0 * xc * t_cur - t_prev;
            let z_next = -z_prev; // T_order(0)
            if order.is_multiple_of(2) {
                // Even order: subtract the exact static DC term T_order(0).
                even_sum += self.harmonics[k] * (t_next - z_next);
            } else {
                direct += self.harmonics[k] * t_next;
            }
            t_prev = t_cur;
            t_cur = t_next;
            z_prev = z_cur;
            z_cur = z_next;
            order += 1;
            k += 1;
        }
        (direct, even_sum)
    }
}

impl AudioNode for ChebyshevShaperNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let in_channels = input.channels();
        let frames = output.active_frames().min(input.active_frames());
        if frames == 0 || out_channels == 0 {
            return;
        }
        let channels = out_channels.min(in_channels).min(self.channels);
        let pole = self.dc_pole;

        for f in 0..frames {
            // Advance the shared smoothed macros exactly once per frame.
            let drive = self.drive.next_sample();
            let mix = self.mix.next_sample();
            let out_level = self.output.next_sample();
            let dry_gain = 1.0 - mix;

            for ch in 0..channels {
                let x = {
                    let v = input.channel(ch)[f];
                    if v.is_finite() { v } else { 0.0 }
                };
                // Clamp into the domain where the Chebyshev identity holds and
                // the polynomials stay bounded.
                let xc = (drive * x).clamp(-1.0, 1.0);
                let (direct, even_in) = self.shape(xc);

                // Only the even-order content carries DC (odd-order polynomials
                // are odd functions, hence DC-free), so the one-pole / one-zero
                // DC blocker runs on that part alone: y = x - x1 + R * y1. The
                // odd-order sum bypasses it and keeps its phase unchanged.
                let x1 = self.dc_x1[ch];
                let y1 = self.dc_y1[ch];
                let blocked = flush_denormal(even_in - x1 + pole * y1);
                self.dc_x1[ch] = even_in;
                self.dc_y1[ch] = blocked;

                let wet = direct + blocked;
                let y = out_level * (dry_gain * x + mix * wet);
                output.channel_mut(ch)[f] = flush_denormal(y);
            }
        }

        // Output channels without a matching input / state are silenced.
        for ch in channels..out_channels {
            for s in output.channel_mut(ch)[..frames].iter_mut() {
                *s = 0.0;
            }
        }
    }

    fn reset(&mut self) {
        for s in &mut self.dc_x1 {
            *s = 0.0;
        }
        for s in &mut self.dc_y1 {
            *s = 0.0;
        }
        self.drive = Smoothed::new(self.drive.target());
        self.mix = Smoothed::new(self.mix.target());
        self.output = Smoothed::new(self.output.target());
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
        ops::sin(x)
    }

    fn sine_buffer(freq: Sample, amp: Sample, frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, frames);
        buf.set_active_frames(frames);
        for (n, s) in buf.channel_mut(0).iter_mut().enumerate() {
            let t = n as Sample / SR as Sample;
            *s = amp * ops_sin(TAU * freq * t);
        }
        buf
    }

    fn run(node: &mut ChebyshevShaperNode, input: AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let layout = input.layout();
        let mut output = AudioBuffer::new(layout, frames.max(1));
        output.set_active_frames(frames);
        let c = ctx(frames);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);
        let [out] = outputs;
        out
    }

    fn rising_sign_changes(samples: &[Sample], skip: usize) -> usize {
        let mut count = 0;
        let mut prev = 0.0_f32;
        for &s in &samples[skip..] {
            if prev <= 0.0 && s > 0.0 {
                count += 1;
            }
            if s != 0.0 {
                prev = s;
            }
        }
        count
    }

    fn only_harmonic(index: usize) -> [Sample; MAX_HARMONICS] {
        let mut h = [0.0; MAX_HARMONICS];
        h[index] = 1.0;
        h
    }

    #[test]
    fn default_params_are_sane() {
        let p = ChebyshevShaperParams::default();
        assert_eq!(p, p.sanitised());
        assert_eq!(p.drive, DEFAULT_DRIVE);
        assert_eq!(p.mix, DEFAULT_MIX);
        assert_eq!(p.output, DEFAULT_OUTPUT);
        assert_eq!(p.harmonics[0], 1.0);
        assert!(p.harmonics[1..].iter().all(|&h| h == 0.0));
    }

    #[test]
    fn sanitise_clamps_ranges() {
        let mut harmonics = [10.0; MAX_HARMONICS];
        harmonics[0] = -10.0;
        let p = ChebyshevShaperParams {
            drive: 100.0,
            harmonics,
            mix: 5.0,
            output: 9.0,
        }
        .sanitised();
        assert_eq!(p.drive, MAX_DRIVE);
        assert_eq!(p.mix, 1.0);
        assert_eq!(p.output, MAX_OUTPUT);
        assert_eq!(p.harmonics[0], -MAX_HARMONIC_GAIN);
        assert_eq!(p.harmonics[1], MAX_HARMONIC_GAIN);
    }

    #[test]
    fn sanitise_replaces_non_finite() {
        let mut harmonics = [Sample::NAN; MAX_HARMONICS];
        harmonics[2] = Sample::INFINITY;
        let p = ChebyshevShaperParams {
            drive: Sample::NAN,
            harmonics,
            mix: Sample::INFINITY,
            output: Sample::NEG_INFINITY,
        }
        .sanitised();
        assert_eq!(p.drive, DEFAULT_DRIVE);
        assert_eq!(p.mix, DEFAULT_MIX);
        assert_eq!(p.output, DEFAULT_OUTPUT);
        assert!(p.harmonics.iter().all(|&h| h == 0.0));
    }

    #[test]
    fn new_reports_channels_and_layout() {
        let node =
            ChebyshevShaperNode::new(SR, ChannelLayout::Stereo, ChebyshevShaperParams::default());
        assert_eq!(node.channels(), 2);
        assert_eq!(node.layout(), ChannelLayout::Stereo);
        assert_eq!(node.sample_rate(), SR);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node =
            ChebyshevShaperNode::new(SR, ChannelLayout::Mono, ChebyshevShaperParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 256);
        input.set_active_frames(256);
        let out = run(&mut node, input);
        assert!(out.channel(0).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn fundamental_only_is_near_passthrough() {
        // T_1(x) = x, so with only the fundamental weighted the wet path is the
        // (DC-blocked) input itself.
        let mut node = ChebyshevShaperNode::new(
            SR,
            ChannelLayout::Mono,
            ChebyshevShaperParams {
                drive: 1.0,
                harmonics: only_harmonic(0),
                mix: 1.0,
                output: 1.0,
            },
        );
        let input = sine_buffer(1_000.0, 0.5, 4_800);
        let reference: Vec<Sample> = input.channel(0).to_vec();
        let out = run(&mut node, input);
        // After the DC blocker settles the shaped output tracks the input.
        for n in 1_000..4_800 {
            assert!((out.channel(0)[n] - reference[n]).abs() < 1e-2);
        }
    }

    #[test]
    fn second_harmonic_doubles_frequency() {
        let mut node = ChebyshevShaperNode::new(
            SR,
            ChannelLayout::Mono,
            ChebyshevShaperParams {
                drive: 1.0,
                harmonics: only_harmonic(1),
                mix: 1.0,
                output: 1.0,
            },
        );
        let freq = 300.0;
        let frames = SR as usize;
        let out = run(&mut node, sine_buffer(freq, 1.0, frames));
        let input_cycles =
            rising_sign_changes(sine_buffer(freq, 1.0, frames).channel(0), 2_000);
        let out_cycles = rising_sign_changes(out.channel(0), 2_000);
        let ratio = out_cycles as f32 / input_cycles as f32;
        assert!((ratio - 2.0).abs() < 0.1, "ratio={ratio}");
    }

    #[test]
    fn third_harmonic_triples_frequency() {
        let mut node = ChebyshevShaperNode::new(
            SR,
            ChannelLayout::Mono,
            ChebyshevShaperParams {
                drive: 1.0,
                harmonics: only_harmonic(2),
                mix: 1.0,
                output: 1.0,
            },
        );
        let freq = 300.0;
        let frames = SR as usize;
        let out = run(&mut node, sine_buffer(freq, 1.0, frames));
        let input_cycles =
            rising_sign_changes(sine_buffer(freq, 1.0, frames).channel(0), 2_000);
        let out_cycles = rising_sign_changes(out.channel(0), 2_000);
        let ratio = out_cycles as f32 / input_cycles as f32;
        assert!((ratio - 3.0).abs() < 0.15, "ratio={ratio}");
    }

    #[test]
    fn output_is_bounded_for_large_input() {
        // All harmonics at max; |T_k| <= 1 so |wet| <= sum|h| = 8 * 4 = 32,
        // output trim 1 -> output stays finite and bounded regardless of input.
        let harmonics = [MAX_HARMONIC_GAIN; MAX_HARMONICS];
        let mut node = ChebyshevShaperNode::new(
            SR,
            ChannelLayout::Mono,
            ChebyshevShaperParams { drive: MAX_DRIVE, harmonics, mix: 1.0, output: 1.0 },
        );
        let out = run(&mut node, sine_buffer(220.0, 10.0, 2_000));
        for &s in out.channel(0) {
            assert!(s.is_finite());
            assert!(s.abs() <= MAX_HARMONIC_GAIN * MAX_HARMONICS as f32 + 1e-3);
        }
    }

    #[test]
    fn dry_mix_is_passthrough() {
        let mut node = ChebyshevShaperNode::new(
            SR,
            ChannelLayout::Mono,
            ChebyshevShaperParams {
                drive: 3.0,
                harmonics: only_harmonic(3),
                mix: 0.0,
                output: 1.0,
            },
        );
        let input = sine_buffer(440.0, 0.7, 2_000);
        let reference: Vec<Sample> = input.channel(0).to_vec();
        let out = run(&mut node, input);
        for (o, i) in out.channel(0).iter().zip(reference.iter()) {
            assert!((o - i).abs() < 1e-6, "o={o} i={i}");
        }
    }

    #[test]
    fn output_trim_scales_result() {
        let make = |trim: Sample| {
            ChebyshevShaperNode::new(
                SR,
                ChannelLayout::Mono,
                ChebyshevShaperParams {
                    drive: 1.0,
                    harmonics: only_harmonic(1),
                    mix: 1.0,
                    output: trim,
                },
            )
        };
        let peak = |trim: Sample| {
            let mut node = make(trim);
            run(&mut node, sine_buffer(300.0, 1.0, 4_800))
                .channel(0)
                .iter()
                .skip(1_000)
                .fold(0.0_f32, |m, s| m.max(s.abs()))
        };
        let full = peak(1.0);
        let half = peak(0.5);
        assert!((full - 2.0 * half).abs() < 0.05 * full, "full={full} half={half}");
    }

    #[test]
    fn drive_changes_curve() {
        // Fundamental only, but a hard drive clamps the sine toward a square,
        // so the driven output no longer matches the clean input.
        let mut node = ChebyshevShaperNode::new(
            SR,
            ChannelLayout::Mono,
            ChebyshevShaperParams {
                drive: 6.0,
                harmonics: only_harmonic(0),
                mix: 1.0,
                output: 1.0,
            },
        );
        let input = sine_buffer(500.0, 1.0, 4_800);
        let reference: Vec<Sample> = input.channel(0).to_vec();
        let out = run(&mut node, input);
        let max_diff = out
            .channel(0)
            .iter()
            .skip(1_000)
            .zip(reference.iter().skip(1_000))
            .fold(0.0_f32, |m, (o, i)| m.max((o - i).abs()));
        assert!(max_diff > 0.1, "max_diff={max_diff}");
    }

    #[test]
    fn dc_from_even_harmonic_is_removed() {
        // A DC input through T_2 = 2x^2 - 1 is a constant; the DC blocker drives
        // the steady-state output to zero.
        let mut node = ChebyshevShaperNode::new(
            SR,
            ChannelLayout::Mono,
            ChebyshevShaperParams {
                drive: 1.0,
                harmonics: only_harmonic(1),
                mix: 1.0,
                output: 1.0,
            },
        );
        let frames = SR as usize; // one second lets the 20 Hz blocker settle
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        input.set_active_frames(frames);
        for s in input.channel_mut(0).iter_mut() {
            *s = 0.5;
        }
        let out = run(&mut node, input);
        let tail = out.channel(0)[frames - 1];
        assert!(tail.abs() < 1e-2, "tail={tail}");
    }

    #[test]
    fn non_finite_input_treated_as_silence() {
        let mut node =
            ChebyshevShaperNode::new(SR, ChannelLayout::Mono, ChebyshevShaperParams::default());
        let mut input = sine_buffer(300.0, 1.0, 512);
        input.channel_mut(0)[100] = Sample::NAN;
        input.channel_mut(0)[200] = Sample::INFINITY;
        let out = run(&mut node, input);
        assert!(out.channel(0).iter().all(|s| s.is_finite()));
    }

    #[test]
    fn reset_clears_state() {
        let mut node = ChebyshevShaperNode::new(
            SR,
            ChannelLayout::Mono,
            ChebyshevShaperParams {
                drive: 1.0,
                harmonics: only_harmonic(1),
                mix: 1.0,
                output: 1.0,
            },
        );
        let _ = run(&mut node, sine_buffer(300.0, 1.0, 2_000));
        node.reset();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 256);
        input.set_active_frames(256);
        let out = run(&mut node, input);
        // After reset the DC-blocker memory is clear, so silence stays silent.
        assert!(out.channel(0).iter().all(|&s| s.abs() < 1e-6));
    }

    #[test]
    fn set_macros_update_targets_and_clamp() {
        let mut node =
            ChebyshevShaperNode::new(SR, ChannelLayout::Mono, ChebyshevShaperParams::default());
        node.set_drive(100.0, Ramp::Immediate);
        node.set_mix(-1.0, Ramp::Immediate);
        node.set_output(0.25, Ramp::Immediate);
        assert_eq!(node.drive(), MAX_DRIVE);
        assert_eq!(node.mix(), 0.0);
        assert!((node.output() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn set_harmonics_updates_and_clamps() {
        let mut node =
            ChebyshevShaperNode::new(SR, ChannelLayout::Mono, ChebyshevShaperParams::default());
        node.set_harmonics(&[0.5, 100.0, -100.0]);
        let h = node.harmonics();
        assert!((h[0] - 0.5).abs() < 1e-6);
        assert_eq!(h[1], MAX_HARMONIC_GAIN);
        assert_eq!(h[2], -MAX_HARMONIC_GAIN);
        // Untouched tail weights stay at their default of zero.
        assert!(h[3..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn set_harmonics_handles_nonfinite() {
        let mut node =
            ChebyshevShaperNode::new(SR, ChannelLayout::Mono, ChebyshevShaperParams::default());
        node.set_harmonics(&[Sample::NAN, Sample::INFINITY]);
        let h = node.harmonics();
        assert_eq!(h[0], 0.0);
        assert_eq!(h[1], 0.0);
    }

    #[test]
    fn all_zero_harmonics_full_wet_is_silence() {
        let mut node = ChebyshevShaperNode::new(
            SR,
            ChannelLayout::Mono,
            ChebyshevShaperParams {
                drive: 1.0,
                harmonics: [0.0; MAX_HARMONICS],
                mix: 1.0,
                output: 1.0,
            },
        );
        let out = run(&mut node, sine_buffer(300.0, 1.0, 1_000));
        assert!(out.channel(0).iter().all(|&s| s.abs() < 1e-6));
    }

    #[test]
    fn extra_output_channels_are_silenced() {
        let mut node =
            ChebyshevShaperNode::new(SR, ChannelLayout::Mono, ChebyshevShaperParams::default());
        let input = sine_buffer(300.0, 1.0, 512);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 512);
        output.set_active_frames(512);
        let c = ctx(512);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);
        assert!(outputs[0].channel(1).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn stereo_correlated_input_is_coherent() {
        let mut node = ChebyshevShaperNode::new(
            SR,
            ChannelLayout::Stereo,
            ChebyshevShaperParams {
                drive: 2.0,
                harmonics: only_harmonic(1),
                mix: 1.0,
                output: 1.0,
            },
        );
        let frames = 4_000;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, frames);
        input.set_active_frames(frames);
        for n in 0..frames {
            let t = n as Sample / SR as Sample;
            let v = ops_sin(TAU * 300.0 * t);
            input.channel_mut(0)[n] = v;
            input.channel_mut(1)[n] = v;
        }
        let out = run(&mut node, input);
        for n in 0..frames {
            assert!((out.channel(0)[n] - out.channel(1)[n]).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_buffer_is_a_no_op() {
        let mut node =
            ChebyshevShaperNode::new(SR, ChannelLayout::Mono, ChebyshevShaperParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 1);
        input.set_active_frames(0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 1);
        output.set_active_frames(0);
        let c = ctx(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }

    #[test]
    fn latency_is_zero() {
        let node =
            ChebyshevShaperNode::new(SR, ChannelLayout::Mono, ChebyshevShaperParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn output_is_finite_and_denormal_free() {
        let mut node = ChebyshevShaperNode::new(
            SR,
            ChannelLayout::Mono,
            ChebyshevShaperParams {
                drive: 2.0,
                harmonics: only_harmonic(1),
                mix: 1.0,
                output: 1.0,
            },
        );
        let out = run(&mut node, sine_buffer(300.0, 1.0, 4_000));
        for &s in out.channel(0) {
            assert!(s.is_finite());
            assert!(s == 0.0 || s.abs() >= f32::MIN_POSITIVE);
        }
    }
}
