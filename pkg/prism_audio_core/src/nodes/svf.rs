//! Topology-preserving state-variable filter (SVF) with simultaneous
//! low-/high-/band-pass, notch, peak, all-pass, bell, and shelving responses.
//!
//! Unlike the Direct Form I [`Biquad`](super::biquad::Biquad) (which realises
//! the RBJ cookbook), this module uses the *trapezoidal-integrated* (TPT) SVF
//! topology. A single bilinear pre-warp `g = tan(pi * fc / fs)` and the damping
//! `k = 1 / Q` drive two integrator states, and every filter response is read
//! out from the same two internal signals by a cheap output mix. The defining
//! property of this topology is that its state stays bounded and artefact-free
//! when the cutoff is swept quickly, which is exactly what modulated filters
//! (auto-wah, envelope filters, filter LFOs) require and where a naive Direct
//! Form I biquad would click or blow up.
//!
//! # Signal model
//!
//! Per input sample `v0`, with per-channel integrator memories `ic1eq` and
//! `ic2eq`, the update is
//!
//! ```text
//! v3 = v0 - ic2eq
//! v1 = a1 * ic1eq + a2 * v3
//! v2 = ic2eq + a2 * ic1eq + a3 * v3
//! ic1eq = 2 * v1 - ic1eq
//! ic2eq = 2 * v2 - ic2eq
//! ```
//!
//! where `a1 = 1 / (1 + g * (g + k))`, `a2 = g * a1`, `a3 = g * a2`. The two
//! internal signals are the band-pass (`v1`) and low-pass (`v2`) outputs; every
//! requested response is a linear mix `y = m0 * v0 + m1 * v1 + m2 * v2` whose
//! coefficients are baked into [`SvfCoeffs`] at design time.
//!
//! # Provenance
//!
//! The trapezoidal SVF is standard public virtual-analogue signal-processing
//! knowledge, documented in Andrew Simper (Cytomic), "Solving the continuous
//! SVF equations using trapezoidal integration and equivalent currents"
//! (2013), and in Vadim Zavalishin, "The Art of VA Filter Design". The
//! coefficient and output-mix formulas below are re-derived from those public
//! papers. This file contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented mathematics.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::buffer::AudioBuffer;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// The response shape read out of the shared SVF core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SvfKind {
    /// 12 dB/octave low-pass.
    LowPass,
    /// 12 dB/octave high-pass.
    HighPass,
    /// Constant-peak-gain band-pass (peak gain equals `Q`).
    BandPass,
    /// Band-reject / notch.
    Notch,
    /// Resonant peak (low minus high), a symmetric bell without a gain term.
    Peak,
    /// All-pass: flat magnitude, frequency-dependent phase.
    AllPass,
    /// Parametric bell (peaking) EQ with `gain_db` boost/cut at `freq_hz`.
    Bell,
    /// Low shelf with `gain_db` boost/cut below the corner.
    LowShelf,
    /// High shelf with `gain_db` boost/cut above the corner.
    HighShelf,
}

/// Design coefficients for the trapezoidal SVF (`a` terms drive the
/// integrators; `m` terms select the output response).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SvfCoeffs {
    /// First integrator gain: `1 / (1 + g * (g + k))`.
    pub a1: Sample,
    /// Second integrator gain: `g * a1`.
    pub a2: Sample,
    /// Third integrator gain: `g * a2`.
    pub a3: Sample,
    /// Output mix weight applied to the raw input `v0`.
    pub m0: Sample,
    /// Output mix weight applied to the band-pass signal `v1`.
    pub m1: Sample,
    /// Output mix weight applied to the low-pass signal `v2`.
    pub m2: Sample,
}

impl SvfCoeffs {
    /// Designs coefficients for `kind` at `freq_hz` with quality `q` (and
    /// `gain_db` for the [`SvfKind::Bell`], [`SvfKind::LowShelf`], and
    /// [`SvfKind::HighShelf`] kinds).
    ///
    /// `freq_hz` is clamped into the open Nyquist band and `q` to a small
    /// positive minimum, so the design is always numerically valid regardless
    /// of the requested values.
    #[must_use]
    pub fn design(
        kind: SvfKind,
        sample_rate: u32,
        freq_hz: Sample,
        q: Sample,
        gain_db: Sample,
    ) -> Self {
        let sr = sample_rate.max(1) as Sample;
        let f0 = freq_hz.clamp(1.0, sr * 0.499);
        let q = q.max(1.0e-4);
        let w = core::f32::consts::PI * f0 / sr;

        // Select the bilinear pre-warp `g`, damping `k`, and output mix.
        let (g, k, m0, m1, m2) = match kind {
            SvfKind::LowPass => (ops::tan(w), 1.0 / q, 0.0, 0.0, 1.0),
            SvfKind::HighPass => {
                let k = 1.0 / q;
                (ops::tan(w), k, 1.0, -k, -1.0)
            }
            SvfKind::BandPass => (ops::tan(w), 1.0 / q, 0.0, 1.0, 0.0),
            SvfKind::Notch => {
                let k = 1.0 / q;
                (ops::tan(w), k, 1.0, -k, 0.0)
            }
            SvfKind::Peak => {
                let k = 1.0 / q;
                (ops::tan(w), k, 1.0, -k, -2.0)
            }
            SvfKind::AllPass => {
                let k = 1.0 / q;
                (ops::tan(w), k, 1.0, -2.0 * k, 0.0)
            }
            SvfKind::Bell => {
                let a = pow10(gain_db / 40.0);
                let k = 1.0 / (q * a);
                (ops::tan(w), k, 1.0, k * (a * a - 1.0), 0.0)
            }
            SvfKind::LowShelf => {
                let a = pow10(gain_db / 40.0);
                let g = ops::tan(w) / ops::sqrt(a);
                (g, 1.0 / q, 1.0, (1.0 / q) * (a - 1.0), a * a - 1.0)
            }
            SvfKind::HighShelf => {
                let a = pow10(gain_db / 40.0);
                let g = ops::tan(w) * ops::sqrt(a);
                (g, 1.0 / q, a * a, (1.0 / q) * (1.0 - a) * a, 1.0 - a * a)
            }
        };

        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        Self {
            a1,
            a2,
            a3,
            m0,
            m1,
            m2,
        }
    }
}

/// A reusable, allocation-free trapezoidal SVF with per-channel integrator
/// state.
///
/// This is the shared DSP core wrapped by [`SvfNode`]. All state is
/// pre-allocated at construction, so [`Svf::process_inplace`] and [`Svf::tick`]
/// are real-time safe (no allocation, no locking, no panic).
#[derive(Debug, Clone)]
pub struct Svf {
    coeffs: SvfCoeffs,
    /// Per-channel integrator memory `[ic1eq, ic2eq]`.
    state: Vec<[Sample; 2]>,
}

impl Svf {
    /// Builds a filter with the given coefficients sized for `channels`
    /// (at least one channel is always allocated).
    #[must_use]
    pub fn new(coeffs: SvfCoeffs, channels: usize) -> Self {
        Self {
            coeffs,
            state: vec![[0.0; 2]; channels.max(1)],
        }
    }

    /// Builds a filter designed from parametric settings.
    #[must_use]
    pub fn from_params(
        kind: SvfKind,
        sample_rate: u32,
        freq_hz: Sample,
        q: Sample,
        gain_db: Sample,
        channels: usize,
    ) -> Self {
        Self::new(
            SvfCoeffs::design(kind, sample_rate, freq_hz, q, gain_db),
            channels,
        )
    }

    /// Replaces the coefficients, preserving integrator state (click-free).
    #[inline]
    pub fn set_coeffs(&mut self, coeffs: SvfCoeffs) {
        self.coeffs = coeffs;
    }

    /// Returns the current coefficients.
    #[inline]
    #[must_use]
    pub fn coeffs(&self) -> SvfCoeffs {
        self.coeffs
    }

    /// Returns the number of channels the filter carries state for.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.state.len()
    }

    /// Processes a single sample `x0` on channel `ch` and returns the output.
    ///
    /// Non-finite inputs are treated as silence so a stray `NaN` or infinity
    /// can never poison the recursive integrator state.
    #[inline]
    pub fn tick(&mut self, ch: usize, x0: Sample) -> Sample {
        let v0 = if x0.is_finite() { x0 } else { 0.0 };
        let c = self.coeffs;
        let st = &mut self.state[ch];
        let (ic1eq, ic2eq) = (st[0], st[1]);

        let v3 = v0 - ic2eq;
        let v1 = c.a1 * ic1eq + c.a2 * v3;
        let v2 = ic2eq + c.a2 * ic1eq + c.a3 * v3;
        st[0] = flush_denormal(2.0 * v1 - ic1eq);
        st[1] = flush_denormal(2.0 * v2 - ic2eq);

        flush_denormal(c.m0 * v0 + c.m1 * v1 + c.m2 * v2)
    }

    /// Filters `buf` in place with the current coefficients, one pass per
    /// channel. Channels beyond the filter's configured width are untouched.
    pub fn process_inplace(&mut self, buf: &mut AudioBuffer) {
        let channels = buf.channels().min(self.state.len());
        let c = self.coeffs;
        for ch in 0..channels {
            let st = &mut self.state[ch];
            let (mut ic1eq, mut ic2eq) = (st[0], st[1]);
            for d in buf.channel_mut(ch).iter_mut() {
                let x0 = *d;
                let v0 = if x0.is_finite() { x0 } else { 0.0 };
                let v3 = v0 - ic2eq;
                let v1 = c.a1 * ic1eq + c.a2 * v3;
                let v2 = ic2eq + c.a2 * ic1eq + c.a3 * v3;
                ic1eq = flush_denormal(2.0 * v1 - ic1eq);
                ic2eq = flush_denormal(2.0 * v2 - ic2eq);
                *d = flush_denormal(c.m0 * v0 + c.m1 * v1 + c.m2 * v2);
            }
            st[0] = ic1eq;
            st[1] = ic2eq;
        }
    }

    /// Clears the per-channel integrator memory.
    pub fn reset(&mut self) {
        for s in &mut self.state {
            *s = [0.0; 2];
        }
    }
}

/// Construction parameters for an [`SvfNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SvfParams {
    /// Response shape.
    pub kind: SvfKind,
    /// Corner / centre frequency in Hz.
    pub freq_hz: Sample,
    /// Quality factor (resonance). Higher `q` is a narrower, taller peak.
    pub q: Sample,
    /// Boost/cut in dB for the bell and shelving kinds (ignored otherwise).
    pub gain_db: Sample,
}

impl Default for SvfParams {
    fn default() -> Self {
        Self {
            kind: SvfKind::LowPass,
            freq_hz: 1_000.0,
            q: core::f32::consts::FRAC_1_SQRT_2,
            gain_db: 0.0,
        }
    }
}

/// A per-channel trapezoidal SVF node (input port 0 -> output port 0).
///
/// The cutoff frequency is driven through a [`Smoothed`] parameter, so
/// [`SvfNode::set_freq_hz`] can be automated at audio rate without clicks; when
/// the smoother is moving, the coefficients are re-derived every sample, which
/// the TPT topology handles without the instability a Direct Form I biquad
/// would exhibit under fast sweeps.
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::svf::{SvfNode, SvfParams};
///
/// let mut node = SvfNode::new(SvfParams::default(), 48_000, 1);
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
/// input.set_active_frames(4);
/// input.channel_mut(0)[0] = 1.0;
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, 4);
/// output.set_active_frames(4);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 4, playhead: 0 };
/// let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
/// node.process(&ctx, &mut io);
/// assert!(output.channel(0).iter().all(|s| s.is_finite()));
/// ```
#[derive(Debug, Clone)]
pub struct SvfNode {
    svf: Svf,
    sample_rate: u32,
    kind: SvfKind,
    q: Sample,
    gain_db: Sample,
    cutoff: Smoothed,
}

impl SvfNode {
    /// Builds a node from `params` for a `channels`-wide signal at
    /// `sample_rate` Hz.
    #[must_use]
    pub fn new(params: SvfParams, sample_rate: u32, channels: usize) -> Self {
        let svf = Svf::from_params(
            params.kind,
            sample_rate,
            params.freq_hz,
            params.q,
            params.gain_db,
            channels,
        );
        Self {
            svf,
            sample_rate,
            kind: params.kind,
            q: params.q,
            gain_db: params.gain_db,
            cutoff: Smoothed::new(params.freq_hz),
        }
    }

    /// Sets the target cutoff frequency, glided over `ramp` to avoid zipper
    /// noise. The change takes effect on the next [`AudioNode::process`] call.
    #[inline]
    pub fn set_freq_hz(&mut self, freq_hz: Sample, ramp: Ramp) {
        self.cutoff.set_target(freq_hz, ramp);
    }

    /// Replaces the response shape (takes effect on the next process call).
    #[inline]
    pub fn set_kind(&mut self, kind: SvfKind) {
        self.kind = kind;
    }

    /// Replaces the quality factor (takes effect on the next process call).
    #[inline]
    pub fn set_q(&mut self, q: Sample) {
        self.q = q;
    }

    /// Replaces the bell/shelf gain in dB (takes effect on the next process
    /// call).
    #[inline]
    pub fn set_gain_db(&mut self, gain_db: Sample) {
        self.gain_db = gain_db;
    }

    #[inline]
    fn design(&self, freq_hz: Sample) -> SvfCoeffs {
        SvfCoeffs::design(self.kind, self.sample_rate, freq_hz, self.q, self.gain_db)
    }
}

impl AudioNode for SvfNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        output.copy_from(input);

        if self.cutoff.is_settled() {
            // Stable cutoff: derive once and run the block with fixed coeffs.
            let coeffs = self.design(self.cutoff.current());
            self.svf.set_coeffs(coeffs);
            self.svf.process_inplace(output);
        } else {
            // Sweeping cutoff: re-derive per sample so the sweep is exact and
            // stable across every channel from the same smoothed value.
            let channels = output.channels().min(self.svf.channels());
            let frames = output.active_frames();
            for i in 0..frames {
                let f = self.cutoff.next_sample();
                self.svf.set_coeffs(self.design(f));
                for ch in 0..channels {
                    let x = output.channel(ch)[i];
                    let y = self.svf.tick(ch, x);
                    output.channel_mut(ch)[i] = y;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.svf.reset();
    }
}

#[inline]
fn pow10(x: Sample) -> Sample {
    ops::exp(x * core::f32::consts::LN_10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use core::f32::consts::{FRAC_1_SQRT_2, PI};

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: 48_000,
            frames,
            playhead: 0,
        }
    }

    /// Runs a unit-amplitude sine of `freq` through `coeffs` and returns the
    /// steady-state magnitude gain (output RMS / input RMS over the tail half).
    fn sine_gain(coeffs: SvfCoeffs, sr: u32, freq: Sample, n: usize) -> Sample {
        let mut svf = Svf::new(coeffs, 1);
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, n);
        buf.set_active_frames(n);
        let w = 2.0 * PI * freq / (sr as Sample);
        for (i, s) in buf.channel_mut(0).iter_mut().enumerate() {
            *s = ops::sin(w * i as Sample);
        }
        svf.process_inplace(&mut buf);
        let half = n / 2;
        let tail = &buf.channel(0)[half..];
        let sum: Sample = tail.iter().map(|v| v * v).sum();
        let rms = ops::sqrt(sum / tail.len() as Sample);
        // Input RMS of a unit sine is 1/sqrt(2).
        rms * ops::sqrt(2.0)
    }

    /// Returns the steady-state DC gain (final sample of a constant-1 input).
    fn dc_gain(coeffs: SvfCoeffs, n: usize) -> Sample {
        let mut svf = Svf::new(coeffs, 1);
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, n);
        buf.set_active_frames(n);
        for s in buf.channel_mut(0).iter_mut() {
            *s = 1.0;
        }
        svf.process_inplace(&mut buf);
        buf.channel(0)[n - 1]
    }

    #[test]
    fn lowpass_passes_dc_blocks_highs() {
        let c = SvfCoeffs::design(SvfKind::LowPass, 48_000, 1_000.0, FRAC_1_SQRT_2, 0.0);
        assert!(dc_gain(c, 8_192) > 0.98);
        assert!(sine_gain(c, 48_000, 16_000.0, 8_192) < 0.2);
    }

    #[test]
    fn highpass_blocks_dc_passes_highs() {
        let c = SvfCoeffs::design(SvfKind::HighPass, 48_000, 1_000.0, FRAC_1_SQRT_2, 0.0);
        assert!(dc_gain(c, 8_192).abs() < 0.05);
        assert!(sine_gain(c, 48_000, 16_000.0, 8_192) > 0.7);
    }

    #[test]
    fn bandpass_peaks_at_centre() {
        let c = SvfCoeffs::design(SvfKind::BandPass, 48_000, 1_000.0, 4.0, 0.0);
        let centre = sine_gain(c, 48_000, 1_000.0, 16_384);
        let low = sine_gain(c, 48_000, 100.0, 16_384);
        let high = sine_gain(c, 48_000, 10_000.0, 16_384);
        assert!(centre > low * 3.0);
        assert!(centre > high * 3.0);
    }

    #[test]
    fn notch_rejects_centre() {
        let c = SvfCoeffs::design(SvfKind::Notch, 48_000, 1_000.0, FRAC_1_SQRT_2, 0.0);
        assert!(sine_gain(c, 48_000, 1_000.0, 16_384) < 0.15);
        assert!(dc_gain(c, 8_192) > 0.9);
    }

    #[test]
    fn allpass_is_magnitude_flat() {
        let c = SvfCoeffs::design(SvfKind::AllPass, 48_000, 1_000.0, FRAC_1_SQRT_2, 0.0);
        for &f in &[120.0, 1_000.0, 6_000.0] {
            let g = sine_gain(c, 48_000, f, 16_384);
            assert!((g - 1.0).abs() < 0.1, "allpass gain at {f} Hz was {g}");
        }
    }

    #[test]
    fn peak_is_finite_and_resonant() {
        let c = SvfCoeffs::design(SvfKind::Peak, 48_000, 1_000.0, 4.0, 0.0);
        let g = sine_gain(c, 48_000, 1_000.0, 16_384);
        assert!(g.is_finite());
        assert!(g > 0.0);
    }

    #[test]
    fn bell_boosts_and_cuts_at_centre() {
        let boost = SvfCoeffs::design(SvfKind::Bell, 48_000, 1_000.0, 2.0, 12.0);
        let cut = SvfCoeffs::design(SvfKind::Bell, 48_000, 1_000.0, 2.0, -12.0);
        let g_boost = sine_gain(boost, 48_000, 1_000.0, 16_384);
        let g_cut = sine_gain(cut, 48_000, 1_000.0, 16_384);
        // +12 dB ~= x3.98, -12 dB ~= x0.25.
        assert!(g_boost > 3.0 && g_boost < 5.0, "bell boost gain {g_boost}");
        assert!(g_cut < 0.4, "bell cut gain {g_cut}");
        // Far from centre the bell is roughly unity.
        assert!((sine_gain(boost, 48_000, 60.0, 16_384) - 1.0).abs() < 0.2);
    }

    #[test]
    fn low_shelf_lifts_lows_only() {
        let c = SvfCoeffs::design(SvfKind::LowShelf, 48_000, 1_000.0, FRAC_1_SQRT_2, 12.0);
        // DC gain ~= 10^(12/20) ~= 3.98.
        let dc = dc_gain(c, 16_384);
        assert!(dc > 3.5 && dc < 4.5, "low shelf DC gain {dc}");
        // Well above the corner it settles back toward unity.
        assert!((sine_gain(c, 48_000, 16_000.0, 16_384) - 1.0).abs() < 0.25);
    }

    #[test]
    fn high_shelf_lifts_highs_only() {
        let c = SvfCoeffs::design(SvfKind::HighShelf, 48_000, 1_000.0, FRAC_1_SQRT_2, 12.0);
        assert!((dc_gain(c, 16_384) - 1.0).abs() < 0.1);
        assert!(sine_gain(c, 48_000, 16_000.0, 16_384) > 3.0);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut svf = Svf::from_params(SvfKind::LowPass, 48_000, 1_000.0, 2.0, 0.0, 2);
        let mut buf = AudioBuffer::new(ChannelLayout::Stereo, 64);
        buf.set_active_frames(64);
        svf.process_inplace(&mut buf);
        for ch in 0..buf.channels() {
            assert!(buf.channel(ch).iter().all(|s| *s == 0.0));
        }
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut svf = Svf::from_params(SvfKind::BandPass, 48_000, 1_000.0, 8.0, 0.0, 1);
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, 8);
        buf.set_active_frames(8);
        let data = buf.channel_mut(0);
        data[0] = Sample::NAN;
        data[1] = Sample::INFINITY;
        data[2] = Sample::NEG_INFINITY;
        data[3] = 1.0;
        svf.process_inplace(&mut buf);
        assert!(buf.channel(0).iter().all(|s| s.is_finite()));
    }

    #[test]
    fn reset_makes_processing_reproducible() {
        let c = SvfCoeffs::design(SvfKind::LowPass, 48_000, 800.0, 3.0, 0.0);
        let mut svf = Svf::new(c, 1);
        let make = || {
            let mut buf = AudioBuffer::new(ChannelLayout::Mono, 32);
            buf.set_active_frames(32);
            for (i, s) in buf.channel_mut(0).iter_mut().enumerate() {
                *s = ops::sin(0.1 * i as Sample);
            }
            buf
        };
        let mut a = make();
        svf.process_inplace(&mut a);
        svf.reset();
        let mut b = make();
        svf.process_inplace(&mut b);
        for (x, y) in a.channel(0).iter().zip(b.channel(0).iter()) {
            assert!((x - y).abs() < 1.0e-6);
        }
    }

    #[test]
    fn channels_are_independent() {
        let c = SvfCoeffs::design(SvfKind::LowPass, 48_000, 1_000.0, 1.0, 0.0);
        let mut svf = Svf::new(c, 2);
        let mut buf = AudioBuffer::new(ChannelLayout::Stereo, 16);
        buf.set_active_frames(16);
        for s in buf.channel_mut(0).iter_mut() {
            *s = 1.0;
        }
        // Channel 1 left silent.
        svf.process_inplace(&mut buf);
        assert!(buf.channel(0).iter().any(|s| *s != 0.0));
        assert!(buf.channel(1).iter().all(|s| *s == 0.0));
    }

    #[test]
    fn extreme_params_do_not_panic() {
        for &(f, q) in &[(0.0, 0.0), (-100.0, -5.0), (1.0e9, 1.0e9)] {
            let c = SvfCoeffs::design(SvfKind::Peak, 48_000, f, q, 6.0);
            assert!(c.a1.is_finite() && c.a2.is_finite() && c.a3.is_finite());
            let g = sine_gain(c, 48_000, 1_000.0, 1_024);
            assert!(g.is_finite());
        }
    }

    #[test]
    fn zero_frames_is_a_no_op() {
        let mut svf = Svf::from_params(SvfKind::LowPass, 48_000, 1_000.0, 1.0, 0.0, 1);
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, 8);
        buf.set_active_frames(0);
        svf.process_inplace(&mut buf);
        // No panic and no writes past the (empty) active region.
        assert_eq!(buf.active_frames(), 0);
    }

    #[test]
    fn node_smoothed_sweep_stays_finite() {
        let mut node = SvfNode::new(
            SvfParams {
                kind: SvfKind::LowPass,
                freq_hz: 400.0,
                q: 6.0,
                gain_db: 0.0,
            },
            48_000,
            2,
        );
        node.set_freq_hz(6_000.0, Ramp::linear_seconds(0.01, 48_000));

        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 256);
        input.set_active_frames(256);
        for ch in 0..2 {
            for (i, s) in input.channel_mut(ch).iter_mut().enumerate() {
                *s = ops::sin(0.05 * i as Sample);
            }
        }
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 256);
        output.set_active_frames(256);

        let c = ctx(256);
        // Two blocks: the first sweeps, the second is settled.
        for _ in 0..2 {
            let mut io =
                ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
            node.process(&c, &mut io);
            for ch in 0..2 {
                assert!(output.channel(ch).iter().all(|s| s.is_finite()));
            }
        }
    }

    #[test]
    fn node_reset_clears_state() {
        let mut node = SvfNode::new(SvfParams::default(), 48_000, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 16);
        input.set_active_frames(16);
        input.channel_mut(0)[0] = 1.0;
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 16);
        output.set_active_frames(16);
        let c = ctx(16);
        {
            let mut io =
                ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
            node.process(&c, &mut io);
        }
        node.reset();
        // Fresh silence through a reset filter yields silence.
        let silent = AudioBuffer::new(ChannelLayout::Mono, 16);
        let mut silent = silent;
        silent.set_active_frames(16);
        let mut out2 = AudioBuffer::new(ChannelLayout::Mono, 16);
        out2.set_active_frames(16);
        let mut io = ProcessIo::new(core::slice::from_ref(&silent), core::slice::from_mut(&mut out2));
        node.process(&c, &mut io);
        assert!(out2.channel(0).iter().all(|s| *s == 0.0));
    }
}
