//! Modal resonator bank: a parallel bank of high-`Q` resonant modes that
//! imposes the ringing "body" of a struck or plucked object onto whatever
//! signal excites it, the classic building block of modal / physical-modelling
//! synthesis for impacts, bells, plates, and resonant bodies.
//!
//! # Model
//!
//! A vibrating object rings at a discrete set of modes, each an exponentially
//! decaying sinusoid at its own frequency, amplitude, and decay time. This node
//! realises that model directly: it runs the input in parallel through up to
//! [`MAX_MODES`] two-pole resonant band-pass sections and sums their outputs.
//! Each mode `m` has a centre frequency `f_m`, a `-60 dB` decay time `t60_m`,
//! and a linear gain `g_m`. The decay time is mapped to the resonator quality
//! factor by `Q = pi * f * t60 / ln(1000)`, the standard relation between a
//! resonant band-pass bandwidth and its reverberation time, so a longer `t60`
//! produces a narrower, longer-ringing mode. An impulse at the input therefore
//! produces the sum of the modes' exponentially decaying sinusoids, exactly the
//! modal-synthesis response.
//!
//! # Relationship
//!
//! Each mode's biquad coefficients are built with the crate's shared
//! [`BiquadCoeffs::design`](crate::nodes::biquad::BiquadCoeffs::design) (RBJ
//! cookbook) `BandPass` section, so this node does not restate the transfer
//! function. Unlike a
//! [`ParametricEqNode`](crate::nodes::effects::parametric_eq::ParametricEqNode)
//! or [`GraphicEqNode`](crate::nodes::effects::graphic_eq::GraphicEqNode),
//! which cascade a few sections in **series** to shape a spectrum, this node
//! sums many high-`Q` sections in **parallel** and drives them hard enough to
//! ring, which no series equaliser produces. It also differs from
//! [`CombResonatorNode`](crate::nodes::effects::comb_resonator::CombResonatorNode)
//! (a single feedback comb with harmonically spaced peaks) and from
//! [`FormantFilterNode`](crate::nodes::effects::formant_filter::FormantFilterNode)
//! (a small fixed set of vowel formants): a modal bank places an arbitrary set
//! of inharmonic, independently decaying resonances. Because the parallel sum
//! must be accumulated per sample, the per-mode difference equation is stepped
//! directly (a per-channel Direct Form I state) rather than through
//! [`Biquad::process_inplace`](crate::nodes::biquad::Biquad::process_inplace),
//! which filters a whole buffer through one section.
//!
//! # Real-time contract
//!
//! The coefficient and gain tables and every per-channel, per-mode state slot
//! are allocated once at construction (sized for [`MAX_MODES`]).
//! [`ModalResonatorNode::process`] performs no allocation, locking, or panic on
//! the hot path; non-finite input samples are treated as silence and each
//! mode's state is denormal-flushed so the resonators cannot stall on
//! denormals. The node adds no latency ([`ModalResonatorNode::latency_frames`]
//! returns zero).
//!
//! # Provenance
//!
//! Modal synthesis (a resonant object modelled as a parallel bank of decaying
//! sinusoidal modes), the two-pole resonant band-pass filter, the RBJ cookbook
//! biquad, and the band-pass `t60`-to-`Q` relation are standard, publicly
//! documented classic DSP and physical-modelling techniques (for example
//! Adrien's modal formulation and Smith's digital-waveguide / resonator
//! literature). This is pure classic DSP with no AI or ML. This module contains
//! **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance
//! Audio, or Web Audio source or derived code**; only the widely documented
//! resonator and biquad formulas are used.

use alloc::{vec, vec::Vec};
use core::f32::consts::{LN_10, PI};

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};
use crate::nodes::biquad::{BiquadCoeffs, BiquadKind};

/// Maximum number of resonant modes the bank can hold.
pub const MAX_MODES: usize = 32;

/// Smallest resonator quality factor a mode may resolve to.
pub const MIN_MODE_Q: Sample = 0.5;

/// Largest resonator quality factor a mode may resolve to; bounds how long a
/// mode can ring and keeps the design numerically comfortable.
pub const MAX_MODE_Q: Sample = 2_000.0;

/// Largest `-60 dB` decay time, in seconds, a mode may request.
pub const MAX_DECAY_S: Sample = 20.0;

/// Default dry / wet mix.
pub const DEFAULT_MODAL_MIX: Sample = 0.5;

/// One resonant mode: a centre frequency, a `-60 dB` decay time, and a gain.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ModalMode {
    /// Mode centre frequency in hertz.
    pub freq_hz: Sample,
    /// Mode `-60 dB` decay time in seconds (longer rings longer).
    pub decay_s: Sample,
    /// Linear output gain for this mode.
    pub gain: Sample,
}

impl Default for ModalMode {
    fn default() -> Self {
        Self {
            freq_hz: 440.0,
            decay_s: 1.0,
            gain: 1.0,
        }
    }
}

impl ModalMode {
    /// Returns the mode with every field clamped into range and non-finite
    /// values replaced by defaults.
    #[must_use]
    fn sanitised(self, sample_rate: u32) -> Self {
        let d = Self::default();
        let nyquist = 0.5 * sample_rate.max(1) as Sample;
        let freq_hz = if self.freq_hz.is_finite() {
            self.freq_hz.clamp(1.0, nyquist * 0.999)
        } else {
            d.freq_hz.min(nyquist * 0.999)
        };
        let decay_s = if self.decay_s.is_finite() {
            self.decay_s.clamp(0.0, MAX_DECAY_S)
        } else {
            d.decay_s
        };
        let gain = if self.gain.is_finite() { self.gain } else { d.gain };
        Self {
            freq_hz,
            decay_s,
            gain,
        }
    }
}

/// Global parameters shared across every mode.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ModalResonatorParams {
    /// Dry / wet mix in `[0, 1]`: `0` is the dry input, `1` is the resonated
    /// bank only.
    pub mix: Sample,
    /// Overall output trim in decibels applied to the summed wet bank.
    pub output_gain_db: Sample,
}

impl Default for ModalResonatorParams {
    fn default() -> Self {
        Self {
            mix: DEFAULT_MODAL_MIX,
            output_gain_db: 0.0,
        }
    }
}

impl ModalResonatorParams {
    #[must_use]
    fn sanitised(self) -> Self {
        let d = Self::default();
        let mix = if self.mix.is_finite() {
            self.mix.clamp(0.0, 1.0)
        } else {
            d.mix
        };
        let output_gain_db = if self.output_gain_db.is_finite() {
            self.output_gain_db.clamp(-60.0, 36.0)
        } else {
            d.output_gain_db
        };
        Self { mix, output_gain_db }
    }
}

/// Maps a mode `-60 dB` decay time to a resonant band-pass quality factor.
fn mode_q(freq_hz: Sample, decay_s: Sample) -> Sample {
    if !decay_s.is_finite() || decay_s <= 0.0 {
        return MIN_MODE_Q;
    }
    // t60 = ln(1000) / (pi * bandwidth) and Q = f / bandwidth, so
    // Q = pi * f * t60 / ln(1000) (ln(1000) == 3 * ln(10)).
    let q = PI * freq_hz * decay_s / (3.0 * LN_10);
    q.clamp(MIN_MODE_Q, MAX_MODE_Q)
}

/// A parallel bank of resonant modes (modal-synthesis resonator).
#[derive(Clone, Debug)]
pub struct ModalResonatorNode {
    channels: usize,
    mode_count: usize,
    coeffs: [BiquadCoeffs; MAX_MODES],
    gains: [Sample; MAX_MODES],
    mix: Sample,
    output_gain: Sample,
    // Per-channel input history (shared by every parallel mode).
    in_x1: Vec<Sample>,
    in_x2: Vec<Sample>,
    // Per-channel, per-mode output history (`channels * MAX_MODES`).
    y1: Vec<Sample>,
    y2: Vec<Sample>,
}

impl ModalResonatorNode {
    /// Builds a modal resonator for `channels` channels at `sample_rate` from
    /// the given `modes` (at most [`MAX_MODES`] are used) and global `params`.
    ///
    /// ```
    /// use prism_audio_core::nodes::effects::modal_resonator::{
    ///     ModalMode, ModalResonatorNode, ModalResonatorParams,
    /// };
    /// use prism_audio_core::graph::AudioNode;
    ///
    /// let modes = [
    ///     ModalMode { freq_hz: 440.0, decay_s: 1.5, gain: 1.0 },
    ///     ModalMode { freq_hz: 880.0, decay_s: 0.8, gain: 0.5 },
    /// ];
    /// let node = ModalResonatorNode::new(48_000, 2, &modes, ModalResonatorParams::default());
    /// // A pure filter bank adds no latency.
    /// assert_eq!(node.latency_frames(), 0);
    /// ```
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        modes: &[ModalMode],
        params: ModalResonatorParams,
    ) -> Self {
        let channels = channels.max(1);
        let params = params.sanitised();
        let mut node = Self {
            channels,
            mode_count: 0,
            coeffs: [BiquadCoeffs::default(); MAX_MODES],
            gains: [0.0; MAX_MODES],
            mix: params.mix,
            output_gain: db_to_linear(params.output_gain_db),
            in_x1: vec![0.0; channels],
            in_x2: vec![0.0; channels],
            y1: vec![0.0; channels * MAX_MODES],
            y2: vec![0.0; channels * MAX_MODES],
        };
        node.set_modes(sample_rate, modes);
        node
    }

    /// Number of active modes in the bank.
    #[must_use]
    pub fn mode_count(&self) -> usize {
        self.mode_count
    }

    /// Replaces the active mode set, rebuilding each mode's coefficients. At
    /// most [`MAX_MODES`] modes are kept. The ringing state of surviving mode
    /// slots is preserved for a click-free transition.
    pub fn set_modes(&mut self, sample_rate: u32, modes: &[ModalMode]) {
        let count = modes.len().min(MAX_MODES);
        for (slot, mode) in modes.iter().take(count).enumerate() {
            let mode = mode.sanitised(sample_rate);
            let q = mode_q(mode.freq_hz, mode.decay_s);
            self.coeffs[slot] =
                BiquadCoeffs::design(BiquadKind::BandPass, sample_rate, mode.freq_hz, q, 0.0);
            self.gains[slot] = mode.gain;
        }
        // Silence any slots that just left the active set.
        for slot in count..self.mode_count {
            self.gains[slot] = 0.0;
            for ch in 0..self.channels {
                self.y1[ch * MAX_MODES + slot] = 0.0;
                self.y2[ch * MAX_MODES + slot] = 0.0;
            }
        }
        self.mode_count = count;
    }

    /// Replaces the global mix / output trim.
    pub fn set_params(&mut self, params: ModalResonatorParams) {
        let params = params.sanitised();
        self.mix = params.mix;
        self.output_gain = db_to_linear(params.output_gain_db);
    }
}

impl AudioNode for ModalResonatorNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let available = output.channels().min(input.channels());
        let channels = available.min(self.channels);
        let frames = output.active_frames().min(input.active_frames());

        let mode_count = self.mode_count;
        let mix = self.mix;
        let dry = 1.0 - mix;
        let wet_gain = mix * self.output_gain;
        let coeffs = &self.coeffs;
        let gains = &self.gains;
        let in_x1 = &mut self.in_x1;
        let in_x2 = &mut self.in_x2;
        let y1 = &mut self.y1;
        let y2 = &mut self.y2;

        for ch in 0..channels {
            let mode_base = ch * MAX_MODES;
            let mut x1 = in_x1[ch];
            let mut x2 = in_x2[ch];
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            for i in 0..frames {
                let x = src[i];
                let x = if x.is_finite() { x } else { 0.0 };

                let mut acc = 0.0;
                for m in 0..mode_count {
                    let c = &coeffs[m];
                    let idx = mode_base + m;
                    let yv = c.b0 * x + c.b1 * x1 + c.b2 * x2 - c.a1 * y1[idx] - c.a2 * y2[idx];
                    let yv = flush_denormal(yv);
                    y2[idx] = y1[idx];
                    y1[idx] = yv;
                    acc += gains[m] * yv;
                }

                x2 = x1;
                x1 = x;

                dst[i] = dry * x + wet_gain * acc;
            }
            in_x1[ch] = x1;
            in_x2[ch] = x2;
        }

        // Pass surplus channels (beyond the processed set) through untouched.
        for ch in channels..available {
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            dst[..frames].copy_from_slice(&src[..frames]);
        }
    }

    fn reset(&mut self) {
        for value in &mut self.in_x1 {
            *value = 0.0;
        }
        for value in &mut self.in_x2 {
            *value = 0.0;
        }
        for value in &mut self.y1 {
            *value = 0.0;
        }
        for value in &mut self.y2 {
            *value = 0.0;
        }
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use bevy_math::ops;
    use core::f32::consts::TAU;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    /// Streams a mono `signal` through the node in one block.
    fn run_mono(node: &mut ModalResonatorNode, signal: &[Sample]) -> Vec<Sample> {
        let len = signal.len();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len);
        input.channel_mut(0).copy_from_slice(signal);
        let output = AudioBuffer::new(ChannelLayout::Mono, len);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        outputs[0].channel(0).to_vec()
    }

    /// Streams a stereo pair through the node in one block.
    fn run_stereo(
        node: &mut ModalResonatorNode,
        left: &[Sample],
        right: &[Sample],
    ) -> (Vec<Sample>, Vec<Sample>) {
        let len = left.len().min(right.len());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.channel_mut(0)[..len].copy_from_slice(&left[..len]);
        input.channel_mut(1)[..len].copy_from_slice(&right[..len]);
        let output = AudioBuffer::new(ChannelLayout::Stereo, len);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        (outputs[0].channel(0).to_vec(), outputs[0].channel(1).to_vec())
    }

    fn impulse(len: usize) -> Vec<Sample> {
        let mut v = vec![0.0; len];
        if !v.is_empty() {
            v[0] = 1.0;
        }
        v
    }

    fn sine(freq: Sample, amp: Sample, len: usize) -> Vec<Sample> {
        (0..len)
            .map(|n| amp * ops::sin(TAU * freq * n as Sample / SR as Sample))
            .collect()
    }

    fn rms(samples: &[Sample]) -> Sample {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: Sample = samples.iter().map(|&x| x * x).sum();
        ops::sqrt(sum / samples.len() as Sample)
    }

    /// Goertzel single-bin power estimate at `freq`.
    fn goertzel(samples: &[Sample], freq: Sample) -> Sample {
        let omega = TAU * freq / SR as Sample;
        let coeff = 2.0 * ops::cos(omega);
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for &x in samples {
            let s0 = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        s1 * s1 + s2 * s2 - coeff * s1 * s2
    }

    fn single_mode(freq: Sample, decay_s: Sample) -> ModalResonatorNode {
        let modes = [ModalMode {
            freq_hz: freq,
            decay_s,
            gain: 1.0,
        }];
        let params = ModalResonatorParams {
            mix: 1.0,
            output_gain_db: 0.0,
        };
        ModalResonatorNode::new(SR, 1, &modes, params)
    }

    #[test]
    fn latency_is_zero() {
        let node = single_mode(1_000.0, 0.5);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = single_mode(1_000.0, 0.5);
        let out = run_mono(&mut node, &vec![0.0; 512]);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn tone_stays_finite() {
        let mut node = single_mode(1_000.0, 0.5);
        let out = run_mono(&mut node, &sine(440.0, 0.5, 4_096));
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn non_finite_input_is_silenced() {
        let mut node = single_mode(1_000.0, 0.5);
        let mut sig = sine(440.0, 0.3, 1_024);
        sig[10] = Sample::NAN;
        sig[20] = Sample::INFINITY;
        sig[30] = Sample::NEG_INFINITY;
        let out = run_mono(&mut node, &sig);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = single_mode(1_000.0, 0.5);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 1);
        input.set_active_frames(0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 1);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }

    #[test]
    fn mix_zero_is_dry_passthrough() {
        let modes = [ModalMode {
            freq_hz: 1_000.0,
            decay_s: 0.5,
            gain: 1.0,
        }];
        let params = ModalResonatorParams {
            mix: 0.0,
            output_gain_db: 0.0,
        };
        let mut node = ModalResonatorNode::new(SR, 1, &modes, params);
        let sig = sine(440.0, 0.5, 512);
        let out = run_mono(&mut node, &sig);
        for (o, i) in out.iter().zip(sig.iter()) {
            assert!((o - i).abs() < 1e-6);
        }
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let modes = [
            ModalMode {
                freq_hz: 1.0e9,
                decay_s: 1.0e6,
                gain: 1.0e6,
            },
            ModalMode {
                freq_hz: -500.0,
                decay_s: -3.0,
                gain: -2.0,
            },
        ];
        let params = ModalResonatorParams {
            mix: 5.0,
            output_gain_db: 200.0,
        };
        let mut node = ModalResonatorNode::new(SR, 1, &modes, params);
        let out = run_mono(&mut node, &sine(440.0, 0.9, 2_048));
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn non_finite_params_fall_back_to_defaults() {
        let params = ModalResonatorParams {
            mix: Sample::NAN,
            output_gain_db: Sample::INFINITY,
        };
        let node = single_mode(1_000.0, 0.5);
        let mut node = node;
        node.set_params(params);
        assert_eq!(node.mix, DEFAULT_MODAL_MIX);
        assert!(node.output_gain.is_finite());
    }

    #[test]
    fn non_finite_mode_is_sanitised() {
        let modes = [ModalMode {
            freq_hz: Sample::NAN,
            decay_s: Sample::INFINITY,
            gain: Sample::NAN,
        }];
        let params = ModalResonatorParams {
            mix: 1.0,
            output_gain_db: 0.0,
        };
        let mut node = ModalResonatorNode::new(SR, 1, &modes, params);
        let out = run_mono(&mut node, &impulse(1_024));
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn impulse_decays() {
        let mut node = single_mode(1_000.0, 0.5);
        let out = run_mono(&mut node, &impulse(SR as usize));
        assert!(out.iter().all(|x| x.is_finite()));
        let early = rms(&out[0..2_400]);
        let late = rms(&out[21_600..24_000]);
        assert!(early > late * 4.0, "early {early} late {late}");
    }

    #[test]
    fn frequency_selectivity() {
        let mut node = single_mode(1_000.0, 0.5);
        let out = run_mono(&mut node, &impulse(SR as usize));
        let on_band = goertzel(&out, 1_000.0);
        let off_band = goertzel(&out, 4_000.0);
        assert!(
            on_band > off_band * 20.0,
            "on {on_band} off {off_band}"
        );
    }

    #[test]
    fn mode_q_increases_with_decay() {
        let short = mode_q(1_000.0, 0.5);
        let long = mode_q(1_000.0, 2.0);
        assert!(long > short);
        assert_eq!(mode_q(1_000.0, 0.0), MIN_MODE_Q);
        assert_eq!(mode_q(1_000.0, -1.0), MIN_MODE_Q);
    }

    #[test]
    fn stereo_channels_are_independent() {
        let modes = [ModalMode {
            freq_hz: 1_000.0,
            decay_s: 0.5,
            gain: 1.0,
        }];
        let params = ModalResonatorParams {
            mix: 1.0,
            output_gain_db: 0.0,
        };
        let mut node = ModalResonatorNode::new(SR, 2, &modes, params);
        let left = impulse(4_096);
        let right = vec![0.0; 4_096];
        let (lo, ro) = run_stereo(&mut node, &left, &right);
        assert!(rms(&lo) > 0.0);
        assert!(ro.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn surplus_channels_pass_through() {
        // Node sized for one channel but fed a stereo buffer: channel 1 passes.
        let mut node = single_mode(1_000.0, 0.5);
        let left = impulse(256);
        let right = sine(440.0, 0.7, 256);
        let (_lo, ro) = run_stereo(&mut node, &left, &right);
        for (o, i) in ro.iter().zip(right.iter()) {
            assert!((o - i).abs() < 1e-6);
        }
    }

    #[test]
    fn reset_restores_fresh_state() {
        let mut node = single_mode(1_000.0, 0.5);
        let sig = sine(440.0, 0.6, 2_048);
        let _ = run_mono(&mut node, &sig);
        node.reset();
        let after = run_mono(&mut node, &sig);
        let mut fresh = single_mode(1_000.0, 0.5);
        let baseline = run_mono(&mut fresh, &sig);
        let max_err = after
            .iter()
            .zip(baseline.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-6, "max_err {max_err}");
    }

    #[test]
    fn set_modes_changes_mode_count() {
        let mut node = single_mode(1_000.0, 0.5);
        assert_eq!(node.mode_count(), 1);
        let modes = [
            ModalMode {
                freq_hz: 300.0,
                decay_s: 0.5,
                gain: 1.0,
            },
            ModalMode {
                freq_hz: 600.0,
                decay_s: 0.5,
                gain: 1.0,
            },
            ModalMode {
                freq_hz: 900.0,
                decay_s: 0.5,
                gain: 1.0,
            },
        ];
        node.set_modes(SR, &modes);
        assert_eq!(node.mode_count(), 3);
    }

    #[test]
    fn set_params_changes_mix() {
        let mut node = single_mode(1_000.0, 0.5);
        node.set_params(ModalResonatorParams {
            mix: 0.25,
            output_gain_db: -6.0,
        });
        assert!((node.mix - 0.25).abs() < 1e-6);
    }

    #[test]
    fn default_params_are_in_range() {
        let p = ModalResonatorParams::default();
        assert_eq!(p.mix, DEFAULT_MODAL_MIX);
        assert_eq!(p.output_gain_db, 0.0);
    }

    #[test]
    fn mode_bank_truncates_to_max() {
        let modes: Vec<ModalMode> = (0..(MAX_MODES + 8))
            .map(|i| ModalMode {
                freq_hz: 100.0 + i as Sample * 50.0,
                decay_s: 0.3,
                gain: 0.1,
            })
            .collect();
        let node = ModalResonatorNode::new(SR, 1, &modes, ModalResonatorParams::default());
        assert_eq!(node.mode_count(), MAX_MODES);
    }
}
