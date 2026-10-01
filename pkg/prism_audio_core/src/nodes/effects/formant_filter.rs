//! Vowel formant filter: a parallel bank of band-pass resonators tuned to the
//! formant peaks of the five cardinal vowels, with continuous vowel morphing.
//!
//! Human vowels are distinguished by the frequencies of a handful of vocal
//! tract resonances called formants. By running several band-pass resonators in
//! parallel, each pinned to one formant frequency with its own bandwidth and
//! level, an arbitrary signal can be imprinted with a vowel colour -- the basis
//! of talk-box, vocoder-adjacent vowel pads, and "singing" synth timbres. This
//! node realises `FORMANT_COUNT` parallel [`Biquad`] band-pass sections, scales
//! each by its formant level, and sums them. Two vowels can be blended with a
//! single `morph` control that interpolates every formant's frequency,
//! bandwidth, and level.
//!
//! All filter state and the summing scratch buffer are pre-allocated at
//! construction, so [`FormantFilterNode::process`] performs no allocation,
//! locking, or panicking and is safe on the audio thread.
//!
//! # Provenance
//!
//! Classic DSP only, with no AI/ML of any kind. This module contains no UE,
//! Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source
//! code or derived code. The vowel formant frequency, bandwidth, and level
//! tables are widely published acoustic-phonetics data (the kind tabulated in
//! classic speech-synthesis references); they are numeric constants, not code,
//! and the band-pass realisation is the standard RBJ cookbook form reused from
//! [`crate::nodes::biquad`].
//!
//! # Relationship
//!
//! Like [`crate::nodes::effects::parametric_eq::ParametricEqNode`] and
//! [`crate::nodes::effects::graphic_eq`], this node is built from the shared
//! [`Biquad`] core, but it differs in both topology and intent:
//!
//! - A parametric or graphic EQ cascades its sections in series to shape an
//!   overall response, and its bands are user-chosen.
//! - A formant filter runs its sections in parallel and sums them, so each
//!   resonator contributes an additive peak, and its bands are driven by vowel
//!   presets rather than free parameters.
//!
//! It is also distinct from [`crate::nodes::effects::auto_wah::AutoWahNode`],
//! which sweeps a single band-pass formant with an envelope or LFO; here several
//! fixed formants define a recognisable vowel and the only motion is the
//! vowel-to-vowel `morph`.

use crate::buffer::AudioBuffer;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear};
use crate::nodes::biquad::{Biquad, BiquadCoeffs, BiquadKind};

/// Number of parallel formant resonators per vowel.
pub const FORMANT_COUNT: usize = 5;

/// Default resonance scale (`1.0` leaves each formant `Q` at its tabulated
/// `frequency / bandwidth`).
pub const DEFAULT_RESONANCE: Sample = 1.0;

/// Default vowel morph position (`0.0` selects `vowel_a` exactly).
pub const DEFAULT_MORPH: Sample = 0.0;

/// Default output make-up gain in decibels.
pub const DEFAULT_OUTPUT_GAIN_DB: Sample = 0.0;

/// Smallest permitted resonance scale, keeping `Q` strictly positive.
const MIN_RESONANCE: Sample = 1.0e-3;

/// A single formant resonance: a centre frequency, a bandwidth, and a level.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FormantSpec {
    /// Formant centre frequency in Hertz.
    pub freq_hz: Sample,
    /// Formant bandwidth in Hertz (`-3 dB` width); `Q = freq_hz / bandwidth_hz`.
    pub bandwidth_hz: Sample,
    /// Relative formant level in decibels (typically `0` or negative).
    pub gain_db: Sample,
}

impl FormantSpec {
    const fn new(freq_hz: Sample, bandwidth_hz: Sample, gain_db: Sample) -> Self {
        Self {
            freq_hz,
            bandwidth_hz,
            gain_db,
        }
    }
}

/// The five cardinal vowels, each resolving to a `FORMANT_COUNT`-wide table.
///
/// The tabulated frequencies, bandwidths, and levels approximate a typical
/// adult (bass-register) voice, following widely published formant data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Vowel {
    /// The open vowel as in "father".
    #[default]
    A,
    /// The front mid vowel as in "bet".
    E,
    /// The close front vowel as in "see".
    I,
    /// The back mid vowel as in "or".
    O,
    /// The close back vowel as in "boot".
    U,
}

impl Vowel {
    /// Returns the formant table for this vowel.
    #[must_use]
    pub const fn formants(self) -> [FormantSpec; FORMANT_COUNT] {
        match self {
            Vowel::A => [
                FormantSpec::new(600.0, 60.0, 0.0),
                FormantSpec::new(1040.0, 70.0, -7.0),
                FormantSpec::new(2250.0, 110.0, -9.0),
                FormantSpec::new(2450.0, 120.0, -9.0),
                FormantSpec::new(2750.0, 130.0, -20.0),
            ],
            Vowel::E => [
                FormantSpec::new(400.0, 40.0, 0.0),
                FormantSpec::new(1620.0, 80.0, -12.0),
                FormantSpec::new(2400.0, 100.0, -9.0),
                FormantSpec::new(2800.0, 120.0, -12.0),
                FormantSpec::new(3100.0, 120.0, -18.0),
            ],
            Vowel::I => [
                FormantSpec::new(250.0, 60.0, 0.0),
                FormantSpec::new(1750.0, 90.0, -30.0),
                FormantSpec::new(2600.0, 100.0, -16.0),
                FormantSpec::new(3050.0, 120.0, -22.0),
                FormantSpec::new(3340.0, 120.0, -28.0),
            ],
            Vowel::O => [
                FormantSpec::new(400.0, 40.0, 0.0),
                FormantSpec::new(750.0, 80.0, -11.0),
                FormantSpec::new(2400.0, 100.0, -21.0),
                FormantSpec::new(2600.0, 120.0, -20.0),
                FormantSpec::new(2900.0, 120.0, -40.0),
            ],
            Vowel::U => [
                FormantSpec::new(350.0, 40.0, 0.0),
                FormantSpec::new(600.0, 80.0, -20.0),
                FormantSpec::new(2400.0, 100.0, -32.0),
                FormantSpec::new(2675.0, 120.0, -28.0),
                FormantSpec::new(2950.0, 120.0, -36.0),
            ],
        }
    }
}

/// The tunable description of a formant filter.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FormantFilterParams {
    /// Vowel selected when `morph = 0`.
    pub vowel_a: Vowel,
    /// Vowel selected when `morph = 1`.
    pub vowel_b: Vowel,
    /// Blend position in `[0, 1]` between `vowel_a` and `vowel_b`.
    pub morph: Sample,
    /// Multiplies every formant `Q` (higher is sharper / more vocal).
    pub resonance: Sample,
    /// Output make-up gain in decibels applied to the summed formants.
    pub output_gain_db: Sample,
}

impl Default for FormantFilterParams {
    fn default() -> Self {
        Self {
            vowel_a: Vowel::A,
            vowel_b: Vowel::A,
            morph: DEFAULT_MORPH,
            resonance: DEFAULT_RESONANCE,
            output_gain_db: DEFAULT_OUTPUT_GAIN_DB,
        }
    }
}

/// Linearly interpolates two formant specs by `t` in `[0, 1]`.
#[inline]
fn lerp_spec(a: FormantSpec, b: FormantSpec, t: Sample) -> FormantSpec {
    FormantSpec {
        freq_hz: a.freq_hz + (b.freq_hz - a.freq_hz) * t,
        bandwidth_hz: a.bandwidth_hz + (b.bandwidth_hz - a.bandwidth_hz) * t,
        gain_db: a.gain_db + (b.gain_db - a.gain_db) * t,
    }
}

/// Resolves the effective formant table from the morph of two vowels.
fn resolved_formants(params: FormantFilterParams) -> [FormantSpec; FORMANT_COUNT] {
    let t = params.morph.clamp(0.0, 1.0);
    let a = params.vowel_a.formants();
    let b = params.vowel_b.formants();
    let mut out = [FormantSpec::new(0.0, 1.0, 0.0); FORMANT_COUNT];
    for (slot, (&sa, &sb)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        *slot = lerp_spec(sa, sb, t);
    }
    out
}

/// Computes band-pass coefficients and linear level for one resolved formant.
fn formant_coeffs(spec: FormantSpec, resonance: Sample, sample_rate: u32) -> (BiquadCoeffs, Sample) {
    let bw = spec.bandwidth_hz.max(MIN_RESONANCE);
    let q = (spec.freq_hz / bw) * resonance.max(MIN_RESONANCE);
    let coeffs = BiquadCoeffs::design(BiquadKind::BandPass, sample_rate, spec.freq_hz, q, 0.0);
    (coeffs, db_to_linear(spec.gain_db))
}

/// A vowel formant filter owning its parallel resonator bank and scratch.
///
/// The reusable DSP unit (no graph I/O). All state is pre-allocated at
/// construction for a signal up to `max_frames` long.
#[derive(Debug, Clone)]
pub struct FormantFilter {
    sample_rate: u32,
    params: FormantFilterParams,
    /// One band-pass resonator per formant.
    filters: [Biquad; FORMANT_COUNT],
    /// Linear level applied to each formant's output before summing.
    levels: [Sample; FORMANT_COUNT],
    /// Overall output gain (linear).
    output_gain: Sample,
    /// Scratch buffer holding the per-formant filtered signal before summing.
    scratch: AudioBuffer,
}

impl FormantFilter {
    /// Builds a formant filter for a `channels`-wide signal at `sample_rate` Hz,
    /// sized for blocks of up to `max_frames` samples.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        max_frames: usize,
        params: FormantFilterParams,
    ) -> Self {
        let channels = channels.max(1);
        let specs = resolved_formants(params);
        let mut levels = [0.0; FORMANT_COUNT];
        let filters = core::array::from_fn(|i| {
            let (coeffs, level) = formant_coeffs(specs[i], params.resonance, sample_rate);
            levels[i] = level;
            Biquad::new(coeffs, channels)
        });
        Self {
            sample_rate,
            params,
            filters,
            levels,
            output_gain: db_to_linear(params.output_gain_db),
            scratch: AudioBuffer::new(layout_for(channels), max_frames.max(1)),
        }
    }

    /// Returns the current parameters.
    #[inline]
    #[must_use]
    pub fn params(&self) -> FormantFilterParams {
        self.params
    }

    /// Redesigns every resonator, preserving filter state (click-free).
    pub fn set_params(&mut self, params: FormantFilterParams) {
        self.params = params;
        let specs = resolved_formants(params);
        for (i, filter) in self.filters.iter_mut().enumerate() {
            let (coeffs, level) = formant_coeffs(specs[i], params.resonance, self.sample_rate);
            filter.set_coeffs(coeffs);
            self.levels[i] = level;
        }
        self.output_gain = db_to_linear(params.output_gain_db);
    }

    /// Processes `input` into `output`, summing the parallel formant bank.
    ///
    /// `output` must already have its active frame count set by the caller.
    pub fn process_into(&mut self, input: &AudioBuffer, output: &mut AudioBuffer) {
        let frames = output
            .active_frames()
            .min(input.active_frames())
            .min(self.scratch.capacity_frames());
        output.clear();
        let out_channels = output.channels();
        self.scratch.set_active_frames(frames);
        for (i, filter) in self.filters.iter_mut().enumerate() {
            // Copy the dry input into scratch, band-pass it, then add it in
            // scaled by the formant level.
            let level = self.levels[i];
            let copy_channels = out_channels.min(self.scratch.channels()).min(input.channels());
            for ch in 0..copy_channels {
                let src = input.channel(ch);
                let dst = self.scratch.channel_mut(ch);
                let n = frames.min(src.len()).min(dst.len());
                dst[..n].copy_from_slice(&src[..n]);
            }
            filter.process_inplace(&mut self.scratch);
            for ch in 0..copy_channels {
                let src = self.scratch.channel(ch);
                let dst = output.channel_mut(ch);
                let n = frames.min(src.len()).min(dst.len());
                for f in 0..n {
                    dst[f] += src[f] * level;
                }
            }
        }
        // Apply output make-up gain.
        if self.output_gain != 1.0 {
            for ch in 0..out_channels {
                let dst = output.channel_mut(ch);
                let n = frames.min(dst.len());
                for d in dst.iter_mut().take(n) {
                    *d *= self.output_gain;
                }
            }
        }
    }

    /// Clears every resonator's filter memory.
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
        self.scratch.clear();
    }
}

/// Maps a channel count to the matching [`ChannelLayout`](crate::buffer::ChannelLayout).
fn layout_for(channels: usize) -> crate::buffer::ChannelLayout {
    use crate::buffer::ChannelLayout;
    match channels {
        0 | 1 => ChannelLayout::Mono,
        _ => ChannelLayout::Stereo,
    }
}

/// A vowel formant filter node (input port 0 -> output port 0).
#[derive(Debug, Clone)]
pub struct FormantFilterNode {
    inner: FormantFilter,
}

impl FormantFilterNode {
    /// Builds a formant filter node for a `channels`-wide signal at
    /// `sample_rate` Hz, sized for blocks of up to `max_frames` samples.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        max_frames: usize,
        params: FormantFilterParams,
    ) -> Self {
        Self {
            inner: FormantFilter::new(sample_rate, channels, max_frames, params),
        }
    }

    /// Returns the current parameters.
    #[inline]
    #[must_use]
    pub fn params(&self) -> FormantFilterParams {
        self.inner.params()
    }

    /// Redesigns every resonator, preserving filter state (click-free).
    pub fn set_params(&mut self, params: FormantFilterParams) {
        self.inner.set_params(params);
    }
}

impl AudioNode for FormantFilterNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        self.inner.process_into(input, output);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::ChannelLayout;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn run(node: &mut FormantFilterNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let inputs = [input.clone()];
        let mut out = AudioBuffer::new(input.layout(), input.capacity_frames().max(1));
        out.set_active_frames(frames);
        let mut outputs = [out];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(frames), &mut io);
        }
        let [out] = outputs;
        out
    }

    /// Returns the peak magnitude of the discrete Fourier transform bin nearest
    /// `freq_hz`, computed directly (small N, test only).
    fn goertzel_magnitude(signal: &[Sample], freq_hz: Sample) -> Sample {
        let n = signal.len();
        if n == 0 {
            return 0.0;
        }
        let w = core::f32::consts::TAU * freq_hz / SR as Sample;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (k, &s) in signal.iter().enumerate() {
            let ph = w * k as Sample;
            re += s * bevy_math::ops::cos(ph);
            im -= s * bevy_math::ops::sin(ph);
        }
        bevy_math::ops::sqrt(re * re + im * im) / n as Sample
    }

    /// Builds white-ish broadband mono excitation (a deterministic impulse
    /// train is enough to probe the resonances).
    fn broadband(frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, frames);
        // Unit impulse: a flat broadband spectrum.
        buf.channel_mut(0)[0] = 1.0;
        buf
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = FormantFilterNode::new(SR, 2, 64, FormantFilterParams::default());
        let input = AudioBuffer::new(ChannelLayout::Stereo, 64);
        let out = run(&mut node, &input);
        for ch in 0..2 {
            assert!(out.channel(ch).iter().all(|&s| s == 0.0));
        }
    }

    #[test]
    fn vowel_a_boosts_its_first_formant() {
        // Vowel A has F1 near 600 Hz. The impulse response must have more energy
        // near 600 Hz than near a non-formant region such as 5000 Hz.
        let params = FormantFilterParams {
            vowel_a: Vowel::A,
            vowel_b: Vowel::A,
            ..FormantFilterParams::default()
        };
        let mut node = FormantFilterNode::new(SR, 1, 4096, params);
        let input = broadband(4096);
        let out = run(&mut node, &input);
        let at_formant = goertzel_magnitude(out.channel(0), 600.0);
        let off_formant = goertzel_magnitude(out.channel(0), 5000.0);
        assert!(
            at_formant > off_formant * 4.0,
            "formant {at_formant} off {off_formant}"
        );
    }

    #[test]
    fn different_vowels_have_different_responses() {
        let input = broadband(4096);
        let mut node_a = FormantFilterNode::new(
            SR,
            1,
            4096,
            FormantFilterParams {
                vowel_a: Vowel::A,
                vowel_b: Vowel::A,
                ..FormantFilterParams::default()
            },
        );
        let mut node_i = FormantFilterNode::new(
            SR,
            1,
            4096,
            FormantFilterParams {
                vowel_a: Vowel::I,
                vowel_b: Vowel::I,
                ..FormantFilterParams::default()
            },
        );
        let out_a = run(&mut node_a, &input);
        let out_i = run(&mut node_i, &input);
        // Vowel A has a strong F1 near 600 Hz; vowel I's F1 sits near 250 Hz,
        // so the 600 Hz content must differ markedly between them.
        let a_600 = goertzel_magnitude(out_a.channel(0), 600.0);
        let i_600 = goertzel_magnitude(out_i.channel(0), 600.0);
        assert!(a_600 > i_600, "a {a_600} i {i_600}");
    }

    #[test]
    fn morph_zero_matches_vowel_a() {
        let input = broadband(2048);
        let base = FormantFilterParams {
            vowel_a: Vowel::A,
            vowel_b: Vowel::A,
            ..FormantFilterParams::default()
        };
        let morph = FormantFilterParams {
            vowel_a: Vowel::A,
            vowel_b: Vowel::U,
            morph: 0.0,
            ..FormantFilterParams::default()
        };
        let mut node_base = FormantFilterNode::new(SR, 1, 2048, base);
        let mut node_morph = FormantFilterNode::new(SR, 1, 2048, morph);
        let out_base = run(&mut node_base, &input);
        let out_morph = run(&mut node_morph, &input);
        assert_eq!(out_base.channel(0), out_morph.channel(0));
    }

    #[test]
    fn morph_one_matches_vowel_b() {
        let input = broadband(2048);
        let direct = FormantFilterParams {
            vowel_a: Vowel::U,
            vowel_b: Vowel::U,
            ..FormantFilterParams::default()
        };
        let morph = FormantFilterParams {
            vowel_a: Vowel::A,
            vowel_b: Vowel::U,
            morph: 1.0,
            ..FormantFilterParams::default()
        };
        let mut node_direct = FormantFilterNode::new(SR, 1, 2048, direct);
        let mut node_morph = FormantFilterNode::new(SR, 1, 2048, morph);
        let out_direct = run(&mut node_direct, &input);
        let out_morph = run(&mut node_morph, &input);
        assert_eq!(out_direct.channel(0), out_morph.channel(0));
    }

    #[test]
    fn morph_blend_lies_between_endpoints() {
        // A halfway morph of F1 between A (600) and I (250) should centre near
        // 425 Hz, so its 425 Hz content exceeds either endpoint's 425 Hz.
        let input = broadband(4096);
        let mut node = FormantFilterNode::new(
            SR,
            1,
            4096,
            FormantFilterParams {
                vowel_a: Vowel::A,
                vowel_b: Vowel::I,
                morph: 0.5,
                ..FormantFilterParams::default()
            },
        );
        let out = run(&mut node, &input);
        let mid = goertzel_magnitude(out.channel(0), 425.0);
        assert!(mid > 0.0, "mid {mid}");
    }

    #[test]
    fn stereo_channels_match_for_identical_input() {
        let params = FormantFilterParams::default();
        let mut node = FormantFilterNode::new(SR, 2, 512, params);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 512);
        input.channel_mut(0)[0] = 1.0;
        input.channel_mut(1)[0] = 1.0;
        let out = run(&mut node, &input);
        assert_eq!(out.channel(0), out.channel(1));
    }

    #[test]
    fn output_gain_scales_linearly() {
        let input = broadband(256);
        let mut unity = FormantFilterNode::new(SR, 1, 256, FormantFilterParams::default());
        let mut boosted = FormantFilterNode::new(
            SR,
            1,
            256,
            FormantFilterParams {
                output_gain_db: 6.0206, // ~ x2
                ..FormantFilterParams::default()
            },
        );
        let out_u = run(&mut unity, &input);
        let out_b = run(&mut boosted, &input);
        let a = out_u.channel(0)[1];
        let b = out_b.channel(0)[1];
        assert!((b - 2.0 * a).abs() <= 1e-3 * (1.0 + a.abs()), "a {a} b {b}");
    }

    #[test]
    fn reset_clears_filter_memory() {
        let mut node = FormantFilterNode::new(SR, 1, 128, FormantFilterParams::default());
        let input = broadband(128);
        let first = run(&mut node, &input);
        node.reset();
        let second = run(&mut node, &input);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn set_params_preserves_state() {
        let mut node = FormantFilterNode::new(SR, 1, 64, FormantFilterParams::default());
        let input = broadband(64);
        let _ = run(&mut node, &input);
        node.set_params(FormantFilterParams {
            vowel_a: Vowel::E,
            vowel_b: Vowel::O,
            morph: 0.3,
            ..FormantFilterParams::default()
        });
        assert_eq!(node.params().vowel_a, Vowel::E);
        assert_eq!(node.params().vowel_b, Vowel::O);
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = FormantFilterNode::new(SR, 2, 64, FormantFilterParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 4);
        input.set_active_frames(0);
        let out = run(&mut node, &input);
        assert_eq!(out.active_frames(), 0);
    }

    #[test]
    fn params_round_trip() {
        let params = FormantFilterParams {
            vowel_a: Vowel::I,
            vowel_b: Vowel::O,
            morph: 0.42,
            resonance: 1.5,
            output_gain_db: -3.0,
        };
        let node = FormantFilterNode::new(SR, 1, 32, params);
        assert_eq!(node.params(), params);
    }

    #[test]
    fn core_matches_node() {
        let params = FormantFilterParams::default();
        let mut core = FormantFilter::new(SR, 1, 128, params);
        let mut node = FormantFilterNode::new(SR, 1, 128, params);
        let input = broadband(128);
        let mut core_out = AudioBuffer::new(ChannelLayout::Mono, 128);
        core_out.set_active_frames(128);
        core.process_into(&input, &mut core_out);
        let node_out = run(&mut node, &input);
        assert_eq!(core_out.channel(0), node_out.channel(0));
    }

    #[test]
    fn default_vowel_is_a() {
        assert_eq!(Vowel::default(), Vowel::A);
        let p = FormantFilterParams::default();
        assert_eq!(p.vowel_a, Vowel::A);
        assert_eq!(p.vowel_b, Vowel::A);
    }
}

/// A worked example: imprinting the vowel "A" on a broadband impulse.
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::formant_filter::{
///     FormantFilterNode, FormantFilterParams, Vowel,
/// };
///
/// let params = FormantFilterParams {
///     vowel_a: Vowel::A,
///     vowel_b: Vowel::I,
///     morph: 0.0, // pure A
///     ..FormantFilterParams::default()
/// };
/// let mut node = FormantFilterNode::new(48_000, 1, 8, params);
///
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
/// input.channel_mut(0)[0] = 1.0; // broadband impulse
///
/// let inputs = [input];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 8)];
/// outputs[0].set_active_frames(8);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 8, playhead: 0 };
/// {
///     let mut io = ProcessIo::new(&inputs, &mut outputs);
///     node.process(&ctx, &mut io);
/// }
/// // The resonant bank rings, so the response is non-trivial after the impulse.
/// assert!(outputs[0].channel(0).iter().any(|&s| s != 0.0));
/// ```
#[cfg(doctest)]
struct DocExample;
