//! Channel vocoder: cross-synthesis that imprints the spectral envelope of a
//! modulator (typically a voice) onto a carrier (typically a synth), yielding
//! the classic "talking synth" / robot-choir timbre.
//!
//! A bank of `VOCODER_BANDS` constant-`Q` band-pass filters analyses the
//! modulator into log-spaced frequency bands. A per-band envelope follower
//! measures how much energy the modulator has in each band. A matching
//! band-pass bank splits the carrier into the same bands, each band is scaled
//! by the modulator's envelope for that band, and the scaled bands are summed.
//! The carrier therefore "speaks" with the moving formants of the modulator.
//!
//! The modulator is read from input port 0 and the carrier from input port 1
//! (mirroring the side-chain convention of
//! [`crate::nodes::dynamics::ducking`]). When no carrier is wired the node
//! emits silence.
//!
//! All filter state, envelope memory, and scratch buffers are pre-allocated at
//! construction, so [`VocoderNode::process`] performs no allocation, locking,
//! or panicking and is safe on the audio thread.
//!
//! # Provenance
//!
//! Classic DSP only, with no AI/ML of any kind. This module contains no UE,
//! Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source
//! code or derived code. The analysis/synthesis filter-bank vocoder dates to
//! Dudley's 1930s channel vocoder and is textbook signal processing; the
//! band-pass sections are the standard RBJ cookbook form reused from
//! [`crate::nodes::biquad`] and the per-band envelope follower uses the shared
//! one-pole time constant from
//! [`crate::nodes::dynamics::detector::time_to_coef`].
//!
//! # Relationship
//!
//! Like [`crate::nodes::effects::formant_filter::FormantFilterNode`], this node
//! is built from a parallel [`Biquad`] band-pass bank, but the two differ in
//! intent and signal flow:
//!
//! - A formant filter imprints *fixed, preset* vowel formants onto a single
//!   input; its band levels are static vowel-table constants.
//! - A vocoder imprints the *time-varying measured* envelope of a live
//!   modulator onto a separate carrier; its band levels move with the
//!   modulator and it requires two inputs.
//!
//! It is distinct from [`crate::nodes::effects::ring_modulator`] (a single
//! bipolar multiply with no filtering), from
//! [`crate::nodes::effects::auto_wah`] (one swept band), and from
//! [`crate::nodes::dynamics::multiband`] (which splits one signal to compress
//! each band, never cross-synthesising two signals).

use alloc::vec;
use alloc::vec::Vec;

use crate::buffer::{AudioBuffer, ChannelLayout};
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear};
use crate::nodes::biquad::{Biquad, BiquadCoeffs, BiquadKind};
use crate::nodes::dynamics::detector::time_to_coef;

/// Number of constant-`Q` analysis/synthesis bands.
pub const VOCODER_BANDS: usize = 16;

/// Default low edge of the band bank in Hertz.
pub const DEFAULT_LOW_HZ: Sample = 80.0;

/// Default high edge of the band bank in Hertz.
pub const DEFAULT_HIGH_HZ: Sample = 12_000.0;

/// Default envelope attack time in milliseconds.
pub const DEFAULT_ATTACK_MS: Sample = 2.0;

/// Default envelope release time in milliseconds.
pub const DEFAULT_RELEASE_MS: Sample = 15.0;

/// Default output make-up gain in decibels.
pub const DEFAULT_OUTPUT_GAIN_DB: Sample = 0.0;

/// Lowest permitted band edge, keeping the filter design well conditioned.
const MIN_LOW_HZ: Sample = 20.0;

/// The tunable description of a channel vocoder.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VocoderParams {
    /// Low edge of the band bank in Hertz.
    pub low_hz: Sample,
    /// High edge of the band bank in Hertz.
    pub high_hz: Sample,
    /// Envelope attack time in milliseconds (how fast bands open).
    pub attack_ms: Sample,
    /// Envelope release time in milliseconds (how fast bands close).
    pub release_ms: Sample,
    /// Output make-up gain in decibels applied to the summed bands.
    pub output_gain_db: Sample,
}

impl Default for VocoderParams {
    fn default() -> Self {
        Self {
            low_hz: DEFAULT_LOW_HZ,
            high_hz: DEFAULT_HIGH_HZ,
            attack_ms: DEFAULT_ATTACK_MS,
            release_ms: DEFAULT_RELEASE_MS,
            output_gain_db: DEFAULT_OUTPUT_GAIN_DB,
        }
    }
}

/// Sanitises the requested band range against the Nyquist limit, guaranteeing
/// `MIN_LOW_HZ <= low < high <= 0.49 * sample_rate` with `high >= 2 * low`.
fn sanitize_range(low_hz: Sample, high_hz: Sample, sample_rate: u32) -> (Sample, Sample) {
    let nyquist = (sample_rate.max(1) as Sample) * 0.49;
    let high = high_hz.clamp(MIN_LOW_HZ * 2.0, nyquist.max(MIN_LOW_HZ * 2.0));
    let low = low_hz.clamp(MIN_LOW_HZ, high * 0.5);
    (low, high)
}

/// Geometric ratio between adjacent band centres for the sanitised range.
#[inline]
fn band_ratio(low: Sample, high: Sample) -> Sample {
    // `low < high` is guaranteed by `sanitize_range`, so the ratio exceeds 1.
    bevy_math::ops::powf(high / low, 1.0 / (VOCODER_BANDS as Sample - 1.0))
}

/// Log-spaced band centre frequencies for the sanitised range.
fn band_centers(low: Sample, high: Sample) -> [Sample; VOCODER_BANDS] {
    let ratio = band_ratio(low, high);
    core::array::from_fn(|i| low * bevy_math::ops::powf(ratio, i as Sample))
}

/// Constant-`Q` quality factor matching the geometric band spacing.
fn band_q(low: Sample, high: Sample) -> Sample {
    let ratio = band_ratio(low, high);
    // Geometric band edges give bandwidth `f * (ratio - 1) / sqrt(ratio)`, so
    // `Q = f / bandwidth = sqrt(ratio) / (ratio - 1)`.
    bevy_math::ops::sqrt(ratio) / (ratio - 1.0)
}

/// Builds the `VOCODER_BANDS` band-pass coefficient set for a sanitised range.
fn band_coeffs(low: Sample, high: Sample, sample_rate: u32) -> [BiquadCoeffs; VOCODER_BANDS] {
    let centers = band_centers(low, high);
    let q = band_q(low, high);
    core::array::from_fn(|i| {
        BiquadCoeffs::design(BiquadKind::BandPass, sample_rate, centers[i], q, 0.0)
    })
}

/// Maps a channel count to the matching [`ChannelLayout`].
fn layout_for(channels: usize) -> ChannelLayout {
    match channels {
        0 | 1 => ChannelLayout::Mono,
        _ => ChannelLayout::Stereo,
    }
}

/// A channel vocoder owning its analysis and synthesis banks and scratch.
///
/// The reusable DSP unit (no graph I/O). All state is pre-allocated at
/// construction for a signal up to `max_frames` long across `channels`.
#[derive(Debug, Clone)]
pub struct Vocoder {
    sample_rate: u32,
    channels: usize,
    params: VocoderParams,
    /// Band-pass bank analysing the modulator.
    mod_bank: [Biquad; VOCODER_BANDS],
    /// Band-pass bank splitting the carrier.
    car_bank: [Biquad; VOCODER_BANDS],
    /// Envelope attack one-pole coefficient (larger = slower).
    attack_coef: Sample,
    /// Envelope release one-pole coefficient (larger = slower).
    release_coef: Sample,
    /// Overall output gain (linear).
    output_gain: Sample,
    /// Per-band, per-channel envelope memory (`band * channels + channel`).
    envelopes: Vec<Sample>,
    /// Scratch holding the band-filtered modulator.
    scratch_mod: AudioBuffer,
    /// Scratch holding the band-filtered carrier.
    scratch_car: AudioBuffer,
}

impl Vocoder {
    /// Builds a vocoder for a `channels`-wide signal at `sample_rate` Hz, sized
    /// for blocks of up to `max_frames` samples.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        max_frames: usize,
        params: VocoderParams,
    ) -> Self {
        let channels = channels.max(1);
        let (low, high) = sanitize_range(params.low_hz, params.high_hz, sample_rate);
        let coeffs = band_coeffs(low, high, sample_rate);
        let mod_bank = core::array::from_fn(|i| Biquad::new(coeffs[i], channels));
        let car_bank = core::array::from_fn(|i| Biquad::new(coeffs[i], channels));
        let layout = layout_for(channels);
        let cap = max_frames.max(1);
        Self {
            sample_rate,
            channels,
            params,
            mod_bank,
            car_bank,
            attack_coef: time_to_coef(params.attack_ms, sample_rate),
            release_coef: time_to_coef(params.release_ms, sample_rate),
            output_gain: db_to_linear(params.output_gain_db),
            envelopes: vec![0.0; VOCODER_BANDS * channels],
            scratch_mod: AudioBuffer::new(layout, cap),
            scratch_car: AudioBuffer::new(layout, cap),
        }
    }

    /// Returns the current parameters.
    #[inline]
    #[must_use]
    pub fn params(&self) -> VocoderParams {
        self.params
    }

    /// Returns the number of analysis/synthesis bands.
    #[inline]
    #[must_use]
    pub fn bands(&self) -> usize {
        VOCODER_BANDS
    }

    /// Redesigns both banks and envelope times, preserving filter and envelope
    /// state (click-free).
    pub fn set_params(&mut self, params: VocoderParams) {
        self.params = params;
        let (low, high) = sanitize_range(params.low_hz, params.high_hz, self.sample_rate);
        let coeffs = band_coeffs(low, high, self.sample_rate);
        for (i, (m, c)) in self.mod_bank.iter_mut().zip(self.car_bank.iter_mut()).enumerate() {
            m.set_coeffs(coeffs[i]);
            c.set_coeffs(coeffs[i]);
        }
        self.attack_coef = time_to_coef(params.attack_ms, self.sample_rate);
        self.release_coef = time_to_coef(params.release_ms, self.sample_rate);
        self.output_gain = db_to_linear(params.output_gain_db);
    }

    /// Cross-synthesises `carrier` with the band envelopes of `modulator`,
    /// writing the sum into `output`.
    ///
    /// `output` must already have its active frame count set by the caller.
    pub fn process_into(
        &mut self,
        modulator: &AudioBuffer,
        carrier: &AudioBuffer,
        output: &mut AudioBuffer,
    ) {
        let frames = output
            .active_frames()
            .min(modulator.active_frames())
            .min(carrier.active_frames())
            .min(self.scratch_mod.capacity_frames());
        output.clear();
        let active_ch = self
            .channels
            .min(modulator.channels())
            .min(carrier.channels())
            .min(output.channels());
        self.scratch_mod.set_active_frames(frames);
        self.scratch_car.set_active_frames(frames);
        let attack = self.attack_coef;
        let release = self.release_coef;
        for band in 0..VOCODER_BANDS {
            // Analyse the modulator and split the carrier into this band.
            for ch in 0..active_ch {
                let src = modulator.channel(ch);
                let dst = self.scratch_mod.channel_mut(ch);
                let n = frames.min(src.len()).min(dst.len());
                dst[..n].copy_from_slice(&src[..n]);
            }
            for ch in 0..active_ch {
                let src = carrier.channel(ch);
                let dst = self.scratch_car.channel_mut(ch);
                let n = frames.min(src.len()).min(dst.len());
                dst[..n].copy_from_slice(&src[..n]);
            }
            self.mod_bank[band].process_inplace(&mut self.scratch_mod);
            self.car_bank[band].process_inplace(&mut self.scratch_car);
            // Follow the modulator envelope and modulate the carrier band.
            for ch in 0..active_ch {
                let idx = band * self.channels + ch;
                let mut env = self.envelopes[idx];
                let m = self.scratch_mod.channel(ch);
                let c = self.scratch_car.channel(ch);
                let out = output.channel_mut(ch);
                let n = frames.min(m.len()).min(c.len()).min(out.len());
                for f in 0..n {
                    let rect = m[f].abs();
                    let coef = if rect > env { attack } else { release };
                    env = coef * env + (1.0 - coef) * rect;
                    out[f] += c[f] * env;
                }
                self.envelopes[idx] = env;
            }
        }
        if self.output_gain != 1.0 {
            for ch in 0..active_ch {
                let out = output.channel_mut(ch);
                let n = frames.min(out.len());
                for d in out.iter_mut().take(n) {
                    *d *= self.output_gain;
                }
            }
        }
    }

    /// Clears all filter memory and band envelopes.
    pub fn reset(&mut self) {
        for (m, c) in self.mod_bank.iter_mut().zip(self.car_bank.iter_mut()) {
            m.reset();
            c.reset();
        }
        for e in &mut self.envelopes {
            *e = 0.0;
        }
        self.scratch_mod.clear();
        self.scratch_car.clear();
    }
}

/// A channel vocoder node (modulator on input 0, carrier on input 1 -> output 0).
///
/// When no carrier is connected on input port 1, the node emits silence.
#[derive(Debug, Clone)]
pub struct VocoderNode {
    inner: Vocoder,
}

impl VocoderNode {
    /// Builds a vocoder node for a `channels`-wide signal at `sample_rate` Hz,
    /// sized for blocks of up to `max_frames` samples.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        max_frames: usize,
        params: VocoderParams,
    ) -> Self {
        Self {
            inner: Vocoder::new(sample_rate, channels, max_frames, params),
        }
    }

    /// Returns the current parameters.
    #[inline]
    #[must_use]
    pub fn params(&self) -> VocoderParams {
        self.inner.params()
    }

    /// Returns the number of analysis/synthesis bands.
    #[inline]
    #[must_use]
    pub fn bands(&self) -> usize {
        self.inner.bands()
    }

    /// Redesigns both banks and envelope times, preserving state (click-free).
    pub fn set_params(&mut self, params: VocoderParams) {
        self.inner.set_params(params);
    }
}

impl AudioNode for VocoderNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (inputs, outputs) = io.split();
        let Some(output) = outputs.first_mut() else {
            return;
        };
        match (inputs.first(), inputs.get(1)) {
            (Some(modulator), Some(carrier)) => {
                self.inner.process_into(modulator, carrier, output);
            }
            _ => output.clear(),
        }
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

    /// Runs the node with a modulator and carrier, returning the output.
    fn run(
        node: &mut VocoderNode,
        modulator: &AudioBuffer,
        carrier: &AudioBuffer,
    ) -> AudioBuffer {
        let frames = modulator.active_frames().min(carrier.active_frames());
        let inputs = [modulator.clone(), carrier.clone()];
        let mut out = AudioBuffer::new(modulator.layout(), modulator.capacity_frames().max(1));
        out.set_active_frames(frames);
        let mut outputs = [out];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(frames), &mut io);
        }
        let [out] = outputs;
        out
    }

    /// Discrete single-bin magnitude at `freq_hz` (small N, test only).
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

    /// A deterministic sine of `freq_hz` filling a mono buffer.
    fn sine(freq_hz: Sample, frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, frames);
        let w = core::f32::consts::TAU * freq_hz / SR as Sample;
        let ch = buf.channel_mut(0);
        for (k, s) in ch.iter_mut().enumerate() {
            *s = bevy_math::ops::sin(w * k as Sample);
        }
        buf
    }

    /// A broadband-ish carrier: a deterministic xorshift noise burst.
    fn noise(frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, frames);
        let mut state: u32 = 0x1234_5678;
        let ch = buf.channel_mut(0);
        for s in ch.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *s = (state as Sample / u32::MAX as Sample) * 2.0 - 1.0;
        }
        buf
    }

    #[test]
    fn silent_modulator_gives_silence() {
        let mut node = VocoderNode::new(SR, 1, 512, VocoderParams::default());
        let modulator = AudioBuffer::new(ChannelLayout::Mono, 512);
        let carrier = noise(512);
        let out = run(&mut node, &modulator, &carrier);
        assert!(out.channel(0).iter().all(|&s| s.abs() <= 1.0e-6));
    }

    #[test]
    fn silent_carrier_gives_silence() {
        let mut node = VocoderNode::new(SR, 1, 512, VocoderParams::default());
        let modulator = noise(512);
        let carrier = AudioBuffer::new(ChannelLayout::Mono, 512);
        let out = run(&mut node, &modulator, &carrier);
        assert!(out.channel(0).iter().all(|&s| s.abs() <= 1.0e-6));
    }

    #[test]
    fn missing_carrier_emits_silence() {
        let mut node = VocoderNode::new(SR, 1, 256, VocoderParams::default());
        let modulator = noise(256);
        let inputs = [modulator];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 256)];
        outputs[0].set_active_frames(256);
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(256), &mut io);
        }
        let [out] = outputs;
        assert!(out.channel(0).iter().all(|&s| s == 0.0));
    }

    /// Integrated magnitude over a frequency range (averages out noise so a
    /// single random carrier bin cannot dominate the comparison).
    fn band_energy(signal: &[Sample], lo_hz: Sample, hi_hz: Sample) -> Sample {
        let steps = 24;
        let mut sum = 0.0;
        for i in 0..steps {
            let t = i as Sample / (steps as Sample - 1.0);
            let f = lo_hz + (hi_hz - lo_hz) * t;
            sum += goertzel_magnitude(signal, f);
        }
        sum
    }

    #[test]
    fn imprints_modulator_band_onto_carrier() {
        // A modulator tone near 1 kHz should let the carrier's 1 kHz region
        // through far more strongly than a region where the modulator is quiet.
        let frames = 8192;
        let modulator = sine(1000.0, frames);
        let carrier = noise(frames);
        let mut node = VocoderNode::new(SR, 1, frames, VocoderParams::default());
        let out = run(&mut node, &modulator, &carrier);
        let near = band_energy(out.channel(0), 850.0, 1150.0);
        let far = band_energy(out.channel(0), 7000.0, 9000.0);
        assert!(near > far * 2.0, "near {near} far {far}");
    }

    #[test]
    fn tracks_modulator_frequency() {
        // Different modulator tones should emphasise different carrier regions.
        let frames = 8192;
        let carrier = noise(frames);
        let mut low = VocoderNode::new(SR, 1, frames, VocoderParams::default());
        let mut high = VocoderNode::new(SR, 1, frames, VocoderParams::default());
        let out_low = run(&mut low, &sine(300.0, frames), &carrier);
        let out_high = run(&mut high, &sine(5000.0, frames), &carrier);
        let low_at_300 = goertzel_magnitude(out_low.channel(0), 300.0);
        let high_at_300 = goertzel_magnitude(out_high.channel(0), 300.0);
        assert!(low_at_300 > high_at_300, "low {low_at_300} high {high_at_300}");
    }

    #[test]
    fn output_gain_scales_linearly() {
        let frames = 1024;
        let modulator = noise(frames);
        let carrier = noise(frames);
        let mut unity = VocoderNode::new(SR, 1, frames, VocoderParams::default());
        let mut boosted = VocoderNode::new(
            SR,
            1,
            frames,
            VocoderParams {
                output_gain_db: 6.0206, // ~ x2
                ..VocoderParams::default()
            },
        );
        let out_u = run(&mut unity, &modulator, &carrier);
        let out_b = run(&mut boosted, &modulator, &carrier);
        let mut max_err = 0.0f32;
        for (a, b) in out_u.channel(0).iter().zip(out_b.channel(0)) {
            let err = (b - 2.0 * a).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err <= 1.0e-3, "max_err {max_err}");
    }

    #[test]
    fn stereo_channels_match_for_identical_input() {
        let frames = 512;
        let mut node = VocoderNode::new(SR, 2, frames, VocoderParams::default());
        let mut modulator = AudioBuffer::new(ChannelLayout::Stereo, frames);
        let mut carrier = AudioBuffer::new(ChannelLayout::Stereo, frames);
        let mono_mod = noise(frames);
        let mono_car = sine(2000.0, frames);
        for ch in 0..2 {
            modulator.channel_mut(ch).copy_from_slice(mono_mod.channel(0));
            carrier.channel_mut(ch).copy_from_slice(mono_car.channel(0));
        }
        let out = run(&mut node, &modulator, &carrier);
        assert_eq!(out.channel(0), out.channel(1));
    }

    #[test]
    fn reset_clears_state() {
        let frames = 256;
        let modulator = noise(frames);
        let carrier = sine(1500.0, frames);
        let mut node = VocoderNode::new(SR, 1, frames, VocoderParams::default());
        let first = run(&mut node, &modulator, &carrier);
        node.reset();
        let second = run(&mut node, &modulator, &carrier);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn set_params_updates_params() {
        let mut node = VocoderNode::new(SR, 1, 64, VocoderParams::default());
        node.set_params(VocoderParams {
            low_hz: 120.0,
            high_hz: 8000.0,
            attack_ms: 5.0,
            release_ms: 40.0,
            output_gain_db: -3.0,
        });
        assert_eq!(node.params().low_hz, 120.0);
        assert_eq!(node.params().release_ms, 40.0);
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = VocoderNode::new(SR, 2, 64, VocoderParams::default());
        let mut modulator = AudioBuffer::new(ChannelLayout::Stereo, 4);
        let mut carrier = AudioBuffer::new(ChannelLayout::Stereo, 4);
        modulator.set_active_frames(0);
        carrier.set_active_frames(0);
        let out = run(&mut node, &modulator, &carrier);
        assert_eq!(out.active_frames(), 0);
    }

    #[test]
    fn params_round_trip() {
        let params = VocoderParams {
            low_hz: 100.0,
            high_hz: 10_000.0,
            attack_ms: 3.0,
            release_ms: 20.0,
            output_gain_db: 1.5,
        };
        let node = VocoderNode::new(SR, 1, 32, params);
        assert_eq!(node.params(), params);
    }

    #[test]
    fn core_matches_node() {
        let frames = 512;
        let params = VocoderParams::default();
        let modulator = noise(frames);
        let carrier = sine(1200.0, frames);
        let mut core = Vocoder::new(SR, 1, frames, params);
        let mut node = VocoderNode::new(SR, 1, frames, params);
        let mut core_out = AudioBuffer::new(ChannelLayout::Mono, frames);
        core_out.set_active_frames(frames);
        core.process_into(&modulator, &carrier, &mut core_out);
        let node_out = run(&mut node, &modulator, &carrier);
        assert_eq!(core_out.channel(0), node_out.channel(0));
    }

    #[test]
    fn band_count_is_reported() {
        let node = VocoderNode::new(SR, 1, 32, VocoderParams::default());
        assert_eq!(node.bands(), VOCODER_BANDS);
    }

    #[test]
    fn band_centers_are_monotonic() {
        let (low, high) = sanitize_range(80.0, 12_000.0, SR);
        let centers = band_centers(low, high);
        for pair in centers.windows(2) {
            assert!(pair[1] > pair[0], "centers not increasing: {centers:?}");
        }
        assert!(centers[0] >= low - 1.0);
        assert!(centers[VOCODER_BANDS - 1] <= high + 1.0);
    }
}

/// A worked example: making a noise carrier "speak" a 1 kHz modulator tone.
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::vocoder::{VocoderNode, VocoderParams};
///
/// let mut node = VocoderNode::new(48_000, 1, 16, VocoderParams::default());
///
/// // Modulator: a steady tone. Carrier: a simple ramp (broadband enough).
/// let mut modulator = AudioBuffer::new(ChannelLayout::Mono, 16);
/// let mut carrier = AudioBuffer::new(ChannelLayout::Mono, 16);
/// for i in 0..16 {
///     modulator.channel_mut(0)[i] = if i % 2 == 0 { 0.5 } else { -0.5 };
///     carrier.channel_mut(0)[i] = (i as f32 / 16.0) * 2.0 - 1.0;
/// }
///
/// let inputs = [modulator, carrier];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 16)];
/// outputs[0].set_active_frames(16);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 16, playhead: 0 };
/// {
///     let mut io = ProcessIo::new(&inputs, &mut outputs);
///     node.process(&ctx, &mut io);
/// }
/// // The carrier is shaped by the modulator's band energy, so it is finite.
/// assert!(outputs[0].channel(0).iter().all(|s| s.is_finite()));
/// ```
#[cfg(doctest)]
struct DocExample;
