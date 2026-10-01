//! Graphic equaliser: a fixed bank of ISO-spaced peaking bands whose only
//! control is a per-band gain, cascaded from the shared [`Biquad`] core.
//!
//! A graphic EQ is the hardware-console / live-sound counterpart of a
//! parametric EQ. Instead of freely tunable frequency and `Q`, it exposes a
//! fixed row of "sliders" at standard ISO centre frequencies (octave or
//! third-octave spacing); the engineer only pushes each band's gain up or
//! down. Every band is a constant-`Q` RBJ peaking section, and the bands are
//! cascaded in series so the combined response is their product.
//!
//! # The model (constant-Q ISO band bank)
//!
//! For a spacing of `1 / fraction` octaves, band `k` sits at the ISO centre
//! frequency `1000 * 2^e` Hz (octave: `e = k - 5`; third-octave:
//! `e = (k - 17) / 3`), anchored to the `1 kHz` reference. Every band shares a
//! single constant quality factor derived from its bandwidth,
//!
//! ```text
//! Q = 1 / (2^(1 / (2 * fraction)) - 2^(-1 / (2 * fraction)))
//! ```
//!
//! which is `1.4142` for octave bands and `4.3187` for third-octave bands, the
//! textbook values that make adjacent flat bands sum to a smooth response. A
//! band whose nominal centre lands at or above the Nyquist limit is clamped by
//! [`BiquadCoeffs::design`], so the filter stays stable at every sample rate.
//! A band sitting at exactly `0 dB` is collapsed to the pass-through identity
//! (`b0 = 1`, every other coefficient `0`) instead of its RBJ peaking form, so a
//! flat band -- and therefore a flat graphic EQ -- is a bit-exact pass-through
//! while still keeping its delay state warm for the next click-free move.
//!
//! # Real-time contract
//!
//! All per-band, per-channel filter state is pre-allocated in
//! [`GraphicEqNode::new`]. [`process`](crate::graph::AudioNode::process)
//! performs no allocation, takes no locks, and cannot panic.
//! [`set_gain`](GraphicEqNode::set_gain) redesigns one band's coefficients in
//! place while preserving its filter state, so slider moves are click-free.
//!
//! # Relationship
//!
//! - [`parametric_eq`](super::parametric_eq): the parametric EQ exposes fully
//!   tunable per-band frequency, `Q`, and shape; this graphic EQ fixes the
//!   frequencies and `Q` to the ISO grid and exposes only gain, trading
//!   flexibility for the familiar slider workflow. Both cascade the same
//!   [`Biquad`] peaking sections, so this node does not re-derive filter math.
//! - [`stereo_width`](super::stereo_width) / dynamics: those reshape the stereo
//!   image or level envelope; a graphic EQ only reshapes the magnitude
//!   spectrum.
//!
//! # Provenance
//!
//! The constant-`Q` graphic equaliser with ISO octave / third-octave centre
//! frequencies is standard audio-engineering practice (ISO 266 preferred
//! frequencies; RBJ "Audio EQ Cookbook" peaking sections). This module
//! contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code**; it is implemented purely
//! from that publicly documented theory on top of this crate's own
//! [`Biquad`] core.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::nodes::biquad::{Biquad, BiquadCoeffs, BiquadKind};

/// Reference anchor frequency (Hz) that every ISO band is derived from.
pub const REFERENCE_FREQUENCY_HZ: Sample = 1000.0;
/// Number of bands in an octave-spaced graphic EQ.
pub const OCTAVE_BAND_COUNT: usize = 10;
/// Number of bands in a third-octave-spaced graphic EQ.
pub const THIRD_OCTAVE_BAND_COUNT: usize = 31;
/// Maximum absolute per-band boost / cut in decibels.
pub const MAX_BAND_GAIN_DB: Sample = 24.0;

/// Band spacing of a [`GraphicEqNode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum GraphicEqSpacing {
    /// One band per octave (10 bands, `31.25 Hz` to `16 kHz`).
    Octave,
    /// Three bands per octave (31 bands, roughly `20 Hz` to `20 kHz`).
    ThirdOctave,
}

impl GraphicEqSpacing {
    /// Number of bands in this spacing.
    #[inline]
    #[must_use]
    pub fn band_count(self) -> usize {
        match self {
            Self::Octave => OCTAVE_BAND_COUNT,
            Self::ThirdOctave => THIRD_OCTAVE_BAND_COUNT,
        }
    }

    /// Bands per octave: `1` for octave, `3` for third-octave spacing.
    #[inline]
    #[must_use]
    pub fn fraction(self) -> Sample {
        match self {
            Self::Octave => 1.0,
            Self::ThirdOctave => 3.0,
        }
    }

    /// Nominal ISO centre frequency (Hz) of band `index`, anchored at
    /// [`REFERENCE_FREQUENCY_HZ`]. Indices outside `0..band_count` return
    /// `None`.
    #[must_use]
    pub fn band_frequency_hz(self, index: usize) -> Option<Sample> {
        if index >= self.band_count() {
            return None;
        }
        let exponent = match self {
            Self::Octave => index as Sample - 5.0,
            Self::ThirdOctave => (index as Sample - 17.0) / 3.0,
        };
        Some(REFERENCE_FREQUENCY_HZ * ops::powf(2.0, exponent))
    }
}

/// Constant quality factor shared by every band at the given `fraction`.
fn band_q(fraction: Sample) -> Sample {
    let half = 1.0 / (2.0 * fraction);
    let hi = ops::powf(2.0, half);
    1.0 / (hi - 1.0 / hi)
}

/// Designs one band's coefficients.
///
/// At exactly `0 dB` the band uses the pass-through identity coefficients
/// (`b0 = 1`, all others `0`) rather than the RBJ peaking form, which is only
/// unity to within the reciprocal rounding of `1 / a0`. The identity form is
/// bit-exact unity *and* keeps the delay state tracking the input, so a later
/// boost or cut from `0 dB` stays click-free.
fn band_coeffs(sample_rate: u32, freq_hz: Sample, q: Sample, gain_db: Sample) -> BiquadCoeffs {
    if gain_db == 0.0 {
        return BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
    }
    BiquadCoeffs::design(BiquadKind::Peaking, sample_rate, freq_hz, q, gain_db)
}

/// A fixed-grid graphic equaliser (input port 0 -> output port 0).
///
/// Bands are constructed flat (`0 dB`) and adjusted through
/// [`set_gain`](Self::set_gain). The cascade is applied in ascending frequency
/// order.
#[derive(Debug, Clone)]
pub struct GraphicEqNode {
    sample_rate: u32,
    spacing: GraphicEqSpacing,
    /// Shared constant `Q` for every band.
    q: Sample,
    /// Current per-band gains in decibels.
    gains: Vec<Sample>,
    /// One peaking [`Biquad`] per band, sharing the channel width.
    sections: Vec<Biquad>,
}

impl GraphicEqNode {
    /// Builds a flat graphic EQ for `channels` channels at `sample_rate` Hz
    /// with the given band `spacing`.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    /// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
    /// use prism_audio_core::nodes::effects::graphic_eq::{
    ///     GraphicEqNode, GraphicEqSpacing,
    /// };
    ///
    /// let mut eq = GraphicEqNode::new(48_000, 1, GraphicEqSpacing::Octave);
    /// let mut input = AudioBuffer::new(ChannelLayout::Mono, 32);
    /// input.channel_mut(0)[0] = 1.0;
    /// let mut output = AudioBuffer::new(ChannelLayout::Mono, 32);
    /// let inputs = [input.clone()];
    /// let mut outputs = [output];
    /// let ctx = RenderContext { sample_rate: 48_000, frames: 32, playhead: 0 };
    /// let mut io = ProcessIo::new(&inputs, &mut outputs);
    /// eq.process(&ctx, &mut io);
    /// // A flat graphic EQ is a bit-exact pass-through.
    /// assert_eq!(outputs[0].channel(0), inputs[0].channel(0));
    /// ```
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, spacing: GraphicEqSpacing) -> Self {
        let sample_rate = sample_rate.max(1);
        let count = spacing.band_count();
        let q = band_q(spacing.fraction());
        let mut sections = Vec::with_capacity(count);
        for index in 0..count {
            let freq_hz = spacing
                .band_frequency_hz(index)
                .unwrap_or(REFERENCE_FREQUENCY_HZ);
            sections.push(Biquad::new(
                band_coeffs(sample_rate, freq_hz, q, 0.0),
                channels,
            ));
        }
        Self {
            sample_rate,
            spacing,
            q,
            gains: vec![0.0; count],
            sections,
        }
    }

    /// The band spacing of this equaliser.
    #[inline]
    #[must_use]
    pub fn spacing(&self) -> GraphicEqSpacing {
        self.spacing
    }

    /// The number of bands.
    #[inline]
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.gains.len()
    }

    /// The nominal ISO centre frequency (Hz) of band `index`, or `None` if the
    /// index is out of range.
    #[inline]
    #[must_use]
    pub fn band_frequency_hz(&self, index: usize) -> Option<Sample> {
        self.spacing.band_frequency_hz(index)
    }

    /// The current gain of band `index` in decibels, or `None` if the index is
    /// out of range.
    #[inline]
    #[must_use]
    pub fn gain_db(&self, index: usize) -> Option<Sample> {
        self.gains.get(index).copied()
    }

    /// Sets band `index` to `gain_db` decibels (clamped to
    /// `[-MAX_BAND_GAIN_DB, MAX_BAND_GAIN_DB]`), redesigning that band's
    /// coefficients while preserving its filter state so the move is
    /// click-free. Out-of-range indices are ignored.
    pub fn set_gain(&mut self, index: usize, gain_db: Sample) {
        let gain_db = gain_db.clamp(-MAX_BAND_GAIN_DB, MAX_BAND_GAIN_DB);
        if let (Some(slot), Some(section)) =
            (self.gains.get_mut(index), self.sections.get_mut(index))
        {
            *slot = gain_db;
            let freq_hz = self
                .spacing
                .band_frequency_hz(index)
                .unwrap_or(REFERENCE_FREQUENCY_HZ);
            section.set_coeffs(band_coeffs(self.sample_rate, freq_hz, self.q, gain_db));
        }
    }
}

impl AudioNode for GraphicEqNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        output.copy_from(input);
        for section in &mut self.sections {
            section.process_inplace(output);
        }
    }

    fn reset(&mut self) {
        for section in &mut self.sections {
            section.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use crate::nodes::biquad::BiquadNode;

    const SR: u32 = 48_000;

    fn render(node: &mut GraphicEqNode, input: &AudioBuffer) -> AudioBuffer {
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

    fn tone(freq_hz: Sample, frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, frames);
        buf.set_active_frames(frames);
        let w = core::f32::consts::TAU * freq_hz / SR as Sample;
        let data = buf.channel_mut(0);
        for (n, s) in data.iter_mut().enumerate() {
            *s = ops::sin(w * n as Sample);
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
    fn flat_eq_is_bit_exact_unity() {
        let mut eq = GraphicEqNode::new(SR, 2, GraphicEqSpacing::Octave);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 64);
        for ch in 0..input.channels() {
            let data = input.channel_mut(ch);
            for (i, s) in data.iter_mut().enumerate() {
                *s = i as Sample - 20.0;
            }
        }
        let out = render(&mut eq, &input);
        assert_eq!(out.channel(0), input.channel(0));
        assert_eq!(out.channel(1), input.channel(1));
    }

    #[test]
    fn band_counts_match_spacing() {
        assert_eq!(GraphicEqSpacing::Octave.band_count(), OCTAVE_BAND_COUNT);
        assert_eq!(
            GraphicEqSpacing::ThirdOctave.band_count(),
            THIRD_OCTAVE_BAND_COUNT
        );
        let eq = GraphicEqNode::new(SR, 1, GraphicEqSpacing::ThirdOctave);
        assert_eq!(eq.band_count(), THIRD_OCTAVE_BAND_COUNT);
    }

    #[test]
    fn octave_centre_frequencies_are_iso() {
        let s = GraphicEqSpacing::Octave;
        assert!((s.band_frequency_hz(5).unwrap() - 1_000.0).abs() < 1e-3);
        assert!((s.band_frequency_hz(0).unwrap() - 31.25).abs() < 1e-3);
        assert!((s.band_frequency_hz(9).unwrap() - 16_000.0).abs() < 1e-1);
        assert_eq!(s.band_frequency_hz(10), None);
    }

    #[test]
    fn third_octave_is_anchored_at_1k() {
        let s = GraphicEqSpacing::ThirdOctave;
        assert!((s.band_frequency_hz(17).unwrap() - 1_000.0).abs() < 1e-3);
        // One third-octave up is 1000 * 2^(1/3) ~= 1259.9 Hz.
        assert!((s.band_frequency_hz(18).unwrap() - 1_259.92).abs() < 0.5);
    }

    #[test]
    fn constant_q_matches_textbook_values() {
        assert!((band_q(1.0) - 1.414_213_6).abs() < 1e-4);
        assert!((band_q(3.0) - 4.318_7).abs() < 1e-3);
    }

    #[test]
    fn single_band_matches_standalone_biquad() {
        // Boosting one band must be sample-identical to the equivalent
        // standalone peaking BiquadNode at the same centre, Q, and gain.
        let mut eq = GraphicEqNode::new(SR, 1, GraphicEqSpacing::Octave);
        eq.set_gain(5, 6.0); // 1 kHz band
        let q = band_q(1.0);
        let mut node = BiquadNode::new(BiquadKind::Peaking, SR, 1_000.0, q, 6.0, 1);

        let mut src = AudioBuffer::new(ChannelLayout::Mono, 128);
        src.channel_mut(0)[0] = 1.0;
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 128,
            playhead: 0,
        };
        let inputs_a = [src.clone()];
        let mut out_a = [AudioBuffer::new(ChannelLayout::Mono, 128)];
        let mut io_a = ProcessIo::new(&inputs_a, &mut out_a);
        eq.process(&ctx, &mut io_a);

        let inputs_b = [src.clone()];
        let mut out_b = [AudioBuffer::new(ChannelLayout::Mono, 128)];
        let mut io_b = ProcessIo::new(&inputs_b, &mut out_b);
        node.process(&ctx, &mut io_b);

        for (a, b) in out_a[0].channel(0).iter().zip(out_b[0].channel(0)) {
            assert!((a - b).abs() < 1e-6, "cascade {a} vs node {b}");
        }
    }

    #[test]
    fn boost_raises_band_cut_lowers_it() {
        let dry = bin_magnitude(tone(1_000.0, 8_192).channel(0), 1_000.0);

        let mut boost = GraphicEqNode::new(SR, 1, GraphicEqSpacing::Octave);
        boost.set_gain(5, 12.0);
        let wet_boost = bin_magnitude(render(&mut boost, &tone(1_000.0, 8_192)).channel(0), 1_000.0);

        let mut cut = GraphicEqNode::new(SR, 1, GraphicEqSpacing::Octave);
        cut.set_gain(5, -12.0);
        let wet_cut = bin_magnitude(render(&mut cut, &tone(1_000.0, 8_192)).channel(0), 1_000.0);

        assert!(wet_boost > dry * 1.5, "boost should raise 1 kHz: {wet_boost} vs {dry}");
        assert!(wet_cut < dry * 0.75, "cut should lower 1 kHz: {wet_cut} vs {dry}");
    }

    #[test]
    fn distant_band_leaves_tone_untouched() {
        // Boosting the 31.25 Hz band must barely affect a 1 kHz tone.
        let mut eq = GraphicEqNode::new(SR, 1, GraphicEqSpacing::Octave);
        eq.set_gain(0, 12.0);
        let dry = bin_magnitude(tone(1_000.0, 8_192).channel(0), 1_000.0);
        let wet = bin_magnitude(render(&mut eq, &tone(1_000.0, 8_192)).channel(0), 1_000.0);
        assert!((wet - dry).abs() < dry * 0.1, "far band changed 1 kHz: {wet} vs {dry}");
    }

    #[test]
    fn gain_is_clamped() {
        let mut eq = GraphicEqNode::new(SR, 1, GraphicEqSpacing::Octave);
        eq.set_gain(5, 100.0);
        assert_eq!(eq.gain_db(5), Some(MAX_BAND_GAIN_DB));
        eq.set_gain(5, -100.0);
        assert_eq!(eq.gain_db(5), Some(-MAX_BAND_GAIN_DB));
    }

    #[test]
    fn out_of_range_set_is_noop() {
        let mut eq = GraphicEqNode::new(SR, 1, GraphicEqSpacing::Octave);
        eq.set_gain(99, 6.0);
        assert_eq!(eq.band_count(), OCTAVE_BAND_COUNT);
        assert_eq!(eq.gain_db(99), None);
    }

    #[test]
    fn impulse_response_is_stable_and_finite() {
        let mut eq = GraphicEqNode::new(SR, 1, GraphicEqSpacing::ThirdOctave);
        for i in 0..eq.band_count() {
            eq.set_gain(i, if i % 2 == 0 { 6.0 } else { -6.0 });
        }
        let mut src = AudioBuffer::new(ChannelLayout::Mono, 1_024);
        src.set_active_frames(1_024);
        src.channel_mut(0)[0] = 1.0;
        let out = render(&mut eq, &src);
        let mut tail = 0.0f32;
        for &s in &out.channel(0)[900..] {
            assert!(s.is_finite());
            tail += s * s;
        }
        assert!(tail < 1e-3, "impulse response did not decay: {tail}");
    }

    #[test]
    fn reset_clears_filter_state() {
        let mut eq = GraphicEqNode::new(SR, 1, GraphicEqSpacing::Octave);
        eq.set_gain(5, 12.0);
        let _ = render(&mut eq, &tone(1_000.0, 256));
        eq.reset();
        let mut src = AudioBuffer::new(ChannelLayout::Mono, 4);
        src.set_active_frames(4);
        let out = render(&mut eq, &src);
        assert!(out.channel(0).iter().all(|&s| s == 0.0), "state not cleared");
    }

    #[test]
    fn accessors_report_configuration() {
        let eq = GraphicEqNode::new(SR, 1, GraphicEqSpacing::ThirdOctave);
        assert_eq!(eq.spacing(), GraphicEqSpacing::ThirdOctave);
        assert_eq!(eq.gain_db(0), Some(0.0));
        assert!(eq.band_frequency_hz(17).is_some());
        assert_eq!(eq.band_frequency_hz(99), None);
    }

    #[test]
    fn low_sample_rate_clamps_high_bands() {
        // At 8 kHz the 16 kHz octave band is above Nyquist; the designer clamps
        // it so the filter stays finite and stable rather than panicking.
        let mut eq = GraphicEqNode::new(8_000, 1, GraphicEqSpacing::Octave);
        eq.set_gain(9, 12.0);
        let mut src = AudioBuffer::new(ChannelLayout::Mono, 256);
        src.set_active_frames(256);
        src.channel_mut(0)[0] = 1.0;
        let ctx = RenderContext {
            sample_rate: 8_000,
            frames: 256,
            playhead: 0,
        };
        let inputs = [src];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        eq.process(&ctx, &mut io);
        assert!(outputs[0].channel(0).iter().all(|&s| s.is_finite()));
    }
}
