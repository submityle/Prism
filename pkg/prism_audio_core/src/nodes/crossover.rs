//! Linkwitz-Riley multi-band crossover: splits one signal into `N` frequency
//! bands whose sum reconstructs the input with a flat magnitude response.
//!
//! This is the band-splitting primitive that multi-band dynamics (a multi-band
//! compressor, a de-esser), band-wise spatialisation, and mastering chains all
//! build on. Each split point is a fourth-order Linkwitz-Riley (LR4, 24 dB per
//! octave) crossover, realised as two cascaded Butterworth second-order
//! sections (Q = 1/sqrt(2)) for the low branch and two for the high branch.
//!
//! # Reconstruction
//!
//! An LR4 low-pass summed with its complementary LR4 high-pass equals a
//! second-order all-pass: the two bands are in phase at the crossover (each
//! -6 dB there) and their sum has unity magnitude at every frequency. For more
//! than two bands the low bands are additionally passed through the all-pass
//! equivalent of every *higher* crossover so that the running sum stays a pure
//! all-pass. Concretely, with crossovers `f_0 < f_1 < ... < f_{M-1}` the raw
//! serial split gives
//!
//! ```text
//! sum = LP_0 + HP_0 * (LP_1 + HP_1 * (... ))
//! ```
//!
//! which is only all-pass once band `k` is multiplied by the all-pass sections
//! `AP_{k+1} * ... * AP_{M-1}`. After that compensation the sum of all bands is
//! `AP_0 * AP_1 * ... * AP_{M-1}` -- a cascade of all-passes, hence magnitude
//! flat (with the phase response inherent to Linkwitz-Riley networks).
//!
//! # Real-time contract
//!
//! All filter state and the internal carry buffer are allocated in
//! [`LinkwitzRileyCrossover::new`]. [`LinkwitzRileyCrossover::process_block`]
//! performs no allocation, takes no locks, and cannot panic: mismatched
//! channel counts, short band slices, and zero-length blocks all degrade
//! gracefully.
//!
//! # Provenance
//!
//! The Linkwitz-Riley topology is standard public loudspeaker-crossover theory
//! (S. Linkwitz, "Active Crossover Networks for Noninverting Loudspeakers",
//! JAES 1976; the LR4 = squared-Butterworth construction is textbook). The
//! second-order all-pass and Butterworth section coefficients are the
//! canonical Robert Bristow-Johnson audio-EQ cookbook equations. No source or
//! derivative code from any commercial or open engine (UE, Unity, Godot,
//! Wwise, FMOD, Steam Audio) was consulted or copied; only the shared public
//! mathematics is used, and it is expressed on top of this crate's own
//! [`Biquad`] unit.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::buffer::{AudioBuffer, ChannelLayout};
use crate::math::Sample;
use crate::nodes::biquad::{Biquad, BiquadCoeffs, BiquadKind};

/// Maximum number of crossover frequencies (hence `MAX_BANDS - 1`).
pub const MAX_CROSSOVERS: usize = 7;

/// Maximum number of output bands a single crossover can produce.
pub const MAX_BANDS: usize = MAX_CROSSOVERS + 1;

/// Butterworth quality factor for a maximally flat second-order section.
const BUTTERWORTH_Q: Sample = core::f32::consts::FRAC_1_SQRT_2;

/// Builds normalised coefficients for a second-order RBJ all-pass at `fc`.
///
/// This is the phase-matching filter whose response equals an LR4 low-pass
/// plus high-pass sum, used to keep the multi-band reconstruction flat.
#[must_use]
fn allpass2_coeffs(sample_rate: u32, fc: Sample, q: Sample) -> BiquadCoeffs {
    let sr = sample_rate.max(1) as Sample;
    let f0 = fc.clamp(1.0, sr * 0.499);
    let q = q.max(1.0e-4);
    let w0 = 2.0 * core::f32::consts::PI * f0 / sr;
    let (sin_w0, cos_w0) = ops::sin_cos(w0);
    let alpha = sin_w0 / (2.0 * q);
    let a0 = 1.0 + alpha;
    let inv_a0 = 1.0 / a0;
    BiquadCoeffs {
        b0: (1.0 - alpha) * inv_a0,
        b1: (-2.0 * cos_w0) * inv_a0,
        b2: (1.0 + alpha) * inv_a0,
        a1: (-2.0 * cos_w0) * inv_a0,
        a2: (1.0 - alpha) * inv_a0,
    }
}

/// A fourth-order Linkwitz-Riley multi-band splitter.
///
/// Constructed from an ascending list of crossover frequencies; a list of `M`
/// crossovers yields `M + 1` bands. Feed a block through
/// [`process_block`](Self::process_block) to fill one output buffer per band.
#[derive(Debug, Clone)]
pub struct LinkwitzRileyCrossover {
    channels: usize,
    num_bands: usize,
    /// Two cascaded Butterworth low-pass sections per crossover (LR4 low).
    lp: Vec<[Biquad; 2]>,
    /// Two cascaded Butterworth high-pass sections per crossover (LR4 high).
    hp: Vec<[Biquad; 2]>,
    /// Per-band all-pass phase compensation (empty for the top band).
    comp: Vec<Vec<Biquad>>,
    /// Scratch holding the running high-passed remainder between stages.
    carry: AudioBuffer,
}

impl LinkwitzRileyCrossover {
    /// Builds a crossover for `layout` handling blocks up to `max_frames`.
    ///
    /// `crossover_freqs` is copied, sorted ascending, and clamped into the open
    /// Nyquist band; at most [`MAX_CROSSOVERS`] entries are honoured. An empty
    /// list produces a single pass-through band equal to the input.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        layout: ChannelLayout,
        crossover_freqs: &[Sample],
        max_frames: usize,
    ) -> Self {
        let channels = layout.channel_count().max(1);
        let sr = sample_rate.max(1) as Sample;
        let nyquist = sr * 0.499;

        // Copy, clamp, and sort the crossover frequencies deterministically.
        let mut freqs: Vec<Sample> = Vec::with_capacity(MAX_CROSSOVERS);
        for &f in crossover_freqs.iter().take(MAX_CROSSOVERS) {
            freqs.push(f.clamp(1.0, nyquist));
        }
        freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

        let m = freqs.len();
        let num_bands = m + 1;

        let mut lp: Vec<[Biquad; 2]> = Vec::with_capacity(m);
        let mut hp: Vec<[Biquad; 2]> = Vec::with_capacity(m);
        for &fc in &freqs {
            let lp_c = BiquadCoeffs::design(BiquadKind::LowPass, sample_rate, fc, BUTTERWORTH_Q, 0.0);
            let hp_c =
                BiquadCoeffs::design(BiquadKind::HighPass, sample_rate, fc, BUTTERWORTH_Q, 0.0);
            lp.push([Biquad::new(lp_c, channels), Biquad::new(lp_c, channels)]);
            hp.push([Biquad::new(hp_c, channels), Biquad::new(hp_c, channels)]);
        }

        // Band `k` is phase-matched by the all-pass of every crossover above it.
        let mut comp: Vec<Vec<Biquad>> = Vec::with_capacity(num_bands);
        for k in 0..num_bands {
            let mut sections: Vec<Biquad> = Vec::new();
            let mut j = k + 1;
            while j < m {
                let ap = allpass2_coeffs(sample_rate, freqs[j], BUTTERWORTH_Q);
                sections.push(Biquad::new(ap, channels));
                j += 1;
            }
            comp.push(sections);
        }

        let carry = AudioBuffer::new(layout, max_frames.max(1));

        Self {
            channels,
            num_bands,
            lp,
            hp,
            comp,
            carry,
        }
    }

    /// Number of output bands this crossover produces (`crossovers + 1`).
    #[inline]
    #[must_use]
    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    /// Channel count the crossover was configured for.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Splits `input` into per-band buffers, filling `bands[0..num_bands]`.
    ///
    /// Band 0 is the lowest frequency range and the last band is the highest.
    /// Extra `bands` entries are left untouched; if fewer than `num_bands` are
    /// supplied the missing high bands are simply not written. The active frame
    /// count of each written band is set to match `input`.
    pub fn process_block(&mut self, input: &AudioBuffer, bands: &mut [AudioBuffer]) {
        let written = self.num_bands.min(bands.len());
        if written == 0 {
            return;
        }
        let frames = input.active_frames();

        // carry <- input
        self.carry.set_active_frames(frames);
        copy_active(&mut self.carry, input);

        let m = self.lp.len();
        for j in 0..m {
            // Low band j = LR4 low-pass of the current carry.
            if j < bands.len() {
                bands[j].set_active_frames(frames);
                copy_active(&mut bands[j], &self.carry);
                self.lp[j][0].process_inplace(&mut bands[j]);
                self.lp[j][1].process_inplace(&mut bands[j]);
            }
            // carry <- LR4 high-pass of the current carry (feeds the next stage).
            self.hp[j][0].process_inplace(&mut self.carry);
            self.hp[j][1].process_inplace(&mut self.carry);
        }

        // Top band is whatever remains after every high-pass stage.
        if m < bands.len() {
            bands[m].set_active_frames(frames);
            copy_active(&mut bands[m], &self.carry);
        }

        // All-pass phase compensation so the band sum stays magnitude flat.
        for (k, band) in bands.iter_mut().enumerate().take(written) {
            for ap in &mut self.comp[k] {
                ap.process_inplace(band);
            }
        }
    }

    /// Clears every filter's memory and the internal carry buffer.
    pub fn reset(&mut self) {
        for pair in &mut self.lp {
            pair[0].reset();
            pair[1].reset();
        }
        for pair in &mut self.hp {
            pair[0].reset();
            pair[1].reset();
        }
        for sections in &mut self.comp {
            for ap in sections {
                ap.reset();
            }
        }
        self.carry.clear();
    }
}

/// Copies the active samples of `src` into `dst` without panicking.
///
/// Unlike [`AudioBuffer::copy_from`] this tolerates differing layouts and
/// channel counts by copying the overlapping channels and frames only.
fn copy_active(dst: &mut AudioBuffer, src: &AudioBuffer) {
    let channels = dst.channels().min(src.channels());
    for ch in 0..channels {
        let src_ch = src.channel(ch);
        let dst_ch = dst.channel_mut(ch);
        let n = dst_ch.len().min(src_ch.len());
        dst_ch[..n].copy_from_slice(&src_ch[..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::ChannelLayout;

    const SR: u32 = 48_000;

    fn make_bands(count: usize, frames: usize) -> Vec<AudioBuffer> {
        (0..count)
            .map(|_| AudioBuffer::new(ChannelLayout::Mono, frames))
            .collect()
    }

    fn fill_sine(buf: &mut AudioBuffer, freq: Sample, sample_rate: u32) {
        let sr = sample_rate as Sample;
        let step = 2.0 * core::f32::consts::PI * freq / sr;
        let data = buf.channel_mut(0);
        let mut phase = 0.0;
        for d in data.iter_mut() {
            *d = ops::sin(phase);
            phase += step;
        }
    }

    /// RMS of the tail (after settling) of a mono buffer.
    fn tail_rms(buf: &AudioBuffer, skip: usize) -> Sample {
        let data = buf.channel(0);
        let start = skip.min(data.len());
        let slice = &data[start..];
        if slice.is_empty() {
            return 0.0;
        }
        let mut acc = 0.0;
        for &s in slice {
            acc += s * s;
        }
        ops::sqrt(acc / slice.len() as Sample)
    }

    /// Sums all band buffers into `out` (mono, zeroed first).
    fn sum_bands(bands: &[AudioBuffer], out: &mut AudioBuffer) {
        out.clear();
        let dst = out.channel_mut(0);
        for band in bands {
            let src = band.channel(0);
            let n = dst.len().min(src.len());
            for i in 0..n {
                dst[i] += src[i];
            }
        }
    }

    #[test]
    fn single_band_is_identity() {
        let frames = 256;
        let mut xo = LinkwitzRileyCrossover::new(SR, ChannelLayout::Mono, &[], frames);
        assert_eq!(xo.num_bands(), 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        fill_sine(&mut input, 1_000.0, SR);
        let mut bands = make_bands(1, frames);
        xo.process_block(&input, &mut bands);
        for (a, b) in bands[0].channel(0).iter().zip(input.channel(0)) {
            assert!((a - b).abs() < 1.0e-6, "identity band should equal input");
        }
    }

    #[test]
    fn two_way_reconstructs_flat() {
        let frames = 4_096;
        let skip = 2_048;
        for &freq in &[80.0, 500.0, 1_000.0, 4_000.0, 12_000.0] {
            let mut xo = LinkwitzRileyCrossover::new(SR, ChannelLayout::Mono, &[1_000.0], frames);
            let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
            fill_sine(&mut input, freq, SR);
            let mut bands = make_bands(2, frames);
            xo.process_block(&input, &mut bands);
            let mut sum = AudioBuffer::new(ChannelLayout::Mono, frames);
            sum_bands(&bands, &mut sum);
            let in_rms = tail_rms(&input, skip);
            let sum_rms = tail_rms(&sum, skip);
            let ratio = sum_rms / in_rms;
            assert!(
                (ratio - 1.0).abs() < 0.05,
                "2-way sum magnitude flat at {freq} Hz (ratio {ratio})"
            );
        }
    }

    #[test]
    fn three_way_reconstructs_flat() {
        let frames = 8_192;
        let skip = 4_096;
        for &freq in &[60.0, 300.0, 1_000.0, 3_000.0, 9_000.0] {
            let mut xo =
                LinkwitzRileyCrossover::new(SR, ChannelLayout::Mono, &[400.0, 4_000.0], frames);
            assert_eq!(xo.num_bands(), 3);
            let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
            fill_sine(&mut input, freq, SR);
            let mut bands = make_bands(3, frames);
            xo.process_block(&input, &mut bands);
            let mut sum = AudioBuffer::new(ChannelLayout::Mono, frames);
            sum_bands(&bands, &mut sum);
            let in_rms = tail_rms(&input, skip);
            let sum_rms = tail_rms(&sum, skip);
            let ratio = sum_rms / in_rms;
            assert!(
                (ratio - 1.0).abs() < 0.06,
                "3-way sum magnitude flat at {freq} Hz (ratio {ratio})"
            );
        }
    }

    #[test]
    fn low_band_rejects_high_frequency() {
        let frames = 4_096;
        let skip = 2_048;
        let mut xo = LinkwitzRileyCrossover::new(SR, ChannelLayout::Mono, &[1_000.0], frames);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        fill_sine(&mut input, 8_000.0, SR);
        let mut bands = make_bands(2, frames);
        xo.process_block(&input, &mut bands);
        let low = tail_rms(&bands[0], skip);
        let high = tail_rms(&bands[1], skip);
        assert!(high > 0.5, "high band keeps the 8 kHz tone (rms {high})");
        assert!(low < 0.05, "low band rejects the 8 kHz tone (rms {low})");
    }

    #[test]
    fn high_band_rejects_low_frequency() {
        let frames = 4_096;
        let skip = 2_048;
        let mut xo = LinkwitzRileyCrossover::new(SR, ChannelLayout::Mono, &[1_000.0], frames);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        fill_sine(&mut input, 100.0, SR);
        let mut bands = make_bands(2, frames);
        xo.process_block(&input, &mut bands);
        let low = tail_rms(&bands[0], skip);
        let high = tail_rms(&bands[1], skip);
        assert!(low > 0.5, "low band keeps the 100 Hz tone (rms {low})");
        assert!(high < 0.05, "high band rejects the 100 Hz tone (rms {high})");
    }

    #[test]
    fn both_bands_minus_six_db_at_crossover() {
        let frames = 8_192;
        let skip = 4_096;
        let fc = 1_000.0;
        let mut xo = LinkwitzRileyCrossover::new(SR, ChannelLayout::Mono, &[fc], frames);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        fill_sine(&mut input, fc, SR);
        let mut bands = make_bands(2, frames);
        xo.process_block(&input, &mut bands);
        let in_rms = tail_rms(&input, skip);
        let low = tail_rms(&bands[0], skip) / in_rms;
        let high = tail_rms(&bands[1], skip) / in_rms;
        // LR4 is -6 dB (half amplitude) in each band at the crossover.
        assert!((low - 0.5).abs() < 0.05, "low band -6 dB at fc (ratio {low})");
        assert!((high - 0.5).abs() < 0.05, "high band -6 dB at fc (ratio {high})");
    }

    #[test]
    fn three_way_mid_band_isolation() {
        let frames = 8_192;
        let skip = 4_096;
        let mut xo =
            LinkwitzRileyCrossover::new(SR, ChannelLayout::Mono, &[400.0, 4_000.0], frames);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        fill_sine(&mut input, 1_200.0, SR);
        let mut bands = make_bands(3, frames);
        xo.process_block(&input, &mut bands);
        let low = tail_rms(&bands[0], skip);
        let mid = tail_rms(&bands[1], skip);
        let high = tail_rms(&bands[2], skip);
        assert!(mid > 0.6, "mid band keeps 1.2 kHz (rms {mid})");
        assert!(low < 0.1, "low band rejects 1.2 kHz (rms {low})");
        assert!(high < 0.1, "high band rejects 1.2 kHz (rms {high})");
    }

    #[test]
    fn dc_lands_in_low_band() {
        let frames = 2_048;
        let skip = 1_024;
        let mut xo = LinkwitzRileyCrossover::new(SR, ChannelLayout::Mono, &[1_000.0], frames);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        for d in input.channel_mut(0).iter_mut() {
            *d = 1.0;
        }
        let mut bands = make_bands(2, frames);
        xo.process_block(&input, &mut bands);
        // Low band passes DC; high band blocks it.
        let low = bands[0].channel(0)[skip];
        let high = bands[1].channel(0)[skip];
        assert!((low - 1.0).abs() < 0.02, "DC passes the low band (got {low})");
        assert!(high.abs() < 0.02, "DC blocked from the high band (got {high})");
    }

    #[test]
    fn output_is_finite_and_reset_clears_tail() {
        let frames = 1_024;
        let mut xo =
            LinkwitzRileyCrossover::new(SR, ChannelLayout::Stereo, &[500.0, 5_000.0], frames);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, frames);
        fill_sine(&mut input, 2_000.0, SR);
        let ch0: Vec<Sample> = input.channel(0).to_vec();
        input.channel_mut(1).copy_from_slice(&ch0);
        let mut bands: Vec<AudioBuffer> = (0..3)
            .map(|_| AudioBuffer::new(ChannelLayout::Stereo, frames))
            .collect();
        xo.process_block(&input, &mut bands);
        for band in &bands {
            for ch in 0..band.channels() {
                for &s in band.channel(ch) {
                    assert!(s.is_finite(), "outputs must stay finite");
                }
            }
        }
        // After reset, a silent block yields silence (no lingering state).
        xo.reset();
        let silence = AudioBuffer::new(ChannelLayout::Stereo, frames);
        xo.process_block(&silence, &mut bands);
        for band in &bands {
            for ch in 0..band.channels() {
                for &s in band.channel(ch) {
                    assert!(s.abs() < 1.0e-6, "reset should clear the filter tail");
                }
            }
        }
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let frames = 64;
        let mut xo = LinkwitzRileyCrossover::new(SR, ChannelLayout::Mono, &[1_000.0], frames);
        let input = AudioBuffer::new(ChannelLayout::Mono, frames);
        // Empty band slice.
        let mut none: [AudioBuffer; 0] = [];
        xo.process_block(&input, &mut none);
        // Fewer bands than produced.
        let mut one = make_bands(1, frames);
        xo.process_block(&input, &mut one);
        // Zero active frames.
        let mut zero_input = AudioBuffer::new(ChannelLayout::Mono, frames);
        zero_input.set_active_frames(0);
        let mut bands = make_bands(2, frames);
        xo.process_block(&zero_input, &mut bands);
    }

    #[test]
    fn crossover_count_is_clamped() {
        let many: Vec<Sample> = (1..=20).map(|i| 100.0 * i as Sample).collect();
        let xo = LinkwitzRileyCrossover::new(SR, ChannelLayout::Mono, &many, 256);
        assert_eq!(xo.num_bands(), MAX_BANDS, "crossover count clamps to MAX");
    }
}
