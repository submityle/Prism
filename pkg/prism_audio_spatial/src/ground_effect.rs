//! ISO 9613-2 ground effect attenuation (`A_gr`).
//!
//! Outdoors, sound that reaches a receiver is a mix of the direct wave and the
//! wave reflected off the ground. Depending on the ground surface and the
//! source / receiver geometry, that ground-reflected path interferes with the
//! direct path and either boosts (hard, reflective ground) or attenuates
//! (soft, porous ground) the sound in a frequency-dependent way. This module
//! is a control-rate estimator of the ground attenuation term `A_gr` from the
//! general method of **ISO 9613-2:1996, section 7.3.1**. It turns the source
//! height, the receiver height, the horizontal distance, and a ground factor
//! into a per-octave-band attenuation in decibels (positive = attenuation,
//! negative = gain) and the matching linear gains. It performs no per-sample
//! DSP.
//!
//! # Ground factor and the three regions
//!
//! The ground factor `G` lies in `[0, 1]`: `0` is acoustically hard ground
//! (paving, water, concrete) that reflects strongly, and `1` is porous, soft
//! ground (grass, farmland, snow). ISO 9613-2 splits the propagation into
//! three regions, each with its own ground factor:
//!
//! - the **source region**, near the source, factor `G_s`;
//! - the **receiver region**, near the receiver, factor `G_r`;
//! - the **middle region** between them, factor `G_m`.
//!
//! The total ground attenuation is the sum of the three regional
//! contributions:
//!
//! `A_gr = A_s(G_s, h_s) + A_r(G_r, h_r) + A_m(G_m, q)`.
//!
//! # Table 3 formulas
//!
//! For the source and receiver regions, each octave band contributes (with
//! `h` the source or receiver height and `d_p` the horizontal distance):
//!
//! - 63 Hz: `-1.5`
//! - 125 Hz: `-1.5 + G * a(h)`
//! - 250 Hz: `-1.5 + G * b(h)`
//! - 500 Hz: `-1.5 + G * c(h)`
//! - 1000 Hz: `-1.5 + G * d(h)`
//! - 2000, 4000, 8000 Hz: `-1.5 * (1 - G)`
//!
//! where the height functions are
//!
//! - `a(h) = 1.5 + 3.0 * exp(-0.12 * (h - 5)^2) * (1 - exp(-d_p / 50))`
//!   `        + 5.7 * exp(-0.09 * h^2) * (1 - exp(-2.8e-6 * d_p^2))`
//! - `b(h) = 1.5 + 8.6 * exp(-0.09 * h^2) * (1 - exp(-d_p / 50))`
//! - `c(h) = 1.5 + 14.0 * exp(-0.46 * h^2) * (1 - exp(-d_p / 50))`
//! - `d(h) = 1.5 + 5.0 * exp(-0.9 * h^2) * (1 - exp(-d_p / 50))`
//!
//! For the middle region, with `q = 0` when `d_p <= 30 * (h_s + h_r)` and
//! `q = 1 - 30 * (h_s + h_r) / d_p` otherwise, each octave band contributes:
//!
//! - 63 Hz: `-3 * q`
//! - 125 Hz and above: `-3 * q * (1 - G_m)`
//!
//! Hard ground (`G = 0`) yields `A_s = A_r = -1.5` at every band, so the
//! ground reflection is a gain; soft ground (`G = 1`) yields positive
//! attenuation across the low-mid bands, which is the classic "ground dip".
//!
//! # Relationship
//!
//! `A_gr` is one additive term of the ISO 9613-2 outdoor propagation budget.
//! The other terms already live in this crate: geometric divergence in
//! [`crate::attenuation`], atmospheric absorption in [`crate::air`], and the
//! barrier / screen term in [`crate::diffraction`]. The overall octave-band
//! attenuation is `A_div + A_atm + A_gr + A_bar + ...`, summed in decibels.
//!
//! # Control rate, not audio rate
//!
//! Every query operates on stack scalars and fixed-size arrays; there is no
//! heap allocation, no locking, and no panicking. Degenerate geometry (a
//! non-positive distance, a negative height) and non-finite inputs return safe
//! finite values, and the ground factors are clamped to `[0, 1]`. All
//! transcendental math routes through [`bevy_math::ops`], never through `f32`
//! intrinsics.
//!
//! # Provenance
//!
//! This is the textbook ISO ground-effect model: the three-region ground
//! factor scheme and the Table 3 octave-band height functions of ISO 9613-2:1996,
//! "Acoustics -- Attenuation of sound during propagation outdoors -- Part 2:
//! General method of calculation". This module is engine-agnostic and contains
//! **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google
//! Resonance Audio source or derived code**; it is implemented purely from that
//! publicly documented acoustics standard.

use bevy_math::ops;

use prism_audio_core::math::{Sample, db_to_linear};

use crate::material_library::{OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT};

/// Smallest divisor used to keep ratios finite for degenerate inputs.
const MIN_DIVISOR: Sample = 1e-9;

/// An ISO 9613-2 ground-effect estimator described by the source and receiver
/// heights, the horizontal distance, and the three-region ground factors.
///
/// Build one with [`GroundEffect::from_geometry`] (a single ground factor
/// shared by all three regions) or [`GroundEffect::from_regions`] (independent
/// source / middle / receiver factors), then query the per-octave-band
/// attenuation in decibels, the matching linear gains, or a broadband value at
/// an arbitrary frequency.
///
/// The octave bands line up with [`OCTAVE_BAND_CENTERS`], so the result can be
/// combined directly with [`crate::air`] atmospheric absorption or
/// [`crate::diffraction`] barrier losses.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GroundEffect {
    /// Ground factor of the source region in `[0, 1]`.
    g_source: Sample,
    /// Ground factor of the middle region in `[0, 1]`.
    g_middle: Sample,
    /// Ground factor of the receiver region in `[0, 1]`.
    g_receiver: Sample,
    /// Source height above the ground in metres (non-negative).
    source_height_m: Sample,
    /// Receiver height above the ground in metres (non-negative).
    receiver_height_m: Sample,
    /// Horizontal source-to-receiver distance in metres (non-negative).
    horizontal_distance_m: Sample,
}

/// Clamps a ground factor into `[0, 1]`, mapping non-finite input to `0`.
#[inline]
#[must_use]
fn clamp_ground(g: Sample) -> Sample {
    if g.is_finite() { g.clamp(0.0, 1.0) } else { 0.0 }
}

/// Clamps a height to be finite and non-negative.
#[inline]
#[must_use]
fn clamp_height(h: Sample) -> Sample {
    if h.is_finite() { h.max(0.0) } else { 0.0 }
}

/// Clamps a distance to be finite and non-negative.
#[inline]
#[must_use]
fn clamp_distance(d: Sample) -> Sample {
    if d.is_finite() { d.max(0.0) } else { 0.0 }
}

impl GroundEffect {
    /// Builds a ground-effect estimator with a single ground factor shared by
    /// all three regions (the most common case).
    ///
    /// The ground factor is clamped to `[0, 1]`, heights are clamped to be
    /// non-negative, and the distance is clamped to be non-negative; non-finite
    /// inputs fall back to safe finite values.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::ground_effect::GroundEffect;
    ///
    /// // Soft grassy ground, source and receiver two metres up, 100 m apart.
    /// let soft = GroundEffect::from_geometry(2.0, 2.0, 100.0, 1.0);
    /// // Hard reflective ground for comparison.
    /// let hard = GroundEffect::from_geometry(2.0, 2.0, 100.0, 0.0);
    /// // Soft ground attenuates the mid bands more than hard ground.
    /// assert!(soft.attenuation_db(2) > hard.attenuation_db(2));
    /// ```
    #[must_use]
    pub fn from_geometry(
        source_height_m: Sample,
        receiver_height_m: Sample,
        horizontal_distance_m: Sample,
        ground_factor: Sample,
    ) -> Self {
        let g = clamp_ground(ground_factor);
        Self::from_regions(
            g,
            g,
            g,
            source_height_m,
            receiver_height_m,
            horizontal_distance_m,
        )
    }

    /// Builds a ground-effect estimator with independent source, middle, and
    /// receiver ground factors.
    ///
    /// All factors are clamped to `[0, 1]`, heights are clamped to be
    /// non-negative, and the distance is clamped to be non-negative; non-finite
    /// inputs fall back to safe finite values.
    #[must_use]
    pub fn from_regions(
        g_source: Sample,
        g_middle: Sample,
        g_receiver: Sample,
        source_height_m: Sample,
        receiver_height_m: Sample,
        horizontal_distance_m: Sample,
    ) -> Self {
        Self {
            g_source: clamp_ground(g_source),
            g_middle: clamp_ground(g_middle),
            g_receiver: clamp_ground(g_receiver),
            source_height_m: clamp_height(source_height_m),
            receiver_height_m: clamp_height(receiver_height_m),
            horizontal_distance_m: clamp_distance(horizontal_distance_m),
        }
    }

    /// The source-region ground factor in `[0, 1]`.
    #[must_use]
    pub fn ground_factor_source(&self) -> Sample {
        self.g_source
    }

    /// The middle-region ground factor in `[0, 1]`.
    #[must_use]
    pub fn ground_factor_middle(&self) -> Sample {
        self.g_middle
    }

    /// The receiver-region ground factor in `[0, 1]`.
    #[must_use]
    pub fn ground_factor_receiver(&self) -> Sample {
        self.g_receiver
    }

    /// The source height above the ground in metres.
    #[must_use]
    pub fn source_height(&self) -> Sample {
        self.source_height_m
    }

    /// The receiver height above the ground in metres.
    #[must_use]
    pub fn receiver_height(&self) -> Sample {
        self.receiver_height_m
    }

    /// The horizontal source-to-receiver distance in metres.
    #[must_use]
    pub fn horizontal_distance(&self) -> Sample {
        self.horizontal_distance_m
    }

    /// The middle-region geometry weight `q`.
    ///
    /// `q = 0` when `d_p <= 30 * (h_s + h_r)` (short range, no middle region),
    /// otherwise `q = 1 - 30 * (h_s + h_r) / d_p`, approaching `1` at long
    /// range.
    #[must_use]
    fn middle_q(&self) -> Sample {
        let hs_hr = self.source_height_m + self.receiver_height_m;
        let d_p = self.horizontal_distance_m;
        let threshold = 30.0 * hs_hr;
        if d_p > threshold {
            1.0 - threshold / d_p.max(MIN_DIVISOR)
        } else {
            0.0
        }
    }

    /// The source- or receiver-region attenuation for one octave band, using
    /// the ISO 9613-2 Table 3 height functions.
    #[must_use]
    fn region_attenuation(g: Sample, h: Sample, d_p: Sample, band: usize) -> Sample {
        match band {
            0 => -1.5,
            1 => -1.5 + g * height_a(h, d_p),
            2 => -1.5 + g * height_b(h, d_p),
            3 => -1.5 + g * height_c(h, d_p),
            4 => -1.5 + g * height_d(h, d_p),
            _ => -1.5 * (1.0 - g),
        }
    }

    /// The middle-region attenuation for one octave band.
    #[must_use]
    fn middle_attenuation(&self, band: usize) -> Sample {
        let q = self.middle_q();
        if band == 0 {
            -3.0 * q
        } else {
            -3.0 * q * (1.0 - self.g_middle)
        }
    }

    /// The total ground attenuation in decibels for octave band `band_index`
    /// (positive = attenuation, negative = gain from constructive reflection).
    ///
    /// The index is clamped to the last band, so out-of-range indices are safe.
    #[must_use]
    pub fn attenuation_db(&self, band_index: usize) -> Sample {
        let band = band_index.min(OCTAVE_BAND_COUNT - 1);
        let d_p = self.horizontal_distance_m;
        let a_s = Self::region_attenuation(self.g_source, self.source_height_m, d_p, band);
        let a_r = Self::region_attenuation(self.g_receiver, self.receiver_height_m, d_p, band);
        let a_m = self.middle_attenuation(band);
        a_s + a_r + a_m
    }

    /// The ground attenuation in decibels for every octave band.
    #[must_use]
    pub fn band_attenuations_db(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        let mut out = [0.0; OCTAVE_BAND_COUNT];
        for (band, slot) in out.iter_mut().enumerate() {
            *slot = self.attenuation_db(band);
        }
        out
    }

    /// The linear gain `10^(-A/20)` for every octave band.
    ///
    /// The gain exceeds `1` on bands where the ground reflection is
    /// constructive (a negative attenuation), which is expected over hard
    /// ground.
    #[must_use]
    pub fn band_gains(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        let mut out = [1.0; OCTAVE_BAND_COUNT];
        for (band, slot) in out.iter_mut().enumerate() {
            *slot = db_to_linear(-self.attenuation_db(band));
        }
        out
    }

    /// The broadband ground attenuation in decibels at `freq_hz`, interpolated
    /// linearly in `log(frequency)` between the two bracketing octave-band
    /// values.
    ///
    /// Frequencies at or below 63 Hz return the lowest band; frequencies at or
    /// above 8 kHz return the highest band (the spectrum is clamped, never
    /// extrapolated). At a band centre it equals that band's value.
    #[must_use]
    pub fn broadband_attenuation_db(&self, freq_hz: Sample) -> Sample {
        let values = self.band_attenuations_db();
        if !freq_hz.is_finite() || freq_hz <= OCTAVE_BAND_CENTERS[0] {
            return values[0];
        }
        let last = OCTAVE_BAND_COUNT - 1;
        if freq_hz >= OCTAVE_BAND_CENTERS[last] {
            return values[last];
        }
        let log_f = ops::ln(freq_hz);
        for (centres, vals) in OCTAVE_BAND_CENTERS.windows(2).zip(values.windows(2)) {
            let c_hi = centres[1];
            if freq_hz <= c_hi {
                let log_lo = ops::ln(centres[0]);
                let log_hi = ops::ln(c_hi);
                let span = log_hi - log_lo;
                let t = if span > MIN_DIVISOR {
                    (log_f - log_lo) / span
                } else {
                    0.0
                };
                return vals[0] + (vals[1] - vals[0]) * t;
            }
        }
        values[last]
    }

    /// The broadband linear gain at `freq_hz`, i.e.
    /// `10^(-broadband_attenuation_db / 20)`.
    #[must_use]
    pub fn broadband_gain(&self, freq_hz: Sample) -> Sample {
        db_to_linear(-self.broadband_attenuation_db(freq_hz))
    }
}

/// ISO 9613-2 Table 3 height function `a` (used at 125 Hz).
#[inline]
#[must_use]
fn height_a(h: Sample, d_p: Sample) -> Sample {
    let dh = h - 5.0;
    let term_low = 3.0 * ops::exp(-0.12 * dh * dh) * (1.0 - ops::exp(-d_p / 50.0));
    let term_high = 5.7 * ops::exp(-0.09 * h * h) * (1.0 - ops::exp(-2.8e-6 * d_p * d_p));
    1.5 + term_low + term_high
}

/// ISO 9613-2 Table 3 height function `b` (used at 250 Hz).
#[inline]
#[must_use]
fn height_b(h: Sample, d_p: Sample) -> Sample {
    1.5 + 8.6 * ops::exp(-0.09 * h * h) * (1.0 - ops::exp(-d_p / 50.0))
}

/// ISO 9613-2 Table 3 height function `c` (used at 500 Hz).
#[inline]
#[must_use]
fn height_c(h: Sample, d_p: Sample) -> Sample {
    1.5 + 14.0 * ops::exp(-0.46 * h * h) * (1.0 - ops::exp(-d_p / 50.0))
}

/// ISO 9613-2 Table 3 height function `d` (used at 1000 Hz).
#[inline]
#[must_use]
fn height_d(h: Sample, d_p: Sample) -> Sample {
    1.5 + 5.0 * ops::exp(-0.9 * h * h) * (1.0 - ops::exp(-d_p / 50.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn hard_ground_is_uniform_minus_three() {
        // Hard ground, short range so the middle region is inactive (q = 0):
        // every band is A_s + A_r = -1.5 + -1.5 = -3 dB (constructive gain).
        let hard = GroundEffect::from_geometry(2.0, 2.0, 100.0, 0.0);
        for band in 0..OCTAVE_BAND_COUNT {
            assert!(
                approx(hard.attenuation_db(band), -3.0, 1e-4),
                "band {band}"
            );
        }
    }

    #[test]
    fn sixty_three_hz_constant_term() {
        // The 63 Hz source/receiver term is a pure -1.5 constant regardless of
        // the ground factor, so with q = 0 the band is exactly -3 dB.
        let soft = GroundEffect::from_geometry(2.0, 2.0, 100.0, 1.0);
        let hard = GroundEffect::from_geometry(2.0, 2.0, 100.0, 0.0);
        assert!(approx(soft.attenuation_db(0), -3.0, 1e-4));
        assert!(approx(hard.attenuation_db(0), -3.0, 1e-4));
    }

    #[test]
    fn soft_ground_attenuates_mid_bands_more() {
        // Soft (G = 1) versus hard (G = 0) over the same geometry: soft ground
        // has positive attenuation across the 125-1000 Hz bands, hard is -3.
        let soft = GroundEffect::from_geometry(2.0, 2.0, 100.0, 1.0);
        let hard = GroundEffect::from_geometry(2.0, 2.0, 100.0, 0.0);
        for band in 1..=4 {
            assert!(
                soft.attenuation_db(band) > hard.attenuation_db(band),
                "band {band}"
            );
        }
        // The strongest ground dip is around 250 Hz (band 2).
        assert!(approx(soft.attenuation_db(2), 10.376, 5e-2));
    }

    #[test]
    fn matches_iso_table_at_five_hundred_hz() {
        // Hand-verified against the Table 3 c() function at h = 2, d_p = 100.
        let soft = GroundEffect::from_geometry(2.0, 2.0, 100.0, 1.0);
        // A_gr(500) = 2 * (-1.5 + c(2, 100)) = 2 * 1.9225 = 3.845 dB.
        assert!(approx(soft.attenuation_db(3), 3.845, 5e-2));
    }

    #[test]
    fn high_bands_vanish_for_soft_ground() {
        // For soft ground the 2/4/8 kHz source and receiver terms are
        // -1.5*(1-1) = 0, and with q = 0 the total is 0 dB.
        let soft = GroundEffect::from_geometry(2.0, 2.0, 100.0, 1.0);
        for band in 5..OCTAVE_BAND_COUNT {
            assert!(approx(soft.attenuation_db(band), 0.0, 1e-4), "band {band}");
        }
    }

    #[test]
    fn middle_region_weight_grows_with_distance() {
        // The 63 Hz band equals -3 - 3q, isolating the middle weight q.
        // h_s + h_r = 4, threshold = 120 m.
        let near = GroundEffect::from_geometry(2.0, 2.0, 200.0, 0.0);
        let far = GroundEffect::from_geometry(2.0, 2.0, 400.0, 0.0);
        // q(200) = 1 - 120/200 = 0.4 -> -3 - 1.2 = -4.2
        assert!(approx(near.attenuation_db(0), -4.2, 1e-3));
        // q(400) = 1 - 120/400 = 0.7 -> -3 - 2.1 = -5.1
        assert!(approx(far.attenuation_db(0), -5.1, 1e-3));
        // More distance -> larger q -> more negative 63 Hz value.
        assert!(far.attenuation_db(0) < near.attenuation_db(0));
    }

    #[test]
    fn middle_region_inactive_at_short_range() {
        // d_p = 100 < 120 -> q = 0, so the middle region contributes nothing.
        let g = GroundEffect::from_geometry(2.0, 2.0, 100.0, 0.5);
        assert!(approx(g.middle_q(), 0.0, 1e-9));
    }

    #[test]
    fn band_gains_match_attenuations() {
        let g = GroundEffect::from_geometry(2.0, 2.0, 100.0, 0.7);
        let atts = g.band_attenuations_db();
        let gains = g.band_gains();
        for (att, gain) in atts.iter().zip(gains.iter()) {
            assert!(approx(*gain, db_to_linear(-att), 1e-6));
        }
    }

    #[test]
    fn hard_ground_gain_exceeds_unity() {
        // Hard ground gives -3 dB attenuation = a gain above 1.
        let hard = GroundEffect::from_geometry(2.0, 2.0, 100.0, 0.0);
        for gain in hard.band_gains() {
            assert!(gain > 1.0);
        }
    }

    #[test]
    fn broadband_at_band_centre_equals_band_value() {
        let g = GroundEffect::from_geometry(2.0, 2.0, 100.0, 1.0);
        for (band, &f) in OCTAVE_BAND_CENTERS.iter().enumerate() {
            let via_band = g.attenuation_db(band);
            let via_broadband = g.broadband_attenuation_db(f);
            assert!(approx(via_band, via_broadband, 1e-3), "band {band}");
        }
    }

    #[test]
    fn broadband_interpolates_between_bands() {
        let g = GroundEffect::from_geometry(2.0, 2.0, 100.0, 1.0);
        // Between 125 Hz (band 1, ~1.98) and 250 Hz (band 2, ~10.38): the
        // interpolated value lies strictly between the two.
        let lo = g.attenuation_db(1);
        let hi = g.attenuation_db(2);
        let mid = g.broadband_attenuation_db(177.0);
        let (min, max) = if lo < hi { (lo, hi) } else { (hi, lo) };
        assert!(mid > min && mid < max, "mid {mid} lo {lo} hi {hi}");
    }

    #[test]
    fn broadband_clamps_endpoints() {
        let g = GroundEffect::from_geometry(2.0, 2.0, 100.0, 1.0);
        let low = g.broadband_attenuation_db(20.0);
        let high = g.broadband_attenuation_db(20_000.0);
        assert!(approx(low, g.attenuation_db(0), 1e-3));
        assert!(approx(high, g.attenuation_db(OCTAVE_BAND_COUNT - 1), 1e-3));
    }

    #[test]
    fn regional_factors_are_stored_and_clamped() {
        let g = GroundEffect::from_regions(1.5, -0.2, 0.4, 3.0, 1.0, 50.0);
        assert!(approx(g.ground_factor_source(), 1.0, 1e-9));
        assert!(approx(g.ground_factor_middle(), 0.0, 1e-9));
        assert!(approx(g.ground_factor_receiver(), 0.4, 1e-9));
        assert!(approx(g.source_height(), 3.0, 1e-9));
        assert!(approx(g.receiver_height(), 1.0, 1e-9));
        assert!(approx(g.horizontal_distance(), 50.0, 1e-9));
    }

    #[test]
    fn non_finite_and_degenerate_inputs_are_safe() {
        let nan = GroundEffect::from_geometry(Sample::NAN, Sample::NAN, Sample::NAN, Sample::NAN);
        assert!(nan.source_height().is_finite());
        assert!(nan.horizontal_distance().is_finite());
        for band in 0..OCTAVE_BAND_COUNT {
            assert!(nan.attenuation_db(band).is_finite(), "band {band}");
            assert!(nan.band_gains()[band].is_finite(), "band {band}");
        }
        // Zero distance and negative heights clamp safely.
        let zero = GroundEffect::from_geometry(-3.0, -1.0, 0.0, 0.5);
        assert!(approx(zero.source_height(), 0.0, 1e-9));
        assert!(zero.attenuation_db(2).is_finite());
        assert!(zero.broadband_gain(Sample::INFINITY).is_finite());
    }
}
