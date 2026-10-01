//! Seat-dip effect: grazing attenuation over rows of seats or audience.
//!
//! In a concert hall, sound that travels at a shallow (near-grazing) angle over
//! rows of seats and seated listeners suffers a pronounced low-frequency notch,
//! typically centred somewhere between about 100 Hz and 300 Hz. The dip arises
//! because the direct grazing wave interferes destructively with the wave that
//! dips into and reflects out of the periodic gaps between seat rows: the gaps
//! behave like shallow open resonators whose quarter-wave resonance sits in the
//! low-mid range, and the periodicity of the rows adds odd-harmonic comb
//! structure. This module is a control-rate estimator that turns the seat-row
//! geometry (row spacing, seat height) and the source / receiver geometry into
//! a per-octave-band insertion loss in decibels (positive = attenuation),
//! aligned with [`OCTAVE_BAND_CENTERS`]. It performs no per-sample DSP.
//!
//! # Model
//!
//! Three independent factors multiply a bounded maximum dip
//! [`MAX_SEAT_DIP_DB`]:
//!
//! - a **grazing weight** `(1 - sin(theta))` clamped to `[0, 1]`, where
//!   `sin(theta)` is the sine of the ray elevation above the seat plane,
//!   computed from the mean height above the seats and the horizontal distance
//!   as `mean_height / sqrt(mean_height^2 + horizontal^2)`. A near-vertical ray
//!   (`sin(theta)` approaching `1`) gets almost no dip; a near-grazing ray
//!   (`sin(theta)` approaching `0`) gets the full dip. Using the sine directly
//!   avoids needing an arctangent;
//! - a **row weight** `1 - exp(-rows_crossed / ROW_SATURATION)` that saturates
//!   towards `1` as more seat rows are grazed, where
//!   `rows_crossed = horizontal / row_spacing`. More rows deepen the dip;
//! - a **comb shape** that sums Gaussian notches in `log(frequency)` at the odd
//!   quarter-wave harmonics `(2k + 1) * f0` for `k = 0, 1, 2`, each weighted by
//!   `1 / (2k + 1)`. The fundamental notch frequency is the quarter-wave
//!   resonance of the effective seat-gap depth,
//!   `f0 = sound_speed / (4 * seat_height)`. For a 0.45 m seat height at
//!   343 m/s that is about 190 Hz, inside the documented band.
//!
//! The product is clamped to `[0, MAX_SEAT_DIP_DB]`, so the result is always a
//! finite, non-negative attenuation. Because `f0` is low, the high octave bands
//! lie far from every comb notch and attenuate essentially nothing, which
//! reproduces the measured shape: a strong low-frequency dip that fades to zero
//! at high frequencies.
//!
//! # Relationship
//!
//! The seat-dip insertion loss is one more additive, per-octave-band
//! attenuation term, expressed in the same decibel-and-linear-gain convention
//! as the ISO 9613-2 outdoor terms in this crate: geometric divergence in
//! [`crate::attenuation`], atmospheric absorption in [`crate::air`], the
//! ground effect in [`crate::ground_effect`], and barrier diffraction in
//! [`crate::diffraction`], aggregated by [`crate::outdoor_propagation`]. Unlike
//! those outdoor terms it is an indoor, hall-geometry phenomenon, but it plugs
//! into the same `A_total = sum of bands` budget and can be combined by adding
//! its decibels to any of them.
//!
//! # Control rate, not audio rate
//!
//! Every query operates on stack scalars and fixed-size arrays; there is no
//! heap allocation, no locking, and no panicking. Degenerate geometry (a zero
//! or negative seat height, a zero row spacing, a zero distance) and
//! non-finite inputs return a safe, all-zero spectrum. All transcendental math
//! routes through [`bevy_math::ops`], never through `f32` intrinsics.
//!
//! # Provenance
//!
//! This is a textbook composite of two publicly documented ideas: the
//! quarter-wave open-resonator dip of the seat gaps and the grazing-incidence
//! interference first reported for audience seating by Schultz and Watters
//! (1964), with later measurement and modelling by Sessler and West and by
//! Davies, Lovetri and colleagues. It is engine-agnostic and contains **no
//! Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance
//! Audio source or derived code**; it is implemented purely from that publicly
//! documented concert-hall acoustics literature.

use bevy_math::ops;

use prism_audio_core::math::{Sample, db_to_linear};

use crate::material_library::{OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT};

/// The maximum seat-dip insertion loss in decibels at the notch, before the
/// grazing and row weights scale it down.
pub const MAX_SEAT_DIP_DB: Sample = 20.0;

/// Smallest divisor used to keep ratios finite for degenerate inputs.
const MIN_DIVISOR: Sample = 1e-9;

/// Width of each comb notch, in natural-log-frequency units (about 0.6 octave).
const NOTCH_SIGMA: Sample = 0.6;

/// Number of grazed rows at which the row weight reaches `1 - 1/e`.
const ROW_SATURATION: Sample = 6.0;

/// Number of odd quarter-wave comb harmonics summed into the dip shape.
const COMB_HARMONICS: usize = 3;

/// The seat-row and source / receiver geometry that drives the seat-dip effect.
///
/// Heights are measured above the plane of the seat tops. Build one directly or
/// start from [`SeatDipGeometry::default`] (typical raked-stalls values) and
/// override fields, then pass it to [`seat_dip_attenuation_db`] or
/// [`SeatDipEffect::new`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SeatDipGeometry {
    /// Front-to-back spacing between seat rows in metres.
    pub row_spacing_m: Sample,
    /// Height of the seat tops above the floor in metres (the resonator depth).
    pub seat_height_m: Sample,
    /// Source height above the seat plane in metres.
    pub source_height_m: Sample,
    /// Receiver height above the seat plane in metres.
    pub receiver_height_m: Sample,
    /// Horizontal source-to-receiver distance across the seats in metres.
    pub horizontal_distance_m: Sample,
}

impl Default for SeatDipGeometry {
    fn default() -> Self {
        Self {
            row_spacing_m: 0.9,
            seat_height_m: 0.45,
            source_height_m: 1.5,
            receiver_height_m: 1.2,
            horizontal_distance_m: 20.0,
        }
    }
}

impl SeatDipGeometry {
    /// Builds a geometry from its components. No clamping is applied here;
    /// degenerate or non-finite values are handled safely at evaluation time.
    #[must_use]
    pub fn new(
        row_spacing_m: Sample,
        seat_height_m: Sample,
        source_height_m: Sample,
        receiver_height_m: Sample,
        horizontal_distance_m: Sample,
    ) -> Self {
        Self {
            row_spacing_m,
            seat_height_m,
            source_height_m,
            receiver_height_m,
            horizontal_distance_m,
        }
    }

    /// Whether every field is finite.
    #[must_use]
    fn is_finite(&self) -> bool {
        self.row_spacing_m.is_finite()
            && self.seat_height_m.is_finite()
            && self.source_height_m.is_finite()
            && self.receiver_height_m.is_finite()
            && self.horizontal_distance_m.is_finite()
    }

    /// The sine of the ray elevation above the seat plane, in `[0, 1]`.
    ///
    /// Computed as `mean_height / sqrt(mean_height^2 + horizontal^2)` from the
    /// mean of the source and receiver heights above the seats. A grazing ray
    /// returns a value near `0`; a steep ray returns a value near `1`.
    #[must_use]
    fn grazing_sine(&self) -> Sample {
        let mean_h = 0.5 * (self.source_height_m.max(0.0) + self.receiver_height_m.max(0.0));
        let horizontal = self.horizontal_distance_m.max(0.0);
        let hyp = ops::sqrt(mean_h * mean_h + horizontal * horizontal);
        if hyp < MIN_DIVISOR {
            0.0
        } else {
            (mean_h / hyp).clamp(0.0, 1.0)
        }
    }
}

/// Evaluates the seat-dip insertion loss per octave band, in decibels.
///
/// The returned array is aligned with [`OCTAVE_BAND_CENTERS`]; each value is a
/// non-negative, finite attenuation (positive = attenuation). Degenerate
/// geometry (a non-positive seat height, a non-positive row spacing, a
/// non-positive distance) and any non-finite input yield an all-zero spectrum.
///
/// # Examples
///
/// ```
/// use prism_audio_spatial::seat_dip_effect::{seat_dip_attenuation_db, SeatDipGeometry};
///
/// let geometry = SeatDipGeometry::default();
/// let bands = seat_dip_attenuation_db(&geometry, 343.0);
/// // Every band is a finite, non-negative attenuation.
/// assert!(bands.iter().all(|a| a.is_finite() && *a >= 0.0));
/// // The low-mid bands dip more than the top octave.
/// assert!(bands[1] > bands[7]);
/// ```
#[must_use]
pub fn seat_dip_attenuation_db(
    geometry: &SeatDipGeometry,
    sound_speed: Sample,
) -> [Sample; OCTAVE_BAND_COUNT] {
    let mut out = [0.0; OCTAVE_BAND_COUNT];

    if !geometry.is_finite() || !sound_speed.is_finite() {
        return out;
    }
    let c = sound_speed;
    let seat_height = geometry.seat_height_m;
    let row_spacing = geometry.row_spacing_m;
    let horizontal = geometry.horizontal_distance_m;
    if c <= 0.0 || seat_height <= 0.0 || row_spacing <= 0.0 || horizontal <= 0.0 {
        return out;
    }

    // Fundamental quarter-wave resonance of the seat gaps.
    let f0 = c / (4.0 * seat_height);
    if !f0.is_finite() || f0 <= 0.0 {
        return out;
    }

    // Grazing weight: full dip at grazing incidence, none at vertical.
    let grazing_weight = (1.0 - geometry.grazing_sine()).clamp(0.0, 1.0);

    // Row weight: saturating with the number of grazed rows.
    let rows_crossed = horizontal / row_spacing;
    let row_weight = (1.0 - ops::exp(-rows_crossed / ROW_SATURATION)).clamp(0.0, 1.0);

    let scale = MAX_SEAT_DIP_DB * grazing_weight * row_weight;
    if scale <= 0.0 {
        return out;
    }

    for (slot, &centre) in out.iter_mut().zip(OCTAVE_BAND_CENTERS.iter()) {
        let comb = comb_shape(centre, f0);
        *slot = (scale * comb).clamp(0.0, MAX_SEAT_DIP_DB);
    }
    out
}

/// Sums the odd quarter-wave comb notches at `(2k + 1) * f0` evaluated at
/// `freq_hz`, each a Gaussian in `log(frequency)` weighted by `1 / (2k + 1)`.
#[must_use]
fn comb_shape(freq_hz: Sample, f0: Sample) -> Sample {
    if freq_hz <= 0.0 {
        return 0.0;
    }
    let log_f = ops::ln(freq_hz);
    let mut sum = 0.0;
    let mut k = 0;
    while k < COMB_HARMONICS {
        let harmonic = (2 * k + 1) as Sample;
        let centre = harmonic * f0;
        let log_c = ops::ln(centre);
        let z = (log_f - log_c) / NOTCH_SIGMA;
        sum += ops::exp(-0.5 * z * z) / harmonic;
        k += 1;
    }
    sum
}

/// A seat-dip estimator built from a [`SeatDipGeometry`] and a sound speed.
///
/// Caches the per-octave-band attenuation so repeated queries are plain array
/// reads. Mirrors the query surface of the sibling attenuation modules
/// ([`crate::ground_effect`], [`crate::diffraction`]): per-band decibels,
/// per-band linear gains, and a broadband value interpolated in
/// `log(frequency)`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SeatDipEffect {
    /// The source geometry.
    geometry: SeatDipGeometry,
    /// The sound speed in metres per second used for the resonance frequency.
    sound_speed: Sample,
    /// Cached per-octave-band attenuation in decibels.
    bands: [Sample; OCTAVE_BAND_COUNT],
}

impl SeatDipEffect {
    /// Builds an estimator from a geometry and a sound speed, evaluating and
    /// caching the per-octave-band attenuation.
    #[must_use]
    pub fn new(geometry: SeatDipGeometry, sound_speed: Sample) -> Self {
        let bands = seat_dip_attenuation_db(&geometry, sound_speed);
        Self {
            geometry,
            sound_speed,
            bands,
        }
    }

    /// The geometry this estimator was built from.
    #[must_use]
    pub fn geometry(&self) -> SeatDipGeometry {
        self.geometry
    }

    /// The sound speed this estimator was built with.
    #[must_use]
    pub fn sound_speed(&self) -> Sample {
        self.sound_speed
    }

    /// The seat-dip attenuation in decibels for one octave band. A band index
    /// at or beyond [`OCTAVE_BAND_COUNT`] is clamped to the highest band.
    #[must_use]
    pub fn attenuation_db(&self, band_index: usize) -> Sample {
        let index = band_index.min(OCTAVE_BAND_COUNT - 1);
        self.bands[index]
    }

    /// The per-octave-band seat-dip attenuation in decibels, aligned with
    /// [`OCTAVE_BAND_CENTERS`].
    #[must_use]
    pub fn band_attenuations_db(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        self.bands
    }

    /// The per-octave-band linear gains, i.e. `10^(-attenuation_db / 20)`.
    #[must_use]
    pub fn band_gains(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        let mut gains = [0.0; OCTAVE_BAND_COUNT];
        for (slot, &att) in gains.iter_mut().zip(self.bands.iter()) {
            *slot = db_to_linear(-att);
        }
        gains
    }

    /// The broadband seat-dip attenuation in decibels at `freq_hz`, interpolated
    /// linearly in `log(frequency)` between the two bracketing octave-band
    /// values.
    ///
    /// Frequencies at or below 63 Hz return the lowest band; frequencies at or
    /// above 8 kHz return the highest band (clamped, never extrapolated). At a
    /// band centre it equals that band's value.
    #[must_use]
    pub fn broadband_attenuation_db(&self, freq_hz: Sample) -> Sample {
        if !freq_hz.is_finite() || freq_hz <= OCTAVE_BAND_CENTERS[0] {
            return self.bands[0];
        }
        let last = OCTAVE_BAND_COUNT - 1;
        if freq_hz >= OCTAVE_BAND_CENTERS[last] {
            return self.bands[last];
        }
        let log_f = ops::ln(freq_hz);
        for (centres, vals) in OCTAVE_BAND_CENTERS.windows(2).zip(self.bands.windows(2)) {
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
        self.bands[last]
    }

    /// The broadband linear gain at `freq_hz`, i.e.
    /// `10^(-broadband_attenuation_db / 20)`.
    #[must_use]
    pub fn broadband_gain(&self, freq_hz: Sample) -> Sample {
        db_to_linear(-self.broadband_attenuation_db(freq_hz))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOUND_SPEED: Sample = 343.0;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    fn argmax_band(bands: &[Sample; OCTAVE_BAND_COUNT]) -> usize {
        let mut best = 0;
        let mut best_v = bands[0];
        for (i, &v) in bands.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best
    }

    #[test]
    fn band_count_is_eight() {
        let bands = seat_dip_attenuation_db(&SeatDipGeometry::default(), SOUND_SPEED);
        assert_eq!(bands.len(), OCTAVE_BAND_COUNT);
        assert_eq!(bands.len(), 8);
    }

    #[test]
    fn attenuation_is_non_negative_and_finite() {
        let bands = seat_dip_attenuation_db(&SeatDipGeometry::default(), SOUND_SPEED);
        for a in bands {
            assert!(a.is_finite());
            assert!(a >= 0.0);
            assert!(a <= MAX_SEAT_DIP_DB + 1e-4);
        }
    }

    #[test]
    fn shallower_grazing_attenuates_more() {
        // Lower listener/source over the seats -> shallower grazing -> deeper dip.
        let grazing = SeatDipGeometry::new(0.9, 0.45, 0.3, 0.3, 25.0);
        let steep = SeatDipGeometry::new(0.9, 0.45, 4.0, 4.0, 25.0);
        let a_grazing = seat_dip_attenuation_db(&grazing, SOUND_SPEED);
        let a_steep = seat_dip_attenuation_db(&steep, SOUND_SPEED);
        let band = argmax_band(&a_grazing);
        assert!(a_grazing[band] > a_steep[band]);
    }

    #[test]
    fn low_frequency_dips_more_than_high_frequency() {
        let bands = seat_dip_attenuation_db(&SeatDipGeometry::default(), SOUND_SPEED);
        // The 125 Hz band dips far more than the 8 kHz band.
        assert!(bands[1] > bands[7]);
        assert!(bands[2] > bands[6]);
    }

    #[test]
    fn near_vertical_incidence_has_negligible_dip() {
        // Very high source and receiver over a short span -> steep ray.
        let steep = SeatDipGeometry::new(0.9, 0.45, 20.0, 20.0, 1.0);
        let bands = seat_dip_attenuation_db(&steep, SOUND_SPEED);
        for a in bands {
            assert!(a < 1.0, "expected near-zero dip, got {a}");
        }
    }

    #[test]
    fn taller_seats_move_the_notch_lower() {
        // Larger seat height lowers f0 = c / (4 h), moving the peak band down.
        let shallow = SeatDipGeometry::new(0.9, 0.3, 0.3, 0.3, 30.0);
        let deep = SeatDipGeometry::new(0.9, 1.2, 0.3, 0.3, 30.0);
        let band_shallow = argmax_band(&seat_dip_attenuation_db(&shallow, SOUND_SPEED));
        let band_deep = argmax_band(&seat_dip_attenuation_db(&deep, SOUND_SPEED));
        assert!(band_deep <= band_shallow);
        assert!(band_deep < band_shallow || band_deep == 0);
    }

    #[test]
    fn more_rows_crossed_deepen_the_dip() {
        let few = SeatDipGeometry::new(2.0, 0.45, 0.3, 0.3, 4.0);
        let many = SeatDipGeometry::new(0.5, 0.45, 0.3, 0.3, 40.0);
        let band = 1;
        let a_few = seat_dip_attenuation_db(&few, SOUND_SPEED)[band];
        let a_many = seat_dip_attenuation_db(&many, SOUND_SPEED)[band];
        assert!(a_many >= a_few);
    }

    #[test]
    fn row_weight_is_monotonic_in_distance() {
        let band = 1;
        let short = SeatDipGeometry::new(0.9, 0.45, 0.3, 0.3, 5.0);
        let long = SeatDipGeometry::new(0.9, 0.45, 0.3, 0.3, 50.0);
        // Keep grazing comparable by using the same low heights; longer span is
        // shallower too, so attenuation can only grow.
        let a_short = seat_dip_attenuation_db(&short, SOUND_SPEED)[band];
        let a_long = seat_dip_attenuation_db(&long, SOUND_SPEED)[band];
        assert!(a_long >= a_short);
    }

    #[test]
    fn non_finite_inputs_are_safe() {
        let nan_geom = SeatDipGeometry::new(Sample::NAN, 0.45, 1.0, 1.0, 20.0);
        for a in seat_dip_attenuation_db(&nan_geom, SOUND_SPEED) {
            assert_eq!(a, 0.0);
        }
        let ok = SeatDipGeometry::default();
        for a in seat_dip_attenuation_db(&ok, Sample::INFINITY) {
            assert_eq!(a, 0.0);
        }
    }

    #[test]
    fn zero_geometry_is_safe() {
        let zero = SeatDipGeometry::new(0.0, 0.0, 0.0, 0.0, 0.0);
        for a in seat_dip_attenuation_db(&zero, SOUND_SPEED) {
            assert_eq!(a, 0.0);
        }
    }

    #[test]
    fn zero_seat_height_is_safe() {
        let geom = SeatDipGeometry::new(0.9, 0.0, 0.3, 0.3, 20.0);
        for a in seat_dip_attenuation_db(&geom, SOUND_SPEED) {
            assert_eq!(a, 0.0);
        }
    }

    #[test]
    fn negative_inputs_do_not_panic() {
        let geom = SeatDipGeometry::new(-1.0, -1.0, -1.0, -1.0, -1.0);
        for a in seat_dip_attenuation_db(&geom, SOUND_SPEED) {
            assert!(a.is_finite());
            assert_eq!(a, 0.0);
        }
    }

    #[test]
    fn band_gains_match_decibels() {
        let effect = SeatDipEffect::new(SeatDipGeometry::default(), SOUND_SPEED);
        let gains = effect.band_gains();
        let bands = effect.band_attenuations_db();
        for (g, att) in gains.iter().zip(bands.iter()) {
            assert!(approx(*g, db_to_linear(-att), 1e-6));
            assert!(*g <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn broadband_at_band_centre_equals_band_value() {
        let effect = SeatDipEffect::new(SeatDipGeometry::default(), SOUND_SPEED);
        for (band, &f) in OCTAVE_BAND_CENTERS.iter().enumerate() {
            let via_band = effect.attenuation_db(band);
            let via_broadband = effect.broadband_attenuation_db(f);
            assert!(approx(via_band, via_broadband, 1e-3), "band {band}");
        }
    }

    #[test]
    fn broadband_interpolates_between_bands() {
        let effect = SeatDipEffect::new(SeatDipGeometry::default(), SOUND_SPEED);
        let lo = effect.attenuation_db(1);
        let hi = effect.attenuation_db(2);
        let mid = effect.broadband_attenuation_db(177.0);
        let min = lo.min(hi);
        let max = lo.max(hi);
        assert!(mid >= min - 1e-4 && mid <= max + 1e-4);
    }

    #[test]
    fn broadband_clamps_endpoints() {
        let effect = SeatDipEffect::new(SeatDipGeometry::default(), SOUND_SPEED);
        assert!(approx(
            effect.broadband_attenuation_db(10.0),
            effect.attenuation_db(0),
            1e-6
        ));
        assert!(approx(
            effect.broadband_attenuation_db(40_000.0),
            effect.attenuation_db(OCTAVE_BAND_COUNT - 1),
            1e-6
        ));
        assert!(effect.broadband_attenuation_db(Sample::INFINITY).is_finite());
    }

    #[test]
    fn band_index_is_clamped() {
        let effect = SeatDipEffect::new(SeatDipGeometry::default(), SOUND_SPEED);
        assert_eq!(
            effect.attenuation_db(999),
            effect.attenuation_db(OCTAVE_BAND_COUNT - 1)
        );
    }

    #[test]
    fn accessors_round_trip() {
        let geom = SeatDipGeometry::default();
        let effect = SeatDipEffect::new(geom, SOUND_SPEED);
        assert_eq!(effect.geometry(), geom);
        assert!(approx(effect.sound_speed(), SOUND_SPEED, 0.0));
    }

    #[test]
    fn grazing_sine_is_bounded() {
        let geom = SeatDipGeometry::new(0.9, 0.45, 2.0, 2.0, 10.0);
        let s = geom.grazing_sine();
        assert!((0.0..=1.0).contains(&s));
    }
}
