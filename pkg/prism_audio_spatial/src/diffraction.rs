//! Edge / barrier diffraction insertion loss (Maekawa).
//!
//! When a rigid screen or an occluding edge stands between a source and a
//! listener, the sound that reaches the listener has to bend around the edge.
//! The extra path it travels relative to the blocked straight line is the
//! diffraction *path-length difference* `delta`, and the resulting attenuation
//! in the geometric shadow is the classic **Maekawa (1968)** barrier insertion
//! loss. This module is a control-rate estimator: it turns a single path
//! difference into a frequency-dependent (per octave band) insertion loss in
//! decibels and the matching linear gains, ready to drive an occluded or
//! diffracted propagation path. It performs no per-sample DSP.
//!
//! # Fresnel number and insertion loss
//!
//! For a diffracting edge the Fresnel number is
//!
//! `N = 2 * delta / lambda = 2 * delta * f / c`
//!
//! where `delta` is the path-length difference in metres, `f` is the frequency
//! in hertz, and `c` is the speed of sound. `N` grows with both the detour and
//! the frequency, which is why high frequencies are shadowed more strongly. In
//! the geometric shadow (`N > 0`) the Maekawa insertion loss is
//!
//! `IL_dB = 5 + 20 * log10( x / tanh(x) )`, with `x = sqrt(2 * PI * N)`,
//!
//! clamped to an upper limit. This module reuses the crate's canonical Maekawa
//! curve, [`crate::propagation::maekawa_attenuation_db`], evaluated at a Fresnel
//! number computed with a configurable speed of sound, so the shape (including
//! the shadow-boundary floor near `5 dB` and the upper clamp) matches the rest
//! of the acoustics stack. A non-positive path difference means the listener is
//! not in the geometric shadow, so no diffraction loss is applied.
//!
//! # Difference from [`crate::propagation`]
//!
//! [`crate::propagation`] exposes per-frequency free functions
//! ([`crate::propagation::fresnel_number`],
//! [`crate::propagation::maekawa_attenuation_db`],
//! [`crate::propagation::diffraction_gain`]) that assume a fixed speed of
//! sound and are used to build filter cutoffs. This module packages the same
//! physics as a small stateful [`Diffraction`] value parameterised by the path
//! difference and a configurable speed of sound, and reports results over the
//! eight octave bands shared with [`crate::material_library`], aligning it with
//! [`crate::source_directivity`] and [`crate::reverberant_field`].
//!
//! # Control rate, not audio rate
//!
//! Every query operates on stack scalars and fixed-size arrays; there is no
//! heap allocation, no locking, and no panicking. Degenerate geometry (a
//! non-positive detour, a non-positive speed of sound) and non-finite inputs
//! return safe finite values. All transcendental math routes through
//! [`bevy_math::ops`], never through `f32` intrinsics.
//!
//! # Provenance
//!
//! This is the textbook barrier-diffraction model: the Fresnel number and the
//! Maekawa insertion-loss curve as presented in Z. Maekawa, "Noise reduction by
//! screens" (Applied Acoustics, 1968), the Kurze-Anderson formulation, and the
//! screen term of ISO 9613-2. This module is engine-agnostic and contains **no
//! Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance
//! Audio source or derived code**; it is implemented purely from that publicly
//! documented acoustics knowledge.

use bevy_math::Vec3;
use bevy_math::ops;


use prism_audio_core::math::{Sample, db_to_linear};

use crate::early_reflections::DEFAULT_SOUND_SPEED;
use crate::material_library::{OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT};
use crate::propagation::maekawa_attenuation_db;

/// Smallest divisor used to keep ratios finite for degenerate inputs.
const MIN_DIVISOR: Sample = 1e-9;

/// A barrier / edge diffraction estimator described by a path-length
/// difference and a speed of sound.
///
/// Build one with [`Diffraction::from_path_difference`] or
/// [`Diffraction::from_geometry`], then query the per-octave-band insertion
/// loss in decibels, the matching linear gains, or a broadband value at an
/// arbitrary frequency.
///
/// The octave bands line up with [`OCTAVE_BAND_CENTERS`], so the result can be
/// combined directly with [`crate::material_library`] absorption or
/// [`crate::source_directivity`] directivity spectra.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Diffraction {
    /// The diffraction path-length difference `delta` in metres. A value at or
    /// below zero means the listener is in the illuminated zone (no shadow).
    path_difference_m: Sample,
    /// The speed of sound in metres per second used to form the Fresnel number.
    sound_speed: Sample,
}

impl Diffraction {
    /// Builds a diffraction estimator from a path-length difference (metres)
    /// and a speed of sound (metres per second).
    ///
    /// A non-finite path difference becomes `0` (no shadow). A non-finite or
    /// non-positive speed of sound falls back to [`DEFAULT_SOUND_SPEED`].
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::diffraction::Diffraction;
    ///
    /// // A half-metre detour around an edge, at the standard speed of sound.
    /// let d = Diffraction::from_path_difference(0.5, 343.0);
    /// // High frequencies are shadowed more than low frequencies.
    /// let low = d.insertion_loss_db(0);
    /// let high = d.insertion_loss_db(7);
    /// assert!(high > low);
    /// ```
    #[must_use]
    pub fn from_path_difference(delta_m: Sample, sound_speed: Sample) -> Self {
        let path_difference_m = if delta_m.is_finite() { delta_m } else { 0.0 };
        let sound_speed = if sound_speed.is_finite() && sound_speed > 0.0 {
            sound_speed
        } else {
            DEFAULT_SOUND_SPEED
        };
        Self {
            path_difference_m,
            sound_speed,
        }
    }

    /// Builds a diffraction estimator from the diffraction geometry: the source
    /// position, the diffracting edge point, and the receiver position.
    ///
    /// The path-length difference is the detour over the edge minus the blocked
    /// straight line,
    ///
    /// `delta = (|source - edge| + |edge - receiver|) - |source - receiver|`,
    ///
    /// which is non-negative by the triangle inequality and is zero when the
    /// edge lies on the straight line.
    #[must_use]
    pub fn from_geometry(
        source: Vec3,
        edge: Vec3,
        receiver: Vec3,
        sound_speed: Sample,
    ) -> Self {
        let over = length(source - edge) + length(edge - receiver);
        let direct = length(source - receiver);
        Self::from_path_difference(over - direct, sound_speed)
    }

    /// The stored path-length difference `delta` in metres.
    #[must_use]
    pub fn path_difference(&self) -> Sample {
        self.path_difference_m
    }

    /// The stored speed of sound in metres per second.
    #[must_use]
    pub fn sound_speed(&self) -> Sample {
        self.sound_speed
    }

    /// The Fresnel number `N = 2 * delta * f / c` at frequency `freq_hz`.
    ///
    /// Non-finite frequencies return `0`. `N` is zero or negative when the
    /// listener is not in the geometric shadow.
    #[must_use]
    pub fn fresnel_number(&self, freq_hz: Sample) -> Sample {
        if !freq_hz.is_finite() {
            return 0.0;
        }
        let c = self.sound_speed.max(MIN_DIVISOR);
        2.0 * self.path_difference_m * freq_hz / c
    }

    /// The Maekawa insertion loss in decibels at an arbitrary frequency.
    ///
    /// Returns `0` when the listener is not in the geometric shadow (a
    /// non-positive path difference) and for non-finite frequencies.
    #[must_use]
    pub fn insertion_loss_db_at(&self, freq_hz: Sample) -> Sample {
        if self.path_difference_m <= 0.0 || !freq_hz.is_finite() {
            return 0.0;
        }
        maekawa_attenuation_db(self.fresnel_number(freq_hz))
    }

    /// The Maekawa insertion loss in decibels for octave band `band_index`.
    ///
    /// The index is clamped to the last band, so out-of-range indices are
    /// safe.
    #[must_use]
    pub fn insertion_loss_db(&self, band_index: usize) -> Sample {
        let band = band_index.min(OCTAVE_BAND_COUNT - 1);
        self.insertion_loss_db_at(OCTAVE_BAND_CENTERS[band])
    }

    /// The insertion loss in decibels for every octave band.
    #[must_use]
    pub fn band_losses_db(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        let mut out = [0.0; OCTAVE_BAND_COUNT];
        for (slot, &f) in out.iter_mut().zip(OCTAVE_BAND_CENTERS.iter()) {
            *slot = self.insertion_loss_db_at(f);
        }
        out
    }

    /// The linear gain `10^(-IL/20)` in `(0, 1]` for every octave band.
    #[must_use]
    pub fn band_gains(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        let mut out = [1.0; OCTAVE_BAND_COUNT];
        for (slot, &f) in out.iter_mut().zip(OCTAVE_BAND_CENTERS.iter()) {
            *slot = db_to_linear(-self.insertion_loss_db_at(f));
        }
        out
    }

    /// The broadband insertion loss in decibels at `freq_hz`, interpolated
    /// linearly in `log(frequency)` between the two bracketing octave-band
    /// values.
    ///
    /// Frequencies at or below 63 Hz return the lowest band; frequencies at or
    /// above 8 kHz return the highest band (the spectrum is clamped, never
    /// extrapolated). At a band centre it equals that band's value.
    #[must_use]
    pub fn broadband_loss_db(&self, freq_hz: Sample) -> Sample {
        let losses = self.band_losses_db();
        if !freq_hz.is_finite() || freq_hz <= OCTAVE_BAND_CENTERS[0] {
            return losses[0];
        }
        let last = OCTAVE_BAND_COUNT - 1;
        if freq_hz >= OCTAVE_BAND_CENTERS[last] {
            return losses[last];
        }
        let log_f = ops::ln(freq_hz);
        for (centres, values) in OCTAVE_BAND_CENTERS.windows(2).zip(losses.windows(2)) {
            let c_lo = centres[0];
            let c_hi = centres[1];
            if freq_hz <= c_hi {
                let log_lo = ops::ln(c_lo);
                let log_hi = ops::ln(c_hi);
                let span = log_hi - log_lo;
                let t = if span > MIN_DIVISOR {
                    (log_f - log_lo) / span
                } else {
                    0.0
                };
                return values[0] + (values[1] - values[0]) * t;
            }
        }
        losses[last]
    }

    /// The broadband linear gain in `(0, 1]` at `freq_hz`, i.e.
    /// `10^(-broadband_loss_db / 20)`.
    #[must_use]
    pub fn broadband_gain(&self, freq_hz: Sample) -> Sample {
        db_to_linear(-self.broadband_loss_db(freq_hz))
    }
}

/// Euclidean length of a vector using [`bevy_math::ops::sqrt`] on the dot
/// product, avoiding `Vec3::length` (which would route through an `f32`
/// intrinsic).
#[inline]
#[must_use]
fn length(v: Vec3) -> Sample {
    ops::sqrt(v.dot(v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::MAX_DIFFRACTION_DB;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn line_of_sight_has_no_loss() {
        // Zero and negative detours: listener not in shadow.
        let flat = Diffraction::from_path_difference(0.0, 343.0);
        for b in 0..OCTAVE_BAND_COUNT {
            assert!(approx(flat.insertion_loss_db(b), 0.0, 1e-9));
            assert!(approx(flat.band_gains()[b], 1.0, 1e-6));
        }
        let lit = Diffraction::from_path_difference(-1.0, 343.0);
        assert!(approx(lit.insertion_loss_db_at(1_000.0), 0.0, 1e-9));
    }

    #[test]
    fn high_frequencies_are_shadowed_more() {
        let d = Diffraction::from_path_difference(0.2, 343.0);
        let losses = d.band_losses_db();
        for pair in losses.windows(2) {
            assert!(pair[1] >= pair[0], "not monotonic: {pair:?}");
        }
        // Strictly greater across the full span for a genuine detour.
        assert!(losses[OCTAVE_BAND_COUNT - 1] > losses[0] + 5.0);
    }

    #[test]
    fn matches_known_maekawa_value_at_unit_fresnel_number() {
        // delta = c / (2 f) gives N = 1 at f. Pick f = 1000 Hz.
        let c = 343.0;
        let f = 1_000.0;
        let delta = c / (2.0 * f);
        let d = Diffraction::from_path_difference(delta, c);
        assert!(approx(d.fresnel_number(f), 1.0, 1e-4));
        // Maekawa IL at N = 1 is about 13.097 dB.
        assert!(approx(d.insertion_loss_db_at(f), 13.097, 1e-2));
    }

    #[test]
    fn loss_is_clamped_to_upper_limit() {
        // A huge detour drives every audible band to the clamp.
        let d = Diffraction::from_path_difference(100.0, 343.0);
        for b in 0..OCTAVE_BAND_COUNT {
            let il = d.insertion_loss_db(b);
            assert!(il <= MAX_DIFFRACTION_DB + 1e-3);
        }
        assert!(approx(
            d.insertion_loss_db(OCTAVE_BAND_COUNT - 1),
            MAX_DIFFRACTION_DB,
            1e-3
        ));
    }

    #[test]
    fn band_gains_match_losses() {
        let d = Diffraction::from_path_difference(0.3, 343.0);
        let losses = d.band_losses_db();
        let gains = d.band_gains();
        for (il, g) in losses.iter().zip(gains.iter()) {
            let expected = db_to_linear(-il);
            assert!(approx(*g, expected, 1e-6));
            assert!(*g > 0.0 && *g <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn broadband_at_band_centre_equals_band_value() {
        let d = Diffraction::from_path_difference(0.25, 343.0);
        for (b, &f) in OCTAVE_BAND_CENTERS.iter().enumerate() {
            let via_band = d.insertion_loss_db(b);
            let via_broadband = d.broadband_loss_db(f);
            assert!(approx(via_band, via_broadband, 1e-3), "band {b}");
        }
    }

    #[test]
    fn broadband_interpolates_between_bands() {
        let d = Diffraction::from_path_difference(0.25, 343.0);
        // A frequency between 1 kHz (band 4) and 2 kHz (band 5).
        let lo = d.insertion_loss_db(4);
        let hi = d.insertion_loss_db(5);
        let mid = d.broadband_loss_db(1_414.0);
        assert!(mid > lo && mid < hi, "mid {mid} lo {lo} hi {hi}");
    }

    #[test]
    fn broadband_clamps_endpoints() {
        let d = Diffraction::from_path_difference(0.25, 343.0);
        let low = d.broadband_loss_db(20.0);
        let high = d.broadband_loss_db(20_000.0);
        assert!(approx(low, d.insertion_loss_db(0), 1e-3));
        assert!(approx(high, d.insertion_loss_db(OCTAVE_BAND_COUNT - 1), 1e-3));
    }

    #[test]
    fn geometry_computes_path_difference() {
        // Source and receiver two metres apart on the x axis, edge one metre
        // above the midpoint: detour = 2 * sqrt(1 + 1) - 2 = 2*sqrt(2) - 2.
        let source = Vec3::new(-1.0, 0.0, 0.0);
        let receiver = Vec3::new(1.0, 0.0, 0.0);
        let edge = Vec3::new(0.0, 1.0, 0.0);
        let d = Diffraction::from_geometry(source, edge, receiver, 343.0);
        let expected = 2.0 * ops::sqrt(2.0) - 2.0;
        assert!(approx(d.path_difference(), expected, 1e-4));
        assert!(d.path_difference() > 0.0);
    }

    #[test]
    fn edge_on_line_has_zero_detour() {
        // Edge exactly on the straight line: no detour, no loss.
        let source = Vec3::new(-1.0, 0.0, 0.0);
        let receiver = Vec3::new(1.0, 0.0, 0.0);
        let edge = Vec3::new(0.0, 0.0, 0.0);
        let d = Diffraction::from_geometry(source, edge, receiver, 343.0);
        assert!(approx(d.path_difference(), 0.0, 1e-4));
        assert!(approx(d.insertion_loss_db_at(1_000.0), 0.0, 1e-9));
    }

    #[test]
    fn non_finite_and_degenerate_inputs_are_safe() {
        let nan = Diffraction::from_path_difference(Sample::NAN, Sample::NAN);
        assert!(nan.path_difference().is_finite());
        assert!(nan.sound_speed() > 0.0);
        assert!(approx(nan.sound_speed(), DEFAULT_SOUND_SPEED, 1e-6));
        let d = Diffraction::from_path_difference(0.5, 343.0);
        assert!(d.insertion_loss_db_at(Sample::NAN).is_finite());
        assert!(approx(d.insertion_loss_db_at(Sample::NAN), 0.0, 1e-9));
        assert!(d.fresnel_number(Sample::INFINITY).is_finite());
        // Zero speed of sound falls back to the default, not a divide by zero.
        let zc = Diffraction::from_path_difference(0.5, 0.0);
        assert!(zc.fresnel_number(1_000.0).is_finite());
        assert!(approx(zc.sound_speed(), DEFAULT_SOUND_SPEED, 1e-6));
    }

    #[test]
    fn higher_speed_of_sound_lowers_fresnel_number() {
        // N = 2 delta f / c, so a larger c shrinks N and the loss.
        let slow = Diffraction::from_path_difference(0.3, 300.0);
        let fast = Diffraction::from_path_difference(0.3, 400.0);
        assert!(fast.fresnel_number(2_000.0) < slow.fresnel_number(2_000.0));
        assert!(fast.insertion_loss_db_at(2_000.0) < slow.insertion_loss_db_at(2_000.0));
    }
}
