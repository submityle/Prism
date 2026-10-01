//! Direction-aware early-reflection weighting.
//!
//! A directional source does not feed every early reflection equally: a bounce
//! that leaves the source off-axis (into the rear hemisphere of a cardioid
//! voice, say) carries less energy than one that leaves along the forward axis,
//! and because directivity grows with frequency the effect is stronger in the
//! high bands. This module applies that idea. It is a control-rate combiner
//! that takes a [`crate::source_directivity::SourceDirectivity`] plus a source
//! forward axis and weights a set of image-source early-reflection taps
//! (produced by [`crate::early_reflections`]) by the per-octave-band radiation
//! gain along each tap's emission direction.
//!
//! It performs no per-sample DSP and re-implements neither the directivity
//! pattern nor the image-source geometry: it consumes the public outputs of the
//! two source modules and multiplies them together.
//!
//! # Emission direction
//!
//! A [`crate::early_reflections::ReflectionTap`] exposes only its arrival
//! direction, the listener-local unit vector from the image source to the
//! listener (`-Z` forward, `+X` right, `+Y` up). For the zero-order direct
//! path the image coincides with the source, so that arrival direction equals
//! the direction from the source to the listener, which is exactly the
//! emission direction. For higher-order reflections the true emission direction
//! (source toward the first reflection point) is not recoverable from the tap
//! alone, so this module uses the tap arrival direction as a proxy for the
//! emission direction. This is exact for the direct path and a far-field
//! approximation for reflections, where the incoming ray direction is a
//! reasonable stand-in for the direction along which the source radiated.
//!
//! The supplied source forward axis must be expressed in the same frame as the
//! taps (the listener-local frame). The off-axis cosine for a tap is then the
//! dot product of the (unit) forward axis and the (unit) tap direction, fed to
//! the directivity pattern.
//!
//! # Weighting
//!
//! For a tap with linear amplitude `g` and off-axis cosine `c`, the weighted
//! per-band amplitude is `g * d(band, c)`, where `d` is the source radiation
//! gain. The total per-band energy over a tap set adds incoherently, i.e. the
//! sum of squared weighted amplitudes, and the broadband send gain is the
//! root-mean-square of those band energies.
//!
//! # Control rate, not audio rate
//!
//! Every query operates on stack scalars and fixed eight-element arrays; there
//! is no heap allocation, no locking, and no panicking. Degenerate and
//! non-finite inputs fall back to safe finite values (a zero forward axis
//! defaults to `-Z`, a non-finite tap direction is treated as on-axis).
//! All transcendental math routes through [`bevy_math::ops`].
//!
//! # Relationship
//!
//! This module is the combining layer between two authoritative sources and
//! owns none of their physics:
//!
//! - the radiation pattern comes from
//!   [`crate::source_directivity`] ([`SourceDirectivity`]),
//! - the reflection taps come from
//!   [`crate::early_reflections`] ([`ReflectionTap`]).
//!
//! Its octave bands line up with
//! [`crate::material_library::OCTAVE_BAND_CENTERS`], so the weighted spectra
//! can be applied to an octave-band graphic equaliser directly.
//!
//! # Provenance
//!
//! The idea that source directivity shapes the energy of individual early
//! reflections is standard geometrical room acoustics (a directional source
//! illuminates its image sources unevenly). This module is engine-agnostic and
//! contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code**; it is implemented purely by
//! composing this crate's own directivity and image-source modules.

use bevy_math::Vec3;
use bevy_math::ops;

use prism_audio_core::math::Sample;

use crate::early_reflections::ReflectionTap;
use crate::material_library::OCTAVE_BAND_COUNT;
use crate::source_directivity::{DirectivityPreset, SourceDirectivity};

/// A direction-aware early-reflection weighter.
///
/// Holds a source radiation directivity and a forward axis, then weights
/// image-source reflection taps by the per-octave-band radiation gain along
/// each tap's (proxy) emission direction.
///
/// Build one with [`DirectionalEarlyReflections::new`] or
/// [`DirectionalEarlyReflections::from_preset`], then query per-tap weighted
/// band spectra, per-tap broadband gains, or the aggregate per-band energy and
/// send gain of a whole tap set.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DirectionalEarlyReflections {
    /// The source radiation pattern applied to each tap.
    directivity: SourceDirectivity,
    /// The source forward axis (unit), in the listener-local tap frame.
    forward: Vec3,
}

impl DirectionalEarlyReflections {
    /// Builds a weighter from a directivity and a source forward axis.
    ///
    /// The forward axis is normalised; a zero or non-finite axis defaults to
    /// `-Z` (the listener-local forward direction), so the result is always a
    /// finite unit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use bevy_math::Vec3;
    /// use prism_audio_spatial::early_reflections::ReflectionTap;
    /// use prism_audio_spatial::reflection_directivity::DirectionalEarlyReflections;
    /// use prism_audio_spatial::source_directivity::{DirectivityPreset, SourceDirectivity};
    ///
    /// let dir = SourceDirectivity::from_preset(DirectivityPreset::Cardioid);
    /// let weighter = DirectionalEarlyReflections::new(dir, Vec3::NEG_Z);
    /// // A tap arriving from straight ahead is on-axis: full gain.
    /// let front = ReflectionTap {
    ///     delay_samples: 100,
    ///     gain: 1.0,
    ///     direction: Vec3::NEG_Z,
    ///     order: 1,
    ///     is_direct: false,
    /// };
    /// assert!((weighter.emission_cos(&front) - 1.0).abs() < 1e-6);
    /// assert!((weighter.tap_band_gains(&front)[0] - 1.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn new(directivity: SourceDirectivity, source_forward: Vec3) -> Self {
        Self {
            directivity,
            forward: sanitise_direction(source_forward, Vec3::NEG_Z),
        }
    }

    /// Builds a weighter from a named directivity preset and a forward axis.
    #[must_use]
    pub fn from_preset(preset: DirectivityPreset, source_forward: Vec3) -> Self {
        Self::new(SourceDirectivity::from_preset(preset), source_forward)
    }

    /// The source radiation directivity.
    #[must_use]
    pub fn directivity(&self) -> &SourceDirectivity {
        &self.directivity
    }

    /// The (unit) source forward axis in the listener-local tap frame.
    #[must_use]
    pub fn source_forward(&self) -> Vec3 {
        self.forward
    }

    /// The off-axis cosine between the source forward axis and a tap's (proxy)
    /// emission direction, clamped to `[-1, 1]`.
    ///
    /// A non-finite or zero tap direction is treated as on-axis (`1.0`).
    #[must_use]
    pub fn emission_cos(&self, tap: &ReflectionTap) -> Sample {
        let dir = sanitise_direction(tap.direction, self.forward);
        self.forward.dot(dir).clamp(-1.0, 1.0)
    }

    /// The weighted per-octave-band amplitude gain for a single tap.
    ///
    /// Each band is `tap.gain * d(band, cos)`, where `d` is the source
    /// radiation gain along the tap's emission direction.
    #[must_use]
    pub fn tap_band_gains(&self, tap: &ReflectionTap) -> [Sample; OCTAVE_BAND_COUNT] {
        let cos = self.emission_cos(tap);
        let g = finite_gain(tap.gain);
        let mut out = self.directivity.band_gains(cos);
        for slot in &mut out {
            *slot *= g;
        }
        out
    }

    /// The weighted broadband amplitude gain for a single tap at `freq_hz`.
    ///
    /// The directivity sharpness is interpolated in `log(frequency)` between the
    /// neighbouring octave bands (clamped, never extrapolated, at the end
    /// bands), applied to the radiation pattern, then scaled by the tap gain.
    #[must_use]
    pub fn tap_broadband_gain(&self, tap: &ReflectionTap, freq_hz: Sample) -> Sample {
        let cos = self.emission_cos(tap);
        let g = finite_gain(tap.gain);
        g * self.directivity.broadband_gain(cos, freq_hz)
    }

    /// The total per-octave-band energy of a tap set: the incoherent sum of
    /// squared weighted amplitudes over every tap.
    #[must_use]
    pub fn total_band_energy(&self, taps: &[ReflectionTap]) -> [Sample; OCTAVE_BAND_COUNT] {
        let mut energy = [0.0; OCTAVE_BAND_COUNT];
        for tap in taps {
            let band = self.tap_band_gains(tap);
            for (acc, &amp) in energy.iter_mut().zip(band.iter()) {
                *acc += amp * amp;
            }
        }
        energy
    }

    /// The broadband early-reflection send gain of a tap set: the
    /// root-mean-square of the per-band total energies.
    ///
    /// This is a single scalar amplitude suitable for an early-reflection send
    /// bus; it is `0` for an empty tap set.
    #[must_use]
    pub fn total_send_gain(&self, taps: &[ReflectionTap]) -> Sample {
        let energy = self.total_band_energy(taps);
        let mut sum = 0.0;
        for &e in &energy {
            sum += e;
        }
        let mean = sum / OCTAVE_BAND_COUNT as Sample;
        ops::sqrt(mean.max(0.0))
    }
}

/// Normalises a direction, falling back to `default` (already unit) when the
/// input is zero-length or non-finite.
fn sanitise_direction(dir: Vec3, default: Vec3) -> Vec3 {
    if !dir.is_finite() {
        return default;
    }
    let normalised = dir.normalize_or_zero();
    if normalised.length_squared() > 0.5 {
        normalised
    } else {
        default
    }
}

/// Clamps a tap gain to a finite, non-negative value.
fn finite_gain(gain: Sample) -> Sample {
    if gain.is_finite() { gain.max(0.0) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material_library::OCTAVE_BAND_CENTERS;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    fn tap(gain: Sample, direction: Vec3, is_direct: bool) -> ReflectionTap {
        ReflectionTap {
            delay_samples: 50,
            gain,
            direction,
            order: if is_direct { 0 } else { 1 },
            is_direct,
        }
    }

    #[test]
    fn omni_gains_equal_tap_gain() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Omni, Vec3::NEG_Z);
        // Omni radiates uniformly, so every band keeps the tap gain regardless
        // of direction.
        for &dir in &[Vec3::NEG_Z, Vec3::Z, Vec3::X, Vec3::Y] {
            let gains = w.tap_band_gains(&tap(0.5, dir, false));
            for (band, &v) in gains.iter().enumerate() {
                assert!(approx(v, 0.5, 1e-6), "band {band}");
            }
        }
    }

    #[test]
    fn omni_total_band_energy_is_uniform() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Omni, Vec3::NEG_Z);
        let taps = [
            tap(0.5, Vec3::NEG_Z, true),
            tap(0.25, Vec3::X, false),
            tap(0.125, Vec3::Y, false),
        ];
        let energy = w.total_band_energy(&taps);
        let expected = 0.5 * 0.5 + 0.25 * 0.25 + 0.125 * 0.125;
        for (band, &v) in energy.iter().enumerate() {
            assert!(approx(v, expected, 1e-6), "band {band}");
        }
    }

    #[test]
    fn cardioid_rear_tap_is_silenced() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Cardioid, Vec3::NEG_Z);
        // A tap arriving from directly behind the forward axis is a rear-null.
        let rear = tap(1.0, Vec3::Z, false);
        let gains = w.tap_band_gains(&rear);
        for (band, &v) in gains.iter().enumerate() {
            assert!(approx(v, 0.0, 1e-6), "band {band}");
        }
    }

    #[test]
    fn cardioid_on_axis_is_full_gain() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Cardioid, Vec3::NEG_Z);
        let front = tap(0.8, Vec3::NEG_Z, false);
        let gains = w.tap_band_gains(&front);
        for (band, &v) in gains.iter().enumerate() {
            assert!(approx(v, 0.8, 1e-6), "band {band}");
        }
    }

    #[test]
    fn sharper_high_band_attenuates_off_axis_more() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Trumpet, Vec3::NEG_Z);
        // A 90-degree side tap: the sharp high band is attenuated more than the
        // near-omni low band.
        let side = tap(1.0, Vec3::X, false);
        let gains = w.tap_band_gains(&side);
        let last = OCTAVE_BAND_COUNT - 1;
        assert!(gains[last] < gains[0], "hi {} lo {}", gains[last], gains[0]);
    }

    #[test]
    fn emission_cos_matches_geometry() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Omni, Vec3::NEG_Z);
        assert!(approx(w.emission_cos(&tap(1.0, Vec3::NEG_Z, false)), 1.0, 1e-6));
        assert!(approx(w.emission_cos(&tap(1.0, Vec3::Z, false)), -1.0, 1e-6));
        assert!(approx(w.emission_cos(&tap(1.0, Vec3::X, false)), 0.0, 1e-6));
    }

    #[test]
    fn total_send_gain_matches_manual_for_omni() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Omni, Vec3::NEG_Z);
        let taps = [tap(0.6, Vec3::NEG_Z, true), tap(0.3, Vec3::X, false)];
        // Omni: every band energy equals gain0^2 + gain1^2, so the RMS over
        // bands is just sqrt(that).
        let per_band = 0.6 * 0.6 + 0.3 * 0.3;
        assert!(approx(w.total_send_gain(&taps), ops::sqrt(per_band), 1e-6));
    }

    #[test]
    fn broadband_at_band_centre_equals_band_value() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Trumpet, Vec3::X);
        let t = tap(0.7, Vec3::NEG_Z, false);
        let gains = w.tap_band_gains(&t);
        for (band, &freq) in OCTAVE_BAND_CENTERS.iter().enumerate() {
            let via_broadband = w.tap_broadband_gain(&t, freq);
            assert!(approx(via_broadband, gains[band], 1e-4), "band {band}");
        }
    }

    #[test]
    fn broadband_interpolates_between_bands() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Trumpet, Vec3::X);
        let t = tap(1.0, Vec3::NEG_Z, false);
        let gains = w.tap_band_gains(&t);
        // 177 Hz lies between the 125 Hz (index 1) and 250 Hz (index 2) bands.
        let mid = w.tap_broadband_gain(&t, 177.0);
        let (lo, hi) = (gains[1], gains[2]);
        let (min, max) = if lo < hi { (lo, hi) } else { (hi, lo) };
        assert!(mid >= min - 1e-4 && mid <= max + 1e-4, "mid {mid} lo {lo} hi {hi}");
    }

    #[test]
    fn broadband_clamps_endpoints() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Trumpet, Vec3::X);
        let t = tap(1.0, Vec3::NEG_Z, false);
        let gains = w.tap_band_gains(&t);
        let low = w.tap_broadband_gain(&t, 20.0);
        let high = w.tap_broadband_gain(&t, 20_000.0);
        assert!(approx(low, gains[0], 1e-4));
        assert!(approx(high, gains[OCTAVE_BAND_COUNT - 1], 1e-4));
    }

    #[test]
    fn zero_forward_defaults_to_neg_z() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Cardioid, Vec3::ZERO);
        assert!(approx(w.source_forward().length(), 1.0, 1e-6));
        // Default forward is -Z, so a -Z tap is on-axis.
        assert!(approx(w.emission_cos(&tap(1.0, Vec3::NEG_Z, false)), 1.0, 1e-6));
    }

    #[test]
    fn non_finite_inputs_are_safe() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Trumpet, Vec3::NEG_Z);
        let bad = tap(Sample::NAN, Vec3::new(Sample::NAN, 1.0, 0.0), false);
        let gains = w.tap_band_gains(&bad);
        for (band, &v) in gains.iter().enumerate() {
            assert!(v.is_finite(), "band {band}");
        }
        assert!(w.tap_broadband_gain(&bad, Sample::INFINITY).is_finite());
        assert!(w.total_send_gain(&[bad]).is_finite());
    }

    #[test]
    fn empty_tap_set_has_zero_energy() {
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Voice, Vec3::NEG_Z);
        let energy = w.total_band_energy(&[]);
        for (band, &v) in energy.iter().enumerate() {
            assert!(approx(v, 0.0, 1e-9), "band {band}");
        }
        assert!(approx(w.total_send_gain(&[]), 0.0, 1e-9));
    }

    #[test]
    fn direct_path_uses_exact_emission_direction() {
        // For the direct path the tap direction is exactly the emission
        // direction, so a cardioid facing the listener passes it at full gain.
        let w = DirectionalEarlyReflections::from_preset(DirectivityPreset::Cardioid, Vec3::NEG_Z);
        let direct = tap(1.0, Vec3::NEG_Z, true);
        let gains = w.tap_band_gains(&direct);
        for (band, &v) in gains.iter().enumerate() {
            assert!(approx(v, 1.0, 1e-6), "band {band}");
        }
    }
}
