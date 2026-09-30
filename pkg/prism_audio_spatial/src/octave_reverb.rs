//! Octave-band reverberation time: a per-band RT60 spectrum driving a
//! multi-band feedback-delay-network (FDN) reverb.
//!
//! Real rooms do not have a single reverberation time. Surface absorption rises
//! with frequency (porous materials, air absorption), so the high-frequency
//! tail decays faster than the low-frequency tail -- the characteristic "warm"
//! late field. [`crate::room_acoustics`] collapses a room to one broadband
//! RT60; this module keeps the frequency dependence by computing an
//! Eyring reverberation time in **each** standard octave band from the
//! per-band, per-surface absorption spectra supplied by
//! [`crate::material_library`].
//!
//! The output is an eight-band RT60 spectrum aligned with
//! [`OCTAVE_BAND_CENTERS`], queryable at any frequency (log-frequency
//! interpolation) and convertible to per-band FDN feedback gains for a
//! frequency-dependent reverberator.
//!
//! # The model (classic statistical acoustics)
//!
//! For each octave band the six face absorption coefficients are combined into
//! a band total absorption `A_band = sum_i(S_i * alpha_i(band))` and a band mean
//! `alpha_bar(band) = A_band / S`, then fed to the Eyring reverberation formula
//! `RT60(band) = k * V / (-S * ln(1 - alpha_bar(band)))` with the metric Sabine
//! constant `k = 0.161 s/m`. Eyring (rather than Sabine) is used because it
//! stays physical as absorption approaches unity. Geometry (volume `V`, surface
//! area `S`, per-axis face areas) comes from the [`ShoeboxRoom`] exactly as in
//! [`crate::room_acoustics`].
//!
//! An FDN with a delay line of length `delay` seconds must lose 60 dB of level
//! over `RT60` seconds, so each pass multiplies by
//! `g = 10^(-3 * delay / RT60)` (since `-60 dB` is a factor `10^-3`). This is
//! the per-band feedback gain [`OctaveReverb::fdn_decay_gains`] returns.
//!
//! # Control rate, not audio rate
//!
//! Everything here is a small scalar computation over eight bands: no
//! allocation, no locking, no panics, and no `f32` intrinsics (all logarithms
//! and exponentials route through [`bevy_math::ops`]). Degenerate geometry
//! (zero volume, full absorption, non-positive RT60 or delay) returns safe
//! finite values, never a `NaN` or an infinity.
//!
//! # Provenance
//!
//! This is the textbook per-band application of C. F. Eyring's reverberation
//! formula (*J. Acoust. Soc. Am.*, 1930), as presented in standard room
//! acoustics references such as H. Kuttruff's *Room Acoustics* and L. Beranek's
//! *Acoustics*; the `-60 dB` FDN decay-gain relation is the classic
//! Schroeder/Jot reverberator time-constant. This module is engine-agnostic and
//! contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code**; it is implemented purely
//! from that publicly documented acoustics knowledge.

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::early_reflections::ShoeboxRoom;
use crate::material_library::{
    Material, MaterialAbsorption, OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT,
};
use crate::room_acoustics::eyring_rt60;

/// Smallest divisor used to keep ratios finite for degenerate geometry.
const MIN_DIVISOR: Sample = 1e-9;

/// Level ratio for a 60 dB decay, `10^(-60/20) = 10^-3`. The FDN feedback gain
/// per delay pass is this factor raised to `delay / RT60`.
const MINUS_60_DB_RATIO_EXPONENT: Sample = -3.0;

/// A per-octave-band reverberation-time spectrum in seconds.
///
/// The eight values line up with [`OCTAVE_BAND_CENTERS`] (63 Hz to 8 kHz). A
/// well-behaved spectrum falls with frequency because surface absorption rises
/// with frequency, so the high bands decay faster than the low bands.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OctaveReverb {
    /// RT60 in seconds per octave band, aligned with [`OCTAVE_BAND_CENTERS`].
    rt60: [Sample; OCTAVE_BAND_COUNT],
}

impl OctaveReverb {
    /// Builds the spectrum from a room and one [`MaterialAbsorption`] per face.
    ///
    /// `per_face` follows the [`ShoeboxRoom::wall_absorption`] face order
    /// `[x_low, x_high, y_low, y_high, z_low, z_high]`. For every octave band
    /// the six faces contribute their band absorption weighted by face area,
    /// and the band RT60 is the Eyring reverberation time for that band mean
    /// absorption.
    ///
    /// [`ShoeboxRoom::wall_absorption`]: crate::early_reflections::ShoeboxRoom::wall_absorption
    #[must_use]
    pub fn from_shoebox_material(
        room: &ShoeboxRoom,
        per_face: &[MaterialAbsorption; 6],
    ) -> Self {
        let size = room.size();
        let lx = size.x.max(0.0);
        let ly = size.y.max(0.0);
        let lz = size.z.max(0.0);
        let volume = (lx * ly * lz).max(0.0);

        // Face areas by axis: the two x-faces each span (ly * lz), and so on,
        // matching room_acoustics::from_shoebox.
        let area_x = ly * lz;
        let area_y = lx * lz;
        let area_z = lx * ly;
        let surface_area = (2.0 * (area_x + area_y + area_z)).max(0.0);

        let faces = [
            per_face[0].bands(),
            per_face[1].bands(),
            per_face[2].bands(),
            per_face[3].bands(),
            per_face[4].bands(),
            per_face[5].bands(),
        ];

        let mut rt60 = [0.0; OCTAVE_BAND_COUNT];
        for (band, slot) in rt60.iter_mut().enumerate() {
            let total_absorption = area_x * (faces[0][band] + faces[1][band])
                + area_y * (faces[2][band] + faces[3][band])
                + area_z * (faces[4][band] + faces[5][band]);
            let mean_absorption = total_absorption / surface_area.max(MIN_DIVISOR);
            *slot = eyring_rt60(volume, surface_area, mean_absorption);
        }
        Self { rt60 }
    }

    /// Convenience: all six faces share one [`Material`].
    ///
    /// # Examples
    ///
    /// ```
    /// use bevy_math::Vec3;
    /// use prism_audio_spatial::early_reflections::ShoeboxRoom;
    /// use prism_audio_spatial::material_library::Material;
    /// use prism_audio_spatial::octave_reverb::OctaveReverb;
    ///
    /// let room = ShoeboxRoom::new(Vec3::ZERO, Vec3::new(10.0, 4.0, 8.0), [0.0; 6]);
    /// let hard = OctaveReverb::uniform(&room, Material::Concrete);
    /// let soft = OctaveReverb::uniform(&room, Material::AcousticTile);
    ///
    /// // A hard concrete room reverberates longer than an absorptive one.
    /// assert!(hard.broadband_rt60() > soft.broadband_rt60());
    ///
    /// // Feedback gains for a 50 ms delay line stay strictly below unity.
    /// for g in hard.fdn_decay_gains(0.05) {
    ///     assert!((0.0..1.0).contains(&g));
    /// }
    /// ```
    #[must_use]
    pub fn uniform(room: &ShoeboxRoom, material: Material) -> Self {
        let absorption = material.absorption();
        let per_face = [absorption; 6];
        Self::from_shoebox_material(room, &per_face)
    }

    /// Builds a spectrum directly from eight RT60 values (seconds), clamping
    /// each to be non-negative.
    #[must_use]
    pub fn from_bands(rt60: [Sample; OCTAVE_BAND_COUNT]) -> Self {
        let mut clamped = rt60;
        for r in &mut clamped {
            *r = r.max(0.0);
        }
        Self { rt60: clamped }
    }

    /// The eight per-band RT60 values (seconds), aligned with
    /// [`OCTAVE_BAND_CENTERS`].
    #[must_use]
    pub fn rt60_bands(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        self.rt60
    }

    /// RT60 (seconds) at an arbitrary frequency, interpolated linearly in
    /// `log(frequency)` between the two bracketing octave bands.
    ///
    /// Frequencies at or below 63 Hz return the lowest band, at or above 8 kHz
    /// return the highest band (clamped, never extrapolated). A non-finite or
    /// non-positive frequency falls back to the lowest band.
    #[must_use]
    pub fn rt60_at(&self, freq_hz: Sample) -> Sample {
        if !freq_hz.is_finite() || freq_hz <= OCTAVE_BAND_CENTERS[0] {
            return self.rt60[0];
        }
        let last = OCTAVE_BAND_COUNT - 1;
        if freq_hz >= OCTAVE_BAND_CENTERS[last] {
            return self.rt60[last];
        }
        let log_f = ops::ln(freq_hz);
        for (centres, values) in
            OCTAVE_BAND_CENTERS.windows(2).zip(self.rt60.windows(2))
        {
            let c_lo = centres[0];
            let c_hi = centres[1];
            if freq_hz <= c_hi {
                let log_lo = ops::ln(c_lo);
                let log_hi = ops::ln(c_hi);
                let span = log_hi - log_lo;
                let t = if span > 0.0 { (log_f - log_lo) / span } else { 0.0 };
                let v = values[0] + (values[1] - values[0]) * t;
                return v.max(0.0);
            }
        }
        self.rt60[last]
    }

    /// A single broadband RT60 (seconds): the mean of the 500 Hz and 1 kHz
    /// bands.
    ///
    /// These two mid-frequency octaves are the conventional single-number
    /// reverberation time (`T_mid`) in room-acoustics practice.
    #[must_use]
    pub fn broadband_rt60(&self) -> Sample {
        // Indices 3 and 4 are the 500 Hz and 1000 Hz bands.
        0.5 * (self.rt60[3] + self.rt60[4])
    }

    /// Per-band FDN feedback gains for a delay line of `delay_seconds`.
    ///
    /// Each gain is `g = 10^(-3 * delay / RT60)`, the factor that loses 60 dB
    /// over one RT60. Bands with a non-positive RT60, and a non-positive delay,
    /// map to gain `0` (immediate decay) rather than a `NaN` or an infinity.
    /// Every gain is clamped to `[0, 1)` so the loop is always stable.
    #[must_use]
    pub fn fdn_decay_gains(&self, delay_seconds: Sample) -> [Sample; OCTAVE_BAND_COUNT] {
        let mut gains = [0.0; OCTAVE_BAND_COUNT];
        if !delay_seconds.is_finite() || delay_seconds <= 0.0 {
            return gains;
        }
        let ln10 = core::f32::consts::LN_10;
        for (slot, &rt) in gains.iter_mut().zip(self.rt60.iter()) {
            if rt > 0.0 {
                let exponent = MINUS_60_DB_RATIO_EXPONENT * delay_seconds / rt;
                let g = ops::exp(exponent * ln10);
                // Keep strictly below unity so the FDN loop stays stable.
                *slot = g.clamp(0.0, 0.999_999);
            }
        }
        gains
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn test_room() -> ShoeboxRoom {
        ShoeboxRoom::new(Vec3::ZERO, Vec3::new(10.0, 4.0, 8.0), [0.0; 6])
    }

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn hard_room_reverberates_longer_than_soft_room() {
        let room = test_room();
        let hard = OctaveReverb::uniform(&room, Material::Concrete);
        let soft = OctaveReverb::uniform(&room, Material::AcousticTile);
        let hb = hard.rt60_bands();
        let sb = soft.rt60_bands();
        for (h, s) in hb.iter().zip(sb.iter()) {
            assert!(h > s, "hard {h} !> soft {s}");
        }
        assert!(hard.broadband_rt60() > soft.broadband_rt60());
    }

    #[test]
    fn high_frequency_decays_faster_than_low_for_carpet() {
        // Carpet absorbs much more at high frequency, so high-band RT60 < low.
        let room = test_room();
        let rev = OctaveReverb::uniform(&room, Material::Carpet);
        let bands = rev.rt60_bands();
        // 4 kHz band (index 6) should decay faster than 250 Hz band (index 2).
        assert!(bands[6] < bands[2], "hi {} !< lo {}", bands[6], bands[2]);
    }

    #[test]
    fn all_rt60_values_are_finite_and_non_negative() {
        let room = test_room();
        for material in [Material::Concrete, Material::Carpet, Material::Water] {
            for r in OctaveReverb::uniform(&room, material).rt60_bands() {
                assert!(r.is_finite() && r >= 0.0, "bad rt60 {r}");
            }
        }
    }

    #[test]
    fn full_absorption_is_near_zero_and_finite() {
        let room = test_room();
        let per_face = [MaterialAbsorption::new([1.0; OCTAVE_BAND_COUNT]); 6];
        let rev = OctaveReverb::from_shoebox_material(&room, &per_face);
        for r in rev.rt60_bands() {
            assert!(r.is_finite(), "not finite {r}");
            assert!(r < 0.2, "expected near-zero, got {r}");
        }
    }

    #[test]
    fn zero_volume_room_is_safe() {
        let room = ShoeboxRoom::new(Vec3::ZERO, Vec3::ZERO, [0.2; 6]);
        let rev = OctaveReverb::uniform(&room, Material::Concrete);
        for r in rev.rt60_bands() {
            assert!(approx(r, 0.0, 1e-6), "expected 0, got {r}");
        }
    }

    #[test]
    fn rt60_at_band_center_returns_that_band() {
        let room = test_room();
        let rev = OctaveReverb::uniform(&room, Material::Wood);
        let bands = rev.rt60_bands();
        for (i, centre) in OCTAVE_BAND_CENTERS.iter().enumerate() {
            assert!(approx(rev.rt60_at(*centre), bands[i], 1e-3), "band {i}");
        }
    }

    #[test]
    fn rt60_at_clamps_beyond_range() {
        let room = test_room();
        let rev = OctaveReverb::uniform(&room, Material::Brick);
        let bands = rev.rt60_bands();
        assert!(approx(rev.rt60_at(10.0), bands[0], 1e-4));
        assert!(approx(rev.rt60_at(20000.0), bands[OCTAVE_BAND_COUNT - 1], 1e-4));
        // Degenerate frequencies do not panic.
        let _ = rev.rt60_at(Sample::NAN);
        let _ = rev.rt60_at(-5.0);
        let _ = rev.rt60_at(0.0);
    }

    #[test]
    fn rt60_at_interpolates_between_bands() {
        let rev = OctaveReverb::from_bands([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]);
        // 707 Hz is the geometric midpoint of the 500 and 1000 Hz bands, whose
        // RT60 values are 1.0 and 0.5; log-linear interpolation gives ~0.75.
        let v = rev.rt60_at(707.0);
        assert!(approx(v, 0.75, 0.02), "v={v}");
    }

    #[test]
    fn from_bands_clamps_negatives() {
        let rev = OctaveReverb::from_bands([-1.0, 0.5, -0.2, 2.0, 0.0, 1.0, -3.0, 0.3]);
        let bands = rev.rt60_bands();
        assert!(bands.iter().all(|&r| r >= 0.0));
        assert!(approx(bands[0], 0.0, 1e-9));
        assert!(approx(bands[3], 2.0, 1e-9));
    }

    #[test]
    fn fdn_gains_are_monotonic_in_rt60() {
        // Longer RT60 -> gain closer to 1 for a fixed delay.
        let rev = OctaveReverb::from_bands([2.0, 1.0, 0.5, 0.25, 0.5, 1.0, 2.0, 4.0]);
        let gains = rev.fdn_decay_gains(0.05);
        assert!(gains.iter().all(|&g| (0.0..1.0).contains(&g)));
        // Band 0 (RT60 2.0) has a larger gain than band 3 (RT60 0.25).
        assert!(gains[0] > gains[3], "{} !> {}", gains[0], gains[3]);
        // Band 7 (RT60 4.0) has the largest gain overall.
        assert!(gains[7] > gains[0]);
    }

    #[test]
    fn fdn_gain_matches_minus_60_db_over_one_rt60() {
        // Over exactly one RT60 of accumulated delay, the level should drop by
        // 60 dB, i.e. a factor of 1e-3. With delay == rt60 the single-pass gain
        // is 10^-3.
        let rt = 0.5;
        let rev = OctaveReverb::from_bands([rt; OCTAVE_BAND_COUNT]);
        let gains = rev.fdn_decay_gains(rt);
        for g in gains {
            assert!(approx(g, 1e-3, 1e-5), "g={g}");
        }
    }

    #[test]
    fn fdn_gains_degrade_gracefully() {
        // Non-positive delay -> all zero. Zero RT60 band -> zero gain.
        let rev = OctaveReverb::from_bands([0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0]);
        assert!(rev.fdn_decay_gains(0.0).iter().all(|&g| g == 0.0));
        assert!(rev.fdn_decay_gains(-1.0).iter().all(|&g| g == 0.0));
        let gains = rev.fdn_decay_gains(0.05);
        assert!(approx(gains[1], 0.0, 1e-9), "zero-rt60 band should be 0");
        assert!(gains[0] > 0.0);
        assert!(gains.iter().all(|&g| g.is_finite()));
    }

    #[test]
    fn broadband_is_mean_of_mid_bands() {
        let rev = OctaveReverb::from_bands([2.0, 2.0, 2.0, 0.8, 1.2, 0.5, 0.5, 0.5]);
        assert!(approx(rev.broadband_rt60(), 1.0, 1e-6));
    }
}
