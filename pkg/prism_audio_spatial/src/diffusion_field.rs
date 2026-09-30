//! Late diffuse-field energy: a control-rate description of the scattered,
//! direction-less reverberant tail.
//!
//! [`crate::scattering`] splits each wall reflection into a specular part (which
//! keeps its mirror-image direction, handled by [`crate::early_reflections`])
//! and a diffuse part that radiates with a Lambert cosine law. Once enough
//! diffuse reflections have accumulated, the sound field becomes statistically
//! isotropic: the *late diffuse field*. This module aggregates the diffuse
//! energy shed by a set of surfaces and derives the classic statistical
//! descriptors a downstream feedback-delay-network (FDN) reverberator needs:
//!
//! - a per-octave-band diffuse energy density aligned with
//!   [`OCTAVE_BAND_CENTERS`];
//! - the Schroeder echo density (reflections per second), which grows with the
//!   square of time;
//! - the mixing time (early-to-late transition) from room volume;
//! - a diffusion coefficient (diffuse-to-reflected energy ratio);
//! - per-band late wet-send gains suggested for driving an FDN.
//!
//! It performs no per-sample DSP. The actual delay-line filtering lives in the
//! core reverb; here everything is a small scalar computation over eight bands.
//!
//! # The model (classic statistical acoustics)
//!
//! For a surface receiving incident band energy `E` with absorption `alpha` and
//! scattering coefficient `s`, the reflected energy is `E * (1 - alpha)` and the
//! diffuse share of that is `E * (1 - alpha) * s`. Summing over surfaces gives a
//! per-band diffuse energy density.
//!
//! The number of image sources arriving by time `t` in a room of volume `V` is
//! `N(t) = (4/3) * PI * c^3 * t^3 / V`, so the echo density (arrivals per
//! second) is its derivative `dN/dt = 4 * PI * c^3 * t^2 / V`: it rises with
//! `t^2` and is lower in larger rooms. The mixing time, the instant the field
//! is effectively diffuse, is classically estimated as `t_mix ~ sqrt(V)`
//! milliseconds (Polack / Jot).
//!
//! # Control rate, not audio rate
//!
//! Everything here is a small scalar computation: no allocation, no locking, no
//! panics, and no `f32` intrinsics (all roots route through
//! [`bevy_math::ops`]). Degenerate inputs (zero volume, empty surface set,
//! non-finite time) return safe finite values, never a `NaN` or an infinity.
//!
//! # Provenance
//!
//! This is the textbook statistical-acoustics description of the diffuse late
//! field: M. R. Schroeder's echo-density growth, the diffuse-field energy
//! balance in H. Kuttruff's *Room Acoustics*, and the mixing-time estimate from
//! J.-D. Polack and the Jot/Gardner FDN reverberator literature. This module is
//! engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented acoustics knowledge.

use bevy_math::ops;
use core::f32::consts::PI;

use prism_audio_core::math::Sample;

use crate::early_reflections::ShoeboxRoom;
use crate::material_library::{MaterialAbsorption, OCTAVE_BAND_COUNT};
use crate::octave_reverb::OctaveReverb;
use crate::scattering::{ScatteringSpectrum, diffuse_fraction};

/// The speed of sound in dry air at room temperature, in metres per second.
pub const DEFAULT_SOUND_SPEED: Sample = 343.0;

/// Upper clamp on echo density (reflections per second) to keep the estimate
/// finite for very late times or vanishing volumes.
pub const MAX_ECHO_DENSITY: Sample = 1.0e9;

/// Smallest divisor used to keep ratios finite for degenerate geometry.
const MIN_DIVISOR: Sample = 1e-9;

/// A control-rate aggregate of the late diffuse reverberant field.
///
/// Accumulates the diffuse energy shed by one or more surfaces and exposes the
/// classic statistical descriptors used to drive a late reverberator. The eight
/// band energies line up with [`OCTAVE_BAND_CENTERS`].
///
/// [`OCTAVE_BAND_CENTERS`]: crate::material_library::OCTAVE_BAND_CENTERS
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DiffusionField {
    /// Accumulated diffuse energy per octave band.
    diffuse_energy: [Sample; OCTAVE_BAND_COUNT],
    /// Accumulated total reflected energy per octave band (specular + diffuse).
    reflected_energy: [Sample; OCTAVE_BAND_COUNT],
    /// Room volume in cubic metres.
    volume: Sample,
    /// Room interior surface area in square metres.
    surface_area: Sample,
}

impl DiffusionField {
    /// Creates an empty field for a room, with zero accumulated energy.
    ///
    /// Volume and surface area are taken from the [`ShoeboxRoom`] geometry;
    /// call [`DiffusionField::add_surface`] to accumulate diffuse energy.
    #[must_use]
    pub fn new(room: &ShoeboxRoom) -> Self {
        let size = room.size();
        let lx = size.x.max(0.0);
        let ly = size.y.max(0.0);
        let lz = size.z.max(0.0);
        let volume = (lx * ly * lz).max(0.0);
        let surface_area = (2.0 * (lx * ly + ly * lz + lx * lz)).max(0.0);
        Self {
            diffuse_energy: [0.0; OCTAVE_BAND_COUNT],
            reflected_energy: [0.0; OCTAVE_BAND_COUNT],
            volume,
            surface_area,
        }
    }

    /// Accumulates the diffuse energy from one surface.
    ///
    /// `incident` is the per-band energy striking the surface, `scattering` its
    /// scattering spectrum, and `absorption` its absorption spectrum. For each
    /// band the reflected energy is `incident * (1 - alpha)` and the diffuse
    /// share of that is `reflected * s`. Both the diffuse and total reflected
    /// energies are accumulated (the latter feeds the diffusion coefficient).
    /// Negative or non-finite contributions are clamped to zero.
    pub fn add_surface(
        &mut self,
        incident: &[Sample; OCTAVE_BAND_COUNT],
        scattering: &ScatteringSpectrum,
        absorption: &MaterialAbsorption,
    ) {
        let s_bands = scattering.bands();
        let a_bands = absorption.bands();
        for band in 0..OCTAVE_BAND_COUNT {
            let e = incident[band];
            let e = if e.is_finite() { e.max(0.0) } else { 0.0 };
            let alpha = a_bands[band].clamp(0.0, 1.0);
            let reflected = e * (1.0 - alpha);
            let diffuse = reflected * diffuse_fraction(s_bands[band]);
            self.reflected_energy[band] += reflected;
            self.diffuse_energy[band] += diffuse;
        }
    }

    /// Convenience: builds a field for a room whose six faces share one surface,
    /// seeding unit incident energy in every band.
    ///
    /// # Examples
    ///
    /// ```
    /// use bevy_math::Vec3;
    /// use prism_audio_spatial::diffusion_field::DiffusionField;
    /// use prism_audio_spatial::early_reflections::ShoeboxRoom;
    /// use prism_audio_spatial::material_library::Material;
    /// use prism_audio_spatial::scattering::SurfaceScatter;
    ///
    /// let small = ShoeboxRoom::new(Vec3::ZERO, Vec3::new(4.0, 3.0, 5.0), [0.0; 6]);
    /// let big = ShoeboxRoom::new(Vec3::ZERO, Vec3::new(20.0, 12.0, 30.0), [0.0; 6]);
    /// let scat = SurfaceScatter::RoughBrick.scattering();
    /// let abs = Material::Brick.absorption();
    ///
    /// let f_small = DiffusionField::from_uniform_shoebox(&small, &scat, &abs);
    /// let f_big = DiffusionField::from_uniform_shoebox(&big, &scat, &abs);
    ///
    /// // A larger room mixes to a diffuse field later.
    /// assert!(f_big.mixing_time_ms() > f_small.mixing_time_ms());
    /// ```
    #[must_use]
    pub fn from_uniform_shoebox(
        room: &ShoeboxRoom,
        scattering: &ScatteringSpectrum,
        absorption: &MaterialAbsorption,
    ) -> Self {
        let mut field = Self::new(room);
        field.add_surface(&[1.0; OCTAVE_BAND_COUNT], scattering, absorption);
        field
    }

    /// The accumulated per-band diffuse energy density, aligned with
    /// [`OCTAVE_BAND_CENTERS`]. Always non-negative.
    ///
    /// [`OCTAVE_BAND_CENTERS`]: crate::material_library::OCTAVE_BAND_CENTERS
    #[must_use]
    pub fn band_energy(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        self.diffuse_energy
    }

    /// The unweighted mean of the eight diffuse band energies.
    #[must_use]
    pub fn broadband_energy(&self) -> Sample {
        let mut sum = 0.0;
        for e in &self.diffuse_energy {
            sum += *e;
        }
        sum / OCTAVE_BAND_COUNT as Sample
    }

    /// Room volume in cubic metres.
    #[must_use]
    pub fn volume(&self) -> Sample {
        self.volume
    }

    /// Room interior surface area in square metres.
    #[must_use]
    pub fn surface_area(&self) -> Sample {
        self.surface_area
    }

    /// The diffusion coefficient: the mean band ratio of diffuse to total
    /// reflected energy, in `[0, 1]`.
    ///
    /// A value near `0` means the field is dominated by specular reflections;
    /// near `1` almost all reflected energy is scattered. Bands with no
    /// reflected energy contribute `0`.
    #[must_use]
    pub fn diffusion_coefficient(&self) -> Sample {
        let mut sum = 0.0;
        for band in 0..OCTAVE_BAND_COUNT {
            let reflected = self.reflected_energy[band];
            if reflected > MIN_DIVISOR {
                sum += (self.diffuse_energy[band] / reflected).clamp(0.0, 1.0);
            }
        }
        sum / OCTAVE_BAND_COUNT as Sample
    }

    /// The mixing time in milliseconds, the early-to-late transition.
    ///
    /// Uses the classic Polack/Jot estimate `t_mix ~ sqrt(V)` with `V` in cubic
    /// metres and the result in milliseconds; larger rooms mix later. Returns
    /// `0` for a degenerate (zero-volume) room.
    #[must_use]
    pub fn mixing_time_ms(&self) -> Sample {
        if self.volume <= 0.0 {
            return 0.0;
        }
        ops::sqrt(self.volume)
    }

    /// The Schroeder echo density (reflections per second) at time `time_s`.
    ///
    /// Evaluates `4 * PI * c^3 * t^2 / V`: the density rises with the square of
    /// time and is lower in larger rooms. Non-finite or non-positive times, and
    /// zero-volume rooms, return `0`; the result is clamped to
    /// [`MAX_ECHO_DENSITY`].
    #[must_use]
    pub fn echo_density(&self, time_s: Sample, sound_speed: Sample) -> Sample {
        if !time_s.is_finite() || time_s <= 0.0 {
            return 0.0;
        }
        if self.volume <= MIN_DIVISOR {
            return 0.0;
        }
        let c = if sound_speed.is_finite() && sound_speed > 0.0 {
            sound_speed
        } else {
            DEFAULT_SOUND_SPEED
        };
        let density = 4.0 * PI * c * c * c * time_s * time_s / self.volume;
        density.clamp(0.0, MAX_ECHO_DENSITY)
    }

    /// Per-band late wet-send gains, normalised so the loudest band is `1`.
    ///
    /// Each gain is the band diffuse energy divided by the maximum band energy,
    /// giving a relative wet-send suggestion in `[0, 1]`. When no diffuse energy
    /// has been accumulated, all gains are `0`.
    #[must_use]
    pub fn late_send_gains(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        let mut max_energy = 0.0;
        for e in &self.diffuse_energy {
            if *e > max_energy {
                max_energy = *e;
            }
        }
        let mut gains = [0.0; OCTAVE_BAND_COUNT];
        if max_energy <= MIN_DIVISOR {
            return gains;
        }
        for (slot, e) in gains.iter_mut().zip(self.diffuse_energy.iter()) {
            *slot = (*e / max_energy).clamp(0.0, 1.0);
        }
        gains
    }

    /// Per-band FDN feedback gains for a delay line of `delay_seconds`,
    /// pre-multiplied by this field's normalised late wet-send gains.
    ///
    /// Combines the per-band decay of `reverb` (from its RT60 spectrum) with the
    /// relative diffuse energy of this field, giving a single per-band late gain
    /// suggestion. Every value stays in `[0, 1)`.
    #[must_use]
    pub fn fdn_late_gains(
        &self,
        reverb: &OctaveReverb,
        delay_seconds: Sample,
    ) -> [Sample; OCTAVE_BAND_COUNT] {
        let feedback = reverb.fdn_decay_gains(delay_seconds);
        let send = self.late_send_gains();
        let mut gains = [0.0; OCTAVE_BAND_COUNT];
        for band in 0..OCTAVE_BAND_COUNT {
            gains[band] = (feedback[band] * send[band]).clamp(0.0, 0.999_999);
        }
        gains
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material_library::Material;
    use crate::scattering::SurfaceScatter;
    use bevy_math::Vec3;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    fn room(x: Sample, y: Sample, z: Sample) -> ShoeboxRoom {
        ShoeboxRoom::new(Vec3::ZERO, Vec3::new(x, y, z), [0.0; 6])
    }

    #[test]
    fn larger_room_has_longer_mixing_time() {
        let small = DiffusionField::new(&room(4.0, 3.0, 5.0));
        let big = DiffusionField::new(&room(20.0, 12.0, 30.0));
        assert!(big.mixing_time_ms() > small.mixing_time_ms());
        // sqrt(V) exactly.
        assert!(approx(small.mixing_time_ms(), ops::sqrt(60.0), 1e-3));
    }

    #[test]
    fn echo_density_rises_with_time_and_clamps() {
        let field = DiffusionField::new(&room(10.0, 4.0, 8.0));
        let d1 = field.echo_density(0.02, DEFAULT_SOUND_SPEED);
        let d2 = field.echo_density(0.04, DEFAULT_SOUND_SPEED);
        assert!(d2 > d1, "density should rise with time");
        // t^2 law: doubling time quadruples density.
        assert!(approx(d2 / d1, 4.0, 1e-3), "ratio {}", d2 / d1);
        // Very late time is clamped, not infinite.
        let huge = field.echo_density(1.0e6, DEFAULT_SOUND_SPEED);
        assert!(huge.is_finite() && huge <= MAX_ECHO_DENSITY);
    }

    #[test]
    fn echo_density_lower_in_larger_room() {
        let small = DiffusionField::new(&room(4.0, 3.0, 5.0));
        let big = DiffusionField::new(&room(20.0, 12.0, 30.0));
        let ds = small.echo_density(0.05, DEFAULT_SOUND_SPEED);
        let db = big.echo_density(0.05, DEFAULT_SOUND_SPEED);
        assert!(db < ds, "larger room should have lower echo density");
    }

    #[test]
    fn echo_density_degenerate_inputs_are_safe() {
        let field = DiffusionField::new(&room(10.0, 4.0, 8.0));
        assert!(approx(field.echo_density(0.0, DEFAULT_SOUND_SPEED), 0.0, 1e-9));
        assert!(approx(field.echo_density(-1.0, DEFAULT_SOUND_SPEED), 0.0, 1e-9));
        assert!(approx(field.echo_density(Sample::NAN, DEFAULT_SOUND_SPEED), 0.0, 1e-9));
        // Non-positive sound speed falls back to the default (finite result).
        assert!(field.echo_density(0.05, -5.0).is_finite());
        // Zero-volume room -> zero density.
        let flat = DiffusionField::new(&room(0.0, 0.0, 0.0));
        assert!(approx(flat.echo_density(0.05, DEFAULT_SOUND_SPEED), 0.0, 1e-9));
    }

    #[test]
    fn high_absorption_lowers_diffuse_energy() {
        let r = room(10.0, 4.0, 8.0);
        let scat = SurfaceScatter::RoughBrick.scattering();
        let soft = DiffusionField::from_uniform_shoebox(&r, &scat, &Material::Carpet.absorption());
        let hard = DiffusionField::from_uniform_shoebox(&r, &scat, &Material::Concrete.absorption());
        // Concrete reflects far more energy than carpet, so more of it scatters.
        assert!(hard.broadband_energy() > soft.broadband_energy());
    }

    #[test]
    fn high_scattering_raises_diffuse_energy() {
        let r = room(10.0, 4.0, 8.0);
        let abs = Material::Concrete.absorption();
        let flat = DiffusionField::from_uniform_shoebox(
            &r,
            &SurfaceScatter::Flat.scattering(),
            &abs,
        );
        let diffuser = DiffusionField::from_uniform_shoebox(
            &r,
            &SurfaceScatter::Diffuser.scattering(),
            &abs,
        );
        assert!(diffuser.broadband_energy() > flat.broadband_energy());
    }

    #[test]
    fn band_energy_is_non_negative() {
        let r = room(10.0, 4.0, 8.0);
        for material in [Material::Concrete, Material::Carpet, Material::Glass] {
            let field = DiffusionField::from_uniform_shoebox(
                &r,
                &SurfaceScatter::RoughBrick.scattering(),
                &material.absorption(),
            );
            for e in field.band_energy() {
                assert!(e.is_finite() && e >= 0.0, "bad energy {e}");
            }
        }
    }

    #[test]
    fn add_surface_accumulates_and_clamps_bad_input() {
        let mut field = DiffusionField::new(&room(10.0, 4.0, 8.0));
        let scat = SurfaceScatter::Diffuser.scattering();
        let abs = Material::Concrete.absorption();
        field.add_surface(&[1.0; OCTAVE_BAND_COUNT], &scat, &abs);
        let after_one = field.broadband_energy();
        field.add_surface(&[1.0; OCTAVE_BAND_COUNT], &scat, &abs);
        let after_two = field.broadband_energy();
        assert!(after_two > after_one, "second surface should add energy");
        // Negative / non-finite incident energy contributes nothing (no panic).
        let before = field.broadband_energy();
        field.add_surface(&[-5.0; OCTAVE_BAND_COUNT], &scat, &abs);
        field.add_surface(&[Sample::NAN; OCTAVE_BAND_COUNT], &scat, &abs);
        assert!(approx(field.broadband_energy(), before, 1e-6));
    }

    #[test]
    fn diffusion_coefficient_in_unit_range_and_tracks_scattering() {
        let r = room(10.0, 4.0, 8.0);
        let abs = Material::Concrete.absorption();
        let flat = DiffusionField::from_uniform_shoebox(
            &r,
            &SurfaceScatter::Flat.scattering(),
            &abs,
        );
        let diffuser = DiffusionField::from_uniform_shoebox(
            &r,
            &SurfaceScatter::Diffuser.scattering(),
            &abs,
        );
        for c in [flat.diffusion_coefficient(), diffuser.diffusion_coefficient()] {
            assert!((0.0..=1.0).contains(&c), "coeff out of range {c}");
        }
        assert!(diffuser.diffusion_coefficient() > flat.diffusion_coefficient());
        // Empty field: no reflected energy -> zero coefficient, no divide by zero.
        let empty = DiffusionField::new(&r);
        assert!(approx(empty.diffusion_coefficient(), 0.0, 1e-9));
    }

    #[test]
    fn late_send_gains_are_normalised() {
        let r = room(10.0, 4.0, 8.0);
        let field = DiffusionField::from_uniform_shoebox(
            &r,
            &SurfaceScatter::Diffuser.scattering(),
            &Material::Concrete.absorption(),
        );
        let gains = field.late_send_gains();
        let mut max = 0.0;
        for g in gains {
            assert!((0.0..=1.0).contains(&g), "gain out of range {g}");
            if g > max {
                max = g;
            }
        }
        assert!(approx(max, 1.0, 1e-6), "peak band should normalise to 1");
        // Empty field -> all zero.
        let empty = DiffusionField::new(&r);
        assert!(empty.late_send_gains().iter().all(|&g| g == 0.0));
    }

    #[test]
    fn fdn_late_gains_combine_decay_and_send() {
        let r = room(10.0, 4.0, 8.0);
        let scat = SurfaceScatter::Diffuser.scattering();
        let abs = Material::Concrete.absorption();
        let field = DiffusionField::from_uniform_shoebox(&r, &scat, &abs);
        let reverb = OctaveReverb::uniform(&r, Material::Concrete);
        let gains = field.fdn_late_gains(&reverb, 0.05);
        for g in gains {
            assert!((0.0..1.0).contains(&g), "gain out of range {g}");
        }
        // A zero-length delay collapses the feedback factor and hence the gains.
        let zero_delay = field.fdn_late_gains(&reverb, 0.0);
        assert!(zero_delay.iter().all(|&g| g == 0.0));
    }
}
