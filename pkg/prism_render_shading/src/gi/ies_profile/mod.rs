//! IES photometric light profile sampling — CPU golden reference.
//!
//! This module turns a measured **IES / LM-63** luminaire photometry file (a
//! polar table of luminous intensity in candela) into a real-time-samplable
//! *direction → normalised intensity* function.  It is the backend-neutral
//! numerical reference the WESL/GPU twin pass must reproduce: given a world
//! direction and a luminaire orientation, it returns the dimensionless beam
//! shape multiplier in `[0, 1]` used to modulate a punctual light.
//!
//! The work is split into three focused submodules:
//! * [`grid`] — the raw candela table on a (non-uniform) polar grid plus
//!   defensive bilinear [`PhotometricGrid::sample`] with `theta` clamping and
//!   periodic `phi` wrapping.
//! * [`normalize`] — peak / mean-flux normalisation references, the spherical
//!   `sin(theta)`-weighted flux integral, and the world→luminaire polar mapping
//!   ([`world_to_local_angles`]).
//! * [`symmetry`] — LM-63 azimuthal symmetry classes and the pure `phi`-folding
//!   that unfolds a stored wedge back to the full circle ([`fold_phi`]).
//!
//! The high-level entry point [`intensity_for_direction`] composes them: it maps
//! the world direction into the luminaire frame, folds the azimuth into the
//! profile's stored wedge, bilinearly samples the candela table, and divides by
//! the pre-computed normalisation reference, clamping the result to `[0, 1]`.
//!
//! # Conventions
//! * Angles inside the profile are in **degrees** to match LM-63; only the
//!   coordinate mapping and flux integral touch radians, via
//!   [`bevy_math::ops`].
//! * The luminaire frame is right-handed with `forward` as the polar axis
//!   (`theta = 0`) and `up` seeding the azimuth origin (see
//!   [`normalize::luminaire_basis`]).
//! * Deterministic pure functions, no RNG / I/O / GPU / unsafe.  Degenerate
//!   input (empty table, zero vectors, non-finite angles) yields `0` rather than
//!   `NaN`/`inf`.  Only profile construction allocates.
//!
//! # References
//! * IESNA LM-63, *Standard File Format for Electronic Transfer of Photometric
//!   Data*.
//! * IESNA LM-79; Ashdown 1993, *Near-Field Photometry*.
//! * Pharr, Jakob & Humphreys, *Physically Based Rendering* (goniometric lights).

pub mod grid;
pub mod normalize;
pub mod symmetry;

use bevy_math::Vec3;

pub use grid::PhotometricGrid;
pub use normalize::{
    luminaire_basis, luminous_flux, mean_intensity, measured_solid_angle, peak_normalize,
    reference_intensity, world_to_local_angles, NormalizeMode,
};
pub use symmetry::{fold_phi, Symmetry};

/// A ready-to-sample IES luminaire profile.
///
/// Bundles the candela [`grid`], its azimuthal [`Symmetry`] class, and the
/// pre-computed normalisation reference so repeated direction queries avoid
/// re-scanning the table.  Build via [`IesProfile::new`].
#[derive(Clone, Debug, PartialEq)]
pub struct IesProfile {
    /// The measured candela table.
    grid: PhotometricGrid,
    /// Azimuthal symmetry class controlling `phi` folding.
    symmetry: Symmetry,
    /// Normalisation mode used to derive [`reference`](Self::reference).
    mode: NormalizeMode,
    /// Reference candela that raw samples are divided by (`0` ⇒ no light).
    reference: f32,
}

impl IesProfile {
    /// Builds a profile from a grid, symmetry class, and normalisation mode,
    /// pre-computing the normalisation reference.
    #[inline]
    pub fn new(grid: PhotometricGrid, symmetry: Symmetry, mode: NormalizeMode) -> Self {
        let reference = reference_intensity(&grid, mode);
        Self {
            grid,
            symmetry,
            mode,
            reference,
        }
    }

    /// Builds a profile whose symmetry class is inferred from the grid's last
    /// horizontal angle (the LM-63 symmetry marker).
    #[inline]
    pub fn from_grid(grid: PhotometricGrid, mode: NormalizeMode) -> Self {
        let symmetry = match grid.horizontal_angles().last() {
            Some(&last) if grid.horizontal_count() == 1 => {
                // A single horizontal angle is fully rotational regardless of
                // the recorded value.
                let _ = last;
                Symmetry::Rotational
            }
            Some(&last) => Symmetry::from_last_horizontal_angle(last),
            None => Symmetry::Rotational,
        };
        Self::new(grid, symmetry, mode)
    }

    /// The underlying candela table.
    #[inline]
    pub fn grid(&self) -> &PhotometricGrid {
        &self.grid
    }

    /// The azimuthal symmetry class.
    #[inline]
    pub fn symmetry(&self) -> Symmetry {
        self.symmetry
    }

    /// The normalisation mode.
    #[inline]
    pub fn mode(&self) -> NormalizeMode {
        self.mode
    }

    /// The pre-computed normalisation reference candela.
    #[inline]
    pub fn reference(&self) -> f32 {
        self.reference
    }

    /// Samples the profile at a luminaire-local `(theta, phi)` in **degrees**,
    /// folding the azimuth and dividing by the normalisation reference.
    ///
    /// Returns the normalised intensity in `[0, 1]` (`0` when the reference is
    /// degenerate).
    #[inline]
    pub fn sample_local(&self, theta_deg: f32, phi_deg: f32) -> f32 {
        if !(self.reference > 0.0) {
            return 0.0;
        }
        let folded_phi = fold_phi(phi_deg, self.symmetry);
        let raw = self.grid.sample(theta_deg, folded_phi);
        let normalised = raw / self.reference;
        if normalised.is_finite() {
            normalised.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Tunables for a direction query.
///
/// `clamp_unit` keeps the output within `[0, 1]`; disabling it lets
/// [`NormalizeMode::MeanFlux`] profiles report their true relative brightness
/// (`> 1` in the brightest directions).  `intensity_scale` is a final linear
/// gain (e.g. a dimmer).  Defaults: clamp on, unit gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IesSampleParams {
    /// Clamp the normalised intensity to `[0, 1]` before scaling.
    pub clamp_unit: bool,
    /// Final multiplicative gain applied after normalisation.
    pub intensity_scale: f32,
}

impl Default for IesSampleParams {
    #[inline]
    fn default() -> Self {
        Self {
            clamp_unit: true,
            intensity_scale: 1.0,
        }
    }
}

impl IesSampleParams {
    /// Builds parameters with a clamped, finite `intensity_scale` (`>= 0`).
    #[inline]
    pub fn new(clamp_unit: bool, intensity_scale: f32) -> Self {
        let scale = if intensity_scale.is_finite() && intensity_scale >= 0.0 {
            intensity_scale
        } else {
            1.0
        };
        Self {
            clamp_unit,
            intensity_scale: scale,
        }
    }
}

/// Normalised IES intensity for a world-space direction, using default
/// [`IesSampleParams`].
///
/// `world_dir` is the direction from the luminaire toward the shaded point (it
/// need not be normalised).  `luminaire_forward` is the aim axis and
/// `luminaire_up` seeds the azimuth origin.  Returns a value in `[0, 1]`.
#[inline]
pub fn intensity_for_direction(
    profile: &IesProfile,
    world_dir: Vec3,
    luminaire_forward: Vec3,
    luminaire_up: Vec3,
) -> f32 {
    intensity_for_direction_params(
        profile,
        world_dir,
        luminaire_forward,
        luminaire_up,
        IesSampleParams::default(),
    )
}

/// Normalised IES intensity for a world-space direction with explicit
/// [`IesSampleParams`].
///
/// Maps the direction into the luminaire frame, samples the normalised profile,
/// then applies the optional unit clamp and final gain.  Always finite.
#[inline]
pub fn intensity_for_direction_params(
    profile: &IesProfile,
    world_dir: Vec3,
    luminaire_forward: Vec3,
    luminaire_up: Vec3,
    params: IesSampleParams,
) -> f32 {
    let (theta_deg, phi_deg) =
        world_to_local_angles(world_dir, luminaire_forward, luminaire_up);

    // sample_local already clamps to [0, 1]; when the caller opts out of the
    // unit clamp, recompute the raw normalised value instead.
    let base = if params.clamp_unit {
        profile.sample_local(theta_deg, phi_deg)
    } else {
        let reference = profile.reference();
        if reference > 0.0 {
            let folded_phi = fold_phi(phi_deg, profile.symmetry());
            let raw = profile.grid().sample(theta_deg, folded_phi) / reference;
            if raw.is_finite() { raw.max(0.0) } else { 0.0 }
        } else {
            0.0
        }
    };

    let scale = if params.intensity_scale.is_finite() && params.intensity_scale >= 0.0 {
        params.intensity_scale
    } else {
        1.0
    };
    let out = base * scale;
    if out.is_finite() { out.max(0.0) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// A narrow spotlight: candela peaks at nadir (theta=0) and falls to 0 by
    /// 90°, rotationally symmetric.
    fn spotlight() -> IesProfile {
        let grid = PhotometricGrid::new(
            alloc::vec![0.0, 45.0, 90.0, 180.0],
            alloc::vec![0.0],
            alloc::vec![1000.0, 500.0, 0.0, 0.0],
        )
        .unwrap();
        IesProfile::from_grid(grid, NormalizeMode::Peak)
    }

    #[test]
    fn from_grid_infers_rotational_for_single_azimuth() {
        let p = spotlight();
        assert_eq!(p.symmetry(), Symmetry::Rotational);
        assert!(approx(p.reference(), 1000.0, 1e-3));
    }

    #[test]
    fn peak_direction_is_unity() {
        let p = spotlight();
        // Forward = -Z (aim). Direction along aim -> theta 0 -> peak.
        let v = intensity_for_direction(&p, Vec3::NEG_Z, Vec3::NEG_Z, Vec3::Y);
        assert!(approx(v, 1.0, 1e-3));
    }

    #[test]
    fn back_hemisphere_is_dark() {
        let p = spotlight();
        let v = intensity_for_direction(&p, Vec3::Z, Vec3::NEG_Z, Vec3::Y);
        assert!(approx(v, 0.0, 1e-4));
    }

    #[test]
    fn mid_angle_matches_table() {
        let p = spotlight();
        // Direction at 45 deg from aim in the X plane.
        let dir = (Vec3::NEG_Z + Vec3::X).normalize();
        let v = intensity_for_direction(&p, dir, Vec3::NEG_Z, Vec3::Y);
        // 500/1000 = 0.5 at theta=45.
        assert!(approx(v, 0.5, 1e-2));
    }

    #[test]
    fn rotational_profile_ignores_azimuth() {
        let p = spotlight();
        let dir_x = (Vec3::NEG_Z + Vec3::X).normalize();
        let dir_y = (Vec3::NEG_Z + Vec3::Y).normalize();
        let vx = intensity_for_direction(&p, dir_x, Vec3::NEG_Z, Vec3::Y);
        let vy = intensity_for_direction(&p, dir_y, Vec3::NEG_Z, Vec3::Y);
        assert!(approx(vx, vy, 1e-4));
    }

    #[test]
    fn params_scale_and_unclamped() {
        // Bilateral profile normalised by mean flux so some directions > 1.
        let grid = PhotometricGrid::new(
            alloc::vec![0.0, 90.0, 180.0],
            alloc::vec![0.0, 180.0],
            alloc::vec![1000.0, 0.0, 0.0, 1000.0, 0.0, 0.0],
        )
        .unwrap();
        let p = IesProfile::new(grid, Symmetry::Bilateral, NormalizeMode::MeanFlux);
        // Clamped (default) stays <= 1.
        let clamped = intensity_for_direction(&p, Vec3::NEG_Z, Vec3::NEG_Z, Vec3::Y);
        assert!(clamped <= 1.0 + 1e-4);
        // Unclamped may exceed 1 at the peak since mean < peak.
        let unclamped = intensity_for_direction_params(
            &p,
            Vec3::NEG_Z,
            Vec3::NEG_Z,
            Vec3::Y,
            IesSampleParams::new(false, 1.0),
        );
        assert!(unclamped >= clamped - 1e-4);
        // A gain of 0 kills the output.
        let zero = intensity_for_direction_params(
            &p,
            Vec3::NEG_Z,
            Vec3::NEG_Z,
            Vec3::Y,
            IesSampleParams::new(true, 0.0),
        );
        assert!(approx(zero, 0.0, 1e-6));
    }

    #[test]
    fn default_params_are_clamped_unit_gain() {
        let d = IesSampleParams::default();
        assert!(d.clamp_unit);
        assert!(approx(d.intensity_scale, 1.0, 1e-6));
    }

    #[test]
    fn params_new_sanitises_scale() {
        let p = IesSampleParams::new(false, f32::NAN);
        assert!(approx(p.intensity_scale, 1.0, 1e-6));
        let n = IesSampleParams::new(true, -3.0);
        assert!(approx(n.intensity_scale, 1.0, 1e-6));
    }

    #[test]
    fn degenerate_profile_returns_zero() {
        let p = IesProfile::new(
            PhotometricGrid::uniform_single(0.0),
            Symmetry::Rotational,
            NormalizeMode::Peak,
        );
        assert_eq!(p.reference(), 0.0);
        let v = intensity_for_direction(&p, Vec3::NEG_Z, Vec3::NEG_Z, Vec3::Y);
        assert!(approx(v, 0.0, 1e-6));
    }

    #[test]
    fn bilateral_unfolds_symmetrically() {
        // phi=0 bright, phi=180 dark at theta=90; unfolding mirrors back half.
        let grid = PhotometricGrid::new(
            alloc::vec![0.0, 90.0, 180.0],
            alloc::vec![0.0, 180.0],
            // h0 (phi 0):   [500, 1000, 500]
            // h1 (phi 180): [500,    0, 500]
            alloc::vec![500.0, 1000.0, 500.0, 500.0, 0.0, 500.0],
        )
        .unwrap();
        let p = IesProfile::new(grid, Symmetry::Bilateral, NormalizeMode::Peak);
        // theta=90, phi=90 (front side interpolated) vs phi=270 (mirror of 90).
        let front = p.sample_local(90.0, 90.0);
        let mirror = p.sample_local(90.0, 270.0);
        assert!(approx(front, mirror, 1e-4));
    }

    #[test]
    fn all_outputs_finite_and_nonnegative() {
        let p = spotlight();
        let dirs = [
            Vec3::NEG_Z,
            Vec3::Z,
            Vec3::X,
            Vec3::Y,
            Vec3::ZERO,
            Vec3::new(0.3, -0.5, 0.8),
        ];
        for d in dirs {
            let v = intensity_for_direction(&p, d, Vec3::NEG_Z, Vec3::Y);
            assert!(v.is_finite() && v >= 0.0 && v <= 1.0 + 1e-4);
        }
    }
}
