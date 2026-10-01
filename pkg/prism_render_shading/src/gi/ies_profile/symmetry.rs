//! LM-63 azimuthal symmetry: folding arbitrary `phi` into the stored range.
//!
//! IES / LM-63 files exploit a luminaire's azimuthal symmetry to store only a
//! representative wedge of horizontal angles and let the reader *unfold* it to
//! the full circle.  The last horizontal angle in the file encodes the symmetry
//! class:
//!
//! | Last horizontal angle | Symmetry        | Stored `phi` range | Unfolding          |
//! |-----------------------|-----------------|--------------------|--------------------|
//! | `0`                   | fully rotational| single value       | ignore `phi`       |
//! | `90`                  | quadrant        | `0..90`            | mirror into a quad |
//! | `180`                 | bilateral       | `0..180`           | mirror about 0/180 |
//! | `360`                 | none            | `0..360`           | wrap only          |
//!
//! This module provides the pure *fold* functions that map any query azimuth
//! (in degrees, any sign / magnitude) back into the stored wedge so the
//! underlying [`PhotometricGrid`](super::grid::PhotometricGrid) can be sampled
//! directly.  The reflections are chosen so the unfolded field is continuous
//! across every mirror plane and seam.
//!
//! # Conventions
//! * All angles are in **degrees**.  Inputs are first reduced modulo 360° into
//!   `[0, 360)`; the fold then mirrors into the class's stored range.
//! * Folds are *pure* and allocation-free: they depend only on the symmetry
//!   class, not on any particular grid instance.
//! * Outputs are finite and lie within the stored range of the class; a
//!   non-finite input folds to `0`.
//!
//! # References
//! * IESNA LM-63, horizontal-angle symmetry conventions (`0 / 90 / 180 / 360`).

/// Full azimuthal period in degrees.
const PERIOD: f32 = 360.0;

/// Azimuthal symmetry class of an IES luminaire, per LM-63.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Symmetry {
    /// Fully rotationally symmetric: a single horizontal angle; `phi` ignored.
    Rotational,
    /// Quadrant symmetric: data stored over `0..90°`, mirrored into each quad.
    Quadrant,
    /// Bilaterally symmetric: data stored over `0..180°`, mirrored about the
    /// 0°/180° plane.
    Bilateral,
    /// No azimuthal symmetry: data stored over the full `0..360°`.
    None,
}

impl Default for Symmetry {
    #[inline]
    fn default() -> Self {
        Symmetry::None
    }
}

impl Symmetry {
    /// Classifies a luminaire from the **last** horizontal angle of its LM-63
    /// table (the file's symmetry marker).
    ///
    /// Recognised markers are `0`, `90`, `180`, `360`; anything else (including
    /// non-finite input) falls back to [`Symmetry::None`] so no data is lost by
    /// an unexpected marker.
    #[inline]
    pub fn from_last_horizontal_angle(last_phi_deg: f32) -> Self {
        if !last_phi_deg.is_finite() {
            return Symmetry::None;
        }
        // Compare against the canonical markers with a small tolerance.
        let close = |target: f32| (last_phi_deg - target).abs() <= 0.5;
        if close(0.0) {
            Symmetry::Rotational
        } else if close(90.0) {
            Symmetry::Quadrant
        } else if close(180.0) {
            Symmetry::Bilateral
        } else {
            Symmetry::None
        }
    }

    /// Inclusive stored azimuth range `(lo, hi)` in degrees for this class.
    ///
    /// Rotational tables conceptually store a single angle at `0°`, so the range
    /// is degenerate `(0, 0)`.
    #[inline]
    pub fn stored_phi_range(self) -> (f32, f32) {
        match self {
            Symmetry::Rotational => (0.0, 0.0),
            Symmetry::Quadrant => (0.0, 90.0),
            Symmetry::Bilateral => (0.0, 180.0),
            Symmetry::None => (0.0, 360.0),
        }
    }
}

/// Reduces an arbitrary azimuth to `[0, 360)` degrees, mapping non-finite input
/// to `0`.
#[inline]
fn wrap_360(phi_deg: f32) -> f32 {
    if !phi_deg.is_finite() {
        return 0.0;
    }
    phi_deg.rem_euclid(PERIOD)
}

/// Folds an arbitrary query azimuth into the stored wedge for `symmetry`.
///
/// The returned angle (degrees) lies within [`Symmetry::stored_phi_range`] and
/// can be fed straight to the grid's horizontal lookup.  The mapping is
/// continuous across each mirror plane, so bilinear sampling stays smooth.
#[inline]
pub fn fold_phi(phi_deg: f32, symmetry: Symmetry) -> f32 {
    let p = wrap_360(phi_deg); // [0, 360)
    match symmetry {
        Symmetry::Rotational => 0.0,
        Symmetry::None => p,
        Symmetry::Bilateral => fold_bilateral(p),
        Symmetry::Quadrant => fold_quadrant(p),
    }
}

/// Folds `phi ∈ [0, 360)` into `[0, 180]` by mirroring the back half about the
/// 0°/180° plane (`phi -> 360 - phi` for `phi > 180`).
#[inline]
fn fold_bilateral(p: f32) -> f32 {
    if p > 180.0 {
        PERIOD - p
    } else {
        p
    }
}

/// Folds `phi ∈ [0, 360)` into `[0, 90]` by first mirroring into `[0, 180]`
/// (bilateral) and then mirroring about the 90° plane (`phi -> 180 - phi` for
/// `phi > 90`).
#[inline]
fn fold_quadrant(p: f32) -> f32 {
    let half = fold_bilateral(p); // [0, 180]
    if half > 90.0 {
        180.0 - half
    } else {
        half
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4
    }

    #[test]
    fn classification_from_marker() {
        assert_eq!(Symmetry::from_last_horizontal_angle(0.0), Symmetry::Rotational);
        assert_eq!(Symmetry::from_last_horizontal_angle(90.0), Symmetry::Quadrant);
        assert_eq!(Symmetry::from_last_horizontal_angle(180.0), Symmetry::Bilateral);
        assert_eq!(Symmetry::from_last_horizontal_angle(360.0), Symmetry::None);
        // Unknown marker -> None (lossless default).
        assert_eq!(Symmetry::from_last_horizontal_angle(270.0), Symmetry::None);
        assert_eq!(Symmetry::from_last_horizontal_angle(f32::NAN), Symmetry::None);
    }

    #[test]
    fn stored_ranges() {
        assert_eq!(Symmetry::Rotational.stored_phi_range(), (0.0, 0.0));
        assert_eq!(Symmetry::Quadrant.stored_phi_range(), (0.0, 90.0));
        assert_eq!(Symmetry::Bilateral.stored_phi_range(), (0.0, 180.0));
        assert_eq!(Symmetry::None.stored_phi_range(), (0.0, 360.0));
    }

    #[test]
    fn rotational_ignores_phi() {
        for &p in &[0.0_f32, 45.0, 123.0, 270.0, 359.0, -30.0, 720.0] {
            assert!(approx(fold_phi(p, Symmetry::Rotational), 0.0));
        }
    }

    #[test]
    fn none_only_wraps() {
        assert!(approx(fold_phi(10.0, Symmetry::None), 10.0));
        assert!(approx(fold_phi(370.0, Symmetry::None), 10.0));
        assert!(approx(fold_phi(-10.0, Symmetry::None), 350.0));
        // Stays strictly below 360.
        let v = fold_phi(360.0, Symmetry::None);
        assert!(v >= 0.0 && v < 360.0);
    }

    #[test]
    fn bilateral_mirrors_back_half() {
        assert!(approx(fold_phi(30.0, Symmetry::Bilateral), 30.0));
        assert!(approx(fold_phi(150.0, Symmetry::Bilateral), 150.0));
        // 210 mirrors to 150, 330 mirrors to 30.
        assert!(approx(fold_phi(210.0, Symmetry::Bilateral), 150.0));
        assert!(approx(fold_phi(330.0, Symmetry::Bilateral), 30.0));
        // Endpoints map onto themselves / plane.
        assert!(approx(fold_phi(180.0, Symmetry::Bilateral), 180.0));
        assert!(approx(fold_phi(0.0, Symmetry::Bilateral), 0.0));
    }

    #[test]
    fn bilateral_output_in_range() {
        let mut p = -720.0_f32;
        while p <= 720.0 {
            let f = fold_phi(p, Symmetry::Bilateral);
            assert!(f >= 0.0 && f <= 180.0 + 1e-4, "phi={p} -> {f}");
            p += 7.0;
        }
    }

    #[test]
    fn quadrant_mirrors_into_first_quad() {
        assert!(approx(fold_phi(10.0, Symmetry::Quadrant), 10.0));
        // 100 -> bilateral 100 -> quad 80.
        assert!(approx(fold_phi(100.0, Symmetry::Quadrant), 80.0));
        // 170 -> bilateral 170 -> quad 10.
        assert!(approx(fold_phi(170.0, Symmetry::Quadrant), 10.0));
        // 190 -> bilateral 170 -> quad 10.
        assert!(approx(fold_phi(190.0, Symmetry::Quadrant), 10.0));
        // 280 -> wrap 280 -> bilateral 80 -> quad 80.
        assert!(approx(fold_phi(280.0, Symmetry::Quadrant), 80.0));
        // Plane 90 maps to itself.
        assert!(approx(fold_phi(90.0, Symmetry::Quadrant), 90.0));
    }

    #[test]
    fn quadrant_output_in_range() {
        let mut p = -720.0_f32;
        while p <= 720.0 {
            let f = fold_phi(p, Symmetry::Quadrant);
            assert!(f >= 0.0 && f <= 90.0 + 1e-4, "phi={p} -> {f}");
            p += 3.0;
        }
    }

    #[test]
    fn fold_is_continuous_across_mirror() {
        // Approaching 180 from both sides for bilateral should agree.
        let a = fold_phi(179.9, Symmetry::Bilateral);
        let b = fold_phi(180.1, Symmetry::Bilateral);
        assert!((a - b).abs() < 0.3);
        // Approaching 90 from both sides for quadrant should agree.
        let c = fold_phi(89.9, Symmetry::Quadrant);
        let d = fold_phi(90.1, Symmetry::Quadrant);
        assert!((c - d).abs() < 0.3);
    }

    #[test]
    fn non_finite_folds_to_zero() {
        assert_eq!(fold_phi(f32::NAN, Symmetry::None), 0.0);
        assert_eq!(fold_phi(f32::INFINITY, Symmetry::Bilateral), 0.0);
    }
}
