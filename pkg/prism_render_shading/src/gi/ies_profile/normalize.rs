//! Normalisation and world→luminaire coordinate mapping for IES profiles.
//!
//! Raw IES candela values are absolute photometric intensities (often tens of
//! thousands of candela).  Real-time shading wants a dimensionless *shape*
//! multiplier that peaks near `1`, so this module turns a
//! [`PhotometricGrid`](super::grid::PhotometricGrid) into a normalisation
//! reference and provides the geometry that maps a world-space direction into
//! the luminaire's polar frame so the grid can be sampled.
//!
//! Two normalisation references are offered:
//! * [`NormalizeMode::Peak`] divides by the table's maximum candela, so the
//!   brightest measured direction maps to exactly `1` and every other direction
//!   to `[0, 1]`.  This is the usual choice for a punctual spotlight cookie.
//! * [`NormalizeMode::MeanFlux`] divides by the *mean* intensity over the
//!   measured solid angle (`total luminous flux / measured solid angle`), which
//!   keeps the average energy fixed when swapping profiles; individual
//!   directions may exceed `1` and are clamped by the caller.
//!
//! The flux integral uses the standard spherical measure with a `sin(theta)`
//! Jacobian, evaluated by the trapezoidal rule over the grid's (possibly
//! non-uniform) angle axes:
//!
//! ```text
//! flux = integral_phi integral_theta  I(theta, phi) * sin(theta) d(theta) d(phi)
//! ```
//!
//! # Conventions
//! * The luminaire frame is right-handed with the aim (`forward`) direction as
//!   the polar axis: `theta = 0` points along `forward`, `theta` grows toward
//!   the back hemisphere, and `phi` is the azimuth measured from the projection
//!   of `up`, increasing right-handed about `forward`.
//! * Both returned angles are in **degrees** to match the grid axes; `theta` is
//!   in `[0, 180]`, `phi` in `[0, 360)`.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt`/`abs` via inherent
//!   methods.  Direction inputs are renormalised defensively and degenerate
//!   bases fall back to a stable world frame; results are finite, never `NaN`.
//!
//! # References
//! * IESNA LM-63; IESNA LM-79 (flux/efficacy conventions).
//! * Pharr, Jakob & Humphreys, *Physically Based Rendering*, spherical
//!   integration with the `sin(theta)` measure.

use bevy_math::{ops, Vec3};
use core::f32::consts::{PI, TAU};

use super::grid::PhotometricGrid;

/// Degrees-per-radian conversion factor.
const RAD_TO_DEG: f32 = 180.0 / PI;

/// Smallest squared length treated as a usable direction.
const MIN_LEN_SQ: f32 = 1.0e-12;

/// Smallest reference intensity / solid angle treated as non-degenerate.
const MIN_REFERENCE: f32 = 1.0e-9;

/// Which reference the raw candela values are divided by to reach `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalizeMode {
    /// Divide by the peak candela: brightest direction → `1`.
    Peak,
    /// Divide by the mean intensity over the measured solid angle.
    MeanFlux,
}

impl Default for NormalizeMode {
    #[inline]
    fn default() -> Self {
        NormalizeMode::Peak
    }
}

/// Normalises `v`, falling back to `fallback` for degenerate (zero-length or
/// non-finite) input so downstream dot products stay well defined.
#[inline]
fn safe_normalize(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.is_finite() && v.length_squared() > MIN_LEN_SQ {
        v.normalize()
    } else {
        fallback
    }
}

/// Peak-normalises a raw candela value against `max_candela`, clamped to
/// `[0, 1]`.  Returns `0` when the reference is non-positive or non-finite.
#[inline]
pub fn peak_normalize(raw: f32, max_candela: f32) -> f32 {
    if !max_candela.is_finite() || max_candela <= MIN_REFERENCE || !raw.is_finite() {
        return 0.0;
    }
    (raw / max_candela).clamp(0.0, 1.0)
}

/// Trapezoidal integral of `I(theta) * sin(theta)` across one horizontal row.
///
/// `vertical` holds ascending polar angles in degrees; `row` holds the matching
/// candela values.  Returns `0` for fewer than two samples (no interval to
/// integrate) or mismatched lengths.
#[inline]
fn row_theta_integral(vertical: &[f32], row: &[f32]) -> f32 {
    let n = vertical.len();
    if n < 2 || row.len() != n {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for i in 0..n - 1 {
        let t0 = vertical[i] / RAD_TO_DEG;
        let t1 = vertical[i + 1] / RAD_TO_DEG;
        let dt = t1 - t0;
        if !dt.is_finite() || dt <= 0.0 {
            continue;
        }
        let f0 = sanitize(row[i]) * ops::sin(t0);
        let f1 = sanitize(row[i + 1]) * ops::sin(t1);
        let term = 0.5 * (f0 + f1) * dt;
        if term.is_finite() {
            sum += term;
        }
    }
    sum
}

/// Clamps a candela value to a finite, non-negative number.
#[inline]
fn sanitize(c: f32) -> f32 {
    if c.is_finite() && c >= 0.0 { c } else { 0.0 }
}

/// Builds a borrowed view of one horizontal row of the candela buffer.
#[inline]
fn row_slice(grid: &PhotometricGrid, h: usize) -> &[f32] {
    let nv = grid.vertical_count();
    let start = h * nv;
    &grid.candela()[start..start + nv]
}

/// Total luminous flux of the grid: the spherical integral of intensity with a
/// `sin(theta)` measure, in lumens (candela·steradian).
///
/// Integrates over the *stored* angle ranges.  A single horizontal sample is
/// treated as fully axially symmetric and multiplied by the full `2*PI`
/// azimuth; otherwise the azimuth is integrated by the trapezoidal rule.
#[inline]
pub fn luminous_flux(grid: &PhotometricGrid) -> f32 {
    if grid.is_empty() {
        return 0.0;
    }
    let vertical = grid.vertical_angles();
    let horizontal = grid.horizontal_angles();
    let nh = horizontal.len();

    if nh == 1 {
        let flux = row_theta_integral(vertical, row_slice(grid, 0)) * TAU;
        return if flux.is_finite() && flux >= 0.0 { flux } else { 0.0 };
    }

    let mut flux = 0.0_f32;
    let mut prev = row_theta_integral(vertical, row_slice(grid, 0));
    for h in 0..nh - 1 {
        let next = row_theta_integral(vertical, row_slice(grid, h + 1));
        let p0 = horizontal[h] / RAD_TO_DEG;
        let p1 = horizontal[h + 1] / RAD_TO_DEG;
        let dp = p1 - p0;
        if dp.is_finite() && dp > 0.0 {
            let term = 0.5 * (prev + next) * dp;
            if term.is_finite() {
                flux += term;
            }
        }
        prev = next;
    }
    if flux.is_finite() && flux >= 0.0 { flux } else { 0.0 }
}

/// Measured solid angle of the grid: the same integral as [`luminous_flux`] but
/// with unit intensity, i.e. `integral sin(theta) d(theta) d(phi)` over the
/// stored ranges, in steradians.
#[inline]
pub fn measured_solid_angle(grid: &PhotometricGrid) -> f32 {
    if grid.is_empty() {
        return 0.0;
    }
    let vertical = grid.vertical_angles();
    let horizontal = grid.horizontal_angles();
    let nv = vertical.len();
    let nh = horizontal.len();

    // The theta integral of sin(theta) is identical for every row, so compute it
    // once against a unit row.
    let unit_row = {
        // Reuse row_theta_integral with a borrowed slice of ones is awkward in
        // no_std without allocation, so inline the trapezoid here.
        if nv < 2 {
            0.0
        } else {
            let mut s = 0.0_f32;
            for i in 0..nv - 1 {
                let t0 = vertical[i] / RAD_TO_DEG;
                let t1 = vertical[i + 1] / RAD_TO_DEG;
                let dt = t1 - t0;
                if dt.is_finite() && dt > 0.0 {
                    s += 0.5 * (ops::sin(t0) + ops::sin(t1)) * dt;
                }
            }
            s
        }
    };

    if nh == 1 {
        let sa = unit_row * TAU;
        return if sa.is_finite() && sa >= 0.0 { sa } else { 0.0 };
    }

    let mut sa = 0.0_f32;
    for h in 0..nh - 1 {
        let p0 = horizontal[h] / RAD_TO_DEG;
        let p1 = horizontal[h + 1] / RAD_TO_DEG;
        let dp = p1 - p0;
        if dp.is_finite() && dp > 0.0 {
            sa += unit_row * dp;
        }
    }
    if sa.is_finite() && sa >= 0.0 { sa } else { 0.0 }
}

/// Mean intensity over the measured solid angle (`flux / solid_angle`).
///
/// Falls back to the peak candela when the solid angle is degenerate, so the
/// result is always a usable positive reference for a non-empty grid.
#[inline]
pub fn mean_intensity(grid: &PhotometricGrid) -> f32 {
    let sa = measured_solid_angle(grid);
    if sa > MIN_REFERENCE {
        let m = luminous_flux(grid) / sa;
        if m.is_finite() && m > 0.0 {
            return m;
        }
    }
    grid.max_candela()
}

/// Reference intensity that raw candela are divided by for a given mode.
///
/// Returns `0` only for an empty / all-zero table; callers treat a `0`
/// reference as "no normalisation possible" and emit `0` intensity.
#[inline]
pub fn reference_intensity(grid: &PhotometricGrid, mode: NormalizeMode) -> f32 {
    let reference = match mode {
        NormalizeMode::Peak => grid.max_candela(),
        NormalizeMode::MeanFlux => mean_intensity(grid),
    };
    if reference.is_finite() && reference > 0.0 {
        reference
    } else {
        0.0
    }
}

/// Right-handed orthonormal luminaire basis `(right, up, forward)` built from a
/// possibly non-orthogonal / degenerate `forward`/`up` pair.
///
/// `forward` is the polar axis (`theta = 0`).  `up` seeds the azimuth origin and
/// is re-orthogonalised; if it is parallel to `forward` a stable world axis is
/// substituted.
#[inline]
pub fn luminaire_basis(forward: Vec3, up: Vec3) -> (Vec3, Vec3, Vec3) {
    let f = safe_normalize(forward, Vec3::NEG_Z);
    let up_hint = safe_normalize(up, Vec3::Y);

    // right = up x forward; re-seed if up is (anti-)parallel to forward.
    let mut right = up_hint.cross(f);
    if right.length_squared() <= MIN_LEN_SQ {
        right = f.cross(Vec3::Y);
        if right.length_squared() <= MIN_LEN_SQ {
            right = f.cross(Vec3::X);
        }
    }
    let right = safe_normalize(right, Vec3::X);
    // true_up = forward x right completes a right-handed frame (right x up = f).
    let true_up = safe_normalize(f.cross(right), Vec3::Y);
    (right, true_up, f)
}

/// Maps a world-space direction into the luminaire's polar `(theta, phi)` frame,
/// both in **degrees** (`theta ∈ [0, 180]`, `phi ∈ [0, 360)`).
///
/// `world_dir` need not be normalised; a degenerate direction maps to the aim
/// axis (`theta = 0`).  The basis is built by [`luminaire_basis`].
#[inline]
pub fn world_to_local_angles(world_dir: Vec3, forward: Vec3, up: Vec3) -> (f32, f32) {
    let (right, true_up, f) = luminaire_basis(forward, up);
    let d = safe_normalize(world_dir, f);

    let cos_theta = d.dot(f).clamp(-1.0, 1.0);
    let theta = ops::acos(cos_theta); // radians, [0, PI]

    let x = d.dot(right);
    let y = d.dot(true_up);
    let mut phi = ops::atan2(y, x); // radians, (-PI, PI]
    if !phi.is_finite() {
        phi = 0.0;
    }
    if phi < 0.0 {
        phi += TAU;
    }

    let theta_deg = (theta * RAD_TO_DEG).clamp(0.0, 180.0);
    let mut phi_deg = phi * RAD_TO_DEG;
    if !phi_deg.is_finite() {
        phi_deg = 0.0;
    }
    // Keep strictly in [0, 360).
    phi_deg = phi_deg.rem_euclid(360.0);
    (theta_deg, phi_deg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn peak_normalize_maps_to_unit() {
        assert!(approx(peak_normalize(50.0, 100.0), 0.5, 1e-6));
        assert!(approx(peak_normalize(200.0, 100.0), 1.0, 1e-6)); // clamped
        assert_eq!(peak_normalize(10.0, 0.0), 0.0);
        assert_eq!(peak_normalize(f32::NAN, 100.0), 0.0);
    }

    #[test]
    fn isotropic_flux_matches_four_pi() {
        // Unit intensity everywhere, single azimuth -> flux = 4*PI.
        let g = PhotometricGrid::new(
            alloc::vec![0.0, 90.0, 180.0],
            alloc::vec![0.0],
            alloc::vec![1.0, 1.0, 1.0],
        )
        .unwrap();
        let flux = luminous_flux(&g);
        // Trapezoid of sin over [0,PI] with 3 samples underestimates slightly.
        assert!(flux > 0.0 && flux < 4.0 * PI);
        // Solid angle with unit intensity equals the flux here.
        assert!(approx(flux, measured_solid_angle(&g), 1e-4));
    }

    #[test]
    fn finer_grid_approaches_four_pi_solid_angle() {
        // 181 polar samples at 1-degree steps -> sphere solid angle ~ 4*PI.
        let mut vertical = Vec::new();
        let mut row = Vec::new();
        let mut t = 0.0;
        while t <= 180.0 + 1e-3 {
            vertical.push(t);
            row.push(1.0);
            t += 1.0;
        }
        let g = PhotometricGrid::new(vertical, alloc::vec![0.0], row).unwrap();
        let sa = measured_solid_angle(&g);
        assert!(approx(sa, 4.0 * PI, 1e-2));
    }

    #[test]
    fn mean_intensity_falls_back_to_peak_when_degenerate() {
        // Single polar sample -> no theta interval -> solid angle 0.
        let g = PhotometricGrid::uniform_single(77.0);
        assert!(approx(mean_intensity(&g), 77.0, 1e-6));
        assert!(approx(reference_intensity(&g, NormalizeMode::MeanFlux), 77.0, 1e-6));
    }

    #[test]
    fn reference_intensity_zero_for_empty() {
        let g = PhotometricGrid::uniform_single(0.0);
        assert_eq!(reference_intensity(&g, NormalizeMode::Peak), 0.0);
        assert_eq!(reference_intensity(&g, NormalizeMode::MeanFlux), 0.0);
    }

    #[test]
    fn basis_is_orthonormal_right_handed() {
        let (r, u, f) = luminaire_basis(Vec3::NEG_Z, Vec3::Y);
        assert!(approx(r.length(), 1.0, 1e-5));
        assert!(approx(u.length(), 1.0, 1e-5));
        assert!(approx(f.length(), 1.0, 1e-5));
        assert!(approx(r.dot(u), 0.0, 1e-5));
        assert!(approx(r.dot(f), 0.0, 1e-5));
        assert!(approx(u.dot(f), 0.0, 1e-5));
        // right x up == forward.
        assert!(r.cross(u).abs_diff_eq(f, 1e-4));
    }

    #[test]
    fn forward_direction_maps_to_theta_zero() {
        let f = Vec3::NEG_Z;
        let (theta, _phi) = world_to_local_angles(f, f, Vec3::Y);
        assert!(approx(theta, 0.0, 1e-3));
    }

    #[test]
    fn opposite_direction_maps_to_theta_180() {
        let f = Vec3::NEG_Z;
        let (theta, _phi) = world_to_local_angles(-f, f, Vec3::Y);
        assert!(approx(theta, 180.0, 1e-3));
    }

    #[test]
    fn perpendicular_direction_maps_to_theta_90() {
        let f = Vec3::NEG_Z;
        // up = +Y, so right = up x f = Y x (-Z) = -X. Direction +X -> phi near 180.
        let (theta, phi) = world_to_local_angles(Vec3::X, f, Vec3::Y);
        assert!(approx(theta, 90.0, 1e-3));
        assert!(phi >= 0.0 && phi < 360.0);
    }

    #[test]
    fn degenerate_inputs_do_not_nan() {
        let (theta, phi) = world_to_local_angles(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        assert!(theta.is_finite() && phi.is_finite());
        assert!(theta >= 0.0 && theta <= 180.0);
        assert!(phi >= 0.0 && phi < 360.0);
    }

    #[test]
    fn parallel_up_falls_back_gracefully() {
        // up parallel to forward -> basis must still be orthonormal.
        let (r, u, f) = luminaire_basis(Vec3::Y, Vec3::Y);
        assert!(approx(r.dot(f), 0.0, 1e-4));
        assert!(approx(u.dot(f), 0.0, 1e-4));
        assert!(approx(r.length(), 1.0, 1e-4));
    }
}
