//! The Shirley-Chiu concentric square-to-disk map.
//!
//! Mapping a unit square onto the unit disk is the primitive that lets a
//! low-discrepancy (`QMC`) sequence drive aperture and hemisphere sampling
//! without the stratification loss of rejection sampling. The naive polar map
//! `(r, theta) = (sqrt(u), 2*pi*v)` is area-preserving but distorts distances
//! badly near the centre, clumping stratified samples; Peter Shirley and
//! Kenneth Chiu's concentric map instead folds the square into four wedges and
//! sends concentric square rings to concentric circles, so neighbouring square
//! samples stay neighbours on the disk. The result is area-preserving (uniform
//! input gives uniform-by-area output) and, crucially, keeps the discrepancy of
//! the input sequence — exactly what the thin-lens aperture sample wants when it
//! is pulled from the same scrambled `QMC` stream as the sub-pixel jitter.
//!
//! The crate's determinism policy forbids calling `sin`/`cos`, so the only
//! transcendental in the whole reference tracer is `sqrt`. The wedge angle here
//! never leaves `[-pi/4, pi/4]`, so this module evaluates `sin`/`cos` with fixed
//! minimax-grade truncated Taylor polynomials (pure multiplies and adds, carried
//! in `f64`). Over `[-pi/4, pi/4]` the residual is below `2e-9`, far tighter than
//! `f32` can represent, so the map is numerically indistinguishable from an exact
//! trigonometric implementation while staying a purely algebraic routine.

use super::sampler::Sample2;

/// A quarter of pi; the half-width of a single concentric wedge's angular sweep.
const FRAC_PI_4: f64 = core::f64::consts::FRAC_PI_4;

/// Evaluates `sin(x)` for `x` in `[-pi/4, pi/4]` with a truncated Taylor series
/// carried to the eleventh order.
///
/// The series `x - x^3/6 + x^5/120 - x^7/5040 + x^9/362880 - x^11/39916800` has
/// a leading omitted term on the order of `x^13 / 6227020800`, which stays below
/// `1e-11` for `|x| <= pi/4`. Evaluation is Horner-factored so it costs only multiplies and
/// adds, respecting the crate's "no `sin`/`cos`" determinism policy.
fn sin_quarter(x: f64) -> f64 {
    let x2 = x * x;
    x * (1.0
        + x2 * (-1.0 / 6.0
            + x2 * (1.0 / 120.0
                + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362_880.0 + x2 * (-1.0 / 39_916_800.0))))))
}

/// Evaluates `cos(x)` for `x` in `[-pi/4, pi/4]` with a truncated Taylor series
/// carried to the twelfth order.
///
/// The series `1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320 - x^10/3628800 +
/// x^12/479001600` has a leading omitted term on the order of
/// `x^14 / 87178291200`, below `1e-13` for `|x| <= pi/4`. Like [`sin_quarter`] it is Horner-factored into multiplies and
/// adds so no transcendental call is made.
fn cos_quarter(x: f64) -> f64 {
    let x2 = x * x;
    1.0 + x2
        * (-0.5
            + x2 * (1.0 / 24.0
                + x2 * (-1.0 / 720.0
                    + x2 * (1.0 / 40320.0
                        + x2 * (-1.0 / 3_628_800.0 + x2 * (1.0 / 479_001_600.0))))))
}

/// Maps a unit-square sample in `[0, 1]^2` onto the unit disk `x^2 + y^2 <= 1`
/// with the area-preserving concentric map.
///
/// Uniform input is sent to a uniform-by-area distribution on the disk, and the
/// map preserves the discrepancy of a low-discrepancy input sequence, so a `QMC`
/// stream warped through here stratifies the disk far better than rejection
/// sampling would. The exact centre of the square has no well-defined wedge
/// angle and is sent to the origin.
#[must_use]
pub(crate) fn concentric_disk(sample: Sample2) -> (f32, f32) {
    // Remap the unit square into `[-1, 1]^2` so the wedges are symmetric about
    // the origin.
    let a = 2.0 * f64::from(sample.x) - 1.0;
    let b = 2.0 * f64::from(sample.y) - 1.0;
    // The square centre maps to the disk centre; a sum of squares is `<= 0`
    // only when both coordinates are exactly zero, so no angle is formed.
    if a * a + b * b <= 0.0 {
        return (0.0, 0.0);
    }
    if a * a > b * b {
        // Horizontal wedge: the radius is `|a|` and the angle sweeps
        // `[-pi/4, pi/4]` as `b/a` runs over `[-1, 1]`.
        let phi = FRAC_PI_4 * (b / a);
        ((a * cos_quarter(phi)) as f32, (a * sin_quarter(phi)) as f32)
    } else {
        // Vertical wedge: fold through the diagonal so the polynomial argument
        // stays within `[-pi/4, pi/4]`. With `t = (pi/4)*(a/b)` the exact angle
        // is `pi/2 - t`, and `cos(pi/2 - t) = sin(t)`, `sin(pi/2 - t) = cos(t)`.
        let t = FRAC_PI_4 * (a / b);
        ((b * sin_quarter(t)) as f32, (b * cos_quarter(t)) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference `pi/2` used to validate the folded (vertical-wedge) angle.
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;

    #[test]
    fn polynomials_match_closed_form_angles() {
        // sin(0) = 0, cos(0) = 1.
        assert!(sin_quarter(0.0).abs() < 1e-12);
        assert!((cos_quarter(0.0) - 1.0).abs() < 1e-12);
        // sin(pi/6) = 1/2, cos(pi/6) = sqrt(3)/2 (pi/6 < pi/4, in range).
        let pi_6 = FRAC_PI_2 / 3.0;
        assert!((sin_quarter(pi_6) - 0.5).abs() < 1e-9);
        assert!((cos_quarter(pi_6) - (3.0_f64).sqrt() / 2.0).abs() < 1e-9);
        // sin(pi/4) = cos(pi/4) = sqrt(2)/2 at the wedge boundary.
        let root_half = (2.0_f64).sqrt() / 2.0;
        assert!((sin_quarter(FRAC_PI_4) - root_half).abs() < 1e-9);
        assert!((cos_quarter(FRAC_PI_4) - root_half).abs() < 1e-9);
    }

    #[test]
    fn pythagorean_identity_holds_across_the_wedge() {
        // sin^2 + cos^2 = 1 everywhere the polynomials are used.
        for i in 0..=64 {
            let x = -FRAC_PI_4 + (2.0 * FRAC_PI_4) * (f64::from(i) / 64.0);
            let s = sin_quarter(x);
            let c = cos_quarter(x);
            assert!((s * s + c * c - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn sin_is_monotone_increasing_over_the_wedge() {
        let mut prev = sin_quarter(-FRAC_PI_4);
        for i in 1..=64 {
            let x = -FRAC_PI_4 + (2.0 * FRAC_PI_4) * (f64::from(i) / 64.0);
            let s = sin_quarter(x);
            assert!(s > prev);
            prev = s;
        }
    }

    #[test]
    fn centre_maps_to_the_origin() {
        let (x, y) = concentric_disk(Sample2 { x: 0.5, y: 0.5 });
        assert!(x.abs() < 1e-6);
        assert!(y.abs() < 1e-6);
    }

    #[test]
    fn every_sample_lands_inside_the_unit_disk() {
        // A dense grid (including the boundary) never escapes the unit disk.
        for j in 0..=32 {
            for i in 0..=32 {
                let s = Sample2 {
                    x: (i as f32) / 32.0,
                    y: (j as f32) / 32.0,
                };
                let (x, y) = concentric_disk(s);
                assert!(x * x + y * y <= 1.0 + 1e-5);
            }
        }
    }

    #[test]
    fn square_corners_reach_the_unit_circle() {
        // The four square corners sit on the disk boundary (radius 1).
        for &(u, v) in &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            let (x, y) = concentric_disk(Sample2 { x: u, y: v });
            let r = (x * x + y * y).sqrt();
            assert!((r - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn map_is_area_preserving() {
        // The concentric map carries uniform density on the square to uniform
        // density on the disk, i.e. its Jacobian determinant is a constant equal
        // to the disk-to-square area ratio (pi / 1 = pi for a `[0, 1]^2` domain).
        // Estimate it by central differences at several interior points chosen
        // off the wedge-switching diagonal, where the map is smooth.
        const H: f64 = 1.0e-3;
        let eval = |u: f64, v: f64| -> (f64, f64) {
            let (x, y) = concentric_disk(Sample2 {
                x: u as f32,
                y: v as f32,
            });
            (f64::from(x), f64::from(y))
        };
        for &(u, v) in &[
            (0.3, 0.75),
            (0.8, 0.55),
            (0.15, 0.6),
            (0.9, 0.35),
            (0.6, 0.95),
            (0.45, 0.1),
        ] {
            let (xup, yup) = eval(u + H, v);
            let (xum, yum) = eval(u - H, v);
            let (xvp, yvp) = eval(u, v + H);
            let (xvm, yvm) = eval(u, v - H);
            let dx_du = (xup - xum) / (2.0 * H);
            let dy_du = (yup - yum) / (2.0 * H);
            let dx_dv = (xvp - xvm) / (2.0 * H);
            let dy_dv = (yvp - yvm) / (2.0 * H);
            let jacobian = (dx_du * dy_dv - dx_dv * dy_du).abs();
            // pi, with slack for the f32 round-trip and finite-difference error.
            assert!(
                (jacobian - core::f64::consts::PI).abs() < 2.0e-2,
                "jacobian at ({u}, {v}) was {jacobian}, expected pi"
            );
        }
    }
}
