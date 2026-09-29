//! Spectral dispersion: per-wavelength refractive index and colour fringing.
//!
//! Water refracts short (blue) wavelengths slightly more than long (red) ones,
//! so a single refracted ray splits into a faint rainbow fringe that is most
//! visible at grazing angles and through thick water. This module models that
//! with the classic `Cauchy` two-term index law and turns the per-channel
//! indices into the ordered screen-space refraction offsets the refraction pass
//! uses to sample the scene colour three times.
//!
//! Normal dispersion means the index rises as wavelength falls, so
//! `n_red < n_green < n_blue`. A higher index bends the ray more (`Snell`'s
//! law), giving the blue channel the largest lateral shift. These orderings are
//! what the tests below pin down. Only `sqrt` is used, there are no `f32`
//! equality tests, and there is no AI/ML.

use super::EPS;

/// Reference RGB wavelengths in micrometres used to sample the `Cauchy` law:
/// red `0.700`, green `0.546`, blue `0.440`.
pub const RGB_WAVELENGTHS_UM: [f32; 3] = [0.700, 0.546, 0.440];

/// Per-channel refractive indices for the red, green, and blue samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbIor {
    /// Refractive index at the red reference wavelength.
    pub r: f32,
    /// Refractive index at the green reference wavelength.
    pub g: f32,
    /// Refractive index at the blue reference wavelength.
    pub b: f32,
}

/// Evaluates the `Cauchy` refractive-index law `n(lambda) = a + b / lambda^2`.
///
/// `a` is the baseline index (about `1.324` for water) and `b` the dispersion
/// coefficient in micrometre-squared. The index falls monotonically as the
/// wavelength grows, so redder light has the lower index. A degenerate
/// non-positive wavelength clamps to a tiny positive value to avoid dividing by
/// zero.
#[must_use]
pub fn cauchy_ior(a: f32, b: f32, wavelength_um: f32) -> f32 {
    let lambda = wavelength_um.max(EPS);
    a + b / (lambda * lambda)
}

/// Samples the `Cauchy` law at the three [`RGB_WAVELENGTHS_UM`].
///
/// Returns indices satisfying `r < g < b` for any positive dispersion
/// coefficient `b`, the normal-dispersion ordering the refraction pass relies
/// on.
#[must_use]
pub fn spectral_iors(a: f32, b: f32) -> RgbIor {
    RgbIor {
        r: cauchy_ior(a, b, RGB_WAVELENGTHS_UM[0]),
        g: cauchy_ior(a, b, RGB_WAVELENGTHS_UM[1]),
        b: cauchy_ior(a, b, RGB_WAVELENGTHS_UM[2]),
    }
}

/// Transmitted-ray sine for one channel from `Snell`'s law.
///
/// Given the sine of the incidence angle in air and the water index for a
/// channel, returns `sin(theta_incident) / ior`, clamped to `0..=1`. A larger
/// index yields a smaller transmitted sine, i.e. a ray bent closer to the
/// normal.
#[must_use]
pub fn channel_transmitted_sine(sin_incidence: f32, ior: f32) -> f32 {
    let n = ior.max(EPS);
    (sin_incidence.clamp(0.0, 1.0) / n).clamp(0.0, 1.0)
}

/// Angular colour spread between the red and blue refracted rays.
///
/// Returns `sin_t(red) - sin_t(blue) >= 0`: because blue has the higher index
/// it bends more (smaller transmitted sine), so red trails behind and the
/// difference measures the width of the visible colour fringe. The spread grows
/// with the incidence angle, which is why dispersion reads strongest at grazing
/// angles. The result is non-negative for the normal-dispersion ordering.
#[must_use]
pub fn dispersion_spread(iors: RgbIor, sin_incidence: f32) -> f32 {
    let s = sin_incidence.clamp(0.0, 1.0);
    let sine_r = channel_transmitted_sine(s, iors.r);
    let sine_b = channel_transmitted_sine(s, iors.b);
    (sine_r - sine_b).max(0.0)
}

/// Per-channel screen-space refraction offset magnitudes.
///
/// Scales each channel's transmitted sine by the refraction `strength` (a
/// screen-space displacement gain folded with the water thickness). The blue
/// offset is the smallest and the red the largest, matching the transmitted-
/// sine ordering, so sampling the scene colour at these three offsets produces
/// the coloured fringe. All three offsets are non-negative.
#[must_use]
pub fn dispersion_offsets(iors: RgbIor, sin_incidence: f32, strength: f32) -> [f32; 3] {
    let s = sin_incidence.clamp(0.0, 1.0);
    let gain = strength.max(0.0);
    [
        channel_transmitted_sine(s, iors.r) * gain,
        channel_transmitted_sine(s, iors.g) * gain,
        channel_transmitted_sine(s, iors.b) * gain,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cauchy_index_falls_with_wavelength() {
        let blue = cauchy_ior(1.324, 0.003, 0.440);
        let red = cauchy_ior(1.324, 0.003, 0.700);
        assert!(blue > red, "shorter wavelength has the higher index");
        // Degenerate wavelength does not divide by zero.
        assert!(cauchy_ior(1.324, 0.003, 0.0).is_finite());
    }

    #[test]
    fn spectral_iors_are_ordered_r_lt_g_lt_b() {
        let iors = spectral_iors(1.324, 0.003);
        assert!(iors.r < iors.g);
        assert!(iors.g < iors.b);
        // Zero dispersion collapses the three indices together.
        let flat = spectral_iors(1.33, 0.0);
        assert!((flat.r - flat.b).abs() < EPS);
    }

    #[test]
    fn transmitted_sine_bends_more_at_higher_index() {
        let low = channel_transmitted_sine(0.8, 1.33);
        let high = channel_transmitted_sine(0.8, 1.34);
        assert!(high < low, "higher index bends closer to the normal");
        // Total internal reflection style clamp keeps the sine in range.
        assert!(channel_transmitted_sine(1.0, 0.5) <= 1.0 + EPS);
    }

    #[test]
    fn spread_is_non_negative_and_grows_with_incidence() {
        let iors = spectral_iors(1.324, 0.006);
        let grazing = dispersion_spread(iors, 0.95);
        let steep = dispersion_spread(iors, 0.2);
        assert!(grazing >= steep, "grazing angles disperse more");
        assert!(steep >= 0.0);
        // Monotonic across the incidence range.
        let mut prev = dispersion_spread(iors, 0.0);
        let mut s = 0.0;
        while s <= 1.0 {
            let v = dispersion_spread(iors, s);
            assert!(v + EPS >= prev, "spread must not shrink with incidence");
            prev = v;
            s += 0.05;
        }
    }

    #[test]
    fn offsets_are_ordered_and_non_negative() {
        let iors = spectral_iors(1.324, 0.006);
        let o = dispersion_offsets(iors, 0.7, 2.0);
        // Blue bends most so carries the smallest lateral offset.
        assert!(o[2] < o[0], "blue offset below red offset");
        assert!(o.iter().all(|&x| x >= 0.0));
        // Zero strength yields no offsets.
        let z = dispersion_offsets(iors, 0.7, 0.0);
        assert!(z.iter().all(|&x| x.abs() < EPS));
    }
}
