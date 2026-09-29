//! Spectral night-sky and twilight tinting for the volumetric cloud subsystem
//! (design section 8b, testability section 16).
//!
//! The atmosphere itself is the shared base lighting service; this module holds
//! only the small, read-only *spectral* helpers the cloud/sky hookup needs when
//! the scene wants a physically flavoured night sky and sunset. It never
//! rewrites the shared atmosphere `LUT`; it supplies deterministic `CPU`
//! reference math the `GPU` `WESL` kernels mirror with native intrinsics:
//!
//! - [`SpectralBands`] — a partition-of-unity set of wavelength-bucket weights
//!   (`RGB` or an arbitrary bucket count). Construction is *energy conserving*:
//!   weights are floored to non-negative and renormalised so they sum to one.
//! - [`rayleigh_phase`] — the normalised `Rayleigh` scattering phase
//!   `3/(16*PI) * (1 + cos^2 theta)`, whose solid-angle integral is one; this
//!   is what tints a clear sky blue and reddens the low sun.
//! - [`ozone_absorption`] — a non-negative `ozone` absorption coefficient
//!   (`Chappuis` plus `Huggins` band approximation) as a function of wavelength
//!   in nanometres; it deepens twilight blues.
//! - [`sunset_reddening`] — a bounded, monotone reddening amount driven by the
//!   solar altitude: the lower the sun, the redder the light.
//! - [`spectral_to_rgb`] — collapses a [`SpectralBands`] distribution into a
//!   linear `RGB` [`Vec3`] via a non-negative `CIE`-flavoured response, keeping
//!   the total weight conserved and never emitting a negative channel.
//!
//! All functions are pure, deterministic, and never `panic`: out-of-range
//! inputs are clamped rather than propagating `NaN`. The determinism policy
//! allows only [`f32::sqrt`] among the float intrinsics, so every exponential
//! routes through [`super::math::exp_approx`]. Wavelengths are given in the
//! `nanometre` (`nm`) unit throughout.

#![forbid(unsafe_code)]

use alloc::vec;
use alloc::vec::Vec;

use super::math::{clamp, exp_approx, lerp, saturate, smoothstep, EPS, PI};
use super::Vec3;

/// Normalisation constant `3 / (16 * PI)` of the `Rayleigh` phase function, so
/// its integral over the full sphere of solid angle equals one.
const RAYLEIGH_NORM: f32 = 3.0 / (16.0 * PI);

/// Centre wavelength (`nm`) of the broad visible `Chappuis` `ozone` band.
const CHAPPUIS_CENTER_NM: f32 = 602.0;

/// Gaussian half-width (`nm`) of the `Chappuis` `ozone` band.
const CHAPPUIS_WIDTH_NM: f32 = 90.0;

/// Peak absorption coefficient of the `Chappuis` `ozone` band (dimensionless,
/// relative units); non-negative by construction.
const CHAPPUIS_PEAK: f32 = 0.06;

/// Centre wavelength (`nm`) of the near-`UV` `Huggins` `ozone` band.
const HUGGINS_CENTER_NM: f32 = 320.0;

/// Gaussian half-width (`nm`) of the `Huggins` `ozone` band.
const HUGGINS_WIDTH_NM: f32 = 40.0;

/// Peak absorption coefficient of the `Huggins` `ozone` band (relative units).
const HUGGINS_PEAK: f32 = 0.01;

/// Solar altitude (radians) above which no sunset reddening is applied; below
/// it the reddening ramps smoothly toward its maximum at the horizon.
const REDDEN_MAX_ALTITUDE: f32 = 0.35;

/// `CIE`-flavoured red response centre wavelength (`nm`).
const R_CENTER_NM: f32 = 610.0;

/// `CIE`-flavoured green response centre wavelength (`nm`).
const G_CENTER_NM: f32 = 550.0;

/// `CIE`-flavoured blue response centre wavelength (`nm`).
const B_CENTER_NM: f32 = 465.0;

/// Gaussian half-width (`nm`) shared by the three `RGB` response lobes.
const RGB_WIDTH_NM: f32 = 50.0;

/// Shortest visible wavelength (`nm`) mapped to a bucket in [`spectral_to_rgb`].
const VIS_MIN_NM: f32 = 380.0;

/// Longest visible wavelength (`nm`) mapped to a bucket in [`spectral_to_rgb`].
const VIS_MAX_NM: f32 = 700.0;

/// A non-negative Gaussian lobe `peak * exp(-((x - center) / width)^2)`.
///
/// `width` is floored to a small positive value so a degenerate lobe never
/// divides by zero, and `peak` is floored to zero so the result is always
/// `>= 0`. Used to shape the `ozone` bands and the `CIE`-flavoured `RGB`
/// response, both of which must stay non-negative.
#[must_use]
fn gaussian(x: f32, center: f32, width: f32, peak: f32) -> f32 {
    let w = width.max(1e-3);
    let d = (x - center) / w;
    peak.max(0.0) * exp_approx(-d * d)
}

/// A non-negative, `L1`-normalised `CIE`-flavoured `RGB` response for a single
/// wavelength (`nm`).
///
/// Three Gaussian lobes (red/green/blue) are summed and then divided by their
/// total so the returned [`Vec3`] channels are non-negative and sum to one
/// (a partition of unity). When the wavelength falls outside every lobe and the
/// total collapses below [`EPS`], a neutral grey `1/3` per channel is returned
/// so the response never divides by zero and total weight is still conserved.
#[must_use]
fn cie_rgb_response(nm: f32) -> Vec3 {
    let r = gaussian(nm, R_CENTER_NM, RGB_WIDTH_NM, 1.0);
    let g = gaussian(nm, G_CENTER_NM, RGB_WIDTH_NM, 1.0);
    let b = gaussian(nm, B_CENTER_NM, RGB_WIDTH_NM, 1.0);
    let sum = r + g + b;
    if sum <= EPS {
        return Vec3::splat(1.0 / 3.0);
    }
    Vec3::new(r / sum, g / sum, b / sum)
}

/// A partition-of-unity set of spectral wavelength-bucket weights.
///
/// The weights are always non-negative and (when non-empty) sum to one, so a
/// [`SpectralBands`] describes an energy-conserving spectral power
/// distribution. Buckets are ordered from short to long wavelength; the count
/// is arbitrary (three for `RGB`, more for a finer spectrum).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpectralBands {
    /// Per-bucket non-negative weights that sum to one (empty when there are no
    /// buckets). Kept private so the partition-of-unity invariant established by
    /// [`SpectralBands::new`] can never be violated from outside.
    weights: Vec<f32>,
}

impl SpectralBands {
    /// Builds a normalised band set from raw `weights`.
    ///
    /// Each weight is floored to zero (negative energy is unphysical), then the
    /// whole set is renormalised so it sums to one. When every weight is zero
    /// (or the total is below [`EPS`]) the energy is spread uniformly across the
    /// buckets so the result is still a valid partition of unity. An empty input
    /// yields an empty band set. This never `panic`s.
    #[must_use]
    pub fn new(mut weights: Vec<f32>) -> Self {
        let n = weights.len();
        if n == 0 {
            return Self { weights };
        }
        let mut sum = 0.0;
        let mut i = 0;
        while i < n {
            if weights[i] < 0.0 {
                weights[i] = 0.0;
            }
            sum += weights[i];
            i += 1;
        }
        if sum <= EPS {
            let uniform = 1.0 / (n as f32);
            let mut j = 0;
            while j < n {
                weights[j] = uniform;
                j += 1;
            }
        } else {
            let inv = 1.0 / sum;
            let mut j = 0;
            while j < n {
                weights[j] *= inv;
                j += 1;
            }
        }
        Self { weights }
    }

    /// Builds a three-bucket (`RGB`) band set from raw red/green/blue weights,
    /// normalised by [`SpectralBands::new`].
    #[must_use]
    pub fn from_rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(vec![r, g, b])
    }

    /// Borrows the normalised per-bucket weights.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// The number of wavelength buckets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether there are no buckets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// The sum of the (normalised) weights: one for a non-empty band set, zero
    /// for an empty one.
    #[must_use]
    pub fn sum(&self) -> f32 {
        let mut total = 0.0;
        let mut i = 0;
        while i < self.weights.len() {
            total += self.weights[i];
            i += 1;
        }
        total
    }
}

/// The normalised `Rayleigh` scattering phase function.
///
/// `cos_theta` is the cosine of the scattering angle; it is clamped to
/// `-1..=1` so out-of-range inputs never leave the valid domain. The value is
/// `3/(16*PI) * (1 + cos^2 theta)`, whose integral over the sphere of solid
/// angle is one. This is the angular signature that makes a clear sky blue and
/// forward/back-scatter symmetric.
#[must_use]
pub fn rayleigh_phase(cos_theta: f32) -> f32 {
    let c = clamp(cos_theta, -1.0, 1.0);
    RAYLEIGH_NORM * (1.0 + c * c)
}

/// A non-negative `ozone` absorption coefficient at `wavelength_nm` (`nm`).
///
/// Modelled as the sum of the broad visible `Chappuis` band and the near-`UV`
/// `Huggins` band, each a non-negative Gaussian lobe. Because both lobes are
/// `>= 0`, the returned coefficient is always non-negative for any wavelength,
/// including out-of-range ones (they simply fall to zero far from the bands).
#[must_use]
pub fn ozone_absorption(wavelength_nm: f32) -> f32 {
    let chappuis = gaussian(
        wavelength_nm,
        CHAPPUIS_CENTER_NM,
        CHAPPUIS_WIDTH_NM,
        CHAPPUIS_PEAK,
    );
    let huggins = gaussian(
        wavelength_nm,
        HUGGINS_CENTER_NM,
        HUGGINS_WIDTH_NM,
        HUGGINS_PEAK,
    );
    chappuis + huggins
}

/// The sunset reddening amount in `0..=1` as a function of solar altitude.
///
/// `sun_altitude` is the sun's angular altitude in radians (zero at the
/// horizon, positive above it, negative below). The amount is
/// `saturate(1 - smoothstep(0, REDDEN_MAX_ALTITUDE, sun_altitude))`: it is one
/// at and below the horizon (full reddening), decreases *monotonically* as the
/// sun climbs, and is zero once the sun rises above [`REDDEN_MAX_ALTITUDE`]. It
/// is therefore bounded to `0..=1` and monotone non-increasing in altitude, so
/// a lower sun always yields at least as much red shift.
#[must_use]
pub fn sunset_reddening(sun_altitude: f32) -> f32 {
    saturate(1.0 - smoothstep(0.0, REDDEN_MAX_ALTITUDE, sun_altitude))
}

/// Collapses a spectral [`SpectralBands`] distribution into a linear `RGB`
/// [`Vec3`].
///
/// Each bucket is assigned a representative wavelength at its centre within the
/// visible span (`VIS_MIN_NM..=VIS_MAX_NM`), converted to a non-negative,
/// `L1`-normalised `CIE`-flavoured `RGB` response via [`cie_rgb_response`], and
/// accumulated weighted by the (normalised) bucket weight. Because each per-band
/// response sums to one across its channels, the total of the returned `RGB`
/// channels equals the sum of the band weights (one for a normalised, non-empty
/// band set), so the mapping conserves weight and never produces a negative
/// channel. An empty band set maps to [`Vec3::ZERO`].
#[must_use]
pub fn spectral_to_rgb(bands: &SpectralBands) -> Vec3 {
    let n = bands.len();
    if n == 0 {
        return Vec3::ZERO;
    }
    let weights = bands.weights();
    let mut acc = Vec3::ZERO;
    let mut i = 0;
    while i < n {
        let frac = ((i as f32) + 0.5) / (n as f32);
        let nm = lerp(VIS_MIN_NM, VIS_MAX_NM, frac);
        let response = cie_rgb_response(nm);
        acc = acc.add(response.scale(weights[i]));
        i += 1;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for weight-conservation equalities.
    const WEIGHT_TOL: f32 = 1e-6;

    #[test]
    fn bands_normalise_to_partition_of_unity() {
        let bands = SpectralBands::from_rgb(2.0, 1.0, 1.0);
        assert!((bands.sum() - 1.0).abs() < WEIGHT_TOL);
        let w = bands.weights();
        let mut i = 0;
        while i < w.len() {
            assert!(w[i] >= 0.0, "weight must be non-negative");
            i += 1;
        }
        // The 2:1:1 ratio survives normalisation.
        assert!((w[0] - 0.5).abs() < 1e-4);
        assert!((w[1] - 0.25).abs() < 1e-4);
        assert!((w[2] - 0.25).abs() < 1e-4);
    }

    #[test]
    fn bands_clamp_negative_weights() {
        let bands = SpectralBands::new(vec![-3.0, 1.0, 3.0]);
        assert!((bands.sum() - 1.0).abs() < WEIGHT_TOL);
        let w = bands.weights();
        let mut i = 0;
        while i < w.len() {
            assert!(w[i] >= 0.0, "negative energy must be floored");
            i += 1;
        }
        // The clamped-away first bucket contributes nothing.
        assert!(w[0].abs() < WEIGHT_TOL);
    }

    #[test]
    fn bands_all_zero_spreads_uniformly() {
        let bands = SpectralBands::new(vec![0.0, 0.0, 0.0, 0.0]);
        assert!((bands.sum() - 1.0).abs() < WEIGHT_TOL);
        let w = bands.weights();
        let mut i = 0;
        while i < w.len() {
            assert!((w[i] - 0.25).abs() < 1e-4, "uniform fill expected");
            i += 1;
        }
    }

    #[test]
    fn bands_empty_is_safe() {
        let bands = SpectralBands::new(Vec::new());
        assert!(bands.is_empty());
        assert_eq!(bands.len(), 0);
        assert_eq!(bands.sum(), 0.0);
        assert_eq!(spectral_to_rgb(&bands), Vec3::ZERO);
    }

    #[test]
    fn ozone_absorption_is_non_negative_across_the_spectrum() {
        let mut nm = 250.0;
        while nm <= 800.0 {
            assert!(
                ozone_absorption(nm) >= 0.0,
                "ozone absorption must be non-negative at {nm} nm"
            );
            nm += 5.0;
        }
        // Far out-of-range wavelengths still stay non-negative and finite.
        assert!(ozone_absorption(-100.0) >= 0.0);
        assert!(ozone_absorption(5000.0) >= 0.0);
    }

    #[test]
    fn ozone_absorption_is_deterministic() {
        assert_eq!(ozone_absorption(602.0), ozone_absorption(602.0));
        assert_eq!(ozone_absorption(320.0), ozone_absorption(320.0));
    }

    #[test]
    fn rayleigh_phase_integrates_to_one_over_the_sphere() {
        // Integral over solid angle is 2*PI * integral over mu in [-1, 1].
        let n = 2000;
        let dmu = 2.0 / (n as f32);
        let mut integral = 0.0;
        let mut i = 0;
        while i < n {
            let mu = -1.0 + ((i as f32) + 0.5) * dmu;
            integral += rayleigh_phase(mu) * dmu;
            i += 1;
        }
        integral *= 2.0 * PI;
        assert!(
            (integral - 1.0).abs() < 1e-2,
            "rayleigh phase must be normalised, got {integral}"
        );
    }

    #[test]
    fn rayleigh_phase_clamps_out_of_range_cosine() {
        // Beyond +-1 the value saturates to the clamped endpoint, no panic.
        assert_eq!(rayleigh_phase(5.0), rayleigh_phase(1.0));
        assert_eq!(rayleigh_phase(-5.0), rayleigh_phase(-1.0));
        assert!(rayleigh_phase(0.0) > 0.0);
    }

    #[test]
    fn sunset_reddening_is_bounded_and_monotone_non_increasing() {
        let mut prev = sunset_reddening(-0.3);
        assert!((0.0..=1.0).contains(&prev));
        let mut i = 1;
        while i <= 100 {
            let altitude = -0.3 + (i as f32) / 100.0 * 1.3;
            let cur = sunset_reddening(altitude);
            assert!((0.0..=1.0).contains(&cur), "reddening {cur} out of range");
            assert!(
                cur <= prev + 1e-4,
                "reddening must not increase as the sun rises"
            );
            prev = cur;
            i += 1;
        }
    }

    #[test]
    fn sunset_reddening_hits_its_endpoints() {
        // At and below the horizon the light is fully reddened.
        assert!((sunset_reddening(-0.1) - 1.0).abs() < 1e-4);
        assert!((sunset_reddening(0.0) - 1.0).abs() < 1e-4);
        // Well above REDDEN_MAX_ALTITUDE there is no reddening.
        assert!(sunset_reddening(1.0) < 1e-4);
    }

    #[test]
    fn spectral_to_rgb_conserves_weight_and_stays_non_negative() {
        let bands = SpectralBands::from_rgb(0.3, 0.5, 0.2);
        let rgb = spectral_to_rgb(&bands);
        assert!(rgb.x >= 0.0 && rgb.y >= 0.0 && rgb.z >= 0.0);
        // Total channel weight equals the (unit) band weight sum.
        let total = rgb.x + rgb.y + rgb.z;
        assert!(
            (total - 1.0).abs() < 1e-4,
            "spectral-to-rgb must conserve weight, got {total}"
        );
    }

    #[test]
    fn spectral_to_rgb_many_buckets_conserves_weight() {
        let bands = SpectralBands::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let rgb = spectral_to_rgb(&bands);
        assert!(rgb.x >= 0.0 && rgb.y >= 0.0 && rgb.z >= 0.0);
        let total = rgb.x + rgb.y + rgb.z;
        assert!((total - 1.0).abs() < 1e-4, "weight not conserved: {total}");
    }

    #[test]
    fn spectral_to_rgb_is_deterministic() {
        let bands = SpectralBands::from_rgb(0.4, 0.4, 0.2);
        assert_eq!(spectral_to_rgb(&bands), spectral_to_rgb(&bands));
    }
}
