//! OpenPBR EON rough-diffuse reference lobe.
//!
//! OpenPBR's energy-preserving diffuse response is the Fujii Oren-Nayar lobe
//! plus the reciprocal multiple-scattering term from `EON_comp`.  The
//! directional albedo `Ê_ON(μ, σ)` and its cosine-weighted average are baked
//! from the specification's own `FON_brdf` equation.  Keeping the LUT in the
//! reference crate makes the CPU oracle independent of the real-time crate,
//! while the constants and interpolation are intentionally identical to the
//! real-time OpenPBR implementation.
//!
//! The lobe is sampled with the same cosine distribution as its Oren-Nayar
//! component.  The compensation changes only the value, not the sampling
//! density, so the estimator remains unbiased and the reported `pdf` stays
//! exact.

use super::oren_nayar::{OrenNayar, OrenNayarSample};
use super::sampler::SampleSource;
use super::{Vec3, INV_PI};

/// OpenPBR's authored diffuse roughness maps to `sigma ∈ [0, π/2]`.
pub const MAX_SIGMA: f32 = core::f32::consts::FRAC_PI_2;

/// Number of directional-cosine samples in the baked albedo table.
pub const N_MU: usize = 17;
/// Number of roughness rows in the baked albedo table.
pub const N_SIGMA: usize = 9;

/// `Ê_ON(μ, σ)` sampled on a uniform `(μ, sigma / (π/2))` grid.
const ALBEDO: [f32; N_MU * N_SIGMA] = [
    1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102,
    1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 1.000102, 0.986645,
    0.983973, 0.981022, 0.978077, 0.975073, 0.972048, 0.969053, 0.966109, 0.963232, 0.960478,
    0.957871, 0.955444, 0.953240, 0.951300, 0.949674, 0.948444, 0.947782, 0.922654, 0.917026,
    0.910807, 0.904613, 0.898283, 0.891915, 0.885615, 0.879407, 0.873357, 0.867555, 0.862069,
    0.856957, 0.852316, 0.848229, 0.844811, 0.842220, 0.840847, 0.846674, 0.839602, 0.831781,
    0.823991, 0.816032, 0.808029, 0.800101, 0.792296, 0.784691, 0.777392, 0.770494, 0.764063,
    0.758232, 0.753093, 0.748796, 0.745533, 0.743797, 0.787386, 0.779612, 0.771021, 0.762455,
    0.753710, 0.744912, 0.736201, 0.727622, 0.719264, 0.711242, 0.703662, 0.696595, 0.690184,
    0.684535, 0.679813, 0.676232, 0.674332, 0.746080, 0.737933, 0.728924, 0.719954, 0.710785,
    0.701562, 0.692436, 0.683445, 0.674682, 0.666280, 0.658332, 0.650927, 0.644210, 0.638288,
    0.633338, 0.629584, 0.627586, 0.717770, 0.709404, 0.700156, 0.690942, 0.681529, 0.672060,
    0.662687, 0.653458, 0.644460, 0.635832, 0.627671, 0.620066, 0.613169, 0.607089, 0.602005,
    0.598151, 0.596103, 0.698071, 0.689569, 0.680169, 0.670805, 0.661239, 0.651613, 0.642086,
    0.632702, 0.623559, 0.614790, 0.606495, 0.598766, 0.591756, 0.585576, 0.580408, 0.576491,
    0.574419, 0.684031, 0.675436, 0.665937, 0.656470, 0.646799, 0.637073, 0.627444, 0.617960,
    0.608717, 0.599852, 0.591468, 0.583657, 0.576569, 0.570325, 0.565099, 0.561141, 0.559036,
];

/// `⟨Ê_ON⟩(σ)` on the same roughness grid.
const AVERAGE: [f32; N_SIGMA] = [
    1.000102, 0.957581, 0.861456, 0.769724, 0.702815, 0.657445, 0.626761, 0.605569, 0.590533,
];

/// Bilinear lookup of the unit-albedo directional Oren-Nayar albedo.
#[must_use]
pub fn directional_albedo(mu: f32, sigma: f32) -> f32 {
    let x = mu.clamp(0.0, 1.0) * (N_MU - 1) as f32;
    let y = (sigma.max(0.0) / MAX_SIGMA).clamp(0.0, 1.0) * (N_SIGMA - 1) as f32;
    let i0 = (x as usize).min(N_MU - 2);
    let j0 = (y as usize).min(N_SIGMA - 2);
    let fx = x - i0 as f32;
    let fy = y - j0 as f32;
    let a = ALBEDO[j0 * N_MU + i0];
    let b = ALBEDO[j0 * N_MU + i0 + 1];
    let c = ALBEDO[(j0 + 1) * N_MU + i0];
    let d = ALBEDO[(j0 + 1) * N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    (top + (bottom - top) * fy).clamp(0.0, 1.0)
}

/// Linear lookup of the cosine-weighted mean unit-albedo response.
#[must_use]
pub fn average_albedo(sigma: f32) -> f32 {
    let y = (sigma.max(0.0) / MAX_SIGMA).clamp(0.0, 1.0) * (N_SIGMA - 1) as f32;
    let j0 = (y as usize).min(N_SIGMA - 2);
    let fy = y - j0 as f32;
    (AVERAGE[j0] + (AVERAGE[j0 + 1] - AVERAGE[j0]) * fy).clamp(0.0, 1.0)
}

/// The dimensionless `ρ_ms` coefficient from OpenPBR `EON_comp`.
#[must_use]
pub fn multiple_scattering_albedo(rho: f32, sigma: f32) -> f32 {
    let e = average_albedo(sigma).min(1.0 - 1.0e-4);
    let rho = rho.clamp(0.0, 1.0);
    let ratio = e / (1.0 - e);
    let denominator = (1.0 - rho * (1.0 - e)).max(1.0e-4);
    rho * rho * ratio / denominator
}

/// The reciprocal OpenPBR `EON_comp` BRDF term for a fixed direction pair.
#[must_use]
pub fn compensation(albedo: Vec3, cos_o: f32, cos_i: f32, sigma: f32) -> Vec3 {
    let sigma = sigma.max(0.0);
    if sigma <= 0.0 {
        return Vec3::ZERO;
    }
    let e_i = directional_albedo(cos_i, sigma);
    let e_o = directional_albedo(cos_o, sigma);
    let directional_gap = (1.0 - e_i).max(0.0) * (1.0 - e_o).max(0.0);
    let scale = directional_gap * INV_PI;
    Vec3::new(
        multiple_scattering_albedo(albedo.x, sigma) * scale,
        multiple_scattering_albedo(albedo.y, sigma) * scale,
        multiple_scattering_albedo(albedo.z, sigma) * scale,
    )
}

/// Energy-preserving OpenPBR rough diffuse response `f_ON + f_comp`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EonDiffuse {
    albedo: Vec3,
    sigma: f32,
    base: OrenNayar,
}

impl EonDiffuse {
    /// Builds an EON lobe from a base colour and OpenPBR `sigma` in radians.
    #[must_use]
    pub fn new(albedo: Vec3, sigma: f32) -> Self {
        Self {
            albedo,
            sigma,
            base: OrenNayar::new(albedo, sigma),
        }
    }

    /// Evaluates `f_ON + f_comp` for a pair of directions above `normal`.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::ZERO;
        }
        self.base
            .evaluate(wo, wi, normal)
            .add(compensation(self.albedo, cos_o, cos_i, self.sigma))
    }

    /// Returns the exact cosine-sampling density used by the base Oren-Nayar
    /// lobe. EON changes the BRDF value, not the sampling distribution.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        self.base.pdf(wo, wi, normal)
    }

    /// Samples the cosine distribution and reports the compensated BRDF value.
    #[must_use]
    pub fn sample(
        &self,
        wo: Vec3,
        normal: Vec3,
        rng: &mut impl SampleSource,
    ) -> Option<OrenNayarSample> {
        let oriented = normal.faced_toward(wo);
        let mut sample = self.base.sample(wo, oriented, rng)?;
        sample.value = self.evaluate(wo, sample.direction, oriented);
        Some(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_pt::sampler::Rng;

    const N: Vec3 = Vec3::new(0.0, 0.0, 1.0);

    fn view_at(mu: f32) -> Vec3 {
        let sin = (1.0 - mu * mu).max(0.0).sqrt();
        Vec3::new(sin, 0.0, mu).normalize_or_zero()
    }

    #[test]
    fn smooth_limit_is_exact_lambert_and_has_no_compensation() {
        let albedo = Vec3::new(0.2, 0.5, 0.9);
        let eon = EonDiffuse::new(albedo, 0.0);
        let wo = view_at(0.61);
        let wi = view_at(0.37);
        assert_eq!(compensation(albedo, 0.61, 0.37, 0.0), Vec3::ZERO);
        assert_eq!(eon.evaluate(wo, wi, N), albedo.scale(INV_PI));
    }

    #[test]
    fn lut_anchors_match_the_realtime_openpbr_table() {
        assert_eq!(directional_albedo(0.0, 0.0), 1.0);
        assert!((directional_albedo(0.0, MAX_SIGMA) - 0.684031).abs() < 1.0e-6);
        assert!((directional_albedo(1.0, MAX_SIGMA) - 0.559036).abs() < 1.0e-6);
        assert!((average_albedo(MAX_SIGMA) - 0.590533).abs() < 1.0e-6);
        assert!(average_albedo(0.7) < average_albedo(0.2));
    }

    #[test]
    fn compensation_matches_the_eon_equation_for_a_fixed_pair() {
        let albedo = Vec3::new(0.25, 0.5, 0.75);
        let sigma = 0.6 * MAX_SIGMA;
        let cos_o = 0.63;
        let cos_i = 0.91;
        let e_o = directional_albedo(cos_o, sigma);
        let e_i = directional_albedo(cos_i, sigma);
        let mean = average_albedo(sigma);
        let ratio = mean / (1.0 - mean);
        let gap = (1.0 - e_o) * (1.0 - e_i) * INV_PI;
        let expected = Vec3::new(
            albedo.x * albedo.x * ratio / (1.0 - albedo.x * (1.0 - mean)) * gap,
            albedo.y * albedo.y * ratio / (1.0 - albedo.y * (1.0 - mean)) * gap,
            albedo.z * albedo.z * ratio / (1.0 - albedo.z * (1.0 - mean)) * gap,
        );
        let actual = compensation(albedo, cos_o, cos_i, sigma);
        assert!(
            actual.sub(expected).length() < 1.0e-7,
            "{actual:?} vs {expected:?}"
        );

        // Independent golden values generated from the OpenPBR EON equation
        // at this exact parameter tuple. These keep the test from passing if
        // both the implementation and its expected-value expression drift in
        // the same way.
        let golden = Vec3::new(0.0051952518, 0.0228602128, 0.0571539083);
        assert!(
            actual.sub(golden).length() < 1.0e-6,
            "{actual:?} differs from EON golden {golden:?}"
        );
    }

    #[test]
    fn sampling_pdf_and_value_are_consistent() {
        let eon = EonDiffuse::new(Vec3::new(0.7, 0.5, 0.3), 0.8 * MAX_SIGMA);
        let wo = view_at(0.58);
        let mut rng = Rng::seed(0xE0A2_026);
        for _ in 0..10_000 {
            let Some(sample) = eon.sample(wo, N, &mut rng) else {
                continue;
            };
            assert!((sample.pdf - eon.pdf(wo, sample.direction, N)).abs() < 1.0e-6);
            assert!(
                sample
                    .value
                    .sub(eon.evaluate(wo, sample.direction, N))
                    .length()
                    < 1.0e-6
            );
        }
    }

    #[test]
    fn white_furnace_recovers_the_rough_diffuse_energy() {
        let wo = view_at(0.55);
        let eon = EonDiffuse::new(Vec3::ONE, 0.8 * MAX_SIGMA);
        let mut rng = Rng::seed(0xE0F_ACE);
        let mut sum = 0.0_f64;
        let samples = 1u32 << 17;
        for _ in 0..samples {
            let Some(sample) = eon.sample(wo, N, &mut rng) else {
                continue;
            };
            let cos_i = N.dot(sample.direction).max(0.0);
            sum += f64::from(sample.value.x * cos_i / sample.pdf);
        }
        let albedo = sum / f64::from(samples);
        assert!(
            (albedo - 1.0).abs() < 1.5e-2,
            "white-furnace albedo {albedo}"
        );
    }
}
