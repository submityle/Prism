//! `GGX` multiple-scattering energy compensation for particle `PBR` shading
//! (design §17, "共享 `PBR` closure（`GGX` + 多散近似）"; see
//! `docs/prism_particle_engine_design_zh.md` §17, which calls for a single-plus
//! multiple-scattering specular response so lit/mesh particles never go
//! energy-deficient — "避免烟发死黑").
//!
//! A single-scattering microfacet `BRDF` (the `GGX`/Smith specular lobe used by
//! the shared `PBR` closure) loses energy as roughness grows: light that would
//! bounce a second or third time between microfacets is simply dropped, so
//! rough metals and bright energy particles darken toward grazing angles. This
//! module owns the `CPU` reference for the *multiple-scattering* term that puts
//! that energy back, matching the Kulla-Conty 2017 ("Revisiting Physically
//! Based Shading at Imageworks") construction with the analytic `Fresnel`
//! average popularized by Karis and Fdez-Agüera. A future `GPU` draw kernel can
//! reproduce it bit for bit. It is distinct from the specular anti-aliasing
//! roughness remap ([`super::specular_aa`]) and the `Fresnel` rim term
//! ([`super::fresnel_rim`]); it consumes a roughness/`mu`/`F0` triple and emits
//! *compensated energy*, not a lobe shape.
//!
//! Four analytic pieces compose here, all transcendental-free:
//!
//! 1. [`single_scatter_directional_albedo`] — the directional albedo `E_ss(mu,
//!    roughness)` of the single-scattering `GGX` lobe under a white furnace
//!    (`F = 1`), as a pure rational fit in the Karis `EnvBRDFApprox` spirit (a
//!    roughness-scaled, grazing-weighted deficit) with no `exp2`/`powf`.
//! 2. [`average_albedo`] — the cosine-weighted hemispherical average
//!    `E_avg(roughness)`. It is the *exact* closed-form integral of the `E_ss`
//!    fit (`E_avg = 2∫₀¹ E_ss(mu)·mu dmu`), hence a polynomial in the linear
//!    roughness `alpha`, so the two stay perfectly consistent.
//! 3. [`average_fresnel`] — the analytic average `Fresnel`
//!    `F_avg = F0 + (1 - F0)/21`, the exact cosine-weighted hemispherical
//!    integral of Schlick's `(1 - cos)^5` `Fresnel` (see
//!    [`super::fresnel_rim::fresnel_schlick`], whose pointwise form this
//!    averages rather than duplicates).
//! 4. [`multiscatter_fresnel_scale`] and [`multiscatter_directional_albedo`] —
//!    the Kulla-Conty multiple-scattering term. The `Fresnel` scale is
//!    `F_ms = F_avg²·E_avg / (1 - F_avg·(1 - E_avg))`; the multiple-scattering
//!    directional albedo is `F_ms·(1 - E_ss(mu))`. The matching bidirectional
//!    lobe `f_ms(mu_o, mu_i)` is [`kulla_conty_multiscatter_brdf`].
//!
//! The headline property, proven exactly here, is the **white-furnace energy
//! balance**: with `F0 = 1` the `Fresnel` average is `1`, the scale collapses to
//! `F_ms = 1`, and the single-plus-multiple directional albedo is
//! `E_ss + (1 - E_ss) = 1` for every `mu` and roughness. No chromatic absorption
//! is invented, and compensation vanishes smoothly as roughness → 0.
//!
//! No trigonometric, exponential, or power-function call is used anywhere: fixed
//! powers are integer multiplies and the only transcendental the determinism
//! lint would allow (`sqrt`) is not even needed, so the result is deterministic
//! and portable. `RGB` reflectance uses [`super::Vec3`]; scalars are `f32`.

use super::Vec3;

/// `pi` bound to the compile-time `core::f32::consts::PI` constant. This is a
/// constant reference, not a transcendental call (which the determinism lint
/// bans) and not a bare float literal (which would trip `clippy::approx_constant`).
/// It appears only in the bidirectional [`kulla_conty_multiscatter_brdf`]
/// normalization and cancels out of every directional-albedo result.
const PI: f32 = core::f32::consts::PI;

/// Roughness-independent floor of the single-scattering energy deficit per unit
/// linear roughness (the head-on, `mu = 1` loss slope). Chosen so the `alpha =
/// 1` furnace deficit sits in the physically observed `GGX` range (`≈ 0.35`
/// head-on) without a tabulated `LUT`.
const DEFICIT_BASE: f32 = 0.35;

/// Extra energy deficit weighting toward grazing angles (multiplying
/// `(1 - mu)²`). Together with [`DEFICIT_BASE`] it caps the worst-case deficit
/// at `0.60` (`mu = 0`, `alpha = 1`), keeping `E_ss` strictly inside `(0, 1]`.
const DEFICIT_GRAZING: f32 = 0.25;

/// Smallest denominator tolerated before a ratio is treated as degenerate, so a
/// vanishing missing-energy term (`1 - E_avg → 0` as roughness → 0) never
/// divides by zero or yields a `NaN`.
const MIN_DENOM: f32 = 1e-6;

/// Clamps a scalar into `0..=1` without branching on floating-point equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Directional albedo `E_ss(mu, roughness)` of the single-scattering `GGX` lobe
/// under a white furnace (reflectance at normal incidence `F0 = 1`).
///
/// `mu` is the view cosine `NdotV` and `roughness` is the artist-facing
/// *perceptual* roughness; both are clamped to `0..=1`. Internally the fit uses
/// the *linear* roughness `alpha = perceptual²` (the `NDF` alpha, matching
/// [`super::specular_aa`]). The energy deficit `1 - E_ss` grows linearly with
/// `alpha` and quadratically toward grazing via `(1 - mu)²`:
///
/// `E_ss = 1 - alpha·(DEFICIT_BASE + DEFICIT_GRAZING·(1 - mu)²)`.
///
/// This is a pure rational fit in the Karis `EnvBRDFApprox` spirit (scale/bias
/// polynomials), evaluating the fixed square by an integer multiply rather than
/// `powf`/`powi`. At `roughness = 0` it returns `1` for every `mu` (a perfect
/// mirror reflects all energy); it increases monotonically with `mu` and
/// decreases monotonically with `roughness`, staying inside `[0.4, 1]`.
#[must_use]
pub fn single_scatter_directional_albedo(mu: f32, roughness: f32) -> f32 {
    let cos_theta = clamp01(mu);
    let perceptual = clamp01(roughness);
    // Linear roughness (the `NDF` alpha) is the square of perceptual roughness.
    let alpha = perceptual * perceptual;
    let grazing = 1.0 - cos_theta;
    // Fixed power of two by integer multiply (never `powf`/`powi`).
    let grazing_sq = grazing * grazing;
    let deficit = alpha * (DEFICIT_BASE + DEFICIT_GRAZING * grazing_sq);
    1.0 - deficit
}

/// Cosine-weighted hemispherical average albedo `E_avg(roughness)` of the
/// single-scattering lobe.
///
/// This is the *exact* closed form of `E_avg = 2∫₀¹ E_ss(mu)·mu dmu` for the
/// [`single_scatter_directional_albedo`] fit, which integrates to a polynomial
/// in the linear roughness `alpha`:
///
/// `E_avg = 1 - alpha·(DEFICIT_BASE + DEFICIT_GRAZING/6)`.
///
/// Deriving it analytically (rather than fitting a separate curve) keeps the
/// average perfectly consistent with the directional fit, which is what makes
/// the Kulla-Conty furnace balance exact. It returns `1` at `roughness = 0` and
/// decreases monotonically with roughness, staying inside `(0, 1]`.
#[must_use]
pub fn average_albedo(roughness: f32) -> f32 {
    let perceptual = clamp01(roughness);
    let alpha = perceptual * perceptual;
    // `DEFICIT_GRAZING / 6` is the `∫₀¹ (1 - mu)²·mu dmu = 1/12` contribution
    // doubled by the `2∫` cosine-weight normalization.
    1.0 - alpha * (DEFICIT_BASE + DEFICIT_GRAZING / 6.0)
}

/// Analytic average `Fresnel` `F_avg = F0 + (1 - F0)/21`.
///
/// This is the exact cosine-weighted hemispherical average of Schlick's
/// `Fresnel` `F0 + (1 - F0)·(1 - cos)^5` (the pointwise form lives in
/// [`super::fresnel_rim::fresnel_schlick`]; the `1/21` tail is `∫₀¹ (1 - c)^5·2c
/// dc`). `f0` is clamped to `0..=1`; the result lies in `[F0, 1]`, equals `1/21`
/// at `F0 = 0`, and reaches `1` at `F0 = 1`.
#[must_use]
pub fn average_fresnel(f0: f32) -> f32 {
    let base = clamp01(f0);
    // `(1 - F0)/21` is the closed-form Schlick tail integral.
    base + (1.0 - base) / 21.0
}

/// Per-channel [`average_fresnel`] for an `RGB` reflectance-at-normal-incidence
/// `F0` carried as a [`super::Vec3`] (common for coloured-metal particles).
#[must_use]
pub fn average_fresnel_rgb(f0: Vec3) -> Vec3 {
    Vec3::new(
        average_fresnel(f0.x),
        average_fresnel(f0.y),
        average_fresnel(f0.z),
    )
}

/// Kulla-Conty multiple-scattering `Fresnel` scale
/// `F_ms = F_avg²·E_avg / (1 - F_avg·(1 - E_avg))`.
///
/// It accounts for the extra `Fresnel` reflectance accrued over the (geometric)
/// series of inter-microfacet bounces. Both inputs are clamped to `0..=1` and
/// the denominator is floored at [`MIN_DENOM`]. With a white furnace
/// (`F_avg = 1`) it collapses to exactly `1` for any `E_avg > 0`, which is the
/// algebraic root of the energy-conservation guarantee.
#[must_use]
pub fn multiscatter_fresnel_scale(e_avg: f32, f_avg: f32) -> f32 {
    let energy = clamp01(e_avg);
    let fresnel = clamp01(f_avg);
    let numerator = fresnel * fresnel * energy;
    let denominator = 1.0 - fresnel * (1.0 - energy);
    numerator / denominator.max(MIN_DENOM)
}

/// Directional albedo of the Kulla-Conty multiple-scattering term,
/// `F_ms·(1 - E_ss(mu, roughness))`.
///
/// This is the energy that single scattering dropped, scaled by the
/// multiple-scattering `Fresnel` [`multiscatter_fresnel_scale`]. It is
/// non-negative, vanishes as `roughness → 0` (where `1 - E_ss → 0`), and —
/// because the linear growth of the missing energy dominates the mild decay of
/// `F_ms` — increases monotonically with roughness. Adding it to the single
/// scattering reflectance restores a white furnace to unit energy.
#[must_use]
pub fn multiscatter_directional_albedo(mu: f32, roughness: f32, f0: f32) -> f32 {
    let e_ss = single_scatter_directional_albedo(mu, roughness);
    let e_avg = average_albedo(roughness);
    let f_avg = average_fresnel(f0);
    let scale = multiscatter_fresnel_scale(e_avg, f_avg);
    scale * (1.0 - e_ss)
}

/// The bidirectional Kulla-Conty multiple-scattering lobe
/// `f_ms(mu_o, mu_i) = F_ms·(1 - E_ss(mu_o))·(1 - E_ss(mu_i)) / (π·(1 - E_avg))`.
///
/// `mu_out`/`mu_in` are the outgoing/incoming view-and-light cosines. The lobe
/// is a separable, energy-redistributing complement to the single-scattering
/// `GGX` lobe; integrating it against the incoming cosine over the hemisphere
/// recovers [`multiscatter_directional_albedo`] exactly (because
/// [`average_albedo`] is the true integral of the `E_ss` fit). As `roughness →
/// 0` the missing-energy factor `1 - E_avg → 0`, so the lobe is clamped to `0`
/// to avoid a degenerate `0/0`.
///
/// This matches the design §17 "多散近似" closure; the Imageworks `F_ms` form
/// used here is the energy-exact variant of the paraphrased spec formula and
/// coincides with it in the white-furnace limit (`F_avg = 1 ⇒ F_ms = 1`).
#[must_use]
pub fn kulla_conty_multiscatter_brdf(mu_out: f32, mu_in: f32, roughness: f32, f0: f32) -> f32 {
    let e_avg = average_albedo(roughness);
    let one_minus_e_avg = 1.0 - e_avg;
    if one_minus_e_avg < MIN_DENOM {
        // Roughness → 0: no missing energy to redistribute.
        return 0.0;
    }
    let e_ss_out = single_scatter_directional_albedo(mu_out, roughness);
    let e_ss_in = single_scatter_directional_albedo(mu_in, roughness);
    let f_avg = average_fresnel(f0);
    let scale = multiscatter_fresnel_scale(e_avg, f_avg);
    scale * (1.0 - e_ss_out) * (1.0 - e_ss_in) / (PI * one_minus_e_avg)
}

/// Combines single scattering with the multiple-scattering compensation into
/// one specular directional albedo.
///
/// `single_scatter_albedo` is the caller's `Fresnel`-weighted single-scattering
/// directional albedo (for example the split-sum `F0·scale + bias` of the `GGX`
/// lobe, or an `IBL` prefiltered specular sample). The compensation
/// [`multiscatter_directional_albedo`] is added on top. For a white furnace
/// (`F0 = 1`, `single_scatter_albedo = E_ss`) the result is exactly `1`.
#[must_use]
pub fn compensated_specular_albedo(
    single_scatter_albedo: f32,
    mu: f32,
    roughness: f32,
    f0: f32,
) -> f32 {
    single_scatter_albedo + multiscatter_directional_albedo(mu, roughness, f0)
}

/// Per-channel [`compensated_specular_albedo`] for `RGB` reflectance carried as
/// [`super::Vec3`] (coloured metals / tinted energy particles).
#[must_use]
pub fn compensated_specular_albedo_rgb(
    single_scatter_albedo: Vec3,
    mu: f32,
    roughness: f32,
    f0: Vec3,
) -> Vec3 {
    Vec3::new(
        compensated_specular_albedo(single_scatter_albedo.x, mu, roughness, f0.x),
        compensated_specular_albedo(single_scatter_albedo.y, mu, roughness, f0.y),
        compensated_specular_albedo(single_scatter_albedo.z, mu, roughness, f0.z),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        average_albedo, average_fresnel, average_fresnel_rgb, compensated_specular_albedo,
        compensated_specular_albedo_rgb, kulla_conty_multiscatter_brdf,
        multiscatter_directional_albedo, multiscatter_fresnel_scale,
        single_scatter_directional_albedo, Vec3, PI,
    };

    /// Comparison tolerance for the analytic (closed-form) relations.
    const CMP_EPS: f32 = 1e-6;
    /// Looser tolerance for the numerically integrated furnace check.
    const QUAD_EPS: f32 = 1e-3;

    #[must_use]
    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() <= CMP_EPS
    }

    #[test]
    fn e_ss_is_unity_at_zero_roughness() {
        for &mu in &[0.0_f32, 0.1, 0.5, 0.9, 1.0] {
            assert!(approx_eq(single_scatter_directional_albedo(mu, 0.0), 1.0));
        }
    }

    #[test]
    fn e_ss_stays_inside_unit_range() {
        for &mu in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            for &r in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                let e = single_scatter_directional_albedo(mu, r);
                assert!(e > 0.0);
                assert!(e <= 1.0 + CMP_EPS);
            }
        }
    }

    #[test]
    fn e_ss_increases_toward_head_on() {
        // Fixed roughness: more energy survives as the view approaches head-on.
        let r = 0.8;
        let grazing = single_scatter_directional_albedo(0.1, r);
        let mid = single_scatter_directional_albedo(0.5, r);
        let head_on = single_scatter_directional_albedo(1.0, r);
        assert!(grazing < mid);
        assert!(mid < head_on);
    }

    #[test]
    fn e_ss_decreases_with_roughness() {
        let mu = 0.6;
        let smooth = single_scatter_directional_albedo(mu, 0.2);
        let rough = single_scatter_directional_albedo(mu, 0.6);
        let roughest = single_scatter_directional_albedo(mu, 1.0);
        assert!(rough < smooth);
        assert!(roughest < rough);
    }

    #[test]
    fn e_ss_clamps_out_of_range_inputs() {
        // Over-unity cosine clamps to head-on; negative clamps to grazing.
        assert!(approx_eq(
            single_scatter_directional_albedo(2.0, 0.5),
            single_scatter_directional_albedo(1.0, 0.5),
        ));
        assert!(approx_eq(
            single_scatter_directional_albedo(-1.0, 0.5),
            single_scatter_directional_albedo(0.0, 0.5),
        ));
        // Over-unity roughness clamps to alpha = 1.
        assert!(approx_eq(
            single_scatter_directional_albedo(0.5, 5.0),
            single_scatter_directional_albedo(0.5, 1.0),
        ));
    }

    #[test]
    fn e_avg_is_unity_at_zero_roughness_and_decreases() {
        assert!(approx_eq(average_albedo(0.0), 1.0));
        let smooth = average_albedo(0.3);
        let rough = average_albedo(0.7);
        let roughest = average_albedo(1.0);
        assert!(smooth < 1.0);
        assert!(rough < smooth);
        assert!(roughest < rough);
        assert!(roughest > 0.0);
    }

    #[test]
    fn e_avg_matches_the_quadrature_of_e_ss() {
        // The closed form must equal the cosine-weighted hemispherical integral
        // 2 * ∫₀¹ E_ss(mu) * mu dmu of the directional fit for every roughness.
        let samples = 8192_u32;
        for &r in &[0.1_f32, 0.4, 0.7, 1.0] {
            let mut acc = 0.0;
            for i in 0..samples {
                let mu = (i as f32 + 0.5) / samples as f32;
                acc += single_scatter_directional_albedo(mu, r) * mu;
            }
            let integral = acc / samples as f32;
            let e_avg_quadrature = 2.0 * integral;
            assert!((e_avg_quadrature - average_albedo(r)).abs() <= QUAD_EPS);
        }
    }

    #[test]
    fn average_fresnel_endpoints_and_bounds() {
        // F0 = 0 -> 1/21; F0 = 1 -> 1; result always within [F0, 1].
        assert!(approx_eq(average_fresnel(0.0), 1.0 / 21.0));
        assert!(approx_eq(average_fresnel(1.0), 1.0));
        for &f0 in &[0.0_f32, 0.04, 0.2, 0.5, 0.9, 1.0] {
            let f_avg = average_fresnel(f0);
            assert!(f_avg >= f0 - CMP_EPS);
            assert!(f_avg <= 1.0 + CMP_EPS);
        }
    }

    #[test]
    fn average_fresnel_is_monotone_in_f0() {
        let low = average_fresnel(0.04);
        let mid = average_fresnel(0.3);
        let high = average_fresnel(0.8);
        assert!(low < mid);
        assert!(mid < high);
    }

    #[test]
    fn average_fresnel_rgb_is_per_channel() {
        let f0 = Vec3::new(0.04, 0.3, 0.9);
        let avg = average_fresnel_rgb(f0);
        assert!(approx_eq(avg.x, average_fresnel(0.04)));
        assert!(approx_eq(avg.y, average_fresnel(0.3)));
        assert!(approx_eq(avg.z, average_fresnel(0.9)));
    }

    #[test]
    fn multiscatter_fresnel_scale_is_unity_for_white_furnace() {
        // F_avg = 1 => F_ms = 1 for any E_avg > 0; this is the furnace root.
        for &e_avg in &[0.6_f32, 0.75, 0.9, 1.0] {
            assert!(approx_eq(multiscatter_fresnel_scale(e_avg, 1.0), 1.0));
        }
    }

    #[test]
    fn compensation_vanishes_at_zero_roughness() {
        for &mu in &[0.0_f32, 0.5, 1.0] {
            for &f0 in &[0.04_f32, 0.5, 1.0] {
                assert!(approx_eq(multiscatter_directional_albedo(mu, 0.0, f0), 0.0));
            }
        }
    }

    #[test]
    fn compensation_is_non_negative() {
        for &mu in &[0.05_f32, 0.4, 1.0] {
            for &r in &[0.0_f32, 0.3, 0.6, 1.0] {
                for &f0 in &[0.0_f32, 0.04, 0.5, 1.0] {
                    assert!(multiscatter_directional_albedo(mu, r, f0) >= 0.0);
                }
            }
        }
    }

    #[test]
    fn compensation_increases_with_roughness() {
        let mu = 0.7;
        for &f0 in &[0.04_f32, 0.5, 1.0] {
            let mut previous = -1.0_f32;
            for &r in &[0.0_f32, 0.2, 0.4, 0.6, 0.8, 1.0] {
                let ms = multiscatter_directional_albedo(mu, r, f0);
                assert!(ms > previous);
                previous = ms;
            }
        }
    }

    #[test]
    fn white_furnace_conserves_energy() {
        // The gold standard: single + multiple scattering returns unit energy
        // for a white furnace (F0 = 1), across view angle and roughness.
        for &mu in &[0.05_f32, 0.25, 0.5, 0.75, 1.0] {
            for &r in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                let e_ss = single_scatter_directional_albedo(mu, r);
                let total = compensated_specular_albedo(e_ss, mu, r, 1.0);
                assert!(approx_eq(total, 1.0));
            }
        }
    }

    #[test]
    fn high_roughness_white_furnace_is_unity() {
        // The hardest furnace case (roughest, grazing) still conserves energy.
        let mu = 0.08;
        let r = 1.0;
        let e_ss = single_scatter_directional_albedo(mu, r);
        assert!(e_ss < 1.0); // energy really was lost by single scattering
        let total = compensated_specular_albedo(e_ss, mu, r, 1.0);
        assert!(approx_eq(total, 1.0));
    }

    #[test]
    fn compensated_albedo_adds_non_negative_energy() {
        // For any coloured F0, compensation never removes energy.
        let single = 0.3;
        for &r in &[0.2_f32, 0.6, 1.0] {
            let total = compensated_specular_albedo(single, 0.6, r, 0.5);
            assert!(total >= single - CMP_EPS);
        }
    }

    #[test]
    fn bidirectional_lobe_integrates_to_directional_albedo() {
        // ∫ f_ms(mu_o, mu_i) * mu_i dω_i over the hemisphere, reduced to the 1D
        // cosine-weighted integral 2π * ∫₀¹ f_ms * mu dmu (no trig needed), must
        // match the closed-form multiscatter directional albedo.
        let samples = 8192_u32;
        for &mu_out in &[0.2_f32, 0.6, 0.95] {
            for &f0 in &[0.04_f32, 0.5, 1.0] {
                let r = 0.75;
                let mut acc = 0.0;
                for i in 0..samples {
                    let mu = (i as f32 + 0.5) / samples as f32;
                    acc += kulla_conty_multiscatter_brdf(mu_out, mu, r, f0) * mu;
                }
                let integral = acc / samples as f32;
                let dir_albedo = 2.0 * PI * integral;
                let expected = multiscatter_directional_albedo(mu_out, r, f0);
                assert!((dir_albedo - expected).abs() <= QUAD_EPS);
            }
        }
    }

    #[test]
    fn bidirectional_lobe_is_zero_at_zero_roughness() {
        assert!(approx_eq(
            kulla_conty_multiscatter_brdf(0.5, 0.5, 0.0, 0.5),
            0.0,
        ));
    }

    #[test]
    fn rgb_compensation_matches_scalar_per_channel() {
        let single = Vec3::new(0.2, 0.35, 0.5);
        let f0 = Vec3::new(0.04, 0.5, 0.95);
        let mu = 0.55;
        let r = 0.6;
        let rgb = compensated_specular_albedo_rgb(single, mu, r, f0);
        assert!(approx_eq(
            rgb.x,
            compensated_specular_albedo(single.x, mu, r, f0.x),
        ));
        assert!(approx_eq(
            rgb.y,
            compensated_specular_albedo(single.y, mu, r, f0.y),
        ));
        assert!(approx_eq(
            rgb.z,
            compensated_specular_albedo(single.z, mu, r, f0.z),
        ));
    }

    #[test]
    fn evaluation_is_deterministic() {
        // Identical inputs must produce bit-identical outputs (no equality
        // comparison on raw floats: compare the bit patterns).
        let a = compensated_specular_albedo(0.42, 0.37, 0.63, 0.21);
        let b = compensated_specular_albedo(0.42, 0.37, 0.63, 0.21);
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
