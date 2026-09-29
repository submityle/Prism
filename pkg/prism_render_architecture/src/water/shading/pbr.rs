//! Physically based water lighting response (design §5.1).
//!
//! Computes the `Schlick` `Fresnel` reflectance from the water `IOR`, selects
//! the `SSR` -> `RT` -> probe reflection tier, derives the micro-surface foam
//! mask from the surface `Jacobian`, roughens the `GGX` response under foam, and
//! evaluates the grazing subsurface back-transmission (浪尖背光透光). Every
//! function is pure and deterministic.

use super::super::EPS;
use super::{PbrResponse, PbrShadingParams, ReflectionTier, SurfaceShadingInputs};

/// Base reflectance `F0` from a dielectric index of refraction.
///
/// `F0 = ((ior - 1) / (ior + 1))^2`. For water (`ior ≈ 1.33`) this is `≈ 0.02`,
/// the canonical low-`F0` water reflectance.
#[must_use]
pub fn f0_from_ior(ior: f32) -> f32 {
    let n = (ior - 1.0) / (ior + 1.0);
    n * n
}

/// `Schlick`'s `Fresnel` approximation.
///
/// `F(cos) = F0 + (1 - F0) * (1 - cos)^5`. The result is bounded in
/// `[F0, 1]`, equals `F0` looking straight down (`cos = 1`), and rises to `1`
/// at grazing (`cos = 0`), decreasing monotonically as `cos` increases.
#[must_use]
pub fn fresnel_schlick(f0: f32, cos: f32) -> f32 {
    let c = cos.clamp(0.0, 1.0);
    let one_minus = 1.0 - c;
    let p2 = one_minus * one_minus;
    let p5 = p2 * p2 * one_minus;
    f0 + (1.0 - f0) * p5
}

/// Resolve the [`PbrResponse`] for one surface sample.
#[must_use]
pub fn plan_pbr(params: PbrShadingParams, inputs: SurfaceShadingInputs) -> PbrResponse {
    let f0 = if params.f0_override > EPS {
        params.f0_override
    } else {
        f0_from_ior(inputs.ior)
    };
    let cos_view = inputs.cos_view.clamp(0.0, 1.0);
    let fresnel = fresnel_schlick(f0, cos_view);

    let reflection_tier = if inputs.ssr_confidence >= params.ssr_min_confidence {
        ReflectionTier::ScreenSpace
    } else if inputs.ray_budget >= params.rt_min_budget {
        ReflectionTier::RayTraced
    } else {
        ReflectionTier::Probe
    };

    let foam_mask = if params.foam_fold_threshold > EPS {
        ((params.foam_fold_threshold - inputs.jacobian) / params.foam_fold_threshold)
            .clamp(0.0, 1.0)
    } else {
        0.0
    };

    let base_roughness = params.base_roughness.clamp(0.0, 1.0);
    let specular_roughness = (base_roughness + foam_mask * (1.0 - base_roughness)).clamp(0.0, 1.0);
    let subsurface_transmission = (params.grazing_transmission * (1.0 - cos_view)).clamp(0.0, 1.0);

    PbrResponse {
        f0,
        fresnel,
        reflection_tier,
        foam_mask,
        specular_roughness,
        subsurface_transmission,
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::underwater::RgbColor;
    use super::*;

    fn fixture_params() -> PbrShadingParams {
        PbrShadingParams {
            f0_override: 0.0,
            foam_fold_threshold: 1.0,
            base_roughness: 0.08,
            grazing_transmission: 0.6,
            ssr_min_confidence: 0.5,
            rt_min_budget: 0.25,
        }
    }

    fn fixture_inputs() -> SurfaceShadingInputs {
        SurfaceShadingInputs {
            cos_view: 0.5,
            ior: 1.33,
            jacobian: 0.4,
            water_color: RgbColor {
                r: 0.1,
                g: 0.3,
                b: 0.5,
            },
            specular_intensity: 0.8,
            caustic_intensity: 0.6,
            flow_speed: 1.2,
            depth: 1.0,
            dist_to_shore: 1.0,
            ssr_confidence: 0.7,
            ray_budget: 0.5,
        }
    }

    #[test]
    fn f0_of_water_is_about_two_percent() {
        let f0 = f0_from_ior(1.33);
        assert!((f0 - 0.02).abs() < 5e-3, "water F0 {f0} not near 0.02");
    }

    #[test]
    fn fresnel_equals_f0_at_normal_incidence() {
        let f0 = 0.02;
        assert!((fresnel_schlick(f0, 1.0) - f0).abs() < EPS);
    }

    #[test]
    fn fresnel_reaches_one_at_grazing() {
        assert!((fresnel_schlick(0.02, 0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn fresnel_is_bounded_and_monotonic_in_cos() {
        let f0 = 0.02;
        let mut prev = fresnel_schlick(f0, 0.0);
        let mut cos = 0.05;
        while cos <= 1.0 + EPS {
            let cur = fresnel_schlick(f0, cos);
            assert!(
                (f0 - EPS..=1.0 + EPS).contains(&cur),
                "fresnel {cur} out of range"
            );
            assert!(cur <= prev + EPS, "fresnel rose with cos at {cos}");
            prev = cur;
            cos += 0.05;
        }
    }

    #[test]
    fn reflection_tier_falls_back_ssr_rt_probe() {
        let params = fixture_params();
        let mut inputs = fixture_inputs();

        inputs.ssr_confidence = 0.9;
        inputs.ray_budget = 0.9;
        assert_eq!(
            plan_pbr(params, inputs).reflection_tier,
            ReflectionTier::ScreenSpace
        );

        inputs.ssr_confidence = 0.1;
        assert_eq!(
            plan_pbr(params, inputs).reflection_tier,
            ReflectionTier::RayTraced
        );

        inputs.ray_budget = 0.1;
        assert_eq!(
            plan_pbr(params, inputs).reflection_tier,
            ReflectionTier::Probe
        );
    }

    #[test]
    fn foam_mask_rises_as_jacobian_folds() {
        let params = fixture_params();
        let mut inputs = fixture_inputs();
        inputs.jacobian = 1.0;
        let flat = plan_pbr(params, inputs).foam_mask;
        inputs.jacobian = 0.2;
        let compressed = plan_pbr(params, inputs).foam_mask;
        inputs.jacobian = 0.0;
        let folded = plan_pbr(params, inputs).foam_mask;
        assert!(flat.abs() < EPS, "flat surface should carry no foam");
        assert!(compressed > flat && folded > compressed);
        assert!((0.0..=1.0).contains(&folded));
    }

    #[test]
    fn foam_roughens_the_surface() {
        let params = fixture_params();
        let mut inputs = fixture_inputs();
        inputs.jacobian = 1.0;
        let calm = plan_pbr(params, inputs).specular_roughness;
        inputs.jacobian = 0.0;
        let foamy = plan_pbr(params, inputs).specular_roughness;
        assert!(foamy > calm, "foam should raise roughness");
        assert!((0.0..=1.0).contains(&foamy));
    }

    #[test]
    fn subsurface_transmission_peaks_at_grazing() {
        let params = fixture_params();
        let mut inputs = fixture_inputs();
        inputs.cos_view = 1.0;
        let head_on = plan_pbr(params, inputs).subsurface_transmission;
        inputs.cos_view = 0.0;
        let grazing = plan_pbr(params, inputs).subsurface_transmission;
        assert!(head_on.abs() < EPS);
        assert!(grazing > head_on);
        assert!((0.0..=1.0).contains(&grazing));
    }

    #[test]
    fn plan_pbr_is_deterministic() {
        let params = fixture_params();
        let inputs = fixture_inputs();
        assert_eq!(plan_pbr(params, inputs), plan_pbr(params, inputs));
    }
}
