//! Independent CPU oracle for the layered OpenPBR/über material response.
//!
//! This is a material-only reference.  It intentionally owns no realtime
//! evaluator, GPU/WESL code, texture lookup, resolve state, or frame-graph
//! state.  The equations are composed from the independent specification
//! transcription in [`crate::openpbr_reference`], so a production evaluator
//! cannot certify itself by calling this module.

use core::f32::consts::{FRAC_PI_2, PI};

use crate::openpbr_reference as spec;

const NORMAL: [f32; 3] = [0.0, 1.0, 0.0];
const MIN_LENGTH_SQUARED: f32 = 1.0e-12;

/// Parameters for the reference principled material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrincipledReference {
    /// Linear RGB base color.
    pub base_color: [f32; 3],
    /// Base-layer weight.
    pub base_weight: f32,
    /// Metallic interpolation between dielectric and conductor response.
    pub metallic: f32,
    /// Base dielectric relative IOR.
    pub specular_ior: f32,
    /// Base specular weight.
    pub specular_weight: f32,
    /// Base dielectric/conductor specular tint.
    pub specular_color: [f32; 3],
    /// Diffuse perceptual roughness.
    pub diffuse_roughness: f32,
    /// Base GGX perceptual roughness.
    pub specular_roughness: f32,
    /// Base GGX anisotropy.
    pub anisotropy: f32,
    /// Coat weight.
    pub coat_weight: f32,
    /// Coat relative IOR.
    pub coat_ior: f32,
    /// Coat two-way normal-incidence transmittance color.
    pub coat_color: [f32; 3],
    /// Coat perceptual roughness.
    pub coat_roughness: f32,
    /// OpenPBR coat darkening control.
    pub coat_darkening: f32,
}

impl Default for PrincipledReference {
    fn default() -> Self {
        Self {
            base_color: [1.0; 3],
            base_weight: 1.0,
            metallic: 0.0,
            specular_ior: 1.5,
            specular_weight: 1.0,
            specular_color: [1.0; 3],
            diffuse_roughness: 0.0,
            specular_roughness: 0.5,
            anisotropy: 0.0,
            coat_weight: 0.0,
            coat_ior: 1.5,
            coat_color: [1.0; 3],
            coat_roughness: 0.1,
            coat_darkening: 1.0,
        }
    }
}

/// Terms returned by one oracle evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayeredReferenceValue {
    /// Base BRDF before the coat acts on it.
    pub base_brdf: [f32; 3],
    /// Coat GGX BRDF contribution.
    pub coat_brdf: [f32; 3],
    /// Per-channel coat transmission, absorption, and darkening factor.
    pub base_factor: [f32; 3],
    /// Scalar OpenPBR darkening factor.
    pub darkening: f32,
    /// Base BRDF after layering.
    pub layered_base_brdf: [f32; 3],
    /// Final layered BRDF.
    pub total_brdf: [f32; 3],
}

impl PrincipledReference {
    /// Evaluates the reference BRDF for two local unit directions.
    ///
    /// The local normal is `+Y`; both directions point away from the surface.
    #[must_use]
    pub fn evaluate_brdf(&self, view: [f32; 3], light: [f32; 3]) -> [f32; 3] {
        self.evaluate_layers(view, light).total_brdf
    }

    /// Evaluates the reference and keeps the layer terms inspectable.
    #[must_use]
    pub fn evaluate_layers(&self, view: [f32; 3], light: [f32; 3]) -> LayeredReferenceValue {
        let view = normalize_or(view, NORMAL);
        let light = normalize_or(light, NORMAL);
        let n_dot_v = dot(NORMAL, view);
        let n_dot_l = dot(NORMAL, light);
        if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
            return LayeredReferenceValue::zero();
        }

        let half = normalize_or(add(view, light), NORMAL);
        let n_dot_h = dot(NORMAL, half).max(0.0);
        let v_dot_h = dot(view, half).clamp(0.0, 1.0);
        if n_dot_h <= 0.0 {
            return LayeredReferenceValue::zero();
        }

        let metallic = finite_clamp(self.metallic, 0.0, 1.0);
        let base_weight = finite_clamp(self.base_weight, 0.0, 1.0);
        let coat_weight = finite_clamp(self.coat_weight, 0.0, 1.0);
        let base_roughness = spec::coat_roughened_base_roughness(
            self.specular_roughness,
            self.coat_roughness,
            coat_weight,
        );
        let base_eta = spec::coat_specular_ior_ratio(
            self.specular_ior,
            self.coat_ior,
            coat_weight,
        );
        let dielectric_f = spec::base_dielectric_fresnel_rgb(
            v_dot_h,
            base_eta,
            self.specular_weight,
            self.specular_color,
        );
        let metal_f = spec::base_metal_fresnel_rgb(
            v_dot_h,
            base_weight,
            self.base_color,
            self.specular_weight,
            self.specular_color,
        );
        let fresnel = lerp3(dielectric_f, metal_f, metallic);
        let (alpha_t, alpha_b) = spec::anisotropy_to_alpha(base_roughness, self.anisotropy);
        let base_specular = spec::microfacet_brdf(
            fresnel,
            half[0],
            half[2],
            n_dot_h,
            view[0],
            view[2],
            light[0],
            light[2],
            n_dot_v,
            n_dot_l,
            alpha_t,
            alpha_b,
        );

        let sigma = finite_clamp(self.diffuse_roughness, 0.0, 1.0) * FRAC_PI_2;
        let s = dot(view, light) - n_dot_v * n_dot_l;
        let diffuse = diffuse_brdf(self.base_color, n_dot_v, n_dot_l, s, sigma);
        let diffuse = array_scale(diffuse, 1.0 - metallic);
        let base_brdf = array_scale(array_add(diffuse, base_specular), base_weight);

        let coat_ior = self.coat_ior.max(spec::MIN_IOR);
        let (coat_alpha_t, coat_alpha_b) = spec::anisotropy_to_alpha(self.coat_roughness, 0.0);
        let coat_f = coat_weight * spec::dielectric_fresnel(v_dot_h, coat_ior);
        let coat_brdf = spec::microfacet_brdf(
            [coat_f; 3],
            half[0],
            half[2],
            n_dot_h,
            view[0],
            view[2],
            light[0],
            light[2],
            n_dot_v,
            n_dot_l,
            coat_alpha_t,
            coat_alpha_b,
        );

        let view_f = spec::dielectric_fresnel(n_dot_v, coat_ior);
        let light_f = spec::dielectric_fresnel(n_dot_l, coat_ior);
        let slab_transmission = (1.0 - coat_weight * view_f).max(0.0)
            * (1.0 - coat_weight * light_f).max(0.0);
        let base_roughness_estimate = spec::base_roughness_estimate(
            self.specular_roughness,
            self.specular_weight,
            self.specular_ior,
            metallic,
        );
        let internal_reflection = spec::internal_reflection_general(
            coat_ior,
            n_dot_v,
            base_roughness_estimate,
        );
        let delta = spec::coat_darkening_delta(
            arithmetic_mean(self.base_color),
            internal_reflection,
        );
        let darkening = spec::modulated_darkening_factor(
            coat_weight,
            self.coat_darkening,
            delta,
        );
        let absorption = spec::coat_absorption_rgb(
            self.coat_color,
            n_dot_l,
            n_dot_v,
            coat_ior,
        );
        let base_factor = array_scale(absorption, slab_transmission * darkening).map(|value| {
            1.0 + coat_weight * (value - 1.0)
        });
        let layered_base_brdf = array_mul(base_brdf, base_factor);
        let total_brdf = array_add(layered_base_brdf, coat_brdf);

        LayeredReferenceValue {
            base_brdf,
            coat_brdf,
            base_factor,
            darkening,
            layered_base_brdf,
            total_brdf,
        }
    }

    /// Integrates the BRDF against a unit white environment.
    ///
    /// This deterministic midpoint rule is intended for regression gates and
    /// diagnostics, not for final image production.  `theta_samples` and
    /// `phi_samples` are explicit to make the cost and convergence visible.
    #[must_use]
    pub fn white_furnace(
        &self,
        view: [f32; 3],
        theta_samples: usize,
        phi_samples: usize,
    ) -> [f32; 3] {
        if theta_samples == 0 || phi_samples == 0 {
            return [0.0; 3];
        }
        let d_theta = FRAC_PI_2 / theta_samples as f32;
        let d_phi = 2.0 * PI / phi_samples as f32;
        let mut result = [0.0; 3];
        for theta_index in 0..theta_samples {
            let theta = (theta_index as f32 + 0.5) * d_theta;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            for phi_index in 0..phi_samples {
                let phi = (phi_index as f32 + 0.5) * d_phi;
                let light = [
                    sin_theta * phi.cos(),
                    cos_theta,
                    sin_theta * phi.sin(),
                ];
                let brdf = self.evaluate_brdf(view, light);
                let measure = sin_theta * d_theta * d_phi;
                for channel in 0..3 {
                    result[channel] += brdf[channel] * cos_theta * measure;
                }
            }
        }
        result
    }
}

impl LayeredReferenceValue {
    fn zero() -> Self {
        Self {
            base_brdf: [0.0; 3],
            coat_brdf: [0.0; 3],
            base_factor: [0.0; 3],
            darkening: 1.0,
            layered_base_brdf: [0.0; 3],
            total_brdf: [0.0; 3],
        }
    }
}

fn diffuse_brdf(
    base_color: [f32; 3],
    cos_o: f32,
    cos_i: f32,
    s: f32,
    sigma: f32,
) -> [f32; 3] {
    let e_i = spec::eon_directional_albedo(cos_i, sigma);
    let e_o = spec::eon_directional_albedo(cos_o, sigma);
    let mut result = [0.0; 3];
    for channel in 0..3 {
        let rho = finite_clamp(base_color[channel], 0.0, 1.0);
        let rho_ms = spec::eon_rho_ms(rho, sigma);
        let compensation = (rho_ms / PI) * (1.0 - e_i).max(0.0) * (1.0 - e_o).max(0.0);
        result[channel] = rho * spec::fon_unit_brdf(cos_i, cos_o, s, sigma) + compensation;
    }
    result
}

fn finite_clamp(value: f32, low: f32, high: f32) -> f32 {
    if value.is_finite() {
        value.clamp(low, high)
    } else {
        low
    }
}

fn arithmetic_mean(value: [f32; 3]) -> f32 {
    ((value[0] + value[1] + value[2]) / 3.0).clamp(0.0, 1.0)
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length_squared = dot(value, value);
    if !length_squared.is_finite() || length_squared <= MIN_LENGTH_SQUARED {
        return fallback;
    }
    let inverse_length = length_squared.sqrt().recip();
    [
        value[0] * inverse_length,
        value[1] * inverse_length,
        value[2] * inverse_length,
    ]
}

fn lerp3(a: [f32; 3], b: [f32; 3], weight: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * weight,
        a[1] + (b[1] - a[1]) * weight,
        a[2] + (b[2] - a[2]) * weight,
    ]
}

fn array_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn array_mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn array_scale(a: [f32; 3], scale: f32) -> [f32; 3] {
    [a[0] * scale, a[1] * scale, a[2] * scale]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view_at(cosine: f32) -> [f32; 3] {
        let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
        [sine, cosine, 0.0]
    }

    #[test]
    fn zero_coat_is_an_exact_layer_identity() {
        let material = PrincipledReference {
            base_color: [0.34, 0.57, 0.81],
            metallic: 0.25,
            specular_roughness: 0.37,
            diffuse_roughness: 0.42,
            anisotropy: 0.3,
            coat_weight: 0.0,
            ..Default::default()
        };
        let value = material.evaluate_layers(view_at(0.72), view_at(0.61));
        assert_eq!(value.base_factor, [1.0; 3]);
        assert_eq!(value.darkening, 1.0);
        assert_eq!(value.coat_brdf, [0.0; 3]);
        assert_eq!(value.layered_base_brdf, value.base_brdf);
        assert_eq!(value.total_brdf, value.base_brdf);
    }

    #[test]
    fn coat_darkening_only_reduces_the_base_term() {
        let common = PrincipledReference {
            base_color: [0.8, 0.45, 0.2],
            metallic: 0.0,
            specular_roughness: 0.28,
            diffuse_roughness: 0.25,
            coat_weight: 1.0,
            coat_roughness: 0.22,
            coat_color: [0.65, 0.9, 0.4],
            ..Default::default()
        };
        let no_darkening = PrincipledReference {
            coat_darkening: 0.0,
            ..common
        }
        .evaluate_layers(view_at(0.58), view_at(0.77));
        let darkened = PrincipledReference {
            coat_darkening: 1.0,
            ..common
        }
        .evaluate_layers(view_at(0.58), view_at(0.77));
        assert!(darkened.darkening < 1.0);
        assert_eq!(darkened.coat_brdf, no_darkening.coat_brdf);
        for channel in 0..3 {
            assert!(darkened.layered_base_brdf[channel] <= no_darkening.layered_base_brdf[channel]);
        }
    }

    #[test]
    fn layered_brdf_is_reciprocal_for_symmetric_terms() {
        let material = PrincipledReference {
            base_color: [0.7, 0.3, 0.9],
            metallic: 0.35,
            specular_roughness: 0.42,
            diffuse_roughness: 0.18,
            coat_weight: 0.8,
            coat_roughness: 0.3,
            coat_color: [0.9, 0.8, 0.7],
            ..Default::default()
        };
        let a = material.evaluate_brdf(view_at(0.81), view_at(0.63));
        let b = material.evaluate_brdf(view_at(0.63), view_at(0.81));
        for channel in 0..3 {
            assert!((a[channel] - b[channel]).abs() < 2.0e-5);
        }
    }

    #[test]
    fn white_furnace_stays_within_the_energy_budget() {
        let material = PrincipledReference {
            base_color: [0.6, 0.8, 1.0],
            metallic: 0.0,
            diffuse_roughness: 0.8,
            specular_roughness: 0.7,
            coat_weight: 1.0,
            coat_roughness: 0.65,
            coat_darkening: 1.0,
            ..Default::default()
        };
        for view in [[0.0, 1.0, 0.0], view_at(0.55), view_at(0.2)] {
            let albedo = material.white_furnace(view, 16, 32);
            for (channel, value) in albedo.into_iter().enumerate() {
                assert!(value.is_finite(), "channel {channel}: {value}");
                assert!(value <= 1.02, "channel {channel}: {value}");
                assert!(value >= -1.0e-5, "channel {channel}: {value}");
            }
        }
    }

    #[test]
    fn non_finite_inputs_degrade_to_finite_reference_values() {
        let material = PrincipledReference {
            base_color: [f32::NAN, f32::INFINITY, -f32::INFINITY],
            metallic: f32::NAN,
            coat_weight: f32::INFINITY,
            ..Default::default()
        };
        let value = material.evaluate_brdf([f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(value.into_iter().all(|channel| channel.is_finite() && channel >= 0.0));
    }
}
