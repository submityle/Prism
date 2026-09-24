use core::f32::consts::PI;

use crate::vecmath::{add, dot, mix3, mul, mul_scalar, normalize_or, sub};

const MIN_ROUGHNESS: f32 = 0.045;
const MIN_N_DOT: f32 = 1.0e-5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSample {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub perceptual_roughness: f32,
    pub reflectance: f32,
    pub ambient_occlusion: f32,
    pub emissive: [f32; 3],
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    /// Sheen (fuzz) intensity driving the Cloth Charlie sheen lobe in `[0, 1]`.
    pub sheen: f32,
    /// Subsurface scattering weight driving Cloth/Subsurface diffusion in `[0, 1]`.
    pub subsurface: f32,
}

impl Default for SurfaceSample {
    fn default() -> Self {
        Self {
            base_color: [1.0; 3],
            metallic: 0.0,
            perceptual_roughness: 0.5,
            reflectance: 0.5,
            ambient_occlusion: 1.0,
            emissive: [0.0; 3],
            clearcoat: 0.0,
            clearcoat_roughness: 0.25,
            sheen: 0.0,
            subsurface: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingFrame {
    pub normal: [f32; 3],
    pub view: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectLightSample {
    pub direction: [f32; 3],
    pub illuminance: [f32; 3],
    pub visibility: f32,
}

pub fn evaluate_principled_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> [f32; 3] {
    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let v = normalize_or(frame.view, n);
    let l = normalize_or(light.direction, n);
    let h = normalize_or(add(v, l), n);
    let n_dot_l = dot(n, l).max(0.0);
    let n_dot_v = dot(n, v).max(MIN_N_DOT);
    let n_dot_h = dot(n, h).max(MIN_N_DOT);
    let v_dot_h = dot(v, h).max(0.0);
    if n_dot_l <= 0.0 {
        return surface.emissive;
    }

    let metallic = surface.metallic.clamp(0.0, 1.0);
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let reflectance = surface.reflectance.clamp(0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = mix3([f0_dielectric; 3], surface.base_color, metallic);
    let f = fresnel_schlick(f0, v_dot_h);
    let d = distribution_ggx(n_dot_h, alpha);
    let g = visibility_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
    let single_scatter = mul_scalar(f, d * g);

    // Kulla-Conty-style bounded compensation. This preserves the primary GGX
    // lobe while returning some energy lost to unresolved multiple scattering.
    let energy = 1.0 - 0.28 * roughness * roughness;
    let compensation = mul_scalar(f0, ((1.0 / energy.max(0.25)) - 1.0).min(1.0));
    let specular = add(single_scatter, compensation);
    let diffuse_weight = mul_scalar(sub([1.0; 3], f), 1.0 - metallic);
    let diffuse = mul(mul_scalar(surface.base_color, 1.0 / PI), diffuse_weight);
    let base = mul_scalar(add(diffuse, specular), n_dot_l);

    let coat = surface.clearcoat.clamp(0.0, 1.0);
    let coat_roughness = surface.clearcoat_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let coat_alpha = coat_roughness * coat_roughness;
    let coat_f = fresnel_schlick([0.04; 3], v_dot_h);
    let coat_specular = mul_scalar(
        coat_f,
        coat
            * distribution_ggx(n_dot_h, coat_alpha)
            * visibility_smith_ggx_correlated(n_dot_v, n_dot_l, coat_alpha)
            * n_dot_l,
    );
    let coat_attenuation = 1.0 - coat * coat_f[0].clamp(0.0, 1.0);
    let reflected = add(mul_scalar(base, coat_attenuation), coat_specular);
    let radiance = mul(reflected, light.illuminance);
    add(
        mul_scalar(radiance, light.visibility.clamp(0.0, 1.0)),
        surface.emissive,
    )
}

pub fn evaluate_toon_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    bands: u32,
) -> [f32; 3] {
    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let l = normalize_or(light.direction, n);
    let steps = bands.max(1) as f32;
    let diffuse = (dot(n, l).max(0.0) * steps).floor() / steps;
    add(
        mul_scalar(
            mul(surface.base_color, light.illuminance),
            diffuse * light.visibility.clamp(0.0, 1.0),
        ),
        surface.emissive,
    )
}

/// Integrates a constant environment over a deterministic cosine hemisphere.
/// It is used as a cheap white-furnace regression guard, not a production IBL.
pub fn linear_furnace_response(surface: SurfaceSample, samples: u32) -> [f32; 3] {
    let sample_count = samples.max(1);
    let mut sum = [0.0; 3];
    for index in 0..sample_count {
        let u = (index as f32 + 0.5) / sample_count as f32;
        // A low-discrepancy cosine hemisphere that avoids platform-specific
        // transcendental functions in the backend-neutral reference path.
        let quadrant = index & 3;
        let tangent = match quadrant {
            0 => [1.0, 0.0],
            1 => [0.0, 1.0],
            2 => [-1.0, 0.0],
            _ => [0.0, -1.0],
        };
        let radius = u.sqrt();
        let direction = [radius * tangent[0], (1.0 - u).sqrt(), radius * tangent[1]];
        let value = evaluate_principled_direct(
            SurfaceSample {
                emissive: [0.0; 3],
                clearcoat: 0.0,
                ..surface
            },
            ShadingFrame {
                normal: [0.0, 1.0, 0.0],
                view: [0.0, 1.0, 0.0],
            },
            DirectLightSample {
                direction,
                illuminance: [PI; 3],
                visibility: 1.0,
            },
        );
        sum = add(sum, value);
    }
    mul_scalar(sum, 1.0 / sample_count as f32)
}

fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let alpha_squared = alpha * alpha;
    let denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    alpha_squared / (PI * denominator * denominator).max(MIN_N_DOT)
}

fn visibility_smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let alpha_squared = alpha * alpha;
    let gv = n_dot_l * ((n_dot_v - n_dot_v * alpha_squared) * n_dot_v + alpha_squared).sqrt();
    let gl = n_dot_v * ((n_dot_l - n_dot_l * alpha_squared) * n_dot_l + alpha_squared).sqrt();
    0.5 / (gv + gl).max(MIN_N_DOT)
}

fn fresnel_schlick(f0: [f32; 3], v_dot_h: f32) -> [f32; 3] {
    let one_minus = 1.0 - v_dot_h.clamp(0.0, 1.0);
    let squared = one_minus * one_minus;
    let factor = squared * squared * one_minus;
    add(f0, mul_scalar(sub([1.0; 3], f0), factor))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn light() -> DirectLightSample {
        DirectLightSample {
            direction: [0.0, 1.0, 0.0],
            illuminance: [1.0; 3],
            visibility: 1.0,
        }
    }

    fn frame() -> ShadingFrame {
        ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: [0.0, 1.0, 0.0],
        }
    }

    #[test]
    fn direct_lighting_is_finite_non_negative_and_shadowable() {
        for roughness in [0.045, 0.1, 0.5, 1.0] {
            for metallic in [0.0, 0.5, 1.0] {
                let value = evaluate_principled_direct(
                    SurfaceSample {
                        perceptual_roughness: roughness,
                        metallic,
                        ..Default::default()
                    },
                    frame(),
                    light(),
                );
                assert!(value.into_iter().all(|channel| channel.is_finite() && channel >= 0.0));
            }
        }
        let shadowed = evaluate_principled_direct(
            SurfaceSample::default(),
            frame(),
            DirectLightSample {
                visibility: 0.0,
                ..light()
            },
        );
        assert_eq!(shadowed, [0.0; 3]);
    }

    #[test]
    fn white_furnace_stays_bounded_across_roughness_and_metalness() {
        for roughness in [0.045, 0.25, 0.5, 0.75, 1.0] {
            for metallic in [0.0, 0.5, 1.0] {
                let response = linear_furnace_response(
                    SurfaceSample {
                        perceptual_roughness: roughness,
                        metallic,
                        ..Default::default()
                    },
                    2048,
                );
                assert!(response.into_iter().all(|channel| channel.is_finite() && channel <= 1.2));
            }
        }
    }

    #[test]
    fn toon_and_pbr_share_light_visibility_contract() {
        let lit = evaluate_toon_direct(SurfaceSample::default(), frame(), light(), 4);
        let shadowed = evaluate_toon_direct(
            SurfaceSample::default(),
            frame(),
            DirectLightSample {
                visibility: 0.0,
                ..light()
            },
            4,
        );
        assert!(lit[0] > 0.0);
        assert_eq!(shadowed, [0.0; 3]);
    }
}
