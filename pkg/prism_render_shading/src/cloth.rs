//! Cloth (fabric) BSDF direct-lighting lobe.
//!
//! This is the backend-neutral golden reference for the fabric shading class.
//! It is the byte-for-byte numerical twin of `cloth.wesl` (`cloth_direct`) so
//! the GPU resolve pass agrees with the CPU reference within tolerance.
//!
//! The model follows the physically based cloth closure popularised by
//! Filament / Unreal:
//!
//! * **Sheen specular** uses the Estevez-Kulla *Charlie* distribution together
//!   with the Ashikhmin/Neubelt visibility term.  The lobe is tinted by the
//!   `sheen` weight and deliberately carries **no Fresnel** term - fabric fuzz
//!   is a forward-scattering micro-fibre response, not a dielectric interface.
//! * **Diffuse** is Lambertian, optionally softened by an energy-conserving
//!   *wrap* term and tinted by a subsurface-color approximation when the
//!   `subsurface` weight is non-zero, giving the characteristic soft, waxy
//!   look of thick fabrics (velvet, felt).
//!
//! Emissive is folded in exactly once (added back after the per-light term and
//! returned directly when the surface faces away), matching the convention of
//! `evaluate_principled_direct` so the resolve integrators can accumulate over
//! many lights without multiplying self-illumination.

use core::f32::consts::PI;

use bevy_math::ops;

use crate::vecmath::{add, dot, mul, mul_scalar, normalize_or, saturate3};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Reciprocal of pi, matching the GPU `INV_PI` constant.
const INV_PI: f32 = 1.0 / PI;
/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;
/// Dot-product floor shared with the principled lobe (`MIN_N_DOT`).
const MIN_N_DOT: f32 = 1.0e-5;
/// `sin^2(theta_h)` floor for the Charlie distribution (1/128), preventing a
/// singular `pow` at grazing half-vectors. Mirrors Filament's `D_Charlie`.
const MIN_SIN2: f32 = 0.007_812_5;
/// Diffuse wrap width used when subsurface scattering is enabled.
const WRAP_WIDTH: f32 = 0.5;

/// Estevez-Kulla *Charlie* sheen distribution. Mirrors `cloth_distribution_charlie`.
fn distribution_charlie(roughness: f32, n_dot_h: f32) -> f32 {
    let inv_r = 1.0 / roughness;
    let sin2h = (1.0 - n_dot_h * n_dot_h).max(MIN_SIN2);
    (2.0 + inv_r) * ops::powf(sin2h, inv_r * 0.5) * (1.0 / (2.0 * PI))
}

/// Ashikhmin/Neubelt visibility term for the sheen lobe. Mirrors
/// `cloth_visibility_ashikhmin`.
fn visibility_ashikhmin(n_dot_v: f32, n_dot_l: f32) -> f32 {
    1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)).max(MIN_N_DOT)
}

/// Energy-conserving diffuse wrap `saturate((x + w) / (1 + w)^2)`. Mirrors
/// `cloth_fd_wrap`.
fn fd_wrap(x: f32, w: f32) -> f32 {
    ((x + w) / ((1.0 + w) * (1.0 + w))).clamp(0.0, 1.0)
}

/// Evaluates the cloth BSDF for a single analytic light.
///
/// Returns linear radiance for the light plus the surface emissive term. When
/// the surface faces away from the light the emissive term is returned on its
/// own, exactly like the principled lobe.
pub fn evaluate_cloth_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> [f32; 3] {
    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let v = normalize_or(frame.view, n);
    let l = normalize_or(light.direction, n);
    let h = normalize_or(add(v, l), n);

    let n_dot_l = dot(n, l).max(0.0);
    if n_dot_l <= 0.0 {
        return surface.emissive;
    }
    let n_dot_v = dot(n, v).max(MIN_N_DOT);
    let n_dot_h = dot(n, h).max(MIN_N_DOT);
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);

    // Sheen specular: Charlie distribution * Ashikhmin visibility, tinted by
    // the scalar sheen weight (sheen_color = vec3(sheen)); no Fresnel term.
    let sheen = surface.sheen.clamp(0.0, 1.0);
    let d = distribution_charlie(roughness, n_dot_h);
    let vis = visibility_ashikhmin(n_dot_v, n_dot_l);
    let sheen_specular = [sheen * d * vis; 3];

    // Diffuse: Lambertian base, optionally softened by the subsurface wrap and
    // tinted toward the base color to fake short-range scattering.
    let subsurface = surface.subsurface.clamp(0.0, 1.0);
    let mut diffuse = mul_scalar(surface.base_color, INV_PI);
    if subsurface > 0.0 {
        diffuse = mul_scalar(diffuse, fd_wrap(n_dot_l, WRAP_WIDTH));
        let tint = saturate3([
            surface.base_color[0] * subsurface + n_dot_l,
            surface.base_color[1] * subsurface + n_dot_l,
            surface.base_color[2] * subsurface + n_dot_l,
        ]);
        diffuse = mul(diffuse, tint);
    }

    let radiance = mul_scalar(
        mul(add(diffuse, sheen_specular), light.illuminance),
        n_dot_l * light.visibility.clamp(0.0, 1.0),
    );
    add(radiance, surface.emissive)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> ShadingFrame {
        ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        }
    }

    fn light() -> DirectLightSample {
        DirectLightSample {
            direction: [0.0, 1.0, 0.0],
            illuminance: [1.0; 3],
            visibility: 1.0,
        }
    }

    fn base_surface() -> SurfaceSample {
        SurfaceSample {
            base_color: [0.5, 0.4, 0.3],
            perceptual_roughness: 0.6,
            sheen: 0.5,
            ..Default::default()
        }
    }

    #[test]
    fn cloth_is_finite_and_non_negative_across_parameters() {
        for roughness in [0.045, 0.2, 0.5, 1.0] {
            for sheen in [0.0, 0.5, 1.0] {
                for subsurface in [0.0, 0.5, 1.0] {
                    let value = evaluate_cloth_direct(
                        SurfaceSample {
                            perceptual_roughness: roughness,
                            sheen,
                            subsurface,
                            ..base_surface()
                        },
                        frame(),
                        light(),
                    );
                    assert!(
                        value.into_iter().all(|c| c.is_finite() && c >= 0.0),
                        "roughness={roughness} sheen={sheen} subsurface={subsurface} -> {value:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn back_facing_returns_only_emissive() {
        let surface = SurfaceSample {
            emissive: [0.1, 0.2, 0.3],
            ..base_surface()
        };
        let away = DirectLightSample {
            direction: [0.0, -1.0, 0.0],
            ..light()
        };
        assert_eq!(evaluate_cloth_direct(surface, frame(), away), surface.emissive);
    }

    #[test]
    fn zero_visibility_leaves_only_emissive() {
        let surface = SurfaceSample {
            emissive: [0.05, 0.05, 0.05],
            ..base_surface()
        };
        let shadowed = DirectLightSample {
            visibility: 0.0,
            ..light()
        };
        assert_eq!(
            evaluate_cloth_direct(surface, frame(), shadowed),
            surface.emissive
        );
    }

    #[test]
    fn increasing_sheen_brightens_the_specular_lobe() {
        // A grazing view/light configuration maximises the Charlie lobe.
        let grazing_frame = ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: normalize_or([1.0, 0.2, 0.0], [0.0, 1.0, 0.0]),
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let grazing_light = DirectLightSample {
            direction: normalize_or([-1.0, 0.2, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let low = evaluate_cloth_direct(
            SurfaceSample { sheen: 0.1, ..base_surface() },
            grazing_frame,
            grazing_light,
        );
        let high = evaluate_cloth_direct(
            SurfaceSample { sheen: 0.9, ..base_surface() },
            grazing_frame,
            grazing_light,
        );
        let sum = |c: [f32; 3]| c[0] + c[1] + c[2];
        assert!(
            sum(high) > sum(low),
            "sheen should brighten the lobe: low={low:?} high={high:?}"
        );
    }

    #[test]
    fn subsurface_changes_the_diffuse_response() {
        let without = evaluate_cloth_direct(
            SurfaceSample { subsurface: 0.0, ..base_surface() },
            frame(),
            light(),
        );
        let with = evaluate_cloth_direct(
            SurfaceSample { subsurface: 1.0, ..base_surface() },
            frame(),
            light(),
        );
        assert_ne!(without, with, "subsurface weight must alter the diffuse term");
    }
}
