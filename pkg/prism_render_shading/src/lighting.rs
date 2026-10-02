use core::f32::consts::PI;

use bevy_math::ops;

use crate::tangent::orthonormal_basis;
use crate::vecmath::{add, cross, dot, mix3, mul, mul_scalar, normalize_or, sub};

const MIN_ROUGHNESS: f32 = 0.045;
const MIN_N_DOT: f32 = 1.0e-5;
/// Lower bound on the per-axis GGX roughness so the anisotropic lobe stays
/// finite when `aspect` drives one axis toward a perfect mirror (UE floors the
/// anisotropic roughness identically).
const MIN_ANISOTROPIC_ALPHA: f32 = 1.0e-3;

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
    /// Optical thickness in `[0, 1]` driving Subsurface back-transmission
    /// (0 = paper-thin, full transmission; 1 = opaque, no transmission).
    pub thickness: f32,
    /// Signed anisotropy in `[-1, 1]`. Positive elongates the specular
    /// highlight along the tangent, negative along the bitangent, `0` is
    /// isotropic (Disney/Burley `aspect` mapping).
    pub anisotropy: f32,
    /// Rotation of the anisotropy axes around the surface normal, in radians
    /// (UE `AnisotropyRotation`). `0` keeps the authored tangent as the major
    /// axis.
    pub anisotropy_rotation: f32,
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
            thickness: 0.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingFrame {
    pub normal: [f32; 3],
    pub view: [f32; 3],
    /// Unit tangent (major anisotropy axis) orthogonal to `normal`.
    pub tangent: [f32; 3],
    /// Unit bitangent carrying the frame handedness, orthogonal to both.
    pub bitangent: [f32; 3],
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

    // Anisotropic axes. The tangent frame is re-orthonormalized against the
    // shaded normal (interpolation drifts it) and rotated around the normal by
    // `anisotropy_rotation` before the Burley `aspect` split maps `alpha` onto
    // the two axes. With `anisotropy == 0` the axes collapse to `alpha` and the
    // anisotropic `D`/`V` reduce exactly to the isotropic GGX pair.
    let (tangent, bitangent) = anisotropic_axes(
        n,
        frame.tangent,
        frame.bitangent,
        surface.anisotropy_rotation,
    );
    let anisotropy = surface.anisotropy.clamp(-1.0, 1.0);
    let aspect = (1.0 - 0.9 * anisotropy).sqrt();
    let ax = (alpha / aspect).max(MIN_ANISOTROPIC_ALPHA);
    let ay = (alpha * aspect).max(MIN_ANISOTROPIC_ALPHA);
    let d = distribution_ggx_anisotropic(dot(tangent, h), dot(bitangent, h), n_dot_h, ax, ay);
    let g = visibility_smith_ggx_anisotropic(
        dot(tangent, v),
        dot(bitangent, v),
        dot(tangent, l),
        dot(bitangent, l),
        n_dot_v,
        n_dot_l,
        ax,
        ay,
    );
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
        coat * distribution_ggx(n_dot_h, coat_alpha)
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
                tangent: [1.0, 0.0, 0.0],
                bitangent: [0.0, 0.0, -1.0],
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

pub(crate) fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let alpha_squared = alpha * alpha;
    let denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    alpha_squared / (PI * denominator * denominator).max(MIN_N_DOT)
}

pub(crate) fn visibility_smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let alpha_squared = alpha * alpha;
    let gv = n_dot_l * ((n_dot_v - n_dot_v * alpha_squared) * n_dot_v + alpha_squared).sqrt();
    let gl = n_dot_v * ((n_dot_l - n_dot_l * alpha_squared) * n_dot_l + alpha_squared).sqrt();
    0.5 / (gv + gl).max(MIN_N_DOT)
}

/// Re-orthonormalizes the interpolated tangent frame against `normal` and
/// rotates it around the normal by `rotation` radians (UE `AnisotropyRotation`).
///
/// The returned `(tangent, bitangent)` are unit length and orthogonal to
/// `normal`; the bitangent keeps the handedness carried by `basis_bitangent`.
fn anisotropic_axes(
    normal: [f32; 3],
    basis_tangent: [f32; 3],
    basis_bitangent: [f32; 3],
    rotation: f32,
) -> ([f32; 3], [f32; 3]) {
    // Fall back to the canonical Duff basis when the interpolated tangent has
    // collapsed onto the normal.
    let (fallback_t, _fallback_b) = orthonormal_basis(normal);
    let projected = sub(
        basis_tangent,
        mul_scalar(normal, dot(normal, basis_tangent)),
    );
    let tangent0 = normalize_or(projected, fallback_t);
    let handedness = if dot(cross(normal, tangent0), basis_bitangent) < 0.0 {
        -1.0
    } else {
        1.0
    };
    let bitangent0 = mul_scalar(cross(normal, tangent0), handedness);
    // Rotate within the tangent plane. `ops::{cos,sin}` route through the
    // deterministic libm backend so the CPU golden matches the WESL twin.
    let cos_r = ops::cos(rotation);
    let sin_r = ops::sin(rotation);
    let tangent = add(mul_scalar(tangent0, cos_r), mul_scalar(bitangent0, sin_r));
    let bitangent = sub(mul_scalar(bitangent0, cos_r), mul_scalar(tangent0, sin_r));
    (tangent, bitangent)
}

/// Anisotropic GGX/Trowbridge-Reitz distribution (UE `D_GGXaniso`). Reduces to
/// [`distribution_ggx`] when `ax == ay`.
pub(crate) fn distribution_ggx_anisotropic(
    x_dot_h: f32,
    y_dot_h: f32,
    n_dot_h: f32,
    ax: f32,
    ay: f32,
) -> f32 {
    let dx = x_dot_h / ax;
    let dy = y_dot_h / ay;
    let d = dx * dx + dy * dy + n_dot_h * n_dot_h;
    1.0 / (PI * ax * ay * d * d).max(MIN_N_DOT)
}

/// Height-correlated anisotropic Smith visibility (UE `Vis_SmithJointAniso`).
/// Reduces to [`visibility_smith_ggx_correlated`] when `ax == ay`.
pub(crate) fn visibility_smith_ggx_anisotropic(
    x_dot_v: f32,
    y_dot_v: f32,
    x_dot_l: f32,
    y_dot_l: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    ax: f32,
    ay: f32,
) -> f32 {
    let ax_v = ax * x_dot_v;
    let ay_v = ay * y_dot_v;
    let lambda_v = n_dot_l * (ax_v * ax_v + ay_v * ay_v + n_dot_v * n_dot_v).sqrt();
    let ax_l = ax * x_dot_l;
    let ay_l = ay * y_dot_l;
    let lambda_l = n_dot_v * (ax_l * ax_l + ay_l * ay_l + n_dot_l * n_dot_l).sqrt();
    0.5 / (lambda_v + lambda_l).max(MIN_N_DOT)
}

pub(crate) fn fresnel_schlick(f0: [f32; 3], v_dot_h: f32) -> [f32; 3] {
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
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
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
                assert!(value
                    .into_iter()
                    .all(|channel| channel.is_finite() && channel >= 0.0));
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
                assert!(response
                    .into_iter()
                    .all(|channel| channel.is_finite() && channel <= 1.2));
            }
        }
    }

    #[test]
    fn anisotropy_zero_matches_isotropic_ggx() {
        // With `ax == ay` the anisotropic `D`/`V` must collapse exactly onto
        // the isotropic GGX pair for any half-vector, so leaving anisotropy at
        // its default is numerically free.
        let n = [0.0, 1.0, 0.0];
        let t = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, -1.0];
        for alpha in [0.05_f32, 0.2, 0.5, 0.9] {
            for raw in [
                [0.0, 1.0, 0.0],
                [0.2, 0.9, 0.1],
                [0.5, 0.7, 0.5],
                [-0.3, 0.8, 0.2],
            ] {
                let h = normalize_or(raw, n);
                let n_dot_h = dot(n, h).max(MIN_N_DOT);
                let iso = distribution_ggx(n_dot_h, alpha);
                let aniso =
                    distribution_ggx_anisotropic(dot(t, h), dot(b, h), n_dot_h, alpha, alpha);
                assert!(
                    (iso - aniso).abs() <= 1.0e-4 * iso.max(1.0),
                    "D {iso} vs {aniso}"
                );
            }
        }
        for alpha in [0.1_f32, 0.5, 1.0] {
            let v = normalize_or([0.3, 0.9, 0.1], n);
            let l = normalize_or([-0.2, 0.8, 0.4], n);
            let n_dot_v = dot(n, v).max(MIN_N_DOT);
            let n_dot_l = dot(n, l).max(0.0);
            let iso = visibility_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
            let aniso = visibility_smith_ggx_anisotropic(
                dot(t, v),
                dot(b, v),
                dot(t, l),
                dot(b, l),
                n_dot_v,
                n_dot_l,
                alpha,
                alpha,
            );
            assert!(
                (iso - aniso).abs() <= 1.0e-4 * iso.max(1.0),
                "V {iso} vs {aniso}"
            );
        }
    }

    #[test]
    fn positive_anisotropy_stretches_the_tangent_lobe() {
        // Positive anisotropy widens the lobe along the tangent (ax > ay), so a
        // half-vector tilted along the tangent stays brighter than the same
        // tilt along the bitangent.
        let alpha = 0.3_f32;
        let aspect = (1.0 - 0.9 * 0.8_f32).sqrt();
        let ax = alpha / aspect;
        let ay = alpha * aspect;
        assert!(ax > ay);
        let sin = 0.5_f32;
        let cos = (1.0 - sin * sin).sqrt();
        // Tangent-aligned tilt: XoH = sin, YoH = 0, NoH = cos.
        let d_tangent = distribution_ggx_anisotropic(sin, 0.0, cos, ax, ay);
        // Bitangent-aligned tilt: XoH = 0, YoH = sin.
        let d_bitangent = distribution_ggx_anisotropic(0.0, sin, cos, ax, ay);
        assert!(d_tangent > d_bitangent, "{d_tangent} !> {d_bitangent}");
    }

    #[test]
    fn anisotropy_rotation_turns_the_major_axis_onto_the_bitangent() {
        // A quarter-turn must rotate the authored tangent onto the bitangent so
        // the anisotropy highlight follows the artist-controlled rotation.
        let n = [0.0, 1.0, 0.0];
        let t = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, -1.0];
        let (unrotated_t, _unrotated_b) = anisotropic_axes(n, t, b, 0.0);
        assert!(dot(unrotated_t, t) > 0.999, "{unrotated_t:?}");
        let (rotated_t, _rotated_b) = anisotropic_axes(n, t, b, core::f32::consts::FRAC_PI_2);
        assert!(dot(rotated_t, b) > 0.999, "{rotated_t:?}");
        // The rotated frame stays orthonormal to the shaded normal.
        assert!(dot(rotated_t, n).abs() < 1.0e-6);
    }
}
