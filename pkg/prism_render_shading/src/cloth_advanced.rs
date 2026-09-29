//! Advanced cloth (fabric) BSDF direct-lighting closure.
//!
//! Highest-fidelity sibling of [`crate::evaluate_cloth_direct`]. Where the base
//! `cloth.rs` lobe is a bare `Charlie` sheen plus a Lambertian/wrap diffuse,
//! this reference layers the full production-grade fabric response and is the
//! byte-for-byte numerical twin of `cloth_advanced.wesl`
//! (`cloth_advanced_direct`):
//!
//! * **Energy-conserving sheen** - the `Estevez`-`Kulla` *`Charlie`*
//!   distribution together with the Ashikhmin/Neubelt visibility term, scaled
//!   by an albedo-compensation polynomial fit to the `Estevez`-`Kulla`
//!   directional-albedo table so the fuzz lobe conserves energy as roughness
//!   grows, and darkening the diffuse base by the sheen directional albedo
//!   (the `Filament` "sheen scaling" energy-conservation rule).
//! * **Woven warp/weft anisotropy** - an `Ashikhmin`-`Shirley` dual-tangent
//!   anisotropic highlight. The authored tangent frame is re-orthonormalized
//!   against the shaded normal and rotated by `anisotropy_rotation`, then split
//!   into two thread roughnesses (warp along the tangent, weft along the
//!   bitangent) via the Burley `aspect` mapping.
//! * **Thin double-sided transmission** - a `DICE`-style back-light wrap lobe
//!   modulated by `surface.thickness` so thin fabrics glow when back-lit.
//! * **Fibre multiple-scattering compensation** - a `Turquin` 2019 style
//!   bounded energy-compensation multiplier that restores the sheen energy lost
//!   to single scatter.
//! * **Thin-film interference** - a per-wavelength interference `Fresnel` tint
//!   whose phase is driven by `surface.thickness` (iridescent sheen).
//! * **Tension-driven wrinkle blend** - `anisotropy` doubles as a tension
//!   proxy that sharpens the half-vector response, mixing in slack-cloth
//!   wrinkle scattering.
//!
//! Emissive is folded in exactly once and every lobe is scaled by the light
//! visibility, so a fully occluded light returns the emissive term on its own
//! and the closure stays linearly accumulable across many lights. The front
//! lobes are gated by `n_dot_l` (they vanish when the surface faces away) while
//! the thin-transmission lobe is double-sided, so a back-lit thin fabric still
//! transmits.

use core::f32::consts::PI;

use bevy_math::ops;

use crate::vecmath::{add, cross, dot, mul, mul_scalar, normalize_or, sub};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Reciprocal of pi, matching the GPU `INV_PI` constant.
const INV_PI: f32 = 1.0 / PI;
/// Two pi, the thin-film interference phase scale.
const TWO_PI: f32 = 2.0 * PI;
/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;
/// Dot-product floor shared with the principled lobe (`MIN_N_DOT`).
const MIN_N_DOT: f32 = 1.0e-5;
/// `sin^2(theta_h)` floor for the `Charlie`/`Ashikhmin`-`Shirley` lobes (1/128),
/// preventing a singular `pow` at grazing half-vectors.
const MIN_SIN2: f32 = 0.007_812_5;
/// Index of refraction of the thin surface film (soap-bubble regime).
const FILM_IOR: f32 = 1.35;
/// Maximum optical film thickness (nanometres) at `surface.thickness == 1`.
const FILM_MAX_NM: f32 = 600.0;
/// Red interference wavelength (nanometres).
const LAMBDA_R: f32 = 650.0;
/// Green interference wavelength (nanometres).
const LAMBDA_G: f32 = 550.0;
/// Blue interference wavelength (nanometres).
const LAMBDA_B: f32 = 450.0;
/// Normal-incidence dielectric reflectance used by the thin-film `Fresnel`.
const FILM_F0: f32 = 0.04;
/// Back-light wrap distortion for the thin-transmission lobe (`DICE`).
const TRANS_DISTORTION: f32 = 0.2;
/// Forward-scatter sharpness of the thin-transmission lobe (`DICE`).
const TRANS_POWER: f32 = 4.0;

/// `Estevez`-`Kulla` energy-conserving *`Charlie`* sheen distribution.
///
/// Evaluates the bare `Charlie` `D` term and multiplies it by an
/// albedo-compensation polynomial fit to the `Estevez`-`Kulla` directional
/// albedo: rougher fuzz scatters wider and loses single-scatter energy, so the
/// lobe is scaled up to conserve it. Mirrors `cloth_advanced_distribution_charlie_ec`.
fn distribution_charlie_ec(roughness: f32, n_dot_h: f32) -> f32 {
    let inv_r = 1.0 / roughness;
    let sin2h = (1.0 - n_dot_h * n_dot_h).max(MIN_SIN2);
    let charlie = (2.0 + inv_r) * ops::powf(sin2h, inv_r * 0.5) * (1.0 / TWO_PI);
    // Albedo-compensation multiplier (fit to the Estevez-Kulla table); bounded
    // and >= 1 so it only ever restores lost energy.
    let comp = 1.0 + roughness * (0.309 - 0.288 * roughness);
    charlie * comp
}

/// Ashikhmin/Neubelt visibility term for the sheen lobe. Mirrors
/// `cloth_advanced_visibility_ashikhmin`.
fn visibility_ashikhmin(n_dot_v: f32, n_dot_l: f32) -> f32 {
    1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)).max(MIN_N_DOT)
}

/// Sheen directional albedo (hemispherical reflectance) approximation.
///
/// A bounded polynomial fit to the `Estevez`-`Kulla` sheen albedo table: fuzz
/// reflects more at grazing view angles and with higher roughness. Used both to
/// darken the diffuse base (energy conservation) and to drive the
/// multiple-scattering compensation. Mirrors `cloth_advanced_sheen_albedo`.
fn sheen_albedo(n_dot_v: f32, roughness: f32) -> f32 {
    let grazing = (1.0 - n_dot_v).clamp(0.0, 1.0);
    let grazing3 = grazing * grazing * grazing;
    let fresnel_like = FILM_F0 + (1.0 - FILM_F0) * grazing3;
    (fresnel_like * (0.25 + 0.75 * roughness)).clamp(0.0, 1.0)
}

/// `Turquin` 2019 style multiple-scattering energy compensation.
///
/// Restores the sheen energy lost to single scatter as a bounded additive
/// series driven by the sheen directional albedo `e` and the `sheen` weight.
/// Returns a multiplier `>= 1`. Mirrors `cloth_advanced_multiscatter_comp`.
fn multiscatter_comp(e: f32, sheen: f32) -> f32 {
    let e = e.clamp(0.0, 1.0);
    1.0 + sheen * (1.0 - e)
}

/// Re-orthonormalized, `anisotropy_rotation`-rotated warp/weft tangent basis.
///
/// Mirrors the principled `anisotropic_axes` helper: the authored tangent is
/// projected off the shaded normal, the bitangent handedness is restored, and
/// both axes are rotated within the tangent plane. Returns `(warp, weft)`.
/// Mirrors `cloth_advanced_woven_axes`.
fn woven_axes(
    normal: [f32; 3],
    basis_tangent: [f32; 3],
    basis_bitangent: [f32; 3],
    rotation: f32,
) -> ([f32; 3], [f32; 3]) {
    let projected = sub(
        basis_tangent,
        mul_scalar(normal, dot(normal, basis_tangent)),
    );
    let tangent0 = normalize_or(projected, basis_tangent);
    let handedness = if dot(cross(normal, tangent0), basis_bitangent) < 0.0 {
        -1.0
    } else {
        1.0
    };
    let bitangent0 = mul_scalar(cross(normal, tangent0), handedness);
    let cos_r = ops::cos(rotation);
    let sin_r = ops::sin(rotation);
    let warp = add(mul_scalar(tangent0, cos_r), mul_scalar(bitangent0, sin_r));
    let weft = sub(mul_scalar(bitangent0, cos_r), mul_scalar(tangent0, sin_r));
    (warp, weft)
}

/// `Ashikhmin`-`Shirley` woven warp/weft anisotropic highlight (scalar).
///
/// Splits `perceptual_roughness` into two thread roughnesses via the Burley
/// `aspect` mapping (`anisotropy`), converts them into `Ashikhmin`-`Shirley`
/// specular exponents and evaluates the anisotropic distribution about the
/// rotated warp/weft basis, divided by the `Ashikhmin`-`Shirley` geometry
/// denominator. Mirrors `cloth_advanced_anisotropic_woven`.
#[expect(
    clippy::too_many_arguments,
    reason = "The woven lobe needs the full surface, frame and geometry cosines to stay a leaf helper mirrored byte-for-byte by the WESL twin."
)]
fn anisotropic_woven(
    surface: SurfaceSample,
    frame: ShadingFrame,
    n: [f32; 3],
    v: [f32; 3],
    h: [f32; 3],
    n_dot_h: f32,
    n_dot_v: f32,
    n_dot_l: f32,
) -> f32 {
    let (warp, weft) = woven_axes(
        n,
        frame.tangent,
        frame.bitangent,
        surface.anisotropy_rotation,
    );
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let anisotropy = surface.anisotropy.clamp(-1.0, 1.0);
    let aspect = ops::sqrt(1.0 - 0.9 * anisotropy);
    let r_warp = (roughness / aspect).clamp(MIN_ROUGHNESS, 1.0);
    let r_weft = (roughness * aspect).clamp(MIN_ROUGHNESS, 1.0);
    // Ashikhmin-Shirley exponents from the two thread roughnesses.
    let nu = (2.0 / (r_warp * r_warp) - 2.0).max(0.0);
    let nv = (2.0 / (r_weft * r_weft) - 2.0).max(0.0);
    let h_dot_t = dot(h, warp);
    let h_dot_b = dot(h, weft);
    let sin2h = (1.0 - n_dot_h * n_dot_h).max(MIN_SIN2);
    let exponent = (nu * h_dot_t * h_dot_t + nv * h_dot_b * h_dot_b) / sin2h;
    let norm = ops::sqrt((nu + 1.0) * (nv + 1.0)) * (1.0 / TWO_PI);
    let d = norm * ops::powf(n_dot_h, exponent);
    let v_dot_h = dot(v, h).max(MIN_N_DOT);
    let denom = 4.0 * v_dot_h * n_dot_v.max(n_dot_l);
    d / denom.max(MIN_N_DOT)
}

/// `DICE`-style thin double-sided transmission (back-light) term.
///
/// The transmitted light direction is the incoming direction pushed along the
/// surface normal (the wrap distortion); the view-aligned forward-scatter lobe
/// is raised to `TRANS_POWER` and attenuated by `surface.thickness` so thin
/// fabrics transmit more. Mirrors `cloth_advanced_thin_transmission`.
fn thin_transmission(n: [f32; 3], v: [f32; 3], l: [f32; 3], thickness: f32) -> f32 {
    // vLTLight = L + N * distortion; forward scatter peaks along -vLTLight.
    let lt = add(l, mul_scalar(n, TRANS_DISTORTION));
    let forward = dot(v, mul_scalar(lt, -1.0)).clamp(0.0, 1.0);
    let lobe = ops::powf(forward, TRANS_POWER);
    let thin = 1.0 - thickness.clamp(0.0, 1.0);
    lobe * thin
}

/// Thin-film interference `Fresnel` tint (per channel).
///
/// A dielectric Schlick `Fresnel` modulated by a per-wavelength interference
/// factor `0.5 + 0.5 * cos(phase)`, where the phase is the optical path
/// `2 * n_film * d * cos(theta)` over the wavelength. `surface.thickness`
/// drives the film thickness `d`, so `thickness == 0` yields a flat achromatic
/// `Fresnel` and thicker films sweep through iridescent hues. Mirrors
/// `cloth_advanced_thin_film_fresnel`.
fn thin_film_fresnel(n_dot_v: f32, thickness: f32) -> [f32; 3] {
    let t = thickness.clamp(0.0, 1.0);
    let cos_t = n_dot_v.clamp(0.0, 1.0);
    let one_minus = 1.0 - cos_t;
    let sq = one_minus * one_minus;
    let fresnel = FILM_F0 + (1.0 - FILM_F0) * sq * sq * one_minus;
    let optical = 2.0 * FILM_IOR * (FILM_MAX_NM * t) * cos_t;
    let ir = 0.5 + 0.5 * ops::cos(TWO_PI * optical / LAMBDA_R);
    let ig = 0.5 + 0.5 * ops::cos(TWO_PI * optical / LAMBDA_G);
    let ib = 0.5 + 0.5 * ops::cos(TWO_PI * optical / LAMBDA_B);
    // Blend achromatic (t = 0) toward the interference colours with thickness,
    // then scale by the Fresnel weight.
    [
        (1.0 + (ir - 1.0) * t) * fresnel,
        (1.0 + (ig - 1.0) * t) * fresnel,
        (1.0 + (ib - 1.0) * t) * fresnel,
    ]
}

/// Tension-driven wrinkle half-vector blend.
///
/// Uses `|anisotropy|` as a tension proxy: taut cloth (high tension) sharpens
/// the half-vector response (pushing `n_dot_h` toward 1), while slack cloth
/// keeps the softer authored response. Returns the blended, clamped `n_dot_h`.
/// Mirrors `cloth_advanced_wrinkle_normal_blend`.
fn wrinkle_normal_blend(n_dot_h: f32, anisotropy: f32) -> f32 {
    let tension = anisotropy.abs().clamp(0.0, 1.0);
    let sharpened = ops::powf(n_dot_h, 1.0 + tension);
    let blended = n_dot_h + (sharpened - n_dot_h) * tension;
    blended.clamp(MIN_N_DOT, 1.0)
}

/// Evaluates the advanced cloth BSDF for a single analytic light.
///
/// Returns linear radiance for the light plus the surface emissive term. Every
/// lobe is scaled by the light visibility, so a fully occluded light returns
/// the emissive term on its own. The front-facing sheen/woven lobes are gated
/// by `n_dot_l`, while the thin-transmission lobe is double-sided so a back-lit
/// thin fabric still transmits.
pub fn evaluate_cloth_advanced_direct(
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
    let n_dot_h_raw = dot(n, h).max(MIN_N_DOT);
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let sheen = surface.sheen.clamp(0.0, 1.0);
    let visibility = light.visibility.clamp(0.0, 1.0);

    // Tension-driven wrinkle blend on the half-vector cosine.
    let n_dot_h = wrinkle_normal_blend(n_dot_h_raw, surface.anisotropy);

    // Energy-conserving Charlie sheen (white fuzz), plus its multiple-scatter
    // compensation and the diffuse-darkening directional albedo.
    let d = distribution_charlie_ec(roughness, n_dot_h);
    let vis = visibility_ashikhmin(n_dot_v, n_dot_l);
    let e = sheen_albedo(n_dot_v, roughness);
    let ms = multiscatter_comp(e, sheen);
    let sheen_lobe = sheen * d * vis * ms;

    // Woven warp/weft anisotropic threads, tinted by the thin-film interference.
    let woven = anisotropic_woven(surface, frame, n, v, h, n_dot_h, n_dot_v, n_dot_l);
    let film = thin_film_fresnel(n_dot_v, surface.thickness);
    let specular = [
        sheen_lobe + woven * film[0],
        sheen_lobe + woven * film[1],
        sheen_lobe + woven * film[2],
    ];

    // Diffuse: Lambertian base darkened by the sheen directional albedo so the
    // sheen + diffuse energy stays bounded (Filament sheen scaling).
    let diff_scale = (1.0 - sheen * e).clamp(0.0, 1.0);
    let diffuse = mul_scalar(surface.base_color, INV_PI * diff_scale);

    // Front-facing radiance (naturally zero when the surface faces away).
    let front = mul_scalar(
        mul(add(diffuse, specular), light.illuminance),
        n_dot_l * visibility,
    );

    // Thin double-sided transmission (back-light), tinted by the base color.
    let trans = thin_transmission(n, v, l, surface.thickness);
    let back = mul_scalar(
        mul(surface.base_color, light.illuminance),
        trans * visibility,
    );

    add(add(front, back), surface.emissive)
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
            metallic: 0.0,
            perceptual_roughness: 0.6,
            reflectance: 0.5,
            ambient_occlusion: 1.0,
            emissive: [0.0, 0.0, 0.0],
            clearcoat: 0.0,
            clearcoat_roughness: 0.25,
            sheen: 0.5,
            subsurface: 0.0,
            thickness: 0.3,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
        }
    }

    fn sum(c: [f32; 3]) -> f32 {
        c[0] + c[1] + c[2]
    }

    #[test]
    fn finite_and_non_negative_across_parameters() {
        for roughness in [0.045, 0.2, 0.5, 1.0] {
            for sheen in [0.0, 0.5, 1.0] {
                for anisotropy in [-1.0, -0.3, 0.0, 0.7, 1.0] {
                    for thickness in [0.0, 0.4, 1.0] {
                        for rotation in [0.0, 1.2, 3.0] {
                            let value = evaluate_cloth_advanced_direct(
                                SurfaceSample {
                                    perceptual_roughness: roughness,
                                    sheen,
                                    anisotropy,
                                    anisotropy_rotation: rotation,
                                    thickness,
                                    ..base_surface()
                                },
                                frame(),
                                light(),
                            );
                            assert!(
                                value.into_iter().all(|c| c.is_finite() && c >= 0.0),
                                "roughness={roughness} sheen={sheen} anisotropy={anisotropy} \
                                 thickness={thickness} rotation={rotation} -> {value:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_visibility_returns_only_emissive() {
        let surface = SurfaceSample {
            emissive: [0.05, 0.06, 0.07],
            ..base_surface()
        };
        let shadowed = DirectLightSample {
            visibility: 0.0,
            ..light()
        };
        assert_eq!(
            evaluate_cloth_advanced_direct(surface, frame(), shadowed),
            surface.emissive
        );
    }

    #[test]
    fn increasing_sheen_brightens_the_grazing_specular() {
        // A grazing view/light configuration maximises the Charlie fuzz lobe.
        // Base color is black so only the (white) sheen lobe contributes and the
        // energy-conserving diffuse darkening cannot mask the sheen gain.
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
        let low = evaluate_cloth_advanced_direct(
            SurfaceSample {
                base_color: [0.0, 0.0, 0.0],
                sheen: 0.1,
                ..base_surface()
            },
            grazing_frame,
            grazing_light,
        );
        let high = evaluate_cloth_advanced_direct(
            SurfaceSample {
                base_color: [0.0, 0.0, 0.0],
                sheen: 0.9,
                ..base_surface()
            },
            grazing_frame,
            grazing_light,
        );
        assert!(
            sum(high) > sum(low),
            "sheen should brighten the grazing lobe: low={low:?} high={high:?}"
        );
    }

    #[test]
    fn back_lit_thin_fabric_transmits() {
        // Light coming from behind the surface (n_dot_l <= 0) still transmits
        // through a thin fabric, so the result exceeds the emissive floor.
        let surface = SurfaceSample {
            emissive: [0.01, 0.01, 0.01],
            thickness: 0.0,
            ..base_surface()
        };
        // View from above, light from directly below (straight through).
        let backlit = DirectLightSample {
            direction: [0.0, -1.0, 0.0],
            ..light()
        };
        let value = evaluate_cloth_advanced_direct(surface, frame(), backlit);
        assert!(
            sum(value) > sum(surface.emissive),
            "back-lit thin fabric should transmit above emissive: {value:?}"
        );
    }

    #[test]
    fn thinner_fabric_transmits_more_than_thick() {
        let backlit = DirectLightSample {
            direction: [0.0, -1.0, 0.0],
            ..light()
        };
        let thin = evaluate_cloth_advanced_direct(
            SurfaceSample {
                thickness: 0.0,
                ..base_surface()
            },
            frame(),
            backlit,
        );
        let thick = evaluate_cloth_advanced_direct(
            SurfaceSample {
                thickness: 1.0,
                ..base_surface()
            },
            frame(),
            backlit,
        );
        assert!(
            sum(thin) > sum(thick),
            "thinner fabric should transmit more: thin={thin:?} thick={thick:?}"
        );
    }

    #[test]
    fn thin_film_thickness_tints_the_specular() {
        // With a woven highlight present, a non-zero film thickness must shift
        // the per-channel response away from the achromatic thickness-0 case.
        let woven_frame = ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: normalize_or([0.4, 0.9, 0.0], [0.0, 1.0, 0.0]),
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let woven_light = DirectLightSample {
            direction: normalize_or([-0.4, 0.9, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let base = SurfaceSample {
            base_color: [1.0, 1.0, 1.0],
            sheen: 0.0,
            anisotropy: 0.8,
            ..base_surface()
        };
        let flat = evaluate_cloth_advanced_direct(
            SurfaceSample {
                thickness: 0.0,
                ..base
            },
            woven_frame,
            woven_light,
        );
        let filmed = evaluate_cloth_advanced_direct(
            SurfaceSample {
                thickness: 0.7,
                ..base
            },
            woven_frame,
            woven_light,
        );
        // The interference must break the channel symmetry the flat case keeps.
        let flat_spread = (flat[0] - flat[2]).abs();
        let film_spread = (filmed[0] - filmed[2]).abs();
        assert!(
            film_spread > flat_spread,
            "thin-film thickness should tint the specular: flat={flat:?} filmed={filmed:?}"
        );
    }

    #[test]
    fn anisotropy_rotation_changes_the_woven_response() {
        let woven_frame = ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: normalize_or([0.4, 0.9, 0.2], [0.0, 1.0, 0.0]),
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let woven_light = DirectLightSample {
            direction: normalize_or([-0.3, 0.9, 0.3], [0.0, 1.0, 0.0]),
            ..light()
        };
        let unrotated = evaluate_cloth_advanced_direct(
            SurfaceSample {
                sheen: 0.0,
                anisotropy: 0.9,
                anisotropy_rotation: 0.0,
                ..base_surface()
            },
            woven_frame,
            woven_light,
        );
        let rotated = evaluate_cloth_advanced_direct(
            SurfaceSample {
                sheen: 0.0,
                anisotropy: 0.9,
                anisotropy_rotation: core::f32::consts::FRAC_PI_2,
                ..base_surface()
            },
            woven_frame,
            woven_light,
        );
        assert_ne!(
            unrotated, rotated,
            "rotating the woven warp/weft basis must change the highlight"
        );
    }

    #[test]
    fn higher_sheen_darkens_the_diffuse_base() {
        // Energy conservation (Filament sheen scaling): a bright, diffuse-
        // dominated fabric viewed head-on must lose diffuse energy to the sheen
        // layer as the sheen weight rises.
        let head_on = SurfaceSample {
            base_color: [0.9, 0.9, 0.9],
            perceptual_roughness: 1.0,
            thickness: 0.0,
            ..base_surface()
        };
        let grazing_frame = ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: normalize_or([1.0, 0.25, 0.0], [0.0, 1.0, 0.0]),
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let grazing_light = DirectLightSample {
            direction: normalize_or([-1.0, 0.25, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let no_sheen = evaluate_cloth_advanced_direct(
            SurfaceSample {
                sheen: 0.0,
                ..head_on
            },
            grazing_frame,
            grazing_light,
        );
        let full_sheen = evaluate_cloth_advanced_direct(
            SurfaceSample {
                sheen: 1.0,
                ..head_on
            },
            grazing_frame,
            grazing_light,
        );
        assert!(
            sum(full_sheen) < sum(no_sheen),
            "sheen must take energy from the diffuse base: none={no_sheen:?} full={full_sheen:?}"
        );
    }

    #[test]
    fn evaluation_is_deterministic() {
        let a = evaluate_cloth_advanced_direct(base_surface(), frame(), light());
        let b = evaluate_cloth_advanced_direct(base_surface(), frame(), light());
        assert_eq!(a, b);
    }
}
