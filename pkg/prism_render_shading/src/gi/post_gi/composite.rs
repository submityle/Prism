//! Final GI / AO lighting composite — backend-neutral CPU golden.
//!
//! This is the last numerical step before the display tone map
//! ([`super::tonemap`]): it folds direct lighting, indirect diffuse GI,
//! glossy specular GI and the occlusion terms into one pre-exposed HDR colour,
//! applies the auto-exposure multiplier ([`super::auto_exposure`]) and the
//! bloom overlay ([`super::bloom`]), and clamps the result.
//!
//! The composite follows the standard decoupled-occlusion rule used across AAA
//! deferred shading: ambient occlusion attenuates the *diffuse* indirect term,
//! while a separate *specular occlusion* attenuates the glossy indirect term so
//! shiny surfaces are not over-darkened by a diffuse AO buffer. Two refinements
//! from the literature are included:
//!
//! * **Multi-bounce AO** (Jimenez et al., GTAO, SIGGRAPH 2016): a per-albedo
//!   cubic fit that colours and lifts the AO so dark cavities do not crush to
//!   black, approximating inter-reflection within the occluded region.
//! * **Bent-normal specular occlusion** (Lagarde & de Rousiers, Frostbite):
//!   `saturate(pow(NoV + ao, exp2(-16*roughness - 1)) - 1 + ao)`, which couples
//!   the scalar AO, the view angle and the roughness so rough surfaces recover
//!   more of their reflection than mirror-like ones.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe / global state.
//! * Transcendental math via [`bevy_math::ops`]; vectors via [`bevy_math`].
//! * Radiance, albedo and occlusion inputs are floored to their valid ranges
//!   (`[0, inf)` light, `[0, 1]` occlusion) and the output is clamped to a
//!   finite non-negative ceiling. No path can emit `NaN` or `inf`.
//! * `f32` storage mirrors the layout the WESL/GPU twin consumes.

use bevy_math::{ops, Vec3};

/// Rec. 709 luminance weights (linear sRGB primaries).
pub const LUMINANCE_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Default ceiling the final composite is clamped to, keeping HDR values in a
/// well-conditioned float range before tone mapping.
pub const DEFAULT_HDR_CEILING: f32 = 65504.0;

/// Rec. 709 relative luminance of a linear RGB sample, floored to zero.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    (rgb[0] * LUMINANCE_WEIGHTS[0] + rgb[1] * LUMINANCE_WEIGHTS[1] + rgb[2] * LUMINANCE_WEIGHTS[2])
        .max(0.0)
}

#[inline]
fn floor_light(rgb: [f32; 3]) -> [f32; 3] {
    [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]
}

/// Jimenez GTAO multi-bounce AO: turns a scalar visibility `ao` into a
/// per-channel multiplier that depends on the surface `albedo`, lifting dark
/// cavities toward the albedo colour instead of pure black.
///
/// Uses the published cubic fit `a*v^3 - b*v^2 + c*v` evaluated per channel,
/// with `(a, b, c)` derived from the channel albedo. `ao` is clamped to
/// `[0, 1]`; the result is `>= ao` per channel (never darker than scalar AO)
/// and `<= 1`.
#[must_use]
pub fn multi_bounce_ao(ao: f32, albedo: [f32; 3]) -> [f32; 3] {
    let v = ao.clamp(0.0, 1.0);
    let mut out = [0.0_f32; 3];
    for (o, &a) in out.iter_mut().zip(albedo.iter()) {
        let a = a.clamp(0.0, 1.0);
        let x = a * 2.0404 - 0.3324;
        let y = a * -4.7951 + 0.6417;
        let z = a * 2.7552 + 0.6903;
        // a*v^3 - b*v^2 + c*v, factored via Horner for stability.
        let m = ((x * v + y) * v + z) * v;
        *o = m.clamp(v, 1.0);
    }
    out
}

/// Lagarde / Frostbite specular (reflection) occlusion from scalar AO.
///
/// `saturate(pow(NoV + ao, exp2(-16*roughness - 1)) - 1 + ao)`. Rougher
/// surfaces (`roughness -> 1`) recover more reflection, mirror-like surfaces
/// (`roughness -> 0`) keep the AO closer to the diffuse term. All inputs are
/// clamped to their valid ranges so the `pow` base/exponent stay finite.
#[must_use]
pub fn specular_occlusion(n_dot_v: f32, ao: f32, roughness: f32) -> f32 {
    let nov = n_dot_v.clamp(0.0, 1.0);
    let ao = ao.clamp(0.0, 1.0);
    let r = roughness.clamp(0.0, 1.0);
    let exponent = ops::exp2(-16.0 * r - 1.0);
    let base = (nov + ao).max(0.0);
    (ops::powf(base, exponent) - 1.0 + ao).clamp(0.0, 1.0)
}

/// Scalar occlusion for a light arriving along `light_dir`, given the shading
/// point's bent-normal cone.
///
/// `bent_normal` is the (unit) average unoccluded direction, `cos_aperture` is
/// the cosine of the cone half-angle, and `visibility` is the scalar AO. A
/// light inside the open cone is fully visible; one on the cone edge is
/// attenuated smoothly to zero via a `smoothstep` on `cos(angle)`, then scaled
/// by `visibility`. Degenerate (zero-length) directions fall back to the
/// scalar `visibility`.
#[must_use]
pub fn bent_normal_occlusion(
    bent_normal: Vec3,
    cos_aperture: f32,
    visibility: f32,
    light_dir: Vec3,
) -> f32 {
    let vis = visibility.clamp(0.0, 1.0);
    let bn_len = bent_normal.length();
    let l_len = light_dir.length();
    if bn_len <= f32::MIN_POSITIVE || l_len <= f32::MIN_POSITIVE {
        return vis;
    }
    let cos_angle = (bent_normal.dot(light_dir) / (bn_len * l_len)).clamp(-1.0, 1.0);
    let edge = cos_aperture.clamp(-1.0, 1.0);
    // smoothstep from the cone edge (0) to fully inside (edge -> 1).
    let t = ((cos_angle - edge) / (1.0 - edge).max(1.0e-4)).clamp(0.0, 1.0);
    let cone = t * t * (3.0 - 2.0 * t);
    (cone * vis).clamp(0.0, 1.0)
}

/// The decoupled light terms fed into [`compose_lighting`].
///
/// All radiances are pre-exposed linear HDR and are floored to non-negative on
/// use. The occlusion fields are clamped to `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompositeInput {
    /// Direct (analytic light) radiance — not attenuated by ambient occlusion.
    pub direct: [f32; 3],
    /// Indirect diffuse GI radiance (irradiance * albedo).
    pub diffuse_gi: [f32; 3],
    /// Indirect specular / glossy GI radiance.
    pub specular_gi: [f32; 3],
    /// Surface albedo, used to colour the multi-bounce AO.
    pub albedo: [f32; 3],
    /// Scalar ambient occlusion in `[0, 1]` (`1` fully unoccluded).
    pub ao: f32,
    /// Specular occlusion in `[0, 1]` (`1` fully unoccluded).
    pub specular_occlusion: f32,
}

impl Default for CompositeInput {
    fn default() -> Self {
        Self {
            direct: [0.0, 0.0, 0.0],
            diffuse_gi: [0.0, 0.0, 0.0],
            specular_gi: [0.0, 0.0, 0.0],
            albedo: [1.0, 1.0, 1.0],
            ao: 1.0,
            specular_occlusion: 1.0,
        }
    }
}

/// Composites direct + occluded indirect lighting into one pre-exposed HDR
/// colour: `direct + diffuse_gi * multiBounceAO(ao, albedo) + specular_gi *
/// specular_occlusion`.
///
/// With `ao == 1` and `specular_occlusion == 1` the occlusion terms are the
/// identity, so the result is the energy-conserving sum of the three light
/// terms. Negative inputs are floored before use.
#[must_use]
pub fn compose_lighting(input: CompositeInput) -> [f32; 3] {
    let direct = floor_light(input.direct);
    let diffuse = floor_light(input.diffuse_gi);
    let specular = floor_light(input.specular_gi);
    let ao_rgb = multi_bounce_ao(input.ao, input.albedo);
    let spec_occ = input.specular_occlusion.clamp(0.0, 1.0);
    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        out[c] = (direct[c] + diffuse[c] * ao_rgb[c] + specular[c] * spec_occ).max(0.0);
    }
    out
}

/// Applies a linear exposure multiplier to a colour, flooring the multiplier to
/// zero so a bad (negative) exposure cannot invert the image.
#[must_use]
pub fn apply_exposure(color: [f32; 3], exposure: f32) -> [f32; 3] {
    let e = exposure.max(0.0);
    [
        (color[0] * e).max(0.0),
        (color[1] * e).max(0.0),
        (color[2] * e).max(0.0),
    ]
}

/// Clamps a colour to `[0, ceiling]` per channel so a stray `NaN`/`inf` never
/// escapes the composite: `NaN` becomes `0`, `+inf` saturates to `ceiling`, and
/// `-inf` floors to `0`.
#[must_use]
pub fn clamp_output(color: [f32; 3], ceiling: f32) -> [f32; 3] {
    let hi = if ceiling.is_finite() && ceiling > 0.0 {
        ceiling
    } else {
        DEFAULT_HDR_CEILING
    };
    let mut out = [0.0_f32; 3];
    for (o, &c) in out.iter_mut().zip(color.iter()) {
        // `clamp` maps +inf -> hi and -inf -> 0; only NaN needs a guard.
        *o = if c.is_nan() { 0.0 } else { c.clamp(0.0, hi) };
    }
    out
}

/// Lerp a `scene` colour with its `bloom` overlay by `intensity`
/// (`scene + (bloom - scene) * intensity`), floored to zero. Mirrors
/// [`super::bloom::combine`] so the composite and bloom modules agree.
#[must_use]
pub fn mix_bloom(scene: [f32; 3], bloom: [f32; 3], intensity: f32) -> [f32; 3] {
    [
        (scene[0] + (bloom[0] - scene[0]) * intensity).max(0.0),
        (scene[1] + (bloom[1] - scene[1]) * intensity).max(0.0),
        (scene[2] + (bloom[2] - scene[2]) * intensity).max(0.0),
    ]
}

/// Full composite tail: [`compose_lighting`], then [`apply_exposure`], then
/// [`mix_bloom`] with the pre-exposed `bloom` overlay, then [`clamp_output`].
///
/// `bloom` is expected to already be exposure-matched to the scene (the bloom
/// pyramid runs on pre-exposed radiance); this applies the same exposure to the
/// composited scene before mixing so both share one scale. The result is the
/// display-ready pre-tonemap HDR colour.
#[must_use]
pub fn composite_frame(
    input: CompositeInput,
    exposure: f32,
    bloom: [f32; 3],
    bloom_intensity: f32,
    ceiling: f32,
) -> [f32; 3] {
    let lit = compose_lighting(input);
    let exposed = apply_exposure(lit, exposure);
    let mixed = mix_bloom(exposed, floor_light(bloom), bloom_intensity);
    clamp_output(mixed, ceiling)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
        approx(a[2], b[2]);
    }

    #[test]
    fn multi_bounce_ao_endpoints() {
        // Fully unoccluded is identity; fully occluded is black.
        approx3(multi_bounce_ao(1.0, [0.5, 0.5, 0.5]), [1.0, 1.0, 1.0]);
        approx3(multi_bounce_ao(0.0, [0.5, 0.5, 0.5]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn multi_bounce_ao_never_darker_than_scalar() {
        for &ao in &[0.1_f32, 0.3, 0.6, 0.9] {
            let m = multi_bounce_ao(ao, [0.8, 0.2, 0.05]);
            for c in 0..3 {
                assert!(m[c] >= ao - 1.0e-4, "channel {c}: {} < {ao}", m[c]);
                assert!(m[c] <= 1.0 + 1.0e-4);
            }
        }
    }

    #[test]
    fn specular_occlusion_bounds_and_monotonicity() {
        // Fully occluded diffuse -> fully occluded specular.
        approx(specular_occlusion(0.5, 0.0, 0.5), 0.0);
        // Fully open -> fully visible.
        approx(specular_occlusion(1.0, 1.0, 0.5), 1.0);
        // Rougher surface recovers at least as much reflection for equal AO.
        let smooth = specular_occlusion(0.5, 0.5, 0.0);
        let rough = specular_occlusion(0.5, 0.5, 1.0);
        assert!(rough >= smooth - 1.0e-4, "rough {rough} smooth {smooth}");
        for &r in &[0.0_f32, 0.5, 1.0] {
            let v = specular_occlusion(0.5, 0.4, r);
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn bent_normal_occlusion_cone() {
        let bn = Vec3::Z;
        let cos_ap = ops::cos(core::f32::consts::FRAC_PI_4); // 45-degree cone.
        // Light along the axis is fully visible (scaled by visibility).
        approx(bent_normal_occlusion(bn, cos_ap, 1.0, Vec3::Z), 1.0);
        // Light perpendicular (outside the cone) is occluded.
        approx(bent_normal_occlusion(bn, cos_ap, 1.0, Vec3::X), 0.0);
        // Degenerate directions fall back to scalar visibility.
        approx(bent_normal_occlusion(Vec3::ZERO, cos_ap, 0.7, Vec3::Z), 0.7);
    }

    #[test]
    fn compose_is_energy_conserving_when_unoccluded() {
        let input = CompositeInput {
            direct: [0.5, 0.4, 0.3],
            diffuse_gi: [0.2, 0.1, 0.05],
            specular_gi: [0.1, 0.1, 0.1],
            albedo: [0.8, 0.8, 0.8],
            ao: 1.0,
            specular_occlusion: 1.0,
        };
        let out = compose_lighting(input);
        approx3(out, [0.8, 0.6, 0.45]);
    }

    #[test]
    fn compose_attenuates_indirect_but_not_direct() {
        let occ = CompositeInput {
            direct: [1.0, 1.0, 1.0],
            diffuse_gi: [1.0, 1.0, 1.0],
            specular_gi: [1.0, 1.0, 1.0],
            albedo: [0.0, 0.0, 0.0],
            ao: 0.0,
            specular_occlusion: 0.0,
        };
        // Black albedo + zero AO kills indirect entirely; direct survives.
        approx3(compose_lighting(occ), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn exposure_scales_and_floors() {
        approx3(apply_exposure([2.0, 4.0, 8.0], 0.5), [1.0, 2.0, 4.0]);
        approx3(apply_exposure([2.0, 4.0, 8.0], -1.0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn clamp_output_handles_non_finite() {
        let out = clamp_output([f32::NAN, f32::INFINITY, -5.0], 10.0);
        approx3(out, [0.0, 10.0, 0.0]); // NaN->0, +inf->ceiling, -5->0
        // Bad ceiling falls back to the HDR default.
        let big = clamp_output([1.0e9, 0.0, 0.0], -1.0);
        approx(big[0], DEFAULT_HDR_CEILING);
    }

    #[test]
    fn mix_bloom_endpoints() {
        let scene = [0.2, 0.4, 0.6];
        let bloom = [1.0, 1.0, 1.0];
        approx3(mix_bloom(scene, bloom, 0.0), scene);
        approx3(mix_bloom(scene, bloom, 1.0), bloom);
    }

    #[test]
    fn composite_frame_pipeline() {
        let input = CompositeInput {
            direct: [1.0, 1.0, 1.0],
            diffuse_gi: [0.0, 0.0, 0.0],
            specular_gi: [0.0, 0.0, 0.0],
            albedo: [1.0, 1.0, 1.0],
            ao: 1.0,
            specular_occlusion: 1.0,
        };
        // Exposure 0.5, no bloom -> 0.5; clamp keeps it.
        let out = composite_frame(input, 0.5, [0.0, 0.0, 0.0], 0.0, DEFAULT_HDR_CEILING);
        approx3(out, [0.5, 0.5, 0.5]);
        // Full bloom overlay replaces the scene (then exposure already applied).
        let out2 = composite_frame(input, 1.0, [2.0, 2.0, 2.0], 1.0, DEFAULT_HDR_CEILING);
        approx3(out2, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn composite_frame_never_nan() {
        let input = CompositeInput {
            direct: [f32::INFINITY, 0.0, 0.0],
            ..CompositeInput::default()
        };
        let out = composite_frame(input, 1.0, [0.0, 0.0, 0.0], 0.0, 100.0);
        for c in out {
            assert!(c.is_finite());
        }
    }
}
