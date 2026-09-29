//! Stylized (NPR) hair shading front end - anime "angel-ring" strand look.
//!
//! Backend-neutral golden reference for the stylized hair response and the
//! numerical twin of `stylized_hair.wesl` (`stylized_hair_direct`). Where
//! [`crate::hair`] is the physically based Marschner R/TT/TRT strand BSDF, this
//! module is its non-photoreal sibling: the Guilty Gear / Genshin / Honkai
//! lineage of cel-shaded hair whose signature is the sharp, banded
//! `anisotropic` highlight ring (the "angel ring" / tobikke) that slides around
//! the head as the view moves.
//!
//! It is a genuine layered response, not a toy:
//!
//! * **Diffuse ramp** - a wrapped (half-Lambert) cosine quantized into `bands`
//!   cel steps with softenable edges, exactly the ramp math the generic
//!   [`crate::stylized`] front end uses, modulated by the base color.
//! * **Two shifted anisotropic highlights** - a `Kajiya-Kay` strand highlight
//!   (`sin` of the angle between the shifted strand tangent and the half
//!   vector, raised to a sharpness exponent) evaluated twice: a tight white
//!   primary band shifted toward the root and a broader tinted secondary band
//!   shifted toward the tip (Scheuermann's two-tone hair specular). Each is
//!   thresholded through a `smoothstep` so it reads as a crisp ink-edged ring
//!   rather than a photoreal falloff.
//! * **Rim / edge light** - a view Fresnel gated to the lit hemisphere so
//!   silhouettes catch the key light; it deliberately survives shadow so
//!   back-lit hair keeps a readable outline.
//!
//! The strand direction is the shading-frame `tangent`; the highlight shift is
//! along the surface `normal`, matching how real-time anime hair rigs bias the
//! tangent to place the ring. Emissive is folded in exactly once, matching the
//! other lobes so the resolve integrator can accumulate over many lights
//! without multiplying self-illumination. Every operation maps one-to-one onto
//! a WESL builtin, keeping this reference byte-for-byte in step with the GPU
//! twin.

use bevy_math::ops;

use crate::vecmath::{add, dot, mul, mul_scalar, normalize_or};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;

/// Scalar linear interpolation, matching the GPU `mix` intrinsic.
#[inline]
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Hermite `smoothstep`, matching the GPU `smoothstep` intrinsic. Returns `0`
/// below `edge0`, `1` above `edge1`, and a smooth cubic in between. A collapsed
/// or inverted edge pair degrades to a hard step at `edge0`.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Tunable controls for the stylized hair front end.
///
/// The [`Default`] value is a production anime hair preset (four-band
/// half-Lambert ramp, a tight white primary ring and a broad tinted secondary
/// ring, plus a soft rim). [`StylizedHairParams::matte`] disables both
/// highlights and the rim for a pure cel-ramp look, which is the analytic
/// baseline the tests pin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StylizedHairParams {
    /// Number of cel quantization bands for the diffuse term (`>= 1`).
    pub bands: u32,
    /// Cosine wrap in `[0, 1]`. `0` is pure Lambert; `0.5` is half-Lambert.
    pub wrap: f32,
    /// Softness of the diffuse band transitions in `[0, 1]`.
    pub ramp_softness: f32,
    /// Tangent shift of the primary highlight along the normal (root-ward when
    /// negative), placing the primary ring.
    pub primary_shift: f32,
    /// Sharpness exponent of the primary highlight (larger => tighter ring).
    pub primary_exponent: f32,
    /// Intensity of the primary highlight; `0` disables it.
    pub primary_intensity: f32,
    /// Threshold in `[0, 1]` where the primary ring flips on.
    pub primary_threshold: f32,
    /// Half-width of the primary ring edge; `0` yields a hard ink edge.
    pub primary_softness: f32,
    /// Linear tint of the primary highlight (classically near-white).
    pub primary_color: [f32; 3],
    /// Tangent shift of the secondary highlight along the normal (tip-ward).
    pub secondary_shift: f32,
    /// Sharpness exponent of the secondary highlight.
    pub secondary_exponent: f32,
    /// Intensity of the secondary highlight; `0` disables it.
    pub secondary_intensity: f32,
    /// Threshold in `[0, 1]` where the secondary ring flips on.
    pub secondary_threshold: f32,
    /// Half-width of the secondary ring edge.
    pub secondary_softness: f32,
    /// Linear tint of the secondary highlight (classically hair-toned).
    pub secondary_color: [f32; 3],
    /// Rim (edge) light intensity; `0` disables the rim.
    pub rim_intensity: f32,
    /// Fresnel exponent controlling how tightly the rim hugs the silhouette.
    pub rim_power: f32,
    /// Linear tint of the rim light.
    pub rim_color: [f32; 3],
}

impl Default for StylizedHairParams {
    fn default() -> Self {
        Self {
            bands: 4,
            wrap: 0.5,
            ramp_softness: 0.1,
            primary_shift: -0.06,
            primary_exponent: 96.0,
            primary_intensity: 1.0,
            primary_threshold: 0.35,
            primary_softness: 0.08,
            primary_color: [1.0, 1.0, 1.0],
            secondary_shift: 0.1,
            secondary_exponent: 24.0,
            secondary_intensity: 0.5,
            secondary_threshold: 0.25,
            secondary_softness: 0.2,
            secondary_color: [0.6, 0.45, 0.3],
            rim_intensity: 0.3,
            rim_power: 4.0,
            rim_color: [1.0, 1.0, 1.0],
        }
    }
}

impl StylizedHairParams {
    /// A pure cel-ramp preset: both highlight rings and the rim are disabled, so
    /// the output is exactly the diffuse ramp times base color, illuminance and
    /// shadow. This is the analytic baseline for regression tests.
    #[must_use]
    pub const fn matte(bands: u32) -> Self {
        Self {
            bands,
            wrap: 0.5,
            ramp_softness: 0.0,
            primary_shift: 0.0,
            primary_exponent: 64.0,
            primary_intensity: 0.0,
            primary_threshold: 0.5,
            primary_softness: 0.05,
            primary_color: [1.0, 1.0, 1.0],
            secondary_shift: 0.0,
            secondary_exponent: 16.0,
            secondary_intensity: 0.0,
            secondary_threshold: 0.5,
            secondary_softness: 0.1,
            secondary_color: [1.0, 1.0, 1.0],
            rim_intensity: 0.0,
            rim_power: 4.0,
            rim_color: [1.0, 1.0, 1.0],
        }
    }
}

/// Re-shapes the cosine term into an optionally wrapped, quantized cel ramp.
///
/// With `wrap == 0` and `softness == 0` this is `floor(max(N.L, 0) * bands) /
/// bands`, matching the generic stylized front end so the whole engine bands
/// diffuse identically.
fn hair_ramp(n_dot_l: f32, bands: u32, wrap: f32, softness: f32) -> f32 {
    let steps = bands.max(1) as f32;
    let wrap = wrap.clamp(0.0, 1.0);
    let lit = if wrap > 0.0 {
        ((n_dot_l + wrap) / (1.0 + wrap)).clamp(0.0, 1.0)
    } else {
        n_dot_l.max(0.0)
    };
    let scaled = lit * steps;
    let level = ops::floor(scaled);
    let lower = level / steps;
    if softness <= 0.0 {
        return lower;
    }
    let frac = scaled - level;
    let upper = ((level + 1.0) / steps).min(1.0);
    let half = (softness * 0.5).clamp(0.0, 0.5);
    let t = smoothstep(0.5 - half, 0.5 + half, frac);
    mix(lower, upper, t)
}

/// One shifted, thresholded `Kajiya-Kay` anisotropic highlight band.
///
/// The strand tangent is biased along the normal by `shift`, the `sin` of the
/// angle between that shifted tangent and the half vector is raised to
/// `exponent`, and the result is pushed through a `smoothstep` threshold so the
/// highlight reads as a crisp anime ring. Returns the scalar band weight in
/// `[0, 1]` before tint/intensity.
fn hair_highlight_band(
    tangent: [f32; 3],
    normal: [f32; 3],
    half: [f32; 3],
    shift: f32,
    exponent: f32,
    threshold: f32,
    softness: f32,
) -> f32 {
    let shifted = normalize_or(add(tangent, mul_scalar(normal, shift)), tangent);
    let t_dot_h = dot(shifted, half).clamp(-1.0, 1.0);
    let sin_t_h = (1.0 - t_dot_h * t_dot_h).max(0.0).sqrt();
    let raw = ops::powf(sin_t_h, exponent.max(1.0));
    let edge = (softness * 0.5).clamp(0.0, 0.5);
    smoothstep(threshold - edge, threshold + edge, raw)
}

/// Evaluates the stylized hair front end for a single analytic light.
///
/// Returns linear radiance for the light plus the surface emissive term. The
/// diffuse ramp and highlight rings are gated to the lit hemisphere and
/// attenuated by the clamped analytic visibility; the rim survives shadow so
/// back-lit silhouettes stay readable. Emissive is folded in exactly once.
pub fn evaluate_stylized_hair_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    params: &StylizedHairParams,
) -> [f32; 3] {
    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let v = normalize_or(frame.view, n);
    let l = normalize_or(light.direction, n);
    let t = normalize_or(frame.tangent, [1.0, 0.0, 0.0]);

    let visibility = light.visibility.clamp(0.0, 1.0);
    let n_dot_l = dot(n, l);
    let lit_gate = n_dot_l.max(0.0);

    // Diffuse cel ramp modulated by base color, illuminance and visibility.
    let ramp = hair_ramp(n_dot_l, params.bands, params.wrap, params.ramp_softness);
    let mut color = mul_scalar(
        mul(surface.base_color, light.illuminance),
        ramp * visibility,
    );

    if lit_gate > 0.0 {
        let h = normalize_or(add(v, l), n);
        // A glossier strand tightens both rings, matching the PBR sibling.
        let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
        let gloss = 1.0 - roughness;

        // Primary white ring, shifted toward the root.
        if params.primary_intensity > 0.0 {
            let exponent = params.primary_exponent * mix(0.5, 1.0, gloss);
            let band = hair_highlight_band(
                t,
                n,
                h,
                params.primary_shift,
                exponent,
                params.primary_threshold,
                params.primary_softness,
            );
            let weight = params.primary_intensity * band * lit_gate * visibility;
            color = add(
                color,
                mul(mul_scalar(params.primary_color, weight), light.illuminance),
            );
        }

        // Secondary tinted ring, shifted toward the tip.
        if params.secondary_intensity > 0.0 {
            let exponent = params.secondary_exponent * mix(0.5, 1.0, gloss);
            let band = hair_highlight_band(
                t,
                n,
                h,
                params.secondary_shift,
                exponent,
                params.secondary_threshold,
                params.secondary_softness,
            );
            let weight = params.secondary_intensity * band * lit_gate * visibility;
            let tint = mul(params.secondary_color, surface.base_color);
            color = add(color, mul(mul_scalar(tint, weight), light.illuminance));
        }
    }

    // Rim / edge light: a view Fresnel gated to the lit hemisphere, unshadowed.
    if params.rim_intensity > 0.0 && lit_gate > 0.0 {
        let n_dot_v = dot(n, v).max(0.0);
        let rim = ops::powf((1.0 - n_dot_v).max(0.0), params.rim_power.max(0.0));
        let weight = params.rim_intensity * rim * lit_gate;
        color = add(
            color,
            mul(mul_scalar(params.rim_color, weight), light.illuminance),
        );
    }

    add(color, surface.emissive)
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
            base_color: [0.4, 0.25, 0.1],
            perceptual_roughness: 0.4,
            ..Default::default()
        }
    }

    fn approx(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
    }

    #[test]
    fn matte_preset_is_pure_ramp() {
        // With highlights and rim disabled the output is exactly ramp * base *
        // illuminance * visibility (+ emissive, which is zero here).
        let s = base_surface();
        let f = frame();
        let li = light();
        let params = StylizedHairParams::matte(4);
        let got = evaluate_stylized_hair_direct(s, f, li, &params);

        let n_dot_l = 1.0f32;
        let ramp = hair_ramp(n_dot_l, 4, 0.5, 0.0);
        let want = [
            s.base_color[0] * ramp,
            s.base_color[1] * ramp,
            s.base_color[2] * ramp,
        ];
        assert!(approx(got, want, 1.0e-6), "got {got:?} want {want:?}");
    }

    #[test]
    fn ramp_quantizes_into_bands() {
        // A matte four-band ramp only ever emits multiples of base/4 (times the
        // ramp levels), so the luminance is one of a small discrete set.
        let s = base_surface();
        let params = StylizedHairParams::matte(4);
        let mut seen_levels: Vec<f32> = Vec::new();
        for i in 0..=16 {
            let angle = core::f32::consts::FRAC_PI_2 * (i as f32) / 16.0;
            let l = [ops::sin(angle), ops::cos(angle), 0.0];
            let li = DirectLightSample {
                direction: l,
                illuminance: [1.0; 3],
                visibility: 1.0,
            };
            let got = evaluate_stylized_hair_direct(s, frame(), li, &params);
            // Recover the ramp level from the red channel.
            let level = got[0] / s.base_color[0];
            if !seen_levels.iter().any(|v: &f32| (v - level).abs() < 1.0e-5) {
                seen_levels.push(level);
            }
        }
        // Four bands with half-Lambert wrap can visit at most five discrete
        // levels (0/4..4/4); never a continuous gradient.
        assert!(
            seen_levels.len() <= 5,
            "ramp not quantized: {seen_levels:?}"
        );
    }

    #[test]
    fn highlight_adds_energy_over_matte() {
        // The default anime preset must be strictly brighter than the matte
        // baseline at the specular peak (the rings add energy).
        let s = base_surface();
        let f = frame();
        let li = light();
        let matte = evaluate_stylized_hair_direct(s, f, li, &StylizedHairParams::matte(4));
        let styled = evaluate_stylized_hair_direct(s, f, li, &StylizedHairParams::default());
        assert!(
            styled[0] >= matte[0] && (styled[0] - matte[0]).abs() > 1.0e-4,
            "highlight added no energy: styled {styled:?} matte {matte:?}"
        );
    }

    #[test]
    fn occluded_light_returns_only_emissive() {
        // Zero visibility kills diffuse and both rings; only emissive survives.
        let mut s = base_surface();
        s.emissive = [0.05, 0.02, 0.01];
        let li = DirectLightSample {
            direction: [0.0, 1.0, 0.0],
            illuminance: [1.0; 3],
            visibility: 0.0,
        };
        let got = evaluate_stylized_hair_direct(s, frame(), li, &StylizedHairParams::default());
        assert!(approx(got, s.emissive, 1.0e-6), "got {got:?}");
    }

    #[test]
    fn back_hemisphere_has_no_diffuse_or_highlight() {
        // A light fully behind the surface leaves only emissive: the wrapped
        // ramp floors to zero and both rings are gated off.
        let s = base_surface();
        let li = DirectLightSample {
            direction: [0.0, -1.0, 0.0],
            illuminance: [1.0; 3],
            visibility: 1.0,
        };
        let got = evaluate_stylized_hair_direct(s, frame(), li, &StylizedHairParams::default());
        assert!(approx(got, s.emissive, 1.0e-6), "got {got:?}");
    }

    #[test]
    fn deterministic_across_calls() {
        // The closure is pure: identical inputs yield byte-identical outputs.
        let s = base_surface();
        let f = frame();
        let li = light();
        let params = StylizedHairParams::default();
        let a = evaluate_stylized_hair_direct(s, f, li, &params);
        let b = evaluate_stylized_hair_direct(s, f, li, &params);
        assert_eq!(a, b);
    }
}
