//! Stylized (non-photoreal / NPR) direct-lighting front end.
//!
//! This is the backend-neutral golden reference for the [`Stylized`] illumination
//! axis (`MaterialShadingClass::Npr`).  It is deliberately structured like the
//! other one-lobe modules (`cloth`, `hair`, `clearcoat`, ...) so the stylized
//! response is a first-class front end rather than a single special-case branch
//! buried in the lighting integrator.  It backs the `Stylized` illumination
//! axis (`prism_render_material::Illumination::Stylized`).
//!
//! # Feature set
//!
//! A production "next-gen AAA" stylized character/prop look (Guilty Gear /
//! Genshin / Honkai lineage) is not one effect but a stack of controllable
//! responses.  [`evaluate_stylized_direct`] composes them per analytic light:
//!
//! * **Diffuse ramp** - the cosine term is optionally wrapped (half-Lambert)
//!   and quantized into `bands` cel steps.  `ramp_softness` softens the band
//!   edges so the transition can range from a hard ink line to a smooth toon
//!   gradient.
//! * **Stepped shadow** - the analytic shadow visibility is re-shaped through a
//!   threshold/softness so soft PCF penumbrae read as crisp stylized shadow
//!   shapes instead of a photoreal gradient.
//! * **Stylized specular** - a thresholded blob highlight (the anime "hot
//!   spot") driven by `N.H` sharpened by material gloss, tinted and toggled by
//!   `specular_intensity`.
//! * **Rim / edge light** - a view-facing Fresnel rim gated to the lit
//!   hemisphere so silhouettes catch the key light; it intentionally survives
//!   shadow so back-lit edges stay readable.
//!
//! With [`StylizedParams::legacy_toon`] the advanced lobes are disabled and the
//! output is **byte-for-byte identical** to the historical banded toon lobe, so
//! `evaluate_toon_direct` (and its WESL twin `toon_direct`) remains an exact
//! wrapper.  A dedicated test pins that equivalence.

use bevy_math::ops;

use crate::vecmath::{add, dot, mul, mul_scalar, normalize_or};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;
/// Lowest stylized specular exponent (broad, soft highlight at full roughness).
const MIN_SPECULAR_EXPONENT: f32 = 2.0;
/// Highest stylized specular exponent (tight hot spot at mirror gloss).
const MAX_SPECULAR_EXPONENT: f32 = 256.0;

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

/// Tunable controls for the stylized front end.
///
/// The [`Default`] value reproduces the historical four-band toon lobe: pure
/// Lambert, hard band edges, linear shadow pass-through, and no specular or rim
/// contribution.  Non-default values opt into the richer NPR stack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StylizedParams {
    /// Number of cel quantization bands for the diffuse term (`>= 1`).
    pub bands: u32,
    /// Cosine wrap in `[0, 1]`. `0` is pure Lambert (`max(N.L, 0)`); `0.5` is
    /// the classic half-Lambert that lifts terminator shadows off the surface.
    pub wrap: f32,
    /// Softness of the band transitions in `[0, 1]`. `0` gives hard ink-line
    /// steps; larger values smooth each step toward a continuous ramp.
    pub ramp_softness: f32,
    /// Shadow visibility threshold in `[0, 1]` where the stepped shadow flips.
    pub shadow_threshold: f32,
    /// Half-width of the shadow transition. `0` (with `shadow_threshold == 0`)
    /// preserves the linear analytic visibility; otherwise the visibility is
    /// re-shaped into a stylized hard/soft edge around the threshold.
    pub shadow_softness: f32,
    /// Stylized specular intensity. `0` disables the highlight entirely.
    pub specular_intensity: f32,
    /// Highlight cutoff in `[0, 1]` applied to the sharpened `N.H` response.
    pub specular_threshold: f32,
    /// Half-width of the highlight edge; `0` yields a hard-edged blob.
    pub specular_softness: f32,
    /// Linear tint of the stylized highlight.
    pub specular_color: [f32; 3],
    /// Rim (edge) light intensity. `0` disables the rim.
    pub rim_intensity: f32,
    /// Fresnel exponent controlling how tightly the rim hugs the silhouette.
    pub rim_power: f32,
    /// Linear tint of the rim light.
    pub rim_color: [f32; 3],
}

impl Default for StylizedParams {
    fn default() -> Self {
        Self::legacy_toon(4)
    }
}

impl StylizedParams {
    /// Parameters that reproduce the historical banded toon lobe exactly:
    /// pure Lambert, hard bands, linear shadow, no specular, no rim.
    #[must_use]
    pub const fn legacy_toon(bands: u32) -> Self {
        Self {
            bands,
            wrap: 0.0,
            ramp_softness: 0.0,
            shadow_threshold: 0.0,
            shadow_softness: 0.0,
            specular_intensity: 0.0,
            specular_threshold: 0.5,
            specular_softness: 0.05,
            specular_color: [1.0; 3],
            rim_intensity: 0.0,
            rim_power: 4.0,
            rim_color: [1.0; 3],
        }
    }

    /// Convenience constructor for the legacy path with a chosen band count.
    #[must_use]
    pub const fn with_bands(bands: u32) -> Self {
        Self::legacy_toon(bands)
    }
}

/// Re-shapes the cosine term into a (optionally wrapped) quantized cel ramp.
///
/// With `wrap == 0` and `ramp_softness == 0` this is exactly
/// `floor(max(N.L, 0) * bands) / bands`, matching the historical toon lobe.
fn stylized_ramp(n_dot_l: f32, bands: u32, wrap: f32, softness: f32) -> f32 {
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
    // Blend toward the next band across a window centred on the band edge.
    let frac = scaled - level;
    let upper = ((level + 1.0) / steps).min(1.0);
    let half = (softness * 0.5).clamp(0.0, 0.5);
    let t = smoothstep(0.5 - half, 0.5 + half, frac);
    mix(lower, upper, t)
}

/// Re-shapes the analytic shadow visibility into a stylized step.
///
/// With `shadow_threshold == 0` and `shadow_softness == 0` the clamped analytic
/// visibility passes through unchanged (matching the historical toon lobe).
fn stylized_shadow(visibility: f32, threshold: f32, softness: f32) -> f32 {
    let v = visibility.clamp(0.0, 1.0);
    if softness <= 0.0 {
        if threshold <= 0.0 {
            return v;
        }
        return if v >= threshold { 1.0 } else { 0.0 };
    }
    smoothstep(threshold - softness, threshold + softness, v)
}

/// Evaluates the stylized front end for a single analytic light.
///
/// Returns linear radiance for the light plus the surface emissive term. When
/// the wrapped cosine term is fully dark the diffuse/specular contribution is
/// zero, but a configured rim can still light the silhouette; the emissive term
/// is always folded in exactly once, matching the other lobes so the resolve
/// integrators can accumulate across many lights without double-counting
/// self-illumination.
pub fn evaluate_stylized_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    params: &StylizedParams,
) -> [f32; 3] {
    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let v = normalize_or(frame.view, n);
    let l = normalize_or(light.direction, n);

    let n_dot_l = dot(n, l);
    let lit_gate = n_dot_l.max(0.0);
    let shadow = stylized_shadow(
        light.visibility,
        params.shadow_threshold,
        params.shadow_softness,
    );

    // Diffuse cel ramp modulated by the base color, incident illuminance and
    // stylized shadow.
    let ramp = stylized_ramp(n_dot_l, params.bands, params.wrap, params.ramp_softness);
    let mut color = mul_scalar(mul(surface.base_color, light.illuminance), ramp * shadow);

    // Stylized specular hot spot: a sharpened, thresholded `N.H` blob gated to
    // the lit hemisphere and attenuated by the stylized shadow.
    if params.specular_intensity > 0.0 && lit_gate > 0.0 {
        let h = normalize_or(add(v, l), n);
        let n_dot_h = dot(n, h).max(0.0);
        let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
        let gloss = 1.0 - roughness;
        let exponent = mix(MIN_SPECULAR_EXPONENT, MAX_SPECULAR_EXPONENT, gloss * gloss);
        let raw = ops::powf(n_dot_h, exponent);
        let half = (params.specular_softness * 0.5).clamp(0.0, 0.5);
        let blob = smoothstep(
            params.specular_threshold - half,
            params.specular_threshold + half,
            raw,
        );
        let weight = params.specular_intensity * blob * lit_gate * shadow;
        color = add(
            color,
            mul(mul_scalar(params.specular_color, weight), light.illuminance),
        );
    }

    // Rim / edge light: a view Fresnel gated to the lit hemisphere. It is
    // intentionally *not* shadowed so back-lit silhouettes stay readable.
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

/// Historical banded toon lobe, preserved as an exact wrapper over
/// [`evaluate_stylized_direct`] with [`StylizedParams::legacy_toon`].
///
/// Mirrors the WESL `toon_direct`; kept so existing callers and goldens stay
/// bit-for-bit stable while the richer stylized stack is opted into per
/// material.
pub fn evaluate_toon_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    bands: u32,
) -> [f32; 3] {
    evaluate_stylized_direct(surface, frame, light, &StylizedParams::legacy_toon(bands))
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

    /// The historical banded formula, kept verbatim as the equivalence oracle.
    fn legacy_toon_reference(
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

    #[test]
    fn legacy_params_match_historical_toon_bit_for_bit() {
        for bands in [1u32, 2, 3, 4, 8, 16] {
            for &dir in &[
                [0.0, 1.0, 0.0],
                [0.3, 0.9, 0.1],
                [0.7, 0.5, 0.2],
                [-0.4, 0.8, 0.3],
                [0.0, -1.0, 0.0],
            ] {
                for visibility in [0.0f32, 0.37, 1.0] {
                    let surface = SurfaceSample {
                        base_color: [0.6, 0.4, 0.3],
                        emissive: [0.02, 0.01, 0.05],
                        ..Default::default()
                    };
                    let sample = DirectLightSample {
                        direction: dir,
                        illuminance: [0.9, 1.0, 1.1],
                        visibility,
                    };
                    let expected = legacy_toon_reference(surface, frame(), sample, bands);
                    let actual = evaluate_toon_direct(surface, frame(), sample, bands);
                    assert_eq!(
                        actual, expected,
                        "bands={bands} dir={dir:?} vis={visibility} must match legacy exactly"
                    );
                }
            }
        }
    }

    #[test]
    fn toon_and_pbr_share_light_visibility_contract() {
        // Migrated from lighting.rs: a facing light lights the top band and a
        // fully shadowed light collapses to zero (plus zero emissive here).
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

    #[test]
    fn output_is_finite_and_non_negative_across_the_stack() {
        let params = StylizedParams {
            bands: 3,
            wrap: 0.5,
            ramp_softness: 0.4,
            shadow_threshold: 0.5,
            shadow_softness: 0.1,
            specular_intensity: 1.5,
            specular_threshold: 0.3,
            specular_softness: 0.1,
            specular_color: [1.0, 0.9, 0.7],
            rim_intensity: 0.8,
            rim_power: 3.0,
            rim_color: [0.4, 0.6, 1.0],
        };
        for &dir in &[
            [0.0, 1.0, 0.0],
            [0.6, 0.6, 0.5],
            [-0.5, 0.7, 0.5],
            [0.0, -0.5, 0.87],
        ] {
            for &view in &[[0.0, 1.0, 0.0], [0.9, 0.3, 0.3], [0.2, 0.2, 0.95]] {
                for visibility in [0.0f32, 0.5, 1.0] {
                    let f = ShadingFrame { view, ..frame() };
                    let sample = DirectLightSample {
                        direction: dir,
                        illuminance: [1.0, 0.95, 0.9],
                        visibility,
                    };
                    let value = evaluate_stylized_direct(
                        SurfaceSample {
                            base_color: [0.5, 0.4, 0.35],
                            perceptual_roughness: 0.4,
                            ..Default::default()
                        },
                        f,
                        sample,
                        &params,
                    );
                    assert!(
                        value.into_iter().all(|c| c.is_finite() && c >= 0.0),
                        "dir={dir:?} view={view:?} vis={visibility} -> {value:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn ramp_softness_smooths_band_edges() {
        // A direction landing near a band edge: hard bands snap it down, a soft
        // ramp blends it up toward the next band.
        let dir = normalize_or([0.0, 0.62, 0.78], [0.0, 1.0, 0.0]);
        let sample = DirectLightSample {
            direction: dir,
            ..light()
        };
        let surface = SurfaceSample {
            base_color: [1.0; 3],
            ..Default::default()
        };
        let hard =
            evaluate_stylized_direct(surface, frame(), sample, &StylizedParams::legacy_toon(4));
        let soft = evaluate_stylized_direct(
            surface,
            frame(),
            sample,
            &StylizedParams {
                ramp_softness: 0.9,
                ..StylizedParams::legacy_toon(4)
            },
        );
        assert_ne!(hard, soft, "softness must change the ramp near a band edge");
        assert!(
            soft[0] >= hard[0],
            "softening blends up toward the next band"
        );
    }

    #[test]
    fn half_lambert_wrap_lifts_the_terminator() {
        // A grazing light (N.L near 0) is dark under pure Lambert but lifted by
        // the half-Lambert wrap.
        let dir = normalize_or([0.98, 0.2, 0.0], [0.0, 1.0, 0.0]);
        let sample = DirectLightSample {
            direction: dir,
            ..light()
        };
        let surface = SurfaceSample {
            base_color: [1.0; 3],
            ..Default::default()
        };
        // Enough bands that the wrapped cosine climbs into a higher cel step
        // than the near-terminator Lambert term (which snaps to the dark band).
        let lambert =
            evaluate_stylized_direct(surface, frame(), sample, &StylizedParams::legacy_toon(4));
        let wrapped = evaluate_stylized_direct(
            surface,
            frame(),
            sample,
            &StylizedParams {
                wrap: 0.5,
                ..StylizedParams::legacy_toon(4)
            },
        );
        let sum = |c: [f32; 3]| c[0] + c[1] + c[2];
        assert!(
            sum(wrapped) > sum(lambert),
            "half-Lambert must lift the terminator: lambert={lambert:?} wrapped={wrapped:?}"
        );
    }

    #[test]
    fn stepped_shadow_hardens_the_penumbra() {
        // A mid visibility reads as a soft grey under linear pass-through but is
        // pushed to black once a threshold above it is applied.
        let sample = DirectLightSample {
            visibility: 0.4,
            ..light()
        };
        let surface = SurfaceSample {
            base_color: [1.0; 3],
            ..Default::default()
        };
        let linear =
            evaluate_stylized_direct(surface, frame(), sample, &StylizedParams::legacy_toon(4));
        let stepped = evaluate_stylized_direct(
            surface,
            frame(),
            sample,
            &StylizedParams {
                shadow_threshold: 0.6,
                shadow_softness: 0.0,
                ..StylizedParams::legacy_toon(4)
            },
        );
        assert!(linear[0] > 0.0, "linear shadow keeps partial light");
        assert_eq!(
            stepped, [0.0; 3],
            "threshold above visibility cuts to black"
        );
    }

    #[test]
    fn stylized_specular_adds_a_bounded_hot_spot() {
        // Aligned view/light on a smooth surface maximises the blob; enabling
        // the highlight must add energy over the diffuse-only response.
        let surface = SurfaceSample {
            base_color: [0.2; 3],
            perceptual_roughness: 0.1,
            ..Default::default()
        };
        let diffuse_only =
            evaluate_stylized_direct(surface, frame(), light(), &StylizedParams::legacy_toon(4));
        let with_spec = evaluate_stylized_direct(
            surface,
            frame(),
            light(),
            &StylizedParams {
                specular_intensity: 2.0,
                specular_threshold: 0.2,
                specular_softness: 0.1,
                specular_color: [1.0; 3],
                ..StylizedParams::legacy_toon(4)
            },
        );
        let sum = |c: [f32; 3]| c[0] + c[1] + c[2];
        assert!(
            sum(with_spec) > sum(diffuse_only),
            "specular must brighten the aligned hot spot"
        );
    }

    #[test]
    fn rim_light_survives_shadow() {
        // A grazing view over a shadowed light: the diffuse term is gone but the
        // rim still lights the silhouette because it is intentionally unshadowed.
        let grazing_view = normalize_or([0.95, 0.31, 0.0], [0.0, 1.0, 0.0]);
        let f = ShadingFrame {
            view: grazing_view,
            ..frame()
        };
        let sample = DirectLightSample {
            visibility: 0.0,
            ..light()
        };
        let surface = SurfaceSample {
            base_color: [0.3; 3],
            ..Default::default()
        };
        let no_rim = evaluate_stylized_direct(surface, f, sample, &StylizedParams::legacy_toon(4));
        let with_rim = evaluate_stylized_direct(
            surface,
            f,
            sample,
            &StylizedParams {
                rim_intensity: 1.0,
                rim_power: 2.0,
                rim_color: [1.0; 3],
                ..StylizedParams::legacy_toon(4)
            },
        );
        assert_eq!(no_rim, [0.0; 3], "shadowed diffuse is fully dark");
        let sum = |c: [f32; 3]| c[0] + c[1] + c[2];
        assert!(
            sum(with_rim) > 0.0,
            "rim must light the shadowed silhouette"
        );
    }
}
