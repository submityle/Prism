//! Stylized (NPR) multi-ring "angel ring" highlight stack for anime hair.
//!
//! [`crate::stylized_hair`] ships a fixed two-ring anime hair response (a tight
//! white primary band plus a broad tinted secondary band). Top-tier cel-shaded
//! hair in the Guilty Gear / Genshin / Honkai lineage layers *more* than two
//! bands: a sharp core ring, one or two softer satellite rings, and often a
//! low, wide sheen. This module owns that **N-ring extension** as its own
//! concern: an ordered stack of independently tuned [`AngelRing`] bands whose
//! additive contribution composes on top of the diffuse cel ramp.
//!
//! Each ring is a shifted, thresholded `Kajiya-Kay` anisotropic highlight: the
//! `sin` of the angle between a (shifted) strand tangent and the half vector is
//! raised to a sharpness exponent and pushed through a `smoothstep` threshold so
//! it reads as a crisp ink-edged band rather than a photoreal falloff. The band
//! math is deliberately value-for-value identical to
//! [`crate::stylized_hair`]'s `hair_highlight_band` at `tangent_decouple == 0`,
//! so the two modules stay coherent and a future shared `WESL` twin can mirror
//! both.
//!
//! **Tangent decoupling** is the signature of a true "angel ring": the band is
//! not rigidly locked to the local strand flow but biased toward a fixed
//! reference axis (typically the head-up direction expressed in the shading
//! frame) so the ring sits as a stable horizontal halo across the whole head
//! regardless of how individual strands curl. `tangent_decouple` in `[0, 1]`
//! blends the strand tangent toward that reference axis; `0` follows the strand
//! (matching the base module), `1` fully pins the band to the reference axis.
//!
//! The stack is a pure, deterministic function: array in, color out. Nothing
//! samples a real random source and no input panics — a zero-length reference
//! axis falls back to the strand tangent, degenerate exponents/softness are
//! clamped, disabled rings (`intensity == 0`) contribute nothing, and an empty
//! stack returns black. Emissive and the diffuse ramp are intentionally *not*
//! folded in here: this module returns only the additive highlight radiance for
//! one analytic light so the resolve integrator (or
//! [`crate::stylized_hair`]) can accumulate rings across many lights without
//! double-counting the base shade.

use bevy_math::ops;

use crate::vecmath::{add, dot, mix3, mul_scalar, normalize_or};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Roughness floor shared with the principled/hair lobes (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;

/// Maximum rings a caller is expected to stack; the evaluator itself accepts any
/// slice length, but presets and `GPU`-facing uniforms size to this bound.
pub const MAX_ANGEL_RINGS: usize = 8;

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

/// One stylized anisotropic highlight band in the angel-ring stack.
///
/// A ring is placed by shifting the (optionally decoupled) strand tangent along
/// the surface normal by `shift`, then thresholding a `Kajiya-Kay` `sin` power
/// through a `smoothstep`. `color` is the linear tint applied to the band
/// weight; `intensity` scales it and `0` disables the ring entirely.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngelRing {
    /// Tangent shift along the normal that places the band (root-ward when
    /// negative, tip-ward when positive).
    pub shift: f32,
    /// Sharpness exponent (larger => tighter band); effectively `>= 1`.
    pub exponent: f32,
    /// Band intensity; `0` disables this ring.
    pub intensity: f32,
    /// `smoothstep` center in `[0, 1]` where the band flips on.
    pub threshold: f32,
    /// Half-width of the band edge; `0` yields a hard ink edge.
    pub softness: f32,
    /// Linear tint of the band.
    pub color: [f32; 3],
    /// Blend of the strand tangent toward the reference axis in `[0, 1]`. `0`
    /// follows the strand flow; `1` pins the band to the reference axis (a
    /// stable head-space halo).
    pub tangent_decouple: f32,
}

impl AngelRing {
    /// A disabled ring (zero intensity), useful as a fixed-size array filler.
    pub const DISABLED: AngelRing = AngelRing {
        shift: 0.0,
        exponent: 16.0,
        intensity: 0.0,
        threshold: 0.5,
        softness: 0.1,
        color: [1.0, 1.0, 1.0],
        tangent_decouple: 0.0,
    };

    /// Returns a copy with all controls clamped into their valid ranges. The
    /// evaluator applies the same clamps internally, so calling this is
    /// optional; it exists for callers that want to store normalized presets.
    #[must_use]
    pub fn sanitized(self) -> AngelRing {
        AngelRing {
            shift: self.shift,
            exponent: self.exponent.max(1.0),
            intensity: self.intensity.max(0.0),
            threshold: self.threshold.clamp(0.0, 1.0),
            softness: self.softness.clamp(0.0, 1.0),
            color: [
                self.color[0].max(0.0),
                self.color[1].max(0.0),
                self.color[2].max(0.0),
            ],
            tangent_decouple: self.tangent_decouple.clamp(0.0, 1.0),
        }
    }
}

/// Evaluates a single ring's scalar band weight in `[0, 1]`, before tint and
/// intensity.
///
/// `tangent` is the strand direction, `reference_axis` the fixed band axis (a
/// zero-length axis falls back to `tangent`), `normal` the surface normal, and
/// `half` the view/light half vector. `gloss` in `[0, 1]` tightens the band on
/// glossier strands, matching the base module. At `tangent_decouple == 0` and
/// `gloss` applied identically, this reproduces
/// [`crate::stylized_hair`]'s band value-for-value.
#[must_use]
pub fn angel_ring_band(
    ring: AngelRing,
    tangent: [f32; 3],
    reference_axis: [f32; 3],
    normal: [f32; 3],
    half: [f32; 3],
    gloss: f32,
) -> f32 {
    let decouple = ring.tangent_decouple.clamp(0.0, 1.0);
    let axis = normalize_or(reference_axis, tangent);
    // Bias the strand tangent toward the reference axis; `decouple == 0` returns
    // the strand tangent exactly (mix factor 0), keeping parity with the base
    // two-ring module.
    let base = normalize_or(mix3(tangent, axis, decouple), tangent);
    let shifted = normalize_or(add(base, mul_scalar(normal, ring.shift)), base);
    let t_dot_h = dot(shifted, half).clamp(-1.0, 1.0);
    let sin_t_h = (1.0 - t_dot_h * t_dot_h).max(0.0).sqrt();
    let exponent = (ring.exponent * mix(0.5, 1.0, gloss.clamp(0.0, 1.0))).max(1.0);
    let raw = ops::powf(sin_t_h, exponent);
    let edge = (ring.softness * 0.5).clamp(0.0, 0.5);
    smoothstep(ring.threshold - edge, ring.threshold + edge, raw)
}

/// Accumulates the additive highlight radiance of an angel-ring stack for one
/// analytic light.
///
/// Returns only the summed ring contribution (linear RGB); the caller adds it
/// on top of the diffuse cel ramp and emissive. The whole stack is gated to the
/// lit hemisphere and attenuated by the clamped analytic visibility, so a fully
/// occluded or back-lit light yields black. Rings are summed in slice order;
/// disabled rings are skipped. An empty slice returns `[0, 0, 0]`.
#[must_use]
pub fn accumulate_angel_rings(
    rings: &[AngelRing],
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    reference_axis: [f32; 3],
) -> [f32; 3] {
    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let v = normalize_or(frame.view, n);
    let l = normalize_or(light.direction, n);
    let t = normalize_or(frame.tangent, [1.0, 0.0, 0.0]);

    let visibility = light.visibility.clamp(0.0, 1.0);
    let lit_gate = dot(n, l).max(0.0);
    if lit_gate <= 0.0 || visibility <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let h = normalize_or(add(v, l), n);
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let gloss = 1.0 - roughness;

    let mut color = [0.0, 0.0, 0.0];
    for ring in rings {
        if ring.intensity <= 0.0 {
            continue;
        }
        let band = angel_ring_band(*ring, t, reference_axis, n, h, gloss);
        let weight = ring.intensity.max(0.0) * band * lit_gate * visibility;
        if weight <= 0.0 {
            continue;
        }
        // Tint the band weight and fold in the analytic light color in one
        // step, then accumulate additively across the stack.
        let tinted = [
            ring.color[0].max(0.0) * weight * light.illuminance[0],
            ring.color[1].max(0.0) * weight * light.illuminance[1],
            ring.color[2].max(0.0) * weight * light.illuminance[2],
        ];
        color = add(color, tinted);
    }
    color
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

    fn bits_eq(a: [f32; 3], b: [f32; 3]) -> bool {
        a[0].to_bits() == b[0].to_bits()
            && a[1].to_bits() == b[1].to_bits()
            && a[2].to_bits() == b[2].to_bits()
    }

    /// A ring that lands squarely on the peak (`sin(T, H) == 1`) for the
    /// axis-aligned test frame, so the band evaluates to exactly `1`.
    fn peak_ring() -> AngelRing {
        AngelRing {
            shift: 0.0,
            exponent: 32.0,
            intensity: 0.7,
            threshold: 0.5,
            softness: 0.1,
            color: [0.2, 0.4, 0.6],
            tangent_decouple: 0.0,
        }
    }

    #[test]
    fn empty_stack_is_black() {
        let got = accumulate_angel_rings(&[], base_surface(), frame(), light(), [0.0, 1.0, 0.0]);
        assert!(approx(got, [0.0, 0.0, 0.0], 1e-6));
    }

    #[test]
    fn single_peak_ring_matches_hand_computed_radiance() {
        // Axis-aligned frame: tangent = +X, normal/view/light = +Y, so the half
        // vector is +Y, dot(T, H) = 0, sin(T, H) = 1, raw = 1 >= threshold, band
        // = 1. lit_gate = dot(N, L) = 1, visibility = 1, illuminance = 1, so the
        // result is exactly ring.color * intensity.
        let ring = peak_ring();
        let got =
            accumulate_angel_rings(&[ring], base_surface(), frame(), light(), [0.0, 1.0, 0.0]);
        let expected = [
            ring.color[0] * ring.intensity,
            ring.color[1] * ring.intensity,
            ring.color[2] * ring.intensity,
        ];
        assert!(
            approx(got, expected, 1e-6),
            "got {got:?} expected {expected:?}"
        );
    }

    #[test]
    fn band_peaks_at_one_on_axis() {
        // The scalar band alone is exactly 1 at the peak.
        let band = angel_ring_band(
            peak_ring(),
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            0.6,
        );
        assert!((band - 1.0).abs() < 1e-6, "band = {band}");
    }

    #[test]
    fn disabled_ring_contributes_nothing() {
        let mut ring = peak_ring();
        ring.intensity = 0.0;
        let got =
            accumulate_angel_rings(&[ring], base_surface(), frame(), light(), [0.0, 1.0, 0.0]);
        assert!(approx(got, [0.0, 0.0, 0.0], 1e-6));
    }

    #[test]
    fn stack_is_additive() {
        // A two-ring stack equals the sum of each ring evaluated alone.
        let r0 = peak_ring();
        let r1 = AngelRing {
            shift: 0.05,
            exponent: 12.0,
            intensity: 0.35,
            threshold: 0.3,
            softness: 0.4,
            color: [0.9, 0.2, 0.05],
            tangent_decouple: 0.5,
        };
        let s = base_surface();
        let f = frame();
        let li = light();
        let axis = [0.0, 1.0, 0.0];
        let stacked = accumulate_angel_rings(&[r0, r1], s, f, li, axis);
        let a = accumulate_angel_rings(&[r0], s, f, li, axis);
        let b = accumulate_angel_rings(&[r1], s, f, li, axis);
        let summed = add(a, b);
        assert!(
            approx(stacked, summed, 1e-6),
            "stacked {stacked:?} summed {summed:?}"
        );
    }

    #[test]
    fn zero_visibility_is_black() {
        let mut li = light();
        li.visibility = 0.0;
        let got =
            accumulate_angel_rings(&[peak_ring()], base_surface(), frame(), li, [0.0, 1.0, 0.0]);
        assert!(approx(got, [0.0, 0.0, 0.0], 1e-6));
    }

    #[test]
    fn back_lit_is_black() {
        let mut li = light();
        li.direction = [0.0, -1.0, 0.0];
        let got =
            accumulate_angel_rings(&[peak_ring()], base_surface(), frame(), li, [0.0, 1.0, 0.0]);
        assert!(approx(got, [0.0, 0.0, 0.0], 1e-6));
    }

    #[test]
    fn zero_reference_axis_falls_back_to_tangent() {
        // A zero-length reference axis must not panic; with the axis falling
        // back to the tangent, any decouple factor collapses to the strand-flow
        // band (decouple == 0).
        let mut ring = peak_ring();
        ring.tangent_decouple = 1.0;
        let with_zero_axis = angel_ring_band(
            ring,
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            0.6,
        );
        let mut coupled = ring;
        coupled.tangent_decouple = 0.0;
        let with_coupled = angel_ring_band(
            coupled,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            0.6,
        );
        assert!((with_zero_axis - with_coupled).abs() < 1e-6);
    }

    #[test]
    fn tangent_decouple_biases_the_band() {
        // With a reference axis distinct from the tangent, decoupling must move
        // the band: a fully decoupled ring differs from a fully coupled one.
        let tangent = [1.0, 0.0, 0.0];
        let axis = normalize_or([0.0, 0.0, 1.0], tangent);
        let half = normalize_or([0.3, 1.0, 0.6], [0.0, 1.0, 0.0]);
        let normal = [0.0, 1.0, 0.0];
        // A broad, low-exponent band keeps the response inside the smoothstep
        // ramp so decoupling the tangent visibly shifts the weight.
        let broad = AngelRing {
            shift: 0.0,
            exponent: 2.0,
            intensity: 0.5,
            threshold: 0.5,
            softness: 1.0,
            color: [1.0, 1.0, 1.0],
            tangent_decouple: 0.0,
        };
        let mut coupled = broad;
        coupled.tangent_decouple = 0.0;
        let mut decoupled = broad;
        decoupled.tangent_decouple = 1.0;
        let b_coupled = angel_ring_band(coupled, tangent, axis, normal, half, 0.6);
        let b_decoupled = angel_ring_band(decoupled, tangent, axis, normal, half, 0.6);
        assert!((b_coupled - b_decoupled).abs() > 1e-4);
    }

    #[test]
    fn tangent_decouple_is_clamped() {
        // An out-of-range decouple factor behaves like its clamped counterpart.
        let tangent = [1.0, 0.0, 0.0];
        let axis = normalize_or([0.0, 0.0, 1.0], tangent);
        let half = normalize_or([0.3, 1.0, 0.6], [0.0, 1.0, 0.0]);
        let normal = [0.0, 1.0, 0.0];
        let mut over = peak_ring();
        over.tangent_decouple = 2.5;
        let mut pinned = peak_ring();
        pinned.tangent_decouple = 1.0;
        let b_over = angel_ring_band(over, tangent, axis, normal, half, 0.6);
        let b_pinned = angel_ring_band(pinned, tangent, axis, normal, half, 0.6);
        assert!((b_over - b_pinned).abs() < 1e-6);
    }

    #[test]
    fn evaluation_is_deterministic() {
        let s = base_surface();
        let f = frame();
        let li = light();
        let axis = [0.0, 1.0, 0.0];
        let rings = [peak_ring(), AngelRing::DISABLED];
        let a = accumulate_angel_rings(&rings, s, f, li, axis);
        let b = accumulate_angel_rings(&rings, s, f, li, axis);
        assert!(bits_eq(a, b));
    }

    #[test]
    fn sanitized_clamps_all_controls() {
        let dirty = AngelRing {
            shift: -3.0,
            exponent: 0.2,
            intensity: -1.0,
            threshold: 1.7,
            softness: 4.0,
            color: [-0.5, 2.0, -0.1],
            tangent_decouple: 5.0,
        };
        let s = dirty.sanitized();
        assert!(s.exponent >= 1.0);
        assert!(s.intensity >= 0.0);
        assert!(s.threshold >= 0.0 && s.threshold <= 1.0);
        assert!(s.softness >= 0.0 && s.softness <= 1.0);
        assert!(s.tangent_decouple >= 0.0 && s.tangent_decouple <= 1.0);
        assert!(s.color[0] >= 0.0 && s.color[1] >= 0.0 && s.color[2] >= 0.0);
        // Shift is unbounded (root-ward / tip-ward placement) and preserved.
        assert!((s.shift - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn disabled_constant_is_inert() {
        let got = accumulate_angel_rings(
            &[AngelRing::DISABLED; MAX_ANGEL_RINGS],
            base_surface(),
            frame(),
            light(),
            [0.0, 1.0, 0.0],
        );
        assert!(approx(got, [0.0, 0.0, 0.0], 1e-6));
    }
}
