//! Near / focus / far layer separation and compositing for depth of field.
//!
//! A single gather pass cannot reproduce real depth of field because the
//! *foreground* behaves differently from the background: a near, out-of-focus
//! object must bleed its blurred silhouette *over* the sharp mid-ground, with a
//! soft, partially transparent edge, while the background is simply blurred
//! behind the focus plane.  The established fix (McIntosh et al.; Abadie,
//! SIGGRAPH 2018) splits the frame by signed CoC into a near layer, a sharp
//! focus layer, and a far layer, then composites them with premultiplied alpha:
//! the far-blurred colour is dissolved into the sharp layer by its far CoC, and
//! the near-blurred colour is laid *over* that result with a foreground alpha
//! derived from the near CoC.
//!
//! This module is the backend-neutral CPU reference for that layering and
//! `over` compositing.  The per-layer blur itself is produced elsewhere (see
//! [`super::gather`]); here we only derive the coverage alphas and combine the
//! already-blurred layers.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, I/O, GPU, allocation, or `unsafe`.
//! * Signed, normalised CoC in `[-1, 1]` drives the split (see
//!   [`super::coc::LensParams::coc_normalized`]): negative = near/foreground,
//!   positive = far/background, zero = sharp focus.
//! * Alpha is premultiplied.  A premultiplied pixel is a `Vec4` whose `xyz` is
//!   `colour * alpha` and whose `w` is `alpha`; this makes `over` a plain
//!   `top + bottom * (1 - top.a)`.
//! * No transcendental functions are required; the smootherstep edge is
//!   polynomial, so there is no [`bevy_math::ops`] dependency.
//! * Defensive clamping everywhere: alphas are clamped to `[0, 1]`, colours to
//!   the finite non-negative octant, and no path emits `NaN`/`inf`.

use bevy_math::{Vec3, Vec4};

/// Tunable thresholds for the near-layer foreground alpha ramp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerParams {
    /// Normalised near-CoC magnitude at which the foreground becomes fully
    /// opaque (alpha `1`).  In `(0, 1]`.
    ///
    /// Below this the alpha ramps up from `0`, giving the soft foreground edge.
    pub near_full_coc: f32,
    /// Normalised far-CoC at which the background is fully taken from the
    /// blurred far layer rather than the sharp focus layer.  In `(0, 1]`.
    pub far_full_coc: f32,
}

impl Default for LayerParams {
    #[inline]
    fn default() -> Self {
        Self {
            near_full_coc: 0.5,
            far_full_coc: 0.5,
        }
    }
}

impl LayerParams {
    #[inline]
    fn sanitized(self) -> Self {
        #[inline]
        fn pos_unit(x: f32, fallback: f32) -> f32 {
            if x.is_finite() && x > 0.0 {
                x.min(1.0)
            } else {
                fallback
            }
        }
        Self {
            near_full_coc: pos_unit(self.near_full_coc, 0.5),
            far_full_coc: pos_unit(self.far_full_coc, 0.5),
        }
    }
}

/// Clamps a colour to the finite non-negative octant.
#[inline]
fn sanitize_color(c: Vec3) -> Vec3 {
    Vec3::new(
        if c.x.is_finite() { c.x.max(0.0) } else { 0.0 },
        if c.y.is_finite() { c.y.max(0.0) } else { 0.0 },
        if c.z.is_finite() { c.z.max(0.0) } else { 0.0 },
    )
}

/// Clamps a scalar to `[0, 1]`, mapping non-finite input to `0`.
#[inline]
fn sanitize_unit(x: f32) -> f32 {
    if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }
}

/// Smootherstep `6t^5 - 15t^4 + 10t^3` on `[0, 1]`, clamped at the ends.
///
/// Used for a C2-continuous foreground edge so the near silhouette has no
/// visible banding where it fades into transparency.
#[inline]
fn smootherstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Foreground (near-layer) alpha from a signed, normalised CoC.
///
/// Only the *near* side (negative CoC) produces coverage: the magnitude is
/// mapped through [`smootherstep`] so alpha is `0` at the focus plane, ramps up
/// across `[0, near_full_coc]`, and saturates at `1`.  A far or in-focus pixel
/// (CoC `>= 0`) has zero foreground alpha.
pub fn foreground_alpha(coc_normalized: f32, params: LayerParams) -> f32 {
    let params = params.sanitized();
    let c = if coc_normalized.is_finite() { coc_normalized } else { 0.0 };
    let near_mag = (-c).max(0.0); // only the negative (near) side counts
    smootherstep(near_mag / params.near_full_coc)
}

/// Background blend factor from a signed, normalised CoC.
///
/// Only the *far* side (positive CoC) produces coverage, ramped through
/// [`smootherstep`] over `[0, far_full_coc]`.  Used to dissolve the blurred far
/// layer into the sharp focus layer.  A near or in-focus pixel yields `0`.
pub fn background_blend(coc_normalized: f32, params: LayerParams) -> f32 {
    let params = params.sanitized();
    let c = if coc_normalized.is_finite() { coc_normalized } else { 0.0 };
    let far_mag = c.max(0.0); // only the positive (far) side counts
    smootherstep(far_mag / params.far_full_coc)
}

/// Packs a straight colour and alpha into a premultiplied `Vec4`.
///
/// Returns `(colour * alpha, alpha)` with both operands sanitised.
#[inline]
pub fn premultiply(color: Vec3, alpha: f32) -> Vec4 {
    let a = sanitize_unit(alpha);
    let c = sanitize_color(color);
    Vec4::new(c.x * a, c.y * a, c.z * a, a)
}

/// Porter-Duff `over`: composites premultiplied `top` onto premultiplied
/// `bottom`.
///
/// With premultiplied inputs this is simply `top + bottom * (1 - top.a)`.  Both
/// alphas are assumed already clamped to `[0, 1]`; the result stays
/// premultiplied and finite.
#[inline]
pub fn over(top: Vec4, bottom: Vec4) -> Vec4 {
    let ta = sanitize_unit(top.w);
    let inv = 1.0 - ta;
    let top = Vec4::new(top.x, top.y, top.z, ta);
    top + bottom * inv
}

/// Converts a premultiplied pixel back to a straight (un-premultiplied) colour.
///
/// Divides `xyz` by `w`; a zero (fully transparent) alpha yields black.  Always
/// finite.
#[inline]
pub fn unpremultiply(p: Vec4) -> Vec3 {
    let a = sanitize_unit(p.w);
    if a <= 0.0 {
        Vec3::ZERO
    } else {
        sanitize_color(Vec3::new(p.x, p.y, p.z) / a)
    }
}

/// One pixel's worth of already-blurred depth-of-field layers.
///
/// `sharp` is the original in-focus colour; `far_blurred` is the background
/// gather result; `near_blurred` is the foreground gather result.  Signed
/// normalised CoC drives how they combine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DofLayers {
    /// Original sharp colour at this pixel.
    pub sharp: Vec3,
    /// Background-layer blurred colour (used where CoC is positive/far).
    pub far_blurred: Vec3,
    /// Foreground-layer blurred colour (used where CoC is negative/near).
    pub near_blurred: Vec3,
    /// Signed, normalised CoC at this pixel in `[-1, 1]`.
    pub coc_normalized: f32,
}

/// Composites the three layers into the final straight-colour pixel.
///
/// Steps, mirroring Abadie's near/far pipeline:
/// 1. Dissolve `far_blurred` into `sharp` by [`background_blend`] to form the
///    opaque "behind" colour.
/// 2. Derive the foreground alpha from [`foreground_alpha`], premultiply
///    `near_blurred` by it, and lay it `over` the behind colour.
///
/// When the pixel is in focus (CoC `0`) both coverages vanish and the result is
/// exactly `sharp`.
pub fn composite(layers: DofLayers, params: LayerParams) -> Vec3 {
    let params = params.sanitized();
    let sharp = sanitize_color(layers.sharp);
    let far = sanitize_color(layers.far_blurred);
    let near = sanitize_color(layers.near_blurred);
    let coc = if layers.coc_normalized.is_finite() {
        layers.coc_normalized.clamp(-1.0, 1.0)
    } else {
        0.0
    };

    // 1. Background: lerp sharp -> far by the far coverage.
    let far_t = background_blend(coc, params);
    let behind = sharp.lerp(far, far_t);

    // 2. Foreground: near layer laid over the opaque background.
    let near_a = foreground_alpha(coc, params);
    let near_pm = premultiply(near, near_a);
    let behind_pm = Vec4::new(behind.x, behind.y, behind.z, 1.0);
    let out = over(near_pm, behind_pm);
    // Background alpha is 1, so the result is already opaque; return its colour.
    sanitize_color(Vec3::new(out.x, out.y, out.z))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn approx_vec(a: Vec3, b: Vec3, eps: f32) -> bool {
        approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps)
    }

    #[test]
    fn alpha_stays_in_unit_range() {
        let p = LayerParams::default();
        for k in -20..=20 {
            let coc = k as f32 / 20.0;
            let fa = foreground_alpha(coc, p);
            let bb = background_blend(coc, p);
            assert!((0.0..=1.0).contains(&fa), "fg alpha {fa} out of range");
            assert!((0.0..=1.0).contains(&bb), "bg blend {bb} out of range");
        }
    }

    #[test]
    fn focus_plane_has_no_coverage() {
        let p = LayerParams::default();
        assert!(approx(foreground_alpha(0.0, p), 0.0, 1.0e-7));
        assert!(approx(background_blend(0.0, p), 0.0, 1.0e-7));
    }

    #[test]
    fn only_near_side_drives_foreground() {
        let p = LayerParams::default();
        // Positive (far) CoC must not produce foreground coverage.
        assert_eq!(foreground_alpha(0.8, p), 0.0);
        // Negative (near) CoC must.
        assert!(foreground_alpha(-0.8, p) > 0.0);
    }

    #[test]
    fn only_far_side_drives_background() {
        let p = LayerParams::default();
        assert_eq!(background_blend(-0.8, p), 0.0);
        assert!(background_blend(0.8, p) > 0.0);
    }

    #[test]
    fn foreground_alpha_is_monotonic_in_near_coc() {
        // Deeper foreground blur (more negative CoC) => more coverage.
        let p = LayerParams::default();
        let mut prev = -1.0_f32;
        for k in 0..=20 {
            let coc = -(k as f32 / 20.0);
            let a = foreground_alpha(coc, p);
            assert!(a >= prev - 1.0e-6, "fg alpha not monotonic at {coc}: {a} < {prev}");
            prev = a;
        }
        assert!(approx(prev, 1.0, 1.0e-6), "deep foreground should saturate: {prev}");
    }

    #[test]
    fn all_in_focus_composites_to_sharp() {
        // CoC == 0 everywhere: the composite must return the original colour,
        // untouched by the (irrelevant) blurred layers.
        let layers = DofLayers {
            sharp: Vec3::new(0.2, 0.5, 0.8),
            far_blurred: Vec3::new(1.0, 0.0, 0.0),
            near_blurred: Vec3::new(0.0, 1.0, 0.0),
            coc_normalized: 0.0,
        };
        let out = composite(layers, LayerParams::default());
        assert!(approx_vec(out, layers.sharp, 1.0e-6), "focus composite drifted: {out:?}");
    }

    #[test]
    fn full_near_coverage_shows_foreground() {
        // Saturated near CoC: the near-blurred colour fully covers the pixel.
        let layers = DofLayers {
            sharp: Vec3::new(0.9, 0.9, 0.9),
            far_blurred: Vec3::ZERO,
            near_blurred: Vec3::new(0.1, 0.2, 0.3),
            coc_normalized: -1.0,
        };
        let out = composite(layers, LayerParams::default());
        assert!(approx_vec(out, layers.near_blurred, 1.0e-5), "foreground not opaque: {out:?}");
    }

    #[test]
    fn full_far_coverage_shows_background_blur() {
        // Saturated far CoC with no foreground: result == far-blurred colour.
        let layers = DofLayers {
            sharp: Vec3::new(0.9, 0.1, 0.1),
            far_blurred: Vec3::new(0.1, 0.1, 0.9),
            near_blurred: Vec3::ZERO,
            coc_normalized: 1.0,
        };
        let out = composite(layers, LayerParams::default());
        assert!(approx_vec(out, layers.far_blurred, 1.0e-5), "background not shown: {out:?}");
    }

    #[test]
    fn foreground_diffusion_is_monotonic_in_composite() {
        // As the foreground CoC deepens, the composite moves monotonically from
        // the sharp colour toward the near-blurred colour (component-wise).
        let sharp = Vec3::new(1.0, 1.0, 1.0);
        let near = Vec3::new(0.0, 0.0, 0.0);
        let p = LayerParams::default();
        let mut prev = 2.0_f32; // luminance proxy = x channel, starts above max
        for k in 0..=20 {
            let coc = -(k as f32 / 20.0);
            let layers = DofLayers {
                sharp,
                far_blurred: Vec3::ZERO,
                near_blurred: near,
                coc_normalized: coc,
            };
            let out = composite(layers, p);
            assert!(out.x <= prev + 1.0e-6, "diffusion not monotonic at {coc}: {} > {prev}", out.x);
            prev = out.x;
        }
    }

    #[test]
    fn over_is_porter_duff() {
        // Opaque top completely hides the bottom.
        let top = premultiply(Vec3::new(0.3, 0.4, 0.5), 1.0);
        let bottom = premultiply(Vec3::new(0.9, 0.9, 0.9), 1.0);
        let out = over(top, bottom);
        assert!(approx_vec(Vec3::new(out.x, out.y, out.z), Vec3::new(0.3, 0.4, 0.5), 1.0e-6));
        // Fully transparent top leaves the bottom untouched.
        let clear = premultiply(Vec3::ONE, 0.0);
        let out2 = over(clear, bottom);
        assert!(approx_vec(Vec3::new(out2.x, out2.y, out2.z), Vec3::new(0.9, 0.9, 0.9), 1.0e-6));
    }

    #[test]
    fn premultiply_roundtrips() {
        let c = Vec3::new(0.2, 0.7, 0.4);
        let pm = premultiply(c, 0.5);
        assert!(approx(pm.w, 0.5, 1.0e-7));
        assert!(approx_vec(unpremultiply(pm), c, 1.0e-6));
        // Transparent pixel unpremultiplies to black.
        assert!(approx_vec(unpremultiply(premultiply(c, 0.0)), Vec3::ZERO, 0.0));
    }

    #[test]
    fn composite_is_finite_for_adversarial_input() {
        let layers = DofLayers {
            sharp: Vec3::new(f32::NAN, 0.5, 0.5),
            far_blurred: Vec3::new(f32::INFINITY, 0.0, 0.0),
            near_blurred: Vec3::new(0.0, f32::NAN, 0.0),
            coc_normalized: f32::NAN,
        };
        let out = composite(layers, LayerParams { near_full_coc: 0.0, far_full_coc: -1.0 });
        assert!(out.is_finite(), "composite not finite: {out:?}");
    }
}
