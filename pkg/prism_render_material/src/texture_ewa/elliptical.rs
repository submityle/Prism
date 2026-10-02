//! **Elliptical Weighted Average (EWA)** anisotropic texture filtering
//! (Heckbert 1986/1989; Greene & Heckbert 1986).
//!
//! Hardware anisotropic filtering -- and this crate's own
//! [`filter_resolved`](crate::filter_resolved) -- approximates the pixel's
//! projected texture footprint by averaging a short line of uniformly weighted
//! probe taps along the major axis. EWA instead treats the footprint as the
//! **ellipse** it really is and gathers every texel inside it under a Gaussian
//! reconstruction weight, so it is the reference-quality anisotropic filter:
//! no tap-count stair-stepping, no over-blur across the minor axis, exactly the
//! "textbook" result a high-end offline sampler produces.
//!
//! The pixel footprint in texel space is spanned by the two screen-space
//! partial derivatives of the texture coordinate,
//! `grad_x = d(u, v)/dx` and `grad_y = d(u, v)/dy` (both in texels per pixel).
//! Following Heckbert, those two vectors define an implicit conic
//!
//! ```text
//! r2(s, t) = A s^2 + B s t + C t^2
//! ```
//!
//! (with `s, t` the texel offset from the sample centre) whose `r2 < 1` region
//! is the filter ellipse. A unit **reconstruction** term is added to `A` and
//! `C` so the footprint never collapses below ~1 texel, giving graceful
//! magnification. Each covered texel is weighted by the clamped Gaussian
//! `exp(-alpha * r2) - exp(-alpha)` (so the weight reaches `0` continuously at
//! the `r2 = 1` boundary) and the result is renormalised -- a **convex
//! combination** of the covered texels, so a constant region is preserved and
//! the output never leaves the sampled range. Eccentricity is capped
//! (`MAX_ANISOTROPY`) by lengthening the minor axis, bounding the work exactly
//! as a production EWA sampler does.
//!
//! Properties the oracles below pin down:
//! * an **isotropic** footprint (`grad_x = (h, 0)`, `grad_y = (0, h)`) collapses
//!   the conic to the circle `r2 = (s^2 + t^2) / (h^2 + 1)`, so EWA equals an
//!   independently evaluated **radial** Gaussian gather -- proving the conic
//!   coefficients (`B = 0`, `A = C`) are right;
//! * an **axis-aligned** footprint (`grad_x = (a, 0)`, `grad_y = (0, b)`) gives
//!   `r2 = s^2 / (a^2 + 1) + t^2 / (b^2 + 1)`, matching an independent
//!   per-axis evaluation;
//! * a constant plane is preserved and the output stays within the sampled
//!   range for any footprint;
//! * a zero-gradient footprint sampled at an integer texel returns that exact
//!   texel (the reconstruction term keeps only the centre tap).
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML -- so a
//! CPU golden matches a GPU compute EWA to floating-point tolerance.
//!
//! # References
//! * Heckbert, "Survey of Texture Mapping" (1986) and *Fundamentals of Texture
//!   Mapping and Image Warping* (1989), ch. 3 (EWA).
//! * Greene & Heckbert, "Creating Raster Omnimax Images ... the Elliptical
//!   Weighted Average Filter" (1986).

use bevy_math::ops;

use crate::{linear_to_srgb, srgb_to_linear, ColorSpace, Rgba8Image, WrapMode};

/// Gaussian falloff of the reconstruction filter; `alpha = 2` is Heckbert's
/// recommended value (the weight has decayed to `exp(-2) ~ 0.135` at the
/// ellipse edge before the boundary subtraction).
const ALPHA: f32 = 2.0;

/// Largest major/minor axis ratio before the minor axis is lengthened (blurred)
/// to cap the number of covered texels, matching a production EWA sampler.
const MAX_ANISOTROPY: f32 = 16.0;

/// Wrap an integer tap index into `[0, n)` (local copy of the shared helper so
/// this module stays self-contained).
#[inline]
#[must_use]
fn wrap_index(i: i64, n: i64, mode: WrapMode) -> usize {
    match mode {
        WrapMode::Repeat => i.rem_euclid(n) as usize,
        WrapMode::MirroredRepeat => {
            let p = i.rem_euclid(2 * n);
            let m = if p >= n { 2 * n - 1 - p } else { p };
            m as usize
        }
        _ => i.clamp(0, n - 1) as usize,
    }
}

/// The resolved filter ellipse: conic coefficients `(a, b, c)` for
/// `r2 = a s^2 + b s t + c t^2` plus the inclusive integer texel bounding box
/// `[s0, s1] x [t0, t1]` that encloses `r2 < 1`.
struct Ellipse {
    a: f32,
    b: f32,
    c: f32,
    s0: i64,
    s1: i64,
    t0: i64,
    t1: i64,
}

/// Build the filter ellipse from the sample centre and the two screen-space UV
/// gradients. Returns `None` for a non-finite / degenerate setup.
fn resolve_ellipse(u: f32, v: f32, grad_x: [f32; 2], grad_y: [f32; 2]) -> Option<Ellipse> {
    if !u.is_finite()
        || !v.is_finite()
        || !grad_x[0].is_finite()
        || !grad_x[1].is_finite()
        || !grad_y[0].is_finite()
        || !grad_y[1].is_finite()
    {
        return None;
    }

    // Cap eccentricity by lengthening the minor axis. The conic is symmetric in
    // the two gradient vectors, so which one is "major" only matters here.
    let (mut dx, mut dy) = (grad_x, grad_y);
    let len2 = |p: [f32; 2]| p[0] * p[0] + p[1] * p[1];
    if len2(dx) < len2(dy) {
        core::mem::swap(&mut dx, &mut dy);
    }
    let major = ops::sqrt(len2(dx));
    let minor = ops::sqrt(len2(dy));
    if minor > 0.0 && minor * MAX_ANISOTROPY < major {
        let scale = major / (minor * MAX_ANISOTROPY);
        dy = [dy[0] * scale, dy[1] * scale];
    }

    // Heckbert conic with the unit reconstruction term on A and C.
    let mut a = dx[1] * dx[1] + dy[1] * dy[1] + 1.0;
    let mut b = -2.0 * (dx[0] * dx[1] + dy[0] * dy[1]);
    let mut c = dx[0] * dx[0] + dy[0] * dy[0] + 1.0;

    let f = a * c - b * b * 0.25;
    if !f.is_finite() || f <= 0.0 {
        return None;
    }
    let inv_f = 1.0 / f;
    a *= inv_f;
    b *= inv_f;
    c *= inv_f;

    // Axis-aligned bounding box of the r2 < 1 ellipse.
    let det = -b * b + 4.0 * a * c;
    if !det.is_finite() || det <= 0.0 {
        return None;
    }
    let inv_det = 1.0 / det;
    let s_half = 2.0 * inv_det * ops::sqrt(det * c);
    let t_half = 2.0 * inv_det * ops::sqrt(a * det);
    if !s_half.is_finite() || !t_half.is_finite() {
        return None;
    }
    let s0 = ops::floor(u - s_half) as i64;
    let s1 = ops::floor(u + s_half) as i64;
    let t0 = ops::floor(v - t_half) as i64;
    let t1 = ops::floor(v + t_half) as i64;

    Some(Ellipse {
        a,
        b,
        c,
        s0,
        s1,
        t0,
        t1,
    })
}

/// Nearest-texel fallback for a degenerate footprint.
#[inline]
fn nearest(plane: &[f32], w: usize, wi: i64, hi: i64, u: f32, v: f32, wrap: WrapMode) -> f32 {
    let sx = wrap_index(ops::round(u) as i64, wi, wrap);
    let sy = wrap_index(ops::round(v) as i64, hi, wrap);
    plane[sy * w + sx]
}

/// Elliptical-weighted-average filter a single-channel `width * height`
/// row-major plane at continuous texel coordinate `(u, v)`, with the pixel
/// footprint given by the screen-space UV gradients `grad_x = d(u, v)/dx` and
/// `grad_y = d(u, v)/dy` (texels per pixel), resolving borders through `wrap`.
///
/// Returns `0.0` for an empty / mismatched plane; falls back to the nearest
/// texel for a degenerate (non-finite) footprint.
#[must_use]
pub fn ewa_sample_plane(
    plane: &[f32],
    width: u32,
    height: u32,
    u: f32,
    v: f32,
    grad_x: [f32; 2],
    grad_y: [f32; 2],
    wrap: WrapMode,
) -> f32 {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || plane.len() != w * h {
        return 0.0;
    }
    let wi = width as i64;
    let hi = height as i64;

    let Some(e) = resolve_ellipse(u, v, grad_x, grad_y) else {
        return nearest(plane, w, wi, hi, u, v, wrap);
    };
    let edge = ops::exp(-ALPHA);

    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    for it in e.t0..=e.t1 {
        let tt = it as f32 - v;
        let sy = wrap_index(it, hi, wrap);
        let srow = sy * w;
        for is in e.s0..=e.s1 {
            let ss = is as f32 - u;
            let r2 = e.a * ss * ss + e.b * ss * tt + e.c * tt * tt;
            if r2 < 1.0 {
                let weight = ops::exp(-ALPHA * r2) - edge;
                let sx = wrap_index(is, wi, wrap);
                sum += weight * plane[srow + sx];
                wsum += weight;
            }
        }
    }
    if wsum > 0.0 {
        sum / wsum
    } else {
        nearest(plane, w, wi, hi, u, v, wrap)
    }
}

/// Elliptical-weighted-average filter an `RGBA8` image at continuous texel
/// coordinate `(u, v)`, with the pixel footprint given by the screen-space UV
/// gradients, gamma-correctly under `space`, resolving borders through `wrap`.
///
/// Colour is filtered in scene-linear light; alpha linearly (`/255`). The
/// weights are evaluated once and shared across the four channels. Returns an
/// opaque black texel for an empty image; falls back to the nearest texel for a
/// degenerate footprint.
#[must_use]
pub fn ewa_sample_rgba8(
    src: &Rgba8Image,
    u: f32,
    v: f32,
    grad_x: [f32; 2],
    grad_y: [f32; 2],
    wrap: WrapMode,
    space: ColorSpace,
) -> [u8; 4] {
    let (width, height) = (src.width(), src.height());
    let w = width as usize;
    if width == 0 || height == 0 {
        return [0, 0, 0, 255];
    }
    let wi = width as i64;
    let hi = height as i64;
    let texels = src.as_slice();

    let lift = |t: [u8; 4]| -> [f32; 4] {
        let (r, g, b) = match space {
            ColorSpace::Linear => (
                f32::from(t[0]) / 255.0,
                f32::from(t[1]) / 255.0,
                f32::from(t[2]) / 255.0,
            ),
            ColorSpace::Srgb => (
                srgb_to_linear(t[0]),
                srgb_to_linear(t[1]),
                srgb_to_linear(t[2]),
            ),
        };
        [r, g, b, f32::from(t[3]) / 255.0]
    };

    let round_u8 = |x: f32| -> u8 {
        let cc = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
        (cc * 255.0 + 0.5).floor() as u8
    };
    let encode = |lin: [f32; 4]| -> [u8; 4] {
        let au = round_u8(lin[3]);
        match space {
            ColorSpace::Linear => [round_u8(lin[0]), round_u8(lin[1]), round_u8(lin[2]), au],
            ColorSpace::Srgb => [
                linear_to_srgb(lin[0]),
                linear_to_srgb(lin[1]),
                linear_to_srgb(lin[2]),
                au,
            ],
        }
    };

    let nearest_color = || -> [u8; 4] {
        let sx = wrap_index(ops::round(u) as i64, wi, wrap);
        let sy = wrap_index(ops::round(v) as i64, hi, wrap);
        encode(lift(texels[sy * w + sx]))
    };

    let Some(e) = resolve_ellipse(u, v, grad_x, grad_y) else {
        return nearest_color();
    };
    let edge = ops::exp(-ALPHA);

    let mut acc = [0.0f32; 4];
    let mut wsum = 0.0f32;
    for it in e.t0..=e.t1 {
        let tt = it as f32 - v;
        let sy = wrap_index(it, hi, wrap);
        let srow = sy * w;
        for is in e.s0..=e.s1 {
            let ss = is as f32 - u;
            let r2 = e.a * ss * ss + e.b * ss * tt + e.c * tt * tt;
            if r2 < 1.0 {
                let weight = ops::exp(-ALPHA * r2) - edge;
                let sx = wrap_index(is, wi, wrap);
                let lin = lift(texels[srow + sx]);
                acc[0] += weight * lin[0];
                acc[1] += weight * lin[1];
                acc[2] += weight * lin[2];
                acc[3] += weight * lin[3];
                wsum += weight;
            }
        }
    }
    if wsum > 0.0 {
        let inv = 1.0 / wsum;
        encode([acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv])
    } else {
        nearest_color()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const WRAPS: [WrapMode; 3] = [
        WrapMode::ClampToEdge,
        WrapMode::Repeat,
        WrapMode::MirroredRepeat,
    ];

    fn checker_plane(w: u32, h: u32) -> Vec<f32> {
        (0..(w * h))
            .map(|i| {
                let x = i % w;
                let y = i / w;
                ((x * 13 + y * 7) % 11) as f32 * 0.08 + 0.05
            })
            .collect()
    }

    /// Reference radial / axis-aligned gather sharing EWA's exact weight law but
    /// an independently written conic, swept over the whole (small) plane.
    fn reference_gather(
        plane: &[f32],
        w: u32,
        h: u32,
        u: f32,
        v: f32,
        inv_a2: f32,
        inv_c2: f32,
        wrap: WrapMode,
    ) -> f32 {
        let edge = ops::exp(-ALPHA);
        let (wi, hi) = (w as i64, h as i64);
        let mut sum = 0.0f32;
        let mut wsum = 0.0f32;
        for it in -(h as i64)..(2 * h as i64) {
            let tt = it as f32 - v;
            for is in -(w as i64)..(2 * w as i64) {
                let ss = is as f32 - u;
                let r2 = ss * ss * inv_a2 + tt * tt * inv_c2;
                if r2 < 1.0 {
                    let weight = ops::exp(-ALPHA * r2) - edge;
                    let sx = wrap_index(is, wi, wrap);
                    let sy = wrap_index(it, hi, wrap);
                    sum += weight * plane[sy * w as usize + sx];
                    wsum += weight;
                }
            }
        }
        sum / wsum
    }

    #[test]
    fn isotropic_footprint_matches_radial_gaussian() {
        let (w, h) = (13u32, 11u32);
        let plane = checker_plane(w, h);
        for &h_grad in &[0.8f32, 1.5, 3.0] {
            let inv = 1.0 / (h_grad * h_grad + 1.0);
            // Interior sample so Clamp wrap does not reach the border.
            let (u, v) = (6.3f32, 5.4f32);
            let got = ewa_sample_plane(
                &plane,
                w,
                h,
                u,
                v,
                [h_grad, 0.0],
                [0.0, h_grad],
                WrapMode::ClampToEdge,
            );
            let want = reference_gather(&plane, w, h, u, v, inv, inv, WrapMode::ClampToEdge);
            assert!((got - want).abs() < 1.0e-5, "h={h_grad} {got} vs {want}");
        }
    }

    #[test]
    fn axis_aligned_footprint_matches_separable_radial() {
        let (w, h) = (15u32, 13u32);
        let plane = checker_plane(w, h);
        let (u, v) = (7.2f32, 6.1f32);
        for &(a, b) in &[(3.0f32, 1.0f32), (2.0, 2.5), (4.0, 0.7)] {
            let got = ewa_sample_plane(
                &plane,
                w,
                h,
                u,
                v,
                [a, 0.0],
                [0.0, b],
                WrapMode::ClampToEdge,
            );
            let want = reference_gather(
                &plane,
                w,
                h,
                u,
                v,
                1.0 / (a * a + 1.0),
                1.0 / (b * b + 1.0),
                WrapMode::ClampToEdge,
            );
            assert!((got - want).abs() < 1.0e-5, "a={a} b={b} {got} vs {want}");
        }
    }

    #[test]
    fn constant_plane_is_preserved() {
        let (w, h) = (9u32, 7u32);
        let plane = vec![0.41f32; (w * h) as usize];
        for &wrap in &WRAPS {
            for &g in &[[1.0f32, 0.0], [2.0, 1.0], [0.0, 3.0]] {
                let out = ewa_sample_plane(&plane, w, h, 4.3, 3.6, g, [-g[1], g[0]], wrap);
                assert!((out - 0.41).abs() < 1.0e-6, "{out}");
            }
        }
    }

    #[test]
    fn output_stays_within_sampled_range() {
        let (w, h) = (12u32, 12u32);
        let plane = checker_plane(w, h);
        let lo = plane.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = plane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for &wrap in &WRAPS {
            for &g in &[[2.0f32, 0.5], [1.0, 1.5], [3.0, 0.0]] {
                let out = ewa_sample_plane(&plane, w, h, 5.5, 6.5, g, [-g[1], g[0]], wrap);
                assert!(
                    out >= lo - 1.0e-6 && out <= hi + 1.0e-6,
                    "{out} [{lo},{hi}]"
                );
            }
        }
    }

    #[test]
    fn zero_gradient_integer_coord_returns_the_texel() {
        let (w, h) = (8u32, 6u32);
        let plane = checker_plane(w, h);
        for &wrap in &WRAPS {
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let out = ewa_sample_plane(
                        &plane,
                        w,
                        h,
                        x as f32,
                        y as f32,
                        [0.0, 0.0],
                        [0.0, 0.0],
                        wrap,
                    );
                    let want = plane[(y * w + x) as usize];
                    assert!((out - want).abs() < 1.0e-6, "x={x} y={y} {out} vs {want}");
                }
            }
        }
    }

    #[test]
    fn empty_or_mismatched_plane_is_zero() {
        assert_eq!(
            ewa_sample_plane(
                &[],
                0,
                0,
                0.0,
                0.0,
                [1.0, 0.0],
                [0.0, 1.0],
                WrapMode::Repeat
            ),
            0.0
        );
        let plane = vec![0.5f32; 10];
        assert_eq!(
            ewa_sample_plane(
                &plane,
                4,
                4,
                1.0,
                1.0,
                [1.0, 0.0],
                [0.0, 1.0],
                WrapMode::Repeat
            ),
            0.0
        );
    }

    #[test]
    fn image_constant_is_preserved() {
        let texels = vec![[70u8, 130, 190, 255]; 8 * 6];
        let img = Rgba8Image::new(8, 6, texels).unwrap();
        for &space in &[ColorSpace::Linear, ColorSpace::Srgb] {
            let out = ewa_sample_rgba8(
                &img,
                4.4,
                3.3,
                [2.0, 0.5],
                [-0.5, 2.0],
                WrapMode::Repeat,
                space,
            );
            assert_eq!(out, [70, 130, 190, 255], "constant image preserved");
        }
    }

    #[test]
    fn image_zero_gradient_integer_coord_returns_the_texel() {
        let texels: Vec<[u8; 4]> = (0..8 * 6)
            .map(|i| [i as u8 * 2, 255 - i as u8, i as u8 * 3, 255])
            .collect();
        let img = Rgba8Image::new(8, 6, texels).unwrap();
        for &space in &[ColorSpace::Linear, ColorSpace::Srgb] {
            let out = ewa_sample_rgba8(
                &img,
                3.0,
                2.0,
                [0.0, 0.0],
                [0.0, 0.0],
                WrapMode::ClampToEdge,
                space,
            );
            assert_eq!(
                out,
                img.as_slice()[2 * 8 + 3],
                "nearest texel at integer coord"
            );
        }
    }
}
