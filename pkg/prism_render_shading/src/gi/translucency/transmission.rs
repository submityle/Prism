//! Thickness-based translucent transmission — the CPU golden reference for
//! back-lit subsurface light transport (skin, wax, foliage leaves, marble).
//!
//! When light strikes the far side of a thin or soft object some of it survives
//! the trip through the material and emerges on the shaded side.  This module
//! models that transport with Beer-Lambert attenuation over a *signed local
//! thickness* estimate combined with a *wrap-around* diffuse term that lets the
//! lit region bleed past the geometric terminator:
//!
//! ```text
//!   T(d)       = exp(-sigma_t · |d|)              (Beer-Lambert)
//!   wrap(x, w) = max(0, (x + w) / (1 + w))        (wrap-around diffuse)
//!   L_trans    = light · subsurface · T(d) · wrap(-N·L, w)
//! ```
//!
//! The extinction `sigma_t` can be authored directly or derived from a
//! `subsurface_color` and a reference thickness: a slab of the reference
//! thickness then transmits exactly that colour (`color = exp(-sigma_t · ref)`).
//! The *signed* thickness convention lets callers pass a depth difference whose
//! sign distinguishes a point in front of the occluder from one behind it;
//! attenuation always uses the magnitude.
//!
//! # Conventions
//! * `thickness` is a signed local path length in world units; attenuation uses
//!   `|thickness|`, clamped non-negative.  A sign is informational only.
//! * `sigma_t` is a non-negative linear-RGB extinction coefficient (per world
//!   unit); each channel is treated independently.
//! * `subsurface_color` / `light_color` are non-negative linear-RGB radiances
//!   in `[0, ∞)`; colours used to derive extinction are clamped to `(0, 1]`.
//! * `wrap` is a non-negative wrap factor; `wrap = 0` is ordinary half-Lambert
//!   cutoff, larger values push light further around the body.  `n_dot_l` is a
//!   clamped `[-1, 1]` cosine; back-lighting corresponds to `N·L < 0`.
//! * Transmittance lies in `[0, 1]`; every result is finite (never `NaN`).
//! * Spectral quantities are linear-RGB [`Vec3`]s matching the GPU twin.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   and no `unsafe`.  Transcendental maths goes through [`bevy_math::ops`].

use bevy_math::{ops, Vec3};

/// Largest optical depth evaluated before the exponential is treated as zero.
const MAX_OPTICAL_DEPTH: f32 = 80.0;

/// Smallest reference thickness used when deriving extinction from a colour.
const MIN_REFERENCE: f32 = 1.0e-6;

/// Lower bound on a transmission colour channel when inverting to extinction.
const MIN_COLOR: f32 = 1.0e-4;

/// Beer-Lambert transmittance `T = exp(-sigma_t · |thickness|)` (scalar).
///
/// Uses the magnitude of a signed `thickness`, so a point measured in front of
/// or behind the occluder attenuates identically.  `sigma_t` and the thickness
/// magnitude are clamped non-negative, giving `T(0) = 1`, a monotonic decrease
/// in both factors, and a result in `[0, 1]`.
#[inline]
pub fn beer_lambert(sigma_t: f32, thickness: f32) -> f32 {
    let tau = optical_depth(sigma_t, thickness);
    let t = ops::exp(-tau.min(MAX_OPTICAL_DEPTH));
    t.clamp(0.0, 1.0)
}

/// Spectral Beer-Lambert transmittance: per-channel `exp(-sigma_t · |d|)`.
///
/// Each RGB channel of `sigma_t` attenuates independently over the shared
/// signed `thickness`; every component of the result lies in `[0, 1]`.
#[inline]
pub fn beer_lambert_rgb(sigma_t: Vec3, thickness: f32) -> Vec3 {
    Vec3::new(
        beer_lambert(sigma_t.x, thickness),
        beer_lambert(sigma_t.y, thickness),
        beer_lambert(sigma_t.z, thickness),
    )
}

/// Optical depth `tau = sigma_t · |thickness|` (dimensionless), non-negative.
///
/// Both the extinction and the (absolute) thickness are clamped non-negative,
/// so `tau ≥ 0` and non-finite inputs collapse to `0`.
#[inline]
pub fn optical_depth(sigma_t: f32, thickness: f32) -> f32 {
    let sigma_t = clamp_non_negative(sigma_t);
    let thickness = clamp_non_negative(abs_finite(thickness));
    let tau = sigma_t * thickness;
    if tau.is_finite() {
        tau.max(0.0)
    } else {
        0.0
    }
}

/// Derives a per-channel extinction `sigma_t` from a transmission colour.
///
/// Inverts `color = exp(-sigma_t · reference)` to `sigma_t = -ln(color) /
/// reference`, so a slab of `reference` thickness transmits exactly `color`.
/// Each colour channel is clamped to `(0, 1]` (a fully opaque `0` would imply
/// infinite extinction) and `reference` is clamped to a small positive value.
/// Every returned channel is non-negative and finite.
#[inline]
pub fn extinction_from_color(color: Vec3, reference: f32) -> Vec3 {
    let reference = clamp_finite(reference, MIN_REFERENCE, f32::MAX);
    Vec3::new(
        channel_extinction(color.x, reference),
        channel_extinction(color.y, reference),
        channel_extinction(color.z, reference),
    )
}

/// Transmission colour after travelling `thickness` through a medium whose
/// `reference`-thickness colour is `color`.
///
/// Equal to `color^{|thickness| / reference}` evaluated per channel, i.e. the
/// Beer-Lambert transmittance of [`extinction_from_color`].  At
/// `|thickness| = reference` the result equals `color`; at `thickness = 0` it is
/// white.  Each channel lies in `[0, 1]`.
#[inline]
pub fn transmittance_from_color(color: Vec3, thickness: f32, reference: f32) -> Vec3 {
    let reference = clamp_finite(reference, MIN_REFERENCE, f32::MAX);
    let t = clamp_non_negative(abs_finite(thickness));
    let exponent = t / reference;
    Vec3::new(
        channel_pow(color.x, exponent),
        channel_pow(color.y, exponent),
        channel_pow(color.z, exponent),
    )
}

/// Wrap-around (half-Lambert-style) diffuse response `max(0, (x + w)/(1 + w))`.
///
/// Softens the diffuse terminator by shifting the cosine `x = N·L` by `wrap`
/// before clamping, so light wraps `wrap`-worth past the 90° boundary.  With
/// `wrap = 0` this is the ordinary clamped cosine; the result is in `[0, 1]`.
#[inline]
pub fn wrap_diffuse(n_dot_l: f32, wrap: f32) -> f32 {
    let x = clamp_finite(n_dot_l, -1.0, 1.0);
    let w = clamp_non_negative(wrap);
    let value = (x + w) / (1.0 + w);
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Back-lit transmitted radiance through a thin translucent slab.
///
/// Combines the Beer-Lambert attenuation of the supplied `sigma_t` over the
/// signed `thickness` with a wrap-around diffuse term driven by the *back*
/// cosine `-(N·L)` (light arriving from behind the surface), then tints the
/// result by `subsurface_color` and the incident `light_color`:
///
/// ```text
///   L = light_color · subsurface_color · exp(-sigma_t·|d|) · wrap(-N·L, wrap).
/// ```
///
/// All colours are clamped non-negative and the result is finite and
/// non-negative per channel.
#[inline]
pub fn back_transmission(
    n_dot_l: f32,
    thickness: f32,
    sigma_t: Vec3,
    light_color: Vec3,
    subsurface_color: Vec3,
    wrap: f32,
) -> Vec3 {
    let attenuation = beer_lambert_rgb(sigma_t, thickness);
    let back = wrap_diffuse(-clamp_finite(n_dot_l, -1.0, 1.0), wrap);
    let out = sanitize_rgb(light_color) * sanitize_rgb(subsurface_color) * attenuation * back;
    sanitize_rgb(out)
}

/// Back-lit transmitted radiance using a colour-derived extinction.
///
/// Convenience over [`back_transmission`] that first converts
/// `subsurface_color` and a `reference` thickness into an extinction via
/// [`extinction_from_color`], so the authored colour is reached at the reference
/// thickness.  The same colour also tints the transmitted light.
#[inline]
pub fn back_transmission_colored(
    n_dot_l: f32,
    thickness: f32,
    subsurface_color: Vec3,
    reference: f32,
    light_color: Vec3,
    wrap: f32,
) -> Vec3 {
    let sigma_t = extinction_from_color(subsurface_color, reference);
    back_transmission(
        n_dot_l,
        thickness,
        sigma_t,
        light_color,
        subsurface_color,
        wrap,
    )
}

/// Per-channel extinction for one transmission-colour component.
#[inline]
fn channel_extinction(color: f32, reference: f32) -> f32 {
    let c = clamp_finite(color, MIN_COLOR, 1.0);
    let sigma = -ops::ln(c) / reference;
    if sigma.is_finite() {
        sigma.max(0.0)
    } else {
        0.0
    }
}

/// `color^exponent` for a clamped transmission-colour channel.
#[inline]
fn channel_pow(color: f32, exponent: f32) -> f32 {
    let c = clamp_finite(color, 0.0, 1.0);
    let e = clamp_non_negative(exponent);
    if c <= 0.0 {
        // `0^0 = 1`, otherwise fully absorbed.
        return if e <= 0.0 { 1.0 } else { 0.0 };
    }
    let value = ops::powf(c, e);
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Absolute value with non-finite inputs mapped to `0`.
#[inline]
fn abs_finite(value: f32) -> f32 {
    if value.is_finite() {
        value.abs()
    } else {
        0.0
    }
}

/// Clamps `value` into `[lo, hi]`, mapping non-finite inputs to `lo`.
#[inline]
fn clamp_finite(value: f32, lo: f32, hi: f32) -> f32 {
    if value.is_finite() {
        value.clamp(lo, hi)
    } else {
        lo
    }
}

/// Clamps `value` to be non-negative and finite.
#[inline]
fn clamp_non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Replaces any non-finite channel with `0` and clamps every channel
/// non-negative.
#[inline]
fn sanitize_rgb(rgb: Vec3) -> Vec3 {
    Vec3::new(
        if rgb.x.is_finite() { rgb.x.max(0.0) } else { 0.0 },
        if rgb.y.is_finite() { rgb.y.max(0.0) } else { 0.0 },
        if rgb.z.is_finite() { rgb.z.max(0.0) } else { 0.0 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beer_lambert_monotonic_and_bounded() {
        assert!((beer_lambert(1.0, 0.0) - 1.0).abs() < 1e-6);
        let mut prev = 1.0;
        for d in 1..=20 {
            let t = beer_lambert(0.5, d as f32);
            assert!(t <= prev + 1e-6, "not monotonic at d={d}");
            assert!((0.0..=1.0).contains(&t));
            prev = t;
        }
        let t = beer_lambert(2.0, 3.0);
        assert!((t - ops::exp(-6.0)).abs() < 1e-6, "t={t}");
    }

    #[test]
    fn beer_lambert_uses_absolute_thickness() {
        for d in [0.5f32, 1.0, 2.0, 4.0] {
            assert!((beer_lambert(0.7, d) - beer_lambert(0.7, -d)).abs() < 1e-7);
        }
    }

    #[test]
    fn beer_lambert_rgb_is_per_channel() {
        let t = beer_lambert_rgb(Vec3::new(0.0, 1.0, 2.0), 1.0);
        assert!((t.x - 1.0).abs() < 1e-6);
        assert!((t.y - ops::exp(-1.0)).abs() < 1e-6);
        assert!((t.z - ops::exp(-2.0)).abs() < 1e-6);
    }

    #[test]
    fn extinction_round_trips_reference_color() {
        let color = Vec3::new(0.8, 0.4, 0.1);
        let reference = 1.5;
        let sigma = extinction_from_color(color, reference);
        // A slab of the reference thickness transmits the authored colour.
        let t = beer_lambert_rgb(sigma, reference);
        assert!((t - color).abs().max_element() < 1e-4, "t={t} color={color}");
    }

    #[test]
    fn transmittance_from_color_matches_reference() {
        let color = Vec3::new(0.9, 0.5, 0.2);
        let reference = 2.0;
        let at_ref = transmittance_from_color(color, reference, reference);
        assert!((at_ref - color).abs().max_element() < 1e-4, "at_ref={at_ref}");
        let at_zero = transmittance_from_color(color, 0.0, reference);
        assert!((at_zero - Vec3::ONE).abs().max_element() < 1e-6, "at_zero={at_zero}");
        // Doubling the thickness squares the transmission colour.
        let at_double = transmittance_from_color(color, 2.0 * reference, reference);
        assert!((at_double - color * color).abs().max_element() < 1e-4);
    }

    #[test]
    fn wrap_diffuse_is_bounded_and_monotonic() {
        let w = 0.5;
        let mut prev = -1.0;
        for i in -10..=10 {
            let x = i as f32 / 10.0;
            let v = wrap_diffuse(x, w);
            assert!((0.0..=1.0).contains(&v), "x={x} v={v}");
            assert!(v >= prev - 1e-6, "not monotonic at x={x}");
            prev = v;
        }
        // Zero wrap is the ordinary clamped cosine.
        assert!((wrap_diffuse(0.5, 0.0) - 0.5).abs() < 1e-6);
        assert_eq!(wrap_diffuse(-0.5, 0.0), 0.0);
        // Positive wrap lifts the fully back-lit value above zero.
        assert!(wrap_diffuse(-1.0, 1.0) >= 0.0);
        assert!(wrap_diffuse(-0.5, 1.0) > 0.0);
    }

    #[test]
    fn back_transmission_grows_as_slab_thins() {
        let sigma = Vec3::splat(1.0);
        let light = Vec3::splat(3.0);
        let ss = Vec3::new(0.9, 0.5, 0.3);
        // Light from behind: N·L negative.
        let thin = back_transmission(-0.3, 0.5, sigma, light, ss, 0.5);
        let thick = back_transmission(-0.3, 4.0, sigma, light, ss, 0.5);
        assert!(thin.x > thick.x, "thin={thin} thick={thick}");
        assert!(thin.is_finite() && thick.is_finite());
    }

    #[test]
    fn back_transmission_vanishes_without_back_light() {
        // Fully front-lit (N·L = 1) gives no back-side wrap contribution.
        let out = back_transmission(1.0, 1.0, Vec3::splat(0.5), Vec3::ONE, Vec3::ONE, 0.0);
        assert!(out.max_element() < 1e-6, "out={out}");
    }

    #[test]
    fn colored_variant_reaches_authored_color_at_reference() {
        let ss = Vec3::new(0.7, 0.3, 0.1);
        let reference = 1.0;
        // Pure back-lighting (N·L = -1) with wrap=1 => back wrap = 1.
        let out = back_transmission_colored(-1.0, reference, ss, reference, Vec3::ONE, 1.0);
        // L = light(1) * ss * exp(-sigma*ref)=ss * ss = ss^2.
        let expected = ss * ss;
        assert!((out - expected).abs().max_element() < 2e-3, "out={out} exp={expected}");
    }

    #[test]
    fn is_deterministic() {
        let a = back_transmission(-0.4, 1.1, Vec3::splat(0.6), Vec3::ONE, Vec3::splat(0.5), 0.3);
        let b = back_transmission(-0.4, 1.1, Vec3::splat(0.6), Vec3::ONE, Vec3::splat(0.5), 0.3);
        assert_eq!(a, b);
        assert_eq!(wrap_diffuse(0.2, 0.4), wrap_diffuse(0.2, 0.4));
    }

    #[test]
    fn no_nan_on_degenerate_inputs() {
        assert!(beer_lambert(f32::NAN, f32::NAN).is_finite());
        assert!(optical_depth(f32::INFINITY, f32::NAN).is_finite());
        assert!(extinction_from_color(Vec3::splat(f32::NAN), 0.0).is_finite());
        assert!(transmittance_from_color(Vec3::splat(f32::NAN), f32::NAN, 0.0).is_finite());
        assert!(wrap_diffuse(f32::NAN, f32::NAN).is_finite());
        let out = back_transmission(
            f32::NAN,
            f32::NAN,
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::NAN),
            f32::NAN,
        );
        assert!(out.is_finite());
    }
}
