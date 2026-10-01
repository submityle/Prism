//! Radial (transverse / lateral) chromatic aberration — CPU golden reference.
//!
//! A real lens cannot focus every wavelength to the same point: its refractive
//! index varies with colour, so the red, green and blue images are magnified by
//! slightly different amounts.  Away from the optical axis this *lateral*
//! chromatic aberration shows up as coloured fringes that grow with the
//! distance from the image centre, which is exactly the look real-time
//! post-process pipelines (Unreal Engine's "Scene Fringe", most film-emulation
//! stacks) reproduce by resampling each channel at a radius-dependent scale:
//!
//! ```text
//! r2    = |uv - center|^2
//! uv_c  = center + (uv - center) * (1 + k_c * r2)      // c in {R, G, B}
//! ```
//!
//! Each channel `c` has its own coefficient `k_c`; `k > 0` magnifies (pushes the
//! sample outward), `k < 0` minifies, and `k = 0` is the identity.  Because the
//! scale depends on `r2`, the centre of the frame is untouched and the fringe
//! widens quadratically toward the corners — the physical behaviour of lateral
//! CA.  The green channel is usually left near zero (the reference focus) while
//! red and blue are pushed in opposite directions.
//!
//! This module provides the per-channel sample-coordinate computation plus two
//! recombination paths: a closure-driven resampler (for callers that can sample
//! an image at arbitrary UVs) and a prefetched-triple combiner (for callers
//! that already fetched one colour per channel coordinate).
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions would go through [`bevy_math::ops`]; this module
//!   needs none (only multiplies and a dot product), so the maths is exact.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit `Vec2` centre).
//! * Channel order is always `[R, G, B]`.
//! * Defensive clamping everywhere: coefficients and coordinates are forced
//!   finite and bounded so a degenerate UV or coefficient never emits
//!   `NaN`/`inf`.

use bevy_math::Vec2;

/// Largest magnitude allowed for a per-channel aberration coefficient.
///
/// The scale factor is `1 + k * r2`; with `r2` bounded by a few units across a
/// normalised frame this keeps the sample displacement finite and prevents a
/// pathological coefficient from folding the sample coordinate through the
/// centre.
pub const MAX_COEFFICIENT: f32 = 16.0;

/// Index of the red channel in the `[R, G, B]` convention.
pub const R: usize = 0;
/// Index of the green channel in the `[R, G, B]` convention.
pub const G: usize = 1;
/// Index of the blue channel in the `[R, G, B]` convention.
pub const B: usize = 2;

/// Clamp a coefficient to a finite value in `[-MAX_COEFFICIENT, MAX_COEFFICIENT]`.
///
/// Non-finite input (`NaN`/`inf`) collapses to `0` (the identity coefficient).
#[inline]
fn sanitize_coeff(k: f32) -> f32 {
    if k.is_finite() {
        k.clamp(-MAX_COEFFICIENT, MAX_COEFFICIENT)
    } else {
        0.0
    }
}

/// Replace a non-finite coordinate component with a fallback.
#[inline]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Sanitize a `Vec2`, falling back to `fallback` per component when non-finite.
#[inline]
fn sanitize_vec2(v: Vec2, fallback: Vec2) -> Vec2 {
    Vec2::new(finite_or(v.x, fallback.x), finite_or(v.y, fallback.y))
}

/// Per-channel lateral chromatic-aberration parameters.
///
/// The transform is radial about [`ChromaticAberration::center`]; each of the
/// three channels is magnified by `1 + k_c * r2`, where `r2` is the squared
/// distance from the centre in UV space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChromaticAberration {
    /// Optical-axis location in UV space (the frame centre, usually `(0.5, 0.5)`).
    pub center: Vec2,
    /// Per-channel radial coefficients `[k_r, k_g, k_b]`.
    pub k: [f32; 3],
}

impl Default for ChromaticAberration {
    /// Identity: centred at `(0.5, 0.5)` with all coefficients `0`.
    fn default() -> Self {
        Self {
            center: Vec2::splat(0.5),
            k: [0.0; 3],
        }
    }
}

impl ChromaticAberration {
    /// Construct from an explicit centre and per-channel coefficients.
    ///
    /// The centre is forced finite (falling back to `(0.5, 0.5)` per component)
    /// and every coefficient is clamped to `[-MAX_COEFFICIENT, MAX_COEFFICIENT]`,
    /// so the resulting transform is always well behaved.
    #[must_use]
    pub fn new(center: Vec2, k: [f32; 3]) -> Self {
        Self {
            center: sanitize_vec2(center, Vec2::splat(0.5)),
            k: [sanitize_coeff(k[R]), sanitize_coeff(k[G]), sanitize_coeff(k[B])],
        }
    }

    /// Construct a symmetric red/blue split about a green reference.
    ///
    /// Red is pushed outward by `+strength`, blue inward by `-strength`, and
    /// green is left at the focus (`0`).  This is the common artist control: a
    /// single slider that fans the extreme channels apart around the centre.
    #[must_use]
    pub fn symmetric(center: Vec2, strength: f32) -> Self {
        let s = sanitize_coeff(strength);
        Self::new(center, [s, 0.0, -s])
    }

    /// The scale factor `1 + k_c * r2` applied to channel `c` at squared radius
    /// `r2`.  Clamped to be non-negative so the sample coordinate can never be
    /// reflected through the centre by an aggressive negative coefficient.
    #[inline]
    #[must_use]
    pub fn channel_scale(&self, channel: usize, r2: f32) -> f32 {
        let k = self.k[channel.min(B)];
        (1.0 + k * r2).max(0.0)
    }

    /// Per-channel sample coordinates for the pixel being shaded at `uv`.
    ///
    /// Returns `[uv_r, uv_g, uv_b]`, each computed as
    /// `center + (uv - center) * channel_scale(c, r2)`.  At the centre all three
    /// coincide with `center`; with all coefficients `0` all three equal `uv`.
    #[must_use]
    pub fn channel_uv(&self, uv: Vec2) -> [Vec2; 3] {
        let uv = sanitize_vec2(uv, self.center);
        let delta = uv - self.center;
        let r2 = delta.length_squared();
        [
            self.center + delta * self.channel_scale(R, r2),
            self.center + delta * self.channel_scale(G, r2),
            self.center + delta * self.channel_scale(B, r2),
        ]
    }

    /// Resample an image through the aberration using a caller-supplied sampler.
    ///
    /// `sampler(uv)` must return the full `[R, G, B]` colour at `uv` (typically a
    /// bilinear texture fetch with clamp-to-edge addressing).  The output takes
    /// channel `c` from the sample at that channel's coordinate, so the three
    /// channels are fetched at three slightly different radii.
    ///
    /// The result is floored to zero per channel; a sampler that returns a
    /// non-finite component contributes `0` for that component so the composite
    /// stays finite.
    #[must_use]
    pub fn sample<F>(&self, uv: Vec2, sampler: F) -> [f32; 3]
    where
        F: Fn(Vec2) -> [f32; 3],
    {
        let coords = self.channel_uv(uv);
        let sr = sampler(coords[R]);
        let sg = sampler(coords[G]);
        let sb = sampler(coords[B]);
        combine_channels(sr, sg, sb)
    }
}

/// Free-function form of [`ChromaticAberration::channel_uv`].
///
/// Computes the three per-channel sample coordinates for `uv` about `center`
/// with coefficients `k = [k_r, k_g, k_b]`.  Inputs are sanitized exactly as in
/// [`ChromaticAberration::new`].
#[must_use]
pub fn chromatic_offsets(uv: Vec2, center: Vec2, k: [f32; 3]) -> [Vec2; 3] {
    ChromaticAberration::new(center, k).channel_uv(uv)
}

/// Combine three prefetched colours — one fetched per channel coordinate — into
/// the final aberrated pixel.
///
/// `r_sample`, `g_sample` and `b_sample` are the colours fetched at the red,
/// green and blue coordinates from [`ChromaticAberration::channel_uv`]; the
/// result keeps only the matching channel from each.  Each output component is
/// floored to zero and forced finite.
#[must_use]
pub fn resample_rgb(r_sample: [f32; 3], g_sample: [f32; 3], b_sample: [f32; 3]) -> [f32; 3] {
    combine_channels(r_sample, g_sample, b_sample)
}

/// Shared recombination: pick channel `c` from the `c`-th prefetched colour and
/// clamp to a finite, non-negative value.
#[inline]
fn combine_channels(r_sample: [f32; 3], g_sample: [f32; 3], b_sample: [f32; 3]) -> [f32; 3] {
    [
        finite_or(r_sample[R], 0.0).max(0.0),
        finite_or(g_sample[G], 0.0).max(0.0),
        finite_or(b_sample[B], 0.0).max(0.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn approx_vec(a: Vec2, b: Vec2, eps: f32) -> bool {
        approx(a.x, b.x, eps) && approx(a.y, b.y, eps)
    }

    #[test]
    fn center_has_no_offset() {
        let ca = ChromaticAberration::symmetric(Vec2::splat(0.5), 0.8);
        let uvs = ca.channel_uv(Vec2::splat(0.5));
        for uv in uvs {
            assert!(approx_vec(uv, Vec2::splat(0.5), EPS), "centre moved: {uv:?}");
        }
    }

    #[test]
    fn zero_coefficients_are_identity() {
        let ca = ChromaticAberration::new(Vec2::splat(0.5), [0.0; 3]);
        let p = Vec2::new(0.9, 0.2);
        for uv in ca.channel_uv(p) {
            assert!(approx_vec(uv, p, EPS), "k=0 should be identity: {uv:?}");
        }
    }

    #[test]
    fn edge_channel_separation_is_monotonic() {
        // Symmetric split: red pushes outward, blue inward. The gap between the
        // red and blue sample coordinates must grow as we leave the centre.
        let ca = ChromaticAberration::symmetric(Vec2::splat(0.5), 0.5);
        let dir = Vec2::new(1.0, 0.0);
        let mut prev = -1.0_f32;
        for i in 0..=16 {
            let t = i as f32 / 16.0; // radius 0 .. 0.5
            let uv = Vec2::splat(0.5) + dir * (0.5 * t);
            let c = ca.channel_uv(uv);
            let sep = (c[R] - c[B]).length();
            assert!(sep + EPS >= prev, "separation not monotonic at t={t}");
            prev = sep;
        }
        assert!(prev > 0.0, "edge separation should be positive");
    }

    #[test]
    fn transform_is_point_symmetric_about_center() {
        let center = Vec2::splat(0.5);
        let ca = ChromaticAberration::symmetric(center, 0.7);
        let delta = Vec2::new(0.3, -0.15);
        let plus = ca.channel_uv(center + delta);
        let minus = ca.channel_uv(center - delta);
        for c in 0..3 {
            // The two sample coordinates must be mirror images about the centre.
            let mirrored = 2.0 * center - plus[c];
            assert!(approx_vec(minus[c], mirrored, EPS), "channel {c} not symmetric");
        }
    }

    #[test]
    fn sampler_path_matches_prefetched_path() {
        let ca = ChromaticAberration::symmetric(Vec2::splat(0.5), 0.6);
        let uv = Vec2::new(0.85, 0.35);
        // A deterministic "image": colour encodes the coordinate.
        let sampler = |p: Vec2| [p.x, p.y, p.x + p.y];
        let coords = ca.channel_uv(uv);
        let via_sampler = ca.sample(uv, sampler);
        let via_prefetched = resample_rgb(
            sampler(coords[R]),
            sampler(coords[G]),
            sampler(coords[B]),
        );
        for c in 0..3 {
            assert!(approx(via_sampler[c], via_prefetched[c], EPS), "channel {c} mismatch");
        }
    }

    #[test]
    fn free_function_matches_method() {
        let center = Vec2::new(0.4, 0.55);
        let k = [0.3, -0.1, -0.4];
        let uv = Vec2::new(0.1, 0.9);
        let via_fn = chromatic_offsets(uv, center, k);
        let via_method = ChromaticAberration::new(center, k).channel_uv(uv);
        for c in 0..3 {
            assert!(approx_vec(via_fn[c], via_method[c], EPS), "channel {c} mismatch");
        }
    }

    #[test]
    fn non_finite_inputs_stay_finite() {
        let ca = ChromaticAberration::new(Vec2::new(f32::NAN, 0.5), [f32::INFINITY, 0.0, -1.0]);
        let uvs = ca.channel_uv(Vec2::new(f32::INFINITY, 0.2));
        for uv in uvs {
            assert!(uv.x.is_finite() && uv.y.is_finite(), "coord not finite: {uv:?}");
        }
        let out = ca.sample(Vec2::new(0.9, 0.1), |_| [f32::NAN, 1.0, f32::INFINITY]);
        for c in out {
            assert!(c.is_finite(), "colour not finite: {c}");
        }
    }

    #[test]
    fn negative_coefficient_cannot_reflect_through_center() {
        // A huge negative coefficient would drive the scale negative; the clamp
        // floors it at zero so the sample collapses to the centre instead of
        // flipping to the opposite side.
        let ca = ChromaticAberration::new(Vec2::splat(0.5), [-MAX_COEFFICIENT, 0.0, 0.0]);
        let uv = Vec2::new(1.0, 0.5); // r2 = 0.25
        let coords = ca.channel_uv(uv);
        // scale = max(0, 1 - 16 * 0.25) = 0 => red collapses to centre.
        assert!(approx_vec(coords[R], Vec2::splat(0.5), EPS), "red should collapse to centre");
    }
}
