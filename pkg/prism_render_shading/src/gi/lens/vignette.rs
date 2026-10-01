//! Vignetting and lens distortion — CPU golden reference.
//!
//! Two independent optical effects live here, both parametrised by the
//! normalised distance from the frame centre.
//!
//! * **Natural vignetting (`cos^4` law).**  Off-axis image points receive less
//!   irradiance than the centre: the classic photometric result is that
//!   illumination falls as `cos^4(θ)`, where `θ` is the field angle.  Writing
//!   the field angle in terms of a focal-length-like parameter `f`,
//!   `cos(θ) = f / sqrt(f² + r²)`, so
//!
//!   ```text
//!   cos^4(θ) = 1 / (1 + (r / f)^2)^2
//!   ```
//!
//!   which is the exact closed form used by [`natural_vignette`] (no trig
//!   needed, and it matches `cos(atan(r / f))^4` to floating point).
//! * **Artistic vignetting.**  A separate, art-directable darkening that keeps
//!   the frame flat inside an inner radius and ramps down with a Hermite
//!   `smoothstep` out to an outer radius — the knob film-emulation stacks and
//!   Unreal Engine's vignette expose.
//!
//! * **Brown–Conrady radial distortion.**  A real lens maps the ideal radius
//!   `r` to a distorted radius `r_d = r * (1 + k1 r² + k2 r⁴)` (barrel for
//!   `k < 0`, pincushion for `k > 0`).  [`distort`] applies it; [`undistort`]
//!   inverts it with a fixed-point iteration so a distorted sample can be
//!   mapped back to the ideal grid.
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * The `cos^4` reference in tests uses [`bevy_math::ops`] (`atan`/`cos`); the
//!   shipping path uses the exact closed form and needs no transcendental.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit `Vec2`).
//! * Defensive clamping everywhere: focal length, radii and distortion inputs
//!   are sanitized so no path emits `NaN`/`inf`.

use bevy_math::Vec2;

/// Smallest focal length / denominator used, to avoid divide-by-zero.
pub const EPSILON: f32 = 1.0e-6;

/// Number of fixed-point iterations used to invert the radial distortion.
pub const UNDISTORT_ITERATIONS: u32 = 12;

/// Replace a non-finite scalar with a fallback.
#[inline]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Sanitize a `Vec2` component-wise against a fallback.
#[inline]
fn sanitize_vec2(v: Vec2, fallback: Vec2) -> Vec2 {
    Vec2::new(finite_or(v.x, fallback.x), finite_or(v.y, fallback.y))
}

/// Hermite `smoothstep`, returning `0` at/below `edge0`, `1` at/above `edge1`.
///
/// A non-ascending `[edge0, edge1]` degenerates to a hard step at `edge0`, so a
/// mis-ordered radius range never divides by zero.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if !(span > 0.0) {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Natural `cos^4` vignetting factor at normalised radius `r` for focal-length
/// parameter `focal`.
///
/// Returns `1 / (1 + (r / f)^2)^2`, i.e. `cos^4` of the field angle: `1` on the
/// optical axis and strictly decreasing with `r`.  `focal` is floored to
/// `EPSILON` and `r` is treated as `|r|`, so the result is always finite and in
/// `(0, 1]`.
#[must_use]
pub fn natural_vignette(r: f32, focal: f32) -> f32 {
    let f = finite_or(focal, 1.0).max(EPSILON);
    let r = finite_or(r, 0.0).abs();
    let ratio = r / f;
    let denom = 1.0 + ratio * ratio;
    let inv = 1.0 / denom;
    (inv * inv).clamp(0.0, 1.0)
}

/// Art-directable vignette factor at normalised radius `r`.
///
/// Returns `1` for `r <= inner`, then ramps down via `smoothstep` to
/// `1 - amount` at `r >= outer`.  `amount` is clamped to `[0, 1]`, so the result
/// stays in `[1 - amount, 1] ⊆ [0, 1]` and is non-increasing in `r`.
#[must_use]
pub fn artistic_vignette(r: f32, inner: f32, outer: f32, amount: f32) -> f32 {
    let r = finite_or(r, 0.0).abs();
    let inner = finite_or(inner, 0.0).max(0.0);
    let outer = finite_or(outer, 0.0).max(0.0);
    let amount = finite_or(amount, 0.0).clamp(0.0, 1.0);
    let t = smoothstep(inner, outer, r);
    (1.0 - amount * t).clamp(0.0, 1.0)
}

/// Combined vignette: centre, focal length (natural `cos^4`) and an art darken.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vignette {
    /// Optical centre in UV space (usually `(0.5, 0.5)`).
    pub center: Vec2,
    /// Focal-length parameter controlling the `cos^4` falloff rate.
    pub focal: f32,
    /// Inner radius: no artistic darkening within it.
    pub inner: f32,
    /// Outer radius: full artistic darkening beyond it.
    pub outer: f32,
    /// Strength of the artistic darkening in `[0, 1]`.
    pub amount: f32,
}

impl Default for Vignette {
    /// A gentle default: centred, mild `cos^4`, light artistic darken.
    fn default() -> Self {
        Self {
            center: Vec2::splat(0.5),
            focal: 1.0,
            inner: 0.4,
            outer: 0.9,
            amount: 0.5,
        }
    }
}

impl Vignette {
    /// Construct, sanitizing the centre (finite), focal length (`>= EPSILON`),
    /// radii (non-negative) and amount (`[0, 1]`).
    #[must_use]
    pub fn new(center: Vec2, focal: f32, inner: f32, outer: f32, amount: f32) -> Self {
        Self {
            center: sanitize_vec2(center, Vec2::splat(0.5)),
            focal: finite_or(focal, 1.0).max(EPSILON),
            inner: finite_or(inner, 0.0).max(0.0),
            outer: finite_or(outer, 0.0).max(0.0),
            amount: finite_or(amount, 0.0).clamp(0.0, 1.0),
        }
    }

    /// Normalised radius of `uv` from the centre.
    #[inline]
    #[must_use]
    pub fn radius(&self, uv: Vec2) -> f32 {
        (sanitize_vec2(uv, self.center) - self.center).length()
    }

    /// Combined vignette multiplier at `uv`: the product of the natural `cos^4`
    /// term and the artistic darkening, clamped to `[0, 1]`.  Equals `1` at the
    /// centre and is non-increasing with radius.
    #[must_use]
    pub fn evaluate(&self, uv: Vec2) -> f32 {
        let r = self.radius(uv);
        let natural = natural_vignette(r, self.focal);
        let art = artistic_vignette(r, self.inner, self.outer, self.amount);
        (natural * art).clamp(0.0, 1.0)
    }

    /// Apply the vignette to a linear RGB colour (per-channel multiply).
    #[must_use]
    pub fn apply(&self, uv: Vec2, color: [f32; 3]) -> [f32; 3] {
        let v = self.evaluate(uv);
        [
            (finite_or(color[0], 0.0) * v).max(0.0),
            (finite_or(color[1], 0.0) * v).max(0.0),
            (finite_or(color[2], 0.0) * v).max(0.0),
        ]
    }
}

/// Clamp distortion coefficients to a finite range.
#[inline]
fn sanitize_k(k: f32) -> f32 {
    finite_or(k, 0.0).clamp(-4.0, 4.0)
}

/// Brown–Conrady radial distortion coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BrownConrady {
    /// Second-order radial coefficient `k1`.
    pub k1: f32,
    /// Fourth-order radial coefficient `k2`.
    pub k2: f32,
}

impl BrownConrady {
    /// Construct, clamping both coefficients to a finite, bounded range.
    #[must_use]
    pub fn new(k1: f32, k2: f32) -> Self {
        Self {
            k1: sanitize_k(k1),
            k2: sanitize_k(k2),
        }
    }

    /// Forward distortion of an ideal radius `r`: `r * (1 + k1 r² + k2 r⁴)`.
    #[inline]
    #[must_use]
    pub fn distort_radius(&self, r: f32) -> f32 {
        distort(r, self.k1, self.k2)
    }

    /// Inverse of [`BrownConrady::distort_radius`] via fixed-point iteration.
    #[inline]
    #[must_use]
    pub fn undistort_radius(&self, r_d: f32) -> f32 {
        undistort(r_d, self.k1, self.k2)
    }

    /// Forward-distort a UV point about `center`.
    ///
    /// Scales the offset `uv - center` by the radial distortion factor, so the
    /// centre is a fixed point.
    #[must_use]
    pub fn distort_point(&self, uv: Vec2, center: Vec2) -> Vec2 {
        let center = sanitize_vec2(center, Vec2::splat(0.5));
        let uv = sanitize_vec2(uv, center);
        let delta = uv - center;
        let r = delta.length();
        if r <= EPSILON {
            return center;
        }
        let factor = (self.distort_radius(r) / r).max(0.0);
        center + delta * factor
    }

    /// Inverse-distort a UV point about `center` (maps a distorted sample back
    /// to the ideal grid).
    #[must_use]
    pub fn undistort_point(&self, uv: Vec2, center: Vec2) -> Vec2 {
        let center = sanitize_vec2(center, Vec2::splat(0.5));
        let uv = sanitize_vec2(uv, center);
        let delta = uv - center;
        let r_d = delta.length();
        if r_d <= EPSILON {
            return center;
        }
        let r = self.undistort_radius(r_d);
        let factor = (r / r_d).max(0.0);
        center + delta * factor
    }
}

/// Forward Brown–Conrady radial distortion of a scalar radius.
///
/// Returns `r * (1 + k1 r² + k2 r⁴)`.  The multiplicative factor is floored to
/// zero so an aggressive negative coefficient cannot produce a negative radius,
/// and `r` is treated as `|r|`.  With `k1 = k2 = 0` this is the identity.
#[must_use]
pub fn distort(r: f32, k1: f32, k2: f32) -> f32 {
    let r = finite_or(r, 0.0).abs();
    let k1 = sanitize_k(k1);
    let k2 = sanitize_k(k2);
    let r2 = r * r;
    let factor = (1.0 + k1 * r2 + k2 * r2 * r2).max(0.0);
    r * factor
}

/// Inverse of [`distort`]: recover the ideal radius whose distortion is `r_d`.
///
/// Uses the fixed-point iteration `r <- r_d / (1 + k1 r² + k2 r⁴)` seeded at
/// `r = r_d`, which converges for the mild distortions used in practice.  The
/// denominator is floored to `EPSILON` so the iteration never divides by zero,
/// and the result is non-negative and finite.  With `k1 = k2 = 0` it is the
/// identity.
#[must_use]
pub fn undistort(r_d: f32, k1: f32, k2: f32) -> f32 {
    let r_d = finite_or(r_d, 0.0).abs();
    let k1 = sanitize_k(k1);
    let k2 = sanitize_k(k2);
    if r_d <= EPSILON {
        return r_d;
    }
    let mut r = r_d;
    for _ in 0..UNDISTORT_ITERATIONS {
        let r2 = r * r;
        let denom = (1.0 + k1 * r2 + k2 * r2 * r2).max(EPSILON);
        r = r_d / denom;
    }
    r.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    const EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn center_brightness_is_one() {
        let v = Vignette::default();
        assert!(approx(v.evaluate(v.center), 1.0, EPS), "centre not unity");
        assert!(approx(natural_vignette(0.0, 1.0), 1.0, EPS));
        assert!(approx(artistic_vignette(0.0, 0.4, 0.9, 0.5), 1.0, EPS));
    }

    #[test]
    fn edge_darkening_is_monotonic() {
        let v = Vignette::default();
        let dir = Vec2::new(1.0, 0.0);
        let mut prev = f32::INFINITY;
        for i in 0..=20 {
            let r = i as f32 / 20.0; // 0 .. 1
            let uv = v.center + dir * r;
            let cur = v.evaluate(uv);
            assert!(cur <= prev + EPS, "vignette rose at r={r}: {cur} > {prev}");
            assert!((0.0..=1.0).contains(&cur), "out of range at r={r}: {cur}");
            prev = cur;
        }
    }

    #[test]
    fn natural_matches_cos4_reference() {
        let f = 1.3_f32;
        for i in 0..=16 {
            let r = i as f32 / 16.0 * 1.5;
            let closed = natural_vignette(r, f);
            // Reference: cos(atan(r / f))^4 via transcendental ops.
            let theta = ops::atan(r / f);
            let c = ops::cos(theta);
            let reference = c * c * c * c;
            assert!(approx(closed, reference, 1.0e-5), "cos^4 mismatch at r={r}: {closed} vs {reference}");
        }
    }

    #[test]
    fn distort_identity_when_k_zero() {
        for i in 0..=10 {
            let r = i as f32 / 10.0;
            assert!(approx(distort(r, 0.0, 0.0), r, EPS), "distort k=0 not identity at r={r}");
            assert!(approx(undistort(r, 0.0, 0.0), r, EPS), "undistort k=0 not identity at r={r}");
        }
    }

    #[test]
    fn distort_undistort_round_trips() {
        let bc = BrownConrady::new(0.15, 0.03); // mild barrel
        for i in 1..=12 {
            let r = i as f32 / 12.0; // 0 .. 1 ideal radius
            let r_d = bc.distort_radius(r);
            let back = bc.undistort_radius(r_d);
            assert!(approx(back, r, 1.0e-4), "round trip failed at r={r}: back={back}");
        }
    }

    #[test]
    fn distort_undistort_round_trips_pincushion() {
        let bc = BrownConrady::new(-0.1, -0.02); // mild pincushion
        for i in 1..=12 {
            let r = i as f32 / 12.0;
            let r_d = bc.distort_radius(r);
            let back = bc.undistort_radius(r_d);
            assert!(approx(back, r, 1.0e-3), "pincushion round trip at r={r}: back={back}");
        }
    }

    #[test]
    fn point_distortion_fixes_the_center() {
        let bc = BrownConrady::new(0.2, 0.05);
        let center = Vec2::splat(0.5);
        assert!((bc.distort_point(center, center) - center).length() <= EPS);
        assert!((bc.undistort_point(center, center) - center).length() <= EPS);
    }

    #[test]
    fn point_round_trip_is_identity() {
        let bc = BrownConrady::new(0.12, 0.02);
        let center = Vec2::splat(0.5);
        let uv = Vec2::new(0.9, 0.2);
        let distorted = bc.distort_point(uv, center);
        let back = bc.undistort_point(distorted, center);
        assert!((back - uv).length() <= 1.0e-4, "point round trip failed: {back:?}");
    }

    #[test]
    fn non_finite_inputs_stay_finite() {
        assert!(natural_vignette(f32::NAN, f32::INFINITY).is_finite());
        assert!(artistic_vignette(f32::INFINITY, f32::NAN, 0.5, f32::INFINITY).is_finite());
        assert!(distort(f32::NAN, f32::INFINITY, f32::NAN).is_finite());
        assert!(undistort(f32::INFINITY, f32::NAN, f32::INFINITY).is_finite());
        let v = Vignette::new(Vec2::new(f32::NAN, 0.5), -1.0, f32::NAN, 0.0, 2.0);
        assert!(v.evaluate(Vec2::new(f32::INFINITY, 0.1)).is_finite());
        let bc = BrownConrady::new(f32::NAN, f32::INFINITY);
        let p = bc.distort_point(Vec2::new(f32::NAN, 0.3), Vec2::splat(0.5));
        assert!(p.x.is_finite() && p.y.is_finite());
    }
}
