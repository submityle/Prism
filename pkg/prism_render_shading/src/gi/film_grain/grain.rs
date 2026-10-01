//! Newson-style stochastic film-grain intensity — CPU golden reference.
//!
//! Photographic film grain is the random clumping of developed silver-halide
//! crystals. Perceptually it is a **signal-dependent**, high-frequency texture:
//! shadows and mid-tones carry visible grain while bright highlights look
//! comparatively clean (the emulsion saturates and the crystals pack densely).
//! This module synthesises that look from a *deterministic* hash so the WESL /
//! GPU twin can reproduce the exact same field pixel-for-pixel.
//!
//! It is deliberately distinct from the ordered / blue-noise dither in
//! [`crate::gi::sharpen::deband`]: that pass hides quantization banding with a
//! structured, signal-*independent* perturbation, whereas this pass models a
//! physically-motivated, luminance-weighted stochastic grain.
//!
//! The pipeline is:
//!
//! * [`hash_u32`] / [`hash_pixel`] — a PCG-style integer bit-mix giving a
//!   well-distributed 32-bit value from pixel coordinate + seed.
//! * [`uniform01`] — maps a hashed `u32` to a uniform `[0, 1)` float.
//! * [`value_noise`] — a smooth (bilinearly interpolated, smoothstep-faded)
//!   lattice noise in `[0, 1)` for low-frequency grain *clumping*.
//! * [`grain_sample`] — a zero-mean triangular white-noise sample in `(-1, 1)`
//!   combined with a touch of zero-mean value noise for clumping.
//! * [`luminance`] / [`luminance_response`] — Rec.709 luma and the grain gain
//!   curve: strong in shadows/mids, rolling to zero in the highlights.
//! * [`film_grain`] — the final per-pixel signed grain delta, response-weighted
//!   and scaled by `strength`.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, IO, GPU, global state, or `unsafe`.
//! * Integer hashing uses only wrapping bit arithmetic (Wang/PCG style).
//! * Transcendental-free; only `floor` (via [`bevy_math::ops`]) is used for the
//!   value-noise lattice.
//! * `strength` is clamped to `[0, 1]`; `strength = 0` is a strict zero grain.
//! * All float inputs are sanitized to be finite; outputs are finite and the
//!   signed grain stays within `[-strength, strength]`.

use bevy_math::{ops, UVec2, Vec3};

/// Rec.709 luma weight for the red channel.
const LUMA_R: f32 = 0.2126;
/// Rec.709 luma weight for the green channel.
const LUMA_G: f32 = 0.7152;
/// Rec.709 luma weight for the blue channel.
const LUMA_B: f32 = 0.0722;

/// Fraction of grain energy coming from low-frequency value-noise clumping.
///
/// The remainder is fine per-pixel white noise. Both terms are zero-mean, so
/// the mix stays zero-mean. Kept modest so grain reads as fine film texture
/// rather than blotches.
const CLUMP_MIX: f32 = 0.3;

/// Shadow-lift weight in [`luminance_response`].
///
/// Blends a `(1 - luma)` shadow term with the `4·l·(1-l)` mid-tone parabola so
/// dark regions keep grain (shadows are never fully clean) while the response
/// still peaks in the mid-tones and falls to zero at full white.
const SHADOW_BIAS: f32 = 0.4;

/// Replaces a non-finite scalar with `fallback`.
#[inline]
#[must_use]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Smoothstep fade `3t² - 2t³` for `t` in `[0, 1]` (Hermite, C¹ continuous).
#[inline]
#[must_use]
fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// PCG-style 32-bit integer hash: a deterministic, well-distributed bit mix.
///
/// Uses the standard RXS-M-XS-flavoured multiply/xorshift pipeline on wrapping
/// `u32` arithmetic. Pure and reproducible on any backend; adjacent inputs
/// avalanche to uncorrelated outputs.
#[must_use]
pub fn hash_u32(x: u32) -> u32 {
    // PCG output permutation on an LCG-advanced state.
    let state = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Combines two 32-bit lanes into one hash, decorrelating the axes.
#[inline]
#[must_use]
fn hash_combine(a: u32, b: u32) -> u32 {
    // Mix `b` in with a large odd constant (golden-ratio derived) then re-hash.
    hash_u32(a ^ b.wrapping_mul(0x9E37_79B9))
}

/// Hashes a pixel coordinate and `seed` to a 32-bit value.
///
/// Deterministic in all three inputs; distinct pixels and seeds avalanche to
/// independent outputs, which is what makes the grain field reproducible.
#[must_use]
pub fn hash_pixel(p: UVec2, seed: u32) -> u32 {
    let hx = hash_u32(p.x);
    let hy = hash_u32(p.y.wrapping_add(0x85EB_CA6B));
    hash_combine(hash_combine(hx, hy), hash_u32(seed))
}

/// Maps a hashed `u32` to a uniform float in `[0, 1)`.
///
/// Uses the top 24 bits divided by `2²⁴`, giving an exactly representable
/// uniform grid with no rounding to `1.0`.
#[inline]
#[must_use]
pub fn uniform01(h: u32) -> f32 {
    const SCALE: f32 = 1.0 / 16_777_216.0; // 1 / 2^24
    (h >> 8) as f32 * SCALE
}

/// Smooth lattice value noise in `[0, 1)` for integer-grid cell `(x, y)`.
///
/// Hashes the four surrounding lattice corners of the (fractional) coordinate
/// `(x, y)` and bilinearly blends them with a smoothstep fade, producing
/// low-frequency clumping. `seed` offsets the whole field. Non-finite
/// coordinates fall back to the origin.
#[must_use]
pub fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let xf = finite_or(x, 0.0);
    let yf = finite_or(y, 0.0);

    let x0 = ops::floor(xf);
    let y0 = ops::floor(yf);
    let fx = smoothstep01(xf - x0);
    let fy = smoothstep01(yf - y0);

    // Wrap lattice indices into u32 via a signed→unsigned bit cast.
    let ix = x0 as i32 as u32;
    let iy = y0 as i32 as u32;

    let c00 = uniform01(hash_pixel(UVec2::new(ix, iy), seed));
    let c10 = uniform01(hash_pixel(UVec2::new(ix.wrapping_add(1), iy), seed));
    let c01 = uniform01(hash_pixel(UVec2::new(ix, iy.wrapping_add(1)), seed));
    let c11 = uniform01(hash_pixel(UVec2::new(ix.wrapping_add(1), iy.wrapping_add(1)), seed));

    let top = c00 + (c10 - c00) * fx;
    let bot = c01 + (c11 - c01) * fx;
    (top + (bot - top) * fy).clamp(0.0, 1.0)
}

/// A zero-mean triangular white-noise sample in `(-1, 1)` for a pixel.
///
/// Formed as the difference of two independent uniforms drawn from decorrelated
/// hashes. The triangular PDF concentrates mass near zero, matching the gentle,
/// mostly-low-amplitude character of real grain while remaining strictly
/// zero-mean (so it cannot bias the image DC).
#[must_use]
pub fn triangular_noise(p: UVec2, seed: u32) -> f32 {
    let u1 = uniform01(hash_pixel(p, seed));
    let u2 = uniform01(hash_pixel(p, seed ^ 0x68E3_1DA4));
    // Difference of two uniforms: triangular on (-1, 1), mean 0.
    u1 - u2
}

/// The combined zero-mean grain sample in `(-1, 1)` for a pixel.
///
/// Mixes fine triangular white noise with a small amount of zero-mean value
/// noise for low-frequency clumping. Both terms are zero-mean so the sum is
/// zero-mean; the result is clamped defensively to `[-1, 1]`.
#[must_use]
pub fn grain_sample(p: UVec2, seed: u32) -> f32 {
    let white = triangular_noise(p, seed);
    // Value noise in [0,1) recentred to a zero-mean clump term in (-1, 1).
    let clump = 2.0 * value_noise(p.x as f32 / 3.0, p.y as f32 / 3.0, seed ^ 0x1234_5678) - 1.0;
    let mix = (1.0 - CLUMP_MIX) * white + CLUMP_MIX * clump;
    mix.clamp(-1.0, 1.0)
}

/// Rec.709 luminance of a linear-RGB colour, sanitized to be finite / `>= 0`.
#[must_use]
pub fn luminance(color: Vec3) -> f32 {
    let r = finite_or(color.x, 0.0).max(0.0);
    let g = finite_or(color.y, 0.0).max(0.0);
    let b = finite_or(color.z, 0.0).max(0.0);
    LUMA_R * r + LUMA_G * g + LUMA_B * b
}

/// Grain gain curve in `[0, 1]` as a function of luma.
///
/// Blends a shadow-lift term `(1 - l)` with the mid-tone parabola `4·l·(1-l)`
/// (peak `1` at `l = 0.5`). With [`SHADOW_BIAS`] the response is strong in
/// shadows, strongest in the mid-tones, and rolls to exactly `0` at full white
/// — the classic "shadows/mids grainy, highlights clean" film behaviour.
///
/// `luma` is clamped to `[0, 1]`; non-finite input yields `0`.
#[must_use]
pub fn luminance_response(luma: f32) -> f32 {
    let l = finite_or(luma, 0.0).clamp(0.0, 1.0);
    let mid = 4.0 * l * (1.0 - l);
    let shadow = 1.0 - l;
    (SHADOW_BIAS * shadow + (1.0 - SHADOW_BIAS) * mid).clamp(0.0, 1.0)
}

/// The final signed film-grain delta for a pixel.
///
/// Returns `grain_sample · luminance_response(luma) · strength`, a zero-mean
/// perturbation in `[-strength, strength]`. `strength` is clamped to `[0, 1]`;
/// `strength = 0` yields exactly `0`. Deterministic in `(p, luma, seed)`.
#[must_use]
pub fn film_grain(p: UVec2, luma: f32, seed: u32, strength: f32) -> f32 {
    let s = finite_or(strength, 0.0).clamp(0.0, 1.0);
    if s == 0.0 {
        return 0.0;
    }
    let gain = luminance_response(luma);
    let sample = grain_sample(p, seed);
    (sample * gain * s).clamp(-s, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The PCG hash avalanches: consecutive inputs give very different outputs,
    /// and it is a deterministic pure function.
    #[test]
    fn hash_is_deterministic_and_avalanches() {
        assert_eq!(hash_u32(12345), hash_u32(12345));
        let a = hash_u32(0);
        let b = hash_u32(1);
        assert_ne!(a, b);
        // Low-bit flip should flip many output bits (avalanche).
        let diff = (a ^ b).count_ones();
        assert!(diff >= 8, "weak avalanche: {diff} bits differ");
    }

    /// `uniform01` stays in `[0, 1)` across a sweep of hashed inputs.
    #[test]
    fn uniform01_range() {
        for i in 0..10_000u32 {
            let u = uniform01(hash_u32(i));
            assert!((0.0..1.0).contains(&u), "u={u} i={i}");
        }
        // Extremes map inside the half-open range.
        assert_eq!(uniform01(0), 0.0);
        assert!(uniform01(u32::MAX) < 1.0);
    }

    /// Value noise stays in `[0, 1)`, is deterministic, and is continuous
    /// (small coordinate steps give small output steps).
    #[test]
    fn value_noise_range_and_continuity() {
        for i in 0..64 {
            let x = i as f32 * 0.25;
            let y = (i as f32 * 0.17) + 1.0;
            let v = value_noise(x, y, 7);
            assert!((0.0..1.0).contains(&v), "v={v}");
        }
        assert_eq!(value_noise(3.2, 4.7, 9), value_noise(3.2, 4.7, 9));
        // Continuity: a tiny step cannot jump a large amount.
        let a = value_noise(10.0, 10.0, 3);
        let b = value_noise(10.01, 10.0, 3);
        assert!((a - b).abs() < 0.1, "a={a} b={b}");
        // Non-finite falls back to origin.
        assert_eq!(value_noise(f32::NAN, 2.0, 1), value_noise(0.0, 2.0, 1));
    }

    /// The triangular sample is zero-mean over many pixels and stays in (-1, 1).
    #[test]
    fn triangular_is_zero_mean_and_bounded() {
        let mut sum = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..64u32 {
            for x in 0..64u32 {
                let v = triangular_noise(UVec2::new(x, y), 42);
                assert!((-1.0..=1.0).contains(&v), "v={v}");
                sum += v as f64;
                n += 1.0;
            }
        }
        let mean = (sum / n).abs();
        assert!(mean < 0.02, "mean={mean} not near zero");
    }

    /// The combined grain sample is deterministic, bounded, and zero-mean.
    #[test]
    fn grain_sample_is_zero_mean_and_deterministic() {
        assert_eq!(grain_sample(UVec2::new(5, 9), 1), grain_sample(UVec2::new(5, 9), 1));
        let mut sum = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..96u32 {
            for x in 0..96u32 {
                let v = grain_sample(UVec2::new(x, y), 123);
                assert!((-1.0..=1.0).contains(&v), "v={v}");
                sum += v as f64;
                n += 1.0;
            }
        }
        let mean = (sum / n).abs();
        assert!(mean < 0.03, "mean={mean} not near zero");
    }

    /// Luminance uses Rec.709 weights and is non-negative / finite.
    #[test]
    fn luminance_rec709() {
        assert!((luminance(Vec3::new(1.0, 1.0, 1.0)) - 1.0).abs() < 1.0e-6);
        assert_eq!(luminance(Vec3::new(0.0, 1.0, 0.0)), LUMA_G);
        let s = luminance(Vec3::new(f32::NAN, -1.0, f32::INFINITY));
        assert!(s.is_finite() && s >= 0.0, "s={s}");
    }

    /// The response is zero at white, positive in shadows (shadow lift), and
    /// peaks in the mid-tones.
    #[test]
    fn luminance_response_shape() {
        let shadow = luminance_response(0.0);
        let mid = luminance_response(0.5);
        let high = luminance_response(1.0);
        assert!(shadow > 0.0, "shadows should keep grain: {shadow}");
        assert!(mid > shadow, "mids should exceed shadows: mid={mid} shadow={shadow}");
        assert!(high.abs() < 1.0e-6, "highlights should be clean: {high}");
        // Stays within [0, 1] across the full range.
        for i in 0..=100 {
            let r = luminance_response(i as f32 / 100.0);
            assert!((0.0..=1.0).contains(&r), "r={r}");
        }
    }

    /// `strength = 0` yields exactly zero grain.
    #[test]
    fn zero_strength_is_zero() {
        for &(x, y) in &[(0u32, 0u32), (3, 5), (17, 42), (100, 7)] {
            assert_eq!(film_grain(UVec2::new(x, y), 0.4, 11, 0.0), 0.0);
        }
    }

    /// The grain delta is deterministic, bounded by `strength`, and varies
    /// between neighbouring pixels.
    #[test]
    fn film_grain_bounded_and_varies() {
        let s = 0.5;
        let a = film_grain(UVec2::new(10, 20), 0.5, 7, s);
        assert_eq!(a, film_grain(UVec2::new(10, 20), 0.5, 7, s));
        assert!(a.abs() <= s + 1.0e-6, "a={a}");
        let b = film_grain(UVec2::new(11, 20), 0.5, 7, s);
        assert!(a != b, "expected spatial variation a={a} b={b}");
    }

    /// Averaged over a tile at fixed luma the grain delta is ~zero-mean: the
    /// perturbation does not shift the image's DC level.
    #[test]
    fn film_grain_mean_is_near_zero() {
        let mut sum = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..80u32 {
            for x in 0..80u32 {
                sum += film_grain(UVec2::new(x, y), 0.5, 99, 1.0) as f64;
                n += 1.0;
            }
        }
        let mean = (sum / n).abs();
        assert!(mean < 0.02, "mean={mean}");
    }

    /// Degenerate inputs stay finite.
    #[test]
    fn degenerate_inputs_are_finite() {
        let g = film_grain(UVec2::new(1, 1), f32::NAN, 0, f32::INFINITY);
        assert!(g.is_finite(), "g={g}");
    }
}
