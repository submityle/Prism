//! Noise dithering — interleaved-gradient / white noise + TPDF remap.
//!
//! Where [`crate::gi::debanding::ordered`] uses a periodic Bayer threshold,
//! this module supplies *aperiodic* dither sources and the triangular-PDF
//! remap that makes dither statistically ideal for quantization:
//!
//! * [`interleaved_gradient_noise`] — Jiménez's "Interleaved Gradient Noise"
//!   (SIGGRAPH 2014). A single cheap closure of pixel coordinates that looks
//!   pleasingly blue-noise-like under a `3×3` neighbourhood, in `[0, 1)`.
//! * [`white_noise`] — a PCG-style hashed uniform in `[0, 1)` keyed by pixel
//!   coordinate and a seed; spectrally flat but fully decorrelated.
//! * [`remap_tri`] — maps **two** independent uniform `[0, 1)` draws to a
//!   symmetric **triangular** distribution on `[-1, 1]`. The difference of two
//!   rectangular (uniform) sources is the classic *triangular-PDF (TPDF)*
//!   dither: it has exactly zero mean and, crucially, decouples the quantization
//!   error's first *and* second moments from the signal, eliminating the noise
//!   modulation that a single rectangular (RPDF) dither leaves behind.
//! * [`tpdf_dither`] — convenience: draws two decorrelated white-noise uniforms
//!   for a pixel and returns the TPDF sample in `[-1, 1]`.
//!
//! All functions are deterministic and transcendental-free (only `fract` via
//! [`bevy_math::ops`] and integer bit-mixing), so the WESL / GPU twin can
//! reproduce every sample exactly.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, IO, GPU, global state, or `unsafe`.
//! * Float inputs are sanitized to be finite before use; outputs are finite.
//! * Uniform samples are in `[0, 1)`; the TPDF sample is in `[-1, 1]`.
//! * Integer hashing uses only wrapping arithmetic (PCG / Wang style).
//!
//! # References
//! * J. Jiménez, "Next Generation Post Processing in Call of Duty: Advanced
//!   Warfare", SIGGRAPH 2014 (Interleaved Gradient Noise).
//! * R. A. Wannamaker et al., "A Theory of Nonsubtractive Dither", IEEE Trans.
//!   Signal Processing, 2000 (TPDF properties).
//! * M. Jarzynski & M. Olano, "Hash Functions for GPU Rendering", JCGT 2020.

/// Replaces a non-finite scalar with `fallback`.
#[inline]
#[must_use]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Interleaved Gradient Noise (Jiménez 2014) at pixel `(x, y)`, in `[0, 1)`.
///
/// Evaluates `fract(C₂ · fract(C₀·x + C₁·y))` with the original magic
/// constants `C₀ = 0.06711056`, `C₁ = 0.00583715`, `C₂ = 52.9829189`. The inner
/// `fract` keeps the argument bounded so the outer product does not lose
/// precision; the result approximates a blue-noise spectrum over small
/// neighbourhoods at essentially no cost.
///
/// Inputs are sanitized to finite values; the output is finite and in `[0, 1)`.
#[inline]
#[must_use]
pub fn interleaved_gradient_noise(x: f32, y: f32) -> f32 {
    const C0: f32 = 0.067_110_56;
    const C1: f32 = 0.005_837_15;
    const C2: f32 = 52.982_918_9;
    let x = finite_or(x, 0.0);
    let y = finite_or(y, 0.0);
    let inner = (C0 * x + C1 * y).fract();
    let v = (C2 * inner.fract()).fract();
    // `fract` can return a tiny negative value for negative arguments; fold it
    // back into `[0, 1)` so the contract holds for any coordinate sign.
    if v < 0.0 { v + 1.0 } else { v }
}

/// Animated IGN: offsets the coordinate by a per-frame golden shift.
///
/// Adds `frame · 5.588238` to `x` (the shift Jiménez recommends for temporal
/// animation) so successive frames draw decorrelated-but-stable patterns. The
/// output remains a finite uniform in `[0, 1)`.
#[inline]
#[must_use]
pub fn interleaved_gradient_noise_animated(x: f32, y: f32, frame: u32) -> f32 {
    const FRAME_SHIFT: f32 = 5.588_238;
    interleaved_gradient_noise(x + (frame as f32) * FRAME_SHIFT, y)
}

/// PCG-style 32-bit integer hash: a deterministic, well-distributed bit mix.
///
/// Standard RXS-M-XS permutation over an LCG step, on wrapping arithmetic.
/// Adjacent inputs avalanche to uncorrelated outputs.
#[inline]
#[must_use]
pub fn hash_u32(x: u32) -> u32 {
    let state = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Combines two 32-bit lanes into one hash, decorrelating the axes.
#[inline]
#[must_use]
fn hash_combine(a: u32, b: u32) -> u32 {
    hash_u32(a ^ b.wrapping_mul(0x9E37_79B9))
}

/// Hashes a pixel coordinate and `seed` to a 32-bit value.
///
/// Deterministic in all three inputs; distinct pixels/seeds avalanche to
/// independent outputs.
#[inline]
#[must_use]
pub fn hash_coords(x: u32, y: u32, seed: u32) -> u32 {
    let hx = hash_u32(x);
    let hy = hash_u32(y.wrapping_add(0x85EB_CA6B));
    hash_combine(hash_combine(hx, hy), hash_u32(seed))
}

/// Maps a hashed `u32` to a uniform float in `[0, 1)`.
///
/// Uses the top 24 bits divided by `2²⁴`, an exactly representable uniform grid
/// that never rounds up to `1.0`.
#[inline]
#[must_use]
pub fn uniform01(h: u32) -> f32 {
    const SCALE: f32 = 1.0 / 16_777_216.0; // 1 / 2^24
    (h >> 8) as f32 * SCALE
}

/// Hashed white-noise uniform in `[0, 1)` for pixel `(x, y)` and `seed`.
///
/// Spectrally flat (no blue-noise shaping) but fully decorrelated and
/// deterministic — the raw RPDF source for [`tpdf_dither`].
#[inline]
#[must_use]
pub fn white_noise(x: u32, y: u32, seed: u32) -> f32 {
    uniform01(hash_coords(x, y, seed))
}

/// Remaps two independent uniform `[0, 1)` draws to a triangular `[-1, 1]`.
///
/// Returns `u1 − u2`. The difference of two independent rectangular (uniform)
/// variates has a symmetric triangular PDF on `(-1, 1)` peaked at `0`, with
/// mean exactly `0`. This is the canonical **TPDF dither**: subtracting one
/// uniform source from another yields the two-LSB-wide triangular perturbation
/// that renders the total quantization error independent of the signal (zero
/// noise modulation), unlike a single RPDF dither.
///
/// Both inputs are sanitized and clamped to `[0, 1)` before use, so the output
/// is always finite and within `[-1, 1]`.
#[inline]
#[must_use]
pub fn remap_tri(u1: f32, u2: f32) -> f32 {
    let a = finite_or(u1, 0.5).clamp(0.0, 1.0);
    let b = finite_or(u2, 0.5).clamp(0.0, 1.0);
    a - b
}

/// Triangular-PDF dither sample in `[-1, 1]` for pixel `(x, y)` and `seed`.
///
/// Draws two decorrelated white-noise uniforms (the second keyed by a
/// perturbed seed so the two draws are independent) and feeds them to
/// [`remap_tri`]. Deterministic, finite, zero-mean over many pixels.
#[inline]
#[must_use]
pub fn tpdf_dither(x: u32, y: u32, seed: u32) -> f32 {
    let u1 = white_noise(x, y, seed);
    let u2 = white_noise(x, y, seed.wrapping_add(0x632B_E593));
    remap_tri(u1, u2)
}

/// Maps a `[0, 1)` uniform to the zero-mean dither range `[-0.5, 0.5)`.
///
/// A convenience for feeding a single RPDF source (IGN or white noise) into the
/// one-LSB-wide dither convention shared with [`crate::gi::debanding::ordered`].
#[inline]
#[must_use]
pub fn to_centered(u: f32) -> f32 {
    finite_or(u, 0.5).clamp(0.0, 1.0) - 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    /// IGN is deterministic and stays within `[0, 1)` across a tile.
    #[test]
    fn ign_range_and_determinism() {
        for y in 0..64 {
            for x in 0..64 {
                let xf = x as f32;
                let yf = y as f32;
                let v = interleaved_gradient_noise(xf, yf);
                assert!(v.is_finite(), "non-finite IGN at {x},{y}");
                assert!((0.0..1.0).contains(&v), "v={v}");
                assert_eq!(v, interleaved_gradient_noise(xf, yf));
            }
        }
    }

    /// IGN handles negative and non-finite coordinates without leaving `[0, 1)`.
    #[test]
    fn ign_robust_inputs() {
        for &(x, y) in &[(-3.0, -7.0), (f32::NAN, 2.0), (1.0, f32::INFINITY)] {
            let v = interleaved_gradient_noise(x, y);
            assert!(v.is_finite(), "non-finite IGN for {x},{y}");
            assert!((0.0..1.0).contains(&v), "v={v}");
        }
    }

    /// Averaged over a tile IGN is close to the uniform mean of `0.5`.
    #[test]
    fn ign_mean_near_half() {
        let mut sum = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..128 {
            for x in 0..128 {
                sum += interleaved_gradient_noise(x as f32, y as f32) as f64;
                n += 1.0;
            }
        }
        let mean = sum / n;
        assert!((mean - 0.5).abs() < 0.02, "mean={mean}");
    }

    /// The animated variant equals the static one at frame zero and differs
    /// afterwards.
    #[test]
    fn ign_animation_offset() {
        let base = interleaved_gradient_noise(10.0, 20.0);
        assert_eq!(base, interleaved_gradient_noise_animated(10.0, 20.0, 0));
        let next = interleaved_gradient_noise_animated(10.0, 20.0, 1);
        assert!(base != next, "frames should decorrelate: {base} vs {next}");
    }

    /// White noise is deterministic, in `[0, 1)`, and varies across pixels.
    #[test]
    fn white_noise_basic() {
        let a = white_noise(4, 9, 7);
        assert_eq!(a, white_noise(4, 9, 7));
        assert!((0.0..1.0).contains(&a), "a={a}");
        assert!(a != white_noise(5, 9, 7), "spatial variation expected");
        assert!(a != white_noise(4, 9, 8), "seed variation expected");
    }

    /// White noise averages to ~0.5 over a tile (unbiased uniform).
    #[test]
    fn white_noise_mean_near_half() {
        let mut sum = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..100u32 {
            for x in 0..100u32 {
                sum += white_noise(x, y, 1234) as f64;
                n += 1.0;
            }
        }
        let mean = sum / n;
        assert!((mean - 0.5).abs() < 0.01, "mean={mean}");
    }

    /// `remap_tri` is bounded to `[-1, 1]`, zero-mean, and symmetric.
    #[test]
    fn remap_tri_bounds_and_symmetry() {
        assert!((remap_tri(0.5, 0.5)).abs() < 1.0e-7);
        assert_eq!(remap_tri(0.0, 0.0), 0.0);
        // Full span endpoints.
        assert!(remap_tri(0.999, 0.0) <= 1.0);
        assert!(remap_tri(0.0, 0.999) >= -1.0);
        // Antisymmetry: swapping the draws negates the result.
        let (u, v) = (0.3, 0.8);
        assert!((remap_tri(u, v) + remap_tri(v, u)).abs() < 1.0e-7);
    }

    /// The TPDF sample is bounded and its mean over many pixels is ~zero.
    #[test]
    fn tpdf_dither_zero_mean() {
        let mut sum = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..100u32 {
            for x in 0..100u32 {
                let d = tpdf_dither(x, y, 99);
                assert!((-1.0..=1.0).contains(&d), "d={d}");
                sum += d as f64;
                n += 1.0;
            }
        }
        let mean = (sum / n).abs();
        assert!(mean < 0.01, "mean={mean}");
    }

    /// The TPDF variance approaches the theoretical `1/6` of a unit-width
    /// triangular distribution on `[-1, 1]`.
    #[test]
    fn tpdf_variance_matches_theory() {
        let mut sum = 0.0_f64;
        let mut sq = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..200u32 {
            for x in 0..200u32 {
                let d = tpdf_dither(x, y, 5) as f64;
                sum += d;
                sq += d * d;
                n += 1.0;
            }
        }
        let mean = sum / n;
        let var = sq / n - mean * mean;
        // Triangular on [-1, 1] has variance 1/6 ≈ 0.1667.
        assert!((var - 1.0 / 6.0).abs() < 0.01, "var={var}");
    }

    /// `to_centered` maps `[0, 1)` into `[-0.5, 0.5)`.
    #[test]
    fn to_centered_range() {
        assert!((to_centered(0.5)).abs() < 1.0e-7);
        assert_eq!(to_centered(0.0), -0.5);
        assert!(to_centered(0.999) < 0.5);
        assert_eq!(to_centered(f32::NAN), 0.0);
    }
}
