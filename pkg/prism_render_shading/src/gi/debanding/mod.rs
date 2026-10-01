//! Dithering / debanding CPU golden references.
//!
//! Low-bit-depth quantization of a smooth gradient produces visible contour
//! lines ("banding"). Debanding hides them by adding a small, sub-step
//! perturbation *before* quantization so the local average of the quantized
//! pixels tracks the true value and the eye perceives a smooth ramp again.
//!
//! This is **quantization-time dithering**, deliberately distinct from the
//! photographic grain in [`crate::gi::film_grain`]: that pass models
//! signal-dependent sensor/emulsion noise, whereas this pass is a
//! signal-independent anti-banding perturbation tied to the output bit depth.
//!
//! The submodules provide the three ingredients:
//!
//! * [`ordered`] — periodic Bayer / dispersed-dot threshold maps
//!   ([`bayer_threshold`]), recursively generated and normalized to a zero-mean
//!   `[-0.5, 0.5)`.
//! * [`noise_dither`] — aperiodic dither sources (Interleaved Gradient Noise
//!   [`interleaved_gradient_noise`], hashed [`white_noise`]) and the
//!   triangular-PDF remap [`remap_tri`] / [`tpdf_dither`] that makes dither
//!   statistically ideal.
//! * [`quantize`] — round-to-nearest [`quantize`] and dithered
//!   [`quantize_dithered`] at a given bit depth, plus error metrics.
//!
//! [`apply_debanding`] ties them together: pick a [`DebandMethod`], derive the
//! appropriate dither for the pixel, scale by [`DebandParams::strength`], and
//! quantize. Everything is deterministic so the WESL / GPU twin reproduces it
//! pixel-for-pixel.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, IO, GPU, global state, or `unsafe`.
//! * Signals are display-referred and sanitized to `[0, 1]`; `bits` is clamped
//!   to the [`quantize`] range; outputs are finite and in `[0, 1]`.
//! * `strength = 0` reduces to plain undithered quantization.
//!
//! # References
//! * L. Schuchman, "Dither Signals and Their Effect on Quantization Noise",
//!   IEEE TCT, 1964.
//! * J. Jiménez, "Next Generation Post Processing in Call of Duty: Advanced
//!   Warfare", SIGGRAPH 2014.

pub mod noise_dither;
pub mod ordered;
pub mod quantize;

pub use noise_dither::{
    hash_coords, hash_u32, interleaved_gradient_noise, interleaved_gradient_noise_animated,
    remap_tri, to_centered, tpdf_dither, uniform01, white_noise,
};
pub use ordered::{
    bayer_threshold, bayer_threshold01, bayer_value, normalize_size, BAYER_2, BAYER_4, BAYER_8,
    MAX_BAYER_SIZE,
};
pub use quantize::{
    clamp_bits, dithered_error, levels, quantization_error, quantize, quantize_dithered,
    quantize_step, MAX_BITS, MIN_BITS,
};

/// Selects which dither source [`apply_debanding`] uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DebandMethod {
    /// Periodic Bayer / ordered threshold (one-LSB RPDF, from [`ordered`]).
    #[default]
    Ordered,
    /// Interleaved Gradient Noise, remapped to a one-LSB RPDF dither.
    Ign,
    /// Triangular-PDF white-noise dither (two-LSB, from [`noise_dither`]).
    Tpdf,
}

/// Tunable parameters for [`apply_debanding`].
///
/// `strength` scales the dither amplitude (`0` disables dithering, `1` is the
/// nominal amplitude); `seed` keys the hashed noise sources; `bayer_size` is
/// the ordered-dither tile edge (normalized to a power of two internally).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebandParams {
    /// Dither amplitude multiplier, clamped to `[0, 1]`.
    pub strength: f32,
    /// Seed for the hashed [`DebandMethod::Ign`] / [`DebandMethod::Tpdf`] sources.
    pub seed: u32,
    /// Ordered-dither tile edge; normalized by [`ordered::normalize_size`].
    pub bayer_size: u32,
}

impl Default for DebandParams {
    /// Full-strength dither, seed `0`, and an `8×8` Bayer tile.
    #[inline]
    fn default() -> Self {
        Self {
            strength: 1.0,
            seed: 0,
            bayer_size: 8,
        }
    }
}

/// Replaces a non-finite scalar with `fallback`.
#[inline]
#[must_use]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Computes the signed dither sample (in dither-step units) for one pixel.
///
/// `Ordered` and `Ign` return a one-LSB RPDF perturbation in `[-0.5, 0.5)`;
/// `Tpdf` returns a two-LSB triangular perturbation in `[-1, 1]`. The caller
/// scales this by `strength` and (in [`quantize_dithered`]) by one step `q`.
#[inline]
#[must_use]
fn dither_sample(x: u32, y: u32, method: DebandMethod, params: &DebandParams) -> f32 {
    match method {
        DebandMethod::Ordered => bayer_threshold(x, y, params.bayer_size),
        DebandMethod::Ign => {
            // IGN is seeded by offsetting the coordinate lattice; keep it stable
            // and in `[0, 1)` before centering to the one-LSB RPDF range.
            let sx = x.wrapping_add(params.seed) as f32;
            let sy = y as f32;
            to_centered(interleaved_gradient_noise(sx, sy))
        }
        DebandMethod::Tpdf => tpdf_dither(x, y, params.seed),
    }
}

/// High-level debanding: dither `value` for its pixel and quantize to `bits`.
///
/// Pipeline:
/// 1. Sanitize `value` to `[0, 1]` and clamp `strength` to `[0, 1]`.
/// 2. Draw the method's dither sample for pixel `(x, y)` and scale by
///    `strength` (see [`dither_sample`]).
/// 3. Return [`quantize_dithered`], i.e. `quantize(value + dither · q, bits)`.
///
/// With `strength = 0` this is exactly [`quantize`]. The result is
/// deterministic in every input, finite, and within `[0, 1]`.
#[inline]
#[must_use]
pub fn apply_debanding(
    value: f32,
    x: u32,
    y: u32,
    bits: u32,
    method: DebandMethod,
    params: DebandParams,
) -> f32 {
    let v = finite_or(value, 0.0).clamp(0.0, 1.0);
    let strength = finite_or(params.strength, 0.0).clamp(0.0, 1.0);
    let dither = dither_sample(x, y, method, &params) * strength;
    quantize_dithered(v, dither, bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Defaults are the documented full-strength 8×8 ordered configuration.
    #[test]
    fn default_params() {
        let p = DebandParams::default();
        assert_eq!(p.strength, 1.0);
        assert_eq!(p.seed, 0);
        assert_eq!(p.bayer_size, 8);
        assert_eq!(DebandMethod::default(), DebandMethod::Ordered);
    }

    /// `strength = 0` collapses to plain quantization for every method.
    #[test]
    fn zero_strength_is_plain_quantization() {
        let params = DebandParams {
            strength: 0.0,
            ..DebandParams::default()
        };
        for &m in &[DebandMethod::Ordered, DebandMethod::Ign, DebandMethod::Tpdf] {
            for &v in &[0.0_f32, 0.3, 0.5, 0.77, 1.0] {
                let out = apply_debanding(v, 3, 11, 8, m, params);
                assert_eq!(out, quantize(v, 8), "method {m:?} v={v}");
            }
        }
    }

    /// Output is always finite and within `[0, 1]`, even for hostile inputs.
    #[test]
    fn output_always_in_range() {
        let params = DebandParams::default();
        for &m in &[DebandMethod::Ordered, DebandMethod::Ign, DebandMethod::Tpdf] {
            for &v in &[f32::NAN, f32::INFINITY, -2.0, 0.0, 0.5, 1.0, 2.0] {
                for bits in [0u32, 1, 4, 8, 20] {
                    let out = apply_debanding(v, 7, 3, bits, m, params);
                    assert!(out.is_finite(), "non-finite out m={m:?} v={v} bits={bits}");
                    assert!((0.0..=1.0).contains(&out), "out={out}");
                }
            }
        }
    }

    /// Every method removes the DC bias of raw quantization on a tile.
    #[test]
    fn methods_reduce_dc_bias() {
        let value = 0.37_f32;
        let bits = 4;
        let undithered_bias = (quantize(value, bits) - value).abs() as f64;
        for &m in &[DebandMethod::Ordered, DebandMethod::Ign, DebandMethod::Tpdf] {
            let params = DebandParams {
                bayer_size: 16,
                ..DebandParams::default()
            };
            let mut sum = 0.0_f64;
            let mut n = 0.0_f64;
            for y in 0..64u32 {
                for x in 0..64u32 {
                    sum += apply_debanding(value, x, y, bits, m, params) as f64;
                    n += 1.0;
                }
            }
            let bias = (sum / n - value as f64).abs();
            assert!(
                bias < undithered_bias,
                "method {m:?} did not reduce bias: {bias} vs {undithered_bias}"
            );
            assert!(bias < 0.02, "method {m:?} residual bias {bias}");
        }
    }

    /// The result is deterministic across repeated calls.
    #[test]
    fn deterministic() {
        let params = DebandParams::default();
        let a = apply_debanding(0.4, 12, 34, 6, DebandMethod::Tpdf, params);
        let b = apply_debanding(0.4, 12, 34, 6, DebandMethod::Tpdf, params);
        assert_eq!(a, b);
    }
}
