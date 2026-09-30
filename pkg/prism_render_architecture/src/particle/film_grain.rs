//! Animated film-grain post-processing, the device-free `CPU` gold standard
//! (design §16, §21).
//!
//! Physical film records light on silver-halide crystals of finite size, so a
//! developed frame carries a stochastic, per-frame speckle whose visibility is
//! *luminance dependent*: shadows and midtones show the grain plainly while
//! bright highlights bleach it out. Modern engines re-create this look as a
//! screen-space post pass rather than shipping baked grain plates. This module
//! is the `CPU`-verifiable contract for that pass: a future `WESL` kernel binds
//! the same parameters and reproduces the identical `RGBA` result bit for bit.
//!
//! The pipeline is four self-contained integer/rational pieces:
//!
//! * **Avalanche hash.** [`hash_u32`] is a hand-rolled `xor-shift` plus
//!   odd-constant-multiply finalizer with strong bit diffusion, so one input
//!   bit flips roughly half the output bits. [`grain_hash01`] decorrelates the
//!   pixel coordinates and the frame index with odd multipliers, runs the mix,
//!   and normalizes the top 24 bits into `[0, 1)`.
//! * **Sized value noise.** [`value_noise01`] quantizes the pixel grid into
//!   integer *cells* of a chosen size, hashes the four cell corners, and blends
//!   them with a `smoothstep` (`t*t*(3-2t)`) interpolation. A large cell yields
//!   low-frequency (coarse) grain; a cell of one pixel yields the finest,
//!   per-pixel grain. No transcendental function is used.
//! * **Luminance response.** [`FilmGrainParams::response`] is a rational
//!   polynomial (no `exp`/`pow`): a shadow-boosted numerator over a
//!   highlight-rolled-off denominator, so the grain weight strictly decreases
//!   from shadows through midtones to highlights.
//! * **Blend.** [`FilmGrainParams::apply`] composites the signed grain onto the
//!   `RGB` color with either an additive blend or a polynomial (`Pegtop`) soft
//!   light, both clamped to `[0, 1]`.
//!
//! Unlike [`super::temporal_dither`] (an *ordered* `Bayer` threshold for
//! stochastic alpha), this module produces *random* hashed grain modulated by a
//! luminance response, and it composites in the color domain — it never touches
//! an alpha test or an alpha dissolve the way [`super::alpha_hashed`] and
//! [`super::alpha_erosion`] do.
//!
//! Every value is produced with integer arithmetic plus `f32::floor`/`sqrt`
//! (neither of which this module actually needs beyond `floor`-free integer
//! quantization), `smoothstep`, and rational polynomials, so the reference is
//! deterministic and portable. The parameter block is exported as a flat
//! `std430` uniform layout via [`FilmGrainParams::to_std430`].

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Odd decorrelation multiplier applied to the pixel `x` coordinate before the
/// avalanche mix (a bijection modulo `2^32`).
const ODD_X: u32 = 0x9E37_79B1;

/// Odd decorrelation multiplier applied to the pixel `y` coordinate.
const ODD_Y: u32 = 0x85EB_CA77;

/// Odd decorrelation multiplier applied to the frame index, giving each frame a
/// distinct grain field.
const ODD_FRAME: u32 = 0xC2B2_AE3D;

/// First odd multiply constant of the [`hash_u32`] finalizer.
const MIX_A: u32 = 0x7FEB_352D;

/// Second odd multiply constant of the [`hash_u32`] finalizer.
const MIX_B: u32 = 0x846C_A68B;

/// `2^24`, the largest power of two exactly representable in an `f32` mantissa;
/// used as the normalization divisor so every hashed value is exact.
const NORM_24: f32 = 16_777_216.0;

/// `Rec. 709` luma weight for the red channel.
const LUMA_R: f32 = 0.2126;

/// `Rec. 709` luma weight for the green channel.
const LUMA_G: f32 = 0.7152;

/// `Rec. 709` luma weight for the blue channel.
const LUMA_B: f32 = 0.0722;

/// Widens a small non-negative `u32` to `f32`.
///
/// Every caller passes either a hash reduced to 24 bits or a within-cell offset
/// strictly below the cell size, both far below `2^24`, so the widening is
/// exact and loses no precision. The crate does not enable the pedantic cast
/// lints, matching the surrounding `as f32` arithmetic in sibling modules.
fn u32_to_f32(v: u32) -> f32 {
    v as f32
}

/// Clamps a scalar into the closed unit interval `[0, 1]` without an `f32`
/// equality comparison.
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// The cubic `smoothstep` easing `t*t*(3-2t)` after clamping `t` into `[0, 1]`.
///
/// This is the transcendental-free `S`-curve used to blend adjacent grain
/// cells: flat slope at both ends removes the faceting a linear blend leaves.
fn smoothstep01(t: f32) -> f32 {
    let t = clamp01(t);
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation `a + (b - a) * t`.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Pure `u32` avalanche hash: a `xor-shift` and odd-constant-multiply finalizer
/// with strong bit diffusion.
///
/// The arithmetic is entirely integer (wrapping multiplies and logical shifts),
/// so the grain field is deterministic and reproduces bit for bit on a `GPU`
/// kernel. Flipping one input bit changes roughly half of the output bits.
#[must_use]
pub fn hash_u32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(MIX_A);
    h ^= h >> 15;
    h = h.wrapping_mul(MIX_B);
    h ^= h >> 16;
    h
}

/// A per-pixel, per-frame hashed value in `[0, 1)`.
///
/// The pixel coordinates and frame index are decorrelated by odd multipliers,
/// combined, and run through [`hash_u32`]; the top 24 bits are normalized so
/// the result is exact and strictly below one. Identical inputs always agree;
/// distinct inputs almost always differ.
#[must_use]
pub fn grain_hash01(x: u32, y: u32, frame: u32) -> f32 {
    let a = x.wrapping_mul(ODD_X);
    let b = y.wrapping_mul(ODD_Y);
    let c = frame.wrapping_mul(ODD_FRAME);
    let h = hash_u32(a ^ b ^ c);
    u32_to_f32(h >> 8) / NORM_24
}

/// The within-cell fraction of `coord` along one axis, in `[0, 1)`.
///
/// `cell` is the (clamped) integer cell size; the remainder `coord % cell` is
/// always strictly below `cell`, so the quotient lands in `[0, 1)`.
fn cell_fraction(coord: u32, cell: u32) -> f32 {
    u32_to_f32(coord % cell) / u32_to_f32(cell)
}

/// Sized value noise in `[0, 1)` sampled at pixel `(x, y)` for `frame`.
///
/// The pixel grid is quantized into integer cells of `cell_size` pixels (a size
/// of zero is treated as one). The four surrounding cell corners are hashed
/// with [`grain_hash01`] and bilinearly blended using a [`smoothstep01`] of the
/// within-cell fractions, so a larger `cell_size` produces coarser, lower
/// frequency grain and `cell_size == 1` produces the finest per-pixel grain.
#[must_use]
pub fn value_noise01(x: u32, y: u32, frame: u32, cell_size: u32) -> f32 {
    let cell = cell_size.max(1);
    let gx = x / cell;
    let gy = y / cell;
    let fx = cell_fraction(x, cell);
    let fy = cell_fraction(y, cell);
    let sx = smoothstep01(fx);
    let sy = smoothstep01(fy);
    let gx1 = gx.wrapping_add(1);
    let gy1 = gy.wrapping_add(1);
    let c00 = grain_hash01(gx, gy, frame);
    let c10 = grain_hash01(gx1, gy, frame);
    let c01 = grain_hash01(gx, gy1, frame);
    let c11 = grain_hash01(gx1, gy1, frame);
    let top = lerp(c00, c10, sx);
    let bottom = lerp(c01, c11, sx);
    lerp(top, bottom, sy)
}

/// Signed grain in `[-1, 1)` sampled at pixel `(x, y)` for `frame`.
///
/// Maps the unit-interval [`value_noise01`] into a zero-centered speckle so it
/// can darken or lighten the base color symmetrically.
#[must_use]
pub fn signed_grain(x: u32, y: u32, frame: u32, cell_size: u32) -> f32 {
    value_noise01(x, y, frame, cell_size) * 2.0 - 1.0
}

/// `Rec. 709` relative luminance of a linear `RGB` triple, clamped to `[0, 1]`.
#[must_use]
pub fn luma709(color_rgb: [f32; 3]) -> f32 {
    clamp01(LUMA_R * color_rgb[0] + LUMA_G * color_rgb[1] + LUMA_B * color_rgb[2])
}

/// How the signed grain is composited onto the base color.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GrainBlend {
    /// Add the signed grain directly, then clamp to `[0, 1]`.
    Additive,
    /// Polynomial (`Pegtop`) soft light: gentler, contrast-preserving grain that
    /// leaves solid black and white nearly untouched.
    SoftLight,
}

impl GrainBlend {
    /// The stable `std430` discriminant code for this blend mode.
    #[must_use]
    pub fn code(self) -> u32 {
        match self {
            Self::Additive => 0,
            Self::SoftLight => 1,
        }
    }
}

/// Number of scalar fields packed into the [`FilmGrainParams`] `std430` block.
const FIELD_COUNT: usize = 6;

/// Bundled film-grain parameters plus the sampling and compositing logic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmGrainParams {
    /// Overall grain amplitude; `0.0` disables the effect (identity).
    pub intensity: f32,
    /// Grain cell size in pixels (`0` is treated as `1`); larger is coarser.
    pub cell_size: u32,
    /// Extra grain weight lifted into the shadows by the response numerator.
    pub shadow_boost: f32,
    /// How sharply the response rolls the grain off in the highlights.
    pub highlight_rolloff: f32,
    /// The compositing mode.
    pub blend: GrainBlend,
}

impl FilmGrainParams {
    /// Byte size of the `std430` packing: [`FIELD_COUNT`] scalars rounded up to
    /// whole `vec4` slots so the block honors the 16-byte `std430` base
    /// alignment a `GPU` uniform block expects.
    pub const STD430_SIZE: usize = FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

    /// Builds a parameter block from its fields.
    #[must_use]
    pub fn new(
        intensity: f32,
        cell_size: u32,
        shadow_boost: f32,
        highlight_rolloff: f32,
        blend: GrainBlend,
    ) -> Self {
        Self {
            intensity,
            cell_size,
            shadow_boost,
            highlight_rolloff,
            blend,
        }
    }

    /// The luminance response weight in `(0, 1 + shadow_boost]` for luma `l`.
    ///
    /// A rational polynomial (no transcendental function): the shadow-boosted
    /// numerator `1 + shadow_boost * (1 - l)` strictly decreases in `l` while the
    /// denominator `1 + highlight_rolloff * l * l` strictly increases, so the
    /// weight is strictly larger in the shadows than in the highlights. Both
    /// tunables are clamped non-negative so the denominator can never vanish.
    #[must_use]
    pub fn response(&self, l: f32) -> f32 {
        let l = clamp01(l);
        let boost = self.shadow_boost.max(0.0);
        let rolloff = self.highlight_rolloff.max(0.0);
        let numerator = 1.0 + boost * (1.0 - l);
        let denominator = 1.0 + rolloff * l * l;
        numerator / denominator
    }

    /// The signed grain amount to composite at pixel `(x, y)` for `frame` given
    /// the base color, i.e. `signed_grain * response(luma) * intensity`.
    #[must_use]
    pub fn grain_amount(&self, color_rgb: [f32; 3], x: u32, y: u32, frame: u32) -> f32 {
        let grain = signed_grain(x, y, frame, self.cell_size);
        let weight = self.response(luma709(color_rgb));
        grain * weight * self.intensity
    }

    /// Composites the grain onto one channel value using this block's blend.
    ///
    /// Both paths clamp into `[0, 1]`. The additive path adds the signed amount
    /// directly; the soft-light path maps the amount into a `[0, 1]` blend layer
    /// centered on `0.5` and applies the polynomial `Pegtop` soft light
    /// `(1 - 2b) * a * a + 2 * a * b`, which is the identity when `b == 0.5`.
    fn composite_channel(&self, base: f32, amount: f32) -> f32 {
        match self.blend {
            GrainBlend::Additive => clamp01(base + amount),
            GrainBlend::SoftLight => {
                let a = clamp01(base);
                let b = clamp01(0.5 + 0.5 * amount);
                clamp01((1.0 - 2.0 * b) * a * a + 2.0 * a * b)
            }
        }
    }

    /// Applies film grain to a linear `RGB` triple at pixel `(x, y)` for
    /// `frame`, returning the composited color with every channel in `[0, 1]`.
    ///
    /// The grain is monochrome: one signed amount, derived from the pixel's
    /// luminance response, modulates all three channels equally, matching how
    /// physical film grain rides the whole image rather than one primary.
    #[must_use]
    pub fn apply(&self, color_rgb: [f32; 3], x: u32, y: u32, frame: u32) -> [f32; 3] {
        let amount = self.grain_amount(color_rgb, x, y, frame);
        [
            self.composite_channel(color_rgb[0], amount),
            self.composite_channel(color_rgb[1], amount),
            self.composite_channel(color_rgb[2], amount),
        ]
    }

    /// Packs the parameters into their `std430` uniform-block bytes.
    ///
    /// Little-endian words in order: `intensity`, `shadow_boost`,
    /// `highlight_rolloff` as `f32`; `cell_size` and the blend code as `u32`;
    /// the trailing words are zero padding so the block spans whole `vec4`
    /// slots ([`FilmGrainParams::STD430_SIZE`] bytes).
    #[must_use]
    pub fn to_std430(&self) -> [u8; Self::STD430_SIZE] {
        let words: [[u8; 4]; 8] = [
            self.intensity.to_le_bytes(),
            self.shadow_boost.to_le_bytes(),
            self.highlight_rolloff.to_le_bytes(),
            self.cell_size.to_le_bytes(),
            self.blend.code().to_le_bytes(),
            0u32.to_le_bytes(),
            0u32.to_le_bytes(),
            0u32.to_le_bytes(),
        ];
        let mut bytes = [0u8; Self::STD430_SIZE];
        for (slot, word) in bytes.chunks_exact_mut(4).zip(words.iter()) {
            slot.copy_from_slice(word);
        }
        bytes
    }

    /// Packs a slice of parameter blocks into one contiguous `std430` byte
    /// buffer (element stride [`FilmGrainParams::STD430_SIZE`]), the layout a
    /// `GPU` storage array of grain params binds.
    #[must_use]
    pub fn pack_slice(params: &[Self]) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(Self::STD430_SIZE * params.len());
        for p in params {
            buffer.extend_from_slice(&p.to_std430());
        }
        buffer
    }

    /// Total `std430` byte size of a storage buffer holding `count` packed
    /// blocks, clamped up to a single element per the shared [`storage_bytes`]
    /// rule.
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(Self::STD430_SIZE, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the `f32` comparisons the tests use in place of a
    /// direct `==` on floating point.
    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn params() -> FilmGrainParams {
        FilmGrainParams::new(0.25, 3, 1.5, 4.0, GrainBlend::Additive)
    }

    #[test]
    fn hash_is_deterministic() {
        for i in 0..256u32 {
            assert_eq!(hash_u32(i), hash_u32(i));
        }
    }

    #[test]
    fn hash01_stays_in_unit_interval() {
        for x in 0..40u32 {
            for y in 0..40u32 {
                let v = grain_hash01(x, y, 7);
                assert!((0.0..1.0).contains(&v));
            }
        }
    }

    #[test]
    fn hash_avalanche_flips_many_bits() {
        // Adjacent inputs should differ in a large fraction of their output
        // bits on average — a hallmark of a good avalanche finalizer.
        let mut total = 0u32;
        let samples = 512u32;
        for i in 0..samples {
            let d = hash_u32(i) ^ hash_u32(i.wrapping_add(1));
            total += d.count_ones();
        }
        let avg = total / samples;
        assert!(avg >= 12, "average hamming distance too low: {avg}");
    }

    #[test]
    fn hash01_decorrelates_axes_and_frame() {
        let base = grain_hash01(3, 5, 9);
        assert!(!approx(base, grain_hash01(4, 5, 9)));
        assert!(!approx(base, grain_hash01(3, 6, 9)));
        assert!(!approx(base, grain_hash01(3, 5, 10)));
    }

    #[test]
    fn value_noise_stays_in_unit_interval() {
        for x in 0..64u32 {
            for y in 0..64u32 {
                let v = value_noise01(x, y, 2, 5);
                assert!((0.0..1.0).contains(&v));
            }
        }
    }

    #[test]
    fn value_noise_is_deterministic() {
        assert!(approx(
            value_noise01(11, 23, 4, 6),
            value_noise01(11, 23, 4, 6)
        ));
    }

    #[test]
    fn value_noise_zero_cell_is_treated_as_one() {
        assert!(approx(
            value_noise01(9, 13, 1, 0),
            value_noise01(9, 13, 1, 1)
        ));
    }

    #[test]
    fn value_noise_anchors_at_cell_corners() {
        // At an exact cell corner the fractions are zero, so the noise equals
        // the raw corner hash.
        let cell = 4;
        let gx = 3u32;
        let gy = 2u32;
        let v = value_noise01(gx * cell, gy * cell, 8, cell);
        assert!(approx(v, grain_hash01(gx, gy, 8)));
    }

    #[test]
    fn value_noise_interpolates_within_corner_bounds() {
        // A bilinear smoothstep blend can never leave the min/max of the four
        // corner hashes it blends.
        let cell = 8;
        let gx = 1u32;
        let gy = 1u32;
        let c00 = grain_hash01(gx, gy, 3);
        let c10 = grain_hash01(gx + 1, gy, 3);
        let c01 = grain_hash01(gx, gy + 1, 3);
        let c11 = grain_hash01(gx + 1, gy + 1, 3);
        let lo = c00.min(c10).min(c01).min(c11);
        let hi = c00.max(c10).max(c01).max(c11);
        let v = value_noise01(gx * cell + 3, gy * cell + 5, 3, cell);
        assert!(v >= lo - CMP_EPS && v <= hi + CMP_EPS);
    }

    #[test]
    fn different_cell_sizes_change_the_field() {
        // The same pixel sampled at a fine and a coarse cell size should differ.
        let fine = value_noise01(37, 41, 6, 1);
        let coarse = value_noise01(37, 41, 6, 16);
        assert!(!approx(fine, coarse));
    }

    #[test]
    fn frame_animation_changes_the_grain() {
        let a = value_noise01(20, 20, 100, 4);
        let b = value_noise01(20, 20, 101, 4);
        assert!(!approx(a, b));
    }

    #[test]
    fn signed_grain_is_zero_centered_and_bounded() {
        for x in 0..48u32 {
            let g = signed_grain(x, 7, 3, 2);
            assert!((-1.0..1.0).contains(&g));
        }
    }

    #[test]
    fn smoothstep_pins_its_endpoints() {
        assert!(approx(smoothstep01(0.0), 0.0));
        assert!(approx(smoothstep01(1.0), 1.0));
        assert!(approx(smoothstep01(0.5), 0.5));
    }

    #[test]
    fn luma_weights_sum_to_one() {
        assert!(approx(luma709([1.0, 1.0, 1.0]), 1.0));
        assert!(approx(luma709([0.0, 0.0, 0.0]), 0.0));
    }

    #[test]
    fn response_suppresses_highlights_below_shadows() {
        let p = params();
        let shadow = p.response(0.05);
        let mid = p.response(0.5);
        let highlight = p.response(0.95);
        assert!(highlight < mid);
        assert!(mid < shadow);
    }

    #[test]
    fn response_is_monotonic_decreasing() {
        let p = params();
        let mut prev = p.response(0.0);
        for step in 1..=20u32 {
            let l = u32_to_f32(step) / 20.0;
            let cur = p.response(l);
            assert!(cur <= prev + CMP_EPS);
            prev = cur;
        }
    }

    #[test]
    fn zero_intensity_is_identity_additive() {
        let p = FilmGrainParams::new(0.0, 3, 1.0, 2.0, GrainBlend::Additive);
        let color = [0.3, 0.6, 0.9];
        let out = p.apply(color, 12, 34, 5);
        assert!(approx(out[0], color[0]));
        assert!(approx(out[1], color[1]));
        assert!(approx(out[2], color[2]));
    }

    #[test]
    fn zero_intensity_is_identity_soft_light() {
        let p = FilmGrainParams::new(0.0, 3, 1.0, 2.0, GrainBlend::SoftLight);
        let color = [0.2, 0.5, 0.7];
        let out = p.apply(color, 8, 16, 2);
        assert!(approx(out[0], color[0]));
        assert!(approx(out[1], color[1]));
        assert!(approx(out[2], color[2]));
    }

    #[test]
    fn apply_additive_stays_bounded() {
        let p = FilmGrainParams::new(2.0, 2, 3.0, 0.0, GrainBlend::Additive);
        for x in 0..32u32 {
            for y in 0..32u32 {
                let out = p.apply([0.5, 0.5, 0.5], x, y, 9);
                for c in out {
                    assert!((0.0..=1.0).contains(&c));
                }
            }
        }
    }

    #[test]
    fn apply_soft_light_stays_bounded() {
        let p = FilmGrainParams::new(2.0, 2, 3.0, 1.0, GrainBlend::SoftLight);
        for x in 0..32u32 {
            for y in 0..32u32 {
                let out = p.apply([0.1, 0.5, 0.95], x, y, 4);
                for c in out {
                    assert!((0.0..=1.0).contains(&c));
                }
            }
        }
    }

    #[test]
    fn blend_codes_are_stable_and_distinct() {
        assert_eq!(GrainBlend::Additive.code(), 0);
        assert_eq!(GrainBlend::SoftLight.code(), 1);
    }

    #[test]
    fn std430_spans_two_vec4_slots() {
        assert_eq!(FilmGrainParams::STD430_SIZE, 2 * VEC4_STRIDE);
        assert_eq!(FilmGrainParams::STD430_SIZE % VEC4_STRIDE, 0);
    }

    #[test]
    fn std430_round_trips_the_scalar_fields() {
        let p = params();
        let bytes = p.to_std430();
        assert_eq!(bytes.len(), FilmGrainParams::STD430_SIZE);
        assert!(approx(
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            p.intensity
        ));
        assert!(approx(
            f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            p.shadow_boost
        ));
        assert!(approx(
            f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            p.highlight_rolloff
        ));
        assert_eq!(
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            p.cell_size
        );
        assert_eq!(
            u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            p.blend.code()
        );
    }

    #[test]
    fn pack_slice_and_storage_bytes_agree() {
        let a = params();
        let b = FilmGrainParams::new(0.5, 8, 0.0, 1.0, GrainBlend::SoftLight);
        let buffer = FilmGrainParams::pack_slice(&[a, b]);
        assert_eq!(buffer.len(), 2 * FilmGrainParams::STD430_SIZE);
        assert_eq!(&buffer[..FilmGrainParams::STD430_SIZE], &a.to_std430());
        assert_eq!(&buffer[FilmGrainParams::STD430_SIZE..], &b.to_std430());
        assert_eq!(
            FilmGrainParams::gpu_storage_bytes(0),
            FilmGrainParams::STD430_SIZE
        );
        assert_eq!(
            FilmGrainParams::gpu_storage_bytes(2),
            2 * FilmGrainParams::STD430_SIZE
        );
    }
}
