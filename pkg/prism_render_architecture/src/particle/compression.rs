//! Particle attribute compression and bandwidth optimization (design §27).
//!
//! Attribute buffers dominate a particle system's per-frame streaming cost, so
//! Ember stores most channels in a compressed form and expands them only inside
//! the shading/simulation kernels. This module owns the `CPU`-verifiable
//! contract for that scheme: it models one [`AttributeEncoding`] per attribute
//! semantic, provides the exact bit-level pack/unpack reference the future
//! `GPU` kernel must agree with, and estimates the before/after streaming
//! bandwidth of a whole [`AttributeLayoutPlan`].
//!
//! Following production `VFX` engines, the encodings are chosen per semantic:
//!
//! * positions/velocities use `fp16` (half precision) or a relative uniform
//!   quantization over a known range,
//! * colours use packed `RGBA8` (`unorm`),
//! * normals/tangents use octahedral (`oct`) encoding into two `snorm16`
//!   channels (purely algebraic, no transcendental calls),
//! * ages, sizes, and other bounded scalars use `fp16`,
//! * scalar `PBR` factors (roughness/metallic) use `unorm8`.
//!
//! Every reference codec here is closed-form and uses at most `sqrt`
//! (via [`super::Vec3::normalize_or_zero`]); no transcendental functions are
//! called, keeping the `CPU` reference bit-reproducible against the `GPU` path.
//! All conversions clamp and guard so a malformed input can never produce
//! `NaN`, an out-of-range integer, or an overflowing multiplication.

use super::attributes::{
    shading_input_attributes, AttributeFormat, AttributeLayoutPlan, AttributeSemantic,
};
use super::Vec3;
use super::{EmberShadingModel, EPS_LEN_SQ};

/// The compressed storage encoding chosen for one attribute buffer (design §27).
///
/// The encoding is orthogonal to the logical [`AttributeFormat`]: the format
/// describes the value the kernels see (for example a three-float vector) while
/// the encoding describes how those components are packed on the device. Because
/// [`AttributeEncoding::QuantizedRelative`] carries `f32` range bounds this enum
/// deliberately derives only [`PartialEq`] (no `Eq`/`Hash`): comparing the
/// bounds is an approximate, not a bitwise-total, operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttributeEncoding {
    /// No compression: each component stays a full 32-bit float.
    F32Raw,
    /// No compression: each component stays a full 32-bit unsigned integer
    /// (identifiers and flags that must survive exactly).
    U32Raw,
    /// Half precision (`fp16`): each component is a 16-bit float.
    Fp16,
    /// Per-component unsigned normalized 8-bit (`unorm8`), values in `0..=1`.
    Unorm8,
    /// Per-component signed normalized 8-bit (`snorm8`), values in `-1..=1`.
    Snorm8,
    /// Four channels packed as unsigned normalized bytes (`RGBA8`, `unorm`).
    UnormRgba8,
    /// Four channels packed as signed normalized bytes (`RGBA8`, `snorm`).
    SnormRgba8,
    /// A unit vector octahedral-encoded (`oct`) into two `snorm16` channels.
    OctNormal16,
    /// Uniform quantization of each component into `bits` bits over the closed
    /// range `min..=max`.
    QuantizedRelative {
        /// Bits per component (clamped to `1..=32` when applied).
        bits: u8,
        /// Inclusive lower bound of the quantization range.
        min: f32,
        /// Inclusive upper bound of the quantization range.
        max: f32,
    },
}

impl AttributeEncoding {
    /// The tightly packed byte size of one encoded element of `format`.
    ///
    /// Sub-byte quantizations report their bit-packed content size (rounded up
    /// to whole bytes), reflecting the best case where adjacent small channels
    /// share device words; the fixed packed codecs (`RGBA8`, `oct`) report their
    /// natural word size.
    #[must_use]
    pub const fn encoded_size_bytes(self, format: AttributeFormat) -> u32 {
        let components = component_count(format);
        match self {
            AttributeEncoding::F32Raw | AttributeEncoding::U32Raw => 4 * components,
            AttributeEncoding::Fp16 => 2 * components,
            AttributeEncoding::Unorm8 | AttributeEncoding::Snorm8 => components,
            AttributeEncoding::UnormRgba8
            | AttributeEncoding::SnormRgba8
            | AttributeEncoding::OctNormal16 => 4,
            AttributeEncoding::QuantizedRelative { bits, .. } => {
                let total_bits = (bits as u32) * components;
                total_bits.div_ceil(8)
            }
        }
    }
}

/// The number of scalar components an [`AttributeFormat`] carries.
#[must_use]
const fn component_count(format: AttributeFormat) -> u32 {
    match format {
        AttributeFormat::F32 | AttributeFormat::U32 => 1,
        AttributeFormat::Vec2 => 2,
        AttributeFormat::Vec3 => 3,
        AttributeFormat::Vec4 => 4,
    }
}

/// Selects the default compression encoding for an attribute `semantic` under a
/// given shading `model` (design §27).
///
/// Motion and lifetime channels use `fp16`; normals/tangents use octahedral
/// `snorm16`; scalar `PBR` factors use `unorm8`; identifiers stay raw. Colour is
/// the one model-dependent choice: a lit model treats colour as an albedo in
/// `0..=1` and packs it as `RGBA8`, while an unlit (additive/energy) model needs
/// the extra dynamic range and keeps colour in `fp16`.
#[must_use]
pub fn recommend_encoding(
    semantic: AttributeSemantic,
    model: EmberShadingModel,
) -> AttributeEncoding {
    match semantic {
        AttributeSemantic::Position
        | AttributeSemantic::Velocity
        | AttributeSemantic::Age
        | AttributeSemantic::Lifetime
        | AttributeSemantic::Size
        | AttributeSemantic::Rotation
        | AttributeSemantic::Scale
        | AttributeSemantic::Emissive => AttributeEncoding::Fp16,
        AttributeSemantic::Color => {
            if model.needs_lighting() {
                AttributeEncoding::UnormRgba8
            } else {
                AttributeEncoding::Fp16
            }
        }
        AttributeSemantic::Normal | AttributeSemantic::Tangent => AttributeEncoding::OctNormal16,
        AttributeSemantic::Roughness | AttributeSemantic::Metallic => AttributeEncoding::Unorm8,
        AttributeSemantic::ShadingParams => AttributeEncoding::UnormRgba8,
        AttributeSemantic::Alive
        | AttributeSemantic::ParticleId
        | AttributeSemantic::RibbonId
        | AttributeSemantic::SortKey
        | AttributeSemantic::MaterialId => AttributeEncoding::U32Raw,
        AttributeSemantic::Custom(_) => AttributeEncoding::F32Raw,
    }
}

// --------------------------------------------------------------------------
// Half precision (`fp16`) bit-level conversion.
// --------------------------------------------------------------------------

/// Rounds `value >> shift` to nearest, ties to even, without losing the bits
/// shifted out. `shift` must be in `1..=31`.
#[must_use]
const fn round_shift_to_nearest_even(value: u32, shift: u32) -> u32 {
    let lsb = 1u32 << shift;
    let half = lsb >> 1;
    let remainder = value & (lsb - 1);
    let truncated = value >> shift;
    if remainder > half || (remainder == half && (truncated & 1) == 1) {
        truncated + 1
    } else {
        truncated
    }
}

/// Converts an `f32` to the bit pattern of its nearest `fp16` (half) value.
///
/// Handles the whole `IEEE`-754 range: signed zero, subnormals (both toward and
/// away from the half subnormal range), overflow to `Inf`, and `Inf`/`NaN`
/// passthrough. Rounding is round-to-nearest, ties to even, matching a
/// conformant `GPU` pack. No transcendental function is used.
#[must_use]
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    // `Inf`/`NaN`: the float exponent field is all ones.
    if exp == 0xff {
        if mant != 0 {
            // Quiet `NaN` with a set mantissa so it never decodes back to `Inf`.
            return sign | 0x7e00;
        }
        return sign | 0x7c00;
    }

    // Re-bias the exponent from the 127-offset float form to the 15-offset half
    // form.
    let half_exp = exp - 127 + 15;

    if half_exp >= 0x1f {
        // Finite but too large for half: saturate to `Inf`.
        return sign | 0x7c00;
    }

    if half_exp <= 0 {
        if half_exp < -10 {
            // Smaller than the smallest half subnormal: flush to signed zero.
            return sign;
        }
        // Half subnormal: restore the implicit leading one, then round-shift the
        // 24-bit significand down into the 10-bit subnormal field.
        let significand = mant | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let half_mant = round_shift_to_nearest_even(significand, shift) as u16;
        return sign | half_mant;
    }

    // Normalized half: round the 23-bit mantissa down to 10 bits. A rounding
    // carry propagates into the exponent through the addition, which also turns
    // a rounded-up maximum-finite value into `Inf` exactly as `IEEE` requires.
    let half_mant = round_shift_to_nearest_even(mant, 13) as u16;
    sign | (((half_exp as u16) << 10) + half_mant)
}

/// Converts an `fp16` (half) bit pattern back to the exactly represented `f32`.
///
/// Inverse of [`f32_to_f16_bits`] on every finite half value; half `Inf`/`NaN`
/// decode to float `Inf`/`NaN`, and half subnormals are renormalized into the
/// float's larger exponent range. No transcendental function is used.
#[must_use]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);

    let out_bits = if exp == 0 {
        if mant == 0 {
            // Signed zero.
            sign
        } else {
            // Subnormal half: shift the mantissa up until the implicit one
            // reaches bit 10, decrementing the exponent for each shift.
            let mut m = mant;
            let mut exponent: i32 = -1;
            loop {
                exponent += 1;
                m <<= 1;
                if (m & 0x0400) != 0 {
                    break;
                }
            }
            m &= 0x03ff;
            // 127 (float bias) - 15 (half bias) = 112.
            let e = (112 - exponent) as u32;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        if mant == 0 {
            // Infinity.
            sign | 0x7f80_0000
        } else {
            // Quiet `NaN`, carrying the payload up into the float mantissa.
            sign | 0x7fc0_0000 | (mant << 13)
        }
    } else {
        // Normalized: re-bias the exponent and widen the mantissa.
        let e = exp + 112;
        sign | (e << 23) | (mant << 13)
    };

    f32::from_bits(out_bits)
}

/// Packs a [`Vec3`] into three `fp16` channels (position/velocity storage).
#[must_use]
pub fn vec3_to_f16(v: Vec3) -> [u16; 3] {
    [
        f32_to_f16_bits(v.x),
        f32_to_f16_bits(v.y),
        f32_to_f16_bits(v.z),
    ]
}

/// Unpacks three `fp16` channels back into a [`Vec3`].
#[must_use]
pub fn f16_to_vec3(bits: [u16; 3]) -> Vec3 {
    Vec3::new(
        f16_bits_to_f32(bits[0]),
        f16_bits_to_f32(bits[1]),
        f16_bits_to_f32(bits[2]),
    )
}

// --------------------------------------------------------------------------
// Normalized integer quantization.
// --------------------------------------------------------------------------

/// Encodes a value in `0..=1` as an 8-bit unsigned normalized (`unorm8`) code.
/// Out-of-range inputs are clamped, so the result is always in `0..=255`.
#[must_use]
pub fn unorm8_encode(value: f32) -> u8 {
    let clamped = value.clamp(0.0, 1.0);
    (clamped * 255.0).round() as u8
}

/// Decodes an 8-bit unsigned normalized (`unorm8`) code back into `0..=1`.
#[must_use]
pub fn unorm8_decode(code: u8) -> f32 {
    f32::from(code) / 255.0
}

/// Encodes a value in `-1..=1` as an 8-bit signed normalized (`snorm8`) code.
/// The code range is symmetric (`-127..=127`); `-128` is never produced.
#[must_use]
pub fn snorm8_encode(value: f32) -> i8 {
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * 127.0).round() as i8
}

/// Decodes an 8-bit signed normalized (`snorm8`) code back into `-1..=1`.
#[must_use]
pub fn snorm8_decode(code: i8) -> f32 {
    (f32::from(code) / 127.0).max(-1.0)
}

/// Encodes a value in `-1..=1` as a 16-bit signed normalized (`snorm16`) code.
#[must_use]
pub fn snorm16_encode(value: f32) -> i16 {
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * 32767.0).round() as i16
}

/// Decodes a 16-bit signed normalized (`snorm16`) code back into `-1..=1`.
#[must_use]
pub fn snorm16_decode(code: i16) -> f32 {
    (f32::from(code) / 32767.0).max(-1.0)
}

/// Packs four channels in `0..=1` as an `RGBA8` (`unorm`) quad.
#[must_use]
pub fn unorm_rgba8_encode(rgba: [f32; 4]) -> [u8; 4] {
    [
        unorm8_encode(rgba[0]),
        unorm8_encode(rgba[1]),
        unorm8_encode(rgba[2]),
        unorm8_encode(rgba[3]),
    ]
}

/// Unpacks an `RGBA8` (`unorm`) quad back into four channels in `0..=1`.
#[must_use]
pub fn unorm_rgba8_decode(code: [u8; 4]) -> [f32; 4] {
    [
        unorm8_decode(code[0]),
        unorm8_decode(code[1]),
        unorm8_decode(code[2]),
        unorm8_decode(code[3]),
    ]
}

/// Packs four channels in `-1..=1` as an `RGBA8` (`snorm`) quad.
#[must_use]
pub fn snorm_rgba8_encode(rgba: [f32; 4]) -> [i8; 4] {
    [
        snorm8_encode(rgba[0]),
        snorm8_encode(rgba[1]),
        snorm8_encode(rgba[2]),
        snorm8_encode(rgba[3]),
    ]
}

/// Unpacks an `RGBA8` (`snorm`) quad back into four channels in `-1..=1`.
#[must_use]
pub fn snorm_rgba8_decode(code: [i8; 4]) -> [f32; 4] {
    [
        snorm8_decode(code[0]),
        snorm8_decode(code[1]),
        snorm8_decode(code[2]),
        snorm8_decode(code[3]),
    ]
}

// --------------------------------------------------------------------------
// Octahedral unit-vector encoding.
// --------------------------------------------------------------------------

/// Returns `+1.0` when `x >= 0`, otherwise `-1.0` (the octahedral fold sign).
#[must_use]
fn nonneg_sign(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Octahedral-encodes a direction into two components in `-1..=1`.
///
/// The input is `L1`-normalized (so any nonzero-length vector is accepted) and
/// folded onto the octahedron; a (numerically) zero vector maps to `(0, 0)`,
/// which decodes to the canonical `+Z` axis. The transform is purely algebraic.
#[must_use]
pub fn oct_encode(direction: Vec3) -> (f32, f32) {
    let l1 = direction.x.abs() + direction.y.abs() + direction.z.abs();
    if l1 <= EPS_LEN_SQ {
        return (0.0, 0.0);
    }
    let inv = 1.0 / l1;
    let px = direction.x * inv;
    let py = direction.y * inv;
    let pz = direction.z * inv;
    if pz >= 0.0 {
        (px, py)
    } else {
        (
            (1.0 - py.abs()) * nonneg_sign(px),
            (1.0 - px.abs()) * nonneg_sign(py),
        )
    }
}

/// Decodes an octahedral pair back into a unit [`Vec3`] (inverse of
/// [`oct_encode`] up to the quantization error of the storage format).
#[must_use]
pub fn oct_decode(encoded: (f32, f32)) -> Vec3 {
    let (mut x, mut y) = encoded;
    let z = 1.0 - x.abs() - y.abs();
    if z < 0.0 {
        let folded_x = (1.0 - y.abs()) * nonneg_sign(x);
        let folded_y = (1.0 - x.abs()) * nonneg_sign(y);
        x = folded_x;
        y = folded_y;
    }
    Vec3::new(x, y, z).normalize_or_zero()
}

/// Octahedral-encodes a unit vector into two `snorm16` channels (4 bytes).
#[must_use]
pub fn oct_encode_snorm16(direction: Vec3) -> [i16; 2] {
    let (ex, ey) = oct_encode(direction);
    [snorm16_encode(ex), snorm16_encode(ey)]
}

/// Decodes two `snorm16` channels back into a unit [`Vec3`].
#[must_use]
pub fn oct_decode_snorm16(code: [i16; 2]) -> Vec3 {
    oct_decode((snorm16_decode(code[0]), snorm16_decode(code[1])))
}

// --------------------------------------------------------------------------
// Relative uniform quantization.
// --------------------------------------------------------------------------

/// The largest code an unsigned quantizer of `bits` bits can emit.
#[must_use]
const fn max_code(bits: u8) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else if bits == 0 {
        0
    } else {
        (1u32 << bits) - 1
    }
}

/// Uniformly quantizes `value` into `bits` bits over the closed range
/// `min..=max`.
///
/// `bits` is clamped to `1..=32`. A degenerate range (`max <= min`) or a
/// zero-width quantizer is a no-op that returns code `0`. The value is clamped
/// into range first, so the code is always in `0..=max_code(bits)` and no
/// overflow can occur.
#[must_use]
pub fn quantize_relative(value: f32, min: f32, max: f32, bits: u8) -> u32 {
    if bits == 0 {
        return 0;
    }
    let bits = bits.min(32);
    let range = max - min;
    if range <= 0.0 {
        return 0;
    }
    let levels = max_code(bits);
    let t = ((value - min) / range).clamp(0.0, 1.0);
    (t * levels as f32).round() as u32
}

/// Reconstructs the representative value of a relative-quantized `code`.
///
/// Inverse of [`quantize_relative`]. `bits` is clamped to `1..=32`; a degenerate
/// quantizer returns `min`. The code is clamped to the valid level range so the
/// result never leaves `min..=max`.
#[must_use]
pub fn dequantize_relative(code: u32, min: f32, max: f32, bits: u8) -> f32 {
    if bits == 0 {
        return min;
    }
    let bits = bits.min(32);
    let levels = max_code(bits);
    if levels == 0 {
        return min;
    }
    let range = max - min;
    let t = code.min(levels) as f32 / levels as f32;
    min + t * range
}

// --------------------------------------------------------------------------
// Bandwidth estimation.
// --------------------------------------------------------------------------

/// The per-frame streaming bandwidth of a layout before and after compression
/// (design §27).
///
/// Byte counts are the content bytes streamed for one frame across every
/// attribute the shading model actually consumes; ping-pong buffers are counted
/// once per live copy. The struct stores integer byte counts only, so it is a
/// total-equality value; the derived compression ratio is computed on demand.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BandwidthEstimate {
    /// The pool capacity the estimate was computed for.
    pub capacity: u32,
    /// Bytes streamed per frame with every attribute stored uncompressed.
    pub uncompressed_bytes_per_frame: u64,
    /// Bytes streamed per frame with the recommended encodings applied.
    pub compressed_bytes_per_frame: u64,
}

impl BandwidthEstimate {
    /// Bytes saved per frame by compression (never negative; saturates at zero
    /// if an encoding were somehow larger than the raw form).
    #[must_use]
    pub const fn saved_bytes_per_frame(&self) -> u64 {
        self.uncompressed_bytes_per_frame
            .saturating_sub(self.compressed_bytes_per_frame)
    }

    /// The compression ratio `uncompressed / compressed`.
    ///
    /// Returns `1.0` for an empty layout (nothing to compress), so callers never
    /// divide by zero.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        if self.compressed_bytes_per_frame == 0 {
            return 1.0;
        }
        self.uncompressed_bytes_per_frame as f32 / self.compressed_bytes_per_frame as f32
    }
}

/// Estimates the per-frame streaming bandwidth of `plan` before and after
/// applying the recommended encodings for `model` (design §27).
///
/// Shading-input attributes (normals, roughness, ...) are only counted when the
/// shading `model` actually consumes them, reusing
/// [`shading_input_attributes`] rather than duplicating that policy. Every other
/// attribute is always counted. The uncompressed baseline uses each format's
/// tightly packed content size so the ratio reflects real payload shrinkage.
#[must_use]
pub fn estimate_bandwidth(
    plan: &AttributeLayoutPlan,
    model: EmberShadingModel,
) -> BandwidthEstimate {
    let enabled_shading = shading_input_attributes(model);
    let mut uncompressed: u64 = 0;
    let mut compressed: u64 = 0;

    for attribute in &plan.attributes {
        if attribute.semantic.is_shading_input() && !enabled_shading.contains(&attribute.semantic) {
            continue;
        }
        let elements = u64::from(plan.capacity) * u64::from(attribute.copies);
        let raw_bytes = u64::from(attribute.format.packed_size()) * elements;
        let encoding = recommend_encoding(attribute.semantic, model);
        let encoded_bytes = u64::from(encoding.encoded_size_bytes(attribute.format)) * elements;
        uncompressed += raw_bytes;
        compressed += encoded_bytes;
    }

    BandwidthEstimate {
        capacity: plan.capacity,
        uncompressed_bytes_per_frame: uncompressed,
        compressed_bytes_per_frame: compressed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::attributes::{AttributeAccess, AttributeUsage};

    /// Epsilon for comparing reconstructed floats that survive a lossless or
    /// near-lossless round trip.
    const EPS: f32 = 1e-6;
    /// Angular/component error bound for `snorm16` octahedral round trips.
    const OCT16_EPS: f32 = 5e-3;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    // ---- fp16 round trip -------------------------------------------------

    fn f16_round_trip(value: f32) -> f32 {
        f16_bits_to_f32(f32_to_f16_bits(value))
    }

    #[test]
    fn fp16_preserves_signed_zero_bitwise() {
        assert_eq!(f16_round_trip(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(f16_round_trip(-0.0).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn fp16_round_trips_exact_representable_values() {
        for &v in &[1.0f32, -1.0, 2.0, -2.0, 0.5, 0.25, -0.75, 65504.0, -65504.0] {
            let r = f16_round_trip(v);
            assert!(close(r, v, EPS), "value {v} round-tripped to {r}");
        }
    }

    #[test]
    fn fp16_round_trips_typical_value_within_relative_tolerance() {
        let v = 1.0f32 / 3.0;
        let r = f16_round_trip(v);
        // Half has ~11 bits of mantissa: relative error below 2^-10.
        assert!(close(r, v, v.abs() * (1.0 / 1024.0)));
    }

    #[test]
    fn fp16_round_trips_subnormal_half_value() {
        // 2^-20 is exactly representable as a half subnormal (a multiple of
        // 2^-24), so it must round-trip exactly.
        let v = 1.0f32 / 1_048_576.0;
        let r = f16_round_trip(v);
        assert!(close(r, v, EPS), "subnormal {v} round-tripped to {r}");
        assert!(r > 0.0);
    }

    #[test]
    fn fp16_overflow_saturates_to_infinity() {
        assert!(f16_round_trip(1.0e30).is_infinite());
        assert!(f16_round_trip(1.0e30).is_sign_positive());
        assert!(f16_round_trip(-1.0e30).is_infinite());
        assert!(f16_round_trip(-1.0e30).is_sign_negative());
    }

    #[test]
    fn fp16_passes_through_infinity_and_nan() {
        assert!(f16_round_trip(f32::INFINITY).is_infinite());
        assert!(f16_round_trip(f32::INFINITY).is_sign_positive());
        assert!(f16_round_trip(f32::NEG_INFINITY).is_infinite());
        assert!(f16_round_trip(f32::NEG_INFINITY).is_sign_negative());
        assert!(f16_round_trip(f32::NAN).is_nan());
    }

    #[test]
    fn fp16_tiny_value_flushes_to_zero() {
        // Far below the smallest half subnormal (~5.96e-8) flushes to +0.
        let r = f16_round_trip(1.0e-12);
        assert_eq!(r.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn vec3_fp16_round_trip() {
        let v = Vec3::new(1.5, -2.25, 0.125);
        let r = f16_to_vec3(vec3_to_f16(v));
        assert!(close(r.x, v.x, EPS));
        assert!(close(r.y, v.y, EPS));
        assert!(close(r.z, v.z, EPS));
    }

    // ---- unorm / snorm boundaries ---------------------------------------

    #[test]
    fn unorm8_boundaries_and_midpoint() {
        assert_eq!(unorm8_encode(0.0), 0);
        assert_eq!(unorm8_encode(1.0), 255);
        // Clamps out-of-range inputs.
        assert_eq!(unorm8_encode(-0.5), 0);
        assert_eq!(unorm8_encode(2.0), 255);
        assert!(close(unorm8_decode(0), 0.0, EPS));
        assert!(close(unorm8_decode(255), 1.0, EPS));
        // Round trip is bounded by one quantization step (1/255).
        for &v in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let r = unorm8_decode(unorm8_encode(v));
            assert!(close(r, v, 1.0 / 255.0));
        }
    }

    #[test]
    fn snorm8_boundaries_and_zero() {
        assert_eq!(snorm8_encode(1.0), 127);
        assert_eq!(snorm8_encode(-1.0), -127);
        assert_eq!(snorm8_encode(0.0), 0);
        // Clamps out-of-range inputs and never emits -128.
        assert_eq!(snorm8_encode(-2.0), -127);
        assert_eq!(snorm8_encode(2.0), 127);
        assert!(close(snorm8_decode(127), 1.0, EPS));
        assert!(close(snorm8_decode(-127), -1.0, EPS));
        assert!(close(snorm8_decode(0), 0.0, EPS));
        // The reserved -128 code clamps to -1.
        assert!(close(snorm8_decode(-128), -1.0, EPS));
    }

    #[test]
    fn rgba8_round_trips_within_step() {
        let c = [0.0f32, 0.5, 1.0, 0.25];
        let r = unorm_rgba8_decode(unorm_rgba8_encode(c));
        for i in 0..4 {
            assert!(close(r[i], c[i], 1.0 / 255.0));
        }
        let s = [-1.0f32, -0.5, 0.5, 1.0];
        let rs = snorm_rgba8_decode(snorm_rgba8_encode(s));
        for i in 0..4 {
            assert!(close(rs[i], s[i], 1.0 / 127.0));
        }
    }

    // ---- octahedral encoding --------------------------------------------

    fn assert_oct_round_trip(dir: Vec3) {
        let n = dir.normalize_or_zero();
        let r = oct_decode_snorm16(oct_encode_snorm16(n));
        // The reconstructed vector stays unit length ...
        assert!(close(r.length(), 1.0, OCT16_EPS), "length {}", r.length());
        // ... and points in almost the same direction (dot ~ 1).
        assert!(close(n.dot(r), 1.0, OCT16_EPS), "dot {}", n.dot(r));
    }

    #[test]
    fn oct_round_trips_axes_and_diagonals() {
        assert_oct_round_trip(Vec3::new(1.0, 0.0, 0.0));
        assert_oct_round_trip(Vec3::new(0.0, 1.0, 0.0));
        assert_oct_round_trip(Vec3::new(0.0, 0.0, 1.0));
        assert_oct_round_trip(Vec3::new(0.0, 0.0, -1.0));
        assert_oct_round_trip(Vec3::new(1.0, 1.0, 1.0));
        assert_oct_round_trip(Vec3::new(-1.0, 2.0, -3.0));
        assert_oct_round_trip(Vec3::new(0.3, -0.7, 0.4));
    }

    #[test]
    fn oct_encode_of_zero_is_origin_and_decodes_to_unit() {
        let (x, y) = oct_encode(Vec3::ZERO);
        assert!(close(x, 0.0, EPS));
        assert!(close(y, 0.0, EPS));
        // The canonical fallback decodes to the +Z axis (a valid unit vector).
        let r = oct_decode((0.0, 0.0));
        assert!(close(r.length(), 1.0, EPS));
        assert!(close(r.z, 1.0, EPS));
    }

    // ---- relative quantization ------------------------------------------

    #[test]
    fn relative_quantization_hits_range_endpoints() {
        let (min, max, bits) = (-10.0f32, 10.0f32, 8u8);
        assert_eq!(quantize_relative(min, min, max, bits), 0);
        assert_eq!(quantize_relative(max, min, max, bits), 255);
        // Out-of-range clamps to the endpoints.
        assert_eq!(quantize_relative(-100.0, min, max, bits), 0);
        assert_eq!(quantize_relative(100.0, min, max, bits), 255);
        assert!(close(dequantize_relative(0, min, max, bits), min, EPS));
        assert!(close(dequantize_relative(255, min, max, bits), max, EPS));
    }

    #[test]
    fn relative_quantization_round_trip_is_step_bounded() {
        let (min, max, bits) = (-5.0f32, 15.0f32, 12u8);
        let levels = ((1u32 << bits) - 1) as f32;
        let step = (max - min) / levels;
        for &v in &[-5.0f32, -1.0, 0.0, 3.25, 7.5, 15.0] {
            let code = quantize_relative(v, min, max, bits);
            let r = dequantize_relative(code, min, max, bits);
            assert!(close(r, v, step), "value {v} -> {r} (step {step})");
        }
    }

    #[test]
    fn relative_quantization_bit_widths_change_resolution() {
        let (min, max) = (0.0f32, 1.0f32);
        assert_eq!(quantize_relative(1.0, min, max, 1), 1);
        assert_eq!(quantize_relative(1.0, min, max, 4), 15);
        assert_eq!(quantize_relative(1.0, min, max, 10), 1023);
        // Bits above 32 are clamped to 32 (no panic, no overflow).
        assert_eq!(quantize_relative(min, min, max, 40), 0);
    }

    #[test]
    fn relative_quantization_degenerate_inputs_are_no_ops() {
        // Zero bits: encode -> 0, decode -> min.
        assert_eq!(quantize_relative(0.5, 0.0, 1.0, 0), 0);
        assert!(close(dequantize_relative(7, 0.0, 1.0, 0), 0.0, EPS));
        // Degenerate range (max <= min): encode -> 0.
        assert_eq!(quantize_relative(0.5, 1.0, 1.0, 8), 0);
        assert_eq!(quantize_relative(0.5, 2.0, 1.0, 8), 0);
    }

    // ---- encoded sizes ---------------------------------------------------

    #[test]
    fn encoded_sizes_are_correct() {
        assert_eq!(
            AttributeEncoding::F32Raw.encoded_size_bytes(AttributeFormat::Vec3),
            12
        );
        assert_eq!(
            AttributeEncoding::Fp16.encoded_size_bytes(AttributeFormat::Vec3),
            6
        );
        assert_eq!(
            AttributeEncoding::Fp16.encoded_size_bytes(AttributeFormat::F32),
            2
        );
        assert_eq!(
            AttributeEncoding::OctNormal16.encoded_size_bytes(AttributeFormat::Vec3),
            4
        );
        assert_eq!(
            AttributeEncoding::UnormRgba8.encoded_size_bytes(AttributeFormat::Vec4),
            4
        );
        assert_eq!(
            AttributeEncoding::Unorm8.encoded_size_bytes(AttributeFormat::F32),
            1
        );
        assert_eq!(
            AttributeEncoding::U32Raw.encoded_size_bytes(AttributeFormat::U32),
            4
        );
        // 3 components x 10 bits = 30 bits -> 4 bytes.
        let q = AttributeEncoding::QuantizedRelative {
            bits: 10,
            min: -1.0,
            max: 1.0,
        };
        assert_eq!(q.encoded_size_bytes(AttributeFormat::Vec3), 4);
        // 1 component x 12 bits -> 2 bytes.
        let q1 = AttributeEncoding::QuantizedRelative {
            bits: 12,
            min: 0.0,
            max: 1.0,
        };
        assert_eq!(q1.encoded_size_bytes(AttributeFormat::F32), 2);
    }

    // ---- encoding selection ---------------------------------------------

    #[test]
    fn recommend_encoding_matches_semantics() {
        let lit = EmberShadingModel::Pbr;
        let unlit = EmberShadingModel::Unlit;
        assert_eq!(
            recommend_encoding(AttributeSemantic::Position, lit),
            AttributeEncoding::Fp16
        );
        assert_eq!(
            recommend_encoding(AttributeSemantic::Normal, lit),
            AttributeEncoding::OctNormal16
        );
        assert_eq!(
            recommend_encoding(AttributeSemantic::Roughness, lit),
            AttributeEncoding::Unorm8
        );
        assert_eq!(
            recommend_encoding(AttributeSemantic::MaterialId, lit),
            AttributeEncoding::U32Raw
        );
        assert_eq!(
            recommend_encoding(AttributeSemantic::Custom(7), lit),
            AttributeEncoding::F32Raw
        );
        // Colour depends on the model: albedo (lit) packs to RGBA8, energy
        // (unlit) keeps fp16 dynamic range.
        assert_eq!(
            recommend_encoding(AttributeSemantic::Color, lit),
            AttributeEncoding::UnormRgba8
        );
        assert_eq!(
            recommend_encoding(AttributeSemantic::Color, unlit),
            AttributeEncoding::Fp16
        );
    }

    // ---- bandwidth estimation -------------------------------------------

    fn usage(s: AttributeSemantic, access: AttributeAccess) -> AttributeUsage {
        let format = s.default_format().unwrap_or(AttributeFormat::F32);
        AttributeUsage::new(s, format, access)
    }

    #[test]
    fn bandwidth_estimate_compresses_and_excludes_disabled_shading() {
        let rw = AttributeAccess::READ.union(AttributeAccess::WRITE);
        let usages = [
            usage(AttributeSemantic::Position, rw),
            usage(AttributeSemantic::Color, AttributeAccess::WRITE),
            usage(AttributeSemantic::Normal, AttributeAccess::READ),
            usage(AttributeSemantic::Roughness, AttributeAccess::READ),
            usage(AttributeSemantic::Tangent, AttributeAccess::READ),
        ];
        let plan = AttributeLayoutPlan::build(1000, &usages);

        let unlit = estimate_bandwidth(&plan, EmberShadingModel::Unlit);
        let pbr = estimate_bandwidth(&plan, EmberShadingModel::Pbr);

        // Compression always shrinks the payload.
        assert!(unlit.compressed_bytes_per_frame < unlit.uncompressed_bytes_per_frame);
        assert!(pbr.compressed_bytes_per_frame < pbr.uncompressed_bytes_per_frame);
        assert!(unlit.compression_ratio() > 1.0);
        assert!(pbr.compression_ratio() > 1.0);

        // Unlit excludes every shading input (normal/roughness/tangent), so it
        // streams strictly fewer bytes than PBR, which enables normal +
        // roughness (but still not the tangent PBR does not request).
        assert!(unlit.uncompressed_bytes_per_frame < pbr.uncompressed_bytes_per_frame);
        assert!(unlit.compressed_bytes_per_frame < pbr.compressed_bytes_per_frame);
        assert_eq!(unlit.capacity, 1000);
        assert!(pbr.saved_bytes_per_frame() > 0);
    }

    #[test]
    fn bandwidth_estimate_of_empty_plan_has_unit_ratio() {
        let plan = AttributeLayoutPlan::build(512, &[]);
        let est = estimate_bandwidth(&plan, EmberShadingModel::Pbr);
        assert_eq!(est.uncompressed_bytes_per_frame, 0);
        assert_eq!(est.compressed_bytes_per_frame, 0);
        assert!(close(est.compression_ratio(), 1.0, EPS));
        assert_eq!(est.saved_bytes_per_frame(), 0);
    }

    #[test]
    fn bandwidth_estimate_counts_pbr_tangent_only_when_present() {
        // A plan that includes a tangent, compared under NPR (which does not
        // consume a tangent) vs a custom model (which requests the full set).
        let usages = [
            usage(AttributeSemantic::Position, AttributeAccess::READ),
            usage(AttributeSemantic::Tangent, AttributeAccess::READ),
        ];
        let plan = AttributeLayoutPlan::build(256, &usages);
        let npr = estimate_bandwidth(&plan, EmberShadingModel::Npr);
        let custom = estimate_bandwidth(&plan, EmberShadingModel::Custom(1));
        // NPR does not read a tangent, so it excludes it; custom includes it.
        assert!(custom.uncompressed_bytes_per_frame > npr.uncompressed_bytes_per_frame);
    }
}
