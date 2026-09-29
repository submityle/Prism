//! Motion-vector quantization and mask packing for the compact velocity target.
//!
//! The `GPU` velocity buffer does not store full-precision floats per pixel:
//! motion vectors are quantized to a signed fixed-point pair, and the
//! reactive / transparency / confidence signals are packed into a small number
//! of bytes so the temporal resolve reads one compact texel per pixel. This
//! module is the `CPU` reference for that encoding; the matching `GPU`
//! pack/unpack in the resolve kernel is pending the `GPU` backend.
//!
//! Two representations are provided:
//! - **Velocity** is normalized by a configurable maximum pixel displacement,
//!   clamped to `[-1, 1]`, then quantized to a signed 16-bit pair (`snorm16`).
//!   The maximum bounds the representable motion; anything faster saturates,
//!   which is the correct behavior for a velocity target (extreme motion is
//!   handled by the tile-max dilation, not by more precision here).
//! - **Masks** (reactive, transparency, confidence) are `unorm8` values packed
//!   into a `u32`, leaving one byte for per-pixel flags.

use super::{clamp01, MotionSample, Vec2};

/// Full-scale value of a signed 16-bit fixed-point channel.
const SNORM16_SCALE: f32 = 32767.0;

/// Configuration for velocity quantization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VelocityEncoding {
    /// The pixel displacement that maps to full scale (`+/-1.0` normalized).
    /// Velocities beyond this saturate. Must be positive; construction clamps
    /// it up to a small floor so decoding never divides by zero.
    pub max_velocity_pixels: f32,
}

impl Default for VelocityEncoding {
    fn default() -> Self {
        // A half-1080p-height default; callers scale this to their target.
        Self {
            max_velocity_pixels: 540.0,
        }
    }
}

impl VelocityEncoding {
    /// Smallest allowed full-scale, so normalization never divides by zero.
    const MIN_SCALE: f32 = 1.0;

    /// Builds an encoding, clamping the full-scale up to [`VelocityEncoding::MIN_SCALE`].
    #[must_use]
    pub fn new(max_velocity_pixels: f32) -> Self {
        let clamped = if max_velocity_pixels.is_nan() || max_velocity_pixels < Self::MIN_SCALE {
            Self::MIN_SCALE
        } else {
            max_velocity_pixels
        };
        Self {
            max_velocity_pixels: clamped,
        }
    }

    /// Quantizes a pixel-space velocity to a signed 16-bit pair.
    #[must_use]
    pub fn encode(&self, velocity_pixels: Vec2) -> [i16; 2] {
        [
            encode_snorm16(velocity_pixels.x / self.max_velocity_pixels),
            encode_snorm16(velocity_pixels.y / self.max_velocity_pixels),
        ]
    }

    /// Dequantizes a signed 16-bit pair back to a pixel-space velocity. The
    /// result differs from the input by at most one quantization step per axis.
    #[must_use]
    pub fn decode(&self, encoded: [i16; 2]) -> Vec2 {
        Vec2::new(
            decode_snorm16(encoded[0]) * self.max_velocity_pixels,
            decode_snorm16(encoded[1]) * self.max_velocity_pixels,
        )
    }

    /// The worst-case round-trip error per axis, in pixels: one half
    /// quantization step. Useful for tests and for choosing a full-scale.
    #[must_use]
    pub fn quantization_step_pixels(&self) -> f32 {
        self.max_velocity_pixels / SNORM16_SCALE
    }
}

/// Encodes a normalized value in `[-1, 1]` to `snorm16`, clamping out-of-range
/// inputs (and `NaN`) so saturation is deterministic.
#[must_use]
pub fn encode_snorm16(normalized: f32) -> i16 {
    let clamped = clamp_signed_unit(normalized);
    // Round-to-nearest via a half-step bias, then truncate.
    let scaled = clamped * SNORM16_SCALE;
    let biased = if scaled >= 0.0 {
        scaled + 0.5
    } else {
        scaled - 0.5
    };
    biased as i16
}

/// Decodes a `snorm16` value back to `[-1, 1]`.
#[must_use]
pub fn decode_snorm16(encoded: i16) -> f32 {
    let v = f32::from(encoded) / SNORM16_SCALE;
    clamp_signed_unit(v)
}

/// Quantizes a `[0, 1]` value to `unorm8`, clamping (and resolving `NaN`) first.
#[must_use]
pub fn encode_unorm8(value: f32) -> u8 {
    let scaled = clamp01(value) * 255.0 + 0.5;
    scaled as u8
}

/// Decodes a `unorm8` value back to `[0, 1]`.
#[must_use]
pub fn decode_unorm8(value: u8) -> f32 {
    f32::from(value) / 255.0
}

/// The per-pixel masks packed alongside the velocity vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PackedMasks(pub u32);

impl PackedMasks {
    const REACTIVE_SHIFT: u32 = 0;
    const TRANSPARENCY_SHIFT: u32 = 8;
    const CONFIDENCE_SHIFT: u32 = 16;
    const FLAGS_SHIFT: u32 = 24;

    /// Packs reactive, transparency, and confidence (`[0, 1]` each) into a
    /// `u32`, reserving the top byte for `flags`.
    #[must_use]
    pub fn pack(reactive: f32, transparency: f32, confidence: f32, flags: u8) -> Self {
        let r = u32::from(encode_unorm8(reactive)) << Self::REACTIVE_SHIFT;
        let t = u32::from(encode_unorm8(transparency)) << Self::TRANSPARENCY_SHIFT;
        let c = u32::from(encode_unorm8(confidence)) << Self::CONFIDENCE_SHIFT;
        let f = u32::from(flags) << Self::FLAGS_SHIFT;
        Self(r | t | c | f)
    }

    /// The reactive mask in `[0, 1]`.
    #[must_use]
    pub fn reactive(self) -> f32 {
        decode_unorm8(self.byte(Self::REACTIVE_SHIFT))
    }

    /// The transparency coverage in `[0, 1]`.
    #[must_use]
    pub fn transparency(self) -> f32 {
        decode_unorm8(self.byte(Self::TRANSPARENCY_SHIFT))
    }

    /// The reprojection confidence in `[0, 1]`.
    #[must_use]
    pub fn confidence(self) -> f32 {
        decode_unorm8(self.byte(Self::CONFIDENCE_SHIFT))
    }

    /// The per-pixel flag byte.
    #[must_use]
    pub fn flags(self) -> u8 {
        self.byte(Self::FLAGS_SHIFT)
    }

    fn byte(self, shift: u32) -> u8 {
        ((self.0 >> shift) & 0xFF) as u8
    }
}

/// A fully encoded per-pixel motion texel: a quantized velocity plus the packed
/// masks. This is the compact record the `GPU` velocity target stores.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EncodedMotion {
    /// `snorm16` velocity pair (normalized by the encoding's full-scale).
    pub velocity: [i16; 2],
    /// Packed reactive / transparency / confidence / flags.
    pub masks: PackedMasks,
}

/// Per-pixel flag bits stored in the top byte of [`PackedMasks`].
pub mod flags {
    /// The pixel's history was rejected as disoccluded this frame.
    pub const DISOCCLUDED: u8 = 1 << 0;
    /// The pixel belongs to a transparent/blended surface.
    pub const TRANSPARENT: u8 = 1 << 1;
    /// The pixel was revealed by streaming and has no valid history.
    pub const STREAMING_REVEAL: u8 = 1 << 2;
}

/// Encodes a [`MotionSample`] into its compact texel form.
#[must_use]
pub fn encode_sample(
    sample: MotionSample,
    encoding: VelocityEncoding,
    extra_flags: u8,
) -> EncodedMotion {
    let mut flag_bits = extra_flags;
    if sample.transparency > 0.0 {
        flag_bits |= flags::TRANSPARENT;
    }
    EncodedMotion {
        velocity: encoding.encode(sample.velocity()),
        masks: PackedMasks::pack(
            sample.reactive,
            sample.transparency,
            sample.reprojection_confidence,
            flag_bits,
        ),
    }
}

/// Clamps to `[-1, 1]`, resolving `NaN` to `0.0` deterministically.
fn clamp_signed_unit(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        // `x` is finite here, so the standard clamp cannot propagate `NaN`.
        x.clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn snorm16_endpoints_and_zero() {
        assert_eq!(encode_snorm16(1.0), 32767);
        assert_eq!(encode_snorm16(-1.0), -32767);
        assert_eq!(encode_snorm16(0.0), 0);
        // Saturation and NaN.
        assert_eq!(encode_snorm16(5.0), 32767);
        assert_eq!(encode_snorm16(-5.0), -32767);
        assert_eq!(encode_snorm16(f32::NAN), 0);
    }

    #[test]
    fn snorm16_round_trips_within_one_step() {
        let step = 1.0 / SNORM16_SCALE;
        for i in -20..=20 {
            let v = i as f32 * 0.05;
            let clamped = v.clamp(-1.0, 1.0);
            let back = decode_snorm16(encode_snorm16(v));
            assert!(approx(back, clamped, step), "{v} -> {back}");
        }
    }

    #[test]
    fn velocity_encoding_round_trips_within_step() {
        let enc = VelocityEncoding::new(256.0);
        let step = enc.quantization_step_pixels();
        for &(x, y) in &[(0.0, 0.0), (10.0, -20.0), (255.0, -255.0), (-256.0, 256.0)] {
            let v = Vec2::new(x, y);
            let back = enc.decode(enc.encode(v));
            assert!(approx(back.x, x, step * 1.01), "x {x} -> {}", back.x);
            assert!(approx(back.y, y, step * 1.01), "y {y} -> {}", back.y);
        }
    }

    #[test]
    fn velocity_beyond_full_scale_saturates() {
        let enc = VelocityEncoding::new(100.0);
        let back = enc.decode(enc.encode(Vec2::new(1000.0, -1000.0)));
        assert!(approx(back.x, 100.0, 0.1));
        assert!(approx(back.y, -100.0, 0.1));
    }

    #[test]
    fn encoding_new_clamps_bad_full_scale() {
        assert_eq!(VelocityEncoding::new(-3.0).max_velocity_pixels, 1.0);
        assert_eq!(VelocityEncoding::new(f32::NAN).max_velocity_pixels, 1.0);
        assert_eq!(VelocityEncoding::new(0.0).max_velocity_pixels, 1.0);
    }

    #[test]
    fn unorm8_round_trips() {
        assert_eq!(encode_unorm8(0.0), 0);
        assert_eq!(encode_unorm8(1.0), 255);
        assert_eq!(encode_unorm8(2.0), 255);
        assert_eq!(encode_unorm8(f32::NAN), 0);
        assert!(approx(decode_unorm8(encode_unorm8(0.5)), 0.5, 1.0 / 255.0));
    }

    #[test]
    fn packed_masks_preserve_each_channel() {
        let p = PackedMasks::pack(0.25, 0.5, 0.75, flags::DISOCCLUDED | flags::TRANSPARENT);
        assert!(approx(p.reactive(), 0.25, 1.0 / 255.0));
        assert!(approx(p.transparency(), 0.5, 1.0 / 255.0));
        assert!(approx(p.confidence(), 0.75, 1.0 / 255.0));
        assert_eq!(p.flags(), flags::DISOCCLUDED | flags::TRANSPARENT);
    }

    #[test]
    fn packed_masks_channels_do_not_bleed() {
        // Full reactive, zero elsewhere: only the low byte is set.
        let p = PackedMasks::pack(1.0, 0.0, 0.0, 0);
        assert_eq!(p.0 & 0xFF, 255);
        assert_eq!(p.0 >> 8, 0);
    }

    #[test]
    fn encode_sample_sets_transparent_flag() {
        let sample = MotionSample {
            velocity_pixels: [12.0, -8.0],
            reprojection_confidence: 1.0,
            reactive: 0.0,
            transparency: 0.4,
            surface_id: 3,
        };
        let enc = encode_sample(sample, VelocityEncoding::new(128.0), 0);
        assert_ne!(enc.masks.flags() & flags::TRANSPARENT, 0);
        assert!(approx(enc.masks.transparency(), 0.4, 1.0 / 255.0));
        // Opaque sample keeps the flag clear.
        let opaque = MotionSample {
            transparency: 0.0,
            ..sample
        };
        let enc2 = encode_sample(opaque, VelocityEncoding::default(), 0);
        assert_eq!(enc2.masks.flags() & flags::TRANSPARENT, 0);
    }
}
