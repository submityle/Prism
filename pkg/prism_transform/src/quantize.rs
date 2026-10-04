//! §24.2 Pose quantization (`Quantization`) — network / storage / `GPU` bandwidth.
//!
//! A pose that is replicated over the wire, written to a save file, or streamed
//! to the `GPU` rarely needs full `f32` precision. This module provides
//! **deterministic**, **error-bounded** codecs for the three parts of a
//! [`crate::Transform`], mirroring the techniques shipped by production
//! networked engines:
//!
//! - **Rotation** — quaternion *smallest-three* encoding. A unit quaternion has
//!   one component whose absolute value is at least `1/2`; drop it, store the
//!   `2`-bit index of the dropped (largest) component plus the other three
//!   quantized components, and reconstruct the dropped one from the unit-length
//!   constraint on decode. With the default `10` bits per stored component this
//!   is a `32`-bit rotation ([`QuatQuantizer`] / [`QuatQuantized`]).
//! - **Translation** — bounded uniform quantization inside a caller-supplied
//!   axis-aligned box ([`BoundedQuantizer`]), or a layered coarse-cell + in-cell
//!   fixed-point split for large worlds ([`LayeredTranslation`] /
//!   [`LayeredQuantizer`]). Both expose a closed-form worst-case error bound.
//! - **Scale** — a `1`-bit "is this unit scale?" bypass ([`ScaleQuantizer`] /
//!   [`ScaleQuantized`]); only non-unit scales pay for three quantized
//!   components.
//!
//! [`PoseQuantizer`] ties the three together into a single [`QuantizedPose`]
//! codec for a whole [`crate::Transform`].
//!
//! Every codec is pure `no_std` integer/`f32` arithmetic with a fixed rounding
//! rule (round-half-up on a non-negative mapped value), so encode/decode is
//! bit-identical across platforms and the round-trip error never exceeds the
//! advertised bound. Decoding a value that was produced by [`BoundedQuantizer`]
//! on the exact grid is exact; off-grid values are snapped to the nearest grid
//! point, so the maximum absolute error is half of one quantization step.

use prism_math::{Quat, Vec3};

use crate::Transform;

/// `1 / sqrt(2)`, the tight bound on each non-largest component of a unit
/// quaternion and therefore the half-range the smallest-three codec quantizes
/// those components over.
const INV_SQRT2: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// Map a value in `[-range, range]` to an unsigned integer in `[0, max]`
/// (`max == (1 << bits) - 1`) using round-half-up on the non-negative mapped
/// coordinate, so the rule is identical on every platform.
#[inline]
fn quantize_signed(value: f32, range: f32, bits: u32) -> u32 {
    let max = ((1u64 << bits) - 1) as f32;
    // Normalize to [0, 1], clamp so out-of-range inputs saturate rather than
    // wrap, then round half up.
    let normalized = ((value / range) * 0.5 + 0.5).clamp(0.0, 1.0);
    let scaled = normalized * max + 0.5;
    // `scaled` is in [0, max + 0.5]; the floor lands in [0, max].
    let code = scaled as u32;
    let max_u = ((1u64 << bits) - 1) as u32;
    if code > max_u { max_u } else { code }
}

/// Inverse of [`quantize_signed`]: map an integer code back to the centre of its
/// bucket in `[-range, range]`.
#[inline]
fn dequantize_signed(code: u32, range: f32, bits: u32) -> f32 {
    let max = ((1u64 << bits) - 1) as f32;
    ((code as f32 / max) * 2.0 - 1.0) * range
}

/// Map a value in `[min, max]` to an unsigned integer in `[0, (1 << bits) - 1]`
/// with the same round-half-up rule as [`quantize_signed`].
#[inline]
fn quantize_unsigned(value: f32, min: f32, max: f32, bits: u32) -> u32 {
    let span = max - min;
    let levels = ((1u64 << bits) - 1) as f32;
    if span <= 0.0 {
        return 0;
    }
    let normalized = ((value - min) / span).clamp(0.0, 1.0);
    let scaled = normalized * levels + 0.5;
    let code = scaled as u32;
    let max_u = ((1u64 << bits) - 1) as u32;
    if code > max_u { max_u } else { code }
}

/// Inverse of [`quantize_unsigned`].
#[inline]
fn dequantize_unsigned(code: u32, min: f32, max: f32, bits: u32) -> f32 {
    let span = max - min;
    let levels = ((1u64 << bits) - 1) as f32;
    if levels <= 0.0 {
        return min;
    }
    min + (code as f32 / levels) * span
}

// ---------------------------------------------------------------------------
// Rotation: smallest-three quaternion codec
// ---------------------------------------------------------------------------

/// A quaternion compressed with the smallest-three scheme.
///
/// `largest` is the index (`0..=3` for `x`, `y`, `z`, `w`) of the component
/// that was dropped; `a`, `b`, `c` are the quantized remaining components in
/// ascending index order. The dropped component is always reconstructed with a
/// non-negative sign because the codec canonicalizes the quaternion (a
/// quaternion and its negation represent the same rotation).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QuatQuantized {
    /// Index (`0..=3`) of the largest-magnitude, dropped component.
    pub largest: u8,
    /// First kept component (lowest surviving index), quantized.
    pub a: u16,
    /// Second kept component, quantized.
    pub b: u16,
    /// Third kept component (highest surviving index), quantized.
    pub c: u16,
}

/// Smallest-three quaternion codec with a configurable per-component bit width.
///
/// The default [`QuatQuantizer::DEFAULT`] uses `10` bits per stored component,
/// which together with the `2`-bit index packs into a single `32`-bit word via
/// [`QuatQuantized::to_bits`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QuatQuantizer {
    bits: u32,
}

impl Default for QuatQuantizer {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl QuatQuantizer {
    /// The `10`-bits-per-component configuration (`32`-bit packed rotation).
    pub const DEFAULT: Self = Self { bits: 10 };

    /// Build a codec with `bits` bits per stored component.
    ///
    /// `bits` is clamped to `2..=16` so each component fits a [`u16`] and the
    /// quantization grid is never degenerate.
    #[inline]
    pub const fn new(bits: u32) -> Self {
        let bits = if bits < 2 {
            2
        } else if bits > 16 {
            16
        } else {
            bits
        };
        Self { bits }
    }

    /// Bits used per stored component.
    #[inline]
    pub const fn bits(self) -> u32 {
        self.bits
    }

    /// Encode a (not necessarily normalized) quaternion.
    ///
    /// The input is normalized and sign-canonicalized (so the dropped component
    /// is non-negative) before the three smallest components are quantized over
    /// `[-1/sqrt(2), 1/sqrt(2)]`.
    pub fn encode(self, q: Quat) -> QuatQuantized {
        let q = q.normalize();
        let comps = [q.x, q.y, q.z, q.w];

        // Find the largest-magnitude component.
        let mut largest = 0usize;
        let mut largest_abs = abs_f32(comps[0]);
        let mut i = 1usize;
        while i < 4 {
            let a = abs_f32(comps[i]);
            if a > largest_abs {
                largest_abs = a;
                largest = i;
            }
            i += 1;
        }

        // Canonicalize sign so the dropped component is non-negative; negating a
        // quaternion leaves the rotation unchanged.
        let sign = if comps[largest] < 0.0 { -1.0 } else { 1.0 };

        let mut kept = [0.0f32; 3];
        let mut k = 0usize;
        let mut j = 0usize;
        while j < 4 {
            if j != largest {
                kept[k] = comps[j] * sign;
                k += 1;
            }
            j += 1;
        }

        QuatQuantized {
            largest: largest as u8,
            a: quantize_signed(kept[0], INV_SQRT2, self.bits) as u16,
            b: quantize_signed(kept[1], INV_SQRT2, self.bits) as u16,
            c: quantize_signed(kept[2], INV_SQRT2, self.bits) as u16,
        }
    }

    /// Decode back to a unit quaternion.
    ///
    /// The dropped component is reconstructed from the unit-length constraint
    /// `w^2 + x^2 + y^2 + z^2 = 1` with a non-negative sign, matching the
    /// canonicalization performed by [`QuatQuantizer::encode`].
    pub fn decode(self, q: QuatQuantized) -> Quat {
        let a = dequantize_signed(u32::from(q.a), INV_SQRT2, self.bits);
        let b = dequantize_signed(u32::from(q.b), INV_SQRT2, self.bits);
        let c = dequantize_signed(u32::from(q.c), INV_SQRT2, self.bits);

        let sum_sq = a * a + b * b + c * c;
        let dropped = sqrt_f32((1.0 - sum_sq).max(0.0));

        let largest = q.largest as usize & 0b11;
        let mut comps = [0.0f32; 4];
        let mut k = 0usize;
        let kept = [a, b, c];
        let mut j = 0usize;
        while j < 4 {
            if j == largest {
                comps[j] = dropped;
            } else {
                comps[j] = kept[k];
                k += 1;
            }
            j += 1;
        }

        Quat::from_xyzw(comps[0], comps[1], comps[2], comps[3]).normalize()
    }

    /// Worst-case absolute error of any single stored component after a
    /// round-trip: half of one quantization step over `[-1/sqrt(2),
    /// 1/sqrt(2)]`.
    #[inline]
    pub fn max_component_error(self) -> f32 {
        let levels = ((1u64 << self.bits) - 1) as f32;
        INV_SQRT2 / levels
    }
}

impl QuatQuantized {
    /// Pack into a single `32`-bit word, valid only for the default `10`-bit
    /// configuration (`2`-bit index + `3 * 10`-bit components).
    ///
    /// Layout (low to high): `a` in bits `0..10`, `b` in bits `10..20`, `c` in
    /// bits `20..30`, `largest` in bits `30..32`.
    #[inline]
    pub const fn to_bits(self) -> u32 {
        (self.a as u32 & 0x3ff)
            | ((self.b as u32 & 0x3ff) << 10)
            | ((self.c as u32 & 0x3ff) << 20)
            | (((self.largest as u32) & 0x3) << 30)
    }

    /// Unpack a word produced by [`QuatQuantized::to_bits`].
    #[inline]
    pub const fn from_bits(word: u32) -> Self {
        Self {
            a: (word & 0x3ff) as u16,
            b: ((word >> 10) & 0x3ff) as u16,
            c: ((word >> 20) & 0x3ff) as u16,
            largest: ((word >> 30) & 0x3) as u8,
        }
    }
}

// ---------------------------------------------------------------------------
// Translation: bounded uniform quantizer
// ---------------------------------------------------------------------------

/// A translation quantized by a [`BoundedQuantizer`]: three integer codes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QuantizedVec3 {
    /// Per-axis integer codes (`x`, `y`, `z`).
    pub codes: [u32; 3],
}

/// Uniform quantizer over an axis-aligned box `[min, max]` with a configurable
/// per-axis bit width.
///
/// A value is snapped to the nearest of `2^bits` evenly spaced grid points per
/// axis, so the worst-case absolute error per axis is
/// [`BoundedQuantizer::max_abs_error`] (half of one step).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BoundedQuantizer {
    min: Vec3,
    max: Vec3,
    bits: u32,
}

impl BoundedQuantizer {
    /// Build a quantizer over `[min, max]` with `bits` bits per axis.
    ///
    /// `bits` is clamped to `1..=32`.
    #[inline]
    pub fn new(min: Vec3, max: Vec3, bits: u32) -> Self {
        let bits = bits.clamp(1, 32);
        Self { min, max, bits }
    }

    /// The lower corner of the quantization box.
    #[inline]
    pub fn min(self) -> Vec3 {
        self.min
    }

    /// The upper corner of the quantization box.
    #[inline]
    pub fn max(self) -> Vec3 {
        self.max
    }

    /// Bits used per axis.
    #[inline]
    pub fn bits(self) -> u32 {
        self.bits
    }

    /// Encode a translation, saturating components outside `[min, max]`.
    #[inline]
    pub fn encode(self, v: Vec3) -> QuantizedVec3 {
        QuantizedVec3 {
            codes: [
                quantize_unsigned(v.x, self.min.x, self.max.x, self.bits),
                quantize_unsigned(v.y, self.min.y, self.max.y, self.bits),
                quantize_unsigned(v.z, self.min.z, self.max.z, self.bits),
            ],
        }
    }

    /// Decode back to the centre of each axis bucket.
    #[inline]
    pub fn decode(self, q: QuantizedVec3) -> Vec3 {
        Vec3::new(
            dequantize_unsigned(q.codes[0], self.min.x, self.max.x, self.bits),
            dequantize_unsigned(q.codes[1], self.min.y, self.max.y, self.bits),
            dequantize_unsigned(q.codes[2], self.min.z, self.max.z, self.bits),
        )
    }

    /// Worst-case absolute per-axis error after a round-trip (half of one
    /// step).
    #[inline]
    pub fn max_abs_error(self) -> Vec3 {
        let levels = ((1u64 << self.bits) - 1) as f32;
        let span = self.max - self.min;
        Vec3::new(
            0.5 * span.x / levels,
            0.5 * span.y / levels,
            0.5 * span.z / levels,
        )
    }
}

// ---------------------------------------------------------------------------
// Translation: layered coarse-cell + in-cell quantizer (big world)
// ---------------------------------------------------------------------------

/// A translation split into an integer world cell plus an in-cell quantized
/// offset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LayeredTranslation {
    /// Integer cell coordinates (`x`, `y`, `z`) on the coarse grid.
    pub cell: [i32; 3],
    /// Quantized offset inside the cell, over `[0, cell_size)` per axis.
    pub offset: [u32; 3],
}

/// Layered translation quantizer: a coarse `cell_size` grid carries global
/// position cheaply, and `bits` fine bits resolve the position inside each
/// cell. Precision is uniform everywhere (unlike a single global quantizer,
/// whose absolute step grows with the world extent).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LayeredQuantizer {
    cell_size: f32,
    bits: u32,
}

impl LayeredQuantizer {
    /// Build a layered quantizer with the given `cell_size` (world units per
    /// cell edge) and `bits` fine bits per axis.
    ///
    /// `cell_size` is forced positive and `bits` is clamped to `1..=32`.
    #[inline]
    pub fn new(cell_size: f32, bits: u32) -> Self {
        let cell_size = if cell_size > 0.0 { cell_size } else { 1.0 };
        let bits = bits.clamp(1, 32);
        Self { cell_size, bits }
    }

    /// World units per cell edge.
    #[inline]
    pub fn cell_size(self) -> f32 {
        self.cell_size
    }

    /// Fine bits per axis.
    #[inline]
    pub fn bits(self) -> u32 {
        self.bits
    }

    /// Encode a world-space translation.
    #[inline]
    pub fn encode(self, v: Vec3) -> LayeredTranslation {
        let (cx, ox) = self.encode_axis(v.x);
        let (cy, oy) = self.encode_axis(v.y);
        let (cz, oz) = self.encode_axis(v.z);
        LayeredTranslation {
            cell: [cx, cy, cz],
            offset: [ox, oy, oz],
        }
    }

    /// Decode back to a world-space translation.
    #[inline]
    pub fn decode(self, q: LayeredTranslation) -> Vec3 {
        Vec3::new(
            self.decode_axis(q.cell[0], q.offset[0]),
            self.decode_axis(q.cell[1], q.offset[1]),
            self.decode_axis(q.cell[2], q.offset[2]),
        )
    }

    /// Worst-case absolute per-axis error after a round-trip (half of one fine
    /// step).
    #[inline]
    pub fn max_abs_error(self) -> f32 {
        let levels = ((1u64 << self.bits) - 1) as f32;
        0.5 * self.cell_size / levels
    }

    #[inline]
    fn encode_axis(self, value: f32) -> (i32, u32) {
        // Floor division into cells, then quantize the in-cell remainder over
        // `[0, cell_size]`.
        let cell_f = floor_f32(value / self.cell_size);
        let cell = cell_f as i32;
        let remainder = value - cell_f * self.cell_size;
        let code = quantize_unsigned(remainder, 0.0, self.cell_size, self.bits);
        (cell, code)
    }

    #[inline]
    fn decode_axis(self, cell: i32, code: u32) -> f32 {
        let remainder = dequantize_unsigned(code, 0.0, self.cell_size, self.bits);
        cell as f32 * self.cell_size + remainder
    }
}

// ---------------------------------------------------------------------------
// Scale: 1-bit unit-scale bypass + bounded fallback
// ---------------------------------------------------------------------------

/// A quantized scale: either flagged as unit scale (no payload) or three
/// quantized components.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ScaleQuantized {
    /// `true` when the scale was within tolerance of `(1, 1, 1)` and no
    /// component payload is stored.
    pub is_unit: bool,
    /// Per-axis codes, meaningful only when `is_unit` is `false`.
    pub codes: [u32; 3],
}

/// Scale codec with a `1`-bit "is this unit scale?" fast path. Only non-unit
/// scales pay for three quantized components, matching the design-doc
/// observation that most entities keep unit scale.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ScaleQuantizer {
    min: f32,
    max: f32,
    bits: u32,
    tolerance: f32,
}

impl ScaleQuantizer {
    /// Build a scale codec over the per-component range `[min, max]` with `bits`
    /// bits per axis and a `tolerance` for the unit-scale test.
    ///
    /// `bits` is clamped to `1..=32`.
    #[inline]
    pub fn new(min: f32, max: f32, bits: u32, tolerance: f32) -> Self {
        let bits = bits.clamp(1, 32);
        Self {
            min,
            max,
            bits,
            tolerance: abs_f32(tolerance),
        }
    }

    /// Encode a scale. Components within `tolerance` of `1` on every axis take
    /// the unit-scale fast path.
    #[inline]
    pub fn encode(self, scale: Vec3) -> ScaleQuantized {
        if abs_f32(scale.x - 1.0) <= self.tolerance
            && abs_f32(scale.y - 1.0) <= self.tolerance
            && abs_f32(scale.z - 1.0) <= self.tolerance
        {
            return ScaleQuantized {
                is_unit: true,
                codes: [0; 3],
            };
        }
        ScaleQuantized {
            is_unit: false,
            codes: [
                quantize_unsigned(scale.x, self.min, self.max, self.bits),
                quantize_unsigned(scale.y, self.min, self.max, self.bits),
                quantize_unsigned(scale.z, self.min, self.max, self.bits),
            ],
        }
    }

    /// Decode a scale, returning exactly `(1, 1, 1)` for the unit fast path.
    #[inline]
    pub fn decode(self, q: ScaleQuantized) -> Vec3 {
        if q.is_unit {
            return Vec3::ONE;
        }
        Vec3::new(
            dequantize_unsigned(q.codes[0], self.min, self.max, self.bits),
            dequantize_unsigned(q.codes[1], self.min, self.max, self.bits),
            dequantize_unsigned(q.codes[2], self.min, self.max, self.bits),
        )
    }

    /// Worst-case absolute per-axis error for a non-unit scale (half of one
    /// step). Unit-scale values decode exactly.
    #[inline]
    pub fn max_abs_error(self) -> f32 {
        let levels = ((1u64 << self.bits) - 1) as f32;
        0.5 * (self.max - self.min) / levels
    }
}

// ---------------------------------------------------------------------------
// Whole-pose codec
// ---------------------------------------------------------------------------

/// A fully quantized [`crate::Transform`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QuantizedPose {
    /// Quantized translation.
    pub translation: QuantizedVec3,
    /// Quantized rotation.
    pub rotation: QuatQuantized,
    /// Quantized scale.
    pub scale: ScaleQuantized,
}

/// Combines a translation, rotation, and scale codec into a single
/// [`crate::Transform`] codec.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PoseQuantizer {
    /// Translation codec.
    pub translation: BoundedQuantizer,
    /// Rotation codec.
    pub rotation: QuatQuantizer,
    /// Scale codec.
    pub scale: ScaleQuantizer,
}

impl PoseQuantizer {
    /// Build a pose codec from its three component codecs.
    #[inline]
    pub fn new(
        translation: BoundedQuantizer,
        rotation: QuatQuantizer,
        scale: ScaleQuantizer,
    ) -> Self {
        Self {
            translation,
            rotation,
            scale,
        }
    }

    /// Encode a local transform.
    #[inline]
    pub fn encode(self, t: &Transform) -> QuantizedPose {
        QuantizedPose {
            translation: self.translation.encode(t.translation),
            rotation: self.rotation.encode(t.rotation),
            scale: self.scale.encode(t.scale),
        }
    }

    /// Decode back to a local transform.
    #[inline]
    pub fn decode(self, q: &QuantizedPose) -> Transform {
        Transform {
            translation: self.translation.decode(q.translation),
            rotation: self.rotation.decode(q.rotation),
            scale: self.scale.decode(q.scale),
        }
    }

    /// Worst-case absolute translation error per axis after a round-trip.
    #[inline]
    pub fn max_translation_error(self) -> Vec3 {
        self.translation.max_abs_error()
    }

    /// Worst-case absolute scale error per axis for a non-unit scale.
    #[inline]
    pub fn max_scale_error(self) -> f32 {
        self.scale.max_abs_error()
    }

    /// Worst-case absolute error of any single stored quaternion component.
    #[inline]
    pub fn max_rotation_component_error(self) -> f32 {
        self.rotation.max_component_error()
    }
}

// ---------------------------------------------------------------------------
// Small `no_std` float helpers (avoid pulling in `std`)
// ---------------------------------------------------------------------------

#[inline]
fn abs_f32(x: f32) -> f32 {
    libm::fabsf(x)
}

#[inline]
fn sqrt_f32(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[inline]
fn floor_f32(x: f32) -> f32 {
    libm::floorf(x)
}
