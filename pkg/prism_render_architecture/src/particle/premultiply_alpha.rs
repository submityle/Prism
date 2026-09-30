//! Premultiplied-alpha conversion and the `Porter-Duff` compositing algebra
//! (design §5, §12).
//!
//! Blending is where a particle system finally decides *how* a fragment's colour
//! is written over what is already in the target. The classic reference is the
//! `Porter-Duff` operator family: twelve ways two coverage-weighted colours can
//! be combined, each expressed as `result = src * Fa + dst * Fb` for a pair of
//! per-operator blend factors that depend only on the two alphas. This module
//! owns the `CPU`-verifiable *maths* of that algebra plus the premultiplied /
//! straight (direct) alpha conversion the operators are defined on.
//!
//! # Distinction from the other transparency modules
//!
//! This module is deliberately *only* the conversion and the operator algebra:
//!
//! * [`super::oit`] owns order-independent-transparency *weighting* and the
//!   weighted accumulation / revealage buffers; it does not define the twelve
//!   operators and this module does not re-use its accumulator types.
//! * [`super::soft_particle`] owns the *depth fade* that softens a particle
//!   where it intersects opaque geometry; it feeds an alpha into this algebra
//!   but shares none of its types.
//!
//! Everything here is a small, closed arithmetic expression: no lookup tables,
//! no iteration count, nothing device-specific. The [`RgbaColor::to_std430`]
//! byte layout is the only `GPU`-facing surface and it is pure integer /
//! little-endian arithmetic.
//!
//! # Premultiplied vs straight alpha
//!
//! A [`Rgba`] value carries *straight* (un-premultiplied) colour: its `rgb` is
//! the surface colour independent of coverage, and `a` is the coverage. A
//! [`PremulRgba`] value carries colour already scaled by coverage
//! (`rgb *= a`). The `Porter-Duff` operators are defined on premultiplied
//! colour because the linear combination `src * Fa + dst * Fb` is only correct
//! once coverage is folded into the colour. The two wrapper types keep the
//! convention in the type system so a straight colour can never be composited
//! by mistake.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Alphas at or below this magnitude are treated as fully transparent when
/// dividing colour back out in [`premul_to_straight`], so the reciprocal can
/// never explode. This is a runtime guard, distinct from the test-only
/// comparison epsilon.
const ALPHA_FLOOR: f32 = 1e-6;

/// A *straight* (un-premultiplied / direct) `RGBA` colour: `rgb` is the surface
/// colour and `a` is coverage, the two kept independent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgba {
    /// Red channel, surface colour independent of coverage.
    pub r: f32,
    /// Green channel, surface colour independent of coverage.
    pub g: f32,
    /// Blue channel, surface colour independent of coverage.
    pub b: f32,
    /// Alpha / coverage in `0..=1`.
    pub a: f32,
}

impl Rgba {
    /// Builds a straight `RGBA` colour from its four channels.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

/// A *premultiplied* `RGBA` colour: `rgb` has already been scaled by coverage
/// (`rgb *= a`). The `Porter-Duff` operators in this module are defined on this
/// representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PremulRgba {
    /// Red channel scaled by coverage.
    pub r: f32,
    /// Green channel scaled by coverage.
    pub g: f32,
    /// Blue channel scaled by coverage.
    pub b: f32,
    /// Alpha / coverage in `0..=1`.
    pub a: f32,
}

impl PremulRgba {
    /// Number of scalar channels, so the `std430` slot size is derived rather
    /// than hard-coded.
    const FIELD_COUNT: usize = 4;

    /// Byte size of the single `vec4<f32>` `std430` slot a premultiplied colour
    /// occupies (four little-endian `f32` lanes).
    pub const STD430_SIZE: usize = Self::FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

    /// Builds a premultiplied `RGBA` colour from its four channels.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Fully transparent premultiplied colour (all channels zero).
    pub const TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);

    /// Forms the per-channel linear combination `self * fa + other * fb`, the
    /// shape every `Porter-Duff` operator reduces to. Named `combine` rather
    /// than an arithmetic verb because it fuses a scale and a sum across all
    /// four lanes at once.
    #[must_use]
    fn combine(self, other: Self, fa: f32, fb: f32) -> Self {
        Self {
            r: self.r * fa + other.r * fb,
            g: self.g * fa + other.g * fb,
            b: self.b * fa + other.b * fb,
            a: self.a * fa + other.a * fb,
        }
    }

    /// Forms the per-channel sum `self + other`, each lane clamped to `1.0`.
    /// This backs the additive `plus` / `add` operators.
    #[must_use]
    fn saturating_sum(self, other: Self) -> Self {
        Self {
            r: f32::clamp(self.r + other.r, 0.0, 1.0),
            g: f32::clamp(self.g + other.g, 0.0, 1.0),
            b: f32::clamp(self.b + other.b, 0.0, 1.0),
            a: f32::clamp(self.a + other.a, 0.0, 1.0),
        }
    }

    /// Encodes the premultiplied colour as one `std430` `vec4<f32>` slot:
    /// `[r, g, b, a]` as consecutive little-endian `f32` lanes, matching the
    /// layout a `GPU` kernel binds.
    #[must_use]
    pub fn to_std430(self) -> [u8; Self::STD430_SIZE] {
        let mut bytes = [0u8; Self::STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.r.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.g.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.b.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.a.to_le_bytes());
        bytes
    }

    /// Total `std430` storage-buffer byte size for `count` premultiplied
    /// colours, following the shared clamp-to-one-element rule.
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(Self::STD430_SIZE, count)
    }
}

/// Byte size of a single premultiplied colour's `std430` `vec4<f32>` slot.
pub const PREMUL_STD430_SIZE: usize = PremulRgba::STD430_SIZE;

/// Converts a straight colour to premultiplied form by folding coverage into
/// the colour (`rgb *= a`). The alpha lane is preserved.
#[must_use]
pub fn straight_to_premul(color: Rgba) -> PremulRgba {
    PremulRgba {
        r: color.r * color.a,
        g: color.g * color.a,
        b: color.b * color.a,
        a: color.a,
    }
}

/// Converts a premultiplied colour back to straight form by dividing coverage
/// out (`rgb /= a`). When the alpha is at or below [`ALPHA_FLOOR`] the colour is
/// fully transparent and the straight `rgb` is undefined, so a zeroed colour is
/// returned instead of dividing by (near-)zero.
#[must_use]
pub fn premul_to_straight(color: PremulRgba) -> Rgba {
    if color.a > ALPHA_FLOOR {
        let inv = 1.0 / color.a;
        Rgba {
            r: color.r * inv,
            g: color.g * inv,
            b: color.b * inv,
            a: color.a,
        }
    } else {
        Rgba::new(0.0, 0.0, 0.0, 0.0)
    }
}

/// The `Porter-Duff` compositing operators, plus the two additive blends.
///
/// Each `Porter-Duff` operator is a pair of blend factors `(Fa, Fb)` that depend
/// only on the source and destination alphas; [`composite`] applies
/// `result = src * Fa + dst * Fb`. `Plus` and `Add` are the additive
/// (`lighter`) blend, a per-channel clamped sum that is not part of the twelve
/// canonical operators but is ubiquitous for glows and sparks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendOp {
    /// `(0, 0)` — both inputs discarded, result fully transparent.
    Clear,
    /// `(1, 0)` — source only.
    Src,
    /// `(0, 1)` — destination only.
    Dst,
    /// `(1, 1 - αs)` — source drawn over destination (the common default).
    SrcOver,
    /// `(1 - αd, 1)` — destination drawn over source.
    DstOver,
    /// `(αd, 0)` — source shown only where the destination is present.
    SrcIn,
    /// `(0, αs)` — destination shown only where the source is present.
    DstIn,
    /// `(1 - αd, 0)` — source shown only where the destination is absent.
    SrcOut,
    /// `(0, 1 - αs)` — destination shown only where the source is absent.
    DstOut,
    /// `(αd, 1 - αs)` — source atop destination (source clipped to destination).
    SrcAtop,
    /// `(1 - αd, αs)` — destination atop source (destination clipped to source).
    DstAtop,
    /// `(1 - αd, 1 - αs)` — the non-overlapping parts of each.
    Xor,
    /// Additive `lighter` blend: per-channel sum clamped to `1.0`.
    Plus,
    /// Alias of [`BlendOp::Plus`]: the additive blend under its `add` name.
    Add,
}

impl BlendOp {
    /// Returns the `(Fa, Fb)` `Porter-Duff` blend factors for the two alphas, or
    /// `None` for the additive operators which are not a linear combination of
    /// two factors.
    #[must_use]
    fn porter_duff_factors(self, alpha_src: f32, alpha_dst: f32) -> Option<(f32, f32)> {
        let factors = match self {
            Self::Clear => (0.0, 0.0),
            Self::Src => (1.0, 0.0),
            Self::Dst => (0.0, 1.0),
            Self::SrcOver => (1.0, 1.0 - alpha_src),
            Self::DstOver => (1.0 - alpha_dst, 1.0),
            Self::SrcIn => (alpha_dst, 0.0),
            Self::DstIn => (0.0, alpha_src),
            Self::SrcOut => (1.0 - alpha_dst, 0.0),
            Self::DstOut => (0.0, 1.0 - alpha_src),
            Self::SrcAtop => (alpha_dst, 1.0 - alpha_src),
            Self::DstAtop => (1.0 - alpha_dst, alpha_src),
            Self::Xor => (1.0 - alpha_dst, 1.0 - alpha_src),
            Self::Plus | Self::Add => return None,
        };
        Some(factors)
    }
}

/// Composites premultiplied `src` over / with premultiplied `dst` under the
/// given operator, returning the premultiplied result.
///
/// For every `Porter-Duff` operator this evaluates `result = src * Fa + dst * Fb`
/// with the operator's alpha-dependent factors. `Plus` / `Add` instead return
/// the per-channel sum clamped to `1.0`.
#[must_use]
pub fn composite(op: BlendOp, src: PremulRgba, dst: PremulRgba) -> PremulRgba {
    match op.porter_duff_factors(src.a, dst.a) {
        Some((fa, fb)) => src.combine(dst, fa, fb),
        None => src.saturating_sum(dst),
    }
}

/// Composites each `src[i]` over the matching `dst[i]` with [`BlendOp::SrcOver`],
/// the workhorse particle blend. The result length is the shorter of the two
/// inputs so a length mismatch can never index out of bounds.
#[must_use]
pub fn composite_over_batch(src: &[PremulRgba], dst: &[PremulRgba]) -> Vec<PremulRgba> {
    src.iter()
        .zip(dst.iter())
        .map(|(&s, &d)| composite(BlendOp::SrcOver, s, d))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn premul_close(a: PremulRgba, b: PremulRgba) -> bool {
        approx(a.r, b.r) && approx(a.g, b.g) && approx(a.b, b.b) && approx(a.a, b.a)
    }

    fn straight_close(a: Rgba, b: Rgba) -> bool {
        approx(a.r, b.r) && approx(a.g, b.g) && approx(a.b, b.b) && approx(a.a, b.a)
    }

    #[test]
    fn straight_to_premul_scales_rgb_by_alpha() {
        let p = straight_to_premul(Rgba::new(1.0, 0.5, 0.25, 0.5));
        assert!(premul_close(p, PremulRgba::new(0.5, 0.25, 0.125, 0.5)));
    }

    #[test]
    fn premul_to_straight_divides_rgb_by_alpha() {
        let s = premul_to_straight(PremulRgba::new(0.5, 0.25, 0.125, 0.5));
        assert!(straight_close(s, Rgba::new(1.0, 0.5, 0.25, 0.5)));
    }

    #[test]
    fn straight_premul_roundtrip_positive_alpha() {
        let original = Rgba::new(0.8, 0.4, 0.2, 0.6);
        let back = premul_to_straight(straight_to_premul(original));
        assert!(straight_close(back, original));
    }

    #[test]
    fn roundtrip_holds_across_many_alphas() {
        let base = Rgba::new(0.9, 0.3, 0.7, 0.0);
        for step in 1..=32u32 {
            let alpha = f32::from(u16::try_from(step).unwrap()) / 32.0;
            let original = Rgba::new(base.r, base.g, base.b, alpha);
            let back = premul_to_straight(straight_to_premul(original));
            assert!(straight_close(back, original), "alpha {alpha} failed");
        }
    }

    #[test]
    fn premul_to_straight_zero_alpha_is_protected() {
        let s = premul_to_straight(PremulRgba::new(0.0, 0.0, 0.0, 0.0));
        assert!(straight_close(s, Rgba::new(0.0, 0.0, 0.0, 0.0)));
    }

    #[test]
    fn straight_to_premul_zero_alpha_zeros_rgb() {
        let p = straight_to_premul(Rgba::new(1.0, 1.0, 1.0, 0.0));
        assert!(premul_close(p, PremulRgba::TRANSPARENT));
    }

    #[test]
    fn src_over_opaque_source_equals_source() {
        let src = PremulRgba::new(0.4, 0.3, 0.2, 1.0);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.7);
        let out = composite(BlendOp::SrcOver, src, dst);
        assert!(premul_close(out, src));
    }

    #[test]
    fn src_over_transparent_source_equals_dest() {
        let src = PremulRgba::TRANSPARENT;
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.7);
        let out = composite(BlendOp::SrcOver, src, dst);
        assert!(premul_close(out, dst));
    }

    #[test]
    fn clear_yields_transparent() {
        let src = PremulRgba::new(0.4, 0.3, 0.2, 1.0);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.7);
        let out = composite(BlendOp::Clear, src, dst);
        assert!(premul_close(out, PremulRgba::TRANSPARENT));
    }

    #[test]
    fn src_operator_is_identity_on_source() {
        let src = PremulRgba::new(0.4, 0.3, 0.2, 0.8);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.7);
        assert!(premul_close(composite(BlendOp::Src, src, dst), src));
    }

    #[test]
    fn dst_operator_is_identity_on_dest() {
        let src = PremulRgba::new(0.4, 0.3, 0.2, 0.8);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.7);
        assert!(premul_close(composite(BlendOp::Dst, src, dst), dst));
    }

    #[test]
    fn src_in_scales_source_by_dest_alpha() {
        let src = PremulRgba::new(0.4, 0.2, 0.6, 0.8);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.5);
        let out = composite(BlendOp::SrcIn, src, dst);
        let expected = PremulRgba::new(0.4 * 0.5, 0.2 * 0.5, 0.6 * 0.5, 0.8 * 0.5);
        assert!(premul_close(out, expected));
    }

    #[test]
    fn dst_in_scales_dest_by_source_alpha() {
        let src = PremulRgba::new(0.4, 0.2, 0.6, 0.8);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.5);
        let out = composite(BlendOp::DstIn, src, dst);
        let expected = PremulRgba::new(0.1 * 0.8, 0.9 * 0.8, 0.5 * 0.8, 0.5 * 0.8);
        assert!(premul_close(out, expected));
    }

    #[test]
    fn src_out_scales_source_by_inverse_dest_alpha() {
        let src = PremulRgba::new(0.4, 0.2, 0.6, 0.8);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.5);
        let out = composite(BlendOp::SrcOut, src, dst);
        let f = 1.0 - 0.5;
        let expected = PremulRgba::new(0.4 * f, 0.2 * f, 0.6 * f, 0.8 * f);
        assert!(premul_close(out, expected));
    }

    #[test]
    fn dst_out_scales_dest_by_inverse_source_alpha() {
        let src = PremulRgba::new(0.4, 0.2, 0.6, 0.8);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.5);
        let out = composite(BlendOp::DstOut, src, dst);
        let f = 1.0 - 0.8;
        let expected = PremulRgba::new(0.1 * f, 0.9 * f, 0.5 * f, 0.5 * f);
        assert!(premul_close(out, expected));
    }

    #[test]
    fn dst_over_matches_factors() {
        let src = PremulRgba::new(0.4, 0.2, 0.6, 0.8);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.5);
        let out = composite(BlendOp::DstOver, src, dst);
        let fa = 1.0 - 0.5;
        let expected = PremulRgba::new(
            0.4 * fa + 0.1,
            0.2 * fa + 0.9,
            0.6 * fa + 0.5,
            0.8 * fa + 0.5,
        );
        assert!(premul_close(out, expected));
    }

    #[test]
    fn src_atop_matches_factors() {
        let src = PremulRgba::new(0.4, 0.2, 0.6, 0.8);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.5);
        let out = composite(BlendOp::SrcAtop, src, dst);
        let fa = 0.5;
        let fb = 1.0 - 0.8;
        let expected = PremulRgba::new(
            0.4 * fa + 0.1 * fb,
            0.2 * fa + 0.9 * fb,
            0.6 * fa + 0.5 * fb,
            0.8 * fa + 0.5 * fb,
        );
        assert!(premul_close(out, expected));
    }

    #[test]
    fn dst_atop_matches_factors() {
        let src = PremulRgba::new(0.4, 0.2, 0.6, 0.8);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.5);
        let out = composite(BlendOp::DstAtop, src, dst);
        let fa = 1.0 - 0.5;
        let fb = 0.8;
        let expected = PremulRgba::new(
            0.4 * fa + 0.1 * fb,
            0.2 * fa + 0.9 * fb,
            0.6 * fa + 0.5 * fb,
            0.8 * fa + 0.5 * fb,
        );
        assert!(premul_close(out, expected));
    }

    #[test]
    fn xor_matches_factors() {
        let src = PremulRgba::new(0.4, 0.2, 0.6, 0.8);
        let dst = PremulRgba::new(0.1, 0.9, 0.5, 0.5);
        let out = composite(BlendOp::Xor, src, dst);
        let fa = 1.0 - 0.5;
        let fb = 1.0 - 0.8;
        let expected = PremulRgba::new(
            0.4 * fa + 0.1 * fb,
            0.2 * fa + 0.9 * fb,
            0.6 * fa + 0.5 * fb,
            0.8 * fa + 0.5 * fb,
        );
        assert!(premul_close(out, expected));
    }

    #[test]
    fn plus_clamps_each_channel_to_one() {
        let src = PremulRgba::new(0.7, 0.2, 0.9, 0.6);
        let dst = PremulRgba::new(0.5, 0.3, 0.4, 0.7);
        let out = composite(BlendOp::Plus, src, dst);
        let expected = PremulRgba::new(1.0, 0.5, 1.0, 1.0);
        assert!(premul_close(out, expected));
    }

    #[test]
    fn add_is_same_additive_blend_as_plus() {
        let src = PremulRgba::new(0.7, 0.2, 0.9, 0.6);
        let dst = PremulRgba::new(0.5, 0.3, 0.4, 0.7);
        let via_add = composite(BlendOp::Add, src, dst);
        let via_plus = composite(BlendOp::Plus, src, dst);
        assert!(premul_close(via_add, via_plus));
    }

    #[test]
    fn src_over_is_approximately_associative() {
        let a = PremulRgba::new(0.3, 0.1, 0.2, 0.5);
        let b = PremulRgba::new(0.2, 0.4, 0.1, 0.4);
        let c = PremulRgba::new(0.1, 0.2, 0.3, 0.6);
        let left = composite(BlendOp::SrcOver, composite(BlendOp::SrcOver, a, b), c);
        let right = composite(BlendOp::SrcOver, a, composite(BlendOp::SrcOver, b, c));
        assert!(premul_close(left, right));
    }

    #[test]
    fn std430_size_is_sixteen() {
        assert_eq!(PREMUL_STD430_SIZE, 16);
        assert_eq!(PremulRgba::STD430_SIZE, VEC4_STRIDE);
    }

    #[test]
    fn std430_roundtrips_through_little_endian_lanes() {
        let color = PremulRgba::new(0.25, 0.5, 0.75, 0.125);
        let bytes = color.to_std430();
        let r = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let g = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let b = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let a = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert!(premul_close(PremulRgba::new(r, g, b, a), color));
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(PremulRgba::gpu_storage_bytes(0), VEC4_STRIDE);
        assert_eq!(PremulRgba::gpu_storage_bytes(1), VEC4_STRIDE);
        assert_eq!(PremulRgba::gpu_storage_bytes(8), 8 * VEC4_STRIDE);
    }

    #[test]
    fn batch_matches_per_element_src_over() {
        let src = [
            PremulRgba::new(0.4, 0.3, 0.2, 1.0),
            PremulRgba::TRANSPARENT,
            PremulRgba::new(0.2, 0.2, 0.2, 0.5),
        ];
        let dst = [
            PremulRgba::new(0.1, 0.9, 0.5, 0.7),
            PremulRgba::new(0.6, 0.1, 0.3, 0.8),
            PremulRgba::new(0.3, 0.3, 0.3, 0.4),
        ];
        let batch = composite_over_batch(&src, &dst);
        assert_eq!(batch.len(), 3);
        for i in 0..3 {
            let expected = composite(BlendOp::SrcOver, src[i], dst[i]);
            assert!(premul_close(batch[i], expected), "index {i} mismatch");
        }
    }

    #[test]
    fn batch_length_is_shorter_input() {
        let src = [PremulRgba::TRANSPARENT, PremulRgba::TRANSPARENT];
        let dst = [PremulRgba::TRANSPARENT];
        assert_eq!(composite_over_batch(&src, &dst).len(), 1);
        assert_eq!(composite_over_batch(&dst, &src).len(), 1);
    }
}
