//! `CPU`-verifiable contract for the fixed-function `stencil` test/operation
//! state machine that gates particle draws (design §9 render pass, §16 shading
//! router `OIT`/mask stages).
//!
//! Every production `GPU`-driven `VFX` engine leans on the hardware `stencil`
//! buffer to mask particle draws: decal receivers, portal/mirror regions,
//! soft-particle write regions, and the `NPR` outline pass all program the same
//! fixed-function unit that `DirectX` (`D3D12_DEPTH_STENCIL_DESC`) and `Vulkan`
//! (`VkStencilOpState`) expose. That unit is two mirrored *faces* (front/back),
//! each holding a compare function, a reference value, a read mask, a write
//! mask, and the three operations selected by the compare/depth outcome. This
//! module owns the pure, device-free half of that contract: the compare
//! predicate, the eight `stencil` operations, per-face resolution, the masked
//! write-back, and the `std430` byte layout a pipeline record binds.
//!
//! The primitive here is an unsigned integer `stencil` value (`u32`, holding an
//! 8-bit hardware value in its low bits by convention). All arithmetic is exact
//! integer bit-twiddling — masks, `wrapping_add`/`saturating_sub`, bitwise
//! complement — so the `CPU` reference and the eventual `GPU` path agree bit for
//! bit and nothing here can panic, overflow, or divide by zero.
//!
//! 1. [`CompareFunc`] — the eight comparison predicates, each applied to the
//!    read-masked reference and buffer values via [`CompareFunc::test`].
//! 2. [`StencilOp`] — the eight buffer operations, applied by
//!    [`StencilOp::apply`] against a caller-supplied maximum representable value.
//! 3. [`write_masked`] — the `(old & !mask) | (new & mask)` write-back rule.
//! 4. [`StencilFace`] — one programmed face; [`StencilFace::resolve`] runs the
//!    full test → op-select → write-back pipeline.
//! 5. [`StencilState`] — the front/back face pair a pipeline binds, with its
//!    `std430` byte layout ([`StencilState::std430_bytes`],
//!    [`gpu_storage_bytes`]).
//!
//! Scope boundary: this module is *only* the `stencil` unit. Depth comparison,
//! depth bias, and depth-range policy live in the sibling `depth_*` contracts;
//! [`StencilFace::resolve`] consumes an already-computed `depth_passed` boolean
//! and never re-derives the depth test itself.

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};
use alloc::vec::Vec;

/// Number of programmable words packed per face in the `std430` record: the
/// seven live fields plus one padding word so the face is a multiple of 16
/// bytes.
pub const FACE_WORD_COUNT: usize = 8;

/// `std430` byte size of one packed [`StencilFace`] (a multiple of 16 bytes as
/// a `WebGPU` struct-array element requires).
pub const FACE_STD430_SIZE: usize = FACE_WORD_COUNT * U32_STRIDE;

/// `std430` byte size of one packed [`StencilState`] (front face + back face).
pub const STATE_STD430_SIZE: usize = FACE_STD430_SIZE * 2;

/// The comparison predicate a face applies between the read-masked reference
/// value and the read-masked buffer value.
///
/// The predicate reads as `reference op value`, matching the `DirectX` and
/// `Vulkan` convention: [`CompareFunc::Less`] passes when the reference is less
/// than the stored value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareFunc {
    /// The test never passes.
    Never,
    /// Passes when `reference < value`.
    Less,
    /// Passes when `reference == value`.
    Equal,
    /// Passes when `reference <= value`.
    LessEqual,
    /// Passes when `reference > value`.
    Greater,
    /// Passes when `reference != value`.
    NotEqual,
    /// Passes when `reference >= value`.
    GreaterEqual,
    /// The test always passes.
    Always,
}

impl CompareFunc {
    /// Every predicate in stable enumerant order.
    pub const ALL: [Self; 8] = [
        Self::Never,
        Self::Less,
        Self::Equal,
        Self::LessEqual,
        Self::Greater,
        Self::NotEqual,
        Self::GreaterEqual,
        Self::Always,
    ];

    /// Stable `u32` encoding matching the enumerant order (`Never == 0`).
    #[must_use]
    pub fn to_u32(&self) -> u32 {
        match self {
            Self::Never => 0,
            Self::Less => 1,
            Self::Equal => 2,
            Self::LessEqual => 3,
            Self::Greater => 4,
            Self::NotEqual => 5,
            Self::GreaterEqual => 6,
            Self::Always => 7,
        }
    }

    /// Evaluate the predicate after masking both operands with `read_mask`.
    ///
    /// The reference and buffer values are each masked with `read_mask` via a
    /// bitwise conjunction before the comparison, exactly as the hardware
    /// `stencil` unit applies its read mask.
    #[must_use]
    pub fn test(&self, ref_val: u32, stencil_val: u32, read_mask: u32) -> bool {
        let masked_ref = ref_val & read_mask;
        let masked_val = stencil_val & read_mask;
        match self {
            Self::Never => false,
            Self::Less => masked_ref < masked_val,
            Self::Equal => masked_ref == masked_val,
            Self::LessEqual => masked_ref <= masked_val,
            Self::Greater => masked_ref > masked_val,
            Self::NotEqual => masked_ref != masked_val,
            Self::GreaterEqual => masked_ref >= masked_val,
            Self::Always => true,
        }
    }
}

/// The operation applied to the `stencil` buffer value once the test/depth
/// outcome selects it.
///
/// The clamp/wrap variants use a caller-supplied `max_val` (the largest value
/// the hardware `stencil` bit-depth can hold — `255` for an 8-bit buffer)
/// instead of the full `u32` range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StencilOp {
    /// Leave the current value unchanged.
    Keep,
    /// Set the value to zero.
    Zero,
    /// Replace the value with the face reference value.
    Replace,
    /// Increment, saturating at `max_val`.
    IncrementClamp,
    /// Decrement, saturating at zero.
    DecrementClamp,
    /// Bitwise-invert the current value.
    Invert,
    /// Increment, wrapping to zero once `max_val` is exceeded.
    IncrementWrap,
    /// Decrement, wrapping to `max_val` from zero.
    DecrementWrap,
}

impl StencilOp {
    /// Every operation in stable enumerant order.
    pub const ALL: [Self; 8] = [
        Self::Keep,
        Self::Zero,
        Self::Replace,
        Self::IncrementClamp,
        Self::DecrementClamp,
        Self::Invert,
        Self::IncrementWrap,
        Self::DecrementWrap,
    ];

    /// Stable `u32` encoding matching the enumerant order (`Keep == 0`).
    #[must_use]
    pub fn to_u32(&self) -> u32 {
        match self {
            Self::Keep => 0,
            Self::Zero => 1,
            Self::Replace => 2,
            Self::IncrementClamp => 3,
            Self::DecrementClamp => 4,
            Self::Invert => 5,
            Self::IncrementWrap => 6,
            Self::DecrementWrap => 7,
        }
    }

    /// Apply the operation to `current`, using `ref_val` for
    /// [`StencilOp::Replace`] and `max_val` as the wrap/clamp ceiling.
    #[must_use]
    pub fn apply(&self, current: u32, ref_val: u32, max_val: u32) -> u32 {
        match self {
            Self::Keep => current,
            Self::Zero => 0,
            Self::Replace => ref_val,
            Self::IncrementClamp => current.saturating_add(1).min(max_val),
            Self::DecrementClamp => current.saturating_sub(1),
            Self::Invert => !current,
            Self::IncrementWrap => {
                if current >= max_val {
                    0
                } else {
                    current.wrapping_add(1)
                }
            }
            Self::DecrementWrap => {
                if current == 0 {
                    max_val
                } else {
                    current.wrapping_sub(1)
                }
            }
        }
    }
}

/// Combine the surviving bits of `old` with the incoming bits of `new` under a
/// write mask: `(old & !write_mask) | (new & write_mask)`.
///
/// Only the bits set in `write_mask` are taken from `new`; every other bit is
/// preserved from `old`, matching the hardware `stencil` write-mask rule.
#[must_use]
pub fn write_masked(old: u32, new: u32, write_mask: u32) -> u32 {
    (old & !write_mask) | (new & write_mask)
}

/// One programmed `stencil` face: the compare predicate, its reference value
/// and masks, and the three operations the test/depth outcome selects between.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilFace {
    /// Comparison predicate applied by [`CompareFunc::test`].
    pub compare: CompareFunc,
    /// Reference value compared against the buffer and used by
    /// [`StencilOp::Replace`].
    pub reference: u32,
    /// Mask applied to both operands before the compare.
    pub read_mask: u32,
    /// Mask restricting which bits [`StencilFace::resolve`] writes back.
    pub write_mask: u32,
    /// Operation applied when the compare fails.
    pub fail_op: StencilOp,
    /// Operation applied when the compare passes but the depth test fails.
    pub depth_fail_op: StencilOp,
    /// Operation applied when both the compare and depth test pass.
    pub pass_op: StencilOp,
}

impl StencilFace {
    /// A face that always passes and leaves the buffer untouched with full
    /// masks (`0xFF_FF_FF_FF`) — the neutral "`stencil` disabled" record.
    #[must_use]
    pub fn keep_always() -> Self {
        Self {
            compare: CompareFunc::Always,
            reference: 0,
            read_mask: 0xFF_FF_FF_FF,
            write_mask: 0xFF_FF_FF_FF,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::Keep,
        }
    }

    /// Select the operation this face applies for a given compare/depth
    /// outcome, without touching the buffer.
    #[must_use]
    pub fn selected_op(&self, compare_passed: bool, depth_passed: bool) -> StencilOp {
        if !compare_passed {
            self.fail_op
        } else if depth_passed {
            self.pass_op
        } else {
            self.depth_fail_op
        }
    }

    /// Run the full face pipeline: compare, select the operation from the
    /// compare/`depth_passed` outcome, apply it, then write the result back
    /// under [`StencilFace::write_mask`].
    #[must_use]
    pub fn resolve(&self, stencil_val: u32, depth_passed: bool, max_val: u32) -> u32 {
        let compare_passed = self
            .compare
            .test(self.reference, stencil_val, self.read_mask);
        let op = self.selected_op(compare_passed, depth_passed);
        let new_val = op.apply(stencil_val, self.reference, max_val);
        write_masked(stencil_val, new_val, self.write_mask)
    }

    /// Pack the face into its `std430` word array (seven live fields plus one
    /// zero padding word).
    #[must_use]
    pub fn to_std430(&self) -> [u32; FACE_WORD_COUNT] {
        [
            self.compare.to_u32(),
            self.reference,
            self.read_mask,
            self.write_mask,
            self.fail_op.to_u32(),
            self.depth_fail_op.to_u32(),
            self.pass_op.to_u32(),
            0,
        ]
    }

    /// Serialize the packed face as little-endian `std430` bytes.
    #[must_use]
    pub fn std430_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(FACE_STD430_SIZE);
        for word in self.to_std430() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The front/back `stencil` face pair a pipeline record binds.
///
/// Front-facing and back-facing primitives are gated by independent faces, so
/// two-sided masking (portals, hollow decals) programs each side separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilState {
    /// Face applied to front-facing primitives.
    pub front: StencilFace,
    /// Face applied to back-facing primitives.
    pub back: StencilFace,
}

impl StencilState {
    /// A state whose two faces both keep-and-always-pass (the neutral
    /// "`stencil` disabled" record).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            front: StencilFace::keep_always(),
            back: StencilFace::keep_always(),
        }
    }

    /// A symmetric state where both faces share the same programming.
    #[must_use]
    pub fn uniform(face: StencilFace) -> Self {
        Self {
            front: face,
            back: face,
        }
    }

    /// Resolve the appropriate face for a primitive's facing.
    ///
    /// `front_facing` selects [`StencilState::front`] when `true`, otherwise
    /// [`StencilState::back`].
    #[must_use]
    pub fn resolve(
        &self,
        front_facing: bool,
        stencil_val: u32,
        depth_passed: bool,
        max_val: u32,
    ) -> u32 {
        let face = if front_facing {
            &self.front
        } else {
            &self.back
        };
        face.resolve(stencil_val, depth_passed, max_val)
    }

    /// Serialize the front face followed by the back face as little-endian
    /// `std430` bytes.
    #[must_use]
    pub fn std430_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(STATE_STD430_SIZE);
        bytes.extend_from_slice(&self.front.std430_bytes());
        bytes.extend_from_slice(&self.back.std430_bytes());
        bytes
    }
}

/// Total `std430` storage-buffer byte size for `count` packed [`StencilState`]
/// records, clamped up to a single element for a non-empty `WebGPU` binding.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(STATE_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_all_covers_eight_distinct() {
        assert_eq!(CompareFunc::ALL.len(), 8);
        for (i, a) in CompareFunc::ALL.iter().enumerate() {
            for (j, b) in CompareFunc::ALL.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn compare_to_u32_is_dense_and_ordered() {
        for (i, func) in CompareFunc::ALL.iter().enumerate() {
            let expected = u32::try_from(i).expect("index fits in u32");
            assert_eq!(func.to_u32(), expected);
        }
    }

    #[test]
    fn compare_never_and_always() {
        assert!(!CompareFunc::Never.test(3, 3, 0xFF));
        assert!(CompareFunc::Always.test(0, 999, 0xFF));
    }

    #[test]
    fn compare_ordering_predicates() {
        let mask = 0xFF;
        assert!(CompareFunc::Less.test(2, 5, mask));
        assert!(!CompareFunc::Less.test(5, 2, mask));
        assert!(CompareFunc::Greater.test(5, 2, mask));
        assert!(!CompareFunc::Greater.test(2, 5, mask));
        assert!(CompareFunc::LessEqual.test(5, 5, mask));
        assert!(CompareFunc::GreaterEqual.test(5, 5, mask));
        assert!(CompareFunc::Equal.test(7, 7, mask));
        assert!(CompareFunc::NotEqual.test(7, 8, mask));
    }

    #[test]
    fn compare_applies_read_mask_to_both_operands() {
        // Low nibble differs, high bits differ; masking to low nibble makes
        // them equal.
        let a = 0xF3;
        let b = 0x03;
        assert!(!CompareFunc::Equal.test(a, b, 0xFF));
        assert!(CompareFunc::Equal.test(a, b, 0x0F));
    }

    #[test]
    fn compare_full_mask_matches_raw() {
        let mask = 0xFF_FF_FF_FF;
        assert!(CompareFunc::Less.test(10, 20, mask));
        assert!(CompareFunc::NotEqual.test(10, 20, mask));
    }

    #[test]
    fn op_all_covers_eight_distinct() {
        assert_eq!(StencilOp::ALL.len(), 8);
        for (i, a) in StencilOp::ALL.iter().enumerate() {
            for (j, b) in StencilOp::ALL.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn op_to_u32_is_dense_and_ordered() {
        for (i, op) in StencilOp::ALL.iter().enumerate() {
            let expected = u32::try_from(i).expect("index fits in u32");
            assert_eq!(op.to_u32(), expected);
        }
    }

    #[test]
    fn op_keep_and_zero_and_replace() {
        assert_eq!(StencilOp::Keep.apply(42, 7, 255), 42);
        assert_eq!(StencilOp::Zero.apply(42, 7, 255), 0);
        assert_eq!(StencilOp::Replace.apply(42, 7, 255), 7);
    }

    #[test]
    fn op_increment_clamp_saturates_at_max() {
        assert_eq!(StencilOp::IncrementClamp.apply(10, 0, 255), 11);
        assert_eq!(StencilOp::IncrementClamp.apply(255, 0, 255), 255);
        assert_eq!(StencilOp::IncrementClamp.apply(300, 0, 255), 255);
    }

    #[test]
    fn op_decrement_clamp_saturates_at_zero() {
        assert_eq!(StencilOp::DecrementClamp.apply(10, 0, 255), 9);
        assert_eq!(StencilOp::DecrementClamp.apply(0, 0, 255), 0);
    }

    #[test]
    fn op_increment_wrap_wraps_at_max() {
        assert_eq!(StencilOp::IncrementWrap.apply(10, 0, 255), 11);
        assert_eq!(StencilOp::IncrementWrap.apply(255, 0, 255), 0);
        assert_eq!(StencilOp::IncrementWrap.apply(255, 0, 15), 0);
        assert_eq!(StencilOp::IncrementWrap.apply(15, 0, 15), 0);
    }

    #[test]
    fn op_decrement_wrap_wraps_at_zero() {
        assert_eq!(StencilOp::DecrementWrap.apply(10, 0, 255), 9);
        assert_eq!(StencilOp::DecrementWrap.apply(0, 0, 255), 255);
        assert_eq!(StencilOp::DecrementWrap.apply(0, 0, 15), 15);
    }

    #[test]
    fn op_invert_is_bitwise_complement() {
        assert_eq!(StencilOp::Invert.apply(0x00, 0, 255), 0xFF_FF_FF_FF);
        assert_eq!(StencilOp::Invert.apply(0xFF_FF_FF_FF, 0, 255), 0x00);
        assert_eq!(StencilOp::Invert.apply(0x0F, 0, 255), 0xFF_FF_FF_F0);
    }

    #[test]
    fn write_masked_selects_bits() {
        assert_eq!(write_masked(0xAA, 0x55, 0x00), 0xAA);
        assert_eq!(write_masked(0xAA, 0x55, 0xFF), 0x55);
        assert_eq!(write_masked(0xF0, 0x0F, 0x0F), 0xFF);
        assert_eq!(write_masked(0xFF, 0x00, 0x0F), 0xF0);
    }

    #[test]
    fn write_masked_preserves_unmasked_bits() {
        let old = 0x1234_5678;
        let new = 0xFFFF_FFFF;
        let mask = 0x0000_FF00;
        assert_eq!(write_masked(old, new, mask), 0x1234_FF78);
    }

    #[test]
    fn face_selected_op_matches_outcome() {
        let face = StencilFace {
            compare: CompareFunc::Always,
            reference: 1,
            read_mask: 0xFF,
            write_mask: 0xFF,
            fail_op: StencilOp::Zero,
            depth_fail_op: StencilOp::IncrementClamp,
            pass_op: StencilOp::Replace,
        };
        assert_eq!(face.selected_op(false, true), StencilOp::Zero);
        assert_eq!(face.selected_op(false, false), StencilOp::Zero);
        assert_eq!(face.selected_op(true, true), StencilOp::Replace);
        assert_eq!(face.selected_op(true, false), StencilOp::IncrementClamp);
    }

    #[test]
    fn face_resolve_pass_op_on_full_pass() {
        let face = StencilFace {
            compare: CompareFunc::Equal,
            reference: 5,
            read_mask: 0xFF,
            write_mask: 0xFF,
            fail_op: StencilOp::Zero,
            depth_fail_op: StencilOp::DecrementClamp,
            pass_op: StencilOp::Replace,
        };
        // Compare passes (5 == 5) and depth passes -> Replace with reference.
        assert_eq!(face.resolve(5, true, 255), 5);
    }

    #[test]
    fn face_resolve_depth_fail_op() {
        let face = StencilFace {
            compare: CompareFunc::Equal,
            reference: 5,
            read_mask: 0xFF,
            write_mask: 0xFF,
            fail_op: StencilOp::Zero,
            depth_fail_op: StencilOp::IncrementClamp,
            pass_op: StencilOp::Replace,
        };
        // Compare passes but depth fails -> IncrementClamp on current value.
        assert_eq!(face.resolve(5, false, 255), 6);
    }

    #[test]
    fn face_resolve_fail_op() {
        let face = StencilFace {
            compare: CompareFunc::Equal,
            reference: 5,
            read_mask: 0xFF,
            write_mask: 0xFF,
            fail_op: StencilOp::Zero,
            depth_fail_op: StencilOp::IncrementClamp,
            pass_op: StencilOp::Replace,
        };
        // Compare fails (5 != 9) -> Zero regardless of depth outcome.
        assert_eq!(face.resolve(9, true, 255), 0);
        assert_eq!(face.resolve(9, false, 255), 0);
    }

    #[test]
    fn face_resolve_honors_write_mask() {
        let face = StencilFace {
            compare: CompareFunc::Always,
            reference: 0xFF,
            read_mask: 0xFF,
            write_mask: 0x0F,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::Replace,
        };
        // Replace to 0xFF but only low nibble is writable over old 0x30.
        assert_eq!(face.resolve(0x30, true, 255), 0x3F);
    }

    #[test]
    fn keep_always_is_identity() {
        let face = StencilFace::keep_always();
        for &val in &[0u32, 1, 100, 255, 0xFF_FF_FF_FF] {
            assert_eq!(face.resolve(val, true, 255), val);
            assert_eq!(face.resolve(val, false, 255), val);
        }
    }

    #[test]
    fn state_resolve_selects_face_by_facing() {
        let front = StencilFace {
            compare: CompareFunc::Always,
            reference: 1,
            read_mask: 0xFF,
            write_mask: 0xFF,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::Replace,
        };
        let back = StencilFace {
            compare: CompareFunc::Always,
            reference: 2,
            read_mask: 0xFF,
            write_mask: 0xFF,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::Replace,
        };
        let state = StencilState { front, back };
        assert_eq!(state.resolve(true, 0, true, 255), 1);
        assert_eq!(state.resolve(false, 0, true, 255), 2);
    }

    #[test]
    fn state_disabled_is_identity_both_faces() {
        let state = StencilState::disabled();
        assert_eq!(state.resolve(true, 77, true, 255), 77);
        assert_eq!(state.resolve(false, 77, false, 255), 77);
    }

    #[test]
    fn state_uniform_shares_programming() {
        let face = StencilFace::keep_always();
        let state = StencilState::uniform(face);
        assert_eq!(state.front, state.back);
    }

    #[test]
    fn face_std430_size_is_multiple_of_sixteen() {
        let bytes = StencilFace::keep_always().std430_bytes();
        assert_eq!(bytes.len(), FACE_STD430_SIZE);
        assert_eq!(bytes.len() % 16, 0);
    }

    #[test]
    fn face_std430_word_layout() {
        let face = StencilFace {
            compare: CompareFunc::Greater,
            reference: 0x1122_3344,
            read_mask: 0x00FF_00FF,
            write_mask: 0xFF00_FF00,
            fail_op: StencilOp::Zero,
            depth_fail_op: StencilOp::Invert,
            pass_op: StencilOp::Replace,
        };
        let words = face.to_std430();
        assert_eq!(words[0], CompareFunc::Greater.to_u32());
        assert_eq!(words[1], 0x1122_3344);
        assert_eq!(words[2], 0x00FF_00FF);
        assert_eq!(words[3], 0xFF00_FF00);
        assert_eq!(words[4], StencilOp::Zero.to_u32());
        assert_eq!(words[5], StencilOp::Invert.to_u32());
        assert_eq!(words[6], StencilOp::Replace.to_u32());
        assert_eq!(words[7], 0);
    }

    #[test]
    fn face_std430_bytes_are_little_endian() {
        let face = StencilFace {
            compare: CompareFunc::Never,
            reference: 0x0403_0201,
            read_mask: 0,
            write_mask: 0,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::Keep,
        };
        let bytes = face.std430_bytes();
        // Word 1 (reference) starts at byte offset 4.
        assert_eq!(&bytes[4..8], &[0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn state_std430_size_and_split() {
        let state = StencilState::disabled();
        let bytes = state.std430_bytes();
        assert_eq!(bytes.len(), STATE_STD430_SIZE);
        assert_eq!(bytes.len() % 16, 0);
        let front_bytes = state.front.std430_bytes();
        assert_eq!(&bytes[..FACE_STD430_SIZE], front_bytes.as_slice());
    }

    #[test]
    fn gpu_storage_bytes_clamps_empty_to_one() {
        assert_eq!(gpu_storage_bytes(0), STATE_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), STATE_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(4), STATE_STD430_SIZE * 4);
    }

    #[test]
    fn gpu_storage_bytes_never_overflows() {
        // Saturating multiply keeps a degenerate count from wrapping small.
        assert_eq!(gpu_storage_bytes(usize::MAX), usize::MAX);
    }

    #[test]
    fn wrap_and_clamp_differ_at_boundaries() {
        let max = 7;
        assert_eq!(StencilOp::IncrementClamp.apply(max, 0, max), max);
        assert_eq!(StencilOp::IncrementWrap.apply(max, 0, max), 0);
        assert_eq!(StencilOp::DecrementClamp.apply(0, 0, max), 0);
        assert_eq!(StencilOp::DecrementWrap.apply(0, 0, max), max);
    }
}
