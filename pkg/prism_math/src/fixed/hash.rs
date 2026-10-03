//! A tiny deterministic state hasher for fixed-point simulation state.
//!
//! [`StateHasher`] folds the raw integer bits of [`Fixed`] values (and vectors)
//! into a 64-bit FNV-1a digest. Because it consumes only the authoritative
//! integer `raw` bits — never a float — the digest is bit-identical on every
//! platform for the same sequence of inputs. This is the hook the design doc
//! §16 calls out ("fixed 档可输出每帧状态哈希，供联机 desync 检测"): hash the
//! fixed-point world state each frame and compare across peers to detect
//! divergence.

use super::vec::{FxVec2, FxVec3, FxVec4};
use super::{Fixed, I16F16};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A deterministic 64-bit FNV-1a hasher over fixed-point state.
#[derive(Clone, Copy, Debug)]
pub struct StateHasher {
    state: u64,
}

impl Default for StateHasher {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl StateHasher {
    /// A fresh hasher seeded with the FNV-1a offset basis.
    #[inline]
    pub const fn new() -> Self {
        Self { state: FNV_OFFSET }
    }

    /// Absorb one raw 64-bit integer, little-endian byte by byte.
    #[inline]
    pub fn write_u64(&mut self, value: u64) {
        let mut v = value;
        let mut i = 0;
        while i < 8 {
            let byte = (v & 0xff) as u64;
            self.state = (self.state ^ byte).wrapping_mul(FNV_PRIME);
            v >>= 8;
            i += 1;
        }
    }

    /// Absorb a signed 64-bit integer (reinterpreted as unsigned).
    #[inline]
    pub fn write_i64(&mut self, value: i64) {
        self.write_u64(value as u64);
    }

    /// Absorb a [`Fixed`] by its raw Q32.32 bits.
    #[inline]
    pub fn write_fixed(&mut self, value: Fixed) {
        self.write_i64(value.to_bits());
    }

    /// Absorb an [`I16F16`] by its raw Q16.16 bits.
    #[inline]
    pub fn write_i16f16(&mut self, value: I16F16) {
        self.write_i64(value.to_bits() as i64);
    }

    /// Absorb an [`FxVec2`].
    #[inline]
    pub fn write_fxvec2(&mut self, value: FxVec2) {
        for raw in value.to_bits() {
            self.write_i64(raw);
        }
    }
    /// Absorb an [`FxVec3`].
    #[inline]
    pub fn write_fxvec3(&mut self, value: FxVec3) {
        for raw in value.to_bits() {
            self.write_i64(raw);
        }
    }
    /// Absorb an [`FxVec4`].
    #[inline]
    pub fn write_fxvec4(&mut self, value: FxVec4) {
        for raw in value.to_bits() {
            self.write_i64(raw);
        }
    }

    /// The current 64-bit digest.
    #[inline]
    pub const fn finish(self) -> u64 {
        self.state
    }
}
