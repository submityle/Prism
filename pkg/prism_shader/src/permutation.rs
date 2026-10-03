//! Stable, order-independent identifier for a shader permutation.
//!
//! A [`PermutationId`] is a 64-bit fingerprint of a [`ShaderDefs`] set. It is
//! used as a cache key: the same logical permutation (same names bound to the
//! same values) always produces the same id across runs and platforms, because
//! it is folded in [`ShaderDefs`]'s canonical name-sorted order with a fixed
//! `FNV-1a` hash and an explicit, versioned encoding (no `Hash`-trait /
//! `RandomState` nondeterminism, no pointer or allocation-order dependence).

use crate::def::{ShaderDefValue, ShaderDefs};

/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Encoding version mixed into every id so a future change to the folding
/// scheme cannot silently collide with ids persisted by an older version.
const ENCODING_VERSION: u8 = 1;

/// Folds one byte into a running `FNV-1a` accumulator.
#[must_use]
const fn fnv1a_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ byte as u64).wrapping_mul(FNV_PRIME)
}

/// Folds a byte slice into a running `FNV-1a` accumulator.
#[must_use]
fn fnv1a_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash = fnv1a_byte(hash, byte);
    }
    hash
}

/// Folds a tag byte then a little-endian `u64` into the accumulator.
#[must_use]
fn fnv1a_tagged_u64(hash: u64, tag: u8, value: u64) -> u64 {
    let hash = fnv1a_byte(hash, tag);
    fnv1a_bytes(hash, &value.to_le_bytes())
}

/// A deterministic 64-bit fingerprint of a shader permutation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Debug)]
pub struct PermutationId(pub u64);

impl PermutationId {
    /// The id of the empty permutation (no defs).
    #[must_use]
    pub fn empty() -> Self {
        Self::of(&ShaderDefs::new())
    }

    /// Computes the id of a def set.
    ///
    /// Each def contributes its name (length-prefixed so `"ab"`+`"c"` cannot
    /// alias `"a"`+`"bc"`), a type tag, and its value. The count is folded last
    /// so a prefix of another permutation cannot collide with it.
    #[must_use]
    pub fn of(defs: &ShaderDefs) -> Self {
        let mut hash = fnv1a_byte(FNV_OFFSET, ENCODING_VERSION);
        for (name, value) in defs.iter() {
            // Length-prefix the name to keep the stream unambiguous.
            hash = fnv1a_bytes(hash, &(name.len() as u64).to_le_bytes());
            hash = fnv1a_bytes(hash, name.as_bytes());
            let (tag, payload) = match value {
                ShaderDefValue::Bool(flag) => (0u8, u64::from(flag)),
                ShaderDefValue::Int(signed) => (1u8, signed as u64),
                ShaderDefValue::UInt(unsigned) => (2u8, u64::from(unsigned)),
            };
            hash = fnv1a_tagged_u64(hash, tag, payload);
        }
        hash = fnv1a_tagged_u64(hash, 0xff, defs.len() as u64);
        Self(hash)
    }

    /// The raw 64-bit value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}
