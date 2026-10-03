//! Deterministic per-frame world state hashing (design §14: 逐帧状态哈希去同步).
//!
//! Rollback networking (Quantum / GGPO form) detects desync by comparing a
//! compact hash of each simulated frame between peers. The hash must be
//! build-stable and independent of allocation order, so it is folded in a fixed,
//! deterministic traversal order: entities ascending by [`Entity::to_bits`], and
//! within the world, columns ascending by [`ComponentId`].
//!
//! Honesty: a value only contributes its *bytes* to the hash when its component
//! was registered as hashable
//! ([`register_snapshot_component_hashable`](crate::World::register_snapshot_component_hashable)).
//! Components that are merely cloneable (e.g. ones holding `f32`, which is not
//! [`core::hash::Hash`]) still contribute their *structural* presence —
//! `(entity, component_id)` — but not their value. The hash therefore always
//! detects structural divergence and detects value divergence for every
//! hash-registered component.

/// A small, dependency-free [`core::hash::Hasher`] implementing 64-bit FNV-1a.
///
/// FNV-1a is chosen for determinism and simplicity (no SipHash keys, no std):
/// identical byte streams always fold to the same 64-bit value on every target,
/// which is exactly the contract a cross-peer desync check needs.
pub struct FnvHasher {
    state: u64,
}

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

impl FnvHasher {
    /// Create a hasher seeded with the FNV offset basis.
    #[inline]
    pub fn new() -> Self {
        Self { state: FNV_OFFSET }
    }
}

impl Default for FnvHasher {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl core::hash::Hasher for FnvHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut hash = self.state;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        self.state = hash;
    }
}
