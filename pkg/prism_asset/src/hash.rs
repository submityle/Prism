//! Frozen FNV-1a hash primitives for stable asset identity.
//!
//! Asset identity ([`StableGuid`](crate::StableGuid), [`AssetTypeId`](crate::AssetTypeId))
//! is a **persisted, cross-run/-platform contract**: the same source must hash
//! to the same bits on every machine, this run and ten years from now. The
//! derivation therefore cannot depend on anything that might change out from
//! under it — not `core::hash::Hasher` (unspecified byte order), not a
//! randomized `BuildHasher`, and not an upstream utility crate that could
//! refactor its algorithm or pull in `std`.
//!
//! So the kernel owns its hash primitives directly: canonical FNV-1a in 64- and
//! 128-bit widths, computed byte-by-byte with the published offset basis and
//! prime. These constants and this iteration order are a **frozen contract**;
//! changing them reshuffles every stored guid and must go through an explicit
//! format-version bump and remap table (see `docs/prism_asset_design_zh.md`
//! §20). The values intentionally match the reference FNV-1a specification, so
//! external tooling can reproduce an identity without linking this crate.
//!
//! The code is pure `core` (no `alloc`, no `std`), keeping the identity core
//! `no_std`-clean and dependency-free.

/// FNV-1a 64-bit offset basis (reference FNV spec).
pub(crate) const FNV64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime (reference FNV spec).
pub(crate) const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;
/// FNV-1a 128-bit offset basis (reference FNV spec).
pub(crate) const FNV128_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
/// FNV-1a 128-bit prime (reference FNV spec).
pub(crate) const FNV128_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

/// Canonical FNV-1a 64-bit hash of a byte slice.
#[inline]
#[must_use]
pub(crate) fn fnv1a_64(bytes: &[u8]) -> u64 {
    fnv1a_64_fold(FNV64_OFFSET, bytes)
}

/// Folds `bytes` into an existing FNV-1a 64-bit accumulator, so an identity can
/// be built incrementally across several fields while remaining bit-identical
/// to hashing the concatenation.
#[inline]
#[must_use]
pub(crate) fn fnv1a_64_fold(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV64_PRIME);
    }
    hash
}

/// Canonical FNV-1a 128-bit hash of a byte slice.
#[inline]
#[must_use]
pub(crate) fn fnv1a_128(bytes: &[u8]) -> u128 {
    fnv1a_128_fold(FNV128_OFFSET, bytes)
}

/// Folds `bytes` into an existing FNV-1a 128-bit accumulator (see
/// [`fnv1a_64_fold`] for the incremental-build rationale).
#[inline]
#[must_use]
pub(crate) fn fnv1a_128_fold(mut hash: u128, bytes: &[u8]) -> u128 {
    for &b in bytes {
        hash ^= u128::from(b);
        hash = hash.wrapping_mul(FNV128_PRIME);
    }
    hash
}
