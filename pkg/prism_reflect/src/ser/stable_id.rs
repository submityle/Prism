//! [`StableTypeId`]: a deterministic, cross-run/cross-build type identifier.
//!
//! Rust's [`core::any::TypeId`] is *not* stable across compilations, so it
//! cannot be written into a save file or sent over the network (design §21).
//! `StableTypeId` is derived purely from a type's fully-qualified **path**
//! (its [`core::any::type_name`]) via a fixed-seed
//! [FNV-1a](https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function) 64-bit
//! hash. The same type path always hashes to the same value regardless of
//! build, platform, or run, which is exactly the stability contract the
//! serializer needs for versioned type tagging in the stream.
//!
//! The hash algorithm and the "identity is the type path" rule are a frozen
//! on-disk contract (design §22 risk 1): changing either invalidates every
//! previously written stream, so they must not drift. Renames are handled by
//! an explicit old-id → new-type mapping layer (design §21, a later milestone),
//! never by silently changing this hash.

/// The FNV-1a 64-bit offset basis (the fixed seed).
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// The FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A deterministic type identifier derived from a type's fully-qualified path.
///
/// Unlike [`core::any::TypeId`], a `StableTypeId` is reproducible across builds
/// and platforms because it depends only on the type path string, so it is safe
/// to persist in save files and network streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StableTypeId(u64);

impl StableTypeId {
    /// Compute the stable id of an explicit type **path** string.
    ///
    /// This is the canonical constructor: identity is defined entirely by the
    /// path bytes, hashed with the frozen FNV-1a seed/prime. It is a `const fn`
    /// so ids can be materialised at compile time.
    #[must_use]
    pub const fn of_path(path: &str) -> Self {
        let bytes = path.as_bytes();
        let mut hash = FNV_OFFSET_BASIS;
        let mut i = 0;
        while i < bytes.len() {
            hash ^= bytes[i] as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            i += 1;
        }
        Self(hash)
    }

    /// Compute the stable id of the type `T` from its
    /// [`core::any::type_name`].
    ///
    /// Two distinct types never share a path, so their ids differ (modulo the
    /// astronomically unlikely 64-bit hash collision); the same type always
    /// yields the same id across builds.
    #[must_use]
    pub fn of_type<T: ?Sized + 'static>() -> Self {
        Self::of_path(::core::any::type_name::<T>())
    }

    /// Reconstruct a stable id from its raw 64-bit value (e.g. read back from a
    /// stream).
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw 64-bit hash value, as written to the wire.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

impl ::core::fmt::Display for StableTypeId {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        write!(f, "StableTypeId({:#018x})", self.0)
    }
}
