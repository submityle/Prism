//! Stable, type-erased asset type identity.
//!
//! The runtime stores one arena per asset type and erases types at the handle
//! boundary ([`UntypedHandle`](crate::UntypedHandle),
//! [`UntypedAssetId`](crate::UntypedAssetId)). To route an erased id back to
//! the right arena — and to reject a mismatched `typed::<B>()` instead of
//! silently reading the wrong store — each asset type carries an
//! [`AssetTypeId`].
//!
//! Unlike [`core::any::TypeId`], an `AssetTypeId` is derived from a **stable
//! type name** declared by the asset ([`Asset::TYPE_NAME`](crate::Asset)), so
//! it is identical across runs, builds, and platforms. That makes it safe to
//! persist in containers and content catalogs, which `TypeId` is not.

use crate::hash::fnv1a_64;

/// A stable 64-bit identity for an asset type.
///
/// Derived from a type's declared stable name via the kernel's frozen FNV-1a
/// 64-bit hash ([`crate::hash`]), so it reproduces exactly across runs and
/// platforms. Treat the derivation as a frozen contract: renaming a type's
/// [`Asset::TYPE_NAME`](crate::Asset) changes its id and must go through a
/// migration like any other persisted identity.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct AssetTypeId(u64);

impl AssetTypeId {
    /// Derives the type id from a stable type name.
    #[must_use]
    pub fn of_name(name: &str) -> Self {
        Self(fnv1a_64(name.as_bytes()))
    }

    /// Derives the type id for an [`Asset`](crate::Asset) implementor.
    #[must_use]
    pub fn of<A: crate::Asset>() -> Self {
        Self::of_name(A::TYPE_NAME)
    }

    /// The raw 64-bit value, for serialization.
    #[must_use]
    pub const fn to_u64(self) -> u64 {
        self.0
    }

    /// Reconstructs a type id from a raw 64-bit value read back from storage.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }
}
