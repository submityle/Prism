//! Stable asset identity: generational slot ids and typed/untyped asset ids.

use crate::asset::Asset;
use crate::type_id::AssetTypeId;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

/// A generational handle into an [`Assets`](crate::Assets) arena.
///
/// The `index` selects a slot; the `generation` disambiguates reuse. When a
/// slot is freed and later refilled its generation is incremented, so an
/// [`AssetIndex`] captured before the free compares unequal to the new
/// occupant and will not accidentally resolve to it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct AssetIndex {
    index: u32,
    generation: u32,
}

impl AssetIndex {
    /// Creates an index from raw parts. Normally produced by
    /// [`Assets`](crate::Assets); exposed for serialization round-trips.
    #[must_use]
    pub const fn from_parts(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// The slot position within the arena.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The reuse counter for this slot.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// A typed asset id: an [`AssetIndex`] tagged with the asset type `A`.
///
/// The type tag is zero-cost (`PhantomData<fn() -> A>`), so `AssetId<A>` is
/// `Copy`, `Send`, and `Sync` regardless of `A`'s bounds and never owns an `A`.
pub struct AssetId<A: ?Sized> {
    index: AssetIndex,
    marker: PhantomData<fn() -> A>,
}

impl<A: ?Sized> AssetId<A> {
    /// Wraps an [`AssetIndex`] as a typed id.
    #[must_use]
    pub const fn new(index: AssetIndex) -> Self {
        Self {
            index,
            marker: PhantomData,
        }
    }

    /// The underlying generational slot id.
    #[must_use]
    pub const fn index(self) -> AssetIndex {
        self.index
    }
}

impl<A: Asset> AssetId<A> {
    /// Erases the compile-time type tag into a runtime [`AssetTypeId`], so the
    /// resulting [`UntypedAssetId`] can later be re-typed with a checked
    /// [`UntypedAssetId::typed`] that rejects a cross-type mismatch (design
    /// §20). Not `const` because the type id is a runtime hash of
    /// [`Asset::TYPE_NAME`].
    #[must_use]
    pub fn untyped(self) -> UntypedAssetId {
        UntypedAssetId {
            index: self.index,
            type_id: AssetTypeId::of::<A>(),
        }
    }
}

// Manual impls: deriving would wrongly require `A: Trait` even though the tag
// is a function pointer and carries no `A`.
impl<A: ?Sized> Clone for AssetId<A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<A: ?Sized> Copy for AssetId<A> {}
impl<A: ?Sized> PartialEq for AssetId<A> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl<A: ?Sized> Eq for AssetId<A> {}
impl<A: ?Sized> PartialOrd for AssetId<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<A: ?Sized> Ord for AssetId<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index.cmp(&other.index)
    }
}
impl<A: ?Sized> Hash for AssetId<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}
impl<A: ?Sized> fmt::Debug for AssetId<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AssetId").field(&self.index).finish()
    }
}

/// A type-erased asset id, used by untyped handles and the dependency graph.
///
/// Unlike [`AssetId`], it carries the asset's [`AssetTypeId`] *as data* so the
/// kernel can store heterogeneous ids together (handles, the dependency graph,
/// soft references) and still re-type them safely: [`UntypedAssetId::typed`]
/// returns `None` when the requested type does not match the one the id was
/// minted for, turning a silent wrong-store lookup into a diagnosable miss
/// (design §20). Equality, ordering, and hashing all include the type id, so a
/// slot index reused across asset types never collides.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct UntypedAssetId {
    index: AssetIndex,
    type_id: AssetTypeId,
}

impl UntypedAssetId {
    /// Wraps an [`AssetIndex`] together with the asset's [`AssetTypeId`].
    #[must_use]
    pub const fn new(index: AssetIndex, type_id: AssetTypeId) -> Self {
        Self { index, type_id }
    }

    /// The underlying generational slot id.
    #[must_use]
    pub const fn index(self) -> AssetIndex {
        self.index
    }

    /// The asset type this id was minted for.
    #[must_use]
    pub const fn type_id(self) -> AssetTypeId {
        self.type_id
    }

    /// Whether this id was minted for asset type `A`.
    #[must_use]
    pub fn is<A: Asset>(self) -> bool {
        self.type_id == AssetTypeId::of::<A>()
    }

    /// Re-applies the compile-time type tag `A`, but only if it matches the
    /// type this id was minted for; otherwise returns `None`. This is the
    /// type-safe erasure boundary of the kernel (design §20): a
    /// `UntypedAssetId` for a texture cannot be silently viewed as a mesh id.
    #[must_use]
    pub fn typed<A: Asset>(self) -> Option<AssetId<A>> {
        if self.is::<A>() {
            Some(AssetId::new(self.index))
        } else {
            None
        }
    }

    /// Re-applies a type tag without checking it, for callers that already hold
    /// a type invariant for this id (for example a typed [`Handle`] re-reading
    /// its own id). Prefer [`UntypedAssetId::typed`] at any boundary where the
    /// type is not already guaranteed.
    #[must_use]
    pub const fn typed_unchecked<A: ?Sized>(self) -> AssetId<A> {
        AssetId::new(self.index)
    }
}

impl<A: Asset> From<AssetId<A>> for UntypedAssetId {
    fn from(id: AssetId<A>) -> Self {
        id.untyped()
    }
}
