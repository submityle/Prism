//! Stable asset identity: generational slot ids and typed/untyped asset ids.

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

    /// Erases the type tag.
    #[must_use]
    pub const fn untyped(self) -> UntypedAssetId {
        UntypedAssetId { index: self.index }
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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct UntypedAssetId {
    index: AssetIndex,
}

impl UntypedAssetId {
    /// Wraps an [`AssetIndex`] without a type tag.
    #[must_use]
    pub const fn new(index: AssetIndex) -> Self {
        Self { index }
    }

    /// The underlying generational slot id.
    #[must_use]
    pub const fn index(self) -> AssetIndex {
        self.index
    }

    /// Re-applies a type tag. The caller asserts the id was minted for `A`;
    /// this is a pure value re-wrap and cannot cause memory unsafety, but
    /// resolving a mismatched id in [`Assets`](crate::Assets) simply misses.
    #[must_use]
    pub const fn typed<A: ?Sized>(self) -> AssetId<A> {
        AssetId::new(self.index)
    }
}

impl<A: ?Sized> From<AssetId<A>> for UntypedAssetId {
    fn from(id: AssetId<A>) -> Self {
        id.untyped()
    }
}
