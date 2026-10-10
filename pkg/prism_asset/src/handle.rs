//! Reference-counted asset handles.

use crate::asset::Asset;
use crate::guid::StableGuid;
use crate::id::{AssetId, UntypedAssetId};
use crate::type_id::AssetTypeId;
use alloc::sync::{Arc, Weak};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);

/// A process-unique identifier for a distinct handle allocation.
///
/// Every [`Assets::insert`](crate::Assets::insert) mints one `HandleId`;
/// cloning a [`Handle`] shares it. Two handles with the same `HandleId` are
/// clones of one another, whereas two independent handles to the same asset
/// share an [`AssetId`] but have different `HandleId`s.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct HandleId(u64);

impl HandleId {
    /// The raw integer value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A thread-safe abandonment signal shared between an [`Assets`](crate::Assets)
/// arena and every strong handle it mints.
///
/// Deferred reclaim (design §6.2) must never touch arena storage from a
/// handle's `Drop`: the handle may drop on any thread (for example a render
/// world releasing a GPU-resident asset) while the arena is owned by another.
/// Instead, the last strong handle to drop bumps a monotonic counter here, and
/// the arena consults it at a deterministic reclaim point
/// ([`Assets::collect_releases`](crate::Assets::collect_releases)) to decide
/// whether a reclaim scan is even needed.
///
/// This is a `no_std`, allocation-free, lock-free, and `unsafe`-free design: a
/// single [`AtomicU64`] epoch replaces the lock-based MPSC queue a `std` build
/// could use, trading per-id delivery for a coalesced "something was abandoned
/// since you last looked" edge that the arena turns into one batched scan.
pub(crate) struct ReleaseSignal {
    dropped: AtomicU64,
}

impl ReleaseSignal {
    /// Creates a signal with a zero drop count.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            dropped: AtomicU64::new(0),
        })
    }

    /// Records that one strong handle's last reference was dropped. Called from
    /// [`HandleInner`]'s `Drop`; uses `Release` so a later `Acquire` load in the
    /// arena observes every preceding handle mutation.
    fn mark_dropped(&self) {
        self.dropped.fetch_add(1, AtomicOrdering::Release);
    }

    /// The monotonic count of abandoned strong handles observed so far.
    pub(crate) fn dropped_count(&self) -> u64 {
        self.dropped.load(AtomicOrdering::Acquire)
    }
}

/// Shared payload behind every strong, weak, and untyped handle to one asset.
pub(crate) struct HandleInner {
    id: UntypedAssetId,
    handle_id: HandleId,
    release: Arc<ReleaseSignal>,
}

impl HandleInner {
    /// Allocates a fresh shared payload with a unique [`HandleId`], wired to the
    /// minting arena's [`ReleaseSignal`] so its drop is observable.
    pub(crate) fn new_arc(id: UntypedAssetId, release: Arc<ReleaseSignal>) -> Arc<Self> {
        let handle_id = HandleId(NEXT_HANDLE_ID.fetch_add(1, AtomicOrdering::Relaxed));
        Arc::new(Self {
            id,
            handle_id,
            release,
        })
    }
}

impl Drop for HandleInner {
    fn drop(&mut self) {
        // Runs when the last *strong* handle drops: the arena and any weak
        // handles hold only `Weak<HandleInner>`, so `Arc` runs this destructor
        // as soon as the strong count hits zero (the allocation itself lingers
        // until the weak count also drains). We only signal; the arena reclaims
        // the slot later at its own deterministic point (design §6.2).
        self.release.mark_dropped();
    }
}

/// A strong, reference-counted handle that keeps its asset alive.
pub struct Handle<A: ?Sized> {
    inner: Arc<HandleInner>,
    marker: PhantomData<fn() -> A>,
}

impl<A: ?Sized> Handle<A> {
    pub(crate) fn from_arc(inner: Arc<HandleInner>) -> Self {
        Self {
            inner,
            marker: PhantomData,
        }
    }

    /// The typed id this handle points at.
    ///
    /// A live [`Handle<A>`] is only ever created for an allocation minted as
    /// `A`, so re-applying the tag is sound; we use the unchecked re-type
    /// because `A` may be `?Sized` and cannot run the [`Asset`]-bounded check.
    #[must_use]
    pub fn id(&self) -> AssetId<A> {
        self.inner.id.typed_unchecked()
    }

    /// The type-erased id this handle points at.
    #[must_use]
    pub fn untyped_id(&self) -> UntypedAssetId {
        self.inner.id
    }

    /// This handle's allocation identity (shared by clones).
    #[must_use]
    pub fn handle_id(&self) -> HandleId {
        self.inner.handle_id
    }

    /// The number of live strong handles to this asset allocation.
    #[must_use]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Produces a non-owning [`WeakHandle`] to the same asset.
    #[must_use]
    pub fn downgrade(&self) -> WeakHandle<A> {
        WeakHandle {
            inner: Arc::downgrade(&self.inner),
            marker: PhantomData,
        }
    }

    /// Erases the asset type while keeping the strong reference.
    #[must_use]
    pub fn untyped(&self) -> UntypedHandle {
        UntypedHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<A: ?Sized> Clone for Handle<A> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            marker: PhantomData,
        }
    }
}

impl<A: ?Sized> PartialEq for Handle<A> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.id == other.inner.id
    }
}

impl<A: ?Sized> Eq for Handle<A> {}

impl<A: ?Sized> PartialOrd for Handle<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: ?Sized> Ord for Handle<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.id.cmp(&other.inner.id)
    }
}

impl<A: ?Sized> Hash for Handle<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.id.hash(state);
    }
}

impl<A: ?Sized> fmt::Debug for Handle<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("id", &self.inner.id)
            .field("handle_id", &self.inner.handle_id)
            .field("strong_count", &self.strong_count())
            .finish()
    }
}

/// A non-owning handle that does not keep its asset alive.
pub struct WeakHandle<A: ?Sized> {
    inner: Weak<HandleInner>,
    marker: PhantomData<fn() -> A>,
}

impl<A: ?Sized> WeakHandle<A> {
    /// Attempts to upgrade to a strong [`Handle`], succeeding only while at
    /// least one strong handle still exists.
    #[must_use]
    pub fn upgrade(&self) -> Option<Handle<A>> {
        self.inner.upgrade().map(Handle::from_arc)
    }

    /// The number of live strong handles (0 once the asset is reclaimable).
    #[must_use]
    pub fn strong_count(&self) -> usize {
        self.inner.strong_count()
    }
}

impl<A: ?Sized> Clone for WeakHandle<A> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            marker: PhantomData,
        }
    }
}

impl<A: ?Sized> fmt::Debug for WeakHandle<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeakHandle")
            .field("strong_count", &self.strong_count())
            .finish()
    }
}

/// A type-erased strong handle.
#[derive(Clone)]
pub struct UntypedHandle {
    inner: Arc<HandleInner>,
}

impl UntypedHandle {
    /// The type-erased id this handle points at.
    #[must_use]
    pub fn id(&self) -> UntypedAssetId {
        self.inner.id
    }

    /// This handle's allocation identity.
    #[must_use]
    pub fn handle_id(&self) -> HandleId {
        self.inner.handle_id
    }

    /// The number of live strong handles to this asset allocation.
    #[must_use]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Re-applies the compile-time type tag `A`, yielding a typed [`Handle`]
    /// that shares this handle's strong reference, but only if `A` matches the
    /// type the underlying id was minted for; otherwise returns `None`. This is
    /// the type-safe erasure boundary for handles (design §20).
    #[must_use]
    pub fn typed<A: Asset>(&self) -> Option<Handle<A>> {
        if self.inner.id.is::<A>() {
            Some(Handle::from_arc(Arc::clone(&self.inner)))
        } else {
            None
        }
    }

    /// Re-applies a type tag without checking it, for callers that already hold
    /// a type invariant for this handle. Prefer [`UntypedHandle::typed`] at any
    /// boundary where the type is not already guaranteed.
    #[must_use]
    pub fn typed_unchecked<A: ?Sized>(&self) -> Handle<A> {
        Handle::from_arc(Arc::clone(&self.inner))
    }
}

impl PartialEq for UntypedHandle {
    fn eq(&self, other: &Self) -> bool {
        self.inner.id == other.inner.id
    }
}

impl Eq for UntypedHandle {}

impl Hash for UntypedHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.id.hash(state);
    }
}

impl fmt::Debug for UntypedHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UntypedHandle")
            .field("id", &self.inner.id)
            .field("handle_id", &self.inner.handle_id)
            .finish()
    }
}

/// A persistent, non-owning reference to an asset by its [`StableGuid`].
///
/// Where [`Handle`] is a *runtime* strong reference (it keeps the asset alive
/// and knows its live [`AssetIndex`](crate::AssetIndex)), a `SoftHandle` is a
/// *persistent* reference that stores only the asset's stable identity. It does
/// **not** force the target to load and does **not** keep it resident, so a
/// scene can hold millions of soft references (per design §23.2) at the cost of
/// a 16-byte guid plus an 8-byte type tag, resolving them to real handles lazily
/// through the loader (a later milestone) only when something is actually used.
///
/// Carrying the [`AssetTypeId`] alongside the guid lets a resolver reject a
/// cross-type reference (a `SoftHandle<Mesh>` pointed at a texture guid) instead
/// of silently loading the wrong store.
pub struct SoftHandle<A: ?Sized> {
    guid: StableGuid,
    type_id: AssetTypeId,
    marker: PhantomData<fn() -> A>,
}

impl<A: ?Sized> SoftHandle<A> {
    /// Creates a soft reference to the asset with the given stable identity and
    /// type tag.
    #[must_use]
    pub const fn new(guid: StableGuid, type_id: AssetTypeId) -> Self {
        Self {
            guid,
            type_id,
            marker: PhantomData,
        }
    }

    /// The reserved null soft handle, referring to no asset.
    #[must_use]
    pub const fn null(type_id: AssetTypeId) -> Self {
        Self::new(StableGuid::NIL, type_id)
    }

    /// The persistent identity this soft handle points at.
    #[must_use]
    pub const fn guid(&self) -> StableGuid {
        self.guid
    }

    /// The declared asset type of the target.
    #[must_use]
    pub const fn type_id(&self) -> AssetTypeId {
        self.type_id
    }

    /// Whether this soft handle refers to no asset
    /// ([`StableGuid::NIL`](crate::StableGuid::NIL)).
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.guid.is_nil()
    }
}

impl<A: Asset> SoftHandle<A> {
    /// Creates a soft reference to the asset at `path`, deriving both its stable
    /// guid and its type tag from the typed target `A`.
    #[must_use]
    pub fn from_path(path: &str) -> Self {
        Self::new(StableGuid::from_path(path), AssetTypeId::of::<A>())
    }
}

// Manual impls: `PhantomData<fn() -> A>` means these never actually need `A` to
// implement the trait, but `derive` would wrongly add that bound.
impl<A: ?Sized> Clone for SoftHandle<A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<A: ?Sized> Copy for SoftHandle<A> {}
impl<A: ?Sized> PartialEq for SoftHandle<A> {
    fn eq(&self, other: &Self) -> bool {
        self.guid == other.guid && self.type_id == other.type_id
    }
}
impl<A: ?Sized> Eq for SoftHandle<A> {}
impl<A: ?Sized> Hash for SoftHandle<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.guid.hash(state);
        self.type_id.hash(state);
    }
}
impl<A: ?Sized> fmt::Debug for SoftHandle<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SoftHandle")
            .field("guid", &self.guid)
            .field("type_id", &self.type_id)
            .finish()
    }
}
