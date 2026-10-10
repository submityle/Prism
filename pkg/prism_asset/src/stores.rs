//! Type-erased registry of per-type [`Assets<A>`] arenas (design §7).
//!
//! The kernel stores one arena per asset type so that `Assets<Mesh>` and
//! `Assets<Image>` never share a slot namespace. The [`AssetServer`] and the
//! load pipeline, however, work with [`UntypedAssetId`]s and
//! [`ErasedLoadedAsset`](crate::ErasedLoadedAsset) values that are only known
//! by [`AssetTypeId`] at runtime. [`AssetStores`] bridges the two: it owns a
//! `AssetTypeId → Box<dyn ErasedAssetStore>` table and routes an erased id or a
//! boxed value to the correct arena, downcasting at exactly one audited
//! boundary.
//!
//! ## Why one erased boundary
//! Erasing types is where a wrong-store bug hides (a texture value written into
//! the mesh arena). Every entry point here re-checks the type: routing an
//! [`UntypedAssetId`] uses its embedded [`AssetTypeId`]; inserting a boxed value
//! downcasts to the arena's concrete type and returns [`StoreError::TypeMismatch`]
//! instead of silently mis-storing. That turns the classic erased-storage bug
//! into a diagnosable error (design §20).
//!
//! This module is `no_std + alloc`: it needs only [`core::any::Any`] and the
//! `alloc` collections, so the registry is available in headless and `no_std`
//! builds and reused by the std-gated server.

use crate::error::AssetErrorId;
use crate::id::{AssetId, UntypedAssetId};
use crate::load_state::LoadState;
use crate::storage::Assets;
use crate::type_id::AssetTypeId;
use crate::{Asset, UntypedHandle};
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use core::any::Any;
use core::fmt;

/// Why a type-erased store operation could not be routed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StoreError {
    /// No arena is registered for the requested asset type. Register it with
    /// [`AssetStores::register`] before loading assets of that type.
    UnknownType(AssetTypeId),
    /// A boxed value or id did not match the arena's concrete type. `expected`
    /// is the arena's type; the actual type is erased and cannot be named.
    TypeMismatch {
        /// The asset type the targeted arena stores.
        expected: AssetTypeId,
    },
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::UnknownType(t) => write!(f, "no asset store registered for {t:?}"),
            StoreError::TypeMismatch { expected } => {
                write!(f, "value does not match store type {expected:?}")
            }
        }
    }
}

/// The object-safe, type-erased view of one [`Assets<A>`] arena.
///
/// Implemented once, generically, for every `Assets<A>`; the registry stores
/// these as trait objects keyed by [`AssetTypeId`]. Methods that take or return
/// values use `Box<dyn Any + Send>` (the same shape the loader produces) and
/// re-check the type at the boundary.
pub trait ErasedAssetStore: Any + Send + Sync {
    /// The asset type this arena stores.
    fn asset_type(&self) -> AssetTypeId;

    /// Reserves an ahead-of-time slot, returning a strong [`UntypedHandle`]
    /// that keeps the slot alive until the load fulfills or fails it.
    fn reserve_erased(&mut self) -> UntypedHandle;

    /// Inserts a ready value, returning its strong handle.
    ///
    /// # Errors
    /// [`StoreError::TypeMismatch`] if `value` is not this arena's type.
    fn insert_erased(&mut self, value: Box<dyn Any + Send>) -> Result<UntypedHandle, StoreError>;

    /// Fulfills a previously reserved slot with a decoded value. Returns
    /// whether the id was live (an `Ok(false)` means the id was stale).
    ///
    /// # Errors
    /// [`StoreError::TypeMismatch`] if `value` or `id` is not this arena's type.
    fn fulfill_erased(
        &mut self,
        id: UntypedAssetId,
        value: Box<dyn Any + Send>,
    ) -> Result<bool, StoreError>;

    /// Marks a reserved slot failed. Returns whether the id was live.
    fn fail_erased(&mut self, id: UntypedAssetId, error: AssetErrorId) -> bool;

    /// The load state of `id` within this arena ([`LoadState::NotLoaded`] if the
    /// id does not belong to this arena's type or is stale).
    fn load_state_erased(&self, id: UntypedAssetId) -> LoadState;

    /// Whether `id` resolves to a live slot in this arena.
    fn contains_erased(&self, id: UntypedAssetId) -> bool;

    /// Reclaims slots whose last strong handle has dropped. Returns the count.
    fn remove_unused(&mut self) -> usize;

    /// Advances the arena's grace frame and reclaims slots past their grace
    /// window. Returns the count reclaimed this call.
    fn collect_releases(&mut self) -> usize;

    /// The number of live slots.
    fn len(&self) -> usize;

    /// Whether the arena has no live slots.
    fn is_empty(&self) -> bool;

    /// Upcast for downcasting the trait object back to its `Assets<A>`.
    fn as_any(&self) -> &dyn Any;

    /// Mutable upcast for downcasting the trait object back to its `Assets<A>`.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<A: Asset> ErasedAssetStore for Assets<A> {
    fn asset_type(&self) -> AssetTypeId {
        AssetTypeId::of::<A>()
    }

    fn reserve_erased(&mut self) -> UntypedHandle {
        self.reserve().untyped()
    }

    fn insert_erased(&mut self, value: Box<dyn Any + Send>) -> Result<UntypedHandle, StoreError> {
        match value.downcast::<A>() {
            Ok(v) => Ok(self.insert(*v).untyped()),
            Err(_) => Err(StoreError::TypeMismatch {
                expected: AssetTypeId::of::<A>(),
            }),
        }
    }

    fn fulfill_erased(
        &mut self,
        id: UntypedAssetId,
        value: Box<dyn Any + Send>,
    ) -> Result<bool, StoreError> {
        let typed_id: AssetId<A> = id.typed::<A>().ok_or(StoreError::TypeMismatch {
            expected: AssetTypeId::of::<A>(),
        })?;
        let v = value
            .downcast::<A>()
            .map_err(|_| StoreError::TypeMismatch {
                expected: AssetTypeId::of::<A>(),
            })?;
        Ok(self.fulfill(typed_id, *v))
    }

    fn fail_erased(&mut self, id: UntypedAssetId, error: AssetErrorId) -> bool {
        match id.typed::<A>() {
            Some(typed_id) => self.fail(typed_id, error),
            None => false,
        }
    }

    fn load_state_erased(&self, id: UntypedAssetId) -> LoadState {
        match id.typed::<A>() {
            Some(typed_id) => self.load_state(typed_id),
            None => LoadState::NotLoaded,
        }
    }

    fn contains_erased(&self, id: UntypedAssetId) -> bool {
        match id.typed::<A>() {
            Some(typed_id) => self.contains(typed_id),
            None => false,
        }
    }

    fn remove_unused(&mut self) -> usize {
        Assets::<A>::remove_unused(self)
    }

    fn collect_releases(&mut self) -> usize {
        Assets::<A>::collect_releases(self)
    }

    fn len(&self) -> usize {
        Assets::<A>::len(self)
    }

    fn is_empty(&self) -> bool {
        Assets::<A>::is_empty(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A registry of per-type [`Assets<A>`] arenas, keyed by [`AssetTypeId`].
///
/// Owns the heterogeneous storage the server routes erased loads into. Lookups
/// are deterministic (a [`BTreeMap`] ordered by type id) so iteration order is
/// stable across runs (design §16).
#[derive(Default)]
pub struct AssetStores {
    stores: BTreeMap<AssetTypeId, Box<dyn ErasedAssetStore>>,
}

impl AssetStores {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a fresh arena for asset type `A`. Returns `true` if it was
    /// newly created, `false` if an arena for `A` already existed (left intact).
    pub fn register<A: Asset>(&mut self) -> bool {
        let key = AssetTypeId::of::<A>();
        if self.stores.contains_key(&key) {
            return false;
        }
        self.stores.insert(key, Box::new(Assets::<A>::new()));
        true
    }

    /// Whether an arena is registered for asset type `A`.
    #[must_use]
    pub fn contains_type_of<A: Asset>(&self) -> bool {
        self.stores.contains_key(&AssetTypeId::of::<A>())
    }

    /// Whether an arena is registered for the runtime `type_id`.
    #[must_use]
    pub fn contains_type(&self, type_id: AssetTypeId) -> bool {
        self.stores.contains_key(&type_id)
    }

    /// The number of registered arenas.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stores.len()
    }

    /// Whether no arenas are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stores.is_empty()
    }

    /// A shared reference to the typed arena for `A`, if registered.
    #[must_use]
    pub fn arena<A: Asset>(&self) -> Option<&Assets<A>> {
        self.stores
            .get(&AssetTypeId::of::<A>())
            .and_then(|s| s.as_any().downcast_ref::<Assets<A>>())
    }

    /// A mutable reference to the typed arena for `A`, if registered.
    #[must_use]
    pub fn arena_mut<A: Asset>(&mut self) -> Option<&mut Assets<A>> {
        self.stores
            .get_mut(&AssetTypeId::of::<A>())
            .and_then(|s| s.as_any_mut().downcast_mut::<Assets<A>>())
    }

    /// Reads a value by typed id, if its arena is registered and the slot is
    /// ready.
    #[must_use]
    pub fn get<A: Asset>(&self, id: AssetId<A>) -> Option<&A> {
        self.arena::<A>().and_then(|a| a.get(id))
    }

    /// Eagerly inserts a ready, statically-typed value into `A`'s arena,
    /// auto-registering the arena if needed. Returns a strong
    /// [`UntypedHandle`] holding the slot alive.
    ///
    /// This is the type-safe counterpart to [`AssetStores::insert_erased`]:
    /// because the type is known at the call site no routing can fail, so it
    /// cannot return [`StoreError`]. Prefer it when the caller already holds a
    /// concrete `A` (eager/synchronous inserts); use `insert_erased` only on
    /// the type-erased loader boundary.
    pub fn insert<A: Asset>(&mut self, value: A) -> UntypedHandle {
        self.register::<A>();
        self.arena_mut::<A>()
            .expect("arena registered above")
            .insert(value)
            .untyped()
    }

    /// Reserves an ahead-of-time slot in `A`'s arena, auto-registering the arena
    /// if needed. Returns a strong [`UntypedHandle`] holding the slot alive.
    pub fn reserve<A: Asset>(&mut self) -> UntypedHandle {
        self.register::<A>();
        self.stores
            .get_mut(&AssetTypeId::of::<A>())
            .expect("arena registered above")
            .reserve_erased()
    }

    /// Inserts a ready, type-erased value, routing it to its arena by
    /// `type_id` and auto-registering nothing (the arena must exist).
    ///
    /// # Errors
    /// [`StoreError::UnknownType`] if no arena is registered for `type_id`;
    /// [`StoreError::TypeMismatch`] if `value` is not that arena's type.
    pub fn insert_erased(
        &mut self,
        type_id: AssetTypeId,
        value: Box<dyn Any + Send>,
    ) -> Result<UntypedHandle, StoreError> {
        self.stores
            .get_mut(&type_id)
            .ok_or(StoreError::UnknownType(type_id))?
            .insert_erased(value)
    }

    /// Fulfills a reserved slot with a type-erased value, routed by the id's
    /// embedded type.
    ///
    /// # Errors
    /// [`StoreError::UnknownType`] / [`StoreError::TypeMismatch`] as above.
    pub fn fulfill_erased(
        &mut self,
        id: UntypedAssetId,
        value: Box<dyn Any + Send>,
    ) -> Result<bool, StoreError> {
        self.stores
            .get_mut(&id.type_id())
            .ok_or(StoreError::UnknownType(id.type_id()))?
            .fulfill_erased(id, value)
    }

    /// Marks a reserved slot failed, routed by the id's embedded type. Returns
    /// whether the id was live; `false` also if its arena is unregistered.
    pub fn fail_erased(&mut self, id: UntypedAssetId, error: AssetErrorId) -> bool {
        match self.stores.get_mut(&id.type_id()) {
            Some(s) => s.fail_erased(id, error),
            None => false,
        }
    }

    /// The load state of `id`, routed by its embedded type. Returns
    /// [`LoadState::NotLoaded`] if its arena is unregistered or the slot stale.
    #[must_use]
    pub fn load_state(&self, id: UntypedAssetId) -> LoadState {
        match self.stores.get(&id.type_id()) {
            Some(s) => s.load_state_erased(id),
            None => LoadState::NotLoaded,
        }
    }

    /// Whether `id` resolves to a live slot (routed by its embedded type).
    #[must_use]
    pub fn contains(&self, id: UntypedAssetId) -> bool {
        match self.stores.get(&id.type_id()) {
            Some(s) => s.contains_erased(id),
            None => false,
        }
    }

    /// Reclaims abandoned slots across every arena. Returns the total count.
    pub fn remove_unused(&mut self) -> usize {
        self.stores.values_mut().map(|s| s.remove_unused()).sum()
    }

    /// Advances every arena's grace frame and reclaims expired slots. Returns
    /// the total count reclaimed this call.
    pub fn collect_releases(&mut self) -> usize {
        self.stores.values_mut().map(|s| s.collect_releases()).sum()
    }

    /// The total number of live slots across all arenas.
    #[must_use]
    pub fn total_assets(&self) -> usize {
        self.stores.values().map(|s| s.len()).sum()
    }
}
