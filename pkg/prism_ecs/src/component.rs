//! Component types, their stable [`ComponentId`]s, and the registry that tracks
//! per-type metadata (layout, drop glue, storage strategy).
//!
//! A [`Component`] is any `Send + Sync + 'static` type that is stored on
//! entities. Each distinct component type registered in a [`World`] is assigned
//! a dense [`ComponentId`] on first use; the registry records the runtime
//! [`Layout`] and type-erased drop function needed by the columnar storage.
//!
//! [`World`]: crate::world::World

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::alloc::Layout;
use core::any::{type_name, TypeId};

use crate::collections::HashMap;
use crate::component_hooks::ComponentHooks;
use crate::storage::SharedValue;

/// A type that can be stored on entities as a component.
///
/// This is a marker trait; it is implemented via `#[derive(Component)]` in the
/// companion macro crate (future `prism_ecs_macros`), or manually for a plain
/// data type. The `Send + Sync + 'static` bounds make components safe to store
/// in type-erased columns and to iterate across threads.
pub trait Component: Send + Sync + 'static {
    /// The storage strategy for this component type. Defaults to columnar
    /// [`StorageType::Table`]; override to [`StorageType::SparseSet`] for
    /// components that are added/removed extremely frequently (design §6).
    const STORAGE: StorageType = StorageType::Table;

    /// Install any storage-specific glue for this component into `components`
    /// immediately after it is first registered under `id` (design §6). The
    /// default is a no-op; shared components (whose `STORAGE` is
    /// [`StorageType::Shared`]) override this to register their
    /// [`SharedBoxFn`] value-boxing glue via [`Components::set_shared_box`] so
    /// the structural code can intern their values. Called exactly once, in the
    /// new-id branch of [`Components::register`].
    #[inline]
    fn install_storage_glue(_components: &mut Components, _id: ComponentId) {}
}

/// How a component type is physically stored.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum StorageType {
    /// Columnar Structure-of-Arrays within the archetype's table. Fastest to
    /// iterate and SIMD-friendly; this is the default and backs chunked,
    /// change-tracked, SIMD-iterable component columns (design §6).
    Table,
    /// A sparse set keyed by entity index, stored out-of-band in the world's
    /// [`SparseSets`](crate::storage::SparseSets) registry rather than in an
    /// archetype table. Insert/remove is O(1) and never moves the entity
    /// between archetypes, so it suits tags and components toggled extremely
    /// frequently (design §6 "增删不搬迁 archetype"). Spawn/insert/remove,
    /// `contains`, change detection, and every query path (fetch, iter, filter,
    /// `par_iter`) route sparse components through that registry.
    SparseSet,
    /// Unity-style shared component: the value is de-duplicated into a single
    /// interned copy and used as a *batch key* that splits the archetype, so
    /// every entity sharing one value is grouped into the same archetype
    /// variant (design §6, §15 GPU 批次键). The value is immutable through
    /// queries — it is a key, not per-entity data — so only shared `&T` /
    /// `Option<&T>` reads and `With`/`Without` membership are available;
    /// mutable or change-tracked access (`&mut T`, `Ref`, `Added`, `Changed`)
    /// is rejected at query construction. Changing an entity's shared value is
    /// a structural move to the archetype variant for the new value.
    Shared,
}

/// A dense, stable identifier for a registered component type within one
/// [`World`](crate::world::World).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ComponentId(u32);

impl ComponentId {
    /// Construct a [`ComponentId`] from its raw index. Primarily for internal
    /// use and tests; ids are normally minted by [`Components::register`].
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The raw dense index backing this id.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Type-erased drop glue for a single component value.
///
/// # Safety
/// The pointer must point to a valid, initialized value of the component type
/// this function was created for, and the value must not be used afterwards.
pub type DropFn = unsafe fn(*mut u8);

/// Type-erased clone glue for snapshot capture/restore (design §14 / §16.5).
///
/// Clones the component value at `src` into the uninitialized destination
/// `dst`, producing an independently-owned copy.
///
/// # Safety
/// `src` must point at a valid, initialized value of this component type.
/// `dst` must point at writable, correctly-aligned memory of at least the
/// component [`Layout`]'s size, currently uninitialized; on return it holds a
/// fully-initialized, separately-owned clone that the caller takes ownership of
/// (and must eventually drop via the component's [`DropFn`]).
pub type CloneFn = unsafe fn(src: *const u8, dst: *mut u8);

/// Type-erased deterministic hash glue for state hashing (design §14: 逐帧状态
/// 哈希去同步).
///
/// Folds the component value at `ptr` into `hasher` via the component type's
/// [`core::hash::Hash`] implementation, giving a build-stable contribution to a
/// world state hash.
///
/// # Safety
/// `ptr` must point at a valid, initialized value of this component type.
pub type SnapshotHashFn = unsafe fn(ptr: *const u8, hasher: &mut dyn core::hash::Hasher);

/// Type-erased constructor for a required component's default value (design
/// §16.1).
///
/// It produces exactly one owned value and hands a pointer to it to `out`,
/// following the same ownership contract as
/// [`Bundle::get_components`](crate::bundle::Bundle::get_components): the
/// callback takes ownership of the bytes, so the constructor must not also drop
/// the value. Wrapped in [`Arc`] so a single constructor can be shared across
/// every component that transitively requires it, and so [`ComponentInfo`] can
/// cheaply clone it into flattened closures.
pub type RequiredCtor = Arc<dyn Fn(&mut dyn FnMut(*mut u8)) + Send + Sync>;

/// Type-erased glue that moves a shared component value out of a bundle pointer
/// and boxes it as a [`SharedValue`] for interning (design §6 SharedComponent).
///
/// Built by [`shared_box_of`] and installed on a component's
/// [`ComponentInfo`] via [`Components::set_shared_box`]; the structural spawn /
/// insert paths call it to turn a just-written bundle pointer into an owned,
/// de-duplicatable value for the shared value pool.
///
/// # Safety
/// `ptr` must point at a valid, initialized value of the component type this
/// glue was created for; the value is moved out (read by value), so the source
/// must be treated as moved-from and never dropped again by the caller.
pub type SharedBoxFn = unsafe fn(ptr: *mut u8) -> Box<dyn SharedValue>;

/// One entry in a component's required-components set (design §16.1): the id of
/// a component to auto-insert whenever the requiring component is inserted, and
/// the constructor that supplies its default value.
#[derive(Clone)]
pub struct RequiredComponent {
    id: ComponentId,
    ctor: RequiredCtor,
}

impl RequiredComponent {
    /// The id of the required component.
    #[inline]
    pub fn id(&self) -> ComponentId {
        self.id
    }

    /// A fresh [`Arc`] handle to this entry's default-value constructor.
    #[inline]
    pub fn ctor(&self) -> RequiredCtor {
        Arc::clone(&self.ctor)
    }
}

/// Runtime metadata for one registered component type.
pub struct ComponentInfo {
    id: ComponentId,
    name: Cow<'static, str>,
    layout: Layout,
    storage: StorageType,
    type_id: Option<TypeId>,
    drop: Option<DropFn>,
    /// Type-erased clone glue, present once the component is registered for
    /// snapshotting via [`Components::register_cloneable`] (design §14/§16.5).
    clone: Option<CloneFn>,
    /// Type-erased deterministic hash glue, present once the component is
    /// registered via [`Components::register_hashable`] (design §14).
    snapshot_hash: Option<SnapshotHashFn>,
    hooks: ComponentHooks,
    /// Flattened (transitive, first-wins) required components (design §16.1).
    required: Vec<RequiredComponent>,
    /// Type-erased shared-value box glue, present once a shared component
    /// installs it (design §6 SharedComponent), otherwise `None`.
    shared_box: Option<SharedBoxFn>,
}

impl ComponentInfo {
    /// This component's id.
    #[inline]
    pub fn id(&self) -> ComponentId {
        self.id
    }

    /// Human-readable component name (the Rust type name for statically-typed
    /// components; a user-provided name for dynamically-registered ones).
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Memory layout of a single value of this component.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Storage strategy declared for this component.
    #[inline]
    pub fn storage(&self) -> StorageType {
        self.storage
    }

    /// The Rust [`TypeId`], when this component was registered from a concrete
    /// Rust type. Dynamically-registered components return `None`.
    #[inline]
    pub fn type_id(&self) -> Option<TypeId> {
        self.type_id
    }

    /// The type-erased drop function, or `None` for trivially-droppable
    /// (`needs_drop == false`) components.
    #[inline]
    pub fn drop_fn(&self) -> Option<DropFn> {
        self.drop
    }

    /// The type-erased clone function, present once this component has been
    /// registered for snapshotting (design §14/§16.5), otherwise `None`.
    #[inline]
    pub fn clone_fn(&self) -> Option<CloneFn> {
        self.clone
    }

    /// The type-erased deterministic hash function, present once this component
    /// has been registered as hashable (design §14), otherwise `None`.
    #[inline]
    pub fn snapshot_hash_fn(&self) -> Option<SnapshotHashFn> {
        self.snapshot_hash
    }

    /// The lifecycle [`ComponentHooks`] registered for this component (design
    /// §12). Empty unless hooks were attached via
    /// [`Components::set_hooks`] or
    /// [`World::register_component_hooks`](crate::world::World::register_component_hooks).
    #[inline]
    pub fn hooks(&self) -> &ComponentHooks {
        &self.hooks
    }

    /// The flattened, transitive set of components this component requires
    /// (design §16.1), in breadth-first "nearest requirer wins" order. Empty
    /// unless requirements were declared via
    /// [`Components::register_required`] /
    /// [`World::register_required_component`](crate::world::World::register_required_component).
    #[inline]
    pub fn required(&self) -> &[RequiredComponent] {
        &self.required
    }

    /// The type-erased shared-value box glue, present once a shared component
    /// has installed it via [`Component::install_storage_glue`] /
    /// [`Components::set_shared_box`] (design §6), otherwise `None`.
    #[inline]
    pub fn shared_box_fn(&self) -> Option<SharedBoxFn> {
        self.shared_box
    }
}

/// Build an [`unsafe`] drop function for `T`, or `None` if `T` needs no drop.
fn drop_fn_of<T>() -> Option<DropFn> {
    if core::mem::needs_drop::<T>() {
        /// Drop glue for `T` behind an erased pointer.
        ///
        /// # Safety
        /// `ptr` must point at a valid, initialized `T` that is not used
        /// afterwards, matching the [`DropFn`] contract.
        unsafe fn drop_ptr<T>(ptr: *mut u8) {
            // SAFETY: see the enclosing function's contract; `ptr` is a valid,
            // aligned, initialized `*mut T`.
            unsafe { core::ptr::drop_in_place(ptr.cast::<T>()) }
        }
        Some(drop_ptr::<T> as DropFn)
    } else {
        None
    }
}

/// Build type-erased clone glue for `T: Clone`.
fn clone_fn_of<T: Clone>() -> CloneFn {
    /// Clone `T` from `src` into uninitialized `dst`.
    ///
    /// # Safety
    /// Honors the [`CloneFn`] contract: `src` is a valid `&T`; `dst` is
    /// uninitialized, aligned storage for one `T` that this write initializes.
    unsafe fn clone_ptr<T: Clone>(src: *const u8, dst: *mut u8) {
        // SAFETY: `src` points at a valid, initialized `T` (caller contract).
        let value: T = unsafe { (*src.cast::<T>()).clone() };
        // SAFETY: `dst` is aligned, writable storage for one `T` and is
        // currently uninitialized, so a plain write (no drop of old) is correct
        // and transfers ownership of `value` into the destination.
        unsafe { dst.cast::<T>().write(value) }
    }
    clone_ptr::<T>
}

/// Build type-erased deterministic hash glue for `T: Hash`.
fn snapshot_hash_fn_of<T: core::hash::Hash>() -> SnapshotHashFn {
    /// Fold the `T` at `ptr` into `hasher`.
    ///
    /// # Safety
    /// Honors the [`SnapshotHashFn`] contract: `ptr` is a valid `&T`.
    unsafe fn hash_ptr<T: core::hash::Hash>(ptr: *const u8, hasher: &mut dyn core::hash::Hasher) {
        // SAFETY: `ptr` points at a valid, initialized `T` (caller contract).
        let value: &T = unsafe { &*ptr.cast::<T>() };
        value.hash(&mut HasherShim(hasher));
    }
    hash_ptr::<T>
}

/// Build type-erased [`SharedBoxFn`] glue for a shared component `T` (design
/// §6). Used by `#[derive(Component)]` with `storage = "shared"` and by manual
/// shared-component impls from [`Component::install_storage_glue`].
pub fn shared_box_of<T: Send + Sync + Eq + core::hash::Hash + 'static>() -> SharedBoxFn {
    /// Move the `T` at `ptr` out by value and box it as a `dyn SharedValue`.
    ///
    /// # Safety
    /// Honors the [`SharedBoxFn`] contract: `ptr` points at a valid,
    /// initialized `T` that is moved out and must not be used afterwards.
    unsafe fn box_ptr<T: Send + Sync + Eq + core::hash::Hash + 'static>(
        ptr: *mut u8,
    ) -> Box<dyn SharedValue> {
        // SAFETY: `ptr` points at a valid, initialized `T` (caller contract);
        // we read it out by value, taking ownership, and the caller treats the
        // source bytes as moved-from.
        let value: T = unsafe { ptr.cast::<T>().read() };
        Box::new(value)
    }
    box_ptr::<T>
}

/// Adapts a `&mut dyn Hasher` so `Hash::hash` (generic over `H: Hasher`) can
/// drive it; forwards every write through the trait object.
struct HasherShim<'a>(&'a mut dyn core::hash::Hasher);

impl core::hash::Hasher for HasherShim<'_> {
    #[inline]
    fn finish(&self) -> u64 {
        self.0.finish()
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}

/// The per-world registry of component metadata.
///
/// Assigns a dense [`ComponentId`] to each distinct component type the first
/// time it is seen, and remembers the mapping from Rust [`TypeId`] to
/// [`ComponentId`] so repeated lookups are cheap.
#[derive(Default)]
pub struct Components {
    infos: Vec<ComponentInfo>,
    by_type: HashMap<TypeId, ComponentId>,
    /// Set once any component gains a non-empty [`ComponentHooks`] set, letting
    /// the structural paths skip all hook bookkeeping with a single branch in
    /// the overwhelmingly common hook-free case (design §12).
    hooks_registered: bool,
    /// Directly-declared required-component edges: `requirer -> its immediate
    /// requirements` (design §16.1). The flattened transitive closure is cached
    /// on each [`ComponentInfo::required`] and rebuilt whenever an edge changes.
    direct_required: HashMap<ComponentId, Vec<RequiredComponent>>,
    /// Set once any required-component edge is declared, letting the structural
    /// paths skip required-component expansion in the common case.
    required_registered: bool,
}

impl Components {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            infos: Vec::new(),
            by_type: HashMap::default(),
            hooks_registered: false,
            direct_required: HashMap::default(),
            required_registered: false,
        }
    }

    /// Number of registered components.
    #[inline]
    pub fn len(&self) -> usize {
        self.infos.len()
    }

    /// Whether no components are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }

    /// Register the Rust component type `T` (idempotently) and return its id.
    pub fn register<T: Component>(&mut self) -> ComponentId {
        let type_id = TypeId::of::<T>();
        if let Some(&id) = self.by_type.get(&type_id) {
            return id;
        }
        let id = ComponentId(self.infos.len() as u32);
        self.infos.push(ComponentInfo {
            id,
            name: Cow::Borrowed(type_name::<T>()),
            layout: Layout::new::<T>(),
            storage: T::STORAGE,
            type_id: Some(type_id),
            drop: drop_fn_of::<T>(),
            clone: None,
            snapshot_hash: None,
            hooks: ComponentHooks::new(),
            required: Vec::new(),
            shared_box: None,
        });
        self.by_type.insert(type_id, id);
        T::install_storage_glue(self, id);
        id
    }

    /// Register `T` (idempotently) and attach type-erased clone glue so it can
    /// participate in world [`snapshot`](crate::world::World::snapshot) /
    /// [`restore`](crate::world::World::restore) (design §14/§16.5). Re-calling
    /// is idempotent and refreshes the glue. Returns the component id.
    pub fn register_cloneable<T: Component + Clone>(&mut self) -> ComponentId {
        let id = self.register::<T>();
        let info = &mut self.infos[id.index() as usize];
        info.clone = Some(clone_fn_of::<T>());
        id
    }

    /// Register `T` (idempotently) with both clone glue and deterministic hash
    /// glue, so it contributes to a world
    /// [`state_hash`](crate::world::snapshot::WorldSnapshot::state_hash) (design
    /// §14). Re-calling is idempotent and refreshes the glue. Returns the
    /// component id.
    pub fn register_hashable<T: Component + Clone + core::hash::Hash>(&mut self) -> ComponentId {
        let id = self.register_cloneable::<T>();
        let info = &mut self.infos[id.index() as usize];
        info.snapshot_hash = Some(snapshot_hash_fn_of::<T>());
        id
    }

    /// Attach type-erased clone glue to an already-registered dynamic component
    /// `id` (design §16.2 dynamic components + §14 snapshot). Returns `false`
    /// if `id` is not registered. `clone` must honor the [`CloneFn`] contract
    /// for this component's [`Layout`].
    pub fn set_clone_fn(&mut self, id: ComponentId, clone: CloneFn) -> bool {
        match self.infos.get_mut(id.index() as usize) {
            Some(info) => {
                info.clone = Some(clone);
                true
            }
            None => false,
        }
    }

    /// Attach the type-erased shared-value box glue to an already-registered
    /// component `id` (design §6 SharedComponent). Returns `false` if `id` is
    /// not registered. Normally called from
    /// [`Component::install_storage_glue`] for components whose `STORAGE` is
    /// [`StorageType::Shared`], via [`shared_box_of`]; also usable for
    /// dynamically-registered shared components (design §16.2).
    pub fn set_shared_box(&mut self, id: ComponentId, shared_box: SharedBoxFn) -> bool {
        match self.infos.get_mut(id.index() as usize) {
            Some(info) => {
                info.shared_box = Some(shared_box);
                true
            }
            None => false,
        }
    }

    /// Register a dynamically-described component (no Rust `TypeId`).
    ///
    /// Supports the data-driven / scripting / editor use cases of design §16.2.
    /// The caller supplies the physical [`Layout`], storage strategy, and
    /// optional drop glue. Each call mints a fresh id (dynamic components are
    /// not deduplicated by name).
    pub fn register_dynamic(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        layout: Layout,
        storage: StorageType,
        drop: Option<DropFn>,
    ) -> ComponentId {
        let id = ComponentId(self.infos.len() as u32);
        self.infos.push(ComponentInfo {
            id,
            name: name.into(),
            layout,
            storage,
            type_id: None,
            drop,
            clone: None,
            snapshot_hash: None,
            hooks: ComponentHooks::new(),
            required: Vec::new(),
            shared_box: None,
        });
        id
    }

    /// Attach (replacing any existing) the lifecycle [`ComponentHooks`] for an
    /// already-registered component `id` (design §12). Returns `false` if `id`
    /// is not registered.
    pub fn set_hooks(&mut self, id: ComponentId, hooks: ComponentHooks) -> bool {
        match self.infos.get_mut(id.index() as usize) {
            Some(info) => {
                let non_empty = !hooks.is_empty();
                info.hooks = hooks;
                self.hooks_registered |= non_empty;
                true
            }
            None => false,
        }
    }

    /// Whether any registered component currently carries a non-empty
    /// [`ComponentHooks`] set. A cheap global gate for the structural paths
    /// (design §12). Monotonic: clearing a component's hooks does not reset it,
    /// which only risks a redundant (still correct) per-id hook scan.
    #[inline]
    pub fn has_hooks(&self) -> bool {
        self.hooks_registered
    }

    /// Declare that `requirer` requires `required`, auto-constructed via `ctor`
    /// whenever `requirer` is inserted and `required` is absent (design §16.1).
    ///
    /// Re-declaring the same edge replaces its constructor. Rebuilds the
    /// transitive, de-duplicated closure cached on every component. Returns
    /// `false` (and does nothing) if either id is not registered, or if the
    /// edge is a direct self-requirement.
    pub fn register_required(
        &mut self,
        requirer: ComponentId,
        required: ComponentId,
        ctor: RequiredCtor,
    ) -> bool {
        if requirer == required {
            return false;
        }
        if self.info(requirer).is_none() || self.info(required).is_none() {
            return false;
        }
        let edges = self.direct_required.entry(requirer).or_default();
        if let Some(existing) = edges.iter_mut().find(|e| e.id == required) {
            existing.ctor = ctor;
        } else {
            edges.push(RequiredComponent { id: required, ctor });
        }
        self.required_registered = true;
        self.recompute_required_closures();
        true
    }

    /// Whether any required-component edge has been declared — a cheap global
    /// gate for the structural paths (design §16.1).
    #[inline]
    pub fn has_required(&self) -> bool {
        self.required_registered
    }

    /// Rebuild the flattened transitive required-component closure for every
    /// component from the `direct_required` edge set. Breadth-first so the
    /// requirement nearest an explicitly-inserted component wins when the same
    /// id is reachable through multiple paths; cycle-safe via a visited set
    /// seeded with the root (a component never requires itself).
    fn recompute_required_closures(&mut self) {
        let n = self.infos.len();
        for i in 0..n {
            let root = ComponentId(i as u32);
            let mut out: Vec<RequiredComponent> = Vec::new();
            let mut seen: Vec<ComponentId> = alloc::vec![root];
            let mut queue: Vec<ComponentId> = Vec::new();
            // Seed with the root's direct requirements.
            if let Some(direct) = self.direct_required.get(&root) {
                for rc in direct {
                    if !seen.contains(&rc.id) {
                        seen.push(rc.id);
                        out.push(rc.clone());
                        queue.push(rc.id);
                    }
                }
            }
            // Expand breadth-first.
            let mut qi = 0;
            while qi < queue.len() {
                let cur = queue[qi];
                qi += 1;
                if let Some(direct) = self.direct_required.get(&cur) {
                    for rc in direct {
                        if !seen.contains(&rc.id) {
                            seen.push(rc.id);
                            out.push(rc.clone());
                            queue.push(rc.id);
                        }
                    }
                }
            }
            self.infos[i].required = out;
        }
    }

    /// Look up the id previously assigned to Rust type `T`, without
    /// registering it.
    #[inline]
    pub fn id_of<T: Component>(&self) -> Option<ComponentId> {
        self.by_type.get(&TypeId::of::<T>()).copied()
    }

    /// Fetch the metadata for a registered component id.
    #[inline]
    pub fn info(&self, id: ComponentId) -> Option<&ComponentInfo> {
        self.infos.get(id.0 as usize)
    }
}

/// A sorted, de-duplicated set of component ids identifying an archetype.
///
/// Stored sorted so two sets with the same members compare and hash equal
/// regardless of insertion order — this is what lets the archetype graph find
/// the canonical archetype for any component combination.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct ComponentSet {
    ids: Box<[ComponentId]>,
}

impl ComponentSet {
    /// Build a set from an iterator of ids, sorting and de-duplicating.
    pub fn from_ids(iter: impl IntoIterator<Item = ComponentId>) -> Self {
        let mut ids: Vec<ComponentId> = iter.into_iter().collect();
        ids.sort_unstable();
        ids.dedup();
        Self {
            ids: ids.into_boxed_slice(),
        }
    }

    /// The ids in this set, sorted ascending.
    #[inline]
    pub fn ids(&self) -> &[ComponentId] {
        &self.ids
    }

    /// Number of components in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the set is empty (the "unit"/empty archetype).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Whether `id` is a member.
    #[inline]
    pub fn contains(&self, id: ComponentId) -> bool {
        self.ids.binary_search(&id).is_ok()
    }

    /// Return a new set with `id` added (no-op if already present).
    pub fn with(&self, id: ComponentId) -> Self {
        if self.contains(id) {
            return self.clone();
        }
        let mut ids: Vec<ComponentId> = self.ids.to_vec();
        let pos = ids.partition_point(|&x| x < id);
        ids.insert(pos, id);
        Self {
            ids: ids.into_boxed_slice(),
        }
    }

    /// Return a new set with `id` removed (no-op if absent).
    pub fn without(&self, id: ComponentId) -> Self {
        if !self.contains(id) {
            return self.clone();
        }
        let ids: Vec<ComponentId> = self.ids.iter().copied().filter(|&x| x != id).collect();
        Self {
            ids: ids.into_boxed_slice(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct A(#[allow(dead_code)] u32);
    impl Component for A {}
    struct B;
    impl Component for B {
        const STORAGE: StorageType = StorageType::SparseSet;
    }

    #[test]
    fn register_is_idempotent_and_dense() {
        let mut c = Components::new();
        let a0 = c.register::<A>();
        let a1 = c.register::<A>();
        let b = c.register::<B>();
        assert_eq!(a0, a1);
        assert_ne!(a0, b);
        assert_eq!(a0.index(), 0);
        assert_eq!(b.index(), 1);
        assert_eq!(c.len(), 2);
        assert_eq!(c.id_of::<A>(), Some(a0));
    }

    #[test]
    fn info_records_layout_and_storage() {
        let mut c = Components::new();
        let a = c.register::<A>();
        let b = c.register::<B>();
        assert_eq!(c.info(a).unwrap().layout(), Layout::new::<A>());
        assert_eq!(c.info(a).unwrap().storage(), StorageType::Table);
        assert_eq!(c.info(b).unwrap().storage(), StorageType::SparseSet);
    }

    #[test]
    fn component_set_is_order_independent() {
        let a = ComponentId::new(5);
        let b = ComponentId::new(2);
        let c = ComponentId::new(9);
        let s1 = ComponentSet::from_ids([a, b, c]);
        let s2 = ComponentSet::from_ids([c, a, b, a]);
        assert_eq!(s1, s2);
        assert_eq!(s1.ids(), &[b, a, c]);
        assert!(s1.contains(a));
        assert!(!s1.contains(ComponentId::new(1)));
    }

    #[test]
    fn component_set_with_without() {
        let base = ComponentSet::from_ids([ComponentId::new(1), ComponentId::new(3)]);
        let added = base.with(ComponentId::new(2));
        assert_eq!(
            added.ids(),
            &[
                ComponentId::new(1),
                ComponentId::new(2),
                ComponentId::new(3)
            ]
        );
        let removed = added.without(ComponentId::new(3));
        assert_eq!(removed.ids(), &[ComponentId::new(1), ComponentId::new(2)]);
        // Idempotent edges.
        assert_eq!(base.with(ComponentId::new(1)), base);
        assert_eq!(base.without(ComponentId::new(99)), base);
    }
}
