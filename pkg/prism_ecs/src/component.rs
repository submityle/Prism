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
use alloc::vec::Vec;
use core::alloc::Layout;
use core::any::{TypeId, type_name};

use crate::collections::HashMap;

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
}

/// How a component type is physically stored.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum StorageType {
    /// Columnar Structure-of-Arrays within the archetype's table. Fastest to
    /// iterate and SIMD-friendly; this is the default and the only strategy
    /// wired up in M0.
    Table,
    /// A sparse set keyed by entity index. Insert/remove is O(1) without an
    /// archetype move. Reserved for M2; registered components may declare it
    /// but the M0 storage treats every component as [`StorageType::Table`].
    SparseSet,
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

/// Runtime metadata for one registered component type.
pub struct ComponentInfo {
    id: ComponentId,
    name: Cow<'static, str>,
    layout: Layout,
    storage: StorageType,
    type_id: Option<TypeId>,
    drop: Option<DropFn>,
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

/// The per-world registry of component metadata.
///
/// Assigns a dense [`ComponentId`] to each distinct component type the first
/// time it is seen, and remembers the mapping from Rust [`TypeId`] to
/// [`ComponentId`] so repeated lookups are cheap.
#[derive(Default)]
pub struct Components {
    infos: Vec<ComponentInfo>,
    by_type: HashMap<TypeId, ComponentId>,
}

impl Components {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            infos: Vec::new(),
            by_type: HashMap::default(),
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
        });
        self.by_type.insert(type_id, id);
        id
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
        });
        id
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
            &[ComponentId::new(1), ComponentId::new(2), ComponentId::new(3)]
        );
        let removed = added.without(ComponentId::new(3));
        assert_eq!(removed.ids(), &[ComponentId::new(1), ComponentId::new(2)]);
        // Idempotent edges.
        assert_eq!(base.with(ComponentId::new(1)), base);
        assert_eq!(base.without(ComponentId::new(99)), base);
    }
}
