//! Archetypes: the canonical grouping of entities by their exact component set.
//!
//! An [`Archetype`] owns the [`Table`] that stores every entity sharing one
//! [`ComponentSet`]. The [`Archetypes`] registry maps each distinct component
//! set to a stable [`ArchetypeId`] so that any two entities with the same
//! components land in the same archetype and are iterated together.
//!
//! Structural changes (adding/removing a component) move an entity from one
//! archetype to another. For M0 the destination archetype is resolved by a
//! hash lookup on the target component set; the per-archetype add/remove edge
//! cache that turns this into an O(1) pointer-follow is an M2 refinement
//! (design §5.3, §9).

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::{ComponentId, ComponentSet, Components};
use crate::storage::{SharedValue, SharedValueId, Table};

/// A stable, dense identifier for an [`Archetype`] within a world.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ArchetypeId(u32);

impl ArchetypeId {
    /// Sentinel used by [`EntityLocation::EMPTY`](crate::entity::EntityLocation).
    pub const INVALID: Self = Self(u32::MAX);

    /// The archetype holding entities with no components (the "empty"
    /// archetype), always allocated at construction.
    pub const EMPTY: Self = Self(0);

    /// Construct an [`ArchetypeId`] from its raw index.
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The raw dense index.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// A single shared-component binding carried by an archetype: the shared
/// [`ComponentId`], the interned [`SharedValueId`] every entity here shares,
/// and a canonical [`Arc`] handle keeping the value alive for the archetype's
/// lifetime (design §6 SharedComponent). Bindings are stored sorted by
/// [`ComponentId`] so lookup is a binary search and the derived archetype key
/// is canonical.
pub type SharedBinding = (ComponentId, SharedValueId, Arc<dyn SharedValue>);

/// One archetype: a (non-shared) component set plus the table storing its
/// entities, together with the shared-component value bindings that split it
/// from otherwise-identical archetypes (design §6).
pub struct Archetype {
    id: ArchetypeId,
    components: ComponentSet,
    shared: Box<[SharedBinding]>,
    table: Table,
}

impl Archetype {
    /// This archetype's id.
    #[inline]
    pub fn id(&self) -> ArchetypeId {
        self.id
    }

    /// The exact set of component ids every entity here has.
    #[inline]
    pub fn components(&self) -> &ComponentSet {
        &self.components
    }

    /// Whether entities here have the table/sparse component `id`. Shared
    /// bindings are *not* reported here (they are archetype-level batch keys,
    /// not table columns); query [`Archetype::shared_binding`] for those.
    #[inline]
    pub fn contains(&self, id: ComponentId) -> bool {
        self.components.contains(id)
    }

    /// The interned [`SharedValueId`] bound to shared component `id` in this
    /// archetype, or `None` if this archetype has no binding for `id` (design
    /// §6). This is the precise, per-archetype membership test for a shared
    /// component.
    #[inline]
    pub fn shared_binding(&self, id: ComponentId) -> Option<SharedValueId> {
        self.shared
            .binary_search_by_key(&id, |&(cid, _, _)| cid)
            .ok()
            .map(|i| self.shared[i].1)
    }

    /// A canonical [`Arc`] handle to the interned value bound to shared
    /// component `id` in this archetype, or `None` if unbound (design §6). The
    /// value is immutable; downcast via
    /// [`SharedValue::as_any`](crate::storage::SharedValue::as_any) to read it.
    #[inline]
    pub fn shared_arc(&self, id: ComponentId) -> Option<&Arc<dyn SharedValue>> {
        self.shared
            .binary_search_by_key(&id, |&(cid, _, _)| cid)
            .ok()
            .map(|i| &self.shared[i].2)
    }

    /// All shared-component bindings on this archetype, sorted by
    /// [`ComponentId`] (design §6).
    #[inline]
    pub fn shared_bindings(&self) -> &[SharedBinding] {
        &self.shared
    }

    /// Number of entities in this archetype.
    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether this archetype currently holds no entities.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Immutable access to the backing table.
    #[inline]
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Mutable access to the backing table.
    #[inline]
    pub fn table_mut(&mut self) -> &mut Table {
        &mut self.table
    }
}

/// The canonical identity of an archetype: its non-shared [`ComponentSet`]
/// plus its sorted shared-value bindings. Two archetypes with the same table
/// component set but different shared values have distinct keys, which is how
/// a shared component splits an archetype by value (design §6). Shared bindings
/// are reduced to `(ComponentId, SharedValueId)` here (the [`Arc`] handle is
/// identity-irrelevant and lives on the [`Archetype`]).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
struct ArchetypeKey {
    set: ComponentSet,
    shared: Box<[(ComponentId, SharedValueId)]>,
}

impl ArchetypeKey {
    /// Build a canonical key, sorting the shared bindings by [`ComponentId`].
    fn new(set: ComponentSet, mut shared: Vec<(ComponentId, SharedValueId)>) -> Self {
        shared.sort_unstable_by_key(|&(cid, _)| cid);
        Self {
            set,
            shared: shared.into_boxed_slice(),
        }
    }

    /// Derive the key for an archetype's table set and (already-materialized)
    /// shared bindings.
    fn from_bindings(set: &ComponentSet, shared: &[SharedBinding]) -> Self {
        Self::new(
            set.clone(),
            shared.iter().map(|&(cid, sid, _)| (cid, sid)).collect(),
        )
    }
}

/// The per-world registry of archetypes.
pub struct Archetypes {
    archetypes: Vec<Archetype>,
    by_key: HashMap<ArchetypeKey, ArchetypeId>,
}

impl Archetypes {
    /// Create a registry pre-populated with the empty archetype at
    /// [`ArchetypeId::EMPTY`].
    pub fn new() -> Self {
        let mut this = Self {
            archetypes: Vec::new(),
            by_key: HashMap::default(),
        };
        let empty = ComponentSet::default();
        let id = this.insert(empty, &[], &Components::new());
        debug_assert_eq!(id, ArchetypeId::EMPTY);
        this
    }

    /// Number of archetypes.
    #[inline]
    pub fn len(&self) -> usize {
        self.archetypes.len()
    }

    /// Whether there are no archetypes (never true after [`Archetypes::new`]).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.archetypes.is_empty()
    }

    /// Borrow an archetype by id.
    #[inline]
    pub fn get(&self, id: ArchetypeId) -> Option<&Archetype> {
        self.archetypes.get(id.0 as usize)
    }

    /// Mutably borrow an archetype by id.
    #[inline]
    pub fn get_mut(&mut self, id: ArchetypeId) -> Option<&mut Archetype> {
        self.archetypes.get_mut(id.0 as usize)
    }

    /// Mutably borrow two distinct archetypes at once (for structural moves).
    ///
    /// # Panics
    /// Panics if `a == b` or either id is out of range.
    pub fn get_pair_mut(
        &mut self,
        a: ArchetypeId,
        b: ArchetypeId,
    ) -> (&mut Archetype, &mut Archetype) {
        assert_ne!(a, b, "cannot borrow the same archetype twice");
        let (lo, hi) = (a.0.min(b.0) as usize, a.0.max(b.0) as usize);
        let (left, right) = self.archetypes.split_at_mut(hi);
        let lo_ref = &mut left[lo];
        let hi_ref = &mut right[0];
        if a.0 < b.0 {
            (lo_ref, hi_ref)
        } else {
            (hi_ref, lo_ref)
        }
    }

    /// All archetypes, in id order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Archetype> {
        self.archetypes.iter()
    }

    /// All archetypes mutably, in id order.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Archetype> {
        self.archetypes.iter_mut()
    }

    /// Get the id for `set`, creating the archetype (and its empty table) if it
    /// does not yet exist. `components` supplies the layout/drop metadata needed
    /// to build the table columns.
    pub fn get_or_create(&mut self, set: &ComponentSet, components: &Components) -> ArchetypeId {
        self.get_or_create_shared(set, &[], components)
    }

    /// Get the id for the archetype identified by table component `set` *and*
    /// the given shared-value `bindings`, creating it (and its empty table) if
    /// it does not yet exist (design §6 SharedComponent archetype split).
    ///
    /// `bindings` need not be sorted; the derived key is canonicalized. The
    /// supplied [`Arc`] handles are retained on the created archetype to keep
    /// each interned value alive for the archetype's lifetime (shared values
    /// are immortal once an archetype binds them). `components` supplies the
    /// layout/drop metadata for the table columns of `set`.
    pub fn get_or_create_shared(
        &mut self,
        set: &ComponentSet,
        bindings: &[SharedBinding],
        components: &Components,
    ) -> ArchetypeId {
        let key = ArchetypeKey::from_bindings(set, bindings);
        if let Some(&id) = self.by_key.get(&key) {
            return id;
        }
        self.insert(set.clone(), bindings, components)
    }

    /// Drop the `by_key` lookup entry of every **empty** archetype whose shared
    /// bindings include `(cid, sid)`, after that `(cid, sid)` has had its last
    /// reference released from its value pool (design §6 SharedComponent 生命周期).
    ///
    /// A freed [`SharedValueId`] may be recycled by the pool to name a *different*
    /// value. If a now-stale empty archetype keyed on the old `(cid, sid)` stayed
    /// in `by_key`, a later spawn of that different value with the same table set
    /// would collide with it in [`Archetypes::get_or_create_shared`] and route the
    /// entity into an archetype still holding the *old* value's [`Arc`] — a silent
    /// wrong-value read. Evicting the lookup entry closes that aliasing window.
    ///
    /// The [`Archetype`] struct is left tombstoned in the `archetypes` vec so
    /// existing [`ArchetypeId`]s (dense vec indices) stay stable; only the
    /// `by_key` entry is removed, so the tombstone can never again be returned by
    /// a lookup and a fresh archetype is minted on demand instead. This is safe
    /// because `refcount(cid, sid) == 0` implies no live entity binds `(cid, sid)`,
    /// so every archetype binding it is empty and unreferenced by any
    /// [`EntityLocation`](crate::entity::EntityLocation).
    pub fn evict_shared_binding(&mut self, cid: ComponentId, sid: SharedValueId) {
        let Self { archetypes, by_key } = self;
        by_key.retain(|_key, id| {
            let arch = &archetypes[id.index() as usize];
            !(arch.is_empty()
                && arch
                    .shared_bindings()
                    .iter()
                    .any(|&(c, s, _)| c == cid && s == sid))
        });
    }

    fn insert(
        &mut self,
        set: ComponentSet,
        bindings: &[SharedBinding],
        components: &Components,
    ) -> ArchetypeId {
        let id = ArchetypeId(self.archetypes.len() as u32);
        let columns = set.ids().iter().map(|&cid| {
            let info = components
                .info(cid)
                .expect("component in archetype set must be registered");
            (cid, info.layout(), info.drop_fn())
        });
        let table = Table::new(columns);
        let mut shared: Vec<SharedBinding> = bindings.to_vec();
        shared.sort_unstable_by_key(|&(cid, _, _)| cid);
        let shared = shared.into_boxed_slice();
        let key = ArchetypeKey::from_bindings(&set, &shared);
        self.archetypes.push(Archetype {
            id,
            components: set,
            shared,
            table,
        });
        self.by_key.insert(key, id);
        id
    }
}

impl Default for Archetypes {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{Component, Components};

    struct A(#[allow(dead_code)] u32);
    impl Component for A {}
    struct B(#[allow(dead_code)] u64);
    impl Component for B {}

    #[test]
    fn empty_archetype_exists_at_id_zero() {
        let archetypes = Archetypes::new();
        assert_eq!(archetypes.len(), 1);
        let empty = archetypes.get(ArchetypeId::EMPTY).unwrap();
        assert!(empty.components().is_empty());
        assert!(empty.is_empty());
    }

    #[test]
    fn get_or_create_dedups_by_set() {
        let mut comps = Components::new();
        let a = comps.register::<A>();
        let b = comps.register::<B>();
        let mut archetypes = Archetypes::new();

        let set_ab = ComponentSet::from_ids([a, b]);
        let set_ba = ComponentSet::from_ids([b, a]);
        let id1 = archetypes.get_or_create(&set_ab, &comps);
        let id2 = archetypes.get_or_create(&set_ba, &comps);
        assert_eq!(id1, id2);
        assert_eq!(archetypes.len(), 2); // empty + {A,B}

        let set_a = ComponentSet::from_ids([a]);
        let id3 = archetypes.get_or_create(&set_a, &comps);
        assert_ne!(id1, id3);
        assert_eq!(archetypes.len(), 3);
        assert!(archetypes.get(id3).unwrap().contains(a));
        assert!(!archetypes.get(id3).unwrap().contains(b));
    }
}
