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

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::{ComponentId, ComponentSet, Components};
use crate::storage::Table;

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

/// One archetype: a component set plus the table storing its entities.
pub struct Archetype {
    id: ArchetypeId,
    components: ComponentSet,
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

    /// Whether entities here have component `id`.
    #[inline]
    pub fn contains(&self, id: ComponentId) -> bool {
        self.components.contains(id)
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

/// The per-world registry of archetypes.
pub struct Archetypes {
    archetypes: Vec<Archetype>,
    by_set: HashMap<ComponentSet, ArchetypeId>,
}

impl Archetypes {
    /// Create a registry pre-populated with the empty archetype at
    /// [`ArchetypeId::EMPTY`].
    pub fn new() -> Self {
        let mut this = Self {
            archetypes: Vec::new(),
            by_set: HashMap::default(),
        };
        let empty = ComponentSet::default();
        let id = this.insert(empty, &Components::new());
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

    /// Get the id for `set`, creating the archetype (and its empty table) if it
    /// does not yet exist. `components` supplies the layout/drop metadata needed
    /// to build the table columns.
    pub fn get_or_create(&mut self, set: &ComponentSet, components: &Components) -> ArchetypeId {
        if let Some(&id) = self.by_set.get(set) {
            return id;
        }
        self.insert(set.clone(), components)
    }

    fn insert(&mut self, set: ComponentSet, components: &Components) -> ArchetypeId {
        let id = ArchetypeId(self.archetypes.len() as u32);
        let columns = set.ids().iter().map(|&cid| {
            let info = components
                .info(cid)
                .expect("component in archetype set must be registered");
            (cid, info.layout(), info.drop_fn())
        });
        let table = Table::new(columns);
        self.archetypes.push(Archetype {
            id,
            components: set.clone(),
            table,
        });
        self.by_set.insert(set, id);
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
