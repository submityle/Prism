//! The per-world registry of [`ComponentSparseSet`] columns (design §6).
//!
//! Table components live inside archetype tables; sparse components live *out
//! of band* here, keyed by [`Entity`], one [`ComponentSparseSet`] per
//! sparse-declared [`ComponentId`]. Keeping them in a dedicated registry (not
//! on the archetype) is exactly what lets insert/remove avoid archetype
//! fragmentation: toggling a sparse component never changes which archetype an
//! entity belongs to (design §6 "增删不搬迁 archetype").
//!
//! Sets are created lazily the first time a sparse component is written, from
//! the physical [`Layout`] and drop glue recorded in the component registry, so
//! a declared-but-unused sparse component costs nothing.

use alloc::vec::Vec;
use core::alloc::Layout;

use crate::component::{ComponentId, DropFn};
use crate::entity::Entity;
use crate::storage::sparse::ComponentSparseSet;

/// Registry mapping each sparse [`ComponentId`] to its [`ComponentSparseSet`].
///
/// Indexed directly by [`ComponentId::index`]; the slot is `None` until the
/// component is first written. Table components never allocate a slot.
#[derive(Default)]
pub struct SparseSets {
    /// `sets[id] = Some(set)` once the sparse component `id` has been written.
    sets: Vec<Option<ComponentSparseSet>>,
}

impl SparseSets {
    /// A new, empty registry.
    #[inline]
    pub fn new() -> Self {
        Self { sets: Vec::new() }
    }

    /// Number of sparse components that currently have a backing set.
    #[inline]
    pub fn len(&self) -> usize {
        self.sets.iter().filter(|s| s.is_some()).count()
    }

    /// Whether no sparse component has a backing set yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sets.iter().all(Option::is_none)
    }

    /// The set for `id`, if one has been created.
    #[inline]
    pub fn get(&self, id: ComponentId) -> Option<&ComponentSparseSet> {
        self.sets.get(id.index() as usize).and_then(Option::as_ref)
    }

    /// The mutable set for `id`, if one has been created.
    #[inline]
    pub fn get_mut(&mut self, id: ComponentId) -> Option<&mut ComponentSparseSet> {
        self.sets
            .get_mut(id.index() as usize)
            .and_then(Option::as_mut)
    }

    /// The set for `id`, creating it from `layout`/`drop` on first use.
    pub fn get_or_init(
        &mut self,
        id: ComponentId,
        layout: Layout,
        drop: Option<DropFn>,
    ) -> &mut ComponentSparseSet {
        let index = id.index() as usize;
        if index >= self.sets.len() {
            self.sets.resize_with(index + 1, || None);
        }
        self.sets[index].get_or_insert_with(|| ComponentSparseSet::new(layout, drop))
    }

    /// Whether `entity` has a value in the set for `id`.
    #[inline]
    pub fn contains(&self, id: ComponentId, entity: Entity) -> bool {
        self.get(id).is_some_and(|set| set.contains(entity))
    }

    /// Collect every sparse [`ComponentId`] that currently holds a value for
    /// `entity`, in ascending id order. Used by the despawn/removal paths to
    /// enumerate an entity's out-of-band components for lifecycle hooks
    /// (design §12).
    pub fn ids_for(&self, entity: Entity) -> Vec<ComponentId> {
        self.sets
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                slot.as_ref()
                    .filter(|set| set.contains(entity))
                    .map(|_| ComponentId::new(i as u32))
            })
            .collect()
    }

    /// Remove `entity`'s value from the set for `id`, dropping it. Returns
    /// `true` if a value was present and removed.
    #[inline]
    pub fn remove(&mut self, id: ComponentId, entity: Entity) -> bool {
        self.get_mut(id).is_some_and(|set| set.remove(entity))
    }

    /// Remove `entity` from every sparse set it appears in (despawn path).
    pub fn remove_entity_from_all(&mut self, entity: Entity) {
        for set in self.sets.iter_mut().flatten() {
            set.remove(entity);
        }
    }

    /// Clamp every stored tick in every set against `this_run` (design §10
    /// tick-wrap guard), mirroring the table-column pass.
    pub fn check_change_ticks(&mut self, this_run: crate::change::Tick) {
        for set in self.sets.iter_mut().flatten() {
            set.check_change_ticks(this_run);
        }
    }
}
