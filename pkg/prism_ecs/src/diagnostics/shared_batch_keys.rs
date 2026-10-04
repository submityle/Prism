//! Shared-component batch-key census (design §6 SharedComponent / §15 GPU 批次键
//! / §16.6).
//!
//! A [`Shared`](crate::component::StorageType::Shared) component is not stored
//! per entity: its *value* is de-duplicated into one interned
//! [`SharedValueId`] and used as a per-archetype **batch key** (design §6, the
//! Unity shared-component shape). Every entity carrying the same value lands in
//! the same archetype variant, and distinct values fragment the archetype
//! graph. This is exactly the key a GPU-driven renderer groups by: entities
//! sharing a `(mesh, material)` batch key collapse into one instanced /
//! indirect draw (design §15, the Horizon / Insomniac shape), so the *fan-out*
//! of a batch key — how many entities amortise behind it — is the instancing
//! quality signal.
//!
//! Two forces pull against each other and this census surfaces both:
//!
//! * **Instancing amortisation** wants *few* batch keys each backing *many*
//!   entities: a key behind one thousand entities is one draw amortising a
//!   thousand instances. A key behind a single entity
//!   ([`singleton`](SharedComponentBatchReport::singleton_batch_key_count)) is
//!   a draw amortising nothing.
//! * **Archetype fragmentation** (design §22 risk #5) warns the opposite way:
//!   every distinct shared value mints another archetype variant, so a shared
//!   component with a huge
//!   [`distinct_values`](SharedComponentBatchEntry::distinct_values) is a
//!   fragmentation hot spot — a high-cardinality value mistakenly declared
//!   `Shared` instead of a plain table column.
//!
//! Unlike [`storage_distribution`](super::storage_distribution), which counts
//! `Shared` components at *registration* (a type-level census), and
//! [`component_distribution`](super::component_distribution), which attributes
//! archetype spread to *table* components (shared bindings are archetype-level
//! keys, not table columns, so they are invisible there), this report reads the
//! *live* bindings off the archetype graph and keys on the
//! `(ComponentId, SharedValueId)` batch key itself.
//!
//! # What it surfaces
//! * Per **batch key** ([`SharedBatchKeyEntry`]): the archetype count the key
//!   spans (one shared value can bind several archetypes that differ only in
//!   their table columns) and the live entity fan-out behind it.
//! * Per **shared component** ([`SharedComponentBatchEntry`]): distinct value
//!   count (fragmentation), total archetypes and entities, and the
//!   largest / smallest per-value fan-out.
//! * **World roll-ups**: total batch keys, singleton and empty keys, and the
//!   singleton share — a direct read on draw-call pressure vs. amortisation.
//!
//! # Honest scope
//! Counts describe *live bindings on the archetype graph*: a shared component
//! that is registered but never bound to any archetype has no batch key and is
//! not reported (mirroring [`component_distribution`]'s used-only policy). An
//! archetype that still exists after its entities despawned contributes an
//! [`empty`](SharedComponentBatchReport::empty_batch_key_count) batch key
//! (archetype present, zero fan-out). Everything is read-only and deterministic
//! (design §14): batch keys are sorted by `(component id, value id)` and
//! component entries by component id, independent of archetype creation or
//! hashing order.

use alloc::string::String;
use alloc::vec::Vec;

use crate::archetype::Archetypes;
use crate::collections::HashMap;
use crate::component::{ComponentId, Components};
use crate::storage::SharedValueId;

/// Integer permille (`parts per thousand`) of `num / den`, returning `0` when
/// `den` is zero.
#[inline]
fn permille(num: u64, den: u64) -> u64 {
    (num * 1000).checked_div(den).unwrap_or(0)
}

/// One shared-component **batch key**: a `(component, value)` pair, the number
/// of archetypes that bind it, and the live entity fan-out behind it (design
/// §6 / §15).
///
/// A single interned value can bind more than one archetype when those
/// archetypes differ only in their *table* columns (e.g. `{Position, Batch(7)}`
/// and `{Position, Velocity, Batch(7)}` both bind `Batch -> 7`); the GPU still
/// groups all of them under the one batch key.
#[derive(Clone, Debug)]
pub struct SharedBatchKeyEntry {
    /// The shared component this batch key belongs to.
    pub component: ComponentId,
    /// The interned value id that forms the key (design §6).
    pub value: SharedValueId,
    /// Number of archetypes that bind this `(component, value)` key.
    pub archetype_count: usize,
    /// Total live entities across those archetypes — the instancing fan-out
    /// amortised behind one draw (design §15). `0` for a key held only by
    /// now-empty archetypes.
    pub entity_count: usize,
}

/// Per shared-component roll-up of its batch keys: distinct value count,
/// archetype and entity totals, and per-value fan-out extremes (design §6 /
/// §15 / §22 risk #5).
#[derive(Clone, Debug)]
pub struct SharedComponentBatchEntry {
    /// The shared component described.
    pub component: ComponentId,
    /// Registered name of the component (from
    /// [`ComponentInfo::name`](crate::component::ComponentInfo::name)).
    pub name: String,
    /// Number of distinct interned values currently bound — the batch-key
    /// cardinality this component induces, and the archetype-fragmentation
    /// pressure it adds (design §22 risk #5).
    pub distinct_values: usize,
    /// Total archetypes carrying a binding for this component (summed across
    /// its batch keys; one value can span several archetypes).
    pub archetype_count: usize,
    /// Total live entities carrying this shared component.
    pub entity_count: usize,
    /// Largest entity fan-out behind a single value of this component — the
    /// best-amortised draw it produces (design §15).
    pub max_value_fanout: usize,
    /// Smallest entity fan-out behind a single value of this component — the
    /// worst-amortised draw (`0` when some value is held only by empty
    /// archetypes).
    pub min_value_fanout: usize,
    /// Number of this component's values held only by now-empty archetypes
    /// (zero fan-out).
    pub empty_value_count: usize,
}

impl SharedComponentBatchEntry {
    /// Mean entities amortised behind each distinct value, rounded down; `0`
    /// when there are no values. A low mean against a high
    /// [`distinct_values`](Self::distinct_values) signals a high-cardinality
    /// value better modelled as a plain table column than a shared batch key.
    #[inline]
    pub fn mean_value_fanout(&self) -> u64 {
        (self.entity_count as u64)
            .checked_div(self.distinct_values as u64)
            .unwrap_or(0)
    }
}

/// Read-only census of every live shared-component batch key in a world, plus
/// per-component roll-ups and world-level draw-call-pressure totals (design §6
/// / §15 / §16.6).
#[derive(Clone, Debug)]
pub struct SharedComponentBatchReport {
    batch_key_count: usize,
    total_bound_entities: usize,
    singleton_batch_key_count: usize,
    empty_batch_key_count: usize,
    max_distinct_values: usize,
    components: Vec<SharedComponentBatchEntry>,
    batch_keys: Vec<SharedBatchKeyEntry>,
}

impl SharedComponentBatchReport {
    /// Build the census from a world via
    /// [`World::archetypes`](crate::world::World::archetypes) and
    /// [`World::components`](crate::world::World::components).
    #[inline]
    pub fn capture(world: &crate::world::World) -> Self {
        Self::from_parts(world.archetypes(), world.components())
    }

    /// Build the census directly from an [`Archetypes`] graph and the component
    /// [`registry`](Components) (used for component names).
    ///
    /// Batch keys are returned sorted by `(component id, value id)` and
    /// component entries by component id (design §14).
    pub fn from_parts(archetypes: &Archetypes, components: &Components) -> Self {
        // (component index, value index) -> (archetype count, entity count).
        let mut keys: HashMap<(u32, u32), (usize, usize)> = HashMap::default();
        for archetype in archetypes.iter() {
            let entities = archetype.len();
            for &(component, value, _) in archetype.shared_bindings() {
                let slot = keys.entry((component.index(), value.index())).or_insert((0, 0));
                slot.0 += 1;
                slot.1 += entities;
            }
        }

        let mut batch_keys: Vec<SharedBatchKeyEntry> = keys
            .into_iter()
            .map(|((component, value), (archetype_count, entity_count))| SharedBatchKeyEntry {
                component: ComponentId::new(component),
                value: SharedValueId::new(value),
                archetype_count,
                entity_count,
            })
            .collect();
        batch_keys.sort_unstable_by_key(|key| (key.component.index(), key.value.index()));

        let batch_key_count = batch_keys.len();
        let mut total_bound_entities = 0usize;
        let mut singleton_batch_key_count = 0usize;
        let mut empty_batch_key_count = 0usize;

        // Fold batch keys into per-component accumulators. `batch_keys` is
        // already ordered by component id, so component entries emerge sorted.
        let mut components_out: Vec<SharedComponentBatchEntry> = Vec::new();
        for key in &batch_keys {
            total_bound_entities += key.entity_count;
            if key.entity_count == 1 {
                singleton_batch_key_count += 1;
            }
            if key.entity_count == 0 {
                empty_batch_key_count += 1;
            }

            let fresh = components_out
                .last()
                .map(|entry| entry.component != key.component)
                .unwrap_or(true);
            if fresh {
                let id = key.component;
                let name = components
                    .info(id)
                    .map(|info| String::from(info.name()))
                    .unwrap_or_default();
                components_out.push(SharedComponentBatchEntry {
                    component: id,
                    name,
                    distinct_values: 0,
                    archetype_count: 0,
                    entity_count: 0,
                    max_value_fanout: 0,
                    min_value_fanout: usize::MAX,
                    empty_value_count: 0,
                });
            }

            let entry = components_out
                .last_mut()
                .expect("an entry was just ensured for this component");
            entry.distinct_values += 1;
            entry.archetype_count += key.archetype_count;
            entry.entity_count += key.entity_count;
            if key.entity_count > entry.max_value_fanout {
                entry.max_value_fanout = key.entity_count;
            }
            if key.entity_count < entry.min_value_fanout {
                entry.min_value_fanout = key.entity_count;
            }
            if key.entity_count == 0 {
                entry.empty_value_count += 1;
            }
        }

        let mut max_distinct_values = 0usize;
        for entry in &mut components_out {
            if entry.min_value_fanout == usize::MAX {
                entry.min_value_fanout = 0;
            }
            if entry.distinct_values > max_distinct_values {
                max_distinct_values = entry.distinct_values;
            }
        }

        Self {
            batch_key_count,
            total_bound_entities,
            singleton_batch_key_count,
            empty_batch_key_count,
            max_distinct_values,
            components: components_out,
            batch_keys,
        }
    }

    /// Number of shared components with at least one live binding.
    #[inline]
    pub fn shared_component_count(&self) -> usize {
        self.components.len()
    }

    /// Whether the world has no live shared-component bindings.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.batch_keys.is_empty()
    }

    /// Total distinct `(component, value)` batch keys — the number of instanced
    /// draw groups the shared components induce (design §15).
    #[inline]
    pub fn batch_key_count(&self) -> usize {
        self.batch_key_count
    }

    /// Total live entities carrying some shared component (an entity carrying
    /// two shared components is counted once per component).
    #[inline]
    pub fn total_bound_entities(&self) -> usize {
        self.total_bound_entities
    }

    /// Number of batch keys backed by exactly one entity — draws that amortise
    /// a single instance, the worst instancing outcome (design §15).
    #[inline]
    pub fn singleton_batch_key_count(&self) -> usize {
        self.singleton_batch_key_count
    }

    /// Number of batch keys held only by now-empty archetypes (zero fan-out).
    #[inline]
    pub fn empty_batch_key_count(&self) -> usize {
        self.empty_batch_key_count
    }

    /// Permille (parts per thousand) of batch keys that are
    /// [`singleton`](Self::singleton_batch_key_count), `0` when there are no
    /// batch keys. A high share means most draws amortise a single instance.
    #[inline]
    pub fn singleton_permille(&self) -> u64 {
        permille(self.singleton_batch_key_count as u64, self.batch_key_count as u64)
    }

    /// Largest [`distinct_values`](SharedComponentBatchEntry::distinct_values)
    /// across all shared components — the peak batch-key cardinality, and the
    /// archetype-fragmentation high-water mark (design §22 risk #5). `0` when
    /// empty.
    #[inline]
    pub fn max_distinct_values(&self) -> usize {
        self.max_distinct_values
    }

    /// Mean entities amortised per batch key across the whole world, rounded
    /// down; `0` when there are no batch keys. The world-level instancing
    /// amortisation (design §15).
    #[inline]
    pub fn mean_entities_per_batch_key(&self) -> u64 {
        (self.total_bound_entities as u64)
            .checked_div(self.batch_key_count as u64)
            .unwrap_or(0)
    }

    /// Per-shared-component roll-ups, sorted by ascending component id.
    #[inline]
    pub fn components(&self) -> &[SharedComponentBatchEntry] {
        &self.components
    }

    /// Per-batch-key entries, sorted by `(component id, value id)`.
    #[inline]
    pub fn batch_keys(&self) -> &[SharedBatchKeyEntry] {
        &self.batch_keys
    }

    /// The shared component inducing the most distinct batch keys — the biggest
    /// fragmentation driver — with the lowest component id winning ties, or
    /// `None` when empty.
    pub fn most_fragmented(&self) -> Option<&SharedComponentBatchEntry> {
        let mut best: Option<&SharedComponentBatchEntry> = None;
        for entry in &self.components {
            match best {
                Some(current) if current.distinct_values >= entry.distinct_values => {}
                _ => best = Some(entry),
            }
        }
        best
    }

    /// The roll-up for shared component `component`, or `None` if it has no live
    /// binding.
    pub fn component(&self, component: ComponentId) -> Option<&SharedComponentBatchEntry> {
        self.components
            .binary_search_by_key(&component.index(), |entry| entry.component.index())
            .ok()
            .map(|i| &self.components[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{Component, ComponentId, Components, StorageType};
    use crate::world::World;

    #[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
    struct Position(i32);
    impl Component for Position {}

    #[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
    struct Velocity(i32);
    impl Component for Velocity {}

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct BatchKey(u32);
    impl Component for BatchKey {
        const STORAGE: StorageType = StorageType::Shared;
        fn install_storage_glue(components: &mut Components, id: ComponentId) {
            components.set_shared_box(id, crate::component::shared_box_of::<Self>());
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Layer(u8);
    impl Component for Layer {
        const STORAGE: StorageType = StorageType::Shared;
        fn install_storage_glue(components: &mut Components, id: ComponentId) {
            components.set_shared_box(id, crate::component::shared_box_of::<Self>());
        }
    }

    #[test]
    fn empty_world_has_no_batch_keys() {
        let world = World::new();
        let report = SharedComponentBatchReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.batch_key_count(), 0);
        assert_eq!(report.shared_component_count(), 0);
        assert_eq!(report.total_bound_entities(), 0);
        assert_eq!(report.singleton_permille(), 0);
        assert_eq!(report.mean_entities_per_batch_key(), 0);
        assert!(report.most_fragmented().is_none());
    }

    #[test]
    fn world_without_shared_components_has_no_batch_keys() {
        let mut world = World::new();
        world.spawn(Position(1));
        world.spawn((Position(2), Velocity(3)));
        let report = SharedComponentBatchReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.shared_component_count(), 0);
    }

    #[test]
    fn distinct_values_split_into_separate_batch_keys() {
        let mut world = World::new();
        // Two entities share value 10; one entity has value 20.
        world.spawn((Position(1), BatchKey(10)));
        world.spawn((Position(2), BatchKey(10)));
        world.spawn((Position(3), BatchKey(20)));
        let report = SharedComponentBatchReport::capture(&world);

        assert_eq!(report.shared_component_count(), 1);
        assert_eq!(report.batch_key_count(), 2);
        assert_eq!(report.total_bound_entities(), 3);

        let batch = world.components().id_of::<BatchKey>().unwrap();
        let entry = report.component(batch).unwrap();
        assert_eq!(entry.distinct_values, 2);
        assert_eq!(entry.entity_count, 3);
        assert_eq!(entry.max_value_fanout, 2);
        assert_eq!(entry.min_value_fanout, 1);
        assert_eq!(entry.empty_value_count, 0);
        assert_eq!(entry.mean_value_fanout(), 1); // 3 / 2 rounded down.
    }

    #[test]
    fn one_value_spans_archetypes_differing_in_table_columns() {
        let mut world = World::new();
        // Same shared value 7, but different table column sets => two
        // archetypes, one batch key.
        world.spawn((Position(1), BatchKey(7)));
        world.spawn((Position(2), Velocity(5), BatchKey(7)));
        let report = SharedComponentBatchReport::capture(&world);

        assert_eq!(report.batch_key_count(), 1);
        let key = &report.batch_keys()[0];
        assert_eq!(key.archetype_count, 2);
        assert_eq!(key.entity_count, 2);
    }

    #[test]
    fn singleton_batch_keys_counted() {
        let mut world = World::new();
        world.spawn((Position(1), BatchKey(1)));
        world.spawn((Position(2), BatchKey(2)));
        world.spawn((Position(3), BatchKey(3)));
        let report = SharedComponentBatchReport::capture(&world);
        assert_eq!(report.batch_key_count(), 3);
        assert_eq!(report.singleton_batch_key_count(), 3);
        assert_eq!(report.singleton_permille(), 1000);
        assert_eq!(report.mean_entities_per_batch_key(), 1);
    }

    #[test]
    fn two_shared_components_rolled_up_independently() {
        let mut world = World::new();
        world.spawn((Position(1), BatchKey(1), Layer(0)));
        world.spawn((Position(2), BatchKey(1), Layer(1)));
        let report = SharedComponentBatchReport::capture(&world);

        let batch = world.components().id_of::<BatchKey>().unwrap();
        let layer = world.components().id_of::<Layer>().unwrap();

        // BatchKey: one value (1) behind both entities.
        let batch_entry = report.component(batch).unwrap();
        assert_eq!(batch_entry.distinct_values, 1);
        assert_eq!(batch_entry.entity_count, 2);
        assert_eq!(batch_entry.max_value_fanout, 2);

        // Layer: two distinct values (0, 1), one entity each.
        let layer_entry = report.component(layer).unwrap();
        assert_eq!(layer_entry.distinct_values, 2);
        assert_eq!(layer_entry.entity_count, 2);
        assert_eq!(layer_entry.max_value_fanout, 1);

        assert_eq!(report.shared_component_count(), 2);
        assert_eq!(report.max_distinct_values(), 2);
        // Layer fragments more than BatchKey.
        assert_eq!(report.most_fragmented().unwrap().component, layer);
    }

    #[test]
    fn batch_keys_sorted_by_component_then_value() {
        let mut world = World::new();
        world.spawn((Position(1), BatchKey(30)));
        world.spawn((Position(2), BatchKey(10)));
        world.spawn((Position(3), BatchKey(20)));
        let report = SharedComponentBatchReport::capture(&world);

        let order: Vec<(u32, u32)> = report
            .batch_keys()
            .iter()
            .map(|k| (k.component.index(), k.value.index()))
            .collect();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(order, sorted);
    }

    #[test]
    fn components_sorted_by_id() {
        let mut world = World::new();
        world.spawn((Position(1), Layer(0)));
        world.spawn((Position(2), BatchKey(1)));
        let report = SharedComponentBatchReport::capture(&world);
        let ids: Vec<u32> = report.components().iter().map(|e| e.component.index()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn from_parts_matches_capture() {
        let mut world = World::new();
        world.spawn((Position(1), BatchKey(1)));
        world.spawn((Position(2), BatchKey(2)));
        let via_capture = SharedComponentBatchReport::capture(&world);
        let via_parts =
            SharedComponentBatchReport::from_parts(world.archetypes(), world.components());
        assert_eq!(via_capture.batch_key_count(), via_parts.batch_key_count());
        assert_eq!(
            via_capture.total_bound_entities(),
            via_parts.total_bound_entities()
        );
        assert_eq!(
            via_capture.shared_component_count(),
            via_parts.shared_component_count()
        );
    }
}
