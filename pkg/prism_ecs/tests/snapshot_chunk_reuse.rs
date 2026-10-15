//! Payload clone/comparison counts prove actual chunk reuse, not just counters
//! in the snapshot API. All counters are test-local and non-semantic.

use prism_ecs::prelude::*;
use prism_ecs::world::snapshot::{SnapshotRing, SnapshotValue};
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Default)]
struct Counts {
    clones: AtomicUsize,
    comparisons: AtomicUsize,
    drops: AtomicUsize,
}

impl Counts {
    fn reset(&self) {
        self.clones.store(0, Ordering::Relaxed);
        self.comparisons.store(0, Ordering::Relaxed);
    }

    fn assert(&self, clones: usize, comparisons: usize) {
        assert_eq!(self.clones.load(Ordering::Relaxed), clones, "value clones");
        assert_eq!(
            self.comparisons.load(Ordering::Relaxed),
            comparisons,
            "value comparisons"
        );
    }
}

struct Payload {
    value: Vec<u64>,
    padding: [u8; 1024],
    counts: Arc<Counts>,
}

impl Payload {
    fn new(value: u64, counts: &Arc<Counts>) -> Self {
        Self {
            value: vec![value],
            padding: [0; 1024],
            counts: counts.clone(),
        }
    }
}

impl Clone for Payload {
    fn clone(&self) -> Self {
        self.counts.clones.fetch_add(1, Ordering::Relaxed);
        Self {
            value: self.value.clone(),
            padding: self.padding,
            counts: self.counts.clone(),
        }
    }
}

impl PartialEq for Payload {
    fn eq(&self, other: &Self) -> bool {
        self.counts.comparisons.fetch_add(1, Ordering::Relaxed);
        self.value == other.value && self.padding == other.padding
    }
}

impl Drop for Payload {
    fn drop(&mut self) {
        self.counts.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Component, Clone, PartialEq)]
struct TableValue(Payload);

#[derive(Component, Clone, PartialEq)]
#[component(storage = "SparseSet")]
struct SparseValue(Payload);

// SAFETY: Vec/padding are independently cloned and cannot mutate through shared
// aliases. Counts are diagnostics only and excluded from the component value.
unsafe impl SnapshotValue for TableValue {}
// SAFETY: same independently owned Payload contract as TableValue.
unsafe impl SnapshotValue for SparseValue {}

#[derive(Component, Clone)]
struct Tag;

fn table_world() -> (World, Arc<Counts>, Vec<Entity>, usize) {
    let mut world = World::new();
    let counts = Arc::new(Counts::default());
    let id = world.register_snapshot_component_reusable::<TableValue>();
    world.register_snapshot_component_equatable::<TableValue>();
    let first = world.spawn(TableValue(Payload::new(0, &counts)));
    let chunk_rows = world
        .archetypes()
        .iter()
        .find(|arch| arch.contains(id))
        .unwrap()
        .table()
        .rows_per_chunk();
    let mut entities = vec![first];
    for i in 1..chunk_rows * 3 + 1 {
        entities.push(world.spawn(TableValue(Payload::new(i as u64, &counts))));
    }
    counts.reset();
    (world, counts, entities, chunk_rows)
}

#[test]
fn clean_capture_diff_and_reconstruction_do_not_clone_or_compare_values() {
    let (mut world, counts, entities, _) = table_world();
    let base = world.snapshot_incremental(None);
    counts.assert(entities.len(), 0);
    counts.reset();
    let clean = world.snapshot_incremental(Some(&base));
    let delta = base.diff(&clean);
    assert_eq!(delta.changed_cell_count(), 0);
    assert_eq!(delta.reused_cell_count(), entities.len());
    let replay = delta.apply(&base);
    counts.assert(0, 0);
    world.restore(&replay);
    for (i, entity) in entities.iter().enumerate() {
        assert_eq!(
            world.get::<TableValue>(*entity).unwrap().0.value,
            [i as u64]
        );
    }
}

#[test]
fn immutable_buffers_outlive_world_and_drop_each_captured_value_once() {
    let (mut world, counts, entities, _) = table_world();
    let base = world.snapshot_incremental(None);
    let clean = world.snapshot_incremental(Some(&base));
    let delta = base.diff(&clean);
    let replay = delta.apply(&base);
    counts.assert(entities.len(), 0);
    drop(world);
    assert_eq!(counts.drops.load(Ordering::Relaxed), entities.len());
    drop(base);
    drop(clean);
    drop(delta);
    assert_eq!(counts.drops.load(Ordering::Relaxed), entities.len());
    drop(replay);
    assert_eq!(counts.drops.load(Ordering::Relaxed), entities.len() * 2);
}

#[test]
fn repeated_same_tick_writes_clone_only_the_touched_table_chunk() {
    let (mut world, counts, entities, chunk_rows) = table_world();
    let base = world.snapshot_incremental(None);
    counts.reset();
    let entity = entities[chunk_rows + 1];
    // No explicit tick advance: the incremental API closes its own boundary.
    world.get_mut::<TableValue>(entity).unwrap().0.value[0] = 100;
    world
        .get_mut::<TableValue>(entity)
        .unwrap()
        .0
        .value
        .push(200);
    world.get_mut::<TableValue>(entity).unwrap().0.value[0] = 300;
    let changed = world.snapshot_incremental(Some(&base));
    counts.assert(chunk_rows, 0);
    let delta = base.diff(&changed);
    assert_eq!(delta.changed_cell_count(), chunk_rows);
    assert_eq!(delta.reused_cell_count(), entities.len() - chunk_rows);
    let replay = delta.apply(&base);
    counts.assert(chunk_rows, 0);
    counts.reset();
    let _clean = world.snapshot_incremental(Some(&replay));
    counts.assert(0, 0);
    world.restore(&base);
    assert_eq!(
        world.get::<TableValue>(entity).unwrap().0.value,
        [entity_index(&entities, entity)]
    );
    world.restore(&replay);
    assert_eq!(world.get::<TableValue>(entity).unwrap().0.value, [300, 200]);
}

fn entity_index(entities: &[Entity], entity: Entity) -> u64 {
    entities.iter().position(|&e| e == entity).unwrap() as u64
}

#[test]
fn read_only_hot_base_cannot_hide_a_later_write_in_the_same_tick() {
    let (mut world, counts, entities, _) = table_world();
    let read_only = world.snapshot();
    counts.reset();
    world.get_mut::<TableValue>(entities[0]).unwrap().0.value[0] = 91;
    let target = world.snapshot_incremental(Some(&read_only));
    // All initial buffers were hot and unsealed: conservatively recapture.
    counts.assert(entities.len(), 0);
    world.restore(&target);
    assert_eq!(world.get::<TableValue>(entities[0]).unwrap().0.value, [91]);
    world.restore(&read_only);
    assert_eq!(world.get::<TableValue>(entities[0]).unwrap().0.value, [0]);
}

#[test]
fn sparse_clean_and_dirty_chunks_preserve_removal_reinsertion_and_restore() {
    let mut world = World::new();
    let counts = Arc::new(Counts::default());
    world.register_snapshot_component_reusable::<SparseValue>();
    world.register_snapshot_component_equatable::<SparseValue>();
    let chunk_rows = prism_ecs::storage::rows_per_chunk(size_of::<SparseValue>());
    let entities = (0..chunk_rows * 3 + 1)
        .map(|i| world.spawn(SparseValue(Payload::new(i as u64, &counts))))
        .collect::<Vec<_>>();
    let base = world.snapshot_incremental(None);
    counts.reset();
    let clean = world.snapshot_incremental(Some(&base));
    counts.assert(0, 0);
    world
        .get_mut::<SparseValue>(entities[chunk_rows + 1])
        .unwrap()
        .0
        .value = vec![400, 500];
    let changed = world.snapshot_incremental(Some(&clean));
    counts.assert(chunk_rows, 0);
    let delta = clean.diff(&changed);
    assert_eq!(delta.changed_cell_count(), chunk_rows);
    let replay = delta.apply(&clean);
    counts.assert(chunk_rows, 0);
    world.remove::<SparseValue>(entities[1]);
    let removed = world.snapshot_incremental(Some(&replay));
    world.insert(entities[1], SparseValue(Payload::new(999, &counts)));
    let inserted = world.snapshot_incremental(Some(&removed));
    counts.reset();
    let _clean = world.snapshot_incremental(Some(&inserted));
    counts.assert(0, 0);
    world.restore(&removed);
    assert!(world.get::<SparseValue>(entities[1]).is_none());
    assert_eq!(
        world
            .get::<SparseValue>(entities[chunk_rows + 1])
            .unwrap()
            .0
            .value,
        [400, 500]
    );
    world.restore(&inserted);
    assert_eq!(
        world.get::<SparseValue>(entities[1]).unwrap().0.value,
        [999]
    );
    world.restore(&base);
    for (i, entity) in entities.iter().enumerate() {
        assert_eq!(
            world.get::<SparseValue>(*entity).unwrap().0.value,
            [i as u64]
        );
    }
    counts.reset();
    let after_restore = world.snapshot_incremental(Some(&base));
    counts.assert(entities.len(), 0);
    counts.reset();
    let _clean = world.snapshot_incremental(Some(&after_restore));
    counts.assert(0, 0);
}

fn sparse_world() -> (World, Arc<Counts>, Vec<Entity>, usize) {
    let mut world = World::new();
    let counts = Arc::new(Counts::default());
    world.register_snapshot_component_reusable::<SparseValue>();
    world.register_snapshot_component_equatable::<SparseValue>();
    let chunk_rows = prism_ecs::storage::rows_per_chunk(size_of::<SparseValue>());
    let entities = (0..chunk_rows * 3 + 1)
        .map(|i| world.spawn(SparseValue(Payload::new(i as u64, &counts))))
        .collect();
    (world, counts, entities, chunk_rows)
}

#[test]
fn sparse_old_tick_override_cannot_reuse_stale_payload() {
    let (mut world, counts, entities, chunk_rows) = sparse_world();
    let entity = entities[chunk_rows + 1];
    let original_ticks = world.get_ticks::<SparseValue>(entity).unwrap();
    let base = world.snapshot_incremental(None);
    counts.reset();
    {
        let mut value = world.get_mut::<SparseValue>(entity).unwrap();
        value.0.value = vec![700, 800];
        value.set_last_changed(original_ticks.changed);
    }
    let changed = world.snapshot_incremental(Some(&base));
    counts.assert(chunk_rows, 0);
    let ticks = world.get_ticks::<SparseValue>(entity).unwrap();
    assert_eq!(ticks.added, original_ticks.added);
    assert_eq!(ticks.changed, original_ticks.changed);
    let delta = base.diff(&changed);
    assert_eq!(delta.changed_cell_count(), chunk_rows);
    assert_eq!(delta.reused_cell_count(), entities.len() - chunk_rows);
    let replay = delta.apply(&base);
    counts.assert(chunk_rows, 0);
    counts.reset();
    let _clean = world.snapshot_incremental(Some(&replay));
    counts.assert(0, 0);
    world.restore(&replay);
    assert_eq!(
        world.get::<SparseValue>(entity).unwrap().0.value,
        [700, 800]
    );
    let restored = world.get_ticks::<SparseValue>(entity).unwrap();
    assert_eq!(restored.added, original_ticks.added);
    assert_eq!(restored.changed, original_ticks.changed);
    world.restore(&base);
    assert_eq!(
        world.get::<SparseValue>(entity).unwrap().0.value,
        [entity_index(&entities, entity)]
    );
}

#[test]
fn sparse_query_writes_with_old_ticks_invalidate_cold_read_only_bases() {
    let (mut world, counts, entities, chunk_rows) = sparse_world();
    let entity = entities[chunk_rows + 1];
    let old_tick = world.get_ticks::<SparseValue>(entity).unwrap().changed;
    // Make the initial values cold, allowing even a read-only baseline to be
    // reused under the tick-only implementation. Each mutation happens at the
    // same world tick as its read-only baseline and restores the old metadata.
    world.increment_change_tick();
    let required = world.query::<&mut SparseValue>();
    let optional = world.query::<Option<&mut SparseValue>>();
    let base = world.snapshot();
    let write_tick = world.change_tick();
    counts.reset();
    {
        let mut value = required.get_mut(&mut world, entity).unwrap();
        value.0.value[0] = 900;
        value.0.value.push(800);
        assert_eq!(value.0.value.pop(), Some(800));
        value.set_last_changed(old_tick);
    }
    assert_eq!(world.change_tick(), write_tick);
    let first = world.snapshot_incremental(Some(&base));
    counts.assert(chunk_rows, 0);
    // The optional fetch path must invalidate its own cold baseline too.
    let second_base = world.snapshot();
    let second_tick = world.change_tick();
    counts.reset();
    {
        let mut value = optional.get_mut(&mut world, entity).unwrap().unwrap();
        value.0.value.push(901);
        value.set_last_changed(old_tick);
    }
    assert_eq!(world.change_tick(), second_tick);
    let second = world.snapshot_incremental(Some(&second_base));
    counts.assert(chunk_rows, 0);
    world.restore(&second);
    assert_eq!(
        world.get::<SparseValue>(entity).unwrap().0.value,
        [900, 901]
    );
    world.restore(&first);
    assert_eq!(world.get::<SparseValue>(entity).unwrap().0.value, [900]);
    world.restore(&base);
    assert_eq!(
        world.get::<SparseValue>(entity).unwrap().0.value,
        [entity_index(&entities, entity)]
    );
}

#[test]
fn table_restore_invalidates_provenance_even_when_ticks_rewind() {
    let (mut world, counts, entities, _) = table_world();
    let base = world.snapshot_incremental(None);
    world.get_mut::<TableValue>(entities[0]).unwrap().0.value[0] = 123;
    let future = world.snapshot_incremental(Some(&base));
    world.restore(&base);
    counts.reset();
    let restored = world.snapshot_incremental(Some(&future));
    counts.assert(entities.len(), 0);
    counts.reset();
    let _clean = world.snapshot_incremental(Some(&restored));
    counts.assert(0, 0);
    world.restore(&future);
    assert_eq!(world.get::<TableValue>(entities[0]).unwrap().0.value, [123]);
    world.restore(&restored);
    assert_eq!(world.get::<TableValue>(entities[0]).unwrap().0.value, [0]);
}

#[test]
fn history_eviction_and_structural_migration_keep_clean_chunk_provenance() {
    let (mut world, counts, entities, _) = table_world();
    world.register_snapshot_component::<Tag>();
    let mut ring = SnapshotRing::new(3);
    ring.capture(0, &mut world);
    counts.reset();
    for frame in 1..8 {
        ring.capture(frame, &mut world);
        let _reconstructed = ring.get(frame).unwrap();
    }
    counts.assert(0, 0);
    assert_eq!(ring.frames().collect::<Vec<_>>(), [5, 6, 7]);
    // Entity migration has unchanged value/ticks but new storage provenance.
    // Delta encoding must retain the new buffer, even for an equal payload.
    world.insert(entities[0], Tag);
    ring.capture(8, &mut world);
    assert!(counts.clones.load(Ordering::Relaxed) > 0);
    counts.reset();
    for frame in 9..15 {
        ring.capture(frame, &mut world);
        let _reconstructed = ring.latest().unwrap();
    }
    counts.assert(0, 0);
    let last = ring.get(14).unwrap();
    world.restore(&last);
    assert!(world.get::<Tag>(entities[0]).is_some());
    for (i, entity) in entities.iter().enumerate() {
        assert_eq!(
            world.get::<TableValue>(*entity).unwrap().0.value,
            [i as u64]
        );
    }
}

#[test]
fn schema_reload_expires_reuse_until_explicit_contract_registration() {
    use core::alloc::Layout;
    use core::convert::Infallible;
    use prism_ecs::component::StorageType;
    use prism_ecs::world::DynamicComponentLayout;

    let (mut world, counts, entities, _) = table_world();
    // SAFETY: unused primitive component descriptors have no drop/clone glue.
    let old_layout = unsafe { DynamicComponentLayout::new(Layout::new::<u32>(), None, None, None) };
    let id = world
        .register_dynamic_component("snapshot.reuse.schema", StorageType::Table, old_layout)
        .unwrap();
    let base = world.snapshot_incremental(None);
    // SAFETY: migration output is a u64 matching this descriptor.
    let new_layout = unsafe { DynamicComponentLayout::new(Layout::new::<u64>(), None, None, None) };
    world
        .reload_dynamic_component(id, new_layout, |_, _, output| {
            // SAFETY: output was allocated for the new u64 layout.
            unsafe { output.write(0u64) };
            Ok::<_, Infallible>(())
        })
        .unwrap();
    counts.reset();
    let conservative = world.snapshot_incremental(Some(&base));
    counts.assert(entities.len(), 0);
    counts.reset();
    let still_conservative = world.snapshot_incremental(Some(&conservative));
    counts.assert(entities.len(), 0);
    world.register_snapshot_component_reusable::<TableValue>();
    counts.reset();
    let renewed = world.snapshot_incremental(Some(&still_conservative));
    counts.assert(entities.len(), 0);
    counts.reset();
    let _clean = world.snapshot_incremental(Some(&renewed));
    counts.assert(0, 0);
}

#[derive(Component)]
struct MissingClone;

#[test]
fn missing_clone_glue_does_not_advance_ticks_or_change_history() {
    let (mut world, counts, _, _) = table_world();
    let mut ring = SnapshotRing::new(2);
    ring.capture(1, &mut world);
    world.spawn(MissingClone);
    let before = world.change_tick();
    counts.reset();
    assert!(world.try_snapshot_incremental(None).is_err());
    assert!(ring.try_capture(2, &mut world).is_err());
    assert_eq!(world.change_tick(), before);
    assert_eq!(ring.frames().collect::<Vec<_>>(), [1]);
    counts.assert(0, 0);
}

#[derive(Component)]
struct CloneOnly {
    value: Arc<AtomicUsize>,
    clones: Arc<AtomicUsize>,
}

impl Clone for CloneOnly {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::Relaxed);
        Self {
            value: Arc::new(AtomicUsize::new(self.value.load(Ordering::Relaxed))),
            clones: self.clones.clone(),
        }
    }
}

#[test]
fn clone_only_interior_mutation_is_not_mistaken_for_an_immutable_chunk() {
    let mut world = World::new();
    world.register_snapshot_component::<CloneOnly>();
    let clones = Arc::new(AtomicUsize::new(0));
    let entity = world.spawn(CloneOnly {
        value: Arc::new(AtomicUsize::new(1)),
        clones: clones.clone(),
    });
    let base = world.snapshot_incremental(None);
    // Shared-reference mutation is legal for Clone-only types, not SnapshotValue.
    world
        .get::<CloneOnly>(entity)
        .unwrap()
        .value
        .store(2, Ordering::Relaxed);
    let changed = world.snapshot_incremental(Some(&base));
    assert_eq!(clones.load(Ordering::Relaxed), 2);
    world.restore(&base);
    assert_eq!(
        world
            .get::<CloneOnly>(entity)
            .unwrap()
            .value
            .load(Ordering::Relaxed),
        1
    );
    world.restore(&changed);
    assert_eq!(
        world
            .get::<CloneOnly>(entity)
            .unwrap()
            .value
            .load(Ordering::Relaxed),
        2
    );
}
