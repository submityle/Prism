//! Capture a [`WorldSnapshot`] from a live [`World`] (design §14 snapshot).
//!
//! Ordinary capture clones live component values into independent storage.
//! Incremental capture may instead retain version-matched, immutable buffers
//! from an earlier snapshot. The public incremental entry point closes a tick
//! boundary before releasing its unique World borrow. The traversal is
//! deterministic — entities sorted ascending by [`Entity::to_bits`], components
//! ascending by [`ComponentId`] — so unchanged state yields identically ordered
//! snapshots and, with deterministic value glue, a stable
//! [`state_hash`](super::WorldSnapshot::state_hash).

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::change::Tick;
use crate::collections::HashMap;
use crate::component::{ComponentId, StorageType};
use crate::entity::Entity;
use crate::world::World;

use super::column::{SnapshotChunk, SnapshotColumn};
use super::reuse::{ChunkKey, ChunkOrigin};
use super::WorldSnapshot;

/// Attempt to capture `world`. Returns `Err` listing every component that is
/// resident on some entity but has no clone glue registered (via
/// [`register_snapshot_component`](World::register_snapshot_component)); such a
/// component cannot be snapshotted, so the capture is refused wholesale rather
/// than silently dropping state.
pub(super) fn capture(world: &World) -> Result<WorldSnapshot, Vec<ComponentId>> {
    capture_with_base(world, None, false)
}

pub(super) fn capture_incremental(
    world: &World,
    previous: Option<&WorldSnapshot>,
) -> Result<WorldSnapshot, Vec<ComponentId>> {
    capture_with_base(world, previous, true)
}

fn capture_with_base(
    world: &World,
    previous: Option<&WorldSnapshot>,
    sealed: bool,
) -> Result<WorldSnapshot, Vec<ComponentId>> {
    assert!(
        !world.entities.has_outstanding_reservations(),
        "snapshot requires deferred entity reservations to be flushed first"
    );
    let entities = collect_entities(world);

    // Validate clone glue for every component that actually has a holder.
    let mut missing: Vec<ComponentId> = Vec::new();
    let component_count = world.components.len();
    for i in 0..component_count {
        let id = ComponentId::new(i as u32);
        if component_has_holder(world, id) {
            let info = world.components.info(id).expect("registered component");
            if info.clone_fn().is_none() {
                missing.push(id);
            }
        }
    }
    if !missing.is_empty() {
        return Err(missing);
    }

    // Build one column per component that has holders, in ascending id order.
    let mut columns: Vec<SnapshotColumn> = Vec::new();
    for i in 0..component_count {
        let id = ComponentId::new(i as u32);
        let info = world.components.info(id).expect("registered component");
        let Some(clone) = info.clone_fn() else {
            continue;
        };
        let previous_column = previous
            .filter(|base| {
                base.world_id == world.id()
                    && base.schema_generation == world.components.schema_generation()
            })
            .and_then(|base| base.columns.iter().find(|column| column.component == id));
        if let Some(column) =
            capture_column(world, id, &entities, clone, info, previous_column, sealed)
        {
            columns.push(column);
        }
    }

    Ok(WorldSnapshot {
        identity: Arc::new(()),
        world_id: world.id(),
        schema_generation: world.components.schema_generation(),
        change_tick: world.change_tick,
        last_change_tick: world.last_change_tick,
        entities_state: world.entities.capture_state(),
        entities,
        columns,
        resources: super::resource::capture_resources(world),
        relations: super::SnapshotRelations::capture(&world.relations),
    })
}

/// Gather every entity that holds at least one component (table or sparse),
/// sorted ascending by [`Entity::to_bits`] and deduplicated. Component-less live
/// entities are intentionally omitted: the allocator state alone restores their
/// liveness and generation.
fn collect_entities(world: &World) -> Vec<Entity> {
    let mut entities: Vec<Entity> = Vec::new();
    for arch in world.archetypes.iter() {
        // Entities that hold only shared components still live in a table (whose
        // component set is empty, since a shared id does not fragment the set —
        // design §6), so include any archetype that carries either table columns
        // or shared-value bindings.
        if !arch.components().is_empty() || !arch.shared_bindings().is_empty() {
            entities.extend_from_slice(arch.table().entities());
        }
    }
    let component_count = world.components.len();
    for i in 0..component_count {
        let id = ComponentId::new(i as u32);
        if world
            .components
            .info(id)
            .is_some_and(|info| info.storage() == StorageType::SparseSet)
            && let Some(set) = world.sparse_sets.get(id)
        {
            entities.extend_from_slice(set.entities());
        }
    }
    entities.sort_unstable_by_key(|e| e.to_bits());
    entities.dedup();
    entities
}

/// Whether component `id` is resident on at least one entity.
fn component_has_holder(world: &World, id: ComponentId) -> bool {
    match world.components.info(id).map(|i| i.storage()) {
        Some(StorageType::Table) => world
            .archetypes
            .iter()
            .any(|arch| arch.contains(id) && !arch.table().is_empty()),
        Some(StorageType::SparseSet) => world.sparse_sets.get(id).is_some_and(|s| !s.is_empty()),
        // A shared component holds entities through every archetype that binds a
        // value for it; the entities themselves live in that archetype's table
        // (design §6 archetype split).
        Some(StorageType::Shared) => world
            .archetypes
            .iter()
            .any(|arch| arch.shared_binding(id).is_some() && !arch.table().is_empty()),
        None => false,
    }
}

/// Build the snapshot column for component `id`, or `None` if nothing holds it.
/// `entities` is the sorted global entity list used to resolve each holder's
/// row index; holders are emitted in ascending row order for determinism.
fn capture_column(
    world: &World,
    id: ComponentId,
    entities: &[Entity],
    clone: crate::component::CloneFn,
    info: &crate::component::ComponentInfo,
    previous: Option<&SnapshotColumn>,
    sealed: bool,
) -> Option<SnapshotColumn> {
    let storage = info.storage();
    let reusable = super::reuse::reusable(world, id);
    if reusable && storage != StorageType::Shared {
        return capture_reusable_column(world, id, entities, clone, info, previous, sealed);
    }
    // (row_index, raw value ptr, added, changed) for every holder.
    let mut holders: Vec<(u32, *const u8, Tick, Tick)> = Vec::new();

    match storage {
        StorageType::Table => {
            for arch in world.archetypes.iter() {
                if !arch.contains(id) {
                    continue;
                }
                let table = arch.table();
                let Some(column) = table.column(id) else {
                    continue;
                };
                for (row, &entity) in table.entities().iter().enumerate() {
                    let slot = entities
                        .binary_search_by_key(&entity.to_bits(), |e| e.to_bits())
                        .expect("holder entity is in the captured set");
                    // SAFETY: `row < table.len()` and `column` stores `id`.
                    let ptr = unsafe { column.get_ptr(row) } as *const u8;
                    holders.push((
                        slot as u32,
                        ptr,
                        column.added_tick(row),
                        column.changed_tick(row),
                    ));
                }
            }
        }
        StorageType::SparseSet => {
            let set = world.sparse_sets.get(id)?;
            for &entity in set.entities() {
                let slot = entities
                    .binary_search_by_key(&entity.to_bits(), |e| e.to_bits())
                    .expect("holder entity is in the captured set");
                // SAFETY: `entity` is present in `set`.
                let ptr = unsafe { set.get_ptr(entity) }.expect("present entity has a value");
                let added = set.added_tick(entity).expect("present entity has ticks");
                let changed = set.changed_tick(entity).expect("present entity has ticks");
                holders.push((slot as u32, ptr as *const u8, added, changed));
            }
        }
        StorageType::Shared => {
            // A shared value is interned once and shared by every entity in the
            // binding archetype; it carries no per-entity change ticks (design
            // §6: immutable, archetype-wide). We still record one cloned copy
            // per holder entity so restore can re-intern the exact value and
            // re-fragment the archetype, using a fixed zero tick so capture is
            // deterministic and a snapshot roundtrip preserves semantic state.
            for arch in world.archetypes.iter() {
                if arch.shared_binding(id).is_none() {
                    continue;
                }
                let arc = arch
                    .shared_arc(id)
                    .expect("archetype with a shared binding exposes its value");
                let value_ptr = arc.value_ptr();
                for &entity in arch.table().entities() {
                    let slot = entities
                        .binary_search_by_key(&entity.to_bits(), |e| e.to_bits())
                        .expect("holder entity is in the captured set");
                    holders.push((slot as u32, value_ptr, Tick::new(0), Tick::new(0)));
                }
            }
        }
    }

    if holders.is_empty() {
        return None;
    }

    // Deterministic order: ascending owning-entity index.
    holders.sort_unstable_by_key(|(slot, ..)| *slot);

    let mut column = SnapshotColumn::new(
        id,
        storage,
        info.layout(),
        clone,
        info.drop_fn(),
        info.snapshot_hash_fn(),
    );
    column.equality = super::hash::component_equality(world, id);
    if reusable {
        column.enable_sharing();
    }
    for (slot, ptr, added, changed) in holders {
        // SAFETY: `ptr` points at the live value of this component for the
        // holder entity; `push_cloned` only reads it to clone, never moves it.
        unsafe { column.push_cloned(ptr, added, changed, slot) };
    }
    Some(column)
}

#[cfg(test)]
mod sparse_revision_tests {
    use super::*;
    use crate::component::Component;
    use crate::world::snapshot::SnapshotValue;
    use alloc::vec;
    use core::mem::ManuallyDrop;

    #[derive(Clone, PartialEq)]
    struct SparseValue(Vec<u32>);

    impl Component for SparseValue {
        const STORAGE: StorageType = StorageType::SparseSet;
    }

    // SAFETY: the value owns its Vec and cloning isolates all mutable state.
    unsafe impl SnapshotValue for SparseValue {}

    #[test]
    fn sparse_reinsertion_with_identical_membership_and_ticks_recaptures_payload() {
        let mut world = World::new();
        let id = world.register_snapshot_component_reusable::<SparseValue>();
        world.register_snapshot_component_equatable::<SparseValue>();
        let first = world.spawn(SparseValue(vec![1]));
        let last = world.spawn(SparseValue(vec![2]));
        let ticks = world.get_ticks::<SparseValue>(last).unwrap();
        let base = world.snapshot_incremental(None);
        let set = world.sparse_sets.get_mut(id).unwrap();
        assert!(set.remove(last));
        let replacement = ManuallyDrop::new(SparseValue(vec![99]));
        // SAFETY: removed entity, initialized exact component type; transfer
        // ownership into the set while intentionally replaying identical ticks.
        unsafe {
            set.insert_with_ticks(
                last,
                core::ptr::from_ref(&*replacement).cast(),
                ticks.added,
                ticks.changed,
            );
        }
        assert_eq!(set.entities(), &[first, last]);
        assert_eq!(set.added_tick(last), Some(ticks.added));
        assert_eq!(set.changed_tick(last), Some(ticks.changed));
        let target = world.snapshot_incremental(Some(&base));
        let delta = base.diff(&target);
        // Both rows occupy the same logical chunk, whose identity changed.
        assert_eq!(delta.changed_cell_count(), 2);
        let replay = delta.apply(&base);
        world.restore(&replay);
        assert_eq!(world.get::<SparseValue>(first).unwrap().0, [1]);
        assert_eq!(world.get::<SparseValue>(last).unwrap().0, [99]);
        let restored = world.get_ticks::<SparseValue>(last).unwrap();
        assert_eq!(restored.added, ticks.added);
        assert_eq!(restored.changed, ticks.changed);
        world.restore(&base);
        assert_eq!(world.get::<SparseValue>(last).unwrap().0, [2]);
    }

    #[test]
    fn sparse_same_tick_overwrites_with_identical_ticks_recapture_each_value() {
        let mut world = World::new();
        let id = world.register_snapshot_component_reusable::<SparseValue>();
        world.register_snapshot_component_equatable::<SparseValue>();
        let entity = world.spawn(SparseValue(vec![1]));
        let ticks = world.get_ticks::<SparseValue>(entity).unwrap();
        let base = world.snapshot_incremental(None);
        let write_tick = world.change_tick();
        let mut previous = capture_with_base(&world, Some(&base), false).unwrap();
        for value in [2, 3] {
            let replacement = ManuallyDrop::new(SparseValue(vec![value]));
            // SAFETY: exact component type and owned replacement; preserve the
            // original tick to model replay without changing value identity.
            assert!(!unsafe {
                world.sparse_sets.get_mut(id).unwrap().insert(
                    entity,
                    core::ptr::from_ref(&*replacement).cast(),
                    ticks.changed,
                )
            });
            // Exercise repeated captures inside the same tick. Unlike the
            // public incremental boundary, this internal read-only capture
            // does not advance time or claim to seal the source.
            let target = capture_with_base(&world, Some(&previous), false).unwrap();
            assert_eq!(world.change_tick(), write_tick);
            let delta = previous.diff(&target);
            assert_eq!(delta.changed_cell_count(), 1);
            previous = delta.apply(&previous);
        }
        world.restore(&previous);
        assert_eq!(world.get::<SparseValue>(entity).unwrap().0, [3]);
        world.restore(&base);
        assert_eq!(world.get::<SparseValue>(entity).unwrap().0, [1]);
    }
}

/// Visit storage versions/membership first; clone value buffers only on a
/// version miss. Holder sorting and tick metadata remain deterministic and are
/// deliberately separate from immutable value ownership.
fn capture_reusable_column(
    world: &World,
    id: ComponentId,
    entities: &[Entity],
    clone: crate::component::CloneFn,
    info: &crate::component::ComponentInfo,
    previous: Option<&SnapshotColumn>,
    sealed: bool,
) -> Option<SnapshotColumn> {
    let make_column = || {
        let mut column = SnapshotColumn::new(
            id,
            info.storage(),
            info.layout(),
            clone,
            info.drop_fn(),
            info.snapshot_hash_fn(),
        );
        column.equality = super::hash::component_equality(world, id);
        column
    };
    let mut old = HashMap::new();
    // Sorting holders can interleave buffers. Deduplicate by Arc identity
    // before hashing metadata, especially sparse membership vectors.
    let mut seen = HashMap::new();
    if let Some(previous) = previous {
        for chunk in previous.shared_chunks() {
            if seen.insert(Arc::as_ptr(chunk) as usize, ()).is_some() {
                continue;
            }
            if let Some(origin) = &chunk.origin {
                if origin.reusable_in(world) {
                    old.insert(origin.key.clone(), chunk.clone());
                }
            }
        }
    }
    let origin = |key| {
        Some(ChunkOrigin {
            key,
            archetype_generation: world.archetype_generation(),
            capture_tick: world.change_tick(),
            sealed,
        })
    };
    let mut holders: Vec<(u32, Arc<SnapshotChunk>, usize, Tick, Tick)> = Vec::new();
    let entity_row = |entity: Entity| {
        entities
            .binary_search_by_key(&entity.to_bits(), |e| e.to_bits())
            .expect("holder entity is in captured set") as u32
    };
    match info.storage() {
        StorageType::Table => {
            for arch in world.archetypes.iter() {
                let table = arch.table();
                let Some(column) = table.column(id) else {
                    continue;
                };
                for (index, window) in table.iter_chunks().enumerate() {
                    let key = ChunkKey::Table {
                        incarnation: window.incarnation,
                        structural_epoch: window.meta.structural_epoch(),
                        version: column.chunk_version(index),
                        revision: column.chunk_revision(index),
                        rows: window.rows,
                    };
                    let chunk = if let Some(chunk) = old.get(&key) {
                        Arc::clone(chunk)
                    } else {
                        let mut buffer = make_column();
                        for row in window.first_row..window.first_row + window.rows {
                            // SAFETY: the live row stores this component and
                            // make_column captured exactly its value glue.
                            unsafe {
                                buffer.push_cloned(
                                    column.get_ptr(row),
                                    column.added_tick(row),
                                    column.changed_tick(row),
                                    0,
                                );
                            }
                        }
                        buffer.into_chunk(origin(key))
                    };
                    for local in 0..window.rows {
                        let row = window.first_row + local;
                        holders.push((
                            entity_row(table.entities()[row]),
                            chunk.clone(),
                            local,
                            column.added_tick(row),
                            column.changed_tick(row),
                        ));
                    }
                }
            }
        }
        StorageType::SparseSet => {
            let set = world.sparse_sets.get(id)?;
            let rows_per_chunk = crate::storage::rows_per_chunk(info.layout().size());
            for (index, members) in set.entities().chunks(rows_per_chunk).enumerate() {
                let added: Vec<_> = members
                    .iter()
                    .map(|&e| set.added_tick(e).unwrap())
                    .collect();
                let changed: Vec<_> = members
                    .iter()
                    .map(|&e| set.changed_tick(e).unwrap())
                    .collect();
                let revisions: Option<Vec<_>> =
                    members.iter().map(|&e| set.snapshot_revision(e)).collect();
                // Exhausted identities cannot safely be reused, including by
                // a later capture that receives this buffer as its baseline.
                let key = revisions.map(|revisions| ChunkKey::Sparse {
                    index,
                    entities: members.to_vec(),
                    added: added.clone(),
                    changed: changed.clone(),
                    revisions,
                });
                let chunk = if let Some(chunk) = key.as_ref().and_then(|key| old.get(key)) {
                    Arc::clone(chunk)
                } else {
                    let mut buffer = make_column();
                    for (local, &entity) in members.iter().enumerate() {
                        // SAFETY: a member of this set stores this component.
                        unsafe {
                            buffer.push_cloned(
                                set.get_ptr(entity).unwrap(),
                                added[local],
                                changed[local],
                                0,
                            )
                        };
                    }
                    buffer.into_chunk(key.and_then(origin))
                };
                for (local, &entity) in members.iter().enumerate() {
                    holders.push((
                        entity_row(entity),
                        chunk.clone(),
                        local,
                        added[local],
                        changed[local],
                    ));
                }
            }
        }
        StorageType::Shared => unreachable!("shared capture uses the interned-value path"),
    }
    if holders.is_empty() {
        return None;
    }
    holders.sort_unstable_by_key(|holder| holder.0);
    let mut column = make_column();
    column.enable_sharing();
    for (row, chunk, local, added, changed) in holders {
        column.push_shared(chunk, local, added, changed, row);
    }
    Some(column)
}
