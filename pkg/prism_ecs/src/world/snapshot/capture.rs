//! Capture a [`WorldSnapshot`] from a live [`World`] (design §14 snapshot).
//!
//! Capture is read-only: it clones each live component value (through the
//! component's [`CloneFn`] glue) into the snapshot's own owned storage, leaving
//! the world untouched so prediction can keep advancing it. The traversal is
//! deterministic — entities sorted ascending by [`Entity::to_bits`], components
//! ascending by [`ComponentId`] — so the same world always yields byte-identical
//! snapshots and a stable [`state_hash`](super::WorldSnapshot::state_hash).

use alloc::vec::Vec;

use crate::change::Tick;
use crate::component::{ComponentId, StorageType};
use crate::entity::Entity;
use crate::world::World;

use super::column::SnapshotColumn;
use super::WorldSnapshot;

/// Attempt to capture `world`. Returns `Err` listing every component that is
/// resident on some entity but has no clone glue registered (via
/// [`register_snapshot_component`](World::register_snapshot_component)); such a
/// component cannot be snapshotted, so the capture is refused wholesale rather
/// than silently dropping state.
pub(super) fn capture(world: &World) -> Result<WorldSnapshot, Vec<ComponentId>> {
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
        if let Some(column) = capture_column(world, id, &entities, clone, info) {
            columns.push(column);
        }
    }

    Ok(WorldSnapshot {
        change_tick: world.change_tick,
        last_change_tick: world.last_change_tick,
        entities_state: world.entities.capture_state(),
        entities,
        columns,
        resources: super::resource::capture_resources(world),
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
) -> Option<SnapshotColumn> {
    let storage = info.storage();
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
            // deterministic and a snapshot roundtrip is byte-for-byte stable.
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
    for (slot, ptr, added, changed) in holders {
        // SAFETY: `ptr` points at the live value of this component for the
        // holder entity; `push_cloned` only reads it to clone, never moves it.
        unsafe { column.push_cloned(ptr, added, changed, slot) };
    }
    Some(column)
}
