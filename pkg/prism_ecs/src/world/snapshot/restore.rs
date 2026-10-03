//! Restore a live [`World`] to a captured [`WorldSnapshot`] (design §14).
//!
//! Restore is the inverse of [`capture`](super::capture): it tears down the
//! world's current archetype tables and sparse sets (their [`Drop`] frees the
//! live values), reinstates the entity allocator exactly (generations, liveness,
//! free list), then re-materialises every captured value — cloned *from* the
//! snapshot so the snapshot stays intact for repeated replays — with its exact
//! original added/changed ticks (design §14: a restore is tick-for-tick
//! equivalent to the capture, so change detection is deterministic across a
//! rollback).
//!
//! Determinism: entities are placed in ascending [`Entity::to_bits`] order (the
//! snapshot's entity order), and within an entity, table components fragment the
//! same archetype a fresh spawn would. Hooks/observers do not fire and
//! [`Resources`](crate::resource::Resources) are untouched (see [module docs](super)).

use alloc::vec;
use alloc::vec::Vec;

use crate::archetype::Archetypes;
use crate::component::{ComponentId, ComponentSet, StorageType};
use crate::entity::EntityLocation;
use crate::storage::SparseSets;
use crate::world::World;

use super::WorldSnapshot;

/// Restore `world` to `snapshot`. See [module docs](super) for scope.
pub(super) fn restore(world: &mut World, snapshot: &WorldSnapshot) {
    // 1. Tear down live storage. Dropping the old registries runs their `Drop`,
    //    freeing every live component value exactly once; the empty archetype is
    //    re-seeded by `Archetypes::new`.
    world.archetypes = Archetypes::new();
    world.sparse_sets = SparseSets::new();

    // 2. Reinstate the allocator: exact generations, liveness, and free list.
    //    Every live slot's location is reset to `EMPTY` and re-stamped below.
    world.entities.restore_state(&snapshot.entities_state);

    // 3. Restore the change-tick cursors so change detection resumes exactly.
    world.change_tick = snapshot.change_tick;
    world.last_change_tick = snapshot.last_change_tick;

    // 4. Invert the columnar layout into a per-entity plan: for each captured
    //    entity (by row index), the (column, slot) pairs it owns.
    let n = snapshot.entities.len();
    let mut per_entity: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (col_idx, col) in snapshot.columns.iter().enumerate() {
        for (slot, &row) in col.rows.iter().enumerate() {
            per_entity[row as usize].push((col_idx, slot));
        }
    }

    // 5. Re-materialise each captured entity in ascending order.
    for (row, &entity) in snapshot.entities.iter().enumerate() {
        let plan = &per_entity[row];

        // The archetype is fragmented by this entity's *table* components only;
        // sparse components are routed out of band (design §6). An entity that
        // holds only sparse components lands in the empty archetype.
        let table_set = ComponentSet::from_ids(
            plan.iter()
                .map(|&(ci, _)| &snapshot.columns[ci])
                .filter(|col| col.storage == StorageType::Table)
                .map(|col| col.component),
        );
        let archetype_id = world.archetypes.get_or_create(&table_set, &world.components);

        // Allocate the row, then fill each table column with a fresh clone and
        // its exact ticks.
        let trow = {
            let table = world
                .archetypes
                .get_mut(archetype_id)
                .expect("archetype just created")
                .table_mut();
            let trow = table.allocate(entity);
            for &(ci, slot) in plan {
                let col = &snapshot.columns[ci];
                if col.storage != StorageType::Table {
                    continue;
                }
                let (added, changed) = (col.added[slot], col.changed[slot]);
                let fill = table.column_for_fill(col.component);
                // SAFETY: `slot < col.len()`; the closure moves the cloned value
                // into the column exactly once via `push_with_ticks`.
                unsafe {
                    col.with_cloned_value(slot, |ptr| fill.push_with_ticks(ptr, added, changed));
                }
            }
            trow
        };

        // Sparse components: ensure the set exists (reusing the column's layout
        // and drop glue), then insert the cloned value with its exact ticks.
        for &(ci, slot) in plan {
            let col = &snapshot.columns[ci];
            if col.storage != StorageType::SparseSet {
                continue;
            }
            let (added, changed) = (col.added[slot], col.changed[slot]);
            let set = world
                .sparse_sets
                .get_or_init(col.component, col.layout, col.drop);
            // SAFETY: `slot < col.len()`; `entity` is placed exactly once (fresh
            // restore), so `insert_with_ticks`'s absent-entity contract holds;
            // the closure moves the cloned value in exactly once.
            unsafe {
                col.with_cloned_value(slot, |ptr| {
                    set.insert_with_ticks(entity, ptr, added, changed);
                });
            }
        }

        world.entities.set_location(
            entity,
            EntityLocation {
                archetype_id,
                row: trow as u32,
            },
        );
    }

    // 6. Component-less live entities were not captured as columns; re-place
    //    them into the empty archetype so their location matches a fresh spawn.
    for entity in world.entities.live_unplaced() {
        let empty = ComponentSet::from_ids(core::iter::empty::<ComponentId>());
        let archetype_id = world.archetypes.get_or_create(&empty, &world.components);
        let trow = world
            .archetypes
            .get_mut(archetype_id)
            .expect("empty archetype")
            .table_mut()
            .allocate(entity);
        world.entities.set_location(
            entity,
            EntityLocation {
                archetype_id,
                row: trow as u32,
            },
        );
    }

    // 7. Owning groups survive the restore as declarations, but their packed
    //    prefixes referenced the torn-down entities; rebuild each group's
    //    membership against the freshly re-materialised storage so iteration is
    //    correct post-rollback (design §6 / §14). No-op without groups.
    world.rebuild_all_owning_groups();
}
