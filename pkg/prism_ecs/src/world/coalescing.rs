//! Single-migration commit for unobservable adjacent structural commands.

use alloc::vec::Vec;

use super::World;
use crate::command::coalescing::CoalescedChange;
use crate::component::StorageType;
use crate::entity::Entity;

impl World {
    /// Commit a run already checked by command replay. All touched components
    /// are drop-free, not shared, and have no hooks, requirements or observers.
    /// Untouched components (including owned values) are relocated, not cloned.
    pub(crate) fn commit_coalesced_structural(
        &mut self,
        entity: Entity,
        changes: Vec<CoalescedChange>,
    ) {
        let Some(loc) = self.entities.location(entity) else {
            return;
        };
        let src_id = loc.archetype_id;
        let arch = self.archetypes.get(src_id).expect("placed live entity");
        let current = arch.components().clone();
        let bindings = arch.shared_bindings().to_vec();
        let mut final_set = current.clone();
        for change in &changes {
            if self.components.info(change.id).unwrap().storage() == StorageType::Table {
                final_set = if change.value.is_some() {
                    final_set.with(change.id)
                } else {
                    final_set.without(change.id)
                };
            }
        }
        let pairs = arch.relation_pairs().to_vec();
        let dst_id = self.archetypes.get_or_create_with_pairs(
            &final_set,
            &bindings,
            &pairs,
            &self.components,
        );
        let now = self.change_tick;
        let row = loc.row as usize;

        // Sparse changes never move table rows. Remove before re-inserting
        // when the lifetime was broken, so Added<T> is reset exactly once.
        for change in &changes {
            let info = self.components.info(change.id).unwrap();
            if info.storage() != StorageType::SparseSet {
                continue;
            }
            let set = self
                .sparse_sets
                .get_or_init(change.id, info.layout(), info.drop_fn());
            if change.removed || change.value.is_none() {
                set.remove(entity);
            }
            if let Some(value) = &change.value {
                // SAFETY: eligibility checked the component type and drop glue;
                // the bundle's bytes are moved into its correctly aligned set.
                unsafe {
                    set.insert(entity, value.as_ptr().cast(), now);
                }
            }
        }

        let reset_added = changes
            .iter()
            .any(|c| c.removed && c.value.is_some() && current.contains(c.id));
        if dst_id == src_id && !reset_added {
            let table = self.archetypes.get_mut(src_id).unwrap().table_mut();
            for change in &changes {
                if final_set.contains(change.id)
                    && let Some(value) = &change.value
                {
                    // SAFETY: this is an existing, drop-free table component
                    // with the same registered layout, at a live row.
                    unsafe {
                        table.column_mut(change.id).unwrap().replace(
                            row,
                            value.as_ptr().cast(),
                            now,
                        );
                    }
                }
            }
        } else if dst_id == src_id {
            // Net shape unchanged, but remove/reinsert starts a new lifetime.
            // Rebuild one row in the same table with exact ticks. Allocating
            // never relocates existing chunk blocks. The appended row swaps
            // back into the original row when the source is forgotten.
            let table = self.archetypes.get_mut(src_id).unwrap().table_mut();
            table.allocate(entity);
            for &id in final_set.ids() {
                let column = table.column_mut(id).unwrap();
                let change = changes.iter().find(|c| c.id == id && c.value.is_some());
                let added = column.added_tick(row);
                let changed = column.changed_tick(row);
                // SAFETY: every column has the original initialized row and a
                // vacant appended slot. Byte copies move (not clone) unchanged
                // owned values; the original is forgotten below. Changed values
                // are drop-free, so forgetting their outgoing value is valid.
                unsafe {
                    if let Some(change) = change {
                        column.push_with_ticks(
                            change.value.as_ref().unwrap().as_ptr().cast(),
                            if change.removed { now } else { added },
                            now,
                        );
                    } else {
                        let ptr = column.get_ptr(row);
                        column.push_with_ticks(ptr, added, changed);
                    }
                }
            }
            // SAFETY: all source columns were moved or overwritten drop-free;
            // the fully initialized appended row replaces the original row.
            let swapped = unsafe { table.swap_remove_row(row, current.ids()) };
            debug_assert_eq!(swapped, Some(entity));
            self.entities.set_location(entity, loc);
        } else {
            let dst_row = self
                .archetypes
                .get_mut(dst_id)
                .unwrap()
                .table_mut()
                .allocate(entity);
            {
                let (src, dst) = self.archetypes.get_pair_mut(src_id, dst_id);
                for &id in final_set.ids() {
                    let change = changes.iter().find(|c| c.id == id && c.value.is_some());
                    let source = src.table().column(id);
                    // SAFETY: source and destination share one registry. Every
                    // destination column is filled once, either by moving its
                    // source value (with ticks) or consuming staged bundle bytes.
                    // Outgoing touched values are drop-free; retained source
                    // values are forgotten by finish_move below.
                    unsafe {
                        let column = dst.table_mut().column_for_fill(id);
                        if let Some(change) = change {
                            let added = if change.removed {
                                now
                            } else {
                                source.map_or(now, |c| c.added_tick(row))
                            };
                            column.push_with_ticks(
                                change.value.as_ref().unwrap().as_ptr().cast(),
                                added,
                                now,
                            );
                        } else {
                            let source = source.expect("unchanged resident column");
                            column.push_with_ticks(
                                source.get_ptr(row),
                                source.added_tick(row),
                                source.changed_tick(row),
                            );
                        }
                    }
                }
            }
            // Removed and overwritten source values are drop-free. Every other
            // source value has moved exactly once; no drop glue may run here.
            self.finish_move(entity, loc, src_id, current.ids(), dst_id, dst_row);
        }

        if !self.owning_groups.is_empty() {
            let touched: Vec<_> = changes.iter().map(|c| c.id).collect();
            self.update_owning_groups(entity, &touched);
        }
    }
}
