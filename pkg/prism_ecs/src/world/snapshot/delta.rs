//! Frame-to-frame snapshot delta encoding (design §14: 世界快照 + delta,
//! EnTT snapshot 形态, chunk 增量编码控内存).
//!
//! A [`WorldSnapshot`] owns an independent copy of every captured value, so a
//! naïve rollback history of one full snapshot per frame costs memory ∝ world
//! size × frame count. A [`SnapshotDelta`] instead encodes the difference from a
//! *base* snapshot to a *target* snapshot: it keeps the target's lightweight
//! structural metadata (allocator state, entity list, per-column glue + holder
//! layout) in full, but only *owns* the component values of cells that actually
//! changed or were added relative to the base. Cells whose value and ticks are
//! unchanged are stored as a cheap back-reference into the base snapshot.
//!
//! This makes rollback history cost scale with *change*, not world size: keep
//! one anchor [`WorldSnapshot`] plus a chain of deltas, and
//! [`apply`](SnapshotDelta::apply) reconstructs the captured target's values and
//! ticks using its clone/comparison contracts (design §20: 快照 roundtrip 等价).
//! Explicit immutable chunks use buffer identity, without value comparisons or
//! extra clones. A new dirty buffer is retained even if some individual values
//! compare equal, preserving its capture provenance for subsequent frames.
//! Clone-only cells without typed comparison glue are conservatively fresh.
//! Metadata remains proportional to holders; these are not byte-cost metrics.

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::change::Tick;
use crate::collections::HashMap;
use crate::component::ComponentId;
use crate::entity::EntitiesState;

use super::column::SnapshotColumn;
use super::resource::{clone_resources, SnapshotResource};
use super::WorldSnapshot;

/// Where a single reconstructed cell's value comes from when
/// [`applying`](SnapshotDelta::apply) a delta.
enum CellSource {
    /// Reuse the value+ticks at this slot of the *base* snapshot's matching
    /// column (unchanged since the base).
    Base(u32),
    /// Clone the value+ticks at this slot of the delta's own `fresh` column
    /// (changed or newly added relative to the base).
    Fresh(u32),
}

/// The per-component diff between a base and target snapshot column. Owns only
/// the freshly-changed values (`fresh`); unchanged cells point back into the
/// base via [`CellSource::Base`].
struct ColumnDelta {
    /// Which component this column captures (matches the target column).
    component: ComponentId,
    /// Owning-entity index (into the target snapshot's entity list) for each
    /// holder, ascending — identical to the target column's `rows`.
    target_rows: Vec<u32>,
    /// Per-holder source selector, parallel to `target_rows`.
    source: Vec<CellSource>,
    /// The freshly changed/added values, in [`CellSource::Fresh`] index order.
    fresh: SnapshotColumn,
}

/// A differential encoding from a base [`WorldSnapshot`] to a target one
/// (design §14). Owns only the component values that changed relative to the
/// base, so a rollback history costs memory ∝ change. Reconstruct the target
/// with [`apply`](Self::apply) against the same base.
pub struct SnapshotDelta {
    base_identity: Arc<()>,
    target_identity: Arc<()>,
    base_world_id: crate::world::WorldId,
    base_schema_generation: u64,
    world_id: crate::world::WorldId,
    schema_generation: u64,
    /// The target world change tick.
    change_tick: Tick,
    /// The target world one-shot-read baseline tick.
    last_change_tick: Tick,
    /// The target entity-allocator state (generations + liveness + free list).
    entities_state: EntitiesState,
    /// The target entity list, ascending by [`Entity::to_bits`](crate::entity::Entity::to_bits).
    entities: Vec<crate::entity::Entity>,
    /// One delta per component column present in the target, ascending by id.
    columns: Vec<ColumnDelta>,
    /// The target's opt-in resources, copied in full (design §14/§16.5).
    /// Resources are singletons, so they are not delta-compressed — carrying
    /// them verbatim keeps a reconstructed snapshot value/tick-equivalent to a
    /// direct capture at negligible cost.
    resources: Vec<SnapshotResource>,
    relations: super::SnapshotRelations,
}

impl SnapshotDelta {
    /// The number of component columns the target holds.
    #[inline]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// The number of fresh logical cells (changed, newly added, or in a newly
    /// captured immutable chunk). Shared-buffer references need not own a new
    /// value allocation, so this is not an exact retained-byte count.
    pub fn changed_cell_count(&self) -> usize {
        self.columns.iter().map(|c| c.fresh.len()).sum()
    }

    /// The number of cells that were reused unchanged from the base snapshot.
    pub fn reused_cell_count(&self) -> usize {
        self.columns
            .iter()
            .map(|c| {
                c.source
                    .iter()
                    .filter(|s| matches!(s, CellSource::Base(_)))
                    .count()
            })
            .sum()
    }

    /// Reconstruct the target [`WorldSnapshot`] from this delta against `base`.
    ///
    /// The result preserves the values and ticks of the snapshot the delta was
    /// [diffed](WorldSnapshot::diff) from (design §20 roundtrip 等价): unchanged
    /// cells come from `base`'s matching column, changed/added cells from the
    /// delta's own `fresh` store. Immutable buffers share ownership; Clone-only
    /// cells are cloned into independently owned storage.
    ///
    /// # Panics
    /// Panics before cloning any value if `base` does not have the exact capture
    /// lineage this delta was diffed against. An independently captured snapshot
    /// is rejected even if its hash and values match. Applying a preceding delta
    /// preserves its target lineage, so reconstructed snapshots support chains.
    pub fn apply(&self, base: &WorldSnapshot) -> WorldSnapshot {
        assert!(
            Arc::ptr_eq(&base.identity, &self.base_identity),
            "delta requires the exact base snapshot it was diffed against"
        );
        assert_eq!(
            base.world_id, self.base_world_id,
            "delta base belongs to another world"
        );
        assert_eq!(
            base.schema_generation, self.base_schema_generation,
            "delta base schema mismatch"
        );
        let mut columns = Vec::with_capacity(self.columns.len());
        for cd in &self.columns {
            let base_col = base.columns.iter().find(|c| c.component == cd.component);
            let mut col = cd.fresh.empty_like();
            for (i, &target_row) in cd.target_rows.iter().enumerate() {
                match cd.source[i] {
                    CellSource::Base(bs) => {
                        let bc = base_col.expect(
                            "SnapshotDelta::apply: base snapshot is missing a column a Base cell \
                             references; apply against the exact base it was diffed from",
                        );
                        // SAFETY: `bs` indexes a holder of `bc` recorded at diff
                        // time; `bc` shares this column's component glue/layout.
                        unsafe { col.push_cloned_from(bc, bs as usize, target_row) };
                    }
                    CellSource::Fresh(fs) => {
                        // SAFETY: `fs` indexes a value stored in `cd.fresh` at
                        // diff time; `fresh` shares this column's glue/layout.
                        unsafe { col.push_cloned_from(&cd.fresh, fs as usize, target_row) };
                    }
                }
            }
            columns.push(col);
        }

        WorldSnapshot {
            identity: self.target_identity.clone(),
            world_id: self.world_id,
            schema_generation: self.schema_generation,
            change_tick: self.change_tick,
            last_change_tick: self.last_change_tick,
            entities_state: self.entities_state.clone(),
            entities: self.entities.clone(),
            columns,
            resources: clone_resources(&self.resources),
            relations: self.relations.clone(),
        }
    }
}

impl WorldSnapshot {
    /// Encode the difference from `self` (the base) to `target` as a
    /// [`SnapshotDelta`] (design §14). Unchanged immutable buffer cells or
    /// Clone-only cells proven equal by typed comparison back-reference the
    /// base. New immutable buffers remain fresh, preserving storage provenance.
    /// Missing/incompatible comparison glue on Clone-only cells is conservative.
    /// Components the target no longer holds simply vanish (not emitted).
    pub fn diff(&self, target: &WorldSnapshot) -> SnapshotDelta {
        let mut columns = Vec::with_capacity(target.columns.len());
        let same_schema =
            self.world_id == target.world_id && self.schema_generation == target.schema_generation;

        // Target columns are already ascending by component id, so the delta's
        // columns inherit that deterministic order.
        for tc in &target.columns {
            let base_col = same_schema
                .then(|| self.columns.iter().find(|c| c.component == tc.component))
                .flatten();

            // Map base owning-entity bits -> base holder slot, for O(1) reuse
            // lookup. Empty when the base lacks this component entirely.
            let mut base_slot_of: HashMap<u64, usize> = HashMap::new();
            if let Some(bc) = base_col {
                for (bslot, &brow) in bc.rows.iter().enumerate() {
                    base_slot_of.insert(self.entities[brow as usize].to_bits(), bslot);
                }
            }

            let mut fresh = tc.empty_like();
            let mut target_rows: Vec<u32> = Vec::with_capacity(tc.rows.len());
            let mut source: Vec<CellSource> = Vec::with_capacity(tc.rows.len());

            for (tslot, &trow) in tc.rows.iter().enumerate() {
                let ent_bits = target.entities[trow as usize].to_bits();
                let reuse = base_slot_of.get(&ent_bits).copied().filter(|&bslot| {
                    let bc =
                        base_col.expect("base_slot_of is only populated when base_col is Some");
                    let ticks_eq = bc.added[bslot] == tc.added[tslot]
                        && bc.changed[bslot] == tc.changed[tslot];
                    ticks_eq
                        && (bc.shares_value(bslot, tc, tslot)
                            || (!tc.uses_shared_values()
                                && bc.value_eq(bslot, tc, tslot) == Some(true)))
                });

                match reuse {
                    Some(bslot) => {
                        target_rows.push(trow);
                        source.push(CellSource::Base(bslot as u32));
                    }
                    None => {
                        let fresh_idx = fresh.len() as u32;
                        // SAFETY: `tslot < tc.len()`; `fresh` shares `tc`'s glue
                        // and layout, so the clone is type-correct.
                        unsafe { fresh.push_cloned_from(tc, tslot, trow) };
                        target_rows.push(trow);
                        source.push(CellSource::Fresh(fresh_idx));
                    }
                }
            }

            columns.push(ColumnDelta {
                component: tc.component,
                target_rows,
                source,
                fresh,
            });
        }

        SnapshotDelta {
            base_identity: self.identity.clone(),
            target_identity: target.identity.clone(),
            base_world_id: self.world_id,
            base_schema_generation: self.schema_generation,
            world_id: target.world_id,
            schema_generation: target.schema_generation,
            change_tick: target.change_tick,
            last_change_tick: target.last_change_tick,
            entities_state: target.entities_state.clone(),
            entities: target.entities.clone(),
            columns,
            resources: clone_resources(&target.resources),
            relations: target.relations.clone(),
        }
    }
}
