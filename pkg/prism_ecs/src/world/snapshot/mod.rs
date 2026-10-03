//! Structured, differential world snapshots (design §14 / §16.5).
//!
//! A [`WorldSnapshot`] is an independent, owned copy of a [`World`]'s entity +
//! component + change-tick state at one instant. Snapshots power three AAA
//! subsystems that all share the same primitive:
//!
//! - **Rollback networking** (Quantum / GGPO form): keep the last confirmed
//!   authoritative frame as a snapshot, then on a late input `restore` it and
//!   replay inputs. A fixed-capacity [`SnapshotRing`] bounds rollback memory,
//!   and [`SnapshotDelta`] encodes frame-to-frame diffs (EnTT snapshot form) so
//!   history costs scale with *change*, not with world size.
//! - **Time-travel debugging** (`prism_ui_timetravel`) and **editor undo**:
//!   capture a ring of snapshots and `restore` to any of them.
//! - **Desync detection**: [`state_hash`](WorldSnapshot::state_hash) folds a
//!   deterministic per-frame hash two peers can compare (逐帧状态哈希去同步).
//!
//! # What a snapshot does and does not cover
//! A snapshot captures every live entity's allocator slot (generation +
//! liveness + free list), every component value held by an entity (table or
//! sparse) together with its exact added/changed ticks, and the world's change
//! tick cursors. It deliberately does **not** capture [`Resources`] — those use
//! `Box<dyn Any>` and need their own clone glue, scoped to a follow-up — nor
//! does `restore` fire component hooks or observers (a restore re-materializes
//! state rather than re-running gameplay). The [`Components`] registry itself is
//! left intact: snapshots are registry-bound (EnTT style), so capture and
//! restore must happen against the same component registry.
//!
//! # Opt-in glue
//! [`Component`] is neither [`Clone`] nor [`core::hash::Hash`], so snapshotting
//! is opt-in per component: call
//! [`register_snapshot_component`](World::register_snapshot_component) (clone
//! glue, required to be captured) and optionally
//! [`register_snapshot_component_hashable`](World::register_snapshot_component_hashable)
//! (adds value hashing). A component without clone glue that is resident on an
//! entity makes [`snapshot`](World::snapshot) refuse the capture, surfaced as
//! the error list of [`try_snapshot`](World::try_snapshot).

mod capture;
mod column;
mod delta;
mod hash;
mod restore;
mod rollback;

use alloc::vec::Vec;

use crate::change::Tick;
use crate::component::{Component, ComponentId};
use crate::entity::{Entity, EntitiesState};
use crate::world::World;

use column::SnapshotColumn;

pub use delta::SnapshotDelta;
pub use hash::FnvHasher;
pub use rollback::SnapshotRing;

/// An owned, frozen copy of a [`World`]'s entity + component + tick state
/// (design §14). See the [module docs](self) for scope and guarantees.
pub struct WorldSnapshot {
    /// The world's monotonic change counter at capture time.
    change_tick: Tick,
    /// The world's one-shot-read baseline tick at capture time.
    last_change_tick: Tick,
    /// The captured entity-allocator state (generations + liveness + free list).
    entities_state: EntitiesState,
    /// Every captured entity, ascending by [`Entity::to_bits`]. Columns index
    /// into this list by position (`SnapshotColumn::rows`).
    entities: Vec<Entity>,
    /// One column per component that has holders, ascending by [`ComponentId`].
    columns: Vec<SnapshotColumn>,
}

impl WorldSnapshot {
    /// The number of entities captured (those holding at least one component).
    #[inline]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// The number of live entities the captured allocator represents (including
    /// component-less live entities not present in [`entities`](Self::entities)).
    #[inline]
    pub fn live_entity_count(&self) -> u32 {
        self.entities_state.live_len()
    }

    /// The number of component columns captured.
    #[inline]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// The world change tick frozen at capture time.
    #[inline]
    pub fn change_tick(&self) -> Tick {
        self.change_tick
    }

    /// The captured entities, ascending by [`Entity::to_bits`].
    #[inline]
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Compute the deterministic per-frame state hash (design §14 逐帧状态哈希).
    ///
    /// The fold order is fixed and allocation-independent: first every entity's
    /// bits ascending, then every column ascending by [`ComponentId`], and
    /// within a column every holder ascending by its entity-row index, folding
    /// `(component index, owning entity bits)` structurally plus — for
    /// components registered as *hashable* — the value's bytes. Two worlds with
    /// identical structure therefore always agree structurally, and agree on
    /// values for every hash-registered component (see [`hash`] for the honesty
    /// boundary around non-hashable components such as those holding `f32`).
    pub fn state_hash(&self) -> u64 {
        use core::hash::Hasher;
        let mut hasher = FnvHasher::new();
        hasher.write_u64(self.change_tick.get() as u64);
        hasher.write_u64(self.entities_state.live_len() as u64);
        hasher.write_u64(self.entities.len() as u64);
        for e in &self.entities {
            hasher.write_u64(e.to_bits());
        }
        for col in &self.columns {
            hasher.write_u64(col.component.index() as u64);
            hasher.write_u64(col.len() as u64);
            for i in 0..col.len() {
                let row = col.rows[i];
                hasher.write_u64(row as u64);
                hasher.write_u64(self.entities[row as usize].to_bits());
                // SAFETY: `i < col.len()`.
                let hashed = unsafe { col.hash_value(i, &mut hasher) };
                // Fold a presence marker so a hashable and a non-hashable column
                // of equal structure never collide trivially.
                hasher.write_u8(hashed as u8);
            }
        }
        hasher.finish()
    }

    /// Structural + byte-exact equality against `other`: identical tick cursors,
    /// identical allocator state, identical entity list, and column-for-column
    /// identical holders, ticks, and value bytes. This is the invariant a
    /// snapshot roundtrip and a delta apply must satisfy (design §20: 快照
    /// roundtrip 等价), and unlike [`state_hash`](Self::state_hash) it compares
    /// *all* value bytes regardless of hash registration.
    pub fn structurally_eq(&self, other: &WorldSnapshot) -> bool {
        if self.change_tick != other.change_tick
            || self.last_change_tick != other.last_change_tick
            || self.entities_state != other.entities_state
            || self.entities != other.entities
            || self.columns.len() != other.columns.len()
        {
            return false;
        }
        for (a, b) in self.columns.iter().zip(&other.columns) {
            if a.component != b.component
                || a.len() != b.len()
                || a.rows != b.rows
                || a.added != b.added
                || a.changed != b.changed
            {
                return false;
            }
            for i in 0..a.len() {
                // SAFETY: `i < a.len() == b.len()`.
                let (ab, bb) = unsafe { (a.value_bytes(i), b.value_bytes(i)) };
                if ab != bb {
                    return false;
                }
            }
        }
        true
    }
}

impl World {
    /// Register `T` so it can be captured by [`snapshot`](World::snapshot) /
    /// restored by [`restore`](World::restore) (design §14/§16.5). Attaches
    /// type-erased clone glue; idempotent. Returns the component id.
    pub fn register_snapshot_component<T: Component + Clone>(&mut self) -> ComponentId {
        self.components_mut().register_cloneable::<T>()
    }

    /// Register `T` with clone glue *and* deterministic value-hash glue, so it
    /// contributes its value to [`state_hash`](WorldSnapshot::state_hash)
    /// (design §14). Idempotent. Returns the component id.
    pub fn register_snapshot_component_hashable<T: Component + Clone + core::hash::Hash>(
        &mut self,
    ) -> ComponentId {
        self.components_mut().register_hashable::<T>()
    }

    /// Capture a [`WorldSnapshot`] of the current world state.
    ///
    /// # Panics
    /// Panics if any component resident on an entity lacks clone glue; register
    /// it first with [`register_snapshot_component`](World::register_snapshot_component),
    /// or use [`try_snapshot`](World::try_snapshot) to handle the gap.
    pub fn snapshot(&self) -> WorldSnapshot {
        match capture::capture(self) {
            Ok(snap) => snap,
            Err(missing) => panic!(
                "World::snapshot() missing clone glue for {} resident component(s): {:?}; \
                 call register_snapshot_component::<T>() for each, or use try_snapshot()",
                missing.len(),
                missing
            ),
        }
    }

    /// Capture a [`WorldSnapshot`], or return every resident component id that
    /// lacks clone glue (and therefore cannot be snapshotted) on failure.
    pub fn try_snapshot(&self) -> Result<WorldSnapshot, Vec<ComponentId>> {
        capture::capture(self)
    }

    /// Restore this world to the state frozen in `snapshot` (design §14).
    ///
    /// Rebuilds the entity allocator, archetype tables, and sparse sets to
    /// exactly reproduce the snapshot — including per-value added/changed ticks
    /// — so a `restore` is byte-equivalent to the capture. Component hooks and
    /// observers do **not** fire, and [`Resources`](crate::resource::Resources)
    /// are left untouched; the [`Components`](crate::component::Components)
    /// registry must be the same one the snapshot was captured against.
    pub fn restore(&mut self, snapshot: &WorldSnapshot) {
        restore::restore(self, snapshot);
    }
}


#[cfg(test)]
mod tests;
