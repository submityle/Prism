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
//! tick cursors, relation policies and both ordered adjacency indexes. Resources
//! are strictly opt-in through `register_snapshot_resource`; infrastructure
//! resources are left untouched. Restore does not fire component hooks or
//! observers (a restore re-materializes
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
//! (adds value hashing). Register typed semantic comparison separately with
//! [`register_snapshot_component_equatable`](World::register_snapshot_component_equatable)
//! to enable equality checks and unchanged-cell delta reuse. A component
//! implementing [`SnapshotValue`] can additionally register immutable chunk
//! sharing with [`World::register_snapshot_component_reusable`], then capture
//! through [`World::snapshot_incremental`] or [`SnapshotRing::capture`].
//! Metadata still visits holders and resources/relations remain full captures.
//! A component
//! without clone glue that is resident on an
//! entity makes [`snapshot`](World::snapshot) refuse the capture, surfaced as
//! the error list of [`try_snapshot`](World::try_snapshot).

mod capture;
mod column;
mod delta;
#[cfg(feature = "determinism")]
mod determinism;
mod hash;
mod resource;
mod restore;
mod reuse;
mod rollback;

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::change::Tick;
use crate::component::{Component, ComponentId};
use crate::entity::{EntitiesState, Entity};
use crate::relation::{RelationKind, Relations};
use crate::world::World;

use column::SnapshotColumn;
use resource::SnapshotResource;

pub use delta::SnapshotDelta;
#[cfg(feature = "determinism")]
pub use determinism::{
    locate_divergence, FrameHash, FrameHashLog, SnapshotDivergence, TickDivergence,
};
pub use hash::FnvHasher;
pub use reuse::SnapshotValue;
pub use rollback::SnapshotRing;

/// An owned, frozen copy of a [`World`]'s entity + component + tick state
/// (design §14). See the [module docs](self) for scope and guarantees.
pub struct WorldSnapshot {
    /// Exact capture lineage; retained by deltas to validate every base before
    /// consulting slot references. Reconstructed targets retain this identity.
    identity: Arc<()>,
    /// Restore is registry-bound; a different world's dense component ids may
    /// denote entirely different Rust types, even when their layouts match.
    world_id: crate::world::WorldId,
    /// Runtime component ABI revision at capture, including value-only reloads.
    schema_generation: u64,
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
    /// Captured opt-in resources, ascending by [`ResourceId`](crate::resource::ResourceId)
    /// (design §14/§16.5). Only resources explicitly registered via
    /// [`register_snapshot_resource`](World::register_snapshot_resource) and
    /// currently holding a value appear here; see the [`resource`] module for
    /// the opt-in contract and honesty boundary.
    resources: Vec<SnapshotResource>,
    relations: SnapshotRelations,
}

/// Preserve both adjacency orders because targets()/sources() expose them.
/// Hash-map bucket order is deliberately excluded from the canonical view.
#[derive(Clone)]
struct SnapshotRelations {
    registry: Relations,
    kinds: Vec<(ComponentId, RelationKind)>,
    outgoing: Vec<(ComponentId, Entity, Vec<Entity>)>,
    incoming: Vec<(ComponentId, Entity, Vec<Entity>)>,
}

impl SnapshotRelations {
    fn capture(registry: &Relations) -> Self {
        let mut kinds: Vec<_> = registry
            .iter_kinds()
            .map(|(id, kind)| (id, *kind))
            .collect();
        kinds.sort_unstable_by_key(|(id, _)| id.index());
        let edges: Vec<_> = registry.index().iter_edges().collect();
        let mut sources: Vec<_> = edges.iter().map(|&(id, source, _)| (id, source)).collect();
        sources.sort_unstable_by_key(|(id, entity)| (id.index(), entity.to_bits()));
        sources.dedup();
        let outgoing = sources
            .into_iter()
            .map(|(id, source)| (id, source, registry.index().targets(id, source).to_vec()))
            .collect();
        let mut targets: Vec<_> = edges.iter().map(|&(id, _, target)| (id, target)).collect();
        targets.sort_unstable_by_key(|(id, entity)| (id.index(), entity.to_bits()));
        targets.dedup();
        let incoming = targets
            .into_iter()
            .map(|(id, target)| (id, target, registry.index().sources(id, target).to_vec()))
            .collect();
        Self {
            registry: registry.clone(),
            kinds,
            outgoing,
            incoming,
        }
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.kinds == other.kinds
            && self.outgoing == other.outgoing
            && self.incoming == other.incoming
    }

    fn hash_state(&self, hasher: &mut impl core::hash::Hasher) {
        use crate::relation::CleanupPolicy;
        fn policy(policy: CleanupPolicy) -> u8 {
            match policy {
                CleanupPolicy::Remove => 0,
                CleanupPolicy::Delete => 1,
                CleanupPolicy::Panic => 2,
            }
        }
        hasher.write_u64(self.kinds.len() as u64);
        for (id, kind) in &self.kinds {
            hasher.write_u32(id.index());
            hasher.write_u8(kind.fragmenting as u8);
            hasher.write_u8(kind.transitive as u8);
            hasher.write_u8(kind.exclusive as u8);
            hasher.write_u8(policy(kind.on_delete));
            hasher.write_u8(policy(kind.on_delete_target));
        }
        for adjacency in [&self.outgoing, &self.incoming] {
            hasher.write_u64(adjacency.len() as u64);
            for (id, entity, neighbors) in adjacency {
                hasher.write_u32(id.index());
                hasher.write_u64(entity.to_bits());
                hasher.write_u64(neighbors.len() as u64);
                for neighbor in neighbors {
                    hasher.write_u64(neighbor.to_bits());
                }
            }
        }
    }
}

impl WorldSnapshot {
    /// Whether every captured component/resource value contributes a semantic
    /// hash. A matching hash remains probabilistic, even when this is true.
    pub fn value_hash_complete(&self) -> bool {
        self.columns.iter().all(|col| col.hash.is_some())
            && self.resources.iter().all(|res| res.value_hash().is_some())
    }

    /// Number of captured directed relationship edges.
    pub fn relation_count(&self) -> usize {
        self.relations.registry.index().edge_count()
    }
    /// Runtime component schema revision captured with this snapshot.
    #[inline]
    pub fn schema_generation(&self) -> u64 {
        self.schema_generation
    }

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

    /// The number of opt-in resources captured with a value (design §14/§16.5).
    #[inline]
    pub fn resource_count(&self) -> usize {
        self.resources.len()
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
    /// Folds schema revision, both tick cursors, complete allocator state
    /// (including dead generations and free-list order), sorted entities and
    /// columns, storage/layout, per-cell ticks, opt-in resources, and relation
    /// policies plus both ordered adjacency indexes. World identity and capture
    /// lineage are deliberately excluded so independent runs can compare.
    ///
    /// Values contribute only through explicitly registered semantic hash glue;
    /// see [`value_hash_complete`](Self::value_hash_complete). A matching hash
    /// never proves equality: omitted values and collisions remain possible.
    /// Identical registry ids, storage ABI, and deterministic user `Hash`
    /// implementations are required for cross-run comparison.
    pub fn state_hash(&self) -> u64 {
        use core::hash::Hasher;
        let mut hasher = FnvHasher::new();
        hasher.write_u64(self.schema_generation);
        hasher.write_u64(self.change_tick.get() as u64);
        hasher.write_u64(self.last_change_tick.get() as u64);
        self.entities_state.hash_state(&mut hasher);
        hasher.write_u64(self.entities.len() as u64);
        for e in &self.entities {
            hasher.write_u64(e.to_bits());
        }
        hasher.write_u64(self.columns.len() as u64);
        for col in &self.columns {
            hasher.write_u64(col.component.index() as u64);
            hasher.write_u8(match col.storage {
                crate::component::StorageType::Table => 0,
                crate::component::StorageType::SparseSet => 1,
                crate::component::StorageType::Shared => 2,
            });
            hasher.write_u64(col.layout.size() as u64);
            hasher.write_u64(col.layout.align() as u64);
            hasher.write_u64(col.len() as u64);
            for i in 0..col.len() {
                let row = col.rows[i];
                hasher.write_u64(row as u64);
                hasher.write_u64(self.entities[row as usize].to_bits());
                hasher.write_u64(col.added[i].get() as u64);
                hasher.write_u64(col.changed[i].get() as u64);
                // SAFETY: `i < col.len()`.
                let hashed = unsafe { col.hash_value(i, &mut hasher) };
                // Fold a presence marker so a hashable and a non-hashable column
                // of equal structure never collide trivially.
                hasher.write_u8(hashed as u8);
            }
        }
        // Opt-in resources (design §14): fold each registered resource's id and,
        // for hashable ones, its value hash; clone-only resources fold only a
        // presence marker (honesty boundary — no value bytes available).
        hasher.write_u64(self.resources.len() as u64);
        for res in &self.resources {
            hasher.write_u64(res.id().index() as u64);
            match res.value_hash() {
                Some(h) => {
                    hasher.write_u8(1);
                    hasher.write_u64(h);
                }
                None => hasher.write_u8(0),
            }
        }
        self.relations.hash_state(&mut hasher);
        hasher.finish()
    }

    /// Structural + semantic equality against `other`: identical tick cursors,
    /// identical allocator state, identical entity list, and column-for-column
    /// identical holders, ticks, relation policies/ordered adjacency, and values.
    /// Values need separately registered typed comparison glue; missing glue
    /// makes this return false conservatively. Hash equality never establishes
    /// value equality. World identity is excluded so independent runs can compare.
    pub fn structurally_eq(&self, other: &WorldSnapshot) -> bool {
        if self.schema_generation != other.schema_generation
            || self.change_tick != other.change_tick
            || self.last_change_tick != other.last_change_tick
            || self.entities_state != other.entities_state
            || self.entities != other.entities
            || self.columns.len() != other.columns.len()
            || !self.relations.equivalent(&other.relations)
        {
            return false;
        }
        for (a, b) in self.columns.iter().zip(&other.columns) {
            if a.component != b.component
                || a.storage != b.storage
                || a.layout != b.layout
                || a.len() != b.len()
                || a.rows != b.rows
                || a.added != b.added
                || a.changed != b.changed
            {
                return false;
            }
            for i in 0..a.len() {
                if a.value_eq(i, b, i) != Some(true) {
                    return false;
                }
            }
        }
        // Typed resource comparison has the same conservative unknown case as
        // component comparison; matching hashes are insufficient evidence.
        if self.resources.len() != other.resources.len() {
            return false;
        }
        for (a, b) in self.resources.iter().zip(&other.resources) {
            if a.value_eq(b) != Some(true) {
                return false;
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

    /// Register resource `T` so its value is captured by
    /// [`snapshot`](World::snapshot) and restored by [`restore`](World::restore)
    /// (design §14/§16.5). Opt-in: resources not registered this way are left
    /// untouched by snapshot/restore, so infrastructure singletons never block
    /// or distort a rollback. Idempotent; returns the resource id.
    pub fn register_snapshot_resource<T: crate::resource::Resource + Clone>(
        &mut self,
    ) -> crate::resource::ResourceId {
        self.resources_mut().register_snapshot::<T>()
    }

    /// Register resource `T` with clone glue *and* deterministic value hashing,
    /// so it contributes its value to
    /// [`state_hash`](WorldSnapshot::state_hash) (design §14). Idempotent;
    /// returns the resource id.
    pub fn register_snapshot_resource_hashable<
        T: crate::resource::Resource + Clone + core::hash::Hash,
    >(
        &mut self,
    ) -> crate::resource::ResourceId {
        self.resources_mut().register_snapshot_hashable::<T>()
    }

    /// Capture a [`WorldSnapshot`] of the current world state.
    ///
    /// # Panics
    /// Panics if any component resident on an entity lacks clone glue; register
    /// it first with [`register_snapshot_component`](World::register_snapshot_component),
    /// or use [`try_snapshot`](World::try_snapshot) to handle the gap.
    /// Also panics if deferred entity reservations have not been flushed.
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
    ///
    /// # Panics
    /// Panics if deferred entity reservations have not been flushed.
    pub fn try_snapshot(&self) -> Result<WorldSnapshot, Vec<ComponentId>> {
        capture::capture(self)
    }

    /// Capture using unchanged immutable component chunks from `previous`.
    /// Pass None for the first capture. Reuse requires explicit registration
    /// through [`register_snapshot_component_reusable`](Self::register_snapshot_component_reusable).
    /// Other components still use their ordinary Clone capture contract.
    ///
    /// After successful capture this advances the world's change tick once,
    /// without changing the captured tick or the last-change baseline. The
    /// unique borrow and this boundary distinguish writes between captures even
    /// when callers do not advance a frame/system tick themselves. Restore,
    /// schema replacement and foreign-world bases invalidate capture reuse.
    /// Structural metadata, resources and relations are still captured in full.
    /// Sparse chunks scan membership/ticks, but do not read clean value payloads.
    /// Writes that deliberately bypass change detection require a fresh capture
    /// (`previous = None`) instead of reusing an earlier base.
    pub fn snapshot_incremental(&mut self, previous: Option<&WorldSnapshot>) -> WorldSnapshot {
        self.try_snapshot_incremental(previous)
            .unwrap_or_else(|missing| {
                panic!("incremental snapshot missing clone glue for {missing:?}")
            })
    }

    /// Fallible [`snapshot_incremental`](Self::snapshot_incremental). Missing
    /// clone glue leaves the world tick unchanged. Outstanding reservations
    /// panic, just as for [`try_snapshot`](Self::try_snapshot).
    pub fn try_snapshot_incremental(
        &mut self,
        previous: Option<&WorldSnapshot>,
    ) -> Result<WorldSnapshot, Vec<ComponentId>> {
        let snapshot = capture::capture_incremental(self, previous)?;
        self.increment_change_tick();
        Ok(snapshot)
    }

    /// Restore this world to the state frozen in `snapshot` (design §14).
    ///
    /// Rebuilds the entity allocator, archetype tables, and sparse sets to
    /// exactly reproduce the snapshot — including per-value added/changed ticks
    /// — together with relationship policies and ordered adjacency. Component
    /// hooks and observers do **not** fire. Registered snapshot resources are
    /// restored; other resources are left untouched. The component
    /// registry must be the same one the snapshot was captured against.
    ///
    /// # Panics
    /// Panics before changing the world if the snapshot belongs to another
    /// world or predates a successful runtime component schema replacement.
    pub fn restore(&mut self, snapshot: &WorldSnapshot) {
        restore::restore(self, snapshot);
    }
}

#[cfg(test)]
mod state_tests;
#[cfg(test)]
mod tests;
