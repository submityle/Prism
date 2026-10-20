//! Component-observer bindings for the glitch-free reaction graph.
//!
//! Sources collect the change bus's deterministic entity batches. A schedule
//! flush first collects every source, then settles derived nodes in dependency
//! order. Writes made by effects are observed on the next flush. Component
//! discovery uses the existing chunk/tick change bus; graph execution itself
//! visits only dirty nodes. The standalone graph remains usable without a World.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::component::{Component, ComponentId, StorageType};
use crate::entity::Entity;
use crate::reaction::{NodeId, ReactionGraph};
use crate::world::{World, WorldId};

/// A reaction node bound to one World, preventing cross-world handle reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReactionId {
    world: WorldId,
    node: NodeId,
}

/// The component bindings and graph owned by a World.
#[derive(Default)]
pub(crate) struct WorldReactions {
    graph: ReactionGraph<World>,
    sources: BTreeMap<ComponentId, ReactionId>,
    /// Frozen while a batch is executing or waiting for a panic retry.
    changes: BTreeMap<usize, Vec<Entity>>,
    /// New deliveries never modify a partially acknowledged batch.
    pending_changes: BTreeMap<usize, Vec<Entity>>,
    active_batch: bool,
    flushing: bool,
    deferred: Vec<NodeId>,
}

impl WorldReactions {
    pub(crate) fn is_flushing(&self) -> bool {
        self.flushing
    }

    #[cfg(test)]
    pub(crate) fn reaction_stats(&self) -> crate::reaction::ReactionStats {
        self.graph.stats()
    }
}

impl World {
    /// Get the unique reaction source for changes of `T` in this World.
    ///
    /// Registration starts a fresh tick: existing values do not fire, and
    /// writes made immediately after registration do. Shared components are
    /// immutable batch keys and do not support tick-based change subscription.
    pub fn reaction_source<T: Component>(&mut self) -> ReactionId {
        assert!(
            !self.reactions.flushing,
            "cannot register reactions while settling effects"
        );
        assert!(
            T::STORAGE != StorageType::Shared,
            "shared components do not have change ticks"
        );
        let component = self.components_mut().register::<T>();
        if let Some(&source) = self.reactions.sources.get(&component) {
            return source;
        }
        let source = ReactionId {
            world: self.id(),
            node: self.reactions.graph.add_source(),
        };
        self.reactions.sources.insert(component, source);
        self.observe_changed::<T, _>(move |ctx| {
            let changed = ctx.changed().to_vec();
            let batch = ctx
                .world
                .reactions
                .pending_changes
                .entry(source.node.index())
                .or_default();
            // Merge repeated writes within the next batch, while preserving
            // the captured batch needed by effects that have not succeeded.
            batch.extend(changed);
            batch.sort_unstable_by_key(|entity| (entity.index(), entity.generation()));
            batch.dedup();
            ctx.world.mark_reaction_dirty(source);
        });
        source
    }

    /// Register a derived effect. Dependencies must belong to this World.
    ///
    /// New nodes start clean. Their first run follows a source change or an
    /// explicit `mark_reaction_dirty`. Dependencies precede this new node, so
    /// the World graph is acyclic by construction.
    pub fn add_reaction<F>(&mut self, dependencies: &[ReactionId], effect: F) -> ReactionId
    where
        F: FnMut(&mut World) + Send + Sync + 'static,
    {
        assert!(
            !self.reactions.flushing,
            "cannot register reactions while settling effects"
        );
        let deps: Vec<_> = dependencies
            .iter()
            .map(|&id| {
                self.validate_reaction(id);
                id.node
            })
            .collect();
        ReactionId {
            world: self.id(),
            node: self.reactions.graph.add_derived(&deps, effect),
        }
    }

    /// Schedule a node and its downstream effects. During an effect, marks are
    /// queued for the next batch, preventing self-feeding loops in this one.
    /// Marks made between retries also wait until the unfinished batch succeeds.
    pub fn mark_reaction_dirty(&mut self, id: ReactionId) {
        self.validate_reaction(id);
        if self.reactions.flushing || self.reactions.active_batch {
            self.reactions.deferred.push(id.node);
        } else {
            self.reactions.graph.mark_dirty(id.node);
        }
    }

    /// The current batch's changed entities for a component source.
    ///
    /// An unchanged source and derived nodes return an empty slice. The slice
    /// is retained until every effect in this batch succeeds, including across
    /// panic retries. New source changes wait in the next batch. Copy the slice
    /// inside an effect if an external consumer needs to retain a delivery.
    /// This freezes entity identities, not component values: reads on retry
    /// observe the current World state.
    pub fn reaction_changes(&self, source: ReactionId) -> &[Entity] {
        self.validate_reaction(source);
        self.reactions
            .changes
            .get(&source.node.index())
            .map_or(&[], Vec::as_slice)
    }

    fn validate_reaction(&self, id: ReactionId) {
        assert_eq!(id.world, self.id(), "reaction belongs to a different World");
    }

    /// Settle pending effects without collecting component changes.
    ///
    /// Usually called automatically after `flush_changed_observers`. Returns
    /// the number of executed effects. Recursive flushes and flushes requested
    /// inside change-observer delivery are no-ops. The graph
    /// is restored even if an effect panics, and uncompleted effects can retry.
    /// A retry uses the original source batch and skips completed effects. New
    /// changes and explicit marks wait for a subsequent flush after that retry
    /// succeeds. Side effects performed before a panic are not rolled back.
    /// If an effect replaces the World, no further old effects run and the old
    /// graph is discarded without modifying the replacement's graph.
    pub fn flush_reactions(&mut self) -> usize {
        if self.reactions.flushing || self.change_observers.is_flushing() {
            return 0;
        }
        if !self.reactions.active_batch {
            self.reactions.changes = core::mem::take(&mut self.reactions.pending_changes);
            for node in self.reactions.deferred.drain(..) {
                self.reactions.graph.mark_dirty(node);
            }
        }
        if self.reactions.graph.dirty_count() == 0 {
            return 0;
        }
        self.increment_change_tick();
        let world_id = self.id();
        self.reactions.flushing = true;
        self.reactions.active_batch = true;
        let graph = core::mem::take(&mut self.reactions.graph);
        let mut guard = ReactionFlush {
            world: self,
            world_id,
            graph: Some(graph),
            completed: false,
        };
        let count = guard
            .graph
            .as_mut()
            .expect("owned reaction graph")
            .flush_while(guard.world, |world| world.id() == world_id);
        guard.completed = true;
        count
    }
}

/// Restores ownership on normal completion and on unwinding.
struct ReactionFlush<'w> {
    world: &'w mut World,
    world_id: WorldId,
    graph: Option<ReactionGraph<World>>,
    completed: bool,
}

impl Drop for ReactionFlush<'_> {
    fn drop(&mut self) {
        if self.world.id() != self.world_id {
            return;
        }
        self.world.reactions.graph = self.graph.take().expect("owned reaction graph");
        self.world.reactions.flushing = false;
        // An unfinished effect needs its captured source batch when retried.
        if self.completed {
            self.world.reactions.changes.clear();
            self.world.reactions.active_batch = false;
        }
    }
}
