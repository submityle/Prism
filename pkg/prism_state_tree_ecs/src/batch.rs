//! Reusable deterministic batch-gather storage for `StateTree` instances.
//!
//! The cursor semantics remain in the core runtime. This module only owns the
//! adapter-side entity gather buffer, so repeated `tick_all` calls do not
//! allocate a new entity vector after the first capacity growth.

use alloc::vec::Vec;

use prism_ecs::{entity::Entity, query::With, world::World};

use crate::{tick_entities, BatchTickError, BatchTickReport, StateTreeInstance};

/// Reusable gather scratch for deterministic whole-world `StateTree` batches.
///
/// The scratch is not a second runtime and carries no semantic state. It can
/// be owned by a schedule/resource and reused across frames. Entity handles
/// are gathered and sorted by generational identity before ticking.
pub struct BatchRuntime {
    entities: Vec<Entity>,
}

impl BatchRuntime {
    /// Creates an empty batch runtime with optional gather capacity reserved.
    pub fn with_capacity(capacity: usize) -> Result<Self, BatchTickError> {
        let mut entities = Vec::new();
        entities
            .try_reserve_exact(capacity)
            .map_err(|_| BatchTickError::AllocationFailed)?;
        Ok(Self { entities })
    }

    /// Creates an empty batch runtime without reserving memory.
    pub const fn new() -> Self {
        Self {
            entities: Vec::new(),
        }
    }

    /// Returns the retained gather capacity.
    pub fn capacity(&self) -> usize {
        self.entities.capacity()
    }

    /// Returns the number of handles retained from the last gather.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Returns whether the retained gather is empty.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Clears only the reusable gather scratch.
    pub fn clear(&mut self) {
        self.entities.clear();
    }

    /// Gathers, sorts and ticks every materialized `StateTree` instance.
    ///
    /// The entity scratch is reused. The returned report still owns its
    /// per-entry outcomes, because runtime failures include post-failure
    /// active configurations and callers must be able to retain the report
    /// independently of the next batch.
    pub fn tick_all(&mut self, world: &mut World) -> Result<BatchTickReport, BatchTickError> {
        self.entities.clear();
        let query = world.query_filtered::<Entity, With<StateTreeInstance>>();
        let count = query.iter(world).count();
        self.entities
            .try_reserve(count.saturating_sub(self.entities.capacity()))
            .map_err(|_| BatchTickError::AllocationFailed)?;
        self.entities.extend(query.iter(world));
        self.entities.sort_unstable();
        Ok(tick_entities(world, &self.entities))
    }

    /// Ticks an explicit ordered list without changing the retained gather.
    pub fn tick_entities(&mut self, world: &mut World, entities: &[Entity]) -> BatchTickReport {
        tick_entities(world, entities)
    }
}

impl Default for BatchRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use prism_ecs::world::World;
    use prism_state_tree_core::{MicrostepLimit, StateId, StateNode, TreeDef};

    use super::*;

    fn asset() -> crate::StateTreeAsset {
        let plan = TreeDef {
            root: StateId(0),
            nodes: vec![
                StateNode::compound(StateId(0), None, vec![StateId(1)], StateId(1), None),
                StateNode::atomic(StateId(1), Some(StateId(0))),
            ],
        }
        .compile()
        .unwrap();
        crate::StateTreeAsset::new(plan)
    }

    #[test]
    fn reuses_gather_capacity_and_keeps_deterministic_order() {
        let mut world = World::new();
        let a = world.spawn(StateTreeInstance::new(asset(), 2, MicrostepLimit(4)).unwrap());
        let b = world.spawn(StateTreeInstance::new(asset(), 2, MicrostepLimit(4)).unwrap());
        let mut batch = BatchRuntime::with_capacity(1).unwrap();
        let first = batch.tick_all(&mut world).unwrap();
        let capacity = batch.capacity();
        let second = batch.tick_all(&mut world).unwrap();

        assert!(capacity >= 2);
        assert_eq!(batch.capacity(), capacity);
        assert_eq!(first.len(), 2);
        assert_eq!(second.len(), 2);
        assert_eq!(first.entries()[0].entity, a.min(b));
        assert_eq!(first.entries()[1].entity, a.max(b));
    }
}
