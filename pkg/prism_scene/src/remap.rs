//! The per-instance [`RemapTable`] (design §6.1).
//!
//! A template's internal entity-valued fields store [`LocalEntityId`]s, not
//! live handles. Instantiation reserves the entities once, then a `RemapTable`
//! maps each local id to *this instance's* live [`Entity`]. Because every
//! instance carries its own table, a template's internal references always
//! resolve inside that instance — instantiating the same template twice yields
//! two independent subtrees and never crosses a reference between them.
//!
//! Base-segment ids and reserved-segment ids (structural additions) are stored
//! in two dense runs, so lookup is O(1) with no filler handle needed.

use alloc::vec::Vec;

use prism_ecs::entity::Entity;

use crate::identity::LocalEntityId;

/// Maps a template-local id to the live ECS entity of one instance (design §6.1).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RemapTable {
    base: Vec<Entity>,
    reserved: Vec<Entity>,
}

impl RemapTable {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty table with room for `base_capacity` base entries.
    #[must_use]
    pub fn with_base_capacity(base_capacity: usize) -> Self {
        Self {
            base: Vec::with_capacity(base_capacity),
            reserved: Vec::new(),
        }
    }

    /// Appends the next base-segment mapping (id = current base length),
    /// returning the local id it was assigned.
    pub fn push_base(&mut self, entity: Entity) -> LocalEntityId {
        let id = LocalEntityId::new(self.base.len() as u32);
        self.base.push(entity);
        id
    }

    /// Records a reserved-segment mapping at `offset` (0-based). The offset must
    /// be at most the current reserved length; a larger offset is rejected so a
    /// gap can never be silently introduced.
    pub fn set_reserved(&mut self, offset: u32, entity: Entity) -> bool {
        let index = offset as usize;
        if index < self.reserved.len() {
            self.reserved[index] = entity;
            true
        } else if index == self.reserved.len() {
            self.reserved.push(entity);
            true
        } else {
            false
        }
    }

    /// Sets the mapping for `local`, routing to the base or reserved run by its
    /// segment. A base id must be within the current base length (or exactly
    /// one past it, which appends). Returns whether the mapping was stored.
    pub fn set(&mut self, local: LocalEntityId, entity: Entity) -> bool {
        if local.is_reserved() {
            self.set_reserved(local.index() - LocalEntityId::RESERVED_BASE, entity)
        } else {
            let index = local.index() as usize;
            if index < self.base.len() {
                self.base[index] = entity;
                true
            } else if index == self.base.len() {
                self.base.push(entity);
                true
            } else {
                false
            }
        }
    }

    /// Resolves a local id to its live entity, if mapped.
    #[must_use]
    pub fn get(&self, local: LocalEntityId) -> Option<Entity> {
        if local.is_reserved() {
            self.reserved
                .get((local.index() - LocalEntityId::RESERVED_BASE) as usize)
                .copied()
        } else {
            self.base.get(local.index() as usize).copied()
        }
    }

    /// Iterates mapped base entities in deterministic local-id order.
    pub fn base_entities(&self) -> impl ExactSizeIterator<Item = (LocalEntityId, Entity)> + '_ {
        self.base
            .iter()
            .copied()
            .enumerate()
            .map(|(index, entity)| (LocalEntityId::new(index as u32), entity))
    }

    /// Number of mapped base entities.
    #[must_use]
    pub fn base_len(&self) -> usize {
        self.base.len()
    }

    /// Number of mapped reserved (structurally-added) entities.
    #[must_use]
    pub fn reserved_len(&self) -> usize {
        self.reserved.len()
    }

    /// Total mapped entities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.base.len() + self.reserved.len()
    }

    /// Whether no entity is mapped.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.base.is_empty() && self.reserved.is_empty()
    }

    /// Clears all mappings, keeping capacity.
    pub fn clear(&mut self) {
        self.base.clear();
        self.reserved.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::RemapTable;
    use crate::identity::LocalEntityId;
    use prism_ecs::entity::Entity;

    fn entity(index: u32, generation: u32) -> Entity {
        Entity::from_bits((u64::from(generation) << 32) | u64::from(index))
            .expect("generation is non-zero")
    }

    #[test]
    fn push_base_assigns_sequential_ids() {
        let mut table = RemapTable::new();
        assert!(table.is_empty());
        let a = table.push_base(entity(10, 1));
        let b = table.push_base(entity(11, 1));
        assert_eq!(a, LocalEntityId::new(0));
        assert_eq!(b, LocalEntityId::new(1));
        assert_eq!(table.get(a), Some(entity(10, 1)));
        assert_eq!(table.get(b), Some(entity(11, 1)));
        assert_eq!(table.base_len(), 2);
        assert_eq!(table.get(LocalEntityId::new(2)), None);
    }

    #[test]
    fn reserved_segment_is_independent_of_base() {
        let mut table = RemapTable::new();
        table.push_base(entity(0, 1));
        table.push_base(entity(1, 1));

        assert!(table.set_reserved(0, entity(100, 2)));
        assert!(table.set_reserved(1, entity(101, 2)));
        assert_eq!(table.get(LocalEntityId::reserved(0)), Some(entity(100, 2)));
        assert_eq!(table.get(LocalEntityId::reserved(1)), Some(entity(101, 2)));
        // Reserved ids do not disturb the base run.
        assert_eq!(table.get(LocalEntityId::new(0)), Some(entity(0, 1)));
        assert_eq!(table.base_len(), 2);
        assert_eq!(table.reserved_len(), 2);
        assert_eq!(table.len(), 4);
    }

    #[test]
    fn set_routes_by_segment_and_rejects_gaps() {
        let mut table = RemapTable::new();
        assert!(table.set(LocalEntityId::new(0), entity(5, 1)));
        // Exactly one past the end appends.
        assert!(table.set(LocalEntityId::new(1), entity(6, 1)));
        // A gap is rejected.
        assert!(!table.set(LocalEntityId::new(3), entity(9, 1)));

        // Reserved: offset 0 ok, offset 2 (gap) rejected.
        assert!(table.set(LocalEntityId::reserved(0), entity(7, 1)));
        assert!(!table.set(LocalEntityId::reserved(2), entity(8, 1)));
    }

    #[test]
    fn overwrite_keeps_position() {
        let mut table = RemapTable::new();
        table.push_base(entity(0, 1));
        assert!(table.set(LocalEntityId::new(0), entity(0, 2)));
        assert_eq!(table.get(LocalEntityId::new(0)), Some(entity(0, 2)));
        assert_eq!(table.base_len(), 1);
    }
}
