//! Cell ↔ entity bucketing and weak cross-cell references (design §13.1; §22
//! risk 4).
//!
//! [`CellStreamer`](crate::partition::cell::CellStreamer) decides *which cells*
//! should load or evict, but it is deliberately `World`-independent and knows
//! nothing about the entities those cells contain. Two bookkeeping pieces close
//! that gap on the CPU side, both pure and fully testable:
//!
//! * [`CellEntityIndex`] — a bidirectional map between a
//!   [`CellCoord`] and the entities currently resident in it. When the streamer
//!   emits a [`StreamingDelta`](crate::partition::cell::StreamingDelta)
//!   `to_unload`, the scene drains exactly those cells'
//!   entities from the index and despawns them — no full-world scan, and in a
//!   deterministic order (design §14).
//! * [`WeakEntity`] / [`WeakRefs`] — the "stable id + weak handle" discipline
//!   from design §22 risk 4. A streamed-out entity is despawned, so any handle
//!   another cell kept to it is now stale. Because [`Entity`] is a generational
//!   index, a weak handle *resolves* against the live [`World`] and fails safely
//!   (generation mismatch → `None`) instead of dangling or aliasing a recycled
//!   slot.
//!
//! Neither type performs I/O or despawns on its own: they are the accounting
//! the owning scene drives around the streamer's schedule.

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::entity::Entity;
use crate::partition::cell::CellCoord;
use crate::world::World;

/// A non-owning, generation-checked handle to an entity that may have been
/// streamed out (design §13.1; §22 risk 4).
///
/// A `WeakEntity` is just an [`Entity`] with explicit "may be dead" semantics.
/// Resolving it against the authoritative [`World`] returns the live handle only
/// if the slot still carries the recorded generation; once the target is
/// despawned (e.g. its cell unloaded) the generation no longer matches and
/// resolution yields `None`. This is what makes cross-cell references safe to
/// hold across streaming without dangling or silently retargeting a recycled
/// slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WeakEntity {
    entity: Entity,
}

impl WeakEntity {
    /// Wraps a (possibly already-dead) entity handle as a weak reference.
    #[inline]
    pub const fn new(entity: Entity) -> Self {
        Self { entity }
    }

    /// The raw stored handle, regardless of whether it is still live.
    ///
    /// Prefer [`resolve`](Self::resolve) when you intend to touch the entity;
    /// this accessor is for serialization and equality bookkeeping.
    #[inline]
    pub const fn entity(self) -> Entity {
        self.entity
    }

    /// Whether the referenced entity is still live in `world`.
    #[inline]
    pub fn is_alive(self, world: &World) -> bool {
        world.contains(self.entity)
    }

    /// Resolves to the live handle, or `None` if the entity has been despawned
    /// (generation mismatch) or its slot recycled.
    #[inline]
    pub fn resolve(self, world: &World) -> Option<Entity> {
        if world.contains(self.entity) {
            Some(self.entity)
        } else {
            None
        }
    }
}

impl From<Entity> for WeakEntity {
    #[inline]
    fn from(entity: Entity) -> Self {
        Self::new(entity)
    }
}

/// A collection of [`WeakEntity`] references, as a cell/entity typically keeps
/// to neighbours that may live in other (possibly streamed-out) cells
/// (design §22 risk 4).
///
/// The list stores handles verbatim; it never despawns. Call
/// [`prune`](Self::prune) after a streaming pass to drop references whose
/// targets were evicted, or iterate the live subset with
/// [`iter_alive`](Self::iter_alive) without mutating.
#[derive(Clone, Debug, Default)]
pub struct WeakRefs {
    refs: Vec<WeakEntity>,
}

impl WeakRefs {
    /// An empty reference list.
    #[inline]
    pub const fn new() -> Self {
        Self { refs: Vec::new() }
    }

    /// Appends a weak reference. Duplicates are allowed; use
    /// [`contains`](Self::contains) to dedup at call sites that need it.
    #[inline]
    pub fn push(&mut self, weak: impl Into<WeakEntity>) {
        self.refs.push(weak.into());
    }

    /// Number of stored references, live or dead.
    #[inline]
    pub fn len(&self) -> usize {
        self.refs.len()
    }

    /// Whether the list holds no references.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }

    /// Drops every reference.
    #[inline]
    pub fn clear(&mut self) {
        self.refs.clear();
    }

    /// Whether `entity` is referenced (by raw handle, ignoring liveness).
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.refs.iter().any(|w| w.entity() == entity)
    }

    /// Removes every reference whose target is no longer live in `world`,
    /// returning how many were dropped. Order of the survivors is preserved.
    pub fn prune(&mut self, world: &World) -> usize {
        let before = self.refs.len();
        self.refs.retain(|w| w.is_alive(world));
        before - self.refs.len()
    }

    /// Iterates the raw stored handles (live or dead).
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = WeakEntity> + '_ {
        self.refs.iter().copied()
    }

    /// Iterates only the references that still resolve in `world`, as live
    /// [`Entity`] handles.
    pub fn iter_alive<'w>(&'w self, world: &'w World) -> impl Iterator<Item = Entity> + 'w {
        self.refs.iter().filter_map(move |w| w.resolve(world))
    }
}

/// A bidirectional index between world-partition cells and the entities
/// resident in them (design §13.1).
///
/// The scene maintains this as it spawns entities ([`assign`](Self::assign)) and
/// as despawns happen ([`remove`](Self::remove)). On a streaming eviction it
/// drains the affected cells ([`take_cell`](Self::take_cell) /
/// [`take_cells`](Self::take_cells)) to learn exactly which entities to despawn,
/// in a deterministic order, without scanning the whole world.
///
/// The index is pure bookkeeping: it stores [`Entity`] handles only and never
/// touches a [`World`]. The `of_entity` reverse map makes moves and removals
/// `O(1)` amortised instead of a per-cell scan.
#[derive(Clone, Debug, Default)]
pub struct CellEntityIndex {
    /// cell -> entities currently assigned to it (unsorted; sorted on output).
    by_cell: HashMap<CellCoord, Vec<Entity>>,
    /// entity -> its current cell, so moves/removals need no scan.
    of_entity: HashMap<Entity, CellCoord>,
}

impl CellEntityIndex {
    /// An empty index.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Assigns `entity` to `cell`, moving it out of any previous cell.
    ///
    /// Returns the entity's previous cell if it was already tracked (which may
    /// equal `cell`, in which case the call is a no-op), or `None` if this is
    /// the first time the entity is seen.
    pub fn assign(&mut self, entity: Entity, cell: CellCoord) -> Option<CellCoord> {
        match self.of_entity.get(&entity).copied() {
            Some(prev) if prev == cell => Some(prev),
            Some(prev) => {
                self.detach_from_cell(entity, prev);
                self.by_cell.entry(cell).or_default().push(entity);
                self.of_entity.insert(entity, cell);
                Some(prev)
            }
            None => {
                self.by_cell.entry(cell).or_default().push(entity);
                self.of_entity.insert(entity, cell);
                None
            }
        }
    }

    /// Removes `entity` from the index (e.g. on despawn), returning the cell it
    /// was in, or `None` if it was not tracked.
    pub fn remove(&mut self, entity: Entity) -> Option<CellCoord> {
        let cell = self.of_entity.remove(&entity)?;
        self.detach_from_cell(entity, cell);
        Some(cell)
    }

    /// The cell `entity` is currently assigned to, or `None` if untracked.
    #[inline]
    pub fn cell_of(&self, entity: Entity) -> Option<CellCoord> {
        self.of_entity.get(&entity).copied()
    }

    /// Whether `entity` is tracked by the index.
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.of_entity.contains_key(&entity)
    }

    /// The entities currently resident in `cell`, sorted by handle for a
    /// deterministic order (design §14). Returns an empty vector for an
    /// untracked or empty cell.
    pub fn entities_in(&self, cell: CellCoord) -> Vec<Entity> {
        let mut out = self.by_cell.get(&cell).cloned().unwrap_or_default();
        out.sort_unstable();
        out
    }

    /// Drains every entity assigned to `cell`, removing them from the index and
    /// returning them sorted by handle. Used on cell unload to despawn exactly
    /// that cell's entities.
    pub fn take_cell(&mut self, cell: CellCoord) -> Vec<Entity> {
        let mut out = self.by_cell.remove(&cell).unwrap_or_default();
        for &entity in &out {
            self.of_entity.remove(&entity);
        }
        out.sort_unstable();
        out
    }

    /// Drains several cells at once (e.g. a streaming delta's `to_unload`),
    /// returning all their entities. The result is grouped by cell in sorted
    /// `(x, y, z)` cell order and sorted by handle within each cell, so the
    /// despawn order is fully deterministic regardless of input ordering or
    /// map iteration (design §14). Duplicate input cells are processed once.
    pub fn take_cells(&mut self, cells: &[CellCoord]) -> Vec<Entity> {
        let mut ordered: Vec<CellCoord> = cells.to_vec();
        ordered.sort_unstable_by(|a, b| a.x.cmp(&b.x).then(a.y.cmp(&b.y)).then(a.z.cmp(&b.z)));
        ordered.dedup();

        let mut out = Vec::new();
        for cell in ordered {
            out.extend(self.take_cell(cell));
        }
        out
    }

    /// Number of cells that currently hold at least one entity.
    #[inline]
    pub fn cell_count(&self) -> usize {
        self.by_cell.len()
    }

    /// Total number of tracked entities across all cells.
    #[inline]
    pub fn entity_count(&self) -> usize {
        self.of_entity.len()
    }

    /// Whether the index tracks no entities.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.of_entity.is_empty()
    }

    /// Removes `entity` from `cell`'s bucket, dropping the bucket if it becomes
    /// empty. The caller is responsible for the `of_entity` side.
    fn detach_from_cell(&mut self, entity: Entity, cell: CellCoord) {
        if let Some(bucket) = self.by_cell.get_mut(&cell) {
            if let Some(pos) = bucket.iter().position(|&e| e == entity) {
                bucket.swap_remove(pos);
            }
            if bucket.is_empty() {
                self.by_cell.remove(&cell);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(x: i32, y: i32, z: i32) -> CellCoord {
        CellCoord::new(x, y, z)
    }

    #[test]
    fn weak_entity_resolves_until_despawn() {
        let mut world = World::new();
        let e = world.spawn(());
        let weak = WeakEntity::new(e);

        assert!(weak.is_alive(&world));
        assert_eq!(weak.resolve(&world), Some(e));
        assert_eq!(weak.entity(), e);

        assert!(world.despawn(e));
        assert!(!weak.is_alive(&world));
        assert_eq!(weak.resolve(&world), None);
    }

    #[test]
    fn weak_entity_does_not_alias_recycled_slot() {
        let mut world = World::new();
        let a = world.spawn(());
        let weak = WeakEntity::from(a);
        assert!(world.despawn(a));

        // Recycling the slot bumps its generation; the stale weak handle must
        // not resolve to the new occupant.
        let b = world.spawn(());
        assert_eq!(a.index(), b.index(), "slot should be recycled for this test");
        assert_ne!(a.generation(), b.generation());
        assert_eq!(weak.resolve(&world), None);
        assert!(world.contains(b));
    }

    #[test]
    fn weak_refs_prune_drops_dead_only() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());

        let mut refs = WeakRefs::new();
        refs.push(a);
        refs.push(b);
        refs.push(c);
        assert_eq!(refs.len(), 3);
        assert!(refs.contains(b));

        assert!(world.despawn(b));
        let live: Vec<Entity> = refs.iter_alive(&world).collect();
        assert_eq!(live, alloc::vec![a, c]);

        let dropped = refs.prune(&world);
        assert_eq!(dropped, 1);
        assert_eq!(refs.len(), 2);
        assert!(!refs.contains(b));
        // Order of survivors is preserved.
        let remaining: Vec<Entity> = refs.iter().map(|w| w.entity()).collect();
        assert_eq!(remaining, alloc::vec![a, c]);
    }

    #[test]
    fn index_assign_tracks_cell() {
        let mut world = World::new();
        let e = world.spawn(());
        let mut idx = CellEntityIndex::new();

        assert_eq!(idx.assign(e, cell(1, 0, 0)), None);
        assert_eq!(idx.cell_of(e), Some(cell(1, 0, 0)));
        assert!(idx.contains(e));
        assert_eq!(idx.entity_count(), 1);
        assert_eq!(idx.cell_count(), 1);
    }

    #[test]
    fn index_assign_same_cell_is_noop() {
        let mut world = World::new();
        let e = world.spawn(());
        let mut idx = CellEntityIndex::new();
        idx.assign(e, cell(2, 2, 2));

        assert_eq!(idx.assign(e, cell(2, 2, 2)), Some(cell(2, 2, 2)));
        assert_eq!(idx.entities_in(cell(2, 2, 2)), alloc::vec![e]);
        assert_eq!(idx.entity_count(), 1);
    }

    #[test]
    fn index_reassign_moves_between_cells() {
        let mut world = World::new();
        let e = world.spawn(());
        let mut idx = CellEntityIndex::new();
        idx.assign(e, cell(0, 0, 0));

        assert_eq!(idx.assign(e, cell(5, 0, 0)), Some(cell(0, 0, 0)));
        assert_eq!(idx.cell_of(e), Some(cell(5, 0, 0)));
        assert!(idx.entities_in(cell(0, 0, 0)).is_empty());
        assert_eq!(idx.entities_in(cell(5, 0, 0)), alloc::vec![e]);
        assert_eq!(idx.entity_count(), 1);
        // The emptied source cell is no longer counted.
        assert_eq!(idx.cell_count(), 1);
    }

    #[test]
    fn index_remove_untracks() {
        let mut world = World::new();
        let e = world.spawn(());
        let mut idx = CellEntityIndex::new();
        idx.assign(e, cell(1, 1, 1));

        assert_eq!(idx.remove(e), Some(cell(1, 1, 1)));
        assert!(!idx.contains(e));
        assert_eq!(idx.cell_of(e), None);
        assert!(idx.is_empty());
        assert_eq!(idx.remove(e), None);
    }

    #[test]
    fn index_entities_in_is_sorted() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        let mut idx = CellEntityIndex::new();
        // Insert out of order.
        idx.assign(c, cell(0, 0, 0));
        idx.assign(a, cell(0, 0, 0));
        idx.assign(b, cell(0, 0, 0));

        assert_eq!(idx.entities_in(cell(0, 0, 0)), alloc::vec![a, b, c]);
    }

    #[test]
    fn take_cell_drains_and_untracks() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let mut idx = CellEntityIndex::new();
        idx.assign(a, cell(3, 3, 3));
        idx.assign(b, cell(3, 3, 3));

        let drained = idx.take_cell(cell(3, 3, 3));
        assert_eq!(drained, alloc::vec![a, b]);
        assert!(idx.is_empty());
        assert_eq!(idx.cell_count(), 0);
        // Draining again yields nothing.
        assert!(idx.take_cell(cell(3, 3, 3)).is_empty());
    }

    #[test]
    fn take_cells_is_deterministic_by_cell_then_handle() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        let d = world.spawn(());
        let mut idx = CellEntityIndex::new();
        idx.assign(b, cell(1, 0, 0));
        idx.assign(a, cell(1, 0, 0));
        idx.assign(d, cell(0, 0, 0));
        idx.assign(c, cell(0, 0, 0));

        // Pass cells out of order and with a duplicate: result must be grouped
        // in sorted (x,y,z) cell order, sorted by handle within each cell.
        let drained = idx.take_cells(&[cell(1, 0, 0), cell(0, 0, 0), cell(1, 0, 0)]);
        assert_eq!(drained, alloc::vec![c, d, a, b]);
        assert!(idx.is_empty());
    }
}
