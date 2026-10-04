//! Sparse-set storage occupancy / keyspace-fragmentation census
//! (design §6 四态存储 / §16.6).
//!
//! Every other occupancy report in this module describes the chunked
//! Table-backed path and explicitly excludes [`SparseSet`] components, which
//! live out-of-band in the world's
//! [`SparseSets`](crate::storage::SparseSets) registry (design §6 "增删不搬迁
//! archetype"). This module is the counterpart for that path: it accounts the
//! memory a sparse set actually costs and surfaces the one failure mode the
//! strategy is prone to — keyspace fragmentation.
//!
//! # The dense / sparse tradeoff
//! A [`ComponentSparseSet`](crate::storage::ComponentSparseSet) keeps two
//! parallel structures: a *dense* side (packed component payload plus a
//! dense-row → owning-[`Entity`](crate::entity::Entity) map and two per-row
//! change-tick vectors) whose length is the live value count, and a *sparse*
//! index array that grows to cover the largest entity index ever inserted,
//! mapping entity index → dense row. The sparse array buys O(1) insert / remove
//! / lookup without ever moving the entity between archetypes, but it is sized
//! by the *entity-index span*, not by the live count. When the live entities
//! carry scattered high indices, the sparse index array dwarfs the dense
//! payload — memory paid for keys that are not there.
//!
//! This census reports, per sparse-set component and world-wide, the live
//! count, the dense payload / bookkeeping bytes, the sparse-index bytes implied
//! by the current index span, and a `sparse_overhead_permille` that expresses
//! how much of the resident memory is the index array rather than data —
//! surfacing sparse sets that have become mostly empty index space.
//!
//! # Honest bounds
//! The figures are derived from the sparse set's current members (the only
//! read-only surface it exposes: [`len`](crate::storage::ComponentSparseSet::len)
//! and [`entities`](crate::storage::ComponentSparseSet::entities)). The real
//! sparse array never shrinks on removal, so the index span computed from the
//! current maximum member index is a **lower bound**: the true array may be
//! larger (and the true fragmentation worse) if higher-indexed entities were
//! inserted and later removed. The accounting never mutates the set, so it is
//! safe to drive from a diagnostics system.

use alloc::vec::Vec;

use crate::component::{ComponentId, StorageType};
use crate::entity::Entity;
use crate::storage::ComponentSparseSet;
use crate::world::World;

/// Bytes of per-dense-row bookkeeping a sparse set keeps alongside the payload:
/// the dense-row → owning [`Entity`] map plus the *added* and *changed* tick
/// vectors (design §6 / §10). This is overhead the Table path folds into chunk
/// metadata, so it is accounted separately here.
const DENSE_ROW_BOOKKEEPING_BYTES: usize =
    size_of::<Entity>() + 2 * size_of::<u32>();

/// Bytes per entry in the sparse index array (an entity index → dense row
/// `u32` slot).
const SPARSE_SLOT_BYTES: usize = size_of::<u32>();

/// One sparse-set component's occupancy and memory accounting (design §6).
///
/// All fields are a read-only snapshot and are public so a report can be
/// assembled from synthetic entries in tests. Byte figures are derived as
/// described in the [module docs](self); the sparse-index figure is a lower
/// bound because the backing array never shrinks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SparseSetOccupancyEntry {
    /// The component's stable id.
    pub id: ComponentId,
    /// Entities with a value in this set (the dense length).
    pub live_count: usize,
    /// The largest entity index among current members, or `0` when empty (see
    /// [`index_span`](Self::index_span) for the span this implies).
    pub max_entity_index: u32,
    /// Lower bound on the sparse index array length: `max_entity_index + 1`, or
    /// `0` when the set is empty. The real array may be larger after removals.
    pub index_span: u32,
    /// Size in bytes of one stored component value ([`Layout::size`]).
    pub element_bytes: usize,
    /// Dense component payload bytes: `live_count * element_bytes`.
    pub dense_payload_bytes: usize,
    /// Dense bookkeeping bytes (entity map + two tick vectors):
    /// `live_count * `[`DENSE_ROW_BOOKKEEPING_BYTES`].
    pub dense_bookkeeping_bytes: usize,
    /// Sparse index array bytes implied by [`index_span`](Self::index_span):
    /// `index_span * `[`SPARSE_SLOT_BYTES`] (a lower bound).
    pub sparse_index_bytes: usize,
}

impl SparseSetOccupancyEntry {
    /// Build an entry for one sparse set given its component id and element
    /// size, scanning the current members for the index span.
    fn from_set(id: ComponentId, element_bytes: usize, set: &ComponentSparseSet) -> Self {
        let live_count = set.len();
        let mut max_entity_index = 0u32;
        for entity in set.entities() {
            let index = entity.index();
            if index > max_entity_index {
                max_entity_index = index;
            }
        }
        let index_span = if live_count == 0 {
            0
        } else {
            // `+ 1` because indices are zero-based; cannot overflow because a
            // live member means `max_entity_index < u32::MAX` in practice, but
            // guard anyway.
            max_entity_index.saturating_add(1)
        };
        Self {
            id,
            live_count,
            max_entity_index,
            index_span,
            element_bytes,
            dense_payload_bytes: live_count * element_bytes,
            dense_bookkeeping_bytes: live_count * DENSE_ROW_BOOKKEEPING_BYTES,
            sparse_index_bytes: index_span as usize * SPARSE_SLOT_BYTES,
        }
    }

    /// Total dense-side bytes: payload plus bookkeeping.
    #[inline]
    pub const fn dense_bytes(&self) -> usize {
        self.dense_payload_bytes + self.dense_bookkeeping_bytes
    }

    /// Total resident bytes attributed to this set: dense plus the sparse index
    /// array.
    #[inline]
    pub const fn resident_bytes(&self) -> usize {
        self.dense_bytes() + self.sparse_index_bytes
    }

    /// How densely the live keys fill the index span, in per-mille (`1000` =
    /// every slot in `[0, max_entity_index]` is live). An empty set reports
    /// `1000` (it wastes no index space).
    pub fn keyspace_density_permille(&self) -> u64 {
        if self.index_span == 0 {
            return 1000;
        }
        let permille = self.live_count as u64 * 1000 / self.index_span as u64;
        if permille > 1000 { 1000 } else { permille }
    }

    /// Fraction of [`resident_bytes`](Self::resident_bytes) spent on the sparse
    /// index array rather than data, in per-mille. High values mark a set that
    /// is mostly empty keyspace. `0` when the set holds nothing resident.
    pub fn sparse_overhead_permille(&self) -> u64 {
        let resident = self.resident_bytes();
        if resident == 0 {
            return 0;
        }
        self.sparse_index_bytes as u64 * 1000 / resident as u64
    }

    /// Whether the set is registered but currently holds no values.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.live_count == 0
    }
}

/// Whole-world sparse-set occupancy census (design §6 / §16.6).
///
/// Built by [`from_world`](Self::from_world); entries are sorted ascending by
/// [`ComponentId`] for a deterministic report. Components declared with
/// [`StorageType::SparseSet`] that have no set allocated yet (registered but
/// never inserted into) are not entries — they are tallied in
/// [`unmaterialized_count`](Self::unmaterialized_count).
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct SparseSetOccupancyReport {
    entries: Vec<SparseSetOccupancyEntry>,
    unmaterialized_count: usize,
}

impl SparseSetOccupancyReport {
    /// Census every sparse-set component in `world`.
    ///
    /// Walks the component registry; for each type whose storage is
    /// [`StorageType::SparseSet`] it builds an entry when a backing set exists,
    /// or counts it as unmaterialized when the type is registered but no set
    /// has been allocated (nothing was ever inserted). Read-only.
    pub fn from_world(world: &World) -> Self {
        let components = world.components();
        let sparse_sets = world.sparse_sets();
        let len = components.len();
        let mut entries = Vec::new();
        let mut unmaterialized_count = 0;

        for i in 0..len {
            let Some(info) = components.info(ComponentId::new(i as u32)) else {
                continue;
            };
            if info.storage() != StorageType::SparseSet {
                continue;
            }
            let id = info.id();
            match sparse_sets.get(id) {
                Some(set) => {
                    entries.push(SparseSetOccupancyEntry::from_set(
                        id,
                        info.layout().size(),
                        set,
                    ));
                }
                None => unmaterialized_count += 1,
            }
        }

        Self::from_entries(entries, unmaterialized_count)
    }

    /// Assemble a report from an explicit entry set and unmaterialized tally,
    /// sorting the entries ascending by [`ComponentId`].
    ///
    /// [`from_world`](Self::from_world) builds the real entries; this
    /// lower-level constructor keeps the aggregate math independent of a live
    /// world.
    pub fn from_entries(
        mut entries: Vec<SparseSetOccupancyEntry>,
        unmaterialized_count: usize,
    ) -> Self {
        entries.sort_unstable_by_key(|e| e.id);
        Self {
            entries,
            unmaterialized_count,
        }
    }

    /// The per-component entries, ascending by component id.
    #[inline]
    pub fn entries(&self) -> &[SparseSetOccupancyEntry] {
        &self.entries
    }

    /// Number of materialized sparse sets in the census.
    #[inline]
    pub fn sparse_set_count(&self) -> usize {
        self.entries.len()
    }

    /// Components declared [`StorageType::SparseSet`] but with no backing set
    /// allocated yet (registered, never inserted into). They cost registry
    /// bookkeeping without holding data.
    #[inline]
    pub fn unmaterialized_count(&self) -> usize {
        self.unmaterialized_count
    }

    /// Whether the census carries no materialized sparse sets.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total live values across every sparse set.
    #[inline]
    pub fn total_live(&self) -> usize {
        self.entries.iter().map(|e| e.live_count).sum()
    }

    /// Total dense bytes (payload + bookkeeping) across every sparse set.
    #[inline]
    pub fn total_dense_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.dense_bytes()).sum()
    }

    /// Total sparse index-array bytes across every sparse set (a lower bound).
    #[inline]
    pub fn total_sparse_index_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.sparse_index_bytes).sum()
    }

    /// Total resident bytes (dense + sparse index) across every sparse set.
    #[inline]
    pub fn total_resident_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.resident_bytes()).sum()
    }

    /// Fraction of total resident bytes spent on sparse index arrays rather
    /// than data, in per-mille. `0` when nothing is resident.
    pub fn overall_sparse_overhead_permille(&self) -> u64 {
        let resident = self.total_resident_bytes();
        if resident == 0 {
            return 0;
        }
        self.total_sparse_index_bytes() as u64 * 1000 / resident as u64
    }

    /// The sparse set holding the most live values (ties resolve to the lowest
    /// id), or `None` when the census is empty.
    pub fn busiest(&self) -> Option<ComponentId> {
        // Entries are ascending by id; a strictly-greater test keeps the lowest
        // id on a tie.
        self.entries
            .iter()
            .reduce(|best, e| if e.live_count > best.live_count { e } else { best })
            .map(|e| e.id)
    }

    /// The non-empty sparse set with the highest sparse-index overhead (ties,
    /// including when none has members, resolve to the lowest id), or `None`
    /// when no set currently holds any values.
    pub fn most_fragmented(&self) -> Option<ComponentId> {
        self.entries
            .iter()
            .filter(|e| e.live_count > 0)
            .reduce(|best, e| {
                if e.sparse_overhead_permille() > best.sparse_overhead_permille() {
                    e
                } else {
                    best
                }
            })
            .map(|e| e.id)
    }

    /// The entry for `component`, or `None` if it has no materialized set.
    #[inline]
    pub fn entry(&self, component: ComponentId) -> Option<SparseSetOccupancyEntry> {
        self.entries
            .binary_search_by(|e| e.id.cmp(&component))
            .ok()
            .map(|i| self.entries[i])
    }

    /// Whether `component` has a materialized set in the census.
    #[inline]
    pub fn contains(&self, component: ComponentId) -> bool {
        self.entries
            .binary_search_by(|e| e.id.cmp(&component))
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Anchor(i32);
    impl Component for Anchor {}

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Charge(i32);
    impl Component for Charge {
        const STORAGE: StorageType = StorageType::SparseSet;
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Flagged;
    impl Component for Flagged {
        const STORAGE: StorageType = StorageType::SparseSet;
    }

    fn entry(
        id: u32,
        live: usize,
        max_index: u32,
        element_bytes: usize,
    ) -> SparseSetOccupancyEntry {
        let index_span = if live == 0 { 0 } else { max_index + 1 };
        SparseSetOccupancyEntry {
            id: ComponentId::new(id),
            live_count: live,
            max_entity_index: max_index,
            index_span,
            element_bytes,
            dense_payload_bytes: live * element_bytes,
            dense_bookkeeping_bytes: live * DENSE_ROW_BOOKKEEPING_BYTES,
            sparse_index_bytes: index_span as usize * SPARSE_SLOT_BYTES,
        }
    }

    #[test]
    fn fresh_world_has_no_sparse_sets() {
        let world = World::new();
        let report = SparseSetOccupancyReport::from_world(&world);
        assert!(report.is_empty());
        assert_eq!(report.sparse_set_count(), 0);
        assert_eq!(report.unmaterialized_count(), 0);
        assert_eq!(report.total_live(), 0);
        assert_eq!(report.total_resident_bytes(), 0);
        assert_eq!(report.overall_sparse_overhead_permille(), 0);
        assert_eq!(report.busiest(), None);
        assert_eq!(report.most_fragmented(), None);
    }

    #[test]
    fn materialized_set_counts_live_and_payload() {
        let mut world = World::new();
        for i in 0..3i32 {
            world.spawn((Anchor(i), Charge(i * 10)));
        }
        let report = SparseSetOccupancyReport::from_world(&world);
        assert_eq!(report.sparse_set_count(), 1);
        assert_eq!(report.unmaterialized_count(), 0);
        assert_eq!(report.total_live(), 3);
        let id = report.busiest().unwrap();
        let e = report.entry(id).unwrap();
        assert_eq!(e.live_count, 3);
        assert_eq!(e.element_bytes, size_of::<i32>());
        assert_eq!(e.dense_payload_bytes, 3 * size_of::<i32>());
        assert_eq!(
            e.dense_bookkeeping_bytes,
            3 * DENSE_ROW_BOOKKEEPING_BYTES
        );
        // Three contiguous low-index entities pack the keyspace densely.
        assert!(e.index_span >= 3);
        assert!(e.keyspace_density_permille() > 0);
    }

    #[test]
    fn declared_but_never_inserted_is_unmaterialized() {
        let mut world = World::new();
        // Registered as a sparse-set component, but no entity ever carries it,
        // so no backing set is allocated.
        world.register_component::<Charge>();
        let report = SparseSetOccupancyReport::from_world(&world);
        assert_eq!(report.sparse_set_count(), 0);
        assert_eq!(report.unmaterialized_count(), 1);
        assert!(report.is_empty());
    }

    #[test]
    fn two_sparse_components_are_both_censused() {
        let mut world = World::new();
        world.spawn((Anchor(0), Charge(1), Flagged));
        world.spawn((Anchor(1), Charge(2)));
        let report = SparseSetOccupancyReport::from_world(&world);
        assert_eq!(report.sparse_set_count(), 2);
        // Charge has two members, Flagged one; Charge is busiest.
        assert_eq!(report.total_live(), 3);
        let busiest = report.busiest().unwrap();
        assert_eq!(report.entry(busiest).unwrap().live_count, 2);
    }

    #[test]
    fn scattered_keys_report_high_sparse_overhead() {
        // One live value at index 1000: dense is tiny, the sparse index array
        // spans 1001 slots, so most resident bytes are index overhead.
        let e = entry(0, 1, 1000, 4);
        assert_eq!(e.index_span, 1001);
        assert_eq!(e.sparse_index_bytes, 1001 * SPARSE_SLOT_BYTES);
        // Dense = 4 payload + 16 bookkeeping = 20; sparse index = 4004.
        assert!(e.sparse_overhead_permille() > 900);
        assert!(e.keyspace_density_permille() < 10);
    }

    #[test]
    fn empty_entry_wastes_no_keyspace() {
        let e = entry(0, 0, 0, 4);
        assert!(e.is_empty());
        assert_eq!(e.index_span, 0);
        assert_eq!(e.sparse_index_bytes, 0);
        assert_eq!(e.resident_bytes(), 0);
        assert_eq!(e.keyspace_density_permille(), 1000);
        assert_eq!(e.sparse_overhead_permille(), 0);
    }

    #[test]
    fn most_fragmented_ignores_empty_and_breaks_ties_to_lowest_id() {
        let report = SparseSetOccupancyReport::from_entries(
            alloc::vec![
                // Dense, low overhead.
                entry(1, 100, 99, 4),
                // Scattered, high overhead.
                entry(4, 1, 500, 4),
                // Equally scattered as id 4, higher id — loses the tie.
                entry(9, 1, 500, 4),
                // Empty — excluded from the ranking entirely.
                entry(2, 0, 0, 4),
            ],
            0,
        );
        let ids: Vec<_> = report.entries().iter().map(|e| e.id.index()).collect();
        assert_eq!(ids, alloc::vec![1, 2, 4, 9]);
        assert_eq!(report.most_fragmented(), Some(ComponentId::new(4)));
        // The dense set is the fullest.
        assert_eq!(report.busiest(), Some(ComponentId::new(1)));
    }

    #[test]
    fn rollup_sums_across_sets() {
        let report = SparseSetOccupancyReport::from_entries(
            alloc::vec![entry(0, 2, 3, 8), entry(1, 1, 1000, 4)],
            3,
        );
        assert_eq!(report.unmaterialized_count(), 3);
        assert_eq!(report.total_live(), 3);
        let expect_dense = (2 * 8 + 2 * DENSE_ROW_BOOKKEEPING_BYTES)
            + (4 + DENSE_ROW_BOOKKEEPING_BYTES);
        assert_eq!(report.total_dense_bytes(), expect_dense);
        let expect_sparse = 4 * SPARSE_SLOT_BYTES + 1001 * SPARSE_SLOT_BYTES;
        assert_eq!(report.total_sparse_index_bytes(), expect_sparse);
        assert_eq!(
            report.total_resident_bytes(),
            expect_dense + expect_sparse
        );
        assert!(report.overall_sparse_overhead_permille() > 0);
    }
}
