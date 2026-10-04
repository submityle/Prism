//! Storage-strategy distribution over registered component types
//! (design §6 four-state storage / §16.6 / §17).
//!
//! Every diagnostic in this module so far measures *runtime* occupancy — how
//! many rows, chunks, or bytes a component or archetype holds right now. This
//! module takes a complementary, *registration-time* view: it classifies every
//! component type the world knows about by its [`StorageType`] and summarises
//! the physical shape of each bucket.
//!
//! The design's storage model (§6) splits components across three physical
//! strategies — [`Table`](StorageType::Table) (columnar SoA, SIMD-friendly,
//! the default), [`SparseSet`](StorageType::SparseSet) (out-of-band, O(1)
//! toggle, for tags / frequently flipped state), and
//! [`Shared`](StorageType::Shared) (interned batch key). Choosing the right
//! strategy per component is an architecture decision with real performance
//! consequences, and it is easy to get wrong silently: a large hot component
//! accidentally left on `SparseSet` loses chunk SIMD; a per-entity-mutated
//! value mistakenly declared `Shared` would be rejected at query construction;
//! a swarm of zero-sized tags on `Table` fragments the archetype graph. This
//! report gives an editor or CI a single glance at the storage mix so those
//! mistakes surface (design §6 / §17 / §22 risk #5).
//!
//! Unlike the occupancy reports, this one needs only the component
//! [`registry`](Components): it describes *types*, not live instances, so it is
//! well-defined even on a freshly built, entity-less world.
//!
//! # What each bucket reports
//! For the component types sharing one [`StorageType`]:
//!
//! * [`component_count`](StorageBucketEntry::component_count) — how many types.
//! * [`zero_sized_count`](StorageBucketEntry::zero_sized_count) — marker / tag
//!   types whose [`Layout::size`](core::alloc::Layout::size) is `0`.
//! * [`dynamic_count`](StorageBucketEntry::dynamic_count) — types registered at
//!   runtime without a Rust [`TypeId`](core::any::TypeId) (design §16.2 dynamic
//!   components / editor / script bridge).
//! * [`nontrivial_drop_count`](StorageBucketEntry::nontrivial_drop_count) —
//!   types carrying a drop glue function (non-`Copy` payloads needing teardown
//!   on despawn / snapshot restore).
//! * [`total_element_bytes`](StorageBucketEntry::total_element_bytes),
//!   [`max_element_bytes`](StorageBucketEntry::max_element_bytes), and
//!   [`max_align`](StorageBucketEntry::max_align) — the per-element size and
//!   alignment profile, feeding the SIMD / cache-layout discussion (design
//!   §17 列对齐 + SIMD).
//!
//! # Scope and determinism
//! This is a pure classification of registered types; it reads no world state
//! and allocates no entities, so it reports identical facts regardless of how
//! many entities exist. Capture is `O(components)`, read-only, and
//! deterministic: buckets are ordered by the design's storage enumeration
//! (`Table`, then `SparseSet`, then `Shared`) and only non-empty buckets are
//! emitted.

use alloc::vec::Vec;

use crate::component::{ComponentId, Components, StorageType};
use crate::world::World;

/// Fixed, design-order rank for a [`StorageType`] so buckets sort
/// deterministically without requiring `Ord` on the enum. Mirrors the §6
/// enumeration order Table → SparseSet → Shared.
#[inline]
const fn storage_rank(storage: StorageType) -> u8 {
    match storage {
        StorageType::Table => 0,
        StorageType::SparseSet => 1,
        StorageType::Shared => 2,
    }
}

/// Aggregate shape of the component types sharing one [`StorageType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageBucketEntry {
    /// The physical storage strategy this bucket describes.
    pub storage: StorageType,
    /// Number of registered component types using this storage strategy.
    pub component_count: usize,
    /// How many of those types are zero-sized (marker / tag components).
    pub zero_sized_count: usize,
    /// How many were registered dynamically at runtime (no Rust `TypeId`).
    pub dynamic_count: usize,
    /// How many carry drop glue (non-trivial teardown on despawn / restore).
    pub nontrivial_drop_count: usize,
    /// Sum of per-element sizes (bytes) over the types in this bucket.
    pub total_element_bytes: usize,
    /// Largest single-element size (bytes) in this bucket; `0` when every type
    /// is zero-sized.
    pub max_element_bytes: usize,
    /// Largest element alignment (bytes) in this bucket; relevant to SIMD
    /// column alignment (design §17). `1` when the bucket holds only
    /// byte-aligned or zero-sized types.
    pub max_align: usize,
}

impl StorageBucketEntry {
    /// Mean per-element size (bytes) across this bucket's types, or `0.0` when
    /// the bucket is empty.
    #[inline]
    pub fn average_element_bytes(&self) -> f32 {
        if self.component_count == 0 {
            0.0
        } else {
            self.total_element_bytes as f32 / self.component_count as f32
        }
    }

    /// Number of types in this bucket that carry a non-zero-sized payload.
    #[inline]
    pub fn sized_count(&self) -> usize {
        self.component_count - self.zero_sized_count
    }
}

/// Internal per-storage accumulator.
#[derive(Default)]
struct Acc {
    component_count: usize,
    zero_sized_count: usize,
    dynamic_count: usize,
    nontrivial_drop_count: usize,
    total_element_bytes: usize,
    max_element_bytes: usize,
    max_align: usize,
}

/// A whole-registry breakdown of component types by physical storage strategy
/// (design §6 / §16.6 / §17).
///
/// Produced by [`StorageDistributionReport::capture`].
/// [`buckets`](Self::buckets) holds one entry per non-empty [`StorageType`],
/// ordered by the design's storage enumeration (`Table`, `SparseSet`,
/// `Shared`) for determinism.
#[derive(Debug, Clone, Default)]
pub struct StorageDistributionReport {
    /// Non-empty storage buckets in design-enumeration order.
    pub buckets: Vec<StorageBucketEntry>,
    /// Total number of registered component types across all buckets.
    pub component_count: usize,
    /// Total number of zero-sized (marker) types across all buckets.
    pub zero_sized_count: usize,
    /// Sum of per-element sizes (bytes) across all registered types.
    pub total_element_bytes: usize,
}

impl StorageDistributionReport {
    /// Capture a storage-strategy breakdown of `world`'s component registry.
    #[inline]
    pub fn capture(world: &World) -> Self {
        Self::from_components(world.components())
    }

    /// Classify every registered component type in `components` by storage
    /// strategy.
    pub fn from_components(components: &Components) -> Self {
        // Three physical strategies (design §6); index by `storage_rank`.
        let mut accs: [Acc; 3] = Default::default();

        let len = components.len();
        for i in 0..len {
            let Some(info) = components.info(ComponentId::new(i as u32)) else {
                continue;
            };
            let layout = info.layout();
            let size = layout.size();
            let acc = &mut accs[storage_rank(info.storage()) as usize];
            acc.component_count += 1;
            acc.total_element_bytes += size;
            if size == 0 {
                acc.zero_sized_count += 1;
            }
            if size > acc.max_element_bytes {
                acc.max_element_bytes = size;
            }
            if layout.align() > acc.max_align {
                acc.max_align = layout.align();
            }
            if info.type_id().is_none() {
                acc.dynamic_count += 1;
            }
            if info.drop_fn().is_some() {
                acc.nontrivial_drop_count += 1;
            }
        }

        let order = [StorageType::Table, StorageType::SparseSet, StorageType::Shared];
        let mut buckets = Vec::new();
        for storage in order {
            let acc = &accs[storage_rank(storage) as usize];
            if acc.component_count == 0 {
                continue;
            }
            buckets.push(StorageBucketEntry {
                storage,
                component_count: acc.component_count,
                zero_sized_count: acc.zero_sized_count,
                dynamic_count: acc.dynamic_count,
                nontrivial_drop_count: acc.nontrivial_drop_count,
                total_element_bytes: acc.total_element_bytes,
                max_element_bytes: acc.max_element_bytes,
                // A populated bucket always has align >= 1; normalise the
                // all-zero-sized case (where the loop still saw align 1).
                max_align: acc.max_align.max(1),
            });
        }

        let component_count = buckets.iter().map(|b| b.component_count).sum();
        let zero_sized_count = buckets.iter().map(|b| b.zero_sized_count).sum();
        let total_element_bytes = buckets.iter().map(|b| b.total_element_bytes).sum();

        Self {
            buckets,
            component_count,
            zero_sized_count,
            total_element_bytes,
        }
    }

    /// Whether no component types are registered (nothing to classify).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// The bucket for `storage`, if any registered type uses it.
    pub fn bucket(&self, storage: StorageType) -> Option<&StorageBucketEntry> {
        self.buckets.iter().find(|b| b.storage == storage)
    }

    /// The storage strategy backing the most registered component types, or
    /// `None` when the registry is empty. Ties resolve to the design's
    /// enumeration order (`Table` before `SparseSet` before `Shared`).
    pub fn dominant_storage(&self) -> Option<StorageType> {
        self.buckets
            .iter()
            .max_by(|a, b| {
                a.component_count
                    .cmp(&b.component_count)
                    // Buckets are already in design order; on a tie prefer the
                    // earlier (lower-rank) strategy by reversing the rank cmp.
                    .then_with(|| storage_rank(b.storage).cmp(&storage_rank(a.storage)))
            })
            .map(|b| b.storage)
    }

    /// Mean per-element size (bytes) across every registered type, or `0.0`
    /// when the registry is empty.
    #[inline]
    pub fn average_element_bytes(&self) -> f32 {
        if self.component_count == 0 {
            0.0
        } else {
            self.total_element_bytes as f32 / self.component_count as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::world::World;

    #[derive(Debug)]
    struct Wide {
        _a: u64,
        _b: u64,
    }
    impl Component for Wide {}

    #[derive(Debug)]
    struct TableTag;
    impl Component for TableTag {}

    #[derive(Debug)]
    struct Sparse {
        _a: u32,
    }
    impl Component for Sparse {
        const STORAGE: StorageType = StorageType::SparseSet;
    }

    #[test]
    fn empty_registry_classifies_nothing() {
        let world = World::new();
        let report = StorageDistributionReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.component_count, 0);
        assert_eq!(report.total_element_bytes, 0);
        assert!(report.dominant_storage().is_none());
        assert_eq!(report.average_element_bytes(), 0.0);
    }

    #[test]
    fn buckets_sum_to_report_totals() {
        let mut world = World::new();
        world.register_component::<Wide>();
        world.register_component::<TableTag>();
        world.register_component::<Sparse>();

        let report = StorageDistributionReport::capture(&world);

        // Within-report partition invariants: buckets exactly cover the registry.
        let sum_count: usize = report.buckets.iter().map(|b| b.component_count).sum();
        let sum_bytes: usize = report.buckets.iter().map(|b| b.total_element_bytes).sum();
        let sum_zst: usize = report.buckets.iter().map(|b| b.zero_sized_count).sum();
        assert_eq!(sum_count, report.component_count);
        assert_eq!(sum_bytes, report.total_element_bytes);
        assert_eq!(sum_zst, report.zero_sized_count);
        assert_eq!(report.component_count, 3);
    }

    #[test]
    fn table_and_sparse_buckets_separate() {
        let mut world = World::new();
        world.register_component::<Wide>();
        world.register_component::<TableTag>();
        world.register_component::<Sparse>();

        let report = StorageDistributionReport::capture(&world);

        let table = report.bucket(StorageType::Table).unwrap();
        // Wide + TableTag both default to Table storage.
        assert_eq!(table.component_count, 2);
        assert_eq!(table.zero_sized_count, 1); // TableTag is zero-sized.
        assert_eq!(table.sized_count(), 1); // Wide.
        assert_eq!(table.max_element_bytes, 16); // Wide = two u64.
        assert!(table.max_align >= 8);

        let sparse = report.bucket(StorageType::SparseSet).unwrap();
        assert_eq!(sparse.component_count, 1);
        assert_eq!(sparse.zero_sized_count, 0);
        assert_eq!(sparse.max_element_bytes, 4); // Sparse = u32.

        // No shared components registered.
        assert!(report.bucket(StorageType::Shared).is_none());
    }

    #[test]
    fn buckets_in_design_enumeration_order() {
        let mut world = World::new();
        // Register sparse first to prove ordering is by storage rank, not
        // registration order.
        world.register_component::<Sparse>();
        world.register_component::<Wide>();

        let report = StorageDistributionReport::capture(&world);
        let ranks: Vec<u8> = report.buckets.iter().map(|b| storage_rank(b.storage)).collect();
        let mut sorted = ranks.clone();
        sorted.sort_unstable();
        assert_eq!(ranks, sorted);
        // Table (rank 0) must precede SparseSet (rank 1).
        assert_eq!(report.buckets[0].storage, StorageType::Table);
        assert_eq!(report.buckets[1].storage, StorageType::SparseSet);
    }

    #[test]
    fn dominant_storage_tracks_largest_bucket() {
        let mut world = World::new();
        world.register_component::<Wide>();
        world.register_component::<TableTag>();
        world.register_component::<Sparse>();

        // Table has two types, SparseSet one → Table dominates.
        let report = StorageDistributionReport::capture(&world);
        assert_eq!(report.dominant_storage(), Some(StorageType::Table));
    }
}
