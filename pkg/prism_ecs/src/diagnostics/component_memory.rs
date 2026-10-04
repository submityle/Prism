//! Component-first byte-level memory accounting (design §16.6 / §5.3 / §17).
//!
//! [`memory_footprint`](super::memory_footprint) answers "which *archetype*
//! wastes the most resident bytes"; this module takes the dual, component-first
//! view: "which *component type* dominates resident memory?" It is the byte
//! analogue of [`component_distribution`](super::component_distribution), which
//! ranks components by archetype spread and row counts but never by bytes.
//!
//! For each component type the report sums the resident payload of its column
//! across every archetype that carries it. A single archetype reserves
//! `capacity * element_size` bytes for one component's column (its `BlobVec`),
//! so summing over archetypes yields the total bytes that component's columns
//! occupy world-wide. This points an editor memory panel straight at the
//! heaviest component types — the ones worth shrinking, splitting into a
//! `SparseSet`, or demoting to a shared component (design §6 / §17).
//!
//! # Honest byte model
//! Figures reuse exactly the same reserved-payload model as
//! [`memory_footprint`](super::memory_footprint#honest-byte-model):
//!
//! * `element_size` is the component's
//!   [`Layout::size`](core::alloc::Layout::size) — pure element size, no
//!   padding.
//! * `reserved_bytes` is `reserved_rows * element_size`, where `reserved_rows`
//!   is the summed chunk capacity (`chunk_count * rows_per_chunk`) of every
//!   archetype carrying the component.
//! * `live_bytes` is `live_rows * element_size`.
//! * `wasted_bytes` is `(reserved_rows - live_rows) * element_size`.
//!
//! By construction the summed [`reserved_bytes`](ComponentMemoryEntry::reserved_bytes)
//! across all components equals the whole-world `total_allocated_bytes` from
//! [`MemoryFootprintReport`](super::memory_footprint::MemoryFootprintReport):
//! the two reports partition the same reserved payload, one by component and one
//! by archetype.
//!
//! # Scope
//! Figures describe the chunked Table-backed storage only (design §6).
//! `SparseSet` components are not laid out in archetype chunks and contribute no
//! bytes. Zero-sized (marker) components reserve no payload and are omitted, as
//! are components that appear only in archetypes with no allocated chunks.
//!
//! Capture is `O(archetypes * components_per_archetype)`, read-only, and
//! deterministic: entries are ranked by descending `reserved_bytes` with an
//! ascending component-id tie-break.

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::{ComponentId, Components};
use crate::diagnostics::inspector::WorldReport;
use crate::world::World;

/// Byte-level memory facts for a single component type across the archetype
/// graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentMemoryEntry {
    /// The component's stable id within the world.
    pub id: ComponentId,
    /// Size in bytes of a single value of this component (no padding).
    pub element_size: usize,
    /// Number of distinct archetypes whose identity includes this component and
    /// that reserve at least one chunk of payload for it.
    pub archetype_count: usize,
    /// Number of those archetypes that currently hold at least one live row.
    pub populated_archetype_count: usize,
    /// Live entity rows carrying this component, summed across archetypes.
    pub live_rows: usize,
    /// Reserved rows (summed chunk capacity) backing this component's columns.
    pub reserved_rows: usize,
}

impl ComponentMemoryEntry {
    /// Reserved column payload for this component world-wide
    /// (`reserved_rows * element_size`), live or not.
    #[inline]
    pub fn reserved_bytes(&self) -> usize {
        self.reserved_rows * self.element_size
    }

    /// Payload backing live rows carrying this component
    /// (`live_rows * element_size`).
    #[inline]
    pub fn live_bytes(&self) -> usize {
        self.live_rows * self.element_size
    }

    /// Reserved payload behind empty slots in this component's columns
    /// (`(reserved_rows - live_rows) * element_size`).
    #[inline]
    pub fn wasted_bytes(&self) -> usize {
        self.reserved_rows.saturating_sub(self.live_rows) * self.element_size
    }

    /// Fraction of this component's reserved payload that is live, in
    /// `[0.0, 1.0]`; `0.0` when it reserves nothing.
    #[inline]
    pub fn byte_occupancy(&self) -> f32 {
        if self.reserved_rows == 0 {
            0.0
        } else {
            self.live_rows as f32 / self.reserved_rows as f32
        }
    }
}

/// Internal per-component accumulator.
#[derive(Default)]
struct Acc {
    element_size: usize,
    archetype_count: usize,
    populated_archetype_count: usize,
    live_rows: usize,
    reserved_rows: usize,
}

/// A whole-world, component-first byte-level memory summary with a
/// heaviest-first ranking (design §16.6 / §17).
///
/// Produced by [`ComponentMemoryReport::capture`].
/// [`entries`](Self::entries) holds every component that reserves column
/// payload, sorted by descending [`reserved_bytes`](ComponentMemoryEntry::reserved_bytes)
/// (heaviest component first), with an ascending component-id tie-break for
/// determinism.
#[derive(Debug, Clone, Default)]
pub struct ComponentMemoryReport {
    /// Payload-reserving components, heaviest first.
    pub entries: Vec<ComponentMemoryEntry>,
    /// Total registered component types, including any that reserve no payload.
    pub component_count: usize,
    /// Total reserved column payload across all components (= the archetype-side
    /// `total_allocated_bytes`).
    pub total_reserved_bytes: usize,
    /// Total live-entity payload across all components.
    pub total_live_bytes: usize,
    /// Total reserved payload behind empty slots across all components.
    pub total_wasted_bytes: usize,
}

impl ComponentMemoryReport {
    /// Capture a component-first memory summary of `world`.
    #[inline]
    pub fn capture(world: &World) -> Self {
        Self::from_world_report(&WorldReport::capture(world), world.components())
    }

    /// Fold an existing [`WorldReport`] into a component-first memory summary,
    /// reading element sizes from `components`. `components` must be the same
    /// registry the report was captured from so component ids resolve.
    pub fn from_world_report(report: &WorldReport, components: &Components) -> Self {
        let mut acc: HashMap<ComponentId, Acc> = HashMap::new();

        for archetype in &report.archetypes {
            let occ = &archetype.occupancy;
            let capacity = occ.chunk_count * occ.rows_per_chunk;
            // An archetype with no allocated chunks reserves no column bytes.
            if capacity == 0 {
                continue;
            }
            let populated = occ.live_rows > 0;
            for &id in &archetype.components {
                let size = components.info(id).map_or(0, |info| info.layout().size());
                // Zero-sized components reserve no payload; skip to keep the
                // report byte-relevant and the sum invariant exact.
                if size == 0 {
                    continue;
                }
                let slot = acc.entry(id).or_default();
                slot.element_size = size;
                slot.archetype_count += 1;
                slot.live_rows += occ.live_rows;
                slot.reserved_rows += capacity;
                if populated {
                    slot.populated_archetype_count += 1;
                }
            }
        }

        let mut entries: Vec<ComponentMemoryEntry> = acc
            .into_iter()
            .map(|(id, a)| ComponentMemoryEntry {
                id,
                element_size: a.element_size,
                archetype_count: a.archetype_count,
                populated_archetype_count: a.populated_archetype_count,
                live_rows: a.live_rows,
                reserved_rows: a.reserved_rows,
            })
            .collect();

        // Heaviest reserved payload first; component id (dense index) breaks ties
        // deterministically — `HashMap` iteration order is otherwise arbitrary.
        entries.sort_by(|a, b| {
            b.reserved_bytes()
                .cmp(&a.reserved_bytes())
                .then_with(|| a.id.index().cmp(&b.id.index()))
        });

        let total_reserved_bytes = entries.iter().map(|e| e.reserved_bytes()).sum();
        let total_live_bytes = entries.iter().map(|e| e.live_bytes()).sum();
        let total_wasted_bytes = entries.iter().map(|e| e.wasted_bytes()).sum();

        Self {
            entries,
            component_count: report.component_count,
            total_reserved_bytes,
            total_live_bytes,
            total_wasted_bytes,
        }
    }

    /// Whether no component reserves column payload (nothing to analyze).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of payload-reserving components analyzed.
    #[inline]
    pub fn analyzed_count(&self) -> usize {
        self.entries.len()
    }

    /// World-wide byte occupancy across all component columns: live payload over
    /// reserved payload, in `[0.0, 1.0]`; `0.0` when nothing is reserved.
    #[inline]
    pub fn byte_occupancy(&self) -> f32 {
        if self.total_reserved_bytes == 0 {
            0.0
        } else {
            self.total_live_bytes as f32 / self.total_reserved_bytes as f32
        }
    }

    /// World-wide internal fragmentation in bytes: the unused fraction of
    /// reserved payload (`1.0 - byte_occupancy`).
    #[inline]
    pub fn internal_fragmentation(&self) -> f32 {
        1.0 - self.byte_occupancy()
    }

    /// The component reserving the most resident bytes, or `None` when none
    /// reserve payload. Ties resolve to the lowest component id.
    #[inline]
    pub fn heaviest(&self) -> Option<&ComponentMemoryEntry> {
        self.entries.first()
    }

    /// The entry for `id`, if that component reserves payload.
    pub fn entry(&self, id: ComponentId) -> Option<&ComponentMemoryEntry> {
        self.entries.iter().find(|e| e.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::diagnostics::memory_footprint::MemoryFootprintReport;
    use crate::world::World;

    #[derive(Debug)]
    struct Wide {
        _a: u64,
        _b: u64,
        _c: u64,
        _d: u64,
    }
    impl Component for Wide {}

    #[derive(Debug)]
    struct Narrow {
        _a: u8,
    }
    impl Component for Narrow {}

    #[derive(Debug)]
    struct Marker;
    impl Component for Marker {}

    #[test]
    fn empty_world_has_no_payload() {
        let world = World::new();
        let report = ComponentMemoryReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.total_reserved_bytes, 0);
        assert_eq!(report.byte_occupancy(), 0.0);
        assert!(report.heaviest().is_none());
    }

    #[test]
    fn element_size_and_live_bytes_track_component() {
        let mut world = World::new();
        for _ in 0..3u32 {
            world.spawn(Wide {
                _a: 0,
                _b: 0,
                _c: 0,
                _d: 0,
            });
        }
        let report = ComponentMemoryReport::capture(&world);
        assert_eq!(report.analyzed_count(), 1);
        let e = report.heaviest().unwrap();
        assert_eq!(e.element_size, 32);
        assert_eq!(e.live_rows, 3);
        assert_eq!(e.live_bytes(), 3 * 32);
        assert_eq!(e.reserved_bytes(), e.reserved_rows * 32);
        assert_eq!(
            e.reserved_bytes(),
            e.live_bytes() + e.wasted_bytes()
        );
        assert_eq!(report.total_live_bytes, 3 * 32);
    }

    #[test]
    fn marker_component_is_omitted() {
        let mut world = World::new();
        // Wide + Marker on the same entity: only Wide's column reserves bytes.
        world.spawn((
            Wide {
                _a: 0,
                _b: 0,
                _c: 0,
                _d: 0,
            },
            Marker,
        ));
        let report = ComponentMemoryReport::capture(&world);
        assert_eq!(report.analyzed_count(), 1);
        assert_eq!(report.heaviest().unwrap().element_size, 32);
    }

    #[test]
    fn two_components_rank_and_sum() {
        let mut world = World::new();
        // One shared archetype holding both columns, several rows.
        for _ in 0..5u32 {
            world.spawn((
                Wide {
                    _a: 0,
                    _b: 0,
                    _c: 0,
                    _d: 0,
                },
                Narrow { _a: 0 },
            ));
        }
        let report = ComponentMemoryReport::capture(&world);
        assert_eq!(report.analyzed_count(), 2);

        // Same archetype → same reserved_rows, so Wide (32 B) outweighs Narrow (1 B).
        assert_eq!(report.heaviest().unwrap().element_size, 32);
        assert!(
            report.entries[0].reserved_bytes() >= report.entries[1].reserved_bytes()
        );

        // Totals equal the sum of entries.
        let sum_reserved: usize = report.entries.iter().map(|e| e.reserved_bytes()).sum();
        assert_eq!(report.total_reserved_bytes, sum_reserved);
    }

    #[test]
    fn component_sum_matches_archetype_total() {
        let mut world = World::new();
        world.spawn((
            Wide {
                _a: 1,
                _b: 2,
                _c: 3,
                _d: 4,
            },
            Narrow { _a: 7 },
        ));
        for _ in 0..4u32 {
            world.spawn(Wide {
                _a: 0,
                _b: 0,
                _c: 0,
                _d: 0,
            });
        }
        world.spawn(Narrow { _a: 9 });

        let by_component = ComponentMemoryReport::capture(&world);
        let by_archetype = MemoryFootprintReport::capture(&world);

        // The two reports partition the same reserved payload.
        assert_eq!(
            by_component.total_reserved_bytes,
            by_archetype.total_allocated_bytes
        );
        assert_eq!(
            by_component.total_live_bytes,
            by_archetype.total_live_bytes
        );
        assert_eq!(
            by_component.total_wasted_bytes,
            by_archetype.total_wasted_bytes
        );
    }
}
