//! Component distribution diagnostics (design §16.6 / §5.3 / §22 risk #5).
//!
//! Where [`archetype_fragmentation`](super::archetype_fragmentation) looks at
//! fragmentation *archetype-first* (how thinly spread live entities are across
//! archetypes), this module takes the dual, *component-first* view: for each
//! registered component type, across how many distinct archetypes does it
//! appear, and how are its live instances spread over them?
//!
//! A component that participates in a large number of archetypes is a direct
//! driver of archetype explosion (design §22 risk #5): every extra component
//! combination it co-occurs with mints another archetype, and iteration pays
//! per-archetype setup for each. Surfacing the "most spread" components points
//! straight at the fragmenting-relation or tag-combinatorics hot spots the
//! design warns about (design §11 / §5.3), complementing the archetype-centric
//! ranking.
//!
//! The report folds a single [`WorldReport`](super::inspector::WorldReport):
//! for every archetype it attributes that archetype's occupancy to each
//! component in the archetype's identity. Only components that appear in at
//! least one archetype are reported; a registered-but-never-used component has
//! no distribution footprint.
//!
//! Capture is `O(archetypes * components_per_archetype)`, read-only, and
//! deterministic: entries are ranked by descending archetype spread with a
//! component-id tie-break.
//!
//! # Scope
//! Figures describe the chunked Table-backed storage (design §6), mirroring
//! [`WorldReport`]. `SparseSet` components are not laid out in archetype chunks
//! and are therefore outside these counts.

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::ComponentId;
use crate::diagnostics::inspector::WorldReport;
use crate::world::World;

/// Distribution facts for a single component type across the archetype graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentDistributionEntry {
    /// The component's stable id within the world.
    pub id: ComponentId,
    /// Number of distinct archetypes whose identity includes this component,
    /// including archetypes that currently hold no live rows.
    pub archetype_count: usize,
    /// Number of those archetypes that currently hold at least one live row.
    pub populated_archetype_count: usize,
    /// Total live entity rows carrying this component, summed across every
    /// archetype that includes it (its live instance count).
    pub live_rows: usize,
    /// Total allocated 16 KiB chunks across every archetype that includes this
    /// component (design §5.3).
    pub chunk_count: usize,
}

impl ComponentDistributionEntry {
    /// Archetypes that include this component but currently hold no live rows
    /// (`archetype_count - populated_archetype_count`). High values signal
    /// stale / drained archetypes keeping this component in the graph.
    #[inline]
    pub fn empty_archetype_count(&self) -> usize {
        self.archetype_count
            .saturating_sub(self.populated_archetype_count)
    }

    /// Average live rows per *populated* archetype carrying this component
    /// (`live_rows / populated_archetype_count`). Returns `0.0` when no
    /// archetype is populated so callers never divide by zero.
    ///
    /// A low average over a high [`archetype_count`](Self::archetype_count) is
    /// the component-explosion smell: the component is sprinkled across many
    /// archetypes that each hold almost nothing (design §22 risk #5).
    #[inline]
    pub fn avg_rows_per_populated_archetype(&self) -> f32 {
        if self.populated_archetype_count == 0 {
            0.0
        } else {
            self.live_rows as f32 / self.populated_archetype_count as f32
        }
    }

    /// Average live rows across *all* archetypes carrying this component,
    /// including empty ones (`live_rows / archetype_count`). Returns `0.0` when
    /// the component appears in no archetype.
    #[inline]
    pub fn avg_rows_per_archetype(&self) -> f32 {
        if self.archetype_count == 0 {
            0.0
        } else {
            self.live_rows as f32 / self.archetype_count as f32
        }
    }

    /// Whether this component appears in more than one archetype — i.e. it
    /// participates in archetype combinatorics rather than being confined to a
    /// single component set.
    #[inline]
    pub fn is_spread(&self) -> bool {
        self.archetype_count > 1
    }
}

/// A whole-world component-distribution summary with a most-spread-first
/// ranking (design §16.6).
///
/// Produced by [`ComponentDistributionReport::capture`].
/// [`entries`](Self::entries) holds every component that appears in at least
/// one archetype, sorted by descending
/// [`archetype_count`](ComponentDistributionEntry::archetype_count) (most
/// spread first), with an ascending component-id tie-break for determinism.
#[derive(Debug, Clone, Default)]
pub struct ComponentDistributionReport {
    /// Components present in at least one archetype, most spread first.
    pub entries: Vec<ComponentDistributionEntry>,
    /// Total registered component types in the world, including any that appear
    /// in no archetype (registered but never spawned onto an entity).
    pub component_count: usize,
    /// Total archetype memberships across all components (the summed
    /// [`archetype_count`](ComponentDistributionEntry::archetype_count)). Equal
    /// to the total number of component slots across every archetype identity;
    /// a useful proxy for overall archetype-graph width.
    pub total_memberships: usize,
    /// Largest [`archetype_count`](ComponentDistributionEntry::archetype_count)
    /// observed, i.e. the spread of the most widely scattered component. Zero
    /// when no component appears in any archetype.
    pub max_archetype_count: usize,
}

impl ComponentDistributionReport {
    /// Capture a component-distribution summary of `world`.
    #[inline]
    pub fn capture(world: &World) -> Self {
        Self::from_world_report(&WorldReport::capture(world))
    }

    /// Fold an existing [`WorldReport`] into a component-distribution summary,
    /// avoiding a second walk of the world when a structural snapshot is
    /// already in hand.
    pub fn from_world_report(report: &WorldReport) -> Self {
        // Accumulate per-component rollups keyed by component id.
        let mut acc: HashMap<ComponentId, Acc> = HashMap::new();
        let mut total_memberships = 0;

        for archetype in &report.archetypes {
            let occ = &archetype.occupancy;
            let populated = occ.live_rows > 0;
            for &component in &archetype.components {
                total_memberships += 1;
                let slot = acc.entry(component).or_default();
                slot.archetype_count += 1;
                slot.live_rows += occ.live_rows;
                slot.chunk_count += occ.chunk_count;
                if populated {
                    slot.populated_archetype_count += 1;
                }
            }
        }

        let mut entries: Vec<ComponentDistributionEntry> = acc
            .into_iter()
            .map(|(id, a)| ComponentDistributionEntry {
                id,
                archetype_count: a.archetype_count,
                populated_archetype_count: a.populated_archetype_count,
                live_rows: a.live_rows,
                chunk_count: a.chunk_count,
            })
            .collect();

        // Most spread first; component id (dense index) breaks ties
        // deterministically — `HashMap` iteration order is otherwise arbitrary.
        entries.sort_by(|a, b| {
            b.archetype_count
                .cmp(&a.archetype_count)
                .then_with(|| a.id.index().cmp(&b.id.index()))
        });

        let max_archetype_count = entries.first().map_or(0, |e| e.archetype_count);

        Self {
            entries,
            component_count: report.component_count,
            total_memberships,
            max_archetype_count,
        }
    }

    /// Whether no component appears in any archetype (nothing to analyze).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of components that appear in at least one archetype.
    ///
    /// May be less than [`component_count`](Self::component_count): a component
    /// can be registered yet never spawned onto an entity, so it has no
    /// distribution footprint and no entry.
    #[inline]
    pub fn analyzed_count(&self) -> usize {
        self.entries.len()
    }

    /// Registered components that appear in no archetype
    /// (`component_count - analyzed_count`).
    #[inline]
    pub fn unused_component_count(&self) -> usize {
        self.component_count.saturating_sub(self.entries.len())
    }

    /// The most widely spread component (highest archetype count), or `None`
    /// when no component appears in any archetype. Ties resolve to the lowest
    /// component id.
    #[inline]
    pub fn most_spread(&self) -> Option<&ComponentDistributionEntry> {
        self.entries.first()
    }

    /// The entry for `id`, if that component appears in at least one archetype.
    pub fn entry(&self, id: ComponentId) -> Option<&ComponentDistributionEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Number of reported components that appear in more than one archetype
    /// (those participating in archetype combinatorics).
    #[inline]
    pub fn spread_component_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_spread()).count()
    }
}

/// Mutable per-component accumulator used while folding archetypes.
#[derive(Default)]
struct Acc {
    archetype_count: usize,
    populated_archetype_count: usize,
    live_rows: usize,
    chunk_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::world::World;

    struct Position;
    impl Component for Position {}
    struct Velocity;
    impl Component for Velocity {}
    struct Tag;
    impl Component for Tag {}

    #[test]
    fn empty_world_has_no_components_to_analyze() {
        let world = World::new();
        let report = ComponentDistributionReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.analyzed_count(), 0);
        assert_eq!(report.total_memberships, 0);
        assert_eq!(report.max_archetype_count, 0);
        assert!(report.most_spread().is_none());
    }

    #[test]
    fn shared_component_spreads_across_archetypes() {
        let mut world = World::new();
        // Position lives in three distinct archetypes; Velocity and Tag in one
        // each. Position is therefore the most spread component.
        world.spawn(Position); //            {Position}
        world.spawn((Position, Velocity)); // {Position, Velocity}
        world.spawn((Position, Tag)); //      {Position, Tag}

        let report = ComponentDistributionReport::capture(&world);

        let pos = world.components().id_of::<Position>().unwrap();
        let vel = world.components().id_of::<Velocity>().unwrap();
        let tag = world.components().id_of::<Tag>().unwrap();

        let pos_entry = report.entry(pos).unwrap();
        assert_eq!(pos_entry.archetype_count, 3, "Position in all three archetypes");
        assert_eq!(pos_entry.populated_archetype_count, 3);
        assert_eq!(pos_entry.live_rows, 3, "one Position per spawned entity");
        assert!(pos_entry.is_spread());

        assert_eq!(report.entry(vel).unwrap().archetype_count, 1);
        assert_eq!(report.entry(vel).unwrap().live_rows, 1);
        assert_eq!(report.entry(tag).unwrap().archetype_count, 1);

        // Most spread first.
        assert_eq!(report.most_spread().unwrap().id, pos);
        assert_eq!(report.max_archetype_count, 3);
        assert_eq!(report.spread_component_count(), 1, "only Position spans >1 archetype");
    }

    #[test]
    fn entries_ranked_by_descending_spread_with_id_tiebreak() {
        let mut world = World::new();
        world.spawn(Position);
        world.spawn((Position, Velocity));
        world.spawn((Position, Tag));

        let report = ComponentDistributionReport::capture(&world);

        for pair in report.entries.windows(2) {
            let (hi, lo) = (&pair[0], &pair[1]);
            assert!(
                hi.archetype_count > lo.archetype_count
                    || (hi.archetype_count == lo.archetype_count
                        && hi.id.index() <= lo.id.index()),
                "entries must be most-spread-first, id-ascending on ties"
            );
        }
    }

    #[test]
    fn live_rows_accumulate_across_archetypes() {
        let mut world = World::new();
        // Two entities in {Position}, three in {Position, Velocity}: Position
        // has five live instances spread over two archetypes.
        world.spawn(Position);
        world.spawn(Position);
        world.spawn((Position, Velocity));
        world.spawn((Position, Velocity));
        world.spawn((Position, Velocity));

        let report = ComponentDistributionReport::capture(&world);
        let pos = world.components().id_of::<Position>().unwrap();
        let entry = report.entry(pos).unwrap();
        assert_eq!(entry.archetype_count, 2);
        assert_eq!(entry.populated_archetype_count, 2);
        assert_eq!(entry.live_rows, 5);
        assert_eq!(entry.avg_rows_per_populated_archetype(), 2.5);
        assert_eq!(entry.avg_rows_per_archetype(), 2.5);
        assert_eq!(entry.empty_archetype_count(), 0);
    }

    #[test]
    fn unused_registered_component_has_no_entry() {
        let mut world = World::new();
        // Register Velocity without ever spawning it onto an entity.
        world.register_component::<Velocity>();
        world.spawn(Position);

        let report = ComponentDistributionReport::capture(&world);
        let vel = world.components().id_of::<Velocity>().unwrap();
        assert!(report.entry(vel).is_none(), "unused component has no footprint");
        assert!(report.unused_component_count() >= 1);
        assert!(report.component_count > report.analyzed_count());
    }

    #[test]
    fn from_world_report_matches_direct_capture() {
        let mut world = World::new();
        world.spawn(Position);
        world.spawn((Position, Velocity));
        world.spawn((Position, Tag));

        let via_world = ComponentDistributionReport::capture(&world);
        let via_report =
            ComponentDistributionReport::from_world_report(&WorldReport::capture(&world));

        assert_eq!(via_world.entries, via_report.entries);
        assert_eq!(via_world.component_count, via_report.component_count);
        assert_eq!(via_world.total_memberships, via_report.total_memberships);
        assert_eq!(via_world.max_archetype_count, via_report.max_archetype_count);
    }
}
