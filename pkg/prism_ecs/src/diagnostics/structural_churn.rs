//! Structural churn profile (design §16.6 / §9 / §10).
//!
//! [`FrameChangeProfile`](super::frame_profile::FrameChangeProfile) ranks
//! systems by how many *cells* they dirtied and reports entity / archetype
//! deltas as **net** figures: a system that spawns five entities and another
//! that despawns five net out to zero movement over the frame. Net is the right
//! lens for "how much did the world's size change", but it hides *structural
//! work* — the archetype migrations and allocator traffic that the design's
//! command/structural-change batching targets (design §9), and whose cost the
//! headline "成本 ∝ 变化量" (design §10) is about containing.
//!
//! This module takes the dual, **gross** lens over the same
//! [`StepObservation`] trace: it accumulates absolute live-entity deltas and
//! actual archetype creations plus permanent evictions. Archetype moves add
//! even when they cancel within one step. Within a fully measured trace, the
//! gap between gross and net — the *oscillation* — is pure structural thrash:
//! churn that produced no lasting size change and is the
//! prime candidate for command batching or scheduling changes (design §9).
//! Registry-reset observations retain live net deltas but contribute no gross
//! archetype churn; oscillation across those gaps is only the measured excess.
//!
//! Aggregation is by schedule node, so concatenating several frames' traces (or
//! a node observed more than once) sums cleanly, and a node that oscillates
//! across frames surfaces its thrash at the node level. Frame boundaries carry
//! no node and are ignored. [`entries`](StructuralChurnProfile::entries) are
//! ranked by descending total churn with a deterministic node-index tie-break,
//! so repeated captures of the same workload rank identically.

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::diagnostics::step_inspector::StepObservation;
use crate::schedule::graph::Schedule;

/// Gross structural movement a single schedule node drove over a trace.
///
/// *Gross* entity figures sum absolute per-observation live-count deltas;
/// *gross* archetype figures sum actual creations and permanent evictions.
/// *Net* figures sum signed live-count deltas. At a registry reset (e.g. world
/// restore), archetype gross is unavailable and contributes zero, while net
/// still describes the size change. Oscillation is the non-negative excess
/// of measured gross over the absolute net.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemChurnEntry {
    /// The schedule node index this entry aggregates.
    pub node: usize,
    /// Whether any observation of this node actually ran.
    pub ran: bool,
    /// Number of observations in which this node ran (vs. was gated off).
    pub run_count: usize,
    /// Number of observations in which this node observably moved entities or
    /// archetypes (a non-zero live delta or measured archetype churn).
    pub migrating_observation_count: usize,
    /// Summed absolute live-entity movement across the trace
    /// (`Σ |entity_delta|`).
    pub gross_entity_churn: u64,
    /// Summed archetype creations plus permanent evictions across the trace.
    /// Registry-reset observations contribute zero because their counters
    /// belong to different generations.
    pub gross_archetype_churn: u64,
    /// Net live-entity delta across the trace (negative for net despawns).
    pub net_entity_delta: i64,
    /// Net archetype-count delta across the trace.
    pub net_archetype_delta: isize,
}

impl SystemChurnEntry {
    /// Total gross structural movement: `gross_entity_churn +
    /// gross_archetype_churn`. The primary ranking key.
    #[inline]
    pub fn total_churn(&self) -> u64 {
        self.gross_entity_churn + self.gross_archetype_churn
    }

    /// Entity-count oscillation: gross entity movement that produced no lasting
    /// size change (`gross_entity_churn - |net_entity_delta|`). High values are
    /// spawn/despawn thrash a command batch could absorb (design §9).
    #[inline]
    pub fn entity_oscillation(&self) -> u64 {
        self.gross_entity_churn
            .saturating_sub(self.net_entity_delta.unsigned_abs())
    }

    /// Archetype-count oscillation: gross archetype movement with no lasting
    /// change (`gross_archetype_churn - |net_archetype_delta|`).
    #[inline]
    pub fn archetype_oscillation(&self) -> u64 {
        self.gross_archetype_churn
            .saturating_sub(self.net_archetype_delta.unsigned_abs() as u64)
    }

    /// Whether this node moved any structure at all over the trace.
    #[inline]
    pub fn is_migrating(&self) -> bool {
        self.total_churn() > 0 || self.migrating_observation_count > 0
    }

    /// Resolve this node's system name against `schedule` (its `type_name`,
    /// design §16.6 inspector label), or `None` if the node is out of range.
    pub fn name<'s>(&self, schedule: &'s Schedule) -> Option<&'s str> {
        schedule.nodes.get(self.node).map(|n| n.system.name())
    }
}

/// A trace's structural churn folded into a per-system ranking (design §16.6).
///
/// Produced by [`StructuralChurnProfile::from_trace`].
/// [`entries`](Self::entries) is sorted by [`total_churn`](SystemChurnEntry::total_churn)
/// descending, with ties broken by ascending node index for determinism.
#[derive(Debug, Clone, Default)]
pub struct StructuralChurnProfile {
    /// One entry per schedule node observed in the trace, highest churn first.
    pub entries: Vec<SystemChurnEntry>,
    /// Total gross entity movement across every system (`Σ |entity_delta|`).
    pub total_gross_entity_churn: u64,
    /// Total measured archetype creations plus permanent evictions across
    /// every system. Registry-reset observations contribute zero.
    pub total_gross_archetype_churn: u64,
    /// Net live-entity delta across the whole trace.
    pub total_net_entity_delta: i64,
    /// Net archetype-count delta across the whole trace.
    pub total_net_archetype_delta: isize,
    /// Number of distinct nodes that moved any structure.
    pub migrating_system_count: usize,
}

impl StructuralChurnProfile {
    /// Fold a [`StepObservation`] trace into a ranked per-system churn profile.
    ///
    /// Frame-boundary observations are ignored. Observations are aggregated by
    /// schedule node, so a node seen more than once sums its contributions.
    /// Read-only: it borrows the trace and allocates its own report.
    pub fn from_trace(trace: &[StepObservation]) -> Self {
        let mut by_node: HashMap<usize, SystemChurnEntry> = HashMap::new();
        for obs in trace {
            let Some(node) = obs.node() else {
                continue;
            };
            let entry = by_node.entry(node).or_insert(SystemChurnEntry {
                node,
                ran: false,
                run_count: 0,
                migrating_observation_count: 0,
                gross_entity_churn: 0,
                gross_archetype_churn: 0,
                net_entity_delta: 0,
                net_archetype_delta: 0,
            });

            if obs.ran() {
                entry.ran = true;
                entry.run_count += 1;
            }

            let ed = obs.entity_delta();
            let ad = obs.archetype_delta();
            let archetype_churn = obs.gross_archetype_churn();
            entry.gross_entity_churn += ed.unsigned_abs();
            entry.gross_archetype_churn += archetype_churn;
            entry.net_entity_delta += ed;
            entry.net_archetype_delta += ad;
            if ed != 0 || ad != 0 || archetype_churn != 0 {
                entry.migrating_observation_count += 1;
            }
        }

        let mut entries: Vec<SystemChurnEntry> = by_node.into_values().collect();
        // Highest churn first; node index breaks ties deterministically.
        entries.sort_by(|a, b| {
            b.total_churn()
                .cmp(&a.total_churn())
                .then_with(|| a.node.cmp(&b.node))
        });

        let mut total_gross_entity_churn = 0;
        let mut total_gross_archetype_churn = 0;
        let mut total_net_entity_delta = 0;
        let mut total_net_archetype_delta = 0;
        let mut migrating_system_count = 0;
        for e in &entries {
            total_gross_entity_churn += e.gross_entity_churn;
            total_gross_archetype_churn += e.gross_archetype_churn;
            total_net_entity_delta += e.net_entity_delta;
            total_net_archetype_delta += e.net_archetype_delta;
            if e.is_migrating() {
                migrating_system_count += 1;
            }
        }

        Self {
            entries,
            total_gross_entity_churn,
            total_gross_archetype_churn,
            total_net_entity_delta,
            total_net_archetype_delta,
            migrating_system_count,
        }
    }

    /// Whether no system was observed (an empty or boundary-only trace).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of distinct systems profiled.
    #[inline]
    pub fn system_count(&self) -> usize {
        self.entries.len()
    }

    /// Total gross structural movement across the trace: entity + archetype.
    #[inline]
    pub fn total_gross_churn(&self) -> u64 {
        self.total_gross_entity_churn + self.total_gross_archetype_churn
    }

    /// World-wide entity oscillation: gross entity movement that produced no
    /// lasting size change (`total_gross_entity_churn - |total_net_entity_delta|`).
    /// A large value means the frame did a lot of spawn/despawn work that
    /// netted out — the batching opportunity the design targets (design §9).
    #[inline]
    pub fn total_entity_oscillation(&self) -> u64 {
        self.total_gross_entity_churn
            .saturating_sub(self.total_net_entity_delta.unsigned_abs())
    }

    /// World-wide archetype oscillation
    /// (`total_gross_archetype_churn - |total_net_archetype_delta|`).
    #[inline]
    pub fn total_archetype_oscillation(&self) -> u64 {
        self.total_gross_archetype_churn
            .saturating_sub(self.total_net_archetype_delta.unsigned_abs() as u64)
    }

    /// The highest-churn system, or `None` for an empty profile. This is the
    /// first ranked entry, so ties resolve to the lowest node index.
    #[inline]
    pub fn hottest(&self) -> Option<&SystemChurnEntry> {
        self.entries.first()
    }

    /// The entry for `node`, if it was observed.
    pub fn entry(&self, node: usize) -> Option<&SystemChurnEntry> {
        self.entries.iter().find(|e| e.node == node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Commands;
    use crate::component::{Component, ComponentId, Components, StorageType};
    use crate::diagnostics::step_inspector::SteppingInspector;
    use crate::schedule::{IntoSystemConfigs, Schedule};
    use crate::system::Query;
    use crate::world::World;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Marker;
    impl Component for Marker {}

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct SharedKey(u32);
    impl Component for SharedKey {
        const STORAGE: StorageType = StorageType::Shared;

        fn install_storage_glue(components: &mut Components, id: ComponentId) {
            components.set_shared_box(id, crate::component::shared_box_of::<Self>());
        }
    }

    fn shared_cycle(world: &mut World) {
        let entity = world.spawn(SharedKey(0));
        assert!(world.despawn(entity));
    }

    // Spawns five Marker entities via deferred commands: +5 entities.
    fn spawner(mut commands: Commands) {
        for _ in 0..5 {
            commands.spawn(Marker);
        }
    }
    // Despawns every Marker entity it can see: -5 entities (after the spawner
    // ran earlier in the same frame).
    fn despawner(q: Query<(crate::entity::Entity, &Marker)>, mut commands: Commands) {
        for (e, _) in q.iter() {
            commands.entity(e).despawn();
        }
    }

    fn build() -> Schedule {
        let mut schedule = Schedule::new();
        schedule.add_systems((spawner, despawner).chain());
        schedule
    }

    fn profile_one_frame() -> (Schedule, StructuralChurnProfile) {
        let mut world = World::new();
        let mut schedule = build();
        let mut inspector = SteppingInspector::new();
        let trace = inspector.capture_frame(&mut schedule, &mut world);
        let profile = StructuralChurnProfile::from_trace(&trace);
        (schedule, profile)
    }

    fn entry_named<'a>(
        profile: &'a StructuralChurnProfile,
        schedule: &Schedule,
        needle: &str,
    ) -> &'a SystemChurnEntry {
        profile
            .entries
            .iter()
            .find(|e| e.name(schedule).unwrap().contains(needle))
            .unwrap_or_else(|| panic!("missing entry for {needle}"))
    }

    #[test]
    fn empty_trace_is_empty() {
        let profile = StructuralChurnProfile::from_trace(&[]);
        assert!(profile.is_empty());
        assert_eq!(profile.system_count(), 0);
        assert_eq!(profile.total_gross_churn(), 0);
        assert_eq!(profile.migrating_system_count, 0);
        assert!(profile.hottest().is_none());
    }

    /// A spawn-then-despawn frame nets to zero entities but has gross entity
    /// churn of ten — the thrash net accounting hides.
    #[test]
    fn gross_churn_counts_opposing_moves_that_net_out() {
        let (schedule, profile) = profile_one_frame();

        let spawner = entry_named(&profile, &schedule, "spawner");
        assert_eq!(spawner.net_entity_delta, 5);
        assert_eq!(spawner.gross_entity_churn, 5);
        assert!(spawner.ran);

        let despawner = entry_named(&profile, &schedule, "despawner");
        assert_eq!(despawner.net_entity_delta, -5);
        assert_eq!(despawner.gross_entity_churn, 5);

        // Frame-level: net cancels, gross does not.
        assert_eq!(
            profile.total_net_entity_delta, 0,
            "spawns and despawns cancel"
        );
        assert_eq!(
            profile.total_gross_entity_churn, 10,
            "but ten entities were actually moved"
        );
        assert_eq!(
            profile.total_entity_oscillation(),
            10,
            "all entity movement was pure oscillation this frame"
        );
        assert_eq!(profile.migrating_system_count, 2);
    }

    #[test]
    fn same_step_archetype_churn_accumulates_despite_zero_net_delta() {
        let mut world = World::new();
        let mut schedule = Schedule::new();
        schedule.add_systems(shared_cycle);
        let mut inspector = SteppingInspector::new();
        let mut trace = inspector.capture_frame(&mut schedule, &mut world);
        let profile = StructuralChurnProfile::from_trace(&trace);
        let entry = profile.hottest().unwrap();

        assert_eq!(entry.gross_entity_churn, 0);
        assert_eq!(entry.net_archetype_delta, 0);
        assert_eq!(entry.gross_archetype_churn, 2);
        assert_eq!(entry.archetype_oscillation(), 2);
        assert_eq!(entry.migrating_observation_count, 1);
        assert!(entry.is_migrating());
        assert_eq!(profile.migrating_system_count, 1);
        assert_eq!(profile.total_gross_archetype_churn, 2);
        assert_eq!(profile.total_net_archetype_delta, 0);
        assert_eq!(profile.total_archetype_oscillation(), 2);

        trace.extend(inspector.capture_frame(&mut schedule, &mut world));
        let repeated = StructuralChurnProfile::from_trace(&trace);
        let entry = repeated.hottest().unwrap();
        assert_eq!(entry.run_count, 2);
        assert_eq!(entry.migrating_observation_count, 2);
        assert_eq!(entry.gross_archetype_churn, 4);
        assert_eq!(entry.net_archetype_delta, 0);
        assert_eq!(repeated.total_gross_archetype_churn, 4);
        assert_eq!(repeated.total_archetype_oscillation(), 4);
    }

    #[test]
    fn net_archetype_decline_counts_permanent_evictions() {
        let mut world = World::new();
        let first = world.spawn(SharedKey(0));
        let second = world.spawn(SharedKey(1));
        let mut schedule = Schedule::new();
        schedule.add_systems(move |world: &mut World| {
            assert!(world.despawn(first));
            assert!(world.despawn(second));
        });
        let trace = SteppingInspector::new().capture_frame(&mut schedule, &mut world);
        let profile = StructuralChurnProfile::from_trace(&trace);
        let entry = profile.hottest().unwrap();

        assert_eq!(entry.net_archetype_delta, -2);
        assert_eq!(entry.gross_archetype_churn, 2);
        assert_eq!(entry.archetype_oscillation(), 0);
        assert_eq!(profile.total_net_archetype_delta, -2);
        assert_eq!(profile.total_gross_archetype_churn, 2);
        assert_eq!(profile.total_archetype_oscillation(), 0);
    }

    #[test]
    fn restore_keeps_net_delta_without_fabricating_gross_churn() {
        let mut world = World::new();
        let empty = world.snapshot();
        for _ in 0..3 {
            shared_cycle(&mut world);
        }
        // Empty ordinary archetypes stay live until restore rebuilds the
        // registry. Their disappearance remains a net delta, not an eviction.
        let entity = world.spawn(Marker);
        assert!(world.despawn(entity));
        let mut schedule = Schedule::new();
        schedule
            .add_systems((move |world: &mut World| world.restore(&empty), shared_cycle).chain());
        let trace = SteppingInspector::new().capture_frame(&mut schedule, &mut world);
        assert!(trace[0].archetype_registry_reset);
        let profile = StructuralChurnProfile::from_trace(&trace);
        let reset = profile.entry(trace[0].node().unwrap()).unwrap();

        assert_eq!(reset.net_archetype_delta, -1);
        assert_eq!(reset.gross_archetype_churn, 0);
        assert_eq!(reset.archetype_oscillation(), 0);
        assert_eq!(reset.migrating_observation_count, 1);
        assert!(reset.is_migrating());
        assert_eq!(profile.migrating_system_count, 2);
        assert_eq!(profile.total_net_archetype_delta, -1);
        assert_eq!(profile.total_gross_archetype_churn, 2);
        assert_eq!(profile.total_archetype_oscillation(), 1);
    }

    /// Entries rank by descending total churn with a node-index tie-break.
    #[test]
    fn entries_rank_by_descending_churn() {
        let (_schedule, profile) = profile_one_frame();
        for pair in profile.entries.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            assert!(
                a.total_churn() > b.total_churn()
                    || (a.total_churn() == b.total_churn() && a.node < b.node),
                "entries must be churn-descending, node-ascending on ties"
            );
        }
    }

    /// Concatenating two frames sums each node's churn rather than duplicating
    /// entries, and the per-node oscillation accumulates.
    #[test]
    fn repeated_nodes_aggregate_churn() {
        let mut world = World::new();
        let mut schedule = build();
        let mut inspector = SteppingInspector::new();
        let mut trace = inspector.capture_frame(&mut schedule, &mut world);
        trace.extend(inspector.capture_frame(&mut schedule, &mut world));

        let profile = StructuralChurnProfile::from_trace(&trace);
        assert_eq!(profile.system_count(), 2, "still two distinct nodes");

        let spawner = entry_named(&profile, &schedule, "spawner");
        assert_eq!(
            spawner.gross_entity_churn, 10,
            "five spawned in each of two frames"
        );
        assert_eq!(spawner.net_entity_delta, 10);
        assert_eq!(spawner.run_count, 2);

        assert_eq!(profile.total_gross_entity_churn, 20);
        assert_eq!(profile.total_net_entity_delta, 0, "each frame nets to zero");
        assert_eq!(profile.total_entity_oscillation(), 20);
    }

    /// `entry(node)` resolves an observed node and rejects an unobserved one.
    #[test]
    fn entry_lookup_by_node() {
        let (schedule, profile) = profile_one_frame();
        let spawner = entry_named(&profile, &schedule, "spawner");
        let node = spawner.node;
        assert_eq!(profile.entry(node).unwrap().node, node);
        assert!(profile.entry(usize::MAX).is_none());
    }
}
