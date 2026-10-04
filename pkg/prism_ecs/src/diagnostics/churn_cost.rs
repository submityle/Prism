//! Churn cost correlation: timing × *gross* structural churn joined per system
//! (design §16.6 / §9 / §10 / §17).
//!
//! [`system_cost`](super::system_cost) joins per-system self time with the
//! *net* change volume from [`frame_profile`](super::frame_profile): changed /
//! added cells and net entity / archetype deltas. Net accounting is exactly
//! what hides thrash — a system that spawns five entities and despawns five
//! others nets to zero and looks free, yet it moved ten entities between
//! archetypes and burned real time doing it.
//!
//! This module is the *gross* dual of [`system_cost`](super::system_cost): it
//! joins the same profiler self time against
//! [`structural_churn`](super::structural_churn), whose figures count absolute
//! entity / archetype movement so opposing moves add instead of cancelling. The
//! join surfaces the signal net cost accounting cannot:
//!
//! * **expensive oscillator** (self time high *and* oscillation non-zero) — a
//!   system burning time thrashing structure that nets to little or nothing;
//!   the prime command-batching target (design §9 并行 ECB / §10 相等保护写).
//! * **expensive mover** (self time high, gross churn high, little
//!   oscillation) — doing genuine one-directional structural work; a
//!   batching / archetype-layout candidate.
//! * **compute-bound** (self time high, gross churn zero) — a read-only or
//!   math-heavy system; the SIMD / owning-group candidate (design §17), the
//!   same classification [`system_cost`](super::system_cost) draws from net
//!   data, confirmed here from the gross side.
//!
//! The join is by **system name**: profiler self time by span label (via
//! [`FlameGraph::hotspots`]), gross churn by the schedule node's `type_name`
//! (via [`SystemChurnEntry::name`](super::structural_churn::SystemChurnEntry::name)).
//! Churn entries whose node name cannot be resolved against the schedule are
//! skipped. The join is read-only: it borrows a [`FlameGraph`], a
//! [`StructuralChurnProfile`], and the [`Schedule`] used to resolve node names,
//! and allocates its own report. [`entries`](ChurnCostProfile::entries) rank by
//! descending self time, with gross-churn and name tie-breaks for determinism.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;

use crate::collections::HashMap;
use crate::diagnostics::profiler::FlameGraph;
use crate::diagnostics::structural_churn::StructuralChurnProfile;
use crate::schedule::graph::Schedule;

/// One system's joined self time and gross structural churn.
///
/// A system may be present on only one side of the join: [`has_timing`] is set
/// when it appeared in the [`FlameGraph`], [`has_churn`] when it appeared in the
/// [`StructuralChurnProfile`]. When both are set the entry is *matched*.
///
/// [`has_timing`]: Self::has_timing
/// [`has_churn`]: Self::has_churn
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChurnCostEntry {
    /// The system name both sides joined on.
    pub name: String,
    /// Self time aggregated from the flame graph (zero for churn-only).
    pub duration: Duration,
    /// Whether this system contributed self time to the flame graph.
    pub has_timing: bool,
    /// Gross entity movement (`Σ |entity_delta|`) folded from the churn profile.
    pub gross_entity_churn: u64,
    /// Gross archetype movement (`Σ |archetype_delta|`) folded from the churn
    /// profile.
    pub gross_archetype_churn: u64,
    /// Entity oscillation: gross entity movement beyond the net delta
    /// (`gross - |net|`), the thrash net accounting hides.
    pub entity_oscillation: u64,
    /// Net live-entity delta folded from the churn profile.
    pub net_entity_delta: i64,
    /// How many times this system ran across the trace.
    pub run_count: usize,
    /// Whether this system contributed gross churn (was observed running).
    pub has_churn: bool,
}

impl ChurnCostEntry {
    /// Total gross structural movement: `gross_entity_churn +
    /// gross_archetype_churn`.
    #[inline]
    pub fn total_gross_churn(&self) -> u64 {
        self.gross_entity_churn + self.gross_archetype_churn
    }

    /// Whether both timing and churn data were joined for this system.
    #[inline]
    pub fn is_matched(&self) -> bool {
        self.has_timing && self.has_churn
    }

    /// Expensive oscillator: timed with non-zero self time *and* non-zero
    /// entity oscillation — time burned thrashing structure that nets out. The
    /// command-batching target net cost accounting misses (design §9 / §10).
    #[inline]
    pub fn is_expensive_oscillator(&self) -> bool {
        self.has_timing && !self.duration.is_zero() && self.entity_oscillation > 0
    }

    /// Compute-bound: timed with non-zero self time but no gross structural
    /// churn — a read-only / math-heavy system (SIMD / owning-group candidate,
    /// design §17).
    #[inline]
    pub fn is_compute_bound(&self) -> bool {
        self.has_timing && !self.duration.is_zero() && self.total_gross_churn() == 0
    }

    /// Average self time per unit of gross churn, in nanoseconds, or `None`
    /// when nothing was moved. A high value flags a system expensive relative
    /// to the structure it actually moved.
    #[inline]
    pub fn nanos_per_churn(&self) -> Option<u128> {
        let churn = self.total_gross_churn();
        if churn == 0 {
            None
        } else {
            Some(self.duration.as_nanos() / churn as u128)
        }
    }
}

/// Per-system timing and gross structural churn joined into one ranked table
/// (design §16.6).
///
/// Produced by [`ChurnCostProfile::join`]. [`entries`](Self::entries) is sorted
/// by descending [`duration`](ChurnCostEntry::duration), with descending
/// [`total_gross_churn`](ChurnCostEntry::total_gross_churn) and then ascending
/// name as deterministic tie-breaks.
#[derive(Debug, Clone, Default)]
pub struct ChurnCostProfile {
    /// One entry per distinct system name seen on either side, hottest first.
    pub entries: Vec<ChurnCostEntry>,
    /// Total self time across every timed system.
    pub total_duration: Duration,
    /// Total gross structural churn across every observed system
    /// (`Σ gross_entity_churn + gross_archetype_churn`).
    pub total_gross_churn: u64,
    /// Total gross entity movement across every observed system
    /// (`Σ gross_entity_churn`), kept separate so frame-level oscillation can
    /// be derived against the net delta below.
    pub total_gross_entity_churn: u64,
    /// Net live-entity delta summed across every observed system. Spawns and
    /// despawns in different systems cancel here even though each contributes
    /// gross churn — that gap is the cross-system thrash.
    pub total_net_entity_delta: i64,
    /// Frame-level entity oscillation: gross entity movement beyond the net
    /// delta (`total_gross_entity_churn - |total_net_entity_delta|`). Unlike the
    /// per-entry [`entity_oscillation`](ChurnCostEntry::entity_oscillation),
    /// which only catches a *single* system that thrashes across the trace,
    /// this catches a spawner / despawner *pair* whose moves cancel at the
    /// frame boundary — the cross-system command-batching target (design §9 /
    /// §10).
    pub total_entity_oscillation: u64,
    /// Number of systems present on both sides (timed and observed).
    pub matched_count: usize,
    /// Number of systems that were timed but moved no structure.
    pub timing_only_count: usize,
    /// Number of systems that moved structure but were not timed.
    pub churn_only_count: usize,
}

/// Churn-side figures aggregated by system name before the join.
struct ChurnAcc {
    gross_entity_churn: u64,
    gross_archetype_churn: u64,
    entity_oscillation: u64,
    net_entity_delta: i64,
    run_count: usize,
}

impl ChurnCostProfile {
    /// Join a [`FlameGraph`]'s per-label self time with a
    /// [`StructuralChurnProfile`]'s per-system gross churn, resolving
    /// churn-side node names against `schedule`.
    ///
    /// Both sides aggregate by system name: flame-graph self time by span
    /// label (via [`FlameGraph::hotspots`]) and gross churn by the schedule
    /// node's `type_name`. Churn entries whose node name cannot be resolved
    /// against `schedule` are skipped. Read-only.
    pub fn join(
        flame: &FlameGraph,
        churn: &StructuralChurnProfile,
        schedule: &Schedule,
    ) -> Self {
        // Fold gross churn by resolved system name. Several nodes sharing a
        // name sum together, matching how the flame graph aggregates labels.
        let mut churn_by_name: HashMap<String, ChurnAcc> = HashMap::new();
        for e in &churn.entries {
            let Some(name) = e.name(schedule) else {
                continue;
            };
            let acc = churn_by_name.entry(name.to_string()).or_insert(ChurnAcc {
                gross_entity_churn: 0,
                gross_archetype_churn: 0,
                entity_oscillation: 0,
                net_entity_delta: 0,
                run_count: 0,
            });
            acc.gross_entity_churn += e.gross_entity_churn;
            acc.gross_archetype_churn += e.gross_archetype_churn;
            acc.entity_oscillation += e.entity_oscillation();
            acc.net_entity_delta += e.net_entity_delta;
            acc.run_count += e.run_count;
        }

        let mut entries: Vec<ChurnCostEntry> = Vec::new();

        // Timed systems first; attach churn data when the name matches and
        // remove the matched accumulator so only churn-only systems remain.
        for (label, duration) in flame.hotspots() {
            match churn_by_name.remove(&label) {
                Some(acc) => entries.push(ChurnCostEntry {
                    name: label,
                    duration,
                    has_timing: true,
                    gross_entity_churn: acc.gross_entity_churn,
                    gross_archetype_churn: acc.gross_archetype_churn,
                    entity_oscillation: acc.entity_oscillation,
                    net_entity_delta: acc.net_entity_delta,
                    run_count: acc.run_count,
                    has_churn: true,
                }),
                None => entries.push(ChurnCostEntry {
                    name: label,
                    duration,
                    has_timing: true,
                    gross_entity_churn: 0,
                    gross_archetype_churn: 0,
                    entity_oscillation: 0,
                    net_entity_delta: 0,
                    run_count: 0,
                    has_churn: false,
                }),
            }
        }

        // Remaining churn accumulators had no timing span.
        for (name, acc) in churn_by_name {
            entries.push(ChurnCostEntry {
                name,
                duration: Duration::ZERO,
                has_timing: false,
                gross_entity_churn: acc.gross_entity_churn,
                gross_archetype_churn: acc.gross_archetype_churn,
                entity_oscillation: acc.entity_oscillation,
                net_entity_delta: acc.net_entity_delta,
                run_count: acc.run_count,
                has_churn: true,
            });
        }

        // Hottest self time first; gross churn then name break ties.
        entries.sort_by(|a, b| {
            b.duration
                .cmp(&a.duration)
                .then_with(|| b.total_gross_churn().cmp(&a.total_gross_churn()))
                .then_with(|| a.name.cmp(&b.name))
        });

        let mut total_duration = Duration::ZERO;
        let mut total_gross_churn = 0;
        let mut total_gross_entity_churn = 0;
        let mut total_net_entity_delta: i64 = 0;
        let mut matched_count = 0;
        let mut timing_only_count = 0;
        let mut churn_only_count = 0;
        for e in &entries {
            total_duration = total_duration.saturating_add(e.duration);
            total_gross_churn += e.total_gross_churn();
            total_gross_entity_churn += e.gross_entity_churn;
            total_net_entity_delta += e.net_entity_delta;
            match (e.has_timing, e.has_churn) {
                (true, true) => matched_count += 1,
                (true, false) => timing_only_count += 1,
                (false, true) => churn_only_count += 1,
                (false, false) => {}
            }
        }
        // Frame-level oscillation is gross-vs-net across *all* systems, not the
        // sum of per-system oscillations: a spawner (+5) and a despawner (-5)
        // each move monotonically (per-system oscillation 0) yet the frame
        // thrashes ten entities that net to zero.
        let total_entity_oscillation =
            total_gross_entity_churn - total_net_entity_delta.unsigned_abs();

        Self {
            entries,
            total_duration,
            total_gross_churn,
            total_gross_entity_churn,
            total_net_entity_delta,
            total_entity_oscillation,
            matched_count,
            timing_only_count,
            churn_only_count,
        }
    }

    /// Whether no system was present on either side.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of distinct systems in the joined table.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The slowest system by self time (the first ranked entry), or `None` when
    /// the profile is empty.
    #[inline]
    pub fn hottest_by_time(&self) -> Option<&ChurnCostEntry> {
        self.entries.first()
    }

    /// The system that moved the most structure, or `None` when the profile is
    /// empty. Ties resolve to the lowest name for determinism.
    pub fn hottest_by_churn(&self) -> Option<&ChurnCostEntry> {
        self.entries.iter().max_by(|a, b| {
            a.total_gross_churn()
                .cmp(&b.total_gross_churn())
                .then_with(|| b.name.cmp(&a.name))
        })
    }

    /// The most time-expensive *single-system* oscillator — the slowest system
    /// (first ranked) whose own entity movement across the trace exceeds its
    /// net delta — or `None` when none qualify. This catches a system that
    /// thrashes structure within itself (e.g. alternating spawn / despawn
    /// across frames); a spawner / despawner *pair* that only cancels at the
    /// frame boundary shows up in
    /// [`total_entity_oscillation`](Self::total_entity_oscillation) instead,
    /// not here. The headline per-system command-batching target (design §9 /
    /// §10).
    pub fn worst_oscillator(&self) -> Option<&ChurnCostEntry> {
        self.entries.iter().find(|e| e.is_expensive_oscillator())
    }

    /// The entry for the system named `name`, if present.
    pub fn entry(&self, name: &str) -> Option<&ChurnCostEntry> {
        self.entries.iter().find(|e| e.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Commands;
    use crate::component::Component;
    use crate::diagnostics::profiler::SpanNode;
    use crate::diagnostics::step_inspector::SteppingInspector;
    use crate::entity::Entity;
    use crate::schedule::{IntoSystemConfigs, Schedule};
    use crate::system::Query;
    use crate::world::World;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Marker;
    impl Component for Marker {}

    // +5 entities via deferred commands.
    fn spawner(mut commands: Commands) {
        for _ in 0..5 {
            commands.spawn(Marker);
        }
    }
    // -5 entities: despawns every Marker the spawner created earlier this frame.
    fn despawner(q: Query<(Entity, &Marker)>, mut commands: Commands) {
        for (e, _) in q.iter() {
            commands.entity(e).despawn();
        }
    }

    fn build() -> Schedule {
        let mut schedule = Schedule::new();
        schedule.add_systems((spawner, despawner).chain());
        schedule
    }

    fn name_of(schedule: &Schedule, needle: &str) -> String {
        schedule
            .nodes
            .iter()
            .map(|n| n.system.name())
            .find(|name| name.contains(needle))
            .unwrap_or_else(|| panic!("missing system {needle}"))
            .to_string()
    }

    fn capture() -> (Schedule, StructuralChurnProfile) {
        let mut world = World::new();
        let mut schedule = build();
        let mut inspector = SteppingInspector::new();
        let trace = inspector.capture_frame(&mut schedule, &mut world);
        let churn = StructuralChurnProfile::from_trace(&trace);
        (schedule, churn)
    }

    #[test]
    fn empty_join_is_empty() {
        let schedule = build();
        let profile = ChurnCostProfile::join(
            &FlameGraph::from_roots(Vec::new()),
            &StructuralChurnProfile::from_trace(&[]),
            &schedule,
        );
        assert!(profile.is_empty());
        assert_eq!(profile.len(), 0);
        assert_eq!(profile.total_gross_churn, 0);
        assert!(profile.hottest_by_time().is_none());
        assert!(profile.worst_oscillator().is_none());
    }

    /// Both systems are timed and moved structure; the slower one (spawner) is
    /// an expensive oscillator because the frame's moves net out but gross
    /// churn is non-zero.
    #[test]
    fn join_matches_timing_and_gross_churn_by_name() {
        let (schedule, churn) = capture();
        let spawner_name = name_of(&schedule, "spawner");
        let despawner_name = name_of(&schedule, "despawner");

        // Synthetic flame graph: spawner slower than despawner.
        let flame = FlameGraph::from_roots(alloc::vec![
            SpanNode::new(spawner_name.clone(), Duration::from_micros(90), Vec::new()),
            SpanNode::new(despawner_name.clone(), Duration::from_micros(40), Vec::new()),
        ]);

        let profile = ChurnCostProfile::join(&flame, &churn, &schedule);
        assert_eq!(profile.len(), 2);
        assert_eq!(profile.matched_count, 2);
        assert_eq!(profile.churn_only_count, 0);

        // Ranked by self time: spawner first.
        assert_eq!(profile.hottest_by_time().unwrap().name, spawner_name);

        let spawner = profile.entry(&spawner_name).unwrap();
        assert!(spawner.is_matched());
        assert_eq!(spawner.gross_entity_churn, 5);
        assert_eq!(spawner.net_entity_delta, 5);
        assert!(!spawner.is_compute_bound());

        let despawner = profile.entry(&despawner_name).unwrap();
        assert_eq!(despawner.gross_entity_churn, 5);
        assert_eq!(despawner.net_entity_delta, -5);

        // The spawner creates the `Marker` archetype on its first spawn
        // (+1 archetype); the despawner only empties it (no archetype removed).
        assert_eq!(spawner.gross_archetype_churn, 1);
        assert_eq!(despawner.gross_archetype_churn, 0);
        // Frame-level: ten entities moved (nothing net survives) plus the one
        // archetype the spawner created — gross churn counts entity + archetype.
        assert_eq!(profile.total_gross_churn, 11);
        // Frame-level: the spawner (+5) and despawner (-5) net to zero yet moved
        // ten entities — pure cross-system oscillation. The archetype creation
        // is net, not oscillation.
        assert_eq!(profile.total_gross_entity_churn, 10);
        assert_eq!(profile.total_net_entity_delta, 0);
        assert_eq!(profile.total_entity_oscillation, 10);

        // Neither system oscillates *by itself* this frame — each moves
        // monotonically — so there is no single-system oscillator to blame; the
        // thrash lives in the spawner/despawner pair (frame-level above).
        assert!(profile.worst_oscillator().is_none());
        assert_eq!(spawner.entity_oscillation, 0);
        assert_eq!(despawner.entity_oscillation, 0);
    }

    /// A churny system never timed shows up as churn-only with zero duration.
    #[test]
    fn churn_without_timing_is_churn_only() {
        let (schedule, churn) = capture();
        let despawner_name = name_of(&schedule, "despawner");

        // Only the despawner is timed.
        let flame = FlameGraph::from_roots(alloc::vec![SpanNode::new(
            despawner_name.clone(),
            Duration::from_micros(40),
            Vec::new(),
        )]);

        let profile = ChurnCostProfile::join(&flame, &churn, &schedule);
        let spawner_name = name_of(&schedule, "spawner");
        let spawner = profile.entry(&spawner_name).unwrap();
        assert!(!spawner.has_timing);
        assert!(spawner.has_churn);
        assert_eq!(spawner.duration, Duration::ZERO);
        assert_eq!(profile.churn_only_count, 1);
        assert_eq!(profile.matched_count, 1);
    }

    /// A timed system with no structural movement is compute-bound and
    /// timing-only.
    #[test]
    fn timing_without_churn_is_compute_bound() {
        let (schedule, churn) = capture();
        let flame = FlameGraph::from_roots(alloc::vec![SpanNode::new(
            String::from("ghost_system"),
            Duration::from_micros(50),
            Vec::new(),
        )]);

        let profile = ChurnCostProfile::join(&flame, &churn, &schedule);
        let ghost = profile.entry("ghost_system").unwrap();
        assert!(ghost.has_timing);
        assert!(!ghost.has_churn);
        assert!(ghost.is_compute_bound());
        assert_eq!(ghost.total_gross_churn(), 0);
        assert!(ghost.nanos_per_churn().is_none());
        assert_eq!(profile.timing_only_count, 1);
    }

    /// A *single* system that spawns on one frame and despawns on the next
    /// oscillates by itself: gross entity churn ten, net zero across the
    /// two-frame trace — exactly the per-system thrash `worst_oscillator` is
    /// meant to surface (contrast the spawner/despawner pair above, whose
    /// oscillation is only frame-level).
    #[test]
    fn single_system_oscillator_is_flagged() {
        use crate::resource::Resource;
        use crate::system::ResMut;

        #[derive(Debug)]
        struct Phase(bool);
        impl Resource for Phase {}

        // Spawns five on `true` frames, despawns every Marker on `false`
        // frames, flipping each run. Over two frames: +5 then -5.
        fn churner(
            mut phase: ResMut<Phase>,
            q: Query<(Entity, &Marker)>,
            mut commands: Commands,
        ) {
            if phase.0 {
                for _ in 0..5 {
                    commands.spawn(Marker);
                }
            } else {
                for (e, _) in q.iter() {
                    commands.entity(e).despawn();
                }
            }
            phase.0 = !phase.0;
        }

        let mut world = World::new();
        world.insert_resource(Phase(true));
        let mut schedule = Schedule::new();
        schedule.add_systems(churner);

        let mut inspector = SteppingInspector::new();
        let mut trace = inspector.capture_frame(&mut schedule, &mut world); // +5
        trace.extend(inspector.capture_frame(&mut schedule, &mut world)); // -5
        let churn = StructuralChurnProfile::from_trace(&trace);

        let churner_name = name_of(&schedule, "churner");
        let flame = FlameGraph::from_roots(alloc::vec![SpanNode::new(
            churner_name.clone(),
            Duration::from_micros(70),
            Vec::new(),
        )]);

        let profile = ChurnCostProfile::join(&flame, &churn, &schedule);
        let churner = profile.entry(&churner_name).unwrap();
        assert!(churner.is_matched());
        assert_eq!(churner.gross_entity_churn, 10, "five up, five down");
        assert_eq!(churner.net_entity_delta, 0, "nets to nothing");
        assert_eq!(churner.entity_oscillation, 10);
        assert!(churner.is_expensive_oscillator());
        assert!(!churner.is_compute_bound());

        // Now a single system owns the thrash, so it is the headline target —
        // and the frame-level total agrees.
        assert_eq!(profile.worst_oscillator().unwrap().name, churner_name);
        assert_eq!(profile.total_entity_oscillation, 10);
    }
}
