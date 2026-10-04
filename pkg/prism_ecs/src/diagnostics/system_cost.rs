//! System cost correlation: timing × change-volume joined per system
//! (design §16.6 / §17 / §10).
//!
//! [`profiler`](super::profiler) answers *how long* each system ran (self time,
//! a [`FlameGraph`]); [`frame_profile`](super::frame_profile) answers *how much*
//! each system changed ([`FrameChangeProfile`]). On their own neither tells you
//! which systems are the ones worth optimizing. The interesting signal is the
//! *join* of the two:
//!
//! * **hot and churny** (time + storage change both high) — the prime target
//!   for command batching / scheduling work (design §9/§10).
//! * **compute-bound** (time high, storage change zero) — a read-only or
//!   math-heavy system; the SIMD / owning-group candidate (design §17).
//! * **change-only** (changed storage but never timed) — a system the profiler
//!   run did not instrument, surfaced so the two views can be reconciled.
//!
//! This module performs that join by **system name**: profiler self time is
//! aggregated by span label, change volume is aggregated by the schedule node's
//! `type_name`, and the two are matched on the shared name. Several schedule
//! nodes (or spans) that share a name sum together, matching how each source
//! aggregates internally. The join is read-only: it borrows a [`FlameGraph`], a
//! [`FrameChangeProfile`], and the [`Schedule`] used to resolve node names, and
//! allocates its own report. [`entries`](SystemCostProfile::entries) rank by
//! descending self time, with total-cell and name tie-breaks for determinism.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;

use crate::collections::HashMap;
use crate::diagnostics::frame_profile::FrameChangeProfile;
use crate::diagnostics::profiler::FlameGraph;
use crate::schedule::graph::Schedule;

/// One system's joined timing and change-volume figures.
///
/// A system may be present on only one side of the join: [`has_timing`] is set
/// when it appeared in the [`FlameGraph`], [`has_change`] when it appeared in
/// the [`FrameChangeProfile`]. When both are set the entry is *matched*.
///
/// [`has_timing`]: Self::has_timing
/// [`has_change`]: Self::has_change
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemCostEntry {
    /// The system name both sides joined on.
    pub name: String,
    /// Self time aggregated from the flame graph (zero for change-only).
    pub duration: Duration,
    /// Whether this system contributed self time to the flame graph.
    pub has_timing: bool,
    /// Changed-cell count folded from the change profile.
    pub changed_cells: usize,
    /// Added-cell count folded from the change profile.
    pub added_cells: usize,
    /// Dirty-chunk count folded from the change profile.
    pub dirty_chunks: usize,
    /// Net live-entity delta folded from the change profile.
    pub entity_delta: i64,
    /// Net archetype-count delta folded from the change profile.
    pub archetype_delta: isize,
    /// Whether this system contributed change volume (was observed running).
    pub has_change: bool,
}

impl SystemCostEntry {
    /// Total touched cells: `changed_cells + added_cells`.
    #[inline]
    pub fn total_cells(&self) -> usize {
        self.changed_cells + self.added_cells
    }

    /// Whether both timing and change data were joined for this system.
    #[inline]
    pub fn is_matched(&self) -> bool {
        self.has_timing && self.has_change
    }

    /// Hot *and* churny: timed with non-zero self time and non-zero touched
    /// cells. The command-batching / scheduling optimization target
    /// (design §9/§10).
    #[inline]
    pub fn is_hot_and_churny(&self) -> bool {
        self.has_timing && !self.duration.is_zero() && self.total_cells() > 0
    }

    /// Compute-bound: timed with non-zero self time but no storage change. A
    /// read-only / math-heavy system — the SIMD / owning-group candidate
    /// (design §17).
    #[inline]
    pub fn is_compute_bound(&self) -> bool {
        self.has_timing && !self.duration.is_zero() && self.total_cells() == 0
    }

    /// Average self time spent per touched cell, in nanoseconds, or `None` when
    /// no cells were touched. A rough per-unit-work efficiency figure: a high
    /// value flags a system that is expensive relative to the work it did.
    #[inline]
    pub fn nanos_per_cell(&self) -> Option<u128> {
        let cells = self.total_cells();
        if cells == 0 {
            None
        } else {
            Some(self.duration.as_nanos() / cells as u128)
        }
    }
}

/// Per-system timing and change volume joined into one ranked table
/// (design §16.6).
///
/// Produced by [`SystemCostProfile::join`]. [`entries`](Self::entries) is sorted
/// by descending [`duration`](SystemCostEntry::duration), with descending
/// [`total_cells`](SystemCostEntry::total_cells) and then ascending name as
/// deterministic tie-breaks.
#[derive(Debug, Clone, Default)]
pub struct SystemCostProfile {
    /// One entry per distinct system name seen on either side, hottest first.
    pub entries: Vec<SystemCostEntry>,
    /// Total self time across every timed system.
    pub total_duration: Duration,
    /// Total touched cells across every changed system.
    pub total_cells: usize,
    /// Number of systems present on both sides (timed and changed).
    pub matched_count: usize,
    /// Number of systems that were timed but changed no storage.
    pub timing_only_count: usize,
    /// Number of systems that changed storage but were not timed.
    pub change_only_count: usize,
}

/// Change-side figures aggregated by system name before the join.
struct ChangeAcc {
    changed_cells: usize,
    added_cells: usize,
    dirty_chunks: usize,
    entity_delta: i64,
    archetype_delta: isize,
}

impl SystemCostProfile {
    /// Join a [`FlameGraph`]'s per-label self time with a
    /// [`FrameChangeProfile`]'s per-system change volume, resolving change-side
    /// node names against `schedule`.
    ///
    /// Both sides aggregate by system name: flame-graph self time by span
    /// label (via [`FlameGraph::hotspots`]) and change volume by the schedule
    /// node's `type_name`. Change entries whose node name cannot be resolved
    /// against `schedule` are skipped. Read-only.
    pub fn join(
        flame: &FlameGraph,
        changes: &FrameChangeProfile,
        schedule: &Schedule,
    ) -> Self {
        // Fold change volume by resolved system name.
        let mut change_by_name: HashMap<String, ChangeAcc> = HashMap::new();
        for e in &changes.entries {
            let Some(name) = e.name(schedule) else {
                continue;
            };
            let acc = change_by_name
                .entry(name.to_string())
                .or_insert(ChangeAcc {
                    changed_cells: 0,
                    added_cells: 0,
                    dirty_chunks: 0,
                    entity_delta: 0,
                    archetype_delta: 0,
                });
            acc.changed_cells += e.changed_cells;
            acc.added_cells += e.added_cells;
            acc.dirty_chunks += e.dirty_chunks;
            acc.entity_delta += e.entity_delta;
            acc.archetype_delta += e.archetype_delta;
        }

        let mut entries: Vec<SystemCostEntry> = Vec::new();

        // Timed systems first; attach change data when the name matches, and
        // remove the matched change accumulator so only change-only systems
        // remain afterwards.
        for (label, duration) in flame.hotspots() {
            let matched = change_by_name.remove(&label);
            let (changed_cells, added_cells, dirty_chunks, entity_delta, archetype_delta, has_change) =
                match matched {
                    Some(acc) => (
                        acc.changed_cells,
                        acc.added_cells,
                        acc.dirty_chunks,
                        acc.entity_delta,
                        acc.archetype_delta,
                        true,
                    ),
                    None => (0, 0, 0, 0, 0, false),
                };
            entries.push(SystemCostEntry {
                name: label,
                duration,
                has_timing: true,
                changed_cells,
                added_cells,
                dirty_chunks,
                entity_delta,
                archetype_delta,
                has_change,
            });
        }

        // Remaining change accumulators had no timing span.
        for (name, acc) in change_by_name {
            entries.push(SystemCostEntry {
                name,
                duration: Duration::ZERO,
                has_timing: false,
                changed_cells: acc.changed_cells,
                added_cells: acc.added_cells,
                dirty_chunks: acc.dirty_chunks,
                entity_delta: acc.entity_delta,
                archetype_delta: acc.archetype_delta,
                has_change: true,
            });
        }

        // Hottest self time first; total cells then name break ties.
        entries.sort_by(|a, b| {
            b.duration
                .cmp(&a.duration)
                .then_with(|| b.total_cells().cmp(&a.total_cells()))
                .then_with(|| a.name.cmp(&b.name))
        });

        let mut total_duration = Duration::ZERO;
        let mut total_cells = 0;
        let mut matched_count = 0;
        let mut timing_only_count = 0;
        let mut change_only_count = 0;
        for e in &entries {
            total_duration = total_duration.saturating_add(e.duration);
            total_cells += e.total_cells();
            match (e.has_timing, e.has_change) {
                (true, true) => matched_count += 1,
                (true, false) => timing_only_count += 1,
                (false, true) => change_only_count += 1,
                (false, false) => {}
            }
        }

        Self {
            entries,
            total_duration,
            total_cells,
            matched_count,
            timing_only_count,
            change_only_count,
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
    pub fn hottest_by_time(&self) -> Option<&SystemCostEntry> {
        self.entries.first()
    }

    /// The system with the most touched cells, or `None` when the profile is
    /// empty. Ties resolve to the lowest name for determinism.
    pub fn hottest_by_churn(&self) -> Option<&SystemCostEntry> {
        self.entries.iter().max_by(|a, b| {
            a.total_cells()
                .cmp(&b.total_cells())
                .then_with(|| b.name.cmp(&a.name))
        })
    }

    /// The entry for the system named `name`, if present.
    pub fn entry(&self, name: &str) -> Option<&SystemCostEntry> {
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
    use crate::schedule::{IntoSystemConfigs, Schedule};
    use crate::system::Query;
    use crate::world::World;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Marker;
    impl Component for Marker {}

    // Spawns five Marker entities: touches storage (adds cells).
    fn spawner(mut commands: Commands) {
        for _ in 0..5 {
            commands.spawn(Marker);
        }
    }
    // Reads Marker without writing: pure compute, no storage change.
    fn reader(q: Query<&Marker>) {
        let mut n = 0usize;
        for _ in q.iter() {
            n += 1;
        }
        core::hint::black_box(n);
    }

    fn build() -> Schedule {
        let mut schedule = Schedule::new();
        schedule.add_systems((spawner, reader).chain());
        schedule
    }

    /// Resolve each node's `type_name` so the synthetic flame graph can use the
    /// same labels the change profile will join on.
    fn name_of(schedule: &Schedule, needle: &str) -> String {
        schedule
            .nodes
            .iter()
            .map(|n| n.system.name())
            .find(|name| name.contains(needle))
            .unwrap_or_else(|| panic!("missing system {needle}"))
            .to_string()
    }

    fn capture() -> (Schedule, FrameChangeProfile) {
        let mut world = World::new();
        let mut schedule = build();
        let mut inspector = SteppingInspector::new();
        let trace = inspector.capture_frame(&mut schedule, &mut world);
        let changes = FrameChangeProfile::from_trace(&trace);
        (schedule, changes)
    }

    #[test]
    fn empty_join_is_empty() {
        let schedule = build();
        let profile = SystemCostProfile::join(
            &FlameGraph::from_roots(Vec::new()),
            &FrameChangeProfile::from_trace(&[]),
            &schedule,
        );
        assert!(profile.is_empty());
        assert_eq!(profile.len(), 0);
        assert_eq!(profile.total_cells, 0);
        assert!(profile.hottest_by_time().is_none());
        assert!(profile.hottest_by_churn().is_none());
    }

    /// A matched join: spawner is hot-and-churny (timed + added cells), reader
    /// is compute-bound (timed, zero cells).
    #[test]
    fn join_matches_timing_and_change_by_name() {
        let (schedule, changes) = capture();
        let spawner_name = name_of(&schedule, "spawner");
        let reader_name = name_of(&schedule, "reader");

        // Synthetic flame graph: spawner slower than reader.
        let flame = FlameGraph::from_roots(alloc::vec![
            SpanNode::new(spawner_name.clone(), Duration::from_micros(90), Vec::new()),
            SpanNode::new(reader_name.clone(), Duration::from_micros(30), Vec::new()),
        ]);

        let profile = SystemCostProfile::join(&flame, &changes, &schedule);
        assert_eq!(profile.len(), 2);
        assert_eq!(profile.matched_count, 2, "both systems were timed and observed");
        assert_eq!(profile.change_only_count, 0);

        // Ranked by self time: spawner first.
        assert_eq!(profile.hottest_by_time().unwrap().name, spawner_name);

        let spawner = profile.entry(&spawner_name).unwrap();
        assert!(spawner.is_matched());
        assert!(spawner.is_hot_and_churny(), "spawner added cells and took time");
        assert!(!spawner.is_compute_bound());
        assert!(spawner.added_cells > 0);
        assert_eq!(spawner.entity_delta, 5);

        let reader = profile.entry(&reader_name).unwrap();
        assert!(reader.is_matched());
        assert!(reader.is_compute_bound(), "reader took time but changed nothing");
        assert!(!reader.is_hot_and_churny());
        assert_eq!(reader.total_cells(), 0);
        assert!(reader.nanos_per_cell().is_none());

        // Churn ranking picks the spawner (only one with touched cells).
        assert_eq!(profile.hottest_by_churn().unwrap().name, spawner_name);
    }

    /// A system that changed storage but was never timed shows up as
    /// change-only with zero duration.
    #[test]
    fn change_without_timing_is_change_only() {
        let (schedule, changes) = capture();
        let reader_name = name_of(&schedule, "reader");

        // Flame graph only times the reader, not the spawner.
        let flame = FlameGraph::from_roots(alloc::vec![SpanNode::new(
            reader_name.clone(),
            Duration::from_micros(30),
            Vec::new(),
        )]);

        let profile = SystemCostProfile::join(&flame, &changes, &schedule);
        let spawner_name = name_of(&schedule, "spawner");
        let spawner = profile.entry(&spawner_name).unwrap();
        assert!(!spawner.has_timing);
        assert!(spawner.has_change);
        assert_eq!(spawner.duration, Duration::ZERO);
        assert_eq!(profile.change_only_count, 1);
        assert_eq!(profile.matched_count, 1, "only the reader was on both sides");
    }

    /// A timed system absent from the change profile is timing-only.
    #[test]
    fn timing_without_change_is_timing_only() {
        let (schedule, changes) = capture();
        let flame = FlameGraph::from_roots(alloc::vec![SpanNode::new(
            String::from("ghost_system"),
            Duration::from_micros(50),
            Vec::new(),
        )]);

        let profile = SystemCostProfile::join(&flame, &changes, &schedule);
        let ghost = profile.entry("ghost_system").unwrap();
        assert!(ghost.has_timing);
        assert!(!ghost.has_change);
        assert_eq!(ghost.total_cells(), 0);
        assert_eq!(profile.timing_only_count, 1);
    }
}
