//! Frame-level change-volume profile (design §16.6 / §10).
//!
//! [`SteppingInspector::capture_frame`](super::step_inspector::SteppingInspector::capture_frame)
//! yields a per-system [`StepObservation`] trace in execution order, each
//! carrying a [`ChangeReport`](super::change_volume::ChangeReport) scoped to
//! exactly that system's change-tick window (design §23.4 × §16.6). This module
//! folds such a trace into a ranked profile: it answers "which systems dirtied
//! the most storage this frame?" — the data behind a devtools "hot systems"
//! panel and the headline invariant "成本 ∝ 变化量" (design §10) applied at
//! system granularity.
//!
//! Aggregation is by schedule node, so a trace that concatenates several frames
//! (or a system that somehow appears more than once) sums cleanly. Frame
//! boundaries carry no node and are ignored. The resulting [`entries`] are
//! sorted dirtiest-first with a deterministic node-index tie-break, so repeated
//! captures of the same workload rank identically regardless of hash order.
//!
//! [`entries`]: FrameChangeProfile::entries

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::diagnostics::step_inspector::StepObservation;
use crate::schedule::graph::Schedule;

/// Per-system change-volume totals aggregated over a frame trace.
///
/// Deltas are *net* across every observation of the node (a spawn then despawn
/// cancels out). `changed_cells` / `added_cells` / `dirty_chunks` are sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemChangeEntry {
    /// The schedule node index this entry aggregates.
    pub node: usize,
    /// Whether any observation of this node actually ran (vs. being gated off
    /// every time it was stepped).
    pub ran: bool,
    /// Summed changed cells (per-component value writes) across the frame.
    pub changed_cells: usize,
    /// Summed added cells (newly inserted components) across the frame.
    pub added_cells: usize,
    /// Summed dirty chunks across the frame.
    pub dirty_chunks: usize,
    /// Net live-entity delta across the frame (negative for net despawns).
    pub entity_delta: i64,
    /// Net archetype-count delta across the frame.
    pub archetype_delta: isize,
}

impl SystemChangeEntry {
    /// Total dirtied cells: `changed_cells + added_cells`. This is the primary
    /// ranking key for the frame profile.
    #[inline]
    pub fn total_cells(&self) -> usize {
        self.changed_cells + self.added_cells
    }

    /// Whether this system observably touched storage over the frame: it
    /// changed or added cells, or moved the entity / archetype counts. A system
    /// that ran but mutated nothing returns `false` (a no-op), exactly the
    /// signal a devtools panel uses to grey it out.
    #[inline]
    pub fn touched_storage(&self) -> bool {
        self.changed_cells > 0
            || self.added_cells > 0
            || self.entity_delta != 0
            || self.archetype_delta != 0
    }

    /// Resolve this system's name against `schedule` (its `type_name`, design
    /// §16.6 inspector label), or `None` if the node is out of range.
    pub fn name<'s>(&self, schedule: &'s Schedule) -> Option<&'s str> {
        schedule.nodes.get(self.node).map(|n| n.system.name())
    }
}

/// A frame's change volume folded into a per-system ranking (design §16.6).
///
/// Produced by [`FrameChangeProfile::from_trace`]. [`entries`](Self::entries)
/// is sorted by [`total_cells`](SystemChangeEntry::total_cells) descending, with
/// ties broken by ascending node index for determinism.
#[derive(Debug, Clone)]
pub struct FrameChangeProfile {
    /// One entry per schedule node observed in the trace, dirtiest first.
    pub entries: Vec<SystemChangeEntry>,
    /// Total changed cells across every system in the frame.
    pub total_changed_cells: usize,
    /// Total added cells across every system in the frame.
    pub total_added_cells: usize,
    /// Total dirty chunks across every system in the frame.
    pub total_dirty_chunks: usize,
    /// Number of distinct nodes that ran at least once.
    pub ran_count: usize,
    /// Number of distinct nodes that were skipped every time they were
    /// observed (gated off, never ran).
    pub skipped_count: usize,
}

impl FrameChangeProfile {
    /// Fold a [`StepObservation`] trace into a ranked per-system profile.
    ///
    /// Frame-boundary observations are ignored. Observations are aggregated by
    /// schedule node, so a system seen more than once sums its contributions.
    /// Read-only: it borrows the trace and allocates its own report.
    pub fn from_trace(trace: &[StepObservation]) -> Self {
        let mut by_node: HashMap<usize, SystemChangeEntry> = HashMap::new();
        for obs in trace {
            let Some(node) = obs.node() else {
                continue;
            };
            let entry = by_node.entry(node).or_insert(SystemChangeEntry {
                node,
                ran: false,
                changed_cells: 0,
                added_cells: 0,
                dirty_chunks: 0,
                entity_delta: 0,
                archetype_delta: 0,
            });
            entry.ran |= obs.ran();
            entry.changed_cells += obs.change.changed_cells;
            entry.added_cells += obs.change.added_cells;
            entry.dirty_chunks += obs.change.dirty_chunks;
            entry.entity_delta += obs.entity_delta();
            entry.archetype_delta += obs.archetype_delta();
        }

        let mut entries: Vec<SystemChangeEntry> = by_node.into_values().collect();
        // Dirtiest first; ascending node index breaks ties deterministically.
        entries.sort_by(|a, b| {
            b.total_cells()
                .cmp(&a.total_cells())
                .then_with(|| a.node.cmp(&b.node))
        });

        let mut total_changed_cells = 0;
        let mut total_added_cells = 0;
        let mut total_dirty_chunks = 0;
        let mut ran_count = 0;
        let mut skipped_count = 0;
        for entry in &entries {
            total_changed_cells += entry.changed_cells;
            total_added_cells += entry.added_cells;
            total_dirty_chunks += entry.dirty_chunks;
            if entry.ran {
                ran_count += 1;
            } else {
                skipped_count += 1;
            }
        }

        Self {
            entries,
            total_changed_cells,
            total_added_cells,
            total_dirty_chunks,
            ran_count,
            skipped_count,
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

    /// Total dirtied cells across the frame: changed + added.
    #[inline]
    pub fn total_cells(&self) -> usize {
        self.total_changed_cells + self.total_added_cells
    }

    /// The dirtiest system (highest [`total_cells`](SystemChangeEntry::total_cells)),
    /// or `None` for an empty profile. This is the first ranked entry, so ties
    /// resolve to the lowest node index.
    #[inline]
    pub fn hottest(&self) -> Option<&SystemChangeEntry> {
        self.entries.first()
    }

    /// Number of systems that ran but did not observably touch storage (no-ops
    /// a panel can grey out).
    pub fn noop_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.ran && !e.touched_storage())
            .count()
    }

    /// The entry for `node`, if it was observed.
    pub fn entry(&self, node: usize) -> Option<&SystemChangeEntry> {
        self.entries.iter().find(|e| e.node == node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Commands;
    use crate::component::Component;
    use crate::diagnostics::step_inspector::SteppingInspector;
    use crate::resource::Resource;
    use crate::schedule::{resource_exists, IntoSystemConfigs, Schedule};
    use crate::system::{Query, ResMut};
    use crate::world::World;
    use alloc::vec::Vec;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Cd(i32);
    impl Component for Cd {}

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Spawned;
    impl Component for Spawned {}

    #[derive(Debug, Default, PartialEq, Clone)]
    struct Log(Vec<u32>);
    impl Resource for Log {}

    #[derive(Debug, Default)]
    struct Gate;
    impl Resource for Gate {}

    // Reads only: runs but dirties nothing.
    fn reader(q: Query<&Cd>, mut log: ResMut<Log>) {
        log.0.push(q.iter().count() as u32);
    }
    // Mutates every Cd: three changed cells, no new rows.
    fn mutator(mut q: Query<&mut Cd>) {
        for mut c in q.iter_mut() {
            c.0 += 1;
        }
    }
    // Spawns five rows via deferred commands: five added cells, +5 entities.
    fn spawner(mut commands: Commands) {
        for _ in 0..5 {
            commands.spawn(Spawned);
        }
    }
    // Gated off (Gate resource never inserted): never runs.
    fn gated(mut log: ResMut<Log>) {
        log.0.push(999);
    }

    fn fresh_world() -> World {
        let mut world = World::new();
        world.insert_resource(Log::default());
        world.spawn(Cd(0));
        world.spawn(Cd(0));
        world.spawn(Cd(0));
        world
    }

    fn build() -> Schedule {
        let mut schedule = Schedule::new();
        schedule.add_systems((reader, mutator, spawner).chain());
        schedule.add_systems(gated.run_if(resource_exists::<Gate>()));
        schedule
    }

    fn profile_one_frame() -> (Schedule, FrameChangeProfile) {
        let mut world = fresh_world();
        let mut schedule = build();
        let mut inspector = SteppingInspector::new();
        let trace = inspector.capture_frame(&mut schedule, &mut world);
        let profile = FrameChangeProfile::from_trace(&trace);
        (schedule, profile)
    }

    fn entry_named<'a>(
        profile: &'a FrameChangeProfile,
        schedule: &Schedule,
        needle: &str,
    ) -> &'a SystemChangeEntry {
        profile
            .entries
            .iter()
            .find(|e| e.name(schedule).unwrap().contains(needle))
            .unwrap_or_else(|| panic!("missing entry for {needle}"))
    }

    /// An empty or boundary-only trace folds to an empty profile.
    #[test]
    fn empty_trace_is_empty() {
        let profile = FrameChangeProfile::from_trace(&[]);
        assert!(profile.is_empty());
        assert_eq!(profile.system_count(), 0);
        assert_eq!(profile.total_cells(), 0);
        assert_eq!(profile.ran_count, 0);
        assert_eq!(profile.skipped_count, 0);
        assert!(profile.hottest().is_none());
    }

    /// Per-system aggregation attributes change volume to the exact system:
    /// the mutator reports its three changed cells, the spawner its five added
    /// cells, the reader nothing, and the gated system never ran.
    #[test]
    fn per_system_change_is_attributed() {
        let (schedule, profile) = profile_one_frame();

        // reader, mutator, spawner, gated — four distinct nodes.
        assert_eq!(profile.system_count(), 4);

        let reader = entry_named(&profile, &schedule, "reader");
        assert!(reader.ran);
        assert!(!reader.touched_storage(), "read-only system is a no-op");
        assert_eq!(reader.total_cells(), 0);

        let mutator = entry_named(&profile, &schedule, "mutator");
        assert!(mutator.ran);
        assert_eq!(mutator.changed_cells, 3, "all three Cd cells written");
        assert_eq!(mutator.added_cells, 0, "no new rows");
        assert_eq!(mutator.entity_delta, 0);

        let spawner = entry_named(&profile, &schedule, "spawner");
        assert!(spawner.ran);
        assert_eq!(spawner.added_cells, 5, "five Spawned cells added");
        assert_eq!(spawner.entity_delta, 5);

        let gated = entry_named(&profile, &schedule, "gated");
        assert!(!gated.ran, "gate resource absent");
        assert!(!gated.touched_storage());
    }

    /// Entries rank dirtiest-first and the totals sum every system.
    #[test]
    fn entries_rank_dirtiest_first_and_totals_sum() {
        let (schedule, profile) = profile_one_frame();

        // Totals are independent of ordering. A freshly spawned component is
        // both *added* and *changed* in the same tick, so the spawner's five
        // rows show up in both the changed and added totals.
        assert_eq!(
            profile.total_changed_cells, 8,
            "mutator's 3 writes + spawner's 5 fresh cells"
        );
        assert_eq!(profile.total_added_cells, 5, "only the spawner added cells");
        assert_eq!(profile.total_cells(), 13);
        assert_eq!(profile.ran_count, 3, "reader, mutator, spawner ran");
        assert_eq!(profile.skipped_count, 1, "gated never ran");

        // Descending total_cells with ascending-node tie-break.
        for pair in profile.entries.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            assert!(
                a.total_cells() > b.total_cells()
                    || (a.total_cells() == b.total_cells() && a.node < b.node),
                "entries must be sorted dirtiest-first, node-ascending on ties"
            );
        }

        // spawner (5 cells) outranks mutator (3), which outranks the two no-ops.
        let hottest = profile.hottest().unwrap();
        assert!(
            hottest.name(&schedule).unwrap().contains("spawner"),
            "spawner adds the most cells this frame"
        );
    }

    /// A system that ran but touched nothing is a no-op; a gated system is not
    /// (it never ran).
    #[test]
    fn noop_count_tracks_ran_but_inert_systems() {
        let (schedule, profile) = profile_one_frame();
        assert_eq!(
            profile.noop_count(),
            1,
            "only the reader ran without touching storage"
        );
        let reader = entry_named(&profile, &schedule, "reader");
        assert!(reader.ran && !reader.touched_storage());
    }

    /// Concatenating two frames' traces sums each node's contribution rather
    /// than producing duplicate entries.
    #[test]
    fn repeated_nodes_aggregate() {
        let mut world = fresh_world();
        let mut schedule = build();
        let mut inspector = SteppingInspector::new();
        let mut trace = inspector.capture_frame(&mut schedule, &mut world);
        trace.extend(inspector.capture_frame(&mut schedule, &mut world));

        let profile = FrameChangeProfile::from_trace(&trace);
        // Still four distinct nodes despite two frames of observations.
        assert_eq!(profile.system_count(), 4);
        let mutator = entry_named(&profile, &schedule, "mutator");
        assert_eq!(
            mutator.changed_cells, 6,
            "three cells changed in each of two frames"
        );
        let spawner = entry_named(&profile, &schedule, "spawner");
        assert_eq!(spawner.added_cells, 10, "five added in each of two frames");
        assert_eq!(spawner.entity_delta, 10);
    }

    /// `entry(node)` resolves an observed node and rejects an unobserved one.
    #[test]
    fn entry_lookup_by_node() {
        let (schedule, profile) = profile_one_frame();
        let mutator = entry_named(&profile, &schedule, "mutator");
        let node = mutator.node;
        assert_eq!(profile.entry(node).unwrap().node, node);
        assert!(profile.entry(usize::MAX).is_none());
    }
}
