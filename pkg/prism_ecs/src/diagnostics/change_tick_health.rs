//! Change-detection tick-saturation / wraparound-headroom census
//! (design §10 / §14 / §16.6).
//!
//! Change detection stores a `u32` [`Tick`] per component value (added /
//! changed) and a coarse `chunk_version` per chunk, all compared *relatively*
//! to the world's current change tick via wrapping arithmetic (design §10
//! 双层 tick). A `u32` counter wraps, so the scheme stays correct only while
//! every stored tick is periodically clamped by
//! [`World::check_change_ticks`](crate::world::World::check_change_ticks): once
//! a stored tick ages past [`Tick::MAX_CHANGE_AGE`] relative to the current
//! tick, a wrapped old tick could alias a very recent one and manufacture a
//! false `Changed<T>` positive. Clamping prevents the alias, but a world that
//! lets ticks drift that far without a maintenance pass is at risk between
//! passes — and in a deterministic simulation (design §14) that drift is a
//! correctness, not just a performance, concern.
//!
//! This module reports, per archetype and world-wide, how *old* the oldest
//! stored tick is relative to the current change tick, how much headroom
//! remains before the clamp bound, and how many cells have already aged past
//! it. It is a read-only health gauge: run it to decide whether a
//! `check_change_ticks` pass is due, or as a CI assertion that a long-running
//! fixture never approaches the bound.
//!
//! # What it surfaces
//!
//! * **tick age** — [`max_age`](TickAgeReport::max_age) is the oldest stored
//!   tick's distance (in ticks) behind the current change tick. Fresh writes
//!   sit at age `0`; the figure grows every frame a value is left untouched.
//! * **clamp headroom** — [`headroom`](TickAgeReport::headroom) /
//!   [`saturation_permille`](TickAgeReport::saturation_permille) express how
//!   close the oldest tick is to [`Tick::MAX_CHANGE_AGE`]; `1000` per-mille
//!   means it has reached the bound.
//! * **maintenance due** — [`needs_check`](TickAgeReport::needs_check) trips at
//!   [`Tick::CHECK_TICK_THRESHOLD`], the suggested interval between clamp
//!   passes, well before correctness is threatened.
//! * **saturation** — [`is_saturated`](TickAgeReport::is_saturated) /
//!   [`total_saturated_cells`](TickAgeReport::total_saturated_cells) flag cells
//!   that have already aged past the clamp bound (a pass is overdue).
//!
//! # Scope
//! Only Table-backed storage carries chunk versioning and per-row ticks, so
//! these figures cover the chunked columnar path (design §6); `SparseSet`
//! components are outside this report. The accounting never mutates ticks (it
//! does not clamp), so it is safe to drive from a diagnostics system.

use alloc::vec::Vec;

use crate::archetype::ArchetypeId;
use crate::change::Tick;
use crate::world::World;

/// One archetype's change-detection tick-age accounting (design §10 / §14).
///
/// Every age is measured backwards from the world's change tick at capture
/// time (`this_run - stored`, wrapping), so a freshly written value reports age
/// `0`. All fields are a read-only snapshot and are public so a report can be
/// assembled from synthetic entries in tests.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArchetypeTickAgeEntry {
    /// The archetype's stable id.
    pub id: ArchetypeId,
    /// Component cells (per-row values) examined across every column.
    pub cell_count: usize,
    /// Chunk-version slots examined across every column (coarse layer).
    pub chunk_version_count: usize,
    /// Oldest per-row *added* tick age in the archetype.
    pub max_added_age: u32,
    /// Oldest per-row *changed* tick age in the archetype.
    pub max_changed_age: u32,
    /// Oldest coarse *chunk-version* tick age in the archetype.
    pub max_chunk_version_age: u32,
    /// Cells whose added or changed tick has aged past
    /// [`Tick::MAX_CHANGE_AGE`] (a clamp pass is overdue for them).
    pub saturated_cells: usize,
}

impl ArchetypeTickAgeEntry {
    /// The oldest tick age in the archetype across the fine (added / changed)
    /// and coarse (chunk-version) layers.
    #[inline]
    pub const fn oldest_age(&self) -> u32 {
        let fine = if self.max_added_age > self.max_changed_age {
            self.max_added_age
        } else {
            self.max_changed_age
        };
        if fine > self.max_chunk_version_age {
            fine
        } else {
            self.max_chunk_version_age
        }
    }

    /// Whether the oldest tick in the archetype has aged past the clamp bound
    /// [`Tick::MAX_CHANGE_AGE`] — a `check_change_ticks` pass is overdue.
    #[inline]
    pub const fn is_saturated(&self) -> bool {
        self.oldest_age() > Tick::MAX_CHANGE_AGE
    }

    /// Whether any individual cell in the archetype has aged past the clamp
    /// bound.
    #[inline]
    pub const fn has_saturated_cells(&self) -> bool {
        self.saturated_cells != 0
    }
}

/// Whole-world change-detection tick-age census (design §10 / §14 / §16.6).
///
/// Built by [`from_world`](Self::from_world); the entries are sorted ascending
/// by [`ArchetypeId`] for a deterministic report, and the roll-up accessors
/// aggregate every archetype. All ages are relative to
/// [`this_run`](Self::this_run), the world's change tick at capture time.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct TickAgeReport {
    this_run: Tick,
    entries: Vec<ArchetypeTickAgeEntry>,
}

impl TickAgeReport {
    /// Measure the tick age of every Table-backed archetype in `world`.
    ///
    /// For each archetype this walks every column's chunk versions (coarse
    /// layer) and per-row added / changed ticks (fine layer), recording the
    /// oldest age in each and counting cells already past the clamp bound.
    /// Read-only: it never clamps a tick.
    pub fn from_world(world: &World) -> Self {
        let this_run = world.change_tick();
        let archetypes = world.archetypes();
        let mut entries = Vec::with_capacity(archetypes.len());

        for archetype in archetypes.iter() {
            let table = archetype.table();
            let component_ids = archetype.components().ids();

            let mut cell_count = 0;
            let mut chunk_version_count = 0;
            let mut max_added_age = 0u32;
            let mut max_changed_age = 0u32;
            let mut max_chunk_version_age = 0u32;
            let mut saturated_cells = 0;

            for &id in component_ids {
                if let Some(column) = table.column(id) {
                    let chunks = column.chunk_count();
                    for chunk in 0..chunks {
                        let age = column.chunk_version(chunk).age_since(this_run);
                        if age > max_chunk_version_age {
                            max_chunk_version_age = age;
                        }
                    }
                    chunk_version_count += chunks;

                    let rows = column.len();
                    for row in 0..rows {
                        let added_age = column.added_tick(row).age_since(this_run);
                        let changed_age = column.changed_tick(row).age_since(this_run);
                        if added_age > max_added_age {
                            max_added_age = added_age;
                        }
                        if changed_age > max_changed_age {
                            max_changed_age = changed_age;
                        }
                        let cell_age = if added_age > changed_age {
                            added_age
                        } else {
                            changed_age
                        };
                        if cell_age > Tick::MAX_CHANGE_AGE {
                            saturated_cells += 1;
                        }
                    }
                    cell_count += rows;
                }
            }

            entries.push(ArchetypeTickAgeEntry {
                id: archetype.id(),
                cell_count,
                chunk_version_count,
                max_added_age,
                max_changed_age,
                max_chunk_version_age,
                saturated_cells,
            });
        }

        Self::from_entries(this_run, entries)
    }

    /// Assemble a report from an explicit change tick and entry set, sorting
    /// the entries ascending by [`ArchetypeId`].
    ///
    /// [`from_world`](Self::from_world) builds the real entries; this
    /// lower-level constructor keeps the aggregate math independent of a live
    /// world.
    pub fn from_entries(this_run: Tick, mut entries: Vec<ArchetypeTickAgeEntry>) -> Self {
        entries.sort_unstable_by_key(|e| e.id);
        Self { this_run, entries }
    }

    /// The world's change tick at capture time (every age is relative to this).
    #[inline]
    pub const fn this_run(&self) -> Tick {
        self.this_run
    }

    /// The per-archetype entries, ascending by archetype id.
    #[inline]
    pub fn entries(&self) -> &[ArchetypeTickAgeEntry] {
        &self.entries
    }

    /// Number of archetypes in the census.
    #[inline]
    pub fn archetype_count(&self) -> usize {
        self.entries.len()
    }

    /// Whether the census carries no archetypes at all.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total component cells examined across every archetype.
    #[inline]
    pub fn total_cells(&self) -> usize {
        self.entries.iter().map(|e| e.cell_count).sum()
    }

    /// Total chunk-version slots examined across every archetype.
    #[inline]
    pub fn total_chunk_versions(&self) -> usize {
        self.entries.iter().map(|e| e.chunk_version_count).sum()
    }

    /// Total cells that have aged past the clamp bound across every archetype.
    #[inline]
    pub fn total_saturated_cells(&self) -> usize {
        self.entries.iter().map(|e| e.saturated_cells).sum()
    }

    /// The oldest tick age anywhere in the world (`0` for an empty census or a
    /// world whose every tick is current).
    pub fn max_age(&self) -> u32 {
        self.entries.iter().map(|e| e.oldest_age()).max().unwrap_or(0)
    }

    /// The archetype carrying the oldest tick (ties, including an all-current
    /// world, resolve to the lowest id), or `None` when the census is empty.
    pub fn oldest_archetype(&self) -> Option<ArchetypeId> {
        // Entries are ascending by id, so a strictly-greater test keeps the
        // lowest id on a tie.
        self.entries
            .iter()
            .reduce(|best, e| {
                if e.oldest_age() > best.oldest_age() {
                    e
                } else {
                    best
                }
            })
            .map(|e| e.id)
    }

    /// Ticks of headroom before the oldest tick reaches the clamp bound
    /// [`Tick::MAX_CHANGE_AGE`]; `0` once the bound is reached or passed.
    #[inline]
    pub fn headroom(&self) -> u32 {
        Tick::MAX_CHANGE_AGE.saturating_sub(self.max_age())
    }

    /// Oldest tick age as a fraction of the clamp bound, in per-mille
    /// (`1000` = the bound has been reached). Saturates at `1000` once past it.
    pub fn saturation_permille(&self) -> u64 {
        let age = self.max_age() as u64;
        let bound = Tick::MAX_CHANGE_AGE as u64;
        let permille = age * 1000 / bound;
        if permille > 1000 { 1000 } else { permille }
    }

    /// Whether the oldest tick has aged past the clamp bound — a
    /// [`check_change_ticks`](crate::world::World::check_change_ticks) pass is
    /// overdue and change detection could otherwise alias a wrapped tick.
    #[inline]
    pub fn is_saturated(&self) -> bool {
        self.max_age() > Tick::MAX_CHANGE_AGE
    }

    /// Whether the oldest tick has reached the suggested maintenance interval
    /// [`Tick::CHECK_TICK_THRESHOLD`] — a clamp pass is due, with correctness
    /// headroom still to spare.
    #[inline]
    pub fn needs_check(&self) -> bool {
        self.max_age() >= Tick::CHECK_TICK_THRESHOLD
    }

    /// The entry for `archetype`, or `None` if it is not in the census.
    #[inline]
    pub fn entry(&self, archetype: ArchetypeId) -> Option<ArchetypeTickAgeEntry> {
        self.entries
            .binary_search_by(|e| e.id.cmp(&archetype))
            .ok()
            .map(|i| self.entries[i])
    }

    /// Whether `archetype` is in the census.
    #[inline]
    pub fn contains(&self, archetype: ArchetypeId) -> bool {
        self.entries
            .binary_search_by(|e| e.id.cmp(&archetype))
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;

    #[derive(Debug, PartialEq)]
    struct Position(f32, f32);
    impl Component for Position {}

    fn entry(id: u32, added: u32, changed: u32, chunk: u32, saturated: usize) -> ArchetypeTickAgeEntry {
        ArchetypeTickAgeEntry {
            id: ArchetypeId::new(id),
            cell_count: 1,
            chunk_version_count: 1,
            max_added_age: added,
            max_changed_age: changed,
            max_chunk_version_age: chunk,
            saturated_cells: saturated,
        }
    }

    #[test]
    fn fresh_world_has_zero_age_and_full_headroom() {
        let world = World::new();
        let report = TickAgeReport::from_world(&world);
        assert_eq!(report.max_age(), 0);
        assert_eq!(report.headroom(), Tick::MAX_CHANGE_AGE);
        assert_eq!(report.saturation_permille(), 0);
        assert!(!report.is_saturated());
        assert!(!report.needs_check());
        assert_eq!(report.total_cells(), 0);
        assert_eq!(report.total_saturated_cells(), 0);
    }

    #[test]
    fn recent_writes_age_by_the_tick_delta() {
        let mut world = World::new();
        for i in 0..5u32 {
            world.spawn(Position(i as f32, 0.0));
        }
        // Advance the change tick three ticks past the spawn stamp.
        world.increment_change_tick();
        world.increment_change_tick();
        world.increment_change_tick();
        let report = TickAgeReport::from_world(&world);
        // Every stored tick is now exactly three ticks behind the current one.
        assert_eq!(report.max_age(), 3);
        assert_eq!(report.headroom(), Tick::MAX_CHANGE_AGE - 3);
        assert_eq!(report.saturation_permille(), 0);
        assert!(!report.is_saturated());
        assert!(!report.needs_check());
        assert_eq!(report.total_saturated_cells(), 0);
        assert!(report.total_cells() >= 5);
        // The populated archetype, not the empty one, carries the oldest tick.
        let oldest = report.oldest_archetype().unwrap();
        let e = report.entry(oldest).unwrap();
        assert_eq!(e.oldest_age(), 3);
    }

    #[test]
    fn check_change_ticks_pass_resets_the_age() {
        // Even if we cannot cheaply drive the world tick to the bound, a clamp
        // pass must never *raise* an age: after it, recent writes stay recent.
        let mut world = World::new();
        world.spawn(Position(0.0, 0.0));
        world.increment_change_tick();
        world.check_change_ticks();
        let report = TickAgeReport::from_world(&world);
        assert!(report.max_age() <= 1);
        assert!(!report.needs_check());
    }

    #[test]
    fn needs_check_trips_at_the_threshold_without_saturating() {
        // CHECK_TICK_THRESHOLD < MAX_CHANGE_AGE, so a tick aged exactly to the
        // maintenance interval is due for a pass but not yet a correctness risk.
        let report = TickAgeReport::from_entries(
            Tick::new(0),
            alloc::vec![entry(0, 0, Tick::CHECK_TICK_THRESHOLD, 0, 0)],
        );
        assert_eq!(report.max_age(), Tick::CHECK_TICK_THRESHOLD);
        assert!(report.needs_check());
        assert!(!report.is_saturated());
        assert_eq!(report.total_saturated_cells(), 0);
    }

    #[test]
    fn half_bound_reports_five_hundred_permille() {
        // MAX_CHANGE_AGE is even, so half of it lands on an exact 500 per-mille.
        let half = Tick::MAX_CHANGE_AGE / 2;
        let report =
            TickAgeReport::from_entries(Tick::new(0), alloc::vec![entry(0, half, 0, 0, 0)]);
        assert_eq!(report.max_age(), half);
        assert_eq!(report.saturation_permille(), 500);
        assert!(report.needs_check());
        assert!(!report.is_saturated());
    }

    #[test]
    fn past_the_bound_is_saturated_and_clamped_to_full_permille() {
        let report = TickAgeReport::from_entries(
            Tick::new(0),
            alloc::vec![entry(0, 0, 0, Tick::MAX_CHANGE_AGE + 1, 1)],
        );
        assert!(report.is_saturated());
        assert_eq!(report.headroom(), 0);
        assert_eq!(report.saturation_permille(), 1000);
        assert_eq!(report.total_saturated_cells(), 1);
        assert!(report.entries()[0].is_saturated());
        assert!(report.entries()[0].has_saturated_cells());
    }

    #[test]
    fn entries_sort_and_oldest_breaks_ties_to_lowest_id() {
        // Supplied out of id order; two archetypes share the oldest age.
        let report = TickAgeReport::from_entries(
            Tick::new(0),
            alloc::vec![entry(7, 0, 100, 0, 0), entry(2, 100, 0, 0, 0), entry(5, 0, 0, 10, 0)],
        );
        let ids: Vec<_> = report.entries().iter().map(|e| e.id.index()).collect();
        assert_eq!(ids, alloc::vec![2, 5, 7]);
        assert_eq!(report.max_age(), 100);
        // Ids 2 and 7 both have age 100; the lowest id wins.
        assert_eq!(report.oldest_archetype(), Some(ArchetypeId::new(2)));
        assert!(report.contains(ArchetypeId::new(5)));
        assert!(!report.contains(ArchetypeId::new(3)));
        assert_eq!(report.entry(ArchetypeId::new(5)).unwrap().oldest_age(), 10);
    }

    #[test]
    fn empty_census_rolls_up_to_zero() {
        let report = TickAgeReport::from_entries(Tick::new(0), alloc::vec![]);
        assert!(report.is_empty());
        assert_eq!(report.archetype_count(), 0);
        assert_eq!(report.max_age(), 0);
        assert_eq!(report.oldest_archetype(), None);
        assert_eq!(report.headroom(), Tick::MAX_CHANGE_AGE);
        assert!(!report.is_saturated());
    }
}
