//! Entity-allocator occupancy and location-table integrity census (design
//! §5.1 / §5.3 / §20 / §16.6).
//!
//! The [`Entities`] allocator (design §5.1) is the spine of the world: a dense
//! `meta` table of per-slot generations plus a free-list of recycled slots.
//! Every live handle resolves through it to an [`EntityLocation`] — an
//! `(ArchetypeId, row)` coordinate into the archetype graph (design §5.3). Two
//! independent structures therefore encode the *same* truth: the allocator says
//! which slots are live and where they sit, and the archetype tables physically
//! hold those entities row by row. When those two disagree the world is
//! corrupt in exactly the way generational-index ECSs are supposed to make
//! impossible (design §20 invariants: stable ids, generation strictly
//! increasing, sub-world migration preserving generation).
//!
//! This census reads both sides once and reconciles them:
//!
//! * **Allocator occupancy** — live handles against the high-water slot
//!   [`capacity`](Entities::capacity) and the recycled
//!   [`free`](EntityResidencyReport::free_slots) slots waiting to be re-handed.
//!   A world that has churned heavily shows a capacity far above its live count
//!   (a deep free-list), which sizes every dense side table (design §5.1).
//! * **Recycle pressure** — how many live handles sit on a *reused* slot
//!   (`generation > 1`) and the peak generation reached. Generation is a
//!   wrapping `NonZeroU32` (design §5.1): the peak is the distance travelled
//!   toward the ~4-billion-cycle wrap after which stale handles could alias, so
//!   it is the long-run safety signal for a streaming world (design §13.1).
//! * **Location integrity** — walking every archetype table and resolving each
//!   stored entity back through the allocator: it must be
//!   [`live`](Entities::contains), its recorded
//!   [`location`](Entities::location) must point back to the very archetype and
//!   row it was found in, and no index may appear in two tables at once. Any
//!   mismatch is a structural-move or free-list bug the integrity counters
//!   surface directly.
//!
//! Unlike [`inspector`](super::inspector), which measures chunk-row geometry
//! (capacity and occupancy of the SoA chunks *inside* each archetype), this
//! report measures the entity-*handle* allocator and the correctness of the
//! handle-to-row mapping — an orthogonal axis. It is also distinct from the
//! partition-gated [`dormancy_census`](super::dormancy_census), which buckets
//! the id-space of the *dormant subset*; this report reconciles the *whole*
//! live population against its physical placement.
//!
//! # Honest scope
//! Everything is read-only and allocation-light (one small index set for
//! duplicate detection) and deterministic (design §14): the figures are pure
//! counts and the walk order does not affect them. A handle that is live but
//! not yet placed in any table — reserved-and-flushed but not spawned, or a
//! restore in progress before its tables are rebuilt (design §14) — is counted
//! in [`live_entities`](EntityResidencyReport::live_entities) but not in
//! [`placed_entities`](EntityResidencyReport::placed_entities); the difference
//! is reported as [`unplaced_entities`](EntityResidencyReport::unplaced_entities)
//! rather than mistaken for corruption.

use crate::archetype::Archetypes;
use crate::collections::HashMap;
use crate::entity::Entities;

/// Integer permille (`parts per thousand`) of `num / den`, returning `0` when
/// `den` is zero.
#[inline]
fn permille(num: u64, den: u64) -> u64 {
    (num * 1000).checked_div(den).unwrap_or(0)
}

/// Read-only reconciliation of the [`Entities`] allocator against the physical
/// entity placement in the archetype graph (design §5.1 / §5.3 / §20 / §16.6).
///
/// Build it with [`capture`](Self::capture) from a world, or
/// [`from_parts`](Self::from_parts) from an allocator and archetype graph
/// directly. All fields are private; read them through the accessors.
#[derive(Clone, Debug)]
pub struct EntityResidencyReport {
    live_entities: usize,
    slot_capacity: usize,
    placed_entities: usize,
    recycled_slots: usize,
    max_generation: u32,
    generation_sum: u64,
    misplaced_entities: usize,
    dangling_placements: usize,
    duplicate_placements: usize,
}

impl EntityResidencyReport {
    /// Build the census from a world via
    /// [`World::entities`](crate::world::World::entities) and
    /// [`World::archetypes`](crate::world::World::archetypes).
    #[inline]
    pub fn capture(world: &crate::world::World) -> Self {
        Self::from_parts(world.archetypes(), world.entities())
    }

    /// Build the census directly from the archetype graph and the entity
    /// allocator, reconciling the two.
    ///
    /// Every entity physically stored in an archetype table is resolved back
    /// through `entities`: a mismatch between where it is stored and the
    /// allocator's recorded [`location`](Entities::location) increments
    /// [`misplaced_entities`](Self::misplaced_entities), a stored handle the
    /// allocator does not recognise as live increments
    /// [`dangling_placements`](Self::dangling_placements), and an index that
    /// appears in more than one table increments
    /// [`duplicate_placements`](Self::duplicate_placements).
    pub fn from_parts(archetypes: &Archetypes, entities: &Entities) -> Self {
        let live_entities = entities.len() as usize;
        let slot_capacity = entities.capacity() as usize;

        let mut placed_entities = 0usize;
        let mut recycled_slots = 0usize;
        let mut max_generation = 0u32;
        let mut generation_sum = 0u64;
        let mut misplaced_entities = 0usize;
        let mut dangling_placements = 0usize;
        let mut duplicate_placements = 0usize;
        let mut seen: HashMap<u32, ()> = HashMap::default();

        for archetype in archetypes.iter() {
            let archetype_id = archetype.id();
            for (row, &entity) in archetype.table().entities().iter().enumerate() {
                placed_entities += 1;

                let generation = entity.generation();
                if generation > 1 {
                    recycled_slots += 1;
                }
                if generation > max_generation {
                    max_generation = generation;
                }
                generation_sum += generation as u64;

                if seen.insert(entity.index(), ()).is_some() {
                    duplicate_placements += 1;
                }

                if entities.contains(entity) {
                    match entities.location(entity) {
                        Some(location)
                            if location.archetype_id == archetype_id
                                && location.row as usize == row => {}
                        _ => misplaced_entities += 1,
                    }
                } else {
                    dangling_placements += 1;
                }
            }
        }

        Self {
            live_entities,
            slot_capacity,
            placed_entities,
            recycled_slots,
            max_generation,
            generation_sum,
            misplaced_entities,
            dangling_placements,
            duplicate_placements,
        }
    }

    /// Number of currently-live entity handles the allocator tracks
    /// ([`Entities::len`]).
    #[inline]
    pub fn live_entities(&self) -> usize {
        self.live_entities
    }

    /// High-water slot count of the allocator metadata table
    /// ([`Entities::capacity`]) — the width every dense per-slot side table
    /// must span (design §5.1).
    #[inline]
    pub fn slot_capacity(&self) -> usize {
        self.slot_capacity
    }

    /// Number of entities physically found in an archetype table (summed across
    /// the whole graph).
    #[inline]
    pub fn placed_entities(&self) -> usize {
        self.placed_entities
    }

    /// Recycled slots sitting on the free list: allocated capacity not backing
    /// a live handle (`slot_capacity - live_entities`, saturating). A deep
    /// free-list signals heavy spawn/despawn churn (design §5.1 / §13.1).
    #[inline]
    pub fn free_slots(&self) -> usize {
        self.slot_capacity.saturating_sub(self.live_entities)
    }

    /// Live handles not yet placed in any archetype table
    /// (`live_entities - placed_entities`, saturating): reserved-and-flushed
    /// handles awaiting a spawn, or a restore in progress before its tables are
    /// rebuilt (design §14). `0` for a settled world.
    #[inline]
    pub fn unplaced_entities(&self) -> usize {
        self.live_entities.saturating_sub(self.placed_entities)
    }

    /// Permille (parts per thousand) of slot capacity that is live, `0` when no
    /// slots are allocated. The density of the allocator table.
    #[inline]
    pub fn occupancy_permille(&self) -> u64 {
        permille(self.live_entities as u64, self.slot_capacity as u64)
    }

    /// Permille (parts per thousand) of slot capacity sitting idle on the free
    /// list, `0` when no slots are allocated.
    #[inline]
    pub fn free_permille(&self) -> u64 {
        permille(self.free_slots() as u64, self.slot_capacity as u64)
    }

    /// Number of placed entities whose slot has been reused at least once
    /// (`generation > 1`) — the recycled share of the live population.
    #[inline]
    pub fn recycled_slots(&self) -> usize {
        self.recycled_slots
    }

    /// Permille (parts per thousand) of placed entities that occupy a recycled
    /// slot, `0` when nothing is placed. A direct read on free-list reuse.
    #[inline]
    pub fn recycle_permille(&self) -> u64 {
        permille(self.recycled_slots as u64, self.placed_entities as u64)
    }

    /// Highest generation reached by any live placed handle — the distance
    /// travelled toward the wrapping `NonZeroU32` recycle horizon (design
    /// §5.1). `0` when nothing is placed; `1` for a never-recycled world.
    #[inline]
    pub fn max_generation(&self) -> u32 {
        self.max_generation
    }

    /// Mean generation across placed handles, rounded down (`1` + average
    /// recycle depth); `0` when nothing is placed.
    #[inline]
    pub fn mean_generation(&self) -> u64 {
        self.generation_sum
            .checked_div(self.placed_entities as u64)
            .unwrap_or(0)
    }

    /// Placed entities whose allocator-recorded [`location`](Entities::location)
    /// did not match the archetype and row they were physically stored in — a
    /// structural-move or location-table desync (design §20). `0` in a healthy
    /// world.
    #[inline]
    pub fn misplaced_entities(&self) -> usize {
        self.misplaced_entities
    }

    /// Placed handles the allocator does not recognise as live (stale
    /// generation or freed slot) — a free-list / despawn bug (design §20). `0`
    /// in a healthy world.
    #[inline]
    pub fn dangling_placements(&self) -> usize {
        self.dangling_placements
    }

    /// Entity indices found in more than one archetype table — a duplicate
    /// placement that must never occur (design §20). `0` in a healthy world.
    #[inline]
    pub fn duplicate_placements(&self) -> usize {
        self.duplicate_placements
    }

    /// Whether the live count equals the placed count, i.e. every live handle
    /// is materialised in a table and none is placed twice.
    #[inline]
    pub fn live_matches_placed(&self) -> bool {
        self.live_entities == self.placed_entities
    }

    /// Whether the allocator and the archetype graph fully agree: no misplaced,
    /// dangling, or duplicate placements, and the live and placed counts match.
    /// The one-shot "is the world structurally sound" verdict (design §20).
    #[inline]
    pub fn is_consistent(&self) -> bool {
        self.misplaced_entities == 0
            && self.dangling_placements == 0
            && self.duplicate_placements == 0
            && self.live_matches_placed()
    }

    /// Whether the world holds no live or placed entities.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live_entities == 0 && self.placed_entities == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::world::World;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct Position(i32);
    impl Component for Position {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct Velocity(i32);
    impl Component for Velocity {}

    #[test]
    fn empty_world_is_consistent_and_empty() {
        let world = World::new();
        let report = EntityResidencyReport::capture(&world);
        assert!(report.is_empty());
        assert!(report.is_consistent());
        assert_eq!(report.live_entities(), 0);
        assert_eq!(report.slot_capacity(), 0);
        assert_eq!(report.placed_entities(), 0);
        assert_eq!(report.free_slots(), 0);
        assert_eq!(report.unplaced_entities(), 0);
        assert_eq!(report.occupancy_permille(), 0);
        assert_eq!(report.recycle_permille(), 0);
        assert_eq!(report.max_generation(), 0);
        assert_eq!(report.mean_generation(), 0);
    }

    #[test]
    fn fresh_spawns_are_fully_placed_and_consistent() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn(Position(i));
        }
        let report = EntityResidencyReport::capture(&world);
        assert_eq!(report.live_entities(), 5);
        assert_eq!(report.placed_entities(), 5);
        assert_eq!(report.slot_capacity(), 5);
        assert_eq!(report.free_slots(), 0);
        assert_eq!(report.unplaced_entities(), 0);
        assert_eq!(report.occupancy_permille(), 1000);
        assert_eq!(report.recycled_slots(), 0);
        assert_eq!(report.max_generation(), 1);
        assert_eq!(report.mean_generation(), 1);
        assert!(report.is_consistent());
    }

    #[test]
    fn component_less_entities_are_placed_in_the_empty_archetype() {
        let mut world = World::new();
        world.spawn(());
        world.spawn(());
        let report = EntityResidencyReport::capture(&world);
        assert_eq!(report.live_entities(), 2);
        assert_eq!(report.placed_entities(), 2);
        assert_eq!(report.unplaced_entities(), 0);
        assert!(report.is_consistent());
    }

    #[test]
    fn despawn_leaves_recycled_free_slots() {
        let mut world = World::new();
        let a = world.spawn(Position(0));
        let _b = world.spawn(Position(1));
        let c = world.spawn(Position(2));
        let _d = world.spawn(Position(3));
        assert!(world.despawn(a));
        assert!(world.despawn(c));

        let report = EntityResidencyReport::capture(&world);
        assert_eq!(report.live_entities(), 2);
        assert_eq!(report.placed_entities(), 2);
        assert_eq!(report.slot_capacity(), 4);
        assert_eq!(report.free_slots(), 2);
        assert_eq!(report.free_permille(), 500);
        assert_eq!(report.occupancy_permille(), 500);
        assert!(report.is_consistent());
    }

    #[test]
    fn respawn_recycles_slots_and_bumps_generation() {
        let mut world = World::new();
        let a = world.spawn(Position(0));
        let b = world.spawn(Position(1));
        assert!(world.despawn(a));
        assert!(world.despawn(b));
        // Two fresh spawns must reuse the two freed slots, bumping generation.
        world.spawn(Position(2));
        world.spawn(Position(3));

        let report = EntityResidencyReport::capture(&world);
        assert_eq!(report.live_entities(), 2);
        assert_eq!(report.placed_entities(), 2);
        assert_eq!(report.slot_capacity(), 2);
        assert_eq!(report.recycled_slots(), 2);
        assert_eq!(report.recycle_permille(), 1000);
        assert!(report.max_generation() >= 2);
        assert!(report.mean_generation() >= 2);
        assert!(report.is_consistent());
    }

    #[test]
    fn mixed_archetypes_reconcile_and_stay_consistent() {
        let mut world = World::new();
        world.spawn(Position(0));
        world.spawn((Position(1), Velocity(1)));
        world.spawn((Position(2), Velocity(2)));
        world.spawn(());
        let report = EntityResidencyReport::capture(&world);
        assert_eq!(report.live_entities(), 4);
        assert_eq!(report.placed_entities(), 4);
        assert_eq!(report.misplaced_entities(), 0);
        assert_eq!(report.dangling_placements(), 0);
        assert_eq!(report.duplicate_placements(), 0);
        assert!(report.is_consistent());
    }

    #[test]
    fn capture_matches_from_parts() {
        let mut world = World::new();
        let a = world.spawn(Position(0));
        world.spawn((Position(1), Velocity(1)));
        assert!(world.despawn(a));
        world.spawn(Position(2));

        let via_capture = EntityResidencyReport::capture(&world);
        let via_parts =
            EntityResidencyReport::from_parts(world.archetypes(), world.entities());
        assert_eq!(via_capture.live_entities(), via_parts.live_entities());
        assert_eq!(via_capture.placed_entities(), via_parts.placed_entities());
        assert_eq!(via_capture.slot_capacity(), via_parts.slot_capacity());
        assert_eq!(via_capture.recycled_slots(), via_parts.recycled_slots());
        assert_eq!(via_capture.max_generation(), via_parts.max_generation());
        assert_eq!(via_capture.is_consistent(), via_parts.is_consistent());
    }
}
