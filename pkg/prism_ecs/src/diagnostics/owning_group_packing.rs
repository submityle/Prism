//! Owning-group packing / population diagnostic (design §6 存储模型 四态
//! "OwningGroup", §17 "owning group 完美打包", §16.6).
//!
//! An [`OwningGroup`](crate::storage::OwningGroup) is the fourth storage state
//! (design §6): it claims a fixed component set and keeps every entity that
//! owns **all** of those components packed, hole-free, into the contiguous
//! prefix `[0, len())` of its dense storage so a super-hot query iterates it as
//! a branch-free linear scan (design §17 "超热查询完美打包，迭代线性零分支").
//!
//! An owning group is the most expensive storage state to *declare*: it imposes
//! a packing order on its owned components (forbidding any other group from
//! owning them — the single-owner invariant, design §20) and runs O(1)
//! pack/unpack bookkeeping on every structural change touching those components.
//! That cost only pays off when a group actually has members to iterate. This
//! report is a per-group census of each declared group's owned set and packed
//! population so a performance tool or CI can spot a group that was declared
//! for a hot query but is **cold or empty** — paying the maintenance cost
//! without earning the fast iteration it exists for.
//!
//! # What is measured
//! For each declared group, read from the group's own counters (no entity
//! scan):
//!
//! * `owned` / `owned_count` — the component set the group claims and its width;
//! * `packed_members` = [`OwningGroup::len`](crate::storage::OwningGroup::len) —
//!   entities owning *every* owned component, i.e. the branch-free iteration
//!   length and the whole reason the group exists;
//! * `tracked_entities` =
//!   [`OwningGroup::tracked_len`](crate::storage::OwningGroup::tracked_len) and
//!   the derived `unpacked = tracked_entities - packed_members`.
//!
//! # On the tracked-but-unpacked region
//! The storage distinguishes a packed prefix `[0, len())` from a tracked tail
//! `[len(), tracked_len())` and exposes staged
//! [`track`](crate::storage::OwningGroup::track) /
//! [`pack`](crate::storage::OwningGroup::pack) entry points. The standard
//! structural driver
//! ([`World::update_owning_groups`](crate::world::World)), however, only ever
//! inserts an entity once it owns the *whole* set (packing it immediately) and
//! removes it otherwise, so through the public `World` API an entity is never
//! parked in the unpacked tail: `tracked_entities == packed_members` and
//! `unpacked == 0` for every group. This report still reads and reports the two
//! counters faithfully — they diverge only if a caller drives the storage's
//! staged `track`/`pack` directly — but the primary, reachable signal here is
//! the **packed population census** and declared-but-empty detection, not tail
//! overhead.
//!
//! # Scope and determinism
//! Owning-group data lives on the [`World`] (not the component registry), so
//! this report reads [`World::owning_group`](crate::world::World::owning_group)
//! directly rather than through a `Components` registry. It is a pure read of
//! each group's counters: `O(groups)` with no entity traversal, fully
//! deterministic. [`entries`](OwningGroupPackingReport::entries) lists groups in
//! [`OwningGroupId`] (declaration) order; report totals are order-independent.
//! The figures are capability facts, not errors — a group can legitimately be
//! empty for a scene that has not spawned its members yet — so they are
//! surfaced as counts and per-entry predicates for a caller to weigh, never
//! enforced here.

use alloc::vec::Vec;

use crate::storage::OwningGroupId;
use crate::world::World;

#[cfg(doc)]
use crate::component::ComponentId;

/// Packing / population state of a single owning group (design §6 / §17).
///
/// All counts are taken from the group's own packed / tracked counters at
/// capture time; see the [module docs](self) for the dense-region layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwningGroupPackingEntry {
    /// The group this entry describes.
    pub group: OwningGroupId,
    /// The component types the group owns, in declaration order
    /// ([`ComponentId`]).
    pub owned: Vec<crate::component::ComponentId>,
    /// Number of component types owned (`owned.len()`, cached for ranking).
    pub owned_count: usize,
    /// Perfectly-packed members: entities owning *every* component, i.e. the
    /// branch-free iteration length
    /// ([`OwningGroup::len`](crate::storage::OwningGroup::len)).
    pub packed_members: usize,
    /// Entities tracked in the owned storage, packed or not
    /// ([`OwningGroup::tracked_len`](crate::storage::OwningGroup::tracked_len)).
    /// Equals `packed_members` under the standard structural driver; see the
    /// [module docs](self).
    pub tracked_entities: usize,
    /// Tracked-but-not-member entities (`tracked_entities - packed_members`).
    /// `0` under the standard structural driver (see the [module docs](self));
    /// nonzero only if staged `track`/`pack` is driven directly.
    pub unpacked: usize,
}

impl OwningGroupPackingEntry {
    /// Whether the group has no packed members — it is declared but iterating
    /// it yields nothing. A group that stays empty is pure declaration /
    /// maintenance cost with no iteration payoff.
    #[inline]
    pub fn is_empty_group(&self) -> bool {
        self.packed_members == 0
    }

    /// Whether the group has members and its dense storage is one unbroken
    /// member run with no tracked tail. This is the normal healthy state under
    /// the standard structural driver for any non-empty group.
    #[inline]
    pub fn is_perfectly_packed(&self) -> bool {
        self.packed_members > 0 && self.unpacked == 0
    }

    /// Whether the group tracks entities that are not members (`unpacked > 0`).
    /// Always `false` under the standard structural driver; see the [module
    /// docs](self).
    #[inline]
    pub fn has_tracking_overhead(&self) -> bool {
        self.unpacked > 0
    }

    /// Packing density in per-mille (0..=1000): `packed_members * 1000 /
    /// tracked_entities`. Integer-only to stay `no_std`/float-free. Returns `0`
    /// when nothing is tracked (an empty group has no density to speak of).
    /// `1000` means every tracked entity is a packed member — the normal value
    /// for any non-empty group under the standard structural driver.
    #[inline]
    pub fn density_permille(&self) -> u32 {
        if self.tracked_entities == 0 {
            return 0;
        }
        ((self.packed_members as u64 * 1000) / self.tracked_entities as u64) as u32
    }
}

/// Per-group packing / population census across every owning group declared on
/// a world (design §6 / §17 / §16.6).
///
/// Built by [`capture`](Self::capture) / [`from_world`](Self::from_world).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OwningGroupPackingReport {
    /// Per-group entries in [`OwningGroupId`] (declaration) order.
    pub entries: Vec<OwningGroupPackingEntry>,
    /// Number of owning groups declared on the world.
    pub group_count: usize,
    /// Sum of `packed_members` across all groups.
    pub total_packed_members: usize,
    /// Sum of `tracked_entities` across all groups.
    pub total_tracked_entities: usize,
    /// Sum of `unpacked` across all groups (`0` under the standard driver).
    pub total_unpacked: usize,
    /// Number of groups with no packed members ([`is_empty_group`]): declared
    /// but cold.
    ///
    /// [`is_empty_group`]: OwningGroupPackingEntry::is_empty_group
    pub empty_group_count: usize,
    /// Number of non-empty groups whose storage is one unbroken member run
    /// ([`is_perfectly_packed`]).
    ///
    /// [`is_perfectly_packed`]: OwningGroupPackingEntry::is_perfectly_packed
    pub perfectly_packed_count: usize,
}

impl OwningGroupPackingReport {
    /// Capture the owning-group packing / population census of `world`.
    #[inline]
    pub fn capture(world: &World) -> Self {
        Self::from_world(world)
    }

    /// Build the report by reading every owning group declared on `world`.
    ///
    /// Owning-group state lives on the [`World`], so — unlike the registry-based
    /// diagnostics — this does not take a `Components`; it walks
    /// `0..world.owning_group_count()` and reads each group's counters.
    pub fn from_world(world: &World) -> Self {
        let count = world.owning_group_count();
        let mut report = Self {
            group_count: count,
            ..Self::default()
        };

        for i in 0..count {
            let id = OwningGroupId::new(i as u32);
            let Some(group) = world.owning_group(id) else {
                continue;
            };

            let packed_members = group.len();
            let tracked_entities = group.tracked_len();
            // `tracked_len >= len` is a storage invariant (the packed prefix is
            // a subset of the dense storage); saturating keeps the diagnostic
            // honest rather than panicking if that ever regressed.
            let unpacked = tracked_entities.saturating_sub(packed_members);
            let owned: Vec<crate::component::ComponentId> = group.owned().to_vec();
            let entry = OwningGroupPackingEntry {
                group: id,
                owned_count: owned.len(),
                owned,
                packed_members,
                tracked_entities,
                unpacked,
            };

            report.total_packed_members += packed_members;
            report.total_tracked_entities += tracked_entities;
            report.total_unpacked += unpacked;
            report.empty_group_count += entry.is_empty_group() as usize;
            report.perfectly_packed_count += entry.is_perfectly_packed() as usize;

            report.entries.push(entry);
        }

        report
    }

    /// Whether the world declares no owning groups.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The packing entry for `group`, if declared.
    pub fn entry(&self, group: OwningGroupId) -> Option<&OwningGroupPackingEntry> {
        self.entries.iter().find(|e| e.group == group)
    }

    /// The declared-but-cold groups (zero packed members), in declaration
    /// order: owning groups paying maintenance cost with nothing to iterate.
    pub fn empty_groups(&self) -> Vec<&OwningGroupPackingEntry> {
        self.entries.iter().filter(|e| e.is_empty_group()).collect()
    }

    /// The subset of groups carrying a tracked-but-unpacked tail, in
    /// declaration order. Empty under the standard structural driver; see the
    /// [module docs](self).
    pub fn groups_with_overhead(&self) -> Vec<&OwningGroupPackingEntry> {
        self.entries
            .iter()
            .filter(|e| e.has_tracking_overhead())
            .collect()
    }

    /// The least-populated group: fewest `packed_members` first, ties broken by
    /// lowest [`OwningGroupId`] for a stable, deterministic pick. `None` on a
    /// world with no groups. Surfaces the coldest declared hot-query candidate.
    pub fn least_populated(&self) -> Option<&OwningGroupPackingEntry> {
        self.entries.iter().min_by(|a, b| {
            a.packed_members
                .cmp(&b.packed_members)
                .then_with(|| a.group.index().cmp(&b.group.index()))
        })
    }

    /// Whether every declared group has members and is a hole-free member run.
    /// Vacuously `true` on a world with no groups.
    #[inline]
    pub fn all_perfectly_packed(&self) -> bool {
        self.perfectly_packed_count == self.group_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;

    #[derive(Debug, PartialEq)]
    struct A(u32);
    #[derive(Debug, PartialEq)]
    struct B(u32);
    #[derive(Debug, PartialEq)]
    struct C(u32);

    impl Component for A {}
    impl Component for B {}
    impl Component for C {}

    #[test]
    fn empty_world_has_no_groups() {
        let world = World::new();
        let report = OwningGroupPackingReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.group_count, 0);
        // No groups ⇒ vacuously all-packed, and nothing to rank.
        assert!(report.all_perfectly_packed());
        assert!(report.least_populated().is_none());
    }

    #[test]
    fn full_members_pack_and_partials_are_not_tracked() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();
        let group = world.register_owning_group(&[a, b]).unwrap();

        // Three full members pack. Partials owning only one owned component are
        // *not* tracked at all: the structural driver only inserts an entity
        // once it owns the whole set (module docs), so no unpacked tail forms.
        world.spawn((A(1), B(2)));
        world.spawn((A(3), B(4)));
        world.spawn((A(5), B(6)));
        world.spawn(A(7));
        world.spawn(A(8));

        let report = OwningGroupPackingReport::capture(&world);
        let entry = report.entry(group).unwrap();

        assert_eq!(entry.owned, alloc::vec![a, b]);
        assert_eq!(entry.owned_count, 2);
        assert_eq!(entry.packed_members, 3);
        // Partials are never parked in the tail: tracked == packed.
        assert_eq!(entry.tracked_entities, 3);
        assert_eq!(entry.unpacked, 0);
        assert!(!entry.has_tracking_overhead());
        assert!(entry.is_perfectly_packed());
        assert!(!entry.is_empty_group());
        assert_eq!(entry.density_permille(), 1000);
    }

    #[test]
    fn empty_group_is_declared_but_cold() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();
        let group = world.register_owning_group(&[a, b]).unwrap();
        // No matching entity ever spawned.

        let report = OwningGroupPackingReport::capture(&world);
        let entry = report.entry(group).unwrap();

        assert!(entry.is_empty_group());
        assert!(!entry.is_perfectly_packed());
        assert_eq!(entry.packed_members, 0);
        assert_eq!(entry.density_permille(), 0);
        assert_eq!(report.empty_group_count, 1);
        assert!(!report.all_perfectly_packed());
        assert!(report.empty_groups().iter().any(|e| e.group == group));
    }

    #[test]
    fn all_members_full_is_perfectly_packed() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();
        let group = world.register_owning_group(&[a, b]).unwrap();

        world.spawn((A(1), B(2)));
        world.spawn((A(3), B(4)));

        let report = OwningGroupPackingReport::capture(&world);
        let entry = report.entry(group).unwrap();

        assert!(entry.is_perfectly_packed());
        assert!(!entry.has_tracking_overhead());
        assert_eq!(entry.packed_members, 2);
        assert_eq!(entry.tracked_entities, 2);
        assert_eq!(entry.unpacked, 0);
        assert_eq!(entry.density_permille(), 1000);
        assert!(report.all_perfectly_packed());
        assert_eq!(report.perfectly_packed_count, 1);
    }

    #[test]
    fn insert_completes_and_remove_empties_membership() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();
        let group = world.register_owning_group(&[a, b]).unwrap();

        // Owning only A: not a member and not tracked.
        let e = world.spawn(A(1));
        let r0 = OwningGroupPackingReport::capture(&world);
        let e0 = r0.entry(group).unwrap();
        assert_eq!(e0.packed_members, 0);
        assert_eq!(e0.tracked_entities, 0);
        assert!(e0.is_empty_group());

        // Gaining B completes the group: fully packed.
        world.insert(e, B(2));
        let r1 = OwningGroupPackingReport::capture(&world);
        let e1 = r1.entry(group).unwrap();
        assert_eq!(e1.packed_members, 1);
        assert_eq!(e1.unpacked, 0);
        assert!(e1.is_perfectly_packed());

        // Losing B drops it from the group entirely.
        world.remove::<B>(e);
        let r2 = OwningGroupPackingReport::capture(&world);
        let e2 = r2.entry(group).unwrap();
        assert_eq!(e2.packed_members, 0);
        assert!(e2.is_empty_group());
    }

    #[test]
    fn totals_sum_over_groups_and_least_populated_ranks() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();
        let c = world.register_component::<C>();

        // Group 0 (a, b): two full members. Group 1 (c): one member. Owned
        // sets must be disjoint — the single-owner invariant forbids two
        // groups claiming the same component (design §20).
        let g0 = world.register_owning_group(&[a, b]).unwrap();
        let g1 = world.register_owning_group(&[c]).unwrap();

        world.spawn((A(1), B(2)));
        world.spawn((A(3), B(4)));
        world.spawn(C(5));

        let report = OwningGroupPackingReport::capture(&world);
        assert_eq!(report.group_count, 2);

        // Totals are the per-entry sums.
        let packed: usize = report.entries.iter().map(|e| e.packed_members).sum();
        let tracked: usize = report.entries.iter().map(|e| e.tracked_entities).sum();
        let unpacked: usize = report.entries.iter().map(|e| e.unpacked).sum();
        assert_eq!(packed, report.total_packed_members);
        assert_eq!(packed, 3);
        assert_eq!(tracked, report.total_tracked_entities);
        assert_eq!(unpacked, report.total_unpacked);
        assert_eq!(unpacked, 0);

        // g1 has 1 member, g0 has 2, so the least-populated group is g1.
        let coldest = report.least_populated().unwrap();
        assert_eq!(coldest.group, g1);
        assert_eq!(coldest.packed_members, 1);
        assert_eq!(report.entry(g0).unwrap().packed_members, 2);

        // Both groups are non-empty and hole-free under the standard driver.
        assert!(report.all_perfectly_packed());
        assert!(report.groups_with_overhead().is_empty());
        assert!(report.empty_groups().is_empty());
    }
}
