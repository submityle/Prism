//! Weak cross-cell reference table health census (design §13.1 / §22 risk 4 /
//! §16.6).
//!
//! World-partition streaming (design §13.1) despawns an entity when its cell
//! unloads, so any handle another cell kept to it goes stale. The engine's
//! answer (design §22 risk 4) is the generational [`WeakEntity`]: a stored
//! handle resolves against the live [`World`] only while the slot still carries
//! the recorded generation, and fails safely (`None`) once the target is
//! streamed out or its slot recycled. A [`WeakRefs`] table holds such handles
//! verbatim — duplicates allowed, never self-pruning — and the owning scene is
//! expected to [`prune`](WeakRefs::prune) it after each streaming pass.
//!
//! Whether that discipline is actually being followed is invisible from the
//! table alone: `len()` counts handles live *and* dead, and nothing reports how
//! many have gone stale or how badly the list has accreted duplicates. This
//! report resolves every stored handle against the authoritative [`World`] once
//! and makes the table's health legible:
//!
//! * **liveness** — how many stored references (and how many distinct targets)
//!   still resolve versus dangle, as both counts and a permille *prune
//!   pressure* (the fraction a [`prune`](WeakRefs::prune) would drop);
//! * **multiplicity** — distinct targets versus total handles, the duplicate
//!   count, and the maximum fan-in (how many times the most-referenced target
//!   appears), flagging a table that is accreting redundant handles;
//! * a deterministic **per-target breakdown** — each distinct target with its
//!   reference count and liveness, ordered by [`Entity::to_bits`] (design §14).
//!
//! The census is read-only: it resolves handles with
//! [`WeakEntity::is_alive`](WeakEntity::is_alive) and never mutates the table or
//! the world. It runs in `O(n)` over the stored handles plus an `O(t log t)`
//! sort of the `t` distinct targets.

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::entity::Entity;
use crate::partition::streaming::WeakRefs;
use crate::world::World;

/// Integer permille (`parts per thousand`) of `num / den`, returning `0` when
/// `den` is zero.
#[inline]
fn permille(num: u64, den: u64) -> u64 {
    (num * 1000).checked_div(den).unwrap_or(0)
}

/// One distinct target of a [`WeakRefs`] table: an entity handle referenced one
/// or more times, with its reference count and current liveness
/// (design §13.1 / §22 risk 4).
#[derive(Clone, Copy, Debug)]
pub struct WeakTargetEntry {
    /// The stored (possibly dead) entity handle this target refers to.
    pub entity: Entity,
    /// How many stored references point at this target (fan-in; `>= 1`).
    pub ref_count: usize,
    /// Whether the target still resolves in the audited [`World`].
    pub is_alive: bool,
}

/// Read-only health census of a [`WeakRefs`] table resolved against a live
/// [`World`]: liveness / prune pressure, duplicate multiplicity, and a
/// deterministic per-target breakdown (design §13.1 / §22 risk 4 / §16.6).
#[derive(Clone, Debug)]
pub struct WeakReferenceHealth {
    total_refs: usize,
    alive_ref_count: usize,
    alive_target_count: usize,
    max_fan_in: usize,
    targets: Vec<WeakTargetEntry>,
}

impl WeakReferenceHealth {
    /// Censuses a [`WeakRefs`] table against `world`: resolves every stored
    /// handle, deduplicates targets, and rolls up liveness and multiplicity.
    /// Read-only; `O(n + t log t)`.
    pub fn from_refs(refs: &WeakRefs, world: &World) -> Self {
        // Accumulate per distinct target handle: (entity, ref_count, is_alive).
        let mut by_target: HashMap<u64, (Entity, usize, bool)> = HashMap::default();
        let mut total_refs = 0usize;
        let mut alive_ref_count = 0usize;

        for weak in refs.iter() {
            let entity = weak.entity();
            let alive = weak.is_alive(world);
            total_refs += 1;
            if alive {
                alive_ref_count += 1;
            }
            let slot = by_target
                .entry(entity.to_bits())
                .or_insert((entity, 0, alive));
            slot.1 += 1;
        }

        let mut targets: Vec<WeakTargetEntry> = by_target
            .into_values()
            .map(|(entity, ref_count, is_alive)| WeakTargetEntry {
                entity,
                ref_count,
                is_alive,
            })
            .collect();
        targets.sort_unstable_by_key(|t| t.entity.to_bits());

        let alive_target_count = targets.iter().filter(|t| t.is_alive).count();
        let max_fan_in = targets.iter().map(|t| t.ref_count).max().unwrap_or(0);

        Self {
            total_refs,
            alive_ref_count,
            alive_target_count,
            max_fan_in,
            targets,
        }
    }

    /// Total stored references, counting duplicates (equals
    /// [`WeakRefs::len`]).
    #[inline]
    pub fn total_refs(&self) -> usize {
        self.total_refs
    }

    /// Whether the table holds no references.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_refs == 0
    }

    /// Number of distinct target handles referenced.
    #[inline]
    pub fn unique_target_count(&self) -> usize {
        self.targets.len()
    }

    /// Redundant references: stored handles beyond one per distinct target
    /// (`total_refs - unique_target_count`).
    #[inline]
    pub fn duplicate_ref_count(&self) -> usize {
        self.total_refs - self.targets.len()
    }

    /// Whether any target is referenced more than once.
    #[inline]
    pub fn has_duplicate_targets(&self) -> bool {
        self.max_fan_in > 1
    }

    /// The largest fan-in: how many times the most-referenced target appears.
    /// `0` when the table is empty.
    #[inline]
    pub fn max_fan_in(&self) -> usize {
        self.max_fan_in
    }

    /// Stored references whose target still resolves in the world.
    #[inline]
    pub fn alive_ref_count(&self) -> usize {
        self.alive_ref_count
    }

    /// Stored references whose target has been despawned (counting duplicates).
    #[inline]
    pub fn dangling_ref_count(&self) -> usize {
        self.total_refs - self.alive_ref_count
    }

    /// Distinct targets that still resolve in the world.
    #[inline]
    pub fn alive_target_count(&self) -> usize {
        self.alive_target_count
    }

    /// Distinct targets that have been despawned.
    #[inline]
    pub fn dangling_target_count(&self) -> usize {
        self.targets.len() - self.alive_target_count
    }

    /// Prune pressure in permille: the fraction of stored references a
    /// [`prune`](WeakRefs::prune) would drop (`dangling_ref × 1000 / total`).
    /// `0` when empty.
    pub fn dangling_ref_permille(&self) -> u64 {
        permille(self.dangling_ref_count() as u64, self.total_refs as u64)
    }

    /// Fraction of distinct targets that have gone stale, in permille.
    /// `0` when empty.
    pub fn dangling_target_permille(&self) -> u64 {
        permille(
            self.dangling_target_count() as u64,
            self.targets.len() as u64,
        )
    }

    /// Whether any stored reference dangles.
    #[inline]
    pub fn has_dangling(&self) -> bool {
        self.alive_ref_count < self.total_refs
    }

    /// Whether every stored reference still resolves (vacuously `true` when
    /// empty).
    #[inline]
    pub fn all_alive(&self) -> bool {
        self.alive_ref_count == self.total_refs
    }

    /// How many references a [`prune`](WeakRefs::prune) would remove right now
    /// (equals [`dangling_ref_count`](Self::dangling_ref_count)).
    #[inline]
    pub fn prune_would_drop(&self) -> usize {
        self.dangling_ref_count()
    }

    /// The per-target breakdown, ordered by [`Entity::to_bits`].
    #[inline]
    pub fn targets(&self) -> &[WeakTargetEntry] {
        &self.targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_table_is_empty() {
        let world = World::new();
        let refs = WeakRefs::new();
        let health = WeakReferenceHealth::from_refs(&refs, &world);
        assert!(health.is_empty());
        assert_eq!(health.total_refs(), 0);
        assert_eq!(health.unique_target_count(), 0);
        assert_eq!(health.duplicate_ref_count(), 0);
        assert_eq!(health.alive_ref_count(), 0);
        assert_eq!(health.dangling_ref_count(), 0);
        assert_eq!(health.max_fan_in(), 0);
        assert_eq!(health.dangling_ref_permille(), 0);
        assert!(!health.has_dangling());
        assert!(health.all_alive());
        assert!(health.targets().is_empty());
    }

    #[test]
    fn all_live_targets_report_no_dangling() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        let mut refs = WeakRefs::new();
        refs.push(a);
        refs.push(b);
        refs.push(c);
        let health = WeakReferenceHealth::from_refs(&refs, &world);
        assert_eq!(health.total_refs(), 3);
        assert_eq!(health.unique_target_count(), 3);
        assert_eq!(health.alive_ref_count(), 3);
        assert_eq!(health.dangling_ref_count(), 0);
        assert_eq!(health.alive_target_count(), 3);
        assert_eq!(health.dangling_target_count(), 0);
        assert!(health.all_alive());
        assert!(!health.has_dangling());
        assert_eq!(health.dangling_ref_permille(), 0);
        assert_eq!(health.prune_would_drop(), 0);
    }

    #[test]
    fn despawned_target_dangles_and_is_counted() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let mut refs = WeakRefs::new();
        refs.push(a);
        refs.push(b);
        assert!(world.despawn(b));
        let health = WeakReferenceHealth::from_refs(&refs, &world);
        assert_eq!(health.total_refs(), 2);
        assert_eq!(health.alive_ref_count(), 1);
        assert_eq!(health.dangling_ref_count(), 1);
        assert_eq!(health.alive_target_count(), 1);
        assert_eq!(health.dangling_target_count(), 1);
        assert!(health.has_dangling());
        assert!(!health.all_alive());
        assert_eq!(health.dangling_ref_permille(), 500);
        assert_eq!(health.prune_would_drop(), 1);
    }

    #[test]
    fn duplicates_counted_and_fan_in_tracked() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let mut refs = WeakRefs::new();
        refs.push(a);
        refs.push(a);
        refs.push(a);
        refs.push(b);
        let health = WeakReferenceHealth::from_refs(&refs, &world);
        assert_eq!(health.total_refs(), 4);
        assert_eq!(health.unique_target_count(), 2);
        assert_eq!(health.duplicate_ref_count(), 2);
        assert!(health.has_duplicate_targets());
        assert_eq!(health.max_fan_in(), 3);
    }

    #[test]
    fn dangling_duplicates_scale_prune_pressure() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let mut refs = WeakRefs::new();
        // Three handles to b, one to a; despawn b -> 3 of 4 refs dangle.
        refs.push(a);
        refs.push(b);
        refs.push(b);
        refs.push(b);
        assert!(world.despawn(b));
        let health = WeakReferenceHealth::from_refs(&refs, &world);
        assert_eq!(health.dangling_ref_count(), 3);
        assert_eq!(health.dangling_target_count(), 1);
        assert_eq!(health.alive_target_count(), 1);
        assert_eq!(health.dangling_ref_permille(), 750);
        assert_eq!(health.dangling_target_permille(), 500);
        assert_eq!(health.prune_would_drop(), 3);
    }

    #[test]
    fn targets_are_deduped_and_sorted() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        let mut refs = WeakRefs::new();
        // Push out of order and with duplicates.
        refs.push(c);
        refs.push(a);
        refs.push(b);
        refs.push(a);
        let health = WeakReferenceHealth::from_refs(&refs, &world);
        let targets = health.targets();
        assert_eq!(targets.len(), 3);
        // Sorted ascending by to_bits, every target accounted for once.
        let mut prev = 0u64;
        let mut total: usize = 0;
        for (i, t) in targets.iter().enumerate() {
            if i > 0 {
                assert!(t.entity.to_bits() > prev);
            }
            prev = t.entity.to_bits();
            total += t.ref_count;
            assert!(t.is_alive);
        }
        assert_eq!(total, health.total_refs());
    }

    #[test]
    fn mixed_liveness_rolls_up() {
        let mut world = World::new();
        let alive = world.spawn(());
        let dead1 = world.spawn(());
        let dead2 = world.spawn(());
        let mut refs = WeakRefs::new();
        refs.push(alive);
        refs.push(dead1);
        refs.push(dead2);
        assert!(world.despawn(dead1));
        assert!(world.despawn(dead2));
        let health = WeakReferenceHealth::from_refs(&refs, &world);
        assert_eq!(health.alive_ref_count(), 1);
        assert_eq!(health.dangling_ref_count(), 2);
        assert_eq!(health.alive_target_count(), 1);
        assert_eq!(health.dangling_target_count(), 2);
        // One of three distinct targets alive.
        assert_eq!(health.dangling_target_permille(), 666);
    }
}
