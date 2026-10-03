//! Entity dormancy (design §13.2).
//!
//! Not every entity needs to be visited every frame. A settled rigid body, a
//! finished AI agent, a prop the player walked away from — these are *dormant*:
//! they hold their state but are pulled out of the active scheduling set so the
//! per-frame cost scales with the number of *awake* entities, not the total
//! population (the MassEntity philosophy, design §13.2). A dormant entity is
//! woken explicitly — by a gameplay event, a collision, an interest source
//! moving into range (design §13.1), or an LOD band change (design §13.2 /
//! [`super::lod`]).
//!
//! This module is the pure, `World`-independent bookkeeping half: a
//! [`DormancySet`] tracks which entities are dormant and records *wake events*
//! so the scheduler can re-admit exactly the entities that woke this frame.
//! It performs no scheduling itself; the schedule layer drains
//! [`DormancySet::drain_woken`] each frame and re-includes those entities.
//!
//! Determinism (design §14): every ordered output is sorted by
//! [`Entity::to_bits`], so a given sequence of sleep/wake calls yields a
//! frame-stable, platform-independent result.

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::Component;
use crate::entity::Entity;

/// Marker [`Component`] tagging an entity as dormant (design §13.2).
///
/// Attaching `Dormant` lets queries and the scheduler cheaply skip the entity
/// without destroying its storage, mirroring the `Disabled` pattern (design
/// §23.1) but reserved for the "asleep until woken" case. The authoritative
/// awake/asleep set is [`DormancySet`]; this tag is the per-entity reflection
/// of it for query filtering.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Dormant;

impl Component for Dormant {}

/// Tracks the set of dormant entities and the wake events raised since the last
/// drain (design §13.2).
///
/// The scheduler puts an entity to [`sleep`](Self::sleep) to remove it from the
/// active set, and [`wake`](Self::wake)s it in response to an event. Each frame
/// the scheduler calls [`drain_woken`](Self::drain_woken) to learn which
/// entities must be re-admitted to the active schedule.
#[derive(Default)]
pub struct DormancySet {
    /// Currently-dormant entities. `()` value — this is a set keyed by entity.
    dormant: HashMap<Entity, ()>,
    /// Entities woken since the last [`drain_woken`](Self::drain_woken).
    woken: Vec<Entity>,
}

impl DormancySet {
    /// Creates an empty dormancy set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Puts `entity` to sleep (removes it from the active set).
    ///
    /// Returns `true` if the entity was awake and is now dormant, `false` if it
    /// was already dormant. If the entity has a pending wake event that has not
    /// been drained yet, sleeping it again cancels that pending wake.
    pub fn sleep(&mut self, entity: Entity) -> bool {
        self.woken.retain(|&e| e != entity);
        self.dormant.insert(entity, ()).is_none()
    }

    /// Wakes `entity`, recording a wake event for the next
    /// [`drain_woken`](Self::drain_woken).
    ///
    /// Returns `true` if the entity was dormant and is now awake, `false` if it
    /// was already awake (in which case no duplicate wake event is recorded).
    pub fn wake(&mut self, entity: Entity) -> bool {
        if self.dormant.remove(&entity).is_some() {
            // Guard against a stale duplicate (e.g. sleep→wake→sleep→wake would
            // have cleared it in `sleep`, but be defensive against misuse).
            if !self.woken.contains(&entity) {
                self.woken.push(entity);
            }
            true
        } else {
            false
        }
    }

    /// Whether `entity` is currently dormant.
    pub fn is_dormant(&self, entity: Entity) -> bool {
        self.dormant.contains_key(&entity)
    }

    /// Number of currently-dormant entities.
    pub fn len(&self) -> usize {
        self.dormant.len()
    }

    /// Whether no entity is currently dormant.
    pub fn is_empty(&self) -> bool {
        self.dormant.is_empty()
    }

    /// Number of wake events pending drain.
    pub fn pending_wake_count(&self) -> usize {
        self.woken.len()
    }

    /// Drains and returns the entities woken since the last call, sorted by
    /// [`Entity::to_bits`] for deterministic re-admission (design §14).
    ///
    /// After this call the pending-wake list is empty.
    pub fn drain_woken(&mut self) -> Vec<Entity> {
        let mut out = core::mem::take(&mut self.woken);
        out.sort_unstable_by_key(|e| e.to_bits());
        out
    }

    /// Iterates the currently-dormant entities in unspecified order.
    ///
    /// Use [`sorted_dormant`](Self::sorted_dormant) when deterministic order is
    /// required.
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.dormant.keys().copied()
    }

    /// All currently-dormant entities, sorted by [`Entity::to_bits`].
    pub fn sorted_dormant(&self) -> Vec<Entity> {
        let mut out: Vec<Entity> = self.dormant.keys().copied().collect();
        out.sort_unstable_by_key(|e| e.to_bits());
        out
    }

    /// Wakes every dormant entity at once, recording a wake event for each
    /// (e.g. a cell streaming in, design §13.1). Returns the number woken.
    pub fn wake_all(&mut self) -> usize {
        let mut woken_now = self.sorted_dormant();
        let count = woken_now.len();
        self.dormant.clear();
        for e in woken_now.drain(..) {
            if !self.woken.contains(&e) {
                self.woken.push(e);
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a test entity; `gen` must be nonzero (enforced by `from_bits`).
    fn ent(index: u32, generation: u32) -> Entity {
        Entity::from_bits(((generation as u64) << 32) | index as u64)
            .expect("test entity needs a nonzero generation")
    }

    #[test]
    fn sleep_marks_dormant() {
        let mut d = DormancySet::new();
        let e = ent(1, 1);
        assert!(d.sleep(e));
        assert!(d.is_dormant(e));
        assert_eq!(d.len(), 1);
        // Sleeping again is idempotent.
        assert!(!d.sleep(e));
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn wake_removes_and_records_event() {
        let mut d = DormancySet::new();
        let e = ent(2, 1);
        d.sleep(e);
        assert!(d.wake(e));
        assert!(!d.is_dormant(e));
        assert_eq!(d.pending_wake_count(), 1);
        // Waking an already-awake entity is a no-op with no duplicate event.
        assert!(!d.wake(e));
        assert_eq!(d.pending_wake_count(), 1);
    }

    #[test]
    fn drain_woken_is_sorted_and_clears() {
        let mut d = DormancySet::new();
        let a = ent(5, 1);
        let b = ent(2, 1);
        let c = ent(9, 1);
        d.sleep(a);
        d.sleep(b);
        d.sleep(c);
        d.wake(c);
        d.wake(a);
        d.wake(b);
        let woken = d.drain_woken();
        assert_eq!(woken, alloc::vec![b, a, c]); // sorted by bits: index 2,5,9
        assert_eq!(d.pending_wake_count(), 0);
        // Second drain is empty.
        assert!(d.drain_woken().is_empty());
    }

    #[test]
    fn sleep_cancels_pending_wake() {
        let mut d = DormancySet::new();
        let e = ent(3, 1);
        d.sleep(e);
        d.wake(e);
        assert_eq!(d.pending_wake_count(), 1);
        // Re-sleeping before drain cancels the stale wake event.
        d.sleep(e);
        assert_eq!(d.pending_wake_count(), 0);
        assert!(d.is_dormant(e));
    }

    #[test]
    fn generation_distinguishes_entities() {
        let mut d = DormancySet::new();
        let old = ent(4, 1);
        let recycled = ent(4, 2);
        d.sleep(old);
        assert!(d.is_dormant(old));
        assert!(!d.is_dormant(recycled));
    }

    #[test]
    fn wake_all_wakes_everything() {
        let mut d = DormancySet::new();
        d.sleep(ent(1, 1));
        d.sleep(ent(2, 1));
        d.sleep(ent(3, 1));
        assert_eq!(d.wake_all(), 3);
        assert!(d.is_empty());
        assert_eq!(d.drain_woken().len(), 3);
    }

    #[test]
    fn sorted_dormant_is_deterministic() {
        let mut d = DormancySet::new();
        d.sleep(ent(7, 1));
        d.sleep(ent(1, 1));
        d.sleep(ent(4, 1));
        let sorted = d.sorted_dormant();
        assert_eq!(sorted, alloc::vec![ent(1, 1), ent(4, 1), ent(7, 1)]);
    }
}
