//! Persistent, cross-frame contact impulse cache for warm-starting.
//!
//! Iterative contact solvers (`PGS`, `TGS`, substep `XPBD`) converge far faster
//! and stack far more stably when each frame's solve *starts* from the previous
//! frame's converged solution instead of from rest. That is *warm-starting*: the
//! converged normal Lagrange multiplier of every contact is cached under a
//! stable, order-independent identity and fed back as the initial multiplier of
//! the matching contact next frame. New contacts (no cache entry) seed to `0`,
//! and contacts that separated (no constraint this frame) are pruned.
//!
//! This module is the *cache* — a pure data structure with no solver logic. The
//! warm-started solve itself lives beside its cold twin in
//! [`cpu_resolve_contacts_warm`](super::cpu_resolve_contacts_warm), which seeds
//! its multipliers from [`ContactCache::seed`], applies the matching impulse,
//! solves, and writes the converged multipliers back with
//! [`ContactCache::store`].
//!
//! # Stable identity
//!
//! Two overlapping particles are identified by the order-normalised pair
//! [`ContactKey`] `(min(a, b), max(a, b))`, so the same contact keys to the same
//! slot regardless of which particle the narrow phase happened to label `a`.
//! (A richer per-feature identity is what an oriented-box manifold cache will
//! need later; for the sphere pairs the solver handles today the pair *is* the
//! feature.)
//!
//! Provenance: standard warm-starting of an iterative constraint solver
//! (Catto 2005 sequential impulses; Müller et al. substep `XPBD`). No Unreal
//! Engine source or derived code.

use std::collections::HashMap;

use super::constraint::ContactConstraint;

/// The stable, order-independent identity of a contacting particle pair.
///
/// Constructed with the lower particle index first so that a contact keys to the
/// same slot whether the narrow phase labelled the pair `(a, b)` or `(b, a)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ContactKey {
    /// The lower of the two particle indices.
    lo: u32,
    /// The higher of the two particle indices.
    hi: u32,
}

impl ContactKey {
    /// Builds the order-independent key for the pair `(a, b)`.
    #[must_use]
    pub fn new(a: u32, b: u32) -> ContactKey {
        if a <= b {
            ContactKey { lo: a, hi: b }
        } else {
            ContactKey { lo: b, hi: a }
        }
    }

    /// The lower particle index of the pair.
    #[must_use]
    pub fn lo(self) -> u32 {
        self.lo
    }

    /// The higher particle index of the pair.
    #[must_use]
    pub fn hi(self) -> u32 {
        self.hi
    }
}

impl From<&ContactConstraint> for ContactKey {
    fn from(constraint: &ContactConstraint) -> ContactKey {
        ContactKey::new(constraint.a, constraint.b)
    }
}

/// A cross-frame store of each contact's last converged normal multiplier.
///
/// Carry one [`ContactCache`] per solved contact set across frames: pass it to
/// [`cpu_resolve_contacts_warm`](super::cpu_resolve_contacts_warm) every frame
/// and it seeds, then re-stores, itself. An empty cache is a cold start (every
/// contact seeds to `0`), so the first warmed frame is identical to the cold
/// solver.
#[derive(Clone, Debug, Default)]
pub struct ContactCache {
    /// Maps each order-normalised pair to its last converged multiplier.
    impulses: HashMap<ContactKey, f32>,
}

impl ContactCache {
    /// Creates an empty cache (a cold start).
    #[must_use]
    pub fn new() -> ContactCache {
        ContactCache {
            impulses: HashMap::new(),
        }
    }

    /// The number of contacts currently cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.impulses.len()
    }

    /// Whether the cache holds no contacts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.impulses.is_empty()
    }

    /// The cached multiplier for `key`, or `0.0` when the pair is new.
    #[must_use]
    pub fn get(&self, key: ContactKey) -> f32 {
        self.impulses.get(&key).copied().unwrap_or(0.0)
    }

    /// Produces the warm-start seed multiplier for each constraint, aligned
    /// one-to-one with `constraints`.
    ///
    /// A persistent contact yields its previous converged multiplier; a
    /// brand-new contact (absent from the cache) yields `0.0`, so a cold cache
    /// seeds every contact to `0` and the warmed solve degenerates exactly to
    /// the cold one.
    #[must_use]
    pub fn seed(&self, constraints: &[ContactConstraint]) -> Vec<f32> {
        constraints
            .iter()
            .map(|constraint| self.get(ContactKey::from(constraint)))
            .collect()
    }

    /// Rebuilds the cache from this frame's `constraints` and their converged
    /// `lambdas`, pruning every pair that has no constraint this frame.
    ///
    /// The two slices must be aligned one-to-one (the solver stores the exact
    /// list it seeded from). Rebuilding from scratch — rather than updating in
    /// place — is what prunes departed contacts: a pair that produced no
    /// constraint this frame simply is not re-inserted. When the same pair
    /// appears twice (which the sphere narrow phase never emits) the last
    /// multiplier wins.
    ///
    /// # Panics
    ///
    /// Panics when `constraints` and `lambdas` have different lengths, which
    /// would mean the solver seeded and converged mismatched lists.
    pub fn store(&mut self, constraints: &[ContactConstraint], lambdas: &[f32]) {
        assert_eq!(
            constraints.len(),
            lambdas.len(),
            "constraint and multiplier lists must be aligned one-to-one"
        );
        self.impulses.clear();
        for (constraint, &lambda) in constraints.iter().zip(lambdas.iter()) {
            self.impulses.insert(ContactKey::from(constraint), lambda);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_order_independent() {
        // The whole point of the key: (a, b) and (b, a) collapse to one slot.
        assert_eq!(ContactKey::new(3, 7), ContactKey::new(7, 3));
        let key = ContactKey::new(9, 2);
        assert_eq!(key.lo(), 2);
        assert_eq!(key.hi(), 9);
    }

    #[test]
    fn key_from_constraint_normalises_order() {
        let forward = ContactConstraint::new(1, 4, 2.0, 0.0);
        let reversed = ContactConstraint::new(4, 1, 2.0, 0.0);
        assert_eq!(ContactKey::from(&forward), ContactKey::from(&reversed));
    }

    #[test]
    fn empty_cache_seeds_every_contact_to_zero() {
        // A cold cache must seed zeros so the first warmed frame equals the cold
        // solve — the property the solver's parity test relies on.
        let cache = ContactCache::new();
        assert!(cache.is_empty());
        let cons = vec![
            ContactConstraint::new(0, 1, 2.0, 0.0),
            ContactConstraint::new(2, 3, 2.0, 0.0),
        ];
        assert_eq!(cache.seed(&cons), vec![0.0, 0.0]);
    }

    #[test]
    fn store_then_seed_round_trips_persistent_contacts() {
        // Store two converged multipliers, then confirm a matching constraint
        // list seeds back those exact values regardless of endpoint order.
        let mut cache = ContactCache::new();
        let stored = vec![
            ContactConstraint::new(0, 1, 2.0, 0.0),
            ContactConstraint::new(5, 2, 2.0, 0.0),
        ];
        cache.store(&stored, &[0.25, 0.5]);
        assert_eq!(cache.len(), 2);
        // Re-query with reversed endpoints to prove the key normalises.
        let query = vec![
            ContactConstraint::new(1, 0, 2.0, 0.0),
            ContactConstraint::new(2, 5, 2.0, 0.0),
        ];
        assert_eq!(cache.seed(&query), vec![0.25, 0.5]);
    }

    #[test]
    fn store_prunes_departed_contacts() {
        // A pair present last frame but gone this frame must not survive: the
        // rebuild only re-inserts the pairs it is given.
        let mut cache = ContactCache::new();
        cache.store(&[ContactConstraint::new(0, 1, 2.0, 0.0)], &[0.3]);
        assert_eq!(cache.len(), 1);
        cache.store(&[], &[]);
        assert!(cache.is_empty());
        assert_eq!(cache.get(ContactKey::new(0, 1)), 0.0);
    }

    #[test]
    fn get_is_zero_for_absent_pairs() {
        let mut cache = ContactCache::new();
        cache.store(&[ContactConstraint::new(0, 1, 2.0, 0.0)], &[0.4]);
        assert_eq!(cache.get(ContactKey::new(0, 1)), 0.4);
        assert_eq!(cache.get(ContactKey::new(4, 8)), 0.0);
    }

    #[test]
    #[should_panic(expected = "aligned one-to-one")]
    fn store_rejects_mismatched_lengths() {
        let mut cache = ContactCache::new();
        cache.store(&[ContactConstraint::new(0, 1, 2.0, 0.0)], &[0.1, 0.2]);
    }
}
