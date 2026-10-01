//! Persistent, cross-frame distance-constraint multiplier cache for
//! warm-starting the `XPBD` stretch solver.
//!
//! Iterative position-based solvers (substep `XPBD`, `TGS`) converge far faster
//! and hold a loaded structure far more stably when each frame's solve *starts*
//! from the previous frame's converged solution instead of from rest. That is
//! *warm-starting*: the converged Lagrange multiplier of every distance
//! constraint is cached under a stable, order-independent identity and fed back
//! as the initial multiplier of the matching constraint next frame. New
//! constraints (no cache entry) seed to `0`, and constraints that disappeared
//! (no constraint this frame) are pruned.
//!
//! This module is the *cache* — a pure data structure with no solver logic. The
//! warm-started solve itself lives beside its cold twin in
//! [`cpu_solve_warm`](super::cpu_solve_warm), which seeds its multipliers from
//! [`DistanceCache::seed`], applies the matching correction, solves, and writes
//! the converged multipliers back with [`DistanceCache::store`].
//!
//! # Stable identity
//!
//! Two coupled particles are identified by the order-normalised pair
//! [`DistanceKey`] `(min(a, b), max(a, b))`, so the same constraint keys to the
//! same slot regardless of which endpoint the caller happened to label `a`. A
//! distance constraint is fully identified by its particle pair (unlike an
//! oriented-box contact manifold, which needs a per-feature identity), so the
//! pair *is* the feature here.
//!
//! # Signed multipliers
//!
//! Unlike the one-sided contact constraint (whose multiplier is clamped
//! non-negative because a contact can only push), a distance constraint is
//! two-sided: it both pulls particles together when stretched and pushes them
//! apart when compressed, so its converged multiplier may be positive or
//! negative. The cache therefore stores the raw signed `f32` with no clamp, and
//! the warm-start applies it unconditionally (see [`cpu_solve_warm`]).
//!
//! Provenance: standard warm-starting of an iterative constraint solver (Müller
//! et al. substep `XPBD`). No Unreal Engine source or derived code.

use std::collections::HashMap;

use super::constraint::DistanceConstraint;

/// The stable, order-independent identity of a coupled particle pair.
///
/// Constructed with the lower particle index first so that a constraint keys to
/// the same slot whether the caller labelled the pair `(a, b)` or `(b, a)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DistanceKey {
    /// The lower of the two particle indices.
    lo: u32,
    /// The higher of the two particle indices.
    hi: u32,
}

impl DistanceKey {
    /// Builds the order-independent key for the pair `(a, b)`.
    #[must_use]
    pub fn new(a: u32, b: u32) -> DistanceKey {
        if a <= b {
            DistanceKey { lo: a, hi: b }
        } else {
            DistanceKey { lo: b, hi: a }
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

impl From<&DistanceConstraint> for DistanceKey {
    fn from(constraint: &DistanceConstraint) -> DistanceKey {
        DistanceKey::new(constraint.a, constraint.b)
    }
}

/// A cross-frame store of each distance constraint's last converged multiplier.
///
/// Carry one [`DistanceCache`] per solved constraint set across frames: pass it
/// to [`cpu_solve_warm`](super::cpu_solve_warm) every frame and it seeds, then
/// re-stores, itself. An empty cache is a cold start (every constraint seeds to
/// `0`), so the first warmed frame is identical to the cold solver.
#[derive(Clone, Debug, Default)]
pub struct DistanceCache {
    /// Maps each order-normalised pair to its last converged multiplier.
    impulses: HashMap<DistanceKey, f32>,
}

impl DistanceCache {
    /// Creates an empty cache (a cold start).
    #[must_use]
    pub fn new() -> DistanceCache {
        DistanceCache {
            impulses: HashMap::new(),
        }
    }

    /// The number of constraints currently cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.impulses.len()
    }

    /// Whether the cache holds no constraints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.impulses.is_empty()
    }

    /// The cached multiplier for `key`, or `0.0` when the pair is new.
    #[must_use]
    pub fn get(&self, key: DistanceKey) -> f32 {
        self.impulses.get(&key).copied().unwrap_or(0.0)
    }

    /// Produces the warm-start seed multiplier for each constraint, aligned
    /// one-to-one with `constraints`.
    ///
    /// A persistent constraint yields its previous converged multiplier; a
    /// brand-new constraint (absent from the cache) yields `0.0`, so a cold
    /// cache seeds every constraint to `0` and the warmed solve degenerates
    /// exactly to the cold one.
    #[must_use]
    pub fn seed(&self, constraints: &[DistanceConstraint]) -> Vec<f32> {
        constraints
            .iter()
            .map(|constraint| self.get(DistanceKey::from(constraint)))
            .collect()
    }

    /// Rebuilds the cache from this frame's `constraints` and their converged
    /// `lambdas`, pruning every pair that has no constraint this frame.
    ///
    /// The two slices must be aligned one-to-one (the solver stores the exact
    /// list it seeded from). Rebuilding from scratch — rather than updating in
    /// place — is what prunes departed constraints: a pair that produced no
    /// constraint this frame simply is not re-inserted. When the same pair
    /// appears twice the last multiplier wins.
    ///
    /// # Panics
    ///
    /// Panics when `constraints` and `lambdas` have different lengths, which
    /// would mean the solver seeded and converged mismatched lists.
    pub fn store(&mut self, constraints: &[DistanceConstraint], lambdas: &[f32]) {
        assert_eq!(
            constraints.len(),
            lambdas.len(),
            "constraint and multiplier lists must be aligned one-to-one"
        );
        self.impulses.clear();
        for (constraint, &lambda) in constraints.iter().zip(lambdas.iter()) {
            self.impulses.insert(DistanceKey::from(constraint), lambda);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_order_independent() {
        // The whole point of the key: (a, b) and (b, a) collapse to one slot.
        assert_eq!(DistanceKey::new(3, 7), DistanceKey::new(7, 3));
        let key = DistanceKey::new(9, 2);
        assert_eq!(key.lo(), 2);
        assert_eq!(key.hi(), 9);
    }

    #[test]
    fn key_from_constraint_normalises_order() {
        let forward = DistanceConstraint::new(1, 4, 2.0, 0.0);
        let reversed = DistanceConstraint::new(4, 1, 2.0, 0.0);
        assert_eq!(DistanceKey::from(&forward), DistanceKey::from(&reversed));
    }

    #[test]
    fn empty_cache_seeds_every_constraint_to_zero() {
        // A cold cache must seed zeros so the first warmed frame equals the cold
        // solve — the property the solver's parity test relies on.
        let cache = DistanceCache::new();
        assert!(cache.is_empty());
        let cons = vec![
            DistanceConstraint::new(0, 1, 2.0, 0.0),
            DistanceConstraint::new(2, 3, 2.0, 0.0),
        ];
        assert_eq!(cache.seed(&cons), vec![0.0, 0.0]);
    }

    #[test]
    fn store_then_seed_round_trips_persistent_constraints() {
        // Store two converged multipliers (one negative to prove signed values
        // survive), then confirm a matching list seeds back those exact values
        // regardless of endpoint order.
        let mut cache = DistanceCache::new();
        let stored = vec![
            DistanceConstraint::new(0, 1, 2.0, 0.0),
            DistanceConstraint::new(5, 2, 2.0, 0.0),
        ];
        cache.store(&stored, &[0.25, -0.5]);
        assert_eq!(cache.len(), 2);
        // Re-query with reversed endpoints to prove the key normalises.
        let query = vec![
            DistanceConstraint::new(1, 0, 2.0, 0.0),
            DistanceConstraint::new(2, 5, 2.0, 0.0),
        ];
        assert_eq!(cache.seed(&query), vec![0.25, -0.5]);
    }

    #[test]
    fn store_prunes_departed_constraints() {
        // A pair present last frame but gone this frame must not survive: the
        // rebuild only re-inserts the pairs it is given.
        let mut cache = DistanceCache::new();
        cache.store(&[DistanceConstraint::new(0, 1, 2.0, 0.0)], &[0.3]);
        assert_eq!(cache.len(), 1);
        cache.store(&[], &[]);
        assert!(cache.is_empty());
        assert_eq!(cache.get(DistanceKey::new(0, 1)), 0.0);
    }

    #[test]
    fn get_is_zero_for_absent_pairs() {
        let mut cache = DistanceCache::new();
        cache.store(&[DistanceConstraint::new(0, 1, 2.0, 0.0)], &[0.4]);
        assert_eq!(cache.get(DistanceKey::new(0, 1)), 0.4);
        assert_eq!(cache.get(DistanceKey::new(4, 8)), 0.0);
    }

    #[test]
    #[should_panic(expected = "aligned one-to-one")]
    fn store_rejects_mismatched_lengths() {
        let mut cache = DistanceCache::new();
        cache.store(&[DistanceConstraint::new(0, 1, 2.0, 0.0)], &[0.1, 0.2]);
    }
}
