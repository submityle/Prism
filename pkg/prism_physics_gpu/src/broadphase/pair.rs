//! A broad-phase candidate pair.
//!
//! [`CandidatePair`] stores the two particle indices of a potential collision
//! with the invariant `a < b`, so a pair has one canonical form regardless of
//! discovery order. This canonicalisation is what lets the parity test compare
//! the `GPU` output (produced in nondeterministic atomic-append order) against
//! the `CPU` golden set by sorting both.
//!
//! Provenance: trivial index pair; no Unreal Engine source or derived code.

/// An unordered pair of particle indices, stored canonically as `a < b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CandidatePair {
    /// The smaller particle index.
    pub a: u32,
    /// The larger particle index.
    pub b: u32,
}

impl CandidatePair {
    /// Creates a canonical pair, swapping so that `a < b`.
    ///
    /// # Panics
    ///
    /// Panics if `i == j`, which is never a valid collision pair.
    #[must_use]
    pub fn new(i: u32, j: u32) -> CandidatePair {
        assert!(
            i != j,
            "a candidate pair must reference two distinct particles"
        );
        if i < j {
            CandidatePair { a: i, b: j }
        } else {
            CandidatePair { a: j, b: i }
        }
    }
}
