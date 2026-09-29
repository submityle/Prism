//! Exact integer ratios for utilization and drop-rate reporting.
//!
//! Work-graph accounting reports fractions such as queue utilization
//! (occupancy over capacity) and drop rate (dropped over produced). Reporting
//! those as `f32` invites the classic floating-point equality traps: two runs
//! that accept the same work can disagree in the last mantissa bit, and golden
//! comparisons need epsilons instead of exact matches.
//!
//! [`Ratio`] sidesteps that entirely. It stores a numerator and denominator as
//! integers, normalizes to lowest terms on construction, and compares by
//! cross-multiplication in a wider integer type. Equal fractions therefore have
//! identical representations, so `==`, [`Ord`], and hashing are all exact and
//! deterministic. Callers that still want a scalar can ask for parts-per-million
//! via [`Ratio::per_million`], which rounds down through a `u128` intermediate
//! and never overflows for `u64` inputs.

/// An exact, normalized non-negative ratio of two integers.
///
/// A denominator of zero is treated as the ratio `0/1`, which keeps utilization
/// of an empty (zero-capacity) queue well defined at zero rather than undefined.
/// Construction always reduces the fraction to lowest terms, so two ratios are
/// equal exactly when they denote the same rational value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Ratio {
    numerator: u64,
    denominator: u64,
}

impl Ratio {
    /// The ratio `0/1`.
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };

    /// The ratio `1/1`.
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    /// Creates a normalized ratio `numerator / denominator`.
    ///
    /// A zero denominator collapses to [`Ratio::ZERO`]. Otherwise the fraction
    /// is reduced by its greatest common divisor so equal values share one
    /// representation.
    #[must_use]
    pub const fn new(numerator: u64, denominator: u64) -> Self {
        if denominator == 0 || numerator == 0 {
            return Self::ZERO;
        }
        let divisor = gcd(numerator, denominator);
        Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        }
    }

    /// The reduced numerator.
    #[must_use]
    pub const fn numerator(self) -> u64 {
        self.numerator
    }

    /// The reduced denominator; never zero.
    #[must_use]
    pub const fn denominator(self) -> u64 {
        self.denominator
    }

    /// Returns `true` when the ratio is exactly zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.numerator == 0
    }

    /// Returns `true` when the ratio is greater than or equal to one.
    #[must_use]
    pub const fn is_saturated(self) -> bool {
        self.numerator >= self.denominator
    }

    /// The ratio expressed in parts-per-million, rounded toward zero.
    ///
    /// The multiply happens in `u128`, so no `u64` numerator/denominator pair
    /// overflows. A ratio of one yields `1_000_000`; ratios above one (possible
    /// only via [`Ratio::new`] with `numerator > denominator`) exceed it.
    #[must_use]
    pub const fn per_million(self) -> u64 {
        let scaled = (self.numerator as u128) * 1_000_000u128;
        (scaled / self.denominator as u128) as u64
    }

    /// The ratio expressed in whole percent, rounded toward zero.
    #[must_use]
    pub const fn percent(self) -> u64 {
        let scaled = (self.numerator as u128) * 100u128;
        (scaled / self.denominator as u128) as u64
    }
}

impl Default for Ratio {
    fn default() -> Self {
        Self::ZERO
    }
}

impl PartialOrd for Ratio {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ratio {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Cross-multiply in u128 to compare a/b vs c/d without division.
        let lhs = (self.numerator as u128) * (other.denominator as u128);
        let rhs = (other.numerator as u128) * (self.denominator as u128);
        lhs.cmp(&rhs)
    }
}

/// Greatest common divisor via the binary-free Euclidean algorithm.
///
/// Both inputs are assumed non-zero by the sole caller; the loop still
/// terminates for any input because the remainder strictly decreases.
const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_denominator_is_zero() {
        assert_eq!(Ratio::new(5, 0), Ratio::ZERO);
        assert!(Ratio::new(5, 0).is_zero());
    }

    #[test]
    fn equal_values_share_representation() {
        assert_eq!(Ratio::new(1, 2), Ratio::new(2, 4));
        assert_eq!(Ratio::new(3, 9), Ratio::new(1, 3));
        assert_eq!(Ratio::new(0, 7), Ratio::ZERO);
    }

    #[test]
    fn ordering_is_exact() {
        assert!(Ratio::new(1, 3) < Ratio::new(1, 2));
        assert!(Ratio::new(2, 3) > Ratio::new(1, 2));
        assert!(Ratio::new(2, 4) == Ratio::new(1, 2));
        assert!(Ratio::ONE.is_saturated());
        assert!(Ratio::new(5, 4).is_saturated());
        assert!(!Ratio::new(3, 4).is_saturated());
    }

    #[test]
    fn per_million_and_percent_round_down() {
        assert_eq!(Ratio::ONE.per_million(), 1_000_000);
        assert_eq!(Ratio::new(1, 2).per_million(), 500_000);
        assert_eq!(Ratio::new(1, 3).per_million(), 333_333);
        assert_eq!(Ratio::new(1, 3).percent(), 33);
        assert_eq!(Ratio::new(3, 4).percent(), 75);
        assert_eq!(Ratio::ZERO.per_million(), 0);
    }

    #[test]
    fn per_million_handles_large_inputs() {
        let r = Ratio::new(u64::MAX, u64::MAX);
        assert_eq!(r, Ratio::ONE);
        assert_eq!(r.per_million(), 1_000_000);
    }

    #[test]
    fn determinism_of_construction() {
        for n in 0u64..50 {
            for d in 1u64..50 {
                assert_eq!(Ratio::new(n, d), Ratio::new(n, d));
                assert_eq!(Ratio::new(n, d), Ratio::new(n * 2, d * 2));
            }
        }
    }
}
