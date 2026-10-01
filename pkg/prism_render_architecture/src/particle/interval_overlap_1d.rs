//! One-dimensional closed-interval algebra for particle lifetime windows.
//!
//! Many particle subsystems reduce to reasoning about a single axis: a
//! particle lifetime window `[spawn, death]`, one axis of an `AABB`, a slice of
//! the simulation time line, or the overlap test between two scheduled emitter
//! bursts. This module models that axis as a closed interval `[min, max]` and
//! offers the full set of membership, overlap, and set-combination queries the
//! higher layers lean on.
//!
//! Every comparison is a plain `f32` ordering (`<`, `<=`, `>`, `>=`); exact
//! floating-point equality (`==` / `!=`) is never used, so denormal spawn times
//! and accumulated drift never produce a spurious "equal" verdict. Where a
//! canonical ordering is required (sorting before a sweep merge) the total
//! order from `f32::total_cmp` is used instead. The algebra stays inside the
//! elementary `min` / `max` / `abs` / `clamp` repertoire so it runs identically
//! on a `CPU` core or a `GPU` lane, with no transcendental detour and no
//! `unsafe`.
//!
//! The empty interval is encoded as `[+inf, -inf]`, which makes `is_empty`
//! a single ordering test and lets `intersect` return it naturally when two
//! intervals do not meet. The `everything` interval spans `[-inf, +inf]`.

extern crate alloc;

use alloc::vec::Vec;
use core::cmp::Ordering;

/// Absolute tolerance used by the touch/adjacency tests in this module.
///
/// Two interval endpoints that sit within this distance of each other are
/// treated as touching for the purpose of merge adjacency, so sweep merges do
/// not leave hairline gaps from accumulated `f32` drift.
pub const INTERVAL_EPSILON: f32 = 1.0e-6;

/// A closed one-dimensional interval `[min, max]`.
///
/// The canonical form keeps `min <= max`; constructors enforce that invariant.
/// The sentinel empty value `[+inf, -inf]` deliberately violates it so that
/// `is_empty` and the set operations can detect emptiness with a single
/// ordering comparison. Deriving `PartialEq` is intentional: structural
/// comparison of the two fields is convenient for sentinels, while hand-written
/// floating-point equality is avoided everywhere else.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    /// Lower (inclusive) bound of the interval.
    pub min: f32,
    /// Upper (inclusive) bound of the interval.
    pub max: f32,
}

impl Interval {
    /// Builds an interval from two bounds, swapping them when `min > max`.
    ///
    /// The result is always in canonical `min <= max` form, so callers may pass
    /// endpoints in either order.
    #[must_use]
    pub fn new(min: f32, max: f32) -> Self {
        if min > max {
            Self { min: max, max: min }
        } else {
            Self { min, max }
        }
    }

    /// Builds a degenerate interval covering the single value `v`.
    #[must_use]
    pub fn point(v: f32) -> Self {
        Self { min: v, max: v }
    }

    /// Returns the canonical empty interval `[+inf, -inf]`.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }

    /// Returns the all-covering interval `[-inf, +inf]`.
    #[must_use]
    pub fn everything() -> Self {
        Self {
            min: f32::NEG_INFINITY,
            max: f32::INFINITY,
        }
    }

    /// Reports whether this interval contains no points.
    ///
    /// True exactly when `min > max`, which only the empty sentinel (or a
    /// value that collapsed below emptiness) satisfies.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.min > self.max
    }

    /// Reports whether the scalar `v` lies within the closed interval.
    ///
    /// An empty interval contains nothing, which falls out of the `min <= max`
    /// invariant without a special case.
    #[must_use]
    pub fn contains(&self, v: f32) -> bool {
        self.min <= v && v <= self.max
    }

    /// Reports whether `other` is wholly contained in this interval.
    ///
    /// An empty `other` is a subset of everything, so this returns `true`; a
    /// non-empty `other` inside an empty `self` returns `false`.
    #[must_use]
    pub fn contains_interval(&self, other: Interval) -> bool {
        if other.is_empty() {
            return true;
        }
        if self.is_empty() {
            return false;
        }
        self.min <= other.min && other.max <= self.max
    }

    /// Reports whether the two intervals share at least one point.
    ///
    /// Touching at a single endpoint counts as overlap. Either interval being
    /// empty yields `false`.
    #[must_use]
    pub fn overlaps(&self, other: Interval) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        self.min <= other.max && other.min <= self.max
    }

    /// Returns the length `max - min`, clamped to be non-negative.
    ///
    /// The empty interval reports `0.0`.
    #[must_use]
    pub fn length(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        (self.max - self.min).max(0.0)
    }

    /// Returns the midpoint `(min + max) * 0.5` of the interval.
    #[must_use]
    pub fn center(&self) -> f32 {
        (self.min + self.max) * 0.5
    }

    /// Clamps the scalar `v` into the closed interval.
    ///
    /// For an empty interval the bounds are degenerate, so the value is pinned
    /// to `min` to avoid relying on an invalid `clamp` range.
    #[must_use]
    pub fn clamp_value(&self, v: f32) -> f32 {
        if self.is_empty() {
            return self.min;
        }
        v.clamp(self.min, self.max)
    }

    /// Returns the intersection of the two intervals.
    ///
    /// When the intervals do not meet the canonical empty interval is returned.
    #[must_use]
    pub fn intersect(&self, other: Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return Interval::empty();
        }
        let lo = self.min.max(other.min);
        let hi = self.max.min(other.max);
        if lo > hi {
            Interval::empty()
        } else {
            Interval { min: lo, max: hi }
        }
    }

    /// Returns the smallest interval enclosing both operands (their hull).
    ///
    /// An empty operand contributes nothing, so the hull of an interval with
    /// the empty interval is the interval itself.
    #[must_use]
    pub fn hull(&self, other: Interval) -> Interval {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return *self;
        }
        Interval {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Returns the gap (empty space) between the two intervals.
    ///
    /// Overlapping or touching intervals report `0.0`; an empty operand also
    /// reports `0.0` since there is no meaningful separation to measure.
    #[must_use]
    pub fn gap(&self, other: Interval) -> f32 {
        if self.is_empty() || other.is_empty() {
            return 0.0;
        }
        if self.max < other.min {
            return other.min - self.max;
        }
        if other.max < self.min {
            return self.min - other.max;
        }
        0.0
    }

    /// Returns the interval grown by `amount` on both sides.
    ///
    /// A negative `amount` shrinks the interval; `new` re-normalizes the result
    /// so an over-shrunk interval collapses rather than inverting. The empty
    /// interval stays empty.
    #[must_use]
    pub fn expand(&self, amount: f32) -> Interval {
        if self.is_empty() {
            return Interval::empty();
        }
        let lo = self.min - amount;
        let hi = self.max + amount;
        if lo > hi {
            // Over-shrunk by a negative `amount`: collapse to the midpoint of
            // the inverted bounds instead of letting the interval flip.
            let mid = (lo + hi) * 0.5;
            Interval::new(mid, mid)
        } else {
            Interval::new(lo, hi)
        }
    }

    /// Returns the interval shifted by `delta` along the axis.
    ///
    /// The empty interval stays empty.
    #[must_use]
    pub fn translate(&self, delta: f32) -> Interval {
        if self.is_empty() {
            return Interval::empty();
        }
        Interval {
            min: self.min + delta,
            max: self.max + delta,
        }
    }
}

/// Merges a slice of intervals into a minimal set of disjoint intervals.
///
/// The input need not be pre-sorted: a working copy is sorted by `min` using
/// the total order from `f32::total_cmp`, then a single sweep fuses any
/// interval that overlaps or touches (within `INTERVAL_EPSILON`) the current
/// run. Empty intervals are skipped. The returned intervals are sorted by
/// `min` and are pairwise non-overlapping.
#[must_use]
pub fn merge_sorted(intervals: &[Interval]) -> Vec<Interval> {
    let mut sorted: Vec<Interval> = intervals
        .iter()
        .copied()
        .filter(|i| !i.is_empty())
        .collect();
    sorted.sort_by(|a, b| a.min.total_cmp(&b.min));

    let mut merged: Vec<Interval> = Vec::new();
    for current in sorted {
        if let Some(last) = merged.last_mut() {
            // Touching or overlapping if the next lower bound does not clear the
            // running upper bound by more than the adjacency tolerance.
            if current.min <= last.max + INTERVAL_EPSILON {
                if current.max > last.max {
                    last.max = current.max;
                }
                continue;
            }
        }
        merged.push(current);
    }
    merged
}

/// Returns the total length covered by the union of the intervals.
///
/// The intervals are merged first so overlapping coverage is only counted once.
#[must_use]
pub fn total_covered_length(intervals: &[Interval]) -> f32 {
    let merged = merge_sorted(intervals);
    let mut total = 0.0;
    for interval in &merged {
        total += interval.length();
    }
    total
}

/// Orders two intervals by lower bound, then upper bound, via `total_cmp`.
///
/// This is factored out so the `PartialOrd` implementation can delegate to the
/// total `Ord`, avoiding a divergent partial ordering.
impl Ord for Interval {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.min.total_cmp(&other.min) {
            Ordering::Equal => self.max.total_cmp(&other.max),
            non_equal => non_equal,
        }
    }
}

impl PartialOrd for Interval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Interval {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for value comparisons inside the tests.
    const EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn new_swaps_inverted_bounds() {
        let i = Interval::new(5.0, 1.0);
        assert!(approx(i.min, 1.0));
        assert!(approx(i.max, 5.0));
    }

    #[test]
    fn new_keeps_ordered_bounds() {
        let i = Interval::new(-2.0, 3.0);
        assert!(approx(i.min, -2.0));
        assert!(approx(i.max, 3.0));
    }

    #[test]
    fn point_is_degenerate() {
        let i = Interval::point(2.5);
        assert!(approx(i.min, 2.5));
        assert!(approx(i.max, 2.5));
        assert!(approx(i.length(), 0.0));
    }

    #[test]
    fn empty_reports_empty() {
        let i = Interval::empty();
        assert!(i.is_empty());
    }

    #[test]
    fn empty_contains_nothing() {
        let i = Interval::empty();
        assert!(!i.contains(0.0));
        assert!(!i.contains(f32::MAX));
    }

    #[test]
    fn everything_contains_extremes() {
        let i = Interval::everything();
        assert!(i.contains(0.0));
        assert!(i.contains(1.0e30));
        assert!(i.contains(-1.0e30));
    }

    #[test]
    fn everything_is_not_empty() {
        let i = Interval::everything();
        assert!(!i.is_empty());
    }

    #[test]
    fn contains_inside() {
        let i = Interval::new(0.0, 10.0);
        assert!(i.contains(5.0));
    }

    #[test]
    fn contains_both_boundaries() {
        let i = Interval::new(0.0, 10.0);
        assert!(i.contains(0.0));
        assert!(i.contains(10.0));
    }

    #[test]
    fn contains_outside() {
        let i = Interval::new(0.0, 10.0);
        assert!(!i.contains(-0.5));
        assert!(!i.contains(10.5));
    }

    #[test]
    fn contains_interval_true() {
        let outer = Interval::new(0.0, 10.0);
        let inner = Interval::new(2.0, 8.0);
        assert!(outer.contains_interval(inner));
    }

    #[test]
    fn contains_interval_false_when_crossing() {
        let outer = Interval::new(0.0, 10.0);
        let crossing = Interval::new(5.0, 15.0);
        assert!(!outer.contains_interval(crossing));
    }

    #[test]
    fn contains_interval_handles_empty_other() {
        let outer = Interval::new(0.0, 10.0);
        assert!(outer.contains_interval(Interval::empty()));
    }

    #[test]
    fn empty_does_not_contain_nonempty_interval() {
        let empty = Interval::empty();
        let other = Interval::new(0.0, 1.0);
        assert!(!empty.contains_interval(other));
    }

    #[test]
    fn overlaps_separate_is_false() {
        let a = Interval::new(0.0, 1.0);
        let b = Interval::new(2.0, 3.0);
        assert!(!a.overlaps(b));
    }

    #[test]
    fn overlaps_touching_is_true() {
        let a = Interval::new(0.0, 1.0);
        let b = Interval::new(1.0, 2.0);
        assert!(a.overlaps(b));
    }

    #[test]
    fn overlaps_contained_is_true() {
        let a = Interval::new(0.0, 10.0);
        let b = Interval::new(3.0, 4.0);
        assert!(a.overlaps(b));
    }

    #[test]
    fn overlaps_with_empty_is_false() {
        let a = Interval::new(0.0, 10.0);
        assert!(!a.overlaps(Interval::empty()));
        assert!(!Interval::empty().overlaps(a));
    }

    #[test]
    fn intersect_overlapping() {
        let a = Interval::new(0.0, 5.0);
        let b = Interval::new(3.0, 10.0);
        let r = a.intersect(b);
        assert!(approx(r.min, 3.0));
        assert!(approx(r.max, 5.0));
    }

    #[test]
    fn intersect_disjoint_is_empty() {
        let a = Interval::new(0.0, 1.0);
        let b = Interval::new(2.0, 3.0);
        assert!(a.intersect(b).is_empty());
    }

    #[test]
    fn intersect_touching_is_point() {
        let a = Interval::new(0.0, 1.0);
        let b = Interval::new(1.0, 2.0);
        let r = a.intersect(b);
        assert!(!r.is_empty());
        assert!(approx(r.min, 1.0));
        assert!(approx(r.max, 1.0));
    }

    #[test]
    fn intersect_with_empty_is_empty() {
        let a = Interval::new(0.0, 1.0);
        assert!(a.intersect(Interval::empty()).is_empty());
    }

    #[test]
    fn hull_of_disjoint() {
        let a = Interval::new(0.0, 1.0);
        let b = Interval::new(4.0, 6.0);
        let r = a.hull(b);
        assert!(approx(r.min, 0.0));
        assert!(approx(r.max, 6.0));
    }

    #[test]
    fn hull_with_empty_is_identity() {
        let a = Interval::new(2.0, 7.0);
        let r = a.hull(Interval::empty());
        assert!(approx(r.min, 2.0));
        assert!(approx(r.max, 7.0));
        let r2 = Interval::empty().hull(a);
        assert!(approx(r2.min, 2.0));
        assert!(approx(r2.max, 7.0));
    }

    #[test]
    fn gap_of_disjoint() {
        let a = Interval::new(0.0, 1.0);
        let b = Interval::new(4.0, 5.0);
        assert!(approx(a.gap(b), 3.0));
        assert!(approx(b.gap(a), 3.0));
    }

    #[test]
    fn gap_of_overlapping_is_zero() {
        let a = Interval::new(0.0, 5.0);
        let b = Interval::new(3.0, 8.0);
        assert!(approx(a.gap(b), 0.0));
    }

    #[test]
    fn gap_of_touching_is_zero() {
        let a = Interval::new(0.0, 2.0);
        let b = Interval::new(2.0, 4.0);
        assert!(approx(a.gap(b), 0.0));
    }

    #[test]
    fn length_of_normal_interval() {
        let a = Interval::new(-1.0, 4.0);
        assert!(approx(a.length(), 5.0));
    }

    #[test]
    fn length_of_empty_is_zero() {
        assert!(approx(Interval::empty().length(), 0.0));
    }

    #[test]
    fn length_of_point_is_zero() {
        assert!(approx(Interval::point(3.0).length(), 0.0));
    }

    #[test]
    fn center_is_midpoint() {
        let a = Interval::new(2.0, 8.0);
        assert!(approx(a.center(), 5.0));
    }

    #[test]
    fn clamp_below_pins_to_min() {
        let a = Interval::new(0.0, 10.0);
        assert!(approx(a.clamp_value(-3.0), 0.0));
    }

    #[test]
    fn clamp_above_pins_to_max() {
        let a = Interval::new(0.0, 10.0);
        assert!(approx(a.clamp_value(42.0), 10.0));
    }

    #[test]
    fn clamp_inside_is_identity() {
        let a = Interval::new(0.0, 10.0);
        assert!(approx(a.clamp_value(4.0), 4.0));
    }

    #[test]
    fn expand_grows_both_sides() {
        let a = Interval::new(2.0, 4.0);
        let r = a.expand(1.0);
        assert!(approx(r.min, 1.0));
        assert!(approx(r.max, 5.0));
    }

    #[test]
    fn expand_negative_can_collapse() {
        let a = Interval::new(0.0, 2.0);
        let r = a.expand(-2.0);
        assert!(approx(r.min, r.max));
    }

    #[test]
    fn translate_shifts_both_bounds() {
        let a = Interval::new(1.0, 3.0);
        let r = a.translate(5.0);
        assert!(approx(r.min, 6.0));
        assert!(approx(r.max, 8.0));
    }

    #[test]
    fn merge_sorted_fuses_overlapping() {
        let input = [Interval::new(0.0, 3.0), Interval::new(2.0, 5.0)];
        let out = merge_sorted(&input);
        assert_eq!(out.len(), 1);
        assert!(approx(out[0].min, 0.0));
        assert!(approx(out[0].max, 5.0));
    }

    #[test]
    fn merge_sorted_fuses_touching() {
        let input = [Interval::new(0.0, 1.0), Interval::new(1.0, 2.0)];
        let out = merge_sorted(&input);
        assert_eq!(out.len(), 1);
        assert!(approx(out[0].min, 0.0));
        assert!(approx(out[0].max, 2.0));
    }

    #[test]
    fn merge_sorted_absorbs_contained() {
        let input = [Interval::new(0.0, 10.0), Interval::new(2.0, 4.0)];
        let out = merge_sorted(&input);
        assert_eq!(out.len(), 1);
        assert!(approx(out[0].min, 0.0));
        assert!(approx(out[0].max, 10.0));
    }

    #[test]
    fn merge_sorted_keeps_disjoint() {
        let input = [Interval::new(0.0, 1.0), Interval::new(3.0, 4.0)];
        let out = merge_sorted(&input);
        assert_eq!(out.len(), 2);
        assert!(approx(out[0].min, 0.0));
        assert!(approx(out[0].max, 1.0));
        assert!(approx(out[1].min, 3.0));
        assert!(approx(out[1].max, 4.0));
    }

    #[test]
    fn merge_sorted_handles_unsorted_input() {
        let input = [
            Interval::new(5.0, 6.0),
            Interval::new(0.0, 1.0),
            Interval::new(0.5, 2.0),
        ];
        let out = merge_sorted(&input);
        assert_eq!(out.len(), 2);
        assert!(approx(out[0].min, 0.0));
        assert!(approx(out[0].max, 2.0));
        assert!(approx(out[1].min, 5.0));
        assert!(approx(out[1].max, 6.0));
    }

    #[test]
    fn merge_sorted_skips_empty() {
        let input = [
            Interval::empty(),
            Interval::new(1.0, 2.0),
            Interval::empty(),
        ];
        let out = merge_sorted(&input);
        assert_eq!(out.len(), 1);
        assert!(approx(out[0].min, 1.0));
        assert!(approx(out[0].max, 2.0));
    }

    #[test]
    fn total_covered_length_overlapping() {
        let input = [Interval::new(0.0, 3.0), Interval::new(2.0, 5.0)];
        assert!(approx(total_covered_length(&input), 5.0));
    }

    #[test]
    fn total_covered_length_disjoint() {
        let input = [Interval::new(0.0, 1.0), Interval::new(3.0, 4.0)];
        assert!(approx(total_covered_length(&input), 2.0));
    }

    #[test]
    fn total_covered_length_empty_slice_is_zero() {
        let input: [Interval; 0] = [];
        assert!(approx(total_covered_length(&input), 0.0));
    }

    #[test]
    fn ordering_is_by_min_then_max() {
        let a = Interval::new(0.0, 5.0);
        let b = Interval::new(0.0, 7.0);
        let c = Interval::new(1.0, 2.0);
        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&c), Ordering::Less);
        assert_eq!(a.partial_cmp(&a), Some(Ordering::Equal));
    }
}
