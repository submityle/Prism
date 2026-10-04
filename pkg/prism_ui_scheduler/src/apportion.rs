//! Lossless proportional apportionment of an integer budget.
//!
//! The scheduler often needs to divide one indivisible resource — a frame's
//! microsecond [`FrameBudget`], a fixed step quota — among several lanes in
//! proportion to per-lane weights. Naive `total * weight / sum` truncates every
//! share toward zero, so the parts sum to *less* than the whole and a few
//! microseconds leak away every frame; naive rounding instead risks handing out
//! more than exists. Neither is acceptable when the budget is a hard deadline.
//!
//! This module uses **Hamilton's largest-remainder method**: every share first
//! takes its integer floor, then the leftover units (always fewer than the
//! number of weights) go one each to the shares whose fractional remainders are
//! largest, ties broken toward the lower index. The result provably sums to the
//! original `total` whenever any weight is non-zero, so no unit is created or
//! lost, and each share stays within one unit of its exact proportional quota.
//!
//! All arithmetic is exact integer math (widened to [`u128`] to avoid overflow
//! when `total * weight` is large), so apportionment is deterministic and
//! identical across platforms — a requirement for replay and headless parity.
//!
//! # Example
//!
//! ```
//! use prism_ui_scheduler::apportion;
//!
//! // A 100 µs budget split 1:2:3 — floors are 16, 33, 50 (sum 99); the single
//! // leftover microsecond goes to the largest remainder (the first share).
//! assert_eq!(apportion(100, &[1, 2, 3]), [17, 33, 50]);
//! ```

use alloc::vec::Vec;

use crate::budget::FrameBudget;
use crate::lane::Lane;

/// Apportions `total` across `weights` by Hamilton's largest-remainder method.
///
/// Returns one share per weight, in input order. When the weights sum to a
/// non-zero value the shares sum **exactly** to `total`; when every weight is
/// zero (or `weights` is empty) there is nothing to apportion against and every
/// share is zero. Each share differs from its exact proportional quota
/// `total * weight / sum` by less than one unit.
///
/// Arithmetic is performed in [`u128`], so `total` and the weights may span the
/// full [`u64`] range without overflow.
#[must_use]
pub fn apportion(total: u64, weights: &[u64]) -> Vec<u64> {
    let n = weights.len();
    let mut shares: Vec<u64> = alloc::vec![0; n];

    let weight_sum: u128 = weights.iter().map(|&w| u128::from(w)).sum();
    if weight_sum == 0 {
        return shares;
    }

    let total_u = u128::from(total);
    // Each entry's floor share plus the fractional remainder it leaves behind,
    // tagged with its index so ties resolve toward the lower index.
    let mut remainders: Vec<(u128, usize)> = Vec::with_capacity(n);
    let mut base_sum: u128 = 0;
    for (i, &w) in weights.iter().enumerate() {
        let scaled = total_u * u128::from(w);
        let floor = scaled / weight_sum;
        shares[i] = floor as u64;
        base_sum += floor;
        remainders.push((scaled % weight_sum, i));
    }

    // `base_sum <= total_u` and the shortfall is strictly less than `n`, so it
    // can always be spread one unit per distinct index.
    let leftover = (total_u - base_sum) as usize;

    // Largest remainder first; equal remainders favor the lower index.
    remainders.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for &(_, idx) in remainders.iter().take(leftover) {
        shares[idx] += 1;
    }

    shares
}

/// Apportions a [`FrameBudget`]'s microseconds across weighted [`Lane`]s.
///
/// The returned pairs preserve input order and lane, with each lane carrying
/// its apportioned microsecond share; the shares sum to the budget whenever any
/// weight is non-zero. This hands each priority lane a proportional slice of the
/// frame without losing microseconds to rounding.
///
/// # Example
///
/// ```
/// use prism_ui_scheduler::{apportion_lanes, FrameBudget, Lane};
///
/// let split = apportion_lanes(
///     FrameBudget::from_micros(1_000),
///     &[(Lane::Input, 3), (Lane::Visible, 1)],
/// );
/// assert_eq!(split, [(Lane::Input, 750), (Lane::Visible, 250)]);
/// ```
#[must_use]
pub fn apportion_lanes(budget: FrameBudget, weighted: &[(Lane, u64)]) -> Vec<(Lane, u64)> {
    let weights: Vec<u64> = weighted.iter().map(|&(_, w)| w).collect();
    let shares = apportion(budget.micros(), &weights);
    weighted
        .iter()
        .zip(shares)
        .map(|(&(lane, _), micros)| (lane, micros))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{apportion, apportion_lanes};
    use crate::budget::FrameBudget;
    use crate::lane::Lane;
    use alloc::vec::Vec;

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    #[test]
    fn satisfies_largest_remainder_invariants_over_random_inputs() {
        let mut rng = SplitMix64(0x5CED_0000_0000_0001);
        for _ in 0..5_000 {
            let n = 1 + rng.below(12) as usize;
            let weights: Vec<u64> = (0..n).map(|_| rng.below(1_000)).collect();
            let total = rng.below(1_000_000);

            let shares = apportion(total, &weights);
            assert_eq!(shares.len(), n);

            // Independently recompute floors, remainders and the leftover count.
            let weight_sum: u128 = weights.iter().map(|&w| u128::from(w)).sum();
            if weight_sum == 0 {
                assert!(shares.iter().all(|&s| s == 0));
                continue;
            }
            let total_u = u128::from(total);
            let floors: Vec<u128> = weights
                .iter()
                .map(|&w| total_u * u128::from(w) / weight_sum)
                .collect();
            let rems: Vec<u128> = weights
                .iter()
                .map(|&w| total_u * u128::from(w) % weight_sum)
                .collect();
            let base: u128 = floors.iter().sum();
            let leftover = (total_u - base) as usize;

            // Exact conservation: the parts sum to the whole.
            let sum: u128 = shares.iter().map(|&s| u128::from(s)).sum();
            assert_eq!(sum, total_u, "weights={weights:?} total={total}");

            // Every share is its floor or floor + 1, and exactly `leftover`
            // shares were bumped.
            let mut bumped = Vec::new();
            for (i, &s) in shares.iter().enumerate() {
                let s = u128::from(s);
                assert!(s == floors[i] || s == floors[i] + 1);
                if s == floors[i] + 1 {
                    bumped.push(i);
                }
            }
            assert_eq!(bumped.len(), leftover);

            // Ordering: no bumped index may have a strictly worse remainder key
            // than any unbumped one (remainder desc, then index asc).
            for &i in &bumped {
                for j in 0..n {
                    if u128::from(shares[j]) == floors[j] {
                        let i_better = rems[i] > rems[j] || (rems[i] == rems[j] && i < j);
                        assert!(i_better, "bad tie-break i={i} j={j} weights={weights:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn equal_weights_bias_low_indices() {
        // 10 split three equal ways: 3,3,3 then one leftover to index 0.
        assert_eq!(apportion(10, &[1, 1, 1]), [4, 3, 3]);
    }

    #[test]
    fn documented_ratio_split() {
        assert_eq!(apportion(100, &[1, 2, 3]), [17, 33, 50]);
    }

    #[test]
    fn zero_weight_entries_get_nothing() {
        assert_eq!(apportion(10, &[0, 1, 1]), [0, 5, 5]);
    }

    #[test]
    fn all_zero_weights_yield_zero_shares() {
        assert_eq!(apportion(10, &[0, 0]), [0, 0]);
    }

    #[test]
    fn empty_weights_yield_empty() {
        assert_eq!(apportion(10, &[]), Vec::<u64>::new());
    }

    #[test]
    fn zero_total_yields_zero_shares() {
        assert_eq!(apportion(0, &[1, 2, 3]), [0, 0, 0]);
    }

    #[test]
    fn single_weight_takes_everything() {
        assert_eq!(apportion(42, &[7]), [42]);
    }

    #[test]
    fn lanes_preserve_order_and_sum_to_budget() {
        let split = apportion_lanes(
            FrameBudget::from_micros(1_000),
            &[(Lane::Input, 3), (Lane::Visible, 1)],
        );
        assert_eq!(split, [(Lane::Input, 750), (Lane::Visible, 250)]);
        let sum: u64 = split.iter().map(|&(_, m)| m).sum();
        assert_eq!(sum, 1_000);
    }

    #[test]
    fn lanes_with_no_weight_split_nothing() {
        let split = apportion_lanes(
            FrameBudget::from_micros(500),
            &[(Lane::Immediate, 0), (Lane::Idle, 0)],
        );
        assert_eq!(split, [(Lane::Immediate, 0), (Lane::Idle, 0)]);
    }
}
