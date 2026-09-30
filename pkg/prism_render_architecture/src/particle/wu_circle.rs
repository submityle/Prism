//! Xiaolin Wu anti-aliased circle rasterization for the particle debug /
//! gizmo and soft-splat contracts (design §12-§13, §16, §29).
//!
//! Where [`crate::particle::midpoint_circle`] answers "which integer lattice
//! cells lie closest to the ideal ring" (a hard, alias-prone silhouette),
//! several particle stages instead need "how much of each edge pixel does the
//! ring cover". A smooth influence-radius overlay, a round soft-splat outline
//! drawn with sub-pixel smoothness, and the `CPU` reference for a smooth-circle
//! `GPU` kernel all want *fractional* pixel coverage so the ring reads as a
//! clean anti-aliased curve instead of a jagged staircase. This module owns
//! that contract: it turns a center and radius into a deterministic list of
//! `(x, y, coverage)` samples, where `coverage` in `[0, 1]` is the fraction of
//! the pixel painted by the ideal circle.
//!
//! # Algorithm
//! This is Xiaolin Wu's anti-aliased circle, expressed without any
//! trigonometry. Walking the first octant one integer column `x` at a time
//! from `x = 0` up to the diagonal `x = r / sqrt(2)`, the exact ring height is
//! `y = sqrt(r^2 - x^2)`. That height lands between two pixel rows: the lower
//! row `floor(y)` receives the majority weight `1 - frac` and the row directly
//! above it, `floor(y) + 1`, receives the remainder `frac`, where
//! `frac = y - floor(y)`. The two coverages a single column emits therefore sum
//! to one. Each octant `(x, y)` sample is mirrored through the circle's eight
//! symmetries, so an eighth of the walk paints the whole ring.
//!
//! # Coincident-pixel merge
//! On the axes (`x == 0`) and near the diagonal (`x == y`) several of the eight
//! reflections land on the same lattice cell, and a swapped reflection of one
//! column can coincide with a direct reflection of another. Rather than emit
//! duplicates, coincident samples are merged by keeping the *maximum* coverage
//! for each cell, which both deduplicates the exact axis/diagonal repeats and
//! keeps the strongest edge estimate where two octants meet. The merged cells
//! are then returned sorted (ascending by `x`, then `y`), so the output is
//! duplicate-free and deterministic.
//!
//! # Hand-rolled math only
//! The whole routine uses nothing beyond `+`, `-`, `*`, `/`, `f32::sqrt`,
//! `f32::floor`, `f32::abs`, `f32::min`, `f32::max`, and `f32::clamp`, plus an
//! integer `div_ceil` for the loop bound. There is no `sin`/`cos`, no
//! `exp`/`ln`, no `atan`, no `powf`, no `round`, and no `ceil`: rounding to the
//! nearest integer is spelled `(v + 0.5).floor()`, the integer part is
//! `v.floor()`, and the fractional part is `v - v.floor()`. Floating-point
//! values are never compared with `==` or `!=`; near-zero tests go through
//! explicit epsilon constants.
//!
//! # Strict scope
//! This module only rasterizes a *single anti-aliased circle ring* into
//! fractional coverage. It is deliberately distinct from the integer
//! midpoint/Bresenham ring in [`crate::particle::midpoint_circle`] (hard,
//! coverage-free lattice points) and from the anti-aliased *straight segment*
//! in [`crate::particle::wu_line`]. It draws no filled disks, ellipses, arcs,
//! or lines, and it neither imports nor mutates any sibling module; all state
//! lives in local values.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// Coverage at or below this magnitude is treated as "nothing painted" and the
/// sample is dropped from the output. Well under a single 8-bit intensity step
/// (`1/255`), so it never discards a visible pixel.
const COVERAGE_EPS: f32 = 1.0e-6;

/// Radii at or below this magnitude are treated as the degenerate "no ring"
/// case: a zero (or numerically negligible) radius has no anti-aliased edge to
/// paint, so [`rasterize`] returns the single center pixel at full coverage.
const RADIUS_EPS: f32 = 1.0e-6;

/// Integer part of `v` as an `f32` (its floor).
#[inline]
fn ipart(v: f32) -> f32 {
    v.floor()
}

/// Fractional part of `v` in `[0, 1)`, computed as `v - floor(v)` so it stays
/// correct for negative inputs (e.g. `fpart(-0.25) == 0.75`).
#[inline]
fn fpart(v: f32) -> f32 {
    v - v.floor()
}

/// Rounds `v` to the nearest integer (ties toward `+inf`) as an `f32`, spelled
/// with `floor` so no banned `round`/`ceil` is used.
#[inline]
fn round_nearest(v: f32) -> f32 {
    (v + 0.5).floor()
}

/// Records one octant sample at cell `(px, py)` (already an offset from the
/// center) into the merge map, keeping the maximum coverage seen for that cell
/// and dropping near-zero coverage entirely.
///
/// The coverage is clamped to `[0, 1]` first so a marginally-out-of-range value
/// from floating-point error can never escape into the output.
#[inline]
fn merge_sample(acc: &mut BTreeMap<(i32, i32), f32>, px: i32, py: i32, coverage: f32) {
    let c = coverage.clamp(0.0, 1.0);
    if c <= COVERAGE_EPS {
        return;
    }
    acc.entry((px, py))
        .and_modify(|existing| *existing = existing.max(c))
        .or_insert(c);
}

/// Emits the eight symmetric reflections of an octant offset `(x, y)` about the
/// center, all carrying the same `coverage`.
///
/// The eight reflections are the four sign combinations of `(±x, ±y)` plus the
/// same four with the axes swapped `(±y, ±x)`. When `(x, y)` sits on an axis
/// (`x == 0`) or the diagonal (`x == y`) some reflections coincide; the merge
/// map collapses those repeats by keeping the maximum coverage, so this helper
/// emits the full eight unconditionally.
#[inline]
fn emit_octant_symmetry(
    acc: &mut BTreeMap<(i32, i32), f32>,
    cx: i32,
    cy: i32,
    x: i32,
    y: i32,
    coverage: f32,
) {
    merge_sample(acc, cx + x, cy + y, coverage);
    merge_sample(acc, cx - x, cy + y, coverage);
    merge_sample(acc, cx + x, cy - y, coverage);
    merge_sample(acc, cx - x, cy - y, coverage);
    merge_sample(acc, cx + y, cy + x, coverage);
    merge_sample(acc, cx - y, cy + x, coverage);
    merge_sample(acc, cx + y, cy - x, coverage);
    merge_sample(acc, cx - y, cy - x, coverage);
}

/// Rasterizes the anti-aliased ring of radius `r` centered at the lattice point
/// `(cx, cy)` into an ordered list of `(x, y, coverage)` samples.
///
/// `coverage` lies in `[0, 1]` and is the fraction of the pixel painted by the
/// ideal circle. Each octant column contributes a lower pixel `floor(y)` with
/// weight `1 - frac` and the pixel above it with weight `frac`, so a column's
/// two coverages sum to one; the eight-way symmetry mirrors that walk into the
/// full ring. Coincident axis/diagonal reflections are merged by keeping the
/// maximum coverage, and the result is sorted (ascending by `x`, then `y`) and
/// duplicate-free. Special cases:
///
/// * `r <= 0` (within [`RADIUS_EPS`]) is the degenerate "no ring" case and
///   returns the single center pixel `(cx, cy, 1.0)`.
/// * `r < 0` is not a valid radius and returns an empty vector.
///
/// The output is deterministic: identical inputs yield bit-for-bit identical
/// samples, and translating the center by an integer offset shifts every
/// coordinate by that offset while leaving the coverages unchanged.
#[must_use]
pub fn rasterize(cx: i32, cy: i32, r: f32) -> Vec<(i32, i32, f32)> {
    // A NaN or negative radius is not a drawable ring.
    if r.is_nan() || r < 0.0 {
        return Vec::new();
    }

    let mut acc: BTreeMap<(i32, i32), f32> = BTreeMap::new();

    // Degenerate radius: a single fully-covered center pixel, no edge.
    if r <= RADIUS_EPS {
        acc.insert((cx, cy), 1.0);
        return acc.into_iter().map(|((x, y), c)| (x, y, c)).collect();
    }

    let r_sq = r * r;

    // Walk the first octant: x from 0 up to and including the diagonal x == y,
    // i.e. while 2*x^2 <= r^2. The nearest integer to r is always a generous
    // upper bound on that diagonal column (r / sqrt(2) < r); `div_ceil` on the
    // unsigned span rounds the guard up by one whole pixel with pure integer
    // arithmetic and no transcendental, so the diagonal is never missed.
    let x_limit = round_nearest(r) as i64;
    let x_span = if x_limit < 0 { 0u64 } else { x_limit as u64 };
    let x_max = ((x_span + 1).div_ceil(1)).max(1) as i64;

    let mut x: i64 = 0;
    while x <= x_max {
        let xf = x as f32;
        // Stop once we pass the 45-degree diagonal (2*x^2 > r^2).
        if 2.0 * xf * xf > r_sq {
            break;
        }
        let inside = r_sq - xf * xf;
        // Guard against a tiny negative from floating-point error.
        let y = if inside <= 0.0 { 0.0 } else { inside.sqrt() };

        let y_lo = ipart(y);
        let frac = fpart(y);

        let xi = x as i32;
        let y_lo_i = y_lo as i32;

        // Lower row keeps the majority weight; the row above takes the rest.
        emit_octant_symmetry(&mut acc, cx, cy, xi, y_lo_i, 1.0 - frac);
        emit_octant_symmetry(&mut acc, cx, cy, xi, y_lo_i + 1, frac);

        x += 1;
    }

    acc.into_iter().map(|((x, y), c)| (x, y, c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::{BTreeMap as Map, BTreeSet};
    use alloc::vec::Vec;

    /// Builds a lookup from cell to coverage for convenient assertions.
    fn coverage_map(samples: &[(i32, i32, f32)]) -> Map<(i32, i32), f32> {
        samples.iter().map(|&(x, y, c)| ((x, y), c)).collect()
    }

    /// The set of covered cells (coverage discarded).
    fn cell_set(samples: &[(i32, i32, f32)]) -> BTreeSet<(i32, i32)> {
        samples.iter().map(|&(x, y, _)| (x, y)).collect()
    }

    #[test]
    fn negative_radius_is_empty() {
        assert!(rasterize(0, 0, -1.0).is_empty());
        assert!(rasterize(3, -4, -0.001).is_empty());
    }

    #[test]
    fn nan_radius_is_empty() {
        assert!(rasterize(0, 0, f32::NAN).is_empty());
    }

    #[test]
    fn zero_radius_is_single_center_pixel() {
        let s = rasterize(2, 5, 0.0);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].0, 2);
        assert_eq!(s[0].1, 5);
        assert!((s[0].2 - 1.0).abs() <= COVERAGE_EPS);
    }

    #[test]
    fn negligible_radius_is_single_center_pixel() {
        let s = rasterize(-3, 7, RADIUS_EPS * 0.5);
        assert_eq!(s.len(), 1);
        assert_eq!((s[0].0, s[0].1), (-3, 7));
        assert!((s[0].2 - 1.0).abs() <= COVERAGE_EPS);
    }

    #[test]
    fn all_coverage_within_unit_interval() {
        for ri in 1..80 {
            let r = ri as f32 * 0.5;
            for &(_, _, c) in &rasterize(0, 0, r) {
                assert!(
                    (0.0..=1.0).contains(&c),
                    "coverage {c} out of [0,1] at r = {r}"
                );
            }
        }
    }

    #[test]
    fn coverage_strictly_positive_after_drop() {
        // Near-zero coverage is dropped, so every emitted sample is meaningful.
        for ri in 1..60 {
            let r = ri as f32 * 0.75;
            for &(_, _, c) in &rasterize(0, 0, r) {
                assert!(c > COVERAGE_EPS, "kept a near-zero sample at r = {r}");
            }
        }
    }

    #[test]
    fn output_is_sorted_and_unique() {
        for ri in 1..70 {
            let r = ri as f32 * 0.6;
            let s = rasterize(0, 0, r);
            for w in s.windows(2) {
                let a = (w[0].0, w[0].1);
                let b = (w[1].0, w[1].1);
                assert!(a < b, "not strictly sorted at r = {r}: {a:?} !< {b:?}");
            }
        }
    }

    #[test]
    fn no_duplicate_cells() {
        for ri in 1..70 {
            let r = ri as f32 * 0.6;
            let s = rasterize(0, 0, r);
            assert_eq!(cell_set(&s).len(), s.len(), "duplicate cell at r = {r}");
        }
    }

    #[test]
    fn four_fold_axis_mirror_symmetry() {
        // The covered-cell set is symmetric under (x,-y) and (-x,y) about center.
        for ri in 2..40 {
            let r = ri as f32;
            let set = cell_set(&rasterize(0, 0, r));
            for &(x, y) in &set {
                assert!(set.contains(&(-x, y)), "missing (-x,y) at r = {r}");
                assert!(set.contains(&(x, -y)), "missing (x,-y) at r = {r}");
                assert!(set.contains(&(-x, -y)), "missing (-x,-y) at r = {r}");
            }
        }
    }

    #[test]
    fn eight_fold_diagonal_mirror_symmetry() {
        // The covered-cell set is symmetric under the axis swap (x,y) -> (y,x).
        for ri in 2..40 {
            let r = ri as f32;
            let set = cell_set(&rasterize(0, 0, r));
            for &(x, y) in &set {
                assert!(set.contains(&(y, x)), "missing swapped (y,x) at r = {r}");
                assert!(set.contains(&(-y, x)), "missing (-y,x) at r = {r}");
                assert!(set.contains(&(y, -x)), "missing (y,-x) at r = {r}");
                assert!(set.contains(&(-y, -x)), "missing (-y,-x) at r = {r}");
            }
        }
    }

    #[test]
    fn coverage_matches_across_eight_reflections() {
        // Not just the cells: the coverage value is identical across the eight
        // symmetric images of any octant cell.
        let r = 17.0;
        let map = coverage_map(&rasterize(0, 0, r));
        for (&(x, y), &c) in &map {
            for &(rx, ry) in &[
                (-x, y),
                (x, -y),
                (-x, -y),
                (y, x),
                (-y, x),
                (y, -x),
                (-y, -x),
            ] {
                let rc = *map.get(&(rx, ry)).expect("reflection missing");
                assert!(
                    (rc - c).abs() <= COVERAGE_EPS,
                    "coverage mismatch: ({x},{y})={c} vs ({rx},{ry})={rc}"
                );
            }
        }
    }

    #[test]
    fn column_pair_coverages_sum_to_one() {
        // For an interior octant column the lower and upper pixel coverages sum
        // to one (the whole point of Wu's split).
        let r = 10.0;
        let map = coverage_map(&rasterize(0, 0, r));
        // Columns 1..=6 sit strictly below the diagonal for r = 10.
        for x in 1..=6 {
            let xf = x as f32;
            let y = (r * r - xf * xf).sqrt();
            let lo = y.floor() as i32;
            let cov_lo = *map.get(&(x, lo)).unwrap_or(&0.0);
            let cov_hi = *map.get(&(x, lo + 1)).unwrap_or(&0.0);
            assert!(
                (cov_lo + cov_hi - 1.0).abs() <= 1.0e-4,
                "column {x} coverages {cov_lo}+{cov_hi} != 1"
            );
        }
    }

    #[test]
    fn diagonal_neighbourhood_coverage_sums_to_one() {
        // Near the main diagonal (x ~ y) the ring still splits its weight
        // between two rows that sum to one.
        let r = 20.0;
        let map = coverage_map(&rasterize(0, 0, r));
        let x = 13; // below diagonal (13 < 20/sqrt(2) ~ 14.14)
        let xf = x as f32;
        let y = (r * r - xf * xf).sqrt();
        let lo = y.floor() as i32;
        let cov_lo = *map.get(&(x, lo)).unwrap_or(&0.0);
        let cov_hi = *map.get(&(x, lo + 1)).unwrap_or(&0.0);
        assert!((cov_lo + cov_hi - 1.0).abs() <= 1.0e-4);
    }

    #[test]
    fn axis_extreme_points_are_full_coverage() {
        // For an integer radius the four axis extrema land exactly on a lattice
        // row/column, so their coverage is (essentially) 1.
        let r = 12.0;
        let map = coverage_map(&rasterize(0, 0, r));
        for &(x, y) in &[(12, 0), (-12, 0), (0, 12), (0, -12)] {
            let c = *map.get(&(x, y)).expect("axis extreme missing");
            assert!(c >= 1.0 - 1.0e-3, "axis extreme ({x},{y}) coverage {c} < 1");
        }
    }

    #[test]
    fn axis_extreme_points_present_for_many_integer_radii() {
        for ri in 1..50 {
            let r = ri as f32;
            let set = cell_set(&rasterize(0, 0, r));
            assert!(set.contains(&(ri, 0)), "missing (+r,0) at r = {r}");
            assert!(set.contains(&(-ri, 0)), "missing (-r,0) at r = {r}");
            assert!(set.contains(&(0, ri)), "missing (0,+r) at r = {r}");
            assert!(set.contains(&(0, -ri)), "missing (0,-r) at r = {r}");
        }
    }

    #[test]
    fn all_samples_lie_near_the_ideal_ring() {
        // Every covered pixel is within a pixel of the mathematical circle:
        // its center's distance to the ring is at most ~sqrt(2)/2 plus slack.
        for ri in 3..60 {
            let r = ri as f32;
            for &(x, y, _) in &rasterize(0, 0, r) {
                let d = ((x * x + y * y) as f32).sqrt();
                assert!(
                    (d - r).abs() <= 1.5,
                    "pixel ({x},{y}) too far from ring r = {r}: d = {d}"
                );
            }
        }
    }

    #[test]
    fn point_count_grows_with_radius() {
        // Sampled at a coarse stride the ring's pixel count increases with r.
        let counts: Vec<usize> = (2..40)
            .step_by(4)
            .map(|ri| rasterize(0, 0, ri as f32).len())
            .collect();
        for w in counts.windows(2) {
            assert!(w[1] > w[0], "count did not grow: {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn point_count_trends_upward() {
        // The anti-aliased pixel count is not monotone step-to-step: at radii
        // with many integer right-triangle legs (e.g. r = 65 = 16^2+63^2 = ...)
        // many columns' upper row lands exactly on the ring, so its frac is
        // ~0 and that near-zero sample is dropped, dipping the total. The count
        // still scales linearly (~8/sqrt(2) columns, two rows each), so across
        // a factor-of-two radius jump the growth is unambiguous.
        for ri in 20..90 {
            let big = rasterize(0, 0, ri as f32).len();
            let half = rasterize(0, 0, (ri / 2) as f32).len();
            assert!(
                big > half,
                "count did not grow from r/2: {half} (r={}) -> {big} (r={ri})",
                ri / 2
            );
        }
    }

    #[test]
    fn translation_shifts_coordinates_only() {
        // An integer translation of the center shifts every coordinate by the
        // same offset and leaves the coverages unchanged.
        let base = rasterize(0, 0, 14.0);
        let shifted = rasterize(-25, 40, 14.0);
        let base_map = coverage_map(&base);
        let shifted_map = coverage_map(&shifted);
        assert_eq!(base.len(), shifted.len());
        for (&(x, y), &c) in &base_map {
            let sc = *shifted_map
                .get(&(x - 25, y + 40))
                .expect("shifted cell missing");
            assert!(
                (sc - c).abs() <= COVERAGE_EPS,
                "coverage changed under translation: {c} vs {sc}"
            );
        }
    }

    #[test]
    fn translation_preserves_coverage_multiset() {
        // The bag of coverage values is invariant under translation.
        let mut a: Vec<u32> = rasterize(0, 0, 9.0).iter().map(|s| s.2.to_bits()).collect();
        let mut b: Vec<u32> = rasterize(100, -70, 9.0)
            .iter()
            .map(|s| s.2.to_bits())
            .collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn deterministic_bit_for_bit() {
        for ri in 1..40 {
            let r = ri as f32 * 0.5;
            let a = rasterize(3, -2, r);
            let b = rasterize(3, -2, r);
            assert_eq!(a.len(), b.len());
            for (p, q) in a.iter().zip(b.iter()) {
                assert_eq!(p.0, q.0);
                assert_eq!(p.1, q.1);
                assert_eq!(p.2.to_bits(), q.2.to_bits(), "coverage bits differ");
            }
        }
    }

    #[test]
    fn merge_keeps_maximum_on_coincidence() {
        // The merge helper keeps the larger coverage for a repeated cell.
        let mut acc: Map<(i32, i32), f32> = Map::new();
        merge_sample(&mut acc, 4, 7, 0.3);
        merge_sample(&mut acc, 4, 7, 0.9);
        merge_sample(&mut acc, 4, 7, 0.5);
        assert!((acc[&(4, 7)] - 0.9).abs() <= COVERAGE_EPS);
    }

    #[test]
    fn merge_drops_near_zero_coverage() {
        let mut acc: Map<(i32, i32), f32> = Map::new();
        merge_sample(&mut acc, 1, 1, COVERAGE_EPS * 0.5);
        assert!(acc.is_empty());
    }

    #[test]
    fn merge_clamps_out_of_range_coverage() {
        let mut acc: Map<(i32, i32), f32> = Map::new();
        merge_sample(&mut acc, 0, 0, 1.4);
        merge_sample(&mut acc, 2, 2, -0.2);
        assert!((acc[&(0, 0)] - 1.0).abs() <= COVERAGE_EPS);
        assert!(!acc.contains_key(&(2, 2)), "negative coverage should drop");
    }

    #[test]
    fn small_radius_one_is_well_formed() {
        let s = rasterize(0, 0, 1.0);
        assert!(!s.is_empty());
        let set = cell_set(&s);
        // The four unit-axis extrema must be present at full coverage.
        for &p in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
            assert!(set.contains(&p), "missing {p:?} at r = 1");
        }
        for &(_, _, c) in &s {
            assert!((0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn small_fractional_radius_is_well_formed() {
        // A radius between the degenerate guard and 1 still produces a valid,
        // bounded, symmetric ring.
        let r = 0.5;
        let s = rasterize(0, 0, r);
        assert!(!s.is_empty());
        let set = cell_set(&s);
        for &(x, y) in &set {
            assert!(set.contains(&(-x, y)) && set.contains(&(x, -y)));
        }
        for &(_, _, c) in &s {
            assert!(c > COVERAGE_EPS && c <= 1.0);
        }
    }

    #[test]
    fn helpers_avoid_transcendental_rounding() {
        // ipart / fpart / round_nearest behave for negative inputs too.
        assert!((ipart(-0.25) - (-1.0)).abs() <= COVERAGE_EPS);
        assert!((fpart(-0.25) - 0.75).abs() <= 1.0e-6);
        assert!((round_nearest(2.5) - 3.0).abs() <= COVERAGE_EPS);
        assert!((round_nearest(2.49) - 2.0).abs() <= COVERAGE_EPS);
    }

    #[test]
    fn samples_stay_within_radius_bounding_box() {
        for ri in 1..60 {
            let r = ri as f32;
            let bound = ri + 1; // ceil(r) + 1 pixel of anti-alias slack
            for &(x, y, _) in &rasterize(0, 0, r) {
                assert!(
                    x.abs() <= bound && y.abs() <= bound,
                    "escaped box at r = {r}: ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn large_radius_is_well_formed() {
        let r = 500.0;
        let s = rasterize(0, 0, r);
        assert!(!s.is_empty());
        for &(x, y, c) in &s {
            assert!((0.0..=1.0).contains(&c));
            let d = ((x as f32) * (x as f32) + (y as f32) * (y as f32)).sqrt();
            assert!((d - r).abs() <= 1.5, "pixel far from ring at r = {r}");
        }
    }

    #[test]
    fn distinct_from_axis_only_when_expected() {
        // Sanity: a ring of radius 5 covers strictly more than the 4 axis
        // extrema (there is a genuine anti-aliased body).
        let s = rasterize(0, 0, 5.0);
        assert!(s.len() > 4, "ring r = 5 unexpectedly sparse: {}", s.len());
    }
}
