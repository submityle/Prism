//! Integer midpoint (Bresenham) circle rasterization for particle sprite and
//! debug-gizmo footprints (design §12-§13, §16).
//!
//! Several particle stages need the *pixel-exact* boundary of a circle on an
//! integer lattice: a debug overlay draws an emitter's influence radius; a
//! screen-space culling reference wants the discrete silhouette of a round
//! splat; and an atlas-authoring pass stamps circular masks into a texture. All
//! of them share one requirement — a rasterizer that turns a center and radius
//! into the set of lattice points closest to the ideal circle, with no gaps and
//! no floating-point drift. This module owns that contract.
//!
//! The boundary is produced by the classic *midpoint circle* algorithm. It
//! walks the second octant (`x` from `0` up to `y`, with `y` starting at the
//! radius) and, at each step, keeps a single integer *decision variable* that
//! decides whether the next pixel stays on the current row or drops one row
//! inward. Every plotted octant point is reflected through the circle's eight
//! symmetries, so a single eighth of the work draws the whole ring. Points that
//! land on an axis (`x == 0`) or on a diagonal (`x == y`) map onto fewer than
//! eight distinct reflections; the final [`rasterize`] output is sorted and
//! deduplicated so those coincident reflections collapse to one lattice point.
//!
//! [`filled_disk`] fills the *solid* disk `x^2 + y^2 <= r^2` with horizontal
//! spans: one contiguous run of pixels per scan row, its half-width found by an
//! integer scan. The filled disk is defined directly by the squared-distance
//! inequality, so it is exact and complete (every lattice point inside the
//! mathematical disk is present, and nothing outside it is).
//!
//! # Strict scope
//! This module only rasterizes circles and fills disks on an integer lattice.
//! It does not draw ellipses, arcs, lines, or anti-aliased coverage; it does
//! not build convex hulls ([`super::convex_hull_2d`]), test polygon containment
//! ([`super::point_in_polygon`]), or measure polygon area
//! ([`super::polygon_area_2d`]). It neither imports nor reconstructs those
//! sibling contracts and keeps its own tiny integer helpers.
//!
//! # No transcendental or floating-point math
//! The rasterizer is pure integer arithmetic: additions, integer multiplies,
//! and comparisons on the decision variable. There is no `sqrt`, no `sin`,
//! `cos`, `atan`, `exp`, `ln`, `powf`, `ceil`, `round`, and no `f32` anywhere in
//! the module. Squared-distance comparisons widen to [`i64`] so that even large
//! radii cannot overflow the intermediate products.

use alloc::vec::Vec;

/// The squared Euclidean distance `dx^2 + dy^2`, computed in [`i64`] so the
/// intermediate products never overflow for any [`i32`] offset.
///
/// This is the only "distance" the module speaks: the midpoint algorithm and
/// the disk-fill inequality both compare *squared* lengths, which keeps the
/// whole contract in exact integer arithmetic with no `sqrt`.
#[must_use]
pub fn dist_sq(dx: i32, dy: i32) -> i64 {
    let x = dx as i64;
    let y = dy as i64;
    x * x + y * y
}

/// Pushes the eight symmetric reflections of an octant point `(x, y)` about the
/// center `(cx, cy)`.
///
/// The eight reflections are the four sign combinations of `(±x, ±y)` plus the
/// same four with the axes swapped `(±y, ±x)`. When `(x, y)` sits on an axis
/// (`x == 0`) or a diagonal (`x == y`), some of these reflections coincide;
/// [`rasterize`] removes the resulting duplicates in a single sort-and-dedup
/// pass, so this helper can emit the full eight unconditionally.
fn push_octant_symmetry(out: &mut Vec<(i32, i32)>, cx: i32, cy: i32, x: i32, y: i32) {
    out.push((cx + x, cy + y));
    out.push((cx - x, cy + y));
    out.push((cx + x, cy - y));
    out.push((cx - x, cy - y));
    out.push((cx + y, cy + x));
    out.push((cx - y, cy + x));
    out.push((cx + y, cy - x));
    out.push((cx - y, cy - x));
}

/// Rasterizes the boundary of the circle of radius `r` centered at `(cx, cy)`
/// into the set of integer lattice points closest to the ideal circle.
///
/// The result is sorted (ascending by `x`, then by `y`) and free of duplicates,
/// which makes the output deterministic and lets the axis/diagonal reflections
/// collapse cleanly. Special cases:
///
/// * `r == 0` returns the single center point `(cx, cy)`.
/// * `r < 0` is not a valid radius and returns an empty vector.
///
/// Every returned point `p` satisfies the midpoint-algorithm error bound
/// `|dist_sq(p - center) - r*r| <= 2*r + 1`: it lies within a pixel of the
/// mathematical circle. Because the octant walk covers `x in [0, y]` and the
/// eight-way symmetry mirrors it, the ring is closed with no gaps between
/// neighboring plotted pixels.
#[must_use]
pub fn rasterize(cx: i32, cy: i32, r: i32) -> Vec<(i32, i32)> {
    let mut pts: Vec<(i32, i32)> = Vec::new();
    if r < 0 {
        return pts;
    }
    if r == 0 {
        pts.push((cx, cy));
        return pts;
    }

    let mut x: i32 = 0;
    let mut y: i32 = r;
    // Midpoint decision variable, initialized to `1 - r`. When it is negative
    // the ideal circle passes above the midpoint, so the next pixel keeps the
    // current row; otherwise it drops one row inward.
    let mut d: i32 = 1 - r;
    while x <= y {
        push_octant_symmetry(&mut pts, cx, cy, x, y);
        if d < 0 {
            // Stay on the same row: advance x only.
            d += 2 * x + 3;
        } else {
            // Step inward: advance x and drop y by one row.
            d += 2 * (x - y) + 5;
            y -= 1;
        }
        x += 1;
    }

    pts.sort_unstable();
    pts.dedup();
    pts
}

/// Fills the solid disk `x^2 + y^2 <= r*r` centered at `(cx, cy)` with one
/// contiguous horizontal span of lattice points per scan row.
///
/// The half-width of each row is found by an integer scan (grow the half-width
/// while the next column still satisfies the squared-distance inequality), so
/// the disk is defined exactly by `x^2 + y^2 <= r*r` with no `sqrt`. The output
/// is emitted row by row from `cy - r` to `cy + r`, and within each row from
/// left to right, making it deterministic. Special cases:
///
/// * `r == 0` returns the single center point `(cx, cy)`.
/// * `r < 0` returns an empty vector.
///
/// The result is complete (every lattice point inside the mathematical disk is
/// present) and sound (no point outside the disk is included). Its size equals
/// the number of lattice points in the disk.
#[must_use]
pub fn filled_disk(cx: i32, cy: i32, r: i32) -> Vec<(i32, i32)> {
    let mut pts: Vec<(i32, i32)> = Vec::new();
    if r < 0 {
        return pts;
    }
    if r == 0 {
        pts.push((cx, cy));
        return pts;
    }

    let r_sq = (r as i64) * (r as i64);
    let mut dy = -r;
    while dy <= r {
        let yy = (dy as i64) * (dy as i64);
        // Largest half-width `dx >= 0` with `dx^2 + dy^2 <= r^2`.
        let mut dx: i32 = 0;
        while dist_sq(dx + 1, dy) <= r_sq {
            dx += 1;
        }
        debug_assert!(yy + (dx as i64) * (dx as i64) <= r_sq);
        let mut px = cx - dx;
        let row_y = cy + dy;
        while px <= cx + dx {
            pts.push((px, row_y));
            px += 1;
        }
        dy += 1;
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    fn as_set(pts: &[(i32, i32)]) -> BTreeSet<(i32, i32)> {
        pts.iter().copied().collect()
    }

    #[test]
    fn dist_sq_is_exact_squared_length() {
        assert_eq!(dist_sq(3, 4), 25);
        assert_eq!(dist_sq(-3, -4), 25);
        assert_eq!(dist_sq(0, 0), 0);
        assert_eq!(dist_sq(5, 0), 25);
    }

    #[test]
    fn dist_sq_does_not_overflow_for_large_offsets() {
        // 46_340^2 + 46_340^2 far exceeds i32::MAX but fits comfortably in i64.
        assert_eq!(dist_sq(46_340, 46_340), 2 * 46_340i64 * 46_340i64);
    }

    #[test]
    fn radius_zero_is_a_single_center_point() {
        let pts = rasterize(7, -3, 0);
        assert_eq!(pts, alloc::vec![(7, -3)]);
    }

    #[test]
    fn negative_radius_rasterize_is_empty() {
        assert!(rasterize(0, 0, -1).is_empty());
        assert!(rasterize(2, 2, -100).is_empty());
    }

    #[test]
    fn radius_one_is_the_four_axis_neighbors() {
        let pts = rasterize(0, 0, 1);
        let expected = as_set(&[(0, 1), (0, -1), (1, 0), (-1, 0)]);
        assert_eq!(as_set(&pts), expected);
        assert_eq!(pts.len(), 4);
    }

    #[test]
    fn radius_two_expected_point_set() {
        let pts = rasterize(0, 0, 2);
        let expected = as_set(&[
            (0, 2),
            (0, -2),
            (2, 0),
            (-2, 0),
            (1, 2),
            (-1, 2),
            (1, -2),
            (-1, -2),
            (2, 1),
            (-2, 1),
            (2, -1),
            (-2, -1),
        ]);
        assert_eq!(as_set(&pts), expected);
        assert_eq!(pts.len(), 12);
    }

    #[test]
    fn radius_three_point_count_is_sixteen() {
        assert_eq!(rasterize(0, 0, 3).len(), 16);
    }

    #[test]
    fn radius_five_point_count_is_twenty_eight() {
        assert_eq!(rasterize(0, 0, 5).len(), 28);
    }

    #[test]
    fn rasterize_output_is_sorted_ascending() {
        let pts = rasterize(0, 0, 7);
        for w in pts.windows(2) {
            assert!(w[0] < w[1], "not strictly ascending: {:?}", w);
        }
    }

    #[test]
    fn rasterize_output_has_no_duplicates() {
        for r in 0..40 {
            let pts = rasterize(0, 0, r);
            let set = as_set(&pts);
            assert_eq!(set.len(), pts.len(), "duplicate points at r = {r}");
        }
    }

    #[test]
    fn rasterize_is_deterministic() {
        for r in 0..25 {
            assert_eq!(rasterize(3, -4, r), rasterize(3, -4, r));
        }
    }

    #[test]
    fn every_point_is_near_the_true_circle() {
        for r in 1..200 {
            let r_sq = (r as i64) * (r as i64);
            let bound = 2 * (r as i64) + 1;
            for &(x, y) in &rasterize(0, 0, r) {
                let err = (dist_sq(x, y) - r_sq).abs();
                assert!(err <= bound, "r = {r}, point ({x},{y}), err = {err}");
            }
        }
    }

    #[test]
    fn rasterize_has_full_eight_way_symmetry() {
        let pts = rasterize(0, 0, 11);
        let set = as_set(&pts);
        for &(x, y) in &pts {
            for cand in [
                (x, y),
                (-x, y),
                (x, -y),
                (-x, -y),
                (y, x),
                (-y, x),
                (y, -x),
                (-y, -x),
            ] {
                assert!(set.contains(&cand), "missing symmetric point {:?}", cand);
            }
        }
    }

    #[test]
    fn rasterize_is_translation_invariant() {
        let base = rasterize(0, 0, 9);
        let shifted = rasterize(100, -50, 9);
        let mapped: BTreeSet<(i32, i32)> = base.iter().map(|&(x, y)| (x + 100, y - 50)).collect();
        assert_eq!(as_set(&shifted), mapped);
    }

    #[test]
    fn axis_points_are_present_for_each_radius() {
        for r in 1..30 {
            let set = as_set(&rasterize(0, 0, r));
            assert!(set.contains(&(r, 0)), "missing (+r,0) at r = {r}");
            assert!(set.contains(&(-r, 0)), "missing (-r,0) at r = {r}");
            assert!(set.contains(&(0, r)), "missing (0,+r) at r = {r}");
            assert!(set.contains(&(0, -r)), "missing (0,-r) at r = {r}");
        }
    }

    #[test]
    fn diagonal_point_collapses_to_four_reflections() {
        // r = 3 plots the diagonal octant point (2, 2); its four reflections
        // must all be present and there must be exactly four of them.
        let set = as_set(&rasterize(0, 0, 3));
        for cand in [(2, 2), (-2, 2), (2, -2), (-2, -2)] {
            assert!(
                set.contains(&cand),
                "missing diagonal reflection {:?}",
                cand
            );
        }
    }

    #[test]
    fn ring_has_no_radial_gaps() {
        // Every ring point must have at least one 8-connected neighbor also on
        // the ring, i.e. the circle is closed with no isolated pixels.
        for r in 2..60 {
            let pts = rasterize(0, 0, r);
            let set = as_set(&pts);
            for &(x, y) in &pts {
                let mut has_neighbor = false;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        if set.contains(&(x + dx, y + dy)) {
                            has_neighbor = true;
                        }
                    }
                }
                assert!(has_neighbor, "isolated point ({x},{y}) at r = {r}");
            }
        }
    }

    #[test]
    fn point_count_grows_within_reasonable_bounds() {
        for r in 1..100 {
            let n = rasterize(0, 0, r).len();
            // A rasterized circle of radius r has on the order of 8r/sqrt(2)
            // points; bound it loosely but meaningfully.
            assert!(n >= 4, "too few points at r = {r}: {n}");
            assert!(
                n as i64 <= 8 * (r as i64) + 8,
                "too many points at r = {r}: {n}"
            );
        }
    }

    #[test]
    fn point_count_is_monotone_nondecreasing() {
        let mut prev = 0usize;
        for r in 1..80 {
            let n = rasterize(0, 0, r).len();
            assert!(n >= prev, "count dropped from {prev} to {n} at r = {r}");
            prev = n;
        }
    }

    #[test]
    fn filled_disk_radius_zero_is_single_point() {
        assert_eq!(filled_disk(2, 2, 0), alloc::vec![(2, 2)]);
    }

    #[test]
    fn filled_disk_negative_radius_is_empty() {
        assert!(filled_disk(0, 0, -1).is_empty());
    }

    #[test]
    fn filled_disk_radius_one_is_plus_shape() {
        let set = as_set(&filled_disk(0, 0, 1));
        let expected = as_set(&[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]);
        assert_eq!(set, expected);
    }

    #[test]
    fn filled_disk_every_point_is_inside_disk() {
        for r in 0..80 {
            let r_sq = (r as i64) * (r as i64);
            for &(x, y) in &filled_disk(0, 0, r) {
                assert!(dist_sq(x, y) <= r_sq, "point ({x},{y}) outside r = {r}");
            }
        }
    }

    #[test]
    fn filled_disk_is_complete() {
        // Every lattice point satisfying x^2 + y^2 <= r^2 must be present.
        for r in 0..40 {
            let r_sq = (r as i64) * (r as i64);
            let set = as_set(&filled_disk(0, 0, r));
            for x in -r..=r {
                for y in -r..=r {
                    if dist_sq(x, y) <= r_sq {
                        assert!(set.contains(&(x, y)), "missing ({x},{y}) at r = {r}");
                    }
                }
            }
        }
    }

    #[test]
    fn filled_disk_has_no_duplicates() {
        for r in 0..50 {
            let pts = filled_disk(0, 0, r);
            assert_eq!(as_set(&pts).len(), pts.len(), "duplicate at r = {r}");
        }
    }

    #[test]
    fn filled_disk_rows_are_contiguous_spans() {
        // Within any scan row the x-coordinates must be a contiguous run.
        let pts = filled_disk(0, 0, 12);
        let mut by_row: alloc::collections::BTreeMap<i32, Vec<i32>> =
            alloc::collections::BTreeMap::new();
        for &(x, y) in &pts {
            by_row.entry(y).or_default().push(x);
        }
        for (_, mut xs) in by_row {
            xs.sort_unstable();
            for w in xs.windows(2) {
                assert_eq!(w[1] - w[0], 1, "gap in row span: {:?}", w);
            }
        }
    }

    #[test]
    fn filled_disk_is_symmetric() {
        let set = as_set(&filled_disk(0, 0, 15));
        for &(x, y) in set.iter() {
            assert!(set.contains(&(-x, y)));
            assert!(set.contains(&(x, -y)));
            assert!(set.contains(&(y, x)));
        }
    }

    #[test]
    fn filled_disk_is_translation_invariant() {
        let base = filled_disk(0, 0, 8);
        let shifted = filled_disk(-20, 30, 8);
        let mapped: BTreeSet<(i32, i32)> = base.iter().map(|&(x, y)| (x - 20, y + 30)).collect();
        assert_eq!(as_set(&shifted), mapped);
    }

    #[test]
    fn filled_disk_count_matches_manual_count() {
        for r in 0..30 {
            let r_sq = (r as i64) * (r as i64);
            let mut manual = 0usize;
            for x in -r..=r {
                for y in -r..=r {
                    if dist_sq(x, y) <= r_sq {
                        manual += 1;
                    }
                }
            }
            assert_eq!(filled_disk(0, 0, r).len(), manual, "count mismatch r = {r}");
        }
    }

    #[test]
    fn filled_disk_is_deterministic() {
        for r in 0..20 {
            assert_eq!(filled_disk(1, 1, r), filled_disk(1, 1, r));
        }
    }

    #[test]
    fn filled_disk_contains_center_for_positive_radius() {
        for r in 0..20 {
            assert!(as_set(&filled_disk(5, 6, r)).contains(&(5, 6)));
        }
    }

    #[test]
    fn boundary_points_are_within_filled_bounding_box() {
        // The rasterized ring never escapes the [-r, r] box around the center.
        for r in 1..60 {
            for &(x, y) in &rasterize(0, 0, r) {
                assert!(
                    x >= -r && x <= r && y >= -r && y <= r,
                    "escaped box at r = {r}"
                );
            }
        }
    }

    #[test]
    fn large_radius_rasterize_is_well_formed() {
        let r = 1000;
        let pts = rasterize(0, 0, r);
        let r_sq = (r as i64) * (r as i64);
        let bound = 2 * (r as i64) + 1;
        assert!(!pts.is_empty());
        for &(x, y) in &pts {
            assert!((dist_sq(x, y) - r_sq).abs() <= bound);
        }
    }
}
