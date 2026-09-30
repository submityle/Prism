//! Integer Bresenham line rasterization for the particle debug / trail
//! contracts (design §12-§13, §29).
//!
//! Several particle stages need to walk the *discrete grid cells a straight
//! segment crosses*: a debug-draw overlay wants the pixels of a velocity or
//! bounds gizmo; a screen-space trail wants the cells a spark sweeps between two
//! frames; and a deterministic ribbon rasterizer wants the exact same integer
//! cell walk on the `CPU` reference path that the `GPU` kernel will later
//! reproduce bit for bit. This module owns the small, purely-integer contract
//! those stages share: turning a pair of integer endpoints into the ordered
//! list of grid cells the segment passes through, endpoints included.
//!
//! The walk is Bresenham's classic error-term line algorithm, generalized to
//! cover all eight octants at once. It keeps a single integer error accumulator
//! `err = dx - dy` and, each iteration, steps the *dominant* axis by one while
//! stepping the *minor* axis only when the accumulated error crosses zero. No
//! division, no floating point, and no transcendental math is involved: the
//! entire walk is `+`, `-`, `*2`, comparison, and integer sign.
//!
//! # Determinism under reversal
//! Callers rely on a segment and its reverse producing the *same* set of cells
//! in opposite order (a forward trail and its rewind must overlap exactly). The
//! naive error algorithm is not symmetric under endpoint swap because ties break
//! toward the start point. To make [`rasterize`] symmetric *by construction*,
//! the walk is always computed from the lexicographically smaller endpoint and
//! the result is reversed when the caller asked for the opposite direction.
//! Thus `rasterize(a, b)` is exactly the reverse of `rasterize(b, a)` for every
//! pair of endpoints.
//!
//! # Guaranteed properties
//! For any endpoints the returned walk (see the tests):
//! * includes both endpoints as its first and last cell,
//! * has exactly [`step_count`] cells, i.e. `max(|dx|, |dy|) + 1`,
//! * is 8-connected (consecutive cells touch, sharing an edge or a corner),
//! * advances the dominant axis by exactly one cell per step and never revisits
//!   a cell, and
//! * is the exact reverse of the walk between the swapped endpoints.
//!
//! # Strict scope
//! This module only rasterizes a *single straight segment* into grid cells. It
//! deliberately does not draw thick lines, anti-aliased (`Wu`) lines, circles,
//! polygons, or filled shapes, and it neither imports nor mutates any sibling
//! module. Its arithmetic is self-contained integer math widened to `i64` so a
//! segment spanning the full `i32` range can never overflow the error term.

use alloc::vec::Vec;

/// The step direction along one axis: `-1`, `0`, or `+1`.
///
/// A zero step means the two endpoints share that coordinate (a perfectly
/// horizontal or vertical segment on that axis).
#[inline]
fn axis_step(from: i64, to: i64) -> i64 {
    if from < to {
        1
    } else if from > to {
        -1
    } else {
        0
    }
}

/// Number of grid cells [`rasterize`] returns for the given endpoints:
/// `max(|dx|, |dy|) + 1`, where `dx = x1 - x0` and `dy = y1 - y0`.
///
/// This is the count of cells along the *dominant* axis (the longer of the two
/// spans), plus one because both endpoints are inclusive. It is symmetric in
/// its arguments — swapping the endpoints does not change the count — and is
/// exactly `1` for a degenerate single-point segment.
///
/// The differences are computed in `i64` so the full `i32` coordinate range
/// cannot overflow.
#[must_use]
pub fn step_count(x0: i32, y0: i32, x1: i32, y1: i32) -> usize {
    let dx = (i64::from(x1) - i64::from(x0)).unsigned_abs();
    let dy = (i64::from(y1) - i64::from(y0)).unsigned_abs();
    // `max(dx, dy)` fits in `u64`; `+ 1` cannot overflow because the maximum
    // possible span between two `i32` values is far below `u64::MAX`.
    (dx.max(dy) + 1) as usize
}

/// Rasterizes the straight segment from `(x0, y0)` to `(x1, y1)` into the
/// ordered list of grid cells it passes through, endpoints included.
///
/// The result covers all eight octants (horizontal, vertical, both diagonals,
/// and every shallow / steep slope of either sign), starts at `(x0, y0)`, ends
/// at `(x1, y1)`, contains exactly [`step_count`] cells, and is 8-connected.
/// A degenerate segment whose endpoints coincide yields a single cell.
///
/// `rasterize(a, b)` is the exact reverse of `rasterize(b, a)`.
#[must_use]
pub fn rasterize(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    // Compute the walk from the lexicographically smaller endpoint so the
    // tie-breaking is orientation-independent, then flip for the caller's
    // direction. This makes the forward and reverse walks exact mirrors.
    if (x0, y0) <= (x1, y1) {
        walk(x0, y0, x1, y1)
    } else {
        let mut cells = walk(x1, y1, x0, y0);
        cells.reverse();
        cells
    }
}

/// The raw Bresenham walk from `(x0, y0)` toward `(x1, y1)`.
///
/// All arithmetic is widened to `i64` so `2 * err`, `dx - dy`, and the per-step
/// coordinate advance cannot overflow for any `i32` endpoints. The dominant
/// axis advances by exactly one cell on every iteration, so the loop runs
/// [`step_count`] times and terminates exactly on the endpoint.
fn walk(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let xe = i64::from(x1);
    let ye = i64::from(y1);
    let mut x = i64::from(x0);
    let mut y = i64::from(y0);

    let dx = (xe - x).abs();
    let dy = (ye - y).abs();
    let sx = axis_step(x, xe);
    let sy = axis_step(y, ye);

    let mut err = dx - dy;
    let mut cells = Vec::with_capacity(step_count(x0, y0, x1, y1));

    loop {
        cells.push((x as i32, y as i32));
        if x == xe && y == ye {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }

    cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// True when consecutive cells are 8-neighbors: each axis differs by at
    /// most one and the pair is not identical.
    fn is_8_connected(cells: &[(i32, i32)]) -> bool {
        cells.windows(2).all(|w| {
            let (ax, ay) = w[0];
            let (bx, by) = w[1];
            let dx = (bx - ax).abs();
            let dy = (by - ay).abs();
            dx <= 1 && dy <= 1 && (dx != 0 || dy != 0)
        })
    }

    /// True when no cell repeats.
    fn all_distinct(cells: &[(i32, i32)]) -> bool {
        for i in 0..cells.len() {
            for j in (i + 1)..cells.len() {
                if cells[i] == cells[j] {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn single_point_yields_one_cell() {
        assert_eq!(rasterize(4, -7, 4, -7), vec![(4, -7)]);
        assert_eq!(rasterize(0, 0, 0, 0), vec![(0, 0)]);
    }

    #[test]
    fn step_count_single_point_is_one() {
        assert_eq!(step_count(4, -7, 4, -7), 1);
        assert_eq!(step_count(0, 0, 0, 0), 1);
    }

    #[test]
    fn horizontal_positive_is_exact_run() {
        assert_eq!(
            rasterize(0, 3, 4, 3),
            vec![(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)]
        );
    }

    #[test]
    fn horizontal_negative_is_exact_run() {
        assert_eq!(
            rasterize(4, 3, 0, 3),
            vec![(4, 3), (3, 3), (2, 3), (1, 3), (0, 3)]
        );
    }

    #[test]
    fn vertical_positive_is_exact_run() {
        assert_eq!(
            rasterize(-2, 0, -2, 4),
            vec![(-2, 0), (-2, 1), (-2, 2), (-2, 3), (-2, 4)]
        );
    }

    #[test]
    fn vertical_negative_is_exact_run() {
        assert_eq!(
            rasterize(-2, 4, -2, 0),
            vec![(-2, 4), (-2, 3), (-2, 2), (-2, 1), (-2, 0)]
        );
    }

    #[test]
    fn diagonal_pos_pos_is_pure_diagonal() {
        assert_eq!(rasterize(0, 0, 3, 3), vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
    }

    #[test]
    fn diagonal_pos_neg_is_pure_diagonal() {
        assert_eq!(
            rasterize(0, 0, 3, -3),
            vec![(0, 0), (1, -1), (2, -2), (3, -3)]
        );
    }

    #[test]
    fn diagonal_neg_pos_is_pure_diagonal() {
        assert_eq!(
            rasterize(0, 0, -3, 3),
            vec![(0, 0), (-1, 1), (-2, 2), (-3, 3)]
        );
    }

    #[test]
    fn diagonal_neg_neg_is_pure_diagonal() {
        assert_eq!(
            rasterize(0, 0, -3, -3),
            vec![(0, 0), (-1, -1), (-2, -2), (-3, -3)]
        );
    }

    #[test]
    fn octant0_shallow_pos_pos_known_line() {
        // dx=6, dy=2: dominant x. Classic 6:2 Bresenham staircase.
        assert_eq!(
            rasterize(0, 0, 6, 2),
            vec![(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 2), (6, 2)]
        );
    }

    #[test]
    fn octant1_steep_pos_pos_known_line() {
        // dx=2, dy=6: dominant y — the transpose of the octant-0 case.
        assert_eq!(
            rasterize(0, 0, 2, 6),
            vec![(0, 0), (0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        );
    }

    #[test]
    fn octant2_steep_neg_pos() {
        let cells = rasterize(0, 0, -2, 6);
        assert_eq!(cells.first(), Some(&(0, 0)));
        assert_eq!(cells.last(), Some(&(-2, 6)));
        assert_eq!(cells.len(), 7);
        assert!(is_8_connected(&cells));
    }

    #[test]
    fn octant3_shallow_neg_pos() {
        let cells = rasterize(0, 0, -6, 2);
        assert_eq!(cells.first(), Some(&(0, 0)));
        assert_eq!(cells.last(), Some(&(-6, 2)));
        assert_eq!(cells.len(), 7);
        assert!(is_8_connected(&cells));
    }

    #[test]
    fn octant4_shallow_neg_neg() {
        let cells = rasterize(0, 0, -6, -2);
        assert_eq!(cells.first(), Some(&(0, 0)));
        assert_eq!(cells.last(), Some(&(-6, -2)));
        assert_eq!(cells.len(), 7);
        assert!(is_8_connected(&cells));
    }

    #[test]
    fn octant5_steep_neg_neg() {
        let cells = rasterize(0, 0, -2, -6);
        assert_eq!(cells.first(), Some(&(0, 0)));
        assert_eq!(cells.last(), Some(&(-2, -6)));
        assert_eq!(cells.len(), 7);
        assert!(is_8_connected(&cells));
    }

    #[test]
    fn octant6_steep_pos_neg() {
        let cells = rasterize(0, 0, 2, -6);
        assert_eq!(cells.first(), Some(&(0, 0)));
        assert_eq!(cells.last(), Some(&(2, -6)));
        assert_eq!(cells.len(), 7);
        assert!(is_8_connected(&cells));
    }

    #[test]
    fn octant7_shallow_pos_neg() {
        let cells = rasterize(0, 0, 6, -2);
        assert_eq!(cells.first(), Some(&(0, 0)));
        assert_eq!(cells.last(), Some(&(6, -2)));
        assert_eq!(cells.len(), 7);
        assert!(is_8_connected(&cells));
    }

    #[test]
    fn endpoints_are_always_included() {
        let pairs = [
            (0, 0, 10, 3),
            (-5, 2, 7, -8),
            (3, 3, 3, 3),
            (100, -40, -100, 40),
            (1, 2, 1, 99),
        ];
        for (x0, y0, x1, y1) in pairs {
            let cells = rasterize(x0, y0, x1, y1);
            assert_eq!(cells.first(), Some(&(x0, y0)));
            assert_eq!(cells.last(), Some(&(x1, y1)));
        }
    }

    #[test]
    fn cell_count_matches_step_count() {
        let pairs = [
            (0, 0, 10, 3),
            (-5, 2, 7, -8),
            (3, 3, 3, 3),
            (100, -40, -100, 40),
            (1, 2, 1, 99),
            (-13, -13, 13, 13),
            (0, 0, 17, 5),
            (0, 0, 5, 17),
        ];
        for (x0, y0, x1, y1) in pairs {
            let cells = rasterize(x0, y0, x1, y1);
            assert_eq!(cells.len(), step_count(x0, y0, x1, y1));
        }
    }

    #[test]
    fn step_count_is_max_span_plus_one() {
        assert_eq!(step_count(0, 0, 6, 2), 7);
        assert_eq!(step_count(0, 0, 2, 6), 7);
        assert_eq!(step_count(-3, -3, 3, 3), 7);
        assert_eq!(step_count(0, 0, 100, 0), 101);
        assert_eq!(step_count(0, 0, 0, 100), 101);
    }

    #[test]
    fn step_count_is_order_independent() {
        assert_eq!(step_count(0, 0, 6, 2), step_count(6, 2, 0, 0));
        assert_eq!(step_count(-5, 2, 7, -8), step_count(7, -8, -5, 2));
    }

    #[test]
    fn walk_is_8_connected_over_many_lines() {
        for x1 in -8..=8 {
            for y1 in -8..=8 {
                let cells = rasterize(0, 0, x1, y1);
                assert!(
                    is_8_connected(&cells),
                    "not 8-connected for endpoint ({x1}, {y1})"
                );
            }
        }
    }

    #[test]
    fn walk_has_no_duplicate_cells() {
        for x1 in -6..=6 {
            for y1 in -6..=6 {
                let cells = rasterize(0, 0, x1, y1);
                assert!(
                    all_distinct(&cells),
                    "duplicate cell for endpoint ({x1}, {y1})"
                );
            }
        }
    }

    #[test]
    fn reverse_endpoints_yield_reversed_cells() {
        for x1 in -8..=8 {
            for y1 in -8..=8 {
                let forward = rasterize(0, 0, x1, y1);
                let mut backward = rasterize(x1, y1, 0, 0);
                backward.reverse();
                assert_eq!(
                    forward, backward,
                    "reversal mismatch for endpoint ({x1}, {y1})"
                );
            }
        }
    }

    #[test]
    fn count_matches_step_count_over_grid() {
        for x1 in -9..=9 {
            for y1 in -9..=9 {
                let cells = rasterize(0, 0, x1, y1);
                assert_eq!(cells.len(), step_count(0, 0, x1, y1));
            }
        }
    }

    #[test]
    fn dominant_axis_advances_by_exactly_one_each_step() {
        // For a shallow line the x-span is dominant, so |Δx| == 1 every step
        // and |Δy| is 0 or 1.
        let cells = rasterize(0, 0, 20, 7);
        for w in cells.windows(2) {
            let dx = (w[1].0 - w[0].0).abs();
            let dy = (w[1].1 - w[0].1).abs();
            assert_eq!(dx, 1);
            assert!(dy <= 1);
        }
    }

    #[test]
    fn dominant_axis_advances_for_steep_line() {
        // For a steep line the y-span is dominant, so |Δy| == 1 every step.
        let cells = rasterize(0, 0, 7, 20);
        for w in cells.windows(2) {
            let dx = (w[1].0 - w[0].0).abs();
            let dy = (w[1].1 - w[0].1).abs();
            assert_eq!(dy, 1);
            assert!(dx <= 1);
        }
    }

    #[test]
    fn shallow_line_x_is_strictly_monotonic() {
        let cells = rasterize(-5, -2, 12, 3);
        for w in cells.windows(2) {
            assert!(w[1].0 > w[0].0, "x should strictly increase");
            assert!(w[1].1 >= w[0].1, "y should be non-decreasing");
        }
    }

    #[test]
    fn translation_shifts_every_cell() {
        let base = rasterize(0, 0, 9, 4);
        let shifted = rasterize(100, -50, 109, -46);
        assert_eq!(base.len(), shifted.len());
        for (b, s) in base.iter().zip(shifted.iter()) {
            assert_eq!(s.0 - b.0, 100);
            assert_eq!(s.1 - b.1, -50);
        }
    }

    #[test]
    fn long_line_endpoints_and_count() {
        let cells = rasterize(-1000, -3, 1000, 7);
        assert_eq!(cells.first(), Some(&(-1000, -3)));
        assert_eq!(cells.last(), Some(&(1000, 7)));
        assert_eq!(cells.len(), 2001);
        assert!(is_8_connected(&cells));
    }

    #[test]
    fn symmetric_diagonal_is_its_own_transpose() {
        // The (dx == dy) diagonal is the fixed point of the shallow/steep
        // symmetry: transposing coordinates yields the transposed walk.
        let cells = rasterize(0, 0, 5, 5);
        let transposed: Vec<(i32, i32)> =
            rasterize(0, 0, 5, 5).iter().map(|&(x, y)| (y, x)).collect();
        assert_eq!(cells, transposed);
    }

    #[test]
    fn negative_quadrant_shallow_known_line() {
        // Mirror of the octant-0 staircase into the (-x, -y) quadrant.
        assert_eq!(
            rasterize(0, 0, -6, -2),
            vec![
                (0, 0),
                (-1, 0),
                (-2, -1),
                (-3, -1),
                (-4, -1),
                (-5, -2),
                (-6, -2)
            ]
        );
    }

    #[test]
    fn all_octant_endpoints_reach_target_exactly() {
        let targets = [
            (6, 2),
            (2, 6),
            (-2, 6),
            (-6, 2),
            (-6, -2),
            (-2, -6),
            (2, -6),
            (6, -2),
        ];
        for (x1, y1) in targets {
            let cells = rasterize(0, 0, x1, y1);
            assert_eq!(cells.last(), Some(&(x1, y1)));
            assert_eq!(cells.first(), Some(&(0, 0)));
            assert_eq!(cells.len(), step_count(0, 0, x1, y1));
            assert!(is_8_connected(&cells));
        }
    }

    #[test]
    fn one_step_neighbors_have_two_cells() {
        // Any endpoint one cell away yields exactly the two endpoints.
        assert_eq!(rasterize(0, 0, 1, 0), vec![(0, 0), (1, 0)]);
        assert_eq!(rasterize(0, 0, 0, 1), vec![(0, 0), (0, 1)]);
        assert_eq!(rasterize(0, 0, 1, 1), vec![(0, 0), (1, 1)]);
        assert_eq!(rasterize(0, 0, -1, -1), vec![(0, 0), (-1, -1)]);
    }
}
