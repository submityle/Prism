//! Xiaolin Wu anti-aliased line rasterization for the particle debug / trail
//! and soft-ribbon contracts (design §12-§13, §29).
//!
//! Where [`crate::particle::bresenham_line`] answers "which integer grid cells
//! does a segment cross", several particle stages instead need "how much of
//! each pixel does a thin bright streak cover". A spark trail, a velocity
//! gizmo drawn with sub-pixel smoothness, a soft ribbon edge, and the `CPU`
//! reference for a smooth-line `GPU` kernel all want *fractional* pixel
//! coverage so the streak reads as a clean anti-aliased line instead of a
//! jagged staircase. This module owns that contract: it turns a pair of
//! floating-point endpoints into an ordered list of `(x, y, coverage)` samples,
//! where `coverage` in `[0, 1]` is the fraction of the pixel the line paints.
//!
//! # Algorithm
//! The rasterizer is Xiaolin Wu's classic anti-aliased line algorithm. Walking
//! the *major* (longer) axis one integer step at a time, the line's exact
//! position along the *minor* axis lands between two pixel rows/columns; the
//! fractional part splits full brightness between the nearer pixel (weight
//! `1 - frac`) and its neighbor (weight `frac`), so the two coverages a single
//! minor step emits always sum to one. The endpoints are attenuated by an
//! `x-gap` term so a segment that starts or ends mid-pixel does not paint a
//! full-brightness cap.
//!
//! # Steep lines
//! When the segment rises faster than it runs (`|dy| > |dx|`) the roles of the
//! two axes are swapped so the walk is always along the longer axis; the
//! emitted coordinates are swapped back into the caller's frame. This keeps the
//! minor-axis gradient in `[-1, 1]` and guarantees exactly two candidate pixels
//! per major step for both shallow and steep lines.
//!
//! # Hand-rolled math only
//! The whole routine uses nothing beyond `+`, `-`, `*`, `/`, `f32::floor`,
//! `f32::abs`, and `f32::clamp`. There is no `sin`/`cos`, no `exp`/`ln`, no
//! `round`, and no `ceil`: rounding to the nearest integer is spelled
//! `(v + 0.5).floor()`, the integer part is `v.floor()`, and the fractional
//! part is `v - v.floor()`. Floating-point values are never compared with `==`
//! or `!=`; near-zero tests go through explicit epsilon constants.
//!
//! # Guaranteed properties
//! For any endpoints the returned samples (see the tests):
//! * carry `coverage` clamped to `[0, 1]`, with zero-coverage pixels omitted,
//! * split every interior major step into two pixels whose coverages sum to
//!   one (up to rounding),
//! * reduce to a single covered pixel per step for axis-aligned lines, giving
//!   exact `1.0` coverage along the body of a horizontal or vertical run,
//! * include the two endpoint pixels (attenuated by their sub-pixel gap), and
//! * are identical (as a set) when the endpoints are swapped, and identical
//!   run-to-run for identical input (determinism).
//!
//! # Strict scope
//! This module only rasterizes a *single anti-aliased straight segment*. It is
//! deliberately distinct from the integer Bresenham walk in
//! [`crate::particle::bresenham_line`] and draws no thick lines, curves,
//! circles, or filled shapes. It neither imports nor mutates any sibling
//! module; all state lives in local values.

use alloc::vec::Vec;

/// Coverage at or below this magnitude is treated as "nothing painted" and the
/// sample is dropped from the output. Well under a single 8-bit intensity step
/// (`1/255`), so it never discards a visible pixel.
const COVERAGE_EPS: f32 = 1.0e-6;

/// Span (after axis canonicalization) at or below this magnitude marks a
/// degenerate segment whose endpoints coincide; the gradient is then defined as
/// `1.0` purely to keep the arithmetic finite (the main loop stays empty).
const DEGENERATE_EPS: f32 = 1.0e-6;

/// Integer part of `v` as an `f32` (its floor).
#[inline]
fn ipart(v: f32) -> f32 {
    v.floor()
}

/// Fractional part of `v` in `[0, 1)`, computed as `v - floor(v)` so it is
/// correct for negative inputs (e.g. `fpart(-0.25) == 0.75`).
#[inline]
fn fpart(v: f32) -> f32 {
    v - v.floor()
}

/// Reverse fractional part, `1 - fpart(v)`, in `(0, 1]`.
#[inline]
fn rfpart(v: f32) -> f32 {
    1.0 - fpart(v)
}

/// Rounds `v` to the nearest integer (ties toward `+∞`) as an `f32`, spelled
/// with `floor` so no banned `round`/`ceil` is used.
#[inline]
fn round_nearest(v: f32) -> f32 {
    (v + 0.5).floor()
}

/// Pushes one anti-aliased sample, mapping the internal `(major, minor)` walk
/// frame back into the caller's `(x, y)` frame and dropping near-zero coverage.
///
/// In the shallow (`steep == false`) case the major axis is `x`; in the steep
/// case the axes were swapped up front, so the internal minor coordinate is the
/// caller's `x` and the internal major coordinate is the caller's `y`.
#[inline]
fn emit(out: &mut Vec<(i32, i32, f32)>, steep: bool, major: i32, minor: i32, coverage: f32) {
    let c = coverage.clamp(0.0, 1.0);
    if c <= COVERAGE_EPS {
        return;
    }
    if steep {
        out.push((minor, major, c));
    } else {
        out.push((major, minor, c));
    }
}

/// Rasterizes the anti-aliased straight segment from `(x0, y0)` to `(x1, y1)`
/// into an ordered list of `(x, y, coverage)` samples.
///
/// `coverage` lies in `[0, 1]` and is the fraction of the pixel painted by the
/// line. Interior major-axis steps contribute two samples whose coverages sum
/// to one; axis-aligned lines collapse to a single full-coverage pixel per step
/// along their body. The two endpoints are attenuated by their sub-pixel gap so
/// a segment beginning or ending mid-pixel does not paint a hard cap. Samples
/// with (clamped) coverage at or below [`COVERAGE_EPS`] are omitted.
///
/// The result is deterministic and independent of endpoint order: swapping the
/// two endpoints yields the same set of samples.
#[must_use]
pub fn rasterize(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<(i32, i32, f32)> {
    let mut out: Vec<(i32, i32, f32)> = Vec::new();

    let (mut ax, mut ay, mut bx, mut by) = (x0, y0, x1, y1);

    // Walk the longer axis: swap x/y when the line is steeper than 45°.
    let steep = (by - ay).abs() > (bx - ax).abs();
    if steep {
        core::mem::swap(&mut ax, &mut ay);
        core::mem::swap(&mut bx, &mut by);
    }
    // Always march left-to-right along the (now dominant) x axis.
    if ax > bx {
        core::mem::swap(&mut ax, &mut bx);
        core::mem::swap(&mut ay, &mut by);
    }

    let dx = bx - ax;
    let dy = by - ay;
    // After canonicalization |dy| <= |dx|, so the gradient magnitude is <= 1.
    // Only a coincident-endpoint segment has a ~zero run; define its gradient
    // as 1.0 to keep the math finite (the interior loop stays empty anyway).
    let gradient = if dx.abs() <= DEGENERATE_EPS {
        1.0
    } else {
        dy / dx
    };

    // First endpoint.
    let xend0 = round_nearest(ax);
    let yend0 = ay + gradient * (xend0 - ax);
    let xgap0 = rfpart(ax + 0.5);
    let xpxl1 = xend0 as i32;
    let ypxl1 = ipart(yend0) as i32;
    let cov0_near = rfpart(yend0) * xgap0;
    let cov0_far = fpart(yend0) * xgap0;

    // Second endpoint.
    let xend1 = round_nearest(bx);
    let yend1 = by + gradient * (xend1 - bx);
    let xgap1 = fpart(bx + 0.5);
    let xpxl2 = xend1 as i32;
    let ypxl2 = ipart(yend1) as i32;
    let cov1_near = rfpart(yend1) * xgap1;
    let cov1_far = fpart(yend1) * xgap1;

    if xpxl1 == xpxl2 {
        // The segment is shorter than a pixel along the major axis: both
        // endpoints land in the same column. Merge their (up to four) pixel
        // contributions by minor coordinate so no coordinate repeats.
        let mut acc: Vec<(i32, f32)> = Vec::new();
        let mut add = |minor: i32, coverage: f32| {
            for entry in acc.iter_mut() {
                if entry.0 == minor {
                    entry.1 += coverage;
                    return;
                }
            }
            acc.push((minor, coverage));
        };
        add(ypxl1, cov0_near);
        add(ypxl1 + 1, cov0_far);
        add(ypxl2, cov1_near);
        add(ypxl2 + 1, cov1_far);
        for (minor, coverage) in acc {
            emit(&mut out, steep, xpxl1, minor, coverage);
        }
        return out;
    }

    // First endpoint column.
    emit(&mut out, steep, xpxl1, ypxl1, cov0_near);
    emit(&mut out, steep, xpxl1, ypxl1 + 1, cov0_far);

    // Interior: one major step per column, two anti-aliased pixels each.
    let mut intery = yend0 + gradient;
    let mut x = xpxl1 + 1;
    while x < xpxl2 {
        let base = ipart(intery) as i32;
        emit(&mut out, steep, x, base, rfpart(intery));
        emit(&mut out, steep, x, base + 1, fpart(intery));
        intery += gradient;
        x += 1;
    }

    // Second endpoint column.
    emit(&mut out, steep, xpxl2, ypxl2, cov1_near);
    emit(&mut out, steep, xpxl2, ypxl2 + 1, cov1_far);

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparing coverages and gradients.
    const EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    /// Sums coverage per major-axis coordinate. `major_is_x` selects which
    /// coordinate is the dominant walk axis (`x` for shallow, `y` for steep).
    fn column_sums(pixels: &[(i32, i32, f32)], major_is_x: bool) -> Vec<(i32, f32)> {
        let mut sums: Vec<(i32, f32)> = Vec::new();
        for &(x, y, c) in pixels {
            let major = if major_is_x { x } else { y };
            if let Some(entry) = sums.iter_mut().find(|e| e.0 == major) {
                entry.1 += c;
            } else {
                sums.push((major, c));
            }
        }
        sums
    }

    fn sorted(mut pixels: Vec<(i32, i32, f32)>) -> Vec<(i32, i32, f32)> {
        pixels.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        pixels
    }

    #[test]
    fn horizontal_interior_coverage_is_exactly_one() {
        let px = rasterize(0.0, 0.0, 5.0, 0.0);
        // Interior columns x = 1..=4 must be a single pixel at y = 0, cov 1.
        for x in 1..=4 {
            let hits: Vec<_> = px.iter().filter(|p| p.0 == x).collect();
            assert_eq!(hits.len(), 1, "column {x} should have one pixel");
            assert_eq!(hits[0].1, 0);
            assert!(approx(hits[0].2, 1.0), "coverage {}", hits[0].2);
        }
    }

    #[test]
    fn horizontal_endpoints_are_attenuated() {
        let px = rasterize(0.0, 0.0, 5.0, 0.0);
        let first = px.iter().find(|p| p.0 == 0).unwrap();
        let last = px.iter().find(|p| p.0 == 5).unwrap();
        assert!(approx(first.2, 0.5), "start coverage {}", first.2);
        assert!(approx(last.2, 0.5), "end coverage {}", last.2);
    }

    #[test]
    fn horizontal_pixel_count_and_row() {
        let px = rasterize(0.0, 0.0, 5.0, 0.0);
        // Six columns (0..=5), each a single non-zero pixel on row 0.
        assert_eq!(px.len(), 6);
        assert!(px.iter().all(|p| p.1 == 0));
    }

    #[test]
    fn vertical_interior_coverage_is_exactly_one() {
        let px = rasterize(5.0, 0.0, 5.0, 3.0);
        for y in 1..=2 {
            let hits: Vec<_> = px.iter().filter(|p| p.1 == y).collect();
            assert_eq!(hits.len(), 1, "row {y} should have one pixel");
            assert_eq!(hits[0].0, 5);
            assert!(approx(hits[0].2, 1.0));
        }
    }

    #[test]
    fn vertical_line_x_is_constant() {
        let px = rasterize(5.0, 0.0, 5.0, 8.0);
        assert!(px.iter().all(|p| p.0 == 5), "all samples share x = 5");
        // Endpoints at y = 0 and y = 8 are present.
        assert!(px.iter().any(|p| p.1 == 0));
        assert!(px.iter().any(|p| p.1 == 8));
    }

    #[test]
    fn diagonal_45_paints_only_the_diagonal() {
        let px = rasterize(0.0, 0.0, 5.0, 5.0);
        assert!(px.iter().all(|p| p.0 == p.1), "every pixel is on x == y");
        // Interior diagonal pixels are full coverage.
        for k in 1..=4 {
            let hit = px.iter().find(|p| p.0 == k).unwrap();
            assert!(approx(hit.2, 1.0), "diag ({k}) coverage {}", hit.2);
        }
    }

    #[test]
    fn diagonal_45_endpoints_attenuated() {
        let px = rasterize(0.0, 0.0, 5.0, 5.0);
        let first = px.iter().find(|p| p.0 == 0).unwrap();
        let last = px.iter().find(|p| p.0 == 5).unwrap();
        assert!(approx(first.2, 0.5));
        assert!(approx(last.2, 0.5));
    }

    #[test]
    fn coverage_is_within_unit_interval_over_many_lines() {
        for x1 in -6..=6 {
            for y1 in -6..=6 {
                let px = rasterize(0.3, -0.2, x1 as f32 + 0.1, y1 as f32 - 0.4);
                for p in &px {
                    assert!(p.2 >= 0.0 && p.2 <= 1.0, "coverage {} out of range", p.2);
                }
            }
        }
    }

    #[test]
    fn coverage_is_never_negative() {
        let px = rasterize(-3.7, 2.1, 9.4, -5.9);
        assert!(px.iter().all(|p| p.2 >= 0.0));
    }

    #[test]
    fn no_pixel_exceeds_full_coverage() {
        let px = rasterize(-3.7, 2.1, 9.4, -5.9);
        assert!(px.iter().all(|p| p.2 <= 1.0));
    }

    #[test]
    fn shallow_interior_columns_sum_to_one() {
        let px = rasterize(0.0, 0.0, 10.0, 3.0);
        let sums = column_sums(&px, true);
        // Interior columns are x = 1..=9 (exclude endpoints 0 and 10).
        for (major, s) in sums {
            if major > 0 && major < 10 {
                assert!(approx(s, 1.0), "column {major} sum {s}");
            }
        }
    }

    #[test]
    fn steep_interior_rows_sum_to_one() {
        let px = rasterize(0.0, 0.0, 3.0, 10.0);
        // Steep line: the major axis is y.
        let sums = column_sums(&px, false);
        for (major, s) in sums {
            if major > 0 && major < 10 {
                assert!(approx(s, 1.0), "row {major} sum {s}");
            }
        }
    }

    #[test]
    fn steep_line_major_axis_is_y() {
        let px = rasterize(0.0, 0.0, 3.0, 12.0);
        // One major step per y row: y spans 0..=12 => 13 distinct rows.
        let mut rows: Vec<i32> = px.iter().map(|p| p.1).collect();
        rows.sort_unstable();
        rows.dedup();
        assert_eq!(rows.len(), 13);
    }

    #[test]
    fn shallow_line_major_axis_is_x() {
        let px = rasterize(0.0, 0.0, 12.0, 3.0);
        let mut cols: Vec<i32> = px.iter().map(|p| p.0).collect();
        cols.sort_unstable();
        cols.dedup();
        assert_eq!(cols.len(), 13);
    }

    #[test]
    fn steep_is_transpose_of_shallow() {
        // Rasterizing the transposed endpoints must yield transposed samples.
        let shallow = sorted(rasterize(0.0, 0.0, 10.0, 4.0));
        let steep = rasterize(0.0, 0.0, 4.0, 10.0);
        let steep_t = sorted(steep.iter().map(|&(x, y, c)| (y, x, c)).collect());
        assert_eq!(shallow.len(), steep_t.len());
        for (a, b) in shallow.iter().zip(steep_t.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!(approx(a.2, b.2), "coverage {} vs {}", a.2, b.2);
        }
    }

    #[test]
    fn endpoints_are_present() {
        let px = rasterize(1.0, 1.0, 9.0, 4.0);
        assert!(
            px.iter().any(|p| p.0 == 1 && p.1 == 1),
            "start pixel present"
        );
        assert!(px.iter().any(|p| p.0 == 9 && p.1 == 4), "end pixel present");
    }

    #[test]
    fn swapping_endpoints_yields_same_sample_set() {
        let forward = sorted(rasterize(1.5, -2.0, 11.5, 3.0));
        let backward = sorted(rasterize(11.5, 3.0, 1.5, -2.0));
        assert_eq!(forward.len(), backward.len());
        for (a, b) in forward.iter().zip(backward.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!(approx(a.2, b.2), "coverage {} vs {}", a.2, b.2);
        }
    }

    #[test]
    fn swapping_endpoints_steep_line() {
        let forward = sorted(rasterize(-2.0, 1.5, 3.0, 11.5));
        let backward = sorted(rasterize(3.0, 11.5, -2.0, 1.5));
        assert_eq!(forward, backward);
    }

    #[test]
    fn output_is_deterministic() {
        let a = rasterize(0.25, -1.75, 7.5, 4.5);
        let b = rasterize(0.25, -1.75, 7.5, 4.5);
        assert_eq!(a, b);
    }

    #[test]
    fn single_point_is_covered() {
        let px = rasterize(2.0, 3.0, 2.0, 3.0);
        assert!(!px.is_empty(), "a degenerate point still paints something");
        // All coverage is concentrated at the single pixel column.
        let total: f32 = px.iter().map(|p| p.2).sum();
        assert!(approx(total, 1.0), "point total coverage {total}");
    }

    #[test]
    fn subpixel_segment_stays_in_one_column() {
        // Shorter than a pixel along x: both endpoints round to the same column.
        let px = rasterize(0.1, 0.0, 0.3, 0.0);
        let cols: Vec<i32> = {
            let mut c: Vec<i32> = px.iter().map(|p| p.0).collect();
            c.sort_unstable();
            c.dedup();
            c
        };
        assert_eq!(cols.len(), 1, "subpixel run collapses to a single column");
    }

    #[test]
    fn fractional_start_point_shifts_gap() {
        // Starting exactly on a pixel boundary vs mid-pixel changes the first
        // endpoint's rounded column.
        let flush = rasterize(0.0, 0.0, 6.0, 0.0);
        let mid = rasterize(0.5, 0.0, 6.0, 0.0);
        let flush_first = flush.iter().find(|p| p.0 == 0).map(|p| p.2);
        // At x0 = 0.5, round_nearest(0.5) = 1, so the first column is 1, not 0.
        let mid_zero = mid.iter().find(|p| p.0 == 0);
        assert!(flush_first.is_some());
        assert!(
            mid_zero.is_none(),
            "mid-pixel start does not paint column 0"
        );
    }

    #[test]
    fn gradient_splits_between_two_rows() {
        // A gentle slope must produce at least one column with two partial
        // pixels whose coverages are both strictly inside (0, 1).
        let px = rasterize(0.0, 0.0, 8.0, 3.0);
        let mut found_split = false;
        for x in 1..=7 {
            let hits: Vec<_> = px.iter().filter(|p| p.0 == x).collect();
            if hits.len() == 2 {
                let a = hits[0].2;
                let b = hits[1].2;
                if a > EPS && a < 1.0 - EPS && b > EPS && b < 1.0 - EPS {
                    assert!(approx(a + b, 1.0), "split sum {}", a + b);
                    found_split = true;
                }
            }
        }
        assert!(found_split, "expected an anti-aliased two-pixel split");
    }

    #[test]
    fn shallow_negative_slope_columns_sum_to_one() {
        let px = rasterize(0.0, 0.0, 10.0, -4.0);
        let sums = column_sums(&px, true);
        for (major, s) in sums {
            if major > 0 && major < 10 {
                assert!(approx(s, 1.0), "column {major} sum {s}");
            }
        }
    }

    #[test]
    fn steep_negative_slope_rows_sum_to_one() {
        let px = rasterize(0.0, 0.0, -4.0, 10.0);
        let sums = column_sums(&px, false);
        for (major, s) in sums {
            if major > 0 && major < 10 {
                assert!(approx(s, 1.0), "row {major} sum {s}");
            }
        }
    }

    #[test]
    fn reversed_horizontal_matches_forward() {
        let forward = sorted(rasterize(0.0, 2.0, 7.0, 2.0));
        let backward = sorted(rasterize(7.0, 2.0, 0.0, 2.0));
        assert_eq!(forward, backward);
    }

    #[test]
    fn negative_coordinates_are_handled() {
        let px = rasterize(-5.0, -5.0, -1.0, -3.0);
        assert!(!px.is_empty());
        // Endpoints of a shallow line span the rounded x range.
        assert!(px.iter().any(|p| p.0 == -5));
        assert!(px.iter().any(|p| p.0 == -1));
        assert!(px.iter().all(|p| p.2 >= 0.0 && p.2 <= 1.0));
    }

    #[test]
    fn interior_two_pixel_split_sums_to_one_generic() {
        let px = rasterize(-1.0, 0.5, 9.0, 4.2);
        let sums = column_sums(&px, true);
        let mut counted = 0;
        for (major, s) in sums {
            // Skip the two extreme columns (endpoints carry the x-gap).
            if major > -1 && major < 9 {
                assert!(approx(s, 1.0), "column {major} sum {s}");
                counted += 1;
            }
        }
        assert!(counted > 0);
    }

    #[test]
    fn gradient_produces_expected_first_step() {
        // For slope 0.5 starting flush at the origin, the first interior column
        // (x = 1) sits at y = 0.5, so both pixels get ~0.5 coverage.
        let px = rasterize(0.0, 0.0, 8.0, 4.0);
        let hits: Vec<_> = px.iter().filter(|p| p.0 == 1).collect();
        assert_eq!(hits.len(), 2);
        assert!(approx(hits[0].2, 0.5));
        assert!(approx(hits[1].2, 0.5));
        assert_eq!(hits[0].1, 0);
        assert_eq!(hits[1].1, 1);
    }

    #[test]
    fn steep_line_paints_two_columns_per_row_interior() {
        // A steep, non-integer slope splits interior rows across two x columns.
        let px = rasterize(0.0, 0.0, 4.0, 8.0);
        let hits: Vec<_> = px.iter().filter(|p| p.1 == 1).collect();
        assert_eq!(hits.len(), 2);
        let s: f32 = hits.iter().map(|p| p.2).sum();
        assert!(approx(s, 1.0));
    }

    #[test]
    fn all_samples_have_nonzero_coverage() {
        let px = rasterize(0.0, 0.0, 13.0, 5.0);
        assert!(px.iter().all(|p| p.2 > COVERAGE_EPS));
    }

    #[test]
    fn long_line_endpoints_and_bounds() {
        let px = rasterize(0.0, 0.0, 100.0, 40.0);
        assert!(px.iter().any(|p| p.0 == 0 && p.1 == 0));
        assert!(px.iter().any(|p| p.0 == 100 && p.1 == 40));
        assert!(px.iter().all(|p| p.2 >= 0.0 && p.2 <= 1.0));
    }

    #[test]
    fn horizontal_and_vertical_are_transposes() {
        let h = sorted(rasterize(0.0, 3.0, 6.0, 3.0));
        let v = rasterize(3.0, 0.0, 3.0, 6.0);
        let v_t = sorted(v.iter().map(|&(x, y, c)| (y, x, c)).collect());
        assert_eq!(h.len(), v_t.len());
        for (a, b) in h.iter().zip(v_t.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!(approx(a.2, b.2));
        }
    }
}
